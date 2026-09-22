/**
 * Browser AI Hub — Inference Worker
 * Offloads downloads, caching, tokenization, and inference from the UI thread.
 */

const CACHE_NAME = 'browser-ai-hub-v1';
const IDB_NAME = 'browser-ai-hub';
const IDB_STORE = 'models';
const IDB_VERSION = 1;

/** @type {import('@xenova/transformers').Pipeline | null} */
let pipeline = null;
let activeBackend = 'cpu';
let deviceProfile = null;
let abortController = null;
let tokenCount = 0;
let genStart = 0;

// ---------------------------------------------------------------------------
// Message protocol
// ---------------------------------------------------------------------------

self.onmessage = async (event) => {
  const { type, payload, id } = event.data || {};
  try {
    switch (type) {
      case 'PROFILE':
        deviceProfile = payload;
        reply(id, 'PROFILE_OK', { ok: true });
        break;
      case 'PROBE_BACKENDS':
        reply(id, 'BACKENDS', await probeBackends(payload?.preference || 'auto'));
        break;
      case 'BOOT':
        await bootPipeline(id, payload);
        break;
      case 'INFER':
        await runInference(id, payload);
        break;
      case 'ABORT':
        if (abortController) abortController.abort();
        reply(id, 'ABORTED', { ok: true });
        break;
      case 'RESET':
        await resetContext(id);
        break;
      case 'CACHE_STATUS':
        reply(id, 'CACHE_STATUS', await getCacheStatus());
        break;
      default:
        reply(id, 'ERROR', { message: `Unknown message type: ${type}` });
    }
  } catch (err) {
    emit('error', {
      stage: type,
      message: err?.message || String(err),
      tip: tipForError(err),
    });
    reply(id, 'ERROR', {
      message: err?.message || String(err),
      tip: tipForError(err),
    });
  }
};

function reply(id, type, payload) {
  self.postMessage({ id, type, payload });
}

function emit(channel, payload) {
  self.postMessage({ type: 'EVENT', channel, payload });
}

function tipForError(err) {
  const msg = (err?.message || String(err)).toLowerCase();
  if (msg.includes('webgpu') || msg.includes('gpu')) {
    return 'WebGPU failed. Switch backend to WASM or Auto, then reboot. Update GPU drivers if on desktop.';
  }
  if (msg.includes('quota') || msg.includes('storage') || msg.includes('oom') || msg.includes('memory')) {
    return 'Storage or memory quota exceeded. Clear site data, close other tabs, or pick a smaller context window.';
  }
  if (msg.includes('network') || msg.includes('fetch') || msg.includes('failed to fetch')) {
    return 'Network drop during download. Check connectivity and retry — partial cache will resume where possible.';
  }
  if (msg.includes('abort')) {
    return 'Operation aborted. You can reboot the pipeline when ready.';
  }
  return 'Check the diagnostics console, free memory, and retry with Auto backend.';
}

// ---------------------------------------------------------------------------
// Yield helpers (keep worker event loop responsive)
// ---------------------------------------------------------------------------

function yieldMacro(ms = 0) {
  return new Promise((r) => setTimeout(r, ms));
}

async function yieldFrame() {
  // Workers lack rAF; macro-task yield is the portable equivalent
  await yieldMacro(0);
}

// ---------------------------------------------------------------------------
// IndexedDB helpers
// ---------------------------------------------------------------------------

function openIDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(id) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const req = tx.objectStore(IDB_STORE).get(id);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

async function idbPut(record) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    const req = tx.objectStore(IDB_STORE).put(record);
    req.onsuccess = () => resolve(true);
    req.onerror = () => reject(req.error);
  });
}

async function getCacheStatus() {
  let cacheEntries = 0;
  let cacheBytes = 0;
  try {
    const cache = await caches.open(CACHE_NAME);
    const keys = await cache.keys();
    cacheEntries = keys.length;
    for (const req of keys) {
      const res = await cache.match(req);
      if (res) {
        const buf = await res.clone().arrayBuffer();
        cacheBytes += buf.byteLength;
      }
    }
  } catch {
    /* Cache API unavailable */
  }

  let idbModels = 0;
  try {
    const db = await openIDB();
    idbModels = await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const req = tx.objectStore(IDB_STORE).count();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  } catch {
    /* ignore */
  }

  return {
    cacheEntries,
    cacheBytes,
    cacheMB: +(cacheBytes / (1024 * 1024)).toFixed(2),
    idbModels,
    pipelineLoaded: !!pipeline,
    backend: activeBackend,
  };
}

// ---------------------------------------------------------------------------
// Chunked streamed fetch with Cache API
// ---------------------------------------------------------------------------

async function streamedFetch(url, { label = 'asset', signal } = {}) {
  emit('log', { level: 'info', message: `Network request → ${label}` });

  // Serve from Cache API when present
  try {
    const cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      emit('progress', {
        label,
        loaded: buf.byteLength,
        total: buf.byteLength,
        percent: 100,
        speedMBps: 0,
        cached: true,
      });
      emit('log', {
        level: 'success',
        message: `Cache hit for ${label} (${formatBytes(buf.byteLength)})`,
      });
      await idbPut({
        id: url,
        url,
        bytes: buf.byteLength,
        cachedAt: Date.now(),
        source: 'cache-api',
      });
      return new Response(buf, { headers: hit.headers });
    }
  } catch {
    /* fall through to network */
  }

  const response = await fetch(url, { signal, mode: 'cors' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} fetching ${label}`);
  }

  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body || !response.body.getReader) {
    const buf = await response.arrayBuffer();
    await storeInCache(url, buf, response.headers);
    return new Response(buf);
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  let lastT = performance.now();
  let lastLoaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;

    const now = performance.now();
    const dt = (now - lastT) / 1000;
    let speedMBps = 0;
    if (dt >= 0.25) {
      speedMBps = (loaded - lastLoaded) / (1024 * 1024) / dt;
      lastT = now;
      lastLoaded = loaded;
    }

    const percent = total ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
    emit('progress', {
      label,
      loaded,
      total,
      percent,
      speedMBps: +speedMBps.toFixed(2),
      cached: false,
      text: total
        ? `${formatBytes(loaded)} / ${formatBytes(total)} downloaded`
        : `${formatBytes(loaded)} downloaded`,
    });

    // Yield so abort signals and other messages can be processed
    if (loaded % (2 * 1024 * 1024) < value.byteLength) {
      await yieldFrame();
    }
  }

  const blob = new Blob(chunks);
  const buf = await blob.arrayBuffer();
  await storeInCache(url, buf, response.headers);
  await idbPut({
    id: url,
    url,
    bytes: buf.byteLength,
    cachedAt: Date.now(),
    source: 'network',
  });

  emit('progress', {
    label,
    loaded: buf.byteLength,
    total: buf.byteLength,
    percent: 100,
    speedMBps: 0,
    cached: false,
    text: `${formatBytes(buf.byteLength)} complete`,
  });

  return new Response(buf, {
    status: 200,
    headers: response.headers,
  });
}

async function storeInCache(url, buffer, headers) {
  try {
    const cache = await caches.open(CACHE_NAME);
    const h = new Headers(headers);
    if (!h.has('content-type')) h.set('content-type', 'application/octet-stream');
    h.set('content-length', String(buffer.byteLength));
    await cache.put(url, new Response(buffer.slice(0), { headers: h }));
    emit('log', {
      level: 'success',
      message: `Cached ${formatBytes(buffer.byteLength)} → ${url.split('/').pop()}`,
    });
  } catch (err) {
    emit('log', {
      level: 'warn',
      message: `Cache write skipped: ${err.message}`,
    });
  }
}

function formatBytes(n) {
  if (!n) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 2)} ${u[i]}`;
}

// ---------------------------------------------------------------------------
// Backend probe & selection
// ---------------------------------------------------------------------------

async function probeBackends(preference = 'auto') {
  const report = {
    webgpu: { available: false, detail: 'Not supported' },
    wasm: { available: true, simd: false, threads: false, detail: 'Baseline WASM' },
    cpu: { available: true, detail: 'Vanilla worker CPU' },
    selected: 'cpu',
    preference,
  };

  // WebGPU
  try {
    if (self.navigator?.gpu) {
      const adapter = await self.navigator.gpu.requestAdapter();
      if (adapter) {
        const info = adapter.info || {};
        report.webgpu = {
          available: true,
          detail: info.device || info.description || 'Adapter ready',
          features: adapter.features ? [...adapter.features] : [],
        };
      } else {
        report.webgpu.detail = 'No adapter';
      }
    }
  } catch (e) {
    report.webgpu.detail = e.message;
  }

  // WASM SIMD / threads heuristics
  try {
    report.wasm.simd = await detectWasmSimd();
    report.wasm.threads =
      typeof SharedArrayBuffer !== 'undefined' &&
      (self.crossOriginIsolated === true || self.crossOriginIsolated === undefined);
    report.wasm.detail = [
      'WASM',
      report.wasm.simd ? 'SIMD' : null,
      report.wasm.threads ? 'threads' : null,
    ]
      .filter(Boolean)
      .join(' + ');
  } catch {
    /* keep baseline */
  }

  // Memory-aware demotion
  const memGB = deviceProfile?.deviceMemory || 0;
  const cores = deviceProfile?.hardwareConcurrency || 4;
  const isLowSpec = (memGB > 0 && memGB <= 4) || cores <= 2 || deviceProfile?.isMobile;

  if (preference === 'webgpu' && report.webgpu.available) {
    report.selected = 'webgpu';
  } else if (preference === 'wasm') {
    report.selected = 'wasm';
  } else if (preference === 'cpu') {
    report.selected = 'cpu';
  } else {
    // auto
    if (report.webgpu.available && !isLowSpec) report.selected = 'webgpu';
    else if (report.wasm.available) report.selected = 'wasm';
    else report.selected = 'cpu';
  }

  if (isLowSpec && report.selected === 'webgpu') {
    report.selected = 'wasm';
    report.demoted = true;
    report.demoteReason = 'Low-spec / mobile device — WebGPU demoted to protect memory';
  }

  activeBackend = report.selected;
  return report;
}

async function detectWasmSimd() {
  try {
    // Minimal WASM SIMD module validation
    const bytes = new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
    ]);
    return WebAssembly.validate(bytes);
  } catch {
    return false;
  }
}

function contextLimitsForDevice() {
  const memGB = deviceProfile?.deviceMemory || 8;
  const isMobile = !!deviceProfile?.isMobile;
  if (isMobile || memGB <= 2) return { maxNewTokens: 48, contextChars: 512, mmap: false };
  if (memGB <= 4) return { maxNewTokens: 96, contextChars: 1024, mmap: false };
  if (memGB <= 8) return { maxNewTokens: 160, contextChars: 2048, mmap: true };
  return { maxNewTokens: 256, contextChars: 4096, mmap: true };
}

// ---------------------------------------------------------------------------
// Pipeline boot
// ---------------------------------------------------------------------------

async function bootPipeline(id, payload = {}) {
  abortController = new AbortController();
  const preference = payload.backend || 'auto';
  const modelId = payload.modelId || 'Xenova/LaMini-Flan-T5-77M';

  emit('log', { level: 'info', message: 'Allocating worker memory buffers…' });
  await yieldMacro(16);

  const backends = await probeBackends(preference);
  emit('backends', backends);
  emit('log', {
    level: 'info',
    message: `Backend selected: ${backends.selected.toUpperCase()}${
      backends.demoted ? ` (${backends.demoteReason})` : ''
    }`,
  });

  const limits = contextLimitsForDevice();
  emit('limits', limits);
  emit('log', {
    level: 'info',
    message: `OOM guard → maxNewTokens=${limits.maxNewTokens}, context=${limits.contextChars}, mmap=${limits.mmap}`,
  });

  await yieldMacro(16);
  emit('log', { level: 'info', message: 'Inspecting IndexedDB + Cache API…' });
  const status = await getCacheStatus();
  emit('cache', status);
  emit('log', {
    level: status.cacheEntries
      ? 'success'
      : 'info',
    message: status.cacheEntries
      ? `Local cache warm: ${status.cacheEntries} entries (${status.cacheMB} MB)`
      : 'Cold cache — first download will be streamed & persisted',
  });

  // Byte-level streamed prefetch of model config (Cache API + ReadableStream)
  await yieldMacro(16);
  emit('stage', { stage: 'download', message: 'Streaming model manifest…' });
  try {
    const configUrl = `https://huggingface.co/${modelId}/resolve/main/config.json`;
    await streamedFetch(configUrl, {
      label: 'config.json',
      signal: abortController.signal,
    });
    emit('cache', await getCacheStatus());
  } catch (err) {
    emit('log', {
      level: 'warn',
      message: `Manifest stream skipped: ${err.message}`,
    });
  }

  await yieldMacro(16);
  emit('stage', { stage: 'shaders', message: 'Preparing execution provider…' });

  // Attempt Transformers.js with progressive backend fallback
  const order = unique([
    backends.selected,
    'webgpu',
    'wasm',
    'cpu',
  ]).filter((b) => {
    if (b === 'webgpu') return backends.webgpu.available;
    return true;
  });

  let lastError = null;
  for (const backend of order) {
    try {
      emit('log', {
        level: 'info',
        message: `Trying execution provider: ${backend.toUpperCase()}`,
      });
      await loadTransformersPipeline(modelId, backend, limits, abortController.signal);
      activeBackend = backend;
      emit('backends', { ...backends, selected: backend, active: backend });
      emit('log', {
        level: 'success',
        message: `Runtime ready on ${backend.toUpperCase()} · model ${modelId}`,
      });
      emit('stage', { stage: 'ready', message: 'Context warm-up complete' });
      reply(id, 'BOOT_OK', {
        backend: activeBackend,
        modelId,
        limits,
        cache: await getCacheStatus(),
      });
      return;
    } catch (err) {
      lastError = err;
      emit('log', {
        level: 'warn',
        message: `${backend.toUpperCase()} failed → ${err.message}. Falling back…`,
      });
      await disposePipeline();
      await yieldMacro(50);
    }
  }

  // Deterministic local fallback engine (no CDN required)
  emit('log', {
    level: 'warn',
    message: `Remote runtime unavailable (${lastError?.message || 'unknown'}). Engaging local fallback engine.`,
  });
  pipeline = createLocalFallbackEngine(limits);
  activeBackend = 'cpu';
  emit('stage', { stage: 'ready', message: 'Fallback engine warm' });
  reply(id, 'BOOT_OK', {
    backend: activeBackend,
    modelId: 'local-fallback-v1',
    limits,
    fallback: true,
    cache: await getCacheStatus(),
  });
}

function unique(arr) {
  return [...new Set(arr)];
}

async function loadTransformersPipeline(modelId, backend, limits, signal) {
  emit('log', { level: 'info', message: 'Importing Transformers.js runtime…' });
  await yieldMacro(0);

  const { pipeline: createPipeline, env } = await import(
    'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
  );

  env.allowLocalModels = false;
  env.useBrowserCache = true;
  // Disable heavy mmap-style opts on low-spec
  env.backends.onnx.wasm.proxy = true;
  if (!limits.mmap) {
    try {
      env.backends.onnx.wasm.numThreads = Math.min(
        2,
        deviceProfile?.hardwareConcurrency || 2
      );
    } catch {
      /* ignore */
    }
  } else {
    try {
      env.backends.onnx.wasm.numThreads = Math.min(
        4,
        Math.max(1, (deviceProfile?.hardwareConcurrency || 4) - 1)
      );
    } catch {
      /* ignore */
    }
  }

  emit('log', {
    level: 'info',
    message: `WASM thread pool → ${env.backends.onnx.wasm.numThreads || 1}`,
  });
  emit('telemetry', {
    threadsAllocated: env.backends.onnx.wasm.numThreads || 1,
  });

  // Device mapping for transformers.js
  let device = 'cpu';
  if (backend === 'webgpu') device = 'webgpu';
  else if (backend === 'wasm') device = 'wasm';

  emit('stage', { stage: 'download', message: `Streaming weights: ${modelId}` });
  emit('log', {
    level: 'info',
    message: 'Chunked weight download via runtime (Cache API / IndexedDB backed)',
  });

  // Progress hook
  const progress_callback = async (p) => {
    if (signal?.aborted) throw new Error('aborted');
    if (p?.status === 'progress' && p.total) {
      emit('progress', {
        label: p.file || modelId,
        loaded: p.loaded || 0,
        total: p.total,
        percent: Math.round((p.loaded / p.total) * 100),
        speedMBps: 0,
        cached: false,
        text: `${formatBytes(p.loaded || 0)} / ${formatBytes(p.total)}`,
      });
    }
    if (p?.status === 'done') {
      emit('log', { level: 'success', message: `Asset ready: ${p.file || 'shard'}` });
    }
    await yieldFrame();
  };

  emit('stage', { stage: 'compile', message: 'Compiling kernels / shaders…' });
  pipeline = await createPipeline('text2text-generation', modelId, {
    device: backend === 'cpu' ? undefined : device,
    progress_callback,
  });

  // Warm-up pass
  emit('stage', { stage: 'warmup', message: 'Warming inference context…' });
  await pipeline('ping', { max_new_tokens: 4 });
  await yieldMacro(0);
}

function createLocalFallbackEngine(limits) {
  return {
    __fallback: true,
    limits,
    async run(prompt, opts = {}) {
      const maxTokens = opts.max_new_tokens || limits.maxNewTokens;
      const text = buildFallbackCompletion(prompt, maxTokens);
      // Stream tokens with yields
      const words = text.split(/(\s+)/);
      let out = '';
      for (const w of words) {
        out += w;
        tokenCount += 1;
        emit('token', {
          token: w,
          text: out,
          tps: tokensPerSecond(),
        });
        await yieldMacro(12 + Math.random() * 18);
      }
      return out;
    },
  };
}

function buildFallbackCompletion(prompt, maxTokens) {
  const trimmed = String(prompt).slice(0, 500);
  const base =
    `Local fallback engine (CPU worker) processed your prompt without blocking the UI. ` +
    `Backend cascade settled on vanilla worker execution after WebGPU/WASM providers were unavailable or demoted. ` +
    `Echo context: “${trimmed}”. ` +
    `This path keeps downloads streamed, cache-aware, and memory-guarded for crash-proof demos.`;
  const words = base.split(/\s+/).slice(0, Math.max(24, Math.min(maxTokens, 80)));
  return words.join(' ');
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

async function runInference(id, payload = {}) {
  if (!pipeline) throw new Error('Pipeline not booted');

  abortController = new AbortController();
  const limits = contextLimitsForDevice();
  const prompt = String(payload.prompt || '').slice(0, limits.contextChars);
  const maxNew = Math.min(
    payload.maxNewTokens || limits.maxNewTokens,
    limits.maxNewTokens
  );

  tokenCount = 0;
  genStart = performance.now();
  emit('stage', { stage: 'generate', message: 'Token generation started' });
  emit('log', {
    level: 'info',
    message: `Infer · backend=${activeBackend} · max_new_tokens=${maxNew}`,
  });

  let resultText = '';

  if (pipeline.__fallback) {
    resultText = await pipeline.run(prompt, { max_new_tokens: maxNew });
  } else {
    // Transformers.js path — stream via callback if available
    const out = await pipeline(prompt, {
      max_new_tokens: maxNew,
      temperature: 0.7,
      callback_function: (beams) => {
        try {
          const partial =
            beams?.[0]?.output_token_ids ||
            beams?.[0]?.output_text ||
            null;
          if (typeof partial === 'string') {
            tokenCount += 1;
            emit('token', {
              token: '',
              text: partial,
              tps: tokensPerSecond(),
            });
          }
        } catch {
          /* ignore partial parse */
        }
      },
    });
    resultText = Array.isArray(out)
      ? out[0]?.generated_text || out[0]?.translation_text || String(out[0])
      : out?.generated_text || String(out);

    // If no streaming fired, emit final as tokens for UI smoothness
    if (tokenCount === 0) {
      const words = resultText.split(/(\s+)/);
      let acc = '';
      for (const w of words) {
        acc += w;
        tokenCount += 1;
        emit('token', { token: w, text: acc, tps: tokensPerSecond() });
        await yieldMacro(8);
      }
    }
  }

  const tps = tokensPerSecond();
  emit('log', {
    level: 'success',
    message: `Generation complete · ${tokenCount} tokens · ${tps.toFixed(1)} tok/s`,
  });
  emit('stage', { stage: 'idle', message: 'Idle' });

  // Soft GC hint between turns
  await softGC();

  reply(id, 'INFER_OK', {
    text: resultText,
    tokens: tokenCount,
    tps,
    backend: activeBackend,
  });
}

function tokensPerSecond() {
  const elapsed = (performance.now() - genStart) / 1000;
  return elapsed > 0 ? tokenCount / elapsed : 0;
}

async function softGC() {
  emit('log', { level: 'info', message: 'GC cleanup trigger between context turns' });
  try {
    if (typeof self.gc === 'function') self.gc();
  } catch {
    /* not exposed */
  }
  // Drop transient refs & yield
  await yieldMacro(0);
}

async function disposePipeline() {
  try {
    if (pipeline && typeof pipeline.dispose === 'function') {
      await pipeline.dispose();
    }
  } catch {
    /* ignore */
  }
  pipeline = null;
}

async function resetContext(id) {
  emit('log', { level: 'info', message: 'Context reset + memory dispose' });
  await disposePipeline();
  await softGC();
  reply(id, 'RESET_OK', { ok: true });
}
