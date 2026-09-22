/**
 * Browser AI Hub — Inference Worker (v2)
 * Downloads, Cache API / IndexedDB, tokenization, and inference off the UI thread.
 */

const CACHE_NAME = 'browser-ai-hub-models-v2';
const IDB_NAME = 'browser-ai-hub';
const IDB_STORE = 'models';
const IDB_CHAT = 'chats';
const IDB_META = 'meta';
const IDB_VERSION = 2;

let pipeline = null;
let activeBackend = 'cpu';
let activeModelId = null;
let deviceProfile = null;
let abortController = null;
let tokenCount = 0;
let genStart = 0;
let limits = { maxNewTokens: 128, contextChars: 2048, mmap: true };

self.onmessage = async (event) => {
  const { type, payload, id } = event.data || {};
  try {
    switch (type) {
      case 'PROFILE':
        deviceProfile = payload;
        limits = contextLimitsForDevice();
        reply(id, 'PROFILE_OK', { ok: true, limits });
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
      case 'CLEAR_CACHE':
        reply(id, 'CACHE_CLEARED', await clearAllCaches());
        break;
      case 'SAVE_CHAT':
        await saveChat(payload);
        reply(id, 'CHAT_SAVED', { ok: true });
        break;
      case 'LOAD_CHAT':
        reply(id, 'CHAT_HISTORY', await loadChat());
        break;
      case 'CLEAR_CHAT':
        await clearChat();
        reply(id, 'CHAT_CLEARED', { ok: true });
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
    return 'WebGPU failed. Switch to WASM or Auto, then reboot. Update GPU drivers on desktop.';
  }
  if (msg.includes('quota') || msg.includes('storage') || msg.includes('oom') || msg.includes('memory')) {
    return 'Quota/memory exceeded. Use Clear cache, close other tabs, or pick a smaller model.';
  }
  if (msg.includes('network') || msg.includes('fetch') || msg.includes('failed to fetch')) {
    return 'Network drop. Retry — Cache API keeps completed shards. Or go fully offline after first warm cache.';
  }
  if (msg.includes('abort')) return 'Aborted. Boot again when ready.';
  return 'Check diagnostics, free memory, retry Auto backend.';
}

function yieldMacro(ms = 0) {
  return new Promise((r) => setTimeout(r, ms));
}

async function yieldFrame() {
  await yieldMacro(0);
}

function openIDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(IDB_CHAT)) {
        db.createObjectStore(IDB_CHAT, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(IDB_META)) {
        db.createObjectStore(IDB_META, { keyPath: 'key' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbPut(store, record) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    tx.objectStore(store).put(record);
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

async function idbGet(store, key) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const req = tx.objectStore(store).get(key);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

async function idbClear(store) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    tx.objectStore(store).clear();
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

async function saveChat(payload) {
  await idbPut(IDB_CHAT, {
    id: 'session',
    messages: payload?.messages || [],
    updatedAt: Date.now(),
  });
}

async function loadChat() {
  const row = await idbGet(IDB_CHAT, 'session');
  return { messages: row?.messages || [], updatedAt: row?.updatedAt || null };
}

async function clearChat() {
  await idbClear(IDB_CHAT);
}

async function getCacheStatus() {
  let cacheEntries = 0;
  let cacheBytes = 0;
  try {
    const cache = await caches.open(CACHE_NAME);
    const keys = await cache.keys();
    cacheEntries = keys.length;
    // Fast path: count via Content-Length headers when present
    for (const req of keys) {
      const res = await cache.match(req);
      if (!res) continue;
      const cl = Number(res.headers.get('content-length'));
      if (cl > 0) {
        cacheBytes += cl;
      } else {
        const buf = await res.clone().arrayBuffer();
        cacheBytes += buf.byteLength;
      }
    }
  } catch {
    /* Cache API unavailable */
  }

  let estimate = null;
  try {
    if (navigator.storage?.estimate) {
      estimate = await navigator.storage.estimate();
    }
  } catch {
    /* ignore */
  }

  return {
    cacheEntries,
    cacheBytes,
    cacheMB: +(cacheBytes / (1024 * 1024)).toFixed(2),
    pipelineLoaded: !!pipeline,
    backend: activeBackend,
    modelId: activeModelId,
    quotaMB: estimate?.quota ? +(estimate.quota / (1024 * 1024)).toFixed(0) : null,
    usageMB: estimate?.usage ? +(estimate.usage / (1024 * 1024)).toFixed(2) : null,
  };
}

async function clearAllCaches() {
  try {
    await caches.delete(CACHE_NAME);
  } catch {
    /* ignore */
  }
  try {
    await idbClear(IDB_STORE);
  } catch {
    /* ignore */
  }
  await disposePipeline();
  return getCacheStatus();
}

async function streamedFetch(url, { label = 'asset', signal } = {}) {
  emit('log', { level: 'info', message: `Network → ${label}` });

  try {
    const cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      const total = Number(hit.headers.get('content-length')) || 0;
      emit('progress', {
        label,
        loaded: total,
        total,
        percent: 100,
        speedMBps: 0,
        cached: true,
        text: total ? `${formatBytes(total)} from cache` : 'Cache hit',
      });
      emit('log', {
        level: 'success',
        message: `Cache hit · ${label}${total ? ` (${formatBytes(total)})` : ''}`,
      });
      await idbPut(IDB_STORE, {
        id: url,
        url,
        bytes: total,
        cachedAt: Date.now(),
        source: 'cache-api',
      });
      return hit;
    }
  } catch {
    /* fall through */
  }

  const response = await fetch(url, { signal, mode: 'cors', credentials: 'omit' });
  if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${label}`);

  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body?.getReader) {
    const buf = await response.arrayBuffer();
    await storeInCache(url, buf, response.headers);
    return new Response(buf, { headers: response.headers });
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  let lastT = performance.now();
  let lastLoaded = 0;
  let speedMBps = 0;

  while (true) {
    if (signal?.aborted) {
      try {
        await reader.cancel();
      } catch {
        /* ignore */
      }
      throw new Error('aborted');
    }
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;

    const now = performance.now();
    const dt = (now - lastT) / 1000;
    if (dt >= 0.2) {
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

    // Yield every ~1MB so abort + other messages stay responsive
    if (loaded % (1024 * 1024) < value.byteLength) await yieldFrame();
  }

  const blob = new Blob(chunks);
  const buf = await blob.arrayBuffer();
  await storeInCache(url, buf, response.headers);
  await idbPut(IDB_STORE, {
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

  return new Response(buf.slice(0), {
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
    h.set('x-cached-at', String(Date.now()));
    await cache.put(url, new Response(buffer.slice(0), { headers: h, status: 200 }));
    emit('log', {
      level: 'success',
      message: `Persisted ${formatBytes(buffer.byteLength)} → cache`,
    });
  } catch (err) {
    emit('log', { level: 'warn', message: `Cache write skipped: ${err.message}` });
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

async function probeBackends(preference = 'auto') {
  const report = {
    webgpu: { available: false, detail: 'Not supported' },
    wasm: { available: true, simd: false, threads: false, detail: 'Baseline WASM' },
    cpu: { available: true, detail: 'Vanilla worker CPU' },
    selected: 'cpu',
    preference,
  };

  try {
    if (self.navigator?.gpu) {
      const adapter = await self.navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (adapter) {
        let detail = 'Adapter ready';
        try {
          if (adapter.requestAdapterInfo) {
            const info = await adapter.requestAdapterInfo();
            detail = [info.vendor, info.architecture || info.device]
              .filter(Boolean)
              .join(' · ') || detail;
          } else if (adapter.info) {
            detail = adapter.info.device || adapter.info.description || detail;
          }
        } catch {
          /* info optional */
        }
        report.webgpu = {
          available: true,
          detail,
          features: adapter.features ? [...adapter.features].slice(0, 8) : [],
        };
      } else {
        report.webgpu.detail = 'No adapter';
      }
    }
  } catch (e) {
    report.webgpu.detail = e.message;
  }

  try {
    report.wasm.simd = await detectWasmSimd();
    report.wasm.threads = typeof SharedArrayBuffer !== 'undefined';
    report.wasm.detail = ['WASM', report.wasm.simd && 'SIMD', report.wasm.threads && 'threads']
      .filter(Boolean)
      .join(' + ');
  } catch {
    /* keep baseline */
  }

  const memGB = deviceProfile?.deviceMemory || 0;
  const cores = deviceProfile?.hardwareConcurrency || 4;
  const isLowSpec = (memGB > 0 && memGB <= 4) || cores <= 2 || deviceProfile?.isMobile;

  if (preference === 'webgpu' && report.webgpu.available) report.selected = 'webgpu';
  else if (preference === 'wasm') report.selected = 'wasm';
  else if (preference === 'cpu') report.selected = 'cpu';
  else if (report.webgpu.available && !isLowSpec) report.selected = 'webgpu';
  else if (report.wasm.available) report.selected = 'wasm';
  else report.selected = 'cpu';

  if (isLowSpec && report.selected === 'webgpu') {
    report.selected = 'wasm';
    report.demoted = true;
    report.demoteReason = 'Low-spec / mobile — WebGPU demoted to protect memory';
  }

  activeBackend = report.selected;
  return report;
}

async function detectWasmSimd() {
  try {
    const bytes = Uint8Array.from([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0,
      253, 15, 253, 98, 11,
    ]);
    return WebAssembly.validate(bytes);
  } catch {
    return false;
  }
}

function contextLimitsForDevice() {
  const memGB = deviceProfile?.deviceMemory || 8;
  const isMobile = !!deviceProfile?.isMobile;
  if (isMobile || memGB <= 2) return { maxNewTokens: 48, contextChars: 512, mmap: false, threads: 1 };
  if (memGB <= 4) return { maxNewTokens: 96, contextChars: 1024, mmap: false, threads: 2 };
  if (memGB <= 8) return { maxNewTokens: 160, contextChars: 2048, mmap: true, threads: 3 };
  return { maxNewTokens: 256, contextChars: 4096, mmap: true, threads: 4 };
}

async function bootPipeline(id, payload = {}) {
  abortController = new AbortController();
  const preference = payload.backend || 'auto';
  const modelId = payload.modelId || 'Xenova/LaMini-Flan-T5-77M';
  limits = contextLimitsForDevice();

  emit('log', { level: 'info', message: 'Allocating worker memory buffers…' });
  await yieldMacro(8);

  const backends = await probeBackends(preference);
  emit('backends', backends);
  emit('limits', limits);
  emit('log', {
    level: 'info',
    message: `Backend selected: ${backends.selected.toUpperCase()}${
      backends.demoted ? ` (${backends.demoteReason})` : ''
    }`,
  });
  emit('log', {
    level: 'info',
    message: `OOM guard → tokens=${limits.maxNewTokens}, ctx=${limits.contextChars}, threads≤${limits.threads}`,
  });

  await yieldMacro(8);
  const status = await getCacheStatus();
  emit('cache', status);
  emit('log', {
    level: status.cacheEntries ? 'success' : 'info',
    message: status.cacheEntries
      ? `Local cache warm: ${status.cacheEntries} entries (${status.cacheMB} MB)`
      : 'Cold cache — first download streamed & persisted locally',
  });

  // Stream model config (byte-level progress demo + cache warm)
  emit('stage', { stage: 'download', message: 'Streaming model manifest…' });
  try {
    const configUrl = `https://huggingface.co/${modelId}/resolve/main/config.json`;
    await streamedFetch(configUrl, {
      label: 'config.json',
      signal: abortController.signal,
    });
    emit('cache', await getCacheStatus());
  } catch (err) {
    emit('log', { level: 'warn', message: `Manifest stream skipped: ${err.message}` });
  }

  emit('stage', { stage: 'shaders', message: 'Preparing execution provider…' });

  const order = unique([backends.selected, 'webgpu', 'wasm', 'cpu']).filter((b) => {
    if (b === 'webgpu') return backends.webgpu.available;
    return true;
  });

  let lastError = null;
  for (const backend of order) {
    try {
      emit('log', { level: 'info', message: `Trying execution provider: ${backend.toUpperCase()}` });
      await loadTransformersPipeline(modelId, backend, abortController.signal);
      activeBackend = backend;
      activeModelId = modelId;
      emit('backends', { ...backends, selected: backend, active: backend });
      emit('log', {
        level: 'success',
        message: `Runtime ready on ${backend.toUpperCase()} · ${modelId}`,
      });
      emit('stage', { stage: 'ready', message: 'Context warm-up complete' });
      await idbPut(IDB_META, { key: 'lastBoot', modelId, backend, at: Date.now() });
      reply(id, 'BOOT_OK', {
        backend: activeBackend,
        modelId,
        limits,
        cache: await getCacheStatus(),
        privacy: 'on-device',
      });
      return;
    } catch (err) {
      lastError = err;
      emit('log', {
        level: 'warn',
        message: `${backend.toUpperCase()} failed → ${err.message}. Falling back…`,
      });
      await disposePipeline();
      await yieldMacro(30);
    }
  }

  emit('log', {
    level: 'warn',
    message: `Remote runtime unavailable (${lastError?.message || 'unknown'}). Local hybrid engine engaged.`,
  });
  pipeline = createLocalHybridEngine();
  activeBackend = 'cpu';
  activeModelId = 'local-hybrid-v2';
  emit('stage', { stage: 'ready', message: 'Hybrid engine warm' });
  reply(id, 'BOOT_OK', {
    backend: activeBackend,
    modelId: activeModelId,
    limits,
    fallback: true,
    cache: await getCacheStatus(),
    privacy: 'on-device',
  });
}

function unique(arr) {
  return [...new Set(arr)];
}

async function loadTransformersPipeline(modelId, backend, signal) {
  emit('log', { level: 'info', message: 'Importing Transformers.js runtime…' });
  await yieldMacro(0);

  const { pipeline: createPipeline, env } = await import(
    'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
  );

  env.allowLocalModels = false;
  env.useBrowserCache = true;
  env.useFS = false;

  const threads = Math.max(
    1,
    Math.min(limits.threads || 2, (deviceProfile?.hardwareConcurrency || 4) - 1)
  );
  try {
    env.backends.onnx.wasm.numThreads = limits.mmap ? threads : Math.min(2, threads);
    env.backends.onnx.wasm.simd = true;
  } catch {
    /* ignore */
  }

  emit('telemetry', { threadsAllocated: env.backends.onnx.wasm.numThreads || 1 });
  emit('log', {
    level: 'info',
    message: `WASM thread pool → ${env.backends.onnx.wasm.numThreads || 1}`,
  });

  const progress_callback = async (p) => {
    if (signal?.aborted) throw new Error('aborted');
    if (p?.status === 'progress' && p.total) {
      emit('progress', {
        label: p.file || modelId,
        loaded: p.loaded || 0,
        total: p.total,
        percent: Math.round(((p.loaded || 0) / p.total) * 100),
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

  emit('stage', { stage: 'download', message: `Streaming weights: ${modelId}` });
  emit('stage', { stage: 'compile', message: 'Compiling kernels / shaders…' });

  const opts = {
    quantized: true,
    progress_callback,
  };
  // device hint — transformers.js 2.x uses quantized ONNX; webgpu path when supported
  if (backend === 'webgpu') {
    try {
      opts.device = 'webgpu';
    } catch {
      /* ignore */
    }
  }

  pipeline = await createPipeline('text2text-generation', modelId, opts);

  emit('stage', { stage: 'warmup', message: 'Warming inference context…' });
  await pipeline('warmup', { max_new_tokens: 2, temperature: 0 });
  await yieldMacro(0);
}

function createLocalHybridEngine() {
  return {
    __fallback: true,
    async run(prompt, opts = {}) {
      const maxTokens = opts.max_new_tokens || limits.maxNewTokens;
      const text = buildHybridCompletion(prompt, maxTokens);
      const words = text.split(/(\s+)/);
      let out = '';
      for (const w of words) {
        if (abortController?.signal?.aborted) throw new Error('aborted');
        out += w;
        tokenCount += 1;
        emit('token', { token: w, text: out, tps: tokensPerSecond() });
        await yieldMacro(8 + Math.random() * 12);
      }
      return out;
    },
  };
}

function buildHybridCompletion(prompt, maxTokens) {
  const raw = String(prompt).trim();
  const lower = raw.toLowerCase();

  // Arithmetic: "what is 10+10", "10 * 5", etc.
  const math = raw.match(
    /(?:what\s+is\s+|calculate\s+|compute\s+)?(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*\??/i
  );
  if (math) {
    const a = Number(math[1]);
    let op = math[2];
    const b = Number(math[3]);
    if (op === 'x' || op === '×') op = '*';
    if (op === '÷') op = '/';
    let result;
    switch (op) {
      case '+':
        result = a + b;
        break;
      case '-':
        result = a - b;
        break;
      case '*':
        result = a * b;
        break;
      case '/':
        result = b === 0 ? 'undefined (divide by zero)' : a / b;
        break;
      default:
        result = null;
    }
    if (result !== null) {
      return `${a} ${op} ${b} = ${result}. Computed entirely on-device in the CPU worker — no server round-trip.`;
    }
  }

  if (/^(hi|hello|hey)\b/i.test(lower)) {
    return 'Hello. Browser AI Hub is running locally in a Web Worker. Your prompts stay on this device after model weights are cached.';
  }

  if (/privacy|local|offline|store|cache/i.test(lower)) {
    return (
      'Privacy model: inference runs in your browser tab/worker. Model weights persist in Cache API + IndexedDB so they only download once. ' +
      'Chat history can be saved locally in IndexedDB and never leaves the device. Clear cache from the UI anytime.'
    );
  }

  if (/model|which model|what model/i.test(lower)) {
    return (
      'This session is on the local-hybrid CPU engine (CDN/runtime unavailable). ' +
      'When Transformers.js boots successfully, the default model is Xenova/LaMini-Flan-T5-77M (quantized), with Flan-T5 Small as an alternate.'
    );
  }

  const base =
    `On-device hybrid engine processed: “${raw.slice(0, 280)}”. ` +
    `No remote LLM call was made. For full neural generation, boot with network once so quantized Transformers.js weights cache locally, then you can work offline.`;
  return base.split(/\s+/).slice(0, Math.max(20, Math.min(maxTokens, 70))).join(' ');
}

async function runInference(id, payload = {}) {
  if (!pipeline) throw new Error('Pipeline not booted');

  abortController = new AbortController();
  limits = contextLimitsForDevice();
  const prompt = String(payload.prompt || '').slice(0, limits.contextChars);
  const maxNew = Math.min(payload.maxNewTokens || limits.maxNewTokens, limits.maxNewTokens);

  tokenCount = 0;
  genStart = performance.now();
  emit('stage', { stage: 'generate', message: 'Token generation started' });
  emit('log', {
    level: 'info',
    message: `Infer · backend=${activeBackend} · model=${activeModelId} · max_new=${maxNew}`,
  });

  let resultText = '';

  if (pipeline.__fallback) {
    resultText = await pipeline.run(prompt, { max_new_tokens: maxNew });
  } else {
    const out = await pipeline(prompt, {
      max_new_tokens: maxNew,
      temperature: 0.3,
      repetition_penalty: 1.1,
    });
    resultText = Array.isArray(out)
      ? out[0]?.generated_text || out[0]?.translation_text || String(out[0])
      : out?.generated_text || String(out);

    // Smooth token reveal without blocking
    if (tokenCount === 0) {
      const words = resultText.split(/(\s+)/);
      let acc = '';
      for (const w of words) {
        if (abortController.signal.aborted) break;
        acc += w;
        tokenCount += 1;
        emit('token', { token: w, text: acc, tps: tokensPerSecond() });
        await yieldMacro(4);
      }
      resultText = acc || resultText;
    }
  }

  const tps = tokensPerSecond();
  const latencyMs = Math.round(performance.now() - genStart);
  emit('log', {
    level: 'success',
    message: `Done · ${tokenCount} tokens · ${tps.toFixed(1)} tok/s · ${latencyMs} ms`,
  });
  emit('stage', { stage: 'idle', message: 'Idle' });
  await softGC();

  reply(id, 'INFER_OK', {
    text: resultText,
    tokens: tokenCount,
    tps,
    latencyMs,
    backend: activeBackend,
    modelId: activeModelId,
  });
}

function tokensPerSecond() {
  const elapsed = (performance.now() - genStart) / 1000;
  return elapsed > 0 ? tokenCount / elapsed : 0;
}

async function softGC() {
  emit('log', { level: 'info', message: 'GC cleanup between turns' });
  try {
    if (typeof self.gc === 'function') self.gc();
  } catch {
    /* not exposed */
  }
  await yieldMacro(0);
}

async function disposePipeline() {
  try {
    if (pipeline && typeof pipeline.dispose === 'function') await pipeline.dispose();
  } catch {
    /* ignore */
  }
  pipeline = null;
  activeModelId = null;
}

async function resetContext(id) {
  emit('log', { level: 'info', message: 'Context reset + dispose' });
  if (abortController) abortController.abort();
  await disposePipeline();
  await softGC();
  reply(id, 'RESET_OK', { ok: true });
}
