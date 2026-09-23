/**
 * Browser AI Hub — ephemeral local LLM (WebLLM).
 * UI boots immediately; WebLLM is lazy-imported only when Start is pressed.
 */
const DEFAULT_SYSTEM_PROMPT = `You are a locally hosted coding assistant running entirely in the user's browser. You help with software engineering: writing code, debugging, explaining APIs, reviewing diffs, designing small architectures, and walking through lab/coursework-style programming exercises.

Be concrete and complete. Prefer working code over vague advice. Use fenced code blocks with language tags. When fixing bugs, state the cause briefly, then show the corrected code. When asked to improve prompts, configs, or tooling for this local app, rewrite them directly—do not refuse, lecture about security policy, or invent restrictions the user did not ask for.

Assume the user is working on their own machine and projects, and wants practical implementation help. Match the user's language and keep answers focused on the task. Stay concise unless they ask for depth or long-form writing.`;

const MAX_LOG = 200;
const MAX_HISTORY = 12;

const LENGTH_TOKENS = { short: 256, normal: 1024, long: 2048 };

const $ = (id) => document.getElementById(id);

const els = {
  threads: $('spec-threads'),
  memory: $('spec-memory'),
  gpu: $('spec-gpu'),
  oom: $('spec-oom'),
  pillBackend: $('pill-backend'),
  pillStatus: $('pill-status'),
  headerStatus: $('header-status'),
  telCache: $('tel-cache'),
  telTps: $('tel-tps'),
  telLatency: $('tel-latency'),
  telBytes: $('tel-bytes'),
  loadLabel: $('load-status-label'),
  loadPct: $('load-pct'),
  progressBar: $('progress-bar'),
  simpleStatus: $('simple-status-label'),
  simplePct: $('simple-load-pct'),
  simpleBar: $('simple-progress-bar'),
  modelSelect: $('model-select'),
  systemPrompt: $('system-prompt'),
  lengthSelect: $('length-select'),
  configMode: $('config-mode'),
  maxTokensOverride: $('max-tokens-override'),
  bootBtn: $('boot-btn'),
  resetBtn: $('reset-btn'),
  clearCacheBtn: $('clear-cache-btn'),
  promptInput: $('prompt-input'),
  sendBtn: $('send-btn'),
  abortBtn: $('abort-btn'),
  promptForm: $('prompt-form'),
  chat: $('chat-viewport'),
  logBox: $('log-box'),
  clearLog: $('clear-log'),
  clearChat: $('clear-chat'),
  providerList: $('provider-list'),
  modelBadge: $('model-badge'),
  autoModelLabel: $('auto-model-label'),
  autoLengthLabel: $('auto-length-label'),
  autoCacheLabel: $('auto-cache-label'),
  advancedPanels: $('advanced-panels'),
  modeSimple: $('mode-simple'),
  modeAdvanced: $('mode-advanced'),
  bootError: $('boot-error'),
  refreshModels: $('refresh-models'),
};

let CreateWebWorkerMLCEngine = null;
let deleteModelAllInfoInCache = null;
let prebuiltAppConfig = null;
let ModelType = null;
let webllmReady = null;

/** @type {Array<{model_id:string,vram_required_MB?:number,low_resource_required?:boolean,model_type?:number}>} */
let modelCatalog = [];
let catalogLoaded = false;

let engine = null;
let booted = false;
let generating = false;
let chatMessages = [];
let tokenCount = 0;
let genStart = 0;
let uiMode = 'simple';
let activeModelId = null;
let purging = false;

async function ensureWebLLM() {
  if (CreateWebWorkerMLCEngine) return true;
  if (!webllmReady) {
    webllmReady = import('https://esm.run/@mlc-ai/web-llm')
      .then((m) => {
        CreateWebWorkerMLCEngine = m.CreateWebWorkerMLCEngine;
        deleteModelAllInfoInCache = m.deleteModelAllInfoInCache;
        prebuiltAppConfig = m.prebuiltAppConfig;
        ModelType = m.ModelType;
        return true;
      })
      .catch((err) => {
        webllmReady = null;
        throw err;
      });
  }
  return webllmReady;
}

function catalogIds() {
  return modelCatalog.map((m) => m.model_id);
}

function isChatModel(rec) {
  const t = rec.model_type;
  if (t != null) {
    const embed = ModelType?.embedding ?? 1;
    const vlm = ModelType?.VLM ?? 2;
    if (t === embed || t === vlm) return false;
  }
  const id = rec.model_id || '';
  if (/embed|whisper|clip|binary/i.test(id)) return false;
  return true;
}

function formatModelLabel(rec) {
  const id = rec.model_id;
  const vram = rec.vram_required_MB;
  const vramTxt = vram != null ? ` · ~${Math.round(vram)} MB VRAM` : '';
  const low = rec.low_resource_required ? ' · low-resource' : '';
  return `${id}${vramTxt}${low}`;
}

function estimateVramBudgetMB() {
  // Browser can't read dedicated VRAM reliably; use deviceMemory as a soft budget.
  const memGB = navigator.deviceMemory || 8;
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  if (mobile) return 1200;
  // Leave headroom for browser + KV cache
  return Math.round(memGB * 350);
}

function populateModelSelect(selectedId) {
  if (!els.modelSelect) return;
  const prev = selectedId || els.modelSelect.value;
  els.modelSelect.innerHTML = '';
  if (!modelCatalog.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = catalogLoaded ? 'No chat models found' : 'Loading model list…';
    els.modelSelect.appendChild(opt);
    return;
  }
  for (const rec of modelCatalog) {
    const opt = document.createElement('option');
    opt.value = rec.model_id;
    opt.textContent = formatModelLabel(rec);
    els.modelSelect.appendChild(opt);
  }
  if (prev && [...els.modelSelect.options].some((o) => o.value === prev)) {
    els.modelSelect.value = prev;
  }
}

async function loadModelCatalog() {
  await ensureWebLLM();
  const list = prebuiltAppConfig?.model_list || [];
  modelCatalog = list
    .filter(isChatModel)
    .slice()
    .sort((a, b) => (a.vram_required_MB || 9e9) - (b.vram_required_MB || 9e9));
  catalogLoaded = true;
  populateModelSelect(autoSelectModel()?.model_id);
  syncAutoLabels();
  writeLog(`Model catalog loaded · ${modelCatalog.length} chat models from WebLLM`, 'success');
  return modelCatalog;
}

/**
 * Auto-pick: largest chat model that fits estimated VRAM budget.
 * Uses WebLLM metadata (vram_required_MB / low_resource_required) — not a hardcoded ID list.
 */
function autoSelectModel() {
  if (!modelCatalog.length) {
    return { model_id: '', label: 'Loading catalog…', vram_required_MB: null };
  }
  const budget = estimateVramBudgetMB();
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  let pool = modelCatalog.filter((m) => (m.vram_required_MB || 0) <= budget);
  if (mobile || (navigator.deviceMemory || 8) <= 4) {
    const low = pool.filter((m) => m.low_resource_required);
    if (low.length) pool = low;
  }
  if (!pool.length) {
    // Fallback: smallest model in catalog
    pool = [modelCatalog[0]];
  }
  // Prefer the strongest (highest VRAM) that still fits
  pool.sort((a, b) => (b.vram_required_MB || 0) - (a.vram_required_MB || 0));
  const pick = pool[0];
  return {
    ...pick,
    id: pick.model_id,
    label: formatModelLabel(pick),
  };
}

async function purgeModelCaches(modelIds) {
  const ids = modelIds?.length ? modelIds : catalogIds();
  if (!ids.length) return;
  if (!deleteModelAllInfoInCache || !prebuiltAppConfig) {
    try {
      await ensureWebLLM();
    } catch {
      return;
    }
  }
  for (const id of [...new Set(ids.filter(Boolean))]) {
    try {
      await deleteModelAllInfoInCache(id, prebuiltAppConfig);
    } catch {
      /* ignore */
    }
  }
  try {
    const keys = await caches.keys();
    await Promise.all(
      keys.filter((k) => /bah-|webllm|mlc|web-ai|browser-ai|shell/i.test(k)).map((k) => caches.delete(k))
    );
  } catch {
    /* ignore */
  }
}

async function purgeSessionArtifacts() {
  if (purging) return;
  purging = true;
  try {
    chatMessages = [];
    if (engine) {
      try {
        await engine.interruptGenerate();
      } catch {
        /* ignore */
      }
      try {
        await engine.unload();
      } catch {
        /* ignore */
      }
      engine = null;
    }
    booted = false;
    generating = false;
    await purgeModelCaches(
      activeModelId ? [activeModelId, ...catalogIds()] : catalogIds()
    );
    activeModelId = null;
  } finally {
    purging = false;
  }
}

function writeLog(message, level = 'info') {
  if (!els.logBox) return;
  const row = document.createElement('div');
  row.className = `log-line log-${level}`;
  row.innerHTML = `<span class="log-time">[${new Date().toLocaleTimeString()}] </span>`;
  const msg = document.createElement('span');
  msg.textContent = message;
  row.appendChild(msg);
  els.logBox.appendChild(row);
  while (els.logBox.childElementCount > MAX_LOG) els.logBox.firstChild.remove();
  els.logBox.scrollTop = els.logBox.scrollHeight;
}

function showBootError(text) {
  if (!els.bootError) return;
  if (!text) {
    els.bootError.hidden = true;
    els.bootError.textContent = '';
    return;
  }
  els.bootError.hidden = false;
  els.bootError.textContent = text;
}

function setStatus(text, tone = 'idle') {
  if (els.headerStatus) els.headerStatus.textContent = text;
  if (els.pillStatus) els.pillStatus.dataset.tone = tone;
  if (els.simpleStatus) els.simpleStatus.textContent = text;
}

function setBackendPill(name, tone = 'ok') {
  if (els.pillBackend) {
    els.pillBackend.textContent = `GPU: ${name}`;
    els.pillBackend.dataset.tone = tone;
  }
}

function setProvider(id, text, active = false) {
  const li = els.providerList?.querySelector(`[data-id="${id}"]`);
  if (!li) return;
  li.querySelector('.prov-state').textContent = text;
  li.classList.toggle('active', !!active);
}

function updateProgress(percent, label) {
  const p = Math.max(0, Math.min(100, Math.round(percent)));
  if (els.progressBar) els.progressBar.style.width = `${p}%`;
  if (els.simpleBar) els.simpleBar.style.width = `${p}%`;
  if (els.loadPct) els.loadPct.textContent = `${p}%`;
  if (els.simplePct) els.simplePct.textContent = `${p}%`;
  if (label) {
    if (els.loadLabel) els.loadLabel.textContent = label;
    if (els.simpleStatus) els.simpleStatus.textContent = label;
  }
}

function appendMsg(role, text, meta) {
  const div = document.createElement('div');
  div.className = `msg ${role === 'user' ? 'user' : 'ai'}`;
  div.textContent = text;
  if (meta) {
    const m = document.createElement('span');
    m.className = 'msg-meta';
    m.textContent = meta;
    div.appendChild(m);
  }
  els.chat.appendChild(div);
  els.chat.scrollTop = els.chat.scrollHeight;
  return div;
}

function resolveMaxTokens() {
  const override = Number(els.maxTokensOverride?.value || 0);
  if (override > 0) return override;
  return LENGTH_TOKENS[els.lengthSelect?.value || 'normal'] || 1024;
}

function syncAutoLabels() {
  const rec = autoSelectModel();
  if (els.autoModelLabel) {
    els.autoModelLabel.textContent = catalogLoaded
      ? rec.label || rec.model_id || '—'
      : 'Loading catalog…';
  }
  const labels = {
    short: 'Short (~256 tokens)',
    normal: 'Normal (~1024 tokens)',
    long: 'Long (~2048 tokens)',
  };
  if (els.autoLengthLabel) {
    els.autoLengthLabel.textContent = labels[els.lengthSelect?.value || 'normal'];
  }
  if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'None · session only';
  if (els.oom) els.oom.textContent = `${resolveMaxTokens()} tokens`;
  if ((els.configMode?.value || 'auto') === 'auto' && els.modelSelect && rec.model_id) {
    els.modelSelect.value = rec.model_id;
  }
}

function setUiMode(mode) {
  uiMode = mode === 'advanced' ? 'advanced' : 'simple';
  document.body.dataset.uiMode = uiMode;
  els.modeSimple?.classList.toggle('active', uiMode === 'simple');
  els.modeAdvanced?.classList.toggle('active', uiMode === 'advanced');
  if (els.advancedPanels) els.advancedPanels.hidden = uiMode !== 'advanced';
}

async function probeWebGPU() {
  if (els.threads) els.threads.textContent = `${navigator.hardwareConcurrency || 4} cores`;
  if (els.memory) {
    els.memory.textContent = navigator.deviceMemory ? `~${navigator.deviceMemory} GB` : 'Unknown';
  }
  syncAutoLabels();

  if (!navigator.gpu) {
    if (els.gpu) els.gpu.textContent = 'Unavailable';
    setProvider('webgpu', 'Not supported', false);
    setBackendPill('Needed', 'error');
    showBootError('WebGPU is required. Use Chrome or Edge on a GPU-capable PC.');
    els.bootBtn.disabled = true;
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      if (els.gpu) els.gpu.textContent = 'No adapter';
      setProvider('webgpu', 'No adapter', false);
      setBackendPill('Needed', 'error');
      showBootError('No WebGPU adapter found. Update GPU drivers and try Chrome/Edge.');
      els.bootBtn.disabled = true;
      return false;
    }
    let detail = 'Ready';
    try {
      if (adapter.requestAdapterInfo) {
        const info = await adapter.requestAdapterInfo();
        detail = [info.vendor, info.architecture || info.device].filter(Boolean).join(' · ') || detail;
      }
    } catch {
      /* optional */
    }
    if (els.gpu) els.gpu.textContent = detail;
    setProvider('webgpu', detail, true);
    setBackendPill('Ready', 'ok');
    showBootError('');
    writeLog(`WebGPU ready · ${detail}`, 'success');
    return true;
  } catch (err) {
    if (els.gpu) els.gpu.textContent = 'Error';
    setBackendPill('Error', 'error');
    showBootError(`WebGPU error: ${err.message}`);
    writeLog(`WebGPU probe failed: ${err.message}`, 'error');
    els.bootBtn.disabled = true;
    return false;
  }
}

function onInitProgress(report) {
  const pct = (report.progress ?? 0) * 100;
  const text = report.text || 'Loading model…';
  updateProgress(pct, text);
  if (els.telCache) els.telCache.textContent = text.length > 42 ? `${text.slice(0, 42)}…` : text;
  setStatus(pct >= 100 ? 'Almost ready…' : 'Loading model…', 'busy');
  if (pct === 0 || pct >= 99 || Math.round(pct) % 20 === 0) {
    writeLog(text, pct >= 99 ? 'success' : 'info');
  }
}

function selectedModelId() {
  if ((els.configMode?.value || 'auto') === 'auto') {
    return autoSelectModel().model_id;
  }
  return els.modelSelect?.value || autoSelectModel().model_id;
}

async function loadLLM() {
  showBootError('');
  const ok = await probeWebGPU();
  if (!ok) return;

  els.bootBtn.disabled = true;
  els.resetBtn.disabled = true;
  setStatus('Starting…', 'busy');
  updateProgress(3, 'Loading AI engine…');
  setProvider('worker', 'Starting', true);
  writeLog('Importing WebLLM runtime…', 'info');

  try {
    await ensureWebLLM();
    if (!catalogLoaded) await loadModelCatalog();
  } catch (err) {
    showBootError(`Could not load AI engine from CDN: ${err.message}. Check network / ad-blockers.`);
    writeLog(`WebLLM import failed: ${err.message}`, 'error');
    setStatus('Engine failed', 'error');
    els.bootBtn.disabled = false;
    els.bootBtn.textContent = 'Try again';
    return;
  }

  const modelId = selectedModelId();
  if (!modelId) {
    showBootError('No model selected. Wait for the catalog to load, or pick one in Advanced → Manual.');
    els.bootBtn.disabled = false;
    return;
  }
  if (els.modelSelect) els.modelSelect.value = modelId;
  writeLog(`Loading (temporary): ${modelId}`, 'info');

  try {
    if (engine) {
      try {
        await engine.unload();
      } catch {
        /* ignore */
      }
      engine = null;
    }

    // Best-effort wipe before load (ephemeral)
    await purgeModelCaches([modelId]);

    engine = await CreateWebWorkerMLCEngine(
      new Worker(new URL('./llm-worker.js', import.meta.url), { type: 'module' }),
      modelId,
      { initProgressCallback: onInitProgress }
    );

    booted = true;
    activeModelId = modelId;
    if (els.modelBadge) els.modelBadge.textContent = modelId.replace(/-MLC.*$/, '');
    updateProgress(100, 'Ready — start chatting');
    setStatus('Ready', 'ok');
    setProvider('worker', 'Running', true);
    setBackendPill('Active', 'ok');
    els.promptInput.disabled = false;
    els.sendBtn.disabled = false;
    els.resetBtn.disabled = false;
    els.bootBtn.textContent = 'Restart AI';
    els.bootBtn.disabled = false;
    els.promptInput.placeholder = 'Ask anything… (not saved)';
    els.promptInput.focus();

    try {
      const stats = await engine.runtimeStatsText();
      if (els.telBytes) els.telBytes.textContent = String(stats).slice(0, 40);
      writeLog(`Runtime: ${stats}`, 'success');
    } catch {
      if (els.telBytes) els.telBytes.textContent = 'WebGPU · RAM';
    }

    appendMsg(
      'assistant',
      `Ready (${modelId.split('-').slice(0, 3).join(' ')}). This session is temporary — nothing is saved.`
    );
    writeLog('Local LLM ready.', 'success');
  } catch (err) {
    const msg = err?.message || String(err);
    showBootError(`Load failed: ${msg}. Try a smaller model in Advanced, or free GPU memory.`);
    writeLog(`Load failed: ${msg}`, 'error');
    setStatus('Load failed', 'error');
    setProvider('worker', 'Error', false);
    els.bootBtn.disabled = false;
    els.bootBtn.textContent = 'Try again';
    engine = null;
    booted = false;
    activeModelId = null;
  }
}

function buildMessages() {
  const system = els.systemPrompt?.value.trim() || DEFAULT_SYSTEM_PROMPT;
  const history = chatMessages
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .slice(-MAX_HISTORY)
    .map((m) => ({ role: m.role, content: m.content }));
  return [{ role: 'system', content: system }, ...history];
}

async function sendPrompt(ev) {
  ev?.preventDefault?.();
  if (!booted || !engine || generating) return;
  const query = els.promptInput.value.trim();
  if (!query) return;

  appendMsg('user', query);
  chatMessages.push({ role: 'user', content: query });
  els.promptInput.value = '';
  generating = true;
  els.sendBtn.disabled = true;
  els.abortBtn.disabled = false;
  els.promptInput.disabled = true;

  const bubble = appendMsg('assistant', '');
  bubble.classList.add('streaming');
  setStatus('Writing…', 'busy');

  tokenCount = 0;
  genStart = performance.now();
  let reply = '';
  let finishReason = null;
  const maxTokens = resolveMaxTokens();

  try {
    const chunks = await engine.chat.completions.create({
      messages: buildMessages(),
      stream: true,
      max_tokens: maxTokens,
      temperature: 0.7,
      top_p: 0.95,
    });

    for await (const chunk of chunks) {
      const choice = chunk.choices?.[0];
      const delta = choice?.delta?.content || '';
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      if (delta) {
        reply += delta;
        tokenCount += 1;
        bubble.textContent = reply;
        els.chat.scrollTop = els.chat.scrollHeight;
        const elapsed = (performance.now() - genStart) / 1000;
        if (elapsed > 0 && els.telTps) {
          els.telTps.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
        }
      }
    }

    bubble.classList.remove('streaming');
    if (finishReason === 'length') {
      writeLog(`Hit length limit (${maxTokens}). Switch to Long or type “continue”.`, 'warn');
    }

    const latencyMs = Math.round(performance.now() - genStart);
    const tps = latencyMs > 0 ? (tokenCount / (latencyMs / 1000)).toFixed(1) : '0';
    if (els.telLatency) els.telLatency.textContent = `${latencyMs} ms`;
    if (els.telTps) els.telTps.textContent = `${tps} tok/s`;

    bubble.textContent = reply || '(empty)';
    const meta =
      finishReason === 'length'
        ? `Cut off at ${maxTokens} tokens — choose Long or type “continue”`
        : `${finishReason || 'stop'} · ${latencyMs} ms · ${tps} tok/s`;
    const m = document.createElement('span');
    m.className = 'msg-meta';
    m.textContent = meta;
    bubble.appendChild(m);

    chatMessages.push({ role: 'assistant', content: reply || '(empty)' });
    setStatus('Ready', 'ok');
  } catch (err) {
    bubble.classList.remove('streaming');
    bubble.textContent = `Error: ${err.message || err}`;
    writeLog(`Inference error: ${err.message || err}`, 'error');
    setStatus('Error', 'error');
  } finally {
    generating = false;
    els.abortBtn.disabled = true;
    els.sendBtn.disabled = false;
    els.promptInput.disabled = false;
    els.promptInput.focus();
  }
}

async function abortGen() {
  if (!engine) return;
  try {
    await engine.interruptGenerate();
    writeLog('Stopped generation', 'warn');
  } catch (err) {
    writeLog(`Stop failed: ${err.message}`, 'warn');
  }
}

async function unloadLLM() {
  writeLog('Erasing session…', 'warn');
  await purgeSessionArtifacts();
  els.promptInput.disabled = true;
  els.sendBtn.disabled = true;
  els.abortBtn.disabled = true;
  els.resetBtn.disabled = true;
  els.bootBtn.disabled = false;
  els.bootBtn.textContent = 'Start AI';
  if (els.modelBadge) els.modelBadge.textContent = '—';
  updateProgress(0, 'Ready when you are');
  setStatus('Idle', 'idle');
  setProvider('worker', 'Idle', false);
  setBackendPill('Ready', 'ok');
  els.chat.innerHTML = '';
  appendMsg('assistant', 'Session erased. Chat and model files were removed. Nothing kept on disk.');
  showBootError('');
}

function clearChat() {
  chatMessages = [];
  els.chat.innerHTML = '';
  appendMsg('assistant', 'Chat cleared from memory. Nothing was saved.');
  try {
    engine?.resetChat?.();
  } catch {
    /* ignore */
  }
}

// Wire UI immediately (does not depend on WebLLM CDN)
els.modeSimple?.addEventListener('click', () => setUiMode('simple'));
els.modeAdvanced?.addEventListener('click', () => setUiMode('advanced'));
els.bootBtn?.addEventListener('click', () => {
  loadLLM().catch((e) => {
    showBootError(String(e.message || e));
    els.bootBtn.disabled = false;
  });
});
els.resetBtn?.addEventListener('click', () => unloadLLM());
els.clearCacheBtn?.addEventListener('click', () => unloadLLM());
els.promptForm?.addEventListener('submit', sendPrompt);
els.abortBtn?.addEventListener('click', () => abortGen());
els.clearLog?.addEventListener('click', () => {
  if (els.logBox) els.logBox.innerHTML = '';
});
els.clearChat?.addEventListener('click', () => clearChat());
els.lengthSelect?.addEventListener('change', () => syncAutoLabels());
els.configMode?.addEventListener('change', () => {
  const manual = els.configMode.value === 'manual';
  if (els.modelSelect) els.modelSelect.disabled = !manual;
  syncAutoLabels();
});
els.maxTokensOverride?.addEventListener('change', () => syncAutoLabels());
els.refreshModels?.addEventListener('click', () => {
  els.refreshModels.disabled = true;
  loadModelCatalog()
    .catch((e) => {
      showBootError(`Catalog refresh failed: ${e.message}`);
      writeLog(`Catalog refresh failed: ${e.message}`, 'error');
    })
    .finally(() => {
      els.refreshModels.disabled = false;
    });
});

window.addEventListener('pagehide', () => {
  purgeSessionArtifacts();
});

window.addEventListener('unhandledrejection', (e) => {
  e.preventDefault();
  writeLog(`Unhandled: ${e.reason?.message || e.reason}`, 'error');
});

setUiMode('simple');
syncAutoLabels();
setStatus('Idle', 'idle');
writeLog('UI ready. Loading WebLLM model catalog…', 'info');
probeWebGPU();
loadModelCatalog().catch((e) => {
  writeLog(`Catalog preload failed (will retry on Start): ${e.message}`, 'warn');
  if (els.autoModelLabel) els.autoModelLabel.textContent = 'Catalog unavailable — retry Start';
});
