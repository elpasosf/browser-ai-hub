/**
 * Browser AI Hub — main-thread controller
 * UI stays free; all heavy work is in worker.js
 */

const isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
const deviceProfile = {
  hardwareConcurrency: navigator.hardwareConcurrency || 4,
  deviceMemory: navigator.deviceMemory || null,
  isMobile,
  userAgent: navigator.userAgent,
};

const PREFS_KEY = 'bah.prefs.v1';
const MAX_LOG_LINES = 200;

const $ = (id) => document.getElementById(id);
const els = {
  threads: $('spec-threads'),
  memory: $('spec-memory'),
  alloc: $('spec-alloc'),
  oom: $('spec-oom'),
  pillBackend: $('pill-backend'),
  pillStatus: $('pill-status'),
  pillPrivacy: $('pill-privacy'),
  headerStatus: $('header-status'),
  telCache: $('tel-cache'),
  telSpeed: $('tel-speed'),
  telBytes: $('tel-bytes'),
  telTps: $('tel-tps'),
  telLatency: $('tel-latency'),
  loadLabel: $('load-status-label'),
  loadPct: $('load-pct'),
  progressBar: $('progress-bar'),
  progressTrack: $('progress-track'),
  backendSelect: $('backend-select'),
  modelSelect: $('model-select'),
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
};

let worker = null;
let reqSeq = 0;
const pending = new Map();
let booted = false;
let streamingBubble = null;
let chatMessages = [];
let generating = false;

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}');
  } catch {
    return {};
  }
}

function savePrefs(patch) {
  const next = { ...loadPrefs(), ...patch };
  localStorage.setItem(PREFS_KEY, JSON.stringify(next));
}

function writeLog(message, level = 'info') {
  const row = document.createElement('div');
  row.className = `log-line log-${level}`;
  const t = new Date().toLocaleTimeString();
  const time = document.createElement('span');
  time.className = 'log-time';
  time.textContent = `[${t}] `;
  const msg = document.createElement('span');
  msg.className = 'log-msg';
  msg.textContent = message;
  row.append(time, msg);
  els.logBox.appendChild(row);
  while (els.logBox.childElementCount > MAX_LOG_LINES) {
    els.logBox.firstElementChild.remove();
  }
  els.logBox.scrollTop = els.logBox.scrollHeight;
}

function setStatus(text, tone = 'idle') {
  els.headerStatus.textContent = text;
  els.pillStatus.dataset.tone = tone;
}

function setBackendPill(name) {
  els.pillBackend.textContent = `Backend: ${name || '—'}`;
  const n = String(name || '').toLowerCase();
  els.pillBackend.dataset.tone =
    n.includes('webgpu') ? 'ok' : n.includes('wasm') ? 'warn' : 'idle';
}

function updateProgress(percent, label) {
  const p = Math.max(0, Math.min(100, percent | 0));
  els.progressBar.style.width = `${p}%`;
  els.loadPct.textContent = `${p}%`;
  els.progressTrack.setAttribute('aria-valuenow', String(p));
  if (label) els.loadLabel.textContent = label;
}

function setProviderState(id, text, active = false) {
  const li = els.providerList.querySelector(`[data-id="${id}"]`);
  if (!li) return;
  li.querySelector('.prov-state').textContent = text;
  li.classList.toggle('active', active);
}

function yieldToUI() {
  return new Promise((r) => {
    requestAnimationFrame(() => setTimeout(r, 0));
  });
}

function appendMsg(role, text, meta) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
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

function persistChatSoon() {
  const run = () => {
    callWorker('SAVE_CHAT', { messages: chatMessages }).catch(() => {});
  };
  if ('requestIdleCallback' in window) {
    requestIdleCallback(run, { timeout: 1500 });
  } else {
    setTimeout(run, 400);
  }
}

window.addEventListener('unhandledrejection', (e) => {
  e.preventDefault();
  const msg = e.reason?.message || String(e.reason);
  writeLog(`Unhandled rejection caught: ${msg}`, 'error');
  writeLog('Tip: reboot with Auto backend or Clear cache if quota-related.', 'warn');
  setStatus('Error trapped', 'error');
});

window.addEventListener('error', (e) => {
  writeLog(`Runtime error: ${e.message}`, 'error');
  setStatus('Error trapped', 'error');
});

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (ev) => onWorkerMessage(ev.data);
  worker.onerror = (err) => {
    writeLog(`Worker crash: ${err.message}`, 'error');
    writeLog('Tip: serve over http(s). Retry Auto backend.', 'warn');
    setStatus('Worker error', 'error');
    els.bootBtn.disabled = false;
  };
  return worker;
}

function callWorker(type, payload = {}) {
  const id = ++reqSeq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, type });
    ensureWorker().postMessage({ id, type, payload });
  });
}

function onWorkerMessage(data) {
  if (!data) return;
  if (data.type === 'EVENT') {
    handleEvent(data.channel, data.payload);
    return;
  }
  const waiter = pending.get(data.id);
  if (!waiter) return;
  pending.delete(data.id);
  if (data.type === 'ERROR') {
    writeLog(data.payload?.message || 'Worker error', 'error');
    if (data.payload?.tip) writeLog(`Tip: ${data.payload.tip}`, 'warn');
    waiter.reject(new Error(data.payload?.message || 'Worker error'));
    return;
  }
  waiter.resolve(data.payload);
}

function handleEvent(channel, payload) {
  switch (channel) {
    case 'log':
      writeLog(payload.message, payload.level || 'info');
      break;
    case 'progress': {
      const pct = payload.percent ?? 0;
      const label = payload.cached
        ? `Cache restore · ${payload.label || 'asset'}`
        : payload.text || `Downloading ${payload.label || 'asset'}`;
      updateProgress(pct, label);
      if (payload.speedMBps != null) {
        els.telSpeed.textContent = `${Number(payload.speedMBps).toFixed(2)} MB/s`;
      }
      if (payload.total || payload.loaded) {
        els.telBytes.textContent = `${fmt(payload.loaded)} / ${fmt(payload.total || payload.loaded)}`;
      }
      setStatus(payload.cached ? 'Loading from cache' : 'Streaming weights', 'busy');
      break;
    }
    case 'backends':
      applyBackendReport(payload);
      break;
    case 'cache':
      renderCache(payload);
      break;
    case 'limits':
      els.oom.textContent = `${payload.maxNewTokens} tok · ctx ${payload.contextChars}`;
      break;
    case 'telemetry':
      if (payload.threadsAllocated != null) {
        els.alloc.textContent = `${payload.threadsAllocated} workers`;
      }
      break;
    case 'stage':
      setStatus(payload.message || payload.stage, 'busy');
      updateProgress(stagePercent(payload.stage), payload.message || payload.stage);
      break;
    case 'token':
      if (streamingBubble) {
        // Keep meta node if present
        const meta = streamingBubble.querySelector('.msg-meta');
        streamingBubble.textContent = payload.text || '';
        if (meta) streamingBubble.appendChild(meta);
        els.chat.scrollTop = els.chat.scrollHeight;
      }
      if (payload.tps != null) {
        els.telTps.textContent = `${Number(payload.tps).toFixed(1)} tok/s`;
      }
      break;
    case 'error':
      writeLog(payload.message, 'error');
      if (payload.tip) writeLog(`Tip: ${payload.tip}`, 'warn');
      setStatus('Recoverable error', 'error');
      break;
    default:
      break;
  }
}

function renderCache(payload) {
  if (!payload) return;
  const quota =
    payload.quotaMB != null ? ` · ${payload.usageMB}/${payload.quotaMB} MB used` : '';
  els.telCache.textContent = payload.cacheEntries
    ? `${payload.cacheEntries} files · ${payload.cacheMB} MB${quota}`
    : `Cold (empty)${quota}`;
}

function stagePercent(stage) {
  return (
    {
      shaders: 18,
      download: 48,
      compile: 72,
      warmup: 90,
      ready: 100,
      generate: 100,
      idle: 100,
    }[stage] ?? 50
  );
}

function applyBackendReport(r) {
  setProviderState(
    'webgpu',
    r.webgpu?.available ? r.webgpu.detail || 'Ready' : r.webgpu?.detail || 'Unavailable',
    r.selected === 'webgpu' || r.active === 'webgpu'
  );
  setProviderState(
    'wasm',
    r.wasm?.detail || 'Baseline',
    r.selected === 'wasm' || r.active === 'wasm'
  );
  setProviderState(
    'cpu',
    r.cpu?.detail || 'Worker',
    r.selected === 'cpu' || r.active === 'cpu'
  );
  const active = r.active || r.selected;
  if (active) setBackendPill(active.toUpperCase());
  if (r.demoted) writeLog(r.demoteReason || 'Backend demoted for memory safety', 'warn');
}

function fmt(n) {
  if (!n) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i ? 2 : 0)} ${u[i]}`;
}

async function initProfile() {
  const prefs = loadPrefs();
  if (prefs.backend) els.backendSelect.value = prefs.backend;
  if (prefs.modelId) els.modelSelect.value = prefs.modelId;

  els.threads.textContent = `${deviceProfile.hardwareConcurrency} cores`;
  els.memory.textContent = deviceProfile.deviceMemory
    ? `~${deviceProfile.deviceMemory} GB`
    : 'Unknown';
  els.alloc.textContent = 'Pending boot';
  els.oom.textContent = 'Profiling…';
  els.pillPrivacy.dataset.tone = 'ok';
  writeLog('Main thread online · privacy: on-device inference', 'info');
  await yieldToUI();

  try {
    ensureWorker();
    const profile = await callWorker('PROFILE', deviceProfile);
    if (profile?.limits) {
      els.oom.textContent = `${profile.limits.maxNewTokens} tok · ctx ${profile.limits.contextChars}`;
    }
    const backends = await callWorker('PROBE_BACKENDS', {
      preference: els.backendSelect.value,
    });
    applyBackendReport(backends);
    const cache = await callWorker('CACHE_STATUS');
    renderCache(cache);
    const history = await callWorker('LOAD_CHAT');
    if (history?.messages?.length) {
      chatMessages = history.messages;
      els.chat.innerHTML = '';
      for (const m of chatMessages) {
        appendMsg(m.role, m.text, m.meta);
      }
      writeLog(`Restored ${chatMessages.length} local chat messages`, 'success');
    }
    writeLog(`Probe complete · preferred ${backends.selected.toUpperCase()}`, 'success');
    setStatus('Ready to boot', 'idle');
  } catch (err) {
    writeLog(`Probe failed: ${err.message}`, 'error');
    setProviderState('cpu', 'Available', true);
    setBackendPill('CPU');
  }
}

async function boot() {
  els.bootBtn.disabled = true;
  els.backendSelect.disabled = true;
  els.modelSelect.disabled = true;
  els.resetBtn.disabled = true;
  setStatus('Booting pipeline', 'busy');
  updateProgress(5, 'Spawning worker loader…');
  savePrefs({ backend: els.backendSelect.value, modelId: els.modelSelect.value });
  writeLog(`Boot · backend=${els.backendSelect.value} · model=${els.modelSelect.value}`, 'info');
  await yieldToUI();

  try {
    ensureWorker();
    await callWorker('PROFILE', deviceProfile);
    const result = await callWorker('BOOT', {
      backend: els.backendSelect.value,
      modelId: els.modelSelect.value,
    });

    booted = true;
    setBackendPill((result.backend || 'cpu').toUpperCase());
    els.modelBadge.textContent = result.modelId || '—';
    updateProgress(100, result.fallback ? 'Hybrid engine ready' : 'Ready');
    setStatus('Ready', 'ok');
    els.promptInput.disabled = false;
    els.sendBtn.disabled = false;
    els.promptInput.placeholder = 'Ask the local model…';
    els.promptInput.focus();
    els.resetBtn.disabled = false;
    els.bootBtn.textContent = 'Pipeline active';
    renderCache(result.cache);
    const note = result.fallback
      ? 'Local hybrid engine active (CDN/runtime unavailable). Math + privacy Q&A work on-device; full neural gen needs a one-time weight download.'
      : `Pipeline live on ${String(result.backend).toUpperCase()} · ${result.modelId}. Weights cached locally — subsequent boots skip the network.`;
    appendMsg('ai', note);
    chatMessages.push({ role: 'ai', text: note });
    persistChatSoon();
    writeLog('Pipeline sequence finished successfully', 'success');
  } catch (err) {
    writeLog(`Boot failed: ${err.message}`, 'error');
    setStatus('Boot failed', 'error');
    els.bootBtn.disabled = false;
    els.backendSelect.disabled = false;
    els.modelSelect.disabled = false;
    els.bootBtn.textContent = 'Retry boot';
  }
}

async function sendPrompt(ev) {
  ev?.preventDefault?.();
  if (!booted || generating) return;
  const query = els.promptInput.value.trim();
  if (!query) return;

  appendMsg('user', query);
  chatMessages.push({ role: 'user', text: query });
  els.promptInput.value = '';
  generating = true;
  els.sendBtn.disabled = true;
  els.abortBtn.disabled = false;
  els.promptInput.disabled = true;
  streamingBubble = appendMsg('ai', '');
  streamingBubble.classList.add('streaming');
  setStatus('Generating', 'busy');
  els.telTps.textContent = '0.0 tok/s';
  els.telLatency.textContent = '—';

  try {
    const result = await callWorker('INFER', { prompt: query });
    if (streamingBubble) {
      streamingBubble.classList.remove('streaming');
      const meta = `${result.backend || '?'} · ${result.modelId || '?'} · ${
        result.latencyMs ?? '?'
      } ms · ${(result.tps || 0).toFixed(1)} tok/s`;
      streamingBubble.textContent = result.text || '(empty)';
      const m = document.createElement('span');
      m.className = 'msg-meta';
      m.textContent = meta;
      streamingBubble.appendChild(m);
      chatMessages.push({ role: 'ai', text: result.text || '', meta });
    }
    if (result.tps != null) els.telTps.textContent = `${Number(result.tps).toFixed(1)} tok/s`;
    if (result.latencyMs != null) els.telLatency.textContent = `${result.latencyMs} ms`;
    setStatus('Ready', 'ok');
    persistChatSoon();
  } catch (err) {
    if (streamingBubble) {
      streamingBubble.classList.remove('streaming');
      streamingBubble.textContent = `Error: ${err.message}`;
    }
    writeLog(`Inference error: ${err.message}`, 'error');
    setStatus('Inference error', 'error');
  } finally {
    streamingBubble = null;
    generating = false;
    els.abortBtn.disabled = true;
    els.sendBtn.disabled = false;
    els.promptInput.disabled = false;
    els.promptInput.focus();
  }
}

async function abortGen() {
  try {
    await callWorker('ABORT');
    writeLog('Abort requested', 'warn');
  } catch {
    /* ignore */
  }
}

async function resetPipeline() {
  writeLog('Resetting context & disposing weights…', 'warn');
  try {
    await callWorker('RESET');
  } catch {
    /* ignore */
  }
  booted = false;
  generating = false;
  els.promptInput.disabled = true;
  els.sendBtn.disabled = true;
  els.abortBtn.disabled = true;
  els.promptInput.placeholder = 'Initialize pipeline first…';
  els.bootBtn.disabled = false;
  els.backendSelect.disabled = false;
  els.modelSelect.disabled = false;
  els.resetBtn.disabled = true;
  els.bootBtn.textContent = 'Boot AI pipeline';
  els.modelBadge.textContent = '—';
  updateProgress(0, 'Standby');
  setStatus('Idle', 'idle');
  appendMsg('ai', 'Context cleared. Boot again when ready.');
}

async function clearCache() {
  writeLog('Clearing model Cache API + IndexedDB…', 'warn');
  try {
    const status = await callWorker('CLEAR_CACHE');
    renderCache(status);
    writeLog('Cache cleared', 'success');
    if (booted) await resetPipeline();
  } catch (err) {
    writeLog(`Clear cache failed: ${err.message}`, 'error');
  }
}

async function clearChatHistory() {
  chatMessages = [];
  els.chat.innerHTML = '';
  appendMsg(
    'ai',
    'Chat cleared from local IndexedDB. Model weights (if cached) are untouched.'
  );
  try {
    await callWorker('CLEAR_CHAT');
  } catch {
    /* ignore */
  }
}

els.bootBtn.addEventListener('click', () => boot().catch((e) => writeLog(e.message, 'error')));
els.resetBtn.addEventListener('click', () =>
  resetPipeline().catch((e) => writeLog(e.message, 'error'))
);
els.clearCacheBtn.addEventListener('click', () =>
  clearCache().catch((e) => writeLog(e.message, 'error'))
);
els.promptForm.addEventListener('submit', sendPrompt);
els.abortBtn.addEventListener('click', () => abortGen());
els.clearLog.addEventListener('click', () => {
  els.logBox.innerHTML = '';
  writeLog('Console cleared', 'info');
});
els.clearChat.addEventListener('click', () => clearChatHistory());
els.backendSelect.addEventListener('change', () =>
  savePrefs({ backend: els.backendSelect.value })
);
els.modelSelect.addEventListener('change', () =>
  savePrefs({ modelId: els.modelSelect.value })
);

// Register service worker for app-shell caching (https / localhost only)
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}

requestAnimationFrame(() => {
  writeLog('System initialized. Ready to safely spawn loader.', 'info');
  initProfile();
});
