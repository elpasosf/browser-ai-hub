/**
 * Browser AI Hub — WebLLM local chat with Simple (auto) + Advanced modes.
 * Privacy: NOTHING is persisted. Chat, prefs, and model weights are session-only.
 */
import {
  CreateWebWorkerMLCEngine,
  deleteModelAllInfoInCache,
  prebuiltAppConfig,
} from 'https://esm.run/@mlc-ai/web-llm';

const MAX_LOG = 220;
const MAX_HISTORY = 12;
const LEGACY_KEYS = [
  'bah.webllm.prefs.v1',
  'bah.webllm.prefs.v2',
  'bah.webllm.chat.v1',
  'bah.prefs.v1',
];

const MODELS = {
  tiny: {
    id: 'SmolLM2-360M-Instruct-q4f16_1-MLC',
    label: 'SmolLM2 360M (fastest)',
  },
  small: {
    id: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    label: 'Qwen2.5 0.5B (balanced)',
  },
  medium: {
    id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
    label: 'Llama 3.2 1B (smarter)',
  },
  large: {
    id: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
    label: 'Qwen2.5 1.5B (best quality)',
  },
};

const LENGTH_TOKENS = {
  short: 256,
  normal: 1024,
  long: 2048,
};

const ALL_MODEL_IDS = [
  ...Object.values(MODELS).map((m) => m.id),
  'TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC',
  'Phi-3.5-mini-instruct-q4f16_1-MLC-1k',
];

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
  telSpeed: $('tel-speed'),
  telTps: $('tel-tps'),
  telLatency: $('tel-latency'),
  telBytes: $('tel-bytes'),
  loadLabel: $('load-status-label'),
  loadPct: $('load-pct'),
  progressBar: $('progress-bar'),
  progressTrack: $('progress-track'),
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
};

/** @type {import('@mlc-ai/web-llm').MLCEngineInterface | null} */
let engine = null;
let booted = false;
let generating = false;
let chatMessages = [];
let tokenCount = 0;
let genStart = 0;
let uiMode = 'simple';
let recommended = MODELS.medium;
let activeModelId = null;
let purging = false;

function scrubLegacyStorage() {
  for (const k of LEGACY_KEYS) {
    try {
      localStorage.removeItem(k);
    } catch {
      /* ignore */
    }
    try {
      sessionStorage.removeItem(k);
    } catch {
      /* ignore */
    }
  }
}

async function purgeModelCaches(modelIds = ALL_MODEL_IDS) {
  const ids = [...new Set(modelIds.filter(Boolean))];
  for (const id of ids) {
    try {
      await deleteModelAllInfoInCache(id, prebuiltAppConfig);
    } catch {
      /* ignore */
    }
  }
  try {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((k) => /bah-|webllm|mlc|web-ai|browser-ai/i.test(k))
        .map((k) => caches.delete(k))
    );
  } catch {
    /* ignore */
  }
}

async function unregisterServiceWorkers() {
  if (!('serviceWorker' in navigator)) return;
  try {
    const regs = await navigator.serviceWorker.getRegistrations();
    await Promise.all(regs.map((r) => r.unregister()));
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
    await purgeModelCaches(activeModelId ? [activeModelId, ...ALL_MODEL_IDS] : ALL_MODEL_IDS);
    activeModelId = null;
  } finally {
    purging = false;
  }
}

function writeLog(message, level = 'info') {
  if (!els.logBox) return;
  const row = document.createElement('div');
  row.className = `log-line log-${level}`;
  const time = document.createElement('span');
  time.className = 'log-time';
  time.textContent = `[${new Date().toLocaleTimeString()}] `;
  const msg = document.createElement('span');
  msg.textContent = message;
  row.append(time, msg);
  els.logBox.appendChild(row);
  while (els.logBox.childElementCount > MAX_LOG) els.logBox.firstElementChild.remove();
  els.logBox.scrollTop = els.logBox.scrollHeight;
}

function setStatus(text, tone = 'idle') {
  els.headerStatus.textContent = text;
  els.pillStatus.dataset.tone = tone;
  if (els.simpleStatus) els.simpleStatus.textContent = text;
}

function setBackendPill(name, tone = 'ok') {
  els.pillBackend.textContent = `GPU: ${name}`;
  els.pillBackend.dataset.tone = tone;
}

function setProvider(id, text, active = false) {
  const li = els.providerList?.querySelector(`[data-id="${id}"]`);
  if (!li) return;
  li.querySelector('.prov-state').textContent = text;
  li.classList.toggle('active', active);
}

function updateProgress(percent, label) {
  const p = Math.max(0, Math.min(100, Math.round(percent)));
  const apply = (bar, pctEl, labelEl) => {
    if (bar) bar.style.width = `${p}%`;
    if (pctEl) pctEl.textContent = `${p}%`;
    if (labelEl && label) labelEl.textContent = label;
  };
  apply(els.progressBar, els.loadPct, els.loadLabel);
  apply(els.simpleBar, els.simplePct, els.simpleStatus);
  els.progressTrack?.setAttribute('aria-valuenow', String(p));
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

function autoSelectModel() {
  const mem = navigator.deviceMemory || 8;
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  if (mobile || mem <= 4) recommended = MODELS.tiny;
  else if (mem <= 8) recommended = MODELS.small;
  else if (mem <= 16) recommended = MODELS.medium;
  else recommended = MODELS.large;
  return recommended;
}

function resolveMaxTokens() {
  const override = Number(els.maxTokensOverride?.value || 0);
  if (override > 0) return override;
  const len = els.lengthSelect?.value || 'normal';
  return LENGTH_TOKENS[len] || LENGTH_TOKENS.normal;
}

function syncAutoLabels() {
  const rec = autoSelectModel();
  if (els.autoModelLabel) els.autoModelLabel.textContent = rec.label;
  const len = els.lengthSelect?.value || 'normal';
  const labels = {
    short: 'Short (~256 tokens)',
    normal: 'Normal (~1024 tokens)',
    long: 'Long (~2048 tokens)',
  };
  if (els.autoLengthLabel) els.autoLengthLabel.textContent = labels[len] || labels.normal;
  if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'None · session only';
  if (els.oom) els.oom.textContent = `${resolveMaxTokens()} max tokens`;
  if ((els.configMode?.value || 'auto') === 'auto') {
    els.modelSelect.value = rec.id;
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

  if (!('gpu' in navigator)) {
    if (els.gpu) els.gpu.textContent = 'Unavailable';
    setProvider('webgpu', 'Not supported', false);
    setBackendPill('Needed', 'error');
    writeLog('WebGPU missing. Use Chrome or Edge.', 'error');
    els.bootBtn.disabled = true;
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      if (els.gpu) els.gpu.textContent = 'No adapter';
      setProvider('webgpu', 'No adapter', false);
      setBackendPill('Needed', 'error');
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
    writeLog(`WebGPU ready · ${detail}`, 'success');
    return true;
  } catch (err) {
    if (els.gpu) els.gpu.textContent = 'Error';
    writeLog(`WebGPU probe failed: ${err.message}`, 'error');
    els.bootBtn.disabled = true;
    return false;
  }
}

function onInitProgress(report) {
  const pct = (report.progress ?? 0) * 100;
  const text = report.text || 'Loading model…';
  updateProgress(pct, text);
  if (els.telCache) {
    els.telCache.textContent = text.length > 48 ? `${text.slice(0, 48)}…` : text;
  }
  setStatus(pct >= 100 ? 'Almost ready…' : 'Loading into memory…', 'busy');
  if (pct === 0 || pct >= 99 || Math.round(pct) % 15 === 0) {
    writeLog(text, pct >= 99 ? 'success' : 'info');
  }
}

function selectedModelId() {
  if ((els.configMode?.value || 'auto') === 'auto') {
    return autoSelectModel().id;
  }
  return els.modelSelect.value;
}

async function loadLLM() {
  const ok = await probeWebGPU();
  if (!ok) return;

  const modelId = selectedModelId();
  els.modelSelect.value = modelId;

  els.bootBtn.disabled = true;
  els.modelSelect.disabled = true;
  els.systemPrompt.disabled = true;
  els.resetBtn.disabled = true;
  setStatus('Starting…', 'busy');
  updateProgress(2, 'Starting AI engine…');
  setProvider('worker', 'Starting', true);
  writeLog(`Loading (temporary): ${modelId}`, 'info');
  writeLog('Session-only: chat & model files erased when you leave or stop.', 'warn');

  try {
    if (engine) {
      try {
        await engine.unload();
      } catch {
        /* ignore */
      }
      engine = null;
    }
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
    if (els.clearCacheBtn) els.clearCacheBtn.disabled = false;
    els.bootBtn.textContent = uiMode === 'simple' ? 'Restart AI' : 'Reload model';
    els.bootBtn.disabled = false;
    els.modelSelect.disabled = false;
    els.systemPrompt.disabled = false;
    els.promptInput.placeholder = 'Ask anything… (not saved)';
    els.promptInput.focus();

    try {
      const stats = await engine.runtimeStatsText();
      if (els.telBytes) els.telBytes.textContent = String(stats).slice(0, 42);
      writeLog(`Runtime: ${stats}`, 'success');
    } catch {
      if (els.telBytes) els.telBytes.textContent = 'WebGPU · RAM only';
    }

    appendMsg(
      'assistant',
      `Ready with ${modelId.split('-').slice(0, 3).join(' ')}. Nothing is saved — chat and model data are temporary for this tab only.`
    );
    writeLog('Local LLM ready (ephemeral session).', 'success');
  } catch (err) {
    writeLog(`Load failed: ${err.message || err}`, 'error');
    writeLog('Tip: Advanced → smaller model, or close other GPU apps.', 'warn');
    setStatus('Couldn’t start', 'error');
    setProvider('worker', 'Error', false);
    els.bootBtn.disabled = false;
    els.modelSelect.disabled = false;
    els.systemPrompt.disabled = false;
    els.bootBtn.textContent = 'Try again';
    engine = null;
    booted = false;
    activeModelId = null;
  }
}

function buildMessages() {
  const system =
    els.systemPrompt.value.trim() ||
    'You are a helpful assistant. Give complete answers unless asked to be brief.';
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
      stream_options: { include_usage: true },
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
      if (chunk.usage && els.telSpeed) {
        els.telSpeed.textContent = `in ${chunk.usage.prompt_tokens || '?'} · out ${
          chunk.usage.completion_tokens || '?'
        }`;
      }
    }

    bubble.classList.remove('streaming');

    if (finishReason === 'length') {
      writeLog(
        `Stopped at max length (${maxTokens} tokens). Set reply length to Long, or type “continue”.`,
        'warn'
      );
    }

    const latencyMs = Math.round(performance.now() - genStart);
    const tps = latencyMs > 0 ? (tokenCount / (latencyMs / 1000)).toFixed(1) : '0';
    if (els.telLatency) els.telLatency.textContent = `${latencyMs} ms`;
    if (els.telTps) els.telTps.textContent = `${tps} tok/s`;
    const meta =
      finishReason === 'length'
        ? `Cut off at ${maxTokens} tokens — choose Long or type “continue” · ${tps} tok/s`
        : `${finishReason || 'stop'} · ≤${maxTokens} tok · ${latencyMs} ms · ${tps} tok/s`;
    bubble.textContent = reply || '(empty)';
    const m = document.createElement('span');
    m.className = 'msg-meta';
    m.textContent = meta;
    bubble.appendChild(m);

    chatMessages.push({ role: 'assistant', content: reply || '(empty)' });
    setStatus('Ready', 'ok');
    writeLog(`Done · reason=${finishReason || 'stop'} · ${tps} tok/s`, 'success');
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
  writeLog('Ending session — clearing chat + model cache…', 'warn');
  const model = activeModelId;
  chatMessages = [];
  try {
    if (engine) {
      try {
        await engine.interruptGenerate();
      } catch {
        /* ignore */
      }
      await engine.unload();
    }
  } catch {
    /* ignore */
  }
  engine = null;
  booted = false;
  generating = false;
  await purgeModelCaches(model ? [model, ...ALL_MODEL_IDS] : ALL_MODEL_IDS);
  activeModelId = null;

  els.promptInput.disabled = true;
  els.sendBtn.disabled = true;
  els.abortBtn.disabled = true;
  els.resetBtn.disabled = true;
  els.bootBtn.disabled = false;
  els.bootBtn.textContent = uiMode === 'simple' ? 'Start AI' : 'Load local LLM';
  if (els.modelBadge) els.modelBadge.textContent = '—';
  updateProgress(0, 'Ready when you are');
  setStatus('Idle', 'idle');
  setProvider('worker', 'Idle', false);
  els.chat.innerHTML = '';
  appendMsg(
    'assistant',
    'Session cleared. Chat history and model files were removed. Nothing was kept on disk.'
  );
}

function clearChat() {
  chatMessages = [];
  els.chat.innerHTML = '';
  appendMsg('assistant', 'Chat wiped from memory. Nothing was written to disk.');
  if (engine) {
    engine.resetChat?.().catch(() => {});
  }
}

els.modeSimple?.addEventListener('click', () => setUiMode('simple'));
els.modeAdvanced?.addEventListener('click', () => setUiMode('advanced'));
els.bootBtn.addEventListener('click', () => loadLLM());
els.resetBtn.addEventListener('click', () => unloadLLM());
els.clearCacheBtn?.addEventListener('click', () => unloadLLM());
els.promptForm.addEventListener('submit', sendPrompt);
els.abortBtn.addEventListener('click', () => abortGen());
els.clearLog?.addEventListener('click', () => {
  if (els.logBox) els.logBox.innerHTML = '';
  writeLog('Console cleared', 'info');
});
els.clearChat.addEventListener('click', () => clearChat());
els.lengthSelect?.addEventListener('change', () => syncAutoLabels());
els.configMode?.addEventListener('change', () => {
  syncAutoLabels();
  els.modelSelect.disabled = els.configMode.value === 'auto';
});
els.maxTokensOverride?.addEventListener('change', () => syncAutoLabels());

window.addEventListener('unhandledrejection', (e) => {
  e.preventDefault();
  writeLog(`Unhandled: ${e.reason?.message || e.reason}`, 'error');
  setStatus('Error', 'error');
});

window.addEventListener('pagehide', () => {
  purgeSessionArtifacts();
});
window.addEventListener('beforeunload', () => {
  purgeSessionArtifacts();
});

scrubLegacyStorage();
unregisterServiceWorkers();
els.modelSelect.disabled = (els.configMode?.value || 'auto') === 'auto';
setUiMode('simple');

requestAnimationFrame(() => {
  writeLog('Ephemeral mode: no chat history, no prefs, no lasting model cache.', 'warn');
  syncAutoLabels();
  probeWebGPU();
  purgeModelCaches().then(() => writeLog('Prior model caches cleared.', 'success'));
});
