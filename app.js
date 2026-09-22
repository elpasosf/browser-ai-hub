/**
 * Browser AI Hub — WebLLM local chat with Simple (auto) + Advanced modes.
 */
import { CreateWebWorkerMLCEngine } from 'https://esm.run/@mlc-ai/web-llm';

const PREFS_KEY = 'bah.webllm.prefs.v2';
const CHAT_KEY = 'bah.webllm.chat.v1';
const MAX_LOG = 220;
const MAX_HISTORY = 12;

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

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}');
  } catch {
    return {};
  }
}

function savePrefs(patch) {
  localStorage.setItem(PREFS_KEY, JSON.stringify({ ...loadPrefs(), ...patch }));
}

function loadChat() {
  try {
    return JSON.parse(localStorage.getItem(CHAT_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveChat() {
  localStorage.setItem(CHAT_KEY, JSON.stringify(chatMessages.slice(-40)));
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

/** Auto-pick model from device memory / mobile */
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
  if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'This browser only';
  if (els.oom) els.oom.textContent = `${resolveMaxTokens()} max tokens`;

  // In Auto config, keep select aligned with recommendation
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
  savePrefs({ uiMode });
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
  const friendly = /cache|Cached|Fetch cache/i.test(text)
    ? 'Loading from your browser cache…'
    : /download|Fetch|Loading/i.test(text)
      ? text
      : text;
  updateProgress(pct, friendly);
  if (els.telCache) {
    els.telCache.textContent = text.length > 48 ? `${text.slice(0, 48)}…` : text;
  }
  setStatus(pct >= 100 ? 'Almost ready…' : /cache/i.test(text) ? 'Loading from cache' : 'Downloading model', 'busy');
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
  savePrefs({
    modelId,
    system: els.systemPrompt.value,
    length: els.lengthSelect.value,
    configMode: els.configMode?.value || 'auto',
    maxTokensOverride: els.maxTokensOverride?.value || '0',
  });

  els.bootBtn.disabled = true;
  els.modelSelect.disabled = true;
  els.systemPrompt.disabled = true;
  els.resetBtn.disabled = true;
  setStatus('Starting…', 'busy');
  updateProgress(2, 'Starting AI engine…');
  setProvider('worker', 'Starting', true);
  writeLog(`Loading: ${modelId}`, 'info');
  writeLog('First visit downloads weights into THIS browser. Later visits reuse Cache API (not GitHub).', 'warn');

  try {
    if (engine) {
      try {
        await engine.unload();
      } catch {
        /* ignore */
      }
      engine = null;
    }

    engine = await CreateWebWorkerMLCEngine(
      new Worker(new URL('./llm-worker.js', import.meta.url), { type: 'module' }),
      modelId,
      { initProgressCallback: onInitProgress }
    );

    booted = true;
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
    els.promptInput.placeholder = 'Ask anything…';
    els.promptInput.focus();

    try {
      const stats = await engine.runtimeStatsText();
      if (els.telBytes) els.telBytes.textContent = String(stats).slice(0, 42);
      writeLog(`Runtime: ${stats}`, 'success');
    } catch {
      if (els.telBytes) els.telBytes.textContent = 'WebGPU · browser cache';
    }

    appendMsg(
      'assistant',
      `Ready. Using ${modelId.split('-').slice(0, 3).join(' ')}. Reply length: ${els.lengthSelect.value} (up to ${resolveMaxTokens()} tokens). Ask for long essays with length set to Long if needed.`
    );
    writeLog('Local LLM ready.', 'success');
  } catch (err) {
    writeLog(`Load failed: ${err.message || err}`, 'error');
    writeLog('Tip: switch to Advanced → smaller model, or close other GPU apps.', 'warn');
    setStatus('Couldn’t start', 'error');
    setProvider('worker', 'Error', false);
    els.bootBtn.disabled = false;
    els.modelSelect.disabled = false;
    els.systemPrompt.disabled = false;
    els.bootBtn.textContent = 'Try again';
    engine = null;
    booted = false;
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
  saveChat();

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
    saveChat();
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
  writeLog('Freeing GPU memory…', 'warn');
  try {
    if (engine) await engine.unload();
  } catch {
    /* ignore */
  }
  engine = null;
  booted = false;
  generating = false;
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
  appendMsg(
    'assistant',
    'AI stopped and GPU memory freed. The model file usually stays in your browser cache for a fast restart.'
  );
}

function clearChat() {
  chatMessages = [];
  saveChat();
  els.chat.innerHTML = '';
  appendMsg('assistant', 'New chat started. Cached model weights are unchanged.');
}

// --- UI mode ---
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
els.lengthSelect?.addEventListener('change', () => {
  savePrefs({ length: els.lengthSelect.value });
  syncAutoLabels();
});
els.configMode?.addEventListener('change', () => {
  savePrefs({ configMode: els.configMode.value });
  syncAutoLabels();
  els.modelSelect.disabled = els.configMode.value === 'auto';
});
els.modelSelect?.addEventListener('change', () => savePrefs({ modelId: els.modelSelect.value }));
els.maxTokensOverride?.addEventListener('change', () => {
  savePrefs({ maxTokensOverride: els.maxTokensOverride.value });
  syncAutoLabels();
});

window.addEventListener('unhandledrejection', (e) => {
  e.preventDefault();
  writeLog(`Unhandled: ${e.reason?.message || e.reason}`, 'error');
  setStatus('Error', 'error');
});

// restore
const prefs = loadPrefs();
if (prefs.length) els.lengthSelect.value = prefs.length;
if (prefs.configMode) els.configMode.value = prefs.configMode;
if (prefs.maxTokensOverride) els.maxTokensOverride.value = prefs.maxTokensOverride;
if (prefs.modelId) els.modelSelect.value = prefs.modelId;
if (prefs.system) els.systemPrompt.value = prefs.system;
els.modelSelect.disabled = (els.configMode?.value || 'auto') === 'auto';
setUiMode(prefs.uiMode || 'simple');

const prior = loadChat();
if (prior.length) {
  chatMessages = prior;
  els.chat.innerHTML = '';
  for (const m of chatMessages) appendMsg(m.role, m.content);
}

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}

requestAnimationFrame(() => {
  writeLog('Browser AI Hub ready', 'info');
  syncAutoLabels();
  probeWebGPU();
});
