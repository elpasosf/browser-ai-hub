/**
 * Browser AI Hub — real local LLM via WebLLM (MLC) + WebGPU worker.
 */
import { CreateWebWorkerMLCEngine } from 'https://esm.run/@mlc-ai/web-llm';

const PREFS_KEY = 'bah.webllm.prefs.v1';
const CHAT_KEY = 'bah.webllm.chat.v1';
const MAX_LOG = 220;
const MAX_HISTORY = 16;

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
  modelSelect: $('model-select'),
  systemPrompt: $('system-prompt'),
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

/** @type {import('@mlc-ai/web-llm').MLCEngineInterface | null} */
let engine = null;
let booted = false;
let generating = false;
let chatMessages = []; // {role, content}
let tokenCount = 0;
let genStart = 0;

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
}

function setBackendPill(name, tone = 'ok') {
  els.pillBackend.textContent = `Backend: ${name}`;
  els.pillBackend.dataset.tone = tone;
}

function setProvider(id, text, active = false) {
  const li = els.providerList.querySelector(`[data-id="${id}"]`);
  if (!li) return;
  li.querySelector('.prov-state').textContent = text;
  li.classList.toggle('active', active);
}

function updateProgress(percent, label) {
  const p = Math.max(0, Math.min(100, Math.round(percent)));
  els.progressBar.style.width = `${p}%`;
  els.loadPct.textContent = `${p}%`;
  els.progressTrack.setAttribute('aria-valuenow', String(p));
  if (label) els.loadLabel.textContent = label;
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

function contextBudget() {
  const mem = navigator.deviceMemory || 8;
  if (mem <= 4) return { label: 'Strict · 512 ctx', maxTokens: 128 };
  if (mem <= 8) return { label: 'Balanced · 1k ctx', maxTokens: 256 };
  return { label: 'Full · 2k+ ctx', maxTokens: 512 };
}

async function probeWebGPU() {
  els.threads.textContent = `${navigator.hardwareConcurrency || 4} cores`;
  els.memory.textContent = navigator.deviceMemory ? `~${navigator.deviceMemory} GB` : 'Unknown';
  els.oom.textContent = contextBudget().label;

  if (!('gpu' in navigator)) {
    els.gpu.textContent = 'Unavailable';
    setProvider('webgpu', 'Not supported', false);
    setBackendPill('NONE', 'error');
    writeLog('WebGPU missing. Use Chrome/Edge 113+ on a GPU-capable device.', 'error');
    els.bootBtn.disabled = true;
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      els.gpu.textContent = 'No adapter';
      setProvider('webgpu', 'No adapter', false);
      setBackendPill('NONE', 'error');
      writeLog('WebGPU adapter not found. Update GPU drivers / try another browser.', 'error');
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
    els.gpu.textContent = detail;
    setProvider('webgpu', detail, true);
    setBackendPill('WebGPU', 'ok');
    writeLog(`WebGPU ready · ${detail}`, 'success');
    return true;
  } catch (err) {
    els.gpu.textContent = 'Error';
    setProvider('webgpu', err.message, false);
    writeLog(`WebGPU probe failed: ${err.message}`, 'error');
    els.bootBtn.disabled = true;
    return false;
  }
}

function onInitProgress(report) {
  const pct = (report.progress ?? 0) * 100;
  const text = report.text || 'Loading model…';
  updateProgress(pct, text);
  els.telCache.textContent = text.length > 48 ? `${text.slice(0, 48)}…` : text;
  setStatus(pct >= 100 ? 'Warming kernels' : 'Downloading / caching', 'busy');
  // Throttle log spam: only milestone-ish updates
  if (pct === 0 || pct >= 99 || Math.round(pct) % 10 === 0) {
    writeLog(text, pct >= 99 ? 'success' : 'info');
  }
}

async function loadLLM() {
  const ok = await probeWebGPU();
  if (!ok) return;

  const modelId = els.modelSelect.value;
  savePrefs({ modelId, system: els.systemPrompt.value });

  els.bootBtn.disabled = true;
  els.modelSelect.disabled = true;
  els.systemPrompt.disabled = true;
  els.resetBtn.disabled = true;
  setStatus('Loading local LLM', 'busy');
  updateProgress(1, 'Spawning WebLLM worker…');
  setProvider('worker', 'Starting', true);
  writeLog(`Loading WebLLM model: ${modelId}`, 'info');
  writeLog('First run downloads weights into Cache API (can take several minutes).', 'warn');

  try {
    // Unload previous engine if any
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
    els.modelBadge.textContent = modelId.replace(/-MLC.*$/, '');
    updateProgress(100, 'Local LLM ready');
    setStatus('Ready', 'ok');
    setProvider('worker', 'Running', true);
    setProvider('webgpu', els.gpu.textContent || 'Active', true);
    setBackendPill('WebGPU', 'ok');
    els.promptInput.disabled = false;
    els.sendBtn.disabled = false;
    els.resetBtn.disabled = false;
    els.clearCacheBtn.disabled = false;
    els.bootBtn.textContent = 'Reload model';
    els.bootBtn.disabled = false;
    els.modelSelect.disabled = false;
    els.systemPrompt.disabled = false;
    els.promptInput.placeholder = 'Chat with your local LLM…';
    els.promptInput.focus();

    try {
      const stats = await engine.runtimeStatsText();
      els.telBytes.textContent = String(stats).slice(0, 42);
      writeLog(`Runtime: ${stats}`, 'success');
    } catch {
      els.telBytes.textContent = 'WebGPU · cached';
    }

    const hello =
      `Local LLM loaded: ${modelId}. Inference runs on your GPU via WebGPU in a Web Worker. Weights stay in this browser's cache.`;
    appendMsg('assistant', hello);
    writeLog('Pipeline ready — ask anything.', 'success');
  } catch (err) {
    writeLog(`Load failed: ${err.message || err}`, 'error');
    writeLog(
      'Tip: pick a smaller model (Qwen 0.5B / SmolLM2 360M), free VRAM, use Chrome/Edge, and serve over http://localhost (not file://).',
      'warn'
    );
    setStatus('Load failed', 'error');
    setProvider('worker', 'Error', false);
    els.bootBtn.disabled = false;
    els.modelSelect.disabled = false;
    els.systemPrompt.disabled = false;
    els.bootBtn.textContent = 'Retry load';
    engine = null;
    booted = false;
  }
}

function buildMessages() {
  const system = els.systemPrompt.value.trim() || 'You are a helpful assistant.';
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
  setStatus('Generating', 'busy');
  els.telTps.textContent = '…';
  els.telLatency.textContent = '…';

  tokenCount = 0;
  genStart = performance.now();
  let reply = '';
  const budget = contextBudget();

  try {
    const chunks = await engine.chat.completions.create({
      messages: buildMessages(),
      stream: true,
      stream_options: { include_usage: true },
      max_tokens: budget.maxTokens,
      temperature: 0.7,
      top_p: 0.95,
    });

    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.delta?.content || '';
      if (delta) {
        reply += delta;
        tokenCount += 1;
        bubble.textContent = reply;
        els.chat.scrollTop = els.chat.scrollHeight;
        const elapsed = (performance.now() - genStart) / 1000;
        if (elapsed > 0) els.telTps.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
      }
      if (chunk.usage) {
        els.telSpeed.textContent = `prompt ${chunk.usage.prompt_tokens || '?'} · out ${
          chunk.usage.completion_tokens || '?'
        }`;
      }
    }

    bubble.classList.remove('streaming');
    const latencyMs = Math.round(performance.now() - genStart);
    const tps = latencyMs > 0 ? (tokenCount / (latencyMs / 1000)).toFixed(1) : '0';
    els.telLatency.textContent = `${latencyMs} ms`;
    els.telTps.textContent = `${tps} tok/s`;
    const meta = `WebLLM · ${els.modelSelect.value.split('-')[0]} · ${latencyMs} ms · ${tps} tok/s`;
    const m = document.createElement('span');
    m.className = 'msg-meta';
    m.textContent = meta;
    bubble.appendChild(m);

    chatMessages.push({ role: 'assistant', content: reply || '(empty)' });
    saveChat();
    setStatus('Ready', 'ok');
    writeLog(`Generation done · ${tokenCount} chunks · ${tps} tok/s · ${latencyMs} ms`, 'success');

    try {
      const stats = await engine.runtimeStatsText();
      els.telBytes.textContent = String(stats).slice(0, 42);
    } catch {
      /* ignore */
    }
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
    writeLog('Interrupt requested', 'warn');
  } catch (err) {
    writeLog(`Interrupt failed: ${err.message}`, 'warn');
  }
}

async function unloadLLM() {
  writeLog('Unloading engine from GPU memory…', 'warn');
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
  els.bootBtn.textContent = 'Load local LLM';
  els.modelBadge.textContent = '—';
  updateProgress(0, 'Standby');
  setStatus('Idle', 'idle');
  setProvider('worker', 'Idle', false);
  appendMsg('assistant', 'Model unloaded from GPU. Browser Cache API may still hold weights for fast reload.');
}

function clearChat() {
  chatMessages = [];
  saveChat();
  els.chat.innerHTML = '';
  appendMsg(
    'assistant',
    'Chat cleared (local only). Model weights in Cache API are unchanged.'
  );
}

// --- wire up ---
els.bootBtn.addEventListener('click', () => loadLLM());
els.resetBtn.addEventListener('click', () => unloadLLM());
els.clearCacheBtn.addEventListener('click', () => unloadLLM());
els.promptForm.addEventListener('submit', sendPrompt);
els.abortBtn.addEventListener('click', () => abortGen());
els.clearLog.addEventListener('click', () => {
  els.logBox.innerHTML = '';
  writeLog('Console cleared', 'info');
});
els.clearChat.addEventListener('click', () => clearChat());
els.modelSelect.addEventListener('change', () => savePrefs({ modelId: els.modelSelect.value }));

window.addEventListener('unhandledrejection', (e) => {
  e.preventDefault();
  writeLog(`Unhandled: ${e.reason?.message || e.reason}`, 'error');
  setStatus('Error trapped', 'error');
});

// restore prefs / chat
const prefs = loadPrefs();
if (prefs.modelId) els.modelSelect.value = prefs.modelId;
if (prefs.system) els.systemPrompt.value = prefs.system;
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
  writeLog('Browser AI Hub · WebLLM local LLM runtime', 'info');
  writeLog('Serve via http://localhost (npx serve .) — file:// will fail.', 'warn');
  probeWebGPU();
});
