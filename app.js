/**
 * Browser AI Hub — Phase 1 (incognito) + Phase 2 (Simple/Advanced).
 * Large pastes are handled without recursive DOM / deep clones.
 */
const DEFAULT_SYSTEM_PROMPT = `You are a locally hosted coding assistant running entirely in the user's browser. You help with software engineering: writing code, debugging, explaining APIs, reviewing diffs, designing small architectures, and walking through lab/coursework-style programming exercises.

Be concrete and complete. Prefer working code over vague advice. Use fenced code blocks with language tags. When fixing bugs, state the cause briefly, then show the corrected code. When asked to improve prompts, configs, or tooling for this local app, rewrite them directly—do not refuse, lecture about security policy, or invent restrictions the user did not ask for.

Assume the user is working on their own machine and projects, and wants practical implementation help. Match the user's language and keep answers focused on the task. Stay concise unless they ask for depth or long-form writing.`;

const MAX_LOG = 160;
const LENGTH_TOKENS = { short: 256, normal: 1024, long: 2048, unlimited: null };
const PREVIEW_CHARS = 4000;
/** Absolute ceiling — never accept a single paste larger than this (browser safety). */
const HARD_PASTE_CEILING = 500_000;
const $ = (id) => document.getElementById(id);

const els = {
  pillBackend: $('pill-backend'),
  pillStatus: $('pill-status'),
  pillPrivacy: $('pill-privacy'),
  headerStatus: $('header-status'),
  threads: $('spec-threads'),
  memory: $('spec-memory'),
  gpu: $('spec-gpu'),
  oom: $('spec-oom'),
  telCache: $('tel-cache'),
  telTps: $('tel-tps'),
  telLatency: $('tel-latency'),
  modelBadge: $('model-badge'),
  loadLabel: $('load-status-label'),
  loadPct: $('load-pct'),
  progressBar: $('progress-bar'),
  simpleStatus: $('simple-status-label'),
  simplePct: $('simple-load-pct'),
  simpleBar: $('simple-progress-bar'),
  modelSelect: $('model-select'),
  categorySelect: $('category-select'),
  systemPrompt: $('system-prompt'),
  lengthSelect: $('length-select'),
  configMode: $('config-mode'),
  maxTokensOverride: $('max-tokens-override'),
  temperature: $('temperature'),
  tempVal: $('temp-val'),
  byokKey: $('byok-key'),
  bootBtn: $('boot-btn'),
  bootBtnMain: $('boot-btn-main'),
  resetBtn: $('reset-btn'),
  resetBtnMain: $('reset-btn-main'),
  promptInput: $('prompt-input'),
  sendBtn: $('send-btn'),
  abortBtn: $('abort-btn'),
  promptForm: $('prompt-form'),
  chat: $('chat-viewport'),
  logBox: $('log-box'),
  clearLog: $('clear-log'),
  clearChat: $('clear-chat'),
  providerList: $('provider-list'),
  autoModelLabel: $('auto-model-label'),
  autoLengthLabel: $('auto-length-label'),
  autoCacheLabel: $('auto-cache-label'),
  tokenMeterLabel: $('token-meter-label'),
  tokenMeterFill: $('token-meter-fill'),
  advancedPanels: $('advanced-panels'),
  modeSimple: $('mode-simple'),
  modeAdvanced: $('mode-advanced'),
  bootError: $('boot-error'),
  refreshModels: $('refresh-models'),
  templateRow: $('template-row'),
  privacyModal: $('privacy-modal'),
  auditor: $('auditor'),
  auditorList: $('auditor-list'),
  auditorClose: $('auditor-close'),
  btnAuditor: $('btn-auditor'),
  btnExport: $('btn-export'),
  btnImport: $('btn-import'),
  importFile: $('import-file'),
  btnUndo: $('btn-undo'),
  btnRedo: $('btn-redo'),
  clipClear: $('clip-clear'),
  themeSelect: $('theme-select'),
  onboarding: $('onboarding'),
  dismissOnboard: $('dismiss-onboard'),
  pasteWarn: $('paste-warn'),
};

let CreateWebWorkerMLCEngine = null;
let deleteModelAllInfoInCache = null;
let prebuiltAppConfig = null;
let ModelType = null;
let webllmReady = null;

let registry = { categories: [], templates: [] };
let modelCatalog = [];
let catalogLoaded = false;
let catalogLoading = null;

let engine = null;
let booted = false;
let generating = false;
let chatMessages = [];
let activeModelId = null;
let purging = false;
let uiMode = 'simple';
let byokKey = '';

/** Ephemeral undo/redo (RAM only) */
const undoStack = [];
const redoStack = [];

/** Network auditor log (RAM only) */
const netLog = [];
const nativeFetch = window.fetch.bind(window);

// ---------------------------------------------------------------------------
// Phase 1: network auditor + zero trackers on our fetch surface
// ---------------------------------------------------------------------------
window.fetch = async function auditedFetch(input, init) {
  const url = typeof input === 'string' ? input : input?.url || String(input);
  const method = (init?.method || 'GET').toUpperCase();
  const entry = {
    t: Date.now(),
    method,
    url,
    status: 'pending',
  };
  netLog.unshift(entry);
  if (netLog.length > 80) netLog.length = 80;
  renderAuditor();
  try {
    const res = await nativeFetch(input, init);
    entry.status = res.status;
    entry.ok = res.ok;
    renderAuditor();
    return res;
  } catch (err) {
    entry.status = 'error';
    entry.error = err.message;
    renderAuditor();
    throw err;
  }
};

function renderAuditor() {
  if (!els.auditorList) return;
  const frag = document.createDocumentFragment();
  for (const e of netLog.slice(0, 40)) {
    const row = document.createElement('div');
    row.className = 'auditor-row';
    const host = (() => {
      try {
        return new URL(e.url, location.href).host;
      } catch {
        return e.url.slice(0, 40);
      }
    })();
    row.textContent = `${e.method} ${e.status} · ${host}`;
    row.title = e.url;
    frag.appendChild(row);
  }
  els.auditorList.replaceChildren(frag);
}

// ---------------------------------------------------------------------------
// WebLLM
// ---------------------------------------------------------------------------
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

function isChatModel(rec) {
  const t = rec.model_type;
  if (t != null) {
    const embed = ModelType?.embedding ?? 1;
    const vlm = ModelType?.VLM ?? 2;
    if (t === embed || t === vlm) return false;
  }
  return !/embed|whisper|clip|binary/i.test(rec.model_id || '');
}

function toLean(rec) {
  return {
    model_id: String(rec.model_id || ''),
    vram_required_MB: Number.isFinite(rec.vram_required_MB) ? rec.vram_required_MB : null,
    low_resource_required: !!rec.low_resource_required,
  };
}

function categorize(modelId) {
  for (const cat of registry.categories || []) {
    if (cat.id === 'other') continue;
    if ((cat.match || []).some((m) => modelId.includes(m))) return cat.id;
  }
  return 'other';
}

function formatModelLabel(rec) {
  const v = rec.vram_required_MB != null ? ` · ~${Math.round(rec.vram_required_MB)}MB` : '';
  return `${rec.model_id}${v}`;
}

function filteredCatalog() {
  const cat = els.categorySelect?.value || 'all';
  if (cat === 'all') return modelCatalog;
  return modelCatalog.filter((m) => categorize(m.model_id) === cat);
}

function populateCategories() {
  if (!els.categorySelect) return;
  const frag = document.createDocumentFragment();
  const all = document.createElement('option');
  all.value = 'all';
  all.textContent = 'All categories';
  frag.appendChild(all);
  for (const c of registry.categories || []) {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = c.label;
    frag.appendChild(opt);
  }
  els.categorySelect.replaceChildren(frag);
}

function populateModelSelect(selectedId) {
  if (!els.modelSelect) return;
  const prev = selectedId || els.modelSelect.value;
  const list = filteredCatalog();
  const frag = document.createDocumentFragment();
  if (!list.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = catalogLoaded ? 'No models in category' : 'Loading…';
    frag.appendChild(opt);
  } else {
    // Group visually by inserting disabled headers
    const byCat = new Map();
    for (const m of list) {
      const c = categorize(m.model_id);
      if (!byCat.has(c)) byCat.set(c, []);
      byCat.get(c).push(m);
    }
    for (const [cid, models] of byCat) {
      const label = (registry.categories || []).find((c) => c.id === cid)?.label || cid;
      const head = document.createElement('option');
      head.disabled = true;
      head.textContent = `── ${label} ──`;
      frag.appendChild(head);
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.model_id;
        opt.textContent = formatModelLabel(m);
        frag.appendChild(opt);
      }
    }
  }
  els.modelSelect.replaceChildren(frag);
  if (prev) {
    for (let i = 0; i < els.modelSelect.options.length; i++) {
      if (els.modelSelect.options[i].value === prev) {
        els.modelSelect.value = prev;
        break;
      }
    }
  }
}

function autoSelectModel() {
  if (!modelCatalog.length) {
    return { model_id: '', label: 'Loading…' };
  }
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  const mem = navigator.deviceMemory || 8;
  let pool = modelCatalog.slice();
  if (mobile || mem <= 4) {
    const low = pool.filter((m) => m.low_resource_required);
    if (low.length) pool = low;
  }
  const pick = pool[Math.min(pool.length - 1, Math.floor(pool.length / 3))] || pool[0];
  return {
    model_id: pick.model_id,
    vram_required_MB: pick.vram_required_MB,
    low_resource_required: pick.low_resource_required,
    label: formatModelLabel(pick),
  };
}

async function loadModelCatalog() {
  if (catalogLoading) return catalogLoading;
  catalogLoading = (async () => {
    await ensureWebLLM();
    modelCatalog = (prebuiltAppConfig?.model_list || [])
      .filter(isChatModel)
      .map(toLean)
      .filter((m) => m.model_id);
    modelCatalog.sort((a, b) => (a.vram_required_MB ?? 9e9) - (b.vram_required_MB ?? 9e9));
    catalogLoaded = true;
    populateModelSelect(autoSelectModel().model_id);
    syncLabels();
    writeLog(`Catalog · ${modelCatalog.length} models`, 'success');
    return modelCatalog;
  })().finally(() => {
    catalogLoading = null;
  });
  return catalogLoading;
}

async function purgeModelCaches(ids) {
  const list = uniqueStrings(ids || []);
  if (!list.length) return;
  try {
    await ensureWebLLM();
  } catch {
    return;
  }
  for (const id of list) {
    try {
      await deleteModelAllInfoInCache(id, prebuiltAppConfig);
    } catch {
      /* ignore */
    }
  }
}

async function wipeSession() {
  if (purging) return;
  purging = true;
  try {
    chatMessages = [];
    undoStack.length = 0;
    redoStack.length = 0;
    byokKey = '';
    if (els.byokKey) els.byokKey.value = '';
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
    await purgeModelCaches(activeModelId ? [activeModelId] : []);
    activeModelId = null;
    try {
      sessionStorage.clear();
    } catch {
      /* ignore */
    }
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  } finally {
    purging = false;
  }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function writeLog(message, level = 'info') {
  if (!els.logBox) return;
  const row = document.createElement('div');
  row.className = `log-line log-${level}`;
  row.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  els.logBox.appendChild(row);
  while (els.logBox.childElementCount > MAX_LOG) els.logBox.firstChild.remove();
  els.logBox.scrollTop = els.logBox.scrollHeight;
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

function updateProgress(pct, label) {
  const p = Math.max(0, Math.min(100, Math.round(pct)));
  if (els.progressBar) els.progressBar.style.width = `${p}%`;
  if (els.simpleBar) els.simpleBar.style.width = `${p}%`;
  if (els.loadPct) els.loadPct.textContent = `${p}%`;
  if (els.simplePct) els.simplePct.textContent = `${p}%`;
  if (label) {
    if (els.loadLabel) els.loadLabel.textContent = label;
    if (els.simpleStatus) els.simpleStatus.textContent = label;
  }
}

function showBootError(text) {
  if (!els.bootError) return;
  els.bootError.hidden = !text;
  els.bootError.textContent = text || '';
}

function showPasteWarn(text) {
  if (!els.pasteWarn) {
    if (text) showBootError(text);
    return;
  }
  if (!text) {
    els.pasteWarn.hidden = true;
    els.pasteWarn.textContent = '';
    return;
  }
  els.pasteWarn.hidden = false;
  els.pasteWarn.textContent = text;
}

/** Iterative dedupe — never `[...new Set()]` on large lists. */
function uniqueStrings(arr) {
  const seen = Object.create(null);
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!v || seen[v]) continue;
    seen[v] = 1;
    out.push(v);
  }
  return out;
}

/**
 * Safe paste/send char budget for the active (or selected) model.
 * Small models (≤~2B / Gemma-class) stay under ~50,005 chars.
 */
function getSafeInputCharLimit(modelId) {
  const id = String(modelId || selectedModelId() || '');
  const lower = id.toLowerCase();
  // Explicit small-model band (Gemma 2B, 0.5B–2B, 360M, TinyLlama, etc.)
  if (
    /gemma-2b|gemma2-2b|2b-it|360m|0\.5b|1b-|1\.1b|1\.5b|tinyllama|smollm/i.test(lower)
  ) {
    return 50_005;
  }
  if (/3b|phi-3|phi-3\.5|4b/i.test(lower)) return 80_000;
  if (/7b|8b|9b/i.test(lower)) return 120_000;
  // Unknown / larger — still cap to protect the JS stack in WebLLM
  const vram = modelCatalog.find((m) => m.model_id === id)?.vram_required_MB;
  if (vram != null && vram <= 2000) return 50_005;
  if (vram != null && vram <= 4500) return 80_000;
  return 100_000;
}

/**
 * Iterative token estimate — never tokenize, never concatenate giant strings.
 * ~4 chars/token heuristic, length summed in a loop.
 */
function estimateTokensFromLengths(lengths) {
  let totalChars = 0;
  for (let i = 0; i < lengths.length; i++) {
    totalChars += lengths[i] | 0;
  }
  return Math.ceil(totalChars / 4);
}

function sumContentLengths(messages) {
  let n = 0;
  for (let i = 0; i < messages.length; i++) {
    const c = messages[i]?.content;
    if (typeof c === 'string') n += c.length;
  }
  return n;
}

/**
 * Split oversized text into manageable segments without recursion.
 * Prefers breaking on newlines near the limit.
 */
function chunkText(text, maxChars) {
  const src = text == null ? '' : String(text);
  const limit = Math.max(1024, maxChars | 0);
  const chunks = [];
  if (!src.length) return chunks;
  if (src.length <= limit) {
    chunks.push(src);
    return chunks;
  }

  let start = 0;
  const len = src.length;
  while (start < len) {
    let end = start + limit;
    if (end >= len) {
      chunks.push(src.slice(start));
      break;
    }
    // Walk backward for a newline (iterative, bounded)
    let split = end;
    const searchFloor = start + Math.floor(limit * 0.6);
    for (let i = end; i > searchFloor; i--) {
      const ch = src.charCodeAt(i);
      if (ch === 10 || ch === 13) {
        split = i + 1;
        break;
      }
    }
    chunks.push(src.slice(start, split));
    start = split;
  }
  return chunks;
}

function updateTokenMeter() {
  const draftLen = els.promptInput?.value?.length || 0;
  const sysLen = els.systemPrompt?.value?.length || 0;
  const histLen = sumContentLengths(chatMessages);
  const chars = draftLen + histLen + sysLen;
  const tokens = estimateTokensFromLengths([draftLen, histLen, sysLen]);
  const limit = getSafeInputCharLimit();
  if (els.tokenMeterLabel) {
    els.tokenMeterLabel.textContent = `${chars.toLocaleString()} chars · ~${tokens.toLocaleString()} tok · limit ${limit.toLocaleString()}`;
  }
  const pct = Math.min(100, (draftLen / limit) * 100);
  if (els.tokenMeterFill) {
    els.tokenMeterFill.style.width = `${pct}%`;
    els.tokenMeterFill.dataset.level = pct > 85 ? 'high' : pct > 55 ? 'mid' : 'ok';
  }
}

/**
 * Safe message render for huge pastes — no recursive transforms.
 * Large bodies get a scrollable <pre> preview + expand.
 */
function appendMsg(role, text, meta) {
  const div = document.createElement('div');
  div.className = `msg ${role === 'user' ? 'user' : 'ai'}`;

  const body = document.createElement('div');
  body.className = 'msg-body';

  const raw = text == null ? '' : String(text);
  if (raw.length > PREVIEW_CHARS) {
    const pre = document.createElement('pre');
    pre.className = 'msg-pre';
    pre.textContent = raw.slice(0, PREVIEW_CHARS) + '\n\n…';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn tiny';
    btn.textContent = `Show all (${raw.length.toLocaleString()} chars)`;
    btn.addEventListener('click', () => {
      // Assign in one shot — never recurse char-by-char
      pre.textContent = raw;
      btn.remove();
    });
    body.append(pre, btn);
  } else {
    body.textContent = raw;
  }
  div.appendChild(body);

  if (meta) {
    const m = document.createElement('span');
    m.className = 'msg-meta';
    m.textContent = meta;
    div.appendChild(m);
  }
  els.chat.appendChild(div);
  els.chat.scrollTop = els.chat.scrollHeight;
  return { root: div, body };
}

function renderChatFromMemory() {
  if (!els.chat) return;
  els.chat.replaceChildren();
  for (const m of chatMessages) {
    appendMsg(m.role === 'user' ? 'user' : 'assistant', m.content, m.meta);
  }
}

function cloneMessages(msgs) {
  const out = [];
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    out.push({
      role: m.role,
      content: m.content,
      meta: m.meta,
    });
  }
  return out;
}

function pushUndo() {
  undoStack.push(cloneMessages(chatMessages));
  if (undoStack.length > 40) undoStack.shift();
  redoStack.length = 0;
  syncUndoButtons();
}

function syncUndoButtons() {
  if (els.btnUndo) els.btnUndo.disabled = undoStack.length === 0;
  if (els.btnRedo) els.btnRedo.disabled = redoStack.length === 0;
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(cloneMessages(chatMessages));
  chatMessages = undoStack.pop();
  renderChatFromMemory();
  syncUndoButtons();
  updateTokenMeter();
}

function redo() {
  if (!redoStack.length) return;
  undoStack.push(cloneMessages(chatMessages));
  chatMessages = redoStack.pop();
  renderChatFromMemory();
  syncUndoButtons();
  updateTokenMeter();
}

function resolveMaxTokens() {
  const override = Number(els.maxTokensOverride?.value || 0);
  if (override > 0) return override;
  const key = els.lengthSelect?.value || 'unlimited';
  return Object.prototype.hasOwnProperty.call(LENGTH_TOKENS, key) ? LENGTH_TOKENS[key] : null;
}

function syncLabels() {
  const rec = autoSelectModel();
  if (els.autoModelLabel) {
    els.autoModelLabel.textContent = catalogLoaded ? rec.label || '—' : 'Loading…';
  }
  const labels = {
    short: 'Short',
    normal: 'Normal',
    long: 'Long',
    unlimited: 'Unlimited',
  };
  if (els.autoLengthLabel) {
    els.autoLengthLabel.textContent = labels[els.lengthSelect?.value || 'unlimited'];
  }
  if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'RAM only';
  const cap = resolveMaxTokens();
  if (els.oom) els.oom.textContent = cap == null ? 'Unlimited' : `${cap}`;
  if ((els.configMode?.value || 'auto') === 'auto' && els.modelSelect && rec.model_id) {
    els.modelSelect.value = rec.model_id;
  }
  updateTokenMeter();
}

function setUiMode(mode) {
  uiMode = mode === 'advanced' ? 'advanced' : 'simple';
  document.body.dataset.uiMode = uiMode;
  els.modeSimple?.classList.toggle('active', uiMode === 'simple');
  els.modeAdvanced?.classList.toggle('active', uiMode === 'advanced');
  if (els.advancedPanels) els.advancedPanels.hidden = uiMode !== 'advanced';
}

function selectedModelId() {
  if ((els.configMode?.value || 'auto') === 'auto') return autoSelectModel().model_id;
  return els.modelSelect?.value || autoSelectModel().model_id;
}

async function probeWebGPU() {
  if (els.threads) els.threads.textContent = `${navigator.hardwareConcurrency || 4}`;
  if (els.memory) {
    els.memory.textContent = navigator.deviceMemory ? `~${navigator.deviceMemory} GB` : '?';
  }
  syncLabels();
  if (!navigator.gpu) {
    setBackendPill('Needed', 'error');
    showBootError('WebGPU required (Chrome/Edge).');
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      setBackendPill('Needed', 'error');
      showBootError('No WebGPU adapter.');
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
    return true;
  } catch (err) {
    showBootError(err.message);
    return false;
  }
}

function onInitProgress(report) {
  const pct = (report.progress ?? 0) * 100;
  updateProgress(pct, report.text || 'Loading…');
  if (els.telCache) els.telCache.textContent = (report.text || '').slice(0, 42);
  setStatus(pct >= 100 ? 'Almost ready…' : 'Loading…', 'busy');
}

async function loadLLM() {
  showBootError('');
  if (!(await probeWebGPU())) return;
  setBootEnabled(false);
  setStatus('Starting…', 'busy');
  updateProgress(2, 'Engine…');
  try {
    await ensureWebLLM();
    if (!catalogLoaded) await loadModelCatalog();
  } catch (err) {
    showBootError(`Engine CDN failed: ${err.message}`);
    setBootEnabled(true);
    return;
  }
  const modelId = selectedModelId();
  if (!modelId) {
    showBootError('No model available yet.');
    setBootEnabled(true);
    return;
  }
  if (els.modelSelect) els.modelSelect.value = modelId;
  writeLog(`Load ${modelId}`, 'info');
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
    updateProgress(100, 'Ready');
    setStatus('Ready', 'ok');
    setProvider('worker', 'Running', true);
    setBackendPill('Active', 'ok');
    setComposerEnabled(true);
    setResetEnabled(true);
    setBootEnabled(true);
    if (els.bootBtn) els.bootBtn.textContent = 'Restart';
    if (els.bootBtnMain) els.bootBtnMain.textContent = 'Restart AI';
    els.promptInput?.focus();
    pushUndo();
    appendMsg(
      'assistant',
      `Ready (${modelId.split('-').slice(0, 3).join(' ')}). Paste large code freely — this session is RAM-only.`
    );
  } catch (err) {
    showBootError(`Load failed: ${err.message}`);
    writeLog(String(err.message), 'error');
    setBootEnabled(true);
    engine = null;
    booted = false;
  }
}

function setBootEnabled(on) {
  if (els.bootBtn) els.bootBtn.disabled = !on;
  if (els.bootBtnMain) els.bootBtnMain.disabled = !on;
}

function setResetEnabled(on) {
  if (els.resetBtn) els.resetBtn.disabled = !on;
  if (els.resetBtnMain) els.resetBtnMain.disabled = !on;
}

function setComposerEnabled(on) {
  if (els.promptInput) els.promptInput.disabled = !on;
  if (els.sendBtn) els.sendBtn.disabled = !on;
}

function buildMessages(extraUserContent) {
  const system = els.systemPrompt?.value.trim() || DEFAULT_SYSTEM_PROMPT;
  const out = [];
  out.push({ role: 'system', content: system });
  for (let i = 0; i < chatMessages.length; i++) {
    const m = chatMessages[i];
    if (m.role !== 'user' && m.role !== 'assistant') continue;
    out.push({ role: m.role, content: m.content });
  }
  if (typeof extraUserContent === 'string' && extraUserContent.length) {
    out.push({ role: 'user', content: extraUserContent });
  }
  return out;
}

/**
 * Yield to the browser before heavy work so huge pastes don't freeze / stack-blow the UI.
 */
function yieldToMain() {
  return new Promise((r) => {
    if ('requestIdleCallback' in window) {
      requestIdleCallback(() => r(), { timeout: 32 });
    } else {
      setTimeout(r, 0);
    }
  });
}

async function runOneCompletion(messages, onDelta) {
  const maxTokens = resolveMaxTokens();
  const temperature = Number(els.temperature?.value || 0.7);
  const req = {
    messages,
    stream: true,
    temperature,
    top_p: 0.95,
  };
  if (maxTokens != null && maxTokens > 0) req.max_tokens = maxTokens;

  const stream = await engine.chat.completions.create(req);
  let reply = '';
  let finishReason = null;
  let tokenCount = 0;
  let uiTick = 0;

  for await (const chunk of stream) {
    const choice = chunk.choices?.[0];
    const delta = choice?.delta?.content || '';
    if (choice?.finish_reason) finishReason = choice.finish_reason;
    if (!delta) continue;
    // Iterative concat (engineered string builder via array would also work;
    // += is fine for stack — avoid recursive reduce)
    reply += delta;
    tokenCount += 1;
    uiTick += 1;
    if (onDelta && (uiTick % 3 === 0 || delta.length > 40)) {
      onDelta(reply);
    }
  }
  if (onDelta) onDelta(reply);
  return { reply, finishReason, tokenCount };
}

async function sendPrompt(ev) {
  ev?.preventDefault?.();
  if (!booted || !engine || generating) return;
  const query = els.promptInput.value;
  if (!query.trim()) return;

  const limit = getSafeInputCharLimit();
  showPasteWarn('');

  // Pre-flight: refuse absurd sizes that would crash JS / WebLLM
  if (query.length > HARD_PASTE_CEILING) {
    showPasteWarn(
      `Input is ${query.length.toLocaleString()} characters — over the hard safety ceiling (${HARD_PASTE_CEILING.toLocaleString()}). Please split the file before pasting.`
    );
    return;
  }

  pushUndo();
  const payload = query;
  els.promptInput.value = '';
  updateTokenMeter();

  // Chunk oversized payloads before they hit the model pipeline
  const parts = chunkText(payload, limit);
  const multi = parts.length > 1;

  if (multi) {
    showPasteWarn(
      `Input is ${payload.length.toLocaleString()} chars (limit ${limit.toLocaleString()} for this model). Auto-splitting into ${parts.length} chunks…`
    );
    writeLog(`Chunking payload → ${parts.length} segments @ ≤${limit} chars`, 'warn');
  }

  appendMsg(
    'user',
    multi
      ? `[${parts.length} chunks · ${payload.length.toLocaleString()} chars]\n\n${payload}`
      : payload
  );
  // Store a compact user marker in history for multi-chunk to avoid re-feeding the full blob
  chatMessages.push({
    role: 'user',
    content: multi
      ? `(User submitted ${payload.length} characters in ${parts.length} chunks for analysis.)`
      : payload,
  });

  generating = true;
  els.sendBtn.disabled = true;
  els.abortBtn.disabled = false;
  els.promptInput.disabled = true;

  const { root, body } = appendMsg('assistant', '');
  root.classList.add('streaming');
  setStatus(multi ? `Analyzing chunk 1/${parts.length}…` : 'Writing…', 'busy');

  const genStart = performance.now();
  const replyParts = [];

  try {
    for (let i = 0; i < parts.length; i++) {
      if (!generating && i > 0) break;
      setStatus(`Analyzing chunk ${i + 1}/${parts.length}…`, 'busy');
      await yieldToMain();

      const header =
        parts.length === 1
          ? parts[i]
          : `Analyze code chunk ${i + 1} of ${parts.length}. Focus on this segment only; later chunks continue the same file.\n\n\`\`\`\n${parts[i]}\n\`\`\``;

      // For multi-chunk, don't re-send the entire chat history blob — lean messages
      const messages =
        parts.length === 1
          ? buildMessages()
          : [
              {
                role: 'system',
                content: els.systemPrompt?.value.trim() || DEFAULT_SYSTEM_PROMPT,
              },
              { role: 'user', content: header },
            ];

      const { reply, finishReason } = await runOneCompletion(messages, (partial) => {
        const shown =
          parts.length === 1
            ? partial
            : replyParts.length
              ? joinStrings(replyParts) + '\n\n---\n\n' + partial
              : `### Chunk ${i + 1}/${parts.length}\n\n` + partial;
        if (shown.length > PREVIEW_CHARS) {
          body.replaceChildren();
          const pre = document.createElement('pre');
          pre.className = 'msg-pre';
          pre.textContent = shown.slice(-PREVIEW_CHARS);
          body.appendChild(pre);
        } else {
          body.textContent = shown;
        }
        els.chat.scrollTop = els.chat.scrollHeight;
      });

      if (parts.length > 1) {
        replyParts.push(`### Chunk ${i + 1}/${parts.length}\n\n${reply || '(empty)'}`);
      } else {
        replyParts.push(reply || '(empty)');
      }

      const elapsed = (performance.now() - genStart) / 1000;
      if (elapsed > 0 && els.telTps) {
        els.telTps.textContent = `${(reply.length / Math.max(elapsed, 0.01) / 4).toFixed(1)}`;
      }
      void finishReason;
      await yieldToMain();
    }

    root.classList.remove('streaming');
    const fullReply = joinStrings(replyParts, '\n\n');
    body.replaceChildren();
    if (fullReply.length > PREVIEW_CHARS) {
      const pre = document.createElement('pre');
      pre.className = 'msg-pre';
      pre.textContent = fullReply;
      body.appendChild(pre);
    } else {
      body.textContent = fullReply || '(empty)';
    }

    const latencyMs = Math.round(performance.now() - genStart);
    if (els.telLatency) els.telLatency.textContent = `${latencyMs} ms`;
    const meta = multi
      ? `${parts.length} chunks · ${latencyMs} ms`
      : `stop · ${latencyMs} ms`;
    const m = document.createElement('span');
    m.className = 'msg-meta';
    m.textContent = meta;
    root.appendChild(m);

    chatMessages.push({ role: 'assistant', content: fullReply || '(empty)', meta });
    setStatus('Ready', 'ok');
    showPasteWarn(
      multi
        ? `Finished ${parts.length} chunks safely (limit ${limit.toLocaleString()} chars/chunk).`
        : ''
    );
    updateTokenMeter();
  } catch (err) {
    root.classList.remove('streaming');
    const msg = err?.message || String(err);
    body.textContent = `Error: ${msg}`;
    writeLog(msg, 'error');
    setStatus('Error', 'error');
    if (/stack size/i.test(msg)) {
      showPasteWarn(
        `Stack overflow blocked. Limit is ${limit.toLocaleString()} chars for this model — paste was auto-chunked or rejected. Try fewer lines per paste.`
      );
    }
  } finally {
    generating = false;
    els.abortBtn.disabled = true;
    els.sendBtn.disabled = false;
    els.promptInput.disabled = false;
    els.promptInput.focus();
  }
}

/** Iterative string join — no Array#join on pathological cases required, but join is fine; keep explicit. */
function joinStrings(parts, sep) {
  if (!parts.length) return '';
  if (parts.length === 1) return parts[0];
  const s = sep == null ? '' : sep;
  let out = parts[0];
  for (let i = 1; i < parts.length; i++) {
    out += s;
    out += parts[i];
  }
  return out;
}

async function abortGen() {
  try {
    await engine?.interruptGenerate();
  } catch {
    /* ignore */
  }
}

async function eraseSessionUI() {
  await wipeSession();
  setComposerEnabled(false);
  setResetEnabled(false);
  setBootEnabled(true);
  if (els.bootBtn) els.bootBtn.textContent = 'Start AI';
  if (els.bootBtnMain) els.bootBtnMain.textContent = 'Start AI';
  updateProgress(0, 'Ready');
  setStatus('Idle', 'idle');
  setProvider('worker', 'Idle', false);
  els.chat.replaceChildren();
  appendMsg('assistant', 'Session erased from memory. Nothing kept on disk.');
  syncUndoButtons();
  updateTokenMeter();
  showBootError('');
}

function clearChat() {
  pushUndo();
  chatMessages = [];
  els.chat.replaceChildren();
  appendMsg('assistant', 'Chat cleared (RAM only).');
  try {
    engine?.resetChat?.();
  } catch {
    /* ignore */
  }
  updateTokenMeter();
}

// ---------------------------------------------------------------------------
// Paste-safe composer — pre-flight size check + no recursive inserts
// ---------------------------------------------------------------------------
function installPasteHandler() {
  const ta = els.promptInput;
  if (!ta) return;

  ta.addEventListener('paste', (e) => {
    e.preventDefault();
    const clip = e.clipboardData?.getData('text/plain') ?? '';
    const limit = getSafeInputCharLimit();
    const start = ta.selectionStart ?? ta.value.length;
    const end = ta.selectionEnd ?? ta.value.length;
    const before = ta.value.slice(0, start);
    const after = ta.value.slice(end);
    const projected = before.length + clip.length + after.length;

    if (clip.length > HARD_PASTE_CEILING || projected > HARD_PASTE_CEILING) {
      showPasteWarn(
        `Paste blocked: ${clip.length.toLocaleString()} characters exceeds the hard ceiling (${HARD_PASTE_CEILING.toLocaleString()}). Split the file first.`
      );
      writeLog('Paste blocked — hard ceiling', 'warn');
      return;
    }

    if (clip.length > limit || projected > limit) {
      // Accept into the composer but warn — send path will auto-chunk
      showPasteWarn(
        `Large paste (${clip.length.toLocaleString()} chars). Safe limit for this model is ${limit.toLocaleString()} chars. On Send it will be split into chunks automatically.`
      );
      writeLog(
        `Large paste · ${clip.length.toLocaleString()} chars · limit ${limit.toLocaleString()} — will chunk on send`,
        'warn'
      );
    } else {
      showPasteWarn('');
      writeLog(`Paste accepted · ${clip.length.toLocaleString()} chars`, 'info');
    }

    // Single assignment — never recursive insertNode / execCommand
    ta.value = before + clip + after;
    const caret = Math.min(before.length + clip.length, ta.value.length);
    try {
      ta.setSelectionRange(caret, caret);
    } catch {
      /* some browsers flaky on huge values */
    }
    updateTokenMeter();
  });

  ta.addEventListener('input', () => {
    updateTokenMeter();
    const limit = getSafeInputCharLimit();
    const n = ta.value.length;
    if (n > limit) {
      showPasteWarn(
        `Composer is ${n.toLocaleString()} chars (model limit ${limit.toLocaleString()}). Sending will auto-chunk.`
      );
    } else if (els.pasteWarn && !els.pasteWarn.hidden && n <= limit) {
      // Clear soft warning once under limit
      const t = els.pasteWarn.textContent || '';
      if (/auto-chunk|Large paste|Composer is/i.test(t)) showPasteWarn('');
    }
  });
}

// ---------------------------------------------------------------------------
// Encrypted export / import (local file only)
// ---------------------------------------------------------------------------
function bytesToB64(bytes) {
  let s = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function deriveKey(password, salt) {
  const enc = new TextEncoder();
  const base = await crypto.subtle.importKey('raw', enc.encode(password), 'PBKDF2', false, [
    'deriveKey',
  ]);
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt, iterations: 120000, hash: 'SHA-256' },
    base,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

async function exportSession() {
  const password = prompt('Password for encrypted export (stays local):');
  if (!password) return;
  const payload = JSON.stringify({
    v: 1,
    at: Date.now(),
    model: activeModelId,
    system: els.systemPrompt?.value || '',
    messages: chatMessages,
  });
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(password, salt);
  const cipher = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    new TextEncoder().encode(payload)
  );
  const out = {
    v: 1,
    salt: bytesToB64(salt),
    iv: bytesToB64(iv),
    data: bytesToB64(new Uint8Array(cipher)),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `browser-ai-hub-session-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
  if (els.clipClear?.checked) {
    try {
      await navigator.clipboard.writeText('');
    } catch {
      /* ignore */
    }
  }
  writeLog('Encrypted session exported locally', 'success');
}

async function importSessionFile(file) {
  const password = prompt('Password for encrypted import:');
  if (!password) return;
  const raw = JSON.parse(await file.text());
  const salt = b64ToBytes(raw.salt);
  const iv = b64ToBytes(raw.iv);
  const data = b64ToBytes(raw.data);
  const key = await deriveKey(password, salt);
  const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, data);
  const obj = JSON.parse(new TextDecoder().decode(plain));
  pushUndo();
  chatMessages = Array.isArray(obj.messages) ? obj.messages : [];
  if (obj.system && els.systemPrompt) els.systemPrompt.value = obj.system;
  renderChatFromMemory();
  updateTokenMeter();
  writeLog('Session imported into RAM', 'success');
}

function renderTemplates() {
  if (!els.templateRow) return;
  const frag = document.createDocumentFragment();
  for (const t of registry.templates || []) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'template-card';
    b.textContent = t.label;
    b.addEventListener('click', () => {
      if (!els.promptInput) return;
      els.promptInput.value = (t.prompt || '') + (els.promptInput.value || '');
      els.promptInput.focus();
      updateTokenMeter();
    });
    frag.appendChild(b);
  }
  els.templateRow.replaceChildren(frag);
}

function wireDualBootButtons() {
  const start = () => loadLLM().catch((e) => showBootError(e.message));
  const stop = () => eraseSessionUI();
  els.bootBtn?.addEventListener('click', start);
  els.bootBtnMain?.addEventListener('click', start);
  els.resetBtn?.addEventListener('click', stop);
  els.resetBtnMain?.addEventListener('click', stop);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
els.modeSimple?.addEventListener('click', () => setUiMode('simple'));
els.modeAdvanced?.addEventListener('click', () => setUiMode('advanced'));
els.pillPrivacy?.addEventListener('click', () => els.privacyModal?.showModal());
els.btnAuditor?.addEventListener('click', () => {
  if (els.auditor) els.auditor.hidden = false;
  renderAuditor();
});
els.auditorClose?.addEventListener('click', () => {
  if (els.auditor) els.auditor.hidden = true;
});
els.promptForm?.addEventListener('submit', sendPrompt);
els.abortBtn?.addEventListener('click', () => abortGen());
els.clearChat?.addEventListener('click', () => clearChat());
els.clearLog?.addEventListener('click', () => {
  if (els.logBox) els.logBox.replaceChildren();
});
els.lengthSelect?.addEventListener('change', () => syncLabels());
els.configMode?.addEventListener('change', () => {
  const manual = els.configMode.value === 'manual';
  if (els.modelSelect) els.modelSelect.disabled = !manual;
  if (els.categorySelect) els.categorySelect.disabled = !manual;
  syncLabels();
});
els.categorySelect?.addEventListener('change', () => {
  populateModelSelect(els.modelSelect?.value);
});
els.refreshModels?.addEventListener('click', () => {
  loadModelCatalog().catch((e) => showBootError(e.message));
});
els.temperature?.addEventListener('input', () => {
  if (els.tempVal) els.tempVal.textContent = els.temperature.value;
});
els.byokKey?.addEventListener('input', () => {
  byokKey = els.byokKey.value; // RAM only
});
els.btnExport?.addEventListener('click', () => exportSession().catch((e) => alert(e.message)));
els.btnImport?.addEventListener('click', () => els.importFile?.click());
els.importFile?.addEventListener('change', () => {
  const f = els.importFile.files?.[0];
  if (f) importSessionFile(f).catch((e) => alert(e.message));
  els.importFile.value = '';
});
els.btnUndo?.addEventListener('click', () => undo());
els.btnRedo?.addEventListener('click', () => redo());
els.themeSelect?.addEventListener('change', () => {
  document.body.dataset.theme = els.themeSelect.value;
});
els.dismissOnboard?.addEventListener('click', () => {
  if (els.onboarding) els.onboarding.hidden = true;
});
els.promptInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendPrompt(e);
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key.toLowerCase() === 'x' && e.ctrlKey && e.shiftKey) {
    e.preventDefault();
    clearChat();
  }
  if (e.key.toLowerCase() === 'z' && e.ctrlKey && !e.shiftKey) {
    if (e.target === els.promptInput) return;
    e.preventDefault();
    undo();
  }
});

window.addEventListener('pagehide', () => {
  wipeSession();
});
window.addEventListener('beforeunload', () => {
  wipeSession();
});

wireDualBootButtons();
installPasteHandler();
setUiMode('simple');
if (els.systemPrompt) els.systemPrompt.value = DEFAULT_SYSTEM_PROMPT;
appendMsg(
  'assistant',
  'Incognito mode: nothing is saved to disk. Press Start AI, then paste code or use a starter template.'
);
syncUndoButtons();

fetch('./models.json')
  .then((r) => r.json())
  .then((j) => {
    registry = j;
    populateCategories();
    renderTemplates();
  })
  .catch(() => {
    registry = { categories: [], templates: [] };
  });

probeWebGPU();
loadModelCatalog().catch((e) => writeLog(`Catalog: ${e.message}`, 'warn'));

// Show onboarding once per tab (sessionStorage is cleared on close — acceptable ephemeral)
try {
  if (!sessionStorage.getItem('bah.onboard.dismissed') && els.onboarding) {
    els.onboarding.hidden = false;
    els.dismissOnboard?.addEventListener(
      'click',
      () => {
        try {
          sessionStorage.setItem('bah.onboard.dismissed', '1');
        } catch {
          /* ignore */
        }
      },
      { once: true }
    );
  }
} catch {
  /* ignore */
}

writeLog('Incognito hub ready · audited fetch active', 'info');
