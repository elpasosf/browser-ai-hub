/**
 * Browser AI Hub — Phase 1–2 + opt-in local model cache (IndexedDB via WebLLM).
 * Chat stays ephemeral; saved model weights persist until the user removes them.
 */
const DEFAULT_SYSTEM_PROMPT = `You are a locally hosted coding assistant running entirely in the user's browser. You help with software engineering: writing code, debugging, explaining APIs, reviewing diffs, designing small architectures, and walking through lab/coursework-style programming exercises.

Be concrete and complete. Prefer working code over vague advice. Use fenced code blocks with language tags. When fixing bugs, state the cause briefly, then show the corrected code. When asked to improve prompts, configs, or tooling for this local app, rewrite them directly—do not refuse, lecture about security policy, or invent restrictions the user did not ask for.

Assume the user is working on their own machine and projects, and wants practical implementation help. Match the user's language and keep answers focused on the task. Stay concise unless they ask for depth or long-form writing.`;

const MAX_LOG = 160;
const LENGTH_TOKENS = { short: 256, normal: 1024, long: 2048, unlimited: null };
const PREVIEW_CHARS = 4000;
/** Absolute ceiling — never accept a single paste larger than this (browser safety). */
const HARD_PASTE_CEILING = 500_000;
/** localStorage key for which model IDs the user opted to keep on disk. */
const KEEP_MODELS_KEY = 'bah.keptModels';
/** Default KV context before dynamic scale-up. */
const DEFAULT_CONTEXT_WINDOW = 4096;
/** Soft threshold — scale or slide when prompt uses this fraction of the window. */
const CONTEXT_SOFT_RATIO = 0.85;
/** Max fallback retries for ContextWindowSizeExceededError. */
const CONTEXT_OVERFLOW_RETRIES = 3;

/**
 * Mode-based presets: system style, output budget, and context strategy.
 * - code: prefer larger fixed context_window_size + chunked file handling
 * - writing: sliding_window_size for long-form history
 * - short: aggressive trim + concise outputs
 */
const MODE_PRESETS = {
  code: {
    id: 'code',
    label: 'Code',
    strategy: 'scale', // grow context_window_size toward model max
    defaultContext: 8192,
    maxOutputTokens: 4096,
    outputReserve: 1024,
    historyTurns: 24,
    temperature: 0.4,
    systemExtra:
      'Mode: CODE. Prioritize complete, runnable code and full-file reasoning. Prefer concrete diffs and fenced blocks. Use available context fully; if the user pasted a large file, work from the provided chunks carefully.',
  },
  writing: {
    id: 'writing',
    label: 'Writing',
    strategy: 'sliding', // context_window_size=-1, sliding_window_size>0
    defaultContext: 4096,
    slidingWindow: 4096,
    attentionSink: 4,
    maxOutputTokens: 2048,
    outputReserve: 768,
    historyTurns: 16,
    temperature: 0.8,
    systemExtra:
      'Mode: WRITING. Optimize for clear long-form prose. Maintain narrative coherence using recent context; older turns may be summarized or slid out of the window.',
  },
  short: {
    id: 'short',
    label: 'Short',
    strategy: 'tight', // keep small fixed window + hard history trim
    defaultContext: 2048,
    maxOutputTokens: 256,
    outputReserve: 256,
    historyTurns: 4,
    temperature: 0.5,
    systemExtra:
      'Mode: SHORT. Reply in a few sentences or a minimal code fix. Do not pad. Ignore tangential history.',
  },
};

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
  autoModeLabel: $('auto-mode-label'),
  autoContextLabel: $('auto-context-label'),
  chatMode: $('chat-mode'),
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
  saveModelBtn: $('save-model-btn'),
  removeModelBtn: $('remove-model-btn'),
  saveModelBtnAdv: $('save-model-btn-adv'),
  removeModelBtnAdv: $('remove-model-btn-adv'),
  modelCacheStatus: $('model-cache-status'),
  modelCacheStatusAdv: $('model-cache-status-adv'),
};

let CreateWebWorkerMLCEngine = null;
let deleteModelAllInfoInCache = null;
let hasModelInCache = null;
let prebuiltAppConfig = null;
let ModelType = null;
let webllmReady = null;
/** Shared AppConfig with IndexedDB cache enabled for large model weights. */
let appConfig = null;

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
/** True while Save Model Locally is downloading/caching. */
let savingModel = false;
/** Last known cache hit for the selected model. */
let selectedModelCached = false;
/** Prevent double auto-load on startup. */
let autoLoadAttempted = false;
/**
 * Active WebLLM chatOpts window configuration for the loaded engine.
 * Exactly one of context_window_size / sliding_window_size is positive (−1 disables).
 */
let engineContext = {
  context_window_size: DEFAULT_CONTEXT_WINDOW,
  sliding_window_size: -1,
  attention_sink_size: -1,
  mode: 'code',
};

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
        hasModelInCache = m.hasModelInCache;
        prebuiltAppConfig = m.prebuiltAppConfig;
        ModelType = m.ModelType;
        // Prefer IndexedDB — larger quota than Cache API for multi‑GB weights
        appConfig = Object.assign({}, prebuiltAppConfig, { useIndexedDBCache: true });
        return true;
      })
      .catch((err) => {
        webllmReady = null;
        throw err;
      });
  }
  return webllmReady;
}

function getAppConfig() {
  if (appConfig) return appConfig;
  if (prebuiltAppConfig) {
    appConfig = Object.assign({}, prebuiltAppConfig, { useIndexedDBCache: true });
  }
  return appConfig;
}

// ---------------------------------------------------------------------------
// Opt-in local model keep-list (tiny preference only — not chat data)
// ---------------------------------------------------------------------------
function getKeptModels() {
  try {
    const raw = localStorage.getItem(KEEP_MODELS_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    const out = [];
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (typeof v === 'string' && v && out.indexOf(v) === -1) out.push(v);
    }
    return out;
  } catch {
    return [];
  }
}

function setKeptModels(ids) {
  try {
    localStorage.setItem(KEEP_MODELS_KEY, JSON.stringify(uniqueStrings(ids || [])));
  } catch {
    /* quota / private mode */
  }
}

function isModelKept(modelId) {
  if (!modelId) return false;
  const kept = getKeptModels();
  for (let i = 0; i < kept.length; i++) {
    if (kept[i] === modelId) return true;
  }
  return false;
}

function markModelKept(modelId) {
  if (!modelId) return;
  const kept = getKeptModels();
  if (kept.indexOf(modelId) === -1) kept.push(modelId);
  setKeptModels(kept);
}

function unmarkModelKept(modelId) {
  if (!modelId) return;
  setKeptModels(getKeptModels().filter((id) => id !== modelId));
}

function setModelCacheStatus(text, tone) {
  const nodes = [els.modelCacheStatus, els.modelCacheStatusAdv];
  for (let i = 0; i < nodes.length; i++) {
    const el = nodes[i];
    if (!el) continue;
    el.textContent = text || '';
    if (tone) el.dataset.tone = tone;
    else delete el.dataset.tone;
  }
}

function setModelCacheButtons({ canSave, canRemove, busy }) {
  const saveBtns = [els.saveModelBtn, els.saveModelBtnAdv];
  const removeBtns = [els.removeModelBtn, els.removeModelBtnAdv];
  for (let i = 0; i < saveBtns.length; i++) {
    if (saveBtns[i]) saveBtns[i].disabled = !!busy || !canSave;
  }
  for (let i = 0; i < removeBtns.length; i++) {
    if (removeBtns[i]) removeBtns[i].disabled = !!busy || !canRemove;
  }
}

async function refreshModelCacheStatus(modelId) {
  const id = modelId || selectedModelId();
  if (!id) {
    selectedModelCached = false;
    setModelCacheStatus('Select a model to check local cache.', 'idle');
    setModelCacheButtons({ canSave: false, canRemove: false, busy: savingModel });
    if (els.autoCacheLabel) els.autoCacheLabel.textContent = '—';
    return false;
  }
  let cached = false;
  try {
    await ensureWebLLM();
    if (typeof hasModelInCache === 'function') {
      cached = !!(await hasModelInCache(id, getAppConfig()));
    }
  } catch {
    cached = false;
  }
  selectedModelCached = cached;
  const kept = isModelKept(id);
  if (cached && kept) {
    setModelCacheStatus(`Saved locally: ${id.split('-').slice(0, 3).join(' ')} — fast start available.`, 'ok');
    if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'Local IndexedDB';
  } else if (cached) {
    setModelCacheStatus(`Weights found in browser storage for this model. Click Save to keep them across visits.`, 'ok');
    if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'Cached (session)';
  } else {
    setModelCacheStatus(`Not saved locally. Start AI or Save Model Locally to download from the CDN.`, 'idle');
    if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'Remote / RAM';
  }
  setModelCacheButtons({
    canSave: !savingModel && !generating,
    canRemove: cached || kept,
    busy: savingModel || generating,
  });
  return cached;
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
  const cfg = getAppConfig();
  for (const id of list) {
    try {
      await deleteModelAllInfoInCache(id, cfg);
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
    engineContext = {
      context_window_size: DEFAULT_CONTEXT_WINDOW,
      sliding_window_size: -1,
      attention_sink_size: -1,
      mode: selectedChatMode(),
    };
    // Preserve opt-in saved models; purge only ephemeral (non-kept) weights
    const kept = getKeptModels();
    if (activeModelId && kept.indexOf(activeModelId) === -1) {
      await purgeModelCaches([activeModelId]);
    }
    activeModelId = null;
    try {
      sessionStorage.clear();
    } catch {
      /* ignore */
    }
    try {
      // Clear ephemeral keys but restore the keep-list preference
      localStorage.clear();
      setKeptModels(kept);
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

function estimateMessagesTokens(messages) {
  // ~4 chars/token + small per-message role overhead
  let chars = 0;
  let count = 0;
  for (let i = 0; i < messages.length; i++) {
    const c = messages[i]?.content;
    if (typeof c === 'string') chars += c.length;
    count += 1;
  }
  return Math.ceil(chars / 4) + count * 4;
}

function selectedChatMode() {
  const v = els.chatMode?.value || 'code';
  return MODE_PRESETS[v] ? v : 'code';
}

function getModePreset(modeId) {
  return MODE_PRESETS[modeId || selectedChatMode()] || MODE_PRESETS.code;
}

/**
 * Model-advertised max context (from WebLLM record overrides) with safe VRAM caps.
 */
function getModelMaxContext(modelId) {
  const id = String(modelId || activeModelId || selectedModelId() || '');
  const full = (prebuiltAppConfig?.model_list || []).find((m) => m.model_id === id);
  const o = full?.overrides || {};
  if (Number.isFinite(o.context_window_size) && o.context_window_size > 0) {
    return o.context_window_size;
  }
  if (Number.isFinite(o.sliding_window_size) && o.sliding_window_size > 0) {
    return o.sliding_window_size;
  }
  const lower = id.toLowerCase();
  if (/1m|1000k/.test(lower)) return 32768;
  if (/128k/.test(lower)) return 16384;
  if (/64k|32k/.test(lower)) return 16384;
  if (/16k|8192|8k/.test(lower)) return 8192;
  if (/gemma-2b|0\.5b|360m|1b-|tinyllama|smollm/i.test(lower)) return 8192;
  if (/3b|phi-3|7b|8b/.test(lower)) return 8192;
  return 8192;
}

function effectiveContextBudget() {
  if (engineContext.sliding_window_size > 0) return engineContext.sliding_window_size;
  if (engineContext.context_window_size > 0) return engineContext.context_window_size;
  return DEFAULT_CONTEXT_WINDOW;
}

function buildChatOptsForMode(modeId, contextSize) {
  const preset = getModePreset(modeId);
  const modelMax = getModelMaxContext();
  const size = Math.min(
    modelMax,
    Math.max(1024, contextSize || preset.defaultContext || DEFAULT_CONTEXT_WINDOW)
  );

  if (preset.strategy === 'sliding') {
    const slide = Math.min(modelMax, Math.max(2048, preset.slidingWindow || size));
    return {
      context_window_size: -1,
      sliding_window_size: slide,
      attention_sink_size: preset.attentionSink >= 0 ? preset.attentionSink : 4,
    };
  }

  // scale + tight: fixed KV window (sliding disabled)
  const ctx =
    preset.strategy === 'tight'
      ? Math.min(size, preset.defaultContext || 2048)
      : size;
  return {
    context_window_size: ctx,
    sliding_window_size: -1,
    attention_sink_size: -1,
  };
}

function chatOptsEqual(a, b) {
  if (!a || !b) return false;
  return (
    a.context_window_size === b.context_window_size &&
    a.sliding_window_size === b.sliding_window_size &&
    (a.attention_sink_size || -1) === (b.attention_sink_size || -1)
  );
}

function applyEngineContextState(chatOpts, modeId) {
  engineContext = {
    context_window_size: chatOpts.context_window_size,
    sliding_window_size: chatOpts.sliding_window_size,
    attention_sink_size: chatOpts.attention_sink_size ?? -1,
    mode: modeId || selectedChatMode(),
  };
  if (els.autoContextLabel) {
    const label =
      engineContext.sliding_window_size > 0
        ? `slide ${engineContext.sliding_window_size}`
        : `ctx ${engineContext.context_window_size}`;
    els.autoContextLabel.textContent = label;
  }
}

/**
 * Reload the live engine with new context / sliding-window chatOpts (same weights).
 */
async function reloadEngineContext(chatOpts, modeId) {
  if (!engine || !activeModelId) {
    applyEngineContextState(chatOpts, modeId);
    return;
  }
  if (chatOptsEqual(chatOpts, engineContext)) {
    applyEngineContextState(chatOpts, modeId);
    return;
  }
  writeLog(
    `Resizing context → ctx=${chatOpts.context_window_size} slide=${chatOpts.sliding_window_size}`,
    'warn'
  );
  setStatus('Adjusting context window…', 'busy');
  await engine.reload(activeModelId, chatOpts);
  applyEngineContextState(chatOpts, modeId);
  setStatus('Ready', 'ok');
}

/**
 * Before a prompt: estimate tokens and scale context_window_size or enable sliding window.
 */
async function ensureDynamicContext(promptTokens, modeId) {
  const preset = getModePreset(modeId);
  const modelMax = getModelMaxContext();
  const reserve = preset.outputReserve || 512;
  const needed = promptTokens + reserve;
  const current = effectiveContextBudget();

  if (preset.strategy === 'sliding') {
    let slide = Math.max(preset.slidingWindow || DEFAULT_CONTEXT_WINDOW, current);
    if (needed > slide * CONTEXT_SOFT_RATIO) {
      slide = Math.min(modelMax, Math.max(needed, slide * 2));
    }
    slide = Math.min(modelMax, Math.max(2048, slide));
    await reloadEngineContext(buildChatOptsForMode('writing', slide), 'writing');
    return effectiveContextBudget();
  }

  if (preset.strategy === 'tight') {
    const tight = Math.min(modelMax, preset.defaultContext || 2048);
    await reloadEngineContext(buildChatOptsForMode('short', tight), 'short');
    return effectiveContextBudget();
  }

  // code / scale
  let target = current > 0 ? current : DEFAULT_CONTEXT_WINDOW;
  if (needed > target * CONTEXT_SOFT_RATIO) {
    // Step up: 4096 → 8192 → 16384 → modelMax
    const steps = [4096, 8192, 16384, modelMax];
    target = modelMax;
    for (let i = 0; i < steps.length; i++) {
      if (steps[i] >= needed) {
        target = Math.min(modelMax, steps[i]);
        break;
      }
    }
    target = Math.min(modelMax, Math.max(needed, target));
    writeLog(
      `Prompt ~${promptTokens} tok → scaling context_window_size to ${target} (max ${modelMax})`,
      'warn'
    );
  } else {
    target = Math.min(modelMax, Math.max(target, preset.defaultContext || DEFAULT_CONTEXT_WINDOW));
  }
  await reloadEngineContext(buildChatOptsForMode('code', target), 'code');
  return effectiveContextBudget();
}

/**
 * Fit messages under a token budget: keep system + newest turns; compress oversized bodies.
 */
function fitMessagesToBudget(messages, budgetTokens, modeId) {
  const preset = getModePreset(modeId);
  const maxTurns = preset.historyTurns || 12;
  const out = [];
  for (let i = 0; i < messages.length; i++) {
    out.push({
      role: messages[i].role,
      content: typeof messages[i].content === 'string' ? messages[i].content : '',
    });
  }

  // Drop oldest user/assistant pairs beyond historyTurns
  const system = out.length && out[0].role === 'system' ? out[0] : null;
  let rest = system ? out.slice(1) : out.slice();
  while (rest.length > maxTurns * 2) {
    rest.shift();
  }

  const rebuild = () => {
    const m = [];
    if (system) m.push(system);
    for (let i = 0; i < rest.length; i++) m.push(rest[i]);
    return m;
  };

  let fitted = rebuild();
  let tokens = estimateMessagesTokens(fitted);

  // Drop oldest history until under budget
  while (tokens > budgetTokens && rest.length > 1) {
    rest.shift();
    // Keep message order valid: prefer starting on user after system
    while (rest.length && rest[0].role === 'assistant') rest.shift();
    fitted = rebuild();
    tokens = estimateMessagesTokens(fitted);
  }

  // Compress largest remaining message (usually the latest paste) if still over
  if (tokens > budgetTokens) {
    const room = Math.max(256, budgetTokens - (system ? estimateMessagesTokens([system]) : 0) - 64);
    for (let i = fitted.length - 1; i >= 0; i--) {
      if (fitted[i].role === 'system') continue;
      const text = fitted[i].content;
      const approx = Math.ceil(text.length / 4);
      if (approx <= room) continue;
      const keepChars = Math.max(400, room * 4);
      const head = Math.floor(keepChars * 0.55);
      const tail = keepChars - head;
      fitted[i] = {
        role: fitted[i].role,
        content:
          text.slice(0, head) +
          `\n\n…[compressed for context window; ${text.length.toLocaleString()} chars total]…\n\n` +
          text.slice(-tail),
      };
      tokens = estimateMessagesTokens(fitted);
      if (tokens <= budgetTokens) break;
    }
  }

  return fitted;
}

/**
 * Summarize-style compression of older history into one system note (iterative, no recursion).
 */
function compressOldestHistory(messages) {
  if (!messages.length) return messages;
  const out = [];
  let i = 0;
  if (messages[0]?.role === 'system') {
    out.push({ role: 'system', content: messages[0].content });
    i = 1;
  }
  if (messages.length - i <= 2) {
    for (; i < messages.length; i++) out.push({ role: messages[i].role, content: messages[i].content });
    return out;
  }
  // Fold the oldest two turns into a brief note, keep the rest
  const dropped = [];
  const take = Math.min(2, messages.length - i - 1);
  for (let k = 0; k < take; k++) {
    dropped.push(messages[i + k]);
  }
  i += take;
  let summary = 'Earlier context (compressed): ';
  for (let k = 0; k < dropped.length; k++) {
    const t = dropped[k].content || '';
    summary += `[${dropped[k].role}] ${t.slice(0, 180)}${t.length > 180 ? '…' : ''} `;
  }
  out.push({ role: 'system', content: summary.slice(0, 1200) });
  for (; i < messages.length; i++) {
    out.push({ role: messages[i].role, content: messages[i].content });
  }
  return out;
}

function isContextWindowError(err) {
  const name = err?.name || '';
  const msg = err?.message || String(err || '');
  return (
    name === 'ContextWindowSizeExceededError' ||
    /ContextWindowSizeExceeded/i.test(msg) ||
    /Prompt tokens exceed context window size/i.test(msg) ||
    /exceed context window/i.test(msg)
  );
}

/**
 * Fallback when WebLLM throws ContextWindowSizeExceededError:
 * compress / slice oldest history, optionally bump window, then retry.
 */
async function fallbackAfterContextOverflow(messages, modeId, attempt) {
  writeLog(`Context overflow fallback · attempt ${attempt + 1}`, 'warn');
  let next = compressOldestHistory(messages);
  next = fitMessagesToBudget(
    next,
    Math.max(512, Math.floor(effectiveContextBudget() * 0.55)),
    modeId
  );

  const preset = getModePreset(modeId);
  const modelMax = getModelMaxContext();
  if (attempt === 0 && preset.strategy !== 'tight') {
    // First fallback: scale window or switch to sliding
    if (preset.strategy === 'sliding' || attempt >= 1) {
      await reloadEngineContext(
        {
          context_window_size: -1,
          sliding_window_size: Math.min(modelMax, Math.max(4096, effectiveContextBudget())),
          attention_sink_size: 4,
        },
        modeId
      );
    } else {
      const bumped = Math.min(modelMax, Math.max(effectiveContextBudget() * 2, 8192));
      await reloadEngineContext(buildChatOptsForMode('code', bumped), modeId);
    }
  } else if (attempt >= 1) {
    // Hard switch to sliding window as last resort
    await reloadEngineContext(
      {
        context_window_size: -1,
        sliding_window_size: Math.min(modelMax, 4096),
        attention_sink_size: 4,
      },
      modeId
    );
    next = fitMessagesToBudget(next, Math.floor(effectiveContextBudget() * 0.5), 'short');
  }

  return next;
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
  if (key !== 'unlimited' && Object.prototype.hasOwnProperty.call(LENGTH_TOKENS, key)) {
    return LENGTH_TOKENS[key];
  }
  // Mode default when reply length is "unlimited"
  return getModePreset().maxOutputTokens;
}

function resolveTemperature() {
  const preset = getModePreset();
  const raw = els.temperature?.value;
  // If user hasn't touched advanced temp, prefer mode default when in simple mode
  if (uiMode === 'simple' && preset.temperature != null) return preset.temperature;
  return Number(raw || preset.temperature || 0.7);
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
    unlimited: 'Mode default',
  };
  if (els.autoLengthLabel) {
    els.autoLengthLabel.textContent = labels[els.lengthSelect?.value || 'unlimited'];
  }
  const preset = getModePreset();
  if (els.autoModeLabel) els.autoModeLabel.textContent = preset.label;
  if (els.autoContextLabel && !booted) {
    els.autoContextLabel.textContent =
      preset.strategy === 'sliding'
        ? `slide ${preset.slidingWindow || 4096}`
        : `ctx ${preset.defaultContext || DEFAULT_CONTEXT_WINDOW}`;
  }
  const cap = resolveMaxTokens();
  if (els.oom) els.oom.textContent = cap == null ? 'Unlimited' : `${cap}`;
  if ((els.configMode?.value || 'auto') === 'auto' && els.modelSelect && rec.model_id) {
    els.modelSelect.value = rec.model_id;
  }
  updateTokenMeter();
  refreshModelCacheStatus(rec.model_id).catch(() => {});
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

async function loadLLM(opts) {
  const options = opts || {};
  showBootError('');
  if (!(await probeWebGPU())) return;
  setBootEnabled(false);
  setModelCacheButtons({ canSave: false, canRemove: selectedModelCached, busy: true });
  setStatus(options.fromCache ? 'Loading from local cache…' : 'Starting…', 'busy');
  updateProgress(2, options.fromCache ? 'Local cache…' : 'Engine…');
  try {
    await ensureWebLLM();
    if (!catalogLoaded) await loadModelCatalog();
  } catch (err) {
    showBootError(`Engine CDN failed: ${err.message}`);
    setBootEnabled(true);
    refreshModelCacheStatus().catch(() => {});
    return;
  }
  const modelId = options.modelId || selectedModelId();
  if (!modelId) {
    showBootError('No model available yet.');
    setBootEnabled(true);
    refreshModelCacheStatus().catch(() => {});
    return;
  }
  if (els.modelSelect) els.modelSelect.value = modelId;
  const wasCached = await refreshModelCacheStatus(modelId);
  writeLog(
    wasCached || options.fromCache
      ? `Load ${modelId} (prefer local cache)`
      : `Load ${modelId} (remote download)`,
    'info'
  );
  try {
    if (engine) {
      try {
        await engine.unload();
      } catch {
        /* ignore */
      }
      engine = null;
    }
    // Do NOT purge before load — that defeated local caching.
    // Ephemeral models are cleared on tab close via wipeSession if not kept.
    const modeId = selectedChatMode();
    const chatOpts = buildChatOptsForMode(modeId);
    engine = await CreateWebWorkerMLCEngine(
      new Worker(new URL('./llm-worker.js', import.meta.url), { type: 'module' }),
      modelId,
      {
        initProgressCallback: onInitProgress,
        appConfig: getAppConfig(),
      },
      chatOpts
    );
    applyEngineContextState(chatOpts, modeId);
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
    const kept = isModelKept(modelId);
    const src = wasCached || options.fromCache ? 'local cache' : 'CDN (now cached in browser)';
    appendMsg(
      'assistant',
      `Ready (${modelId.split('-').slice(0, 3).join(' ')}) via ${src}.${
        kept ? ' This model is saved for fast startups.' : ' Chat is RAM-only; use Save Model Locally to keep weights.'
      }`
    );
    await refreshModelCacheStatus(modelId);
  } catch (err) {
    showBootError(`Load failed: ${err.message}`);
    writeLog(String(err.message), 'error');
    setBootEnabled(true);
    engine = null;
    booted = false;
    await refreshModelCacheStatus(modelId);
  }
}

/**
 * Download / retain the selected model in IndexedDB for faster future startups.
 */
async function saveModelLocally() {
  if (savingModel || generating) return;
  const modelId = selectedModelId();
  if (!modelId) {
    setModelCacheStatus('No model selected.', 'error');
    return;
  }
  savingModel = true;
  showBootError('');
  setModelCacheButtons({ canSave: false, canRemove: false, busy: true });
  setModelCacheStatus(`Saving ${modelId.split('-').slice(0, 3).join(' ')} to local IndexedDB…`, 'busy');
  setStatus('Saving model locally…', 'busy');
  writeLog(`Save Model Locally · ${modelId}`, 'info');

  try {
    await ensureWebLLM();
    // Ensure weights are present: load (or reload) so WebLLM writes to IndexedDB
    if (!booted || activeModelId !== modelId) {
      await loadLLM({ modelId, fromCache: false });
      if (!booted) throw new Error('Model load did not complete — nothing saved.');
    } else {
      // Already running — mark kept; weights should already be in cache from load
      updateProgress(60, 'Verifying cache…');
    }

    markModelKept(modelId);
    // Give the cache backend a tick, then verify
    await yieldToMain();
    let cached = false;
    if (typeof hasModelInCache === 'function') {
      cached = !!(await hasModelInCache(modelId, getAppConfig()));
    }
    if (!cached) {
      // Rare: engine loaded from memory path without durable write — force reload once
      updateProgress(40, 'Writing to IndexedDB…');
      if (engine) {
        try {
          await engine.reload(modelId);
        } catch {
          /* reload optional */
        }
      }
      cached = typeof hasModelInCache === 'function'
        ? !!(await hasModelInCache(modelId, getAppConfig()))
        : true;
    }

    selectedModelCached = !!cached;
    updateProgress(100, cached ? 'Saved locally' : 'Marked for keep');
    setStatus(cached ? 'Model saved locally' : 'Ready', 'ok');
    setModelCacheStatus(
      cached
        ? `Success — ${modelId.split('-').slice(0, 3).join(' ')} is saved in IndexedDB for fast startups.`
        : `Keep-list updated. Re-open after Start AI finishes caching if status still shows remote.`,
      cached ? 'ok' : 'busy'
    );
    writeLog(cached ? `Cached OK · ${modelId}` : `Keep marked · ${modelId}`, 'info');
    if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'Local IndexedDB';
  } catch (err) {
    unmarkModelKept(modelId);
    setModelCacheStatus(`Save failed: ${err.message}`, 'error');
    showBootError(`Save failed: ${err.message}`);
    writeLog(String(err.message), 'error');
    setStatus('Error', 'error');
  } finally {
    savingModel = false;
    await refreshModelCacheStatus(modelId);
  }
}

/**
 * Clear cached weights for the selected (or active) model from IndexedDB / Cache.
 */
async function removeSavedModel() {
  if (savingModel || generating) return;
  const modelId = selectedModelId() || activeModelId;
  if (!modelId) {
    setModelCacheStatus('No model selected.', 'error');
    return;
  }
  const ok = window.confirm(
    `Remove saved model weights for:\n\n${modelId}\n\nThis frees browser storage. Next Start AI will re-download from the CDN.`
  );
  if (!ok) return;

  savingModel = true;
  setModelCacheButtons({ canSave: false, canRemove: false, busy: true });
  setModelCacheStatus('Removing saved model from browser storage…', 'busy');
  setStatus('Removing local model…', 'busy');
  writeLog(`Remove Saved Model · ${modelId}`, 'info');

  try {
    // If this model is currently loaded, unload first so files aren't locked
    if (engine && activeModelId === modelId) {
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
      booted = false;
      activeModelId = null;
      setComposerEnabled(false);
      setResetEnabled(false);
      if (els.bootBtn) els.bootBtn.textContent = 'Start AI';
      if (els.bootBtnMain) els.bootBtnMain.textContent = 'Start AI';
    }

    await purgeModelCaches([modelId]);
    unmarkModelKept(modelId);
    selectedModelCached = false;
    updateProgress(0, 'Removed');
    setStatus('Local model removed', 'ok');
    setModelCacheStatus('Saved model cleared from browser storage.', 'ok');
    if (els.autoCacheLabel) els.autoCacheLabel.textContent = 'Remote / RAM';
    writeLog(`Purged cache · ${modelId}`, 'info');
    appendMsg('assistant', `Removed local cache for ${modelId.split('-').slice(0, 3).join(' ')}.`);
  } catch (err) {
    setModelCacheStatus(`Remove failed: ${err.message}`, 'error');
    showBootError(`Remove failed: ${err.message}`);
    writeLog(String(err.message), 'error');
  } finally {
    savingModel = false;
    await refreshModelCacheStatus(modelId);
  }
}

/**
 * On startup: if the selected/recommended model is already saved, load from local cache.
 */
async function autoLoadIfCached() {
  if (autoLoadAttempted || booted) return;
  autoLoadAttempted = true;
  try {
    await ensureWebLLM();
    if (!catalogLoaded) await loadModelCatalog();
    syncLabels();
    const modelId = selectedModelId();
    if (!modelId) {
      setModelCacheStatus('No model available yet.', 'idle');
      return;
    }
    const cached = await refreshModelCacheStatus(modelId);
    const kept = isModelKept(modelId);
    if (cached && kept) {
      setModelCacheStatus('Local model found — loading for quick startup…', 'busy');
      writeLog(`Auto-load from IndexedDB · ${modelId}`, 'info');
      await loadLLM({ modelId, fromCache: true });
    } else if (cached) {
      setModelCacheStatus(
        'Weights found in browser storage. Press Start AI for a fast load, or Save Model Locally to keep them.',
        'ok'
      );
    } else {
      setModelCacheStatus(
        'No local model yet. Press Start AI (CDN) or Save Model Locally to download and keep weights.',
        'idle'
      );
    }
  } catch (err) {
    writeLog(`Auto-load check: ${err.message}`, 'warn');
    setModelCacheStatus('Could not check local cache — use Start AI to load from the CDN.', 'error');
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

function buildMessages(extraUserContent, modeId) {
  const preset = getModePreset(modeId);
  const base = els.systemPrompt?.value.trim() || DEFAULT_SYSTEM_PROMPT;
  // Append mode instruction without wiping a custom system prompt
  const system = base.includes(preset.systemExtra)
    ? base
    : `${base}\n\n${preset.systemExtra}`;
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

/**
 * Core completion call (single attempt).
 */
async function createCompletionOnce(messages, onDelta) {
  const maxTokens = resolveMaxTokens();
  const temperature = resolveTemperature();
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

/**
 * Prepare messages for the active mode: estimate tokens, resize context, fit history.
 */
async function prepareMessagesForMode(rawMessages, modeId) {
  const mid = modeId || selectedChatMode();
  let messages = rawMessages;
  let promptTokens = estimateMessagesTokens(messages);

  // Dynamic context window / sliding window adjustment
  const budget = await ensureDynamicContext(promptTokens, mid);
  const reserve = getModePreset(mid).outputReserve || 512;
  const fitBudget = Math.max(512, budget - reserve);

  messages = fitMessagesToBudget(messages, fitBudget, mid);
  promptTokens = estimateMessagesTokens(messages);

  if (els.tokenMeterLabel) {
    const draftLen = els.promptInput?.value?.length || 0;
    els.tokenMeterLabel.textContent = `~${promptTokens.toLocaleString()} tok · window ${budget} · ${getModePreset(mid).label}`;
    void draftLen;
  }

  writeLog(
    `Context prep · ~${promptTokens} tok / window ${budget} · mode ${mid}`,
    promptTokens > budget * CONTEXT_SOFT_RATIO ? 'warn' : 'info'
  );
  return { messages, promptTokens, budget, modeId: mid };
}

/**
 * Error-fallback wrapper: retries on ContextWindowSizeExceededError with compression.
 */
async function runOneCompletion(messages, onDelta, modeId) {
  const mid = modeId || selectedChatMode();
  let prepared = await prepareMessagesForMode(messages, mid);
  let attempt = 0;

  while (attempt <= CONTEXT_OVERFLOW_RETRIES) {
    try {
      return await createCompletionOnce(prepared.messages, onDelta);
    } catch (err) {
      if (!isContextWindowError(err) || attempt >= CONTEXT_OVERFLOW_RETRIES) {
        throw err;
      }
      writeLog(err.message || 'ContextWindowSizeExceededError', 'warn');
      setStatus('Context full — compressing & retrying…', 'busy');
      const fallbackMsgs = await fallbackAfterContextOverflow(
        prepared.messages,
        mid,
        attempt
      );
      prepared = await prepareMessagesForMode(fallbackMsgs, mid);
      attempt += 1;
    }
  }
  throw new Error('Context window exceeded after fallback retries.');
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
  const modeId = selectedChatMode();

  try {
    for (let i = 0; i < parts.length; i++) {
      if (!generating && i > 0) break;
      setStatus(`Analyzing chunk ${i + 1}/${parts.length}…`, 'busy');
      await yieldToMain();

      const header =
        parts.length === 1
          ? parts[i]
          : `Analyze code chunk ${i + 1} of ${parts.length}. Focus on this segment only; later chunks continue the same file.\n\n\`\`\`\n${parts[i]}\n\`\`\``;

      // code mode keeps chunked full-file parsing; writing/short use leaner history
      let messages;
      if (parts.length === 1) {
        messages = buildMessages(undefined, modeId);
      } else {
        const preset = getModePreset(modeId);
        const base = els.systemPrompt?.value.trim() || DEFAULT_SYSTEM_PROMPT;
        const system = base.includes(preset.systemExtra)
          ? base
          : `${base}\n\n${preset.systemExtra}`;
        messages = [
          { role: 'system', content: system },
          { role: 'user', content: header },
        ];
      }

      const { reply, finishReason } = await runOneCompletion(
        messages,
        (partial) => {
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
      },
        modeId
      );

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
    if (isContextWindowError(err)) {
      showPasteWarn(
        'Context window was exceeded even after auto-compress. Try Short mode, clear chat, or paste a smaller slice.'
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
  const kept = getKeptModels();
  appendMsg(
    'assistant',
    kept.length
      ? `Session erased from memory. ${kept.length} saved model(s) remain in local IndexedDB for fast reload.`
      : 'Session erased from memory. No models are saved locally.'
  );
  syncUndoButtons();
  updateTokenMeter();
  showBootError('');
  await refreshModelCacheStatus();
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

  const save = () => saveModelLocally().catch((e) => showBootError(e.message));
  const remove = () => removeSavedModel().catch((e) => showBootError(e.message));
  els.saveModelBtn?.addEventListener('click', save);
  els.saveModelBtnAdv?.addEventListener('click', save);
  els.removeModelBtn?.addEventListener('click', remove);
  els.removeModelBtnAdv?.addEventListener('click', remove);
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
els.chatMode?.addEventListener('change', () => {
  syncLabels();
  if (!booted || !engine || !activeModelId) return;
  const modeId = selectedChatMode();
  const chatOpts = buildChatOptsForMode(modeId);
  reloadEngineContext(chatOpts, modeId)
    .then(() => writeLog(`Chat mode → ${modeId}`, 'info'))
    .catch((e) => {
      writeLog(`Mode switch failed: ${e.message}`, 'error');
      showBootError(`Mode switch failed: ${e.message}`);
    });
});
els.configMode?.addEventListener('change', () => {
  const manual = els.configMode.value === 'manual';
  if (els.modelSelect) els.modelSelect.disabled = !manual;
  if (els.categorySelect) els.categorySelect.disabled = !manual;
  syncLabels();
  refreshModelCacheStatus().catch(() => {});
});
els.categorySelect?.addEventListener('change', () => {
  populateModelSelect(els.modelSelect?.value);
  refreshModelCacheStatus().catch(() => {});
});
els.modelSelect?.addEventListener('change', () => {
  refreshModelCacheStatus().catch(() => {});
});
els.refreshModels?.addEventListener('click', () => {
  loadModelCatalog()
    .then(() => refreshModelCacheStatus())
    .catch((e) => showBootError(e.message));
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
  'Chat stays in RAM for this tab. Use Save Model Locally to keep weights in IndexedDB for faster next visits.'
);
syncUndoButtons();
setModelCacheButtons({ canSave: true, canRemove: false, busy: false });

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
loadModelCatalog()
  .then(() => autoLoadIfCached())
  .catch((e) => writeLog(`Catalog: ${e.message}`, 'warn'));

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
