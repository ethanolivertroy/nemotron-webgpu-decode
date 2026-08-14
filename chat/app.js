import { AutoTokenizer, env } from '@huggingface/transformers';
import { requestGpu } from '../kernels/gpu.js';
import { DecodeEngine } from '../engine/runtime.js';
import { NANO4B } from '../engine/config.js';

const MODEL = 'onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX';
const MAX_NEW = 128;

env.allowRemoteModels = true;
env.useBrowserCache = true;

const loadBtn = document.getElementById('loadBtn');
const sendBtn = document.getElementById('sendBtn');
const promptEl = document.getElementById('prompt');
const form = document.getElementById('form');
const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');
const dotEl = document.getElementById('dot');
const tpsEl = document.getElementById('tps');
const barEl = document.getElementById('bar');
const gpuWarn = document.getElementById('gpuWarn');
const noThinkEl = document.getElementById('noThink');

let tokenizer = null;
let engine = null;
let busy = false;
const messages = [];

function setStatus(text, kind) {
  statusEl.textContent = text;
  dotEl.className = 'dot' + (kind ? ` ${kind}` : '');
}

function addBubble(role, text) {
  const el = document.createElement('div');
  el.className = `bubble ${role === 'user' ? 'user' : 'asst'}`;
  el.textContent = text;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
  return el;
}

function promptIdsFor(history) {
  const text = tokenizer.apply_chat_template(history, {
    add_generation_prompt: true,
    tokenize: false,
  });
  return tokenizer.encode(text, { add_special_tokens: false });
}

async function load() {
  if (!navigator.gpu) {
    setStatus('WebGPU missing. Use Chrome.', 'err');
    gpuWarn.textContent = 'WebGPU is not available in this browser. Open this page in Chrome 113+ with a real GPU.';
    return;
  }
  loadBtn.disabled = true;
  setStatus('Requesting GPU…', 'busy');
  try {
    const gpu = await requestGpu();
    if (!gpu.device.features.has('subgroups')) {
      throw new Error('This GPU does not expose the subgroups feature the q4 GEMV kernels need.');
    }
    setStatus('Loading tokenizer…', 'busy');
    tokenizer = await AutoTokenizer.from_pretrained(MODEL);
    const got = {};
    const totals = {};
    setStatus('Downloading q4 weights (~2.5 GB)…', 'busy');
    engine = await DecodeEngine.create(gpu, {
      onProgress: (name, n, total) => {
        got[name] = n;
        if (total) totals[name] = total;
        const g = Object.values(got).reduce((a, b) => a + b, 0);
        const t = Object.values(totals).reduce((a, b) => a + b, 0) || 1;
        barEl.style.width = `${Math.min(100, (100 * g) / t)}%`;
        const pct = Math.floor((100 * g) / t);
        setStatus(`Downloading weights ${pct}%`, 'busy');
      },
    });
    promptEl.disabled = false;
    sendBtn.disabled = false;
    setStatus('Ready', 'ready');
    loadBtn.textContent = 'Model loaded';
    promptEl.focus();
  } catch (e) {
    loadBtn.disabled = false;
    setStatus('Load failed', 'err');
    addBubble('assistant', String(e && e.message ? e.message : e));
  }
}

async function send(ev) {
  ev.preventDefault();
  if (!engine || busy) return;
  let content = promptEl.value.trim();
  if (!content) return;
  if (noThinkEl.checked && !content.includes('/no_think')) {
    content = `/no_think ${content}`;
  }
  promptEl.value = '';
  messages.push({ role: 'user', content });
  addBubble('user', content.replace(/^\/no_think\s+/, ''));
  const asst = addBubble('assistant', '');
  busy = true;
  sendBtn.disabled = true;
  setStatus('Generating…', 'busy');
  tpsEl.textContent = '';
  try {
    const ids = promptIdsFor(messages);
    if (ids.length >= NANO4B.maxSeq - 8) {
      throw new Error('Conversation is too long for this demo (max 256 tokens of context). Refresh to reset.');
    }
    const outIds = [];
    const t0 = performance.now();
    for await (const id of engine.generateStream(ids, { maxNew: MAX_NEW })) {
      outIds.push(id);
      asst.textContent = tokenizer.decode(outIds, { skip_special_tokens: true });
      logEl.scrollTop = logEl.scrollHeight;
      const dt = (performance.now() - t0) / 1000;
      if (dt > 0.05) tpsEl.textContent = `${(outIds.length / dt).toFixed(1)} tok/s`;
    }
    const text = tokenizer.decode(outIds, { skip_special_tokens: true });
    messages.push({ role: 'assistant', content: text });
    setStatus('Ready', 'ready');
  } catch (e) {
    asst.textContent = String(e && e.message ? e.message : e);
    setStatus('Error', 'err');
  } finally {
    busy = false;
    sendBtn.disabled = false;
    promptEl.focus();
  }
}

loadBtn.addEventListener('click', load);
form.addEventListener('submit', send);

if (!navigator.gpu) {
  loadBtn.disabled = true;
  setStatus('WebGPU missing', 'err');
} else {
  navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).then((adapter) => {
    if (!adapter) return;
    const info = adapter.info || {};
    gpuWarn.textContent = `GPU: ${info.vendor || '?'} ${info.architecture || ''} · shader-f16=${adapter.features.has('shader-f16')} · subgroups=${adapter.features.has('subgroups')}`;
  });
}
