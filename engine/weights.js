import { createBuffer, createEmptyStorage } from '../kernels/gpu.js';
import index from './weight-index.json';
import aNegExp from './a-neg-exp.json';
import { NANO4B, HF_ONNX } from './config.js';

const CHUNK = 16 << 20;

export function keepTensor(name) {
  if (name.endsWith('conv1d.weight')) return false;
  if (name.includes('prefill')) return false;
  if (name.includes('A_neg_exp')) return false;
  if (name.includes('D_f32_unsq')) return false;
  return true;
}

function writeChunks(device, gpuBuf, bytes, dstOffset = 0) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let off = 0;
  while (off < u8.byteLength) {
    let n = Math.min(CHUNK, u8.byteLength - off);
    n -= n % 4;
    if (n <= 0) break;
    device.queue.writeBuffer(gpuBuf, dstOffset + off, u8, off, n);
    off += n;
  }
}

function localWeights() {
  return typeof location !== 'undefined' &&
    (location.hostname === '127.0.0.1' || location.hostname === 'localhost');
}

async function fetchFile(name, onProgress) {
  const url = localWeights() ? `/weights/${name}` : `${HF_ONNX}/${name}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`weights ${name}: HTTP ${res.status}${localWeights() ? ' (download into harness/weights/ first)' : ` from ${url}`}`);
  }
  const total = Number(res.headers.get('content-length') || 0);
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    onProgress(name, got, total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.byteLength;
  }
  return out;
}

export async function loadWeights(device, { onProgress } = {}) {
  const files = {};
  const names = [...new Set(index.external.filter((t) => keepTensor(t.name)).map((t) => t.location))];
  for (const loc of names) {
    files[loc] = await fetchFile(loc, onProgress);
  }

  const buffers = new Map();
  const meta = new Map();
  for (const t of index.external) {
    if (!keepTensor(t.name)) continue;
    const src = files[t.location];
    const slice = src.subarray(t.offset, t.offset + t.length);
    const gpu = createEmptyStorage(device, t.length);
    writeChunks(device, gpu, slice);
    buffers.set(t.name, gpu);
    meta.set(t.name, { dims: t.dims, dtype: t.dtype, length: t.length });
  }

  const byName = Object.fromEntries(index.external.map((t) => [t.name, t]));
  for (const [layer, values] of Object.entries(aNegExp)) {
    const dtT = byName[`model.layers.${layer}.mamba.dt_bias_expanded_decode`];
    const dT = byName[`model.layers.${layer}.mamba.D_expanded_decode`];
    if (!dtT || !dT) continue;
    const packed = new Float32Array(7680 + 7680 + 96);
    const dtSrc = files[dtT.location];
    const dSrc = files[dT.location];
    packed.set(new Float32Array(dtSrc.buffer, dtSrc.byteOffset + dtT.offset, 7680), 0);
    packed.set(new Float32Array(dSrc.buffer, dSrc.byteOffset + dT.offset, 7680), 7680);
    packed.set(Float32Array.from(values), 15360);
    const name = `model.layers.${layer}.mamba.ssd_aux`;
    buffers.set(name, createBuffer(device, packed, GPUBufferUsage.STORAGE));
    meta.set(name, { dims: [packed.length], dtype: 1, length: packed.byteLength });
  }

  for (const loc of names) files[loc] = null;
  return { buffers, meta, hidden: NANO4B.hidden, vocab: NANO4B.vocab };
}

export function q4Meta(meta, name) {
  const t = meta.get(name);
  if (!t) throw new Error(`missing ${name}`);
  const [N, nBlocks] = t.dims;
  return { N, nBlocks, K: nBlocks * 32 };
}
