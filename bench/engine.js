/**
 * Engine GEMV microbench: we own dispatch.
 *
 * Compares ORT MatMulNBits (8 N-columns / 128 threads, Metal dump 059)
 * against:
 *   col   : 1 thread / column, each re-reads A (naive)
 *   shm   : 256 threads / 256 columns, A tiled in shared memory
 *   sg4   : WG=256, 4 columns, split-K + subgroup reduce
 *
 * Persistent buffers. GPU timestamps when available, else readback-flushed
 * wall clock. Synthetic tok/s is GEMV-only (no RMS / Mamba scan / attn).
 */
import { requestGpu, adapterSummary, compileCompute, createBuffer, createEmptyStorage } from '../kernels/gpu.js';
import { NANO4B, mambaInner, qkvOut } from '../engine/config.js';
import gemvSgWgsl from '../kernels/matmul_nbits_sg.wgsl?raw';
import gemvSg32Wgsl from '../kernels/matmul_nbits_sg32.wgsl?raw';
import gemvF16Wgsl from '../kernels/matmul_nbits_sg_f16.wgsl?raw';
import gemvMmaWgsl from '../kernels/matmul_nbits_mma.wgsl?raw';
import ortTileWgsl from '../harness/shaders/wgsl/059_MatMulNBits.wgsl?raw';

const ITERS = parseInt(new URLSearchParams(location.search).get('iters') || '30', 10);
const WARMUP = 32;

function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s;
  };
}

function packRandomQ4(N, K, seed) {
  const nBlocks = Math.ceil(K / 32);
  const packed = new Uint32Array(N * nBlocks * 4);
  const scales = new Float32Array(N * nBlocks);
  const rnd = lcg(seed);
  for (let i = 0; i < packed.length; i++) packed[i] = rnd();
  for (let i = 0; i < scales.length; i++) scales[i] = 0.04 + (rnd() & 255) / 2048;
  const zeros = new Uint32Array(Math.ceil((N * nBlocks) / 8)).fill(0x88888888);
  const A = new Float32Array(K);
  for (let i = 0; i < K; i++) A[i] = ((rnd() & 0xffff) / 0xffff) * 2 - 1;
  return { packed, scales, zeros, A, nBlocks, N, K };
}

function packU32(fields) {
  const buf = new ArrayBuffer(Math.max(16, fields.length * 4));
  const view = new DataView(buf);
  fields.forEach((f, i) => view.setUint32(i * 4, f, true));
  return buf;
}

function uniformBuffer(device, bytes) {
  const buf = device.createBuffer({
    size: Math.max(16, bytes.byteLength),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(bytes));
  buf.unmap();
  return buf;
}

function bind(device, pipeline, buffers) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
  });
}

async function timeDispatch(device, pipeline, bg, wgx, iters, readBuf) {
  const hasTs = device.features.has('timestamp-query');
  const querySet = hasTs ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const resolveBuf = hasTs
    ? device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    : null;
  const staging = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encode = (n, withTs) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(
      withTs && querySet
        ? {
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: 0,
              endingOfPassWriteIndex: 1,
            },
          }
        : {},
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    for (let i = 0; i < n; i++) pass.dispatchWorkgroups(wgx, 1, 1);
    pass.end();
    if (withTs && querySet && resolveBuf) {
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuf, 0);
      encoder.copyBufferToBuffer(resolveBuf, 0, staging, 0, 16);
    } else {
      encoder.copyBufferToBuffer(readBuf, 0, staging, 0, 16);
    }
    device.queue.submit([encoder.finish()]);
  };

  device.pushErrorScope('validation');
  encode(WARMUP, false);
  await staging.mapAsync(GPUMapMode.READ);
  staging.unmap();
  const warmErr = await device.popErrorScope();
  if (warmErr) throw new Error(warmErr.message);

  device.pushErrorScope('validation');
  const t0 = performance.now();
  encode(iters, !!querySet);
  await staging.mapAsync(GPUMapMode.READ);
  const wallMs = (performance.now() - t0) / iters;
  let gpuMs = wallMs;
  if (querySet) {
    const ns = new BigUint64Array(staging.getMappedRange().slice(0));
    const delta = Number(ns[1] - ns[0]);
    if (delta > 0) gpuMs = delta / 1e6 / iters;
    staging.unmap();
  } else {
    staging.unmap();
  }
  const err = await device.popErrorScope();
  if (err) throw new Error(err.message);
  staging.destroy();
  if (resolveBuf) resolveBuf.destroy();
  if (querySet) querySet.destroy();
  return { gpu_ms: gpuMs, wall_ms: wallMs };
}

function destroyAll(bufs) {
  for (const b of bufs) b.destroy();
}

async function benchShape(device, pipelines, shape) {
  const { K, N, label } = shape;
  const data = packRandomQ4(N, K, 7 + N + K);
  const aBuf = createBuffer(device, data.A, GPUBufferUsage.STORAGE);
  const bBuf = createBuffer(device, data.packed, GPUBufferUsage.STORAGE);
  const sBuf = createBuffer(device, data.scales, GPUBufferUsage.STORAGE);
  const zBuf = createBuffer(device, data.zeros, GPUBufferUsage.STORAGE);
  const yOrt = createEmptyStorage(device, N * 4);
  const ySg = createEmptyStorage(device, N * 4);
  const ySg32 = createEmptyStorage(device, N * 4);
  const yF16 = createEmptyStorage(device, N * 4);
  const yMma = createEmptyStorage(device, N * 4);

  const colU = uniformBuffer(device, packU32([data.K, data.N, data.nBlocks, 0]));
  const numNTile = Math.ceil(N / 8);
  const ortU = uniformBuffer(
    device,
    packU32([
      1,
      N,
      K,
      K / 4,
      data.nBlocks,
      32,
      data.nBlocks,
      data.nBlocks,
      numNTile,
      1,
      0,
      1,
    ]),
  );

  const sgBg = pipelines.sg ? bind(device, pipelines.sg, [aBuf, bBuf, sBuf, zBuf, ySg, colU]) : null;
  const sg32Bg = pipelines.sg32 ? bind(device, pipelines.sg32, [aBuf, bBuf, sBuf, zBuf, ySg32, colU]) : null;
  const f16Bg = pipelines.f16 ? bind(device, pipelines.f16, [aBuf, bBuf, sBuf, zBuf, yF16, colU]) : null;
  const mmaBg = pipelines.mma ? bind(device, pipelines.mma, [aBuf, bBuf, sBuf, zBuf, yMma, colU]) : null;
  const ortBg = bind(device, pipelines.ort, [aBuf, bBuf, sBuf, zBuf, yOrt, ortU]);

  const sg = sgBg ? await timeDispatch(device, pipelines.sg, sgBg, Math.ceil(N / 4), ITERS, ySg) : { gpu_ms: 0 };
  const sg32 = sg32Bg ? await timeDispatch(device, pipelines.sg32, sg32Bg, Math.ceil(N / 8), ITERS, ySg32) : { gpu_ms: 0 };
  const f16 = f16Bg ? await timeDispatch(device, pipelines.f16, f16Bg, Math.ceil(N / 4), ITERS, yF16) : { gpu_ms: 0 };
  const mma = mmaBg ? await timeDispatch(device, pipelines.mma, mmaBg, Math.ceil(N / 8), ITERS, yMma) : { gpu_ms: 0 };
  const ort = await timeDispatch(device, pipelines.ort, ortBg, numNTile, ITERS, yOrt);

  destroyAll([aBuf, bBuf, sBuf, zBuf, ySg, ySg32, yF16, yMma, yOrt, colU, ortU]);
  const round = (x) => +x.toFixed(3);
  return {
    label,
    K,
    N,
    sg4_ms: round(sg.gpu_ms),
    sg32_ms: round(sg32.gpu_ms),
    f16_ms: round(f16.gpu_ms),
    mma_ms: round(mma.gpu_ms),
    ort_ms: round(ort.gpu_ms),
    sg4_vs_ort: sg.gpu_ms ? +(ort.gpu_ms / sg.gpu_ms).toFixed(2) : null,
    sg32_vs_ort: sg32.gpu_ms ? +(ort.gpu_ms / sg32.gpu_ms).toFixed(2) : null,
    f16_vs_ort: f16.gpu_ms ? +(ort.gpu_ms / f16.gpu_ms).toFixed(2) : null,
    mma_vs_ort: mma.gpu_ms ? +(ort.gpu_ms / mma.gpu_ms).toFixed(2) : null,
  };
}

function estimateDecode(rows, key) {
  const by = Object.fromEntries(rows.map((r) => [r.label, r]));
  const gemv = (name, count) => (by[name]?.[key] ?? 0) * count;
  void mambaInner;
  return (
    gemv('h_h', NANO4B.nMamba * 2 + NANO4B.nAttn) +
    gemv('mlp_up', NANO4B.nMlp) +
    gemv('mlp_down', NANO4B.nMlp) +
    gemv('qkv', NANO4B.nAttn) +
    gemv('lm_head', 1)
  );
}

async function main() {
  const log = document.getElementById('log');
  const say = (s) => {
    log.textContent += s + '\n';
    console.log('BENCH_LOG ' + s);
  };
  try {
    const gpu = await requestGpu({ forceFallback: false });
    say('adapter=' + JSON.stringify(adapterSummary(gpu)));
    async function tryCompile(code, label) {
      try {
        return (await compileCompute(gpu.device, code, label)).pipeline;
      } catch (e) {
        say(`compile skip ${label}: ${e.message.split('\n')[0]}`);
        return null;
      }
    }
    const sg = await tryCompile(gemvSgWgsl, 'engine-gemv-sg4');
    const sg32 = await tryCompile(gemvSg32Wgsl, 'engine-gemv-sg32');
    const f16 = await tryCompile(gemvF16Wgsl, 'engine-gemv-f16');
    const mma = await tryCompile(gemvMmaWgsl, 'engine-gemv-mma');
    const ort = await tryCompile(ortTileWgsl, 'engine-ort-tile8');
    const pipelines = { sg, sg32, f16, mma, ort };

    const shapes = [
      { label: 'h_h', K: NANO4B.hidden, N: NANO4B.hidden },
      { label: 'mlp_up', K: NANO4B.hidden, N: NANO4B.intermediate },
      { label: 'mlp_down', K: NANO4B.intermediate, N: NANO4B.hidden },
      { label: 'qkv', K: NANO4B.hidden, N: qkvOut() },
      { label: 'lm_head', K: NANO4B.hidden, N: NANO4B.vocab },
    ];

    const rows = [];
    for (const sh of shapes) {
      say(`shape ${sh.label} K=${sh.K} N=${sh.N} packing…`);
      const row = await benchShape(gpu.device, pipelines, sh);
      rows.push(row);
      say(
        `GEMV ${row.label}: sg4=${row.sg4_ms} sg32=${row.sg32_ms} f16=${row.f16_ms} mma=${row.mma_ms} ort=${row.ort_ms}`,
      );
    }

    const sgMs = estimateDecode(rows, 'sg4_ms');
    const sg32Ms = estimateDecode(rows, 'sg32_ms');
    const f16Ms = estimateDecode(rows, 'f16_ms');
    const mmaMs = estimateDecode(rows, 'mma_ms');
    const ortMs = estimateDecode(rows, 'ort_ms');
    const estimate = {
      note: 'GEMV-only synthetic decode. f16 and mma are experimental.',
      sg4_ms: +sgMs.toFixed(2),
      sg32_ms: +sg32Ms.toFixed(2),
      f16_ms: +f16Ms.toFixed(2),
      mma_ms: +mmaMs.toFixed(2),
      ort_ms: +ortMs.toFixed(2),
      sg4_tok_s: +(1000 / sgMs).toFixed(1),
      sg32_tok_s: +(1000 / sg32Ms).toFixed(1),
      f16_tok_s: f16Ms ? +(1000 / f16Ms).toFixed(1) : null,
      mma_tok_s: mmaMs ? +(1000 / mmaMs).toFixed(1) : null,
      ort_tok_s: +(1000 / ortMs).toFixed(1),
    };
    say(
      `SYNTHETIC: sg4=${estimate.sg4_tok_s} sg32=${estimate.sg32_tok_s} f16=${estimate.f16_tok_s} mma=${estimate.mma_tok_s} ort=${estimate.ort_tok_s}`,
    );

    const result = {
      type: 'engine-gemv',
      adapter: adapterSummary(gpu),
      hidden: NANO4B.hidden,
      iters: ITERS,
      rows,
      estimate,
    };
    console.log('BENCH_RESULT ' + JSON.stringify(result));
    log.textContent = JSON.stringify(result, null, 2);
    try {
      await fetch('/__results', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(result),
      });
    } catch {
      /* collector absent */
    }
  } catch (err) {
    console.log('BENCH_ERROR ' + err + '\n' + (err.stack || ''));
    document.getElementById('log').textContent = 'ERROR: ' + err + '\n' + (err.stack || '');
  }
}

main();
