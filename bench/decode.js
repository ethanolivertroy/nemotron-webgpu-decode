import { AutoTokenizer, env } from '@huggingface/transformers';
import { requestGpu, adapterSummary } from '../kernels/gpu.js';
import { DecodeEngine } from '../engine/runtime.js';

const MODEL = 'onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX';
const PROMPT = 'Write a haiku about GPU kernels.';
const MAX_NEW = parseInt(new URLSearchParams(location.search).get('tokens') || '128', 10);

env.allowRemoteModels = true;
env.useBrowserCache = new URLSearchParams(location.search).get('cache') !== 'off';

const logEl = document.getElementById('log');
const lines = [];
function say(s) {
  lines.push(s);
  logEl.textContent = lines.join('\n');
  console.log('BENCH_LOG ' + s);
}

async function postResults(payload) {
  try {
    await fetch('/__results', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch {
    /* headless may still pick up console */
  }
  console.log('BENCH_RESULT ' + JSON.stringify(payload));
}

async function main() {
  try {
    say('requesting gpu');
    const gpu = await requestGpu();
    const adapter = adapterSummary(gpu);
    say('adapter ' + JSON.stringify(adapter));
    if (!gpu.device.features.has('subgroups')) {
      throw new Error('subgroups feature required for sg4 GEMV');
    }

    const tLoad0 = performance.now();
    say('loading tokenizer');
    const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
    say('loading q4 weights');
    let lastPct = '';
    const engine = await DecodeEngine.create(gpu, {
      onProgress: (name, got, total) => {
        if (!total) return;
        const pct = Math.floor((got / total) * 10) * 10;
        const key = `${name}:${pct}`;
        if (pct === 0 || key === lastPct) return;
        lastPct = key;
        say(`download ${name} ${pct}%`);
      },
    });
    const templated = tokenizer.apply_chat_template(
      [{ role: 'user', content: PROMPT }],
      { add_generation_prompt: true, tokenize: false },
    );
    const promptIds = tokenizer.encode(templated, { add_special_tokens: false });
    say(`prompt tokens=${promptIds.length} via=chat_template ids=${promptIds.slice(0, 8).join(',')}`);
    const loadMs = +(performance.now() - tLoad0).toFixed(1);

    say('warmup 8 tokens');
    engine.reset();
    await engine.generate(promptIds, 8);

    say(`timed generate ${MAX_NEW} tokens`);
    engine.reset();
    const t0 = performance.now();
    const outIds = await engine.generate(promptIds, MAX_NEW);
    const ms = performance.now() - t0;
    const tokPerSec = +(outIds.length / (ms / 1000)).toFixed(2);
    const gpuMs = engine.gpuSamples ? engine.gpuNs / 1e6 : null;
    const gpuMsPerTok = gpuMs != null ? gpuMs / engine.gpuSamples : null;
    const gpuTokPerSec = gpuMsPerTok != null ? +(1000 / gpuMsPerTok).toFixed(2) : null;
    const text = tokenizer.decode(outIds, { skip_special_tokens: true });
    say(`load=${loadMs}ms tokens=${outIds.length} ms=${ms.toFixed(1)} tok/s=${tokPerSec}`);
    if (gpuMs != null) {
      say(`gpu=${gpuMs.toFixed(1)}ms samples=${engine.gpuSamples} gpu_ms/tok=${gpuMsPerTok.toFixed(3)} gpu_tok/s=${gpuTokPerSec}`);
    }
    say('output: ' + text.slice(0, 240));

    const payload = {
      type: 'decode-engine',
      adapter,
      prompt: PROMPT,
      prompt_tokens: promptIds.length,
      load_ms: loadMs,
      tokens: outIds.length,
      ms: +ms.toFixed(1),
      tokPerSec,
      gpu_ms: gpuMs != null ? +gpuMs.toFixed(1) : null,
      gpu_samples: engine.gpuSamples,
      gpu_ms_per_tok: gpuMsPerTok != null ? +gpuMsPerTok.toFixed(3) : null,
      gpu_tokPerSec: gpuTokPerSec,
      output: text.slice(0, 500),
      note: 'chat template, GPU-resident token, real q4 zp, fused relu2, f16 mlp, skip prefill lm_head',
    };
    await postResults(payload);
  } catch (e) {
    const msg = e && e.stack ? e.stack : String(e);
    say('BENCH_ERROR ' + msg);
    await postResults({ type: 'decode-engine', error: msg });
  }
}

main();
