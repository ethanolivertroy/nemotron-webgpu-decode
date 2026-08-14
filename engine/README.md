# From-scratch decode engine

ORT shader-body swaps cannot 3-4x Nemotron. This folder is a custom WebGPU
decode runtime: we own workgroup size, dispatch, fusion, and the layer loop.

## Why not the ORT path

Stock MatMulNBits launches 128 threads to write 8 N-columns. We cannot
change that grid from `createShaderModule`. Round 2 body swaps were +2%.
GEMV-only owned dispatch (sg4) is 140 tok/s vs ORT tile 107 tok/s. Stock
full-model is 70 tok/s, so the rest of the gap is ORT graph tax plus the
Mamba/attn subgraph. This engine fuses that subgraph and submits one
command encoder per token.

## Decode loop (matches ONNX q4 then-branch)

Pre-norm RMS, mixer, residual add. Mixers:

- MLP: q4 GEMV up, relu2, q4 GEMV down
- Mamba-2: q4 in_proj, conv1d k=4 + silu, grouped SSD, gated group RMS, q4 out_proj
- Attention (layers 12, 17, 24, 32): q/k/v GEMV, KV append, GQA softmax, o GEMV. No RoPE (`do_rotary: 0`)

Final RMS, lm_head GEMV, GPU argmax. Next token stays on GPU (embed reads the
argmax buffer). q4 zero-points are loaded from the ONNX data files.

GEMV: sg4 (`matmul_nbits_sg.wgsl`, WG=256, 4 columns) for most shapes; sg32 (`matmul_nbits_sg32.wgsl`, WG=32, 8 columns) when K=3136 and N is in [8192, 65536) except mlp_up; f16 inner-product (`matmul_nbits_sg_f16.wgsl`) for mlp_up and mlp_down. Mamba SSD dispatches one workgroup per (head, dim).

## Weights

Not in git. Download the two ONNX data files into `harness/weights/`:

```
node scripts/fetch-weights.mjs
```

Files: `model_q4.onnx_data` (~2.0 GB) and `model_q4.onnx_data_1` (~443 MB)
from `onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX`.

## Bench vs stock 70 tok/s

```
CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT=5179 PAGE_TIMEOUT_MS=2400000
CHROME_EXTRA_ARGS="--no-first-run --no-default-browser-check --disable-sync"
node scripts/headless-bench.mjs --page decode --tokens 128 --angle metal --profile-dir /tmp/nemotron-m5-chrome-profile
```

Same gate prompt as stock, with the instruct chat template, 128 new tokens, greedy.

M5 Max (Chrome 151, Metal): fused engine **115.04 tok/s** wall vs stock **69.76 tok/s** (1.64x) on the chat-templated prompt (24 ids). GPU timestamps: 7.02 ms/step (**142 GPU tok/s**). First token id matches stock ORT. Raw: `harness/metal/m5max-decode-chat-template.json`.

The earlier **129.11 tok/s** figure used a raw 8-token `encode()` of the same string (no chat template) and collapsed to newlines. That was a prompt-format bug, not a zp packing bug. Embed / RMS / in_proj maxAbs vs CPU is ~1e-7.

