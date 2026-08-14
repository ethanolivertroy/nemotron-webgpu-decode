# Measured results (M5 Max)

MacBook Pro M5 Max, 64 GB, Chrome 151, WebGPU to Metal
(`vendor: apple`, `architecture: metal-3`, `shaderF16: true`).
Gate prompt uses the instruct chat template. 128 new tokens, greedy.
Do not mix these with software-adapter numbers.

## Full-model decode

| Mode | tok/s | ms | notes |
|------|-------|-----|-------|
| stock ORT | **69.76** | 1834.9 | transformers.js + ORT WebGPU |
| ORT shader-body swap | 71.30 | 1795.3 | +2.2%, ORT still owns dispatch |
| fused engine, first loop | 100.47 | 1274.1 | owned GEMV + fused Mamba/attn |
| one-pass + RW activations | 95.94 | 1334.2 | regression: GEMV A as read_write |
| + parallel Mamba SSD, sg32 | 116.63 | 1097.5 | GPU 141 tok/s |
| + GPU-resident next token | 125.45 | 1020.4 | GPU 140 tok/s |
| + f16 GEMV mlp_up / mlp_down | 129.11 | 991.4 | raw 8-token encode, not chat template |
| + chat template (24 prompt ids) | **115.04** | 1112.7 | GPU 142 tok/s, first token matches ORT |

Chat streaming with overlapped readback measures about 112 tok/s wall on the
same machine. The 129 figure is the earlier raw-encode iteration. Do not treat
it as the chat-template number.

Raw: `harness/metal/m5max-decode-chat-template.json`.

## GEMV microbench (lm_head 3136 x 131072 q4)

| Kernel | ms | vs ORT tile |
|--------|-----|-------------|
| ORT tile (WG=128, 8 cols) | 0.634 | 1.00x |
| sg4 (WG=256, 4 cols) | 0.488 | 1.30x |
| f16 inner product | 0.488 | 1.30x |
| sg32 (WG=32, 8 cols) | 0.500 | 1.27x |
| subgroup-matrix 8x8 | 2.642 | 0.24x |

f16 wins mlp_up and mlp_down. Those two shapes use the f16 kernel. lm_head stays sg4.

Raw: `harness/metal/m5max-engine-gemv-f16.json`.

Written by Grok 4.6.
