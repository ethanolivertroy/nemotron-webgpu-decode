---
title: Nemotron 3 Nano WebGPU
emoji: 🟠
colorFrom: gray
colorTo: blue
sdk: static
header: mini
pinned: false
models:
  - onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX
---

# Nemotron 3 Nano WebGPU

In-browser WebGPU decode for Nemotron 3 Nano 4B q4. Chrome with WebGPU
required. About 2.5 GB of weights are fetched from the Hub at runtime.

On an M5 Max this engine is 115 tok/s wall / 142 tok/s GPU vs stock ONNX
Runtime at 70 tok/s (1.64x) on the chat-templated prompt. Kernels written
by Grok 4.6 in Cursor.

Source: [github.com/ethanolivertroy/nemotron-webgpu-decode](https://github.com/ethanolivertroy/nemotron-webgpu-decode)
