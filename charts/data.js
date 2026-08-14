// Measured on MacBook Pro M5 Max, 64 GB, Chrome 151, WebGPU -> Metal
// (vendor: apple, architecture: metal-3). 128 greedy tokens, same prompt.
// Do not mix these with SwiftShader numbers.

export const chartData = {
  sample: false,
  env: 'M5 Max / Chrome 151 / Metal',
  decode_tok_s: {
    labels: ['stock ORT', 'ORT shader swap', 'fused engine v1', 'raw encode', 'chat template'],
    values: [69.76, 71.3, 100.47, 129.11, 115.04],
  },
  gpu_vs_wall: {
    labels: ['stock wall', 'engine wall', 'engine GPU'],
    values: [69.76, 115.04, 142.48],
  },
  iteration_tok_s: {
    labels: [
      'stock ORT',
      'r2 body swap',
      'owned GEMV loop',
      'parallel Mamba SSD',
      'GPU-resident token',
      'f16 MLP raw encode',
      'chat template',
    ],
    values: [69.76, 71.3, 100.47, 116.63, 125.45, 129.11, 115.04],
  },
  gemv_lm_head_ms: {
    labels: ['ORT tile', 'sg4', 'f16', 'sg32', 'subgroup matrix'],
    values: [0.634, 0.488, 0.488, 0.5, 2.642],
  },
};
