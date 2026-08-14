/** Nemotron 3 Nano 4B (nemotron_h) decode-time shapes. */
export const NANO4B = {
  hidden: 3136,
  intermediate: 12544,
  vocab: 131072,
  layers: 42,
  nMamba: 21,
  nMlp: 17,
  nAttn: 4,
  heads: 40,
  kvHeads: 8,
  headDim: 128,
  mambaHeads: 96,
  mambaHeadDim: 80,
  ssmState: 128,
  convKernel: 4,
  nGroups: 8,
  rmsEps: 1e-5,
  dtMin: 0.001,
  mambaIn: 17504,
  convDim: 9728,
  headsPerGroup: 12,
  groupRms: 960,
  maxSeq: 256,
  eosTokenId: 11,
};

export const HF_ONNX =
  'https://huggingface.co/onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX/resolve/main/onnx';

export const ATTN_LAYERS = [12, 17, 24, 32];

export const LAYER_TYPES = [
  'mamba', 'mlp', 'mamba', 'mlp', 'mamba', 'mlp', 'mamba', 'mamba', 'mlp',
  'mamba', 'mlp', 'mamba', 'attention', 'mlp', 'mamba', 'mlp', 'mamba',
  'attention', 'mlp', 'mamba', 'mlp', 'mamba', 'mlp', 'mamba', 'attention',
  'mlp', 'mamba', 'mlp', 'mamba', 'mlp', 'mamba', 'mamba', 'attention',
  'mlp', 'mamba', 'mamba', 'mamba', 'mlp', 'mamba', 'mlp', 'mamba', 'mlp',
];

export function mambaInner() {
  return NANO4B.mambaHeads * NANO4B.mambaHeadDim;
}

export function qkvOut() {
  return NANO4B.heads * NANO4B.headDim + 2 * NANO4B.kvHeads * NANO4B.headDim;
}
