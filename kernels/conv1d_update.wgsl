// Causal conv1d update for Mamba-2 decode (kernel=4, silu).
// state layout: [channels, 4]. Roll left, append x, y = silu(dot(state, w) + b).
// Dispatch ceil(channels / 256).

struct Params {
  channels: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn silu(v: f32) -> f32 {
  return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= params.channels) { return; }
  let base = ch * 4u;
  let s0 = state[base + 1u];
  let s1 = state[base + 2u];
  let s2 = state[base + 3u];
  let s3 = X[ch];
  state[base + 0u] = s0;
  state[base + 1u] = s1;
  state[base + 2u] = s2;
  state[base + 3u] = s3;
  let acc = s0 * weight[base + 0u] + s1 * weight[base + 1u]
    + s2 * weight[base + 2u] + s3 * weight[base + 3u] + bias[ch];
  Y[ch] = silu(acc);
}
