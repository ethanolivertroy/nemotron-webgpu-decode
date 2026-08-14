// z = y * silu(gate), then RMS over last dim in 8 groups of 960, times weight.
// Dispatch 8 workgroups.

struct Params {
  group_size: u32,
  n_groups: u32,
  eps: f32,
  _pad: f32,
};

@group(0) @binding(0) var<storage, read_write> Y: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> Z: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 64u;
var<workgroup> red: array<f32, WG>;

fn silu(v: f32) -> f32 {
  return v / (1.0 + exp(-v));
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wg.x;
  let tid = lid.x;
  if (g >= params.n_groups) { return; }
  let gs = params.group_size;
  let base = g * gs;
  var ss = 0.0;
  var i = tid;
  loop {
    if (i >= gs) { break; }
    let v = Y[base + i] * silu(gate[base + i]);
    Z[base + i] = v;
    ss = ss + v * v;
    i = i + WG;
  }
  red[tid] = ss;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(red[0] / f32(gs) + params.eps);
  i = tid;
  loop {
    if (i >= gs) { break; }
    Z[base + i] = Z[base + i] * inv * weight[base + i];
    i = i + WG;
  }
}
