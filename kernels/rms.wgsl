// Pre-norm RMS (no residual). Dispatch one workgroup per row.

struct Params {
  hidden: u32,
  rows: u32,
  eps: f32,
  _pad: f32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 64u;
var<workgroup> red: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= params.rows) { return; }
  let tid = lid.x;
  let hidden = params.hidden;
  let base = row * hidden;
  var ss: f32 = 0.0;
  var i = tid;
  loop {
    if (i >= hidden) { break; }
    let v = X[base + i];
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
  let inv = inverseSqrt(red[0] / f32(hidden) + params.eps);
  i = tid;
  loop {
    if (i >= hidden) { break; }
    Y[base + i] = X[base + i] * inv * weight[i];
    i = i + WG;
  }
}
