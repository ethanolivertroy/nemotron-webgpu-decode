// Grouped Mamba-2 SSD decode (seq=1). Matches the ONNX then-branch.
// aux layout: dt_bias[96*80] | D[96*80] | A_neg_exp[96]
// B/C are per group (8 groups, 12 heads).
// One workgroup per (head, head_dim) so the 80-long dim loop is parallel.
// Dispatch n_heads * head_dim workgroups of 32.

struct Params {
  n_heads: u32,
  head_dim: u32,
  ssm_n: u32,
  heads_per_group: u32,
  dt_min: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> dt: array<f32>;
@group(0) @binding(4) var<storage, read> aux: array<f32>;
@group(0) @binding(5) var<storage, read_write> H: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

const WG: u32 = 32u;
const DT_BIAS: u32 = 0u;
const D_OFF: u32 = 7680u;
const A_OFF: u32 = 15360u;
var<workgroup> red: array<f32, WG>;

fn softplus(v: f32) -> f32 {
  if (v > 20.0) { return v; }
  return log(1.0 + exp(v));
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let hd = params.head_dim;
  let idx = wg.x;
  let h = idx / hd;
  let d = idx % hd;
  let tid = lid.x;
  if (h >= params.n_heads) { return; }
  let N = params.ssm_n;
  let group = h / params.heads_per_group;
  let Av = aux[A_OFF + h];
  let dt_s = max(softplus(dt[h] + aux[DT_BIAS + h * hd + d]), params.dt_min);
  let dA = exp(dt_s * Av);
  let xv = X[h * hd + d];
  let hOff = (h * hd + d) * N;
  let bOff = group * N;

  var local = 0.0;
  var n = tid;
  loop {
    if (n >= N) { break; }
    let hv = H[hOff + n] * dA + dt_s * B[bOff + n] * xv;
    H[hOff + n] = hv;
    local = local + C[bOff + n] * hv;
    n = n + WG;
  }
  red[tid] = local;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (tid == 0u) {
    Y[h * hd + d] = red[0] + aux[D_OFF + h * hd + d] * xv;
  }
}
