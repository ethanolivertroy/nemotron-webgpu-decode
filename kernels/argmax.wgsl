// Two-level argmax: 256 workgroups each take a slice of N, then WG 0 merges.
// Binding 2 holds 256 candidate (value, index) pairs as f32/u32 overlay.

struct Params {
  n: u32,
  n_slice: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> partial: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> best_v: array<f32, WG>;
var<workgroup> best_i: array<u32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn partial_max(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let start = wg.x * params.n_slice;
  var bv = -3.402823466e38;
  var bi = start;
  var i = start + tid;
  let end = min(start + params.n_slice, params.n);
  loop {
    if (i >= end) { break; }
    let v = X[i];
    if (v > bv) {
      bv = v;
      bi = i;
    }
    i = i + WG;
  }
  best_v[tid] = bv;
  best_i[tid] = bi;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      if (best_v[tid + stride] > best_v[tid]) {
        best_v[tid] = best_v[tid + stride];
        best_i[tid] = best_i[tid + stride];
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (tid == 0u) {
    partial[wg.x * 2u] = bitcast<u32>(best_v[0]);
    partial[wg.x * 2u + 1u] = best_i[0];
  }
}

@compute @workgroup_size(WG, 1, 1)
fn merge_max(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var bv = -3.402823466e38;
  var bi = 0u;
  var s = tid;
  loop {
    if (s >= 256u) { break; }
    let v = bitcast<f32>(partial[s * 2u]);
    let i = partial[s * 2u + 1u];
    if (v > bv) {
      bv = v;
      bi = i;
    }
    s = s + WG;
  }
  best_v[tid] = bv;
  best_i[tid] = bi;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      if (best_v[tid + stride] > best_v[tid]) {
        best_v[tid] = best_v[tid + stride];
        best_i[tid] = best_i[tid + stride];
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (tid == 0u) {
    out_idx[0] = best_i[0];
  }
}
