@group(0) @binding(0) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(1) var<storage, read_write> partial: array<u32>;

const WG: u32 = 256u;
var<workgroup> best_v: array<f32, WG>;
var<workgroup> best_i: array<u32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
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
