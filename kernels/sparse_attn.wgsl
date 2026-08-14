// kernel: decode GQA attention (4 sparse-attention layers)
// replaces: GroupQueryAttention / MultiHeadAttention softmax+score+AV
// target: QKV GEMM is MatMulNBits; this kernel is the remaining SDPA
//
// Design: one workgroup per query head. Threads cooperatively score a
// strided slice of keys, softmax via shared max/sum, then accumulate V.
// GQA maps query head h -> kv head h / (H/Hkv). Online softmax in f32
// (decode seqQ=1, so flash tiling along T is a simple loop). Workgroup
// 64. Causal mask: keys 0..seq-1 are all visible at decode. Scale is
// 1/sqrt(head_dim). No KV write here (ORT already fused cache update).

struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 64u;
const NEG_INF: f32 = -3.4028234663852886e38;

var<workgroup> red: array<f32, WG>;
var<workgroup> scores: array<f32, 256>; // seq <= 256 in the harness

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let tid = lid.x;
  if (h >= params.n_heads) {
    return;
  }
  let group = params.n_heads / params.n_kv_heads;
  let hkv = h / group;
  let hd = params.head_dim;
  let seq = params.seq;
  let scale = inverseSqrt(f32(hd));
  let qOff = h * hd;

  var t = tid;
  loop {
    if (t >= seq) { break; }
    var dot = 0.0;
    let kOff = (t * params.n_kv_heads + hkv) * hd;
    var d = 0u;
    loop {
      if (d >= hd) { break; }
      dot = dot + Q[qOff + d] * K[kOff + d];
      d = d + 1u;
    }
    scores[t] = dot * scale;
    t = t + WG;
  }
  workgroupBarrier();

  var thread_max = NEG_INF;
  t = tid;
  loop {
    if (t >= seq) { break; }
    thread_max = max(thread_max, scores[t]);
    t = t + WG;
  }
  red[tid] = thread_max;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = max(red[tid], red[tid + stride]); }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let row_max = red[0];
  workgroupBarrier();

  var thread_sum = 0.0;
  t = tid;
  loop {
    if (t >= seq) { break; }
    let p = exp(scores[t] - row_max);
    scores[t] = p;
    thread_sum = thread_sum + p;
    t = t + WG;
  }
  red[tid] = thread_sum;
  workgroupBarrier();
  stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv_sum = 1.0 / red[0];

  var d = tid;
  loop {
    if (d >= hd) { break; }
    var acc = 0.0;
    var tt = 0u;
    loop {
      if (tt >= seq) { break; }
      let vOff = (tt * params.n_kv_heads + hkv) * hd;
      acc = acc + scores[tt] * inv_sum * V[vOff + d];
      tt = tt + 1u;
    }
    O[qOff + d] = acc;
    d = d + WG;
  }
}
