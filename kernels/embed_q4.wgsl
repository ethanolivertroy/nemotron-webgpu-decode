// GatherBlockQuantized row: bits=4, block=32, gather_axis=0, quantize_axis=1.
// packed is [vocab, hidden/2] u8. zp is 4-bit packed 8 per u32.
// Token id lives in a storage buffer so decode can keep it on GPU.

struct Params {
  hidden: u32,
  n_blocks: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> zp: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> token_buf: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 64u;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let hidden = params.hidden;
  let n_blocks = params.n_blocks;
  let tok = token_buf[0];
  let row_u32 = hidden / 8u;
  var i = tid;
  loop {
    if (i >= hidden) { break; }
    let blk = i / 32u;
    let scale = scales[tok * n_blocks + blk];
    let nib = tok * n_blocks + blk;
    let zero = f32((zp[nib / 8u] >> ((nib % 8u) * 4u)) & 15u);
    let u32_idx = tok * row_u32 + i / 8u;
    let nibble = (i % 8u) * 4u;
    let q = f32((packed[u32_idx] >> nibble) & 15u);
    Y[i] = (q - zero) * scale;
    i = i + WG;
  }
}
