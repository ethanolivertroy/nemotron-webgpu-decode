struct Params {
  seq: u32,
  n_kv: u32,
  hd: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params.n_kv * params.hd;
  let i = gid.x;
  if (i >= n) { return; }
  cache[params.seq * n + i] = src[i];
}
