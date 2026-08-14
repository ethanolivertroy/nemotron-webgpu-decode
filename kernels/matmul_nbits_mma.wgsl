// Decode GEMV using chromium subgroup-matrix 8x8 f16 MMA.
// One subgroup (WG=32) writes 8 N-columns. A is padded to 8 rows (M=1).
// Expected to lose to sg4/sg32 on M=1; measured anyway.

enable subgroups;
enable f16;
enable chromium_experimental_subgroup_matrix;

struct Params {
  K: u32,
  N: u32,
  n_blocks: u32,
  epilogue: u32,
};

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zp: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 32u;
const N_COLS: u32 = 8u;

var<workgroup> tile_a: array<f16, 64>;
var<workgroup> tile_b: array<f16, 64>;
var<workgroup> tile_c: array<f16, 64>;

fn load_zp(col: u32, blk: u32) -> f32 {
  let nib = col * params.n_blocks + blk;
  return f32((zp[nib / 8u] >> ((nib % 8u) * 4u)) & 15u);
}

@compute @workgroup_size(WG, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let col0 = wg.x * N_COLS;
  let n = params.N;
  let n_blocks = params.n_blocks;

  var matC: subgroup_matrix_result<f16, 8, 8>;

  for (var blk = 0u; blk < n_blocks; blk++) {
    let a_base = blk * 8u;
    // One q4 block is 32 K-values = four 8-wide MMA K-steps.
    for (var step = 0u; step < 4u; step++) {
      let a_vec = A[a_base + step * 2u + tid / 16u];
      if (tid < 8u) {
        let lane = tid;
        let aval = f16(A[a_base + step * 2u + lane / 4u][lane % 4u]);
        for (var row = 0u; row < 8u; row++) {
          tile_a[row * 8u + lane] = aval;
        }
      }
      let col = col0 + (tid % 8u);
      let k_local = tid / 8u;
      if (col < n && k_local < 8u) {
        let packed = B[col * n_blocks + blk];
        let scale = scales[col * n_blocks + blk];
        let zero = load_zp(col, blk);
        let word = packed[step];
        let lo = unpack4xU8(word & 0x0F0F0F0Fu);
        let hi = unpack4xU8((word >> 4u) & 0x0F0F0F0Fu);
        let vals = array<f32, 8>(
          f32(lo.x), f32(hi.x), f32(lo.y), f32(hi.y),
          f32(lo.z), f32(hi.z), f32(lo.w), f32(hi.w),
        );
        tile_b[k_local * 8u + (tid % 8u)] = f16((vals[k_local] - zero) * scale);
      }
      workgroupBarrier();
      var matA: subgroup_matrix_left<f16, 8, 8> =
        subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_a, 0u, false, 8u);
      var matB: subgroup_matrix_right<f16, 8, 8> =
        subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_b, 0u, true, 8u);
      matC = subgroupMatrixMultiplyAccumulate(matA, matB, matC);
      workgroupBarrier();
    }
  }

  subgroupMatrixStore(&tile_c, 0u, matC, false, 8u);
  workgroupBarrier();
  if (tid < N_COLS) {
    let col = col0 + tid;
    if (col < n) {
      Y[col] = f32(tile_c[tid]);
    }
  }
}
