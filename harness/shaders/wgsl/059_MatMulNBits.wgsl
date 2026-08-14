enable subgroups;
const workgroup_size_x: u32 = 128;
const workgroup_size_y: u32 = 1;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> input_b: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales_b: array<f32>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
struct Uniforms {
  M: u32,
  N: u32,
  K: u32,
  K_of_a: u32,
  K_of_b: u32,
  block_size: u32,
  blocks_per_col: u32,
  zero_blocks_per_col: u32,
  num_N_tile: u32,
  batch_count: u32,
  weight_idx: u32,
  dispatch_M: u32
};
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

alias input_a_value_t = vec4<f32>;
alias input_b_value_t = vec4<u32>;
alias input_b_indices_t = vec3<u32>;
alias scales_b_value_t = f32;
alias scales_b_indices_t = vec2<u32>;
alias output_element_t = f32;

  alias output_type = output_element_t;
  const default_zero_point = 8;
  const bit_mask = 0xFu;
  const elements_in_uint32:u32 = 32u / 4;
  fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> output_type
  {
    if (row < r_dim && col < c_dim) {
      let offset = row * c_dim + col;

      let array_index = offset / elements_in_uint32;
      let component_index = offset % elements_in_uint32;
      let packed_value = zero_points[array_index];

      let shift_amount = component_index * 4;

      let masked_value = (packed_value >> shift_amount) & bit_mask;
      return output_type(masked_value);
    }
    return output_type(0);
  }
var<workgroup> tile_A : array<input_a_value_t, 256>;
var<workgroup> inter_results: array<array<output_element_t, 32>, 8>;

fn loadSHMA(batch: u32, a_global: u32, kidx: u32, col: u32)
{
    let k_offset = kidx / 4 + col;
    if (batch < uniforms.batch_count && k_offset < uniforms.K_of_a) {
        tile_A[col] = input_a[batch * uniforms.M * uniforms.K_of_a + a_global * uniforms.K_of_a + k_offset];
    } else {
        tile_A[col] = input_a_value_t(0);
    }
}


@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32) {
  let global_idx = global_id.x;
  let workgroup_idx = workgroup_id.x;

  let batch = workgroup_idx / (uniforms.dispatch_M * uniforms.num_N_tile);
  let a_global = (workgroup_idx / uniforms.num_N_tile) % uniforms.dispatch_M;
  const b_base_offset : u32 = 0;
  const b_scale_offset : u32 = 0;
  const actual_weight_idx : u32 = 0;
  let b_global_base = (workgroup_idx % uniforms.num_N_tile) * 8;

  let a_row = a_global;
  let idx = local_idx % 32;
  let idy = local_idx / 32;

  for (var kidx = 0u; kidx < uniforms.K; kidx += 1024)
  {
    for (var id = local_idx; id < 256; id += workgroup_size_x)
    {
      loadSHMA(batch, a_row, kidx, id);
    }
    workgroupBarrier();

    for (var local_row_offset = 0u; local_row_offset < 8; local_row_offset += 4)
    {
      var b_global = b_global_base + local_row_offset + idy;
      var k_offset = kidx / 32 + idx;
      if (b_global < uniforms.N && k_offset < uniforms.K_of_b)
      {
        let block_idx = (kidx + idx * 32) / uniforms.block_size;
        let scale_b = scales_b[b_global * uniforms.blocks_per_col + block_idx + b_scale_offset];
        let zero = mm_read_zero(b_global, block_idx, uniforms.N, uniforms.zero_blocks_per_col);
        var b_value = input_b[b_global * uniforms.K_of_b + k_offset + b_base_offset];

        var sum = output_element_t(0);
        var a_offset = idx * (8 / 4) * 4;
        for (var i = 0u; i < 4; i++) {
            let b_value_lower = vec4<output_element_t>(unpack4xU8(b_value[i] & 0x0F0F0F0Fu)) - vec4<output_element_t>(zero);
            let b_value_upper = vec4<output_element_t>(unpack4xU8((b_value[i] >> 4) & 0x0F0F0F0Fu)) - vec4<output_element_t>(zero);
            let b0 = vec4<output_element_t>(b_value_lower[0], b_value_upper[0], b_value_lower[1], b_value_upper[1]) * scale_b;
            let b1 = vec4<output_element_t>(b_value_lower[2], b_value_upper[2], b_value_lower[3], b_value_upper[3]) * scale_b;
            sum += dot(tile_A[a_offset], b0) + dot(tile_A[a_offset + 1], b1);
            a_offset += 2;
        }
        inter_results[local_row_offset + idy][idx] += sum;
      }
    }
    workgroupBarrier();
  }

  if (batch >= uniforms.batch_count) {
    return;
  }

  if (local_idx < 8) {
    var output_value = output_element_t(0);
    for (var b = 0u; b < 32; b++) {
      output_value += inter_results[local_idx][b];
    }
    let b_global =  b_global_base + local_idx;
    let output_idx = batch * uniforms.dispatch_M * uniforms.N + a_global * uniforms.N + b_global;
    if (b_global < uniforms.N) {
      output[output_idx]=output_value;;
    }
  }

}
