/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * four_bit_expert_probe: how fast one MoE layer's routed experts can run when the weights are 4 bits wide,
 * on each card, for the two 4-bit formats on the table:
 *
 *   e2m1   the model's own experts: sign + 3-bit float per weight, one power-of-two scale per 32 weights
 *   int4   the AutoRound pack: unsigned 4-bit integer per weight (value q - 8), one fp16 scale per 32 weights
 *
 * It answers one question before any engine code exists: against today's EXL3 kernels (0.255 ms on the CMP 170HX
 * and 0.099 ms on the RTX PRO 6000 for the routed experts of one layer at one row), what does a 4-bit kernel cost?
 *
 * Shapes are DeepSeek-V4.1-Flash's: hidden 5120, expert inner width 2304, six experts per row. For each row and
 * each of its experts: gate = W1 x, up = W3 x, h = silu(min(gate, L)) * clamp(up, -L, L) * route weight,
 * y += W2 h. Weights are synthetic (a hash of their position), so the probe needs no model file and the host can
 * recompute any expert exactly: every run checks a few results against a double-precision reference, and
 * --corrupt flips one stored word first, where the check must fail.
 *
 * Kernel design (the prototype of the engine's 4-bit linear):
 *   - tensor-core instruction m16n8k16 with the WEIGHTS as its 16x16 operand and the input vector as one of its
 *     eight columns, fp32 accumulation;
 *   - weights stored in the order the instruction's lanes consume them: one 16-byte load per lane covers 64
 *     inputs, and the 32 lanes of a warp read 512 consecutive bytes;
 *   - a nibble becomes an fp16 by bit placement only (e2m1: the three magnitude bits are the low exponent and top
 *     mantissa bits, zero and one half land on fp16's subnormals; int4: the nibble is the mantissa of 16 + q);
 *     the group scale, and for int4 the "- 24 * sum of the group's inputs" correction, are applied once per 32
 *     inputs in fp32;
 *   - one block owns 16 output rows of one job; its warps each take a slice of the input width (and, for gate
 *     and up, one of the two matrices), so no warp runs a long serial chain; the slices' partial sums meet in
 *     shared memory and are added in a fixed order. No atomics: the result does not depend on timing.
 *     (A first version gave each warp the whole input width: correct, but bound by that chain, 156 us on the RTX
 *     PRO 6000 and 269 us on the CMP 170HX at one row.)
 * Each job here is one (row, expert) pair with one input column; rows that share an expert are a later step.
 *
 *   four_bit_expert_probe --format e2m1|int4 [--experts N] [--steps N] [--corrupt]
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN_SIZE 5120
#define EXPERT_INNER_SIZE 2304
#define MATRICES_PER_EXPERT_INPUT_SIDE 2         /* gate and up share the input */
#define EXPERTS_PER_ROW 6
#define SWIGLU_LIMIT 10.0f

#define WEIGHTS_PER_SCALE_GROUP 32
#define MMA_ROWS 16                              /* output rows one warp owns (the instruction's M) */
#define MMA_INPUTS 16                            /* inputs one instruction consumes (its K) */
#define MMA_STEPS_PER_LOAD 4                     /* one 16-byte load per lane = four instructions = 64 inputs */
#define INPUTS_PER_LOAD (MMA_INPUTS * MMA_STEPS_PER_LOAD)
#define MMA_STEPS_PER_SCALE_GROUP (WEIGHTS_PER_SCALE_GROUP / MMA_INPUTS)
#define SCALE_GROUPS_PER_LOAD (INPUTS_PER_LOAD / WEIGHTS_PER_SCALE_GROUP)
#define LANES_PER_WARP 32
#define LANES_PER_LANE_GROUP 4                   /* lanes that share an instruction row pair */
#define GATE_UP_INPUT_SLICES 4                    /* warps that share the input width of one gate or up tile */
#define GATE_UP_WARPS (MATRICES_PER_EXPERT_INPUT_SIDE * GATE_UP_INPUT_SLICES)
#define DOWN_INPUT_SLICES 4                      /* warps that share the input width of one down tile */
#define FILL_THREADS_PER_BLOCK 256
#define NIBBLE_BITS 4
#define NIBBLE_MASK 15u
#define HALVES_PER_REGISTER 2
#define REGISTERS_PER_OPERAND_A 4
#define BITS_PER_HALF 16

#define FORMAT_E2M1 0
#define FORMAT_INT4 1
/* e2m1: placing the three magnitude bits at fp16 bits 9..11 yields the value times 2^-14 (see header). */
#define E2M1_MAGNITUDE_MASK 0x0E000E00u
#define E2M1_SIGN_MASK 0x80008000u
#define E2M1_MAGNITUDE_BIT 9
#define E2M1_SIGN_BIT 12                         /* shift that moves nibble bit 3 to fp16 bit 15 */
#define E2M1_PLACEMENT_EXPONENT 14
#define E8M0_BIAS 127
#define FP32_EXPONENT_SHIFT 23
/* int4: 0x4C00 is fp16 16.0; the nibble at bits 6..9 makes it 16 + q exactly. True weight = (16 + q) - 24. */
#define INT4_BASE_PATTERN 0x4C004C00u
#define INT4_MANTISSA_MASK 0x03C003C0u
#define INT4_MANTISSA_BIT 6
#define INT4_PLACEMENT_OFFSET 24.0f

#define DEFAULT_EXPERT_COUNT 128
#define DEFAULT_STEP_COUNT 200
#define WARM_UP_STEPS 10
#define MAX_ROWS 8
#define MAX_JOBS (MAX_ROWS * EXPERTS_PER_ROW)
#define GATE_MATRIX 0
#define UP_MATRIX 1
#define DOWN_MATRIX 2
#define MATRIX_KINDS 3
#define JOBS_CHECKED_PER_RUN 2
#define CORRUPTED_BYTES (1024u * 1024u)             /* --corrupt overwrites this much of one gate matrix */
#define CORRUPTED_BYTE_VALUE 0x5A
#define RELATIVE_TOLERANCE 2.0e-3
#define BYTES_PER_GB 1.0e9
#define BYTES_PER_MIB (1024.0 * 1024.0)
#define MICROSECONDS_PER_MILLISECOND 1000.0
#define MAX_CARDS 16
#define E2M1_SCALE_EXPONENT_BASE 118             /* synthetic scales 2^-9 .. 2^-6 */
#define E2M1_SCALE_EXPONENT_SPREAD 3u
#define INT4_SCALE_STEPS 7u                      /* synthetic scales 1/1024 .. 8/1024 */
#define INT4_SCALE_DIVISOR 1024.0f
#define INPUT_VALUE_STEPS 255u                   /* synthetic inputs -2 .. 2 in steps of 1/64 */
#define INPUT_VALUE_OFFSET 128.0f
#define INPUT_VALUE_DIVISOR 64.0f

static const int row_counts[] = {1, 2, 4, 6, 8};
#define ROW_COUNT_CASES (sizeof(row_counts) / sizeof(row_counts[0]))

struct expert_job {
    uint32_t expert_index;
    uint32_t row_index;
    float route_weight;
    uint32_t padding;
};

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t check_cuda_status = (call);                                                       \
        if (check_cuda_status != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,                      \
                    cudaGetErrorString(check_cuda_status));                                           \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

/* ---- synthetic data: the same functions on the host and on the card ---- */

__host__ __device__ inline uint32_t mix_bits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__host__ __device__ inline uint32_t matrix_seed(uint32_t expert_index, uint32_t matrix_kind)
{
    return mix_bits(expert_index * MATRIX_KINDS + matrix_kind + 1u);
}

__host__ __device__ inline uint32_t weight_nibble(uint32_t seed, uint32_t output_row, uint32_t input_column, uint32_t input_size)
{
    return mix_bits(seed + output_row * input_size + input_column) & NIBBLE_MASK;
}

__host__ __device__ inline uint32_t scale_code(uint32_t seed, uint32_t output_row, uint32_t group, uint32_t group_count)
{
    return mix_bits(~seed + output_row * group_count + group);
}

__host__ __device__ inline float input_value(uint32_t row_index, uint32_t column, uint32_t step)
{
    uint32_t code = mix_bits(0x1234567u + step * 977u + row_index * HIDDEN_SIZE + column) & INPUT_VALUE_STEPS;
    return ((float)code - INPUT_VALUE_OFFSET) / INPUT_VALUE_DIVISOR;
}

static double e2m1_value(uint32_t nibble)
{
    static const double magnitudes[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    double magnitude = magnitudes[nibble & 7u];
    return (nibble & 8u) ? -magnitude : magnitude;
}

static double host_weight(int weight_format, uint32_t seed, uint32_t output_row, uint32_t input_column, uint32_t input_size)
{
    uint32_t nibble = weight_nibble(seed, output_row, input_column, input_size);
    uint32_t group_count = input_size / WEIGHTS_PER_SCALE_GROUP;
    uint32_t code = scale_code(seed, output_row, input_column / WEIGHTS_PER_SCALE_GROUP, group_count);
    if (weight_format == FORMAT_E2M1) {
        int exponent = (int)(E2M1_SCALE_EXPONENT_BASE + (code & E2M1_SCALE_EXPONENT_SPREAD)) - E8M0_BIAS;
        return e2m1_value(nibble) * ldexp(1.0, exponent);
    }
    return ((double)nibble - 8.0) * (double)(1u + (code & INT4_SCALE_STEPS)) / (double)INT4_SCALE_DIVISOR;
}

/* ---- storage sizes ---- */

static size_t packed_words(size_t output_size, size_t input_size)
{
    return output_size * input_size / (2 * sizeof(uint32_t));          /* 4 bits per weight */
}

static size_t scale_entries(size_t output_size, size_t input_size)
{
    return output_size * (input_size / WEIGHTS_PER_SCALE_GROUP);
}

/* ---- fill kernels: write the synthetic weights in the order the lanes consume them ---- */

__global__ void fill_packed_kernel(uint32_t *packed, uint32_t first_expert, uint32_t matrix_kind_base, uint32_t matrices_per_expert,
                                   uint32_t output_size, uint32_t input_size, uint64_t word_count)
{
    uint64_t words_per_matrix = (uint64_t)output_size * input_size / (2 * sizeof(uint32_t));
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t word_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; word_index < word_count; word_index += thread_total) {
        uint64_t matrix_index = word_index / words_per_matrix;
        uint64_t word_in_matrix = word_index % words_per_matrix;
        uint32_t seed = matrix_seed(first_expert + (uint32_t)(matrix_index / matrices_per_expert),
                                    matrix_kind_base + (uint32_t)(matrix_index % matrices_per_expert));
        uint32_t step_in_load = (uint32_t)(word_in_matrix % MMA_STEPS_PER_LOAD);
        uint32_t lane = (uint32_t)(word_in_matrix / MMA_STEPS_PER_LOAD % LANES_PER_WARP);
        uint64_t load_index = word_in_matrix / (MMA_STEPS_PER_LOAD * LANES_PER_WARP);
        uint32_t loads_per_tile = input_size / INPUTS_PER_LOAD;
        uint32_t tile = (uint32_t)(load_index / loads_per_tile);
        uint32_t first_input = ((uint32_t)(load_index % loads_per_tile) * MMA_STEPS_PER_LOAD + step_in_load) * MMA_INPUTS;
        uint32_t lane_group = lane / LANES_PER_LANE_GROUP;
        uint32_t lane_in_group = lane % LANES_PER_LANE_GROUP;
        uint32_t word = 0;
        for (uint32_t operand_register = 0; operand_register < REGISTERS_PER_OPERAND_A; operand_register++) {
            uint32_t output_row = tile * MMA_ROWS + lane_group + (operand_register & 1u) * (MMA_ROWS / 2);
            for (uint32_t half = 0; half < HALVES_PER_REGISTER; half++) {
                uint32_t input_column = first_input + lane_in_group * 2 + half + (operand_register >> 1) * (MMA_INPUTS / 2);
                word |= weight_nibble(seed, output_row, input_column, input_size) << (operand_register * NIBBLE_BITS + half * BITS_PER_HALF);
            }
        }
        packed[word_index] = word;
    }
}

template <int weight_format, typename scale_type>
__global__ void fill_scales_kernel(scale_type *scales, uint32_t first_expert, uint32_t matrix_kind_base, uint32_t matrices_per_expert,
                                   uint32_t output_size, uint32_t input_size, uint64_t entry_count)
{
    uint32_t group_count = input_size / WEIGHTS_PER_SCALE_GROUP;
    uint64_t entries_per_matrix = (uint64_t)output_size * group_count;
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t entry_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; entry_index < entry_count; entry_index += thread_total) {
        uint64_t matrix_index = entry_index / entries_per_matrix;
        uint64_t entry_in_matrix = entry_index % entries_per_matrix;
        uint32_t seed = matrix_seed(first_expert + (uint32_t)(matrix_index / matrices_per_expert),
                                    matrix_kind_base + (uint32_t)(matrix_index % matrices_per_expert));
        /* stored as [tile][group][row in tile] */
        uint32_t row_in_tile = (uint32_t)(entry_in_matrix % MMA_ROWS);
        uint32_t group = (uint32_t)(entry_in_matrix / MMA_ROWS % group_count);
        uint32_t tile = (uint32_t)(entry_in_matrix / MMA_ROWS / group_count);
        uint32_t code = scale_code(seed, tile * MMA_ROWS + row_in_tile, group, group_count);
        if (weight_format == FORMAT_E2M1) {
            scales[entry_index] = (scale_type)(E2M1_SCALE_EXPONENT_BASE + (code & E2M1_SCALE_EXPONENT_SPREAD));
        } else {
            scales[entry_index] = (scale_type)__half_as_ushort(__float2half((float)(1u + (code & INT4_SCALE_STEPS)) / INT4_SCALE_DIVISOR));
        }
    }
}

__global__ void fill_input_kernel(__half *layer_input, float *input_group_sums, uint32_t row_count, uint32_t step)
{
    /* one block per row; the group sums are the int4 correction term (unused for e2m1) */
    uint32_t row_index = blockIdx.x;
    if (row_index >= row_count) {
        return;
    }
    for (uint32_t column = threadIdx.x; column < HIDDEN_SIZE; column += blockDim.x) {
        layer_input[(size_t)row_index * HIDDEN_SIZE + column] = __float2half(input_value(row_index, column, step));
    }
    for (uint32_t group = threadIdx.x; group < HIDDEN_SIZE / WEIGHTS_PER_SCALE_GROUP; group += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t offset = 0; offset < WEIGHTS_PER_SCALE_GROUP; offset++) {
            sum += input_value(row_index, group * WEIGHTS_PER_SCALE_GROUP + offset, step);
        }
        input_group_sums[(size_t)row_index * (HIDDEN_SIZE / WEIGHTS_PER_SCALE_GROUP) + group] = sum;
    }
}

/* ---- the kernels under test ---- */

__device__ __forceinline__ void tensor_core_m16n8k16(float accumulator[4], const uint32_t weights[REGISTERS_PER_OPERAND_A],
                                                     uint32_t input_low, uint32_t input_high)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(accumulator[0]), "=f"(accumulator[1]), "=f"(accumulator[2]), "=f"(accumulator[3])
                 : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(input_low), "r"(input_high),
                   "f"(accumulator[0]), "f"(accumulator[1]), "f"(accumulator[2]), "f"(accumulator[3]));
}

/* Eight nibbles of one packed word become the four fp16 pairs of the instruction's weight operand. */
template <int weight_format>
__device__ __forceinline__ void place_nibbles(uint32_t word, uint32_t weights[REGISTERS_PER_OPERAND_A])
{
    if (weight_format == FORMAT_E2M1) {
        weights[0] = ((word << E2M1_MAGNITUDE_BIT) & E2M1_MAGNITUDE_MASK) | ((word << E2M1_SIGN_BIT) & E2M1_SIGN_MASK);
        weights[1] = ((word << (E2M1_MAGNITUDE_BIT - NIBBLE_BITS)) & E2M1_MAGNITUDE_MASK) | ((word << (E2M1_SIGN_BIT - NIBBLE_BITS)) & E2M1_SIGN_MASK);
        weights[2] = ((word << (E2M1_MAGNITUDE_BIT - 2 * NIBBLE_BITS)) & E2M1_MAGNITUDE_MASK) | ((word << (E2M1_SIGN_BIT - 2 * NIBBLE_BITS)) & E2M1_SIGN_MASK);
        weights[3] = ((word >> (3 * NIBBLE_BITS - E2M1_MAGNITUDE_BIT)) & E2M1_MAGNITUDE_MASK) | (word & E2M1_SIGN_MASK);
    } else {
        weights[0] = ((word << INT4_MANTISSA_BIT) & INT4_MANTISSA_MASK) | INT4_BASE_PATTERN;
        weights[1] = ((word << (INT4_MANTISSA_BIT - NIBBLE_BITS)) & INT4_MANTISSA_MASK) | INT4_BASE_PATTERN;
        weights[2] = ((word >> (2 * NIBBLE_BITS - INT4_MANTISSA_BIT)) & INT4_MANTISSA_MASK) | INT4_BASE_PATTERN;
        weights[3] = ((word >> (3 * NIBBLE_BITS - INT4_MANTISSA_BIT)) & INT4_MANTISSA_MASK) | INT4_BASE_PATTERN;
    }
}

template <int weight_format, typename scale_type>
__device__ __forceinline__ float group_scale(scale_type stored)
{
    if (weight_format == FORMAT_E2M1) {
        return __uint_as_float(((uint32_t)stored + E2M1_PLACEMENT_EXPONENT) << FP32_EXPONENT_SHIFT);
    }
    return __half2float(__ushort_as_half((unsigned short)stored));
}

/*
 * One warp, one tile of 16 output rows, one slice of the input width (loads first_load .. first_load + load_count):
 * result[0] is output row (lane / 4) and result[1] is output row (lane / 4 + 8), both for the single input column;
 * they are valid in lanes with lane % 4 == 0.
 */
template <int weight_format, typename scale_type>
__device__ __forceinline__ void tile_slice_product(const uint4 *tile_packed, const scale_type *tile_scales, const __half *input,
                                                   const float *input_group_sums, uint32_t first_load, uint32_t load_count,
                                                   float result[2])
{
    uint32_t lane = threadIdx.x % LANES_PER_WARP;
    uint32_t lane_group = lane / LANES_PER_LANE_GROUP;
    uint32_t lane_in_group = lane % LANES_PER_LANE_GROUP;
    bool carries_input = lane_group == 0;                 /* the input vector is instruction column 0 */
    bool carries_result = lane_in_group == 0;             /* instruction column 0 of rows lane_group and lane_group + 8 */
    const uint32_t *input_pairs = (const uint32_t *)input;
    float row_low = 0.0f, row_high = 0.0f;
    for (uint32_t load_index = first_load; load_index < first_load + load_count; load_index++) {
        uint4 packed = tile_packed[(size_t)load_index * LANES_PER_WARP + lane];
        uint32_t words[MMA_STEPS_PER_LOAD] = {packed.x, packed.y, packed.z, packed.w};
        for (uint32_t group_in_load = 0; group_in_load < SCALE_GROUPS_PER_LOAD; group_in_load++) {
            float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t step_in_group = 0; step_in_group < MMA_STEPS_PER_SCALE_GROUP; step_in_group++) {
                uint32_t step = group_in_load * MMA_STEPS_PER_SCALE_GROUP + step_in_group;
                uint32_t first_input = (load_index * MMA_STEPS_PER_LOAD + step) * MMA_INPUTS;
                uint32_t weights[REGISTERS_PER_OPERAND_A];
                uint32_t input_low = 0, input_high = 0;
                place_nibbles<weight_format>(words[step], weights);
                if (carries_input) {
                    input_low = input_pairs[first_input / HALVES_PER_REGISTER + lane_in_group];
                    input_high = input_pairs[(first_input + MMA_INPUTS / 2) / HALVES_PER_REGISTER + lane_in_group];
                }
                tensor_core_m16n8k16(accumulator, weights, input_low, input_high);
            }
            if (carries_result) {
                uint32_t group = load_index * SCALE_GROUPS_PER_LOAD + group_in_load;
                const scale_type *group_scales = tile_scales + (size_t)group * MMA_ROWS;
                float low = accumulator[0], high = accumulator[2];
                if (weight_format == FORMAT_INT4) {
                    float correction = INT4_PLACEMENT_OFFSET * input_group_sums[group];
                    low -= correction;
                    high -= correction;
                }
                row_low += low * group_scale<weight_format, scale_type>(group_scales[lane_group]);
                row_high += high * group_scale<weight_format, scale_type>(group_scales[lane_group + MMA_ROWS / 2]);
            }
        }
    }
    result[0] = row_low;
    result[1] = row_high;
}

/* The slices of one tile meet here: one partial sum per (warp, output row), added by the row's thread in warp order. */
__device__ __forceinline__ void publish_partial(float partial_sums[][MMA_ROWS], const float result[2])
{
    uint32_t lane = threadIdx.x % LANES_PER_WARP;
    if (lane % LANES_PER_LANE_GROUP == 0) {
        uint32_t warp = threadIdx.x / LANES_PER_WARP;
        partial_sums[warp][lane / LANES_PER_LANE_GROUP] = result[0];
        partial_sums[warp][lane / LANES_PER_LANE_GROUP + MMA_ROWS / 2] = result[1];
    }
}

__device__ __forceinline__ float sum_slices_in_order(const float partial_sums[][MMA_ROWS], uint32_t first_warp, uint32_t slice_count, uint32_t row)
{
    float sum = 0.0f;
    for (uint32_t slice = 0; slice < slice_count; slice++) {
        sum += partial_sums[first_warp + slice][row];
    }
    return sum;
}

/*
 * gate and up of every job: grid (tiles of the inner width, jobs), eight warps per block: warps 0..3 take the four
 * input slices of the gate matrix, warps 4..7 those of the up matrix. Writes h as fp16.
 */
template <int weight_format, typename scale_type>
__global__ void gate_up_kernel(const uint4 *gate_up_packed, const scale_type *gate_up_scales, const __half *layer_input,
                               const float *input_group_sums, const struct expert_job *jobs, __half *expert_hidden)
{
    __shared__ float partial_sums[GATE_UP_WARPS][MMA_ROWS];
    struct expert_job job = jobs[blockIdx.y];
    uint32_t tile = blockIdx.x;
    uint32_t warp = threadIdx.x / LANES_PER_WARP;
    uint32_t matrix_kind = warp / GATE_UP_INPUT_SLICES;                 /* GATE_MATRIX or UP_MATRIX */
    uint32_t loads_per_slice = HIDDEN_SIZE / INPUTS_PER_LOAD / GATE_UP_INPUT_SLICES;
    size_t loads_per_tile = (size_t)(HIDDEN_SIZE / INPUTS_PER_LOAD) * LANES_PER_WARP;
    size_t scales_per_tile = (size_t)(HIDDEN_SIZE / WEIGHTS_PER_SCALE_GROUP) * MMA_ROWS;
    size_t tiles_per_matrix = EXPERT_INNER_SIZE / MMA_ROWS;
    size_t matrix = (size_t)job.expert_index * MATRICES_PER_EXPERT_INPUT_SIDE + matrix_kind;
    float result[2];
    tile_slice_product<weight_format, scale_type>(
        gate_up_packed + (matrix * tiles_per_matrix + tile) * loads_per_tile,
        gate_up_scales + (matrix * tiles_per_matrix + tile) * scales_per_tile,
        layer_input + (size_t)job.row_index * HIDDEN_SIZE,
        input_group_sums + (size_t)job.row_index * (HIDDEN_SIZE / WEIGHTS_PER_SCALE_GROUP),
        (warp % GATE_UP_INPUT_SLICES) * loads_per_slice, loads_per_slice, result);
    publish_partial(partial_sums, result);
    __syncthreads();
    if (threadIdx.x < MMA_ROWS) {
        uint32_t row = threadIdx.x;
        float gate_value = sum_slices_in_order(partial_sums, GATE_MATRIX * GATE_UP_INPUT_SLICES, GATE_UP_INPUT_SLICES, row);
        float up_value = sum_slices_in_order(partial_sums, UP_MATRIX * GATE_UP_INPUT_SLICES, GATE_UP_INPUT_SLICES, row);
        gate_value = fminf(gate_value, SWIGLU_LIMIT);
        up_value = fminf(fmaxf(up_value, -SWIGLU_LIMIT), SWIGLU_LIMIT);
        float silu = gate_value / (1.0f + expf(-gate_value));
        expert_hidden[(size_t)blockIdx.y * EXPERT_INNER_SIZE + tile * MMA_ROWS + row] = __float2half(silu * up_value * job.route_weight);
    }
}

/* the int4 correction term of the down projection: sums of h per group of 32, one block per job */
__global__ void hidden_group_sums_kernel(const __half *expert_hidden, float *hidden_group_sums)
{
    const __half *hidden = expert_hidden + (size_t)blockIdx.x * EXPERT_INNER_SIZE;
    for (uint32_t group = threadIdx.x; group < EXPERT_INNER_SIZE / WEIGHTS_PER_SCALE_GROUP; group += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t offset = 0; offset < WEIGHTS_PER_SCALE_GROUP; offset++) {
            sum += __half2float(hidden[group * WEIGHTS_PER_SCALE_GROUP + offset]);
        }
        hidden_group_sums[(size_t)blockIdx.x * (EXPERT_INNER_SIZE / WEIGHTS_PER_SCALE_GROUP) + group] = sum;
    }
}

/* down projection of every job: grid (tiles of the hidden width, jobs), four warps per block, one input slice each. */
template <int weight_format, typename scale_type>
__global__ void down_kernel(const uint4 *down_packed, const scale_type *down_scales, const __half *expert_hidden,
                            const float *hidden_group_sums, const struct expert_job *jobs, float *job_output)
{
    __shared__ float partial_sums[DOWN_INPUT_SLICES][MMA_ROWS];
    struct expert_job job = jobs[blockIdx.y];
    uint32_t tile = blockIdx.x;
    uint32_t warp = threadIdx.x / LANES_PER_WARP;
    uint32_t loads_per_slice = EXPERT_INNER_SIZE / INPUTS_PER_LOAD / DOWN_INPUT_SLICES;
    size_t loads_per_tile = (size_t)(EXPERT_INNER_SIZE / INPUTS_PER_LOAD) * LANES_PER_WARP;
    size_t scales_per_tile = (size_t)(EXPERT_INNER_SIZE / WEIGHTS_PER_SCALE_GROUP) * MMA_ROWS;
    size_t tiles_per_matrix = HIDDEN_SIZE / MMA_ROWS;
    float result[2];
    tile_slice_product<weight_format, scale_type>(
        down_packed + ((size_t)job.expert_index * tiles_per_matrix + tile) * loads_per_tile,
        down_scales + ((size_t)job.expert_index * tiles_per_matrix + tile) * scales_per_tile,
        expert_hidden + (size_t)blockIdx.y * EXPERT_INNER_SIZE,
        hidden_group_sums + (size_t)blockIdx.y * (EXPERT_INNER_SIZE / WEIGHTS_PER_SCALE_GROUP),
        warp * loads_per_slice, loads_per_slice, result);
    publish_partial(partial_sums, result);
    __syncthreads();
    if (threadIdx.x < MMA_ROWS) {
        job_output[(size_t)blockIdx.y * HIDDEN_SIZE + tile * MMA_ROWS + threadIdx.x] =
            sum_slices_in_order(partial_sums, 0, DOWN_INPUT_SLICES, threadIdx.x);
    }
}

/* the six experts of a row, added in slot order: grid (rows) */
__global__ void row_sum_kernel(const float *job_output, float *row_output)
{
    for (uint32_t column = threadIdx.x; column < HIDDEN_SIZE; column += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < EXPERTS_PER_ROW; slot++) {
            sum += job_output[((size_t)blockIdx.x * EXPERTS_PER_ROW + slot) * HIDDEN_SIZE + column];
        }
        row_output[(size_t)blockIdx.x * HIDDEN_SIZE + column] = sum;
    }
}

/* ---- host side ---- */

static float half_bits_to_float(uint16_t bits)
{
    int exponent = (bits >> 10) & 31;
    int mantissa = bits & 1023;
    double value = exponent == 0 ? ldexp((double)mantissa, -24) : ldexp((double)(mantissa + 1024), exponent - 25);
    return (float)((bits & 0x8000) ? -value : value);
}

static uint16_t float_to_half_bits(float value)
{
    __half half = __float2half(value);                    /* the host version rounds to nearest even, as the card does */
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

/* The exact result of one job, in double, with h rounded to fp16 the way the kernel stores it. */
static void host_reference(int weight_format, const struct expert_job *job, uint32_t step, double *reference_output)
{
    static double hidden[EXPERT_INNER_SIZE];
    static float input[HIDDEN_SIZE];
    for (uint32_t column = 0; column < HIDDEN_SIZE; column++) {
        input[column] = half_bits_to_float(float_to_half_bits(input_value(job->row_index, column, step)));
    }
    uint32_t gate_seed = matrix_seed(job->expert_index, GATE_MATRIX);
    uint32_t up_seed = matrix_seed(job->expert_index, UP_MATRIX);
    uint32_t down_seed = matrix_seed(job->expert_index, DOWN_MATRIX);
    for (uint32_t inner = 0; inner < EXPERT_INNER_SIZE; inner++) {
        double gate = 0.0, up = 0.0;
        for (uint32_t column = 0; column < HIDDEN_SIZE; column++) {
            gate += host_weight(weight_format, gate_seed, inner, column, HIDDEN_SIZE) * input[column];
            up += host_weight(weight_format, up_seed, inner, column, HIDDEN_SIZE) * input[column];
        }
        gate = fmin(gate, (double)SWIGLU_LIMIT);
        up = fmin(fmax(up, -(double)SWIGLU_LIMIT), (double)SWIGLU_LIMIT);
        double value = gate / (1.0 + exp(-gate)) * up * (double)job->route_weight;
        hidden[inner] = half_bits_to_float(float_to_half_bits((float)value));
    }
    for (uint32_t column = 0; column < HIDDEN_SIZE; column++) {
        double sum = 0.0;
        for (uint32_t inner = 0; inner < EXPERT_INNER_SIZE; inner++) {
            sum += host_weight(weight_format, down_seed, column, inner, EXPERT_INNER_SIZE) * hidden[inner];
        }
        reference_output[column] = sum;
    }
}

struct card_buffers {
    uint4 *gate_up_packed;
    uint4 *down_packed;
    void *gate_up_scales;
    void *down_scales;
    __half *layer_input;
    float *input_group_sums;
    struct expert_job *jobs;
    __half *expert_hidden;
    float *hidden_group_sums;
    float *job_output;
    float *row_output;
};

template <int weight_format, typename scale_type>
static void launch_layer(const struct card_buffers *buffers, uint32_t row_count)
{
    uint32_t job_count = row_count * EXPERTS_PER_ROW;
    dim3 gate_up_grid(EXPERT_INNER_SIZE / MMA_ROWS, job_count);
    dim3 down_grid(HIDDEN_SIZE / MMA_ROWS, job_count);
    gate_up_kernel<weight_format, scale_type><<<gate_up_grid, GATE_UP_WARPS * LANES_PER_WARP>>>(
        buffers->gate_up_packed, (const scale_type *)buffers->gate_up_scales, buffers->layer_input, buffers->input_group_sums,
        buffers->jobs, buffers->expert_hidden);
    if (weight_format == FORMAT_INT4) {
        hidden_group_sums_kernel<<<job_count, EXPERT_INNER_SIZE / WEIGHTS_PER_SCALE_GROUP>>>(buffers->expert_hidden, buffers->hidden_group_sums);
    }
    down_kernel<weight_format, scale_type><<<down_grid, DOWN_INPUT_SLICES * LANES_PER_WARP>>>(
        buffers->down_packed, (const scale_type *)buffers->down_scales, buffers->expert_hidden, buffers->hidden_group_sums,
        buffers->jobs, buffers->job_output);
    row_sum_kernel<<<row_count, FILL_THREADS_PER_BLOCK>>>(buffers->job_output, buffers->row_output);
}

static void choose_jobs(struct expert_job *jobs, uint32_t row_count, uint32_t expert_count, uint32_t step)
{
    for (uint32_t row = 0; row < row_count; row++) {
        for (uint32_t slot = 0; slot < EXPERTS_PER_ROW; slot++) {
            uint32_t expert = mix_bits(0xABCDEFu + step * 131u + row * 17u + slot) % expert_count;
            for (uint32_t earlier = 0; earlier < slot; earlier++) {            /* six distinct experts per row */
                if (jobs[row * EXPERTS_PER_ROW + earlier].expert_index == expert) {
                    expert = (expert + 1) % expert_count;
                    earlier = (uint32_t)-1;
                }
            }
            jobs[row * EXPERTS_PER_ROW + slot].expert_index = expert;
            jobs[row * EXPERTS_PER_ROW + slot].row_index = row;
            jobs[row * EXPERTS_PER_ROW + slot].route_weight = (float)(slot + 1) / (float)(EXPERTS_PER_ROW * (EXPERTS_PER_ROW + 1) / 2);
            jobs[row * EXPERTS_PER_ROW + slot].padding = 0;
        }
    }
}

template <int weight_format, typename scale_type>
static int probe_card(int card_index, uint32_t expert_count, uint32_t step_count, int corrupt_one_word)
{
    cudaDeviceProp properties;
    struct card_buffers buffers;
    struct expert_job host_jobs[MAX_JOBS];
    size_t gate_up_words = packed_words(EXPERT_INNER_SIZE, HIDDEN_SIZE) * MATRICES_PER_EXPERT_INPUT_SIDE * expert_count;
    size_t down_words = packed_words(HIDDEN_SIZE, EXPERT_INNER_SIZE) * expert_count;
    size_t gate_up_scale_count = scale_entries(EXPERT_INNER_SIZE, HIDDEN_SIZE) * MATRICES_PER_EXPERT_INPUT_SIDE * expert_count;
    size_t down_scale_count = scale_entries(HIDDEN_SIZE, EXPERT_INNER_SIZE) * expert_count;
    size_t bytes_per_expert = (gate_up_words + down_words) * sizeof(uint32_t) / expert_count
                              + (gate_up_scale_count + down_scale_count) * sizeof(scale_type) / expert_count;
    cudaEvent_t started, stopped;
    int failures = 0;

    CHECK_CUDA(cudaSetDevice(card_index));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, card_index));
    CHECK_CUDA(cudaMalloc(&buffers.gate_up_packed, gate_up_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&buffers.down_packed, down_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&buffers.gate_up_scales, gate_up_scale_count * sizeof(scale_type)));
    CHECK_CUDA(cudaMalloc(&buffers.down_scales, down_scale_count * sizeof(scale_type)));
    CHECK_CUDA(cudaMalloc(&buffers.layer_input, (size_t)MAX_ROWS * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.input_group_sums, (size_t)MAX_ROWS * (HIDDEN_SIZE / WEIGHTS_PER_SCALE_GROUP) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.jobs, sizeof(host_jobs)));
    CHECK_CUDA(cudaMalloc(&buffers.expert_hidden, (size_t)MAX_JOBS * EXPERT_INNER_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.hidden_group_sums, (size_t)MAX_JOBS * (EXPERT_INNER_SIZE / WEIGHTS_PER_SCALE_GROUP) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.job_output, (size_t)MAX_JOBS * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.row_output, (size_t)MAX_ROWS * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaEventCreate(&started));
    CHECK_CUDA(cudaEventCreate(&stopped));

    unsigned int fill_blocks = (unsigned int)properties.multiProcessorCount * 8;
    fill_packed_kernel<<<fill_blocks, FILL_THREADS_PER_BLOCK>>>((uint32_t *)buffers.gate_up_packed, 0, GATE_MATRIX, MATRICES_PER_EXPERT_INPUT_SIDE,
                                                           EXPERT_INNER_SIZE, HIDDEN_SIZE, gate_up_words);
    fill_packed_kernel<<<fill_blocks, FILL_THREADS_PER_BLOCK>>>((uint32_t *)buffers.down_packed, 0, DOWN_MATRIX, 1, HIDDEN_SIZE, EXPERT_INNER_SIZE, down_words);
    fill_scales_kernel<weight_format, scale_type><<<fill_blocks, FILL_THREADS_PER_BLOCK>>>(
        (scale_type *)buffers.gate_up_scales, 0, GATE_MATRIX, MATRICES_PER_EXPERT_INPUT_SIDE, EXPERT_INNER_SIZE, HIDDEN_SIZE, gate_up_scale_count);
    fill_scales_kernel<weight_format, scale_type><<<fill_blocks, FILL_THREADS_PER_BLOCK>>>(
        (scale_type *)buffers.down_scales, 0, DOWN_MATRIX, 1, HIDDEN_SIZE, EXPERT_INNER_SIZE, down_scale_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("card %d %s (sm_%d%d, %d SMs): %s, %u experts of %.2f MiB in memory\n", card_index, properties.name, properties.major,
           properties.minor, properties.multiProcessorCount, weight_format == FORMAT_E2M1 ? "e2m1" : "int4", expert_count,
           (double)bytes_per_expert / BYTES_PER_MIB);

    for (size_t case_index = 0; case_index < ROW_COUNT_CASES; case_index++) {
        uint32_t row_count = (uint32_t)row_counts[case_index];
        uint32_t job_count = row_count * EXPERTS_PER_ROW;
        double total_milliseconds = 0.0;
        for (uint32_t step = 0; step < step_count + WARM_UP_STEPS; step++) {
            choose_jobs(host_jobs, row_count, expert_count, step);
            CHECK_CUDA(cudaMemcpy(buffers.jobs, host_jobs, job_count * sizeof(struct expert_job), cudaMemcpyHostToDevice));
            fill_input_kernel<<<row_count, FILL_THREADS_PER_BLOCK>>>(buffers.layer_input, buffers.input_group_sums, row_count, step);
            if (step == 0 && corrupt_one_word) {
                /* overwrite the start of the first checked job's gate matrix */
                size_t word_offset = (size_t)host_jobs[0].expert_index * MATRICES_PER_EXPERT_INPUT_SIDE * packed_words(EXPERT_INNER_SIZE, HIDDEN_SIZE);
                CHECK_CUDA(cudaMemset((uint32_t *)buffers.gate_up_packed + word_offset, CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES));
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(started));
            launch_layer<weight_format, scale_type>(&buffers, row_count);
            CHECK_CUDA(cudaEventRecord(stopped));
            CHECK_CUDA(cudaEventSynchronize(stopped));
            CHECK_CUDA(cudaGetLastError());
            if (step >= WARM_UP_STEPS) {
                float milliseconds = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&milliseconds, started, stopped));
                total_milliseconds += (double)milliseconds;
            }
            if (step == 0) {
                /* check the first and the last job of the first step against the host */
                static float card_output[HIDDEN_SIZE];
                static double reference_output[HIDDEN_SIZE];
                for (uint32_t checked = 0; checked < JOBS_CHECKED_PER_RUN; checked++) {
                    uint32_t job_index = checked == 0 ? 0 : job_count - 1;
                    double squared_sum = 0.0, worst_error = 0.0;
                    CHECK_CUDA(cudaMemcpy(card_output, buffers.job_output + (size_t)job_index * HIDDEN_SIZE, sizeof(card_output), cudaMemcpyDeviceToHost));
                    host_reference(weight_format, &host_jobs[job_index], step, reference_output);
                    for (uint32_t column = 0; column < HIDDEN_SIZE; column++) {
                        squared_sum += reference_output[column] * reference_output[column];
                        worst_error = fmax(worst_error, fabs((double)card_output[column] - reference_output[column]));
                    }
                    double root_mean_square = sqrt(squared_sum / HIDDEN_SIZE);
                    int within_tolerance = worst_error <= RELATIVE_TOLERANCE * root_mean_square;
                    printf("  %u row(s), job %u (expert %u): worst error %.3e against rms %.3e: %s\n", row_count, job_index,
                           host_jobs[job_index].expert_index, worst_error, root_mean_square, within_tolerance ? "matches the reference" : "DIFFERS");
                    if (within_tolerance == (corrupt_one_word && checked == 0)) {
                        failures++;                              /* the corrupted job must differ, every other job must match */
                    }
                }
            }
        }
        if (corrupt_one_word) {
            break;                                               /* the corruption stays in memory: one case shows it */
        }
        double milliseconds_per_layer = total_milliseconds / step_count;
        double bytes_read = (double)job_count * (double)bytes_per_expert;
        printf("  %u row(s): %8.1f us per layer for the routed experts, %6.1f MB read, %6.0f GB/s\n", row_count,
               milliseconds_per_layer * MICROSECONDS_PER_MILLISECOND, bytes_read / 1.0e6,
               bytes_read / BYTES_PER_GB / (milliseconds_per_layer / 1000.0));
    }
    CHECK_CUDA(cudaEventDestroy(started));
    CHECK_CUDA(cudaEventDestroy(stopped));
    CHECK_CUDA(cudaFree(buffers.row_output));
    CHECK_CUDA(cudaFree(buffers.job_output));
    CHECK_CUDA(cudaFree(buffers.hidden_group_sums));
    CHECK_CUDA(cudaFree(buffers.expert_hidden));
    CHECK_CUDA(cudaFree(buffers.jobs));
    CHECK_CUDA(cudaFree(buffers.input_group_sums));
    CHECK_CUDA(cudaFree(buffers.layer_input));
    CHECK_CUDA(cudaFree(buffers.down_scales));
    CHECK_CUDA(cudaFree(buffers.gate_up_scales));
    CHECK_CUDA(cudaFree(buffers.down_packed));
    CHECK_CUDA(cudaFree(buffers.gate_up_packed));
    return failures;
}

int main(int argument_count, char **arguments)
{
    int weight_format = -1;
    long expert_count = DEFAULT_EXPERT_COUNT;
    long step_count = DEFAULT_STEP_COUNT;
    int corrupt_one_word = 0;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        const char *argument = arguments[argument_index];
        int has_value = argument_index + 1 < argument_count;
        if (strcmp(argument, "--corrupt") == 0) {
            corrupt_one_word = 1;
        } else if (strcmp(argument, "--format") == 0 && has_value) {
            const char *name = arguments[++argument_index];
            weight_format = strcmp(name, "e2m1") == 0 ? FORMAT_E2M1 : strcmp(name, "int4") == 0 ? FORMAT_INT4 : -1;
        } else if (strcmp(argument, "--experts") == 0 && has_value) {
            expert_count = strtol(arguments[++argument_index], NULL, 10);
        } else if (strcmp(argument, "--steps") == 0 && has_value) {
            step_count = strtol(arguments[++argument_index], NULL, 10);
        } else {
            weight_format = -1;
            break;
        }
    }
    if (weight_format < 0 || expert_count < EXPERTS_PER_ROW || expert_count > 4096 || step_count < 1 || step_count > 1000000) {
        fprintf(stderr, "usage: %s --format e2m1|int4 [--experts N] [--steps N] [--corrupt]\n", arguments[0]);
        return EXIT_FAILURE;
    }
    int card_count = 0;
    int failures = 0;
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (card_count > MAX_CARDS) {
        card_count = MAX_CARDS;
    }
    for (int card_index = 0; card_index < card_count; card_index++) {
        if (weight_format == FORMAT_E2M1) {
            failures += probe_card<FORMAT_E2M1, uint8_t>(card_index, (uint32_t)expert_count, (uint32_t)step_count, corrupt_one_word);
        } else {
            failures += probe_card<FORMAT_INT4, uint16_t>(card_index, (uint32_t)expert_count, (uint32_t)step_count, corrupt_one_word);
        }
    }
    if (failures != 0) {
        fprintf(stderr, "%d check(s) did not come out as required\n", failures);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
