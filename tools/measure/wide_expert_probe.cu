/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * wide_expert_probe: what the routed experts of one layer cost at a prefill chunk (4,096 rows, 24,576 row-expert pairs over
 * 384 experts) when the weights stay the checkpoint's own 4-bit e2m1. Today's engine (trellis weights: it reconstructs every
 * expert, then runs batched products) needs 48.0 ms per layer on the CMP 170HX and 12.7 ms on the RTX PRO 6000. Three forms:
 *   thin    the decode kernel's arithmetic for many rows: every row keeps the bits of its one-row call (sampled pairs are
 *           launched again one row at a time and compared bit for bit);
 *   wide    the same operands in the order that suits a prefill; checked against the reference only;
 *   cublas  the bar: cublasGemmEx (fp16 in, fp32 out) on weights decoded to fp16 beforehand (25 GiB on the card) and inputs
 *           dequantized to fp16, one call per expert and matrix: the tensor cores with no format work.
 * linear_kernel's comment has the two kernels. Arithmetic, as DeepSeek's reference (inference/model.py linear(), kernel.py
 * act_quant_kernel and fp4_gemm_kernel): the input of every product is quantized first, per block of 32: amax (at least
 * 1e-4), a scale 2^ceil(log2(amax / 448)) from the fp32 product, value / scale clamped to +-448 and rounded to E4M3 (nearest,
 * ties to even). A block's product uses the E4M3 values and the e2m1 weights, accumulates in fp32 and is multiplied once by
 * the two scales. gate = W1 x, up = W3 x, h = silu(min(gate, L)) * clamp(up, -L, L) * route weight in fp32, quantized again,
 * pair output = W2 h; a row's six pair outputs are added in slot order. Stages, all on the card: rows grouped by expert with
 * a counting sort in a fixed order, input quantization, gate and up, swiglu with hidden quantization, down, row sum.
 * Data is synthetic (a hash of its position). Every run checks sampled rows stage by stage: quantized bytes and exponents
 * against a host quantizer that searches the nearest code (exact), products against a double-precision reference fed with
 * the card's own quantized input (tolerance), row sums (exact). --corrupt overwrites a megabyte of one expert's gate matrix
 * first, where the gate and up check of every sampled pair routed to that expert must fail.
 *   wide_expert_probe [--form thin|wide|cublas|all] [--rows N] [--steps N] [--corrupt]      (link with -lcublas)
 * Measurement tool: not part of the engine.
 */
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    /* the model; a matrix, a form, a stage and a layout of fill_kernel by number */
    HIDDEN_SIZE = 5120, EXPERT_INNER_SIZE = 2304, EXPERT_COUNT = 384, EXPERTS_PER_ROW = 6, WEIGHTS_PER_SCALE_GROUP = 32,
    ROUTE_WEIGHT_DIVISOR = EXPERTS_PER_ROW * (EXPERTS_PER_ROW + 1) / 2, GATE_MATRIX = 0, UP_MATRIX, DOWN_MATRIX, MATRIX_KINDS,
    GATE_UP_MATRICES = 2, ONE_MATRIX = 1, GATE_UP_SIDE = 0, DOWN_SIDE, MATRIX_SIDES,     /* gate and up share the input and a buffer */
    FORM_THIN = 0, FORM_WIDE, FORM_CUBLAS, FORM_COUNT, LAYOUT_PACKED = 0, LAYOUT_SCALES, LAYOUT_DECODED, LAYOUT_INPUT,
    STAGE_GROUP_ROWS = 0, STAGE_INPUT_FORMAT, STAGE_GATE_UP, STAGE_HIDDEN_FORMAT, STAGE_DOWN, STAGE_ROW_SUM, STAGE_COUNT,
    /* m16n8k16: 16 output rows (M), 16 inputs (K), 8 rows of the chunk (N); a lane holds four registers of operand A, two
     * of B and four sums (two output rows by two columns); a 16-byte load per lane is four instructions = two scale groups */
    MMA_ROWS = 16, MMA_HALF_ROWS = MMA_ROWS / 2, MMA_INPUTS = 16, MMA_HALF_INPUTS = MMA_INPUTS / 2, MMA_COLUMNS = 8,
    MMA_STEPS_PER_SCALE_GROUP = WEIGHTS_PER_SCALE_GROUP / MMA_INPUTS, SCALE_GROUPS_PER_LOAD = 2,
    MMA_STEPS_PER_LOAD = MMA_STEPS_PER_SCALE_GROUP * SCALE_GROUPS_PER_LOAD, INPUTS_PER_LOAD = MMA_INPUTS * MMA_STEPS_PER_LOAD,
    LANES_PER_WARP = 32, LANES_PER_LANE_GROUP = 4, HALVES_PER_REGISTER = 2, REGISTERS_PER_OPERAND_A = 4, REGISTERS_PER_OPERAND_B = 2,
    ACCUMULATORS_PER_LANE = 4, BATCH_COLUMNS_PER_LANE = 2, WORDS_PER_ACTIVATION_GROUP = LANES_PER_LANE_GROUP * MMA_STEPS_PER_SCALE_GROUP,
    NIBBLE_BITS = 4, NIBBLE_MASK = 15, NIBBLES_PER_WORD = 8, BITS_PER_HALF = 16, BITS_PER_BYTE = 8, BYTES_PER_WORD = 4, BYTE_MASK = 0xFF,
    /* the kernels: column groups (instructions side by side) of a block; warps that share the input width of a thin tile;
     * a wide block's warps; scale groups of inputs a wide block stages; the extent of a shared array a form does not use */
    CHUNK_COLUMN_GROUPS = 8, CHUNK_ROWS = CHUNK_COLUMN_GROUPS * MMA_COLUMNS, THIN_SLICES = 4, WIDE_WARPS = CHUNK_COLUMN_GROUPS,
    STAGED_GROUPS = 4, UNUSED_EXTENT = 1, LARGEST_GROUP_COUNT = HIDDEN_SIZE / WEIGHTS_PER_SCALE_GROUP,
    INNER_GROUP_COUNT = EXPERT_INNER_SIZE / WEIGHTS_PER_SCALE_GROUP, LARGEST_SCALE_WORDS = LARGEST_GROUP_COUNT / BYTES_PER_WORD,
    ELEMENTS_PER_BLOCK = CHUNK_COLUMN_GROUPS * LANES_PER_WARP * ACCUMULATORS_PER_LANE, HELPER_THREADS_PER_BLOCK = 256, FILL_BLOCKS_PER_SM = 8,
    MAX_THREADS_PER_BLOCK = 1024,
    /* e2m1: the three magnitude bits at fp16 bits 9..11 are the value times 2^-14 (0 and 0.5 are subnormals) */
    E2M1_MAGNITUDE_BIT = 9, E2M1_SIGN_BIT = 12, E2M1_PLACEMENT_EXPONENT = 14, E2M1_SIGN_NIBBLE = 8, E2M1_MAGNITUDE_NIBBLE = 7, E2M1_MAGNITUDES = 8,
    /* E4M3: sign, four exponent bits (bias 7), three mantissa bits; exponent 0 is subnormal; 0x7E is 448, the largest.
     * Bytes 0 and 2 of a stored word become the halves of one register ("even"), 1 and 3 of the next: the value times 2^-8 */
    E4M3_SIGN_BIT = 0x80, E4M3_EXPONENT_SHIFT = 3, E4M3_EXPONENT_MASK = 15, E4M3_MANTISSA_MASK = 7, E4M3_IMPLICIT_ONE = 8,
    E4M3_LARGEST_CODE = 0x7E, E4M3_PLACEMENT_EXPONENT = 8, E4M3_EVEN_SIGN_SHIFT = 8, E4M3_EVEN_MAGNITUDE_SHIFT = 7, E4M3_ODD_MAGNITUDE_SHIFT = 1,
    /* fp32 to E4M3 by rounding the mantissa to three bits in place (the bias is just under half of the dropped bits, the
     * lowest kept bit adds the rest, the carry runs into the exponent). A group's sum is multiplied by 2^(weight code +
     * activation code - 232): the last name is the offset of that power's fp32 exponent field, which must stay in 1 .. 254
     * (the probe's codes give 118 .. 141; the engine must check the pack's) */
    FP32_DROPPED_BITS = 20, FP32_ROUNDING_BIAS = 0x7FFFF, FP32_TO_E4M3_CODE_OFFSET = (127 - 7) << E4M3_EXPONENT_SHIFT,
    FP32_EXPONENT_SHIFT = 23, FP32_EXPONENT_MASK = 0xFF, FP32_MANTISSA_MASK = 0x7FFFFF, FP32_SIGN_SHIFT = 31, E8M0_BIAS = 127,
    E8M0_INVERSE_BASE = 2 * E8M0_BIAS, COMBINED_EXPONENT_OFFSET = E8M0_BIAS - E2M1_PLACEMENT_EXPONENT - E4M3_PLACEMENT_EXPONENT,
    /* synthetic data: weight scales 2^-9 .. 2^-6; inputs -2 .. 2 in steps of 1/64, divided by 1, 2, 4 or 8 per block of 32 */
    WEIGHT_SCALE_EXPONENT_BASE = 118, WEIGHT_SCALE_EXPONENT_SPREAD = 3, INPUT_VALUE_STEPS = 255, INPUT_SHIFT_SPREAD = 3,
    INPUT_SEED = 0x1234567, INPUT_SHIFT_SEED = 0x7654321, ROUTING_SEED = 0xABCDEF, ROUTING_ROW_STRIDE = 17,
    /* the run: rows that are checked, with their six pairs each; output elements compared per product of a checked pair,
     * half of them among the first 92 output rows, inside the megabyte --corrupt overwrites (packed or decoded) */
    DEFAULT_ROW_COUNT = 4096, MAX_ROW_COUNT = 16384, DEFAULT_STEP_COUNT = 20, MAX_STEP_COUNT = 100000, WARM_UP_STEPS = 2,
    DECIMAL_BASE = 10, SAMPLED_ROWS = 4, SAMPLED_PAIRS = SAMPLED_ROWS * EXPERTS_PER_ROW, SAMPLED_OUTPUTS = 16,
    LEADING_SAMPLES = SAMPLED_OUTPUTS / 2, LEADING_SAMPLE_STRIDE = 13, CORRUPTED_BYTES = 1024 * 1024, CORRUPTED_BYTE_VALUE = 0x5A,
    STALE_BYTE_VALUE = 0xFF, STALE_BYTE_ONE_ROW = 0xEE, SM_80_MAJOR = 8, SM_120_MAJOR = 12, MAX_CARD_ALLOCATIONS = 32, MAX_CARDS = 16
};
static_assert(EXPERT_INNER_SIZE % (MMA_ROWS * WIDE_WARPS) == 0 && HIDDEN_SIZE % (MMA_ROWS * WIDE_WARPS) == 0
              && INNER_GROUP_COUNT % (THIN_SLICES * SCALE_GROUPS_PER_LOAD) == 0 && INNER_GROUP_COUNT % STAGED_GROUPS == 0
              && LARGEST_GROUP_COUNT % (THIN_SLICES * SCALE_GROUPS_PER_LOAD) == 0 && LARGEST_GROUP_COUNT % STAGED_GROUPS == 0
              && STAGED_GROUPS % SCALE_GROUPS_PER_LOAD == 0 && INNER_GROUP_COUNT % BYTES_PER_WORD == 0
              && EXPERT_COUNT <= MAX_THREADS_PER_BLOCK && CHUNK_COLUMN_GROUPS == 8,
              "the shapes do not divide into the tiles, slices and staged groups the kernels assume");
#define WEIGHTS_PER_MATRIX ((size_t)EXPERT_INNER_SIZE * HIDDEN_SIZE)     /* of gate, of up and of down alike */
#define SWIGLU_LIMIT 10.0f
#define E2M1_MAGNITUDE_MASK 0x0E000E00u
#define E2M1_SIGN_MASK 0x80008000u
#define E4M3_EVEN_SIGN_MASK 0x00800080u
#define E4M3_EVEN_MAGNITUDE_MASK 0x007F007Fu
#define E4M3_ODD_SIGN_MASK 0x80008000u
#define E4M3_ODD_MAGNITUDE_MASK 0x7F007F00u
#define E4M3_VALUE_DIVISOR 1024.0                /* 2^(bias + mantissa bits) */
#define E4M3_LARGEST_VALUE 448.0f
#define E4M3_SMALLEST_NORMAL 0.015625f           /* 2^-6 */
#define E4M3_SUBNORMAL_STEPS_PER_UNIT 512.0f     /* a subnormal is code * 2^-9 */
#define ACTIVATION_AMAX_FLOOR 1.0e-4f
#define ACTIVATION_SCALE_INVERSE (1.0f / E4M3_LARGEST_VALUE)
#define FREXP_SMALLEST_MANTISSA 0.5f             /* frexpf returns it for a power of two */
#define INPUT_VALUE_OFFSET 128.0f
#define INPUT_VALUE_DIVISOR 64.0f
#define RELATIVE_TOLERANCE 2.0e-3
#define TODAY_LAYER_MILLISECONDS_SM_80 48.0      /* today's engine, routed experts of one layer at 4,096 rows */
#define TODAY_LAYER_MILLISECONDS_SM_120 12.7
#define FLOPS_PER_WEIGHT 2.0
#define FLOPS_PER_TFLOP 1.0e12
#define MILLISECONDS_PER_SECOND 1000.0
#define BYTES_PER_MIB (1024.0 * 1024.0)
#define BYTES_PER_GIB (1024.0 * BYTES_PER_MIB)

struct row_chunk { uint32_t expert_index, first_position, row_count, padding; };      /* up to CHUNK_ROWS positions */

/* One product y = W x for every chunk; a position is a pair's place in the order grouped by expert. */
struct linear_arguments {
    const uint4 *packed;                         /* the matrix of expert 0: [tile][load][lane], 16 bytes */
    const uint8_t *weight_scales;                /* [tile][scale group][row in tile]: one E8M0 code */
    size_t packed_stride, scale_stride;          /* from one expert's matrix to the next */
    const uint32_t *activation_words;            /* [activation row][scale group][lane in group][step]: E4M3 bytes */
    const uint8_t *activation_scales;            /* [activation row][scale group]: one E8M0 code */
    const uint32_t *sorted_pairs;                /* [position]: pair = row * EXPERTS_PER_ROW + slot */
    const struct row_chunk *chunks;              /* [grid.y] */
    float *output;                               /* [position][output size] */
    uint32_t input_size, output_size;
    uint32_t activation_is_layer_input;          /* 1: the activation row is the pair's row; 0: it is the position */
};
static void check_status(int code, const char *call, const char *text, int line) {            /* 0 is success for both libraries */
    if (code != 0) {
        fprintf(stderr, "%s:%d: %s failed: %s (%d)\n", __FILE__, line, call, text, code);
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(call) do { cudaError_t status_ = (call); check_status((int)status_, #call, cudaGetErrorString(status_), __LINE__); } while (0)
#define CHECK_CUBLAS(call) check_status((int)(call), #call, "cuBLAS status", __LINE__)
static void read_card(void *host, const void *on_card, size_t bytes) { CHECK_CUDA(cudaMemcpy(host, on_card, bytes, cudaMemcpyDeviceToHost)); }
static double seconds_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/* ---- synthetic data and the layouts: the same functions on the host and on the card ---- */
__host__ __device__ inline uint32_t mix_bits(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}
__host__ __device__ inline uint32_t matrix_seed(uint32_t expert, uint32_t matrix_kind) { return mix_bits(expert * MATRIX_KINDS + matrix_kind + 1u); }
__host__ __device__ inline uint32_t weight_nibble(uint32_t seed, uint32_t output_row, uint32_t input_column, uint32_t input_size) {
    return mix_bits(seed + output_row * input_size + input_column) & NIBBLE_MASK;
}
/* The E8M0 code of the scale of 32 consecutive weights of one output row. */
__host__ __device__ inline uint32_t weight_scale_exponent(uint32_t seed, uint32_t output_row, uint32_t group, uint32_t group_count) {
    return WEIGHT_SCALE_EXPONENT_BASE + (mix_bits(~seed + output_row * group_count + group) & WEIGHT_SCALE_EXPONENT_SPREAD);
}
/* Exact in fp16. The divisor changes from block to block, so that the rows of one instruction differ in their scales. */
__host__ __device__ inline float input_value(uint32_t row_index, uint32_t column) {
    uint32_t code = mix_bits(INPUT_SEED + row_index * HIDDEN_SIZE + column) & INPUT_VALUE_STEPS;
    uint32_t shift = mix_bits(INPUT_SHIFT_SEED + row_index * LARGEST_GROUP_COUNT + column / WEIGHTS_PER_SCALE_GROUP) & INPUT_SHIFT_SPREAD;
    return ((float)code - INPUT_VALUE_OFFSET) / (INPUT_VALUE_DIVISOR * (float)(1u << shift));
}
__host__ __device__ inline float route_weight(uint32_t slot) { return (float)(slot + 1u) / (float)ROUTE_WEIGHT_DIVISOR; }
/*
 * The lane layout. For m16n8k16 the PTX documentation gives lane L (group g = L / 4, position t = L % 4) these elements
 * of the 16x16 operand A, as four registers of two fp16 halves, low half first: A[g][2t], A[g][2t + 1]; A[g + 8][2t],
 * A[g + 8][2t + 1]; A[g][2t + 8], A[g][2t + 9]; A[g + 8][2t + 8], A[g + 8][2t + 9]; of the 16x8 operand B, as two:
 * B[2t][g], B[2t + 1][g]; B[2t + 8][g], B[2t + 9][g]; and of the 16x8 result D, as four fp32: D[g][2t], D[g][2t + 1],
 * D[g + 8][2t], D[g + 8][2t + 1]. A[i][k] is the weight of output row 16 * tile + i for the k-th of 16 consecutive
 * inputs: a packed word holds a lane's eight nibbles of one instruction, register r's halves at bits 4r and 4r + 16.
 * Column n of B and of D is the n-th row of a column group. A row's quantized input is stored per scale group as
 * [t][step] words: bytes 0 and 2 of a word are the halves of the step's first B register, bytes 1 and 3 of its second.
 * These two functions say where the code of input k of a row lies: in which of the row's words, at which bit.
 */
__host__ __device__ inline uint32_t code_word(uint32_t input) {
    return input / WEIGHTS_PER_SCALE_GROUP * WORDS_PER_ACTIVATION_GROUP + input % MMA_HALF_INPUTS / HALVES_PER_REGISTER * MMA_STEPS_PER_SCALE_GROUP
           + input % WEIGHTS_PER_SCALE_GROUP / MMA_INPUTS;
}
__host__ __device__ inline uint32_t code_shift(uint32_t input) {
    return (input % HALVES_PER_REGISTER * REGISTERS_PER_OPERAND_B + input % MMA_INPUTS / MMA_HALF_INPUTS) * BITS_PER_BYTE;
}
__host__ __device__ inline void place_e4m3_word(uint32_t word, uint32_t *even_register, uint32_t *odd_register) {
    *even_register = ((word & E4M3_EVEN_SIGN_MASK) << E4M3_EVEN_SIGN_SHIFT) | ((word & E4M3_EVEN_MAGNITUDE_MASK) << E4M3_EVEN_MAGNITUDE_SHIFT);
    *odd_register = (word & E4M3_ODD_SIGN_MASK) | ((word & E4M3_ODD_MAGNITUDE_MASK) >> E4M3_ODD_MAGNITUDE_SHIFT);
}
/* Eight e2m1 nibbles of one packed word become the four fp16 pairs of the instruction's weight operand. */
__host__ __device__ inline void place_e2m1_word(uint32_t word, uint32_t weights[REGISTERS_PER_OPERAND_A]) {
    weights[0] = ((word << E2M1_MAGNITUDE_BIT) & E2M1_MAGNITUDE_MASK) | ((word << E2M1_SIGN_BIT) & E2M1_SIGN_MASK);
    weights[1] = ((word << (E2M1_MAGNITUDE_BIT - NIBBLE_BITS)) & E2M1_MAGNITUDE_MASK) | ((word << (E2M1_SIGN_BIT - NIBBLE_BITS)) & E2M1_SIGN_MASK);
    weights[2] = ((word << (E2M1_MAGNITUDE_BIT - 2 * NIBBLE_BITS)) & E2M1_MAGNITUDE_MASK) | ((word << (E2M1_SIGN_BIT - 2 * NIBBLE_BITS)) & E2M1_SIGN_MASK);
    weights[3] = ((word >> (3 * NIBBLE_BITS - E2M1_MAGNITUDE_BIT)) & E2M1_MAGNITUDE_MASK) | (word & E2M1_SIGN_MASK);
}

/* ---- fill kernels: write the synthetic data in the order the kernels under test read it ---- */
/*
 * One matrix after another, [expert][matrix]: packed words [tile][load][lane][step] (LAYOUT_PACKED), scale codes
 * [tile][scale group][row in tile] (LAYOUT_SCALES), or the bar's fp16 weights [output row][input], each the e2m1 value
 * times its group's scale, exact (LAYOUT_DECODED). LAYOUT_INPUT: the layer's input instead, [row][hidden size] fp16.
 * unit_count counts words, codes or fp16 values.
 */
__global__ void fill_kernel(void *destination, uint32_t layout, uint32_t matrix_kind_base, uint32_t matrices_per_expert, uint32_t output_size,
                            uint32_t input_size, uint64_t unit_count) {
    uint64_t weights_per_matrix = (uint64_t)output_size * input_size;
    uint64_t units_per_matrix = layout == LAYOUT_PACKED ? weights_per_matrix / NIBBLES_PER_WORD
                                : layout == LAYOUT_SCALES ? weights_per_matrix / WEIGHTS_PER_SCALE_GROUP : weights_per_matrix;
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    uint32_t group_count = input_size / WEIGHTS_PER_SCALE_GROUP;
    uint32_t loads_per_tile = input_size / INPUTS_PER_LOAD;
    for (uint64_t unit_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; unit_index < unit_count; unit_index += thread_total) {
        uint64_t matrix_index = unit_index / units_per_matrix;
        uint64_t unit = unit_index % units_per_matrix;
        uint32_t seed = matrix_seed((uint32_t)(matrix_index / matrices_per_expert), matrix_kind_base + (uint32_t)(matrix_index % matrices_per_expert));
        if (layout == LAYOUT_INPUT) {
            ((__half *)destination)[unit_index] = __float2half(input_value((uint32_t)(unit_index / HIDDEN_SIZE), (uint32_t)(unit_index % HIDDEN_SIZE)));
        } else if (layout == LAYOUT_PACKED) {
            uint32_t lane = (uint32_t)(unit / MMA_STEPS_PER_LOAD % LANES_PER_WARP);
            uint64_t load_index = unit / (MMA_STEPS_PER_LOAD * LANES_PER_WARP);
            uint32_t first_row = (uint32_t)(load_index / loads_per_tile) * MMA_ROWS + lane / LANES_PER_LANE_GROUP;
            uint32_t first_input = ((uint32_t)(load_index % loads_per_tile) * MMA_STEPS_PER_LOAD + (uint32_t)(unit % MMA_STEPS_PER_LOAD)) * MMA_INPUTS
                                   + lane % LANES_PER_LANE_GROUP * HALVES_PER_REGISTER;
            uint32_t word = 0;
            for (uint32_t operand_register = 0; operand_register < REGISTERS_PER_OPERAND_A; operand_register++) {
                for (uint32_t half = 0; half < HALVES_PER_REGISTER; half++) {
                    word |= weight_nibble(seed, first_row + (operand_register & 1u) * MMA_HALF_ROWS,
                                          first_input + half + (operand_register >> 1) * MMA_HALF_INPUTS, input_size)
                            << (operand_register * NIBBLE_BITS + half * BITS_PER_HALF);
                }
            }
            ((uint32_t *)destination)[unit_index] = word;
        } else if (layout == LAYOUT_SCALES) {
            uint32_t output_row = (uint32_t)(unit / MMA_ROWS / group_count) * MMA_ROWS + (uint32_t)(unit % MMA_ROWS);
            ((uint8_t *)destination)[unit_index] = (uint8_t)weight_scale_exponent(seed, output_row, (uint32_t)(unit / MMA_ROWS % group_count), group_count);
        } else {
            uint32_t output_row = (uint32_t)(unit / input_size);
            uint32_t input_column = (uint32_t)(unit % input_size);
            uint32_t nibble = weight_nibble(seed, output_row, input_column, input_size);
            uint32_t exponent = weight_scale_exponent(seed, output_row, input_column / WEIGHTS_PER_SCALE_GROUP, group_count);
            uint32_t placed = ((nibble & E2M1_MAGNITUDE_NIBBLE) << E2M1_MAGNITUDE_BIT) | ((nibble & E2M1_SIGN_NIBBLE) << E2M1_SIGN_BIT);
            ((__half *)destination)[unit_index] = __float2half(__half2float(__ushort_as_half((unsigned short)placed))
                                                               * __uint_as_float((exponent + E2M1_PLACEMENT_EXPONENT) << FP32_EXPONENT_SHIFT));
        }
    }
}

/* ---- the stages every form shares: grouping, quantization with swiglu, row sum ---- */
/*
 * Counting sort, one block per expert: thread t scans its share of the rows in row order and counts the pairs of this
 * expert and of the experts before it (a row names an expert at most once). A pair's position is the number of pairs of
 * earlier experts plus this expert's pairs in earlier rows: row order within an expert, whatever the timing, no atomics.
 */
__global__ void group_rows_kernel(const uint32_t *pair_experts, uint32_t row_count, uint32_t *sorted_pairs, uint32_t *pair_positions) {
    __shared__ uint32_t thread_own[HELPER_THREADS_PER_BLOCK];
    __shared__ uint32_t thread_before[HELPER_THREADS_PER_BLOCK];
    uint32_t expert_index = blockIdx.x;
    uint32_t rows_per_thread = (row_count + blockDim.x - 1) / blockDim.x;
    uint32_t first_pair = threadIdx.x * rows_per_thread * EXPERTS_PER_ROW;
    uint32_t last_pair = (threadIdx.x + 1) * rows_per_thread < row_count ? (threadIdx.x + 1) * rows_per_thread * EXPERTS_PER_ROW : row_count * EXPERTS_PER_ROW;
    uint32_t own = 0, before = 0, position = 0;
    for (uint32_t pair = first_pair; pair < last_pair; pair++) {
        own += pair_experts[pair] == expert_index ? 1u : 0u;
        before += pair_experts[pair] < expert_index ? 1u : 0u;
    }
    thread_own[threadIdx.x] = own;
    thread_before[threadIdx.x] = before;
    __syncthreads();
    for (uint32_t other = 0; other < blockDim.x; other++) { position += thread_before[other] + (other < threadIdx.x ? thread_own[other] : 0u); }
    for (uint32_t pair = first_pair; pair < last_pair; pair++) {
        if (pair_experts[pair] == expert_index) {
            sorted_pairs[position] = pair;
            pair_positions[pair] = position++;
        }
    }
}
/* fp32 to the nearest E4M3 code, ties to even; the magnitude is clamped to 448 first. */
__device__ __forceinline__ uint32_t e4m3_code(float value) {
    uint32_t sign = (__float_as_uint(value) >> FP32_SIGN_SHIFT) * E4M3_SIGN_BIT;
    float magnitude = fminf(fabsf(value), E4M3_LARGEST_VALUE);
    if (magnitude < E4M3_SMALLEST_NORMAL) { return sign | (uint32_t)rintf(magnitude * E4M3_SUBNORMAL_STEPS_PER_UNIT); }      /* 8: the first normal */
    uint32_t bits = __float_as_uint(magnitude);
    bits += FP32_ROUNDING_BIAS + ((bits >> FP32_DROPPED_BITS) & 1u);
    return sign | ((bits >> FP32_DROPPED_BITS) - FP32_TO_E4M3_CODE_OFFSET);
}
/*
 * The reference's act_quant for blocks of 32, one thread per block: words are a block's 32 codes in lane order, scales
 * its E8M0 code. is_hidden 0: the values are the layer's input, [row][hidden size] fp16. is_hidden 1: they are computed
 * here, h = silu(min(gate, L)) * clamp(up, -L, L) * route weight for [position][inner size], and also kept as fp32.
 * For the bar, dequantized (not NULL) takes every code times its block's scale as fp16, [block][32]; with is_hidden 0 the
 * blocks are then those of the positions (each reads the block of its pair's row) and no words or scales are written.
 */
template <int is_hidden>
__global__ void quantize_kernel(const __half *layer_input, const float *gate_output, const float *up_output, const uint32_t *sorted_pairs,
                                float *hidden, uint32_t *words, uint8_t *scales, __half *dequantized, uint64_t group_total) {
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t group_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; group_index < group_total; group_index += thread_total) {
        uint64_t first_value = group_index * WEIGHTS_PER_SCALE_GROUP, source_group = group_index;
        float weight = is_hidden ? route_weight(sorted_pairs[group_index / INNER_GROUP_COUNT] % EXPERTS_PER_ROW) : 0.0f;
        float values[WEIGHTS_PER_SCALE_GROUP];
        uint32_t group_words[WORDS_PER_ACTIVATION_GROUP] = {0, 0, 0, 0, 0, 0, 0, 0};
        float amax = ACTIVATION_AMAX_FLOOR;
        if (!is_hidden && dequantized != NULL) {          /* the same block of the position's row */
            uint64_t row = sorted_pairs[group_index / LARGEST_GROUP_COUNT] / EXPERTS_PER_ROW;
            source_group = row * LARGEST_GROUP_COUNT + group_index % LARGEST_GROUP_COUNT;
        }
        for (uint32_t input = 0; input < WEIGHTS_PER_SCALE_GROUP; input++) {
            if (is_hidden) {
                float gate_value = fminf(gate_output[first_value + input], SWIGLU_LIMIT);
                float up_value = fminf(fmaxf(up_output[first_value + input], -SWIGLU_LIMIT), SWIGLU_LIMIT);
                values[input] = gate_value / (1.0f + expf(-gate_value)) * up_value * weight;
                hidden[first_value + input] = values[input];
            } else {
                values[input] = __half2float(layer_input[source_group * WEIGHTS_PER_SCALE_GROUP + input]);
            }
            amax = fmaxf(amax, fabsf(values[input]));
        }
        /* 2^ceil(log2(amax / 448)) from the bits of the fp32 product, as the reference's fast_round_scale */
        uint32_t product_bits = __float_as_uint(amax * ACTIVATION_SCALE_INVERSE);
        uint32_t stored_exponent = ((product_bits >> FP32_EXPONENT_SHIFT) & FP32_EXPONENT_MASK) + ((product_bits & FP32_MANTISSA_MASK) != 0 ? 1u : 0u);
        float inverse_scale = __uint_as_float((E8M0_INVERSE_BASE - stored_exponent) << FP32_EXPONENT_SHIFT);
        float placed_scale = __uint_as_float((stored_exponent + E4M3_PLACEMENT_EXPONENT) << FP32_EXPONENT_SHIFT);
        for (uint32_t input = 0; input < WEIGHTS_PER_SCALE_GROUP; input++) {
            uint32_t code = e4m3_code(values[input] * inverse_scale), placed, unused_register;
            group_words[code_word(input)] |= code << code_shift(input);
            place_e4m3_word(code, &placed, &unused_register);                 /* a code in byte 0: the low half of the first register */
            if (dequantized != NULL) {
                dequantized[first_value + input] = __float2half(__half2float(__ushort_as_half((unsigned short)placed)) * placed_scale);
            }
        }
        for (uint32_t word = 0; word < WORDS_PER_ACTIVATION_GROUP && words != NULL; word++) {
            words[group_index * WORDS_PER_ACTIVATION_GROUP + word] = group_words[word];
        }
        if (scales != NULL) { scales[group_index] = (uint8_t)stored_exponent; }
    }
}
/* The six pair outputs of a row, added in slot order: one block per row. */
__global__ void row_sum_kernel(const float *pair_output, const uint32_t *pair_positions, float *row_output) {
    const float *slot_output[EXPERTS_PER_ROW];
    for (uint32_t slot = 0; slot < EXPERTS_PER_ROW; slot++) {
        slot_output[slot] = pair_output + (size_t)pair_positions[blockIdx.x * EXPERTS_PER_ROW + slot] * HIDDEN_SIZE;
    }
    for (uint32_t column = threadIdx.x; column < HIDDEN_SIZE; column += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < EXPERTS_PER_ROW; slot++) { sum += slot_output[slot][column]; }
        row_output[(size_t)blockIdx.x * HIDDEN_SIZE + column] = sum;
    }
}

/* ---- the kernels under test ---- */
__device__ __forceinline__ void tensor_core_m16n8k16(float accumulator[ACCUMULATORS_PER_LANE], const uint32_t weights[REGISTERS_PER_OPERAND_A],
                                                     uint32_t input_low, uint32_t input_high) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(accumulator[0]), "=f"(accumulator[1]), "=f"(accumulator[2]), "=f"(accumulator[3])
                 : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(input_low), "r"(input_high),
                   "f"(accumulator[0]), "f"(accumulator[1]), "f"(accumulator[2]), "f"(accumulator[3]));
}
/*
 * What a block needs of its chunk, in shared memory: the activation row of each of its 64 columns (a column past the
 * chunk's rows repeats the last row: it is computed and never written) and those rows' scale codes, [column][scale group].
 */
__device__ __forceinline__ struct row_chunk stage_chunk(const struct linear_arguments *arguments, uint32_t *column_rows, uint32_t *column_scale_words) {
    struct row_chunk chunk = arguments->chunks[blockIdx.y];
    uint32_t scale_words_per_row = arguments->input_size / WEIGHTS_PER_SCALE_GROUP / BYTES_PER_WORD;
    const uint32_t *scale_words = (const uint32_t *)arguments->activation_scales;
    for (uint32_t column = threadIdx.x; column < CHUNK_ROWS; column += blockDim.x) {
        uint32_t position = chunk.first_position + (column < chunk.row_count ? column : chunk.row_count - 1);
        column_rows[column] = arguments->activation_is_layer_input ? arguments->sorted_pairs[position] / EXPERTS_PER_ROW : position;
    }
    __syncthreads();
    for (uint32_t item = threadIdx.x; item < CHUNK_ROWS * scale_words_per_row; item += blockDim.x) {
        uint32_t column = item / scale_words_per_row;
        column_scale_words[column * LARGEST_SCALE_WORDS + item % scale_words_per_row] =
            scale_words[(size_t)column_rows[column] * scale_words_per_row + item % scale_words_per_row];
    }
    __syncthreads();
    return chunk;
}
/* A lane's two stored words of one scale group as the B registers of its two instructions. */
__device__ __forceinline__ uint4 placed_inputs(uint2 stored) {
    uint4 placed;
    place_e4m3_word(stored.x, &placed.x, &placed.y);
    place_e4m3_word(stored.y, &placed.z, &placed.w);
    return placed;
}
/*
 * One scale group for one column group: two instructions from zero, then each of the lane's four sums takes the group's sum
 * times 2^(weight code + activation code - 232), exact, fused or not. exponent_low and exponent_high: the weight codes of
 * output rows g and g + 8 less COMBINED_EXPONENT_OFFSET; column_scales: the code of column 2t, that of 2t + 1 lies
 * LARGEST_GROUP_COUNT further.
 */
__device__ __forceinline__ void apply_column_group(float sums[ACCUMULATORS_PER_LANE], uint4 inputs, const uint8_t *column_scales,
                                                   uint32_t weights[MMA_STEPS_PER_SCALE_GROUP][REGISTERS_PER_OPERAND_A],
                                                   uint32_t exponent_low, uint32_t exponent_high) {
    float accumulator[ACCUMULATORS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t exponent_even = column_scales[0];
    uint32_t exponent_odd = column_scales[LARGEST_GROUP_COUNT];
    tensor_core_m16n8k16(accumulator, weights[0], inputs.x, inputs.y);
    tensor_core_m16n8k16(accumulator, weights[1], inputs.z, inputs.w);
    sums[0] = fmaf(accumulator[0], __uint_as_float((exponent_low + exponent_even) << FP32_EXPONENT_SHIFT), sums[0]);
    sums[1] = fmaf(accumulator[1], __uint_as_float((exponent_low + exponent_odd) << FP32_EXPONENT_SHIFT), sums[1]);
    sums[2] = fmaf(accumulator[2], __uint_as_float((exponent_high + exponent_even) << FP32_EXPONENT_SHIFT), sums[2]);
    sums[3] = fmaf(accumulator[3], __uint_as_float((exponent_high + exponent_odd) << FP32_EXPONENT_SHIFT), sums[3]);
}
/* Where one of a lane's four sums of a column group goes, or NULL for a column past the chunk's rows. */
__device__ __forceinline__ float *output_element(const struct linear_arguments *arguments, const struct row_chunk *chunk, uint32_t tile,
                                                 uint32_t column_group, uint32_t lane, uint32_t accumulator_index) {
    uint32_t column = column_group * MMA_COLUMNS + lane % LANES_PER_LANE_GROUP * BATCH_COLUMNS_PER_LANE + accumulator_index % BATCH_COLUMNS_PER_LANE;
    uint32_t output_row = tile * MMA_ROWS + lane / LANES_PER_LANE_GROUP + accumulator_index / BATCH_COLUMNS_PER_LANE * MMA_HALF_ROWS;
    return column < chunk->row_count ? arguments->output + (size_t)(chunk->first_position + column) * arguments->output_size + output_row : NULL;
}
/* A lane's four sums of one column group: thin leaves them where its slices meet, wide (meeting_place NULL) writes them out. */
__device__ __forceinline__ void put_sums(float *meeting_place, const struct linear_arguments *arguments, const struct row_chunk *chunk, uint32_t tile,
                                         uint32_t column_group, uint32_t lane, float sum_0, float sum_1, float sum_2, float sum_3) {
    float lane_sums[ACCUMULATORS_PER_LANE] = {sum_0, sum_1, sum_2, sum_3};
    for (uint32_t accumulator_index = 0; accumulator_index < ACCUMULATORS_PER_LANE; accumulator_index++) {
        float *destination = meeting_place != NULL ? meeting_place + accumulator_index
                                                   : output_element(arguments, chunk, tile, column_group, lane, accumulator_index);
        if (destination != NULL) { *destination = lane_sums[accumulator_index]; }
    }
}
/* The eight column groups are written out one by one, with constant indices, so that every sum has a register of its own. */
#define FOR_EACH_COLUMN_GROUP(statement_of) \
    statement_of(0) statement_of(1) statement_of(2) statement_of(3) statement_of(4) statement_of(5) statement_of(6) statement_of(7)

/*
 * y = W x for every chunk; grid.y is the chunk. Lane (g, t) holds, per column group c, the sums of output rows g and g + 8
 * of its tile for columns 8c + 2t and 8c + 2t + 1, and feeds column 8c + g at inputs 2t, 2t + 1, 2t + 8, 2t + 9.
 * is_wide 0 (thin): grid.x is the tile, THIN_SLICES warps, one slice of the input width each; a column group's stored words
 *   come straight from memory, asked for one scale group ahead. Per row the additions are those of a one-row launch: scale
 *   groups in order inside a slice, then the slices in order, in shared memory.
 * is_wide 1: grid.x is a band of WIDE_WARPS tiles, one warp each, the whole input width in one chain. The inputs are staged
 *   STAGED_GROUPS scale groups at a time: the thread at lane (g, t) of warp w holds the stored words of column 8w + g at
 *   position t, asked for while the block worked on the groups before, and writes them to shared memory as fp16
 *   registers; every warp then reads column group c's registers at its own lane.
 */
template <int is_wide>
__global__ void linear_kernel(struct linear_arguments arguments) {
    __shared__ float partial_sums[is_wide ? UNUSED_EXTENT : THIN_SLICES][CHUNK_COLUMN_GROUPS][LANES_PER_WARP][ACCUMULATORS_PER_LANE];
    __shared__ uint4 staged_inputs[is_wide ? STAGED_GROUPS : UNUSED_EXTENT][CHUNK_COLUMN_GROUPS][LANES_PER_WARP];
    __shared__ uint32_t column_rows[CHUNK_ROWS];
    __shared__ uint32_t column_scale_words[CHUNK_ROWS * LARGEST_SCALE_WORDS];
    struct row_chunk chunk = stage_chunk(&arguments, column_rows, column_scale_words);
    uint32_t warp = threadIdx.x / LANES_PER_WARP, lane = threadIdx.x % LANES_PER_WARP;
    uint32_t lane_group = lane / LANES_PER_LANE_GROUP, lane_in_group = lane % LANES_PER_LANE_GROUP;
    uint32_t tile = is_wide ? blockIdx.x * WIDE_WARPS + warp : blockIdx.x;
    uint32_t group_count = arguments.input_size / WEIGHTS_PER_SCALE_GROUP;
    uint32_t first_group = is_wide ? 0 : warp * (group_count / THIN_SLICES);
    uint32_t last_group = is_wide ? group_count : first_group + group_count / THIN_SLICES;
    uint32_t column_groups = (chunk.row_count + MMA_COLUMNS - 1) / MMA_COLUMNS;
    size_t words_per_row = arguments.input_size / BYTES_PER_WORD;
    const uint4 *lane_packed = arguments.packed + chunk.expert_index * arguments.packed_stride
                               + (size_t)tile * (group_count / SCALE_GROUPS_PER_LOAD) * LANES_PER_WARP + lane;
    const uint8_t *row_scales = arguments.weight_scales + chunk.expert_index * arguments.scale_stride + (size_t)tile * group_count * MMA_ROWS + lane_group;
    const uint8_t *scale_bytes = (const uint8_t *)column_scale_words + lane_in_group * BATCH_COLUMNS_PER_LANE * LARGEST_GROUP_COUNT;
    /* wide: what this thread stages (nothing if the chunk has no column group of its warp's number) */
    const uint2 *staged_source = (const uint2 *)(arguments.activation_words + column_rows[(is_wide ? warp : 0) * MMA_COLUMNS + lane_group] * words_per_row)
                                 + lane_in_group;
    bool stages_inputs = is_wide && warp < column_groups;
    const uint2 *column_words[CHUNK_COLUMN_GROUPS];
    uint2 column_ahead[CHUNK_COLUMN_GROUPS], column_now[CHUNK_COLUMN_GROUPS];        /* thin: a column group's stored words */
    uint2 ahead[STAGED_GROUPS];
    float sums[CHUNK_COLUMN_GROUPS][ACCUMULATORS_PER_LANE] = {};
    uint4 packed = make_uint4(0, 0, 0, 0);
    uint4 packed_ahead = lane_packed[(size_t)(first_group / SCALE_GROUPS_PER_LOAD) * LANES_PER_WARP];
#define COLUMN_SOURCE(c)                                                                                                                 \
    column_words[c] = (const uint2 *)(arguments.activation_words + column_rows[c * MMA_COLUMNS + lane_group] * words_per_row) + lane_in_group; \
    column_ahead[c] = is_wide ? make_uint2(0, 0) : column_words[c][(size_t)first_group * LANES_PER_LANE_GROUP];
    FOR_EACH_COLUMN_GROUP(COLUMN_SOURCE)
    for (uint32_t staged_group = 0; staged_group < STAGED_GROUPS; staged_group++) {
        ahead[staged_group] = stages_inputs ? staged_source[(size_t)staged_group * LANES_PER_LANE_GROUP] : make_uint2(0, 0);
    }
    for (uint32_t group = first_group; group < last_group; group++) {
        uint32_t staged_group = group % STAGED_GROUPS;
        uint32_t group_in_load = group % SCALE_GROUPS_PER_LOAD;       /* first groups are even: a load covers two groups */
        uint32_t weights[MMA_STEPS_PER_SCALE_GROUP][REGISTERS_PER_OPERAND_A];
        uint32_t exponent_low = (uint32_t)row_scales[(size_t)group * MMA_ROWS] - COMBINED_EXPONENT_OFFSET;
        uint32_t exponent_high = (uint32_t)row_scales[(size_t)group * MMA_ROWS + MMA_HALF_ROWS] - COMBINED_EXPONENT_OFFSET;
        if (is_wide && staged_group == 0) {
            for (uint32_t staged = 0; staged < STAGED_GROUPS; staged++) { staged_inputs[staged][warp][lane] = placed_inputs(ahead[staged]); }
            __syncthreads();
            for (uint32_t staged = 0; staged < STAGED_GROUPS && stages_inputs && group + STAGED_GROUPS < last_group; staged++) {
                ahead[staged] = staged_source[(size_t)(group + STAGED_GROUPS + staged) * LANES_PER_LANE_GROUP];      /* in flight meanwhile */
            }
        }
        if (group_in_load == 0) {                         /* the load asked for earlier; ask for the next one */
            packed = packed_ahead;
            if (group + SCALE_GROUPS_PER_LOAD < last_group) { packed_ahead = lane_packed[(size_t)(group / SCALE_GROUPS_PER_LOAD + 1) * LANES_PER_WARP]; }
        }
        place_e2m1_word(group_in_load == 0 ? packed.x : packed.z, weights[0]);
        place_e2m1_word(group_in_load == 0 ? packed.y : packed.w, weights[1]);
#define COLUMN_LOAD(c)                                                                                                                   \
    column_now[c] = column_ahead[c];                      /* thin: asked for during the group before; ask for the next */               \
    if (!is_wide && c < column_groups && group + 1 < last_group) {                                                                       \
        column_ahead[c] = column_words[c][(size_t)(group + 1) * LANES_PER_LANE_GROUP];                                                   \
    }
#define COLUMN_GROUP(c)                                                                                                                  \
    if (c < column_groups) {                                                                                                             \
        apply_column_group(sums[c], is_wide ? staged_inputs[staged_group][c][lane] : placed_inputs(column_now[c]),                       \
                           scale_bytes + c * MMA_COLUMNS * LARGEST_GROUP_COUNT + group, weights, exponent_low, exponent_high);           \
    }
        FOR_EACH_COLUMN_GROUP(COLUMN_LOAD)
        FOR_EACH_COLUMN_GROUP(COLUMN_GROUP)
        if (is_wide && staged_group == STAGED_GROUPS - 1) { __syncthreads(); }    /* nobody still reads what the next groups overwrite */
    }
#define COLUMN_RESULT(c) \
    put_sums(is_wide ? NULL : partial_sums[warp][c][lane], &arguments, &chunk, tile, c, lane, sums[c][0], sums[c][1], sums[c][2], sums[c][3]);
    FOR_EACH_COLUMN_GROUP(COLUMN_RESULT)
    __syncthreads();
    for (uint32_t element = threadIdx.x; element < ELEMENTS_PER_BLOCK && !is_wide; element += blockDim.x) {      /* thin: the slices meet */
        uint32_t accumulator_index = element % ACCUMULATORS_PER_LANE;
        uint32_t source_lane = element / ACCUMULATORS_PER_LANE % LANES_PER_WARP;
        uint32_t column_group = element / (ACCUMULATORS_PER_LANE * LANES_PER_WARP);
        float *destination = output_element(&arguments, &chunk, tile, column_group, source_lane, accumulator_index);
        if (destination != NULL) {
            float sum = 0.0f;
            for (uint32_t slice = 0; slice < THIN_SLICES; slice++) { sum += partial_sums[slice][column_group][source_lane][accumulator_index]; }
            *destination = sum;
        }
    }
}

/* ---- host side: the formats decoded directly, the reference, the routing ---- */
static double e2m1_value(uint32_t nibble) {
    static const double magnitudes[E2M1_MAGNITUDES] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (nibble & E2M1_SIGN_NIBBLE) ? -magnitudes[nibble & E2M1_MAGNITUDE_NIBBLE] : magnitudes[nibble & E2M1_MAGNITUDE_NIBBLE];
}
/* E4M3 decoded from its fields: (8 + mantissa) * 2^(exponent - 10), or mantissa * 2^-9 where the exponent is 0. */
static double e4m3_value(uint32_t code) {
    uint32_t exponent = (code >> E4M3_EXPONENT_SHIFT) & E4M3_EXPONENT_MASK;
    uint32_t mantissa = code & E4M3_MANTISSA_MASK;
    double magnitude = (double)(exponent == 0 ? mantissa << 1 : (E4M3_IMPLICIT_ONE + mantissa) << exponent) / E4M3_VALUE_DIVISOR;
    return (code & E4M3_SIGN_BIT) ? -magnitude : magnitude;
}
/* The nearest E4M3 code, found by trying them all; of two equally near codes the even one. */
static uint32_t host_e4m3_code(double value) {
    double magnitude = fmin(fabs(value), (double)E4M3_LARGEST_VALUE);
    double best_error = magnitude;
    uint32_t best_code = 0;
    for (uint32_t code = 1; code <= E4M3_LARGEST_CODE; code++) {
        double error = fabs(e4m3_value(code) - magnitude);
        if (error < best_error || (error == best_error && (code & 1u) == 0)) {
            best_error = error;
            best_code = code;
        }
    }
    return best_code | (signbit(value) ? 1u : 0u) * E4M3_SIGN_BIT;
}
/*
 * The reference's act_quant on the host, by another route than the kernel's: the scale from frexpf of the same fp32
 * product, the codes from the search above. values hold fp32 numbers; words (lane order) must be zero beforehand.
 */
static void host_quantize(const double *values, uint32_t value_count, uint32_t *words, uint8_t *exponents) {
    for (uint32_t group = 0; group < value_count / WEIGHTS_PER_SCALE_GROUP; group++) {
        double amax = (double)ACTIVATION_AMAX_FLOOR;
        int exponent = 0;
        for (uint32_t input = group * WEIGHTS_PER_SCALE_GROUP; input < (group + 1) * WEIGHTS_PER_SCALE_GROUP; input++) {
            amax = fmax(amax, fabs(values[input]));
        }
        float product = (float)amax * ACTIVATION_SCALE_INVERSE;
        float mantissa = frexpf(product, &exponent);          /* product = mantissa * 2^exponent, mantissa in [0.5, 1) */
        exponent -= mantissa == FREXP_SMALLEST_MANTISSA ? 1 : 0;
        exponents[group] = (uint8_t)(exponent + E8M0_BIAS);
        for (uint32_t input = group * WEIGHTS_PER_SCALE_GROUP; input < (group + 1) * WEIGHTS_PER_SCALE_GROUP; input++) {
            words[code_word(input)] |= host_e4m3_code(ldexp(values[input], -exponent)) << code_shift(input);
        }
    }
}
/* One output element in double: per block of 32, the sum of e2m1 weight times E4M3 input, times the two scales. */
static double host_product(uint32_t expert_index, uint32_t matrix_kind, uint32_t output_row, const uint32_t *words, const uint8_t *exponents) {
    uint32_t input_size = matrix_kind == DOWN_MATRIX ? EXPERT_INNER_SIZE : HIDDEN_SIZE;
    uint32_t group_count = input_size / WEIGHTS_PER_SCALE_GROUP;
    uint32_t seed = matrix_seed(expert_index, matrix_kind);
    double sum = 0.0;
    for (uint32_t group = 0; group < group_count; group++) {
        double group_sum = 0.0;
        for (uint32_t input = group * WEIGHTS_PER_SCALE_GROUP; input < (group + 1) * WEIGHTS_PER_SCALE_GROUP; input++) {
            uint32_t code = (words[code_word(input)] >> code_shift(input)) & BYTE_MASK;
            group_sum += e2m1_value(weight_nibble(seed, output_row, input, input_size)) * e4m3_value(code);
        }
        sum += ldexp(group_sum, (int)weight_scale_exponent(seed, output_row, group, group_count) + (int)exponents[group] - 2 * E8M0_BIAS);
    }
    return sum;
}
/* Output rows that are compared: the first half among the first 92 (inside what --corrupt overwrites), the rest spread. */
static uint32_t sampled_output_row(uint32_t sample, uint32_t output_size) {
    if (sample < LEADING_SAMPLES) { return sample * LEADING_SAMPLE_STRIDE % output_size; }
    return (uint32_t)((uint64_t)(output_size - 1) * (sample - LEADING_SAMPLES + 1) / (SAMPLED_OUTPUTS - LEADING_SAMPLES));
}
/* The pairs that are checked: the six of SAMPLED_ROWS rows spread over the chunk, the first of them row 0. */
static uint32_t sampled_pair_index(uint32_t sampled_pair, uint32_t row_count) {
    uint32_t row = (uint32_t)((uint64_t)(row_count - 1) * (sampled_pair / EXPERTS_PER_ROW) / (SAMPLED_ROWS - 1));
    return row * EXPERTS_PER_ROW + sampled_pair % EXPERTS_PER_ROW;
}
struct host_routing {
    uint32_t *pair_experts;                      /* [pair]: six distinct experts per row */
    uint32_t *pair_positions;                    /* [pair]: its place in the order by expert, then by row */
    struct row_chunk *chunks;                    /* every expert's rows, at most CHUNK_ROWS to a chunk; then the sampled pairs alone */
    uint32_t expert_first[EXPERT_COUNT + 1];
    uint32_t smallest_group, largest_group, chunk_count;
};
static void build_routing(struct host_routing *routing, uint32_t row_count) {
    uint32_t pair_count = row_count * EXPERTS_PER_ROW;
    uint32_t next_position[EXPERT_COUNT];
    memset(routing, 0, sizeof(*routing));
    routing->pair_experts = (uint32_t *)calloc(pair_count, sizeof(uint32_t));
    routing->pair_positions = (uint32_t *)calloc(pair_count, sizeof(uint32_t));
    routing->chunks = (struct row_chunk *)calloc(pair_count / CHUNK_ROWS + EXPERT_COUNT + SAMPLED_PAIRS, sizeof(struct row_chunk));
    check_status(routing->pair_experts == NULL || routing->pair_positions == NULL || routing->chunks == NULL, "build_routing",
                 "no host memory for the routing", __LINE__);
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        uint32_t *row_experts = routing->pair_experts + (pair - pair % EXPERTS_PER_ROW);
        uint32_t slot = pair % EXPERTS_PER_ROW;
        uint32_t expert = mix_bits(ROUTING_SEED + pair / EXPERTS_PER_ROW * ROUTING_ROW_STRIDE + slot) % EXPERT_COUNT;
        for (uint32_t earlier = 0; earlier < slot; earlier++) {
            if (row_experts[earlier] == expert) {                 /* taken: try the next expert, from the first slot again */
                expert = (expert + 1) % EXPERT_COUNT;
                earlier = (uint32_t)-1;
            }
        }
        row_experts[slot] = expert;
        routing->expert_first[expert + 1]++;
    }
    routing->smallest_group = pair_count;
    for (uint32_t expert = 0; expert < EXPERT_COUNT; expert++) {
        uint32_t count = routing->expert_first[expert + 1];
        routing->smallest_group = count < routing->smallest_group ? count : routing->smallest_group;
        routing->largest_group = count > routing->largest_group ? count : routing->largest_group;
        for (uint32_t offset = 0; offset < count; offset += CHUNK_ROWS) {
            struct row_chunk *chunk = &routing->chunks[routing->chunk_count++];
            chunk->expert_index = expert;
            chunk->first_position = routing->expert_first[expert] + offset;
            chunk->row_count = count - offset > CHUNK_ROWS ? (uint32_t)CHUNK_ROWS : count - offset;
        }
        routing->expert_first[expert + 1] += routing->expert_first[expert];
        next_position[expert] = routing->expert_first[expert];
    }
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        routing->pair_positions[pair] = next_position[routing->pair_experts[pair]]++;
    }
    for (uint32_t sampled_pair = 0; sampled_pair < SAMPLED_PAIRS; sampled_pair++) {        /* behind the layer's chunks: one row each */
        struct row_chunk *chunk = &routing->chunks[routing->chunk_count + sampled_pair];
        chunk->expert_index = routing->pair_experts[sampled_pair_index(sampled_pair, row_count)];
        chunk->first_position = routing->pair_positions[sampled_pair_index(sampled_pair, row_count)];
        chunk->row_count = 1;
    }
}

/* ---- host side: one card ---- */
struct card_state {
    cudaDeviceProp properties;
    uint32_t row_count, pair_count, allocation_count;
    unsigned int fill_blocks;
    void *allocations[MAX_CARD_ALLOCATIONS];
    uint4 *packed[MATRIX_SIDES];                 /* [expert][gate, up][tile][load][lane] and, for down, [expert][tile][load][lane] */
    uint8_t *weight_scales[MATRIX_SIDES], *input_scales, *hidden_scales;
    __half *decoded[MATRIX_SIDES];               /* the bar's: [expert][gate, up][inner][hidden] and [expert][hidden][inner] */
    __half *layer_input, *grouped;               /* [row][hidden]; the bar's inputs of the product at hand, [position][its input size] */
    uint32_t *pair_experts, *sorted_pairs, *pair_positions, *input_words, *hidden_words;
    struct row_chunk *chunks;                    /* the layer's, then one-row chunks of the sampled pairs */
    float *gate_output, *up_output, *hidden, *pair_output, *row_output;     /* [position][inner] x 3, [position][hidden], [row][hidden] */
    cublasHandle_t cublas;
    cudaEvent_t events[STAGE_COUNT + 1];
};
struct layer_snapshot { float hidden[SAMPLED_PAIRS][EXPERT_INNER_SIZE], pair_output[SAMPLED_PAIRS][HIDDEN_SIZE]; };   /* of the sampled pairs */

template <typename element_type>
static void card_buffer(struct card_state *card, element_type **buffer, size_t element_count) {
    void *memory = NULL;
    check_status(card->allocation_count >= MAX_CARD_ALLOCATIONS, "card_buffer", "MAX_CARD_ALLOCATIONS is too small", __LINE__);
    CHECK_CUDA(cudaMalloc(&memory, element_count * sizeof(element_type)));
    card->allocations[card->allocation_count++] = memory;
    *buffer = (element_type *)memory;
}
/* One product for every chunk (or, with single_rows, for the one-row chunks of the sampled pairs). */
static void launch_products(struct card_state *card, const struct host_routing *routing, int form, uint32_t matrix_kind, int single_rows) {
    int side = matrix_kind == DOWN_MATRIX ? DOWN_SIDE : GATE_UP_SIDE;
    uint32_t input_size = side == DOWN_SIDE ? EXPERT_INNER_SIZE : HIDDEN_SIZE, output_size = side == DOWN_SIDE ? HIDDEN_SIZE : EXPERT_INNER_SIZE;
    size_t matrices_per_expert = side == DOWN_SIDE ? ONE_MATRIX : GATE_UP_MATRICES, matrix_in_expert = side == DOWN_SIDE ? 0 : matrix_kind;
    size_t loads_per_matrix = WEIGHTS_PER_MATRIX / (NIBBLES_PER_WORD * MMA_STEPS_PER_LOAD), scales_per_matrix = WEIGHTS_PER_MATRIX / WEIGHTS_PER_SCALE_GROUP;
    unsigned int chunk_count = single_rows ? (unsigned int)SAMPLED_PAIRS : routing->chunk_count;
    float *output = side == DOWN_SIDE ? card->pair_output : matrix_kind == GATE_MATRIX ? card->gate_output : card->up_output;
    if (form == FORM_CUBLAS) {                            /* row-major [output][input] is column-major input by output: op(A) = A transposed */
        const float one = 1.0f, zero = 0.0f;
        for (uint32_t expert = 0; expert < EXPERT_COUNT; expert++) {
            size_t first = routing->expert_first[expert];
            int rows = (int)(routing->expert_first[expert + 1] - routing->expert_first[expert]);
            if (rows > 0) {
                CHECK_CUBLAS(cublasGemmEx(card->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_size, rows, (int)input_size, &one,
                                          card->decoded[side] + (expert * matrices_per_expert + matrix_in_expert) * WEIGHTS_PER_MATRIX, CUDA_R_16F,
                                          (int)input_size, card->grouped + first * input_size, CUDA_R_16F, (int)input_size, &zero,
                                          output + first * output_size, CUDA_R_32F, (int)output_size, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
        return;
    }
    struct linear_arguments arguments = {
        card->packed[side] + matrix_in_expert * loads_per_matrix, card->weight_scales[side] + matrix_in_expert * scales_per_matrix,
        matrices_per_expert * loads_per_matrix, matrices_per_expert * scales_per_matrix,
        side == DOWN_SIDE ? card->hidden_words : card->input_words, side == DOWN_SIDE ? card->hidden_scales : card->input_scales,
        card->sorted_pairs, card->chunks + (single_rows ? routing->chunk_count : 0), output, input_size, output_size, side == DOWN_SIDE ? 0u : 1u};
    if (form == FORM_WIDE) {
        linear_kernel<1><<<dim3(output_size / (MMA_ROWS * WIDE_WARPS), chunk_count), WIDE_WARPS * LANES_PER_WARP>>>(arguments);
    } else {
        linear_kernel<0><<<dim3(output_size / MMA_ROWS, chunk_count), THIN_SLICES * LANES_PER_WARP>>>(arguments);
    }
}
/* One stage of a layer, queued on the card. */
static void run_stage(struct card_state *card, const struct host_routing *routing, int form, int stage, int single_rows) {
    uint64_t hidden_values = (uint64_t)card->pair_count * EXPERT_INNER_SIZE;
    if (stage == STAGE_GROUP_ROWS) {
        group_rows_kernel<<<EXPERT_COUNT, HELPER_THREADS_PER_BLOCK>>>(card->pair_experts, card->row_count, card->sorted_pairs, card->pair_positions);
    } else if (stage == STAGE_INPUT_FORMAT) {
        quantize_kernel<0><<<card->fill_blocks, HELPER_THREADS_PER_BLOCK>>>(card->layer_input, NULL, NULL, NULL, NULL, card->input_words, card->input_scales,
                                                                            NULL, (uint64_t)card->row_count * LARGEST_GROUP_COUNT);
        if (form == FORM_CUBLAS) {                        /* the bar's inputs: fp16, by position */
            quantize_kernel<0><<<card->fill_blocks, HELPER_THREADS_PER_BLOCK>>>(card->layer_input, NULL, NULL, card->sorted_pairs, NULL, NULL, NULL,
                                                                                card->grouped, (uint64_t)card->pair_count * LARGEST_GROUP_COUNT);
        }
    } else if (stage == STAGE_HIDDEN_FORMAT) {
        quantize_kernel<1><<<card->fill_blocks, HELPER_THREADS_PER_BLOCK>>>(
            NULL, card->gate_output, card->up_output, card->sorted_pairs, card->hidden, card->hidden_words, card->hidden_scales,
            form == FORM_CUBLAS ? card->grouped : NULL, hidden_values / WEIGHTS_PER_SCALE_GROUP);
    } else if (stage == STAGE_GATE_UP) {
        launch_products(card, routing, form, GATE_MATRIX, single_rows);
        launch_products(card, routing, form, UP_MATRIX, single_rows);
    } else if (stage == STAGE_DOWN) {
        launch_products(card, routing, form, DOWN_MATRIX, single_rows);
    } else {
        row_sum_kernel<<<card->row_count, HELPER_THREADS_PER_BLOCK>>>(card->pair_output, card->pair_positions, card->row_output);
    }
}
/* One layer (with single_rows: its products, for the one-row chunks) over outputs that hold a stale byte, then the card's values for the sampled pairs. */
static void run_layer(struct card_state *card, const struct host_routing *routing, int form, int single_rows, struct layer_snapshot *snapshot) {
    int stale_byte = single_rows ? STALE_BYTE_ONE_ROW : STALE_BYTE_VALUE;      /* NaN, and another byte for one row: an unwritten word cannot match */
    CHECK_CUDA(cudaMemset(card->gate_output, stale_byte, (size_t)card->pair_count * EXPERT_INNER_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(card->up_output, stale_byte, (size_t)card->pair_count * EXPERT_INNER_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(card->pair_output, stale_byte, (size_t)card->pair_count * HIDDEN_SIZE * sizeof(float)));
    for (int stage = single_rows ? STAGE_GATE_UP : STAGE_GROUP_ROWS; stage <= (single_rows ? STAGE_DOWN : STAGE_ROW_SUM); stage++) {
        run_stage(card, routing, form, stage, single_rows);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    for (uint32_t sampled_pair = 0; sampled_pair < SAMPLED_PAIRS; sampled_pair++) {
        size_t position = routing->pair_positions[sampled_pair_index(sampled_pair, card->row_count)];
        read_card(snapshot->hidden[sampled_pair], card->hidden + position * EXPERT_INNER_SIZE, sizeof(snapshot->hidden[0]));
        read_card(snapshot->pair_output[sampled_pair], card->pair_output + position * HIDDEN_SIZE, sizeof(snapshot->pair_output[0]));
    }
}
static uint32_t differing_units(const void *first, const void *second, size_t unit_count, size_t unit_bytes) {
    uint32_t differing = 0;
    for (size_t unit = 0; unit < unit_count; unit++) {
        differing += memcmp((const char *)first + unit * unit_bytes, (const char *)second + unit * unit_bytes, unit_bytes) != 0 ? 1u : 0u;
    }
    return differing;
}
/* SAMPLED_OUTPUTS values of the card against the reference: their worst error over the reference's rms; a NaN stays. */
static double worst_ratio(const double *reference, const float *card_values) {
    double squared_sum = 0.0, worst_error = 0.0;
    for (uint32_t sample = 0; sample < SAMPLED_OUTPUTS; sample++) {
        double error = fabs((double)card_values[sample] - reference[sample]);
        squared_sum += reference[sample] * reference[sample];
        if (error != error || error > worst_error) { worst_error = error; }
    }
    return worst_error / sqrt(squared_sum / (double)SAMPLED_OUTPUTS);
}
/* values (fp32 numbers) quantized on the host against one row of the card's words and exponents, which it also returns. */
static uint32_t quantization_differences(const double *values, uint32_t value_count, const uint32_t *words_on_card, const uint8_t *scales_on_card,
                                         uint32_t *card_words, uint8_t *card_exponents) {
    static uint32_t host_words[HIDDEN_SIZE / BYTES_PER_WORD];
    static uint8_t host_exponents[LARGEST_GROUP_COUNT];
    uint32_t group_count = value_count / WEIGHTS_PER_SCALE_GROUP;
    memset(host_words, 0, sizeof(host_words));
    host_quantize(values, value_count, host_words, host_exponents);
    read_card(card_words, words_on_card, value_count);                    /* a byte per value */
    read_card(card_exponents, scales_on_card, group_count);
    return differing_units(card_words, host_words, value_count / BYTES_PER_WORD, BYTES_PER_WORD)
           + differing_units(card_exponents, host_exponents, group_count, 1);
}
/*
 * The card's position of every pair against the host's own stable sort, then the sampled pairs stage by stage. Returns the checks
 * that did not come out as required: positions, quantized words, exponents and row sums exact, every product within tolerance, except
 * the gate and up product of a pair routed to the expert --corrupt overwrote (that of row 0, slot 0), which must be outside it.
 */
static int check_layer(const struct card_state *card, const struct host_routing *routing, const struct layer_snapshot *snapshot, const char *form_name,
                       int corrupt_weights) {
    static double values[HIDDEN_SIZE];
    static uint32_t input_words[HIDDEN_SIZE / BYTES_PER_WORD], hidden_words[EXPERT_INNER_SIZE / BYTES_PER_WORD], card_order[MAX_ROW_COUNT * EXPERTS_PER_ROW];
    static uint8_t input_exponents[LARGEST_GROUP_COUNT], hidden_exponents[INNER_GROUP_COUNT];
    static float expected_sum[HIDDEN_SIZE], card_row[HIDDEN_SIZE];
    uint32_t unequal = 0, row_sums = 0, pairs_not_as_required = 0;           /* unequal: positions, quantized words, exponents */
    double hidden_worst = 0.0, output_worst = 0.0, reference[SAMPLED_OUTPUTS];
    float card_values[SAMPLED_OUTPUTS];
    read_card(card_order, card->pair_positions, card->pair_count * sizeof(uint32_t));           /* the card's sort: the position of every pair */
    unequal += differing_units(card_order, routing->pair_positions, card->pair_count, sizeof(uint32_t));
    for (uint32_t sampled_pair = 0; sampled_pair < SAMPLED_PAIRS; sampled_pair++) {
        uint32_t pair = sampled_pair_index(sampled_pair, card->row_count);
        uint32_t row = pair / EXPERTS_PER_ROW, slot = pair % EXPERTS_PER_ROW, expert = routing->pair_experts[pair];
        size_t position = routing->pair_positions[pair];
        int must_differ = corrupt_weights && expert == routing->pair_experts[0];
        if (slot == 0) {                                  /* a new row: its input, quantized on the host and by the card */
            for (uint32_t column = 0; column < HIDDEN_SIZE; column++) { values[column] = (double)input_value(row, column); }
            unequal += quantization_differences(values, HIDDEN_SIZE, card->input_words + (size_t)row * (HIDDEN_SIZE / BYTES_PER_WORD),
                                                card->input_scales + (size_t)row * LARGEST_GROUP_COUNT, input_words, input_exponents);
            memset(expected_sum, 0, sizeof(expected_sum));
        }
        for (uint32_t output = 0; output < SAMPLED_OUTPUTS; output++) {       /* gate, up and swiglu from the card's quantized input */
            uint32_t inner = sampled_output_row(output, EXPERT_INNER_SIZE);
            double gate_value = fmin(host_product(expert, GATE_MATRIX, inner, input_words, input_exponents), (double)SWIGLU_LIMIT);
            double up_value = fmin(fmax(host_product(expert, UP_MATRIX, inner, input_words, input_exponents), -(double)SWIGLU_LIMIT), (double)SWIGLU_LIMIT);
            reference[output] = gate_value / (1.0 + exp(-gate_value)) * up_value * (double)route_weight(slot);
            card_values[output] = snapshot->hidden[sampled_pair][inner];
        }
        double hidden_ratio = worst_ratio(reference, card_values);
        /* the card's hidden vector, quantized on the host and by the card */
        for (uint32_t inner = 0; inner < EXPERT_INNER_SIZE; inner++) { values[inner] = (double)snapshot->hidden[sampled_pair][inner]; }
        unequal += quantization_differences(values, EXPERT_INNER_SIZE, card->hidden_words + position * (EXPERT_INNER_SIZE / BYTES_PER_WORD),
                                            card->hidden_scales + position * INNER_GROUP_COUNT, hidden_words, hidden_exponents);
        for (uint32_t output = 0; output < SAMPLED_OUTPUTS; output++) {       /* down from the card's quantized hidden vector */
            uint32_t column = sampled_output_row(output, HIDDEN_SIZE);
            reference[output] = host_product(expert, DOWN_MATRIX, column, hidden_words, hidden_exponents);
            card_values[output] = snapshot->pair_output[sampled_pair][column];
        }
        double output_ratio = worst_ratio(reference, card_values);
        hidden_worst = hidden_ratio != hidden_ratio || hidden_ratio > hidden_worst ? hidden_ratio : hidden_worst;
        output_worst = output_ratio != output_ratio || output_ratio > output_worst ? output_ratio : output_worst;
        pairs_not_as_required += ((hidden_ratio <= RELATIVE_TOLERANCE) == (must_differ != 0) ? 1u : 0u) + (output_ratio <= RELATIVE_TOLERANCE ? 0u : 1u);
        for (uint32_t column = 0; column < HIDDEN_SIZE; column++) { expected_sum[column] += snapshot->pair_output[sampled_pair][column]; }     /* fp32 */
        if (slot == EXPERTS_PER_ROW - 1) {
            read_card(card_row, card->row_output + (size_t)row * HIDDEN_SIZE, sizeof(card_row));
            row_sums += differing_units(card_row, expected_sum, HIDDEN_SIZE, sizeof(float));
        }
    }
    int failures = (unequal != 0 ? 1 : 0) + (row_sums != 0 ? 1 : 0) + (int)pairs_not_as_required;
    printf("  %s: %d rows, %d pairs: %u differing positions, quantized words and exponents, %u differing row sums, worst error over rms %.3e (hidden vector) "
           "and %.3e (pair output): %s\n", form_name, (int)SAMPLED_ROWS, (int)SAMPLED_PAIRS, unequal, row_sums, hidden_worst, output_worst,
           failures != 0 ? "NOT AS REQUIRED" : corrupt_weights ? "the pairs of the overwritten expert differ, the others match" : "matches the reference");
    return failures;
}
/* thin only: the sampled pairs again, each as a chunk of one row; every word must be the one of the full launch. */
static int check_one_row_launches(struct card_state *card, const struct host_routing *routing, const struct layer_snapshot *many_rows) {
    static struct layer_snapshot one_row;
    size_t word_count = sizeof(struct layer_snapshot) / sizeof(uint32_t);
    run_layer(card, routing, FORM_THIN, 1, &one_row);       /* over another stale byte than the full launch's: see run_layer */
    uint32_t differing = differing_units(&one_row, many_rows, word_count, sizeof(uint32_t));
    printf("  thin: %d pairs again, one row to a block: %u of %zu words of hidden vector and pair output differ: %s\n", (int)SAMPLED_PAIRS,
           differing, word_count, differing == 0 ? "every row keeps the bits of its one-row call" : "NOT AS REQUIRED");
    return differing != 0 ? 1 : 0;
}
/* The layer step_count times after the warm-up, an event between stages: one line. A layer the host queues no faster than it runs is the host's figure. */
static void measure_form(struct card_state *card, const struct host_routing *routing, int form, const char *form_name, uint32_t step_count) {
    static const char *const stage_names[STAGE_COUNT] = {"group rows", "quantize input", "gate and up", "swiglu and quantize", "down", "row sum"};
    double stage_milliseconds[STAGE_COUNT] = {0.0}, layer_milliseconds = 0.0, queued_milliseconds = 0.0;
    double flops_per_matrix = FLOPS_PER_WEIGHT * (double)WEIGHTS_PER_MATRIX * card->pair_count;         /* of one matrix for every pair */
    double today = card->properties.minor != 0 ? 0.0 : card->properties.major == SM_80_MAJOR ? TODAY_LAYER_MILLISECONDS_SM_80
                   : card->properties.major == SM_120_MAJOR ? TODAY_LAYER_MILLISECONDS_SM_120 : 0.0;       /* figures of sm_80 and sm_120 only */
    for (uint32_t step = 0; step < WARM_UP_STEPS + step_count; step++) {
        double queue_started = seconds_now();             /* the card is idle here: the step before was waited for */
        CHECK_CUDA(cudaEventRecord(card->events[0]));
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            run_stage(card, routing, form, stage, 0);
            CHECK_CUDA(cudaEventRecord(card->events[stage + 1]));
        }
        queued_milliseconds += step >= WARM_UP_STEPS ? (seconds_now() - queue_started) * MILLISECONDS_PER_SECOND / step_count : 0.0;
        CHECK_CUDA(cudaEventSynchronize(card->events[STAGE_COUNT]));
        CHECK_CUDA(cudaGetLastError());
        for (int stage = 0; stage < STAGE_COUNT && step >= WARM_UP_STEPS; stage++) {
            float milliseconds = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, card->events[stage], card->events[stage + 1]));
            stage_milliseconds[stage] += (double)milliseconds / step_count;
            layer_milliseconds += (double)milliseconds / step_count;
        }
    }
    printf("  %s: %.2f ms per layer, %.0f TFLOP/s (", form_name, layer_milliseconds,
           (double)MATRIX_KINDS * flops_per_matrix / (layer_milliseconds / MILLISECONDS_PER_SECOND) / FLOPS_PER_TFLOP);
    for (int stage = 0; stage < STAGE_COUNT; stage++) {
        printf("%s%s %.2f", stage == 0 ? "" : ", ", stage_names[stage], stage_milliseconds[stage]);
        if (stage == STAGE_GATE_UP || stage == STAGE_DOWN) {
            printf(" at %.0f TFLOP/s", (double)(stage == STAGE_DOWN ? ONE_MATRIX : GATE_UP_MATRICES) * flops_per_matrix
                                       / (stage_milliseconds[stage] / MILLISECONDS_PER_SECOND) / FLOPS_PER_TFLOP);
        }
    }
    printf(" ms); the host queues a layer in %.2f ms; today's engine at %d rows: ", queued_milliseconds, (int)DEFAULT_ROW_COUNT);
    if (today > 0.0) { printf("%.1f ms\n", today); } else { printf("no figure for this card\n"); }
}
static int probe_card(int card_index, const struct host_routing *routing, uint32_t row_count, const int *form_enabled, uint32_t step_count,
                      int corrupt_weights) {
    static const char *const form_names[FORM_COUNT] = {"thin", "wide", "cublas"};
    static struct layer_snapshot snapshot;
    static struct card_state card_storage;
    struct card_state *card = &card_storage;
    size_t pair_count = (size_t)row_count * EXPERTS_PER_ROW, inner_values = pair_count * EXPERT_INNER_SIZE, hidden_values = pair_count * HIDDEN_SIZE;
    size_t corrupted_weight = routing->pair_experts[0] * GATE_UP_MATRICES * WEIGHTS_PER_MATRIX;         /* the first of that expert's gate */
    size_t bytes_per_expert = MATRIX_KINDS * (WEIGHTS_PER_MATRIX * BYTES_PER_WORD / NIBBLES_PER_WORD + WEIGHTS_PER_MATRIX / WEIGHTS_PER_SCALE_GROUP);
    int holds_packed = form_enabled[FORM_THIN] || form_enabled[FORM_WIDE], failures = 0;
    memset(card, 0, sizeof(*card));
    CHECK_CUDA(cudaSetDevice(card_index));
    CHECK_CUDA(cudaGetDeviceProperties(&card->properties, card_index));
    card->row_count = row_count;
    card->pair_count = (uint32_t)pair_count;
    card->fill_blocks = (unsigned int)card->properties.multiProcessorCount * FILL_BLOCKS_PER_SM;
#define FILL_WEIGHTS(buffer, layout, unit_count) \
    fill_kernel<<<card->fill_blocks, HELPER_THREADS_PER_BLOCK>>>(buffer, layout, kind, matrices, output_size, input_size, unit_count)
    for (int side = 0; side < MATRIX_SIDES; side++) {     /* gate with up, then down: packed for thin and wide, decoded for the bar */
        uint32_t kind = side == DOWN_SIDE ? DOWN_MATRIX : GATE_MATRIX, matrices = side == DOWN_SIDE ? ONE_MATRIX : GATE_UP_MATRICES;
        uint32_t output_size = side == DOWN_SIDE ? HIDDEN_SIZE : EXPERT_INNER_SIZE, input_size = side == DOWN_SIDE ? EXPERT_INNER_SIZE : HIDDEN_SIZE;
        size_t weights = EXPERT_COUNT * matrices * WEIGHTS_PER_MATRIX;
        if (holds_packed) {
            card_buffer(card, &card->packed[side], weights / (NIBBLES_PER_WORD * MMA_STEPS_PER_LOAD));
            card_buffer(card, &card->weight_scales[side], weights / WEIGHTS_PER_SCALE_GROUP);
            FILL_WEIGHTS(card->packed[side], LAYOUT_PACKED, weights / NIBBLES_PER_WORD);
            FILL_WEIGHTS(card->weight_scales[side], LAYOUT_SCALES, weights / WEIGHTS_PER_SCALE_GROUP);
        }
        if (form_enabled[FORM_CUBLAS]) {                  /* four times the packed weights: a card without the room stops here */
            card_buffer(card, &card->decoded[side], weights);
            FILL_WEIGHTS(card->decoded[side], LAYOUT_DECODED, weights);
        }
    }
    card_buffer(card, &card->pair_experts, pair_count);
    card_buffer(card, &card->sorted_pairs, pair_count);
    card_buffer(card, &card->pair_positions, pair_count);
    card_buffer(card, &card->chunks, routing->chunk_count + SAMPLED_PAIRS);
    card_buffer(card, &card->layer_input, (size_t)row_count * HIDDEN_SIZE);
    card_buffer(card, &card->input_words, (size_t)row_count * HIDDEN_SIZE / BYTES_PER_WORD);
    card_buffer(card, &card->input_scales, (size_t)row_count * LARGEST_GROUP_COUNT);
    card_buffer(card, &card->gate_output, inner_values);
    card_buffer(card, &card->up_output, inner_values);
    card_buffer(card, &card->hidden, inner_values);
    card_buffer(card, &card->hidden_words, inner_values / BYTES_PER_WORD);
    card_buffer(card, &card->hidden_scales, inner_values / WEIGHTS_PER_SCALE_GROUP);
    card_buffer(card, &card->pair_output, hidden_values);
    card_buffer(card, &card->row_output, (size_t)row_count * HIDDEN_SIZE);
    card_buffer(card, &card->grouped, form_enabled[FORM_CUBLAS] ? hidden_values : 1);
    for (int event = 0; event <= STAGE_COUNT; event++) { CHECK_CUDA(cudaEventCreate(&card->events[event])); }
    if (form_enabled[FORM_CUBLAS]) { CHECK_CUBLAS(cublasCreate(&card->cublas)); }
    CHECK_CUDA(cudaMemcpy(card->pair_experts, routing->pair_experts, pair_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(card->chunks, routing->chunks, (routing->chunk_count + SAMPLED_PAIRS) * sizeof(struct row_chunk), cudaMemcpyHostToDevice));
    fill_kernel<<<card->fill_blocks, HELPER_THREADS_PER_BLOCK>>>(card->layer_input, LAYOUT_INPUT, 0, ONE_MATRIX, 1, HIDDEN_SIZE,
                                                                 (uint64_t)row_count * HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    if (corrupt_weights && holds_packed) {
        uint8_t *first_byte = (uint8_t *)card->packed[GATE_UP_SIDE] + corrupted_weight * BYTES_PER_WORD / NIBBLES_PER_WORD;
        CHECK_CUDA(cudaMemset(first_byte, CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES));
    }
    if (corrupt_weights && form_enabled[FORM_CUBLAS]) {
        CHECK_CUDA(cudaMemset(card->decoded[GATE_UP_SIDE] + corrupted_weight, CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES));
    }
    printf("card %d %s (sm_%d%d, %d SMs): %d e2m1 experts of %.2f MiB (%.2f GiB), %u rows, %zu row-expert pairs, %u to %u rows per expert, "
           "%u chunks of up to %d rows\n", card_index, card->properties.name, card->properties.major, card->properties.minor,
           card->properties.multiProcessorCount, (int)EXPERT_COUNT, (double)bytes_per_expert / BYTES_PER_MIB,
           (double)(bytes_per_expert * EXPERT_COUNT) / BYTES_PER_GIB, row_count, pair_count, routing->smallest_group, routing->largest_group,
           routing->chunk_count, (int)CHUNK_ROWS);
    for (int form = 0; form < FORM_COUNT; form++) {
        if (form_enabled[form]) {
            run_layer(card, routing, form, 0, &snapshot);
            failures += check_layer(card, routing, &snapshot, form_names[form], corrupt_weights);
            if (!corrupt_weights) {                       /* with --corrupt the megabyte stays overwritten: the check shows it */
                failures += form == FORM_THIN ? check_one_row_launches(card, routing, &snapshot) : 0;
                measure_form(card, routing, form, form_names[form], step_count);
            }
        }
    }
    for (int event = 0; event <= STAGE_COUNT; event++) { CHECK_CUDA(cudaEventDestroy(card->events[event])); }
    if (form_enabled[FORM_CUBLAS]) { CHECK_CUBLAS(cublasDestroy(card->cublas)); }
    while (card->allocation_count > 0) { CHECK_CUDA(cudaFree(card->allocations[--card->allocation_count])); }
    return failures;
}
int main(int argument_count, char **arguments) {
    static struct host_routing routing;
    int form_enabled[FORM_COUNT] = {1, 1, 1};
    long row_count = DEFAULT_ROW_COUNT, step_count = DEFAULT_STEP_COUNT;
    int corrupt_weights = 0, usage_is_wrong = 0, card_count = 0, failures = 0;
    for (int argument_index = 1; argument_index < argument_count; argument_index += 2) {
        const char *argument = arguments[argument_index];
        const char *value = argument_index + 1 < argument_count ? arguments[argument_index + 1] : "";
        int is_all = strcmp(value, "all") == 0;
        if (strcmp(argument, "--corrupt") == 0) {
            corrupt_weights = 1;
            argument_index--;                             /* the only argument without a value */
        } else if (strcmp(argument, "--form") == 0) {
            form_enabled[FORM_THIN] = is_all || strcmp(value, "thin") == 0;
            form_enabled[FORM_WIDE] = is_all || strcmp(value, "wide") == 0;
            form_enabled[FORM_CUBLAS] = is_all || strcmp(value, "cublas") == 0;
            usage_is_wrong |= !(form_enabled[FORM_THIN] || form_enabled[FORM_WIDE] || form_enabled[FORM_CUBLAS]);
        } else if (strcmp(argument, "--rows") == 0) {
            row_count = strtol(value, NULL, DECIMAL_BASE);                    /* 0 for what is not a number: refused below */
        } else if (strcmp(argument, "--steps") == 0) {
            step_count = strtol(value, NULL, DECIMAL_BASE);
        } else {
            usage_is_wrong = 1;
        }
    }
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (usage_is_wrong || row_count < 1 || row_count > MAX_ROW_COUNT || step_count < 1 || step_count > MAX_STEP_COUNT || card_count < 1) {
        fprintf(stderr, "usage: %s [--form thin|wide|cublas|all] [--rows N] [--steps N] [--corrupt], with at least one card\n", arguments[0]);
        return EXIT_FAILURE;
    }
    build_routing(&routing, (uint32_t)row_count);
    for (int card_index = 0; card_index < card_count && card_index < MAX_CARDS; card_index++) {
        failures += probe_card(card_index, &routing, (uint32_t)row_count, form_enabled, (uint32_t)step_count, corrupt_weights);
    }
    if (failures != 0) { fprintf(stderr, "%d check(s) did not come out as required\n", failures); }
    return failures != 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
