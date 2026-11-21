/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * eight_bit_trunk_probe: what the trunk of one layer (the attention projections and the shared expert) costs when
 * its weights stay the checkpoint's own 8-bit floats, on each card, at 1 to 8 rows.
 *
 * Today's engine stores these matrices in a 3-bit trellis format: at one row its kernels take, per layer, 0.197 ms
 * on the CMP 170HX and 0.093 ms on the RTX PRO 6000 for the trunk linears, plus 0.099 ms and 0.034 ms for the
 * shared expert. The checkpoint stores them as E4M3 bytes with one E8M0 scale (2^(e - 127)) per block of 32 output
 * rows by 32 inputs. The five attention matrices are measured as plain linears y = W x, the shared expert as its
 * chain: gate = W1 x and up = W3 x in one launch, h = silu(min(gate, L)) * clamp(up, -L, L) kept as fp16, y = W2 h.
 * Weights, scales and inputs are synthetic (a hash of their position): every run compares sampled outputs of the
 * first and the last row of the batch with a double-precision reference that decodes E4M3 directly, and --corrupt
 * overwrites a megabyte of one matrix first, where the comparison must fail.
 *
 * Kernel design (the prototype of the engine's 8-bit linear; four_bit_expert_probe is its 4-bit sibling):
 *   - tensor-core instruction m16n8k16 with the WEIGHTS as its 16x16 operand and up to eight rows of the batch as
 *     its eight columns, fp32 accumulation: 1 to 8 rows cost the same instructions;
 *   - weights stored in the order the instruction's lanes consume them: one 16-byte load per lane is two
 *     instructions = 32 inputs = one scale group, and the 32 lanes of a warp read 512 consecutive bytes;
 *   - a byte becomes an fp16 by bit placement only, ((b & 0x80) << 8) | ((b & 0x7F) << 7): the E4M3 value times
 *     2^-8 for every code, subnormal ones included (the host proves it before any card is touched). The 2^8 goes
 *     into the group's scale, applied once per 32 inputs in fp32 (a power of two: exact);
 *   - one block owns 16 output rows, inside one 32-row scale block; its warps each take a slice of the input
 *     width and ask for their next 512 bytes before working on the ones they hold; the slices' partial sums meet
 *     in shared memory and are added in a fixed order: no atomics, nothing depends on timing. The slice count is
 *     a launch parameter: the probe tries 1, 2, 4 and 8 for every matrix and reports the fastest.
 * Every launch reads another copy of its matrix, from a ring COLD_RING_L2_MULTIPLE times the card's L2 cache: the
 * engine reads a layer's weights once per token, from memory (one layer is 162 MB; the RTX PRO 6000 caches 128 MiB).
 *
 *   eight_bit_trunk_probe [--steps N] [--corrupt]
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

#define SWIGLU_LIMIT 10.0f
#define WEIGHTS_PER_SCALE_GROUP 32               /* inputs that share one scale */
#define SCALE_BLOCK_ROWS 32                      /* output rows that share one scale */
#define MMA_ROWS 16                              /* output rows one block owns (the instruction's M) */
#define MMA_HALF_ROWS (MMA_ROWS / 2)
#define MMA_INPUTS 16                            /* inputs one instruction consumes (its K) */
#define MAX_ROWS 8                               /* rows of the batch one instruction carries (its N) */
#define MMA_STEPS_PER_SCALE_GROUP (WEIGHTS_PER_SCALE_GROUP / MMA_INPUTS)
#define TILES_PER_SCALE_BLOCK (SCALE_BLOCK_ROWS / MMA_ROWS)
#define LANES_PER_WARP 32
#define LANES_PER_LANE_GROUP 4                   /* lanes that share an instruction row pair */
#define HALVES_PER_REGISTER 2
#define REGISTERS_PER_OPERAND_A 4
#define REGISTERS_PER_OPERAND_B 2
#define REGISTERS_PER_WORD 2                     /* one stored word fills two registers of the weight operand */
#define WORDS_PER_STEP (REGISTERS_PER_OPERAND_A / REGISTERS_PER_WORD)
#define WORDS_PER_LOAD (WORDS_PER_STEP * MMA_STEPS_PER_SCALE_GROUP)      /* one 16-byte load = one scale group */
#define BYTES_PER_WORD 4
#define BITS_PER_BYTE 8
#define BITS_PER_HALF 16
#define INPUTS_PER_WORD (MMA_INPUTS / WORDS_PER_STEP)                    /* over the four lanes of a lane group */
#define INPUT_PAIRS_PER_STEP (MMA_INPUTS / HALVES_PER_REGISTER)
#define INPUT_PAIRS_PER_REGISTER (INPUT_PAIRS_PER_STEP / REGISTERS_PER_OPERAND_B)
#define INPUT_PAIRS_PER_SCALE_GROUP (WEIGHTS_PER_SCALE_GROUP / HALVES_PER_REGISTER)
#define ACCUMULATORS_PER_LANE 4                  /* two output rows by two rows of the batch */
#define BATCH_COLUMNS_PER_LANE 2
#define SHARED_BYTES_PER_WARP (LANES_PER_WARP * ACCUMULATORS_PER_LANE * sizeof(float))
#define MAX_INPUT_SLICES 8                       /* warps that share the input width of one tile of one matrix */
#define MAX_THREADS_PER_BLOCK 1024
#define FILL_THREADS_PER_BLOCK 256
#define FILL_BLOCKS_PER_SM 8

/* E4M3: sign, four exponent bits with bias 7, three mantissa bits; exponent 0 is subnormal; no infinities. */
#define E4M3_CODES 256u
#define E4M3_SIGN_BIT 0x80u
#define E4M3_MAGNITUDE_MASK 0x7Fu                /* a code with all of these set is one of the two NaN codes */
#define E4M3_EXPONENT_SHIFT 3
#define E4M3_EXPONENT_MASK 15u
#define E4M3_MANTISSA_MASK 7u
#define E4M3_IMPLICIT_ONE 8u                     /* 1.0 in units of the mantissa */
#define E4M3_VALUE_DIVISOR 1024.0                /* 2^(bias + mantissa bits), see e4m3_value */
#define E4M3_PLACEMENT_EXPONENT 8                /* the placement below yields the value times 2^-8 */
/* the placement for bytes 0 and 2 of a stored word at once ("even") and for bytes 1 and 3 ("odd": signs in place) */
#define E4M3_EVEN_SIGN_MASK 0x00800080u
#define E4M3_EVEN_SIGN_SHIFT 8
#define E4M3_EVEN_MAGNITUDE_MASK 0x007F007Fu
#define E4M3_EVEN_MAGNITUDE_SHIFT 7
#define E4M3_ODD_SIGN_MASK 0x80008000u
#define E4M3_ODD_MAGNITUDE_MASK 0x7F007F00u
#define E4M3_ODD_MAGNITUDE_SHIFT 1
#define E8M0_BIAS 127
#define FP32_EXPONENT_SHIFT 23
#define FP16_SIGN_BIT 0x8000u
#define FP16_HALF_MASK 0xFFFFu
#define FP16_EXPONENT_SHIFT 10
#define FP16_EXPONENT_MASK 31
#define FP16_MANTISSA_MASK 1023
#define FP16_IMPLICIT_ONE 1024
#define FP16_SUBNORMAL_EXPONENT (-24)            /* a subnormal is mantissa * 2^-24 */
#define FP16_EXPONENT_OFFSET 25                  /* a normal is (1024 + mantissa) * 2^(exponent - 25) */

#define SCALE_EXPONENT_BASE 114u                 /* synthetic scales 2^-13 .. 2^-10 */
#define SCALE_EXPONENT_SPREAD 3u
#define INPUT_VALUE_STEPS 255u                   /* synthetic inputs -2 .. 2 in steps of 1/64 */
#define INPUT_VALUE_OFFSET 128.0f
#define INPUT_VALUE_DIVISOR 64.0f

#define ATTENTION_MATRICES 5                     /* rows 0 .. 4 of the table: plain linears */
#define GATE_MATRIX ATTENTION_MATRICES
#define UP_MATRIX (GATE_MATRIX + 1)
#define DOWN_MATRIX (UP_MATRIX + 1)
#define LAYER_MATRICES (DOWN_MATRIX + 1)
#define MATRICES_PER_GATE_UP_LAUNCH 2            /* gate and up share the input */
/* launch units: the five linears, gate with up, down; a unit below DOWN_UNIT has the number of its first matrix */
#define GATE_UP_UNIT GATE_MATRIX
#define DOWN_UNIT (GATE_UP_UNIT + 1)
#define LAUNCH_UNITS (DOWN_UNIT + 1)
#define SHARED_EXPERT_UNITS 2
#define LAYER_SEQUENCES 2                        /* what a layer runs in a row: the trunk linears, the shared expert */
/* check units: the five linears, then the shared expert's chain, which starts at the launch unit of that number */
#define SHARED_EXPERT_CHECK GATE_UP_UNIT
#define CHECK_UNITS (SHARED_EXPERT_CHECK + 1)
#define LARGEST_INPUT_SIZE 8192                  /* buffers are sized by these; the table is checked against them */
#define LARGEST_OUTPUT_SIZE 32768
#define INPUT_REGION_HALVES ((size_t)MAX_ROWS * LARGEST_INPUT_SIZE)
#define OUTPUT_REGION_FLOATS ((size_t)MAX_ROWS * LARGEST_OUTPUT_SIZE)

#define DEFAULT_STEP_COUNT 200
#define MAX_STEP_COUNT 1000000
#define DECIMAL_BASE 10
#define WARM_UP_STEPS 10
#define COLD_RING_L2_MULTIPLE 4                  /* bytes a launch unit reads before it meets a copy again, in L2s */
#define FALLBACK_L2_BYTES (128u * 1024u * 1024u) /* for a card that does not report its L2 size */
#define SAMPLED_OUTPUTS 16                       /* output elements compared per checked row of the batch */
#define LEADING_SAMPLES (SAMPLED_OUTPUTS / 2)    /* of which this many among the first output rows */
#define LEADING_SAMPLE_STRIDE 13
#define CORRUPTED_MATRIX GATE_MATRIX
#define CORRUPTED_BYTES (1024u * 1024u)          /* --corrupt overwrites this much of the first copy of that matrix */
#define CORRUPTED_BYTE_VALUE 0x5A
#define STALE_BYTE_VALUE 0xFF                    /* outputs are filled with NaN before every checked launch */
#define RELATIVE_TOLERANCE 2.0e-3
#define TODAY_TRUNK_MICROSECONDS_SM_80 197       /* today's 3-bit kernels at one row, per layer */
#define TODAY_TRUNK_MICROSECONDS_SM_120 93
#define TODAY_SHARED_EXPERT_MICROSECONDS_SM_80 99
#define TODAY_SHARED_EXPERT_MICROSECONDS_SM_120 34
#define SM_80_MAJOR 8
#define SM_120_MAJOR 12
#define BYTES_PER_MB 1.0e6
#define BYTES_PER_GB 1.0e9
#define BYTES_PER_MIB (1024.0 * 1024.0)
#define BYTES_PER_GIB (1024.0 * BYTES_PER_MIB)
#define MICROSECONDS_PER_MILLISECOND 1000.0
#define MICROSECONDS_PER_SECOND 1.0e6
#define MAX_CARDS 16

struct matrix_shape {
    const char *name;
    uint32_t output_size;
    uint32_t input_size;
};

/* The matrices of one layer, all with 32-by-32 scale blocks. */
static const struct matrix_shape layer_matrices[LAYER_MATRICES] = {
    {"attention query A", 1280, 5120},
    {"attention query B", 32768, 1280},
    {"attention key/value", 512, 5120},
    {"attention output A", 8192, 4096},
    {"attention output B", 5120, 8192},
    {"shared expert gate", 2304, 5120},
    {"shared expert up", 2304, 5120},
    {"shared expert down", 5120, 2304},
};

static const uint32_t row_counts[] = {1, 2, 4, 8};
#define ROW_COUNT_CASES (sizeof(row_counts) / sizeof(row_counts[0]))
static const uint32_t slice_candidates[] = {1, 2, 4, 8};
#define SLICE_CANDIDATES (sizeof(slice_candidates) / sizeof(slice_candidates[0]))

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

/* A weight's stored byte: any E4M3 code but the two NaN codes, which become the largest magnitude instead. */
__host__ __device__ inline uint32_t weight_byte(uint32_t seed, uint32_t output_row, uint32_t input_column,
                                                uint32_t input_size)
{
    uint32_t code = mix_bits(seed + output_row * input_size + input_column) & (E4M3_SIGN_BIT | E4M3_MAGNITUDE_MASK);
    if ((code & E4M3_MAGNITUDE_MASK) == E4M3_MAGNITUDE_MASK) {
        code -= 1u;
    }
    return code;
}

/* The E8M0 code of the scale of one block of 32 output rows by 32 inputs. */
__host__ __device__ inline uint32_t scale_exponent(uint32_t seed, uint32_t scale_block, uint32_t group,
                                                   uint32_t group_count)
{
    return SCALE_EXPONENT_BASE + (mix_bits(~seed + scale_block * group_count + group) & SCALE_EXPONENT_SPREAD);
}

__host__ __device__ inline float input_value(uint32_t seed, uint32_t row_index, uint32_t column, uint32_t input_size)
{
    uint32_t code = mix_bits(seed + row_index * input_size + column) & INPUT_VALUE_STEPS;
    return ((float)code - INPUT_VALUE_OFFSET) / INPUT_VALUE_DIVISOR;
}

/*
 * The order of the stored bytes. For m16n8k16 the PTX documentation gives lane L (group g = L / 4, position
 * t = L % 4) these elements of the 16x16 operand A, as four registers of two fp16 halves, low half first:
 *   register 0: A[g][2t],         A[g][2t + 1]
 *   register 1: A[g + 8][2t],     A[g + 8][2t + 1]
 *   register 2: A[g][2t + 8],     A[g][2t + 9]
 *   register 3: A[g + 8][2t + 8], A[g + 8][2t + 9]
 * of the 16x8 operand B, as two registers: B[2t][g], B[2t + 1][g] and B[2t + 8][g], B[2t + 9][g]; and of the 16x8
 * result D, as four fp32: D[g][2t], D[g][2t + 1], D[g + 8][2t], D[g + 8][2t + 1]. A[i][k] is the weight of output
 * row 16 * tile + i for the k-th of 16 consecutive inputs; column n of B and of D is row n of the batch.
 *
 * A matrix is stored as [tile][scale group][lane][16 bytes]. A lane's 16 bytes are four words, one per pair of
 * registers: words 0 and 1 are registers 0, 1 and 2, 3 of the group's first instruction, words 2 and 3 those of
 * its second. Inside a word, bytes 0 and 2 are the low and the high half of the even register and bytes 1 and 3
 * those of the odd one, so each register is two masks and at most two shifts away from its word. Byte q of a lane
 * is therefore W[16 * tile + g + 8 * (q % 2)][32 * group + 8 * (q / 4) + 2t + (q / 2) % 2].
 */
__host__ __device__ inline void place_e4m3_word(uint32_t word, uint32_t *even_register, uint32_t *odd_register)
{
    *even_register = ((word & E4M3_EVEN_SIGN_MASK) << E4M3_EVEN_SIGN_SHIFT)
                     | ((word & E4M3_EVEN_MAGNITUDE_MASK) << E4M3_EVEN_MAGNITUDE_SHIFT);
    *odd_register = (word & E4M3_ODD_SIGN_MASK) | ((word & E4M3_ODD_MAGNITUDE_MASK) >> E4M3_ODD_MAGNITUDE_SHIFT);
}

/* ---- fill kernels: write the synthetic data in the order the kernels under test read it ---- */

/* Every copy of the ring holds the same matrix: word_count is a whole number of words_per_copy. */
__global__ void fill_weights_kernel(uint32_t *words, uint32_t seed, uint32_t input_size, uint64_t words_per_copy,
                                    uint64_t word_count)
{
    uint32_t group_count = input_size / WEIGHTS_PER_SCALE_GROUP;
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t word_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; word_index < word_count;
         word_index += thread_total) {
        uint64_t word_in_copy = word_index % words_per_copy;
        uint32_t word_in_load = (uint32_t)(word_in_copy % WORDS_PER_LOAD);
        uint32_t lane = (uint32_t)(word_in_copy / WORDS_PER_LOAD % LANES_PER_WARP);
        uint64_t load_index = word_in_copy / (WORDS_PER_LOAD * LANES_PER_WARP);
        uint32_t group = (uint32_t)(load_index % group_count);
        uint32_t tile = (uint32_t)(load_index / group_count);
        uint32_t lane_group = lane / LANES_PER_LANE_GROUP;
        uint32_t lane_in_group = lane % LANES_PER_LANE_GROUP;
        uint32_t word = 0;
        for (uint32_t byte_in_word = 0; byte_in_word < BYTES_PER_WORD; byte_in_word++) {
            uint32_t output_row = tile * MMA_ROWS + lane_group + (byte_in_word % REGISTERS_PER_WORD) * MMA_HALF_ROWS;
            uint32_t input_column = group * WEIGHTS_PER_SCALE_GROUP + word_in_load * INPUTS_PER_WORD
                                    + lane_in_group * HALVES_PER_REGISTER + byte_in_word / REGISTERS_PER_WORD;
            word |= weight_byte(seed, output_row, input_column, input_size) << (byte_in_word * BITS_PER_BYTE);
        }
        words[word_index] = word;
    }
}

/* Scales are stored as [scale block of 32 output rows][scale group]: a tile reads its block's row front to back. */
__global__ void fill_scales_kernel(uint8_t *scales, uint32_t seed, uint32_t group_count, uint64_t entries_per_copy,
                                   uint64_t entry_count)
{
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t entry_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; entry_index < entry_count;
         entry_index += thread_total) {
        uint64_t entry_in_copy = entry_index % entries_per_copy;
        uint32_t group = (uint32_t)(entry_in_copy % group_count);
        uint32_t scale_block = (uint32_t)(entry_in_copy / group_count);
        scales[entry_index] = (uint8_t)scale_exponent(seed, scale_block, group, group_count);
    }
}

/* One block per row of the batch. */
__global__ void fill_input_kernel(__half *input, uint32_t seed, uint32_t input_size)
{
    uint32_t row_index = blockIdx.x;
    for (uint32_t column = threadIdx.x; column < input_size; column += blockDim.x) {
        float value = input_value(seed, row_index, column, input_size);
        input[(size_t)row_index * input_size + column] = __float2half(value);
    }
}

/* ---- the kernels under test ---- */

__device__ __forceinline__ void tensor_core_m16n8k16(float accumulator[ACCUMULATORS_PER_LANE],
                                                     const uint32_t weights[REGISTERS_PER_OPERAND_A],
                                                     uint32_t input_low, uint32_t input_high)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(accumulator[0]), "=f"(accumulator[1]), "=f"(accumulator[2]), "=f"(accumulator[3])
        : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(input_low), "r"(input_high),
          "f"(accumulator[0]), "f"(accumulator[1]), "f"(accumulator[2]), "f"(accumulator[3]));
}

/* 2^(e - 127) times the 2^8 that the bit placement took from every weight, written as an fp32 exponent field. */
__device__ __forceinline__ float group_scale(uint32_t stored_exponent)
{
    return __uint_as_float((stored_exponent + E4M3_PLACEMENT_EXPONENT) << FP32_EXPONENT_SHIFT);
}

/*
 * One warp, one tile of 16 output rows, scale groups first_group .. first_group + group_count - 1 of the input width.
 * Lane (g, t) ends with four sums: sums[0] and sums[1] are output row g for rows 2t and 2t + 1 of the batch, sums[2]
 * and sums[3] output row g + 8 for the same two; they go to shared memory, where the slices of a tile meet, as
 * [warp][lane][accumulator]. A lane whose g is not below row_count feeds zeros. input holds MAX_ROWS rows.
 */
__device__ __forceinline__ void tile_slice_product(const uint4 *tile_weights, const uint8_t *block_scales,
                                                   const __half *input, uint32_t input_size, uint32_t row_count,
                                                   uint32_t first_group, uint32_t group_count, float *partial_sums)
{
    float sums[ACCUMULATORS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t lane = threadIdx.x % LANES_PER_WARP;
    uint32_t lane_group = lane / LANES_PER_LANE_GROUP;
    uint32_t lane_in_group = lane % LANES_PER_LANE_GROUP;
    uint32_t last_group = first_group + group_count;
    bool carries_input = lane_group < row_count;          /* instruction column lane_group is that row of the batch */
    const uint32_t *input_pairs = (const uint32_t *)(input + (size_t)lane_group * input_size) + lane_in_group;
    const uint4 *lane_weights = tile_weights + lane;
    uint4 packed_ahead = lane_weights[(size_t)first_group * LANES_PER_WARP];
    for (uint32_t group = first_group; group < last_group; group++) {
        uint4 packed = packed_ahead;
        uint32_t words[WORDS_PER_LOAD] = {packed.x, packed.y, packed.z, packed.w};
        uint32_t inputs[MMA_STEPS_PER_SCALE_GROUP][REGISTERS_PER_OPERAND_B] = {{0, 0}, {0, 0}};
        float accumulator[ACCUMULATORS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
        float scale = group_scale(block_scales[group]);
        if (group + 1 < last_group) {
            packed_ahead = lane_weights[(size_t)(group + 1) * LANES_PER_WARP];    /* in flight during this group */
        }
        if (carries_input) {
            const uint32_t *group_pairs = input_pairs + (size_t)group * INPUT_PAIRS_PER_SCALE_GROUP;
            for (uint32_t step = 0; step < MMA_STEPS_PER_SCALE_GROUP; step++) {
                inputs[step][0] = group_pairs[step * INPUT_PAIRS_PER_STEP];
                inputs[step][1] = group_pairs[step * INPUT_PAIRS_PER_STEP + INPUT_PAIRS_PER_REGISTER];
            }
        }
        for (uint32_t step = 0; step < MMA_STEPS_PER_SCALE_GROUP; step++) {
            uint32_t weights[REGISTERS_PER_OPERAND_A];
            place_e4m3_word(words[step * WORDS_PER_STEP], &weights[0], &weights[1]);
            place_e4m3_word(words[step * WORDS_PER_STEP + 1], &weights[REGISTERS_PER_WORD],
                            &weights[REGISTERS_PER_WORD + 1]);
            tensor_core_m16n8k16(accumulator, weights, inputs[step][0], inputs[step][1]);
        }
        for (uint32_t accumulator_index = 0; accumulator_index < ACCUMULATORS_PER_LANE; accumulator_index++) {
            /* the scale is a power of two: the product is exact, fused or not */
            sums[accumulator_index] = fmaf(accumulator[accumulator_index], scale, sums[accumulator_index]);
        }
    }
    for (uint32_t accumulator_index = 0; accumulator_index < ACCUMULATORS_PER_LANE; accumulator_index++) {
        partial_sums[threadIdx.x * ACCUMULATORS_PER_LANE + accumulator_index] = sums[accumulator_index];
    }
}

/* One element of one matrix: the partial sums of its slices (warps first_warp ..), added in warp order. */
__device__ __forceinline__ float sum_slices_in_order(const float *partial_sums, uint32_t first_warp,
                                                     uint32_t slice_count, uint32_t lane, uint32_t accumulator_index)
{
    float sum = 0.0f;
    for (uint32_t slice = 0; slice < slice_count; slice++) {
        sum += partial_sums[((first_warp + slice) * LANES_PER_WARP + lane) * ACCUMULATORS_PER_LANE + accumulator_index];
    }
    return sum;
}

/*
 * y = W x for up to eight rows: grid (tiles of 16 output rows), slice_count warps per block, one slice of the
 * input width each, slice_count * SHARED_BYTES_PER_WARP of shared memory. input and output are [row][size].
 */
__global__ void linear_kernel(const uint4 *weights, const uint8_t *scales, const __half *input, float *output,
                              uint32_t output_size, uint32_t input_size, uint32_t row_count, uint32_t slice_count)
{
    extern __shared__ float partial_sums[];
    uint32_t tile = blockIdx.x;
    uint32_t slice = threadIdx.x / LANES_PER_WARP;
    uint32_t lane = threadIdx.x % LANES_PER_WARP;
    uint32_t group_count = input_size / WEIGHTS_PER_SCALE_GROUP;
    uint32_t groups_per_slice = group_count / slice_count;
    tile_slice_product(weights + (size_t)tile * group_count * LANES_PER_WARP,
                       scales + (size_t)(tile / TILES_PER_SCALE_BLOCK) * group_count, input, input_size, row_count,
                       slice * groups_per_slice, groups_per_slice, partial_sums);
    __syncthreads();
    if (slice == 0) {
        for (uint32_t accumulator_index = 0; accumulator_index < ACCUMULATORS_PER_LANE; accumulator_index++) {
            uint32_t batch_row = (lane % LANES_PER_LANE_GROUP) * BATCH_COLUMNS_PER_LANE
                                 + accumulator_index % BATCH_COLUMNS_PER_LANE;
            uint32_t output_row = tile * MMA_ROWS + lane / LANES_PER_LANE_GROUP
                                  + (accumulator_index / BATCH_COLUMNS_PER_LANE) * MMA_HALF_ROWS;
            if (batch_row < row_count) {
                output[(size_t)batch_row * output_size + output_row] =
                    sum_slices_in_order(partial_sums, 0, slice_count, lane, accumulator_index);
            }
        }
    }
}

/*
 * gate and up of the shared expert, and its hidden vector: grid (tiles of the inner width), 2 * slice_count warps
 * per block: warps 0 .. slice_count - 1 take the slices of the gate matrix, the others those of the up matrix.
 * Writes h = silu(min(gate, L)) * clamp(up, -L, L) as fp16, [row][inner size].
 */
__global__ void gate_up_kernel(const uint4 *gate_weights, const uint4 *up_weights, const uint8_t *gate_scales,
                               const uint8_t *up_scales, const __half *input, __half *hidden, uint32_t inner_size,
                               uint32_t input_size, uint32_t row_count, uint32_t slice_count)
{
    extern __shared__ float partial_sums[];
    uint32_t tile = blockIdx.x;
    uint32_t warp = threadIdx.x / LANES_PER_WARP;
    uint32_t lane = threadIdx.x % LANES_PER_WARP;
    bool reads_up = warp >= slice_count;
    uint32_t slice = reads_up ? warp - slice_count : warp;
    uint32_t group_count = input_size / WEIGHTS_PER_SCALE_GROUP;
    uint32_t groups_per_slice = group_count / slice_count;
    size_t tile_loads = (size_t)tile * group_count * LANES_PER_WARP;
    size_t block_scales = (size_t)(tile / TILES_PER_SCALE_BLOCK) * group_count;
    tile_slice_product((reads_up ? up_weights : gate_weights) + tile_loads,
                       (reads_up ? up_scales : gate_scales) + block_scales, input, input_size, row_count,
                       slice * groups_per_slice, groups_per_slice, partial_sums);
    __syncthreads();
    if (warp == 0) {
        for (uint32_t accumulator_index = 0; accumulator_index < ACCUMULATORS_PER_LANE; accumulator_index++) {
            uint32_t batch_row = (lane % LANES_PER_LANE_GROUP) * BATCH_COLUMNS_PER_LANE
                                 + accumulator_index % BATCH_COLUMNS_PER_LANE;
            uint32_t output_row = tile * MMA_ROWS + lane / LANES_PER_LANE_GROUP
                                  + (accumulator_index / BATCH_COLUMNS_PER_LANE) * MMA_HALF_ROWS;
            if (batch_row < row_count) {
                float gate_value = sum_slices_in_order(partial_sums, 0, slice_count, lane, accumulator_index);
                float up_value = sum_slices_in_order(partial_sums, slice_count, slice_count, lane, accumulator_index);
                gate_value = fminf(gate_value, SWIGLU_LIMIT);
                up_value = fminf(fmaxf(up_value, -SWIGLU_LIMIT), SWIGLU_LIMIT);
                float silu = gate_value / (1.0f + expf(-gate_value));
                hidden[(size_t)batch_row * inner_size + output_row] = __float2half(silu * up_value);
            }
        }
    }
}

/* ---- host side: the formats, decoded directly ---- */

static float half_bits_to_float(uint16_t bits)
{
    int exponent = (bits >> FP16_EXPONENT_SHIFT) & FP16_EXPONENT_MASK;
    int mantissa = bits & FP16_MANTISSA_MASK;
    double value = exponent == 0 ? ldexp((double)mantissa, FP16_SUBNORMAL_EXPONENT)
                                 : ldexp((double)(mantissa + FP16_IMPLICIT_ONE), exponent - FP16_EXPONENT_OFFSET);
    return (float)((bits & FP16_SIGN_BIT) ? -value : value);
}

static uint16_t float_to_half_bits(float value)
{
    __half half = __float2half(value);                    /* on the host too it rounds to nearest even */
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

/* E4M3 decoded from its fields: (8 + mantissa) * 2^(exponent - 10), or mantissa * 2^-9 where the exponent is 0. */
static double e4m3_value(uint32_t code)
{
    uint32_t exponent = (code >> E4M3_EXPONENT_SHIFT) & E4M3_EXPONENT_MASK;
    uint32_t mantissa = code & E4M3_MANTISSA_MASK;
    uint32_t scaled = exponent == 0 ? mantissa << 1 : (E4M3_IMPLICIT_ONE + mantissa) << exponent;
    double magnitude = (double)scaled / E4M3_VALUE_DIVISOR;
    return (code & E4M3_SIGN_BIT) ? -magnitude : magnitude;
}

/*
 * The claim the kernels rest on: for every code that is not a NaN, at each of the four byte positions of a stored
 * word, the placed fp16 is the E4M3 value times 2^-8. The three other bytes of the word hold the code's
 * complement, so a mask that let a neighbour's bit through would show.
 */
static int placement_matches_direct_decode(void)
{
    for (uint32_t code = 0; code < E4M3_CODES; code++) {
        uint32_t neighbour = ~code & (E4M3_CODES - 1u);
        double expected = ldexp(e4m3_value(code), -E4M3_PLACEMENT_EXPONENT);
        if ((code & E4M3_MAGNITUDE_MASK) == E4M3_MAGNITUDE_MASK) {
            continue;
        }
        for (uint32_t byte_in_word = 0; byte_in_word < BYTES_PER_WORD; byte_in_word++) {
            uint32_t registers[REGISTERS_PER_WORD];
            uint32_t word = 0;
            for (uint32_t position = 0; position < BYTES_PER_WORD; position++) {
                word |= (position == byte_in_word ? code : neighbour) << (position * BITS_PER_BYTE);
            }
            place_e4m3_word(word, &registers[0], &registers[1]);
            uint32_t placed = (registers[byte_in_word % REGISTERS_PER_WORD]
                               >> (byte_in_word / REGISTERS_PER_WORD * BITS_PER_HALF)) & FP16_HALF_MASK;
            if ((double)half_bits_to_float((uint16_t)placed) != expected) {
                fprintf(stderr, "E4M3 code 0x%02x in byte %u of a word is placed as fp16 0x%04x: not its value / 256\n",
                        code, byte_in_word, placed);
                return 0;
            }
        }
    }
    return 1;
}

/* ---- host side: the table, its sizes and its launch units ---- */

static int layer_table_is_usable(void)
{
    for (uint32_t matrix_index = 0; matrix_index < LAYER_MATRICES; matrix_index++) {
        const struct matrix_shape *shape = &layer_matrices[matrix_index];
        if (shape->name == NULL || shape->output_size == 0 || shape->input_size == 0
            || shape->output_size % SCALE_BLOCK_ROWS != 0 || shape->input_size % WEIGHTS_PER_SCALE_GROUP != 0
            || shape->output_size > LARGEST_OUTPUT_SIZE || shape->input_size > LARGEST_INPUT_SIZE) {
            return 0;
        }
    }
    return layer_matrices[GATE_MATRIX].output_size == layer_matrices[UP_MATRIX].output_size
           && layer_matrices[GATE_MATRIX].input_size == layer_matrices[UP_MATRIX].input_size
           && layer_matrices[DOWN_MATRIX].input_size == layer_matrices[GATE_MATRIX].output_size;
}

static uint32_t matrix_seed(uint32_t matrix_index)        /* of its weights; its input's is LAYER_MATRICES further */
{
    return mix_bits(matrix_index + 1u);
}

static size_t weight_bytes(uint32_t matrix_index)
{
    return (size_t)layer_matrices[matrix_index].output_size * layer_matrices[matrix_index].input_size;
}

static size_t scale_entries(uint32_t matrix_index)
{
    return (size_t)(layer_matrices[matrix_index].output_size / SCALE_BLOCK_ROWS)
           * (layer_matrices[matrix_index].input_size / WEIGHTS_PER_SCALE_GROUP);
}

static uint32_t unit_matrix(uint32_t unit)                /* for gate with up: the gate, whose shape the up shares */
{
    return unit == DOWN_UNIT ? DOWN_MATRIX : unit;
}

static uint32_t unit_matrix_count(uint32_t unit)
{
    return unit == GATE_UP_UNIT ? MATRICES_PER_GATE_UP_LAUNCH : 1;
}

static int slice_count_fits(uint32_t unit, uint32_t slice_count)
{
    uint32_t group_count = layer_matrices[unit_matrix(unit)].input_size / WEIGHTS_PER_SCALE_GROUP;
    return slice_count >= 1 && slice_count <= MAX_INPUT_SLICES && group_count % slice_count == 0
           && unit_matrix_count(unit) * slice_count * LANES_PER_WARP <= MAX_THREADS_PER_BLOCK;
}

/* Copies of a matrix in memory: as many as make one pass over them read ring_bytes; gate and up fill one ring. */
static uint32_t ring_copies(uint32_t matrix_index, size_t ring_bytes)
{
    int shares_a_launch = matrix_index == GATE_MATRIX || matrix_index == UP_MATRIX;
    size_t bytes_per_launch = weight_bytes(matrix_index) * (shares_a_launch ? MATRICES_PER_GATE_UP_LAUNCH : 1);
    size_t copies = (ring_bytes + bytes_per_launch - 1) / bytes_per_launch;
    return copies < 1 ? 1u : (uint32_t)copies;
}

/* ---- host side: the reference ---- */

/*
 * Output rows that are compared. The first half walks the first rows of the output: a megabyte from the start of
 * any matrix of the table covers at least its first 128 output rows, so these all read what --corrupt overwrites.
 * The second half spreads over the whole output and ends on its last row.
 */
static uint32_t sampled_output_row(uint32_t sample, uint32_t output_size)
{
    if (sample < LEADING_SAMPLES) {
        return sample * LEADING_SAMPLE_STRIDE % output_size;
    }
    uint64_t spread_step = sample - LEADING_SAMPLES + 1;          /* 1 .. the number of spread samples */
    return (uint32_t)((output_size - 1) * spread_step / (SAMPLED_OUTPUTS - LEADING_SAMPLES));
}

/* One row of the batch of a matrix's synthetic input, as the card holds it. */
static void host_input(uint32_t matrix_index, uint32_t batch_row, double *input)
{
    uint32_t input_size = layer_matrices[matrix_index].input_size;
    for (uint32_t column = 0; column < input_size; column++) {
        float value = input_value(matrix_seed(matrix_index + LAYER_MATRICES), batch_row, column, input_size);
        input[column] = (double)half_bits_to_float(float_to_half_bits(value));
    }
}

/* One output element in double: every byte decoded directly, every group's sum times its scale. */
static double host_row_product(uint32_t matrix_index, uint32_t output_row, const double *input)
{
    uint32_t seed = matrix_seed(matrix_index);
    uint32_t input_size = layer_matrices[matrix_index].input_size;
    uint32_t group_count = input_size / WEIGHTS_PER_SCALE_GROUP;
    double sum = 0.0;
    for (uint32_t group = 0; group < group_count; group++) {
        int exponent = (int)scale_exponent(seed, output_row / SCALE_BLOCK_ROWS, group, group_count) - E8M0_BIAS;
        double group_sum = 0.0;
        for (uint32_t offset = 0; offset < WEIGHTS_PER_SCALE_GROUP; offset++) {
            uint32_t input_column = group * WEIGHTS_PER_SCALE_GROUP + offset;
            group_sum += e4m3_value(weight_byte(seed, output_row, input_column, input_size)) * input[input_column];
        }
        sum += group_sum * ldexp(1.0, exponent);
    }
    return sum;
}

/*
 * The exact sampled outputs of one check unit for one row of the batch. Inputs do not change from launch to
 * launch, so a row's reference is computed once and kept (the shared expert's needs its whole hidden vector,
 * rounded to fp16 the way the kernel stores it).
 */
static const double *host_reference(uint32_t check_unit, uint32_t batch_row)
{
    static double kept[CHECK_UNITS][MAX_ROWS][SAMPLED_OUTPUTS];
    static int is_kept[CHECK_UNITS][MAX_ROWS];
    static double input[LARGEST_INPUT_SIZE];
    static double hidden[LARGEST_INPUT_SIZE];
    uint32_t last_matrix = check_unit == SHARED_EXPERT_CHECK ? DOWN_MATRIX : check_unit;
    const double *last_input = input;
    double *reference = kept[check_unit][batch_row];
    if (is_kept[check_unit][batch_row]) {
        return reference;
    }
    host_input(check_unit, batch_row, input);             /* the chain's check unit has the number of the gate matrix */
    if (check_unit == SHARED_EXPERT_CHECK) {
        for (uint32_t inner = 0; inner < layer_matrices[GATE_MATRIX].output_size; inner++) {
            double gate_value = fmin(host_row_product(GATE_MATRIX, inner, input), (double)SWIGLU_LIMIT);
            double up_value = fmin(fmax(host_row_product(UP_MATRIX, inner, input), -(double)SWIGLU_LIMIT),
                                   (double)SWIGLU_LIMIT);
            double value = gate_value / (1.0 + exp(-gate_value)) * up_value;
            hidden[inner] = (double)half_bits_to_float(float_to_half_bits((float)value));
        }
        last_input = hidden;
    }
    for (uint32_t sample = 0; sample < SAMPLED_OUTPUTS; sample++) {
        uint32_t output_row = sampled_output_row(sample, layer_matrices[last_matrix].output_size);
        reference[sample] = host_row_product(last_matrix, output_row, last_input);
    }
    is_kept[check_unit][batch_row] = 1;
    return reference;
}

/* ---- host side: one card ---- */

struct matrix_on_card {
    uint4 *weights;                              /* [copy][tile][scale group][lane]: 16 bytes */
    uint8_t *scales;                             /* [copy][scale block][scale group]: one E8M0 code */
    __half *input;                               /* [MAX_ROWS][input size]; the down projection's is the hidden */
    float *output;                               /* [MAX_ROWS][output size] */
    uint32_t copy_count;
};

struct card_state {
    struct matrix_on_card matrices[LAYER_MATRICES];
    cudaEvent_t started;
    cudaEvent_t stopped;
};

struct check_tally {
    uint32_t comparisons;
    uint32_t within_tolerance;
    double worst_ratio;                          /* of the comparison whose error is largest against its rms */
    double worst_error;
    double worst_rms;
};

/* One launch of a unit on copy launch_index % copies of its matrices. */
static void launch_unit(const struct card_state *card, uint32_t unit, uint32_t launch_index, uint32_t row_count,
                        uint32_t slice_count)
{
    uint32_t matrix_index = unit_matrix(unit);
    const struct matrix_shape *shape = &layer_matrices[matrix_index];
    const struct matrix_on_card *matrix = &card->matrices[matrix_index];
    uint32_t copy = launch_index % matrix->copy_count;
    size_t load_offset = copy * (weight_bytes(matrix_index) / sizeof(uint4));
    size_t scale_offset = copy * scale_entries(matrix_index);
    unsigned int tile_count = shape->output_size / MMA_ROWS;
    unsigned int thread_count = unit_matrix_count(unit) * slice_count * LANES_PER_WARP;
    size_t shared_bytes = unit_matrix_count(unit) * slice_count * SHARED_BYTES_PER_WARP;
    if (unit == GATE_UP_UNIT) {
        const struct matrix_on_card *up = &card->matrices[UP_MATRIX];
        gate_up_kernel<<<tile_count, thread_count, shared_bytes>>>(
            matrix->weights + load_offset, up->weights + load_offset, matrix->scales + scale_offset,
            up->scales + scale_offset, matrix->input, card->matrices[DOWN_MATRIX].input, shape->output_size,
            shape->input_size, row_count, slice_count);
    } else {
        linear_kernel<<<tile_count, thread_count, shared_bytes>>>(
            matrix->weights + load_offset, matrix->scales + scale_offset, matrix->input, matrix->output,
            shape->output_size, shape->input_size, row_count, slice_count);
    }
}

/* Mean microseconds of a step, unit_count units launched in a row: events around all the steps after the warm-up. */
static double time_units(const struct card_state *card, uint32_t first_unit, uint32_t unit_count,
                         const uint32_t *slice_counts, uint32_t row_count, uint32_t step_count)
{
    float milliseconds = 0.0f;
    for (uint32_t step = 0; step < WARM_UP_STEPS + step_count; step++) {
        if (step == WARM_UP_STEPS) {
            CHECK_CUDA(cudaEventRecord(card->started));
        }
        for (uint32_t unit = first_unit; unit < first_unit + unit_count; unit++) {
            launch_unit(card, unit, step, row_count, slice_counts[unit]);
        }
    }
    CHECK_CUDA(cudaEventRecord(card->stopped));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(card->stopped));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, card->started, card->stopped));
    return (double)milliseconds * MICROSECONDS_PER_MILLISECOND / step_count;
}

/* Compares the sampled outputs of the first and the last row of the batch with the reference. */
static void compare_with_reference(const float *card_output, uint32_t check_unit, uint32_t row_count,
                                   struct check_tally *tally)
{
    static float card_row[LARGEST_OUTPUT_SIZE];
    uint32_t output_size = layer_matrices[check_unit == SHARED_EXPERT_CHECK ? DOWN_MATRIX : check_unit].output_size;
    uint32_t row_stride = row_count > 1 ? row_count - 1 : 1u;      /* rows 0 and row_count - 1, once if they are one */
    for (uint32_t batch_row = 0; batch_row < row_count; batch_row += row_stride) {
        const double *reference = host_reference(check_unit, batch_row);
        double squared_sum = 0.0, worst_error = 0.0;
        CHECK_CUDA(cudaMemcpy(card_row, card_output + (size_t)batch_row * output_size, output_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (uint32_t sample = 0; sample < SAMPLED_OUTPUTS; sample++) {
            double error = fabs((double)card_row[sampled_output_row(sample, output_size)] - reference[sample]);
            squared_sum += reference[sample] * reference[sample];
            if (error != error || error > worst_error) {  /* a NaN from the card stays the worst error */
                worst_error = error;
            }
        }
        double root_mean_square = sqrt(squared_sum / SAMPLED_OUTPUTS);
        double ratio = worst_error / root_mean_square;
        tally->within_tolerance += (worst_error <= RELATIVE_TOLERANCE * root_mean_square) ? 1u : 0u;
        if (tally->comparisons == 0 || ratio != ratio || ratio > tally->worst_ratio) {
            tally->worst_ratio = ratio;
            tally->worst_error = worst_error;
            tally->worst_rms = root_mean_square;
        }
        tally->comparisons++;
    }
}

/*
 * Every check unit with every slice count that fits it, against the reference; one line per unit. Returns the
 * number of units that did not come out as required: every comparison within tolerance, or, for a unit that reads
 * the megabyte --corrupt overwrote, none.
 */
static int check_row_count(const struct card_state *card, uint32_t row_count, int corrupt_weights)
{
    __half *hidden = card->matrices[DOWN_MATRIX].input;
    size_t hidden_bytes = (size_t)MAX_ROWS * layer_matrices[DOWN_MATRIX].input_size * sizeof(__half);
    uint32_t corrupted_matrix = CORRUPTED_MATRIX;
    int failures = 0;
    for (uint32_t check_unit = 0; check_unit < CHECK_UNITS; check_unit++) {
        int is_chain = check_unit == SHARED_EXPERT_CHECK;         /* launch units gate with up, then down */
        uint32_t last_unit = is_chain ? DOWN_UNIT : check_unit;
        uint32_t output_matrix = unit_matrix(last_unit);
        float *card_output = card->matrices[output_matrix].output;
        size_t output_bytes = (size_t)MAX_ROWS * layer_matrices[output_matrix].output_size * sizeof(float);
        int reads_corrupted = is_chain ? corrupted_matrix >= GATE_MATRIX : corrupted_matrix == check_unit;
        struct check_tally tally;
        memset(&tally, 0, sizeof(tally));
        for (uint32_t candidate = 0; candidate < SLICE_CANDIDATES; candidate++) {
            uint32_t slice_count = slice_candidates[candidate];
            if (!slice_count_fits(check_unit, slice_count) && !slice_count_fits(last_unit, slice_count)) {
                continue;
            }
            /* NaN everywhere first: an element that a launch does not write cannot pass */
            CHECK_CUDA(cudaMemset(card_output, STALE_BYTE_VALUE, output_bytes));
            if (is_chain) {
                CHECK_CUDA(cudaMemset(hidden, STALE_BYTE_VALUE, hidden_bytes));
            }
            for (uint32_t unit = check_unit; unit <= last_unit; unit++) {
                /* other copies of a ring are checked too, except that --corrupt overwrote the first */
                launch_unit(card, unit, corrupt_weights ? 0u : candidate, row_count,
                            slice_count_fits(unit, slice_count) ? slice_count : slice_candidates[0]);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            compare_with_reference(card_output, check_unit, row_count, &tally);
        }
        const char *verdict = tally.within_tolerance == tally.comparisons ? "matches the reference"
                              : tally.within_tolerance == 0 ? "DIFFERS" : "DIFFERS in part";
        printf("  %u row(s), %-21s: %u of %u comparisons within tolerance, worst error %.3e against rms %.3e: %s\n",
               row_count, is_chain ? "shared expert" : layer_matrices[check_unit].name, tally.within_tolerance,
               tally.comparisons, tally.worst_error, tally.worst_rms, verdict);
        if (tally.comparisons == 0
            || tally.within_tolerance != (corrupt_weights && reads_corrupted ? 0u : tally.comparisons)) {
            failures++;
        }
    }
    return failures;
}

/* One line for a sequence of the layer, beside today's figure for the card's architecture where there is one. */
static void print_layer_line(uint32_t row_count, const char *name, double microseconds, double bytes_read,
                             const cudaDeviceProp *properties, int today_on_sm_80, int today_on_sm_120)
{
    int today = properties->minor != 0 ? 0 : properties->major == SM_80_MAJOR ? today_on_sm_80
                : properties->major == SM_120_MAJOR ? today_on_sm_120 : 0;
    printf("  %u row(s), %s: %8.1f us per layer, %6.1f MB read, %6.0f GB/s, today's kernels at one row: ", row_count,
           name, microseconds, bytes_read / BYTES_PER_MB,
           bytes_read / BYTES_PER_GB / (microseconds / MICROSECONDS_PER_SECOND));
    if (today > 0) {
        printf("%d us\n", today);
    } else {
        printf("no figure\n");
    }
}

/* Every unit with every slice count that fits it, then the layer's two sequences with the fastest of each. */
static void measure_row_count(const struct card_state *card, const cudaDeviceProp *properties, uint32_t row_count,
                              uint32_t step_count)
{
    uint32_t best_slices[LAUNCH_UNITS] = {0};
    double sequence_bytes[LAYER_SEQUENCES] = {0.0, 0.0};          /* of the trunk linears, of the shared expert */
    for (uint32_t unit = 0; unit < LAUNCH_UNITS; unit++) {
        uint32_t matrix_index = unit_matrix(unit);
        const struct matrix_shape *shape = &layer_matrices[matrix_index];
        size_t matrix_bytes = weight_bytes(matrix_index) + scale_entries(matrix_index);
        double bytes_read = (double)(unit_matrix_count(unit) * matrix_bytes);
        double tried_microseconds[SLICE_CANDIDATES];
        double best_microseconds = 0.0;
        uint32_t best_slice_count = 0;
        const char *separator = "";
        for (uint32_t candidate = 0; candidate < SLICE_CANDIDATES; candidate++) {
            tried_microseconds[candidate] = 0.0;
            if (!slice_count_fits(unit, slice_candidates[candidate])) {
                continue;
            }
            best_slices[unit] = slice_candidates[candidate];
            tried_microseconds[candidate] = time_units(card, unit, 1, best_slices, row_count, step_count);
            if (best_slice_count == 0 || tried_microseconds[candidate] < best_microseconds) {
                best_microseconds = tried_microseconds[candidate];
                best_slice_count = slice_candidates[candidate];
            }
        }
        best_slices[unit] = best_slice_count;
        printf("  %u row(s), %-21s %5u x %4u: %u slice(s), %8.1f us per launch, %6.1f MB read, %6.0f GB/s (", row_count,
               unit == GATE_UP_UNIT ? "shared expert gate+up" : shape->name, shape->output_size, shape->input_size,
               best_slice_count, best_microseconds, bytes_read / BYTES_PER_MB,
               bytes_read / BYTES_PER_GB / (best_microseconds / MICROSECONDS_PER_SECOND));
        for (uint32_t candidate = 0; candidate < SLICE_CANDIDATES; candidate++) {
            if (slice_count_fits(unit, slice_candidates[candidate])) {
                printf("%s%u: %.1f", separator, slice_candidates[candidate], tried_microseconds[candidate]);
                separator = ", ";
            }
        }
        printf(" us)\n");
        sequence_bytes[unit < GATE_UP_UNIT ? 0 : 1] += bytes_read;
    }
    double trunk_microseconds = time_units(card, 0, ATTENTION_MATRICES, best_slices, row_count, step_count);
    double shared_expert_microseconds = time_units(card, GATE_UP_UNIT, SHARED_EXPERT_UNITS, best_slices, row_count,
                                                   step_count);
    print_layer_line(row_count, "trunk linears", trunk_microseconds, sequence_bytes[0], properties,
                     TODAY_TRUNK_MICROSECONDS_SM_80, TODAY_TRUNK_MICROSECONDS_SM_120);
    print_layer_line(row_count, "shared expert", shared_expert_microseconds, sequence_bytes[1], properties,
                     TODAY_SHARED_EXPERT_MICROSECONDS_SM_80, TODAY_SHARED_EXPERT_MICROSECONDS_SM_120);
}

static int probe_card(int card_index, uint32_t step_count, int corrupt_weights)
{
    cudaDeviceProp properties;
    struct card_state card;
    uint8_t *all_weights = NULL;
    uint8_t *all_scales = NULL;
    __half *all_inputs = NULL;
    float *all_outputs = NULL;
    size_t weight_total = 0, scale_total = 0, layer_bytes = 0, weight_offset = 0, scale_offset = 0;
    int failures = 0;
    CHECK_CUDA(cudaSetDevice(card_index));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, card_index));
    size_t l2_bytes = properties.l2CacheSize > 0 ? (size_t)properties.l2CacheSize : (size_t)FALLBACK_L2_BYTES;
    size_t ring_bytes = COLD_RING_L2_MULTIPLE * l2_bytes;
    unsigned int fill_blocks = (unsigned int)properties.multiProcessorCount * FILL_BLOCKS_PER_SM;
    for (uint32_t matrix_index = 0; matrix_index < LAYER_MATRICES; matrix_index++) {
        card.matrices[matrix_index].copy_count = ring_copies(matrix_index, ring_bytes);
        weight_total += card.matrices[matrix_index].copy_count * weight_bytes(matrix_index);
        scale_total += card.matrices[matrix_index].copy_count * scale_entries(matrix_index);
        layer_bytes += weight_bytes(matrix_index) + scale_entries(matrix_index);
    }
    /* one allocation of each kind; every matrix has a region of inputs and one of outputs, sized for the largest */
    CHECK_CUDA(cudaMalloc((void **)&all_weights, weight_total));
    CHECK_CUDA(cudaMalloc((void **)&all_scales, scale_total));
    CHECK_CUDA(cudaMalloc((void **)&all_inputs, LAYER_MATRICES * INPUT_REGION_HALVES * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void **)&all_outputs, LAYER_MATRICES * OUTPUT_REGION_FLOATS * sizeof(float)));
    CHECK_CUDA(cudaMemset(all_inputs, 0, LAYER_MATRICES * INPUT_REGION_HALVES * sizeof(__half)));
    CHECK_CUDA(cudaMemset(all_outputs, 0, LAYER_MATRICES * OUTPUT_REGION_FLOATS * sizeof(float)));
    CHECK_CUDA(cudaEventCreate(&card.started));
    CHECK_CUDA(cudaEventCreate(&card.stopped));
    for (uint32_t matrix_index = 0; matrix_index < LAYER_MATRICES; matrix_index++) {
        const struct matrix_shape *shape = &layer_matrices[matrix_index];
        struct matrix_on_card *matrix = &card.matrices[matrix_index];
        uint64_t words_per_copy = weight_bytes(matrix_index) / BYTES_PER_WORD;
        matrix->weights = (uint4 *)(all_weights + weight_offset);
        matrix->scales = all_scales + scale_offset;
        matrix->input = all_inputs + matrix_index * INPUT_REGION_HALVES;
        matrix->output = all_outputs + matrix_index * OUTPUT_REGION_FLOATS;
        weight_offset += matrix->copy_count * weight_bytes(matrix_index);
        scale_offset += matrix->copy_count * scale_entries(matrix_index);
        fill_weights_kernel<<<fill_blocks, FILL_THREADS_PER_BLOCK>>>(
            (uint32_t *)matrix->weights, matrix_seed(matrix_index), shape->input_size, words_per_copy,
            matrix->copy_count * words_per_copy);
        fill_scales_kernel<<<fill_blocks, FILL_THREADS_PER_BLOCK>>>(
            matrix->scales, matrix_seed(matrix_index), shape->input_size / WEIGHTS_PER_SCALE_GROUP,
            scale_entries(matrix_index), matrix->copy_count * scale_entries(matrix_index));
        if (matrix_index < UP_MATRIX) {                   /* the down projection's input is written by gate with up */
            fill_input_kernel<<<MAX_ROWS, FILL_THREADS_PER_BLOCK>>>(
                matrix->input, matrix_seed(matrix_index + LAYER_MATRICES), shape->input_size);
        }
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    if (corrupt_weights) {
        size_t matrix_bytes = weight_bytes(CORRUPTED_MATRIX);
        CHECK_CUDA(cudaMemset(card.matrices[CORRUPTED_MATRIX].weights, CORRUPTED_BYTE_VALUE,
                              matrix_bytes < CORRUPTED_BYTES ? matrix_bytes : CORRUPTED_BYTES));
    }
    printf("card %d %s (sm_%d%d, %d SMs): e4m3, %d matrices of %.2f MiB in all, rings of %.0f MiB (%d times the L2 "
           "cache), %.2f GiB in memory\n",
           card_index, properties.name, properties.major, properties.minor, properties.multiProcessorCount,
           LAYER_MATRICES, (double)layer_bytes / BYTES_PER_MIB, (double)ring_bytes / BYTES_PER_MIB,
           COLD_RING_L2_MULTIPLE, (double)(weight_total + scale_total) / BYTES_PER_GIB);
    for (size_t case_index = 0; case_index < ROW_COUNT_CASES; case_index++) {
        failures += check_row_count(&card, row_counts[case_index], corrupt_weights);
        if (corrupt_weights) {
            break;                                       /* the megabyte stays overwritten: one case shows it */
        }
        measure_row_count(&card, &properties, row_counts[case_index], step_count);
    }
    CHECK_CUDA(cudaEventDestroy(card.started));
    CHECK_CUDA(cudaEventDestroy(card.stopped));
    CHECK_CUDA(cudaFree(all_outputs));
    CHECK_CUDA(cudaFree(all_inputs));
    CHECK_CUDA(cudaFree(all_scales));
    CHECK_CUDA(cudaFree(all_weights));
    return failures;
}

int main(int argument_count, char **arguments)
{
    long step_count = DEFAULT_STEP_COUNT;
    int corrupt_weights = 0;
    int usage_is_wrong = 0;
    int card_count = 0;
    int failures = 0;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        const char *argument = arguments[argument_index];
        if (strcmp(argument, "--corrupt") == 0) {
            corrupt_weights = 1;
        } else if (strcmp(argument, "--steps") == 0 && argument_index + 1 < argument_count) {
            char *number_end = NULL;
            step_count = strtol(arguments[++argument_index], &number_end, DECIMAL_BASE);
            usage_is_wrong |= number_end == arguments[argument_index] || *number_end != '\0';
        } else {
            usage_is_wrong = 1;
        }
    }
    if (usage_is_wrong || step_count < 1 || step_count > MAX_STEP_COUNT) {
        fprintf(stderr, "usage: %s [--steps N] [--corrupt]\n", arguments[0]);
        return EXIT_FAILURE;
    }
    if (!layer_table_is_usable() || !placement_matches_direct_decode()) {
        fprintf(stderr, "the table of matrices or the placement of E4M3 codes is not what the kernels assume\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (card_count < 1) {
        fprintf(stderr, "no card to measure\n");
        return EXIT_FAILURE;
    }
    for (int card_index = 0; card_index < card_count && card_index < MAX_CARDS; card_index++) {
        failures += probe_card(card_index, (uint32_t)step_count, corrupt_weights);
    }
    if (failures != 0) {
        fprintf(stderr, "%d check(s) did not come out as required\n", failures);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
