/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * shared_rows_expert_probe: what the routed experts of one MoE layer cost at 1 to 8 rows (decode, and the verify
 * step of drafting) when a job is an expert with every row routed to it, and whether a row keeps its bits there.
 *
 * four_bit_expert_probe makes one job per (row, expert) and reads an expert's 17.9 MiB once per row: 8 rows cost
 * 534 us on the RTX PRO 6000 and 1,110 us on a CMP 170HX, against 316 us and 1,098 us for today's kernels. Here the
 * rows of a job are the eight columns of the tensor-core instruction's operand B: an expert is read once whatever
 * the number of its rows. Shapes are DeepSeek-V4.1-Flash's (hidden 5120, inner 2304, six experts per row); weights,
 * routes and inputs are synthetic (hashes of their position). Settled here, on every card:
 *   1. the cost at 1, 2, 4, 6 and 8 rows under three routings: --overlap 0 (every row has six experts of its own),
 *      50 (a row shares three of its six with the row before) and 100 (all rows route to the same six);
 *   2. row exactness, the precondition of lossless drafting: every row of a launch is launched again alone (a batch
 *      of one: same experts, same route weights) and its six expert outputs and their sum are compared bit for bit:
 *      0 words may differ. --break-exactness makes the order in which a block adds its slices depend on the number
 *      of rows in the job: differences must then be reported wherever rows share an expert, and none elsewhere;
 *   3. the reference's activation quantization (model.py linear(), kernel.py act_quant_kernel) before every product,
 *      and its casts to bf16: see quantize_kernel and expert_kernel. The quantization kernel is also timed alone;
 *   4. two ways to turn a stored nibble into an fp16: the prototype's bit placement and a 16-entry table of the
 *      placed patterns in shared memory (--decode; both by default): same bits required, the faster named per card.
 * Checked on every run, for two (row, expert) pairs of every case, each step on its own: the card's quantized
 * input, byte for byte, against a host quantizer written from the definition; the hidden vector against
 * double precision from that input; its quantization against the host's of the card's hidden vector; the output
 * against double precision from the card's quantized hidden vector (a rounding that falls the other way upstream
 * cannot fail a product downstream). --corrupt overwrites a megabyte of the first checked expert's gate matrix: its
 * hidden vector must then differ.
 *
 * The kernels are four_bit_expert_probe's with every column of operand B in use: tile_slice_product says what a lane
 * holds and where the scales are applied, sum_slices_in_order the order of additions, host_row_product where a
 * weight is stored. A row's six experts are added in slot order by a kernel of its own. No atomics anywhere.
 *
 *   shared_rows_expert_probe [--overlap 0|50|100] [--decode placement|table] [--experts N] [--steps N]
 *                            [--break-exactness] [--corrupt]
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN_SIZE 5120
#define EXPERT_INNER_SIZE 2304
#define EXPERTS_PER_ROW 6
#define MAX_ROWS 8                               /* rows of the batch one instruction carries (its N) */
#define MAX_PAIRS (MAX_ROWS * EXPERTS_PER_ROW)   /* a pair is a row at one of its experts: row * 6 + slot */
#define SWIGLU_LIMIT 10.0f
#define GATE_MATRIX 0                            /* an expert's matrices, in the order they are stored */
#define UP_MATRIX 1
#define DOWN_MATRIX 2
#define MATRIX_KINDS 3
#define MATRICES_PER_EXPERT_INPUT_SIDE 2         /* gate and up share the input */
#define MATRIX_WEIGHTS (HIDDEN_SIZE * EXPERT_INNER_SIZE)         /* of each of the three */
#define WEIGHTS_PER_SCALE_GROUP 32               /* inputs that share one scale: of the weights, of the activations */
#define MMA_ROWS 16                              /* outputs one block owns (the instruction's M) */
#define MMA_HALF_ROWS (MMA_ROWS / 2)
#define MMA_INPUTS 16                            /* inputs one instruction consumes (its K) */
#define MMA_HALF_INPUTS (MMA_INPUTS / 2)
#define MMA_STEPS_PER_LOAD 4                     /* one 16-byte load per lane = four instructions = 64 inputs */
#define INPUTS_PER_LOAD (MMA_INPUTS * MMA_STEPS_PER_LOAD)
#define MMA_STEPS_PER_SCALE_GROUP (WEIGHTS_PER_SCALE_GROUP / MMA_INPUTS)
#define SCALE_GROUPS_PER_LOAD (INPUTS_PER_LOAD / WEIGHTS_PER_SCALE_GROUP)
#define LANES_PER_WARP 32
#define LANES_PER_LANE_GROUP 4                   /* lanes that share an instruction row pair */
#define HALVES_PER_REGISTER 2
#define REGISTERS_PER_OPERAND_A 4
#define REGISTERS_PER_OPERAND_B 2
#define ELEMENTS_PER_LANE 4                      /* of the result: two outputs by two columns */
#define COLUMNS_PER_LANE 2
#define OUTPUTS_PER_LANE 2
#define WORDS_PER_SCALE_GROUP (LANES_PER_LANE_GROUP * MMA_STEPS_PER_SCALE_GROUP)     /* of a quantized vector */
#define INPUT_SLICES 4                           /* warps that share the input width of one tile of one matrix */
#define BYTES_PER_WORD 4
#define BITS_PER_BYTE 8
#define BYTE_MASK 0xFFu
#define BITS_PER_HALF 16
#define HALF_MASK 0xFFFFu
#define NIBBLE_BITS 4
#define NIBBLE_MASK 15u
#define NIBBLE_CODES 16u
/* a stored matrix: per (tile of 16 outputs, load of 64 inputs) 32 lanes of four packed words and 8 scale words */
#define TILE_LOADS_PER_MATRIX ((size_t)MATRIX_WEIGHTS / (MMA_ROWS * INPUTS_PER_LOAD))
#define PACKED_WORDS_PER_MATRIX (TILE_LOADS_PER_MATRIX * LANES_PER_WARP * MMA_STEPS_PER_LOAD)
#define SCALE_WORDS_PER_MATRIX (TILE_LOADS_PER_MATRIX * MMA_HALF_ROWS)
#define EXPERT_BYTES (MATRIX_KINDS * (PACKED_WORDS_PER_MATRIX + SCALE_WORDS_PER_MATRIX) * sizeof(uint32_t))
#define QUANTIZED_BYTES(size) ((size) + (size) / WEIGHTS_PER_SCALE_GROUP)   /* a vector's codes, then its exponents */
/* a checked launch's result on the host: [pair][hidden size] expert outputs, then [row][hidden size] sums */
#define ROW_EXPERT_FLOATS ((size_t)EXPERTS_PER_ROW * HIDDEN_SIZE)
#define ROW_SUMS_AT ((size_t)MAX_PAIRS * HIDDEN_SIZE)
#define RESULT_FLOATS (ROW_SUMS_AT + (size_t)MAX_ROWS * HIDDEN_SIZE)
/* e2m1: placing the three magnitude bits at fp16 bits 9..11 yields the value times 2^-14 (0 and 0.5: subnormals). */
#define E2M1_MAGNITUDE_MASK 0x0E000E00u
#define E2M1_SIGN_MASK 0x80008000u
#define E2M1_MAGNITUDE_BIT 9
#define E2M1_SIGN_BIT 12                         /* shift that moves nibble bit 3 to fp16 bit 15 */
#define E2M1_PLACEMENT_EXPONENT 14
/* E4M3: sign, four exponent bits (bias 7), three mantissa bits; exponent 0 is subnormal; the largest value is 448.
 * ((b & 0x80) << 8) | ((b & 0x7F) << 7) is the fp16 of the value times 2^-8, subnormal codes included. */
#define E4M3_CODES 256u
#define E4M3_SIGN_BIT 0x80u
#define E4M3_NAN_MAGNITUDE 0x7Fu
#define E4M3_EXPONENT_SHIFT 3
#define E4M3_EXPONENT_MASK 15u
#define E4M3_EXPONENT_BIAS 7u
#define E4M3_MANTISSA_MASK 7u
#define E4M3_IMPLICIT_ONE 8u                     /* 1.0 in units of the mantissa */
#define E4M3_VALUE_DIVISOR 1024.0                /* 2^(bias + mantissa bits) */
#define E4M3_PLACEMENT_EXPONENT 8
#define E4M3_PAIR_SIGN_MASK 0x00800080u          /* of bytes 0 and 2 of a word at once; bytes 1 and 3: one byte up */
#define E4M3_PAIR_MAGNITUDE_MASK 0x007F007Fu
#define E4M3_MAGNITUDE_SHIFT 7
#define E4M3_MAXIMUM 448.0f
#define E4M3_MAXIMUM_INVERSE (1.0f / E4M3_MAXIMUM)
#define E4M3_SMALLEST_NORMAL 0.015625f           /* 2^-6: codes below it are 2^-9 apart, as those just above it */
#define E4M3_SIGN_FROM_FP32_SHIFT 24             /* moves fp32's sign bit to the code's */
#define AMAX_FLOOR 1.0e-4f                       /* the reference's lower bound of a block's largest magnitude */
#define E8M0_BIAS 127                            /* a scale is 2^(code - 127) */
#define FP32_EXPONENT_SHIFT 23
#define FP32_EXPONENT_BIAS 127u
#define FP32_MANTISSA_MASK 0x007FFFFFu
/* quantizing a value of fp32 exponent field f: codes are 2^(f - 130) apart, the code is 8 * (f - 120) + steps - 8 */
#define E4M3_STEPS_EXPONENT_BASE (2u * FP32_EXPONENT_BIAS + E4M3_EXPONENT_SHIFT)
#define E4M3_CODE_BASE (((FP32_EXPONENT_BIAS - E4M3_EXPONENT_BIAS) << E4M3_EXPONENT_SHIFT) + E4M3_IMPLICIT_ONE)
/* an instruction's sum is the true one times 2^-(14 + 8): the exponent field of both scales and that factor */
#define PRODUCT_EXPONENT_BIAS (E8M0_BIAS - E2M1_PLACEMENT_EXPONENT - E4M3_PLACEMENT_EXPONENT)
#define BF16_ROUNDING_BIAS 0x7FFFu               /* plus the kept half's lowest bit: to nearest, ties to even */
#define BF16_KEPT_MASK 0xFFFF0000u
#define BF16_SIGNIFICANT_BITS 8
#define DECODE_PLACEMENT 0
#define DECODE_TABLE 1
#define DECODE_KINDS 2
#define PACKED_SEED 0x243F6A88u
#define SCALE_SEED 0x85A308D3u
#define SCALE_WORD_BASE 0x76767676u              /* synthetic weight scales: E8M0 codes 118 .. 121, 2^-9 .. 2^-6 */
#define SCALE_WORD_SPREAD 0x03030303u
#define INPUT_SEED 0x1234567u
#define STEP_STRIDE 977u
#define INPUT_VALUE_STEPS 255u                   /* synthetic inputs: multiples of 1/64 in -2 .. 2, */
#define INPUT_VALUE_OFFSET 128.0f
#define INPUT_VALUE_DIVISOR 64.0f
#define INPUT_BLOCK_KINDS 8u                     /* divided per block of 32 by 2^0 .. 2^6 so that scales differ, */
#define INPUT_TINY_SHIFT 16u                     /* or by 2^16 for one block in eight: below the reference's floor */
#define ROUTE_SEED 0xABCDEFu
#define ROUTE_WEIGHT_STEPS 32u                   /* synthetic route weights 1/64 .. 32/64 */
#define ROUTE_WEIGHT_DIVISOR 64.0f
#define OVERLAP_STEP 50                          /* overlaps 0, 50 and 100 percent */
#define PERCENT 100
#define DEFAULT_EXPERT_COUNT 128
#define MIN_EXPERT_COUNT 128                     /* two halves: each holds a step's experts at 12 places or more */
#define MAX_EXPERT_COUNT 512
#define DEFAULT_STEP_COUNT 200
#define MAX_STEP_COUNT 1000000
#define DECIMAL_BASE 10
#define WARM_UP_STEPS 10
#define CHECKED_STEP 0u                          /* the routes and inputs of this step are the checked ones */
#define CHECKED_PAIRS 2
#define DOWN_CHECK_STRIDE 13                     /* every 13th output is compared: all lanes, over the tiles */
#define CORRUPTED_BYTES (1024u * 1024u)          /* --corrupt overwrites this much of one gate matrix */
#define CORRUPTED_BYTE_VALUE 0x5A
#define STALE_BYTE_SHARED 0xFF                   /* the outputs hold this before a checked launch (NaN), and this */
#define STALE_BYTE_ALONE 0xEE                    /* before a row alone: a word left unwritten cannot match */
#define RELATIVE_TOLERANCE 2.0e-3
#define CASE_NAME_BYTES 48
#define TODAY_ONE_ROW_MICROSECONDS_SM_80 255     /* today's kernels: the routed experts of one layer */
#define TODAY_EIGHT_ROWS_MICROSECONDS_SM_80 1098
#define TODAY_ONE_ROW_MICROSECONDS_SM_120 99
#define TODAY_EIGHT_ROWS_MICROSECONDS_SM_120 316
#define SM_80_MAJOR 8
#define SM_120_MAJOR 12
#define BYTES_PER_MB 1.0e6
#define BYTES_PER_GB 1.0e9
#define BYTES_PER_MIB (1024.0 * 1024.0)
#define MICROSECONDS_PER_MILLISECOND 1000.0
#define MICROSECONDS_PER_SECOND 1.0e6
#define MAX_CARDS 16
#define FILL_THREADS_PER_BLOCK 256
#define FILL_BLOCKS_PER_SM 8

static const uint32_t row_counts[] = {1, 2, 4, 6, 8};
#define ROW_COUNT_CASES (sizeof(row_counts) / sizeof(row_counts[0]))
static const char *const decode_names[DECODE_KINDS] = {"placement", "table"};

/* One expert with the rows routed to it, which are a block's instruction columns. */
struct expert_job {
    uint32_t expert_index;
    uint32_t row_count;                          /* columns in use, 1 .. MAX_ROWS */
    uint32_t pair_index[MAX_ROWS];               /* of each column; a column not in use repeats column 0 */
    float route_weight[MAX_ROWS];
};

/* The vectors in front of a product and their quantization: the rows' inputs, or the pairs' hidden vectors. */
struct quantized_vectors {
    float *values;                               /* [vector][size]: inputs exact in fp16; hidden vectors, cast to bf16 */
    uint32_t *words;                             /* the E4M3 bytes: [vector][load][lane position]: 16 bytes */
    uint8_t *exponents;                          /* [vector][block of 32]: E8M0 codes */
    uint32_t size;
    uint32_t pairs_per_vector;                   /* six pairs read a row's input, one its own hidden vector */
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

/* ---- synthetic data and layouts: the same functions on the host and on the card ---- */

__host__ __device__ inline uint32_t mix_bits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

/* A stored word is a hash of its place in its buffer: eight weight nibbles, or the E8M0 codes of four scales. */
__host__ __device__ inline uint32_t stored_word(uint64_t word_index, uint32_t is_scale)
{
    uint32_t bits = mix_bits((is_scale != 0u ? SCALE_SEED : PACKED_SEED) + (uint32_t)word_index);
    return is_scale != 0u ? SCALE_WORD_BASE + (bits & SCALE_WORD_SPREAD) : bits;
}

/* One element of a row's input (see the INPUT_ constants): every value is exact in fp16, and so in bf16 and fp32. */
__host__ __device__ inline float input_value(uint32_t row, uint32_t column, uint32_t step)
{
    uint32_t row_seed = INPUT_SEED + step * STEP_STRIDE + row * HIDDEN_SIZE;
    uint32_t block_kind = mix_bits(~row_seed + column / WEIGHTS_PER_SCALE_GROUP) % INPUT_BLOCK_KINDS;
    uint32_t shift = block_kind < INPUT_BLOCK_KINDS - 1u ? block_kind : INPUT_TINY_SHIFT;
    float value = ((float)(mix_bits(row_seed + column) & INPUT_VALUE_STEPS) - INPUT_VALUE_OFFSET) / INPUT_VALUE_DIVISOR;
    return value / (float)(1u << shift);
}

/* The order of a quantized vector's bytes. Lane position t of the column that carries the vector needs, for each
 * instruction, inputs 2t and 2t + 1 (register 0 of operand B, low half first) and 2t + 8 and 2t + 9 (register 1).
 * Its four bytes are one word: bytes 0 and 2 are register 0, bytes 1 and 3 register 1. activation_offset is the
 * place of a byte's value in its block of 32; the words are stored as [load of 64 values][t][instruction]. */
__host__ __device__ inline uint32_t activation_offset(uint32_t step_in_block, uint32_t lane_position, uint32_t byte_in_word)
{
    return step_in_block * MMA_INPUTS + lane_position * HALVES_PER_REGISTER + byte_in_word / REGISTERS_PER_OPERAND_B
           + (byte_in_word % REGISTERS_PER_OPERAND_B) * MMA_HALF_INPUTS;
}

__host__ __device__ inline uint32_t activation_word_index(uint32_t block, uint32_t lane_position, uint32_t step_in_block)
{
    return ((block / SCALE_GROUPS_PER_LOAD) * LANES_PER_LANE_GROUP + lane_position) * MMA_STEPS_PER_LOAD
           + (block % SCALE_GROUPS_PER_LOAD) * MMA_STEPS_PER_SCALE_GROUP + step_in_block;
}

__global__ void fill_words_kernel(uint32_t *words, uint32_t is_scale, uint64_t word_count)
{
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < word_count; index += thread_total) {
        words[index] = stored_word(index, is_scale);
    }
}

/* One block per row of the launch: slot blockIdx.x of the batch holds row first_row + blockIdx.x. */
__global__ void fill_input_kernel(float *layer_input, uint32_t first_row, uint32_t step)
{
    for (uint32_t column = threadIdx.x; column < HIDDEN_SIZE; column += blockDim.x) {
        layer_input[(size_t)blockIdx.x * HIDDEN_SIZE + column] = input_value(first_row + blockIdx.x, column, step);
    }
}

/* ---- the kernels under test ---- */

/* An fp32 rounded to bf16, its upper half, to nearest with ties to even: the value the cast to bf16 keeps. */
__device__ __forceinline__ float bf16_value(float value)
{
    uint32_t bits = __float_as_uint(value);
    return __uint_as_float((bits + BF16_ROUNDING_BIAS + ((bits >> BITS_PER_HALF) & 1u)) & BF16_KEPT_MASK);
}

/* The E4M3 code nearest to value * inverse_scale, clamped to +-448, ties to the even code. */
__device__ __forceinline__ uint32_t quantize_to_e4m3(float value, float inverse_scale)
{
    float scaled = value * inverse_scale;                         /* by a power of two: exact */
    float magnitude = fminf(fabsf(scaled), E4M3_MAXIMUM);
    uint32_t exponent_field = __float_as_uint(fmaxf(magnitude, E4M3_SMALLEST_NORMAL)) >> FP32_EXPONENT_SHIFT;
    float steps_per_unit = __uint_as_float((E4M3_STEPS_EXPONENT_BASE - exponent_field) << FP32_EXPONENT_SHIFT);
    uint32_t steps = (uint32_t)__float2int_rn(magnitude * steps_per_unit);      /* 0 .. 16, ties to even */
    return ((exponent_field << E4M3_EXPONENT_SHIFT) + steps - E4M3_CODE_BASE)
           | ((__float_as_uint(scaled) >> E4M3_SIGN_FROM_FP32_SHIFT) & E4M3_SIGN_BIT);
}

/* The reference's act_quant for blocks of 32: grid (vectors), one thread per block. The largest magnitude of the
 * block (at least 1e-4) times 1 / 448 in fp32, rounded up to a power of two (read from the fp32 fields, as the
 * reference does), is the scale; a value over the scale, clamped to +-448, becomes the nearest E4M3 code. */
__global__ void quantize_kernel(struct quantized_vectors vectors)
{
    uint32_t block_count = vectors.size / WEIGHTS_PER_SCALE_GROUP;
    uint32_t *vector_words = vectors.words + (size_t)blockIdx.x * (vectors.size / BYTES_PER_WORD);
    for (uint32_t block = threadIdx.x; block < block_count; block += blockDim.x) {
        const float *block_values = vectors.values + (size_t)blockIdx.x * vectors.size + block * WEIGHTS_PER_SCALE_GROUP;
        float largest = 0.0f;
        for (uint32_t offset = 0; offset < WEIGHTS_PER_SCALE_GROUP; offset++) {
            largest = fmaxf(largest, fabsf(block_values[offset]));
        }
        uint32_t ratio_bits = __float_as_uint(fmaxf(largest, AMAX_FLOOR) * E4M3_MAXIMUM_INVERSE);
        uint32_t scale_exponent = (ratio_bits >> FP32_EXPONENT_SHIFT) + ((ratio_bits & FP32_MANTISSA_MASK) != 0u ? 1u : 0u);
        float inverse_scale = __uint_as_float((FP32_EXPONENT_BIAS + E8M0_BIAS - scale_exponent) << FP32_EXPONENT_SHIFT);
        for (uint32_t word_in_block = 0; word_in_block < WORDS_PER_SCALE_GROUP; word_in_block++) {
            uint32_t lane_position = word_in_block / MMA_STEPS_PER_SCALE_GROUP;
            uint32_t step_in_block = word_in_block % MMA_STEPS_PER_SCALE_GROUP;
            uint32_t word = 0;
            for (uint32_t byte_in_word = 0; byte_in_word < BYTES_PER_WORD; byte_in_word++) {
                float value = block_values[activation_offset(step_in_block, lane_position, byte_in_word)];
                word |= quantize_to_e4m3(value, inverse_scale) << (byte_in_word * BITS_PER_BYTE);
            }
            vector_words[activation_word_index(block, lane_position, step_in_block)] = word;
        }
        vectors.exponents[(size_t)blockIdx.x * block_count + block] = (uint8_t)scale_exponent;
    }
}

__device__ __forceinline__ void tensor_core_m16n8k16(float accumulator[ELEMENTS_PER_LANE],
                                                     const uint32_t weights[REGISTERS_PER_OPERAND_A],
                                                     uint32_t input_low, uint32_t input_high)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(accumulator[0]), "=f"(accumulator[1]), "=f"(accumulator[2]), "=f"(accumulator[3])
        : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(input_low), "r"(input_high),
          "f"(accumulator[0]), "f"(accumulator[1]), "f"(accumulator[2]), "f"(accumulator[3]));
}

/* Eight nibbles of one stored word become the four fp16 pairs of the weight operand (register r, half h: the nibble
 * at bit 4r + 16h), each the value times 2^-14: by bit placement, or through the table of the sixteen placed nibbles. */
template <int decode_kind>
__device__ __forceinline__ void decode_nibbles(uint32_t word, const uint32_t *nibble_table, uint32_t weights[REGISTERS_PER_OPERAND_A])
{
    if (decode_kind == DECODE_PLACEMENT) {
        weights[0] = ((word << E2M1_MAGNITUDE_BIT) & E2M1_MAGNITUDE_MASK) | ((word << E2M1_SIGN_BIT) & E2M1_SIGN_MASK);
        weights[1] = ((word << (E2M1_MAGNITUDE_BIT - NIBBLE_BITS)) & E2M1_MAGNITUDE_MASK)
                     | ((word << (E2M1_SIGN_BIT - NIBBLE_BITS)) & E2M1_SIGN_MASK);
        weights[2] = ((word << (E2M1_MAGNITUDE_BIT - 2 * NIBBLE_BITS)) & E2M1_MAGNITUDE_MASK)
                     | ((word << (E2M1_SIGN_BIT - 2 * NIBBLE_BITS)) & E2M1_SIGN_MASK);
        weights[3] = ((word >> (3 * NIBBLE_BITS - E2M1_MAGNITUDE_BIT)) & E2M1_MAGNITUDE_MASK) | (word & E2M1_SIGN_MASK);
    } else {
        for (uint32_t operand_register = 0; operand_register < REGISTERS_PER_OPERAND_A; operand_register++) {
            uint32_t low = (word >> (operand_register * NIBBLE_BITS)) & NIBBLE_MASK;
            uint32_t high = (word >> (operand_register * NIBBLE_BITS + BITS_PER_HALF)) & NIBBLE_MASK;
            weights[operand_register] = nibble_table[low] | (nibble_table[high] << BITS_PER_HALF);
        }
    }
}

/* One warp, one tile of 16 outputs of one job, loads first_load .. first_load + load_count - 1 of the input width.
 * Lane L (g = L / 4, t = L % 4) holds of operand A, the weights, outputs g and g + 8 at inputs 2t, 2t + 1, 2t + 8,
 * 2t + 9 of the instruction's 16; of operand B column g, the vector of one row of the job (a row's quantized input
 * for gate and up, a pair's quantized hidden vector for down), at the same four inputs; of the result, outputs g and
 * g + 8 for columns 2t and 2t + 1. Per load it reads 16 bytes of weights, 16 of its column's vector, the four E8M0
 * codes of its two outputs' scales and the two of each of its two columns'. Per group of 32 inputs each of its four
 * sums is multiplied once by 2^(weight scale + activation scale + 22), built in the exponent field, and added to a
 * running sum (fma: an exact product, one rounding): the sums are stored as [warp][lane][element] for the block. */
template <int decode_kind>
__device__ __forceinline__ void tile_slice_product(const uint4 *tile_packed, const uint32_t *tile_scales,
                                                   struct quantized_vectors vectors, const struct expert_job *job,
                                                   uint32_t first_load, uint32_t load_count, uint32_t *nibble_table,
                                                   float *partial_sums)
{
    float sums[ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t lane = threadIdx.x % LANES_PER_WARP;
    uint32_t lane_group = lane / LANES_PER_LANE_GROUP, lane_position = lane % LANES_PER_LANE_GROUP;
    size_t vector_loads = vectors.size / INPUTS_PER_LOAD;         /* 16-byte loads, and pairs of E8M0 codes, per vector */
    size_t fed_vector = job->pair_index[lane_group] / vectors.pairs_per_vector;
    size_t even_vector = job->pair_index[lane_position * COLUMNS_PER_LANE] / vectors.pairs_per_vector;
    size_t odd_vector = job->pair_index[lane_position * COLUMNS_PER_LANE + 1] / vectors.pairs_per_vector;
    const uint4 *lane_weights = tile_packed + lane;
    const uint32_t *lane_scales = tile_scales + lane_group;
    const uint4 *lane_inputs = (const uint4 *)vectors.words + fed_vector * vector_loads * LANES_PER_LANE_GROUP + lane_position;
    const uint16_t *even_exponents = (const uint16_t *)vectors.exponents + even_vector * vector_loads;
    const uint16_t *odd_exponents = (const uint16_t *)vectors.exponents + odd_vector * vector_loads;
    if (decode_kind == DECODE_TABLE) {                    /* the block's first threads place each nibble once */
        if (threadIdx.x < NIBBLE_CODES) {
            uint32_t placed[REGISTERS_PER_OPERAND_A];
            decode_nibbles<DECODE_PLACEMENT>(threadIdx.x, nibble_table, placed);
            nibble_table[threadIdx.x] = placed[0] & HALF_MASK;
        }
        __syncthreads();
    }
    for (uint32_t load = first_load; load < first_load + load_count; load++) {
        uint4 packed = lane_weights[(size_t)load * LANES_PER_WARP];
        uint4 fed = lane_inputs[(size_t)load * LANES_PER_LANE_GROUP];
        uint32_t weight_words[MMA_STEPS_PER_LOAD] = {packed.x, packed.y, packed.z, packed.w};
        uint32_t input_words[MMA_STEPS_PER_LOAD] = {fed.x, fed.y, fed.z, fed.w};
        uint32_t weight_exponents = lane_scales[(size_t)load * MMA_HALF_ROWS];
        uint32_t activation_exponents[COLUMNS_PER_LANE] = {even_exponents[load], odd_exponents[load]};
        for (uint32_t step = 0; step < MMA_STEPS_PER_LOAD; step += MMA_STEPS_PER_SCALE_GROUP) {
            uint32_t group_in_load = step / MMA_STEPS_PER_SCALE_GROUP;
            float accumulator[ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t group_step = step; group_step < step + MMA_STEPS_PER_SCALE_GROUP; group_step++) {
                uint32_t weights[REGISTERS_PER_OPERAND_A];
                uint32_t word = input_words[group_step], odd_bytes = word >> BITS_PER_BYTE;
                /* four E4M3 bytes become the two fp16 pairs of operand B, each the value times 2^-8 */
                uint32_t input_low = ((word & E4M3_PAIR_SIGN_MASK) << BITS_PER_BYTE)
                                     | ((word & E4M3_PAIR_MAGNITUDE_MASK) << E4M3_MAGNITUDE_SHIFT);
                uint32_t input_high = ((odd_bytes & E4M3_PAIR_SIGN_MASK) << BITS_PER_BYTE)
                                      | ((odd_bytes & E4M3_PAIR_MAGNITUDE_MASK) << E4M3_MAGNITUDE_SHIFT);
                decode_nibbles<decode_kind>(weight_words[group_step], nibble_table, weights);
                tensor_core_m16n8k16(accumulator, weights, input_low, input_high);
            }
            for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
                uint32_t weight_byte = group_in_load * OUTPUTS_PER_LANE + element / COLUMNS_PER_LANE;
                uint32_t column_exponents = activation_exponents[element % COLUMNS_PER_LANE];
                uint32_t exponent_sum = ((weight_exponents >> (weight_byte * BITS_PER_BYTE)) & BYTE_MASK)
                                        + ((column_exponents >> (group_in_load * BITS_PER_BYTE)) & BYTE_MASK);
                /* both scales and the placements' 2^22 as one power of two (the probe's codes keep it normal) */
                float scale = __uint_as_float((exponent_sum - PRODUCT_EXPONENT_BIAS) << FP32_EXPONENT_SHIFT);
                sums[element] = fmaf(accumulator[element], scale, sums[element]);
            }
        }
    }
    for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
        partial_sums[threadIdx.x * ELEMENTS_PER_LANE + element] = sums[element];
    }
}

/* One element of one matrix: the partial sums of its slices (warps first_warp ..), added in slice order. Under
 * --break-exactness the sum of a job of several rows starts at another slice, which depends on their number. */
__device__ __forceinline__ float sum_slices_in_order(const float *partial_sums, uint32_t first_warp, uint32_t lane,
                                                     uint32_t element, uint32_t row_count, uint32_t break_exactness)
{
    uint32_t first_slice = break_exactness != 0u && row_count > 1u ? 1u + (row_count - 2u) % (INPUT_SLICES - 1u) : 0u;
    float sum = 0.0f;
    for (uint32_t position = 0; position < INPUT_SLICES; position++) {
        uint32_t warp = first_warp + (first_slice + position) % INPUT_SLICES;
        sum += partial_sums[(warp * LANES_PER_WARP + lane) * ELEMENTS_PER_LANE + element];
    }
    return sum;
}

/* One product stage of every job: grid (tiles of 16 outputs, jobs); per block INPUT_SLICES warps for each matrix of
 * the stage, one slice of the input width each; the element's own lane of warp 0 adds the slices. Two matrices: gate
 * (warps 0..3) and up (warps 4..7) from the rows' quantized inputs, each rounded to bf16 as the reference's product
 * returns it; h = silu(min(gate, L)) * clamp(up, -L, L) * route weight, cast to bf16 as in Expert.forward, goes to
 * hidden_values, [pair][inner size]. One matrix: down from the pairs' quantized hidden vectors; each row's output
 * of this expert goes as fp32 to pair_output, [pair][hidden size]. */
template <int decode_kind, int matrix_count>
__global__ void expert_kernel(const uint4 *packed, const uint32_t *scales, struct quantized_vectors vectors,
                              const struct expert_job *jobs, float *hidden_values, float *pair_output, uint32_t break_exactness)
{
    __shared__ float partial_sums[matrix_count * INPUT_SLICES * LANES_PER_WARP * ELEMENTS_PER_LANE];
    __shared__ uint32_t nibble_table[NIBBLE_CODES];
    const struct expert_job *job = &jobs[blockIdx.y];
    uint32_t warp = threadIdx.x / LANES_PER_WARP, lane = threadIdx.x % LANES_PER_WARP;
    uint32_t loads = vectors.size / INPUTS_PER_LOAD, loads_per_slice = loads / INPUT_SLICES;
    size_t matrix = (size_t)job->expert_index * MATRIX_KINDS + (matrix_count == 1 ? DOWN_MATRIX : warp / INPUT_SLICES);
    size_t tile_loads = matrix * TILE_LOADS_PER_MATRIX + (size_t)blockIdx.x * loads;
    tile_slice_product<decode_kind>(packed + tile_loads * LANES_PER_WARP, scales + tile_loads * MMA_HALF_ROWS, vectors, job,
                                    (warp % INPUT_SLICES) * loads_per_slice, loads_per_slice, nibble_table, partial_sums);
    __syncthreads();
    for (uint32_t element = 0; element < ELEMENTS_PER_LANE && warp == 0; element++) {
        uint32_t column = (lane % LANES_PER_LANE_GROUP) * COLUMNS_PER_LANE + element % COLUMNS_PER_LANE;
        uint32_t output = blockIdx.x * MMA_ROWS + lane / LANES_PER_LANE_GROUP + (element / COLUMNS_PER_LANE) * MMA_HALF_ROWS;
        float value = sum_slices_in_order(partial_sums, 0, lane, element, job->row_count, break_exactness);
        if (column < job->row_count && matrix_count == 1) {
            pair_output[(size_t)job->pair_index[column] * HIDDEN_SIZE + output] = value;
        } else if (column < job->row_count) {
            float gate_value = fminf(bf16_value(value), SWIGLU_LIMIT);
            float up_value = bf16_value(sum_slices_in_order(partial_sums, UP_MATRIX * INPUT_SLICES, lane, element,
                                                            job->row_count, break_exactness));
            float silu = gate_value / (1.0f + expf(-gate_value));
            up_value = fminf(fmaxf(up_value, -SWIGLU_LIMIT), SWIGLU_LIMIT);
            hidden_values[(size_t)job->pair_index[column] * EXPERT_INNER_SIZE + output] =
                bf16_value(silu * up_value * job->route_weight[column]);
        }
    }
}

/* A row's six experts, each rounded to bf16 as the reference's product returns it, added in slot order, which is
 * ascending expert order, the order of the reference's loop: grid (rows). */
__global__ void row_sum_kernel(const float *pair_output, float *row_output)
{
    for (uint32_t column = threadIdx.x; column < HIDDEN_SIZE; column += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < EXPERTS_PER_ROW; slot++) {
            sum += bf16_value(pair_output[((size_t)blockIdx.x * EXPERTS_PER_ROW + slot) * HIDDEN_SIZE + column]);
        }
        row_output[(size_t)blockIdx.x * HIDDEN_SIZE + column] = sum;
    }
}

/* ---- host side: the formats decoded directly, and the references ---- */

static const double e2m1_values[NIBBLE_CODES] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
static double e4m3_values[E4M3_CODES];

/* E4M3 from its fields: (8 + mantissa) * 2^(exponent - 10), or mantissa * 2^-9 where the exponent is 0. */
static void decode_e4m3_codes(void)
{
    for (uint32_t code = 0; code < E4M3_CODES; code++) {
        uint32_t exponent = (code >> E4M3_EXPONENT_SHIFT) & E4M3_EXPONENT_MASK, mantissa = code & E4M3_MANTISSA_MASK;
        uint32_t scaled = exponent == 0 ? mantissa << 1 : (E4M3_IMPLICIT_ONE + mantissa) << exponent;
        e4m3_values[code] = ((code & E4M3_SIGN_BIT) != 0u ? -1.0 : 1.0) * (double)scaled / E4M3_VALUE_DIVISOR;
    }
}

/* A double rounded to bf16's eight significant bits, to nearest with ties to even (rint in the default mode). */
static double host_round_to_bf16(double value)
{
    int binary_exponent = 0;
    double mantissa = frexp(value, &binary_exponent);     /* value = mantissa * 2^exponent, 0.5 <= |mantissa| < 1 */
    return ldexp(rint(ldexp(mantissa, BF16_SIGNIFICANT_BITS)), binary_exponent - BF16_SIGNIFICANT_BITS);
}

/* The reference's act_quant for one vector, written from its definition and not from the card's handling of bits:
 * per block of 32 the scale is 2^ceil(log2(ratio)), ratio being the largest magnitude (at least 1e-4) times 1 / 448
 * in fp32; each value over the scale, clamped to +-448, becomes the nearest E4M3 value, ties to the even code.
 * quantized gets the codes in the order of the values, then the blocks' E8M0 codes. */
static void host_quantize(const float *values, uint32_t vector_size, uint8_t *quantized)
{
    for (uint32_t block = 0; block < vector_size / WEIGHTS_PER_SCALE_GROUP; block++) {
        const float *block_values = values + block * WEIGHTS_PER_SCALE_GROUP;
        float largest = AMAX_FLOOR;
        for (uint32_t offset = 0; offset < WEIGHTS_PER_SCALE_GROUP; offset++) {
            largest = fmaxf(largest, fabsf(block_values[offset]));
        }
        float ratio = largest * E4M3_MAXIMUM_INVERSE;
        int log2_ceiling = (int)ceil(log2((double)ratio));        /* exact: a float's log2 is not within 1e-8 of a whole */
        quantized[vector_size + block] = (uint8_t)(log2_ceiling + E8M0_BIAS);
        for (uint32_t offset = 0; offset < WEIGHTS_PER_SCALE_GROUP; offset++) {
            double scaled = ldexp((double)block_values[offset], -log2_ceiling);
            double magnitude = fmin(fabs(scaled), (double)E4M3_MAXIMUM);
            uint32_t nearest = 0;
            for (uint32_t code = 1; code < E4M3_NAN_MAGNITUDE; code++) {
                double distance = fabs(e4m3_values[code] - magnitude), nearest_distance = fabs(e4m3_values[nearest] - magnitude);
                if (distance < nearest_distance || (distance == nearest_distance && (code & 1u) == 0u)) {
                    nearest = code;
                }
            }
            quantized[block * WEIGHTS_PER_SCALE_GROUP + offset] = (uint8_t)(nearest | (signbit(scaled) ? E4M3_SIGN_BIT : 0u));
        }
    }
}

/* One vector's quantization read from the card: its codes back in the order of its values, then its E8M0 codes. */
static void fetch_quantized(const struct quantized_vectors *vectors, size_t vector, uint8_t *quantized)
{
    static uint32_t words[HIDDEN_SIZE / BYTES_PER_WORD];
    uint32_t block_count = vectors->size / WEIGHTS_PER_SCALE_GROUP;
    CHECK_CUDA(cudaMemcpy(words, vectors->words + vector * (vectors->size / BYTES_PER_WORD), vectors->size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(quantized + vectors->size, vectors->exponents + vector * block_count, block_count, cudaMemcpyDeviceToHost));
    for (uint32_t word_index = 0; word_index < block_count * WORDS_PER_SCALE_GROUP; word_index++) {
        uint32_t block = word_index / WORDS_PER_SCALE_GROUP;
        uint32_t lane_position = word_index % WORDS_PER_SCALE_GROUP / MMA_STEPS_PER_SCALE_GROUP;
        uint32_t step_in_block = word_index % MMA_STEPS_PER_SCALE_GROUP;
        uint32_t word = words[activation_word_index(block, lane_position, step_in_block)];
        for (uint32_t byte_in_word = 0; byte_in_word < BYTES_PER_WORD; byte_in_word++) {
            quantized[block * WEIGHTS_PER_SCALE_GROUP + activation_offset(step_in_block, lane_position, byte_in_word)] =
                (uint8_t)(word >> (byte_in_word * BITS_PER_BYTE));
        }
    }
}

/* One output of one matrix of one expert in double, from the stored words: per group of 32 inputs the sum of weight
 * times activation, times both scales. It says where the kernels find a weight and a scale: packed words are
 * [expert][matrix][tile of 16 outputs][load of 64 inputs][lane][instruction of the load], eight nibbles each, the
 * nibble of register r and half h of operand A at bit 4r + 16h; scale words are [expert][matrix][tile][load]
 * [lane group g]: byte 2q + r is the E8M0 code of group q of the load for output g + 8r. */
static double host_row_product(uint32_t matrix_kind, uint32_t expert, uint32_t output_row, const uint8_t *quantized)
{
    uint32_t input_size = matrix_kind == DOWN_MATRIX ? EXPERT_INNER_SIZE : HIDDEN_SIZE;
    uint32_t lane_group = output_row % MMA_HALF_ROWS, output_half = output_row % MMA_ROWS / MMA_HALF_ROWS;
    size_t matrix = (size_t)expert * MATRIX_KINDS + matrix_kind;
    size_t first_tile_load = matrix * TILE_LOADS_PER_MATRIX + (size_t)(output_row / MMA_ROWS) * (input_size / INPUTS_PER_LOAD);
    double sum = 0.0;
    for (uint32_t group = 0; group < input_size / WEIGHTS_PER_SCALE_GROUP; group++) {
        size_t tile_load = first_tile_load + group / SCALE_GROUPS_PER_LOAD;
        uint32_t group_in_load = group % SCALE_GROUPS_PER_LOAD;
        uint32_t scale_byte = group_in_load * OUTPUTS_PER_LANE + output_half;
        uint32_t scale_code = (stored_word(tile_load * MMA_HALF_ROWS + lane_group, 1) >> (scale_byte * BITS_PER_BYTE)) & BYTE_MASK;
        double group_sum = 0.0;
        for (uint32_t offset = 0; offset < WEIGHTS_PER_SCALE_GROUP; offset++) {
            uint32_t step = group_in_load * MMA_STEPS_PER_SCALE_GROUP + offset / MMA_INPUTS, input = offset % MMA_INPUTS;
            uint32_t lane = lane_group * LANES_PER_LANE_GROUP + input % MMA_HALF_INPUTS / HALVES_PER_REGISTER;
            uint32_t operand_register = output_half + input / MMA_HALF_INPUTS * OUTPUTS_PER_LANE;
            uint32_t nibble_place = operand_register + input % HALVES_PER_REGISTER * REGISTERS_PER_OPERAND_A;
            uint32_t word = stored_word((tile_load * LANES_PER_WARP + lane) * MMA_STEPS_PER_LOAD + step, 0);
            group_sum += e2m1_values[(word >> (nibble_place * NIBBLE_BITS)) & NIBBLE_MASK]
                         * e4m3_values[quantized[group * WEIGHTS_PER_SCALE_GROUP + offset]];
        }
        sum += ldexp(group_sum, (int)scale_code + (int)quantized[input_size + group] - 2 * E8M0_BIAS);
    }
    return sum;
}

static uint64_t count_differing(const void *first, const void *second, size_t element_count, size_t element_bytes)
{
    uint64_t differing = 0;
    for (size_t element = 0; element < element_count; element++) {
        const char *first_element = (const char *)first + element * element_bytes;
        differing += memcmp(first_element, (const char *)second + element * element_bytes, element_bytes) != 0 ? 1u : 0u;
    }
    return differing;
}

/* ---- host side: options, routes and jobs ---- */

static struct {
    uint32_t expert_count, step_count, break_exactness;
    uint32_t first_overlap, last_overlap;        /* the overlaps and the decodes of this run: all, or the one asked for */
    int first_decode_kind, last_decode_kind, corrupt_weights;
} options = {DEFAULT_EXPERT_COUNT, DEFAULT_STEP_COUNT, 0, 0, PERCENT, DECODE_PLACEMENT, DECODE_TABLE, 0};

/* The synthetic routing. The experts of a step are consecutive: in one half of the experts for even steps, in the
 * other for odd ones, and six further on each time, so that a launch reads nothing a card's cache still holds from
 * the launches before it. Row r takes six of them from the 6r-th on (overlap 0), the 3r-th (50) or the first (100).
 * Slots are in ascending expert order. */
static uint32_t route_expert(uint32_t step, uint32_t overlap_percent, uint32_t row, uint32_t slot)
{
    uint32_t half = options.expert_count / 2;
    uint32_t first = (step % 2) * half + (step / 2) * EXPERTS_PER_ROW % (half - MAX_PAIRS + 1u);
    return first + row * (EXPERTS_PER_ROW * (PERCENT - overlap_percent) / PERCENT) + slot;
}

static float route_weight(uint32_t step, uint32_t row, uint32_t slot)
{
    uint32_t code = mix_bits(~ROUTE_SEED + step * STEP_STRIDE + row * EXPERTS_PER_ROW + slot);
    return (float)(1u + code % ROUTE_WEIGHT_STEPS) / ROUTE_WEIGHT_DIVISOR;
}

/* The jobs of rows first_row .. first_row + row_count - 1, which fill the batch from its slot 0: one job per expert,
 * its columns in the order of the rows. Returns their number. */
static uint32_t build_jobs(uint32_t step, uint32_t overlap_percent, uint32_t first_row, uint32_t row_count, struct expert_job *jobs)
{
    uint32_t job_count = 0;
    for (uint32_t pair = 0; pair < row_count * EXPERTS_PER_ROW; pair++) {
        uint32_t row = first_row + pair / EXPERTS_PER_ROW, slot = pair % EXPERTS_PER_ROW;
        uint32_t expert = route_expert(step, overlap_percent, row, slot), job_index = 0;
        while (job_index < job_count && jobs[job_index].expert_index != expert) {
            job_index++;
        }
        struct expert_job *job = &jobs[job_index];
        if (job_index == job_count) {
            job_count++;
            job->expert_index = expert;
            job->row_count = 0;
            for (uint32_t column = 0; column < MAX_ROWS; column++) {
                job->pair_index[column] = pair;              /* a column not in use reads what column 0 reads */
                job->route_weight[column] = 0.0f;
            }
        }
        job->pair_index[job->row_count] = pair;
        job->route_weight[job->row_count++] = route_weight(step, row, slot);
    }
    return job_count;
}

/* ---- host side: one card ---- */

static struct {
    uint4 *packed;                               /* the experts' weights and scales: see host_row_product */
    uint32_t *scales;
    struct expert_job *jobs;
    struct quantized_vectors input, hidden;      /* of the rows of the batch, of the pairs */
    float *pair_output;                          /* [pair][hidden size]: one expert's output for one row */
    float *row_output;                           /* [row][hidden size] */
    cudaEvent_t started, stopped;
} card;                                          /* the card being probed: one at a time */

static void allocate_vectors(struct quantized_vectors *vectors, size_t vector_count, uint32_t size, uint32_t pairs_per_vector)
{
    CHECK_CUDA(cudaMalloc((void **)&vectors->values, vector_count * size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&vectors->words, vector_count * size));
    CHECK_CUDA(cudaMalloc((void **)&vectors->exponents, vector_count * (size / WEIGHTS_PER_SCALE_GROUP)));
    vectors->size = size;
    vectors->pairs_per_vector = pairs_per_vector;
}

/* What a launch needs that is not timed: the jobs of these rows on the card, and the rows' inputs. */
static uint32_t prepare_layer(uint32_t step, uint32_t overlap_percent, uint32_t first_row, uint32_t row_count)
{
    struct expert_job jobs[MAX_PAIRS];
    uint32_t job_count = build_jobs(step, overlap_percent, first_row, row_count, jobs);
    CHECK_CUDA(cudaMemcpy(card.jobs, jobs, job_count * sizeof(struct expert_job), cudaMemcpyHostToDevice));
    fill_input_kernel<<<row_count, FILL_THREADS_PER_BLOCK>>>(card.input.values, first_row, step);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return job_count;
}

/* The routed experts of one layer: both quantizations, gate with up, down, and the rows' sums. */
template <int decode_kind>
static void launch_layer_as(uint32_t job_count, uint32_t row_count)
{
    dim3 gate_up_grid(EXPERT_INNER_SIZE / MMA_ROWS, job_count), down_grid(HIDDEN_SIZE / MMA_ROWS, job_count);
    unsigned int gate_up_threads = MATRICES_PER_EXPERT_INPUT_SIDE * INPUT_SLICES * LANES_PER_WARP;
    quantize_kernel<<<row_count, HIDDEN_SIZE / WEIGHTS_PER_SCALE_GROUP>>>(card.input);
    expert_kernel<decode_kind, MATRICES_PER_EXPERT_INPUT_SIDE><<<gate_up_grid, gate_up_threads>>>(
        card.packed, card.scales, card.input, card.jobs, card.hidden.values, NULL, options.break_exactness);
    quantize_kernel<<<row_count * EXPERTS_PER_ROW, EXPERT_INNER_SIZE / WEIGHTS_PER_SCALE_GROUP>>>(card.hidden);
    expert_kernel<decode_kind, 1><<<down_grid, INPUT_SLICES * LANES_PER_WARP>>>(
        card.packed, card.scales, card.hidden, card.jobs, NULL, card.pair_output, options.break_exactness);
    row_sum_kernel<<<row_count, FILL_THREADS_PER_BLOCK>>>(card.pair_output, card.row_output);
}

static void (*const launch_layer[DECODE_KINDS])(uint32_t job_count, uint32_t row_count) = {
    launch_layer_as<DECODE_PLACEMENT>, launch_layer_as<DECODE_TABLE>};

/* One launch of the checked step outside the timing, its outputs made stale first: a word it does not write cannot
 * pass for a result. result gets RESULT_FLOATS: the expert outputs as the down stage wrote them, then the sums. */
static void run_checked_layer(int decode_kind, uint32_t overlap_percent, uint32_t first_row, uint32_t row_count, int stale_byte,
                              float *result)
{
    CHECK_CUDA(cudaMemset(card.pair_output, stale_byte, ROW_SUMS_AT * sizeof(float)));
    CHECK_CUDA(cudaMemset(card.row_output, stale_byte, (RESULT_FLOATS - ROW_SUMS_AT) * sizeof(float)));
    uint32_t job_count = prepare_layer(CHECKED_STEP, overlap_percent, first_row, row_count);
    launch_layer[decode_kind](job_count, row_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(result, card.pair_output, ROW_SUMS_AT * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(result + ROW_SUMS_AT, card.row_output, (RESULT_FLOATS - ROW_SUMS_AT) * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

/* The launch that just ran, against the host, for two pairs (the first row at its last expert, the last row at its
 * first: where rows share experts, column 0 and the last column of a job): the quantization of the row's input, the
 * hidden vector from it, the quantization of the card's hidden vector, the output from that. Returns the number of
 * checks that did not come out as required. */
static int check_against_reference(const char *case_name, uint32_t overlap_percent, uint32_t row_count, const float *result)
{
    static uint8_t host_quantized[QUANTIZED_BYTES(HIDDEN_SIZE)], card_quantized[QUANTIZED_BYTES(HIDDEN_SIZE)];
    static float values[HIDDEN_SIZE];
    int failures = 0;
    for (uint32_t checked = 0; checked < CHECKED_PAIRS; checked++) {
        uint32_t pair = checked == 0 ? EXPERTS_PER_ROW - 1u : (row_count - 1u) * EXPERTS_PER_ROW;
        uint32_t row = pair / EXPERTS_PER_ROW, slot = pair % EXPERTS_PER_ROW;
        uint32_t expert = route_expert(CHECKED_STEP, overlap_percent, row, slot);
        double hidden_error = 0.0, hidden_square = 0.0, output_error = 0.0, output_square = 0.0;
        for (uint32_t column = 0; column < HIDDEN_SIZE; column++) {
            values[column] = input_value(row, column, CHECKED_STEP);
        }
        host_quantize(values, HIDDEN_SIZE, host_quantized);
        fetch_quantized(&card.input, row, card_quantized);
        uint64_t input_differences = count_differing(card_quantized, host_quantized, QUANTIZED_BYTES(HIDDEN_SIZE), 1);
        CHECK_CUDA(cudaMemcpy(values, card.hidden.values + (size_t)pair * EXPERT_INNER_SIZE, EXPERT_INNER_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (uint32_t inner = 0; inner < EXPERT_INNER_SIZE; inner++) {
            double gate_value = host_round_to_bf16(host_row_product(GATE_MATRIX, expert, inner, host_quantized));
            double up_value = host_round_to_bf16(host_row_product(UP_MATRIX, expert, inner, host_quantized));
            gate_value = fmin(gate_value, (double)SWIGLU_LIMIT);
            up_value = fmin(fmax(up_value, -(double)SWIGLU_LIMIT), (double)SWIGLU_LIMIT);
            double reference = host_round_to_bf16(gate_value / (1.0 + exp(-gate_value)) * up_value
                                                  * (double)route_weight(CHECKED_STEP, row, slot));
            hidden_error += ((double)values[inner] - reference) * ((double)values[inner] - reference);
            hidden_square += reference * reference;
        }
        host_quantize(values, EXPERT_INNER_SIZE, host_quantized);         /* of the card's own hidden vector */
        fetch_quantized(&card.hidden, pair, card_quantized);
        uint64_t hidden_differences = count_differing(card_quantized, host_quantized, QUANTIZED_BYTES(EXPERT_INNER_SIZE), 1);
        for (uint32_t output = 0; output < HIDDEN_SIZE; output += DOWN_CHECK_STRIDE) {
            double reference = host_row_product(DOWN_MATRIX, expert, output, card_quantized);
            double card_value = (double)result[(size_t)pair * HIDDEN_SIZE + output];
            output_error += (card_value - reference) * (card_value - reference);
            output_square += reference * reference;
        }
        /* written so that a NaN from the card is a difference */
        int hidden_matches = sqrt(hidden_error) <= RELATIVE_TOLERANCE * sqrt(hidden_square);
        int output_matches = sqrt(output_error) <= RELATIVE_TOLERANCE * sqrt(output_square);
        printf("  %s, row %u at expert %u: %llu of %d quantized input bytes and exponents differ; hidden vector off by %.3e "
               "of its rms: %s; %llu of %d quantized hidden bytes and exponents differ; output off by %.3e of its rms: %s\n",
               case_name, row, expert, (unsigned long long)input_differences, QUANTIZED_BYTES(HIDDEN_SIZE),
               sqrt(hidden_error / hidden_square), hidden_matches ? "matches the reference" : "DIFFERS",
               (unsigned long long)hidden_differences, QUANTIZED_BYTES(EXPERT_INNER_SIZE), sqrt(output_error / output_square),
               output_matches ? "matches the reference" : "DIFFERS");
        failures += (hidden_matches == (options.corrupt_weights && checked == 0) ? 1 : 0) + (output_matches ? 0 : 1)
                    + (input_differences != 0 ? 1 : 0) + (hidden_differences != 0 ? 1 : 0);
    }
    return failures;
}

/* The checks of one case, for every decode of the run: the reference (first decode) or the first decode's bits (the
 * other), then each row alone against the same row in the launch. Returns the number that did not come out as required. */
static int check_case(const char *case_name, uint32_t overlap_percent, uint32_t row_count)
{
    static float kept_result[DECODE_KINDS][RESULT_FLOATS], alone_result[RESULT_FLOATS];
    const float *first_result = kept_result[options.first_decode_kind];
    size_t compared_floats = row_count * (ROW_EXPERT_FLOATS + HIDDEN_SIZE);
    int must_differ = options.break_exactness != 0u && overlap_percent > 0u && row_count > 1u;
    int failures = 0;
    if (options.corrupt_weights) {                        /* the start of the gate matrix of the first checked pair */
        uint32_t expert = route_expert(CHECKED_STEP, overlap_percent, 0, EXPERTS_PER_ROW - 1u);
        size_t first_word = expert * (MATRIX_KINDS * PACKED_WORDS_PER_MATRIX);
        CHECK_CUDA(cudaMemset((uint32_t *)card.packed + first_word, CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES));
    }
    for (int decode_kind = options.first_decode_kind; decode_kind <= options.last_decode_kind; decode_kind++) {
        const float *result = kept_result[decode_kind];
        uint64_t differing = 0;
        run_checked_layer(decode_kind, overlap_percent, 0, row_count, STALE_BYTE_SHARED, kept_result[decode_kind]);
        if (decode_kind == options.first_decode_kind) {
            failures += check_against_reference(case_name, overlap_percent, row_count, result);
        } else {                                          /* what the launch did not write is stale in both */
            differing = count_differing(result, first_result, RESULT_FLOATS, sizeof(float));
            printf("  %s, %s: %llu of %zu words differ from the %s decode's: %s\n", case_name, decode_names[decode_kind],
                   (unsigned long long)differing, compared_floats, decode_names[options.first_decode_kind],
                   differing == 0 ? "the same bits" : "DIFFERENT BITS");
            failures += differing != 0 ? 1 : 0;
        }
        differing = 0;
        for (uint32_t row = 0; row < row_count; row++) {
            run_checked_layer(decode_kind, overlap_percent, row, 1, STALE_BYTE_ALONE, alone_result);
            differing += count_differing(result + row * ROW_EXPERT_FLOATS, alone_result, ROW_EXPERT_FLOATS, sizeof(float))
                         + count_differing(result + ROW_SUMS_AT + (size_t)row * HIDDEN_SIZE, alone_result + ROW_SUMS_AT,
                                           HIDDEN_SIZE, sizeof(float));
        }
        printf("  %s, %s: %llu of %zu words differ between a row in this launch and the same row launched alone: %s%s\n",
               case_name, decode_names[decode_kind], (unsigned long long)differing, compared_floats,
               differing == 0 ? "rows are exact" : "ROWS DIFFER",
               must_differ ? " (required: the order of additions was made to depend on the rows of a job)" : "");
        failures += (differing != 0 ? 1 : 0) != must_differ ? 1 : 0;
    }
    return failures;
}

/* Mean microseconds of one step, each started on an idle card. alone_vectors NULL: the routed experts of one layer for
 * the step's routes and inputs; *job_count gets the experts read. Otherwise: the quantization of row_count of them. */
static double time_steps(int decode_kind, uint32_t overlap_percent, uint32_t row_count,
                         const struct quantized_vectors *alone_vectors, uint32_t *job_count)
{
    double total_milliseconds = 0.0;
    for (uint32_t step = 0; step < WARM_UP_STEPS + options.step_count; step++) {
        float milliseconds = 0.0f;
        if (alone_vectors == NULL) {
            *job_count = prepare_layer(step, overlap_percent, 0, row_count);
        }
        CHECK_CUDA(cudaEventRecord(card.started));
        if (alone_vectors == NULL) {
            launch_layer[decode_kind](*job_count, row_count);
        } else {
            quantize_kernel<<<row_count, alone_vectors->size / WEIGHTS_PER_SCALE_GROUP>>>(*alone_vectors);
        }
        CHECK_CUDA(cudaEventRecord(card.stopped));
        CHECK_CUDA(cudaEventSynchronize(card.stopped));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, card.started, card.stopped));
        total_milliseconds += step >= WARM_UP_STEPS ? (double)milliseconds : 0.0;
    }
    return total_milliseconds * MICROSECONDS_PER_MILLISECOND / options.step_count;
}

static int probe_card(int card_index)
{
    cudaDeviceProp properties;
    size_t packed_words = (size_t)options.expert_count * MATRIX_KINDS * PACKED_WORDS_PER_MATRIX;
    size_t scale_words = (size_t)options.expert_count * MATRIX_KINDS * SCALE_WORDS_PER_MATRIX;
    size_t case_count = ((options.last_overlap - options.first_overlap) / OVERLAP_STEP + 1) * ROW_COUNT_CASES;
    double decode_microseconds[DECODE_KINDS] = {0.0, 0.0};
    int failures = 0;
    CHECK_CUDA(cudaSetDevice(card_index));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, card_index));
    int is_sm_80 = properties.major == SM_80_MAJOR && properties.minor == 0;
    int is_sm_120 = properties.major == SM_120_MAJOR && properties.minor == 0;
    unsigned int fill_blocks = (unsigned int)properties.multiProcessorCount * FILL_BLOCKS_PER_SM;
    CHECK_CUDA(cudaMalloc((void **)&card.packed, packed_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void **)&card.scales, scale_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void **)&card.jobs, MAX_PAIRS * sizeof(struct expert_job)));
    CHECK_CUDA(cudaMalloc((void **)&card.pair_output, ROW_SUMS_AT * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&card.row_output, (RESULT_FLOATS - ROW_SUMS_AT) * sizeof(float)));
    allocate_vectors(&card.input, MAX_ROWS, HIDDEN_SIZE, EXPERTS_PER_ROW);
    allocate_vectors(&card.hidden, MAX_PAIRS, EXPERT_INNER_SIZE, 1);
    CHECK_CUDA(cudaEventCreate(&card.started));
    CHECK_CUDA(cudaEventCreate(&card.stopped));
    fill_words_kernel<<<fill_blocks, FILL_THREADS_PER_BLOCK>>>((uint32_t *)card.packed, 0, packed_words);
    fill_words_kernel<<<fill_blocks, FILL_THREADS_PER_BLOCK>>>(card.scales, 1, scale_words);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("card %d %s (sm_%d%d, %d SMs): e2m1 weights, e4m3 activations, %u experts of %.2f MiB in memory\n", card_index,
           properties.name, properties.major, properties.minor, properties.multiProcessorCount, options.expert_count,
           (double)EXPERT_BYTES / BYTES_PER_MIB);
    for (size_t case_index = 0; case_index < case_count; case_index++) {
        uint32_t overlap_percent = options.first_overlap + (uint32_t)(case_index / ROW_COUNT_CASES) * OVERLAP_STEP;
        uint32_t row_count = row_counts[case_index % ROW_COUNT_CASES];
        char case_name[CASE_NAME_BYTES];
        snprintf(case_name, sizeof(case_name), "overlap %3u, %u row(s)", overlap_percent, row_count);
        failures += check_case(case_name, overlap_percent, row_count);
        if (options.corrupt_weights) {
            break;                                       /* the megabyte stays overwritten: one case shows it */
        }
        for (int decode_kind = options.first_decode_kind; decode_kind <= options.last_decode_kind; decode_kind++) {
            uint32_t job_count = 0;
            double microseconds = time_steps(decode_kind, overlap_percent, row_count, NULL, &job_count);
            double bytes_read = (double)job_count * (double)EXPERT_BYTES;
            decode_microseconds[decode_kind] += microseconds;
            printf("  %s, %s: %8.1f us per layer for the routed experts (both quantizations in it), %2u expert(s) read "
                   "once, %6.1f MB read, %6.0f GB/s", case_name, decode_names[decode_kind], microseconds, job_count,
                   bytes_read / BYTES_PER_MB, bytes_read / BYTES_PER_GB / (microseconds / MICROSECONDS_PER_SECOND));
            if (row_count == MAX_ROWS && (is_sm_80 || is_sm_120)) {
                printf(", today's kernels: %d us at eight rows, %d us at one row",
                       is_sm_80 ? TODAY_EIGHT_ROWS_MICROSECONDS_SM_80 : TODAY_EIGHT_ROWS_MICROSECONDS_SM_120,
                       is_sm_80 ? TODAY_ONE_ROW_MICROSECONDS_SM_80 : TODAY_ONE_ROW_MICROSECONDS_SM_120);
            }
            printf("\n");
        }
    }
    for (uint32_t row_count = 1; row_count <= MAX_ROWS && !options.corrupt_weights; row_count *= MAX_ROWS) {
        uint32_t pair_count = row_count * EXPERTS_PER_ROW;
        double input_microseconds = time_steps(0, 0, row_count, &card.input, NULL);
        double hidden_microseconds = time_steps(0, 0, pair_count, &card.hidden, NULL);
        printf("  quantization alone, %u row(s): %5.1f us for the launch of the %u input(s) of %d values (%.2f us each), "
               "%5.1f us for that of the %u hidden vectors of %d values (%.2f us each)\n", row_count, input_microseconds,
               row_count, HIDDEN_SIZE, input_microseconds / row_count, hidden_microseconds, pair_count, EXPERT_INNER_SIZE,
               hidden_microseconds / pair_count);
    }
    if (options.first_decode_kind != options.last_decode_kind && !options.corrupt_weights) {
        int faster = decode_microseconds[DECODE_TABLE] < decode_microseconds[DECODE_PLACEMENT] ? DECODE_TABLE : DECODE_PLACEMENT;
        printf("  faster nibble decode on this card: %s (%.1f us against %.1f us, summed over the cases above)\n",
               decode_names[faster], decode_microseconds[faster], decode_microseconds[DECODE_KINDS - 1 - faster]);
    }
    CHECK_CUDA(cudaEventDestroy(card.started));
    CHECK_CUDA(cudaEventDestroy(card.stopped));
    void *card_memory[] = {card.packed, card.scales, card.jobs, card.pair_output, card.row_output, card.input.values,
                           card.input.words, card.input.exponents, card.hidden.values, card.hidden.words, card.hidden.exponents};
    for (size_t index = 0; index < sizeof(card_memory) / sizeof(card_memory[0]); index++) {
        CHECK_CUDA(cudaFree(card_memory[index]));
    }
    return failures;
}

static int parse_number(const char *text, long minimum, long maximum, long *number)
{
    char *number_end = NULL;
    long parsed = strtol(text, &number_end, DECIMAL_BASE);
    if (number_end == text || *number_end != '\0' || parsed < minimum || parsed > maximum) {
        return 0;
    }
    *number = parsed;
    return 1;
}

int main(int argument_count, char **arguments)
{
    long expert_count = DEFAULT_EXPERT_COUNT, step_count = DEFAULT_STEP_COUNT, overlap = -1;
    int usage_is_wrong = 0, card_count = 0, failures = 0;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        const char *argument = arguments[argument_index];
        const char *value = argument_index + 1 < argument_count ? arguments[argument_index + 1] : NULL;
        if (strcmp(argument, "--corrupt") == 0) {
            options.corrupt_weights = 1;
        } else if (strcmp(argument, "--break-exactness") == 0) {
            options.break_exactness = 1;
        } else if (value == NULL || ++argument_index >= argument_count) {
            usage_is_wrong = 1;                                    /* every other option has a value */
        } else if (strcmp(argument, "--overlap") == 0) {
            usage_is_wrong |= !parse_number(value, 0, PERCENT, &overlap) || overlap % OVERLAP_STEP != 0;
        } else if (strcmp(argument, "--decode") == 0) {
            options.first_decode_kind = strcmp(value, decode_names[DECODE_TABLE]) == 0 ? DECODE_TABLE : DECODE_PLACEMENT;
            options.last_decode_kind = options.first_decode_kind;
            usage_is_wrong |= strcmp(value, decode_names[options.first_decode_kind]) != 0;
        } else if (strcmp(argument, "--experts") == 0) {
            usage_is_wrong |= !parse_number(value, MIN_EXPERT_COUNT, MAX_EXPERT_COUNT, &expert_count);
        } else if (strcmp(argument, "--steps") == 0) {
            usage_is_wrong |= !parse_number(value, 1, MAX_STEP_COUNT, &step_count);
        } else {
            usage_is_wrong = 1;
        }
    }
    if (usage_is_wrong) {
        fprintf(stderr, "usage: %s [--overlap 0|50|100] [--decode placement|table] [--experts N] [--steps N] "
                        "[--break-exactness] [--corrupt]\n", arguments[0]);
        return EXIT_FAILURE;
    }
    if (overlap >= 0) {
        options.first_overlap = options.last_overlap = (uint32_t)overlap;
    }
    options.expert_count = (uint32_t)expert_count;
    options.step_count = (uint32_t)step_count;
    decode_e4m3_codes();
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (card_count < 1) {
        fprintf(stderr, "no card to measure\n");
        return EXIT_FAILURE;
    }
    for (int card_index = 0; card_index < card_count && card_index < MAX_CARDS; card_index++) {
        failures += probe_card(card_index);
    }
    if (failures != 0) {
        fprintf(stderr, "%d check(s) did not come out as required\n", failures);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
