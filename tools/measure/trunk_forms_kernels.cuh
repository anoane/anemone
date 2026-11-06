/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trunk_forms_kernels.cuh: y = W x for the trunk's 8-bit matrices and 1 to 8 rows, as ONE kernel body with
 * compile-time choices, and the quantization that feeds it: package 4 of round 1 of the speed campaign
 * (research/speed/ROUND_1.md, "trunk_forms"). The part down to "the probe's own" is written to move into
 * engine/source as it stands once a choice has won; the rest is the list of kernels the round stamps and the
 * synthetic rings. trunk_forms_probe.cu holds every check. Measurement tool: not part of the engine.
 *
 * Contract (the engine's, engine/include/anemone/linear_records.h, where nothing else is said):
 *   - A block is one tile of 16 outputs, a warp one slice of the input width, a load 64 inputs. Slice i of S holds
 *     loads floor(i L / S) to floor((i + 1) L / S) - 1 and is one float32 chain from +0; the S chains are added
 *     from +0 in ascending order and the sum is rounded once to bfloat16. S is blockDim.x / 32, at most L.
 *   - The rows are the columns of operand B: lane 4 g + t feeds column g and holds the results of outputs g and
 *     g + 8 for columns 2 t and 2 t + 1. A column from row_count on carries row 0 and is stored nowhere.
 *   - Stored E4M3 weights: bytes [tile][block of 32 inputs][lane][16], byte b of a lane the weight of output
 *     16 tile + g + 8 (b % 2) at input 32 block + 8 (b / 4) + 2 t + (b / 2) % 2; behind them one E8M0 code per
 *     32 outputs by 32 inputs, [output / 32][block]. A quantized vector: codes [row][width] in lane order (byte
 *     ((4 l + t) 4 + s) 4 + j is value 64 l + 16 s + 2 t + j / 2 + 8 (j % 2)), exponents [row][width / 32].
 *   - A kernel checks nothing and uses no atomic operation: the host has held every range and count a launch
 *     reads, writes or loops by to its array before the launch (check 10 of research/speed/CONVENTIONS.md).
 *
 * The forms (enum trunk_form), per block of 32 inputs:
 *   CODES          two m16n8k16 f16 instructions, the second on the sum of the first; weights and codes placed
 *                  as 16-bit floats in the loop; chain = fma(sum, 2^(both scale codes), chain). The engine's form.
 *   FP16           the same with rows that are fp16 values in value order, not quantized (the earlier probe's).
 *   PLACED16       the same with operand B read as the quantizer placed it: operands [row][width] of 16-bit halves,
 *                  half ((4 l + t) 4 + s) 4 + 2 u + h is value 64 l + 16 s + 8 u + 2 t + h.
 *   CARRIED_BF16   m16n8k16 bf16: the weights placed as bfloat16 (the value times 2^-120), operand B the
 *                  bfloat16 of the dequantized value times 2^100, so the multiply-add takes the weights' code only.
 *                  The vector's exponent codes must be at most TRUNK_CARRIED_EXPONENT_CODE_MOST: the host holds it.
 *   FP8            sm_120a: one m16n8k32 e4m3 instruction; the weights' bytes after one byte permute a register,
 *                  the codes as stored; the multiply-add outside as in CODES.
 *   FP8_SCALED     sm_120a: the same instruction with both scales inside and a zero accumulator, then one float32
 *                  addition to the chain: the rounding points of CODES.
 *   WHOLE_S8       m16n8k32 s8: the stored bytes read as signed whole numbers, the codes whole numbers (the value
 *                  divided by its scale and by 4, to nearest); the sum is exact; fma((float)sum, 2^(codes), chain).
 *   BF16_WEIGHTS   bfloat16 weights, words [tile][step of 16 inputs][lane][4] (half h of word r the weight of
 *                  output 16 tile + g + 8 (r % 2) at input 16 step + 2 t + 8 (r / 2) + h), and bfloat16 rows in
 *                  value order: m16n8k16 bf16 adding into the chain, no scale, no decode.
 * Loads asked ahead: ahead_weights loads of weights (1: the engine's) and ahead_inputs loads of everything else
 * (0: read where used, the engine's) are held in registers, asked in the order inputs_first says, and never beyond
 * the slice's last load. merged: the grid is the tiles of two matrices that share their input. row_dependent: where
 * a launch has more than one row the sum is rounded to bfloat16 after every slice (to show the row check failing).
 */
#ifndef TRUNK_FORMS_KERNELS_CUH
#define TRUNK_FORMS_KERNELS_CUH
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

enum trunk_form {
    TRUNK_CODES = 0, TRUNK_FP16 = 1, TRUNK_PLACED16 = 2, TRUNK_CARRIED_BF16 = 3, TRUNK_FP8 = 4, TRUNK_FP8_SCALED = 5,
    TRUNK_WHOLE_S8 = 6, TRUNK_BF16_WEIGHTS = 7
};
enum {
    TRUNK_THREADS_PER_WARP = 32, TRUNK_LANES_PER_LANE_GROUP = 4, TRUNK_OUTPUTS_PER_TILE = 16,
    TRUNK_OUTPUTS_PER_HALF_TILE = 8, TRUNK_OUTPUTS_PER_SCALE = 32, TRUNK_TILES_PER_SCALE = 2,
    TRUNK_COLUMNS_PER_LANE = 2, TRUNK_OUTPUT_HALVES = 2, TRUNK_ELEMENTS_PER_LANE = 4, TRUNK_MOST_ROWS = 8,
    TRUNK_THIRD_OF_FOUR = 2, TRUNK_FOURTH_OF_FOUR = 3,       /* of a lane's results, of the registers of operand A */
    TRUNK_MOST_SLICES = 32, TRUNK_VALUES_PER_LOAD = 64, TRUNK_VALUES_PER_BLOCK = 32, TRUNK_VALUES_PER_STEP = 16,
    TRUNK_VALUES_PER_HALF_STEP = 8, TRUNK_VALUES_PER_LANE_PLACE = 2, TRUNK_BLOCKS_PER_LOAD = 2,
    TRUNK_STEPS_PER_BLOCK = 2, TRUNK_STEPS_PER_LOAD = 4,
    TRUNK_WORDS_PER_PART = 4,                    /* a part is 16 bytes of a lane: one load instruction */
    TRUNK_WORDS_PER_STEP = 2, TRUNK_REGISTERS_PER_WORD = 2, TRUNK_HALVES_PER_REGISTER = 2,
    TRUNK_E4M3_PARTS_PER_LOAD = 2, TRUNK_BF16_PARTS_PER_LOAD = 4, TRUNK_WEIGHT_WORDS = 16, TRUNK_FED_PARTS = 2,
    TRUNK_FED_WORDS = 8, TRUNK_REGISTERS_PER_OPERAND_A = 4, TRUNK_REGISTERS_PER_OPERAND_B = 2,
    TRUNK_PART_BYTES = 16, TRUNK_HALF_BYTES = 2, TRUNK_PAIRS_PER_LOAD = 32, TRUNK_PAIRS_PER_REGISTER = 4,
    TRUNK_MERGED_MATRICES = 2, TRUNK_REQUEST_KINDS = 2, TRUNK_SHARED_BYTES_PER_THREAD = 16,
    TRUNK_BYTE_BITS = 8, TRUNK_HALF_BITS = 16, TRUNK_BYTE_MASK = 0xFF,
    /* an E4M3 byte's seven magnitude bits at bits 7 .. 13 of a 16-bit float are its value times 2^-8; at bits
     * 4 .. 10 of a bfloat16 its value times 2^-120; the sign goes to bit 15 of either */
    TRUNK_FP16_MAGNITUDE_BIT = 7, TRUNK_BF16_MAGNITUDE_BIT = 4, TRUNK_E4M3_FP16_EXPONENT = 8,
    TRUNK_E4M3_BF16_EXPONENT = 120, TRUNK_CARRIED_EXPONENT = 100, TRUNK_CARRIED_EXPONENT_CODE_MOST = 145,
    TRUNK_E4M3_SIGN_BIT = 0x80, TRUNK_E4M3_MAGNITUDE_MASK = 0x7F, TRUNK_E4M3_MANTISSA_BITS = 3,
    TRUNK_E4M3_MANTISSA_MASK = 7, TRUNK_E4M3_EXPONENT_MASK = 15, TRUNK_E4M3_IMPLICIT_ONE = 8,
    TRUNK_E4M3_UNITS_EXPONENT = 130,             /* at a float32 exponent field f E4M3 numbers are 2^(f - 130) apart */
    TRUNK_E4M3_CODE_BASE = 968,                  /* ((127 - 7) << 3) + 8: code = 8 f + units - this */
    TRUNK_E8M0_BIAS = 127, TRUNK_WHOLE_CODE_EXPONENT = 2,     /* a whole code is the E4M3-scaled value over 2^2 */
    TRUNK_FLOAT32_EXPONENT_SHIFT = 23, TRUNK_FLOAT32_EXPONENT_MOST = 254, TRUNK_FLOAT32_SIGN_TO_E4M3 = 24,
    /* selectors of prmt: result byte n is byte (selector >> 4 n) & 7 of the eight bytes of two words. Of the two
     * stored words of 16 inputs: bytes 0, 4, 2, 6 are one output at inputs 2 t, 2 t + 8, 2 t + 1, 2 t + 9, the order
     * of a stored code word; bytes 1, 5, 3, 7 the other output's */
    TRUNK_LOW_OUTPUT_SELECTOR = 0x6240, TRUNK_HIGH_OUTPUT_SELECTOR = 0x7351
};
#define TRUNK_E4M3_EVEN_SIGN_MASK 0x00800080u    /* bytes 0 and 2 of a word: the halves of one register */
#define TRUNK_E4M3_EVEN_MAGNITUDE_MASK 0x007F007Fu
#define TRUNK_E4M3_ODD_MAGNITUDE_MASK 0x7F007F00u /* bytes 1 and 3: the halves of the other, their signs in place */
#define TRUNK_HALF_SIGN_MASK 0x80008000u
#define TRUNK_CODE_IN_BOTH_HALVES 0x0101u        /* one scale code for both outputs of a lane */
#define TRUNK_CODE_IN_EVERY_BYTE 0x01010101u     /* a scale operand: whichever lane and byte the instruction takes */
#define TRUNK_FLOAT32_MAGNITUDE_MASK 0x7FFFFFFFu
#define TRUNK_FLOAT32_MANTISSA_MASK 0x007FFFFFu
#define TRUNK_FLOAT32_INFINITY_BITS 0x7F800000u
#define TRUNK_BFLOAT16_ROUNDING_BIAS 0x7FFFu     /* plus the lowest kept bit: to nearest, a tie to even */
#define TRUNK_BFLOAT16_NOT_A_NUMBER 0x7FC0u
#define TRUNK_EVERY_LANE 0xFFFFFFFFu
#define TRUNK_MAGNITUDE_FLOOR 1.0e-4f            /* the reference's lower bound of a block's largest magnitude */
#define TRUNK_E4M3_LARGEST 448.0f
#define TRUNK_E4M3_LARGEST_INVERSE (1.0f / 448.0f)
#define TRUNK_E4M3_SMALLEST_NORMAL 0.015625f     /* 2^-6 */
#define TRUNK_WHOLE_CODE_FACTOR 0.25f            /* 2^-TRUNK_WHOLE_CODE_EXPONENT: 448 becomes 112 */

struct trunk_matrix {                            /* one stored matrix of a launch, and where its results go */
    const unsigned char *stored;                 /* E4M3: its bytes, its scale codes behind; or its bfloat16 words */
    uint16_t *output;                            /* bfloat16 [row][output_size] */
    uint32_t output_size, reserved_zero;
};
struct trunk_product_arguments {                 /* grid: the tiles of its matrices; block 32 S; 16 shared bytes each */
    struct trunk_matrix matrices[TRUNK_MERGED_MATRICES];    /* a merged kernel reads both, any other the first */
    const unsigned char *fed;                    /* [row][input_size]: codes (a byte each), operands or values (two) */
    const unsigned char *exponents;              /* [row][input_size / 32]; not read where the rows carry no code */
    uint32_t input_size, row_count;
};
struct trunk_quantize_arguments {                /* grid (ceil(width / blockDim.x), vectors); block: a multiple of 32 */
    const uint16_t *values;                      /* bfloat16 [vector][width] */
    unsigned char *codes, *exponents;            /* [vector][width] in lane order, [vector][width / 32] */
    uint16_t *operands;                          /* [vector][width]: written by PLACED16 and CARRIED_BF16 only */
    uint32_t width, reserved_zero;
};
struct trunk_lane {                              /* what one lane of a product reads */
    const uint4 *weights;                        /* its 16 bytes of the first part of its tile */
    const unsigned char *weight_scales;          /* the code of the first block of its tile's 32 outputs */
    const unsigned char *fed;                    /* the vector its column feeds, from its lane place on */
    const unsigned char *exponents[TRUNK_COLUMNS_PER_LANE];  /* of the vectors of columns 2 t and 2 t + 1 */
};
struct trunk_held_weights { uint32_t words[TRUNK_WEIGHT_WORDS]; };          /* one load of a lane, in registers */
struct trunk_held_fed {                          /* everything else a lane reads for one load */
    uint32_t words[TRUNK_FED_WORDS];             /* codes: a word an instruction; else a word a register of operand B */
    uint32_t column_codes[TRUNK_COLUMNS_PER_LANE];           /* byte q: the exponent of block q of the load */
    uint32_t weight_codes;                       /* byte 2 q + r: block q of the load, output half r */
};

__device__ __forceinline__ float trunk_bfloat16_value(uint32_t stored_bits)
{
    return __uint_as_float(stored_bits << TRUNK_HALF_BITS);
}

/* A float32 rounded to bfloat16: to nearest, a tie to even; a not-a-number becomes the quiet one. */
__device__ __forceinline__ uint32_t trunk_bfloat16_bits(float value)
{
    const uint32_t bits = __float_as_uint(value);
    const uint32_t rounded =
        (bits + TRUNK_BFLOAT16_ROUNDING_BIAS + ((bits >> TRUNK_HALF_BITS) & 1u)) >> TRUNK_HALF_BITS;
    return (bits & TRUNK_FLOAT32_MAGNITUDE_MASK) > TRUNK_FLOAT32_INFINITY_BITS ? TRUNK_BFLOAT16_NOT_A_NUMBER : rounded;
}

/* 2^(exponent_field - 127) for a field of 1 to 254; +0 below, 2^127 above: no code makes an infinity of a scale. */
__device__ __forceinline__ float trunk_power_of_two(int32_t exponent_field)
{
    const int32_t above_zero = exponent_field < 0 ? 0 : exponent_field;
    const int32_t bounded = above_zero > TRUNK_FLOAT32_EXPONENT_MOST ? TRUNK_FLOAT32_EXPONENT_MOST : above_zero;
    return __uint_as_float((uint32_t)bounded << TRUNK_FLOAT32_EXPONENT_SHIFT);
}

/* sums = A B + sums for the lane's part of each. m16n8k16: 16 inputs, weights[r] outputs g (r even) and g + 8 at
 * inputs 2 t, 2 t + 1 (+ 8 for r of 2 and 3), fed_first and fed_second column g at those inputs. m16n8k32: 32
 * inputs, weights[r] the same outputs at inputs 4 t .. 4 t + 3 (+ 16), a byte each, and so the two fed registers. */
#define TRUNK_OPERANDS " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}"
#define TRUNK_FOUR(constraint, four)             /* four registers as operands of one constraint */                  \
    constraint(four[0]), constraint(four[1]), constraint(four[TRUNK_THIRD_OF_FOUR]),                                 \
        constraint(four[TRUNK_FOURTH_OF_FOUR])
#define TRUNK_TENSOR_CORE(function_name, sum_type, sum_written, sum_read, instruction)                               \
    __device__ __forceinline__ void function_name(sum_type sums[TRUNK_ELEMENTS_PER_LANE],                            \
                                                  const uint32_t weights[TRUNK_REGISTERS_PER_OPERAND_A],             \
                                                  uint32_t fed_first, uint32_t fed_second)                           \
    {                                                                                                                \
        asm volatile(instruction TRUNK_OPERANDS ";\n" : TRUNK_FOUR(sum_written, sums)                                \
                     : TRUNK_FOUR("r", weights), "r"(fed_first), "r"(fed_second), TRUNK_FOUR(sum_read, sums));       \
    }
TRUNK_TENSOR_CORE(trunk_core_f16, float, "=f", "f", "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
TRUNK_TENSOR_CORE(trunk_core_bf16, float, "=f", "f", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
TRUNK_TENSOR_CORE(trunk_core_whole, int32_t, "=r", "r", "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
TRUNK_TENSOR_CORE(trunk_core_e4m3, float, "=f", "f", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
/* The same with a scale code per output and per column inside: every lane hands over its tile's and its column's
 * code in every byte, so whichever lane and byte the instruction takes ({0, 0}: the first) holds the same code. */
__device__ __forceinline__ void trunk_core_e4m3_scaled(float sums[TRUNK_ELEMENTS_PER_LANE],
                                                       const uint32_t weights[TRUNK_REGISTERS_PER_OPERAND_A],
                                                       uint32_t fed_first, uint32_t fed_second, uint32_t weight_scales,
                                                       uint32_t fed_scales)
{
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"
                 TRUNK_OPERANDS ", %14, {0, 0}, %15, {0, 0};\n"
                 : TRUNK_FOUR("=f", sums)
                 : TRUNK_FOUR("r", weights), "r"(fed_first), "r"(fed_second), TRUNK_FOUR("f", sums),
                   "r"(weight_scales), "r"(fed_scales));
}
#else       /* a card without the two forms: their kernels are stamped empty of products and never launched */
__device__ __forceinline__ void trunk_core_e4m3(float *, const uint32_t *, uint32_t, uint32_t) {}
__device__ __forceinline__ void trunk_core_e4m3_scaled(float *, const uint32_t *, uint32_t, uint32_t, uint32_t,
                                                       uint32_t) {}
#endif

/* Four E4M3 bytes of one word as two registers of 16-bit halves: bytes 0 and 2 the low and the high half of the
 * first, bytes 1 and 3 of the second. magnitude_bit says which float the halves are. Placement computes nothing. */
__device__ __forceinline__ void trunk_place_e4m3(uint32_t word, uint32_t magnitude_bit, uint32_t *first_register,
                                                 uint32_t *second_register)
{
    *first_register = ((word & TRUNK_E4M3_EVEN_SIGN_MASK) << TRUNK_BYTE_BITS) |
                      ((word & TRUNK_E4M3_EVEN_MAGNITUDE_MASK) << magnitude_bit);
    *second_register = (word & TRUNK_HALF_SIGN_MASK) |
                       ((word & TRUNK_E4M3_ODD_MAGNITUDE_MASK) >> (TRUNK_BYTE_BITS - magnitude_bit));
}

/* What a form adds to the weights' scale code (and the column's, where its rows carry one) for the float32
 * exponent field of the power of two that undoes its placements and applies the scales. */
__device__ __forceinline__ int32_t trunk_scale_field_bias(int form)
{
    return form == TRUNK_FP16 ? TRUNK_E4M3_FP16_EXPONENT
           : form == TRUNK_CARRIED_BF16 ? TRUNK_E4M3_BF16_EXPONENT - TRUNK_CARRIED_EXPONENT
           : form == TRUNK_FP8 ? -TRUNK_E8M0_BIAS
           : form == TRUNK_WHOLE_S8 ? TRUNK_WHOLE_CODE_EXPONENT - TRUNK_E8M0_BIAS
           : TRUNK_E4M3_FP16_EXPONENT + TRUNK_E4M3_FP16_EXPONENT - TRUNK_E8M0_BIAS;
}

/* The four words of the 16 bytes a lane loads at once. */
__device__ __forceinline__ void trunk_unpack(uint4 part, uint32_t words[TRUNK_WORDS_PER_PART])
{
    words[0] = part.x;
    words[1] = part.y;
    words[TRUNK_THIRD_OF_FOUR] = part.z;
    words[TRUNK_FOURTH_OF_FOUR] = part.w;
}

/* The lane's weights of one load of 64 inputs: two parts of E4M3 bytes, or four of bfloat16 words. */
template <int form>
__device__ __forceinline__ struct trunk_held_weights trunk_fetch_weights(const struct trunk_lane *lane, uint32_t load)
{
    const uint32_t part_count = form == TRUNK_BF16_WEIGHTS ? TRUNK_BF16_PARTS_PER_LOAD : TRUNK_E4M3_PARTS_PER_LOAD;
    struct trunk_held_weights held = {};
    for (uint32_t part = 0; part < part_count; part++) {
        trunk_unpack(lane->weights[((size_t)load * part_count + part) * TRUNK_THREADS_PER_WARP],
                     &held.words[part * TRUNK_WORDS_PER_PART]);
    }
    return held;
}

/* Everything else the lane reads for one load: the two scale codes of its weights, what its column feeds, and the
 * exponents of the two columns it holds results of. */
template <int form>
__device__ __forceinline__ struct trunk_held_fed trunk_fetch_fed(const struct trunk_lane *lane, uint32_t load)
{
    const uint32_t part_count = form == TRUNK_PLACED16 || form == TRUNK_CARRIED_BF16 ? TRUNK_FED_PARTS : 1u;
    struct trunk_held_fed held = {};
    if (form != TRUNK_BF16_WEIGHTS) {
        const unsigned char *block_codes = lane->weight_scales + (size_t)load * TRUNK_BLOCKS_PER_LOAD;
        held.weight_codes = ((uint32_t)block_codes[0] | ((uint32_t)block_codes[1] << TRUNK_HALF_BITS)) *
                            TRUNK_CODE_IN_BOTH_HALVES;
    }
    if (form == TRUNK_FP16 || form == TRUNK_BF16_WEIGHTS) {      /* values in value order: register k is pair 4 k + t */
        const uint32_t *pairs = (const uint32_t *)lane->fed + (size_t)load * TRUNK_PAIRS_PER_LOAD;
        for (uint32_t word = 0; word < TRUNK_FED_WORDS; word++) {
            held.words[word] = pairs[word * TRUNK_PAIRS_PER_REGISTER];
        }
    } else {
        const uint4 *parts = (const uint4 *)lane->fed + (size_t)load * TRUNK_LANES_PER_LANE_GROUP * part_count;
        for (uint32_t part = 0; part < part_count; part++) {
            trunk_unpack(parts[part], &held.words[part * TRUNK_WORDS_PER_PART]);
        }
    }
    if (form == TRUNK_CODES || form == TRUNK_PLACED16 || form == TRUNK_FP8 || form == TRUNK_FP8_SCALED ||
        form == TRUNK_WHOLE_S8) {
        for (uint32_t column = 0; column < TRUNK_COLUMNS_PER_LANE; column++) {
            held.column_codes[column] = ((const uint16_t *)lane->exponents[column])[load];
        }
    }
    return held;
}

/* One slice of one tile for one lane: the chains of its four results over loads first_load .. end_load - 1. The
 * slice has at least one load (S is at most L). chains_out is the lane's place in the block's shared memory. */
template <int form, int ahead_weights, int ahead_inputs, int inputs_first>
__device__ __forceinline__ void trunk_slice_chains(const struct trunk_lane *lane, uint32_t first_load,
                                                   uint32_t end_load, float chains_out[TRUNK_ELEMENTS_PER_LANE])
{
    const uint32_t weights_depth = (uint32_t)ahead_weights;
    const uint32_t fed_depth = ahead_inputs > 0 ? (uint32_t)ahead_inputs : 1u;
    const int inputs_turn = inputs_first != 0 ? 0 : 1;          /* of the two requests of a load: the inputs' */
    const uint32_t weight_magnitude_bit =
        form == TRUNK_CARRIED_BF16 ? TRUNK_BF16_MAGNITUDE_BIT : TRUNK_FP16_MAGNITUDE_BIT;
    struct trunk_held_weights held_weights[ahead_weights];
    struct trunk_held_fed held_fed[ahead_inputs > 0 ? ahead_inputs : 1] = {};
    float chains[TRUNK_ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    /* the slice's first loads, in the order of the form; a place past the slice's end asks for the first again */
    for (int turn = 0; turn < TRUNK_REQUEST_KINDS; turn++) {
        for (int ahead = 0; ahead < (turn == inputs_turn ? ahead_inputs : ahead_weights); ahead++) {
            const uint32_t place = first_load + (uint32_t)ahead;
            if (turn == inputs_turn) {
                held_fed[ahead] = trunk_fetch_fed<form>(lane, place < end_load ? place : first_load);
            } else {
                held_weights[ahead] = trunk_fetch_weights<form>(lane, place < end_load ? place : first_load);
            }
        }
    }
    for (uint32_t load = first_load; load < end_load; load++) {
        const struct trunk_held_weights weights = held_weights[0];
        struct trunk_held_fed fed = held_fed[0];
        for (uint32_t ahead = 1; ahead < weights_depth; ahead++) {
            held_weights[ahead - 1u] = held_weights[ahead];
        }
        for (uint32_t ahead = 1; ahead < fed_depth; ahead++) {
            held_fed[ahead - 1u] = held_fed[ahead];
        }
        for (int turn = 0; turn < TRUNK_REQUEST_KINDS; turn++) {
            if (turn != inputs_turn) {
                if (load + weights_depth < end_load) {
                    held_weights[weights_depth - 1u] = trunk_fetch_weights<form>(lane, load + weights_depth);
                }
            } else if (ahead_inputs == 0) {
                fed = trunk_fetch_fed<form>(lane, load);
            } else if (load + fed_depth < end_load) {
                held_fed[fed_depth - 1u] = trunk_fetch_fed<form>(lane, load + fed_depth);
            }
        }
        if (form == TRUNK_BF16_WEIGHTS) {        /* no scale: the instruction adds 16 products into the chains */
            for (uint32_t step = 0; step < TRUNK_STEPS_PER_LOAD; step++) {
                trunk_core_bf16(chains, &weights.words[step * TRUNK_REGISTERS_PER_OPERAND_A],
                                fed.words[step * TRUNK_REGISTERS_PER_OPERAND_B],
                                fed.words[step * TRUNK_REGISTERS_PER_OPERAND_B + 1u]);
            }
        }
        for (uint32_t block = 0; form != TRUNK_BF16_WEIGHTS && block < TRUNK_BLOCKS_PER_LOAD; block++) {
            float sums[TRUNK_ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (form == TRUNK_FP8 || form == TRUNK_FP8_SCALED || form == TRUNK_WHOLE_S8) {
                uint32_t bytes[TRUNK_REGISTERS_PER_OPERAND_A];        /* a byte permute a register */
                for (uint32_t half = 0; half < TRUNK_STEPS_PER_BLOCK; half++) {
                    const uint32_t *words =
                        &weights.words[(block * TRUNK_STEPS_PER_BLOCK + half) * TRUNK_WORDS_PER_STEP];
                    bytes[half * TRUNK_REGISTERS_PER_WORD] =
                        __byte_perm(words[0], words[1], TRUNK_LOW_OUTPUT_SELECTOR);
                    bytes[half * TRUNK_REGISTERS_PER_WORD + 1u] =
                        __byte_perm(words[0], words[1], TRUNK_HIGH_OUTPUT_SELECTOR);
                }
                const uint32_t fed_first = fed.words[block * TRUNK_STEPS_PER_BLOCK];
                const uint32_t fed_second = fed.words[block * TRUNK_STEPS_PER_BLOCK + 1u];
                if (form == TRUNK_WHOLE_S8) {
                    int32_t whole[TRUNK_ELEMENTS_PER_LANE] = {0, 0, 0, 0};
                    trunk_core_whole(whole, bytes, fed_first, fed_second);
                    for (uint32_t element = 0; element < TRUNK_ELEMENTS_PER_LANE; element++) {
                        sums[element] = (float)whole[element];       /* under 2^19 in magnitude: exact */
                    }
                } else if (form == TRUNK_FP8) {
                    trunk_core_e4m3(sums, bytes, fed_first, fed_second);
                } else {                         /* column_codes[0] is of the vector the lane feeds: trunk_lane_set */
                    const uint32_t weight_code = (fed.weight_codes >> (block * TRUNK_HALF_BITS)) & TRUNK_BYTE_MASK;
                    const uint32_t fed_code = (fed.column_codes[0] >> (block * TRUNK_BYTE_BITS)) & TRUNK_BYTE_MASK;
                    trunk_core_e4m3_scaled(sums, bytes, fed_first, fed_second, weight_code * TRUNK_CODE_IN_EVERY_BYTE,
                                           fed_code * TRUNK_CODE_IN_EVERY_BYTE);
                }
            } else {
                for (uint32_t step = block * TRUNK_STEPS_PER_BLOCK; step < (block + 1u) * TRUNK_STEPS_PER_BLOCK;
                     step++) {
                    uint32_t operand[TRUNK_REGISTERS_PER_OPERAND_A];
                    uint32_t fed_low = fed.words[step * TRUNK_REGISTERS_PER_OPERAND_B];
                    uint32_t fed_high = fed.words[step * TRUNK_REGISTERS_PER_OPERAND_B + 1u];
                    trunk_place_e4m3(weights.words[step * TRUNK_WORDS_PER_STEP], weight_magnitude_bit, &operand[0],
                                     &operand[1]);
                    trunk_place_e4m3(weights.words[step * TRUNK_WORDS_PER_STEP + 1u], weight_magnitude_bit,
                                     &operand[TRUNK_REGISTERS_PER_WORD], &operand[TRUNK_REGISTERS_PER_WORD + 1u]);
                    if (form == TRUNK_CODES) {   /* a code word is one instruction's: the low inputs, the high ones */
                        trunk_place_e4m3(fed.words[step], TRUNK_FP16_MAGNITUDE_BIT, &fed_low, &fed_high);
                    }
                    if (form == TRUNK_CARRIED_BF16) {
                        trunk_core_bf16(sums, operand, fed_low, fed_high);
                    } else {
                        trunk_core_f16(sums, operand, fed_low, fed_high);
                    }
                }
            }
            for (uint32_t element = 0; element < TRUNK_ELEMENTS_PER_LANE; element++) {
                const uint32_t output_half = element / TRUNK_COLUMNS_PER_LANE;
                const uint32_t column_code =
                    fed.column_codes[element % TRUNK_COLUMNS_PER_LANE] >> (block * TRUNK_BYTE_BITS);
                const uint32_t weight_code =
                    fed.weight_codes >> ((block * TRUNK_OUTPUT_HALVES + output_half) * TRUNK_BYTE_BITS);
                int32_t exponent_field = (int32_t)(weight_code & TRUNK_BYTE_MASK) + trunk_scale_field_bias(form);
                if (form != TRUNK_FP16 && form != TRUNK_CARRIED_BF16) {
                    exponent_field += (int32_t)(column_code & TRUNK_BYTE_MASK);
                }
                chains[element] = form == TRUNK_FP8_SCALED
                                      ? chains[element] + sums[element]
                                      : fmaf(sums[element], trunk_power_of_two(exponent_field), chains[element]);
            }
        }
    }
    for (uint32_t element = 0; element < TRUNK_ELEMENTS_PER_LANE; element++) {
        chains_out[element] = chains[element];
    }
}

/* Where the lane of a thread reads: its part of a tile of one stored matrix, and of the launch's vectors. */
template <int form>
__device__ __forceinline__ void trunk_lane_set(struct trunk_lane *lane, const unsigned char *stored,
                                               uint32_t output_size, const struct trunk_product_arguments *arguments,
                                               uint32_t tile, uint32_t lane_index)
{
    const uint32_t lane_group = lane_index / TRUNK_LANES_PER_LANE_GROUP;
    const size_t lane_place = lane_index % TRUNK_LANES_PER_LANE_GROUP;
    const size_t even_column = lane_place * TRUNK_COLUMNS_PER_LANE;
    const size_t width = arguments->input_size, block_count = width / TRUNK_VALUES_PER_BLOCK;
    const size_t fed_row = lane_group < arguments->row_count ? lane_group : 0u;
    const size_t even_row = even_column < arguments->row_count ? even_column : 0u;
    const size_t odd_row = even_column + 1u < arguments->row_count ? even_column + 1u : 0u;
    const size_t parts_per_tile =
        form == TRUNK_BF16_WEIGHTS ? width / TRUNK_VALUES_PER_STEP : block_count;      /* of one lane */
    lane->weights = (const uint4 *)stored + (size_t)tile * parts_per_tile * TRUNK_THREADS_PER_WARP + lane_index;
    lane->weight_scales = stored + (size_t)output_size * width + (size_t)(tile / TRUNK_TILES_PER_SCALE) * block_count;
    if (form == TRUNK_FP16 || form == TRUNK_BF16_WEIGHTS) {
        lane->fed = arguments->fed + (fed_row * width + lane_place * TRUNK_COLUMNS_PER_LANE) * TRUNK_HALF_BYTES;
    } else if (form == TRUNK_PLACED16 || form == TRUNK_CARRIED_BF16) {
        lane->fed = arguments->fed + fed_row * width * TRUNK_HALF_BYTES +
                    lane_place * TRUNK_FED_PARTS * TRUNK_PART_BYTES;
    } else {
        lane->fed = arguments->fed + fed_row * width + lane_place * TRUNK_PART_BYTES;
    }
    lane->exponents[0] = arguments->exponents + (form == TRUNK_FP8_SCALED ? fed_row : even_row) * block_count;
    lane->exponents[1] = arguments->exponents + odd_row * block_count;
}

/* One thread of a product: its slice, the barrier, then warp w finishes the results w, w + S, .. of every lane. */
template <int form, int ahead_weights, int ahead_inputs, int inputs_first, int merged, int row_dependent>
__device__ __forceinline__ void trunk_product_block(struct trunk_product_arguments arguments, float *slice_chains)
{
    const uint32_t warp = threadIdx.x / TRUNK_THREADS_PER_WARP, lane_index = threadIdx.x % TRUNK_THREADS_PER_WARP;
    const uint32_t slice_count = blockDim.x / TRUNK_THREADS_PER_WARP;
    const uint32_t load_count = arguments.input_size / TRUNK_VALUES_PER_LOAD;
    const uint32_t first_tiles = arguments.matrices[0].output_size / TRUNK_OUTPUTS_PER_TILE;
    const bool second = merged != 0 && blockIdx.x >= first_tiles;
    const unsigned char *stored = second ? arguments.matrices[1].stored : arguments.matrices[0].stored;
    uint16_t *output = second ? arguments.matrices[1].output : arguments.matrices[0].output;
    const uint32_t output_size = second ? arguments.matrices[1].output_size : arguments.matrices[0].output_size;
    const uint32_t tile = second ? blockIdx.x - first_tiles : blockIdx.x;
    struct trunk_lane lane;
    trunk_lane_set<form>(&lane, stored, output_size, &arguments, tile, lane_index);
    trunk_slice_chains<form, ahead_weights, ahead_inputs, inputs_first>(
        &lane, warp * load_count / slice_count, (warp + 1u) * load_count / slice_count,
        slice_chains + (size_t)threadIdx.x * TRUNK_ELEMENTS_PER_LANE);
    __syncthreads();
    for (uint32_t element = warp; element < TRUNK_ELEMENTS_PER_LANE; element += slice_count) {
        const uint32_t row = lane_index % TRUNK_LANES_PER_LANE_GROUP * TRUNK_COLUMNS_PER_LANE +
                             element % TRUNK_COLUMNS_PER_LANE;
        const uint32_t output_index = tile * TRUNK_OUTPUTS_PER_TILE + lane_index / TRUNK_LANES_PER_LANE_GROUP +
                                      element / TRUNK_COLUMNS_PER_LANE * TRUNK_OUTPUTS_PER_HALF_TILE;
        if (row < arguments.row_count) {
            float sum = 0.0f;
            for (uint32_t slice = 0; slice < slice_count; slice++) {
                sum += slice_chains[((size_t)slice * TRUNK_THREADS_PER_WARP + lane_index) * TRUNK_ELEMENTS_PER_LANE +
                                    element];
                if (row_dependent != 0 && arguments.row_count > 1u) {
                    sum = trunk_bfloat16_value(trunk_bfloat16_bits(sum));
                }
            }
            output[(size_t)row * output_size + output_index] = (uint16_t)trunk_bfloat16_bits(sum);
        }
    }
}

/* The reference's act_quant with scales rounded up to a power of two, as the engine's kernel has it: a warp takes
 * one block of 32 values of one vector, a lane one value. m = max(largest magnitude, 1e-4) / 448 in float32; the
 * code c has 2^(c - 127) the least power of two not under m; a value over that scale, limited to 448, goes to the
 * nearest E4M3 number, a tie to the even code, the sign kept. WHOLE_S8 stores that quotient over 4, to nearest, as
 * a signed byte. PLACED16 adds the code placed as a 16-bit float, CARRIED_BF16 the bfloat16 of the code's value
 * times 2^(c - 127 + 100), each where operand B wants it. Every lane stores to places of its own. */
template <int form>
__device__ __forceinline__ void trunk_quantize_block(struct trunk_quantize_arguments arguments)
{
    const uint32_t lane_index = threadIdx.x % TRUNK_THREADS_PER_WARP;
    const uint32_t block_index =
        blockIdx.x * (blockDim.x / TRUNK_THREADS_PER_WARP) + threadIdx.x / TRUNK_THREADS_PER_WARP;
    const uint32_t block_count = arguments.width / TRUNK_VALUES_PER_BLOCK;
    const size_t vector = blockIdx.y;
    if (block_index >= block_count) {
        return;
    }
    const float value = trunk_bfloat16_value(
        arguments.values[vector * arguments.width + (size_t)block_index * TRUNK_VALUES_PER_BLOCK + lane_index]);
    float largest = fabsf(value);
    for (uint32_t distance = TRUNK_THREADS_PER_WARP >> 1u; distance != 0; distance >>= 1u) {
        largest = fmaxf(largest, __shfl_xor_sync(TRUNK_EVERY_LANE, largest, (int)distance));
    }
    const uint32_t ratio_bits = __float_as_uint(fmaxf(largest, TRUNK_MAGNITUDE_FLOOR) * TRUNK_E4M3_LARGEST_INVERSE);
    const uint32_t exponent_code = ((ratio_bits >> TRUNK_FLOAT32_EXPONENT_SHIFT) & TRUNK_BYTE_MASK) +
                                   ((ratio_bits & TRUNK_FLOAT32_MANTISSA_MASK) != 0 ? 1u : 0u);
    const float scaled = value * __uint_as_float((TRUNK_E8M0_BIAS + TRUNK_E8M0_BIAS - exponent_code)
                                                 << TRUNK_FLOAT32_EXPONENT_SHIFT);
    /* E4M3 numbers are 2^(f - 130) apart at a float32 exponent field f, and 2^-9 apart under 2^-6: the magnitude
     * in those units, rounded to nearest with a tie to even, is added to the code of the stretch's first number */
    const float magnitude = fminf(fabsf(scaled), TRUNK_E4M3_LARGEST);
    const uint32_t exponent_field =
        __float_as_uint(fmaxf(magnitude, TRUNK_E4M3_SMALLEST_NORMAL)) >> TRUNK_FLOAT32_EXPONENT_SHIFT;
    const uint32_t units = (uint32_t)__float2int_rn(
        magnitude * __uint_as_float((TRUNK_E4M3_UNITS_EXPONENT + TRUNK_E8M0_BIAS - exponent_field)
                                    << TRUNK_FLOAT32_EXPONENT_SHIFT));
    const uint32_t sign = (__float_as_uint(scaled) >> TRUNK_FLOAT32_SIGN_TO_E4M3) & TRUNK_E4M3_SIGN_BIT;
    const uint32_t code = ((exponent_field << TRUNK_E4M3_MANTISSA_BITS) + units - TRUNK_E4M3_CODE_BASE) | sign;
    /* value lane_index of the block is input 2 t + half + 8 upper of instruction `step` of its load */
    const uint32_t step =
        block_index % TRUNK_BLOCKS_PER_LOAD * TRUNK_STEPS_PER_BLOCK + lane_index / TRUNK_VALUES_PER_STEP;
    const uint32_t value_in_step = lane_index % TRUNK_VALUES_PER_STEP;
    const uint32_t lane_place = value_in_step % TRUNK_VALUES_PER_HALF_STEP / TRUNK_VALUES_PER_LANE_PLACE;
    const uint32_t half = value_in_step % TRUNK_VALUES_PER_LANE_PLACE;
    const uint32_t upper = value_in_step / TRUNK_VALUES_PER_HALF_STEP;
    const size_t place = vector * arguments.width +
                         (((size_t)(block_index / TRUNK_BLOCKS_PER_LOAD) * TRUNK_LANES_PER_LANE_GROUP + lane_place) *
                              TRUNK_STEPS_PER_LOAD + step) * TRUNK_WORDS_PER_PART;
    arguments.codes[place + half * TRUNK_REGISTERS_PER_OPERAND_B + upper] = (unsigned char)(
        form == TRUNK_WHOLE_S8 ? (uint32_t)__float2int_rn(scaled * TRUNK_WHOLE_CODE_FACTOR) : code);
    if (form == TRUNK_PLACED16) {
        arguments.operands[place + upper * TRUNK_HALVES_PER_REGISTER + half] = (uint16_t)(
            (sign << TRUNK_BYTE_BITS) | ((code & TRUNK_E4M3_MAGNITUDE_MASK) << TRUNK_FP16_MAGNITUDE_BIT));
    } else if (form == TRUNK_CARRIED_BF16) {     /* whole units times a power of two: exact, eight bits at most */
        const float carried = (float)units * __uint_as_float(
            (exponent_field + exponent_code - (TRUNK_E4M3_UNITS_EXPONENT - TRUNK_CARRIED_EXPONENT))
            << TRUNK_FLOAT32_EXPONENT_SHIFT);
        arguments.operands[place + upper * TRUNK_HALVES_PER_REGISTER + half] = (uint16_t)(
            (__float_as_uint(carried) >> TRUNK_HALF_BITS) | (sign << TRUNK_BYTE_BITS));
    }
    if (lane_index == 0) {
        arguments.exponents[vector * block_count + block_index] = (unsigned char)exponent_code;
    }
}

/* One kernel per choice, the body the same function; no launch bound, as no kernel of the engine states one. */
#define TRUNK_PRODUCT_KERNEL(kernel_name, form, ahead_weights, ahead_inputs, inputs_first, merged, row_dependent) \
    __global__ void kernel_name(struct trunk_product_arguments arguments)                                         \
    {                                                                                                             \
        extern __shared__ float slice_chains[];                                                                   \
        trunk_product_block<form, ahead_weights, ahead_inputs, inputs_first, merged, row_dependent>(arguments,    \
                                                                                                    slice_chains);\
    }
#define TRUNK_QUANTIZE_KERNEL(kernel_name, form)                                                                  \
    __global__ void kernel_name(struct trunk_quantize_arguments arguments) { trunk_quantize_block<form>(arguments); }

/* ---- the probe's own from here on: the kernels of the round, the rings, the touch ---- */

/* id, form, loads of weights ahead, loads of everything else ahead, 1: the inputs asked first, 1: two matrices a
 * launch, 1: rounding by the row count. A kernel is trunk_product_<id>; the first is the engine's kernel. */
#define TRUNK_PRODUCT_KERNELS(STAMP)                         \
    STAMP(codes, TRUNK_CODES, 1, 0, 0, 0, 0)                 \
    STAMP(fp16, TRUNK_FP16, 1, 0, 0, 0, 0)                   \
    STAMP(placed16, TRUNK_PLACED16, 1, 0, 0, 0, 0)           \
    STAMP(carried_bf16, TRUNK_CARRIED_BF16, 1, 0, 0, 0, 0)   \
    STAMP(fp8, TRUNK_FP8, 1, 0, 0, 0, 0)                     \
    STAMP(fp8_scaled, TRUNK_FP8_SCALED, 1, 0, 0, 0, 0)       \
    STAMP(whole_s8, TRUNK_WHOLE_S8, 1, 0, 0, 0, 0)           \
    STAMP(bf16_weights, TRUNK_BF16_WEIGHTS, 1, 0, 0, 0, 0)   \
    STAMP(weights_2, TRUNK_CODES, 2, 0, 0, 0, 0)             \
    STAMP(weights_4, TRUNK_CODES, 4, 0, 0, 0, 0)             \
    STAMP(inputs_1, TRUNK_CODES, 1, 1, 0, 0, 0)              \
    STAMP(inputs_2, TRUNK_CODES, 1, 2, 0, 0, 0)              \
    STAMP(inputs_first, TRUNK_CODES, 1, 0, 1, 0, 0)          \
    STAMP(weights_2_inputs_1, TRUNK_CODES, 2, 1, 0, 0, 0)    \
    STAMP(weights_2_first, TRUNK_CODES, 2, 0, 1, 0, 0)       \
    STAMP(inputs_1_first, TRUNK_CODES, 1, 1, 1, 0, 0)        \
    STAMP(all_ahead, TRUNK_CODES, 2, 1, 1, 0, 0)             \
    STAMP(merged, TRUNK_CODES, 1, 0, 0, 1, 0)                \
    STAMP(row_dependent, TRUNK_CODES, 1, 0, 0, 0, 1)
#define TRUNK_STAMP_KERNEL(id, form, ahead_weights, ahead_inputs, inputs_first, merged, row_dependent) \
    TRUNK_PRODUCT_KERNEL(trunk_product_##id, form, ahead_weights, ahead_inputs, inputs_first, merged, row_dependent)
TRUNK_PRODUCT_KERNELS(TRUNK_STAMP_KERNEL)
TRUNK_QUANTIZE_KERNEL(trunk_quantize_codes, TRUNK_CODES)
TRUNK_QUANTIZE_KERNEL(trunk_quantize_placed16, TRUNK_PLACED16)
TRUNK_QUANTIZE_KERNEL(trunk_quantize_carried_bf16, TRUNK_CARRIED_BF16)
TRUNK_QUANTIZE_KERNEL(trunk_quantize_whole_s8, TRUNK_WHOLE_S8)

enum {
    TRUNK_FILL_E4M3 = 0, TRUNK_FILL_BF16 = 1,    /* what a ring holds */
    TRUNK_MIX_SHIFT_OUTER = 16, TRUNK_MIX_SHIFT_INNER = 15, TRUNK_WORD_BYTES = 4,
    TRUNK_TOUCH_STRIDE_WORDS = 8                 /* one word of every 32 bytes */
};
#define TRUNK_MIX_MULTIPLIER_FIRST 0x7feb352du
#define TRUNK_MIX_MULTIPLIER_SECOND 0x846ca68bu
#define TRUNK_MAGNITUDE_IN_EVERY_BYTE 0x7F7F7F7Fu
#define TRUNK_SCALE_WORD_BASE 0x72727272u        /* four E8M0 codes of 114 .. 117: scales of 2^-13 .. 2^-10 */
#define TRUNK_SCALE_WORD_SPREAD 0x03030303u
#define TRUNK_UPPER_HALF_MASK 0xFFFF0000u
#define TRUNK_BF16_WEIGHT_CENTRE 128.0f          /* a bfloat16 weight is (a stored byte - 128) 2^-19 */
#define TRUNK_BF16_WEIGHT_UNIT 0x1p-19f

__host__ __device__ inline uint32_t trunk_mix_bits(uint32_t value)
{
    value ^= value >> TRUNK_MIX_SHIFT_OUTER;
    value *= TRUNK_MIX_MULTIPLIER_FIRST;
    value ^= value >> TRUNK_MIX_SHIFT_INNER;
    value *= TRUNK_MIX_MULTIPLIER_SECOND;
    value ^= value >> TRUNK_MIX_SHIFT_OUTER;
    return value;
}

/* Word `word` of a stored matrix; every copy of a ring is alike. Four weight bytes, each any E4M3 code but the two
 * that are no number (those lose their lowest bit); from word scale_word on, four E8M0 scale codes. A card fills
 * a ring by this alone; the host finds a weight's byte by the format's layout: neither knows the other's order. */
__host__ __device__ inline uint32_t trunk_stored_word(uint32_t seed, uint64_t word, uint64_t scale_word)
{
    const uint32_t bits = trunk_mix_bits(seed + (uint32_t)word);
    const uint32_t no_number = (((bits & TRUNK_MAGNITUDE_IN_EVERY_BYTE) + TRUNK_CODE_IN_EVERY_BYTE) >>
                                (TRUNK_BYTE_BITS - 1)) & TRUNK_CODE_IN_EVERY_BYTE;
    return word >= scale_word ? TRUNK_SCALE_WORD_BASE + (bits & TRUNK_SCALE_WORD_SPREAD) : bits & ~no_number;
}

/* The weight a bfloat16 matrix holds for a stored byte: a number a bfloat16 has exactly. */
__host__ __device__ inline float trunk_bfloat16_weight(uint32_t byte)
{
    return ((float)(byte & TRUNK_BYTE_MASK) - TRUNK_BF16_WEIGHT_CENTRE) * TRUNK_BF16_WEIGHT_UNIT;
}

struct trunk_fill_arguments {
    uint32_t *words;                             /* word_count of them: copies of words_per_copy each */
    uint64_t word_count, words_per_copy, scale_word;
    uint32_t kind, seed;
};

/* A ring: any grid, any block. BF16: a word is the bfloat16 weights of bytes 0 and 1 of the stored word. */
__global__ void trunk_fill_kernel(struct trunk_fill_arguments arguments)
{
    const uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < arguments.word_count;
         index += thread_total) {
        const uint32_t stored =
            trunk_stored_word(arguments.seed, index % arguments.words_per_copy, arguments.scale_word);
        const uint32_t low_half = __float_as_uint(trunk_bfloat16_weight(stored)) >> TRUNK_HALF_BITS;
        const uint32_t high_half =
            __float_as_uint(trunk_bfloat16_weight(stored >> TRUNK_BYTE_BITS)) & TRUNK_UPPER_HALF_MASK;
        arguments.words[index] = arguments.kind == TRUNK_FILL_BF16 ? (low_half | high_half) : stored;
    }
}

/* Ledger S13: reads a stored matrix into the card's L2 ahead of its product, one word of every 32 bytes; a thread
 * stores the fold of what it read to a word of its own, so that nothing of the reading can be left out. */
__global__ void trunk_touch_kernel(const uint32_t *words, uint64_t word_count, uint32_t *folds)
{
    const uint64_t thread = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x * TRUNK_TOUCH_STRIDE_WORDS;
    uint32_t fold = 0;
    for (uint64_t index = thread * TRUNK_TOUCH_STRIDE_WORDS; index < word_count; index += stride) {
        fold ^= words[index];
    }
    folds[thread] = fold;
}

#endif /* TRUNK_FORMS_KERNELS_CUH */
