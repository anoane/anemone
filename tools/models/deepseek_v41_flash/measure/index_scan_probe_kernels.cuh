/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_probe_kernels.cuh: what package 11a's probe needs beside index_scan_kernels.cuh and the engine never
 * will: the synthetic keys, queries and weights; the rings of keys of each state kind, filled on the card; the
 * queries placed for each form as their producer would; the two bounds a scan is held against (read_only: the
 * baseline's grid and loads, folded; tensor_only: its grid and instructions in its order on operands made in
 * registers); the stamps of every kernel the probe launches; the host's vocabulary of a step (families, forms, work,
 * candidate lists, samples); the instances and the acceptance figures. index_scan_probe.cu checks all of it against
 * index_scan_reference.h.
 *
 * The card's two roundings, written from the reference (both warp-wide, a lane a value of a block of 32):
 *   TRAINED (kernel.py:155-172 with E8M0 scales, called by model.py:546, 552): m = max(the block's largest magnitude,
 *     6 x 2^-126); the scale code c is the float32 exponent field of m x fp32(1/6), one more unless that product is a
 *     power of two; the value times 2^(127 - c), clamped to 6, to the nearest E2M1 value, a tie to the even code;
 *     stored as a signed byte of twice that value, the scale code beside.
 *   FRESH (kernel.py:74-90 with power-of-two scales, as the reference's act_quant): m = max(largest, 1e-4), c from m x
 *     fp32(1/448) the same way; the value times 2^(127 - c), limited to 448, to the nearest E4M3 code, a tie to even.
 * Measurement tool: not part of the engine.
 */
#ifndef INDEX_SCAN_PROBE_KERNELS_CUH
#define INDEX_SCAN_PROBE_KERNELS_CUH
#include <math.h>
#include "index_scan_kernels.cuh"

enum index_state { INDEX_STATE_FP16 = 0, INDEX_STATE_TRAINED, INDEX_STATE_FRESH, INDEX_STATES };
enum index_shape {                               /* the scan kernels of a form, in their stamping order */
    INDEX_SHAPE_GROUP = 0, INDEX_SHAPE_SINGLE, INDEX_SHAPE_ROW_MAJOR, INDEX_SHAPE_LISTED, INDEX_SHAPE_HEAD_ORDER,
    INDEX_SHAPES
};
enum index_bound { INDEX_BOUND_READ = 0, INDEX_BOUND_CHAINED, INDEX_BOUND_INTERLEAVED };
enum { INDEX_BOUND_SIZES = 2, INDEX_BOUND_SHAPES = 6 };     /* a group's kernel and a row alone's, of each bound */
enum { INDEX_THREADS = 128, INDEX_WIDE_THREADS = 512,       /* a block: 4 warps, warps:16's 16 */
       INDEX_KERNEL_SLOTS = INDEX_FORMS * (INDEX_SHAPES + INDEX_BOUND_SHAPES) };   /* index_scans, then index_bounds */
enum {
    /* the kinds of a block of 32 values (CONVENTIONS 5.7): steps of 1/64 to 2; zeros of both signs between them; a
     * block under the fresh floor of 1e-4; E2M1 ties beside 6 x 2^-2; magnitudes over 2^17; E4M3 ties beside 448 x
     * 2^-8; a block of zeros of both signs alone, the only one at the trained floor (scale code 1) */
    INDEX_VALUE_PLAIN = 0, INDEX_VALUE_ZEROS, INDEX_VALUE_UNDER_FLOOR, INDEX_VALUE_E2M1_TIES, INDEX_VALUE_SPREAD,
    INDEX_VALUE_E4M3_TIES, INDEX_VALUE_ZERO_BLOCK, INDEX_VALUE_KINDS, INDEX_VALUE_STEPS = 256, INDEX_VALUE_CENTRE = 128,
    INDEX_VALUE_STEP_EXPONENT = -6, INDEX_UNDER_FLOOR_EXPONENT = -15, INDEX_E2M1_TIE_EXPONENT = -4,
    INDEX_SPREAD_PLACES = 18, INDEX_TIE_UNITS = 17, INDEX_TIE_EXPONENT = 19, INDEX_TIE_EXPONENTS = 14,
    INDEX_EIGHTHS_MASK = 7, INDEX_EIGHTHS = 8, INDEX_SIGN_BIT_OF_HASH = 16, INDEX_E2M1_CODES = 8,
    INDEX_WEIGHT_EXPONENT = -12, INDEX_KEY_SEED = 0x11a, INDEX_QUERY_SEED = 0x21a, INDEX_WEIGHT_SEED = 0x31a,
    INDEX_MIX_SHIFT_OUTER = 16, INDEX_MIX_SHIFT_INNER = 15, INDEX_NIBBLE_BITS = 4, INDEX_NIBBLE_MASK = 15,
    INDEX_E4M3_MANTISSA_BITS = 3, INDEX_E4M3_SIGN_BIT = 0x80, INDEX_E4M3_UNITS_EXPONENT = 130,
    INDEX_E4M3_CODE_BASE = 968,                  /* ((127 - 7) << 3) + 8: code = 8 f + units - this, f a field */
    INDEX_FLOAT32_SIGN_TO_BYTE = 24, INDEX_TWICE_E8M0_BIAS = 254,
    INDEX_HALF_WORDS_PER_KEY = 64, INDEX_HALF_WORDS_PER_RUN = 16, INDEX_HALF_WORDS_PER_ROW = 2048,
    INDEX_HALF_WORDS_PER_TILE_STEP = 128,        /* of a placed FP16 query row: a step of a head tile, 32 lanes of 4 */
    INDEX_WORDS_PER_LOAD = 4, INDEX_DIMS_PER_HALF_STEP = 8, INDEX_PAIR_IN_REGISTER = 2, INDEX_REGISTERS_PER_STEP = 2,
    INDEX_E2M1_QUADRUPLE = 4,                    /* E2M1 midpoints are whole numbers at four times a value */
    INDEX_FILL_THREADS = 256,                    /* of the fills and placements: a thread a word, or a warp a block */
    /* tensor_only: each of a lane's four sums gains 384 over its two head tiles a row and odd tile (B sums to 128
     * over a key's 128 dims, A is 1 in head tile 0 and 2 in head tile 1); B's patterns; a warp's 32 lanes */
    INDEX_BOUND_SUM_GAIN = 384, INDEX_SUMS_PER_LANE = 4, INDEX_BOUND_PATTERNS = 4, INDEX_BOUND_ZERO_SHIFT = 31,
    INDEX_BOUND_SHARED_BYTES = INDEX_SELECT_WARPS * INDEX_WORD_BYTES,      /* a fold a warp, of up to 32 warps */
    INDEX_BOUND_WARP_SUM = INDEX_LANES * INDEX_SUMS_PER_LANE * INDEX_BOUND_SUM_GAIN
};
#define INDEX_FLOPS_PER_KEY 8192.0               /* 32 heads of 128 dims, a multiply and an add each */
#define INDEX_MIX_FIRST 0x7feb352du
#define INDEX_MIX_SECOND 0x846ca68bu
#define INDEX_E2M1_DOUBLED 0xC8643210u           /* nibble c: twice the E2M1 value of code c (0 .. 6 by halves) */
#define INDEX_E2M1_FLOOR 0x1.8p-124f             /* 6 x 2^-126 */
#define INDEX_E2M1_LARGEST 6.0f
#define INDEX_E2M1_LARGEST_INVERSE (1.0f / 6.0f)
#define INDEX_E4M3_FLOOR 1.0e-4f
#define INDEX_E4M3_LARGEST 448.0f
#define INDEX_E4M3_LARGEST_INVERSE (1.0f / 448.0f)
#define INDEX_E4M3_SMALLEST_NORMAL 0x1p-6f
#define INDEX_E2M1_TIES_LARGEST 1.5f             /* 6 x 2^-2 */
#define INDEX_E4M3_TIES_LARGEST 1.75f            /* 448 x 2^-8 */
#define INDEX_FLOAT32_MANTISSA_MASK 0x007FFFFFu
#define INDEX_FP16_ONES 0x3C003C00u              /* tensor_only's operands, two or four values a word: ones, twos, */
#define INDEX_FP16_TWOS 0x40004000u              /* and ones with the first value 2 and the second 0 (skewed) */
#define INDEX_FP16_SKEWED 0x00004000u
#define INDEX_INT8_ONES 0x01010101u
#define INDEX_INT8_TWOS 0x02020202u
#define INDEX_INT8_SKEWED 0x01010002u
#define INDEX_E4M3_ONES 0x38383838u
#define INDEX_E4M3_TWOS 0x40404040u
#define INDEX_E4M3_SKEWED 0x38380040u
#define INDEX_E2M1_ONES 0x08080808u              /* codes 2 (1.0) and 4 (2.0) at bits 2 .. 5 of a byte */
#define INDEX_E2M1_TWOS 0x10101010u
#define INDEX_E2M1_SKEWED 0x08080010u
#define INDEX_SCALE_ONES 0x7F7F7F7Fu             /* 2^0 in every byte */
#define INDEX_HALF_MASK 0xFFFFu

__host__ __device__ inline uint32_t index_mix(uint32_t value)
{
    value ^= value >> INDEX_MIX_SHIFT_OUTER;
    value *= INDEX_MIX_FIRST;
    value ^= value >> INDEX_MIX_SHIFT_INNER;
    value *= INDEX_MIX_SECOND;
    return value ^ (value >> INDEX_MIX_SHIFT_OUTER);
}
__host__ __device__ inline uint32_t index_e2m1_doubled(uint32_t code)   /* code 0 .. 7: 0, 1, 2, 3, 4, 6, 8, 12 */
{
    return (INDEX_E2M1_DOUBLED >> (code * INDEX_NIBBLE_BITS)) & INDEX_NIBBLE_MASK;
}

/* A value at `place` of a block of the given kind, from 32 bits of a hash; every value is an fp16 exactly. */
__host__ __device__ inline float index_corner_value(uint32_t bits, uint32_t kind, uint32_t place)
{
    const float plain =
        ldexpf((float)(bits % INDEX_VALUE_STEPS) - (float)INDEX_VALUE_CENTRE, INDEX_VALUE_STEP_EXPONENT);
    const float sign = ((bits >> INDEX_SIGN_BIT_OF_HASH) & 1u) != 0 ? -1.0f : 1.0f;
    const uint32_t eighths = (bits >> INDEX_BYTE_BITS) & INDEX_EIGHTHS_MASK;
    const uint32_t midpoint = (bits >> INDEX_E4M3_MANTISSA_BITS) % (INDEX_E2M1_CODES - 1u);
    const int tie_exponent = 1 + (int)((bits >> INDEX_E4M3_MANTISSA_BITS) % INDEX_TIE_EXPONENTS) - INDEX_TIE_EXPONENT;
    switch (kind) {
    case INDEX_VALUE_ZEROS: return (place & 1u) != 0 ? plain : sign * 0.0f;
    case INDEX_VALUE_ZERO_BLOCK: return sign * 0.0f;
    case INDEX_VALUE_UNDER_FLOOR: return ldexpf(plain, INDEX_UNDER_FLOOR_EXPONENT);
    case INDEX_VALUE_E2M1_TIES:          /* halfway between two E2M1 values at the scale 2^-2 */
        return place == 0 ? INDEX_E2M1_TIES_LARGEST
                          : sign * ldexpf((float)(index_e2m1_doubled(midpoint) + index_e2m1_doubled(midpoint + 1u)),
                                          INDEX_E2M1_TIE_EXPONENT);
    case INDEX_VALUE_SPREAD:
        return sign * ldexpf(1.0f + (float)eighths / (float)INDEX_EIGHTHS,
                             (int)(place % INDEX_SPREAD_PLACES) - (INDEX_SPREAD_PLACES - 1));
    case INDEX_VALUE_E4M3_TIES:          /* (17 + 2 m) 2^(e - 19): halfway between two E4M3 codes at the scale 2^-8 */
        return place == 0 ? INDEX_E4M3_TIES_LARGEST
                          : sign * ldexpf((float)(INDEX_TIE_UNITS + eighths + eighths), tie_exponent);
    default: return plain;
    }
}
/* Dim `dim` of key `key`; of a query: head `head` of row `row`; a weight: a whole number of 2^-12, under 1/32. */
__host__ __device__ inline float index_key_value(uint32_t key, uint32_t dim)
{
    return index_corner_value(index_mix(INDEX_KEY_SEED + key * INDEX_DIMS + dim),
                              (key + dim / INDEX_DIMS_PER_BLOCK) % INDEX_VALUE_KINDS, dim % INDEX_DIMS_PER_BLOCK);
}
__host__ __device__ inline float index_query_value(uint32_t row, uint32_t head, uint32_t dim)
{
    const uint32_t kind = (row + head + dim / INDEX_DIMS_PER_BLOCK) % INDEX_VALUE_KINDS;
    return index_corner_value(index_mix(INDEX_QUERY_SEED + (row * INDEX_HEADS + head) * INDEX_DIMS + dim), kind,
                              dim % INDEX_DIMS_PER_BLOCK);
}
__host__ __device__ inline float index_weight_value(uint32_t row, uint32_t head)
{
    const uint32_t bits = index_mix(INDEX_WEIGHT_SEED + row * INDEX_HEADS + head);
    return ldexpf((float)(bits % INDEX_VALUE_STEPS) - (float)INDEX_VALUE_CENTRE, INDEX_WEIGHT_EXPONENT);
}

/* The largest magnitude of the warp's 32 values; the scale code its product with `inverse` gives. */
__device__ __forceinline__ uint32_t index_scale_code(float value, float floor, float inverse)
{
    float largest = fabsf(value);
    for (uint32_t distance = INDEX_LANES / INDEX_COLUMNS_PER_LANE; distance != 0; distance >>= 1u) {
        largest = fmaxf(largest, __shfl_xor_sync(INDEX_EVERY_LANE, largest, (int)distance));
    }
    const uint32_t product = __float_as_uint(fmaxf(largest, floor) * inverse);
    return (product >> INDEX_FLOAT32_EXPONENT_SHIFT) + ((product & INDEX_FLOAT32_MANTISSA_MASK) != 0 ? 1u : 0u);
}
/* TRAINED: the signed byte of twice the E2M1 value; *scale_code the block's. */
__device__ __forceinline__ uint32_t index_round_trained(float value, uint32_t *scale_code)
{
    *scale_code = index_scale_code(value, INDEX_E2M1_FLOOR, INDEX_E2M1_LARGEST_INVERSE);
    const float scaled = fminf(fmaxf(value * __uint_as_float((INDEX_TWICE_E8M0_BIAS - *scale_code)
                                                             << INDEX_FLOAT32_EXPONENT_SHIFT), -INDEX_E2M1_LARGEST),
                               INDEX_E2M1_LARGEST);
    const float quadrupled = fabsf(scaled) * (float)INDEX_E2M1_QUADRUPLE;
    uint32_t below = 0, at_or_below = 0;
    for (uint32_t code = 0; code + 1u < INDEX_E2M1_CODES; code++) {
        const float midpoint = (float)(index_e2m1_doubled(code) + index_e2m1_doubled(code + 1u));
        below += quadrupled > midpoint ? 1u : 0u;
        at_or_below += quadrupled >= midpoint ? 1u : 0u;
    }
    const uint32_t doubled = index_e2m1_doubled((below & 1u) == 0 ? below : at_or_below);
    return (scaled < 0.0f ? 0u - doubled : doubled) & INDEX_DIGIT_MASK;
}
/* FRESH: the E4M3 byte; *scale_code the block's. E4M3 numbers are 2^(f - 130) apart at a float32 exponent field f
 * and 2^-9 apart under 2^-6: the magnitude in those units, to nearest with a tie to even, onto the stretch's first. */
__device__ __forceinline__ uint32_t index_round_fresh(float value, uint32_t *scale_code)
{
    *scale_code = index_scale_code(value, INDEX_E4M3_FLOOR, INDEX_E4M3_LARGEST_INVERSE);
    const float scaled = value * __uint_as_float((INDEX_TWICE_E8M0_BIAS - *scale_code) << INDEX_FLOAT32_EXPONENT_SHIFT);
    const float magnitude = fminf(fabsf(scaled), INDEX_E4M3_LARGEST);
    const uint32_t field =
        __float_as_uint(fmaxf(magnitude, INDEX_E4M3_SMALLEST_NORMAL)) >> INDEX_FLOAT32_EXPONENT_SHIFT;
    const uint32_t units = (uint32_t)__float2int_rn(magnitude * __uint_as_float(
        (INDEX_E4M3_UNITS_EXPONENT + INDEX_E8M0_BIAS - field) << INDEX_FLOAT32_EXPONENT_SHIFT));
    return ((field << INDEX_E4M3_MANTISSA_BITS) + units - INDEX_E4M3_CODE_BASE) |
           ((__float_as_uint(scaled) >> INDEX_FLOAT32_SIGN_TO_BYTE) & INDEX_E4M3_SIGN_BIT);
}
template <int state>
__device__ __forceinline__ uint32_t index_round(float value, uint32_t *scale_code)
{
    return state == INDEX_STATE_TRAINED ? index_round_trained(value, scale_code) : index_round_fresh(value, scale_code);
}
__device__ __forceinline__ float index_half_value(float value) { return __half2float(__float2half_rn(value)); }
__device__ __forceinline__ uint32_t index_half_bits(float value) { return __half_as_ushort(__float2half_rn(value)); }
/* Where an 8-bit record or a placed query of codes puts dim d of a block: the byte of a 16-dim step's word. */
__host__ __device__ inline uint32_t index_byte_of_dim(uint32_t dim)
{
    const uint32_t within = dim % INDEX_DIMS_PER_STEP;
    return (within & 1u) * INDEX_PAIR_IN_REGISTER + within / INDEX_DIMS_PER_HALF_STEP;
}
__host__ __device__ inline uint32_t index_place_of_dim(uint32_t dim)             /* its lane place t */
{
    return dim % INDEX_DIMS_PER_HALF_STEP / INDEX_PAIR_IN_REGISTER;
}

/* An FP16 ring: word w is of key w / 64; of its run t, step j, register r: dims 16 j + 2 t + 8 r and the next. */
__global__ void index_fill_half(uint32_t *words, uint64_t word_count)
{
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < word_count; index += stride) {
        const uint32_t key = (uint32_t)(index / INDEX_HALF_WORDS_PER_KEY);
        const uint32_t word = (uint32_t)(index % INDEX_HALF_WORDS_PER_KEY);
        const uint32_t dim = word % INDEX_HALF_WORDS_PER_RUN / INDEX_REGISTERS_PER_STEP * INDEX_DIMS_PER_STEP +
                             word / INDEX_HALF_WORDS_PER_RUN * INDEX_PAIR_IN_REGISTER +
                             (word & 1u) * INDEX_DIMS_PER_HALF_STEP;
        words[index] = index_half_bits(index_key_value(key, dim)) |
                       (index_half_bits(index_key_value(key, dim + 1u)) << INDEX_HALF_BITS);
    }
}
/* An 8-bit ring of a kind: a warp a block of a key at a time, rounding the fp16 of each value. */
template <int state>
__global__ void index_fill_codes(unsigned char *ring, uint64_t key_count)
{
    const uint32_t lane = threadIdx.x % INDEX_LANES;
    const uint64_t warps = (uint64_t)gridDim.x * blockDim.x / INDEX_LANES;
    for (uint64_t index = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / INDEX_LANES;
         index < key_count * INDEX_BLOCKS; index += warps) {
        const uint32_t key = (uint32_t)(index / INDEX_BLOCKS), block = (uint32_t)(index % INDEX_BLOCKS);
        const uint32_t dim = block * INDEX_DIMS_PER_BLOCK + lane;
        uint32_t scale_code = 0;
        const uint32_t code = index_round<state>(index_half_value(index_key_value(key, dim)), &scale_code);
        unsigned char *tile = ring + (size_t)(key / INDEX_KEYS_PER_TILE) * INDEX_CODE_TILE_BYTES;
        tile[key % INDEX_KEYS_PER_TILE * INDEX_CODE_KEY_BYTES + index_place_of_dim(dim) * INDEX_CODE_RUN_BYTES +
             dim / INDEX_DIMS_PER_STEP * INDEX_WORD_BYTES + index_byte_of_dim(dim)] = (unsigned char)code;
        if (lane == 0) {
            tile[INDEX_CODE_SCALES_AT + key % INDEX_KEYS_PER_TILE * INDEX_WORD_BYTES + block] =
                (unsigned char)scale_code;
        }
    }
}
/* Queries as their producer places them. FP16: a thread a word of [row][head tile][step][lane][register]. */
__global__ void index_place_half(const uint16_t *natural, uint32_t *placed, uint32_t row_count)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= row_count * INDEX_HALF_WORDS_PER_ROW) {
        return;
    }
    const uint32_t word = index % INDEX_HALF_WORDS_PER_ROW, lane = word / INDEX_WORDS_PER_LOAD % INDEX_LANES;
    const uint32_t step = word / INDEX_HALF_WORDS_PER_TILE_STEP % INDEX_STEPS;
    const uint32_t operand_register = word % INDEX_WORDS_PER_LOAD;
    const uint32_t head = word / (INDEX_HALF_WORDS_PER_TILE_STEP * INDEX_STEPS) * INDEX_HEADS_PER_TILE +
                          lane / INDEX_LANE_PLACES + (operand_register & 1u) * INDEX_HALF_TILE;
    const uint32_t dim = step * INDEX_DIMS_PER_STEP + lane % INDEX_LANE_PLACES * INDEX_PAIR_IN_REGISTER +
                         operand_register / INDEX_PAIR_IN_REGISTER * INDEX_DIMS_PER_HALF_STEP;
    const uint16_t *values =
        natural + ((size_t)(index / INDEX_HALF_WORDS_PER_ROW) * INDEX_HEADS + head) * INDEX_DIMS + dim;
    placed[index] = (uint32_t)values[0] | ((uint32_t)values[1] << INDEX_HALF_BITS);
}
/* The rows' queries in natural order [row][head][dim], fp16: what their producer would place. */
__global__ void index_fill_queries(uint16_t *natural, uint32_t row_count)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row_count * INDEX_HEADS * INDEX_DIMS) {
        const uint32_t row = index / (INDEX_HEADS * INDEX_DIMS), head = index / INDEX_DIMS % INDEX_HEADS;
        natural[index] = (uint16_t)index_half_bits(index_query_value(row, head, index % INDEX_DIMS));
    }
}
/* Codes of a kind: a warp a block of a head of a row, rounded; byte i of register r of lane 4 g + t (the record). */
template <int state>
__global__ void index_place_codes(const uint16_t *natural, unsigned char *placed, uint32_t row_count)
{
    const uint32_t lane = threadIdx.x % INDEX_LANES;
    const uint32_t index = (blockIdx.x * blockDim.x + threadIdx.x) / INDEX_LANES;
    if (index >= row_count * INDEX_HEADS * INDEX_BLOCKS) {
        return;
    }
    const uint32_t row = index / (INDEX_HEADS * INDEX_BLOCKS), head = index / INDEX_BLOCKS % INDEX_HEADS;
    const uint32_t block = index % INDEX_BLOCKS, dim = block * INDEX_DIMS_PER_BLOCK + lane;
    const uint32_t in_tile = head % INDEX_HEADS_PER_TILE, upper_dims = lane / INDEX_DIMS_PER_STEP;
    const uint32_t operand_register = in_tile / INDEX_HALF_TILE + upper_dims * INDEX_PAIR_IN_REGISTER;
    const uint32_t lane_of_value = in_tile % INDEX_HALF_TILE * INDEX_LANE_PLACES + index_place_of_dim(dim);
    uint32_t scale_code = 0;
    const uint32_t code = index_round<state>(
        __half2float(__ushort_as_half(natural[((size_t)row * INDEX_HEADS + head) * INDEX_DIMS + dim])), &scale_code);
    unsigned char *row_bytes = placed + (size_t)row * INDEX_CODE_QUERY_BYTES;
    row_bytes[((head / INDEX_HEADS_PER_TILE * INDEX_BLOCKS + block) * INDEX_LANES + lane_of_value) * INDEX_LOAD_BYTES +
              operand_register * INDEX_WORD_BYTES + index_byte_of_dim(dim)] = (unsigned char)code;
    if (lane == 0) {
        row_bytes[INDEX_QUERY_SCALES_AT + head * INDEX_WORD_BYTES + block] = (unsigned char)scale_code;
    }
}

/* ---- the two bounds: the baseline's grid and tiles (rows on the fast axis), its loads or its instructions ---- */

/* A word of a form's operand: all ones; all twos (twos); or ones with the first value 2 and the next 0 (skewed). */
template <int form>
__device__ __forceinline__ uint32_t index_bound_word(int twos, int skewed)
{
    return form == INDEX_FORM_HALF || form == INDEX_FORM_PLACED
               ? (twos ? INDEX_FP16_TWOS : skewed ? INDEX_FP16_SKEWED : INDEX_FP16_ONES)
           : form == INDEX_FORM_WHOLE ? (twos ? INDEX_INT8_TWOS : skewed ? INDEX_INT8_SKEWED : INDEX_INT8_ONES)
           : form == INDEX_FORM_E4M3 ? (twos ? INDEX_E4M3_TWOS : skewed ? INDEX_E4M3_SKEWED : INDEX_E4M3_ONES)
           : (twos ? INDEX_E2M1_TWOS : skewed ? INDEX_E2M1_SKEWED : INDEX_E2M1_ONES);
}
/* One step of a head tile as the scan issues it, on operands made in registers whose sums the host knows: A (a word)
 * is 1 in head tile 0 and 2 in head tile 1; B (first, second) is 0 on an even tile and, on an odd one, ones of which
 * the step's pattern skews none, one or both registers, so that a column of B sums to 1 a dim and no two steps have
 * the same operands. HALF: one m16n8k16; PLACED: two (a block); WHOLE (in its own integers), E4M3, E2M1: one
 * m16n8k32. Interleaved, as the scan: HALF on its head tile's chain; an 8-bit block from a fresh sum, `zero`, then
 * added. Chained: every instruction on the row's one sum. */
template <int form, int bound>
__device__ __forceinline__ void index_bound_step(float4 *chain, int4 *whole_chain, uint32_t word, uint32_t first,
                                                 uint32_t second, uint32_t zero)
{
    const uint4 a = make_uint4(word, word, word, word);
    const int fresh = bound == INDEX_BOUND_INTERLEAVED && form != INDEX_FORM_HALF;
    float4 part = make_float4(__uint_as_float(zero), __uint_as_float(zero), __uint_as_float(zero),
                              __uint_as_float(zero)), *sums = fresh ? &part : chain;
    int4 whole_part = make_int4((int)zero, (int)zero, (int)zero, (int)zero), *whole = fresh ? &whole_part : whole_chain;
    if (form == INDEX_FORM_HALF || form == INDEX_FORM_PLACED) {
        index_mma_half(sums, a, first, second);
        if (form == INDEX_FORM_PLACED) {
            index_mma_half(sums, a, first, second);
        }
    } else if (form == INDEX_FORM_WHOLE) {
        index_mma_whole(whole, a, first, second);
    } else if (form == INDEX_FORM_E4M3) {
        index_mma_e4m3(sums, a, first, second);
    } else {
        index_mma_e2m1(sums, a, first, second, INDEX_SCALE_ONES, INDEX_SCALE_ONES);
    }
    if (fresh && form == INDEX_FORM_WHOLE) {
        *whole_chain = make_int4(whole_chain->x + whole_part.x, whole_chain->y + whole_part.y,
                                 whole_chain->z + whole_part.z, whole_chain->w + whole_part.w);
    } else if (fresh) {
        *chain = make_float4(chain->x + part.x, chain->y + part.y, chain->z + part.z, chain->w + part.w);
    }
}
/* read_only: every word the baseline loads, folded by exclusive or, then to 16 bits (FP16) or 8 (codes): the fold
 * of the halves or bytes of every key of the block's tiles; a key's scale word once (lane group 0). tensor_only: the
 * baseline's instructions (16 m16n8k16 or 8 m16n8k32 a row and tile, 65,536 products each); interleaved, as the
 * scan: in its order (per step its rows', per row its two head tiles'), a sum a row and head tile; chained: one sum a
 * row, so that each of a row's instructions waits on the one before, per step the head tiles' turns of the rows (the
 * other rows between two of a row's). A block's fold or sum to folds[block]. */
template <int form, int group_rows, int bound>
__device__ __forceinline__ void index_bound_block(const struct index_scan_arguments *arguments, uint32_t *folds,
                                                  uint32_t *shared)
{
    const uint32_t lane = threadIdx.x % INDEX_LANES, warp = threadIdx.x / INDEX_LANES, warps = blockDim.x / INDEX_LANES;
    const uint32_t lane_group = lane / INDEX_LANE_PLACES, lane_place = lane % INDEX_LANE_PLACES;
    const uint32_t steps = form == INDEX_FORM_HALF ? INDEX_STEPS : INDEX_BLOCKS;
    const int chained = bound == INDEX_BOUND_CHAINED;
    uint32_t extents[group_rows], fold = 0;
    const struct index_place place = index_block_place<group_rows, 1, 0>(arguments, extents);
    /* zeros the assembler cannot see (check 10 holds every extent and tile index under 2^31): A is kept in registers,
     * never made again in a result's, and no row's fresh sum is another's, so that no instruction waits on another
     * or is merged with another (as in the scan, whose operands all differ) */
    const uint32_t hidden = place.first_tile >> INDEX_BOUND_ZERO_SHIFT;
    const uint32_t a_words[INDEX_HEAD_TILES] = {index_bound_word<form>(0, 0) ^ hidden,
                                                index_bound_word<form>(1, 0) ^ hidden};
    float4 sums[group_rows][INDEX_HEAD_TILES];
    int4 whole[group_rows][INDEX_HEAD_TILES];
#pragma unroll
    for (uint32_t row = 0; row < group_rows; row++) {
        sums[row][0] = sums[row][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        whole[row][0] = whole[row][1] = make_int4(0, 0, 0, 0);
    }
    for (uint32_t tile = place.first_tile + warp; tile < place.end_tile; tile += warps) {
        const unsigned char *keys = arguments->keys + (size_t)tile * index_tile_bytes(form);
        if (bound == INDEX_BOUND_READ && form == INDEX_FORM_HALF) {
            const uint4 *run =
                (const uint4 *)(keys + lane_group * INDEX_HALF_KEY_BYTES + lane_place * INDEX_HALF_RUN_BYTES);
#pragma unroll
            for (uint32_t load = 0; load < INDEX_HALF_RUN_LOADS; load++) {
                const uint4 words = run[load];
                fold ^= words.x ^ words.y ^ words.z ^ words.w;
            }
        } else if (bound == INDEX_BOUND_READ) {
            const uint4 *run =
                (const uint4 *)(keys + lane_group * INDEX_CODE_KEY_BYTES + lane_place * INDEX_CODE_RUN_BYTES);
            const uint32_t *key_scales = (const uint32_t *)(keys + INDEX_CODE_SCALES_AT);
            const uint4 first = run[0], second = run[1];
            const uint32_t even = key_scales[lane_place * INDEX_COLUMNS_PER_LANE];
            const uint32_t odd = key_scales[lane_place * INDEX_COLUMNS_PER_LANE + 1u];
            fold ^= first.x ^ first.y ^ first.z ^ first.w ^ second.x ^ second.y ^ second.z ^ second.w ^
                    (lane_group == 0 ? even ^ odd : 0u);
        } else {
#pragma unroll
            for (uint32_t step = 0; step < steps; step++) {
                const uint32_t odd = tile & 1u, pattern = step % INDEX_BOUND_PATTERNS;
                const uint32_t first = odd * index_bound_word<form>(0, (pattern & 1u) != 0);
                const uint32_t second = odd * index_bound_word<form>(0, (pattern >> 1 & 1u) != 0);
#pragma unroll
                for (uint32_t order = 0; order < group_rows * INDEX_HEAD_TILES; order++) {
                    const uint32_t row = chained ? order % group_rows : order / INDEX_HEAD_TILES;
                    const uint32_t head_tile = chained ? order / group_rows : order % INDEX_HEAD_TILES;
                    if (row < place.rows) {
                        index_bound_step<form, bound>(&sums[row][chained ? 0u : head_tile],
                                                      &whole[row][chained ? 0u : head_tile], a_words[head_tile],
                                                      first, second, extents[row] >> INDEX_BOUND_ZERO_SHIFT);
                    }
                }
            }
        }
    }
#pragma unroll
    for (uint32_t row = 0; bound != INDEX_BOUND_READ && row < group_rows; row++) {
        for (uint32_t head_tile = 0; head_tile < INDEX_HEAD_TILES; head_tile++) {   /* whole numbers under 2^24 */
            const float4 sum = sums[row][head_tile];
            const int4 sum_whole = whole[row][head_tile];
            fold += (uint32_t)sum.x + (uint32_t)sum.y + (uint32_t)sum.z + (uint32_t)sum.w + (uint32_t)sum_whole.x +
                    (uint32_t)sum_whole.y + (uint32_t)sum_whole.z + (uint32_t)sum_whole.w;
        }
    }
    const uint32_t halves = (fold ^ (fold >> INDEX_HALF_BITS)) & INDEX_HALF_MASK;   /* of a read: the halves' fold */
    fold = bound != INDEX_BOUND_READ ? fold
           : form == INDEX_FORM_HALF ? halves : index_byte(halves, 0) ^ index_byte(halves, 1);
    for (uint32_t distance = INDEX_LANES / INDEX_COLUMNS_PER_LANE; distance != 0; distance >>= 1u) {
        const uint32_t other = __shfl_xor_sync(INDEX_EVERY_LANE, fold, (int)distance);
        fold = bound == INDEX_BOUND_READ ? fold ^ other : fold + other;
    }
    if (lane == 0) {
        shared[warp] = fold;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t other = 1; other < warps; other++) {
            fold = bound == INDEX_BOUND_READ ? fold ^ shared[other] : fold + shared[other];
        }
        folds[blockIdx.y * gridDim.x + blockIdx.x] = fold;
    }
}

/* ---- the stamps: every kernel the probe launches ---- */

#define INDEX_SCAN_SHAPES(STAMP, name, form)                                                                         \
    STAMP(name##_group, form, INDEX_MOST_GROUP_ROWS, 1, 0, 0)                                                        \
    STAMP(name##_single, form, 1, 1, 0, 0)                                                                           \
    STAMP(name##_row_major, form, 1, 0, 0, 0)                                                                        \
    STAMP(name##_listed, form, 1, 1, 1, 0)                                                                           \
    STAMP(name##_head_order, form, INDEX_MOST_GROUP_ROWS, 1, 0, 1)
/* Every form in its order; a form only sm_120a has (E4M3, and E2M1 behind INDEX_E2M1_SCALED) is stamped in that
 * build alone (CONVENTIONS 1.1), elsewhere ABSENT: no kernel, a row of no function in the tables. */
#define INDEX_SM_120A 120                        /* -DPROBE_SM of the sm_120a build, for the preprocessor */
#if PROBE_SM == INDEX_SM_120A
#define INDEX_E4M3_STAMP(STAMP, ABSENT) STAMP
#else
#define INDEX_E4M3_STAMP(STAMP, ABSENT) ABSENT
#endif
#if PROBE_SM == INDEX_SM_120A && INDEX_E2M1_SCALED
#define INDEX_E2M1_STAMP(STAMP, ABSENT) STAMP
#else
#define INDEX_E2M1_STAMP(STAMP, ABSENT) ABSENT
#endif
#define INDEX_EVERY_FORM(SHAPES, STAMP, ABSENT)                                                                      \
    SHAPES(STAMP, half, INDEX_FORM_HALF) SHAPES(STAMP, whole, INDEX_FORM_WHOLE)                                      \
    SHAPES(INDEX_E2M1_STAMP(STAMP, ABSENT), e2m1, INDEX_FORM_E2M1)                                                   \
    SHAPES(INDEX_E4M3_STAMP(STAMP, ABSENT), e4m3, INDEX_FORM_E4M3) SHAPES(STAMP, placed, INDEX_FORM_PLACED)
#define INDEX_STAMP_NONE(...)
#define INDEX_STAMP_ABSENT_ROW(...) {NULL, "absent"},
#define INDEX_STAMP_SCAN(name, form, group_rows, rows_fast, listed, tree)                                            \
    INDEX_SCAN_KERNEL(index_scan_##name, form, group_rows, rows_fast, listed, tree)
INDEX_EVERY_FORM(INDEX_SCAN_SHAPES, INDEX_STAMP_SCAN, INDEX_STAMP_NONE)
/* the bounds of a form: a row group of 8 or a row alone (the baseline's two), each read, chained, interleaved */
#define INDEX_BOUND_SHAPES(STAMP, name, form)                                                                        \
    STAMP(name##_read_group, form, INDEX_MOST_GROUP_ROWS, INDEX_BOUND_READ)                                          \
    STAMP(name##_read_single, form, 1, INDEX_BOUND_READ)                                                             \
    STAMP(name##_chained_group, form, INDEX_MOST_GROUP_ROWS, INDEX_BOUND_CHAINED)                                    \
    STAMP(name##_chained_single, form, 1, INDEX_BOUND_CHAINED)                                                       \
    STAMP(name##_interleaved_group, form, INDEX_MOST_GROUP_ROWS, INDEX_BOUND_INTERLEAVED)                            \
    STAMP(name##_interleaved_single, form, 1, INDEX_BOUND_INTERLEAVED)
/* a warp's fold in dynamic shared memory (INDEX_BOUND_SHARED_BYTES at least): no static part, so that the padding
 * of --blocks-per-sm may take all an SM's shared memory */
#define INDEX_STAMP_BOUND(name, form, group_rows, bound)                                                             \
    __global__ void index_bound_##name(struct index_scan_arguments arguments, uint32_t *folds)                       \
    {                                                                                                                \
        extern __shared__ uint32_t index_shared_folds[];                                                             \
        index_bound_block<form, group_rows, bound>(&arguments, folds, index_shared_folds);                           \
    }
INDEX_EVERY_FORM(INDEX_BOUND_SHAPES, INDEX_STAMP_BOUND, INDEX_STAMP_NONE)
/* each kernel's address and name, in stamping order: form by form, its shapes in their order; NULL: absent */
struct index_kernel { const void *function; const char *name; };
#define INDEX_STAMP_SCAN_ROW(name, ...) {(const void *)index_scan_##name, "index_scan_" #name},
#define INDEX_STAMP_BOUND_ROW(name, ...) {(const void *)index_bound_##name, "index_bound_" #name},
static const struct index_kernel index_scans[INDEX_FORMS * INDEX_SHAPES] = {
    INDEX_EVERY_FORM(INDEX_SCAN_SHAPES, INDEX_STAMP_SCAN_ROW, INDEX_STAMP_ABSENT_ROW)};
static const struct index_kernel index_bounds[INDEX_FORMS * INDEX_BOUND_SHAPES] = {
    INDEX_EVERY_FORM(INDEX_BOUND_SHAPES, INDEX_STAMP_BOUND_ROW, INDEX_STAMP_ABSENT_ROW)};
#define INDEX_SELECT_KERNELS "index_select_one_block+index_select_spans+index_select_merge"

/* ---- which stamp a launch of the probe takes ---- */

enum {                                           /* what an instance launches; its keys: the case's state, or its own */
    INDEX_SCAN = 0, INDEX_WIDE, INDEX_ROW_MAJOR, INDEX_KEY_MAJOR, INDEX_LISTED, INDEX_READ, INDEX_CHAINED,
    INDEX_INTERLEAVED, INDEX_ONE_BLOCK, INDEX_SPANS, INDEX_SCAN_SELECT, INDEX_HEAD_ORDER,
    INDEX_FOLLOW = 0, INDEX_OWN_HALF, INDEX_OWN_E2M1, INDEX_OWN_PLACED
};
#define INDEX_NO_KERNEL UINT32_MAX
/* The form of an instance's keys in a state: fp16 HALF, trained WHOLE, fresh E4M3 on sm_120a and PLACED on sm_80. */
inline int index_form_of_keys(uint32_t keys, uint32_t state, int sm_120a)
{
    return keys == INDEX_OWN_HALF ? INDEX_FORM_HALF : keys == INDEX_OWN_E2M1 ? INDEX_FORM_E2M1
           : keys == INDEX_OWN_PLACED ? INDEX_FORM_PLACED : state == INDEX_STATE_FP16 ? INDEX_FORM_HALF
           : state == INDEX_STATE_TRAINED ? INDEX_FORM_WHOLE : sm_120a ? INDEX_FORM_E4M3 : INDEX_FORM_PLACED;
}
inline uint32_t index_key_state(int form)        /* what a form's keys are, and its queries (PLACED's: fp16) */
{
    return form == INDEX_FORM_HALF ? (uint32_t)INDEX_STATE_FP16
           : form <= INDEX_FORM_E2M1 ? (uint32_t)INDEX_STATE_TRAINED : (uint32_t)INDEX_STATE_FRESH;
}
inline uint32_t index_query_state(int form)
{
    return form == INDEX_FORM_PLACED ? (uint32_t)INDEX_STATE_FP16 : index_key_state(form);
}
inline uint64_t index_plane_bytes(int form, uint32_t entries)
{
    return (uint64_t)index_tiles_of(entries) * index_tile_bytes(form);
}
inline int index_is_bound(uint32_t family) { return family >= INDEX_READ && family <= INDEX_INTERLEAVED; }
inline int index_is_per_row(uint32_t family)      /* a row a block: checked, and so timed, up to 8 rows */
{
    return family == INDEX_ROW_MAJOR || family == INDEX_KEY_MAJOR || family == INDEX_LISTED;
}
inline const char *index_reads_of(uint32_t family)      /* a step's key plane is cold (CONVENTIONS 3.5); the slab a
                                                           select reads was just written, so warm in L2 on purpose */
{
    return family == INDEX_ONE_BLOCK || family == INDEX_SPANS ? "warm_slab" : family == INDEX_SCAN_SELECT
           ? "cold_keys,warm_slab" : family == INDEX_CHAINED || family == INDEX_INTERLEAVED ? "none" : "cold_keys";
}
inline uint32_t index_group_of(uint32_t family, uint32_t rows)            /* rows a block of a scan-shaped grid */
{
    const int grouped = family == INDEX_SCAN || family == INDEX_WIDE || family == INDEX_SCAN_SELECT ||
                        index_is_bound(family);
    return family == INDEX_HEAD_ORDER || (rows > 1 && grouped) ? (uint32_t)INDEX_MOST_GROUP_ROWS : 1u;
}
/* The stamped scan or bound kernel of a family, form and group size; INDEX_NO_KERNEL where it launches none. */
inline uint32_t index_scan_kernel(uint32_t family, int form, uint32_t group)
{
    const uint32_t shape = family == INDEX_HEAD_ORDER ? INDEX_SHAPE_HEAD_ORDER : family == INDEX_LISTED
                           ? INDEX_SHAPE_LISTED : family == INDEX_ROW_MAJOR ? INDEX_SHAPE_ROW_MAJOR
                           : group == 1 ? INDEX_SHAPE_SINGLE : INDEX_SHAPE_GROUP;
    return family == INDEX_ONE_BLOCK || family == INDEX_SPANS || index_is_bound(family)
               ? INDEX_NO_KERNEL : (uint32_t)form * INDEX_SHAPES + shape;
}
inline uint32_t index_bound_kernel(uint32_t family, int form, uint32_t group)
{
    const uint32_t shape = (family - INDEX_READ) * INDEX_BOUND_SIZES + (group == 1 ? 1u : 0u);
    return !index_is_bound(family) ? INDEX_NO_KERNEL : (uint32_t)form * INDEX_BOUND_SHAPES + shape;
}
inline uint32_t index_threads_of(uint32_t family) { return family == INDEX_WIDE ? INDEX_WIDE_THREADS : INDEX_THREADS; }
/* A scan or bound kernel's place in one table of both, index_scans then index_bounds; INDEX_NO_KERNEL: none. */
inline uint32_t index_slot_of(uint32_t scan_kernel, uint32_t bound_kernel)
{
    return scan_kernel != INDEX_NO_KERNEL ? scan_kernel : bound_kernel != INDEX_NO_KERNEL
           ? (uint32_t)(INDEX_FORMS * INDEX_SHAPES) + bound_kernel : INDEX_NO_KERNEL;
}
/* 0 where a kernel a family launches in a form (its group's or a row alone's) cannot take the family's block, by
 * most_threads (a slot's: a block's threads as the kernel's registers allow; 0 where the build lacks it); *most: that
 * kernel's. On sm_80 the 8-bit group kernels need 129 and 145 registers: warps:16's 512 threads do not fit an SM's
 * 65,536 (i11a of 2026-10-12: "too many resources requested for launch"). */
inline int index_launchable(const uint32_t *most_threads, uint32_t family, int form, uint32_t *most)
{
    for (uint32_t group = 1; group <= INDEX_MOST_GROUP_ROWS; group += INDEX_MOST_GROUP_ROWS - 1u) {
        const uint32_t slot = index_slot_of(index_scan_kernel(family, form, group),
                                            index_bound_kernel(family, form, group));
        *most = slot != INDEX_NO_KERNEL ? most_threads[slot] : 0u;
        if (*most != 0 && *most < index_threads_of(family)) {
            return 0;
        }
    }
    return 1;
}
struct index_step {                              /* the launches of one step, in the order they are made */
    uint32_t instance, form, family, select, scan_kernel, bound_kernel, threads;   /* INDEX_NO_KERNEL: none */
    uint32_t entries;                            /* of its plane: what every extent is held to */
    dim3 grid, select_grid;
    size_t shared_bytes;
    const void *ring, *queries, *scores;         /* the card arrays its pointers lie in */
    struct index_scan_arguments scan;
    struct index_select_arguments choose;
};
/* The scan or bound kernel a step launches on its grid; NULL: none, or a form this build lacks. */
inline const void *index_function_of(const struct index_step *step)
{
    return step->scan_kernel != INDEX_NO_KERNEL ? index_scans[step->scan_kernel].function
           : step->bound_kernel != INDEX_NO_KERNEL ? index_bounds[step->bound_kernel].function : NULL;
}

inline uint32_t index_largest_extent(uint32_t first_row, uint32_t rows, const uint32_t *extents)
{
    uint32_t largest = 0;
    for (uint32_t row = first_row; row < first_row + rows; row++) {
        largest = extents[row] > largest ? extents[row] : largest;
    }
    return largest;
}
/* The tiles of the group of `group` rows from first_row (of `rows`): those of its largest extent. */
inline uint32_t index_group_tiles(uint32_t first_row, uint32_t rows, uint32_t group, const uint32_t *extents)
{
    const uint32_t in_group = rows - first_row < group ? rows - first_row : group;
    return index_tiles_of(index_largest_extent(first_row, in_group, extents));
}
/* Key bytes a step of a family reads and the products it makes, from rows' extents: a select reads a row once. */
inline void index_step_work(uint32_t family, int form, uint32_t rows, const uint32_t *extents, double *bytes,
                            double *flops)
{
    const uint32_t group = index_group_of(family, rows);
    *bytes = *flops = 0.0;
    for (uint32_t first = 0; family != INDEX_ONE_BLOCK && family != INDEX_SPANS && first < rows; first += group) {
        const uint32_t in_group = rows - first < group ? rows - first : group;
        const double tiles = family == INDEX_LISTED ? index_listed_tiles(extents[first])
                                                    : index_group_tiles(first, rows, group, extents);
        *bytes += family == INDEX_CHAINED || family == INDEX_INTERLEAVED ? 0.0 : tiles * index_tile_bytes(form);
        *flops += family == INDEX_READ ? 0.0 : tiles * INDEX_KEYS_PER_TILE * INDEX_FLOPS_PER_KEY * in_group;
    }
    for (uint32_t row = 0; family >= INDEX_ONE_BLOCK && family <= INDEX_SCAN_SELECT && row < rows; row++) {
        *bytes += (double)extents[row] * sizeof(uint16_t);
    }
}
/* A key as the FP16 ring holds it (key_or_row a key), or a row's query of a head (a row): fp16s in dim order. */
inline void index_natural(uint32_t key_or_row, uint32_t head, int query, double values[INDEX_DIMS])
{
    for (uint32_t dim = 0; dim < INDEX_DIMS; dim++) {
        values[dim] = query ? (double)index_query_value(key_or_row, head, dim)
                            : (double)index_key_value(key_or_row, dim);
    }
}
/* A ring of a state filled; the queries of `rows` rows placed for the three forms of query (FP16, TRAINED, E4M3). */
inline void index_fill_ring(uint32_t state, unsigned char *ring, uint64_t tiles, unsigned int blocks)
{
    if (state == INDEX_STATE_FP16) {
        index_fill_half<<<blocks, INDEX_FILL_THREADS>>>((uint32_t *)ring,
                                                        tiles * INDEX_HALF_TILE_BYTES / INDEX_WORD_BYTES);
    } else if (state == INDEX_STATE_TRAINED) {
        index_fill_codes<INDEX_STATE_TRAINED><<<blocks, INDEX_FILL_THREADS>>>(ring, tiles * INDEX_KEYS_PER_TILE);
    } else {
        index_fill_codes<INDEX_STATE_FRESH><<<blocks, INDEX_FILL_THREADS>>>(ring, tiles * INDEX_KEYS_PER_TILE);
    }
}
inline void index_place_queries(uint16_t *natural, unsigned char *const placed[INDEX_STATES], uint32_t rows)
{
    const uint32_t warps = INDEX_FILL_THREADS / INDEX_LANES;
    index_fill_queries<<<(rows * INDEX_HEADS * INDEX_DIMS + INDEX_FILL_THREADS - 1u) / INDEX_FILL_THREADS,
                         INDEX_FILL_THREADS>>>(natural, rows);
    index_place_half<<<(rows * INDEX_HALF_WORDS_PER_ROW + INDEX_FILL_THREADS - 1u) / INDEX_FILL_THREADS,
                       INDEX_FILL_THREADS>>>(natural, (uint32_t *)placed[INDEX_STATE_FP16], rows);
    index_place_codes<INDEX_STATE_TRAINED><<<rows * INDEX_HEADS * INDEX_BLOCKS / warps, INDEX_FILL_THREADS>>>(
        natural, placed[INDEX_STATE_TRAINED], rows);
    index_place_codes<INDEX_STATE_FRESH><<<rows * INDEX_HEADS * INDEX_BLOCKS / warps, INDEX_FILL_THREADS>>>(
        natural, placed[INDEX_STATE_FRESH], rows);
}

/* What a bound's block must hold, as the host sees it: read_only the fold of its tiles' key folds, tensor_only 49,152
 * (32 lanes of 4 sums of 384) a row of its group and odd tile (a word that wraps as the card's does). The group from
 * first_row, a segment. */
inline uint32_t index_bound_expected(uint32_t family, uint32_t first_row, uint32_t segment, uint32_t segment_tiles,
                                     uint32_t rows, const uint32_t *extents, const uint16_t *key_folds)
{
    const uint32_t group = index_group_of(family, rows), in_group = rows - first_row < group ? rows - first_row : group;
    const uint32_t end = index_group_tiles(first_row, rows, group, extents);
    uint32_t expected = 0;
    for (uint32_t tile = segment * segment_tiles; tile < end && tile < (segment + 1u) * segment_tiles; tile++) {
        for (uint32_t key = 0; family == INDEX_READ && key < INDEX_KEYS_PER_TILE; key++) {
            expected ^= key_folds[tile * INDEX_KEYS_PER_TILE + key];
        }
        expected += family == INDEX_READ ? 0u : (tile & 1u) * INDEX_BOUND_WARP_SUM * in_group;
    }
    return expected;
}
/* Words of a listed scan's [row][16,384] that differ from a dense slab's [row][capacity] at the listed entries. */
inline uint64_t index_listed_differing(const uint16_t *listed, const uint16_t *dense, uint32_t capacity,
                                       const uint32_t *lists, const uint32_t *extents, uint32_t rows, uint64_t *of)
{
    uint64_t differing = 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t at = 0; at < index_listed_tiles(extents[row]) * INDEX_KEYS_PER_TILE; at++, (*of)++) {
            const uint32_t tile = lists[(size_t)row * INDEX_CANDIDATE_BLOCKS + at / INDEX_KEYS_PER_TILE];
            const size_t entry = (size_t)tile * INDEX_KEYS_PER_TILE + at % INDEX_KEYS_PER_TILE;
            differing += listed[(size_t)row * INDEX_CANDIDATE_ENTRIES + at] != dense[(size_t)row * capacity + entry]
                             ? 1u : 0u;
        }
    }
    return differing;
}
/* The checks' sample of a row's entries: all up to 4,096, else the first and last 1,024 and 2,048 spread between. */
enum { INDEX_SAMPLES = 4096, INDEX_LEADING = 1024 };
inline uint32_t index_sampled_entry(uint32_t extent, uint32_t sample)
{
    return extent <= INDEX_SAMPLES || sample < INDEX_LEADING ? sample
           : sample >= INDEX_SAMPLES - INDEX_LEADING ? extent - (INDEX_SAMPLES - sample)
           : INDEX_LEADING + (uint32_t)((uint64_t)(sample - INDEX_LEADING) * (extent - INDEX_LEADING - INDEX_LEADING) /
                                       (INDEX_SAMPLES - INDEX_LEADING - INDEX_LEADING));
}
/* A row's candidate blocks (layers 24 .. 36 select among layer 20's 2,048): all it sees, or 2,047 of all but its
 * newest by a hash (selection sampling keeps them ascending and exactly that many), then the newest. */
inline void index_make_list(uint32_t row, uint32_t extent, uint32_t *list)
{
    const uint32_t blocks = index_tiles_of(extent), sampled = blocks > INDEX_CANDIDATE_BLOCKS;
    uint32_t count = 0, needed = INDEX_CANDIDATE_BLOCKS - 1u;
    for (uint32_t block = 0; block + 1u < blocks; block++) {
        if (!sampled || index_mix(index_mix(row) + block) % (blocks - 1u - block) < needed) {
            list[count++] = block;
            needed -= sampled ? 1u : 0u;
        }
    }
    if (blocks > 0) {
        list[count] = blocks - 1u;
    }
}

/* ---- the instances and the acceptance figures (index_scan_probe.cu stamps the table of the harness) ---- */

/* id, variant, launches, keys, class claimed, 1: its words must be the baseline's, the architecture it needs (0:
 * both), 1: it needs the block-scaled E2M1 form. The last joins with --break-exactness. */
#define INDEX_INSTANCES(ROW)                                                                                         \
    ROW(baseline, "baseline", INDEX_SCAN, INDEX_FOLLOW, 'R', 0, 0, 0)                                                 \
    ROW(baseline_again, "baseline_again", INDEX_SCAN, INDEX_FOLLOW, 'R', 1, 0, 0)                                     \
    ROW(wide, "warps:16", INDEX_WIDE, INDEX_FOLLOW, 'R', 1, 0, 0)                                                     \
    ROW(row_major, "scan:per_row,row_major", INDEX_ROW_MAJOR, INDEX_FOLLOW, 'R', 1, 0, 0)                             \
    ROW(key_major, "scan:per_row,key_major", INDEX_KEY_MAJOR, INDEX_FOLLOW, 'R', 1, 0, 0)                             \
    ROW(candidates, "scan:candidates", INDEX_LISTED, INDEX_FOLLOW, 'R', 1, 0, 0)                                      \
    ROW(read_only, "read_only", INDEX_READ, INDEX_FOLLOW, 'C', 0, 0, 0)                                               \
    ROW(chained, "tensor_only:chained", INDEX_CHAINED, INDEX_FOLLOW, 'C', 0, 0, 0)                                    \
    ROW(interleaved, "tensor_only:interleaved", INDEX_INTERLEAVED, INDEX_FOLLOW, 'C', 0, 0, 0)                        \
    ROW(one_block, "select:one_block", INDEX_ONE_BLOCK, INDEX_FOLLOW, 'R', 0, 0, 0)                                   \
    ROW(spans, "select:spans", INDEX_SPANS, INDEX_FOLLOW, 'R', 0, 0, 0)                                               \
    ROW(scan_select, "step:scan_select", INDEX_SCAN_SELECT, INDEX_FOLLOW, 'R', 0, 0, 0)                               \
    ROW(fp16_scan, "scan:fp16", INDEX_SCAN, INDEX_OWN_HALF, 'R', 0, 0, 0)                                             \
    ROW(e2m1, "scan:e2m1_scaled", INDEX_SCAN, INDEX_OWN_E2M1, 'R', 1, INDEX_SM_120A, 1)                               \
    ROW(fp16_queries, "scan:fp16_queries", INDEX_SCAN, INDEX_OWN_PLACED, 'R', 0, INDEX_SM_120A, 0)                    \
    ROW(head_order, "break:head_order", INDEX_HEAD_ORDER, INDEX_FOLLOW, 'C', 0, 0, 0)
#define INDEX_INSTANCE_ID(id, ...) INDEX_INSTANCE_##id,
enum { INDEX_INSTANCES(INDEX_INSTANCE_ID) INDEX_INSTANCE_COUNT };
#define INDEX_NAN ((double)NAN)                  /* a figure that only measures */
/* The acceptance of ROUND_1_attention.md 11a and the 8-bit bars, every bar [guess]: a figure at its rows and entries,
 * in the case states of its mask, of the judged instance (the faster of judged and other); its ratio to the
 * baseline, or the inverse where the bar is "at least" (inverted); scaled_by: the instance whose ratio the bar is
 * multiplied by, so that it lies between two instances (COUNT: none). */
enum { INDEX_ANY = (1 << INDEX_STATES) - 1, INDEX_EIGHT_BIT = INDEX_ANY - (1 << INDEX_STATE_FP16),
       INDEX_FRESH = 1 << INDEX_STATE_FRESH };
static const struct index_figure {
    const char *name;
    uint32_t rows, entries, judged, other, scaled_by, states;
    double bar;
    int inverted;
} index_figures[] = {
#define INDEX_FIGURE(name, rows, entries, judged, other, scaled_by, states, bar, inverted)                           \
    {name, rows, entries, INDEX_INSTANCE_##judged, INDEX_INSTANCE_##other, INDEX_INSTANCE_##scaled_by, states, bar,   \
     inverted},
    INDEX_FIGURE("scan_over_read_only", 1, 1048576, read_only, read_only, COUNT, INDEX_ANY,
                 PROBE_SM == INDEX_SM_120A ? 1.10 : 1.25, 1)
    INDEX_FIGURE("select_over_scan", 1, 1048576, one_block, spans, COUNT, INDEX_ANY, 0.25, 0)
    INDEX_FIGURE("spans_over_one_block", 1, 1048576, spans, spans, one_block, INDEX_ANY, 0.50, 0)
    INDEX_FIGURE("candidates_over_scan", 1, 1048576, candidates, candidates, COUNT, INDEX_ANY, 0.08, 0)
    INDEX_FIGURE("eight_bit_over_fp16", 1, 1048576, fp16_scan, fp16_scan, COUNT, INDEX_EIGHT_BIT, 0.60, 1)
    INDEX_FIGURE("fp16_queries_over_fp16", 1, 1048576, fp16_queries, fp16_queries, fp16_scan, INDEX_FRESH, 0.60, 0)
    INDEX_FIGURE("scan_over_per_row_row_major", 8, 1048576, row_major, row_major, COUNT, INDEX_ANY, 0.56, 1)
    INDEX_FIGURE("key_major_over_scan", 8, 1048576, key_major, key_major, COUNT, INDEX_ANY, INDEX_NAN, 0)
    INDEX_FIGURE("warps_16_over_scan", 8, 1048576, wide, wide, COUNT, INDEX_ANY, INDEX_NAN, 0)
    INDEX_FIGURE("scan_over_tensor_only", 256, 524288, interleaved, chained, COUNT, INDEX_ANY, 1.30, 1)
    INDEX_FIGURE("select_over_scan", 256, 524288, one_block, spans, COUNT, INDEX_ANY, 0.20, 0)
};

#endif /* INDEX_SCAN_PROBE_KERNELS_CUH */
