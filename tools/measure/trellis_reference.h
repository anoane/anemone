/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trellis_reference.h: a tile of the EXL3 trellis and of exlr8 v2 by its definition, for the host half of
 * trellis_expert_probe.cu (speed campaign, round 1, package 7). It knows nothing of a lane, of an instruction or
 * of how a card holds a matrix; of the kernels it holds one contract, a rotation's fp32 order (the last function).
 *
 * A TILE is 16 inputs by 16 outputs: 256 weights of K bits in 8 K words of 32 bits (the file's bytes, little
 * endian). Bit b of its stream is bit 31 - b % 32 of word b / 32. Weight i of the stream has the K bits from bit
 * i K on as its code and the 16 bits that END with its code as its STATE; the first states take their leading
 * bits from the end of the same tile. Weight 8 t + m is input 2 (t % 4) + m % 2 + 8 (m / 2 % 2) and output
 * t / 4 + 8 (m / 4) of the tile.
 * A CODEBOOK gives a state s its value, by arithmetic modulo 2^32 and one fp16 operation:
 *   mul1   b = the sum of the four bytes of s * 0x83DCD12D; the fp16 fused multiply-add of the fp16 with the bits
 *          0x6400 + b (1024 + b), of 0x1EEE and of 0xC931: (1774 b - 905216) / 2^18, rounded once
 *   mcg    x = (s * 0xCBAC1FED & 0x8FFF8FFF) ^ 0x3B603B60; the fp16 sum of the fp16 in x's lower and upper half
 *   exlr8  the mcg value rounded to E4M3: to nearest, of two equally near the even code
 * A MATRIX is diag(suh) Hb T Hb diag(svh): T the tiles' values as [input][output], Hb the Sylvester Hadamard
 * matrix of order 128 in blocks, over the root of 128; a row x gives y = (((x .* suh) Hb) T) Hb .* svh.
 * What it does not hold: the order of tiles in a file (the probe's tiles are synthetic: every bit pattern is a
 * tile), the half-bit widths, the first codebook (3INST), a card's bits of an fp32 sum. Plain C, C++ too.
 * Measurement tool: not part of the engine.
 */
#ifndef TRELLIS_REFERENCE_H
#define TRELLIS_REFERENCE_H
#include <math.h>
#include <stddef.h>
#include <stdint.h>

enum {
    TR_REFERENCE_MUL1 = 0, TR_REFERENCE_MCG, TR_REFERENCE_EXLR8,        /* the codebooks */
    TR_REFERENCE_TILE_WEIGHTS = 256, TR_REFERENCE_STATES = 65536, TR_REFERENCE_STATE_BITS = 16,
    TR_REFERENCE_WORD_BITS = 32, TR_REFERENCE_WORD_BYTES = 4, TR_REFERENCE_BYTE_BITS = 8, TR_REFERENCE_BYTE_MASK = 0xFF,
    TR_REFERENCE_HALF_MASK = 0xFFFF, TR_REFERENCE_HADAMARD = 128,
    /* placement: a lane of the stream is eight weights: two inputs, those eight on, and the same at eight outputs on */
    TR_REFERENCE_LANE_WEIGHTS = 8, TR_REFERENCE_LANES_PER_OUTPUT = 4, TR_REFERENCE_LANE_INPUTS = 2,
    TR_REFERENCE_FAR_HALF = 8, TR_REFERENCE_OUTPUT_WEIGHTS = 4,
    /* fp16: field 0 is mantissa * 2^-24, otherwise (1024 + mantissa) * 2^(field - 25); the least normal is 2^-14 */
    TR_REFERENCE_FP16_EXPONENT_SHIFT = 10, TR_REFERENCE_FP16_EXPONENT_MASK = 31, TR_REFERENCE_FP16_MANTISSA_MASK = 1023,
    TR_REFERENCE_FP16_IMPLICIT_ONE = 1024, TR_REFERENCE_FP16_SIGN = 0x8000, TR_REFERENCE_FP16_SUBNORMAL_EXPONENT = 24,
    TR_REFERENCE_FP16_NORMAL_EXPONENT = 25, TR_REFERENCE_FP16_LEAST_NORMAL_EXPONENT = -14,
    /* E4M3: sign, four exponent bits, three mantissa bits; 0x7F is no number */
    TR_REFERENCE_E4M3_SIGN = 0x80, TR_REFERENCE_E4M3_NAN = 0x7F, TR_REFERENCE_E4M3_EXPONENT_SHIFT = 3,
    TR_REFERENCE_E4M3_EXPONENT_MASK = 15, TR_REFERENCE_E4M3_MANTISSA_MASK = 7, TR_REFERENCE_E4M3_IMPLICIT_ONE = 8,
    TR_REFERENCE_MUL1_FIRST_BITS = 0x6400, TR_REFERENCE_MUL1_STEP_BITS = 0x1EEE, TR_REFERENCE_MUL1_OFFSET_BITS = 0xC931
};
#define TR_REFERENCE_MUL1_MULTIPLIER 0x83DCD12Du
#define TR_REFERENCE_MCG_MULTIPLIER 0xCBAC1FEDu
#define TR_REFERENCE_MCG_KEPT 0x8FFF8FFFu
#define TR_REFERENCE_MCG_FLIPPED 0x3B603B60u
#define TR_REFERENCE_E4M3_DIVISOR 1024.0             /* 2^(bias + mantissa bits) */
#define TR_REFERENCE_HADAMARD_SCALE 0.08838834764831845f        /* float32(1 / sqrt(128)): a kernel's multiplier */

/* An fp16 from its bits, by its definition (an exponent field of 31 is no value of a codebook or a side vector). */
static inline double tr_reference_half(uint32_t bits) {
    const uint32_t exponent = (bits >> TR_REFERENCE_FP16_EXPONENT_SHIFT) & TR_REFERENCE_FP16_EXPONENT_MASK;
    const uint32_t mantissa = bits & TR_REFERENCE_FP16_MANTISSA_MASK;
    const double magnitude = exponent == 0 ? ldexp((double)mantissa, -TR_REFERENCE_FP16_SUBNORMAL_EXPONENT)
                             : ldexp((double)(mantissa + TR_REFERENCE_FP16_IMPLICIT_ONE),
                                     (int)exponent - TR_REFERENCE_FP16_NORMAL_EXPONENT);
    return (bits & TR_REFERENCE_FP16_SIGN) != 0u ? -magnitude : magnitude;
}
/* The bits of the fp16 nearest to a value below 65520 in magnitude, a tie to the even one: the value in units of
 * the last place of its binade (2^-24 below 2^-14), which rint rounds; 2048 units are the next exponent field. */
static inline uint32_t tr_reference_half_bits(double value) {
    const double magnitude = fabs(value);
    const uint32_t sign = signbit(value) ? (uint32_t)TR_REFERENCE_FP16_SIGN : 0u;
    int exponent = 0;
    if (magnitude == 0.0) { return sign; }
    (void)frexp(magnitude, &exponent);               /* magnitude is in [2^(exponent - 1), 2^exponent) */
    const int binade = exponent - 1 < TR_REFERENCE_FP16_LEAST_NORMAL_EXPONENT ? TR_REFERENCE_FP16_LEAST_NORMAL_EXPONENT
                                                                               : exponent - 1;
    const int unit = binade - TR_REFERENCE_FP16_EXPONENT_SHIFT;
    const uint32_t units = (uint32_t)rint(ldexp(magnitude, -unit));
    const uint32_t field_less_one = (uint32_t)(unit + TR_REFERENCE_FP16_SUBNORMAL_EXPONENT);
    return sign | ((field_less_one << TR_REFERENCE_FP16_EXPONENT_SHIFT) + units);
}
/* E4M3 from its fields: (8 + mantissa) * 2^(exponent - 10), or mantissa * 2^-9 where the exponent is 0. */
static inline double tr_reference_e4m3(uint32_t code) {
    const uint32_t exponent = (code >> TR_REFERENCE_E4M3_EXPONENT_SHIFT) & TR_REFERENCE_E4M3_EXPONENT_MASK;
    const uint32_t mantissa = code & TR_REFERENCE_E4M3_MANTISSA_MASK;
    const uint32_t scaled = exponent == 0 ? mantissa << 1 : (TR_REFERENCE_E4M3_IMPLICIT_ONE + mantissa) << exponent;
    return ((code & TR_REFERENCE_E4M3_SIGN) != 0u ? -1.0 : 1.0) * (double)scaled / TR_REFERENCE_E4M3_DIVISOR;
}
/* The E4M3 code nearest to a value, of two equally near the even one; the sign is the value's, of a zero too; a
 * magnitude above the largest code's 448 becomes that code. */
static inline uint32_t tr_reference_e4m3_code(double value) {
    const double magnitude = fabs(value);
    uint32_t nearest = 0;
    for (uint32_t code = 1; code < TR_REFERENCE_E4M3_NAN; code++) {
        const double distance = fabs(tr_reference_e4m3(code) - magnitude);
        const double least = fabs(tr_reference_e4m3(nearest) - magnitude);
        nearest = distance < least || (distance == least && (code & 1u) == 0u) ? code : nearest;
    }
    return nearest | (signbit(value) ? (uint32_t)TR_REFERENCE_E4M3_SIGN : 0u);
}
/* mul1's b of a state: the sum of the four bytes of its product. */
static inline uint32_t tr_reference_byte_sum(uint32_t state) {
    const uint32_t product = state * TR_REFERENCE_MUL1_MULTIPLIER;
    uint32_t sum = 0;
    for (uint32_t byte = 0; byte < TR_REFERENCE_WORD_BYTES; byte++) {
        sum += (product >> (byte * TR_REFERENCE_BYTE_BITS)) & TR_REFERENCE_BYTE_MASK;
    }
    return sum;
}
/* The value a codebook gives a state. The products and the sum are exact in double precision: the one rounding of
 * either rule is the fp16's. code, where not NULL: the E4M3 code exlr8 rounds a value to (of mcg's and exlr8's). */
static inline double tr_reference_weight(int codebook, uint32_t state, uint32_t *code) {
    const uint32_t mixed = (state * TR_REFERENCE_MCG_MULTIPLIER & TR_REFERENCE_MCG_KEPT) ^ TR_REFERENCE_MCG_FLIPPED;
    double value = tr_reference_half(mixed & TR_REFERENCE_HALF_MASK)
                   + tr_reference_half(mixed >> TR_REFERENCE_STATE_BITS);
    if (codebook == TR_REFERENCE_MUL1) {
        value = tr_reference_half(TR_REFERENCE_MUL1_FIRST_BITS + tr_reference_byte_sum(state))
                * tr_reference_half(TR_REFERENCE_MUL1_STEP_BITS) + tr_reference_half(TR_REFERENCE_MUL1_OFFSET_BITS);
    }
    value = tr_reference_half(tr_reference_half_bits(value));
    if (code != NULL) { *code = tr_reference_e4m3_code(value); }
    return codebook == TR_REFERENCE_EXLR8 ? tr_reference_e4m3(tr_reference_e4m3_code(value)) : value;
}

/* The 256 states of one tile in the order of its stream. bits: K; words: the tile's 8 K words. */
static inline void tr_reference_tile_states(uint32_t bits, const uint32_t *words, uint16_t *states) {
    const uint32_t stream_bits = TR_REFERENCE_TILE_WEIGHTS * bits;
    for (uint32_t weight = 0; weight < TR_REFERENCE_TILE_WEIGHTS; weight++) {
        uint32_t state = 0;
        for (uint32_t bit = 0; bit < TR_REFERENCE_STATE_BITS; bit++) {      /* the 16 bits that end with the code */
            const uint32_t at = ((weight + 1u) * bits + stream_bits - TR_REFERENCE_STATE_BITS + bit) % stream_bits;
            const uint32_t word = words[at / TR_REFERENCE_WORD_BITS];
            state = (state << 1) | ((word >> (TR_REFERENCE_WORD_BITS - 1u - at % TR_REFERENCE_WORD_BITS)) & 1u);
        }
        states[weight] = (uint16_t)state;
    }
}
/* Where weight `position` of a tile's stream lies in the tile. */
static inline void tr_reference_place(uint32_t position, uint32_t *input, uint32_t *output) {
    const uint32_t lane = position / TR_REFERENCE_LANE_WEIGHTS, in_lane = position % TR_REFERENCE_LANE_WEIGHTS;
    *input = TR_REFERENCE_LANE_INPUTS * (lane % TR_REFERENCE_LANES_PER_OUTPUT) + in_lane % TR_REFERENCE_LANE_INPUTS
             + TR_REFERENCE_FAR_HALF * (in_lane / TR_REFERENCE_LANE_INPUTS % TR_REFERENCE_LANE_INPUTS);
    *output = lane / TR_REFERENCE_LANES_PER_OUTPUT + TR_REFERENCE_FAR_HALF * (in_lane / TR_REFERENCE_OUTPUT_WEIGHTS);
}

/* One block of 128 values through Hb by the definition, in double precision: element (from, to) of the Sylvester
 * matrix is -1 where from & to has an odd number of bits. */
static inline void tr_reference_hadamard(const double *values, double *rotated) {
    for (uint32_t to = 0; to < TR_REFERENCE_HADAMARD; to++) {
        double sum = 0.0;
        for (uint32_t from = 0; from < TR_REFERENCE_HADAMARD; from++) {
            uint32_t odd = 0;
            for (uint32_t common = from & to; common != 0; common &= common - 1u) { odd ^= 1u; }
            sum += odd != 0u ? -values[from] : values[from];
        }
        rotated[to] = sum / sqrt((double)TR_REFERENCE_HADAMARD);
    }
}
/* The same block in fp32 in the kernels' order, which is the contract of a rotated row's bits: seven stages, the
 * distance doubling from 1, a pair (lower, upper) becoming (lower + upper, lower - upper); then each value times
 * float32(1 / sqrt(128)). In place. The host is built without contraction: every operation here is one rounding. */
static inline void tr_reference_rotate(float *values) {
    for (uint32_t distance = 1; distance < TR_REFERENCE_HADAMARD; distance += distance) {
        for (uint32_t index = 0; index < TR_REFERENCE_HADAMARD; index++) {
            if ((index & distance) == 0u) {
                const float lower = values[index], upper = values[index + distance];
                values[index] = lower + upper;
                values[index + distance] = lower - upper;
            }
        }
    }
    for (uint32_t index = 0; index < TR_REFERENCE_HADAMARD; index++) { values[index] *= TR_REFERENCE_HADAMARD_SCALE; }
}
#endif /* TRELLIS_REFERENCE_H */
