/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_reference.h: the host's side of package 11a's checks (research/speed/ROUND_1_attention.md,
 * "index_scan"), from the definitions alone and in double precision where a definition allows it: the formats'
 * values, the reference's two roundings, a score, the TRAINED contract as whole sums emulated in float32, and the
 * selection by a counting sort. It takes natural-order values ([head][dim], [dim]) and shares nothing with the code
 * that packs or rounds for the card (index_scan_probe_kernels.cuh): index_scan_probe.cu compares the two.
 *   - fp16: sign, five exponent bits of bias 15, ten mantissa bits; E4M3: sign, four of bias 7, three, no infinity,
 *     0x7F no number; E2M1: sign, two of bias 1, one: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
 *   - TRAINED (kernel.py:155-172 with E8M0 scales; model.py:546, 552) and FRESH (kernel.py:74-90 with power-of-two
 *     scales) per block of 32: the scale code is the float32 exponent field of max(largest, floor) x fp32(1 / largest
 *     code), one more unless that product is a power of two; a value is divided by 2^(code - 127), clamped, and goes
 *     to the nearest value of the format, a tie to the even code (engine/tools/oracle/reference_quantization.py has
 *     the same definitions, which the oracle checks the reference against).
 *   - A score: the sum over 32 heads of weight x max(0, dot), all in double.
 *   - The selection: the top min(512, extent) by order key descending (the fp16's bits, the sign flipped, or every
 *     bit where it is set), position ascending; positions ascending, then -1.
 *   - The 8-bit records read back by the layout index_scan_kernels.cuh defines, with constants of their own here: a
 *     tile of 8 keys is 1,056 B, key g's codes at 128 g, its run t at 32 t, a word a step j, byte i dim 16 j + 2 t +
 *     {0, 8, 1, 9}[i], the scale words at 1,024; a placed query row is 4,224 B, [head tile][block][lane][register]
 *     of 16 B, byte i of register r of lane 4 g + t head 16 h + g + 8 (r % 2), dim 32 b + 16 (r / 2) + 2 t +
 *     {0, 8, 1, 9}[i], the scale codes [head][block] at 4,096.
 * Host code only (C, as C++ in a .cu). Measurement tool: not part of the engine.
 */
#ifndef INDEX_SCAN_REFERENCE_H
#define INDEX_SCAN_REFERENCE_H
#include <math.h>
#include <stdint.h>
#include <string.h>

enum {
    REFERENCE_HEADS = 32, REFERENCE_DIMS = 128, REFERENCE_BLOCK = 32, REFERENCE_BLOCKS = 4, REFERENCE_SELECTED = 512,
    REFERENCE_LANE_GROUPS = 8, REFERENCE_HALF_TILE = 8, REFERENCE_HEADS_PER_TILE = 16,
    REFERENCE_HALF_EXPONENT_SHIFT = 10, REFERENCE_HALF_EXPONENT_MASK = 31, REFERENCE_HALF_MANTISSA_MASK = 1023,
    REFERENCE_HALF_BIAS = 15, REFERENCE_HALF_MANTISSA_BITS = 10, REFERENCE_HALF_SIGN = 0x8000,
    REFERENCE_HALF_INFINITY = 0x7C00, REFERENCE_HALF_LEAST_EXPONENT = -14, REFERENCE_HALF_MOST_EXPONENT = 15,
    REFERENCE_E4M3_SIGN = 0x80, REFERENCE_E4M3_MANTISSA_BITS = 3, REFERENCE_E4M3_EXPONENT_MASK = 15,
    REFERENCE_E4M3_MANTISSA_MASK = 7, REFERENCE_E4M3_BIAS = 7, REFERENCE_E4M3_IMPLICIT = 8,
    REFERENCE_E4M3_MAGNITUDES = 127,             /* codes 0 .. 0x7E; 0x7F is no number */
    REFERENCE_E2M1_CODES = 8, REFERENCE_E8M0_BIAS = 127, REFERENCE_FLOAT32_EXPONENT_SHIFT = 23,
    REFERENCE_FLOAT32_EXPONENT_MASK = 255, REFERENCE_KEYS = 65536, REFERENCE_KEY_BITS = 16,
    /* a TRAINED block's whole sum is of codes twice the values: the power of two of two scale codes, a quarter */
    REFERENCE_WHOLE_FIELD_BIAS = 129,
    REFERENCE_KEEP = 0, REFERENCE_TRAINED = 1, REFERENCE_FRESH = 2,    /* what a state does to its values */
    REFERENCE_TILE_KEYS = 8, REFERENCE_TILE_BYTES = 1056, REFERENCE_KEY_BYTES = 128, REFERENCE_RUN_BYTES = 32,
    REFERENCE_SCALES_AT = 1024, REFERENCE_STEPS = 8, REFERENCE_STEP_DIMS = 16, REFERENCE_PLACES = 4,
    REFERENCE_WORD_BYTES = 4, REFERENCE_LOAD_BYTES = 16, REFERENCE_LANES = 32, REFERENCE_QUERY_SCALES_AT = 4096,
    REFERENCE_PAIR = 2, REFERENCE_UPPER_DIMS = 16
};
static const uint32_t reference_byte_dims[REFERENCE_WORD_BYTES] = {0, 8, 1, 9};
#define REFERENCE_FLOAT32_MANTISSA_MASK 0x007FFFFFu
#define REFERENCE_KEY_FLIP_POSITIVE 0x8000u
#define REFERENCE_KEY_FLIP_NEGATIVE 0xFFFFu
#define REFERENCE_E2M1_FLOOR (6.0 * 0x1p-126)     /* kernel.py:165 */
#define REFERENCE_E2M1_LARGEST 6.0
#define REFERENCE_E4M3_FLOOR 1.0e-4               /* kernel.py:76 */
#define REFERENCE_E4M3_LARGEST 448.0
#define REFERENCE_TWICE 2.0                       /* a midpoint is half a sum; TRAINED stores twice a value */
static const double reference_e2m1_values[REFERENCE_E2M1_CODES] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

static inline double reference_half(uint16_t bits)
{
    const int exponent = (bits >> REFERENCE_HALF_EXPONENT_SHIFT) & REFERENCE_HALF_EXPONENT_MASK;
    const int mantissa = bits & REFERENCE_HALF_MANTISSA_MASK;
    const double magnitude = exponent == REFERENCE_HALF_EXPONENT_MASK ? (mantissa != 0 ? NAN : INFINITY)
        : exponent == 0 ? ldexp(mantissa, REFERENCE_HALF_LEAST_EXPONENT - REFERENCE_HALF_MANTISSA_BITS)
                        : ldexp(mantissa + (1 << REFERENCE_HALF_MANTISSA_BITS),
                                exponent - REFERENCE_HALF_BIAS - REFERENCE_HALF_MANTISSA_BITS);
    return (bits & REFERENCE_HALF_SIGN) != 0 ? -magnitude : magnitude;
}
/* A value rounded once to fp16: to nearest, a tie to even, an infinity past the largest. */
static inline uint16_t reference_half_bits(double value)
{
    const uint16_t sign = signbit(value) ? (uint16_t)REFERENCE_HALF_SIGN : 0u;
    int exponent = 0;
    if (value == 0.0 || isnan(value)) {
        return isnan(value) ? (uint16_t)(REFERENCE_HALF_INFINITY | 1u) : sign;
    }
    (void)frexp(fabs(value), &exponent);                    /* 2^(exponent - 1) <= |value| < 2^exponent */
    const int lowest = exponent - 1 < REFERENCE_HALF_LEAST_EXPONENT ? REFERENCE_HALF_LEAST_EXPONENT : exponent - 1;
    const int unit_exponent = lowest - REFERENCE_HALF_MANTISSA_BITS;
    const double units = nearbyint(ldexp(fabs(value), -unit_exponent));      /* the default mode: to nearest, even */
    const double rounded = ldexp(units, unit_exponent);
    if (rounded >= ldexp(1.0, REFERENCE_HALF_MOST_EXPONENT + 1)) {
        return (uint16_t)(sign | REFERENCE_HALF_INFINITY);
    }
    int rounded_exponent = 0;
    (void)frexp(rounded, &rounded_exponent);
    const int field = rounded < ldexp(1.0, REFERENCE_HALF_LEAST_EXPONENT) ? 0
                                                                          : rounded_exponent - 1 + REFERENCE_HALF_BIAS;
    const int mantissa = field == 0 ? (int)ldexp(rounded, REFERENCE_HALF_MANTISSA_BITS - REFERENCE_HALF_LEAST_EXPONENT)
                                    : (int)ldexp(rounded, REFERENCE_HALF_MANTISSA_BITS - (rounded_exponent - 1)) -
                                          (1 << REFERENCE_HALF_MANTISSA_BITS);
    return (uint16_t)(sign | (field << REFERENCE_HALF_EXPONENT_SHIFT) | mantissa);
}
static inline double reference_e4m3(uint32_t code)
{
    const int exponent = (int)((code >> REFERENCE_E4M3_MANTISSA_BITS) & REFERENCE_E4M3_EXPONENT_MASK);
    const int units = (int)(code & REFERENCE_E4M3_MANTISSA_MASK) + (exponent == 0 ? 0 : REFERENCE_E4M3_IMPLICIT);
    const double magnitude = ldexp(units, (exponent == 0 ? 1 : exponent) - REFERENCE_E4M3_BIAS -
                                              REFERENCE_E4M3_MANTISSA_BITS);
    return (code & REFERENCE_E4M3_SIGN) != 0 ? -magnitude : magnitude;
}
/* The scale code of a block: float32 as the reference's kernels compute it. */
static inline uint32_t reference_scale_code(const double values[REFERENCE_BLOCK], double floor, double largest_code)
{
    double largest = 0.0;
    uint32_t bits = 0;
    for (int index = 0; index < REFERENCE_BLOCK; index++) {
        largest = fmax(largest, fabs(values[index]));
    }
    const float product = (float)fmax(largest, (double)(float)floor) * (float)(1.0 / largest_code);
    memcpy(&bits, &product, sizeof(bits));
    return ((bits >> REFERENCE_FLOAT32_EXPONENT_SHIFT) & REFERENCE_FLOAT32_EXPONENT_MASK) +
           ((bits & REFERENCE_FLOAT32_MANTISSA_MASK) != 0 ? 1u : 0u);
}
/* TRAINED: twice each value's E2M1 (a signed whole number, -0 as 0) and the scale code. */
static inline void reference_trained(const double values[REFERENCE_BLOCK], int8_t doubled[REFERENCE_BLOCK],
                                     uint32_t *scale_code)
{
    *scale_code = reference_scale_code(values, REFERENCE_E2M1_FLOOR, REFERENCE_E2M1_LARGEST);
    for (int index = 0; index < REFERENCE_BLOCK; index++) {
        const double scaled = fmin(fmax(ldexp(values[index], REFERENCE_E8M0_BIAS - (int)*scale_code),
                                        -REFERENCE_E2M1_LARGEST), REFERENCE_E2M1_LARGEST);
        int below = 0, at_or_below = 0;
        for (int code = 0; code + 1 < REFERENCE_E2M1_CODES; code++) {
            const double midpoint = (reference_e2m1_values[code] + reference_e2m1_values[code + 1]) / REFERENCE_TWICE;
            below += fabs(scaled) > midpoint ? 1 : 0;
            at_or_below += fabs(scaled) >= midpoint ? 1 : 0;
        }
        const int value_twice = (int)(reference_e2m1_values[(below & 1) == 0 ? below : at_or_below] * REFERENCE_TWICE);
        doubled[index] = (int8_t)(scaled < 0.0 ? -value_twice : value_twice);
    }
}
/* FRESH: each value's E4M3 code (the nearest of the 127 numbers, a tie to the even code, the sign kept). */
static inline void reference_fresh(const double values[REFERENCE_BLOCK], uint8_t codes[REFERENCE_BLOCK],
                                   uint32_t *scale_code)
{
    *scale_code = reference_scale_code(values, REFERENCE_E4M3_FLOOR, REFERENCE_E4M3_LARGEST);
    for (int index = 0; index < REFERENCE_BLOCK; index++) {
        static double numbers[REFERENCE_E4M3_MAGNITUDES];      /* ascending with the code: filled once */
        const double scaled = ldexp(values[index], REFERENCE_E8M0_BIAS - (int)*scale_code);
        const double magnitude = fmin(fabs(scaled), REFERENCE_E4M3_LARGEST);
        uint32_t below = 0, above = REFERENCE_E4M3_MAGNITUDES - 1u;
        for (uint32_t code = numbers[1] == 0.0 ? 0u : (uint32_t)REFERENCE_E4M3_MAGNITUDES;
             code < REFERENCE_E4M3_MAGNITUDES; code++) {
            numbers[code] = reference_e4m3(code);
        }
        while (above - below > 1u) {                           /* numbers[below] <= magnitude <= numbers[above] */
            const uint32_t middle = (below + above) >> 1;
            *(numbers[middle] <= magnitude ? &below : &above) = middle;
        }
        const double low = magnitude - numbers[below], high = numbers[above] - magnitude;
        const uint32_t best = low < high || (low == high && (below & 1u) == 0) ? below : above;
        codes[index] = (uint8_t)(best | (signbit(scaled) ? (uint32_t)REFERENCE_E4M3_SIGN : 0u));
    }
}
/* The values of 128 dims as the state holds them: TRAINED and FRESH round per block; FP16 keeps them. */
static inline void reference_state_values(int rounding, const double natural[REFERENCE_DIMS],
                                          double held[REFERENCE_DIMS])
{
    for (int block = 0; block < REFERENCE_BLOCKS; block++) {
        const double *values = natural + block * REFERENCE_BLOCK;
        int8_t doubled[REFERENCE_BLOCK];
        uint8_t codes[REFERENCE_BLOCK];
        uint32_t scale_code = 0;
        if (rounding == REFERENCE_TRAINED) {
            reference_trained(values, doubled, &scale_code);
        } else if (rounding == REFERENCE_FRESH) {
            reference_fresh(values, codes, &scale_code);
        }
        for (int index = 0; index < REFERENCE_BLOCK; index++) {
            const double scale = ldexp(1.0, (int)scale_code - REFERENCE_E8M0_BIAS);
            held[block * REFERENCE_BLOCK + index] =
                rounding == REFERENCE_TRAINED ? doubled[index] / REFERENCE_TWICE * scale
                : rounding == REFERENCE_FRESH ? reference_e4m3(codes[index]) * scale : values[index];
        }
    }
}
/* Codes and scale codes of 128 values by a state's rounding: TRAINED twice the E2M1 as a signed byte, FRESH E4M3. */
static inline void reference_codes(int rounding, const double values[REFERENCE_DIMS], uint8_t codes[REFERENCE_DIMS],
                                   uint8_t scales[REFERENCE_BLOCKS])
{
    for (int block = 0; block < REFERENCE_BLOCKS; block++) {
        uint32_t scale = 0;
        if (rounding == REFERENCE_TRAINED) {
            reference_trained(values + block * REFERENCE_BLOCK, (int8_t *)codes + block * REFERENCE_BLOCK, &scale);
        } else {
            reference_fresh(values + block * REFERENCE_BLOCK, codes + block * REFERENCE_BLOCK, &scale);
        }
        scales[block] = (uint8_t)scale;
    }
}
/* Key `key` of a plane of 8-bit tiles: its codes in dim order and its scale codes. */
static inline void reference_read_key(const unsigned char *plane, uint32_t key, int8_t codes[REFERENCE_DIMS],
                                      uint8_t scales[REFERENCE_BLOCKS])
{
    const unsigned char *tile = plane + (size_t)(key / REFERENCE_TILE_KEYS) * REFERENCE_TILE_BYTES;
    const unsigned char *own = tile + key % REFERENCE_TILE_KEYS * REFERENCE_KEY_BYTES;
    for (uint32_t at = 0; at < REFERENCE_KEY_BYTES; at++) {        /* [run t][step j][byte i] */
        const uint32_t place = at / REFERENCE_RUN_BYTES, step = at % REFERENCE_RUN_BYTES / REFERENCE_WORD_BYTES;
        codes[step * REFERENCE_STEP_DIMS + place * REFERENCE_PAIR + reference_byte_dims[at % REFERENCE_WORD_BYTES]] =
            (int8_t)own[at];
    }
    memcpy(scales, tile + REFERENCE_SCALES_AT + key % REFERENCE_TILE_KEYS * REFERENCE_WORD_BYTES, REFERENCE_BLOCKS);
}
/* A placed row of query codes: [head][dim] and the scale codes [head][block]. */
static inline void reference_read_queries(const unsigned char *row_bytes,
                                          int8_t codes[REFERENCE_HEADS * REFERENCE_DIMS],
                                          uint8_t scales[REFERENCE_HEADS * REFERENCE_BLOCKS])
{
    for (uint32_t at = 0; at < REFERENCE_QUERY_SCALES_AT; at++) {  /* [head tile][block][lane][register][byte] */
        const uint32_t lane = at / REFERENCE_LOAD_BYTES % REFERENCE_LANES;
        const uint32_t operand_register = at % REFERENCE_LOAD_BYTES / REFERENCE_WORD_BYTES;
        const uint32_t block = at / (REFERENCE_LOAD_BYTES * REFERENCE_LANES) % REFERENCE_BLOCKS;
        const uint32_t head_tile = at / (REFERENCE_LOAD_BYTES * REFERENCE_LANES * REFERENCE_BLOCKS);
        const uint32_t head = head_tile * REFERENCE_HEADS_PER_TILE + lane / REFERENCE_PLACES +
                              (operand_register & 1u) * REFERENCE_HALF_TILE;
        const uint32_t dim = block * REFERENCE_BLOCK + operand_register / REFERENCE_PAIR * REFERENCE_UPPER_DIMS +
                             lane % REFERENCE_PLACES * REFERENCE_PAIR + reference_byte_dims[at % REFERENCE_WORD_BYTES];
        codes[head * REFERENCE_DIMS + dim] = (int8_t)row_bytes[at];
    }
    memcpy(scales, row_bytes + REFERENCE_QUERY_SCALES_AT, REFERENCE_HEADS * REFERENCE_BLOCKS);
}
/* The exclusive or of a key's 128 halves (FP16), or of its 128 codes and 4 scale codes: a read of its record. */
static inline uint16_t reference_key_fold(int rounding, const double values[REFERENCE_DIMS])
{
    uint8_t codes[REFERENCE_DIMS], scales[REFERENCE_BLOCKS];
    uint32_t fold = 0;
    if (rounding != REFERENCE_KEEP) {
        reference_codes(rounding, values, codes, scales);
    }
    for (int dim = 0; dim < REFERENCE_DIMS; dim++) {
        fold ^= rounding == REFERENCE_KEEP ? reference_half_bits(values[dim]) : codes[dim];
    }
    for (int block = 0; rounding != REFERENCE_KEEP && block < REFERENCE_BLOCKS; block++) {
        fold ^= scales[block];
    }
    return (uint16_t)fold;
}
static inline double reference_score(const double query[REFERENCE_HEADS * REFERENCE_DIMS],
                                     const double key[REFERENCE_DIMS], const float weights[REFERENCE_HEADS])
{
    double score = 0.0;
    for (int head = 0; head < REFERENCE_HEADS; head++) {
        double dot = 0.0;
        for (int dim = 0; dim < REFERENCE_DIMS; dim++) {
            dot += query[head * REFERENCE_DIMS + dim] * key[dim];
        }
        score += (double)weights[head] * fmax(dot, 0.0);
    }
    return score;
}
/* The TRAINED contract in float32 from whole codes: per head and block the exact whole sum times 2^(the scale codes
 * less 129), that power a float32 by its definition, added by one fused multiply-add from +0 in block order; ReLU (+0
 * at or under 0) times the weight; a lane group's heads g, g + 8, 16 + g, 24 + g as (a + b) + (c + d); the eight as
 * ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7)); one rounding to fp16. *exact: 1 if every head's chain stayed equal to the
 * exact sum of its scaled whole sums. */
static inline uint16_t reference_whole_score(const int8_t query[REFERENCE_HEADS * REFERENCE_DIMS],
                                             const uint8_t query_scales[REFERENCE_HEADS * REFERENCE_BLOCKS],
                                             const int8_t key[REFERENCE_DIMS],
                                             const uint8_t key_scales[REFERENCE_BLOCKS],
                                             const float weights[REFERENCE_HEADS], int *exact)
{
    float terms[REFERENCE_HEADS], groups[REFERENCE_LANE_GROUPS];
    *exact = 1;
    for (int head = 0; head < REFERENCE_HEADS; head++) {
        float chain = 0.0f;
        double exact_sum = 0.0;
        for (int block = 0; block < REFERENCE_BLOCKS; block++) {
            int64_t whole = 0;
            for (int dim = block * REFERENCE_BLOCK; dim < (block + 1) * REFERENCE_BLOCK; dim++) {
                whole += (int64_t)query[head * REFERENCE_DIMS + dim] * key[dim];
            }
            const int field = query_scales[head * REFERENCE_BLOCKS + block] + key_scales[block] -
                              REFERENCE_WHOLE_FIELD_BIAS;
            const float power = ldexpf(1.0f, field - REFERENCE_E8M0_BIAS);      /* 0, subnormal, infinity too */
            chain = fmaf((float)whole, power, chain);
            exact_sum += (double)whole * (double)power;
            *exact &= (double)chain == exact_sum ? 1 : 0;
        }
        terms[head] = (chain > 0.0f ? chain : 0.0f) * weights[head];
    }
    for (int group = 0; group < REFERENCE_LANE_GROUPS; group++) {
        groups[group] = (terms[group] + terms[group + REFERENCE_HALF_TILE]) +
                        (terms[group + REFERENCE_HEADS_PER_TILE] + terms[group + REFERENCE_HEADS_PER_TILE +
                                                                         REFERENCE_HALF_TILE]);
    }
    for (int width = 1; width < REFERENCE_LANE_GROUPS; width += width) {       /* ((0 + 1) + (2 + 3)) + ... */
        for (int group = 0; group < REFERENCE_LANE_GROUPS; group += width + width) {
            groups[group] = groups[group] + groups[group + width];
        }
    }
    return reference_half_bits((double)groups[0]);
}
static inline uint32_t reference_order_key(uint16_t bits)
{
    return bits ^ ((bits & REFERENCE_HALF_SIGN) != 0 ? REFERENCE_KEY_FLIP_NEGATIVE : REFERENCE_KEY_FLIP_POSITIVE);
}
/* The selection of a row of `extent` scores into selected[512]; ties_highest_first takes the cut's ties from the
 * highest position down (the break of the check). Returns 1 if the cut left ties of its value out. */
static inline int reference_select(const uint16_t *scores, uint32_t extent, int ties_highest_first,
                                   int32_t selected[REFERENCE_SELECTED])
{
    static uint32_t counts[REFERENCE_KEYS];
    const uint32_t need = extent < REFERENCE_SELECTED ? extent : (uint32_t)REFERENCE_SELECTED;
    uint32_t threshold = REFERENCE_KEYS, above = 0, taken = 0, written = 0;
    memset(counts, 0, sizeof(counts));
    for (uint32_t position = 0; position < extent; position++) {
        counts[reference_order_key(scores[position])]++;
    }
    while (threshold > 0 && above < need) {            /* the cut: the key where `need` is reached from the top */
        above += counts[--threshold];
    }
    const uint32_t greater = need == 0 ? 0 : above - counts[threshold], ties_needed = need - greater;
    for (uint32_t step = 0; step < extent; step++) {   /* the ties the cut takes, then every selected in order */
        const uint32_t position = ties_highest_first ? extent - 1u - step : step;
        taken += need > 0 && reference_order_key(scores[position]) == threshold && taken < ties_needed ? 1u : 0u;
        if (taken == ties_needed) {
            const uint32_t last_tie = position;
            for (uint32_t at = 0; at < extent; at++) {
                const uint32_t key = reference_order_key(scores[at]);
                const int is_tie_taken = key == threshold &&
                                         (ties_highest_first ? at >= last_tie : at <= last_tie);
                if (need > 0 && (key > threshold || is_tie_taken)) {
                    selected[written++] = (int32_t)at;
                }
            }
            break;
        }
    }
    for (uint32_t at = written; at < REFERENCE_SELECTED; at++) {
        selected[at] = -1;
    }
    return need > 0 && ties_needed < counts[threshold];
}

#endif /* INDEX_SCAN_REFERENCE_H */
