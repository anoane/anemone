/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kquant_forms_reference.h: what the kernels of kquant_exact_kernels.cuh and kquant_integer_kernels.cuh are held to,
 * on the host, from the definitions alone: it knows nothing of a record, of a lane or of an instruction. A block of
 * K-quant weights by its meaning (Q2_K: weight = d * scale * code - dmin * minimum; Q3_K: weight = d * (scale - 32)
 * * code); the two quantizers of activations (the reference model's E4M3 per 32, the format's 8-bit integers per
 * 256); contract C1 in double precision; the integer contract step by step, integers exact and floats in its
 * fixed order. What it does not hold: the bits of contract C1's fp32 steps (a card's bits are compared with a
 * card's), and the file's byte layout of a block (the probe's records are synthetic: the converter of round 2 is
 * where the file is read). Plain C, for the host half of kquant_forms_probe.cu. Measurement tool: not part of the
 * engine.
 */
#ifndef KQUANT_FORMS_REFERENCE_H
#define KQUANT_FORMS_REFERENCE_H
#include <math.h>
#include <stdint.h>

enum {
    KQ_REFERENCE_BLOCK_WEIGHTS = 256, KQ_REFERENCE_GROUP_WEIGHTS = 16, KQ_REFERENCE_GROUPS = 16,
    /* fp16: exponent field 0 is mantissa * 2^-24, otherwise (1024 + mantissa) * 2^(field - 25) */
    KQ_REFERENCE_FP16_EXPONENT_SHIFT = 10, KQ_REFERENCE_FP16_EXPONENT_MASK = 31, KQ_REFERENCE_FP16_MANTISSA_MASK = 1023,
    KQ_REFERENCE_FP16_IMPLICIT_ONE = 1024, KQ_REFERENCE_FP16_SIGN_SHIFT = 15, KQ_REFERENCE_FP16_SUBNORMAL_EXPONENT = 24,
    KQ_REFERENCE_FP16_NORMAL_EXPONENT = 25,
    /* E4M3: sign, four exponent bits, three mantissa bits; E8M0: 2^(code - 127); bf16: eight significant bits */
    KQ_REFERENCE_E4M3_SIGN = 0x80, KQ_REFERENCE_E4M3_NAN = 0x7F, KQ_REFERENCE_E4M3_EXPONENT_SHIFT = 3,
    KQ_REFERENCE_E4M3_EXPONENT_MASK = 15, KQ_REFERENCE_E4M3_MANTISSA_MASK = 7, KQ_REFERENCE_E4M3_IMPLICIT_ONE = 8,
    KQ_REFERENCE_E8M0_BIAS = 127, KQ_REFERENCE_BF16_BITS = 8, KQ_REFERENCE_ACTIVATION_BLOCK = 32,
    KQ_REFERENCE_INT8_MOST = 127
};
#define KQ_REFERENCE_E4M3_DIVISOR 1024.0         /* 2^(bias + mantissa bits) */
#define KQ_REFERENCE_E4M3_MAXIMUM 448.0f
#define KQ_REFERENCE_AMAX_FLOOR 1.0e-4f          /* the reference model's lower bound of a block's largest magnitude */

struct kq_reference_block {                      /* one block: 256 inputs of one output, in 16 groups of 16 */
    int code[KQ_REFERENCE_BLOCK_WEIGHTS];        /* 0 .. 3 (Q2_K), -4 .. 3 (Q3_K) */
    int scale[KQ_REFERENCE_GROUPS], minimum[KQ_REFERENCE_GROUPS];       /* Q3_K: the scale less 32, and no minimum */
    double d, dmin;                              /* Q3_K: dmin is 0 */
};

/* An fp16 from its bits, by its definition (an exponent field of 31 is not a scale). */
static inline double kq_reference_half(uint32_t bits) {
    const uint32_t exponent = (bits >> KQ_REFERENCE_FP16_EXPONENT_SHIFT) & KQ_REFERENCE_FP16_EXPONENT_MASK;
    const uint32_t mantissa = bits & KQ_REFERENCE_FP16_MANTISSA_MASK;
    const double magnitude = exponent == 0 ? ldexp((double)mantissa, -KQ_REFERENCE_FP16_SUBNORMAL_EXPONENT)
                             : ldexp((double)(mantissa + KQ_REFERENCE_FP16_IMPLICIT_ONE),
                                     (int)exponent - KQ_REFERENCE_FP16_NORMAL_EXPONENT);
    return ((bits >> KQ_REFERENCE_FP16_SIGN_SHIFT) & 1u) != 0u ? -magnitude : magnitude;
}
/* E4M3 from its fields: (8 + mantissa) * 2^(exponent - 10), or mantissa * 2^-9 where the exponent is 0. */
static inline double kq_reference_e4m3(uint32_t code) {
    const uint32_t exponent = (code >> KQ_REFERENCE_E4M3_EXPONENT_SHIFT) & KQ_REFERENCE_E4M3_EXPONENT_MASK;
    const uint32_t mantissa = code & KQ_REFERENCE_E4M3_MANTISSA_MASK;
    const uint32_t scaled = exponent == 0 ? mantissa << 1 : (KQ_REFERENCE_E4M3_IMPLICIT_ONE + mantissa) << exponent;
    return ((code & KQ_REFERENCE_E4M3_SIGN) != 0u ? -1.0 : 1.0) * (double)scaled / KQ_REFERENCE_E4M3_DIVISOR;
}
/* A double rounded to bf16's eight significant bits, to nearest with ties to even (rint in the default mode). */
static inline double kq_reference_bf16(double value) {
    int exponent = 0;
    const double mantissa = frexp(value, &exponent);
    return ldexp(rint(ldexp(mantissa, KQ_REFERENCE_BF16_BITS)), exponent - KQ_REFERENCE_BF16_BITS);
}

/* The reference model's act_quant for one vector, from its definition: per block of 32 the scale is
 * 2^ceil(log2(largest magnitude, at least 1e-4, times 1 / 448 in fp32)); a value over the scale, clamped to +-448,
 * becomes the nearest E4M3 value, ties to the even code. codes: [size]; exponents: [size / 32], E8M0 codes. */
static inline void kq_reference_quantize_e4m3(const float *values, uint32_t size, uint8_t *codes, uint8_t *exponents) {
    for (uint32_t first = 0; first < size; first += KQ_REFERENCE_ACTIVATION_BLOCK) {
        float largest = KQ_REFERENCE_AMAX_FLOOR;
        for (uint32_t offset = 0; offset < KQ_REFERENCE_ACTIVATION_BLOCK; offset++) {
            largest = fmaxf(largest, fabsf(values[first + offset]));
        }
        const float ratio = largest * (1.0f / KQ_REFERENCE_E4M3_MAXIMUM);
        const int log2_ceiling = (int)ceil(log2((double)ratio));    /* exact: a float's log2 is not near a whole */
        exponents[first / KQ_REFERENCE_ACTIVATION_BLOCK] = (uint8_t)(log2_ceiling + KQ_REFERENCE_E8M0_BIAS);
        for (uint32_t offset = 0; offset < KQ_REFERENCE_ACTIVATION_BLOCK; offset++) {
            const double scaled = ldexp((double)values[first + offset], -log2_ceiling);
            const double magnitude = fmin(fabs(scaled), (double)KQ_REFERENCE_E4M3_MAXIMUM);
            uint32_t nearest = 0;
            for (uint32_t code = 1; code < KQ_REFERENCE_E4M3_NAN; code++) {
                const double distance = fabs(kq_reference_e4m3(code) - magnitude);
                const double least = fabs(kq_reference_e4m3(nearest) - magnitude);
                nearest = distance < least || (distance == least && (code & 1u) == 0u) ? code : nearest;
            }
            codes[first + offset] = (uint8_t)(nearest | (signbit(scaled) ? (uint32_t)KQ_REFERENCE_E4M3_SIGN : 0u));
        }
    }
}
/* The format's quantizer of activations (its reference implementation's rule), per block of 256 values, in fp32:
 * with m the value of largest magnitude (of equal ones the first) the inverse scale is -127 / m, an integer is the
 * nearest whole number to the inverse scale times the value (ties to even), at most 127, and the scale is 1 / the
 * inverse scale; a block of zeros has integers 0 and scale 0. integers: [size]; scales: [size / 256]. */
static inline void kq_reference_quantize_int8(const float *values, uint32_t size, int8_t *integers, float *scales) {
    for (uint32_t first = 0; first < size; first += KQ_REFERENCE_BLOCK_WEIGHTS) {
        float largest = 0.0f, inverse_scale = 0.0f;
        for (uint32_t offset = 0; offset < KQ_REFERENCE_BLOCK_WEIGHTS; offset++) {
            largest = fabsf(values[first + offset]) > fabsf(largest) ? values[first + offset] : largest;
        }
        if (largest != 0.0f) { inverse_scale = -(float)KQ_REFERENCE_INT8_MOST / largest; }
        scales[first / KQ_REFERENCE_BLOCK_WEIGHTS] = largest != 0.0f ? 1.0f / inverse_scale : 0.0f;
        for (uint32_t offset = 0; offset < KQ_REFERENCE_BLOCK_WEIGHTS; offset++) {
            const float nearest = rintf(inverse_scale * values[first + offset]);
            integers[first + offset] = (int8_t)fminf(nearest, (float)KQ_REFERENCE_INT8_MOST);
        }
    }
}
static inline int32_t kq_reference_group_sum(const int8_t *integers, uint32_t group) {        /* b of a group */
    int32_t sum = 0;
    for (uint32_t weight = 0; weight < KQ_REFERENCE_GROUP_WEIGHTS; weight++) {
        sum += integers[group * KQ_REFERENCE_GROUP_WEIGHTS + weight];
    }
    return sum;
}

/* One output of one matrix in double precision: its row of blocks times a vector. values: the inputs as a product
 * multiplies them (dequantized, raw, or an integer times its scale); group_sums: Q2_K's X per group of 16 as the
 * card holds it (the minimum term is dmin * minimum * X, as the contracts form it; a Q3_K block has no minimum). */
static inline double kq_reference_product(const struct kq_reference_block *blocks, uint32_t input_size,
                                          const double *values, const double *group_sums) {
    double sum = 0.0;
    for (uint32_t first = 0; first < input_size; first += KQ_REFERENCE_BLOCK_WEIGHTS) {
        const struct kq_reference_block *block = &blocks[first / KQ_REFERENCE_BLOCK_WEIGHTS];
        double scaled = 0.0, minimums = 0.0;
        for (uint32_t index = 0; index < KQ_REFERENCE_BLOCK_WEIGHTS; index++) {
            const int scale = block->scale[index / KQ_REFERENCE_GROUP_WEIGHTS];
            scaled += (double)(scale * block->code[index]) * values[first + index];
        }
        for (uint32_t group = 0; group < KQ_REFERENCE_GROUPS; group++) {
            minimums += (double)block->minimum[group] * group_sums[first / KQ_REFERENCE_GROUP_WEIGHTS + group];
        }
        sum += block->d * scaled - block->dmin * minimums;
    }
    return sum;
}
/* One output under the integer contract, every step in its order (the header of kquant_integer_kernels.cuh has
 * them): per block I, the sum of scale * code * integer, and J, the sum over its groups of minimum * b, as
 * integers; T = fma((float)I, d * x, T), then T = fma((float)J, -(dmin * x), T) where the block has a minimum term
 * (with_minimums: Q2_K); the slices' T added in slice order from +0. integers: [input size]; scales: x per block. */
static inline float kq_reference_integer_output(const struct kq_reference_block *blocks, uint32_t input_size,
                                                uint32_t slices, const int8_t *integers, const float *scales,
                                                int with_minimums) {
    const uint32_t slice_blocks = input_size / KQ_REFERENCE_BLOCK_WEIGHTS / slices;
    float output = 0.0f;
    for (uint32_t slice = 0; slice < slices; slice++) {
        float total = 0.0f;
        for (uint32_t index = slice * slice_blocks; index < (slice + 1u) * slice_blocks; index++) {
            const struct kq_reference_block *block = &blocks[index];
            const int8_t *block_integers = integers + (size_t)index * KQ_REFERENCE_BLOCK_WEIGHTS;
            int32_t scaled = 0, minimums = 0;
            for (uint32_t weight = 0; weight < KQ_REFERENCE_BLOCK_WEIGHTS; weight++) {
                const int scale = block->scale[weight / KQ_REFERENCE_GROUP_WEIGHTS];
                scaled += scale * block->code[weight] * (int32_t)block_integers[weight];
            }
            for (uint32_t group = 0; group < KQ_REFERENCE_GROUPS; group++) {
                minimums += block->minimum[group] * kq_reference_group_sum(block_integers, group);
            }
            total = fmaf((float)scaled, (float)block->d * scales[index], total);
            if (with_minimums) { total = fmaf((float)minimums, -((float)block->dmin * scales[index]), total); }
        }
        output += total;
    }
    return output;
}
#endif /* KQUANT_FORMS_REFERENCE_H */
