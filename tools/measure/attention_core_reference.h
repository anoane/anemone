/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_reference.h: the host's attention core of package 11b (research/speed/ROUND_1_attention.md,
 * "attention_core"), in double precision, written from the definitions: the reference's sparse attention (kernel.py
 * sparse_attn_kernel: a softmax over the row's valid keys with a sink per head added once to the sum; model.py:651
 * the scale 512^-0.5; :781-787 the de-rotation of the last 64 dims) and its stored state (model.py:707 act_quant: E4M3
 * codes with a power-of-two E8M0 scale per 32; :760 fp4_act_quant: E2M1 codes with an E4M3 scale per 16; fp16 rows).
 * A stored row is decoded here from its format's definition alone: nothing is shared with the code that fills the
 * planes or reads them on the card. Measurement tool: not part of the engine.
 *
 * What it computes for one row and head, fed with the card's own previous stages where a stage is a contract of the
 * kernel's (CONVENTIONS 5.1): the scores q . k x scale (the scale as the reference passes it, an fp32); the softmax
 * with the kernel's running maximum and step (a rescale changes no value of the result, only where P is rounded),
 * P rounded to the kind's P format (fp16 or bfloat16) for the products, the sum of the unrounded p; O over sum +
 * exp(sink - m); then the de-rotation with the card's own fp32 angles. The result is rounded once to bfloat16 and
 * the card's row is held to it: the rms of the difference over the rms of the rounded result.
 */
#ifndef ATTENTION_CORE_REFERENCE_H
#define ATTENTION_CORE_REFERENCE_H
#include <math.h>
#include <stdint.h>
#include <string.h>

enum {
    REFERENCE_HEADS = 64, REFERENCE_DIMS = 512, REFERENCE_KEYS = 640, REFERENCE_GROUP_KEYS = 16,
    REFERENCE_ROPE_FIRST_DIM = 448, REFERENCE_ROPE_PAIRS = 32, REFERENCE_PAIR = 2,
    REFERENCE_FORMAT_FP16 = 0, REFERENCE_FORMAT_POW2 = 1, REFERENCE_FORMAT_E2M1 = 2,
    REFERENCE_PROBABILITY_FP16 = 0, REFERENCE_PROBABILITY_BFLOAT16 = 1,
    REFERENCE_CODES = 512,                       /* the codes of an 8-bit row; its scales follow */
    REFERENCE_POW2_BLOCK = 32, REFERENCE_E2M1_BLOCK = 16, REFERENCE_BYTE_BITS = 8, REFERENCE_HALF_BITS = 16,
    /* binary16: 5 exponent bits of bias 15, 10 mantissa bits; E4M3: 4 of bias 7, 3; E2M1: 2 of bias 1, 1 */
    REFERENCE_HALF_MANTISSA_BITS = 10, REFERENCE_HALF_EXPONENT_MASK = 31, REFERENCE_HALF_BIAS = 15,
    REFERENCE_HALF_SIGN_BIT = 15, REFERENCE_HALF_SIGNIFICANT = 11, REFERENCE_HALF_LEAST_EXPONENT = -24,
    REFERENCE_HALF_LEAST_NORMAL = -14,
    REFERENCE_BFLOAT16_SIGNIFICANT = 8, REFERENCE_BFLOAT16_LEAST_EXPONENT = -133,
    REFERENCE_BFLOAT16_LEAST_NORMAL = -126,
    REFERENCE_E4M3_MANTISSA_BITS = 3, REFERENCE_E4M3_EXPONENT_MASK = 15, REFERENCE_E4M3_BIAS = 7,
    REFERENCE_E4M3_SIGN_BIT = 7, REFERENCE_E4M3_NOT_A_NUMBER = 0x7F, REFERENCE_E4M3_MAGNITUDE = 0x7F,
    REFERENCE_E2M1_EXPONENT_MASK = 3, REFERENCE_E2M1_SIGN_BIT = 3, REFERENCE_E2M1_CODE_MASK = 15,
    REFERENCE_E2M1_MANTISSA_BITS = 1, REFERENCE_E2M1_BIAS = 1,
    REFERENCE_E8M0_BIAS = 127, REFERENCE_E8M0_NOT_A_NUMBER = 255
};
#define REFERENCE_SCORE_FLOOR (-1.0e30)          /* the reference kernel's finite floor of a maximum */

struct reference_row {                           /* one row's inputs, as the card holds them */
    const uint16_t *queries;                     /* fp16 [64][512] */
    const double *keys[REFERENCE_KEYS];          /* a key's 512 decoded values; NULL for an index of -1 */
    const float *sinks;                          /* [64] */
    const float *rotation;                       /* [32][2]: cos, sin */
    double step;                                 /* the contract's step: a rise of the maximum above it rescales */
    int probability_format;
};
struct reference_counts { uint64_t rescales, flushed; };   /* of P values a rounding made zero (not exactly zero) */

static inline double reference_half(uint32_t bits)
{
    const uint32_t exponent = (bits >> REFERENCE_HALF_MANTISSA_BITS) & REFERENCE_HALF_EXPONENT_MASK;
    const uint32_t mantissa = bits & ((1u << REFERENCE_HALF_MANTISSA_BITS) - 1u);
    const double sign = ((bits >> REFERENCE_HALF_SIGN_BIT) & 1u) != 0 ? -1.0 : 1.0;
    if (exponent == REFERENCE_HALF_EXPONENT_MASK) {
        return mantissa != 0 ? (double)NAN : sign * (double)INFINITY;
    }
    return exponent == 0 ? sign * ldexp((double)mantissa, REFERENCE_HALF_LEAST_EXPONENT)
                         : sign * ldexp((double)(mantissa | (1u << REFERENCE_HALF_MANTISSA_BITS)),
                                        (int)exponent - REFERENCE_HALF_BIAS - REFERENCE_HALF_MANTISSA_BITS);
}

static inline double reference_bfloat16(uint32_t bits)
{
    const uint32_t widened = (bits & UINT16_MAX) << REFERENCE_HALF_BITS;
    float value;
    memcpy(&value, &widened, sizeof(value));
    return (double)value;
}

static inline double reference_e4m3(uint32_t code)          /* E4M3FN: no infinity, S.1111.111 is NaN */
{
    const uint32_t exponent = (code >> REFERENCE_E4M3_MANTISSA_BITS) & REFERENCE_E4M3_EXPONENT_MASK;
    const uint32_t mantissa = code & ((1u << REFERENCE_E4M3_MANTISSA_BITS) - 1u);
    const double sign = ((code >> REFERENCE_E4M3_SIGN_BIT) & 1u) != 0 ? -1.0 : 1.0;
    if ((code & REFERENCE_E4M3_MAGNITUDE) == REFERENCE_E4M3_NOT_A_NUMBER) {
        return (double)NAN;
    }
    return exponent == 0 ? sign * ldexp((double)mantissa, 1 - REFERENCE_E4M3_BIAS - REFERENCE_E4M3_MANTISSA_BITS)
                         : sign * ldexp((double)(mantissa | (1u << REFERENCE_E4M3_MANTISSA_BITS)),
                                        (int)exponent - REFERENCE_E4M3_BIAS - REFERENCE_E4M3_MANTISSA_BITS);
}

static inline double reference_e2m1(uint32_t code)          /* in the low nibble; a high nibble is no E2M1 code */
{
    const uint32_t exponent = (code >> REFERENCE_E2M1_MANTISSA_BITS) & REFERENCE_E2M1_EXPONENT_MASK;
    const uint32_t mantissa = code & ((1u << REFERENCE_E2M1_MANTISSA_BITS) - 1u);
    const double sign = ((code >> REFERENCE_E2M1_SIGN_BIT) & 1u) != 0 ? -1.0 : 1.0;
    if ((code & ~(uint32_t)REFERENCE_E2M1_CODE_MASK) != 0) {
        return (double)NAN;
    }
    return exponent == 0 ? sign * ldexp((double)mantissa, 1 - REFERENCE_E2M1_BIAS - REFERENCE_E2M1_MANTISSA_BITS)
                         : sign * ldexp((double)(mantissa | (1u << REFERENCE_E2M1_MANTISSA_BITS)),
                                        (int)exponent - REFERENCE_E2M1_BIAS - REFERENCE_E2M1_MANTISSA_BITS);
}

static inline double reference_e8m0(uint32_t code)
{
    return code == REFERENCE_E8M0_NOT_A_NUMBER ? (double)NAN : ldexp(1.0, (int)code - REFERENCE_E8M0_BIAS);
}

/* A stored row's 512 values: fp16 little-endian; E4M3 codes times the E8M0 scale of their 32; E2M1 codes times the
 * E4M3 scale of their 16. Only the format's own bytes are read: an 8-bit record is 528 or 544 bytes. */
static inline void reference_row_values(int format, const uint8_t *record, double values[REFERENCE_DIMS])
{
    for (uint32_t dim = 0; dim < REFERENCE_DIMS; dim++) {
        values[dim] = format == REFERENCE_FORMAT_FP16
            ? reference_half(record[REFERENCE_PAIR * dim] |
                             ((uint32_t)record[REFERENCE_PAIR * dim + 1u] << REFERENCE_BYTE_BITS))
            : format == REFERENCE_FORMAT_POW2
            ? reference_e4m3(record[dim]) * reference_e8m0(record[REFERENCE_CODES + dim / REFERENCE_POW2_BLOCK])
            : reference_e2m1(record[dim]) * reference_e4m3(record[REFERENCE_CODES + dim / REFERENCE_E2M1_BLOCK]);
    }
}

/* value rounded to `significant` bits, nearest with a tie to even, below 2^least_normal to multiples of
 * 2^least_exponent; 0 stays 0 with its sign. Values here are far from either format's largest number. */
static inline double reference_rounded(double value, int significant, int least_normal, int least_exponent)
{
    int exponent = 0;
    if (value == 0.0 || isnan(value)) {
        return value;
    }
    (void)frexp(value, &exponent);
    const int unit = exponent - 1 >= least_normal ? exponent - significant : least_exponent;
    return ldexp(nearbyint(ldexp(value, -unit)), unit);
}

static inline uint32_t reference_bfloat16_bits(double value)        /* rounded once from double */
{
    const float rounded = (float)reference_rounded(value, REFERENCE_BFLOAT16_SIGNIFICANT,
                                                   REFERENCE_BFLOAT16_LEAST_NORMAL, REFERENCE_BFLOAT16_LEAST_EXPONENT);
    uint32_t bits;
    memcpy(&bits, &rounded, sizeof(bits));
    return bits >> REFERENCE_HALF_BITS;
}

/* One row of the core: output [64][512] as doubles (not yet rounded), and the softmax's counts. */
static inline void reference_attention_row(const struct reference_row *row, double *output,
                                           struct reference_counts *counts)
{
    const double scale = (double)(float)(1.0 / sqrt((double)REFERENCE_DIMS));
    const int bfloat16 = row->probability_format == REFERENCE_PROBABILITY_BFLOAT16;
    for (uint32_t head = 0; head < REFERENCE_HEADS; head++) {
        double query[REFERENCE_DIMS];
        double *values = output + (size_t)head * REFERENCE_DIMS, maximum = REFERENCE_SCORE_FLOOR, sum = 0.0;
        for (uint32_t dim = 0; dim < REFERENCE_DIMS; dim++) {
            query[dim] = reference_half(row->queries[(size_t)head * REFERENCE_DIMS + dim]);
        }
        memset(values, 0, REFERENCE_DIMS * sizeof(double));
        for (uint32_t first = 0; first < REFERENCE_KEYS; first += REFERENCE_GROUP_KEYS) {
            double scores[REFERENCE_GROUP_KEYS], largest = -INFINITY;
            for (uint32_t key = 0; key < REFERENCE_GROUP_KEYS; key++) {
                const double *stored = row->keys[first + key];
                double dot = 0.0;
                for (uint32_t dim = 0; stored != NULL && dim < REFERENCE_DIMS; dim++) {
                    dot += query[dim] * stored[dim];
                }
                scores[key] = stored != NULL ? dot * scale : -INFINITY;
                largest = fmax(largest, scores[key]);
            }
            if (largest > maximum + row->step) {             /* the kernel's rule: only where P is rounded shows it */
                const double factor = exp(maximum - largest);
                for (uint32_t dim = 0; dim < REFERENCE_DIMS; dim++) {
                    values[dim] *= factor;
                }
                sum *= factor;
                maximum = largest;
                counts->rescales++;
            }
            for (uint32_t key = 0; key < REFERENCE_GROUP_KEYS; key++) {
                const double probability = exp(scores[key] - maximum);
                const double rounded = bfloat16
                    ? reference_rounded(probability, REFERENCE_BFLOAT16_SIGNIFICANT, REFERENCE_BFLOAT16_LEAST_NORMAL,
                                        REFERENCE_BFLOAT16_LEAST_EXPONENT)
                    : reference_rounded(probability, REFERENCE_HALF_SIGNIFICANT, REFERENCE_HALF_LEAST_NORMAL,
                                        REFERENCE_HALF_LEAST_EXPONENT);
                counts->flushed += probability > 0.0 && rounded == 0.0 ? 1u : 0u;
                sum += probability;
                for (uint32_t dim = 0; row->keys[first + key] != NULL && dim < REFERENCE_DIMS; dim++) {
                    values[dim] += rounded * row->keys[first + key][dim];
                }
            }
        }
        const double denominator = sum + exp((double)row->sinks[head] - maximum);
        for (uint32_t dim = 0; dim < REFERENCE_DIMS; dim++) {
            values[dim] /= denominator;
        }
        for (uint32_t pair = 0; pair < REFERENCE_ROPE_PAIRS; pair++) {        /* (x0 + i x1) (cos - i sin) */
            double *place = values + REFERENCE_ROPE_FIRST_DIM + REFERENCE_PAIR * pair;
            const double cosine = row->rotation[REFERENCE_PAIR * pair];
            const double sine = row->rotation[REFERENCE_PAIR * pair + 1u];
            const double first_value = place[0] * cosine + place[1] * sine;
            place[1] = place[1] * cosine - place[0] * sine;
            place[0] = first_value;
        }
    }
}

/* The card's bfloat16 row against the reference rounded once: the rms of the difference over the rms of the rounded
 * reference (a row whose reference is all zeros: 0 if the card's is too, else infinity). */
static inline double reference_error_over_rms(const double *reference, const uint16_t *card, size_t count)
{
    double error_squares = 0.0, squares = 0.0;
    for (size_t index = 0; index < count; index++) {
        const double rounded = reference_bfloat16(reference_bfloat16_bits(reference[index]));
        const double difference = reference_bfloat16(card[index]) - rounded;
        error_squares += difference * difference;
        squares += rounded * rounded;
    }
    return squares > 0.0 ? sqrt(error_squares / squares) : error_squares == 0.0 ? 0.0 : (double)INFINITY;
}

#endif /* ATTENTION_CORE_REFERENCE_H */
