/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_reference.h: the host reference of the attention core (attention_core_records.h), in double
 * precision from the definitions: the reference's sparse attention (KER:311-387 of its kernel.py: a softmax over a
 * row's keys with the sink of the head added once to its sum), the scale (REF:651 of its model.py) and the turning
 * back of the output (REF:781, REF:392-406). The oracle of the attention core's tests. Test support: no part of the
 * engine.
 *
 * Contract:
 *   - Nothing here calls the engine or the mathematics library (the tests are linked without it): the exponential
 *     is a reduction by ln 2 and a series, a power of two is built from its bits.
 *   - test_attention_core_half_value is the number of FP16 bits; test_attention_half_bits rounds a double once to FP16:
 *     to nearest, a tie to the even fraction, below 2^-14 to whole multiples of 2^-24, from 65520 on to an infinity,
 *     a not-a-number to TEST_ATTENTION_HALF_NOT_A_NUMBER; a zero keeps its sign. bfloat16 is linear_reference.h's.
 *   - test_attention_core_row computes heads first_head .. first_head + head_count - 1 of one row into output
 *     [64][512] (the other heads are not written) from its FP16 queries [64][512], its 640 FP16 keys in the order of
 *     the record header (NULL for an index of -1), the sinks [64] and the 32 (cos, sin) of its use. A score is
 *     q . k in double times ATTENTION_CORE_SCALE. Three forms:
 *       TEST_ATTENTION_KERNEL  the first kernel's: groups of 16 keys in key order; a maximum from
 *              ATTENTION_CORE_SCORE_FLOOR that moves to a group's largest score where that is above it by
 *              more than ATTENTION_CORE_RESCALE_STEP; p = exp(s - m) summed unrounded and rounded to FP16
 *              for the values; O / (sum + exp(sink - m)) rounded to bfloat16; dims 448 .. 511 then turned back from
 *              those values in float32 by test_attention_turn_back's TEST_ATTENTION_TURN_KERNEL form and rounded to
 *              bfloat16 again. The values are bfloat16 numbers.
 *       TEST_ATTENTION_REFERENCE_ORDER  the reference order's kernel: as the first but with one maximum, the row's
 *              largest score (at least the floor), and for the values p rounded to bfloat16, times
 *              ATTENTION_CORE_WEIGHT_SCALE rounded to FP16, over that scale again.
 *       TEST_ATTENTION_EXACT   the definition alone: one softmax with the row's largest score (at least the
 *              floor), p unrounded, the output turned back in double and not rounded: the double result the
 *              oracle's stand-in and the engine are both held to (plan 3.2).
 *     It refuses a NULL argument or heads beyond 64 (STATUS_INVALID_ARGUMENT) and answers
 *     STATUS_OUT_OF_MEMORY where its working memory is not obtained.
 *   - The turning back's forms: TEST_ATTENTION_TURN_KERNEL the record header's, (fmaf(first, cos, second sin),
 *     fmaf(second, cos, -(first sin))), with the host's own fused multiply-add; TEST_ATTENTION_TURN_OTHER the other
 *     product fused in each part; TEST_ATTENTION_TURN_PRODUCTS both products rounded, then added.
 *   - A distance (struct test_attention_distance) collects values (bfloat16 bits) against reference values: the
 *     squares of their differences and of the reference, how many are bit-equal to the reference rounded once to
 *     bfloat16, the largest difference in last places of bfloat16, how many are no finite number where the
 *     reference is one. test_attention_core_within: none of the last, and the rms of the differences at most relative
 *     times the rms of the reference (of a reference of zeros: no difference). test_attention_relative_squared is
 *     the square of that ratio (0 of nothing).
 *   - Every function may be called from any thread.
 */
#ifndef TEST_ATTENTION_CORE_REFERENCE_H
#define TEST_ATTENTION_CORE_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/attention_core_records.h"
#include "anemone/status.h"

#define TEST_ATTENTION_KEYS (MODEL_WINDOW + MODEL_SELECTED)
#define TEST_ATTENTION_ROW_VALUES \
	(MODEL_HEADS_ROW_BYTES / sizeof(uint16_t)) /* of a row: 64 heads */
#define TEST_ATTENTION_HALF_NOT_A_NUMBER 0x7E00u
#define TEST_ATTENTION_TOLERANCE 2.0e-3         /* of a row's rms: the core's contract against its host reference */

enum test_attention_form { TEST_ATTENTION_KERNEL, TEST_ATTENTION_REFERENCE_ORDER, TEST_ATTENTION_EXACT };
enum test_attention_turn_form { TEST_ATTENTION_TURN_KERNEL, TEST_ATTENTION_TURN_OTHER, TEST_ATTENTION_TURN_PRODUCTS,
                                TEST_ATTENTION_TURN_FORMS };

struct test_attention_row {
    const uint16_t *queries;                    /* FP16 [64][512] */
    const uint16_t *keys[TEST_ATTENTION_KEYS];  /* FP16 [512] each; NULL for an index of -1 */
    const float *sinks;                         /* [64] */
    const float *rotation;                      /* [32][cos, sin] */
    uint32_t first_head;
    uint32_t head_count;
    enum test_attention_form form;
};

struct test_attention_distance {
    double error_squares;
    double squares;
    uint64_t value_count;
    uint64_t equal_count;
    uint64_t not_finite_count;
    int32_t largest_places;
    uint32_t reserved_zero;
};

double test_attention_exp(double exponent);
/* Part 0 (the real) or 1 (the imaginary) of (first + i second)(cosine - i sine) in float32, in a form. */
float test_attention_turn_back(float first, float second, float cosine, float sine, uint32_t part,
                               enum test_attention_turn_form form);
/* From stream: a pair, numbers of bfloat16 and FP16 alike in -2 .. 2, and an angle, cos and sin of float32 in -1 .. 1,
 * the first whose part, rounded to bfloat16, is not the kernel's form in form. False where none is found. */
bool test_attention_turn_case(uint64_t stream, uint32_t part,
			      enum test_attention_turn_form form,
			      float pair[MODEL_PAIR_VALUES],
			      float angle[MODEL_PAIR_VALUES]);
double test_attention_core_half_value(uint16_t bits);
uint16_t test_attention_half_bits(double value);

MUST_CHECK result_status
test_attention_core_row(const struct test_attention_row *row, double *output);

/* values [count] as bfloat16 bits against reference [count]. */
void test_attention_distance_add(struct test_attention_distance *distance, const uint16_t *values,
                                 const double *reference, uint64_t count);
double test_attention_relative_squared(const struct test_attention_distance *distance);
bool test_attention_core_within(const struct test_attention_distance *distance, double relative);

#endif
