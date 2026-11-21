/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_attention_state_card_fixture.h: what the card tests of the attention state share: a layer laid out on a card of
 * linear_card_fixture.h with its launches, the step area, what came back, and the tolerances with their counts.
 * Test support: no part of the engine.
 *
 * Contract:
 *   - A layer (struct ds41f_test_attention_layer) of a kind (theta, indexer, pool ratio, and the first of the heads it
 *     runs up to the last) for rows, sessions and a context:
 *     ds41f_test_attention_layer_set_aside sets aside, in the card's work region, every range the kind has, of exactly
 *     the bytes ds41f_attention_state.h counts (by hand here), takes the four norms' ranges and the frequencies' from
 *     the caller, and prepares the rotation step (launch 0) and the layer's launches behind it: launch_count of them.
 *     A range the kind has not is {0, 0}.
 *   - ds41f_test_attention_step_put writes and uploads a step area: the block of the step and grid_rows rows (those
 *     from row_count on: stale rows of no step).
 *   - What came back (received): half_at and float_at read an element of a range; angle_of the (cos, sin) the
 *     rotation step gave a row and use.
 *   - ds41f_test_attention_within: whether card is a rounding to nearest of a value within 2^-20 of the magnitude of
 *     the reference's terms (the error float32 arithmetic may add) of the reference: |card - reference| at most half a
 *     place of the format at the reference plus that. So the nearest value, or either one near a tie; a value cut
 *     toward zero is refused wherever it is not the nearest. ds41f_test_attention_tally counts a value of a check
 *     (equal, inside the tolerance) and fails the test, with a line for each of the first few, for one outside;
 *     ds41f_test_attention_tallies_print prints one line per check that counted a value;
 *     ds41f_test_attention_tallies_expect_counted fails the test for each check below end that counted none;
 *     ds41f_test_attention_tallies_expect_shares for each check but the rotation of whose values fewer than 99.9% are
 *     bit-equal.
 */
#ifndef DS41F_ATTENTION_STATE_CARD_FIXTURE_H
#define DS41F_ATTENTION_STATE_CARD_FIXTURE_H

#include <stdbool.h>
#include <stdint.h>

#include "models/deepseek_v41_flash/attention/ds41f_attention_state.h"
#include "engine/library/block_scaled_linear/linear_card_fixture.h"

#define DS41F_TEST_ATTENTION_NORMS 4u                 /* q_norm, kv_norm, the compressor's, k_norm */
#define DS41F_TEST_ATTENTION_LAUNCHES \
	5u /* the rotation step, then DS41F_ATTENTION_STATE_LAUNCH_COUNT */
#define DS41F_TEST_ATTENTION_BFLOAT16_BITS 7
#define DS41F_TEST_ATTENTION_BFLOAT16_LOWEST (-126)
#define DS41F_TEST_ATTENTION_HALF_BITS 10
#define DS41F_TEST_ATTENTION_HALF_LOWEST (-14)
#define DS41F_TEST_ATTENTION_FLOAT_BITS 23
#define DS41F_TEST_ATTENTION_FLOAT_LOWEST (-126)

enum ds41f_test_attention_norm { DS41F_TEST_ATTENTION_Q_NORM, DS41F_TEST_ATTENTION_KV_NORM,
    DS41F_TEST_ATTENTION_POOL_NORM,
                           DS41F_TEST_ATTENTION_K_NORM };
enum ds41f_test_attention_check {
    DS41F_TEST_ATTENTION_ROTATION, DS41F_TEST_ATTENTION_QR, DS41F_TEST_ATTENTION_WINDOW, DS41F_TEST_ATTENTION_QUERIES,
    DS41F_TEST_ATTENTION_INDEX_QUERIES, DS41F_TEST_ATTENTION_LATENT, DS41F_TEST_ATTENTION_POOL,
        DS41F_TEST_ATTENTION_SLOTS,
    DS41F_TEST_ATTENTION_KEYS, DS41F_TEST_ATTENTION_ALONE, DS41F_TEST_ATTENTION_CHECK_COUNT
};

struct ds41f_test_attention_kind {
    uint32_t theta;
    uint32_t has_index;
    uint32_t pool_ratio;
    uint32_t first_head;                        /* the layer runs heads first_head to the last */
};

struct ds41f_test_attention_layer {
	struct ds41f_attention_state_description layer;
	struct ds41f_attention_rotation_description rotation;
	struct kernel_launch launches[DS41F_TEST_ATTENTION_LAUNCHES];
	uint32_t launch_count;
};

struct ds41f_test_attention_row {
    int32_t position;
    uint32_t session;
};

struct ds41f_test_attention_tally {
    uint64_t values;
    uint64_t equal;
    uint64_t outside;
    uint64_t out_of_range;                      /* the reference's values outside FP16's normal range (risk 8) */
};

bool ds41f_test_attention_layer_set_aside(
	struct test_linear_card *card,
	const struct ds41f_attention_state_kernels *kernels,
	struct ds41f_test_attention_layer *layer_out, struct ds41f_test_attention_kind kind,
	uint32_t rows, uint32_t sessions, uint32_t context,
	const struct linear_range norms[DS41F_TEST_ATTENTION_NORMS],
	struct linear_range frequencies);
bool ds41f_test_attention_step_put(struct test_linear_card *card,
			     struct linear_range step,
			     const struct ds41f_test_attention_row *rows,
			     uint32_t grid_rows, uint32_t row_count,
			     uint64_t sequence);

uint16_t ds41f_test_attention_half_at(const struct test_linear_card *card,
				struct linear_range range, uint64_t index);
float ds41f_test_attention_float_at(const struct test_linear_card *card,
			      struct linear_range range, uint64_t index);
void ds41f_test_attention_angle_of(const struct test_linear_card *card,
			     struct linear_range rotation, uint32_t row,
			     uint32_t use, uint32_t pair, double *cosine_out,
			     double *sine_out);

bool ds41f_test_attention_within(double card, double reference, double magnitude, int32_t fraction_bits,
                                 int32_t lowest);
void ds41f_test_attention_tally(struct ds41f_test_attention_tally *tallies, enum ds41f_test_attention_check check,
                                bool is_equal,
                          bool is_inside, const char *what, uint64_t index);
void ds41f_test_attention_tallies_print(const struct ds41f_test_attention_tally *tallies, const char *what);
void ds41f_test_attention_tallies_expect_counted(const struct ds41f_test_attention_tally *tallies,
						 enum ds41f_test_attention_check end);
void ds41f_test_attention_tallies_expect_shares(const struct ds41f_test_attention_tally *tallies);

#endif
