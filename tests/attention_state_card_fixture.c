/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state_card_fixture.c: the fixture of attention_state_card_fixture.h. Test support: no part of the engine.
 *
 * Contract: attention_state_card_fixture.h. The ranges are counted from model_layout.h by hand, in the order of the
 * members of struct attention_state_description; a kind's ranges are a prefix of that order, as
 * attention_state.c checks them.
 */
#include "attention_state_card_fixture.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "anemone/attention_state_records.h"
#include "attention_state_reference.h"

#define RANGES 21u
#define ERROR_FRACTION 9.5367431640625e-07      /* 2^-20 */
#define HALF_A_PLACE 0.5
#define SHARE_REQUIRED 0.999
#define FAILURES_SHOWN 4u
#define PERCENT 100.0
#define DIMS MODEL_HEAD_DIMS
#define QUERY_RANK MODEL_QUERY_RANK

static const char *const check_names[TEST_ATTENTION_CHECK_COUNT] = {
    "rotation", "qr", "window rows", "queries", "index queries", "latent", "pool entries", "slots", "keys",
    "rows alone"};
static const uint32_t prefixes[] = {10, 12, 19, 21};       /* window only, indexer, ratio 1, ratio 2 */
static const uint32_t norm_indices[TEST_ATTENTION_NORMS] = {3, 6, 12, 16};

static uint64_t range_bytes(const struct attention_state_description *layer,
			    uint32_t index)
{
    const uint64_t rows = layer->row_count, sessions = layer->session_capacity;
    const uint64_t entries = layer->pool_ratio == 0 ? 0 : layer->context_token_limit / layer->pool_ratio;
    const uint64_t state_row = MODEL_STATE_ROW_BYTES, key = MODEL_KEY_BYTES;
    const uint64_t index_row = MODEL_INDEX_QUERIES_ROW_BYTES,
		   float_row = sizeof(float) * DIMS;
    const uint64_t qr_row = sizeof(uint16_t) * QUERY_RANK,
		   heads_row = MODEL_HEADS_ROW_BYTES;
    const uint64_t counts[RANGES] = {
	    STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE * rows,
	    MODEL_ROTATION_ROW_BYTES * rows,
	    qr_row * rows,
	    qr_row,
	    qr_row * rows,
	    state_row * rows,
	    state_row,
	    MODEL_RING_SLOTS * state_row * sessions,
	    heads_row * rows,
	    heads_row * rows,
	    index_row * rows,
	    index_row * rows,
	    state_row,
	    state_row * rows,
	    state_row * entries * sessions,
	    key * rows,
	    key,
	    key * entries * sessions,
	    (layer->pool_ratio == 1 ? state_row : float_row) * rows,
	    float_row * rows,
	    ATTENTION_SLOTS_SESSION_BYTES * sessions
    };
    return counts[index];
}

static struct linear_range *range_at(struct attention_state_description *layer,
				     uint32_t index)
{
	struct linear_range *const ranges[RANGES] = {
		&layer->step,	      &layer->rotation,
		&layer->query_rank,   &layer->query_norm,
		&layer->query_normed, &layer->window_input,
		&layer->window_norm,  &layer->ring,
		&layer->heads,	      &layer->queries,
		&layer->index_heads,  &layer->index_queries,
		&layer->pool_norm,    &layer->latent,
		&layer->pool,	      &layer->key_input,
		&layer->key_norm,     &layer->keys,
		&layer->pool_input,   &layer->pool_score,
		&layer->slots
	};
	return ranges[index];
}

bool test_attention_layer_set_aside(
	struct test_linear_card *card,
	const struct attention_state_kernels *kernels,
	struct test_attention_layer *layer_out, struct test_attention_kind kind,
	uint32_t rows, uint32_t sessions, uint32_t context,
	const struct linear_range norms[TEST_ATTENTION_NORMS],
	struct linear_range frequencies)
{
	struct attention_state_description *const layer = &layer_out->layer;
	memset(layer_out, 0, sizeof(*layer_out));
	*layer = (struct attention_state_description){
		.row_count = rows,
		.session_capacity = sessions,
		.context_token_limit = context,
		.theta = kind.theta,
		.first_head = kind.first_head,
		.head_count = MODEL_HEADS_VALUE - kind.first_head,
		.has_index = kind.has_index,
		.pool_ratio = kind.pool_ratio
	};
	const uint32_t range_count =
		prefixes[kind.pool_ratio != 0 ? 1u + kind.pool_ratio :
						kind.has_index];
	for (uint32_t index = 0; index < range_count; index++) {
		uint32_t norm = 0;
		while (norm < TEST_ATTENTION_NORMS &&
		       norm_indices[norm] != index) {
			norm++;
		}
		*range_at(layer, index) =
			norm < TEST_ATTENTION_NORMS ?
				norms[norm] :
				test_linear_card_work(card, range_bytes(layer,
									index));
	}
	layer_out->rotation = (struct attention_rotation_description){
		layer->step, frequencies, layer->rotation, rows, 0
	};
	layer_out->launch_count = kind.pool_ratio != 0 ?
					  TEST_ATTENTION_LAUNCHES :
					  1u + ATTENTION_STATE_QUERIES + 1u;
	return test_linear_card_has_room(card) &&
	       test_linear_expect_status(
		       attention_rotation_prepare(
			       &layer_out->rotation, card->arena_byte_count,
			       kernels, &layer_out->launches[0]),
		       STATUS_OK, "the rotation step is prepared") &&
	       test_linear_expect_status(
		       attention_state_prepare(layer, card->arena_byte_count,
					       kernels,
					       &layer_out->launches[1]),
		       STATUS_OK, "the layer is prepared");
}

bool test_attention_step_put(struct test_linear_card *card,
			     struct linear_range step,
			     const struct test_attention_row *rows,
			     uint32_t grid_rows, uint32_t row_count,
			     uint64_t sequence)
{
	const struct step_block block = { sequence, row_count, 0 };
	bool is_put =
		test_linear_card_put(card, step, 0, &block, sizeof(block));
	for (uint32_t row = 0; row < grid_rows; row++) {
		const struct step_row_record step_row = { rows[row].position, 0,
							  rows[row].session,
							  0 };
		is_put = is_put &&
			 test_linear_card_put(card, step,
					      STEP_BLOCK_BYTES_VALUE +
						      row * sizeof(step_row),
					      &step_row, sizeof(step_row));
	}
    return is_put && test_linear_card_upload(card, step, 0, step.byte_count);
}

uint16_t test_attention_half_at(const struct test_linear_card *card,
				struct linear_range range, uint64_t index)
{
    uint16_t bits = 0;
    (void)test_linear_card_get(card, range, index * sizeof(bits), &bits, sizeof(bits));
    return bits;
}

float test_attention_float_at(const struct test_linear_card *card,
			      struct linear_range range, uint64_t index)
{
    float value = 0.0f;
    (void)test_linear_card_get(card, range, index * sizeof(value), &value, sizeof(value));
    return value;
}

void test_attention_angle_of(const struct test_linear_card *card,
			     struct linear_range rotation, uint32_t row,
			     uint32_t use, uint32_t pair, double *cosine_out,
			     double *sine_out)
{
	const uint64_t at = ((uint64_t)row * MODEL_ROTATION_USES + use) *
				    MODEL_FREQUENCIES * MODEL_PAIR_VALUES +
			    (uint64_t)pair * MODEL_PAIR_VALUES;
	*cosine_out = test_attention_float_at(card, rotation, at);
	*sine_out = test_attention_float_at(card, rotation, at + 1u);
}

bool test_attention_within(double card, double reference, double magnitude, int32_t fraction_bits, int32_t lowest)
{
    const double difference = card > reference ? card - reference : reference - card;
    return difference <=
           HALF_A_PLACE * test_attention_last_place(reference, fraction_bits, lowest) + ERROR_FRACTION * magnitude;
}

void test_attention_tally(struct test_attention_tally *tallies, enum test_attention_check check, bool is_equal,
                          bool is_inside, const char *what, uint64_t index)
{
    struct test_attention_tally *const counted = &tallies[check];
    counted->values++;
    counted->equal += is_equal ? 1u : 0u;
    if (!is_inside) {
        if (counted->outside < FAILURES_SHOWN) {
            test_linear_fail("%s: %s, value %" PRIu64 " outside its tolerance", check_names[check], what, index);
        }
        counted->outside++;
    }
}

void test_attention_tallies_print(const struct test_attention_tally *tallies, const char *what)
{
    for (uint32_t check = 0; check < TEST_ATTENTION_CHECK_COUNT; check++) {
        const struct test_attention_tally *const counted = &tallies[check];
        if (counted->values != 0) {
            (void)printf("%s: %s: %" PRIu64 " values, %" PRIu64 " bit-equal (%.4f%%), %" PRIu64
                         " outside the tolerance, %" PRIu64 " outside FP16's normal range\n",
                         what, check_names[check], counted->values, counted->equal,
                         PERCENT * (double)counted->equal / (double)counted->values, counted->outside,
                         counted->out_of_range);
        }
    }
}

void test_attention_tallies_expect_counted(const struct test_attention_tally *tallies, enum test_attention_check end)
{
    for (uint32_t check = 0; check < (uint32_t)end; check++) {
        if (tallies[check].values == 0) {
            test_linear_fail("%s: no value was checked", check_names[check]);
        }
    }
}

void test_attention_tallies_expect_shares(const struct test_attention_tally *tallies)
{
    for (uint32_t check = TEST_ATTENTION_QR; check < TEST_ATTENTION_CHECK_COUNT; check++) {
        if ((double)tallies[check].equal < SHARE_REQUIRED * (double)tallies[check].values) {
            test_linear_fail("%s: %" PRIu64 " of %" PRIu64 " values bit-equal, fewer than 99.9%%", check_names[check],
                             tallies[check].equal, tallies[check].values);
        }
    }
}
