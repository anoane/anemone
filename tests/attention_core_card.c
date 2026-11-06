/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_card.c: what the card test of the attention core runs its cases with. Contract:
 * attention_core_card.h. What it leaves open is decided here:
 *   - A launch's arrays are set aside in the order of the description's members, each with its guards; the sinks
 *     in the weights region, which the card's digest shows unwritten.
 *   - The reference of a row is computed once per order when a case is made; a plan only picks rows of it.
 */
#include "attention_core_card.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#define ROWS LINEAR_MAXIMUM_ROWS
#define HEADS MODEL_HEADS_VALUE
#define DIMS MODEL_HEAD_DIMS
#define ROW_VALUES TEST_ATTENTION_ROW_VALUES
#define RING_SLOTS MODEL_RING_SLOTS
#define WINDOW MODEL_WINDOW
#define SELECTED MODEL_SELECTED
#define FP16_LARGEST 0x7BFFu                    /* 65504: the word of the break */
#define STEP_SEQUENCE 1u

static const enum test_attention_form forms[TEST_ATTENTION_ORDERS] = {TEST_ATTENTION_KERNEL,
                                                                      TEST_ATTENTION_REFERENCE_ORDER};

void test_attention_case_free(struct test_attention_case *data)
{
    free(data->queries);
    free(data->ring);
    free(data->pool);
    free(data->selection);
    free(data->rotation);
    for (uint32_t order = 0; order < TEST_ATTENTION_ORDERS; order++) {
        free(data->reference[order]);
    }
    memset(data, 0, sizeof(*data));
}

void test_attention_reference_row(const struct test_attention_case *data,
				  uint32_t row, enum linear_order_code order,
				  struct test_attention_row *reference)
{
    const int32_t position = data->step_rows[row].position;
    const uint64_t session = data->step_rows[row].session_slot;
    memset(reference, 0, sizeof(*reference));
    reference->queries = data->queries + (size_t)row * ROW_VALUES;
    for (uint32_t key = 0; key < WINDOW; key++) {
        const int32_t key_position = position - (int32_t)(WINDOW - 1u) + (int32_t)key;
        reference->keys[key] = key_position < 0 ? NULL
                                                : data->ring + (session * RING_SLOTS +
                                                                (uint32_t)key_position % RING_SLOTS) * DIMS;
    }
    for (uint32_t place = 0; data->pool != NULL && place < SELECTED; place++) {
        const int32_t entry = data->selection[(size_t)row * SELECTED + place];
        reference->keys[WINDOW + place] =
            entry < 0 ? NULL : data->pool + (session * data->pool_entries + (uint64_t)entry) * DIMS;
    }
    reference->sinks = data->sinks;
    reference->rotation = data->rotation + (size_t)row * TEST_ATTENTION_ROTATION_FLOATS +
                          data->rotation_use * TEST_ATTENTION_USE_FLOATS;
    reference->head_count = HEADS;
    reference->form = forms[order == LINEAR_ORDER_REFERENCE ? 1u : 0u];
}

bool test_attention_case_hold(struct test_attention_case *data)
{
    const uint64_t ring_values = (uint64_t)data->session_count * RING_SLOTS * DIMS;
    const uint64_t pool_values = (uint64_t)data->session_count * data->pool_entries * DIMS;
    bool held = true;
    data->queries = calloc((size_t)ROWS * ROW_VALUES, sizeof(uint16_t));
    data->ring = malloc(ring_values * sizeof(uint16_t));
    data->pool = data->pool_ratio == 0 ? NULL : malloc(pool_values * sizeof(uint16_t));
    data->selection = malloc((size_t)ROWS * SELECTED * sizeof(int32_t));
    data->rotation = malloc((size_t)ROWS * TEST_ATTENTION_ROTATION_FLOATS * sizeof(float));
    for (uint32_t order = 0; order < TEST_ATTENTION_ORDERS; order++) {
        data->reference[order] = calloc((size_t)ROWS * ROW_VALUES, sizeof(double));
        held = held && data->reference[order] != NULL;
    }
    if (!held || data->queries == NULL || data->ring == NULL || (data->pool == NULL && pool_values != 0) ||
        data->selection == NULL || data->rotation == NULL) {
        test_linear_fail("the arrays of a case are not held");
        return false;
    }
    for (uint64_t index = 0; index < ring_values; index++) {
        data->ring[index] = TEST_ATTENTION_POISON;
    }
    for (uint64_t index = 0; index < pool_values; index++) {
        data->pool[index] = TEST_ATTENTION_POISON;
    }
    for (size_t index = 0; index < (size_t)ROWS * SELECTED; index++) {
        data->selection[index] = -1;
    }
    return true;
}

bool test_attention_case_references(struct test_attention_case *data)
{
    for (uint32_t order = 0; order < TEST_ATTENTION_ORDERS; order++) {
        for (uint32_t row = 0; row < ROWS; row++) {
            struct test_attention_row reference;
	    test_attention_reference_row(
		    data, row, (enum linear_order_code)order, &reference);
	    if (!test_linear_expect_status(
			test_attention_core_row(
				&reference, data->reference[order] +
						    (size_t)row * ROW_VALUES),
			STATUS_OK, "the reference computes the row")) {
		    return false;
	    }
	}
    }
    return true;
}

void test_attention_plan_of_rows(struct test_attention_plan *plan, const char *name, uint32_t first_row,
                                 uint32_t row_count)
{
    memset(plan, 0, sizeof(*plan));
    plan->name = name;
    for (uint32_t row = 0; row < row_count; row++) {
        plan->source_rows[row] = first_row + row;
    }
    plan->program_rows = row_count;
    plan->step_rows = row_count;
    plan->head_count = HEADS;
}

/* The plan's rows of the case into the image of the card, in the ranges of the description. */
static bool plan_lay_out(struct test_linear_card *card,
			 const struct test_attention_case *data,
			 const struct test_attention_plan *plan,
			 const struct attention_core_description *description)
{
    static int32_t far_selection[SELECTED];
    const struct step_block block = { STEP_SEQUENCE, plan->step_rows, 0 };
    const uint16_t largest = FP16_LARGEST;
    bool laid = test_linear_card_put(card, description->step, 0, &block, sizeof(block)) &&
                test_linear_card_put(card, description->ring, 0, data->ring, description->ring.byte_count) &&
                test_linear_card_put(card, description->sinks, 0, data->sinks, sizeof(data->sinks)) &&
                (data->pool == NULL ||
                 test_linear_card_put(card, description->pool, 0, data->pool, description->pool.byte_count));
    for (uint32_t place = 0; place < SELECTED; place++) {
        far_selection[place] = TEST_ATTENTION_FAR_ENTRY;
    }
    for (uint32_t row = 0; laid && row < plan->program_rows; row++) {
        const uint32_t source = plan->source_rows[row];
	struct step_row_record step_row = data->step_rows[source];
	const bool is_far = plan->far_selections_behind && row >= plan->step_rows;
        step_row.position +=
            plan->window_one_later && (uint32_t)step_row.position + 1u < data->context_token_limit ? 1 : 0;
	laid = test_linear_card_put(card, description->step,
				    STEP_BLOCK_BYTES_VALUE +
					    (uint64_t)row *
						    STEP_ROW_BYTES_VALUE,
				    &step_row, sizeof(step_row)) &&
	       test_linear_card_put(card, description->queries,
				    (uint64_t)row * MODEL_HEADS_ROW_BYTES,
				    data->queries + (size_t)source * ROW_VALUES,
				    MODEL_HEADS_ROW_BYTES) &&
	       test_linear_card_put(
		       card, description->rotation,
		       (uint64_t)row * MODEL_ROTATION_ROW_BYTES,
		       data->rotation +
			       (size_t)source * TEST_ATTENTION_ROTATION_FLOATS,
		       MODEL_ROTATION_ROW_BYTES) &&
	       (data->pool == NULL ||
		test_linear_card_put(card, description->selection,
				     (uint64_t)row * MODEL_SELECTION_ROW_BYTES,
				     is_far ? far_selection :
					      data->selection +
						      (size_t)source * SELECTED,
				     MODEL_SELECTION_ROW_BYTES));
    }
    memset(card->image + description->output.byte_offset, TEST_ATTENTION_UNTOUCHED_OUTPUT,
           description->output.byte_count);
    if (laid && plan->largest_word && plan->source_rows[0] == 0) {   /* the newest window row of the case's row 0 */
	    const struct step_row_record *const row = &data->step_rows[0];
	    laid = test_linear_card_put(
		    card, description->ring,
		    ((uint64_t)row->session_slot * RING_SLOTS +
		     (uint32_t)row->position % RING_SLOTS) *
			    MODEL_STATE_ROW_BYTES,
		    &largest, sizeof(largest));
    }
    return laid;
}

bool test_attention_plan_run(struct test_linear_card *card,
			     const struct attention_core_kernels *kernels,
			     const struct test_attention_case *data,
			     const struct test_attention_plan *plan,
			     enum linear_order_code order, uint16_t *output_out)
{
    const uint64_t rows = plan->program_rows;
    struct attention_core_description description;
    struct kernel_launch launches[ATTENTION_CORE_MOST_LAUNCHES];
    uint32_t launch_count = 0;
    memset(&description, 0, sizeof(description));
    test_linear_context("card %" PRIu32 ", %s, %s, order %" PRIu32, card->card_index, data->name, plan->name,
                        (uint32_t)order);
    test_linear_card_clear(card);
    description.step = test_linear_card_work(
	    card, STEP_BLOCK_BYTES_VALUE + rows * STEP_ROW_BYTES_VALUE);
    description.queries =
	    test_linear_card_work(card, rows * MODEL_HEADS_ROW_BYTES);
    description.ring = test_linear_card_work(
	    card,
	    (uint64_t)data->session_count * RING_SLOTS * MODEL_STATE_ROW_BYTES);
    if (data->pool != NULL) {
	    description.pool = test_linear_card_work(
		    card, (uint64_t)data->session_count * data->pool_entries *
				  MODEL_STATE_ROW_BYTES);
	    description.selection = test_linear_card_work(
		    card, rows * MODEL_SELECTION_ROW_BYTES);
    }
    description.rotation =
	    test_linear_card_work(card, rows * MODEL_ROTATION_ROW_BYTES);
    description.output =
	    test_linear_card_work(card, rows * MODEL_HEADS_ROW_BYTES);
    description.scores =
	    test_linear_card_work(card, rows * ATTENTION_CORE_SCORES_ROW_BYTES);
    description.sinks = test_linear_card_weights(card, sizeof(data->sinks));
    description.row_count = plan->program_rows;
    description.session_capacity = data->session_count;
    description.context_token_limit = data->context_token_limit;
    description.pool_ratio = data->pool_ratio;
    description.first_head = plan->first_head;
    description.head_count = plan->head_count;
    description.order = order;
    description.arithmetic_class = card->arithmetic_class;
    description.window_mode = plan->block_window ? ATTENTION_CORE_WINDOW_BLOCK :
						   ATTENTION_CORE_WINDOW_CAUSAL;
    return test_linear_card_has_room(card) &&
	   plan_lay_out(card, data, plan, &description) &&
	   test_linear_expect_status(
		   attention_core_prepare(&description, card->arena_byte_count,
					  kernels, launches, &launch_count),
		   STATUS_OK, "the launches are prepared") &&
	   test_linear_card_load(card) &&
	   test_linear_card_run(card, launches, launch_count) &&
	   test_linear_card_fetch(card) &&
	   test_linear_card_get(card, description.output, 0, output_out,
				description.output.byte_count) &&
	   test_linear_card_expect_untouched(card, description.step,
					     "the step area") &&
	   test_linear_card_expect_untouched(card, description.queries,
					     "the queries") &&
	   test_linear_card_expect_untouched(card, description.ring,
					     "the ring") &&
	   (data->pool == NULL ||
	    (test_linear_card_expect_untouched(card, description.pool,
					       "the pool") &&
	     test_linear_card_expect_untouched(card, description.selection,
					       "the selection"))) &&
	   test_linear_card_expect_untouched(card, description.rotation,
					     "the rotation") &&
	   test_linear_card_expect_weights_untouched(card);
}

bool test_attention_is_untouched(const uint16_t *values, uint64_t count)
{
    const unsigned char *const bytes = (const unsigned char *)values;
    for (uint64_t index = 0; index < count * sizeof(uint16_t); index++) {
        if (bytes[index] != TEST_ATTENTION_UNTOUCHED_OUTPUT) {
            return false;
        }
    }
    return true;
}
