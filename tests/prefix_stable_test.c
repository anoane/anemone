/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Fixed-capacity dispatch and growing-prefix provenance. Host-only; no CUDA
 * initialization. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_internal.h"
#include "server_internal.h"
#include "step_driver.h"

#define CONTEXT 65536u
#define CAPACITY 1024u
#define SMALL_PROGRAM 64u
#define SOURCE_PROMPT 25005u
#define EXTENDED_PROMPT 25017u
#define KEPT_PREFIX 25000u
#define SOURCE_DECODER 24877u
#define SOURCE_GENERATED 64u
#define EXTRA_REACH 5000u
#define COMMON_BEFORE_CONE 22000u
#define COMMON_IN_CONE 24000u
#define WHOLE_CONE_BOUNDARY 22600u
#define PROGRAMS 4u
#define SLOTS 2u
#define VIEWS 2u
#define VIEW_ROWS (CAPACITY / VIEWS)
#define VIEW_BYTES (STEP_BLOCK_BYTES_VALUE + VIEW_ROWS * STEP_ROW_BYTES_VALUE)

enum program_index {
	THIN_PROGRAM,
	SMALL_FULL_PROGRAM,
	PADDED_FULL_PROGRAM,
	PADDED_FRONT_PROGRAM
};

static uint32_t tokens[CONTEXT];
static uint32_t request_tokens[CONTEXT];
static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "prefix stable: %s\n", message);
		failures++;
	}
}

static void padded_dispatch(void)
{
	const struct model_limits limits = { .context_token_limit = CONTEXT,
					     .row_capacity = CAPACITY,
					     .session_capacity = SLOTS };
	const struct execution_program programs[PROGRAMS] = {
		{ .kind = PROGRAM_THIN,
		  .row_count = LINEAR_MAXIMUM_ROWS,
		  .limits = limits,
		  .front_segment_count = 1 },
		{ .kind = PROGRAM_FULL,
		  .row_count = SMALL_PROGRAM,
		  .limits = limits },
		{ .kind = PROGRAM_FULL,
		  .row_count = CAPACITY,
		  .limits = limits },
		{ .kind = PROGRAM_WIDE, .row_count = CAPACITY, .limits = limits }
	};
	const struct execution_program *pointers[PROGRAMS];
	for (uint32_t index = 0; index < PROGRAMS; index++)
		pointers[index] = &programs[index];
	uint32_t counts[SLOTS] = { 0 }, next[SLOTS] = { 0 };
	struct step_driver driver = { .programs = pointers,
				      .program_count = PROGRAMS,
				      .largest_row_count = CAPACITY,
				      .token_counts = counts,
				      .next_positions = next };
	struct step_row_record rows[CAPACITY];
	for (uint32_t index = 0; index < CAPACITY; index++)
		rows[index] = (struct step_row_record){ (int32_t)index, 1, 0,
							STEP_ROW_PADDED };
	for (uint32_t front = 0; front <= 1u; front++) {
		for (uint32_t index = 0; index < CAPACITY; index++)
			rows[index].flags = STEP_ROW_PADDED |
					    (front != 0 ? STEP_ROW_FRONT : 0u);
		for (uint32_t live = 1; live <= CAPACITY; live++) {
			const uint32_t *graphs = NULL;
			const struct execution_program *selected =
				step_program_for_rows(&driver, rows, live,
						      &graphs);
			expect(selected ==
				       &programs[front != 0 ?
							 PADDED_FRONT_PROGRAM :
							 PADDED_FULL_PROGRAM],
			       "every live count selects the same physical chunk capacity");
			expect(step_rows_check(&driver, rows, live, true) ==
				       STATUS_OK,
			       "partial fixed-capacity prompt rows pass real driver validation");
		}
	}
	rows[0].flags = STEP_ROW_PADDED | STEP_ROW_LOGITS;
	expect(step_rows_check(&driver, rows, 1, true) ==
		       STATUS_INVALID_ARGUMENT,
	       "a padded prefill program cannot consume the sampling row");
	rows[0].flags = STEP_ROW_PADDED;
	rows[1].flags = STEP_ROW_PADDED;
	rows[1].session_slot = 1;
	rows[1].position = 0;
	expect(step_rows_check(&driver, rows, SLOTS, true) ==
		       STATUS_INVALID_ARGUMENT,
	       "a padded prefill step cannot mix owners");
	rows[0].flags = 0;
	const uint32_t *graphs = NULL;
	expect(step_program_for_rows(&driver, rows, 1, &graphs) == &programs[0],
	       "unflagged decode retains the original THIN selection");
}

static void empty_view_flags(void)
{
	const struct step_view views[VIEWS] = {
		{ 0, 0, VIEW_ROWS }, { VIEW_BYTES, VIEW_ROWS, VIEW_ROWS }
	};
	const struct program_card_record card = { .views = views,
						  .view_count = VIEWS };
	const struct execution_program program = { .cards = &card,
						   .card_count = 1 };
	const struct step_driver driver = { 0 };
	unsigned char bytes[VIEWS * VIEW_BYTES];
	struct step_row_record row = { .flags = STEP_ROW_PADDED };
	for (uint32_t padded = 0; padded <= 1u; padded++) {
		row.flags = padded != 0 ? STEP_ROW_PADDED : 0u;
		step_views_write(&driver, &program, bytes, 1, &row, 1);
		for (uint32_t view = 0; view < VIEWS; view++) {
			struct step_block header;
			memcpy(&header, bytes + view * VIEW_BYTES,
			       sizeof(header));
			expect(header.sequence == 1 &&
				       header.row_count ==
					       (view == 0 ? 1u : 0u),
			       "an empty half preserves its sequence and zero live count");
			expect(header.flags ==
				       (padded != 0 ? STEP_BLOCK_PADDED : 0u),
			       "padding metadata reaches empty views without reading an inactive row");
		}
	}
}

static void growing_provenance(void)
{
	struct inference_engine engine = {
		.description = { .context_token_limit = CONTEXT,
				 .session_slot_count = 1,
				 .chunks_in_flight = 1,
				 .prefill_prefix_stable = 1,
				 .front_steps = 1,
				 .wide_rows = CAPACITY,
				 .prefill_cone_skip = 1,
				 .prefill_margin = MODEL_WINDOW - 1u },
		.cone_rows = MODEL_CONE_ROWS,
		.wide_rows = CAPACITY,
		.full_rows = CAPACITY
	};
	struct http_server server = {
		.engine = &engine,
		.histories = tokens,
		.description = { .context_token_limit = CONTEXT,
				 .session_slot_count = 1,
				 .prefill_prefix_stable = 1 },
		.exact_prefix_reuse = SERVER_PREFIX_COLD_PLAN
	};
	struct slot_history *const history = &server.slot_histories[0];
	*history = (struct slot_history){
		.token_count = SOURCE_PROMPT + SOURCE_GENERATED,
		.processed_count = SOURCE_PROMPT + SOURCE_GENERATED,
		.reach_count = SOURCE_PROMPT + SOURCE_GENERATED,
		.cold_prompt_count = SOURCE_PROMPT,
		.decoder_from = SOURCE_DECODER
	};
	const uint8_t no_image[SERVER_IMAGE_IDENTITY_BYTES] = { 0 };
	expect(server_prefix_kept(&server, 0, tokens, EXTENDED_PROMPT,
				  no_image) == KEPT_PREFIX,
	       "an extension keeps almost the whole canonical prompt but excludes "
	       "generated THIN rows");
	struct connection connection = {
		.token_ids = tokens,
		.token_count = EXTENDED_PROMPT,
		.waiting_snapshot_slot = NO_WAITING_SNAPSHOT,
		.cache_namespace = { .byte_count = 1, .bytes = { 1 } }
	};
	uint32_t selected_kept = 0;
	history->cache_namespace.byte_count = 1;
	(void)server_chosen_session_slot(&server, &connection, 0,
					 &selected_kept);
	expect(selected_kept == 0,
	       "foreign namespace cannot use growing-prefix provenance");
	history->cache_namespace = connection.cache_namespace;
	(void)server_chosen_session_slot(&server, &connection, 0,
					 &selected_kept);
	expect(selected_kept == KEPT_PREFIX,
	       "matching namespace and growing-prefix provenance both pass");
	request_tokens[COMMON_IN_CONE] = 1;
	expect(server_prefix_kept(&server, 0, request_tokens, EXTENDED_PROMPT,
				  no_image) == WHOLE_CONE_BOUNDARY,
	       "a partial before the complete tail leaves a whole aligned cone");
	request_tokens[COMMON_IN_CONE] = 0;
	request_tokens[COMMON_BEFORE_CONE] = 1;
	expect(server_prefix_kept(&server, 0, request_tokens, EXTENDED_PROMPT,
				  no_image) == COMMON_BEFORE_CONE,
	       "a partial with enough new rows retains its closed frontend prefix");
	request_tokens[COMMON_BEFORE_CONE] = 0;
	expect(server_prefix_kept(&server, 0, tokens, KEPT_PREFIX, no_image) ==
		       0,
	       "shrinking donors do not inherit extension provenance");
	history->reach_count += EXTRA_REACH;
	expect(server_prefix_kept(&server, 0, tokens, EXTENDED_PROMPT,
				  no_image) == 0,
	       "a lost ring needs an actual checkpoint");
	history->checkpoint_count = KEPT_PREFIX;
	expect(server_prefix_kept(&server, 0, tokens, EXTENDED_PROMPT,
				  no_image) == KEPT_PREFIX,
	       "an aligned prompt checkpoint restores extension eligibility beyond "
	       "ring reach");
	history->checkpoint_count++;
	expect(server_prefix_kept(&server, 0, tokens, EXTENDED_PROMPT,
				  no_image) == 0,
	       "an off-phase checkpoint cannot waive the owner rule");
	history->checkpoint_count = KEPT_PREFIX;
	history->from_snapshot = true;
	expect(server_prefix_kept(&server, 0, tokens, EXTENDED_PROMPT,
				  no_image) == 0,
	       "imported state still has no proven canonical provenance");
}

int main(void)
{
	padded_dispatch();
	empty_view_flags();
	growing_provenance();
	(void)printf(
		"prefix_stable_test: %d failures; all live chunk sizes and "
		"growing-prefix provenance\n",
		failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
