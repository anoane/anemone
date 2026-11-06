/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_records.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "prepare_common.h"

#define ARENA_BYTES 4096u
#define SESSION_OFFSET 512u
#define RESULT_OFFSET 1024u
#define RING_OFFSET 2048u
#define CHECKPOINT_OFFSET 3072u
#define RING_ROWS 16u
#define WINDOW_ROWS 7u
#define CONTEXT_TOKENS 32u
#define SAVED_BOUNDARY 20u
#define SLOTS 2u
#define WAIT_LIMIT 1000000000ull
#define POISON_BYTE 0xD3u
#define CHECKPOINT_RANGES 2u
#define OTHER_CHECKPOINT_OFFSET 3584u
#define PIPELINE_DEPTH 2u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "step_checkpoint_test: %s\n", message);
		failures++;
	}
}

static bool snapshot_read(void *context, unsigned char *bytes,
			  uint64_t byte_count)
{
	for (uint64_t index = 0; index < byte_count; index++) {
		bytes[index] = context != NULL ? POISON_BYTE :
						 (unsigned char)index;
	}
	return true;
}

int main(void)
{
	struct device_card *card = NULL;
	struct step_driver *driver = NULL;
	uint32_t kernel = 0;
	uint32_t tokens[SAVED_BOUNDARY] = { 0 };
	test_host_memory_backend_reset(1, ARENA_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	expect(test_host_memory_backend.card_open(0, ARENA_BYTES, &card) ==
		       STATUS_OK,
	       "card opens");
	expect(test_host_memory_backend.kernel_prepare(
		       card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
		       FILL_RECORD_BYTES, &kernel) == STATUS_OK,
	       "kernel prepares");
	const struct fill_record record = { 0 };
	struct kernel_launch launch;
	prepare_launch_fill(&launch, kernel, 1, 1, 1, 0, &record,
			    sizeof(record));
	const struct segment_record segment = { .layer = SEGMENT_NO_LAYER,
						.launch_count = 1 };
	const struct program_card_record area = {
		.launches = &launch,
		.arena_byte_count = ARENA_BYTES,
		.step_offset = 0,
		.session_offset = SESSION_OFFSET,
		.result_offset = RESULT_OFFSET,
		.logits_offset = PROGRAM_NOWHERE,
		.round_offset = PROGRAM_NOWHERE,
		.streams_offset = PROGRAM_NOWHERE,
		.carried_pre_offset = PROGRAM_NOWHERE,
		.engram_rows_offset = PROGRAM_NOWHERE,
		.launch_count = 1
	};
	const struct session_range session = {
		RING_OFFSET, RING_ROWS, RING_ROWS, 1, 1, 0, 0
	};
	struct window_checkpoint range = {
		RING_OFFSET, CHECKPOINT_OFFSET, RING_ROWS, RING_ROWS,
		1,	     WINDOW_ROWS,	0
	};
	struct execution_program program = {
		.cards = &area,
		.segments = &segment,
		.limits = { .context_token_limit = CONTEXT_TOKENS,
			    .row_capacity = 1,
			    .session_capacity = SLOTS,
			    .state_width = STATE_WIDTH_16 },
		.card_count = 1,
		.segment_count = 1,
		.draft_segment_first = 1,
		.front_segment_count = 1,
		.row_count = 1,
		.session_ranges = &session,
		.session_range_count = 1,
		.checkpoints = &range,
		.checkpoint_count = 1
	};
	const struct execution_program *const pointer = &program;
	expect(step_driver_create(&test_host_memory_backend, &card, &pointer, 1,
				  NULL, WAIT_LIMIT, &driver) == STATUS_OK,
	       "driver prepares");
	if (driver != NULL) {
		expect(step_checkpoint_supported(driver),
		       "checkpoint supported");
		expect(!step_checkpoint_supported(NULL),
		       "no null driver checkpoint");
		expect(step_session_load(driver, 1, tokens, SAVED_BOUNDARY,
					 snapshot_read, NULL) == STATUS_OK,
		       "session loads");
		expect(step_checkpoint_save(NULL, 0, 0) ==
			       STATUS_INVALID_ARGUMENT,
		       "null refused");
		expect(step_checkpoint_save(driver, SLOTS, 0) ==
			       STATUS_INVALID_ARGUMENT,
		       "slot refused");
		expect(step_checkpoint_save(driver, 1, SAVED_BOUNDARY - 1u) ==
			       STATUS_INVALID_ARGUMENT,
		       "odd boundary refused");
		expect(step_checkpoint_save(driver, 1,
					    SAVED_BOUNDARY + SLOTS) ==
			       STATUS_INVALID_ARGUMENT,
		       "beyond history refused");
		expect(step_checkpoint_save(driver, 1, SAVED_BOUNDARY) ==
			       STATUS_OK,
		       "wrapped save");
		const unsigned char *const arena =
			test_host_memory_backend_arena_of(card);
		for (uint32_t index = 0; index < WINDOW_ROWS; index++) {
			expect(arena[CHECKPOINT_OFFSET + WINDOW_ROWS + index] ==
				       (unsigned char)((SAVED_BOUNDARY -
							WINDOW_ROWS + index) %
						       RING_ROWS),
			       "packed window bits");
		}
		for (uint32_t slot = 0; slot < SLOTS; slot++) {
			expect(step_session_load(driver, slot, tokens,
						 SAVED_BOUNDARY, snapshot_read,
						 &kernel) == STATUS_OK,
			       "later work overwrites both rings");
		}
		expect(step_checkpoint_restore(driver, 1, SAVED_BOUNDARY) ==
			       STATUS_OK,
		       "wrapped restore");
		for (uint32_t index = 0; index < RING_ROWS; index++) {
			const bool restored =
				index >= SAVED_BOUNDARY - WINDOW_ROWS ||
				index < SAVED_BOUNDARY % RING_ROWS;
			expect(arena[RING_OFFSET + RING_ROWS + index] ==
				       (restored ? (unsigned char)index :
						   POISON_BYTE),
			       "only preceding rows restored in owning slot");
			expect(arena[RING_OFFSET + index] == POISON_BYTE,
			       "other slot untouched");
		}
		expect(step_checkpoint_save(driver, 1, 0) == STATUS_OK,
		       "empty window needs no copy");
		expect(step_pipeline_start(driver, PIPELINE_DEPTH) == STATUS_OK,
		       "pipeline starts before any step");
		const struct step_row_record row = { SAVED_BOUNDARY, 0, 1,
						     STEP_ROW_FRONT };
		struct step_row_result result;
		expect(step_submit(driver, &row, 1) == STATUS_OK,
		       "a front visit goes in flight");
		expect(step_checkpoint_save(driver, 1, SAVED_BOUNDARY) ==
			       STATUS_INVALID_ARGUMENT,
		       "capture refuses unfinished visits");
		expect(step_checkpoint_restore(driver, 0, SAVED_BOUNDARY) ==
			       STATUS_INVALID_ARGUMENT,
		       "restore refuses unfinished visits of another slot too");
		expect(step_finish(driver, &result) == STATUS_OK,
		       "all visits retire");
		expect(step_checkpoint_save(driver, 1, SAVED_BOUNDARY) ==
			       STATUS_OK,
		       "capture succeeds after global retirement");
	}
	step_driver_destroy(driver);
	driver = NULL;
	range.checkpoint_offset = RING_OFFSET;
	expect(step_driver_create(&test_host_memory_backend, &card, &pointer, 1,
				  NULL, WAIT_LIMIT,
				  &driver) == STATUS_INVALID_ARGUMENT,
	       "overlapping checkpoint refused");
	expect(driver == NULL, "refusal leaves output untouched");
	range.checkpoint_offset = ARENA_BYTES;
	expect(step_driver_create(&test_host_memory_backend, &card, &pointer, 1,
				  NULL, WAIT_LIMIT,
				  &driver) == STATUS_INVALID_ARGUMENT,
	       "out of arena checkpoint refused");
	range.checkpoint_offset = RESULT_OFFSET;
	expect(step_driver_create(&test_host_memory_backend, &card, &pointer, 1,
				  NULL, WAIT_LIMIT,
				  &driver) == STATUS_INVALID_ARGUMENT,
	       "checkpoint overlapping results refused");
	range.checkpoint_offset = CHECKPOINT_OFFSET;
	struct window_checkpoint ranges[CHECKPOINT_RANGES] = { range, range };
	ranges[1].ring_offset = CHECKPOINT_OFFSET;
	ranges[1].checkpoint_offset = OTHER_CHECKPOINT_OFFSET;
	program.checkpoints = ranges;
	program.checkpoint_count = CHECKPOINT_RANGES;
	expect(step_driver_create(&test_host_memory_backend, &card, &pointer, 1,
				  NULL, WAIT_LIMIT,
				  &driver) == STATUS_INVALID_ARGUMENT,
	       "checkpoint overlapping another range's ring refused");
	test_host_memory_backend.card_close(card, WAIT_LIMIT);
	expect(test_host_memory_backend_staging_block_count() == 0,
	       "staging retired");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
