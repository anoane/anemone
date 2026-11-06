/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image alternatives are immutable per-step selections; text keeps its original launch and graph tables. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_records.h"
#include "anemone/hooks.h"
#include "anemone/step_images.h"
#include "host_memory_backend.h"
#include "prepare_common.h"
#include "step_driver.h"

#define ARENA_BYTES 65536u
#define CONTEXT_TOKENS 64u
#define STEP_ROWS 2u
#define SESSION_OFFSET 512u
#define RESULT_OFFSET 1024u
#define WAIT_NANOSECONDS 1000000000ull
#define IMAGE_LAUNCH_VALUE 19u
#define TEXT_LAUNCH_VALUE 17u
#define GRAPH_HANDLE_PLUS_ONE 7u
#define ONE_PROGRAM 1u
#define LAST_ROW 1u

static int failures;

static void expect(bool condition, const char *description)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", description);
		failures++;
	}
}

int main(void)
{
	test_host_memory_backend_reset(1, ARENA_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	struct device_card *card = NULL;
	if (test_host_memory_backend.card_open(0, ARENA_BYTES, &card) !=
	    STATUS_OK) {
		return EXIT_FAILURE;
	}
	uint32_t kernel = BACKEND_NO_KERNEL;
	expect(test_host_memory_backend.kernel_prepare(
		       card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
		       sizeof(struct fill_record), &kernel) == STATUS_OK,
	       "prepare host fixture kernel");
	struct kernel_launch text_launch, image_launch;
	const struct fill_record text_record = { .value = TEXT_LAUNCH_VALUE };
	const struct fill_record image_record = { .value = IMAGE_LAUNCH_VALUE };
	prepare_launch_fill(&text_launch, kernel, 1, 1, 1, 0, &text_record,
			    sizeof(text_record));
	prepare_launch_fill(&image_launch, kernel, 1, 1, 1, 0, &image_record,
			    sizeof(image_record));
	struct program_card_record text_card = {
		.launches = &text_launch,
		.arena_byte_count = ARENA_BYTES,
		.step_offset = 0,
		.session_offset = SESSION_OFFSET,
		.result_offset = RESULT_OFFSET,
		.logits_offset = PROGRAM_NOWHERE,
		.streams_offset = PROGRAM_NOWHERE,
		.carried_pre_offset = PROGRAM_NOWHERE,
		.engram_rows_offset = PROGRAM_NOWHERE,
		.round_offset = PROGRAM_NOWHERE,
		.launch_count = 1
	};
	struct program_card_record image_card = text_card;
	image_card.launches = &image_launch;
	const struct segment_record segment = { .card = 0,
						.layer = SEGMENT_NO_LAYER,
						.launch_count = 1 };
	const struct execution_program text = {
		.cards = &text_card,
		.segments = &segment,
		.limits = { .context_token_limit = CONTEXT_TOKENS,
			    .row_capacity = STEP_ROWS,
			    .session_capacity = 1,
			    .state_width = STATE_WIDTH_16 },
		.card_count = 1,
		.segment_count = 1,
		.row_count = STEP_ROWS,
		.draft_segment_first = 1
	};
	struct execution_program image = text;
	image.cards = &image_card;
	const struct execution_program *text_programs[ONE_PROGRAM] = { &text };
	const struct execution_program *image_programs[ONE_PROGRAM] = { &image };
	struct step_driver *driver = NULL;
	expect(step_driver_create(&test_host_memory_backend, &card,
				  text_programs, ONE_PROGRAM, NULL,
				  WAIT_NANOSECONDS, &driver) == STATUS_OK,
	       "create base host driver");
	if (driver != NULL) {
		image.row_count = 1;
		expect(step_image_programs_attach(driver, image_programs,
						  ONE_PROGRAM) ==
				       STATUS_INVALID_ARGUMENT &&
			       driver->image_programs == NULL,
		       "different image row shape is rejected atomically");
		image.row_count = STEP_ROWS;
		image_card.step_offset++;
		expect(step_image_programs_attach(driver, image_programs,
						  ONE_PROGRAM) ==
				       STATUS_INVALID_ARGUMENT &&
			       driver->image_programs == NULL,
		       "different image step area is rejected atomically");
		image_card.step_offset--;
#if CHECKING_HOOKS_ENABLED
		expect(fault_arm("step_driver_memory", 1),
		       "arm image program pointer allocation failure");
		expect(step_image_programs_attach(driver, image_programs,
						  ONE_PROGRAM) ==
				       STATUS_OUT_OF_MEMORY &&
			       driver->image_programs == NULL,
		       "image program allocation failure leaves base driver usable");
		expect(fault_disarm() == 1u,
		       "image program allocation fault reached");
#endif
		expect(step_image_programs_attach(driver, image_programs,
						  ONE_PROGRAM) == STATUS_OK,
		       "compatible image launches attach without changing the base programs");
		expect(step_image_programs_attach(driver, image_programs,
						  ONE_PROGRAM) ==
			       STATUS_INVALID_ARGUMENT,
		       "image alternatives attach once");
		driver->run_graphs =
			calloc(ONE_PROGRAM, sizeof(*driver->run_graphs));
		expect(driver->run_graphs != NULL,
		       "allocate a synthetic base graph table");
		if (driver->run_graphs != NULL) {
			driver->run_graphs[0] = GRAPH_HANDLE_PLUS_ONE;
			struct step_row_record rows[STEP_ROWS] = { { 0 } };
			const uint32_t *graphs = NULL;
			expect(step_program_for_rows(driver, rows, STEP_ROWS,
						     &graphs) == &text &&
				       graphs == driver->run_graphs &&
				       graphs[0] == GRAPH_HANDLE_PLUS_ONE,
			       "text-only rows preserve exact base program and graph pointer");
			rows[LAST_ROW].flags = STEP_ROW_IMAGE;
			expect(step_program_for_rows(driver, rows, STEP_ROWS,
						     &graphs) == &image &&
				       graphs == NULL,
			       "one image row selects the checked eager image program for the whole step");
			rows[LAST_ROW].flags = 0;
			expect(step_program_for_rows(driver, rows, STEP_ROWS,
						     &graphs) == &text &&
				       graphs == driver->run_graphs,
			       "a following text step keeps the base graph independently");
		}
		step_driver_destroy(driver);
	}
	test_host_memory_backend.card_close(card, WAIT_NANOSECONDS);
	expect(test_host_memory_backend_staging_block_count() == 0,
	       "driver releases owned image pointer tables");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
