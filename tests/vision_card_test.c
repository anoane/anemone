/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* One explicitly exposed supported card under gpu_run.sh; synthetic fixtures and an opt-in original-weight check. */
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "vision_fixture.h"
#include "vision_original.h"
#include "vision_reference.h"

#define VISION_CARD_DEADLINE_SECONDS 120u
#define VISION_CARD_WAIT_NANOSECONDS 30000000000ull
#define VISION_CARD_TRACE_ARGUMENT_COUNT 2
#define VISION_CARD_ORIGINAL_ARGUMENT_COUNT 5
#define VISION_CARD_PACK_ARGUMENT 2u
#define VISION_CARD_PATCHES_ARGUMENT 3u
#define VISION_CARD_OUTPUT_ARGUMENT 4u

static const struct backend_operations *const backend = &backend_cuda_value;

static bool kernel_oracle_matches(const struct test_vision_fixture *fixture,
				  const unsigned char *received)
{
	const uint16_t *const expected = test_vision_const_at(
		fixture->arena, fixture->description.output.byte_offset);
	const uint16_t *const actual = test_vision_const_at(
		received, fixture->description.output.byte_offset);
	const uint64_t count =
		fixture->description.output.byte_count / sizeof(uint16_t);
	for (uint64_t index = 0; index < count; index++) {
		const float reference = test_vision_float(expected[index]);
		const float difference =
			fabsf(test_vision_float(actual[index]) - reference);
		if (!isfinite(difference) ||
		    difference >
			    fabsf(reference) * TEST_VISION_DEFAULT_RELATIVE +
				    TEST_VISION_DEFAULT_ABSOLUTE) {
			return false;
		}
	}
	return true;
}

static bool input_unchanged(const struct test_vision_fixture *fixture,
			    const unsigned char *received)
{
	const struct linear_range scratch = fixture->description.scratch;
	const struct linear_range output = fixture->description.output;
	for (uint64_t index = 0; index < TEST_VISION_ARENA_BYTES; index++) {
		const bool written =
			(index >= scratch.byte_offset &&
			 index - scratch.byte_offset < scratch.byte_count) ||
			(index >= output.byte_offset &&
			 index - output.byte_offset < output.byte_count);
		if (!written && received[index] != fixture->arena[index]) {
			return false;
		}
	}
	return true;
}

static bool run_case(struct device_card *card, void *staging,
		     const struct vision_kernels *kernels, uint32_t case_index,
		     bool trace)
{
	struct test_vision_fixture fixture;
	if (!test_vision_fixture_create(case_index, &fixture)) {
		return false;
	}
	memcpy(staging, fixture.arena, TEST_VISION_ARENA_BYTES);
	uint64_t ticket = 0;
	bool passed =
		vision_prepare(&fixture.description, TEST_VISION_ARENA_BYTES,
			       kernels, fixture.launches,
			       VISION_MAXIMUM_LAUNCHES,
			       &fixture.launch_count) == STATUS_OK &&
		backend->upload_begin(card, 0, staging, TEST_VISION_ARENA_BYTES,
				      &ticket) == STATUS_OK &&
		backend->upload_wait(card, ticket,
				     VISION_CARD_WAIT_NANOSECONDS) == STATUS_OK;
	for (uint32_t pass = 0; pass < fixture.launch_count && passed;) {
		const uint32_t count = trace ? 1u : fixture.launch_count;
		passed = backend->launch_run(card, &fixture.launches[pass],
					     count) == STATUS_OK &&
			 backend->launch_wait(card,
					      VISION_CARD_WAIT_NANOSECONDS) ==
				 STATUS_OK;
		if (trace && passed) {
			passed =
				backend->download_begin(card, 0, staging,
							TEST_VISION_ARENA_BYTES,
							&ticket) == STATUS_OK &&
				backend->upload_wait(
					card, ticket,
					VISION_CARD_WAIT_NANOSECONDS) ==
					STATUS_OK;
			if (passed) {
				struct vision_arguments record;
				memcpy(&record, fixture.launches[pass].argument,
				       sizeof(record));
				const uint32_t elements =
					record.rows * record.output_width *
					(fixture.launches[pass].kernel ==
							 kernels->operations
								 [VISION_ROPE] ?
						 VISION_QKV_COMPONENTS :
						 1u);
				const uint16_t *const words =
					test_vision_const_at(
						staging,
						record.destination_offset);
				(void)printf("TRACE %u %u", case_index, pass);
				for (uint32_t element = 0; element < elements;
				     element++) {
					(void)printf(" %04x", words[element]);
				}
				(void)putchar('\n');
			}
		}
		pass += count;
	}
	passed = passed &&
		 backend->download_begin(card, 0, staging,
					 TEST_VISION_ARENA_BYTES,
					 &ticket) == STATUS_OK &&
		 backend->upload_wait(card, ticket,
				      VISION_CARD_WAIT_NANOSECONDS) ==
			 STATUS_OK &&
		 input_unchanged(&fixture, staging);
	/* Both the independent C oracle and the actual official GPU Flash fixtures are required. */
	passed = passed &&
		 vision_prepare(&fixture.description, TEST_VISION_ARENA_BYTES,
				&fixture.kernels, fixture.launches,
				VISION_MAXIMUM_LAUNCHES,
				&fixture.launch_count) == STATUS_OK &&
		 test_vision_reference(fixture.launches, fixture.launch_count,
				       fixture.arena) &&
		 kernel_oracle_matches(&fixture, staging) &&
		 test_vision_fixture_golden(&fixture, staging);
	test_vision_fixture_destroy(&fixture);
	return passed;
}

int main(int argument_count, char **arguments)
{
	if (argument_count == VISION_CARD_ORIGINAL_ARGUMENT_COUNT &&
	    strcmp(arguments[1], "--original") == 0) {
		return test_vision_original_card(
			       arguments[VISION_CARD_PACK_ARGUMENT],
			       arguments[VISION_CARD_PATCHES_ARGUMENT],
			       arguments[VISION_CARD_OUTPUT_ARGUMENT]) ?
			       EXIT_SUCCESS :
			       EXIT_FAILURE;
	}
	const bool trace = argument_count == VISION_CARD_TRACE_ARGUMENT_COUNT &&
			   strcmp(arguments[1], "--trace") == 0;
	if (argument_count != 1 && !trace) {
		return EXIT_FAILURE;
	}
	const char *const guarded = getenv("UNDER_GPU_RUN");
	if (guarded == NULL || strcmp(guarded, "1") != 0) {
		(void)puts("vision card test requires gpu_run.sh");
		return EXIT_FAILURE;
	}
	(void)alarm(VISION_CARD_DEADLINE_SECONDS);
	uint32_t count = 0;
	struct card_description_record description;
	if (backend->card_count(&count) != STATUS_OK || count != 1 ||
	    backend->card_describe(0, &description) != STATUS_OK ||
	    (strstr(description.name, "170HX") == NULL &&
	     strstr(description.name, "RTX PRO 6000") == NULL)) {
		(void)puts(
			"vision card test requires exactly one explicitly exposed CMP 170HX or RTX PRO 6000");
		return EXIT_FAILURE;
	}
	struct device_card *card = NULL;
	void *staging = NULL;
	struct vision_kernels kernels;
	const uint32_t arithmetic_class =
		description.arithmetic_class == ARITHMETIC_CLASS_SM_120 ?
			ARITHMETIC_CLASS_SM_120_NATIVE :
			description.arithmetic_class;
	bool passed = backend->card_open(0, TEST_VISION_ARENA_BYTES, &card) ==
			      STATUS_OK &&
		      backend->staging_allocate(TEST_VISION_ARENA_BYTES,
						&staging) == STATUS_OK &&
		      vision_kernels_prepare(backend, card, arithmetic_class,
					     &kernels) == STATUS_OK;
	for (uint32_t case_index = 0; case_index < TEST_VISION_CASES && passed;
	     case_index++) {
		passed = run_case(card, staging, &kernels, case_index, trace);
	}
	if (card != NULL) {
		backend->card_close(card, VISION_CARD_WAIT_NANOSECONDS);
	}
	if (staging != NULL) {
		backend->staging_free(staging);
	}
	(void)printf("vision complete synthetic tower/projector on %s: %s\n",
		     description.name, passed ? "passed" : "failed");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
