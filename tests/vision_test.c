/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host-only preparation bounds and complete small synthetic tower against the official PyTorch fixture. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vision_fixture.h"
#include "vision_reference.h"

#define EXPECTED_SMALL_LAYERS 2u
#define IMPOSSIBLE_CAPACITY 0u
#define SCALAR_SENTINEL 12345u
#define CHANGED_WORD 0x3F80u
#define TRACE_ARGUMENT_COUNT 2

static bool refusals(struct test_vision_fixture *fixture)
{
	struct vision_description invalid = fixture->description;
	uint32_t count = SCALAR_SENTINEL;
	invalid.grid_width = 0;
	if (vision_prepare(&invalid, TEST_VISION_ARENA_BYTES, &fixture->kernels,
			   fixture->launches, VISION_MAXIMUM_LAUNCHES,
			   &count) != STATUS_INVALID_ARGUMENT ||
	    count != SCALAR_SENTINEL) {
		return false;
	}
	invalid = fixture->description;
	invalid.scratch.byte_offset = invalid.output.byte_offset;
	if (vision_prepare(&invalid, TEST_VISION_ARENA_BYTES, &fixture->kernels,
			   fixture->launches, VISION_MAXIMUM_LAUNCHES,
			   &count) != STATUS_INVALID_ARGUMENT ||
	    count != SCALAR_SENTINEL) {
		return false;
	}
	invalid = fixture->description;
	invalid.patches.byte_count--;
	if (vision_prepare(&invalid, TEST_VISION_ARENA_BYTES, &fixture->kernels,
			   fixture->launches, VISION_MAXIMUM_LAUNCHES,
			   &count) != STATUS_INVALID_ARGUMENT ||
	    count != SCALAR_SENTINEL) {
		return false;
	}
	if (vision_prepare(&fixture->description, TEST_VISION_ARENA_BYTES,
			   &fixture->kernels, fixture->launches,
			   IMPOSSIBLE_CAPACITY,
			   &count) != STATUS_INVALID_ARGUMENT ||
	    count != SCALAR_SENTINEL) {
		return false;
	}
	struct vision_kernels invalid_kernels = fixture->kernels;
	invalid_kernels.arithmetic_class = 0;
	if (vision_prepare(&fixture->description, TEST_VISION_ARENA_BYTES,
			   &invalid_kernels, fixture->launches,
			   VISION_MAXIMUM_LAUNCHES,
			   &count) != STATUS_INVALID_ARGUMENT ||
	    count != SCALAR_SENTINEL) {
		return false;
	}
	struct vision_scratch scratch;
	const struct vision_config official = vision_official_config();
	return vision_scratch_plan(&official, VISION_MAXIMUM_PATCHES,
				   VISION_MAXIMUM_PATCHES, 0,
				   &scratch) == STATUS_INVALID_ARGUMENT &&
	       vision_scratch_plan(&official, 1, 1,
				   UINT64_MAX - VISION_ALIGNMENT + 1u,
				   &scratch) == STATUS_INVALID_ARGUMENT;
}

static bool trace_case(struct test_vision_fixture *fixture, bool math_backend)
{
	for (uint32_t pass = 0; pass < fixture->launch_count; pass++) {
		const struct kernel_launch *const launch =
			&fixture->launches[pass];
		struct vision_arguments record;
		memcpy(&record, launch->argument, sizeof(record));
		if (!(math_backend ? test_vision_math_reference(
					     launch, 1, fixture->arena) :
				     test_vision_reference(launch, 1,
							   fixture->arena))) {
			return false;
		}
		const uint32_t elements = record.rows * record.output_width *
					  (launch->kernel == VISION_ROPE ?
						   VISION_QKV_COMPONENTS :
						   1u);
		const uint16_t *const words = test_vision_const_at(
			fixture->arena, record.destination_offset);
		(void)printf("TRACE %u %u", fixture->case_index, pass);
		for (uint32_t index = 0; index < elements; index++) {
			(void)printf(" %04x", words[index]);
		}
		(void)putchar('\n');
	}
	return true;
}

int main(int argument_count, char **arguments)
{
	const bool trace = argument_count == TRACE_ARGUMENT_COUNT &&
			   (strcmp(arguments[1], "--trace") == 0 ||
			    strcmp(arguments[1], "--trace-math") == 0);
	const bool math_trace = trace &&
				strcmp(arguments[1], "--trace-math") == 0;
	if (argument_count != 1 && !trace) {
		return EXIT_FAILURE;
	}
	for (uint32_t case_index = 0; case_index < TEST_VISION_CASES;
	     case_index++) {
		struct test_vision_fixture fixture;
		if (!test_vision_fixture_create(case_index, &fixture)) {
			return EXIT_FAILURE;
		}
		if (trace) {
			const bool passed = trace_case(&fixture, math_trace);
			test_vision_fixture_destroy(&fixture);
			if (!passed) {
				return EXIT_FAILURE;
			}
			continue;
		}
		bool passed = fixture.launch_count ==
				      EXPECTED_SMALL_LAYERS *
						      VISION_PASSES_PER_LAYER +
					      VISION_FIXED_PASSES &&
			      refusals(&fixture);
		/* Restore the accepted table after refusal probes before running the independent oracle. */
		passed = passed &&
			 vision_prepare(&fixture.description,
					TEST_VISION_ARENA_BYTES,
					&fixture.kernels, fixture.launches,
					VISION_MAXIMUM_LAUNCHES,
					&fixture.launch_count) == STATUS_OK;
		passed = passed &&
			 test_vision_reference(fixture.launches,
					       fixture.launch_count,
					       fixture.arena) &&
			 test_vision_fixture_golden(&fixture, fixture.arena);
		uint16_t *const output = test_vision_at(
			fixture.arena, fixture.description.output.byte_offset);
		output[0] ^= CHANGED_WORD;
		passed = passed &&
			 !test_vision_fixture_golden(&fixture, fixture.arena);
		test_vision_fixture_destroy(&fixture);
		if (passed) {
			passed = test_vision_fixture_create(case_index,
							    &fixture);
			if (passed) {
				passed = test_vision_math_reference(
						 fixture.launches,
						 fixture.launch_count,
						 fixture.arena) &&
					 test_vision_fixture_math_golden(
						 &fixture, fixture.arena);
				test_vision_fixture_destroy(&fixture);
			}
		}
		if (!passed) {
			(void)printf("vision host case %u failed\n",
				     case_index);
			return EXIT_FAILURE;
		}
	}
	(void)puts(
		"vision: four complete BF16 GPU Flash oracle cases, MATH controls and refusal boundaries passed");
	return EXIT_SUCCESS;
}
