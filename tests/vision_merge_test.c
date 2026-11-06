/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host-only learned delimiter layout, exact replication, immutable inputs and refusal boundaries. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vision_merge_fixture.h"
#include "vision_reference.h"

static bool untouched(const void *storage)
{
	const unsigned char *const bytes = storage;
	for (size_t index = 0; index < sizeof(struct kernel_launch); index++) {
		if (bytes[index] != TEST_VISION_MERGE_SENTINEL) {
			return false;
		}
	}
	return true;
}

static bool refusals(const struct test_vision_merge_fixture *fixture,
		     const struct vision_kernels *kernels)
{
	struct kernel_launch launch;
	memset(&launch, TEST_VISION_MERGE_SENTINEL, sizeof(launch));
	if (vision_merge_prepare(
		    &fixture->small_weights, &fixture->small_encoded,
		    &fixture->small_output, TEST_VISION_MERGE_SMALL_WIDTH,
		    TEST_VISION_MERGE_SMALL_HEIGHT,
		    TEST_VISION_MERGE_SMALL_COLUMNS, 0,
		    TEST_VISION_MERGE_SMALL_ARENA_BYTES, kernels,
		    &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	struct linear_range short_output = fixture->small_output;
	short_output.byte_count--;
	if (vision_merge_prepare(&fixture->small_weights,
				 &fixture->small_encoded, &short_output,
				 TEST_VISION_MERGE_SMALL_WIDTH,
				 TEST_VISION_MERGE_SMALL_HEIGHT,
				 TEST_VISION_MERGE_SMALL_COLUMNS, 1,
				 TEST_VISION_MERGE_SMALL_ARENA_BYTES, kernels,
				 &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	struct linear_range overlap = fixture->small_output;
	overlap.byte_offset = fixture->small_encoded.byte_offset;
	if (vision_merge_prepare(&fixture->small_weights,
				 &fixture->small_encoded, &overlap,
				 TEST_VISION_MERGE_SMALL_WIDTH,
				 TEST_VISION_MERGE_SMALL_HEIGHT,
				 TEST_VISION_MERGE_SMALL_COLUMNS, 1,
				 TEST_VISION_MERGE_SMALL_ARENA_BYTES, kernels,
				 &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	struct test_vision_merge_fixture model;
	if (!test_vision_merge_fixture_prepare(true, kernels, &model)) {
		return false;
	}
	const struct linear_range empty = { 0 };
	if (vision_step_merge_prepare(
		    model.embeddings, empty, model.step, model.streams,
		    TEST_VISION_MERGE_STEP_ROWS, 1,
		    TEST_VISION_MERGE_CARD_ARENA_BYTES, kernels,
		    &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	short_output = model.streams;
	short_output.byte_count--;
	if (vision_step_merge_prepare(
		    model.embeddings, model.metadata, model.step, short_output,
		    TEST_VISION_MERGE_STEP_ROWS, 1,
		    TEST_VISION_MERGE_CARD_ARENA_BYTES, kernels,
		    &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	return true;
}

int main(void)
{
	struct vision_kernels kernels = { .arithmetic_class =
						  ARITHMETIC_CLASS_SM_80 };
	for (uint32_t operation = 0; operation < VISION_OPERATIONS;
	     operation++) {
		kernels.operations[operation] = operation;
	}
	struct test_vision_merge_fixture fixture;
	unsigned char *const arena =
		malloc(TEST_VISION_MERGE_SMALL_ARENA_BYTES);
	unsigned char *const initial =
		malloc(TEST_VISION_MERGE_SMALL_ARENA_BYTES);
	bool passed =
		arena != NULL && initial != NULL &&
		test_vision_merge_fixture_prepare(false, &kernels, &fixture);
	if (passed) {
		test_vision_merge_fixture_fill(&fixture, arena);
		memcpy(initial, arena, TEST_VISION_MERGE_SMALL_ARENA_BYTES);
		passed = refusals(&fixture, &kernels) &&
			 test_vision_reference(fixture.launches,
					       fixture.launch_count, arena) &&
			 test_vision_merge_fixture_matches(&fixture, arena,
							   initial);
		arena[fixture.small_output.byte_offset] ^= 1u;
		passed = passed && !test_vision_merge_fixture_matches(
					   &fixture, arena, initial);
	}
	free(initial);
	free(arena);
	(void)printf(
		"vision learned span layout, copies, guards and refusals: %s\n",
		passed ? "passed" : "failed");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
