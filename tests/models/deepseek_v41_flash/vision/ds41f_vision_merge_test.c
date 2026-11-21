/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host-only learned delimiter layout, exact replication, immutable inputs and refusal boundaries. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds41f_vision_merge_fixture.h"
#include "ds41f_vision_reference.h"

static bool untouched(const void *storage)
{
	const unsigned char *const bytes = storage;
	for (size_t index = 0; index < sizeof(struct kernel_launch); index++) {
		if (bytes[index] != DS41F_TEST_VISION_MERGE_SENTINEL) {
			return false;
		}
	}
	return true;
}

static bool refusals(const struct ds41f_test_vision_merge_fixture *fixture,
		     const struct ds41f_vision_kernels *kernels)
{
	struct kernel_launch launch;
	memset(&launch, DS41F_TEST_VISION_MERGE_SENTINEL, sizeof(launch));
	if (ds41f_vision_merge_prepare(
		    &fixture->small_weights, &fixture->small_encoded,
		    &fixture->small_output, DS41F_TEST_VISION_MERGE_SMALL_WIDTH,
		    DS41F_TEST_VISION_MERGE_SMALL_HEIGHT,
		    DS41F_TEST_VISION_MERGE_SMALL_COLUMNS, 0,
		    DS41F_TEST_VISION_MERGE_SMALL_ARENA_BYTES, kernels,
		    &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	struct linear_range short_output = fixture->small_output;
	short_output.byte_count--;
	if (ds41f_vision_merge_prepare(
		    &fixture->small_weights, &fixture->small_encoded,
		    &short_output, DS41F_TEST_VISION_MERGE_SMALL_WIDTH,
		    DS41F_TEST_VISION_MERGE_SMALL_HEIGHT,
		    DS41F_TEST_VISION_MERGE_SMALL_COLUMNS, 1,
		    DS41F_TEST_VISION_MERGE_SMALL_ARENA_BYTES, kernels,
		    &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	struct linear_range overlap = fixture->small_output;
	overlap.byte_offset = fixture->small_encoded.byte_offset;
	if (ds41f_vision_merge_prepare(
		    &fixture->small_weights, &fixture->small_encoded, &overlap,
		    DS41F_TEST_VISION_MERGE_SMALL_WIDTH,
		    DS41F_TEST_VISION_MERGE_SMALL_HEIGHT,
		    DS41F_TEST_VISION_MERGE_SMALL_COLUMNS, 1,
		    DS41F_TEST_VISION_MERGE_SMALL_ARENA_BYTES, kernels,
		    &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	struct ds41f_test_vision_merge_fixture model;
	if (!ds41f_test_vision_merge_fixture_prepare(true, kernels, &model)) {
		return false;
	}
	const struct linear_range empty = { 0 };
	if (ds41f_vision_step_merge_prepare(
		    model.embeddings, empty, model.step, model.streams,
		    DS41F_TEST_VISION_MERGE_STEP_ROWS, 1,
		    DS41F_TEST_VISION_MERGE_CARD_ARENA_BYTES, kernels,
		    &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	short_output = model.streams;
	short_output.byte_count--;
	if (ds41f_vision_step_merge_prepare(
		    model.embeddings, model.metadata, model.step, short_output,
		    DS41F_TEST_VISION_MERGE_STEP_ROWS, 1,
		    DS41F_TEST_VISION_MERGE_CARD_ARENA_BYTES, kernels,
		    &launch) != STATUS_INVALID_ARGUMENT ||
	    !untouched(&launch)) {
		return false;
	}
	return true;
}

int main(void)
{
	struct ds41f_vision_kernels kernels = {
		.arithmetic_class = ARITHMETIC_CLASS_SM_80
	};
	for (uint32_t operation = 0; operation < DS41F_VISION_OPERATIONS;
	     operation++) {
		kernels.operations[operation] = operation;
	}
	struct ds41f_test_vision_merge_fixture fixture;
	unsigned char *const arena =
		malloc(DS41F_TEST_VISION_MERGE_SMALL_ARENA_BYTES);
	unsigned char *const initial =
		malloc(DS41F_TEST_VISION_MERGE_SMALL_ARENA_BYTES);
	bool passed = arena != NULL && initial != NULL &&
		      ds41f_test_vision_merge_fixture_prepare(false, &kernels,
							      &fixture);
	if (passed) {
		ds41f_test_vision_merge_fixture_fill(&fixture, arena);
		memcpy(initial, arena,
		       DS41F_TEST_VISION_MERGE_SMALL_ARENA_BYTES);
		passed = refusals(&fixture, &kernels) &&
			 ds41f_test_vision_reference(fixture.launches,
						     fixture.launch_count,
						     arena) &&
			 ds41f_test_vision_merge_fixture_matches(
				 &fixture, arena, initial);
		arena[fixture.small_output.byte_offset] ^= 1u;
		passed = passed && !ds41f_test_vision_merge_fixture_matches(
					   &fixture, arena, initial);
	}
	free(initial);
	free(arena);
	(void)printf(
		"vision learned span layout, copies, guards and refusals: %s\n",
		passed ? "passed" : "failed");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
