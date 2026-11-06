/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/official_rope.h"

#define ARENA_BYTES (4ull << 20u)
#define INPUT_OFFSET 4096u
#define ROTATION_OFFSET (1ull << 20u)
#define OUTPUT_OFFSET (2ull << 20u)
#define HANDLE 17u
#define UNTOUCHED 0xA5
#define INVALID_INVERSE 2u

static int failures;
static void expect(bool condition, const char *message)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct official_rope_description
description_of(uint32_t width, uint32_t rows, uint32_t heads)
{
	const uint64_t bytes =
		(uint64_t)width * rows * heads * sizeof(uint16_t);
	return (struct official_rope_description){
		{ INPUT_OFFSET, bytes },
		{ ROTATION_OFFSET,
		  (uint64_t)rows * OFFICIAL_ROPE_WIDTH * sizeof(float) },
		{ OUTPUT_OFFSET, bytes },
		width,
		heads,
		rows,
		0
	};
}

static void refused(const struct official_rope_description *description,
		    uint64_t arena, uint32_t kernel)
{
	struct kernel_launch launch, before;
	memset(&launch, UNTOUCHED, sizeof(launch));
	before = launch;
	expect(official_rope_prepare(description, arena, kernel, &launch) ==
		       STATUS_INVALID_ARGUMENT,
	       "invalid descriptor refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal leaves output untouched");
}

int main(void)
{
	static const uint32_t widths[] = { OFFICIAL_ROPE_WIDTH,
					   MODEL_HEAD_DIMS };
	for (uint32_t index = 0; index < sizeof(widths) / sizeof(widths[0]);
	     index++) {
		for (uint32_t rows = 1; rows <= LINEAR_MAXIMUM_ROWS; rows++) {
			for (uint32_t heads = 1; heads <= MODEL_HEADS_VALUE;
			     heads++) {
				struct official_rope_description description =
					description_of(widths[index], rows,
						       heads);
				struct kernel_launch launch = { 0 };
				expect(official_rope_prepare(
					       &description, ARENA_BYTES,
					       HANDLE, &launch) == STATUS_OK,
				       "disjoint regions admitted");
				expect(launch.grid[0] == heads &&
					       launch.grid[1] == rows,
				       "grid follows actual heads and rows");
				description.inverse = 1;
				description.output = description.input;
				expect(official_rope_prepare(
					       &description, ARENA_BYTES,
					       HANDLE, &launch) == STATUS_OK,
				       "in-place inverse admitted");
			}
		}
	}
	struct official_rope_description description =
		description_of(MODEL_HEAD_DIMS, 1, 1);
	refused(NULL, ARENA_BYTES, HANDLE);
	refused(&description, 1, HANDLE);
	refused(&description, ARENA_BYTES, BACKEND_NO_KERNEL);
	description.output.byte_offset =
		INPUT_OFFSET + LINEAR_OFFSET_ALIGNMENT_BYTES;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(MODEL_HEAD_DIMS, 1, 1);
	description.rotation = description.input;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(MODEL_HEAD_DIMS, 1, 1);
	description.inverse = INVALID_INVERSE;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(MODEL_HEAD_DIMS, 1, 1);
	description.output = description.input;
	description.output.byte_count--;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(OFFICIAL_ROPE_WIDTH - 1u, 1, 1);
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(MODEL_HEAD_DIMS, 0, 1);
	refused(&description, ARENA_BYTES, HANDLE);
	description =
		description_of(MODEL_HEAD_DIMS, 1, MODEL_HEADS_VALUE + 1u);
	refused(&description, ARENA_BYTES, HANDLE);
	(void)printf("official_rope_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
