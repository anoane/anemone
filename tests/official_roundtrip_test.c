/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/official_roundtrip.h"

#define ARENA_BYTES (4ull << 20u)
#define INPUT_OFFSET 4096u
#define OUTPUT_OFFSET (2ull << 20u)
#define HANDLE 17u
#define UNTOUCHED 0xA5

static int failures;
static void expect(bool condition, const char *message)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct official_roundtrip_description description_of(uint32_t width,
							    uint32_t rows)
{
	const uint64_t bytes = (uint64_t)width * rows * sizeof(uint16_t);
	return (struct official_roundtrip_description){
		{ INPUT_OFFSET, bytes }, { OUTPUT_OFFSET, bytes }, width, rows
	};
}

static void refused(const struct official_roundtrip_description *description,
		    uint64_t arena, uint32_t kernel)
{
	struct kernel_launch launch, before;
	memset(&launch, UNTOUCHED, sizeof(launch));
	before = launch;
	expect(official_roundtrip_prepare(description, arena, kernel,
					  &launch) == STATUS_INVALID_ARGUMENT,
	       "invalid descriptor refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal leaves output untouched");
}

int main(void)
{
	static const uint32_t widths[] = { OFFICIAL_ROUNDTRIP_BLOCK_VALUES,
					   MODEL_HEAD_DIMS, MODEL_QUERY_RANK,
					   OFFICIAL_ROUNDTRIP_MOST_WIDTH };
	for (uint32_t index = 0; index < sizeof(widths) / sizeof(widths[0]);
	     index++) {
		for (uint32_t rows = 1; rows <= LINEAR_MAXIMUM_ROWS; rows++) {
			struct official_roundtrip_description description =
				description_of(widths[index], rows);
			struct kernel_launch launch = { 0 };
			expect(official_roundtrip_prepare(&description,
							  ARENA_BYTES, HANDLE,
							  &launch) == STATUS_OK,
			       "disjoint ranges admitted");
			expect(launch.grid[1] == rows &&
				       launch.block[0] ==
					       OFFICIAL_ROUNDTRIP_THREADS,
			       "launch spans actual rows");
			description.output = description.input;
			expect(official_roundtrip_prepare(&description,
							  ARENA_BYTES, HANDLE,
							  &launch) == STATUS_OK,
			       "identical ranges admitted");
			description.output.byte_count--;
			refused(&description, ARENA_BYTES, HANDLE);
		}
	}
	struct official_roundtrip_description description =
		description_of(MODEL_HEAD_DIMS, 1);
	refused(&description, 1, HANDLE);
	refused(&description, ARENA_BYTES, BACKEND_NO_KERNEL);
	refused(NULL, ARENA_BYTES, HANDLE);
	description.input.byte_count--;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(MODEL_HEAD_DIMS, 1);
	description.output.byte_offset =
		INPUT_OFFSET + LINEAR_OFFSET_ALIGNMENT_BYTES;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(MODEL_HEAD_DIMS - 1u, 1);
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(OFFICIAL_ROUNDTRIP_MOST_WIDTH + 1u, 1);
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(MODEL_HEAD_DIMS, 0);
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(MODEL_HEAD_DIMS, LINEAR_MAXIMUM_ROWS + 1u);
	refused(&description, ARENA_BYTES, HANDLE);
	(void)printf("official_roundtrip_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
