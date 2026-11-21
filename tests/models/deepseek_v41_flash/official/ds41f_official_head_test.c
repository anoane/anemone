/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "models/deepseek_v41_flash/official/ds41f_official_head.h"

#define ARENA_BYTES (4ull << 30u)
#define WEIGHTS_OFFSET 4096u
#define ROWS_OFFSET (3ull << 30u)
#define OUTPUT_OFFSET (7ull << 29u)
#define HANDLE 17u
#define UNTOUCHED 0xA5
#define REFUSED_ROWS 2u
#define VOCABULARY_SHARDS 4u
#define DEPTH_DIMENSION 2u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct ds41f_official_head_description description_of(uint32_t rows,
							     uint32_t outputs)
{
	return (struct ds41f_official_head_description){
		.weights = { WEIGHTS_OFFSET, (uint64_t)outputs * DS41F_HIDDEN *
						     sizeof(uint16_t) },
		.rows = { ROWS_OFFSET,
			  (uint64_t)rows * DS41F_HIDDEN * sizeof(uint16_t) },
		.output = { OUTPUT_OFFSET,
			    (uint64_t)rows * outputs * sizeof(float) },
		.output_size = outputs,
		.row_count = rows
	};
}

static void refused(const struct ds41f_official_head_description *description,
		    uint64_t arena, uint32_t kernel)
{
	struct kernel_launch launch, before;
	memset(&launch, UNTOUCHED, sizeof(launch));
	before = launch;
	expect(ds41f_official_head_prepare(description, arena, kernel,
					   &launch) == STATUS_INVALID_ARGUMENT,
	       "invalid head descriptor refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal leaves output record untouched");
	expect(ds41f_official_head_reused_prepare(description, arena, kernel,
						  &launch) ==
		       STATUS_INVALID_ARGUMENT,
	       "invalid reused head descriptor refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "reused refusal leaves output record untouched");
}

int main(void)
{
	static const uint32_t row_counts[] = { 1, 3, 4, 8 };
	static const uint32_t output_counts[] = { DS41F_OFFICIAL_HEAD_OUTPUTS,
						  DS41F_VOCABULARY_VALUE /
							  VOCABULARY_SHARDS,
						  DS41F_VOCABULARY_VALUE };
	for (uint32_t r = 0; r < sizeof(row_counts) / sizeof(row_counts[0]);
	     r++) {
		for (uint32_t n = 0;
		     n < sizeof(output_counts) / sizeof(output_counts[0]);
		     n++) {
			struct ds41f_official_head_description description =
				description_of(row_counts[r], output_counts[n]);
			struct kernel_launch launch = { 0 };
			expect(ds41f_official_head_prepare(
				       &description, ARENA_BYTES, HANDLE,
				       &launch) == STATUS_OK,
			       "independent rows and vocabulary cuts admitted");
			expect(launch.kernel == HANDLE &&
				       launch.grid[0] ==
					       output_counts[n] /
						       DS41F_OFFICIAL_HEAD_OUTPUTS &&
				       launch.grid[1] == row_counts[r] &&
				       launch.grid[DEPTH_DIMENSION] == 1 &&
				       launch.block[0] ==
					       DS41F_OFFICIAL_HEAD_OWNERS &&
				       launch.block[1] ==
					       DS41F_OFFICIAL_HEAD_OUTPUTS &&
				       launch.block[DEPTH_DIMENSION] == 1,
			       "canonical16-by4 ownership independent of row/cut width");
			struct ds41f_official_head_arguments arguments;
			memcpy(&arguments, launch.argument, sizeof(arguments));
			expect(arguments.output_size == output_counts[n] &&
				       arguments.row_count == row_counts[r] &&
				       arguments.weights_offset ==
					       WEIGHTS_OFFSET &&
				       arguments.rows_offset == ROWS_OFFSET &&
				       arguments.output_offset == OUTPUT_OFFSET,
			       "record retains caller-owned ranges and actual extents");
			struct kernel_launch reused;
			expect(ds41f_official_head_reused_prepare(
				       &description, ARENA_BYTES, HANDLE,
				       &reused) == STATUS_OK &&
				       reused.grid[1] == 1 &&
				       reused.grid[0] == launch.grid[0] &&
				       memcmp(reused.argument, launch.argument,
					      sizeof(arguments)) == 0 &&
				       memcmp(reused.block, launch.block,
					      sizeof(launch.block)) == 0,
			       "explicit weight reuse retains every row's record/owners");
			description.weights.byte_count--;
			refused(&description, ARENA_BYTES, HANDLE);
		}
	}
	struct ds41f_official_head_description description =
		description_of(1, DS41F_VOCABULARY_VALUE);
	refused(NULL, ARENA_BYTES, HANDLE);
	refused(&description, 1, HANDLE);
	refused(&description, ARENA_BYTES, BACKEND_NO_KERNEL);
	description.rows = description.weights;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(1, DS41F_VOCABULARY_VALUE);
	description.output.byte_offset = ROWS_OFFSET;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(1, DS41F_VOCABULARY_VALUE);
	description.rows.byte_offset++;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(1, DS41F_VOCABULARY_VALUE);
	description.output.byte_count--;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(REFUSED_ROWS, DS41F_VOCABULARY_VALUE);
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(1, DS41F_VOCABULARY_VALUE - 1u);
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(0, DS41F_VOCABULARY_VALUE);
	refused(&description, ARENA_BYTES, HANDLE);
	(void)printf("official_head_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
