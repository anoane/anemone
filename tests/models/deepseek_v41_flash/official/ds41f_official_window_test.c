/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "models/deepseek_v41_flash/official/ds41f_official_window.h"

#define ARENA_BYTES (1ull << 20u)
#define INPUT_OFFSET 4096u
#define WEIGHT_OFFSET (1ull << 16u)
#define ROTATION_OFFSET (2ull << 16u)
#define OUTPUT_OFFSET (3ull << 16u)
#define STATISTICS_OFFSET (4ull << 16u)
#define HANDLE 17u
#define UNTOUCHED_BYTE 0xA5
#define SOURCE_VECTOR 4u
#define SOURCE_EIGHT_ROW_DIVISOR 8u

static int failures;
static void expect(bool condition, const char *what)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", what);
	}
}

static struct ds41f_official_window_description description_of(uint32_t rows)
{
	const uint64_t row_bytes = (uint64_t)DS41F_HEAD_DIMS * sizeof(uint16_t);
	return (struct ds41f_official_window_description){
		.norm = { .input = { INPUT_OFFSET, rows * row_bytes },
			  .weight = { WEIGHT_OFFSET, row_bytes },
			  .output = { OUTPUT_OFFSET, rows * row_bytes },
			  .statistics = { STATISTICS_OFFSET,
					  (uint64_t)rows *
						  DS41F_OFFICIAL_NORM_STATISTICS *
						  sizeof(float) },
			  .width = DS41F_HEAD_DIMS,
			  .row_count = rows },
		.rotation = { ROTATION_OFFSET,
			      (uint64_t)rows *
				      (uint64_t)DS41F_OFFICIAL_ROPE_WIDTH *
				      sizeof(float) }
	};
}

static void refused(const struct ds41f_official_window_description *description,
		    uint32_t reference_rows, uint64_t arena, uint32_t kernel)
{
	struct kernel_launch launch, before;
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	before = launch;
	expect(ds41f_official_window_prepare(description, reference_rows, arena,
					     kernel, &launch) ==
		       STATUS_INVALID_ARGUMENT,
	       "invalid composition refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal preserves output record");
}

int main(void)
{
	static const uint32_t admitted[] = { 1, 3, 4, 8 };
	for (uint32_t actual = 0;
	     actual < sizeof(admitted) / sizeof(admitted[0]); actual++) {
		for (uint32_t reference = 0;
		     reference < sizeof(admitted) / sizeof(admitted[0]);
		     reference++) {
			const uint32_t rows = admitted[actual],
				       shape = admitted[reference];
			const struct ds41f_official_window_description
				description = description_of(rows);
			struct kernel_launch launch;
			const result_status status =
				ds41f_official_window_prepare(&description,
							      shape,
							      ARENA_BYTES,
							      HANDLE, &launch);
			expect(status == STATUS_OK,
			       "crossed actual/reference shape admitted");
			if (status == STATUS_OK) {
				const uint32_t source_owners =
					shape == LINEAR_MAXIMUM_ROWS ?
						DS41F_HEAD_DIMS /
							SOURCE_EIGHT_ROW_DIVISOR :
						DS41F_HEAD_DIMS / SOURCE_VECTOR;
				expect(launch.kernel == HANDLE &&
					       launch.grid[0] == 1 &&
					       launch.grid[1] == rows &&
					       launch.block[0] == source_owners,
				       "actual grid and recorded reference norm owners stay separate");
				struct ds41f_official_window_arguments arguments;
				memcpy(&arguments, launch.argument,
				       sizeof(arguments));
				expect(arguments.rows == rows &&
					       arguments.reserved_zero == 0 &&
					       arguments.output_offset ==
						       OUTPUT_OFFSET &&
					       arguments.rotation_offset ==
						       ROTATION_OFFSET,
				       "record retains actual extent and independent output/coefficient ranges");
			}
		}
	}
	struct ds41f_official_window_description description =
		description_of(1);
	refused(NULL, 1, ARENA_BYTES, HANDLE);
	refused(&description, 0, ARENA_BYTES, HANDLE);
	refused(&description, 1, 1, HANDLE);
	refused(&description, 1, ARENA_BYTES, BACKEND_NO_KERNEL);
	description.rotation = description.norm.input;
	refused(&description, 1, ARENA_BYTES, HANDLE);
	description = description_of(1);
	description.rotation = description.norm.statistics;
	refused(&description, 1, ARENA_BYTES, HANDLE);
	description = description_of(1);
	description.rotation.byte_count--;
	refused(&description, 1, ARENA_BYTES, HANDLE);
	description = description_of(1);
	description.rotation.byte_offset++;
	refused(&description, 1, ARENA_BYTES, HANDLE);
	description = description_of(1);
	description.norm.output = description.norm.input;
	refused(&description, 1, ARENA_BYTES, HANDLE);
	description = description_of(1);
	description.norm.width = DS41F_QUERY_RANK;
	refused(&description, 1, ARENA_BYTES, HANDLE);
	description = description_of(0);
	refused(&description, 1, ARENA_BYTES, HANDLE);
	(void)printf("official_window_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
