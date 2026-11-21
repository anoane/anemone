/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "models/deepseek_v41_flash/official/ds41f_official_router_projection.h"

#define ARENA_BYTES (32ull << 20u)
#define WEIGHTS_OFFSET 4096u
#define ROWS_OFFSET (8ull << 20u)
#define OUTPUT_OFFSET (16ull << 20u)
#define HANDLE 17u
#define UNTOUCHED 0xA5
#define REFUSED_ROWS 2u
#define DEPTH_DIMENSION 2u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct ds41f_official_router_projection_description
description_of(uint32_t rows)
{
	return (struct ds41f_official_router_projection_description){
		.weights = { WEIGHTS_OFFSET, (uint64_t)DS41F_EXPERTS *
						     DS41F_HIDDEN *
						     sizeof(uint16_t) },
		.rows = { ROWS_OFFSET,
			  (uint64_t)rows * DS41F_HIDDEN * sizeof(uint16_t) },
		.output = { OUTPUT_OFFSET,
			    (uint64_t)rows * DS41F_EXPERTS * sizeof(float) },
		.row_count = rows
	};
}

static void
refused(const struct ds41f_official_router_projection_description *description,
	uint64_t arena, uint32_t kernel)
{
	struct kernel_launch launch, before;
	memset(&launch, UNTOUCHED, sizeof(launch));
	before = launch;
	expect(ds41f_official_router_projection_prepare(description, arena,
							kernel, &launch) ==
		       STATUS_INVALID_ARGUMENT,
	       "invalid router product descriptor refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal preserves the caller launch");
}

int main(void)
{
	static const uint32_t rows[] = { 1, 3, 4, 8 };
	for (uint32_t r = 0; r < sizeof(rows) / sizeof(rows[0]); r++) {
		struct ds41f_official_router_projection_description description =
			description_of(rows[r]);
		struct kernel_launch launch = { 0 };
		expect(ds41f_official_router_projection_prepare(
			       &description, ARENA_BYTES, HANDLE, &launch) ==
			       STATUS_OK,
		       "independent actual rows admitted");
		expect(launch.kernel == HANDLE &&
			       launch.grid[0] ==
				       DS41F_EXPERTS /
					       DS41F_OFFICIAL_ROUTER_PROJECTION_OUTPUTS &&
			       launch.grid[1] == rows[r] &&
			       launch.grid[DEPTH_DIMENSION] == 1 &&
			       launch.block[0] ==
				       DS41F_OFFICIAL_ROUTER_PROJECTION_OWNERS &&
			       launch.block[1] ==
				       DS41F_OFFICIAL_ROUTER_PROJECTION_OUTPUTS &&
			       launch.block[DEPTH_DIMENSION] == 1,
		       "32-by4 ownership is independent of actual rows");
		struct ds41f_official_router_projection_arguments arguments;
		memcpy(&arguments, launch.argument, sizeof(arguments));
		expect(arguments.row_count == rows[r] &&
			       arguments.reserved_zero == 0 &&
			       arguments.weights_offset == WEIGHTS_OFFSET &&
			       arguments.rows_offset == ROWS_OFFSET &&
			       arguments.output_offset == OUTPUT_OFFSET,
		       "record retains caller ranges and explicit row extent");
		description.weights.byte_count--;
		refused(&description, ARENA_BYTES, HANDLE);
	}
	struct ds41f_official_router_projection_description description =
		description_of(1);
	refused(NULL, ARENA_BYTES, HANDLE);
	refused(&description, 1, HANDLE);
	refused(&description, ARENA_BYTES, BACKEND_NO_KERNEL);
	description.rows = description.weights;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(1);
	description.output.byte_offset = ROWS_OFFSET;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(1);
	description.rows.byte_offset++;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(1);
	description.output.byte_count--;
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(REFUSED_ROWS);
	refused(&description, ARENA_BYTES, HANDLE);
	description = description_of(0);
	refused(&description, ARENA_BYTES, HANDLE);
	(void)printf("official_router_projection_test: %d failures\n",
		     failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
