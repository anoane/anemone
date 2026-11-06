/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/official_hc_partitioned.h"

#define ARENA_BYTES (4ull << 20u)
#define WEIGHTS_OFFSET 4096u
#define ROWS_OFFSET (2ull << 20u)
#define OUTPUT_OFFSET (3ull << 20u)
#define PARTIALS_OFFSET (7ull << 19u)
#define DENSE_HANDLE 11u
#define DOT_HANDLE 12u
#define SPLIT_HANDLE 13u
#define PARTIAL_HANDLE 21u
#define REDUCE_HANDLE 22u
#define UNTOUCHED_BYTE 0xA5
#define WRONG_SCALE 2.0f

static int failures;
static struct dense_linear_kernels dense;
static const struct official_hc_kernels official = { DOT_HANDLE, SPLIT_HANDLE };
static const struct official_hc_partitioned_kernels partitioned = {
	PARTIAL_HANDLE, REDUCE_HANDLE
};

static void expect(bool condition, const char *what)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", what);
	}
}

static struct dense_linear_description description_of(uint32_t rows)
{
	return (struct dense_linear_description){
		.weights = { WEIGHTS_OFFSET, (uint64_t)MODEL_MIXES *
						     OFFICIAL_HC_INPUTS *
						     sizeof(float) },
		.rows = { ROWS_OFFSET, (uint64_t)rows * OFFICIAL_HC_INPUTS *
					       sizeof(uint16_t) },
		.output = { OUTPUT_OFFSET,
			    (uint64_t)rows * MODEL_MIXES * sizeof(float) },
		.output_size = MODEL_MIXES,
		.input_size = OFFICIAL_HC_INPUTS,
		.row_count = rows,
		.weight_type = DENSE_FLOAT32,
		.output_type = DENSE_FLOAT32,
		.flags = DENSE_ROW_SCALE,
		.output_scale = 1.0f,
		.epsilon = MODEL_NORM_EPSILON
	};
}

static struct linear_range partials_of(uint32_t rows)
{
	return (struct linear_range){ PARTIALS_OFFSET,
				      (uint64_t)rows * MODEL_MIXES *
					      OFFICIAL_HC_PARTITIONS *
					      sizeof(float) };
}

static void refused(const struct dense_linear_description *description,
		    const struct linear_range *partials,
		    const struct official_hc_partitioned_kernels *kernels,
		    uint64_t arena_bytes)
{
	struct kernel_launch launches[OFFICIAL_HC_PARTITIONED_LAUNCHES],
		before[OFFICIAL_HC_PARTITIONED_LAUNCHES];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memcpy(before, launches, sizeof(before));
	expect(official_hc_partitioned_prepare(
		       description, partials, arena_bytes, &dense, &official,
		       kernels, launches) == STATUS_INVALID_ARGUMENT,
	       "invalid partitioned projection refused");
	expect(memcmp(before, launches, sizeof(before)) == 0,
	       "refusal preserves both output records");
}

int main(void)
{
	for (uint32_t index = 0; index < DENSE_INSTANCES; index++)
		dense.products[index] = DENSE_HANDLE;
	static const uint32_t accepted[] = { 1, 3, 4, 8 };
	for (uint32_t index = 0; index < sizeof(accepted) / sizeof(accepted[0]);
	     index++) {
		const uint32_t rows = accepted[index];
		const struct dense_linear_description description =
			description_of(rows);
		struct linear_range partials = partials_of(rows);
		struct kernel_launch launches[OFFICIAL_HC_PARTITIONED_LAUNCHES];
		const result_status status = official_hc_partitioned_prepare(
			&description, &partials, ARENA_BYTES, &dense, &official,
			&partitioned, launches);
		expect(status == STATUS_OK,
		       "admitted partitioned shape prepares");
		if (status == STATUS_OK) {
			expect(launches[0].kernel == PARTIAL_HANDLE &&
				       launches[0].grid[0] == MODEL_MIXES &&
				       launches[0].grid[1] ==
					       OFFICIAL_HC_PARTITIONS &&
				       launches[0].grid[OFFICIAL_HC_ROW_GRID_DIMENSION] ==
					       rows &&
				       launches[0].block[0] ==
					       OFFICIAL_HC_PARTIAL_THREADS,
			       "first pass spans every output, partial and actual row");
			expect(launches[1].kernel == REDUCE_HANDLE &&
				       launches[1].grid[0] == 1 &&
				       launches[1].grid[1] == rows &&
				       launches[1].block[0] ==
					       OFFICIAL_HC_THREADS,
			       "second pass computes one F1 RMS per actual row");
			expect(memcmp(launches[0].argument,
				      launches[1].argument,
				      sizeof(struct official_hc_partitioned_arguments)) ==
				       0,
			       "both passes use the same validated offsets and extent");
		}
		partials.byte_count--;
		refused(&description, &partials, &partitioned, ARENA_BYTES);
	}
	struct dense_linear_description description = description_of(1);
	struct linear_range partials = partials_of(1);
	refused(NULL, &partials, &partitioned, ARENA_BYTES);
	refused(&description, NULL, &partitioned, ARENA_BYTES);
	refused(&description, &partials, NULL, ARENA_BYTES);
	refused(&description, &partials, &partitioned, 1);
	partials.byte_offset = ROWS_OFFSET;
	refused(&description, &partials, &partitioned, ARENA_BYTES);
	partials = partials_of(1);
	partials.byte_offset++;
	refused(&description, &partials, &partitioned, ARENA_BYTES);
	partials = partials_of(1);
	description.output_scale = WRONG_SCALE;
	refused(&description, &partials, &partitioned, ARENA_BYTES);
	description = description_of(0);
	refused(&description, &partials, &partitioned, ARENA_BYTES);
	description = description_of(1);
	struct official_hc_partitioned_kernels missing = partitioned;
	missing.reduce = BACKEND_NO_KERNEL;
	refused(&description, &partials, &missing, ARENA_BYTES);
	(void)printf("official_hc_partitioned_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
