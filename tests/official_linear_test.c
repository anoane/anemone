/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host preparation only: ABI, ownership geometry and refusal atomicity. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/official_linear.h"

#define TEST_INPUTS 128u
#define TEST_OUTPUTS 64u
#define TEST_BITS_PER_WEIGHT 8u
#define TEST_FIRST_OFFSET 4096u
#define TEST_ARENA_BYTES (128ull << 20u)
#define TEST_HANDLE 17u
#define TEST_NATIVE_HANDLE 18u
#define TEST_WOA_HANDLE 19u
#define TEST_ALIGNMENT 16u
#define TEST_UNTOUCHED 0xa5
#define TEST_DEPTH 2u
#define TEST_HALF_TILE_OUTPUTS 16u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct linear_range next_range(uint64_t *cursor, uint64_t bytes)
{
	const struct linear_range range = { *cursor, bytes };
	*cursor = (*cursor + bytes + TEST_ALIGNMENT - 1) / TEST_ALIGNMENT *
		  TEST_ALIGNMENT;
	return range;
}

static struct linear_matrix matrix_of(uint64_t *cursor, uint32_t outputs,
				      uint32_t inputs)
{
	const uint64_t bytes =
		(uint64_t)outputs * inputs +
		(uint64_t)((outputs + LINEAR_VALUES_PER_BLOCK - 1) /
			   LINEAR_VALUES_PER_BLOCK) *
			(inputs / LINEAR_VALUES_PER_BLOCK);
	return (struct linear_matrix){
		.stored = next_range(cursor, bytes),
		.shape = { outputs, inputs, TEST_BITS_PER_WEIGHT, 0 },
		.weight_format = WEIGHT_FORMAT_E4M3_SCALED,
		.slice_count = 1
	};
}

static struct linear_trunk_quantized_description
quantized_of(uint32_t rows, uint32_t matrices, uint32_t outputs)
{
	uint64_t cursor = TEST_FIRST_OFFSET;
	struct linear_trunk_quantized_description description = {
		.matrix_count = matrices, .row_count = rows
	};
	for (uint32_t matrix = 0; matrix < matrices; matrix++) {
		description.matrices[matrix] =
			matrix_of(&cursor, outputs, TEST_INPUTS);
		description.outputs[matrix] = next_range(
			&cursor, (uint64_t)rows * outputs * sizeof(uint16_t));
	}
	description.codes = next_range(&cursor, (uint64_t)rows * TEST_INPUTS);
	description.exponents =
		next_range(&cursor, (uint64_t)rows * TEST_INPUTS /
					    LINEAR_VALUES_PER_BLOCK);
	description.operands = next_range(
		&cursor, (uint64_t)rows * TEST_INPUTS * sizeof(uint16_t));
	return description;
}

static void
refused(const struct linear_trunk_quantized_description *description,
	uint64_t arena_bytes, const struct official_linear_kernels *kernels,
	enum official_linear_form form)
{
	struct kernel_launch launch, before;
	memset(&launch, TEST_UNTOUCHED, sizeof(launch));
	before = launch;
	expect(official_linear_quantized_prepare(description, arena_bytes,
						 kernels, form, &launch) ==
		       STATUS_INVALID_ARGUMENT,
	       "invalid projection refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal preserves the caller launch");
}

static void quantized_cases(void)
{
	const struct official_linear_kernels kernels = { TEST_HANDLE,
							 TEST_NATIVE_HANDLE,
							 TEST_WOA_HANDLE };
	for (uint32_t rows = 1; rows <= LINEAR_MAXIMUM_ROWS; rows++)
		for (uint32_t matrices = 1; matrices <= LINEAR_MERGED_MATRICES;
		     matrices++) {
			const struct linear_trunk_quantized_description
				description = quantized_of(rows, matrices,
							   TEST_OUTPUTS);
			for (uint32_t native = 0; native <= 1; native++) {
				struct kernel_launch launch = { 0 };
				expect(official_linear_quantized_prepare(
					       &description, TEST_ARENA_BYTES,
					       &kernels,
					       native ?
						       OFFICIAL_LINEAR_NATIVE :
						       OFFICIAL_LINEAR_PORTABLE,
					       &launch) == STATUS_OK,
				       "rows and merged matrices admitted");
				expect(launch.kernel ==
						       (native ?
								TEST_NATIVE_HANDLE :
								TEST_HANDLE) &&
					       launch.grid[0] ==
						       matrices * TEST_OUTPUTS /
							       (native ?
									OFFICIAL_LINEAR_NATIVE_OUTPUTS :
									OFFICIAL_LINEAR_PORTABLE_OUTPUTS) &&
					       launch.grid[1] ==
						       (native ? 1 : rows) &&
					       launch.grid[TEST_DEPTH] == 1 &&
					       launch.block[0] ==
						       OFFICIAL_LINEAR_THREADS &&
					       launch.block[1] == 1 &&
					       launch.block[TEST_DEPTH] == 1 &&
					       launch.dynamic_shared_byte_count ==
						       0 &&
					       launch.argument_byte_count ==
						       TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES,
				       "portable and native geometry retains the 72-byte record");
				struct trunk_linear_quantized_arguments
					arguments;
				memcpy(&arguments, launch.argument,
				       sizeof(arguments));
				expect(arguments.row_count == rows &&
					       arguments.input_size ==
						       TEST_INPUTS &&
					       arguments.weights_offset[0] ==
						       description.matrices[0]
							       .stored
							       .byte_offset &&
					       arguments.codes_offset ==
						       description.codes
							       .byte_offset &&
					       arguments.output_offset[0] ==
						       description.outputs[0]
							       .byte_offset &&
					       arguments.output_size[1] ==
						       (matrices == 1 ?
								0 :
								TEST_OUTPUTS),
				       "record contains validated packed inputs and BF16 outputs");
			}
		}
	struct linear_trunk_quantized_description description =
		quantized_of(1, 1, TEST_OUTPUTS);
	refused(NULL, TEST_ARENA_BYTES, &kernels, OFFICIAL_LINEAR_PORTABLE);
	refused(&description, TEST_ARENA_BYTES, NULL, OFFICIAL_LINEAR_PORTABLE);
	refused(&description, 1, &kernels, OFFICIAL_LINEAR_PORTABLE);
	description.operands.byte_count--;
	refused(&description, TEST_ARENA_BYTES, &kernels,
		OFFICIAL_LINEAR_PORTABLE);
	description = quantized_of(1, 1, TEST_OUTPUTS);
	description.outputs[0].byte_offset = description.codes.byte_offset;
	refused(&description, TEST_ARENA_BYTES, &kernels,
		OFFICIAL_LINEAR_PORTABLE);
	description = quantized_of(LINEAR_MAXIMUM_ROWS + 1, 1, TEST_OUTPUTS);
	refused(&description, TEST_ARENA_BYTES, &kernels,
		OFFICIAL_LINEAR_PORTABLE);
	description = quantized_of(1, 1, TEST_HALF_TILE_OUTPUTS);
	refused(&description, TEST_ARENA_BYTES, &kernels,
		OFFICIAL_LINEAR_NATIVE);
	struct kernel_launch half_tile_launch = { 0 };
	expect(official_linear_quantized_prepare(
		       &description, TEST_ARENA_BYTES, &kernels,
		       OFFICIAL_LINEAR_PORTABLE,
		       &half_tile_launch) == STATUS_OK,
	       "portable form retains the pack's half output tile");
	description = quantized_of(1, 1, TEST_OUTPUTS);
	struct official_linear_kernels portable_only = kernels;
	portable_only.quantized_native = BACKEND_NO_KERNEL;
	refused(&description, TEST_ARENA_BYTES, &portable_only,
		OFFICIAL_LINEAR_NATIVE);
}

static void woa_cases(void)
{
	const struct official_linear_kernels kernels = { TEST_HANDLE,
							 TEST_NATIVE_HANDLE,
							 TEST_WOA_HANDLE };
	uint64_t cursor = TEST_FIRST_OFFSET;
	struct linear_trunk_bfloat16_description description = {
		.matrix = matrix_of(&cursor,
				    OFFICIAL_LINEAR_WOA_GROUPS *
					    OFFICIAL_LINEAR_WOA_COLUMNS,
				    OFFICIAL_LINEAR_WOA_INNER),
		.row_count = 1,
		.segment_count = OFFICIAL_LINEAR_WOA_GROUPS
	};
	description.values = next_range(
		&cursor, (uint64_t)LINEAR_MAXIMUM_ROWS *
				 OFFICIAL_LINEAR_WOA_GROUPS *
				 OFFICIAL_LINEAR_WOA_INNER * sizeof(uint16_t));
	description.output = next_range(
		&cursor,
		(uint64_t)LINEAR_MAXIMUM_ROWS * OFFICIAL_LINEAR_WOA_GROUPS *
			OFFICIAL_LINEAR_WOA_COLUMNS * sizeof(uint16_t));
	for (uint32_t rows = 1; rows <= LINEAR_MAXIMUM_ROWS; rows++) {
		description.row_count = rows;
		struct kernel_launch launch = { 0 };
		expect(official_linear_woa_prepare(&description,
						   TEST_ARENA_BYTES, &kernels,
						   &launch) == STATUS_OK,
		       "grouped BF16 form admitted");
		expect(launch.kernel == TEST_WOA_HANDLE &&
			       launch.grid[0] ==
				       OFFICIAL_LINEAR_WOA_COLUMNS /
					       OFFICIAL_LINEAR_WOA_OUTPUTS &&
			       launch.grid[1] == rows &&
			       launch.grid[TEST_DEPTH] ==
				       OFFICIAL_LINEAR_WOA_GROUPS &&
			       launch.block[0] == OFFICIAL_LINEAR_THREADS &&
			       launch.argument_byte_count ==
				       TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES,
		       "WoA retains its 40-byte record and eight independent groups");
	}
	struct kernel_launch launch, before;
	memset(&launch, TEST_UNTOUCHED, sizeof(launch));
	before = launch;
	description.segment_count = 1;
	expect(official_linear_woa_prepare(&description, TEST_ARENA_BYTES,
					   &kernels, &launch) ==
			       STATUS_INVALID_ARGUMENT &&
		       memcmp(&launch, &before, sizeof(launch)) == 0,
	       "a different grouped contract is refused without publishing a launch");
}

int main(void)
{
	quantized_cases();
	woa_cases();
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
