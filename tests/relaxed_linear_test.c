/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * relaxed_linear_test: host-only admission and launch records of class 80's ordered/relaxed trunk products.
 * Fold boundaries are independent of the launch descriptions; refused descriptions leave every output byte
 * and the launch count untouched. The arena is a number: no matrix or device memory is allocated.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/relaxed_linear.h"

#define TEST_ROWS 64u
#define TEST_WIDTH 256u
#define E4M3_BITS_PER_WEIGHT 8u
#define TEST_SLICES 4u
#define TEST_SEGMENTS 2u
#define TEST_SPLITS 2u
#define TEST_MATRIX_BYTES 65600ull /* 256 * 256 codes, 8 * 8 scales */
#define FIRST_MATRIX_OFFSET 4096ull
#define SECOND_MATRIX_OFFSET 131072ull
#define OPERANDS_OFFSET 1048576ull
#define FIRST_OUTPUT_OFFSET 2097152ull
#define SECOND_OUTPUT_OFFSET 3145728ull
#define PARTIALS_OFFSET 4194304ull
#define ARENA_BYTES 8388608ull
#define UNTOUCHED_BYTE 0xA5
#define FIRST_HANDLE 17u
#define LARGE_HANDLE 19u
#define ORDERED_HANDLE 23u
#define ORDERED_LARGE_HANDLE 29u
#define BFLOAT16_HANDLE 31u
#define BFLOAT16_LARGE_HANDLE 37u
#define BFLOAT16_ORDERED_HANDLE 41u
#define BFLOAT16_ORDERED_LARGE_HANDLE 43u
#define DECODE_HANDLE 47u
#define BFLOAT16_DECODE_HANDLE 53u
#define REDUCE_HANDLE 59u
#define FLOAT32_BYTES 4u
#define TEST_CODE_TOP 122u
#define TEST_FOLD_REFERENCE 115
#define INVALID_ROWS_FORM 2u
#define INVALID_FLAG 2u
#define INVALID_MATRIX_COUNT 3u
#define INVALID_SLICES 3u
#define OVERSIZED_WEIGHT_OFFSET 4096ull
#define OVERSIZED_OPERANDS_OFFSET (1ull << 41u)
#define OVERSIZED_OUTPUT_OFFSET (1ull << 42u)
#define OVERSIZED_PARTIALS_OFFSET (1ull << 50u)
#define OVERSIZED_ROWS 1024u
#define STEP_OFFSET 6291456ull /* the step area: its block and TEST_ROWS rows */
#define STEP_BYTES (16ull + TEST_ROWS * 16ull)
#define CONE_LAYER 30u /* a decoder layer */
#define FRONT_LAYER 19u /* not one */
#define DECODE_ROWS 8u

static uint32_t failures;
static const struct relaxed_linear_kernels kernels = {
	.wide_quantized = { FIRST_HANDLE, LARGE_HANDLE },
	.ordered_wide_quantized = { ORDERED_HANDLE, ORDERED_LARGE_HANDLE },
	.wide_bfloat16 = { BFLOAT16_HANDLE, BFLOAT16_LARGE_HANDLE },
	.ordered_wide_bfloat16 = { BFLOAT16_ORDERED_HANDLE,
				   BFLOAT16_ORDERED_LARGE_HANDLE },
	.decode_quantized = DECODE_HANDLE,
	.decode_bfloat16 = BFLOAT16_DECODE_HANDLE,
	.reduce = REDUCE_HANDLE,
};

static void expect(bool condition, const char *what)
{
	if (!condition) {
		printf("relaxed_linear_test: %s\n", what);
		failures++;
	}
}

static struct linear_matrix matrix_at(uint64_t offset)
{
	const struct linear_matrix matrix = {
		.stored = { offset, TEST_MATRIX_BYTES },
		.shape = { TEST_WIDTH, TEST_WIDTH, E4M3_BITS_PER_WEIGHT, 0 },
		.weight_format = WEIGHT_FORMAT_E4M3_SCALED,
		.slice_count = TEST_SLICES,
	};
	return matrix;
}

static struct relaxed_trunk_quantized_description quantized_description(void)
{
	const uint64_t values_bytes =
		linear_values_byte_count(TEST_ROWS, TEST_WIDTH);
	const struct relaxed_trunk_quantized_description description = {
		.matrices = { matrix_at(FIRST_MATRIX_OFFSET),
			      matrix_at(SECOND_MATRIX_OFFSET) },
		.operands = { OPERANDS_OFFSET, values_bytes },
		.outputs = { { FIRST_OUTPUT_OFFSET, values_bytes },
			     { SECOND_OUTPUT_OFFSET, values_bytes } },
		.partials = { PARTIALS_OFFSET, ARENA_BYTES - PARTIALS_OFFSET },
		.matrix_count = LINEAR_MERGED_MATRICES,
		.row_count = TEST_ROWS,
		.k_split_count = 1,
	};
	return description;
}

static struct relaxed_trunk_bfloat16_description bfloat16_description(void)
{
	const struct relaxed_trunk_bfloat16_description description = {
		.matrix = matrix_at(FIRST_MATRIX_OFFSET),
		.values = { OPERANDS_OFFSET,
			    linear_values_byte_count(TEST_ROWS,
						     (uint64_t)TEST_SEGMENTS *
							     TEST_WIDTH) },
		.output = { FIRST_OUTPUT_OFFSET,
			    linear_values_byte_count(TEST_ROWS, TEST_WIDTH) },
		.partials = { PARTIALS_OFFSET, ARENA_BYTES - PARTIALS_OFFSET },
		.row_count = TEST_ROWS,
		.segment_count = TEST_SEGMENTS,
		.k_split_count = 1,
		.fold_reference = TEST_FOLD_REFERENCE,
	};
	return description;
}

static void expect_quantized_refusal(
	const struct relaxed_trunk_quantized_description *description,
	const char *what)
{
	struct kernel_launch actual[RELAXED_MOST_LAUNCHES],
		untouched[RELAXED_MOST_LAUNCHES];
	uint32_t count = UINT32_MAX;
	memset(actual, UNTOUCHED_BYTE, sizeof(actual));
	memcpy(untouched, actual, sizeof(actual));
	expect(relaxed_trunk_quantized_prepare(description, ARENA_BYTES,
					       &kernels, actual, &count) ==
		       STATUS_INVALID_ARGUMENT,
	       what);
	expect(count == UINT32_MAX &&
		       memcmp(actual, untouched, sizeof(actual)) == 0,
	       "quantized refusal leaves launches and count untouched");
}

static void expect_bfloat16_refusal(
	const struct relaxed_trunk_bfloat16_description *description,
	const char *what)
{
	struct kernel_launch actual[RELAXED_MOST_LAUNCHES],
		untouched[RELAXED_MOST_LAUNCHES];
	uint32_t count = UINT32_MAX;
	memset(actual, UNTOUCHED_BYTE, sizeof(actual));
	memcpy(untouched, actual, sizeof(actual));
	expect(relaxed_trunk_bfloat16_prepare(description, ARENA_BYTES,
					      &kernels, actual, &count) ==
		       STATUS_INVALID_ARGUMENT,
	       what);
	expect(count == UINT32_MAX &&
		       memcmp(actual, untouched, sizeof(actual)) == 0,
	       "bfloat16 refusal leaves launches and count untouched");
}

static void test_fold(void)
{
	const uint32_t rows_form_code = INVALID_ROWS_FORM;
	enum relaxed_rows_code rows_form_from_configuration;
	_Static_assert(sizeof(rows_form_from_configuration) ==
			       sizeof(rows_form_code),
		       "configuration rows form has a uint32 representation");
	memcpy(&rows_form_from_configuration, &rows_form_code,
	       sizeof(rows_form_code));
	uint8_t codes[] = { RELAXED_SCALE_SMALLEST, RELAXED_SCALE_LARGEST };
	int32_t reference = 0;
	expect(relaxed_fold_check(RELAXED_QUANTIZED_ROWS, codes, sizeof(codes),
				  &reference) == STATUS_OK &&
		       reference == RELAXED_FOLD_REFERENCE,
	       "quantized fold accepts both scale bounds");
	codes[0]--;
	reference = TEST_FOLD_REFERENCE;
	expect(relaxed_fold_check(RELAXED_QUANTIZED_ROWS, codes, sizeof(codes),
				  &reference) == STATUS_INVALID_ARGUMENT &&
		       reference == TEST_FOLD_REFERENCE,
	       "quantized fold refuses below its lower bound without writing output");
	codes[0]++;
	codes[1]++;
	expect(relaxed_fold_check(RELAXED_QUANTIZED_ROWS, codes, sizeof(codes),
				  &reference) == STATUS_INVALID_ARGUMENT &&
		       reference == TEST_FOLD_REFERENCE,
	       "quantized fold refuses above its upper bound");
	codes[0] = TEST_CODE_TOP - RELAXED_HALF_SCALE_SPREAD;
	codes[1] = TEST_CODE_TOP;
	expect(relaxed_fold_check(RELAXED_BFLOAT16_ROWS, codes, sizeof(codes),
				  &reference) == STATUS_OK &&
		       reference == TEST_FOLD_REFERENCE,
	       "bfloat16 fold accepts the maximum spread and computes its reference");
	codes[0]--;
	expect(relaxed_fold_check(RELAXED_BFLOAT16_ROWS, codes, sizeof(codes),
				  &reference) == STATUS_INVALID_ARGUMENT &&
		       reference == TEST_FOLD_REFERENCE,
	       "bfloat16 fold refuses a wider spread");
	codes[0] = RELAXED_HALF_SCALE_LARGEST;
	expect(relaxed_fold_check(RELAXED_BFLOAT16_ROWS, codes, 1,
				  &reference) == STATUS_OK &&
		       reference == (int32_t)RELAXED_HALF_SCALE_LARGEST -
					    RELAXED_HALF_TOP,
	       "bfloat16 fold accepts the last unclamped reference scale");
	codes[0]++;
	reference = TEST_FOLD_REFERENCE;
	expect(relaxed_fold_check(RELAXED_BFLOAT16_ROWS, codes, 1,
				  &reference) == STATUS_INVALID_ARGUMENT &&
		       reference == TEST_FOLD_REFERENCE,
	       "bfloat16 fold refuses scales whose reference power clamps");
	codes[0] = (uint8_t)RELAXED_HALF_TOP;
	expect(relaxed_fold_check(RELAXED_BFLOAT16_ROWS, codes, 1,
				  &reference) == STATUS_INVALID_ARGUMENT &&
		       reference == TEST_FOLD_REFERENCE,
	       "bfloat16 fold refuses a reference below normal float32 powers");
	expect(relaxed_fold_check(rows_form_from_configuration, codes, 1,
				  &reference) == STATUS_INVALID_ARGUMENT,
	       "fold refuses an unknown row form");
	expect(relaxed_fold_check(RELAXED_QUANTIZED_ROWS, NULL, 1,
				  &reference) == STATUS_INVALID_ARGUMENT,
	       "fold refuses missing scale codes");
	expect(relaxed_fold_check(RELAXED_QUANTIZED_ROWS, codes, 0,
				  &reference) == STATUS_INVALID_ARGUMENT,
	       "fold refuses no scale codes");
	expect(relaxed_fold_check(RELAXED_QUANTIZED_ROWS, codes, 1, NULL) ==
		       STATUS_INVALID_ARGUMENT,
	       "fold refuses missing output");
}

static void test_quantized(void)
{
	struct relaxed_trunk_quantized_description description =
		quantized_description();
	struct kernel_launch launches[RELAXED_MOST_LAUNCHES] = { 0 };
	struct trunk_relaxed_quantized_arguments record;
	uint32_t count = 0;
	description.is_ordered = 1;
	expect(relaxed_trunk_quantized_prepare(&description, ARENA_BYTES,
					       &kernels, launches,
					       &count) == STATUS_OK &&
		       count == 1 && launches[0].kernel == ORDERED_HANDLE,
	       "ordered quantized uses the ordered small tile");
	memcpy(&record, launches[0].argument, sizeof(record));
	expect(record.output_size[0] == TEST_WIDTH &&
		       record.output_size[1] == TEST_WIDTH &&
		       record.weights_offset[1] == SECOND_MATRIX_OFFSET &&
		       record.partials_offset == 0 && record.cone_layer == 0 &&
		       record.step_offset == 0 && launches[0].grid[0] == 1 &&
		       launches[0].grid[1] == TEST_WIDTH *
						      LINEAR_MERGED_MATRICES /
						      RELAXED_WIDE_OUTPUTS,
	       "ordered quantized record covers both matrices and outputs");
	description.is_ordered = 0;
	description.k_split_count = TEST_SPLITS;
	expect(relaxed_trunk_quantized_prepare(&description, ARENA_BYTES,
					       &kernels, launches,
					       &count) == STATUS_OK &&
		       count == RELAXED_MOST_LAUNCHES &&
		       launches[0].kernel == FIRST_HANDLE &&
		       launches[1].kernel == REDUCE_HANDLE &&
		       launches[0].grid[LAUNCH_DIMENSIONS - 1u] == TEST_SPLITS,
	       "relaxed wide splits use product followed by reduction");
	description = quantized_description();
	description.row_count = 1;
	expect(relaxed_trunk_quantized_prepare(&description, ARENA_BYTES,
					       &kernels, launches,
					       &count) == STATUS_OK &&
		       count == 1 && launches[0].kernel == DECODE_HANDLE &&
		       launches[0].block[0] ==
			       TEST_SLICES * LINEAR_THREADS_PER_WARP,
	       "relaxed decode preserves its fixed slice count");
	description.is_ordered = 1;
	expect_quantized_refusal(&description, "ordered decode is refused");
	description = quantized_description();
	description.row_count = 0;
	expect_quantized_refusal(&description, "zero rows are refused");
	description = quantized_description();
	description.row_count = RELAXED_MOST_ROWS + 1u;
	expect_quantized_refusal(&description, "too many rows are refused");
	description = quantized_description();
	description.matrix_count = INVALID_MATRIX_COUNT;
	expect_quantized_refusal(&description, "too many matrices are refused");
	description = quantized_description();
	description.is_ordered = INVALID_FLAG;
	expect_quantized_refusal(&description, "an unknown order is refused");
	description = quantized_description();
	description.k_split_count = 0;
	expect_quantized_refusal(&description, "zero splits are refused");
	description = quantized_description();
	description.k_split_count = TEST_SPLITS;
	description.is_ordered = 1;
	expect_quantized_refusal(&description, "ordered splits are refused");
	description = quantized_description();
	description.matrices[0].reserved_zero = 1;
	expect_quantized_refusal(&description,
				 "matrix reserved bits are refused");
	description = quantized_description();
	description.matrices[0].shape.reserved_must_be_zero = 1;
	expect_quantized_refusal(&description,
				 "shape reserved bits are refused");
	description = quantized_description();
	description.matrices[0].shape.output_size--;
	expect_quantized_refusal(&description,
				 "a partial output tile is refused");
	description = quantized_description();
	description.matrices[0].weight_format = WEIGHT_FORMAT_E2M1_SCALED;
	expect_quantized_refusal(&description,
				 "a different weight format is refused");
	description = quantized_description();
	description.row_count = 1;
	description.matrices[0].slice_count = INVALID_SLICES;
	description.matrices[1].slice_count = INVALID_SLICES;
	expect_quantized_refusal(&description,
				 "unstamped decode slices are refused");
	description = quantized_description();
	description.outputs[1].byte_offset = FIRST_OUTPUT_OFFSET;
	expect_quantized_refusal(&description,
				 "overlapping outputs are refused");
	description = quantized_description();
	description.operands.byte_offset++;
	expect_quantized_refusal(&description,
				 "an unaligned operand is refused");
	description = quantized_description();
	description.operands.byte_count--;
	expect_quantized_refusal(&description, "short operands are refused");
	description = quantized_description();
	description.matrices[0].stored.byte_count--;
	expect_quantized_refusal(&description,
				 "short stored matrix is refused");
	expect_quantized_refusal(NULL, "a missing description is refused");
}

static void test_bfloat16(void)
{
	struct relaxed_trunk_bfloat16_description description =
		bfloat16_description();
	struct kernel_launch launches[RELAXED_MOST_LAUNCHES] = { 0 };
	struct trunk_relaxed_bfloat16_arguments record;
	uint32_t count = 0;
	description.is_ordered = 1;
	expect(relaxed_trunk_bfloat16_prepare(&description, ARENA_BYTES,
					      &kernels, launches,
					      &count) == STATUS_OK &&
		       count == 1 &&
		       launches[0].kernel == BFLOAT16_ORDERED_HANDLE,
	       "ordered bfloat16 uses the ordered small tile");
	memcpy(&record, launches[0].argument, sizeof(record));
	expect(record.segment_count == TEST_SEGMENTS &&
		       record.output_size == TEST_WIDTH &&
		       record.fold_reference == TEST_FOLD_REFERENCE &&
		       record.partials_offset == 0,
	       "grouped bfloat16 record preserves segments");
	description.is_ordered = 0;
	description.row_count = LINEAR_MAXIMUM_ROWS;
	expect(relaxed_trunk_bfloat16_prepare(&description, ARENA_BYTES,
					      &kernels, launches,
					      &count) == STATUS_OK &&
		       count == 1 &&
		       launches[0].kernel == BFLOAT16_DECODE_HANDLE,
	       "bfloat16 decode uses its decode form through eight rows");
	description.is_ordered = 1;
	expect_bfloat16_refusal(&description,
				"ordered bfloat16 decode is refused");
	description = bfloat16_description();
	description.segment_count = 0;
	expect_bfloat16_refusal(&description, "zero segments are refused");
	description = bfloat16_description();
	description.segment_count = TEST_SEGMENTS + 1u;
	expect_bfloat16_refusal(&description,
				"a segment cutting an output tile is refused");
	description = bfloat16_description();
	description.fold_reference = 0;
	expect_bfloat16_refusal(&description,
				"a nonnormal fold power is refused");
	description = bfloat16_description();
	description.values.byte_count--;
	expect_bfloat16_refusal(&description,
				"short grouped values are refused");
	description = bfloat16_description();
	description.output.byte_offset = OPERANDS_OFFSET;
	expect_bfloat16_refusal(&description,
				"output overlapping grouped values is refused");
	description = bfloat16_description();
	description.matrix.reserved_zero = 1;
	expect_bfloat16_refusal(&description,
				"bfloat16 matrix reserved bits are refused");
	expect_bfloat16_refusal(NULL,
				"a missing bfloat16 description is refused");
}

/* A one-split product needs only its selected handle; the retained alternatives are independently usable. */
static void test_selected_handles(void)
{
	struct relaxed_linear_kernels selected = kernels;
	struct relaxed_trunk_quantized_description quantized =
		quantized_description();
	struct relaxed_trunk_bfloat16_description bfloat16 =
		bfloat16_description();
	struct kernel_launch launches[RELAXED_MOST_LAUNCHES] = { 0 };
	uint32_t count = 0;
	selected.reduce = BACKEND_NO_KERNEL;
	quantized.is_ordered = 1;
	bfloat16.is_ordered = 1;
	selected.ordered_wide_quantized[RELAXED_TILE_LARGE] = BACKEND_NO_KERNEL;
	selected.ordered_wide_bfloat16[RELAXED_TILE_LARGE] = BACKEND_NO_KERNEL;
	expect(relaxed_trunk_quantized_prepare(&quantized, ARENA_BYTES,
					       &selected, launches,
					       &count) == STATUS_OK,
	       "ordered quantized does not need unused tile or reduction handles");
	expect(relaxed_trunk_bfloat16_prepare(&bfloat16, ARENA_BYTES, &selected,
					      launches, &count) == STATUS_OK,
	       "ordered BF16 does not need unused tile or reduction handles");
	quantized.is_ordered = 0;
	quantized.row_count = 1;
	selected.wide_quantized[RELAXED_TILE_SMALL] = BACKEND_NO_KERNEL;
	selected.wide_quantized[RELAXED_TILE_LARGE] = BACKEND_NO_KERNEL;
	expect(relaxed_trunk_quantized_prepare(&quantized, ARENA_BYTES,
					       &selected, launches,
					       &count) == STATUS_OK,
	       "decode needs neither unused wide nor reduction handles");
}

/* These ranges describe no allocation. Without the count check the reduction grid silently narrows to 32 bits. */
static void test_oversized_reduction(void)
{
	struct relaxed_trunk_quantized_description description =
		quantized_description();
	description.matrix_count = 1;
	description.row_count = OVERSIZED_ROWS;
	description.k_split_count = TEST_SPLITS;
	description.is_ordered = 0;
	description.matrices[0].shape.output_size =
		UINT32_MAX / RELAXED_WIDE_OUTPUTS * RELAXED_WIDE_OUTPUTS;
	description.matrices[0].stored.byte_offset = OVERSIZED_WEIGHT_OFFSET;
	const struct weight_format_operations *const format =
		weight_format_operations_of(
			description.matrices[0].weight_format);
	expect(format != NULL &&
		       format->stored_byte_count(
			       &description.matrices[0].shape,
			       &description.matrices[0].stored.byte_count) ==
			       STATUS_OK,
	       "the oversized fixture is a stored matrix");
	description.operands = (struct linear_range){
		OVERSIZED_OPERANDS_OFFSET,
		linear_values_byte_count(OVERSIZED_ROWS, TEST_WIDTH)
	};
	description.outputs[0] = (struct linear_range){
		OVERSIZED_OUTPUT_OFFSET,
		linear_values_byte_count(
			OVERSIZED_ROWS,
			description.matrices[0].shape.output_size)
	};
	description.partials = (struct linear_range){
		OVERSIZED_PARTIALS_OFFSET,
		linear_sums_byte_count((uint64_t)TEST_SPLITS * OVERSIZED_ROWS,
				       description.matrices[0].shape.output_size)
	};
	struct kernel_launch launches[RELAXED_MOST_LAUNCHES] = { 0 };
	uint32_t count = UINT32_MAX;
	expect(relaxed_trunk_quantized_prepare(&description, UINT64_MAX,
					       &kernels, launches, &count) ==
			       STATUS_INVALID_ARGUMENT &&
		       count == UINT32_MAX,
	       "a split reduction whose block count passes 32 bits is refused");
}

/* relaxed_trunk_cone_set: a wide launch of either rows takes the step area and the layer; a decode launch
 * stays as it is; a layer outside the decoder's or a short step area is refused. */
static void test_cone(void)
{
	struct relaxed_trunk_quantized_description quantized =
		quantized_description();
	struct relaxed_trunk_bfloat16_description bfloat16 =
		bfloat16_description();
	struct kernel_launch launches[RELAXED_MOST_LAUNCHES] = { 0 };
	struct trunk_relaxed_quantized_arguments quantized_record;
	struct trunk_relaxed_bfloat16_arguments bfloat16_record;
	const struct linear_range step = { STEP_OFFSET, STEP_BYTES };
	const struct linear_range short_step = { STEP_OFFSET,
						 STEP_BYTES - 16u };
	uint32_t count = 0;
	quantized.is_ordered = 1;
	expect(relaxed_trunk_quantized_prepare(&quantized, ARENA_BYTES,
					       &kernels, launches,
					       &count) == STATUS_OK &&
		       relaxed_trunk_cone_set(&kernels, step, CONE_LAYER,
					      ARENA_BYTES,
					      &launches[0]) == STATUS_OK,
	       "an ordered quantized launch takes the cone");
	memcpy(&quantized_record, launches[0].argument, sizeof(quantized_record));
	expect(quantized_record.step_offset == STEP_OFFSET && quantized_record.cone_layer == CONE_LAYER &&
		       quantized_record.row_count == TEST_ROWS,
	       "the quantized record holds the step area and the layer");
	expect(relaxed_trunk_cone_set(&kernels, step, FRONT_LAYER, ARENA_BYTES,
				      &launches[0]) ==
			       STATUS_INVALID_ARGUMENT &&
		       relaxed_trunk_cone_set(&kernels, short_step, CONE_LAYER,
					      ARENA_BYTES, &launches[0]) ==
			       STATUS_INVALID_ARGUMENT,
	       "the cone refuses a front layer and a short step area");
	bfloat16.is_ordered = 1;
	expect(relaxed_trunk_bfloat16_prepare(&bfloat16, ARENA_BYTES, &kernels,
					      launches, &count) == STATUS_OK &&
		       relaxed_trunk_cone_set(&kernels, step, CONE_LAYER,
					      ARENA_BYTES,
					      &launches[0]) == STATUS_OK,
	       "an ordered bfloat16 launch takes the cone");
	memcpy(&bfloat16_record, launches[0].argument, sizeof(bfloat16_record));
	expect(bfloat16_record.step_offset == STEP_OFFSET && bfloat16_record.cone_layer == CONE_LAYER &&
		       bfloat16_record.reserved_zero == 0,
	       "the bfloat16 record holds the step area and the layer");
	quantized.row_count = DECODE_ROWS;
	quantized.is_ordered = 0;
	expect(relaxed_trunk_quantized_prepare(&quantized, ARENA_BYTES,
					       &kernels, launches,
					       &count) == STATUS_OK,
	       "a decode launch");
	memcpy(&quantized_record, launches[0].argument, sizeof(quantized_record));
	const uint32_t decode_kernel = launches[0].kernel;
	expect(relaxed_trunk_cone_set(&kernels, step, CONE_LAYER, ARENA_BYTES,
				      &launches[0]) == STATUS_OK &&
		       launches[0].kernel == decode_kernel &&
		       memcmp(launches[0].argument, &quantized_record,
			      sizeof(quantized_record)) == 0,
	       "a decode launch stays as it is");
}

int main(void)
{
	test_fold();
	test_cone();
	test_quantized();
	test_bfloat16();
	test_selected_handles();
	test_oversized_reduction();
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
