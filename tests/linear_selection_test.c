/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* linear_selection_test: class-80 trunk selection and the legacy launch bytes retained by every other path. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/linear.h"
#include "anemone/relaxed_linear.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define ARENA_BYTES (1ull << 40u)
#define BITS_PER_WEIGHT 8u
#define FIRST_HANDLE 10u
#define FIRST_RELAXED_HANDLE 80u
#define GROUPED_OUTPUTS 8192u
#define GROUPED_INPUTS 4096u
#define GROUPED_SEGMENTS 8u
#define REFERENCE_FOLD 120
#define REFUSED_OUTPUTS \
	144u /* the reference stores 16-output tiles; the ordered kernel requires 128 */
#define REFUSED_REFERENCE 2
#define UNTOUCHED_BYTE 0xA5
#define UNKNOWN_CLASS 81u
#define UNKNOWN_ORDER (CMP_ORDER_RELAXED + 1)
#define INDEXER_SLICES 4u
#define INDEXER_OUTPUTS 4096u
#define INDEXER_INPUTS 1280u
#define WIDE_ROWS 64u
#define SHARED_OUTPUTS 2304u
#define SHARED_INPUTS 5120u
#define ACTIVATION_LIMIT 10.0f

static int failures;
static struct linear_kernels_record kernels;
static struct relaxed_linear_kernels relaxed_kernels;

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)printf("FAILED: %s\n", what);
		failures++;
	}
}

static struct linear_range next_range(uint64_t *cursor, uint64_t bytes)
{
	const struct linear_range range = { *cursor, bytes };
	*cursor += (bytes + LINEAR_OFFSET_ALIGNMENT_BYTES - 1u) /
		   LINEAR_OFFSET_ALIGNMENT_BYTES *
		   LINEAR_OFFSET_ALIGNMENT_BYTES;
	return range;
}

static struct linear_matrix next_matrix(uint64_t *cursor, uint32_t outputs,
					uint32_t inputs)
{
	struct linear_matrix matrix = {
		.shape = { outputs, inputs, BITS_PER_WEIGHT, 0 },
		.weight_format = WEIGHT_FORMAT_E4M3_SCALED,
		.slice_count = 1,
	};
	const struct weight_format_operations *const format =
		weight_format_operations_of(matrix.weight_format);
	uint64_t bytes = 0;
	expect(format != NULL && format->stored_byte_count(&matrix.shape,
							   &bytes) == STATUS_OK,
	       "the fixture's matrix is stored by its format");
	matrix.stored = next_range(cursor, bytes);
	return matrix;
}

static struct linear_trunk_quantized_description
quantized_of(uint32_t outputs, uint32_t inputs, uint32_t rows)
{
	uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
	struct linear_trunk_quantized_description description = {
		.matrix_count = 1, .row_count = rows
	};
	description.matrices[0] = next_matrix(&cursor, outputs, inputs);
	description.codes =
		next_range(&cursor, linear_codes_byte_count(rows, inputs));
	description.exponents =
		next_range(&cursor, linear_exponents_byte_count(rows, inputs));
	description.operands =
		next_range(&cursor, linear_values_byte_count(rows, inputs));
	description.outputs[0] =
		next_range(&cursor, linear_values_byte_count(rows, outputs));
	return description;
}

static struct linear_trunk_bfloat16_description bfloat16_of(uint32_t rows)
{
	uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
	struct linear_trunk_bfloat16_description description = {
		.row_count = rows,
		.segment_count = GROUPED_SEGMENTS,
	};
	description.matrix =
		next_matrix(&cursor, GROUPED_OUTPUTS, GROUPED_INPUTS);
	description.values = next_range(
		&cursor,
		linear_values_byte_count(rows, (uint64_t)GROUPED_SEGMENTS *
						       GROUPED_INPUTS));
	description.output = next_range(
		&cursor, linear_values_byte_count(rows, GROUPED_OUTPUTS));
	return description;
}

static struct linear_shared_expert_description shared_of(uint32_t rows)
{
	uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
	const uint64_t hidden_bytes =
		linear_values_byte_count(rows, SHARED_OUTPUTS);
	struct linear_shared_expert_description description = {
		.row_count = rows,
		.activation_limit = ACTIVATION_LIMIT,
	};
	description.gate = next_matrix(&cursor, SHARED_OUTPUTS, SHARED_INPUTS);
	description.up = next_matrix(&cursor, SHARED_OUTPUTS, SHARED_INPUTS);
	description.down = next_matrix(&cursor, SHARED_INPUTS, SHARED_OUTPUTS);
	description.codes = next_range(
		&cursor, linear_codes_byte_count(rows, SHARED_INPUTS));
	description.exponents = next_range(
		&cursor, linear_exponents_byte_count(rows, SHARED_INPUTS));
	description.operands = next_range(
		&cursor, linear_values_byte_count(rows, SHARED_INPUTS));
	description.gate_output = next_range(&cursor, hidden_bytes);
	description.up_output = next_range(&cursor, hidden_bytes);
	description.hidden = next_range(&cursor, hidden_bytes);
	description.hidden_codes = next_range(
		&cursor, linear_codes_byte_count(rows, SHARED_OUTPUTS));
	description.hidden_exponents = next_range(
		&cursor, linear_exponents_byte_count(rows, SHARED_OUTPUTS));
	description.hidden_operands = next_range(&cursor, hidden_bytes);
	description.output = next_range(
		&cursor, linear_values_byte_count(rows, SHARED_INPUTS));
	return description;
}

static void test_selection_bytes(void)
{
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120,
					    ARITHMETIC_CLASS_SM_120_NATIVE };
	static const uint32_t wide_rows[] = { 64, 256, 1024 };
	static const struct matrix_shape_record shapes[] = {
		{ 1280, 5120, BITS_PER_WEIGHT, 0 },
		{ 512, 5120, BITS_PER_WEIGHT, 0 },
		{ 32768, 1280, BITS_PER_WEIGHT, 0 },
		{ 4096, 1280, BITS_PER_WEIGHT, 0 },
		{ 5120, 8192, BITS_PER_WEIGHT, 0 },
		{ 25600, 6144, BITS_PER_WEIGHT, 0 },
		{ 2304, 5120, BITS_PER_WEIGHT, 0 },
		{ 5120, 2304, BITS_PER_WEIGHT, 0 },
	};
	const int32_t folds[LINEAR_MERGED_MATRICES] = {
		RELAXED_FOLD_REFERENCE,
		RELAXED_FOLD_REFERENCE,
	};
	for (size_t card = 0; card < COUNT_OF(classes); card++) {
		for (uint32_t rows = 1;
		     rows <= LINEAR_MAXIMUM_ROWS + COUNT_OF(wide_rows);
		     rows++) {
			const uint32_t row_count =
				rows <= LINEAR_MAXIMUM_ROWS ?
					rows :
					wide_rows[rows - LINEAR_MAXIMUM_ROWS -
						  1u];
			const bool is_ordered =
				classes[card] == ARITHMETIC_CLASS_SM_80 &&
				row_count > LINEAR_MAXIMUM_ROWS;
			for (size_t shape = 0; shape < COUNT_OF(shapes);
			     shape++) {
				const struct linear_trunk_quantized_description
					description = quantized_of(
						shapes[shape].output_size,
						shapes[shape].input_size,
						row_count);
				struct kernel_launch before = { 0 },
						     after = { 0 };
				expect(linear_trunk_quantized_prepare(
					       &description, ARENA_BYTES,
					       &kernels, &before) == STATUS_OK,
				       "the legacy quantized launch prepares");
				expect(linear_trunk_quantized_select_prepare(
					       &description, ARENA_BYTES,
					       &kernels, &relaxed_kernels,
					       classes[card],
					       CMP_ORDER_REFERENCE, folds,
					       &after) == STATUS_OK,
				       "the selected quantized launch prepares");
				expect(is_ordered ?
					       after.kernel ==
						       relaxed_kernels
							       .ordered_wide_quantized
								       [0] :
					       memcmp(&before, &after,
						      sizeof(before)) == 0,
				       "CMP wide selects ordered; thin and PRO quantized launch bytes are unchanged");
				if (classes[card] != ARITHMETIC_CLASS_SM_80) {
					expect(linear_trunk_quantized_select_prepare(
						       &description,
						       ARENA_BYTES, &kernels,
						       &relaxed_kernels,
						       classes[card],
						       CMP_ORDER_RELAXED, folds,
						       &after) == STATUS_OK &&
						       memcmp(&before, &after,
							      sizeof(before)) ==
							       0,
					       "the relaxed knob preserves every PRO launch byte");
				}
			}
			const struct linear_trunk_bfloat16_description bf16 =
				bfloat16_of(row_count);
			struct kernel_launch before = { 0 }, after = { 0 };
			expect(linear_trunk_bfloat16_prepare(
				       &bf16, ARENA_BYTES, &kernels, &before) ==
					       STATUS_OK &&
				       linear_trunk_bfloat16_select_prepare(
					       &bf16, ARENA_BYTES, &kernels,
					       &relaxed_kernels, classes[card],
					       CMP_ORDER_REFERENCE,
					       REFERENCE_FOLD,
					       &after) == STATUS_OK,
			       "grouped BF16 launches prepare");
			expect(is_ordered ?
				       after.kernel ==
					       relaxed_kernels
						       .ordered_wide_bfloat16[0] :
				       memcmp(&before, &after,
					      sizeof(before)) == 0,
			       "CMP wide grouped BF16 selects ordered; every thin and PRO launch byte is unchanged");
			const struct linear_shared_expert_description shared =
				shared_of(row_count);
			struct kernel_launch
				old_chain[LINEAR_SHARED_EXPERT_LAUNCH_COUNT] = {
					0
				};
			struct kernel_launch
				new_chain[LINEAR_SHARED_EXPERT_LAUNCH_COUNT] = {
					0
				};
			expect(linear_shared_expert_prepare(
				       &shared, ARENA_BYTES, &kernels,
				       old_chain) == STATUS_OK &&
				       linear_shared_expert_select_prepare(
					       &shared, ARENA_BYTES, &kernels,
					       &relaxed_kernels, classes[card],
					       CMP_ORDER_REFERENCE, NULL,
					       new_chain) == STATUS_OK,
			       "shared expert chains prepare");
			expect(is_ordered ?
				       new_chain[LINEAR_SHARED_EXPERT_GATE_UP]
							       .kernel ==
						       relaxed_kernels
							       .ordered_wide_quantized
								       [0] &&
					       new_chain[LINEAR_SHARED_EXPERT_DOWN]
							       .kernel ==
						       relaxed_kernels
							       .ordered_wide_quantized
								       [0] :
				       memcmp(old_chain, new_chain,
					      sizeof(old_chain)) == 0,
			       "shared products select ordered while thin and PRO chains keep every byte");
			expect(memcmp(&old_chain[LINEAR_SHARED_EXPERT_HIDDEN],
				      &new_chain[LINEAR_SHARED_EXPERT_HIDDEN],
				      sizeof(before)) == 0 &&
				       memcmp(&old_chain
						      [LINEAR_SHARED_EXPERT_QUANTIZE],
					      &new_chain
						      [LINEAR_SHARED_EXPERT_QUANTIZE],
					      sizeof(before)) == 0,
			       "the shared hidden step and quantizer keep every launch byte");
		}
	}
}

static void test_relaxed_and_refusals(void)
{
	static const uint32_t rows_tried[] = { 1, 8, 64, 256, 1024 };
	const int32_t fold = RELAXED_FOLD_REFERENCE;
	for (size_t index = 0; index < COUNT_OF(rows_tried); index++) {
		const uint32_t rows = rows_tried[index];
		struct linear_trunk_quantized_description description =
			quantized_of(INDEXER_OUTPUTS, INDEXER_INPUTS, rows);
		struct kernel_launch reference = { 0 }, relaxed = { 0 },
				     fallback = { 0 };
		expect(linear_trunk_quantized_select_prepare(
			       &description, ARENA_BYTES, &kernels,
			       &relaxed_kernels, ARITHMETIC_CLASS_SM_80,
			       CMP_ORDER_REFERENCE, NULL,
			       &reference) == STATUS_OK &&
			       linear_trunk_quantized_select_prepare(
				       &description, ARENA_BYTES, &kernels,
				       &relaxed_kernels, ARITHMETIC_CLASS_SM_80,
				       CMP_ORDER_RELAXED, &fold,
				       &relaxed) == STATUS_OK,
		       "reference and relaxed indexer products prepare");
		expect(rows <= LINEAR_MAXIMUM_ROWS ?
			       relaxed.kernel == relaxed_kernels
							 .decode_quantized &&
				       relaxed.block[0] ==
					       INDEXER_SLICES *
						       LINEAR_THREADS_PER_WARP :
			       relaxed.kernel ==
				       relaxed_kernels.wide_quantized[0],
		       "relaxed decode takes the fixed indexer count and wide takes the relaxed instance");
		expect(linear_trunk_quantized_select_prepare(
			       &description, ARENA_BYTES, &kernels,
			       &relaxed_kernels, ARITHMETIC_CLASS_SM_80,
			       CMP_ORDER_RELAXED, NULL,
			       &fallback) == STATUS_OK &&
			       memcmp(&reference, &fallback,
				      sizeof(reference)) == 0,
		       "a refused fold stays in reference order at thin and wide row counts");
		const struct linear_trunk_bfloat16_description bf16 =
			bfloat16_of(rows);
		expect(linear_trunk_bfloat16_select_prepare(
			       &bf16, ARENA_BYTES, &kernels, &relaxed_kernels,
			       ARITHMETIC_CLASS_SM_80, CMP_ORDER_RELAXED,
			       REFERENCE_FOLD, &relaxed) == STATUS_OK &&
			       relaxed.kernel ==
				       (rows <= LINEAR_MAXIMUM_ROWS ?
						relaxed_kernels.decode_bfloat16 :
						relaxed_kernels
							.wide_bfloat16[0]),
		       "grouped BF16 selects relaxed decode and wide forms");
		const struct linear_shared_expert_description shared =
			shared_of(rows);
		struct kernel_launch chain[LINEAR_SHARED_EXPERT_LAUNCH_COUNT];
		int32_t shared_folds[LINEAR_MERGED_MATRICES + 1u] = { fold,
								      fold,
								      fold };
		expect(linear_shared_expert_select_prepare(
			       &shared, ARENA_BYTES, &kernels, &relaxed_kernels,
			       ARITHMETIC_CLASS_SM_80, CMP_ORDER_RELAXED,
			       shared_folds, chain) == STATUS_OK &&
			       chain[LINEAR_SHARED_EXPERT_GATE_UP].kernel ==
				       (rows <= LINEAR_MAXIMUM_ROWS ?
						relaxed_kernels.decode_quantized :
						relaxed_kernels
							.wide_quantized[0]) &&
			       chain[LINEAR_SHARED_EXPERT_DOWN].kernel ==
				       chain[LINEAR_SHARED_EXPERT_GATE_UP]
					       .kernel,
		       "both shared products take the relaxed form");
		shared_folds[1] = 0;
		expect(linear_shared_expert_select_prepare(
			       &shared, ARENA_BYTES, &kernels, &relaxed_kernels,
			       ARITHMETIC_CLASS_SM_80, CMP_ORDER_RELAXED,
			       shared_folds, chain) == STATUS_OK &&
			       chain[LINEAR_SHARED_EXPERT_GATE_UP].kernel ==
				       (rows <= LINEAR_MAXIMUM_ROWS ?
						kernels.trunk_linear_quantized[0] :
						relaxed_kernels
							.ordered_wide_quantized
								[0]) &&
			       chain[LINEAR_SHARED_EXPERT_DOWN].kernel ==
				       (rows <= LINEAR_MAXIMUM_ROWS ?
						relaxed_kernels.decode_quantized :
						relaxed_kernels
							.wide_quantized[0]),
		       "a refused up fold keeps its merged gate/up reference while down remains relaxed");
	}
}

static void expect_quantized_refusal(
	const struct linear_trunk_quantized_description *description,
	uint64_t arena_bytes, uint32_t arithmetic_class,
	enum cmp_order_code order, const int32_t *folds,
	const struct relaxed_linear_kernels *selected_kernels)
{
	struct kernel_launch launch, untouched;
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	memcpy(&untouched, &launch, sizeof(untouched));
	expect(linear_trunk_quantized_select_prepare(
		       description, arena_bytes, &kernels, selected_kernels,
		       arithmetic_class, order, folds,
		       &launch) == STATUS_INVALID_ARGUMENT &&
		       memcmp(&launch, &untouched, sizeof(launch)) == 0,
	       "a refused selection leaves every output launch byte untouched");
}

static void test_refusals(void)
{
	const uint32_t card = ARITHMETIC_CLASS_SM_80;
	const uint32_t order_code = UNKNOWN_ORDER;
	enum cmp_order_code order_from_configuration;
	_Static_assert(sizeof(order_from_configuration) == sizeof(order_code),
		       "configuration order has a uint32 representation");
	memcpy(&order_from_configuration, &order_code, sizeof(order_code));
	const int32_t fold = RELAXED_FOLD_REFERENCE,
		      invalid_fold = REFUSED_REFERENCE;
	const struct linear_trunk_quantized_description good =
		quantized_of(INDEXER_OUTPUTS, INDEXER_INPUTS, WIDE_ROWS);
	struct linear_trunk_quantized_description bad =
		quantized_of(REFUSED_OUTPUTS, INDEXER_INPUTS, WIDE_ROWS);
	struct kernel_launch launch;
	expect(linear_trunk_quantized_prepare(&bad, ARENA_BYTES, &kernels,
					      &launch) == STATUS_OK,
	       "the legacy ABI accepts a shape the new 128-output tile refuses");
	expect_quantized_refusal(&bad, ARENA_BYTES, card, CMP_ORDER_REFERENCE,
				 NULL, &relaxed_kernels);
	expect_quantized_refusal(&good, ARENA_BYTES, UNKNOWN_CLASS,
				 CMP_ORDER_REFERENCE, NULL, &relaxed_kernels);
	expect_quantized_refusal(&good, ARENA_BYTES, card,
				 order_from_configuration, NULL,
				 &relaxed_kernels);
	expect_quantized_refusal(&good, ARENA_BYTES, card, CMP_ORDER_RELAXED,
				 &invalid_fold, &relaxed_kernels);
	bad = good;
	uint64_t merged_cursor =
		good.outputs[0].byte_offset + good.outputs[0].byte_count;
	bad.matrix_count = LINEAR_MERGED_MATRICES;
	bad.matrices[1] =
		next_matrix(&merged_cursor, INDEXER_OUTPUTS, INDEXER_INPUTS);
	bad.outputs[1] = next_range(&merged_cursor,
				    linear_values_byte_count(WIDE_ROWS,
							     INDEXER_OUTPUTS));
	const int32_t mixed_invalid_folds[LINEAR_MERGED_MATRICES] = {
		0, REFUSED_REFERENCE
	};
	expect_quantized_refusal(&bad, ARENA_BYTES, card, CMP_ORDER_RELAXED,
				 mixed_invalid_folds, &relaxed_kernels);
	expect_quantized_refusal(&good, ARENA_BYTES, card, CMP_ORDER_REFERENCE,
				 NULL, NULL);
	expect_quantized_refusal(NULL, ARENA_BYTES, card, CMP_ORDER_REFERENCE,
				 NULL, &relaxed_kernels);
	bad = good;
	bad.matrices[0].reserved_zero = 1;
	expect_quantized_refusal(&bad, ARENA_BYTES, card, CMP_ORDER_REFERENCE,
				 NULL, &relaxed_kernels);
	bad = good;
	bad.matrices[0].slice_count = 0;
	expect_quantized_refusal(&bad, ARENA_BYTES, card, CMP_ORDER_REFERENCE,
				 NULL, &relaxed_kernels);
	bad = good;
	bad.codes.byte_count--;
	expect_quantized_refusal(&bad, ARENA_BYTES, card, CMP_ORDER_REFERENCE,
				 NULL, &relaxed_kernels);
	bad = good;
	bad.operands.byte_count--;
	expect_quantized_refusal(&bad, ARENA_BYTES, card, CMP_ORDER_RELAXED,
				 &fold, &relaxed_kernels);
	bad = good;
	bad.outputs[0].byte_offset = bad.operands.byte_offset;
	expect_quantized_refusal(&bad, ARENA_BYTES, card, CMP_ORDER_REFERENCE,
				 NULL, &relaxed_kernels);
	expect_quantized_refusal(
		&good,
		good.outputs[0].byte_offset + good.outputs[0].byte_count - 1u,
		card, CMP_ORDER_REFERENCE, NULL, &relaxed_kernels);
	struct relaxed_linear_kernels missing = relaxed_kernels;
	missing.ordered_wide_quantized[0] = BACKEND_NO_KERNEL;
	expect_quantized_refusal(&good, ARENA_BYTES, card, CMP_ORDER_REFERENCE,
				 NULL, &missing);
	const struct linear_trunk_bfloat16_description bf16 =
		bfloat16_of(WIDE_ROWS);
	struct kernel_launch untouched;
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	memcpy(&untouched, &launch, sizeof(untouched));
	expect(linear_trunk_bfloat16_select_prepare(
		       &bf16, ARENA_BYTES, &kernels, &relaxed_kernels, card,
		       CMP_ORDER_RELAXED, -1,
		       &launch) == STATUS_INVALID_ARGUMENT &&
		       memcmp(&launch, &untouched, sizeof(launch)) == 0,
	       "an invalid BF16 fold reference is refused without touching the launch");
	struct linear_shared_expert_description shared = shared_of(WIDE_ROWS);
	struct kernel_launch chain[LINEAR_SHARED_EXPERT_LAUNCH_COUNT];
	struct kernel_launch old_chain[LINEAR_SHARED_EXPERT_LAUNCH_COUNT];
	shared.hidden.byte_count--;
	memset(chain, UNTOUCHED_BYTE, sizeof(chain));
	memcpy(old_chain, chain, sizeof(chain));
	expect(linear_shared_expert_select_prepare(
		       &shared, ARENA_BYTES, &kernels, &relaxed_kernels, card,
		       CMP_ORDER_REFERENCE, NULL,
		       chain) == STATUS_INVALID_ARGUMENT &&
		       memcmp(chain, old_chain, sizeof(chain)) == 0,
	       "the selector preserves validation of shared scratch and leaves the whole refused chain untouched");
}

int main(void)
{
	uint32_t handles[sizeof(kernels) / sizeof(uint32_t)];
	uint32_t relaxed_handles[sizeof(relaxed_kernels) / sizeof(uint32_t)];
	for (size_t index = 0; index < COUNT_OF(handles); index++) {
		handles[index] = FIRST_HANDLE + (uint32_t)index;
	}
	for (size_t index = 0; index < COUNT_OF(relaxed_handles); index++) {
		relaxed_handles[index] = FIRST_RELAXED_HANDLE + (uint32_t)index;
	}
	memcpy(&kernels, handles, sizeof(kernels));
	memcpy(&relaxed_kernels, relaxed_handles, sizeof(relaxed_kernels));
	test_selection_bytes();
	test_relaxed_and_refusals();
	test_refusals();
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
