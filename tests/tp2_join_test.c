/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* TP-only completion admission and record binding, without a CUDA context. */
#include <stdio.h>
#include <string.h>

#include "anemone/tp2_join.h"

#define TEST_ROWS 6u
#define TEST_WIDTH 5120u
#define TEST_HEAD_HALF 16384u
#define TEST_CANONICAL_HEADS 32768u
#define TEST_CAPACITY 1048576ull
#define TEST_OUTPUT_OFFSET (TEST_CAPACITY * 2u)
#define TEST_STEP_OFFSET (TEST_CAPACITY * 3u)
#define TEST_ARENA_BYTES (TEST_CAPACITY * 4u)
#define TEST_SLOT_STRIDE (TEST_CAPACITY / 2u)
#define TEST_HANDLE 41u
#define TEST_POISON 0xA5
#define TEST_UNKNOWN_OPERATION 7u
#define TEST_GRID_COLUMNS 20u
#define TEST_STEP_SHORT_BYTES 8u
#define TEST_BAD_STRIDE 7u
#define TEST_INTERLEAVED_OFFSET (TEST_SLOT_STRIDE / 2u)

static int failures;
static uint32_t kernel_calls;
static bool kernel_failure;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

static MUST_CHECK result_status kernel_prepare_stub(struct device_card *card,
						    const char *file,
						    const char *function,
						    uint32_t bytes,
						    uint32_t *kernel_out)
{
	(void)card;
	kernel_calls++;
	expect(strcmp(file, KERNEL_FILE_TP2_JOIN) == 0 &&
		       strcmp(function, KERNEL_TP2_JOIN) == 0 &&
		       bytes == TP2_JOIN_ARGUMENT_BYTES,
	       "the TP-only kernel name and argument ABI");
	*kernel_out = TEST_HANDLE;
	return kernel_failure ? STATUS_DEVICE_ERROR : STATUS_OK;
}

static struct tp2_join_description description_make(void)
{
	return (struct tp2_join_description){
		.source = { 0, TEST_CAPACITY },
		.output = { TEST_OUTPUT_OFFSET, TEST_CAPACITY },
		.width = TEST_WIDTH,
		.row_count = TEST_ROWS,
		.source_row_stride = TEST_WIDTH,
		.output_row_stride = TEST_WIDTH,
		.operation = TP2_JOIN_BFLOAT16_TO_FLOAT32
	};
}

static void refused(const struct tp2_join_description *description,
		    uint64_t capacity, uint32_t kernel, const char *message)
{
	struct kernel_launch launch;
	unsigned char previous[sizeof(launch)];
	memset(&launch, TEST_POISON, sizeof(launch));
	memcpy(previous, &launch, sizeof(previous));
	const result_status status =
		tp2_join_prepare(description, capacity, kernel, &launch);
	expect(status == STATUS_INVALID_ARGUMENT &&
		       memcmp(previous, &launch, sizeof(previous)) == 0,
	       message);
}

static void kernel_admission(void)
{
	const struct backend_operations backend = {
		.kernel_prepare = kernel_prepare_stub
	};
	const struct backend_operations absent = { 0 };
	struct device_card *const card = (struct device_card *)(uintptr_t)1;
	uint32_t kernel = 0;
	result_status status = tp2_join_kernel_prepare(&backend, card, &kernel);
	expect(status == STATUS_OK && kernel == TEST_HANDLE &&
		       kernel_calls == 1,
	       "one independent kernel binding");
	kernel_failure = true;
	kernel = 0;
	status = tp2_join_kernel_prepare(&backend, card, &kernel);
	expect(status == STATUS_DEVICE_ERROR && kernel == 0,
	       "failed kernel binding leaves output unchanged");
	kernel_failure = false;
	status = tp2_join_kernel_prepare(NULL, card, &kernel);
	expect(status == STATUS_INVALID_ARGUMENT, "missing backend refused");
	status = tp2_join_kernel_prepare(&absent, card, &kernel);
	expect(status == STATUS_INVALID_ARGUMENT,
	       "missing kernel callback refused");
	status = tp2_join_kernel_prepare(&backend, NULL, &kernel);
	expect(status == STATUS_INVALID_ARGUMENT, "missing card refused");
	status = tp2_join_kernel_prepare(&backend, card, NULL);
	expect(status == STATUS_INVALID_ARGUMENT, "missing output refused");
}

static void operations(void)
{
	struct tp2_join_description description = description_make();
	struct kernel_launch launch = { 0 };
	struct tp2_join_arguments arguments = { 0 };
	result_status status = tp2_join_prepare(&description, TEST_ARENA_BYTES,
						TEST_HANDLE, &launch);
	memcpy(&arguments, launch.argument, sizeof(arguments));
	expect(status == STATUS_OK && launch.kernel == TEST_HANDLE &&
		       launch.grid[0] == TEST_GRID_COLUMNS &&
		       launch.grid[1] == TEST_ROWS &&
		       launch.block[0] == TP2_JOIN_THREADS &&
		       launch.argument_byte_count == TP2_JOIN_ARGUMENT_BYTES &&
		       arguments.has_routed_sums == 0 &&
		       arguments.operation == TP2_JOIN_BFLOAT16_TO_FLOAT32 &&
		       arguments.width == TEST_WIDTH &&
		       arguments.source_row_stride == TEST_WIDTH &&
		       arguments.output_row_stride == TEST_WIDTH,
	       "attention partial upcast, geometry and element strides");
	description.routed_sums =
		(struct linear_range){ TEST_CAPACITY, TEST_CAPACITY };
	description.routed_row_stride = TEST_WIDTH;
	status = tp2_join_prepare(&description, TEST_ARENA_BYTES, TEST_HANDLE,
				  &launch);
	memcpy(&arguments, launch.argument, sizeof(arguments));
	expect(status == STATUS_OK && arguments.has_routed_sums == 1 &&
		       arguments.routed_sums_offset == TEST_CAPACITY &&
		       arguments.routed_row_stride == TEST_WIDTH,
	       "FFN FP32 routed sum plus shared BF16 input");
	description.operation = TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16;
	status = tp2_join_prepare(&description, TEST_ARENA_BYTES, TEST_HANDLE,
				  &launch);
	memcpy(&arguments, launch.argument, sizeof(arguments));
	expect(status == STATUS_OK && arguments.has_routed_sums == 1 &&
		       arguments.operation == TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16,
	       "BF16 shared plus FP32 routed sum, one round, BF16 out (the ring's packed rows)");
	description = description_make();
	description.operation = TP2_JOIN_FLOAT32_TO_BFLOAT16;
	status = tp2_join_prepare(&description, TEST_ARENA_BYTES, TEST_HANDLE,
				  &launch);
	memcpy(&arguments, launch.argument, sizeof(arguments));
	expect(status == STATUS_OK &&
		       arguments.operation == TP2_JOIN_FLOAT32_TO_BFLOAT16,
	       "global FP32 result to one final BF16 round");
	description.operation = TP2_JOIN_FLOAT32_COPY;
	status = tp2_join_prepare(&description, TEST_ARENA_BYTES, TEST_HANDLE,
				  &launch);
	memcpy(&arguments, launch.argument, sizeof(arguments));
	expect(status == STATUS_OK &&
		       arguments.operation == TP2_JOIN_FLOAT32_COPY,
	       "local-only collective stub preserves FP32 words");
	description.operation = TP2_JOIN_BFLOAT16_COPY;
	description.width = description.source_row_stride = TEST_HEAD_HALF;
	description.output_row_stride = TEST_CANONICAL_HEADS;
	status = tp2_join_prepare(&description, TEST_ARENA_BYTES, TEST_HANDLE,
				  &launch);
	memcpy(&arguments, launch.argument, sizeof(arguments));
	expect(status == STATUS_OK && arguments.width == TEST_HEAD_HALF &&
		       arguments.output_row_stride == TEST_CANONICAL_HEADS,
	       "compact head rows scatter into canonical rows");
	description.source_row_stride = TEST_CANONICAL_HEADS;
	description.output_row_stride = TEST_HEAD_HALF;
	status = tp2_join_prepare(&description, TEST_ARENA_BYTES, TEST_HANDLE,
				  &launch);
	expect(status == STATUS_OK,
	       "canonical attention rows gather into compact groups");
	for (uint32_t byte = launch.argument_byte_count;
	     byte < LAUNCH_ARGUMENT_BYTES; byte++)
		expect(launch.argument[byte] == 0,
		       "the unused argument suffix stays zero");
}

static void parity_slots(void)
{
	struct tp2_join_description description = description_make();
	description.step = (struct linear_range){ TEST_STEP_OFFSET,
						  STEP_BLOCK_BYTES_VALUE };
	description.output_slot_stride = TEST_SLOT_STRIDE;
	struct kernel_launch launch = { 0 };
	struct tp2_join_arguments arguments = { 0 };
	result_status status = tp2_join_prepare(&description, TEST_ARENA_BYTES,
						TEST_HANDLE, &launch);
	memcpy(&arguments, launch.argument, sizeof(arguments));
	expect(status == STATUS_OK &&
		       arguments.output_slot_stride == TEST_SLOT_STRIDE &&
		       arguments.source_slot_stride == 0 &&
		       arguments.step_offset == TEST_STEP_OFFSET,
	       "publication output follows step parity without changing static source");
	description.source_slot_stride = TEST_SLOT_STRIDE;
	description.output_slot_stride = 0;
	description.operation = TP2_JOIN_FLOAT32_TO_BFLOAT16;
	status = tp2_join_prepare(&description, TEST_ARENA_BYTES, TEST_HANDLE,
				  &launch);
	memcpy(&arguments, launch.argument, sizeof(arguments));
	expect(status == STATUS_OK &&
		       arguments.source_slot_stride == TEST_SLOT_STRIDE &&
		       arguments.output_slot_stride == 0,
	       "global result read follows parity without moving BF16 output");
	struct tp2_join_description interleaved = description;
	interleaved.operation = TP2_JOIN_FLOAT32_COPY;
	interleaved.output_slot_stride = TEST_SLOT_STRIDE;
	interleaved.output.byte_offset = TEST_INTERLEAVED_OFFSET;
	status = tp2_join_prepare(&interleaved, TEST_ARENA_BYTES, TEST_HANDLE,
				  &launch);
	expect(status == STATUS_OK,
	       "interleaved source/output capacities admit disjoint parity spans");
	description.source.byte_count = TEST_SLOT_STRIDE;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"the second source slot must fit");
	description = description_make();
	description.step = (struct linear_range){ TEST_STEP_OFFSET,
						  STEP_BLOCK_BYTES_VALUE };
	description.output_slot_stride = TEST_SLOT_STRIDE;
	description.output.byte_count = TEST_SLOT_STRIDE;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"the second output slot must fit");
	description.output.byte_count = TEST_CAPACITY;
	description.step.byte_count = TEST_STEP_SHORT_BYTES;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"the whole step block must fit");
	description.step =
		(struct linear_range){ TEST_OUTPUT_OFFSET + TEST_SLOT_STRIDE,
				       STEP_BLOCK_BYTES_VALUE };
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"step cannot alias the second output slot");
	description.step = (struct linear_range){ TEST_STEP_OFFSET,
						  STEP_BLOCK_BYTES_VALUE };
	description.output_slot_stride = TEST_BAD_STRIDE;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"unaligned or overlapping slots refused");
	description.output_slot_stride = UINT64_MAX;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"slot offset overflow refused before addition");
}

static void refusals(void)
{
	struct tp2_join_description description = description_make();
	refused(NULL, TEST_ARENA_BYTES, TEST_HANDLE,
		"missing description refused");
	refused(&description, TEST_ARENA_BYTES, BACKEND_NO_KERNEL,
		"unprepared kernel refused");
	result_status status = tp2_join_prepare(&description, TEST_ARENA_BYTES,
						TEST_HANDLE, NULL);
	expect(status == STATUS_INVALID_ARGUMENT, "missing launch refused");
	description.width = 0;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"zero width refused");
	description = description_make();
	description.row_count = 0;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"zero rows refused");
	description.row_count = LINEAR_WIDE_MOST_ROWS + 1u;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"too many rows refused");
	description = description_make();
	description.operation = TEST_UNKNOWN_OPERATION;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"unknown conversion refused");
	description = description_make();
	description.source_row_stride = TEST_WIDTH - 1u;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"overlapping source rows refused");
	description = description_make();
	description.output_row_stride = TEST_WIDTH - 1u;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"overlapping destination rows refused");
	description = description_make();
	description.source_row_stride = UINT32_MAX;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"strided span beyond capacity refused");
	description = description_make();
	description.source.byte_count = TEST_WIDTH;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"short source refused");
	description = description_make();
	description.output.byte_offset = 1;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"unaligned output refused");
	description = description_make();
	description.output = description.source;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"in-place conversion refused");
	description = description_make();
	description.output.byte_count = UINT64_MAX;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"arena extent overflow refused");
	description = description_make();
	description.routed_sums.byte_offset = TEST_CAPACITY;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"absent routed sum has no ignored offset");
	description = description_make();
	description.routed_row_stride = TEST_WIDTH;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"absent routed sum has no ignored stride");
	description.routed_sums =
		(struct linear_range){ TEST_CAPACITY, TEST_CAPACITY };
	description.operation = TP2_JOIN_BFLOAT16_COPY;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"raw copies never add routed sums");
	description = description_make();
	description.step = (struct linear_range){ TEST_STEP_OFFSET,
						  STEP_BLOCK_BYTES_VALUE };
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"static operations have no ignored step block");
	description = description_make();
	description.reserved_zero = 1;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE,
		"reserved fields refused");
}

int main(void)
{
	kernel_admission();
	operations();
	parity_slots();
	refusals();
	if (failures == 0)
		(void)puts(
			"tp2_join_test: independent binding, four operations, "
			"strided rows, parity and refusals passed");
	return failures != 0;
}
