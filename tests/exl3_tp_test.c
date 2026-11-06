/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU-only independent-pass admission; no real backend or device operations. */
#include "anemone/exl3_tp.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_OUTER_SIZE 5120u
#define TEST_INNER_SIZE 2304u
#define TEST_PAIR_COUNT 6u
#define TEST_ARENA_BYTES (96ull << 20u)
#define TEST_WEIGHT_OFFSET 1024u
#define TEST_WEIGHT_BYTES (16ull << 20u)
#define TEST_INPUT_OFFSET (32ull << 20u)
#define TEST_OPERAND_OFFSET (48ull << 20u)
#define TEST_PRODUCT_OFFSET (64ull << 20u)
#define TEST_HIDDEN_OFFSET (80ull << 20u)
#define TEST_KERNEL_HANDLE 7u
#define TEST_ACTIVATION_LIMIT 7.0f
#define TEST_SENTINEL_BYTE 0xa5

/* The real backend keeps this opaque. The fake only checks pointer identity. */
struct device_card {
	uint32_t unused;
};

static const char *expected_module;
static const char *expected_kernel;
static uint32_t expected_argument_bytes;
static uint32_t prepare_calls;
static uint32_t fake_kernel = TEST_KERNEL_HANDLE;
static int callback_failures;

MUST_CHECK static result_status
fake_prepare(struct device_card *card, const char *module, const char *kernel,
	     uint32_t argument_bytes, uint32_t *kernel_out)
{
	if (card == NULL || strcmp(module, expected_module) != 0 ||
	    strcmp(kernel, expected_kernel) != 0 ||
	    argument_bytes != expected_argument_bytes) {
		callback_failures++;
		return STATUS_INVALID_ARGUMENT;
	}
	prepare_calls++;
	*kernel_out = fake_kernel;
	return STATUS_OK;
}

static int refusal_unchanged(struct bb_exl3_pass_description *description,
			     const struct backend_operations *backend,
			     struct device_card *card,
			     result_status expected_status)
{
	struct kernel_launch launch;
	memset(&launch, TEST_SENTINEL_BYTE, sizeof(launch));
	const struct kernel_launch sentinel = launch;
	const result_status status =
		bb_exl3_pass_prepare(description, backend, card, &launch);
	return status != expected_status ||
	       memcmp(&launch, &sentinel, sizeof(launch)) != 0;
}

int main(void)
{
	const struct backend_operations backend = { .kernel_prepare =
							    fake_prepare };
	struct device_card card = { 0 };
	struct bb_exl3_output_shard_plan plan;
	if (bb_exl3_output_shard_prepare(TEST_OUTER_SIZE, TEST_INNER_SIZE,
					 EXL3_MAXIMUM_BITS, BB_EXL3_TP_RANKS, 1,
					 BB_EXL3_REFERENCE_ORDER,
					 &plan) != STATUS_OK) {
		return 1;
	}
	struct bb_exl3_pass_description description = {
		.plan = &plan,
		.record = { .jobs_offset = 0,
			    .input_offset = TEST_INPUT_OFFSET,
			    .operands_offset = TEST_OPERAND_OFFSET,
			    .products_offset = TEST_PRODUCT_OFFSET,
			    .hidden_offset = TEST_HIDDEN_OFFSET,
			    .output_offset = TEST_PRODUCT_OFFSET,
			    .pair_count = TEST_PAIR_COUNT,
			    .pairs_per_row = TEST_PAIR_COUNT,
			    .job_count = 1,
			    .bits_per_weight = EXL3_MAXIMUM_BITS,
			    .activation_limit = TEST_ACTIVATION_LIMIT },
		.arena_bytes = TEST_ARENA_BYTES,
		.weight_offset = TEST_WEIGHT_OFFSET,
		.weight_bytes = TEST_WEIGHT_BYTES
	};
	const char *const names[BB_EXL3_PASSES] = {
		"exl3_input", "exl3_gate_up_k3",
		BB_EXL3_SHARD_ACTIVATION_KERNEL, "exl3_down_k3", "exl3_finish"
	};
	int failures = 0;
	for (uint32_t pass = 0; pass < BB_EXL3_PASSES; pass++) {
		description.pass = (enum bb_exl3_pass)pass;
		const bool down = pass >= BB_EXL3_DOWN_PASS;
		description.record.input_size = down ? plan.down.output_count :
						       plan.gate_up.input_count;
		description.record.inner_size =
			down ? plan.down.input_count :
			       plan.gate_up.output_count;
		expected_module = EXL3_KERNEL_FILE;
		expected_kernel = names[pass];
		expected_argument_bytes =
			pass == BB_EXL3_ACTIVATE_PASS ?
				BB_EXL3_SHARD_ACTIVATION_BYTES :
				EXL3_EXPERT_RECORD_BYTES;
		struct kernel_launch launch;
		if (bb_exl3_pass_prepare(&description, &backend, &card,
					 &launch) != STATUS_OK) {
			return 1;
		}
		const bool product = pass == BB_EXL3_GATE_UP_PASS ||
				     pass == BB_EXL3_DOWN_PASS;
		failures += launch.kernel != TEST_KERNEL_HANDLE;
		failures += launch.block[0] !=
			    (product ? EXL3_PRODUCT_SLICES * EXL3_WARP :
				       EXL3_TRANSFORM);
		if (pass == BB_EXL3_ACTIVATE_PASS) {
			struct bb_exl3_shard_activation_record arguments;
			memcpy(&arguments, launch.argument, sizeof(arguments));
			failures += arguments.down_input_size !=
				    TEST_INNER_SIZE;
			failures += arguments.down_output_size !=
				    plan.down.output_count;
			failures += arguments.down_input_begin !=
				    plan.gate_up.output_begin;
		}
	}
	/* The down pass must reserve all 2304 intermediate channels per pair. */
	description.pass = BB_EXL3_DOWN_PASS;
	description.record.operands_offset =
		TEST_ARENA_BYTES -
		(uint64_t)TEST_PAIR_COUNT * TEST_INNER_SIZE * EXL3_HALF_BYTES +
		EXL3_TILE;
	failures += refusal_unchanged(&description, &backend, &card,
				      STATUS_INVALID_ARGUMENT);
	description.record.operands_offset = TEST_WEIGHT_OFFSET;
	failures += refusal_unchanged(&description, &backend, &card,
				      STATUS_INVALID_ARGUMENT);
	description.record.operands_offset = TEST_OPERAND_OFFSET;
	description.record.activation_limit = INFINITY;
	failures += refusal_unchanged(&description, &backend, &card,
				      STATUS_INVALID_ARGUMENT);
	description.record.activation_limit = TEST_ACTIVATION_LIMIT;
	plan.product_slices = EXL3_RELAXED_SLICES;
	failures += refusal_unchanged(&description, &backend, &card,
				      STATUS_INVALID_ARGUMENT);
	plan.product_slices = EXL3_PRODUCT_SLICES;
	/* Explicit optional order changes dispatch and threads, never by default. */
	plan.order = BB_EXL3_EIGHT_PART_ORDER;
	plan.product_slices = EXL3_RELAXED_SLICES;
	expected_module = EXL3_KERNEL_FILE;
	expected_kernel = "exl3_down_k3_s8";
	expected_argument_bytes = EXL3_EXPERT_RECORD_BYTES;
	struct kernel_launch optional_launch;
	const result_status optional_status = bb_exl3_pass_prepare(
		&description, &backend, &card, &optional_launch);
	failures += optional_status != STATUS_OK;
	if (optional_status == STATUS_OK) {
		failures += optional_launch.block[0] !=
			    EXL3_RELAXED_SLICES * EXL3_WARP;
	}
	fake_kernel = BACKEND_NO_KERNEL;
	failures += refusal_unchanged(&description, &backend, &card,
				      STATUS_UNSUPPORTED);
	fake_kernel = TEST_KERNEL_HANDLE;
	expected_module = EXL3_KERNEL_FILE;
	expected_kernel = EXL3_ROW_SUM;
	expected_argument_bytes = EXPERT_ROW_SUM_ARGUMENT_BYTES;
	struct expert_row_sum_arguments row_sum = {
		.pair_output_offset = TEST_PRODUCT_OFFSET,
		.row_output_offset = TEST_HIDDEN_OFFSET,
		.width = plan.down.output_count,
		.pairs_per_row = TEST_PAIR_COUNT
	};
	struct kernel_launch row_launch;
	const result_status row_status = bb_exl3_row_sum_prepare(
		&row_sum, 1, TEST_ARENA_BYTES, &backend, &card, &row_launch);
	failures += row_status != STATUS_OK;
	if (row_status == STATUS_OK) {
		failures += row_launch.grid[0] !=
			    plan.down.output_count / EXL3_TRANSFORM;
		failures += row_launch.grid[1] != 1;
		const struct kernel_launch unchanged = row_launch;
		row_sum.row_output_offset = row_sum.pair_output_offset;
		failures += bb_exl3_row_sum_prepare(&row_sum, 1,
						    TEST_ARENA_BYTES, &backend,
						    &card, &row_launch) !=
			    STATUS_INVALID_ARGUMENT;
		failures += memcmp(&row_launch, &unchanged,
				   sizeof(row_launch)) != 0;
	}
	/* Product selection changes only the module/name and x geometry.
	 * The complete 80-byte arithmetic record stays exact. */
	description.product_tiles = BB_EXL3_PRODUCT_FOUR_TILES;
	expected_module = BB_EXL3_PRODUCT_KERNEL_FILE;
	expected_kernel = "exl3_c_down_k3_s8_t4_g1";
	expected_argument_bytes = EXL3_EXPERT_RECORD_BYTES;
	struct kernel_launch product_launch;
	unsigned char expected_record[sizeof(description.record)];
	memcpy(expected_record, &description.record, sizeof(expected_record));
	const result_status product_status = bb_exl3_pass_prepare(
		&description, &backend, &card, &product_launch);
	failures += product_status != STATUS_OK;
	if (product_status == STATUS_OK) {
		failures += product_launch.grid[0] !=
			    plan.down.output_count / EXL3_TILE /
				    BB_EXL3_PRODUCT_FOUR_TILES;
		failures += product_launch.block[0] !=
			    EXL3_RELAXED_SLICES * EXL3_WARP;
		failures += memcmp(product_launch.argument, expected_record,
				   sizeof(expected_record)) != 0;
	}
	description.product_tiles = 1;
	failures += refusal_unchanged(&description, &backend, &card,
				      STATUS_UNSUPPORTED);
	plan.order = BB_EXL3_REFERENCE_ORDER;
	plan.product_slices = EXL3_PRODUCT_SLICES;
	expected_kernel = "exl3_c_down_k3_t1_g1";
	const result_status exact_status = bb_exl3_pass_prepare(
		&description, &backend, &card, &product_launch);
	failures += exact_status != STATUS_OK;
	if (exact_status == STATUS_OK) {
		failures += product_launch.grid[0] !=
			    plan.down.output_count / EXL3_TILE;
		failures += product_launch.block[0] !=
			    EXL3_PRODUCT_SLICES * EXL3_WARP;
		failures += memcmp(product_launch.argument, expected_record,
				   sizeof(expected_record)) != 0;
	}
	failures += callback_failures;
	printf("EXL3 independent pass admission: %s (%u fake prepares)\n",
	       failures == 0 ? "PASS" : "FAIL", prepare_calls);
	return failures == 0 ? 0 : 1;
}
