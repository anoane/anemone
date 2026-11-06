/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/official_router_order.h"

#define TEST_ARENA_BYTES (128ull << 20u)
#define TEST_WEIGHTS_OFFSET (16ull << 20u)
#define TEST_WEIGHTS_BYTES (64ull << 20u)
#define TEST_STEP_OFFSET 4096u
#define TEST_SCORES_OFFSET (1ull << 16u)
#define TEST_BIAS_OFFSET (2ull << 16u)
#define TEST_OFFSETS_OFFSET (3ull << 16u)
#define TEST_AFFINITIES_OFFSET (4ull << 16u)
#define TEST_JOBS_OFFSET (5ull << 16u)
#define TEST_HANDLE 17u
#define TEST_UNTOUCHED 0xa5
#define TEST_INNER_SIZE 2304u
#define TEST_WEIGHT_BITS 4u
#define TEST_UNADMITTED_ROWS 2u
#define TEST_TABLE_ENTRIES (MODEL_EXPERTS * MODEL_EXPERT_MATRICES)

static int failures;
static uint64_t expert_offsets[TEST_TABLE_ENTRIES];
static struct linear_routed_experts_description experts;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct router_description description_of(void)
{
	for (uint32_t entry = 0; entry < TEST_TABLE_ENTRIES; entry++)
		expert_offsets[entry] = TEST_WEIGHTS_OFFSET;
	experts = (struct linear_routed_experts_description){
		.jobs = { TEST_JOBS_OFFSET,
			  (uint64_t)MODEL_EXPERTS_PER_ROW * EXPERT_JOB_BYTES },
		.expert_weights = { TEST_WEIGHTS_OFFSET, TEST_WEIGHTS_BYTES },
		.gate_up_shape = { TEST_INNER_SIZE, MODEL_HIDDEN,
				   TEST_WEIGHT_BITS, 0 },
		.down_shape = { MODEL_HIDDEN, TEST_INNER_SIZE, TEST_WEIGHT_BITS,
				0 },
		.weight_format = WEIGHT_FORMAT_E2M1_SCALED,
		.job_count = MODEL_EXPERTS_PER_ROW,
		.row_count = 1,
		.pairs_per_row = MODEL_EXPERTS_PER_ROW
	};
	return (struct router_description){
		.experts = &experts,
		.expert_offsets = expert_offsets,
		.step = { TEST_STEP_OFFSET, STEP_BLOCK_BYTES_VALUE },
		.scores = { TEST_SCORES_OFFSET, MODEL_ROUTER_ROW_BYTES },
		.bias = { TEST_BIAS_OFFSET, MODEL_ROUTER_ROW_BYTES },
		.offsets = { TEST_OFFSETS_OFFSET, MODEL_EXPERT_OFFSETS_BYTES },
		.affinities = { TEST_AFFINITIES_OFFSET,
				MODEL_ROUTER_ROW_BYTES },
		.order = LINEAR_ORDER_REFERENCE,
		.expert_count = MODEL_EXPERTS,
		.pick_count = MODEL_EXPERTS_PER_ROW
	};
}

static void refused(const struct router_description *description,
		    uint64_t arena_bytes, uint32_t kernel)
{
	struct kernel_launch launch;
	memset(&launch, TEST_UNTOUCHED, sizeof(launch));
	const struct kernel_launch before = launch;
	expect(official_router_order_prepare(description, arena_bytes, kernel,
					     &launch) ==
		       STATUS_INVALID_ARGUMENT,
	       "unsupported or unsafe router descriptor refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal leaves prepared launch unchanged");
}

int main(void)
{
	struct router_description description = description_of();
	struct kernel_launch launch, original;
	const struct router_kernels_record kernels = { .select = TEST_HANDLE };
	const result_status original_status = router_select_prepare(
		&description, TEST_ARENA_BYTES, &kernels, &original);
	const result_status status = official_router_order_prepare(
		&description, TEST_ARENA_BYTES, TEST_HANDLE, &launch);
	expect(status == STATUS_OK && original_status == STATUS_OK,
	       "existing original-weight router description admitted");
	if (status == STATUS_OK && original_status == STATUS_OK) {
		expect(memcmp(original.argument, launch.argument,
			      ROUTER_SELECT_ARGUMENT_BYTES) == 0,
		       "post-select pass preserves every original range and job contract");
		expect(launch.grid[0] == 1 && launch.grid[1] == 1 &&
			       launch.block[0] == 1 &&
			       launch.dynamic_shared_byte_count == 0 &&
			       launch.argument_byte_count ==
				       ROUTER_SELECT_ARGUMENT_BYTES,
		       "serial in-place pass needs no additional arena scratch");
	}
	refused(NULL, TEST_ARENA_BYTES, TEST_HANDLE);
	refused(&description, TEST_ARENA_BYTES, BACKEND_NO_KERNEL);
	refused(&description, 1, TEST_HANDLE);
	description.order = LINEAR_ORDER_FIXED;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of();
	experts.row_count = TEST_UNADMITTED_ROWS;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of();
	description.expert_count = MODEL_DRAFT_EXPERTS;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of();
	description.affinities = description.bias;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of();
	description.affinities.byte_count--;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	(void)printf("official_router_order_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
