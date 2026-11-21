/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "models/deepseek_v41_flash/official/ds41f_official_attention.h"

#define TEST_ARENA_BYTES (64ull << 20u)
#define TEST_STEP_OFFSET 4096u
#define TEST_QUERY_OFFSET (1ull << 16u)
#define TEST_RING_OFFSET (1ull << 20u)
#define TEST_POOL_OFFSET (8ull << 20u)
#define TEST_SELECTION_OFFSET (32ull << 20u)
#define TEST_SINKS_OFFSET (33ull << 20u)
#define TEST_ROTATION_OFFSET (34ull << 20u)
#define TEST_OUTPUT_OFFSET (35ull << 20u)
#define TEST_CONTEXT 16384u
#define TEST_HANDLE 17u
#define TEST_UNTOUCHED 0xa5
#define TEST_UNADMITTED_ROWS 2u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct ds41f_attention_core_description description_of(uint32_t ratio)
{
	return (struct ds41f_attention_core_description){
		.step = { TEST_STEP_OFFSET,
			  STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE },
		.queries = { TEST_QUERY_OFFSET, DS41F_HEADS_ROW_BYTES },
		.ring = { TEST_RING_OFFSET,
			  (uint64_t)DS41F_RING_SLOTS * DS41F_STATE_ROW_BYTES },
		.pool = { TEST_POOL_OFFSET,
			  ratio == 0 ? 0 :
				       (uint64_t)TEST_CONTEXT / ratio *
					       DS41F_STATE_ROW_BYTES },
		.selection = { TEST_SELECTION_OFFSET,
			       DS41F_SELECTION_ROW_BYTES },
		.sinks = { TEST_SINKS_OFFSET,
			   (uint64_t)DS41F_HEADS_VALUE * sizeof(float) },
		.rotation = { TEST_ROTATION_OFFSET, DS41F_ROTATION_ROW_BYTES },
		.output = { TEST_OUTPUT_OFFSET, DS41F_HEADS_ROW_BYTES },
		.row_count = 1,
		.session_capacity = 1,
		.context_token_limit = TEST_CONTEXT,
		.pool_ratio = ratio,
		.head_count = DS41F_HEADS_VALUE,
		.order = LINEAR_ORDER_REFERENCE,
		.arithmetic_class = ARITHMETIC_CLASS_SM_80,
		.window_mode = DS41F_ATTENTION_CORE_WINDOW_CAUSAL
	};
}

static void refused(const struct ds41f_attention_core_description *description,
		    uint32_t operand_types, uint64_t arena_bytes,
		    uint32_t kernel)
{
	struct kernel_launch launch;
	memset(&launch, TEST_UNTOUCHED, sizeof(launch));
	const struct kernel_launch before = launch;
	expect(ds41f_official_attention_prepare(description, operand_types,
						arena_bytes, kernel, &launch) ==
		       STATUS_INVALID_ARGUMENT,
	       "unsupported or unsafe core descriptor refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal leaves prepared launch unchanged");
}

int main(void)
{
	static const uint32_t ratios[] = { 0, DS41F_DIRECT_RATIO,
					   DS41F_POOLED_RATIO };
	for (uint32_t index = 0; index < sizeof(ratios) / sizeof(ratios[0]);
	     index++) {
		const uint32_t ratio = ratios[index];
		const struct ds41f_attention_core_description description =
			description_of(ratio);
		for (uint32_t types = 0;
		     types <= DS41F_OFFICIAL_ATTENTION_OPERAND_TYPES; types++) {
			struct kernel_launch launch;
			const result_status status =
				ds41f_official_attention_prepare(
					&description, types, TEST_ARENA_BYTES,
					TEST_HANDLE, &launch);
			expect(status == STATUS_OK,
			       "independent finite carrier storage types admitted");
			if (status != STATUS_OK)
				continue;
			struct ds41f_official_attention_arguments arguments;
			memcpy(&arguments, launch.argument, sizeof(arguments));
			expect(launch.kernel == TEST_HANDLE &&
				       launch.grid[0] == 1 &&
				       launch.grid[1] == DS41F_HEADS_VALUE &&
				       launch.block[0] ==
					       DS41F_OFFICIAL_ATTENTION_THREADS &&
				       launch.dynamic_shared_byte_count ==
					       DS41F_OFFICIAL_ATTENTION_SHARED_BYTES,
			       "one independent full-head CTA grid with bounded private stage");
			expect(arguments.operand_types == types &&
				       arguments.reserved_zero == 0 &&
				       arguments.source.scores_offset == 0 &&
				       arguments.source.queries_offset ==
					       TEST_QUERY_OFFSET &&
				       arguments.source.output_offset ==
					       TEST_OUTPUT_OFFSET &&
				       arguments.source.ring_session_stride ==
					       (uint64_t)DS41F_RING_SLOTS *
						       DS41F_STATE_ROW_BYTES,
			       "physical ring, output and storage contracts retained without global scratch");
			expect(arguments.source.rotation_offset ==
				       TEST_ROTATION_OFFSET +
					       (ratio == 0 ?
							0 :
							DS41F_ROTATION_ROW_BYTES /
								DS41F_ROTATION_USES),
			       "theta use follows persistent pool ownership");
		}
	}
	struct ds41f_attention_core_description description =
		description_of(DS41F_DIRECT_RATIO);
	refused(NULL, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	refused(&description, DS41F_OFFICIAL_ATTENTION_OPERAND_TYPES + 1u,
		TEST_ARENA_BYTES, TEST_HANDLE);
	refused(&description, 1, TEST_ARENA_BYTES, BACKEND_NO_KERNEL);
	refused(&description, 1, 1, TEST_HANDLE);
	description.queries = description.output;
	refused(&description, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(0);
	description.row_count = TEST_UNADMITTED_ROWS;
	refused(&description, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(0);
	description.session_capacity = TEST_UNADMITTED_ROWS;
	refused(&description, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(0);
	description.first_head = DS41F_ATTENTION_CORE_HEADS_PER_BLOCK;
	refused(&description, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(0);
	description.order = LINEAR_ORDER_FIXED;
	refused(&description, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(0);
	description.output_compact = 1;
	refused(&description, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(0);
	description.cone_layer = DS41F_DECODER_FIRST_LAYER;
	refused(&description, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(0);
	description.window_mode = DS41F_ATTENTION_CORE_WINDOW_BLOCK;
	refused(&description, 1, TEST_ARENA_BYTES, TEST_HANDLE);
	(void)printf("official_attention_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
