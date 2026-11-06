/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Cold bounds for the component-only EP2 filter. No backend callback or CUDA context is used. */
#include <stdio.h>
#include <string.h>
#include "anemone/ep_filter.h"
#include "anemone/model_layout.h"

#define TEST_ROWS 6u
#define TEST_ARENA (4ull << 20u)
#define TEST_HIDDEN_OFFSET (1ull << 20u)
#define TEST_OUTPUT_OFFSET (2ull << 20u)
#define TEST_DUMMY_OFFSET (3ull << 20u)
#define TEST_HANDLE 41u
#define TEST_POISON 0xA5

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

static void refused(const struct ep_filter_description *description,
		    uint64_t arena, uint32_t handle)
{
	struct kernel_launch launch;
	unsigned char before[sizeof(launch)];
	memset(&launch, TEST_POISON, sizeof(launch));
	memcpy(before, &launch, sizeof(before));
	expect(ep_filter_prepare(description, arena, handle, &launch) ==
			       STATUS_INVALID_ARGUMENT &&
		       memcmp(before, &launch, sizeof(before)) == 0,
	       "refusal preserves the entire launch");
}

int main(void)
{
	const uint32_t pairs = TEST_ROWS * MODEL_EXPERTS_PER_ROW;
	const struct ep_filter_description valid = {
		{ 0, linear_jobs_byte_count(pairs) },
		{ TEST_HIDDEN_OFFSET,
		  linear_values_byte_count(pairs, EP_FILTER_HIDDEN_WIDTH) },
		{ TEST_OUTPUT_OFFSET,
		  linear_values_byte_count(pairs, MODEL_HIDDEN) },
		TEST_DUMMY_OFFSET,
		TEST_ROWS
	};
	struct kernel_launch launch;
	struct ep_filter_arguments arguments;
	const result_status status =
		ep_filter_prepare(&valid, TEST_ARENA, TEST_HANDLE, &launch);
	expect(status == STATUS_OK, "full routed BF16 arrays prepare");
	if (status == STATUS_OK) {
		memcpy(&arguments, launch.argument, sizeof(arguments));
		expect(arguments.job_count == pairs &&
			       arguments.pair_count == pairs &&
			       arguments.hidden_width ==
				       EP_FILTER_HIDDEN_WIDTH &&
			       arguments.output_width == MODEL_HIDDEN &&
			       arguments.unowned_gate_offset ==
				       TEST_DUMMY_OFFSET,
		       "the filter uses full BF16 pair geometry");
	}
	struct ep_filter_description changed = valid;
	changed.hidden.byte_count--;
	refused(&changed, TEST_ARENA, TEST_HANDLE);
	changed = valid;
	changed.output.byte_offset = changed.hidden.byte_offset;
	refused(&changed, TEST_ARENA, TEST_HANDLE);
	changed = valid;
	changed.jobs.byte_offset++;
	refused(&changed, TEST_ARENA, TEST_HANDLE);
	changed = valid;
	changed.row_count = 0;
	refused(&changed, TEST_ARENA, TEST_HANDLE);
	changed.row_count = LINEAR_MAXIMUM_ROWS + 1u;
	refused(&changed, TEST_ARENA, TEST_HANDLE);
	changed = valid;
	changed.unowned_gate_offset = TEST_ARENA;
	refused(&changed, TEST_ARENA, TEST_HANDLE);
	changed = valid;
	changed.unowned_gate_offset++;
	refused(&changed, TEST_ARENA, TEST_HANDLE);
	refused(&valid, TEST_OUTPUT_OFFSET, TEST_HANDLE);
	refused(&valid, TEST_ARENA, BACKEND_NO_KERNEL);
	(void)puts(failures == 0 ? "ep_filter_test PASS" :
				   "ep_filter_test FAILED");
	return failures == 0 ? 0 : 1;
}
