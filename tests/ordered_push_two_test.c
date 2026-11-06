/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Three-phase admission, unchanged arguments, overflow refusal and kernel binding for owner-slice push. */
#include <stdio.h>
#include <string.h>

#include "anemone/ordered_push_two.h"

#define TWO_TEST_ELEMENTS 5120u
#define TWO_TEST_RANKS 5u
#define TWO_TEST_STEP_OFFSET 0u
#define TWO_TEST_TABLE_OFFSET 128u
#define TWO_TEST_REGION_OFFSET 1024u
#define TWO_TEST_SIGNAL_BYTES                \
	((uint64_t)ORDERED_PUSH_TWO_PHASES * \
	 ORDERED_COLLECTIVE_MAXIMUM_RANKS * ORDERED_COLLECTIVE_LINE_BYTES)
#define TWO_TEST_ARRAY_BYTES \
	((uint64_t)ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS * sizeof(float))
#define TWO_TEST_INPUT_OFFSET TWO_TEST_SIGNAL_BYTES
#define TWO_TEST_OUTPUT_OFFSET (TWO_TEST_INPUT_OFFSET + TWO_TEST_ARRAY_BYTES)
#define TWO_TEST_INBOX_OFFSET (TWO_TEST_OUTPUT_OFFSET + TWO_TEST_ARRAY_BYTES)
#define TWO_TEST_STRIDE          \
	(TWO_TEST_INBOX_OFFSET + \
	 ORDERED_COLLECTIVE_MAXIMUM_RANKS * TWO_TEST_ARRAY_BYTES)
#define TWO_TEST_ARENA_BYTES \
	(TWO_TEST_REGION_OFFSET + ORDERED_COLLECTIVE_SLOTS * TWO_TEST_STRIDE)
#define TWO_TEST_DEADLINE 1000000u
#define TWO_TEST_HANDLE 29u
#define TWO_TEST_POISON 0xA5

static int failures;
static bool kernel_failure;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

static MUST_CHECK result_status
two_test_kernel_prepare(struct device_card *card, const char *file,
			const char *name, uint32_t bytes, uint32_t *kernel_out)
{
	(void)card;
	expect(strcmp(file, KERNEL_FILE_ORDERED_PUSH_TWO) == 0 &&
		       strcmp(name, KERNEL_ORDERED_PUSH_TWO) == 0 &&
		       bytes == ORDERED_PUSH_ARGUMENT_BYTES,
	       "separate two-shot module and unchanged argument ABI");
	if (kernel_failure)
		return STATUS_UNSUPPORTED;
	*kernel_out = TWO_TEST_HANDLE;
	return STATUS_OK;
}

static struct ordered_push_description two_test_description(void)
{
	return (struct ordered_push_description){
		.arguments = { .peer_table_offset = TWO_TEST_TABLE_OFFSET,
			       .region_offset = TWO_TEST_REGION_OFFSET,
			       .input_offset = TWO_TEST_INPUT_OFFSET,
			       .output_offset = TWO_TEST_OUTPUT_OFFSET,
			       .inbox_offset = TWO_TEST_INBOX_OFFSET,
			       .step_offset = TWO_TEST_STEP_OFFSET,
			       .parity_stride = TWO_TEST_STRIDE,
			       .deadline_cycles = TWO_TEST_DEADLINE,
			       .inbox_rank_stride = TWO_TEST_ARRAY_BYTES,
			       .element_count = TWO_TEST_ELEMENTS,
			       .rank_count = TWO_TEST_RANKS },
		.arena_byte_count = TWO_TEST_ARENA_BYTES
	};
}

static void kernel_admission(void)
{
	const struct backend_operations backend = {
		.kernel_prepare = two_test_kernel_prepare
	};
	struct device_card *const card = (struct device_card *)(uintptr_t)1;
	uint32_t kernel = BACKEND_NO_KERNEL;
	expect(ordered_push_two_kernel_prepare(&backend, card, &kernel) ==
			       STATUS_OK &&
		       kernel == TWO_TEST_HANDLE,
	       "bind separate two-shot kernel");
	kernel_failure = true;
	expect(ordered_push_two_kernel_prepare(&backend, card, &kernel) ==
			       STATUS_UNSUPPORTED &&
		       kernel == TWO_TEST_HANDLE,
	       "kernel failure preserves output");
	expect(ordered_push_two_kernel_prepare(NULL, card, &kernel) ==
		       STATUS_INVALID_ARGUMENT,
	       "null backend refuses");
}

int main(void)
{
	kernel_admission();
	struct ordered_push_description description = two_test_description();
	struct kernel_launch launch;
	for (uint32_t count = 1; count <= ORDERED_COLLECTIVE_MAXIMUM_RANKS;
	     count++) {
		description.arguments.rank_count = count;
		for (uint32_t rank = 0; rank < count; rank++) {
			description.arguments.rank = rank;
			for (uint32_t format = ORDERED_PUSH_FP32;
			     format <= ORDERED_PUSH_BF16; format++) {
				description.arguments.payload_format = format;
				expect(ordered_push_two_prepare(
					       &description, TWO_TEST_HANDLE,
					       &launch) == STATUS_OK &&
					       launch.kernel ==
						       TWO_TEST_HANDLE &&
					       launch.argument_byte_count ==
						       ORDERED_PUSH_ARGUMENT_BYTES &&
					       memcmp(launch.argument,
						      &description.arguments,
						      sizeof(description
								     .arguments)) ==
						       0,
				       "valid owner-slice plan retains immutable record");
			}
		}
	}
	description = two_test_description();
	description.arguments.input_offset = (uint64_t)ORDERED_PUSH_PHASES *
					     TWO_TEST_RANKS *
					     ORDERED_COLLECTIVE_LINE_BYTES;
	expect(ordered_push_prepare(&description, TWO_TEST_HANDLE, &launch) ==
		       STATUS_OK,
	       "two-phase prefix is valid for one shot");
	memset(&launch, TWO_TEST_POISON, sizeof(launch));
	const struct kernel_launch before = launch;
	expect(ordered_push_two_prepare(&description, TWO_TEST_HANDLE,
					&launch) == STATUS_INVALID_ARGUMENT &&
		       memcmp(&launch, &before, sizeof(launch)) == 0,
	       "two-shot refuses overlap with third phase");
	description = two_test_description();
	description.arguments.inbox_rank_stride = UINT64_MAX;
	expect(ordered_push_two_prepare(&description, TWO_TEST_HANDLE,
					&launch) == STATUS_INVALID_ARGUMENT &&
		       memcmp(&launch, &before, sizeof(launch)) == 0,
	       "inherited overflow refuses and preserves output");
	expect(ordered_push_two_prepare(NULL, TWO_TEST_HANDLE, &launch) ==
		       STATUS_INVALID_ARGUMENT,
	       "null description refuses");
	(void)printf("ordered_push_two_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? 0 : 1;
}
