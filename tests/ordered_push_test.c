/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Push-launch admission, overflow/disjointness fuzzing and qualified writer-route fallback. */
#include <stdio.h>
#include <string.h>

#include "anemone/ordered_push.h"

#define PUSH_TEST_RANKS 5u
#define PUSH_TEST_ELEMENTS 5120u
#define PUSH_TEST_TABLE_OFFSET 128u
#define PUSH_TEST_REGION_OFFSET 512u
#define PUSH_TEST_INPUT_OFFSET 1280u
#define PUSH_TEST_ARRAY_BYTES (PUSH_TEST_ELEMENTS * sizeof(float))
#define PUSH_TEST_OUTPUT_OFFSET (PUSH_TEST_INPUT_OFFSET + PUSH_TEST_ARRAY_BYTES)
#define PUSH_TEST_INBOX_OFFSET (PUSH_TEST_OUTPUT_OFFSET + PUSH_TEST_ARRAY_BYTES)
#define PUSH_TEST_STRIDE \
	(PUSH_TEST_INBOX_OFFSET + PUSH_TEST_RANKS * PUSH_TEST_ARRAY_BYTES)
#define PUSH_TEST_ARENA_BYTES \
	(PUSH_TEST_REGION_OFFSET + ORDERED_COLLECTIVE_SLOTS * PUSH_TEST_STRIDE)
#define PUSH_TEST_DEADLINE 1000000ull
#define PUSH_TEST_HANDLE 17u
#define PUSH_TEST_POISON 0xA5
enum push_test_fault {
	PUSH_ZERO_RANKS,
	PUSH_EXCESS_RANKS,
	PUSH_EXCESS_RANK,
	PUSH_ZERO_ELEMENTS,
	PUSH_EXCESS_ELEMENTS,
	PUSH_BAD_FORMAT,
	PUSH_RESERVED,
	PUSH_ZERO_DEADLINE,
	PUSH_HOST_SELF,
	PUSH_HOST_EXCESS,
	PUSH_REGION_ALIGNMENT,
	PUSH_STRIDE_OVERFLOW,
	PUSH_TABLE_ALIGNMENT,
	PUSH_STEP_ALIGNMENT,
	PUSH_INBOX_OVERFLOW,
	PUSH_ZERO_INBOX,
	PUSH_SIGNAL_OVERLAP,
	PUSH_ARRAY_OVERLAP,
	PUSH_INBOX_RANGE,
	PUSH_CAPACITY,
	PUSH_TEST_FAULTS
};
#define PUSH_TEST_WRONG_ALIGNMENT 2u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

static struct ordered_push_description push_description(void)
{
	return (struct ordered_push_description){
		.arguments = { .peer_table_offset = PUSH_TEST_TABLE_OFFSET,
			       .region_offset = PUSH_TEST_REGION_OFFSET,
			       .input_offset = PUSH_TEST_INPUT_OFFSET,
			       .output_offset = PUSH_TEST_OUTPUT_OFFSET,
			       .inbox_offset = PUSH_TEST_INBOX_OFFSET,
			       .step_offset = 0,
			       .parity_stride = PUSH_TEST_STRIDE,
			       .deadline_cycles = PUSH_TEST_DEADLINE,
			       .inbox_rank_stride = PUSH_TEST_ARRAY_BYTES,
			       .element_count = PUSH_TEST_ELEMENTS,
			       .rank_count = PUSH_TEST_RANKS },
		.arena_byte_count = PUSH_TEST_ARENA_BYTES
	};
}

static void launch_admission(void)
{
	struct ordered_push_description description = push_description();
	struct kernel_launch launch;
	expect(ordered_push_prepare(&description, PUSH_TEST_HANDLE, &launch) ==
		       STATUS_OK,
	       "valid FP32 push");
	expect(launch.kernel == PUSH_TEST_HANDLE && launch.grid[0] == 1 &&
		       launch.block[0] == ORDERED_COLLECTIVE_THREADS &&
		       launch.argument_byte_count ==
			       ORDERED_PUSH_ARGUMENT_BYTES &&
		       memcmp(launch.argument, &description.arguments,
			      sizeof(description.arguments)) == 0,
	       "launch preserves immutable record");
	description.arguments.payload_format = ORDERED_PUSH_BF16;
	expect(ordered_push_prepare(&description, PUSH_TEST_HANDLE, &launch) ==
		       STATUS_OK,
	       "valid BF16 push");
	memset(&launch, PUSH_TEST_POISON, sizeof(launch));
	const struct kernel_launch absent_before = launch;
	expect(ordered_push_prepare(&description, BACKEND_NO_KERNEL, &launch) ==
			       STATUS_INVALID_ARGUMENT &&
		       memcmp(&launch, &absent_before, sizeof(launch)) == 0,
	       "absent kernel is refused without changing output");
	for (uint32_t fault = 0; fault < PUSH_TEST_FAULTS; fault++) {
		description = push_description();
		struct ordered_push_arguments *const arguments =
			&description.arguments;
		/* Explicit malformed records span every bound, storage region and graph-mask check. */
		switch (fault) {
		case PUSH_ZERO_RANKS:
			arguments->rank_count = 0;
			break;
		case PUSH_EXCESS_RANKS:
			arguments->rank_count =
				ORDERED_COLLECTIVE_MAXIMUM_RANKS + 1u;
			break;
		case PUSH_EXCESS_RANK:
			arguments->rank = PUSH_TEST_RANKS;
			break;
		case PUSH_ZERO_ELEMENTS:
			arguments->element_count = 0;
			break;
		case PUSH_EXCESS_ELEMENTS:
			arguments->element_count =
				ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS + 1u;
			break;
		case PUSH_BAD_FORMAT:
			arguments->payload_format = ORDERED_PUSH_BF16 + 1u;
			break;
		case PUSH_RESERVED:
			arguments->reserved_zero = 1;
			break;
		case PUSH_ZERO_DEADLINE:
			arguments->deadline_cycles = 0;
			break;
		case PUSH_HOST_SELF:
			arguments->incoming_host_mask = 1;
			break;
		case PUSH_HOST_EXCESS:
			arguments->incoming_host_mask = 1u << PUSH_TEST_RANKS;
			break;
		case PUSH_REGION_ALIGNMENT:
			arguments->region_offset++;
			break;
		case PUSH_STRIDE_OVERFLOW:
			arguments->parity_stride = UINT64_MAX;
			break;
		case PUSH_TABLE_ALIGNMENT:
			arguments->peer_table_offset++;
			break;
		case PUSH_STEP_ALIGNMENT:
			arguments->step_offset++;
			break;
		case PUSH_INBOX_OVERFLOW:
			arguments->inbox_rank_stride = UINT64_MAX;
			break;
		case PUSH_ZERO_INBOX:
			arguments->inbox_rank_stride = 0;
			break;
		case PUSH_SIGNAL_OVERLAP:
			arguments->input_offset = 0;
			break;
		case PUSH_ARRAY_OVERLAP:
			arguments->output_offset = arguments->input_offset;
			break;
		case PUSH_INBOX_RANGE:
			arguments->inbox_offset =
				UINT64_MAX - PUSH_TEST_WRONG_ALIGNMENT;
			break;
		case PUSH_CAPACITY:
			description.arena_byte_count--;
			break;
		default:
			break;
		}
		memset(&launch, PUSH_TEST_POISON, sizeof(launch));
		const struct kernel_launch before = launch;
		expect(ordered_push_prepare(&description, PUSH_TEST_HANDLE,
					    &launch) ==
				       STATUS_INVALID_ARGUMENT &&
			       memcmp(&launch, &before, sizeof(launch)) == 0,
		       "malformed push refuses without changing output");
	}
	for (uint64_t overlap = 0; overlap < STEP_BLOCK_BYTES_VALUE;
	     overlap++) {
		description = push_description();
		description.arguments.peer_table_offset = overlap;
		expect(ordered_push_prepare(&description, PUSH_TEST_HANDLE,
					    &launch) == STATUS_INVALID_ARGUMENT,
		       "step/table overlap fuzz refuses");
	}
}

static void routes_admission(void)
{
	struct card_description_record cards[PUSH_TEST_RANKS] = { 0 };
	uint32_t qualified[PUSH_TEST_RANKS], host[PUSH_TEST_RANKS];
	for (uint32_t rank = 0; rank < PUSH_TEST_RANKS; rank++)
		qualified[rank] = 1u << rank;
	struct ordered_collective_route_description description = {
		.cards = cards,
		.direct_reader_masks = qualified,
		.rank_count = PUSH_TEST_RANKS,
		.element_count = PUSH_TEST_ELEMENTS,
		.operation = ORDERED_ALL_REDUCE
	};
	expect(ordered_push_routes_prepare(&description, host) == STATUS_OK,
	       "denied writers use host");
	const uint32_t all = (1u << PUSH_TEST_RANKS) - 1u;
	for (uint32_t rank = 0; rank < PUSH_TEST_RANKS; rank++) {
		expect(host[rank] == (all & ~(1u << rank)),
		       "every denied incoming writer uses host");
		qualified[rank] = all;
	}
	expect(ordered_push_routes_prepare(&description, host) == STATUS_OK,
	       "qualified direct writers");
	for (uint32_t rank = 0; rank < PUSH_TEST_RANKS; rank++)
		expect(host[rank] == 0,
		       "no historical read-policy preference applied");
}

int main(void)
{
	launch_admission();
	routes_admission();
	(void)printf("ordered_push_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? 0 : 1;
}
