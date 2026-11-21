/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host bounds/ownership controls for the experimental masked peer copy. No backend/device calls. */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "engine/library/tensor_parallel/peer_row_copy.h"

#define TEST_ARENA_BYTES (8ull << 20u)
#define TEST_KERNEL 7u
#define TEST_CUT_BYTES (18612u * 4u)
#define TEST_FULL_ROW_BYTES (129280u * 4u)
#define TEST_SOURCE_OFFSET 4096u
#define TEST_DESTINATION_OFFSET 65536u
#define TEST_PEER_OFFSET 64u
#define TEST_STEP_OFFSET 128u
#define TEST_SENTINEL_BYTE 0x5Au
#define TEST_ROWS 8u /* the rows a description covers, and its ranges hold */
#define TEST_ROWS_PAST 9 /* a row more than the ranges hold */
#define TEST_PEER_BYTES 64u
#define TEST_STEP_BYTES 512u
#define TEST_SHORT_STRIDE 16 /* a source stride shorter than a row */
#define TEST_STRIDE_WRAP \
	15u /* a stride UINT64_MAX - 15: the range's end overflows */

static unsigned failures, accepted, rejected;

static struct head_offload_description description(uint32_t rows)
{
	return (struct head_offload_description){
		{ TEST_SOURCE_OFFSET, (uint64_t)(TEST_ROWS * TEST_CUT_BYTES) },
		{ TEST_PEER_OFFSET, TEST_PEER_BYTES },
		{ TEST_STEP_OFFSET, TEST_STEP_BYTES },
		{ TEST_DESTINATION_OFFSET,
		  (uint64_t)(TEST_ROWS * TEST_FULL_ROW_BYTES) },
		TEST_ARENA_BYTES,
		(uint64_t)TEST_CUT_BYTES,
		(uint64_t)TEST_FULL_ROW_BYTES,
		TEST_CUT_BYTES,
		rows,
		TEST_CUT_BYTES,
		0
	};
}

static void expect(bool value, const char *message)
{
	if (!value) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

static void refuses(const struct head_offload_description *d, uint32_t kernel)
{
	struct kernel_launch launch, before;
	memset(&launch, TEST_SENTINEL_BYTE, sizeof(launch));
	before = launch;
	expect(head_offload_prepare(d, TEST_ARENA_BYTES, kernel, &launch) ==
			       STATUS_INVALID_ARGUMENT &&
		       memcmp(&launch, &before, sizeof(launch)) == 0,
	       "a refused description leaves every launch byte unchanged");
	rejected++;
}

int main(void)
{
	static const uint32_t rows[] = { 1u, 2u, 5u, 8u };
	for (uint32_t index = 0; index < sizeof(rows) / sizeof(rows[0]);
	     index++) {
		struct head_offload_description d = description(rows[index]);
		struct kernel_launch launch;
		expect(head_offload_prepare(&d, TEST_ARENA_BYTES, TEST_KERNEL,
					    &launch) == STATUS_OK,
		       "full-vocabulary strided cut is admitted");
		struct head_offload_arguments arguments;
		memcpy(&arguments, launch.argument, sizeof(arguments));
		expect(arguments.destination_offset == TEST_DESTINATION_OFFSET +
							       TEST_CUT_BYTES &&
			       arguments.row_count == rows[index] &&
			       arguments.source_stride ==
				       (uint64_t)TEST_CUT_BYTES &&
			       arguments.destination_stride ==
				       (uint64_t)TEST_FULL_ROW_BYTES &&
			       launch.grid[1] == rows[index] &&
			       launch.grid[0] != 0 &&
			       launch.grid[0] <= HEAD_OFFLOAD_MOST_BLOCKS &&
			       launch.block[0] == HEAD_OFFLOAD_THREADS,
		       "cut offset and row strides preserve original output coordinates");
		accepted++;
	}
	struct head_offload_description d = description(TEST_ROWS);
	refuses(NULL, TEST_KERNEL);
	refuses(&d, BACKEND_NO_KERNEL);
	d.row_count = 0;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.row_count = TEST_ROWS_PAST;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.row_bytes = 0;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.row_bytes--;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.source_stride--;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.destination_stride--;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.column_bytes++;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.column_bytes = TEST_FULL_ROW_BYTES;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.source_stride = TEST_SHORT_STRIDE;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.source.byte_count--;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.destination.byte_count = (uint64_t)TEST_CUT_BYTES;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.step.byte_count = STEP_BLOCK_BYTES_VALUE;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.peer_word.byte_count = 0;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.peer_word.byte_offset = d.step.byte_offset;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.source.byte_offset++;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.destination.byte_offset++;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.source_stride = UINT64_MAX - TEST_STRIDE_WRAP;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.destination_stride = UINT64_MAX - TEST_STRIDE_WRAP;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.destination_arena_byte_count = TEST_DESTINATION_OFFSET;
	refuses(&d, TEST_KERNEL);
	d = description(TEST_ROWS);
	d.reserved_zero = 1;
	refuses(&d, TEST_KERNEL);
	(void)printf(
		"head_offload_test: %u failures; %u admitted shapes; %u refused ownership/bounds controls\n",
		failures, accepted, rejected);
	return failures == 0 ? 0 : 1;
}
