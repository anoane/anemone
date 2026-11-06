/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/official_norm.h"
#include "host_memory_backend.h"

#define ARENA_BYTES (1ull << 20u)
#define WEIGHT_OFFSET 4096u
#define INPUT_OFFSET (1ull << 16u)
#define OUTPUT_OFFSET (1ull << 18u)
#define STATISTICS_OFFSET (1ull << 19u)
#define KERNEL_HANDLE 17u
#define UNTOUCHED 0xA5
#define BAD_WIDTH 64u
#define BAD_ROWS 2u
#define MAX_THREADS OFFICIAL_NORM_MOST_THREADS
#define HALF_THREADS (MAX_THREADS / 2u)
#define QUARTER_THREADS (HALF_THREADS / 2u)
#define EIGHTH_THREADS (QUARTER_THREADS / 2u)

static int failures;

static void expect(bool condition, const char *what)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", what);
	}
}

static struct official_norm_description description_of(uint32_t width,
						       uint32_t rows)
{
	const uint64_t bytes = (uint64_t)width * sizeof(uint16_t);
	return (struct official_norm_description){
		.input = { INPUT_OFFSET, bytes * rows },
		.weight = { WEIGHT_OFFSET, bytes },
		.output = { OUTPUT_OFFSET, bytes * rows },
		.statistics = { STATISTICS_OFFSET,
				(uint64_t)rows * OFFICIAL_NORM_STATISTICS *
					sizeof(float) },
		.width = width,
		.row_count = rows
	};
}

static void
refusal_with_reference(const struct official_norm_description *description,
		       uint32_t reference_rows, uint64_t arena, uint32_t kernel,
		       const char *what)
{
	struct kernel_launch output, before;
	memset(&output, UNTOUCHED, sizeof(output));
	memcpy(&before, &output, sizeof(before));
	const result_status status =
		reference_rows == 0 ?
			official_norm_prepare(description, arena, kernel,
					      &output) :
			official_norm_shape_prepare(description, reference_rows,
						    arena, kernel, &output);
	expect(status == STATUS_INVALID_ARGUMENT, what);
	expect(memcmp(&before, &output, sizeof(output)) == 0,
	       "failed norm preparation leaves output untouched");
}

static void refusal(const struct official_norm_description *description,
		    uint64_t arena, uint32_t kernel, const char *what)
{
	refusal_with_reference(description, 0, arena, kernel, what);
}

MUST_CHECK static result_status
prepare_failure(struct device_card *card, const char *file,
		const char *function, uint32_t bytes, uint32_t *handle)
{
	(void)card;
	(void)file;
	(void)function;
	(void)bytes;
	(void)handle;
	return STATUS_DEVICE_ERROR;
}

int main(void)
{
	static const uint32_t widths[] = { MODEL_HEAD_DIMS, MODEL_QUERY_RANK,
					   MODEL_HIDDEN };
	static const uint32_t rows[] = { 1, OFFICIAL_NORM_ODD_ROWS,
					 MODEL_STREAMS, LINEAR_MAXIMUM_ROWS };
	static const uint32_t threads[][sizeof(rows) / sizeof(rows[0])] = {
		{ QUARTER_THREADS, QUARTER_THREADS, QUARTER_THREADS,
		  EIGHTH_THREADS },
		{ HALF_THREADS, HALF_THREADS, QUARTER_THREADS, EIGHTH_THREADS },
		{ MAX_THREADS, HALF_THREADS, QUARTER_THREADS, EIGHTH_THREADS }
	};
	for (uint32_t width = 0; width < sizeof(widths) / sizeof(widths[0]);
	     width++) {
		for (uint32_t row = 0; row < sizeof(rows) / sizeof(rows[0]);
		     row++) {
			const struct official_norm_description description =
				description_of(widths[width], rows[row]);
			struct kernel_launch launch;
			const result_status status =
				official_norm_prepare(&description, ARENA_BYTES,
						      KERNEL_HANDLE, &launch);
			expect(status == STATUS_OK,
			       "supported norm shape admitted");
			if (status == STATUS_OK) {
				struct official_norm_arguments arguments;
				memcpy(&arguments, launch.argument,
				       sizeof(arguments));
				expect(launch.kernel == KERNEL_HANDLE &&
					       launch.block[0] ==
						       threads[width][row] &&
					       launch.grid[0] == 1 &&
					       launch.grid[1] == rows[row] &&
					       launch.argument_byte_count ==
						       sizeof(arguments),
				       "norm geometry and record");
				expect(arguments.width == widths[width] &&
					       arguments.reserved_zero == 0 &&
					       arguments.mean_factor ==
						       1.0f / (float)widths
								       [width] &&
					       arguments.epsilon ==
						       MODEL_NORM_EPSILON,
				       "norm statistic constants");
			}
			for (uint32_t reference = 0;
			     reference < sizeof(rows) / sizeof(rows[0]);
			     reference++) {
				const result_status shaped =
					official_norm_shape_prepare(
						&description, rows[reference],
						ARENA_BYTES, KERNEL_HANDLE,
						&launch);
				expect(shaped == STATUS_OK,
				       "independent actual/reference norm rows admitted");
				if (shaped == STATUS_OK)
					expect(launch.grid[1] == rows[row] &&
						       launch.block[0] ==
							       threads[width]
								      [reference],
					       "norm grid uses actual extent; tree uses reference rows");
			}
		}
	}
	struct official_norm_description description =
		description_of(MODEL_HIDDEN, 1);
	refusal_with_reference(&description, BAD_ROWS, ARENA_BYTES,
			       KERNEL_HANDLE,
			       "unqualified reference shape refused");
	description.row_count = LINEAR_MAXIMUM_ROWS;
	refusal_with_reference(
		&description, 1, ARENA_BYTES, KERNEL_HANDLE,
		"small reference shape cannot hide an invalid actual memory extent");
	description = description_of(MODEL_HIDDEN, 1);
	refusal(NULL, ARENA_BYTES, KERNEL_HANDLE, "null description refused");
	refusal(&description, ARENA_BYTES, BACKEND_NO_KERNEL,
		"missing norm handle refused");
	refusal(&description, 1, KERNEL_HANDLE, "out of arena refused");
	description.row_count = BAD_ROWS;
	refusal(&description, ARENA_BYTES, KERNEL_HANDLE,
		"unqualified row shape refused");
	description = description_of(BAD_WIDTH, 1);
	refusal(&description, ARENA_BYTES, KERNEL_HANDLE,
		"unqualified width refused");
	description = description_of(MODEL_HIDDEN, 1);
	description.output.byte_offset = description.input.byte_offset;
	refusal(&description, ARENA_BYTES, KERNEL_HANDLE,
		"input/output alias refused");
	description = description_of(MODEL_HIDDEN, 1);
	description.statistics.byte_offset = description.weight.byte_offset;
	refusal(&description, ARENA_BYTES, KERNEL_HANDLE,
		"statistics/weight alias refused");
	description = description_of(MODEL_HIDDEN, 1);
	description.input.byte_count--;
	refusal(&description, ARENA_BYTES, KERNEL_HANDLE,
		"short input refused");
	description = description_of(MODEL_HIDDEN, 1);
	description.statistics.byte_offset++;
	refusal(&description, ARENA_BYTES, KERNEL_HANDLE,
		"misaligned statistics refused");
	struct device_card *card = NULL;
	test_host_memory_backend_reset(1, ARENA_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	const result_status opened =
		test_host_memory_backend.card_open(0, ARENA_BYTES, &card);
	expect(opened == STATUS_OK, "host card opens");
	if (opened == STATUS_OK) {
		uint32_t kernel = KERNEL_HANDLE;
		struct backend_operations backend = test_host_memory_backend;
		backend.kernel_prepare = prepare_failure;
		expect(official_norm_kernel_prepare(&backend, card, &kernel) ==
				       STATUS_DEVICE_ERROR &&
			       kernel == KERNEL_HANDLE,
		       "kernel preparation failure preserves caller handle");
		expect(official_norm_kernel_prepare(&test_host_memory_backend,
						    card,
						    &kernel) == STATUS_OK &&
			       kernel != BACKEND_NO_KERNEL,
		       "norm kernel prepared");
		test_host_memory_backend.card_close(card, 1);
	}
	(void)printf("official_norm_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
