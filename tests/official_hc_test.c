/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/official_hc.h"
#include "host_memory_backend.h"

#define ARENA_BYTES (4ull << 20u)
#define WEIGHTS_OFFSET 4096u
#define ROWS_OFFSET (2ull << 20u)
#define OUTPUT_OFFSET (3ull << 20u)
#define DENSE_HANDLE 11u
#define DOT_HANDLE 12u
#define SPLIT_HANDLE 13u
#define UNTOUCHED_BYTE 0xA5
#define WRONG_SCALE 2.0f

static int failures;
static struct dense_linear_kernels dense;
static const struct official_hc_kernels official = { DOT_HANDLE, SPLIT_HANDLE };

static struct dense_linear_description description_of(uint32_t rows)
{
	return (struct dense_linear_description){
		.weights = { WEIGHTS_OFFSET, (uint64_t)MODEL_MIXES *
						     OFFICIAL_HC_INPUTS *
						     sizeof(float) },
		.rows = { ROWS_OFFSET, (uint64_t)rows * OFFICIAL_HC_INPUTS *
					       sizeof(uint16_t) },
		.output = { OUTPUT_OFFSET,
			    (uint64_t)rows * MODEL_MIXES * sizeof(float) },
		.output_size = MODEL_MIXES,
		.input_size = OFFICIAL_HC_INPUTS,
		.row_count = rows,
		.weight_type = DENSE_FLOAT32,
		.output_type = DENSE_FLOAT32,
		.flags = DENSE_ROW_SCALE,
		.output_scale = 1.0f,
		.epsilon = MODEL_NORM_EPSILON
	};
}

static void expect(bool condition, const char *what)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", what);
	}
}

MUST_CHECK static result_status
preparation_failure(struct device_card *card, const char *file,
		    const char *function, uint32_t bytes, uint32_t *handle)
{
	(void)card;
	(void)file;
	(void)function;
	(void)bytes;
	(void)handle;
	return STATUS_DEVICE_ERROR;
}

static void kernel_selection(void)
{
	struct device_card *card = NULL;
	test_host_memory_backend_reset(1, ARENA_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	const result_status opened =
		test_host_memory_backend.card_open(0, ARENA_BYTES, &card);
	expect(opened == STATUS_OK, "host card opens");
	if (opened != STATUS_OK)
		return;
	struct official_hc_kernels kernels = { BACKEND_NO_KERNEL,
					       BACKEND_NO_KERNEL };
	expect(official_hc_kernels_prepare(&test_host_memory_backend, card,
					   &kernels) == STATUS_OK,
	       "direct kernels prepare");
	const struct official_hc_kernels direct = kernels;
	expect(official_hc_staged_prepare(&test_host_memory_backend, card,
					  &kernels) == STATUS_OK &&
		       kernels.dot == direct.dot &&
		       kernels.split != direct.split,
	       "staging replaces only multi-row selection");
	const struct official_hc_kernels before = kernels;
	struct backend_operations failed = test_host_memory_backend;
	failed.kernel_prepare = preparation_failure;
	expect(official_hc_staged_prepare(&failed, card, &kernels) ==
			       STATUS_DEVICE_ERROR &&
		       memcmp(&kernels, &before, sizeof(kernels)) == 0,
	       "failed staged preparation preserves handles");
	expect(official_hc_kernels_prepare(&failed, card, &kernels) ==
			       STATUS_DEVICE_ERROR &&
		       memcmp(&kernels, &before, sizeof(kernels)) == 0,
	       "failed direct preparation preserves handles");
	expect(official_hc_staged_prepare(NULL, card, &kernels) ==
		       STATUS_INVALID_ARGUMENT,
	       "missing backend refused");
	expect(official_hc_staged_prepare(&test_host_memory_backend, NULL,
					  &kernels) == STATUS_INVALID_ARGUMENT,
	       "missing card refused");
	test_host_memory_backend.card_close(card, 1);
}

static void refused(const struct dense_linear_description *description,
		    const struct official_hc_kernels *kernels, uint64_t arena)
{
	struct kernel_launch launch, before;
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	before = launch;
	expect(official_hc_prepare(description, arena, &dense, kernels,
				   &launch) == STATUS_INVALID_ARGUMENT,
	       "invalid input refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "failed prepare leaves output untouched");
}

int main(void)
{
	kernel_selection();
	for (uint32_t instance = 0; instance < DENSE_INSTANCES; instance++)
		dense.products[instance] = DENSE_HANDLE;
	static const uint32_t accepted[] = { 1, 3, 4, 8 };
	static const uint32_t rejected[] = { 0, 2, 5, 6, 7, 9, UINT32_MAX };
	for (uint32_t index = 0; index < sizeof(accepted) / sizeof(accepted[0]);
	     index++) {
		const struct dense_linear_description description =
			description_of(accepted[index]);
		struct kernel_launch launch = { 0 }, baseline = { 0 };
		expect(dense_linear_prepare(&description, ARENA_BYTES, &dense,
					    &baseline) == STATUS_OK,
		       "ordinary projection prepares");
		expect(official_hc_prepare(&description, ARENA_BYTES, &dense,
					   &official, &launch) == STATUS_OK,
		       "source-order projection prepares");
		expect(launch.kernel == (accepted[index] == 1 ? DOT_HANDLE :
								SPLIT_HANDLE) &&
			       launch.block[0] == OFFICIAL_HC_THREADS &&
			       launch.grid[0] == MODEL_MIXES &&
			       launch.grid[1] == accepted[index],
		       "correct source-order handle and geometry");
		expect(memcmp(launch.argument, baseline.argument,
			      sizeof(launch.argument)) == 0,
		       "validated dense arguments preserved byte for byte");
	}
	/* A serial engine may target a captured prompt GEMM, and conversely.
     * Validate real extents before recording the independent reference shape. */
	for (uint32_t actual = 0;
	     actual < sizeof(accepted) / sizeof(accepted[0]); actual++) {
		const struct dense_linear_description description =
			description_of(accepted[actual]);
		for (uint32_t reference = 0;
		     reference < sizeof(accepted) / sizeof(accepted[0]);
		     reference++) {
			struct kernel_launch launch = { 0 };
			expect(official_hc_shape_prepare(
				       &description, accepted[reference],
				       ARENA_BYTES, &dense, &official,
				       &launch) == STATUS_OK,
			       "independent reference shape prepares");
			struct dense_linear_arguments arguments;
			memcpy(&arguments, launch.argument, sizeof(arguments));
			expect(arguments.row_count == accepted[reference] &&
				       launch.grid[1] == accepted[actual] &&
				       launch.kernel ==
					       (accepted[reference] == 1 ?
							DOT_HANDLE :
							SPLIT_HANDLE),
			       "reference arithmetic and actual launch extents remain independent");
		}
		for (uint32_t index = 0;
		     index < sizeof(rejected) / sizeof(rejected[0]); index++) {
			struct kernel_launch launch, before;
			memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
			before = launch;
			expect(official_hc_shape_prepare(
				       &description, rejected[index],
				       ARENA_BYTES, &dense, &official,
				       &launch) == STATUS_INVALID_ARGUMENT &&
				       memcmp(&launch, &before,
					      sizeof(launch)) == 0,
			       "unqualified reference shape rejected without mutation");
		}
		struct dense_linear_description short_input = description;
		short_input.rows.byte_count--;
		struct kernel_launch launch, before;
		memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
		before = launch;
		expect(official_hc_shape_prepare(
			       &short_input, accepted[0], ARENA_BYTES, &dense,
			       &official, &launch) == STATUS_INVALID_ARGUMENT &&
			       memcmp(&launch, &before, sizeof(launch)) == 0,
		       "actual input range checked even for smaller reference shape");
	}
	for (uint32_t index = 0; index < sizeof(rejected) / sizeof(rejected[0]);
	     index++) {
		const struct dense_linear_description description =
			description_of(rejected[index]);
		refused(&description, &official, ARENA_BYTES);
	}
	struct dense_linear_description description = description_of(1);
	description.output.byte_offset = description.rows.byte_offset;
	refused(&description, &official, ARENA_BYTES);
	description = description_of(1);
	description.output_scale = WRONG_SCALE;
	refused(&description, &official, ARENA_BYTES);
	description = description_of(1);
	description.epsilon = NAN;
	refused(&description, &official, ARENA_BYTES);
	description = description_of(1);
	description.flags |= DENSE_CONE_ROWS;
	refused(&description, &official, ARENA_BYTES);
	description = description_of(1);
	description.weight_type = DENSE_BFLOAT16;
	refused(&description, &official, ARENA_BYTES);
	description = description_of(1);
	description.output_type = DENSE_BFLOAT16;
	refused(&description, &official, ARENA_BYTES);
	description = description_of(1);
	refused(&description, &official, 1);
	struct official_hc_kernels missing = official;
	missing.dot = BACKEND_NO_KERNEL;
	refused(&description, &missing, ARENA_BYTES);
	refused(NULL, &official, ARENA_BYTES);
	refused(&description, NULL, ARENA_BYTES);
	(void)printf("official_hc_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
