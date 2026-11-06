/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Extra image ranges and step flags are validated before any image-aware launch is published. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/image_routing.h"
#include "anemone/engram.h"

#define TEST_ALIGNMENT 64u
#define TEST_ROWS 2u
#define TEST_MATRIX_WIDTH 64u
#define TEST_MATRIX_BITS 4u
#define TEST_MATRIX_BYTES 2176u
#define FIRST_OFFSET 4096u
#define TEXT_KERNEL 7u
#define IMAGE_KERNEL 11u
#define IMAGE_WIDE_KERNEL 12u
#define IMAGE_GATE_KERNEL 13u
#define FILL_BYTE 0xa5
#define TWO_RANGES 2u

static int failures;

static void expect(bool condition, const char *description)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", description);
		failures++;
	}
}

static struct linear_range reserve(uint64_t *cursor, uint64_t bytes)
{
	*cursor = (*cursor + TEST_ALIGNMENT - 1u) / TEST_ALIGNMENT *
		  TEST_ALIGNMENT;
	const struct linear_range range = { *cursor, bytes };
	*cursor += bytes;
	return range;
}

static void test_router(void)
{
	uint64_t cursor = FIRST_OFFSET;
	struct linear_routed_experts_description experts = {
		.gate_up_shape = { TEST_MATRIX_WIDTH, TEST_MATRIX_WIDTH,
				   TEST_MATRIX_BITS, 0 },
		.down_shape = { TEST_MATRIX_WIDTH, TEST_MATRIX_WIDTH,
				TEST_MATRIX_BITS, 0 },
		.weight_format = WEIGHT_FORMAT_E2M1_SCALED,
		.row_count = TEST_ROWS,
		.pairs_per_row = MODEL_EXPERTS_PER_ROW,
		.job_count = TEST_ROWS * MODEL_EXPERTS_PER_ROW
	};
	const struct router_kernels_record base = { TEXT_KERNEL,
						    TEXT_KERNEL,
						    TEXT_KERNEL,
						    TEXT_KERNEL,
						    TEXT_KERNEL,
						    TEXT_KERNEL,
						    0 };
	const struct image_routing_kernels kernels = { IMAGE_KERNEL,
						       IMAGE_WIDE_KERNEL,
						       IMAGE_GATE_KERNEL };
	uint64_t offsets[MODEL_EXPERTS * MODEL_EXPERT_MATRICES] = { 0 };
	struct router_description description = {
		.experts = &experts,
		.expert_offsets = offsets,
		.expert_count = MODEL_EXPERTS,
		.pick_count = MODEL_EXPERTS_PER_ROW,
		.step = reserve(&cursor,
				STEP_BLOCK_BYTES_VALUE +
					TEST_ROWS * STEP_ROW_BYTES_VALUE),
		.scores = reserve(&cursor,
				  (uint64_t)TEST_ROWS * MODEL_ROUTER_ROW_BYTES),
		.bias = reserve(&cursor, MODEL_ROUTER_ROW_BYTES),
		.offsets = reserve(&cursor, sizeof(offsets)),
		.affinities = reserve(&cursor, (uint64_t)TEST_ROWS *
						       MODEL_ROUTER_ROW_BYTES)
	};
	experts.jobs =
		reserve(&cursor, linear_jobs_byte_count(experts.job_count));
	experts.expert_weights = reserve(&cursor, TEST_MATRIX_BYTES);
	const struct linear_range vision_bias =
		reserve(&cursor, MODEL_ROUTER_ROW_BYTES);
	for (size_t index = 0; index < sizeof(offsets) / sizeof(offsets[0]);
	     index++) {
		offsets[index] = experts.expert_weights.byte_offset;
	}
	struct kernel_launch ordinary = { 0 }, image = { 0 };
	expect(router_select_prepare(&description, cursor, &base, &ordinary) ==
		       STATUS_OK,
	       "base synthetic router description is valid");
	expect(image_router_select_prepare(&description, vision_bias, cursor,
					   &base, &kernels,
					   &image) == STATUS_OK &&
		       image.kernel == IMAGE_KERNEL &&
		       image.argument_byte_count == IMAGE_ROUTER_ARGUMENT_BYTES,
	       "image router uses its own prepared record and kernel");
	struct image_router_arguments record = { 0 };
	memcpy(&record, image.argument, sizeof(record));
	expect(memcmp(&record.base, ordinary.argument, sizeof(record.base)) ==
			       0 &&
		       record.vision_bias_offset == vision_bias.byte_offset,
	       "image variant retains every base numerical field");
	struct kernel_launch untouched;
	memset(&image, FILL_BYTE, sizeof(image));
	untouched = image;
	expect(image_router_select_prepare(&description, description.scores,
					   cursor, &base, &kernels,
					   &image) == STATUS_INVALID_ARGUMENT &&
		       memcmp((const unsigned char *)&image,
			      (const unsigned char *)&untouched,
			      sizeof(image)) == 0,
	       "vision bias cannot alias a base range and failed prepare leaves the launch unchanged");
	description.step.byte_count = STEP_BLOCK_BYTES_VALUE;
	expect(image_router_select_prepare(&description, vision_bias, cursor,
					   &base, &kernels,
					   &image) == STATUS_INVALID_ARGUMENT,
	       "step range includes every row flag used by the variant");
	description.step.byte_count =
		STEP_BLOCK_BYTES_VALUE + TEST_ROWS * STEP_ROW_BYTES_VALUE;
	struct linear_range short_bias = vision_bias;
	short_bias.byte_count--;
	expect(image_router_select_prepare(&description, short_bias, cursor,
					   &base, &kernels,
					   &image) == STATUS_INVALID_ARGUMENT,
	       "vision bias range is not a byte short");
}

static void test_engram(void)
{
	uint64_t cursor = FIRST_OFFSET;
	const struct engram_rows_kernels base = { TEXT_KERNEL, TEXT_KERNEL };
	const struct image_routing_kernels kernels = { IMAGE_KERNEL,
						       IMAGE_WIDE_KERNEL,
						       IMAGE_GATE_KERNEL };
	const struct engram_rows_description description = {
		.rows = reserve(&cursor,
				(uint64_t)TEST_ROWS * ENGRAM_ROWS_ROW_BYTES),
		.values = reserve(&cursor, (uint64_t)TEST_ROWS *
						   ENGRAM_ROWS_VALUES *
						   sizeof(uint16_t)),
		.projection = reserve(&cursor, (uint64_t)TEST_ROWS *
						       ENGRAM_ROWS_PROJECTION *
						       sizeof(uint16_t)),
		.query_weight =
			reserve(&cursor, (uint64_t)MODEL_STREAMS *
						 MODEL_HIDDEN_ROW_BYTES),
		.key_weight = reserve(&cursor, (uint64_t)MODEL_STREAMS *
						       MODEL_HIDDEN_ROW_BYTES),
		.streams = reserve(&cursor, (uint64_t)TEST_ROWS *
						    MODEL_STREAMS_ROW_BYTES),
		.rows_per_token = 1,
		.row_count = TEST_ROWS
	};
	const struct linear_range step =
		reserve(&cursor, STEP_BLOCK_BYTES_VALUE +
					 TEST_ROWS * STEP_ROW_BYTES_VALUE);
	struct kernel_launch ordinary[ENGRAM_ROWS_LAUNCH_COUNT] = { { 0 } },
			     image[ENGRAM_ROWS_LAUNCH_COUNT] = { { 0 } };
	expect(engram_rows_prepare(&description, cursor, &base, ordinary) ==
			       STATUS_OK &&
		       image_engram_rows_prepare(&description, step, cursor,
						 &base, &kernels,
						 image) == STATUS_OK,
	       "image-aware Engram prepares valid disjoint buffers");
	expect(memcmp((const unsigned char
			       *)&ordinary[ENGRAM_ROWS_VALUES_LAUNCH],
		      (const unsigned char *)&image[ENGRAM_ROWS_VALUES_LAUNCH],
		      sizeof(ordinary[0])) == 0 &&
		       image[ENGRAM_ROWS_GATE_LAUNCH].kernel ==
			       IMAGE_GATE_KERNEL,
	       "Engram values launch is unchanged and only its gate selects the image variant");
	expect(image_engram_rows_prepare(&description, description.streams,
					 cursor, &base, &kernels,
					 image) == STATUS_INVALID_ARGUMENT,
	       "Engram step flags share no bytes with hidden streams");
	struct linear_range short_step = step;
	short_step.byte_count--;
	expect(image_engram_rows_prepare(&description, short_step, cursor,
					 &base, &kernels,
					 image) == STATUS_INVALID_ARGUMENT,
	       "Engram step range holds the last row flag");
}

int main(void)
{
	test_router();
	test_engram();
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
