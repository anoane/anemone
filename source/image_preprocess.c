/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/image.h"

#define RESIZE_PAIR 2u
#define RESIZE_HALF 0.5
#define RESIZE_QUARTER 0.25
#define RESIZE_CUBIC_SUPPORT 2.0
#define RESIZE_CUBIC_A (-0.5)
#define RESIZE_CUBIC_OUTER_SHIFT 5.0
#define RESIZE_CUBIC_OUTER_LINEAR 8.0
#define RESIZE_CUBIC_OUTER_CONSTANT 4.0
#define RESIZE_CUBIC_INNER_QUADRATIC (-2.5)
#define RESIZE_CUBIC_INNER_CUBIC 1.5
#define RESIZE_COEFFICIENT_BITS 22u
#define RESIZE_COEFFICIENT_SCALE (1u << RESIZE_COEFFICIENT_BITS)
#define RESIZE_SAMPLE_MAXIMUM 255
#define RESIZE_PAD_VALUE 127u
#define RESIZE_SAMPLE_DIVISOR 255.0f
#define RESIZE_NORMALIZE_OFFSET 0.5f
#define RESIZE_NORMALIZE_DIVISOR 0.5f
#define BF16_SHIFT 16u
#define BF16_ROUND_BIAS 32767u
#define FLOAT_EXPONENT_MASK UINT32_C(0x7f800000)
#define FLOAT_MANTISSA_MASK UINT32_C(0x007fffff)
#define BF16_QUIET_NAN UINT32_C(0x0040)

struct resize_axis {
	int32_t *coefficients;
	uint32_t *starts;
	uint32_t *counts;
	uint32_t kernel;
};
static bool image_plan_equal(const struct image_plan *left,
			     const struct image_plan *right)
{
	return left->width == right->width && left->height == right->height &&
	       left->vit_width == right->vit_width &&
	       left->vit_height == right->vit_height &&
	       left->llm_width == right->llm_width &&
	       left->llm_height == right->llm_height &&
	       left->token_count == right->token_count &&
	       left->patch_elements == right->patch_elements;
}
MUST_CHECK result_status image_grid(uint32_t width, uint32_t height,
				    struct image_plan *plan_out)
{
	if (plan_out == NULL || width == 0 || height == 0 ||
	    width > IMAGE_MAX_DIMENSION || height > IMAGE_MAX_DIMENSION ||
	    (uint64_t)width * height > IMAGE_MAX_PIXELS)
		return STATUS_INVALID_ARGUMENT;
	if ((uint64_t)width * height < IMAGE_MIN_PIXELS) {
		double ratio = sqrt((double)IMAGE_MIN_PIXELS /
				    ((double)width * height));
		width = (uint32_t)((double)width * ratio);
		height = (uint32_t)((double)height * ratio);
	}
	uint32_t best_width =
		((width + IMAGE_PATCH_SIZE - 1u) / IMAGE_PATCH_SIZE) *
		IMAGE_PATCH_SIZE;
	uint32_t best_height =
		((height + IMAGE_PATCH_SIZE - 1u) / IMAGE_PATCH_SIZE) *
		IMAGE_PATCH_SIZE;
	uint32_t cell = IMAGE_PATCH_SIZE * IMAGE_DOWNSAMPLE;
	uint32_t llm_width = (best_width + cell - 1u) / cell;
	uint32_t llm_height = (best_height + cell - 1u) / cell;
	if ((uint64_t)llm_height * (llm_width + 1u) + RESIZE_PAIR >
	    IMAGE_MAX_TOKENS) {
		double aspect = (double)height / width;
		double maximum_width =
			sqrt((IMAGE_MAX_TOKENS - RESIZE_PAIR) / aspect +
			     RESIZE_QUARTER) -
			RESIZE_HALF;
		double maximum_height = maximum_width * aspect;
		if (maximum_width < 1.0) {
			best_height = ((IMAGE_MAX_TOKENS - RESIZE_PAIR) /
				       RESIZE_PAIR) *
				      cell;
			best_width = cell;
		} else if (maximum_height < 1.0) {
			best_height = cell;
			best_width =
				(IMAGE_MAX_TOKENS - RESIZE_PAIR - 1u) * cell;
		} else {
			double horizontal = floor(maximum_width) * cell / width;
			double vertical = floor(maximum_height) * cell / height;
			double ratio = horizontal < vertical ? horizontal :
							       vertical;
			best_height = (uint32_t)floor(height * ratio /
						      IMAGE_PATCH_SIZE) *
				      IMAGE_PATCH_SIZE;
			best_width = (uint32_t)floor(width * ratio /
						     IMAGE_PATCH_SIZE) *
				     IMAGE_PATCH_SIZE;
		}
		llm_width = (best_width + cell - 1u) / cell;
		llm_height = (best_height + cell - 1u) / cell;
	}
	struct image_plan result = {
		.width = best_width,
		.height = best_height,
		.vit_width = best_width / IMAGE_PATCH_SIZE,
		.vit_height = best_height / IMAGE_PATCH_SIZE,
		.llm_width = llm_width,
		.llm_height = llm_height,
		.token_count = llm_height * (llm_width + 1u) + RESIZE_PAIR,
		.patch_elements =
			(size_t)best_width * best_height * IMAGE_CHANNELS
	};
	if (best_width == 0 || best_height == 0 ||
	    result.token_count > IMAGE_MAX_TOKENS ||
	    (uint64_t)result.vit_width * result.vit_height > IMAGE_MAX_PATCHES)
		return STATUS_INVALID_ARGUMENT;
	*plan_out = result;
	return STATUS_OK;
}
static double resize_cubic(double position)
{
	position = fabs(position);
	if (position < 1.0)
		return ((RESIZE_CUBIC_INNER_CUBIC * position +
			 RESIZE_CUBIC_INNER_QUADRATIC) *
			position) *
			       position +
		       1.0;
	if (position < RESIZE_CUBIC_SUPPORT)
		return (((position - RESIZE_CUBIC_OUTER_SHIFT) * position +
			 RESIZE_CUBIC_OUTER_LINEAR) *
				position -
			RESIZE_CUBIC_OUTER_CONSTANT) *
		       RESIZE_CUBIC_A;
	return 0;
}
static void resize_axis_destroy(struct resize_axis *axis)
{
	free(axis->coefficients);
	free(axis->starts);
	free(axis->counts);
	*axis = (struct resize_axis){ 0 };
}
static MUST_CHECK result_status resize_axis_prepare(uint32_t source_size,
						    uint32_t target_size,
						    struct resize_axis *axis_out)
{
	double scale = (double)source_size / target_size;
	double filter_scale = scale < 1.0 ? 1.0 : scale;
	double support = RESIZE_CUBIC_SUPPORT * filter_scale;
	double inverse_filter_scale = 1.0 / filter_scale;
	uint32_t kernel = (uint32_t)ceil(support) * RESIZE_PAIR + 1u;
	struct resize_axis axis = { .kernel = kernel };
	axis.coefficients = malloc((size_t)target_size * kernel *
				   sizeof(*axis.coefficients));
	axis.starts = malloc((size_t)target_size * sizeof(*axis.starts));
	axis.counts = malloc((size_t)target_size * sizeof(*axis.counts));
	double *weights = malloc((size_t)kernel * sizeof(*weights));
	if (axis.coefficients == NULL || axis.starts == NULL ||
	    axis.counts == NULL || weights == NULL) {
		free(weights);
		resize_axis_destroy(&axis);
		return STATUS_OUT_OF_MEMORY;
	}
	for (uint32_t target = 0; target < target_size; ++target) {
		double center = ((double)target + RESIZE_HALF) * scale;
		int64_t first = (int64_t)(center - support + RESIZE_HALF);
		int64_t last = (int64_t)(center + support + RESIZE_HALF);
		if (first < 0)
			first = 0;
		if (last > source_size)
			last = source_size;
		uint32_t count = (uint32_t)(last - first);
		double total = 0;
		for (uint32_t offset = 0; offset < count; ++offset) {
			weights[offset] =
				resize_cubic(((double)offset + (double)first -
					      center + RESIZE_HALF) *
					     inverse_filter_scale);
			total += weights[offset];
		}
		axis.starts[target] = (uint32_t)first;
		axis.counts[target] = count;
		for (uint32_t offset = 0; offset < count; ++offset) {
			double weight = weights[offset] / total;
			axis.coefficients[(size_t)target * kernel + offset] =
				(int32_t)(weight * RESIZE_COEFFICIENT_SCALE +
					  (weight < 0 ? -RESIZE_HALF :
							RESIZE_HALF));
		}
	}
	free(weights);
	*axis_out = axis;
	return STATUS_OK;
}
static uint8_t resize_finish(int64_t sum)
{
	/* Division emulates arithmetic right shift even for a negative cubic lobe. */
	if (sum <= 0)
		return 0;
	int64_t value = sum / RESIZE_COEFFICIENT_SCALE;
	return (uint8_t)(value > RESIZE_SAMPLE_MAXIMUM ? RESIZE_SAMPLE_MAXIMUM :
							 value);
}
static uint32_t resize_round_even(double value)
{
	double integral = floor(value);
	double fraction = value - integral;
	uint32_t result = (uint32_t)integral;
	if (fraction > RESIZE_HALF ||
	    (fraction == RESIZE_HALF && (result & 1u) != 0))
		++result;
	return result;
}
static uint16_t resize_bfloat16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	if ((bits & FLOAT_EXPONENT_MASK) == FLOAT_EXPONENT_MASK &&
	    (bits & FLOAT_MANTISSA_MASK) != 0)
		return (uint16_t)((bits >> BF16_SHIFT) | BF16_QUIET_NAN);
	bits += BF16_ROUND_BIAS + ((bits >> BF16_SHIFT) & 1u);
	return (uint16_t)(bits >> BF16_SHIFT);
}
MUST_CHECK result_status image_preprocess(const struct image_rgb *image,
					  const struct image_plan *plan,
					  uint16_t *patches_out,
					  size_t patch_capacity)
{
	if (image == NULL || image->pixels == NULL || plan == NULL ||
	    patches_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct image_plan expected;
	result_status status =
		image_grid(image->width, image->height, &expected);
	if (status != STATUS_OK)
		return status;
	if (!image_plan_equal(plan, &expected) ||
	    patch_capacity < plan->patch_elements)
		return STATUS_INVALID_ARGUMENT;
	uint32_t resized_width = plan->width, resized_height = plan->height;
	double source_aspect = (double)image->width / image->height;
	double target_aspect = (double)plan->width / plan->height;
	if (source_aspect > target_aspect)
		resized_height = resize_round_even((double)image->height /
						   image->width * plan->width);
	else if (source_aspect < target_aspect)
		resized_width = resize_round_even((double)image->width /
						  image->height * plan->height);
	if (resized_width == 0 || resized_height == 0)
		return STATUS_INVALID_ARGUMENT;
	uint32_t pad_left = resize_round_even(
		(double)(plan->width - resized_width) * RESIZE_HALF);
	uint32_t pad_top = resize_round_even(
		(double)(plan->height - resized_height) * RESIZE_HALF);
	size_t intermediate_bytes =
		(size_t)resized_width * image->height * IMAGE_CHANNELS;
	if (intermediate_bytes > IMAGE_MAX_DECOMPRESSED_BYTES)
		return STATUS_INVALID_ARGUMENT;
	uint8_t *intermediate = malloc(intermediate_bytes);
	uint8_t *resized =
		malloc((size_t)resized_width * resized_height * IMAGE_CHANNELS);
	struct resize_axis horizontal = { 0 }, vertical = { 0 };
	if (intermediate == NULL || resized == NULL) {
		status = STATUS_OUT_OF_MEMORY;
		goto cleanup;
	}
	status = resize_axis_prepare(image->width, resized_width, &horizontal);
	if (status != STATUS_OK)
		goto cleanup;
	status = resize_axis_prepare(image->height, resized_height, &vertical);
	if (status != STATUS_OK)
		goto cleanup;
	for (uint32_t row = 0; row < image->height; ++row)
		for (uint32_t column = 0; column < resized_width; ++column)
			for (unsigned channel = 0; channel < IMAGE_CHANNELS;
			     ++channel) {
				int64_t sum =
					RESIZE_COEFFICIENT_SCALE / RESIZE_PAIR;
				for (uint32_t offset = 0;
				     offset < horizontal.counts[column];
				     ++offset)
					sum += (int64_t)image->pixels
						       [((size_t)row *
								 image->width +
							 horizontal
								 .starts[column] +
							 offset) *
								IMAGE_CHANNELS +
							channel] *
					       horizontal.coefficients
						       [(size_t)column *
								horizontal.kernel +
							offset];
				intermediate[((size_t)row * resized_width +
					      column) *
						     IMAGE_CHANNELS +
					     channel] = resize_finish(sum);
			}
	for (uint32_t row = 0; row < resized_height; ++row)
		for (uint32_t column = 0; column < resized_width; ++column)
			for (unsigned channel = 0; channel < IMAGE_CHANNELS;
			     ++channel) {
				int64_t sum =
					RESIZE_COEFFICIENT_SCALE / RESIZE_PAIR;
				for (uint32_t offset = 0;
				     offset < vertical.counts[row]; ++offset)
					sum += (int64_t)intermediate
						       [((size_t)(vertical.starts
									  [row] +
								  offset) *
								 resized_width +
							 column) *
								IMAGE_CHANNELS +
							channel] *
					       vertical.coefficients
						       [(size_t)row *
								vertical.kernel +
							offset];
				resized[((size_t)row * resized_width + column) *
						IMAGE_CHANNELS +
					channel] = resize_finish(sum);
			}
	for (uint32_t patch_row = 0; patch_row < plan->vit_height; ++patch_row)
		for (uint32_t patch_column = 0; patch_column < plan->vit_width;
		     ++patch_column)
			for (unsigned channel = 0; channel < IMAGE_CHANNELS;
			     ++channel)
				for (uint32_t row = 0; row < IMAGE_PATCH_SIZE;
				     ++row)
					for (uint32_t column = 0;
					     column < IMAGE_PATCH_SIZE;
					     ++column) {
						uint32_t pixel_row =
							patch_row *
								IMAGE_PATCH_SIZE +
							row;
						uint32_t pixel_column =
							patch_column *
								IMAGE_PATCH_SIZE +
							column;
						unsigned sample =
							RESIZE_PAD_VALUE;
						if (pixel_row >= pad_top &&
						    pixel_row - pad_top <
							    resized_height &&
						    pixel_column >= pad_left &&
						    pixel_column - pad_left <
							    resized_width)
							sample = resized
								[((size_t)(pixel_row -
									   pad_top) *
									  resized_width +
								  pixel_column -
								  pad_left) *
									 IMAGE_CHANNELS +
								 channel];
						float value =
							(float)sample /
							RESIZE_SAMPLE_DIVISOR;
						value = (value -
							 RESIZE_NORMALIZE_OFFSET) /
							RESIZE_NORMALIZE_DIVISOR;
						size_t output =
							(((((size_t)patch_row *
								    plan->vit_width +
							    patch_column) *
								   IMAGE_CHANNELS +
							   channel) *
								  IMAGE_PATCH_SIZE +
							  row) *
							 IMAGE_PATCH_SIZE) +
							column;
						patches_out[output] =
							resize_bfloat16(value);
					}
	status = STATUS_OK;
cleanup:
	resize_axis_destroy(&horizontal);
	resize_axis_destroy(&vertical);
	free(intermediate);
	free(resized);
	return status;
}
