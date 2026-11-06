/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Synthetic tensors share a documented integer recipe with tools/vision/make_fixtures.py, not an RNG. */
#include "vision_fixture.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vision_reference.h"

#define FIXTURE_WIDTH 16u
#define FIXTURE_HEADS 2u
#define FIXTURE_LAYERS 2u
#define FIXTURE_INTERMEDIATE 12u
#define FIXTURE_PATCH 2u
#define FIXTURE_FIRST_OFFSET 256u
#define FIXTURE_INDEX_MULTIPLIER 17u
#define FIXTURE_TENSOR_MULTIPLIER 11u
#define FIXTURE_MODULUS 31u
#define FIXTURE_CENTER 15
#define FIXTURE_DIVISOR 64.0f
#define FIXTURE_NAME_CAPACITY 128u
#define FIXTURE_EXPECTED_WORDS 256u
#define FIXTURE_BFLOAT16_NORM_CASE 1u

static const uint32_t fixture_shapes[TEST_VISION_CASES][VISION_HALF_DIVISOR] = {
	{ 1, 1 },
	{ 2, 3 },
	{ 4, 5 },
	{ 11, 12 }
};

static struct linear_range tensor_add(struct test_vision_fixture *fixture,
				      uint64_t *offset, uint32_t *tensor_id,
				      uint32_t elements, bool norm)
{
	const uint64_t bytes =
		(uint64_t)elements *
		(norm && fixture->case_index != FIXTURE_BFLOAT16_NORM_CASE ?
			 sizeof(float) :
			 sizeof(uint16_t));
	const struct linear_range range = { .byte_offset = *offset,
					    .byte_count = bytes };
	for (uint32_t index = 0; index < elements; index++) {
		const uint32_t pattern =
			(index * FIXTURE_INDEX_MULTIPLIER +
			 *tensor_id * FIXTURE_TENSOR_MULTIPLIER) %
			FIXTURE_MODULUS;
		const float value = (float)((int32_t)pattern - FIXTURE_CENTER) /
				    FIXTURE_DIVISOR;
		if (norm && fixture->case_index != FIXTURE_BFLOAT16_NORM_CASE) {
			float *const values =
				test_vision_at(fixture->arena, *offset);
			values[index] = 1.0f + value;
		} else {
			uint16_t *const values =
				test_vision_at(fixture->arena, *offset);
			values[index] =
				test_vision_bfloat(norm ? 1.0f + value : value);
		}
	}
	(*tensor_id)++;
	*offset += (bytes + VISION_ALIGNMENT - 1u) / VISION_ALIGNMENT *
		   VISION_ALIGNMENT;
	return range;
}

bool test_vision_fixture_create(uint32_t case_index,
				struct test_vision_fixture *fixture)
{
	if (fixture == NULL || case_index >= TEST_VISION_CASES) {
		return false;
	}
	memset(fixture, 0, sizeof(*fixture));
	fixture->arena = calloc(1, TEST_VISION_ARENA_BYTES);
	if (fixture->arena == NULL) {
		return false;
	}
	fixture->case_index = case_index;
	struct vision_config config = vision_official_config();
	config.layer_count = FIXTURE_LAYERS;
	config.width = FIXTURE_WIDTH;
	config.heads = FIXTURE_HEADS;
	config.intermediate_width = FIXTURE_INTERMEDIATE;
	config.language_width = FIXTURE_WIDTH;
	config.patch_size = FIXTURE_PATCH;
	uint64_t offset = FIXTURE_FIRST_OFFSET;
	uint32_t tensor_id = 0;
	struct vision_weights *const weights = &fixture->weights;
	const uint32_t patch_width =
		VISION_CHANNELS * FIXTURE_PATCH * FIXTURE_PATCH;
	weights->patch = tensor_add(fixture, &offset, &tensor_id,
				    FIXTURE_WIDTH * patch_width, false);
	weights->patch_bias =
		tensor_add(fixture, &offset, &tensor_id, FIXTURE_WIDTH, false);
	for (uint32_t layer = 0; layer < FIXTURE_LAYERS; layer++) {
		struct vision_layer_weights *const current =
			&weights->layers[layer];
		current->norm1 = tensor_add(fixture, &offset, &tensor_id,
					    FIXTURE_WIDTH, true);
		current->qkv = tensor_add(fixture, &offset, &tensor_id,
					  VISION_QKV_COMPONENTS *
						  FIXTURE_WIDTH * FIXTURE_WIDTH,
					  false);
		current->qkv_bias = tensor_add(
			fixture, &offset, &tensor_id,
			VISION_QKV_COMPONENTS * FIXTURE_WIDTH, false);
		current->attention_output =
			tensor_add(fixture, &offset, &tensor_id,
				   FIXTURE_WIDTH * FIXTURE_WIDTH, false);
		current->attention_output_bias = tensor_add(
			fixture, &offset, &tensor_id, FIXTURE_WIDTH, false);
		current->norm2 = tensor_add(fixture, &offset, &tensor_id,
					    FIXTURE_WIDTH, true);
		current->gate_up =
			tensor_add(fixture, &offset, &tensor_id,
				   VISION_GATE_COMPONENTS *
					   FIXTURE_INTERMEDIATE * FIXTURE_WIDTH,
				   false);
		current->down = tensor_add(fixture, &offset, &tensor_id,
					   FIXTURE_WIDTH * FIXTURE_INTERMEDIATE,
					   false);
	}
	weights->final_norm =
		tensor_add(fixture, &offset, &tensor_id, FIXTURE_WIDTH, true);
	weights->projector1 = tensor_add(fixture, &offset, &tensor_id,
					 FIXTURE_WIDTH * FIXTURE_WIDTH *
						 VISION_DOWNSAMPLE_RATIO *
						 VISION_DOWNSAMPLE_RATIO,
					 false);
	weights->projector1_bias =
		tensor_add(fixture, &offset, &tensor_id, FIXTURE_WIDTH, false);
	weights->projector2 = tensor_add(fixture, &offset, &tensor_id,
					 FIXTURE_WIDTH * FIXTURE_WIDTH, false);
	weights->projector2_bias =
		tensor_add(fixture, &offset, &tensor_id, FIXTURE_WIDTH, false);
	weights->image_start =
		tensor_add(fixture, &offset, &tensor_id, FIXTURE_WIDTH, false);
	weights->image_newline =
		tensor_add(fixture, &offset, &tensor_id, FIXTURE_WIDTH, false);
	weights->image_end =
		tensor_add(fixture, &offset, &tensor_id, FIXTURE_WIDTH, false);
	const uint32_t height = fixture_shapes[case_index][0];
	const uint32_t width = fixture_shapes[case_index][1];
	const struct linear_range patches =
		tensor_add(fixture, &offset, &tensor_id,
			   height * width * patch_width, false);
	const uint64_t output_bytes =
		(uint64_t)((height + config.downsample_ratio - 1u) /
			   config.downsample_ratio) *
		((width + config.downsample_ratio - 1u) /
		 config.downsample_ratio) *
		FIXTURE_WIDTH * sizeof(uint16_t);
	const struct linear_range output = { .byte_offset = offset,
					     .byte_count = output_bytes };
	offset += (output_bytes + VISION_ALIGNMENT - 1u) / VISION_ALIGNMENT *
		  VISION_ALIGNMENT;
	struct vision_scratch scratch;
	if (vision_scratch_plan(&config, height, width, offset, &scratch) !=
		    STATUS_OK ||
	    scratch.byte_count > TEST_VISION_ARENA_BYTES - offset) {
		test_vision_fixture_destroy(fixture);
		return false;
	}
	fixture->description = (struct vision_description){
		.config = config,
		.weights = weights,
		.patches = patches,
		.output = output,
		.scratch = { .byte_offset = offset,
			     .byte_count = scratch.byte_count },
		.grid_height = height,
		.grid_width = width
	};
	fixture->kernels.arithmetic_class = ARITHMETIC_CLASS_SM_80;
	for (uint32_t operation = 0; operation < VISION_OPERATIONS;
	     operation++) {
		fixture->kernels.operations[operation] = operation;
	}
	if (vision_prepare(&fixture->description, TEST_VISION_ARENA_BYTES,
			   &fixture->kernels, fixture->launches,
			   VISION_MAXIMUM_LAUNCHES,
			   &fixture->launch_count) != STATUS_OK) {
		test_vision_fixture_destroy(fixture);
		return false;
	}
	return true;
}

void test_vision_fixture_destroy(struct test_vision_fixture *fixture)
{
	if (fixture != NULL) {
		free(fixture->arena);
		fixture->arena = NULL;
	}
}

static bool fixture_file_close(FILE *file, bool success)
{
	return fclose(file) == 0 && success;
}

static bool fixture_golden(const struct test_vision_fixture *fixture,
			   const unsigned char *arena, bool math_backend)
{
	char name[FIXTURE_NAME_CAPACITY];
	const int length = snprintf(name, sizeof(name),
				    "tools/vision/fixtures/vit_%s%u.hex",
				    math_backend ? "math_" : "",
				    fixture->case_index);
	if (length <= 0 || (uint32_t)length >= sizeof(name)) {
		return false;
	}
	FILE *const file = fopen(name, "rb");
	if (file == NULL) {
		return false;
	}
	const uint64_t elements =
		fixture->description.output.byte_count / sizeof(uint16_t);
	if (elements > FIXTURE_EXPECTED_WORDS) {
		return fixture_file_close(file, false);
	}
	uint16_t expected[FIXTURE_EXPECTED_WORDS];
	for (uint32_t index = 0; index < (uint32_t)elements; index++) {
		if (fscanf(file, "%hx", &expected[index]) != 1) {
			return fixture_file_close(file, false);
		}
	}
	int next = fgetc(file);
	while (next == ' ' || next == '\n' || next == '\r' || next == '\t') {
		next = fgetc(file);
	}
	const bool read = next == EOF && ferror(file) == 0;
	const bool closed = fclose(file) == 0;
	if (!read || !closed) {
		return false;
	}
	const uint16_t *const actual = test_vision_const_at(
		arena, fixture->description.output.byte_offset);
	uint32_t different_words = 0;
	for (uint64_t index = 0; index < elements; index++) {
		different_words += actual[index] != expected[index] ? 1u : 0u;
		const float wanted = test_vision_float(expected[index]);
		const float difference =
			fabsf(test_vision_float(actual[index]) - wanted);
		if ((!math_backend && actual[index] != expected[index]) ||
		    !isfinite(difference) ||
		    difference >
			    fabsf(wanted) * (math_backend ?
						     TEST_VISION_GOLDEN_TOLERANCE :
						     TEST_VISION_DEFAULT_RELATIVE) +
				    (math_backend ?
					     TEST_VISION_GOLDEN_ABSOLUTE :
					     TEST_VISION_DEFAULT_ABSOLUTE)) {
			(void)printf(
				"vision golden case %u element %llu: %.9g expected %.9g\n",
				fixture->case_index, (unsigned long long)index,
				(double)test_vision_float(actual[index]),
				(double)wanted);
			return false;
		}
	}
	if (!math_backend) {
		(void)printf(
			"vision GPU Flash case %u: %u different BF16 words of %llu\n",
			fixture->case_index, different_words,
			(unsigned long long)elements);
	}
	return true;
}

bool test_vision_fixture_golden(const struct test_vision_fixture *fixture,
				const unsigned char *arena)
{
	return fixture_golden(fixture, arena, false);
}

bool test_vision_fixture_math_golden(const struct test_vision_fixture *fixture,
				     const unsigned char *arena)
{
	return fixture_golden(fixture, arena, true);
}
