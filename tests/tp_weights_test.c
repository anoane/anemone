/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original-format shard bytes against the independent checkpoint-layout/decode oracle, including scale edges. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "anemone/tp_weights.h"
#include "weight_format_reference.h"

#define SOURCE_OUTPUTS 80u
#define SOURCE_INPUTS 192u
#define MOST_CODES (SOURCE_OUTPUTS * SOURCE_INPUTS)
#define MOST_SCALES (MOST_CODES / 32u)
#define MOST_STORED 16384u
#define OUTPUT_TILE 16u
#define FP8_SCALE_OUTPUTS 32u
#define SCALE_INPUTS 32u
#define MXFP4_LOAD_INPUTS 64u
#define MXFP4_BITS 4u
#define FP8_BITS 8u
#define TEST_POISON 0xA5
#define MXFP4_CODE_MASK 15u
#define FP8_CODE_MASK 0x7Eu
#define FP8_SIGN_SHIFT 7u
#define CODE_OUTPUT_MULTIPLIER 17u
#define CODE_INPUT_MULTIPLIER 29u
#define SCALE_OUTPUT_MULTIPLIER 7u
#define SCALE_INPUT_MULTIPLIER 11u
#define SCALE_COUNT 17u
#define SCALE_BASE 120u
#define TEST_WRONG_WIDTH 2u
enum invalid_rectangle {
	EMPTY_SOURCE_OUTPUTS,
	EMPTY_SOURCE_INPUTS,
	WRONG_SOURCE_BITS,
	SOURCE_RESERVED,
	EMPTY_OUTPUTS,
	EMPTY_INPUTS,
	LARGE_FIRST_OUTPUT,
	LARGE_FIRST_INPUT,
	LARGE_OUTPUT_COUNT,
	LARGE_INPUT_COUNT,
	MISALIGNED_FIRST_OUTPUT,
	MISALIGNED_FIRST_INPUT,
	MISALIGNED_OUTPUT_COUNT,
	MISALIGNED_INPUT_COUNT,
	FIRST_OUTPUT_AT_END,
	FIRST_INPUT_AT_END,
	MISALIGNED_SOURCE_OUTPUTS,
	MISALIGNED_SOURCE_INPUTS,
	MXFP4_HALF_LOAD_ORIGIN,
	OUTPUTS_PAST_END,
	INVALID_RECTANGLE_COUNT
};

static int failures;
static uint32_t rectangles;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

static struct tensor_shard_description
description_of(enum weight_format_code format)
{
	return (struct tensor_shard_description){
		.format = format,
		.source_shape = { SOURCE_OUTPUTS, SOURCE_INPUTS,
				  format == WEIGHT_FORMAT_E2M1_SCALED ?
					  MXFP4_BITS :
					  FP8_BITS,
				  0 },
		.output_count = SOURCE_OUTPUTS,
		.input_count = SOURCE_INPUTS
	};
}

static void test_rectangles(enum weight_format_code format)
{
	uint8_t scales[MOST_SCALES] = { 0 };
	uint8_t codes[MOST_CODES], source[MOST_STORED],
		destination[MOST_STORED];
	struct tensor_shard_description description = description_of(format);
	const bool mxfp4 = format == WEIGHT_FORMAT_E2M1_SCALED;
	const uint32_t input_granule = mxfp4 ? MXFP4_LOAD_INPUTS : SCALE_INPUTS;
	const uint32_t output_granule = mxfp4 ? OUTPUT_TILE : FP8_SCALE_OUTPUTS;
	const uint32_t scale_rows =
		mxfp4 ? SOURCE_OUTPUTS :
			(SOURCE_OUTPUTS + FP8_SCALE_OUTPUTS - 1) /
				FP8_SCALE_OUTPUTS;
	for (uint32_t output = 0; output < SOURCE_OUTPUTS; output++) {
		for (uint32_t input = 0; input < SOURCE_INPUTS; input++) {
			const uint32_t value = output * CODE_OUTPUT_MULTIPLIER +
					       input * CODE_INPUT_MULTIPLIER;
			codes[output * SOURCE_INPUTS + input] =
				(uint8_t)(mxfp4 ? value & MXFP4_CODE_MASK :
						  (value & FP8_CODE_MASK) |
							  ((output & 1u)
							   << FP8_SIGN_SHIFT));
		}
	}
	for (uint32_t output = 0; output < scale_rows; output++) {
		for (uint32_t input = 0; input < SOURCE_INPUTS / SCALE_INPUTS;
		     input++) {
			scales[output * (SOURCE_INPUTS / SCALE_INPUTS) + input] =
				(uint8_t)(SCALE_BASE +
					  (output * SCALE_OUTPUT_MULTIPLIER +
					   input * SCALE_INPUT_MULTIPLIER) %
						  SCALE_COUNT);
		}
	}
	struct test_weight_format_sizes sizes;
	if (test_weight_format_sizes_of(format, &description.source_shape,
					&sizes) != STATUS_OK ||
	    test_weight_format_lay_out(
		    format, &description.source_shape, codes, sizeof(codes),
		    scales,
		    (uint64_t)scale_rows * (SOURCE_INPUTS / SCALE_INPUTS),
		    source, sizes.stored_byte_count) != STATUS_OK) {
		expect(false, "independent source lay-out");
		return;
	}
	for (uint32_t first_output = 0; first_output < SOURCE_OUTPUTS;
	     first_output += output_granule) {
		for (uint32_t outputs = OUTPUT_TILE;
		     outputs <= SOURCE_OUTPUTS - first_output;
		     outputs += OUTPUT_TILE) {
			for (uint32_t first_input = 0;
			     first_input < SOURCE_INPUTS;
			     first_input += input_granule) {
				for (uint32_t inputs = input_granule;
				     inputs <= SOURCE_INPUTS - first_input;
				     inputs += input_granule) {
					description.first_output = first_output;
					description.output_count = outputs;
					description.first_input = first_input;
					description.input_count = inputs;
					struct tensor_shard_plan plan;
					memset(destination, TEST_POISON,
					       sizeof(destination));
					if (tensor_shard_prepare(&description,
								 &plan) !=
						    STATUS_OK ||
					    tensor_shard_extract(
						    &plan, source,
						    sizes.stored_byte_count,
						    destination,
						    sizeof(destination)) !=
						    STATUS_OK) {
						expect(false,
						       "valid rectangle is extracted");
						return;
					}
					rectangles++;
					for (uint32_t output = 0;
					     output < outputs; output++) {
						for (uint32_t input = 0;
						     input < inputs; input++) {
							struct test_weight_format_weight
								original,
								shard;
							const result_status original_status = test_weight_format_read_weight(
								format,
								&description
									 .source_shape,
								source,
								sizes.stored_byte_count,
								first_output +
									output,
								first_input +
									input,
								&original);
							const result_status shard_status =
								test_weight_format_read_weight(
									format,
									&plan.shard_shape,
									destination,
									plan.shard_byte_count,
									output,
									input,
									&shard);
							expect(original_status ==
									       STATUS_OK &&
								       shard_status ==
									       STATUS_OK &&
								       original.code ==
									       shard.code &&
								       original.scale_code ==
									       shard.scale_code &&
								       original.value ==
									       shard.value,
							       "every shard code and scale is "
							       "the original checkpoint's");
						}
					}
					for (uint64_t index =
						     plan.shard_byte_count;
					     index < sizeof(destination);
					     index++) {
						expect(destination[index] ==
							       TEST_POISON,
						       "destination suffix stays untouched");
					}
					if (outputs == SOURCE_OUTPUTS &&
					    inputs == SOURCE_INPUTS) {
						expect(memcmp(source,
							      destination,
							      (size_t)sizes
								      .stored_byte_count) ==
							       0,
						       "whole-matrix extraction keeps every stored byte");
						struct test_weight_format_weight
							original,
							corrupted;
						destination[0] ^= 1u;
						const result_status original_status =
							test_weight_format_read_weight(
								format,
								&description
									 .source_shape,
								source,
								sizes.stored_byte_count,
								0, 0,
								&original);
						const result_status corrupt_status =
							test_weight_format_read_weight(
								format,
								&plan.shard_shape,
								destination,
								plan.shard_byte_count,
								0, 0,
								&corrupted);
						expect(original_status ==
								       STATUS_OK &&
							       corrupt_status ==
								       STATUS_OK &&
							       original.code !=
								       corrupted
									       .code,
						       "one changed stored bit is detected by the independent oracle");
					}
				}
			}
		}
	}
}

static bool same_shape(const struct matrix_shape_record *first,
		       const struct matrix_shape_record *second)
{
	return first->output_size == second->output_size &&
	       first->input_size == second->input_size &&
	       first->bits_per_weight == second->bits_per_weight &&
	       first->reserved_must_be_zero == second->reserved_must_be_zero;
}

static bool same_plan(const struct tensor_shard_plan *first,
		      const struct tensor_shard_plan *second)
{
	return first->description.format == second->description.format &&
	       same_shape(&first->description.source_shape,
			  &second->description.source_shape) &&
	       first->description.first_output ==
		       second->description.first_output &&
	       first->description.output_count ==
		       second->description.output_count &&
	       first->description.first_input ==
		       second->description.first_input &&
	       first->description.input_count ==
		       second->description.input_count &&
	       same_shape(&first->shard_shape, &second->shard_shape) &&
	       first->source_byte_count == second->source_byte_count &&
	       first->shard_byte_count == second->shard_byte_count;
}

static void test_refusals(void)
{
	const struct tensor_shard_description good =
		description_of(WEIGHT_FORMAT_E2M1_SCALED);
	struct tensor_shard_plan plan, before;
	memset(&plan, TEST_POISON, sizeof(plan));
	before = plan;
	expect(tensor_shard_prepare(NULL, &plan) == STATUS_INVALID_ARGUMENT &&
		       same_plan(&plan, &before),
	       "null description leaves scalar output alone");
	expect(tensor_shard_prepare(&good, NULL) == STATUS_INVALID_ARGUMENT,
	       "null plan refused");
	for (uint32_t failure = 0; failure < INVALID_RECTANGLE_COUNT;
	     failure++) {
		struct tensor_shard_description bad = good;
		switch (failure) {
		case EMPTY_SOURCE_OUTPUTS:
			bad.source_shape.output_size = 0;
			break;
		case EMPTY_SOURCE_INPUTS:
			bad.source_shape.input_size = 0;
			break;
		case WRONG_SOURCE_BITS:
			bad.source_shape.bits_per_weight = FP8_BITS;
			break;
		case SOURCE_RESERVED:
			bad.source_shape.reserved_must_be_zero = 1;
			break;
		case EMPTY_OUTPUTS:
			bad.output_count = 0;
			break;
		case EMPTY_INPUTS:
			bad.input_count = 0;
			break;
		case LARGE_FIRST_OUTPUT:
			bad.first_output = UINT32_MAX;
			break;
		case LARGE_FIRST_INPUT:
			bad.first_input = UINT32_MAX;
			break;
		case LARGE_OUTPUT_COUNT:
			bad.output_count = UINT32_MAX;
			break;
		case LARGE_INPUT_COUNT:
			bad.input_count = UINT32_MAX;
			break;
		case MISALIGNED_FIRST_OUTPUT:
			bad.first_output = 1;
			break;
		case MISALIGNED_FIRST_INPUT:
			bad.first_input = 1;
			break;
		case MISALIGNED_OUTPUT_COUNT:
			bad.output_count = 1;
			break;
		case MISALIGNED_INPUT_COUNT:
			bad.input_count = TEST_WRONG_WIDTH;
			break;
		case FIRST_OUTPUT_AT_END:
			bad.first_output = SOURCE_OUTPUTS;
			break;
		case FIRST_INPUT_AT_END:
			bad.first_input = SOURCE_INPUTS;
			break;
		case MISALIGNED_SOURCE_OUTPUTS:
			bad.source_shape.output_size = SOURCE_OUTPUTS - 1;
			break;
		case MISALIGNED_SOURCE_INPUTS:
			bad.source_shape.input_size = SOURCE_INPUTS - 1;
			break;
		case MXFP4_HALF_LOAD_ORIGIN:
			bad.first_input = SCALE_INPUTS;
			bad.input_count = MXFP4_LOAD_INPUTS;
			break;
		default:
			bad.first_output = OUTPUT_TILE;
			break;
		}
		expect(tensor_shard_prepare(&bad, &plan) ==
				       STATUS_INVALID_ARGUMENT &&
			       same_plan(&plan, &before),
		       "invalid rectangle leaves scalar output alone");
	}
	struct tensor_shard_description unsupported = good;
	unsupported.format = WEIGHT_FORMAT_FLOAT16;
	expect(tensor_shard_prepare(&unsupported, &plan) == STATUS_UNSUPPORTED,
	       "unimplemented weight format refused");
	unsupported = description_of(WEIGHT_FORMAT_E4M3_SCALED);
	unsupported.first_output = OUTPUT_TILE;
	unsupported.output_count = OUTPUT_TILE;
	expect(tensor_shard_prepare(&unsupported, &plan) == STATUS_UNSUPPORTED,
	       "misaligned shared FP8 output scales cannot be copied losslessly");
	uint8_t source[MOST_STORED], destination[MOST_STORED],
		poison[MOST_STORED];
	memset(source, 0, sizeof(source));
	memset(destination, TEST_POISON, sizeof(destination));
	memcpy(poison, destination, sizeof(poison));
	if (tensor_shard_prepare(&good, &plan) != STATUS_OK) {
		expect(false, "valid plan for refusal tests");
		return;
	}
	expect(tensor_shard_extract(NULL, source, plan.source_byte_count,
				    destination, sizeof(destination)) ==
		       STATUS_INVALID_ARGUMENT,
	       "null extraction plan refused");
	expect(tensor_shard_extract(&plan, NULL, plan.source_byte_count,
				    destination, sizeof(destination)) ==
		       STATUS_INVALID_ARGUMENT,
	       "null source refused");
	expect(tensor_shard_extract(&plan, source, plan.source_byte_count, NULL,
				    sizeof(destination)) ==
		       STATUS_INVALID_ARGUMENT,
	       "null destination refused");
	expect(tensor_shard_extract(&plan, source, plan.source_byte_count - 1,
				    destination, sizeof(destination)) ==
		       STATUS_INVALID_ARGUMENT,
	       "short source refused");
	expect(tensor_shard_extract(&plan, source, plan.source_byte_count,
				    destination, plan.shard_byte_count - 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "short destination refused");
	expect(tensor_shard_extract(&plan, source, plan.source_byte_count,
				    source,
				    sizeof(source)) == STATUS_INVALID_ARGUMENT,
	       "overlapping buffers refused");
	before = plan;
	plan.shard_byte_count++;
	expect(tensor_shard_extract(&plan, source, plan.source_byte_count,
				    destination, sizeof(destination)) ==
		       STATUS_INVALID_ARGUMENT,
	       "tampered byte count refused");
	plan = before;
	plan.shard_shape.output_size = OUTPUT_TILE;
	expect(tensor_shard_extract(&plan, source, plan.source_byte_count,
				    destination, sizeof(destination)) ==
		       STATUS_INVALID_ARGUMENT,
	       "tampered shard shape refused");
	expect(memcmp(poison, destination, sizeof(poison)) == 0,
	       "all extraction refusals precede the first write");
}

#define EXL3_TEST_INPUTS 384u
#define EXL3_TEST_OUTPUTS 512u
#define EXL3_TEST_TRANSFORM 128u
#define EXL3_TEST_TILE 16u
#define EXL3_TEST_BYTES_PER_TILE_BIT 32u
#define EXL3_TEST_BITS_PER_BYTE 8u
#define EXL3_TEST_MIN_BITS 2u
#define EXL3_TEST_MAX_BITS 3u
#define EXL3_TEST_BAD_BITS 4u
#define EXL3_TEST_GUARD_BYTES 32u
#define EXL3_TEST_STORED_BYTES 75520u
#define EXL3_TEST_BUFFER_BYTES \
	(EXL3_TEST_STORED_BYTES + EXL3_TEST_GUARD_BYTES * 2u)
#define EXL3_TEST_INPUT_HASH 19u
#define EXL3_TEST_OUTPUT_HASH 31u
#define EXL3_TEST_BYTE_HASH 7u
#define EXL3_TEST_BITS_HASH 11u
#define EXL3_TEST_INPUT_SCALE_BASE 0x3400u
#define EXL3_TEST_OUTPUT_SCALE_BASE 0xB000u
#define EXL3_TEST_OUTPUT_SCALE_STEP 3u
#define EXL3_TEST_NAN_SCALE 0x7E01u
#define EXL3_TEST_INFINITY_SCALE 0x7C00u

enum exl3_invalid_rectangle {
	EXL3_BAD_SOURCE_INPUT,
	EXL3_BAD_SOURCE_OUTPUT,
	EXL3_BAD_BITS,
	EXL3_BAD_RESERVED,
	EXL3_EMPTY_INPUT,
	EXL3_EMPTY_OUTPUT,
	EXL3_BAD_INPUT_ORIGIN,
	EXL3_BAD_OUTPUT_ORIGIN,
	EXL3_BAD_INPUT_COUNT,
	EXL3_BAD_OUTPUT_COUNT,
	EXL3_INPUT_PAST_END,
	EXL3_OUTPUT_PAST_END,
	EXL3_INPUT_ORIGIN_OVERFLOW,
	EXL3_OUTPUT_COUNT_OVERFLOW,
	EXL3_INVALID_RECTANGLES
};

static uint8_t exl3_code_byte(uint32_t input_tile, uint32_t output_tile,
			      uint32_t byte, uint32_t bits)
{
	return (uint8_t)(input_tile * EXL3_TEST_INPUT_HASH +
			 output_tile * EXL3_TEST_OUTPUT_HASH +
			 byte * EXL3_TEST_BYTE_HASH +
			 bits * EXL3_TEST_BITS_HASH);
}

static uint16_t exl3_input_scale(uint32_t channel)
{
	return (uint16_t)(EXL3_TEST_INPUT_SCALE_BASE + channel);
}

static uint16_t exl3_output_scale(uint32_t channel)
{
	return (uint16_t)(EXL3_TEST_OUTPUT_SCALE_BASE +
			  channel * EXL3_TEST_OUTPUT_SCALE_STEP);
}

/* Source fill walks flat trellis bytes, independently of the rectangle oracle's nested tile traversal. */
static void exl3_source_fill(uint8_t *source, uint32_t bits)
{
	const uint32_t tile_bytes = EXL3_TEST_BYTES_PER_TILE_BIT * bits;
	const uint32_t code_bytes = EXL3_TEST_INPUTS * EXL3_TEST_OUTPUTS /
				    EXL3_TEST_BITS_PER_BYTE * bits;
	for (uint32_t byte = 0; byte < code_bytes; byte++) {
		const uint32_t tile = byte / tile_bytes;
		source[byte] = exl3_code_byte(
			tile / (EXL3_TEST_OUTPUTS / EXL3_TEST_TILE),
			tile % (EXL3_TEST_OUTPUTS / EXL3_TEST_TILE),
			byte % tile_bytes, bits);
	}
	for (uint32_t channel = 0;
	     channel < EXL3_TEST_INPUTS + EXL3_TEST_OUTPUTS; channel++) {
		const uint16_t scale =
			channel < EXL3_TEST_INPUTS ?
				exl3_input_scale(channel) :
				exl3_output_scale(channel - EXL3_TEST_INPUTS);
		memcpy(source + code_bytes + (size_t)channel * sizeof(scale),
		       &scale, sizeof(scale));
	}
}

/* The oracle visits every original tile and scale by global coordinates; no production packer is called. */
static void
exl3_rectangle_reference(uint8_t *expected,
			 const struct tensor_shard_description *description)
{
	const uint32_t bits = description->source_shape.bits_per_weight;
	const uint32_t tile_bytes = EXL3_TEST_BYTES_PER_TILE_BIT * bits;
	uint32_t next = 0;
	for (uint32_t input = 0;
	     input < description->input_count / EXL3_TEST_TILE; input++) {
		for (uint32_t output = 0;
		     output < description->output_count / EXL3_TEST_TILE;
		     output++) {
			for (uint32_t byte = 0; byte < tile_bytes; byte++) {
				expected[next++] = exl3_code_byte(
					description->first_input /
							EXL3_TEST_TILE +
						input,
					description->first_output /
							EXL3_TEST_TILE +
						output,
					byte, bits);
			}
		}
	}
	for (uint32_t input = 0; input < description->input_count; input++) {
		const uint16_t scale =
			exl3_input_scale(description->first_input + input);
		memcpy(expected + next, &scale, sizeof(scale));
		next += (uint32_t)sizeof(scale);
	}
	for (uint32_t output = 0; output < description->output_count;
	     output++) {
		const uint16_t scale =
			exl3_output_scale(description->first_output + output);
		memcpy(expected + next, &scale, sizeof(scale));
		next += (uint32_t)sizeof(scale);
	}
}

static struct tensor_shard_description exl3_description_of(uint32_t bits)
{
	const struct tensor_shard_description description = {
		.format = WEIGHT_FORMAT_EXL3,
		.source_shape = { EXL3_TEST_OUTPUTS, EXL3_TEST_INPUTS, bits,
				  0 },
		.output_count = EXL3_TEST_OUTPUTS,
		.input_count = EXL3_TEST_INPUTS
	};
	return description;
}

static void test_exl3_rectangles(void)
{
	uint8_t source[EXL3_TEST_BUFFER_BYTES],
		destination[EXL3_TEST_BUFFER_BYTES],
		expected[EXL3_TEST_BUFFER_BYTES];
	for (uint32_t bits = EXL3_TEST_MIN_BITS; bits <= EXL3_TEST_MAX_BITS;
	     bits++) {
		memset(source, TEST_POISON, sizeof(source));
		exl3_source_fill(source + EXL3_TEST_GUARD_BYTES, bits);
		for (uint32_t first_input = 0; first_input < EXL3_TEST_INPUTS;
		     first_input += EXL3_TEST_TRANSFORM) {
			for (uint32_t inputs = EXL3_TEST_TRANSFORM;
			     inputs <= EXL3_TEST_INPUTS - first_input;
			     inputs += EXL3_TEST_TRANSFORM) {
				for (uint32_t first_output = 0;
				     first_output < EXL3_TEST_OUTPUTS;
				     first_output += EXL3_TEST_TRANSFORM) {
					for (uint32_t outputs =
						     EXL3_TEST_TRANSFORM;
					     outputs <=
					     EXL3_TEST_OUTPUTS - first_output;
					     outputs += EXL3_TEST_TRANSFORM) {
						struct tensor_shard_description
							description =
								exl3_description_of(
									bits);
						description.first_input =
							first_input;
						description.input_count =
							inputs;
						description.first_output =
							first_output;
						description.output_count =
							outputs;
						struct tensor_shard_plan plan;
						memset(destination, TEST_POISON,
						       sizeof(destination));
						memset(expected, TEST_POISON,
						       sizeof(expected));
						exl3_rectangle_reference(
							expected +
								EXL3_TEST_GUARD_BYTES,
							&description);
						if (tensor_shard_prepare(
							    &description,
							    &plan) !=
							    STATUS_OK ||
						    tensor_shard_extract(
							    &plan,
							    source +
								    EXL3_TEST_GUARD_BYTES,
							    plan.source_byte_count,
							    destination +
								    EXL3_TEST_GUARD_BYTES,
							    EXL3_TEST_STORED_BYTES) !=
							    STATUS_OK) {
							expect(false,
							       "valid EXL3 generic rectangle is extracted");
							return;
						}
						rectangles++;
						expect(memcmp(destination,
							      expected,
							      sizeof(destination)) ==
							       0,
						       "EXL3 tiles, scales and both destination guards "
						       "match independent coordinates");
						destination
							[EXL3_TEST_GUARD_BYTES] ^=
							1u;
						expect(memcmp(destination,
							      expected,
							      sizeof(destination)) !=
							       0,
						       "EXL3 independent raw-byte oracle detects "
						       "deliberate one-byte corruption");
					}
				}
			}
		}
	}
}

static void test_exl3_refusals(void)
{
	struct tensor_shard_description good =
		exl3_description_of(EXL3_TEST_MAX_BITS);
	good.first_input = good.first_output = EXL3_TEST_TRANSFORM;
	good.input_count = good.output_count = EXL3_TEST_TRANSFORM;
	struct tensor_shard_plan plan, before;
	memset(&plan, TEST_POISON, sizeof(plan));
	before = plan;
	for (uint32_t invalid = 0; invalid < EXL3_INVALID_RECTANGLES;
	     invalid++) {
		struct tensor_shard_description bad = good;
		switch (invalid) {
		case EXL3_BAD_SOURCE_INPUT:
			bad.source_shape.input_size--;
			break;
		case EXL3_BAD_SOURCE_OUTPUT:
			bad.source_shape.output_size--;
			break;
		case EXL3_BAD_BITS:
			bad.source_shape.bits_per_weight = EXL3_TEST_BAD_BITS;
			break;
		case EXL3_BAD_RESERVED:
			bad.source_shape.reserved_must_be_zero = 1;
			break;
		case EXL3_EMPTY_INPUT:
			bad.input_count = 0;
			break;
		case EXL3_EMPTY_OUTPUT:
			bad.output_count = 0;
			break;
		case EXL3_BAD_INPUT_ORIGIN:
			bad.first_input++;
			break;
		case EXL3_BAD_OUTPUT_ORIGIN:
			bad.first_output++;
			break;
		case EXL3_BAD_INPUT_COUNT:
			bad.input_count--;
			break;
		case EXL3_BAD_OUTPUT_COUNT:
			bad.output_count--;
			break;
		case EXL3_INPUT_PAST_END:
			bad.first_input = EXL3_TEST_INPUTS;
			break;
		case EXL3_OUTPUT_PAST_END:
			bad.first_output = EXL3_TEST_OUTPUTS;
			break;
		case EXL3_INPUT_ORIGIN_OVERFLOW:
			bad.first_input = UINT32_MAX;
			break;
		default:
			bad.output_count = UINT32_MAX;
			break;
		}
		expect(tensor_shard_prepare(&bad, &plan) ==
				       STATUS_INVALID_ARGUMENT &&
			       same_plan(&plan, &before),
		       "EXL3 invalid generic geometry preserves every scalar output");
	}
	uint8_t source[EXL3_TEST_BUFFER_BYTES],
		before_source[EXL3_TEST_BUFFER_BYTES],
		destination[EXL3_TEST_BUFFER_BYTES],
		poison[EXL3_TEST_BUFFER_BYTES];
	memset(source, TEST_POISON, sizeof(source));
	exl3_source_fill(source + EXL3_TEST_GUARD_BYTES, EXL3_TEST_MAX_BITS);
	memcpy(before_source, source, sizeof(source));
	memset(destination, TEST_POISON, sizeof(destination));
	memcpy(poison, destination, sizeof(poison));
	if (tensor_shard_prepare(&good, &plan) != STATUS_OK) {
		expect(false, "valid EXL3 plan for negative controls");
		return;
	}
	expect(tensor_shard_extract(&plan, source + EXL3_TEST_GUARD_BYTES,
				    plan.source_byte_count - 1u,
				    destination + EXL3_TEST_GUARD_BYTES,
				    EXL3_TEST_STORED_BYTES) ==
		       STATUS_INVALID_ARGUMENT,
	       "EXL3 short source is refused");
	expect(tensor_shard_extract(&plan, source + EXL3_TEST_GUARD_BYTES,
				    plan.source_byte_count + 1u,
				    destination + EXL3_TEST_GUARD_BYTES,
				    EXL3_TEST_STORED_BYTES) ==
		       STATUS_INVALID_ARGUMENT,
	       "EXL3 extra source byte is refused");
	expect(tensor_shard_extract(&plan, source + EXL3_TEST_GUARD_BYTES,
				    plan.source_byte_count,
				    destination + EXL3_TEST_GUARD_BYTES,
				    plan.shard_byte_count - 1u) ==
		       STATUS_INVALID_ARGUMENT,
	       "EXL3 short destination is refused");
	expect(tensor_shard_extract(
		       &plan, source + EXL3_TEST_GUARD_BYTES,
		       plan.source_byte_count, source + EXL3_TEST_GUARD_BYTES,
		       plan.shard_byte_count) == STATUS_INVALID_ARGUMENT,
	       "EXL3 complete overlap is refused");
	expect(tensor_shard_extract(&plan, source + EXL3_TEST_GUARD_BYTES,
				    plan.source_byte_count,
				    source + EXL3_TEST_GUARD_BYTES + 1u,
				    plan.shard_byte_count) ==
		       STATUS_INVALID_ARGUMENT,
	       "EXL3 partial overlap is refused");
	before = plan;
	plan.shard_byte_count++;
	expect(tensor_shard_extract(&plan, source + EXL3_TEST_GUARD_BYTES,
				    plan.source_byte_count,
				    destination + EXL3_TEST_GUARD_BYTES,
				    EXL3_TEST_STORED_BYTES) ==
		       STATUS_INVALID_ARGUMENT,
	       "EXL3 changed prepared byte count is refused");
	plan = before;
	plan.shard_shape.input_size++;
	expect(tensor_shard_extract(&plan, source + EXL3_TEST_GUARD_BYTES,
				    plan.source_byte_count,
				    destination + EXL3_TEST_GUARD_BYTES,
				    EXL3_TEST_STORED_BYTES) ==
		       STATUS_INVALID_ARGUMENT,
	       "EXL3 changed prepared shape is refused");
	expect(memcmp(source, before_source, sizeof(source)) == 0 &&
		       memcmp(destination, poison, sizeof(destination)) == 0,
	       "EXL3 refusals precede every source/destination write");
	plan = before;
	const uint32_t source_codes = EXL3_TEST_INPUTS * EXL3_TEST_OUTPUTS /
				      EXL3_TEST_BITS_PER_BYTE *
				      EXL3_TEST_MAX_BITS;
	const uint32_t local_codes = good.input_count * good.output_count /
				     EXL3_TEST_BITS_PER_BYTE *
				     EXL3_TEST_MAX_BITS;
	const uint16_t nan_scale = EXL3_TEST_NAN_SCALE,
		       infinity_scale = EXL3_TEST_INFINITY_SCALE;
	memcpy(source + EXL3_TEST_GUARD_BYTES + source_codes +
		       (size_t)good.first_input * sizeof(nan_scale),
	       &nan_scale, sizeof(nan_scale));
	memcpy(source + EXL3_TEST_GUARD_BYTES + source_codes +
		       (size_t)(EXL3_TEST_INPUTS + good.first_output) *
			       sizeof(infinity_scale),
	       &infinity_scale, sizeof(infinity_scale));
	expect(tensor_shard_extract(&plan, source + EXL3_TEST_GUARD_BYTES,
				    plan.source_byte_count,
				    destination + EXL3_TEST_GUARD_BYTES,
				    EXL3_TEST_STORED_BYTES) == STATUS_OK,
	       "EXL3 raw copy delegates finite-scale admission to model adapter");
	uint16_t copied_nan = 0, copied_infinity = 0;
	memcpy(&copied_nan, destination + EXL3_TEST_GUARD_BYTES + local_codes,
	       sizeof(copied_nan));
	memcpy(&copied_infinity,
	       destination + EXL3_TEST_GUARD_BYTES + local_codes +
		       (size_t)good.input_count * sizeof(copied_infinity),
	       sizeof(copied_infinity));
	expect(copied_nan == nan_scale && copied_infinity == infinity_scale,
	       "EXL3 raw transport preserves nonfinite scale bits without selecting them for a model");
}

/* Plain 16-bit matrices: every row range of a small matrix is its source bytes; inputs are never cut. */
static void test_plain_rows(void)
{
	enum { PLAIN_OUTPUTS = 12, PLAIN_INPUTS = 6, PLAIN_ROW_BYTES = PLAIN_INPUTS * 2 };
	uint8_t source[PLAIN_OUTPUTS * PLAIN_ROW_BYTES], destination[PLAIN_OUTPUTS * PLAIN_ROW_BYTES + 1];
	for (uint32_t index = 0; index < sizeof(source); index++)
		source[index] = (uint8_t)(index * 7u + 3u);
	for (uint32_t first = 0; first < PLAIN_OUTPUTS; first++) {
		for (uint32_t count = 1; first + count <= PLAIN_OUTPUTS; count++) {
			const struct tensor_shard_description description = {
				.format = WEIGHT_FORMAT_NONE,
				.source_shape = { PLAIN_OUTPUTS, PLAIN_INPUTS,
						  TENSOR_SHARD_PLAIN_BITS, 0 },
				.first_output = first,
				.output_count = count,
				.first_input = 0,
				.input_count = PLAIN_INPUTS
			};
			struct tensor_shard_plan plan;
			memset(destination, 0xA5, sizeof(destination));
			const bool done =
				tensor_shard_prepare(&description, &plan) ==
					STATUS_OK &&
				plan.shard_byte_count ==
					(uint64_t)count * PLAIN_ROW_BYTES &&
				plan.source_byte_count == sizeof(source) &&
				tensor_shard_extract(&plan, source,
						     sizeof(source),
						     destination,
						     sizeof(destination)) ==
					STATUS_OK;
			expect(done && memcmp(destination, source + first * PLAIN_ROW_BYTES,
					      (size_t)count * PLAIN_ROW_BYTES) == 0 &&
				       destination[count * PLAIN_ROW_BYTES] == 0xA5,
			       "a plain row range copies its source rows and nothing behind");
			rectangles++;
		}
	}
	struct tensor_shard_description bad = {
		.format = WEIGHT_FORMAT_NONE,
		.source_shape = { PLAIN_OUTPUTS, PLAIN_INPUTS,
				  TENSOR_SHARD_PLAIN_BITS, 0 },
		.first_output = 0,
		.output_count = 1,
		.first_input = 2,
		.input_count = PLAIN_INPUTS - 2
	};
	struct tensor_shard_plan plan;
	expect(tensor_shard_prepare(&bad, &plan) == STATUS_INVALID_ARGUMENT,
	       "a plain matrix's inputs are never cut");
	bad.first_input = 0;
	bad.input_count = PLAIN_INPUTS;
	bad.source_shape.bits_per_weight = 0;
	expect(tensor_shard_prepare(&bad, &plan) == STATUS_INVALID_ARGUMENT,
	       "a plain matrix names 16-bit elements");
}

int main(void)
{
	test_plain_rows();
	test_rectangles(WEIGHT_FORMAT_E2M1_SCALED);
	test_rectangles(WEIGHT_FORMAT_E4M3_SCALED);
	test_refusals();
	test_exl3_rectangles();
	test_exl3_refusals();
	if (failures != 0) {
		return 1;
	}
	(void)printf(
		"tp_weights_test: %u rectangles preserve every original code and scale; refusals pass\n",
		rectangles);
	return 0;
}
