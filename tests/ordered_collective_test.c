/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Ordered peer launch admission and its staged fallback, including cancellation-sensitive sums. */
#include <fenv.h>
#include <stdio.h>
#include <string.h>

#include "anemone/ordered_collective.h"

#define TEST_RANKS 5u
#define TEST_ELEMENTS 7u
#define TEST_ARRAY_BYTES (TEST_ELEMENTS * sizeof(float))
#define TEST_STEP_OFFSET 0u
#define TEST_TABLE_OFFSET 128u
#define TEST_REGION_OFFSET 512u
#define TEST_STRIDE 512ull
#define TEST_ARENA_BYTES \
	(TEST_REGION_OFFSET + ORDERED_COLLECTIVE_SLOTS * TEST_STRIDE)
#define TEST_INPUT_OFFSET ORDERED_COLLECTIVE_SIGNAL_BYTES
#define TEST_OUTPUT_OFFSET (TEST_INPUT_OFFSET + TEST_ARRAY_BYTES)
#define TEST_SCRATCH_OFFSET (TEST_OUTPUT_OFFSET + TEST_ARRAY_BYTES)
#define TEST_DEADLINE 1000000u
#define TEST_CANCELLATION 16777216.0f
#define TEST_SENTINEL (-99.0f)
#define TEST_WRONG_ALIGNMENT 2u
#define TEST_POISON_BYTE 0xA5
#define TEST_HANDLE_BASE 16u

static int failures;
static uint32_t prepared_count, fail_at;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

/* Compare the stored FP32 words, including the sign of zero; numeric equality is insufficient. */
static bool same_bits(const float *actual, const float *expected,
		      uint32_t count)
{
	for (uint32_t index = 0; index < count; index++) {
		uint32_t actual_word = 0, expected_word = 0;
		memcpy(&actual_word, &actual[index], sizeof(actual_word));
		memcpy(&expected_word, &expected[index], sizeof(expected_word));
		if (actual_word != expected_word) {
			return false;
		}
	}
	return true;
}

static struct ordered_collective_description description_of(void)
{
	return (struct ordered_collective_description){
		.arguments = { .peer_table_offset = TEST_TABLE_OFFSET,
			       .region_offset = TEST_REGION_OFFSET,
			       .input_offset = TEST_INPUT_OFFSET,
			       .output_offset = TEST_OUTPUT_OFFSET,
			       .scratch_offset = TEST_SCRATCH_OFFSET,
			       .step_offset = TEST_STEP_OFFSET,
			       .parity_stride = TEST_STRIDE,
			       .deadline_cycles = TEST_DEADLINE,
			       .element_count = TEST_ELEMENTS,
			       .rank_count = TEST_RANKS },
		.arena_byte_count = TEST_ARENA_BYTES,
		.operation = ORDERED_ALL_REDUCE
	};
}

static MUST_CHECK result_status kernel_prepare(struct device_card *card,
					       const char *file,
					       const char *name, uint32_t bytes,
					       uint32_t *kernel_out)
{
	(void)card;
	expect(strcmp(file, KERNEL_FILE_ORDERED_COLLECTIVE) == 0 &&
		       name != NULL &&
		       bytes == ORDERED_COLLECTIVE_ARGUMENT_BYTES,
	       "kernel names use the collective ABI");
	prepared_count++;
	if (prepared_count == fail_at) {
		return STATUS_UNSUPPORTED;
	}
	*kernel_out = TEST_HANDLE_BASE + prepared_count;
	return STATUS_OK;
}

static void test_prepare(void)
{
	const struct backend_operations backend = { .kernel_prepare =
							    kernel_prepare };
	struct device_card *const card = (struct device_card *)(uintptr_t)1;
	struct ordered_collective_kernels kernels = { 0 }, before;
	memset(&kernels, TEST_POISON_BYTE, sizeof(kernels));
	before = kernels;
	for (fail_at = 1; fail_at <= ORDERED_COLLECTIVE_OPERATIONS; fail_at++) {
		prepared_count = 0;
		expect(ordered_collective_kernels_prepare(&backend, card,
							  &kernels) ==
				       STATUS_UNSUPPORTED &&
			       memcmp(&kernels, &before, sizeof(kernels)) == 0,
		       "failed kernel lookup leaves output untouched");
	}
	fail_at = 0;
	prepared_count = 0;
	expect(ordered_collective_kernels_prepare(&backend, card, &kernels) ==
			       STATUS_OK &&
		       prepared_count == ORDERED_COLLECTIVE_OPERATIONS,
	       "every collective entry prepared");
	struct kernel_launch launch, untouched;
	memset(&launch, TEST_POISON_BYTE, sizeof(launch));
	untouched = launch;
	for (uint32_t operation = 0; operation < ORDERED_COLLECTIVE_OPERATIONS;
	     operation++) {
		struct ordered_collective_description description =
			description_of();
		description.operation =
			(enum ordered_collective_operation)operation;
		expect(ordered_collective_prepare(&description, &kernels,
						  &launch) == STATUS_OK &&
			       launch.grid[0] == 1 &&
			       launch.block[0] == ORDERED_COLLECTIVE_THREADS &&
			       launch.kernel == kernels.handles[operation] &&
			       memcmp(launch.argument, &description.arguments,
				      sizeof(description.arguments)) == 0,
		       "immutable record and single CTA geometry");
	}
	launch = untouched;
	struct ordered_collective_description invalid[] = {
		description_of(), description_of(), description_of(),
		description_of(), description_of(), description_of(),
		description_of(), description_of(), description_of(),
		description_of(), description_of(), description_of(),
		description_of(), description_of(), description_of()
	};
	enum invalid_case {
		NO_ELEMENTS,
		MANY_ELEMENTS,
		NO_RANKS,
		MANY_RANKS,
		BAD_RANK,
		RESERVED,
		NO_DEADLINE,
		BAD_REGION,
		BAD_STRIDE,
		OVERFLOW_STRIDE,
		TABLE_OVERLAP,
		STEP_OVERLAP,
		ARRAY_OVERLAP,
		SIGNAL_OVERLAP,
		BAD_OPERATION,
		INVALID_CASES
	};
	_Static_assert(sizeof(invalid) / sizeof(invalid[0]) == INVALID_CASES,
		       "all invalid cases initialized");
	invalid[NO_ELEMENTS].arguments.element_count = 0;
	invalid[MANY_ELEMENTS].arguments.element_count =
		ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS + 1u;
	invalid[NO_RANKS].arguments.rank_count = 0;
	invalid[MANY_RANKS].arguments.rank_count =
		ORDERED_COLLECTIVE_MAXIMUM_RANKS + 1u;
	invalid[BAD_RANK].arguments.rank = TEST_RANKS;
	invalid[RESERVED].arguments.reserved_zero = 1;
	invalid[NO_DEADLINE].arguments.deadline_cycles = 0;
	invalid[BAD_REGION].arguments.region_offset += TEST_WRONG_ALIGNMENT;
	invalid[BAD_STRIDE].arguments.parity_stride++;
	invalid[OVERFLOW_STRIDE].arguments.parity_stride =
		UINT64_MAX - ORDERED_COLLECTIVE_LINE_BYTES + 1u;
	invalid[TABLE_OVERLAP].arguments.peer_table_offset = TEST_REGION_OFFSET;
	invalid[STEP_OVERLAP].arguments.step_offset = TEST_TABLE_OFFSET;
	invalid[ARRAY_OVERLAP].arguments.output_offset = TEST_INPUT_OFFSET;
	invalid[SIGNAL_OVERLAP].arguments.input_offset = 0;
	invalid[BAD_OPERATION].operation = ORDERED_COLLECTIVE_OPERATIONS;
	for (uint32_t index = 0; index < INVALID_CASES; index++) {
		expect(ordered_collective_prepare(&invalid[index], &kernels,
						  &launch) ==
				       STATUS_INVALID_ARGUMENT &&
			       memcmp(&launch, &untouched, sizeof(launch)) == 0,
		       "invalid collective refuses without modifying the launch");
	}
	expect(ordered_collective_prepare(NULL, &kernels, &launch) ==
			       STATUS_INVALID_ARGUMENT &&
		       ordered_collective_prepare(&invalid[0], NULL, &launch) ==
			       STATUS_INVALID_ARGUMENT &&
		       ordered_collective_prepare(&invalid[0], &kernels,
						  NULL) ==
			       STATUS_INVALID_ARGUMENT,
	       "null descriptions, kernels and output refused");
}

static void test_host(void)
{
	static const float order[TEST_RANKS] = { TEST_CANCELLATION, 1.0f,
						 -TEST_CANCELLATION, 1.0f,
						 1.0f };
	float values[TEST_RANKS][TEST_ELEMENTS],
		actual[TEST_RANKS][TEST_ELEMENTS];
	float gathered[TEST_RANKS][TEST_ELEMENTS], expected[TEST_ELEMENTS];
	const float *inputs[TEST_RANKS];
	float *outputs[TEST_RANKS], *gather_outputs[TEST_RANKS];
	for (uint32_t rank = 0; rank < TEST_RANKS; rank++) {
		inputs[rank] = values[rank];
		outputs[rank] = actual[rank];
		gather_outputs[rank] = gathered[rank];
		for (uint32_t element = 0; element < TEST_ELEMENTS; element++) {
			values[rank][element] = order[rank];
			actual[rank][element] = TEST_SENTINEL;
		}
	}
	for (uint32_t element = 0; element < TEST_ELEMENTS; element++) {
		float sum = 0.0f;
		for (uint32_t rank = 0; rank < TEST_RANKS; rank++) {
			sum = (float)((double)sum +
				      (double)values[rank][element]);
		}
		expected[element] = sum;
	}
	for (uint32_t operation = 0; operation <= ORDERED_TWO_SHOT;
	     operation++) {
		expect(ordered_collective_host(
			       inputs, outputs, TEST_RANKS, TEST_ELEMENTS,
			       (enum ordered_collective_operation)operation) ==
			       STATUS_OK,
		       "staged fallback accepts ordered all-reduce");
		for (uint32_t rank = 0; rank < TEST_RANKS; rank++) {
			expect(same_bits(actual[rank], expected, TEST_ELEMENTS),
			       "cancellation-sensitive fallback has reference bits on every rank");
			for (uint32_t element = 0; element < TEST_ELEMENTS;
			     element++) {
				actual[rank][element] = TEST_SENTINEL;
			}
		}
	}
	for (uint32_t rank = 0; rank < TEST_RANKS; rank++) {
		inputs[rank] = values[TEST_RANKS - 1u - rank];
	}
	expect(ordered_collective_host(inputs, outputs, TEST_RANKS,
				       TEST_ELEMENTS,
				       ORDERED_ALL_REDUCE) == STATUS_OK &&
		       !same_bits(actual[0], expected, TEST_ELEMENTS),
	       "the bit gate rejects a changed summation order");
	for (uint32_t rank = 0; rank < TEST_RANKS; rank++) {
		inputs[rank] = values[rank];
		for (uint32_t element = 0; element < TEST_ELEMENTS; element++) {
			actual[rank][element] = TEST_SENTINEL;
		}
	}
	expect(ordered_collective_host(inputs, outputs, TEST_RANKS,
				       TEST_ELEMENTS,
				       ORDERED_REDUCE_SCATTER) == STATUS_OK,
	       "staged reduce-scatter accepts");
	const uint32_t slice = (TEST_ELEMENTS + TEST_RANKS - 1u) / TEST_RANKS;
	for (uint32_t rank = 0; rank < TEST_RANKS; rank++) {
		for (uint32_t element = 0; element < TEST_ELEMENTS; element++) {
			expect(actual[rank][element] ==
				       (element / slice == rank ?
						expected[element] :
						TEST_SENTINEL),
			       "reduce-scatter writes only the owner's indexed slice");
		}
		inputs[rank] = actual[rank];
	}
	expect(ordered_collective_host(inputs, gather_outputs, TEST_RANKS,
				       TEST_ELEMENTS,
				       ORDERED_GATHER) == STATUS_OK,
	       "staged gather accepts");
	for (uint32_t rank = 0; rank < TEST_RANKS; rank++) {
		expect(same_bits(gathered[rank], expected, TEST_ELEMENTS),
		       "standalone scatter and gather compose to identical all-reduce bits");
	}
	inputs[0] = gathered[0];
	expect(ordered_collective_host(inputs, gather_outputs, TEST_RANKS,
				       TEST_ELEMENTS, ORDERED_ALL_REDUCE) ==
		       STATUS_INVALID_ARGUMENT,
	       "overlapping fallback buffers refused");
	inputs[0] = NULL;
	expect(ordered_collective_host(inputs, outputs, TEST_RANKS,
				       TEST_ELEMENTS, ORDERED_ALL_REDUCE) ==
			       STATUS_INVALID_ARGUMENT &&
		       ordered_collective_host(
			       NULL, outputs, TEST_RANKS, TEST_ELEMENTS,
			       ORDERED_ALL_REDUCE) == STATUS_INVALID_ARGUMENT,
	       "invalid fallback inputs refused");
	inputs[0] = values[0];
	if (fesetround(FE_UPWARD) == 0) {
		expect(ordered_collective_host(
			       inputs, outputs, TEST_RANKS, TEST_ELEMENTS,
			       ORDERED_ALL_REDUCE) == STATUS_INVALID_ARGUMENT,
		       "fallback refuses another rounding mode");
		expect(fesetround(FE_TONEAREST) == 0, "rounding restored");
	}
}

int main(void)
{
	test_prepare();
	test_host();
	(void)printf("ordered_collective_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? 0 : 1;
}
