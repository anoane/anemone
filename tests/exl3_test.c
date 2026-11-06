/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "anemone/exl3.h"
#include "exl3_reference.h"
#include "attention_state_reference.h"

#define WIDTH 128u
#define WIDE_INPUT 5120u
#define INNER 2304u
#define K2_BYTES 2963968ull
#define K3_BYTES 4438528ull
#define SENTINEL 0xBA5Eull
#define K2 2u
#define K3 3u
#define BAD_BITS 4u
#define CODEBOOK_STATES 65536u
#define SCALE_BITS 0x1EEEu
#define BIAS_BITS 0xC931u
#define MAXIMUM_BYTE_SUM 1020u
#define BASE_VALUE 1024.0
#define TILE 16u
#define TILE_VALUES 256u
#define OUTPUT_OFFSET 65536ull
#define ARENA_BYTES 131072ull
#define FAKE_KERNEL 1u

static bool expect(bool condition, const char *what)
{
	if (!condition) {
		(void)fprintf(stderr, "EXL3 failure: %s\n", what);
	}
	return condition;
}

int main(void)
{
	bool passed = true;
	const struct weight_format_operations *const format =
		&exl3_weight_operations;
	struct matrix_shape_record shape = { INNER, WIDE_INPUT, K2, 0 };
	uint64_t count = SENTINEL;
	passed =
		expect(format->stored_byte_count(&shape, &count) == STATUS_OK &&
			       count == K2_BYTES,
		       "real K2 byte count") &&
		passed;
	shape.bits_per_weight = K3;
	passed =
		expect(format->stored_byte_count(&shape, &count) == STATUS_OK &&
			       count == K3_BYTES,
		       "real K3 byte count") &&
		passed;
	shape.bits_per_weight = BAD_BITS;
	count = SENTINEL;
	passed = expect(format->stored_byte_count(&shape, &count) ==
					STATUS_INVALID_ARGUMENT &&
				count == SENTINEL,
			"unsupported bitrate preserves output") &&
		 passed;
	shape.bits_per_weight = K2;
	shape.input_size--;
	passed = expect(format->stored_byte_count(&shape, &count) ==
					STATUS_INVALID_ARGUMENT &&
				count == SENTINEL,
			"partial transform refused") &&
		 passed;
	passed = expect(format->stored_byte_count(NULL, &count) ==
					STATUS_INVALID_ARGUMENT &&
				count == SENTINEL,
			"null shape refused") &&
		 passed;
	shape.input_size = WIDE_INPUT;
	shape.reserved_must_be_zero = 1;
	passed = expect(format->stored_byte_count(&shape, &count) ==
				STATUS_INVALID_ARGUMENT,
			"reserved field refused") &&
		 passed;
	for (uint32_t state = 0; state < CODEBOOK_STATES; state++) {
		const double decoded =
			test_attention_half_value(test_exl3_codebook(state));
		const double low =
			BASE_VALUE * test_attention_half_value(SCALE_BITS) +
			test_attention_half_value(BIAS_BITS);
		const double high =
			(BASE_VALUE + MAXIMUM_BYTE_SUM) *
				test_attention_half_value(SCALE_BITS) +
			test_attention_half_value(BIAS_BITS);
		passed = expect(decoded >= low && decoded <= high,
				"codebook finite range") &&
			 passed;
	}
	bool seen[TILE_VALUES] = { false };
	for (uint32_t input = 0; input < TILE; input++) {
		for (uint32_t output = 0; output < TILE; output++) {
			const uint32_t position =
				test_exl3_position(input, output);
			if (position >= TILE_VALUES || seen[position]) {
				passed = expect(false,
						"permutation bijection") &&
					 passed;
			} else {
				seen[position] = true;
			}
		}
	}
	struct exl3_reconstruct_record record = { 0,	 OUTPUT_OFFSET,
						  WIDTH, WIDTH,
						  K3,	 0 };
	struct kernel_launch launch, before;
	memset(&launch, 0, sizeof(launch));
	passed = expect(exl3_reconstruct_prepare(&record, ARENA_BYTES,
						 FAKE_KERNEL,
						 &launch) == STATUS_OK,
			"valid disjoint launch") &&
		 passed;
	before = launch;
	record.output_offset = 0;
	passed = expect(exl3_reconstruct_prepare(&record, ARENA_BYTES,
						 FAKE_KERNEL, &launch) ==
					STATUS_INVALID_ARGUMENT &&
				memcmp(&launch, &before, sizeof(launch)) == 0,
			"overlap refused without changing launch") &&
		 passed;
	record.output_offset = OUTPUT_OFFSET;
	passed = expect(exl3_reconstruct_prepare(&record, OUTPUT_OFFSET,
						 FAKE_KERNEL, &launch) ==
					STATUS_INVALID_ARGUMENT &&
				memcmp(&launch, &before, sizeof(launch)) == 0,
			"short arena refused without changing launch") &&
		 passed;
	(void)printf(
		"EXL3 CPU %s: shape/admission, 65536 codebook states, permutation, launch bounds\n",
		passed ? "PASS" : "FAIL");
	return passed ? 0 : 1;
}
