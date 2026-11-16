/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "exl3_reference.h"
#include "attention_state_reference.h"

#define STATE_BITS 16u
#define WORD_BITS 32u
#define BYTE_BITS 8u
#define BYTE_MASK 255u
#define MUL1_MULTIPLIER 0x83DCD12Du
#define MUL1_BASE 0x6400u
#define MUL1_SCALE 0x1EEEu
#define MUL1_BIAS 0xC931u
#define TILE_WIDTH 16u
#define TILE_VALUES 256u
#define WARP_LANES 32u
#define LANE_VALUES 8u
#define LANE_GROUP 4u
#define ROW_PAIR 2u
#define HIGH_ROW_BIT 2u
#define HIGH_COLUMN_BIT 4u
#define HALF_TILE 8u
#define WHOLE_WIDTH_MOST 8u
#define HALF_WIDTH_SCALE 16u

uint16_t test_exl3_codebook(uint32_t state)
{
	const uint32_t hashed = state * MUL1_MULTIPLIER;
	uint32_t sum = 0;
	for (uint32_t shift = 0; shift < WORD_BITS; shift += BYTE_BITS) {
		sum += (hashed >> shift) & BYTE_MASK;
	}
	const double value =
		test_attention_half_value((uint16_t)(MUL1_BASE + sum));
	return test_attention_half_nearest(
		value * test_attention_half_value(MUL1_SCALE) +
		test_attention_half_value(MUL1_BIAS));
}

uint32_t test_exl3_position(uint32_t input, uint32_t output)
{
	for (uint32_t lane = 0; lane < WARP_LANES; lane++) {
		for (uint32_t place = 0; place < LANE_VALUES; place++) {
			const uint32_t row =
				(lane % LANE_GROUP) * ROW_PAIR + (place & 1u) +
				((place & HIGH_ROW_BIT) != 0 ? HALF_TILE : 0);
			const uint32_t column =
				lane / LANE_GROUP +
				((place & HIGH_COLUMN_BIT) != 0 ? HALF_TILE :
								  0);
			if (row == input && column == output) {
				return lane * LANE_VALUES + place;
			}
		}
	}
	return UINT32_MAX;
}

/* Bits ring position `position` steps: a whole K's K, a half K's KA or KA + 1 (odd positions) */
static uint32_t ring_step(uint32_t stored_width, uint32_t position)
{
	return stored_width <= WHOLE_WIDTH_MOST ?
		       stored_width :
		       stored_width / HALF_WIDTH_SCALE + (position & 1u);
}

uint16_t test_exl3_state(const unsigned char *tiles, uint32_t inputs,
			 uint32_t outputs, uint32_t stored_width,
			 uint32_t input, uint32_t output)
{
	(void)inputs;
	uint32_t ring_bits = 0;
	for (uint32_t position = 0; position < TILE_VALUES; position++) {
		ring_bits += ring_step(stored_width, position);
	}
	const uint64_t tile =
		(uint64_t)(input / TILE_WIDTH) * (outputs / TILE_WIDTH) +
		output / TILE_WIDTH;
	const unsigned char *const bytes = tiles + tile * ring_bits / BYTE_BITS;
	const uint32_t position =
		test_exl3_position(input % TILE_WIDTH, output % TILE_WIDTH);
	uint32_t window_end = 0;
	for (uint32_t step = 0; step <= position; step++) {
		window_end += ring_step(stored_width, step);
	}
	uint32_t state = 0;
	for (uint32_t bit = 0; bit < STATE_BITS; bit++) {
		const uint32_t ring_bit =
			(window_end + ring_bits - STATE_BITS + bit) % ring_bits;
		/* MSB-first in little-endian uint32 words: bit 0 is a word's least significant */
		const uint32_t word_bit = WORD_BITS - 1u - ring_bit % WORD_BITS;
		const uint32_t byte =
			ring_bit / WORD_BITS * (WORD_BITS / BYTE_BITS) +
			word_bit / BYTE_BITS;
		state = state * ROW_PAIR +
			(((uint32_t)bytes[byte] >> (word_bit % BYTE_BITS)) &
			 1u);
	}
	return (uint16_t)state;
}

uint16_t test_exl3_reconstruct(const unsigned char *tiles, uint32_t inputs,
			       uint32_t outputs, uint32_t stored_width,
			       uint32_t input, uint32_t output)
{
	return test_exl3_codebook(test_exl3_state(tiles, inputs, outputs,
						  stored_width, input, output));
}
