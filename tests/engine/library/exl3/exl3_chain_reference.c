/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "exl3_chain_reference.h"

#include "exl3_reference.h"
#include "support/mma_class_reference.h"

#define TILE_INPUTS 16u

void test_exl3_decode_matrix(const unsigned char *tiles, uint32_t inputs,
			     uint32_t outputs, uint32_t stored_width,
			     uint16_t *weights_out)
{
	for (uint32_t input = 0; input < inputs; input++)
		for (uint32_t output = 0; output < outputs; output++)
			weights_out[(uint64_t)input * outputs + output] =
				test_exl3_reconstruct(tiles, inputs, outputs,
						      stored_width, input,
						      output);
}

/* One instruction's row of A: the weights of K tile `tile` at `output`. */
static void tile_row(const uint16_t *weights, uint32_t outputs, uint32_t tile,
		     uint32_t output, uint16_t row[TEST_MMA_PRODUCTS])
{
	for (uint32_t k = 0; k < TEST_MMA_PRODUCTS; k++)
		row[k] = weights[((uint64_t)tile * TILE_INPUTS + k) * outputs +
				 output];
}

/* One output of one column, fed its column's operands: the slices' chains, then their sum in slice order. */
static float chain_output(uint32_t sum_class, const uint16_t *weights,
			  uint32_t inputs, uint32_t outputs, uint32_t slices,
			  const uint16_t *fed, uint32_t output)
{
	const uint32_t tile_count = inputs / TILE_INPUTS;
	float sum = 0.0f;
	for (uint32_t slice = 0; slice < slices; slice++) {
		float running = 0.0f;
		const uint32_t end = tile_count * (slice + 1u) / slices;
		for (uint32_t tile = tile_count * slice / slices; tile < end;
		     tile++) {
			uint16_t row[TEST_MMA_PRODUCTS];
			tile_row(weights, outputs, tile, output, row);
			running = test_mma_class_sum(
				sum_class, running, row,
				fed + (uint64_t)tile * TILE_INPUTS);
		}
		sum += running;
	}
	return sum;
}

void test_exl3_chain(uint32_t sum_class, const uint16_t *weights,
		     uint32_t inputs, uint32_t outputs, uint32_t slices,
		     const uint16_t *operands, uint32_t columns,
		     float *products_out)
{
	for (uint32_t column = 0; column < columns; column++)
		for (uint32_t output = 0; output < outputs; output++)
			products_out[(uint64_t)column * outputs + output] =
				chain_output(sum_class, weights, inputs,
					     outputs, slices,
					     operands +
						     (uint64_t)column * inputs,
					     output);
}
