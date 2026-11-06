/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Exhaustive BF16 special/sign/mantissa coverage at every admitted shared exponent, plus independent mixed pairs.
 * This qualifies the integer codec leaves, not peer publication, packet reuse or a model collective. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bf16_wire_codec.h"

#define BF16_VALUES 65536u
#define PAIR_MASK 65535u
#define HASH_MULTIPLIER 2654435761u
#define HASH_INCREMENT 1013904223u
#define RANDOM_PAIRS 1000000u

static uint64_t checked;
static int failures;

static void check(uint32_t word, uint32_t top)
{
	const uint32_t packed = bf16_wire_pair_pack(word, top);
	const uint32_t got = bf16_wire_pair_unpack(
		packed & PAIR_MASK, packed >> BF16_WIRE_HALF_BITS, top);
	if (got != word) {
		if (failures == 0)
			(void)printf("FAILED: BF16 wire %08" PRIx32
				     " -> %08" PRIx32 " at exponent %" PRIu32
				     "\n",
				     word, got, top);
		failures++;
	}
	checked++;
}

int main(void)
{
	for (uint32_t value = 0; value < BF16_VALUES; value++) {
		const uint32_t word = value | (value << BF16_WIRE_HALF_BITS);
		const uint32_t exponent = bf16_wire_low_exponent(word);
		for (uint32_t delta = 0;
		     delta <= BF16_WIRE_DELTA_MASK &&
		     exponent + delta <= BF16_WIRE_EXPONENT_MASK;
		     delta++)
			check(word, exponent + delta);
	}
	uint32_t mixed = HASH_INCREMENT;
	for (uint32_t pair = 0; pair < RANDOM_PAIRS; pair++) {
		mixed = mixed * HASH_MULTIPLIER + HASH_INCREMENT;
		const uint32_t low = bf16_wire_low_exponent(mixed);
		const uint32_t high = bf16_wire_high_exponent(mixed);
		const uint32_t top = low > high ? low : high;
		const uint32_t least = low < high ? low : high;
		if (top - least <= BF16_WIRE_DELTA_MASK)
			check(mixed, top);
	}
	(void)printf("BF16 wire integer codec: %" PRIu64
		     " pairs, %d failures\n",
		     checked, failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
