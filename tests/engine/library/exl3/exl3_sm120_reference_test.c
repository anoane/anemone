/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_sm120_reference_test.c: the host oracles of the EXL3 products on both classes of card, which the EXL3 card test
 * then holds the cards to.
 *   - mma_class_reference's two sum rules on instructions made by hand, whose answers follow from the rule:
 *     - both classes agree where a sum is exact;
 *     - they differ either way where it is not: class 120 keeps four 2^-25 products that class 80 truncates, and
 *       class 80 keeps eight 2^-26 products in their own group that class 120 truncates;
 *     - cancellation, a negative sum, a subnormal input, overflow to the largest finite value, and not-finite inputs.
 *     Then random instructions, each class within a bound of the exact sum.
 *   - exl3_chain_reference's products:
 *     - the known answer of an all-zero trellis: every weight the codebook's value of state 0, -3.453125, so 256 inputs
 *       of 1.0 give -884 exactly, on both classes and at any slice count;
 *     - random K2 and K3 matrices on both classes within a bound of the exact sum, the classes differing somewhere.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine/backend/backend_seam.h"
#include "models/deepseek_v41_flash/attention/ds41f_attention_state_reference.h"
#include "exl3_chain_reference.h"
#include "exl3_reference.h"
#include "support/mma_class_reference.h"

#define HALF_ZERO 0x0000u
#define HALF_ONE 0x3C00u
#define HALF_MINUS_ONE 0xBC00u
#define HALF_TWO_TO_MINUS_12 0x0C00u
#define HALF_MINUS_TWO_TO_MINUS_12 0x8C00u
#define HALF_TWO_TO_MINUS_13 0x0800u
#define HALF_SMALLEST_SUBNORMAL 0x0001u /* 2^-24 */
#define HALF_LARGEST_FINITE 0x7BFFu /* 65504 */
#define HALF_INFINITY 0x7C00u
#define HALF_NOT_A_NUMBER 0x7E00u
#define FLOAT_ONE_BITS 0x3F800000u
#define FLOAT_ONE_AND_ULP_BITS 0x3F800001u /* 1 + 2^-23 */
#define FLOAT_MINUS_ONE_BITS 0xBF800000u
#define FLOAT_MINUS_ONE_AND_ULP_BITS 0xBF800001u
#define FLOAT_TWO_TO_MINUS_25_BITS 0x33000000u
#define FLOAT_TWO_TO_MINUS_24_BITS 0x33800000u
#define FLOAT_136_BITS 0x43080000u
#define FLOAT_136_AND_HALF_BITS 0x43088000u
#define FLOAT_LARGEST_FINITE_BITS 0x7F7FFFFFu
#define FLOAT_NOT_A_NUMBER_BITS 0x7FC00000u
#define FLOAT_MINUS_884_BITS 0xC45D0000u
#define RUNNING_HALF 0.5f
#define SMALL_PRODUCTS 4u /* class 120's case: 2^-25 four times */
#define TINY_PRODUCTS \
	8u /* class 80's case: 2^-26 eight times, a group of its own */
#define CLASS_80_GROUP 8u /* products 8 to 15: class 80's second group */
#define CODEBOOK_STATE_ZERO_BITS 0xC2E8u /* -3.453125 */
#define RANDOM_SEED 0x9E3779B97F4A7C15ull
#define XORSHIFT_FIRST 13
#define XORSHIFT_SECOND 7
#define XORSHIFT_THIRD 17
#define RANDOM_INSTRUCTIONS 4096u
#define HALF_BIAS 15u
#define HALF_FIELD_SHIFT 10u
#define HALF_FRACTION_MASK 0x03FFu
#define HALF_SIGN_BIT 0x8000u
#define EXPONENT_SPREAD 6u /* random inputs' exponents within 2^-6 to 2^6 */
#define EXPONENT_CHOICES 13u /* 2 EXPONENT_SPREAD + 1 */
#define ZERO_ONE_IN 8u /* a random input is zero once in this many */
#define RUNNING_SCALE 64.0f /* a random running sum: within about this */
#define RANDOM_SHIFT 32
#define RANDOM_UNIT 2.3283064365386963e-10 /* 2^-32 */
#define SIGNED_RANDOM_OFFSET 0.5
#define BOUND_TERMS \
	64.0 /* the bound: this many units of 2^(anchor - 23), far above any truncation */
#define FLOAT_ULP 1.1920928955078125e-07 /* 2^-23 */
#define PRODUCT_HEADROOM \
	4.0 /* a product of significands below 2 each is below 4 times 2^(ea + eb) */
#define SLICE_HEADROOM \
	2.0 /* a slice's running sum, a difference of two prefix sums, is at most twice the largest */
#define CHAIN_INPUTS 256u
#define CHAIN_OUTPUTS 32u
#define CHAIN_COLUMNS 8u
#define CHAIN_TILE 16u
#define CHAIN_SLICES 4u
#define CHAIN_MORE_SLICES 8u
#define ZERO_TRELLIS_SUM (-884.0f)
#define BITS_PER_WORD 32u
#define BITS_K2 2u
#define BITS_K3 3u

/* 1 to 16 as 16-bit floats */
static const uint16_t half_integers[TEST_MMA_PRODUCTS] = {
	0x3C00u, 0x4000u, 0x4200u, 0x4400u, 0x4500u, 0x4600u, 0x4700u, 0x4800u,
	0x4880u, 0x4900u, 0x4980u, 0x4A00u, 0x4A80u, 0x4B00u, 0x4B80u, 0x4C00u
};

static int failures;

static uint32_t bits_of(float number)
{
	uint32_t bits = 0;
	memcpy(&bits, &number, sizeof(bits));
	return bits;
}

static float float_of(uint32_t bits)
{
	float number = 0.0f;
	memcpy(&number, &bits, sizeof(number));
	return number;
}

static void expect_sum(const char *what, float running, const uint16_t *row,
		       const uint16_t *column, uint32_t class_80_bits,
		       uint32_t class_120_bits)
{
	const uint32_t class_80 = bits_of(test_mma_class_sum(
		ARITHMETIC_CLASS_SM_80, running, row, column));
	const uint32_t emulated = bits_of(test_mma_class_sum(
		ARITHMETIC_CLASS_SM_120, running, row, column));
	const uint32_t class_120 = bits_of(test_mma_class_sum(
		ARITHMETIC_CLASS_SM_120_NATIVE, running, row, column));
	if (class_80 != class_80_bits || emulated != class_80_bits ||
	    class_120 != class_120_bits) {
		failures++;
		(void)printf(
			"FAIL %s: class 80 %08x (emulated %08x), expected %08x; class 120 %08x, expected %08x\n",
			what, class_80, emulated, class_80_bits, class_120,
			class_120_bits);
	}
}

static void known_sums(void)
{
	uint16_t row[TEST_MMA_PRODUCTS], column[TEST_MMA_PRODUCTS];
	memset(row, 0, sizeof(row));
	memset(column, 0, sizeof(column));
	expect_sum("zeros", 0.0f, row, column, 0u, 0u);
	/* class 120 keeps four 2^-25 beside 1 (25 bits below the anchor); class 80 truncates them (24 bits) */
	row[CLASS_80_GROUP] = HALF_ONE;
	column[CLASS_80_GROUP] = HALF_ONE;
	for (uint32_t k = 1; k <= SMALL_PRODUCTS; k++) {
		row[CLASS_80_GROUP + k] = HALF_TWO_TO_MINUS_13;
		column[CLASS_80_GROUP + k] = HALF_TWO_TO_MINUS_12;
	}
	expect_sum("class 120's 25th bit", 0.0f, row, column, FLOAT_ONE_BITS,
		   FLOAT_ONE_AND_ULP_BITS);
	for (uint32_t k = 0; k <= SMALL_PRODUCTS; k++)
		column[CLASS_80_GROUP + k] =
			k == 0 ? HALF_MINUS_ONE : HALF_MINUS_TWO_TO_MINUS_12;
	expect_sum("toward zero, negative", 0.0f, row, column,
		   FLOAT_MINUS_ONE_BITS, FLOAT_MINUS_ONE_AND_ULP_BITS);
	/* class 80 sums eight 2^-26 alone in its first group (exactly 2^-23); class 120 truncates each beside 1 */
	memset(row, 0, sizeof(row));
	memset(column, 0, sizeof(column));
	for (uint32_t k = 0; k < TINY_PRODUCTS; k++) {
		row[k] = HALF_TWO_TO_MINUS_13;
		column[k] = HALF_TWO_TO_MINUS_13;
	}
	row[CLASS_80_GROUP] = HALF_ONE;
	column[CLASS_80_GROUP] = HALF_ONE;
	expect_sum("class 80's own group", 0.0f, row, column,
		   FLOAT_ONE_AND_ULP_BITS, FLOAT_ONE_BITS);
	/* 1 - 1 + 2^-25 in one group of class 80: the small product is below its window */
	memset(row, 0, sizeof(row));
	memset(column, 0, sizeof(column));
	row[0] = HALF_ONE;
	column[0] = HALF_ONE;
	row[1] = HALF_MINUS_ONE;
	column[1] = HALF_ONE;
	row[SMALL_PRODUCTS] = HALF_TWO_TO_MINUS_13;
	column[SMALL_PRODUCTS] = HALF_TWO_TO_MINUS_12;
	expect_sum("cancellation", 0.0f, row, column, 0u,
		   FLOAT_TWO_TO_MINUS_25_BITS);
	/* exact: 1 + 2 + ... + 16, and with a running sum of a half */
	for (uint32_t k = 0; k < TEST_MMA_PRODUCTS; k++) {
		row[k] = half_integers[k];
		column[k] = HALF_ONE;
	}
	expect_sum("exact", 0.0f, row, column, FLOAT_136_BITS, FLOAT_136_BITS);
	expect_sum("exact, running", RUNNING_HALF, row, column,
		   FLOAT_136_AND_HALF_BITS, FLOAT_136_AND_HALF_BITS);
	/* a subnormal input takes the smallest normal exponent for its anchor */
	memset(row, 0, sizeof(row));
	memset(column, 0, sizeof(column));
	row[0] = HALF_SMALLEST_SUBNORMAL;
	column[0] = HALF_ONE;
	expect_sum("subnormal", 0.0f, row, column, FLOAT_TWO_TO_MINUS_24_BITS,
		   FLOAT_TWO_TO_MINUS_24_BITS);
	/* past the largest finite float32: truncated toward zero, it stays the largest */
	row[0] = HALF_LARGEST_FINITE;
	column[0] = HALF_LARGEST_FINITE;
	expect_sum("largest finite", float_of(FLOAT_LARGEST_FINITE_BITS), row,
		   column, FLOAT_LARGEST_FINITE_BITS,
		   FLOAT_LARGEST_FINITE_BITS);
	/* any input or running sum that is not finite, even times zero */
	memset(row, 0, sizeof(row));
	memset(column, 0, sizeof(column));
	row[SMALL_PRODUCTS] = HALF_INFINITY;
	expect_sum("infinite weight", 0.0f, row, column,
		   FLOAT_NOT_A_NUMBER_BITS, FLOAT_NOT_A_NUMBER_BITS);
	row[SMALL_PRODUCTS] = HALF_ZERO;
	column[TEST_MMA_PRODUCTS - 1u] = HALF_NOT_A_NUMBER;
	expect_sum("not-a-number input", 0.0f, row, column,
		   FLOAT_NOT_A_NUMBER_BITS, FLOAT_NOT_A_NUMBER_BITS);
	column[TEST_MMA_PRODUCTS - 1u] = HALF_ZERO;
	expect_sum("not-a-number running", float_of(FLOAT_NOT_A_NUMBER_BITS),
		   row, column, FLOAT_NOT_A_NUMBER_BITS,
		   FLOAT_NOT_A_NUMBER_BITS);
}

static uint64_t random_state = RANDOM_SEED;

static uint32_t random_word(void)
{
	random_state ^= random_state << XORSHIFT_FIRST;
	random_state ^= random_state >> XORSHIFT_SECOND;
	random_state ^= random_state << XORSHIFT_THIRD;
	return (uint32_t)(random_state >> RANDOM_SHIFT);
}

/* A finite 16-bit float of either sign within 2^-EXPONENT_SPREAD to 2^EXPONENT_SPREAD, zero once in ZERO_ONE_IN. */
static uint16_t random_half(void)
{
	const uint32_t word = random_word();
	if (word % ZERO_ONE_IN == 0)
		return HALF_ZERO;
	const uint32_t field = HALF_BIAS - EXPONENT_SPREAD +
			       (word >> HALF_FIELD_SHIFT) % EXPONENT_CHOICES;
	return (uint16_t)((word & HALF_SIGN_BIT) | (field << HALF_FIELD_SHIFT) |
			  (random_word() & HALF_FRACTION_MASK));
}

/* |value| <= BOUND_TERMS units of 2^(e - 23) for the largest term e: what truncation can lose is far below. */
static bool within_bound(double value, double exact, double largest_term)
{
	return fabs(value - exact) <= BOUND_TERMS * largest_term * FLOAT_ULP;
}

static void random_sums(void)
{
	uint32_t differing = 0;
	for (uint32_t index = 0; index < RANDOM_INSTRUCTIONS; index++) {
		uint16_t row[TEST_MMA_PRODUCTS], column[TEST_MMA_PRODUCTS];
		const float running =
			(float)(((double)random_word() * RANDOM_UNIT -
				 SIGNED_RANDOM_OFFSET) *
				(double)RUNNING_SCALE);
		double exact = (double)running, largest = fabs(exact);
		for (uint32_t k = 0; k < TEST_MMA_PRODUCTS; k++) {
			row[k] = random_half();
			column[k] = random_half();
			const double product =
				test_attention_half_value(row[k]) *
				test_attention_half_value(column[k]);
			exact += product;
			largest =
				fmax(largest, PRODUCT_HEADROOM * fabs(product));
		}
		const float class_80 = test_mma_class_sum(
			ARITHMETIC_CLASS_SM_80, running, row, column);
		const float class_120 = test_mma_class_sum(
			ARITHMETIC_CLASS_SM_120_NATIVE, running, row, column);
		if (!within_bound((double)class_80, exact, largest) ||
		    !within_bound((double)class_120, exact, largest)) {
			failures++;
			(void)printf(
				"FAIL random instruction %u: class 80 %.9g, class 120 %.9g, exact %.17g\n",
				index, (double)class_80, (double)class_120,
				exact);
		}
		differing += bits_of(class_80) != bits_of(class_120) ? 1u : 0u;
	}
	(void)printf("random instructions: %u, the classes differ on %u\n",
		     RANDOM_INSTRUCTIONS, differing);
}

static void chain_zero_trellis(void)
{
	const uint32_t words =
		CHAIN_INPUTS * CHAIN_OUTPUTS * BITS_K3 / BITS_PER_WORD;
	uint32_t *const tiles = calloc(words, sizeof(*tiles));
	uint16_t *const weights =
		calloc((size_t)CHAIN_INPUTS * CHAIN_OUTPUTS, sizeof(*weights));
	uint16_t *const operands =
		calloc((size_t)CHAIN_COLUMNS * CHAIN_INPUTS, sizeof(*operands));
	float *const products = calloc((size_t)CHAIN_COLUMNS * CHAIN_OUTPUTS,
				       sizeof(*products));
	if (tiles == NULL || weights == NULL || operands == NULL ||
	    products == NULL) {
		failures++;
		(void)printf("FAIL zero trellis: no memory\n");
	} else {
		test_exl3_decode_matrix((const unsigned char *)tiles,
					CHAIN_INPUTS, CHAIN_OUTPUTS, BITS_K3,
					weights);
		bool decoded = true;
		for (uint32_t index = 0; index < CHAIN_INPUTS * CHAIN_OUTPUTS;
		     index++)
			decoded = decoded &&
				  weights[index] == CODEBOOK_STATE_ZERO_BITS;
		if (!decoded ||
		    test_exl3_codebook(0) != CODEBOOK_STATE_ZERO_BITS) {
			failures++;
			(void)printf(
				"FAIL zero trellis: a weight is not the codebook's state 0\n");
		}
		for (uint32_t index = 0; index < CHAIN_COLUMNS * CHAIN_INPUTS;
		     index++)
			operands[index] = HALF_ONE;
		const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					     ARITHMETIC_CLASS_SM_120_NATIVE };
		const uint32_t slice_counts[] = { CHAIN_SLICES,
						  CHAIN_MORE_SLICES };
		for (uint32_t form = 0;
		     form < sizeof(classes) / sizeof(classes[0]); form++)
			for (uint32_t cut = 0;
			     cut <
			     sizeof(slice_counts) / sizeof(slice_counts[0]);
			     cut++) {
				test_exl3_chain(classes[form], weights,
						CHAIN_INPUTS, CHAIN_OUTPUTS,
						slice_counts[cut], operands,
						CHAIN_COLUMNS, products);
				bool known = true;
				for (uint32_t index = 0;
				     index < CHAIN_COLUMNS * CHAIN_OUTPUTS;
				     index++)
					known = known &&
						bits_of(products[index]) ==
							FLOAT_MINUS_884_BITS;
				if (!known) {
					failures++;
					(void)printf(
						"FAIL zero trellis: class %u, %u slices: %.9g, expected %.9g\n",
						classes[form],
						slice_counts[cut],
						(double)products[0],
						(double)ZERO_TRELLIS_SUM);
				}
			}
	}
	free(tiles);
	free(weights);
	free(operands);
	free(products);
}

/* One output of a random chain on both classes against the exact sum; true where the classes differ. */
static bool chain_output_checked(uint32_t bits, uint32_t column,
				 uint32_t output, const uint16_t *weights,
				 const uint16_t *operands, float class_80,
				 float class_120)
{
	double exact = 0.0, largest = 0.0;
	for (uint32_t input = 0; input < CHAIN_INPUTS; input++) {
		const double product =
			test_attention_half_value(
				weights[input * CHAIN_OUTPUTS + output]) *
			test_attention_half_value(
				operands[column * CHAIN_INPUTS + input]);
		exact += product;
		largest = fmax(largest, fmax(PRODUCT_HEADROOM * fabs(product),
					     fabs(exact)));
	}
	/* every instruction and slice sum may lose its share: the bound grows with them */
	const uint32_t tiles =
		CHAIN_INPUTS / CHAIN_TILE; /* exact: whole tiles */
	const double steps = SLICE_HEADROOM * (double)(tiles + CHAIN_SLICES);
	if (!within_bound((double)class_80, exact, steps * largest) ||
	    !within_bound((double)class_120, exact, steps * largest)) {
		failures++;
		(void)printf("FAIL random chain K%u column %u output %u: "
			     "class 80 %.9g, class 120 %.9g, exact %.17g\n",
			     bits, column, output, (double)class_80,
			     (double)class_120, exact);
	}
	return bits_of(class_80) != bits_of(class_120);
}

static void chain_random(uint32_t bits, uint32_t *differing)
{
	const uint32_t words =
		CHAIN_INPUTS * CHAIN_OUTPUTS * bits / BITS_PER_WORD;
	uint32_t *const tiles = calloc(words, sizeof(*tiles));
	uint16_t *const weights =
		calloc((size_t)CHAIN_INPUTS * CHAIN_OUTPUTS, sizeof(*weights));
	uint16_t *const operands =
		calloc((size_t)CHAIN_COLUMNS * CHAIN_INPUTS, sizeof(*operands));
	float *const class_80 = calloc((size_t)CHAIN_COLUMNS * CHAIN_OUTPUTS,
				       sizeof(*class_80));
	float *const class_120 = calloc((size_t)CHAIN_COLUMNS * CHAIN_OUTPUTS,
					sizeof(*class_120));
	if (tiles == NULL || weights == NULL || operands == NULL ||
	    class_80 == NULL || class_120 == NULL) {
		failures++;
		(void)printf("FAIL random chain: no memory\n");
	} else {
		for (uint32_t word = 0; word < words; word++)
			tiles[word] = random_word();
		for (uint32_t index = 0; index < CHAIN_COLUMNS * CHAIN_INPUTS;
		     index++)
			operands[index] = random_half();
		test_exl3_decode_matrix((const unsigned char *)tiles,
					CHAIN_INPUTS, CHAIN_OUTPUTS, bits,
					weights);
		test_exl3_chain(ARITHMETIC_CLASS_SM_80, weights, CHAIN_INPUTS,
				CHAIN_OUTPUTS, CHAIN_SLICES, operands,
				CHAIN_COLUMNS, class_80);
		test_exl3_chain(ARITHMETIC_CLASS_SM_120_NATIVE, weights,
				CHAIN_INPUTS, CHAIN_OUTPUTS, CHAIN_SLICES,
				operands, CHAIN_COLUMNS, class_120);
		for (uint32_t at = 0; at < CHAIN_COLUMNS * CHAIN_OUTPUTS; at++)
			*differing += chain_output_checked(bits,
							   at / CHAIN_OUTPUTS,
							   at % CHAIN_OUTPUTS,
							   weights, operands,
							   class_80[at],
							   class_120[at]) ?
					      1u :
					      0u;
	}
	free(tiles);
	free(weights);
	free(operands);
	free(class_80);
	free(class_120);
}

int main(void)
{
	known_sums();
	random_sums();
	chain_zero_trellis();
	uint32_t differing = 0;
	chain_random(BITS_K2, &differing);
	chain_random(BITS_K3, &differing);
	if (differing == 0) {
		failures++;
		(void)printf(
			"FAIL random chains: the classes never differ, so the oracle cannot tell them apart\n");
	}
	(void)printf("random chains: %u outputs, the classes differ on %u\n",
		     CHAIN_COLUMNS * CHAIN_OUTPUTS * (BITS_K3 - BITS_K2 + 1u),
		     differing);
	(void)printf("EXL3 SM_120 reference %s: %d failures\n",
		     failures == 0 ? "PASS" : "FAIL", failures);
	return failures == 0 ? 0 : 1;
}
