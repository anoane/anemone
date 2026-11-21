/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "mma_class_reference.h"

#include <stdbool.h>
#include <string.h>

#include "engine/backend/backend_seam.h"

#define HALF_FRACTION_BITS 10
#define HALF_FRACTION_MASK 0x03FFu
#define HALF_HIDDEN_BIT 0x0400u
#define HALF_FIELD_MASK 0x1Fu
#define HALF_SIGN_BIT 0x8000u
#define HALF_BIAS 15
#define PRODUCT_FRACTION_BITS 20 /* two inputs' 10 fraction bits */
#define FLOAT_FRACTION_BITS 23
#define FLOAT_FRACTION_MASK 0x007FFFFFu
#define FLOAT_HIDDEN_BIT 0x00800000u
#define FLOAT_FIELD_MASK 0xFFu
#define FLOAT_SIGN_BIT 0x80000000u
#define FLOAT_BIAS 127
#define FLOAT_SMALLEST_EXPONENT (-126)
#define FLOAT_LARGEST_FINITE_BITS 0x7F7FFFFFu
#define FLOAT_NOT_A_NUMBER_BITS 0x7FC00000u
#define CLASS_80_GROUP_PRODUCTS 8u
#define CLASS_80_KEPT_BITS 24
#define CLASS_120_KEPT_BITS 25
#define LAST_SHIFT 63 /* a magnitude shifted right by more is zero */
#define NO_ANCHOR INT32_MIN

/* A 16-bit float as significand times 2^(exponent - HALF_FRACTION_BITS) (a subnormal's exponent the smallest normal
 * one's), or a float32 as significand times 2^(exponent - FLOAT_FRACTION_BITS). */
struct reference_value {
	int64_t significand;
	int32_t exponent;
	bool is_finite;
};

static struct reference_value half_decode(uint16_t bits)
{
	const uint32_t field = ((uint32_t)bits >> HALF_FRACTION_BITS) &
			       HALF_FIELD_MASK;
	const int64_t magnitude =
		(int64_t)((bits & HALF_FRACTION_MASK) |
			  (field != 0 ? HALF_HIDDEN_BIT : 0u));
	const struct reference_value value = {
		(bits & HALF_SIGN_BIT) != 0 ? -magnitude : magnitude,
		(int32_t)(field != 0 ? field : 1u) - HALF_BIAS,
		field != HALF_FIELD_MASK
	};
	return value;
}

static struct reference_value float_decode(float number)
{
	uint32_t bits = 0;
	memcpy(&bits, &number, sizeof(bits));
	const uint32_t field = (bits >> FLOAT_FRACTION_BITS) & FLOAT_FIELD_MASK;
	const int64_t magnitude =
		(int64_t)((bits & FLOAT_FRACTION_MASK) |
			  (field != 0 ? FLOAT_HIDDEN_BIT : 0u));
	const struct reference_value value = {
		(bits & FLOAT_SIGN_BIT) != 0 ? -magnitude : magnitude,
		(int32_t)(field != 0 ? field : 1u) - FLOAT_BIAS,
		field != FLOAT_FIELD_MASK
	};
	return value;
}

static float float_of_bits(uint32_t bits)
{
	float number = 0.0f;
	memcpy(&number, &bits, sizeof(number));
	return number;
}

/* value times 2^shift, truncated toward zero; |value| is below 2^40 here, so no shift overflows. */
static int64_t scaled_toward_zero(int64_t value, int32_t shift)
{
	const uint64_t magnitude = value < 0 ? (uint64_t)-value :
					       (uint64_t)value;
	const uint64_t scaled = shift >= 0	    ? magnitude << shift :
				-shift > LAST_SHIFT ? 0u :
						      magnitude >> -shift;
	return value < 0 ? -(int64_t)scaled : (int64_t)scaled;
}

/* units times 2^unit, truncated toward zero to float32. */
static float float_toward_zero(int64_t units, int32_t unit)
{
	if (units == 0)
		return 0.0f;
	const uint64_t magnitude = units < 0 ? (uint64_t)-units :
					       (uint64_t)units;
	const uint32_t sign = units < 0 ? FLOAT_SIGN_BIT : 0u;
	int32_t lead = 0;
	while (lead < LAST_SHIFT && (magnitude >> (lead + 1)) != 0)
		lead++;
	const int32_t exponent = unit + lead;
	if (exponent > FLOAT_BIAS)
		return float_of_bits(sign | FLOAT_LARGEST_FINITE_BITS);
	uint64_t significand =
		lead > FLOAT_FRACTION_BITS ?
			magnitude >> (lead - FLOAT_FRACTION_BITS) :
			magnitude << (FLOAT_FRACTION_BITS - lead);
	if (exponent < FLOAT_SMALLEST_EXPONENT) {
		const int32_t extra = FLOAT_SMALLEST_EXPONENT - exponent;
		significand = extra > LAST_SHIFT ? 0u : significand >> extra;
		return float_of_bits(sign | (uint32_t)significand);
	}
	return float_of_bits(
		sign |
		((uint32_t)(exponent + FLOAT_BIAS) << FLOAT_FRACTION_BITS) |
		((uint32_t)significand & FLOAT_FRACTION_MASK));
}

/* One group: running and products first to first + count - 1, kept_bits below the anchor. */
static float group_sum(float running, const struct reference_value *rows,
		       const struct reference_value *columns, uint32_t first,
		       uint32_t count, int32_t kept_bits)
{
	const struct reference_value sum = float_decode(running);
	int32_t anchor = sum.significand != 0 ? sum.exponent : NO_ANCHOR;
	for (uint32_t k = first; k < first + count; k++) {
		const int32_t exponent = rows[k].exponent + columns[k].exponent;
		if (rows[k].significand != 0 && columns[k].significand != 0 &&
		    exponent > anchor)
			anchor = exponent;
	}
	if (anchor == NO_ANCHOR)
		return 0.0f;
	const int32_t unit = anchor - kept_bits;
	int64_t units = scaled_toward_zero(
		sum.significand, sum.exponent - FLOAT_FRACTION_BITS - unit);
	for (uint32_t k = first; k < first + count; k++)
		units += scaled_toward_zero(
			rows[k].significand * columns[k].significand,
			rows[k].exponent + columns[k].exponent -
				PRODUCT_FRACTION_BITS - unit);
	return float_toward_zero(units, unit);
}

float test_mma_class_sum(uint32_t sum_class, float running,
			 const uint16_t row[TEST_MMA_PRODUCTS],
			 const uint16_t column[TEST_MMA_PRODUCTS])
{
	struct reference_value rows[TEST_MMA_PRODUCTS],
		columns[TEST_MMA_PRODUCTS];
	bool is_finite = float_decode(running).is_finite;
	for (uint32_t k = 0; k < TEST_MMA_PRODUCTS; k++) {
		rows[k] = half_decode(row[k]);
		columns[k] = half_decode(column[k]);
		is_finite = is_finite && rows[k].is_finite &&
			    columns[k].is_finite;
	}
	if (!is_finite)
		return float_of_bits(FLOAT_NOT_A_NUMBER_BITS);
	if (sum_class == ARITHMETIC_CLASS_SM_120_NATIVE)
		return group_sum(running, rows, columns, 0, TEST_MMA_PRODUCTS,
				 CLASS_120_KEPT_BITS);
	const float first = group_sum(running, rows, columns, 0,
				      CLASS_80_GROUP_PRODUCTS,
				      CLASS_80_KEPT_BITS);
	return group_sum(first, rows, columns, CLASS_80_GROUP_PRODUCTS,
			 TEST_MMA_PRODUCTS - CLASS_80_GROUP_PRODUCTS,
			 CLASS_80_KEPT_BITS);
}
