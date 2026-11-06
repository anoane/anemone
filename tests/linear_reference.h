/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_reference.h: the host reference of the linear on the cards: what the kernels of linear_records.h must give,
 * from what the model's reference code computes (model.py linear(), Expert.forward and MoE.forward; kernel.py
 * act_quant_kernel with scales rounded to a power of two). The oracle of the card tests of the linear. Test support:
 * no part of the engine.
 *
 * Contract:
 *   - It is written from the definitions, with no call into the engine and no formula of a kernel: a quantized value
 *     is the E4M3 number a search over all of them finds nearest; a weight and its scale are read through
 *     weight_format_reference.h; a sum is a sum of doubles. No function of the mathematics library is called (the
 *     tests are linked without it): a power of two is a product of the powers an E8M0 byte stands for, and the
 *     exponential is its series.
 *   - bfloat16 is the upper half of a float32: a sign, 8 exponent bits of bias 127 and 7 fraction bits. Its last
 *     place is 2^-7 of its leading bit, and 2^-133 below 2^-126. test_linear_bfloat16_nearest rounds a double once:
 *     to nearest, a tie to the even fraction, to infinity from half a last place above the largest number, a
 *     not-a-number to 0x7FC0; the sign of a zero is kept. The ordinal of a bfloat16 is a whole number that is one
 *     apart for neighbouring values and 0 for both zeros: a difference of ordinals is a difference in last places.
 *   - Quantization (test_linear_quantize), for each block of 32 consecutive values: m is the largest magnitude, at
 *     least float32(1e-4), times float32(1 / 448), a float32 product; the scale is the smallest power of two that is
 *     not below m, and its E8M0 code the exponent; each value is divided by the scale (exactly, in double), its
 *     magnitude limited to 448, and replaced by the nearest of the 127 E4M3 magnitudes 0x00 to 0x7E, of two equally
 *     near the one of the even code; the code keeps the sign bit of the value, also where it is a zero. The codes
 *     come in the ORDER OF THE VALUES. test_linear_codes_lay_out puts them in the lane order of linear_records.h by
 *     walking the stored bytes; test_linear_code_place goes the other way, from a value to its byte, and shares no
 *     formula with it. test_linear_operands_lay_out makes the operands of linear_records.h of codes and exponents in
 *     the order of the values: the bfloat16 nearest each code's value times its scale times 2^100 (which it is
 *     exactly), a zero keeping the code's sign; an infinity of the code's sign for a block whose exponent is above
 *     TEST_LINEAR_CARRIED_EXPONENT_LARGEST; in operand order, by walking the halves.
 *   - A product (test_linear_products) is, for one output and one fed vector, the sum over the blocks of 32 inputs in
 *     ascending order of (the sum of weight code value times fed value) times the weights' scale times the vector's
 *     scale: sum. magnitude_sum is the same sum over the magnitudes of the products. A fed value of a quantized
 *     vector is what its code stands for, and the sum of a block is then exact (41 bits hold it). A fed value of a
 *     row that is not quantized is the 16-bit float nearest its bfloat16 (test_linear_half_value: a tie to even,
 *     multiples of 2^-24 below 2^-14, an infinity from 65520), as linear_records.h says the kernel feeds it; such a
 *     row has no scale. With segment_count above 1 the outputs are cut into as many equal runs, and run i reads
 *     the fed values from i * input_size on.
 *   - Tolerance (test_linear_output_is_within_tolerance). A card adds the same products in float32 and in the
 *     order of linear_records.h, then rounds once to bfloat16. With y the sum above, m its magnitude_sum, B the
 *     blocks of 32 inputs and S the slices, the card's float32 sum lies within
 *         bound = (T + B + S) * 2^-24 * m + B * 2^-149
 *     of y: each of the B fused multiply-adds and of the S additions rounds once, by at most 2^-24 of a partial sum
 *     that m bounds, or by 2^-149 where the result is subnormal; T = TEST_LINEAR_TENSOR_CORE_ALLOWANCE stands for
 *     the rounding inside the tensor-core instructions, which is defined nowhere: 64 is what cutting each of the 32
 *     products of every block to the float32 grid of the largest would cost, twice over. The rounding to bfloat16
 *     moves the result by at most half a last place. So an output c passes when |c - y| <= last_place(c) / 2 +
 *     bound. In last places of bfloat16 that is one half plus (T + B + S) * 2^-16 * m / |y| at most: 0.004 * m / |y|
 *     for the 160 blocks and 16 slices of a hidden width of 5120. An output whose products do not cancel (m / |y|
 *     under 100) is therefore the double result rounded once, or its neighbour; one whose products cancel to a
 *     thousandth of their magnitudes may lie four further. An infinite c passes where y plus the bound rounds to
 *     that infinity; where y is no finite number (a row that is not quantized held 65536 or more) c passes when it
 *     is none either. test_linear_compare counts, over many outputs: those whose y is a finite number, those of
 *     them bit-equal to test_linear_bfloat16_nearest(y), those of all outside the tolerance, and the largest
 *     difference in last places.
 *   - The hidden step (test_linear_hidden_value), in double: g = gate, or limit where gate is above it; u = up
 *     limited to -limit .. limit; g / (1 + exp(-g)) * u * route weight. Where -g is above 128 ln 2 the float32
 *     exponential of a card is infinite and the quotient a zero: so it is here. For finite gate and up; a
 *     not-a-number stays one. test_linear_hidden_range is what the fused kernel may store for a gate and an up it
 *     does not show: every bfloat16 within the tolerance of either product is tried at its ends and its middle, and
 *     the hidden values of all of them, each rounded to bfloat16, span the range, widened by one last place for the
 *     float32 arithmetic and the exponential of the card. central is the hidden value of the two products rounded
 *     once.
 *   - test_linear_row_sum is the float32 sum of a row's pair outputs from +0 in ascending slot order, in float32
 *     additions: a card that adds in that order has these bits.
 *   - Refused with STATUS_INVALID_ARGUMENT: a NULL argument; a width that is 0 or no multiple of 32; a value
 *     to quantize that is an infinity or a not-a-number; a code or a scale code that is no number (the E4M3 bytes
 *     0x7F and 0xFF, the E8M0 byte 0xFF); a matrix weight_format_reference.h refuses, or whose block of 32 inputs
 *     holds two scales; outputs beyond the matrix; a vector_count of 0; a segment_count that is 0 or does not divide
 *     the outputs. STATUS_OUT_OF_MEMORY where the working memory of a product is not obtained. On a refusal
 *     the outputs are unspecified.
 *   - Every function may be called from any thread.
 */
#ifndef TEST_LINEAR_REFERENCE_H
#define TEST_LINEAR_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/status.h"
#include "anemone/weight_format_seam.h"

#define TEST_LINEAR_VALUES_PER_BLOCK 32u            /* values of a vector, and inputs of a matrix, that share a scale */
#define TEST_LINEAR_TENSOR_CORE_ALLOWANCE 64.0      /* T of the tolerance */
#define TEST_LINEAR_BFLOAT16_NOT_A_NUMBER 0x7FC0u   /* what a not-a-number is rounded to */
#define TEST_LINEAR_NO_OUTPUT UINT64_MAX            /* in place of the index of an output */
#define TEST_LINEAR_CARRIED_EXPONENT_LARGEST 146u   /* the largest exponent of a block whose operands are finite */

double test_linear_bfloat16_value(uint16_t bits);   /* an infinity or a not-a-number where the bits are one */
bool test_linear_bfloat16_is_finite(uint16_t bits);
uint16_t test_linear_bfloat16_nearest(double value);
int32_t test_linear_bfloat16_ordinal(uint16_t bits);
uint16_t test_linear_bfloat16_at(int32_t ordinal);  /* of an ordinal; +0 for 0; an infinity beyond the infinities */
double test_linear_bfloat16_last_place(uint16_t bits); /* the distance from a finite bfloat16 to the next away from 0 */
double test_linear_half_value(uint16_t bfloat16_bits); /* the 16-bit float nearest a bfloat16, as a double */

/* values [width] as bfloat16 bits; codes_out [width] in the order of the values; exponents_out [width / 32]. */
MUST_CHECK result_status test_linear_quantize(const uint16_t *values,
					      uint32_t width,
					      uint8_t *codes_out,
					      uint8_t *exponents_out);

/* codes [width] in the order of the values; stored_out [width] in lane order. */
MUST_CHECK result_status test_linear_codes_lay_out(const uint8_t *codes,
						   uint32_t width,
						   uint8_t *stored_out);

/* The byte of a vector in lane order that holds the code of its value value_index. */
uint32_t test_linear_code_place(uint32_t value_index);

/* codes [width] in the order of the values, exponents [width / 32]; operands_out [width] in operand order. */
MUST_CHECK result_status test_linear_operands_lay_out(const uint8_t *codes,
						      const uint8_t *exponents,
						      uint32_t width,
						      uint16_t *operands_out);

/* What a product is fed: fed_values_out [width], from codes in the order of their values, or from bfloat16 bits. */
MUST_CHECK result_status test_linear_fed_values_of_codes(
	const uint8_t *codes, uint32_t width, double *fed_values_out);
MUST_CHECK result_status test_linear_fed_values_of_bfloat16(
	const uint16_t *values, uint32_t width, double *fed_values_out);

struct test_linear_product {
    double sum;                                     /* y: the blocks of 32 inputs added in ascending order */
    double magnitude_sum;                           /* m: the same sum over the magnitudes of the products */
};

/* The outputs first_output .. first_output + output_count - 1 of a stored matrix for vector_count fed vectors:
 * fed_values [vector_count][segment_count * input_size]; fed_exponents [vector_count][segment_count * input_size /
 * 32], or NULL for rows that are not quantized; products_out [vector_count][output_count]. */
MUST_CHECK result_status test_linear_products(
	enum weight_format_code format, const struct matrix_shape_record *shape,
	const uint8_t *stored, uint64_t stored_byte_count,
	uint32_t first_output, uint32_t output_count, const double *fed_values,
	const uint8_t *fed_exponents, uint32_t vector_count,
	uint32_t segment_count, struct test_linear_product *products_out);

bool test_linear_output_is_within_tolerance(uint16_t output_bits, const struct test_linear_product *product,
                                            uint32_t input_size, uint32_t slice_count);

/* What test_linear_compare has counted so far; a test starts it as all zeros. */
struct test_linear_comparison {
    uint64_t output_count;                          /* outputs whose sum is a finite number */
    uint64_t equal_count;                           /* of them: bit-equal to the double result rounded once */
    uint64_t outside_count;                         /* outside the tolerance */
    uint32_t largest_distance;                      /* in last places, among outputs and results that are finite */
    uint32_t reserved_zero;
};

/* Adds output_count outputs to a comparison: outputs [output_count] as bfloat16 bits against products
 * [output_count]. The index of the first output outside the tolerance, or TEST_LINEAR_NO_OUTPUT. */
uint64_t test_linear_compare(const uint16_t *outputs, const struct test_linear_product *products,
                             uint64_t output_count, uint32_t input_size, uint32_t slice_count,
                             struct test_linear_comparison *comparison);

double test_linear_hidden_value(double gate, double up, double limit, double route_weight);

struct test_linear_hidden_range {
    int32_t lowest_ordinal;                         /* the ordinals a card's hidden value may have: lowest .. highest */
    int32_t highest_ordinal;
    uint16_t central_bits;                          /* the hidden value of the two products rounded once */
    uint16_t reserved_zero;
};

/* gate and up: the two products of one output for one vector, of input_size inputs in slice_count slices each. */
void test_linear_hidden_range(const struct test_linear_product *gate, const struct test_linear_product *up,
                              uint32_t input_size, uint32_t slice_count, double limit, double route_weight,
                              struct test_linear_hidden_range *range_out);

/* pair_outputs [pair_count][width] as bfloat16 bits, the pairs of ONE row in ascending slot order: the sum of
 * their values at column. */
float test_linear_row_sum(const uint16_t *pair_outputs, uint32_t pair_count, uint32_t width, uint32_t column);

#endif
