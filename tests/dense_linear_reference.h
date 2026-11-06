/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_linear_reference.h: the host reference of the dense products (dense_linear_records.h), in double, and the
 * bound a card's output is held to (first_token_plan.md 3.2). The oracle of the card test of the family; its own
 * test is dense_linear_test. Test support: no part of the engine.
 *
 * Contract:
 *   - It is written from the definitions, with no call into the engine and no formula of the kernel, and calls no
 *     function of the mathematics library (the tests are linked without it). Weights are [output][input] of a type
 *     of enum dense_type_code, rows bfloat16 [row][input_size].
 *   - test_dense_reference: for each row and output, y = the sum over the inputs of weight times value in double
 *     (each product exact) and m = the sum of their magnitudes; with DENSE_ROW_SCALE both times
 *     (q / input_size + epsilon)^-1/2, q the sum of the row's squares. sums_out and magnitudes_out are
 *     [row_count][output_count], of outputs first_output on.
 *   - test_dense_output_holds: with bound = TEST_DENSE_TOLERANCE m + TEST_DENSE_SUBNORMAL_ALLOWANCE and s the output
 *     scale, a float32 output c holds where |c - y s| <= bound s; a bfloat16 one where it is finite and lies between
 *     the kernel's bfloat16 (rounded, scaled in float32, rounded again) of y - bound and of y + bound, rounding keeping
 *     the order.
 *   - test_dense_reciprocal_square_root of a positive normal double: within 2^-50 of 1 / sqrt.
 *   - Every function may be called from any thread.
 */
#ifndef TEST_DENSE_LINEAR_REFERENCE_H
#define TEST_DENSE_LINEAR_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#define TEST_DENSE_TOLERANCE 9.5367431640625e-07             /* 2^-20 of the magnitudes of the products (plan 3.2) */
#define TEST_DENSE_SUBNORMAL_ALLOWANCE 7.174648137343064e-43  /* 2^-140: what roundings below the normal numbers add */

double test_dense_magnitude(double value);
double test_dense_reciprocal_square_root(double value);
void test_dense_reference(uint32_t weight_type, const unsigned char *weights, uint32_t input_size, uint32_t flags,
                          float epsilon, const uint16_t *rows, uint32_t row_count, uint32_t first_output,
                          uint32_t output_count, double *sums_out, double *magnitudes_out);
bool test_dense_output_holds(uint32_t output_type, float output_scale, const unsigned char *outputs, uint64_t index,
                             double sum, double magnitude_sum);

#endif
