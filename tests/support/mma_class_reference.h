/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * mma_class_reference.h: the host's model of one output of an m16n8k16 instruction of 16-bit floats with float32
 * sums, as each class of card adds it (the rule tools/measure/cross_class measured, which kernel_class_80.cuh states),
 * written without the device code:
 *   - class 80: two groups, products 0 to 7 and then 8 to 15, each with the running sum. Every term is aligned to the
 *     group's anchor: the largest exponent sum ea + eb of its nonzero products (a subnormal input's exponent the
 *     smallest normal one's, its significand unnormalized) and the running sum's own exponent. Each term is truncated
 *     toward zero to a multiple of 2^(anchor - 24), the terms are added exactly, and the sum is truncated toward zero
 *     to float32 (a magnitude past the largest finite one becomes it);
 *   - class 120: one group of all 16 products and the running sum, to multiples of 2^(anchor - 25).
 * An input or a running sum that is not finite gives the not-a-number 0x7FC00000, as the emulation does (output by
 * output: class_80_instruction_by_element); the EXL3 card test holds both classes' cards to it.
 */
#ifndef TEST_MMA_CLASS_REFERENCE_H
#define TEST_MMA_CLASS_REFERENCE_H

#include <stdint.h>

#define TEST_MMA_PRODUCTS 16u

/* sum_class: ARITHMETIC_CLASS_SM_80 or ARITHMETIC_CLASS_SM_120 (its emulation of class 80) for class 80's sums,
 * ARITHMETIC_CLASS_SM_120_NATIVE for class 120's own; row and column the 16 inputs of the output, bits of 16-bit
 * floats. */
float test_mma_class_sum(uint32_t sum_class, float running,
			 const uint16_t row[TEST_MMA_PRODUCTS],
			 const uint16_t column[TEST_MMA_PRODUCTS]);

#endif
