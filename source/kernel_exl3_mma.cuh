/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_exl3_mma.cuh: the tensor-core product of the EXL3 kernels, accumulator = A B + accumulator for one m16n8k16
 * instruction of 16-bit floats with float32 sums, in the PTX ISA's operand layouts: A the decoded weights of 16
 * outputs (four registers a lane), B up to eight columns, one job row each (lane 4 g + t: inputs 2 t, 2 t + 1 and
 * 2 t + 8, 2 t + 9 of column g). It is the one step of an EXL3 product whose bits depend on the card: every other
 * step is integer or IEEE arithmetic, the same on both classes. Three builds of one call:
 *   - sm_80 (class 80): the instruction, class 80's sums;
 *   - sm_120a with LINEAR_CLASS_FORMS (the *_native modules, ARITHMETIC_CLASS_SM_120_NATIVE, the PRO's default): the
 *     instruction, class 120's own sums, whose bits differ from class 80's (kernel_class_80.cuh);
 *   - sm_120a without it (the plain modules, ARITHMETIC_CLASS_SM_120, --pro_emulation=1): class 80's sums without its
 *     tensor cores, output by output (class_80_instruction_by_element), so a product's bits do not depend on its card.
 * column_count: the columns in use, from column 0; the emulation computes only those, the instruction all eight (the
 * caller stores only those in use).
 */
#ifndef KERNEL_EXL3_MMA_CUH
#define KERNEL_EXL3_MMA_CUH

#include <stdint.h>

#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(LINEAR_CLASS_FORMS)
#define EXL3_MMA_EMULATES_CLASS_80 1
#include "kernel_linear_common.cuh"
#else
#define EXL3_MMA_EMULATES_CLASS_80 0
#endif

/* A lane's results: output g (low) or g + 8 (high) by column 2 t (even) or 2 t + 1 (odd). */
enum exl3_mma_result {
	EXL3_MMA_LOW_EVEN = 0,
	EXL3_MMA_LOW_ODD = 1,
	EXL3_MMA_HIGH_EVEN = 2,
	EXL3_MMA_HIGH_ODD = 3,
	EXL3_MMA_RESULTS = 4
};
/* A lane's weight registers: output g (low) or g + 8 (high) at inputs 2 t, 2 t + 1 (low) or 2 t + 8, 2 t + 9 (high). */
enum exl3_mma_weight_register {
	EXL3_MMA_LOW_OUTPUT_LOW_INPUTS = 0,
	EXL3_MMA_HIGH_OUTPUT_LOW_INPUTS = 1,
	EXL3_MMA_LOW_OUTPUT_HIGH_INPUTS = 2,
	EXL3_MMA_HIGH_OUTPUT_HIGH_INPUTS = 3,
	EXL3_MMA_WEIGHT_REGISTERS = 4
};
#define EXL3_MMA_COLUMNS 8u

__device__ static __forceinline__ void
exl3_mma_f16(float accumulator[EXL3_MMA_RESULTS],
	     const uint32_t weights[EXL3_MMA_WEIGHT_REGISTERS],
	     uint32_t fed_low_inputs, uint32_t fed_high_inputs,
	     uint32_t column_count)
{
#if EXL3_MMA_EMULATES_CLASS_80
	class_80_instruction_by_element(accumulator, weights, fed_low_inputs,
					fed_high_inputs, false, column_count);
#else
	(void)column_count;
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
		     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
		     : "+f"(accumulator[EXL3_MMA_LOW_EVEN]),
		       "+f"(accumulator[EXL3_MMA_LOW_ODD]),
		       "+f"(accumulator[EXL3_MMA_HIGH_EVEN]),
		       "+f"(accumulator[EXL3_MMA_HIGH_ODD])
		     : "r"(weights[EXL3_MMA_LOW_OUTPUT_LOW_INPUTS]),
		       "r"(weights[EXL3_MMA_HIGH_OUTPUT_LOW_INPUTS]),
		       "r"(weights[EXL3_MMA_LOW_OUTPUT_HIGH_INPUTS]),
		       "r"(weights[EXL3_MMA_HIGH_OUTPUT_HIGH_INPUTS]),
		       "r"(fed_low_inputs), "r"(fed_high_inputs));
#endif
}

#endif
