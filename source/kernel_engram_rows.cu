/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_engram_rows.cu: the n-gram rows of a token into bfloat16 values, and the gate that adds the n-gram value
 * into the streams. Device code only, built once per architecture into the module "kernel_engram_rows".
 *
 * Contract: include/anemone/engram_rows_records.h. Here: an E4M3 byte becomes a 16-bit float by cuda_fp8.h's exact
 * conversion, then a float32; a thread of the gate keeps its values of the stream in registers from the sums to the
 * writes, and adds the warps' sums itself, in the one order, so that every thread holds the same gate.
 */
#include <cuda_fp8.h>

#include "anemone/engram_rows_records.h"
#include "kernel_linear_common.cuh"
#include "kernel_engram_gate.cuh"

#define E8M0_SMALLEST_BITS 0x00400000u          /* 2^-127, the number of code 0: a subnormal float32 */
#define E8M0_NOT_A_NUMBER 0xFFu
#define FLOAT32_NOT_A_NUMBER_BITS 0x7FC00000u
extern "C" __global__ void
engram_values_entry(struct engram_values_arguments arguments,
		    unsigned char *arena)
{
	LINEAR_DEVICE_CHECK(blockDim.x == ENGRAM_ROWS_VALUES);
	const size_t table_row = (size_t)blockIdx.y * gridDim.x + blockIdx.x;
	const unsigned char *const row = arena + arguments.rows_offset +
					 table_row * ENGRAM_ROWS_ROW_BYTES;
	const uint32_t code = row[ENGRAM_ROWS_VALUES +
				  threadIdx.x / ENGRAM_ROWS_VALUES_PER_SCALE];
	const float scale =
		code == E8M0_NOT_A_NUMBER ?
			__uint_as_float(FLOAT32_NOT_A_NUMBER_BITS) :
		code == 0 ? __uint_as_float(E8M0_SMALLEST_BITS) :
			    __uint_as_float(code << FLOAT32_EXPONENT_SHIFT);
	const float number = __half2float(
		__half(__nv_cvt_fp8_to_halfraw(row[threadIdx.x], __NV_E4M3)));
	uint16_t *const values = (uint16_t *)(arena + arguments.values_offset) +
				 table_row * ENGRAM_ROWS_VALUES;
	values[threadIdx.x] = (uint16_t)linear_bfloat16_bits(number * scale);
}

extern "C" __global__ void __launch_bounds__(ENGRAM_ROWS_GATE_THREADS, 1)
	engram_gate_entry(struct engram_gate_arguments arguments,
			  unsigned char *arena)
{
    engram_gate(arguments, arena);
}
