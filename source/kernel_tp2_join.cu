/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* TP-only local vector completion and strided BF16 placement; no synchronization or payload arrivals. */
#include "anemone/tp2_join_records.h"
#include "kernel_linear_common.cuh"

/* RING_PACK / RING_UNPACK: an element's place in the ring's owner-major order of row_count rows of width (width a
 * multiple of four): owner o of quarter o of its slab's elements, slab after slab, owner after owner. A slab's place
 * in 32 bits (a slab's elements fit them); a slab of a multiple of four rows (every slab of a chunk: its rows are a
 * multiple of 8) gives each owner whole rows, without a division by a run-time value. The same places as the 64-bit
 * divisions it replaces (rb01: 39 us a 512-row join on the 640-column card). */
__device__ static __forceinline__ uint64_t ring_place(uint64_t row, uint32_t column, uint32_t rows, uint32_t width)
{
	const uint32_t slab = (uint32_t)(row / TP2_JOIN_RING_SLAB_ROWS);
	const uint32_t row_in_slab = (uint32_t)(row % TP2_JOIN_RING_SLAB_ROWS);
	const uint32_t slab_rows =
		rows - slab * TP2_JOIN_RING_SLAB_ROWS <
				TP2_JOIN_RING_SLAB_ROWS ?
			rows - slab * TP2_JOIN_RING_SLAB_ROWS :
			TP2_JOIN_RING_SLAB_ROWS;
	uint32_t owner, in_quarter;
	if (slab_rows % TP2_JOIN_RING_OWNERS == 0) {
		const uint32_t owner_rows = slab_rows / TP2_JOIN_RING_OWNERS;
		owner = row_in_slab / owner_rows;
		in_quarter = (row_in_slab - owner * owner_rows) * width + column;
	} else {
		const uint32_t quarter =
			slab_rows * width / TP2_JOIN_RING_OWNERS;
		const uint32_t in_slab = row_in_slab * width + column;
		owner = in_slab / quarter;
		in_quarter = in_slab - owner * quarter;
	}
	return (uint64_t)owner *
		       ((uint64_t)rows * width / TP2_JOIN_RING_OWNERS) +
	       (uint64_t)slab * (TP2_JOIN_RING_SLAB_ROWS * width /
				 TP2_JOIN_RING_OWNERS) +
	       in_quarter;
}

extern "C" __global__ void tp2_join(struct tp2_join_arguments arguments,
				    unsigned char *arena)
{
	const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t row = blockIdx.y;
	if (column >= arguments.width || row >= arguments.row_count)
		return;
	const uint64_t source_element =
		row * arguments.source_row_stride + column;
	const uint64_t output_element =
		row * arguments.output_row_stride + column;
	const uint64_t parity =
		arguments.source_slot_stride != 0 ||
				arguments.output_slot_stride != 0 ?
			((const struct step_block *)(arena +
						     arguments.step_offset))
					->sequence &
				1u :
			0;
	const unsigned char *const source =
		arena + arguments.source_offset +
		parity * arguments.source_slot_stride;
	unsigned char *const output = arena + arguments.output_offset +
				      parity * arguments.output_slot_stride;
	if (arguments.operation == TP2_JOIN_RING_PACK) {
		float value = linear_bfloat16_value(((const uint16_t *)source)[source_element]);
		if (arguments.has_routed_sums != 0)
			value = __fadd_rn(((const float *)(arena + arguments.routed_sums_offset))
						  [row * arguments.routed_row_stride + column],
					  value);
		((uint16_t *)output)[ring_place(row, column, arguments.row_count, arguments.width)] =
			(uint16_t)linear_bfloat16_bits(value);
	} else if (arguments.operation == TP2_JOIN_RING_UNPACK) {
		((uint16_t *)output)[output_element] = ((const uint16_t *)source)[ring_place(
			row, column, arguments.row_count, arguments.width)];
	} else if (arguments.operation == TP2_JOIN_BFLOAT16_COPY) {
		((uint16_t *)output)[output_element] =
			((const uint16_t *)source)[source_element];
	} else if (arguments.operation == TP2_JOIN_FLOAT32_COPY) {
		((uint32_t *)output)[output_element] =
			((const uint32_t *)source)[source_element];
	} else if (arguments.operation == TP2_JOIN_FLOAT32_TO_BFLOAT16) {
		const float value = ((const float *)source)[source_element];
		((uint16_t *)output)[output_element] =
			(uint16_t)linear_bfloat16_bits(value);
	} else if (arguments.operation == TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16) {
		float value = linear_bfloat16_value(
			((const uint16_t *)source)[source_element]);
		if (arguments.has_routed_sums != 0) {
			const float routed = ((
				const float *)(arena +
					       arguments.routed_sums_offset))
				[row * arguments.routed_row_stride + column];
			value = __fadd_rn(routed, value);
		}
		((uint16_t *)output)[output_element] =
			(uint16_t)linear_bfloat16_bits(value);
	} else {
		const uint16_t bits =
			((const uint16_t *)source)[source_element];
		float value = linear_bfloat16_value(bits);
		if (arguments.has_routed_sums != 0) {
			const float routed = ((
				const float *)(arena +
					       arguments.routed_sums_offset))
				[row * arguments.routed_row_stride + column];
			value = __fadd_rn(routed, value);
		}
		((float *)output)[output_element] = value;
	}
}
