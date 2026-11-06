/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Lossless transport of the original activation quantizer's bytes. No requantization is performed on receive.
 * Four equal owners supply lane-ordered E4M3 codes followed by UE8M0 scales and zero padding in each wire row.
 * The receiver reconstructs exactly the original code, scale and carried BF16 operand planes.
 * input is [pair][owner][wire_pitch]; outputs use the original full-width lane/operand layouts.
 * Host admission is exactly local_width576, wire_pitch608, four owners and1..48 pairs. The separate producer
 * takes one owner's contiguous BF16 rows with the same geometry. No entry admits arbitrary widths or pitches. */
#ifndef QUANTIZED_WIRE_RECORDS_H
#define QUANTIZED_WIRE_RECORDS_H
#include "anemone/step_records.h"
#define QUANTIZED_WIRE_PRODUCER_ARGUMENT_BYTES 32u
#define KERNEL_QUANTIZE_WIRE "quantize_wire"
#define QUANTIZED_WIRE_ARGUMENT_BYTES 48u
#define QUANTIZED_WIRE_OWNERS 4u
#define QUANTIZED_WIRE_LOCAL_WIDTH 576u
#define QUANTIZED_WIRE_PITCH 608u
#define KERNEL_FILE_QUANTIZED_WIRE "kernel_quantized_wire"
#define KERNEL_QUANTIZED_WIRE "quantized_wire"
#define KERNEL_EXPERT_GATE_UP_WIRE "expert_gate_up_wire_1"
struct quantized_wire_arguments {
	uint64_t input_offset, codes_offset, exponents_offset, operands_offset;
	uint32_t pairs, local_width, wire_pitch, reserved_zero;
};
struct quantize_wire_arguments {
	uint64_t values_offset, wire_offset;
	uint32_t width, pairs, wire_pitch, reserved_zero;
};
STEP_RECORD_HAS(quantize_wire_arguments,
		QUANTIZED_WIRE_PRODUCER_ARGUMENT_BYTES);
STEP_RECORD_HAS(quantized_wire_arguments, QUANTIZED_WIRE_ARGUMENT_BYTES);
#endif
