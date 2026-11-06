/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* TP-only local conversions and strided canonical placement. Contract: anemone/tp2_join.h. */
#include "anemone/tp2_join.h"

#include "anemone/hooks.h"
#include "prepare_common.h"

#define JOIN_MOST_RANGES 6u
#define JOIN_MAXIMUM_WIDTH (MODEL_HEADS_VALUE * MODEL_HEAD_DIMS)

_Static_assert(TP2_JOIN_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES,
	       "the TP join record fits a launch");

result_status tp2_join_kernel_prepare(const struct backend_operations *backend,
				      struct device_card *card,
				      uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL) {
		LOG_ERROR(
			"the local TP completion: missing backend, card or kernel output");
		return STATUS_INVALID_ARGUMENT;
	}
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_TP2_JOIN, KERNEL_TP2_JOIN,
		TP2_JOIN_ARGUMENT_BYTES, &kernel);
	if (status == STATUS_OK)
		*kernel_out = kernel;
	return status;
}

static uint64_t strided_bytes(uint32_t rows, uint32_t width, uint32_t stride,
			      uint32_t element_bytes)
{
	return ((uint64_t)(rows - 1u) * stride + width) * element_bytes;
}

static bool capacity_is_valid(const struct linear_range *range,
			      uint64_t touched, uint64_t slot_stride,
			      uint64_t arena)
{
	return range->byte_offset % LINEAR_OFFSET_ALIGNMENT_BYTES == 0 &&
	       slot_stride % LINEAR_OFFSET_ALIGNMENT_BYTES == 0 &&
	       range->byte_offset <= arena &&
	       range->byte_count <= arena - range->byte_offset &&
	       touched <= range->byte_count &&
	       (slot_stride == 0 ||
		(slot_stride >= touched &&
		 slot_stride <= range->byte_count - touched));
}

result_status tp2_join_prepare(const struct tp2_join_description *description,
			       uint64_t arena_byte_count, uint32_t kernel,
			       struct kernel_launch *launch_out)
{
	static const char what[] = "the local TP completion";
	if (!prepare_arguments_are_given(what, description, &kernel,
					 launch_out))
		return STATUS_INVALID_ARGUMENT;
	const uint32_t width = description->width,
		       rows = description->row_count;
	const bool has_routed = description->routed_sums.byte_count != 0;
	const bool has_step = description->source_slot_stride != 0 ||
			      description->output_slot_stride != 0;
	if (width == 0 || width > JOIN_MAXIMUM_WIDTH || rows == 0 ||
	    rows > LINEAR_WIDE_MOST_ROWS ||
	    description->operation >= TP2_JOIN_OPERATION_COUNT ||
	    description->source_row_stride < width ||
	    description->output_row_stride < width ||
	    (has_routed &&
	     ((description->operation != TP2_JOIN_BFLOAT16_TO_FLOAT32 &&
	       description->operation != TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16 &&
	       description->operation != TP2_JOIN_RING_PACK) ||
	      description->routed_row_stride < width)) ||
	    /* the ring's owner-major rows are dense: width a multiple of the owners */
	    ((description->operation == TP2_JOIN_RING_PACK ||
	      description->operation == TP2_JOIN_RING_UNPACK) &&
	     (width % TP2_JOIN_RING_OWNERS != 0 ||
	      (description->operation == TP2_JOIN_RING_PACK ?
		       description->output_row_stride :
		       description->source_row_stride) != width)) ||
	    (!has_routed && (description->routed_sums.byte_offset != 0 ||
			     description->routed_row_stride != 0)) ||
	    (!has_step && (description->step.byte_offset != 0 ||
			   description->step.byte_count != 0))) {
		LOG_ERROR(
			"%s: invalid operation, row/slot dimensions or optional input",
			what);
		return STATUS_INVALID_ARGUMENT;
	}
	if (!prepare_reserved_is_zero(what, description->reserved_zero) ||
	    !prepare_kernel_is_prepared(what, kernel, KERNEL_TP2_JOIN))
		return STATUS_INVALID_ARGUMENT;
	const uint32_t input_bytes =
		description->operation == TP2_JOIN_FLOAT32_TO_BFLOAT16 ||
				description->operation ==
					TP2_JOIN_FLOAT32_COPY ?
			LINEAR_SUM_BYTES :
			LINEAR_BFLOAT16_BYTES;
	const uint32_t output_bytes =
		description->operation == TP2_JOIN_BFLOAT16_TO_FLOAT32 ||
				description->operation ==
					TP2_JOIN_FLOAT32_COPY ?
			LINEAR_SUM_BYTES :
			LINEAR_BFLOAT16_BYTES;
	const uint64_t source_bytes = strided_bytes(
		rows, width, description->source_row_stride, input_bytes);
	const uint64_t destination_bytes = strided_bytes(
		rows, width, description->output_row_stride, output_bytes);
	const uint64_t routed_bytes = strided_bytes(
		rows, width, description->routed_row_stride, LINEAR_SUM_BYTES);
	if (!capacity_is_valid(&description->source, source_bytes,
			       description->source_slot_stride,
			       arena_byte_count) ||
	    !capacity_is_valid(&description->output, destination_bytes,
			       description->output_slot_stride,
			       arena_byte_count) ||
	    (has_routed &&
	     !capacity_is_valid(&description->routed_sums, routed_bytes, 0,
				arena_byte_count)) ||
	    (has_step &&
	     !capacity_is_valid(&description->step, STEP_BLOCK_BYTES_VALUE, 0,
				arena_byte_count))) {
		LOG_ERROR(
			"%s: strided data, second parity slot or step leaves its capacity",
			what);
		return STATUS_INVALID_ARGUMENT;
	}
	struct linear_range used[JOIN_MOST_RANGES] = {
		{ description->source.byte_offset, source_bytes },
		{ description->output.byte_offset, destination_bytes }
	};
	uint32_t count = 0;
	struct checked_range ranges[JOIN_MOST_RANGES] = { 0 };
	ranges[count++] = (struct checked_range){ "the source", &used[0],
						  source_bytes, false };
	ranges[count++] = (struct checked_range){ "the output", &used[1],
						  destination_bytes, false };
	if (description->source_slot_stride != 0) {
		used[count] = (struct linear_range){
			description->source.byte_offset +
				description->source_slot_stride,
			source_bytes
		};
		ranges[count] =
			(struct checked_range){ "the second source slot",
						&used[count], source_bytes,
						false };
		count++;
	}
	if (description->output_slot_stride != 0) {
		used[count] = (struct linear_range){
			description->output.byte_offset +
				description->output_slot_stride,
			destination_bytes
		};
		ranges[count] =
			(struct checked_range){ "the second output slot",
						&used[count], destination_bytes,
						false };
		count++;
	}
	if (has_routed) {
		used[count] = (struct linear_range){
			description->routed_sums.byte_offset, routed_bytes
		};
		ranges[count] = (struct checked_range){ "the routed sums",
							&used[count],
							routed_bytes, false };
		count++;
	}
	if (has_step) {
		used[count] =
			(struct linear_range){ description->step.byte_offset,
					       STEP_BLOCK_BYTES_VALUE };
		ranges[count] =
			(struct checked_range){ "the step block", &used[count],
						STEP_BLOCK_BYTES_VALUE, false };
		count++;
	}
	if (!prepare_ranges_are_valid(what, ranges, count, arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct tp2_join_arguments arguments = {
		.source_offset = description->source.byte_offset,
		.routed_sums_offset = description->routed_sums.byte_offset,
		.output_offset = description->output.byte_offset,
		.step_offset = description->step.byte_offset,
		.source_slot_stride = description->source_slot_stride,
		.output_slot_stride = description->output_slot_stride,
		.width = width,
		.row_count = rows,
		.source_row_stride = description->source_row_stride,
		.routed_row_stride = description->routed_row_stride,
		.output_row_stride = description->output_row_stride,
		.operation = description->operation,
		.has_routed_sums = has_routed ? 1u : 0u,
		.reserved_zero = 0
	};
	const uint32_t blocks =
		(width + TP2_JOIN_THREADS - 1u) / TP2_JOIN_THREADS;
	prepare_launch_fill(launch_out, kernel, blocks, rows, TP2_JOIN_THREADS,
			    0, &arguments, TP2_JOIN_ARGUMENT_BYTES);
	return STATUS_OK;
}
