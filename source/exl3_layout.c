/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Lossless EXL3 load-time packing. No device or token-loop dependency. */
#include "anemone/exl3_layout.h"

#include <stdint.h>
#include <string.h>

static uint64_t exl3_code_bytes(uint32_t inputs, uint32_t outputs,
				uint32_t bits_per_weight)
{
	return (uint64_t)inputs * outputs / EXL3_BITS_PER_BYTE *
	       bits_per_weight;
}

static uint64_t exl3_matrix_bytes(uint32_t inputs, uint32_t outputs,
				  uint32_t bits_per_weight)
{
	return exl3_code_bytes(inputs, outputs, bits_per_weight) +
	       ((uint64_t)inputs + outputs) * EXL3_HALF_BYTES;
}

result_status
bb_exl3_matrix_slice_prepare(const struct bb_exl3_matrix_slice *slice,
			     struct bb_exl3_slice_sizes *sizes_out)
{
	if (slice == NULL || sizes_out == NULL || slice->reserved_zero != 0 ||
	    slice->input_count == 0 || slice->output_count == 0 ||
	    slice->input_size % EXL3_TRANSFORM != 0 ||
	    slice->output_size % EXL3_TRANSFORM != 0 ||
	    slice->input_begin % EXL3_TRANSFORM != 0 ||
	    slice->input_count % EXL3_TRANSFORM != 0 ||
	    slice->output_begin % EXL3_TRANSFORM != 0 ||
	    slice->output_count % EXL3_TRANSFORM != 0 ||
	    slice->input_begin > slice->input_size ||
	    slice->input_count > slice->input_size - slice->input_begin ||
	    slice->output_begin > slice->output_size ||
	    slice->output_count > slice->output_size - slice->output_begin ||
	    slice->bits_per_weight < EXL3_MINIMUM_BITS ||
	    slice->bits_per_weight > EXL3_MAXIMUM_BITS) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct bb_exl3_slice_sizes sizes = {
		.source_bytes = exl3_matrix_bytes(slice->input_size,
						  slice->output_size,
						  slice->bits_per_weight),
		.packed_bytes = exl3_matrix_bytes(slice->input_count,
						  slice->output_count,
						  slice->bits_per_weight)
	};
	*sizes_out = sizes;
	return STATUS_OK;
}

result_status
bb_exl3_matrix_slice_copy(const struct bb_exl3_matrix_slice *slice,
			  const void *source, size_t source_bytes,
			  void *destination, size_t destination_bytes)
{
	if (slice == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	/* Snapshot metadata before touching a destination that may contain it. */
	const struct bb_exl3_matrix_slice description = *slice;
	slice = &description;
	struct bb_exl3_slice_sizes sizes;
	if (source == NULL || destination == NULL ||
	    bb_exl3_matrix_slice_prepare(slice, &sizes) != STATUS_OK ||
	    sizes.source_bytes > source_bytes ||
	    sizes.packed_bytes > destination_bytes) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uintptr_t source_address = (uintptr_t)source;
	const uintptr_t destination_address = (uintptr_t)destination;
	if (source_bytes > UINTPTR_MAX - source_address ||
	    destination_bytes > UINTPTR_MAX - destination_address ||
	    (source_address < destination_address + destination_bytes &&
	     destination_address < source_address + source_bytes)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const unsigned char *const source_values = source;
	unsigned char *const destination_values = destination;
	const uint64_t source_row_bytes = exl3_code_bytes(
		EXL3_TILE, slice->output_size, slice->bits_per_weight);
	const uint64_t copied_row_bytes = exl3_code_bytes(
		EXL3_TILE, slice->output_count, slice->bits_per_weight);
	const uint64_t first_code_byte =
		(uint64_t)(slice->input_begin / EXL3_TILE) * source_row_bytes +
		exl3_code_bytes(EXL3_TILE, slice->output_begin,
				slice->bits_per_weight);
	const uint32_t code_rows = slice->input_count / EXL3_TILE;
	for (uint32_t row = 0; row < code_rows; row++) {
		memcpy(destination_values + (uint64_t)row * copied_row_bytes,
		       source_values + first_code_byte +
			       (uint64_t)row * source_row_bytes,
		       (size_t)copied_row_bytes);
	}
	const uint64_t source_codes = exl3_code_bytes(
		slice->input_size, slice->output_size, slice->bits_per_weight);
	const uint64_t destination_codes =
		exl3_code_bytes(slice->input_count, slice->output_count,
				slice->bits_per_weight);
	const size_t input_scale_bytes =
		(size_t)slice->input_count * EXL3_HALF_BYTES;
	const size_t output_scale_bytes =
		(size_t)slice->output_count * EXL3_HALF_BYTES;
	memcpy(destination_values + destination_codes,
	       source_values + source_codes +
		       (uint64_t)slice->input_begin * EXL3_HALF_BYTES,
	       input_scale_bytes);
	memcpy(destination_values + destination_codes + input_scale_bytes,
	       source_values + source_codes +
		       ((uint64_t)slice->input_size + slice->output_begin) *
			       EXL3_HALF_BYTES,
	       output_scale_bytes);
	return STATUS_OK;
}

result_status
bb_exl3_output_shard_prepare(uint32_t outer_size, uint32_t inner_size,
			     uint32_t bits_per_weight, uint32_t rank_count,
			     uint32_t rank, enum bb_exl3_product_order order,
			     struct bb_exl3_output_shard_plan *plan_out)
{
	if (plan_out == NULL ||
	    (rank_count != 1 && rank_count != BB_EXL3_TP_RANKS) ||
	    rank >= rank_count ||
	    (order != BB_EXL3_REFERENCE_ORDER &&
	     order != BB_EXL3_EIGHT_PART_ORDER) ||
	    outer_size % EXL3_TRANSFORM != 0 ||
	    inner_size % EXL3_TRANSFORM != 0 ||
	    outer_size / EXL3_TRANSFORM < rank_count ||
	    inner_size / EXL3_TRANSFORM < rank_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t outer_blocks = outer_size / EXL3_TRANSFORM;
	const uint32_t inner_blocks = inner_size / EXL3_TRANSFORM;
	const uint32_t outer_begin =
		outer_blocks * rank / rank_count * EXL3_TRANSFORM;
	const uint32_t outer_end =
		outer_blocks * (rank + 1) / rank_count * EXL3_TRANSFORM;
	const uint32_t inner_begin =
		inner_blocks * rank / rank_count * EXL3_TRANSFORM;
	const uint32_t inner_end =
		inner_blocks * (rank + 1) / rank_count * EXL3_TRANSFORM;
	const struct bb_exl3_output_shard_plan plan = {
		.gate_up = { outer_size, inner_size, 0, outer_size, inner_begin,
			     inner_end - inner_begin, bits_per_weight, 0 },
		.down = { inner_size, outer_size, 0, inner_size, outer_begin,
			  outer_end - outer_begin, bits_per_weight, 0 },
		.rank_count = rank_count,
		.rank = rank,
		.product_slices = order == BB_EXL3_REFERENCE_ORDER ?
					  EXL3_PRODUCT_SLICES :
					  EXL3_RELAXED_SLICES,
		.order = order
	};
	struct bb_exl3_slice_sizes sizes;
	const result_status gate_status =
		bb_exl3_matrix_slice_prepare(&plan.gate_up, &sizes);
	if (gate_status != STATUS_OK) {
		return gate_status;
	}
	const result_status down_status =
		bb_exl3_matrix_slice_prepare(&plan.down, &sizes);
	if (down_status != STATUS_OK) {
		return down_status;
	}
	*plan_out = plan;
	return STATUS_OK;
}
