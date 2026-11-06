/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copy storage tiles and their independent scale rows; all sizes and bounds are checked before the first copy. */
#include "anemone/tp_weights.h"
#include "anemone/exl3_layout.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define OUTPUTS_PER_TILE 16u
#define FP8_OUTPUTS_PER_SCALE 32u
#define FP8_INPUTS_PER_GROUP 32u
#define MXFP4_INPUTS_PER_GROUP 64u
#define CODE_BYTES_PER_GROUP 512ull
#define MXFP4_SCALE_BYTES_PER_GROUP 32ull
#define MXFP4_WEIGHTS_PER_BYTE 2u

/* The shared EXL3 packer owns trellis and scale placement; this adapter only maps the generic rectangle. */
static struct bb_exl3_matrix_slice
tensor_exl3_slice_of(const struct tensor_shard_description *description)
{
	const struct bb_exl3_matrix_slice slice = {
		.input_size = description->source_shape.input_size,
		.output_size = description->source_shape.output_size,
		.input_begin = description->first_input,
		.input_count = description->input_count,
		.output_begin = description->first_output,
		.output_count = description->output_count,
		.bits_per_weight = description->source_shape.bits_per_weight
	};
	return slice;
}

MUST_CHECK result_status
tensor_shard_prepare(const struct tensor_shard_description *description,
		     struct tensor_shard_plan *plan_out)
{
	if (description == NULL || plan_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (description->format == WEIGHT_FORMAT_NONE) {
		/* plain 16-bit rows: a contiguous range of whole rows */
		const struct matrix_shape_record *const source =
			&description->source_shape;
		const uint64_t row_bytes = (uint64_t)source->input_size *
					   TENSOR_SHARD_PLAIN_BITS / 8u;
		if (source->bits_per_weight != TENSOR_SHARD_PLAIN_BITS ||
		    source->reserved_must_be_zero != 0 ||
		    source->output_size == 0 || source->input_size == 0 ||
		    description->output_count == 0 ||
		    description->first_output > source->output_size ||
		    description->output_count >
			    source->output_size - description->first_output ||
		    description->first_input != 0 ||
		    description->input_count != source->input_size) {
			return STATUS_INVALID_ARGUMENT;
		}
		*plan_out = (struct tensor_shard_plan){
			.description = *description,
			.shard_shape = { .output_size =
						 description->output_count,
					 .input_size = source->input_size,
					 .bits_per_weight =
						 source->bits_per_weight },
			.source_byte_count = source->output_size * row_bytes,
			.shard_byte_count =
				description->output_count * row_bytes
		};
		return STATUS_OK;
	}
	const struct weight_format_operations *const format =
		weight_format_operations_of(description->format);
	if (format == NULL) {
		return STATUS_UNSUPPORTED;
	}
	struct tensor_shard_plan prepared = {
		.description = *description,
		.shard_shape = { .output_size = description->output_count,
				 .input_size = description->input_count,
				 .bits_per_weight = description->source_shape
							    .bits_per_weight }
	};
	result_status status = format->stored_byte_count(
		&description->source_shape, &prepared.source_byte_count);
	if (status != STATUS_OK) {
		return status;
	}
	if (description->format == WEIGHT_FORMAT_EXL3) {
		const struct bb_exl3_matrix_slice slice =
			tensor_exl3_slice_of(description);
		struct bb_exl3_slice_sizes sizes;
		status = bb_exl3_matrix_slice_prepare(&slice, &sizes);
		if (status != STATUS_OK) {
			return status;
		}
		if (sizes.source_bytes != prepared.source_byte_count) {
			return STATUS_INVALID_ARGUMENT;
		}
		prepared.shard_byte_count = sizes.packed_bytes;
		*plan_out = prepared;
		return STATUS_OK;
	}
	const uint32_t input_granule =
		description->format == WEIGHT_FORMAT_E2M1_SCALED ?
			MXFP4_INPUTS_PER_GROUP :
			FP8_INPUTS_PER_GROUP;
	if (description->first_output > description->source_shape.output_size ||
	    description->output_count == 0 ||
	    description->output_count > description->source_shape.output_size -
						description->first_output ||
	    description->first_input > description->source_shape.input_size ||
	    description->input_count == 0 ||
	    description->input_count > description->source_shape.input_size -
					       description->first_input ||
	    description->first_output % OUTPUTS_PER_TILE != 0 ||
	    description->first_input % input_granule != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	status = format->stored_byte_count(&prepared.shard_shape,
					   &prepared.shard_byte_count);
	if (status != STATUS_OK) {
		return status;
	}
	/* An FP8 block shares one scale between two output tiles. Moving an odd tile to output zero would combine
	 * different source scales inside one destination block; a byte copy cannot represent that rectangle. */
	if (description->format == WEIGHT_FORMAT_E4M3_SCALED &&
	    description->first_output % FP8_OUTPUTS_PER_SCALE != 0) {
		return STATUS_UNSUPPORTED;
	}
	*plan_out = prepared;
	return STATUS_OK;
}

static bool same_shape(const struct matrix_shape_record *first,
		       const struct matrix_shape_record *second)
{
	return first->output_size == second->output_size &&
	       first->input_size == second->input_size &&
	       first->bits_per_weight == second->bits_per_weight &&
	       first->reserved_must_be_zero == second->reserved_must_be_zero;
}

MUST_CHECK result_status
tensor_shard_extract(const struct tensor_shard_plan *plan, const void *source,
		     uint64_t source_byte_count, void *destination,
		     uint64_t destination_byte_count)
{
	if (plan == NULL || source == NULL || destination == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct tensor_shard_plan checked;
	const result_status status =
		tensor_shard_prepare(&plan->description, &checked);
	if (status != STATUS_OK) {
		return status;
	}
	const uintptr_t source_address = (uintptr_t)source;
	const uintptr_t destination_address = (uintptr_t)destination;
	if (source_byte_count != checked.source_byte_count ||
	    destination_byte_count < checked.shard_byte_count ||
	    plan->source_byte_count != checked.source_byte_count ||
	    plan->shard_byte_count != checked.shard_byte_count ||
	    !same_shape(&plan->shard_shape, &checked.shard_shape) ||
	    source_byte_count > SIZE_MAX ||
	    checked.shard_byte_count > SIZE_MAX ||
	    source_byte_count > UINTPTR_MAX - source_address ||
	    checked.shard_byte_count > UINTPTR_MAX - destination_address ||
	    (source_address < destination_address + checked.shard_byte_count &&
	     destination_address < source_address + source_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct tensor_shard_description description = checked.description;
	if (description.format == WEIGHT_FORMAT_NONE) {
		memcpy(destination,
		       (const uint8_t *)source +
			       (uint64_t)description.first_output *
				       (checked.shard_byte_count /
					description.output_count),
		       (size_t)checked.shard_byte_count);
		return STATUS_OK;
	}
	if (description.format == WEIGHT_FORMAT_EXL3) {
		const struct bb_exl3_matrix_slice slice =
			tensor_exl3_slice_of(&description);
		return bb_exl3_matrix_slice_copy(
			&slice, source, (size_t)source_byte_count, destination,
			(size_t)checked.shard_byte_count);
	}
	const bool mxfp4 = description.format == WEIGHT_FORMAT_E2M1_SCALED;
	const uint32_t input_granule = mxfp4 ? MXFP4_INPUTS_PER_GROUP :
					       FP8_INPUTS_PER_GROUP;
	const uint64_t source_groups =
		description.source_shape.input_size / input_granule;
	const uint64_t shard_groups = description.input_count / input_granule;
	const uint64_t first_group = description.first_input / input_granule;
	const uint64_t first_tile = description.first_output / OUTPUTS_PER_TILE;
	const uint64_t tile_count = description.output_count / OUTPUTS_PER_TILE;
	const uint64_t code_divisor = mxfp4 ? MXFP4_WEIGHTS_PER_BYTE : 1;
	const uint64_t source_codes =
		(uint64_t)description.source_shape.output_size *
		description.source_shape.input_size / code_divisor;
	const uint64_t shard_codes = (uint64_t)description.output_count *
				     description.input_count / code_divisor;
	const uint8_t *const source_bytes = source;
	uint8_t *const destination_bytes = destination;
	for (uint64_t tile = 0; tile < tile_count; tile++) {
		const uint64_t source_group =
			(first_tile + tile) * source_groups + first_group;
		memcpy(destination_bytes +
			       tile * shard_groups * CODE_BYTES_PER_GROUP,
		       source_bytes + source_group * CODE_BYTES_PER_GROUP,
		       (size_t)(shard_groups * CODE_BYTES_PER_GROUP));
		if (mxfp4) {
			memcpy(destination_bytes + shard_codes +
				       tile * shard_groups *
					       MXFP4_SCALE_BYTES_PER_GROUP,
			       source_bytes + source_codes +
				       source_group *
					       MXFP4_SCALE_BYTES_PER_GROUP,
			       (size_t)(shard_groups *
					MXFP4_SCALE_BYTES_PER_GROUP));
		}
	}
	if (!mxfp4) {
		const uint64_t first_block =
			description.first_output / FP8_OUTPUTS_PER_SCALE;
		const uint64_t block_count =
			((uint64_t)description.output_count +
			 FP8_OUTPUTS_PER_SCALE - 1) /
			FP8_OUTPUTS_PER_SCALE;
		for (uint64_t block = 0; block < block_count; block++) {
			memcpy(destination_bytes + shard_codes +
				       block * shard_groups,
			       source_bytes + source_codes +
				       (first_block + block) * source_groups +
				       first_group,
			       (size_t)shard_groups);
		}
	}
	return STATUS_OK;
}
