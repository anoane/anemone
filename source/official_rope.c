/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_rope.h"

#include <stddef.h>
#include "prepare_common.h"

#define DISJOINT_RANGES 3u
#define IN_PLACE_RANGES 2u

result_status
official_rope_kernel_prepare(const struct backend_operations *backend,
			     struct device_card *card, uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_ROPE, KERNEL_OFFICIAL_ROPE,
		OFFICIAL_ROPE_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status
official_rope_prepare(const struct official_rope_description *description,
		      uint64_t arena_byte_count, uint32_t kernel,
		      struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	if ((description->width != OFFICIAL_ROPE_WIDTH &&
	     description->width != MODEL_HEAD_DIMS) ||
	    description->heads == 0 || description->heads > MODEL_HEADS_VALUE ||
	    description->rows == 0 || description->rows > LINEAR_MAXIMUM_ROWS ||
	    description->inverse > 1)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t bytes = (uint64_t)description->width *
			       description->heads * description->rows *
			       sizeof(uint16_t);
	const struct checked_range ranges[] = {
		{ "input", &description->input, bytes, false },
		{ "rotation", &description->rotation,
		  (uint64_t)description->rows * OFFICIAL_ROPE_WIDTH *
			  sizeof(float),
		  false },
		{ "output", &description->output, bytes, false }
	};
	const bool in_place = description->input.byte_offset ==
				      description->output.byte_offset &&
			      description->input.byte_count ==
				      description->output.byte_count;
	if (!prepare_ranges_are_valid("an official rotation", ranges,
				      in_place ? IN_PLACE_RANGES :
						 DISJOINT_RANGES,
				      arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct official_rope_arguments arguments = {
		.input_offset = description->input.byte_offset,
		.rotation_offset = description->rotation.byte_offset,
		.output_offset = description->output.byte_offset,
		.width = description->width,
		.heads = description->heads,
		.rows = description->rows,
		.inverse = description->inverse
	};
	prepare_launch_fill(launch_out, kernel, description->heads,
			    description->rows, OFFICIAL_ROPE_THREADS, 0,
			    &arguments, sizeof(arguments));
	return STATUS_OK;
}
