/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_norm.h"

#include <stdbool.h>
#include <stddef.h>

#include "prepare_common.h"

#define COUNT_OF(values) ((uint32_t)(sizeof(values) / sizeof((values)[0])))
#define VECTOR_VALUES 4u
#define TWO 2u

result_status
official_norm_kernel_prepare(const struct backend_operations *backend,
			     struct device_card *card, uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_NORM, KERNEL_OFFICIAL_NORM,
		OFFICIAL_NORM_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

static bool rows_are_admitted(uint32_t rows)
{
	return rows == 1 || rows == OFFICIAL_NORM_ODD_ROWS ||
	       rows == MODEL_STREAMS || rows == LINEAR_MAXIMUM_ROWS;
}

result_status
official_norm_shape_prepare(const struct official_norm_description *description,
			    uint32_t reference_rows, uint64_t arena_byte_count,
			    uint32_t kernel, struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t width = description->width,
		       rows = description->row_count;
	if ((width != MODEL_HEAD_DIMS && width != MODEL_QUERY_RANK &&
	     width != MODEL_HIDDEN) ||
	    !rows_are_admitted(rows) || !rows_are_admitted(reference_rows))
		return STATUS_INVALID_ARGUMENT;
	const uint64_t row_bytes = (uint64_t)width * sizeof(uint16_t);
	const struct checked_range ranges[] = {
		{ "input", &description->input, row_bytes * rows, false },
		{ "weight", &description->weight, row_bytes, false },
		{ "output", &description->output, row_bytes * rows, false },
		{ "statistics", &description->statistics,
		  (uint64_t)rows * OFFICIAL_NORM_STATISTICS * sizeof(float),
		  false }
	};
	if (!prepare_ranges_are_valid("an official norm", ranges,
				      COUNT_OF(ranges), arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	uint32_t threads = 1, row_groups = 1;
	while (threads * TWO <= width / VECTOR_VALUES &&
	       threads * TWO <= OFFICIAL_NORM_MOST_THREADS)
		threads *= TWO;
	while (row_groups * TWO <= reference_rows)
		row_groups *= TWO;
	while (threads * row_groups > OFFICIAL_NORM_MOST_THREADS)
		threads /= TWO;
	const struct official_norm_arguments arguments = {
		.input_offset = description->input.byte_offset,
		.weight_offset = description->weight.byte_offset,
		.output_offset = description->output.byte_offset,
		.statistics_offset = description->statistics.byte_offset,
		.width = width,
		.mean_factor = 1.0f / (float)width,
		.epsilon = MODEL_NORM_EPSILON
	};
	prepare_launch_fill(launch_out, kernel, 1, rows, threads, 0, &arguments,
			    sizeof(arguments));
	return STATUS_OK;
}

result_status
official_norm_prepare(const struct official_norm_description *description,
		      uint64_t arena_byte_count, uint32_t kernel,
		      struct kernel_launch *launch_out)
{
	return official_norm_shape_prepare(
		description, description == NULL ? 0 : description->row_count,
		arena_byte_count, kernel, launch_out);
}
