/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "ds41f_official_window.h"

#include <stddef.h>
#include "engine/library/prepare_checks.h"

#define RANGE_COUNT 5u

result_status
ds41f_official_window_kernel_prepare(const struct backend_operations *backend,
				     struct device_card *card,
				     uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, DS41F_MODULE_OFFICIAL_WINDOW,
		DS41F_KERNEL_OFFICIAL_WINDOW,
		DS41F_OFFICIAL_WINDOW_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status ds41f_official_window_prepare(
	const struct ds41f_official_window_description *description,
	uint32_t reference_rows, uint64_t arena_byte_count, uint32_t kernel,
	struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL ||
	    description->norm.width != DS41F_HEAD_DIMS)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch validated;
	const result_status status = ds41f_official_norm_shape_prepare(
		&description->norm, reference_rows, arena_byte_count, kernel,
		&validated);
	if (status != STATUS_OK)
		return status;
	const struct ds41f_official_norm_description *const norm =
		&description->norm;
	const uint64_t row_bytes = (uint64_t)DS41F_HEAD_DIMS * sizeof(uint16_t);
	const struct checked_range ranges[] = {
		{ "input", &norm->input, row_bytes * norm->row_count, false },
		{ "weight", &norm->weight, row_bytes, false },
		{ "rotation", &description->rotation,
		  (uint64_t)norm->row_count *
			  (uint64_t)DS41F_OFFICIAL_ROPE_WIDTH * sizeof(float),
		  false },
		{ "output", &norm->output, row_bytes * norm->row_count, false },
		{ "statistics", &norm->statistics,
		  (uint64_t)norm->row_count * DS41F_OFFICIAL_NORM_STATISTICS *
			  sizeof(float),
		  false }
	};
	if (!prepare_ranges_are_valid("an official window composition", ranges,
				      RANGE_COUNT, arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct ds41f_official_window_arguments arguments = {
		.input_offset = norm->input.byte_offset,
		.weight_offset = norm->weight.byte_offset,
		.rotation_offset = description->rotation.byte_offset,
		.output_offset = norm->output.byte_offset,
		.statistics_offset = norm->statistics.byte_offset,
		.rows = norm->row_count,
		.mean_factor = 1.0f / (float)DS41F_HEAD_DIMS,
		.epsilon = DS41F_NORM_EPSILON
	};
	prepare_launch_fill(launch_out, kernel, 1, norm->row_count,
			    validated.block[0], 0, &arguments,
			    sizeof(arguments));
	return STATUS_OK;
}
