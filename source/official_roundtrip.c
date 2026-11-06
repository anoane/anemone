/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_roundtrip.h"

#include <stddef.h>
#include "prepare_common.h"

#define RANGE_COUNT 2u

result_status
official_roundtrip_kernel_prepare(const struct backend_operations *backend,
				  struct device_card *card,
				  uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_ROUNDTRIP, KERNEL_OFFICIAL_ROUNDTRIP,
		OFFICIAL_ROUNDTRIP_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status official_roundtrip_prepare(
	const struct official_roundtrip_description *description,
	uint64_t arena_byte_count, uint32_t kernel,
	struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t width = description->width,
		       rows = description->row_count;
	if (width == 0 || width > OFFICIAL_ROUNDTRIP_MOST_WIDTH ||
	    width % OFFICIAL_ROUNDTRIP_BLOCK_VALUES != 0 || rows == 0 ||
	    rows > LINEAR_MAXIMUM_ROWS)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t bytes = (uint64_t)width * rows * sizeof(uint16_t);
	const struct checked_range ranges[] = {
		{ "input", &description->input, bytes, false },
		{ "output", &description->output, bytes, false }
	};
	const bool in_place = description->input.byte_offset ==
				      description->output.byte_offset &&
			      description->input.byte_count ==
				      description->output.byte_count;
	if (!prepare_ranges_are_valid("an official round trip", ranges,
				      in_place ? 1u : RANGE_COUNT,
				      arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct official_roundtrip_arguments arguments = {
		description->input.byte_offset, description->output.byte_offset,
		width, rows
	};
	prepare_launch_fill(launch_out, kernel,
			    (width + OFFICIAL_ROUNDTRIP_THREADS - 1u) /
				    OFFICIAL_ROUNDTRIP_THREADS,
			    rows, OFFICIAL_ROUNDTRIP_THREADS, 0, &arguments,
			    sizeof(arguments));
	return STATUS_OK;
}
