/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/ep_filter.h"
#include "prepare_common.h"
#include "anemone/model_layout.h"
#include <stddef.h>

#define FILTER_RANGES 3u

result_status ep_filter_kernel_prepare(const struct backend_operations *backend,
				       struct device_card *card,
				       uint32_t *kernel_out)
{
	if (backend == NULL || card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	if (backend->kernel_prepare == NULL)
		return STATUS_UNSUPPORTED;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_EP_FILTER, KERNEL_EP_FILTER,
		EP_FILTER_ARGUMENT_BYTES, &kernel);
	if (status == STATUS_OK)
		*kernel_out = kernel;
	return status;
}

result_status ep_filter_prepare(const struct ep_filter_description *description,
				uint64_t arena_byte_count, uint32_t kernel,
				struct kernel_launch *launch_out)
{
	static const char what[] = "the component EP2 job filter";
	if (!prepare_arguments_are_given(what, description, &kernel,
					 launch_out))
		return STATUS_INVALID_ARGUMENT;
	const uint32_t rows = description->row_count;
	if (rows == 0 || rows > LINEAR_MAXIMUM_ROWS ||
	    description->unowned_gate_offset % LINEAR_OFFSET_ALIGNMENT_BYTES !=
		    0 ||
	    description->unowned_gate_offset >= arena_byte_count ||
	    !prepare_kernel_is_prepared(what, kernel, KERNEL_EP_FILTER))
		return STATUS_INVALID_ARGUMENT;
	const uint32_t pairs = rows * MODEL_EXPERTS_PER_ROW;
	const struct checked_range ranges[FILTER_RANGES] = {
		{ "the jobs", &description->jobs, linear_jobs_byte_count(pairs),
		  false },
		{ "the hidden pairs", &description->hidden,
		  linear_values_byte_count(pairs, EP_FILTER_HIDDEN_WIDTH),
		  false },
		{ "the output pairs", &description->output,
		  linear_values_byte_count(pairs, MODEL_HIDDEN), false }
	};
	if (!prepare_ranges_are_valid(what, ranges, FILTER_RANGES,
				      arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct ep_filter_arguments arguments = {
		description->jobs.byte_offset,
		description->hidden.byte_offset,
		description->output.byte_offset,
		description->unowned_gate_offset,
		pairs,
		pairs,
		EP_FILTER_HIDDEN_WIDTH,
		MODEL_HIDDEN
	};
	prepare_launch_fill(launch_out, kernel, EP_FILTER_BLOCKS, 1u,
			    EP_FILTER_THREADS, 0, &arguments,
			    sizeof(arguments));
	return STATUS_OK;
}
