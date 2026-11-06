/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_index.h"

#include <stddef.h>
#include "prepare_common.h"

result_status
official_index_kernel_prepare(const struct backend_operations *backend,
			      struct device_card *card, uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_INDEX, KERNEL_OFFICIAL_INDEX,
		OFFICIAL_INDEX_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status
official_index_prepare(const struct index_scan_description *description,
		       uint64_t arena_byte_count, uint32_t kernel,
		       struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL || description->row_count != 1 ||
	    description->session_capacity != 1 ||
	    description->context_token_limit > MODEL_SELECTED ||
	    description->step_row_first != 0 || description->split_rows > 1u)
		return STATUS_INVALID_ARGUMENT;
	const struct index_scan_kernels kernels = { .scan = kernel,
						    .select = kernel,
						    .candidates = kernel };
	struct kernel_launch validated[INDEX_SCAN_LAUNCH_COUNT];
	uint32_t count = 0;
	const result_status status = index_scan_prepare(
		description, arena_byte_count, &kernels, validated, &count);
	if (status != STATUS_OK)
		return status;
	const struct official_index_arguments arguments = {
		.step_offset = description->step.byte_offset,
		.selection_offset = description->selection.byte_offset,
		.candidates_offset =
			description->candidate_role == INDEX_CANDIDATES_SOURCE ?
				description->candidates.byte_offset :
				INDEX_NO_CANDIDATES,
		.context_token_limit = description->context_token_limit,
		.ratio = description->ratio
	};
	prepare_launch_fill(launch_out, kernel, 1, 1, OFFICIAL_INDEX_THREADS, 0,
			    &arguments, OFFICIAL_INDEX_ARGUMENT_BYTES);
	return STATUS_OK;
}
