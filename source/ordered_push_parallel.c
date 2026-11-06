/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/ordered_push_parallel.h"

#include <stddef.h>
#include <string.h>

#define PARALLEL_PUBLISH 0u
#define PARALLEL_REDUCE 1u

result_status ordered_push_parallel_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t kernels_out[ORDERED_PUSH_PARALLEL_LAUNCHES])
{
	if (backend == NULL || card == NULL || kernels_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	if (backend->kernel_prepare == NULL)
		return STATUS_UNSUPPORTED;
	const char *const names[ORDERED_PUSH_PARALLEL_LAUNCHES] = {
		KERNEL_ORDERED_PUSH_PARALLEL_PUBLISH,
		KERNEL_ORDERED_PUSH_PARALLEL_REDUCE
	};
	uint32_t kernels[ORDERED_PUSH_PARALLEL_LAUNCHES];
	result_status status = STATUS_OK;
	for (uint32_t index = 0;
	     index < ORDERED_PUSH_PARALLEL_LAUNCHES && status == STATUS_OK;
	     index++)
		status = backend->kernel_prepare(
			card, KERNEL_FILE_ORDERED_PUSH_PARALLEL, names[index],
			ORDERED_PUSH_ARGUMENT_BYTES, &kernels[index]);
	if (status == STATUS_OK)
		(void)memcpy(kernels_out, kernels, sizeof(kernels));
	return status;
}

result_status ordered_push_parallel_prepare(
	const struct ordered_push_description *description,
	const uint32_t kernels[ORDERED_PUSH_PARALLEL_LAUNCHES],
	struct kernel_launch launches_out[ORDERED_PUSH_PARALLEL_LAUNCHES])
{
	if (kernels == NULL || launches_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launches[ORDERED_PUSH_PARALLEL_LAUNCHES];
	result_status status =
		ordered_push_prepare(description, kernels[PARALLEL_PUBLISH],
				     &launches[PARALLEL_PUBLISH]);
	if (status == STATUS_OK)
		status = ordered_push_prepare(description,
					      kernels[PARALLEL_REDUCE],
					      &launches[PARALLEL_REDUCE]);
	if (status == STATUS_OK) {
		launches[PARALLEL_PUBLISH].grid[0] =
			ORDERED_PUSH_PARALLEL_BLOCKS;
		(void)memcpy(launches_out, launches, sizeof(launches));
	}
	return status;
}
