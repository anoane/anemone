/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "ds41f_official_router_order.h"

#include <stddef.h>
#include "engine/library/prepare_checks.h"

result_status
ds41f_official_router_order_kernel_prepare(const struct backend_operations *backend,
				     struct device_card *card,
				     uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status =
		backend->kernel_prepare(card, DS41F_MODULE_OFFICIAL_ROUTER_ORDER,
					DS41F_KERNEL_OFFICIAL_ROUTER_ORDER,
					DS41F_ROUTER_SELECT_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status
ds41f_official_router_order_prepare(const struct ds41f_router_description *description,
			      uint64_t arena_byte_count, uint32_t kernel,
			      struct kernel_launch *launch_out)
{
	if (description == NULL || description->experts == NULL ||
	    launch_out == NULL || kernel == BACKEND_NO_KERNEL ||
	    description->experts->row_count != 1 ||
	    description->expert_count != DS41F_EXPERTS ||
	    description->pick_count != DS41F_EXPERTS_PER_ROW ||
	    description->order != LINEAR_ORDER_REFERENCE)
		return STATUS_INVALID_ARGUMENT;
	const struct router_kernels_record kernels = { .select = kernel };
	struct kernel_launch validated;
	const result_status status = ds41f_router_select_prepare(
		description, arena_byte_count, &kernels, &validated);
	if (status != STATUS_OK)
		return status;
	prepare_launch_fill(launch_out, kernel, 1, 1, 1, 0, validated.argument,
			    DS41F_ROUTER_SELECT_ARGUMENT_BYTES);
	return STATUS_OK;
}
