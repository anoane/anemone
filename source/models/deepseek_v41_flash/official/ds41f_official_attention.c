/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "ds41f_official_attention.h"

#include <stddef.h>
#include <string.h>
#include "engine/library/prepare_checks.h"

result_status
ds41f_official_attention_kernel_prepare(const struct backend_operations *backend,
				  struct device_card *card,
				  uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, DS41F_MODULE_OFFICIAL_ATTENTION, DS41F_KERNEL_OFFICIAL_ATTENTION,
		DS41F_OFFICIAL_ATTENTION_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status
ds41f_official_attention_prepare(const struct ds41f_attention_core_description *description,
			   uint32_t operand_types, uint64_t arena_byte_count,
			   uint32_t kernel, struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL || description->row_count != 1 ||
	    description->session_capacity != 1 ||
	    description->first_head != 0 ||
	    description->head_count != DS41F_HEADS_VALUE ||
	    description->order != LINEAR_ORDER_REFERENCE ||
	    description->window_mode != DS41F_ATTENTION_CORE_WINDOW_CAUSAL ||
	    description->output_compact != 0 || description->cone_layer != 0 ||
	    operand_types > DS41F_OFFICIAL_ATTENTION_OPERAND_TYPES)
		return STATUS_INVALID_ARGUMENT;
	/* Reuse the common core's range/count admission. This fused form needs no
	 * scores range, which that API checks only for its separate-score form. */
	struct ds41f_attention_core_description checked = *description;
	checked.order = LINEAR_ORDER_FIXED;
	const struct ds41f_attention_core_kernels kernels = {
		.attention_core = kernel,
		.attention_core_scores = kernel,
		.attention_core_values = kernel,
		.attention_core_block = BACKEND_NO_KERNEL
	};
	struct kernel_launch validated[DS41F_ATTENTION_CORE_MOST_LAUNCHES];
	uint32_t count = 0;
	const result_status status = ds41f_attention_core_prepare(
		&checked, arena_byte_count, &kernels, validated, &count);
	if (status != STATUS_OK)
		return status;
	struct ds41f_official_attention_arguments arguments = {
		.operand_types = operand_types
	};
	memcpy(&arguments.source, validated[0].argument,
	       sizeof(arguments.source));
	prepare_launch_fill(launch_out, kernel, 1, DS41F_HEADS_VALUE,
			    DS41F_OFFICIAL_ATTENTION_THREADS,
			    DS41F_OFFICIAL_ATTENTION_SHARED_BYTES, &arguments,
			    DS41F_OFFICIAL_ATTENTION_ARGUMENT_BYTES);
	return STATUS_OK;
}
