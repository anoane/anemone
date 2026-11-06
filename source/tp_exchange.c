/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* TP-only active-row and shared-channel phase preparation. */
#include "anemone/tp_exchange.h"

#include <stddef.h>

#include "prepare_common.h"
#include "ordered_push_internal.h"

_Static_assert(TP_EXCHANGE_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES,
	       "a TP phase fits the existing launch record");

result_status
tp_exchange_kernel_prepare(const struct backend_operations *backend,
			   struct device_card *card, uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	return backend->kernel_prepare(card, KERNEL_FILE_TP_EXCHANGE,
				       KERNEL_TP_EXCHANGE,
				       TP_EXCHANGE_ARGUMENT_BYTES, kernel_out);
}

result_status
tp_exchange_prepare(const struct ordered_push_description *description,
		    uint32_t layer, uint32_t layer_count, uint32_t kernel,
		    struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL || layer_count == 0 ||
	    layer_count > TP_EXCHANGE_PHASES || layer >= layer_count ||
	    description->arguments.element_count % MODEL_HIDDEN != 0)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch checked;
	const result_status status = ordered_push_prepare_capacity(
		description, kernel, TP_EXCHANGE_MAXIMUM_ELEMENTS, &checked);
	if (status != STATUS_OK)
		return status;
	const struct tp_exchange_arguments arguments = { description->arguments,
							 layer, layer_count,
							 0 };
	prepare_launch_fill(&checked, kernel, 1, 1, ORDERED_COLLECTIVE_THREADS,
			    0, &arguments, sizeof(arguments));
	*launch_out = checked;
	return STATUS_OK;
}

result_status tp_exchange_wide_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t form, uint32_t kernels_out[TP_EXCHANGE_WIDE_LAUNCHES])
{
	/* the copy-engine form runs no PUBLISH: READY stands in its place */
	static const char *const publishes[] = {
		KERNEL_TP_EXCHANGE_PUBLISH, KERNEL_TP_EXCHANGE_OWNER_PUBLISH,
		KERNEL_TP_EXCHANGE_OWNER_RING_PUBLISH, KERNEL_TP_EXCHANGE_READY,
		KERNEL_TP_EXCHANGE_READY
	};
	static const char *const sums[] = {
		KERNEL_TP_EXCHANGE_SUM, KERNEL_TP_EXCHANGE_OWNER_SUM,
		KERNEL_TP_EXCHANGE_OWNER_RING_SUM,
		KERNEL_TP_EXCHANGE_CE_OWNER_SUM,
		KERNEL_TP_EXCHANGE_CE_OWNER_RING_SUM
	};
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL ||
	    form > TP_EXCHANGE_FORM_CE_OWNER_RING_SUMS)
		return STATUS_INVALID_ARGUMENT;
	const char *const names[TP_EXCHANGE_WIDE_LAUNCHES] = {
		publishes[form], KERNEL_TP_EXCHANGE_READY, sums[form],
		KERNEL_TP_EXCHANGE_DONE
	};
	uint32_t kernels[TP_EXCHANGE_WIDE_LAUNCHES];
	result_status status = STATUS_OK;
	for (uint32_t index = 0;
	     index < TP_EXCHANGE_WIDE_LAUNCHES && status == STATUS_OK; index++)
		status = backend->kernel_prepare(card, KERNEL_FILE_TP_EXCHANGE,
						 names[index],
						 TP_EXCHANGE_ARGUMENT_BYTES,
						 &kernels[index]);
	if (status == STATUS_OK)
		for (uint32_t index = 0; index < TP_EXCHANGE_WIDE_LAUNCHES;
		     index++)
			kernels_out[index] = kernels[index];
	return status;
}

result_status tp_exchange_wide_prepare(
	const struct ordered_push_description *description, uint32_t layer,
	uint32_t layer_count, const uint32_t kernels[TP_EXCHANGE_WIDE_LAUNCHES],
	uint32_t form,
	struct kernel_launch launches_out[TP_EXCHANGE_WIDE_LAUNCHES])
{
	static const uint32_t blocks[TP_EXCHANGE_WIDE_LAUNCHES] = {
		TP_EXCHANGE_WIDE_BLOCKS, 1u, TP_EXCHANGE_WIDE_BLOCKS, 1u
	};
	if (kernels == NULL || launches_out == NULL || description == NULL)
		return STATUS_INVALID_ARGUMENT;
	const struct ordered_push_arguments *const arguments =
		&description->arguments;
	/* the owner forms: direct routes, the 16-byte vectors of every array they touch; the ring's four ranks and wire */
	if (form > TP_EXCHANGE_FORM_CE_OWNER_RING_SUMS ||
	    (form == TP_EXCHANGE_FORM_CE_OWNER_SUMS &&
	     (arguments->payload_format != ORDERED_PUSH_BF16 ||
	      arguments->rank_count == 0 ||
	      arguments->element_count %
			      (arguments->rank_count * MODEL_HIDDEN) !=
		      0)) ||
	    ((form == TP_EXCHANGE_FORM_OWNER_RING_SUMS ||
	      form == TP_EXCHANGE_FORM_CE_OWNER_RING_SUMS) &&
	     (arguments->rank_count != TP_EXCHANGE_OWNER_RING_RANKS ||
	      arguments->payload_format != ORDERED_PUSH_BF16)) ||
	    (form != TP_EXCHANGE_FORM_PUSH &&
	     (arguments->incoming_host_mask != 0 ||
	      (arguments->region_offset | arguments->input_offset |
	       arguments->output_offset | arguments->inbox_offset |
	       arguments->inbox_rank_stride | arguments->parity_stride) %
			      TP_EXCHANGE_OWNER_ALIGNMENT !=
		      0)))
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launches[TP_EXCHANGE_WIDE_LAUNCHES];
	result_status status = STATUS_OK;
	for (uint32_t index = 0;
	     index < TP_EXCHANGE_WIDE_LAUNCHES && status == STATUS_OK;
	     index++) {
		status = tp_exchange_prepare(description, layer, layer_count,
					     kernels[index], &launches[index]);
		if (status == STATUS_OK)
			launches[index].grid[0] = blocks[index];
	}
	if (status == STATUS_OK)
		for (uint32_t index = 0; index < TP_EXCHANGE_WIDE_LAUNCHES;
		     index++)
			launches_out[index] = launches[index];
	return status;
}
