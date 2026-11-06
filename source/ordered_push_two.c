/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The separate three-phase owner-slice operation reuses the checked one-shot geometry and argument ABI. */
#include "anemone/ordered_push_two.h"

#include <stddef.h>

result_status
ordered_push_two_kernel_prepare(const struct backend_operations *backend,
				struct device_card *card, uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	return backend->kernel_prepare(card, KERNEL_FILE_ORDERED_PUSH_TWO,
				       KERNEL_ORDERED_PUSH_TWO,
				       ORDERED_PUSH_ARGUMENT_BYTES, kernel_out);
}

result_status
ordered_push_two_prepare(const struct ordered_push_description *description,
			 uint32_t kernel, struct kernel_launch *launch_out)
{
	if (description == NULL)
		return STATUS_INVALID_ARGUMENT;
	const struct ordered_push_arguments *const arguments =
		&description->arguments;
	const uint64_t signal_bytes = (uint64_t)ORDERED_PUSH_TWO_PHASES *
				      arguments->rank_count *
				      ORDERED_COLLECTIVE_LINE_BYTES;
	if (arguments->input_offset < signal_bytes ||
	    arguments->output_offset < signal_bytes ||
	    arguments->inbox_offset < signal_bytes)
		return STATUS_INVALID_ARGUMENT;
	return ordered_push_prepare(description, kernel, launch_out);
}
