/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Component-only EP2 job filter. The launch is ordered after the ordinary router and before expert products.
 * Each job whose gate offset equals unowned_gate_offset is made empty. All pair hidden/output BF16 words are
 * cleared, so the omitted experts contribute +0 and subsequent quantization cannot read stale hidden data.
 * Owned jobs retain their original offsets, pair indices, route weights and row count. No rank exchange occurs.
 */
#ifndef EP_FILTER_RECORDS_H
#define EP_FILTER_RECORDS_H

#include <stdint.h>
#include "anemone/linear_records.h"

#define KERNEL_FILE_EP_FILTER "kernel_ep_filter"
#define KERNEL_EP_FILTER "ep_filter"
#define EP_FILTER_ARGUMENT_BYTES 48u
#define EP_FILTER_THREADS 256u
#define EP_FILTER_BLOCKS 16u
#define EP_FILTER_HIDDEN_WIDTH 2304u

struct ep_filter_arguments {
	uint64_t jobs_offset;
	uint64_t hidden_offset;
	uint64_t output_offset;
	uint64_t unowned_gate_offset;
	uint32_t job_count;
	uint32_t pair_count;
	uint32_t hidden_width;
	uint32_t output_width;
};
LINEAR_RECORD_HAS(ep_filter_arguments, EP_FILTER_ARGUMENT_BYTES);
#endif
