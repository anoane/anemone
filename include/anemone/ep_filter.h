/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Component-only, thin EP2 filter; no model/serving admission. All 384 experts retain the global router IDs.
 * Unowned table entries alias one resident dummy expert whose gate offset cannot name an owned expert.
 * The caller proves that ownership/table invariant at plan creation. This filter never changes routing.
 *
 * Description: 1..8 rows, six picks per row, job_count==pair_count==6*rows; hidden/output are BF16 full2304/5120
 * pair arrays. The three touched ranges are aligned, bounded, disjoint; the dummy offset is aligned inside the
 * arena. Preparation allocates nothing, submits no work and leaves outputs untouched on refusal. Launch in the
 * same stream between router and full-width routed experts, with no simultaneous reuse of these arrays.
 */
#ifndef EP_FILTER_H
#define EP_FILTER_H
#include "anemone/ep_filter_records.h"
#include "anemone/linear.h"

struct ep_filter_description {
	struct linear_range jobs;
	struct linear_range hidden;
	struct linear_range output;
	uint64_t unowned_gate_offset;
	uint32_t row_count;
};
MUST_CHECK result_status
ep_filter_kernel_prepare(const struct backend_operations *backend,
			 struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
ep_filter_prepare(const struct ep_filter_description *description,
		  uint64_t arena_byte_count, uint32_t kernel,
		  struct kernel_launch *launch_out);
#endif
