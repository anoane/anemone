/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Cold validator shared with TP's explicitly larger kernel. Caller supplies that kernel's element limit.
 * The generic public entry point retains its existing maximum. All count, alignment, alias and extent checks
 * are identical; no launch is published on refusal. This does not qualify a kernel or a remote allocation.
 */
#ifndef ORDERED_PUSH_INTERNAL_H
#define ORDERED_PUSH_INTERNAL_H

#include "anemone/ordered_push.h"

MUST_CHECK result_status ordered_push_prepare_capacity(
	const struct ordered_push_description *description, uint32_t kernel,
	uint32_t maximum_elements, struct kernel_launch *launch_out);

/* Route the full actual count using only the caller-qualified directed capability masks. Like the public
 * push planner's zero-row policy, this ignores common_numa_root and does not apply the read collective's
 * historical host preference or measured overrides. Only ALL_REDUCE without paired scatter/gather is valid.
 * The caller supplies its explicit kernel bound; valid bounds are positive and no greater than UINT32_MAX
 * minus the collective thread count. Output may alias input masks and remains untouched on refusal.
 * This pure metadata operation does not qualify a transfer, allocation or kernel. Public limits stay fixed.
 */
MUST_CHECK result_status ordered_push_routes_prepare_capacity(
	const struct ordered_collective_route_description *description,
	uint32_t maximum_elements, uint32_t *host_writer_masks_out);

#endif
