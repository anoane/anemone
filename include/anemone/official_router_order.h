/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent post-selection normalization for the pinned original M1 call.
 * Run after the ordinary select, before any expert gate/up reads its jobs.
 * The select's six ascending expert jobs and finite affinities/bias are inputs.
 * Only route_weight[0] changes: weight offsets, pair indices, job ordering and
 * every other byte remain unchanged. A padded width32 unstable bitonic order
 * reproduces the qualified source topk order before its six-value sum.
 * There is no global scratch or optional diagnostic write. Unsupported shapes
 * fail at prepare; a malformed device job causes no partial weight update.
 * A refusal preserves launch_out. This pass does not choose expert membership.
 */
#ifndef OFFICIAL_ROUTER_ORDER_H
#define OFFICIAL_ROUTER_ORDER_H
#include "anemone/router.h"

#include "anemone/official_router_order_records.h"

MUST_CHECK result_status official_router_order_kernel_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t *kernel_out);
MUST_CHECK result_status official_router_order_prepare(
	const struct router_description *description, uint64_t arena_byte_count,
	uint32_t kernel, struct kernel_launch *launch_out);
#endif
