/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Component-only candidate: a parallel payload kernel followed on the SAME stream by the one-CTA READY/sum/DONE
 * kernel. It preserves ordered_push's full64 sequence, finite/nonfinite handling, casts, ascending-rank sums and
 * completion/ownership contract. Every payload writer system-fences before its kernel completes; the later kernel
 * publishes READY only after all payload CTAs retire. This is separate publication, with no GEMV overlap credit.
 *
 * Both ranks/groups must queue both launches before any outgoing wait. No other work/flight may reuse their
 * mailboxes until every participant retires. There is no shared-layer phase/active-row adaptation in this candidate:
 * each description names a dedicated per-layer/site channel and exactly the prepared element_count.
 *
 * Kernel preparation loads metadata only and leaves both output handles untouched on failure. Launch preparation
 * inherits ordered_push bounds/disjointness and maximum elements, touches no card, allocates nothing, and leaves
 * both outputs untouched on failure. No numerical, transport, model, capacity or performance gate is implied.
 */
#ifndef ORDERED_PUSH_PARALLEL_H
#define ORDERED_PUSH_PARALLEL_H

#include "anemone/ordered_push.h"

#include "anemone/ordered_push_parallel_records.h"

MUST_CHECK result_status ordered_push_parallel_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t kernels_out[ORDERED_PUSH_PARALLEL_LAUNCHES]);
MUST_CHECK result_status ordered_push_parallel_prepare(
	const struct ordered_push_description *description,
	const uint32_t kernels[ORDERED_PUSH_PARALLEL_LAUNCHES],
	struct kernel_launch launches_out[ORDERED_PUSH_PARALLEL_LAUNCHES]);

#endif
