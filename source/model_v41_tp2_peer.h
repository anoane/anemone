/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Startup actual-allocation qualification for pair/quad groups; singleton stages retain self-only grants.
 * p2p_v1 all-CMP route qualification is prerequisite; this oracle rechecks the current backend, contexts, protocol
 * and model arenas. CMP/PRO edges are never enabled. PP handoffs stay unchanged. No host fallback or key is read.
 *
 * Call after opening all canonical arenas and before model_load, with no concurrent work/close. One site per group
 * tests1 and row_capacity HIDDEN rows, sequences2..5 and both parities, using changing finite fractional inputs,
 * independent host BF16 wire casts and a +0 then ascending-rank FP32 bit oracle. Wide-capacity shared regions also
 * test increasing first/middle/last layer phases in each model step, a changing active row count, and inactive-output
 * canaries. Every group member submits before any wait. The route planner validates the full row_capacity*HIDDEN
 * count against TP_EXCHANGE_MAXIMUM_ELEMENTS, using only the tested directed capabilities (no measured
 * override or common-root host preference). It must retain every tested direct edge. Generic public route and
 * collective limits remain unchanged.
 *
 * One bounded pinned scratch block relinquishes through staging_retire on every exit; failed wait is not completion.
 * Caller must drain the entire group before any close/reuse. Success leaves disposable test data/flags/STEP;
 * model_load reinitializes arenas/channels before runtime sequence1. A whole-group drain precedes scratch reset.
 * No timings, model quality or capacity are qualified. Output is untouched on failure; success masks include only
 * self and the tested writers in each group. All model metadata/mappings outlive any possible peer reader.
 */
#ifndef MODEL_V41_TP2_PEER_H
#define MODEL_V41_TP2_PEER_H

#include "model_v41_tp2_model.h"

MUST_CHECK result_status v41_tp2_peer_qualify(
	const struct v41_tp2_model *model,
	const struct backend_operations *backend,
	struct device_card *const *cards, uint64_t wait_limit_nanoseconds,
	uint32_t writer_masks_out[V41_DATA_TP2_PROGRAM_CARDS]);

/* Cold teardown barrier, before closing ANY participant. Attempt launch_wait on every non-NULL card in logical
 * order, even after an earlier error. NULL entries are unopened slots. Any failed/unsupported CMP wait retains
 * both members of its pair; an uncertain PRO retains only itself. Caller closes only cards whose bits are clear
 * and retains all model metadata/mappings while any bit remains. A close attempt is not completion proof.
 *
 * On valid arguments, retained_out is always published, including failure, and the first non-OK wait is returned.
 * An absent launch_wait retains every open participant's group and returns UNSUPPORTED (no open cards: OK).
 * NULL backend/cards/output or zero wait is INVALID_ARGUMENT and leaves output untouched. No allocation occurs.
 */
MUST_CHECK result_status
v41_tp2_peer_drain(const struct backend_operations *backend,
		   struct device_card *const cards[V41_DATA_TP2_PROGRAM_CARDS],
		   uint64_t wait_limit_nanoseconds, uint32_t *retained_out);

/* Same cold drain, with a fixed CMP group width2 or4. Any uncertain member retains its complete group. */
MUST_CHECK result_status
v41_tp_group_drain(const struct backend_operations *backend,
		   struct device_card *const cards[V41_DATA_TP2_PROGRAM_CARDS],
		   uint64_t wait_limit_nanoseconds, uint32_t members,
		   uint32_t *retained_out);

#endif
