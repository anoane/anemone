/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_plan_internal.h: what the TP plan units share (split from model_v41.c by whole items).
 */
#ifndef DS41F_TP_PLAN_INTERNAL_H
#define DS41F_TP_PLAN_INTERNAL_H

#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "ds41f_tp_layer.h"
#include "ds41f_tp_stage.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/exl3/exl3.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark_records.h"
#include "ds41f_draft_share_records.h"
#include "engine/library/sampling/sampler_shard_records.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine/backend/fill_records.h"
#include "engine/library/block_scaled_linear/expert_relaxed.h"
#include "common/hooks.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/library/block_scaled_linear/relaxed_linear.h"
#include "engine/library/prepare_checks.h"

#include "models/deepseek_v41_flash/plan/ds41f_plan.h"

/* Load-only layer inventory. Sorting carries the destination descriptor with its original source extent. */
struct ds41f_tp2_pending_tensor {
	struct tp_load_job job;
	struct linear_range *stored;
	uint64_t *expert_offset;
	uint64_t byte_count;
	uint32_t extent;
};
int ds41f_tp2_pending_compare(const void *first, const void *second);
MUST_CHECK result_status ds41f_tp2_tensor_find(
	const struct pack_record *pack, const char *name,
	const struct ds41f_tensor_rule *rule, uint32_t card, uint32_t rank,
	uint32_t rank_count, bool split_outputs, bool split_inputs,
	bool balanced_shards, struct ds41f_tensor *tensor_out,
	struct ds41f_tp2_pending_tensor *pending);

#endif
