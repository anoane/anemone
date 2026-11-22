/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tensors.h: the tensor rules the plan and the TP plans read (split from model_v41.c by whole items).
 */
#ifndef DS41F_TENSORS_H
#define DS41F_TENSORS_H

#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_layer.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_stage.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/exl3/exl3.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark_records.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_draft_share_records.h"
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

extern const struct ds41f_tensor_rule
	ds41f_layer_rules[DS41F_TENSOR_LAYER_COUNT];
extern const struct ds41f_tensor_rule
	ds41f_expert_rules[DS41F_TENSOR_EXPERT_COUNT];
extern const struct ds41f_tensor_rule ds41f_global_rules[DS41F_TENSOR_GLOBALS];
extern const struct ds41f_tensor_rule
	ds41f_draft_router_rules[DS41F_TENSOR_ROUTER_BIAS -
				 DS41F_TENSOR_ROUTER + 1u];
extern const struct ds41f_tensor_rule
	ds41f_draft_rules[DS41F_TENSOR_DRAFT_GLOBALS];

#endif
