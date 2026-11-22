/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_model.c: the operations table, the EXL3 descriptor model and the entry (split from model_v41_program.c by
 * whole items).
 */
#include "common/checked_size.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_layer.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_program.h"
#include "models/model_registry.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "models/deepseek_v41_flash/attention/ds41f_attention_core.h"
#include "models/deepseek_v41_flash/attention/ds41f_attention_output.h"
#include "models/deepseek_v41_flash/attention/ds41f_attention_state.h"
#include "engine/backend/fill_records.h"
#include "engine/library/bf16_linear/dense_linear.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "engine/library/bf16_linear/dense_stream_records.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_draft_share_records.h"
#include "models/deepseek_v41_flash/engram/ds41f_engram_rows.h"
#include "engine/backend/copy_records.h"
#include "engine/library/block_scaled_linear/expert_relaxed.h"
#include "engine/library/exl3/exl3.h"
#include "engine/library/exl3/exl3_tc.h"
#include "engine/library/exl3/exl3_tp.h"
#include "engine/library/moe/expert_producer.h"
#include "common/hooks.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_hyper_connection.h"
#include "engine/library/tensor_parallel/peer_row_copy.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"
#include "models/deepseek_v41_flash/vision/ds41f_image_routing.h"
#include "models/deepseek_v41_flash/moe/ds41f_router.h"
#include "engine/library/block_scaled_linear/relaxed_linear.h"
#include "engine/library/sampling/sampler.h"
#include "engine/library/sampling/sampler_shard_records.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_site_front.h"
#include "engine/library/tensor_parallel/tp_join.h"
#include "engine/library/moe/ep_filter.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/wire_codecs/quantized_wire_records.h"
#include "engine/library/tensor_parallel/tp_exchange.h"
#include "engine/library/collectives/ordered_push_parallel.h"
#include "engine/library/prepare_checks.h"

#include "models/deepseek_v41_flash/builder/ds41f_builder.h"

static const struct model_program_operations ds41f_program_operations = {
	DS41F_MODEL_NAME,	ds41f_describe,	       ds41f_state_byte_count,
	ds41f_plan_create,	ds41f_plan_destroy,    ds41f_arena_byte_count,
	ds41f_arena_initialise, ds41f_program_prepare, ds41f_program_destroy
};

/* Its routed experts as EXL3 maps and the tensor-core MoE read them (exl3_descriptor.h), its established packs among
 * them: the registry's entry (model_registry.h) */
static const struct exl3_established_pack v41_established[] =
	DS41F_1017_ESTABLISHED_EXL3_PACKS;
static const struct exl3_descriptor_model v41_exl3 = {
	.model_type = DS41F_CONFIG_MODEL_TYPE,
	.layers = DS41F_LAYERS,
	.experts_per_layer = DS41F_EXPERTS,
	.hidden = DS41F_HIDDEN,
	.intermediate = DS41F_EXPERT_INNER,
	.matrices = { { "w1", DS41F_EXPERT_INNER, DS41F_HIDDEN },
		      { "w2", DS41F_HIDDEN, DS41F_EXPERT_INNER },
		      { "w3", DS41F_EXPERT_INNER, DS41F_HIDDEN } },
	.established = v41_established,
	.established_count =
		sizeof(v41_established) / sizeof(v41_established[0]),
	.routes_per_row = DS41F_EXPERTS_PER_ROW,
	.cone_first_layer = DS41F_DECODER_FIRST_LAYER
};
const struct model_entry ds41f_model_entry = { &ds41f_program_operations,
					       &v41_exl3 };
