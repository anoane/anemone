/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_layer_program.c: the TP2 layer program (split from model_v41_program.c by whole items).
 */
#include "common/checked_size.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "ds41f_tp_layer.h"
#include "ds41f_tp_program.h"
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
#include "ds41f_draft_share_records.h"
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

result_status ds41f_tp2_layer_program_prepare(
	const struct ds41f_tp2_layer_plan *plan,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *descriptions, uint32_t rows,
	uint32_t slice_count, bool local_stub,
	struct ds41f_tp2_layer_program *program_out)
{
	if (plan == NULL || plan->model == NULL || backend == NULL ||
	    cards == NULL || descriptions == NULL || program_out == NULL ||
	    plan->card >= plan->model->card_count ||
	    cards[plan->card] == NULL ||
	    descriptions[plan->card].arithmetic_class !=
		    ARITHMETIC_CLASS_SM_80 ||
	    rows == 0 || rows > plan->model->limits.row_capacity ||
	    rows > LINEAR_MAXIMUM_ROWS || slice_count == 0 ||
	    slice_count > LINEAR_MAXIMUM_SLICES ||
	    (slice_count & (slice_count - 1u)) != 0 ||
	    (plan->rank_count != 1u && plan->rank_count != DS41F_TP2_RANKS) ||
	    plan->rank >= plan->rank_count)
		return STATUS_INVALID_ARGUMENT;
	struct ds41f_builder *const build = calloc(1, sizeof(*build));
	struct ds41f_made_program *const made =
		calloc(1, sizeof(*made) + (size_t)DS41F_TP2_LAYER_LAUNCHES *
						  sizeof(struct kernel_launch));
	if (build == NULL || made == NULL) {
		free(build);
		free(made);
		return STATUS_OUT_OF_MEMORY;
	}
	*build = (struct ds41f_builder){
		.tensor_layer = plan->rank_count == DS41F_TP2_RANKS ? plan :
								      NULL,
		.tensor_slices = slice_count,
		.tensor_local_stub = local_stub,
		.plan = plan->model,
		.backend = backend,
		.cards = cards,
		.descriptions = descriptions,
		.made = made,
		.rows = rows,
		.chunk_rows = rows,
		.program_rows = rows,
		.slab_count = 1,
		.card = plan->card,
		.head_stream_kernel = BACKEND_NO_KERNEL,
		.head_mma_kernel = BACKEND_NO_KERNEL
	};
	build->launches[plan->card] = made->launches;
	build->launch_capacity[plan->card] = DS41F_TP2_LAYER_LAUNCHES;
	made->program.kind = PROGRAM_THIN;
	result_status status = ds41f_card_select(build, plan->card);
	if (status == DS41F_OK && build->tensor_layer != NULL) {
		status = tp2_join_kernel_prepare(backend, cards[plan->card],
						 &build->tensor_join_kernel);
		status =
			plan->parallel_publish ?
				DS41F_THEN(ordered_push_parallel_kernels_prepare(
					backend, cards[plan->card],
					build->tensor_parallel_kernels)) :
				DS41F_THEN(ordered_push_kernel_prepare(
					backend, cards[plan->card],
					&build->tensor_push_kernel));
	}
	if (plan->experts_partitioned)
		status = DS41F_THEN(ep_filter_kernel_prepare(
			backend, cards[plan->card],
			&build->tensor_ep_filter_kernel));
	status = DS41F_THEN(ds41f_step_launches(build, false));
	const struct ds41f_tp2_layer_plan *const expert_shards =
		build->tensor_layer;
	if (plan->attention_replicated)
		build->tensor_layer = NULL;
	status = DS41F_THEN(ds41f_layer_steps(
		build, 0, 0, 0, DS41F_FRONT_STEPS | DS41F_REST_STEPS));
	build->tensor_layer = expert_shards;
	uint32_t first_launch = 0;
	status = DS41F_THEN(ds41f_ffn_launches(build, 0, &first_launch));
	if (status == DS41F_OK) {
		struct ds41f_tp2_layer_program result = { 0 };
		result.launch_count = made->cards[plan->card].launch_count;
		memcpy(result.launches, made->launches,
		       (size_t)result.launch_count *
			       sizeof(result.launches[0]));
		*program_out = result;
	}
	free(made);
	free(build);
	return status;
}
