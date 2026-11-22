/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_program_internal.h: what the TP program units share (split from model_v41_program.c by whole items).
 */
#ifndef DS41F_TP_PROGRAM_INTERNAL_H
#define DS41F_TP_PROGRAM_INTERNAL_H

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

#define DS41F_TP2_PROGRAM_ROTATION_PAIRS 2u
#define DS41F_TP2_PROGRAM_FRONT_PAIRS 1u
#define DS41F_TP2_PROGRAM_PAIRS                            \
	(DS41F_LAYERS + DS41F_TP2_PROGRAM_ROTATION_PAIRS + \
	 DS41F_TP2_PROGRAM_FRONT_PAIRS)
uint32_t ds41f_tp2_program_layer_card(
	const struct ds41f_tp2_program_description *description,
	uint32_t layer);
bool ds41f_tp2_program_exl3_all(
	const struct ds41f_tp2_program_description *description);
bool ds41f_tp2_program_replicated(
	const struct ds41f_tp2_program_description *description);
bool ds41f_tp2_program_is_valid(
	const struct ds41f_tp2_program_description *description,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *card_descriptions,
	enum program_kind kind, uint32_t rows);
MUST_CHECK result_status ds41f_tp2_program_select(
	struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description,
	struct ds41f_tp2_layer_plan *tensor, uint32_t card, uint32_t layer);
MUST_CHECK result_status ds41f_tp2_program_segment(struct ds41f_builder *build,
						   uint32_t layer,
						   uint32_t first_launch);
MUST_CHECK result_status ds41f_tp2_program_copies(struct ds41f_builder *build,
						  uint32_t layer,
						  uint32_t copied);
MUST_CHECK result_status ds41f_tp2_program_streams(struct ds41f_builder *build,
						   uint32_t destination);
MUST_CHECK result_status ds41f_tp2_program_pair_add(
	struct ds41f_builder *build, struct step_companion_pair *pairs,
	uint32_t *pair_count, uint32_t first_segment);
MUST_CHECK result_status ds41f_tp2_program_rotations(
	struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description,
	struct ds41f_tp2_layer_plan *tensor, struct step_companion_pair *pairs,
	uint32_t *pair_count, uint32_t first_card, uint32_t first_layer,
	bool embeds);
MUST_CHECK result_status ds41f_tp2_program_boundary(
	struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description,
	uint32_t layer);
MUST_CHECK result_status
ds41f_tp2_program_layer_steps(struct ds41f_builder *build, uint32_t layer,
			      uint32_t source, uint32_t group, bool front);
MUST_CHECK result_status ds41f_tp2_program_layer_pair(
	struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description,
	struct ds41f_tp2_layer_plan *tensor, struct step_companion_pair *pairs,
	uint32_t *pair_count, uint32_t layer, uint32_t source, uint32_t group,
	bool front);

#endif
