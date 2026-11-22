/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_plan.h: the plan and card records and their calls (split from model_v41.c by whole items).
 */
#ifndef DS41F_PLAN_H
#define DS41F_PLAN_H

#include "model_v41.h"
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

#define DS41F_NAME_CAPACITY \
	64 /* "layers.<l>.ffn.experts.<e>.w1.weight" and the longest, with NUL */
#define DS41F_LAYER_PREFIX_CAPACITY \
	20 /* "layers.<l>." of any 32-bit l, with NUL */
#define DS41F_MATRIX_DIMENSIONS 2u
#define DS41F_BIT(index) (1ull << (index))

/* A tensor as the programs read it: element type, outputs and inputs (0 for a vector), format and its bits, the
 * layers that have it. The shapes are REF's and CFG's (ds41f_geometry.h); the formats the converter's. */
struct ds41f_tensor_rule {
	const char *name;
	uint32_t element_type;
	uint32_t dimensions[DS41F_MATRIX_DIMENSIONS];
	uint32_t weight_format;
	uint32_t bits_per_weight;
	uint64_t layers;
};
_Static_assert(
	DS41F_WALK_LAYERS == DS41F_DRAFT_FIRST_LAYER + DS41F_DRAFT_STAGES &&
		DS41F_DRAFT_LAYERS ==
			DS41F_BIT(DS41F_WALK_LAYERS) -
				DS41F_BIT(DS41F_DRAFT_FIRST_LAYER) &&
		DS41F_DRAFT_TAP_LAYERS ==
			DS41F_BIT(DS41F_LAYERS) -
				DS41F_BIT(DS41F_DRAFT_FIRST_TAP_LAYER) &&
		DS41F_EMBEDDING_COPY_PIECE_BYTES % PACK_DIGEST_WORD_BYTES == 0,
	"the draft's stages follow the 40 layers, its taps are their last; the copy moves whole digest words");
uint64_t ds41f_sum(uint64_t first, uint64_t second);
uint64_t ds41f_product(uint64_t first, uint64_t second);
uint32_t ds41f_latest_of(uint64_t mask, uint32_t layer);
MUST_CHECK result_status ds41f_model_check(const struct pack_record *pack);
MUST_CHECK result_status ds41f_limits_check(const struct program_limits *limits);
uint64_t ds41f_image_offset_of(const struct ds41f_card *card);
uint64_t ds41f_take(uint64_t *next, uint64_t byte_count);
bool ds41f_card_lay_out_attention(struct ds41f_plan *plan,
				  struct ds41f_card *card,
				  uint64_t attention_layers);
bool ds41f_card_lay_out(struct ds41f_plan *plan, struct ds41f_card *card);
void ds41f_layer_prefix_of(uint32_t layer,
			   char prefix[DS41F_LAYER_PREFIX_CAPACITY]);

#endif
