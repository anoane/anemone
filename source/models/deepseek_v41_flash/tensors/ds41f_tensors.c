/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tensors.c: the tensor rules: layer, expert, global, draft router, draft (split from model_v41.c by whole
 * items).
 */
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
#include "ds41f_tensors.h"

#define EVERY 0xFFFFFFFFFFull /* bit l for each of the 40 layers */
#define WALKED (EVERY | DS41F_DRAFT_LAYERS) /* and the draft's stages */

enum {
	BF16 = PACK_ELEMENT_BFLOAT16,
	F32 = PACK_ELEMENT_FLOAT32,
	U8 = PACK_ELEMENT_UINT8
};
enum {
	PLAIN = WEIGHT_FORMAT_NONE,
	E4M3 = WEIGHT_FORMAT_E4M3_SCALED,
	E2M1 = WEIGHT_FORMAT_E2M1_SCALED
};
const struct ds41f_tensor_rule ds41f_layer_rules[DS41F_TENSOR_LAYER_COUNT] = {
	[DS41F_TENSOR_ATTENTION_NORM] = { "attn_norm.weight",
					  BF16,
					  { 5120, 0 },
					  PLAIN,
					  0,
					  WALKED },
	[DS41F_TENSOR_FFN_NORM] = { "ffn_norm.weight",
				    BF16,
				    { 5120, 0 },
				    PLAIN,
				    0,
				    WALKED },
	[DS41F_TENSOR_ATTENTION_MIX] = { "hc_attn_fn",
					 F32,
					 { 24, 20480 },
					 PLAIN,
					 0,
					 WALKED },
	[DS41F_TENSOR_ATTENTION_MIX_BASE] = { "hc_attn_base",
					      F32,
					      { 24, 0 },
					      PLAIN,
					      0,
					      WALKED },
	[DS41F_TENSOR_ATTENTION_MIX_SCALE] = { "hc_attn_scale",
					       F32,
					       { 3, 0 },
					       PLAIN,
					       0,
					       WALKED },
	[DS41F_TENSOR_FFN_MIX] = { "hc_ffn_fn",
				   F32,
				   { 24, 20480 },
				   PLAIN,
				   0,
				   WALKED },
	[DS41F_TENSOR_FFN_MIX_BASE] = { "hc_ffn_base",
					F32,
					{ 24, 0 },
					PLAIN,
					0,
					WALKED },
	[DS41F_TENSOR_FFN_MIX_SCALE] = { "hc_ffn_scale",
					 F32,
					 { 3, 0 },
					 PLAIN,
					 0,
					 WALKED },
	[DS41F_TENSOR_SINK] = { "attn.attn_sink",
				F32,
				{ 64, 0 },
				PLAIN,
				0,
				WALKED },
	[DS41F_TENSOR_QUERY_DOWN] = { "attn.wq_a.weight",
				      U8,
				      { 1280, 5120 },
				      E4M3,
				      8,
				      WALKED },
	[DS41F_TENSOR_QUERY_NORM] = { "attn.q_norm.weight",
				      BF16,
				      { 1280, 0 },
				      PLAIN,
				      0,
				      WALKED },
	[DS41F_TENSOR_QUERY_UP] = { "attn.wq_b.weight",
				    U8,
				    { 32768, 1280 },
				    E4M3,
				    8,
				    WALKED },
	[DS41F_TENSOR_WINDOW] = { "attn.wkv.weight",
				  U8,
				  { 512, 5120 },
				  E4M3,
				  8,
				  WALKED },
	[DS41F_TENSOR_WINDOW_NORM] = { "attn.kv_norm.weight",
				       BF16,
				       { 512, 0 },
				       PLAIN,
				       0,
				       WALKED },
	[DS41F_TENSOR_OUTPUT_GROUPS] = { "attn.wo_a.weight",
					 U8,
					 { 8192, 4096 },
					 E4M3,
					 8,
					 WALKED },
	[DS41F_TENSOR_OUTPUT] = { "attn.wo_b.weight",
				  U8,
				  { 5120, 8192 },
				  E4M3,
				  8,
				  WALKED },
	[DS41F_TENSOR_ROUTER] = { "ffn.gate.weight",
				  BF16,
				  { 384, 5120 },
				  PLAIN,
				  0,
				  EVERY },
	[DS41F_TENSOR_ROUTER_BIAS] = { "ffn.gate.bias",
				       F32,
				       { 384, 0 },
				       PLAIN,
				       0,
				       EVERY },
	[DS41F_TENSOR_SHARED_GATE] = { "ffn.shared_experts.w1.weight",
				       U8,
				       { 2304, 5120 },
				       E4M3,
				       8,
				       WALKED },
	[DS41F_TENSOR_SHARED_UP] = { "ffn.shared_experts.w3.weight",
				     U8,
				     { 2304, 5120 },
				     E4M3,
				     8,
				     WALKED },
	[DS41F_TENSOR_SHARED_DOWN] = { "ffn.shared_experts.w2.weight",
				       U8,
				       { 5120, 2304 },
				       E4M3,
				       8,
				       WALKED },
	[DS41F_TENSOR_COMPRESSOR] = { "attn.compressor.wkv.weight",
				      BF16,
				      { 512, 5120 },
				      PLAIN,
				      0,
				      DS41F_SOURCE_LAYERS },
	[DS41F_TENSOR_COMPRESSOR_GATE] = { "attn.compressor.wgate.weight",
					   BF16,
					   { 512, 5120 },
					   PLAIN,
					   0,
					   DS41F_POOLED_LAYERS },
	[DS41F_TENSOR_COMPRESSOR_NORM] = { "attn.compressor.norm.weight",
					   BF16,
					   { 512, 0 },
					   PLAIN,
					   0,
					   DS41F_SOURCE_LAYERS },
	[DS41F_TENSOR_KEY] = { "attn.indexer.wk.weight",
			       BF16,
			       { 128, 512 },
			       PLAIN,
			       0,
			       DS41F_SOURCE_LAYERS },
	[DS41F_TENSOR_KEY_NORM] = { "attn.indexer.k_norm.weight",
				    BF16,
				    { 128, 0 },
				    PLAIN,
				    0,
				    DS41F_SOURCE_LAYERS },
	[DS41F_TENSOR_INDEX_QUERY_UP] = { "attn.indexer.wq_b.weight",
					  U8,
					  { 4096, 1280 },
					  E4M3,
					  8,
					  DS41F_INDEX_LAYERS },
	[DS41F_TENSOR_INDEX_WEIGHTS] = { "attn.indexer.weights_proj.weight",
					 BF16,
					 { 32, 5120 },
					 PLAIN,
					 0,
					 DS41F_INDEX_LAYERS },
	[DS41F_TENSOR_ENGRAM] = { "engram.wkv.weight",
				  U8,
				  { 25600, 6144 },
				  E4M3,
				  8,
				  DS41F_ENGRAM_LAYERS },
	[DS41F_TENSOR_ENGRAM_QUERY] = { "engram.q_weight",
					BF16,
					{ 4, 5120 },
					PLAIN,
					0,
					DS41F_ENGRAM_LAYERS },
	[DS41F_TENSOR_ENGRAM_KEY] = { "engram.k_weight",
				      BF16,
				      { 4, 5120 },
				      PLAIN,
				      0,
				      DS41F_ENGRAM_LAYERS },
};
const struct ds41f_tensor_rule ds41f_expert_rules[DS41F_TENSOR_EXPERT_COUNT] = {
	[DS41F_TENSOR_GATE] = { "w1.weight", U8, { 2304, 5120 }, E2M1, 4, EVERY },
	[DS41F_TENSOR_UP] = { "w3.weight", U8, { 2304, 5120 }, E2M1, 4, EVERY },
	[DS41F_TENSOR_DOWN] = { "w2.weight", U8, { 5120, 2304 }, E2M1, 4, EVERY },
};
const struct ds41f_tensor_rule ds41f_global_rules[DS41F_TENSOR_GLOBALS] = {
	[DS41F_TENSOR_EMBEDDING] = { "embed.weight",
				     BF16,
				     { 129280, 5120 },
				     PLAIN,
				     0,
				     EVERY },
	[DS41F_TENSOR_NORM] = { "norm.weight",
				BF16,
				{ 5120, 0 },
				PLAIN,
				0,
				EVERY },
	[DS41F_TENSOR_HEAD] = { "head.weight",
				BF16,
				{ 129280, 5120 },
				PLAIN,
				0,
				EVERY },
};
/* A draft stage's router in place of a layer's (CFG:11; its bias_vl is not read: drafts are text, REF:1129), and the
 * draft's globals (dspark_v1/MAP_A_draft_model.md 2.1). */
const struct ds41f_tensor_rule ds41f_draft_router_rules[DS41F_TENSOR_ROUTER_BIAS -
							DS41F_TENSOR_ROUTER +
							1u] = {
	{ "ffn.gate.weight", BF16, { 128, 5120 }, PLAIN, 0, DS41F_DRAFT_LAYERS },
	{ "ffn.gate.bias", F32, { 128, 0 }, PLAIN, 0, DS41F_DRAFT_LAYERS },
};
const struct ds41f_tensor_rule ds41f_draft_rules[DS41F_TENSOR_DRAFT_GLOBALS] = {
	[DS41F_TENSOR_DRAFT_MAIN_PROJECTION] = { "mtp.0.main_proj.weight",
						 U8,
						 { 5120, 15360 },
						 E4M3,
						 8,
						 EVERY },
	[DS41F_TENSOR_DRAFT_MAIN_NORM] = { "mtp.0.main_norm.weight",
					   BF16,
					   { 5120, 0 },
					   PLAIN,
					   0,
					   EVERY },
	[DS41F_TENSOR_DRAFT_NORM] = { "mtp.2.norm.weight",
				      BF16,
				      { 5120, 0 },
				      PLAIN,
				      0,
				      EVERY },
	[DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING] = { "mtp.2.markov_head.embed.weight",
						  BF16,
						  { 129280, 256 },
						  PLAIN,
						  0,
						  EVERY },
	[DS41F_TENSOR_DRAFT_MARKOV_HEAD] = { "mtp.2.markov_head.head.weight",
					     BF16,
					     { 129280, 256 },
					     PLAIN,
					     0,
					     EVERY },
	[DS41F_TENSOR_DRAFT_CONFIDENCE] = { "mtp.2.confidence_head.proj.weight",
					    BF16,
					    { 1, 5376 },
					    PLAIN,
					    0,
					    EVERY },
};
_Static_assert(
	DS41F_ARRAY_FREQUENCIES == 0 && DS41F_TENSOR_GATE == 0 &&
		DS41F_TENSOR_UP == 1 &&
		DS41F_TENSOR_EXPERT_COUNT == DS41F_EXPERT_MATRICES &&
		EVERY == DS41F_BIT(DS41F_LAYERS) - 1u &&
		DS41F_ARRAY_ALIGNMENT_BYTES % PACK_DIGEST_WORD_BYTES == 0 &&
		PACK_DIGEST_WORD_BYTES % FILL_WORD_BYTES == 0,
	"the image is the expert tables, then the frequency words, whole words of a digest and of a fill; a "
	"table's row is gate, up, down");
