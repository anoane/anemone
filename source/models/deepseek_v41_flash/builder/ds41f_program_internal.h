/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_internal.h: the builder record every unit fills, shared helpers and prototypes (split from
 * model_v41_program.c by whole items).
 */
#ifndef DS41F_PROGRAM_INTERNAL_H
#define DS41F_PROGRAM_INTERNAL_H

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

#include "ds41f_builder.h"

#define DS41F_ARRAY(name) \
	(b->arrays[DS41F_ARRAY_##name]) /* of the card, for the next launches' rows */
#define DS41F_WEIGHTS(name) \
	(&weights[DS41F_TENSOR_##name]) /* a tensor of the layer */
#define DS41F_ELEMENT_BF16 DENSE_BFLOAT16
#define DS41F_ELEMENT_F32 DENSE_FLOAT32
#define DS41F_TRUNK LINEAR_FAMILY_TRUNK_QUANTIZED

/* An array's bytes a row and besides, and its kind of rows (enum ds41f_rows). */
#define DS41F_BYTES_OF(array, row_bytes, fixed_bytes, holder, rows) \
	[DS41F_ARRAY_##array] = { (uint64_t)(row_bytes),            \
				  (uint64_t)(fixed_bytes), rows },
static const struct ds41f_array_bytes {
	uint64_t row_bytes, fixed_bytes;
	uint32_t rows;
} ds41f_array_bytes[DS41F_ARRAY_COUNT] = { DS41F_ARRAYS(DS41F_BYTES_OF) };

/* Public handle arrays use the record header's slice-stamp order. */
#define LINEAR_SLICE_STAMP(count) EXL3_ATTENTION_SLICE_INSTANCE_##count,
enum exl3_attention_slice_instance {
	LINEAR_EVERY_SLICE_COUNT EXL3_ATTENTION_SLICE_INSTANCE_COUNT
};
#undef LINEAR_SLICE_STAMP
_Static_assert(EXL3_ATTENTION_SLICE_INSTANCE_COUNT == LINEAR_SLICE_INSTANCES,
	       "EXL3 attention uses the public linear-handle ordering");
extern const struct linear_range ds41f_no_range;
bool ds41f_card_has_exl3(const struct ds41f_plan *plan, uint32_t card);
void ds41f_checkpoints_make(const struct ds41f_plan *plan,
			    struct window_checkpoint *ranges);
bool ds41f_plan_is_valid(const struct ds41f_plan *plan);
struct linear_range ds41f_packed_rows(const struct ds41f_builder *b,
				      enum ds41f_array array,
				      uint64_t row_bytes);
void ds41f_span_select(struct ds41f_builder *b, uint32_t span);
void ds41f_draft_slab_select(struct ds41f_builder *b, uint32_t slab);
void ds41f_piece_select(struct ds41f_builder *b, uint32_t piece);
void ds41f_draft_swap(struct ds41f_builder *b, bool begins);
struct kernel_launch *ds41f_slots(struct ds41f_builder *b, uint32_t count);
bool ds41f_trunk_mma_ready(struct ds41f_builder *b,
			   const struct matrix_shape_record *shape);
void ds41f_trunk_mma_swap(struct ds41f_builder *b, struct kernel_launch *launch,
			  const struct matrix_shape_record *shape);
bool ds41f_batch_thin_bits_of(const struct ds41f_builder *b);
bool ds41f_trunk_mma_wide_applies(
	const struct ds41f_builder *b,
	const struct linear_trunk_bfloat16_description *product);
void ds41f_trunk_mma_pair_merge(struct ds41f_builder *b);
MUST_CHECK result_status
ds41f_trunk_mma_wide(struct ds41f_builder *b, struct kernel_launch *launch,
		     const struct linear_trunk_bfloat16_description *product);
uint32_t ds41f_tc_most_rows(const struct ds41f_builder *b);
MUST_CHECK result_status ds41f_trunk_mma_wide_rows(
	struct ds41f_builder *b, struct kernel_launch *launch,
	const struct linear_trunk_bfloat16_description *product);
MUST_CHECK result_status ds41f_put(struct ds41f_builder *b,
				   const struct kernel_launch *launch);
MUST_CHECK result_status ds41f_official_norm(struct ds41f_builder *b,
					     struct linear_range input,
					     struct linear_range weight,
					     struct linear_range output,
					     uint32_t width);
MUST_CHECK result_status ds41f_official_roundtrip(struct ds41f_builder *b,
						  struct linear_range values,
						  uint32_t width);
void ds41f_copies_hand_off(struct ds41f_builder *b, uint32_t layer,
			   uint32_t copied);
MUST_CHECK result_status ds41f_slices_of(
	const struct ds41f_builder *b, enum linear_family_code family,
	const struct matrix_shape_record *shape, uint32_t *slice_count_out);
MUST_CHECK result_status ds41f_matrix_of(const struct ds41f_builder *b,
					 const struct ds41f_tensor *tensor,
					 enum linear_family_code family,
					 struct linear_matrix *matrix_out);
MUST_CHECK result_status ds41f_dense_cone(struct ds41f_builder *b,
					  const struct ds41f_tensor *weights,
					  struct linear_range rows,
					  struct linear_range output,
					  uint32_t output_type, uint32_t flags,
					  float output_scale,
					  uint32_t cone_layer);
MUST_CHECK result_status ds41f_quantize_into(struct ds41f_builder *b,
					     struct linear_range values,
					     uint32_t width,
					     struct linear_range codes,
					     struct linear_range exponents,
					     struct linear_range operands);
MUST_CHECK result_status ds41f_quantize(struct ds41f_builder *b,
					struct linear_range values,
					uint32_t width);
MUST_CHECK result_status ds41f_trunk_cone_at(struct ds41f_builder *b,
					     struct kernel_launch *launch,
					     uint32_t cone_layer);
MUST_CHECK result_status ds41f_trunk_of(
	struct ds41f_builder *b, struct linear_range codes,
	struct linear_range exponents, struct linear_range operands,
	const struct ds41f_tensor *first, struct linear_range first_output,
	const struct ds41f_tensor *second, struct linear_range second_output);
MUST_CHECK result_status ds41f_trunk(struct ds41f_builder *b,
				     const struct ds41f_tensor *first,
				     struct linear_range first_output,
				     const struct ds41f_tensor *second,
				     struct linear_range second_output);
MUST_CHECK result_status ds41f_exl3_trunk_bfloat16(
	struct ds41f_builder *b, const struct ds41f_tensor *matrix,
	struct linear_range values, struct linear_range output);
MUST_CHECK result_status
ds41f_groups_product(struct ds41f_builder *b,
		     const struct linear_trunk_bfloat16_description *groups,
		     int32_t fold_reference);
MUST_CHECK result_status ds41f_site_prepare(struct ds41f_builder *b,
					    const struct ds41f_tensor *weights,
					    bool is_ffn,
					    struct kernel_launch *launches);
MUST_CHECK result_status ds41f_site_begin(struct ds41f_builder *b,
					  const struct ds41f_tensor *weights,
					  bool is_ffn,
					  struct kernel_launch *launches);
MUST_CHECK result_status
ds41f_site_front_of(struct ds41f_builder *b, const struct ds41f_tensor *weights,
		    bool is_ffn, const struct ds41f_router_description *router,
		    enum ds41f_site_front_kind kind);
MUST_CHECK result_status
ds41f_site_front(struct ds41f_builder *b, const struct ds41f_tensor *weights,
		 bool is_ffn, const struct ds41f_router_description *router);
void ds41f_front_branch_mark(struct ds41f_builder *b, uint32_t side);
bool ds41f_trunk_is_relaxed(const struct ds41f_builder *b,
			    const struct ds41f_tensor *tensor);
MUST_CHECK result_status ds41f_tensor_row_copy(struct ds41f_builder *b,
					       struct linear_range source,
					       struct linear_range output,
					       uint32_t width,
					       uint32_t source_stride,
					       uint32_t output_stride);
struct linear_range ds41f_dense_slot(const struct ds41f_builder *b,
				     uint32_t width);
MUST_CHECK result_status ds41f_dense_gather(struct ds41f_builder *b,
					    uint32_t site, uint32_t width,
					    struct linear_range output);
struct linear_range ds41f_dense_own(const struct ds41f_builder *b,
				    uint32_t width);
MUST_CHECK result_status ds41f_dense_gather_slabs(struct ds41f_builder *b,
						  uint32_t site, uint32_t width,
						  struct linear_range output);
bool ds41f_pool_gather_applies(const struct ds41f_builder *b);
MUST_CHECK result_status ds41f_pool_rows_gather(struct ds41f_builder *b,
						uint64_t kv_row_bytes,
						uint32_t first, uint32_t rows,
						bool scores);
MUST_CHECK result_status ds41f_tensor_complete_ce_rings(
	struct ds41f_builder *b, uint32_t site, struct linear_range values,
	struct linear_range routed);
MUST_CHECK result_status ds41f_tensor_complete(struct ds41f_builder *b,
					       uint32_t site,
					       struct linear_range values,
					       struct linear_range routed);
MUST_CHECK result_status ds41f_tensor_complete_owner_rings(
	struct ds41f_builder *b, uint32_t site, struct linear_range values,
	struct linear_range routed);
MUST_CHECK result_status ds41f_tensor_complete_slabs(
	struct ds41f_builder *b, uint32_t site, struct linear_range values,
	struct linear_range routed);
MUST_CHECK result_status ds41f_engram_launches(struct ds41f_builder *b,
					       uint32_t layer, uint32_t steps);
void ds41f_front_end(struct ds41f_builder *b, uint32_t layer,
		     uint32_t *first_launch);
MUST_CHECK result_status ds41f_exact_gather(struct ds41f_builder *b,
					    uint32_t site,
					    struct linear_range input,
					    struct linear_range output,
					    uint32_t bytes, uint32_t vectors);
MUST_CHECK result_status ds41f_exact_gather_bank(
	struct ds41f_builder *b, uint32_t site, uint64_t ring_offset,
	uint64_t bank_bytes, uint64_t row_bytes, struct linear_range input,
	struct linear_range output, uint32_t bytes, uint32_t vectors);
MUST_CHECK result_status ds41f_ffn_post(struct ds41f_builder *b,
					uint32_t layer);
MUST_CHECK result_status ds41f_draft_wide_layer(struct ds41f_builder *b,
						uint32_t layer);
MUST_CHECK result_status ds41f_router_scores(struct ds41f_builder *b,
					     const struct ds41f_tensor *router);
MUST_CHECK result_status ds41f_batch_draft_scratch(struct ds41f_builder *b);
MUST_CHECK result_status ds41f_original_head_walk(struct ds41f_builder *b);
MUST_CHECK result_status ds41f_router_gather(struct ds41f_builder *b,
					     const struct ds41f_tensor *router);

#endif
