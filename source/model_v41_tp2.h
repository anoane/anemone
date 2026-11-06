/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original-weight layer-zero TP2 prototype. The PP model entry points and default programs are unchanged.
 * Plans select one physical card, one or two tensor ranks and a caller-owned arena prefix. All tensor jobs
 * are prepared at load; every expert remains resident. State/KV, router and mHC are replicated. This first
 * layer has no compressed-pool, indexer or Engram dependencies. No full-model placement admission is implied.
 * Program preparation returns ordinary graph-capturable launches; the caller queues both ranks before waiting.
 * Separate attention/FFN mailboxes use the same positive step sequence and independent two-slot generations.
 */
#ifndef MODEL_V41_TP2_H
#define MODEL_V41_TP2_H

#include "model_v41.h"
#include "anemone/ordered_push.h"
#include "anemone/tp_load.h"

#define TP2_RANKS 2u
#define TP4_RANKS 4u
#define TP_LOCAL_VECTORS 2u
#define TP4_INNER_TILE 128u
#define TP4_SMALL_INNER \
	(V41_DATA_EXPERT_INNER / TP4_RANKS / TP4_INNER_TILE * TP4_INNER_TILE)
#define TP4_LARGE_INNER (V41_DATA_EXPERT_INNER / TP2_RANKS - TP4_SMALL_INNER)
#define TP2_SITES 2u
#define TP2_LAYER_JOBS \
	(V41_DATA_LAYER_TENSOR_COUNT + MODEL_EXPERTS * MODEL_EXPERT_MATRICES)
#define TP2_LAYER_LAUNCHES 96u
#define TP2_DEFAULT_SLICES 4u
#define TP2_DEADLINE_CYCLES 5000000000ull

struct bb_exl3_output_shard_plan;

struct v41_tp2_layer_plan {
	struct model_plan_record *model;
	struct tp_load_job jobs[TP2_LAYER_JOBS];
	struct ordered_push_description collectives[TP2_SITES];
	uint64_t peer_regions
		[TP2_SITES]
		[TP4_RANKS]; /* each rank's region of the site's channel */
	struct linear_range compact, exact_hidden, exact_sums;
	uint64_t ring_offsets[TP2_SITES], ring_bank_bytes[TP2_SITES],
		ring_row_bytes[TP2_SITES];
	bool exact_output_shards, original_code_wire, bf16_ring;
	bool original_wide_experts; /* original full-K expert jobs over the chunk, lossless gathers per slab */
	bool exact_dense; /* model_v41_tp2_stage.h: the attention's dense products output-axis shards, two gathers */
	bool exl3_fast_products; /* model_v41_tp2_stage.h: kernel_exl3_wide's EXL3 transforms and products */
	bool exl3_wide_entry; /* model_v41_tp2_stage.h: a chunk's EXL3 experts in one call */
	bool exl3_tc_moe; /* model_v41_tp2_stage.h: GNmac's tensor-core routed experts (exl3_tc.h) */
	uint32_t exl3_expert_rotation; /* model_v41_tp2_stage.h: the routed experts' chunks rotated per expert (1, 2) */
	bool trunk_shape_slices; /* model_v41_tp2_stage.h: thin trunk products at their shape's measured count */
	uint32_t ring_rows; /* model_v41_tp2_stage.h: rows one ring gather carries */
	struct linear_range exl3_tc_workspace, exl3_tcp_workspace;
	struct linear_range
		exl3_tail_workspace; /* model_v41_tp2_stage.h tail_overlay: tail programs' TC MoE */
	struct linear_range
		trunk_mma_partials; /* model_v41_tp2_stage.h: trunk_mma's _64 partials */
	struct linear_range
		trunk_mma_set_partials; /* model_v41_tp2_stage.h tail_sets: the row sets' partials */
	bool tail_sets; /* model_v41_tp2_stage.h: a tail program's trunk products one launch over its 64-row groups */
	struct linear_range
		router_partials; /* model_v41_tp2_stage.h: router_mma's partials, or none */
	bool ordered_wide; /* model_v41_tp2_stage.h: wide BF16-row attention trunk products by the ordered wide form */
	uint32_t graph_branches; /* model_v41_tp2_stage.h: the TC-MoE FFN's shared expert a graph branch */
	bool bf16_slab_rings; /* model_v41_tp2_stage.h: a piece's attention completed a slab at a time by the BF16 ring */
	uint32_t wide_exchange; /* model_v41_tp2_stage.h: a piece's ordered push by tp_exchange.h's wide form */
	uint32_t split_overlap; /* model_v41_tp2_stage.h: a chunked layer in two halves, the sums beside the other half */
	uint32_t split_least_rows; /* model_v41_tp2_stage.h: split_overlap's least chunk rows */
	bool launches_untrimmed; /* model_v41_tp2_stage.h: --tp_launch_trims=0, the launches as before 3 November */
	bool chunk_core; /* model_v41_tp2_stage.h: a chunked step's attention core over the chunk */
	struct linear_range dense_input;
	struct linear_range
		index_scores; /* index_span: a span's scores and annex (model_v41_tp2_stage.h) */
	uint32_t index_rows; /* index_span: the most rows of a scan launch (0: a slab's) */
	uint64_t dense_ring_offsets[TP2_SITES],
		dense_ring_bank_bytes[TP2_SITES];
	struct linear_range stub_values[TP2_SITES];
	uint32_t job_count;
	uint32_t card;
	uint32_t rank;
	uint32_t rank_count;
	uint32_t layer_index;
	bool shared_channels;
	bool parallel_publish; /* component-only candidate, no full-model admission */
	bool attention_replicated; /* one reduction: full attention, TP2 routed/shared experts */
	bool experts_partitioned; /* component-only EP2, full owned experts and TP2 shared expert */
	uint64_t unowned_gate_offset; /* aliases all unowned table entries to one resident dummy expert */
	const struct bb_exl3_output_shard_plan *
		exl3_shard; /* immutable stage-owned target metadata; NULL for native */
};

struct v41_tp2_layer_program {
	struct kernel_launch launches[TP2_LAYER_LAUNCHES];
	uint32_t launch_count;
};

MUST_CHECK result_status v41_tp2_layer_plan_create(
	const struct pack_record *pack, const struct model_limits *limits,
	uint32_t card, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, struct v41_tp2_layer_plan **plan_out);
/* Component candidate: replicated_attention keeps full attention weights/products on both ranks, eliminating
 * attention's rank reduction. Routed/shared experts retain TP2. The extra weight bytes are reserved at creation;
 * no full-model placement or capacity admission is implied. The original create API always selects false. */
MUST_CHECK result_status v41_tp2_layer_variant_create(
	const struct pack_record *pack, const struct model_limits *limits,
	uint32_t card, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, bool replicated_attention,
	struct v41_tp2_layer_plan **plan_out);
/* Component-only EP2 choice: expert e is owned by rank e%2. Each rank loads its192full experts plus one dummy
 * unowned expert, whose offsets alias every unowned table entry. The filter empties those jobs after unchanged
 * routing. Shared expert and attention retain their independent TP2/replicated choices. No serving admission. */
MUST_CHECK result_status v41_tp2_layer_choices_create(
	const struct pack_record *pack, const struct model_limits *limits,
	uint32_t card, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, bool replicated_attention,
	bool partitioned_experts, struct v41_tp2_layer_plan **plan_out);
void v41_tp2_layer_plan_destroy(struct v41_tp2_layer_plan *plan);

/* rank_count=1 prepares the original PP layer. A TP local stub copies each site's saved globally reduced vector
 * from an untimed real operation. The caller pins identical inputs/state and verifies real/stub output identity,
 * so their difference does not silently include changed FFN routing. No result may use an uninitialized stub.
 * slice_count is an explicit TP-only measured candidate, never a change to the PP default table.
 */
MUST_CHECK result_status v41_tp2_layer_program_prepare(
	const struct v41_tp2_layer_plan *plan,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *descriptions, uint32_t rows,
	uint32_t slice_count, bool local_stub,
	struct v41_tp2_layer_program *program_out);

#endif
