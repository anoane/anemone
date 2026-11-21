/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original-weight layer-zero TP2 prototype. The PP model entry points and default programs are unchanged.
 * Plans select one physical card, one or two tensor ranks and a caller-owned arena prefix. All tensor jobs
 * are prepared at load; every expert remains resident. State/KV, router and mHC are replicated. This first
 * layer has no compressed-pool, indexer or Engram dependencies. No full-model placement admission is implied.
 * Program preparation returns ordinary graph-capturable launches; the caller queues both ranks before waiting.
 * Separate attention/FFN mailboxes use the same positive step sequence and independent two-slot generations.
 */
#ifndef DS41F_TP_LAYER_H
#define DS41F_TP_LAYER_H

#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "engine/library/collectives/ordered_push.h"
#include "engine/storage/load/tp_load.h"

#define DS41F_TP2_RANKS 2u
#define DS41F_TP4_RANKS 4u
#define DS41F_TP_LOCAL_VECTORS 2u
#define DS41F_TP4_INNER_TILE 128u
#define DS41F_TP4_SMALL_INNER                                          \
	(DS41F_EXPERT_INNER / DS41F_TP4_RANKS / DS41F_TP4_INNER_TILE * \
	 DS41F_TP4_INNER_TILE)
#define DS41F_TP4_LARGE_INNER \
	(DS41F_EXPERT_INNER / DS41F_TP2_RANKS - DS41F_TP4_SMALL_INNER)
#define DS41F_TP2_SITES 2u
#define DS41F_TP2_LAYER_JOBS \
	(DS41F_TENSOR_LAYER_COUNT + DS41F_EXPERTS * DS41F_EXPERT_MATRICES)
#define DS41F_TP2_LAYER_LAUNCHES 96u
#define DS41F_TP2_DEFAULT_SLICES 4u
#define DS41F_TP2_DEADLINE_CYCLES 5000000000ull

struct bb_exl3_output_shard_plan;

struct ds41f_tp2_layer_plan {
	struct ds41f_plan *model;
	struct tp_load_job jobs[DS41F_TP2_LAYER_JOBS];
	struct ordered_push_description collectives[DS41F_TP2_SITES];
	uint64_t peer_regions
		[DS41F_TP2_SITES]
		[DS41F_TP4_RANKS]; /* each rank's region of the site's channel */
	struct linear_range compact, exact_hidden, exact_sums;
	uint64_t ring_offsets[DS41F_TP2_SITES],
		ring_bank_bytes[DS41F_TP2_SITES],
		ring_row_bytes[DS41F_TP2_SITES];
	bool exact_output_shards, original_code_wire, bf16_ring;
	bool original_wide_experts; /* original full-K expert jobs over the chunk, lossless gathers per slab */
	bool exact_dense; /* ds41f_tp_stage.h: the attention's dense products output-axis shards, two gathers */
	bool exl3_fast_products; /* ds41f_tp_stage.h: kernel_exl3_wide's EXL3 transforms and products */
	bool exl3_wide_entry; /* ds41f_tp_stage.h: a chunk's EXL3 experts in one call */
	bool exl3_tc_moe; /* ds41f_tp_stage.h: GNmac's tensor-core routed experts (exl3_tc.h) */
	uint32_t exl3_expert_rotation; /* ds41f_tp_stage.h: the routed experts' chunks rotated per expert (1, 2) */
	bool trunk_shape_slices; /* ds41f_tp_stage.h: thin trunk products at their shape's measured count */
	uint32_t ring_rows; /* ds41f_tp_stage.h: rows one ring gather carries */
	struct linear_range exl3_tc_workspace, exl3_tcp_workspace;
	struct linear_range
		exl3_tail_workspace; /* ds41f_tp_stage.h tail_overlay: tail programs' TC MoE */
	struct linear_range
		trunk_mma_partials; /* ds41f_tp_stage.h: trunk_mma's _64 partials */
	struct linear_range
		trunk_mma_set_partials; /* ds41f_tp_stage.h tail_sets: the row sets' partials */
	bool tail_sets; /* ds41f_tp_stage.h: a tail program's trunk products one launch over its 64-row groups */
	struct linear_range
		router_partials; /* ds41f_tp_stage.h: router_mma's partials, or none */
	bool ordered_wide; /* ds41f_tp_stage.h: wide BF16-row attention trunk products by the ordered wide form */
	uint32_t graph_branches; /* ds41f_tp_stage.h: the TC-MoE FFN's shared expert a graph branch */
	bool bf16_slab_rings; /* ds41f_tp_stage.h: a piece's attention completed a slab at a time by the BF16
				 ring */
	uint32_t wide_exchange; /* ds41f_tp_stage.h: a piece's ordered push by tp_exchange.h's wide form */
	uint32_t split_overlap; /* ds41f_tp_stage.h: a chunked layer in two halves, the sums beside the other
				   half */
	uint32_t split_least_rows; /* ds41f_tp_stage.h: split_overlap's least chunk rows */
	bool launches_untrimmed; /* ds41f_tp_stage.h: --tp_launch_trims=0, the launches as before 3 November */
	bool chunk_core; /* ds41f_tp_stage.h: a chunked step's attention core over the chunk */
	struct linear_range dense_input;
	struct linear_range
		index_scores; /* index_span: a span's scores and annex (ds41f_tp_stage.h) */
	uint32_t index_rows; /* index_span: the most rows of a scan launch (0: a slab's) */
	uint64_t dense_ring_offsets[DS41F_TP2_SITES],
		dense_ring_bank_bytes[DS41F_TP2_SITES];
	struct linear_range stub_values[DS41F_TP2_SITES];
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

struct ds41f_tp2_layer_program {
	struct kernel_launch launches[DS41F_TP2_LAYER_LAUNCHES];
	uint32_t launch_count;
};

MUST_CHECK result_status ds41f_tp2_layer_plan_create(
	const struct pack_record *pack, const struct program_limits *limits,
	uint32_t card, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, struct ds41f_tp2_layer_plan **plan_out);
/* Component candidate: replicated_attention keeps full attention weights/products on both ranks, eliminating
 * attention's rank reduction. Routed/shared experts retain TP2. The extra weight bytes are reserved at creation;
 * no full-model placement or capacity admission is implied. The original create API always selects false. */
MUST_CHECK result_status ds41f_tp2_layer_variant_create(
	const struct pack_record *pack, const struct program_limits *limits,
	uint32_t card, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, bool replicated_attention,
	struct ds41f_tp2_layer_plan **plan_out);
/* Component-only EP2 choice: expert e is owned by rank e%2. Each rank loads its192full experts plus one dummy
 * unowned expert, whose offsets alias every unowned table entry. The filter empties those jobs after unchanged
 * routing. Shared expert and attention retain their independent TP2/replicated choices. No serving admission. */
MUST_CHECK result_status ds41f_tp2_layer_choices_create(
	const struct pack_record *pack, const struct program_limits *limits,
	uint32_t card, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, bool replicated_attention,
	bool partitioned_experts, struct ds41f_tp2_layer_plan **plan_out);
void ds41f_tp2_layer_plan_destroy(struct ds41f_tp2_layer_plan *plan);

/* rank_count=1 prepares the original PP layer. A TP local stub copies each site's saved globally reduced vector
 * from an untimed real operation. The caller pins identical inputs/state and verifies real/stub output identity,
 * so their difference does not silently include changed FFN routing. No result may use an uninitialized stub.
 * slice_count is an explicit TP-only measured candidate, never a change to the PP default table.
 */
MUST_CHECK result_status ds41f_tp2_layer_program_prepare(
	const struct ds41f_tp2_layer_plan *plan,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *descriptions, uint32_t rows,
	uint32_t slice_count, bool local_stub,
	struct ds41f_tp2_layer_program *program_out);

#endif
