/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Private cold/load owner for the experimental text TP/PP layouts. Exact PP entry points are unchanged.
 * Layouts are pairs14/15, quad29, or pair14/single7/single8; native PRO4 retains L29..39/head/DSpark.
 * Limits: context>=16384, state16, reference CMP products, draft5..7, rows<=8 or64..1024, serial flights.
 * Exact quad uses full-K output-axis products and lossless owner gathers. Other TP rank sums are relaxed;
 * BF16 quad uses rotating-order BF16 ring sums at thin sites and the legacy ordered wire at wide sites.
 * Original code/scale transport changes no quantizer result, is optional, and is default off in the CLI.
 * attention_owner is a default-off cold API control for exact quad only. Whole source/index groups own their
 * attention state and execute the original products, followed by lossless residual copies to all FFN ranks.
 * Weights/shared scratch remain replicated; source-20 PRO mirrors remain. This recipe uses a distinct identity
 * and per-layer graph boundaries. Its arena accounting is not runtime fit, speed or output qualification.
 * create reads pack metadata only, owns five stage plans/images, canonical and per-card borrowed-image views,
 * merged GPU jobs and HOST_ONLY map. It checks arena+backend reserve+kept-free against each free-memory snapshot
 * before the caller opens GPU arenas. Static accounting does not qualify runtime peaks or five-slot capacity.
 *
 * load is single-use. Caller cards are already open with exactly canonical arena sizes and stay immutable/live.
 * qualified_writer_masks[owner] is caller evidence of actual current-stack two-rank protocol qualification;
 * every writer in each TP group must be set. peer_connect alone is not qualification. There is no host-route fallback.
 * No model admission is implied by preparation. MAIN owns protocol/numerical/model/capacity gates.
 * load owns two pinned workspaces and one aligned RAM arena. It streams every HOST_ONLY extent, loads selected
 * GPU tensors, calls unchanged arena_initialise with the resolved global embedding, and attaches every channel's
 * direct peer table. A startup TLS staging adapter lends the existing output workspace to arena_initialise;
 * nested/reentrant adapter use is refused. Both workspaces transfer to staging_retire on every load exit, so
 * failed waits never free unknown in-flight sources. Backend staging_retire is mandatory. No hidden retained
 * initializer workspace exists. No callback retains caller/model metadata.
 * Ready/host_arena/host_offsets and successful load stats publish only after the complete load/setup succeeds.
 * Failed loads are not reusable. Caller must drain/close all cards after failure before reusing peer mappings.
 * Destroy accepts NULL; Engram must already be closed, programs/drivers destroyed and cards drained. It frees
 * host RAM, plain canonical/views and each stage owner once (never base-plan-destroys borrowed images twice).
 * No token-path I/O/allocation or image support is provided here. WIDE/FULL plans reserve their maximum rows;
 * the caller must provide serial whole-flight group scheduling and qualify numerical behavior and runtime peaks.
 */
/* QUAD_ALL's default retains full-K EXL3 output shards and lossless gathers. An explicit
 * exact_output_shards=false + wire_format=BF16 instead selects512/640 inner shards, local FP32 expert sums,
 * sharded dense/shared work and BF16 partial sums. This changes arithmetic and is held for separate quality
 * gates. Its Stage1 expert products retain four parts. Rank0 owns the unsharded head and native draft.
 * Load and program description retain ordinary validation; main owns component and runtime gates. */
#ifndef MODEL_V41_TP2_MODEL_H
#define MODEL_V41_TP2_MODEL_H

#include "model_v41_tp2_program.h"

struct v41_tp2_model_description {
	struct model_limits limits;
	struct card_description_record cards[V41_DATA_TP2_PROGRAM_CARDS];
	uint64_t kept_free_bytes;
	uint32_t wire_format; /* explicit relaxed wire cast, zero retains the initial FP32 recipe */
	bool exact_output_shards, original_code_wire;
	bool original_wide_experts; /* default off, original exact QUAD: chunk-wide jobs and slab lossless gathers */
	bool original_batch; /* default off, exact original QUAD: BATCH48, legacy native PRO heads and verifier walks */
	bool attention_owner; /* default off, exact quad only; lossless residual distribution precedes the FFN */
	enum bb_exl3_product_order
		exl3_product_order; /* cold QUAD_ALL only; zero preserves four-part order */
	bool head_shards; /* default off, QUAD_ALL only: the target head and sampler cut by outputs over the ranks */
	bool exact_dense; /* default off, exact QUAD_ALL only: the attention's wq_b/wo_a/wo_b output shards, 2 gathers */
	bool draft_shards; /* default off, with head_shards: the draft's head, Markov bias and sampler cut too */
	bool draft_layers; /* default off, BF16 recipe, with draft_shards, no batch: the draft's layers TP (stage.h) */
	bool batch; /* default off, QUAD_ALL or explicit original_batch: BATCH programs can be prepared */
	bool batch_wide_head; /* default off, with batch: BATCH heads by the wide dense_stream form, one launch */
	bool batch_draft_rows; /* default off, with batch, no draft shards: one draft of all verifiers' rows in slabs */
	bool batch_context_slabs; /* default off, with batch: a step's draft context in slabs of 8 rows (stage.h) */
	bool batch_thin_bits; /* with batch: a BATCH program's rows the thin steps' bits (model_v41_tp2_stage.h) */
	uint32_t batch_grouped_rows; /* with batch_thin_bits: grouped trunk forms to so many rows (stage.h) */
	uint32_t batch_tc_paired; /* with batch_thin_bits: TC MoE short forms paired from so many rows (stage.h) */
	bool batch_draft_slots_head; /* with batch_wide_head: one verifier's draft head over the slots (stage.h) */
	uint32_t batch_row_sets; /* with batch_thin_bits: trunk row sets above batch_grouped_rows (stage.h) */
	uint32_t batch_core_once; /* a BATCH program's attention core (2: and window writes) once (stage.h) */
	uint32_t front_branch; /* thin coefficient fronts as graph side branches: 1 attention, 2 both (stage.h) */
	uint32_t core_heads; /* the attention core's most heads a block, 0 the rule (stage.h) */
	bool decode_round_once; /* default off: target decode BF16 partials summed in FP32, one rounding */
	bool draft_bf16_rows; /* the draft layers' attention products of BF16 rows (stage.h) */
	uint32_t batch_draft_wide; /* a batched draft's sites summed and posted once over its rows (stage.h) */
	bool draft_bf16_operands; /* class 80: the draft's quantized products of BF16 operands (stage.h) */
	bool ring_column_owners; /* the BF16 ring sums with column owners (model_v41_tp2_stage.h) */
	bool head_stream; /* default off, QUAD_ALL only: the CMP head products by dense_stream's kernel (same bits) */
	bool exl3_fast_products; /* EXL3 tensor layers: kernel_exl3_wide's transforms/products (identical words) */
	bool exl3_wide_entry; /* EXL3 tensor layers, chunked steps: the experts once over the chunk (both recipes) */
	bool exl3_tc_moe; /* default off, BF16 recipe EXL3 tensor layers: GNmac's tensor-core MoE (exl3_tc.h) */
	uint32_t exl3_tc_product_blocks; /* decode TC product grid: 0 keeps the measured geometry */
	uint32_t exl3_expert_rotation; /* default 0, with exl3_tc_moe: per-expert chunk rotation 1 or 2 (exl3_tc_records.h) */
	bool head_mma; /* default off, BF16 recipe: the head products on the tensor cores (kernel_head_mma) */
	bool trunk_mma; /* default off, BF16 recipe: thin BF16-row trunk products on the tensor cores (kernel_trunk_mma) */
	bool trunk_mma_fused; /* with trunk_mma: the _64 form adds its own splits (model_v41_tp2_stage.h) */
	bool trunk_mma_thin_splits; /* with trunk_mma: the _64 form's splits the thin form's (model_v41_tp2_stage.h) */
	bool router_mma; /* default off, BF16 recipe: a BATCH step's router by split-K MMAs (model_v41_tp2_stage.h) */
	bool router_mma_wide; /* with router_mma: prefill (wide) steps' routers too (model_v41_tp2_stage.h) */
	uint32_t gather_balance; /* a chunk's gathered index scans and router rows on the narrower ranks (stage.h) */
	bool pool_chunk; /* a chunk's pool entries and keys ahead of its slabs' scans (model_v41_tp2_stage.h) */
	bool pool_gather; /* a pooled source layer's compressor products quartered and gathered (stage.h) */
	bool router_first_whole; /* the first half's router on every rank, not gathered (stage.h) */
	bool tcp_two_blocks; /* the prefill TC MoE's GEMMs at two blocks a multiprocessor (stage.h) */
	bool queries_piece; /* a wide program's queries over each piece with wq_b, not a span at a time (stage.h) */
	bool index_span; /* a chunk's index scans over spans of rows, not a slab at a time (stage.h) */
	bool core_compact_halves; /* a split half's core into the compact rows, no copy (stage.h) */
	uint32_t dense_skinny_bf16; /* the BF16 skinny dense products' outputs a block above 48 rows (stage.h) */
	uint32_t dense_skinny_f32; /* the float32-weight skinny products' outputs a block above 48 rows (stage.h) */
	uint32_t dense_staged; /* 0, or the staged wide dense form's outputs a block (stage.h) */
	uint32_t graph_parts; /* a card's step graph behind the cone in this many parts (model_v41_tp2_stage.h) */
	bool ordered_wide; /* wide EXL3 attention trunk products by the ordered wide form (model_v41_tp2_stage.h) */
	uint32_t graph_branches; /* TC-MoE layers' shared expert a graph branch: 1 thin, 2 BATCH too (stage.h) */
	bool piece_rings; /* BF16 rings over 64-row pieces in wide steps (model_v41_tp2_stage.h) */
	bool trunk_shape_slices; /* thin trunk products at their shard shape's measured slice count (same bits) */
	bool bf16_slab_rings; /* default off, BF16 recipe chunked steps: the attention summed a slab at a time (ring) */
	uint32_t wide_exchange; /* BF16 recipe chunked steps: a piece's ordered push by many CTAs, V41_DATA_TP_EXCHANGE_*
				   (model_v41_tp2_stage.h; the same bits) */
	uint32_t split_overlap; /* default 0, BF16 recipe chunked steps: layers in two halves, sums beside work (stage.h) */
	uint32_t split_least_rows; /* split_overlap: chunks of fewer rows whole (model_v41_tp2_stage.h) */
	bool launches_untrimmed; /* --tp_launch_trims=0: the launches as before 3 November (model_v41_tp2_stage.h) */
	uint32_t full_split_overlap; /* the full programs' split_overlap mode; 0: split_overlap's (stage.h) */
	uint32_t tail_rows; /* 0, or a full program of these rows takes the 64-row forms a 64-row group (stage.h) */
	bool tail_overlay; /* tail_rows: the tail programs' TC MoE workspace over the prefill form's (stage.h) */
	bool tail_sets; /* tail_rows: a tail program's trunk products one launch over its 64-row groups (stage.h) */
	bool thin_tail; /* tail_rows: full programs of 9 to 63 rows in a thin step's forms (stage.h) */
	uint32_t full_cone_behind; /* 0, or the full programs' dead halves' bound (stage.h) */
	bool dead_rows_trimmed; /* with full_cone_behind: the sums' copies of the rows the cone reads alone (stage.h) */
	bool chunk_core; /* default off, chunked steps: the attention core over the chunk (model_v41_tp2_stage.h) */
	bool replicated_embedding; /* QUAD_ALL: every rank embeds its own rows (the same bits; no streams leave card 0
				      at a step's start through the host) */
	uint32_t tensor_layout; /* model_v41_tp2_program.h; zero retains the initial two pairs */
};

struct v41_tp2_model_load_request {
	const struct backend_operations *backend;
	struct device_card *const *cards;
	const struct storage_operations_record *storage;
	const struct storage_reader *reader;
	uint64_t wait_limit_nanoseconds;
	uint32_t qualified_writer_masks[V41_DATA_TP2_PROGRAM_CARDS];
	/* 0: the serial loads (host arena, then the cards); N: both at once, each on N threads (tp_load.h) */
	uint32_t load_threads;
	uint32_t load_reads_per_copy; /* load_threads != 0: reads in flight a copy at most; 0: no limit */
	uint32_t load_host_threads; /* load_threads != 0: the host arena's threads; 0: load_threads */
	uint32_t load_copy_split; /* load_threads != 0: tp_load.h's tp_load_copies.split */
};

struct v41_tp2_model {
	struct v41_tp2_model_description description;
	struct v41_tp2_stage_plan *stages[V41_DATA_TP2_PROGRAM_CARDS];
	struct model_plan_record *canonical;
	struct model_plan_record *views[V41_DATA_TP2_PROGRAM_CARDS];
	struct tp_load_job *jobs;
	struct tp_load_plan gpu_load;
	struct tp_host_load_plan host_load;
	struct tp_load_result gpu_result;
	struct tp_host_load_result host_result;
	uint64_t *extent_offsets; /* owned preparation map; host_offsets publishes it only after successful load */
	void *host_arena;
	const uint64_t *
		host_offsets; /* NULL until ready; internal host_load.extent_offsets owns the allocation */
	uint64_t host_bytes;
	uint64_t recipe_identity;
		uint64_t output_workspace_bytes;
		struct tp_load_copies
			copies; /* load_threads != 0: what each copy delivered (tp_load.h) */
		bool load_attempted;
		bool ready;
};

MUST_CHECK result_status
v41_tp2_model_create(const struct pack_record *pack,
		     const struct v41_tp2_model_description *description,
		     struct v41_tp2_model **model_out);
void v41_tp2_model_destroy(struct v41_tp2_model *model);

/* Failure leaves result fields/ready/attachable RAM unpublished; model metadata remains inspectable. */
MUST_CHECK result_status
v41_tp2_model_load(struct v41_tp2_model *model,
		   const struct v41_tp2_model_load_request *request);

/* Description borrowing this model; program preparation remains separately owned/qualified.
 * QUAD_ALL publishes four actual cards and NULL inactive slot4. Refusal leaves the output unchanged. */
MUST_CHECK result_status v41_tp2_model_program_description(
	const struct v41_tp2_model *model,
	struct v41_tp2_program_description *description_out);

/* Exact frontend byte binding against the verified HOST_ONLY RAM map. Ready is required. Same metadata/byte
 * refusals as front_end_pack.h; no synthetic ordinary GPU load layout is made. */
MUST_CHECK result_status v41_tp2_model_front_end_match(
	const struct v41_tp2_model *model, const void *external_tables,
	uint64_t external_bytes);

#endif
