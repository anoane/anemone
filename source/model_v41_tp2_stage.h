/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Cold text-only TP/PP stage inventory. No GPU, runtime or capacity admission is performed.
 *
 * One result owns one card's selected target tensors and all resident expert rectangles, optionally the target
 * embedding or the unsharded head/DSpark tail. TP2/TP4 shards the original query/output/shared/routed
 * matrices; router, mHC, KV/indexer and Engram GPU matrices remain replicated. Host Engram tables, vision weights
 * and vision state are excluded explicitly. Jobs are dynamically allocated, sorted by source extent and laid out
 * monotonically. The prepared load plan reports exact bounded host workspaces; no tensor payload is read here.
 *
 * KV/key source sizes use the same context formulas as PP for all four producers. One card_lay_out call accounts
 * shared scratch, owned rings/checkpoints and required owned/external pool/key/selection copies once per stage.
 * A head/draft stage must be unsharded, end at layer40 and include taps37..39 with a configured draft block.
 * Its existing V41_DRAFT layout reserves the draft embedding copy once. The global embed remains unresolved when
 * include_embedding is false; this module neither duplicates that weight nor calls arena_initialise.
 *
 * model is incomplete global metadata for a later merged plan, NOT an admitted seam plan. Unowned layer/global
 * cards use the out-of-range backend card sentinel. Never call arena_initialise/program_prepare on it standalone.
 * TP channels reserve two sites and two parities. Row capacities<=8 keep one channel per layer/site;
 * larger capacities reuse one region per distinct site, with layer phases supplied by tp_exchange.h. Full-sized
 * compact/scratch and channel rows are accounted at creation. Graphs, modules, backend and driver peaks remain
 * outside these arena totals. Ranks1/2/4 are supported; TP4 expert shards use aligned512/640/512/640 widths.
 * Exact quad instead replicates attention/shared tensors, uses balanced576 gate/up and1280 down output shards,
 * and reserves lossless hidden/FP32 owner gathers. Its default five1M replicated-state arenas exceed CMP capacity.
 * attention_owner is a separate default-off exact quad control: it keeps all tensors/scratch, but only the closed
 * attention group's persistent state on each CMP. The executable program distributes complete residuals before
 * FFN work. Native PRO source-20 mirrors remain resident. No capacity/runtime or numerical gate is implied.
 * CPU totals are necessary accounting, not a five-slot capacity pass or runtime qualification.
 *
 * exl3_all_targets is a separate cold-only full40 quad: four CMP stages output-shard full-K EXL3 target experts,
 * replicate original dense/shared weights, and rank0 owns embedding/head/full native draft. Its head exception
 * never shards draft weights or implies executable program admission. Original-code wire and attention_owner
 * are refused. Four-part is default; explicit eight-part metadata has a distinct model recipe identity.
 *
 * Creation validates the real pack, selected shapes and limits, allocates only at load, and leaves plan_out
 * untouched on any failure. The result and its immutable jobs/load description borrow pack until destruction.
 */
/* QUAD_ALL's default retains full-K EXL3 output shards and lossless gathers. An explicit
 * exact_output_shards=false + wire_format=BF16 instead selects512/640 inner shards, local FP32 expert sums,
 * sharded dense/shared work and BF16 partial sums. This changes arithmetic and is held for separate quality
 * gates. Its Stage1 expert products retain four parts. Rank0 owns the unsharded head and native draft.
 * Load and program description retain ordinary validation; main owns component and runtime gates. */
#ifndef MODEL_V41_TP2_STAGE_H
#define MODEL_V41_TP2_STAGE_H

#include "model_v41_tp2.h"
#include "anemone/exl3_layout.h"

#define TP4_ATTENTION_SECOND_FIRST 8u
#define TP_MOST_GRAPH_PARTS 8u /* --tp_graph_parts */
#define V41_DATA_TRUNK_MMA_PARTIAL_VALUES \
	16384u /* trunk_mma's _64 form: k_split_count times outputs, at most */
/* and at the thin steps' splits (batch_thin_bits, trunk_mma_thin_splits): the largest split times outputs of the trunk
 * shard shapes (split 4 x wq_b's 8,192, split 16 x wo_a's 2,048), so a 48- or 64-row step's rows keep the thin bits */
#define V41_DATA_TRUNK_MMA_THIN_PARTIAL_VALUES 32768u
#define V41_DATA_TRUNK_MMA_COUNTERS \
	256u /* its fused form's: a counter an output block (128 outputs), at the end */
#define V41_DATA_ROUTER_MMA_SPLITS \
	8u /* router_mma's splits of the 5,120 inputs (GNmac's router: 8) */
#define V41_DATA_TP_EXCHANGE_PUSH \
	1u /* wide_exchange: tp_exchange.h's wide form */
#define V41_DATA_SPLIT_OVERLAP 1u /* split_overlap: both sums beside work */
#define V41_DATA_SPLIT_SERIAL \
	2u /* the halves one behind the other, no branch */
#define V41_DATA_SPLIT_ATTENTION_ONLY \
	3u /* only the attention's sums beside work */
#define V41_DATA_SPLIT_FFN_ONLY 4u /* only the FFN's */
#define V41_DATA_TP_SPLIT_MOST_ROWS \
	(2u * LINEAR_WIDE_MOST_ROWS) /* split_overlap: two wide launches */
#define V41_DATA_SPLIT_PIPELINE \
	5u /* 1, the attention in halves too, the last sums beside the next layer's */
#define V41_DATA_SPLIT_JOINT \
	6u /* 5 with one routed-expert call over the chunk (each expert decoded once) */
#define V41_DATA_SPLIT_JOINT_DOWN \
	7u /* 6 with that call's down GEMM and combine a half each: A's FFN sums beside
                                          B's (exl3_tc_records.h half_rows) */
#define V41_DATA_GATHER_BALANCE_ALL \
	1u /* gather_balance: every chunked program's gathers */
#define V41_DATA_GATHER_BALANCE_WIDE \
	2u /* the wide programs' alone (rc04: a full step's on Fraser +3.1%) */
#define V41_DATA_GATHER_BALANCE_WEIGHTED \
	3u /* every program's: the wider ranks fewer units, the narrower more */
#define V41_DATA_GATHER_WIDE_SIXTEENTHS \
	3u /* GATHER_BALANCE_WEIGHTED: a wider rank's part (a quarter: 4) */
#define V41_DATA_GATHER_SIXTEENTHS 16u
#define V41_DATA_TP_EXCHANGE_OWNER_SUMS \
	2u /* wide_exchange: its owner sums (half the bytes, the same bits) */
#define V41_DATA_TP_EXCHANGE_OWNER_RINGS \
	3u /* and the FFN site's slab rings as owner ring sums (the same bits) */
#define V41_DATA_TP_EXCHANGE_CE_OWNER_SUMS \
	4u /* owner sums whose bytes copy engines move (the same bits) */
#define V41_DATA_TP_EXCHANGE_CE_RINGS \
	5u /* and the FFN site's slab rings as copy-engine owner ring sums (same bits) */
#define TP4_ATTENTION_THIRD_FIRST 14u
#define TP4_ATTENTION_FOURTH_FIRST 20u
#define TP4_ATTENTION_PREFIX_END 29u
#define V41_DATA_ORIGINAL_BATCH_SESSIONS STEP_BATCH_MAXIMUM_SESSIONS
#define V41_DATA_ORIGINAL_BATCH_ROWS \
	(V41_DATA_ORIGINAL_BATCH_SESSIONS * (MODEL_DRAFT_BLOCK + 1u))

/* Whole source/index groups stay together. The unsharded native tail owns later layers. */
static inline uint32_t v41_tp4_attention_owner(uint32_t layer)
{
	return layer < TP4_ATTENTION_SECOND_FIRST ? 0u :
	       layer < TP4_ATTENTION_THIRD_FIRST  ? 1u :
	       layer < TP4_ATTENTION_FOURTH_FIRST ? TP2_RANKS :
	       layer < TP4_ATTENTION_PREFIX_END	  ? TP4_RANKS - 1u :
						    TP4_RANKS;
}

struct v41_tp2_stage_description {
	struct model_limits limits;
	uint32_t card;
	uint32_t rank;
	uint32_t rank_count;
	uint32_t first_target_layer;
	uint32_t target_layer_count;
	bool include_embedding;
	bool include_head_and_draft;
	bool exact_output_shards, original_code_wire;
	bool original_wide_experts; /* default off, original exact quad prefix: one expert call over the chunk;
				    * gathered pairs/sums span row_capacity, banks remain eight rows */
	bool original_batch; /* default off, exact original quad plus native PRO; legacy per-verifier draft walks */
	bool exl3_all_targets; /* cold full40 quad only; all target experts EXL3, draft remains native/unsharded */
	enum bb_exl3_product_order exl3_product_order;
	bool attention_owner; /* default off: exact quad's persistent state and attention execute on one owner */
	/* default off, exl3_all_targets only (tp4_v3_claude): the target head cut by outputs, rank r's rows
	   [r V/4, (r+1) V/4) of the vocabulary V, each rank's own final norm, head cut and sampler_shard
	   (sampler_shard_records.h); ranks 1-3 then hold the norm, their rows of the head and the result arrays, every
	   rank the sampler's exchange area, peer table and settings word; rank 0 the whole head unless draft_shards. */
	bool head_shards;
	/* default off, exact exl3_all_targets only (tp4_v3_claude's exact dense, attention part): wq_b by heads, wo_a by
	   groups, wo_b by output channels as output-axis shards of the full matrices (every output its full-K chain);
	   each rank runs its 16 heads' attention, its two groups of wo_a into its slot, a lossless gather of the groups,
	   its 1280 outputs of wo_b over the full groups into its slot, a lossless gather of the sublayer output */
	bool exact_dense;
	/* default off, with head_shards: the draft's head, Markov bias and sampler cut the same way; rank 0 shares its
	   draft rows, step area and views (draft_share_records.h); ranks 1-3 then hold the Markov tables too */
	bool draft_shards;
	/* default off, the BF16 recipe with draft_shards and replicated embedding: the draft's three layers
	   tensor-parallel as the target's (GNmac's DSpark: wq_b / wo_a by heads, wo_b by inputs, the shared and routed
	   experts by inner rows, their sums by the BF16 ring), each rank its draft context, its draft rows and its cut;
	   the draft tail then holds one segment a rank; BATCH (with batch_draft_rows and batch_wide_head): each rank
	   the batched draft of every verifier, its slabs' sums by the ring, its own round table and batched draft area */
	bool draft_layers;
	bool batch; /* default off, exl3_all_targets or original_batch: BATCH programs and slab result areas */
	bool batch_draft_rows; /* default off, with batch and no draft shards: the verifiers' drafts as one draft of
				  their rows in slabs (one weight pass a slab), not a walk a verifier */
	bool batch_context_slabs; /* default off, with batch: a BATCH step's draft context (main_proj, main_norm, the wkv
				     products) a slab of 8 rows at a time, by the thin steps' forms, not one piece of its rows */
	bool batch_thin_bits; /* with batch: a BATCH program's target rows the thin steps' bits: trunk_mma's _64
				 form at the 1-8-row form's splits, the head of 9-64 rows at the 8-row form's split
				 (head_mma_64_s4) and the TC MoE's short forms at every row count
				 (exl3_tc_prepare_short) */
	bool ring_column_owners; /* default off, the BF16 recipe: every BF16 ring sum with column owners
				    (EXACT_RING_BF16_SUM_COLUMNS: owner o the o-th quarter of each row, so a
				    row's order of additions is the same at any row index and row count); other bits
				    than the default's row-quarter owners */
	uint32_t batch_grouped_rows; /* with batch_thin_bits: a BATCH step of 9 to this many rows (at most 48) takes
					its trunk products by kernel_trunk_mma's grouped forms (the 1-8-row forms over
					its row groups of 8: the thin bits, one launch, no partials); 0: the _64 form */
	bool batch_draft_slots_head; /* default off, with batch_wide_head: a BATCH program of one verifier runs its
					draft head over the 8 slots of each position (positions x 8 rows), as before;
					off: over its own rows (8-row form), each position's logits to its slot 0 */
	uint32_t batch_tc_paired; /* default 0, with batch_thin_bits: a BATCH step of at least so many rows runs the
				     TC MoE's short forms with their products two slices a warp item
				     (exl3_tc_prepare_paired: the same partials, half the items) */
	bool draft_bf16_rows; /* default off: the draft layers' wq_a, wkv, wq_b and wo_b take their BF16 rows (the
				 hidden, qr and wo_a's groups) by the target's form of BF16 rows and E4M3 weights
				 (exl3_trunk_bfloat16) in place of the rows quantized to E4M3: the drafts' numerics
				 alone (the target's tokens and logits are not the draft's) */
	uint32_t batch_draft_wide; /* default 0, with batch_draft_rows and draft_layers: 1, a batched draft of several
				      slabs sums and posts each draft layer's two sites once over all its rows (one
				      BF16 ring of up to ring_rows rows: the same element sums, so the same bits) in
				      place of a ring and a post a slab; 2, also each draft layer's products over all
				      its rows as a BATCH program's target layers (the wide forms' thin bits), state
				      writes, cores and routed experts a slab at a time; 3, also its window writes
				      and block core once over all its rows (each verifier's block whole, as the
				      reference's: other drafts where the slab walk cut a block; drafts only) */
	bool draft_bf16_operands; /* default off, class 80: every quantization of the draft's (context, layers, experts)
				     writes its rows' own BF16 values as the carried operands (activation_carry:
				     exact), in place of their E4M3 values, the draft's thin site fronts' rows carried
				     again behind them: its quantized products read unrounded rows (GNmac's W8A16 /
				     W4A16 class); the drafts' numerics alone */
	uint32_t core_heads; /* default 0 (the measured rule): the attention core's reference-order blocks of at
				most so many heads (8, 16): more blocks of fewer heads, the same bits
				(attention_core_records.h) */
	uint32_t front_branch; /* default 0: 1, a thin step's attention site front as its input front, then its
				  coefficient front as a graph side branch beside the products, core and sum, joined at
				  the site's post (site_front_records.h: the same bits); 2, also the FFN site's (input
				  front, the separate router, the coefficient front a side run beside the shared
				  expert's) */
	uint32_t batch_core_once; /* default 0 (main.c: 3): 1, a BATCH program of several slabs runs each layer's
				     attention core once over all its rows (the slabs' state writes, queries and
				     scans first): the same bits, a core launch (two in the reference order) a layer
				     instead of one a slab; 2, also its window writes (the prepare's window part) once
				     over the rows, qr the chunk's QUERY_NORM (the prepare's qr bits), the slabs' pool
				     writes alone; 3, pools over the step, slot retention in a separate
				     following launch */
	uint32_t batch_row_sets; /* default 0, with batch_thin_bits: a BATCH step of more than batch_grouped_rows rows
				    (to 64) takes its trunk products by the 1-8-row form (8) or the grouped forms of 2
				    or 3 row groups (16, 24) over sets of so many rows (gridDim.y): the thin bits, no
				    partials or reduce; 0: the _64 form at the thin split */
	bool batch_wide_head; /* default off, with batch: a BATCH step's head (or head cut) one launch of the wide
				 dense_stream form over all its rows (dense_stream_records.h), then the slab samplers */
	bool exl3_fast_products; /* EXL3 tensor layers: kernel_exl3_wide's w_* transforms and _t8_g1 (thin) / _t4_g4p
				    (wide) products, exl3_v3_claude's, identical words; false: the retained chain */
	bool head_mma; /* default off, the BF16 recipe: the head products (target, draft, BATCH wide) by kernel_head_mma */
	bool trunk_mma; /* default off, the BF16 recipe: BF16-row trunk products of 1-8 rows by kernel_trunk_mma */
	bool trunk_mma_fused; /* default off, with trunk_mma: the _64 form adds its own splits (the same bits, slower) */
	bool trunk_mma_thin_splits; /* default off, with trunk_mma: the _64 form splits the input groups as the 1-8-row
				       form of its shape does (where its partials fit): each row the thin steps' bits */
	bool router_mma; /* default off, the BF16 recipe, batch: a BATCH step's router by split-K tensor-core MMAs
			    (dense_linear_records.h's router_mma_arguments): its partials area */
	bool router_mma_wide; /* default off, with router_mma: a wide (prefill) step's router of more than 64 rows too
				 (64-row blocks, kernel_head_mma.cu's router_mma_wide_entry; another order than the
				 dense product: GNmac's router class); the partials of a wide launch's rows */
	uint32_t gather_balance; /* default 0, split_overlap's pipeline: a half's index scans and router scores
				    gathered from the ranks of the narrower expert slice alone (the 512-column ones,
				    which wait for the 640-column ones at every exchange), in equal parts; each row's by
				    the same launches: the same bits. V41_DATA_GATHER_BALANCE_*: every program's,
				    or the wide programs' alone; _WEIGHTED: every rank a part, a wider (640-column) rank
				    V41_DATA_GATHER_WIDE_SIXTEENTHS of the units (ri03k: its MoE ~18 ms a 2,048-row
				    step over a narrower one's), the narrower ranks the rest (whole 8-row groups) */
	bool pool_chunk; /* default off, chunk_core: a chunk's (half's) pool entries and index keys (the compressor, wk,
			    the keys) in one launch a piece of at most 1,024 rows behind its window writes, ahead of its
			    slabs' scans, not a slab at a time: a row sees no entry of a later row (model_layout.h), the
			    compressor takes a row's partner from the row before it in the launch: the same bits */
	bool pool_gather; /* default off, split_overlap's pipeline: a pooled source layer's compressor products (wkv,
			     wgate) of a half's rows a quarter a rank (gather_balance's share), gathered by copy
			     engines (each row's by the product every rank runs: the same bits), not every row on every
			     rank */
	/* default off, exl3_tc_moe: the prefill TC MoE's GEMM launches take kernel_exl3_tcp.cu's _b2 forms (two blocks
	 * a multiprocessor, 128 registers, the k16 steps not unrolled): every output's MMA and add order the same, the
	 * same bits */
	bool tcp_two_blocks;
	bool router_first_whole; /* default off, split_overlap's pipeline: the first half's router scores over its
				    rows on every rank (router_scores), not a quarter each gathered: its gather's copies
				    would share the copy engines with the second half's attention sums beside it; each
				    row's by the same product: the same bits */
	uint32_t dense_skinny_bf16; /* default 0 (the wide form's 1), class 80: the BF16 skinny dense products (index
				       weights, keys: at most 1,024 outputs) above 48 rows at this many outputs a block
				       (dense_linear.h's skinny form; each (output, row)'s chain the same: the same bits) */
	uint32_t dense_skinny_f32; /* default 0 (dense_linear.c's 4), class 80: the float32-weight skinny products (the
				      routers, a pooled source's compressor) above 48 rows at this many outputs a block */
	uint32_t dense_staged; /* 0, or (--dense_staged) 8/16/32/64: the wide dense products of at most
				  5,120 inputs (no row scale) by the staged form, the group's rows read once for
				  that many outputs (dense_linear_records.h: each (output, row)'s chain the same: the same bits) */
	bool index_span; /* default off, a wide program of pool_chunk: a chunk's (half's) index scans and selections over
			    up to V41_DATA_SPAN_ROWS rows a launch (a view's rows from a row on, index_scan.h), not
			    a slab at a time, each row's selection by the kernels a slab's takes (split_rows): the
			    same bits; the scores of a span (within V41_DATA_INDEX_SPAN_BYTES a card, else off) */
	bool core_compact_halves; /* default off, split_overlap's halves: a half's core writes the rank's heads into
				     the compact rows wo_a reads (attention_core.h output_compact), not the full rows
				     and a tp2_join copy; the halves' attention runs in order on the chain (split_layer),
				     the core's values the same bits */
	bool queries_piece; /* default off, a wide program: the queries (attention_queries_entry) over each piece
			       with wq_b, from its compact heads in place, not over each span of 8 slabs from the heads
			       copied into the full rows; each row's by the same elementwise launch: the same bits */
	uint32_t graph_parts; /* a card's step graph behind the cone in this many parts (1: one; model_v41_program.c) */
	bool ordered_wide; /* EXL3 attention trunk products of more than 8 rows by the ordered wide form (same bits) */
	uint32_t graph_branches; /* exl3_tc_moe: 1 (main.c's default) a thin layer's shared expert a branch of the step
				    graph beside its routed experts, the ring their join (backend_seam.h's LAUNCH_*;
				    same launches); 2 a BATCH step's too; 0 one chain */
	bool piece_rings; /* steps of more than 8 rows: BF16 rings over pieces of 64 rows, not slabs (same bits) */
	bool trunk_shape_slices; /* thin trunk products at the slice count measured for their shard shape
				    (model_v41_program.c tp_shape_slices; ordered chains: the same bits) */
	uint32_t exl3_tc_product_blocks; /* decode TC product grid: 0 keeps the measured geometry */
	bool decode_round_once; /* target THIN/BATCH sums by BF16 owner exchange, FP32 adds and one BF16 rounding */
	bool exl3_tc_moe; /* default off, the BF16 recipe's EXL3 tensor layers: GNmac's tensor-core routed experts
			     (exl3_tc.h), their workspace exl3_tc_workspace */
	uint32_t exl3_expert_rotation; /* default 0, with exl3_tc_moe: each target layer's rank holds an expert's chunk
					  by exl3_tc_chunk_of (1 four-way, 2 three-way; exl3_tc_records.h) */
	bool exl3_wide_entry; /* default off, EXL3 tensor layers in chunked steps: the fronts and router scores a piece,
				 ONE wide select and EXL3 expert call over the chunk (the grouped _t4_g4p products with
				 exl3_fast_products; the exact recipe's gathers a slab at a time), then each slab's shared
				 expert, ring sum (or gathered sums) and post as a thin step's */
	bool bf16_slab_rings; /* default off, the BF16 recipe's tensor layers in chunked steps: the attention's sum of a
				 piece of more than 8 rows a slab at a time by the BF16 ring (the thin steps' sums, about a
				 fifth of the bytes) instead of one FP32 ordered push over the piece (another rounding) */
	uint32_t wide_exchange; /* the BF16 recipe's ordered push of a piece of more than 8 rows by tp_exchange.h's wide
				   form (V41_DATA_TP_EXCHANGE_PUSH: four launches, the payload and sums by many CTAs;
				   _OWNER_SUMS: each rank sums its shard, half the bytes; _OWNER_RINGS: also the FFN site's
				   slab rings in one exchange a piece; all the same bits); 0: one CTA */
	uint32_t split_overlap; /* default 0, the BF16 recipe's EXL3 tensor layers in chunked steps: from wo_a on, the layer
			       in two halves of the chunk's rows (each through a half view, model_seam.h), each half's
			       attention and FFN sums a graph side run beside the other half's products (Morrow's
			       PREFILL_REDUCE_OVERLAP); every row's operations unchanged, the same bits */
	/* default off (main.c --tp_launch_trims=1), the BF16 recipe, every trim the same bits: a BF16 ring sum of at
	 * most ring_rows rows reads its values and writes the sum over them (the ring reads a pack's every owner before
	 * it writes it): the attention site's values as they are (a bfloat16 rounded again is itself), a TC-MoE FFN
	 * site's whose shared expert is a graph branch after the combine added the routed sums to the shared rows
	 * (exl3_tc_records.h shared_sum, tp2_join's arithmetic); and an EXL3 layer's wo_b, which reads the groups' BF16
	 * rows, has no quantization of them before it; on: those copies, joins and quantizations (before 3 November) */
	bool launches_untrimmed;
	/* 0, or (--prefill_tail_rows) a full program of more than 64 rows and at most these takes each 64-row group of
	 * its rows in the 64-row chunk's forms: the decode TC MoE over its rows, trunk_mma's _64 form a group (each row
	 * the bits of the exact plan's 64-row tail chunks: every other launch is the same at any row count above 8) */
	uint32_t tail_rows;
	/* with tail_rows (--prefill_tail_overlay, default on): a tail program's decode TC MoE call takes its workspace
	 * from the start of exl3_tc_workspace (then of 64 rows) over the prefill form's exl3_tcp_workspace behind it,
	 * grown where the call needs more: one card stream runs one program at a time, a tail program makes no prefill
	 * form call and a 64-row one no tail call (the same launches, the same bits); off: exl3_tc_workspace of
	 * tail_rows */
	bool tail_overlay;
	/* with tail_rows (--prefill_tail_sets): a tail program's trunk products take one _64-form launch over all its
	 * 64-row groups (row sets: kernel_trunk_mma.cu; each group a one-group launch's bits) and one reduce over all
	 * its rows, their partials in trunk_mma_set_partials; off: a launch (and a reduce) a group */
	bool tail_sets;
	/* with tail_rows (--prefill_thin_tail): a full program of 9 to 63 rows (no ladder or tail program has such rows:
	 * main.c) takes the rows the exact plan's tail leaves short of a 64-row chunk but its row of logits, each row in
	 * a thin step's forms as a BATCH program's (batch_thin_bits: trunk_mma's thin splits or grouped forms, the TC
	 * MoE's short forms, the shared expert's _64 form from the BF16 rows): one step for what took one a 8 rows */
	bool thin_tail;
	/* 0, or (--prefill_dead_halves, front_end_seam.h's engine_full_cone_behind) the least rows between a
	 * full program's last row and its prompt's cone end's last row where the program has
	 * MODEL_FULL_CONE_LEAST_ROWS rows or more: a split decoder layer leaves the sums and gathers of a half
	 * whose last row's cone count (model_layout.h's MODEL_CONE_LAYERS at that bound) reads none of the
	 * layer's outputs of its rows */
	uint32_t full_cone_behind;
	/* with full_cone_behind: each sum of a split decoder layer's half copies only its rows from the first one whose
	 * output there the cone may read (the owner sums' quarters and the owner rings' slabs: the dead rows lead each
	 * segment); the sums of the rows before it, never read, keep stale rows */
	bool dead_rows_trimmed;
	uint32_t full_split_overlap; /* with split_overlap: the full programs' mode (1 to 6), 0 split_overlap's;
				      the full steps' sums are the worse covered (rc04: a 2,048-row FULL step 27%
				      idle on them) and may take another mode than the wide ones: the same bits */
	uint32_t split_least_rows; /* with split_overlap: a chunk of fewer rows runs whole (0: every chunk of halves of more
				      than 64 rows splits); a half decodes every routed expert's weights again, which costs a
				      small chunk more than the overlap saves; the same bits either way */
	bool chunk_core; /* default on (main.c), chunked steps: every slab's state writes and index scan in slab order, then the
			    attention core in one launch over the chunk's rows (8-row launches' blocks, a row each) in
			    place of a core a slab: the window (128) and the chunk (at most 1,024) fit the ring (4,096),
			    so no later slab overwrites what an earlier row reads; the same bits */
	bool head_stream; /* default off, exl3_all_targets only: the head products (whole, cut, draft) by dense_stream's
			     kernel (dense_stream_records.h; the dense product's bits) */
	uint32_t wire_format; /* ordered_push_records.h: FP32 or BF16; local input/output and sums stay FP32 */
};

/* One conservative original BATCH recipe. Both stage creation and program validation use this cold
 * contract; mutating a prepared stage cannot admit an EXL3 draft, a different tail or wider gathers. */
static inline bool v41_original_batch_stage_is_valid(
	const struct v41_tp2_stage_description *description)
{
	const struct model_limits *const limits = &description->limits;
	const bool tail = description->card == TP4_RANKS;
	return description->original_batch && description->batch &&
	       description->card <= TP4_RANKS &&
	       description->exact_output_shards &&
	       !description->original_code_wire &&
	       description->wire_format == 0 &&
	       !description->original_wide_experts &&
	       !description->exl3_all_targets &&
	       !description->attention_owner && !description->head_shards &&
	       !description->draft_shards && !description->draft_layers &&
	       !description->exact_dense && !description->batch_wide_head &&
	       !description->batch_draft_rows &&
	       description->batch_context_slabs &&
	       description->batch_core_once == 0 &&
	       description->split_overlap == 0 &&
	       description->full_split_overlap == 0 &&
	       !description->exl3_wide_entry && !description->exl3_tc_moe &&
	       !description->head_mma && !description->trunk_mma &&
	       !description->router_mma && !description->head_stream &&
	       !description->draft_bf16_rows &&
	       !description->draft_bf16_operands &&
	       limits->state_width == STATE_WIDTH_16 &&
	       limits->session_capacity == V41_DATA_ORIGINAL_BATCH_SESSIONS &&
	       limits->row_capacity >= MODEL_WIDE_LEAST_ROWS &&
	       limits->row_capacity <= LINEAR_WIDE_MOST_ROWS &&
	       limits->linear_profile == LINEAR_PROFILE_PRECISION &&
	       limits->linear_order == LINEAR_ORDER_REFERENCE &&
	       limits->cmp_order == CMP_ORDER_REFERENCE &&
	       (tail ? description->rank == 0 &&
				description->rank_count == 1u &&
				description->first_target_layer ==
					TP4_ATTENTION_PREFIX_END &&
				description->target_layer_count ==
					MODEL_LAYERS -
						TP4_ATTENTION_PREFIX_END &&
				description->include_head_and_draft &&
				!description->include_embedding &&
				limits->draft_positions == MODEL_DRAFT_BLOCK :
		       description->rank == description->card &&
				description->rank_count == TP4_RANKS &&
				description->first_target_layer == 0 &&
				description->target_layer_count ==
					TP4_ATTENTION_PREFIX_END &&
				!description->include_head_and_draft &&
				description->include_embedding ==
					(description->card == 0) &&
				limits->draft_positions == 0);
}

struct v41_tp2_stage_channel {
	struct ordered_push_description collective;
	uint32_t layer;
	uint32_t site;
	/* the ring banks, one a row count r (1 .. ring_rows): ring_bank_bytes + r ring_row_bytes each, back to back
	 * (v41_ring_bank_start; ring_row_bytes 0: every bank ring_bank_bytes) */
	uint64_t ring_offset, ring_bank_bytes, ring_row_bytes;
	uint64_t dense_ring_offset, dense_ring_bank_bytes; /* exact_dense: the site's own banks of the dense gathers */
};

/* Where a channel's ring bank of `rows` rows (1 .. ring_rows) starts past its ring_offset; rows = ring_rows + 1: the
 * banks' bytes. */
static inline uint64_t v41_ring_bank_start(uint64_t bank_bytes,
					   uint64_t row_bytes, uint32_t rows)
{
	return (uint64_t)(rows - 1u) * bank_bytes + row_bytes * (rows - 1u) * rows / 2u;
}

struct v41_tp2_stage_plan {
	struct v41_tp2_stage_description description;
	struct model_plan_record *model;
	struct tp_load_job *jobs;
	struct v41_tp2_stage_channel *channels;
	struct tp_load_plan load;
	struct linear_range compact, exact_hidden, exact_sums;
	struct linear_range
		dense_input; /* exact_dense: the gathers' owner-major slots, [ranks][rows][width] */
	/* index_span: the scores [index_rows][the card's most entries] and the split annex of index_rows rows (0: none) */
	struct linear_range index_scores;
	uint32_t index_rows;
	/* head_shards: the sampler's exchange area, its peer table (the four ranks' areas), its settings copy
	   (sampler_shard_records.h); the result card the table of the four ranks' copies, for its settings push */
	struct linear_range sampler_exchange, sampler_peers, sampler_settings,
		sampler_settings_peers;
	/* draft_shards: ranks 1-3 a share area then the draft's logits [8][V/4], results [8] and Markov rows [8]; rank 0
	   the table of the ranks' share areas and its ticket counters */
	struct linear_range draft_area, draft_peers, draft_counters;
	struct linear_range
		batch_results; /* batch, on a card that samples: one result area a slab */
	struct linear_range
		batch_logits; /* batch_wide_head, on a card that samples: [64][its outputs] */
	/* batch, the result card: the round table (a round a session slot) and the draft areas of verifiers 1 .. session
	   capacity - 1 (verifier 0 takes the draft arrays), v41_batch_draft_* below */
	struct linear_range batch_rounds, batch_drafts;
	/* batch_draft_rows, the result card: one draft of every verifier's rows (draft_records.h's BATCHED): slab blocks,
	   views, results, Markov, collapse and logits of V41_DATA_BATCH_DRAFT_ROWS rows, v41_batch_rows_* */
	struct linear_range batch_rows;
	struct linear_range
		exl3_tc_workspace; /* exl3_tc_moe: the tensor-core MoE workspace of row_capacity rows */
	struct linear_range
		exl3_tcp_workspace; /* exl3_tc_moe: the prefill form's, chunks of 65-2,048 rows */
	struct linear_range
		exl3_tail_workspace; /* tail_rows: a tail program's TC MoE calls' (tail_overlay) */
	/* trunk_mma, steps of more than 8 rows: the _64 form's float32 partials [split][row][output], at most 64 rows of
	   V41_DATA_TRUNK_MMA_THIN_PARTIAL_VALUES (splits times outputs) */
	struct linear_range trunk_mma_partials;
	struct linear_range
		trunk_mma_set_partials; /* tail_sets: [split][tail_rows][output], at most
							       V41_DATA_TRUNK_MMA_PARTIAL_VALUES a row */
	struct linear_range
		router_partials; /* router_mma: [splits][64 rows, router_mma_wide 1,024][384] float32 */
	struct bb_exl3_output_shard_plan exl3_shards[MODEL_LAYERS];
	uint64_t tensor_payload_byte_count;
	uint64_t aligned_weight_byte_count;
	uint64_t base_state_byte_count;
	uint64_t communication_byte_count;
	uint64_t draft_embedding_byte_count; /* already included in base_state_byte_count */
	uint64_t external_pool_sources;
	uint64_t external_key_sources;
	uint64_t external_selection_groups;
	uint32_t job_count;
	uint32_t channel_count;
	uint32_t channel_rows;
	uint32_t ring_rows; /* rows one ring gather carries: 8, or a BATCH step's most on the BF16 recipe (one ring a
			       sublayer over all its rows: the same element sums, so the same bits) */
	bool external_draft_embedding;
	bool vision_excluded;
};

/* A BATCH verifier's draft area (model_seam.h's BATCH; positions P): its draft step area, views, results, Markov
 * rows, collapse rows and logits, each from a multiple of 256 bytes. */
#define V41_DATA_BATCH_DRAFT_ALIGN 256u
static inline uint64_t v41_batch_aligned(uint64_t bytes)
{
	return (bytes + V41_DATA_BATCH_DRAFT_ALIGN - 1u) /
	       V41_DATA_BATCH_DRAFT_ALIGN * V41_DATA_BATCH_DRAFT_ALIGN;
}
enum v41_batch_draft_part {
	V41_DATA_BATCH_DRAFT_STEP,
	V41_DATA_BATCH_DRAFT_VIEWS,
	V41_DATA_BATCH_DRAFT_RESULTS,
	V41_DATA_BATCH_DRAFT_MARKOV,
	V41_DATA_BATCH_DRAFT_COLLAPSE,
	V41_DATA_BATCH_DRAFT_LOGITS,
	V41_DATA_BATCH_DRAFT_PARTS
};
static inline uint64_t v41_batch_draft_bytes(uint32_t part, uint32_t positions)
{
	static const uint64_t row_bytes[V41_DATA_BATCH_DRAFT_PARTS] = {
		STEP_ROW_BYTES_VALUE,	 STEP_DRAFT_VIEW_BYTES,
		STEP_DRAFT_RESULT_BYTES, MODEL_HALVES(MODEL_MARKOV_RANK),
		MODEL_HIDDEN_ROW_BYTES,	 MODEL_LOGITS_ROW_BYTES
	};
	return (part == V41_DATA_BATCH_DRAFT_STEP ? STEP_BLOCK_BYTES_VALUE :
						    0u) +
	       positions * row_bytes[part];
}
static inline uint64_t v41_batch_draft_offset(uint32_t part, uint32_t positions)
{
	uint64_t offset = 0;
	for (uint32_t before = 0; before < part; before++)
		offset += v41_batch_aligned(
			v41_batch_draft_bytes(before, positions));
	return offset;
}

/* batch_draft_rows: the parts of the area of one draft of every verifier's rows (at most 8 verifiers x 7 positions),
 * then (V41_DATA_BATCH_ROWS_HIDDEN) the normed rows position-major, 8 slots a position (draft_records.h's BATCHED;
 * its views, result areas, Markov rows and logits fit the parts sized for 56 rows). */
#define V41_DATA_BATCH_DRAFT_ROWS 56u
#define V41_DATA_BATCH_DRAFT_SLABS \
	(V41_DATA_BATCH_DRAFT_ROWS / LINEAR_MAXIMUM_ROWS)
#define V41_DATA_BATCH_ROWS_HIDDEN V41_DATA_BATCH_DRAFT_PARTS
#define V41_DATA_BATCH_ROWS_PARTS (V41_DATA_BATCH_DRAFT_PARTS + 1u)
static inline uint64_t v41_batch_rows_bytes(uint32_t part)
{
	/* the slab blocks, then one block of all the rows (draft_records.h's WIDE_STEP) */
	const uint64_t slab_block = STEP_BLOCK_BYTES_VALUE +
				    LINEAR_MAXIMUM_ROWS * STEP_ROW_BYTES_VALUE;
	const uint64_t all_rows =
		STEP_BLOCK_BYTES_VALUE +
		(uint64_t)V41_DATA_BATCH_DRAFT_ROWS * STEP_ROW_BYTES_VALUE;
	return part == V41_DATA_BATCH_DRAFT_STEP ?
		       (uint64_t)V41_DATA_BATCH_DRAFT_SLABS * slab_block +
			       all_rows :
	       part == V41_DATA_BATCH_ROWS_HIDDEN ?
		       (uint64_t)V41_DATA_BATCH_DRAFT_ROWS *
			       MODEL_HIDDEN_ROW_BYTES :
		       v41_batch_draft_bytes(part, V41_DATA_BATCH_DRAFT_ROWS);
}
static inline uint64_t v41_batch_rows_offset(uint32_t part)
{
	uint64_t offset = 0;
	for (uint32_t before = 0; before < part; before++)
		offset += v41_batch_aligned(v41_batch_rows_bytes(before));
	return offset;
}

MUST_CHECK result_status
v41_tp2_stage_plan_create(const struct pack_record *pack,
			  const struct v41_tp2_stage_description *description,
			  struct v41_tp2_stage_plan **plan_out);
void v41_tp2_stage_plan_destroy(struct v41_tp2_stage_plan *plan);

#endif
