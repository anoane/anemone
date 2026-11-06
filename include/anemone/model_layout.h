/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_layout.h: the model's numbers, and the layouts of the arrays several units share. C11 and CUDA, including of
 * the engine linear_records.h alone: for the kernel files, the host code that prepares their launches, tests.
 *
 * Contract:
 *   - Each number is the reference's: CFG is its inference/config.json, REF its inference/model.py, KER its
 *     kernel.py, a number after a colon a line. "Ours" marks the engine's own choice.
 *   - An array is little-endian, its rows one after another from its offset (the program names the offsets:
 *     model_seam.h); FP16 is IEEE binary16. A state array ([session]...) keeps session s's part at its offset plus
 *     s times the bytes of one session's part, nothing between: its records carry that stride.
 *       streams       bfloat16 [row][4][5,120]: the residual's four copies between sites (REF:966 type_as)
 *       pre, post     float32 [row][4]; comb float32 [row][4][4]: comb[j][k] weighs stream j in output k (REF:965)
 *       mixes         float32 [row][24]: pre 0 .. 3, post 4 .. 7, comb[j][k] at 8 + 4 j + k (KER:427-431)
 *       hidden        bfloat16 [row][5,120]: a sublayer's normed input, and its output
 *       queries       FP16 [row][64][512]: the main queries, the last 64 dims of each head rotated (REF:771-772)
 *       output        bfloat16 [row][64][512]: the attention output; group g of wo_a reads heads 8 g .. 8 g + 7
 *       index queries FP16 [row][8,192 B]: [head tile h][step j][lane 4 g + t][register r], the halves of head
 *                     16 h + g + 8 (r % 2) at dims 16 j + 2 t + 8 (r / 2) + {0, 1}
 *       index weights bfloat16 [row][32]: the weights projection times 2^-6 (REF:555: 128^-0.5 * 32^-0.5)
 *       rotation      float32 [row][3 uses][32][cos, sin]: use 0 theta 0 at position p, 1 theta 1 at p, 2 theta 1
 *                     at p - 1 (a 2-to-1 entry: REF:542, 756); word i turns dims 2 i and 2 i + 1 of a vector's last
 *                     64 as (real, imaginary) (REF:397)
 *       frequencies   float32 [2 thetas][32] (REF:376-386): theta 0 CFG:30 without YaRN (layers 0 and 1), theta 1
 *                     CFG:51 with YaRN (layers 2 to 39: REF:680-687); an angle is float32(p) times a word (REF:388)
 *       ring          FP16 [session][slots][512]: the window row of position p in slot p mod slots; a row at p reads
 *                     positions p - 127 .. p, none below 0, so no row of a step may write a slot another reads. slots
 *                     is the plan's: the smallest power of two of at least MODEL_RING_SLOTS and the row
 *                     capacity + 127 (wide_prefill.md 2.2); this build's kernels take MODEL_RING_SLOTS alone
 *       pool          FP16 [session][context limit / q][512] of ratio q: entry e holds positions q e .. q e + q - 1,
 *                     written by the row at q e + q - 1, rotated at q e (REF:477-485, 751-761); a row at p sees the
 *                     entries below (p + 1) / q (REF:561-567)
 *       key tiles     [session][entries / 8][2,048 B]: key g of a tile at 256 g, its run t (64 B at 64 t) the FP16 of
 *                     dims 16 j + 2 t + {0, 1, 8, 9}, j = 0 .. 7: key e lies at 256 e
 *       selection     int32 [row][512]: entry indices ascending, then -1 (REF:578-580, without the window's offset)
 *       maxima        FP16 [row][ceil(entries / 64)]: maximum m the largest score of the row's entries 64 m .. 64 m +
 *                     63 in index_scan_records.h's order of scores, -inf where none is visible (wide_prefill.md 2.3)
 *       candidates    uint32 [row][2,048]: the blocks of 8 entries (block b: entries 8 b .. 8 b + 7, a key tile) the
 *                     row's candidate stage keeps (REF:583-610), ascending, then MODEL_NO_CANDIDATE
 *       sorted pairs, runs, expert tiles: a layer's routed experts at a wide step (wide_linear_records.h)
 *       router scores float32 [row][384]: the product, before the score function (REF:811)
 *       logits        float32 [row][129,280] (REF:1012), and the sampler's weights alike: a thin step's row r at r; of
 *                     a wide, cone or full step its flagged rows only, the k-th at k, at most
 *                     MODEL_LOGITS_MOST_ROWS (the step driver refuses a step that flags more); this
 *                     build's head and sampler run thin steps alone
 *       expert offsets uint64 [384][gate, up, down] of a layer: the arena offsets of each expert's matrices (ours)
 *       draft taps    bfloat16 [row][3][5,120]: the stream means at the inputs of layers 37, 38 and 39 (REF:1269)
 *       draft ring    a ring of the layout above, one per draft stage: the context row of position p, or the block
 *                     row the draft wrote there
 *   - The context limit is a multiple of MODEL_CONTEXT_GRANULE (a pool holds whole tiles) and at most
 *     MODEL_CONTEXT_MOST. Below MODEL_CONTEXT_WITHOUT_CANDIDATES the candidate stage keeps every
 *     block (REF:596-610). A plan's row capacity is 1 to MODEL_WIDE_MOST_ROWS.
 *   - Wide steps (wide_prefill.md 2.2, 3): a thin step carries at most LINEAR_MAXIMUM_ROWS rows, a wide one
 *     (a chunk) MODEL_WIDE_LEAST_ROWS or more. Of a wide step the attention middle (wq_b, the queries, the
 *     core) runs in slabs of MODEL_ATTENTION_SLAB_ROWS rows and the index scan with its selection in slabs of
 *     MODEL_SCAN_SLAB_ROWS: slab k is the view of the step's rows k S .. k S + S - 1 (model_seam.h), and a
 *     slab's arrays hold S rows, reused by every slab.
 *   - The cone (wide_prefill.md 2.1, 4): layer MODEL_DECODER_FIRST_LAYER + j computes a prompt's last 1 +
 *     (19 - j) 127 rows, MODEL_CONE_ROWS for j = 0; for j > 0 its window prepare takes the 127 rows before
 *     them too, which layer 20 + j - 1 computed. Layer 20's front (its pool entry, key and window) runs on every row.
 */
#ifndef MODEL_LAYOUT_H
#define MODEL_LAYOUT_H

#include <stdint.h>

#include "anemone/linear_records.h"

#define MODEL_LAYERS 40u /* CFG:5 n_layers */
#define MODEL_HIDDEN 5120u /* CFG:3 dim */
#define MODEL_VOCABULARY_VALUE 129280u /* CFG:2 vocab_size */
#define MODEL_STREAMS 4u /* CFG:40 hc_mult; REF:1258 */
#define MODEL_SITE_VECTORS \
	2u /* pre and post, a weight a stream each (REF:938) */
#define MODEL_MIXES 24u /* REF:938 mix_hc = (2 + hc_mult) * hc_mult */
#define MODEL_SINKHORN_ITERATIONS 20u /* CFG:41 hc_sinkhorn_iters; KER:450 */
#define MODEL_HYPER_EPSILON 1e-6f /* CFG:42 hc_eps; KER:427, 443, 448 */
#define MODEL_NORM_EPSILON \
	1e-20f /* CFG:23 norm_eps: RMS norms (REF:292), REF:359, REF:953 */
#define MODEL_HEADS_VALUE 64u /* CFG:13 n_heads */
#define MODEL_HEAD_DIMS \
	512u /* CFG:21 head_dim: also a window row, a pool entry */
#define MODEL_ROTATED_DIMS \
	64u /* CFG:22 rope_head_dim: the last of a head (REF:706, 772) */
#define MODEL_FIRST_ROTATED_DIM 448u
#define MODEL_QUERY_RANK 1280u /* CFG:20 q_lora_rank: qr */
#define MODEL_OUTPUT_GROUPS 8u /* CFG:24 o_groups (REF:785-787) */
#define MODEL_WINDOW 128u /* CFG:26 window_size */
#define MODEL_RING_SLOTS \
	4096u /* ours: a prefix may be kept 3,968 rows behind the reach */
#define MODEL_RING_MOST_SLOTS \
	8192u /* those of a plan of MODEL_WIDE_MOST_ROWS rows */
#define MODEL_WIDE_MOST_ROWS 4096u /* ours: a chunk (wide_prefill.md 2.2) */
#define MODEL_WIDE_LEAST_ROWS \
	64u /* ours: a step of fewer rows is thin (wide_prefill.md 3) */
#define MODEL_ATTENTION_SLAB_ROWS 1024u /* ours: wide_prefill.md 2.2, 6.1 */
#define MODEL_SCAN_SLAB_ROWS 256u /* ours */
#define MODEL_DECODER_FIRST_LAYER \
	20u /* the first of ratio 1 (CFG:52): every later layer reads its pool */
#define MODEL_CONE_ROWS 2414u /* 1 + 19 x 127: wide_prefill.md 2.1 */
/* The prefill cone: of a prompt row `behind` rows before a row that needs every decoder layer, the decoder layers
 * whose output of the row is read (layer 20 + j reads the window of its 127 rows before, so a row 1..127 behind needs
 * one layer fewer, and so on). */
/* engine.c's exact plan: a full chunk of at least so many rows lies before a tail of at least as many */
#define MODEL_FULL_CONE_LEAST_ROWS 256u
#define MODEL_CONE_LAYERS(behind)                                   \
	(((behind) + MODEL_WINDOW - 2u) / (MODEL_WINDOW - 1u) <     \
			 MODEL_LAYERS - MODEL_DECODER_FIRST_LAYER ? \
		 MODEL_LAYERS - MODEL_DECODER_FIRST_LAYER -         \
			 ((behind) + MODEL_WINDOW - 2u) /           \
				 (MODEL_WINDOW - 1u) :              \
		 0u)
#define MODEL_LOGITS_MOST_ROWS \
	128u /* ours: a gate's flagged rows (wide_prefill.md 2.1, 8) */
#define MODEL_INDEX_HEADS 32u /* CFG:34 index_n_heads */
#define MODEL_INDEX_HEAD_DIMS 128u /* CFG:35 index_head_dim */
#define MODEL_SELECTED 512u /* CFG:36 index_topk */
#define MODEL_POOLED_RATIO 2u /* CFG:52 compress_ratios: layers 2 to 19 */
#define MODEL_DIRECT_RATIO \
	1u /* layers 20 to 39; 0 (the window alone): layers 0 and 1 */
#define MODEL_KEYS_PER_TILE 8u /* ours: the index scan's tile */
#define MODEL_CONTEXT_WITHOUT_CANDIDATES \
	16384u /* CFG:38 x CFG:39: 2,048 blocks of 8 entries */
#define MODEL_CONTEXT_MOST \
	1048576u /* CFG:29 x CFG:31: original_seq_len times rope_factor */
#define MODEL_CONTEXT_GRANULE 16u /* ours */
#define MODEL_CANDIDATE_SOURCE 20u /* CFG:37 candidate_source_layer */
#define MODEL_CANDIDATE_BLOCKS 2048u /* CFG:38 candidate_topk_blocks */
#define MODEL_CANDIDATE_BLOCK_ENTRIES 8u /* CFG:39 candidate_block_size */
#define MODEL_NO_CANDIDATE UINT32_MAX
#define MODEL_ENTRIES_PER_MAXIMUM 64u /* ours: wide_prefill.md 2.3 */
#define MODEL_EXPERTS 384u /* CFG:14 n_routed_experts */
#define MODEL_EXPERTS_PER_ROW 6u /* CFG:16 n_activated_experts */
#define MODEL_EXPERT_MATRICES 3u /* gate, up, down (REF:836-838) */
#define MODEL_ROUTE_SCALE 1.5f /* CFG:18 route_scale (REF:826) */
#define MODEL_THETAS \
	2u /* CFG:30 rope_theta, CFG:51 compress_rope_theta (REF:680-687) */
#define MODEL_FREQUENCIES 32u /* words a theta: rotated dims / 2 (REF:376) */
#define MODEL_PAIR_VALUES \
	2u /* of a rotated pair: two dims, or its cos and sin (REF:397) */
#define MODEL_ROTATION_USES \
	3u /* ours: the angles a step's rows need (REF:542, 680-687, 756) */
#define MODEL_STREAMS_ROW_BYTES 40960u
#define MODEL_HIDDEN_ROW_BYTES 10240u
#define MODEL_SITE_ROW_BYTES 16u /* pre; post */
#define MODEL_COMB_ROW_BYTES 64u
#define MODEL_MIXES_ROW_BYTES 96u
#define MODEL_HEADS_ROW_BYTES 65536u /* queries; output */
#define MODEL_INDEX_QUERIES_ROW_BYTES 8192u
#define MODEL_INDEX_WEIGHTS_ROW_BYTES 64u
#define MODEL_ROTATION_ROW_BYTES 768u
#define MODEL_STATE_ROW_BYTES 1024u /* a window row; a pool entry */
#define MODEL_KEY_BYTES 256u
#define MODEL_TILE_BYTES 2048u
#define MODEL_SELECTION_ROW_BYTES 2048u
#define MODEL_CANDIDATES_ROW_BYTES 8192u
#define MODEL_ROUTER_ROW_BYTES 1536u
#define MODEL_LOGITS_ROW_BYTES 517120u
#define MODEL_EXPERT_OFFSETS_BYTES 9216u /* of a layer */
/* The draft (DSpark, the checkpoint's mtp.0 .. mtp.2): dspark_v1/DESIGN.md 5.1. */
#define MODEL_DRAFT_STAGES 3u /* CFG:6 n_mtp_layers */
#define MODEL_DRAFT_FIRST_LAYER \
	40u /* REF:1213-1214: the stages are layers 40, 41, 42 */
#define MODEL_DRAFT_TAPS 3u /* CFG:9 dspark_target_layer_ids */
#define MODEL_DRAFT_FIRST_TAP_LAYER 37u /* CFG:9: 37, 38, 39, consecutive */
#define MODEL_DRAFT_FEATURES 15360u /* REF:1116: main_proj's inputs, dim x 3 */
#define MODEL_DRAFT_TAP_ROW_BYTES 30720u
#define MODEL_DRAFT_BLOCK 5u /* CFG:7 dspark_block_size: R */
#define MODEL_DRAFT_NOISE_TOKEN 128799u /* CFG:8 dspark_noise_token_id */
#define MODEL_DRAFT_EXPERTS 128u /* CFG:11 dspark_n_routed_experts */
#define MODEL_DRAFT_EXPERTS_PER_ROW 3u /* CFG:12 dspark_n_activated_experts */
#define MODEL_MARKOV_RANK 256u /* CFG:10 dspark_markov_rank */
#define MODEL_DRAFT_CONFIDENCE_INPUTS 5376u /* REF:1121: dim + markov rank */

#ifdef __cplusplus
#define MODEL_LAYOUT_HOLDS(condition) static_assert(condition, #condition)
#else
#define MODEL_LAYOUT_HOLDS(condition) _Static_assert(condition, #condition)
#endif
#define MODEL_HALVES(count) (sizeof(uint16_t) * (count))
#define MODEL_FLOATS(count) (sizeof(float) * (count))
MODEL_LAYOUT_HOLDS(MODEL_MIXES ==
		   (MODEL_SITE_VECTORS + MODEL_STREAMS) * MODEL_STREAMS);
MODEL_LAYOUT_HOLDS(MODEL_FIRST_ROTATED_DIM + MODEL_ROTATED_DIMS ==
		   MODEL_HEAD_DIMS);
MODEL_LAYOUT_HOLDS(MODEL_FREQUENCIES *MODEL_PAIR_VALUES == MODEL_ROTATED_DIMS);
MODEL_LAYOUT_HOLDS(MODEL_RING_SLOTS >=
			   MODEL_WINDOW + LINEAR_MAXIMUM_ROWS - 1u &&
		   (MODEL_RING_SLOTS & (MODEL_RING_SLOTS - 1u)) == 0);
MODEL_LAYOUT_HOLDS(MODEL_CONTEXT_GRANULE ==
			   MODEL_POOLED_RATIO * MODEL_KEYS_PER_TILE &&
		   MODEL_CONTEXT_WITHOUT_CANDIDATES % MODEL_CONTEXT_GRANULE ==
			   0 &&
		   MODEL_CONTEXT_MOST % MODEL_CONTEXT_GRANULE == 0);
/* the most slots, the fewest that hold a wide plan's rows; slabs that cut a chunk whole, logits rows that hold a thin
 * step's; the cone's first layer; the candidates, blocks of a key tile */
MODEL_LAYOUT_HOLDS((MODEL_RING_MOST_SLOTS & (MODEL_RING_MOST_SLOTS - 1u)) ==
			   0 &&
		   MODEL_RING_MOST_SLOTS >=
			   MODEL_WINDOW - 1u + MODEL_WIDE_MOST_ROWS &&
		   (MODEL_RING_MOST_SLOTS >> 1u) <
			   MODEL_WINDOW - 1u + MODEL_WIDE_MOST_ROWS);
MODEL_LAYOUT_HOLDS(MODEL_WIDE_LEAST_ROWS > LINEAR_MAXIMUM_ROWS &&
		   MODEL_LOGITS_MOST_ROWS >= LINEAR_MAXIMUM_ROWS &&
		   MODEL_WIDE_MOST_ROWS % MODEL_ATTENTION_SLAB_ROWS == 0 &&
		   MODEL_ATTENTION_SLAB_ROWS % MODEL_SCAN_SLAB_ROWS == 0);
MODEL_LAYOUT_HOLDS(MODEL_CONE_ROWS == 1u + (MODEL_LAYERS - 1u -
					    MODEL_DECODER_FIRST_LAYER) *
						      (MODEL_WINDOW - 1u) &&
		   MODEL_CONE_ROWS <= MODEL_WIDE_MOST_ROWS);
MODEL_LAYOUT_HOLDS(MODEL_CANDIDATE_BLOCKS *MODEL_CANDIDATE_BLOCK_ENTRIES ==
			   MODEL_CONTEXT_WITHOUT_CANDIDATES &&
		   MODEL_CANDIDATE_BLOCK_ENTRIES == MODEL_KEYS_PER_TILE &&
		   MODEL_ENTRIES_PER_MAXIMUM % MODEL_KEYS_PER_TILE == 0 &&
		   MODEL_CANDIDATES_ROW_BYTES ==
			   sizeof(uint32_t) * MODEL_CANDIDATE_BLOCKS);
MODEL_LAYOUT_HOLDS(MODEL_HIDDEN_ROW_BYTES == MODEL_HALVES(MODEL_HIDDEN) &&
		   MODEL_STREAMS_ROW_BYTES ==
			   MODEL_STREAMS * MODEL_HIDDEN_ROW_BYTES);
MODEL_LAYOUT_HOLDS(MODEL_SITE_ROW_BYTES == MODEL_FLOATS(MODEL_STREAMS) &&
		   MODEL_COMB_ROW_BYTES ==
			   MODEL_STREAMS * MODEL_SITE_ROW_BYTES &&
		   MODEL_MIXES_ROW_BYTES == MODEL_FLOATS(MODEL_MIXES));
MODEL_LAYOUT_HOLDS(MODEL_STATE_ROW_BYTES == MODEL_HALVES(MODEL_HEAD_DIMS) &&
		   MODEL_HEADS_ROW_BYTES ==
			   MODEL_HEADS_VALUE * MODEL_STATE_ROW_BYTES);
MODEL_LAYOUT_HOLDS(MODEL_KEY_BYTES == MODEL_HALVES(MODEL_INDEX_HEAD_DIMS) &&
		   MODEL_TILE_BYTES == MODEL_KEYS_PER_TILE * MODEL_KEY_BYTES &&
		   MODEL_INDEX_WEIGHTS_ROW_BYTES ==
			   MODEL_HALVES(MODEL_INDEX_HEADS) &&
		   MODEL_INDEX_QUERIES_ROW_BYTES ==
			   MODEL_INDEX_HEADS * MODEL_KEY_BYTES);
MODEL_LAYOUT_HOLDS(MODEL_ROTATION_ROW_BYTES ==
		   MODEL_ROTATION_USES * MODEL_FLOATS(MODEL_ROTATED_DIMS));
MODEL_LAYOUT_HOLDS(MODEL_SELECTION_ROW_BYTES ==
			   sizeof(int32_t) * MODEL_SELECTED &&
		   MODEL_ROUTER_ROW_BYTES == MODEL_FLOATS(MODEL_EXPERTS) &&
		   MODEL_LOGITS_ROW_BYTES ==
			   MODEL_FLOATS(MODEL_VOCABULARY_VALUE));
MODEL_LAYOUT_HOLDS(MODEL_EXPERT_OFFSETS_BYTES ==
		   sizeof(uint64_t) * MODEL_EXPERTS * MODEL_EXPERT_MATRICES);
MODEL_LAYOUT_HOLDS(MODEL_DRAFT_FEATURES == MODEL_DRAFT_TAPS * MODEL_HIDDEN &&
		   MODEL_DRAFT_TAP_ROW_BYTES ==
			   MODEL_HALVES(MODEL_DRAFT_FEATURES) &&
		   MODEL_DRAFT_FIRST_TAP_LAYER + MODEL_DRAFT_TAPS ==
			   MODEL_LAYERS &&
		   MODEL_DRAFT_FIRST_LAYER == MODEL_LAYERS);
MODEL_LAYOUT_HOLDS(MODEL_DRAFT_CONFIDENCE_INPUTS ==
			   MODEL_HIDDEN + MODEL_MARKOV_RANK &&
		   MODEL_DRAFT_EXPERTS <= MODEL_EXPERTS &&
		   MODEL_DRAFT_EXPERTS_PER_ROW <= MODEL_EXPERTS_PER_ROW &&
		   MODEL_DRAFT_NOISE_TOKEN < MODEL_VOCABULARY_VALUE &&
		   MODEL_DRAFT_BLOCK < LINEAR_MAXIMUM_ROWS);

#endif
