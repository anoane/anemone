/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_geometry.h: the model's numbers, and the layouts of the arrays several units share. C11 and CUDA, including of
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
 *                     is the plan's: the smallest power of two of at least DS41F_RING_SLOTS and the row
 *                     capacity + 127 (wide_prefill.md 2.2); this build's kernels take DS41F_RING_SLOTS alone
 *       pool          FP16 [session][context limit / q][512] of ratio q: entry e holds positions q e .. q e + q - 1,
 *                     written by the row at q e + q - 1, rotated at q e (REF:477-485, 751-761); a row at p sees the
 *                     entries below (p + 1) / q (REF:561-567)
 *       key tiles     [session][entries / 8][2,048 B]: key g of a tile at 256 g, its run t (64 B at 64 t) the FP16 of
 *                     dims 16 j + 2 t + {0, 1, 8, 9}, j = 0 .. 7: key e lies at 256 e
 *       selection     int32 [row][512]: entry indices ascending, then -1 (REF:578-580, without the window's offset)
 *       maxima        FP16 [row][ceil(entries / 64)]: maximum m the largest score of the row's entries 64 m .. 64 m +
 *                     63 in ds41f_index_scan_records.h's order of scores, -inf where none is visible (wide_prefill.md
 *                     2.3)
 *       candidates    uint32 [row][2,048]: the blocks of 8 entries (block b: entries 8 b .. 8 b + 7, a key tile) the
 *                     row's candidate stage keeps (REF:583-610), ascending, then DS41F_NO_CANDIDATE
 *       sorted pairs, runs, expert tiles: a layer's routed experts at a wide step (expert_tile_records.h)
 *       router scores float32 [row][384]: the product, before the score function (REF:811)
 *       logits        float32 [row][129,280] (REF:1012), and the sampler's weights alike: a thin step's row r at r; of
 *                     a wide, cone or full step its flagged rows only, the k-th at k, at most
 *                     MODEL_LOGITS_MOST_ROWS (the step driver refuses a step that flags more); this
 *                     build's head and sampler run thin steps alone
 *       expert offsets uint64 [384][gate, up, down] of a layer: the arena offsets of each expert's matrices (ours)
 *       draft taps    bfloat16 [row][3][5,120]: the stream means at the inputs of layers 37, 38 and 39 (REF:1269)
 *       draft ring    a ring of the layout above, one per draft stage: the context row of position p, or the block
 *                     row the draft wrote there
 *   - The context limit is a multiple of DS41F_CONTEXT_GRANULE (a pool holds whole tiles) and at most
 *     DS41F_CONTEXT_MOST. Below DS41F_CONTEXT_WITHOUT_CANDIDATES the candidate stage keeps every
 *     block (REF:596-610). A plan's row capacity is 1 to STEP_WIDE_MOST_ROWS.
 *   - Wide steps (wide_prefill.md 2.2, 3): a thin step carries at most LINEAR_MAXIMUM_ROWS rows, a wide one
 *     (a chunk) STEP_WIDE_LEAST_ROWS or more. Of a wide step the attention middle (wq_b, the queries, the
 *     core) runs in slabs of MODEL_ATTENTION_SLAB_ROWS rows and the index scan with its selection in slabs of
 *     MODEL_SCAN_SLAB_ROWS: slab k is the view of the step's rows k S .. k S + S - 1 (model_seam.h), and a
 *     slab's arrays hold S rows, reused by every slab.
 *   - The cone (wide_prefill.md 2.1, 4): layer DS41F_DECODER_FIRST_LAYER + j computes a prompt's last 1 +
 *     (19 - j) 127 rows, DS41F_CONE_ROWS for j = 0; for j > 0 its window prepare takes the 127 rows before
 *     them too, which layer 20 + j - 1 computed. Layer 20's front (its pool entry, key and window) runs on every row.
 */
#ifndef DS41F_GEOMETRY_H
#define DS41F_GEOMETRY_H

#include <stdint.h>

#include "common/record_layout.h"
#include "engine/library/block_scaled_linear/linear_records.h"
#include "engine/program/step_records.h"

#define DS41F_LAYERS 40u /* CFG:5 n_layers */
#define DS41F_HIDDEN 5120u /* CFG:3 dim */
#define DS41F_VOCABULARY_VALUE 129280u /* CFG:2 vocab_size */
#define DS41F_STREAMS 4u /* CFG:40 hc_mult; REF:1258 */
#define DS41F_SITE_VECTORS \
	2u /* pre and post, a weight a stream each (REF:938) */
#define DS41F_MIXES 24u /* REF:938 mix_hc = (2 + hc_mult) * hc_mult */
#define DS41F_SINKHORN_ITERATIONS 20u /* CFG:41 hc_sinkhorn_iters; KER:450 */
#define DS41F_HYPER_EPSILON 1e-6f /* CFG:42 hc_eps; KER:427, 443, 448 */
#define DS41F_NORM_EPSILON \
	1e-20f /* CFG:23 norm_eps: RMS norms (REF:292), REF:359, REF:953 */
#define DS41F_HEADS_VALUE 64u /* CFG:13 n_heads */
#define DS41F_HEAD_DIMS \
	512u /* CFG:21 head_dim: also a window row, a pool entry */
#define DS41F_ROTATED_DIMS \
	64u /* CFG:22 rope_head_dim: the last of a head (REF:706, 772) */
#define DS41F_FIRST_ROTATED_DIM 448u
#define DS41F_QUERY_RANK 1280u /* CFG:20 q_lora_rank: qr */
#define DS41F_OUTPUT_GROUPS 8u /* CFG:24 o_groups (REF:785-787) */
#define DS41F_WINDOW 128u /* CFG:26 window_size */
#define DS41F_RING_SLOTS \
	4096u /* ours: a prefix may be kept 3,968 rows behind the reach */
#define DS41F_DECODER_FIRST_LAYER \
	20u /* the first of ratio 1 (CFG:52): every later layer reads its pool */
#define DS41F_CONE_ROWS 2414u /* 1 + 19 x 127: wide_prefill.md 2.1 */
/* The prefill cone: of a prompt row `behind` rows before a row that needs every decoder layer, the decoder layers
 * whose output of the row is read (layer 20 + j reads the window of its 127 rows before, so a row 1..127 behind needs
 * one layer fewer, and so on). */
/* the rows before its own a row's window reads */
#define DS41F_WINDOW_BEHIND (DS41F_WINDOW - 1u)
#define DS41F_CONE_LAYERS(behind)                                      \
	(((behind) + DS41F_WINDOW_BEHIND - 1u) / DS41F_WINDOW_BEHIND < \
			 DS41F_LAYERS - DS41F_DECODER_FIRST_LAYER ?    \
		 DS41F_LAYERS - DS41F_DECODER_FIRST_LAYER -            \
			 ((behind) + DS41F_WINDOW_BEHIND - 1u) /       \
				 DS41F_WINDOW_BEHIND :                 \
		 0u)
#define DS41F_INDEX_HEADS 32u /* CFG:34 index_n_heads */
#define DS41F_INDEX_HEAD_DIMS 128u /* CFG:35 index_head_dim */
#define DS41F_SELECTED 512u /* CFG:36 index_topk */
#define DS41F_POOLED_RATIO 2u /* CFG:52 compress_ratios: layers 2 to 19 */
#define DS41F_DIRECT_RATIO \
	1u /* layers 20 to 39; 0 (the window alone): layers 0 and 1 */
#define DS41F_KEYS_PER_TILE 8u /* ours: the index scan's tile */
#define DS41F_CONTEXT_WITHOUT_CANDIDATES \
	16384u /* CFG:38 x CFG:39: 2,048 blocks of 8 entries */
#define DS41F_CONTEXT_MOST \
	1048576u /* CFG:29 x CFG:31: original_seq_len times rope_factor */
#define DS41F_CONTEXT_GRANULE 16u /* ours */
#define DS41F_CANDIDATE_SOURCE 20u /* CFG:37 candidate_source_layer */
#define DS41F_CANDIDATE_BLOCKS 2048u /* CFG:38 candidate_topk_blocks */
#define DS41F_NO_CANDIDATE UINT32_MAX
#define DS41F_EXPERTS 384u /* CFG:14 n_routed_experts */
#define DS41F_EXPERTS_PER_ROW 6u /* CFG:16 n_activated_experts */
#define DS41F_EXPERT_MATRICES 3u /* gate, up, down (REF:836-838) */
#define DS41F_ROUTE_SCALE 1.5f /* CFG:18 route_scale (REF:826) */
#define DS41F_THETAS \
	2u /* CFG:30 rope_theta, CFG:51 compress_rope_theta (REF:680-687) */
#define DS41F_FREQUENCIES 32u /* words a theta: rotated dims / 2 (REF:376) */
#define DS41F_PAIR_VALUES \
	2u /* of a rotated pair: two dims, or its cos and sin (REF:397) */
#define DS41F_ROTATION_USES \
	3u /* ours: the angles a step's rows need (REF:542, 680-687, 756) */
#define DS41F_STREAMS_ROW_BYTES 40960u
#define DS41F_HIDDEN_ROW_BYTES 10240u
#define DS41F_SITE_ROW_BYTES 16u /* pre; post */
#define DS41F_COMB_ROW_BYTES 64u
#define DS41F_MIXES_ROW_BYTES 96u
#define DS41F_HEADS_ROW_BYTES 65536u /* queries; output */
#define DS41F_INDEX_QUERIES_ROW_BYTES 8192u
#define DS41F_INDEX_WEIGHTS_ROW_BYTES 64u
#define DS41F_ROTATION_ROW_BYTES 768u
#define DS41F_STATE_ROW_BYTES 1024u /* a window row; a pool entry */
#define DS41F_RING_SESSION_BYTES \
	((uint64_t)DS41F_RING_SLOTS * DS41F_STATE_ROW_BYTES)
#define DS41F_KEY_BYTES 256u
#define DS41F_TILE_BYTES 2048u
#define DS41F_SELECTION_ROW_BYTES 2048u
#define DS41F_CANDIDATES_ROW_BYTES 8192u
#define DS41F_ROUTER_ROW_BYTES 1536u
#define DS41F_LOGITS_ROW_BYTES 517120u
#define DS41F_EXPERT_OFFSETS_BYTES 9216u /* of a layer */
/* The draft (DSpark, the checkpoint's mtp.0 .. mtp.2): dspark_v1/DESIGN.md 5.1. */
#define DS41F_DRAFT_STAGES 3u /* CFG:6 n_mtp_layers */
#define DS41F_DRAFT_FIRST_LAYER \
	40u /* REF:1213-1214: the stages are layers 40, 41, 42 */
#define DS41F_DRAFT_TAPS 3u /* CFG:9 dspark_target_layer_ids */
#define DS41F_DRAFT_FIRST_TAP_LAYER 37u /* CFG:9: 37, 38, 39, consecutive */
#define DS41F_DRAFT_FEATURES 15360u /* REF:1116: main_proj's inputs, dim x 3 */
#define DS41F_DRAFT_TAP_ROW_BYTES 30720u
#define DS41F_DRAFT_BLOCK 5u /* CFG:7 dspark_block_size: R */
#define DS41F_DRAFT_NOISE_TOKEN 128799u /* CFG:8 dspark_noise_token_id */
#define DS41F_DRAFT_EXPERTS 128u /* CFG:11 dspark_n_routed_experts */
#define DS41F_DRAFT_EXPERTS_PER_ROW 3u /* CFG:12 dspark_n_activated_experts */
#define DS41F_MARKOV_RANK 256u /* CFG:10 dspark_markov_rank */
#define DS41F_DRAFT_CONFIDENCE_INPUTS 5376u /* REF:1121: dim + markov rank */

RECORD_HOLDS(DS41F_MIXES ==
	     (DS41F_SITE_VECTORS + DS41F_STREAMS) * DS41F_STREAMS);
RECORD_HOLDS(DS41F_FIRST_ROTATED_DIM + DS41F_ROTATED_DIMS == DS41F_HEAD_DIMS);
RECORD_HOLDS(DS41F_FREQUENCIES *DS41F_PAIR_VALUES == DS41F_ROTATED_DIMS);
RECORD_HOLDS(DS41F_RING_SLOTS >= DS41F_WINDOW + LINEAR_MAXIMUM_ROWS - 1u &&
	     (DS41F_RING_SLOTS & (DS41F_RING_SLOTS - 1u)) == 0);
RECORD_HOLDS(DS41F_CONTEXT_GRANULE ==
		     DS41F_POOLED_RATIO * DS41F_KEYS_PER_TILE &&
	     DS41F_CONTEXT_WITHOUT_CANDIDATES % DS41F_CONTEXT_GRANULE == 0 &&
	     DS41F_CONTEXT_MOST % DS41F_CONTEXT_GRANULE == 0);
/* a wide step's fewest rows, more than a thin launch's; the cone's first layer; the candidates, blocks of a key
 * tile */
RECORD_HOLDS(STEP_WIDE_LEAST_ROWS > LINEAR_MAXIMUM_ROWS);
RECORD_HOLDS(DS41F_CONE_ROWS ==
		     1u + (DS41F_LAYERS - 1u - DS41F_DECODER_FIRST_LAYER) *
				     (DS41F_WINDOW - 1u) &&
	     DS41F_CONE_ROWS <= STEP_WIDE_MOST_ROWS);
RECORD_HOLDS(DS41F_CANDIDATE_BLOCKS *DS41F_KEYS_PER_TILE ==
		     DS41F_CONTEXT_WITHOUT_CANDIDATES &&
	     DS41F_CANDIDATES_ROW_BYTES ==
		     sizeof(uint32_t) * DS41F_CANDIDATE_BLOCKS);
RECORD_HOLDS(DS41F_HIDDEN_ROW_BYTES == RECORD_HALVES(DS41F_HIDDEN) &&
	     DS41F_STREAMS_ROW_BYTES == DS41F_STREAMS * DS41F_HIDDEN_ROW_BYTES);
RECORD_HOLDS(DS41F_SITE_ROW_BYTES == RECORD_FLOATS(DS41F_STREAMS) &&
	     DS41F_COMB_ROW_BYTES == DS41F_STREAMS * DS41F_SITE_ROW_BYTES &&
	     DS41F_MIXES_ROW_BYTES == RECORD_FLOATS(DS41F_MIXES));
RECORD_HOLDS(DS41F_STATE_ROW_BYTES == RECORD_HALVES(DS41F_HEAD_DIMS) &&
	     DS41F_HEADS_ROW_BYTES ==
		     DS41F_HEADS_VALUE * DS41F_STATE_ROW_BYTES);
RECORD_HOLDS(DS41F_KEY_BYTES == RECORD_HALVES(DS41F_INDEX_HEAD_DIMS) &&
	     DS41F_TILE_BYTES == DS41F_KEYS_PER_TILE * DS41F_KEY_BYTES &&
	     DS41F_INDEX_WEIGHTS_ROW_BYTES ==
		     RECORD_HALVES(DS41F_INDEX_HEADS) &&
	     DS41F_INDEX_QUERIES_ROW_BYTES ==
		     DS41F_INDEX_HEADS * DS41F_KEY_BYTES);
RECORD_HOLDS(DS41F_ROTATION_ROW_BYTES ==
	     DS41F_ROTATION_USES * RECORD_FLOATS(DS41F_ROTATED_DIMS));
RECORD_HOLDS(DS41F_SELECTION_ROW_BYTES == sizeof(int32_t) * DS41F_SELECTED &&
	     DS41F_ROUTER_ROW_BYTES == RECORD_FLOATS(DS41F_EXPERTS) &&
	     DS41F_LOGITS_ROW_BYTES == RECORD_FLOATS(DS41F_VOCABULARY_VALUE));
RECORD_HOLDS(DS41F_EXPERT_OFFSETS_BYTES ==
	     sizeof(uint64_t) * DS41F_EXPERTS * DS41F_EXPERT_MATRICES);
RECORD_HOLDS(DS41F_DRAFT_FEATURES == DS41F_DRAFT_TAPS * DS41F_HIDDEN &&
	     DS41F_DRAFT_TAP_ROW_BYTES == RECORD_HALVES(DS41F_DRAFT_FEATURES) &&
	     DS41F_DRAFT_FIRST_TAP_LAYER + DS41F_DRAFT_TAPS == DS41F_LAYERS &&
	     DS41F_DRAFT_FIRST_LAYER == DS41F_LAYERS);
RECORD_HOLDS(DS41F_DRAFT_CONFIDENCE_INPUTS ==
		     DS41F_HIDDEN + DS41F_MARKOV_RANK &&
	     DS41F_DRAFT_EXPERTS <= DS41F_EXPERTS &&
	     DS41F_DRAFT_EXPERTS_PER_ROW <= DS41F_EXPERTS_PER_ROW &&
	     DS41F_DRAFT_NOISE_TOKEN < DS41F_VOCABULARY_VALUE &&
	     DS41F_DRAFT_BLOCK < LINEAR_MAXIMUM_ROWS);
/* a row's routes fit the expert producer's (linear_records.h) */
RECORD_HOLDS(DS41F_EXPERTS_PER_ROW <= EXPERT_MOST_ROUTES_PER_ROW);

#endif
