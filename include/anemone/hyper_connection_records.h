/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * hyper_connection_records.h: the records of the hyper-connection kernels (source/kernel_hyper_connection.cu). C11
 * and CUDA, including of the engine model_layout.h alone.
 *
 * Contract: linear_records.h's for a kernel, its record and its ranges; the arrays are model_layout.h's. Row r is
 * blockIdx.y and no row's bits depend on another's. Arithmetic is float32, each operation rounded (fmaf where
 * written), a bfloat16 rounded to nearest with a tie to even, a not-a-number to 0x7FC0. REF is the reference's
 * inference/model.py, KER its kernel.py:
 *   - hyper_mix_finish: pre, post and comb of the mixes m, scale s and base b, as the source the reference
 *     generates for KER:406-460 computes them: each m s + b one fmaf, the card's expf and IEEE division, eps
 *     MODEL_HYPER_EPSILON, a row's or a column's four values added as (x_0 + x_2) + (x_1 + x_3). Then the
 *     collapse below with the carried pre, which the site's pre then replaces.
 *   - hyper_final, and the collapse of a mix finish (REF:957-960, 288-293): of the carried pre p, h_d =
 *     bfloat16((((0 + p_0 X_0d) + p_1 X_1d) + p_2 X_2d) + p_3 X_3d); q, the sum of the h_d h_d in a fixed tree
 *     (thread t of HYPER_ROW_THREADS adds those of d = 8 t .. 8 t + 7 from +0, a warp its 32 sums and one
 *     warp the warps' sums, zeros behind them, by exchanges at distance 16, 8, 4, 2, 1); the output
 *     bfloat16(w_d (h_d rsqrtf(q float32(1 / 5,120) + MODEL_NORM_EPSILON))) of the norm's weight w (the
 *     reference's library takes a mean as its sum times that factor). hyper_final_kept, the draft's final,
 *     also writes h, the collapse before its norm (the draft's confidence reads it: REF:1147, 1158), to comb_offset
 *     (the final reads no comb).
 *   - hyper_draft_inputs (dspark_v1/DESIGN.md C1), mode HYPER_STREAM_MEAN: the collapse above with
 *     pre (1/4, 1/4, 1/4, 1/4), the mean of the four streams at a tap layer's input (REF:1269), written to columns
 *     column_offset .. column_offset + 5,119 of row r of an output of output_row_values columns; mode
 *     HYPER_ROW_NORM: the final's norm above of a bfloat16 row h by mtp.0.main_norm.weight (REF:1133).
 *   - hyper_post_entry (REF:962-966): stream k becomes bfloat16(post_k y_d + ((((0 + c_0k X_0d) + c_1k X_1d) +
 *     c_2k X_2d) + c_3k X_3d)) in place; with is_combined 1 (the combine), y_d is first bfloat16(s_d + y_d) of the
 *     routed experts' sums s and the shared expert's output y (REF:903-904).
 *   - hyper_embed (REF:1253-1260): the embedding's row of the step row's token id (step_records.h), below
 *     vocabulary_size by the step driver's promise, into the four streams; the carried pre (1, 0, 0, 0).
 *   - Launches: grid[1] the rows; grid[2], block[1] and block[2] 1; no dynamic shared memory. The mix finish has
 *     grid[0] 1 and block[0] HYPER_FINISH_THREADS, the finals and the draft inputs 1 and
 *     HYPER_ROW_THREADS, the others HYPER_PIECES and HYPER_PIECE_THREADS.
 */
#ifndef HYPER_CONNECTION_RECORDS_H
#define HYPER_CONNECTION_RECORDS_H

#include <stdint.h>

#include "anemone/model_layout.h"

#define KERNEL_FILE_HYPER_CONNECTION "kernel_hyper_connection"
#define KERNEL_HYPER_MIX_FINISH "hyper_mix_finish"
#define KERNEL_HYPER_FINAL "hyper_final"
#define KERNEL_HYPER_POST "hyper_post_entry"
#define KERNEL_HYPER_EMBED "hyper_embed"
#define KERNEL_HYPER_DRAFT_INPUTS "hyper_draft_inputs"
#define KERNEL_HYPER_FINAL_KEPT "hyper_final_kept"
#define HYPER_SCALES 3u /* pre, post, comb (KER:416) */
#define HYPER_STREAM_MEAN_PRE \
	0.25f /* each stream's weight in their mean (REF:1269): exact */
#define HYPER_VALUES_PER_THREAD 8u /* one 16-byte load of bfloat16 */
#define HYPER_ROW_THREADS (MODEL_HIDDEN / HYPER_VALUES_PER_THREAD)
#define HYPER_FINISH_THREADS (HYPER_ROW_THREADS + LINEAR_THREADS_PER_WARP)
#define HYPER_PIECE_THREADS \
	128u /* ours, untimed: a row on several multiprocessors */
#define HYPER_PIECES (HYPER_ROW_THREADS / HYPER_PIECE_THREADS)
#define HYPER_FINISH_ARGUMENT_BYTES 72u
#define HYPER_POST_ARGUMENT_BYTES 48u
#define HYPER_EMBED_ARGUMENT_BYTES 40u
#define HYPER_DRAFT_INPUTS_ARGUMENT_BYTES 40u

enum hyper_draft_input_mode { HYPER_STREAM_MEAN = 0, HYPER_ROW_NORM = 1 };

/* The final reads streams, pre and weight and writes hidden (the draft's final also the collapse at comb_offset); its
 * other offsets are not read. */
struct hyper_finish_arguments {
	uint64_t streams_offset;
	uint64_t pre_offset; /* the carried pre: read, then the mix finish writes the site's */
	uint64_t mixes_offset;
	uint64_t scale_offset; /* float32 [HYPER_SCALES] */
	uint64_t base_offset; /* float32 [MODEL_MIXES] */
	uint64_t post_offset; /* written */
	uint64_t comb_offset; /* written */
	uint64_t weight_offset; /* the norm's, bfloat16 [MODEL_HIDDEN] */
	uint64_t hidden_offset; /* written */
};

/* streams read and written; output: the sublayer's, or the shared expert's; sums: read with the combine alone,
 * float32 [row][MODEL_HIDDEN]. */
struct hyper_post_arguments {
	uint64_t streams_offset;
	uint64_t post_offset;
	uint64_t comb_offset;
	uint64_t output_offset;
	uint64_t sums_offset;
	uint32_t is_combined; /* 1: the FFN site's post, 0: the attention site's */
	uint32_t reserved_zero;
};

/* step: the step area; embedding: bfloat16 [vocabulary_size][MODEL_HIDDEN]; streams and pre written. */
struct hyper_embed_arguments {
	uint64_t step_offset;
	uint64_t embedding_offset;
	uint64_t streams_offset;
	uint64_t pre_offset;
	uint32_t vocabulary_size; /* what the checked modules hold a token id to */
	uint32_t reserved_zero;
};

/* input: STREAM_MEAN streams [row][4][5,120], ROW_NORM bfloat16 [row][5,120]; weight: ROW_NORM the norm's, bfloat16
 * [5,120], else 0; output: bfloat16 [row][output_row_values], written from column_offset on, 5,120 values a row. */
struct hyper_draft_inputs_arguments {
	uint64_t input_offset;
	uint64_t weight_offset;
	uint64_t output_offset;
	uint32_t output_row_values; /* MODEL_DRAFT_FEATURES (taps) or MODEL_HIDDEN */
	uint32_t column_offset; /* 5,120 t for tap t; 0 */
	uint32_t mode; /* enum hyper_draft_input_mode */
	uint32_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(
	sizeof(struct hyper_finish_arguments) == HYPER_FINISH_ARGUMENT_BYTES &&
	sizeof(struct hyper_post_arguments) == HYPER_POST_ARGUMENT_BYTES &&
	sizeof(struct hyper_embed_arguments) == HYPER_EMBED_ARGUMENT_BYTES &&
	sizeof(struct hyper_draft_inputs_arguments) ==
		HYPER_DRAFT_INPUTS_ARGUMENT_BYTES);
MODEL_LAYOUT_HOLDS(MODEL_HIDDEN % HYPER_VALUES_PER_THREAD == 0 &&
		   HYPER_ROW_THREADS % LINEAR_THREADS_PER_WARP == 0 &&
		   HYPER_ROW_THREADS % HYPER_PIECE_THREADS == 0);

#endif
