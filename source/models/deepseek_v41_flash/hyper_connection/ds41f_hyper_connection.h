/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_hyper_connection.h: the hyper-connection sites on the cards (first_token_plan.md 2.2): the launches of the
 * kernels of ds41f_hyper_connection_records.h, each prepared once at load from a description that names every arena
 * offset. A site's mix project, its product with hc_attn_fn or hc_ffn_fn, is a launch of dense_linear.h.
 *
 * Contract: linear.h's for the kernels and the prepare functions (once at load, arithmetic on a description, one
 * error line for a refusal, the launches untouched). Refused with STATUS_INVALID_ARGUMENT: a NULL argument; a
 * kernel it launches not prepared; a reserved field that is not zero; a row_count of 0 or above
 * STEP_WIDE_MOST_ROWS (the final collapse: LINEAR_MAXIMUM_ROWS, the head's steps are thin); a
 * vocabulary_size of 0; a range that breaks a rule of prepare_checks.h for the bytes its comment gives, of row_count
 * rows of ds41f_geometry.h's arrays ("exactly": a tensor of the pack, which holds exactly its bytes). A launch takes
 * its rows on grid[1] and a row's bits are its own (ds41f_hyper_connection_records.h): a launch of a chunk's rows, a
 * wide step's, writes the bytes the launches of its slabs of 8 rows write.
 */
#ifndef DS41F_HYPER_CONNECTION_H
#define DS41F_HYPER_CONNECTION_H

#include <stdint.h>

#include "engine/backend/backend_seam.h"
#include "ds41f_hyper_connection_records.h"
#include "engine/library/block_scaled_linear/linear.h"
#include "common/status.h"

struct ds41f_hyper_kernels_record {
	uint32_t mix_finish;
	uint32_t final_collapse;
	uint32_t post;
	uint32_t embed;
	uint32_t draft_inputs; /* the draft's two: ds41f_draft_kernels_prepare (ds41f_dspark.h) */
	uint32_t final_kept;
};
/* Prepares the first four kernels on one card, as linear_kernels_prepare does, and leaves the draft's two
 * BACKEND_NO_KERNEL: a card without the draft keeps its handles (ds41f_dspark.h). */
MUST_CHECK result_status ds41f_hyper_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct ds41f_hyper_kernels_record *kernels_out);

/* A site of a block: its mix finish, between its mix project and its sublayer, and its post after the sublayer,
 * with the combine where routed_sums is a range (the FFN site). */
enum ds41f_hyper_site_launch {
	DS41F_HYPER_SITE_MIX_FINISH = 0,
	DS41F_HYPER_SITE_POST = 1,
	DS41F_HYPER_SITE_LAUNCH_COUNT = 2
};
struct ds41f_hyper_site_description {
	struct linear_range streams; /* read and written: streams */
	struct linear_range
		carried_pre; /* read and written: pre, the previous site's, then this one's */
	struct linear_range mixes; /* read: mixes */
	struct linear_range
		scale; /* read: hc_attn_scale or hc_ffn_scale, exactly */
	struct linear_range base; /* read: hc_attn_base or hc_ffn_base, exactly */
	struct linear_range
		norm_weight; /* read: attn_norm.weight or ffn_norm.weight, exactly */
	struct linear_range post; /* written, then read: post */
	struct linear_range comb; /* written, then read: comb */
	struct linear_range hidden; /* written: hidden, the sublayer's input */
	struct linear_range
		output; /* read: hidden, the sublayer's output or the shared expert's */
	struct linear_range
		routed_sums; /* read: float32 [row][5,120], or {0, 0}: none */
	uint32_t row_count;
	uint32_t reserved_zero;
};
/* launches_out: DS41F_HYPER_SITE_LAUNCH_COUNT launches. */
MUST_CHECK result_status
ds41f_hyper_site_prepare(const struct ds41f_hyper_site_description *description,
			 uint64_t arena_byte_count,
			 const struct ds41f_hyper_kernels_record *kernels,
			 struct kernel_launch *launches_out);

/* The rows' tokens into the streams, before the first layer. */
struct ds41f_hyper_embed_description {
	struct linear_range
		step; /* read: the step area (step_records.h) of row_count rows */
	struct linear_range
		embedding; /* read: embed.weight [vocabulary_size][5,120], exactly */
	struct linear_range streams; /* written: streams */
	struct linear_range carried_pre; /* written: pre */
	uint32_t row_count;
	uint32_t vocabulary_size;
};
MUST_CHECK result_status ds41f_hyper_embed_prepare(
	const struct ds41f_hyper_embed_description *description,
	uint64_t arena_byte_count,
	const struct ds41f_hyper_kernels_record *kernels,
	struct kernel_launch *launch_out);

/* The final collapse with the last site's pre, and the final norm: the head's input. */
struct ds41f_hyper_final_description {
	struct linear_range streams; /* read: streams */
	struct linear_range carried_pre; /* read: pre */
	struct linear_range norm_weight; /* read: norm.weight, exactly */
	struct linear_range hidden; /* written: hidden */
	struct linear_range
		collapse; /* written: the collapse before the norm, hidden's layout, by
                                                       final_kept (not at byte 0); {0, 0} for none */
	uint32_t row_count;
	uint32_t reserved_zero;
};
MUST_CHECK result_status ds41f_hyper_final_prepare(
	const struct ds41f_hyper_final_description *description,
	uint64_t arena_byte_count,
	const struct ds41f_hyper_kernels_record *kernels,
	struct kernel_launch *launch_out);

/* The draft's inputs (ds41f_hyper_connection_records.h): STREAM_MEAN, tap t of the rows (the streams at layer 37 + t's
 * input) into the taps; ROW_NORM, main_norm of main_proj's output. Also refused: a mode outside the enumeration; a tap
 * of DS41F_DRAFT_TAPS or more, or a tap other than 0 for ROW_NORM; a weight given for STREAM_MEAN. */
struct ds41f_hyper_draft_inputs_description {
	struct linear_range
		input; /* read: STREAM_MEAN streams; ROW_NORM bfloat16 [row][5,120] */
	struct linear_range
		weight; /* ROW_NORM, read: mtp.0.main_norm.weight, exactly; else {0, 0} */
	struct linear_range
		output; /* written: STREAM_MEAN draft taps (ds41f_geometry.h), column 5,120
                                                       tap of each row; ROW_NORM bfloat16 [row][5,120] */
	uint32_t row_count; /* 1 to STEP_WIDE_MOST_ROWS: a chunk's rows */
	uint32_t mode; /* enum ds41f_hyper_draft_input_mode */
	uint32_t tap; /* STREAM_MEAN: 0 to 2, the layer less 37; ROW_NORM: 0 */
	uint32_t reserved_zero;
};
MUST_CHECK result_status ds41f_hyper_draft_inputs_prepare(
	const struct ds41f_hyper_draft_inputs_description *description,
	uint64_t arena_byte_count,
	const struct ds41f_hyper_kernels_record *kernels,
	struct kernel_launch *launch_out);

#endif
