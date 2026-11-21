/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_site_front_records.h: the records of the fused fronts of a block's sites (ds41f_site_front.cu, the phases C,
 * D and E of the 25 Oct 2026 report on the oproj and mHC fronts). The site front is a site's mix project, its mix
 * finish and the quantization of the sublayer's input in one launch (the attention site's, ahead of attn.wq_a with
 * attn.wkv); the FFN front is the same for the FFN site with the router's scores and its select in the same launch.
 * C11 and CUDA, including of the engine the record layouts of the linear, the dense products, the hyper connection
 * and the router alone.
 *
 * Contract: the records of the launches each fused launch stands for (dense_linear_records.h,
 * ds41f_hyper_connection_records.h, linear_records.h, ds41f_router_records.h), whose outputs it writes with the same
 * bits: the same chains, trees, roundings and quantization; only launches merge. And:
 *   - site_front_<R> and ffn_front_<R> (R 1 for one row, 8 for 2 to 8 rows; blocks of
 *     DS41F_SITE_FRONT_THREADS, no dynamic shared memory; grid DS41F_SITE_FRONT_BLOCKS(kind, rows)) stand for, in
 *     this order: the mix project (the dense product of mix weights,
 *     float32 [24][20,480], and the streams as its rows, with DENSE_ROW_SCALE, epsilon
 *     DS41F_NORM_EPSILON and output scale 1), the mix finish, and the quantization of hidden (width
 *     DS41F_HIDDEN) into codes, exponents and operands; the FFN front also, between the finish and the
 *     quantization, the router's dense product (bfloat16 router weights [384][5,120] and hidden as its rows, no flag,
 *     output scale 1, into select.scores) and the select of select (its affinities and job table). The mixes, which
 *     no later launch reads, are not stored in a mixes array: the scratch holds them (its mixes, of the launch's
 *     rows), the same bits, until a front writes that scratch again (a capture of the layer reads them there:
 *     ds41f_program_view.h). input_front_<R> and then coefficient_front_<R> stand for the site front
 *     in two launches (the second may run beside later launches up to the site's post: it reads the streams, which
 *     nothing writes before that post, and it writes the pre that the first read): the first stores hidden, codes,
 *     exponents and operands, the second the mixes (scratch), post, comb and the site's pre.
 *   - The row count is select.row_count; the site front reads nothing else of select, which is zero but for it.
 *   - Scratch (struct ds41f_site_front_scratch). Integer atomic additions on its counters order the blocks: the FFN
 *     front's blocks take their parts in the order of their tickets (rows first, the scores' blocks last, which wait
 *     for every row's input: blocks of lower tickets, which wait for nothing); the last block to publish its part
 *     finishes it (the mixes and Sinkhorn; the select). No floating-point result depends on which block that is, nor
 *     on timing. Every counter is zero before the launch and the launch leaves it zero; launches that share a scratch
 *     run one after the other (one stream). The rest carries the mix project's warp sums and square sums from the
 *     blocks that form them to the last, and the mixes.
 *   - The rest of linear_records.h's contract for a kernel holds: whoever prepares the launch checked every range.
 */
#ifndef DS41F_SITE_FRONT_RECORDS_H
#define DS41F_SITE_FRONT_RECORDS_H

#include <stdint.h>

#include "engine/library/bf16_linear/dense_linear_records.h"
#include "ds41f_hyper_connection_records.h"
#include "engine/library/block_scaled_linear/linear_records.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "models/deepseek_v41_flash/moe/ds41f_router_records.h"

#define DS41F_MODULE_SITE_FRONT "ds41f_site_front"
#define DS41F_KERNEL_SITE_FRONT_STEM "site_front_"
#define DS41F_KERNEL_FFN_FRONT_STEM "ffn_front_"
#define DS41F_SITE_FRONT_THREADS \
	256u /* the dense product's block; the select's */
#define DS41F_SITE_FRONT_MIX_INPUTS \
	(DS41F_STREAMS * DS41F_HIDDEN) /* the mix project's */
#define DS41F_SITE_FRONT_LOGICAL_WARPS \
	(DENSE_THREADS_VALUE / LINEAR_THREADS_PER_WARP) /* of an output */
#define DS41F_SITE_FRONT_MIX_PARTS \
	2u /* ours: the site front's blocks of one mix */
#define DS41F_FFN_FRONT_MIX_PARTS \
	1u /* the FFN front's: its blocks fit on a CMP at once */
#define DS41F_FFN_FRONT_EXPERTS \
	4u /* ours: experts of a block of scores (the dense product's wide form) */
#define DS41F_FFN_FRONT_SCORE_BLOCKS (DS41F_EXPERTS / DS41F_FFN_FRONT_EXPERTS)
/* The four kinds of front: the site front, the FFN front, and the site front cut in two launches that run in this
 * order, the input front (its row blocks) and the coefficient front (its mix blocks and the finish). */
enum ds41f_site_front_kind {
	DS41F_SITE_FRONT_KIND_SITE = 0,
	DS41F_SITE_FRONT_KIND_FFN = 1,
	DS41F_SITE_FRONT_KIND_INPUT = 2,
	DS41F_SITE_FRONT_KIND_COEFFICIENTS = 3,
	DS41F_SITE_FRONT_KINDS = 4
};
#define DS41F_SITE_FRONT_MIX_BLOCKS (DS41F_MIXES * DS41F_SITE_FRONT_MIX_PARTS)
#define DS41F_SITE_FRONT_INPUT_PARTS \
	4u /* ours: the input front's blocks of a row, each quantizing a quarter */
/* The blocks of a launch of a kind for rows rows: the FFN front of one row has no row block (its score blocks form
 * and quantize its input). */
#define DS41F_SITE_FRONT_BLOCKS(kind, rows)                        \
	((kind) == DS41F_SITE_FRONT_KIND_FFN ?                     \
		 ((rows) > 1u ? (rows) : 0u) +                     \
			 DS41F_MIXES * DS41F_FFN_FRONT_MIX_PARTS + \
			 DS41F_FFN_FRONT_SCORE_BLOCKS :            \
	 (kind) == DS41F_SITE_FRONT_KIND_INPUT ?                   \
		 (rows) * DS41F_SITE_FRONT_INPUT_PARTS :           \
	 (kind) == DS41F_SITE_FRONT_KIND_COEFFICIENTS ?            \
		 DS41F_SITE_FRONT_MIX_BLOCKS :                     \
		 (rows) + DS41F_SITE_FRONT_MIX_BLOCKS)
#define DS41F_SITE_FRONT_ARGUMENT_BYTES 192u
#define DS41F_SITE_FRONT_SCRATCH_BYTES 7184u
/* SITE_FRONT_STAMP(name, rows, kind) for every function, in the order of struct ds41f_site_front_kernels
 * (a kind's two row forms, the kinds in their order): whoever expands it defines it first and undefines it after. */
#define DS41F_SITE_FRONT_EVERY_INSTANCE                                      \
	SITE_FRONT_STAMP(site, 1, DS41F_SITE_FRONT_KIND_SITE)                \
	SITE_FRONT_STAMP(site, 8, DS41F_SITE_FRONT_KIND_SITE)                \
	SITE_FRONT_STAMP(ffn, 1, DS41F_SITE_FRONT_KIND_FFN)                  \
	SITE_FRONT_STAMP(ffn, 8, DS41F_SITE_FRONT_KIND_FFN)                  \
	SITE_FRONT_STAMP(input, 1, DS41F_SITE_FRONT_KIND_INPUT)              \
	SITE_FRONT_STAMP(input, 8, DS41F_SITE_FRONT_KIND_INPUT)              \
	SITE_FRONT_STAMP(coefficient, 1, DS41F_SITE_FRONT_KIND_COEFFICIENTS) \
	SITE_FRONT_STAMP(coefficient, 8, DS41F_SITE_FRONT_KIND_COEFFICIENTS)
#define DS41F_SITE_FRONT_ROW_FORMS 2u /* one row, 8 */
#define DS41F_SITE_FRONT_INSTANCES \
	(DS41F_SITE_FRONT_KINDS * DS41F_SITE_FRONT_ROW_FORMS)

struct ds41f_site_front_arguments {
	struct ds41f_router_select_arguments
		select; /* the FFN front's select, its scores written here too; the site
                                                       front's: row_count alone */
	uint64_t streams_offset; /* read: streams */
	uint64_t pre_offset; /* read, then written: the carried pre, then the site's */
	uint64_t scale_offset; /* read: float32 [3] */
	uint64_t base_offset; /* read: float32 [24] */
	uint64_t post_offset; /* written */
	uint64_t comb_offset; /* written */
	uint64_t norm_weight_offset; /* read: bfloat16 [5,120] */
	uint64_t hidden_offset; /* written (and read by the FFN front's scores) */
	uint64_t mix_weights_offset; /* read: hc_attn_fn or hc_ffn_fn, float32 [24][20,480] */
	uint64_t codes_offset; /* written: codes [row_count][5,120] (linear_records.h) */
	uint64_t exponents_offset; /* written: exponents [row_count][160] */
	uint64_t operands_offset; /* written: operands [row_count][5,120] */
	uint64_t scratch_offset; /* read and written: struct ds41f_site_front_scratch */
	uint64_t router_weights_offset; /* read by the FFN front: ffn.gate.weight, bfloat16 [384][5,120] */
};

struct ds41f_site_front_scratch {
	uint32_t tickets; /* the FFN front's blocks that took a part */
	uint32_t arrived; /* row blocks that read the carried pre; mix blocks that published */
	uint32_t rows_ready; /* the FFN front's row blocks whose input is stored */
	uint32_t scored; /* the FFN front's score blocks that published */
	float mixes[LINEAR_MAXIMUM_ROWS][DS41F_MIXES];
	float square_sums[DS41F_SITE_FRONT_LOGICAL_WARPS]
			 [LINEAR_MAXIMUM_ROWS]; /* the mix project's warps' */
	float warp_sums[DS41F_MIXES][DS41F_SITE_FRONT_LOGICAL_WARPS]
		       [LINEAR_MAXIMUM_ROWS];
};

RECORD_HAS(ds41f_site_front_arguments, DS41F_SITE_FRONT_ARGUMENT_BYTES);
RECORD_HAS(ds41f_site_front_scratch, DS41F_SITE_FRONT_SCRATCH_BYTES);
RECORD_HOLDS(DS41F_SITE_FRONT_THREADS == DENSE_THREADS_VALUE &&
	     DS41F_SITE_FRONT_THREADS == ROUTER_THREADS &&
	     DS41F_SITE_FRONT_LOGICAL_WARPS % DS41F_SITE_FRONT_MIX_PARTS == 0 &&
	     DS41F_EXPERTS % DS41F_FFN_FRONT_EXPERTS == 0 &&
	     DS41F_SITE_FRONT_MIX_INPUTS %
			     (DENSE_THREADS_VALUE * DENSE_CHUNK_INPUTS) ==
		     0);

#endif
