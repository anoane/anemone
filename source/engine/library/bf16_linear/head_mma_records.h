/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * head_mma_records.h: the BF16 head and router MMA kernels' geometry, forms and launches as the program
 * reads them
 * (split from model_v41_program.c by whole items).
 */
#ifndef HEAD_MMA_RECORDS_H
#define HEAD_MMA_RECORDS_H

#define HEAD_MMA_KERNEL_FILE \
	"kernel_head_mma" /* head_mma (kernel_head_mma.cu): its forms and geometry */
#define HEAD_MMA_KERNEL_8 \
	"head_mma_8_s4" /* head02: 2-8 rows (1 row: the stream is faster) */
#define HEAD_MMA_SPLIT_8 4u
#define HEAD_MMA_LEAST_ROWS 2u
#define HEAD_MMA_WIDE_FORMS \
	3u /* head02: _64_s1 to 24 rows, _s2 to 48, _s4 above */
#define HEAD_MMA_WIDE_THIN_FORM \
	2u /* _64_s4: HEAD_MMA_SPLIT_8, the 8-row form's split */
/* head_wide_product: _64_s1 to so many rows, _64_s2 to HEAD_MMA_WIDE_S2_MOST_ROWS, _64_s4 above */
#define HEAD_MMA_WIDE_S1_MOST_ROWS 24u
#define HEAD_MMA_WIDE_S2_MOST_ROWS 48u
#define HEAD_MMA_WIDE_S4_FORM 2u
_Static_assert((1u << HEAD_MMA_WIDE_THIN_FORM) == HEAD_MMA_SPLIT_8,
	       "the thin wide head splits as the 8-row form");
#define HEAD_MMA_THREADS 256u
#define HEAD_MMA_BLOCK_OUTPUTS \
	128u /* of a split of 1: the _sN forms take 128 / N */
#define HEAD_MMA_STEP_INPUTS 32u
#define HEAD_MMA_MOST_ROWS 64u
#define HEAD_MMA_TILE_OUTPUTS 16u /* a warp's outputs */
#define ROUTER_MMA_WARPS 4u /* router_mma: kernel_head_mma.cu's geometry */
#define ROUTER_MMA_REDUCE_THREADS 256u
/* router_scores' product and its reduce */
#define ROUTER_MMA_LAUNCHES 2u

#endif
