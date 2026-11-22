/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trunk_mma_records.h: the BF16 trunk MMA kernel's geometry, forms and launches as the program
 * reads them (split from
 * model_v41_program.c by whole items).
 */
#ifndef TRUNK_MMA_RECORDS_H
#define TRUNK_MMA_RECORDS_H

#include "engine/library/block_scaled_linear/linear_records.h" /* LINEAR_MAXIMUM_ROWS */

#define TRUNK_MMA_KERNEL_FILE \
	"kernel_trunk_mma" /* trunk_mma (kernel_trunk_mma.cu): forms t<T>_s<S>, geometry */
#define TRUNK_MMA_FORMS 11u /* trunk_mma_forms below */
#define TRUNK_MMA_THREADS 256u
#define TRUNK_MMA_TILE_OUTPUTS 16u
#define TRUNK_MMA_GROUP_INPUTS 32u
#define TRUNK_MMA_WIDE_KERNEL "trunk_mma_64"
#define TRUNK_MMA_THIN_STEM \
	"trunk_mma_8_" /* the 1-8-row forms' names: this, then t<T>_s<S>[_w16] */
#define TRUNK_MMA_GROUPED_FORMS \
	5u /* batch_thin_bits: the forms of 2 to 6 row groups of 8 (16 to 48 rows) */
#define TRUNK_MMA_GROUPED_MOST_ROWS \
	(LINEAR_MAXIMUM_ROWS * (TRUNK_MMA_GROUPED_FORMS + 1u))
#define TRUNK_MMA_NAME_BYTES 64u
#define TRUNK_MMA_PAIR_FORM \
	8u /* trunk_mma_forms' t1_s16: the form of wq_a, wkv and the shared gate / up shards */
#define TRUNK_MMA_SET_MOST_GROUPS \
	3u /* batch_row_sets: row sets of 1 to 3 row groups (8, 16 or 24 rows) */
#define TRUNK_MMA_PAIR_SUFFIX "_pair"
#define TRUNK_MMA_FUSED_KERNEL \
	"trunk_mma_64_fused" /* its splits added by the last of them */
#define TRUNK_MMA_WIDE_TILES \
	8u /* tiles of 16 outputs a block of the _64 form */
#define TRUNK_MMA_WIDE_MOST_ROWS 64u
#define TRUNK_MMA_WIDE_BLOCKS \
	140u /* the _64 form: splits until the launch has about this many blocks */
#define TRUNK_MMA_MOST_SPLITS 16u
#define TRUNK_MMA_REDUCE_THREADS 256u
#define TRUNK_MMA_VALUES_ALIGNMENT \
	16u /* the _64 form stages the rows by 16-byte copies */
/* trunk_mma_form_of: t1_s16 (trunk_mma_forms' 8) to so many tiles, t1_s4 (2) to TRUNK_MMA_S4_MOST_TILES, t1_s2
 * (1) above */
#define TRUNK_MMA_S16_MOST_TILES 128u
#define TRUNK_MMA_S4_MOST_TILES 512u
#define TRUNK_MMA_FORM_T1_S16 8u
#define TRUNK_MMA_FORM_T1_S4 2u
#define TRUNK_MMA_FORM_T1_S2 1u
/* trunk_mma_splits_of: the splits double while the doubled fit */
#define TRUNK_MMA_SPLIT_GROWTH 2u
/* trunk_mma_grouped_kernels[0]: the form of so many row groups */
#define TRUNK_MMA_GROUPED_LEAST_GROUPS 2u
/* trunk_mma_pair_merge: the two launches it merges */
#define TRUNK_MMA_PAIR_LAUNCHES 2u

#endif
