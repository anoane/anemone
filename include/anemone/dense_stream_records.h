/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_stream_records.h (exl3_v3_claude's kernel, tp4_v3_claude's use): kernel_dense_stream's
 * dense_stream_bfloat16_8 (epoch02 17f7430d), the dense product of dense_linear_records.h for bfloat16 weights
 * and 1 to 8 rows with the same record and the same order of additions, in persistent blocks of
 * DENSE_THREADS_VALUE over groups of DENSE_OUTPUTS_PER_BLOCK outputs, the next group's weights in flight;
 * any grid gives the same bits. Taken for head products on class 80 cards where a program's knob selects it (2.5x the
 * dense product's own kernel there: 362 against 909 us for 25,856 x 5,120, 6 rows). DENSE_STREAM_BLOCKS: two a
 * multiprocessor of these CMP 170HX (74, the measured schedule); a launch takes at most one block a group.
 */
#ifndef DENSE_STREAM_RECORDS_H
#define DENSE_STREAM_RECORDS_H

#include "anemone/dense_linear_records.h"

#define KERNEL_FILE_DENSE_STREAM "kernel_dense_stream"
#define KERNEL_DENSE_STREAM "dense_stream_bfloat16_8"
#define DENSE_STREAM_BLOCKS 148u
/* The wide form (exl3_v3_claude src37 2673a3be, its two-group entry; one block a multiprocessor): the same record with
 * row_count 1 .. DENSE_STREAM_WIDE_ROWS and, with DENSE_STEP_ROWS, a row stored where it is below the
 * step block's row_count with LOGITS; every row's bits those of its 8-row group's launch: a BATCH step's head in one
 * weight pass. */
#define KERNEL_DENSE_STREAM_WIDE "dense_stream_bfloat16_wide16"
#define DENSE_STREAM_WIDE_ROWS 64u
#define DENSE_STREAM_WIDE_BLOCKS 74u /* one a multiprocessor */

#endif
