/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_rows.h: the two launches of an n-gram layer's own (model.py:312-325, 350-365): the table rows of each row's
 * token to the values engram.wkv reads, and, once the linear has made their projection (linear_quantize_prepare
 * and linear_trunk_quantized_prepare: values to projection), the gate that adds it into the streams.
 *
 * Contract: linear.h's for the kernels and the prepare functions (once at load, arithmetic on a description, one
 * error line for a refusal, the launches untouched). Refused with STATUS_INVALID_ARGUMENT: a NULL argument; a
 * kernel not prepared; rows_per_token of 0 or above ENGRAM_MAXIMUM_ROWS_PER_TOKEN (engram.h); a row_count of 0
 * or above MODEL_WIDE_MOST_ROWS; a range that breaks a rule of prepare_common.h. Both launches take their rows
 * on grid[1] and a row's bits are its own (engram_rows_records.h): a launch of a chunk's rows, a wide step's, writes
 * the bytes the launches of its slabs of 8 rows write.
 */
#ifndef ENGRAM_ROWS_H
#define ENGRAM_ROWS_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/engram_rows_records.h"
#include "anemone/linear.h"
#include "anemone/status.h"

struct engram_rows_kernels {
	uint32_t values;
	uint32_t gate;
};
/* Prepares both kernels on one card, as linear_kernels_prepare does. */
MUST_CHECK result_status engram_rows_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct engram_rows_kernels *kernels_out);

enum engram_rows_launch {
	ENGRAM_ROWS_VALUES_LAUNCH =
		0, /* engram_values_entry: rows to values, before the linear */
	ENGRAM_ROWS_GATE_LAUNCH =
		1, /* engram_gate_entry: projection into streams, after it */
	ENGRAM_ROWS_LAUNCH_COUNT = 2
};
struct engram_rows_description {
	struct linear_range rows; /* read: rows [row_count][rows_per_token] */
	struct linear_range
		values; /* written: values [row_count][rows_per_token * 256] */
	struct linear_range projection; /* read: projection [row_count] */
	struct linear_range query_weight; /* read: engram.q_weight */
	struct linear_range key_weight; /* read: engram.k_weight */
	struct linear_range streams; /* read and written: streams [row_count] */
	uint32_t rows_per_token; /* of one table: engram_description's */
	uint32_t row_count;
};
/* launches_out: ENGRAM_ROWS_LAUNCH_COUNT launches. */
MUST_CHECK result_status engram_rows_prepare(
	const struct engram_rows_description *description,
	uint64_t arena_byte_count, const struct engram_rows_kernels *kernels,
	struct kernel_launch *launches_out);

#endif
