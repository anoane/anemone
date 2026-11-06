/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_rows_records.h: the records of the two kernels of an n-gram layer (source/kernel_engram_rows.cu) and the
 * arrays they name. C11 and CUDA, including of the engine model_layout.h alone.
 *
 * Contract: linear_records.h's rules for a kernel, its record and its ranges, and, for the rows of grid[1]:
 *   - rows        uint8 [row][grid[0]][ENGRAM_ROWS_ROW_BYTES]: the table rows of the row's token as
 *                 engram_gather_entry gives them (engram.h), 256 E4M3 bytes and an E8M0 code for each 32 of them;
 *     values      bfloat16 [row][grid[0] 256]: value v of table row j at 256 j + v;
 *     projection  bfloat16 [row][ENGRAM_ROWS_PROJECTION]: engram.wkv's output, the key of stream k at k
 *                 hidden, then the value (model.py:353-354); q_weight, k_weight bfloat16 [stream][hidden] each;
 *     streams     model_layout.h's, read and written in place.
 *   - engram_values_entry, a block of 256 threads for a table row (model.py:317-320): an E4M3 byte's number times
 *     2^(c - 127) in float32 (2^-127 for c = 0; a byte or a code that is no number gives a not-a-number), rounded
 *     once to bfloat16 as linear_records.h rounds.
 *   - engram_gate_entry, a block of ENGRAM_ROWS_GATE_THREADS threads for stream k of a row (model.py:
 *     355-365), in float32: of each value h of the stream, its key and w = q_weight k_weight, the sums A of h h, B
 *     of key key and D of (h w) key, each from +0: thread t adds values t + i ENGRAM_ROWS_GATE_THREADS in
 *     ascending i; each lane of a warp adds to its sum that of the lane at xor distance 16, then 8, 4, 2 and 1 (every
 *     lane ends with the same sum); then the warps' sums in ascending order. f = float32(1 / hidden) and eps
 *     (model_layout.h) are the reference's mean and epsilon: rstd = rsqrtf(A f + eps) rsqrtf(B f + eps),
 *     dot = (D rstd) float32(hidden^-0.5), s = copysign(sqrt(max(|dot|, 1e-6)), dot), a not-a-number kept, gate =
 *     1 / (1 + expf(-s)) (rsqrtf and expf: the card's, as the reference's library); h becomes bfloat16(h + gate
 *     value). No row's bits depend on another row.
 */
#ifndef ENGRAM_ROWS_RECORDS_H
#define ENGRAM_ROWS_RECORDS_H

#include <stdint.h>

#include "anemone/model_layout.h"

#define KERNEL_FILE_ENGRAM_ROWS "kernel_engram_rows"
#define KERNEL_ENGRAM_VALUES "engram_values_entry"
#define KERNEL_ENGRAM_GATE "engram_gate_entry"
#define ENGRAM_ROWS_VALUES \
	256u /* of a table row: engram.h's ENGRAM_VALUE_BYTES_VALUE */
#define ENGRAM_ROWS_VALUES_PER_SCALE 32u /* model.py:307 */
#define ENGRAM_ROWS_ROW_BYTES 264u /* engram.h's ENGRAM_ROW_BYTES */
#define ENGRAM_ROWS_PROJECTION \
	25600u /* a key for each stream and the value: (4 + 1) hidden */
#define ENGRAM_ROWS_GATE_THREADS 512u
#define ENGRAM_VALUES_ARGUMENT_BYTES 16u
#define ENGRAM_GATE_ARGUMENT_BYTES 32u

struct engram_values_arguments {
	uint64_t rows_offset;
	uint64_t values_offset;
};

/* grid[0] is MODEL_STREAMS. */
struct engram_gate_arguments {
	uint64_t projection_offset;
	uint64_t query_weight_offset;
	uint64_t key_weight_offset;
	uint64_t streams_offset;
};

MODEL_LAYOUT_HOLDS(sizeof(struct engram_values_arguments) ==
			   ENGRAM_VALUES_ARGUMENT_BYTES &&
		   sizeof(struct engram_gate_arguments) ==
			   ENGRAM_GATE_ARGUMENT_BYTES);
MODEL_LAYOUT_HOLDS(ENGRAM_ROWS_ROW_BYTES ==
			   ENGRAM_ROWS_VALUES +
				   ENGRAM_ROWS_VALUES /
					   ENGRAM_ROWS_VALUES_PER_SCALE &&
		   MODEL_HIDDEN % ENGRAM_ROWS_GATE_THREADS == 0 &&
		   ENGRAM_ROWS_PROJECTION ==
			   (MODEL_STREAMS + 1u) * MODEL_HIDDEN);

#endif
