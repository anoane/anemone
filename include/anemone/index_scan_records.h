/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_records.h: the records of the index scan and its selection (source/kernel_index_scan.cu). C11 and CUDA,
 * of the engine linear_records.h alone: for the kernel file, the host code that prepares its launches, the tests.
 *
 * Contract:
 *   - linear_records.h's rules for a kernel and its record hold: offsets into the arena that whoever prepares a launch
 *     has checked, no atomic operation, nothing that depends on timing. The arrays are model_layout.h's (the step area
 *     of step_records.h, index queries and weights, key tiles, the selection); scores FP16 [row][entry_capacity].
 *   - Rows. Block row r serves row r of the step area at step_offset; at or beyond the step's row_count it reads
 *     nothing more and writes nothing. Row r at position p of session s sees e = (p + 1) / ratio entries (REF:561-567)
 *     of the key tiles at keys_offset + s keys_session_stride. The checked modules trap on a row step_records.h
 *     forbids: p below 0 or from context_token_limit on, s from session_capacity on (the scan).
 *   - The score of an entry below e (REF:554-557): per head the dot of query and key, the m16n8k16 instruction's
 *     float32 sum over the 8 steps of 16 dims in ascending order from +0; r is the dot where it is above 0, else +0;
 *     r times the head's weight (bfloat16, exact in float32); v_g = (r w of head g + of g + 8) + (r w of 16 + g + of
 *     24 + g) for g = 0 .. 7; then ((v_0 + v_1) + (v_2 + v_3)) + ((v_4 + v_5) + (v_6 + v_7)) rounded once to FP16, to
 *     nearest, a tie to even. Its bits depend on its row's query and weights and its key alone (11a's class R). In a
 *     tile that holds an entry below e, one at or beyond e scores -inf; no tile from ceil(e / 8) on is written. This
 *     is the one order: the reference's (an einsum, then a sum over heads) is the library's and cannot be repeated,
 *     so there is no reference-order knob; the selection is held to picks shared (first_token_plan.md 3.2).
 *   - The selection of a row (REF:577-580): of its e scores the top min(512, e) by score descending, then position
 *     ascending (ours: among equal scores the lowest position wins), written as positions ascending, then -1 up to
 *     512. Scores are ordered by their FP16 bits with the sign flipped, or every bit flipped where it is set: a
 *     not-a-number beyond the infinity of its sign, and -0 below +0, the order of the float32 sums (a -0 is a
 *     negative one rounded, or a row whose weights are all negative, which holds no +0).
 *   - The candidate stage (REF:570-576, 584-611): after the candidate source's selection, index_candidates
 *     keeps for each row the min(2,048, b) of its b = ceil(e / 8) blocks of 8 entries (a key tile each) whose
 *     largest score is highest, the block of its newest entry (b - 1) first whatever it scores, then blocks by
 *     score descending and number ascending (ours, as the selection's ties), written ascending, then
 *     MODEL_NO_CANDIDATE up to 2,048. A row of e at most 16,384 keeps every block, as the reference does.
 *     A layer that reads candidates (candidates_offset not INDEX_NO_CANDIDATES) scores its row's kept tiles
 *     alone, tile k at scores k * 8 .. k * 8 + 7 of a row of MODEL_CONTEXT_WITHOUT_CANDIDATES, and selects
 *     among the first min(e, 16,384) of them, written as the entries they stand for (block * 8 + k % 8): the
 *     reference's masking of every other entry to -inf, with positions above e still -inf.
 *   - Launches (grid[2], block[1] and block[2] are 1; rows: those a launch serves):
 *       kernel                    grid[0]  grid[1]                                     block[0]  dynamic shared
 *       index_scan_entry        rows     ceil(entry_capacity / 8 / SEGMENT_TILES)    128       8,192
 *       index_select_entry      rows     1                                           1,024     0
 *       index_candidates  rows     1                                           1,024     0
 *     where a layer that reads candidates has an entry_capacity of 16,384.
 */
#ifndef INDEX_SCAN_RECORDS_H
#define INDEX_SCAN_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"

#define KERNEL_FILE_INDEX_SCAN "kernel_index_scan"
#define KERNEL_FILE_INDEX_SCAN_NATIVE \
	"kernel_index_scan_native" /* class 120 native (backend_seam.h) */
#define KERNEL_INDEX_SCAN "index_scan_entry"
#define KERNEL_INDEX_SELECT "index_select_entry"
#define KERNEL_INDEX_CANDIDATES "index_candidates"
#define KERNEL_INDEX_SPLIT_HISTOGRAM_HIGH "index_split_histogram_high"
#define KERNEL_INDEX_SPLIT_CUT_HIGH "index_split_cut_high"
#define KERNEL_INDEX_SPLIT_HISTOGRAM_LOW "index_split_histogram_low"
#define KERNEL_INDEX_SPLIT_CUT_LOW "index_split_cut_low"
#define KERNEL_INDEX_SPLIT_OUTPUT "index_split_output"
#define INDEX_SCAN_ARGUMENT_BYTES 80u
#define INDEX_SELECT_ARGUMENT_BYTES 48u
#define INDEX_CANDIDATES_ARGUMENT_BYTES 40u
#define INDEX_SPLIT_ARGUMENT_BYTES 64u
#define INDEX_SPLIT_SEGMENTS 64u
#define INDEX_SPLIT_THREADS 256u
#define INDEX_SPLIT_BINS 256u
#define INDEX_SPLIT_ROW_BYTES 66560u
#define INDEX_SPLIT_PADDING_WORDS 60u
#define INDEX_SPLIT_LAUNCH_COUNT 5u
#define INDEX_NO_CANDIDATES \
	UINT64_MAX /* a candidates_offset: the layer reads none */
#define INDEX_SCAN_THREADS 128u /* 4 warps, a tile of 8 keys a warp at a time */
#define INDEX_SELECT_THREADS_VALUE \
	1024u /* 32 warps: a lane of warp 0 for each warp */
/* A segment's tiles at a whole pool, a shorter row spread over all: 512 keys, fastest of 512, 2,048, 8,192 (i11a) */
#define INDEX_SEGMENT_TILES \
	16u /* a row in 4 times the blocks of 64: for the PRO's emulation */
#define INDEX_SCORE_BYTES 2u
#define INDEX_NEGATIVE_INFINITY_VALUE 0xFC00u /* of FP16 */
#define INDEX_NO_POSITION (-1)

/* kernel_index_scan, index_scan_entry: the scores of a segment of the tiles of each row. */
struct index_scan_arguments_record {
	uint64_t step_offset; /* read: the step block and its rows */
	uint64_t keys_offset; /* read: key tiles of session 0 */
	uint64_t keys_session_stride; /* entry_capacity * MODEL_KEY_BYTES */
	uint64_t queries_offset; /* read: index queries [row][8,192 B] */
	uint64_t weights_offset; /* read: index weights bfloat16 [row][32] */
	uint64_t scores_offset; /* written: FP16 [row][entry_capacity], the tiles below e */
	uint64_t candidates_offset; /* read: uint32 [row][2,048], or INDEX_NO_CANDIDATES */
	uint32_t entry_capacity; /* a row of scores: context / ratio, or 16,384 with candidates */
	uint32_t ratio; /* positions an entry holds */
	uint32_t context_token_limit;
	uint32_t session_capacity;
	uint32_t row_first; /* the step row of the launch's row 0 (every record's alike) */
	uint32_t reserved_zero;
};

/* kernel_index_scan, index_select_entry: each row's selection from its scores. */
struct index_select_arguments_record {
	uint64_t step_offset; /* read: the step block and its rows */
	uint64_t scores_offset; /* read: FP16 [row][context / ratio, or 16,384 with candidates] */
	uint64_t selection_offset; /* written: int32 [row][512] */
	uint64_t candidates_offset; /* read: uint32 [row][2,048], or INDEX_NO_CANDIDATES */
	uint32_t context_token_limit;
	uint32_t ratio;
	uint32_t row_first;
	uint32_t reserved_zero;
};

/* kernel_index_scan, index_candidates: each row's kept blocks from the candidate source's scores. */
struct index_candidates_arguments {
	uint64_t step_offset; /* read: the step block and its rows */
	uint64_t scores_offset; /* read: FP16 [row][entry_capacity], the first e of a row */
	uint64_t candidates_offset; /* written: uint32 [row][2,048] */
	uint32_t entry_capacity; /* context_token_limit: the source's ratio is 1 */
	uint32_t context_token_limit;
	uint32_t row_first;
	uint32_t reserved_zero;
};

/* Five ordered launches: histogram high byte, reduce high cut, histogram low byte, reduce exact cut and segment
 * prefixes, output. Histogram/output grids are (rows, segments); cuts are (rows, 1), all with SPLIT_THREADS.
 * is_block is 0 for ordinary selection (candidates_offset may map positions), 1 for candidate-source blocks
 * (ratio is 1 and candidates_offset is NO_CANDIDATES). output_offset names the corresponding output array.
 * segments is 1..SPLIT_SEGMENTS. Scratch is disjoint, linear-record aligned, ROW_BYTES per prepared row; its lifetime
 * spans all five launches. Only live rows are accessed. Every pass overwrites the scratch it next consumes.
 * Integer, fixed-order reductions preserve the legacy cut, first tied positions, ascending output and fill.
 * No atomic operation or inter-block wait is used. A segment owns a contiguous run of whole warp turns;
 * the final pass uses its prefix before applying the same position-ordered local tie/output prefixes.
 */
struct index_split_arguments {
	uint64_t step_offset;
	uint64_t scores_offset;
	uint64_t output_offset;
	uint64_t candidates_offset;
	uint64_t scratch_offset;
	uint32_t context_token_limit;
	uint32_t ratio;
	uint32_t segments;
	uint32_t is_block;
	uint32_t row_first;
	uint32_t reserved_zero;
};

struct index_split_segment {
	uint32_t bins[INDEX_SPLIT_BINS];
	uint32_t greater_high;
	uint32_t ties_before;
	uint32_t outputs_before;
};

struct index_split_scratch {
	uint32_t threshold;
	uint32_t ties_needed;
	uint32_t above_high;
	uint32_t reserved;
	struct index_split_segment segment[INDEX_SPLIT_SEGMENTS];
	uint32_t padding[INDEX_SPLIT_PADDING_WORDS];
};

LINEAR_RECORD_HAS(index_scan_arguments_record, INDEX_SCAN_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(index_select_arguments_record, INDEX_SELECT_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(index_candidates_arguments, INDEX_CANDIDATES_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(index_split_arguments, INDEX_SPLIT_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(index_split_scratch, INDEX_SPLIT_ROW_BYTES);

#endif
