/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_card_test: the index scan and its selection (source/kernel_index_scan.cu) on every card, held to the host
 * reference (index_scan_reference.h) and, given a capture folder, replayed on the reference model's own inputs.
 *
 *     index_scan_card_test [--break ties|extent|word|tree] [<stepwise folder of the 2,048-token text>]
 *
 * Synthetic, on every card: a step of 7 rows in a launch of 8 (the eighth lies beyond the step and must stay as it
 * was) over two sessions, context 16,384. EXACT values at ratios 2 and 1: keys k / 16, queries q / 8, weights w / 64
 * with |k| <= 3, |q| <= 2, |w| <= 8, so every product and sum is exact in float32 and many scores tie across the cut
 * (one row's query is zero: all its scores tie). WIDE values at ratio 1: FP16 of 2^-12 to 2 and weights of 2^-10 to
 * 2^-3, both signs. TREE values at ratio 1: EXACT keys and queries, so every dot and product is exact in float32,
 * and weights that make the head tree's float32 roundings show in the FP16 bits (tree_weights_of). Extents 0, 1, 7,
 * 512, 513 and the whole pool among them. Held:
 *   - EXACT and TREE: every score, the sign of a zero included, is the host's FP16 rounding of its float32 sum in the
 *     records' tree; WIDE: within 2e-3 of the row's rms plus half an FP16 place (first_token_plan.md 3.2); all: -inf
 *     from e to the end of its tile, nothing written from tile ceil(e / 8) on nor in the row beyond the step, no range
 *     read is changed (the fixture's guards lie behind every range);
 *   - every selection is the host's sort of the card's own scores;
 *   - R: each row alone, in a launch of one row, has its bits of the launch of eight.
 * Replay (a folder of tools/oracle/stepwise_capture.py): at every index layer and capture position the folder holds
 * up to its last snapshot, one row at that position with the captured index queries and weights, and the keys of the
 * layer's owner from the first snapshot at or after it (the owner writes each entry once: model.py:546-547): the
 * scores within 2e-3 of the row's rms of the captured scores plus half an FP16 place, the selection the host's sort
 * of the card's scores. Of the rows above 512 entries (below, every entry is picked), at least 99% of the
 * reference's picks shared on average (3.2: the u16 reference keeps its scores in float32, so near ties may part).
 * Each row prints its share and its largest captured |score| (FP16 ends at 65,504). A folder without such a row
 * fails: of U13a's, only the 2,048-token text's stepwise folder has one (snapshots at 1,026 and 2,047; the 435-token
 * prompt's end at 129). The share against the published reference (3.2) is M2d's: the stepwise capture holds the u16
 * arm alone.
 * Breaks, each of which must make the test fail: ties (the host takes the highest position among equal scores),
 * extent (the host's e of one row one more), word (one key value changed on the host only), tree (the host sums the
 * heads one after another).
 * Prints a digest per card of every byte it fetched: every CMP 170HX prints the same one.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/index_scan.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "capture_reader.h"
#include "index_scan_reference.h"
#include "linear_card_fixture.h"

#define TEST_PASSED 0
#define TEST_FAILED 1
#define TEST_WRONG_USAGE 2
#define TIME_LIMIT_SECONDS 3600
#define WORK_CAPACITY (16ull << 20u)            /* the widest case: two sessions of 16,384 keys and its rows */
#define ROWS LINEAR_MAXIMUM_ROWS
#define STEP_ROWS (ROWS - 1u)                   /* the rows of the step; the last row of the launch lies beyond */
#define SESSIONS 2u
#define CONTEXT MODEL_CONTEXT_WITHOUT_CANDIDATES
#define HEADS MODEL_INDEX_HEADS
#define DIMS MODEL_INDEX_HEAD_DIMS
#define QUERY_VALUES ((uint32_t)(HEADS * DIMS))
#define SELECTED MODEL_SELECTED
#define KEYS_PER_TILE MODEL_KEYS_PER_TILE
#define SCORE_BYTES INDEX_SCORE_BYTES
#define SENTINEL 0x5A                           /* every byte of the scores and the selection before a launch */
#define NO_ROW UINT32_MAX
#define TOLERANCE_OF_RMS 2e-3
#define HALF_PLACE_OF_VALUE 0x1p-11             /* half an FP16 place, against a value, and below the normal ones */
#define HALF_PLACE_SUBNORMAL 0x1p-25
#define SHARE_REQUIRED 0.99
#define HALF_FIELD_SHIFT 10u
#define HALF_FRACTION_MASK 0x3FFu
#define BFLOAT16_FIELD_SHIFT 7u
#define BFLOAT16_FRACTION_MASK 0x7Fu
#define SIGN_SHIFT 15u
#define WIDE_KEY_FIELD 3u                       /* FP16 exponent fields 3 .. 15: 2^-12 to below 2 */
#define WIDE_KEY_FIELDS 13u
#define WIDE_WEIGHT_FIELD 117u                  /* bfloat16 fields 117 .. 124: 2^-10 to below 2^-2 */
#define WIDE_WEIGHT_FIELDS 8u
#define LARGE_WEIGHT_FIELD 135u                 /* bfloat16 fields 135 .. 139: 2^8 to below 2^13 */
#define LARGE_WEIGHT_FIELDS 5u
#define HEAD_TILE_HEADS 16u                     /* heads g and g + 8 of a tile: v_g's terms (index_scan_records.h) */
#define HALF_TILE_HEADS 8u
#define MIRROR_DISTANCE 4u                      /* v_g and v_(g + 4) meet only in the last addition */
#define EXACT_KEY_SPREAD 3u                     /* k from -3 to 3, times 2^-4 */
#define EXACT_KEY_EXPONENT (-4)
#define EXACT_QUERY_SPREAD 2u
#define EXACT_QUERY_EXPONENT (-3)
#define EXACT_WEIGHT_SPREAD 8u
#define EXACT_WEIGHT_EXPONENT (-6)
#define FRACTION_SHIFT 8u                       /* hash bits a value's fraction is taken from */
#define KEY_STREAM 0x5100u
#define QUERY_STREAM 0x5200u
#define WEIGHT_STREAM 0x5300u
#define NAME_CAPACITY 96u
#define DIMENSIONS_OF_A_MATRIX 2u
#define BREAK_ARGUMENTS 2                       /* --break and its name */

enum case_kind { CASE_EXACT_POOLED, CASE_EXACT_DIRECT, CASE_WIDE_DIRECT, CASE_TREE_DIRECT, CASE_COUNT };
enum break_kind { BREAK_NONE, BREAK_TIES, BREAK_EXTENT, BREAK_WORD, BREAK_TREE };

/* Each case's positions, row by row; the last row lies beyond the step. */
static const uint32_t case_positions[CASE_COUNT][ROWS] = {
    {0, 1, 13, 1023, 1025, 16383, 9000, 9001},
    {0, 6, 511, 512, 4095, 16383, 1000, 3000},
    {2047, 6, 600, 512, 4095, 16383, 1000, 3000},
    {2, 7, 8, 1000, 4100, 16383, 600, 3000},
};
static const uint32_t case_ratios[CASE_COUNT] = {2, 1, 1, 1};
static const uint32_t case_zero_query_rows[CASE_COUNT] = {NO_ROW, 4, NO_ROW, NO_ROW};
static const uint32_t break_extent_row = 1;     /* the row whose e the host takes one more with --break extent */

/* The index layers: each with the owner of the keys it scores and its ratio (config.json kv_source_layers,
 * index_source_layers, compress_ratios); the snapshot positions of stepwise_recorder.py. */
enum index_layer_field { LAYER, OWNER, RATIO, INDEX_LAYER_FIELDS };
static const uint32_t index_layers[][INDEX_LAYER_FIELDS] = {{2, 2, 2},   {8, 8, 2},   {14, 14, 2}, {20, 20, 1},
                                                            {24, 20, 1}, {28, 20, 1}, {32, 20, 1}, {36, 20, 1}};
static const uint32_t snapshot_positions[] = {129, 1026, 2047};    /* ascending */
#define SNAPSHOTS (sizeof(snapshot_positions) / sizeof(snapshot_positions[0]))

struct scan_case {
    uint32_t ratio;
    uint32_t entries;                           /* of a session's pool */
    uint16_t *keys;                             /* [SESSIONS][entries][128] */
    uint16_t queries[ROWS][QUERY_VALUES];
    uint16_t weights[ROWS][HEADS];
    double *reference;                          /* [ROWS][entries]: the host's scores, before their rounding */
    bool is_exact;                              /* the reference is the float32 tree's, held bit for bit */
};

static enum break_kind broken = BREAK_NONE;

static uint32_t extent_of(uint32_t position, uint32_t ratio)
{
    return (position + 1u) / ratio;
}

/* A whole number from -spread to spread times 2^exponent, as FP16; or (spread 0) FP16 bits of a wide range. */
static uint16_t value_of(uint64_t stream, uint64_t index, uint32_t spread, int exponent)
{
    const uint64_t hash = test_linear_hash(stream, index);
    if (spread != 0) {
        return test_index_half_nearest(ldexp((double)(int64_t)(hash % (spread + spread + 1u)) - (double)spread,
                                             exponent));
    }
    return (uint16_t)((((uint32_t)hash & 1u) << SIGN_SHIFT) |
                      ((WIDE_KEY_FIELD + (uint32_t)(hash >> 1) % WIDE_KEY_FIELDS) << HALF_FIELD_SHIFT) |
                      ((uint32_t)(hash >> FRACTION_SHIFT) & HALF_FRACTION_MASK));
}

/* An EXACT weight, or bfloat16 bits of the WIDE range (both signs) or of the LARGE one (above 0). */
static uint16_t weight_of(uint64_t index, bool is_exact, bool is_large)
{
    const uint64_t hash = test_linear_hash(WEIGHT_STREAM, index);
    if (is_exact) {
        const float value = (float)ldexp((double)(int64_t)(hash % (EXACT_WEIGHT_SPREAD + EXACT_WEIGHT_SPREAD + 1u)) -
                                             (double)EXACT_WEIGHT_SPREAD,
                                         EXACT_WEIGHT_EXPONENT);
        uint32_t bits = 0;
        memcpy(&bits, &value, sizeof(bits));
        return (uint16_t)(bits >> (SIGN_SHIFT + 1u));       /* exact: a few bits of fraction */
    }
    const uint32_t field = is_large ? LARGE_WEIGHT_FIELD + (uint32_t)(hash >> 1) % LARGE_WEIGHT_FIELDS
                                    : WIDE_WEIGHT_FIELD + (uint32_t)(hash >> 1) % WIDE_WEIGHT_FIELDS;
    return (uint16_t)(((is_large ? 0u : (uint32_t)hash & 1u) << SIGN_SHIFT) | (field << BFLOAT16_FIELD_SHIFT) |
                      ((uint32_t)(hash >> FRACTION_SHIFT) & BFLOAT16_FRACTION_MASK));
}

/* The TREE case's weights and queries of a row. In index_scan_records.h's tree v_g holds heads g, g + 8, g + 16 and
 * g + 24. Heads g and g + 16 of g = 0 .. 3 weigh 2^8 to 2^13; those of g + 4 the opposite, with the queries of g and
 * g + 16, so that the large terms cancel in the exact sum but meet only in its last addition, after rounding the
 * small terms beside them. Heads g + 8 and g + 24 weigh as WIDE's. Every term is exact in float32 (a dot has 10 bits,
 * a weight 8). On such values 81% of the FP16 scores differ between this tree and the heads in order (a host
 * simulation of 28,672 scores), so the scores held bit for bit hold the tree. */
static void tree_weights_of(uint32_t row, uint16_t queries[QUERY_VALUES], uint16_t weights[HEADS])
{
    for (uint32_t head = 0; head < HEADS; head++) {
        const bool is_large = head % HEAD_TILE_HEADS < HALF_TILE_HEADS;
        const bool is_mirror = head % HALF_TILE_HEADS >= MIRROR_DISTANCE;
        weights[head] = weight_of(((uint64_t)CASE_TREE_DIRECT * ROWS + row) * HEADS + head, false, is_large);
        if (is_large && is_mirror) {
            weights[head] = (uint16_t)(weights[head - MIRROR_DISTANCE] ^ (1u << SIGN_SHIFT));
            memcpy(&queries[(size_t)head * DIMS], &queries[(size_t)(head - MIRROR_DISTANCE) * DIMS],
                   DIMS * sizeof(uint16_t));
        }
    }
}

/* The keys, queries and weights of a case, and the host's scores of every visible entry of every row. */
static bool case_make(enum case_kind kind, struct scan_case *made)
{
    const bool has_exact_values = kind != CASE_WIDE_DIRECT;
    const enum test_index_sum sum = kind == CASE_WIDE_DIRECT ? TEST_INDEX_SUM_DOUBLE
                                    : broken == BREAK_TREE    ? TEST_INDEX_SUM_HEADS_IN_ORDER
                                                              : TEST_INDEX_SUM_TREE;
    made->ratio = case_ratios[kind];
    made->entries = CONTEXT / made->ratio;
    made->is_exact = sum != TEST_INDEX_SUM_DOUBLE;
    made->keys = calloc((size_t)SESSIONS * made->entries * DIMS, sizeof(uint16_t));
    made->reference = calloc((size_t)ROWS * made->entries, sizeof(double));
    if (made->keys == NULL || made->reference == NULL) {
        test_linear_fail("the host does not hold a case");
        return false;
    }
    for (uint64_t value = 0; value < (uint64_t)SESSIONS * made->entries * DIMS; value++) {
        made->keys[value] = value_of(KEY_STREAM + kind, value, has_exact_values ? EXACT_KEY_SPREAD : 0,
                                     EXACT_KEY_EXPONENT);
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t value = 0; value < QUERY_VALUES; value++) {
            made->queries[row][value] = row == case_zero_query_rows[kind]
                                            ? 0
                                            : value_of(QUERY_STREAM + kind, (uint64_t)row * QUERY_VALUES + value,
                                                       has_exact_values ? EXACT_QUERY_SPREAD : 0, EXACT_QUERY_EXPONENT);
        }
        for (uint32_t head = 0; head < HEADS; head++) {
            made->weights[row][head] =
                weight_of(((uint64_t)kind * ROWS + row) * HEADS + head, kind != CASE_WIDE_DIRECT, false);
        }
        if (kind == CASE_TREE_DIRECT) {
            tree_weights_of(row, made->queries[row], made->weights[row]);
        }
    }
    const uint16_t kept = made->keys[0];
    if (broken == BREAK_WORD) {                 /* the host's first value of session 0's first key is one more */
        made->keys[0] = test_index_half_nearest(test_index_half_value(kept) + 1.0);
    }
    for (uint32_t row = 0; row < STEP_ROWS; row++) {
        const uint32_t session = row % SESSIONS;
        const uint32_t extent = extent_of(case_positions[kind][row], made->ratio);
        test_index_scores(made->queries[row], made->weights[row], made->keys + (size_t)session * made->entries * DIMS,
                          extent, sum, made->reference + (size_t)row * made->entries);
    }
    made->keys[0] = kept;
    return true;
}

/* The step area of `rows` rows from row `first` of a case's positions. */
static void step_put(struct test_linear_card *card, struct linear_range range,
		     const uint32_t *positions, uint32_t first, uint32_t rows,
		     uint32_t step_rows)
{
	const struct step_block block = { .sequence = 1,
					  .row_count = step_rows,
					  .flags = 0 };
	(void)test_linear_card_put(card, range, 0, &block, sizeof(block));
	for (uint32_t row = 0; row < rows; row++) {
		const struct step_row_record step_row = {
			.position = (int32_t)positions[first + row],
			.token_id = 0,
			.session_slot = (first + row) % SESSIONS,
			.flags = 0
		};
		(void)test_linear_card_put(card, range,
					   STEP_BLOCK_BYTES_VALUE +
						   (uint64_t)row *
							   STEP_ROW_BYTES_VALUE,
					   &step_row, sizeof(step_row));
	}
}

/* What a description of one launch names. */
struct scan_ranges {
	struct linear_range step, keys, queries, weights, scores, selection;
};

/* One guarded allocation contains the payload and, for split shapes, its writable annex. The description names
 * only the payload, keeping the fixture's trailing guard behind all bytes the selection may write. */
static struct linear_range scores_work(struct test_linear_card *card,
				       uint32_t rows, uint32_t entries)
{
    const uint64_t payload_bytes = (uint64_t)rows * entries * SCORE_BYTES;
    const uint64_t scratch_bytes =
	    rows <= INDEX_SPLIT_MAX_ROWS && entries >= INDEX_SPLIT_MIN_ENTRIES ?
		    (uint64_t)rows * INDEX_SPLIT_ROW_BYTES :
		    0;
    struct linear_range scores =
	    test_linear_card_work(card, payload_bytes + scratch_bytes);
    scores.byte_count = payload_bytes;
    return scores;
}

static bool launches_of(const struct scan_ranges *ranges, uint32_t ratio,
			uint32_t context, uint32_t sessions, uint32_t rows,
			const struct test_linear_card *card,
			const struct index_scan_kernels *kernels,
			struct kernel_launch *launches_out,
			uint32_t *launch_count_out)
{
	const struct index_scan_description description = {
		.step = ranges->step,
		.keys = ranges->keys,
		.queries = ranges->queries,
		.weights = ranges->weights,
		.scores = ranges->scores,
		.selection = ranges->selection,
		.context_token_limit = context,
		.session_capacity = sessions,
		.ratio = ratio,
		.row_count = rows
	};
	return test_linear_expect_status(
		index_scan_prepare(&description, card->arena_byte_count,
				   kernels, launches_out, launch_count_out),
		STATUS_OK, "the launches are prepared");
}

/* One row's scores held to the host's: bit for bit, the sign of a zero included (exact), or within the tolerance; -inf
 * from e to the end of its tile; the bytes from there on untouched. Gives the square of the row's largest error over
 * its rms (no square root: the tests link no mathematics library). */
static double row_scores_held(const struct test_linear_card *card,
			      struct linear_range scores, uint32_t row,
			      uint32_t capacity, uint32_t extent,
			      const double *reference, const float *captured,
			      bool is_exact)
{
    uint16_t *const bits = malloc((size_t)capacity * SCORE_BYTES);
    const uint32_t written = (extent + KEYS_PER_TILE - 1u) / KEYS_PER_TILE * KEYS_PER_TILE;
    uint32_t wrong = 0;
    double square_sum = 0.0, largest = 0.0;
    if (bits == NULL) {
        test_linear_fail("the host does not hold a row");
        return INFINITY;
    }
    if (!test_linear_card_get(card, scores, (uint64_t)row * capacity * SCORE_BYTES, bits,
                              (uint64_t)capacity * SCORE_BYTES)) {
        free(bits);
        return INFINITY;
    }
    for (uint32_t entry = 0; entry < extent; entry++) {
        const double expected = captured != NULL ? (double)captured[entry] : reference[entry];
        square_sum += expected * expected;
    }
    const double mean_square = extent != 0 ? square_sum / extent : 0.0;
    for (uint32_t entry = 0; entry < capacity; entry++) {
        if (entry >= written) {
            uint16_t sentinel = 0;
            memset(&sentinel, SENTINEL, sizeof(sentinel));
            wrong += bits[entry] != sentinel ? 1u : 0u;
        } else if (entry >= extent) {
		wrong += bits[entry] != INDEX_NEGATIVE_INFINITY_VALUE ? 1u : 0u;
	} else {
            const double expected = captured != NULL ? (double)captured[entry] : reference[entry];
            const double error = fabs(test_index_half_value(bits[entry]) - expected);
            const uint16_t nearest = test_index_half_nearest(expected);
            const double beyond_rounding = error - fabs(expected) * HALF_PLACE_OF_VALUE - HALF_PLACE_SUBNORMAL;
            largest = mean_square > 0.0 && error * error / mean_square > largest ? error * error / mean_square
                                                                                 : largest;
            wrong += is_exact ? (bits[entry] != nearest ? 1u : 0u)
                              : (beyond_rounding <= 0.0 || beyond_rounding * beyond_rounding <=
                                                                    TOLERANCE_OF_RMS * TOLERANCE_OF_RMS * mean_square
                                     ? 0u : 1u);
        }
    }
    if (wrong != 0) {
        test_linear_fail("row %" PRIu32 ": %" PRIu32 " of its %" PRIu32 " score words are not the host's (e %" PRIu32
                         ", %s)", row, wrong, capacity, extent, is_exact ? "bit for bit" : "within the tolerance");
    }
    free(bits);
    return largest;
}

/* A row's selection is the host's sort of the card's own scores; *shared_out: of the picks of `reference` (NULL:
 * none), how many the card's holds. */
static void row_selection_held(const struct test_linear_card *card,
			       struct linear_range scores,
			       struct linear_range selection, uint32_t row,
			       uint32_t capacity, uint32_t extent,
			       const int32_t *reference,
			       uint32_t reference_count, uint32_t *shared_out)
{
    uint16_t *const bits = malloc((size_t)capacity * SCORE_BYTES);
    int32_t chosen[SELECTED], expected[SELECTED];
    if (bits == NULL) {
        test_linear_fail("the host does not hold a row");
        return;
    }
    if (!test_linear_card_get(card, scores, (uint64_t)row * capacity * SCORE_BYTES, bits,
                              (uint64_t)capacity * SCORE_BYTES) ||
        !test_linear_card_get(card, selection, (uint64_t)row * sizeof(chosen), chosen, sizeof(chosen))) {
        free(bits);
        return;
    }
    test_index_select(bits, extent, broken == BREAK_TIES, expected);
    if (memcmp(chosen, expected, sizeof(chosen)) != 0) {
        test_linear_fail("row %" PRIu32 ": the selection is not the host's sort of the card's scores (e %" PRIu32 ")",
                         row, extent);
    }
    for (uint32_t pick = 0; reference != NULL && pick < reference_count; pick++) {
        bool is_shared = false;
        for (uint32_t output = 0; output < SELECTED && !is_shared; output++) {
            is_shared = chosen[output] == reference[pick];
        }
        *shared_out += is_shared ? 1u : 0u;
    }
    free(bits);
}

/* A synthetic case on one card: the launch of 8 rows, then each row of the step alone. */
static void case_run(struct test_linear_card *card,
		     const struct index_scan_kernels *kernels,
		     enum case_kind kind, const struct scan_case *made)
{
    const uint32_t entries = made->entries;
    const uint64_t stride =
	    (uint64_t)entries / KEYS_PER_TILE * MODEL_TILE_BYTES;
    struct kernel_launch launches[INDEX_SCAN_LAUNCH_COUNT];
    uint32_t launch_count = 0;
    struct scan_ranges ranges, alone;
    test_linear_context("card %" PRIu32 ", case %u", card->card_index, (unsigned)kind);
    test_linear_card_clear(card);
    ranges.step = test_linear_card_work(
	    card, STEP_BLOCK_BYTES_VALUE + ROWS * STEP_ROW_BYTES_VALUE);
    ranges.keys = test_linear_card_work(card, SESSIONS * stride);
    ranges.queries = test_linear_card_work(
	    card, (uint64_t)ROWS * MODEL_INDEX_QUERIES_ROW_BYTES);
    ranges.weights = test_linear_card_work(
	    card, (uint64_t)ROWS * MODEL_INDEX_WEIGHTS_ROW_BYTES);
    ranges.scores = scores_work(card, ROWS, entries);
    ranges.selection = test_linear_card_work(
	    card, (uint64_t)ROWS * MODEL_SELECTION_ROW_BYTES);
    alone = ranges;
    alone.step = test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE +
						     STEP_ROW_BYTES_VALUE);
    alone.scores = scores_work(card, 1, entries);
    alone.selection = test_linear_card_work(card, MODEL_SELECTION_ROW_BYTES);
    unsigned char *const staging = malloc(ranges.keys.byte_count);
    if (staging == NULL || !test_linear_card_has_room(card)) {
        (void)test_linear_expect(staging != NULL, "the host holds the key tiles");
        free(staging);
        return;
    }
    memset(staging, SENTINEL, ranges.scores.byte_count);
    bool is_laid_out = test_linear_card_put(card, ranges.scores, 0, staging, ranges.scores.byte_count) &&
                       test_linear_card_put(card, ranges.selection, 0, staging, ranges.selection.byte_count);
    step_put(card, ranges.step, case_positions[kind], 0, ROWS, STEP_ROWS);
    for (uint64_t key = 0; key < (uint64_t)SESSIONS * entries; key++) {
	    test_index_place_key(made->keys + key * DIMS,
				 staging +
					 key / KEYS_PER_TILE * MODEL_TILE_BYTES,
				 (uint32_t)(key % KEYS_PER_TILE));
    }
    is_laid_out = is_laid_out && test_linear_card_put(card, ranges.keys, 0, staging, ranges.keys.byte_count);
    for (uint32_t row = 0; row < ROWS; row++) {
        test_index_place_queries(made->queries[row], staging);
	is_laid_out = is_laid_out &&
		      test_linear_card_put(
			      card, ranges.queries,
			      (uint64_t)row * MODEL_INDEX_QUERIES_ROW_BYTES,
			      staging, MODEL_INDEX_QUERIES_ROW_BYTES) &&
		      test_linear_card_put(
			      card, ranges.weights,
			      (uint64_t)row * MODEL_INDEX_WEIGHTS_ROW_BYTES,
			      made->weights[row], sizeof(made->weights[row]));
    }
    free(staging);
    if (!is_laid_out ||
        !launches_of(&ranges, made->ratio, CONTEXT, SESSIONS, ROWS, card, kernels, launches, &launch_count) ||
        !test_linear_card_load(card) || !test_linear_card_run(card, launches, launch_count) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    (void)test_linear_card_expect_untouched(card, ranges.step, "the step area");
    (void)test_linear_card_expect_untouched(card, ranges.keys, "the key tiles");
    (void)test_linear_card_expect_untouched(card, ranges.queries, "the queries");
    (void)test_linear_card_expect_untouched(card, ranges.weights, "the weights");
    for (uint32_t row = STEP_ROWS; row < ROWS; row++) {
	    const struct linear_range beyond_scores = {
		    ranges.scores.byte_offset +
			    (uint64_t)row * entries * SCORE_BYTES,
		    (uint64_t)entries * SCORE_BYTES
	    };
	    const struct linear_range beyond_selection = {
		    ranges.selection.byte_offset +
			    (uint64_t)row * MODEL_SELECTION_ROW_BYTES,
		    MODEL_SELECTION_ROW_BYTES
	    };
	    (void)test_linear_card_expect_untouched(
		    card, beyond_scores, "the scores of a row beyond the step");
	    (void)test_linear_card_expect_untouched(
		    card, beyond_selection,
		    "the selection of a row beyond the step");
    }
    for (uint32_t row = 0; row < STEP_ROWS; row++) {
        const uint32_t extent = extent_of(case_positions[kind][row], made->ratio);
        const uint32_t host_extent = extent + (broken == BREAK_EXTENT && row == break_extent_row ? 1u : 0u);
        (void)row_scores_held(card, ranges.scores, row, entries, host_extent,
                              made->reference + (size_t)row * entries, NULL, made->is_exact);
        row_selection_held(card, ranges.scores, ranges.selection, row, entries, host_extent, NULL, 0, NULL);
    }
    for (uint32_t row = 0; row < STEP_ROWS; row++) {     /* R: each row alone */
        const uint32_t written = (extent_of(case_positions[kind][row], made->ratio) + KEYS_PER_TILE - 1u) /
                                 KEYS_PER_TILE * KEYS_PER_TILE;
	alone.queries = (struct linear_range){
		ranges.queries.byte_offset +
			(uint64_t)row * MODEL_INDEX_QUERIES_ROW_BYTES,
		MODEL_INDEX_QUERIES_ROW_BYTES
	};
	alone.weights = (struct linear_range){
		ranges.weights.byte_offset +
			(uint64_t)row * MODEL_INDEX_WEIGHTS_ROW_BYTES,
		MODEL_INDEX_WEIGHTS_ROW_BYTES
	};
	step_put(card, alone.step, case_positions[kind], row, 1, 1);
        if (written == 0) {
            continue;                           /* nothing to compare but the selection of -1, held above */
        }
        if (!launches_of(&alone, made->ratio, CONTEXT, SESSIONS, 1, card, kernels, launches, &launch_count) ||
            !test_linear_card_upload(card, alone.step, 0, alone.step.byte_count) ||
            !test_linear_card_run(card, launches, launch_count) ||
            !test_linear_card_fetch(card)) {
            return;
        }
        (void)test_linear_card_expect_same(card, alone.scores, 0, ranges.scores, (uint64_t)row * entries * SCORE_BYTES,
                                           (uint64_t)written * SCORE_BYTES, "R: a row's scores alone");
	(void)test_linear_card_expect_same(
		card, alone.selection, 0, ranges.selection,
		(uint64_t)row * MODEL_SELECTION_ROW_BYTES,
		MODEL_SELECTION_ROW_BYTES, "R: a row's selection alone");
    }
}

/* ---- the replay ---- */

/* A record of FP16 values (held as float32 by the writer, or as float16) of the given shape, as FP16 bits. */
static bool halves_read(const struct test_capture *capture, const char *name, const uint64_t *dimensions,
                        uint32_t dimension_count, uint64_t count, uint16_t *halves_out)
{
    const struct test_capture_tensor *const record = test_capture_find(capture, name);
    if (record != NULL && record->type == TEST_CAPTURE_FLOAT16) {
	    return test_linear_expect_status(
		    test_capture_read(capture, name, TEST_CAPTURE_FLOAT16,
				      dimensions, dimension_count, 0, count,
				      halves_out, count * sizeof(uint16_t)),
		    STATUS_OK, name);
    }
    float *const values = malloc((size_t)count * sizeof(float));
    bool is_read =
	    values != NULL &&
	    test_linear_expect_status(
		    test_capture_read(capture, name, TEST_CAPTURE_FLOAT32,
				      dimensions, dimension_count, 0, count,
				      values, count * sizeof(float)),
		    STATUS_OK, name);
    for (uint64_t value = 0; is_read && value < count; value++) {
        halves_out[value] = test_index_half_nearest((double)values[value]);
        is_read = test_index_half_value(halves_out[value]) == (double)values[value];
    }
    if (values != NULL && !is_read) {
        test_linear_fail("%s holds a value that is no FP16 number", name);
    }
    free(values);
    return is_read;
}

/* A record of one dimension, of count elements of the type. */
static bool vector_read(const struct test_capture *capture, const char *name, enum test_capture_type type,
                        uint64_t count, void *destination, uint64_t element_bytes)
{
    const uint64_t shape[1] = {count};
    return test_linear_expect_status(
	    test_capture_read(capture, name, type, shape, 1, 0, count,
			      destination, count * element_bytes),
	    STATUS_OK, name);
}

/* The share of the reference's picks a replay's rows hold: rows above 512 entries, and the others (all picked). */
struct replay_shares {
    uint32_t rows;
    uint32_t counted_rows;
    double counted_share_sum;
};

/* One captured index layer at one position, if the folder holds its queries and a snapshot of the owner's keys at or
 * after it: false where it does not. */
static bool replay_case(struct test_linear_card *card,
			const struct index_scan_kernels *kernels,
			const struct test_capture *capture,
			const uint32_t *layer, uint32_t position,
			struct replay_shares *shares)
{
    char query_name[NAME_CAPACITY], weights_name[NAME_CAPACITY], keys_name[NAME_CAPACITY];
    char scores_name[NAME_CAPACITY], selection_name[NAME_CAPACITY];
    const uint32_t ratio = layer[RATIO], extent = extent_of(position, ratio);
    const uint32_t context = (position + MODEL_CONTEXT_GRANULE) /
			     MODEL_CONTEXT_GRANULE * MODEL_CONTEXT_GRANULE;
    const uint32_t entries = context / ratio, picks = extent < SELECTED ? extent : SELECTED;
    uint32_t snapshot = 0;
    bool has_keys = false;
    (void)snprintf(query_name, sizeof(query_name), "layer_%" PRIu32 ".position_%" PRIu32 ".indexer.query",
                   layer[LAYER], position);
    for (size_t index = 0; index < SNAPSHOTS && !has_keys; index++) {
        snapshot = snapshot_positions[index];
        (void)snprintf(keys_name, sizeof(keys_name), "layer_%" PRIu32 ".position_%" PRIu32 ".snapshot.keys",
                       layer[OWNER], snapshot);
        has_keys = snapshot >= position && test_capture_find(capture, keys_name) != NULL;
    }
    if (extent == 0 || !has_keys || test_capture_find(capture, query_name) == NULL) {
        return false;                           /* no entry to score, no keys, or not a capture point */
    }
    (void)snprintf(weights_name, sizeof(weights_name), "layer_%" PRIu32 ".position_%" PRIu32 ".indexer.weights",
                   layer[LAYER], position);
    (void)snprintf(scores_name, sizeof(scores_name), "layer_%" PRIu32 ".position_%" PRIu32 ".indexer.scores",
                   layer[LAYER], position);
    (void)snprintf(selection_name, sizeof(selection_name), "layer_%" PRIu32 ".position_%" PRIu32 ".indexer.selection",
                   layer[LAYER], position);
    const uint64_t query_shape[DIMENSIONS_OF_A_MATRIX] = {HEADS, DIMS};
    const uint64_t keys_shape[DIMENSIONS_OF_A_MATRIX] = {extent_of(snapshot, ratio), DIMS};
    static uint16_t query[QUERY_VALUES];
    uint16_t weights[HEADS];
    int32_t reference_picks[SELECTED];
    uint16_t *const keys = calloc((size_t)entries * DIMS, sizeof(uint16_t));
    float *const scores = malloc((size_t)extent * sizeof(float));
    /* the key tiles, and first the scores' sentinel and then the queries, as the image takes them */
    unsigned char *const tiles =
	    malloc((size_t)entries / KEYS_PER_TILE * MODEL_TILE_BYTES +
		   MODEL_INDEX_QUERIES_ROW_BYTES);
    struct kernel_launch launches[INDEX_SCAN_LAUNCH_COUNT];
    uint32_t launch_count = 0;
    struct scan_ranges ranges;
    uint32_t shared = 0;
    test_linear_context("card %" PRIu32 ", replay of layer %" PRIu32 " at position %" PRIu32 ", keys of %" PRIu32,
                        card->card_index, layer[LAYER], position, snapshot);
    test_linear_card_clear(card);
    ranges.step = test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE +
						      STEP_ROW_BYTES_VALUE);
    ranges.keys = test_linear_card_work(
	    card, (uint64_t)entries / KEYS_PER_TILE * MODEL_TILE_BYTES);
    ranges.queries = test_linear_card_work(card, MODEL_INDEX_QUERIES_ROW_BYTES);
    ranges.weights = test_linear_card_work(card, MODEL_INDEX_WEIGHTS_ROW_BYTES);
    ranges.scores = scores_work(card, 1, entries);
    ranges.selection = test_linear_card_work(card, MODEL_SELECTION_ROW_BYTES);
    if (keys == NULL || scores == NULL || tiles == NULL) {
        test_linear_fail("the host does not hold a replay");
    } else if (test_linear_card_has_room(card) &&
               halves_read(capture, query_name, query_shape, DIMENSIONS_OF_A_MATRIX, QUERY_VALUES, query) &&
               halves_read(capture, keys_name, keys_shape, DIMENSIONS_OF_A_MATRIX, (uint64_t)extent * DIMS, keys) &&
               vector_read(capture, weights_name, TEST_CAPTURE_BFLOAT16, HEADS, weights, sizeof(uint16_t)) &&
               vector_read(capture, scores_name, TEST_CAPTURE_FLOAT32, extent, scores, sizeof(float)) &&
               vector_read(capture, selection_name, TEST_CAPTURE_INT32, picks, reference_picks, sizeof(int32_t))) {
        const uint32_t replay_position[1] = {position};
        memset(tiles, SENTINEL, ranges.scores.byte_count);
        bool is_laid_out = test_linear_card_put(card, ranges.scores, 0, tiles, ranges.scores.byte_count);
        step_put(card, ranges.step, replay_position, 0, 1, 1);
        for (uint32_t key = 0; key < entries; key++) {
		test_index_place_key(keys + (size_t)key * DIMS,
				     tiles + (size_t)key / KEYS_PER_TILE *
						     MODEL_TILE_BYTES,
				     key % KEYS_PER_TILE);
	}
        is_laid_out = is_laid_out && test_linear_card_put(card, ranges.keys, 0, tiles, ranges.keys.byte_count);
        test_index_place_queries(query, tiles);
	is_laid_out = is_laid_out &&
		      test_linear_card_put(card, ranges.queries, 0, tiles,
					   MODEL_INDEX_QUERIES_ROW_BYTES) &&
		      test_linear_card_put(card, ranges.weights, 0, weights,
					   sizeof(weights));
	if (is_laid_out && launches_of(&ranges, ratio, context, 1, 1, card, kernels, launches, &launch_count) &&
            test_linear_card_load(card) && test_linear_card_run(card, launches, launch_count) &&
            test_linear_card_fetch(card)) {
            const double largest = row_scores_held(card, ranges.scores, 0, entries, extent, NULL, scores, false);
            double largest_score = 0.0;
            for (uint32_t entry = 0; entry < extent; entry++) {
                const double magnitude = fabs((double)scores[entry]);
                largest_score = magnitude > largest_score ? magnitude : largest_score;
            }
            row_selection_held(card, ranges.scores, ranges.selection, 0, entries, extent, reference_picks, picks,
                               &shared);
            shares->rows++;
            shares->counted_rows += extent > SELECTED ? 1u : 0u;
            shares->counted_share_sum += extent > SELECTED ? (double)shared / picks : 0.0;
            (void)printf("REPLAY card=%" PRIu32 " layer=%" PRIu32 " position=%" PRIu32 " keys_of=%" PRIu32
                         " entries=%" PRIu32 " shared=%" PRIu32 "/%" PRIu32 "%s largest_error_over_rms_squared=%.3g"
                         " largest_score=%.6g\n", card->card_index, layer[LAYER], position, snapshot, extent, shared,
                         picks, extent > SELECTED ? "" : " (all picked: not in the mean)", largest, largest_score);
        }
    }
    free(tiles);
    free(scores);
    free(keys);
    return true;
}

static void card_run(uint32_t card_index, const struct scan_case *cases, const struct test_capture *capture)
{
    static struct test_linear_card card;
    struct index_scan_kernels kernels;
    struct replay_shares shares = {0, 0, 0.0};
    uint32_t replayed = 0;
    if (!test_linear_card_open(&card, card_index, WORK_CAPACITY, 0)) {
        return;
    }
    if (test_linear_expect_status(
		index_scan_kernels_prepare(&backend_cuda_value, card.card,
					   card.arithmetic_class, &kernels),
		STATUS_OK, "the kernels of the index scan are prepared")) {
	    for (uint32_t kind = 0; kind < CASE_COUNT; kind++) {
		    case_run(&card, &kernels, (enum case_kind)kind,
			     &cases[kind]);
	    }
	    for (size_t layer = 0;
		 capture != NULL &&
		 layer < sizeof(index_layers) / sizeof(index_layers[0]);
		 layer++) {
		    for (uint32_t position = 0;
			 position <= snapshot_positions[SNAPSHOTS - 1u];
			 position++) {
			    replayed += replay_case(&card, &kernels, capture,
						    index_layers[layer],
						    position, &shares) ?
						1u :
						0u;
		    }
	    }
    }
    const double mean_share = shares.counted_rows != 0 ? shares.counted_share_sum / shares.counted_rows : 0.0;
    test_linear_context("card %" PRIu32, card_index);
    if (capture != NULL && (replayed == 0 || shares.counted_rows == 0 || mean_share < SHARE_REQUIRED)) {
        test_linear_fail("%" PRIu32 " rows replayed, %" PRIu32 " of them above %u entries, which share %.4f of the "
                         "reference's picks on average: at least one such row (the 2,048-token text's stepwise "
                         "folder), and %.2f", replayed, shares.counted_rows, SELECTED, mean_share, SHARE_REQUIRED);
    }
    (void)printf("card %" PRIu32 ": digest %016" PRIx64 ", %" PRIu32 " rows replayed, %" PRIu32 " above %u entries "
                 "sharing %.4f of the picks\n", card_index, card.digest, shares.rows, shares.counted_rows, SELECTED,
                 mean_share);
    test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
    static struct scan_case cases[CASE_COUNT];
    static const char *const break_names[] = {"", "ties", "extent", "word", "tree"};
    struct test_capture *capture = NULL;
    uint32_t card_count = 0;
    int argument = 1;
    if (argument + 1 < argument_count && strcmp(arguments[argument], "--break") == 0) {
        for (uint32_t kind = BREAK_TIES; kind <= BREAK_TREE; kind++) {
            broken = strcmp(arguments[argument + 1], break_names[kind]) == 0 ? (enum break_kind)kind : broken;
        }
        argument += broken != BREAK_NONE ? BREAK_ARGUMENTS : 0;
    }
    if (argument_count - argument > 1 || (argument < argument_count && arguments[argument][0] == '-')) {
        (void)fprintf(stderr, "usage: index_scan_card_test [--break ties|extent|word|tree] "
                              "[<stepwise folder of the 2,048-token text>]\n");
        return TEST_WRONG_USAGE;
    }
    (void)alarm(TIME_LIMIT_SECONDS);
    if (argument < argument_count &&
	!test_linear_expect_status(test_capture_open(arguments[argument],
						     &capture),
				   STATUS_OK, "the capture folder opens")) {
	    return TEST_FAILED;
    }
    for (uint32_t kind = 0; kind < CASE_COUNT; kind++) {
        (void)case_make((enum case_kind)kind, &cases[kind]);
    }
    if (test_linear_failures() == 0 &&
	test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    card_run(card_index, cases, capture);
	    }
    }
    for (uint32_t kind = 0; kind < CASE_COUNT; kind++) {
        free(cases[kind].keys);
        free(cases[kind].reference);
    }
    test_capture_close(capture);
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "index_scan_card_test FAILED%s%s: %d failure lines\n",
                      broken != BREAK_NONE ? ", break " : "", break_names[broken], test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("index_scan_card_test passed on %" PRIu32 " cards%s\n", card_count,
                 capture != NULL ? ", replay included" : " (no capture folder: nothing replayed)");
    return TEST_PASSED;
}
