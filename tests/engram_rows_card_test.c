/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_rows_card_test: the two kernels of an n-gram layer (source/kernel_engram_rows.cu) on every card, against
 * host references in double precision, and replayed on the reference's own inputs.
 *
 *     engram_rows_card_test [[--break rows|gate] <capture folder> <pack copy> [<pack copy> ...]]
 *
 * On every card, always, through engram_rows_prepare:
 *   - values: 8 rows of 32 table rows that hold each pair of an E4M3 byte and a scale code once, each value
 *     bit-equal to the byte's number times 2^(c - 127) rounded once to float32, then to bfloat16 (model.py:317-320);
 *   - the gate: 8 rows of four streams, a case a row (plain; keys that make every term of the dot positive, a gate
 *     near 1; every one negative, near 0; zero keys, which the clamp of 1e-6 decides; a zero stream; keys whose mean
 *     square is far below epsilon, which then decides; large numbers; a not-a-number in a stream and an infinity in
 *     the value). The stream is zero in its upper half, where the value spans every bfloat16 fraction, so that the
 *     outputs show the gate. Every output lies between the two the card's last two roundings make of the host's gate
 *     moved by the error bound of the card's float32 arithmetic (the sums' longest chain of additions, every
 *     rounding, two last places of rsqrtf and expf), a not-a-number where the arithmetic gives one; the share
 *     bit-equal to the host's gate in float32 is printed;
 *   - launches of 1, 5 and 8 rows of cases moved to other rows (case 5 at row 0; cases 3 to 7 at rows 0 to 4; case
 *     r at row r - 1, case 0 at row 7) give each case the bits it had at its own row of 8, and leave the rows behind
 *     untouched;
 *   - each comparison shown failing: one stored word changed in the host's copy is reported.
 * With a capture folder of tools/oracle/stepwise_capture.py and the pack the reference ran, for the n-gram layers 1
 * and 14 at every position, 8 positions a launch (a last, shorter batch beside stale rows):
 *   - the row ids engram_row_ids_entry gives for the tokens fed are layer_L.ngram.row_ids;
 *   - the values of those table rows, read from the pack, are layer_L.ngram.rows, bit for bit;
 *   - once for each slice count engram.wkv has on the card under the two profiles and the two orders (linear.h), the
 *     chain (values, the linear's quantization and engram.wkv from the pack, the gate) from the streams leaving the
 *     layer before (layer_L-1.block.output): engram.wkv's output at the capture points 99.9% bit-equal to
 *     layer_L.position_P.ngram.projection (the linear's gate; the rest is printed, not bounded: g01 found correct
 *     engram.wkv outputs 2 to 160 places apart), the streams within one bfloat16 place of layer_L.ngram.output on
 *     99.9%;
 *   - the gate alone at the capture points, from the captured projection: within one place on 99.9%.
 * Beside the last two, the share of captured outputs the n-gram add moved more than one place: where a wrong gate
 * can show. --break rows reads the table row behind each row id; --break gate leaves the gate out of the chain and
 * gives the gate alone SHORT_GATE_SHARE of each captured value, a gate 10% short: the test then passes only if the
 * checks the break aims at reported and every other held. Without arguments the replay compares nothing.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/engram.h"
#include "anemone/engram_rows.h"
#include "anemone/engram_rows_records.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/pack.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"
#include "anemone/weight_format_seam.h"
#include "capture_reader.h"
#include "linear_card_fixture.h"
#include "engram_rows_reference.h"
#include "linear_reference.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define TEST_TIME_LIMIT_SECONDS 7200            /* the alarm ends the test here, whatever a drive or a driver does */
#define HIDDEN MODEL_HIDDEN
#define STREAMS MODEL_STREAMS
#define ROWS LINEAR_MAXIMUM_ROWS
#define TABLE_ROWS TEST_ENGRAM_ROWS_TABLE_ROWS
#define CASE_TABLE_ROWS TEST_ENGRAM_ROWS_PAIR_TABLE_ROWS
#define ROW_VALUES ENGRAM_ROWS_VALUES
#define ROW_BYTES ENGRAM_ROWS_ROW_BYTES
#define CASE_VALUES 8192u                       /* of a row of the values case: 32 table rows of 256 */
#define CASE_ROW_BYTES 8448u                    /* its table rows: 32 of 264 bytes */
#define VALUES_WIDTH 6144u                      /* engram.wkv's inputs: 24 table rows of 256 */
#define PROJECTION ENGRAM_ROWS_PROJECTION
#define STREAM_VALUES 20480u                    /* of a row: 4 streams of 5,120 */
#define VALUE_BYTES 2u
#define WORK_CAPACITY (8ull << 20u)             /* eight rows of every array, and their guards */
#define WEIGHTS_CAPACITY (160ull << 20u)        /* engram.wkv: 157,440,000 bytes, read in whole blocks */
#define BFLOAT16_SIGN 0x8000u
#define CHANGED_PLACE 3u                        /* where one stored word is changed to show a check failing */
#define SHORT_GATE_SHARE 0.9                    /* --break gate: what the gate alone adds of each captured value */
#define KNOB_COUNT 4u                           /* two profiles by two orders, each a slice count of engram.wkv */
#define FIGURES_CAPACITY 256u
#define RUNS_OF_MOVED_CASES 3u
#define FIRST_ARGUMENT_AFTER_BREAK 3
#define BREAK_KIND_ARGUMENT 2
#define ENGRAM_LAYER_COUNT 2u
#define NAME_CAPACITY 96u
#define MOST_POSITIONS \
	MODEL_CONTEXT_WITHOUT_CANDIDATES /* the context of the first milestone */
#define PACK_BLOCK ((uint64_t)PACK_BLOCK_BYTES)
#define TABLE_ROWS_PER_BLOCK ((uint64_t)ENGRAM_ROWS_PER_BLOCK)
#define MATRIX_DIMENSIONS 2u
#define MATRIX_BYTES 157440000ull               /* engram.wkv: 25,600 by 6,144 E4M3 and a scale per 32 by 32 */
#define E4M3_BITS_PER_WEIGHT 8u
#define WEIGHT_READ_CAPACITY 45056u             /* 40,960 bytes from anywhere in a 4,096-byte block: 11 blocks */
#define STREAM_DIMENSIONS 4u                    /* [sequence, position, stream, value] */
#define ROWS_DIMENSIONS 4u                      /* [sequence, position, table row, value] */
#define IDS_DIMENSIONS 3u
#define TOKENS_DIMENSIONS 2u
#define POSITION_AXIS 1u

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum break_kind { BREAK_NONE, BREAK_ROWS, BREAK_GATE };
enum replay_check { CHECK_ROW_IDS, CHECK_VALUES, CHECK_PROJECTION, CHECK_CHAIN, CHECK_GATE_ALONE, CHECK_COUNT };
enum launch_of_chain { CHAIN_VALUES, CHAIN_QUANTIZE, CHAIN_PROJECTION, CHAIN_GATE, CHAIN_LAUNCH_COUNT };

static const uint32_t engram_layers[ENGRAM_LAYER_COUNT] = {1, 14};      /* config.json engram_layer_ids, in order */
static const uint32_t moved_row_counts[RUNS_OF_MOVED_CASES] = {1, 5, 8};
static const uint32_t moved_by[RUNS_OF_MOVED_CASES] = {5, 3, 1};     /* row r holds case (r + moved_by) % 8 */
static const char *const check_names[CHECK_COUNT] = {"row ids", "values", "projection", "chain", "gate alone"};
_Static_assert(TEST_ENGRAM_ROWS_GATE_CASE_COUNT == ROWS, "a gate case for each row of a launch");
_Static_assert(VALUES_WIDTH == TABLE_ROWS * ROW_VALUES &&
		       STREAM_VALUES == STREAMS * HIDDEN &&
		       CASE_VALUES == CASE_TABLE_ROWS * ROW_VALUES &&
		       CASE_ROW_BYTES == CASE_TABLE_ROWS * ROW_BYTES &&
		       CASE_TABLE_ROWS >= TABLE_ROWS &&
		       WEIGHT_READ_CAPACITY >=
			       STREAM_VALUES * VALUE_BYTES + PACK_BLOCK_BYTES,
	       "the sizes are the model's");

/* Where a card case keeps its arrays. */
struct card_ranges {
	struct engram_rows_description description;
	struct linear_range codes;
	struct linear_range exponents;
	struct linear_range operands;
};

/* What the comparisons of one kind added up to. */
struct tally {
    uint64_t count;
    uint64_t equal;
    uint64_t within_one;
    uint64_t moved;                             /* captured outputs the n-gram add moved more than one place */
    uint32_t largest_distance;
};

/* The host's copies of a case's arrays. */
static uint8_t host_rows[ROWS * CASE_ROW_BYTES];
static uint16_t host_streams[ROWS * STREAM_VALUES];
static uint16_t host_projection[ROWS * PROJECTION];
static uint16_t host_query_weight[STREAM_VALUES];
static uint16_t host_key_weight[STREAM_VALUES];
static uint16_t card_values[ROWS * CASE_VALUES];        /* of the values case, or of the replay's 24 table rows */
static uint16_t card_streams[ROWS * STREAM_VALUES];
static uint16_t first_values[ROWS * CASE_VALUES];       /* of the launch of 8 rows */
static uint16_t first_streams[ROWS * STREAM_VALUES];
static uint16_t from_capture[ROWS * STREAM_VALUES];     /* of the replay: what the capture holds */
static uint16_t card_projection[ROWS * PROJECTION];
static uint16_t captured_projection[PROJECTION];
static struct card_ranges ranges_of_replay;

/* ---- numbers ---- */

/* Whether two bfloat16 lie within one place of each other (two not-a-numbers do); the distance into a tally. */
static void tally_add(struct tally *tally, uint16_t card_bits, uint16_t reference_bits)
{
    const bool both_missing = isnan(test_linear_bfloat16_value(card_bits)) &&
                              isnan(test_linear_bfloat16_value(reference_bits));
    const int32_t distance = test_linear_bfloat16_ordinal(card_bits) - test_linear_bfloat16_ordinal(reference_bits);
    const uint32_t places = both_missing ? 0u : (uint32_t)(distance < 0 ? -distance : distance);
    const bool is_number = !isnan(test_linear_bfloat16_value(card_bits)) &&
                           !isnan(test_linear_bfloat16_value(reference_bits));
    tally->count++;
    tally->equal += card_bits == reference_bits || both_missing ? 1u : 0u;
    tally->within_one += both_missing || (is_number && places <= 1) ? 1u : 0u;
    if (is_number && places > tally->largest_distance) {
        tally->largest_distance = places;
    }
}

/* Whether the n-gram add moved a captured output more than one place from the stream entering it, into a tally: the
 * outputs where a wrong gate can show in the chain's and the gate's checks. */
static void power_add(struct tally *tally, uint16_t entering_bits, uint16_t captured_bits)
{
    const int32_t distance = test_linear_bfloat16_ordinal(captured_bits) - test_linear_bfloat16_ordinal(entering_bits);
    tally->moved += distance > 1 || distance < -1 ? 1u : 0u;
}

static bool all_zero(const uint16_t *values, size_t count)
{
    for (size_t place = 0; place < count; place++) {
        if (values[place] != 0) {
            return false;
        }
    }
    return true;
}

/* ---- the synthetic cases ---- */

static uint64_t values_differing(const uint16_t *values, uint32_t row_count)
{
    uint64_t differing = 0;
    for (uint32_t table_row = 0; table_row < row_count * CASE_TABLE_ROWS; table_row++) {
        const uint8_t *const row = &host_rows[(size_t)table_row * ROW_BYTES];
        for (uint32_t place = 0; place < ROW_VALUES; place++) {
		const uint32_t scale_code =
			row[ROW_VALUES + place / ENGRAM_ROWS_VALUES_PER_SCALE];
		const uint16_t expected =
			test_engram_rows_value(row[place], scale_code);
		differing += values[(size_t)table_row * ROW_VALUES + place] !=
					     expected ?
				     1u :
				     0u;
        }
    }
    return differing;
}

/* The outputs of rows 0 .. row_count - 1 against the gate references; the first outside named. */
static uint64_t gate_outside(const uint16_t *outputs, uint32_t row_count, uint64_t *equal_count, bool is_named)
{
    uint64_t outside = 0;
    for (uint32_t row = 0; row < row_count; row++) {
        for (uint32_t stream = 0; stream < STREAMS; stream++) {
            const size_t stream_place = ((size_t)row * STREAMS + stream) * HIDDEN;
            const uint16_t *const key = &host_projection[(size_t)row * PROJECTION + (size_t)stream * HIDDEN];
            const uint16_t *const value = &host_projection[(size_t)row * PROJECTION + STREAM_VALUES];
            const struct test_engram_rows_gate reference =
                test_engram_rows_gate_of(&host_streams[stream_place], key, &host_query_weight[(size_t)stream * HIDDEN],
                                         &host_key_weight[(size_t)stream * HIDDEN]);
            uint32_t first = 0;
            const uint64_t count = test_engram_rows_gate_outside(&reference, &host_streams[stream_place], value,
                                                                 &outputs[stream_place], equal_count, &first);
            if (count != 0 && is_named) {
                test_linear_fail("row %" PRIu32 " (case %" PRIu32 "), stream %" PRIu32 ": %" PRIu64 " outputs outside "
                                 "the gate's bound, the first at %" PRIu32 ": 0x%04x, the host's gate %.9g in [%.9g, "
                                 "%.9g]", row, row, stream, count, first, (unsigned)outputs[stream_place + first],
                                 reference.central, reference.lowest, reference.highest);
            }
            outside += count;
        }
    }
    return outside;
}

/* Lays out a card case: every array of the chain for 8 rows of table_rows table rows, in the work region. */
static void card_ranges_set(struct test_linear_card *card, struct card_ranges *ranges, uint32_t table_rows)
{
	struct engram_rows_description *const description =
		&ranges->description;
	test_linear_card_clear(card);
	memset(ranges, 0, sizeof(*ranges));
	description->rows_per_token = table_rows;
	description->row_count = ROWS;
	description->rows = test_linear_card_work(
		card, (uint64_t)ROWS * table_rows * ROW_BYTES);
	description->values = test_linear_card_work(
		card, linear_values_byte_count(ROWS, (uint64_t)table_rows *
							     ROW_VALUES));
	ranges->codes = test_linear_card_work(
		card, linear_codes_byte_count(ROWS, VALUES_WIDTH));
	ranges->exponents = test_linear_card_work(
		card, linear_exponents_byte_count(ROWS, VALUES_WIDTH));
	ranges->operands = test_linear_card_work(
		card, linear_values_byte_count(ROWS, VALUES_WIDTH));
	description->projection =
		test_linear_card_work(card, sizeof(host_projection));
	description->query_weight =
		test_linear_card_work(card, sizeof(host_query_weight));
	description->key_weight =
		test_linear_card_work(card, sizeof(host_key_weight));
	description->streams =
		test_linear_card_work(card, sizeof(host_streams));
}

static bool launches_prepare(const struct card_ranges *ranges,
			     uint32_t row_count, uint64_t arena_byte_count,
			     const struct engram_rows_kernels *kernels,
			     struct kernel_launch *launches_out)
{
	struct engram_rows_description description = ranges->description;
	description.row_count = row_count;
	return test_linear_expect_status(
		engram_rows_prepare(&description, arena_byte_count, kernels,
				    launches_out),
		STATUS_OK, "the launches of the n-gram rows are prepared");
}

/* Runs one launch of row_count rows on the arrays as the image holds them, and fetches the outputs. */
static bool card_case_run(struct test_linear_card *card,
			  const struct card_ranges *ranges, uint32_t row_count,
			  const struct engram_rows_kernels *kernels,
			  uint32_t launch_index)
{
	struct kernel_launch launches[ENGRAM_ROWS_LAUNCH_COUNT];
	const struct engram_rows_description *const description =
		&ranges->description;
	return launches_prepare(ranges, row_count, card->arena_byte_count,
				kernels, launches) &&
	       test_linear_card_upload(card, description->values, 0,
				       description->values.byte_count) &&
	       test_linear_card_upload(card, description->streams, 0,
				       description->streams.byte_count) &&
	       test_linear_card_run(card, &launches[launch_index], 1) &&
	       test_linear_card_fetch(card) &&
	       test_linear_card_get(card, description->values, 0, card_values,
				    description->values.byte_count) &&
	       test_linear_card_get(card, description->streams, 0, card_streams,
				    sizeof(card_streams));
}

/* Row r of the image's table rows, projections and streams given case (r + shift) % 8; the first two uploaded (a run
 * uploads the streams). */
static bool cases_move(struct test_linear_card *card,
		       const struct engram_rows_description *description,
		       uint32_t shift)
{
    bool is_put = true;
    for (uint32_t row = 0; row < ROWS && is_put; row++) {
        const size_t moved = (row + shift) % ROWS;
        is_put = test_linear_card_put(card, description->rows, (uint64_t)row * CASE_ROW_BYTES,
                                      &host_rows[moved * CASE_ROW_BYTES], CASE_ROW_BYTES) &&
                 test_linear_card_put(card, description->projection, (uint64_t)row * PROJECTION * VALUE_BYTES,
                                      &host_projection[moved * PROJECTION], (uint64_t)PROJECTION * VALUE_BYTES) &&
                 test_linear_card_put(card, description->streams, (uint64_t)row * STREAM_VALUES * VALUE_BYTES,
                                      &host_streams[moved * STREAM_VALUES], (uint64_t)STREAM_VALUES * VALUE_BYTES);
    }
    return is_put && test_linear_card_upload(card, description->rows, 0, description->rows.byte_count) &&
           test_linear_card_upload(card, description->projection, 0, description->projection.byte_count);
}

/* Whether row r of outputs (row_values each) holds, below row_count, the bits case (r + shift) % 8 had in the launch
 * of 8 rows (of_eight), and behind it what the image held for that case (before; NULL: zeros). */
static bool moved_rows_hold(const uint16_t *outputs, const uint16_t *of_eight, const uint16_t *before,
                            size_t row_values, uint32_t row_count, uint32_t shift)
{
    bool holds = true;
    for (uint32_t row = 0; row < ROWS; row++) {
        const uint16_t *const output = &outputs[row * row_values];
        const size_t moved = (row + shift) % ROWS * row_values;
        holds = holds && (row < row_count ? memcmp(output, &of_eight[moved], row_values * VALUE_BYTES) == 0
                          : before != NULL ? memcmp(output, &before[moved], row_values * VALUE_BYTES) == 0
                                           : all_zero(output, row_values));
    }
    return holds;
}

static void synthetic_cases(struct test_linear_card *card,
			    const struct engram_rows_kernels *kernels)
{
    struct card_ranges ranges;
    const struct engram_rows_description *const description =
	    &ranges.description;
    uint64_t equal_count = 0;
    uint64_t ignored = 0;
    test_engram_rows_rows_fill(host_rows, ROWS * CASE_TABLE_ROWS);
    test_engram_rows_gate_fill(host_streams, host_projection, host_query_weight, host_key_weight);
    card_ranges_set(card, &ranges, CASE_TABLE_ROWS);
    if (!test_linear_card_has_room(card) ||
        !test_linear_card_put(card, description->rows, 0, host_rows, sizeof(host_rows)) ||
        !test_linear_card_put(card, description->projection, 0, host_projection, sizeof(host_projection)) ||
        !test_linear_card_put(card, description->query_weight, 0, host_query_weight, sizeof(host_query_weight)) ||
        !test_linear_card_put(card, description->key_weight, 0, host_key_weight, sizeof(host_key_weight)) ||
        !test_linear_card_put(card, description->streams, 0, host_streams, sizeof(host_streams)) ||
        !test_linear_card_load(card)) {
        return;
    }
    test_linear_context("card %" PRIu32 ": values", card->card_index);
    if (card_case_run(card, &ranges, ROWS, kernels,
		      ENGRAM_ROWS_VALUES_LAUNCH)) {
	    test_linear_expect(values_differing(card_values, ROWS) == 0,
			       "every value is the host's, bit for bit");
	    memcpy(first_values, card_values, sizeof(first_values));
	    host_rows[CHANGED_PLACE] ^=
		    1u; /* one stored word changed in the host's copy alone */
	    test_linear_expect(values_differing(card_values, ROWS) != 0,
			       "a changed code is reported");
	    host_rows[CHANGED_PLACE] ^= 1u;
    }
    test_linear_context("card %" PRIu32 ": the gate", card->card_index);
    if (card_case_run(card, &ranges, ROWS, kernels, ENGRAM_ROWS_GATE_LAUNCH)) {
	    test_linear_expect(
		    gate_outside(card_streams, ROWS, &equal_count, true) == 0,
		    "every output lies within the bound of the gate");
	    memcpy(first_streams, card_streams, sizeof(first_streams));
	    host_streams[CHANGED_PLACE] ^= BFLOAT16_SIGN;
	    test_linear_expect(gate_outside(card_streams, 1, &ignored, false) !=
				       0,
			       "a changed stream value is reported");
	    host_streams[CHANGED_PLACE] ^= BFLOAT16_SIGN;
	    (void)printf(
		    "card %" PRIu32
		    ": the gate: %.4f%% of %u outputs bit-equal to the host's gate rounded to "
		    "float32\n",
		    card->card_index,
		    test_linear_percent(equal_count,
					(uint64_t)ROWS * STREAM_VALUES),
		    ROWS * STREAM_VALUES);
    }
    for (uint32_t run = 0; run < RUNS_OF_MOVED_CASES; run++) {
        const uint32_t row_count = moved_row_counts[run];
        test_linear_context("card %" PRIu32 ": %" PRIu32 " rows from case %" PRIu32, card->card_index, row_count,
                            moved_by[run]);
        if (!cases_move(card, description, moved_by[run])) {
            return;
        }
	test_linear_expect(
		card_case_run(card, &ranges, row_count, kernels,
			      ENGRAM_ROWS_VALUES_LAUNCH) &&
			moved_rows_hold(card_values, first_values, NULL,
					CASE_VALUES, row_count, moved_by[run]),
		"each case's values are those of its row of 8, and the rows behind are left untouched");
	test_linear_expect(
		card_case_run(card, &ranges, row_count, kernels,
			      ENGRAM_ROWS_GATE_LAUNCH) &&
			moved_rows_hold(card_streams, first_streams,
					host_streams, STREAM_VALUES, row_count,
					moved_by[run]),
		"each case's streams are those of its row of 8, and the rows behind are left untouched");
    }
}

/* ---- the replay on the reference's own inputs ---- */

/* What the replay reads from: the capture, the pack, the n-gram constants, and what the host takes from them once. */
struct replay {
    struct test_capture *capture;
    struct storage_reader *reader;
    struct pack_record *pack;
    struct engram_record *engram;
    uint32_t *token_ids;                        /* [position] */
    uint8_t *table_rows;                        /* [layer][position][TABLE_ROWS][ROW_BYTES], from the pack */
    unsigned char *block;                       /* one pack block, aligned for the reader */
    unsigned char *weight_blocks;               /* the blocks q_weight or k_weight lie in, aligned for the reader */
    uint32_t position_count;
    uint32_t block_alignment_bytes;
    enum break_kind break_kind;
    bool held[CHECK_COUNT];                     /* on every card and layer so far */
};

static void check_report(struct replay *replay, enum replay_check check, bool held, const char *figures)
{
    replay->held[check] = replay->held[check] && held;
    (void)printf("REPLAY %s: %s: %s\n", check_names[check], figures, held ? "held" : "did not hold");
}

/* A tensor of the pack by name, held to its type and sizes; the bytes from the block it begins in to the end of the
 * one it ends in are read into destination (capacity bytes, aligned for the reader); lead_out is where it begins. */
static bool pack_tensor_read(const struct replay *replay, const char *name,
			     const struct pack_tensor *wanted,
			     unsigned char *destination, uint64_t capacity,
			     uint64_t *lead_out)
{
	const uint32_t tensor_index =
		pack_find_tensor(replay->pack, name, (uint32_t)strlen(name));
	if (tensor_index == PACK_NOT_FOUND) {
		test_linear_fail("the pack holds no tensor %s", name);
		return false;
	}
	const struct pack_tensor *const tensor =
		&pack_tensors_of(replay->pack)[tensor_index];
	const struct pack_extent *const extent =
		&pack_extents_of(replay->pack)[tensor->extent_index];
	const uint64_t alignment = replay->block_alignment_bytes;
	const uint64_t file_byte_offset =
		extent->byte_offset + tensor->byte_offset_in_extent;
	const uint64_t lead = file_byte_offset % alignment;
	const uint64_t read_byte_count =
		(lead + tensor->byte_count + alignment - 1) / alignment *
		alignment;
	if (tensor->element_type != wanted->element_type ||
	    tensor->weight_format != wanted->weight_format ||
	    tensor->dimension_count != MATRIX_DIMENSIONS ||
	    tensor->dimensions[0] != wanted->dimensions[0] ||
	    tensor->dimensions[1] != wanted->dimensions[1] ||
	    tensor->byte_count != wanted->byte_count ||
	    read_byte_count > capacity) {
		test_linear_fail("%s is, in the pack, of type %" PRIu32
				 ", format %" PRIu32 ", %" PRIu32 " by %" PRIu32
				 " in %" PRIu64
				 " bytes: not what the n-gram layer reads",
				 name, tensor->element_type,
				 tensor->weight_format, tensor->dimensions[0],
				 tensor->dimensions[1], tensor->byte_count);
		return false;
	}
    *lead_out = lead;
    return test_linear_expect_status(
	    storage_direct.reader_read(replay->reader, 0,
				       file_byte_offset - lead, destination,
				       read_byte_count),
	    STATUS_OK, "a tensor is read from the pack");
}

/* The row of table `table` and that id, as the pack holds it, into row_out. */
static bool table_row_read(const struct replay *replay, uint32_t table, uint64_t row_id, uint8_t *row_out)
{
	const struct engram_description *const description =
		engram_description_of(replay->engram);
	const struct pack_unit_record_2 *const unit = &pack_units_of(
		replay->pack)[description->table_unit_index[table]];
	const struct pack_extent *const extent =
		&pack_extents_of(replay->pack)[unit->first_extent_index];
	const uint64_t block_offset =
		row_id / TABLE_ROWS_PER_BLOCK * PACK_BLOCK;
	if (row_id >= description->row_count[table] ||
	    block_offset >= extent->byte_count) {
		test_linear_fail("row %" PRIu64 " lies outside table %" PRIu32,
				 row_id, table);
		return false;
	}
	if (!test_linear_expect_status(
		    storage_direct.reader_read(replay->reader, 0,
					       extent->byte_offset +
						       block_offset,
					       replay->block, PACK_BLOCK),
		    STATUS_OK, "a block of a table is read from the pack")) {
		return false;
	}
    memcpy(row_out, &replay->block[row_id % TABLE_ROWS_PER_BLOCK * ROW_BYTES], ROW_BYTES);
    return true;
}

/* The row ids of every position against the capture's, and the rows they name read once from the pack (with
 * --break rows, the row behind each). */
static bool replay_rows_gather(struct replay *replay)
{
	const struct engram_description *const description =
		engram_description_of(replay->engram);
	uint64_t row_ids[ENGRAM_LAYER_COUNT * TABLE_ROWS];
	int64_t captured[TABLE_ROWS];
	uint64_t differing = 0;
	char name[NAME_CAPACITY];
	if (description->table_count != ENGRAM_LAYER_COUNT ||
	    description->rows_per_token != TABLE_ROWS) {
		test_linear_fail("the pack has %" PRIu32
				 " n-gram tables of %" PRIu32
				 " rows a token: the model has %u of %u",
				 description->table_count,
				 description->rows_per_token,
				 ENGRAM_LAYER_COUNT, TABLE_ROWS);
		return false;
	}
    for (uint32_t position = 0; position < replay->position_count; position++) {
	    if (!test_linear_expect_status(
			engram_row_ids_entry(replay->engram, replay->token_ids,
					     NULL, position, row_ids),
			STATUS_OK, "the row ids of a token")) {
		    return false;
	    }
	for (uint32_t table = 0; table < ENGRAM_LAYER_COUNT; table++) {
            const uint64_t dimensions[IDS_DIMENSIONS] = {1, replay->position_count, TABLE_ROWS};
            (void)snprintf(name, sizeof(name), "layer_%" PRIu32 ".ngram.row_ids", engram_layers[table]);
	    if (!test_linear_expect_status(
			test_capture_read(
				replay->capture, name, TEST_CAPTURE_INT64,
				dimensions, IDS_DIMENSIONS,
				(uint64_t)position * TABLE_ROWS, TABLE_ROWS,
				captured, sizeof(captured)),
			STATUS_OK, "the captured row ids are read")) {
		    return false;
	    }
	    for (uint32_t place = 0; place < TABLE_ROWS; place++) {
                const uint64_t row_id = row_ids[table * TABLE_ROWS + place];
                const uint64_t read_id = replay->break_kind != BREAK_ROWS ? row_id
                                         : row_id + 1 < description->row_count[table] ? row_id + 1
                                                                                      : row_id - 1;
                const size_t row_place = ((size_t)table * replay->position_count + position) * TABLE_ROWS + place;
                differing += captured[place] < 0 || (uint64_t)captured[place] != row_id ? 1u : 0u;
                if (!table_row_read(replay, table, read_id, &replay->table_rows[row_place * ROW_BYTES])) {
                    return false;
                }
            }
        }
    }
    char figures[NAME_CAPACITY];
    (void)snprintf(figures, sizeof(figures), "%" PRIu64 " of %" PRIu64 " differ", differing,
                   (uint64_t)replay->position_count * ENGRAM_LAYER_COUNT * TABLE_ROWS);
    check_report(replay, CHECK_ROW_IDS, differing == 0, figures);
    return true;
}

static bool replay_open(struct replay *replay, const char *capture_folder, const char *const *copy_paths,
                        uint32_t copy_count)
{
	struct storage_description storage = { 0, 0, 0 };
	int64_t token_id = 0;
	if (!test_linear_expect_status(test_capture_open(capture_folder,
							 &replay->capture),
				       STATUS_OK, "the capture opens")) {
		return false;
	}
    const struct test_capture_tensor *const tokens = test_capture_find(replay->capture, "input.token_ids");
    if (tokens == NULL || tokens->type != TEST_CAPTURE_INT64 || tokens->dimension_count != TOKENS_DIMENSIONS ||
        tokens->dimensions[0] != 1 || tokens->dimensions[POSITION_AXIS] == 0 ||
        tokens->dimensions[POSITION_AXIS] > MOST_POSITIONS) {
        test_linear_fail("the capture holds no input.token_ids of int64 [1, 1 to %u]", MOST_POSITIONS);
        return false;
    }
    replay->position_count = (uint32_t)tokens->dimensions[POSITION_AXIS];
    replay->token_ids = calloc(replay->position_count, sizeof(*replay->token_ids));
    replay->table_rows = calloc((size_t)ENGRAM_LAYER_COUNT * replay->position_count * TABLE_ROWS, ROW_BYTES);
    replay->block = aligned_alloc(PACK_BLOCK, PACK_BLOCK);
    replay->weight_blocks = aligned_alloc(PACK_BLOCK, WEIGHT_READ_CAPACITY);
    if (replay->token_ids == NULL || replay->table_rows == NULL || replay->block == NULL ||
        replay->weight_blocks == NULL) {
        test_linear_fail("the host's copies of the replay are not allocated");
        return false;
    }
    result_status status =
	    storage_direct.reader_open(copy_paths, copy_count, &replay->reader);
    status = status == STATUS_OK ?
		     storage_direct.reader_describe(replay->reader, &storage) :
		     status;
    status = status == STATUS_OK ?
		     pack_open_entry(&storage_direct, replay->reader,
				     &replay->pack) :
		     status;
    status = status == STATUS_OK ? engram_open(&storage_direct, replay->reader,
					       replay->pack, &replay->engram) :
				   status;
    replay->block_alignment_bytes = storage.block_alignment_bytes;
    if (!test_linear_expect_status(status, STATUS_OK,
				   "the pack and its n-gram constants open") ||
	!test_linear_expect(
		storage.block_alignment_bytes != 0 &&
			PACK_BLOCK % storage.block_alignment_bytes == 0 &&
			BACKEND_STAGING_ALIGNMENT_BYTES %
					storage.block_alignment_bytes ==
				0,
		"the reader's blocks divide a pack block and a staging page")) {
	    return false;
    }
    const uint32_t vocabulary_size =
	    engram_description_of(replay->engram)->vocabulary_size;
    for (uint32_t position = 0; position < replay->position_count; position++) {
        const uint64_t dimensions[TOKENS_DIMENSIONS] = {1, replay->position_count};
	if (!test_linear_expect_status(
		    test_capture_read(replay->capture, "input.token_ids",
				      TEST_CAPTURE_INT64, dimensions,
				      TOKENS_DIMENSIONS, position, 1, &token_id,
				      sizeof(token_id)),
		    STATUS_OK, "a token id is read") ||
	    !test_linear_expect(token_id >= 0 &&
					token_id < (int64_t)vocabulary_size,
				"a token id is in the vocabulary")) {
		return false;
	}
	replay->token_ids[position] = (uint32_t)token_id;
    }
    for (uint32_t check = 0; check < CHECK_COUNT; check++) {
        replay->held[check] = true;
    }
    return replay_rows_gather(replay);
}

static void replay_close(struct replay *replay)
{
	engram_close(replay->engram);
	pack_close(replay->pack);
	storage_direct.reader_close(replay->reader);
	test_capture_close(replay->capture);
	free(replay->weight_blocks);
	free(replay->block);
	free(replay->table_rows);
	free(replay->token_ids);
}

/* bfloat16 values of a capture record of [1, positions, ...] from position `first` on, count positions. */
static bool capture_rows_read(const struct replay *replay, const char *name, const uint64_t *dimensions,
                              uint32_t dimension_count, uint64_t values_per_position, uint32_t first, uint32_t count,
                              uint16_t *destination)
{
	return test_linear_expect_status(
		test_capture_read(replay->capture, name, TEST_CAPTURE_BFLOAT16,
				  dimensions, dimension_count,
				  first * values_per_position,
				  count * values_per_position, destination,
				  count * values_per_position * VALUE_BYTES),
		STATUS_OK, name);
}

/* engram.wkv, q_weight and k_weight of a layer from the pack into the case's image; the description of the product,
 * all but its slice count, into product_out. */
static bool
layer_weights_load(struct test_linear_card *card, const struct replay *replay,
		   uint32_t layer, const struct card_ranges *ranges,
		   struct linear_trunk_quantized_description *product_out)
{
	static const struct pack_tensor matrix_expected = {
		.byte_count = MATRIX_BYTES,
		.element_type = PACK_ELEMENT_UINT8,
		.dimensions = { PROJECTION, VALUES_WIDTH },
		.weight_format = WEIGHT_FORMAT_E4M3_SCALED
	};
	static const struct pack_tensor weight_expected = {
		.byte_count = (uint64_t)STREAM_VALUES * VALUE_BYTES,
		.element_type = PACK_ELEMENT_BFLOAT16,
		.dimensions = { STREAMS, HIDDEN }
	};
	static const char *const weight_names[] = { "q_weight", "k_weight" };
	const struct linear_range destinations[] = {
		ranges->description.query_weight, ranges->description.key_weight
	};
	char name[NAME_CAPACITY];
	uint64_t lead = 0;
	const struct linear_range read_range =
		test_linear_card_weights(card, WEIGHTS_CAPACITY);
	unsigned char *const destination =
		test_linear_card_writable(card, read_range);
	(void)snprintf(name, sizeof(name),
		       "layers.%" PRIu32 ".engram.wkv.weight", layer);
	if (destination == NULL ||
	    !pack_tensor_read(replay, name, &matrix_expected, destination,
			      WEIGHTS_CAPACITY, &lead)) {
		return false;
	}
    for (uint32_t weight = 0; weight < COUNT_OF(weight_names); weight++) {
        uint64_t weight_lead = 0;
        (void)snprintf(name, sizeof(name), "layers.%" PRIu32 ".engram.%s", layer, weight_names[weight]);
        if (!pack_tensor_read(replay, name, &weight_expected, replay->weight_blocks, WEIGHT_READ_CAPACITY,
                              &weight_lead) ||
            !test_linear_card_put(card, destinations[weight], 0, &replay->weight_blocks[weight_lead],
                                  weight_expected.byte_count)) {
            return false;
        }
    }
    *product_out = (struct linear_trunk_quantized_description){
	    .matrices = { { .stored = { read_range.byte_offset + lead,
					MATRIX_BYTES },
			    .shape = { PROJECTION, VALUES_WIDTH,
				       E4M3_BITS_PER_WEIGHT, 0 },
			    .weight_format = WEIGHT_FORMAT_E4M3_SCALED } },
	    .codes = ranges->codes,
	    .exponents = ranges->exponents,
	    .operands = ranges->operands,
	    .outputs = { ranges->description.projection },
	    .matrix_count = 1,
	    .row_count = ROWS
    };
    return true;
}

/* The slice counts engram.wkv has on the card under the two profiles and the two orders, each once, into counts_out
 * (KNOB_COUNT); how many, 0 behind a failure line. */
static uint32_t slice_counts_of(const struct test_linear_card *card,
				const struct matrix_shape_record *shape,
				uint32_t *counts_out)
{
	static const enum linear_profile_code profiles[] = {
		LINEAR_PROFILE_PRECISION, LINEAR_PROFILE_SPEED
	};
	static const enum linear_order_code orders[] = {
		LINEAR_ORDER_FIXED, LINEAR_ORDER_REFERENCE
	};
	_Static_assert(COUNT_OF(profiles) * COUNT_OF(orders) == KNOB_COUNT,
		       "every profile with every order");
	uint32_t count = 0;
	for (uint32_t knobs = 0; knobs < KNOB_COUNT; knobs++) {
		uint32_t slice_count = 0;
		if (!test_linear_expect_status(
			    linear_default_slice_count(
				    profiles[knobs / COUNT_OF(orders)],
				    orders[knobs % COUNT_OF(orders)],
				    LINEAR_FAMILY_TRUNK_QUANTIZED, shape,
				    card->arithmetic_class, &slice_count),
			    STATUS_OK, "engram.wkv has its slice count")) {
			return 0;
		}
		bool is_new = true;
		for (uint32_t seen = 0; seen < count; seen++) {
			is_new = is_new && counts_out[seen] != slice_count;
		}
		counts_out[count] = slice_count;
		count += is_new ? 1u : 0u;
	}
    return count;
}

/* The tallies of a layer on a card in one slice count against what the checks ask, reported. */
static void layer_report(struct replay *replay, const struct tally *tallies, uint32_t card_index, uint32_t layer,
                         uint32_t slice_count)
{
    for (uint32_t check = CHECK_VALUES; check < CHECK_COUNT; check++) {
        const struct tally *const tally = &tallies[check];
        const bool is_gate = check == CHECK_CHAIN || check == CHECK_GATE_ALONE;
        const uint64_t passing = is_gate ? tally->within_one : tally->equal;
        const bool held = tally->count != 0 && (check == CHECK_VALUES ? passing == tally->count
                                                                      : test_linear_share_holds(passing, tally->count));
        char power[NAME_CAPACITY] = "";
        char figures[FIGURES_CAPACITY];
        if (is_gate) {
            (void)snprintf(power, sizeof(power), "; the n-gram add moved %.4f%% of them more than one place",
                           test_linear_percent(tally->moved, tally->count));
        }
        (void)snprintf(figures, sizeof(figures), "card %" PRIu32 ", layer %" PRIu32 ", in %" PRIu32 " slices: %.4f%% "
                       "of %" PRIu64 " bit-equal, %.4f%% within one place, at most %" PRIu32 " places apart%s",
                       card_index, layer, slice_count, test_linear_percent(tally->equal, tally->count), tally->count,
                       test_linear_percent(tally->within_one, tally->count), tally->largest_distance, power);
        check_report(replay, (enum replay_check)check, held, figures);
    }
}

/* The projection of row `row` of a batch at a capture point, read, tallied against the card's and put into the image
 * for the gate alone (with --break gate, SHORT_GATE_SHARE of each value). */
static bool capture_point_take(struct test_linear_card *card, const struct replay *replay, const char *point_name,
                               uint32_t row, struct tally *projection_tally)
{
    const uint64_t projection_dimensions[1] = {PROJECTION};
    if (!test_linear_expect_status(
		test_capture_read(replay->capture, point_name,
				  TEST_CAPTURE_BFLOAT16, projection_dimensions,
				  1, 0, PROJECTION, captured_projection,
				  sizeof(captured_projection)),
		STATUS_OK, "a captured projection is read")) {
	    return false;
    }
    for (uint32_t place = 0; place < PROJECTION; place++) {
        tally_add(projection_tally, card_projection[(size_t)row * PROJECTION + place], captured_projection[place]);
    }
    for (uint32_t place = STREAM_VALUES; replay->break_kind == BREAK_GATE && place < PROJECTION; place++) {
        captured_projection[place] =
            test_linear_bfloat16_nearest(SHORT_GATE_SHARE * test_linear_bfloat16_value(captured_projection[place]));
    }
    return test_linear_card_put(card, ranges_of_replay.description.projection,
                                (uint64_t)row * PROJECTION * VALUE_BYTES, captured_projection,
                                sizeof(captured_projection));
}

/* Every position of a layer through the chain, 8 a launch, and the gate alone at the capture points; reported. */
static void replay_positions(struct test_linear_card *card,
			     struct replay *replay, uint32_t table,
			     const struct kernel_launch *chain,
			     uint32_t slice_count)
{
    const uint32_t layer = engram_layers[table];
    const uint64_t stream_dimensions[STREAM_DIMENSIONS] = {1, replay->position_count, STREAMS, HIDDEN};
    const uint64_t rows_dimensions[ROWS_DIMENSIONS] = {1, replay->position_count, TABLE_ROWS, ROW_VALUES};
    const struct engram_rows_description *const described =
	    &ranges_of_replay.description;
    const uint32_t chain_count = replay->break_kind == BREAK_GATE ? CHAIN_GATE : CHAIN_LAUNCH_COUNT;
    struct tally tallies[CHECK_COUNT];
    char before[NAME_CAPACITY];
    char rows_name[NAME_CAPACITY];
    char output_name[NAME_CAPACITY];
    char point_name[NAME_CAPACITY];
    memset(tallies, 0, sizeof(tallies));
    (void)snprintf(before, sizeof(before), "layer_%" PRIu32 ".block.output", layer - 1);
    (void)snprintf(rows_name, sizeof(rows_name), "layer_%" PRIu32 ".ngram.rows", layer);
    (void)snprintf(output_name, sizeof(output_name), "layer_%" PRIu32 ".ngram.output", layer);
    for (uint32_t first = 0; first < replay->position_count; first += ROWS) {
        const uint32_t count = replay->position_count - first < ROWS ? replay->position_count - first : ROWS;
        const uint8_t *const rows =
            &replay->table_rows[((size_t)table * replay->position_count + first) * TABLE_ROWS * ROW_BYTES];
        bool is_point[ROWS] = {false};
        uint32_t point_count = 0;
        if (!capture_rows_read(replay, before, stream_dimensions, STREAM_DIMENSIONS, STREAM_VALUES, first, count,
                               host_streams) ||
            !test_linear_card_put(card, described->rows, 0, rows, (uint64_t)count * TABLE_ROWS * ROW_BYTES) ||
            !test_linear_card_put(card, described->streams, 0, host_streams,
                                  (uint64_t)count * STREAM_VALUES * VALUE_BYTES) ||
            !test_linear_card_upload(card, described->rows, 0, described->rows.byte_count) ||
            !test_linear_card_upload(card, described->streams, 0, described->streams.byte_count) ||
            !test_linear_card_run(card, chain, chain_count) || !test_linear_card_fetch(card) ||
            !test_linear_card_get(card, described->values, 0, card_values, described->values.byte_count) ||
            !test_linear_card_get(card, described->projection, 0, card_projection, sizeof(card_projection)) ||
            !test_linear_card_get(card, described->streams, 0, card_streams, sizeof(card_streams)) ||
            !capture_rows_read(replay, rows_name, rows_dimensions, ROWS_DIMENSIONS, VALUES_WIDTH, first, count,
                               from_capture)) {
            return;
        }
        for (size_t place = 0; place < (size_t)count * VALUES_WIDTH; place++) {
            tally_add(&tallies[CHECK_VALUES], card_values[place], from_capture[place]);
        }
        if (!capture_rows_read(replay, output_name, stream_dimensions, STREAM_DIMENSIONS, STREAM_VALUES, first, count,
                               from_capture)) {
            return;
        }
        for (size_t place = 0; place < (size_t)count * STREAM_VALUES; place++) {
            tally_add(&tallies[CHECK_CHAIN], card_streams[place], from_capture[place]);
            power_add(&tallies[CHECK_CHAIN], host_streams[place], from_capture[place]);
        }
        /* the capture points of the batch: the projection, then the gate alone from the captured one */
        for (uint32_t row = 0; row < count; row++) {
            (void)snprintf(point_name, sizeof(point_name), "layer_%" PRIu32 ".position_%" PRIu32 ".ngram.projection",
                           layer, first + row);
            is_point[row] = test_capture_find(replay->capture, point_name) != NULL;
            if (is_point[row] && !capture_point_take(card, replay, point_name, row, &tallies[CHECK_PROJECTION])) {
                return;
            }
            point_count += is_point[row] ? 1u : 0u;
        }
        if (point_count != 0 &&
            (!test_linear_card_upload(card, described->projection, 0, described->projection.byte_count) ||
             !test_linear_card_upload(card, described->streams, 0, described->streams.byte_count) ||
             !test_linear_card_run(card, &chain[CHAIN_GATE], 1) || !test_linear_card_fetch(card) ||
             !test_linear_card_get(card, described->streams, 0, card_streams, sizeof(card_streams)))) {
            return;
        }
        for (size_t place = 0; point_count != 0 && place < (size_t)count * STREAM_VALUES; place++) {
            if (is_point[place / STREAM_VALUES]) {
                tally_add(&tallies[CHECK_GATE_ALONE], card_streams[place], from_capture[place]);
                power_add(&tallies[CHECK_GATE_ALONE], host_streams[place], from_capture[place]);
            }
        }
    }
    layer_report(replay, tallies, card->card_index, layer, slice_count);
}

/* One layer on one card: its weights loaded once, then every position replayed in each slice count. */
static void replay_layer(struct test_linear_card *card,
			 const struct engram_rows_kernels *kernels,
			 struct replay *replay, uint32_t table)
{
    const uint32_t layer = engram_layers[table];
    const struct engram_rows_description *const described =
	    &ranges_of_replay.description;
    struct linear_trunk_quantized_description product;
    struct kernel_launch own[ENGRAM_ROWS_LAUNCH_COUNT];
    struct kernel_launch chain[CHAIN_LAUNCH_COUNT];
    uint32_t slice_counts[KNOB_COUNT] = {0};
    test_linear_context("card %" PRIu32 ": layer %" PRIu32, card->card_index, layer);
    card_ranges_set(card, &ranges_of_replay, TABLE_ROWS);
    const struct linear_quantize_description quantize = {
	    described->values,
	    ranges_of_replay.codes,
	    ranges_of_replay.exponents,
	    ranges_of_replay.operands,
	    VALUES_WIDTH,
	    ROWS
    };
    if (!test_linear_card_has_room(card) ||
	!layer_weights_load(card, replay, layer, &ranges_of_replay, &product) ||
	!test_linear_card_load(card) ||
	!launches_prepare(&ranges_of_replay, ROWS, card->arena_byte_count,
			  kernels, own) ||
	!test_linear_expect_status(
		linear_quantize_prepare(&quantize, card->arena_byte_count,
					&card->kernels, &chain[CHAIN_QUANTIZE]),
		STATUS_OK, "the quantization of the values is prepared")) {
	    return;
    }
    chain[CHAIN_VALUES] = own[ENGRAM_ROWS_VALUES_LAUNCH];
    chain[CHAIN_GATE] = own[ENGRAM_ROWS_GATE_LAUNCH];
    const uint32_t variant_count = slice_counts_of(card, &product.matrices[0].shape, slice_counts);
    for (uint32_t variant = 0; variant < variant_count; variant++) {
        product.matrices[0].slice_count = slice_counts[variant];
	if (!test_linear_expect_status(
		    linear_trunk_quantized_prepare(
			    &product, card->arena_byte_count, &card->kernels,
			    &chain[CHAIN_PROJECTION]),
		    STATUS_OK, "the product with engram.wkv is prepared")) {
		return;
	}
	replay_positions(card, replay, table, chain, slice_counts[variant]);
    }
}

/* ---- the cards ---- */

static void card_run(uint32_t card_index, struct replay *replay)
{
    struct test_linear_card card;
    struct engram_rows_kernels kernels;
    if (!test_linear_card_open(&card, card_index, WORK_CAPACITY, replay != NULL ? WEIGHTS_CAPACITY : 0)) {
        return;
    }
    if (test_linear_expect_status(
		engram_rows_kernels_prepare(&backend_cuda_value, card.card,
					    &kernels),
		STATUS_OK, "the kernels of the n-gram rows are prepared")) {
	    synthetic_cases(&card, &kernels);
	    for (uint32_t table = 0;
		 replay != NULL && table < ENGRAM_LAYER_COUNT; table++) {
		    replay_layer(&card, &kernels, replay, table);
	    }
	    (void)printf("card %" PRIu32
			 ": digest of everything fetched 0x%016" PRIx64 "\n",
			 card_index, card.digest);
    }
    test_linear_card_close(&card);
}

/* With a break, whether exactly the checks it aims at reported; without one, whether every check held. */
static bool replay_verdict(const struct replay *replay)
{
    bool verdict = true;
    for (uint32_t check = 0; check < CHECK_COUNT; check++) {
        const bool is_aimed_at =
            (replay->break_kind == BREAK_ROWS &&
             (check == CHECK_VALUES || check == CHECK_PROJECTION || check == CHECK_CHAIN)) ||
            (replay->break_kind == BREAK_GATE && (check == CHECK_CHAIN || check == CHECK_GATE_ALONE));
        const bool as_wanted = replay->held[check] != is_aimed_at;
        (void)printf("CHECK %s: %s\n", check_names[check],
                     as_wanted ? (is_aimed_at ? "reported, as the break asks" : "held") : "FAILED");
        verdict = verdict && as_wanted;
    }
    return verdict;
}

int main(int argument_count, char **arguments)
{
    static struct replay replay;
    uint32_t card_count = 0;
    int first_argument = 1;
    if (argument_count > first_argument && strcmp(arguments[first_argument], "--break") == 0) {
        const char *const kind = argument_count > BREAK_KIND_ARGUMENT ? arguments[BREAK_KIND_ARGUMENT] : "";
        replay.break_kind = strcmp(kind, "rows") == 0 ? BREAK_ROWS : strcmp(kind, "gate") == 0 ? BREAK_GATE
                                                                                               : BREAK_NONE;
        first_argument = FIRST_ARGUMENT_AFTER_BREAK;
    }
    const bool is_replayed = argument_count > first_argument;
    const int copy_count = argument_count - first_argument - 1;
    if ((first_argument != 1 &&
	 (replay.break_kind == BREAK_NONE || !is_replayed)) ||
	(is_replayed && copy_count < 1) ||
	copy_count > (int)STORAGE_MAXIMUM_COPIES) {
	    (void)fprintf(
		    stderr,
		    "usage: engram_rows_card_test [[--break rows|gate] <capture folder> <pack copy> "
		    "[<pack copy> ...]]\n");
	    return TEST_WRONG_USAGE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    const bool is_open = !is_replayed ||
                         replay_open(&replay, arguments[first_argument],
                                     (const char *const *)&arguments[first_argument + 1], (uint32_t)copy_count);
    if (is_open &&
	test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    card_run(card_index, is_replayed ? &replay : NULL);
	    }
    }
    const bool replay_holds = !is_replayed || (is_open && replay_verdict(&replay));
    if (is_replayed) {
        replay_close(&replay);
    }
    if (test_linear_failures() != 0 || !replay_holds) {
        (void)fprintf(stderr, "engram_rows_card_test FAILED: %d failure lines%s\n", test_linear_failures(),
                      replay_holds ? "" : ", and the replay's checks are not as they must be");
        return TEST_FAILED;
    }
    (void)printf("engram_rows_card_test passed on %" PRIu32 " cards%s\n", card_count,
                 is_replayed ? "" : " (the replay compared nothing: it needs a capture folder and a pack)");
    return TEST_PASSED;
}
