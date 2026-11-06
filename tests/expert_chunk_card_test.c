/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_chunk_card_test: a wide step's chunk of routed experts on the real cards, composed as the wide walk runs it:
 * the router's wide select (router_wide_prepare: the select, the count, the place and the jobs) and the
 * routed experts of the chunk's rows over the wide select's job table (linear_routed_experts_prepare),
 * against what the engine runs today: per eight rows the thin select (router_select_prepare) and the routed
 * experts of those rows.
 *
 *     expert_chunk_card_test [--time]
 *
 * On every card, experts of 256 to 192 and back (4 and 2 slices), each of the 384 its own three matrices, and chunks
 * of 8, 61 (a last thin step of 5 rows), 1,024 and 4,096 rows, in turn of the fixed and the reference order; plain
 * scores; the first rows of plain values, the others of the edges, quantized once on the card for both chains:
 *   - the wide chain's affinities, hidden values, hidden codes, exponents and operands, pair outputs and row sums are
 *     the thin steps', byte for byte: the thin steps write arrays of the chunk's shape, each step its rows' part.
 *     What the router leaves for the experts (picks, counts, pairs, runs, tiles, job tables) and what both chains
 *     write start as rubbish, of another byte for each chain: a byte neither writes shows as a difference.
 *   - Shown failing once: of the chunk of 61 rows, the route weight of the first pair of the first wide job is changed
 *     on the card (half as large again) and the wide chain's experts run again: that pair's hidden values and outputs
 *     and its row's sums must then differ from the thin steps', and nothing else of any array.
 *   - The ranges only read come back untouched, and so do the guards and the matrices.
 * --time prints, of the chunk of 1,024 rows, the time of the wide select, of the wide chain's experts and of the thin
 * steps (128 of five launches), TIMED_RUNS runs back to back: indicative at this shape; linear_probe times the
 * model's.
 * It FAILS on any difference and when a card has no room.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/router.h"
#include "anemone/router_records.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "anemone/wide_linear_records.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 1800
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define EXPERTS MODEL_EXPERTS
#define PICKS MODEL_EXPERTS_PER_ROW
#define MATRICES MODEL_EXPERT_MATRICES
#define THIN_ROWS LINEAR_MAXIMUM_ROWS
#define MOST_ROWS MODEL_WIDE_MOST_ROWS
#define SORT_ROWS ROUTER_SORT_ROWS
#define UNEVEN_ROWS 61u                         /* seven thin steps and one of 5 rows */
#define TIMED_ROWS 1024u
#define HIDDEN_SIZE 256u                        /* linear_expert_card_test's uneven shape, 6 experts to a row */
#define INNER_SIZE 192u
#define GATE_UP_SLICES 4u
#define DOWN_SLICES 2u
#define E2M1 WEIGHT_FORMAT_E2M1_SCALED
#define E2M1_BITS_PER_WEIGHT 4u
#define GATE_UP_SCALE_CODE_BASE 121u            /* so that gate and up reach beyond the activation limit */
#define DOWN_SCALE_CODE_BASE 118u
#define SCALE_CODE_SPREAD 3u
#define ACTIVATION_LIMIT 10.0f                  /* the model's */
#define VALUES_PER_BLOCK TEST_LINEAR_VALUES_PER_BLOCK
#define VALUE_BYTES ((uint64_t)LINEAR_BFLOAT16_BYTES)
#define SUM_BYTES ((uint64_t)LINEAR_SUM_BYTES)
#define PLAIN_ROWS 4u                           /* of a chunk: the first rows hold plain values, the others the edges */
#define WORK_CAPACITY (192ull << 20u)           /* the four chunks take about 140 MiB */
#define WEIGHTS_CAPACITY (40ull << 20u)         /* 1,152 matrices of 26 KiB */
#define FILL_PIECE_BYTES 65536u
#define WIDE_RUBBISH_BYTE 0xA7
#define THIN_RUBBISH_BYTE 0x5C
#define STEP_SEQUENCE 1u
#define CHUNK_STREAM 0x11000000ull              /* the streams of the hashes, far from those of the other tests */
#define MATRIX_STREAM 0x100ull
#define SCORE_STREAM 0x10000ull
#define ROW_STREAM 0x20000ull
#define HALVES 2u
#define HALF 0.5
#define HASH_FRACTION_SHIFT 40u                 /* the top 24 bits of a hash: a fraction of 2^24 */
#define HASH_FRACTION_UNIT 0x1p-24
#define PLAIN_SPAN 16.0                         /* scores from -8 to 8 */
#define BIAS_SPAN 1.0
#define CHANGED_WEIGHT_FACTOR 1.5f
#define THIN_STEP_LAUNCHES (1u + LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT)
#define WIDE_SELECT_FIRST 1u                    /* of a chunk's launches: the quantization, then the wide chain */
#define WIDE_EXPERTS_FIRST (WIDE_SELECT_FIRST + ROUTER_WIDE_LAUNCH_COUNT)
#define THIN_FIRST (WIDE_EXPERTS_FIRST + LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT)
#define MOST_CHUNK_LAUNCHES (THIN_FIRST + MOST_ROWS / THIN_ROWS * THIN_STEP_LAUNCHES)
#define TIMED_RUNS 10u
#define TIMED_LAUNCHES (TIMED_ROWS / THIN_ROWS * THIN_STEP_LAUNCHES)
#define MOST_ARGUMENTS 2                        /* the program's name and --time */
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum chunk_index { CHUNK_ONE_STEP, CHUNK_UNEVEN, CHUNK_TIMED, CHUNK_MOST, CHUNK_COUNT };
static const uint32_t chunk_rows[CHUNK_COUNT] = {
    [CHUNK_ONE_STEP] = THIN_ROWS, [CHUNK_UNEVEN] = UNEVEN_ROWS, [CHUNK_TIMED] = TIMED_ROWS, [CHUNK_MOST] = MOST_ROWS};

enum chain { CHAIN_WIDE, CHAIN_THIN, CHAIN_COUNT };
static const unsigned char rubbish_of[CHAIN_COUNT] = {WIDE_RUBBISH_BYTE, THIN_RUBBISH_BYTE};

/* What each chain writes, in arrays of the chunk's rows: a row's own, or its six pairs'. */
enum written_array {
    WRITTEN_AFFINITIES,
    WRITTEN_HIDDEN,
    WRITTEN_HIDDEN_CODES,
    WRITTEN_HIDDEN_EXPONENTS,
    WRITTEN_HIDDEN_OPERANDS,
    WRITTEN_PAIR_OUTPUT,
    WRITTEN_ROW_OUTPUT,
    WRITTEN_COUNT
};
static const uint64_t row_bytes_of[WRITTEN_COUNT] = {
	[WRITTEN_AFFINITIES] = MODEL_ROUTER_ROW_BYTES,
	[WRITTEN_HIDDEN] = VALUE_BYTES * PICKS * INNER_SIZE,
	[WRITTEN_HIDDEN_CODES] = (uint64_t)PICKS * INNER_SIZE,
	[WRITTEN_HIDDEN_EXPONENTS] =
		(uint64_t)PICKS * INNER_SIZE / VALUES_PER_BLOCK,
	[WRITTEN_HIDDEN_OPERANDS] = VALUE_BYTES * PICKS * INNER_SIZE,
	[WRITTEN_PAIR_OUTPUT] = VALUE_BYTES * PICKS * HIDDEN_SIZE,
	[WRITTEN_ROW_OUTPUT] = SUM_BYTES * HIDDEN_SIZE
};
static const char *const written_names[WRITTEN_COUNT] = {
    "the affinities",    "the hidden values", "the hidden codes", "the hidden exponents", "the hidden operands",
    "the pair outputs",  "the row sums"};

/* One chunk on one card: its ranges, and where its launches lie in launches. */
struct chunk {
    uint32_t row_count;
    bool is_reference_order;
    struct linear_range wide_step, last_step, scores, values, codes, exponents,
	    operands;
    struct linear_range picks, counts, pairs, runs, tiles, jobs;
    struct linear_range written[CHAIN_COUNT][WRITTEN_COUNT];
    uint32_t first_launch;                      /* the quantization, the wide chain, the thin steps */
    uint32_t launch_count;
};

static const struct matrix_shape_record gate_up_shape = {
	INNER_SIZE, HIDDEN_SIZE, E2M1_BITS_PER_WEIGHT, 0
};
static const struct matrix_shape_record down_shape = { HIDDEN_SIZE, INNER_SIZE,
						       E2M1_BITS_PER_WEIGHT,
						       0 };
static struct router_kernels_record router_kernels;
static uint64_t expert_offsets[EXPERTS * MATRICES];
static float bias_given[EXPERTS];
static struct linear_range bias_range, offsets_range, thin_step_range;
static uint32_t gate_up_slices, down_slices;   /* those the card under test runs */
static struct chunk chunks[CHUNK_COUNT];
static struct kernel_launch launches[CHUNK_COUNT * MOST_CHUNK_LAUNCHES];
static struct expert_job_record wide_jobs[ROUTER_WIDE_JOBS(MOST_ROWS)];

static double hash_fraction(uint64_t stream, uint64_t position)
{
    return (double)(test_linear_hash(stream, position) >> HASH_FRACTION_SHIFT) * HASH_FRACTION_UNIT;
}

static uint64_t clock_nanoseconds(void)
{
    struct timespec now = {0};
    return clock_gettime(CLOCK_MONOTONIC, &now) == 0
               ? (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec : 0;
}

static struct linear_range rows_of(struct linear_range range,
				   uint64_t row_bytes, uint32_t first_row,
				   uint32_t row_count)
{
	return (struct linear_range){ range.byte_offset + first_row * row_bytes,
				      row_count * row_bytes };
}

/* Fills a range of the image with one byte, a piece at a time. */
static bool fill(struct test_linear_card *card, struct linear_range range,
		 unsigned char byte)
{
    static unsigned char piece[FILL_PIECE_BYTES];
    bool is_filled = true;
    memset(piece, byte, sizeof(piece));
    for (uint64_t done = 0; done < range.byte_count && is_filled; done += sizeof(piece)) {
        const uint64_t count = range.byte_count - done < sizeof(piece) ? range.byte_count - done : sizeof(piece);
        is_filled = test_linear_card_put(card, range, done, piece, count);
    }
    return is_filled;
}

/* The routed experts of a chain's rows first_row .. first_row + row_count - 1 over a job table. */
static struct linear_routed_experts_description
experts_of(const struct test_linear_card *card, const struct chunk *chunk,
	   enum chain chain, uint32_t first_row, uint32_t row_count,
	   struct linear_range jobs, uint32_t job_count)
{
	const struct linear_range *const written = chunk->written[chain];
	return (struct linear_routed_experts_description){
		.jobs = jobs,
		.expert_weights = test_linear_card_all_weights(card),
		.codes = rows_of(chunk->codes, HIDDEN_SIZE, first_row,
				 row_count),
		.exponents = rows_of(chunk->exponents,
				     HIDDEN_SIZE / VALUES_PER_BLOCK, first_row,
				     row_count),
		.operands = rows_of(chunk->operands, HIDDEN_SIZE * VALUE_BYTES,
				    first_row, row_count),
		.hidden = rows_of(written[WRITTEN_HIDDEN],
				  row_bytes_of[WRITTEN_HIDDEN], first_row,
				  row_count),
		.hidden_codes = rows_of(written[WRITTEN_HIDDEN_CODES],
					row_bytes_of[WRITTEN_HIDDEN_CODES],
					first_row, row_count),
		.hidden_exponents =
			rows_of(written[WRITTEN_HIDDEN_EXPONENTS],
				row_bytes_of[WRITTEN_HIDDEN_EXPONENTS],
				first_row, row_count),
		.hidden_operands =
			rows_of(written[WRITTEN_HIDDEN_OPERANDS],
				row_bytes_of[WRITTEN_HIDDEN_OPERANDS],
				first_row, row_count),
		.pair_output = rows_of(written[WRITTEN_PAIR_OUTPUT],
				       row_bytes_of[WRITTEN_PAIR_OUTPUT],
				       first_row, row_count),
		.row_output = rows_of(written[WRITTEN_ROW_OUTPUT],
				      row_bytes_of[WRITTEN_ROW_OUTPUT],
				      first_row, row_count),
		.gate_up_shape = gate_up_shape,
		.down_shape = down_shape,
		.weight_format = E2M1,
		.job_count = job_count,
		.row_count = row_count,
		.pairs_per_row = PICKS,
		.gate_up_slice_count = gate_up_slices,
		.down_slice_count = down_slices,
		.activation_limit = ACTIVATION_LIMIT
	};
}

/* Every expert's gate, up and down in the weights region, and the table of their offsets. */
static bool matrices_lay_out(struct test_linear_card *card)
{
    bool is_laid_out = true;
    for (uint32_t entry = 0; entry < EXPERTS * MATRICES && is_laid_out; entry++) {
	    const bool is_down = entry % MATRICES == ROUTER_DOWN;
	    const struct test_linear_matrix_recipe recipe = {
		    .format = E2M1,
		    .shape = is_down ? down_shape : gate_up_shape,
		    .stream = CHUNK_STREAM + MATRIX_STREAM + entry,
		    .code = test_linear_e2m1_code,
		    .scale_code_base = is_down ? DOWN_SCALE_CODE_BASE :
						 GATE_UP_SCALE_CODE_BASE,
		    .scale_code_spread = SCALE_CODE_SPREAD
	    };
	    struct linear_range stored = { 0, 0 };
	    is_laid_out = test_linear_card_matrix(card, &recipe, &stored);
	    expert_offsets[entry] = stored.byte_offset;
    }
    return is_laid_out;
}

/* Sets a chunk's ranges aside, lays its scores and rows into the image, the rubbish into what is written, and
 * prepares its launches from launches[chunk->first_launch]. */
static bool chunk_lay_out(struct test_linear_card *card, struct chunk *chunk, uint32_t chunk_index)
{
    static float score_row[EXPERTS];
    static uint16_t value_row[HIDDEN_SIZE];
    const uint32_t rows = chunk->row_count;
    const struct step_block wide_step = { STEP_SEQUENCE, rows, 0 };
    const struct step_block last_step = { STEP_SEQUENCE, rows % THIN_ROWS, 0 };
    const enum linear_order_code order = chunk->is_reference_order ?
						 LINEAR_ORDER_REFERENCE :
						 LINEAR_ORDER_FIXED;
    chunk->wide_step = test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE);
    chunk->last_step = test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE);
    chunk->scores = test_linear_card_work(card, (uint64_t)rows *
							MODEL_ROUTER_ROW_BYTES);
    chunk->values = test_linear_card_work(card, (uint64_t)rows * HIDDEN_SIZE * VALUE_BYTES);
    chunk->codes = test_linear_card_work(card, (uint64_t)rows * HIDDEN_SIZE);
    chunk->exponents = test_linear_card_work(card, (uint64_t)rows * (HIDDEN_SIZE / VALUES_PER_BLOCK));
    chunk->operands = test_linear_card_work(card, (uint64_t)rows * HIDDEN_SIZE * VALUE_BYTES);
    chunk->picks = test_linear_card_work(
	    card, (uint64_t)rows * PICKS * sizeof(struct router_pick));
    chunk->counts = test_linear_card_work(card, (uint64_t)(rows + SORT_ROWS - 1u) / SORT_ROWS * EXPERTS *
                                                    sizeof(uint32_t));
    chunk->pairs = test_linear_card_work(
	    card, (uint64_t)rows * PICKS * sizeof(struct wide_pair));
    chunk->runs =
	    test_linear_card_work(card, EXPERTS * sizeof(struct wide_run));
    chunk->tiles = test_linear_card_work(
	    card, (uint64_t)ROUTER_WIDE_TILES(rows) *
			  sizeof(struct wide_expert_tile));
    chunk->jobs = test_linear_card_work(
	    card, linear_jobs_byte_count(ROUTER_WIDE_JOBS(rows)));
    for (uint32_t chain = 0; chain < CHAIN_COUNT; chain++) {
        for (uint32_t array = 0; array < WRITTEN_COUNT; array++) {
            chunk->written[chain][array] = test_linear_card_work(card, rows * row_bytes_of[array]);
        }
    }
    const struct linear_range router_scratch[] = { chunk->picks, chunk->counts,
						   chunk->pairs, chunk->runs,
						   chunk->tiles, chunk->jobs };
    bool is_laid_out = test_linear_card_has_room(card) &&
                       test_linear_card_put(card, chunk->wide_step, 0, &wide_step, sizeof(wide_step)) &&
                       test_linear_card_put(card, chunk->last_step, 0, &last_step, sizeof(last_step));
    for (uint32_t row = 0; row < rows && is_laid_out; row++) {
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            const uint64_t position = (uint64_t)(row + 1u) * EXPERTS + expert;
            score_row[expert] =
                (float)((hash_fraction(CHUNK_STREAM + SCORE_STREAM + chunk_index + 1u, position) - HALF) * PLAIN_SPAN);
        }
        test_linear_values(value_row, HIDDEN_SIZE, CHUNK_STREAM + ROW_STREAM + (uint64_t)chunk_index * MOST_ROWS + row,
                           row < PLAIN_ROWS ? TEST_LINEAR_VALUES_PLAIN : TEST_LINEAR_VALUES_EDGES);
	is_laid_out =
		test_linear_card_put(card, chunk->scores,
				     (uint64_t)row * MODEL_ROUTER_ROW_BYTES,
				     score_row, sizeof(score_row)) &&
		test_linear_card_put(card, chunk->values,
				     row * sizeof(value_row), value_row,
				     sizeof(value_row));
    }
    for (uint32_t index = 0; index < COUNT_OF(router_scratch) && is_laid_out; index++) {
        is_laid_out = fill(card, router_scratch[index], WIDE_RUBBISH_BYTE);
    }
    for (uint32_t chain = 0; chain < CHAIN_COUNT && is_laid_out; chain++) {
        for (uint32_t array = 0; array < WRITTEN_COUNT && is_laid_out; array++) {
            is_laid_out = fill(card, chunk->written[chain][array], rubbish_of[chain]);
        }
    }
    struct kernel_launch *const first = &launches[chunk->first_launch];
    const struct linear_quantize_description quantize = {
	    .values = chunk->values,
	    .codes = chunk->codes,
	    .exponents = chunk->exponents,
	    .operands = chunk->operands,
	    .width = HIDDEN_SIZE,
	    .row_count = rows
    };
    const struct router_wide_description wide = {
	    .expert_offsets = expert_offsets,
	    .step = chunk->wide_step,
	    .scores = chunk->scores,
	    .bias = bias_range,
	    .offsets = offsets_range,
	    .expert_weights = test_linear_card_all_weights(card),
	    .affinities = chunk->written[CHAIN_WIDE][WRITTEN_AFFINITIES],
	    .picks = chunk->picks,
	    .counts = chunk->counts,
	    .pairs = chunk->pairs,
	    .runs = chunk->runs,
	    .tiles = chunk->tiles,
	    .jobs = chunk->jobs,
	    .gate_up_shape = gate_up_shape,
	    .down_shape = down_shape,
	    .weight_format = E2M1,
	    .order = order,
	    .row_count = rows
    };
    const struct linear_routed_experts_description wide_experts =
	    experts_of(card, chunk, CHAIN_WIDE, 0, rows, chunk->jobs,
		       ROUTER_WIDE_JOBS(rows));
    is_laid_out =
	    is_laid_out &&
	    test_linear_expect_status(
		    linear_quantize_prepare(&quantize, card->arena_byte_count,
					    &card->kernels, first),
		    STATUS_OK, "the quantization of the chunk is prepared") &&
	    test_linear_expect_status(
		    router_wide_prepare(&wide, card->arena_byte_count,
					&router_kernels,
					first + WIDE_SELECT_FIRST),
		    STATUS_OK, "the wide select is prepared") &&
	    test_linear_expect_status(
		    linear_routed_experts_prepare(
			    &wide_experts, card->arena_byte_count,
			    &card->kernels, first + WIDE_EXPERTS_FIRST),
		    STATUS_OK, "the wide chain's experts are prepared");
    chunk->launch_count = THIN_FIRST;
    for (uint32_t first_row = 0; first_row < rows && is_laid_out; first_row += THIN_ROWS) {
        const uint32_t step_rows = rows - first_row < THIN_ROWS ? rows - first_row : THIN_ROWS;
	const struct linear_range jobs = test_linear_card_work(
		card, linear_jobs_byte_count((uint64_t)step_rows * PICKS));
	const struct linear_routed_experts_description experts =
		experts_of(card, chunk, CHAIN_THIN, first_row, step_rows, jobs,
			   step_rows * PICKS);
	const struct router_description select = {
		.experts = &experts,
		.expert_offsets = expert_offsets,
		.step = step_rows == THIN_ROWS ? thin_step_range :
						 chunk->last_step,
		.scores = rows_of(chunk->scores, MODEL_ROUTER_ROW_BYTES,
				  first_row, step_rows),
		.bias = bias_range,
		.offsets = offsets_range,
		.affinities =
			rows_of(chunk->written[CHAIN_THIN][WRITTEN_AFFINITIES],
				MODEL_ROUTER_ROW_BYTES, first_row, step_rows),
		.order = order,
		.expert_count = MODEL_EXPERTS,
		.pick_count = MODEL_EXPERTS_PER_ROW
	};
	is_laid_out =
		test_linear_card_has_room(card) &&
		fill(card, jobs, THIN_RUBBISH_BYTE) &&
		test_linear_expect_status(
			router_select_prepare(&select, card->arena_byte_count,
					      &router_kernels,
					      first + chunk->launch_count),
			STATUS_OK, "a thin select is prepared") &&
		test_linear_expect_status(
			linear_routed_experts_prepare(
				&experts, card->arena_byte_count,
				&card->kernels,
				first + chunk->launch_count + 1u),
			STATUS_OK, "a thin step's experts are prepared");
	chunk->launch_count += THIN_STEP_LAUNCHES;
    }
    return is_laid_out;
}

/* wide = thin, byte for byte, in every array both chains write; the ranges only read untouched. */
static void chunk_check(struct test_linear_card *card, const struct chunk *chunk)
{
	const struct linear_range only_read[] = {
		chunk->wide_step, chunk->last_step, chunk->scores, chunk->values
	};
	const uint32_t job_capacity = ROUTER_WIDE_JOBS(chunk->row_count);
	uint32_t jobs_used = 0, columns_used = 0, failures = 0;
	test_linear_context("card %" PRIu32 ", a chunk of %" PRIu32
			    " rows, the %s order",
			    card->card_index, chunk->row_count,
			    chunk->is_reference_order ? "reference" : "fixed");
	for (uint32_t array = 0; array < WRITTEN_COUNT; array++) {
		failures +=
			test_linear_card_expect_same(
				card, chunk->written[CHAIN_WIDE][array], 0,
				chunk->written[CHAIN_THIN][array], 0,
				chunk->written[CHAIN_WIDE][array].byte_count,
				written_names[array]) ?
				0u :
				1u;
	}
    for (uint32_t index = 0; index < COUNT_OF(only_read); index++) {
        failures += test_linear_card_expect_untouched(card, only_read[index], "a range only read") ? 0u : 1u;
    }
    if (test_linear_card_get(card, chunk->jobs, 0, wide_jobs, chunk->jobs.byte_count)) {
        for (uint32_t job = 0; job < job_capacity; job++) {
            jobs_used += wide_jobs[job].row_count != 0 ? 1u : 0u;
            columns_used += wide_jobs[job].row_count;
        }
    }
    failures += test_linear_expect(columns_used == chunk->row_count * PICKS, "the wide jobs hold every pair once")
                    ? 0u : 1u;
    (void)printf("card %" PRIu32 ": %" PRIu32 " rows, the %s order: the wide chain (%" PRIu32 " jobs of %" PRIu32
                 ", %.2f pairs a job) against %" PRIu32 " thin steps: %s\n", card->card_index, chunk->row_count,
                 chunk->is_reference_order ? "reference" : "fixed", jobs_used, job_capacity,
                 jobs_used != 0 ? (double)columns_used / jobs_used : 0.0,
                 (chunk->row_count + THIN_ROWS - 1u) / THIN_ROWS, failures == 0 ? "every byte the same" : "DIFFERENT");
}

/* Shown failing once: the route weight of the first pair of the first wide job changed on the card and the wide
 * chain's experts run again; that pair's hidden values and outputs and its row's sums differ, and nothing else. */
static void control_check(struct test_linear_card *card, const struct chunk *chunk)
{
	struct expert_job_record job;
	uint32_t failures = 0;
	test_linear_context("card %" PRIu32 ", a chunk of %" PRIu32
			    " rows, a route weight changed on the card",
			    card->card_index, chunk->row_count);
	if (!test_linear_card_get(card, chunk->jobs, 0, &job, sizeof(job)) ||
	    !test_linear_expect(
		    job.row_count != 0 &&
			    job.pair_index[0] < chunk->row_count * PICKS,
		    "the first wide job holds a pair of the chunk")) {
		return;
	}
    const uint32_t pair = job.pair_index[0], row = pair / PICKS;
    job.route_weight[0] *= CHANGED_WEIGHT_FACTOR;
    if (!test_linear_card_put(card, chunk->jobs, 0, &job, sizeof(job)) ||
	!test_linear_card_upload(card, chunk->jobs, 0, sizeof(job)) ||
	!test_linear_card_run(
		card, &launches[chunk->first_launch + WIDE_EXPERTS_FIRST],
		LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT) ||
	!test_linear_card_fetch(card)) {
	    return;
    }
    for (uint32_t array = 0; array < WRITTEN_COUNT; array++) {
        const bool is_of_rows = array == WRITTEN_AFFINITIES || array == WRITTEN_ROW_OUTPUT;
        const uint64_t unit_bytes = is_of_rows ? row_bytes_of[array] : row_bytes_of[array] / PICKS;
        const uint64_t at = (is_of_rows ? row : pair) * unit_bytes;
        const bool must_differ = array == WRITTEN_HIDDEN || array == WRITTEN_PAIR_OUTPUT || array == WRITTEN_ROW_OUTPUT;
        uint64_t first_differing = 0;
        const uint64_t differing = test_linear_card_differing(card, chunk->written[CHAIN_WIDE][array], 0,
                                                              chunk->written[CHAIN_THIN][array], 0,
                                                              chunk->written[CHAIN_WIDE][array].byte_count,
                                                              &first_differing);
        const uint64_t differing_there =
            array == WRITTEN_AFFINITIES ? 0
                                        : test_linear_card_differing(card, chunk->written[CHAIN_WIDE][array], at,
                                                                     chunk->written[CHAIN_THIN][array], at,
                                                                     unit_bytes, &first_differing);
        failures += differing != differing_there || (must_differ && differing_there == 0) ? 1u : 0u;
        (void)printf("  %s: %" PRIu64 " bytes differ, %" PRIu64 " of them the changed %s's\n", written_names[array],
                     differing, differing_there, is_of_rows ? "row" : "pair");
    }
    (void)test_linear_expect(failures == 0, "a changed route weight changes its pair's hidden values and outputs and "
                                            "its row's sums, and nothing else");
}

/* The time of the wide select, of the wide chain's experts and of the thin steps, TIMED_RUNS runs queued at once,
 * after a run of each that warms the card: host clock around the queue and the wait. */
static void time_chunk(struct test_linear_card *card, const struct chunk *chunk)
{
	static struct kernel_launch queued[TIMED_RUNS * TIMED_LAUNCHES];
	const struct {
		const char *name;
		uint32_t first_launch;
		uint32_t launch_count;
	} parts[] = { { "the wide select", WIDE_SELECT_FIRST,
			ROUTER_WIDE_LAUNCH_COUNT },
		      { "the wide chain's experts", WIDE_EXPERTS_FIRST,
			LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT },
		      { "the thin steps", THIN_FIRST,
			chunk->launch_count - THIN_FIRST } };
    bool is_timed = test_linear_expect(chunk->launch_count - THIN_FIRST <= TIMED_LAUNCHES, "the timed chunk fits");
    for (uint32_t part = 0; part < COUNT_OF(parts) && is_timed; part++) {
        const uint32_t count = parts[part].launch_count;
        for (uint32_t run = 0; run < TIMED_RUNS; run++) {
            memcpy(&queued[(size_t)run * count], &launches[chunk->first_launch + parts[part].first_launch],
                   count * sizeof(queued[0]));
        }
        is_timed = test_linear_card_run(card, queued, count);
        const uint64_t started = clock_nanoseconds();
        is_timed = is_timed && test_linear_card_run(card, queued, TIMED_RUNS * count);
        const uint64_t ended = clock_nanoseconds();
        (void)printf("TIME card %" PRIu32 " (class %" PRIu32 "): %s over %" PRIu32 " rows: %.1f us a run (%u runs "
                     "back to back)\n", card->card_index, card->arithmetic_class, parts[part].name, chunk->row_count,
                     (double)(ended - started) / NANOSECONDS_PER_MICROSECOND / TIMED_RUNS, TIMED_RUNS);
    }
}

/* Every chunk on one card: laid out together, loaded once, run once, then held wide against thin. */
static void card_run(struct test_linear_card *card, bool is_timed)
{
	const struct step_block thin_step = { STEP_SEQUENCE, THIN_ROWS, 0 };
	uint32_t launch_count = 0;
	gate_up_slices = test_linear_card_slices(card, GATE_UP_SLICES, true);
	down_slices = test_linear_card_slices(card, DOWN_SLICES, false);
	test_linear_context("card %" PRIu32 ", the chunks laid out",
			    card->card_index);
	test_linear_card_clear(card);
	bias_range = test_linear_card_work(card, MODEL_ROUTER_ROW_BYTES);
	offsets_range = test_linear_card_work(card, MODEL_EXPERT_OFFSETS_BYTES);
	thin_step_range = test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE);
	for (uint32_t expert = 0; expert < EXPERTS; expert++) {
		bias_given[expert] =
			(float)((hash_fraction(CHUNK_STREAM + SCORE_STREAM,
					       expert) -
				 HALF) *
				BIAS_SPAN);
	}
    bool is_ready = matrices_lay_out(card) && test_linear_card_has_room(card) &&
                    test_linear_card_put(card, bias_range, 0, bias_given, sizeof(bias_given)) &&
                    test_linear_card_put(card, offsets_range, 0, expert_offsets, sizeof(expert_offsets)) &&
                    test_linear_card_put(card, thin_step_range, 0, &thin_step, sizeof(thin_step));
    for (uint32_t index = 0; index < CHUNK_COUNT && is_ready; index++) {
        chunks[index] = (struct chunk){.row_count = chunk_rows[index], .is_reference_order = index % HALVES != 0,
                                       .first_launch = launch_count};
        is_ready = chunk_lay_out(card, &chunks[index], index);
        launch_count += chunks[index].launch_count;
    }
    if (!test_linear_expect(is_ready, "the chunks are laid out") || !test_linear_card_load(card) ||
        !test_linear_card_run(card, launches, launch_count) || !test_linear_card_fetch(card)) {
        return;
    }
    (void)test_linear_card_expect_untouched(card, bias_range, "the bias is only read");
    (void)test_linear_card_expect_untouched(card, offsets_range, "the table of offsets is only read");
    (void)test_linear_card_expect_untouched(card, thin_step_range, "the thin steps' block is only read");
    (void)test_linear_card_expect_weights_untouched(card);
    for (uint32_t index = 0; index < CHUNK_COUNT; index++) {
        chunk_check(card, &chunks[index]);
    }
    control_check(card, &chunks[CHUNK_UNEVEN]);
    if (is_timed) {
        time_chunk(card, &chunks[CHUNK_TIMED]);
    }
}

int main(int argument_count, char **arguments)
{
    static struct test_linear_card card;
    uint32_t card_count = 0;
    const bool is_timed = argument_count == MOST_ARGUMENTS && strcmp(arguments[1], "--time") == 0;
    if (argument_count > MOST_ARGUMENTS || (argument_count == MOST_ARGUMENTS && !is_timed)) {
        (void)fprintf(stderr, "usage: expert_chunk_card_test [--time]\n");
        return TEST_WRONG_USAGE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
					      WEIGHTS_CAPACITY) &&
			test_linear_expect_status(
				router_kernels_prepare(&backend_cuda_value,
						       card.card,
						       &router_kernels),
				STATUS_OK,
				"the router's kernels are prepared")) {
			    card_run(&card, is_timed);
			    (void)printf(
				    "card %" PRIu32 " (class %" PRIu32
				    "): digest of everything it computed 0x%016" PRIX64
				    "\n",
				    card_index, card.arithmetic_class,
				    card.digest);
		    }
		    test_linear_card_close(&card);
	    }
    }
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "expert_chunk_card_test FAILED: %d failures\n", test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("expert_chunk_card_test passed on %" PRIu32 " cards\n", card_count);
    return TEST_PASSED;
}
