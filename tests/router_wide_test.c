/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router_wide_test: the wide select's launches (router_wide_prepare, router.h) and the host reference of its
 * sort (test_router_wide_sort, router_reference.h). No card: a prepare function is arithmetic on a description, the
 * arena is a number.
 *
 *   - The reference: three rows that share experts, of a launch of four, sorted by hand: every pair, run (an empty
 *     run's first pair among them), tile and job, the tiles and jobs behind zero, nothing behind the pairs written;
 *     130 rows that all choose the same six experts: runs of 130 cut into tiles of 64, 64 and 2 and into 16 jobs of 8
 *     and one of 2; what the pair count tells apart.
 *   - router_wide_prepare: for 1, 8, 9, 383, 384, 385 and 4,096 rows and both orders, every byte of the
 *     four launches; offsets of the table at both ends of the experts' range; then every refusal of router.h in a
 *     description otherwise accepted: the status, one error line with the words of its rule, the launches untouched.
 * A failure is one line on the standard output: the case, and what was expected of it.
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
#include "log_expectation.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/router.h"
#include "anemone/router_records.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "anemone/wide_linear_records.h"
#include "router_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define EXPERTS MODEL_EXPERTS
#define PICKS MODEL_EXPERTS_PER_ROW
#define TABLE_ENTRIES (MODEL_EXPERTS * MODEL_EXPERT_MATRICES)
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 2048u
enum made_up_handle {
    SELECT_HANDLE = 17, WIDE_SELECT_HANDLE, WIDE_COUNT_HANDLE, WIDE_PLACE_HANDLE, WIDE_JOBS_HANDLE, WIDE_SMALL_HANDLE
};
#define CHANGED_PAIRS 2u

/* The model's experts, 2304 by 5120 and back at 4 bits, as router_test lays them out. */
#define INNER_SIZE 2304u
#define HIDDEN_SIZE 5120u
#define E2M1_BITS_PER_WEIGHT 4u
#define EXPERT_MATRIX_BYTES 6266880ull
#define EXPERT_WEIGHTS_BYTES 7219445760ull
#define SMALL_DOWN_INPUT_SIZE 1152u
#define SMALL_DOWN_BYTES 3133440ull
#define LAST_UP_ENTRY (TABLE_ENTRIES - 2u)
#define FIRST_OFFSET 8589934592ull              /* 2^33: every offset of the test is beyond 32 bits */
#define HALF_AN_ALIGNMENT 8u

/* By hand from router_records.h and wide_linear_records.h: the launches, and the bytes of each array. */
#define SELECT_THREADS 256u
#define SELECT_ROWS 8u                          /* rows of a block of the select */
#define SORT_THREADS 384u                       /* rows of a block of the count and the place */
#define RECORD_BYTES 120u
#define ROUTER_ROW_BYTES 1536u                  /* float32 [384], uint32 [384] */
#define PICK_ROW_BYTES 48u                      /* six picks of 8 bytes, six sorted pairs of 8 */
#define RUNS_BYTES 3072u
#define TILE_BYTES 32u
#define TILE_PAIRS 64u
#define JOB_BYTES 96u
#define JOB_PAIRS 8u
#define JOB_SLACK ((JOB_PAIRS - 1u) * EXPERTS)  /* the ragged last job of every expert at most */
#define MOST_ROWS 4096u

/* Words of the one error line of each rule of router.h. */
#define RULE_MISSING "a missing argument"
#define RULE_KERNEL "is not prepared"
#define RULE_RESERVED "a reserved field"
#define RULE_COUNTS "the router writes"
#define RULE_ALIGNED "an offset is a multiple of"
#define RULE_ARENA "leaves the arena"
#define RULE_TOUCHED "the launches touch"
#define RULE_SHARED "share a byte"
#define RULE_TABLE "lies wholly inside the range of the experts' matrices"

enum launch_dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };

/* ---- the reference by hand: router_test's three rows, of a launch of four ---- */

#define KNOWN_ROWS 3u
#define KNOWN_LAUNCH_ROWS 4u
#define KNOWN_PAIRS (KNOWN_ROWS * PICKS)
#define KNOWN_TILES 14u
#define OFFSETS_BASE 4096u                      /* a made-up table: entry i at OFFSETS_BASE + 16 i */
static const uint32_t known_experts[KNOWN_ROWS][PICKS] = {
    {1, 5, 9, 13, 17, 21}, {5, 9, 30, 31, 32, 33}, {1, 9, 40, 41, 42, 43}};
static const float known_weights[KNOWN_ROWS][PICKS] = {
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
    {20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f}};
static const struct wide_pair known_pairs[KNOWN_PAIRS] = {
	{ 0, 1.0f },   { 12, 20.0f }, { 1, 2.0f },   { 6, 10.0f },
	{ 2, 3.0f },   { 7, 11.0f },  { 13, 21.0f }, { 3, 4.0f },
	{ 4, 5.0f },   { 5, 6.0f },   { 8, 12.0f },  { 9, 13.0f },
	{ 10, 14.0f }, { 11, 15.0f }, { 14, 22.0f }, { 15, 23.0f },
	{ 16, 24.0f }, { 17, 25.0f }
};
struct known_tile {
    uint32_t expert;
    uint32_t first_pair;
    uint32_t pair_count;
};
static const struct known_tile known_tiles[KNOWN_TILES] = {
    {1, 0, 2},   {5, 2, 2},   {9, 4, 3},   {13, 7, 1},  {17, 8, 1},  {21, 9, 1},  {30, 10, 1},
    {31, 11, 1}, {32, 12, 1}, {33, 13, 1}, {40, 14, 1}, {41, 15, 1}, {42, 16, 1}, {43, 17, 1}};

/* 130 rows that choose the same six: weight of row r, slot s is r + s / 8, exact in float32. */
#define SAME_ROWS 130u
#define SAME_TILES_A_RUN 3u                     /* 64, 64, 2 */
#define SAME_LAST_TILE_PAIRS 2u
#define SAME_JOBS_A_RUN 17u                     /* 16 of 8, then 2 */
#define SAME_LAST_JOB_PAIRS 2u
#define EIGHTHS 8.0f
static const uint32_t same_experts[PICKS] = {2, 50, 100, 150, 200, 383};

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_router_wide_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is pointed at the temporary file");
}

static void capture_end(char *text, size_t text_capacity)
{
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, text_capacity - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
}

static void expect_one_error_line_with(const char *text, const char *words, const char *what)
{
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: %s: no single error line with \"%s\": \"%s\"\n", context_text, what, words, text);
        failures++;
    }
}

static uint32_t tiles_of(uint32_t row_count)
{
    return (row_count * PICKS + TILE_PAIRS - 1u) / TILE_PAIRS + EXPERTS;
}

/* The job table of row_count rows: a job a pair at most, and the pairs in eights with every expert's ragged last. */
static uint32_t jobs_of(uint32_t row_count)
{
    const uint32_t pairs = row_count * PICKS, bound = (pairs + JOB_SLACK) / JOB_PAIRS;
    return pairs < bound ? pairs : bound;
}

static uint64_t made_up_offsets[TABLE_ENTRIES];
static struct wide_pair sorted_pairs[MOST_ROWS * PICKS];
static struct wide_run runs[EXPERTS];
static struct wide_expert_tile tiles[MOST_ROWS * PICKS / TILE_PAIRS + EXPERTS];
static struct expert_job_record
	jobs[(MOST_ROWS * PICKS + JOB_SLACK) / JOB_PAIRS];

/* Every byte of a job is zero (its members one by one: a weight by its bits). */
static bool is_zero_job(const struct expert_job_record *job)
{
    bool is_zero = job->gate_offset == 0 && job->up_offset == 0 && job->down_offset == 0 && job->row_count == 0 &&
                   job->expert == 0;
    for (uint32_t column = 0; column < JOB_PAIRS; column++) {
        uint32_t weight_bits = 0;
        memcpy(&weight_bits, &job->route_weight[column], sizeof(weight_bits));
        is_zero = is_zero && job->pair_index[column] == 0 && weight_bits == 0;
    }
    return is_zero;
}

/* Job `job` of the table: its expert's matrices, and the sorted pairs from first_pair on, pair_count of them; every
 * column behind zero. */
static void expect_job(uint32_t job, uint32_t expert, uint32_t first_pair, uint32_t pair_count)
{
	const uint64_t *const offsets =
		&made_up_offsets[(size_t)expert * MODEL_EXPERT_MATRICES];
	bool columns_held = true;
	for (uint32_t column = 0; column < JOB_PAIRS; column++) {
		const bool is_used = column < pair_count;
		const struct wide_pair expected =
			is_used ? sorted_pairs[first_pair + column] :
				  (struct wide_pair){ 0, 0.0f };
		columns_held =
			columns_held &&
			jobs[job].pair_index[column] == expected.pair_index &&
			jobs[job].route_weight[column] == expected.route_weight;
	}
	expect(jobs[job].gate_offset == offsets[ROUTER_GATE] &&
		       jobs[job].up_offset == offsets[ROUTER_UP] &&
		       jobs[job].down_offset == offsets[ROUTER_DOWN] &&
		       jobs[job].row_count == pair_count &&
		       jobs[job].expert == expert && columns_held,
	       "a job names its expert, its matrices and its sorted pairs, zeros behind them");
}

static void expect_tile(uint32_t tile, uint32_t expert, uint32_t first_pair, uint32_t pair_count)
{
	const uint64_t *const offsets =
		&made_up_offsets[(size_t)expert * MODEL_EXPERT_MATRICES];
	expect(tiles[tile].gate_offset == offsets[ROUTER_GATE] &&
		       tiles[tile].up_offset == offsets[ROUTER_UP] &&
		       tiles[tile].down_offset == offsets[ROUTER_DOWN] &&
		       tiles[tile].first_pair == first_pair &&
		       tiles[tile].pair_count == pair_count,
	       "a tile names its expert's matrices, its first pair and its pairs");
}

static void test_reference(void)
{
    static uint32_t experts[SAME_ROWS][PICKS];
    static float weights[SAME_ROWS][PICKS];
    static const unsigned char zero_tile[sizeof(tiles[0])] = {0};
    for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
	    made_up_offsets[entry] =
		    OFFSETS_BASE +
		    (uint64_t)entry * LINEAR_OFFSET_ALIGNMENT_BYTES;
    }
    context_text = "the sort of three rows of a launch of four, by hand";
    memset(sorted_pairs, UNTOUCHED_BYTE, sizeof(sorted_pairs));
    memset(tiles, UNTOUCHED_BYTE, sizeof(tiles));
    test_router_wide_sort(known_experts, known_weights, KNOWN_ROWS, KNOWN_LAUNCH_ROWS, made_up_offsets, sorted_pairs,
                          runs, tiles);
    expect(test_router_wide_pairs_differ(sorted_pairs, known_pairs, KNOWN_PAIRS) == 0, "every pair as by hand");
    unsigned char untouched[sizeof(sorted_pairs[0])];
    const unsigned char *const behind = (const unsigned char *)sorted_pairs + (size_t)KNOWN_PAIRS * sizeof(untouched);
    memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
    expect(memcmp(behind, untouched, sizeof(untouched)) == 0, "nothing behind the live rows' pairs is written");
    for (uint32_t expert = 0, tile = 0, below = 0; expert < EXPERTS; expert++) {
        const bool is_used = tile < KNOWN_TILES && known_tiles[tile].expert == expert;
        expect(runs[expert].first_pair == below &&
                   runs[expert].pair_count == (is_used ? known_tiles[tile].pair_count : 0),
               "a run holds the pairs of lower experts before it, an empty one too");
        below += is_used ? known_tiles[tile].pair_count : 0;
        tile += is_used ? 1u : 0u;
    }
    for (uint32_t tile = 0; tile < tiles_of(KNOWN_LAUNCH_ROWS); tile++) {
        if (tile < KNOWN_TILES) {
            expect_tile(tile, known_tiles[tile].expert, known_tiles[tile].first_pair, known_tiles[tile].pair_count);
        } else {
            expect(memcmp(&tiles[tile], zero_tile, sizeof(zero_tile)) == 0, "every tile behind is zero");
        }
    }
    memset(jobs, UNTOUCHED_BYTE, sizeof(jobs));
    test_router_wide_jobs(known_experts, known_weights, KNOWN_ROWS, KNOWN_LAUNCH_ROWS, made_up_offsets, jobs);
    for (uint32_t job = 0; job < jobs_of(KNOWN_LAUNCH_ROWS); job++) {   /* every run under 8 pairs: a job a tile */
        if (job < KNOWN_TILES) {
            expect_job(job, known_tiles[job].expert, known_tiles[job].first_pair, known_tiles[job].pair_count);
        } else {
            expect(is_zero_job(&jobs[job]), "every job behind is zero");
        }
    }
    context_text = "what tells two arrays of pairs apart";
    struct wide_pair changed[KNOWN_PAIRS];
    memcpy(changed, known_pairs, sizeof(changed));
    changed[1].route_weight = nextafterf(changed[1].route_weight, INFINITY);
    changed[KNOWN_PAIRS - 1u].pair_index = 0;
    expect(test_router_wide_pairs_differ(changed, known_pairs, KNOWN_PAIRS) == CHANGED_PAIRS &&
               test_router_wide_pairs_differ(changed, known_pairs, 1) == 0,
           "a weight a place up and an index changed: two pairs differ, only those");
    changed[1].route_weight = NAN;
    memcpy(&sorted_pairs[1], &changed[1], sizeof(changed[1]));
    sorted_pairs[1].route_weight = -NAN;
    expect(test_router_wide_pairs_differ(&changed[1], &sorted_pairs[1], 1) == 0, "two not-a-numbers do not differ");
    context_text = "130 rows of the same six experts";
    for (uint32_t row = 0; row < SAME_ROWS; row++) {
        for (uint32_t slot = 0; slot < PICKS; slot++) {
            experts[row][slot] = same_experts[slot];
            weights[row][slot] = (float)row + (float)slot / EIGHTHS;
        }
    }
    test_router_wide_sort((const uint32_t(*)[PICKS])experts, (const float(*)[PICKS])weights, SAME_ROWS, SAME_ROWS,
                          made_up_offsets, sorted_pairs, runs, tiles);
    for (uint32_t slot = 0; slot < PICKS; slot++) {
        const uint32_t first = slot * SAME_ROWS;
        expect(runs[same_experts[slot]].first_pair == first && runs[same_experts[slot]].pair_count == SAME_ROWS,
               "a run of every row");
        for (uint32_t row = 0; row < SAME_ROWS; row++) {
            expect(sorted_pairs[first + row].pair_index == row * PICKS + slot &&
                       sorted_pairs[first + row].route_weight == weights[row][slot],
                   "its rows ascending");
        }
        for (uint32_t cut = 0; cut < SAME_TILES_A_RUN; cut++) {
            expect_tile(slot * SAME_TILES_A_RUN + cut, same_experts[slot], first + cut * TILE_PAIRS,
                        cut + 1u < SAME_TILES_A_RUN ? TILE_PAIRS : SAME_LAST_TILE_PAIRS);
        }
    }
    test_router_wide_jobs((const uint32_t(*)[PICKS])experts, (const float(*)[PICKS])weights, SAME_ROWS, SAME_ROWS,
                          made_up_offsets, jobs);
    for (uint32_t slot = 0; slot < PICKS; slot++) {
        for (uint32_t cut = 0; cut < SAME_JOBS_A_RUN; cut++) {
            expect_job(slot * SAME_JOBS_A_RUN + cut, same_experts[slot], slot * SAME_ROWS + cut * JOB_PAIRS,
                       cut + 1u < SAME_JOBS_A_RUN ? JOB_PAIRS : SAME_LAST_JOB_PAIRS);
        }
    }
    for (uint32_t job = PICKS * SAME_JOBS_A_RUN; job < jobs_of(SAME_ROWS); job++) {
        expect(is_zero_job(&jobs[job]), "every job behind is zero");
    }
    for (uint32_t tile = PICKS * SAME_TILES_A_RUN; tile < tiles_of(SAME_ROWS); tile++) {
        expect(memcmp(&tiles[tile], zero_tile, sizeof(zero_tile)) == 0, "every tile behind is zero");
    }
}

/* ---- the launches ---- */

/* An accepted description: every range one behind another from FIRST_OFFSET on, the experts' matrices first, the
 * arena ending with the last range; the table every matrix of every expert one behind another. */
enum case_range { WEIGHTS_RANGE, STEP_RANGE, SCORES_RANGE, BIAS_RANGE, OFFSETS_RANGE, AFFINITIES_RANGE, PICKS_RANGE,
                  COUNTS_RANGE, PAIRS_RANGE, RUNS_RANGE, TILES_RANGE, JOBS_RANGE, RANGE_COUNT };
static const char *const range_names[RANGE_COUNT] = {
    "the experts' matrices", "the step block", "the router scores", "the bias", "the expert offsets",
    "the affinities", "the picks", "the counts", "the sorted pairs", "the runs", "the expert tiles", "the jobs"};
struct test_case {
	struct router_wide_description description;
	struct router_kernels_record kernels;
	uint64_t offsets[TABLE_ENTRIES];
	uint64_t arena_byte_count;
};

static struct linear_range *range_of(struct test_case *test_case,
				     enum case_range range)
{
	struct router_wide_description *const description =
		&test_case->description;
	struct linear_range *const ranges[RANGE_COUNT] = {
		&description->expert_weights, &description->step,
		&description->scores,	      &description->bias,
		&description->offsets,	      &description->affinities,
		&description->picks,	      &description->counts,
		&description->pairs,	      &description->runs,
		&description->tiles,	      &description->jobs
	};
	return ranges[range];
}

static void case_make(struct test_case *test_case, uint32_t row_count,
		      enum linear_order_code order)
{
    const uint64_t rows = row_count, sort_blocks = (row_count + SORT_THREADS - 1u) / SORT_THREADS;
    const uint64_t byte_counts[RANGE_COUNT] = {
	    EXPERT_WEIGHTS_BYTES,
	    STEP_BLOCK_BYTES_VALUE,
	    rows * ROUTER_ROW_BYTES,
	    ROUTER_ROW_BYTES,
	    MODEL_EXPERT_OFFSETS_BYTES,
	    rows * ROUTER_ROW_BYTES,
	    rows * PICK_ROW_BYTES,
	    sort_blocks * ROUTER_ROW_BYTES,
	    rows * PICK_ROW_BYTES,
	    RUNS_BYTES,
	    (uint64_t)tiles_of(row_count) * TILE_BYTES,
	    (uint64_t)jobs_of(row_count) * JOB_BYTES
    };
    uint64_t cursor = FIRST_OFFSET;
    memset(test_case, 0, sizeof(*test_case));
    for (uint32_t range = 0; range < RANGE_COUNT; range++) {
	    *range_of(test_case, (enum case_range)range) =
		    (struct linear_range){ cursor, byte_counts[range] };
	    cursor += byte_counts[range];
    }
    test_case->arena_byte_count = cursor;
    test_case->description.gate_up_shape =
	    (struct matrix_shape_record){ INNER_SIZE, HIDDEN_SIZE,
					  E2M1_BITS_PER_WEIGHT, 0 };
    test_case->description.down_shape =
	    (struct matrix_shape_record){ HIDDEN_SIZE, INNER_SIZE,
					  E2M1_BITS_PER_WEIGHT, 0 };
    test_case->description.weight_format = WEIGHT_FORMAT_E2M1_SCALED;
    test_case->description.order = order;
    test_case->description.row_count = row_count;
    for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
        test_case->offsets[entry] = FIRST_OFFSET + (uint64_t)entry * EXPERT_MATRIX_BYTES;
    }
    test_case->description.expert_offsets = test_case->offsets;
    test_case->kernels = (struct router_kernels_record){
	    SELECT_HANDLE,     WIDE_SELECT_HANDLE, WIDE_COUNT_HANDLE,
	    WIDE_PLACE_HANDLE, WIDE_JOBS_HANDLE,   WIDE_SMALL_HANDLE,
	    BACKEND_NO_KERNEL
    };
}

MUST_CHECK static result_status case_prepare(const struct test_case *test_case,
					     struct kernel_launch *launches)
{
	return router_wide_prepare(&test_case->description,
				   test_case->arena_byte_count,
				   &test_case->kernels, launches);
}

static void test_launches(void)
{
    static const uint32_t row_counts[] = {1, 8, 9, 383, 384, 385, 4096};
    static const unsigned char zero_bytes[LAUNCH_ARGUMENT_BYTES] = { 0 };
    static struct test_case test_case;
    struct kernel_launch launches[ROUTER_WIDE_LAUNCH_COUNT];
    struct router_wide_arguments record;
    context_text = "an accepted description";
    for (size_t index = 0; index < COUNT_OF(row_counts); index++) {
        const uint32_t row_count = row_counts[index];
	const uint32_t grids[ROUTER_WIDE_LAUNCH_COUNT] = {
		(row_count + SELECT_ROWS - 1u) / SELECT_ROWS,
		(row_count + SORT_THREADS - 1u) / SORT_THREADS,
		(row_count + SORT_THREADS - 1u) / SORT_THREADS, 1
	};
	const uint32_t blocks[ROUTER_WIDE_LAUNCH_COUNT] = {
		SELECT_THREADS, SORT_THREADS, SORT_THREADS, SORT_THREADS
	};
	for (uint32_t order = 0; order <= 1; order++) {
		case_make(&test_case, row_count,
			  order != 0 ? LINEAR_ORDER_REFERENCE :
				       LINEAR_ORDER_FIXED);
		memset(launches, UNTOUCHED_BYTE, sizeof(launches));
		expect(case_prepare(&test_case, launches) == STATUS_OK,
		       "it is accepted");
		for (uint32_t launch = 0; launch < ROUTER_WIDE_LAUNCH_COUNT;
		     launch++) {
			const struct kernel_launch *const made =
				&launches[launch];
			expect(made->kernel == WIDE_SELECT_HANDLE + launch &&
				       made->grid[DIMENSION_X] == 1 &&
				       made->grid[DIMENSION_Y] ==
					       grids[launch] &&
				       made->grid[DIMENSION_Z] == 1 &&
				       made->block[DIMENSION_X] ==
					       blocks[launch] &&
				       made->block[DIMENSION_Y] == 1 &&
				       made->block[DIMENSION_Z] == 1 &&
				       made->dynamic_shared_byte_count == 0 &&
				       made->argument_byte_count ==
					       RECORD_BYTES &&
				       made->branch == LAUNCH_CHAIN &&
				       memcmp(made->argument + RECORD_BYTES,
					      zero_bytes,
					      sizeof(zero_bytes) -
						      RECORD_BYTES) == 0 &&
				       memcmp(made->argument,
					      launches[0].argument,
					      RECORD_BYTES) == 0,
			       "select, count, place and jobs: their kernels, grids and blocks, one record, zeros behind");
		}
	    memcpy(&record, launches[0].argument, sizeof(record));
	    const struct router_wide_description *const description =
		    &test_case.description;
	    expect(record.step_offset == description->step.byte_offset &&
                       record.scores_offset == description->scores.byte_offset &&
                       record.bias_offset == description->bias.byte_offset &&
                       record.expert_offsets_offset == description->offsets.byte_offset &&
                       record.affinities_offset == description->affinities.byte_offset &&
                       record.picks_offset == description->picks.byte_offset &&
                       record.counts_offset == description->counts.byte_offset &&
                       record.pairs_offset == description->pairs.byte_offset &&
                       record.runs_offset == description->runs.byte_offset &&
                       record.tiles_offset == description->tiles.byte_offset &&
                       record.jobs_offset == description->jobs.byte_offset && record.weights_offset == FIRST_OFFSET &&
                       record.gate_up_last_offset == FIRST_OFFSET + EXPERT_WEIGHTS_BYTES - EXPERT_MATRIX_BYTES &&
                       record.down_last_offset == record.gate_up_last_offset && record.row_count == row_count &&
                       record.reference_order == order,
                   "the record names every range, the last offsets of the matrices, the rows and the order");
        }
    }
    context_text = "the table at the ends of the experts' range";
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.offsets[TABLE_ENTRIES - 1u] = FIRST_OFFSET + EXPERT_WEIGHTS_BYTES - EXPERT_MATRIX_BYTES;
    test_case.offsets[0] = FIRST_OFFSET;
    expect(case_prepare(&test_case, launches) == STATUS_OK,
	   "a matrix at either end is accepted");
}

/* router_wide_small_prepare: at most a sort block of rows, the select and the small kernel of the wide
 * prepare's record; more rows refused. */
static void test_small_launches(void)
{
    static const uint32_t row_counts[] = {1, 8, 12, 48, 64, 383, 384};
    static struct test_case test_case;
    struct kernel_launch launches[ROUTER_WIDE_LAUNCH_COUNT],
	    wide[ROUTER_WIDE_LAUNCH_COUNT];
    context_text = "the small wide select";
    for (size_t index = 0; index < COUNT_OF(row_counts); index++) {
        const uint32_t row_count = row_counts[index];
	case_make(&test_case, row_count, LINEAR_ORDER_REFERENCE);
	expect(case_prepare(&test_case, wide) == STATUS_OK &&
		       router_wide_small_prepare(&test_case.description,
						 test_case.arena_byte_count,
						 &test_case.kernels,
						 launches) == STATUS_OK,
	       "both accepted");
	expect(memcmp(&launches[0], &wide[0], sizeof(launches[0])) == 0 && launches[1].kernel == WIDE_SMALL_HANDLE &&
                   launches[1].grid[DIMENSION_X] == 1 && launches[1].grid[DIMENSION_Y] == 1 &&
                   launches[1].block[DIMENSION_X] == SORT_THREADS &&
                   memcmp(launches[1].argument, wide[0].argument, RECORD_BYTES) == 0,
               "the wide select, then one block of the small kernel with the same record");
    }
    case_make(&test_case, SORT_THREADS + 1u, LINEAR_ORDER_REFERENCE);
    expect(router_wide_small_prepare(
		   &test_case.description, test_case.arena_byte_count,
		   &test_case.kernels, launches) == STATUS_INVALID_ARGUMENT,
	   "more rows than a sort block refused");
}

/* One refusal: the status, one error line with the words of its rule, the launches untouched. */
static void expect_refusal(const struct test_case *test_case, const char *words, const char *what)
{
	struct kernel_launch launches[ROUTER_WIDE_LAUNCH_COUNT],
		untouched[ROUTER_WIDE_LAUNCH_COUNT];
	char logged[CAPTURE_CAPACITY];
	context_text = what;
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memcpy(untouched, launches, sizeof(untouched));
	capture_begin();
	expect(case_prepare(test_case, launches) == STATUS_INVALID_ARGUMENT,
	       "it is refused");
	capture_end(logged, sizeof(logged));
	expect_one_error_line_with(logged, words, "the refusal");
	expect(memcmp(launches, untouched, sizeof(launches)) == 0,
	       "the launches are untouched");
}

static void test_refusals(void)
{
    static struct test_case test_case;
    struct kernel_launch launches[ROUTER_WIDE_LAUNCH_COUNT];
    char logged[CAPTURE_CAPACITY];
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    context_text = "missing arguments";
    capture_begin();
    expect(router_wide_prepare(NULL, test_case.arena_byte_count,
			       &test_case.kernels,
			       launches) == STATUS_INVALID_ARGUMENT &&
		   router_wide_prepare(&test_case.description,
				       test_case.arena_byte_count, NULL,
				       launches) == STATUS_INVALID_ARGUMENT &&
		   router_wide_prepare(
			   &test_case.description, test_case.arena_byte_count,
			   &test_case.kernels, NULL) == STATUS_INVALID_ARGUMENT,
	   "no description, kernels or launches");
    capture_end(logged, sizeof(logged));
    test_case.description.expert_offsets = NULL;
    expect_refusal(&test_case, RULE_MISSING, "no table of offsets");
    uint32_t *const wide_handles[] = {&test_case.kernels.wide_select, &test_case.kernels.wide_count,
                                      &test_case.kernels.wide_place, &test_case.kernels.wide_jobs};
    for (size_t index = 0; index < COUNT_OF(wide_handles); index++) {
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    *wide_handles[index] = BACKEND_NO_KERNEL;
	    expect_refusal(&test_case, RULE_KERNEL,
			   "a wide kernel that is not prepared");
    }
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.kernels.select = BACKEND_NO_KERNEL;
    expect(case_prepare(&test_case, launches) == STATUS_OK,
	   "the thin select's kernel is not one of them");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.description.reserved_zero = 1;
    expect_refusal(&test_case, RULE_RESERVED, "a reserved field");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    const uint32_t neither_order = LINEAR_ORDER_REFERENCE + 1u;
    memcpy(&test_case.description.order, &neither_order, sizeof(neither_order));
    expect_refusal(&test_case, RULE_COUNTS, "an order that is neither");
    static const uint32_t wrong_rows[] = {0, MOST_ROWS + 1u, UINT32_MAX};
    for (size_t index = 0; index < COUNT_OF(wrong_rows); index++) {
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    test_case.description.row_count = wrong_rows[index];
	    expect_refusal(&test_case, RULE_COUNTS,
			   "rows the wide select does not take");
    }
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.description.weight_format = WEIGHT_FORMAT_E4M3_SCALED;
    expect_refusal(&test_case, RULE_COUNTS, "matrices of 8 bits");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.description.gate_up_shape.bits_per_weight = E2M1_BITS_PER_WEIGHT - 1u;
    expect_refusal(&test_case, RULE_COUNTS, "a shape the format does not store (gate and up)");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.description.down_shape.output_size = HIDDEN_SIZE - 1u;
    expect_refusal(&test_case, RULE_COUNTS, "a shape the format does not store (down)");
    for (uint32_t range = 0; range < RANGE_COUNT; range++) {
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    range_of(&test_case, (enum case_range)range)->byte_offset +=
		    HALF_AN_ALIGNMENT;
	    expect_refusal(&test_case, RULE_ALIGNED, range_names[range]);
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    range_of(&test_case, (enum case_range)range)->byte_count -= 1u;
	    /* the experts' range holds the larger matrix and more: a byte short, its last matrix leaves it */
	    expect_refusal(&test_case,
			   range == WEIGHTS_RANGE ? RULE_TABLE : RULE_TOUCHED,
			   range_names[range]);
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    range_of(&test_case, (enum case_range)range)->byte_offset =
		    test_case.arena_byte_count;
	    expect_refusal(&test_case, RULE_ARENA, range_names[range]);
    }
    for (uint32_t range = 1; range < RANGE_COUNT; range++) {
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    range_of(&test_case, (enum case_range)range)->byte_offset =
		    range_of(&test_case, (enum case_range)(range - 1u))
			    ->byte_offset;
	    expect_refusal(&test_case, RULE_SHARED, range_names[range]);
    }
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.description.step.byte_offset = test_case.description.tiles.byte_offset;
    expect_refusal(&test_case, RULE_SHARED, "the step block on the tiles, which the place writes");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.arena_byte_count -= 1u;
    expect_refusal(&test_case, RULE_ARENA, "an arena a byte short");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.offsets[MODEL_EXPERT_MATRICES + 1u] += HALF_AN_ALIGNMENT;
    expect_refusal(&test_case, RULE_TABLE, "a matrix of the table not on a multiple of the alignment");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.offsets[0] -= LINEAR_OFFSET_ALIGNMENT_BYTES;
    expect_refusal(&test_case, RULE_TABLE, "a matrix before the experts' range");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.offsets[TABLE_ENTRIES - 1u] += LINEAR_OFFSET_ALIGNMENT_BYTES;
    expect_refusal(&test_case, RULE_TABLE, "a down matrix that leaves the experts' range");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.offsets[TABLE_ENTRIES - 1u] =
	    UINT64_MAX - LINEAR_OFFSET_ALIGNMENT_BYTES + 1u;
    expect_refusal(&test_case, RULE_TABLE, "a matrix whose end wraps");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.description.down_shape.input_size = SMALL_DOWN_INPUT_SIZE;
    test_case.offsets[LAST_UP_ENTRY] = FIRST_OFFSET + EXPERT_WEIGHTS_BYTES - SMALL_DOWN_BYTES;
    expect_refusal(&test_case, RULE_TABLE, "an up matrix where only the smaller down matrix fits");
}

int main(void)
{
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream could not be made\n");
        return EXIT_FAILURE;
    }
    test_reference();
    test_launches();
    test_small_launches();
    test_refusals();
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0,
           "the temporary file is removed");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("router_wide_test passed\n");
    return EXIT_SUCCESS;
}
