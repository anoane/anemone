/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router_wide_card_test: the router's wide select (kernel_router.cu: router_wide_select, _count, _place and
 * _jobs) on the real cards, against the host reference of the router (router_reference.h): the select of the card's
 * own affinities, a host stable sort of the picks and the host's job table of them.
 *
 *     router_wide_card_test [--light] [--time] [--padded] [<capture folder>]
 *
 * The capture folder is a forward capture of tools/oracle with layer_L.mlp.routed_expert_ids and .route_weights (int64
 * and float32 [positions][6], as the 2,048-token text's of batch m1a). Without it, as `make card_tests` runs it, the
 * synthetic cases run and it says that nothing was replayed. --light keeps of them, in the fixed order, the plain kind
 * with all, one fewer and about half the rows live, and the one-expert kind with all (for compute-sanitizer's slower
 * tools: the blocks of no live row, a step of none); --time prints, of the plain case of 4,096 rows, the time a run of
 * the select, of the count, the place and the jobs, of the jobs alone and of all four takes, 100 runs back to back.
 * --padded sets the block's padding flag: inactive picks must be no-expert/zero-weight sentinels. Extra 1K/2K cases
 * poison those picks with a valid expert before running, including empty views and partial halves.
 *
 * On every card, synthetic (a hash of the case, the row and the expert), the four launches:
 *   - launches of 1, 9, 385 and 4,096 rows (plain scores also 8, 64, 383, 384 and 1,000), each with the live rows
 *     (the step block's row_count) all of them, one fewer, about half and, with the release modules, one more (the
 *     checked ones trap on it); both orders; scores of five kinds: plain (|s| < 8); ties (four values and two biases);
 *     special (a not-a-number and both infinities in the first row); one (expert 383's bias +1000: every row routed
 *     to it, a run of every live row); few (every bias but eight experts' -1000: 376 empty experts);
 *   - the affinities of the live rows hold against the double-precision affinity; their picks are, byte for byte (but
 *     weights that are a not-a-number on both sides), test_router_select of the card's own affinities; the counts of
 *     every block of live rows, the first 6 L sorted pairs, the 384 runs and every tile (all of them rubbish before
 *     the run) are test_router_wide_sort's of those picks, every job (rubbish before) test_router_wide_jobs' of them;
 *     what lies behind the live rows (affinities, picks, counts, pairs) is still the rubbish; the ranges only read
 *     come back untouched, and so do the guards;
 *   - thin = wide: router_select_entry over the first and the last eight rows of the plain case of 4,096 rows:
 *     their affinities bit-equal to the wide select's, its job table test_router_jobs of the wide picks of those rows;
 *   - class R: 64 sampled rows of that case in a wide launch of 64 rows, then at row 8 k + k % 8 of one of 512 other
 *     plain rows: their affinities and picks bit-equal;
 *   - shown failing once: the bias of the first expert row 0 chose is lowered on the card; the pairs must then differ
 *     from the ones expected before and be those of the changed bias.
 * Replay, for every layer of the capture, the count, the place and the jobs alone over its routing put in as picks
 * (each row's experts ascending with their weights): its positions (the first 4,096), and twice them where they fit
 * (4,096 rows of 2,048): counts, pairs, runs, tiles and jobs the reference's, the picks untouched; it prints the
 * experts used and the longest run.
 * It FAILS on any other difference, on a capture it cannot read, and when a card has no room.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/router.h"
#include "anemone/router_records.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "anemone/wide_linear_records.h"
#include "capture_reader.h"
#include "linear_card_fixture.h"
#include "router_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define TEST_TIME_LIMIT_SECONDS 1800
#define EXPERTS MODEL_EXPERTS
#define PICKS MODEL_EXPERTS_PER_ROW
#define SORT_ROWS ROUTER_SORT_ROWS
#define THIN_ROWS LINEAR_MAXIMUM_ROWS
#define MOST_ROWS MODEL_WIDE_MOST_ROWS
#define MOST_BLOCKS ((MOST_ROWS + SORT_ROWS - 1u) / SORT_ROWS)
#define MOST_TILES ROUTER_WIDE_TILES(MOST_ROWS)
#define MOST_JOBS ROUTER_WIDE_JOBS(MOST_ROWS)
#define TABLE_ENTRIES (MODEL_EXPERTS * MODEL_EXPERT_MATRICES)
#define WORK_CAPACITY 16777216u                 /* 16 MiB: the largest case is about 12.6 */
/* The experts: matrices of 64 by 64 at 4 bits, every entry of the table its own place, 16 bytes after the one before,
 * in a range that holds the last one whole (router_card_test's). */
#define SMALL_SIZE 64u
#define E2M1_BITS_PER_WEIGHT 4u
#define SMALL_MATRIX_BYTES 2176u
#define WEIGHTS_BYTES \
	(SMALL_MATRIX_BYTES + TABLE_ENTRIES * LINEAR_OFFSET_ALIGNMENT_BYTES)
#define WEIGHTS_CAPACITY 65536u
#define RUBBISH_BYTE 0x5A
#define RUBBISH_BYTES ((size_t)MOST_ROWS * MODEL_ROUTER_ROW_BYTES)
#define STEP_SEQUENCE 1u
#define HALF 0.5
#define HASH_FRACTION_SHIFT 40u                 /* the top 24 bits of a hash: a fraction of 2^24 */
#define HASH_FRACTION_UNIT 0x1p-24
#define PLAIN_SPAN 16.0                         /* scores from -8 to 8 */
#define BIAS_SPAN 1.0
#define STEERING_BIAS 1000.0f
#define FEW_EXPERTS 8u                          /* of the kind few: the experts every row chooses among */
#define FEW_STRIDE 47u                          /* expert 47 k of them */
#define ONE_EXPERT 383u
#define LIVE_KINDS 3u                           /* all, one fewer, about half */
#define HALVES 2u
#define ROWS_ABOVE_THE_LAUNCH     \
	(CHECKING_HOOKS_ENABLED ? \
		 0u :             \
		 1u) /* the checked modules trap on a count above */
#define SAMPLED_ROWS 64u
#define SAMPLE_STRIDE (MOST_ROWS / SAMPLED_ROWS)
#define SAMPLE_STEP 7u                          /* row 64 k + (7 k) % 64: a row of every turn, at every place */
#define SAMPLED_COMPANY_ROWS 512u               /* the launch of 512 the sampled rows also run in, at 8 k + k % 8 */
#define SAMPLED_COMPANY_STREAM 1000000u         /* of the plain rows around them: no case counts this far */
#define NAME_CAPACITY 128u
#define ROUTED_FORMAT "layer_%" PRIu32 ".mlp.%s"
#define RECORD_DIMENSIONS 2u
#define ROUTING_REPEATS 2u                      /* a layer's positions, then twice them */
#define MOST_ARGUMENTS \
	5 /* the program's name, three flags and the capture folder */
#define TIMED_RUNS 100u
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum score_kind { PLAIN, TIES, SPECIAL, ONE, FEW, KIND_COUNT };
static const char *const kind_names[KIND_COUNT] = {"plain", "ties", "special", "one expert", "few experts"};
static const uint32_t every_kind_rows[] = {1, 9, 385, 4096};
static const uint32_t plain_rows[] = {8, 64, 383, 384, 1000};
static const float tied_scores[] = {-1.0f, 0.0f, 0.5f, 1.0f};
static const float tied_biases[] = {0.0f, 0.25f};
enum special_expert { NOT_A_NUMBER_EXPERT, INFINITE_EXPERT, NEGATIVE_INFINITE_EXPERT, SPECIAL_EXPERTS };
static const uint32_t special_experts[SPECIAL_EXPERTS] = {17, 42, 43};

/* One wide launch on one card: its ranges, what it was given, what came back and what was expected. */
struct wide_case {
	struct linear_range step, scores, bias, offsets, affinities, picks,
		counts, pairs, runs, tiles, jobs, weights;
	struct router_wide_description description;
	struct kernel_launch launches[ROUTER_WIDE_LAUNCH_COUNT];
	uint32_t row_count;
	uint32_t live_rows; /* of the step block; the kernels take at most row_count */
	bool reference_order;
	float scores_given[MOST_ROWS][EXPERTS];
	float bias_given[EXPERTS];
	float affinities_back[MOST_ROWS][EXPERTS];
	struct router_pick picks_back[MOST_ROWS][PICKS];
	uint32_t counts_back[MOST_BLOCKS][EXPERTS];
	struct wide_pair pairs_back[MOST_ROWS * PICKS];
	struct wide_run runs_back[EXPERTS];
	struct wide_expert_tile tiles_back[MOST_TILES];
	struct expert_job_record jobs_back[MOST_JOBS];
	uint32_t experts_expected[MOST_ROWS][PICKS];
	float weights_expected[MOST_ROWS][PICKS];
	struct wide_pair pairs_expected[MOST_ROWS * PICKS];
	struct wide_run runs_expected[EXPERTS];
	struct wide_expert_tile tiles_expected[MOST_TILES];
	struct expert_job_record jobs_expected[MOST_JOBS];
};

static uint64_t expert_offsets[TABLE_ENTRIES];
static struct router_kernels_record router_kernels;
static unsigned char rubbish[RUBBISH_BYTES];
static struct wide_case wide_cases[1 + 1];     /* a case, and the 64 sampled rows of one */
static bool is_light = false;
static bool is_timed = false;
static bool is_padded = false;

static uint32_t live_of(const struct wide_case *wide)
{
    return wide->live_rows < wide->row_count ? wide->live_rows : wide->row_count;
}

/* Whether elements from .. to - 1 of an array hold the rubbish written before. */
static bool is_rubbish(const void *array, size_t element_bytes, uint32_t from, uint32_t to)
{
    return from >= to || memcmp((const unsigned char *)array + from * element_bytes, rubbish,
                                (to - from) * element_bytes) == 0;
}

/* Whether two stretches hold the same bytes: the bits of floats, not their values. */
static bool same_bytes(const void *first, const void *second, size_t byte_count)
{
    return memcmp(first, second, byte_count) == 0;
}

static uint32_t bits_of(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double hash_fraction(uint64_t stream, uint64_t position)
{
    return (double)(test_linear_hash(stream, position) >> HASH_FRACTION_SHIFT) * HASH_FRACTION_UNIT;
}

/* The scores and the bias of a synthetic case. */
static void scores_make(struct wide_case *wide, enum score_kind kind, uint64_t stream)
{
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        const bool is_kept = expert % FEW_STRIDE == 0 && expert / FEW_STRIDE < FEW_EXPERTS;
        wide->bias_given[expert] = kind == TIES ? tied_biases[test_linear_hash(stream, expert) % COUNT_OF(tied_biases)]
                                                : (float)((hash_fraction(stream, expert) - HALF) * BIAS_SPAN);
        wide->bias_given[expert] += kind == ONE && expert == ONE_EXPERT ? STEERING_BIAS : 0.0f;
        wide->bias_given[expert] -= kind == FEW && !is_kept ? STEERING_BIAS : 0.0f;
    }
    for (uint32_t row = 0; row < wide->row_count; row++) {
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            const uint64_t position = (uint64_t)(row + 1u) * EXPERTS + expert;
            wide->scores_given[row][expert] =
                kind == TIES ? tied_scores[test_linear_hash(stream, position) % COUNT_OF(tied_scores)]
                             : (float)((hash_fraction(stream, position) - HALF) * PLAIN_SPAN);
        }
    }
    if (kind == SPECIAL) {
        wide->scores_given[0][special_experts[NOT_A_NUMBER_EXPERT]] = NAN;
        wide->scores_given[0][special_experts[INFINITE_EXPERT]] = INFINITY;
        wide->scores_given[0][special_experts[NEGATIVE_INFINITE_EXPERT]] = -INFINITY;
    }
}

/* Sets the ranges of a case aside, every array the launches write full of rubbish, and prepares the launches. With
 * picks_given the host's expected picks of the live rows are put in (the count and the place alone run then). */
static bool case_lay_out(struct test_linear_card *card, struct wide_case *wide, bool picks_given)
{
    const uint64_t rows = wide->row_count, sort_blocks = (rows + SORT_ROWS - 1u) / SORT_ROWS;
    const struct step_block step = { STEP_SEQUENCE, wide->live_rows,
				     is_padded ? STEP_BLOCK_PADDED : 0u };
    test_linear_card_clear(card);
    wide->step = test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE);
    wide->scores = test_linear_card_work(card, rows * MODEL_ROUTER_ROW_BYTES);
    wide->bias = test_linear_card_work(card, MODEL_ROUTER_ROW_BYTES);
    wide->offsets = test_linear_card_work(card, MODEL_EXPERT_OFFSETS_BYTES);
    wide->affinities =
	    test_linear_card_work(card, rows * MODEL_ROUTER_ROW_BYTES);
    wide->picks = test_linear_card_work(card, rows * sizeof(wide->picks_back[0]));
    wide->counts = test_linear_card_work(card, sort_blocks * sizeof(wide->counts_back[0]));
    wide->pairs = test_linear_card_work(card, rows * PICKS * sizeof(wide->pairs_back[0]));
    wide->runs = test_linear_card_work(card, sizeof(wide->runs_back));
    wide->tiles =
	    test_linear_card_work(card, ROUTER_WIDE_TILES(wide->row_count) *
						sizeof(wide->tiles_back[0]));
    wide->jobs =
	    test_linear_card_work(card, ROUTER_WIDE_JOBS(wide->row_count) *
						sizeof(wide->jobs_back[0]));
    wide->weights = test_linear_card_weights(card, WEIGHTS_BYTES);
    for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
	    expert_offsets[entry] =
		    wide->weights.byte_offset +
		    (uint64_t)entry * LINEAR_OFFSET_ALIGNMENT_BYTES;
    }
    wide->description = (struct router_wide_description){
	    .expert_offsets = expert_offsets,
	    .step = wide->step,
	    .scores = wide->scores,
	    .bias = wide->bias,
	    .offsets = wide->offsets,
	    .expert_weights = wide->weights,
	    .affinities = wide->affinities,
	    .picks = wide->picks,
	    .counts = wide->counts,
	    .pairs = wide->pairs,
	    .runs = wide->runs,
	    .tiles = wide->tiles,
	    .jobs = wide->jobs,
	    .gate_up_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT,
			       0 },
	    .down_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT, 0 },
	    .weight_format = WEIGHT_FORMAT_E2M1_SCALED,
	    .order = wide->reference_order ? LINEAR_ORDER_REFERENCE :
					     LINEAR_ORDER_FIXED,
	    .row_count = wide->row_count,
    };
    const struct linear_range written[] = { wide->affinities, wide->picks,
					    wide->counts,     wide->pairs,
					    wide->runs,	      wide->tiles,
					    wide->jobs };
    bool is_laid_out = test_linear_card_has_room(card) &&
                       test_linear_card_put(card, wide->step, 0, &step, sizeof(step)) &&
                       test_linear_card_put(card, wide->scores, 0, wide->scores_given, wide->scores.byte_count) &&
                       test_linear_card_put(card, wide->bias, 0, wide->bias_given, sizeof(wide->bias_given)) &&
                       test_linear_card_put(card, wide->offsets, 0, expert_offsets, sizeof(expert_offsets));
    for (size_t index = 0; index < COUNT_OF(written) && is_laid_out; index++) {
        is_laid_out = test_linear_card_put(card, written[index], 0, rubbish, written[index].byte_count);
    }
    for (uint32_t row = 0; row < live_of(wide) && picks_given && is_laid_out; row++) {
        for (uint32_t slot = 0; slot < PICKS; slot++) {
		wide->picks_back[row][slot] = (struct router_pick){
			wide->experts_expected[row][slot],
			wide->weights_expected[row][slot]
		};
	}
        is_laid_out = test_linear_card_put(card, wide->picks, (uint64_t)row * sizeof(wide->picks_back[0]),
                                           wide->picks_back[row], sizeof(wide->picks_back[0]));
    }
    return is_laid_out &&
	   test_linear_expect_status(router_wide_prepare(&wide->description,
							 card->arena_byte_count,
							 &router_kernels,
							 wide->launches),
				     STATUS_OK, "the launches are prepared");
}

/* Runs the four launches, or the count, the place and the jobs alone, and takes back what they wrote. */
static bool case_run(struct test_linear_card *card, struct wide_case *wide, bool sort_alone)
{
	const uint32_t first_launch = sort_alone ? ROUTER_WIDE_COUNT_LAUNCH :
						   ROUTER_WIDE_SELECT_LAUNCH;
	return test_linear_card_run(card, &wide->launches[first_launch],
				    ROUTER_WIDE_LAUNCH_COUNT - first_launch) &&
	       test_linear_card_fetch(card) &&
	       test_linear_card_get(card, wide->affinities, 0,
				    wide->affinities_back,
				    wide->affinities.byte_count) &&
	       test_linear_card_get(card, wide->picks, 0, wide->picks_back,
				    wide->picks.byte_count) &&
	       test_linear_card_get(card, wide->counts, 0, wide->counts_back,
				    wide->counts.byte_count) &&
	       test_linear_card_get(card, wide->pairs, 0, wide->pairs_back,
				    wide->pairs.byte_count) &&
	       test_linear_card_get(card, wide->runs, 0, wide->runs_back,
				    wide->runs.byte_count) &&
	       test_linear_card_get(card, wide->tiles, 0, wide->tiles_back,
				    wide->tiles.byte_count) &&
	       test_linear_card_get(card, wide->jobs, 0, wide->jobs_back,
				    wide->jobs.byte_count);
}

/* What every case holds to. With selected, the picks the host forms of the card's own affinities are expected (else
 * those put in); then the counts, the sort, and the rubbish behind the live rows. */
static bool case_check(struct test_linear_card *card, struct wide_case *wide, bool selected)
{
    const uint32_t live_rows = live_of(wide), live_blocks = (live_rows + SORT_ROWS - 1u) / SORT_ROWS;
    const uint32_t blocks = (wide->row_count + SORT_ROWS - 1u) / SORT_ROWS;
    static uint32_t counts_expected[MOST_BLOCKS][EXPERTS];
    uint32_t failing_affinities = 0, failing_picks = 0, failing_counts = 0, untouched_failures = 0;
    if (!test_linear_expect(wide->row_count >= 1 && wide->row_count <= MOST_ROWS, "a case of 1 to 4,096 rows")) {
        return false;
    }
    for (uint32_t row = 0; row < live_rows && selected; row++) {
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            failing_affinities += test_router_affinity_holds(wide->scores_given[row][expert],
                                                             wide->affinities_back[row][expert]) ? 0u : 1u;
        }
        test_router_select(wide->affinities_back[row], wide->bias_given, wide->reference_order,
                           wide->experts_expected[row], wide->weights_expected[row]);
    }
    for (uint32_t row = 0; row < live_rows; row++) {
        for (uint32_t slot = 0; slot < PICKS; slot++) {
            const float back = wide->picks_back[row][slot].route_weight, expected = wide->weights_expected[row][slot];
            failing_picks += wide->picks_back[row][slot].expert == wide->experts_expected[row][slot] &&
                                     (bits_of(back) == bits_of(expected) || (back != back && expected != expected))
                                 ? 0u : 1u;
        }
    }
    memset(counts_expected, 0, sizeof(counts_expected));
    for (uint32_t row = 0; row < live_rows; row++) {
        for (uint32_t slot = 0; slot < PICKS; slot++) {
            counts_expected[row / SORT_ROWS][wide->experts_expected[row][slot] % EXPERTS]++;
        }
    }
    for (uint32_t block = 0; block < live_blocks; block++) {
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            failing_counts += wide->counts_back[block][expert] == counts_expected[block][expert] ? 0u : 1u;
        }
    }
    test_router_wide_sort((const uint32_t(*)[PICKS])wide->experts_expected,
                          (const float(*)[PICKS])wide->weights_expected, live_rows, wide->row_count, expert_offsets,
                          wide->pairs_expected, wide->runs_expected, wide->tiles_expected);
    const uint32_t differing_pairs = test_router_wide_pairs_differ(wide->pairs_back, wide->pairs_expected,
                                                                   live_rows * PICKS);
    const bool runs_and_tiles_held =
        memcmp(wide->runs_back, wide->runs_expected, sizeof(wide->runs_back)) == 0 &&
        memcmp(wide->tiles_back, wide->tiles_expected, wide->tiles.byte_count) == 0;
    test_router_wide_jobs((const uint32_t(*)[PICKS])wide->experts_expected,
                          (const float(*)[PICKS])wide->weights_expected, live_rows, wide->row_count, expert_offsets,
                          wide->jobs_expected);
    const uint32_t differing_jobs =
	    test_router_jobs_differ(wide->jobs_back, wide->jobs_expected,
				    ROUTER_WIDE_JOBS(wide->row_count));
    untouched_failures +=
        is_rubbish(wide->affinities_back, sizeof(wide->affinities_back[0]), live_rows, wide->row_count) ? 0u : 1u;
    if (is_padded && selected) {
	    for (uint32_t row = live_rows; row < wide->row_count; row++)
		    for (uint32_t slot = 0; slot < PICKS; slot++)
			    failing_picks +=
				    wide->picks_back[row][slot].expert ==
							    EXPERTS &&
						    bits_of(wide
								    ->picks_back
									    [row]
									    [slot]
								    .route_weight) ==
							    0u ?
					    0u :
					    1u;
    } else {
	    untouched_failures += is_rubbish(wide->picks_back,
					     sizeof(wide->picks_back[0]),
					     live_rows, wide->row_count) ?
					  0u :
					  1u;
    }
    untouched_failures += is_rubbish(wide->counts_back, sizeof(wide->counts_back[0]), live_blocks, blocks) ? 0u : 1u;
    untouched_failures +=
        is_rubbish(wide->pairs_back, sizeof(wide->pairs_back[0]), live_rows * PICKS, wide->row_count * PICKS) ? 0u : 1u;
    if (failing_affinities != 0 || failing_picks != 0 || failing_counts != 0 || differing_pairs != 0 ||
        !runs_and_tiles_held || differing_jobs != 0 || untouched_failures != 0) {
        test_linear_fail("%" PRIu32 " affinities off, %" PRIu32 " picks, %" PRIu32 " counts, %" PRIu32
                         " sorted pairs and %" PRIu32 " jobs not the reference's, runs and tiles %s, %" PRIu32
                         " arrays written behind the live rows", failing_affinities, failing_picks, failing_counts,
                         differing_pairs, differing_jobs, runs_and_tiles_held ? "held" : "NOT the reference's",
                         untouched_failures);
        return false;
    }
    return test_linear_card_expect_untouched(card, wide->step, "the step block is only read") &&
           test_linear_card_expect_untouched(card, wide->scores, "the scores are only read") &&
           test_linear_card_expect_untouched(card, wide->bias, "the bias is only read") &&
           test_linear_card_expect_untouched(card, wide->offsets, "the table of offsets is only read") &&
           (selected || test_linear_card_expect_untouched(card, wide->picks, "the picks put in are only read"));
}

static bool case_full_run(struct test_linear_card *card, struct wide_case *wide)
{
    return case_lay_out(card, wide, false) && test_linear_card_load(card) && case_run(card, wide, false) &&
           case_check(card, wide, true);
}

/* thin = wide: the thin select over the first and the last eight live rows of a case that passed. */
static bool thin_check(struct test_linear_card *card, const struct wide_case *wide)
{
    static float thin_affinities[THIN_ROWS][EXPERTS], thin_weights[THIN_ROWS][PICKS];
    static uint32_t thin_experts[THIN_ROWS][PICKS];
    static struct expert_job_record thin_jobs[THIN_ROWS * PICKS],
	    jobs_expected[THIN_ROWS * PICKS];
    const uint32_t live_rows = live_of(wide);
    if (!test_linear_expect(live_rows >= THIN_ROWS && live_rows <= MOST_ROWS, "the case has two thin windows")) {
        return false;
    }
    const uint32_t firsts[] = {0, live_rows - THIN_ROWS};
    bool is_passed = true;
    for (size_t window = 0; window < COUNT_OF(firsts) && is_passed; window++) {
        const uint32_t first = firsts[window];
	const struct step_block step = { STEP_SEQUENCE, THIN_ROWS, 0 };
	struct kernel_launch launch;
	test_linear_card_clear(card);
	const struct linear_range step_range =
		test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE);
	const struct linear_range scores =
		test_linear_card_work(card, sizeof(thin_affinities));
	const struct linear_range bias =
		test_linear_card_work(card, MODEL_ROUTER_ROW_BYTES);
	const struct linear_range offsets =
		test_linear_card_work(card, MODEL_EXPERT_OFFSETS_BYTES);
	const struct linear_range affinities =
		test_linear_card_work(card, sizeof(thin_affinities));
	const struct linear_range jobs =
		test_linear_card_work(card, sizeof(thin_jobs));
	const struct linear_range weights =
		test_linear_card_weights(card, WEIGHTS_BYTES);
	const struct linear_routed_experts_description experts = {
		.jobs = jobs,
		.expert_weights = weights,
		.gate_up_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT,
				   0 },
		.down_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT,
				0 },
		.weight_format = WEIGHT_FORMAT_E2M1_SCALED,
		.job_count = THIN_ROWS * PICKS,
		.row_count = THIN_ROWS,
		.pairs_per_row = PICKS
	};
	const struct router_description description = {
		.experts = &experts,
		.expert_offsets = expert_offsets,
		.step = step_range,
		.scores = scores,
		.bias = bias,
		.offsets = offsets,
		.affinities = affinities,
		.order = wide->reference_order ? LINEAR_ORDER_REFERENCE :
						 LINEAR_ORDER_FIXED,
		.expert_count = MODEL_EXPERTS,
		.pick_count = MODEL_EXPERTS_PER_ROW
	};
	test_linear_context("card %" PRIu32 ", the thin select over rows %" PRIu32 " to %" PRIu32 " of %" PRIu32,
                            card->card_index, first, first + THIN_ROWS - 1u, wide->row_count);
	is_passed =
		test_linear_card_has_room(card) &&
		weights.byte_offset == wide->weights.byte_offset &&
		test_linear_card_put(card, step_range, 0, &step,
				     sizeof(step)) &&
		test_linear_card_put(
			card, scores, 0,
			(const unsigned char *)wide->scores_given +
				(size_t)first * sizeof(wide->scores_given[0]),
			scores.byte_count) &&
		test_linear_card_put(card, bias, 0, wide->bias_given,
				     sizeof(wide->bias_given)) &&
		test_linear_card_put(card, offsets, 0, expert_offsets,
				     sizeof(expert_offsets)) &&
		test_linear_expect_status(
			router_select_prepare(&description,
					      card->arena_byte_count,
					      &router_kernels, &launch),
			STATUS_OK, "the thin launch is prepared") &&
		test_linear_card_load(card) &&
		test_linear_card_run(card, &launch, 1) &&
		test_linear_card_fetch(card) &&
		test_linear_card_get(card, affinities, 0, thin_affinities,
				     sizeof(thin_affinities)) &&
		test_linear_card_get(card, jobs, 0, thin_jobs,
				     sizeof(thin_jobs));
	memcpy(thin_experts, (const unsigned char *)wide->experts_expected + first * sizeof(thin_experts[0]),
               sizeof(thin_experts));
        memcpy(thin_weights, (const unsigned char *)wide->weights_expected + first * sizeof(thin_weights[0]),
               sizeof(thin_weights));
        test_router_jobs((const uint32_t(*)[PICKS])thin_experts, (const float(*)[PICKS])thin_weights, THIN_ROWS,
                         THIN_ROWS, expert_offsets, jobs_expected);
        is_passed = is_passed &&
                    test_linear_expect(same_bytes(thin_affinities,
                                                  (const unsigned char *)wide->affinities_back +
                                                      (size_t)first * sizeof(wide->affinities_back[0]),
                                                  sizeof(thin_affinities)),
                                       "the thin select's affinities are the wide select's, bit for bit") &&
                    test_linear_expect(test_router_jobs_differ(thin_jobs, jobs_expected, THIN_ROWS * PICKS) == 0,
                                       "the thin select's jobs are those of the wide select's picks");
    }
    return is_passed;
}

/* Row 64 k + 7 k % 64 of a case of 4,096 rows: one of every turn of 64, at every place of a turn. */
static uint32_t sampled_row(uint32_t sample)
{
    return sample * SAMPLE_STRIDE + sample * SAMPLE_STEP % SAMPLE_STRIDE;
}

/* Class R: 64 sampled rows of a case keep their affinities and picks in a wide launch of their own and at row
 * 8 k + k % 8 of one of 512 other plain rows. */
static bool sampled_check(struct test_linear_card *card, const struct wide_case *company)
{
    static const uint32_t launch_rows[] = {SAMPLED_ROWS, SAMPLED_COMPANY_ROWS};
    struct wide_case *const sampled = &wide_cases[1];
    bool is_passed = true;
    for (size_t launch = 0; launch < COUNT_OF(launch_rows) && is_passed; launch++) {
        const uint32_t stride = launch_rows[launch] / SAMPLED_ROWS;
        uint32_t differing = 0;
        sampled->row_count = launch_rows[launch];
        sampled->live_rows = launch_rows[launch];
        sampled->reference_order = company->reference_order;
        scores_make(sampled, PLAIN, SAMPLED_COMPANY_STREAM);
        memcpy(sampled->bias_given, company->bias_given, sizeof(sampled->bias_given));
        for (uint32_t row = 0; row < SAMPLED_ROWS; row++) {
            memcpy(sampled->scores_given[row * stride + row % stride], company->scores_given[sampled_row(row)],
                   sizeof(sampled->scores_given[0]));
        }
        test_linear_context("card %" PRIu32 ", 64 sampled rows of %" PRIu32 " in a launch of %" PRIu32,
                            card->card_index, company->row_count, sampled->row_count);
        if (!case_full_run(card, sampled)) {
            return false;
        }
        for (uint32_t row = 0; row < SAMPLED_ROWS; row++) {
            const uint32_t to = row * stride + row % stride, from = sampled_row(row);
            const size_t affinity_bytes = sizeof(sampled->affinities_back[0]);
            const size_t pick_bytes = sizeof(sampled->picks_back[0]);
            const bool is_same =
                same_bytes(sampled->affinities_back[to], company->affinities_back[from], affinity_bytes) &&
                same_bytes(sampled->picks_back[to], company->picks_back[from], pick_bytes);
            differing += is_same ? 0u : 1u;
        }
        is_passed = test_linear_expect(differing == 0, "a row's affinities and picks are those it has in company");
    }
    return is_passed;
}

/* Shown failing: the bias of the first expert row 0 chose, lowered on the card. */
static bool lowered_bias_check(struct test_linear_card *card, struct wide_case *wide)
{
	static struct wide_pair expected_before[MOST_ROWS * PICKS];
	const uint32_t expert = wide->experts_expected[0][0],
		       pair_count = live_of(wide) * PICKS;
	memcpy(expected_before, wide->pairs_expected, sizeof(expected_before));
	wide->bias_given[expert] -= STEERING_BIAS;
	test_linear_context("card %" PRIu32 ", the bias of expert %" PRIu32
			    " lowered",
			    card->card_index, expert);
	if (!test_linear_card_put(card, wide->bias, 0, wide->bias_given,
				  sizeof(wide->bias_given)) ||
	    !test_linear_card_upload(card, wide->bias, 0,
				     sizeof(wide->bias_given)) ||
	    !case_run(card, wide, false)) {
		return false;
	}
    const bool is_shown = test_router_wide_pairs_differ(wide->pairs_back, expected_before, pair_count) != 0;
    return test_linear_expect(is_shown, "the sorted pairs differ from the ones expected before") &&
           case_check(card, wide, true);
}

static uint64_t clock_nanoseconds(void)
{
    struct timespec now = {0};
    return clock_gettime(CLOCK_MONOTONIC, &now) == 0
               ? (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec : 0;
}

/* The time of the select, of the count, the place and the jobs, of the jobs alone and of all four, TIMED_RUNS runs
 * queued at once, after a run of each that warms the card: host clock around the queue and the wait. */
static bool time_run(struct test_linear_card *card, const struct wide_case *wide)
{
	static struct kernel_launch
		queued[TIMED_RUNS * ROUTER_WIDE_LAUNCH_COUNT];
	static const struct {
		const char *name;
		uint32_t first_launch;
		uint32_t launch_count;
	} parts[] = { { "the select", ROUTER_WIDE_SELECT_LAUNCH, 1 },
		      { "the count, the place and the jobs",
			ROUTER_WIDE_COUNT_LAUNCH,
			ROUTER_WIDE_LAUNCH_COUNT - 1u },
		      { "the jobs", ROUTER_WIDE_JOBS_LAUNCH, 1 },
		      { "all four", ROUTER_WIDE_SELECT_LAUNCH,
			ROUTER_WIDE_LAUNCH_COUNT } };
    bool is_passed = true;
    for (size_t part = 0; part < COUNT_OF(parts) && is_passed; part++) {
        const uint32_t count = parts[part].launch_count;
        for (uint32_t run = 0; run < TIMED_RUNS; run++) {
            memcpy(&queued[(size_t)run * count], &wide->launches[parts[part].first_launch], count * sizeof(queued[0]));
        }
        is_passed = test_linear_card_run(card, queued, count);
        const uint64_t started = clock_nanoseconds();
        is_passed = is_passed && test_linear_card_run(card, queued, TIMED_RUNS * count);
        const uint64_t ended = clock_nanoseconds();
        (void)printf("TIME card %" PRIu32 " (class %" PRIu32 "): %s over %" PRIu32 " rows: %.1f us a run (%u runs "
                     "back to back)\n", card->card_index, card->arithmetic_class, parts[part].name, wide->row_count,
                     (double)(ended - started) / NANOSECONDS_PER_MICROSECOND / TIMED_RUNS, TIMED_RUNS);
    }
    return is_passed;
}

/* Every synthetic case on one card. */
static bool synthetic_run(struct test_linear_card *card)
{
    struct wide_case *const wide = &wide_cases[0];
    uint32_t case_count = 0, failed_count = 0;
    uint64_t stream = 0;
    for (uint32_t kind = 0; kind < KIND_COUNT; kind++) {
        for (size_t index = 0; index < COUNT_OF(every_kind_rows) + (kind == PLAIN ? COUNT_OF(plain_rows) : 0);
             index++) {
            const size_t every_kind = COUNT_OF(every_kind_rows);
            const uint32_t row_count = index < every_kind ? every_kind_rows[index] : plain_rows[index - every_kind];
            const uint32_t live_counts[LIVE_KINDS + 1u] = {row_count, row_count - 1u, row_count / HALVES + 1u,
                                                           row_count + 1u};
            for (uint32_t live = 0; live < LIVE_KINDS + ROWS_ABOVE_THE_LAUNCH; live++) {
                for (uint32_t order = 0; order <= 1; order++) {
                    stream++;
                    const bool is_light_case = index < every_kind && order == 0 && live < LIVE_KINDS &&
                                               (kind == PLAIN || (kind == ONE && live == 0));
                    if (is_light && !is_light_case) {
                        continue;
                    }
                    wide->row_count = row_count;
                    wide->live_rows = live_counts[live];
                    wide->reference_order = order != 0;
                    test_linear_context("card %" PRIu32 ", %s scores, %" PRIu32 " of %" PRIu32 " rows live, %s order",
                                        card->card_index, kind_names[kind], wide->live_rows, row_count,
                                        order != 0 ? "the reference's" : "the fixed");
                    scores_make(wide, (enum score_kind)kind, stream);
                    bool is_passed = case_full_run(card, wide);
                    if (is_passed && kind == PLAIN && row_count == MOST_ROWS && live == 0 && order == 0) {
                        is_passed = thin_check(card, wide) && sampled_check(card, wide) &&
                                    case_full_run(card, wide) && (!is_timed || time_run(card, wide)) &&
                                    lowered_bias_check(card, wide);
                    }
                    case_count++;
                    failed_count += is_passed ? 0u : 1u;
                }
            }
        }
    }
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): %" PRIu32 " synthetic cases, %" PRIu32 " failed\n",
                 card->card_index, card->arithmetic_class, case_count, failed_count);
    return failed_count == 0;
}

/* ---- the replay of a capture's routing ---- */

/* The routing of a layer of positions positions as picks of row_count rows, position r % positions in row r (of the
 * first 4,096 positions at most); false if it cannot be read or is not six distinct experts a position. */
static bool routing_read(const struct test_capture *capture, uint32_t layer, uint64_t positions, struct wide_case *wide)
{
    static int64_t ids[MOST_ROWS][PICKS];
    static float weights[MOST_ROWS][PICKS];
    char ids_name[NAME_CAPACITY], weights_name[NAME_CAPACITY];
    const int ids_length = snprintf(ids_name, sizeof(ids_name), ROUTED_FORMAT, layer, "routed_expert_ids");
    const int weights_length = snprintf(weights_name, sizeof(weights_name), ROUTED_FORMAT, layer, "route_weights");
    const uint64_t dimensions[RECORD_DIMENSIONS] = {positions, PICKS};
    const uint32_t read = positions < MOST_ROWS ? (uint32_t)positions : MOST_ROWS;
    if (ids_length <= 0 || (size_t)ids_length >= sizeof(ids_name) ||
	weights_length <= 0 || (size_t)weights_length >= sizeof(weights_name) ||
	positions == 0 ||
	!test_linear_expect_status(
		test_capture_read(capture, ids_name, TEST_CAPTURE_INT64,
				  dimensions, RECORD_DIMENSIONS, 0,
				  (uint64_t)read * PICKS, ids,
				  sizeof(ids[0]) * read),
		STATUS_OK, ids_name) ||
	!test_linear_expect_status(
		test_capture_read(capture, weights_name, TEST_CAPTURE_FLOAT32,
				  dimensions, RECORD_DIMENSIONS, 0,
				  (uint64_t)read * PICKS, weights,
				  sizeof(weights[0]) * read),
		STATUS_OK, weights_name)) {
	    test_linear_fail("layer %" PRIu32
			     ": no routing of positions of six experts",
			     layer);
	    return false;
    }
    for (uint32_t row = 0; row < wide->row_count; row++) {
        const uint32_t position = row % read;
        bool is_valid = true;
        for (uint32_t slot = 0; slot < PICKS; slot++) {     /* the six by expert ascending: the slots */
            uint32_t place = 0;
            for (uint32_t other = 0; other < PICKS; other++) {
                place += ids[position][other] < ids[position][slot] ? 1u : 0u;
                is_valid = is_valid && (other == slot || ids[position][other] != ids[position][slot]);
            }
            is_valid = is_valid && ids[position][slot] >= 0 && ids[position][slot] < (int64_t)EXPERTS;
            if (!is_valid) {
                test_linear_fail("layer %" PRIu32 ", position %" PRIu32 ": not six distinct experts", layer, position);
                return false;
            }
            wide->experts_expected[row][place] = (uint32_t)ids[position][slot];
            wide->weights_expected[row][place] = weights[position][slot];
        }
    }
    return true;
}

/* Every layer of the capture on one card: its positions, and twice them where they fit. */
static bool replay_run(struct test_linear_card *card, const struct test_capture *capture)
{
    struct wide_case *const wide = &wide_cases[0];
    uint32_t layer_count = 0, failed_count = 0;
    for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
	    char name[NAME_CAPACITY];
	    const int length = snprintf(name, sizeof(name), ROUTED_FORMAT,
					layer, "routed_expert_ids");
	    const struct test_capture_tensor *const tensor =
		    length > 0 && (size_t)length < sizeof(name) ?
			    test_capture_find(capture, name) :
			    NULL;
	    if (tensor == NULL) {
		    continue;
	    }
	    const uint64_t positions = tensor->dimension_count ==
						       RECORD_DIMENSIONS ?
					       tensor->dimensions[0] :
					       0;
	    layer_count++;
	    for (uint32_t times = 1;
		 times <= ROUTING_REPEATS &&
		 (times == 1 || times * positions <= MOST_ROWS);
		 times++) {
		    uint32_t largest = 0, used = 0;
		    wide->row_count = times * positions < MOST_ROWS ?
					      (uint32_t)(times * positions) :
					      MOST_ROWS;
		    wide->live_rows = wide->row_count;
		    test_linear_context("card %" PRIu32 ", layer %" PRIu32
					", its routing %" PRIu32 " times",
					card->card_index, layer, times);
		    const bool is_passed =
			    test_linear_expect(positions != 0,
					       "the routing has positions") &&
			    routing_read(capture, layer, positions, wide) &&
			    case_lay_out(card, wide, true) &&
			    test_linear_card_load(card) &&
			    case_run(card, wide, true) &&
			    case_check(card, wide, false);
		    for (uint32_t expert = 0; expert < EXPERTS && is_passed;
			 expert++) {
			    used += wide->runs_back[expert].pair_count != 0 ?
					    1u :
					    0u;
			    largest =
				    wide->runs_back[expert].pair_count >
						    largest ?
					    wide->runs_back[expert].pair_count :
					    largest;
		    }
		    if (is_passed) {
			    (void)printf("card %" PRIu32 ": layer %" PRIu32
					 ", %" PRIu32 " rows: %" PRIu32
					 " experts used, the "
					 "longest run %" PRIu32 " pairs\n",
					 card->card_index, layer,
					 wide->row_count, used, largest);
		    }
		    failed_count += is_passed ? 0u : 1u;
	    }
    }
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): replayed the routing of %" PRIu32 " layers, %" PRIu32
                 " cases failed\n", card->card_index, card->arithmetic_class, layer_count, failed_count);
    return test_linear_expect(layer_count != 0, "the capture holds routings") && failed_count == 0;
}

static bool padding_run(struct test_linear_card *card)
{
	static const uint32_t capacities[] = { 1024u, 2048u };
	static const uint32_t lives[] = {
		0u,   1u,   7u,	   8u,	  78u,	 428u,
		512u, 595u, 1023u, 1024u, 2047u, 2048u
	};
	const struct router_pick poison[PICKS] = { { 0, 1.0f }, { 0, 1.0f },
						   { 0, 1.0f }, { 0, 1.0f },
						   { 0, 1.0f }, { 0, 1.0f } };
	struct wide_case *const wide = &wide_cases[0];
	uint32_t cases = 0, failed = 0;
	for (size_t capacity = 0; capacity < COUNT_OF(capacities); capacity++) {
		for (size_t live = 0; live < COUNT_OF(lives); live++) {
			if (lives[live] > capacities[capacity])
				continue;
			wide->row_count = capacities[capacity];
			wide->live_rows = lives[live];
			wide->reference_order = true;
			scores_make(wide, PLAIN, cases);
			test_linear_context("card %" PRIu32 ", padded %" PRIu32
					    "/%" PRIu32
					    ", valid inactive poison",
					    card->card_index, wide->live_rows,
					    wide->row_count);
			bool passed = case_lay_out(card, wide, false);
			for (uint32_t row = wide->live_rows;
			     row < wide->row_count && passed; row++)
				passed = test_linear_card_put(
					card, wide->picks,
					(uint64_t)row * sizeof(poison), poison,
					sizeof(poison));
			passed = passed && test_linear_card_load(card) &&
				 case_run(card, wide, false) &&
				 case_check(card, wide, true);
			cases++;
			failed += passed ? 0u : 1u;
		}
	}
	(void)printf("card %" PRIu32 ": %" PRIu32
		     " padded poison cases, %" PRIu32 " failed\n",
		     card->card_index, cases, failed);
	return failed == 0;
}

int main(int argument_count, char **arguments)
{
    static struct test_linear_card card;
    struct test_capture *capture = NULL;
    uint32_t card_count = 0, failed_cards = 0;
    const char *capture_folder = NULL;
    bool is_usage = argument_count <= MOST_ARGUMENTS;
    for (int index = 1; index < argument_count && is_usage; index++) {
        const bool is_flag = arguments[index][0] == '-';
        is_light = is_light || strcmp(arguments[index], "--light") == 0;
        is_timed = is_timed || strcmp(arguments[index], "--time") == 0;
	is_padded = is_padded || strcmp(arguments[index], "--padded") == 0;
	is_usage = is_flag ? strcmp(arguments[index], "--light") == 0 ||
				     strcmp(arguments[index], "--time") == 0 ||
				     strcmp(arguments[index], "--padded") == 0 :
			     capture_folder == NULL;
	capture_folder = is_flag ? capture_folder : arguments[index];
    }
    if (!is_usage) {
	    (void)fprintf(
		    stderr,
		    "usage: router_wide_card_test [--light] [--time] [--padded] [<capture folder>]\n");
	    return TEST_WRONG_USAGE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    memset(rubbish, RUBBISH_BYTE, sizeof(rubbish));
    if (capture_folder != NULL) {
	    (void)test_linear_expect_status(test_capture_open(capture_folder,
							      &capture),
					    STATUS_OK, "the capture opens");
    }
    if (test_linear_failures() == 0 &&
	test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    bool is_passed =
			    test_linear_card_open(&card, card_index,
						  WORK_CAPACITY,
						  WEIGHTS_CAPACITY) &&
			    test_linear_expect_status(
				    router_kernels_prepare(&backend_cuda_value,
							   card.card,
							   &router_kernels),
				    STATUS_OK,
				    "the router's kernels are prepared");
		    is_passed = is_passed && synthetic_run(&card);
		    is_passed = is_passed && (!is_padded || padding_run(&card));
		    is_passed = is_passed &&
				(capture == NULL || replay_run(&card, capture));
		    test_linear_card_close(&card);
		    failed_cards += is_passed ? 0u : 1u;
	    }
    }
    test_capture_close(capture);
    if (test_linear_failures() != 0 || failed_cards != 0) {
        (void)fprintf(stderr, "router_wide_card_test FAILED: %" PRIu32 " cards, %d failure lines\n", failed_cards,
                      test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("router_wide_card_test passed on %" PRIu32 " cards (%s)\n", card_count,
                 capture != NULL ? "synthetic cases and the capture's routings" : "synthetic cases; nothing replayed: "
                                                                                  "it needs a capture folder");
    return TEST_PASSED;
}
