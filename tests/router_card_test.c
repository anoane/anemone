/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router_card_test: the router's kernel (kernel_router.cu) on the real cards, against the host reference of the
 * router (router_reference.h) on synthetic scores and, given a capture of the reference, on the reference's own.
 *
 *     router_card_test [<capture folder> <pack copy> [<pack copy> ...]]
 *
 * The capture folder is one of tools/oracle/stepwise_capture.py: its capture points of the router (layer_L.position_P.
 * router.logits, .scores, .biased_scores, .expert_ids and .route_weights). The pack is one of the original format,
 * whose layers.L.ffn.gate.bias the replay reads. Without arguments, as `make card_tests` runs it, the synthetic cases
 * run and it says that nothing was replayed.
 *
 * On every card, synthetic (a hash of the case, the row and the expert):
 *   - launches of 1, 2, 5 and 8 rows, each with every count of live rows (the step block's row_count) up to its own
 *     and, with the release modules, one above it (the checked ones trap on it), in both orders; scores of five
 *     kinds: plain (|s| < 8); wide (from -120, where e^s is not a normal float32, to 40, beyond the threshold, 20
 *     itself among them); ties (four values and two biases, so that biased affinities tie and the lower expert must
 *     win); shared (each row its neighbour's scores but a few, so that rows choose the same experts); and plain with
 *     a not-a-number and both infinities in the first row;
 *   - the affinities of the live rows hold against the double-precision affinity, those behind them are untouched;
 *   - the job table is, byte for byte (but weights that are a not-a-number on both sides), test_router_jobs of
 *     test_router_select of the card's own affinities: every expert, offset, slot, column and weight; nothing of a
 *     row at or behind the live count; zeros everywhere else, where the table held rubbish before;
 *   - a row's affinities have the same bits in a launch of its own as in company (and so its pairs, which the
 *     reference forms from them alone);
 *   - shown failing once: the bias of the first expert row 0 chose is lowered on the card; the table must then
 *     differ from the one expected before and be the one of the changed bias;
 *   - the guards behind every range and the ranges the kernel only reads come back untouched.
 *   - the draft's router (router_select_draft, dspark_v1/DESIGN.md K2): the same launches, live rows and score
 *     kinds over 128 experts and 3 picks in the fixed order, its table's entries from 128 on zero (the checked
 *     modules trap on reading one: an offset below the experts' range); the affinities, the rows behind and the job
 *     table held as above to test_router_draft_jobs of test_router_select_draft; shown failing once with a lowered
 *     bias. The model's select keeps its bits: its cases above are unchanged.
 * Replay, for every capture point of the router, in launches of up to 8 positions of one layer, in both orders: the
 * affinities bit-equal to the capture's scores, and they plus the pack's bias to its biased scores; the six experts
 * the capture's; with the reference's order every weight bit-equal. Weights of the fixed order are held to the host
 * reference of that order (as in every case); their distance to the capture's is printed, not held. Shown failing
 * once a card: the score and the bias of the first expert the capture chose for row 0 of the first batch lowered;
 * the replay must find the affinities of that row alone, the biased scores of every row and that row's experts
 * apart from the capture's.
 * It FAILS on any other difference, on a capture or a pack it cannot read, and when a card has no room.
 */
#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/draft.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/pack.h"
#include "anemone/router.h"
#include "anemone/router_records.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "anemone/storage_seam.h"
#include "capture_reader.h"
#include "linear_card_fixture.h"
#include "router_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define TEST_TIME_LIMIT_SECONDS 1800
#define ARGUMENTS_BEFORE_THE_COPIES 2           /* the program's name and the capture folder */
#define EXPERTS MODEL_EXPERTS
#define PICKS MODEL_EXPERTS_PER_ROW
#define MOST_ROWS LINEAR_MAXIMUM_ROWS
#define MOST_JOBS (MOST_ROWS * PICKS)
#define TABLE_ENTRIES (MODEL_EXPERTS * MODEL_EXPERT_MATRICES)
#define WORK_CAPACITY 1048576u                  /* far above a case's ranges and guards */
/* The experts of the launches: matrices of 64 by 64 at 4 bits (2,048 bytes and 128 of scales), every entry of the
 * table its own place, 16 bytes after the one before, in a range that holds the last one whole. */
#define SMALL_SIZE 64u
#define E2M1_BITS_PER_WEIGHT 4u
#define SMALL_MATRIX_BYTES 2176u
#define WEIGHTS_BYTES \
	(SMALL_MATRIX_BYTES + TABLE_ENTRIES * LINEAR_OFFSET_ALIGNMENT_BYTES)
#define WEIGHTS_CAPACITY 65536u
#define RUBBISH_BYTE 0x5A
#define HALF 0.5
#define WIDE_KINDS 2u                           /* a wide score is one of the table or one of the span */
#define STEP_SEQUENCE 1u
#define HASH_FRACTION_SHIFT 40u                 /* the top 24 bits of a hash: a fraction of 2^24 */
#define HASH_FRACTION_UNIT 0x1p-24
#define PLAIN_SPAN 16.0                         /* scores from -8 to 8 */
#define WIDE_LOWEST (-120.0)
#define WIDE_SPAN 160.0
#define BIAS_SPAN 1.0                           /* a bias from -0.5 to 0.5 */
#define SHARED_CHANGES 16u                      /* one expert in 16 of a row is not its neighbour's */
#define BIAS_LOWERED 1000.0f
#define ROWS_ABOVE_THE_LAUNCH     \
	(CHECKING_HOOKS_ENABLED ? \
		 0u :             \
		 1u) /* the checked modules trap on a count above */
#define RUBBISH_BYTES \
	(MOST_ROWS *  \
	 MODEL_ROUTER_ROW_BYTES) /* what the affinities and the job table held */
#define MOST_POINTS 65536u
#define NAME_CAPACITY 128u
#define BIAS_NAME_FORMAT "layers.%" PRIu32 ".ffn.gate.bias"
#define RECORD_FORMAT "layer_%" PRIu32 ".position_%" PRIu32 ".router.%s"
#define READ_CAPACITY 8192u                     /* two blocks of the largest alignment: a bias of 1,536 bytes */
#define DECIMAL 10
#define ORDERS 2u                               /* the fixed one and the reference's */
#define DRAFT_EXPERTS MODEL_DRAFT_EXPERTS
#define DRAFT_PICKS MODEL_DRAFT_EXPERTS_PER_ROW
#define MOST_DRAFT_JOBS (MOST_ROWS * DRAFT_PICKS)
#define DRAFT_ROW_BYTES (DRAFT_EXPERTS * sizeof(float))
#define DRAFT_STREAM_BASE 0x10000u              /* the draft's cases take streams of their own */

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum score_kind { PLAIN, WIDE, TIES, SHARED, SPECIAL, KIND_COUNT };
static const char *const kind_names[KIND_COUNT] = {"plain", "wide", "ties", "shared", "special"};
static const uint32_t launch_rows[] = {1, 2, 5, 8};
static const float wide_scores[] = {-120.0f, -103.5f, -88.0f, -87.0f, -50.0f, 0.0f, 19.999998f, 20.0f, 20.000002f,
                                    25.0f, 40.0f};
static const float tied_scores[] = {-1.0f, 0.0f, 0.5f, 1.0f};
static const float tied_biases[] = {0.0f, 0.25f};
enum special_expert { NOT_A_NUMBER_EXPERT, INFINITE_EXPERT, NEGATIVE_INFINITE_EXPERT, SPECIAL_EXPERTS };
static const uint32_t special_experts[SPECIAL_EXPERTS] = {17, 42, 43};

/* One launch on one card: its ranges, what it was given and what came back. */
struct router_case {
	struct linear_range step, scores, bias, offsets, affinities, jobs,
		weights;
	struct linear_routed_experts_description experts;
	struct router_description description;
	struct kernel_launch launch;
	uint32_t row_count;
	uint32_t live_rows;
	float scores_given[MOST_ROWS][EXPERTS];
	float bias_given[EXPERTS];
	float affinities_back[MOST_ROWS][EXPERTS];
	struct expert_job_record jobs_back[MOST_JOBS];
	uint32_t experts_expected[MOST_ROWS][PICKS];
	float weights_expected[MOST_ROWS][PICKS];
	struct expert_job_record jobs_expected[MOST_JOBS];
};

_Static_assert(RUBBISH_BYTES >= MOST_JOBS * EXPERT_JOB_BYTES,
	       "rubbish fills the largest job table too");
static uint64_t expert_offsets[TABLE_ENTRIES];
static struct router_kernels_record router_kernels;

/* Whether two stretches hold the same bytes: the bits of floats, not their values. */
static bool same_bits(const void *first, const void *second, size_t byte_count)
{
    return memcmp(first, second, byte_count) == 0;
}

static double hash_fraction(uint64_t stream, uint64_t position)
{
    return (double)(test_linear_hash(stream, position) >> HASH_FRACTION_SHIFT) * HASH_FRACTION_UNIT;
}

/* The scores and the bias of a synthetic case. */
static void scores_make(struct router_case *router_case, enum score_kind kind, uint64_t stream)
{
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        const uint64_t hash = test_linear_hash(stream, expert);
        router_case->bias_given[expert] = kind == TIES ? tied_biases[hash % COUNT_OF(tied_biases)]
                                                       : (float)((hash_fraction(stream, expert) - HALF) * BIAS_SPAN);
    }
    for (uint32_t row = 0; row < router_case->row_count; row++) {
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            const uint64_t position = (uint64_t)(row + 1u) * EXPERTS + expert;
            const uint64_t hash = test_linear_hash(stream, position);
            const double fraction = hash_fraction(stream, position);
            float score = (float)((fraction - HALF) * PLAIN_SPAN);
            if (kind == WIDE) {
                score = hash % WIDE_KINDS == 0 ? wide_scores[hash / WIDE_KINDS % COUNT_OF(wide_scores)]
                                               : (float)(WIDE_LOWEST + fraction * WIDE_SPAN);
            } else if (kind == TIES) {
                score = tied_scores[hash % COUNT_OF(tied_scores)];
            } else if (kind == SHARED && row != 0 && hash % SHARED_CHANGES != 0) {
                score = router_case->scores_given[row - 1u][expert];
            }
            router_case->scores_given[row][expert] = score;
        }
    }
    if (kind == SPECIAL) {
        router_case->scores_given[0][special_experts[NOT_A_NUMBER_EXPERT]] = NAN;
        router_case->scores_given[0][special_experts[INFINITE_EXPERT]] = INFINITY;
        router_case->scores_given[0][special_experts[NEGATIVE_INFINITE_EXPERT]] = -INFINITY;
    }
}

/* Sets the ranges of a case aside on the card, lays out its image and prepares its launch. */
static bool case_lay_out(struct test_linear_card *card,
			 struct router_case *router_case, uint32_t live_rows,
			 enum linear_order_code order)
{
	const uint64_t rows_byte_count =
		(uint64_t)router_case->row_count * MODEL_ROUTER_ROW_BYTES;
	const uint64_t jobs_byte_count =
		(uint64_t)router_case->row_count * PICKS * EXPERT_JOB_BYTES;
	const struct step_block step = { STEP_SEQUENCE, live_rows, 0 };
	static unsigned char rubbish[RUBBISH_BYTES];
	memset(rubbish, RUBBISH_BYTE, sizeof(rubbish));
	test_linear_card_clear(card);
	router_case->live_rows = live_rows;
	router_case->step = test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE);
	router_case->scores = test_linear_card_work(card, rows_byte_count);
	router_case->bias = test_linear_card_work(card, MODEL_ROUTER_ROW_BYTES);
	router_case->offsets =
		test_linear_card_work(card, MODEL_EXPERT_OFFSETS_BYTES);
	router_case->affinities = test_linear_card_work(card, rows_byte_count);
	router_case->jobs = test_linear_card_work(card, jobs_byte_count);
	router_case->weights = test_linear_card_weights(card, WEIGHTS_BYTES);
	for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
		expert_offsets[entry] =
			router_case->weights.byte_offset +
			(uint64_t)entry * LINEAR_OFFSET_ALIGNMENT_BYTES;
	}
	router_case->experts = (struct linear_routed_experts_description){
		.jobs = router_case->jobs,
		.expert_weights = router_case->weights,
		.gate_up_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT,
				   0 },
		.down_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT,
				0 },
		.weight_format = WEIGHT_FORMAT_E2M1_SCALED,
		.job_count = router_case->row_count * PICKS,
		.row_count = router_case->row_count,
		.pairs_per_row = PICKS,
	};
	router_case->description = (struct router_description){
		.experts = &router_case->experts,
		.expert_offsets = expert_offsets,
		.step = router_case->step,
		.scores = router_case->scores,
		.bias = router_case->bias,
		.offsets = router_case->offsets,
		.affinities = router_case->affinities,
		.order = order,
		.expert_count = EXPERTS,
		.pick_count = PICKS,
	};
	return test_linear_card_has_room(card) &&
	       test_linear_card_put(card, router_case->step, 0, &step,
				    sizeof(step)) &&
	       test_linear_card_put(card, router_case->scores, 0,
				    router_case->scores_given,
				    rows_byte_count) &&
	       test_linear_card_put(card, router_case->bias, 0,
				    router_case->bias_given,
				    MODEL_ROUTER_ROW_BYTES) &&
	       test_linear_card_put(card, router_case->offsets, 0,
				    expert_offsets,
				    MODEL_EXPERT_OFFSETS_BYTES) &&
	       test_linear_card_put(card, router_case->affinities, 0, rubbish,
				    rows_byte_count) &&
	       test_linear_card_put(card, router_case->jobs, 0, rubbish,
				    jobs_byte_count) &&
	       test_linear_expect_status(
		       router_select_prepare(&router_case->description,
					     card->arena_byte_count,
					     &router_kernels,
					     &router_case->launch),
		       STATUS_OK, "the launch is prepared");
}

/* Runs the launch and takes back what it wrote. */
static bool case_run(struct test_linear_card *card, struct router_case *router_case)
{
	const uint64_t rows_byte_count =
		(uint64_t)router_case->row_count * MODEL_ROUTER_ROW_BYTES;
	return test_linear_card_run(card, &router_case->launch, 1) &&
	       test_linear_card_fetch(card) &&
	       test_linear_card_get(card, router_case->affinities, 0,
				    router_case->affinities_back,
				    rows_byte_count) &&
	       test_linear_card_get(card, router_case->jobs, 0,
				    router_case->jobs_back,
				    router_case->jobs.byte_count);
}

/* The table the card must have written, from its own affinities of the live rows. */
static void case_expect(struct router_case *router_case, bool reference_order)
{
    const uint32_t live_rows =
        router_case->live_rows < router_case->row_count ? router_case->live_rows : router_case->row_count;
    for (uint32_t row = 0; row < live_rows; row++) {
        test_router_select(router_case->affinities_back[row], router_case->bias_given, reference_order,
                           router_case->experts_expected[row], router_case->weights_expected[row]);
    }
    test_router_jobs((const uint32_t(*)[PICKS])router_case->experts_expected,
                     (const float(*)[PICKS])router_case->weights_expected, live_rows, router_case->row_count,
                     expert_offsets, router_case->jobs_expected);
}

/* What every case holds to: the affinities, the rows behind, the table, the ranges only read. */
static bool case_check(struct test_linear_card *card, struct router_case *router_case, bool reference_order)
{
    const uint32_t live_rows =
        router_case->live_rows < router_case->row_count ? router_case->live_rows : router_case->row_count;
    uint32_t failing_values = 0, touched_rows = 0;
    unsigned char rubbish_row[MODEL_ROUTER_ROW_BYTES];
    memset(rubbish_row, RUBBISH_BYTE, sizeof(rubbish_row));
    for (uint32_t row = 0; row < router_case->row_count; row++) {
        for (uint32_t expert = 0; expert < EXPERTS && row < live_rows; expert++) {
            failing_values += test_router_affinity_holds(router_case->scores_given[row][expert],
                                                         router_case->affinities_back[row][expert])
                                  ? 0u
                                  : 1u;
        }
        touched_rows +=
            row >= live_rows && !same_bits(router_case->affinities_back[row], rubbish_row, sizeof(rubbish_row)) ? 1u
                                                                                                               : 0u;
    }
    case_expect(router_case, reference_order);
    const uint32_t differing =
        test_router_jobs_differ(router_case->jobs_back, router_case->jobs_expected, router_case->row_count * PICKS);
    if (failing_values != 0 || touched_rows != 0 || differing != 0) {
        test_linear_fail("%" PRIu32 " affinities off, %" PRIu32 " rows behind the live ones written, %" PRIu32
                         " entries of the table not the reference's", failing_values, touched_rows, differing);
    }
    return failing_values == 0 && touched_rows == 0 && differing == 0 &&
           test_linear_card_expect_untouched(card, router_case->step, "the step block is only read") &&
           test_linear_card_expect_untouched(card, router_case->scores, "the scores are only read") &&
           test_linear_card_expect_untouched(card, router_case->bias, "the bias is only read") &&
           test_linear_card_expect_untouched(card, router_case->offsets, "the table of offsets is only read");
}

/* Each row of a case of 8 live rows launched alone: its affinities keep their bits. */
static bool rows_alone_check(struct test_linear_card *card, const struct router_case *company, uint64_t stream)
{
    static struct router_case alone;
    bool is_passed = true;
    for (uint32_t row = 0; row < company->live_rows && is_passed; row++) {
        alone.row_count = 1;
        memcpy(alone.scores_given[0], company->scores_given[row], sizeof(alone.scores_given[0]));
        memcpy(alone.bias_given, company->bias_given, sizeof(alone.bias_given));
        test_linear_context("card %" PRIu32 ", row %" PRIu32 " of stream %" PRIu64 " alone", card->card_index, row,
                            stream);
	is_passed = case_lay_out(card, &alone, 1, LINEAR_ORDER_FIXED) &&
		    test_linear_card_load(card) && case_run(card, &alone) &&
		    case_check(card, &alone, false) &&
		    test_linear_expect(
			    same_bits(alone.affinities_back[0],
				      company->affinities_back[row],
				      sizeof(alone.affinities_back[0])),
			    "a row alone has the affinities it has in company");
    }
    return is_passed;
}

/* Shown failing: the bias of the first expert row 0 chose, lowered on the card. */
static bool lowered_bias_check(struct test_linear_card *card, struct router_case *router_case)
{
    const uint32_t expert = router_case->experts_expected[0][0];
    const uint32_t entry_count = router_case->row_count * PICKS;
    static struct expert_job_record expected_before[MOST_JOBS];
    memcpy(expected_before, router_case->jobs_expected, sizeof(expected_before));
    router_case->bias_given[expert] -= BIAS_LOWERED;
    test_linear_context("card %" PRIu32 ", the bias of expert %" PRIu32 " lowered", card->card_index, expert);
    if (!test_linear_card_put(card, router_case->bias, 0,
			      router_case->bias_given,
			      MODEL_ROUTER_ROW_BYTES) ||
	!test_linear_card_upload(card, router_case->bias, 0,
				 MODEL_ROUTER_ROW_BYTES) ||
	!case_run(card, router_case)) {
	    return false;
    }
    const bool is_shown = test_router_jobs_differ(router_case->jobs_back, expected_before, entry_count) != 0;
    return test_linear_expect(is_shown, "the table differs from the one expected before") &&
           case_check(card, router_case, false);
}

/* Every synthetic case on one card. */
static bool synthetic_run(struct test_linear_card *card)
{
    static struct router_case router_case;
    uint32_t case_count = 0, failed_count = 0;
    uint64_t stream = 0;
    for (uint32_t kind = 0; kind < KIND_COUNT; kind++) {
        for (size_t launch = 0; launch < COUNT_OF(launch_rows); launch++) {
            for (uint32_t live_rows = 1; live_rows <= launch_rows[launch] + ROWS_ABOVE_THE_LAUNCH; live_rows++) {
                for (uint32_t order = 0; order <= 1; order++) {
                    const bool is_reference = order != 0;
                    stream++;
                    router_case.row_count = launch_rows[launch];
                    test_linear_context("card %" PRIu32 ", %s scores, %" PRIu32 " of %" PRIu32 " rows live, %s order",
                                        card->card_index, kind_names[kind], live_rows, launch_rows[launch],
                                        is_reference ? "the reference's" : "the fixed");
                    scores_make(&router_case, (enum score_kind)kind, stream);
		    bool is_passed =
			    case_lay_out(card, &router_case, live_rows,
					 is_reference ? LINEAR_ORDER_REFERENCE :
							LINEAR_ORDER_FIXED) &&
			    test_linear_card_load(card) &&
			    case_run(card, &router_case) &&
			    case_check(card, &router_case, is_reference);
		    if (is_passed && kind == PLAIN && live_rows == MOST_ROWS && launch_rows[launch] == MOST_ROWS &&
                        !is_reference) {
			    is_passed = rows_alone_check(card, &router_case,
							 stream) &&
					case_lay_out(card, &router_case,
						     live_rows,
						     LINEAR_ORDER_FIXED) &&
					test_linear_card_load(card) &&
					case_run(card, &router_case) &&
					case_check(card, &router_case, false) &&
					lowered_bias_check(card, &router_case);
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

/* ---- the draft's router ---- */

/* One launch of the draft's router: 128 experts of 3 picks. */
struct draft_case {
	struct linear_range step, scores, bias, offsets, affinities, jobs,
		weights;
	struct linear_routed_experts_description experts;
	struct router_description description;
	struct kernel_launch launch;
	uint32_t row_count;
	uint32_t live_rows;
	float scores_given[MOST_ROWS][DRAFT_EXPERTS];
	float bias_given[DRAFT_EXPERTS];
	float affinities_back[MOST_ROWS][DRAFT_EXPERTS];
	struct expert_job_record jobs_back[MOST_DRAFT_JOBS];
	uint32_t experts_expected[MOST_ROWS][DRAFT_PICKS];
	float weights_expected[MOST_ROWS][DRAFT_PICKS];
	struct expert_job_record jobs_expected[MOST_DRAFT_JOBS];
};

/* Lays a draft case out (the model's case's table, its entries from 128 on zero), prepares it, runs it and takes back
 * what it wrote. */
static bool draft_case_run(struct test_linear_card *card, struct draft_case *draft)
{
    const uint64_t rows_byte_count = (uint64_t)draft->row_count * DRAFT_ROW_BYTES;
    const uint64_t jobs_byte_count =
	    (uint64_t)draft->row_count * DRAFT_PICKS * EXPERT_JOB_BYTES;
    const struct step_block step = { STEP_SEQUENCE, draft->live_rows, 0 };
    static unsigned char rubbish[RUBBISH_BYTES];
    memset(rubbish, RUBBISH_BYTE, sizeof(rubbish));
    test_linear_card_clear(card);
    draft->step = test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE);
    draft->scores = test_linear_card_work(card, rows_byte_count);
    draft->bias = test_linear_card_work(card, DRAFT_ROW_BYTES);
    draft->offsets = test_linear_card_work(card, MODEL_EXPERT_OFFSETS_BYTES);
    draft->affinities = test_linear_card_work(card, rows_byte_count);
    draft->jobs = test_linear_card_work(card, jobs_byte_count);
    draft->weights = test_linear_card_weights(card, WEIGHTS_BYTES);
    for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
	    expert_offsets[entry] =
		    entry < DRAFT_EXPERTS * MODEL_EXPERT_MATRICES ?
			    draft->weights.byte_offset +
				    (uint64_t)entry *
					    LINEAR_OFFSET_ALIGNMENT_BYTES :
			    0;
    }
    draft->experts = (struct linear_routed_experts_description){
	    .jobs = draft->jobs,
	    .expert_weights = draft->weights,
	    .gate_up_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT,
			       0 },
	    .down_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT, 0 },
	    .weight_format = WEIGHT_FORMAT_E2M1_SCALED,
	    .job_count = draft->row_count * DRAFT_PICKS,
	    .row_count = draft->row_count,
	    .pairs_per_row = DRAFT_PICKS,
    };
    draft->description =
	    (struct router_description){ .experts = &draft->experts,
					 .expert_offsets = expert_offsets,
					 .step = draft->step,
					 .scores = draft->scores,
					 .bias = draft->bias,
					 .offsets = draft->offsets,
					 .affinities = draft->affinities,
					 .order = LINEAR_ORDER_FIXED,
					 .expert_count = DRAFT_EXPERTS,
					 .pick_count = DRAFT_PICKS };
    return test_linear_card_has_room(card) &&
	   test_linear_card_put(card, draft->step, 0, &step, sizeof(step)) &&
	   test_linear_card_put(card, draft->scores, 0, draft->scores_given,
				rows_byte_count) &&
	   test_linear_card_put(card, draft->bias, 0, draft->bias_given,
				DRAFT_ROW_BYTES) &&
	   test_linear_card_put(card, draft->offsets, 0, expert_offsets,
				MODEL_EXPERT_OFFSETS_BYTES) &&
	   test_linear_card_put(card, draft->affinities, 0, rubbish,
				rows_byte_count) &&
	   test_linear_card_put(card, draft->jobs, 0, rubbish,
				jobs_byte_count) &&
	   test_linear_expect_status(
		   router_select_prepare(&draft->description,
					 card->arena_byte_count,
					 &router_kernels, &draft->launch),
		   STATUS_OK, "the draft's launch is prepared") &&
	   test_linear_card_load(card) &&
	   test_linear_card_run(card, &draft->launch, 1) &&
	   test_linear_card_fetch(card) &&
	   test_linear_card_get(card, draft->affinities, 0,
				draft->affinities_back, rows_byte_count) &&
	   test_linear_card_get(card, draft->jobs, 0, draft->jobs_back,
				jobs_byte_count);
}

/* The draft's affinities, rows behind, job table and ranges only read; the expected table left in the case. */
static bool draft_case_check(struct test_linear_card *card, struct draft_case *draft)
{
    const uint32_t live_rows = draft->live_rows < draft->row_count ? draft->live_rows : draft->row_count;
    uint32_t failing_values = 0, touched_rows = 0;
    unsigned char rubbish_row[DRAFT_ROW_BYTES];
    memset(rubbish_row, RUBBISH_BYTE, sizeof(rubbish_row));
    for (uint32_t row = 0; row < draft->row_count; row++) {
        for (uint32_t expert = 0; expert < DRAFT_EXPERTS && row < live_rows; expert++) {
            failing_values +=
                test_router_affinity_holds(draft->scores_given[row][expert], draft->affinities_back[row][expert])
                    ? 0u
                    : 1u;
        }
        touched_rows +=
            row >= live_rows && !same_bits(draft->affinities_back[row], rubbish_row, sizeof(rubbish_row)) ? 1u : 0u;
        if (row < live_rows) {
            test_router_select_draft(draft->affinities_back[row], draft->bias_given, draft->experts_expected[row],
                                     draft->weights_expected[row]);
        }
    }
    test_router_draft_jobs((const uint32_t(*)[DRAFT_PICKS])draft->experts_expected,
                           (const float(*)[DRAFT_PICKS])draft->weights_expected, live_rows, draft->row_count,
                           expert_offsets, draft->jobs_expected);
    const uint32_t differing =
        test_router_jobs_differ(draft->jobs_back, draft->jobs_expected, draft->row_count * DRAFT_PICKS);
    if (failing_values != 0 || touched_rows != 0 || differing != 0) {
        test_linear_fail("the draft's: %" PRIu32 " affinities off, %" PRIu32 " rows behind the live ones written, %"
                         PRIu32 " entries of the table not the reference's", failing_values, touched_rows, differing);
    }
    return failing_values == 0 && touched_rows == 0 && differing == 0 &&
           test_linear_card_expect_untouched(card, draft->step, "the step block is only read") &&
           test_linear_card_expect_untouched(card, draft->scores, "the scores are only read") &&
           test_linear_card_expect_untouched(card, draft->bias, "the bias is only read") &&
           test_linear_card_expect_untouched(card, draft->offsets, "the table of offsets is only read");
}

/* Every draft case on one card: the model's score kinds, cut to 128 experts. */
static bool draft_run(struct test_linear_card *card)
{
    static struct router_case scores_of;
    static struct draft_case draft;
    static struct expert_job_record expected_before[MOST_DRAFT_JOBS];
    uint32_t case_count = 0, failed_count = 0;
    uint64_t stream = DRAFT_STREAM_BASE;
    for (uint32_t kind = 0; kind < KIND_COUNT; kind++) {
        for (size_t launch = 0; launch < COUNT_OF(launch_rows); launch++) {
            for (uint32_t live_rows = 1; live_rows <= launch_rows[launch] + ROWS_ABOVE_THE_LAUNCH; live_rows++) {
                stream++;
                scores_of.row_count = draft.row_count = launch_rows[launch];
                draft.live_rows = live_rows;
                scores_make(&scores_of, (enum score_kind)kind, stream);
                for (uint32_t row = 0; row < draft.row_count; row++) {
                    memcpy(draft.scores_given[row], scores_of.scores_given[row], DRAFT_ROW_BYTES);
                }
                memcpy(draft.bias_given, scores_of.bias_given, DRAFT_ROW_BYTES);
                test_linear_context("card %" PRIu32 ", the draft's router, %s scores, %" PRIu32 " of %" PRIu32
                                    " rows live", card->card_index, kind_names[kind], live_rows, launch_rows[launch]);
                bool is_passed = draft_case_run(card, &draft) && draft_case_check(card, &draft);
                if (is_passed && kind == PLAIN && live_rows == MOST_ROWS && launch_rows[launch] == MOST_ROWS) {
                    memcpy(expected_before, draft.jobs_expected, sizeof(expected_before));
                    draft.bias_given[draft.experts_expected[0][0]] -= BIAS_LOWERED;
                    is_passed = draft_case_run(card, &draft) &&
                                test_linear_expect(test_router_jobs_differ(draft.jobs_back, expected_before,
                                                                           MOST_DRAFT_JOBS) != 0,
                                                   "a lowered bias: the draft's table differs from the one before") &&
                                draft_case_check(card, &draft);
                }
                case_count++;
                failed_count += is_passed ? 0u : 1u;
            }
        }
    }
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): %" PRIu32 " cases of the draft's router, %" PRIu32
                 " failed\n", card->card_index, card->arithmetic_class, case_count, failed_count);
    return failed_count == 0;
}

/* ---- the replay of a capture ---- */

struct replay_sources {
    struct test_capture *capture;
    struct storage_reader *reader;
    struct pack_record *pack;
    uint32_t block_alignment_bytes;
};

struct capture_point {
    uint32_t layer;
    uint32_t position;
};

struct replay_totals {
    uint64_t row_count;
    uint64_t same_experts;
    uint64_t equal_weights[ORDERS];             /* of the fixed order, of the reference's */
    uint64_t weight_count;
    uint32_t largest_places[ORDERS];
};

/* The rows of a batch whose affinities, biased scores, six experts or weights are not the capture's. */
struct replay_differences {
    uint32_t affinities;
    uint32_t biased;
    uint32_t experts;
    uint32_t weights;
};

static struct capture_point points[MOST_POINTS];

static int point_order(const void *first, const void *second)
{
    const struct capture_point *const a = first, *const b = second;
    return a->layer != b->layer ? (a->layer < b->layer ? -1 : 1)
                                : (a->position != b->position ? (a->position < b->position ? -1 : 1) : 0);
}

/* The number that begins text, of digits only, and where it ends; false for none or one above limit. */
static bool number_read(const char *text, unsigned long limit, unsigned long *number_out, const char **end_out)
{
    char *end = NULL;
    if (!isdigit((unsigned char)text[0])) {
        return false;
    }
    *number_out = strtoul(text, &end, DECIMAL);
    *end_out = end;
    return *number_out <= limit;
}

/* Whether a record is the router's logits at a capture point: layer_L.position_P.router.logits. */
static bool point_of(const char *name, struct capture_point *point_out)
{
    static const char layer_prefix[] = "layer_", position_prefix[] = ".position_", record[] = ".router.logits";
    unsigned long layer = 0, position = 0;
    const char *end = name;
    if (strncmp(name, layer_prefix, sizeof(layer_prefix) - 1u) != 0 ||
	!number_read(name + sizeof(layer_prefix) - 1u, MODEL_LAYERS - 1u,
		     &layer, &end) ||
	strncmp(end, position_prefix, sizeof(position_prefix) - 1u) != 0 ||
	!number_read(end + sizeof(position_prefix) - 1u, UINT32_MAX, &position,
		     &end) ||
	strcmp(end, record) != 0) {
	    return false;
    }
    *point_out = (struct capture_point){(uint32_t)layer, (uint32_t)position};
    return true;
}

/* The capture points of the router, by layer and position. */
static uint32_t points_find(const struct test_capture *capture)
{
    uint32_t point_count = 0;
    for (uint32_t index = 0; index < test_capture_tensor_count(capture); index++) {
        const struct test_capture_tensor *const tensor = test_capture_tensor_at(capture, index);
        struct capture_point point = {0, 0};
        if (tensor != NULL && point_of(tensor->name, &point)) {
            if (point_count == MOST_POINTS) {
                test_linear_fail("the capture holds more than %u points of the router", MOST_POINTS);
                return 0;
            }
            points[point_count++] = point;
        }
    }
    qsort(points, point_count, sizeof(points[0]), point_order);
    return point_count;
}

/* The bias of a layer from the pack: a float32 [384] tensor, read in whole blocks of the reader. */
static bool bias_read(const struct replay_sources *sources, uint32_t layer, float *bias_out)
{
	_Alignas(BACKEND_STAGING_ALIGNMENT_BYTES) static unsigned char
		blocks[READ_CAPACITY];
	char name[NAME_CAPACITY];
	const int name_length =
		snprintf(name, sizeof(name), BIAS_NAME_FORMAT, layer);
	const uint32_t tensor_index =
		name_length > 0 && (size_t)name_length < sizeof(name) ?
			pack_find_tensor(sources->pack, name,
					 (uint32_t)name_length) :
			PACK_NOT_FOUND;
	const uint64_t alignment = sources->block_alignment_bytes;
	if (tensor_index == PACK_NOT_FOUND || alignment == 0) {
		test_linear_fail(
			"the pack holds no %s, or its reader states no block",
			name);
		return false;
	}
	const struct pack_tensor *const tensor =
		&pack_tensors_of(sources->pack)[tensor_index];
	const struct pack_extent *const extent =
		&pack_extents_of(sources->pack)[tensor->extent_index];
	const uint64_t file_byte_offset =
		extent->byte_offset + tensor->byte_offset_in_extent;
	const uint64_t lead_byte_count = file_byte_offset % alignment;
	const uint64_t read_byte_count =
		(lead_byte_count + tensor->byte_count + alignment - 1u) /
		alignment * alignment;
	if (tensor->element_type != PACK_ELEMENT_FLOAT32 ||
	    tensor->dimension_count != 1 || tensor->dimensions[0] != EXPERTS ||
	    tensor->byte_count != MODEL_ROUTER_ROW_BYTES ||
	    read_byte_count > sizeof(blocks)) {
		test_linear_fail("%s is not float32 [%u] in the pack", name,
				 EXPERTS);
		return false;
	}
	if (!test_linear_expect_status(
		    storage_direct.reader_read(sources->reader, 0,
					       file_byte_offset -
						       lead_byte_count,
					       blocks, read_byte_count),
		    STATUS_OK, "the bias is read from the pack")) {
		return false;
	}
	memcpy(bias_out, blocks + lead_byte_count, MODEL_ROUTER_ROW_BYTES);
	return true;
}

/* One record of a capture point: its count elements of a type. */
static bool point_read(const struct test_capture *capture, const struct capture_point *point, const char *record,
                       enum test_capture_type type, uint64_t count, void *destination)
{
    char name[NAME_CAPACITY];
    const int name_length = snprintf(name, sizeof(name), RECORD_FORMAT, point->layer, point->position, record);
    const uint64_t dimensions[1] = {count};
    return test_linear_expect(name_length > 0 &&
				      (size_t)name_length < sizeof(name),
			      "a record's name fits") &&
	   test_linear_expect_status(
		   test_capture_read(capture, name, type, dimensions, 1, 0,
				     count, destination,
				     count * test_capture_element_bytes(type)),
		   STATUS_OK, name);
}

static uint32_t places_apart(float first, float second)
{
    uint32_t first_bits = 0, second_bits = 0;
    memcpy(&first_bits, &first, sizeof(first_bits));
    memcpy(&second_bits, &second, sizeof(second_bits));
    return first_bits > second_bits ? first_bits - second_bits : second_bits - first_bits;
}

/* What a run of a batch differs in from the capture, and the distance of its weights; false for a record it cannot
 * read. The card's table holds each row's six as the host forms them from the card's affinities (case_check): so
 * its experts and weights by slot are experts_expected and weights_expected. */
static bool batch_compare(const struct replay_sources *sources, const struct router_case *router_case,
                          const struct capture_point *batch, uint32_t batch_count, bool is_reference,
                          struct replay_totals *totals, struct replay_differences *differences)
{
    float scores_captured[EXPERTS], biased_captured[EXPERTS], biased[EXPERTS], weights_captured[PICKS];
    int64_t experts_captured[PICKS];
    memset(differences, 0, sizeof(*differences));
    for (uint32_t row = 0; row < batch_count; row++) {
        if (!point_read(sources->capture, &batch[row], "scores", TEST_CAPTURE_FLOAT32, EXPERTS, scores_captured) ||
            !point_read(sources->capture, &batch[row], "biased_scores", TEST_CAPTURE_FLOAT32, EXPERTS,
                        biased_captured) ||
            !point_read(sources->capture, &batch[row], "expert_ids", TEST_CAPTURE_INT64, PICKS, experts_captured) ||
            !point_read(sources->capture, &batch[row], "route_weights", TEST_CAPTURE_FLOAT32, PICKS,
                        weights_captured)) {
            return false;
        }
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            biased[expert] = router_case->affinities_back[row][expert] + router_case->bias_given[expert];
        }
        uint32_t same = 0, weights_apart = 0;
        for (uint32_t slot = 0; slot < PICKS; slot++) {
            for (uint32_t pick = 0; pick < PICKS; pick++) {
                if (experts_captured[pick] == (int64_t)router_case->experts_expected[row][slot]) {
                    const uint32_t places =
                        places_apart(weights_captured[pick], router_case->weights_expected[row][slot]);
                    same++;
                    weights_apart += places == 0 ? 0u : 1u;
                    totals->equal_weights[is_reference] += places == 0 ? 1u : 0u;
                    totals->largest_places[is_reference] =
                        places > totals->largest_places[is_reference] ? places : totals->largest_places[is_reference];
                }
            }
        }
        differences->affinities +=
            same_bits(scores_captured, router_case->affinities_back[row], sizeof(scores_captured)) ? 0u : 1u;
        differences->biased += same_bits(biased_captured, biased, sizeof(biased)) ? 0u : 1u;
        differences->experts += same == PICKS ? 0u : 1u;
        differences->weights += weights_apart == 0 ? 0u : 1u;
        totals->row_count += is_reference ? 1u : 0u;
        totals->same_experts += is_reference && same == PICKS ? 1u : 0u;
        totals->weight_count += is_reference ? PICKS : 0u;
    }
    return true;
}

/* One batch of points of one layer, in one order, against the capture: the weights bit-equal in the reference's. */
static bool batch_replay(struct test_linear_card *card, const struct replay_sources *sources,
                         struct router_case *router_case, const struct capture_point *batch, uint32_t batch_count,
                         bool is_reference, struct replay_totals *totals)
{
    struct replay_differences differences;
    if (!case_lay_out(card, router_case, batch_count,
		      is_reference ? LINEAR_ORDER_REFERENCE :
				     LINEAR_ORDER_FIXED) ||
	!test_linear_card_load(card) || !case_run(card, router_case) ||
	!case_check(card, router_case, is_reference) ||
	!batch_compare(sources, router_case, batch, batch_count, is_reference,
		       totals, &differences)) {
	    return false;
    }
    if (differences.affinities != 0 || differences.biased != 0 || differences.experts != 0 ||
        (is_reference && differences.weights != 0)) {
        test_linear_fail("of %" PRIu32 " rows, %" PRIu32 " with affinities, %" PRIu32 " with biased scores, %" PRIu32
                         " with experts and %" PRIu32 " with weights that are not the capture's", batch_count,
                         differences.affinities, differences.biased, differences.experts, differences.weights);
        return false;
    }
    return true;
}

/* Shown failing: the score and the bias of the first expert the capture chose for row 0, lowered. The card still
 * holds to the host reference; the replay must find the affinities of row 0 alone, the biased scores of every row
 * and the experts of row 0 at least apart from the capture's. */
static bool replay_break_check(struct test_linear_card *card, const struct replay_sources *sources,
                               struct router_case *router_case, const struct capture_point *batch,
                               uint32_t batch_count)
{
    struct replay_totals ignored;
    struct replay_differences differences;
    int64_t experts_captured[PICKS];
    memset(&ignored, 0, sizeof(ignored));
    if (!point_read(sources->capture, &batch[0], "expert_ids", TEST_CAPTURE_INT64, PICKS, experts_captured)) {
        return false;
    }
    if (experts_captured[0] < 0 || experts_captured[0] >= (int64_t)EXPERTS) {
        test_linear_fail("the capture's first expert, %" PRId64 ", is none of the layer's", experts_captured[0]);
        return false;
    }
    const uint32_t expert = (uint32_t)experts_captured[0];
    test_linear_context("card %" PRIu32 ", layer %" PRIu32 ": the score and the bias of expert %" PRIu32 " lowered",
                        card->card_index, batch[0].layer, expert);
    router_case->scores_given[0][expert] -= BIAS_LOWERED;
    router_case->bias_given[expert] -= BIAS_LOWERED;
    return case_lay_out(card, router_case, batch_count,
			LINEAR_ORDER_REFERENCE) &&
	   test_linear_card_load(card) && case_run(card, router_case) &&
	   case_check(card, router_case, true) &&
	   batch_compare(sources, router_case, batch, batch_count, true,
			 &ignored, &differences) &&
	   test_linear_expect(
		   differences.affinities == 1 &&
			   differences.biased == batch_count &&
			   differences.experts != 0,
		   "the replay finds the affinities, biased scores and experts apart");
}

/* Every capture point on one card, a layer's points eight at a time. */
static bool replay_run(struct test_linear_card *card, const struct replay_sources *sources, uint32_t point_count)
{
    static struct router_case router_case;
    struct replay_totals totals;
    uint32_t failed_count = 0;
    memset(&totals, 0, sizeof(totals));
    for (uint32_t first = 0; first < point_count;) {
        uint32_t batch_count = 0;
        while (batch_count < MOST_ROWS && first + batch_count < point_count &&
               points[first + batch_count].layer == points[first].layer) {
            batch_count++;
        }
        router_case.row_count = MOST_ROWS;
        test_linear_context("card %" PRIu32 ", layer %" PRIu32 " from position %" PRIu32, card->card_index,
                            points[first].layer, points[first].position);
        bool is_passed = bias_read(sources, points[first].layer, router_case.bias_given);
        for (uint32_t row = 0; row < batch_count && is_passed; row++) {
            is_passed = point_read(sources->capture, &points[first + row], "logits", TEST_CAPTURE_FLOAT32, EXPERTS,
                                   router_case.scores_given[row]);
        }
        for (uint32_t order = 0; order <= 1 && is_passed; order++) {
            is_passed = batch_replay(card, sources, &router_case, &points[first], batch_count, order != 0, &totals);
        }
        if (first == 0 && is_passed) {
            is_passed = replay_break_check(card, sources, &router_case, &points[first], batch_count);
        }
        failed_count += is_passed ? 0u : 1u;
        first += batch_count;
    }
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): replayed %" PRIu64 " rows of %" PRIu32 " points: %" PRIu64
                 " with the reference's six experts; weights bit-equal %" PRIu64 " of %" PRIu64 " (reference order, "
                 "largest %" PRIu32 " places), %" PRIu64 " (fixed order, not held, largest %" PRIu32
                 " places); %" PRIu32 " batches failed\n",
                 card->card_index, card->arithmetic_class, totals.row_count, point_count, totals.same_experts,
                 totals.equal_weights[1], totals.weight_count, totals.largest_places[1], totals.equal_weights[0],
                 totals.largest_places[0], failed_count);
    return failed_count == 0;
}

/* Opens the capture and the pack. */
static bool sources_open(int argument_count, char **arguments, struct replay_sources *sources)
{
	struct storage_description description = { 0, 0, 0 };
	const char *const *const copy_paths =
		(const char *const *)&arguments[ARGUMENTS_BEFORE_THE_COPIES];
	const uint32_t copy_count =
		(uint32_t)(argument_count - ARGUMENTS_BEFORE_THE_COPIES);
	result_status status =
		test_capture_open(arguments[1], &sources->capture);
	status = status == STATUS_OK ?
			 storage_direct.reader_open(copy_paths, copy_count,
						    &sources->reader) :
			 status;
	status = status == STATUS_OK ?
			 storage_direct.reader_describe(sources->reader,
							&description) :
			 status;
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, sources->reader,
					 &sources->pack) :
			 status;
	sources->block_alignment_bytes = description.block_alignment_bytes;
	return test_linear_expect_status(status, STATUS_OK,
					 "the capture and the pack open") &&
	       test_linear_expect(
		       description.block_alignment_bytes != 0 &&
			       BACKEND_STAGING_ALIGNMENT_BYTES %
					       description
						       .block_alignment_bytes ==
				       0,
		       "the reader's blocks divide a staging page");
}

int main(int argument_count, char **arguments)
{
    static struct test_linear_card card;
    struct replay_sources sources = {NULL, NULL, NULL, 0};
    uint32_t card_count = 0, failed_cards = 0, point_count = 0;
    const bool is_replay = argument_count > 1;
    if (argument_count == ARGUMENTS_BEFORE_THE_COPIES ||
	argument_count - ARGUMENTS_BEFORE_THE_COPIES > STORAGE_MAXIMUM_COPIES) {
	    (void)fprintf(
		    stderr,
		    "usage: router_card_test [<capture folder> <pack copy> [<pack copy> ...]]\n");
	    return TEST_WRONG_USAGE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    if (is_replay && sources_open(argument_count, arguments, &sources)) {
        point_count = points_find(sources.capture);
        (void)test_linear_expect(point_count != 0, "the capture holds points of the router");
    }
    if (test_linear_failures() == 0 &&
	test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    struct hyper_kernels_record
			    hyper_kernels; /* the families draft.h fills beside the router's */
		    struct attention_state_kernels state_kernels;
		    struct attention_core_kernels core_kernels;
		    struct draft_kernels_record draft_kernels;
		    bool is_passed =
			    test_linear_card_open(&card, card_index,
						  WORK_CAPACITY,
						  WEIGHTS_CAPACITY) &&
			    test_linear_expect_status(
				    router_kernels_prepare(&backend_cuda_value,
							   card.card,
							   &router_kernels),
				    STATUS_OK,
				    "the router's kernel is prepared") &&
			    test_linear_expect_status(
				    draft_kernels_prepare(
					    &backend_cuda_value, card.card,
					    card.arithmetic_class,
					    &hyper_kernels, &router_kernels,
					    &state_kernels, &core_kernels,
					    &draft_kernels),
				    STATUS_OK,
				    "the draft's kernels are prepared");
		    is_passed = is_passed && synthetic_run(&card) &&
				draft_run(&card);
		    is_passed = is_passed &&
				(point_count == 0 ||
				 replay_run(&card, &sources, point_count));
		    test_linear_card_close(&card);
		    failed_cards += is_passed ? 0u : 1u;
	    }
    }
    pack_close(sources.pack);
    storage_direct.reader_close(sources.reader);
    test_capture_close(sources.capture);
    if (test_linear_failures() != 0 || failed_cards != 0) {
        (void)fprintf(stderr, "router_card_test FAILED: %" PRIu32 " cards, %d failure lines\n", failed_cards,
                      test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("router_card_test passed on %" PRIu32 " cards (%s)\n", card_count,
                 is_replay ? "synthetic cases and the capture replayed" : "synthetic cases; nothing replayed: it "
                                                                          "needs a capture folder and a pack");
    return TEST_PASSED;
}
