/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router_test: source/router.c against router.h and router_records.h, and the host reference of the router
 * (router_reference.h) against values found apart from it. No card: a prepare function is arithmetic on a
 * description, the arena is a number.
 *
 *   - The reference: the affinity at nine scores against values to 25 digits (60-digit decimal arithmetic), at a
 *     not-a-number and the infinities, and what holds against it: 2 last places up but not 5. Rows made so that each
 *     rule of the choice shows, with weights that are exact: the bias steers and the weights come from the
 *     affinities; a not-a-number above all; the lower expert first among equals; +0 above -0. A row whose two orders
 *     of the sum are one place apart (bits computed in float32 apart from this code). A job table of three rows
 *     that share experts, by hand, and what test_router_jobs_differ counts.
 *   - router_kernels_prepare: what it asks of a table (the six kernels in order), its refusals, the status
 *     of a table that fails at each kernel with that kernel named and the handles untouched; on a card in host memory
 *     of class 0 UNSUPPORTED, of class 80 the handles.
 *   - router_select_prepare: for 1 to 8 rows and both orders the launch of an accepted description, every
 *     byte; offsets of the table at both ends of the experts' range, and with a down matrix smaller than gate and up
 *     at the end that only a down matrix may begin at; then every refusal of router.h alone in a description
 *     otherwise accepted: the status, one error line with the words of its rule, the launch untouched. The experts'
 *     nine other ranges are held to the arena, the alignment and no shared byte, never to a size.
 *   - The draft's router (dspark_v1/DESIGN.md C4, H3): kernels_prepare leaves select_draft unprepared; for 1 to 8
 *     rows the launch of the pair (128, 3) by its own kernel, ranges of 128 experts, the table's entries from 128 on
 *     not read; refused: the reference order, its kernel unprepared, 6 pairs a row, ranges a byte short, a pair that
 *     is neither; the reference of 3 picks by hand (the bias steers, a not-a-number first, the lower expert first
 *     among equals, weights exact) and its job table.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <float.h>
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
#include "host_memory_backend.h"
#include "router_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define EXPERTS MODEL_EXPERTS
#define PICKS MODEL_EXPERTS_PER_ROW
#define TABLE_ENTRIES (MODEL_EXPERTS * MODEL_EXPERT_MATRICES)
#define MOST_ROWS LINEAR_MAXIMUM_ROWS
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 2048u
#define CARD_MEMORY_BYTES 1048576u
#define MADE_UP_HANDLE 17u
#define RELATIVE_TOLERANCE 1e-12
#define TWO_PLACES 2u
#define ONE_PLACE_MORE 5u                       /* last places: 4.5 off at least, more than an affinity may be */
#define SMALLEST_NORMAL_EXPONENTIAL_SCORE (-87.3f) /* about ln(FLT_MIN): e^score is a normal float32 above it */

/* The model's experts, from its inference configuration: 2304 by 5120 and back, 4 bits. */
#define INNER_SIZE 2304u
#define HIDDEN_SIZE 5120u
#define E2M1_BITS_PER_WEIGHT 4u
#define EXPERT_MATRIX_BYTES 6266880ull          /* 2304 * 5120 / 2 + 2304 * 5120 / 32 */
#define EXPERT_WEIGHTS_BYTES 7219445760ull      /* 384 experts of three such matrices */
#define SMALL_DOWN_INPUT_SIZE 1152u             /* a down matrix of 5120 by 1152: */
#define SMALL_DOWN_BYTES 3133440ull             /* 5120 * 1152 / 2 + 5120 * 1152 / 32 */
#define LAST_UP_ENTRY (TABLE_ENTRIES - 2u)      /* the last expert's gate, up and down are the last three entries */
#define FIRST_OFFSET 8589934592ull              /* 2^33: every offset of the test is beyond 32 bits */
#define HALF_AN_ALIGNMENT 8u

/* The launch, by hand from router_records.h: one block of 256 threads, a record of 80 bytes; the wide ones' 120. */
#define BLOCK_THREADS 256u
#define RECORD_BYTES 80u
#define WIDE_RECORD_BYTES 120u

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
#define RULE_PAIR "picks a row"
#define DRAFT_EXPERTS MODEL_DRAFT_EXPERTS
#define DRAFT_PICKS MODEL_DRAFT_EXPERTS_PER_ROW
#define DRAFT_HANDLE 29u
#define WEIGHT_OF_ONE_IN_FOUR 0x3EC00000u       /* 1 / 4 * 1.5 = 0.375 */
#define WEIGHT_OF_ONE_IN_TWO 0x3F400000u        /* 1 / 2 * 1.5 = 0.75 */
/* The draft's row by hand: every expert of affinity 1/128 and bias 0 but those named. */
#define DRAFT_OTHER_AFFINITY 0.0078125f
#define DRAFT_TOP_EXPERT 100u                   /* affinity 2, bias 0: biased 2 */
#define DRAFT_TOP_AFFINITY 2.0f
#define DRAFT_TIED_FIRST 7u                     /* affinity 1, bias 1: biased 2, a tie the lower expert wins */
#define DRAFT_TIED_SECOND 50u
#define DRAFT_STEERED_EXPERT 90u                /* affinity 1.5, bias -1: the bias steers it out */
#define DRAFT_STEERED_AFFINITY 1.5f
#define DRAFT_NAN_EXPERT 5u                     /* affinity 0, its bias a not-a-number (the first case) */
#define DRAFT_JOB_ROWS 2u

enum launch_dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };

_Static_assert(sizeof(enum linear_order_code) == sizeof(uint32_t),
	       "an order is set from the bytes of a number");

/* ---- the reference's own values ---- */

struct known_affinity {
    float score;
    double affinity;
};
/* sqrt(softplus(score)) with 60 decimal digits, softplus(score) = score above 20 (the threshold of F.softplus). */
static const struct known_affinity known_affinities[] = {
    {0.0f, 8.3255461115769779922857197e-01},   {21.0f, 4.5825756949558398289923389e+00},
    {20.0f, 4.4721359552300237183430909e+00},  {-20.0f, 4.5399929739090793504464189e-05},
    {-100.0f, 1.9287498479639178205634439e-22}, {3.5f, 1.8787630021566372739982853e+00},
    {-3.5f, 1.7248309561409363999118227e-01},  {19.5f, 4.4158804335487014114391968e+00},
    {-87.5f, 9.9911715682242431609300926e-20},
};

/* A row of the choice: every expert's affinity and bias, the ones a row changes, and what must come of it. */
struct known_expert {
    uint32_t expert;
    float affinity;
    float bias;
};
struct known_row {
    const char *what;
    float affinity;                             /* of every expert the row does not name */
    float bias;
    struct known_expert named[PICKS + PICKS];
    uint32_t named_count;
    uint32_t experts[PICKS];                    /* by slot */
    uint32_t weight_bits[PICKS];                /* the same in both orders */
};
#define WEIGHT_OF_ONE_IN_EIGHT 0x3E400000u      /* 1 / 8 * 1.5 = 0.1875 */
#define WEIGHT_OF_THREE_IN_EIGHT 0x3F100000u    /* 3 / 8 * 1.5 = 0.5625 */
#define WEIGHT_OF_FOUR_IN_EIGHT 0x3F400000u     /* 0.75 */
#define WEIGHT_OF_NEGATIVE_ZERO 0x80000000u
static const struct known_row known_rows[] = {
    {"the bias steers, a not-a-number comes first, the lower expert first among equals", 0.25f, 0.0f,
     {{300, 3.0f, 0.0f}, {7, 1.0f, 1.0f}, {50, 1.0f, 1.0f}, {100, 1.0f, 1.0f}, {200, 1.0f, 1.0f},
      {250, 1.0f, 1.0f}, {260, 1.0f, 1.0f}, {5, 100.0f, -99.0f}},
     8, {7, 50, 100, 200, 250, 300},
     {WEIGHT_OF_ONE_IN_EIGHT, WEIGHT_OF_ONE_IN_EIGHT, WEIGHT_OF_ONE_IN_EIGHT, WEIGHT_OF_ONE_IN_EIGHT,
      WEIGHT_OF_ONE_IN_EIGHT, WEIGHT_OF_THREE_IN_EIGHT}},
    {"+0 above -0", 0.25f, -1.0f,
     {{10, 1.0f, 1.0f}, {20, 1.0f, 1.0f}, {30, 1.0f, 1.0f}, {40, 1.0f, 1.0f}, {90, 4.0f, -4.0f},
      {5, -0.0f, -0.0f}, {6, -0.0f, -0.0f}},
     7, {5, 10, 20, 30, 40, 90},
     {WEIGHT_OF_NEGATIVE_ZERO, WEIGHT_OF_ONE_IN_EIGHT, WEIGHT_OF_ONE_IN_EIGHT, WEIGHT_OF_ONE_IN_EIGHT,
      WEIGHT_OF_ONE_IN_EIGHT, WEIGHT_OF_FOUR_IN_EIGHT}},
};
#define NAN_BIAS_EXPERT 300u                    /* of the first row: its bias is a not-a-number */

/* The row of the two orders: expert 300 of affinity 1 and experts 1 to 5 of 3 * 2^-26, every other 0, no bias. The
 * fixed order adds 5 of them first (1 + 2^-22), the reference's 1 first (1 + 2^-23). Bits from numpy's float32. */
#define TINY_AFFINITY 0x1.8p-25f
#define BIG_EXPERT 300u
static const uint32_t order_row_experts[PICKS] = {1, 2, 3, 4, 5, 300};
struct order_bits {
    uint32_t small;                             /* the weight of experts 1 to 5 */
    uint32_t big;                               /* of expert 300 */
};
static const struct order_bits order_row_bits[] = {{0x338FFFFEu, 0x3FBFFFFDu}, {0x338FFFFFu, 0x3FBFFFFEu}};

/* Three rows that share experts, and their table by hand: experts ascending, the rows of each ascending. */
#define JOB_ROWS 3u
#define JOB_LAUNCH_ROWS 4u
#define JOBS_USED 14u
static const uint32_t job_row_experts[JOB_ROWS][PICKS] = {
    {1, 5, 9, 13, 17, 21}, {5, 9, 30, 31, 32, 33}, {1, 9, 40, 41, 42, 43}};
static const float job_row_weights[JOB_ROWS][PICKS] = {
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
    {20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f}};
struct known_job {
    uint32_t expert;
    uint32_t row_count;
    uint32_t pairs[JOB_ROWS];
    float weights[JOB_ROWS];
};
static const struct known_job known_jobs[JOBS_USED] = {
    {1, 2, {0, 12, 0}, {1.0f, 20.0f, 0.0f}},    {5, 2, {1, 6, 0}, {2.0f, 10.0f, 0.0f}},
    {9, 3, {2, 7, 13}, {3.0f, 11.0f, 21.0f}},   {13, 1, {3, 0, 0}, {4.0f, 0.0f, 0.0f}},
    {17, 1, {4, 0, 0}, {5.0f, 0.0f, 0.0f}},     {21, 1, {5, 0, 0}, {6.0f, 0.0f, 0.0f}},
    {30, 1, {8, 0, 0}, {12.0f, 0.0f, 0.0f}},    {31, 1, {9, 0, 0}, {13.0f, 0.0f, 0.0f}},
    {32, 1, {10, 0, 0}, {14.0f, 0.0f, 0.0f}},   {33, 1, {11, 0, 0}, {15.0f, 0.0f, 0.0f}},
    {40, 1, {14, 0, 0}, {22.0f, 0.0f, 0.0f}},   {41, 1, {15, 0, 0}, {23.0f, 0.0f, 0.0f}},
    {42, 1, {16, 0, 0}, {24.0f, 0.0f, 0.0f}},   {43, 1, {17, 0, 0}, {25.0f, 0.0f, 0.0f}},
};
#define ENTRIES_CHANGED 2u
#define OFFSETS_BASE 4096u                      /* a made-up table: entry i at OFFSETS_BASE + 16 i */

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_router_test_XXXXXX";
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

static uint32_t bits_of(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_of(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* ---- the reference ---- */

static void test_reference_affinities(void)
{
    context_text = "the affinity of the reference";
    for (size_t index = 0; index < COUNT_OF(known_affinities); index++) {
        const struct known_affinity *const known = &known_affinities[index];
        const double value = test_router_affinity(known->score);
        const double difference = value - known->affinity;
        expect(difference <= RELATIVE_TOLERANCE * known->affinity &&
                   -difference <= RELATIVE_TOLERANCE * known->affinity,
               "within 10^-12 of its value");
        /* the float32 nearest the value holds, and so does the one 2 places up (2.5 places off at most) */
        const float nearest = (float)known->affinity;
        expect(test_router_affinity_holds(known->score, nearest) &&
                   test_router_affinity_holds(known->score, float_of(bits_of(nearest) + TWO_PLACES)),
               "the float32 nearest it holds, and the one 2 places up");
        expect(known->score < SMALLEST_NORMAL_EXPONENTIAL_SCORE ||
                   !test_router_affinity_holds(known->score, float_of(bits_of(nearest) + ONE_PLACE_MORE)),
               "5 places up does not hold, where e^score is a normal float32");
    }
    const double of_not_a_number = test_router_affinity(NAN);
    expect(of_not_a_number != of_not_a_number && test_router_affinity_holds(NAN, NAN) &&
               !test_router_affinity_holds(NAN, 1.0f),
           "a not-a-number gives one, and only one holds");
    expect(test_router_affinity(INFINITY) > DBL_MAX && test_router_affinity_holds(INFINITY, INFINITY) &&
               !test_router_affinity_holds(INFINITY, FLT_MAX),
           "+infinity gives +infinity");
    expect(test_router_affinity(-INFINITY) == 0.0 && test_router_affinity(-FLT_MAX) == 0.0 &&
               test_router_affinity_holds(-FLT_MAX, 0.0f) && !test_router_affinity_holds(-FLT_MAX, 1.0f),
           "a score far below 0 gives 0");
}

/* The affinities and the bias of a known row. */
static void known_row_lay_out(const struct known_row *row, float *affinities, float *bias)
{
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        affinities[expert] = row->affinity;
        bias[expert] = row->bias;
    }
    for (uint32_t named = 0; named < row->named_count; named++) {
        affinities[row->named[named].expert] = row->named[named].affinity;
        bias[row->named[named].expert] = row->named[named].bias;
    }
}

static void test_reference_choice(void)
{
    float affinities[EXPERTS], bias[EXPERTS], weights[PICKS];
    uint32_t experts[PICKS];
    for (size_t row_index = 0; row_index < COUNT_OF(known_rows); row_index++) {
        const struct known_row *const row = &known_rows[row_index];
        context_text = row->what;
        known_row_lay_out(row, affinities, bias);
        if (row_index == 0) {
            bias[NAN_BIAS_EXPERT] = NAN;
        }
        for (uint32_t order = 0; order <= 1; order++) {
            test_router_select(affinities, bias, order != 0, experts, weights);
            for (uint32_t slot = 0; slot < PICKS; slot++) {
                expect(experts[slot] == row->experts[slot] && bits_of(weights[slot]) == row->weight_bits[slot],
                       "the experts by slot and their weights, in both orders");
            }
        }
    }
    context_text = "the two orders of the sum";
    memset(affinities, 0, sizeof(affinities));
    memset(bias, 0, sizeof(bias));
    for (uint32_t slot = 0; slot < PICKS; slot++) {
        affinities[order_row_experts[slot]] = order_row_experts[slot] == BIG_EXPERT ? 1.0f : TINY_AFFINITY;
    }
    for (uint32_t order = 0; order <= 1; order++) {
        test_router_select(affinities, bias, order != 0, experts, weights);
        for (uint32_t slot = 0; slot < PICKS; slot++) {
            const bool is_big = experts[slot] == BIG_EXPERT;
            expect(experts[slot] == order_row_experts[slot] &&
                       bits_of(weights[slot]) == (is_big ? order_row_bits[order].big : order_row_bits[order].small),
                   "the fixed order adds the small ones first, the reference's the large one; a place apart");
        }
    }
}

static void test_reference_jobs(void)
{
	struct expert_job_record jobs[JOB_LAUNCH_ROWS * PICKS],
		expected[JOB_LAUNCH_ROWS * PICKS];
	uint64_t offsets[TABLE_ENTRIES];
	context_text = "the job table of three rows of a launch of four";
	for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
		offsets[entry] =
			OFFSETS_BASE +
			(uint64_t)entry * LINEAR_OFFSET_ALIGNMENT_BYTES;
	}
    memset(expected, 0, sizeof(expected));
    for (uint32_t job = 0; job < JOBS_USED; job++) {
        const struct known_job *const known = &known_jobs[job];
	const uint64_t *const expert_offsets =
		&offsets[(size_t)known->expert * MODEL_EXPERT_MATRICES];
	expected[job].gate_offset = expert_offsets[0];
        expected[job].up_offset = expert_offsets[1];
	expected[job].down_offset = expert_offsets[MODEL_EXPERT_MATRICES - 1u];
	expected[job].row_count = known->row_count;
        expected[job].expert = known->expert;
        for (uint32_t column = 0; column < known->row_count; column++) {
            expected[job].pair_index[column] = known->pairs[column];
            expected[job].route_weight[column] = known->weights[column];
        }
    }
    memset(jobs, UNTOUCHED_BYTE, sizeof(jobs));
    test_router_jobs(job_row_experts, job_row_weights, JOB_ROWS, JOB_LAUNCH_ROWS, offsets, jobs);
    expect(test_router_jobs_differ(jobs, expected, JOB_LAUNCH_ROWS * PICKS) == 0, "every member as by hand");
    jobs[JOBS_USED - 1u].route_weight[0] = 0.0f;
    jobs[0].pair_index[1] = 0;
    expect(test_router_jobs_differ(jobs, expected, JOB_LAUNCH_ROWS * PICKS) == ENTRIES_CHANGED,
           "two entries changed, two differ");
    jobs[JOBS_USED - 1u].route_weight[0] = NAN;
    expected[JOBS_USED - 1u].route_weight[0] = -NAN;
    jobs[0].pair_index[1] = expected[0].pair_index[1];
    expect(test_router_jobs_differ(jobs, expected, JOB_LAUNCH_ROWS * PICKS) == 0,
           "two weights that are both a not-a-number do not differ");
}

/* ---- the kernels of a card ---- */

/* The kernels in the order the table is asked for them, the order of struct router_kernels_record. */
enum asked_kernel { SELECT_KERNEL, WIDE_SELECT_KERNEL, WIDE_COUNT_KERNEL, WIDE_PLACE_KERNEL, WIDE_JOBS_KERNEL,
                    WIDE_SMALL_KERNEL, KERNEL_COUNT };
static const struct {
    const char *name;
    uint32_t record_bytes;
} asked_kernels[KERNEL_COUNT] = { { "router_select_entry", RECORD_BYTES },
				  { "router_wide_select", WIDE_RECORD_BYTES },
				  { "router_wide_count", WIDE_RECORD_BYTES },
				  { "router_wide_place", WIDE_RECORD_BYTES },
				  { "router_wide_jobs", WIDE_RECORD_BYTES },
				  { "router_wide_small", WIDE_RECORD_BYTES } };
static uint32_t asked_count = 0;
static uint32_t failing_kernel = KERNEL_COUNT;  /* the one the table answers answer_of_table for; none */
static result_status answer_of_table = STATUS_OK;

MUST_CHECK static result_status
recording_kernel_prepare(struct device_card *card, const char *kernel_file_name,
			 const char *function_name,
			 uint32_t argument_byte_count, uint32_t *kernel_out)
{
    const uint32_t kernel = asked_count % KERNEL_COUNT;
    expect(card != NULL && strcmp(kernel_file_name, "kernel_router") == 0 &&
               strcmp(function_name, asked_kernels[kernel].name) == 0 &&
               argument_byte_count == asked_kernels[kernel].record_bytes,
           "the table is asked for each kernel in order, by its file, its name and the size of its record");
    asked_count++;
    *kernel_out = MADE_UP_HANDLE + kernel;
    return kernel == failing_kernel ? answer_of_table : STATUS_OK;
}

static void test_kernels_prepare(void)
{
	static const struct backend_operations recording_backend = {
		.name = "recording", .kernel_prepare = recording_kernel_prepare
	};
	static const struct backend_operations backend_without_kernels = {
		.name = "no kernels"
	};
	struct router_kernels_record kernels;
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	context_text = "the kernels of the router";
	memset(&kernels, UNTOUCHED_BYTE, sizeof(kernels));
	const struct router_kernels_record untouched = kernels;
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	const struct {
		const struct backend_operations *backend;
		struct device_card *card;
		struct router_kernels_record *kernels;
	} missing[] = { { NULL, card, &kernels },
			{ &backend_without_kernels, card, &kernels },
			{ &recording_backend, NULL, &kernels },
			{ &recording_backend, card, NULL } };
    for (size_t index = 0; index < COUNT_OF(missing); index++) {
        capture_begin();
	expect(router_kernels_prepare(
		       missing[index].backend, missing[index].card,
		       missing[index].kernels) == STATUS_INVALID_ARGUMENT,
	       "a missing table, kernel_prepare, card or place for the handle is refused");
	capture_end(logged, sizeof(logged));
        expect_one_error_line_with(logged, RULE_MISSING, "the refusal");
    }
    answer_of_table = STATUS_DEVICE_ERROR;
    for (failing_kernel = 0; failing_kernel < KERNEL_COUNT; failing_kernel++) {
        asked_count = 0;
        capture_begin();
	expect(router_kernels_prepare(&recording_backend, card, &kernels) ==
			       STATUS_DEVICE_ERROR &&
		       asked_count == failing_kernel + 1u,
	       "the status of a table that fails comes back as it is, and no kernel after it is asked for");
	capture_end(logged, sizeof(logged));
        expect_one_error_line_with(logged, asked_kernels[failing_kernel].name, "the kernel that fails is named");
    }
    capture_begin();
    expect(router_kernels_prepare(&test_host_memory_backend, card, &kernels) ==
		   STATUS_UNSUPPORTED,
	   "a card without a build of the kernels");
    capture_end(logged, sizeof(logged));
    expect(memcmp(&kernels, &untouched, sizeof(kernels)) == 0, "the handles are untouched");
    asked_count = 0;
    expect(router_kernels_prepare(&recording_backend, card, &kernels) ==
			   STATUS_OK &&
		   kernels.select == MADE_UP_HANDLE + SELECT_KERNEL &&
		   kernels.wide_select == MADE_UP_HANDLE + WIDE_SELECT_KERNEL &&
		   kernels.wide_count == MADE_UP_HANDLE + WIDE_COUNT_KERNEL &&
		   kernels.wide_place == MADE_UP_HANDLE + WIDE_PLACE_KERNEL &&
		   kernels.wide_jobs == MADE_UP_HANDLE + WIDE_JOBS_KERNEL &&
		   kernels.wide_small == MADE_UP_HANDLE + WIDE_SMALL_KERNEL &&
		   asked_count == KERNEL_COUNT &&
		   kernels.select_draft == BACKEND_NO_KERNEL,
	   "the table's handles, in the order of the kernels; the draft's left unprepared");
    test_host_memory_backend.card_close(card, 0);
    test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
    test_host_memory_backend_set_arithmetic_class(0, ARITHMETIC_CLASS_SM_80);
    expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES, &card) ==
			   STATUS_OK &&
		   router_kernels_prepare(&test_host_memory_backend, card,
					  &kernels) == STATUS_OK &&
		   kernels.select != BACKEND_NO_KERNEL &&
		   kernels.wide_select != BACKEND_NO_KERNEL &&
		   kernels.wide_count != BACKEND_NO_KERNEL &&
		   kernels.wide_place != BACKEND_NO_KERNEL &&
		   kernels.wide_jobs != BACKEND_NO_KERNEL &&
		   kernels.wide_small != BACKEND_NO_KERNEL,
	   "a card of class 80 in host memory prepares the kernels the program's modules hold");
    test_host_memory_backend.card_close(card, 0);
}

/* ---- the launch ---- */

/* An accepted description: every range one behind another from FIRST_OFFSET on, the experts' matrices first, the
 * experts' arrays the router only holds apart last (sized as the model's), the arena ending with the last range; the
 * table every matrix of every expert one behind another. */
struct test_case {
	struct linear_routed_experts_description experts;
	struct router_description description;
	struct router_kernels_record kernels;
	uint64_t offsets[TABLE_ENTRIES];
	uint64_t arena_byte_count;
};
enum case_range { WEIGHTS_RANGE, JOBS_RANGE, STEP_RANGE, SCORES_RANGE, BIAS_RANGE, OFFSETS_RANGE, AFFINITIES_RANGE,
                  CODES_RANGE, EXPONENTS_RANGE, OPERANDS_RANGE, HIDDEN_RANGE, HIDDEN_CODES_RANGE,
                  HIDDEN_EXPONENTS_RANGE, HIDDEN_OPERANDS_RANGE, PAIR_OUTPUT_RANGE, ROW_OUTPUT_RANGE, RANGE_COUNT };

static struct linear_range *range_of(struct test_case *test_case,
				     enum case_range range)
{
	struct linear_range *const ranges[RANGE_COUNT] = {
		&test_case->experts.expert_weights,
		&test_case->experts.jobs,
		&test_case->description.step,
		&test_case->description.scores,
		&test_case->description.bias,
		&test_case->description.offsets,
		&test_case->description.affinities,
		&test_case->experts.codes,
		&test_case->experts.exponents,
		&test_case->experts.operands,
		&test_case->experts.hidden,
		&test_case->experts.hidden_codes,
		&test_case->experts.hidden_exponents,
		&test_case->experts.hidden_operands,
		&test_case->experts.pair_output,
		&test_case->experts.row_output
	};
	return ranges[range];
}

static void case_make(struct test_case *test_case, uint32_t row_count,
		      enum linear_order_code order)
{
    const uint64_t pair_count = (uint64_t)row_count * PICKS;
    const uint64_t byte_counts[RANGE_COUNT] = {
	    EXPERT_WEIGHTS_BYTES,
	    (uint64_t)row_count * PICKS * EXPERT_JOB_BYTES,
	    STEP_BLOCK_BYTES_VALUE,
	    (uint64_t)row_count * MODEL_ROUTER_ROW_BYTES,
	    MODEL_ROUTER_ROW_BYTES,
	    MODEL_EXPERT_OFFSETS_BYTES,
	    (uint64_t)row_count * MODEL_ROUTER_ROW_BYTES,
	    linear_codes_byte_count(row_count, HIDDEN_SIZE),
	    linear_exponents_byte_count(row_count, HIDDEN_SIZE),
	    linear_values_byte_count(row_count, HIDDEN_SIZE),
	    linear_values_byte_count(pair_count, INNER_SIZE),
	    linear_codes_byte_count(pair_count, INNER_SIZE),
	    linear_exponents_byte_count(pair_count, INNER_SIZE),
	    linear_values_byte_count(pair_count, INNER_SIZE),
	    linear_values_byte_count(pair_count, HIDDEN_SIZE),
	    linear_sums_byte_count(row_count, HIDDEN_SIZE)
    };
    uint64_t cursor = FIRST_OFFSET;
    memset(test_case, 0, sizeof(*test_case));
    for (uint32_t range = 0; range < RANGE_COUNT; range++) {
	    *range_of(test_case, (enum case_range)range) =
		    (struct linear_range){ cursor, byte_counts[range] };
	    cursor += byte_counts[range];
    }
    test_case->arena_byte_count = cursor;
    test_case->experts.gate_up_shape =
	    (struct matrix_shape_record){ INNER_SIZE, HIDDEN_SIZE,
					  E2M1_BITS_PER_WEIGHT, 0 };
    test_case->experts.down_shape =
	    (struct matrix_shape_record){ HIDDEN_SIZE, INNER_SIZE,
					  E2M1_BITS_PER_WEIGHT, 0 };
    test_case->experts.weight_format = WEIGHT_FORMAT_E2M1_SCALED;
    test_case->experts.row_count = row_count;
    test_case->experts.pairs_per_row = PICKS;
    test_case->experts.job_count = row_count * PICKS;
    for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
        test_case->offsets[entry] = FIRST_OFFSET + (uint64_t)entry * EXPERT_MATRIX_BYTES;
    }
    test_case->description.experts = &test_case->experts;
    test_case->description.expert_offsets = test_case->offsets;
    test_case->description.order = order;
    test_case->description.expert_count = EXPERTS;
    test_case->description.pick_count = PICKS;
    test_case->kernels.select = MADE_UP_HANDLE;
    test_case->kernels.select_draft = DRAFT_HANDLE;
}

/* The draft's router: case_make's case with 128 experts of 3 picks; scores, bias, offsets and affinities of 128. */
static void draft_case_make(struct test_case *test_case, uint32_t row_count)
{
	case_make(test_case, row_count, LINEAR_ORDER_FIXED);
	test_case->description.expert_count = DRAFT_EXPERTS;
	test_case->description.pick_count = DRAFT_PICKS;
	test_case->experts.pairs_per_row = DRAFT_PICKS;
	test_case->experts.job_count = row_count * DRAFT_PICKS;
	test_case->experts.jobs.byte_count =
		(uint64_t)row_count * DRAFT_PICKS * EXPERT_JOB_BYTES;
	test_case->description.scores.byte_count =
		(uint64_t)row_count * DRAFT_EXPERTS * sizeof(float);
	test_case->description.bias.byte_count =
		(uint64_t)DRAFT_EXPERTS * sizeof(float);
	test_case->description.offsets.byte_count = (uint64_t)DRAFT_EXPERTS *
						    MODEL_EXPERT_MATRICES *
						    sizeof(uint64_t);
	test_case->description.affinities.byte_count =
		(uint64_t)row_count * DRAFT_EXPERTS * sizeof(float);
	for (uint32_t entry = DRAFT_EXPERTS * MODEL_EXPERT_MATRICES;
	     entry < TABLE_ENTRIES; entry++) {
		test_case->offsets[entry] =
			0; /* the draft's table: entries from 128 on zero, never read */
	}
}

MUST_CHECK static result_status case_prepare(const struct test_case *test_case,
					     struct kernel_launch *launch)
{
	return router_select_prepare(&test_case->description,
				     test_case->arena_byte_count,
				     &test_case->kernels, launch);
}

static void test_launches(void)
{
	static const unsigned char zero_bytes[LAUNCH_ARGUMENT_BYTES] = { 0 };
	static struct test_case test_case;
	struct kernel_launch launch;
	struct router_select_arguments record;
	context_text = "an accepted description";
	for (uint32_t row_count = 1; row_count <= MOST_ROWS; row_count++) {
		for (uint32_t order = 0; order <= 1; order++) {
			case_make(&test_case, row_count,
				  order != 0 ? LINEAR_ORDER_REFERENCE :
					       LINEAR_ORDER_FIXED);
			memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
			expect(case_prepare(&test_case, &launch) == STATUS_OK,
			       "it is accepted");
			memcpy(&record, launch.argument, sizeof(record));
			expect(launch.kernel == MADE_UP_HANDLE &&
				       launch.grid[DIMENSION_X] == 1 &&
				       launch.grid[DIMENSION_Y] == 1 &&
				       launch.grid[DIMENSION_Z] == 1 &&
				       launch.block[DIMENSION_X] ==
					       BLOCK_THREADS &&
				       launch.block[DIMENSION_Y] == 1 &&
				       launch.block[DIMENSION_Z] == 1 &&
				       launch.dynamic_shared_byte_count == 0 &&
				       launch.argument_byte_count ==
					       RECORD_BYTES &&
				       launch.branch == LAUNCH_CHAIN &&
				       memcmp(launch.argument + RECORD_BYTES,
					      zero_bytes,
					      sizeof(zero_bytes) -
						      RECORD_BYTES) == 0,
			       "one block of 256 threads, no shared memory requested, the record and zeros behind it");
			expect(record.step_offset == test_case.description.step
							     .byte_offset &&
				       record.scores_offset ==
					       test_case.description.scores
						       .byte_offset &&
				       record.bias_offset ==
					       test_case.description.bias
						       .byte_offset &&
				       record.expert_offsets_offset ==
					       test_case.description.offsets
						       .byte_offset &&
				       record.affinities_offset ==
					       test_case.description.affinities
						       .byte_offset &&
				       record.jobs_offset ==
					       test_case.experts.jobs
						       .byte_offset &&
				       record.weights_offset == FIRST_OFFSET &&
				       record.gate_up_last_offset ==
					       FIRST_OFFSET +
						       EXPERT_WEIGHTS_BYTES -
						       EXPERT_MATRIX_BYTES &&
				       record.down_last_offset ==
					       record.gate_up_last_offset &&
				       record.row_count == row_count &&
				       record.reference_order == order,
			       "the record names every range, the last offsets of the matrices, the rows and the order");
		}
    }
    context_text = "the table at the ends of the experts' range";
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.offsets[0] = test_case.experts.expert_weights.byte_offset;
    test_case.offsets[TABLE_ENTRIES - 1u] =
        test_case.experts.expert_weights.byte_offset + EXPERT_WEIGHTS_BYTES - EXPERT_MATRIX_BYTES;
    expect(case_prepare(&test_case, &launch) == STATUS_OK,
	   "a matrix at either end is accepted");
    context_text = "a down matrix smaller than gate and up, at the end of the experts' range";
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.experts.down_shape.input_size = SMALL_DOWN_INPUT_SIZE;
    test_case.offsets[TABLE_ENTRIES - 1u] =
        test_case.experts.expert_weights.byte_offset + EXPERT_WEIGHTS_BYTES - SMALL_DOWN_BYTES;
    expect(case_prepare(&test_case, &launch) == STATUS_OK, "it is accepted");
    memcpy(&record, launch.argument, sizeof(record));
    expect(record.down_last_offset == test_case.offsets[TABLE_ENTRIES - 1u] &&
               record.gate_up_last_offset == FIRST_OFFSET + EXPERT_WEIGHTS_BYTES - EXPERT_MATRIX_BYTES,
           "each kind of matrix has its own last offset");
}

/* One refusal: the status, one error line with the words of its rule, the launch untouched. */
static void expect_refusal(const struct test_case *test_case, const char *words, const char *what)
{
	struct kernel_launch launch, untouched;
	char logged[CAPTURE_CAPACITY];
	context_text = what;
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	untouched = launch;
	capture_begin();
	expect(case_prepare(test_case, &launch) == STATUS_INVALID_ARGUMENT,
	       "it is refused");
	capture_end(logged, sizeof(logged));
	expect_one_error_line_with(logged, words, "the refusal");
	expect(memcmp(&launch, &untouched, sizeof(launch)) == 0,
	       "the launch is untouched");
}

static void test_refusals(void)
{
    static struct test_case test_case;
    struct kernel_launch launch;
    char logged[CAPTURE_CAPACITY];
    const char *const range_names[RANGE_COUNT] = {
        "the experts' range", "the job table", "the step block", "the router scores", "the bias", "the expert offsets",
        "the affinities", "the rows' codes", "the rows' exponents", "the rows' operands", "the hidden values",
        "the hidden codes", "the hidden exponents", "the hidden operands", "the pair outputs", "the row outputs"};
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    context_text = "missing arguments";
    capture_begin();
    expect(router_select_prepare(NULL, test_case.arena_byte_count,
				 &test_case.kernels,
				 &launch) == STATUS_INVALID_ARGUMENT &&
		   router_select_prepare(&test_case.description,
					 test_case.arena_byte_count, NULL,
					 &launch) == STATUS_INVALID_ARGUMENT &&
		   router_select_prepare(
			   &test_case.description, test_case.arena_byte_count,
			   &test_case.kernels, NULL) == STATUS_INVALID_ARGUMENT,
	   "no description, kernels or launch");
    capture_end(logged, sizeof(logged));
    test_case.description.experts = NULL;
    expect_refusal(&test_case, RULE_MISSING, "no routed experts");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.description.expert_offsets = NULL;
    expect_refusal(&test_case, RULE_MISSING, "no table of offsets");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.kernels.select = BACKEND_NO_KERNEL;
    expect_refusal(&test_case, RULE_KERNEL, "a kernel that is not prepared");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.description.expert_count = DRAFT_EXPERTS;
    expect_refusal(&test_case, RULE_PAIR, "an expert count and pick count that are neither pair");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.experts.reserved_zero = 1;
    expect_refusal(&test_case, RULE_RESERVED, "a reserved field of the routed experts");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    const uint32_t neither_order = LINEAR_ORDER_REFERENCE + 1u;
    memcpy(&test_case.description.order, &neither_order, sizeof(neither_order));
    expect_refusal(&test_case, RULE_COUNTS, "an order that is neither");
    static const struct {
        uint32_t row_count;
        uint32_t pairs_per_row;
        uint32_t job_count;
    } wrong_counts[] = {{0, PICKS, 0}, {MOST_ROWS + 1u, PICKS, (MOST_ROWS + 1u) * PICKS},
                        {MOST_ROWS, PICKS - 1u, MOST_ROWS * PICKS}, {MOST_ROWS, PICKS, MOST_ROWS * PICKS - 1u}};
    for (size_t index = 0; index < COUNT_OF(wrong_counts); index++) {
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    test_case.experts.row_count = wrong_counts[index].row_count;
	    test_case.experts.pairs_per_row = wrong_counts[index].pairs_per_row;
	    test_case.experts.job_count = wrong_counts[index].job_count;
	    expect_refusal(&test_case, RULE_COUNTS,
			   "rows, pairs or jobs the router does not write");
    }
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.experts.weight_format = WEIGHT_FORMAT_E4M3_SCALED;
    expect_refusal(&test_case, RULE_COUNTS, "matrices of 8 bits");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.experts.gate_up_shape.bits_per_weight = E2M1_BITS_PER_WEIGHT - 1u;
    expect_refusal(&test_case, RULE_COUNTS, "a shape the format does not store (gate and up)");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.experts.down_shape.output_size = HIDDEN_SIZE - 1u;
    expect_refusal(&test_case, RULE_COUNTS, "a shape the format does not store (down)");
    for (uint32_t range = 0; range < RANGE_COUNT; range++) {
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    range_of(&test_case, (enum case_range)range)->byte_offset +=
		    HALF_AN_ALIGNMENT;
	    expect_refusal(&test_case, RULE_ALIGNED, range_names[range]);
	    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
	    range_of(&test_case, (enum case_range)range)->byte_count -= 1u;
	    if (range >= CODES_RANGE) {
		    context_text = range_names[range];
		    expect(case_prepare(&test_case, &launch) == STATUS_OK,
			   "a byte short is the experts' to refuse");
	    } else {
		    /* the experts' range holds the larger matrix and more: a byte short, its last matrix leaves it */
		    expect_refusal(&test_case,
				   range == WEIGHTS_RANGE ? RULE_TABLE :
							    RULE_TOUCHED,
				   range_names[range]);
	    }
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
    test_case.description.step.byte_offset = test_case.experts.pair_output.byte_offset;
    expect_refusal(&test_case, RULE_SHARED, "the step block on the pair outputs, which the experts write");
    case_make(&test_case, MOST_ROWS, LINEAR_ORDER_FIXED);
    test_case.experts.hidden.byte_offset = test_case.description.affinities.byte_offset;
    expect_refusal(&test_case, RULE_SHARED, "the hidden values on the affinities");
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
    test_case.experts.down_shape.input_size = SMALL_DOWN_INPUT_SIZE;
    test_case.offsets[LAST_UP_ENTRY] =
        test_case.experts.expert_weights.byte_offset + EXPERT_WEIGHTS_BYTES - SMALL_DOWN_BYTES;
    expect_refusal(&test_case, RULE_TABLE, "an up matrix where only the smaller down matrix fits");
}

/* ---- the draft's router ---- */

static const uint32_t draft_expected_experts[DRAFT_PICKS] = {DRAFT_TIED_FIRST, DRAFT_TIED_SECOND, DRAFT_TOP_EXPERT};
static const uint32_t draft_expected_weight_bits[DRAFT_PICKS] = {WEIGHT_OF_ONE_IN_FOUR, WEIGHT_OF_ONE_IN_FOUR,
                                                                 WEIGHT_OF_ONE_IN_TWO};
static const uint32_t draft_job_rows[DRAFT_JOB_ROWS][DRAFT_PICKS] = {{7, 50, 100}, {7, 60, 127}};
static const float draft_job_weights[DRAFT_JOB_ROWS][DRAFT_PICKS] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
static const struct known_job draft_jobs[] = {{7, 2, {0, 3, 0}, {1.0f, 4.0f, 0.0f}}, {50, 1, {1, 0, 0}, {2.0f}},
                                              {60, 1, {4, 0, 0}, {5.0f}}, {100, 1, {2, 0, 0}, {3.0f}},
                                              {127, 1, {5, 0, 0}, {6.0f}}};

static void test_draft(void)
{
	static const unsigned char zero_bytes[LAUNCH_ARGUMENT_BYTES] = { 0 };
	static struct test_case test_case;
	struct kernel_launch launch;
	struct router_select_arguments record;
	context_text = "the draft's router, accepted";
	for (uint32_t row_count = 1; row_count <= MOST_ROWS; row_count++) {
		draft_case_make(&test_case, row_count);
		memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
		expect(case_prepare(&test_case, &launch) == STATUS_OK,
		       "it is accepted");
		memcpy(&record, launch.argument, sizeof(record));
		expect(launch.kernel == DRAFT_HANDLE &&
			       launch.grid[DIMENSION_X] == 1 &&
			       launch.grid[DIMENSION_Y] == 1 &&
			       launch.block[DIMENSION_X] == BLOCK_THREADS &&
			       launch.argument_byte_count == RECORD_BYTES &&
			       memcmp(launch.argument + RECORD_BYTES,
				      zero_bytes,
				      sizeof(zero_bytes) - RECORD_BYTES) == 0 &&
			       record.scores_offset ==
				       test_case.description.scores.byte_offset &&
			       record.jobs_offset ==
				       test_case.experts.jobs.byte_offset &&
			       record.row_count == row_count &&
			       record.reference_order == 0,
		       "its own kernel, the select's record in the fixed order");
	}
    context_text = "the draft's router, refused";
    draft_case_make(&test_case, MOST_ROWS);
    test_case.description.order = LINEAR_ORDER_REFERENCE;
    expect_refusal(&test_case, RULE_COUNTS, "the reference order");
    draft_case_make(&test_case, MOST_ROWS);
    test_case.kernels.select_draft = BACKEND_NO_KERNEL;
    expect_refusal(&test_case, RULE_KERNEL, "its kernel not prepared");
    draft_case_make(&test_case, MOST_ROWS);
    test_case.experts.pairs_per_row = PICKS;
    expect_refusal(&test_case, RULE_COUNTS, "the model's six pairs a row");
    draft_case_make(&test_case, MOST_ROWS);
    test_case.description.pick_count = PICKS;
    expect_refusal(&test_case, RULE_PAIR, "128 experts of six picks");
    static const enum case_range draft_ranges[] = {SCORES_RANGE, BIAS_RANGE, OFFSETS_RANGE, AFFINITIES_RANGE,
                                                   JOBS_RANGE};
    for (size_t index = 0; index < COUNT_OF(draft_ranges); index++) {
        draft_case_make(&test_case, MOST_ROWS);
        range_of(&test_case, draft_ranges[index])->byte_count -= 1u;
        expect_refusal(&test_case, RULE_TOUCHED, "a range of the draft's sizes a byte short");
    }
    draft_case_make(&test_case, MOST_ROWS);
    test_case.offsets[DRAFT_EXPERTS * MODEL_EXPERT_MATRICES - 1u] +=
	    HALF_AN_ALIGNMENT;
    expect_refusal(&test_case, RULE_TABLE, "the last draft expert's matrix off the alignment");

    context_text = "the draft's reference by hand";
    float affinities[DRAFT_EXPERTS], bias[DRAFT_EXPERTS], weights[DRAFT_PICKS];
    uint32_t experts[DRAFT_PICKS];
    for (uint32_t expert = 0; expert < DRAFT_EXPERTS; expert++) {
        affinities[expert] = DRAFT_OTHER_AFFINITY;
        bias[expert] = 0.0f;
    }
    affinities[DRAFT_TOP_EXPERT] = DRAFT_TOP_AFFINITY;
    affinities[DRAFT_TIED_FIRST] = affinities[DRAFT_TIED_SECOND] = 1.0f;
    bias[DRAFT_TIED_FIRST] = bias[DRAFT_TIED_SECOND] = 1.0f;
    affinities[DRAFT_STEERED_EXPERT] = DRAFT_STEERED_AFFINITY;
    bias[DRAFT_STEERED_EXPERT] = -1.0f;
    affinities[DRAFT_NAN_EXPERT] = 0.0f;
    bias[DRAFT_NAN_EXPERT] = NAN;
    test_router_select_draft(affinities, bias, experts, weights);
    expect(experts[0] == DRAFT_NAN_EXPERT && experts[1] == DRAFT_TIED_FIRST && experts[DRAFT_PICKS - 1u] ==
               DRAFT_TIED_SECOND && bits_of(weights[0]) == 0 && bits_of(weights[1]) == WEIGHT_OF_ONE_IN_TWO,
           "a not-a-number first, then the lower two of three tied at 2; weights 0, 3/4, 3/4");
    bias[DRAFT_NAN_EXPERT] = 0.0f;
    test_router_select_draft(affinities, bias, experts, weights);
    bool holds = true;
    for (uint32_t slot = 0; slot < DRAFT_PICKS; slot++) {
        holds = holds && experts[slot] == draft_expected_experts[slot] &&
                bits_of(weights[slot]) == draft_expected_weight_bits[slot];
    }
    expect(holds, "the bias steers past the larger affinity of 1.5, three tie at 2: weights 1/4, 1/4, 2/4 of 1.5");
    struct expert_job_record jobs[MOST_ROWS * DRAFT_PICKS],
	    expected[MOST_ROWS * DRAFT_PICKS];
    uint64_t offsets[TABLE_ENTRIES];
    for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
	    offsets[entry] = OFFSETS_BASE +
			     (uint64_t)entry * LINEAR_OFFSET_ALIGNMENT_BYTES;
    }
    memset(expected, 0, sizeof(expected));
    for (size_t job = 0; job < COUNT_OF(draft_jobs); job++) {
	    const uint64_t *const expert_offsets =
		    &offsets[(size_t)draft_jobs[job].expert *
			     MODEL_EXPERT_MATRICES];
	    expected[job].gate_offset = expert_offsets[0];
	    expected[job].up_offset = expert_offsets[1];
	    expected[job].down_offset =
		    expert_offsets[MODEL_EXPERT_MATRICES - 1u];
	    expected[job].row_count = draft_jobs[job].row_count;
	    expected[job].expert = draft_jobs[job].expert;
	    for (uint32_t column = 0; column < draft_jobs[job].row_count;
		 column++) {
		    expected[job].pair_index[column] =
			    draft_jobs[job].pairs[column];
		    expected[job].route_weight[column] =
			    draft_jobs[job].weights[column];
	    }
    }
    memset(jobs, UNTOUCHED_BYTE, sizeof(jobs));
    test_router_draft_jobs(draft_job_rows, draft_job_weights, DRAFT_JOB_ROWS, MOST_ROWS, offsets, jobs);
    expect(test_router_jobs_differ(jobs, expected, MOST_ROWS * DRAFT_PICKS) == 0,
           "the draft's table of two rows of a launch of eight: pairs row * 3 + slot, the rest zero");
}

int main(void)
{
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream could not be made\n");
        return EXIT_FAILURE;
    }
    test_reference_affinities();
    test_reference_choice();
    test_reference_jobs();
    test_kernels_prepare();
    test_launches();
    test_refusals();
    test_draft();
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0 &&
               test_host_memory_backend_open_card_count() == 0,
           "the temporary file is removed and no card is left open");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("router_test passed\n");
    return EXIT_SUCCESS;
}
