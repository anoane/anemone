/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler_card_test: the sampler on the real cards, against the host reference (sampler_reference.h) and, given the
 * capture folder of a stepwise run of the oracle with every row's logits (tools/oracle/stepwise_capture.py
 * --all-logits), against the reference's own logits.
 *
 *     sampler_card_test [--capture <folder>] [--draws <count>] [--break logit|key|cut]
 *
 * It needs the cards free; `make card_tests` runs it without arguments, under compute-sanitizer's memcheck too. Its
 * batch adds `compute-sanitizer --tool racecheck` and `--tool synccheck` of build/profile/sampler_card_test --draws
 * 1024, and build/release/sampler_card_test --capture <folder> --draws 1000000 (plan 3.2's gate). On every card:
 *   - rows: one launch of eight rows of the model's vocabulary, seven of them in the step: greedy rows (plain; a tie
 *     of the largest logit); keyed rows whose largest logit is not finite, which fall back to greedy (+infinity among
 *     not-a-numbers and -infinity; -infinity and a not-a-number alone); keyed rows at temperature 0.6 with top_p 0.95
 *     (not-a-numbers and -infinities among its logits) and below the floor with top_p 1; a row without logits; one
 *     beyond the step. The result's block and every answer and weight are the reference's: a greedy token and its
 *     logit bit for bit; each weight within TEST_SAMPLER_WEIGHT_PLACES of exp in double; a keyed token kept by tau of
 *     a sort of the card's own weights and allowed by the race over them. The answer beyond the step and the weights
 *     of the rows that do not draw come back untouched, and nothing outside the ranges is written (the fixture's
 *     guards, after the launch of eight rows and after those of one).
 *   - each of those seven rows again alone, in a launch of one row: its answer and its weights bit for bit.
 *   - draws: --draws draws (DEFAULT_DRAWS by default; whole batches) of one row at temperature 1 and top_p 0.95, of
 *     DRAW_VOCABULARY synthetic logits (with --capture: the logits of the captured position whose 16 largest lie
 *     closest together), eight a launch and BATCH_LAUNCHES launches a batch, each launch's areas behind their own
 *     guards, at positions 0 up: every weight of the eight copies of the row the same, every token kept and, at the
 *     first RACED_DRAWS positions, allowed by the race of its key; the counts against the kept probabilities by a
 *     chi-square test (Wilson and Hilferty's approximation) of LEAST_CELLS cells or more at p above
 *     CHI_SQUARE_LEAST_P; the first batch, run again, the same tokens.
 *   - with --capture: the greedy answer of every position of logits.all_rows is the token and the logit of
 *     logits.top_ids and logits.top_values at rank 0 (descending, the lowest id first among equals), bit for bit.
 * --break shows the checks of a kind failing: the test then passes only if one of them reported and every other check
 * held. logit raises one logit of the plain greedy row on the card alone (greedy); key gives the reference the seed
 * plus one, and cut top_p 0.9 for 0.95 (keyed).
 */
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/sampler.h"
#include "anemone/sampler_records.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "capture_reader.h"
#include "linear_card_fixture.h"
#include "sampler_reference.h"

#define TEST_TIME_LIMIT_SECONDS 7200            /* the alarm ends the test here, whatever the driver does */
#define WORK_CAPACITY (16ull << 20u)
#define VOCABULARY MODEL_VOCABULARY_VALUE
#define ROWS LINEAR_MAXIMUM_ROWS
#define STEP_ROWS 7u                            /* the eighth row lies beyond the step */
#define SESSIONS 4u
#define SEQUENCE 0x0123456789ABCDEFull
#define UNTOUCHED_BYTE 0x7E                     /* of the weights and the answers no launch may write */
#define SPREAD 16.0                             /* synthetic logits lie in [-16, 16) */
#define HASH_SHIFT 40u                          /* the top 24 bits of a hash */
#define HASH_SCALE 1.1920928955078125e-7        /* 2^-23: 24 bits to [0, 2) */
#define ROW_STREAM 0x5A000000ull
#define DRAW_STREAM 0x5B000000ull
#define TIE_FIRST 4099u
#define TIE_SECOND 77777u
#define NOT_A_NUMBER_STRIDE 7u
#define LOWEST_TOKEN 3u
#define HIGHEST_TOKEN 100003u
#define SEED_CUT 0x243F6A8885A308D3ull
#define SEED_ALL 0x13198A2E03707344ull
#define SEED_DRAWS 0xA4093822299F31D0ull
#define POSITION_CUT 5
#define POSITION_ALL 123456
#define TOP_P 0.95f
#define BROKEN_TOP_P 0.9f
#define COOL_TEMPERATURE 0.6f
#define BELOW_THE_FLOOR \
	1e-7f /* of temperature: SAMPLER_TEMPERATURE_FLOOR is taken */
#define CUT_ODD_STRIDE 97u                      /* a not-a-number and a -infinity among the keyed row's logits */
#define DRAW_VOCABULARY 4096u
#define DEFAULT_DRAWS 65536u
#define RACED_DRAWS 65536u                      /* the draws held to the race; the rest to the kept tokens alone */
#define MOST_DRAWS 100000000ull
#define BATCH_LAUNCHES 128u
#define BATCH_DRAWS ((uint64_t)BATCH_LAUNCHES * ROWS)
#define CHI_SQUARE_LEAST_P 0.001
#define LEAST_EXPECTED 5.0                      /* a cell of the chi-square test expects at least this many */
#define LEAST_CELLS 10u                         /* a test of fewer cells holds the draws to too little */
#define WILSON_HILFERTY_TWO 2.0
#define WILSON_HILFERTY_NINE 9.0
#define NORMAL_QUANTILE 3.090232306167813       /* the standard normal's upper point of CHI_SQUARE_LEAST_P */
#define NEWTON_STEPS 64u
#define HALF 0.5
#define TOP_RANKS 16u                           /* of logits.top_ids and logits.top_values */
#define ALL_ROWS_DIMENSIONS 3u                  /* logits.all_rows: [1, positions, vocabulary] */
#define TOP_DIMENSIONS 2u
#define MESSAGE_CAPACITY 512u
#define REPORTS_SHOWN 8u
#define FAILURES_SHOWN 16u
#define CHECK_FORMAT_INDEX 3
#define CHECK_VALUES_INDEX 4
#define DECIMAL 10
#define OPTION_WORDS 2                          /* a flag and its value */
#define POSITION_AXIS 1u                        /* of logits.all_rows */
#define VOCABULARY_AXIS 2u

enum row_case { ROW_PLAIN, ROW_TIE, ROW_NOT_NUMBERS, ROW_NO_NUMBER, ROW_CUT, ROW_ALL, ROW_NO_LOGITS, ROW_BEYOND };
enum check_kind { KIND_OTHER, KIND_GREEDY, KIND_KEYED };
enum broken { BROKEN_NONE, BROKEN_LOGIT, BROKEN_KEY, BROKEN_CUT };
enum session_slot { SLOT_GREEDY, SLOT_CUT, SLOT_ALL, SLOT_FALLBACK };

static const char *const broken_names[] = {[BROKEN_LOGIT] = "logit", [BROKEN_KEY] = "key", [BROKEN_CUT] = "cut"};
static const enum check_kind broken_kinds[] = {[BROKEN_NONE] = KIND_OTHER, [BROKEN_LOGIT] = KIND_GREEDY,
                                               [BROKEN_KEY] = KIND_KEYED, [BROKEN_CUT] = KIND_KEYED};

/* The settings of each slot, and the slot and position of each row (slot 0 and position 0 unless named). */
static const struct step_session slot_sessions[SESSIONS] = {
	[SLOT_GREEDY] = { 0, 0.0f, 1.0f },
	[SLOT_CUT] = { SEED_CUT, COOL_TEMPERATURE, TOP_P },
	[SLOT_ALL] = { SEED_ALL, BELOW_THE_FLOOR, 1.0f },
	[SLOT_FALLBACK] = { 0, 1.0f, TOP_P }
};
static const uint32_t row_slots[ROWS] = {[ROW_NOT_NUMBERS] = SLOT_FALLBACK, [ROW_NO_NUMBER] = SLOT_FALLBACK,
                                         [ROW_CUT] = SLOT_CUT, [ROW_ALL] = SLOT_ALL};
static const int32_t row_positions[ROWS] = {[ROW_CUT] = POSITION_CUT, [ROW_ALL] = POSITION_ALL};

static enum broken broken = BROKEN_NONE;
static uint64_t reported_count = 0;
static uint64_t failure_lines = 0;

/* A check of a kind: a failure, or, of the kind a break aims at, a report. */
static void check(enum check_kind kind, bool holds, const char *format, ...)
    __attribute__((format(printf, CHECK_FORMAT_INDEX, CHECK_VALUES_INDEX)));
static void check(enum check_kind kind, bool holds, const char *format, ...)
{
    char message[MESSAGE_CAPACITY];
    va_list values;
    if (holds) {
        return;
    }
    va_start(values, format);
    (void)vsnprintf(message, sizeof(message), format, values);
    va_end(values);
    if (broken != BROKEN_NONE && kind == broken_kinds[broken]) {
        reported_count++;
        if (reported_count <= REPORTS_SHOWN) {
            (void)printf("REPORTED: %s\n", message);
        }
    } else if (++failure_lines <= FAILURES_SHOWN) {
        test_linear_fail("%s", message);
    } else {
        test_linear_fail("(more)");
    }
}

/* Whether the host memory of a case was had: a failure line if not. */
static bool memory_was_had(bool was_had)
{
    if (!was_had) {
        test_linear_fail("the host memory of the case cannot be had");
    }
    return was_had;
}

static uint32_t bits_of(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* Synthetic logits: [-SPREAD, SPREAD) by a hash of the stream and the token. */
static void synthetic_logits(float *logits, uint32_t vocabulary_size, uint64_t stream)
{
    for (uint32_t token = 0; token < vocabulary_size; token++) {
        const double unit = (double)(test_linear_hash(stream, token) >> HASH_SHIFT) * HASH_SCALE - 1.0;
        logits[token] = (float)(unit * SPREAD);
    }
}

static bool sampler_prepare(const struct test_linear_card *card,
			    const struct sampler_kernels_record *kernels,
			    const struct sampler_description *description,
			    struct kernel_launch *launch_out)
{
	return test_linear_expect_status(
		sampler_prepare_entry(description, card->arena_byte_count,
				      kernels, launch_out),
		STATUS_OK, "the launch of the sampler is prepared");
}

/* A step area: its block, then its rows. */
static bool put_step(struct test_linear_card *card, struct linear_range area,
		     uint64_t sequence, uint32_t row_count,
		     const struct step_row_record *rows, uint32_t rows_laid)
{
	const struct step_block block = { sequence, row_count, 0 };
	return test_linear_card_put(card, area, 0, &block, sizeof(block)) &&
	       test_linear_card_put(card, area, sizeof(block), rows,
				    (uint64_t)rows_laid * sizeof(*rows));
}

/* Of a result area as last downloaded: its block, and the answer of a row. */
static struct step_result result_of(const struct test_linear_card *card,
				    struct linear_range area)
{
	struct step_result result;
	memset(&result, 0, sizeof(result));
	(void)test_linear_card_get(card, area, 0, &result, sizeof(result));
	return result;
}
static struct step_row_result answer_of(const struct test_linear_card *card,
					struct linear_range area, uint32_t row)
{
	struct step_row_result answer = { STEP_NO_TOKEN, 0.0f };
	(void)test_linear_card_get(
		card, area,
		STEP_RESULT_BYTES + (uint64_t)row * STEP_ROW_RESULT_BYTES,
		&answer, sizeof(answer));
	return answer;
}

/* Downloads one range of the work region into received, at its offset. */
static bool fetch_range(struct test_linear_card *card,
			struct linear_range range)
{
    uint64_t ticket = 0;
    return test_linear_expect_status(backend_cuda_value.download_begin(
					     card->card, range.byte_offset,
					     card->received + range.byte_offset,
					     range.byte_count, &ticket),
				     STATUS_OK, "a range is downloaded") &&
	   test_linear_expect_status(
		   backend_cuda_value.upload_wait(
			   card->card, ticket,
			   TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
		   STATUS_OK, "the download finishes within the limit");
}

/* The bytes of a range as received are those of the image. */
static bool came_back_as_laid(const struct test_linear_card *card,
			      struct linear_range range, uint64_t byte_offset,
			      uint64_t byte_count)
{
    return memcmp(card->received + range.byte_offset + byte_offset, card->image + range.byte_offset + byte_offset,
                  byte_count) == 0;
}

/* ---- the keyed answers ---- */

/* Tau of a row of weights as a card wrote them, of the top_p the reference is given, and the tokens it keeps. */
static uint32_t keep(const float *weights, uint32_t vocabulary_size, float top_p, uint32_t *kept)
{
    uint32_t threshold = 0;
    if (!test_linear_expect(test_sampler_threshold(weights, vocabulary_size,
                                                   broken == BROKEN_CUT && top_p < 1.0f ? BROKEN_TOP_P : top_p,
                                                   &threshold),
                            "the reference sorts the weights")) {
        return 0;
    }
    return test_sampler_kept(weights, vocabulary_size, threshold, kept);
}

/* Each weight within its places of the reference's, of logits whose largest is at greedy. */
static void check_weights(const float *weights, const float *logits, uint32_t vocabulary_size, float temperature,
                          const char *what)
{
    const float largest = logits[test_sampler_greedy(logits, vocabulary_size)];
    uint32_t wrong = 0;
    for (uint32_t token = 0; token < vocabulary_size; token++) {
        wrong += test_sampler_within_places(weights[token], test_sampler_weight(logits[token], largest, temperature),
                                            TEST_SAMPLER_WEIGHT_PLACES)
                     ? 0u
                     : 1u;
    }
    check(KIND_OTHER, wrong == 0, "%s: %" PRIu32 " weights not within %u places of exp in double", what, wrong,
          TEST_SAMPLER_WEIGHT_PLACES);
}

/* A keyed token: kept, and allowed by the race of its key over the kept tokens. */
static bool keyed_token_holds(const float *weights, const uint32_t *kept, uint32_t kept_count, uint64_t seed,
                              int32_t position, int32_t token)
{
    struct test_sampler_race race;
    test_sampler_race(weights, kept, kept_count, broken == BROKEN_KEY ? seed + 1 : seed, position, &race);
    return token >= 0 && test_sampler_race_allows(&race, (uint32_t)token);
}

/* ---- rows ---- */

struct rows_layout {
	struct linear_range logits;
	struct linear_range weights;
	struct linear_range sessions;
	struct linear_range step;
	struct linear_range result;
	struct linear_range alone_step;
	struct linear_range alone_weights;
	struct linear_range alone_result;
};

static bool lay_out_rows(struct test_linear_card *card,
			 struct rows_layout *layout, float *logits,
			 struct step_row_record *rows)
{
    const uint64_t row_bytes = (uint64_t)VOCABULARY * sizeof(float);
    test_linear_card_clear(card);
    layout->logits = test_linear_card_work(card, ROWS * row_bytes);
    layout->weights = test_linear_card_work(card, ROWS * row_bytes);
    layout->sessions = test_linear_card_work(card, sizeof(slot_sessions));
    layout->step = test_linear_card_work(
	    card, STEP_BLOCK_BYTES_VALUE + ROWS * STEP_ROW_BYTES_VALUE);
    layout->result = test_linear_card_work(
	    card, STEP_RESULT_BYTES + ROWS * STEP_ROW_RESULT_BYTES);
    layout->alone_step = test_linear_card_work(
	    card, STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE);
    layout->alone_weights = test_linear_card_work(card, row_bytes);
    layout->alone_result = test_linear_card_work(
	    card, STEP_RESULT_BYTES + STEP_ROW_RESULT_BYTES);
    if (!test_linear_card_has_room(card)) {
        return false;
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        float *const row_logits = logits + (size_t)row * VOCABULARY;
        synthetic_logits(row_logits, VOCABULARY, ROW_STREAM + row);
	rows[row] = (struct step_row_record){ row_positions[row], 0,
					      row_slots[row], STEP_ROW_LOGITS };
    }
    float *const tie = logits + (size_t)ROW_TIE * VOCABULARY;
    tie[TIE_FIRST] = (float)SPREAD;
    tie[TIE_SECOND] = (float)SPREAD;
    float *const odd = logits + (size_t)ROW_NOT_NUMBERS * VOCABULARY;
    for (uint32_t token = 0; token < VOCABULARY; token += NOT_A_NUMBER_STRIDE) {
        odd[token] = NAN;
    }
    odd[LOWEST_TOKEN] = -INFINITY;
    odd[HIGHEST_TOKEN] = INFINITY;
    float *const cut = logits + (size_t)ROW_CUT * VOCABULARY;
    for (uint32_t token = 0; token + 1 < VOCABULARY; token += CUT_ODD_STRIDE) {
        cut[token] = NAN;
        cut[token + 1] = -INFINITY;
    }
    float *const no_number = logits + (size_t)ROW_NO_NUMBER * VOCABULARY;
    for (uint32_t token = 0; token < VOCABULARY; token++) {
        no_number[token] = token == 0 ? NAN : -INFINITY;
    }
    rows[ROW_NO_LOGITS].flags = 0;
    memset(card->image + layout->weights.byte_offset, UNTOUCHED_BYTE, layout->weights.byte_count);
    memset(card->image + layout->result.byte_offset, UNTOUCHED_BYTE, layout->result.byte_count);
    return test_linear_card_put(card, layout->logits, 0, logits, ROWS * row_bytes) &&
           test_linear_card_put(card, layout->sessions, 0, slot_sessions, sizeof(slot_sessions)) &&
           put_step(card, layout->step, SEQUENCE, STEP_ROWS, rows, ROWS);
}

/* The answer of a row against the reference; weights: the row's as the card wrote them. */
static void check_answer(uint32_t row, const float *logits,
			 const float *weights, struct step_row_result answer,
			 uint32_t *kept, const char *what)
{
	const struct step_session session = slot_sessions[row_slots[row]];
	const uint32_t greedy = test_sampler_greedy(logits, VOCABULARY);
	if (row == ROW_NO_LOGITS) {
		check(KIND_OTHER,
		      answer.token_id == STEP_NO_TOKEN &&
			      bits_of(answer.logit) == 0,
		      "%s: row %" PRIu32 " without logits is answered %" PRId32
		      ", not ANEMONE_STEP_NO_TOKEN and +0",
		      what, row, answer.token_id);
	} else if (row != ROW_CUT && row != ROW_ALL) {
		check(KIND_GREEDY,
		      answer.token_id == (int32_t)greedy &&
			      bits_of(answer.logit) == bits_of(logits[greedy]),
		      "%s: row %" PRIu32 ": token %" PRId32
		      " and logit 0x%08" PRIx32 ", greedy %" PRIu32
		      " and 0x%08" PRIx32,
		      what, row, answer.token_id, bits_of(answer.logit), greedy,
		      bits_of(logits[greedy]));
	} else {
		const uint32_t kept_count =
			keep(weights, VOCABULARY, session.top_p, kept);
		const bool in_range = answer.token_id >= 0 &&
				      (uint32_t)answer.token_id < VOCABULARY;
		check(KIND_KEYED,
		      in_range &&
			      keyed_token_holds(
				      weights, kept, kept_count, session.seed,
				      row_positions[row], answer.token_id) &&
			      bits_of(answer.logit) ==
				      bits_of(logits[answer.token_id]),
		      "%s: row %" PRIu32 ": token %" PRId32
		      " is not the race's over the %" PRIu32
		      " kept, or its logit",
		      what, row, answer.token_id, kept_count);
	}
}

static void test_rows(struct test_linear_card *card,
		      const struct sampler_kernels_record *kernels)
{
    const uint64_t row_bytes = (uint64_t)VOCABULARY * sizeof(float);
    float *const logits = malloc(ROWS * row_bytes);
    float *const weights = malloc(row_bytes);
    uint32_t *const kept = malloc((size_t)VOCABULARY * sizeof(uint32_t));
    struct step_row_record rows[ROWS];
    struct rows_layout layout;
    struct kernel_launch launch;
    test_linear_context("card %" PRIu32 ": rows", card->card_index);
    if (!memory_was_had(logits != NULL && weights != NULL && kept != NULL) ||
        !lay_out_rows(card, &layout, logits, rows)) {
        free(kept);
        free(weights);
        free(logits);
        return;
    }
    const struct sampler_description description = { layout.step,
						     layout.sessions,
						     layout.logits,
						     layout.weights,
						     layout.result,
						     VOCABULARY,
						     SESSIONS,
						     ROWS,
						     0 };
    bool ran = sampler_prepare(card, kernels, &description, &launch) && test_linear_card_load(card);
    if (ran) {
        if (broken == BROKEN_LOGIT) {          /* a logit above the largest, on the card alone */
            const float raised = (float)SPREAD;
            const uint64_t byte_offset = (uint64_t)ROW_PLAIN * row_bytes + (uint64_t)LOWEST_TOKEN * sizeof(float);
            unsigned char *const laid = card->image + layout.logits.byte_offset + byte_offset;
            float kept_logit = 0.0f;
            memcpy(&kept_logit, laid, sizeof(kept_logit));
            memcpy(laid, &raised, sizeof(raised));
            (void)test_linear_card_upload(card, layout.logits, byte_offset, sizeof(float));
            memcpy(laid, &kept_logit, sizeof(kept_logit));
        }
        ran = test_linear_card_run(card, &launch, 1) && test_linear_card_fetch(card);
        if (ran) {
		const struct step_result result =
			result_of(card, layout.result);
		check(KIND_OTHER,
		      result.sequence == SEQUENCE &&
			      result.row_count == STEP_ROWS &&
			      result.reserved_zero == 0,
		      "the result's block is the step's");
		for (uint32_t row = 0; row < STEP_ROWS; row++) {
			(void)test_linear_card_get(card, layout.weights,
						   row * row_bytes, weights,
						   row_bytes);
			check_answer(row, logits + (size_t)row * VOCABULARY,
				     weights,
				     answer_of(card, layout.result, row), kept,
				     "eight rows");
			if (row == ROW_CUT || row == ROW_ALL) {
				check_weights(weights,
					      logits + (size_t)row * VOCABULARY,
					      VOCABULARY,
					      slot_sessions[row_slots[row]]
						      .temperature,
					      "a keyed row");
			} else {
				check(KIND_OTHER,
				      came_back_as_laid(card, layout.weights,
							row * row_bytes,
							row_bytes),
				      "the weights of row %" PRIu32
				      ", which does not draw, were written",
				      row);
			}
            }
	    check(KIND_OTHER,
		  came_back_as_laid(card, layout.weights,
				    ROW_BEYOND * row_bytes, row_bytes) &&
			  came_back_as_laid(
				  card, layout.result,
				  STEP_RESULT_BYTES +
					  ROW_BEYOND * STEP_ROW_RESULT_BYTES,
				  STEP_ROW_RESULT_BYTES),
		  "the row beyond the step was answered or wrote weights");
	}
        for (uint32_t row = 0; ran && row < STEP_ROWS; row++) {     /* each row alone: its bits among eight */
		const struct linear_range logits_of_row = {
			layout.logits.byte_offset + row * row_bytes, row_bytes
		};
		const struct sampler_description alone = { layout.alone_step,
							   layout.sessions,
							   logits_of_row,
							   layout.alone_weights,
							   layout.alone_result,
							   VOCABULARY,
							   SESSIONS,
							   1,
							   0 };
		const struct step_row_result among_eight =
			answer_of(card, layout.result, row);
		const bool draws = row == ROW_CUT || row == ROW_ALL;
		if (!put_step(card, layout.alone_step, SEQUENCE + 1 + row, 1,
			      &rows[row], 1) ||
		    !test_linear_card_upload(card, layout.alone_step, 0,
					     layout.alone_step.byte_count) ||
		    !sampler_prepare(card, kernels, &alone, &launch) ||
		    !test_linear_card_run(card, &launch, 1) ||
		    !fetch_range(card, layout.alone_result) ||
		    (draws && !fetch_range(card, layout.alone_weights))) {
			break;
		}
		const struct step_row_result by_itself =
			answer_of(card, layout.alone_result, 0);
		check(KIND_OTHER,
		      result_of(card, layout.alone_result).sequence ==
				      SEQUENCE + 1 + row &&
			      by_itself.token_id == among_eight.token_id &&
			      bits_of(by_itself.logit) ==
				      bits_of(among_eight.logit) &&
			      (!draws ||
			       memcmp(card->received +
					      layout.alone_weights.byte_offset,
				      card->received +
					      layout.weights.byte_offset +
					      row * row_bytes,
				      row_bytes) == 0),
		      "row %" PRIu32
		      " alone is not answered as among eight, with the same weights",
		      row);
        }
        if (ran) {                              /* the guards, after the launches of one row */
            (void)test_linear_card_fetch(card);
        }
    }
    free(kept);
    free(weights);
    free(logits);
}

/* ---- draws ---- */

struct draws {
    float *weights;                             /* of the row, as the card wrote them */
    uint32_t *kept;
    uint64_t *counts;                           /* per token */
    int32_t *first_batch;                       /* the tokens of the first batch */
    bool *is_kept;                              /* per token */
    uint32_t kept_count;
};

/* The square root of a positive double by Newton's steps (the tests have no mathematics library). */
static double square_root(double value)
{
    double root = value > 1.0 ? value : 1.0;
    for (uint32_t step = 0; step < NEWTON_STEPS; step++) {
        root = HALF * (root + value / root);
    }
    return root;
}

/* The value a chi-square of these degrees of freedom exceeds with chance CHI_SQUARE_LEAST_P: Wilson and Hilferty's
 * cube of a normal. */
static double chi_square_bound(double degrees)
{
    const double spread = WILSON_HILFERTY_TWO / (WILSON_HILFERTY_NINE * degrees);
    const double root = 1.0 - spread + NORMAL_QUANTILE * square_root(spread);
    return degrees * root * root * root;
}

/* The counts against the kept probabilities. Cells that expect fewer than LEAST_EXPECTED draws are pooled, and the
 * pool joins the cell of the most probable token if it expects fewer itself. */
static void check_distribution(const struct draws *draws, uint64_t draw_count)
{
    double kept_sum = 0.0;
    double statistic = 0.0;
    double pooled_count = 0.0;
    double pooled_expected = 0.0;
    double largest_count = 0.0;
    double largest_expected = 0.0;
    uint32_t cells = 0;
    if (draws->kept_count == 0) {
        return;                                 /* every draw has failed already */
    }
    for (uint32_t index = 0; index < draws->kept_count; index++) {
        kept_sum += (double)draws->weights[draws->kept[index]];
    }
    for (uint32_t index = 0; index < draws->kept_count; index++) {
        const uint32_t token = draws->kept[index];
        const double expected = (double)draw_count * (double)draws->weights[token] / kept_sum;
        const double count = (double)draws->counts[token];
        if (expected > largest_expected) {      /* the cell of the most probable token is added last */
            if (largest_expected >= LEAST_EXPECTED) {
                statistic += (largest_count - largest_expected) * (largest_count - largest_expected) /
                             largest_expected;
                cells++;
            } else {
                pooled_count += largest_count;
                pooled_expected += largest_expected;
            }
            largest_count = count;
            largest_expected = expected;
        } else if (expected >= LEAST_EXPECTED) {
            statistic += (count - expected) * (count - expected) / expected;
            cells++;
        } else {
            pooled_count += count;
            pooled_expected += expected;
        }
    }
    if (pooled_expected >= LEAST_EXPECTED) {
        statistic += (pooled_count - pooled_expected) * (pooled_count - pooled_expected) / pooled_expected;
        cells++;
    } else {
        largest_count += pooled_count;
        largest_expected += pooled_expected;
    }
    statistic += (largest_count - largest_expected) * (largest_count - largest_expected) / largest_expected;
    cells++;
    const double bound = cells > 1 ? chi_square_bound((double)(cells - 1)) : statistic;
    (void)printf("draws: %" PRIu64 " of %" PRIu32 " kept tokens, %" PRIu32 " cells, chi-square %.1f, at p %g %.1f\n",
                 draw_count, draws->kept_count, cells, statistic, CHI_SQUARE_LEAST_P, bound);
    check(KIND_OTHER, cells >= LEAST_CELLS, "the draws fill %" PRIu32 " cells, fewer than %u: a flatter row or more",
          cells, LEAST_CELLS);
    check(KIND_KEYED, cells < LEAST_CELLS || statistic < bound,
          "the draws are not the kept distribution: chi-square %.1f, at p %g %.1f", statistic, CHI_SQUARE_LEAST_P,
          bound);
}

/* One batch: the step area of each launch uploaded alone (never a guard again), the launches run, and the result
 * areas downloaded at once: results spans them. */
static bool run_batch(struct test_linear_card *card,
		      const struct kernel_launch *launches,
		      const struct linear_range *steps,
		      struct linear_range results, uint64_t batch)
{
	struct step_row_record rows[ROWS];
	for (uint32_t launch = 0; launch < BATCH_LAUNCHES; launch++) {
		const uint64_t step = batch * BATCH_LAUNCHES + launch;
		for (uint32_t row = 0; row < ROWS; row++) {
			rows[row] = (struct step_row_record){
				(int32_t)(step * ROWS + row), 0, 0,
				STEP_ROW_LOGITS
			};
		}
		if (!put_step(card, steps[launch], step + 1, ROWS, rows,
			      ROWS) ||
		    !test_linear_card_upload(card, steps[launch], 0,
					     steps[launch].byte_count)) {
			return false;
		}
    }
    return test_linear_card_run(card, launches, BATCH_LAUNCHES) && fetch_range(card, results);
}

static void test_draws(struct test_linear_card *card,
		       const struct sampler_kernels_record *kernels,
		       const float *logits, uint32_t vocabulary_size,
		       uint64_t draw_count)
{
	static const struct step_session session = { SEED_DRAWS, 1.0f, TOP_P };
	const uint64_t row_bytes = (uint64_t)vocabulary_size * sizeof(float);
	const uint64_t step_bytes =
		STEP_BLOCK_BYTES_VALUE + ROWS * STEP_ROW_BYTES_VALUE;
	const uint64_t result_bytes =
		STEP_RESULT_BYTES + ROWS * STEP_ROW_RESULT_BYTES;
	const uint64_t batch_count =
		(draw_count + BATCH_DRAWS - 1) / BATCH_DRAWS;
	struct kernel_launch *const launches =
		malloc(sizeof(struct kernel_launch) * BATCH_LAUNCHES);
	struct linear_range steps[BATCH_LAUNCHES];
	struct linear_range
		results[BATCH_LAUNCHES]; /* each area behind its own guard */
	struct draws draws = { malloc(row_bytes),
			       malloc((size_t)vocabulary_size *
				      sizeof(uint32_t)),
			       calloc(vocabulary_size, sizeof(uint64_t)),
			       malloc(sizeof(int32_t) * BATCH_LAUNCHES * ROWS),
			       calloc(vocabulary_size, sizeof(bool)),
			       0 };
	bool sound = true;
	test_linear_context("card %" PRIu32 ": draws", card->card_index);
	test_linear_card_clear(card);
	const struct linear_range logits_range =
		test_linear_card_work(card, ROWS * row_bytes);
	const struct linear_range weights_range =
		test_linear_card_work(card, ROWS * row_bytes);
	const struct linear_range sessions =
		test_linear_card_work(card, sizeof(session));
	for (uint32_t launch = 0; launch < BATCH_LAUNCHES; launch++) {
		steps[launch] = test_linear_card_work(card, step_bytes);
	}
    for (uint32_t launch = 0; launch < BATCH_LAUNCHES; launch++) {
        results[launch] = test_linear_card_work(card, result_bytes);
    }
    const struct linear_range all_results = {
	    results[0].byte_offset, results[BATCH_LAUNCHES - 1].byte_offset +
					    result_bytes -
					    results[0].byte_offset
    };
    sound = memory_was_had(launches != NULL && draws.weights != NULL && draws.kept != NULL &&
                           draws.counts != NULL && draws.first_batch != NULL && draws.is_kept != NULL) &&
            test_linear_card_has_room(card) && test_linear_card_put(card, sessions, 0, &session, sizeof(session));
    for (uint32_t row = 0; sound && row < ROWS; row++) {
        sound = test_linear_card_put(card, logits_range, row * row_bytes, logits, row_bytes);
    }
    for (uint32_t launch = 0; sound && launch < BATCH_LAUNCHES; launch++) {
	    const struct sampler_description description = { steps[launch],
							     sessions,
							     logits_range,
							     weights_range,
							     results[launch],
							     vocabulary_size,
							     1,
							     ROWS,
							     0 };
	    sound = sampler_prepare(card, kernels, &description,
				    &launches[launch]);
    }
    sound = sound && test_linear_card_load(card);
    for (uint64_t batch = 0; sound && batch < batch_count; batch++) {
        sound = run_batch(card, launches, steps, all_results, batch) && (batch != 0 || test_linear_card_fetch(card));
        if (sound && batch == 0) {              /* the weights the card wrote: eight copies of one row */
            (void)test_linear_card_get(card, weights_range, 0, draws.weights, row_bytes);
            for (uint32_t row = 1; row < ROWS; row++) {
                check(KIND_OTHER,
                      memcmp(card->received + weights_range.byte_offset + row * row_bytes, draws.weights,
                             row_bytes) == 0,
                      "copy %" PRIu32 " of the row has other weights", row);
            }
            check_weights(draws.weights, logits, vocabulary_size, 1.0f, "the drawn row");
            draws.kept_count = keep(draws.weights, vocabulary_size, TOP_P, draws.kept);
            for (uint32_t index = 0; index < draws.kept_count; index++) {
                draws.is_kept[draws.kept[index]] = true;
            }
        }
        for (uint32_t launch = 0; sound && launch < BATCH_LAUNCHES; launch++) {
            const uint64_t step = batch * BATCH_LAUNCHES + launch;
            check(KIND_OTHER, result_of(card, results[launch]).sequence == step + 1,
                  "launch %" PRIu64 " answered another step", step);
            for (uint32_t row = 0; row < ROWS; row++) {
                const int32_t token = answer_of(card, results[launch], row).token_id;
                const int32_t position = (int32_t)(step * ROWS + row);
                const bool holds = token >= 0 && (uint32_t)token < vocabulary_size && draws.is_kept[token] &&
                                   (position >= (int32_t)RACED_DRAWS ||
                                    keyed_token_holds(draws.weights, draws.kept, draws.kept_count, SEED_DRAWS,
                                                      position, token));
                check(KIND_KEYED, holds, "the draw at position %" PRId32 " is %" PRId32 ", not the race's", position,
                      token);
                draws.counts[holds ? (uint32_t)token : 0] += holds ? 1u : 0u;
                if (batch == 0) {
                    draws.first_batch[launch * ROWS + row] = token;
                }
            }
        }
    }
    if (sound) {
        check_distribution(&draws, batch_count * BATCH_DRAWS);
        sound = run_batch(card, launches, steps, all_results, 0) && test_linear_card_fetch(card);   /* every guard */
        for (uint32_t launch = 0; sound && launch < BATCH_LAUNCHES; launch++) {
            for (uint32_t row = 0; row < ROWS; row++) {
                check(KIND_OTHER,
                      answer_of(card, results[launch], row).token_id == draws.first_batch[launch * ROWS + row],
                      "the first batch, run again, drew another token at launch %" PRIu32 ", row %" PRIu32, launch,
                      row);
            }
        }
    }
    free(draws.is_kept);
    free(draws.first_batch);
    free(draws.counts);
    free(draws.kept);
    free(draws.weights);
    free(launches);
}

/* ---- the capture ---- */

struct capture_logits {
    const struct test_capture *capture;
    int64_t *top_ids;                           /* [positions][TOP_RANKS] */
    float *top_values;
    uint32_t position_count;
};

/* The capture's records of the logits, of the shapes they must have; the top ids and values read. */
static bool capture_logits_open(const struct test_capture *capture, struct capture_logits *logits_out)
{
    const struct test_capture_tensor *const all_rows = test_capture_find(capture, "logits.all_rows");
    memset(logits_out, 0, sizeof(*logits_out));
    if (all_rows == NULL || all_rows->dimension_count != ALL_ROWS_DIMENSIONS || all_rows->dimensions[0] != 1 ||
        all_rows->dimensions[POSITION_AXIS] == 0 || all_rows->dimensions[POSITION_AXIS] > INT32_MAX ||
        all_rows->dimensions[VOCABULARY_AXIS] != VOCABULARY || all_rows->type != TEST_CAPTURE_FLOAT32) {
        test_linear_fail("the capture holds no logits.all_rows of float32 [1][positions][vocabulary]");
        return false;
    }
    const uint32_t position_count = (uint32_t)all_rows->dimensions[POSITION_AXIS];
    const uint64_t top_dimensions[TOP_DIMENSIONS] = {position_count, TOP_RANKS};
    const uint64_t top_count = (uint64_t)position_count * TOP_RANKS;
    logits_out->capture = capture;
    logits_out->position_count = position_count;
    logits_out->top_ids = malloc(top_count * sizeof(int64_t));
    logits_out->top_values = malloc(top_count * sizeof(float));
    return memory_was_had(logits_out->top_ids != NULL &&
			  logits_out->top_values != NULL) &&
	   test_linear_expect_status(
		   test_capture_read(
			   capture, "logits.top_ids", TEST_CAPTURE_INT64,
			   top_dimensions, TOP_DIMENSIONS, 0, top_count,
			   logits_out->top_ids, top_count * sizeof(int64_t)),
		   STATUS_OK, "logits.top_ids [positions][16] is read") &&
	   test_linear_expect_status(
		   test_capture_read(
			   capture, "logits.top_values", TEST_CAPTURE_FLOAT32,
			   top_dimensions, TOP_DIMENSIONS, 0, top_count,
			   logits_out->top_values, top_count * sizeof(float)),
		   STATUS_OK, "logits.top_values [positions][16] is read");
}

static bool capture_rows_read(const struct capture_logits *logits, uint32_t first_position, uint32_t row_count,
                              float *rows_out)
{
    const uint64_t dimensions[ALL_ROWS_DIMENSIONS] = {1, logits->position_count, VOCABULARY};
    const uint64_t count = (uint64_t)row_count * VOCABULARY;
    return test_linear_expect_status(
	    test_capture_read(logits->capture, "logits.all_rows",
			      TEST_CAPTURE_FLOAT32, dimensions,
			      ALL_ROWS_DIMENSIONS,
			      (uint64_t)first_position * VOCABULARY, count,
			      rows_out, count * sizeof(float)),
	    STATUS_OK, "rows of logits.all_rows are read");
}

/* The captured position whose TOP_RANKS largest logits lie closest together, the first among equals: the row top_p
 * keeps the most of, for the draws. */
static uint32_t flattest_position(const struct capture_logits *logits)
{
    uint32_t flattest = 0;
    for (uint32_t position = 1; position < logits->position_count; position++) {
        const float *const top = logits->top_values + (size_t)position * TOP_RANKS;
        const float *const best = logits->top_values + (size_t)flattest * TOP_RANKS;
        flattest = top[0] - top[TOP_RANKS - 1] < best[0] - best[TOP_RANKS - 1] ? position : flattest;
    }
    const float *const chosen = logits->top_values + (size_t)flattest * TOP_RANKS;
    (void)printf("draws: the captured position %" PRIu32 " of %" PRIu32 ", its %u largest logits within %g\n",
                 flattest, logits->position_count, TOP_RANKS, (double)(chosen[0] - chosen[TOP_RANKS - 1]));
    return flattest;
}

/* Every captured position, greedy, eight a launch. */
static void test_replay(struct test_linear_card *card,
			const struct sampler_kernels_record *kernels,
			const struct capture_logits *logits)
{
	static const struct step_session greedy = { 0, 0.0f, 1.0f };
	const uint64_t row_bytes = (uint64_t)VOCABULARY * sizeof(float);
	float *const rows_logits = malloc(ROWS * row_bytes);
	struct step_row_record rows[ROWS];
	struct kernel_launch launch;
	uint32_t equal_count = 0;
	test_linear_context("card %" PRIu32 ": the capture's logits",
			    card->card_index);
	test_linear_card_clear(card);
	const struct linear_range logits_range =
		test_linear_card_work(card, ROWS * row_bytes);
	const struct linear_range weights_range =
		test_linear_card_work(card, ROWS * row_bytes);
	const struct linear_range sessions =
		test_linear_card_work(card, sizeof(greedy));
	const struct linear_range step = test_linear_card_work(
		card, STEP_BLOCK_BYTES_VALUE + ROWS * STEP_ROW_BYTES_VALUE);
	const struct linear_range result = test_linear_card_work(
		card, STEP_RESULT_BYTES + ROWS * STEP_ROW_RESULT_BYTES);
	const struct sampler_description description = { step,
							 sessions,
							 logits_range,
							 weights_range,
							 result,
							 VOCABULARY,
							 1,
							 ROWS,
							 0 };
	bool sound = memory_was_had(rows_logits != NULL) &&
		     test_linear_card_has_room(card) &&
		     test_linear_card_put(card, sessions, 0, &greedy,
					  sizeof(greedy)) &&
		     sampler_prepare(card, kernels, &description, &launch) &&
		     test_linear_card_load(card);
	for (uint32_t first = 0; sound && first < logits->position_count;
	     first += ROWS) {
		const uint32_t row_count =
			logits->position_count - first < ROWS ?
				logits->position_count - first :
				ROWS;
		for (uint32_t row = 0; row < ROWS; row++) {
			rows[row] = (struct step_row_record){
				(int32_t)(first + row), 0, 0, STEP_ROW_LOGITS
			};
		}
		sound = capture_rows_read(logits, first, row_count,
					  rows_logits) &&
			test_linear_card_put(card, logits_range, 0, rows_logits,
					     row_count * row_bytes) &&
			put_step(card, step, first + 1u, row_count, rows,
				 ROWS) &&
			test_linear_card_upload(card, logits_range, 0,
						row_count * row_bytes) &&
			test_linear_card_upload(card, step, 0,
						step.byte_count) &&
			test_linear_card_run(card, &launch, 1) &&
			fetch_range(card, result);
		for (uint32_t row = 0; sound && row < row_count; row++) {
			const uint64_t top =
				(uint64_t)(first + row) * TOP_RANKS;
			const struct step_row_result answer =
				answer_of(card, result, row);
			const uint32_t host = test_sampler_greedy(
				rows_logits + (size_t)row * VOCABULARY,
				VOCABULARY);
			const bool equal =
				answer.token_id == logits->top_ids[top] &&
				bits_of(answer.logit) ==
					bits_of(logits->top_values[top]);
			check(KIND_GREEDY,
			      equal && (int64_t)host == logits->top_ids[top],
			      "position %" PRIu32 ": token %" PRId32
			      " (the host's %" PRIu32
			      "), the reference's argmax %" PRId64,
			      first + row, answer.token_id, host,
			      logits->top_ids[top]);
			equal_count += equal ? 1u : 0u;
		}
    }
    (void)printf("card %" PRIu32 ": %" PRIu32 " of %" PRIu32 " captured positions answered as the reference's argmax\n",
                 card->card_index, equal_count, logits->position_count);
    free(rows_logits);
}

/* ---- the program ---- */

static void test_card(uint32_t card_index, const struct capture_logits *logits, const float *draw_logits,
                      uint32_t draw_vocabulary, uint64_t draw_count)
{
    struct test_linear_card card;
    struct sampler_kernels_record kernels;
    if (!test_linear_card_open(&card, card_index, WORK_CAPACITY, 0)) {
        return;
    }
    if (test_linear_expect_status(sampler_kernels_prepare(&backend_cuda_value,
							  card.card, &kernels),
				  STATUS_OK,
				  "the kernel of the sampler is prepared")) {
	    test_rows(&card, &kernels);
	    test_draws(&card, &kernels, draw_logits, draw_vocabulary,
		       draw_count);
	    if (logits->capture != NULL) {
		    test_replay(&card, &kernels, logits);
	    }
    }
    test_linear_card_close(&card);
}

static bool options_read(int argument_count, char **arguments, const char **capture_folder_out,
                         uint64_t *draw_count_out)
{
    for (int index = 1; index < argument_count; index += OPTION_WORDS) {
        const char *const value = index + 1 < argument_count ? arguments[index + 1] : NULL;
        char *end = NULL;
        if (value != NULL && strcmp(arguments[index], "--capture") == 0) {
            *capture_folder_out = value;
        } else if (value != NULL && strcmp(arguments[index], "--draws") == 0) {
            const unsigned long long count = strtoull(value, &end, DECIMAL);
            if (end == value || *end != '\0' || count == 0 || count > MOST_DRAWS) {
                return false;
            }
            *draw_count_out = count;
        } else if (value != NULL && strcmp(arguments[index], "--break") == 0) {
            broken = BROKEN_NONE;
            for (uint32_t kind = BROKEN_LOGIT; kind <= BROKEN_CUT; kind++) {
                broken = strcmp(value, broken_names[kind]) == 0 ? (enum broken)kind : broken;
            }
            if (broken == BROKEN_NONE) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

int main(int argument_count, char **arguments)
{
    const char *capture_folder = NULL;
    uint64_t draw_count = DEFAULT_DRAWS;
    uint32_t card_count = 0;
    struct test_capture *capture = NULL;
    struct capture_logits logits;
    memset(&logits, 0, sizeof(logits));
    if (!options_read(argument_count, arguments, &capture_folder, &draw_count)) {
        (void)fprintf(stderr, "usage: sampler_card_test [--capture <folder>] [--draws <1 to %llu>] "
                              "[--break logit|key|cut]\n", MOST_DRAWS);
        return EXIT_FAILURE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    const uint32_t draw_vocabulary = capture_folder != NULL ? VOCABULARY : DRAW_VOCABULARY;
    float *const draw_logits = malloc((size_t)draw_vocabulary * sizeof(float));
    if (draw_logits == NULL) {
        (void)memory_was_had(false);
        return EXIT_FAILURE;
    }
    synthetic_logits(draw_logits, draw_vocabulary, DRAW_STREAM);
    if (capture_folder != NULL &&
	(!test_linear_expect_status(test_capture_open(capture_folder, &capture),
				    STATUS_OK, "the capture folder opens") ||
	 !capture_logits_open(capture, &logits) ||
	 !capture_rows_read(&logits, flattest_position(&logits), 1,
			    draw_logits))) {
	    logits.capture = NULL;
    }
    if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards") &&
	(capture_folder == NULL || logits.capture != NULL)) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    test_card(card_index, &logits, draw_logits, draw_vocabulary,
			      draw_count);
	    }
    }
    free(logits.top_values);
    free(logits.top_ids);
    test_capture_close(capture);
    free(draw_logits);
    const int failures = test_linear_failures();
    if (failures != 0 || (broken != BROKEN_NONE && reported_count == 0)) {
        (void)fprintf(stderr, "sampler_card_test FAILED: %d failures, %" PRIu64 " reports%s\n", failures,
                      reported_count, broken != BROKEN_NONE && reported_count == 0 ? " (the break was not seen)" : "");
        return EXIT_FAILURE;
    }
    (void)printf("sampler_card_test passed on %" PRIu32 " cards%s%s\n", card_count,
                 capture_folder != NULL ? ", the capture replayed" : " (no capture: nothing replayed)",
                 broken != BROKEN_NONE ? ", the break reported" : "");
    return EXIT_SUCCESS;
}
