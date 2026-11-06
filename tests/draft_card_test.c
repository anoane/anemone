/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft_card_test: the draft's own kernels (kernel_draft.cu) on the real cards against the host reference
 * (draft_reference.h): dspark_v1/DESIGN.md K5-K7.
 *
 *     draft_card_test [--break verify|markov|confidence]
 *
 * It needs the cards free (`make card_tests`): a card without room FAILS the test. On every card:
 *   - VERIFY: every count of proposals k = 0 .. 7, every first mismatch and every place of the end token, a row of
 *     another session first in the step (the cases of draft_test), each a launch of its own: the round's members
 *     sequence .. reserved_zero, the draft's step area and its views byte for byte the reference's, the round's
 *     proposals and confidences untouched. With the release modules, also a step without a DRAFTS row and steps of
 *     8 rows under a record of 5 (the checked modules trap on both): the reference's empty answer, and the answer of
 *     the record's rows alone.
 *   - MARKOV: a vocabulary of MARKOV_VOCABULARY (its last block part empty), W1 and W2 of plain values, the five
 *     positions in order with forced tokens (the
 *     anchor in the draft's step row 0, the draws in result areas 0 .. 3), each launch adding its bias to its logits
 *     row: every logit and every Markov row byte for byte the reference's.
 *   - CONFIDENCE: five positions of plain collapse and Markov rows, the draft's tokens in its result areas: the
 *     round's proposals and confidence_logits byte for byte the reference's, zeros at places 5 and 6, its first
 *     members untouched; then a draft step of 3 rows: zeros from place 3.
 * Shown failing (--break): one word of the card's copy of an input changed behind the reference's back (a proposal
 * of the first case; W2; a confidence weight), so that check alone must fail.
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
#include "anemone/draft.h"
#include "anemone/status.h"
#include "draft_reference.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 1800
#define WORK_CAPACITY (16ull << 20u)
#define WEIGHTS_CAPACITY \
	(40ull           \
	 << 20u) /* W1 and W2 of MARKOV_VOCABULARY, the confidence's weights */
#define ROWS LINEAR_MAXIMUM_ROWS
#define POSITIONS STEP_DRAFT_MOST
#define MOST_POSITIONS STEP_DRAFT_MOST
#define VOCABULARY MODEL_VOCABULARY_VALUE
#define MARKOV_VOCABULARY \
	32771u /* the Markov bias's: no whole blocks (8 entries), the PRO's room */
#define RANK MODEL_MARKOV_RANK
#define HIDDEN MODEL_HIDDEN
#define CONFIDENCE_INPUTS MODEL_DRAFT_CONFIDENCE_INPUTS
#define END_TOKEN 1u
#define SEQUENCE 7u
#define FIRST_POSITION 100
#define OTHER_SESSION 3u
#define VERIFYING_SESSION 1u
#define TOKEN_BASE 1000
#define OTHER_TOKEN 77
#define ANCHOR_AND_OTHER_ROWS 2u
#define MOST_CASES     \
	((ROWS - 1u) * \
	 ROWS) /* of one count of proposals: mismatches by end places */
#define ROUND_UNTOUCHED 0x5A
#define SHORT_STEP_ROWS 3u /* the confidence's step of fewer rows */
#define SHORT_RECORD_ROWS \
	5u /* a verify record of fewer rows than its step's 8: its rows alone read */
#define STREAM_W1 0x4D570001u
#define STREAM_W2 0x4D570002u
#define STREAM_LOGITS 0x4D570003u
#define STREAM_COLLAPSE 0x43460001u
#define STREAM_MARKOV 0x43460002u
#define STREAM_WEIGHTS 0x43460003u
#define FORCED_TOKEN_STEP \
	40009u /* position j's token: (j + 1) times it, modulo the vocabulary */
#define LOGIT_SPAN 16.0
#define HASH_FRACTION_SHIFT 40u
#define HASH_FRACTION_UNIT 0x1p-24
#define HALF 0.5
#define BREAK_ARGUMENTS 2

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum check { VERIFY_CHECK, MARKOV_CHECK, CONFIDENCE_CHECK, CHECK_COUNT };
static const char *const check_names[CHECK_COUNT] = { "VERIFY", "MARKOV",
						      "CONFIDENCE" };
static const char *const break_names[CHECK_COUNT] = { "verify", "markov",
						      "confidence" };
static uint32_t check_failures[CHECK_COUNT];
static int break_check = -1;

static uint16_t w1[MARKOV_VOCABULARY * RANK];
static uint16_t w2[MARKOV_VOCABULARY * RANK];
static float logits_in[POSITIONS][MARKOV_VOCABULARY];
static float logits_expected[POSITIONS][MARKOV_VOCABULARY];
static float logits_back[POSITIONS][MARKOV_VOCABULARY];
static uint16_t markov_expected[POSITIONS][RANK];
static uint16_t markov_back[POSITIONS][RANK];
static uint16_t collapse_rows[POSITIONS][HIDDEN];
static uint16_t markov_rows[POSITIONS][RANK];
static uint16_t confidence_weights[CONFIDENCE_INPUTS];

/* Whether two stretches hold the same bytes: the bits of floats, not their values. */
static bool same_bits(const void *first, const void *second, size_t byte_count)
{
	return memcmp(first, second, byte_count) == 0;
}

static void check_fail(enum check check, const char *what, uint32_t index)
{
	if (check_failures[check] < ROWS) {
		test_linear_fail("%s: %s (%" PRIu32 ")", check_names[check],
				 what, index);
	}
	check_failures[check]++;
}

static double hash_fraction(uint64_t stream, uint64_t position)
{
	return (double)(test_linear_hash(stream, position) >>
			HASH_FRACTION_SHIFT) *
	       HASH_FRACTION_UNIT;
}

/* ---- VERIFY ---- */

struct verify_case {
	struct step_block block;
	struct step_row_record rows[ROWS];
	struct step_row_result results[ROWS];
	struct linear_range step, result, round, draft_step, views;
};

/* As draft_test: an anchor and k proposals of the verifying session behind a row of another session; the draws the
 * next proposal's token below mismatch, END_TOKEN at end_index (none for 0), another token otherwise. */
static void verify_case_make(uint32_t proposals, uint32_t mismatch,
			     uint32_t end_index, struct verify_case *step)
{
	memset(step, 0, sizeof(*step));
	step->block = (struct step_block){ SEQUENCE,
					   proposals + ANCHOR_AND_OTHER_ROWS,
					   0 };
	step->rows[0] = (struct step_row_record){ FIRST_POSITION, OTHER_TOKEN,
						  OTHER_SESSION,
						  STEP_ROW_LOGITS };
	step->results[0] = (struct step_row_result){ OTHER_TOKEN, 0.0f };
	for (uint32_t index = 0; index <= proposals; index++) {
		const uint32_t drawn_index = index + 1u;
		step->rows[index + 1u] = (struct step_row_record){
			FIRST_POSITION + (int32_t)index,
			TOKEN_BASE + (int32_t)index, VERIFYING_SESSION,
			STEP_ROW_LOGITS | (index == 0 ? STEP_ROW_DRAFTS :
							STEP_ROW_PROPOSED)
		};
		const int32_t drawn =
			drawn_index == end_index ?
				(int32_t)END_TOKEN :
			drawn_index < mismatch ?
				TOKEN_BASE + (int32_t)drawn_index :
				OTHER_TOKEN + (int32_t)drawn_index;
		step->results[index + 1u] =
			(struct step_row_result){ drawn, 0.0f };
	}
}

/* One count of proposals: every case a launch of its own in one run, prepared for record_rows rows; every write
 * against the reference. */
static void verify_batch(struct test_linear_card *card,
			 const struct draft_kernels_record *kernels,
			 uint32_t proposals, bool without_drafts,
			 uint32_t record_rows)
{
	static struct verify_case cases[MOST_CASES];
	static struct kernel_launch launches[MOST_CASES];
	static unsigned char round_untouched[STEP_ROUND_BYTES];
	uint32_t case_count = 0;
	bool is_run = true;
	memset(round_untouched, ROUND_UNTOUCHED, sizeof(round_untouched));
	test_linear_card_clear(card);
	for (uint32_t mismatch = 1; mismatch <= proposals + 1u; mismatch++) {
		for (uint32_t end_index = 0;
		     end_index <= proposals + 1u && case_count < MOST_CASES;
		     end_index++) {
			struct verify_case *const step = &cases[case_count];
			/* the break: of the case that accepts every proposal, the first
			 * proposal changed on the card alone */
			const bool is_broken =
				break_check == VERIFY_CHECK && proposals != 0 &&
				mismatch == proposals + 1u && end_index == 0;
			verify_case_make(proposals, mismatch, end_index, step);
			if (without_drafts) {
				step->rows[1].flags = STEP_ROW_LOGITS;
			}
			if (is_broken) {
				step->rows[ANCHOR_AND_OTHER_ROWS].token_id =
					OTHER_TOKEN;
			}
			step->step = test_linear_card_work(
				card,
				STEP_BLOCK_BYTES_VALUE +
					(uint64_t)ROWS * STEP_ROW_BYTES_VALUE);
			step->result = test_linear_card_work(
				card,
				STEP_RESULT_BYTES +
					(uint64_t)ROWS * STEP_ROW_RESULT_BYTES);
			step->round =
				test_linear_card_work(card, STEP_ROUND_BYTES);
			step->draft_step = test_linear_card_work(
				card, STEP_BLOCK_BYTES_VALUE +
					      (uint64_t)POSITIONS *
						      STEP_ROW_BYTES_VALUE);
			step->views = test_linear_card_work(
				card,
				(uint64_t)POSITIONS * STEP_DRAFT_VIEW_BYTES);
			const struct verify_matches_description description = {
				step->step,
				step->result,
				step->round,
				step->draft_step,
				step->views,
				record_rows,
				POSITIONS,
				END_TOKEN,
				VOCABULARY,
				0,
				0,
				0,
				0
			};
			is_run =
				is_run && test_linear_card_has_room(card) &&
				test_linear_card_put(card, step->step, 0,
						     &step->block,
						     sizeof(step->block)) &&
				test_linear_card_put(card, step->step,
						     STEP_BLOCK_BYTES_VALUE,
						     step->rows,
						     sizeof(step->rows)) &&
				test_linear_card_put(card, step->result,
						     STEP_RESULT_BYTES,
						     step->results,
						     sizeof(step->results)) &&
				test_linear_card_put(card, step->round, 0,
						     round_untouched,
						     sizeof(round_untouched)) &&
				test_linear_expect_status(
					verify_matches_prepare(
						&description,
						card->arena_byte_count, kernels,
						&launches[case_count]),
					STATUS_OK,
					"the verify kernel is prepared");
			if (is_broken) {
				step->rows[ANCHOR_AND_OTHER_ROWS].token_id =
					TOKEN_BASE +
					1; /* the reference's own */
			}
			case_count++;
		}
	}
	is_run = is_run && test_linear_card_load(card) &&
		 test_linear_card_run(card, launches, case_count) &&
		 test_linear_card_fetch(card);
	for (uint32_t index = 0; is_run && index < case_count; index++) {
		struct verify_case *const step = &cases[index];
		struct step_round round;
		struct step_block draft_block;
		struct step_row_record draft_rows[MOST_POSITIONS];
		unsigned char views[MOST_POSITIONS * STEP_DRAFT_VIEW_BYTES];
		memset(&round, ROUND_UNTOUCHED, sizeof(round));
		memset(&draft_block, 0, sizeof(draft_block));
		memset(draft_rows, 0, sizeof(draft_rows));
		memset(views, 0, sizeof(views));
		test_draft_verify(&step->block, step->rows, step->results,
				  record_rows, POSITIONS, END_TOKEN,
				  MODEL_DRAFT_NOISE_TOKEN, &round, &draft_block,
				  draft_rows, views);
		/* the ranges began as zeros: what neither side writes (rows without a DRAFTS row) stays zero on both */
		const bool is_same =
			test_linear_card_expect_bytes(card, step->round, 0,
						      &round, sizeof(round),
						      "the round") &&
			test_linear_card_expect_bytes(
				card, step->draft_step, 0, &draft_block,
				sizeof(draft_block),
				"the draft's step block") &&
			test_linear_card_expect_bytes(
				card, step->draft_step, STEP_BLOCK_BYTES_VALUE,
				draft_rows,
				(uint64_t)POSITIONS * STEP_ROW_BYTES_VALUE,
				"the draft's step rows") &&
			test_linear_card_expect_bytes(
				card, step->views, 0, views,
				(uint64_t)POSITIONS * STEP_DRAFT_VIEW_BYTES,
				"the draft's views");
		if (!is_same) {
			check_fail(VERIFY_CHECK,
				   "a case's writes are not the reference's",
				   proposals);
		}
	}
	if (!is_run) {
		check_fail(VERIFY_CHECK, "a batch did not run", proposals);
	}
}

/* ---- MARKOV, CONFIDENCE, ROUND ---- */

static void inputs_make(void)
{
	test_linear_values(w1, MARKOV_VOCABULARY * RANK, STREAM_W1,
			   TEST_LINEAR_VALUES_PLAIN);
	test_linear_values(w2, MARKOV_VOCABULARY * RANK, STREAM_W2,
			   TEST_LINEAR_VALUES_PLAIN);
	for (uint32_t position = 0; position < POSITIONS; position++) {
		for (uint32_t entry = 0; entry < MARKOV_VOCABULARY; entry++) {
			logits_in[position][entry] =
				(float)((hash_fraction(STREAM_LOGITS + position,
						       entry) -
					 HALF) *
					LOGIT_SPAN);
		}
		test_linear_values(collapse_rows[position], HIDDEN,
				   STREAM_COLLAPSE + position,
				   TEST_LINEAR_VALUES_PLAIN);
		test_linear_values(markov_rows[position], RANK,
				   STREAM_MARKOV + position,
				   TEST_LINEAR_VALUES_PLAIN);
	}
	test_linear_values(confidence_weights, CONFIDENCE_INPUTS,
			   STREAM_WEIGHTS, TEST_LINEAR_VALUES_PLAIN);
}

/* The draft's tokens of a position: the anchor (position 0) in the step row, the draws in the result areas. */
static int32_t forced_token(uint32_t position)
{
	return (int32_t)(((position + 1u) * FORCED_TOKEN_STEP) %
			 MARKOV_VOCABULARY);
}

static bool tokens_put(struct test_linear_card *card,
		       struct linear_range draft_step,
		       struct linear_range results, uint32_t step_rows)
{
	const struct step_block block = { SEQUENCE, step_rows, 0 };
	const struct step_row_record anchor = { FIRST_POSITION, forced_token(0),
						VERIFYING_SESSION,
						STEP_ROW_LOGITS };
	bool is_put = test_linear_card_put(card, draft_step, 0, &block,
					   sizeof(block)) &&
		      test_linear_card_put(card, draft_step,
					   STEP_BLOCK_BYTES_VALUE, &anchor,
					   sizeof(anchor));
	for (uint32_t area = 0; is_put && area < POSITIONS; area++) {
		const struct step_result result = { SEQUENCE, 1, 0 };
		const struct step_row_result drawn = { forced_token(area + 1u),
						       0.0f };
		is_put = test_linear_card_put(card, results,
					      (uint64_t)area *
						      STEP_DRAFT_RESULT_BYTES,
					      &result, sizeof(result)) &&
			 test_linear_card_put(
				 card, results,
				 (uint64_t)area * STEP_DRAFT_RESULT_BYTES +
					 STEP_RESULT_BYTES,
				 &drawn, sizeof(drawn));
	}
	return is_put;
}

static void markov_run(struct test_linear_card *card,
		       const struct draft_kernels_record *kernels)
{
	struct kernel_launch launches[POSITIONS];
	test_linear_card_clear(card);
	const struct linear_range draft_step = test_linear_card_work(
		card, STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE);
	const struct linear_range results = test_linear_card_work(
		card, (uint64_t)POSITIONS * STEP_DRAFT_RESULT_BYTES);
	const struct linear_range logits =
		test_linear_card_work(card, sizeof(logits_in));
	const struct linear_range markov =
		test_linear_card_work(card, sizeof(markov_back));
	const struct linear_range embedding =
		test_linear_card_weights(card, sizeof(w1));
	const struct linear_range head =
		test_linear_card_weights(card, sizeof(w2));
	const uint16_t broken =
		(uint16_t)(w2[(size_t)forced_token(0) * RANK] ^ 1u);
	bool is_run =
		test_linear_card_has_room(card) &&
		tokens_put(card, draft_step, results, POSITIONS) &&
		test_linear_card_put(card, logits, 0, logits_in,
				     sizeof(logits_in)) &&
		test_linear_card_put(card, embedding, 0, w1, sizeof(w1)) &&
		test_linear_card_put(card, head, 0, w2, sizeof(w2)) &&
		(break_check != MARKOV_CHECK ||
		 test_linear_card_put(card, head,
				      (uint64_t)forced_token(0) *
					      DRAFT_MARKOV_ROW_BYTES,
				      &broken, sizeof(broken)));
	for (uint32_t position = 0; is_run && position < POSITIONS;
	     position++) {
		const struct draft_markov_bias_description description = {
			draft_step, results,   embedding,
			head,	    logits,    markov,
			position,   POSITIONS, MARKOV_VOCABULARY,
			0,	    0
		};
		is_run = test_linear_expect_status(
			draft_markov_bias_prepare(&description,
						  card->arena_byte_count,
						  kernels, &launches[position]),
			STATUS_OK, "the Markov bias is prepared");
	}
	is_run = is_run && test_linear_card_load(card) &&
		 test_linear_card_run(card, launches, POSITIONS) &&
		 test_linear_card_fetch(card) &&
		 test_linear_card_get(card, logits, 0, logits_back,
				      sizeof(logits_back)) &&
		 test_linear_card_get(card, markov, 0, markov_back,
				      sizeof(markov_back)) &&
		 test_linear_card_expect_untouched(
			 card, draft_step,
			 "the draft's step area is only read") &&
		 test_linear_card_expect_untouched(
			 card, results, "the result areas are only read") &&
		 test_linear_card_expect_weights_untouched(card);
	if (!is_run) {
		check_fail(MARKOV_CHECK, "the run did not finish", 0);
		return;
	}
	memcpy(logits_expected, logits_in, sizeof(logits_expected));
	for (uint32_t position = 0; position < POSITIONS; position++) {
		test_draft_markov_bias(w1, w2, MARKOV_VOCABULARY,
				       (uint32_t)forced_token(position),
				       logits_expected[position],
				       markov_expected[position]);
		if (!same_bits(logits_back[position], logits_expected[position],
			       sizeof(logits_back[position]))) {
			check_fail(
				MARKOV_CHECK,
				"a position's logits are not the reference's",
				position);
		}
		if (memcmp(markov_back[position], markov_expected[position],
			   sizeof(markov_back[position])) != 0) {
			check_fail(
				MARKOV_CHECK,
				"a Markov row is not W1's row of the position's token",
				position);
		}
	}
}

static void confidence_run(struct test_linear_card *card,
			   const struct draft_kernels_record *kernels)
{
	static const uint32_t step_rows_of[] = { POSITIONS, SHORT_STEP_ROWS };
	static unsigned char round_untouched[STEP_ROUND_BYTES];
	memset(round_untouched, ROUND_UNTOUCHED, sizeof(round_untouched));
	for (size_t case_index = 0;
	     case_index < sizeof(step_rows_of) / sizeof(step_rows_of[0]);
	     case_index++) {
		const uint32_t step_rows = step_rows_of[case_index];
		struct kernel_launch launch;
		test_linear_card_clear(card);
		const struct linear_range draft_step = test_linear_card_work(
			card, STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE);
		const struct linear_range collapse =
			test_linear_card_work(card, sizeof(collapse_rows));
		const struct linear_range markov =
			test_linear_card_work(card, sizeof(markov_rows));
		const struct linear_range results = test_linear_card_work(
			card, (uint64_t)POSITIONS * STEP_DRAFT_RESULT_BYTES);
		const struct linear_range round =
			test_linear_card_work(card, STEP_ROUND_BYTES);
		const struct linear_range weight = test_linear_card_weights(
			card, sizeof(confidence_weights));
		const struct draft_confidence_description description = {
			draft_step, collapse,  markov, weight, results,
			round,	    POSITIONS, 0,      0
		};
		const uint16_t broken = (uint16_t)(confidence_weights[0] ^ 1u);
		struct step_round expected;
		const bool is_run =
			test_linear_card_has_room(card) &&
			tokens_put(card, draft_step, results, step_rows) &&
			test_linear_card_put(card, collapse, 0, collapse_rows,
					     sizeof(collapse_rows)) &&
			test_linear_card_put(card, markov, 0, markov_rows,
					     sizeof(markov_rows)) &&
			test_linear_card_put(card, round, 0, round_untouched,
					     sizeof(round_untouched)) &&
			test_linear_card_put(card, weight, 0,
					     confidence_weights,
					     sizeof(confidence_weights)) &&
			(break_check != CONFIDENCE_CHECK ||
			 test_linear_card_put(card, weight, 0, &broken,
					      sizeof(broken))) &&
			test_linear_expect_status(
				draft_confidence_prepare(&description,
							 card->arena_byte_count,
							 kernels, &launch),
				STATUS_OK, "the confidence is prepared") &&
			test_linear_card_load(card) &&
			test_linear_card_run(card, &launch, 1) &&
			test_linear_card_fetch(card) &&
			test_linear_card_expect_weights_untouched(card);
		if (!is_run) {
			check_fail(CONFIDENCE_CHECK, "the run did not finish",
				   step_rows);
			continue;
		}
		memset(&expected, ROUND_UNTOUCHED, sizeof(expected));
		memset(expected.proposals, 0, sizeof(expected.proposals));
		memset(expected.confidence_logits, 0,
		       sizeof(expected.confidence_logits));
		for (uint32_t place = 0; place < step_rows && place < POSITIONS;
		     place++) {
			expected.proposals[place] = forced_token(place + 1u);
			expected.confidence_logits[place] =
				test_draft_confidence(collapse_rows[place],
						      markov_rows[place],
						      confidence_weights);
		}
		if (!test_linear_card_expect_bytes(card, round, 0, &expected,
						   sizeof(expected),
						   "the round")) {
			check_fail(
				CONFIDENCE_CHECK,
				"the round's proposals and confidences are not the reference's",
				step_rows);
		}
	}
}

static void card_run(uint32_t card_index)
{
	static struct test_linear_card card;
	struct hyper_kernels_record
		hyper_kernels; /* the families draft.h fills beside its own */
	struct router_kernels_record router_kernels;
	struct attention_state_kernels state_kernels;
	struct attention_core_kernels core_kernels;
	struct draft_kernels_record kernels;
	if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
				  WEIGHTS_CAPACITY) &&
	    test_linear_expect_status(
		    draft_kernels_prepare(&backend_cuda_value, card.card,
					  card.arithmetic_class, &hyper_kernels,
					  &router_kernels, &state_kernels,
					  &core_kernels, &kernels),
		    STATUS_OK, "the draft's kernels are prepared")) {
		const uint32_t failures_before[CHECK_COUNT] = {
			check_failures[VERIFY_CHECK],
			check_failures[MARKOV_CHECK],
			check_failures[CONFIDENCE_CHECK]
		};
		test_linear_context("card %" PRIu32 ", VERIFY", card_index);
		for (uint32_t proposals = 0; proposals < ROWS - 1u;
		     proposals++) {
			verify_batch(&card, &kernels, proposals, false, ROWS);
		}
		if (!CHECKING_HOOKS_ENABLED) {
			verify_batch(&card, &kernels, 1u, true, ROWS);
			verify_batch(&card, &kernels,
				     ROWS - ANCHOR_AND_OTHER_ROWS, false,
				     SHORT_RECORD_ROWS);
		}
		test_linear_context("card %" PRIu32 ", MARKOV", card_index);
		markov_run(&card, &kernels);
		test_linear_context("card %" PRIu32 ", CONFIDENCE", card_index);
		confidence_run(&card, &kernels);
		for (uint32_t check = 0; check < CHECK_COUNT; check++) {
			(void)printf("card %" PRIu32 " (class %" PRIu32
				     "): %s %s\n",
				     card_index, card.arithmetic_class,
				     check_names[check],
				     check_failures[check] ==
						     failures_before[check] ?
					     "byte for byte the reference's" :
					     "NOT the reference's");
		}
	}
	test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
	uint32_t card_count = 0;
	if (argument_count == BREAK_ARGUMENTS + 1 &&
	    strcmp(arguments[1], "--break") == 0) {
		for (uint32_t check = 0; check < CHECK_COUNT; check++) {
			break_check = strcmp(arguments[BREAK_ARGUMENTS],
					     break_names[check]) == 0 ?
					      (int)check :
					      break_check;
		}
	}
	if (argument_count != 1 && break_check < 0) {
		(void)fprintf(
			stderr,
			"usage: draft_card_test [--break verify|markov|confidence]\n");
		return TEST_WRONG_USAGE;
	}
	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	inputs_make();
	if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				      STATUS_OK, "the cards are counted") &&
	    test_linear_expect(card_count > 0 &&
				       card_count <= BACKEND_MAXIMUM_CARDS,
			       "the machine has cards for the test")) {
		for (uint32_t card_index = 0; card_index < card_count;
		     card_index++) {
			card_run(card_index);
		}
	}
	if (break_check >= 0) {
		bool only_the_break = check_failures[break_check] != 0;
		for (uint32_t check = 0; check < CHECK_COUNT; check++) {
			only_the_break = only_the_break &&
					 ((int)check == break_check ||
					  check_failures[check] == 0);
		}
		(void)printf("draft_card_test %s: the break of %s\n",
			     only_the_break ? "passed" : "FAILED",
			     break_names[break_check]);
		return only_the_break ? TEST_PASSED : TEST_FAILED;
	}
	if (test_linear_failures() != 0) {
		(void)fprintf(stderr,
			      "draft_card_test FAILED: %d failure lines\n",
			      test_linear_failures());
		return TEST_FAILED;
	}
	(void)printf("draft_card_test passed on %" PRIu32 " cards\n",
		     card_count);
	return TEST_PASSED;
}
