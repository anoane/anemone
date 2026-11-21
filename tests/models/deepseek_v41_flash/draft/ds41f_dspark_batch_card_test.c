/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_dspark_batch_card_test (tp4_v3_b): a BATCH step's draft kernels (ds41f_dspark_records.h's ROUND_TABLE and
 * SLAB_RESULTS, model_seam.h's BATCH) on the real cards against the host reference (ds41f_dspark_reference.h), with no
 * model:
 *
 *     ds41f_dspark_batch_card_test [TRIALS]
 *
 * Each trial (its own step sequence) builds a step of up to 64 rows: 1 to 8 sessions of distinct slots, each either
 * drafting (its DRAFTS anchor and 0 to 7 proposals, consecutive positions) or plain (one row of logits), in a random
 * order; the target's draws in the slab result areas (8 rows a slab), matching each drafting session's proposals up
 * to a random first mismatch, the end token at a random place. Then:
 *   - VERIFY: 8 launches, verifier v the step's v-th DRAFTS row, each with its own draft step area and views, one
 *     round table of 8 slots: for a verifier with a DRAFTS row, its round (at its session's slot), its draft step
 *     area and views byte for byte what ds41f_test_draft_verify gives on that session's rows alone; for one without, an
 *     empty draft block and views; the slots of sessions without a DRAFTS row untouched.
 *   - CONFIDENCE: a launch a verifier over plain collapse and Markov rows and its draft's tokens: the proposals and
 *     confidence logits at its session's slot byte for byte the reference's; a verifier without a DRAFTS row writes
 *     nothing.
 *   - BATCHED: the same step, the 8 verifiers into one draft: slab blocks (draft rows v R .. v R + R - 1, row i in
 *     block i / 8 at place i % 8; verifier 0 heading every block with its rows of the DRAFTS count D x R) and
 *     position-major views (view j: D rows, row v verifier v's row j); then the Markov bias of each position over
 *     every slot (MARKOV_VOCABULARY entries, plain W1 and W2, its tokens the anchors and given draws in the position
 *     result areas) and each verifier's confidence over its collapse rows and those Markov rows: every block header,
 *     row, view, logit, Markov row and the round table byte for byte the reference's placement, slots of no
 *     verifier untouched.
 * A card without room FAILS the test.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "engine/backend/backend_seam.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark.h"
#include "common/status.h"
#include "ds41f_dspark_reference.h"
#include "engine/library/block_scaled_linear/linear_card_fixture.h"
#include "engine/library/block_scaled_linear/linear_reference.h"
#include "support/byte_identity.h"

#define TEST_TIME_LIMIT_SECONDS 1800
#define WORK_CAPACITY (48ull << 20u)
#define WEIGHTS_CAPACITY (48ull << 20u) /* W1 and W2 of MARKOV_VOCABULARY, the confidence's weights */
#define MOST_ROWS DS41F_DRAFT_BATCH_MOST_ROWS
#define SLAB_ROWS DS41F_DRAFT_SLAB_ROWS
#define SLABS (MOST_ROWS / SLAB_ROWS)
#define VERIFIERS 8u
#define SLOTS 8u
#define POSITIONS STEP_DRAFT_MOST
#define HIDDEN DS41F_HIDDEN
#define RANK DS41F_MARKOV_RANK
#define CONFIDENCE_INPUTS DS41F_DRAFT_CONFIDENCE_INPUTS
#define VOCABULARY DS41F_VOCABULARY_VALUE
#define MARKOV_VOCABULARY 32771u /* the batched Markov bias's: no whole last block of 8 entries */
#define BATCH_SLOTS DS41F_DRAFT_BATCH_SLOTS
#define LOGIT_SPAN 8.0
#define LOGIT_STEPS (1u << 16u)
#define STREAM_W1 3u
#define STREAM_W2 4u
#define STREAM_LOGITS 5u
#define END_TOKEN 1u
#define DEFAULT_TRIALS 48u
#define MOST_TRIALS 4096u
#define FIRST_SEQUENCE 1000u
#define TOKEN_BASE 2000
#define OTHER_TOKEN 77
#define ROUND_UNTOUCHED 0xA5
#define SEED 0x0BA7C4D2A5ull
#define STREAM_STEP 1u
#define STREAM_INPUTS 2u
/* a plain value: one of PLAIN_STEPS quarters about PLAIN_CENTRE, as the upper half of its float */
#define PLAIN_STEPS 17u
#define PLAIN_CENTRE 8
#define PLAIN_STEP 0.25f
#define BF16_SHIFT 16u
/* the draws' keys: a trial's index under its stream, a session's draws under its key */
#define STREAM_SHIFT 32u
#define SESSION_KEY_SHIFT 8u
#define DRAW_DRAFTS 100u
#define DRAW_PROPOSALS 101u
#define DRAW_MISMATCH 102u
#define DRAW_END 103u
#define DRAW_POSITION 104u
#define DRAFT_ODDS 4u /* a session drafts unless its draw of 4 is 0 */
#define END_PLACES_MORE 2u /* the end token at one of the proposals + 1 rows, or at none */
#define FIRST_POSITION 100
#define POSITION_SPREAD 4000u
#define SESSION_TOKEN_STRIDE 16u
#define TOKEN_STRIDE 64u /* a verifier's (a slot's) tokens apart */
#define INPUT_SHIFT 16u
#define LOGIT_SHIFT 20u
#define LOGIT_SPAN_HALVES 2.0
#define LAUNCHES_A_VERIFIER 2u
#define MOST_ARGUMENTS 2 /* the program and TRIALS */
#define TEST_PASSED 0
#define TEST_FAILED 1
#define TEST_WRONG_USAGE 2

static uint16_t collapse_rows[VERIFIERS][POSITIONS][HIDDEN];
static uint16_t markov_rows[VERIFIERS][POSITIONS][RANK];
static uint16_t confidence_weights[CONFIDENCE_INPUTS];
static uint16_t w1[MARKOV_VOCABULARY * RANK], w2[MARKOV_VOCABULARY * RANK];
static float logits_in[POSITIONS][BATCH_SLOTS][MARKOV_VOCABULARY];
static float logits_expected[POSITIONS][BATCH_SLOTS][MARKOV_VOCABULARY];
static uint32_t failures;

static uint32_t draw(uint64_t stream, uint64_t position, uint32_t modulus)
{
	return (uint32_t)(test_linear_hash(stream, position) % modulus);
}

/* A plain bfloat16 value of -2 .. 2 in quarters: products and sums stay exact enough to be any order's. */
static uint16_t plain_value(uint64_t stream, uint64_t position)
{
	const float value = (float)((int32_t)draw(stream, position, PLAIN_STEPS) - PLAIN_CENTRE) * PLAIN_STEP;
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)(bits >> BF16_SHIFT);
}

struct trial {
	struct step_block block;
	struct step_row_record rows[MOST_ROWS];
	struct step_row_result results[MOST_ROWS];
	uint32_t row_count, session_count;
	uint32_t slot_of[SLOTS], first_row[SLOTS], rows_of[SLOTS];
	bool drafts[SLOTS];
};

/* Sessions in a random order; a drafting session's draws the next proposal's token below its first mismatch, the
 * end token at its end place (none where 0), another token otherwise. */
static void trial_make(uint32_t index, struct trial *trial)
{
	memset(trial, 0, sizeof(*trial));
	const uint64_t stream = (uint64_t)STREAM_STEP << STREAM_SHIFT | index;
	trial->session_count = 1u + draw(stream, 0, SLOTS);
	uint32_t slots[SLOTS];
	for (uint32_t slot = 0; slot < SLOTS; slot++)
		slots[slot] = slot;
	for (uint32_t place = SLOTS - 1u; place > 0; place--) { /* a random slot for each session */
		const uint32_t other = draw(stream, 1u + place, place + 1u);
		const uint32_t held = slots[place];
		slots[place] = slots[other];
		slots[other] = held;
	}
	uint32_t row = 0;
	for (uint32_t session = 0; session < trial->session_count; session++) {
		const uint64_t key = (uint64_t)session << SESSION_KEY_SHIFT;
		const bool drafts = draw(stream, DRAW_DRAFTS + key, DRAFT_ODDS) != 0;
		const uint32_t proposals = drafts ? draw(stream, DRAW_PROPOSALS + key, POSITIONS + 1u) : 0u;
		const uint32_t mismatch = 1u + draw(stream, DRAW_MISMATCH + key, proposals + 1u);
		const uint32_t end_index = draw(stream, DRAW_END + key, proposals + END_PLACES_MORE);
		const int32_t position = FIRST_POSITION + (int32_t)draw(stream, DRAW_POSITION + key, POSITION_SPREAD);
		trial->slot_of[session] = slots[session];
		trial->first_row[session] = row;
		trial->rows_of[session] = proposals + 1u;
		trial->drafts[session] = drafts;
		for (uint32_t place = 0; place <= proposals; place++, row++) {
			const int32_t token = TOKEN_BASE + (int32_t)(session * SESSION_TOKEN_STRIDE + place);
			trial->rows[row] = (struct step_row_record){
				position + (int32_t)place, token,
				slots[session],
				STEP_ROW_LOGITS |
					(!drafts    ? 0u :
					 place == 0 ? STEP_ROW_DRAFTS :
						      STEP_ROW_PROPOSED)
			};
			const uint32_t drawn_index = place + 1u;
			const int32_t drawn = drawn_index == end_index ? (int32_t)END_TOKEN :
					      drawn_index < mismatch ? token + 1 :
								       OTHER_TOKEN + (int32_t)drawn_index;
			trial->results[row] =
				(struct step_row_result){ drawn, (float)row };
		}
	}
	trial->row_count = row;
	trial->block = (struct step_block){ FIRST_SEQUENCE + index, row, 0 };
}

static void inputs_make(void)
{
	for (uint32_t verifier = 0; verifier < VERIFIERS; verifier++) {
		for (uint32_t position = 0; position < POSITIONS; position++) {
			const uint64_t at = ((uint64_t)verifier * POSITIONS + position) << INPUT_SHIFT;
			for (uint32_t input = 0; input < HIDDEN; input++)
				collapse_rows[verifier][position][input] = plain_value(STREAM_INPUTS, at + input);
			for (uint32_t input = 0; input < RANK; input++)
				markov_rows[verifier][position][input] =
					plain_value(STREAM_INPUTS, at + HIDDEN + input);
		}
	}
	for (uint32_t input = 0; input < CONFIDENCE_INPUTS; input++)
		confidence_weights[input] = plain_value(STREAM_INPUTS + 1u, input);
	test_linear_values(w1, MARKOV_VOCABULARY * RANK, STREAM_W1, TEST_LINEAR_VALUES_PLAIN);
	test_linear_values(w2, MARKOV_VOCABULARY * RANK, STREAM_W2, TEST_LINEAR_VALUES_PLAIN);
	for (uint32_t position = 0; position < POSITIONS; position++)
		for (uint32_t slot = 0; slot < BATCH_SLOTS; slot++)
			for (uint32_t entry = 0; entry < MARKOV_VOCABULARY; entry++)
				logits_in[position][slot][entry] =
					(float)((double)draw(STREAM_LOGITS,
							     ((uint64_t)position * BATCH_SLOTS + slot) << LOGIT_SHIFT |
								     entry,
							     LOGIT_STEPS) /
							(double)LOGIT_STEPS * LOGIT_SPAN -
						LOGIT_SPAN / LOGIT_SPAN_HALVES);
}

static void fail(const char *what, uint32_t trial, uint32_t verifier)
{
	(void)fprintf(stderr, "draft_batch_card_test: trial %" PRIu32 " verifier %" PRIu32 ": %s\n", trial, verifier,
		      what);
	failures++;
}

static void trial_run(struct test_linear_card *card,
		      const struct ds41f_draft_kernels_record *kernels,
		      uint32_t index)
{
	static struct trial trial;
	static unsigned char untouched[SLOTS * STEP_ROUND_BYTES];
	static uint8_t slab_results[SLABS * DS41F_DRAFT_SLAB_RESULT_BYTES];
	trial_make(index, &trial);
	memset(untouched, ROUND_UNTOUCHED, sizeof(untouched));
	memset(slab_results, 0, sizeof(slab_results));
	for (uint32_t row = 0; row < trial.row_count; row++) /* the slabs' result areas, each with its header */
		memcpy(slab_results +
			       (size_t)((row / SLAB_ROWS) * DS41F_DRAFT_SLAB_RESULT_BYTES) +
			       STEP_RESULT_BYTES +
			       (size_t)((row % SLAB_ROWS) * STEP_ROW_RESULT_BYTES),
		       &trial.results[row], sizeof(trial.results[row]));
	test_linear_card_clear(card);
	const struct linear_range step = test_linear_card_work(
		card, STEP_BLOCK_BYTES_VALUE +
			      (uint64_t)MOST_ROWS * STEP_ROW_BYTES_VALUE);
	const struct linear_range results =
		test_linear_card_work(card, sizeof(slab_results));
	const struct linear_range table =
		test_linear_card_work(card, sizeof(untouched));
	const struct linear_range weight =
		test_linear_card_weights(card, sizeof(confidence_weights));
	struct linear_range draft_step[VERIFIERS], views[VERIFIERS],
		collapse[VERIFIERS], markov[VERIFIERS],
		draft_results[VERIFIERS];
	struct kernel_launch launches[LAUNCHES_A_VERIFIER * VERIFIERS];
	bool is_run = test_linear_card_put(card, step, 0, &trial.block,
					   sizeof(trial.block)) &&
		      test_linear_card_put(card, step, STEP_BLOCK_BYTES_VALUE,
					   trial.rows, sizeof(trial.rows)) &&
		      test_linear_card_put(card, results, 0, slab_results,
					   sizeof(slab_results)) &&
		      test_linear_card_put(card, table, 0, untouched,
					   sizeof(untouched)) &&
		      test_linear_card_put(card, weight, 0, confidence_weights,
					   sizeof(confidence_weights));
	for (uint32_t verifier = 0; verifier < VERIFIERS; verifier++) {
		draft_step[verifier] = test_linear_card_work(
			card,
			STEP_BLOCK_BYTES_VALUE +
				(uint64_t)POSITIONS * STEP_ROW_BYTES_VALUE);
		views[verifier] = test_linear_card_work(
			card, (uint64_t)POSITIONS * STEP_DRAFT_VIEW_BYTES);
		collapse[verifier] = test_linear_card_work(card, sizeof(collapse_rows[verifier]));
		markov[verifier] = test_linear_card_work(card, sizeof(markov_rows[verifier]));
		draft_results[verifier] = test_linear_card_work(
			card, (uint64_t)POSITIONS * STEP_DRAFT_RESULT_BYTES);
		const struct ds41f_verify_matches_description verify = {
			step,
			results,
			table,
			draft_step[verifier],
			views[verifier],
			MOST_ROWS,
			POSITIONS,
			END_TOKEN,
			VOCABULARY,
			verifier,
			DS41F_DRAFT_ROUND_TABLE | DS41F_DRAFT_SLAB_RESULTS,
			SLOTS,
			0
		};
		const struct ds41f_draft_confidence_description confidence = {
			draft_step[verifier],
			collapse[verifier],
			markov[verifier],
			weight,
			draft_results[verifier],
			table,
			POSITIONS,
			DS41F_DRAFT_ROUND_TABLE,
			0
		};
		/* the draft's tokens, which the confidence copies into the proposals */
		uint8_t tokens[POSITIONS * STEP_DRAFT_RESULT_BYTES];
		memset(tokens, 0, sizeof(tokens));
		for (uint32_t position = 0; position < POSITIONS; position++) {
			const struct step_row_result drawn = {
				TOKEN_BASE +
					(int32_t)(verifier * TOKEN_STRIDE + position),
				0.0f
			};
			memcpy(tokens + (size_t)(position * STEP_DRAFT_RESULT_BYTES) +
				       STEP_RESULT_BYTES,
			       &drawn, sizeof(drawn));
		}
		is_run =
			is_run && test_linear_card_has_room(card) &&
			test_linear_card_put(card, collapse[verifier], 0,
					     collapse_rows[verifier],
					     sizeof(collapse_rows[verifier])) &&
			test_linear_card_put(card, markov[verifier], 0,
					     markov_rows[verifier],
					     sizeof(markov_rows[verifier])) &&
			test_linear_card_put(card, draft_results[verifier], 0,
					     tokens, sizeof(tokens)) &&
			test_linear_expect_status(
				ds41f_verify_matches_prepare(
					&verify, card->arena_byte_count,
					kernels, &launches[verifier]),
				STATUS_OK, "a verifier is prepared") &&
			test_linear_expect_status(
				ds41f_draft_confidence_prepare(
					&confidence, card->arena_byte_count,
					kernels,
					&launches[VERIFIERS + verifier]),
				STATUS_OK, "a confidence is prepared");
	}
	is_run = is_run && test_linear_card_load(card) &&
		 test_linear_card_run(card, launches, LAUNCHES_A_VERIFIER * VERIFIERS) && test_linear_card_fetch(card);
	if (!is_run) {
		fail("the run did not finish", index, 0);
		return;
	}
	/* the reference: each drafting session alone */
	uint32_t verifier = 0;
	uint8_t expected_table[sizeof(untouched)];
	memcpy(expected_table, untouched, sizeof(untouched));
	for (uint32_t row = 0; row < trial.row_count; row++) {
		if ((trial.rows[row].flags & STEP_ROW_DRAFTS) == 0)
			continue;
		uint32_t session = 0;
		while (trial.first_row[session] != row)
			session++;
		const uint32_t count = trial.rows_of[session];
		const struct step_block alone = { trial.block.sequence, count,
						  0 };
		struct step_round round;
		struct step_block draft_block;
		struct step_row_record draft_rows[POSITIONS];
		unsigned char view_bytes[POSITIONS * STEP_DRAFT_VIEW_BYTES];
		memcpy(&round,
		       expected_table +
			       (size_t)(trial.slot_of[session] * STEP_ROUND_BYTES),
		       sizeof(round));
		memset(&draft_block, 0, sizeof(draft_block));
		memset(draft_rows, 0, sizeof(draft_rows));
		memset(view_bytes, 0, sizeof(view_bytes));
		ds41f_test_draft_verify(&alone, &trial.rows[row], &trial.results[row],
				  count, POSITIONS, END_TOKEN,
				  DS41F_DRAFT_NOISE_TOKEN, &round, &draft_block,
				  draft_rows, view_bytes);
		for (uint32_t position = 0; position < POSITIONS; position++) {
			round.proposals[position] = TOKEN_BASE + (int32_t)(verifier * TOKEN_STRIDE + position);
			round.confidence_logits[position] = ds41f_test_draft_confidence(
				collapse_rows[verifier][position], markov_rows[verifier][position], confidence_weights);
		}
		memcpy(expected_table +
			       (size_t)(trial.slot_of[session] * STEP_ROUND_BYTES),
		       &round, sizeof(round));
		if (!test_linear_card_expect_bytes(
			    card, draft_step[verifier], 0, &draft_block,
			    sizeof(draft_block), "the draft's step block") ||
		    !test_linear_card_expect_bytes(
			    card, draft_step[verifier], STEP_BLOCK_BYTES_VALUE,
			    draft_rows, sizeof(draft_rows),
			    "the draft's step rows") ||
		    !test_linear_card_expect_bytes(
			    card, views[verifier], 0, view_bytes,
			    sizeof(view_bytes), "the draft's views"))
			fail("a verifier's draft is not the reference's", index, verifier);
		verifier++;
	}
	for (; verifier < VERIFIERS; verifier++) { /* no DRAFTS row: an empty draft, no round */
		const struct step_block empty = { trial.block.sequence, 0, 0 };
		bool is_empty = test_linear_card_expect_bytes(card, draft_step[verifier], 0, &empty, sizeof(empty),
							      "an empty draft block");
		for (uint32_t position = 0; position < POSITIONS; position++) {
			const uint32_t offset = position * STEP_DRAFT_VIEW_BYTES;
			is_empty = is_empty &&
				   test_linear_card_expect_bytes(
					   card, views[verifier], offset,
					   &empty, sizeof(empty),
					   "an empty view");
		}
		if (!is_empty)
			fail("a verifier without a DRAFTS row wrote a draft", index, verifier);
	}
	if (!test_linear_card_expect_bytes(card, table, 0, expected_table, sizeof(expected_table), "the round table"))
		fail("the round table is not the reference's", index, VERIFIERS);
}

/* BATCHED (ds41f_dspark_records.h): one draft step of slab blocks and position-major views for every verifier; the
 * Markov bias of each position over every slot; each verifier's confidence from view 0's slot. */
static void trial_run_batched(struct test_linear_card *card,
			      const struct ds41f_draft_kernels_record *kernels,
			      uint32_t index)
{
	enum { DRAFT_ROWS = VERIFIERS * POSITIONS, DRAFT_SLABS = (DRAFT_ROWS + SLAB_ROWS - 1) / SLAB_ROWS };
	static struct trial trial;
	static unsigned char untouched[SLOTS * STEP_ROUND_BYTES];
	static uint8_t slab_results[SLABS * DS41F_DRAFT_SLAB_RESULT_BYTES];
	static uint8_t position_results[POSITIONS * DS41F_DRAFT_SLAB_RESULT_BYTES];
	static float logits_back[POSITIONS][BATCH_SLOTS][MARKOV_VOCABULARY];
	static uint16_t markov_back[POSITIONS][BATCH_SLOTS][RANK];
	trial_make(index, &trial);
	memset(untouched, ROUND_UNTOUCHED, sizeof(untouched));
	memset(slab_results, 0, sizeof(slab_results));
	memset(position_results, 0, sizeof(position_results));
	for (uint32_t row = 0; row < trial.row_count; row++)
		memcpy(slab_results +
			       (size_t)((row / SLAB_ROWS) * DS41F_DRAFT_SLAB_RESULT_BYTES) +
			       STEP_RESULT_BYTES +
			       (size_t)((row % SLAB_ROWS) * STEP_ROW_RESULT_BYTES),
		       &trial.results[row], sizeof(trial.results[row]));
	/* the draws of each position's sampler: slot v's token of position j, as a sampler over view j writes them */
	for (uint32_t position = 0; position < POSITIONS; position++) {
		for (uint32_t slot = 0; slot < BATCH_SLOTS; slot++) {
			const struct step_row_result drawn = {
				TOKEN_BASE + (int32_t)(slot * TOKEN_STRIDE + position),
				0.0f
			};
			memcpy(position_results +
				       (size_t)(position * DS41F_DRAFT_SLAB_RESULT_BYTES) +
				       STEP_RESULT_BYTES +
				       (size_t)(slot * STEP_ROW_RESULT_BYTES),
			       &drawn, sizeof(drawn));
		}
	}
	test_linear_card_clear(card);
	const struct linear_range step = test_linear_card_work(
		card, STEP_BLOCK_BYTES_VALUE +
			      (uint64_t)MOST_ROWS * STEP_ROW_BYTES_VALUE);
	const struct linear_range results =
		test_linear_card_work(card, sizeof(slab_results));
	const struct linear_range table =
		test_linear_card_work(card, sizeof(untouched));
	const struct linear_range draft_step = test_linear_card_work(
		card, (uint64_t)DRAFT_SLABS * DS41F_DRAFT_SLAB_STEP_BYTES);
	const struct linear_range views = test_linear_card_work(
		card, (uint64_t)POSITIONS * DS41F_DRAFT_SLAB_STEP_BYTES);
	const struct linear_range collapse =
		test_linear_card_work(card, sizeof(collapse_rows));
	const struct linear_range markov =
		test_linear_card_work(card, sizeof(markov_back));
	const struct linear_range logits =
		test_linear_card_work(card, sizeof(logits_back));
	const struct linear_range draws =
		test_linear_card_work(card, sizeof(position_results));
	const struct linear_range weight =
		test_linear_card_weights(card, sizeof(confidence_weights));
	const struct linear_range embedding =
		test_linear_card_weights(card, sizeof(w1));
	const struct linear_range head =
		test_linear_card_weights(card, sizeof(w2));
	const struct linear_range view0 = { views.byte_offset,
					    DS41F_DRAFT_SLAB_STEP_BYTES };
	struct kernel_launch launches[LAUNCHES_A_VERIFIER * VERIFIERS + POSITIONS];
	bool is_run =
		test_linear_card_has_room(card) &&
		test_linear_card_put(card, step, 0, &trial.block,
				     sizeof(trial.block)) &&
		test_linear_card_put(card, step, STEP_BLOCK_BYTES_VALUE,
				     trial.rows, sizeof(trial.rows)) &&
		test_linear_card_put(card, results, 0, slab_results,
				     sizeof(slab_results)) &&
		test_linear_card_put(card, table, 0, untouched,
				     sizeof(untouched)) &&
		test_linear_card_put(card, collapse, 0, collapse_rows,
				     sizeof(collapse_rows)) &&
		test_linear_card_put(card, logits, 0, logits_in,
				     sizeof(logits_in)) &&
		test_linear_card_put(card, draws, 0, position_results,
				     sizeof(position_results)) &&
		test_linear_card_put(card, weight, 0, confidence_weights,
				     sizeof(confidence_weights)) &&
		test_linear_card_put(card, embedding, 0, w1, sizeof(w1)) &&
		test_linear_card_put(card, head, 0, w2, sizeof(w2));
	for (uint32_t verifier = 0; verifier < VERIFIERS && is_run; verifier++) {
		const struct ds41f_verify_matches_description verify = {
			step,
			results,
			table,
			draft_step,
			views,
			MOST_ROWS,
			POSITIONS,
			END_TOKEN,
			VOCABULARY,
			verifier,
			DS41F_DRAFT_ROUND_TABLE | DS41F_DRAFT_SLAB_RESULTS | DS41F_DRAFT_BATCHED,
			SLOTS,
			DRAFT_SLABS
		};
		const struct ds41f_draft_confidence_description confidence = {
			view0,
			{ collapse.byte_offset + (uint64_t)verifier *
							 POSITIONS * HIDDEN *
							 sizeof(uint16_t),
			  (uint64_t)POSITIONS * HIDDEN * sizeof(uint16_t) },
			markov,
			weight,
			draws,
			table,
			POSITIONS,
			DS41F_DRAFT_ROUND_TABLE | DS41F_DRAFT_BATCHED,
			verifier
		};
		is_run = test_linear_expect_status(
				 ds41f_verify_matches_prepare(
					 &verify, card->arena_byte_count,
					 kernels, &launches[verifier]),
				 STATUS_OK, "a batched verifier is prepared") &&
			 test_linear_expect_status(
				 ds41f_draft_confidence_prepare(
					 &confidence, card->arena_byte_count,
					 kernels,
					 &launches[VERIFIERS + POSITIONS +
						   verifier]),
				 STATUS_OK, "a batched confidence is prepared");
	}
	for (uint32_t position = 0; position < POSITIONS && is_run; position++) {
		const struct ds41f_draft_markov_bias_description bias = {
			view0,	  draws,	embedding,
			head,	  logits,	markov,
			position, POSITIONS,	MARKOV_VOCABULARY,
			0,	  DS41F_DRAFT_BATCHED
		};
		is_run = test_linear_expect_status(
			ds41f_draft_markov_bias_prepare(
				&bias, card->arena_byte_count, kernels,
				&launches[VERIFIERS + position]),
			STATUS_OK, "a batched Markov bias is prepared");
	}
	is_run = is_run && test_linear_card_load(card) &&
		 test_linear_card_run(card, launches, LAUNCHES_A_VERIFIER * VERIFIERS + POSITIONS) &&
		 test_linear_card_fetch(card) &&
		 test_linear_card_get(card, logits, 0, logits_back, sizeof(logits_back)) &&
		 test_linear_card_get(card, markov, 0, markov_back, sizeof(markov_back));
	if (!is_run) {
		fail("the batched run did not finish", index, 0);
		return;
	}
	uint8_t expected_table[sizeof(untouched)];
	memcpy(expected_table, untouched, sizeof(untouched));
	memcpy(logits_expected, logits_in, sizeof(logits_expected));
	uint32_t verifier = 0;
	for (uint32_t row = 0; row < trial.row_count; row++) {
		if ((trial.rows[row].flags & STEP_ROW_DRAFTS) == 0)
			continue;
		uint32_t session = 0;
		while (trial.first_row[session] != row)
			session++;
		const uint32_t count = trial.rows_of[session];
		const struct step_block alone = { trial.block.sequence, count,
						  0 };
		struct step_round round;
		struct step_block draft_block;
		struct step_row_record draft_rows[POSITIONS];
		unsigned char view_bytes[POSITIONS * STEP_DRAFT_VIEW_BYTES];
		memcpy(&round,
		       expected_table +
			       (size_t)(trial.slot_of[session] * STEP_ROUND_BYTES),
		       sizeof(round));
		memset(view_bytes, 0, sizeof(view_bytes));
		ds41f_test_draft_verify(&alone, &trial.rows[row], &trial.results[row],
				  count, POSITIONS, END_TOKEN,
				  DS41F_DRAFT_NOISE_TOKEN, &round, &draft_block,
				  draft_rows, view_bytes);
		for (uint32_t position = 0; position < POSITIONS; position++) {
			const uint32_t place = verifier * POSITIONS + position;
			/* the Markov bias: the anchor's token at position 0, else the slot's draw of the position
			 * before */
			const uint32_t token =
				position == 0 ? (uint32_t)draft_rows[0].token_id :
						(uint32_t)(TOKEN_BASE +
							   (int32_t)(verifier * TOKEN_STRIDE + position - 1u));
			uint16_t markov_expected[RANK];
			ds41f_test_draft_markov_bias(w1, w2, MARKOV_VOCABULARY, token,
						     logits_expected[position][verifier],
					       markov_expected);
			round.proposals[position] = TOKEN_BASE + (int32_t)(verifier * TOKEN_STRIDE + position);
			round.confidence_logits[position] =
				ds41f_test_draft_confidence(collapse_rows[verifier][position], markov_expected,
						      confidence_weights);
			if (memcmp(markov_back[position][verifier], markov_expected, sizeof(markov_expected)) != 0)
				fail("a batched Markov row is not the reference's", index, verifier);
			if (!test_linear_card_expect_bytes(
				    card, draft_step,
				    (place / SLAB_ROWS) *
						    DS41F_DRAFT_SLAB_STEP_BYTES +
					    STEP_BLOCK_BYTES_VALUE +
					    (place % SLAB_ROWS) *
						    STEP_ROW_BYTES_VALUE,
				    &draft_rows[position],
				    sizeof(draft_rows[position]),
				    "a batched draft row") ||
			    !test_linear_card_expect_bytes(
				    card, views,
				    (uint64_t)position * DS41F_DRAFT_SLAB_STEP_BYTES +
					    STEP_BLOCK_BYTES_VALUE +
					    (uint64_t)verifier *
						    STEP_ROW_BYTES_VALUE,
				    &draft_rows[position],
				    sizeof(draft_rows[position]),
				    "a batched view's slot"))
				fail("a batched verifier's rows or views are not the reference's", index, verifier);
		}
		memcpy(expected_table +
			       (size_t)(trial.slot_of[session] * STEP_ROUND_BYTES),
		       &round, sizeof(round));
		verifier++;
	}
	const uint32_t drafting = verifier;
	if (!bytes_identical(logits_back, logits_expected, sizeof(logits_back)))
		fail("the batched Markov logits are not the reference's (slots of no verifier untouched)",
		     index, drafting);
	for (uint32_t position = 0; position < POSITIONS; position++) {
		const struct step_block header = { trial.block.sequence,
						   drafting, 0 };
		if (!test_linear_card_expect_bytes(
			    card, views,
			    (uint64_t)position * DS41F_DRAFT_SLAB_STEP_BYTES, &header,
			    sizeof(header), "a position view's header"))
			fail("a position view's header is not the reference's", index, position);
	}
	for (uint32_t slab = 0; slab < DRAFT_SLABS; slab++) {
		const uint32_t total = drafting * POSITIONS, first = slab * SLAB_ROWS;
		const struct step_block header = { trial.block.sequence,
						   total <= first ?
							   0u :
						   total - first < SLAB_ROWS ?
							   total - first :
							   SLAB_ROWS,
						   0 };
		if (!test_linear_card_expect_bytes(
			    card, draft_step,
			    (uint64_t)slab * DS41F_DRAFT_SLAB_STEP_BYTES, &header,
			    sizeof(header), "a slab block's header"))
			fail("a slab block's header is not the reference's", index, slab);
	}
	if (!test_linear_card_expect_bytes(card, table, 0, expected_table, sizeof(expected_table),
					   "the batched round table"))
		fail("the batched round table is not the reference's", index, VERIFIERS);
}

int main(int argument_count, char **arguments)
{
	uint32_t trials = DEFAULT_TRIALS, card_count = 0;
	if (argument_count > MOST_ARGUMENTS ||
	    (argument_count == MOST_ARGUMENTS && (sscanf(arguments[1], "%" SCNu32, &trials) != 1 ||
						  trials == 0 || trials > MOST_TRIALS))) {
		(void)fprintf(stderr, "usage: draft_batch_card_test [TRIALS 1..%u]\n", MOST_TRIALS);
		return TEST_WRONG_USAGE;
	}
	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	inputs_make();
	if (!test_linear_expect_status(
		    backend_cuda_value.card_count(&card_count), STATUS_OK,
		    "the cards are counted"))
		return TEST_FAILED;
	for (uint32_t card_index = 0; card_index < card_count; card_index++) {
		static struct test_linear_card card;
		struct ds41f_hyper_kernels_record hyper_kernels;
		struct router_kernels_record router_kernels;
		struct ds41f_attention_state_kernels state_kernels;
		struct ds41f_attention_core_kernels core_kernels;
		struct ds41f_draft_kernels_record kernels;
		if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
					  WEIGHTS_CAPACITY) &&
		    test_linear_expect_status(
			    ds41f_draft_kernels_prepare(
				    &backend_cuda_value, card.card,
				    card.arithmetic_class, &hyper_kernels,
				    &router_kernels, &state_kernels,
				    &core_kernels, &kernels),
			    STATUS_OK, "the draft's kernels are prepared")) {
			const uint32_t before = failures;
			for (uint32_t trial = 0; trial < trials; trial++) {
				trial_run(&card, &kernels, trial);
				trial_run_batched(&card, &kernels, trial);
			}
			(void)printf("card %" PRIu32 " (class %" PRIu32 "): %" PRIu32 " BATCH steps %s\n", card_index,
				     card.arithmetic_class, trials,
				     failures == before ? "byte for byte the reference's" : "NOT the reference's");
		} else {
			failures++;
		}
		test_linear_card_close(&card);
	}
	if (failures != 0 || test_linear_failures() != 0) {
		(void)fprintf(stderr, "draft_batch_card_test FAILED\n");
		return TEST_FAILED;
	}
	(void)printf("draft_batch_card_test passed on %" PRIu32 " cards\n", card_count);
	return TEST_PASSED;
}
