/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * serving_fixture.h: an engine (front_end_seam.h) over a step driver on the backend in host memory, and tables of a
 * tokenizer of bytes, for the tests of the engine and the server. Test support: no part of the engine.
 *
 * Contract:
 *   - test_serving_open resets the backend, opens one card of class 80, programs of 1 and the engine's rows, a driver
 *     (with its pipeline) and an engine named TEST_SERVING_MODEL as test_serving_set_engine last said (by default one
 *     slot, no pipeline, TEST_SERVING_ROWS rows, no speculation); test_serving_close destroys them all (NULL-safe).
 *   - The card's one launch is a stand-in model, which keeps a byte a position of each session (its state): the low
 *     byte of the token a row read there, written before the row draws, at TEST_SERVING_STATE_POSITIONS positions a
 *     slot, or as test_serving_set_ring last said (a position beyond takes the slot of its remainder: a ring). Each
 *     step first sleeps the step delay, and is counted. A row that asks for logits draws, with the row's position as
 *     its logit:
 *       the scripts (the default; one slot): a step that begins at position 0 begins its script again; each row draws
 *       the next token of the script the session's seed names (scripts[seed % count], read up to
 *       TEST_SERVING_END_TOKEN), the end token past its end. The scripts (one at least, whatever the model) must
 *       outlive the engine.
 *       the function model (test_serving_set_function_model): a function of the session's state and settings alone,
 *       so that any steps that read the same tokens draw the same: with a copy period, the byte the state holds that
 *       many positions before the next position (0 before position 0); else a hash of the seed (where the temperature
 *       is above 0), the position and the state's bytes at the row's position and the TEST_SERVING_WINDOW - 1 (or
 *       test_serving_set_ring's window - 1) before it (0 before position 0), which is the end token one time in
 *       TEST_SERVING_END_ODDS, else a byte.
 *   - test_serving_set_cone, before an open: its programs carry a front (model_seam.h) of one launch before the
 *     model's, which follows the rows' positions for the scripts and computes nothing, and a cone of cone_rows rows,
 *     which the engine's description names (0, the default: no front). test_serving_rows_seen counts the rows the
 *     model's launch read since the open.
 *   - A snapshot holds a session's state, a byte a token.
 *   - test_serving_tables: the 256 byte tokens, then TEST_SERVING_REASONING_END_TOKEN "</think>",
 *     TEST_SERVING_END_TOKEN "<END>" and TEST_SERVING_REASONING_BEGIN_TOKEN "<think>": no merge, one block of no
 *     class. The caller frees them.
 *   - test_serving_set_interleaved, before an open: with a depth (2 to STEP_PIPELINE_MOST) the programs run
 *     on two cards, card 0 visited twice a step (model_seam.h's visits) in a pipeline of that depth: card 0 puts each
 *     row's token plus position down, card 1 holds them to its own rows and adds one, card 0's model holds what comes
 *     back to its rows of the step (a visit that read another step's rows faults); no front. 0, the default: the one
 *     card. test_serving_set_fault_position: the model faults on a row at that position (UINT32_MAX: none; an open
 *     resets it), so the card fails and the step with it.
 */
#ifndef TEST_SERVING_FIXTURE_H
#define TEST_SERVING_FIXTURE_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/front_end_seam.h"
#include "anemone/status.h"

#define TEST_SERVING_ROWS 4u
#define TEST_SERVING_MOST_ROWS 8u /* a thin step's: LINEAR_MAXIMUM_ROWS */
#define TEST_SERVING_MOST_SLOTS 4u
#define TEST_SERVING_STATE_POSITIONS 32768u
#define TEST_SERVING_WINDOW 4u                          /* state bytes a draw of the function model reads */
#define TEST_SERVING_END_ODDS 64u
#define TEST_SERVING_MODEL "test-model"
#define TEST_SERVING_REASONING_END_TOKEN 256u
#define TEST_SERVING_END_TOKEN 257u
#define TEST_SERVING_REASONING_BEGIN_TOKEN 258u
#define TEST_SERVING_VOCABULARY 259u

MUST_CHECK result_status
test_serving_open(uint32_t context_tokens, const uint32_t *const *scripts,
		  uint32_t script_count, struct inference_engine **engine_out);
void test_serving_close(void);
/* The model's state on the card: the bytes of slot 0's positions (a snapshot's range, open's context). */
const unsigned char *test_serving_state(void);
void test_serving_set_cone(uint32_t cone_rows);
/* Before an open, kept until set again: slot_count slots (1 to TEST_SERVING_MOST_SLOTS), a pipeline of depth steps
 * (1: none), programs of 1 and rows rows (2 to TEST_SERVING_MOST_ROWS), draft_positions proposals a round at most
 * (front_end_seam.h; 0: no speculation) of n-grams of 4 down to 2 tokens. */
void test_serving_set_engine(uint32_t slot_count, uint32_t depth, uint32_t rows, uint32_t draft_positions);
/* Before open: description.ignore_eos, including invalid values for admission controls. */
void test_serving_set_ignore_eos(uint32_t ignore_eos);
/* Before open: a device draft on the result card. On the function model it returns exact next proposals, except
 * rejected_position (0..4; 5: all right). Its separate persistent ring changes the snapshot identity. */
void test_serving_set_device_block(uint32_t positions);
void test_serving_set_device_draft(bool enabled, uint32_t rejected_position);
/* Before open: one confidence score is negative (5: none); or report a structurally valid wrong emitted count on a
 * round with proposals, testing the engine's authoritative keyed-match check before publishing any flight token. */
void test_serving_set_device_confidence(uint32_t low_position);
/* Before an open, kept until set again: the description's prefill_first, prefill_order and prefill_order_wait
 * (front_end_seam.h; 0, 0, 0 by default). */
void test_serving_set_prefill(uint32_t prefill_first, uint32_t prefill_order, uint32_t prefill_order_wait);
/* Load-time policy override; NULL restores defaults, lookup_min 0 disables hybrid. */
void test_serving_set_draft_policy(const struct draft_policy_record *policy,
				   uint32_t lookup_min);
/* Before open, kept until set again: the engine's draft_learned_acceptance. */
void test_serving_set_learned_acceptance(bool learned);
/* Before open, kept until set again: the engine's draft_session_acceptance (its level). */
void test_serving_set_session_acceptance(uint32_t level);
/* Before open, kept until set again: the engine's draft_start_economic. */
void test_serving_set_start_economic(bool economic);
void test_serving_set_device_round_fault(bool wrong_emitted);
uint64_t test_serving_device_rounds(void);
uint32_t test_serving_device_rejections(void);
/* Before an open, kept until set again: the function model (with copy_period 0 its hash, else the copy) or the
 * scripts. */
void test_serving_set_function_model(bool is_function_model, uint32_t copy_period);
/* Before open: force the function model's end token at this input position (UINT32_MAX: no forced end). */
void test_serving_set_function_end(uint32_t position);
/* Before an open, kept until set again: the state a ring of positions slots (at most TEST_SERVING_STATE_POSITIONS),
 * the function model's draw reading window bytes of it (the model's are 4,096 and 128). */
void test_serving_set_ring(uint32_t positions, uint32_t window);
void test_serving_set_interleaved(uint32_t depth);
void test_serving_set_fault_position(uint32_t position);
uint64_t test_serving_rows_seen(void);
uint64_t test_serving_front_rows_seen(void);
void test_serving_set_step_delay(uint64_t nanoseconds);
uint64_t test_serving_step_count(void);
void *test_serving_tables(uint64_t *byte_count_out);

#endif
