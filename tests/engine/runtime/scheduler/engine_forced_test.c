/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engine_forced_test: engine_submit_forced (front_end_seam.h) over serving_fixture.h's function model, whose draws
 * follow the session's state alone and whose head writes its logits: on the serial path, through a pipeline and in a
 * speculative engine, the events carry the forced tokens to the limit, an end token among them ending nothing; each
 * landed drawing row reaches rows_landed once, in order, before the final event, with its position, its input token
 * (the prompt's last, then the forced token before it), the step's rows and the row's place, and the logits the step
 * left on the card (read there with step_arena_read: argmax the draw, the draw's logit the result's); the card's state
 * is the prompt and the forced tokens but the last; every path gives the same draws and logits; a failing rows_landed
 * fails the command with its status; every refusal queues nothing; and an ordinary command after a forced one gives the
 * events it gives on a fresh engine. Run under the thread sanitizer with the others.
 */
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "engine/backend/backend_seam.h"
#include "engine/input/image/request_images.h"
#include "engine/runtime/scheduler/front_end_seam.h"
#include "engine/runtime/step/step.h"
#include "engine/runtime/step/step_images.h"
#include "engine/program/step_records.h"
#include "engine/input/tokenizer/tokenizer.h"
#include "support/byte_identity.h"
#include "engine/runtime/scheduler/scheduler_internal.h" /* the driver the rows_landed sink reads the logits from */
#include "models/deepseek_v41_flash/plan/model_v41.h" /* the plan record step_images_attach takes */
#include "engine/runtime/scheduler/serving_cursor.h"
#include "support/serving_fixture.h"

#define CONTEXT 256u
#define PROMPT 5u
#define FORCED 40u
#define END_AT 2u /* a forced end token there ends nothing */
#define FORCED_STRIDE \
	37u /* forced token k: (k * stride + first) % 256, mostly not the draw */
#define FORCED_FIRST 11u
#define FORCED_TOKEN_RANGE 256u /* the byte tokens */
#define PROMPT_STRIDE 13u
#define MOST_EVENTS 256u
#define POLL_MILLISECONDS 5000
#define LAST_ROW_PAUSE_NS \
	50000000L /* the last rows_landed waits: a final event before it would be taken */
#define FAILING_AT 3u
/* the tokens up to the failing row's and the FAILED event */
#define FAILED_EVENTS (FAILING_AT + 2u)
#define PIPELINE_DEPTH 2u
#define SPECULATIVE_ROWS 8u
#define DRAFT_DEPTH 4u
#define ROWS_ABOVE_DEPTH \
	(DRAFT_DEPTH + 2u) /* one above speculative_depth + 1 */
#define TWO_SLOTS 2u
#define ORDINARY_LIMIT 200u
#define ORDINARY_SEED 5u
#define NO_ROW UINT32_MAX
#define WRONG_TOP_K 5u
#define WRONG_TEMPERATURE 0.5f
#define WRONG_TOP_P 0.9f
#define CLOSING_BUDGET \
	1u /* a reasoning budget whose first token closes it: no draw's bytes read */
#define WRONG_ROWS 2u /* rows above 1 without speculation */
#define BEYOND_CONTEXT 2u /* new tokens past the room the context leaves */
#define DEVICE_RIGHT 5u /* test_serving_set_device_draft: no proposal wrong */

enum path { SERIAL, PIPELINE, SPECULATIVE, PATHS };
/* engine_submit_forced's refusals, a command each */
enum refusal {
	TEMPERATURE,
	TOP_P,
	TOP_K,
	NO_NEW_TOKENS,
	OTHER_LIMIT,
	NO_TOKENS,
	NO_CALLBACK,
	NO_ROWS,
	ROWS_WITHOUT_SPECULATION,
	BEYOND_THE_CONTEXT,
	REFUSALS
};
/* engine_submit_forced's own refusals of commands engine_submit runs */
enum forced_only { KEPT, BUDGET, IMAGES, BEYOND_LIMIT, FORCED_ONLY };

static int failures = 0;
static const char *context_text = "";
static uint32_t prompt[PROMPT];
/* one beyond: a read past the continuation fails the checks */
static uint32_t forced_tokens[FORCED + 1u];
static const uint32_t no_end[] = { TEST_SERVING_END_TOKEN };
static const uint32_t *const scripts[] = { no_end };
/* What rows_landed saw, a forced position k each: the paths' rows are held to each other's. */
static struct sink {
	struct inference_engine *engine;
	uint32_t calls, positions, fail_at, pause_at;
	uint32_t position[FORCED], input[FORCED], drawn[FORCED];
	uint32_t step_rows[FORCED], row_in_step[FORCED];
	float logits[FORCED][TEST_SERVING_VOCABULARY];
	bool logits_hold, rows_hold;
} sink, reference;

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)printf("FAILED: %s: %s\n", context_text, what);
		failures++;
	}
}

/* The capture's side: each drawing row of the command, read from the card the step left it on. */
MUST_CHECK static result_status
rows_landed(void *context, const struct step_row_record *rows,
	    const struct step_row_result *results, uint32_t step_row_count,
	    uint32_t first_row, uint32_t row_count)
{
	struct sink *const own = context;
	static float read[TEST_SERVING_MOST_ROWS][TEST_SERVING_VOCABULARY];
	const result_status status =
		row_count == 0 || row_count > TEST_SERVING_MOST_ROWS ||
				first_row + row_count > step_row_count ?
			STATUS_INVALID_ARGUMENT :
			step_arena_read(
				own->engine->driver, 0,
				TEST_SERVING_LOGITS_OFFSET +
					(uint64_t)first_row * sizeof(read[0]),
				(uint64_t)row_count * sizeof(read[0]), read);
	if (own->positions + row_count == own->pause_at) {
		const struct timespec pause = { 0, LAST_ROW_PAUSE_NS };
		(void)nanosleep(&pause, NULL);
	}
	own->calls++;
	for (uint32_t row = 0; row < row_count && status == STATUS_OK; row++) {
		const uint32_t k = own->positions;
		if (k == own->fail_at || k >= FORCED)
			return STATUS_INPUT_OUTPUT_ERROR;
		const struct step_row_record *const step_row =
			&rows[first_row + row];
		const struct step_row_result *const result =
			&results[first_row + row];
		uint32_t argmax = 0;
		for (uint32_t column = 1; column < TEST_SERVING_VOCABULARY;
		     column++)
			argmax = read[row][column] > read[row][argmax] ?
					 column :
					 argmax;
		own->position[k] = (uint32_t)step_row->position;
		own->input[k] = (uint32_t)step_row->token_id;
		own->drawn[k] = (uint32_t)result->token_id;
		own->step_rows[k] = step_row_count;
		own->row_in_step[k] = first_row + row;
		memcpy(own->logits[k], read[row], sizeof(read[row]));
		own->logits_hold = own->logits_hold &&
				   argmax == (uint32_t)result->token_id &&
				   read[row][argmax] == result->logit;
		own->rows_hold = own->rows_hold &&
				 (step_row->flags & STEP_ROW_LOGITS) != 0 &&
				 own->position[k] == PROMPT - 1u + k &&
				 own->input[k] ==
					 (k == 0 ? prompt[PROMPT - 1u] :
						   forced_tokens[k - 1u]);
		own->positions++;
	}
	return status;
}

/* Every event until the final one; the count taken. */
static uint32_t events_until_final(struct inference_engine *engine,
				   struct generation_event *events,
				   uint32_t *positions_at_final)
{
	uint32_t count = 0;
	int descriptor = -1;
	expect(engine_event_descriptor(engine, &descriptor) == STATUS_OK,
	       "the descriptor");
	while (count < MOST_EVENTS) {
		struct pollfd poll_entry = { descriptor, POLLIN, 0 };
		bool is_available = false;
		if (poll(&poll_entry, 1, POLL_MILLISECONDS) != 1) {
			expect(false, "an event comes within the poll's time");
			return count;
		}
		expect(engine_next_event(engine, &events[count],
					 &is_available) == STATUS_OK,
		       "next_event");
		if (is_available &&
		    events[count++].event_kind != GENERATION_EVENT_TOKEN) {
			*positions_at_final = sink.positions;
			return count;
		}
	}
	return count;
}

static struct generation_command_record command_of(uint32_t limit)
{
	return (struct generation_command_record){ .new_token_ids = prompt,
						   .new_token_count = PROMPT,
						   .generated_token_limit =
							   limit,
						   .sampler_top_p = 1.0f,
						   .request_tag = 1u };
}

static struct engine_forced_continuation forced_of(uint32_t rows)
{
	return (struct engine_forced_continuation){ forced_tokens, FORCED, rows,
						    rows_landed, &sink };
}

static struct inference_engine *opened(enum path path)
{
	struct inference_engine *engine = NULL;
	test_serving_set_engine(1, path == PIPELINE ? PIPELINE_DEPTH : 1u,
				path == SPECULATIVE ? SPECULATIVE_ROWS :
						      TEST_SERVING_ROWS,
				path == SPECULATIVE ? DRAFT_DEPTH : 0u);
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "the engine opens");
	return engine;
}

/* One forced command of rows drawing rows a step; its events and sink held to the contract. */
static void forced_run(struct inference_engine *engine, uint32_t rows)
{
	static struct generation_event events[MOST_EVENTS];
	const struct generation_command_record command = command_of(FORCED);
	const struct engine_forced_continuation forced = forced_of(rows);
	uint32_t positions_at_final = 0;
	sink = (struct sink){ .engine = engine,
			      .fail_at = NO_ROW,
			      .pause_at = FORCED,
			      .logits_hold = true,
			      .rows_hold = true };
	expect(engine_submit_forced(engine, &command, &forced) == STATUS_OK,
	       "the forced command is submitted");
	const uint32_t count =
		events_until_final(engine, events, &positions_at_final);
	expect(count == FORCED + 1u &&
		       events[FORCED].event_kind ==
			       GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       events[FORCED].session_token_count == PROMPT + FORCED,
	       "the limit alone ends it, the end token among the forced ones not");
	bool tokens_hold = true;
	for (uint32_t k = 0; k < FORCED && k < count; k++)
		tokens_hold = tokens_hold &&
			      events[k].event_kind == GENERATION_EVENT_TOKEN &&
			      events[k].token_id == forced_tokens[k] &&
			      events[k].session_token_count == PROMPT + 1u + k;
	expect(tokens_hold, "every token event carries the forced token");
	expect(sink.positions == FORCED && positions_at_final == FORCED,
	       "every forced row reached rows_landed before the final event");
	expect(sink.rows_hold,
	       "each row's position and input token: the prompt's last, then the forced token before it");
	expect(sink.logits_hold,
	       "the logits read from the card: argmax the draw, the draw's logit the result's");
	const unsigned char *const state = test_serving_state();
	bool state_holds = true;
	for (uint32_t position = 0; position + 1u < PROMPT + FORCED; position++)
		state_holds =
			state_holds &&
			state[position] ==
				(unsigned char)(position < PROMPT ?
							prompt[position] :
							forced_tokens[position -
								      PROMPT]);
	expect(state_holds,
	       "the card's state holds the prompt and the forced tokens but the last");
	bool any_differs = false;
	for (uint32_t k = 0; k < FORCED; k++)
		any_differs = any_differs || sink.drawn[k] != forced_tokens[k];
	expect(any_differs, "the draws are the model's, not the forced tokens");
}

static uint32_t drafter_calls;

/* A drafter whose proposals are never drawn: none may reach a forced row. */
static uint32_t wrong_drafter(struct inference_engine *engine,
			      const struct command_cursor *cursor,
			      uint32_t anchor_position, uint32_t most,
			      uint32_t *proposals_out)
{
	(void)engine;
	(void)cursor;
	(void)anchor_position;
	drafter_calls++;
	for (uint32_t index = 0; index < most; index++)
		proposals_out[index] = TEST_SERVING_REASONING_END_TOKEN;
	return most;
}

static bool sinks_equal(const struct sink *one, const struct sink *other)
{
	return memcmp(one->position, other->position, sizeof(one->position)) ==
		       0 &&
	       memcmp(one->input, other->input, sizeof(one->input)) == 0 &&
	       memcmp(one->drawn, other->drawn, sizeof(one->drawn)) == 0 &&
	       bytes_identical(one->logits, other->logits, sizeof(one->logits));
}

static void test_paths(void)
{
	static const char *const names[PATHS] = {
		"forced, serial path", "forced, a pipeline of two",
		"forced, one row a step in a speculative engine"
	};
	for (uint32_t path = 0; path < PATHS; path++) {
		context_text = names[path];
		struct inference_engine *const engine = opened(path);
		if (engine != NULL)
			forced_run(engine, 1u);
		if (path == SERIAL)
			reference = sink;
		else
			expect(sinks_equal(&sink, &reference),
			       "the same draws and logits as the serial path");
		expect(sink.calls == FORCED,
		       "one row a step: a call a position");
		for (uint32_t k = 1; k < FORCED; k++)
			expect(sink.step_rows[k] == 1u,
			       "a forced row behind the prompt goes alone");
		test_serving_close();
	}
}

/* Verify rows: a generated token's row and the next forced tokens in one step, rows of them, all accepted, the prompt's
 * last row alone; the drafter never asked, no draft run; the rows the reference's. */
static void test_verify_rows(void)
{
	static const uint32_t shapes[] = { 2u, 4u, DRAFT_DEPTH + 1u };
	for (uint32_t device = 0; device <= 1u; device++)
		for (uint32_t shape = 0;
		     shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
			const uint32_t rows = shapes[shape];
			context_text =
				device == 0 ?
					"verify rows, a scripted drafter" :
					"verify rows, a device draft";
			test_serving_set_device_draft(device != 0,
						      DEVICE_RIGHT);
			struct inference_engine *const engine =
				opened(SPECULATIVE);
			drafter_calls = 0;
			const uint64_t device_rounds =
				test_serving_device_rounds();
			if (engine != NULL && device == 0)
				engine->draft = wrong_drafter;
			static struct draft_statistics before, after;
			expect(engine != NULL && engine_draft_statistics(
							 engine, 0, NULL,
							 &before) == STATUS_OK,
			       "the draft statistics before");
			if (engine != NULL)
				forced_run(engine, rows);
			expect(sinks_equal(&sink, &reference),
			       "the reference's draws and logits");
			expect(engine != NULL &&
				       engine_draft_statistics(engine, 0, NULL,
							       &after) ==
					       STATUS_OK &&
				       memcmp(&before, &after,
					      sizeof(before)) == 0,
			       "no draft statistic counts");
			/* read after the final event, which the stepping thread queued after its last write */
			const struct command_cursor *const cursor =
				engine != NULL ? &engine->cursors[0] : NULL;
			expect(cursor != NULL && cursor->rounds == 0 &&
				       cursor->proposals == 0 &&
				       cursor->accepted == 0 &&
				       cursor->depth == 1u &&
				       cursor->cooldown == 0,
			       "no round, depth or cooldown counts");
			expect(drafter_calls == 0 &&
				       test_serving_device_rounds() ==
					       device_rounds,
			       "no drafter asked, no draft run");
			expect(sink.calls ==
				       1u + (FORCED - 1u + rows - 1u) / rows,
			       "a step of rows drawing rows a call");
			expect(sink.step_rows[0] == PROMPT &&
				       sink.row_in_step[0] == PROMPT - 1u,
			       "the prompt's last row draws alone, in the prompt's step");
			bool shaped = true;
			for (uint32_t k = 1; k < FORCED; k++) {
				const uint32_t first = k - (k - 1u) % rows;
				const uint32_t count = FORCED - first < rows ?
							       FORCED - first :
							       rows;
				shaped = shaped && sink.step_rows[k] == count &&
					 sink.row_in_step[k] == k - first;
			}
			expect(shaped,
			       "each verify step: an anchor and the forced tokens behind it");
			static struct generation_event events[MOST_EVENTS];
			const struct generation_command_record ordinary =
				command_of(ORDINARY_LIMIT);
			uint32_t unused = 0;
			expect(engine != NULL &&
				       engine_submit(engine, &ordinary) ==
					       STATUS_OK &&
				       events_until_final(engine, events,
							  &unused) > 1u,
			       "an ordinary command after it");
			expect(device == 0 ? drafter_calls != 0 :
					     test_serving_device_rounds() !=
						     device_rounds,
			       "which its drafter serves");
			const struct engine_forced_continuation deep =
				forced_of(ROWS_ABOVE_DEPTH);
			const struct generation_command_record command =
				command_of(FORCED);
			expect(engine != NULL &&
				       engine_submit_forced(engine, &command,
							    &deep) ==
					       STATUS_INVALID_ARGUMENT,
			       "rows above speculative_depth + 1 refused");
			test_serving_close();
		}
	test_serving_set_device_draft(false, DEVICE_RIGHT);
}

/* A failing rows_landed fails the command with its status, and a forced command after it runs whole. */
static void test_sink_failure(void)
{
	static struct generation_event events[MOST_EVENTS];
	context_text = "a failing rows_landed";
	struct inference_engine *const engine = opened(PIPELINE);
	const struct generation_command_record command = command_of(FORCED);
	const struct engine_forced_continuation forced = forced_of(1u);
	uint32_t positions_at_final = 0;
	sink = (struct sink){ .engine = engine,
			      .fail_at = FAILING_AT,
			      .pause_at = NO_ROW,
			      .logits_hold = true,
			      .rows_hold = true };
	expect(engine != NULL && engine_submit_forced(engine, &command,
						      &forced) == STATUS_OK,
	       "the forced command is submitted");
	const uint32_t count = engine != NULL ?
				       events_until_final(engine, events,
							  &positions_at_final) :
				       0;
	expect(count == FAILED_EVENTS &&
		       events[count - 1u].event_kind ==
			       GENERATION_EVENT_FAILED &&
		       events[count - 1u].status == STATUS_INPUT_OUTPUT_ERROR,
	       "its status in the FAILED event, after the tokens up to it");
	if (engine != NULL)
		forced_run(engine, 1u);
	expect(sinks_equal(&sink, &reference),
	       "the next forced command gives the reference's rows");
	test_serving_close();
}

/* Every refusal queues nothing: the valid command after them runs whole. */
static void test_refusals(void)
{
	context_text = "forced refusals";
	struct inference_engine *engine = opened(SERIAL);
	struct generation_command_record commands[REFUSALS];
	struct engine_forced_continuation forced[REFUSALS];
	for (uint32_t index = 0; index < REFUSALS; index++) {
		commands[index] =
			command_of(index == OTHER_LIMIT ? FORCED - 1u : FORCED);
		forced[index] = forced_of(1u);
	}
	commands[TEMPERATURE].sampler_temperature = WRONG_TEMPERATURE;
	commands[TOP_P].sampler_top_p = WRONG_TOP_P;
	commands[TOP_K].sampler_top_k = WRONG_TOP_K;
	commands[NO_NEW_TOKENS].new_token_count = 0; /* engine_submit's own */
	forced[NO_TOKENS].token_ids = NULL;
	forced[NO_CALLBACK].rows_landed = NULL;
	forced[NO_ROWS].rows = 0;
	forced[ROWS_WITHOUT_SPECULATION].rows = WRONG_ROWS;
	commands[BEYOND_THE_CONTEXT].new_token_count =
		CONTEXT - FORCED + BEYOND_CONTEXT;
	static uint32_t long_prompt[CONTEXT];
	commands[BEYOND_THE_CONTEXT].new_token_ids = long_prompt;
	for (uint32_t index = 0; engine != NULL && index < REFUSALS; index++)
		expect(engine_submit_forced(engine, &commands[index],
					    &forced[index]) ==
			       STATUS_INVALID_ARGUMENT,
		       "refused");
	const struct generation_command_record command = command_of(FORCED);
	expect(engine != NULL && engine_submit_forced(engine, &command, NULL) ==
					 STATUS_INVALID_ARGUMENT,
	       "no continuation");
	forced_tokens[FORCED - 1u] = TEST_SERVING_VOCABULARY;
	struct engine_forced_continuation outside = forced_of(1u);
	expect(engine != NULL &&
		       engine_submit_forced(engine, &command, &outside) ==
			       STATUS_INVALID_ARGUMENT,
	       "a token outside the vocabulary");
	forced_tokens[FORCED - 1u] = FORCED_FIRST;
	struct generation_command_record fits = command_of(FORCED);
	fits.new_token_count = CONTEXT - FORCED + 1u;
	fits.new_token_ids = long_prompt;
	sink = (struct sink){ .engine = engine,
			      .fail_at = NO_ROW,
			      .pause_at = NO_ROW,
			      .logits_hold = true,
			      .rows_hold = true };
	expect(engine != NULL && engine_submit_forced(engine, &fits,
						      &outside) == STATUS_OK,
	       "new tokens and the forced ones but the last fill the context");
	static struct generation_event events[MOST_EVENTS];
	uint32_t positions_at_final = 0;
	const uint32_t count = engine != NULL ?
				       events_until_final(engine, events,
							  &positions_at_final) :
				       0;
	expect(count == FORCED + 1u && sink.positions == FORCED &&
		       events[FORCED].event_kind ==
			       GENERATION_EVENT_FINISHED_AT_LIMIT,
	       "and it runs to its limit");
	if (engine != NULL)
		forced_run(engine, 1u);
	expect(sinks_equal(&sink, &reference),
	       "nothing refused was queued: the next command is the reference");
	test_serving_close();
	test_serving_set_engine(TWO_SLOTS, 1, TEST_SERVING_ROWS, 0);
	engine = NULL;
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "an engine of two slots opens");
	const struct engine_forced_continuation one = forced_of(1u);
	expect(engine != NULL && engine_submit_forced(engine, &command, &one) ==
					 STATUS_INVALID_ARGUMENT,
	       "an engine of more than one slot");
	test_serving_close();
}

/* engine_submit_forced's own refusals, each of a command engine_submit takes and runs to its end: on a serial engine
 * whose slot holds a prior command's tokens, whose driver takes images (attached without a tower: an input of no image
 * runs as text) and with the byte tables' tokenizer, a kept count (even, within the slot's reach), a reasoning budget,
 * images and forced tokens beyond the context (new tokens that fit) are refused, queuing nothing. */
static void test_forced_only_refusals(void)
{
	static const struct chat_images no_image; /* an input of no image */
	static uint32_t beyond_tokens[CONTEXT + BEYOND_CONTEXT];
	static uint32_t long_prompt[CONTEXT];
	static struct generation_event events[MOST_EVENTS];
	static const char *const names[FORCED_ONLY] = {
		"a kept count", "a reasoning budget", "images",
		"forced tokens beyond the context"
	};
	context_text = "forced-only refusals";
	struct inference_engine *const engine = opened(SERIAL);
	struct ds41f_plan *const plan = calloc(1, sizeof(*plan));
	uint64_t table_bytes = 0;
	void *const tables = test_serving_tables(&table_bytes);
	struct text_tokenizer *tokenizer = NULL;
	if (plan != NULL) {
		plan->limits.session_capacity = 1;
		plan->vision.enabled = true;
	}
	uint32_t unused = 0;
	const struct generation_command_record prior = command_of(FORCED);
	const bool is_ready =
		engine != NULL && plan != NULL && tables != NULL &&
		tokenizer_open(tables, table_bytes, &tokenizer) == STATUS_OK &&
		step_images_attach(engine->driver, plan,
				   ARITHMETIC_CLASS_SM_80) == STATUS_OK &&
		step_images_supported(engine->driver) &&
		engine_submit(engine, &prior) == STATUS_OK &&
		events_until_final(engine, events, &unused) != 0;
	expect(is_ready,
	       "a driver that takes images, the tokenizer, a prior command");
	struct generation_command_record commands[FORCED_ONLY];
	struct engine_forced_continuation forced[FORCED_ONLY];
	for (uint32_t index = 0; index < FORCED_ONLY; index++) {
		commands[index] = command_of(FORCED);
		forced[index] = forced_of(1u);
	}
	commands[BUDGET].reasoning_budget = CLOSING_BUDGET;
	commands[BUDGET].reasoning_end_token = TEST_SERVING_REASONING_END_TOKEN;
	commands[BUDGET].reasoning_tokenizer = tokenizer;
	commands[IMAGES].images = &no_image;
	commands[BEYOND_LIMIT].new_token_ids = long_prompt;
	commands[BEYOND_LIMIT].new_token_count = CONTEXT - 1u;
	commands[BEYOND_LIMIT].generated_token_limit = CONTEXT + BEYOND_CONTEXT;
	forced[BEYOND_LIMIT].token_ids = beyond_tokens;
	forced[BEYOND_LIMIT].token_count = CONTEXT + BEYOND_CONTEXT;
	/* the sink of this engine, should a refusal let a forced command through */
	sink = (struct sink){ .engine = engine,
			      .fail_at = NO_ROW,
			      .pause_at = NO_ROW,
			      .logits_hold = true,
			      .rows_hold = true };
	for (uint32_t index = 0; is_ready && index < FORCED_ONLY; index++) {
		if (index == KEPT) {
			/* the slot's processed tokens, even, the prompt after them (it holds the prior command's) */
			(void)pthread_mutex_lock(&engine->lock);
			commands[KEPT].kept_token_count =
				engine->slots[0].processed_count /
				SERVING_KEPT_GRANULE * SERVING_KEPT_GRANULE;
			(void)pthread_mutex_unlock(&engine->lock);
			expect(commands[KEPT].kept_token_count != 0,
			       "the slot holds tokens to keep");
		}
		context_text = names[index];
		expect(engine_submit_forced(engine, &commands[index],
					    &forced[index]) ==
			       STATUS_INVALID_ARGUMENT,
		       "engine_submit_forced refuses it");
		const uint32_t count =
			engine_submit(engine, &commands[index]) == STATUS_OK ?
				events_until_final(engine, events, &unused) :
				0;
		expect(count != 0 &&
			       (events[count - 1u].event_kind ==
					GENERATION_EVENT_FINISHED_AT_LIMIT ||
				events[count - 1u].event_kind ==
					GENERATION_EVENT_FINISHED_AT_END_TOKEN),
		       "engine_submit runs it to its end");
	}
	test_serving_close();
	tokenizer_close(tokenizer);
	free(tables);
	free(plan);
}

/* An ordinary command after a forced one gives what it gives on a fresh engine. */
static void test_ordinary_after(void)
{
	static struct generation_event fresh[MOST_EVENTS], after[MOST_EVENTS];
	uint32_t unused = 0;
	struct generation_command_record command = command_of(ORDINARY_LIMIT);
	command.sampler_seed = ORDINARY_SEED;
	for (uint32_t path = 0; path < PATHS; path++) {
		context_text = "an ordinary command after a forced one";
		struct inference_engine *engine = opened(path);
		expect(engine != NULL &&
			       engine_submit(engine, &command) == STATUS_OK,
		       "the ordinary command on a fresh engine");
		const uint32_t fresh_count =
			engine != NULL ?
				events_until_final(engine, fresh, &unused) :
				0;
		test_serving_close();
		engine = opened(path);
		if (engine != NULL)
			forced_run(engine, 1u);
		expect(engine != NULL &&
			       engine_submit(engine, &command) == STATUS_OK,
		       "the ordinary command after the forced one");
		const uint32_t after_count =
			engine != NULL ?
				events_until_final(engine, after, &unused) :
				0;
		expect(fresh_count == after_count &&
			       memcmp(fresh, after,
				      fresh_count * sizeof(fresh[0])) == 0,
		       "the same events, its end token ending it");
		expect(fresh_count > 1u &&
			       fresh[fresh_count - 1u].event_kind ==
				       GENERATION_EVENT_FINISHED_AT_END_TOKEN,
		       "an ordinary command still ends at the end token");
		test_serving_close();
	}
}

int main(void)
{
	(void)setvbuf(stdout, NULL, _IOLBF, 0);
	for (uint32_t index = 0; index < PROMPT; index++)
		prompt[index] = (index * PROMPT_STRIDE + 1u) % UINT8_MAX;
	for (uint32_t k = 0; k < FORCED; k++)
		forced_tokens[k] =
			(k * FORCED_STRIDE + FORCED_FIRST) % FORCED_TOKEN_RANGE;
	forced_tokens[END_AT] = TEST_SERVING_END_TOKEN;
	test_serving_set_function_model(true, 0);
	test_paths();
	test_verify_rows();
	test_sink_failure();
	test_refusals();
	test_forced_only_refusals();
	test_ordinary_after();
	test_serving_set_function_model(false, 0);
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
	if (failures != 0) {
		(void)printf("engine_forced_test: %d failures\n", failures);
		return EXIT_FAILURE;
	}
	(void)printf("engine_forced_test: passed\n");
	return EXIT_SUCCESS;
}
