/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Prompt checkpoints over the model's actual ring/window sizes: a 5,000-token turn, dropped on the next command.
 * The copy model's period127 does not divide4096, so overwritten ring rows differ from the saved prompt window.
 * Compare every event bit and every byte of the live window with a fresh prefill, also through speculation/fronts.
 */
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/model_layout.h"
#include "serving_fixture.h"

#define CONTEXT 16384u
#define GENERATED 5000u
#define CONTINUED 32u
#define NEW_ROWS 9u
#define PROMPT_CAPACITY 8192u
#define BYTE_VALUES 256u
#define PROMPT_STRIDE 37u
#define PROMPT_TURN 101u
#define PROMPT_FIRST 11u
#define TAG 71u
#define WAIT_MILLISECONDS 10000
#define GRANULE MODEL_POOLED_RATIO
#define WINDOW_ROWS (MODEL_WINDOW - 1u)
#define LAG (MODEL_RING_SLOTS - MODEL_WINDOW)
#define PIPELINE 6u
#define CONE 2u
#define DRAFT 7u
#define FINAL_EVENTS (CONTINUED + 1u)

static int failures;
static uint32_t prompt[PROMPT_CAPACITY];
static const uint32_t script[] = { TEST_SERVING_END_TOKEN };
static const uint32_t *const scripts[] = { script };

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: prompt checkpoint: %s\n", message);
		failures++;
	}
}

static struct generation_event next_event(struct inference_engine *engine)
{
	struct generation_event event = { .event_kind =
						  GENERATION_EVENT_FAILED };
	int descriptor = -1;
	expect(engine_event_descriptor(engine, &descriptor) == STATUS_OK,
	       "event descriptor");
	struct pollfd polling = { descriptor, POLLIN, 0 };
	for (uint32_t attempts = 0; attempts < CONTEXT; attempts++) {
		bool available = false;
		if (engine_next_event(engine, &event, &available) !=
		    STATUS_OK) {
			break;
		}
		if (available) {
			return event;
		}
		if (poll(&polling, 1, WAIT_MILLISECONDS) <= 0) {
			break;
		}
	}
	expect(false, "a bounded event wait completes");
	return event;
}

static uint32_t run_command(struct inference_engine *engine, uint32_t kept,
			    uint32_t new_count, uint32_t limit,
			    struct generation_event *events)
{
	const struct generation_command_record command = {
		.new_token_ids = prompt + kept,
		.session_slot = 0,
		.kept_token_count = kept,
		.new_token_count = new_count,
		.generated_token_limit = limit,
		.sampler_top_p = 1.0f,
		.request_tag = TAG
	};
	if (engine_submit(engine, &command) != STATUS_OK) {
		expect(false, "the command is admitted");
		return 0;
	}
	for (uint32_t count = 0; count <= limit; count++) {
		const struct generation_event event = next_event(engine);
		if (events != NULL && count < FINAL_EVENTS) {
			events[count] = event;
		}
		if (event.event_kind != GENERATION_EVENT_TOKEN) {
			expect(event.event_kind ==
					       GENERATION_EVENT_FINISHED_AT_LIMIT &&
				       event.status == 0,
			       "the command reaches its exact token limit");
			return count + 1u;
		}
	}
	expect(false, "exactly one final event");
	return 0;
}

static struct inference_engine *opened(uint32_t slots, uint32_t depth,
				       uint32_t draft, uint32_t cone)
{
	struct inference_engine *engine = NULL;
	test_serving_set_engine(slots, depth, TEST_SERVING_MOST_ROWS, draft);
	test_serving_set_ring(MODEL_RING_SLOTS, MODEL_WINDOW);
	test_serving_set_function_model(true, WINDOW_ROWS);
	test_serving_set_cone(cone);
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "fixture opens");
	return engine;
}

static void test_reuse(uint32_t prompt_rows, uint32_t slots, uint32_t depth,
		       uint32_t draft, uint32_t cone)
{
	const uint32_t boundary = (prompt_rows - 1u) / GRANULE * GRANULE;
	struct generation_event reused[FINAL_EVENTS], fresh[FINAL_EVENTS];
	unsigned char window[WINDOW_ROWS];
	struct inference_engine *engine = opened(slots, depth, draft, cone);
	if (engine == NULL) {
		return;
	}
	expect(run_command(engine, 0, prompt_rows, GENERATED, NULL) ==
		       GENERATED + 1u,
	       "5,000 tokens generated");
	struct session_state state = { 0 };
	expect(engine_session_state(engine, 0, &state) == STATUS_OK &&
		       state.checkpoint_count == boundary &&
		       state.reach_count - boundary > LAG,
	       "checkpoint survives a turn beyond the ordinary lag");
	const uint32_t old_reach = state.reach_count;
	const uint64_t before = test_serving_rows_seen();
	const uint32_t reused_count =
		run_command(engine, boundary, NEW_ROWS, CONTINUED, reused);
	expect(draft != 0 || test_serving_rows_seen() - before ==
				     NEW_ROWS + CONTINUED - 1u,
	       "only new rows run, with no cone or prompt replay");
	expect(engine_session_state(engine, 0, &state) == STATUS_OK &&
		       state.reach_count >= old_reach,
	       "restore retains the conservative reach for all earlier windows");
	const uint32_t processed = boundary + NEW_ROWS + CONTINUED - 1u;
	const unsigned char *const bytes = test_serving_state();
	for (uint32_t back = 0; back < WINDOW_ROWS; back++) {
		window[back] = bytes[(processed - WINDOW_ROWS + back) %
				     MODEL_RING_SLOTS];
	}
	const struct generation_command_record earlier = {
		.new_token_ids = prompt + boundary - GRANULE,
		.kept_token_count = boundary - GRANULE,
		.new_token_count = NEW_ROWS,
		.sampler_top_p = 1.0f
	};
	expect(engine_submit(engine, &earlier) == STATUS_INVALID_ARGUMENT,
	       "an earlier uncheckpointed prefix is still refused");
	test_serving_close();
	engine = opened(slots, depth, draft, cone);
	if (engine == NULL) {
		return;
	}
	const uint32_t fresh_count =
		run_command(engine, 0, boundary + NEW_ROWS, CONTINUED, fresh);
	expect(reused_count == FINAL_EVENTS && fresh_count == reused_count &&
		       memcmp((const unsigned char *)reused,
			      (const unsigned char *)fresh,
			      sizeof(reused)) == 0,
	       "every event bit equals a fresh prefill");
	const unsigned char *const fresh_bytes = test_serving_state();
	for (uint32_t back = 0; back < WINDOW_ROWS; back++) {
		expect(window[back] ==
			       fresh_bytes[(processed - WINDOW_ROWS + back) %
					   MODEL_RING_SLOTS],
		       "every live window byte equals a fresh prefill");
	}
	test_serving_close();
}

static void test_load_discards_checkpoint(void)
{
	struct inference_engine *const engine = opened(1, PIPELINE, DRAFT, 0);
	if (engine == NULL) {
		return;
	}
	const uint32_t prompt_rows = MODEL_RING_SLOTS + CONTINUED + 1u;
	const uint32_t boundary = prompt_rows - 1u;
	(void)run_command(engine, 0, prompt_rows, GENERATED, NULL);
	struct session_state state = { 0 };
	expect(engine_session_state(engine, 0, &state) == STATUS_OK,
	       "snapshot metadata");
	char path[] = "/tmp/anemone_checkpoint_XXXXXX";
	const int descriptor = mkstemp(path);
	expect(descriptor >= 0 && unlink(path) == 0, "snapshot temporary file");
	if (descriptor >= 0) {
		struct session_command command = { .descriptor = descriptor,
						   .token_count = boundary,
						   .operation = SESSION_SAVE,
						   .request_tag = TAG };
		expect(engine_session(engine, &command) == STATUS_OK,
		       "snapshot save admitted");
		expect(next_event(engine).status == 0, "snapshot saved");
		expect(lseek(descriptor, 0, SEEK_SET) == 0, "snapshot rewound");
		command.token_ids = prompt;
		command.operation = SESSION_LOAD;
		command.reach_count = state.reach_count;
		command.decoder_from = state.decoder_from;
		expect(engine_session(engine, &command) == STATUS_OK,
		       "snapshot load admitted");
		expect(next_event(engine).status == 0, "snapshot loaded");
		expect(engine_session_state(engine, 0, &state) == STATUS_OK &&
			       state.checkpoint_count == 0,
		       "snapshot load discards the local checkpoint");
		const struct generation_command_record generation = {
			.new_token_ids = prompt + boundary,
			.kept_token_count = boundary,
			.new_token_count = NEW_ROWS,
			.sampler_top_p = 1.0f
		};
		expect(engine_submit(engine, &generation) ==
			       STATUS_INVALID_ARGUMENT,
		       "a snapshot without the checkpoint cannot claim its lag exception");
		expect(close(descriptor) == 0, "snapshot closed");
	}
	test_serving_close();
}

int main(void)
{
	static const uint32_t prompts[] = { 1030, 4129, 4130 };
	static const struct settings {
		uint32_t slots, depth, draft, cone, interleaved;
	} settings[] = { { 1, 1, 0, 0, 0 },
			 { 1, PIPELINE, DRAFT, 0, 0 },
			 { TEST_SERVING_MOST_SLOTS, PIPELINE, DRAFT, CONE, 0 },
			 { 1, PIPELINE, 0, 0, PIPELINE },
			 { TEST_SERVING_MOST_SLOTS, PIPELINE, DRAFT, 0, PIPELINE } };
	for (uint32_t index = 0; index < PROMPT_CAPACITY; index++) {
		prompt[index] = (index * PROMPT_STRIDE +
				 index / MODEL_RING_SLOTS * PROMPT_TURN +
				 PROMPT_FIRST) %
				BYTE_VALUES;
	}
	for (uint32_t setting = 0;
	     setting < sizeof(settings) / sizeof(settings[0]); setting++) {
		test_serving_set_interleaved(settings[setting].interleaved);
		for (uint32_t index = 0;
		     index < sizeof(prompts) / sizeof(prompts[0]); index++) {
			test_reuse(prompts[index], settings[setting].slots,
				   settings[setting].depth,
				   settings[setting].draft,
				   settings[setting].cone);
		}
	}
	test_serving_set_interleaved(0);
	test_load_discards_checkpoint();
	test_serving_set_cone(0);
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
	test_serving_set_ring(TEST_SERVING_STATE_POSITIONS,
			      TEST_SERVING_WINDOW);
	test_serving_set_function_model(false, 0);
	(void)printf("prompt checkpoint: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
