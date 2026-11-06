/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/live_stats.h"
#include "json.h"

#define SAMPLE_NOW_NS 4000000000ull
#define SAMPLE_START_NS 1000000000ull
#define SAMPLE_SUMMARY_END_NS 3000000000ull
#define SAMPLE_FIRST_TOKEN_NS 2000000000ull
#define SAMPLE_SECOND_TOKEN_NS 2100000000ull
#define SAMPLE_THIRD_TOKEN_NS 2300000000ull
#define SAMPLE_PREFILL_NS 500000000ull
#define SAMPLE_WALL_SECONDS 1793030400.0
#define SAMPLE_NOW_SECONDS 4.0
#define SAMPLE_START_SECONDS 1.0
#define SAMPLE_GENERATED_TOKENS 10u
#define SAMPLE_PREFILL_TOKENS 20u
#define SAMPLE_PROMPT_TOKENS 12u
#define SAMPLE_REUSED_TOKENS 4u
#define SAMPLE_TOKEN_COUNT 3u
#define SAMPLE_REQUEST_TAG 17u
#define SAMPLE_COLUMNS_WIDE 160u
#define SAMPLE_BAD_COLUMNS 20u
#define SAMPLE_INVALID_JSON "{\"missing\":}"
#define SAMPLE_IDENTIFIER "request-17"
#define SAMPLE_CONTEXT 128u
#define SAMPLE_POSITION 15u
#define SAMPLE_BUSY_NS 500000000ull
#define SAMPLE_WINDOW_BUSY_NS 250000000ull
#define SAMPLE_PHASE_SECONDS 2.0
#define SAMPLE_QUEUE_SECONDS 0.25
#define SAMPLE_STOP_BYTES 2u
#define SAMPLE_FROZEN_SECONDS 3.0
#define TERMINAL_ESCAPE 27
#define SAMPLE_LAYER_MASK 5ull
#define SAMPLE_HEAD_MASK 3ull
#define EXAMPLE_PATH_CAPACITY 4096u
#define EXAMPLE_DIRECTORY_VARIABLE "LIVE_STATS_EXAMPLE_DIRECTORY"
#define SAMPLE_CARD_NOW_NS 5500000000ull
#define SAMPLE_CARD_WINDOW_NS 2000000000ull
#define SAMPLE_CARD_LONG_BUSY_NS 400000000ull
#define SAMPLE_CARD_LONG_WINDOW_NS 4000000000ull
#define SAMPLE_GLOBAL_WINDOW_NS 1000000000ull
#define SAMPLE_GLOBAL_LONG_WINDOW_NS 3000000000ull
#define SAMPLE_GLOBAL_WINDOW_TOKENS 6u
#define SAMPLE_GLOBAL_LONG_TOKENS 9u
#define SAMPLE_GLOBAL_WINDOW_PREFILL 10u
#define SAMPLE_GLOBAL_LONG_PREFILL 18u
#define SAMPLE_REQUEST_WINDOW_TOKENS 2u
#define SAMPLE_REQUEST_WINDOW_PREFILL 5u
#define SAMPLE_REQUEST_LONG_PREFILL 8u
#define TABLE_HEADER_LINES 3u
#define TABLE_FIXTURE_LINES (TABLE_HEADER_LINES + STATS_SLOT_CAPACITY)
#define TABLE_MANY_REQUESTS 3u
#define TABLE_HUGE_TOKENS 1000000u
#define SAMPLE_USER "a_123456789012345678901234567890"
#define SAMPLE_COMPACT_USER "a_1234567890123~"
#define TABLE_USER_PAGES 2u
#define TABLE_FOUR_USERS 4u
#define TABLE_COMPACT_USER_WIDTH 16u
#define TABLE_SHORT_USER_WIDTH 5u
#define TABLE_WIDE_USER_WIDTH 32u
#define TABLE_VIEW_WIDTHS 2u

static int failures;

static void write_example(const char *directory, const char *name,
			  const struct chat_buffer *buffer);

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: %s\n", message);
		failures++;
	}
}

static bool contains(const struct chat_buffer *buffer, const char *text)
{
	return memmem(buffer->bytes, buffer->count, text, strlen(text)) != NULL;
}

static struct json_value test_member(struct json_value object, const char *name)
{
	struct json_value key, value;
	if (object.text == NULL || object.text[object.begin] != '{') {
		return (struct json_value){ 0 };
	}
	for (size_t cursor = object.begin;
	     json_next(object, &cursor, &key, &value);) {
		if (key.end - key.begin == strlen(name) + strlen("\"\"") &&
		    memcmp(key.text + key.begin + 1u, name, strlen(name)) ==
			    0) {
			return value;
		}
	}
	return (struct json_value){ 0 };
}

static bool test_value(struct json_value value, const char *text)
{
	return value.text != NULL && value.end - value.begin == strlen(text) &&
	       memcmp(value.text + value.begin, text, strlen(text)) == 0;
}

static void fixture(struct live_snapshot *snapshot)
{
	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->wall_seconds = SAMPLE_WALL_SECONDS;
	snapshot->now_seconds = SAMPLE_NOW_SECONDS;
	snapshot->started_seconds = SAMPLE_START_SECONDS;
	snapshot->queue_depth = SERVER_DEFAULT_QUEUE_DEPTH;
	snapshot->user_queue_depth =
		SERVER_DEFAULT_QUEUE_DEPTH / SERVER_USER_QUEUE_FRACTION;
	snapshot->user_count = 1;
	(void)snprintf(snapshot->users[0].user, sizeof(snapshot->users[0].user),
		       "%s", SAMPLE_USER);
	(void)snprintf(snapshot->build_id, sizeof(snapshot->build_id), "%s",
		       "test-build");
	(void)snprintf(snapshot->model_alias, sizeof(snapshot->model_alias),
		       "%s", "fixture\"model");
	(void)snprintf(snapshot->card_order, sizeof(snapshot->card_order), "%s",
		       "0");
	(void)snprintf(snapshot->placement, sizeof(snapshot->placement), "%s",
		       "layers 0..1: card 0");
	snapshot->engine.now_ns = SAMPLE_NOW_NS;
	snapshot->engine.started_ns = SAMPLE_START_NS;
	snapshot->engine.generated_tokens = SAMPLE_GENERATED_TOKENS;
	snapshot->engine.prefill_tokens = SAMPLE_PREFILL_TOKENS;
	snapshot->engine.slots_total = STATS_SLOT_CAPACITY;
	snapshot->engine.slots_busy = 1;
	snapshot->engine.card_count = 1;
	snapshot->engine.cards[0].busy_ns = SAMPLE_BUSY_NS;
	snapshot->engine.cards[0].busy_available = true;
	snapshot->engine.cards[0].busy_observed_ns = SAMPLE_CARD_NOW_NS;
	snapshot->engine.cards[0].layer_mask = SAMPLE_LAYER_MASK;
	snapshot->engine.cards[0].head_mask = SAMPLE_HEAD_MASK;
	struct live_window_record *const short_window =
		&snapshot->windows[LIVE_WINDOW_SHORT];
	struct live_window_record *const long_window =
		&snapshot->windows[LIVE_WINDOW_LONG];
	short_window->decode =
		(struct live_measurement){ SAMPLE_GLOBAL_WINDOW_TOKENS,
					   SAMPLE_GLOBAL_WINDOW_NS };
	long_window->decode =
		(struct live_measurement){ SAMPLE_GLOBAL_LONG_TOKENS,
					   SAMPLE_GLOBAL_LONG_WINDOW_NS };
	short_window->prefill =
		(struct live_measurement){ SAMPLE_GLOBAL_WINDOW_PREFILL,
					   SAMPLE_GLOBAL_WINDOW_NS };
	long_window->prefill =
		(struct live_measurement){ SAMPLE_GLOBAL_LONG_PREFILL,
					   SAMPLE_GLOBAL_LONG_WINDOW_NS };
	short_window->cards[0] =
		(struct live_measurement){ SAMPLE_WINDOW_BUSY_NS,
					   SAMPLE_CARD_WINDOW_NS };
	long_window->cards[0] =
		(struct live_measurement){ SAMPLE_CARD_LONG_BUSY_NS,
					   SAMPLE_CARD_LONG_WINDOW_NS };
	short_window->slots[0].decode =
		(struct live_measurement){ SAMPLE_REQUEST_WINDOW_TOKENS,
					   SAMPLE_GLOBAL_WINDOW_NS };
	long_window->slots[0].decode =
		(struct live_measurement){ SAMPLE_REQUEST_WINDOW_TOKENS,
					   SAMPLE_CARD_WINDOW_NS };
	short_window->slots[0].prefill =
		(struct live_measurement){ SAMPLE_REQUEST_WINDOW_PREFILL,
					   SAMPLE_GLOBAL_WINDOW_NS };
	long_window->slots[0].prefill =
		(struct live_measurement){ SAMPLE_REQUEST_LONG_PREFILL,
					   SAMPLE_GLOBAL_LONG_WINDOW_NS };
	snapshot->request_count = 1;
	struct live_request_record *const request = &snapshot->requests[0];
	(void)snprintf(request->user, sizeof(request->user), "%s", SAMPLE_USER);
	request->request_tag = SAMPLE_REQUEST_TAG;
	request->arrival_seconds = SAMPLE_START_SECONDS;
	request->phase_seconds = SAMPLE_PHASE_SECONDS;
	request->queue_wait_seconds = SAMPLE_QUEUE_SECONDS;
	request->rendered_prompt_tokens = SAMPLE_PROMPT_TOKENS;
	request->reused_tokens = SAMPLE_REUSED_TOKENS;
	request->reused_slot_tokens = SAMPLE_REUSED_TOKENS;
	(void)snprintf(request->identifier, sizeof(request->identifier), "%s",
		       SAMPLE_IDENTIFIER);
	(void)snprintf(request->client, sizeof(request->client), "%s",
		       "127.0.0.1:5000");
	(void)snprintf(request->model, sizeof(request->model), "%s", "fixture");
	(void)snprintf(request->phase, sizeof(request->phase), "%s", "decode");
	request->options.top_p = 1.0f;
	request->options.is_streamed = true;
	request->options.stop_count = 1;
	request->options.stop_byte_counts[0] = SAMPLE_STOP_BYTES;
	request->options.stops[0][0] = TERMINAL_ESCAPE;
	request->options.stops[0][1] = '\n';
	request->engine.request_tag = SAMPLE_REQUEST_TAG;
	request->engine.phase = STATS_DECODE;
	request->engine.context_capacity = SAMPLE_CONTEXT;
	request->engine.position = SAMPLE_POSITION;
	request->engine.prompt_tokens = SAMPLE_PROMPT_TOKENS;
	request->engine.prompt_processed = SAMPLE_PROMPT_TOKENS;
	request->engine.reused_tokens = SAMPLE_REUSED_TOKENS;
	request->engine.prefill_ns = SAMPLE_PREFILL_NS;
	request->engine.generated_tokens = SAMPLE_TOKEN_COUNT;
	request->engine.submitted_ns = SAMPLE_START_NS;
	request->engine.started_ns = SAMPLE_START_NS;
	request->engine.first_token_ns = SAMPLE_FIRST_TOKEN_NS;
	request->engine.last_token_ns = SAMPLE_THIRD_TOKEN_NS;
	request->engine.token_times[0] = SAMPLE_FIRST_TOKEN_NS;
	request->engine.token_times[1] = SAMPLE_SECOND_TOKEN_NS;
	request->engine.token_times[SAMPLE_TOKEN_COUNT - 1u] =
		SAMPLE_THIRD_TOKEN_NS;
	request->engine.token_ring_count = SAMPLE_TOKEN_COUNT;
	request->engine.token_ring_next = SAMPLE_TOKEN_COUNT;
	snapshot->engine.slots[0] = request->engine;
	for (uint32_t slot = 0; slot < snapshot->engine.slots_total; slot++) {
		snapshot->engine.sessions[slot] = (struct engine_session_stats){
			.context_capacity = SAMPLE_CONTEXT,
			.idle_since_ns = SAMPLE_START_NS
		};
	}
	snapshot->engine.sessions[0].active = true;
	snapshot->engine.sessions[0].phase = STATS_DECODE;
	snapshot->engine.sessions[0].position = SAMPLE_POSITION;
	snapshot->engine.sessions[0].reach = SAMPLE_POSITION;
	(void)snprintf(snapshot->sessions[0].user,
		       sizeof(snapshot->sessions[0].user), "%s", SAMPLE_USER);
	snapshot->finished_count = LIVE_STATS_FINISHED_CAPACITY;
	for (uint32_t index = 0; index < snapshot->finished_count; index++) {
		snapshot->finished[index] = *request;
		snapshot->finished[index].request_tag =
			SAMPLE_REQUEST_TAG + index + 1u;
		snapshot->finished[index].finished_seconds =
			SAMPLE_FROZEN_SECONDS;
		snapshot->finished[index].engine.finished_ns =
			SAMPLE_SUMMARY_END_NS;
		(void)snprintf(snapshot->finished[index].finish_reason,
			       sizeof(snapshot->finished[index].finish_reason),
			       "%s", "length");
	}
	snapshot->sessions[0].has_last_request = true;
	snapshot->sessions[0].last_request = snapshot->finished[0];
	snapshot->sessions[1].has_last_request = true;
	snapshot->sessions[1].last_request = snapshot->finished[1];
	snapshot->sessions[1].last_request.slot = 1;
	(void)snprintf(snapshot->sessions[1].user,
		       sizeof(snapshot->sessions[1].user), "%s", "prior_user");
	(void)snprintf(snapshot->sessions[1].last_request.user,
		       sizeof(snapshot->sessions[1].last_request.user), "%s",
		       "prior_user");
	snapshot->engine.sessions[1].snapshots_saved = 1;
	snapshot->engine.sessions[1].position = SAMPLE_POSITION;
	snapshot->engine.sessions[1].reach = SAMPLE_POSITION;
	snapshot->engine.sessions[1].snapshot_save_bytes = SAMPLE_PROMPT_TOKENS;
	snapshot->engine.sessions[1].snapshot_save_ns = SAMPLE_BUSY_NS;
}

static void test_sessions(struct live_snapshot *snapshot,
			  struct chat_buffer *json)
{
	struct json_value root = { 0 };
	expect(json_read_document(json->bytes, json->count, &root) ==
		       JSON_DOCUMENT_VALID,
	       "session fixture is valid JSON");
	const struct json_value sessions = test_member(root, "sessions");
	struct json_value session = { 0 };
	uint32_t count = 0;
	for (size_t cursor = sessions.begin;
	     json_next(sessions, &cursor, NULL, &session); count++) {
		if (count == 0) {
			const struct json_value current =
				test_member(session, "current_request");
			const struct json_value prompt =
				test_member(current, "prompt");
			const struct json_value decode =
				test_member(current, "decode");
			expect(test_value(test_member(session, "active"),
					  "true") &&
				       test_value(test_member(session, "state"),
						  "\"decode\"") &&
				       test_value(test_member(session,
							      "position"),
						  "15") &&
				       test_value(test_member(session, "reach"),
						  "15"),
			       "busy slot retains phase and held context");
			expect(test_value(
				       test_member(
					       prompt,
					       "prefill_tokens_per_second_10s"),
				       "5") &&
				       test_value(
					       test_member(
						       prompt,
						       "prefill_tokens_per_second"),
					       "16") &&
				       test_value(
					       test_member(
						       decode,
						       "tokens_per_second_10s"),
					       "2") &&
				       test_value(
					       test_member(
						       decode,
						       "tokens_per_second_60s"),
					       "1") &&
				       test_value(
					       test_member(
						       decode,
						       "tokens_per_second_since_first"),
					       "1") &&
				       test_value(
					       test_member(decode,
							   "ms_per_token_10s"),
					       "500") &&
				       test_value(
					       test_member(
						       decode,
						       "ms_per_token_since_first"),
					       "1000"),
			       "session current request distinguishes live windows from phase averages");
			const struct json_value last =
				test_member(session, "last_request");
			expect(test_value(
				       test_member(test_member(last, "decode"),
						   "tokens_per_second_10s"),
				       "null") &&
				       test_value(
					       test_member(
						       test_member(last,
								   "decode"),
						       "tokens_per_second_since_first"),
					       "2") &&
				       test_value(test_member(
							  test_member(last,
								      "output"),
							  "finish_reason"),
						  "\"length\""),
			       "last summary freezes averages and cannot borrow the current request's windows");
		} else if (count == 1) {
			const struct json_value transfers =
				test_member(session, "snapshots");
			expect(test_value(test_member(session, "active"),
					  "false") &&
				       test_value(test_member(session, "state"),
						  "\"idle\"") &&
				       test_value(test_member(session,
							      "idle_seconds"),
						  "3") &&
				       test_value(test_member(session, "user"),
						  "\"prior_user\"") &&
				       test_value(test_member(session,
							      "current_request"),
						  "null") &&
				       !test_value(test_member(session,
							       "last_request"),
						   "null"),
			       "idle slot retains prior identity, elapsed idle time and its last request");
			expect(test_value(test_member(transfers, "saved"),
					  "1") &&
				       test_value(test_member(transfers,
							      "save_bytes"),
						  "12") &&
				       test_value(test_member(transfers,
							      "save_ms"),
						  "500"),
			       "session snapshot counts, bytes and elapsed milliseconds are present");
		}
	}
	expect(count == STATS_SLOT_CAPACITY,
	       "every busy or idle slot gets exactly one session entry");
	const uint32_t saved_finished = snapshot->finished_count;
	snapshot->finished_count = 0;
	snapshot->engine.sessions[0].operation = SESSION_SAVE;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(json, "\"state\":\"snapshot_save\"") &&
		       contains(json, "\"last_request\":{\"identity\":"),
	       "snapshot operations have an active state and last summaries outlive the finished ring");
	snapshot->engine.sessions[0].operation = 0;
	snapshot->finished_count = saved_finished;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK,
	       "restore session fixture");
}

static void test_json(struct live_snapshot *snapshot, struct chat_buffer *json)
{
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK,
	       "snapshot serializes");
	struct json_value value;
	expect(json_read_document(json->bytes, json->count, &value) ==
		       JSON_DOCUMENT_VALID,
	       "snapshot is a whole well-formed JSON document");
	expect(contains(json, "\"version\":2"),
	       "window semantics use schema version two");
	expect(contains(json, "\"decode_tokens_per_second_interval\":6"),
	       "decode interval uses counter delta");
	expect(contains(json, "\"prefill_tokens_per_second_interval\":10"),
	       "prefill interval uses counter delta");
	expect(contains(json, "\"busy_percent_interval\":12.5"),
	       "card busy interval uses its own observation times rather than the global interval");
	expect(contains(json, "\"decode_tokens_per_second_10s\":6") &&
		       contains(json, "\"decode_tokens_per_second_60s\":3") &&
		       contains(json, "\"prefill_tokens_per_second_10s\":10") &&
		       contains(json, "\"prefill_tokens_per_second_60s\":6") &&
		       contains(json, "\"busy_percent_60s\":10"),
	       "explicit global/card windows retain independent ten- and sixty-second measurements");
	expect(contains(json, "\"layers\":[0,2]") &&
		       contains(
			       json,
			       "\"heads\":[\"language_model\",\"draft_markov\"]"),
	       "card layer and head masks have semantic lists");
	expect(contains(json, "\"free_device_bytes\":null"),
	       "unknown device free memory stays unavailable");
	expect(contains(json, "\"p50_ms\":100") &&
		       contains(json, "\"p95_ms\":200"),
	       "token latency percentiles use chronological fixed ring");
	expect(contains(json, "\"stop_sequences\":[\"\\u001b\\n\"]"),
	       "terminal controls are JSON escaped");
	expect(contains(json, "\"queue_depth\":16"),
	       "global snapshot reports configured queue capacity");
	expect(contains(json, "\"user_queue_depth\":8") &&
		       contains(json,
				"\"queued_requests\":0,\"served_requests\":"),
	       "global and per-user snapshots report queue limits and waiting counts");
	expect(contains(json, "\"elapsed_ms\":2000"),
	       "finished request elapsed stays frozen");
	static const char *const fields[] = { "wall_seconds",
					      "uptime_seconds",
					      "build_id",
					      "model_alias",
					      "card_order",
					      "placement",
					      "slots_total",
					      "slots_busy",
					      "slots_idle",
					      "concurrent_requests",
					      "queued_requests",
					      "queue_depth",
					      "user_queue_depth",
					      "open_connections",
					      "served_requests",
					      "refused_503",
					      "refused_400",
					      "refused_auth",
					      "failed_requests",
					      "cancelled_requests",
					      "prompt_tokens",
					      "cached_tokens",
					      "generated_tokens",
					      "flights_in_air",
					      "steps_landed_per_second",
					      "step_latency",
					      "steps_by_kind",
					      "prompt_checkpoints_saved",
					      "prompt_checkpoints_restored",
					      "snapshots_saved",
					      "snapshots_loaded",
					      "snapshot_save_bytes",
					      "snapshot_load_bytes",
					      "snapshot_save_ms",
					      "snapshot_load_ms",
					      "event_queue_depth",
					      "event_backpressure_stalls",
					      "speculation",
					      "chosen_depth_histogram",
					      "proposals",
					      "reached",
					      "accepted",
					      "acceptance_percent",
					      "mean_accepted_per_round",
					      "tokens_per_round",
					      "probes",
					      "dspark_tails_run",
					      "lookup",
					      "dspark",
					      "engine_index",
					      "name",
					      "pci_bus_id",
					      "arithmetic_class",
					      "layer_mask",
					      "head_mask",
					      "holds_draft",
					      "arena_bytes",
					      "free_device_bytes",
					      "steps",
					      "visits",
					      "busy_ms_interval",
					      "busy_percent_interval",
					      "busy_time_available",
					      "busy_observed_monotonic_ns",
					      "identity",
					      "request_id",
					      "client",
					      "model",
					      "stream",
					      "tools_declared",
					      "reasoning_effort",
					      "temperature",
					      "top_p",
					      "seed",
					      "max_tokens",
					      "stop_sequences",
					      "timeline",
					      "phase",
					      "phase_elapsed_ms",
					      "elapsed_ms",
					      "queue_wait_ms",
					      "rendered_tokens",
					      "reused_tokens",
					      "reused_slot_prefix_tokens",
					      "reused_snapshot_tokens",
					      "reused_prompt_checkpoint_tokens",
					      "processed_tokens",
					      "remaining_tokens",
					      "progress_percent",
					      "prefill_tokens_per_second",
					      "prefill_ms_per_token",
					      "prefill_elapsed_ms",
					      "time_to_first_token_ms",
					      "reasoning_tokens",
					      "content_tokens",
					      "tool_call_tokens",
					      "tokens_per_second_since_first",
					      "ms_per_token_since_first",
					      "tokens_per_second_interval",
					      "ms_per_token_interval",
					      "inter_token_latency",
					      "min_ms",
					      "p50_ms",
					      "p95_ms",
					      "max_ms",
					      "chosen_depth",
					      "last_confidence_bins",
					      "source",
					      "position",
					      "context_capacity",
					      "context_used_percent",
					      "reach",
					      "decoder_from",
					      "checkpoint_G",
					      "kept_lag",
					      "bytes_sent",
					      "events_sent",
					      "finish_reason",
					      "last_error",
					      "finished" };
	for (size_t index = 0; index < sizeof(fields) / sizeof(fields[0]);
	     index++) {
		expect(contains(json, fields[index]), fields[index]);
	}
}

static void expect_width(const struct chat_buffer *text, uint32_t columns)
{
	size_t line_bytes = 0;
	for (size_t index = 0; index < text->count; index++) {
		if (text->bytes[index] == '\n') {
			expect(line_bytes <= columns,
			       "text line stays within requested columns");
			line_bytes = 0;
		} else {
			line_bytes++;
		}
	}
	expect(line_bytes <= columns,
	       "last text line stays within requested columns");
}

/* The step plan's BATCH programs: none without a plan (the fields above), each program's counts with one. */
static void test_batch_programs(struct live_snapshot *snapshot,
				struct chat_buffer *json)
{
	expect(!contains(json, "batch_programs"),
	       "no step plan: no batch_programs");
	snapshot->engine.batch_program_count = 2;
	snapshot->engine.batch_programs[0] =
		(struct stats_batch_program){ 8, 1, 19500, 10, 195000000 };
	snapshot->engine.batch_programs[1] =
		(struct stats_batch_program){ 12, 2, 28600, 0, 0 };
	json->count = 0;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(
			       json,
			       "\"batch_programs\":[{\"rows\":8,\"verifiers\":1,"
			       "\"steps\":10,\"mean_ms\":19.5,\"cost_ms\":19.5},"
			       "{\"rows\":12,\"verifiers\":2,\"steps\":0,"
			       "\"mean_ms\":0,\"cost_ms\":28.6}]"),
	       "a step plan: each program's rows, verifiers, steps, mean and cost");
	snapshot->engine.batch_program_count = 0;
	json->count = 0;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK,
	       "the snapshot serializes again without the plan");
}

static void test_text(const struct chat_buffer *json)
{
	struct chat_buffer text = { 0 };
	expect(live_stats_text(json->bytes, json->count,
			       LIVE_STATS_DEFAULT_COLUMNS, false,
			       &text) == STATUS_OK,
	       "80-column log render");
	expect_width(&text, LIVE_STATS_DEFAULT_COLUMNS);
	expect(text.count > 0 && text.bytes[0] == '[',
	       "log starts with timestamp");
	expect(memchr(text.bytes, TERMINAL_ESCAPE, text.count) == NULL,
	       "log has no terminal controls");
	expect(contains(&text, "global:") && contains(&text, "requests:") &&
		       contains(&text, "finished:"),
	       "text includes global, active and finished blocks");
	expect(contains(&text, "request-17") &&
		       contains(&text, "checkpoint_G") &&
		       contains(&text, "last_confidence_bins"),
	       "text retains per-request identity and nested metrics");
	expect(live_stats_text(json->bytes, json->count, SAMPLE_COLUMNS_WIDE,
			       false, &text) == STATUS_OK,
	       "160-column log render");
	expect_width(&text, SAMPLE_COLUMNS_WIDE);
	expect(live_stats_text(json->bytes, json->count, SAMPLE_COLUMNS_WIDE,
			       true, &text) == STATUS_OK &&
		       text.count > strlen("\033[H\033[J") &&
		       memcmp(text.bytes, "\033[H\033[J",
			      strlen("\033[H\033[J")) == 0,
	       "TTY redraw homes and clears to end");
	expect(live_stats_text(SAMPLE_INVALID_JSON, strlen(SAMPLE_INVALID_JSON),
			       LIVE_STATS_DEFAULT_COLUMNS, false,
			       &text) == STATUS_INVALID_ARGUMENT,
	       "malformed JSON refused");
	expect(live_stats_text(json->bytes, json->count, SAMPLE_BAD_COLUMNS,
			       false, &text) == STATUS_INVALID_ARGUMENT,
	       "invalid text width refused");
	chat_buffer_free(&text);
}

static void test_table(struct live_snapshot *snapshot, struct chat_buffer *json)
{
	static const uint32_t widths[] = { LIVE_STATS_DEFAULT_COLUMNS,
					   SAMPLE_COLUMNS_WIDE };
	struct chat_buffer table = { 0 }, text = { 0 };
	for (size_t index = 0; index < sizeof(widths) / sizeof(widths[0]);
	     index++) {
		const uint32_t columns = widths[index];
		expect(live_stats_table_entry(json->bytes, json->count, columns,
					      false, &table) == STATUS_OK,
		       "compact table renders at 80 and 160 columns");
		expect_width(&table, columns);
		uint32_t lines = 0;
		for (size_t byte = 0; byte < table.count; byte++) {
			lines += table.bytes[byte] == '\n' ? 1u : 0u;
		}
		expect(lines == TABLE_FIXTURE_LINES &&
			       !contains(&table, "global:") &&
			       !contains(&table, "finished:"),
		       "table contains a global line, headings and exactly one concurrent row");
		expect(contains(&table, "cards10% c0=") &&
			       contains(&table, "idle sl=1 user=prior_user") &&
			       contains(&table, columns == SAMPLE_COLUMNS_WIDE ?
							"davg/s" :
							"dav") &&
			       contains(&table, columns == SAMPLE_COLUMNS_WIDE ?
							"p10/s" :
							"p10") &&
			       contains(&table, columns == SAMPLE_COLUMNS_WIDE ?
							"msavg" :
							"mav"),
		       "table contains card-window load, idle context and live/average phase speeds");
		expect(contains(&table, "16:00:00Z") &&
			       contains(&table, "ms/t") &&
			       contains(&table, columns == SAMPLE_COLUMNS_WIDE ?
							"ITL95ms" :
							"95"),
		       "global time, reciprocal rate and latency headings survive both widths");
		expect(contains(&table, columns == SAMPLE_COLUMNS_WIDE ?
						"queued=0/16" :
						"q=0/16"),
		       "table shows queued count and configured capacity");
		expect(contains(&table, "user") &&
			       contains(&table, columns == SAMPLE_COLUMNS_WIDE ?
							SAMPLE_USER :
							SAMPLE_COMPACT_USER),
		       "compact usernames end in a truncation marker; wide usernames remain complete");
		expect(contains(&table,
				columns == SAMPLE_COLUMNS_WIDE ? "ctx%" : "cx"),
		       "last table column fits alongside the username");
		expect(contains(&table, columns == SAMPLE_COLUMNS_WIDE ?
						"decode" :
						" d "),
		       "phase remains readable in wide and compact rows");
		expect(live_stats_text(json->bytes, json->count, columns, false,
				       &text) == STATUS_OK &&
			       text.count > table.count &&
			       memcmp(text.bytes, table.bytes, table.count) ==
				       0,
		       "text starts with the identical table before the full dump");
		expect(contains(&text, SAMPLE_USER) && contains(json, "\"user\":\"" SAMPLE_USER "\""),
		       "username appears in full dump and JSON including copied finished records");
		expect(live_stats_table_entry(json->bytes, json->count, columns,
					      true, &table) == STATUS_OK &&
			       memcmp(table.bytes, "\033[H\033[J",
				      strlen("\033[H\033[J")) == 0,
		       "table alone redraws a terminal");
	}
	const struct live_request_record saved = snapshot->requests[0];
	snapshot->request_count = TABLE_MANY_REQUESTS;
	for (uint32_t index = 1; index < snapshot->request_count; index++) {
		snapshot->requests[index] = saved;
		snapshot->requests[index].slot = UINT32_MAX;
		snapshot->requests[index].engine.first_token_ns = 0;
		snapshot->requests[index].engine.generated_tokens =
			TABLE_HUGE_TOKENS;
		(void)snprintf(snapshot->requests[index].phase,
			       sizeof(snapshot->requests[index].phase), "%s",
			       "queued");
	}
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       live_stats_table_entry(json->bytes, json->count,
					      LIVE_STATS_DEFAULT_COLUMNS, false,
					      &table) == STATUS_OK &&
		       contains(&table, " q ") && contains(&table, "1M"),
	       "queued slots, unavailable rates and million-token counts render compactly");
	expect_width(&table, LIVE_STATS_DEFAULT_COLUMNS);
	expect(memchr(table.bytes, TERMINAL_ESCAPE, table.count) == NULL,
	       "table log mode cannot emit terminal controls");
	snapshot->request_count = 0;
	snapshot->engine.sessions[0].active = false;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       live_stats_table_entry(json->bytes, json->count,
					      SAMPLE_COLUMNS_WIDE, false,
					      &table) == STATUS_OK &&
		       !contains(&table, "decode "),
	       "idle table omits finished request rows");
	expect(live_stats_table_entry(SAMPLE_INVALID_JSON,
				      strlen(SAMPLE_INVALID_JSON),
				      SAMPLE_COLUMNS_WIDE, false,
				      &table) == STATUS_INVALID_ARGUMENT,
	       "table refuses malformed JSON");
	snapshot->request_count = 1;
	snapshot->engine.sessions[0].active = true;
	snapshot->requests[0] = saved;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK,
	       "restore fixture JSON");
	chat_buffer_free(&table);
	chat_buffer_free(&text);
}

static void test_table_users(struct live_snapshot *snapshot,
			     struct chat_buffer *json)
{
	static const char *const names[TABLE_USER_PAGES][TABLE_FOUR_USERS] = {
		{ "a", "alice", "1234567890123456", SAMPLE_USER },
		{ "a", "bob", "carol", "dave" }
	};
	static const uint32_t columns[] = { LIVE_STATS_DEFAULT_COLUMNS,
					    SAMPLE_COLUMNS_WIDE };
	static const char *const
		example_names[TABLE_USER_PAGES][TABLE_VIEW_WIDTHS] = {
			{ "users_long80.txt", "users_long160.txt" },
			{ "users_short80.txt", "users_short160.txt" }
		};
	const struct live_request_record saved = snapshot->requests[0];
	struct chat_buffer table = { 0 }, text = { 0 };
	snapshot->request_count = TABLE_FOUR_USERS;
	for (size_t page = 0; page < TABLE_USER_PAGES; page++) {
		for (size_t index = 0; index < TABLE_FOUR_USERS; index++) {
			snapshot->requests[index] = saved;
			snapshot->requests[index].slot = (uint32_t)index;
			snapshot->requests[index].request_tag =
				SAMPLE_REQUEST_TAG + (uint32_t)index;
			snapshot->requests[index].engine.slot = (uint32_t)index;
			snapshot->requests[index].engine.request_tag =
				snapshot->requests[index].request_tag;
			snapshot->engine.slots[index] =
				snapshot->requests[index].engine;
			snapshot->engine.sessions[index] =
				snapshot->engine.sessions[0];
			for (uint32_t window = 0; window < LIVE_WINDOW_COUNT;
			     window++) {
				snapshot->windows[window].slots[index] =
					snapshot->windows[window].slots[0];
			}
			(void)snprintf(snapshot->requests[index].user,
				       sizeof(snapshot->requests[index].user),
				       "%s", names[page][index]);
			(void)snprintf(snapshot->sessions[index].user,
				       sizeof(snapshot->sessions[index].user),
				       "%s", names[page][index]);
		}
		expect(live_stats_json_entry(snapshot, json) == STATUS_OK,
		       "four-user JSON renders");
		for (size_t index = 0;
		     index < sizeof(columns) / sizeof(columns[0]); index++) {
			const bool wide = columns[index] == SAMPLE_COLUMNS_WIDE;
			const uint32_t user_width =
				wide	  ? TABLE_WIDE_USER_WIDTH :
				page == 0 ? TABLE_COMPACT_USER_WIDTH :
					    TABLE_SHORT_USER_WIDTH;
			expect(live_stats_table_entry(json->bytes, json->count,
						      columns[index], false,
						      &table) == STATUS_OK,
			       "four-user table renders at both widths");
			expect_width(&table, columns[index]);
			const char *const directory =
				getenv(EXAMPLE_DIRECTORY_VARIABLE);
			if (directory != NULL && directory[0] != '\0') {
				write_example(directory,
					      example_names[page][index],
					      &table);
			}
			uint32_t lines = 0;
			for (size_t byte = 0; byte < table.count; byte++) {
				lines += table.bytes[byte] == '\n' ? 1u : 0u;
			}
			expect(lines == TABLE_FIXTURE_LINES,
			       "four active users and four idle slots each retain one row");
			const char *const heading = memmem(table.bytes,
							   table.count, "user",
							   strlen("user"));
			const char *line_begin = heading;
			while (line_begin != NULL && line_begin > table.bytes &&
			       line_begin[-1] != '\n') {
				line_begin--;
			}
			expect(heading != NULL &&
				       (size_t)(heading - line_begin) +
						       strlen("user") ==
					       user_width,
			       "username width follows the longest name on this page, capped only in compact mode");
			expect(!contains(&table, ">") &&
				       !contains(&table, "<1") &&
				       contains(&table, "500"),
			       "numeric metrics retain useful values instead of overflow markers");
			for (size_t user = 0; user < TABLE_FOUR_USERS; user++) {
				const char *const shown =
					!wide && page == 0 &&
							user + 1 ==
								TABLE_FOUR_USERS ?
						SAMPLE_COMPACT_USER :
						names[page][user];
				expect(contains(&table, shown),
				       "all four different username lengths remain identifiable");
			}
			expect(live_stats_text(json->bytes, json->count,
					       columns[index], false,
					       &text) == STATUS_OK &&
				       text.count > table.count &&
				       memcmp(text.bytes, table.bytes,
					      table.count) == 0,
			       "text shares the four-user table and preserves full JSON names in its dump");
		}
	}
	fixture(snapshot);
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK,
	       "restore one-user JSON");
	chat_buffer_free(&table);
	chat_buffer_free(&text);
}

static void test_ring_wrap(struct live_snapshot *snapshot,
			   struct chat_buffer *json)
{
	struct engine_slot_stats *const slot = &snapshot->requests[0].engine;
	slot->token_ring_count = STATS_TOKEN_RING;
	slot->token_ring_next = 1;
	for (uint32_t index = 0; index < STATS_TOKEN_RING; index++) {
		const uint32_t ring_index =
			(slot->token_ring_next + index) % STATS_TOKEN_RING;
		slot->token_times[ring_index] =
			SAMPLE_FIRST_TOKEN_NS +
			(uint64_t)index * SAMPLE_BUSY_NS;
	}
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(
			       json,
			       "\"inter_token_latency\":{\"samples\":63,\"mean_ms\":500,\"min_ms\":500"),
	       "wrapped token ring retains all chronological intervals");
}

static void test_queue_times(struct live_snapshot *snapshot,
			     struct chat_buffer *json)
{
	const struct live_request_record saved = snapshot->requests[0];
	struct live_request_record *const request = &snapshot->requests[0];
	request->slot = UINT32_MAX;
	request->engine.started_ns = 0;
	request->queue_wait_seconds = 0;
	(void)snprintf(request->phase, sizeof(request->phase), "%s", "queued");
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(json, "\"queue_wait_ms\":2000"),
	       "live encoded waiting grows until admission");
	request->slot = 0;
	request->queue_wait_seconds = SAMPLE_QUEUE_SECONDS;
	request->engine.phase = STATS_QUEUED;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(json, "\"queue_wait_ms\":3250"),
	       "engine queued waiting grows after admission");
	request->engine.started_ns = SAMPLE_FIRST_TOKEN_NS;
	request->engine.first_token_ns = SAMPLE_THIRD_TOKEN_NS;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(json, "\"queue_wait_ms\":1250"),
	       "started engine queue wait freezes");
	expect(contains(json, "\"time_to_first_token_ms\":1300"),
	       "first token latency includes pre-engine arrival wait");
	snapshot->requests[0] = saved;
}

static void test_card_busy_windows(struct live_snapshot *snapshot,
				   struct chat_buffer *json)
{
	struct engine_card_stats *const current = &snapshot->engine.cards[0];
	struct live_measurement *const busy =
		&snapshot->windows[LIVE_WINDOW_SHORT].cards[0];
	const struct engine_card_stats saved_current = *current;
	const struct live_measurement saved_busy = *busy;
	current->busy_available = false;
	current->busy_observed_ns = 0;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(json, "\"busy_time_available\":false") &&
		       contains(json, "\"busy_observed_monotonic_ns\":0") &&
		       contains(json, "\"busy_ms_total\":null") &&
		       contains(json, "\"busy_ms_interval\":null") &&
		       contains(json, "\"busy_percent_interval\":null"),
	       "unavailable current busy observation exposes no timings");
	*current = saved_current;
	busy->elapsed_ns = 0;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(json, "\"busy_time_available\":true") &&
		       contains(json, "\"busy_ms_total\":500") &&
		       contains(json, "\"busy_ms_interval\":null") &&
		       contains(json, "\"busy_percent_interval\":null"),
	       "unavailable historical window keeps the cumulative total and omits live timings");
	*busy = saved_busy;
	busy->count = 0;
	expect(live_stats_json_entry(snapshot, json) == STATUS_OK &&
		       contains(json, "\"busy_ms_interval\":0") &&
		       contains(json, "\"busy_percent_interval\":0"),
	       "coherent idle card observation is a measured zero");
	*current = saved_current;
	*busy = saved_busy;
}

static void write_example(const char *directory, const char *name,
			  const struct chat_buffer *buffer)
{
	char path[EXAMPLE_PATH_CAPACITY];
	const int count =
		snprintf(path, sizeof(path), "%s/%s", directory, name);
	if (count <= 0 || (size_t)count >= sizeof(path)) {
		expect(false, "example path fits");
		return;
	}
	FILE *const file = fopen(path, "wb");
	if (file == NULL) {
		expect(false, "example output opens");
		return;
	}
	const size_t written = fwrite(buffer->bytes, 1, buffer->count, file);
	const int closed = fclose(file);
	expect(written == buffer->count, "example output writes completely");
	expect(closed == 0, "example output closes");
}

static void examples(const struct chat_buffer *json)
{
	const char *const directory = getenv(EXAMPLE_DIRECTORY_VARIABLE);
	if (directory == NULL || directory[0] == '\0') {
		return;
	}
	write_example(directory, "fixture.json", json);
	struct chat_buffer text = { 0 };
	if (live_stats_text(json->bytes, json->count,
			    LIVE_STATS_DEFAULT_COLUMNS, false,
			    &text) == STATUS_OK) {
		write_example(directory, "fixture80.txt", &text);
	} else {
		expect(false, "80-column example renders");
	}
	if (live_stats_text(json->bytes, json->count, SAMPLE_COLUMNS_WIDE,
			    false, &text) == STATUS_OK) {
		write_example(directory, "fixture160.txt", &text);
	} else {
		expect(false, "160-column example renders");
	}
	chat_buffer_free(&text);
}

int main(void)
{
	struct live_snapshot *const snapshot = calloc(1, sizeof(*snapshot));
	if (snapshot == NULL) {
		return 1;
	}
	struct chat_buffer json = { 0 };
	fixture(snapshot);
	test_json(snapshot, &json);
	test_batch_programs(snapshot, &json);
	test_sessions(snapshot, &json);
	test_text(&json);
	test_table(snapshot, &json);
	test_table_users(snapshot, &json);
	examples(&json);
	test_card_busy_windows(snapshot, &json);
	test_queue_times(snapshot, &json);
	test_ring_wrap(snapshot, &json);
	expect(live_stats_json_entry(NULL, &json) == STATUS_INVALID_ARGUMENT,
	       "NULL snapshot refused");
	char build[LIVE_STATS_BUILD_BYTES] = { 0 };
	expect(live_stats_build_id(build) == STATUS_OK && build[0] != '\0',
	       "running executable build identity is available");
	chat_buffer_free(&json);
	free(snapshot);
	(void)printf(
		"live statistics record bytes: engine_slot=%zu engine=%zu request=%zu snapshot=%zu\n",
		sizeof(struct engine_slot_stats), sizeof(struct engine_stats),
		sizeof(struct live_request_record),
		sizeof(struct live_snapshot));
	return failures == 0 ? 0 : 1;
}
