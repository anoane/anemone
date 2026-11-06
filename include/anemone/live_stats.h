/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Release live statistics. JSON schema 2 retains interval field names as 10-second window aliases; explicit
 * _10s/_60s rates use one poll-thread sampler, independent of reads or printing. window_seconds_10s/_60s report
 * the actual shorter coverage during startup. Since-start and phase averages retain their original meanings.
 * Fixed request/session records retain no credentials. JSON is the shared wire snapshot;
 * the text renderer shows every JSON field, wraps to the requested columns and escapes terminal controls.
 * Render outside all engine locks. An output buffer may grow up to the existing chat-output bound.
 * Invalid pointers, malformed JSON or columns outside 40..240 are INVALID_ARGUMENT. Allocation failure is
 * OUT_OF_MEMORY. Scalar outputs remain untouched on failure. Snapshot counters use the engine's lifetime.
 */
#ifndef LIVE_STATS_H
#define LIVE_STATS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/chat_completions.h"
#include "anemone/engine_stats.h"
#include "anemone/server_keys.h"
#include "anemone/server.h"
#include "anemone/live_metrics.h"

#define LIVE_STATS_LEAST_INTERVAL_MS 100u
#define LIVE_STATS_MOST_INTERVAL_MS 60000u
#define LIVE_STATS_DEFAULT_INTERVAL_MS 1000u
#define LIVE_STATS_DEFAULT_COLUMNS 80u
#define LIVE_STATS_LEAST_COLUMNS 40u
#define LIVE_STATS_MOST_COLUMNS 240u
#define LIVE_STATS_FINISHED_CAPACITY 8u
/* Sending requests and disconnected active commands can outlive the waiting queue. */
#define LIVE_STATS_REQUEST_CAPACITY \
	(SERVER_MOST_CONNECTIONS + STATS_SLOT_CAPACITY)
#define LIVE_STATS_MODEL_BYTES 256u
#define LIVE_STATS_CLIENT_BYTES 80u
#define LIVE_STATS_PHASE_BYTES 16u
#define LIVE_STATS_REASON_BYTES 32u
#define LIVE_STATS_ERROR_BYTES 256u
#define LIVE_STATS_BUILD_BYTES 96u
#define LIVE_STATS_PLACEMENT_BYTES 4096u
#define LIVE_STATS_CARD_ORDER_BYTES 256u

struct live_request_record {
	char user[SERVER_USER_BYTES]; /* authenticated name; "-" with no key file */
	uint32_t slot; /* UINT32_MAX while queued without a slot */
	uint32_t request_tag;
	char identifier[CHAT_IDENTIFIER_CAPACITY];
	char client[LIVE_STATS_CLIENT_BYTES];
	char model[LIVE_STATS_MODEL_BYTES];
	struct chat_request options;
	bool tools_declared;
	char phase[LIVE_STATS_PHASE_BYTES];
	double arrival_seconds; /* monotonic */
	double phase_seconds;
	double queue_wait_seconds;
	double finished_seconds; /* 0 until done; freezes elapsed and phase times */
	uint32_t rendered_prompt_tokens;
	uint32_t reused_tokens;
	uint32_t reused_slot_tokens;
	uint32_t reused_snapshot_tokens;
	uint32_t reused_checkpoint_tokens;
	uint64_t reasoning_tokens;
	uint64_t content_tokens;
	uint64_t tool_call_tokens;
	uint64_t bytes_sent;
	uint64_t events_sent;
	char finish_reason[LIVE_STATS_REASON_BYTES];
	char last_error[LIVE_STATS_ERROR_BYTES];
	struct engine_slot_stats engine;
};

/* Lifetime completed totals plus current requests at snapshot time. Removed usernames retain their totals.
 * Served counts successful generation commands, refused counts HTTP refusals with a known identity.
 * Unknown credentials are counted only in the existing global refused_auth counter. */
struct live_user {
	char user[SERVER_USER_BYTES];
	uint32_t concurrent_requests;
	uint32_t queued_requests; /* encoded waiters, including requests waiting on a slot's snapshot prerequisite */
	uint64_t served_requests, prompt_tokens, cached_tokens, generated_tokens, refused;
};

/* One entry per engine slot. Current requests live in snapshot.requests; match slot and request_tag.
 * Last request survives idle periods and the finished ring wrapping. User is current, otherwise last. */
struct live_session {
	char user[SERVER_USER_BYTES];
	bool has_last_request;
	struct live_request_record last_request;
};

struct live_snapshot {
	uint32_t user_count;
	struct live_user users[SERVER_MOST_USERS];
	double wall_seconds; /* Unix seconds, fractional */
	double now_seconds; /* monotonic */
	double started_seconds;
	char build_id[LIVE_STATS_BUILD_BYTES];
	char model_alias[LIVE_STATS_MODEL_BYTES];
	char card_order[LIVE_STATS_CARD_ORDER_BYTES];
	char placement[LIVE_STATS_PLACEMENT_BYTES];
	uint32_t queued_requests;
	uint32_t queue_depth; /* configured encoded-waiting capacity, separate from engine slots */
	uint32_t user_queue_depth; /* configured waiting bound per authenticated username; no active-slot cap */
	uint32_t open_connections;
	uint64_t served_requests;
	uint64_t refused_503;
	uint64_t refused_400;
	uint64_t refused_auth;
	uint64_t failed_requests;
	uint64_t cancelled_requests;
	struct engine_stats engine;
	struct live_window_record windows[LIVE_WINDOW_COUNT];
	struct live_session sessions[STATS_SLOT_CAPACITY];
	uint32_t request_count;
	struct live_request_record requests[LIVE_STATS_REQUEST_CAPACITY];
	uint32_t finished_count;
	struct live_request_record finished[LIVE_STATS_FINISHED_CAPACITY];
};

MUST_CHECK result_status live_stats_json_entry(
	const struct live_snapshot *snapshot, struct chat_buffer *output);
/* The running executable's GNU ELF build id in lowercase hexadecimal, or "unavailable" if absent. */
MUST_CHECK result_status
live_stats_build_id(char output[LIVE_STATS_BUILD_BYTES]);
/* Both views start with global, card 10-second busy percentages and headings, within min(columns, 160).
 * Active slots (including snapshot commands) and queued requests have rows; idle slots follow in short form,
 * retaining user/context and idle duration. Session rows show live (10s) and average decode and prefill rates,
 * live/average ms/token, TTFT, ITL p50/p95 and speculation acceptance per round.
 * The global queued figure is current/configured depth; queue capacity is independent of engine slots.
 * At 80 columns phase is its initial, numbers use decimal K/M/G/T/P/E suffixes and headers are abbreviated.
 * The compact user column fits the longest visible username, at least the heading and at most 16 columns;
 * longer names end in '~'. Saved space widens numeric metrics. The wide view retains 32-column usernames.
 * Durations are milliseconds except elapsed seconds; rates are tokens/second. Undefined values are '-'.
 * is_tty prefixes ANSI home and clear-to-end. Text then includes the full per-metric dump at columns width.
 * Values remain JSON escaped, so model aliases, request ids and stop strings cannot inject terminal controls. */
MUST_CHECK result_status live_stats_text(const char *json, size_t byte_count,
					 uint32_t columns, bool is_tty,
					 struct chat_buffer *output);
MUST_CHECK result_status live_stats_table_entry(const char *json,
						size_t byte_count,
						uint32_t columns, bool is_tty,
						struct chat_buffer *output);

#endif
