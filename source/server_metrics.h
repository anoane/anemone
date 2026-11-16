/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* HTTP poller owns these fixed records. No token-worker writes, allocation or locks. */
#ifndef SERVER_METRICS_H
#define SERVER_METRICS_H

#include "anemone/chat_completions.h"

#define METRICS_BUCKET_COUNT 32u
#define METRICS_HTTP_HANDLERS 7u
#define METRICS_HTTP_METHODS 8u
#define METRICS_HTTP_STATUS_CLASSES 4u
#define METRICS_HTTP_LOW_BUCKETS 3u
#define METRICS_TEXT_BYTES 262144u
enum metrics_finish {
	METRICS_FINISH_STOP,
	METRICS_FINISH_LENGTH,
	METRICS_FINISH_TOOLS,
	METRICS_FINISH_ERROR,
	METRICS_FINISH_CANCELLED,
	METRICS_FINISH_COUNT
};

enum metrics_latency {
	METRICS_TTFT,
	METRICS_QUEUE,
	METRICS_PREFILL,
	METRICS_DECODE,
	METRICS_E2E,
	METRICS_INFERENCE,
	METRICS_ITL,
	METRICS_TPOT,
	METRICS_PROMPT_TOKENS,
	METRICS_GENERATION_TOKENS,
	METRICS_MAX_TOKENS,
	METRICS_MAX_GENERATION,
	METRICS_N,
	METRICS_PREEMPTIONS,
	METRICS_COMPUTED_TOKENS,
	METRICS_LATENCY_COUNT
};

struct metrics_histogram {
	uint64_t buckets[METRICS_BUCKET_COUNT +
			 1u]; /* disjoint; last is overflow */
	double sum;
};

struct metrics_http_cell {
	uint64_t requests[METRICS_HTTP_STATUS_CLASSES];
	uint64_t buckets[METRICS_HTTP_LOW_BUCKETS + 1u];
	double seconds, duration_created;
	double requests_created[METRICS_HTTP_STATUS_CLASSES];
};

struct server_metrics {
	bool enabled;
	uint64_t admitted, finished[METRICS_FINISH_COUNT];
	uint64_t admission_rejections;
	uint64_t refused_client, refused_server;
	struct metrics_histogram latencies[METRICS_LATENCY_COUNT];
	struct metrics_http_cell http[METRICS_HTTP_HANDLERS]
				     [METRICS_HTTP_METHODS];
	uint64_t http_count[METRICS_HTTP_HANDLERS],
		request_bytes[METRICS_HTTP_HANDLERS],
		response_bytes[METRICS_HTTP_HANDLERS];
	struct metrics_histogram http_high;
	/* Fixed first-use epochs: 2296 bytes across HTTP cells/handlers on Linux x86-64. */
	double http_created[METRICS_HTTP_HANDLERS];
	/* 256 bytes, prepared once for the immutable serving context. */
	double token_bounds[METRICS_BUCKET_COUNT];
	uint32_t token_context, token_bound_count;
	/* Poller-only scrape scratch: fixed at server allocation, never grown per scrape. */
	char *text; /* METRICS_TEXT_BYTES when enabled, NULL otherwise; server_close owns it */
};

struct http_server;
struct live_request_record;
struct engine_stats;
struct active_command;
struct connection;

void server_metrics_prepare(struct http_server *server);
void server_metrics_process_init(void);
void server_metrics_token(struct http_server *server,
			  struct active_command *active, double now);
void server_metrics_admitted(struct http_server *server,
			     const struct connection *connection);
void server_metrics_http_head(struct connection *connection,
			      bool messages_enabled);
void server_metrics_http_output(struct connection *connection);
void server_metrics_http_done(struct http_server *server,
			      struct connection *connection, double now);

/* The exported model label's startup contract; no allocation or mutation. */
bool server_metrics_model_valid(const char *model);
void server_metrics_first(struct http_server *server,
			  const struct live_request_record *request,
			  double now);
void server_metrics_finished(struct http_server *server,
			     const struct live_request_record *request);
void server_metrics_refused(struct http_server *server, uint32_t route,
			    uint32_t status);
MUST_CHECK result_status server_metrics_process(struct chat_buffer *output);
MUST_CHECK result_status server_metrics_text(struct http_server *server,
					     struct chat_buffer *output);
/* Pure renderer for private immutable snapshots; production caller is the single HTTP poller. */
MUST_CHECK result_status server_metrics_render(const struct http_server *server,
					       const struct engine_stats *engine,
					       struct chat_buffer *output);

#endif
