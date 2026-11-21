/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Real HTTP over the host-memory serving fixture; no CUDA driver calls or model files. */
#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include "engine/server/server_internal.h"
#include "support/serving_fixture.h"
#include "support/stats_tables.h"
#include "common/json.h"
#include "engine/storage/profile/storage_profile.h"
#include "engine/input/tokenizer/tokenizer.h"
#include "support/byte_identity.h"

#define TEST_TEXT_BYTES METRICS_TEXT_BYTES
#define TEST_PORT_FIRST 23000u
#define TEST_PORT_TRIES 128u
#define TEST_CONTEXT 4096u
#define TEST_OUTPUT 4u
#define TEST_LONG_OUTPUT 256u
#define TEST_SECONDS 10
#define TEST_PIPE_ENDS 2u
#define TEST_READERS 4u
#define TEST_SCRAPES 20u
#define TEST_POLL_TRIES 200u
#define TEST_PAUSE_NS 10000000L
#define TEST_STEP_NS 2000000ull
#define TEST_NANOSECONDS 1000000000ull
#define TEST_ARRIVAL 10.0
#define TEST_FINISHED 15.0
#define TEST_FIRST 12.0
#define TEST_QUEUE 0.5
#define TEST_EXPECTED_E2E 20.0
#define TEST_EXPECTED_TTFT 2.0
#define TEST_EXPECTED_TERMINALS 4u
#define TEST_BAD_FLOAT ((double)NAN)
#define TEST_INVALID_UTF8_BYTE 0xFF
#define TEST_GLOBAL_REFUSALS 6u
#define TEST_AUTH_HEADER_BYTES 256u
#define TEST_BODY_BYTES 512u
#define TEST_HTTP_CONTENT_BYTES 7u
#define TEST_DECIMAL_BASE 10

static int failures;
static const char *review_artifact_prefix;
static uint32_t port;
static uint32_t script[TEST_LONG_OUTPUT + 1u];
static const uint32_t *const scripts[] = { script };

struct running {
	struct http_server *server;
	struct inference_engine *engine;
	const struct text_tokenizer *tokenizer;
	const char *key_file, *stats_users;
	int stop[TEST_PIPE_ENDS];
	pthread_t thread;
	result_status status;
};

static void expect(bool good, const char *message)
{
	if (!good) {
		(void)printf("FAILED: %s\n", message);
		failures++;
	}
}

static double value_of(const char *text, const char *name, const char *label)
{
	for (const char *line = text; line != NULL && *line != '\0';) {
		const char *const end = strchr(line, '\n');
		if (strncmp(line, name, strlen(name)) == 0 &&
		    line[strlen(name)] == ' ' && label == NULL) {
			char *parsed = NULL;
			const double value =
				strtod(line + strlen(name) + 1u, &parsed);
			return parsed != line + strlen(name) + 1u &&
					       isfinite(value) ?
				       value :
				       TEST_BAD_FLOAT;
		}
		const char *const boundary = strchr(line, '}');
		if (strncmp(line, name, strlen(name)) == 0 &&
		    line[strlen(name)] == '{' && boundary != NULL &&
		    (end == NULL || boundary < end)) {
			const char *const found =
				label == NULL ? line : strstr(line, label);
			if (found != NULL && found < boundary) {
				char *parsed = NULL;
				const double value =
					strtod(boundary + 1, &parsed);
				return parsed != boundary + 1 &&
						       isfinite(value) ?
					       value :
					       TEST_BAD_FLOAT;
			}
		}
		line = end != NULL ? end + 1 : NULL;
	}
	return TEST_BAD_FLOAT;
}

static uint64_t count_of(const struct metrics_histogram *histogram)
{
	uint64_t count = 0;
	for (uint32_t bucket = 0; bucket <= METRICS_BUCKET_COUNT; bucket++) {
		count += histogram->buckets[bucket];
	}
	return count;
}

#define TEST_ACTUAL_OUTPUT 12u
#define TEST_LARGE_CONTEXT 1048576u
#define TEST_PATH_BYTES 4096u
#define TEST_RETAINED_POSITIONS 100u
#define TEST_LARGE_BODY 9999999999999999999ull

static void review_render(struct http_server *server,
			  const struct engine_stats *engine,
			  struct chat_buffer *output)
{
	output->count = 0;
	output->is_short = false;
	expect(server_metrics_render(server, engine, output) == STATUS_OK,
	       "review snapshot renders");
	chat_buffer_write(output, "", 1);
}

static void review_http(struct http_server *server, const char *head,
			uint32_t route, uint32_t status, uint64_t bytes)
{
	struct connection *const connection = &server->connections[0];
	*connection = (struct connection){
		.is_head_read = true,
		.head = { .route = route, .body_byte_count = bytes },
		.live = { .arrival_seconds = TEST_ARRIVAL },
		.metrics_status = status
	};
	chat_buffer_write(&connection->input, head, strlen(head));
	server_metrics_http_head(connection, false);
	server_metrics_http_done(server, connection, TEST_FINISHED);
	chat_buffer_free(&connection->input);
}

static void review_artifact(const struct chat_buffer *output,
			    const char *suffix)
{
	if (review_artifact_prefix == NULL)
		return;
	char path[TEST_PATH_BYTES];
	const int count = snprintf(path, sizeof(path), "%s.%s.prom",
				   review_artifact_prefix, suffix);
	expect(count > 0 && (size_t)count < sizeof(path),
	       "review artifact path");
	if (count <= 0 || (size_t)count >= sizeof(path))
		return;
	FILE *const file = fopen(path, "w");
	expect(file != NULL, "review artifact opens");
	if (file != NULL) {
		const bool written = fputs(output->bytes, file) >= 0;
		const int closed = fclose(file);
		expect(written && closed == 0,
		       "review artifact written and closed");
	}
}

/* Protocol failures never reach the ASGI application/instrumentator upstream. */
static void test_http_protocol_boundary(void)
{
	struct http_server *const server = calloc(1, sizeof(*server));
	expect(server != NULL, "protocol-boundary storage");
	if (server == NULL)
		return;
	server->description.session_slot_count =
		1; /* the run's tables, then no slot as before */
	expect(server_tables(server) == STATUS_OK, "the run's tables");
	server->description.session_slot_count = 0;
	server->metrics.enabled = true;
	static const struct {
		const char *bytes;
		uint32_t refusal;
	} malformed[] = {
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nX-A: \001\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1 x\r\nHost: x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/2.0\r\nHost: x\r\n\r\n",
		  HTTP_VERSION_NOT_SUPPORTED },
		{ "GET /v1/models invalid\r\nHost: x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "BROKEN\r\nHost: x\r\n\r\n", HTTP_BAD_REQUEST },
		{ "GET\r\nX-A: /v1/models HTTP/1.1\r\nHost: x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\nHost: x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nBad\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nBad Name: x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\n: x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nX\rY: x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nX\nY: x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nX: x\ry\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nX: x\ny\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\n X: y\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length:\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 1x\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 99999999999999999999\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n",
		  HTTP_BAD_REQUEST },
		/* Framing is found in either field order; the refusal stays the first failing line's. */
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nHost: y\r\nBad\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nBad\r\nHost: y\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nBad\r\n\r\n",
		  HTTP_NOT_IMPLEMENTED },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nBad\r\nTransfer-Encoding: chunked\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nExpect: other\r\nBad\r\n\r\n",
		  HTTP_EXPECTATION_FAILED },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nBad\r\nExpect: other\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 9999999999999999999\r\nBad\r\n\r\n",
		  HTTP_CONTENT_TOO_LARGE },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nBad\r\nContent-Length: 9999999999999999999\r\n\r\n",
		  HTTP_BAD_REQUEST },
		{ "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer a\r\n"
		  "Authorization: Bearer b\r\nBad\r\nContent-Length: 2\r\n\r\n{}",
		  HTTP_BAD_REQUEST },
		{ "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nBad\r\nAuthorization: Bearer a\r\n"
		  "Authorization: Bearer b\r\nContent-Length: 2\r\n\r\n{}",
		  HTTP_BAD_REQUEST },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nContent-Length: 1x\r\n\r\n",
		  HTTP_NOT_IMPLEMENTED }
	};
	struct connection *const connection = &server->connections[0];
	/* Existing children make the byte comparison cover every counter, histogram and epoch. */
	review_http(server, "GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n",
		    CHAT_ROUTE_MODELS, HTTP_OK, 0);
	unsigned char before[sizeof(server->metrics)];
	memcpy(before, &server->metrics, sizeof(before));
	for (uint32_t index = 0;
	     index < sizeof(malformed) / sizeof(malformed[0]); index++) {
		const char *const bytes = malformed[index].bytes;
		memcpy(&server->metrics, before, sizeof(before));
		*connection = (struct connection){
			.live = { .arrival_seconds = TEST_ARRIVAL },
			.metrics_status = malformed[index].refusal
		};
		chat_buffer_write(&connection->input, bytes, strlen(bytes));
		connection->is_head_read = server_head_read_metrics(
			bytes, strlen(bytes), true, &connection->head);
		expect(connection->is_head_read &&
			       connection->head.refusal ==
				       malformed[index].refusal,
		       "protocol-boundary parser refuses malformed syntax");
		server_metrics_http_head(connection, false);
		server_metrics_http_done(server, connection, TEST_FINISHED);
		expect(bytes_identical(before, &server->metrics,
				       sizeof(before)),
		       "protocol-malformed head leaves every HTTP family unchanged");
		chat_buffer_free(&connection->input);
	}
	char *const oversized = malloc(SERVER_MOST_HEAD_BYTES);
	expect(oversized != NULL, "oversized-head storage");
	if (oversized != NULL) {
		memcpy(&server->metrics, before, sizeof(before));
		static const char prefix[] = "GET /v1/models HTTP/1.1\r\nX-A: ";
		memset(oversized, 'x', SERVER_MOST_HEAD_BYTES);
		memcpy(oversized, prefix, sizeof(prefix) - 1u);
		*connection = (struct connection){
			.live = { .arrival_seconds = TEST_ARRIVAL },
			.metrics_status = HTTP_FIELDS_TOO_LARGE
		};
		chat_buffer_write(&connection->input, oversized,
				  SERVER_MOST_HEAD_BYTES);
		connection->is_head_read = server_head_read_metrics(
			oversized, SERVER_MOST_HEAD_BYTES, true,
			&connection->head);
		expect(connection->is_head_read &&
			       connection->head.refusal ==
				       HTTP_FIELDS_TOO_LARGE,
		       "unterminated oversized head is refused");
		server_metrics_http_head(connection, false);
		server_metrics_http_done(server, connection, TEST_FINISHED);
		expect(bytes_identical(before, &server->metrics,
				       sizeof(before)),
		       "protocol-oversized head leaves every HTTP family unchanged");
		chat_buffer_free(&connection->input);
		free(oversized);
	}
	/* Even an inconsistent supplied route cannot make path scanning cross the request line. */
	memcpy(&server->metrics, before, sizeof(before));
	review_http(server, "GET\r\nX-A: /v1/models HTTP/1.1\r\n\r\n",
		    CHAT_ROUTE_MODELS, HTTP_BAD_REQUEST, 0);
	expect(bytes_identical(before, &server->metrics, sizeof(before)),
	       "metrics path scan never consumes header bytes as a request target");
	/* Parse ordinary refusals too: excluding every nonzero refusal is incorrect. */
	static const struct {
		const char *bytes;
		uint32_t refusal, status, handler;
	} accepted[] = {
		{ "GET /unknown HTTP/1.1\r\nHost: x\r\n\r\n", HTTP_NOT_FOUND,
		  HTTP_NOT_FOUND, METRICS_HTTP_HANDLERS - 1u },
		{ "POST /v1/models HTTP/1.1\r\nHost: x\r\n\r\n",
		  HTTP_METHOD_NOT_ALLOWED, HTTP_METHOD_NOT_ALLOWED,
		  METRICS_HTTP_HANDLERS - 1u },
		{ "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n",
		  HTTP_LENGTH_REQUIRED, HTTP_LENGTH_REQUIRED,
		  CHAT_ROUTE_COMPLETIONS },
		{ "GET /v1/models HTTP/1.1\r\n\r\n", HTTP_BAD_REQUEST,
		  HTTP_BAD_REQUEST, CHAT_ROUTE_MODELS },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nHost: y\r\n\r\n",
		  HTTP_BAD_REQUEST, HTTP_BAD_REQUEST, CHAT_ROUTE_MODELS },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\n\r\nx",
		  HTTP_BAD_REQUEST, HTTP_BAD_REQUEST, CHAT_ROUTE_MODELS },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 9999999999999999999\r\n\r\n",
		  HTTP_CONTENT_TOO_LARGE, HTTP_CONTENT_TOO_LARGE,
		  CHAT_ROUTE_MODELS },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nExpect: other\r\n\r\n",
		  HTTP_EXPECTATION_FAILED, HTTP_EXPECTATION_FAILED,
		  CHAT_ROUTE_MODELS },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n",
		  HTTP_NOT_IMPLEMENTED, HTTP_NOT_IMPLEMENTED,
		  CHAT_ROUTE_MODELS },
		{ "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\n\r\n{",
		  0, HTTP_BAD_REQUEST, CHAT_ROUTE_COMPLETIONS },
		/* Well-formed fields after an application refusal keep it counted. */
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nX-A: b\r\n\r\n",
		  HTTP_NOT_IMPLEMENTED, HTTP_NOT_IMPLEMENTED,
		  CHAT_ROUTE_MODELS },
		{ "GET /v1/models HTTP/1.1\r\nHost: x\r\nHost: y\r\nContent-Length: 0\r\n\r\n",
		  HTTP_BAD_REQUEST, HTTP_BAD_REQUEST, CHAT_ROUTE_MODELS }
	};
	for (uint32_t index = 0; index < sizeof(accepted) / sizeof(accepted[0]);
	     index++) {
		const char *const bytes = accepted[index].bytes;
		memcpy(&server->metrics, before, sizeof(before));
		const uint64_t prior_count =
			server->metrics.http_count[accepted[index].handler];
		const double prior_duration = server->metrics.http_high.sum;
		*connection = (struct connection){
			.live = { .arrival_seconds = TEST_ARRIVAL },
			.metrics_status = accepted[index].status
		};
		chat_buffer_write(&connection->input, bytes, strlen(bytes));
		connection->is_head_read = server_head_read_metrics(
			bytes, strlen(bytes), true, &connection->head);
		expect(connection->is_head_read &&
			       connection->head.refusal ==
				       accepted[index].refusal,
		       "parser produces the expected routed or application refusal");
		server_metrics_http_head(connection, false);
		server_metrics_http_done(server, connection, TEST_FINISHED);
		expect(!connection->metrics_http_excluded &&
			       connection->metrics_handler ==
				       accepted[index].handler &&
			       server->metrics.http_count[accepted[index]
								  .handler] ==
				       prior_count + 1u &&
			       server->metrics.http_high.sum ==
				       prior_duration + TEST_FINISHED -
					       TEST_ARRIVAL,
		       "parser-accepted HTTP refusals retain count and duration observations");
		chat_buffer_free(&connection->input);
	}
	server_tables_free(server);
	free(server);
}

static void test_parity_review(void)
{
	struct http_server *const server = calloc(1, sizeof(*server));
	expect(server != NULL, "review storage");
	if (server == NULL)
		return;
	server->description.session_slot_count =
		1; /* the run's tables, then no slot as before */
	expect(server_tables(server) == STATUS_OK, "the run's tables");
	server->description.session_slot_count = 0;
	server->metrics.enabled = true;
	server->started_seconds = 1;
	server->model_alias = "anemone";
	server->description.context_token_limit = TEST_LARGE_CONTEXT;
	server->description.session_slot_count = 1;
	static struct test_stats_tables engine_tables;
	memset(&engine_tables, 0, sizeof(engine_tables));
	struct engine_stats engine = { .slots_total = 1,
				       .prompt_tokens = TEST_RETAINED_POSITIONS,
				       .cached_tokens = 1,
				       TEST_STATS_TABLES(engine_tables) };
	engine.sessions[0].position = TEST_RETAINED_POSITIONS;
	engine.sessions[0].context_capacity = TEST_LARGE_CONTEXT;
	struct chat_buffer output = { 0 };
	expect(chat_buffer_room(&output, METRICS_TEXT_BYTES) != NULL,
	       "review output");
	review_render(server, &engine, &output);
	review_artifact(&output, "fresh_large");
	expect(strstr(output.bytes, "# HELP http_requests_created ") == NULL &&
		       strstr(output.bytes,
			      "# HELP http_request_size_bytes_created ") ==
			       NULL &&
		       strstr(output.bytes,
			      "# TYPE http_response_size_bytes_created ") ==
			       NULL &&
		       strstr(output.bytes,
			      "# TYPE http_request_duration_seconds_created ") ==
			       NULL,
	       "review 6: lazy created metadata absent without children");
	expect(value_of(output.bytes, "vllm:kv_cache_usage_perc", NULL) == 0 &&
		       value_of(output.bytes, "anemone_cache_context_tokens",
				NULL) == TEST_RETAINED_POSITIONS,
	       "review 3: idle retained positions are free in vLLM, retained natively");
	engine.sessions[0].active = true;
	review_render(server, &engine, &output);
	expect(value_of(output.bytes, "vllm:kv_cache_usage_perc", NULL) ==
		       (double)TEST_RETAINED_POSITIONS / TEST_LARGE_CONTEXT,
	       "review 3: active positions contribute to vLLM KV usage");
	server->disable_prefix_reuse = true;
	review_render(server, &engine, &output);
	expect(value_of(output.bytes, "vllm:prefix_cache_queries_total",
			NULL) == 0 &&
		       value_of(output.bytes, "vllm:prefix_cache_hits_total",
				NULL) == 0,
	       "review 4: disabled prefix lookup does not record cache queries or hits");
	expect(strstr(output.bytes, "le=\"1e+06\"") != NULL &&
		       strstr(output.bytes, "le=\"1000000.0\"") == NULL,
	       "review 10: Go-style million-token bucket spelling");
	expect(strstr(output.bytes, "kv_cache_max_concurrency=\"1.0\"") != NULL,
	       "review 10: max concurrency is a float label");
	struct connection *const connection = &server->connections[0];
	connection->request.max_tokens = TEST_LARGE_CONTEXT;
	server_metrics_admitted(server, connection);
	expect(count_of(&server->metrics.latencies[METRICS_N]) == 0 &&
		       count_of(&server->metrics
					 .latencies[METRICS_MAX_GENERATION]) ==
			       0,
	       "review 1: parent n and actual maximum are terminal observations");
	struct live_request_record request = {
		.slot = 0,
		.request_tag = 1,
		.arrival_seconds = TEST_ARRIVAL,
		.finished_seconds = TEST_FINISHED,
		.options = { .max_tokens = TEST_LARGE_CONTEXT },
		.engine = { .request_tag = 1,
			    .submitted_ns = 1,
			    .generated_tokens = TEST_ACTUAL_OUTPUT }
	};
	(void)snprintf(request.finish_reason, sizeof(request.finish_reason),
		       "%s", "stop");
	server_metrics_finished(server, &request);
	expect(count_of(&server->metrics.latencies[METRICS_N]) == 1 &&
		       server->metrics.latencies[METRICS_MAX_GENERATION].sum ==
			       TEST_ACTUAL_OUTPUT &&
		       server->metrics.latencies[METRICS_MAX_TOKENS].sum ==
			       TEST_LARGE_CONTEXT,
	       "review 1: actual EOS output differs from admission limit");
	struct metrics_histogram before[METRICS_LATENCY_COUNT];
	memcpy(before, server->metrics.latencies, sizeof(before));
	(void)snprintf(request.finish_reason, sizeof(request.finish_reason),
		       "%s", "cancelled");
	server_metrics_finished(server, &request);
	expect(bytes_identical(before, server->metrics.latencies,
			       sizeof(before)) &&
		       server->metrics.finished[METRICS_FINISH_CANCELLED] == 1,
	       "review 2: client abort keeps native outcome, skips vLLM terminal histograms");
	review_render(server, &engine, &output);
	expect(value_of(output.bytes, "vllm:request_success_total",
			"finished_reason=\"abort\"") == 0,
	       "review 2: client disconnect is not vLLM request_success abort");
	review_http(server, "GET /v1/models HTTP/1.1\r\n\r\n",
		    CHAT_ROUTE_MODELS, HTTP_OK, TEST_LARGE_BODY);
	review_render(server, &engine, &output);
	const double created = value_of(output.bytes, "http_requests_created",
					"handler=\"/v1/models\"");
	expect(value_of(output.bytes, "http_request_size_bytes_created",
			"handler=\"/v1/models\"") == created &&
		       value_of(output.bytes,
				"http_response_size_bytes_created",
				"handler=\"/v1/models\"") == created &&
		       value_of(output.bytes,
				"http_request_duration_seconds_created",
				"handler=\"/v1/models\"") == created,
	       "review 6: every lazy HTTP family records the same first-use event");
	const double bytes = value_of(output.bytes,
				      "http_request_size_bytes_sum",
				      "handler=\"/v1/models\"");
	expect(created > (double)server->started_seconds,
	       "review 6: lazy child created at first use");
	review_http(server, "GET /v1/models HTTP/1.1\r\n\r\n",
		    CHAT_ROUTE_MODELS, HTTP_OK, TEST_LARGE_BODY);
	review_render(server, &engine, &output);
	expect(value_of(output.bytes, "http_requests_created",
			"handler=\"/v1/models\"") == created,
	       "review 6: existing child created timestamp remains stable");
	expect(value_of(output.bytes, "http_request_size_bytes_sum",
			"handler=\"/v1/models\"") >= bytes,
	       "review 8: refused declared request sizes never wrap");
	review_http(server, "POST /v1/models HTTP/1.1\r\n\r\n",
		    CHAT_ROUTE_MODELS, HTTP_METHOD_NOT_ALLOWED, 0);
	review_render(server, &engine, &output);
	expect(value_of(output.bytes, "http_requests_total",
			"handler=\"none\",method=\"POST\"") == 1,
	       "review 11: method mismatch has instrumentator handler none");
	static const char *const excluded[] = { "/metrics", "/health",
						"/load",    "/ping",
						"/version", "/server_info" };
	const uint64_t count_before =
		server->metrics.http_count[METRICS_HTTP_HANDLERS - 1u];
	for (uint32_t index = 0; index < sizeof(excluded) / sizeof(excluded[0]);
	     index++) {
		char head[TEST_BODY_BYTES];
		(void)snprintf(head, sizeof(head),
			       "GET /unknown%s/suffix?x=1 HTTP/1.1\r\n\r\n",
			       excluded[index]);
		review_http(server, head, METRICS_HTTP_HANDLERS - 1u,
			    HTTP_NOT_FOUND, 0);
	}
	expect(server->metrics.http_count[METRICS_HTTP_HANDLERS - 1u] ==
		       count_before,
	       "review 11: pinned path regex exclusions include unknown paths");
	output.count = 0;
	expect(server_metrics_process(&output) == STATUS_OK,
	       "review process warmup");
	struct rlimit original, restricted;
	const int got_limit = getrlimit(RLIMIT_NOFILE, &original);
	expect(got_limit == 0, "review descriptor limit available");
	if (got_limit == 0) {
		restricted = original;
		restricted.rlim_cur = 0;
		const int restricted_status =
			setrlimit(RLIMIT_NOFILE, &restricted);
		expect(restricted_status == 0,
		       "review descriptor exhaustion enabled");
		if (restricted_status == 0) {
			output.count = 0;
			const result_status status =
				server_metrics_process(&output);
			const int restored =
				setrlimit(RLIMIT_NOFILE, &original);
			expect(restored == 0,
			       "review descriptor limit restored");
			expect(status == STATUS_OK,
			       "review 7: optional proc failure does not fail scrape");
		}
	}
	review_http(server, "GET /v1/models HTTP/1.1\r\n\r\n",
		    CHAT_ROUTE_MODELS, HTTP_BAD_REQUEST, 0);
	review_render(server, &engine, &output);
	expect(value_of(output.bytes, "http_requests_created",
			"status=\"4xx\"") >= created &&
		       value_of(output.bytes,
				"http_request_duration_seconds_created",
				"handler=\"/v1/models\"") == created,
	       "review 6: later status child has its own epoch without resetting parent children");
	chat_buffer_free(&output);
	server_tables_free(server);
	free(server);
}

static void test_accounting(void)
{
	struct http_server *const server = calloc(1, sizeof(*server));
	if (server == NULL) {
		expect(false, "accounting fixture allocation");
		return;
	}
	server->description.session_slot_count =
		1; /* the run's tables, then no slot as before */
	expect(server_tables(server) == STATUS_OK, "the run's tables");
	server->description.session_slot_count = 0;
	server->metrics.enabled = true;
	static const char *const reasons[] = { "stop", "length", "tool_calls",
					       "error", "cancelled" };
	for (uint32_t reason = 0; reason < METRICS_FINISH_COUNT; reason++) {
		struct live_request_record request = {
			.slot = 0,
			.request_tag = 1,
			.arrival_seconds = TEST_ARRIVAL,
			.finished_seconds = TEST_FINISHED,
			.queue_wait_seconds = TEST_QUEUE,
			.engine = { .request_tag = 1,
				    .submitted_ns = TEST_NANOSECONDS,
				    .started_ns = TEST_NANOSECONDS,
				    .first_token_ns = TEST_NANOSECONDS,
				    .prefill_ns = TEST_NANOSECONDS,
				    .decode_ns = TEST_NANOSECONDS }
		};
		(void)snprintf(request.finish_reason,
			       sizeof(request.finish_reason), "%s",
			       reasons[reason]);
		server_metrics_finished(server, &request);
		expect(server->metrics.finished[reason] == 1,
		       "each outcome counted exactly once by terminal hook");
	}
	expect(count_of(&server->metrics.latencies[METRICS_E2E]) ==
			       TEST_EXPECTED_TERMINALS &&
		       server->metrics.latencies[METRICS_E2E].sum ==
			       TEST_EXPECTED_E2E,
	       "terminal histograms exclude client aborts, native counters include every outcome");
	struct live_request_record first = { .arrival_seconds = TEST_ARRIVAL };
	server_metrics_first(server, &first, TEST_FIRST);
	server_metrics_first(server, &first, NAN);
	server_metrics_first(server, &first, TEST_ARRIVAL - 1);
	expect(count_of(&server->metrics.latencies[METRICS_TTFT]) == 1 &&
		       server->metrics.latencies[METRICS_TTFT].sum ==
			       TEST_EXPECTED_TTFT,
	       "TTFT uses seconds and rejects nonfinite/negative observations");
	server_metrics_refused(server, CHAT_ROUTE_COMPLETIONS,
			       HTTP_BAD_REQUEST);
	server_metrics_refused(server, CHAT_ROUTE_RESPONSES, HTTP_UNAVAILABLE);
	server_metrics_refused(server, CHAT_ROUTE_METRICS, HTTP_UNAUTHORIZED);
	expect(server->metrics.refused_client == 1 &&
		       server->metrics.refused_server == 1,
	       "HTTP error classes exclude scrape authentication");
	server->metrics.enabled = false;
	server_metrics_first(server, &first, TEST_FIRST);
	expect(count_of(&server->metrics.latencies[METRICS_TTFT]) == 1,
	       "off switch prevents observations");
	server->model_alias = "anemone";
	static struct test_stats_tables engine_tables;
	memset(&engine_tables, 0, sizeof(engine_tables));
	struct engine_stats engine = { .now_ns = TEST_NANOSECONDS,
				       TEST_STATS_TABLES(engine_tables) };
	struct chat_buffer output = { 0 };
	expect(chat_buffer_room(&output, METRICS_TEXT_BYTES) != NULL,
	       "exporter test storage");
	expect(server_metrics_render(server, &engine, &output) == STATUS_OK,
	       "immutable counter snapshot renders");
	chat_buffer_write(&output, "", 1);
	expect(strstr(output.bytes, "le=\"0.1\"") != NULL &&
		       strstr(output.bytes, "0.10000000000000001") == NULL &&
		       strstr(output.bytes, "le=\"0.005\"") != NULL,
	       "histogram labels match documented decimal selectors");
	expect(strstr(output.bytes,
		      "anemone_speculation_proposed_tokens_total") == NULL,
	       "disabled speculation is omitted");
	engine.speculation_mode = STATS_SPECULATION_DEVICE;
	server->description.speculative_depth = 1;
	engine.speculation.source[DRAFT_DEVICE].rounds = 1;
	engine.speculation.source[DRAFT_DEVICE].proposed[0] = 1;
	engine.speculation.source[DRAFT_DEVICE].accepted[0] = 1;
	output.count = 0;
	expect(server_metrics_render(server, &engine, &output) == STATUS_OK,
	       "speculation snapshot renders");
	chat_buffer_write(&output, "", 1);
	expect(value_of(output.bytes,
			"anemone_speculation_proposed_tokens_total",
			NULL) == 1 &&
		       value_of(output.bytes,
				"anemone_speculation_accepted_tokens_total",
				NULL) == 1,
	       "enabled speculation retains existing proposed and accepted token semantics");
	expect(value_of(output.bytes, "vllm:spec_decode_num_drafts_total",
			NULL) == 1 &&
		       value_of(output.bytes,
				"vllm:spec_decode_num_draft_tokens_total",
				NULL) == 1 &&
		       value_of(
			       output.bytes,
			       "vllm:spec_decode_num_accepted_tokens_per_pos_total",
			       "position=\"0\"") == 1,
	       "vLLM speculative drafts and zero-based per-position accepted prefix counts");

	char longest[PACK_MODEL_NAME_CAPACITY];
	memset(longest, '\\', sizeof(longest) - 1u);
	longest[sizeof(longest) - 1u] = '\0';
	server->model_alias = longest;
	server->description.context_token_limit = UINT32_MAX;
	server->description.speculative_depth = DRAFT_STAT_POSITIONS;
	for (uint32_t handler = 0; handler < METRICS_HTTP_HANDLERS; handler++) {
		if (handler == CHAT_ROUTE_HEALTH ||
		    handler == CHAT_ROUTE_METRICS)
			continue;
		server->metrics.http_count[handler] = 1;
		for (uint32_t method = 0; method < METRICS_HTTP_METHODS;
		     method++) {
			struct metrics_http_cell *const cell =
				&server->metrics.http[handler][method];
			for (uint32_t status = 0;
			     status < METRICS_HTTP_STATUS_CLASSES; status++)
				cell->requests[status] = 1;
			cell->buckets[METRICS_HTTP_LOW_BUCKETS] =
				METRICS_HTTP_STATUS_CLASSES;
			cell->seconds = METRICS_HTTP_STATUS_CLASSES;
		}
	}
	output.count = 0;
	expect(server_metrics_render(server, &engine, &output) == STATUS_OK &&
		       output.count < METRICS_TEXT_BYTES,
	       "maximum escaped identity, 32-bit context and all bounded HTTP labels fit startup scratch");
	const char invalid_alias[] = { (char)TEST_INVALID_UTF8_BYTE, '\0' };
	server->model_alias = invalid_alias;
	expect(server_metrics_render(server, &engine, &output) ==
		       STATUS_INVALID_ARGUMENT,
	       "invalid UTF-8 startup identity cannot produce an invalid exposition");
	server->model_alias = "bad\rlabel";
	expect(server_metrics_render(server, &engine, &output) ==
		       STATUS_INVALID_ARGUMENT,
	       "raw carriage return cannot enter LF-only exposition");
	chat_buffer_free(&output);
	server_tables_free(server);
	free(server);
}

static void test_scrape_failure_accounting(void)
{
	struct http_server *const server = calloc(1, sizeof(*server));
	if (server == NULL) {
		expect(false, "scrape failure fixture allocation");
		return;
	}
	server->description.session_slot_count =
		1; /* the run's tables, then no slot as before */
	expect(server_tables(server) == STATUS_OK, "the run's tables");
	server->description.session_slot_count = 0;
	server->metrics.enabled = true;
	server->model_alias = "anemone";
	server->live.user_count = 1;
	(void)snprintf(server->live.users[0].user,
		       sizeof(server->live.users[0].user), "%s", "ops");
	struct connection *const connection = &server->connections[0];
	connection->head.route = CHAT_ROUTE_METRICS;
	(void)snprintf(connection->live.user, sizeof(connection->live.user),
		       "%s", "ops");
	static struct test_stats_tables engine_tables;
	memset(&engine_tables, 0, sizeof(engine_tables));
	struct engine_stats engine = { TEST_STATS_TABLES(engine_tables) };
	struct chat_buffer failed = { .is_short = true };
	expect(server_metrics_render(server, &engine, &failed) ==
		       STATUS_OUT_OF_MEMORY,
	       "failed exporter buffer returns a scrape failure");
	server_stats_refused(server, connection, HTTP_SERVER_ERROR, 0);
	expect(server->live.failed_requests == 0 &&
		       server->live.users[0].refused == 0 &&
		       server->metrics.admitted == 0 &&
		       server->live.finished_count == 0,
	       "failed scrape does not count as generation or user request failure");
	connection->head.route = CHAT_ROUTE_COMPLETIONS;
	server_stats_refused(server, connection, HTTP_SERVER_ERROR, 0);
	expect(server->live.failed_requests == 1 &&
		       server->live.users[0].refused == 1,
	       "generation HTTP errors retain existing accounting");
	server_tables_free(server);
	free(server);
}

static void test_startup_labels(void)
{
	static const char invalid_utf8[] = { (char)TEST_INVALID_UTF8_BYTE,
					     '\0' };
	static const char *const invalid[] = { invalid_utf8, "bad\ralias" };
	for (size_t index = 0; index < sizeof(invalid) / sizeof(invalid[0]);
	     index++) {
		struct server_options options = {
			.bind_address = SERVER_DEFAULT_ADDRESS,
			.port = TEST_PORT_FIRST,
			.model_alias = invalid[index],
			.metrics = true
		};
		struct http_server *server = NULL;
		expect(server_open(&options, &server) ==
				       STATUS_INVALID_ARGUMENT &&
			       server == NULL,
		       "metrics startup rejects an invalid model identity before binding");
		expect(!server_metrics_model_valid(invalid[index]),
		       "shared model label validation rejects invalid data");
		options.metrics = false;
		result_status opened = STATUS_INPUT_OUTPUT_ERROR;
		for (uint32_t attempt = 0;
		     attempt < TEST_PORT_TRIES && opened != STATUS_OK;
		     attempt++) {
			options.port = TEST_PORT_FIRST +
				       (uint32_t)getpid() % TEST_PORT_FIRST +
				       attempt;
			opened = server_open(&options, &server);
		}
		expect(opened == STATUS_OK,
		       "metrics-off alias admission retains the base behavior");
		server_close(server);
	}
	expect(server_metrics_model_valid("valid\\\"\nalias"),
	       "escapeable model identity is admitted");
}

static void *run_server(void *argument)
{
	struct running *const running = argument;
	running->status = server_run(running->server, running->engine,
				     running->tokenizer, running->stop[0]);
	return NULL;
}

static bool start(struct running *running, bool metrics)
{
	static char storage_text[STORAGE_PROFILE_METRICS_BYTES];
	static struct storage_profile profile;
	profile.observation_count = STORAGE_PROFILE_OBSERVATIONS;
	profile.observations[0] = (struct storage_replica_observation){
		.mount = "/storage-fixture",
		.reason = "verified_identity_and_bytes",
		.included = true,
		.load_active = true,
		.lookup_active = true,
		.physical = { 1 },
		.physical_count = 1
	};
	for (uint32_t index = 1; index < STORAGE_PROFILE_OBSERVATIONS;
	     index++) {
		profile.observations[index] = profile.observations[0];
		profile.observations[index].physical[0] = index + 1u;
		(void)snprintf(profile.observations[index].mount,
			       sizeof(profile.observations[index].mount),
			       "/storage-fixture/%u", index);
	}
	FILE *const storage_stream =
		fmemopen(storage_text, sizeof(storage_text), "w");
	if (storage_stream == NULL)
		return false;
	const result_status storage_status =
		storage_profile_metrics(storage_stream, &profile, true);
	const int storage_closed = fclose(storage_stream);
	if (storage_status != STATUS_OK || storage_closed != 0)
		return false;
	if (pipe(running->stop) != 0) {
		return false;
	}
	struct server_options options = {
		.key_file = running->key_file,
		.stats_users = running->stats_users,
		.bind_address = SERVER_DEFAULT_ADDRESS,
		.model_alias = "fixture\\\"\nmodel",
		.metrics = metrics,
		.storage_replica_info = storage_text,
		.storage_replica_info_bytes = strlen(storage_text),
		.disable_prefix_reuse = true,
		.queue_depth = TEST_OUTPUT,
		.user_queue_depth = TEST_OUTPUT,
		.defaults = { .max_tokens = TEST_OUTPUT,
			      .top_p = 1.0f,
			      .is_seeded = true }
	};
	result_status opened = STATUS_INPUT_OUTPUT_ERROR;
	for (uint32_t attempt = 0;
	     attempt < TEST_PORT_TRIES && opened != STATUS_OK; attempt++) {
		port = TEST_PORT_FIRST +
		       ((uint32_t)getpid() % TEST_PORT_FIRST) + attempt;
		options.port = port;
		opened = server_open(&options, &running->server);
	}
	if (opened != STATUS_OK ||
	    pthread_create(&running->thread, NULL, run_server, running) != 0) {
		server_close(running->server);
		(void)close(running->stop[0]);
		(void)close(running->stop[1]);
		return false;
	}
	return true;
}

static void stop(struct running *running)
{
	const char byte = 'x';
	expect(write(running->stop[1], &byte, 1) == 1 &&
		       pthread_join(running->thread, NULL) == 0,
	       "server thread retires");
	expect(running->status == STATUS_OK, "server_run status");
	server_close(running->server);
	expect(close(running->stop[0]) == 0 && close(running->stop[1]) == 0,
	       "stop descriptors close");
}

static int connect_request(const char *request)
{
	const int descriptor = socket(AF_INET, SOCK_STREAM, 0);
	const struct timeval deadline = { .tv_sec = TEST_SECONDS };
	struct sockaddr_in address = { .sin_family = AF_INET,
				       .sin_port = htons((uint16_t)port) };
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if (descriptor < 0 ||
	    setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &deadline,
		       sizeof(deadline)) != 0 ||
	    setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &deadline,
		       sizeof(deadline)) != 0 ||
	    connect(descriptor, (const struct sockaddr *)&address,
		    sizeof(address)) != 0) {
		if (descriptor >= 0) {
			(void)close(descriptor);
		}
		return -1;
	}
	size_t sent = 0;
	while (sent < strlen(request)) {
		const ssize_t count = send(descriptor, request + sent,
					   strlen(request) - sent,
					   MSG_NOSIGNAL);
		if (count <= 0) {
			(void)close(descriptor);
			return -1;
		}
		sent += (size_t)count;
	}
	return descriptor;
}

static bool exchange(const char *request, char *response)
{
	const int descriptor = connect_request(request);
	if (descriptor < 0) {
		return false;
	}
	size_t received = 0;
	ssize_t count = 0;
	while (received < TEST_TEXT_BYTES - 1u &&
	       (count = recv(descriptor, response + received,
			     TEST_TEXT_BYTES - 1u - received, 0)) > 0) {
		received += (size_t)count;
	}
	response[received] = '\0';
	const bool complete = count == 0 && close(descriptor) == 0;
	return complete && received != 0;
}

static bool scrape(char *response)
{
	return exchange("GET /metrics HTTP/1.1\r\nHost: fixture\r\n\r\n",
			response) &&
	       strstr(response, "HTTP/1.1 200 OK\r\n") == response &&
	       strstr(response,
		      "anemone_storage_replica_info{mount=\"/storage-fixture\",role=\"load\",status=\"included\","
		      "reason=\"verified_identity_and_bytes\"} 1\n") != NULL &&
	       strstr(response,
		      "anemone_storage_replica_info{mount=\"/storage-fixture\",role=\"lookup\",status=\"included\","
		      "reason=\"verified_identity_and_bytes\"} 1\n") != NULL &&
	       strstr(response,
		      "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n") !=
		       NULL;
}

static void *scrape_worker(void *argument)
{
	bool *const good = argument;
	char *const response = malloc(TEST_TEXT_BYTES);
	*good = response != NULL;
	for (uint32_t index = 0; *good && index < TEST_SCRAPES; index++) {
		*good = scrape(response) &&
			strstr(response, "le=\"+Inf\"") != NULL;
	}
	free(response);
	return NULL;
}

static void generation_request(char *request, uint32_t limit)
{
	char body[TEST_BODY_BYTES];
	const int count = snprintf(
		body, sizeof(body),
		"{\"messages\":[{\"role\":\"user\",\"content\":\"private-test-prompt\"}],\"max_tokens\":%u}",
		limit);
	expect(count > 0 && (size_t)count < sizeof(body),
	       "generation body fits");
	const int request_count = snprintf(
		request, TEST_TEXT_BYTES,
		"POST /v1/chat/completions HTTP/1.1\r\nHost: fixture\r\nContent-Length: %zu\r\n\r\n%s",
		strlen(body), body);
	expect(request_count > 0 && (size_t)request_count < TEST_TEXT_BYTES,
	       "generation request fits");
}

static bool wait_metric(char *response, const char *name, const char *label,
			double expected)
{
	const struct timespec pause = { .tv_nsec = TEST_PAUSE_NS };
	for (uint32_t attempt = 0; attempt < TEST_POLL_TRIES; attempt++) {
		if (scrape(response) &&
		    value_of(response, name, label) == expected) {
			return true;
		}
		(void)nanosleep(&pause, NULL);
	}
	return false;
}

static struct json_value stats_member(struct json_value object,
				      const char *name)
{
	char bytes[TEST_AUTH_HEADER_BYTES];
	struct json_scratch scratch = { .bytes = bytes,
					.capacity = sizeof(bytes) };
	struct json_value key = { 0 }, value = { 0 };
	for (size_t cursor = object.begin;
	     object.text != NULL && json_next(object, &cursor, &key, &value);) {
		if (json_string_is(&scratch, key, name)) {
			return value;
		}
	}
	return (struct json_value){ 0 };
}

static uint64_t stats_integer(struct json_value object, const char *name)
{
	const struct json_value value = stats_member(object, name);
	expect(value.text != NULL, "statistics counter exists");
	return value.text != NULL ? strtoull(value.text + value.begin, NULL,
					     TEST_DECIMAL_BASE) :
				    0;
}

static void
refusal_counts(const char *auth,
	       uint64_t counts[TEST_GLOBAL_REFUSALS + SERVER_MOST_USERS])
{
	static const char *const fields[TEST_GLOBAL_REFUSALS] = {
		"refused_503",	   "refused_400",     "refused_auth",
		"failed_requests", "served_requests", "cancelled_requests"
	};
	char request[TEST_AUTH_HEADER_BYTES], response[TEST_TEXT_BYTES];
	(void)snprintf(request, sizeof(request),
		       "GET /v1/anemone/stats HTTP/1.1\r\nHost: x\r\n%s\r\n",
		       auth);
	expect(exchange(request, response) &&
		       strstr(response, "HTTP/1.1 200 OK") == response,
	       "authorized JSON statistics succeeds");
	const char *const body = strstr(response, "\r\n\r\n");
	struct json_value document = { 0 };
	expect(body != NULL &&
		       json_read_document(body + strlen("\r\n\r\n"),
					  strlen(body + strlen("\r\n\r\n")),
					  &document) == JSON_DOCUMENT_VALID,
	       "statistics JSON is valid");
	const struct json_value global = stats_member(document, "global");
	for (uint32_t index = 0; index < TEST_GLOBAL_REFUSALS; index++) {
		counts[index] = stats_integer(global, fields[index]);
	}
	for (uint32_t index = 0; index < SERVER_MOST_USERS; index++) {
		counts[TEST_GLOBAL_REFUSALS + index] = 0;
	}
	const struct json_value users = stats_member(global, "users");
	struct json_value user = { 0 };
	uint32_t count = 0;
	for (size_t cursor = users.begin;
	     count < SERVER_MOST_USERS &&
	     json_next(users, &cursor, NULL, &user);) {
		counts[TEST_GLOBAL_REFUSALS + count++] =
			stats_integer(user, "refused");
	}
}

static void test_disabled_routes(struct inference_engine *engine,
				 const struct text_tokenizer *tokenizer,
				 const char *key_path, const char *credentials,
				 char *request, char *response)
{
	struct running running = { .engine = engine,
				   .tokenizer = tokenizer,
				   .key_file = key_path,
				   .stats_users = "ops" };
	if (!start(&running, false)) {
		expect(false, "disabled authenticated server starts");
		return;
	}
	char ops[TEST_AUTH_HEADER_BYTES], reader[TEST_AUTH_HEADER_BYTES],
		unknown[TEST_TEXT_BYTES];
	(void)snprintf(ops, sizeof(ops), "Authorization: Bearer %.*s\r\n",
		       (int)(strchr(credentials, '\n') - credentials),
		       credentials);
	const char *const reader_key = strchr(credentials, '\n') + 1;
	(void)snprintf(reader, sizeof(reader), "Authorization: Bearer %.*s\r\n",
		       (int)(strlen(reader_key) - 1u), reader_key);
	const char *const auth[] = {
		"", "Authorization: Bearer wrong-placeholder-value\r\n", ops,
		reader
	};
	static const struct {
		const char *method, *fields, *body;
	} cases[] = { { "GET", "", "" },
		      { "POST", "Content-Length: 0\r\n", "" },
		      { "POST", "", "" },
		      { "GET", "Content-Length: 1\r\n", "x" } };
	for (size_t item = 0; item < sizeof(cases) / sizeof(cases[0]); item++) {
		for (size_t identity = 0;
		     identity < sizeof(auth) / sizeof(auth[0]); identity++) {
			uint64_t before[TEST_GLOBAL_REFUSALS +
					SERVER_MOST_USERS],
				after_unknown[TEST_GLOBAL_REFUSALS +
					      SERVER_MOST_USERS],
				after_metrics[TEST_GLOBAL_REFUSALS +
					      SERVER_MOST_USERS];
			refusal_counts(ops, before);
			(void)snprintf(
				request, TEST_TEXT_BYTES,
				"%s /unknown HTTP/1.1\r\nHost: x\r\n%s%s\r\n%s",
				cases[item].method, auth[identity],
				cases[item].fields, cases[item].body);
			expect(exchange(request, unknown),
			       "unknown-path response arrives");
			refusal_counts(ops, after_unknown);
			(void)snprintf(
				request, TEST_TEXT_BYTES,
				"%s /metrics HTTP/1.1\r\nHost: x\r\n%s%s\r\n%s",
				cases[item].method, auth[identity],
				cases[item].fields, cases[item].body);
			expect(exchange(request, response) &&
				       strcmp(response, unknown) == 0 &&
				       strstr(response, "HTTP/1.1 404 ") ==
					       response,
			       "disabled metrics matches unknown-path bytes before auth/method/body policy");
			refusal_counts(ops, after_metrics);
			for (uint32_t counter = 0;
			     counter < TEST_GLOBAL_REFUSALS + SERVER_MOST_USERS;
			     counter++) {
				expect(after_unknown[counter] ==
						       before[counter] &&
					       after_metrics[counter] ==
						       after_unknown[counter],
				       "disabled metrics and unknown path have identical zero global/per-user stats "
				       "deltas");
			}
		}
	}
	stop(&running);
}

static void test_authentication(struct inference_engine *engine,
				const struct text_tokenizer *tokenizer,
				char *request, char *response)
{
	char path[] = "/tmp/anemone_metrics_fixture_XXXXXX";
	const int descriptor = mkstemp(path);
	static const char credentials[] =
		"ops_0000000000000000000000000000000000000000000000000000000000000000\n"
		"reader_1111111111111111111111111111111111111111111111111111111111111111\n";
	expect(descriptor >= 0, "disposable authentication fixture opens");
	if (descriptor < 0) {
		return;
	}
	expect(write(descriptor, credentials, sizeof(credentials) - 1u) ==
			       (ssize_t)(sizeof(credentials) - 1u) &&
		       close(descriptor) == 0,
	       "placeholder credentials written privately");
	struct running running = { .engine = engine,
				   .tokenizer = tokenizer,
				   .key_file = path,
				   .stats_users = "ops" };
	if (start(&running, true)) {
		expect(exchange("GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n",
				response) &&
			       strstr(response, "HTTP/1.1 401 ") == response,
		       "metrics requires serving bearer credentials");
		(void)snprintf(
			request, TEST_TEXT_BYTES,
			"GET /metrics HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer %.*s\r\n\r\n",
			(int)(strchr(credentials, '\n') - credentials),
			credentials);
		expect(exchange(request, response) &&
			       strstr(response, "HTTP/1.1 200 OK") ==
				       response &&
			       strstr(response, "0000000000000000") == NULL &&
			       strstr(response, "user=") == NULL,
		       "authorized metrics excludes key and user identity");
		const char *const reader = strchr(credentials, '\n') + 1;
		(void)snprintf(
			request, TEST_TEXT_BYTES,
			"GET /metrics HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer %.*s\r\n\r\n",
			(int)(strlen(reader) - 1u), reader);
		expect(exchange(request, response) &&
			       strstr(response, "HTTP/1.1 403 ") == response,
		       "stats_users restricts metrics");
		expect(exchange(
			       "POST /metrics HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n",
			       response) &&
			       strstr(response, "HTTP/1.1 405 ") == response,
		       "metrics rejects wrong method");
		expect(exchange("GET /metrics?x=1 HTTP/1.1\r\nHost: x\r\n\r\n",
				response) &&
			       strstr(response, "HTTP/1.1 404 ") == response &&
			       strstr(response,
				      "anemone_storage_replica_info") == NULL,
		       "metrics path is exact");
		expect(exchange("GET /health HTTP/1.1\r\nHost: x\r\n\r\n",
				response) &&
			       strstr(response, "HTTP/1.1 200 OK") == response,
		       "health keeps anonymous policy");
		stop(&running);
	} else {
		expect(false, "authenticated server starts");
	}
	test_disabled_routes(engine, tokenizer, path, credentials, request,
			     response);
	expect(unlink(path) == 0, "disposable credentials removed");
}

static void test_visible_http_head(void)
{
	static const char response[] =
		"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\n";
	struct connection connection = { 0 };
	char bytes[TEST_BODY_BYTES] = { 0 };
	const size_t prefix = sizeof(response);
	(void)memcpy(bytes + prefix, response, sizeof(response) - 1u);
	connection.answer.output =
		(struct chat_buffer){ .bytes = bytes,
				      .count = prefix + sizeof(response) - 1u,
				      .sent = prefix };
	server_metrics_http_output(&connection);
	expect(connection.metrics_status == HTTP_OK &&
		       connection.metrics_response_bytes ==
			       TEST_HTTP_CONTENT_BYTES,
	       "HTTP accounting reads a reserved JSON head at its wire offset");
}

int main(int argument_count, char **arguments)
{
	review_artifact_prefix = argument_count > 1 ? arguments[1] : NULL;
	test_visible_http_head();
	test_http_protocol_boundary();
	test_parity_review();
	test_accounting();
	test_scrape_failure_accounting();
	test_startup_labels();
	for (uint32_t index = 0; index < TEST_LONG_OUTPUT; index++) {
		script[index] = 'x';
	}
	script[TEST_LONG_OUTPUT] = TEST_SERVING_END_TOKEN;
	uint64_t table_bytes = 0;
	void *const tables = test_serving_tables(&table_bytes);
	struct text_tokenizer *tokenizer = NULL;
	struct inference_engine *engine = NULL;
	if (tables == NULL ||
	    tokenizer_open(tables, table_bytes, &tokenizer) != STATUS_OK) {
		free(tables);
		return 1;
	}
	char *const response = malloc(TEST_TEXT_BYTES), *const request = malloc(
								TEST_TEXT_BYTES);
	if (response == NULL || request == NULL) {
		free(response);
		free(request);
		tokenizer_close(tokenizer);
		free(tables);
		return 1;
	}

	/* Single-token requests may enqueue the token and terminal in the same poll batch. */
	expect(test_serving_open(TEST_CONTEXT, scripts, 1, &engine) ==
		       STATUS_OK,
	       "single-token engine opens");
	struct running single = { .engine = engine, .tokenizer = tokenizer };
	if (start(&single, true)) {
		double previous_e2e = 0, previous_ttft = 0, previous_http = 0;
		for (uint32_t attempt = 0; attempt < TEST_SCRAPES; attempt++) {
			generation_request(request, 1);
			expect(exchange(request, response),
			       "single-token request returns");
			expect(wait_metric(response,
					   "anemone_requests_finished_total",
					   "finished_reason=\"length\"",
					   attempt + 1u),
			       "single-token terminal lands");
			const double e2e = value_of(
				response,
				"vllm:e2e_request_latency_seconds_sum", NULL);
			const double ttft = value_of(
				response,
				"vllm:time_to_first_token_seconds_sum", NULL);
			expect(e2e - previous_e2e >= ttft - previous_ttft,
			       "terminal latency never precedes first token observation");

			const double http = value_of(
				response, "http_request_duration_seconds_sum",
				"handler=\"/v1/chat/completions\"");
			expect(http - previous_http >= e2e - previous_e2e,
			       "HTTP completion never precedes terminal observation");
			previous_http = http;
			previous_e2e = e2e;
			previous_ttft = ttft;
		}
		stop(&single);
	} else
		expect(false, "single-token server starts");
	test_serving_close();
	/* A new engine/server on each round must reset tokens, outcomes and histograms. */
	for (uint32_t round = 0; round < TEST_PIPE_ENDS; round++) {
		expect(test_serving_open(TEST_CONTEXT, scripts, 1, &engine) ==
			       STATUS_OK,
		       "host engine opens");
		struct running running = { .engine = engine,
					   .tokenizer = tokenizer };
		if (!start(&running, true)) {
			expect(false, "HTTP server starts");
			test_serving_close();
			break;
		}
		expect(scrape(response), "initial metrics scrape");
		expect(value_of(response, "vllm:generation_tokens_total",
				NULL) == 0 &&
			       value_of(response,
					"vllm:e2e_request_latency_seconds_count",
					NULL) == 0,
		       "engine restart resets counters and histograms");
		expect(strstr(response,
			      "model_name=\"fixture\\\\\\\"\\nmodel\"") != NULL,
		       "startup labels escape quote/backslash/newline");
		generation_request(request, TEST_OUTPUT);
		expect(exchange(request, response) &&
			       strstr(response, "HTTP/1.1 200 OK") != NULL,
		       "generation succeeds");
		expect(wait_metric(response, "anemone_requests_finished_total",
				   "finished_reason=\"length\"", 1),
		       "success terminal accounting lands");
		expect(value_of(response, "vllm:generation_tokens_total",
				NULL) == TEST_OUTPUT &&
			       value_of(response,
					"vllm:time_to_first_token_seconds_count",
					NULL) == 1,
		       "successful token and TTFT accounting");

		expect(value_of(response,
				"vllm:inter_token_latency_seconds_count",
				NULL) == TEST_OUTPUT - 1u &&
			       value_of(response,
					"vllm:request_generation_tokens_sum",
					NULL) == TEST_OUTPUT &&
			       value_of(response, "vllm:request_params_n_sum",
					NULL) == 1 &&
			       value_of(response,
					"vllm:request_params_max_tokens_sum",
					NULL) == TEST_OUTPUT &&
			       value_of(response,
					"vllm:iteration_tokens_total_sum",
					NULL) ==
				       value_of(response,
						"anemone_prefill_tokens_total",
						NULL) +
					       TEST_OUTPUT,
		       "vLLM token, parameter and iteration accounting on real host engine");
		expect(value_of(response, "process_resident_memory_bytes",
				NULL) > 0,
		       "Linux process collector emits positive finite resident memory");
		expect(value_of(response, "http_requests_total",
				"handler=\"/v1/chat/completions\"") == 1 &&
			       value_of(response, "http_request_size_bytes_sum",
					"handler=\"/v1/chat/completions\"") >
				       0 &&
			       strstr(response, "handler=\"/metrics\"") == NULL,
		       "HTTP instrumentator counts generation, sizes and excludes scrapes");
		expect(engine_stats_metrics_enable(engine, true) ==
			       STATUS_BUSY_VALUE,
		       "iteration collection cannot be enabled after generation admission");
		expect(strstr(response, "private-test-prompt") == NULL &&
			       strstr(response, "user=") == NULL,
		       "no prompt or user label leaks");
		expect(exchange(
			       "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\n\r\n{",
			       response),
		       "malformed HTTP request returns");
		expect(scrape(response) &&
			       value_of(response,
					"anemone_request_refusals_total",
					"status_class=\"4xx\"") == 1,
		       "early refusal is accounted separately");
		test_serving_set_step_delay(TEST_STEP_NS);
		generation_request(request, TEST_LONG_OUTPUT);
		const int active = connect_request(request);
		expect(active >= 0 && wait_metric(response,
						  "vllm:num_requests_running",
						  NULL, 1),
		       "live running gauge");
		const int queued = connect_request(request);
		expect(queued >= 0 && wait_metric(response,
						  "vllm:num_requests_waiting",
						  NULL, 1),
		       "live waiting gauge");
		pthread_t readers[TEST_READERS];
		bool good[TEST_READERS] = { false };
		uint32_t created = 0;
		for (; created < TEST_READERS; created++) {
			if (pthread_create(&readers[created], NULL,
					   scrape_worker,
					   &good[created]) != 0) {
				break;
			}
		}
		for (uint32_t index = 0; index < created; index++) {
			expect(pthread_join(readers[index], NULL) == 0 &&
				       good[index],
			       "concurrent HTTP scrapes are complete");
		}
		expect(created == TEST_READERS,
		       "all concurrent scrapers started");
		if (queued >= 0) {
			expect(close(queued) == 0, "queued client disconnects");
		}
		expect(wait_metric(response, "anemone_requests_finished_total",
				   "finished_reason=\"cancelled\"", 1),
		       "queued disconnect has one terminal cancellation");
		if (active >= 0) {
			expect(close(active) == 0, "active client disconnects");
		}
		expect(wait_metric(response, "anemone_requests_finished_total",
				   "finished_reason=\"cancelled\"",
				   TEST_PIPE_ENDS),
		       "running cancellation has one terminal outcome");
		expect(value_of(response, "vllm:num_requests_running", NULL) ==
				       0 &&
			       value_of(response, "vllm:num_requests_waiting",
					NULL) == 0 &&
			       value_of(response,
					"anemone_requests_admitted_total",
					NULL) == TEST_PIPE_ENDS + 1u,
		       "admitted equals terminal outcomes after cancellation drain");
		test_serving_set_step_delay(0);
		test_serving_set_fault_position(0);
		generation_request(request, TEST_OUTPUT);
		expect(exchange(request, response),
		       "engine fault response returns");
		expect(wait_metric(response, "anemone_requests_finished_total",
				   "finished_reason=\"error\"", 1),
		       "engine error is accounted once");
		expect(value_of(response, "anemone_model_ready", NULL) == 0,
		       "engine failure clears readiness");
		if (argument_count > 1 && round == 0) {
			const char *const body = strstr(response, "\r\n\r\n");
			FILE *const output = fopen(arguments[1], "w");
			expect(output != NULL && body != NULL,
			       "exposition artifact opens");
			if (output != NULL) {
				expect(body != NULL &&
					       fputs(body + strlen("\r\n\r\n"),
						     output) >= 0 &&
					       fclose(output) == 0,
				       "exposition artifact written");
			}
		}
		stop(&running);
		test_serving_close();
	}
	expect(test_serving_open(TEST_CONTEXT, scripts, 1, &engine) ==
		       STATUS_OK,
	       "disabled engine opens");
	test_authentication(engine, tokenizer, request, response);
	struct running disabled = { .engine = engine, .tokenizer = tokenizer };
	if (start(&disabled, false)) {
		expect(exchange("GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n",
				response) &&
			       strstr(response, "HTTP/1.1 404 ") == response &&
			       strstr(response,
				      "anemone_storage_replica_info") == NULL,
		       "metrics defaults off");
		stop(&disabled);
	} else {
		expect(false, "disabled server starts");
	}
	test_serving_close();
	free(response);
	free(request);
	tokenizer_close(tokenizer);
	free(tables);
	(void)printf("server_metrics_test: %d failures\n", failures);
	return failures == 0 ? 0 : 1;
}
