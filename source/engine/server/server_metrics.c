/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* vLLM metric metadata and bucket contract: see docs/METRICS.md for the pinned upstream revision. */
#include "server_internal.h"
#include "common/hooks.h"
#include "engine/formats/pack/pack.h"
#include "engine/input/tokenizer/tokenizer.h"
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define METRICS_LINE_BYTES 2048u
#define METRICS_ESCAPE_MULTIPLIER 2u
#define METRICS_ESCAPE_BYTES \
	(LIVE_STATS_MODEL_BYTES * METRICS_ESCAPE_MULTIPLIER + 1u)
#define METRICS_FORMAT_PARAMETER 2
#define METRICS_FORMAT_VALUES 3
#define METRICS_NANOSECONDS_PER_SECOND 1000000000.0
#define METRICS_CLIENT_STATUS_END 500u
#define METRICS_TOKEN_SCALE 10u
#define METRICS_TOKEN_FAMILY 6u
#define METRICS_BOUND_TEXT_BYTES 32u
#define METRICS_NAME_BYTES 128u
#define METRICS_SCIENTIFIC_BOUND 1000000.0

static const double bounds_0[] = { 0.001, 0.005, 0.01,	0.02,  0.04, 0.06,
				   0.08,  0.1,	 0.25,	0.5,   0.75, 1.0,
				   2.5,	  5.0,	 7.5,	10.0,  20.0, 40.0,
				   80.0,  160.0, 640.0, 2560.0 };
static const double bounds_1[] = { 0.3,	  0.5,	  0.8,	 1.0,	1.5,   2.0,
				   2.5,	  5.0,	  10.0,	 15.0,	20.0,  30.0,
				   40.0,  50.0,	  60.0,	 120.0, 240.0, 480.0,
				   960.0, 1920.0, 7680.0 };
static const double bounds_2[] = { 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,
				   0.3,	 0.4,	0.5,  0.75,  1.0, 2.5,	5.0,
				   7.5,	 10.0,	20.0, 40.0,  80.0 };
static const double bounds_3[] = { 1,	8,    16,   32,	  64,	128,  256,
				   512, 1024, 2048, 4096, 8192, 16384 };
static const double bounds_4[] = { 1, 2, 5, 10, 20 };
static const double bounds_5[] = { 1, 2, 3, 4, 5, 10, 20 };
struct histogram_definition {
	const char *name, *help;
	uint32_t family;
};
static const struct histogram_definition definitions[METRICS_LATENCY_COUNT] = {
	{ "vllm:time_to_first_token_seconds",
	  "Histogram of time to first token in seconds.", 0 },
	{ "vllm:request_queue_time_seconds",
	  "Histogram of time spent in WAITING phase for request.", 1 },
	{ "vllm:request_prefill_time_seconds",
	  "Histogram of time spent in PREFILL phase for request.", 1 },
	{ "vllm:request_decode_time_seconds",
	  "Histogram of time spent in DECODE phase for request.", 1 },
	{ "vllm:e2e_request_latency_seconds",
	  "Histogram of e2e request latency in seconds.", 1 },
	{ "vllm:request_inference_time_seconds",
	  "Histogram of time spent in RUNNING phase for request.", 1 },
	{ "vllm:inter_token_latency_seconds",
	  "Histogram of inter-token latency in seconds.", 2 },
	{ "vllm:request_time_per_output_token_seconds",
	  "Histogram of time_per_output_token_seconds per request.", 2 },
	{ "vllm:request_prompt_tokens", "Number of prefill tokens processed.",
	  6 },
	{ "vllm:request_generation_tokens",
	  "Number of generation tokens processed.", 6 },
	{ "vllm:request_params_max_tokens",
	  "Histogram of the max_tokens request parameter.", 6 },
	{ "vllm:request_max_num_generation_tokens",
	  "Histogram of maximum number of requested generation tokens.", 6 },
	{ "vllm:request_params_n", "Histogram of the n request parameter.", 4 },
	{ "vllm:request_num_preemptions",
	  "Histogram of the number of times a request was preempted.", 5 },
	{ "vllm:request_prefill_kv_computed_tokens",
	  "Histogram of new KV tokens computed during prefill (excluding cached tokens).",
	  6 },
};
/* By enum chat_calls_repair: the label values of anemone_dsml_repairs_total. */
static const char *const calls_repair_kinds[CHAT_CALLS_REPAIR_KINDS] = {
	"block_start",	   "marker_spelling", "attribute",
	"closing_implied", "marker_stray",    "parameter_repeated",
	"name_folded",	   "name_unknown",    "invoke_unnamed",
	"text_in_block",   "block_as_text",   "cut_at_limit"
};
static const char *const finish_reasons[METRICS_FINISH_COUNT] = {
	"stop", "length", "tool_calls", "error", "cancelled"
};

static const double *const bound_tables[] = { bounds_0, bounds_1, bounds_2,
					      bounds_3, bounds_4, bounds_5 };
static const uint32_t bound_counts[] = {
	sizeof(bounds_0) / sizeof(double), sizeof(bounds_1) / sizeof(double),
	sizeof(bounds_2) / sizeof(double), sizeof(bounds_3) / sizeof(double),
	sizeof(bounds_4) / sizeof(double), sizeof(bounds_5) / sizeof(double)
};

static uint32_t token_bounds(uint32_t context,
			     double output[METRICS_BUCKET_COUNT])
{
	uint32_t count = 0;
	static const uint32_t mantissas[] = { 1, 2, 5 };
	for (uint64_t scale = 1;
	     scale <= context && count < METRICS_BUCKET_COUNT;
	     scale *= METRICS_TOKEN_SCALE) {
		for (uint32_t index = 0;
		     index < sizeof(mantissas) / sizeof(mantissas[0]);
		     index++) {
			const uint64_t value = scale * mantissas[index];
			if (value > context)
				return count;
			output[count++] = (double)value;
		}
	}
	return count;
}

void server_metrics_prepare(struct http_server *server)
{
	if (server->metrics.token_context !=
	    server->description.context_token_limit) {
		server->metrics.token_bound_count =
			token_bounds(server->description.context_token_limit,
				     server->metrics.token_bounds);
		server->metrics.token_context =
			server->description.context_token_limit;
	}
}

static void observed(struct http_server *server, uint32_t kind, double value)
{
	struct metrics_histogram *const histogram =
		&server->metrics.latencies[kind];
	if (!isfinite(value) || value < 0 || !isfinite(histogram->sum + value))
		return;
	const uint32_t family = definitions[kind].family;
	if (family == METRICS_TOKEN_FAMILY)
		server_metrics_prepare(server);
	const double *const bounds = family == METRICS_TOKEN_FAMILY ?
					     server->metrics.token_bounds :
					     bound_tables[family];
	const uint32_t count = family == METRICS_TOKEN_FAMILY ?
				       server->metrics.token_bound_count :
				       bound_counts[family];
	uint32_t bucket = 0;
	while (bucket < count && value > bounds[bucket])
		bucket++;
	histogram->buckets[bucket]++;
	histogram->sum += value;
}

void server_metrics_first(struct http_server *server,
			  const struct live_request_record *request, double now)
{
	if (server->metrics.enabled)
		observed(server, METRICS_TTFT, now - request->arrival_seconds);
}

void server_metrics_token(struct http_server *server,
			  struct active_command *active, double now)
{
	if (!isfinite(now) || now <= 0)
		return;
	ASSERT(active->connection == NO_CONNECTION ||
	       server_tables_present(server));
	const struct live_request_record *const request =
		active->connection != NO_CONNECTION ?
			&server->connections[active->connection].live :
			&active->disconnected_live;
	if (active->generated_token_count == 0) {
		active->metrics_first_seconds = now;
		server_metrics_first(server, request, now);
	} else if (active->metrics_last_seconds > 0) {
		observed(server, METRICS_ITL,
			 now - active->metrics_last_seconds);
	}
	active->metrics_last_seconds = now;
}

void server_metrics_admitted(struct http_server *server,
			     const struct connection *connection)
{
	if (!server->metrics.enabled)
		return;
	server->metrics.admitted++;
	(void)connection;
}

void server_metrics_finished(struct http_server *server,
			     const struct live_request_record *request)
{
	if (!server->metrics.enabled)
		return;
	for (uint32_t reason = 0; reason < METRICS_FINISH_COUNT; reason++) {
		if (strcmp(request->finish_reason, finish_reasons[reason]) ==
		    0) {
			server->metrics.finished[reason]++;
			break;
		}
	}
	/* vLLM's output processor removes client-aborted requests before terminal stats.
     * Keep the native terminal count; token/TTFT observations already made remain valid. */
	if (strcmp(request->finish_reason, "cancelled") == 0)
		return;
	observed(server, METRICS_E2E,
		 request->finished_seconds - request->arrival_seconds);
	const struct engine_slot_stats *const engine = &request->engine;
	const bool has_engine = request->slot != UINT32_MAX &&
				engine->request_tag == request->request_tag &&
				engine->submitted_ns != 0;
	double queue = request->queue_wait_seconds;
	if (has_engine && engine->started_ns >= engine->submitted_ns &&
	    engine->started_ns != 0)
		queue += (double)(engine->started_ns - engine->submitted_ns) /
			 METRICS_NANOSECONDS_PER_SECOND;
	else if (has_engine && engine->finished_ns >= engine->submitted_ns)
		queue += (double)(engine->finished_ns - engine->submitted_ns) /
			 METRICS_NANOSECONDS_PER_SECOND;
	observed(server, METRICS_QUEUE, queue);
	double prefill = 0, decode = 0;
	if (has_engine && engine->started_ns != 0) {
		prefill = (double)engine->prefill_ns /
			  METRICS_NANOSECONDS_PER_SECOND;
		decode = engine->last_token_ns >= engine->first_token_ns &&
					 engine->first_token_ns != 0 ?
				 (double)(engine->last_token_ns -
					  engine->first_token_ns) /
					 METRICS_NANOSECONDS_PER_SECOND :
				 0;
	}
	observed(server, METRICS_PREFILL, prefill);
	observed(server, METRICS_DECODE, decode);
	observed(server, METRICS_INFERENCE, prefill + decode);
	double output_decode = decode;
	if (request->slot < server->description.session_slot_count) {
		ASSERT(server_tables_present(server));
		const struct active_command *const active =
			&server->active[request->slot];
		if (active->request_tag == request->request_tag &&
		    active->metrics_first_seconds > 0 &&
		    active->metrics_last_seconds >=
			    active->metrics_first_seconds)
			output_decode = active->metrics_last_seconds -
					active->metrics_first_seconds;
	}
	observed(server, METRICS_TPOT,
		 engine->generated_tokens > 1 ?
			 output_decode /
				 (double)(engine->generated_tokens - 1u) :
			 0);
	observed(server, METRICS_PROMPT_TOKENS,
		 request->rendered_prompt_tokens);
	observed(server, METRICS_GENERATION_TOKENS,
		 has_engine ? engine->generated_tokens : 0);
	observed(server, METRICS_COMPUTED_TOKENS,
		 request->rendered_prompt_tokens >= request->reused_tokens ?
			 request->rendered_prompt_tokens -
				 request->reused_tokens :
			 0);
	observed(server, METRICS_MAX_TOKENS, request->options.max_tokens);
	observed(server, METRICS_PREEMPTIONS, 0);
	observed(server, METRICS_N, 1);
	observed(server, METRICS_MAX_GENERATION,
		 has_engine ? engine->generated_tokens : 0);
}

void server_metrics_calls(struct http_server *server,
			  const struct chat_answer *answer, bool is_limited)
{
	uint32_t repairs[CHAT_CALLS_REPAIR_KINDS] = { 0 };
	if (server->metrics.enabled)
		chat_answer_repairs(answer, is_limited, repairs);
	for (uint32_t kind = 0; kind < CHAT_CALLS_REPAIR_KINDS; kind++)
		server->metrics.calls_repairs[kind] += repairs[kind];
}

void server_metrics_refused(struct http_server *server, uint32_t route,
			    uint32_t status)
{
	if (server->metrics.enabled &&
	    (route == CHAT_ROUTE_COMPLETIONS || route == CHAT_ROUTE_RESPONSES ||
	     route == CHAT_ROUTE_MESSAGES)) {
		if (status < METRICS_CLIENT_STATUS_END)
			server->metrics.refused_client++;
		else
			server->metrics.refused_server++;
	}
}

/* Formatting is scrape-only. A fixed maximum line also bounds escaped startup labels. */
static void append(struct chat_buffer *output, const char *format, ...)
	__attribute__((format(printf, METRICS_FORMAT_PARAMETER,
			      METRICS_FORMAT_VALUES)));
static void append(struct chat_buffer *output, const char *format, ...)
{
	char line[METRICS_LINE_BYTES];
	va_list arguments;
	va_start(arguments, format);
	const int count = vsnprintf(line, sizeof(line), format, arguments);
	va_end(arguments);
	if (count < 0 || (size_t)count >= sizeof(line)) {
		output->is_short = true;
	} else {
		if (output->capacity != 0 &&
		    (size_t)count > output->capacity - output->count) {
			output->is_short = true;
		} else {
			chat_buffer_write(output, line, (size_t)count);
		}
	}
}

static bool escaped(const char *input, size_t limit,
		    char output[METRICS_ESCAPE_BYTES])
{
	if (input == NULL) {
		input = "";
	}
	const size_t count = strnlen(input, limit);
	if (count == limit ||
	    count > (METRICS_ESCAPE_BYTES - 1u) / METRICS_ESCAPE_MULTIPLIER) {
		return false;
	}
	for (size_t position = 0; position < count;) {
		bool valid = false;
		const uint32_t sequence = tokenizer_utf8_sequence_bytes(
			input, position, count, &valid);
		if (!valid || sequence == 0 || input[position] == '\r') {
			return false;
		}
		position += sequence;
	}
	size_t written = 0;
	for (size_t index = 0; index < count; index++) {
		const char byte = input[index];
		if (byte == '\\' || byte == '"' || byte == '\n') {
			output[written++] = '\\';
		}
		output[written++] = (char)(byte == '\n' ? 'n' : byte);
	}
	output[written] = '\0';
	return true;
}

bool server_metrics_model_valid(const char *model)
{
	char encoded[METRICS_ESCAPE_BYTES];
	return model != NULL && model[0] != '\0' &&
	       escaped(model, PACK_MODEL_NAME_CAPACITY, encoded);
}

static void scalar(const struct http_server *server, struct chat_buffer *output,
		   const char *name, const char *type, const char *help,
		   const char *labels, double value)
{
	if (!isfinite(value)) {
		output->is_short = true;
		return;
	}
	append(output, "# HELP %s %s\n# TYPE %s %s\n%s{%s} %.17g\n", name, help,
	       name, type, name, labels, value);
	if (strcmp(type, "counter") == 0 &&
	    strncmp(name, "vllm:", sizeof("vllm:") - 1u) == 0) {
		const size_t length = strlen(name),
			     suffix = sizeof("_total") - 1u;
		if (length <= suffix || length - suffix > METRICS_NAME_BYTES) {
			output->is_short = true;
			return;
		}
		append(output,
		       "# HELP %.*s_created %s\n# TYPE %.*s_created gauge\n%.*s_created{%s} %.17g\n",
		       (int)(length - suffix), name, help,
		       (int)(length - suffix), name, (int)(length - suffix),
		       name, labels, (double)server->started_seconds);
	}
}

static bool bound_text(double value, char text[METRICS_BOUND_TEXT_BYTES])
{
	const int count = snprintf(text, METRICS_BOUND_TEXT_BYTES,
				   value < METRICS_SCIENTIFIC_BOUND &&
						   value == trunc(value) ?
					   "%.1f" :
					   "%g",
				   value);
	return count > 0 && (uint32_t)count < METRICS_BOUND_TEXT_BYTES;
}

static void histogram(const struct http_server *server,
		      struct chat_buffer *output, const char *name,
		      const char *help, const char *labels,
		      const double *bounds, uint32_t count,
		      const uint64_t *buckets, double sum)
{
	append(output, "# HELP %s %s\n# TYPE %s histogram\n", name, help, name);
	uint64_t cumulative = 0;
	for (uint32_t bucket = 0; bucket <= count; bucket++) {
		cumulative += buckets[bucket];
		if (bucket == count)
			append(output, "%s_bucket{%s%sle=\"+Inf\"} %.17g\n",
			       name, labels, labels[0] != '\0' ? "," : "",
			       (double)cumulative);
		else {
			char text[METRICS_BOUND_TEXT_BYTES];
			if (!bound_text(bounds[bucket], text)) {
				output->is_short = true;
				return;
			}
			append(output, "%s_bucket{%s%sle=\"%s\"} %.17g\n", name,
			       labels, labels[0] != '\0' ? "," : "", text,
			       (double)cumulative);
		}
	}
	if (!isfinite(sum)) {
		output->is_short = true;
		return;
	}
	append(output, "%s_sum{%s} %.17g\n%s_count{%s} %.17g\n", name, labels,
	       sum, name, labels, (double)cumulative);
	append(output,
	       "# HELP %s_created %s\n# TYPE %s_created gauge\n%s_created{%s} %.17g\n",
	       name, help, name, name, labels, (double)server->started_seconds);
}

/* Health is always excluded before lookup. Reuse its reserved cell for opt-in Messages, without growing records. */
#define METRICS_HTTP_MESSAGES_HANDLER 0u
static const char *const http_handlers[METRICS_HTTP_HANDLERS] = {
	"/v1/messages",
	"/v1/models",
	"/v1/chat/completions",
	"/v1/anemone/stats",
	"/v1/responses",
	"/metrics",
	"none"
};
static const char *const http_methods[METRICS_HTTP_METHODS] = {
	"GET", "POST", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS", "OTHER"
};
static const double http_low_bounds[METRICS_HTTP_LOW_BUCKETS] = { 0.1, 0.5, 1 };
static const double http_high_bounds[] = { 0.01, 0.025, 0.05, 0.075, 0.1, 0.25,
					   0.5,	 0.75,	1,    1.5,   2,	  2.5,
					   3,	 3.5,	4,    4.5,   5,	  7.5,
					   10,	 30,	60 };
#define METRICS_HTTP_SUCCESS 200u
#define METRICS_HTTP_SERVER_ERROR 500u
#define METRICS_HTTP_STATUS_SCALE 100u
#define METRICS_HTTP_STATUS_OFFSET 9u
#define METRICS_HTTP_STATUS_DIGITS 3u
#define METRICS_DECIMAL_BASE 10u

void server_metrics_http_head(struct connection *connection,
			      bool messages_enabled)
{
	connection->metrics_method = METRICS_HTTP_METHODS - 1u;
	connection->metrics_handler = METRICS_HTTP_HANDLERS - 1u;
	connection->metrics_http_excluded = true;
	/* Invalid field framing and pre-route refusals never reach the upstream application. */
	if (connection->head.metrics_protocol_error ||
	    (connection->head.refusal != 0 &&
	     connection->head.route == CHAT_ROUTE_HEALTH))
		return;
	const char *const line_end =
		connection->input.count != 0 ?
			memmem(connection->input.bytes, connection->input.count,
			       "\r\n", sizeof("\r\n") - 1u) :
			NULL;
	if (line_end == NULL)
		return;
	const char *const end =
		memchr(connection->input.bytes, ' ',
		       (size_t)(line_end - connection->input.bytes));
	if (end == NULL)
		return;
	const size_t bytes = (size_t)(end - connection->input.bytes);
	for (uint32_t index = 0; index + 1u < METRICS_HTTP_METHODS; index++) {
		if (strlen(http_methods[index]) == bytes &&
		    memcmp(connection->input.bytes, http_methods[index],
			   bytes) == 0) {
			connection->metrics_method = index;
			break;
		}
	}
	const char *const path = end + 1;
	const char *const path_end =
		memchr(path, ' ', (size_t)(line_end - path));
	if (path_end == NULL)
		return;
	connection->metrics_http_excluded = false;
	const char *const query = memchr(path, '?', (size_t)(path_end - path));
	const size_t path_bytes =
		(size_t)((query != NULL ? query : path_end) - path);
	/* Upstream exclusions are regex searches of the URL path, before query parameters. */
	static const char *const excluded[] = { "/metrics", "/health",
						"/load",    "/ping",
						"/version", "/server_info" };
	for (uint32_t index = 0; index < sizeof(excluded) / sizeof(excluded[0]);
	     index++)
		connection->metrics_http_excluded |=
			memmem(path, path_bytes, excluded[index],
			       strlen(excluded[index])) != NULL;
	for (uint32_t handler = 0; handler + 1u < METRICS_HTTP_HANDLERS;
	     handler++) {
		const bool messages = handler == METRICS_HTTP_MESSAGES_HANDLER;
		const bool post = handler == CHAT_ROUTE_COMPLETIONS ||
				  handler == CHAT_ROUTE_RESPONSES || messages;
		if ((!messages || messages_enabled) &&
		    connection->metrics_method == (post ? 1u : 0u) &&
		    strlen(http_handlers[handler]) == path_bytes &&
		    memcmp(path, http_handlers[handler], path_bytes) == 0) {
			connection->metrics_handler = handler;
			break;
		}
	}
}

void server_metrics_http_output(struct connection *connection)
{
	const struct chat_buffer *const output = &connection->answer.output;
	if (connection->metrics_status != 0 || output->sent > output->count ||
	    output->count - output->sent <
		    METRICS_HTTP_STATUS_OFFSET + METRICS_HTTP_STATUS_DIGITS)
		return;
	/* Whole JSON replies reserve room before their head; sent marks the actual wire prefix. */
	const char *const first = output->bytes + output->sent;
	const size_t available = output->count - output->sent;
	if (memcmp(first, "HTTP/1.", sizeof("HTTP/1.") - 1u) != 0)
		return;
	uint32_t status = 0;
	for (uint32_t index = 0; index < METRICS_HTTP_STATUS_DIGITS; index++) {
		const unsigned char byte = (unsigned char)
			first[METRICS_HTTP_STATUS_OFFSET + index];
		if (byte < '0' || byte > '9')
			return;
		status = status * METRICS_DECIMAL_BASE + (uint32_t)(byte - '0');
	}
	if (status < METRICS_HTTP_SUCCESS)
		return; /* 100 Continue is not the response. */
	connection->metrics_status = status;
	static const char field[] = "\r\nContent-Length: ";
	const char *const end =
		memmem(first, available, "\r\n\r\n", sizeof("\r\n\r\n") - 1u);
	const size_t head =
		end != NULL ? (size_t)(end - first) + sizeof("\r\n") - 1u : 0;
	const char *length =
		head != 0 ? memmem(first, head, field, sizeof(field) - 1u) :
			    NULL;
	if (length == NULL)
		return;
	length += sizeof(field) - 1u;
	uint64_t value = 0;
	while (length < first + head && *length >= '0' && *length <= '9') {
		const uint64_t digit = (uint64_t)(unsigned char)(*length - '0');
		if (value > (UINT64_MAX - digit) / METRICS_DECIMAL_BASE)
			return;
		value = value * METRICS_DECIMAL_BASE + digit;
		length++;
	}
	if (length < first + head && *length == '\r')
		connection->metrics_response_bytes = value;
}

void server_metrics_http_done(struct http_server *server,
			      struct connection *connection, double now)
{
	if (!server->metrics.enabled || connection->metrics_http_done ||
	    !connection->is_head_read)
		return;
	connection->metrics_http_done = true;
	const uint32_t handler = connection->metrics_handler <
						 METRICS_HTTP_HANDLERS ?
					 connection->metrics_handler :
					 METRICS_HTTP_HANDLERS - 1u;
	if (connection->metrics_http_excluded)
		return;
	const uint32_t method = connection->metrics_method <
						METRICS_HTTP_METHODS ?
					connection->metrics_method :
					METRICS_HTTP_METHODS - 1u;
	const uint32_t status = connection->metrics_status != 0 ?
					connection->metrics_status :
					METRICS_HTTP_SERVER_ERROR;
	const uint32_t status_class = status / METRICS_HTTP_STATUS_SCALE;
	const double elapsed = now - connection->live.arrival_seconds;
	if (status_class < METRICS_HTTP_SUCCESS / METRICS_HTTP_STATUS_SCALE ||
	    status_class >= METRICS_HTTP_SUCCESS / METRICS_HTTP_STATUS_SCALE +
				    METRICS_HTTP_STATUS_CLASSES ||
	    !isfinite(elapsed) || elapsed < 0)
		return;
	struct metrics_http_cell *const cell =
		&server->metrics.http[handler][method];
	const uint32_t status_index =
		status_class - METRICS_HTTP_SUCCESS / METRICS_HTTP_STATUS_SCALE;
	if (cell->requests[status_index] == 0 || cell->duration_created == 0 ||
	    server->metrics.http_count[handler] == 0) {
		struct timespec timestamp;
		if (clock_gettime(CLOCK_REALTIME, &timestamp) != 0 ||
		    timestamp.tv_sec < 0)
			return;
		const double epoch = (double)timestamp.tv_sec +
				     (double)timestamp.tv_nsec /
					     METRICS_NANOSECONDS_PER_SECOND;
		if (cell->requests[status_index] == 0)
			cell->requests_created[status_index] = epoch;
		if (cell->duration_created == 0)
			cell->duration_created = epoch;
		if (server->metrics.http_count[handler] == 0)
			server->metrics.http_created[handler] = epoch;
	}
	cell->requests[status_index]++;
	uint32_t bucket = 0;
	while (bucket < METRICS_HTTP_LOW_BUCKETS &&
	       elapsed > http_low_bounds[bucket])
		bucket++;
	cell->buckets[bucket]++;
	cell->seconds += elapsed;
	const uint32_t count =
		sizeof(http_high_bounds) / sizeof(http_high_bounds[0]);
	bucket = 0;
	while (bucket < count && elapsed > http_high_bounds[bucket])
		bucket++;
	server->metrics.http_high.buckets[bucket]++;
	server->metrics.http_high.sum += elapsed;
	server->metrics.http_count[handler]++;
	/* Declared Content-Length can be UINT64_MAX even when the request is refused. */
	const uint64_t request_bytes = connection->head.body_byte_count;
	const uint64_t response_bytes = connection->metrics_response_bytes;
	server->metrics.request_bytes[handler] =
		request_bytes > UINT64_MAX -
					server->metrics.request_bytes[handler] ?
			UINT64_MAX :
			server->metrics.request_bytes[handler] + request_bytes;
	server->metrics.response_bytes[handler] =
		response_bytes > UINT64_MAX - server->metrics
						      .response_bytes[handler] ?
			UINT64_MAX :
			server->metrics.response_bytes[handler] +
				response_bytes;
}

static void server_metrics_http_render(const struct http_server *server,
				       struct chat_buffer *output)
{
	append(output,
	       "# HELP http_requests_total Total number of requests by method, status and handler.\n# TYPE "
	       "http_requests_total counter\n");
	for (uint32_t handler = 0; handler < METRICS_HTTP_HANDLERS; handler++) {
		for (uint32_t method = 0; method < METRICS_HTTP_METHODS;
		     method++) {
			const struct metrics_http_cell *const cell =
				&server->metrics.http[handler][method];
			for (uint32_t status = 0;
			     status < METRICS_HTTP_STATUS_CLASSES; status++) {
				if (cell->requests[status] != 0)
					append(output,
					       "http_requests_total{handler=\"%s\",method=\"%s\",status=\"%uxx\"}"
					       " %.17g\n",
					       http_handlers[handler],
					       http_methods[method],
					       status +
						       METRICS_HTTP_SUCCESS /
							       METRICS_HTTP_STATUS_SCALE,
					       (double)cell->requests[status]);
			}
		}
	}
	static const char *const size_names[] = { "http_request_size_bytes",
						  "http_response_size_bytes" };
	static const char *const size_help[] = {
		"Content length of incoming requests by handler. Only value of header is respected. Otherwise "
		"ignored. No percentile calculated. ",
		"Content length of outgoing responses by handler. Only value of header is respected. Otherwise "
		"ignored. No percentile calculated. "
	};
	for (uint32_t kind = 0;
	     kind < sizeof(size_names) / sizeof(size_names[0]); kind++) {
		append(output, "# HELP %s %s\n# TYPE %s summary\n",
		       size_names[kind], size_help[kind], size_names[kind]);
		for (uint32_t handler = 0; handler < METRICS_HTTP_HANDLERS;
		     handler++) {
			if (server->metrics.http_count[handler] == 0)
				continue;
			const double sum =
				(double)(kind == 0 ?
						 server->metrics
							 .request_bytes[handler] :
						 server->metrics.response_bytes
							 [handler]);
			append(output,
			       "%s_sum{handler=\"%s\"} %.17g\n%s_count{handler=\"%s\"} %.17g\n",
			       size_names[kind], http_handlers[handler], sum,
			       size_names[kind], http_handlers[handler],
			       (double)server->metrics.http_count[handler]);
		}
	}
	histogram(
		server, output, "http_request_duration_highr_seconds",
		"Latency with many buckets but no API specific labels. Made for more accurate percentile "
		"calculations. ",
		"", http_high_bounds,
		sizeof(http_high_bounds) / sizeof(http_high_bounds[0]),
		server->metrics.http_high.buckets,
		server->metrics.http_high.sum);
	append(output,
	       "# HELP http_request_duration_seconds Latency with only few buckets by handler. Made to be only "
	       "used if aggregation by handler is important. \n# TYPE http_request_duration_seconds histogram\n");
	for (uint32_t handler = 0; handler < METRICS_HTTP_HANDLERS; handler++) {
		for (uint32_t method = 0; method < METRICS_HTTP_METHODS;
		     method++) {
			const struct metrics_http_cell *const cell =
				&server->metrics.http[handler][method];
			uint64_t cumulative = 0;
			for (uint32_t bucket = 0;
			     bucket <= METRICS_HTTP_LOW_BUCKETS; bucket++)
				cumulative += cell->buckets[bucket];
			if (cumulative == 0)
				continue;
			append(output,
			       "http_request_duration_seconds_sum{handler=\"%s\",method=\"%s\"} "
			       "%.17g\nhttp_request_duration_seconds_count{handler=\"%s\",method=\"%s\"} %.17g\n",
			       http_handlers[handler], http_methods[method],
			       cell->seconds, http_handlers[handler],
			       http_methods[method], (double)cumulative);
			cumulative = 0;
			for (uint32_t bucket = 0;
			     bucket <= METRICS_HTTP_LOW_BUCKETS; bucket++) {
				cumulative += cell->buckets[bucket];
				if (bucket == METRICS_HTTP_LOW_BUCKETS)
					append(output,
					       "http_request_duration_seconds_bucket{handler=\"%s\","
					       "method=\"%s\",le=\"+Inf\"} %.17g\n",
					       http_handlers[handler],
					       http_methods[method],
					       (double)cumulative);
				else {
					char text[METRICS_BOUND_TEXT_BYTES];
					if (!bound_text(http_low_bounds[bucket],
							text)) {
						output->is_short = true;
						return;
					}
					append(output,
					       "http_request_duration_seconds_bucket{handler=\"%s\","
					       "method=\"%s\",le=\"%s\"} %.17g\n",
					       http_handlers[handler],
					       http_methods[method], text,
					       (double)cumulative);
				}
			}
		}
	}
	bool has_children = false;
	for (uint32_t handler = 0; handler < METRICS_HTTP_HANDLERS; handler++)
		has_children |= server->metrics.http_count[handler] != 0;
	if (!has_children)
		return;
	append(output,
	       "# HELP http_requests_created Total number of requests by method, status and handler.\n# TYPE "
	       "http_requests_created gauge\n");
	for (uint32_t handler = 0; handler < METRICS_HTTP_HANDLERS; handler++) {
		for (uint32_t method = 0; method < METRICS_HTTP_METHODS;
		     method++) {
			const struct metrics_http_cell *const cell =
				&server->metrics.http[handler][method];
			for (uint32_t status = 0;
			     status < METRICS_HTTP_STATUS_CLASSES; status++) {
				if (cell->requests[status] != 0)
					append(output,
					       "http_requests_created{handler=\"%s\",method=\"%s\",status=\"%uxx\"}"
					       " %.17g\n",
					       http_handlers[handler],
					       http_methods[method],
					       status +
						       METRICS_HTTP_SUCCESS /
							       METRICS_HTTP_STATUS_SCALE,
					       cell->requests_created[status]);
			}
		}
	}
	for (uint32_t kind = 0;
	     kind < sizeof(size_names) / sizeof(size_names[0]); kind++) {
		append(output,
		       "# HELP %s_created %s\n# TYPE %s_created gauge\n",
		       size_names[kind], size_help[kind], size_names[kind]);
		for (uint32_t handler = 0; handler < METRICS_HTTP_HANDLERS;
		     handler++)
			if (server->metrics.http_count[handler] != 0)
				append(output,
				       "%s_created{handler=\"%s\"} %.17g\n",
				       size_names[kind], http_handlers[handler],
				       server->metrics.http_created[handler]);
	}
	append(output,
	       "# HELP http_request_duration_seconds_created Latency with only few buckets by handler. Made to "
	       "be only used if aggregation by handler is important. \n# TYPE "
	       "http_request_duration_seconds_created gauge\n");
	for (uint32_t handler = 0; handler < METRICS_HTTP_HANDLERS; handler++) {
		for (uint32_t method = 0; method < METRICS_HTTP_METHODS;
		     method++) {
			const struct metrics_http_cell *const cell =
				&server->metrics.http[handler][method];
			uint64_t count = 0;
			for (uint32_t bucket = 0;
			     bucket <= METRICS_HTTP_LOW_BUCKETS; bucket++)
				count += cell->buckets[bucket];
			if (count != 0)
				append(output,
				       "http_request_duration_seconds_created{handler=\"%s\",method=\"%s\"} %.17g\n",
				       http_handlers[handler],
				       http_methods[method],
				       cell->duration_created);
		}
	}
}

result_status server_metrics_render(const struct http_server *server,
				    const struct engine_stats *engine,
				    struct chat_buffer *output)
{
	if (server == NULL || engine == NULL || output == NULL ||
	    engine->slots_total > engine->slot_room ||
	    (engine->slots_total != 0 &&
	     (engine->slots == NULL || engine->sessions == NULL)) ||
	    server->description.session_slot_count > engine->slots_total ||
	    server->waiting_count > SERVER_MOST_WAITING ||
	    engine->started_ns > engine->now_ns)
		return STATUS_INVALID_ARGUMENT;
	char model[METRICS_ESCAPE_BYTES], build[METRICS_ESCAPE_BYTES];
	if (!escaped(server->model_alias != NULL ?
			     server->model_alias :
			     server->description.model_name,
		     LIVE_STATS_MODEL_BYTES, model) ||
	    !escaped(server->live.build_id, LIVE_STATS_BUILD_BYTES, build))
		return STATUS_INVALID_ARGUMENT;
	char labels[METRICS_LINE_BYTES];
	const int label_count = snprintf(labels, sizeof(labels),
					 "model_name=\"%s\",engine=\"0\"",
					 model);
	if (label_count < 0 || (size_t)label_count >= sizeof(labels))
		return STATUS_INVALID_ARGUMENT;
	uint32_t running = 0, waiting = server->waiting_count;
	for (uint32_t slot = 0; slot < server->description.session_slot_count;
	     slot++) {
		ASSERT(server_tables_present(server));
		if (server->active[slot].token_ids != NULL &&
		    server->active[slot].operation == 0) {
			const struct engine_slot_stats *const current =
				&engine->slots[slot];
			const bool queued =
				current->request_tag ==
					server->active[slot].request_tag &&
				current->phase == STATS_QUEUED;
			waiting += queued ? 1u : 0u;
			running += queued ? 0u : 1u;
		}
	}
	scalar(server, output, "vllm:num_requests_running", "gauge",
	       "Number of requests in model execution batches.", labels,
	       running);
	scalar(server, output, "vllm:num_requests_waiting", "gauge",
	       "Number of requests waiting to be processed.", labels, waiting);
	scalar(server, output, "vllm:prompt_tokens_total", "counter",
	       "Number of prefill tokens processed.", labels,
	       (double)engine->prompt_tokens);
	scalar(server, output, "vllm:generation_tokens_total", "counter",
	       "Number of generation tokens processed.", labels,
	       (double)engine->generated_tokens);
	scalar(server, output, "vllm:prompt_tokens_cached_total", "counter",
	       "Number of cached prompt tokens (local + external).", labels,
	       (double)engine->cached_tokens);
	scalar(server, output, "vllm:prefix_cache_queries_total", "counter",
	       "Prefix cache queries, in terms of number of queried tokens.",
	       labels,
	       server->disable_prefix_reuse ? 0 :
					      (double)engine->prompt_tokens);
	scalar(server, output, "vllm:prefix_cache_hits_total", "counter",
	       "Prefix cache hits, in terms of number of cached tokens.",
	       labels,
	       server->disable_prefix_reuse ? 0 :
					      (double)engine->cached_tokens);
	scalar(server, output, "vllm:num_preemptions_total", "counter",
	       "Cumulative number of preemption from the engine.", labels, 0);

	append(output,
	       "# HELP vllm:request_success_total Count of successfully processed requests.\n# TYPE "
	       "vllm:request_success_total counter\n");
	static const char *const reasons[] = { "stop", "length", "abort",
					       "error", "repetition" };
	const uint64_t values[] = {
		server->metrics.finished[METRICS_FINISH_STOP] +
			server->metrics.finished[METRICS_FINISH_TOOLS],
		server->metrics.finished[METRICS_FINISH_LENGTH], 0,
		server->metrics.finished[METRICS_FINISH_ERROR], 0
	};
	for (uint32_t index = 0; index < sizeof(reasons) / sizeof(reasons[0]);
	     index++)
		append(output,
		       "vllm:request_success_total{%s,finished_reason=\"%s\"} %.17g\n",
		       labels, reasons[index], (double)values[index]);
	append(output,
	       "# HELP vllm:request_success_created Count of successfully processed requests.\n# TYPE "
	       "vllm:request_success_created gauge\n");
	for (uint32_t index = 0; index < sizeof(reasons) / sizeof(reasons[0]);
	     index++)
		append(output,
		       "vllm:request_success_created{%s,finished_reason=\"%s\"} %.17g\n",
		       labels, reasons[index], (double)server->started_seconds);

	double token_storage[METRICS_BUCKET_COUNT];
	const bool prepared = server->metrics.token_context ==
			      server->description.context_token_limit;
	const uint32_t token_count =
		prepared ? server->metrics.token_bound_count :
			   token_bounds(server->description.context_token_limit,
					token_storage);
	for (uint32_t kind = 0; kind < METRICS_LATENCY_COUNT; kind++) {
		const uint32_t family = definitions[kind].family;
		const double *const bounds =
			family == METRICS_TOKEN_FAMILY ?
				(prepared ? server->metrics.token_bounds :
					    token_storage) :
				bound_tables[family];
		const uint32_t count = family == METRICS_TOKEN_FAMILY ?
					       token_count :
					       bound_counts[family];
		histogram(server, output, definitions[kind].name,
			  definitions[kind].help, labels, bounds, count,
			  server->metrics.latencies[kind].buckets,
			  server->metrics.latencies[kind].sum);
	}
	histogram(server, output, "vllm:iteration_tokens_total",
		  "Histogram of number of tokens per engine_step.", labels,
		  bounds_3, STATS_ITERATION_BUCKETS, engine->iteration_tokens,
		  (double)engine->iteration_token_sum);
	append(output,
	       "# HELP vllm:num_requests_waiting_by_reason Number of waiting requests by reason. Reason labels: "
	       "'capacity' = waiting for scheduling capacity; 'deferred' = deferred by transient constraints "
	       "(LoRA budget, KV transfer, blocked status). Sum of all reasons equals "
	       "vllm:num_requests_waiting.\n# TYPE vllm:num_requests_waiting_by_reason gauge\n");
	uint32_t deferred = 0;
	for (uint32_t index = 0; index < server->waiting_count; index++) {
		if (server->waiting[index] >= server->connection_limit)
			return STATUS_INVALID_ARGUMENT;
		const struct connection *const connection =
			&server->connections[server->waiting[index]];
		deferred += connection->waiting_snapshot_slot <
					    WAITING_SNAPSHOT_READY ?
				    1u :
				    0u;
	}
	append(output,
	       "vllm:num_requests_waiting_by_reason{%s,reason=\"capacity\"} "
	       "%u\nvllm:num_requests_waiting_by_reason{%s,reason=\"deferred\"} %u\n",
	       labels, waiting >= deferred ? waiting - deferred : 0, labels,
	       deferred);
	append(output,
	       "# HELP vllm:admission_rejections_total Number of admission attempts rejected by a queue limit, "
	       "by reason.\n# TYPE vllm:admission_rejections_total "
	       "counter\nvllm:admission_rejections_total{model_name=\"%s\",reason=\"max_num_queued_reqs\"} "
	       "%.17g\nvllm:admission_rejections_total{model_name=\"%s\",reason=\"max_num_queued_tokens\"} 0\n",
	       model, (double)server->metrics.admission_rejections, model);
	append(output,
	       "# HELP vllm:admission_rejections_created Number of admission attempts rejected by a queue limit, "
	       "by reason.\n# TYPE vllm:admission_rejections_created "
	       "gauge\nvllm:admission_rejections_created{model_name=\"%s\",reason=\"max_num_queued_reqs\"} "
	       "%.17g\nvllm:admission_rejections_created{model_name=\"%s\",reason=\"max_num_queued_tokens\"} "
	       "%.17g\n",
	       model, (double)server->started_seconds, model,
	       (double)server->started_seconds);

	append(output,
	       "# HELP vllm:prompt_tokens_by_source_total Number of prompt tokens by source.\n# TYPE"
	       " vllm:prompt_tokens_by_source_total counter\n");
	static const char *const sources[] = { "local_compute",
					       "local_cache_hit",
					       "external_kv_transfer" };
	const uint64_t source_values[] = { engine->prefill_tokens,
					   engine->cached_tokens, 0 };
	for (uint32_t source = 0; source < sizeof(sources) / sizeof(sources[0]);
	     source++)
		append(output,
		       "vllm:prompt_tokens_by_source_total{%s,source=\"%s\"} %.17g\n",
		       labels, sources[source], (double)source_values[source]);
	append(output,
	       "# HELP vllm:prompt_tokens_by_source_created Number of prompt tokens by source.\n# TYPE"
	       " vllm:prompt_tokens_by_source_created gauge\n");
	for (uint32_t source = 0; source < sizeof(sources) / sizeof(sources[0]);
	     source++)
		append(output,
		       "vllm:prompt_tokens_by_source_created{%s,source=\"%s\"} %.17g\n",
		       labels, sources[source],
		       (double)server->started_seconds);

	uint64_t context_used = 0, active_used = 0, context_capacity = 0;
	for (uint32_t slot = 0; slot < engine->slots_total; slot++) {
		context_used += engine->sessions[slot].position;
		active_used += engine->sessions[slot].active ?
				       engine->sessions[slot].position :
				       0;
		context_capacity += engine->sessions[slot].context_capacity;
	}
	scalar(server, output, "vllm:kv_cache_usage_perc", "gauge",
	       "KV-cache usage. 1 means 100 percent usage.", labels,
	       context_capacity != 0 ?
		       (double)active_used / (double)context_capacity :
		       0);
	append(output,
	       "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n# TYPE "
	       "vllm:cache_config_info gauge\nvllm:cache_config_info{engine=\"0\",kv_cache_size_tokens=\"%" PRIu64
	       "\",kv_cache_max_concurrency=\"%u.0\",enable_prefix_caching=\"%s\"} 1\n",
	       context_capacity, server->description.session_slot_count,
	       server->disable_prefix_reuse ? "False" : "True");
	scalar(server, output, "anemone_requests_admitted_total", "counter",
	       "Normalized generation requests admitted to HTTP queue.", labels,
	       (double)server->metrics.admitted);
	append(output,
	       "# HELP anemone_requests_finished_total Terminal admitted generation requests.\n# TYPE "
	       "anemone_requests_finished_total counter\n");
	for (uint32_t reason = 0; reason < METRICS_FINISH_COUNT; reason++)
		append(output,
		       "anemone_requests_finished_total{%s,finished_reason=\"%s\"} %.17g\n",
		       labels, finish_reasons[reason],
		       (double)server->metrics.finished[reason]);
	append(output,
	       "# HELP anemone_request_refusals_total HTTP generation refusals; may also terminate admitted "
	       "requests.\n# TYPE anemone_request_refusals_total "
	       "counter\nanemone_request_refusals_total{%s,status_class=\"4xx\"} "
	       "%.17g\nanemone_request_refusals_total{%s,status_class=\"5xx\"} %.17g\n",
	       labels, (double)server->metrics.refused_client, labels,
	       (double)server->metrics.refused_server);
	append(output,
	       "# HELP anemone_dsml_repairs_total Tool-call DSML the tolerant reading repaired, by kind.\n# TYPE "
	       "anemone_dsml_repairs_total counter\n");
	for (uint32_t kind = 0; kind < CHAT_CALLS_REPAIR_KINDS; kind++)
		append(output,
		       "anemone_dsml_repairs_total{%s,kind=\"%s\"} %.17g\n",
		       labels, calls_repair_kinds[kind],
		       (double)server->metrics.calls_repairs[kind]);
	scalar(server, output, "anemone_cache_context_tokens", "gauge",
	       "Committed context positions retained across active and idle sessions; not physical KV bytes.",
	       labels, (double)context_used);
	scalar(server, output, "anemone_cache_capacity_tokens", "gauge",
	       "Total configured session context capacity in tokens.", labels,
	       (double)context_capacity);
	scalar(server, output, "anemone_prefill_tokens_total", "counter",
	       "New prompt tokens processed, excluding reuse.", labels,
	       (double)engine->prefill_tokens);
	scalar(server, output, "anemone_model_ready", "gauge",
	       "Model serving loop ready and engine has not failed or begun stopping.",
	       labels,
	       !server->engine_failed && !server->is_stopping ? 1.0 : 0.0);
	scalar(server, output, "anemone_process_uptime_seconds", "gauge",
	       "Seconds since inference engine startup.", labels,
	       (double)(engine->now_ns - engine->started_ns) /
		       METRICS_NANOSECONDS_PER_SECOND);
	append(output,
	       "# HELP anemone_build_info Executable ELF build identity.\n# TYPE anemone_build_info "
	       "gauge\nanemone_build_info{%s,build_id=\"%s\"} 1\n",
	       labels, build);
	if (engine->speculation_mode != STATS_SPECULATION_OFF) {
		uint64_t proposed = 0, accepted = 0, drafts = 0;
		for (uint32_t source = 0; source < DRAFT_SOURCES; source++) {
			drafts += engine->speculation.source[source].rounds;
			for (uint32_t position = 0;
			     position < DRAFT_STAT_POSITIONS; position++) {
				proposed += engine->speculation.source[source]
						    .proposed[position];
				accepted += engine->speculation.source[source]
						    .accepted[position];
			}
		}
		scalar(server, output, "vllm:spec_decode_num_drafts_total",
		       "counter", "Number of spec decoding drafts.", labels,
		       (double)drafts);
		scalar(server, output,
		       "vllm:spec_decode_num_draft_tokens_total", "counter",
		       "Number of draft tokens.", labels, (double)proposed);
		scalar(server, output,
		       "vllm:spec_decode_num_accepted_tokens_total", "counter",
		       "Number of accepted tokens.", labels, (double)accepted);
		append(output,
		       "# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft"
		       " position.\n# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter\n");
		for (uint32_t position = 0;
		     position < server->description.speculative_depth &&
		     position < DRAFT_STAT_POSITIONS;
		     position++) {
			uint64_t value = 0;
			for (uint32_t source = 0; source < DRAFT_SOURCES;
			     source++)
				value += engine->speculation.source[source]
						 .accepted[position];
			append(output,
			       "vllm:spec_decode_num_accepted_tokens_per_pos_total{%s,position=\"%u\"} %.17g\n",
			       labels, position, (double)value);
		}
		append(output,
		       "# HELP vllm:spec_decode_num_accepted_tokens_per_pos_created Accepted tokens per draft "
		       "position.\n# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_created gauge\n");
		for (uint32_t position = 0;
		     position < server->description.speculative_depth &&
		     position < DRAFT_STAT_POSITIONS;
		     position++)
			append(output,
			       "vllm:spec_decode_num_accepted_tokens_per_pos_created{%s,position=\"%u\"} %.17g\n",
			       labels, position,
			       (double)server->started_seconds);

		scalar(server, output,
		       "anemone_speculation_proposed_tokens_total", "counter",
		       "Proposed draft tokens by source statistics.", labels,
		       (double)proposed);
		scalar(server, output,
		       "anemone_speculation_accepted_tokens_total", "counter",
		       "Accepted matching non-end-token draft prefix tokens.",
		       labels, (double)accepted);
	}
	server_metrics_http_render(server, output);
	if (server->storage_replica_info != NULL &&
	    server->storage_replica_info_bytes != 0)
		chat_buffer_write(output, server->storage_replica_info,
				  server->storage_replica_info_bytes);
	return output->is_short ? STATUS_OUT_OF_MEMORY : STATUS_OK;
}

result_status server_metrics_text(struct http_server *server,
				  struct chat_buffer *output)
{
	if (server == NULL || output == NULL || !server->metrics.enabled)
		return STATUS_INVALID_ARGUMENT;
	const result_status status =
		engine_stats_counters(server->engine, &server->sampled);
	if (status != STATUS_OK)
		return status;
	result_status rendered =
		server_metrics_render(server, &server->sampled, output);
	if (rendered == STATUS_OK)
		rendered = server_metrics_process(output);
	return rendered;
}
