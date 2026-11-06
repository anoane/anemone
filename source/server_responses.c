/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Responses SSE is written from the existing decoded text/call boundary, never by parsing Chat wire bytes. */
#include "server_responses.h"

#include "anemone/hooks.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ESCAPE_FACTOR 6u
#define QUOTES 2u
#define NUMBER_BYTES 32u
#define HEADER_BYTES 256u
#define ITEM_ID_BYTES 128u
#define STREAM_HEAD                                                                         \
	"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n" \
	"Connection: close\r\n\r\n"

enum item_kind { ITEM_NONE, ITEM_MESSAGE, ITEM_REASONING };

struct server_responses_output {
	struct chat_buffer items;
	const struct chat_buffer *
		fields; /* connection-owned; never copied into request telemetry */
	char *model;
	uint64_t created, sequence, index;
	uint32_t kind;
	size_t text_begin;
	char item_id[ITEM_ID_BYTES];
};

static void text(struct chat_buffer *buffer, const char *value)
{
	chat_buffer_write(buffer, value, strlen(value));
}

static void number(struct chat_buffer *buffer, uint64_t value)
{
	char digits[NUMBER_BYTES];
	const int count = snprintf(digits, sizeof(digits), "%" PRIu64, value);
	chat_buffer_write(buffer, digits, count > 0 ? (size_t)count : 0);
}

static void real(struct chat_buffer *buffer, float value)
{
	if (!isfinite(value)) {
		text(buffer, "null");
		return;
	}
	char digits[NUMBER_BYTES];
	const int count =
		snprintf(digits, sizeof(digits), "%.9g", (double)value);
	chat_buffer_write(buffer, digits, count > 0 ? (size_t)count : 0);
}

static void string(struct chat_buffer *buffer, const char *bytes, size_t count)
{
	const size_t bound = count * ESCAPE_FACTOR + QUOTES;
	char *const room = count <= CHAT_MOST_OUTPUT_BYTES / ESCAPE_FACTOR ?
				   chat_buffer_room(buffer, bound) :
				   NULL;
	if (room == NULL) {
		buffer->is_short = true;
	} else {
		struct json_writer writer = { room, bound, 0 };
		json_write_string(&writer, (struct json_bytes){ bytes, count });
		buffer->count += writer.length;
	}
}

static void quoted(struct chat_buffer *buffer, const char *value)
{
	string(buffer, value, strlen(value));
}

static void event_begin(struct chat_answer *answer, const char *type)
{
	struct chat_buffer *const output = &answer->output;
	text(output, "event: ");
	text(output, type);
	text(output, "\ndata: {\"type\":");
	quoted(output, type);
	text(output, ",\"sequence_number\":");
	number(output, answer->responses->sequence++);
}

static void event_end(struct chat_answer *answer)
{
	text(&answer->output, "}\n\n");
}

static void event_index(struct chat_answer *answer)
{
	text(&answer->output, ",\"output_index\":");
	number(&answer->output, answer->responses->index);
}

static void event_item_id(struct chat_answer *answer)
{
	event_index(answer);
	text(&answer->output, ",\"item_id\":");
	quoted(&answer->output, answer->responses->item_id);
}

static void part_write(struct chat_buffer *target, bool reasoning,
		       const char *bytes, size_t count)
{
	text(target, reasoning ? "{\"type\":\"summary_text\",\"text\":" :
				 "{\"type\":\"output_text\",\"text\":");
	string(target, bytes, count);
	text(target, reasoning ? "}" : ",\"annotations\":[],\"logprobs\":[]}");
}

static void item_write(struct chat_answer *answer, struct chat_buffer *target,
		       bool done, const char *status)
{
	struct server_responses_output *const state = answer->responses;
	const bool reasoning = state->kind == ITEM_REASONING;
	const struct chat_buffer *const source =
		reasoning ? &answer->reasoning : &answer->content;
	text(target, "{\"id\":");
	quoted(target, state->item_id);
	text(target, ",\"type\":");
	quoted(target, reasoning ? "reasoning" : "message");
	text(target, ",\"status\":");
	quoted(target, status);
	text(target, reasoning ? ",\"encrypted_content\":null,\"summary\":[" :
				 ",\"role\":\"assistant\",\"content\":[");
	if (done) {
		part_write(target, reasoning,
			   source->bytes != NULL ?
				   source->bytes + state->text_begin :
				   "",
			   source->count - state->text_begin);
	}
	text(target, "]}");
}

static void append_item(struct chat_answer *answer, struct chat_buffer *item)
{
	struct server_responses_output *const state = answer->responses;
	text(&state->items, state->index == 0 ? "" : ",");
	chat_buffer_write(&state->items, item->bytes, item->count);
	answer->output.is_short = answer->output.is_short || item->is_short ||
				  state->items.is_short;
	state->index++;
}

static void item_close(struct chat_answer *answer, const char *status)
{
	struct server_responses_output *const state = answer->responses;
	if (state == NULL || state->kind == ITEM_NONE) {
		return;
	}
	const bool reasoning = state->kind == ITEM_REASONING;
	const struct chat_buffer *const source =
		reasoning ? &answer->reasoning : &answer->content;
	const char *const bytes =
		source->bytes != NULL ? source->bytes + state->text_begin : "";
	const size_t count = source->count - state->text_begin;
	if (answer->request->is_streamed) {
		event_begin(answer,
			    reasoning ? "response.reasoning_summary_text.done" :
					"response.output_text.done");
		event_item_id(answer);
		text(&answer->output, reasoning ?
					      ",\"summary_index\":0,\"text\":" :
					      ",\"content_index\":0,\"text\":");
		string(&answer->output, bytes, count);
		if (!reasoning) {
			text(&answer->output, ",\"logprobs\":[]");
		}
		event_end(answer);
		event_begin(answer,
			    reasoning ? "response.reasoning_summary_part.done" :
					"response.content_part.done");
		event_item_id(answer);
		text(&answer->output, reasoning ?
					      ",\"summary_index\":0,\"part\":" :
					      ",\"content_index\":0,\"part\":");
		part_write(&answer->output, reasoning, bytes, count);
		event_end(answer);
	}
	struct chat_buffer item = { 0 };
	item_write(answer, &item, true, status);
	if (answer->request->is_streamed) {
		event_begin(answer, "response.output_item.done");
		event_index(answer);
		text(&answer->output, ",\"item\":");
		chat_buffer_write(&answer->output, item.bytes, item.count);
		event_end(answer);
	}
	append_item(answer, &item);
	chat_buffer_free(&item);
	state->kind = ITEM_NONE;
}

static void item_open(struct chat_answer *answer, bool reasoning)
{
	struct server_responses_output *const state = answer->responses;
	const uint32_t kind = reasoning ? ITEM_REASONING : ITEM_MESSAGE;
	if (state->kind == kind) {
		return;
	}
	item_close(answer, "completed");
	state->kind = kind;
	state->text_begin = reasoning ? answer->reasoning.count :
					answer->content.count;
	(void)snprintf(state->item_id, sizeof(state->item_id), "%s_%s_%" PRIu64,
		       reasoning ? "rs" : "msg", answer->identifier,
		       state->index);
	if (answer->request->is_streamed) {
		event_begin(answer, "response.output_item.added");
		event_index(answer);
		text(&answer->output, ",\"item\":");
		item_write(answer, &answer->output, false, "in_progress");
		event_end(answer);
		event_begin(answer,
			    reasoning ?
				    "response.reasoning_summary_part.added" :
				    "response.content_part.added");
		event_item_id(answer);
		text(&answer->output, reasoning ?
					      ",\"summary_index\":0,\"part\":" :
					      ",\"content_index\":0,\"part\":");
		part_write(&answer->output, reasoning, "", 0);
		event_end(answer);
	}
}

static void response_write(struct chat_answer *answer,
			   struct chat_buffer *target, const char *status,
			   bool done)
{
	struct server_responses_output *const state = answer->responses;
	text(target, "{\"id\":");
	quoted(target, answer->identifier);
	text(target, ",\"object\":\"response\",\"created_at\":");
	number(target, state->created);
	text(target, ",\"status\":");
	quoted(target, status);
	text(target, ",\"error\":");
	text(target,
	     strcmp(status, "failed") == 0 ?
		     "{\"code\":\"server_error\",\"message\":\"" CHAT_ENGINE_FAILED
		     "\"}" :
		     "null");
	text(target, ",\"incomplete_details\":");
	text(target, strcmp(status, "incomplete") == 0 ?
			     "{\"reason\":\"max_output_tokens\"}" :
			     "null");
	text(target, ",\"model\":");
	quoted(target, state->model);
	text(target, ",\"output\":[");
	chat_buffer_write(target, state->items.bytes, state->items.count);
	text(target,
	     "],\"store\":false,\"previous_response_id\":null,\"parallel_tool_calls\":true");
	const struct chat_buffer *const fields = state->fields;
	if (fields != NULL && fields->count != 0) {
		chat_buffer_write(target, fields->bytes, fields->count);
		target->is_short = target->is_short || fields->is_short;
	} else {
		text(target,
		     ",\"instructions\":null,\"tools\":[],\"metadata\":{},\"tool_choice\":\"auto\"");
	}
	text(target, ",\"temperature\":");
	real(target, answer->request->temperature);
	text(target, ",\"top_p\":");
	real(target, answer->request->top_p);
	text(target, ",\"max_output_tokens\":");
	number(target, answer->request->max_tokens);
	text(target,
	     ",\"text\":{\"format\":{\"type\":\"text\"}},\"truncation\":\"disabled\",\"usage\":");
	if (done) {
		text(target, "{\"input_tokens\":");
		number(target, answer->prompt_token_count);
		text(target, ",\"output_tokens\":");
		number(target, answer->completion_token_count);
		text(target, ",\"total_tokens\":");
		number(target, (uint64_t)answer->prompt_token_count +
				       answer->completion_token_count);
		text(target, ",\"input_tokens_details\":{\"cached_tokens\":");
		number(target, answer->cached_token_count);
		text(target,
		     ",\"cache_write_tokens\":0},\"output_tokens_details\":{\"reasoning_tokens\":");
		number(target, answer->reasoning_token_count);
		text(target, "}}");
	} else {
		text(target, "null");
	}
	text(target, "}");
}

void server_responses_begin(struct chat_answer *answer, const char *model,
			    int64_t created)
{
	const struct chat_buffer *const fields =
		answer->pending_responses_fields;
	answer->pending_responses_fields = NULL;
	server_responses_free(answer);
	answer->responses = FAULT_POINT(chat_memory) ?
				    NULL :
				    calloc(1, sizeof(*answer->responses));
	if (answer->responses != NULL) {
		answer->responses->fields = fields;
		answer->responses->model =
			FAULT_POINT(chat_memory) ? NULL : strdup(model);
		answer->responses->created = created > 0 ? (uint64_t)created :
							   0;
	}
	if (answer->responses == NULL || answer->responses->model == NULL) {
		answer->output.is_short = true;
		return;
	}
	if (answer->request->is_streamed) {
		text(&answer->output, STREAM_HEAD);
		event_begin(answer, "response.created");
		text(&answer->output, ",\"response\":");
		response_write(answer, &answer->output, "in_progress", false);
		event_end(answer);
		event_begin(answer, "response.in_progress");
		text(&answer->output, ",\"response\":");
		response_write(answer, &answer->output, "in_progress", false);
		event_end(answer);
	}
}

void server_responses_text(struct chat_answer *answer, const char *bytes,
			   size_t count)
{
	if (answer->responses == NULL || answer->output.is_short) {
		return;
	}
	item_open(answer, answer->is_reasoning);
	struct chat_buffer *const target =
		answer->is_reasoning ? &answer->reasoning : &answer->content;
	chat_buffer_write(target, bytes, count);
	answer->output.is_short = answer->output.is_short || target->is_short;
	if (answer->request->is_streamed) {
		event_begin(answer,
			    answer->is_reasoning ?
				    "response.reasoning_summary_text.delta" :
				    "response.output_text.delta");
		event_item_id(answer);
		text(&answer->output,
		     answer->is_reasoning ? ",\"summary_index\":0,\"delta\":" :
					    ",\"content_index\":0,\"delta\":");
		string(&answer->output, bytes, count);
		if (!answer->is_reasoning) {
			text(&answer->output, ",\"logprobs\":[]");
		}
		event_end(answer);
	}
}

static void call_write(struct chat_answer *answer, struct chat_buffer *target,
		       struct json_bytes name,
		       const struct chat_buffer *arguments, bool done)
{
	char call_id[ITEM_ID_BYTES];
	const char *const separator =
		memmem(name.data, name.count, "::", strlen("::"));
	const size_t prefix =
		separator != NULL ? (size_t)(separator - name.data) : 0;
	const size_t bare = separator != NULL ? prefix + strlen("::") : 0;
	(void)snprintf(call_id, sizeof(call_id), "%s-%" PRIu32,
		       answer->identifier, answer->call_count);
	text(target, "{\"id\":");
	quoted(target, answer->responses->item_id);
	text(target, ",\"type\":\"function_call\",\"status\":");
	quoted(target, done ? "completed" : "in_progress");
	text(target, ",\"call_id\":");
	quoted(target, call_id);
	text(target, ",\"name\":");
	string(target, name.data + bare, name.count - bare);
	if (separator != NULL) {
		text(target, ",\"namespace\":");
		string(target, name.data, prefix);
	}
	text(target, ",\"arguments\":");
	string(target, done && arguments->bytes != NULL ? arguments->bytes : "",
	       done ? arguments->count : 0);
	text(target, "}");
}

void server_responses_call(struct chat_answer *answer, struct json_bytes name,
			   const struct chat_buffer *arguments)
{
	if (answer->responses == NULL || answer->output.is_short) {
		return;
	}
	struct server_responses_output *const state = answer->responses;
	item_close(answer, "completed");
	(void)snprintf(state->item_id, sizeof(state->item_id), "fc_%s_%" PRIu64,
		       answer->identifier, state->index);
	if (answer->request->is_streamed) {
		event_begin(answer, "response.output_item.added");
		event_index(answer);
		text(&answer->output, ",\"item\":");
		call_write(answer, &answer->output, name, arguments, false);
		event_end(answer);
		event_begin(answer, "response.function_call_arguments.delta");
		event_item_id(answer);
		text(&answer->output, ",\"delta\":");
		string(&answer->output, arguments->bytes, arguments->count);
		event_end(answer);
		event_begin(answer, "response.function_call_arguments.done");
		event_item_id(answer);
		text(&answer->output, ",\"arguments\":");
		string(&answer->output, arguments->bytes, arguments->count);
		event_end(answer);
	}
	struct chat_buffer item = { 0 };
	call_write(answer, &item, name, arguments, true);
	if (answer->request->is_streamed) {
		event_begin(answer, "response.output_item.done");
		event_index(answer);
		text(&answer->output, ",\"item\":");
		chat_buffer_write(&answer->output, item.bytes, item.count);
		event_end(answer);
	}
	append_item(answer, &item);
	chat_buffer_free(&item);
}

void server_responses_finish(struct chat_answer *answer, const char *reason)
{
	if (answer->responses == NULL || answer->responses->model == NULL ||
	    answer->output.is_short) {
		answer->output.is_short = true;
		return;
	}
	const char *const status = strcmp(reason, "length") == 0 ?
					   "incomplete" :
				   strcmp(reason, "failed") == 0 ? "failed" :
								   "completed";
	item_close(answer, strcmp(status, "completed") == 0 ? "completed" :
							      "incomplete");
	if (answer->request->is_streamed) {
		event_begin(answer, strcmp(status, "incomplete") == 0 ?
					    "response.incomplete" :
				    strcmp(status, "failed") == 0 ?
					    "response.failed" :
					    "response.completed");
		text(&answer->output, ",\"response\":");
		response_write(answer, &answer->output, status, true);
		event_end(answer);
	} else {
		struct chat_buffer body = { 0 };
		response_write(answer, &body, status, true);
		char head[HEADER_BYTES];
		const int count = snprintf(
			head, sizeof(head),
			"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnectio"
			"n: close\r\n\r\n",
			body.count);
		chat_buffer_write(&answer->output, head,
				  count > 0 && (size_t)count < sizeof(head) ?
					  (size_t)count :
					  0);
		chat_buffer_write(&answer->output, body.bytes, body.count);
		answer->output.is_short = answer->output.is_short ||
					  body.is_short || count <= 0 ||
					  (size_t)count >= sizeof(head);
		chat_buffer_free(&body);
	}
}

void server_responses_fail(struct chat_answer *answer)
{
	if (!answer->is_finished && answer->request->is_streamed) {
		server_responses_finish(answer, "failed");
	} else if (!answer->is_finished) {
		const struct chat_refusal refusal = { HTTP_SERVER_ERROR,
						      CHAT_ENGINE_FAILED, NULL,
						      0, 0 };
		chat_refusal_write(&answer->output, &refusal, NULL,
				   CHAT_ROUTE_RESPONSES);
	}
	answer->is_finished = true;
}

void server_responses_free(struct chat_answer *answer)
{
	if (answer->responses != NULL) {
		chat_buffer_free(&answer->responses->items);
		free(answer->responses->model);
		free(answer->responses);
		answer->responses = NULL;
	}
}
