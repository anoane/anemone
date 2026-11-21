/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Anthropic wire output from shared decoded boundaries, with no inference ownership. */
#include "server_messages.h"
#include <inttypes.h>
#include "models/deepseek_v41_flash/prompt/ds41f_chat_template.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ESCAPE_FACTOR 6u
#define QUOTE_BYTES 2u
#define NUMBER_BYTES 32u
#define HEADER_BYTES 256u
#define TOOL_ID_BYTES 128u
#define STREAM_HEAD \
	"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"

static void text(struct chat_buffer *buffer, const char *value)
{
	chat_buffer_write(buffer, value, strlen(value));
}
static void number(struct chat_buffer *buffer, uint64_t value)
{
	char digits[NUMBER_BYTES];
	const int count = snprintf(digits, sizeof(digits), "%" PRIu64, value);
	if (count <= 0 || (size_t)count >= sizeof(digits))
		buffer->is_short = true;
	else
		chat_buffer_write(buffer, digits, (size_t)count);
}
static void string(struct chat_buffer *buffer, const char *bytes, size_t count)
{
	const size_t bound = count * ESCAPE_FACTOR + QUOTE_BYTES;
	char *const room = count <= CHAT_MOST_OUTPUT_BYTES / ESCAPE_FACTOR ?
				   chat_buffer_room(buffer, bound) :
				   NULL;
	if (room == NULL) {
		buffer->is_short = true;
		return;
	}
	struct json_writer writer = { room, bound, 0 };
	json_write_string(&writer, (struct json_bytes){ bytes, count });
	buffer->count += writer.length;
}
static void quoted(struct chat_buffer *buffer, const char *value)
{
	string(buffer, value, strlen(value));
}
void server_messages_refuse(struct chat_buffer *output,
			    const struct chat_refusal *refusal)
{
	const uint32_t status = refusal->status;
	const char *const type =
		status == HTTP_UNAUTHORIZED	 ? "authentication_error" :
		status == HTTP_FORBIDDEN	 ? "permission_error" :
		status == HTTP_NOT_FOUND	 ? "not_found_error" :
		status == HTTP_CONTENT_TOO_LARGE ? "request_too_large" :
		status == HTTP_UNAVAILABLE	 ? "overloaded_error" :
		status >= HTTP_SERVER_ERROR	 ? "api_error" :
						   "invalid_request_error";
	chat_json_body_begin(output);
	text(output, "{\"type\":\"error\",\"error\":{\"type\":");
	quoted(output, type);
	text(output, ",\"message\":");
	quoted(output, refusal->message != NULL ? refusal->message : type);
	text(output, "}}\n");
	chat_json_body_finish(
		output, status,
		status == HTTP_UNAVAILABLE ? "Retry-After: " CHAT_RETRY_AFTER
					     "\r\n" :
		status == HTTP_METHOD_NOT_ALLOWED ? "Allow: POST\r\n" :
		status == HTTP_UNAUTHORIZED ? "WWW-Authenticate: Bearer\r\n" :
					      "");
}

static void event(struct chat_answer *answer, const char *type, bool indexed)
{
	text(&answer->output, "event: ");
	text(&answer->output, type);
	text(&answer->output, "\ndata: {\"type\":");
	quoted(&answer->output, type);
	if (indexed) {
		text(&answer->output, ",\"index\":");
		number(&answer->output, answer->messages.index);
	}
}
static void end_event(struct chat_answer *answer)
{
	text(&answer->output, "}\n\n");
}
static void usage(struct chat_answer *answer, struct chat_buffer *target,
		  bool initial)
{
	text(target, "{\"input_tokens\":");
	number(target, answer->prompt_token_count);
	text(target, ",\"output_tokens\":");
	number(target, initial ? 0 : answer->completion_token_count);
	text(target, ",\"output_tokens_details\":{\"thinking_tokens\":");
	number(target, initial ? 0 : answer->reasoning_token_count);
	text(target, "}}");
}
static void message(struct chat_answer *answer, struct chat_buffer *target,
		    const char *reason)
{
	text(target, "{\"id\":");
	quoted(target, answer->identifier);
	text(target,
	     ",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"anemone\",\"content\":[");
	if (reason != NULL)
		chat_buffer_write(target, answer->content.bytes,
				  answer->content.count);
	text(target, "],\"stop_reason\":");
	if (reason != NULL)
		quoted(target, reason);
	else
		text(target, "null");
	text(target, ",\"stop_sequence\":");
	if (reason != NULL && strcmp(reason, "stop_sequence") == 0)
		string(target, answer->request->stops[answer->matched_stop],
		       answer->request->stop_byte_counts[answer->matched_stop]);
	else
		text(target, "null");
	text(target, ",\"usage\":");
	usage(answer, target, reason == NULL);
	text(target, "}");
}
static void signature(struct chat_answer *answer, struct chat_buffer *target)
{
	char value[TOOL_ID_BYTES];
	const int count = snprintf(value, sizeof(value),
				   SERVER_SIGNATURE_PREFIX "%u",
				   answer->request->reasoning_effort);
	if (count <= 0 || (size_t)count >= sizeof(value))
		target->is_short = true;
	else
		string(target, value, (size_t)count);
}
static void close_text(struct chat_answer *answer)
{
	struct server_messages_output *const state = &answer->messages;
	if (!state->text_open)
		return;
	if (answer->request->is_streamed) {
		if (state->reasoning) {
			event(answer, "content_block_delta", true);
			text(&answer->output,
			     ",\"delta\":{\"type\":\"signature_delta\",\"signature\":");
			signature(answer, &answer->output);
			text(&answer->output, "}");
			end_event(answer);
		}
		event(answer, "content_block_stop", true);
		end_event(answer);
		state->index++;
	} else {
		struct chat_buffer *const block = &answer->content;
		text(block, state->index == 0 ? "" : ",");
		text(block, state->reasoning ?
				    "{\"type\":\"thinking\",\"thinking\":" :
				    "{\"type\":\"text\",\"text\":");
		string(block, answer->reasoning.bytes, answer->reasoning.count);
		if (state->reasoning) {
			text(block, ",\"signature\":");
			signature(answer, block);
		}
		text(block, "}");
		answer->output.is_short = answer->output.is_short ||
					  block->is_short;
		state->index++;
	}
	answer->reasoning.count = 0;
	state->text_open = false;
}

void server_messages_begin(struct chat_answer *answer)
{
	server_messages_free(answer);
	answer->messages.active = true;
	if (answer->request->is_streamed) {
		text(&answer->output, STREAM_HEAD);
		event(answer, "message_start", false);
		text(&answer->output, ",\"message\":");
		message(answer, &answer->output, NULL);
		end_event(answer);
	}
	if (answer->is_reasoning)
		server_messages_text(answer, "", 0);
}
void server_messages_text(struct chat_answer *answer, const char *bytes,
			  size_t count)
{
	struct server_messages_output *const state = &answer->messages;
	if (!state->active || state->failed)
		return;
	if (state->text_open && state->reasoning != answer->is_reasoning)
		close_text(answer);
	if (!state->text_open) {
		state->text_open = true;
		state->reasoning = answer->is_reasoning;
		if (answer->request->is_streamed) {
			event(answer, "content_block_start", true);
			text(&answer->output,
			     state->reasoning ?
				     ",\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}" :
				     ",\"content_block\":{\"type\":\"text\",\"text\":\"\"}");
			end_event(answer);
		}
	}
	if (answer->request->is_streamed) {
		if (count != 0) {
			event(answer, "content_block_delta", true);
			text(&answer->output,
			     state->reasoning ?
				     ",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":" :
				     ",\"delta\":{\"type\":\"text_delta\",\"text\":");
			string(&answer->output, bytes, count);
			text(&answer->output, "}");
			end_event(answer);
		}
	} else {
		chat_buffer_write(&answer->reasoning, bytes, count);
		answer->output.is_short = answer->output.is_short ||
					  answer->reasoning.is_short;
	}
}
/* Generated arguments must also meet JSON's contract; DSML can contain arbitrary scalar spellings. */

void server_messages_call(struct chat_answer *answer, struct json_bytes name,
			  const struct chat_buffer *arguments)
{
	struct server_messages_output *const state = &answer->messages;
	if (!state->active || state->failed)
		return;
	struct json_value document = { 0 };
	bool valid = name.count > 0 && name.count <= MESSAGES_NAME_BYTES &&
		     answer->call_count < MESSAGES_MOST_TOOLS;
	for (size_t index = 0; index < name.count; index++) {
		const char byte = name.data[index];
		valid = valid && ((byte >= 'a' && byte <= 'z') ||
				  (byte >= 'A' && byte <= 'Z') ||
				  (byte >= '0' && byte <= '9') || byte == '_' ||
				  byte == '-');
	}
	struct json_scratch scratch = { .bytes = state->scratch,
					.capacity = state->scratch_capacity };
	if (valid && scratch.bytes != NULL) {
		const enum json_document parsed = json_read_document(
			arguments->bytes, arguments->count, &document);
		valid = parsed == JSON_DOCUMENT_VALID;
	} else
		valid = false;
	if (valid)
		valid = document.text[document.begin] == '{' &&
			!json_has_unquoted(document, "NI");
	if (valid) {
		const bool repeated = json_has_repeated_key(&scratch, document);
		valid = !repeated && !scratch.is_short;
	}
	if (!valid) {
		server_messages_fail(answer);
		return;
	}
	close_text(answer);
	char identifier[TOOL_ID_BYTES];
	const int count = snprintf(identifier, sizeof(identifier),
				   "toolu_%s_%" PRIu32, answer->identifier,
				   answer->call_count);
	if (count <= 0 || (size_t)count >= sizeof(identifier)) {
		answer->output.is_short = true;
		return;
	}
	struct chat_buffer *const block = answer->request->is_streamed ?
						  &answer->output :
						  &answer->content;
	if (answer->request->is_streamed) {
		event(answer, "content_block_start", true);
		text(block, ",\"content_block\":");
	} else
		text(block, state->index == 0 ? "" : ",");
	text(block, "{\"type\":\"tool_use\",\"id\":");
	quoted(block, identifier);
	text(block, ",\"name\":");
	string(block, name.data, name.count);
	text(block, ",\"input\":");
	if (answer->request->is_streamed)
		text(block, "{}");
	else
		chat_buffer_write(block, arguments->bytes, arguments->count);
	text(block, "}");
	if (answer->request->is_streamed) {
		end_event(answer);
		event(answer, "content_block_delta", true);
		text(&answer->output,
		     ",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":");
		string(&answer->output, arguments->bytes, arguments->count);
		text(&answer->output, "}");
		end_event(answer);
		event(answer, "content_block_stop", true);
		end_event(answer);
		state->index++;
	} else
		state->index++;
	answer->output.is_short = answer->output.is_short || block->is_short;
}
void server_messages_finish(struct chat_answer *answer, const char *reason)
{
	struct server_messages_output *const state = &answer->messages;
	if (!state->active || state->failed)
		return;
	if (state->index == 0 && !state->text_open)
		server_messages_text(answer, "", 0);
	close_text(answer);
	const char *const stop =
		strcmp(reason, "length") == 0 ?
			(answer->completion_token_count <
					 answer->request->max_tokens ?
				 "model_context_window_exceeded" :
				 "max_tokens") :
		strcmp(reason, "tool_calls") == 0 ? "tool_use" :
		answer->matched_stop < answer->request->stop_count ?
						    "stop_sequence" :
						    "end_turn";
	if (answer->request->is_streamed) {
		event(answer, "message_delta", false);
		text(&answer->output, ",\"delta\":{\"stop_reason\":");
		quoted(&answer->output, stop);
		text(&answer->output, ",\"stop_sequence\":");
		if (strcmp(stop, "stop_sequence") == 0)
			string(&answer->output,
			       answer->request->stops[answer->matched_stop],
			       answer->request
				       ->stop_byte_counts[answer->matched_stop]);
		else
			text(&answer->output, "null");
		text(&answer->output, "},\"usage\":{\"output_tokens\":");
		number(&answer->output, answer->completion_token_count);
		text(&answer->output,
		     ",\"output_tokens_details\":{\"thinking_tokens\":");
		number(&answer->output, answer->reasoning_token_count);
		text(&answer->output, "}}");
		end_event(answer);
		event(answer, "message_stop", false);
		end_event(answer);
	} else {
		chat_json_body_begin(&answer->output);
		message(answer, &answer->output, stop);
		text(&answer->output, "\n");
		chat_json_body_finish(&answer->output, HTTP_OK, "");
	}
}
void server_messages_fail(struct chat_answer *answer)
{
	if (!answer->messages.active) {
		answer->output.is_short = true;
		return;
	}
	if (answer->messages.failed)
		return;
	answer->messages.failed = true;
	if (answer->request->is_streamed) {
		text(&answer->output,
		     "event: error\ndata:"
		     " {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"generation failed or"
		     " produced unsupported tool output\"}}\n\n");
	} else {
		const struct chat_refusal refusal = {
			.status = HTTP_SERVER_ERROR,
			.message =
				"generation failed or produced unsupported tool output"
		};
		server_messages_refuse(&answer->output, &refusal);
	}
	answer->is_finished = true;
}
void server_messages_free(struct chat_answer *answer)
{
	answer->messages = (struct server_messages_output){ 0 };
}
