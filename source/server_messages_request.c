/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Validate the protocol before normalizing into the existing trusted Chat template. */
#include "server_messages.h"
#include <math.h>
#include <stdio.h>
#include "anemone/chat_template.h"
#include "anemone/sampler_records.h"
#include <stdlib.h>
#include <string.h>
#define SCRATCH_FACTOR 4u
#define SCRATCH_SLACK 4096u
#define ESCAPE_FACTOR 6u
#define QUOTE_BYTES 2u
#define NUMBER_BYTES 64u
#define DECIMAL_BASE 10
#define MIN_THINKING_BUDGET 1024u
#define LOW_EFFORT 50u
#define ADAPTIVE_DIVISOR 4u
#define ADAPTIVE_HIGH_SHARE 3u
struct reader {
	struct json_scratch scratch;
	struct chat_buffer *body;
	struct chat_refusal *refusal;
	struct json_value pending[MESSAGES_MOST_TOOLS];
	uint32_t pending_count;
	bool has_message;
	uint32_t effort, budget, adaptive_share;
};
static bool kind(struct json_value value, char opening)
{
	return value.text != NULL && value.text[value.begin] == opening;
}
static bool word(struct reader *reader, struct json_value value,
		 const char *text)
{
	const size_t saved = reader->scratch.used;
	const bool equal = json_string_is(&reader->scratch, value, text);
	reader->scratch.used = saved;
	return equal;
}
static struct json_value member(struct reader *reader, struct json_value object,
				const char *name)
{
	struct json_value key = { 0 }, value = { 0 };
	for (size_t cursor = object.begin;
	     kind(object, '{') && json_next(object, &cursor, &key, &value);) {
		if (word(reader, key, name))
			return value;
	}
	return (struct json_value){ 0 };
}
static void refuse(struct reader *reader, const char *message)
{
	if (reader->refusal->status == 0)
		*reader->refusal =
			(struct chat_refusal){ .status = HTTP_BAD_REQUEST,
					       .message = message };
}
static void fields(struct reader *reader, struct json_value object,
		   const char *allowed)
{
	if (!kind(object, '{')) {
		refuse(reader, "expected an object");
		return;
	}
	struct json_value key = { 0 }, value = { 0 };
	for (size_t cursor = object.begin;
	     json_next(object, &cursor, &key, &value);) {
		const size_t saved = reader->scratch.used;
		const struct json_bytes name =
			json_unescape(&reader->scratch, key);
		bool found = false;
		for (const char *begin = allowed; *begin != '\0';) {
			const char *end = strchr(begin, '|');
			end = end != NULL ? end : begin + strlen(begin);
			found = found ||
				(name.count != 0 &&
				 (size_t)(end - begin) == name.count &&
				 memcmp(begin, name.data, name.count) == 0);
			begin = *end != '\0' ? end + 1 : end;
		}
		reader->scratch.used = saved;
		if (!found)
			refuse(reader, "unsupported field in Messages request");
	}
}

static void text(struct reader *reader, struct chat_buffer *target,
		 const char *value)
{
	const size_t count = strlen(value);
	if (target->count > CHAT_MOST_BODY_BYTES ||
	    count > CHAT_MOST_BODY_BYTES - target->count) {
		reader->refusal->status = HTTP_CONTENT_TOO_LARGE;
		return;
	}
	chat_buffer_write(target, value, count);
}
static void raw(struct reader *reader, struct chat_buffer *target,
		struct json_value value)
{
	if (value.text != NULL) {
		const size_t count = value.end - value.begin;
		if (target->count > CHAT_MOST_BODY_BYTES ||
		    count > CHAT_MOST_BODY_BYTES - target->count) {
			reader->refusal->status = HTTP_CONTENT_TOO_LARGE;
			return;
		}
		chat_buffer_write(target, value.text + value.begin, count);
	}
}
static void string(struct reader *reader, struct chat_buffer *target,
		   const char *bytes, size_t count)
{
	struct json_writer measured = { 0 };
	json_write_string(&measured, (struct json_bytes){ bytes, count });
	const size_t bound = measured.length;
	if (target->count > CHAT_MOST_BODY_BYTES ||
	    bound > CHAT_MOST_BODY_BYTES - target->count) {
		reader->refusal->status = HTTP_CONTENT_TOO_LARGE;
		return;
	}
	char *const room = chat_buffer_room(target, bound);
	if (room == NULL) {
		target->is_short = true;
		return;
	}
	struct json_writer writer = { room, bound, 0 };
	json_write_string(&writer, (struct json_bytes){ bytes, count });
	target->count += writer.length;
}
static bool name_valid(struct reader *reader, struct json_value value,
		       bool tool_name)
{
	if (!kind(value, '"'))
		return false;
	const size_t saved = reader->scratch.used;
	const struct json_bytes bytes = json_unescape(&reader->scratch, value);
	bool valid = bytes.count > 0 &&
		     bytes.count <= (tool_name ? MESSAGES_NAME_BYTES :
						 MESSAGES_ID_BYTES);
	for (size_t index = 0; index < bytes.count; index++) {
		const char byte = bytes.data[index];
		valid = valid && ((byte >= 'a' && byte <= 'z') ||
				  (byte >= 'A' && byte <= 'Z') ||
				  (byte >= '0' && byte <= '9') || byte == '_' ||
				  byte == '-');
	}
	reader->scratch.used = saved;
	return valid;
}
static bool equal(struct reader *reader, struct json_value first,
		  struct json_value second)
{
	const size_t saved = reader->scratch.used;
	const struct json_bytes left = json_unescape(&reader->scratch, first),
				right = json_unescape(&reader->scratch, second);
	const bool matches = left.count != 0 && left.count == right.count &&
			     memcmp(left.data, right.data, left.count) == 0;
	reader->scratch.used = saved;
	return matches;
}
static void append_string(struct reader *reader, struct chat_buffer *target,
			  struct json_value value)
{
	if (!kind(value, '"')) {
		refuse(reader, "text must be a string");
		return;
	}
	const size_t saved = reader->scratch.used;
	const struct json_bytes bytes = json_unescape(&reader->scratch, value);
	if (bytes.count != 0)
		chat_buffer_write(target, bytes.data, bytes.count);
	reader->scratch.used = saved;
}
static bool empty_array(struct json_value value)
{
	struct json_value element = { 0 };
	size_t cursor = value.begin;
	return kind(value, '[') && !json_next(value, &cursor, NULL, &element);
}

static void plain_content(struct reader *reader, struct chat_buffer *target,
			  struct json_value content)
{
	if (kind(content, '"')) {
		append_string(reader, target, content);
		return;
	}
	if (!kind(content, '[')) {
		refuse(reader,
		       "content must be a string or nonempty text-block array");
		return;
	}
	struct json_value block = { 0 };
	uint32_t count = 0;
	for (size_t cursor = content.begin;
	     json_next(content, &cursor, NULL, &block);) {
		fields(reader, block, "type|text|citations");
		const struct json_value citations =
			member(reader, block, "citations");
		if (citations.text != NULL && !kind(citations, 'n') &&
		    !empty_array(citations))
			refuse(reader, "citations are unsupported");
		count++;
		if (!word(reader, member(reader, block, "type"), "text") ||
		    count > CHAT_MOST_MESSAGES)
			refuse(reader,
			       "only bounded text blocks are supported here");
		append_string(reader, target, member(reader, block, "text"));
	}
	if (count == 0)
		refuse(reader, "content arrays must be nonempty");
}
static void start_message(struct reader *reader)
{
	text(reader, reader->body, reader->has_message ? "," : "");
	reader->has_message = true;
}
static void tool_result(struct reader *reader, struct json_value block)
{
	fields(reader, block, "type|tool_use_id|content|is_error");
	const struct json_value id = member(reader, block, "tool_use_id"),
				error = member(reader, block, "is_error");
	uint32_t match = reader->pending_count;
	for (uint32_t index = 0; kind(id, '"') && index < reader->pending_count;
	     index++)
		if (reader->pending[index].text != NULL &&
		    equal(reader, id, reader->pending[index]))
			match = index;
	if (match == reader->pending_count) {
		refuse(reader, "orphan or duplicate tool_result");
		return;
	}
	reader->pending[match].text = NULL;
	if (error.text != NULL && !kind(error, 't') && !kind(error, 'f'))
		refuse(reader, "is_error must be boolean");
	struct chat_buffer result = { 0 };
	if (kind(error, 't'))
		text(reader, &result, "Error: ");
	const struct json_value content = member(reader, block, "content");
	if (content.text != NULL)
		plain_content(reader, &result, content);
	start_message(reader);
	text(reader, reader->body, "{\"role\":\"tool\",\"tool_call_id\":");
	raw(reader, reader->body, id);
	text(reader, reader->body, ",\"content\":");
	string(reader, reader->body, result.bytes, result.count);
	text(reader, reader->body, "}");
	reader->body->is_short = reader->body->is_short || result.is_short;
	chat_buffer_free(&result);
}
static void message(struct reader *reader, struct json_value item,
		    bool *assistant_out)
{
	fields(reader, item, "role|content");
	const struct json_value role = member(reader, item, "role"),
				content = member(reader, item, "content");
	const bool assistant = word(reader, role, "assistant");
	if (!assistant && !word(reader, role, "user"))
		refuse(reader, "message role must be user or assistant");
	if (assistant && (!reader->has_message || reader->pending_count != 0))
		refuse(reader,
		       "history must begin with user and resolve pending tools immediately");
	struct chat_buffer flattened = { 0 }, calls = { 0 }, reasoning = { 0 };
	struct json_value signature = { 0 };
	bool has_thinking = false;
	uint32_t blocks = 0, results = 0, call_count = 0;
	bool has_text = false;
	if (kind(content, '"')) {
		append_string(reader, &flattened, content);
		has_text = true;
	} else if (kind(content, '[')) {
		struct json_value block = { 0 };
		for (size_t cursor = content.begin;
		     json_next(content, &cursor, NULL, &block);) {
			blocks++;
			if (blocks > CHAT_MOST_MESSAGES) {
				refuse(reader, "too many content blocks");
				break;
			}
			const struct json_value type =
				member(reader, block, "type");
			if (word(reader, type, "text")) {
				fields(reader, block, "type|text|citations");
				const struct json_value citations =
					member(reader, block, "citations");
				if (citations.text != NULL &&
				    !kind(citations, 'n') &&
				    !empty_array(citations))
					refuse(reader,
					       "citations are unsupported");
				if (call_count != 0)
					refuse(reader,
					       "text after tool_use is unsupported");
				append_string(reader, &flattened,
					      member(reader, block, "text"));
				has_text = true;
			} else if (assistant &&
				   word(reader, type, "thinking")) {
				fields(reader, block,
				       "type|thinking|signature");
				if (has_text || call_count != 0)
					refuse(reader,
					       "thinking must precede text and tool_use");
				append_string(reader, &reasoning,
					      member(reader, block,
						     "thinking"));
				signature = member(reader, block, "signature");
				if (signature.text != NULL &&
				    !kind(signature, '"'))
					refuse(reader,
					       "thinking signature must be a string");
				has_thinking = true;
			} else if (!assistant &&
				   word(reader, type, "tool_result")) {
				if (has_text)
					refuse(reader,
					       "tool_result must precede user text");
				tool_result(reader, block);
				results++;
			} else if (assistant &&
				   word(reader, type, "tool_use")) {
				fields(reader, block,
				       "type|id|name|input|caller|toolset_name");
				const struct json_value caller = member(
								reader, block,
								"caller"),
							toolset = member(
								reader, block,
								"toolset_name");
				if ((caller.text != NULL &&
				     !kind(caller, 'n')) ||
				    (toolset.text != NULL &&
				     !kind(toolset, 'n')))
					refuse(reader,
					       "tool caller and toolset extensions are unsupported");
				const struct json_value
					id = member(reader, block, "id"),
					name = member(reader, block, "name"),
					input = member(reader, block, "input");
				if (!name_valid(reader, id, false) ||
				    !name_valid(reader, name, true) ||
				    !kind(input, '{') ||
				    call_count >= MESSAGES_MOST_TOOLS) {
					refuse(reader,
					       "tool_use requires a supported ID/name and object input");
					continue;
				}
				for (uint32_t index = 0; index < call_count;
				     index++)
					if (equal(reader, id,
						  reader->pending[index]))
						refuse(reader,
						       "duplicate tool_use ID");
				reader->pending[call_count] = id;
				text(reader, &calls,
				     call_count == 0 ? "" : ",");
				call_count++;
				text(reader, &calls, "{\"id\":");
				raw(reader, &calls, id);
				text(reader, &calls,
				     ",\"type\":\"function\",\"function\":{\"name\":");
				raw(reader, &calls, name);
				text(reader, &calls, ",\"arguments\":");
				string(reader, &calls, input.text + input.begin,
				       input.end - input.begin);
				text(reader, &calls, "}}");
			} else
				refuse(reader,
				       "unsupported content block or role");
		}
		if (blocks == 0)
			refuse(reader, "content arrays must be nonempty");
	} else
		refuse(reader, "content is required");
	if (!assistant) {
		for (uint32_t index = 0; index < reader->pending_count; index++)
			if (reader->pending[index].text != NULL)
				refuse(reader, "missing immediate tool_result");
	}
	reader->pending_count = assistant ? call_count : 0;
	if (assistant || has_text || results == 0) {
		start_message(reader);
		text(reader, reader->body,
		     assistant ? "{\"role\":\"assistant\",\"content\":" :
				 "{\"role\":\"user\",\"content\":");
		string(reader, reader->body, flattened.bytes, flattened.count);
		if (has_thinking) {
			text(reader, reader->body, ",\"reasoning_content\":");
			string(reader, reader->body, reasoning.bytes,
			       reasoning.count);
			if (signature.text != NULL) {
				text(reader, reader->body, ",\"name\":");
				raw(reader, reader->body, signature);
			}
		}
		if (call_count != 0) {
			text(reader, reader->body, ",\"tool_calls\":[");
			chat_buffer_write(reader->body, calls.bytes,
					  calls.count);
			text(reader, reader->body, "]");
		}
		text(reader, reader->body, "}");
	}
	reader->body->is_short = reader->body->is_short || flattened.is_short ||
				 calls.is_short || reasoning.is_short;
	chat_buffer_free(&flattened);
	chat_buffer_free(&calls);
	chat_buffer_free(&reasoning);
	*assistant_out = assistant;
}
static void tools(struct reader *reader, struct json_value array)
{
	if (!kind(array, '[')) {
		refuse(reader, "tools must be an array");
		return;
	}
	text(reader, reader->body, ",\"tools\":[");
	struct json_value tool = { 0 }, names[MESSAGES_MOST_TOOLS];
	uint32_t count = 0;
	for (size_t cursor = array.begin;
	     json_next(array, &cursor, NULL, &tool);) {
		fields(reader, tool, "name|description|input_schema|strict");
		const struct json_value name = member(reader, tool, "name"),
					description = member(reader, tool,
							     "description"),
					schema = member(reader, tool,
							"input_schema"),
					strict = member(reader, tool, "strict");
		count++;
		if (count > MESSAGES_MOST_TOOLS) {
			refuse(reader, "too many tools");
			break;
		}
		if (!name_valid(reader, name, true) || !kind(schema, '{') ||
		    !word(reader, member(reader, schema, "type"), "object") ||
		    (description.text != NULL && !kind(description, '"')) ||
		    (strict.text != NULL && !kind(strict, 'f')))
			refuse(reader,
			       "tools require supported names/object schemas; strict generation is unsupported");
		for (uint32_t index = 0; index + 1 < count; index++)
			if (kind(name, '"') && kind(names[index], '"') &&
			    equal(reader, name, names[index]))
				refuse(reader, "duplicate tool name");
		names[count - 1] = name;
		text(reader, reader->body, count == 1 ? "" : ",");
		text(reader, reader->body,
		     "{\"type\":\"function\",\"function\":{\"name\":");
		raw(reader, reader->body, name);
		text(reader, reader->body, ",\"parameters\":");
		raw(reader, reader->body, schema);
		if (description.text != NULL) {
			text(reader, reader->body, ",\"description\":");
			raw(reader, reader->body, description);
		}
		text(reader, reader->body, "}}");
	}
	text(reader, reader->body, "]");
}
static bool number_in(struct json_value value, double minimum, double maximum,
		      bool integer)
{
	char digits[NUMBER_BYTES] = { 0 };
	const size_t count = value.end - value.begin;
	if (value.text == NULL || count == 0 || count >= sizeof(digits))
		return false;
	memcpy(digits, value.text + value.begin, count);
	char *end = NULL;
	const double number = strtod(digits, &end);
	return end == digits + count && isfinite(number) && number >= minimum &&
	       number <= maximum &&
	       (!integer || strspn(digits, "0123456789") == count);
}
static uint32_t integer_value(struct json_value value)
{
	char digits[NUMBER_BYTES] = { 0 };
	const size_t count = value.end - value.begin;
	if (value.text == NULL || count >= sizeof(digits))
		return 0;
	memcpy(digits, value.text + value.begin, count);
	return (uint32_t)strtoul(digits, NULL, DECIMAL_BASE);
}
static void thinking_settings(struct reader *reader, struct json_value thinking,
			      struct json_value output, uint32_t maximum)
{
	uint32_t effort = CHAT_TEMPLATE_DEFAULT_EFFORT;
	if (output.text != NULL) {
		fields(reader, output, "effort");
		const struct json_value level =
			member(reader, output, "effort");
		if (word(reader, level, "low"))
			effort = LOW_EFFORT;
		else if (word(reader, level, "medium") ||
			 word(reader, level, "high") ||
			 word(reader, level, "xhigh"))
			effort = CHAT_TEMPLATE_DEFAULT_EFFORT;
		else if (word(reader, level, "max"))
			effort = CHAT_TEMPLATE_HIGHEST_EFFORT;
		else
			refuse(reader, "unsupported output_config.effort");
	}
	if (thinking.text == NULL) {
		if (output.text != NULL)
			refuse(reader,
			       "effort requires enabled or adaptive thinking");
		return;
	}
	const struct json_value type = member(reader, thinking, "type");
	if (word(reader, type, "disabled")) {
		fields(reader, thinking, "type");
		if (output.text != NULL)
			refuse(reader,
			       "disabled thinking has no reasoning effort");
	} else if (word(reader, type, "enabled")) {
		fields(reader, thinking, "type|budget_tokens");
		const struct json_value budget =
			member(reader, thinking, "budget_tokens");
		if (!number_in(budget, MIN_THINKING_BUDGET, UINT32_MAX, true) ||
		    integer_value(budget) >= maximum)
			refuse(reader,
			       "enabled thinking budget must be at least 1024 and below max_tokens");
		reader->budget = integer_value(budget);
		reader->effort = effort;
	} else if (word(reader, type, "adaptive")) {
		fields(reader, thinking, "type");
		if (maximum < ADAPTIVE_DIVISOR)
			refuse(reader,
			       "adaptive thinking needs at least four output tokens");
		const uint32_t share = effort == LOW_EFFORT ?
					       1u :
				       effort == CHAT_TEMPLATE_HIGHEST_EFFORT ?
					       ADAPTIVE_DIVISOR :
					       ADAPTIVE_HIGH_SHARE;
		reader->adaptive_share = share;
		reader->effort = effort;
	} else
		refuse(reader,
		       "thinking.type must be enabled, adaptive or disabled");
	if (reader->effort != 0) {
		char digits[NUMBER_BYTES];
		const int count = snprintf(digits, sizeof(digits),
					   ",\"reasoning_effort\":%u",
					   reader->effort);
		if (count > 0 && (size_t)count < sizeof(digits))
			chat_buffer_write(reader->body, digits, (size_t)count);
		else
			reader->body->is_short = true;
	}
}

uint32_t server_messages_read(const char *body, size_t count,
			      bool legacy_sampling,
			      const struct chat_request *defaults,
			      struct chat_request *request,
			      struct chat_buffer *converted,
			      struct json_scratch *workspace,
			      struct chat_refusal *refusal)
{
	if (body == NULL || defaults == NULL || request == NULL ||
	    converted == NULL || refusal == NULL)
		return HTTP_SERVER_ERROR;
	if (workspace == NULL || workspace->bytes == NULL)
		return HTTP_SERVER_ERROR;
	*refusal = (struct chat_refusal){ 0 };
	if (count > CHAT_MOST_BODY_BYTES) {
		refusal->status = HTTP_CONTENT_TOO_LARGE;
		return refusal->status;
	}
	struct reader reader = { .body = converted, .refusal = refusal };
	reader.scratch =
		(struct json_scratch){ .bytes = workspace->bytes,
				       .capacity = workspace->capacity };
	struct json_value document = { 0 };
	const enum json_document parsed =
		json_read_document(body, count, &document);
	bool valid = parsed == JSON_DOCUMENT_VALID && kind(document, '{');
	if (valid) {
		const bool repeated =
			json_has_repeated_key(&reader.scratch, document);
		valid = !repeated && !reader.scratch.is_short &&
			!json_has_unquoted(document, "NI");
	}
	if (!valid)
		refuse(&reader,
		       "expected finite UTF-8 JSON without duplicate keys");
	if (refusal->status == 0) {
		fields(&reader, document,
		       "model|max_tokens|messages|system|tools|tool_choice|stream|temperature|top_p|top_k|"
		       "stop_sequences|thinking|output_config");
		const struct json_value model = member(&reader, document,
						       "model"),
					limit = member(&reader, document,
						       "max_tokens"),
					messages = member(&reader, document,
							  "messages");
		if (!kind(model, '"'))
			refuse(&reader, "model is required");
		else if (!word(&reader, model, MESSAGES_MODEL))
			*refusal = (struct chat_refusal){
				.status = HTTP_NOT_FOUND,
				.message = "unknown model"
			};
		if (!number_in(limit, 1, UINT32_MAX, true) ||
		    !kind(messages, '['))
			refuse(&reader,
			       "positive integer max_tokens and messages array are required");
		text(&reader, converted,
		     "{\"model\":\"anemone\",\"messages\":[");
		const struct json_value system =
			member(&reader, document, "system");
		if (system.text != NULL) {
			struct chat_buffer flattened = { 0 };
			plain_content(&reader, &flattened, system);
			text(&reader, converted,
			     "{\"role\":\"system\",\"content\":");
			string(&reader, converted, flattened.bytes,
			       flattened.count);
			text(&reader, converted, "}");
			converted->is_short = converted->is_short ||
					      flattened.is_short;
			chat_buffer_free(&flattened);
		}
		struct json_value item = { 0 };
		uint32_t message_count = 0;
		bool assistant = false;
		for (size_t cursor = messages.begin;
		     refusal->status == 0 &&
		     json_next(messages, &cursor, NULL, &item);) {
			if (++message_count > CHAT_MOST_MESSAGES)
				refuse(&reader, "too many messages");
			if (message_count == 1 && system.text != NULL)
				text(&reader, converted, ",");
			message(&reader, item, &assistant);
		}
		if (message_count == 0 || assistant)
			refuse(&reader,
			       "messages must begin and end with user");
		text(&reader, converted, "]");
		const struct json_value declared = member(&reader, document,
							  "tools"),
					choice = member(&reader, document,
							"tool_choice"),
					thinking = member(&reader, document,
							  "thinking");
		if (declared.text != NULL)
			tools(&reader, declared);
		if (choice.text != NULL) {
			fields(&reader, choice, "type");
			const struct json_value type =
				member(&reader, choice, "type");
			if (!word(&reader, type, "auto") &&
			    !word(&reader, type, "none"))
				refuse(&reader,
				       "only auto/none tool_choice is supported");
			text(&reader, converted, ",\"tool_choice\":");
			raw(&reader, converted, type);
		}
		thinking_settings(&reader, thinking,
				  member(&reader, document, "output_config"),
				  integer_value(limit));

		struct json_value key = { 0 }, value = { 0 };
		for (size_t cursor = document.begin;
		     json_next(document, &cursor, &key, &value);) {
			const bool temperature =
					   word(&reader, key, "temperature"),
				   top_p = word(&reader, key, "top_p"),
				   top_k = word(&reader, key, "top_k");
			if (temperature || top_p || top_k) {
				if (!legacy_sampling ||
				    !(top_k ? number_in(value, 0,
							SAMPLER_MOST_VOCABULARY,
							true) :
					      number_in(value, 0, 1, false)) ||
				    (top_p && number_in(value, 0, 0, false)))
					refuse(&reader,
					       "sampling controls require version 2023-06-01 and supported ranges");
			}
			if (temperature || top_p || top_k ||
			    word(&reader, key, "max_tokens") ||
			    word(&reader, key, "stream") ||
			    word(&reader, key, "stop_sequences")) {
				if ((word(&reader, key, "stream") &&
				     !kind(value, 't') && !kind(value, 'f')) ||
				    (word(&reader, key, "stop_sequences") &&
				     !kind(value, '[')))
					refuse(&reader,
					       "invalid stream or stop_sequences type");
				text(&reader, converted, ",");
				if (word(&reader, key, "stop_sequences"))
					text(&reader, converted, "\"stop\"");
				else
					raw(&reader, converted, key);
				text(&reader, converted, ":");
				raw(&reader, converted, value);
			}
		}
		text(&reader, converted, "}");
	}
	if (refusal->status == 0 && converted->count > CHAT_MOST_BODY_BYTES)
		refusal->status = HTTP_CONTENT_TOO_LARGE;
	if (refusal->status != HTTP_CONTENT_TOO_LARGE && converted->is_short)
		refusal->status = HTTP_SERVER_ERROR;
	if (refusal->status == 0) {
		struct chat_request local = *defaults;
		local.reasoning_effort = 0;
		local.temperature = local.top_p = 1.0f;
		const uint32_t status = chat_request_read(
			converted->bytes, converted->count, MESSAGES_MODEL,
			NULL, &local, request, refusal);
		if (status != 0) {
			refusal->status =
				refusal->status != 0 ? refusal->status : status;
			refusal->member_begin = refusal->member_end = 0;
		} else {
			request->is_messages = true;
			request->thinking_budget = reader.budget;
			request->thinking_adaptive_share =
				reader.adaptive_share;
		}
	}
	if (reader.scratch.is_short && refusal->status != HTTP_SERVER_ERROR) {
		refusal->status = HTTP_BAD_REQUEST;
		refusal->message = "request exceeds normalization workspace";
	}
	const size_t measured = count * SCRATCH_FACTOR + SCRATCH_SLACK;
	explicit_bzero(reader.scratch.bytes,
		       measured < reader.scratch.capacity ?
			       measured :
			       reader.scratch.capacity);
	return refusal->status;
}
