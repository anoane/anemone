/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Stateless Responses requests become ordinary Chat messages before rendering. No opaque state is imported. */
#include "server_responses.h"

#include "anemone/hooks.h"

#include <stdlib.h>
#include <string.h>

#define SCRATCH_FACTOR 4u
#define SCRATCH_SLACK 4096u
#define ESCAPE_FACTOR 6u
#define QUOTES 2u
#define BAD HTTP_BAD_REQUEST

struct reader {
	struct json_scratch scratch;
	struct chat_refusal *refusal;
	struct chat_buffer *body;
	struct chat_buffer text, reasoning, calls;
	struct json_value metadata, tool_choice;
	bool assistant, has_message;
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
	if (kind(object, '{')) {
		for (size_t cursor = object.begin;
		     json_next(object, &cursor, &key, &value);) {
			if (word(reader, key, name)) {
				return value;
			}
		}
	}
	return (struct json_value){ 0 };
}

static void text(struct chat_buffer *buffer, const char *bytes)
{
	chat_buffer_write(buffer, bytes, strlen(bytes));
}

static void raw(struct chat_buffer *buffer, struct json_value value)
{
	if (value.text != NULL) {
		chat_buffer_write(buffer, value.text + value.begin,
				  value.end - value.begin);
	} else {
		text(buffer, "null");
	}
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

static void refuse(struct reader *reader, const char *message)
{
	if (reader->refusal->status == 0) {
		reader->refusal->status = BAD;
		reader->refusal->message = message;
	}
}

static void unescaped(struct reader *reader, struct chat_buffer *target,
		      struct json_value value)
{
	if (!kind(value, '"')) {
		refuse(reader,
		       "Responses text, function names, call IDs and arguments must be strings");
		return;
	}
	const size_t saved = reader->scratch.used;
	const struct json_bytes bytes = json_unescape(&reader->scratch, value);
	chat_buffer_write(target, bytes.data, bytes.count);
	reader->scratch.used = saved;
}

/* Content parts retain Chat's blank-line separators. Images and hosted tool items require a separate adapter. */
static void content(struct reader *reader, struct chat_buffer *target,
		    struct json_value value, const char *part_type)
{
	if (kind(value, '"')) {
		unescaped(reader, target, value);
	} else if (kind(value, '[')) {
		struct json_value part = { 0 };
		size_t count = 0;
		for (size_t cursor = value.begin;
		     json_next(value, &cursor, NULL, &part); count++) {
			const struct json_value type =
				member(reader, part, "type");
			const bool refusal = word(reader, type, "refusal");
			if (!word(reader, type, part_type) &&
			    !(strcmp(part_type, "output_text") == 0 &&
			      word(reader, type, "input_text")) &&
			    !refusal) {
				refuse(reader,
				       "this Responses route accepts text content parts only");
				return;
			}
			text(target, count == 0 ? "" : "\n\n");
			unescaped(reader, target,
				  member(reader, part,
					 refusal ? "refusal" : "text"));
		}
	} else if (value.text != NULL && !kind(value, 'n')) {
		refuse(reader,
		       "Responses message content must be a string or text-parts array");
	}
}

/* The whole input is already validated JSON. Preserve string escapes and avoid copying long tool results twice. */
static void input_content_json(struct reader *reader, struct json_value value)
{
	if (kind(value, '"')) {
		raw(reader->body, value);
		return;
	}
	struct chat_buffer flattened = { 0 };
	content(reader, &flattened, value, "input_text");
	string(reader->body, flattened.bytes, flattened.count);
	reader->body->is_short = reader->body->is_short || flattened.is_short;
	chat_buffer_free(&flattened);
}

static void message_start(struct reader *reader)
{
	text(reader->body, reader->has_message ? "," : "");
	reader->has_message = true;
}

static void assistant_flush(struct reader *reader)
{
	if (!reader->assistant) {
		return;
	}
	message_start(reader);
	text(reader->body, "{\"role\":\"assistant\",\"content\":");
	string(reader->body, reader->text.bytes, reader->text.count);
	if (reader->reasoning.count != 0) {
		text(reader->body, ",\"reasoning_content\":");
		string(reader->body, reader->reasoning.bytes,
		       reader->reasoning.count);
	}
	if (reader->calls.count != 0) {
		text(reader->body, ",\"tool_calls\":[");
		chat_buffer_write(reader->body, reader->calls.bytes,
				  reader->calls.count);
		text(reader->body, "]");
	}
	text(reader->body, "}");
	reader->text.count = reader->reasoning.count = reader->calls.count = 0;
	reader->assistant = false;
}

static void qualified_name(struct reader *reader, struct chat_buffer *target,
			   struct json_value name, struct json_value space)
{
	struct chat_buffer qualified = { 0 };
	if (space.text != NULL && !kind(space, 'n')) {
		unescaped(reader, &qualified, space);
		if (qualified.count == 0) {
			refuse(reader, "a function namespace must be nonempty");
		}
		text(&qualified, "::");
	}
	unescaped(reader, &qualified, name);
	string(target, qualified.bytes, qualified.count);
	target->is_short = target->is_short || qualified.is_short;
	chat_buffer_free(&qualified);
}

static void input_item(struct reader *reader, struct json_value item)
{
	const struct json_value type = member(reader, item, "type"),
				role = member(reader, item, "role");
	if (!kind(item, '{')) {
		refuse(reader, "each Responses input item must be an object");
	} else if (type.text == NULL || word(reader, type, "message")) {
		const bool assistant = word(reader, role, "assistant");
		if (!assistant && !word(reader, role, "user") &&
		    !word(reader, role, "system") &&
		    !word(reader, role, "developer")) {
			refuse(reader,
			       "a Responses message role must be user, assistant, system or developer");
			return;
		}
		if (assistant) {
			reader->assistant = true;
			content(reader, &reader->text,
				member(reader, item, "content"), "output_text");
		} else {
			assistant_flush(reader);
			message_start(reader);
			text(reader->body, "{\"role\":");
			raw(reader->body, role);
			text(reader->body, ",\"content\":");
			input_content_json(reader,
					   member(reader, item, "content"));
			text(reader->body, "}");
		}
	} else if (word(reader, type, "reasoning")) {
		if (json_is_truthy(&reader->scratch,
				   member(reader, item, "encrypted_content"))) {
			refuse(reader,
			       "encrypted reasoning cannot be restored; resend plaintext history");
			return;
		}
		reader->assistant = true;
		const struct json_value summary =
			member(reader, item, "summary");
		const struct json_value full = member(reader, item, "content");
		content(reader, &reader->reasoning,
			json_is_truthy(&reader->scratch, full) ? full : summary,
			json_is_truthy(&reader->scratch, full) ?
				"reasoning_text" :
				"summary_text");
	} else if (word(reader, type, "function_call")) {
		reader->assistant = true;
		text(&reader->calls,
		     reader->calls.count != 0 ? ",{\"id\":" : "{\"id\":");
		const struct json_value call_id =
			member(reader, item, "call_id");
		if (!kind(call_id, '"')) {
			refuse(reader,
			       "a function_call needs its call_id string");
		}
		raw(&reader->calls, call_id);
		text(&reader->calls,
		     ",\"type\":\"function\",\"function\":{\"name\":");
		qualified_name(reader, &reader->calls,
			       member(reader, item, "name"),
			       member(reader, item, "namespace"));
		text(&reader->calls, ",\"arguments\":");
		const struct json_value arguments =
			member(reader, item, "arguments");
		if (!kind(arguments, '"')) {
			refuse(reader,
			       "function_call.arguments must be a JSON string");
		}
		raw(&reader->calls, arguments);
		text(&reader->calls, "}}");
	} else if (word(reader, type, "function_call_output")) {
		assistant_flush(reader);
		const struct json_value call_id =
			member(reader, item, "call_id");
		if (!kind(call_id, '"')) {
			refuse(reader,
			       "a function_call_output needs its call_id string");
		}
		message_start(reader);
		text(reader->body, "{\"role\":\"tool\",\"tool_call_id\":");
		raw(reader->body, call_id);
		text(reader->body, ",\"content\":");
		input_content_json(reader, member(reader, item, "output"));
		text(reader->body, "}");
	} else {
		refuse(reader,
		       "unsupported Responses input item; resend full plaintext messages and function calls");
	}
}

static void tools(struct reader *reader, struct json_value array,
		  struct json_value space, bool *first)
{
	if (!kind(array, '[')) {
		refuse(reader, "Responses tools must be an array");
		return;
	}
	struct json_value tool = { 0 };
	for (size_t cursor = array.begin;
	     json_next(array, &cursor, NULL, &tool);) {
		const struct json_value type = member(reader, tool, "type");
		if (word(reader, type, "namespace") && space.text == NULL) {
			const struct json_value name =
				member(reader, tool, "name");
			if (!kind(name, '"')) {
				refuse(reader,
				       "a tool namespace needs its name string");
			}
			tools(reader, member(reader, tool, "tools"), name,
			      first);
		} else if (word(reader, type, "function")) {
			text(reader->body,
			     *first ? "{\"type\":\"function\",\"function\":{" :
				      ",{\"type\":\"function\",\"function\":{");
			*first = false;
			bool comma = false;
			struct json_value key = { 0 }, value = { 0 };
			for (size_t at = tool.begin;
			     json_next(tool, &at, &key, &value);) {
				if (word(reader, key, "type")) {
					continue;
				}
				text(reader->body, comma ? "," : "");
				comma = true;
				raw(reader->body, key);
				text(reader->body, ":");
				if (word(reader, key, "name")) {
					qualified_name(reader, reader->body,
						       value, space);
				} else {
					raw(reader->body, value);
				}
			}
			text(reader->body, "}}");
		} else {
			refuse(reader,
			       "this Responses route supports function tools and one-level function namespaces only");
		}
	}
}

static void copy_setting(struct reader *reader, const char *name,
			 struct json_value value)
{
	text(reader->body, ",\"");
	text(reader->body, name);
	text(reader->body, "\":");
	raw(reader->body, value);
}

/* Nested generation controls have the same fail-closed policy as Chat's top-level controls. */
static void known_fields(struct reader *reader, struct json_value object,
			 const char *first, const char *second)
{
	struct json_value key = { 0 }, value = { 0 };
	for (size_t at = object.begin;
	     kind(object, '{') && json_next(object, &at, &key, &value);) {
		if (!word(reader, key, first) &&
		    (second == NULL || !word(reader, key, second)) &&
		    json_is_truthy(&reader->scratch, value)) {
			refuse(reader,
			       "unsupported nested Responses generation control");
		}
	}
}

/* json.h accepts Python's nonfinite literals; reflected fields must remain strict JSON on the wire. */
static bool standard_json(struct json_value value)
{
	if (value.text == NULL) {
		return true;
	}
	if (kind(value, 'N') || kind(value, 'I') ||
	    (kind(value, '-') && value.begin + 1u < value.end &&
	     value.text[value.begin + 1u] == 'I')) {
		return false;
	}
	if (kind(value, '[') || kind(value, '{')) {
		struct json_value child = { 0 };
		for (size_t at = value.begin;
		     json_next(value, &at, NULL, &child);) {
			if (!standard_json(child)) {
				return false;
			}
		}
	}
	return true;
}

static void settings(struct reader *reader, struct json_value root)
{
	static const char *const unchanged[] = {
		"model",	"stream",      "max_tokens",
		"temperature",	"top_p",       "seed",
		"stop",		"tool_choice", "parallel_tool_calls",
		"service_tier", "thinking",    "anemone_draft_usage"
	};
	static const char *const identity[] = {
		"user",	      "user_id",
		"metadata",   "prompt_cache_key",
		"cache_salt", "safety_identifier",
		"session_id", "conversation_id"
	};
	struct json_value key = { 0 }, value = { 0 };
	for (size_t cursor = root.begin;
	     json_next(root, &cursor, &key, &value);) {
		if ((word(reader, key, "metadata") ||
		     word(reader, key, "tools")) &&
		    !standard_json(value)) {
			refuse(reader,
			       "metadata and tool definitions require standard JSON numbers, not NaN or Infinity");
			reader->refusal->member_begin = key.begin;
			reader->refusal->member_end = key.end;
			return;
		}
		if (word(reader, key, "metadata")) {
			reader->metadata = value;
		}
		if (word(reader, key, "tool_choice")) {
			reader->tool_choice = value;
		}
		bool handled = false;
		for (size_t index = 0;
		     index < sizeof(unchanged) / sizeof(unchanged[0]);
		     index++) {
			if (word(reader, key, unchanged[index])) {
				copy_setting(reader, unchanged[index], value);
				handled = true;
			}
		}
		for (size_t index = 0;
		     index < sizeof(identity) / sizeof(identity[0]); index++) {
			handled = handled || word(reader, key, identity[index]);
		}
		if (handled || word(reader, key, "input") ||
		    word(reader, key, "instructions") ||
		    word(reader, key, "tools")) {
			continue;
		}
		reader->refusal->member_begin = key.begin;
		reader->refusal->member_end = key.end;
		if (word(reader, key, "store")) {
			if (!kind(value, 'f') && !kind(value, 'n')) {
				refuse(reader,
				       "stateless Responses requires store:false; durable response chains are "
				       "unavailable");
			}
		} else if (word(reader, key, "previous_response_id") ||
			   word(reader, key, "conversation")) {
			if (!kind(value, 'n')) {
				refuse(reader,
				       "stateless Responses cannot restore previous_response_id or conversatio"
				       "n; resend full input");
			}
		} else if (word(reader, key, "max_output_tokens")) {
			copy_setting(reader, "max_completion_tokens", value);
		} else if (word(reader, key, "reasoning")) {
			if (!kind(value, 'n') && !kind(value, '{')) {
				refuse(reader,
				       "reasoning must be an object or null");
			}
			known_fields(reader, value, "effort", "summary");
			const struct json_value summary =
				member(reader, value, "summary");
			if (summary.text != NULL && !kind(summary, 'n') &&
			    !word(reader, summary, "auto")) {
				refuse(reader,
				       "only automatic plaintext reasoning is available; use reasoning.summary:auto");
			}
			const struct json_value effort =
				member(reader, value, "effort");
			if (effort.text != NULL) {
				copy_setting(reader, "reasoning_effort",
					     effort);
			}
		} else if (word(reader, key, "text")) {
			const struct json_value format =
				member(reader, value, "format");
			if (!kind(value, 'n') && !kind(value, '{')) {
				refuse(reader,
				       "text must be an object or null");
			}
			known_fields(reader, value, "format", "verbosity");
			const struct json_value verbosity =
				member(reader, value, "verbosity");
			if (verbosity.text != NULL && !kind(verbosity, 'n') &&
			    !word(reader, verbosity, "medium")) {
				refuse(reader,
				       "text.verbosity supports the medium default only");
			}
			if (format.text != NULL) {
				copy_setting(reader, "response_format", format);
			}
		} else if (word(reader, key, "include")) {
			struct json_value entry = { 0 };
			if (!kind(value, '[') && !kind(value, 'n')) {
				refuse(reader, "include must be an array");
			}
			for (size_t at = value.begin;
			     kind(value, '[') &&
			     json_next(value, &at, NULL, &entry);) {
				if (!word(reader, entry,
					  "reasoning.encrypted_content")) {
					refuse(reader,
					       "only reasoning.encrypted_content is accepted; plaintext reason"
					       "ing is returned");
				}
			}
		} else if (word(reader, key, "stream_options") ||
			   word(reader, key, "client_metadata")) {
			if (!kind(value, 'n') && !kind(value, '{')) {
				refuse(reader,
				       "Responses stream_options and client_metadata must be objects");
			}
			if (word(reader, key, "stream_options")) {
				const struct json_value obfuscate = member(
					reader, value, "include_obfuscation");
				known_fields(reader, value,
					     "include_obfuscation",
					     "include_usage");
				const struct json_value usage =
					member(reader, value, "include_usage");
				if (usage.text != NULL && !kind(usage, 'n') &&
				    !kind(usage, 't') && !kind(usage, 'f')) {
					refuse(reader,
					       "stream_options.include_usage must be boolean");
				}
				if (obfuscate.text != NULL &&
				    !kind(obfuscate, 'n') &&
				    !kind(obfuscate, 'f')) {
					refuse(reader,
					       "stream obfuscation is unavailable; set include_obfuscation:false");
				}
			}
		} else if (word(reader, key, "truncation")) {
			if (!kind(value, 'n') &&
			    !word(reader, value, "disabled")) {
				refuse(reader,
				       "automatic truncation is unavailable; resend a bounded full history");
			}
		} else if (json_is_truthy(&reader->scratch, value)) {
			refuse(reader, "unsupported Responses parameter");
		}
		if (reader->refusal->status != 0) {
			return;
		}
	}
	reader->refusal->member_begin = reader->refusal->member_end = 0;
}

uint32_t server_responses_read_fields(const char *body, size_t count,
				      const char *model, const char *alias,
				      const struct chat_request *defaults,
				      struct chat_request *request,
				      struct chat_buffer *chat_body,
				      struct chat_buffer *fields,
				      struct chat_refusal *refusal)
{
	struct json_value root = { 0 };
	*refusal = (struct chat_refusal){ 0 };
	chat_body->count = chat_body->sent = 0;
	chat_body->is_short = false;
	if (fields != NULL) {
		fields->count = fields->sent = 0;
		fields->is_short = false;
	}
	if (body == NULL || count > CHAT_MOST_BODY_BYTES ||
	    json_read_document(body, count, &root) != JSON_DOCUMENT_VALID ||
	    !kind(root, '{')) {
		refusal->status = count > CHAT_MOST_BODY_BYTES ?
					  HTTP_CONTENT_TOO_LARGE :
					  BAD;
		return refusal->status;
	}
	struct reader reader = { .scratch = { .capacity =
						      count * SCRATCH_FACTOR +
						      SCRATCH_SLACK },
				 .refusal = refusal,
				 .body = chat_body };
	reader.scratch.bytes = FAULT_POINT(chat_memory) ?
				       NULL :
				       malloc(reader.scratch.capacity);
	if (reader.scratch.bytes == NULL) {
		refusal->status = HTTP_SERVER_ERROR;
		return refusal->status;
	}
	if (json_has_repeated_key(&reader.scratch, root)) {
		refuse(&reader, "duplicate JSON object keys are refused");
	}
	text(chat_body, "{\"messages\":[");
	const struct json_value instructions =
		member(&reader, root, "instructions");
	if (instructions.text != NULL && !kind(instructions, 'n')) {
		if (!kind(instructions, '"')) {
			refuse(&reader, "instructions must be a string");
		}
		message_start(&reader);
		text(chat_body, "{\"role\":\"system\",\"content\":");
		raw(chat_body, instructions);
		text(chat_body, "}");
	}
	const struct json_value input = member(&reader, root, "input");
	if (kind(input, '"')) {
		message_start(&reader);
		text(chat_body, "{\"role\":\"user\",\"content\":");
		raw(chat_body, input);
		text(chat_body, "}");
	} else if (kind(input, '[')) {
		struct json_value item = { 0 };
		for (size_t at = input.begin;
		     refusal->status == 0 &&
		     json_next(input, &at, NULL, &item);) {
			input_item(&reader, item);
		}
	} else {
		refuse(&reader,
		       "stateless Responses needs full input as a string or item array");
	}
	assistant_flush(&reader);
	text(chat_body, "]");
	const struct json_value declared = member(&reader, root, "tools");
	if (declared.text != NULL && !kind(declared, 'n')) {
		bool first = true;
		text(chat_body, ",\"tools\":[");
		tools(&reader, declared, (struct json_value){ 0 }, &first);
		text(chat_body, "]");
	}
	settings(&reader, root);
	text(chat_body, "}");
	if (fields != NULL && refusal->status == 0) {
		text(fields, ",\"instructions\":");
		raw(fields, instructions);
		text(fields, ",\"tools\":");
		if (declared.text == NULL || kind(declared, 'n')) {
			text(fields, "[]");
		} else {
			raw(fields, declared);
		}
		text(fields, ",\"metadata\":");
		if (reader.metadata.text == NULL ||
		    kind(reader.metadata, 'n')) {
			text(fields, "{}");
		} else {
			raw(fields, reader.metadata);
		}
		text(fields, ",\"tool_choice\":");
		if (reader.tool_choice.text == NULL ||
		    kind(reader.tool_choice, 'n')) {
			text(fields, "\"auto\"");
		} else {
			raw(fields, reader.tool_choice);
		}
	}
	if ((fields != NULL && fields->is_short) || reader.scratch.is_short ||
	    chat_body->is_short || reader.text.is_short ||
	    reader.reasoning.is_short || reader.calls.is_short) {
		refusal->status = HTTP_SERVER_ERROR;
	} else if (chat_body->count > CHAT_MOST_BODY_BYTES) {
		refusal->status = HTTP_CONTENT_TOO_LARGE;
	} else if (refusal->status == 0) {
		const uint32_t status = chat_request_read(
			chat_body->bytes, chat_body->count, model, alias,
			defaults, request, refusal);
		if (status != 0 &&
		    refusal->member_end > refusal->member_begin) {
			struct json_value key = { chat_body->bytes,
						  refusal->member_begin,
						  refusal->member_end };
			const char *mapped =
				word(&reader, key, "messages") ?
					"input" :
				word(&reader, key, "max_completion_tokens") ?
					"max_output_tokens" :
				word(&reader, key, "response_format") ?
					"text" :
				word(&reader, key, "reasoning_effort") ?
					"reasoning" :
					NULL;
			struct json_value original_key = { 0 }, value = { 0 };
			refusal->member_begin = refusal->member_end = 0;
			for (size_t at = root.begin;
			     json_next(root, &at, &original_key, &value);) {
				const size_t saved = reader.scratch.used;
				const struct json_bytes name =
					json_unescape(&reader.scratch, key);
				const struct json_bytes candidate =
					json_unescape(&reader.scratch,
						      original_key);
				const bool equal =
					mapped != NULL ?
						word(&reader, original_key,
						     mapped) :
						name.count == candidate.count &&
							memcmp(name.data,
							       candidate.data,
							       name.count) == 0;
				reader.scratch.used = saved;
				if (equal) {
					refusal->member_begin =
						original_key.begin;
					refusal->member_end = original_key.end;
					break;
				}
			}
		}
		request->is_responses = status == 0;
	}
	chat_buffer_free(&reader.text);
	chat_buffer_free(&reader.reasoning);
	chat_buffer_free(&reader.calls);
	explicit_bzero(reader.scratch.bytes, reader.scratch.capacity);
	free(reader.scratch.bytes);
	return refusal->status;
}

uint32_t server_responses_read(const char *body, size_t count,
			       const char *model, const char *alias,
			       const struct chat_request *defaults,
			       struct chat_request *request,
			       struct chat_buffer *chat_body,
			       struct chat_refusal *refusal)
{
	return server_responses_read_fields(body, count, model, alias, defaults,
					    request, chat_body, NULL, refusal);
}
