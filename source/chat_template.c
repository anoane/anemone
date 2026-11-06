/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * chat_template.c: a conversation as the model reads it. Contract: chat_template.h.
 *
 * The input is checked whole first, as json.loads reads it (json.h); later readings trust the JSON and check only
 * what the encoder checks. Nothing is copied: an entry (a message after the encoder's merges) is a run of elements of
 * the messages array, and members are looked up when they are needed. A string the template looks into (a member's
 * name, an id, a call's arguments) is unescaped into the scratch memory, used and given back; content and reasoning
 * are unescaped straight into the prompt. The ids of the last assistant message that made calls are the exception:
 * unescaped and sorted once, they stay for the tool results after it. The first refusal on the way is kept, and the
 * prompt is then not used.
 */
#include "anemone/chat_template.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "json.h"

#define DSML "｜DSML｜"
#define BEGIN_OF_SENTENCE "<｜begin▁of▁sentence｜>"
#define END_OF_SENTENCE "<｜end▁of▁sentence｜>"
#define TOOL_RESULT_BEGIN "<tool_result>"
#define TOOL_RESULT_END "</tool_result>"
#define THINKING_BEGIN "<think>"
#define THINKING_END "</think>"
#define IMAGE_PLACEHOLDER "<｜deepseek_image｜>"
#define PARAGRAPH "\n\n"
#define NONE_TEXT "None"                          /* what Python's str.format writes for None */
#define EFFORT_BEGIN "Reasoning Effort: "
#define EFFORT_END " (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
#define RESPONSE_FORMAT_BEGIN "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
#define TOOLS_BEGIN "\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can " \
    "invoke tools by writing a \"<" DSML " calls>\" block like the following:\n\n<" DSML " calls>\n<" DSML         \
    " invoke name=\"$TOOL_NAME\">\n<" DSML " parameter name=\"$PARAMETER_NAME\" string=\"true|false\">"           \
    "$PARAMETER_VALUE</" DSML " parameter>\n...\n</" DSML " invoke>\n<" DSML " invoke name=\"$TOOL_NAME2\">\n...\n" \
    "</" DSML " invoke>\n</" DSML " calls>\n\nString parameters should be specified as is and set `string=\"true\"`" \
    ". For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set "           \
    "`string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by " THINKING_BEGIN "), you MUST output your "  \
    "complete reasoning inside " THINKING_BEGIN "..." THINKING_END " BEFORE any tool calls or final response.\n\n"  \
    "Otherwise, output directly after " THINKING_END " with tool calls or final response.\n\n### Available Tool "   \
    "Schemas\n\n"
#define TOOLS_END "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool " \
    "calls.\n"
#define CALLS_BEGIN "\n\n<" DSML " calls>\n"
#define CALLS_END "\n</" DSML " calls>"
#define INVOKE_BEGIN "<" DSML " invoke name=\""
#define INVOKE_NAME_END "\">\n"
#define INVOKE_END "\n</" DSML " invoke>"
#define PARAMETER_BEGIN "<" DSML " parameter name=\""
#define PARAMETER_END "</" DSML " parameter>"
#define FALLBACK_PARAMETER "arguments"            /* the one parameter of arguments that are no object */
#define NAMESPACE_SEPARATOR "::"
#define SCRATCH_BYTES_PER_INPUT_BYTE 5            /* a call's arguments unescaped twice, and their keys sorted */
#define SCRATCH_SLACK_BYTES 4096                  /* a number's terminating zero, the keys of objects deep inside */
#define ARGUMENT_DECODINGS 2                      /* arguments are JSON, or JSON of a string that is JSON */
#define EFFORT_DIGITS_CAPACITY 16
#define FIRST_TRUSTED_SPANS 16u
#define TRUSTED_SPANS_GROWTH 2u

enum role { ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL, ROLE_LATEST_REMINDER, ROLE_COUNT };
static const char *const role_names[ROLE_COUNT] = {"system", "user", "assistant", "tool", "latest_reminder"};
static const char *const role_tokens[ROLE_COUNT] = {"<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "",
                                                     "<｜latest_reminder｜>"};
enum task { TASK_NONE, TASK_ACTION, TASK_QUERY, TASK_AUTHORITY, TASK_DOMAIN, TASK_TITLE, TASK_READ_URL, TASK_COUNT };
static const char *const task_names[TASK_COUNT] = {"", "action", "query", "authority", "domain", "title", "read_url"};
static const char *const task_tokens[TASK_COUNT] = {"", "<｜action｜>", "<｜query｜>", "<｜authority｜>",
                                                     "<｜domain｜>", "<｜title｜>", "<｜read_url｜>"};
/* The members the template reads, of any object: a message, a tool, a call, its function, a content part. */
enum field { FIELD_ROLE, FIELD_CONTENT, FIELD_REASONING, FIELD_TOOL_CALLS, FIELD_TOOL_CALL_ID, FIELD_TOOLS,
             FIELD_RESPONSE_FORMAT, FIELD_TASK, FIELD_CONTENT_BLOCKS, FIELD_WITHOUT_END_OF_SENTENCE,
             FIELD_FUNCTION, FIELD_NAMESPACE, FIELD_NAME, FIELD_ARGUMENTS, FIELD_IDENTIFIER, FIELD_TYPE, FIELD_TEXT,
             FIELD_IMAGE_URL, FIELD_URL, FIELD_COUNT };
static const char *const field_names[FIELD_COUNT] = {"role", "content", "reasoning_content", "tool_calls",
    "tool_call_id", "tools", "response_format", "task", "content_blocks", "wo_eos", "function", "namespace", "name",
    "arguments", "id", "type", "text", "image_url", "url"};

struct entry {                                    /* a message as the encoder renders it, after its merges */
    struct json_value first;                      /* the first message of the array merged into it */
    size_t cursor;                                /* where the reading of the array stood before it */
    uint32_t message_count;
    uint32_t role;                                /* enum role: ROLE_USER for a run of user and tool messages */
    uint32_t task;
    bool has_own_fields;                          /* false when a tool message began it: the encoder made it */
    bool has_tools;
    bool has_blocks;                              /* a user message of it gives a text or image block */
    bool has_tool_results;
};

struct tool_result {
    struct json_value message;
    uint32_t key;                                 /* the place of its call */
    uint32_t position;                            /* its place among the results: the sort is stable */
};

struct call_identifier {
    struct json_bytes identifier;                 /* first, for bsearch: unescaped, in the scratch memory kept */
    uint32_t call_index;
};

struct context {
	const struct chat_template_options *options;
	struct json_writer *writer;
	struct json_scratch scratch;
	struct json_value messages;
	struct json_value
		tools; /* the request's tools, on the first system entry */
	struct entry *entries;
	uint32_t entry_count;
	int64_t last_user_index; /* -1 when no message counts as the user's */
	bool drops_thinking; /* the option, unless a message has tools */
	bool is_checking_only; /* tools or calls the encoder reads but does not write */
	struct tool_result
		*tool_results; /* CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS of each */
	struct call_identifier *
		call_identifiers; /* those of the last assistant message that made calls, sorted */
	uint32_t identifier_count;
	size_t kept_scratch; /* the scratch memory their bytes hold */
	const char *refusal; /* the first thing refused; NULL while none is */
	struct chat_template_spans *trusted;
	uint32_t trusted_capacity;
	result_status trusted_status;
};

static void refuse(struct context *context, const char *reason)
{
    context->refusal = context->refusal != NULL ? context->refusal : reason;
}

static void write_template_bytes(struct context *context, const char *text, size_t length)
{
    const size_t begin = context->writer->length;
    json_write_bytes(context->writer, text, length);
    struct chat_template_spans *const trusted = context->trusted;
    if (trusted == NULL || length == 0 || context->is_checking_only ||
	context->trusted_status != STATUS_OK) {
	    return;
    }
    if (context->writer->length > TOKENIZER_MAXIMUM_TEXT_BYTES) {
	    context->trusted_status = STATUS_UNSUPPORTED;
	    return;
    }
    if (trusted->count > 0 && trusted->spans[trusted->count - 1u].end == begin) {
        trusted->spans[trusted->count - 1u].end = (uint32_t)context->writer->length;
        return;
    }
    if (trusted->count == context->trusted_capacity) {
        const uint32_t capacity = context->trusted_capacity == 0 ? FIRST_TRUSTED_SPANS
                                  : context->trusted_capacity * TRUSTED_SPANS_GROWTH;
	struct tokenizer_span *const spans =
		FAULT_POINT(chat_template_memory) ?
			NULL :
			realloc(trusted->spans,
				(size_t)capacity * sizeof(*spans));
	if (spans == NULL) {
		context->trusted_status = STATUS_OUT_OF_MEMORY;
		return;
        }
        trusted->spans = spans;
        context->trusted_capacity = capacity;
    }
    trusted->spans[trusted->count++] =
	    (struct tokenizer_span){ (uint32_t)begin,
				     (uint32_t)context->writer->length };
}

static void write_text(struct context *context, const char *text)
{
    write_template_bytes(context, text, strlen(text));
}

static bool is_kind(struct json_value value, char first)
{
    return value.text != NULL && value.text[value.begin] == first;
}

static bool is_truthy(struct context *context, struct json_value value)
{
	return json_is_truthy(&context->scratch, value);
}

static bool is_string_of(struct context *context, struct json_value value, const char *word)
{
	return json_string_is(&context->scratch, value, word);
}

/* The last member of each field's name in an object; text NULL for what it lacks, and for all when it is none. */
static void read_fields(struct context *context, struct json_value object, struct json_value *fields)
{
    struct json_value key = {0}, value = {0};
    memset(fields, 0, FIELD_COUNT * sizeof(fields[0]));
    for (size_t cursor = object.begin;
	 is_kind(object, '{') && json_next(object, &cursor, &key, &value);) {
	    const size_t mark = context->scratch.used;
	    const struct json_bytes name =
		    json_unescape(&context->scratch, key);
	    context->scratch.used = mark;
	    for (size_t field = 0; field < FIELD_COUNT; field++) {
		    const bool is_field =
			    name.count == strlen(field_names[field]) &&
			    memcmp(name.data, field_names[field], name.count) ==
				    0;
		    fields[field] = is_field ? value : fields[field];
	    }
    }
}

/* Whether a string holds a text, or holds it twice: "::" twice in a name is refused by the encoder, which takes
 * what comes before the first one as a namespace and allows no other in the rest. */
static bool holds(struct context *context, struct json_value string, const char *text, bool is_twice)
{
    const size_t mark = context->scratch.used;
    const struct json_bytes bytes = json_unescape(&context->scratch, string);
    const char *const found = memmem(bytes.data, bytes.count, text, strlen(text));
    const size_t rest = found == NULL ? 0 : bytes.count - (size_t)(found - bytes.data) - strlen(text);
    context->scratch.used = mark;
    return found != NULL && (!is_twice || memmem(found + strlen(text), rest, text, strlen(text)) != NULL);
}

/* A value as the encoder's to_json writes it; an object with a key twice is refused (the contract says why). */
static void write_json(struct context *context, struct json_value value)
{
	if (json_has_repeated_key(&context->scratch, value)) {
		refuse(context, "an object written as JSON holds no key twice");
	}
	json_write_canonical(context->writer, &context->scratch, value);
}

/* A tool's function, as the encoder's to_json writes it; namespaces are refused (the contract says why). */
static void write_function(struct context *context, struct json_value tool)
{
    struct json_value fields[FIELD_COUNT] = {{0}}, function[FIELD_COUNT] = {{0}};
    read_fields(context, tool, fields);
    read_fields(context, fields[FIELD_FUNCTION], function);
    if (!is_kind(fields[FIELD_FUNCTION], '{') || function[FIELD_NAMESPACE].text != NULL ||
        (fields[FIELD_NAMESPACE].text != NULL && !is_kind(fields[FIELD_NAMESPACE], 'n')) ||
        !is_kind(function[FIELD_NAME], '"') || holds(context, function[FIELD_NAME], NAMESPACE_SEPARATOR, true)) {
        refuse(context, "a tool is an object whose function has a name, without a namespace");
    } else if (!context->is_checking_only) {
        write_json(context, fields[FIELD_FUNCTION]);
    }
}

static void write_tools(struct context *context, struct json_value tools)
{
    struct json_value tool = {0};
    if (!is_kind(tools, '[')) {
        refuse(context, "tools are an array");
        return;
    }
    write_text(context, TOOLS_BEGIN);
    for (size_t cursor = tools.begin, count = 0;
	 json_next(tools, &cursor, NULL, &tool); count++) {
	    write_text(context, count > 0 ? "\n" : "");
	    write_function(context, tool);
    }
    write_text(context, TOOLS_END);
}

/* One DSML parameter: a string as it is, any other value as JSON; key NULL for the fallback parameter. */
static void write_parameter(struct context *context, const struct json_value *key, struct json_value value)
{
    write_text(context, PARAMETER_BEGIN);
    if (key != NULL) {
	    json_write_unescaped(context->writer, *key);
    }
    write_text(context, key != NULL ? "" : FALLBACK_PARAMETER);
    write_text(context, is_kind(value, '"') ? "\" string=\"true\">" : "\" string=\"false\">");
    if (is_kind(value, '"')) {
	    json_write_unescaped(context->writer, value);
    } else {
	    json_write_canonical(context->writer, &context->scratch, value);
    }
    write_text(context, PARAMETER_END);
}

/* A call's arguments as the encoder's encode_arguments_to_dsml writes them: an object, or a string that holds one
 * as JSON (or JSON of a string that does), member after member; anything else as one parameter. What json.loads
 * reads but json.h does not is refused where the encoder would decode it again or write it. */
static void write_arguments(struct context *context, struct json_value arguments)
{
    const size_t mark = context->scratch.used;
    struct json_value object = is_kind(arguments, '{') ? arguments : (struct json_value){0}, source = arguments;
    struct json_value key = {0}, item = {0};
    for (unsigned decoding = 0; object.text == NULL && decoding < ARGUMENT_DECODINGS && is_kind(source, '"');
         decoding++) {
	    const struct json_bytes text =
		    json_unescape(&context->scratch, source);
	    const enum json_document document =
		    json_read_document(text.data, text.count, &source);
	    if (document == JSON_DOCUMENT_BEYOND_THIS_READER &&
		(is_kind(source, '{') || is_kind(source, '"'))) {
		    refuse(context,
			   "arguments are JSON json.h reads: no deeper than its limit, no lone half of a pair");
	    }
        if (document != JSON_DOCUMENT_VALID) {
            break;
        }
        object = is_kind(source, '{') ? source : object;
    }
    if (object.text != NULL &&
	json_has_repeated_key(&context->scratch, object)) {
	    refuse(context, "an object written as JSON holds no key twice");
    }
    if (object.text == NULL) {
        write_parameter(context, NULL, arguments);
    }
    for (size_t cursor = object.begin, count = 0;
	 object.text != NULL && json_next(object, &cursor, &key, &item);
	 count++) {
	    write_text(context, count > 0 ? "\n" : "");
	    write_parameter(context, &key, item);
    }
    context->scratch.used = mark;
}

static void write_calls(struct context *context, struct json_value calls)
{
    struct json_value call = {0}, fields[FIELD_COUNT] = {{0}}, function[FIELD_COUNT] = {{0}};
    if (!is_kind(calls, '[')) {
        refuse(context, "tool calls are an array");
        return;
    }
    write_text(context, CALLS_BEGIN);
    for (size_t cursor = calls.begin, count = 0;
	 context->refusal == NULL && json_next(calls, &cursor, NULL, &call);
	 count++) {
	    read_fields(context, call, fields);
	    read_fields(context, fields[FIELD_FUNCTION], function);
	    if (!is_kind(function[FIELD_NAME], '"') ||
		function[FIELD_ARGUMENTS].text == NULL ||
		count == CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS ||
		holds(context, function[FIELD_NAME], NAMESPACE_SEPARATOR,
		      true) ||
		(fields[FIELD_NAMESPACE].text != NULL &&
		 !is_kind(fields[FIELD_NAMESPACE], 'n')) ||
		(function[FIELD_NAMESPACE].text != NULL &&
		 !is_kind(function[FIELD_NAMESPACE], 'n'))) {
		    refuse(context,
			   "a call has a function with a name and arguments and no namespace, calls not too many");
		    return;
	    }
	    write_text(context, count > 0 ? "\n" INVOKE_BEGIN : INVOKE_BEGIN);
	    json_write_unescaped(context->writer, function[FIELD_NAME]);
	    write_text(context, INVOKE_NAME_END);
	    if (!context->is_checking_only) {
		    write_arguments(context, function[FIELD_ARGUMENTS]);
	    }
	    write_text(context, INVOKE_END);
    }
    write_text(context, CALLS_END);
}

/* The parts of a content array, text as it is and an image as its placeholder, a blank line before each but the
 * first of count. */
static void write_parts(struct context *context, struct json_value parts, size_t *count)
{
    struct json_value part = {0}, fields[FIELD_COUNT] = {{0}};
    for (size_t cursor = parts.begin; json_next(parts, &cursor, NULL, &part);
	 (*count)++) {
	    read_fields(context, part, fields);
	    write_text(context, *count > 0 ? PARAGRAPH : "");
	    if (is_string_of(context, fields[FIELD_TYPE], "image_url")) {
		    if (context->trusted != NULL &&
			!context->is_checking_only) {
			    struct chat_template_spans *const trusted =
				    context->trusted;
			    struct json_value image[FIELD_COUNT] = { { 0 } };
			    read_fields(context, fields[FIELD_IMAGE_URL],
					image);
			    const struct json_value url =
				    is_kind(fields[FIELD_IMAGE_URL], '"') ?
					    fields[FIELD_IMAGE_URL] :
					    image[FIELD_URL];
			    if (trusted->image_count == IMAGE_MAX_IMAGES ||
				context->writer->length >
					TOKENIZER_MAXIMUM_TEXT_BYTES) {
				    refuse(context,
					   "the number of images exceeds the request bound");
			    } else {
				    trusted->images[trusted->image_count++] =
					    (struct chat_template_image){
						    url.begin, url.end,
						    (uint32_t)context->writer
							    ->length
					    };
			    }
		    }
		    write_text(context, IMAGE_PLACEHOLDER);
	    } else if (fields[FIELD_TEXT].text != NULL) {
		    json_write_unescaped(context->writer, fields[FIELD_TEXT]);
	    }
    }
}

/* What the encoder writes of a content: a string as it is, an array as its parts; for one that is absent (or an
 * empty array) and one that is null, what it writes then. */
static void write_content(struct context *context, struct json_value content, const char *when_absent,
                          const char *when_null)
{
    size_t count = 0;
    if (is_kind(content, '"')) {
	    json_write_unescaped(context->writer, content);
    } else if (is_truthy(context, content)) {
        write_parts(context, content, &count);
    } else {
        write_text(context, is_kind(content, 'n') ? when_null : when_absent);
    }
}

/* The order of the kept ids: by their bytes, then by the place of their call. */
static int compare_identifiers(const void *first, const void *second)
{
    const struct call_identifier *const left = first, *const right = second;
    const int order = json_compare_bytes(&left->identifier, &right->identifier);
    return order != 0 ? order : (left->call_index > right->call_index) - (left->call_index < right->call_index);
}

static int compare_tool_results(const void *first, const void *second)
{
    const struct tool_result *const left = first, *const right = second;
    return left->key != right->key ? (left->key > right->key) - (left->key < right->key)
                                   : (left->position > right->position) - (left->position < right->position);
}

/* The encoder's last_tool_call_order of an assistant message that made calls: the id of each call (its own, else
 * its function's) that is truthy, unescaped once and kept, sorted, each id once with the place of its last call; an
 * id of another type than a string is refused (the contract says why). The ids of the message before are given back. */
static void keep_call_identifiers(struct context *context, struct json_value calls)
{
    struct json_value call = {0}, fields[FIELD_COUNT] = {{0}}, function[FIELD_COUNT] = {{0}};
    struct call_identifier *const identifiers = context->call_identifiers;
    uint32_t kept = 0;
    context->scratch.used = 0;
    context->identifier_count = 0;
    for (size_t cursor = calls.begin, index = 0;
	 is_kind(calls, '[') && index < CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS &&
	 json_next(calls, &cursor, NULL, &call);
	 index++) {
	    read_fields(context, call, fields);
	    read_fields(context, fields[FIELD_FUNCTION], function);
	    const struct json_value identifier =
		    is_truthy(context, fields[FIELD_IDENTIFIER]) ?
			    fields[FIELD_IDENTIFIER] :
			    function[FIELD_IDENTIFIER];
	    if (is_truthy(context, identifier) && !is_kind(identifier, '"')) {
		    refuse(context, "a call's id is a string");
	    } else if (is_truthy(context, identifier)) {
		    identifiers[context->identifier_count++] =
			    (struct call_identifier){ json_unescape(
							      &context->scratch,
							      identifier),
						      (uint32_t)index };
	    }
    }
    qsort(identifiers, context->identifier_count, sizeof(identifiers[0]), compare_identifiers);
    for (uint32_t index = 0; index < context->identifier_count; index++) {      /* the last of a run of one id */
	    kept -= kept > 0 && json_compare_bytes(&identifiers[kept - 1],
						   &identifiers[index]) == 0 ?
			    1 :
			    0;
	    identifiers[kept++] = identifiers[index];
    }
    context->identifier_count = kept;
    context->kept_scratch = context->scratch.used;
}

/* The encoder's sort_tool_results_by_call_order: the tool results of a merged user message, stably, by the place of
 * the last call of their id among those of the last assistant message that made calls (0 when none has it). */
static void sort_tool_results(struct context *context, uint32_t result_count)
{
    struct json_value fields[FIELD_COUNT] = {{0}};
    for (uint32_t result = 0; context->identifier_count > 0 && result < result_count; result++) {
        read_fields(context, context->tool_results[result].message, fields);
        const size_t mark = context->scratch.used;
	const struct json_bytes identifier =
		is_kind(fields[FIELD_TOOL_CALL_ID], '"') ?
			json_unescape(&context->scratch,
				      fields[FIELD_TOOL_CALL_ID]) :
			(struct json_bytes){ "", 0 };
	const struct call_identifier *const found = bsearch(
		&identifier, context->call_identifiers,
		context->identifier_count, sizeof(context->call_identifiers[0]),
		json_compare_bytes);
	context->tool_results[result].key = found != NULL ? found->call_index : 0;   /* last_tool_call_order.get */
        context->scratch.used = mark;
    }
    qsort(context->tool_results, result_count, sizeof(context->tool_results[0]), compare_tool_results);
}

/* A merged user message: the parts of its messages, a tool message as one tool result in the place of the sort. */
static void write_user(struct context *context, const struct entry *entry)
{
    struct json_value fields[FIELD_COUNT] = {{0}}, message = {0};
    uint32_t result_count = 0, written_results = 0;
    size_t cursor = entry->cursor, count = 0;
    for (uint32_t index = 0;
	 index < entry->message_count &&
	 json_next(context->messages, &cursor, NULL, &message);
	 index++) {
	    read_fields(context, message, fields);
	    const bool is_tool = is_string_of(context, fields[FIELD_ROLE],
					      role_names[ROLE_TOOL]);
	    if (is_tool && result_count == CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS) {
		    refuse(context,
			   "a merged user message holds not too many tool results");
		    return;
	    }
	    if (is_tool) {
		    context->tool_results[result_count] =
			    (struct tool_result){ message, 0, result_count };
		    result_count++;
	    }
    }
    sort_tool_results(context, result_count);
    write_text(context, role_tokens[ROLE_USER]);
    cursor = entry->cursor;
    for (uint32_t index = 0;
	 index < entry->message_count &&
	 json_next(context->messages, &cursor, NULL, &message);
	 index++) {
	    read_fields(context, message, fields);
	    const bool is_tool = is_string_of(context, fields[FIELD_ROLE],
					      role_names[ROLE_TOOL]);
	    if (!is_tool && is_kind(fields[FIELD_CONTENT], '[')) {
		    write_parts(context, fields[FIELD_CONTENT], &count);
		    continue;
	    }
	    write_text(context, count++ > 0 ? PARAGRAPH : "");
	    if (is_tool) {
		    read_fields(
			    context,
			    context->tool_results[written_results++].message,
			    fields);
		    write_text(context, TOOL_RESULT_BEGIN);
	    }
	    write_content(context, fields[FIELD_CONTENT], "",
			  is_tool ? NONE_TEXT : "");
	    write_text(context, is_tool ? TOOL_RESULT_END : "");
    }
}

/* What the encoder checks of a value without writing it: not what it would decode or serialize to write it. */
static void check_only(struct context *context, void (*write)(struct context *, struct json_value),
                       struct json_value value)
{
    struct json_writer *const writer = context->writer;
    struct json_writer ignored = {NULL, 0, 0};
    context->writer = &ignored;
    context->is_checking_only = true;
    write(context, value);
    context->is_checking_only = false;
    context->writer = writer;
}

/* One entry, as the encoder's render_message writes the message at its index. */
static void write_entry(struct context *context, uint32_t index)
{
	const struct chat_template_options *const options = context->options;
	const struct entry *const entry = &context->entries[index];
	struct json_value fields[FIELD_COUNT] = { { 0 } };
	char effort[EFFORT_DIGITS_CAPACITY] = { 0 };
	read_fields(context,
		    entry->has_own_fields ? entry->first :
					    (struct json_value){ 0 },
		    fields);
	fields[FIELD_TOOLS] = index == 0 && entry->has_own_fields &&
					      context->tools.text != NULL ?
				      context->tools :
				      fields[FIELD_TOOLS];
	const bool is_last = index + 1 == context->entry_count,
		   has_calls = is_truthy(context, fields[FIELD_TOOL_CALLS]);
	write_text(context, index == 0 && (options->is_thinking ||
					   entry->role == ROLE_SYSTEM) ?
				    role_tokens[ROLE_SYSTEM] :
				    "");
	if (index == 0 && options->is_thinking) {
		const int length = snprintf(effort, sizeof(effort), "%u",
					    options->reasoning_effort);
		write_text(context, EFFORT_BEGIN);
		write_template_bytes(context, effort,
				     length > 0 ? (size_t)length : 0);
		write_text(context, EFFORT_END);
	}
    if (entry->role != ROLE_SYSTEM && is_truthy(context, fields[FIELD_TOOLS])) {
        check_only(context, write_tools, fields[FIELD_TOOLS]);
    }
    if (entry->role != ROLE_ASSISTANT && has_calls) {
        check_only(context, write_calls, fields[FIELD_TOOL_CALLS]);
    }
    if (entry->role == ROLE_USER) {
        write_user(context, entry);
    } else if (entry->role != ROLE_ASSISTANT) {
        write_text(context, entry->role == ROLE_SYSTEM && index == 0 ? "" : role_tokens[entry->role]);
        write_content(context, fields[FIELD_CONTENT], entry->role == ROLE_SYSTEM ? "" : NONE_TEXT,
                      entry->role == ROLE_SYSTEM ? "" : NONE_TEXT);
        if (entry->role == ROLE_SYSTEM && is_truthy(context, fields[FIELD_TOOLS])) {
            write_tools(context, fields[FIELD_TOOLS]);
        }
        if (entry->role == ROLE_SYSTEM && is_truthy(context, fields[FIELD_RESPONSE_FORMAT])) {
            write_text(context, RESPONSE_FORMAT_BEGIN);
            write_json(context, fields[FIELD_RESPONSE_FORMAT]);
        }
    } else {
        const bool follows_task = index > 0 && context->entries[index - 1].task != TASK_NONE;
        if (options->is_thinking && !follows_task && (!context->drops_thinking || index > context->last_user_index)) {
            write_content(context, fields[FIELD_REASONING], "", "");
            write_text(context, THINKING_END);
        }
        write_content(context, fields[FIELD_CONTENT], "", "");
        if (has_calls) {
            write_calls(context, fields[FIELD_TOOL_CALLS]);
            keep_call_identifiers(context, fields[FIELD_TOOL_CALLS]);
        }
        write_text(context, is_last && options->continues_final_message ? "" : END_OF_SENTENCE);
    }
    const uint32_t next_role = is_last ? ROLE_ASSISTANT : context->entries[index + 1].role;
    const bool opens_answer = entry->role == ROLE_USER || (entry->role == ROLE_SYSTEM && index > 0);
    if ((next_role == ROLE_ASSISTANT || next_role == ROLE_LATEST_REMINDER) &&
        (entry->task != TASK_NONE || opens_answer)) {
        const bool is_plain_task = entry->task != TASK_NONE && entry->task != TASK_ACTION;
        const bool opens_thinking = options->is_thinking && (entry->task == TASK_ACTION || !context->drops_thinking ||
                                                             index >= context->last_user_index);
        write_text(context, is_plain_task ? "" : role_tokens[ROLE_ASSISTANT]);
        write_text(context, is_plain_task ? "" : opens_thinking ? THINKING_BEGIN : THINKING_END);
        write_text(context, task_tokens[entry->task]);
    }
}

/* The index of a string among names, or count when it is none of them. */
static uint32_t index_of(struct context *context, struct json_value value, const char *const *names, uint32_t count)
{
    uint32_t index = 0;
    while (index < count && !is_string_of(context, value, names[index])) {
        index++;
    }
    return index;
}

/* What the encoder checks of a content array: every part an object, its text a string without the image
 * placeholder, an image with its source; any other kind of part is refused (the contract says why). */
static void check_parts(struct context *context, struct json_value parts)
{
    struct json_value part = {0}, fields[FIELD_COUNT] = {{0}}, image[FIELD_COUNT] = {{0}};
    for (size_t cursor = parts.begin; json_next(parts, &cursor, NULL, &part);) {
	    read_fields(context, part, fields);
	    read_fields(context, fields[FIELD_IMAGE_URL], image);
	    const bool is_image =
		    is_string_of(context, fields[FIELD_TYPE], "image_url");
	    if (!is_image &&
		(!is_string_of(context, fields[FIELD_TYPE], "text") ||
		 (fields[FIELD_TEXT].text != NULL &&
		  (!is_kind(fields[FIELD_TEXT], '"') ||
		   (context->trusted == NULL &&
		    holds(context, fields[FIELD_TEXT], IMAGE_PLACEHOLDER,
			  false)))))) {
		    refuse(context,
			   "a content part is text without the image placeholder, or an image");
	    } else if (is_image &&
		       !(is_kind(fields[FIELD_IMAGE_URL], '"') ?
				 is_truthy(context, fields[FIELD_IMAGE_URL]) :
				 is_truthy(context, image[FIELD_URL]))) {
		    refuse(context, "an image has a source");
	    }
    }
}

/* The checks of every message, process_image_messages' and merge_tool_messages', and the merges. */
static void read_messages(struct context *context)
{
    struct json_value message = {0}, fields[FIELD_COUNT] = {{0}};
    for (size_t cursor = context->messages.begin, before = cursor;
	 context->refusal == NULL &&
	 json_next(context->messages, &cursor, NULL, &message);
	 before = cursor) {
	    read_fields(context, message, fields);
	    const uint32_t role =
		    is_string_of(context, fields[FIELD_ROLE], "developer") ?
			    ROLE_SYSTEM :
			    index_of(context, fields[FIELD_ROLE], role_names,
				     ROLE_COUNT);
	    const uint32_t task =
		    fields[FIELD_TASK].text == NULL ||
				    is_kind(fields[FIELD_TASK], 'n') ?
			    TASK_NONE :
			    index_of(context, fields[FIELD_TASK],
				     task_names + 1, TASK_COUNT - 1) +
				    1;
	    const struct json_value content = fields[FIELD_CONTENT],
				    reasoning = fields[FIELD_REASONING];
	    const struct json_value identifier = fields[FIELD_TOOL_CALL_ID];
	    if (role == ROLE_COUNT || task == TASK_COUNT) {
		    refuse(context,
			   "a message is an object with a role and a task the encoder knows");
	    } else if (fields[FIELD_CONTENT_BLOCKS].text != NULL ||
		       fields[FIELD_WITHOUT_END_OF_SENTENCE].text != NULL) {
		    refuse(context,
			   "a message holds none of the encoder's own members, content_blocks and wo_eos");
	    } else if ((content.text != NULL && !is_kind(content, '"') &&
			!is_kind(content, '[') && !is_kind(content, 'n')) ||
		       (role == ROLE_USER && is_kind(content, 'n')) ||
		       (reasoning.text != NULL && !is_kind(reasoning, '"') &&
			!is_kind(reasoning, 'n')) ||
		       (identifier.text != NULL && !is_kind(identifier, '"') &&
			!is_kind(identifier, 'n'))) {
		    refuse(context,
			   "content is a string, an array or null, and reasoning and ids are strings or null");
	    } else if (context->trusted == NULL &&
		       ((is_kind(content, '"') &&
			 holds(context, content, IMAGE_PLACEHOLDER, false)) ||
			(is_kind(reasoning, '"') &&
			 holds(context, reasoning, IMAGE_PLACEHOLDER, false)))) {
		    refuse(context, "no text holds the image placeholder");
	    } else if (is_kind(content, '[')) {
		    check_parts(context, content);
	    }
	    /* Request tools need the reference's system branch. A fieldless system entry renders empty content;
         * putting it before merging also preserves the explicit prepend's turn and reasoning indexes. */
	    if (context->entry_count == 0 && role != ROLE_SYSTEM &&
		is_truthy(context, context->tools)) {
		    context->entries[context->entry_count++] =
			    (struct entry){ .role = ROLE_SYSTEM,
					    .has_own_fields = true,
					    .has_tools = true };
	    }
	    const uint32_t last =
		    context->entry_count -
		    1; /* wraps for the first, which merges into nothing */
	    const bool gives_block =
		    role == ROLE_USER &&
		    !(is_kind(content, '[') && !is_truthy(context, content));
	    if (context->entry_count > 0 &&
		context->entries[last].role == ROLE_USER &&
		(role == ROLE_TOOL ||
		 (role == ROLE_USER &&
		  context->entries[last].task == TASK_NONE))) {
		    context->entries[last].message_count++;
		    context->entries[last].has_blocks =
			    context->entries[last].has_blocks || gives_block;
		    context->entries[last].has_tool_results =
			    context->entries[last].has_tool_results ||
			    role == ROLE_TOOL;
		    continue;
	    }
	    const struct json_value tools =
		    context->entry_count == 0 && context->tools.text != NULL ?
			    context->tools :
			    fields[FIELD_TOOLS];
	    context->entries[context->entry_count++] = (struct entry){
		    message,
		    before,
		    1,
		    role == ROLE_TOOL ? (uint32_t)ROLE_USER : role,
		    role == ROLE_TOOL ? (uint32_t)TASK_NONE : task,
		    role != ROLE_TOOL,
		    role != ROLE_TOOL && is_truthy(context, tools),
		    gives_block,
		    role == ROLE_TOOL
	    };
    }
}

static MUST_CHECK result_status
render_prompt(const char *messages_json, size_t messages_byte_count,
	      const char *tools_json, size_t tools_byte_count,
	      const struct chat_template_options *options, char *prompt_out,
	      size_t prompt_capacity, size_t *prompt_byte_count_out,
	      struct chat_template_spans *trusted_out)
{
    struct json_value messages = {0}, tools = {0}, element = {0};
    if (messages_json == NULL || options == NULL || prompt_out == NULL ||
	prompt_byte_count_out == NULL ||
	(tools_json == NULL && tools_byte_count > 0) ||
	messages_byte_count > CHAT_TEMPLATE_MAXIMUM_INPUT_BYTES ||
	tools_byte_count >
		CHAT_TEMPLATE_MAXIMUM_INPUT_BYTES - messages_byte_count ||
	options->reasoning_effort < CHAT_TEMPLATE_LOWEST_EFFORT ||
	options->reasoning_effort > CHAT_TEMPLATE_HIGHEST_EFFORT) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (json_read_document(messages_json, messages_byte_count, &messages) !=
		JSON_DOCUMENT_VALID ||
	!is_kind(messages, '[') ||
	(tools_json != NULL &&
	 json_read_document(tools_json, tools_byte_count, &tools) !=
		 JSON_DOCUMENT_VALID)) {
	    LOG_DEBUG_ENTRY(
		    "the template refuses a request: its JSON is not what json.loads reads");
	    return STATUS_INVALID_ARGUMENT;
    }
    size_t message_count = 0;
    for (size_t cursor = messages.begin;
	 json_next(messages, &cursor, NULL, &element);) {
	    message_count++;
    }
    const size_t scratch_bytes = SCRATCH_BYTES_PER_INPUT_BYTE * (messages_byte_count + tools_byte_count) +
                                 SCRATCH_SLACK_BYTES;
    struct entry *const entries =
	    FAULT_POINT(chat_template_memory) ?
		    NULL :
		    malloc((message_count + 1u) * sizeof(struct entry) +
			   scratch_bytes +
			   CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS *
				   (sizeof(struct tool_result) +
				    sizeof(struct call_identifier)));
    if (entries == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    struct tool_result *const tool_results = (struct tool_result *)(entries + message_count + 1u);
    struct call_identifier *const call_identifiers =
	    (struct call_identifier *)(tool_results +
				       CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS);
    struct json_writer writer = {prompt_out, prompt_capacity, 0};
    struct chat_template_spans trusted = { 0 };
    struct context context = {
	    .options = options,
	    .writer = &writer,
	    .messages = messages,
	    .tools = tools,
	    .entries = entries,
	    .last_user_index = -1,
	    .tool_results = tool_results,
	    .call_identifiers = call_identifiers,
	    .trusted = trusted_out != NULL ? &trusted : NULL,
	    .scratch = { (char *)(call_identifiers +
				  CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS),
			 scratch_bytes, 0, false }
    };
    read_messages(&context);
    bool has_tools = false;
    for (uint32_t index = 0; index < context.entry_count; index++) {
        const struct entry *const entry = &entries[index];          /* tool results alone open no turn of the user's */
        const bool is_turn = (entry->role == ROLE_USER && (entry->has_blocks || !entry->has_tool_results)) ||
                             (entry->role == ROLE_SYSTEM && index > 0);
        context.last_user_index = is_turn ? index : context.last_user_index;
        has_tools = has_tools || entry->has_tools;
    }
    context.drops_thinking = options->drops_thinking && !has_tools;
    if ((tools_json != NULL && message_count == 0) || (options->continues_final_message &&
        (context.entry_count == 0 || entries[context.entry_count - 1].role != ROLE_ASSISTANT))) {
        refuse(&context, "tools go on a first message, and only an assistant's message is continued");
    }
    write_text(&context, BEGIN_OF_SENTENCE);
    for (uint32_t index = 0; context.refusal == NULL && index < context.entry_count; index++) {
        write_entry(&context, index);
        context.scratch.used = context.kept_scratch;          /* what an entry unescaped is given back */
    }
    free(entries);
    if (context.trusted_status != STATUS_OK || context.refusal != NULL ||
	context.scratch.is_short || writer.length > prompt_capacity) {
	    free(trusted.spans);
    }
    if (context.trusted_status != STATUS_OK) {
	    return context.trusted_status;
    }
    if (context.refusal != NULL || context.scratch.is_short) {
	    LOG_DEBUG_ENTRY("the template refuses a request: %s",
			    context.refusal != NULL ? context.refusal :
						      "its scratch runs short");
	    return STATUS_INVALID_ARGUMENT;
    }
    if (writer.length > prompt_capacity) {
	    LOG_DEBUG_ENTRY(
		    "the template's prompt of %zu bytes is above the capacity",
		    writer.length);
	    return STATUS_UNSUPPORTED;
    }
    *prompt_byte_count_out = writer.length;
    if (trusted_out != NULL) {
        *trusted_out = trusted;
    }
    return STATUS_OK;
}

result_status chat_template_render(const char *messages_json,
				   size_t messages_byte_count,
				   const char *tools_json,
				   size_t tools_byte_count,
				   const struct chat_template_options *options,
				   char *prompt_out, size_t prompt_capacity,
				   size_t *prompt_byte_count_out)
{
    return render_prompt(messages_json, messages_byte_count, tools_json, tools_byte_count, options, prompt_out,
                         prompt_capacity, prompt_byte_count_out, NULL);
}

result_status chat_template_render_spans(
	const char *messages_json, size_t messages_byte_count,
	const char *tools_json, size_t tools_byte_count,
	const struct chat_template_options *options, char *prompt_out,
	size_t prompt_capacity, size_t *prompt_byte_count_out,
	struct chat_template_spans *trusted_out)
{
	return trusted_out == NULL ?
		       STATUS_INVALID_ARGUMENT :
		       render_prompt(messages_json, messages_byte_count,
				     tools_json, tools_byte_count, options,
				     prompt_out, prompt_capacity,
				     prompt_byte_count_out, trusted_out);
}
