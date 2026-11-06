/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Template provenance and literal payload token ids, on byte-only host tables with real control spellings. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/chat_template.h"
#include "anemone/hooks.h"
#include "anemone/tokenizer.h"

#define TEST_TIME_LIMIT_SECONDS 600
#define PROMPT_CAPACITY 32768u
#define JSON_CAPACITY 8192u
#define PAYLOAD_CAPACITY 1024u
#define ID_CAPACITY PROMPT_CAPACITY
#define MARKER_BEGIN "LITERAL_BEGIN:"
#define MARKER_END ":LITERAL_END"
#define DEFAULT_EFFORT 75u
#define FAULT_FIRST 1u
#define FAULT_SPANS 2u
#define BYTE_TOKENS TOKENIZER_BYTE_VALUES
#define DSML "｜DSML｜"

struct marker {
	const char *text, *escaped;
};

static const struct marker markers[] = {
	{ "<｜begin▁of▁sentence｜>",
	  "\\u003c\\uff5cbegin\\u2581of\\u2581sentence\\uff5c\\u003e" },
	{ "<｜end▁of▁sentence｜>",
	  "\\u003c\\uff5cend\\u2581of\\u2581sentence\\uff5c\\u003e" },
	{ "<｜System｜>", "\\u003c\\uff5cSystem\\uff5c\\u003e" },
	{ "<｜User｜>", "\\u003c\\uff5cUser\\uff5c\\u003e" },
	{ "<｜Assistant｜>", "\\u003c\\uff5cAssistant\\uff5c\\u003e" },
	{ "<think>", "\\u003cthink\\u003e" },
	{ "</think>", "\\u003c/think\\u003e" },
	{ DSML, "\\uff5cDSML\\uff5c" },
	{ "<tool_result>", "\\u003ctool_result\\u003e" },
	{ "</tool_result>", "\\u003c/tool_result\\u003e" },
	{ "<｜deepseek_image｜>",
	  "\\u003c\\uff5cdeepseek_image\\uff5c\\u003e" },
	{ "<" DSML " calls>", "\\u003c\\uff5cDSML\\uff5c calls\\u003e" },
	{ "</" DSML " calls>", "\\u003c/\\uff5cDSML\\uff5c calls\\u003e" },
	{ "<" DSML " invoke", "\\u003c\\uff5cDSML\\uff5c invoke" },
	{ "</" DSML " invoke>", "\\u003c/\\uff5cDSML\\uff5c invoke\\u003e" },
	{ "<" DSML " parameter", "\\u003c\\uff5cDSML\\uff5c parameter" },
	{ "</" DSML " parameter>",
	  "\\u003c/\\uff5cDSML\\uff5c parameter\\u003e" }
};
#define MARKER_COUNT (sizeof(markers) / sizeof(markers[0]))

static int failures;

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", what);
		failures++;
	}
}

/* No merge can make a control id. Every ordinary piece is exactly its byte ids, independently of the splits. */
static void *open_tables(struct text_tokenizer **tokenizer)
{
	const uint32_t vocabulary = BYTE_TOKENS + (uint32_t)MARKER_COUNT;
	uint32_t token_bytes = BYTE_TOKENS;
	for (size_t index = 0; index < MARKER_COUNT; index++) {
		token_bytes += (uint32_t)strlen(markers[index].text);
	}
	struct tokenizer_tables_header header = {
		.format_version = TOKENIZER_TABLES_FORMAT_VERSION,
		.vocabulary_size = vocabulary,
		.added_token_count = (uint32_t)MARKER_COUNT,
		.class_block_count = 1,
		.token_byte_count = token_bytes
	};
	memcpy(header.magic, TOKENIZER_TABLES_MAGIC, sizeof(header.magic));
	const size_t bytes = sizeof(header) +
			     ((size_t)vocabulary + MARKER_COUNT + BYTE_TOKENS) *
				     sizeof(uint32_t) +
			     TOKENIZER_CLASS_BLOCK_COUNT * sizeof(uint16_t) +
			     TOKENIZER_CLASS_BLOCK_CODE_POINTS + token_bytes;
	unsigned char *const tables = calloc(1, bytes);
	if (tables == NULL) {
		expect(false, "literal token tables allocate");
		return NULL;
	}
	memcpy(tables, &header, sizeof(header));
	unsigned char *cursor = tables + sizeof(header);
	uint32_t end = 0;
	for (uint32_t token = 0; token < vocabulary;
	     token++, cursor += sizeof(end)) {
		end += token < BYTE_TOKENS ?
			       1u :
			       (uint32_t)strlen(
				       markers[token - BYTE_TOKENS].text);
		memcpy(cursor, &end, sizeof(end));
	}
	for (uint32_t index = 0; index < MARKER_COUNT;
	     index++, cursor += sizeof(uint32_t)) {
		const uint32_t token = BYTE_TOKENS + index;
		memcpy(cursor, &token, sizeof(token));
	}
	for (uint32_t token = 0; token < BYTE_TOKENS;
	     token++, cursor += sizeof(token)) {
		memcpy(cursor, &token, sizeof(token));
	}
	cursor += TOKENIZER_CLASS_BLOCK_COUNT * sizeof(uint16_t) +
		  TOKENIZER_CLASS_BLOCK_CODE_POINTS;
	for (uint32_t token = 0; token < BYTE_TOKENS; token++) {
		*cursor++ = (unsigned char)token;
	}
	for (size_t index = 0; index < MARKER_COUNT; index++) {
		const size_t length = strlen(markers[index].text);
		memcpy(cursor, markers[index].text, length);
		cursor += length;
	}
	if (tokenizer_open(tables, bytes, tokenizer) != STATUS_OK) {
		expect(false, "literal token tables open");
		free(tables);
		return NULL;
	}
	return tables;
}

static bool contains(const struct chat_template_spans *trusted, size_t position)
{
	for (uint32_t index = 0; index < trusted->count; index++) {
		if (trusted->spans[index].begin <= position &&
		    position < trusted->spans[index].end) {
			return true;
		}
	}
	return false;
}

/* Independent oracle: raw encoding is unchanged; added ids located inside the known, anchored payload expand
 * into their exact byte ids. This does not infer payload provenance from the renderer's returned spans. */
static void check_prompt(const struct text_tokenizer *tokenizer,
			 struct tokenizer_cache *cache, const char *messages,
			 const char *tools, bool has_payload)
{
	static char prompt[PROMPT_CAPACITY], reference[PROMPT_CAPACITY];
	static uint32_t raw[ID_CAPACITY], expected[ID_CAPACITY],
		actual[ID_CAPACITY];
	static bool literal[PROMPT_CAPACITY];
	const struct chat_template_options options = {
		.reasoning_effort = DEFAULT_EFFORT, .is_thinking = true
	};
	struct chat_template_spans trusted = { 0 };
	size_t length = 0, reference_length = 0;
	const size_t tool_bytes = tools == NULL ? 0 : strlen(tools);
	const result_status rendered = chat_template_render_spans(
		messages, strlen(messages), tools, tool_bytes, &options, prompt,
		sizeof(prompt), &length, &trusted);
	expect(rendered == STATUS_OK,
	       "literal payloads render without refusal");
	if (rendered != STATUS_OK) {
		return;
	}
	const result_status raw_rendered = chat_template_render(
		messages, strlen(messages), tools, tool_bytes, &options,
		reference, sizeof(reference), &reference_length);
	const bool has_image_text =
		memmem(prompt, length,
		       MARKER_BEGIN "<｜deepseek_image｜>" MARKER_END,
		       strlen(MARKER_BEGIN
			      "<｜deepseek_image｜>" MARKER_END)) != NULL;
	expect((raw_rendered == STATUS_OK && reference_length == length &&
		memcmp(reference, prompt, length) == 0) ||
		       (has_image_text &&
			raw_rendered == STATUS_INVALID_ARGUMENT),
	       "prompt bytes stay exact; old raw rendering alone refuses image-placeholder text");
	memset(literal, 0, length);
	size_t payloads = 0;
	for (size_t position = 0; position < length;) {
		const char *const begin =
			memmem(prompt + position, length - position,
			       MARKER_BEGIN, strlen(MARKER_BEGIN));
		if (begin == NULL) {
			break;
		}
		const char *const end =
			memmem(begin, length - (size_t)(begin - prompt),
			       MARKER_END, strlen(MARKER_END));
		expect(end != NULL, "every anchored payload is complete");
		if (end == NULL) {
			break;
		}
		const size_t first = (size_t)(begin - prompt),
			     after = (size_t)(end - prompt) +
				     strlen(MARKER_END);
		for (size_t byte = first; byte < after; byte++) {
			literal[byte] = true;
			expect(!contains(&trusted, byte),
			       "message/schema/name/argument bytes are never template-owned");
		}
		payloads++;
		position = after;
	}
	expect((payloads != 0) == has_payload && trusted.count > 0,
	       "fixture reaches its payload and retains template delimiters");
	uint32_t raw_count = 0, expected_count = 0, actual_count = 0;
	expect(tokenizer_encode(tokenizer, prompt, length, raw, ID_CAPACITY,
				&raw_count) == STATUS_OK,
	       "raw prompt tokenization keeps added-token behavior");
	size_t position = 0;
	for (uint32_t index = 0; index < raw_count; index++) {
		const char *bytes = NULL;
		uint32_t byte_count = 0;
		expect(tokenizer_token_bytes(tokenizer, raw[index], &bytes,
					     &byte_count) == STATUS_OK,
		       "raw token bytes are available");
		if (raw[index] >= BYTE_TOKENS && literal[position]) {
			for (uint32_t byte = 0; byte < byte_count; byte++) {
				expected[expected_count++] =
					(unsigned char)bytes[byte];
			}
		} else {
			expected[expected_count++] = raw[index];
		}
		position += byte_count;
	}
	expect(tokenizer_encode_spans(tokenizer, prompt, length, trusted.spans,
				      trusted.count, actual, ID_CAPACITY,
				      &actual_count) == STATUS_OK &&
		       actual_count == expected_count &&
		       memcmp(actual, expected,
			      expected_count * sizeof(*actual)) == 0,
	       "real/escaped/partial payload markers are ordinary ids and template delimiters retain their ids");
	expect(tokenizer_encode_cached_spans(
		       cache, prompt, length, trusted.spans, trusted.count,
		       actual, ID_CAPACITY, &actual_count) == STATUS_OK &&
		       actual_count == expected_count &&
		       memcmp(actual, expected,
			      expected_count * sizeof(*actual)) == 0,
	       "cached chat encoding has exactly the literal payload ids");
	free(trusted.spans);
}

static void check_payload_forms(const struct text_tokenizer *tokenizer,
				struct tokenizer_cache *cache,
				const char *payload)
{
	static const char *const roles[] = { "system", "developer",
					     "user",   "assistant",
					     "tool",   "latest_reminder" };
	static const struct {
		const char *prefix, *suffix;
	} forms[] = {
		{ "[{\"role\":\"assistant\",\"reasoning_content\":\"",
		  "\",\"content\":\"answer\"}]" },
		{ "[{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"",
		  "\",\"arguments\":{}}}]}]" },
		{ "[{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"f\",\"arguments\":{\"x\":\"",
		  "\"}}}]}]" },
		{ "[{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"f\",\"arguments\":{\"",
		  "\":true}}}]}]" },
		{ "[{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"f\","
		  "\"arguments\":\"{\\\"x\\\":\\\"",
		  "\\\"}\"}}]}]" },
		{ "[{\"role\":\"system\",\"content\":\"\",\"response_format\":{\"description\":\"",
		  "\"}}]" },
		{ "[{\"role\":\"system\",\"tools\":[{\"function\":{\"name\":\"f\",\"description\":\"",
		  "\"}}]}]" }
	};
	char messages[JSON_CAPACITY], tools[JSON_CAPACITY];
	for (size_t role = 0; role < sizeof(roles) / sizeof(roles[0]); role++) {
		(void)snprintf(messages, sizeof(messages),
			       "[{\"role\":\"%s\",\"content\":\"%s\"}]",
			       roles[role], payload);
		check_prompt(tokenizer, cache, messages, NULL, true);
		(void)snprintf(
			messages, sizeof(messages),
			"[{\"role\":\"%s\",\"content\":[{\"type\":\"text\",\"text\":\"%s\"}]}]",
			roles[role], payload);
		check_prompt(tokenizer, cache, messages, NULL, true);
	}
	for (size_t form = 0; form < sizeof(forms) / sizeof(forms[0]); form++) {
		(void)snprintf(messages, sizeof(messages), "%s%s%s",
			       forms[form].prefix, payload, forms[form].suffix);
		check_prompt(tokenizer, cache, messages, NULL, true);
	}
	(void)snprintf(
		tools, sizeof(tools),
		"[{\"function\":{\"name\":\"%s\",\"description\":\"%s\","
		"\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"enum\":[\"%s\"]}}}}}]",
		payload, payload, payload);
	check_prompt(tokenizer, cache,
		     "[{\"role\":\"user\",\"content\":\"q\"}]", tools, true);
}

static void test_payloads(const struct text_tokenizer *tokenizer,
			  struct tokenizer_cache *cache)
{
	char payload[PAYLOAD_CAPACITY];
	for (size_t marker = 0; marker < MARKER_COUNT; marker++) {
		for (uint32_t escaped = 0; escaped <= 1; escaped++) {
			(void)snprintf(payload, sizeof(payload),
				       MARKER_BEGIN "%s" MARKER_END,
				       escaped != 0 ? markers[marker].escaped :
						      markers[marker].text);
			check_payload_forms(tokenizer, cache, payload);
		}
	}
	static const char *const partials[] = { "<｜System｜",
						"<｜end▁of▁sentence", "<think",
						"｜DSM", "<tool_result" };
	for (size_t index = 0; index < sizeof(partials) / sizeof(partials[0]);
	     index++) {
		(void)snprintf(payload, sizeof(payload),
			       MARKER_BEGIN "%s" MARKER_END, partials[index]);
		check_payload_forms(tokenizer, cache, payload);
	}
	check_prompt(
		tokenizer, cache,
		"[{\"role\":\"user\",\"content\":\"ab ordinary text 123\"}]",
		NULL, false);
	check_prompt(
		tokenizer, cache,
		"[{\"role\":\"system\",\"content\":\"\"},{\"role\":\"user\",\"content\":["
		"{\"type\":\"text\",\"text\":\"\"}]},{\"role\":\"assistant\",\"reasoning_content\":\"\","
		"\"content\":\"\"}]",
		NULL, false);
}

static void test_span_contract(const struct text_tokenizer *tokenizer,
			       struct tokenizer_cache *cache)
{
	static const char text[] = "<think>";
	enum { SPLIT = 3, SPAN_PAIR = 2 };
	const struct tokenizer_span whole = { 0, sizeof(text) - 1u };
	const struct tokenizer_span split[SPAN_PAIR] = {
		{ 0, SPLIT }, { SPLIT, sizeof(text) - 1u }
	};
	const struct tokenizer_span overlap[SPAN_PAIR] = {
		{ 0, SPLIT }, { SPLIT - 1u, sizeof(text) - 1u }
	};
	const struct tokenizer_span outside = { 0, sizeof(text) };
	uint32_t ids[ID_CAPACITY] = { 0 }, count = 0;
	expect(tokenizer_encode_cached_spans(cache, text, strlen(text), &whole,
					     1, ids, ID_CAPACITY,
					     &count) == STATUS_OK &&
		       count == 1 && ids[0] >= BYTE_TOKENS,
	       "a complete trusted marker emits its control id");
	expect(tokenizer_encode_cached_spans(cache, text, strlen(text), split,
					     SPAN_PAIR, ids, ID_CAPACITY,
					     &count) == STATUS_OK &&
		       count == strlen(text),
	       "a marker crossing two adjacent trusted spans is literal even after a trusted cache hit");
	expect(tokenizer_encode_cached_spans(cache, text, strlen(text), NULL, 0,
					     ids, ID_CAPACITY,
					     &count) == STATUS_OK &&
		       count == strlen(text),
	       "the same cached text with no trusted span becomes literal");
	expect(tokenizer_encode_cached(cache, text, strlen(text), ids,
				       ID_CAPACITY, &count) == STATUS_OK &&
		       count == 1 && ids[0] >= BYTE_TOKENS,
	       "raw cache use restores today's complete marker behavior");
	count = UINT32_MAX;
	expect(tokenizer_encode_spans(tokenizer, text, strlen(text), NULL, 1,
				      ids, ID_CAPACITY,
				      &count) == STATUS_INVALID_ARGUMENT &&
		       tokenizer_encode_spans(tokenizer, text, strlen(text),
					      overlap, SPAN_PAIR, ids,
					      ID_CAPACITY, &count) ==
			       STATUS_INVALID_ARGUMENT &&
		       tokenizer_encode_spans(tokenizer, text, strlen(text),
					      &outside, 1, ids, ID_CAPACITY,
					      &count) ==
			       STATUS_INVALID_ARGUMENT &&
		       count == UINT32_MAX,
	       "bad provenance is refused without publishing an output count");
	expect(tokenizer_encode_cached_spans(cache, "\xff", 1, NULL, 0, ids,
					     ID_CAPACITY, &count) ==
			       STATUS_INVALID_ARGUMENT &&
		       count == UINT32_MAX &&
		       tokenizer_cache_common_prefix(cache, text,
						     strlen(text)) == 0,
	       "chat offsets require valid UTF-8 and failure invalidates cached data");
	char prompt[PROMPT_CAPACITY];
	struct chat_template_spans trusted = { 0 };
	const struct chat_template_options options = { .reasoning_effort =
							       DEFAULT_EFFORT };
	const char *const messages = "[{\"role\":\"user\",\"content\":\"q\"}]";
	size_t length = SIZE_MAX;
	expect(chat_template_render_spans(messages, strlen(messages), NULL, 0,
					  &options, prompt, sizeof(prompt),
					  &length,
					  NULL) == STATUS_INVALID_ARGUMENT &&
		       chat_template_render_spans(messages, strlen(messages),
						  NULL, 0, &options, prompt, 1,
						  &length, &trusted) ==
			       STATUS_UNSUPPORTED &&
		       length == SIZE_MAX && trusted.spans == NULL &&
		       trusted.count == 0,
	       "span result and length stay unchanged on template refusal or insufficient output");
#if CHECKING_HOOKS_ENABLED
	for (uint32_t fault = FAULT_FIRST; fault <= FAULT_SPANS; fault++) {
		expect(fault_arm("chat_template_memory", fault),
		       "template allocation failure is armed");
		expect(chat_template_render_spans(
			       messages, strlen(messages), NULL, 0, &options,
			       prompt, sizeof(prompt), &length,
			       &trusted) == STATUS_OUT_OF_MEMORY &&
			       fault_disarm() >= fault && length == SIZE_MAX &&
			       trusted.spans == NULL && trusted.count == 0,
		       "scratch and span allocation failures preserve outputs and release their owned memory");
	}
#endif
}

int main(void)
{
	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	struct text_tokenizer *tokenizer = NULL;
	void *const tables = open_tables(&tokenizer);
	struct tokenizer_cache *cache = NULL;
	if (tables != NULL &&
	    tokenizer_cache_open(tokenizer, PROMPT_CAPACITY, ID_CAPACITY,
				 &cache) == STATUS_OK) {
		test_payloads(tokenizer, cache);
		test_span_contract(tokenizer, cache);
	} else {
		expect(false, "payload cache opens");
	}
	tokenizer_cache_close(cache);
	tokenizer_close(tokenizer);
	free(tables);
	if (failures != 0) {
		return EXIT_FAILURE;
	}
	(void)printf(
		"chat_payload_test passed (real/escaped/partial markers, all roles, reasoning, tools and caches)\n");
	return EXIT_SUCCESS;
}
