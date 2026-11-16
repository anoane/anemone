/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU-only external-input harness. PARSER_FUZZ_TARGET selects a parser;
 * PARSER_FUZZING replaces the seed runner with the libFuzzer entry point. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/chat_completions.h"
#include "anemone/chat_template.h"
#include "anemone/server.h"
#include "json.h"
#include "parser_fuzz_cases.h"
#include "server_messages.h"

#define FUZZ_INPUT_BYTES 65536u
#define FUZZ_OUTPUT_BYTES 131072u
#define FUZZ_SCRATCH_FACTOR 8u
#define FUZZ_SCRATCH_SLACK 4096u
#define FUZZ_REASONING_EFFORT 50u
#define FUZZ_PATH_BYTES 4096u
#define FUZZ_STABLE_TURNS_BIT 2u /* input bits that pick template options */
#define FUZZ_KEEPS_THINKING_BIT 4u
#ifndef PARSER_FUZZ_TARGET
#define PARSER_FUZZ_TARGET FUZZ_HTTP
#endif

static void test_http(const char *bytes, size_t count)
{
	struct server_head head = { 0 };
	/* metrics and Messages on: every route and header branch (x-api-key, anthropic-version) is reachable */
	const bool complete =
		server_head_read_options(bytes, count, true, true, &head);
	if (complete && (head.head_byte_count > count ||
			 head.head_byte_count > SERVER_MOST_HEAD_BYTES ||
			 head.key_begin > head.key_end ||
			 head.key_end > head.head_byte_count)) {
		abort();
	}
}

static void test_chat(const char *bytes, size_t count)
{
	const struct chat_request defaults = { .is_seeded = true,
					       .max_tokens = 1,
					       .top_p = 1.0f };
	struct chat_request request = { 0 };
	struct chat_refusal refusal = { 0 };
	const uint32_t status = chat_request_read(
		bytes, count, "anemone", NULL, &defaults, &request, &refusal);
	if (status == 0 && (request.messages_begin > request.messages_end ||
			    request.messages_end > count ||
			    request.tools_begin > request.tools_end ||
			    request.tools_end > count)) {
		abort();
	}
}

static void test_template(const char *bytes, size_t count)
{
	char output[FUZZ_OUTPUT_BYTES];
	struct chat_template_spans spans = { 0 };
	const struct chat_template_options options = {
		.reasoning_effort = FUZZ_REASONING_EFFORT,
		.is_thinking = count != 0 && (count % FUZZ_SCRATCH_FACTOR) != 0,
		.drops_thinking = (count & FUZZ_KEEPS_THINKING_BIT) == 0,
		.stable_turns = (count & FUZZ_STABLE_TURNS_BIT) != 0
	};
	size_t written = 0;
	/* A NUL separates independently mutated messages and tools documents. */
	const char *const separator = memchr(bytes, '\0', count);
	const size_t messages_count =
		separator != NULL ? (size_t)(separator - bytes) : count;
	const char *const tools = separator != NULL ? separator + 1 : NULL;
	const size_t tools_count =
		separator != NULL ? count - messages_count - 1u : 0;
	const result_status status = chat_template_render_spans(
		bytes, messages_count, tools, tools_count, &options, output,
		sizeof(output), &written, &spans);
	if (status == STATUS_OK) {
		if (written > sizeof(output)) {
			abort();
		}
		for (uint32_t index = 0; index < spans.count; index++) {
			if (spans.spans[index].begin > spans.spans[index].end ||
			    spans.spans[index].end > written) {
				abort();
			}
		}
	}
	free(spans.spans);
}

/* The Messages body parser: its normalized Chat body's ranges stay inside it, and it is always freed. */
static void test_messages(const char *bytes, size_t count)
{
	static char workspace_bytes[FUZZ_SCRATCH_FACTOR * FUZZ_INPUT_BYTES +
				    FUZZ_SCRATCH_SLACK];
	struct json_scratch workspace = { .bytes = workspace_bytes,
					  .capacity = sizeof(workspace_bytes) };
	const struct chat_request defaults = { .is_seeded = true,
					       .max_tokens = 1,
					       .top_p = 1.0f };
	struct chat_request request = { 0 };
	struct chat_buffer converted = { 0 };
	struct chat_refusal refusal = { 0 };
	const uint32_t status = server_messages_read(
		bytes, count, (count & 1u) != 0, &defaults, &request,
		&converted, &workspace, &refusal);
	const bool inside = request.messages_begin <= request.messages_end &&
			    request.messages_end <= converted.count &&
			    request.tools_begin <= request.tools_end &&
			    request.tools_end <= converted.count;
	chat_buffer_free(&converted);
	if (status == 0 && !inside) {
		abort();
	}
}

static void test_json(const char *bytes, size_t count)
{
	struct json_value value = { 0 };
	const enum json_document status =
		json_read_document(bytes, count, &value);
	if (status == JSON_DOCUMENT_VALID) {
		char scratch_bytes[FUZZ_SCRATCH_FACTOR * FUZZ_INPUT_BYTES +
				   FUZZ_SCRATCH_SLACK];
		char output[FUZZ_OUTPUT_BYTES];
		struct json_scratch scratch = { .bytes = scratch_bytes,
						.capacity =
							sizeof(scratch_bytes) };
		struct json_writer writer = { .bytes = output,
					      .capacity = sizeof(output) };
		const bool repeated = json_has_repeated_key(&scratch, value);
		if (!repeated && !scratch.is_short) {
			json_write_canonical(&writer, &scratch, value);
		}
	}
}

/* libFuzzer's entry point: the symbol its main calls is LLVMFuzzerTestOneInput */
int parser_fuzz_one_input(const uint8_t *data,
			  size_t size) __asm__("LLVMFuzzerTestOneInput");
int parser_fuzz_one_input(const uint8_t *data, size_t size)
{
	if (size > FUZZ_INPUT_BYTES) {
		return 0;
	}
	/* Exact-size storage catches reads beyond the caller's length. */
	char *const bytes = malloc(size != 0 ? size : 1u);
	if (bytes == NULL) {
		return 0;
	}
	memcpy(bytes, data, size);
	if (PARSER_FUZZ_TARGET == FUZZ_HTTP) {
		test_http(bytes, size);
	} else if (PARSER_FUZZ_TARGET == FUZZ_CHAT) {
		test_chat(bytes, size);
	} else if (PARSER_FUZZ_TARGET == FUZZ_TEMPLATE) {
		test_template(bytes, size);
	} else if (PARSER_FUZZ_TARGET == FUZZ_MESSAGES) {
		test_messages(bytes, size);
	} else {
		test_json(bytes, size);
	}
	free(bytes);
	return 0;
}

#ifndef PARSER_FUZZING

int main(int argument_count, char **arguments)
{
	for (size_t index = 0;
	     index < sizeof(test_parser_seeds) / sizeof(test_parser_seeds[0]);
	     index++) {
		const struct test_parser_seed *const seed =
			&test_parser_seeds[index];
		if (seed->target != PARSER_FUZZ_TARGET) {
			continue;
		}
		if (parser_fuzz_one_input((const uint8_t *)seed->bytes,
					  seed->count) != 0) {
			return 1;
		}
		if (argument_count > 1) {
			char path[FUZZ_PATH_BYTES];
			const int length = snprintf(path, sizeof(path),
						    "%s/seed_%zu", arguments[1],
						    index);
			if (length < 0 || (size_t)length >= sizeof(path)) {
				return 1;
			}
			FILE *const stream = fopen(path, "wb");
			if (stream == NULL) {
				return 1;
			}
			const bool written = fwrite(seed->bytes, 1, seed->count,
						    stream) == seed->count;
			const int closed = fclose(stream);
			if (!written || closed != 0) {
				return 1;
			}
		}
	}
	return 0;
}
#endif
