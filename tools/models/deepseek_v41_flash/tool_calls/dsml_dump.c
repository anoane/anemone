/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* dsml_dump: the answer layer's output for every case of a dsml_corpus.py corpus, as one SHA-256 line each, so that
 * two source trees can be shown to give the same bytes (--dsml_parsing=reference against v0.0.5-alpha; the tolerant
 * reading against v0.0.5-alpha on the reference's own spellings).
 *
 *     dsml_dump CORPUS reference|tolerant [CASE] > LISTING
 *
 * With CASE (its index in the corpus), that case's outputs themselves follow each line, for a difference to be read.
 * Each case is read as tests/engine/server/api/chat_tool_calls_test.c reads it (its tools, stops and thinking, as the server reads a
 * request), then fed whole, one byte at a time, cycling 1..6 bytes, in pieces of 2..6, and split in two at every
 * SPLIT_STEP-th byte, to Chat Completions, Messages and Responses, whole and streamed. A line is
 *     case feeding size api streamed sha256(output) calls
 * A tree without the knob (v0.0.5-alpha) has only the reference rule; its listing is compared with the other tree's in
 * reference mode. Check tool: not part of the engine; built by tools/models/deepseek_v41_flash/tool_calls/dsml_dump.mk.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdbool.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if __has_include("engine/server/api/chat_completions.h") /* the module folders' layout */
#include "engine/server/api/chat_completions.h"
#include "common/json.h"
#include "common/sha256.h"
#else /* v0.0.5-alpha's layout */
#include "anemone/chat_completions.h"
#include "json.h"
#include "sha256.h"
#endif

#define BODY_CAPACITY 65536u
#define SCRATCH_CAPACITY 1048576u
#define LARGEST_PIECE 6u
#define SPLIT_STEP 16u
#define HEX_BASE 16
#define HEX_DIGITS_PER_BYTE 2u

enum feeding { FEED_WHOLE, FEED_FIXED, FEED_CYCLING, FEED_SPLIT };
enum api { API_CHAT, API_MESSAGES, API_RESPONSES, API_COUNT };

struct text {
	char *bytes;
	size_t count;
};

static size_t appended(char *body, size_t length, const char *text,
		       size_t count, bool is_string)
{
	if (length >= BODY_CAPACITY) {
		return BODY_CAPACITY;
	}
	struct json_writer writer = { body + length, BODY_CAPACITY - length,
				      0 };
	if (is_string) {
		json_write_string(&writer, (struct json_bytes){ text, count });
	} else {
		json_write_bytes(&writer, text, count);
	}
	return writer.length <= writer.capacity ? length + writer.length :
						  BODY_CAPACITY;
}

static size_t list_appended(char *body, size_t length, const struct text *list,
			    const char *member)
{
	for (size_t at = 0, index = 0; at < list->count; index++) {
		const size_t count = strlen(list->bytes + at);
		length = appended(body, length, index == 0 ? member : ",",
				  strlen(index == 0 ? member : ","), false);
		if (strcmp(member, ",\"tools\":[") == 0) {
			length = appended(
				body, length,
				"{\"type\":\"function\",\"function\":{\"name\":",
				strlen("{\"type\":\"function\",\"function\":{\"name\":"),
				false);
		}
		length = appended(body, length, list->bytes + at, count, true);
		if (strcmp(member, ",\"tools\":[") == 0) {
			length = appended(body, length, "}}", strlen("}}"),
					  false);
		}
		at += count + 1u;
		length = at >= list->count ?
				 appended(body, length, "]", 1, false) :
				 length;
	}
	return length;
}

static void hex_read(const char *hex, struct text *text, bool is_list)
{
	const size_t length = strcspn(hex, "\r\n");
	free(text->bytes);
	text->bytes = malloc(length + 1u);
	text->count = 0;
	for (size_t at = 0; text->bytes != NULL && at < length;) {
		if (hex[at] == ',') {
			text->bytes[text->count++] = '\0';
			at++;
			continue;
		}
		const char digits[] = { hex[at],
					at + 1 < length ? hex[at + 1] : '0',
					'\0' };
		text->bytes[text->count++] =
			(char)strtol(digits, NULL, HEX_BASE);
		at += HEX_DIGITS_PER_BYTE;
	}
	if (text->bytes != NULL && is_list && length != 0) {
		text->bytes[text->count++] = '\0';
	}
}

static unsigned long raw_case = ULONG_MAX;

static void dumped(unsigned long index, const struct text *stream,
		   const struct text *tools, bool has_tools,
		   const struct text *stops, bool is_thinking, bool is_limited,
		   bool is_reference)
{
	static char body[BODY_CAPACITY], scratch[SCRATCH_CAPACITY];
	size_t length = appended(
		body, 0,
		"{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"reasoning_effort\":",
		strlen("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"reasoning_effort\":"),
		false);
	length = appended(body, length, is_thinking ? "\"high\"" : "\"none\"",
			  strlen(is_thinking ? "\"high\"" : "\"none\""), false);
	length = has_tools ?
			 list_appended(body, length, tools, ",\"tools\":[") :
			 length;
	length = list_appended(body, length, stops, ",\"stop\":[");
	length = appended(body, length, "}", 1, false);
	struct chat_request defaults = { .max_tokens = UINT32_MAX,
					 .top_p = 1.0f,
					 .is_seeded = true };
#ifdef CHAT_MARKER_SPACE_MAX /* the knob's tree; v0.0.5-alpha has only the reference rule */
	defaults.is_calls_reference = is_reference;
#else
	(void)is_reference;
#endif
	struct chat_request *const request = calloc(1, sizeof(*request));
	struct chat_refusal refusal = { 0 };
	if (request == NULL || length >= BODY_CAPACITY ||
	    chat_request_read(body, length, "test-model", NULL, &defaults,
			      request, &refusal) != 0) {
		(void)printf("%lu unread\n", index);
		free(request);
		return;
	}
	struct chat_buffer names = {
		0
	}; /* the names the server keeps for an answer */
#ifdef CHAT_MARKER_SPACE_MAX
	chat_tool_names_kept(body + request->tools_begin,
			     request->tools_end - request->tools_begin, &names);
#endif
	for (uint32_t api = API_CHAT; api < API_COUNT; api++) {
		for (uint32_t streamed = 0; streamed <= 1; streamed++) {
			for (uint32_t feeding = FEED_WHOLE;
			     feeding <= FEED_SPLIT; feeding++) {
				const size_t last =
					feeding == FEED_FIXED ? LARGEST_PIECE :
					feeding == FEED_SPLIT ? stream->count :
								1u;
				for (size_t size = 1; size <= last;
				     size +=
				     feeding == FEED_SPLIT ? SPLIT_STEP : 1u) {
					struct chat_answer answer = { 0 };
					request->is_streamed = streamed != 0;
					request->includes_usage = true;
					request->is_messages = api ==
							       API_MESSAGES;
					request->is_responses = api ==
								API_RESPONSES;
					chat_answer_begin(&answer, request,
							  "test-model",
							  "chatcmpl-1", 1, 3u);
#ifdef CHAT_MARKER_SPACE_MAX
					answer.tool_names = &names;
#endif
					answer.messages.scratch = scratch;
					answer.messages.scratch_capacity =
						sizeof(scratch);
					for (size_t position = 0, turn = 0;
					     position < stream->count; turn++) {
						const size_t left =
							stream->count -
							position;
						const size_t wanted =
							feeding == FEED_WHOLE ?
								left :
							feeding == FEED_FIXED ?
								size :
							feeding == FEED_CYCLING ?
								1u + turn % LARGEST_PIECE :
							position < size ?
								size - position :
								left;
						const size_t piece =
							wanted < left ? wanted :
									left;
						(void)chat_answer_token(
							&answer,
							stream->bytes +
								position,
							(uint32_t)piece);
						position += piece;
					}
					if (!is_limited) {
						(void)chat_answer_token(
							&answer, NULL, 0);
					}
					chat_answer_finish(
						&answer,
						is_limited ? "length" : "stop");
					unsigned char
						digest[SHA256_DIGEST_BYTES] = {
							0
						};
					const size_t count =
						answer.output.bytes != NULL ?
							answer.output.count -
								answer.output
									.sent :
							0;
					const bool hashed =
						sha256_of(
							count != 0 ?
								answer.output.bytes +
									answer.output
										.sent :
								"",
							count,
							digest) == STATUS_OK;
					(void)printf("%lu %u %zu %u %u ", index,
						     feeding, size, api,
						     streamed);
					for (size_t byte = 0;
					     byte < sizeof(digest); byte++) {
						(void)printf("%02x",
							     digest[byte]);
					}
					(void)printf(" %u%s\n",
						     answer.call_count,
						     hashed ? "" : " unhashed");
					if (index == raw_case) {
						(void)fwrite(
							answer.output.bytes +
								answer.output
									.sent,
							1, count, stdout);
						(void)printf("\n");
					}
					chat_answer_free(&answer);
				}
			}
		}
	}
	chat_buffer_free(&names);
	free(request);
}

int main(int argc, char **argv)
{
	FILE *const file = argc == 3 || argc == 4 ? fopen(argv[1], "rb") : NULL;
	if (file == NULL) {
		(void)fprintf(
			stderr,
			"usage: dsml_dump CORPUS reference|tolerant [CASE]\n");
		return 2;
	}
	const bool is_reference = strcmp(argv[2], "reference") == 0;
	raw_case = argc == 4 ? strtoul(argv[3], NULL, 10) : raw_case;
	struct text stream = { 0 }, tools = { 0 }, stops = { 0 };
	bool has_tools = false, is_thinking = false, is_limited = false;
	unsigned long index = 0;
	char *line = NULL;
	size_t capacity = 0;
	while (getline(&line, &capacity, file) > 0) {
		const char *const value =
			strchr(line, ' ') != NULL ? strchr(line, ' ') + 1 : "";
		if (strncmp(line, "tools ", strlen("tools ")) == 0) {
			has_tools = value[0] != '-';
			hex_read(has_tools ? value : "", &tools, true);
		} else if (strncmp(line, "stops ", strlen("stops ")) == 0) {
			hex_read(value, &stops, true);
		} else if (strncmp(line, "thinking ", strlen("thinking ")) ==
			   0) {
			is_thinking = value[0] == '1';
		} else if (strncmp(line, "finish ", strlen("finish ")) == 0) {
			is_limited =
				strncmp(value, "length", strlen("length")) == 0;
		} else if (strncmp(line, "stream ", strlen("stream ")) == 0) {
			hex_read(value, &stream, false);
		} else if (strncmp(line, "end", strlen("end")) == 0) {
			dumped(index++, &stream, &tools, has_tools, &stops,
			       is_thinking, is_limited, is_reference);
		}
	}
	free(line);
	free(stream.bytes);
	free(tools.bytes);
	free(stops.bytes);
	(void)fclose(file);
	return 0;
}
