/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host-only serving work, using the production parsers, template and tokenizer. No engine or CUDA calls.
 * Arguments: front_end.tables needle.json [iterations [cached]]. Times are milliseconds, CLOCK_MONOTONIC_RAW.
 * The first body is the supplied JSON; the second adds an assistant reply and another user turn. The small case
 * keeps one tenth of the needle's decoded content and its last 256 bytes, on UTF-8 boundaries. File loading,
 * fixture construction and tokenizer-open are outside timing. One matching idle slot and three empty idle slots
 * model server.c's chosen_session_slot; no generated tokens, snapshot I/O, transport or worker work is simulated.
 * Cached mode opens four bounded entries outside timing for each request pair, selects the longest byte prefix
 * inside tokenize_ms, and checks every cached result against full tokenization outside timing.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/chat_completions.h"
#include "anemone/chat_template.h"
#include "anemone/tokenizer.h"
#include "json.h"
#include "serving_cursor.h"

#define CONTEXT_TOKENS 1048576u
#define PROMPT_CAPACITY CHAT_MOST_OUTPUT_BYTES
#define DEFAULT_ITERATIONS 11u
#define MOST_ITERATIONS 1000u
#define FILE_LIMIT CHAT_MOST_OUTPUT_BYTES
#define SMALL_DIVISOR 10u
#define QUESTION_TAIL_BYTES 256u
#define SESSION_SLOTS 4u
#define NANOSECONDS_PER_MILLISECOND 1000000.0
#define MILLISECONDS_PER_SECOND 1000.0
#define DECIMAL_BASE 10
#define ARGUMENT_COUNT 3
#define ITERATIONS_ARGUMENT 3
#define MODE_ARGUMENT 4
#define CACHE_LEAST_PREFIX_BYTES 256u
#define CACHE_SHARED_FRACTION 2u
#define OK STATUS_OK

struct body {
	char *bytes;
	size_t count;
};

struct benchmark {
	struct text_tokenizer *tokenizer;
	char *prompt;
	uint32_t *encoded;
	uint32_t *history;
	uint32_t *reference;
	uint32_t history_count;
	uint32_t next_cache;
	struct tokenizer_cache *caches[SESSION_SLOTS];
	bool is_cached;
};

static double milliseconds(void)
{
	struct timespec now;
	if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
		perror("clock_gettime");
		exit(EXIT_FAILURE);
	}
	return (double)now.tv_sec * MILLISECONDS_PER_SECOND +
	       (double)now.tv_nsec / NANOSECONDS_PER_MILLISECOND;
}

static struct body read_file(const char *path)
{
	struct body result = { 0 };
	FILE *const input = fopen(path, "rb");
	if (input == NULL) {
		return result;
	}
	const int seeked = fseek(input, 0, SEEK_END);
	const long length = seeked == 0 ? ftell(input) : -1;
	if (length > 0 && (unsigned long)length <= FILE_LIMIT &&
	    fseek(input, 0, SEEK_SET) == 0) {
		result.bytes = malloc((size_t)length + 1u);
		if (result.bytes != NULL) {
			result.count =
				fread(result.bytes, 1, (size_t)length, input);
			if (result.count != (size_t)length) {
				free(result.bytes);
				result = (struct body){ 0 };
			} else {
				result.bytes[result.count] = '\0';
			}
		}
	}
	if (fclose(input) != 0) {
		free(result.bytes);
		result = (struct body){ 0 };
	}
	return result;
}

static bool request_read(struct body body, struct chat_request *request)
{
	const struct chat_request defaults = { .max_tokens = 1,
					       .top_p = 1.0f,
					       .is_seeded = true };
	struct chat_refusal refusal = { 0 };
	const uint32_t status = chat_request_read(body.bytes, body.count,
						  "deepseek", NULL, &defaults,
						  request, &refusal);
	if (status != 0) {
		fprintf(stderr, "request refused: %" PRIu32 " %s\n", status,
			refusal.message == NULL ? "" : refusal.message);
	}
	return status == 0;
}

static struct body follow_up(struct body first)
{
	static const char added[] =
		",{\"role\":\"assistant\",\"content\":\"I have read the document.\"},"
		"{\"role\":\"user\",\"content\":\"Continue with one short sentence.\"}";
	struct chat_request request;
	struct body result = { 0 };
	if (!request_read(first, &request) || request.messages_end == 0) {
		return result;
	}
	const size_t insertion = request.messages_end - 1u;
	result.count = first.count + sizeof(added) - 1u;
	result.bytes = malloc(result.count);
	if (result.bytes != NULL) {
		memcpy(result.bytes, first.bytes, insertion);
		memcpy(result.bytes + insertion, added, sizeof(added) - 1u);
		memcpy(result.bytes + insertion + sizeof(added) - 1u,
		       first.bytes + insertion, first.count - insertion);
	}
	return result;
}

/* The fixture is a one-message needle. All construction is checked by the real reader before timing. */
static struct body small_needle(struct body full)
{
	struct body result = { 0 };
	struct chat_request request;
	if (!request_read(full, &request)) {
		return result;
	}
	char *const scratch_bytes = malloc(full.count);
	if (scratch_bytes == NULL) {
		return result;
	}
	struct json_scratch scratch = { scratch_bytes, full.count, 0, false };
	const struct json_value messages = { full.bytes, request.messages_begin,
					     request.messages_end };
	struct json_value message = { 0 }, content = { 0 }, member = { 0 },
			  key = { 0 };
	size_t cursor = messages.begin;
	if (json_next(messages, &cursor, NULL, &message)) {
		cursor = message.begin;
		while (json_next(message, &cursor, &key, &member)) {
			if (json_string_is(&scratch, key, "content")) {
				content = member;
			}
		}
	}
	if (content.text != NULL && content.text[content.begin] == '"') {
		const struct json_bytes decoded =
			json_unescape(&scratch, content);
		size_t prefix = 0, tail = 0;
		while (!scratch.is_short &&
		       prefix < decoded.count / SMALL_DIVISOR) {
			bool valid = false;
			prefix += tokenizer_utf8_sequence_bytes(
				decoded.data, prefix, decoded.count, &valid);
			if (!valid) {
				scratch.is_short = true;
			}
		}
		tail = prefix;
		while (!scratch.is_short &&
		       tail + QUESTION_TAIL_BYTES < decoded.count) {
			bool valid = false;
			tail += tokenizer_utf8_sequence_bytes(
				decoded.data, tail, decoded.count, &valid);
			if (!valid) {
				scratch.is_short = true;
			}
		}
		if (!scratch.is_short) {
			/* Join the decoded prefix and the question's tail before JSON escaping them again. */
			char *const shortened =
				malloc(prefix + decoded.count - tail);
			result.bytes = malloc(full.count);
			if (shortened != NULL && result.bytes != NULL) {
				memcpy(shortened, decoded.data, prefix);
				memcpy(shortened + prefix, decoded.data + tail,
				       decoded.count - tail);
				struct json_writer writer = { result.bytes,
							      full.count, 0 };
				json_write_bytes(&writer, full.bytes,
						 content.begin);
				json_write_string(
					&writer,
					(struct json_bytes){
						shortened,
						prefix + decoded.count -
							tail });
				json_write_bytes(&writer,
						 full.bytes + content.end,
						 full.count - content.end);
				result.count = writer.length <=
							       writer.capacity ?
						       writer.length :
						       0;
			}
			free(shortened);
		}
	}
	free(scratch_bytes);
	if (result.count == 0) {
		free(result.bytes);
		result.bytes = NULL;
	}
	return result;
}

/* Exactly server.c's kept_tokens arithmetic for a completed prompt with no generated rows or checkpoint.
 * Other idle slots are empty, hence keep zero; the matching slot wins whenever its kept count is positive. */
static uint32_t chosen_prefix(const struct benchmark *benchmark,
			      const uint32_t *tokens, uint32_t count)
{
	uint32_t best = 0;
	for (uint32_t slot = 0; slot < SESSION_SLOTS; slot++) {
		const uint32_t history_count =
			slot == 0 ? benchmark->history_count : 0;
		const uint32_t most = history_count < count ? history_count :
							      count;
		uint32_t common = 0;
		while (common < most &&
		       benchmark->history[common] == tokens[common]) {
			common++;
		}
		uint32_t kept = common < count ? common : count - 1u;
		kept -= kept % SERVING_KEPT_GRANULE;
		kept = history_count - kept <= SERVING_KEPT_LAG ? kept : 0;
		best = kept > best ? kept : best;
	}
	return best;
}

static bool measure(struct benchmark *benchmark, struct body body,
		    const char *scale, const char *turn, unsigned iteration)
{
	struct chat_request request;
	size_t prompt_bytes = 0;
	uint32_t token_count = 0;
	const double begin = milliseconds();
	if (!request_read(body, &request)) {
		return false;
	}
	const double parsed = milliseconds();
	const struct chat_template_options options = {
		.reasoning_effort = request.reasoning_effort != 0 ?
					    request.reasoning_effort :
					    CHAT_TEMPLATE_HIGHEST_EFFORT,
		.is_thinking = request.reasoning_effort != 0,
		.drops_thinking = true
	};
	if (chat_template_render(body.bytes + request.messages_begin,
				 request.messages_end - request.messages_begin,
				 request.tools_end > request.tools_begin ?
					 body.bytes + request.tools_begin :
					 NULL,
				 request.tools_end - request.tools_begin,
				 &options, benchmark->prompt, PROMPT_CAPACITY,
				 &prompt_bytes) != OK) {
		fputs("render failed\n", stderr);
		return false;
	}
	const double rendered = milliseconds();
	result_status encoded = OK;
	if (benchmark->is_cached) {
		uint32_t best = 0, chosen = 0;
		for (uint32_t index = 0; index < SESSION_SLOTS; index++) {
			const uint32_t common = tokenizer_cache_common_prefix(
				benchmark->caches[index], benchmark->prompt,
				prompt_bytes);
			if (common > best && common >= prompt_bytes / CACHE_SHARED_FRACTION) {
				best = common;
				chosen = index;
			}
		}
		if (best < CACHE_LEAST_PREFIX_BYTES) {
			chosen = benchmark->next_cache;
			benchmark->next_cache = (chosen + 1u) % SESSION_SLOTS;
		}
		encoded = tokenizer_encode_cached(benchmark->caches[chosen],
						  benchmark->prompt,
						  prompt_bytes,
						  benchmark->encoded,
						  CONTEXT_TOKENS, &token_count);
	} else {
		encoded = tokenizer_encode(benchmark->tokenizer,
					   benchmark->prompt, prompt_bytes,
					   benchmark->encoded, CONTEXT_TOKENS,
					   &token_count);
	}
	if (encoded != OK || token_count == 0) {
		fputs("encode failed\n", stderr);
		return false;
	}
	const double tokenized = milliseconds();
	uint32_t *const queued = malloc((size_t)token_count * sizeof(*queued));
	if (queued == NULL) {
		return false;
	}
	memcpy(queued, benchmark->encoded,
	       (size_t)token_count * sizeof(*queued));
	const double copied = milliseconds();
	const uint32_t kept = chosen_prefix(benchmark, queued, token_count);
	const double matched = milliseconds();
	memcpy(benchmark->history, queued,
	       (size_t)token_count * sizeof(*queued));
	benchmark->history_count = token_count;
	const double recorded = milliseconds();
	free(queued);
	const double finished = milliseconds();
	if (benchmark->is_cached) {
		uint32_t reference_count = 0;
		if (tokenizer_encode(benchmark->tokenizer, benchmark->prompt,
				     prompt_bytes, benchmark->reference,
				     CONTEXT_TOKENS, &reference_count) != OK ||
		    reference_count != token_count ||
		    memcmp(benchmark->reference, benchmark->encoded,
			   (size_t)token_count * sizeof(uint32_t)) != 0) {
			fputs("cached tokens differ from full tokenization\n",
			      stderr);
			return false;
		}
	}
	printf("%s,%s,%u,%zu,%zu,%" PRIu32 ",%" PRIu32
	       ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
	       scale, turn, iteration, body.count, prompt_bytes, token_count,
	       kept, parsed - begin, rendered - parsed, tokenized - rendered,
	       copied - tokenized, matched - copied, recorded - matched,
	       finished - recorded, finished - begin);
	return true;
}

static bool measure_pair(struct benchmark *benchmark, struct body first,
			 struct body follow, const char *scale,
			 unsigned iteration)
{
	bool passed = true;
	benchmark->history_count = 0;
	benchmark->next_cache = 0;
	for (uint32_t index = 0;
	     benchmark->is_cached && passed && index < SESSION_SLOTS; index++) {
		passed = tokenizer_cache_open(benchmark->tokenizer,
					      PROMPT_CAPACITY, CONTEXT_TOKENS,
					      &benchmark->caches[index]) == OK;
	}
	passed = passed &&
		 measure(benchmark, first, scale, "first", iteration) &&
		 measure(benchmark, follow, scale, "follow", iteration);
	for (uint32_t index = 0; index < SESSION_SLOTS; index++) {
		tokenizer_cache_close(benchmark->caches[index]);
		benchmark->caches[index] = NULL;
	}
	return passed;
}

int main(int argument_count, char **arguments)
{
	if (argument_count < ARGUMENT_COUNT ||
	    argument_count > MODE_ARGUMENT + 1 ||
	    (argument_count > MODE_ARGUMENT &&
	     strcmp(arguments[MODE_ARGUMENT], "cached") != 0)) {
		fputs("usage: request_front_end front_end.tables needle.json [iterations [cached]]\n",
		      stderr);
		return EXIT_FAILURE;
	}
	unsigned iterations = DEFAULT_ITERATIONS;
	if (argument_count > ITERATIONS_ARGUMENT) {
		char *end = NULL;
		errno = 0;
		const unsigned long parsed = strtoul(
			arguments[ITERATIONS_ARGUMENT], &end, DECIMAL_BASE);
		if (errno != 0 || end == arguments[ITERATIONS_ARGUMENT] ||
		    *end != '\0' || parsed == 0 || parsed > MOST_ITERATIONS) {
			return EXIT_FAILURE;
		}
		iterations = (unsigned)parsed;
	}
	struct body tables = read_file(arguments[1]);
	struct body full = read_file(arguments[ARGUMENT_COUNT - 1]);
	struct body small = full.bytes == NULL ? (struct body){ 0 } :
						 small_needle(full);
	struct body full_follow = full.bytes == NULL ? (struct body){ 0 } :
						       follow_up(full);
	struct body small_follow = small.bytes == NULL ? (struct body){ 0 } :
							 follow_up(small);
	struct benchmark benchmark = {
		.prompt = malloc(PROMPT_CAPACITY),
		.encoded = malloc((size_t)CONTEXT_TOKENS * sizeof(uint32_t)),
		.history = malloc((size_t)CONTEXT_TOKENS * sizeof(uint32_t)),
		.reference = malloc((size_t)CONTEXT_TOKENS * sizeof(uint32_t)),
		.is_cached = argument_count > MODE_ARGUMENT
	};
	bool passed = tables.bytes != NULL && full.bytes != NULL &&
		      small.bytes != NULL && full_follow.bytes != NULL &&
		      small_follow.bytes != NULL && benchmark.prompt != NULL &&
		      benchmark.encoded != NULL && benchmark.history != NULL &&
		      benchmark.reference != NULL &&
		      tokenizer_open(tables.bytes, tables.count,
				     &benchmark.tokenizer) == OK;
	puts("scale,turn,iteration,body_bytes,prompt_bytes,tokens,kept,parse_ms,render_ms,tokenize_ms,queue_copy_ms,"
	     "prefix_ms,history_copy_ms,cleanup_ms,total_ms");
	for (unsigned iteration = 0; passed && iteration < iterations;
	     iteration++) {
		passed = measure_pair(&benchmark, small, small_follow, "100k",
				      iteration) &&
			 measure_pair(&benchmark, full, full_follow, "1m",
				      iteration);
	}
	tokenizer_close(benchmark.tokenizer);
	free(benchmark.prompt);
	free(benchmark.encoded);
	free(benchmark.history);
	free(benchmark.reference);
	free(tables.bytes);
	free(full.bytes);
	free(small.bytes);
	free(full_follow.bytes);
	free(small_follow.bytes);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
