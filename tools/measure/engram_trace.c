/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU-only trace generation through the production tokenizer, template and Engram hash.
 * trace PACK TOKEN_IDS_U32 ROW_IDS_U64; tokenize TABLES MESSAGES_JSON TOKEN_IDS_U32.
 * Output files are IDs only. No GPU API is called. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "anemone/engram.h"
#include "anemone/chat_template.h"
#include "anemone/tokenizer.h"

#define MAX_INPUT_BYTES (64u << 20u)
#define MAX_TOKENS 1048576u
#define ARGUMENTS 5
#define NANOSECONDS_PER_SECOND 1000000000ull
static void *read_file(const char *path, size_t *bytes_out)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		return NULL;
	long count = fseek(file, 0, SEEK_END) == 0 ? ftell(file) : -1;
	if (count <= 0 || (unsigned long)count > MAX_INPUT_BYTES ||
	    fseek(file, 0, SEEK_SET) != 0) {
		(void)fclose(file);
		return NULL;
	}
	void *bytes = malloc((size_t)count);
	if (bytes == NULL ||
	    fread(bytes, 1, (size_t)count, file) != (size_t)count ||
	    fclose(file) != 0) {
		free(bytes);
		return NULL;
	}
	*bytes_out = (size_t)count;
	return bytes;
}
static int tokenize(const char *tables_path, const char *messages_path,
		    const char *output_path)
{
	size_t table_bytes = 0, message_bytes = 0, prompt_bytes = 0;
	void *tables = read_file(tables_path, &table_bytes),
	     *messages = read_file(messages_path, &message_bytes);
	char *prompt = malloc(MAX_INPUT_BYTES);
	uint32_t *ids = malloc((size_t)MAX_TOKENS * sizeof(*ids));
	struct text_tokenizer *tokenizer = NULL;
	const struct chat_template_options options = {
		.reasoning_effort = CHAT_TEMPLATE_HIGHEST_EFFORT,
		.is_thinking = false,
		.drops_thinking = true
	};
	result_status status =
		tables != NULL && messages != NULL && prompt != NULL &&
				ids != NULL ?
			tokenizer_open(tables, table_bytes, &tokenizer) :
			STATUS_OUT_OF_MEMORY;
	if (status == STATUS_OK)
		status = chat_template_render(messages, message_bytes, NULL, 0,
					      &options, prompt, MAX_INPUT_BYTES,
					      &prompt_bytes);
	uint32_t tokens = 0;
	if (status == STATUS_OK)
		status = tokenizer_encode(tokenizer, prompt, prompt_bytes, ids,
					  MAX_TOKENS, &tokens);
	FILE *output = status == STATUS_OK ? fopen(output_path, "wbx") : NULL;
	if (output == NULL ||
	    fwrite(ids, sizeof(*ids), tokens, output) != tokens ||
	    fclose(output) != 0)
		status = STATUS_INPUT_OUTPUT_ERROR;
	(void)printf("{\"status\":%d,\"tokens\":%u}\n", (int)status, tokens);
	tokenizer_close(tokenizer);
	free(tables);
	free(messages);
	free(prompt);
	free(ids);
	return status == STATUS_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
static int trace(const char *pack_path, const char *tokens_path,
		 const char *output_path)
{
	size_t token_bytes = 0;
	uint32_t *tokens = read_file(tokens_path, &token_bytes);
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct engram_record *engram = NULL;
	result_status status =
		tokens != NULL && token_bytes % sizeof(*tokens) == 0 &&
				token_bytes / sizeof(*tokens) <= MAX_TOKENS ?
			STATUS_OK :
			STATUS_INVALID_ARGUMENT;
	if (status == STATUS_OK)
		status = storage_direct.reader_open(&pack_path, 1, &reader);
	if (status == STATUS_OK)
		status = pack_open_entry(&storage_direct, reader, &pack);
	if (status == STATUS_OK)
		status = engram_open(&storage_direct, reader, pack, &engram);
	const struct engram_description *description =
		engram_description_of(engram);
	FILE *output = status == STATUS_OK ? fopen(output_path, "wbx") : NULL;
	if (output == NULL)
		status = STATUS_INPUT_OUTPUT_ERROR;
	const uint64_t count = token_bytes / sizeof(*tokens);
	uint64_t ids[ENGRAM_MAXIMUM_TABLES * ENGRAM_MAXIMUM_ROWS_PER_TOKEN];
	struct timespec begin = { 0 }, end = { 0 };
	if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0)
		status = STATUS_TIMED_OUT;
	for (uint64_t position = 0; status == STATUS_OK && position < count;
	     position++) {
		status = engram_row_ids_entry(engram, tokens, NULL, position,
					      ids);
		if (status == STATUS_OK && description != NULL &&
		    fwrite(ids, sizeof(*ids),
			   (size_t)description->table_count *
				   description->rows_per_token,
			   output) != (size_t)description->table_count *
					      description->rows_per_token)
			status = STATUS_INPUT_OUTPUT_ERROR;
	}
	if (clock_gettime(CLOCK_MONOTONIC, &end) != 0)
		status = STATUS_TIMED_OUT;
	if (output != NULL && fclose(output) != 0)
		status = STATUS_INPUT_OUTPUT_ERROR;
	(void)printf(
		"{\"status\":%d,\"tokens\":%llu,\"tables\":%u,\"rows_per_token_per_table\":%u,\"hash_and_write_ns\":%llu}\n",
		(int)status, (unsigned long long)count,
		description != NULL ? description->table_count : 0,
		description != NULL ? description->rows_per_token : 0,
		(unsigned long long)((end.tv_sec - begin.tv_sec) *
					     (int64_t)NANOSECONDS_PER_SECOND +
				     end.tv_nsec - begin.tv_nsec));
	engram_close(engram);
	pack_close(pack);
	storage_direct.reader_close(reader);
	free(tokens);
	return status == STATUS_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
int main(int argument_count, char **arguments)
{
	if (argument_count != ARGUMENTS)
		return EXIT_FAILURE;
	if (strcmp(arguments[1], "trace") == 0)
		return trace(arguments[2], arguments[3], arguments[4]);
	if (strcmp(arguments[1], "tokenize") == 0)
		return tokenize(arguments[2], arguments[3], arguments[4]);
	return EXIT_FAILURE;
}
