/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The complete API image template/token expansion against CPU fixtures from the official reference.
 * Optional arguments: <converted official tokenizer tables> <make_image_chat_golden.py output>.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/chat_completions.h"
#include "anemone/chat_images.h"
#include "anemone/chat_template.h"
#include "anemone/tokenizer.h"

#define GOLDEN_MAGIC "ANEMVLG1"
#define GOLDEN_MAGIC_BYTES 8u
#define TOKEN_CAPACITY 32768u
#define PROMPT_CAPACITY 131072u
#define TEST_EFFORT 75u
#define FILE_LIMIT (32ul * 1024ul * 1024ul)
#define ARGUMENT_COUNT 3
#define TABLE_ARGUMENT 1
#define FIXTURE_ARGUMENT 2
#define IMAGE_FIELDS 4u
#define TOKEN_START_FIELD 0u
#define TOKEN_COUNT_FIELD 1u
#define VIT_HEIGHT_FIELD 2u
#define VIT_WIDTH_FIELD 3u

struct fixture_reader {
	const unsigned char *bytes;
	size_t count, used;
	bool failed;
};

static int failures;

static void expect(bool valid, const char *description)
{
	if (!valid) {
		(void)fprintf(stderr, "FAILED: %s\n", description);
		failures++;
	}
}

static void *read_file(const char *path, size_t *count_out)
{
	FILE *const input = fopen(path, "rb");
	if (input == NULL) {
		return NULL;
	}
	unsigned char *bytes = NULL;
	long count = -1;
	if (fseek(input, 0, SEEK_END) == 0) {
		count = ftell(input);
	}
	if (count >= 0 && (unsigned long)count <= FILE_LIMIT &&
	    fseek(input, 0, SEEK_SET) == 0) {
		bytes = malloc((size_t)count + 1u);
		if (bytes != NULL &&
		    fread(bytes, 1, (size_t)count, input) != (size_t)count) {
			free(bytes);
			bytes = NULL;
		}
	}
	const int closed = fclose(input);
	if (closed != 0) {
		free(bytes);
		bytes = NULL;
	}
	if (bytes != NULL) {
		*count_out = (size_t)count;
	}
	return bytes;
}

static const unsigned char *take(struct fixture_reader *reader, size_t count)
{
	if (reader->failed || count > reader->count - reader->used) {
		reader->failed = true;
		return NULL;
	}
	const unsigned char *const bytes = reader->bytes + reader->used;
	reader->used += count;
	return bytes;
}

static uint32_t take_integer(struct fixture_reader *reader)
{
	uint32_t value = 0;
	const void *const bytes = take(reader, sizeof(value));
	if (bytes != NULL) {
		memcpy(&value, bytes, sizeof(value));
	}
	return value;
}

static void compare_case(const struct text_tokenizer *tokenizer,
			 struct fixture_reader *reader)
{
	const uint32_t messages_bytes = take_integer(reader);
	const char *const messages = (const char *)take(reader, messages_bytes);
	const uint32_t reference_bytes = take_integer(reader);
	const char *const reference =
		(const char *)take(reader, reference_bytes);
	const uint32_t reference_count = take_integer(reader);
	const void *const reference_ids =
		take(reader, (size_t)reference_count * sizeof(uint32_t));
	const uint32_t image_count = take_integer(reader);
	uint32_t metadata[IMAGE_MAX_IMAGES][IMAGE_FIELDS] = { { 0 } };
	if (image_count > IMAGE_MAX_IMAGES) {
		reader->failed = true;
	}
	for (uint32_t index = 0; index < image_count && !reader->failed;
	     index++) {
		for (uint32_t field = 0; field < IMAGE_FIELDS; field++) {
			metadata[index][field] = take_integer(reader);
		}
	}
	if (reader->failed) {
		return;
	}
	static char prompt[PROMPT_CAPACITY];
	static uint32_t tokens[TOKEN_CAPACITY];
	size_t prompt_bytes = 0;
	uint32_t token_count = 0;
	struct chat_template_spans trusted = { 0 };
	struct chat_images images = { 0 };
	struct chat_refusal refusal = { 0 };
	const struct chat_template_options options = {
		.reasoning_effort = TEST_EFFORT, .drops_thinking = true
	};
	const result_status rendered = chat_template_render_spans(
		messages, messages_bytes, NULL, 0, &options, prompt,
		sizeof(prompt), &prompt_bytes, &trusted);
	const bool prepared = rendered == STATUS_OK &&
			      chat_images_prepare(messages, messages_bytes,
						  &trusted, &images,
						  &refusal) == 0;
	expect(prepared,
	       "official image conversations render, decode and preprocess");
	if (prepared) {
		expect(prompt_bytes == reference_bytes &&
			       memcmp(prompt, reference, prompt_bytes) == 0,
		       "text and image template bytes equal encoding.py");
		const result_status encoded = tokenizer_encode_spans(
			tokenizer, prompt, prompt_bytes, trusted.spans,
			trusted.count, tokens, TOKEN_CAPACITY, &token_count);
		const result_status expanded =
			encoded != STATUS_OK ?
				encoded :
				chat_images_expand(&images, tokens,
						   TOKEN_CAPACITY,
						   &token_count);
		expect(expanded == STATUS_OK &&
			       token_count == reference_count &&
			       memcmp(tokens, reference_ids,
				      (size_t)token_count * sizeof(*tokens)) ==
				       0,
		       "expanded image token IDs equal official prepare_vl_inputs");
		expect(images.count == image_count, "reference image count");
		for (uint32_t index = 0;
		     index < images.count && index < image_count; index++) {
			expect(images.images[index].token_begin ==
					       metadata[index]
						       [TOKEN_START_FIELD] &&
				       images.images[index].plan.token_count ==
					       metadata[index]
						       [TOKEN_COUNT_FIELD] &&
				       images.images[index].plan.vit_height ==
					       metadata[index]
						       [VIT_HEIGHT_FIELD] &&
				       images.images[index].plan.vit_width ==
					       metadata[index][VIT_WIDTH_FIELD],
			       "image token starts and grids follow reference prompt order");
		}
	}
	chat_images_destroy(&images);
	free(trusted.spans);
}

int main(int argument_count, char **arguments)
{
	if (argument_count == 1) {
		(void)printf(
			"image_chat_golden_test: supply tokenizer tables and official image chat fixture to compare\n");
		return EXIT_SUCCESS;
	}
	if (argument_count != ARGUMENT_COUNT) {
		return EXIT_FAILURE;
	}
	size_t tables_bytes = 0, fixture_bytes = 0;
	void *const tables =
		read_file(arguments[TABLE_ARGUMENT], &tables_bytes);
	void *const fixture =
		read_file(arguments[FIXTURE_ARGUMENT], &fixture_bytes);
	struct text_tokenizer *tokenizer = NULL;
	if (tables == NULL || fixture == NULL ||
	    tokenizer_open(tables, tables_bytes, &tokenizer) != STATUS_OK) {
		free(tables);
		free(fixture);
		return EXIT_FAILURE;
	}
	struct fixture_reader reader = { fixture, fixture_bytes, 0, false };
	const void *const magic = take(&reader, GOLDEN_MAGIC_BYTES);
	const uint32_t cases = take_integer(&reader);
	expect(magic != NULL &&
		       memcmp(magic, GOLDEN_MAGIC, GOLDEN_MAGIC_BYTES) == 0 &&
		       cases > 0,
	       "official image fixture header");
	for (uint32_t index = 0;
	     index < cases && !reader.failed && failures == 0; index++) {
		compare_case(tokenizer, &reader);
	}
	expect(!reader.failed && reader.used == reader.count,
	       "complete official image fixture consumed");
	tokenizer_close(tokenizer);
	free(tables);
	free(fixture);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
