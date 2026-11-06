/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image URL admission, prompt provenance, typed-span expansion and content-bound prefix identity. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_seam.h"
#include "anemone/chat_completions.h"
#include "anemone/chat_images.h"
#include "anemone/chat_template.h"
#include "anemone/hooks.h"

#define PNG_URL "data:image/png;base64,AA=="
#define JPEG_URL "data:image/jpeg;base64,AQ=="
#define IMAGE_PART \
	"{\"type\":\"image_url\",\"image_url\":{\"url\":\"" PNG_URL "\"}}"
#define PLACEHOLDER "<｜deepseek_image｜>"
#define TEXT_CAPACITY 8192u
#define URL_GROUP_BYTES 4u
#define URL_OUTPUT_BYTES 3u
#define TEST_TOKENS 3u
#define TOKEN_CAPACITY 16u
#define TEST_SPAN_FIRST 4u
#define TEST_SPAN_SECOND 5u
#define SOURCE_TOKENS 5u
#define EXPANDED_TOKENS 12u
#define FIRST_IMAGE_START 1u
#define SECOND_IMAGE_START 6u
#define SECOND_IMAGE_INDEX 1u
#define LAST_TOKEN_INDEX 11u
#define FIRST_TEXT_TOKEN 17u
#define MIDDLE_TEXT_TOKEN 18u
#define LAST_TEXT_TOKEN 19u
#define HASH_OFFSET 7u
#define HASH_WORDS 32u
#define TEST_EFFORT 75u
#define ONE_IMAGE 1u
#define TWO_IMAGES 2u
#define INVALID_COUNT 99u
#define API_DEFAULT_LIMIT 8u

static int failures;

static void expect(bool condition, const char *description)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", description);
		failures++;
	}
}

static void test_urls(void)
{
	static const struct {
		const char *url, *code;
	} refused[] = {
		{ "https://example.invalid/image.png", CHAT_IMAGE_URL_REMOTE },
		{ "http://example.invalid/image.jpg", CHAT_IMAGE_URL_REMOTE },
		{ "/tmp/image.png", CHAT_IMAGE_URL_INVALID },
		{ "file:///tmp/image.png", CHAT_IMAGE_URL_INVALID },
		{ "data:image/gif;base64,AA==", CHAT_IMAGE_FORMAT_UNSUPPORTED },
		{ "data:image/webp;base64,AA==",
		  CHAT_IMAGE_FORMAT_UNSUPPORTED },
		{ "data:image/png;base64,", CHAT_IMAGE_URL_INVALID },
		{ "data:image/png;base64,A", CHAT_IMAGE_URL_INVALID },
		{ "data:image/png;base64,AB==", CHAT_IMAGE_URL_INVALID },
		{ "data:image/png;base64,AAF=", CHAT_IMAGE_URL_INVALID },
		{ "data:image/png;base64,AA=A", CHAT_IMAGE_URL_INVALID },
		{ "data:image/png;base64,AA==AA==", CHAT_IMAGE_URL_INVALID },
		{ "data:image/png;base64,AA\n=", CHAT_IMAGE_URL_INVALID },
		{ "data:image/png;base64,!!!!", CHAT_IMAGE_URL_INVALID }
	};
	for (size_t index = 0; index < sizeof(refused) / sizeof(refused[0]);
	     index++) {
		struct chat_refusal refusal = { 0 };
		size_t count = INVALID_COUNT;
		expect(chat_image_url_decode(
			       refused[index].url, strlen(refused[index].url),
			       NULL, 0, &count, &refusal) == HTTP_BAD_REQUEST &&
			       refusal.code != NULL &&
			       strcmp(refusal.code, refused[index].code) == 0 &&
			       count == INVALID_COUNT,
		       "malformed and remote image URL names its refusal without changing count");
	}
	uint8_t bytes[URL_OUTPUT_BYTES] = { 1 };
	struct chat_refusal refusal = { 0 };
	size_t count = 0;
	expect(chat_image_url_decode(PNG_URL, strlen(PNG_URL), bytes,
				     sizeof(bytes), &count, &refusal) == 0 &&
		       count == 1u && bytes[0] == 0,
	       "PNG data URL validates and decodes base64");
	expect(chat_image_url_decode(JPEG_URL, strlen(JPEG_URL), bytes,
				     sizeof(bytes), &count, &refusal) == 0 &&
		       count == 1u && bytes[0] == 1u,
	       "JPEG data URL validates and decodes base64");
	const size_t encoded = ((size_t)IMAGE_MAX_BYTES + 1u) /
			       URL_OUTPUT_BYTES * URL_GROUP_BYTES;
	const size_t prefix = strlen("data:image/png;base64,");
	char *const url = malloc(prefix + encoded + 1u);
	expect(url != NULL, "allocate bounded oversized data URL fixture");
	if (url != NULL) {
		memcpy(url, "data:image/png;base64,", prefix);
		memset(url + prefix, 'A', encoded);
		url[prefix + encoded] = '\0';
		expect(chat_image_url_decode(url, prefix + encoded, NULL, 0,
					     &count,
					     &refusal) == HTTP_BAD_REQUEST &&
			       strcmp(refusal.code, CHAT_IMAGE_BYTES_LIMIT) ==
				       0,
		       "decoded bytes bound is checked before any base64 decoding");
		url[prefix + encoded - 1u] = '=';
		expect(chat_image_url_decode(url, prefix + encoded, NULL, 0,
					     &count, &refusal) == 0 &&
			       count == (size_t)IMAGE_MAX_BYTES,
		       "image bytes exactly at the named bound are admitted");
		free(url);
	}
}

static void test_request_images(void)
{
	static const char *const roles[] = { "user", "tool", "system",
					     "developer", "assistant" };
	const struct chat_request defaults = { .max_tokens = API_DEFAULT_LIMIT,
					       .is_seeded = true };
	for (size_t index = 0; index < sizeof(roles) / sizeof(roles[0]);
	     index++) {
		char body[TEXT_CAPACITY] = { 0 };
		const int length =
			snprintf(body, sizeof(body),
				 "{\"messages\":[{\"content\":[" IMAGE_PART
				 "],\"role\":\"%s\"}]}",
				 roles[index]);
		struct chat_request request = { 0 };
		struct chat_refusal refusal = { 0 };
		const bool allowed = strcmp(roles[index], "user") == 0 ||
				     strcmp(roles[index], "tool") == 0;
		const uint32_t status =
			length <= 0 ? HTTP_SERVER_ERROR :
				      chat_request_read(body, (size_t)length,
							"test", NULL, &defaults,
							&request, &refusal);
		expect(allowed ?
			       status == 0 && request.image_count == ONE_IMAGE :
			       status == HTTP_BAD_REQUEST &&
				       refusal.code != NULL &&
				       strcmp(refusal.code,
					      CHAT_IMAGE_ROLE_INVALID) == 0,
		       "images are admitted only in user and tool messages, independent of member order");
	}
	char body[TEXT_CAPACITY] =
		"{\"messages\":[{\"role\":\"user\",\"content\":[";
	size_t used = strlen(body);
	for (uint32_t index = 0; index <= IMAGE_MAX_IMAGES; index++) {
		const char *const separator = index == 0 ? "" : ",";
		memcpy(body + used, separator, strlen(separator) + 1u);
		used += strlen(separator);
		memcpy(body + used, IMAGE_PART, strlen(IMAGE_PART) + 1u);
		used += strlen(IMAGE_PART);
	}
	memcpy(body + used, "]}]}", strlen("]}]}") + 1u);
	used += strlen("]}]}");
	struct chat_request request = { 0 };
	struct chat_refusal refusal = { 0 };
	expect(chat_request_read(body, used, "test", NULL, &defaults, &request,
				 &refusal) == HTTP_BAD_REQUEST &&
		       refusal.code != NULL &&
		       strcmp(refusal.code, CHAT_IMAGE_COUNT_LIMIT) == 0,
	       "request image count is bounded before a decoder is called");
}

static void test_template_order(void)
{
	static const char messages[] =
		"[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":["
		"{\"id\":\"b\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}},"
		"{\"id\":\"a\",\"type\":\"function\",\"function\":{\"name\":\"g\",\"arguments\":\"{}\"}}]},"
		"{\"role\":\"tool\",\"tool_call_id\":\"a\",\"content\":[" IMAGE_PART
		"]},"
		"{\"role\":\"tool\",\"tool_call_id\":\"b\",\"content\":["
		"{\"type\":\"image_url\",\"image_url\":\"" JPEG_URL "\"}]}]";
	const struct chat_template_options options = { .reasoning_effort =
							       TEST_EFFORT };
	char prompt[TEXT_CAPACITY] = { 0 };
	size_t prompt_bytes = 0;
	struct chat_template_spans trusted = { 0 };
	const result_status status = chat_template_render_spans(
		messages, strlen(messages), NULL, 0, &options, prompt,
		sizeof(prompt), &prompt_bytes, &trusted);
	expect(status == STATUS_OK && trusted.image_count == TWO_IMAGES,
	       "tool result images have a template origin record");
	if (status == STATUS_OK && trusted.image_count == TWO_IMAGES) {
		expect(memcmp(messages + trusted.images[0].url_begin + 1u,
			      JPEG_URL, strlen(JPEG_URL)) == 0 &&
			       memcmp(messages +
					      trusted.images[SECOND_IMAGE_INDEX]
						      .url_begin +
					      1u,
				      PNG_URL, strlen(PNG_URL)) == 0,
		       "image sources follow the sorted tool results rather than their wire order");
		for (uint32_t index = 0; index < trusted.image_count; index++) {
			const uint32_t offset =
				trusted.images[index].prompt_byte_offset;
			expect(offset + strlen(PLACEHOLDER) <= prompt_bytes &&
				       memcmp(prompt + offset, PLACEHOLDER,
					      strlen(PLACEHOLDER)) == 0,
			       "image source offset points to its trusted placeholder");
		}
	}
	free(trusted.spans);
}

static void test_prepare_failures(void)
{
	static const char messages[] =
		"[{\"role\":\"user\",\"content\":[" IMAGE_PART "]}]";
	const struct chat_template_options options = { .reasoning_effort =
							       TEST_EFFORT };
	char prompt[TEXT_CAPACITY] = { 0 };
	size_t prompt_bytes = 0;
	struct chat_template_spans trusted = { 0 };
	const result_status status = chat_template_render_spans(
		messages, strlen(messages), NULL, 0, &options, prompt,
		sizeof(prompt), &prompt_bytes, &trusted);
	expect(status == STATUS_OK, "prepare failure fixture renders");
	if (status != STATUS_OK) {
		return;
	}
	struct chat_images images = { 0 };
	struct chat_refusal refusal = { 0 };
	expect(chat_images_prepare(messages, strlen(messages), &trusted,
				   &images, &refusal) == HTTP_BAD_REQUEST &&
		       refusal.code != NULL && images.count == 0,
	       "corrupt decoded bytes give a named 400 and no prepared images");
#if CHECKING_HOOKS_ENABLED
	for (uint32_t occurrence = 1; occurrence <= TWO_IMAGES; occurrence++) {
		refusal = (struct chat_refusal){ 0 };
		expect(fault_arm("chat_memory", occurrence),
		       "arm image prepare allocation failure");
		const uint32_t failed =
			chat_images_prepare(messages, strlen(messages),
					    &trusted, &images, &refusal);
		expect(failed == HTTP_SERVER_ERROR &&
			       refusal.status == failed && images.count == 0,
		       "every image prepare allocation failure sets HTTP500 and releases partial inputs");
		expect(fault_disarm() == occurrence,
		       "allocation failure point was reached");
	}
#endif
	chat_images_destroy(&images);
	free(trusted.spans);
}

static void test_span_expansion(void)
{
	struct chat_images images = { .count = TWO_IMAGES };
	images.images[0].plan.token_count = TEST_SPAN_FIRST;
	images.images[SECOND_IMAGE_INDEX].plan.token_count = TEST_SPAN_SECOND;
	uint32_t tokens[TOKEN_CAPACITY] = {
		FIRST_TEXT_TOKEN, CHAT_IMAGE_TOKEN_ID, MIDDLE_TEXT_TOKEN,
		CHAT_IMAGE_TOKEN_ID, LAST_TEXT_TOKEN
	};
	uint32_t count = SOURCE_TOKENS;
	expect(chat_images_expand(&images, tokens, TOKEN_CAPACITY, &count) ==
			       STATUS_OK &&
		       count == EXPANDED_TOKENS &&
		       images.images[0].token_begin == FIRST_IMAGE_START &&
		       images.images[SECOND_IMAGE_INDEX].token_begin ==
			       SECOND_IMAGE_START &&
		       tokens[0] == FIRST_TEXT_TOKEN &&
		       tokens[SECOND_IMAGE_START - 1u] == MIDDLE_TEXT_TOKEN &&
		       tokens[LAST_TOKEN_INDEX] == LAST_TEXT_TOKEN,
	       "image token spans expand in reference row order");
	for (uint32_t index = FIRST_IMAGE_START;
	     index < SECOND_IMAGE_START - 1u; index++) {
		expect(tokens[index] == CHAT_IMAGE_TOKEN_ID,
		       "every typed image position retains reference ID");
	}
	uint32_t too_short[TEST_TOKENS] = { FIRST_TEXT_TOKEN,
					    CHAT_IMAGE_TOKEN_ID,
					    LAST_TEXT_TOKEN };
	images.count = ONE_IMAGE;
	count = TEST_TOKENS;
	expect(chat_images_expand(&images, too_short, TEST_TOKENS, &count) ==
			       STATUS_UNSUPPORTED &&
		       count == TEST_TOKENS &&
		       too_short[1] == CHAT_IMAGE_TOKEN_ID,
	       "token capacity fails before changing tokens or their count");
}

static void test_identity(void)
{
	/* hashlib: revision-2 domain, then little-endian uint64 math, count=1, position=7, length=3 and "abc".
	 * The standalone recipe is followup/image_identity_fixture.py; no engine implementation is imported. */
	static const uint64_t math_identities[] = {
		0, ARITHMETIC_CLASS_SM_80, ARITHMETIC_CLASS_SM_120_NATIVE
	};
	static const uint8_t expected[][HASH_WORDS] = {
		{ 0x08, 0x2b, 0x41, 0x23, 0xbc, 0x9d, 0x6c, 0x2c,
		  0x62, 0x22, 0xc6, 0xa3, 0x08, 0xe4, 0x41, 0xba,
		  0xf5, 0xac, 0x91, 0x36, 0x95, 0x0f, 0xde, 0xc7,
		  0x37, 0x3b, 0x64, 0x87, 0x64, 0xc3, 0x33, 0x00 },
		{ 0xff, 0x27, 0x97, 0x98, 0x8f, 0x20, 0x16, 0x7f,
		  0x45, 0x59, 0xa5, 0x05, 0xa1, 0x0a, 0x4b, 0xe6,
		  0xbb, 0xe4, 0x9d, 0xa3, 0x28, 0x63, 0x7e, 0x9a,
		  0xad, 0x63, 0xb5, 0xcb, 0x4a, 0x84, 0xe5, 0xb0 },
		{ 0xc2, 0x0b, 0x61, 0x06, 0x6e, 0x1c, 0x52, 0x7b,
		  0x12, 0xbb, 0x14, 0x2e, 0x10, 0x4a, 0xbd, 0x26,
		  0x59, 0x2d, 0xf6, 0xdf, 0x9a, 0xcb, 0x65, 0xc0,
		  0x0e, 0x8d, 0x91, 0xa7, 0x1b, 0x9e, 0x3d, 0x60 }
	};
	uint8_t bytes[] = { 'a', 'b', 'c' };
	struct chat_images images = { .count = ONE_IMAGE };
	images.images[0].bytes = bytes;
	images.images[0].byte_count = sizeof(bytes);
	images.images[0].token_begin = HASH_OFFSET;
	uint8_t first[CHAT_IMAGE_IDENTITY_BYTES] = { 0 },
		second[CHAT_IMAGE_IDENTITY_BYTES] = { 0 };
	expect(chat_images_identity(&images, 0, first) == STATUS_OK &&
		       memcmp(first, expected[0], sizeof(first)) == 0,
	       "SHA256 matches the independent hashlib fixture");
	for (size_t index = 1;
	     index < sizeof(math_identities) / sizeof(math_identities[0]);
	     index++) {
		expect(chat_images_identity(&images, math_identities[index],
					    second) == STATUS_OK &&
			       memcmp(second, expected[index],
				      sizeof(second)) == 0 &&
			       memcmp(first, second, sizeof(first)) != 0,
		       "encoder arithmetic changes image identity and matches the independent known answer");
	}
	bytes[0] = 'd';
	expect(chat_images_identity(&images, 0, second) == STATUS_OK &&
		       memcmp(first, second, sizeof(first)) != 0,
	       "different image bytes change the prefix identity");
	bytes[0] = 'a';
	images.images[0].token_begin++;
	expect(chat_images_identity(&images, 0, second) == STATUS_OK &&
		       memcmp(first, second, sizeof(first)) != 0,
	       "moving an image changes the prefix identity");
	images.count = 0;
	memset(second, 0, sizeof(second));
	for (size_t index = 0;
	     index < sizeof(math_identities) / sizeof(math_identities[0]);
	     index++) {
		memset(first, 1, sizeof(first));
		expect(chat_images_identity(&images, math_identities[index],
					    first) == STATUS_OK &&
			       memcmp(first, second, sizeof(first)) == 0,
		       "text-only identity preserves the zero key across encoder arithmetic");
	}
}

int main(void)
{
	test_urls();
	test_request_images();
	test_template_order();
	test_prepare_failures();
	test_span_expansion();
	test_identity();
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
