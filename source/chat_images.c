/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Bounded data URL admission and request-owned image preparation. Contract: chat_images.h. */
#include "anemone/chat_images.h"

#include <stdlib.h>
#include <string.h>

#include "anemone/chat_completions.h"
#include "anemone/chat_template.h"
#include "anemone/hooks.h"
#include "json.h"

#define PNG_DATA_PREFIX "data:image/png;base64,"
#define JPEG_DATA_PREFIX "data:image/jpeg;base64,"
#define BASE64_GROUP_BYTES 4u
#define BASE64_OUTPUT_BYTES 3u
#define BASE64_LOW_BITS 15u
#define BASE64_PAD_BITS 3u
#define BASE64_FIRST_SHIFT 2u
#define BASE64_SECOND_SHIFT 4u
#define BASE64_THIRD_SHIFT 6u
static const char base64_alphabet[] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static uint32_t image_internal_failure(struct chat_refusal *refusal)
{
	if (refusal != NULL) {
		*refusal = (struct chat_refusal){ .status = HTTP_SERVER_ERROR };
	}
	return HTTP_SERVER_ERROR;
}

static uint32_t image_refuse(struct chat_refusal *refusal, const char *code,
			     const char *message)
{
	*refusal =
		(struct chat_refusal){ HTTP_BAD_REQUEST, message, code, 0, 0 };
	return HTTP_BAD_REQUEST;
}

static uint32_t base64_digit(char letter)
{
	const char *const found =
		letter == '\0' ? NULL : strchr(base64_alphabet, letter);
	return found != NULL ? (uint32_t)(found - base64_alphabet) : UINT32_MAX;
}

uint32_t chat_image_url_decode(const char *url, size_t url_bytes,
			       uint8_t *decoded, size_t decoded_capacity,
			       size_t *byte_count_out,
			       struct chat_refusal *refusal_out)
{
	if (url == NULL || byte_count_out == NULL || refusal_out == NULL) {
		return image_internal_failure(refusal_out);
	}
	if ((url_bytes >= strlen("http://") &&
	     memcmp(url, "http://", strlen("http://")) == 0) ||
	    (url_bytes >= strlen("https://") &&
	     memcmp(url, "https://", strlen("https://")) == 0)) {
		return image_refuse(refusal_out, CHAT_IMAGE_URL_REMOTE,
				    "remote image URLs are not fetched");
	}
	size_t prefix = 0;
	if (url_bytes >= strlen(PNG_DATA_PREFIX) &&
	    memcmp(url, PNG_DATA_PREFIX, strlen(PNG_DATA_PREFIX)) == 0) {
		prefix = strlen(PNG_DATA_PREFIX);
	} else if (url_bytes >= strlen(JPEG_DATA_PREFIX) &&
		   memcmp(url, JPEG_DATA_PREFIX, strlen(JPEG_DATA_PREFIX)) ==
			   0) {
		prefix = strlen(JPEG_DATA_PREFIX);
	} else {
		const bool is_data = url_bytes >= strlen("data:") &&
				     memcmp(url, "data:", strlen("data:")) == 0;
		return image_refuse(
			refusal_out,
			is_data ? CHAT_IMAGE_FORMAT_UNSUPPORTED :
				  CHAT_IMAGE_URL_INVALID,
			"image_url requires data:image/png;base64, or data:image/jpeg;base64,");
	}
	const size_t encoded_count = url_bytes - prefix;
	if (encoded_count == 0 || encoded_count % BASE64_GROUP_BYTES != 0) {
		return image_refuse(refusal_out, CHAT_IMAGE_URL_INVALID,
				    "image base64 has invalid length");
	}
	const size_t padding =
		(url[url_bytes - 1u] == '=' ? 1u : 0u) +
		(url[url_bytes - BASE64_FIRST_SHIFT] == '=' ? 1u : 0u);
	const size_t count =
		encoded_count / BASE64_GROUP_BYTES * BASE64_OUTPUT_BYTES -
		padding;
	if (count > (size_t)IMAGE_MAX_BYTES) {
		return image_refuse(refusal_out, CHAT_IMAGE_BYTES_LIMIT,
				    "an image exceeds ANEMONE_IMAGE_MAX_BYTES");
	}
	if (decoded != NULL && count > decoded_capacity) {
		return image_internal_failure(refusal_out);
	}
	size_t written = 0;
	for (size_t cursor = prefix; cursor < url_bytes;
	     cursor += BASE64_GROUP_BYTES) {
		const uint32_t first = base64_digit(url[cursor]);
		const uint32_t second = base64_digit(url[cursor + 1u]);
		const uint32_t third =
			url[cursor + BASE64_FIRST_SHIFT] == '=' ?
				0 :
				base64_digit(url[cursor + BASE64_FIRST_SHIFT]);
		const uint32_t fourth =
			url[cursor + BASE64_OUTPUT_BYTES] == '=' ?
				0 :
				base64_digit(url[cursor + BASE64_OUTPUT_BYTES]);
		const bool final = cursor + BASE64_GROUP_BYTES == url_bytes;
		const bool third_padding = url[cursor + BASE64_FIRST_SHIFT] ==
					   '=';
		const bool fourth_padding = url[cursor + BASE64_OUTPUT_BYTES] ==
					    '=';
		if (first == UINT32_MAX || second == UINT32_MAX ||
		    third == UINT32_MAX || fourth == UINT32_MAX ||
		    ((third_padding || fourth_padding) && !final) ||
		    (third_padding && !fourth_padding) ||
		    (third_padding && (second & BASE64_LOW_BITS) != 0) ||
		    (fourth_padding && !third_padding &&
		     (third & BASE64_PAD_BITS) != 0)) {
			return image_refuse(refusal_out, CHAT_IMAGE_URL_INVALID,
					    "image base64 is malformed");
		}
		if (decoded != NULL) {
			decoded[written] =
				(uint8_t)((first << BASE64_FIRST_SHIFT) |
					  (second >> BASE64_SECOND_SHIFT));
			if (written + 1u < count) {
				decoded[written + 1u] =
					(uint8_t)((second
						   << BASE64_SECOND_SHIFT) |
						  (third >>
						   BASE64_FIRST_SHIFT));
			}
			if (written + BASE64_FIRST_SHIFT < count) {
				decoded[written + BASE64_FIRST_SHIFT] =
					(uint8_t)((third
						   << BASE64_THIRD_SHIFT) |
						  fourth);
			}
		}
		written += BASE64_OUTPUT_BYTES;
	}
	*byte_count_out = count;
	return 0;
}

void chat_images_destroy(struct chat_images *images)
{
	if (images != NULL) {
		for (uint32_t index = 0;
		     index < images->count && index < IMAGE_MAX_IMAGES;
		     index++) {
			free(images->images[index].bytes);
			free(images->images[index].patches);
		}
		*images = (struct chat_images){ 0 };
	}
}

uint32_t chat_images_prepare(const char *messages_json,
			     size_t messages_byte_count,
			     const struct chat_template_spans *rendered,
			     struct chat_images *images_out,
			     struct chat_refusal *refusal_out)
{
	if (messages_json == NULL || rendered == NULL || images_out == NULL ||
	    refusal_out == NULL || rendered->image_count > IMAGE_MAX_IMAGES) {
		return image_internal_failure(refusal_out);
	}
	struct chat_images images = { 0 };
	uint32_t result = 0;
	/* Validate every URL before allocating a decoded image or invoking a decoder. */
	struct json_bytes urls[IMAGE_MAX_IMAGES] = { { 0 } };
	size_t byte_counts[IMAGE_MAX_IMAGES] = { 0 };
	char *const scratch_bytes =
		rendered->image_count == 0 ? NULL :
		FAULT_POINT(chat_memory)   ? NULL :
					     malloc(messages_byte_count);
	if (rendered->image_count != 0 && scratch_bytes == NULL) {
		return image_internal_failure(refusal_out);
	}
	struct json_scratch scratch = { scratch_bytes, messages_byte_count, 0,
					false };
	for (uint32_t index = 0; index < rendered->image_count && result == 0;
	     index++) {
		const struct chat_template_image source =
			rendered->images[index];
		if (source.url_begin >= source.url_end ||
		    source.url_end > messages_byte_count ||
		    messages_json[source.url_begin] != '"') {
			result = HTTP_SERVER_ERROR;
			break;
		}
		urls[index] = json_unescape(
			&scratch,
			(struct json_value){ messages_json, source.url_begin,
					     source.url_end });
		if (scratch.is_short) {
			result = HTTP_SERVER_ERROR;
			break;
		}
		result = chat_image_url_decode(urls[index].data,
					       urls[index].count, NULL, 0,
					       &byte_counts[index],
					       refusal_out);
	}
	for (uint32_t index = 0; index < rendered->image_count && result == 0;
	     index++) {
		struct chat_image_input *const input = &images.images[index];
		images.count++;
		input->byte_count = byte_counts[index];
		input->bytes = FAULT_POINT(chat_memory) ?
				       NULL :
				       malloc(input->byte_count);
		if (input->bytes == NULL) {
			result = HTTP_SERVER_ERROR;
			break;
		}
		result = chat_image_url_decode(urls[index].data,
					       urls[index].count, input->bytes,
					       input->byte_count,
					       &input->byte_count, refusal_out);
		if (result != 0) {
			break;
		}
		struct image_rgb rgb = { 0 };
		result_status status =
			image_decode(input->bytes, input->byte_count, &rgb);
		if (status == STATUS_OK) {
			status =
				image_grid(rgb.width, rgb.height, &input->plan);
		}
		if (status == STATUS_OK) {
			input->patches =
				FAULT_POINT(chat_memory) ?
					NULL :
					malloc(input->plan.patch_elements *
					       sizeof(*input->patches));
			status = input->patches == NULL ?
					 STATUS_OUT_OF_MEMORY :
					 image_preprocess(
						 &rgb, &input->plan,
						 input->patches,
						 input->plan.patch_elements);
		}
		image_rgb_destroy(&rgb);
		if (status == STATUS_OUT_OF_MEMORY) {
			result = HTTP_SERVER_ERROR;
		} else if (status != STATUS_OK) {
			result = image_refuse(
				refusal_out,
				status == STATUS_UNSUPPORTED ?
					CHAT_IMAGE_FORMAT_UNSUPPORTED :
					CHAT_IMAGE_DATA_INVALID,
				status == STATUS_UNSUPPORTED ?
					"the image format or JPEG profile is unsupported" :
					"the image is corrupt or exceeds decoded image bounds");
		}
	}
	free(scratch_bytes);
	if (result != 0) {
		chat_images_destroy(&images);
		return result == HTTP_SERVER_ERROR ?
			       image_internal_failure(refusal_out) :
			       result;
	}
	*images_out = images;
	return 0;
}

result_status chat_images_expand(struct chat_images *images, uint32_t *tokens,
				 uint32_t token_capacity, uint32_t *token_count)
{
	if (images == NULL || tokens == NULL || token_count == NULL ||
	    *token_count > token_capacity || images->count > IMAGE_MAX_IMAGES) {
		return STATUS_INVALID_ARGUMENT;
	}
	uint32_t found = 0, expanded_count = *token_count;
	uint32_t placeholder_positions[IMAGE_MAX_IMAGES] = { 0 };
	for (uint32_t index = 0; index < *token_count; index++) {
		if (tokens[index] == CHAT_IMAGE_TOKEN_ID) {
			if (found >= images->count ||
			    found >= IMAGE_MAX_IMAGES ||
			    images->images[found].plan.token_count == 0 ||
			    images->images[found].plan.token_count - 1u >
				    token_capacity - expanded_count) {
				return STATUS_UNSUPPORTED;
			}
			placeholder_positions[found] = index;
			expanded_count +=
				images->images[found].plan.token_count - 1u;
			found++;
		}
	}
	if (found != images->count) {
		return STATUS_INVALID_ARGUMENT;
	}
	uint32_t source = *token_count, destination = expanded_count;
	for (uint32_t remaining = images->count;
	     remaining > 0 && remaining <= IMAGE_MAX_IMAGES; remaining--) {
		const uint32_t image_index = remaining - 1u;
		struct chat_image_input *const image =
			&images->images[image_index];
		const uint32_t position = placeholder_positions[image_index];
		const uint32_t trailing = source - position - 1u;
		destination -= trailing;
		memmove(tokens + destination, tokens + position + 1u,
			(size_t)trailing * sizeof(*tokens));
		const uint32_t begin = destination - image->plan.token_count;
		while (destination > begin) {
			tokens[--destination] = CHAT_IMAGE_TOKEN_ID;
		}
		image->token_begin = begin;
		source = position;
	}
	*token_count = expanded_count;
	return STATUS_OK;
}

#define SHA_WORD_BYTES 4u
#define SHA_BLOCK_BYTES 64u
#define SHA_STATE_WORDS 8u
#define SHA_BLOCK_WORDS 16u
#define SHA_ROUNDS 64u
#define SHA_WORD_BITS 32u
#define SHA_BYTE_BITS 8u
#define SHA_LENGTH_BYTES 8u
#define SHA_LENGTH_OFFSET 56u
#define SHA_PAD_BYTE 0x80u
#define SHA_BYTE_MASK 0xffu
#define SHA_SMALL_FIRST_ROTATE 7u
#define SHA_SMALL_SECOND_ROTATE 18u
#define SHA_SMALL_SHIFT 3u
#define SHA_SMALL_THIRD_ROTATE 17u
#define SHA_SMALL_FOURTH_ROTATE 19u
#define SHA_SMALL_SECOND_SHIFT 10u
#define SHA_LARGE_FIRST_ROTATE 6u
#define SHA_LARGE_SECOND_ROTATE 11u
#define SHA_LARGE_THIRD_ROTATE 25u
#define SHA_LARGE_FOURTH_ROTATE 2u
#define SHA_LARGE_FIFTH_ROTATE 13u
#define SHA_LARGE_SIXTH_ROTATE 22u
#define SHA_SCHEDULE_FIRST_OFFSET 2u
#define SHA_SCHEDULE_SECOND_OFFSET 7u
#define SHA_SCHEDULE_THIRD_OFFSET 15u
#define SHA_STATE_LAST 7u
#define SHA_STATE_CHOOSE 4u
#define SHA_STATE_CHOOSE_SECOND 5u
#define SHA_STATE_CHOOSE_THIRD 6u
#define SHA_STATE_MAJORITY_SECOND 2u
/* Increment the encoder revision whenever image preprocessing or vision math can change cached LLM state.
 * Text-only identities return the unchanged all-zero key before hashing this revisioned domain. */
#define IMAGE_IDENTITY_DOMAIN "anemone-image-prefix-v2"

static const uint32_t sha_round_constants[SHA_ROUNDS] = {
	0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
	0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
	0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
	0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
	0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
	0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
	0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
	0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
	0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
	0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
	0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
	0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
	0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};
static const uint32_t sha_initial_state[SHA_STATE_WORDS] = {
	0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
	0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

struct image_sha {
	uint32_t state[SHA_STATE_WORDS];
	uint8_t block[SHA_BLOCK_BYTES];
	size_t used;
	uint64_t byte_count;
};

static uint32_t sha_rotate(uint32_t word, uint32_t shift)
{
	return (word >> shift) | (word << (SHA_WORD_BITS - shift));
}

static void sha_block(struct image_sha *hash)
{
	uint32_t schedule[SHA_ROUNDS] = { 0 }, state[SHA_STATE_WORDS] = { 0 };
	for (uint32_t index = 0; index < SHA_BLOCK_WORDS; index++) {
		for (uint32_t byte = 0; byte < SHA_WORD_BYTES; byte++) {
			schedule[index] =
				(schedule[index] << SHA_BYTE_BITS) |
				hash->block[index * SHA_WORD_BYTES + byte];
		}
	}
	for (uint32_t index = SHA_BLOCK_WORDS; index < SHA_ROUNDS; index++) {
		const uint32_t first =
			schedule[index - SHA_SCHEDULE_THIRD_OFFSET];
		const uint32_t second =
			schedule[index - SHA_SCHEDULE_FIRST_OFFSET];
		const uint32_t first_sigma =
			sha_rotate(first, SHA_SMALL_FIRST_ROTATE) ^
			sha_rotate(first, SHA_SMALL_SECOND_ROTATE) ^
			(first >> SHA_SMALL_SHIFT);
		const uint32_t second_sigma =
			sha_rotate(second, SHA_SMALL_THIRD_ROTATE) ^
			sha_rotate(second, SHA_SMALL_FOURTH_ROTATE) ^
			(second >> SHA_SMALL_SECOND_SHIFT);
		schedule[index] = schedule[index - SHA_BLOCK_WORDS] +
				  first_sigma +
				  schedule[index - SHA_SCHEDULE_SECOND_OFFSET] +
				  second_sigma;
	}
	memcpy(state, hash->state, sizeof(state));
	for (uint32_t round = 0; round < SHA_ROUNDS; round++) {
		const uint32_t choose_word = state[SHA_STATE_CHOOSE];
		const uint32_t choose =
			(choose_word & state[SHA_STATE_CHOOSE_SECOND]) ^
			(~choose_word & state[SHA_STATE_CHOOSE_THIRD]);
		const uint32_t large_first =
			sha_rotate(choose_word, SHA_LARGE_FIRST_ROTATE) ^
			sha_rotate(choose_word, SHA_LARGE_SECOND_ROTATE) ^
			sha_rotate(choose_word, SHA_LARGE_THIRD_ROTATE);
		const uint32_t first = state[SHA_STATE_LAST] + large_first +
				       choose + sha_round_constants[round] +
				       schedule[round];
		const uint32_t large_second =
			sha_rotate(state[0], SHA_LARGE_FOURTH_ROTATE) ^
			sha_rotate(state[0], SHA_LARGE_FIFTH_ROTATE) ^
			sha_rotate(state[0], SHA_LARGE_SIXTH_ROTATE);
		const uint32_t majority =
			(state[0] & state[1]) ^
			(state[0] & state[SHA_STATE_MAJORITY_SECOND]) ^
			(state[1] & state[SHA_STATE_MAJORITY_SECOND]);
		for (uint32_t index = SHA_STATE_LAST; index > 0; index--) {
			state[index] = state[index - 1u];
		}
		state[SHA_STATE_CHOOSE] += first;
		state[0] = first + large_second + majority;
	}
	for (uint32_t index = 0; index < SHA_STATE_WORDS; index++) {
		hash->state[index] += state[index];
	}
	hash->used = 0;
}

static void sha_update(struct image_sha *hash, const uint8_t *bytes,
		       size_t byte_count)
{
	hash->byte_count += byte_count;
	while (byte_count > 0) {
		const size_t available = SHA_BLOCK_BYTES - hash->used;
		const size_t count = byte_count < available ? byte_count :
							      available;
		memcpy(hash->block + hash->used, bytes, count);
		hash->used += count;
		bytes += count;
		byte_count -= count;
		if (hash->used == SHA_BLOCK_BYTES) {
			sha_block(hash);
		}
	}
}

static void sha_integer(struct image_sha *hash, uint64_t value)
{
	uint8_t bytes[SHA_LENGTH_BYTES] = { 0 };
	for (uint32_t index = 0; index < SHA_LENGTH_BYTES; index++) {
		bytes[index] = (uint8_t)(value & SHA_BYTE_MASK);
		value >>= SHA_BYTE_BITS;
	}
	sha_update(hash, bytes, sizeof(bytes));
}

result_status
chat_images_identity(const struct chat_images *images, uint64_t math_identity,
		     uint8_t identity_out[CHAT_IMAGE_IDENTITY_BYTES])
{
	if (images == NULL || identity_out == NULL ||
	    images->count > IMAGE_MAX_IMAGES) {
		return STATUS_INVALID_ARGUMENT;
	}
	for (uint32_t index = 0; index < images->count; index++) {
		if (images->images[index].bytes == NULL ||
		    images->images[index].byte_count == 0 ||
		    images->images[index].byte_count >
			    (size_t)IMAGE_MAX_BYTES) {
			return STATUS_INVALID_ARGUMENT;
		}
	}
	if (images->count == 0) {
		memset(identity_out, 0, CHAT_IMAGE_IDENTITY_BYTES);
		return STATUS_OK;
	}
	struct image_sha hash = { 0 };
	memcpy(hash.state, sha_initial_state, sizeof(hash.state));
	sha_update(&hash, (const uint8_t *)IMAGE_IDENTITY_DOMAIN,
		   strlen(IMAGE_IDENTITY_DOMAIN));
	sha_integer(&hash, math_identity);
	sha_integer(&hash, images->count);
	for (uint32_t index = 0; index < images->count; index++) {
		const struct chat_image_input *const image =
			&images->images[index];
		sha_integer(&hash, image->token_begin);
		sha_integer(&hash, image->byte_count);
		sha_update(&hash, image->bytes, image->byte_count);
	}
	const uint64_t bit_count = hash.byte_count * SHA_BYTE_BITS;
	hash.block[hash.used++] = SHA_PAD_BYTE;
	if (hash.used > SHA_LENGTH_OFFSET) {
		memset(hash.block + hash.used, 0, SHA_BLOCK_BYTES - hash.used);
		sha_block(&hash);
	}
	memset(hash.block + hash.used, 0, SHA_LENGTH_OFFSET - hash.used);
	for (uint32_t index = 0; index < SHA_LENGTH_BYTES; index++) {
		hash.block[SHA_BLOCK_BYTES - 1u - index] =
			(uint8_t)(bit_count >> (index * SHA_BYTE_BITS));
	}
	sha_block(&hash);
	for (uint32_t index = 0; index < CHAT_IMAGE_IDENTITY_BYTES; index++) {
		identity_out[index] =
			(uint8_t)(hash.state[index / SHA_WORD_BYTES] >>
				  ((SHA_WORD_BYTES - 1u -
				    index % SHA_WORD_BYTES) *
				   SHA_BYTE_BITS));
	}
	return STATUS_OK;
}
