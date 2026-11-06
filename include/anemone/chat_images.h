/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * chat_images.h: bounded data URLs and the request-owned vision inputs in rendered prompt order.
 *
 * URL validation accepts only PNG/JPEG base64 data URLs with canonical padding, never a path or remote URL.
 * A NULL decoded buffer checks the entire URL and its decoded byte bound without decoding or allocating.
 * Prepare receives a checked template result and the same messages JSON. It decodes and preprocesses only after
 * request_read has checked every image's bytes and the total count. Success transfers all input buffers to the
 * caller; destroy releases them. On failure the output record is untouched and refusal names the image error.
 * Expand replaces each trusted image token by its planned span, all positions retaining token id 129264. Literal
 * payload spellings do not produce that added token. The image spans are in order and token_begin is their first
 * position. It validates all capacity/count bounds before changing either the token array or image metadata.
 */
#ifndef CHAT_IMAGES_H
#define CHAT_IMAGES_H

#include <stddef.h>
#include <stdint.h>

#include "anemone/image.h"
#include "anemone/status.h"

#define CHAT_IMAGE_TOKEN_ID 129264u
#define CHAT_IMAGE_IDENTITY_BYTES 32u
#define CHAT_IMAGE_URL_INVALID "invalid_image_url"
#define CHAT_IMAGE_URL_REMOTE "remote_image_url"
#define CHAT_IMAGE_FORMAT_UNSUPPORTED "unsupported_image_format"
#define CHAT_IMAGE_BYTES_LIMIT "image_bytes_limit"
#define CHAT_IMAGE_COUNT_LIMIT "image_count_limit"
#define CHAT_IMAGE_ROLE_INVALID "image_role_not_allowed"
#define CHAT_IMAGE_DATA_INVALID "invalid_image_data"

struct chat_refusal;
struct chat_template_spans;

struct chat_image_input {
	uint8_t *bytes;
	uint16_t *patches;
	size_t byte_count;
	struct image_plan plan;
	uint32_t token_begin;
};

struct chat_images {
	struct chat_image_input images[IMAGE_MAX_IMAGES];
	uint32_t count;
};

MUST_CHECK uint32_t chat_image_url_decode(const char *url, size_t url_bytes,
					  uint8_t *decoded,
					  size_t decoded_capacity,
					  size_t *byte_count_out,
					  struct chat_refusal *refusal_out);
MUST_CHECK uint32_t chat_images_prepare(
	const char *messages_json, size_t messages_byte_count,
	const struct chat_template_spans *rendered,
	struct chat_images *images_out, struct chat_refusal *refusal_out);
MUST_CHECK result_status chat_images_expand(struct chat_images *images,
					    uint32_t *tokens,
					    uint32_t token_capacity,
					    uint32_t *token_count);
/* SHA-256 over the encoder revision domain, math identity, image count, ordered token offsets and
 * length-delimited original bytes. Scalars are uint64 little-endian. math_identity binds the encoder's
 * selected arithmetic class/mode. With no images the identity is all zero for every math identity,
 * preserving existing text-only reuse and snapshot keys. */
MUST_CHECK result_status
chat_images_identity(const struct chat_images *images, uint64_t math_identity,
		     uint8_t identity_out[CHAT_IMAGE_IDENTITY_BYTES]);
void chat_images_destroy(struct chat_images *images);

#endif
