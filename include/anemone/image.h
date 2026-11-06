/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Bounded image decoding and reference BF16 patch preparation. Host C11 only.
 * Decode accepts PNG (all standard depths/types, Adam7) and 8-bit baseline JPEG.
 * No I/O, external libraries, metadata execution or implicit orientation is performed.
 * Decode owns a fresh RGB8 allocation; destroy releases it. Failed calls leave scalar
 * outputs untouched. Buffers being filled are unspecified after failure. PNG CRC,
 * zlib checksums and all entropy/dimension bounds are checked before success.
 * Grid is the reference min_pixels/max_n_token plan with max_wh_ratio=None.
 * Preprocess uses centered RGB(127) padding, Pillow-compatible bicubic filtering,
 * uint8 rounding after each resize axis, float32 normalization and BF16 ties-to-even.
 */
#ifndef IMAGE_H
#define IMAGE_H
#include <stddef.h>
#include <stdint.h>
#include "anemone/status.h"

#define IMAGE_MAX_BYTES 8388608u /* 8 MiB */
#define IMAGE_MAX_IMAGES 8u
#define IMAGE_MAX_DIMENSION 32768u
#define IMAGE_MAX_PIXELS 16777216u /* 16 Mi pixels */
#define IMAGE_MAX_DECOMPRESSED_BYTES 134217728u /* 128 MiB */
#define IMAGE_PATCH_SIZE 14u
#define IMAGE_CHANNELS 3u
#define IMAGE_DOWNSAMPLE 3u
#define IMAGE_MIN_PIXELS 295936u
#define IMAGE_MAX_TOKENS 1024u
#define IMAGE_TOKEN_ID 129264u
#define IMAGE_PATCH_ELEMENTS 588u /* 3 channels of 14 by 14 pixels */
#define IMAGE_MAX_PATCHES 9216u /* 1024 cells of 3 by 3 patches */

struct image_rgb {
	uint8_t *pixels;
	uint32_t width;
	uint32_t height;
};
struct image_plan {
	uint32_t width;
	uint32_t height;
	uint32_t vit_height;
	uint32_t vit_width;
	uint32_t llm_height;
	uint32_t llm_width;
	uint32_t token_count;
	size_t patch_elements;
};
MUST_CHECK result_status image_decode(const void *bytes, size_t byte_count,
				      struct image_rgb *image_out);
void image_rgb_destroy(struct image_rgb *image);
MUST_CHECK result_status image_grid(uint32_t width, uint32_t height,
				    struct image_plan *plan_out);
MUST_CHECK result_status image_preprocess(const struct image_rgb *image,
					  const struct image_plan *plan,
					  uint16_t *patches_out,
					  size_t patch_capacity);
#endif
