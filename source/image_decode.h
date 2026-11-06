/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef IMAGE_DECODE_H
#define IMAGE_DECODE_H
#include "anemone/image.h"
MUST_CHECK result_status image_jpeg_decode(const uint8_t *bytes,
					   size_t byte_count,
					   struct image_rgb *image_out);
#endif
