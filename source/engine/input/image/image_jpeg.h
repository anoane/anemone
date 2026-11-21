/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef IMAGE_JPEG_H
#define IMAGE_JPEG_H
#include "image.h"
MUST_CHECK result_status image_jpeg_decode(const uint8_t *bytes,
					   size_t byte_count,
					   struct image_rgb *image_out);
#endif
