/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Scalar ViT oracles: FP32 online Flash state and BF16 storage boundaries; a separate double mathematics control. */
#ifndef TEST_VISION_REFERENCE_H
#define TEST_VISION_REFERENCE_H

#include <stdbool.h>
#include "anemone/vision.h"

void *test_vision_at(unsigned char *arena, uint64_t byte_offset);
const void *test_vision_const_at(const unsigned char *arena,
				 uint64_t byte_offset);
float test_vision_float(uint16_t bits);
uint16_t test_vision_bfloat(float value);
bool test_vision_math_reference(const struct kernel_launch *launches,
				uint32_t count, unsigned char *arena);
bool test_vision_reference(const struct kernel_launch *launches, uint32_t count,
			   unsigned char *arena);

#endif
