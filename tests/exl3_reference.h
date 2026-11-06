/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent double-precision EXL3 oracle: bit-walking ring and forward
 * tensor-core permutation. No device decoder or prepared launch is reused. */
#ifndef TEST_EXL3_REFERENCE_H
#define TEST_EXL3_REFERENCE_H
#include <stdint.h>
uint16_t test_exl3_codebook(uint32_t state);
uint32_t test_exl3_position(uint32_t input, uint32_t output);
uint16_t test_exl3_reconstruct(const uint32_t *tiles, uint32_t inputs,
			       uint32_t outputs, uint32_t bits, uint32_t input,
			       uint32_t output);
#endif
