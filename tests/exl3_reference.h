/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent double-precision EXL3 oracle: bit-walking ring and forward
 * tensor-core permutation. No device decoder or prepared launch is reused.
 * Widths are stored widths (exl3_records.h): K for a whole K, 16 K for a half K,
 * whose ring steps KA bits at even positions and KA + 1 at odd ones. */
#ifndef TEST_EXL3_REFERENCE_H
#define TEST_EXL3_REFERENCE_H
#include <stdint.h>
uint16_t test_exl3_codebook(uint32_t state);
uint32_t test_exl3_position(uint32_t input, uint32_t output);
/* The 16-bit trellis state of weight (input, output), and its FP16 value; tiles are the stored bytes (little-endian
 * uint32 words) */
uint16_t test_exl3_state(const unsigned char *tiles, uint32_t inputs,
			 uint32_t outputs, uint32_t stored_width,
			 uint32_t input, uint32_t output);
uint16_t test_exl3_reconstruct(const unsigned char *tiles, uint32_t inputs,
			       uint32_t outputs, uint32_t stored_width,
			       uint32_t input, uint32_t output);
#endif
