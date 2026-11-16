/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_chain_reference.h: the host's EXL3 routed-expert product, the words kernel_exl3.cu's
 * exl3_{gate_up,down}_k{2,3} write (and every form that writes the same: kernel_exl3_wide.cu's products), for the card
 * class whose sums it models (mma_class_reference.h). For output o and column c (a job row):
 *   - slice s of `slices` takes the K tiles from tile_count s / slices to tile_count (s + 1) / slices, tile_count
 *     being inputs / 16;
 *   - each K tile t is one instruction on the slice's running sum (from +0): its 16 products the decoded weights of
 *     inputs 16 t to 16 t + 15 at output o by column c's 16-bit operands of the same inputs;
 *   - the slices' sums are added in slice order from +0 in float32.
 * The weights come decoded: test_exl3_decode_matrix (exl3_reference.h, the bit-walking oracle), [inputs][outputs].
 */
#ifndef TEST_EXL3_CHAIN_REFERENCE_H
#define TEST_EXL3_CHAIN_REFERENCE_H

#include <stdint.h>

/* weights_out: [inputs][outputs] 16-bit floats of the trellis tiles (the stored bytes) of an inputs x outputs matrix
 * of a stored width (exl3_records.h). */
void test_exl3_decode_matrix(const unsigned char *tiles, uint32_t inputs,
			     uint32_t outputs, uint32_t stored_width,
			     uint16_t *weights_out);

/* operands: [columns][inputs] 16-bit floats; products_out: [columns][outputs] float32. */
void test_exl3_chain(uint32_t sum_class, const uint16_t *weights,
		     uint32_t inputs, uint32_t outputs, uint32_t slices,
		     const uint16_t *operands, uint32_t columns,
		     float *products_out);

#endif
