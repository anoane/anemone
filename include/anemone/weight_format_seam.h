/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * weight_format_seam.h: how the bytes of a matrix become weights.
 *
 * Contract:
 *   - A weight format says how many bytes a matrix of a given shape occupies in a pack, in exactly the form the
 *     kernels read (no conversion at load). The engine never asks a format anything per token: the answers are
 *     taken once at load. Which kernels decode a format on a card is the backend's knowledge: backend_seam.h
 *     gains that query (format_supported) with the first linear kernels.
 *   - A shape carries bits_per_weight because one pack mixes widths: this EXL3 implementation stores 2 or 3
 *     bits per weight, chosen per matrix. Every other format has one figure and refuses another.
 *   - An extent is a multiple of the pack's extent alignment. A matrix inside it starts at a multiple of
 *     WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES, with zero fill up to the next one; stored_byte_count is the
 *     bytes of the matrix itself, without that fill.
 *   - Every format has a host reference decode in double precision among the tests, declared in
 *     tests/weight_format_reference.h; it is the oracle of its kernels and no part of the engine.
 *   - A pack stores the values of the enumeration: a value is never reused or renumbered; new formats are
 *     appended.
 * A new format is a new entry here, its kernels in every backend, and its reference decode among the tests.
 *
 * The two formats of DeepSeek's checkpoints, as the kernels read them. Both are in LANE ORDER: the 32 lanes of a
 * warp read 512 consecutive bytes and hold, with no bit moved across lanes, the 16-by-16 weight operand of the
 * tensor-core instruction m16n8k16. Below, lane = 4 g + t with g = 0..7 and t = 0..3; a scale is an E8M0 byte c
 * that stands for 2^(c - 127); every count is of the matrix W[output][input].
 *   WEIGHT_FORMAT_E2M1_SCALED: output_size a multiple of 16, input_size a multiple of 64, bits_per_weight
 *   4. A weight is the checkpoint's 4-bit code (bit 3 the sign; bits 2..0 are 0, 0.5, 1, 1.5, 2, 3, 4, 6), and
 *   the 32 consecutive inputs 32 k .. 32 k + 31 of an output share one scale.
 *     words   uint32 [tile = output / 16][load = input / 64][lane][step 0..3]: the nibble at bit 4 r + 16 h of a
 *             word is W[16 tile + g + 8 (r % 2)][64 load + 16 step + 2 t + 8 (r / 2) + h], r = 0..3, h = 0..1;
 *     scales  uint32 [tile][load][g]: byte 2 q + r of a word is the scale of inputs 64 load + 32 q .. + 31 of
 *             output 16 tile + g + 8 r, q = 0..1, r = 0..1.
 *     The words of the whole matrix lie first, its scales behind them: stored_byte_count is
 *     output_size * input_size / 2 + output_size * input_size / 32.
 *   WEIGHT_FORMAT_E4M3_SCALED: output_size a multiple of 16, input_size a multiple of 32, bits_per_weight
 *   8. A weight is an E4M3 byte (sign, 4 exponent bits of bias 7, 3 mantissa bits; exponent 0 is subnormal), and
 *   a block of 32 outputs by 32 inputs shares one scale; the last block may hold 16 outputs.
 *     bytes   uint8 [tile = output / 16][group = input / 32][lane][16]: byte b of a lane is
 *             W[16 tile + g + 8 (b % 2)][32 group + 8 (b / 4) + 2 t + (b / 2) % 2];
 *     scales  uint8 [block = output / 32][group], blocks counted with the last one, whole or half.
 *     The bytes of the whole matrix lie first, its scales behind them: stored_byte_count is
 *     output_size * input_size + ((output_size + 31) / 32) * (input_size / 32), every term counted in 64 bits.
 *   A pack holds no code that is no number: the converter refuses the E4M3 bytes 0x7F and 0xFF and the scale
 *   byte 0xFF in a source, and its verify refuses them in a pack. What a kernel makes of them is not specified.
 *   Measured (tools/measure: four_bit_expert_probe, eight_bit_trunk_probe, shared_rows_expert_probe): read this
 *   way a card decodes at 87% of its read rate, and a stored code becomes a 16-bit float by bit placement alone.
 */
#ifndef WEIGHT_FORMAT_SEAM_H
#define WEIGHT_FORMAT_SEAM_H

#include <stdint.h>

#include "anemone/status.h"

/* One 512-bit load of the host, and a multiple of the digest's word (pack_digest.h). */
#define WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES 64

enum weight_format_code {
	WEIGHT_FORMAT_NONE =
		0, /* no matrix of a format: a pack's tensor of plain elements */
	WEIGHT_FORMAT_FLOAT16 = 1, /* unquantized */
	WEIGHT_FORMAT_EXL3 =
		2, /* mul1 trellis, 2 or 3 bits per weight; exl3_records.h */
	WEIGHT_FORMAT_E2M1_SCALED =
		3, /* sign and 3-bit float, one power-of-two scale per 32 weights */
	WEIGHT_FORMAT_INT4_SCALED =
		4, /* 4-bit integer, one fp16 scale per 32 weights */
	WEIGHT_FORMAT_E4M3_SCALED =
		5, /* 8-bit float, one power-of-two scale per 32 by 32 weights */
	WEIGHT_FORMAT_COUNT = 6 /* number of entries; not a format */
};

struct matrix_shape_record {
	uint32_t output_size;
	uint32_t input_size;
	uint32_t bits_per_weight; /* EXL3: 2 or 3; a format of one width checks that this is its figure */
	uint32_t reserved_must_be_zero;
};

struct weight_format_operations {
	const char *name;
	/* First users: the converter, and the loader's check of a pack index. A shape the format does not store (a
     * size of zero or one its blocks do not divide, a bits_per_weight that is not the format's, a reserved field
     * that is not zero, a byte count beyond 64 bits) is STATUS_INVALID_ARGUMENT. */
	MUST_CHECK result_status (*stored_byte_count)(
		const struct matrix_shape_record *shape,
		uint64_t *byte_count_out);
};

/* The table of a format, or NULL for WEIGHT_FORMAT_NONE, a value outside the enumeration or a format this
 * build does not carry (EXL3 mul1 K2/K3, E2M1_SCALED and E4M3_SCALED). */
const struct weight_format_operations *
weight_format_operations_of(enum weight_format_code format);

#endif
