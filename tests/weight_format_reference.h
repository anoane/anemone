/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * weight_format_reference.h: the host reference of the weight formats: where each weight and each scale of a matrix
 * lies in its stored bytes, and what each stands for in double precision. The oracle of the formats' kernels and of
 * the bytes a converter writes. Test support: no part of the engine.
 *
 * Contract:
 *   - It knows WEIGHT_FORMAT_E2M1_SCALED and WEIGHT_FORMAT_E4M3_SCALED and is written from the two
 *     layouts weight_format_seam.h draws, with no call into the engine: its sizes are a second count of what the
 *     engine's stored_byte_count gives.
 *   - A matrix comes as the checkpoint holds it: codes, one byte for each weight in the order [output][input] (a
 *     4-bit code 0 to 15, or an E4M3 byte), and scale_codes, E8M0 bytes in the checkpoint's own arrangement:
 *       E2M1_SCALED   [output_size][input_size / 32]: inputs 32 k .. 32 k + 31 of an output share its scale k
 *       E4M3_SCALED   [(output_size + 31) / 32][input_size / 32]: outputs 32 b .. 32 b + 31 by inputs 32 k ..
 *                     32 k + 31 share scale k of block b; the last block holds the 16 outputs that are left where
 *                     output_size is no multiple of 32
 *   - test_weight_format_lay_out writes every byte of the stored form and reads nothing of it. The two functions
 *     test_weight_format_lay_out and test_weight_format_read_weight are written independently of each other: the
 *     lay-out walks the stored words and bytes and asks, with the formulas of the seam, which weight or scale
 *     lies in each; the read starts from a weight and computes where it and its scale lie. Neither calls the other
 *     and they share no formula, so a matrix that is laid out and read back holds one against the other.
 *   - Values. An E2M1 code: bit 3 is the sign, bits 2..0 choose 0, 0.5, 1, 1.5, 2, 3, 4 or 6; code 8 is minus
 *     zero. An E4M3 byte: bit 7 is the sign, bits 6..3 the exponent e of bias 7, bits 2..0 the mantissa m; it is
 *     (1 + m / 8) * 2^(e - 7), and m / 8 * 2^-6 where e is 0; byte 0x80 is minus zero. An E8M0 byte c is
 *     2^(c - 127). The value of a weight is the value of its code times the value of its scale, which a double
 *     holds exactly.
 *   - Decided here, because the seam does not say it: a code that the checkpoint's own format reads as
 *     not-a-number is not-a-number in this reference. These are the E4M3 bytes 0x7F and 0xFF and the E8M0 byte
 *     0xFF. By their fields they would be +480, -480 and 2^128, and a kernel that makes a 16-bit float of a weight
 *     by bit placement does make +480 and -480 of the two bytes: the test of a kernel fills no such code, or says
 *     what it expects of one. Lay-out and read move these codes like any other. A number that is no code of its
 *     format at all (above 15, above 255) is not-a-number too.
 *   - Refused with STATUS_INVALID_ARGUMENT: a NULL argument, a format other than the two, a shape its
 *     format does not store (the rules of stored_byte_count in weight_format_seam.h), a count that is not the
 *     shape's, an E2M1 code above 15, an output or an input beyond the shape. On a refusal the outputs are
 *     untouched; the stored bytes of a lay-out that is refused are unspecified.
 *   - Every function may be called from any thread.
 */
#ifndef TEST_WEIGHT_FORMAT_REFERENCE_H
#define TEST_WEIGHT_FORMAT_REFERENCE_H

#include <stdint.h>

#include "anemone/status.h"
#include "anemone/weight_format_seam.h"

#define TEST_WEIGHT_FORMAT_E2M1_CODE_COUNT 16u
#define TEST_WEIGHT_FORMAT_BYTE_CODE_COUNT 256u             /* the E4M3 bytes, and the E8M0 bytes */
#define TEST_WEIGHT_FORMAT_E4M3_NOT_A_NUMBER_MAGNITUDE 0x7Fu /* a byte with all of these bits set: 0x7F and 0xFF */
#define TEST_WEIGHT_FORMAT_E8M0_NOT_A_NUMBER 0xFFu

/* What a matrix of a shape holds. */
struct test_weight_format_sizes {
    uint64_t weight_count;                      /* output_size * input_size: the codes of the checkpoint's form */
    uint64_t scale_count;                       /* the scale codes of the checkpoint's form, and of the stored one */
    uint64_t stored_byte_count;                 /* the weights as they are stored, then scale_count bytes */
};

/* One weight as its matrix stores it. */
struct test_weight_format_weight {
    uint32_t code;                              /* the 4-bit code, or the E4M3 byte */
    uint32_t scale_code;                        /* the E8M0 byte of the scale it shares */
    double value;                               /* the value of the code times the value of the scale */
};

double test_weight_format_e2m1_value(uint32_t code);
double test_weight_format_e4m3_value(uint32_t code);
double test_weight_format_e8m0_value(uint32_t code);

MUST_CHECK result_status test_weight_format_sizes_of(
	enum weight_format_code format, const struct matrix_shape_record *shape,
	struct test_weight_format_sizes *sizes_out);

/* code_count, scale_code_count and stored_byte_count are what the caller holds behind each pointer: they must be
 * the three sizes of the shape. */
MUST_CHECK result_status test_weight_format_lay_out(
	enum weight_format_code format, const struct matrix_shape_record *shape,
	const uint8_t *codes, uint64_t code_count, const uint8_t *scale_codes,
	uint64_t scale_code_count, uint8_t *stored, uint64_t stored_byte_count);

/* The weight W[output_index][input_index] of a stored matrix of stored_byte_count bytes. */
MUST_CHECK result_status test_weight_format_read_weight(
	enum weight_format_code format, const struct matrix_shape_record *shape,
	const uint8_t *stored, uint64_t stored_byte_count,
	uint32_t output_index, uint32_t input_index,
	struct test_weight_format_weight *weight_out);

#endif
