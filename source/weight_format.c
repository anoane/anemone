/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * weight_format.c: the weight formats this build carries and the bytes a matrix of each occupies.
 * Contract: weight_format_seam.h, which draws both layouts.
 *
 * Both formats store the codes of the whole matrix first and its scales behind them, in tiles of 16 outputs, with
 * one scale for 32 consecutive inputs: one check of a shape serves both, each with its own figures.
 */
#include "anemone/weight_format_seam.h"
#include "anemone/exl3.h"

#include <stdbool.h>
#include <stddef.h>

#define OUTPUTS_PER_TILE 16                     /* the outputs of one tensor-core operand */
#define INPUTS_PER_SCALE 32                     /* consecutive inputs of an output that share a scale */
#define E2M1_INPUTS_PER_LOAD 64                 /* the inputs of a tile that the 512 bytes of one load hold */
#define E2M1_BITS_PER_WEIGHT 4
#define E2M1_WEIGHTS_PER_BYTE 2
#define E4M3_BITS_PER_WEIGHT 8
#define E4M3_OUTPUTS_PER_SCALE 32               /* the outputs of a block that share its scales; the last may hold 16 */

/* True for a shape the format stores unless its byte count passes 64 bits: no size is zero, the outputs are whole
 * tiles and the inputs whole blocks, bits_per_weight is the format's figure and the reserved field is zero. */
static bool shape_is_stored(const struct matrix_shape_record *shape,
			    uint32_t inputs_per_block, uint32_t bits_per_weight)
{
    return shape->output_size != 0 && shape->output_size % OUTPUTS_PER_TILE == 0 && shape->input_size != 0 &&
           shape->input_size % inputs_per_block == 0 && shape->bits_per_weight == bits_per_weight &&
           shape->reserved_must_be_zero == 0;
}

MUST_CHECK static result_status
e2m1_scaled_stored_byte_count(const struct matrix_shape_record *shape,
			      uint64_t *byte_count_out)
{
    if (shape == NULL || byte_count_out == NULL ||
        !shape_is_stored(shape, E2M1_INPUTS_PER_LOAD, E2M1_BITS_PER_WEIGHT)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    /* The product of two 32-bit sizes is below 2^64, and a half and a 32nd of it add up to less than it: no shape
     * of this format has a byte count beyond 64 bits. Both divisions are exact: the product is whole loads. */
    const uint64_t weight_count = (uint64_t)shape->output_size * shape->input_size;
    *byte_count_out = weight_count / E2M1_WEIGHTS_PER_BYTE + weight_count / INPUTS_PER_SCALE;
    return STATUS_OK;
}

MUST_CHECK static result_status
e4m3_scaled_stored_byte_count(const struct matrix_shape_record *shape,
			      uint64_t *byte_count_out)
{
    if (shape == NULL || byte_count_out == NULL || !shape_is_stored(shape, INPUTS_PER_SCALE, E4M3_BITS_PER_WEIGHT)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    /* One byte to a weight: below 2^64. The blocks are counted in 64 bits, with the last one, whole or half: at
     * most 2^27 of them by fewer than 2^27 groups of inputs. Only the sum of the two can pass 64 bits. */
    const uint64_t weight_count = (uint64_t)shape->output_size * shape->input_size;
    const uint64_t scale_count = ((uint64_t)shape->output_size + E4M3_OUTPUTS_PER_SCALE - 1) / E4M3_OUTPUTS_PER_SCALE *
                                 (shape->input_size / INPUTS_PER_SCALE);
    if (scale_count > UINT64_MAX - weight_count) {
	    return STATUS_INVALID_ARGUMENT;
    }
    *byte_count_out = weight_count + scale_count;
    return STATUS_OK;
}

/* By the values a pack stores. The entry of a format this build does not carry is empty: it has no function. */
static const struct weight_format_operations carried_operations[WEIGHT_FORMAT_COUNT] = {
	[WEIGHT_FORMAT_E2M1_SCALED] = { .name = "e2m1_scaled",
					.stored_byte_count =
						e2m1_scaled_stored_byte_count },
	[WEIGHT_FORMAT_E4M3_SCALED] = { .name = "e4m3_scaled",
					.stored_byte_count =
						e4m3_scaled_stored_byte_count },
};

const struct weight_format_operations *
weight_format_operations_of(enum weight_format_code format)
{
	if (format == WEIGHT_FORMAT_EXL3) {
		return &exl3_weight_operations;
	}
	const struct weight_format_operations *operations = NULL;
	/* a pack gives any 32 bits: the value is held against both ends, whatever integer the enumeration is */
	if ((int)format >= 0 && format < WEIGHT_FORMAT_COUNT) {
		operations = &carried_operations[format];
	}
    return operations != NULL && operations->stored_byte_count != NULL ? operations : NULL;
}
