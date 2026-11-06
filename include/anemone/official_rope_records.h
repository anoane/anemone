/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef OFFICIAL_ROPE_RECORDS_H
#define OFFICIAL_ROPE_RECORDS_H

#include <stdint.h>
#include "anemone/model_layout.h"

#define KERNEL_FILE_OFFICIAL_ROPE "kernel_official_rope"
#define KERNEL_OFFICIAL_ROPE "official_rope"
#define OFFICIAL_ROPE_ARGUMENT_BYTES 48u
#define OFFICIAL_ROPE_THREADS 256u
#define OFFICIAL_ROPE_WIDTH (MODEL_FREQUENCIES * MODEL_PAIR_VALUES)

/* Contiguous BF16 [rows,heads,width]; coefficients F32 [rows,32,2].
 * Width 64 rotates the whole vector; width 512 copies its first 448 values.
 * Coefficients belong to the caller's recorded source schedule. No generation
 * or cache scatter is performed here. Inverse conjugates the supplied sine.
 */
struct official_rope_arguments {
	uint64_t input_offset;
	uint64_t rotation_offset;
	uint64_t output_offset;
	uint32_t width;
	uint32_t heads;
	uint32_t rows;
	uint32_t inverse;
	uint64_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(sizeof(struct official_rope_arguments) ==
		   OFFICIAL_ROPE_ARGUMENT_BYTES);
#endif
