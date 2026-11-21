/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef DS41F_OFFICIAL_ROPE_RECORDS_H
#define DS41F_OFFICIAL_ROPE_RECORDS_H

#include <stdint.h>
#include "models/deepseek_v41_flash/ds41f_geometry.h"

#define DS41F_MODULE_OFFICIAL_ROPE "ds41f_official_rope"
#define DS41F_KERNEL_OFFICIAL_ROPE "official_rope"
#define DS41F_OFFICIAL_ROPE_ARGUMENT_BYTES 48u
#define DS41F_OFFICIAL_ROPE_THREADS 256u
#define DS41F_OFFICIAL_ROPE_WIDTH (DS41F_FREQUENCIES * DS41F_PAIR_VALUES)

/* Contiguous BF16 [rows,heads,width]; coefficients F32 [rows,32,2].
 * Width 64 rotates the whole vector; width 512 copies its first 448 values.
 * Coefficients belong to the caller's recorded source schedule. No generation
 * or cache scatter is performed here. Inverse conjugates the supplied sine.
 */
struct ds41f_official_rope_arguments {
	uint64_t input_offset;
	uint64_t rotation_offset;
	uint64_t output_offset;
	uint32_t width;
	uint32_t heads;
	uint32_t rows;
	uint32_t inverse;
	uint64_t reserved_zero;
};

RECORD_HOLDS(sizeof(struct ds41f_official_rope_arguments) ==
	     DS41F_OFFICIAL_ROPE_ARGUMENT_BYTES);
#endif
