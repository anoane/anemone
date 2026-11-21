/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef DS41F_OFFICIAL_WINDOW_RECORDS_H
#define DS41F_OFFICIAL_WINDOW_RECORDS_H

#include "ds41f_official_norm_records.h"
#include "ds41f_official_rope_records.h"

#define DS41F_MODULE_OFFICIAL_WINDOW "ds41f_official_window"
#define DS41F_KERNEL_OFFICIAL_WINDOW "official_window"
#define DS41F_OFFICIAL_WINDOW_ARGUMENT_BYTES 56u

/* BF16 input/output [rows,512], gain [512], F32 coefficients [rows,32,2]
 * and statistics [rows,2]. The reference norm extent selects the launch's X
 * owners; it does not change the actual grid or any array's row extent.
 */
struct ds41f_official_window_arguments {
	uint64_t input_offset;
	uint64_t weight_offset;
	uint64_t rotation_offset;
	uint64_t output_offset;
	uint64_t statistics_offset;
	uint32_t rows;
	uint32_t reserved_zero;
	float mean_factor;
	float epsilon;
};

RECORD_HOLDS(sizeof(struct ds41f_official_window_arguments) ==
	     DS41F_OFFICIAL_WINDOW_ARGUMENT_BYTES);
#endif
