/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef DS41F_OFFICIAL_HEAD_RECORDS_H
#define DS41F_OFFICIAL_HEAD_RECORDS_H

#include <stdint.h>
#include "models/deepseek_v41_flash/ds41f_geometry.h"

#define DS41F_MODULE_OFFICIAL_HEAD "ds41f_official_head"
#define DS41F_KERNEL_OFFICIAL_HEAD "official_head"
#define DS41F_KERNEL_OFFICIAL_HEAD_REUSED "official_head_reused"
#define DS41F_OFFICIAL_HEAD_ARGUMENT_BYTES 32u
#define DS41F_OFFICIAL_HEAD_OWNERS 16u
#define DS41F_OFFICIAL_HEAD_OUTPUTS 4u
#define DS41F_OFFICIAL_HEAD_THREADS \
	(DS41F_OFFICIAL_HEAD_OWNERS * DS41F_OFFICIAL_HEAD_OUTPUTS)
#define DS41F_OFFICIAL_HEAD_ODD_ROWS 3u

/* Each output retains16 strided serial RN chains, then distance8/4/2/1.
 * Actual rows are independent M1 arithmetic targets. A contiguous vocabulary
 * cut keeps the full-head order; it does not retarget a smaller library GEMM.
 */
struct ds41f_official_head_arguments {
	uint64_t weights_offset;
	uint64_t rows_offset;
	uint64_t output_offset;
	uint32_t output_size;
	uint32_t row_count;
};

RECORD_HOLDS(sizeof(struct ds41f_official_head_arguments) ==
	     DS41F_OFFICIAL_HEAD_ARGUMENT_BYTES);
#endif
