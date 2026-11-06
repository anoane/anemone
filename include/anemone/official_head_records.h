/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef OFFICIAL_HEAD_RECORDS_H
#define OFFICIAL_HEAD_RECORDS_H

#include <stdint.h>
#include "anemone/model_layout.h"

#define KERNEL_FILE_OFFICIAL_HEAD "kernel_official_head"
#define KERNEL_OFFICIAL_HEAD "official_head"
#define KERNEL_OFFICIAL_HEAD_REUSED "official_head_reused"
#define OFFICIAL_HEAD_ARGUMENT_BYTES 32u
#define OFFICIAL_HEAD_OWNERS 16u
#define OFFICIAL_HEAD_OUTPUTS 4u
#define OFFICIAL_HEAD_THREADS (OFFICIAL_HEAD_OWNERS * OFFICIAL_HEAD_OUTPUTS)
#define OFFICIAL_HEAD_ODD_ROWS 3u

/* Each output retains16 strided serial RN chains, then distance8/4/2/1.
 * Actual rows are independent M1 arithmetic targets. A contiguous vocabulary
 * cut keeps the full-head order; it does not retarget a smaller library GEMM.
 */
struct official_head_arguments {
	uint64_t weights_offset;
	uint64_t rows_offset;
	uint64_t output_offset;
	uint32_t output_size;
	uint32_t row_count;
};

MODEL_LAYOUT_HOLDS(sizeof(struct official_head_arguments) ==
		   OFFICIAL_HEAD_ARGUMENT_BYTES);
#endif
