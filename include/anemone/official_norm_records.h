/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent pinned-source vector4 norm candidate; BF16 input/weight/output.
 * Each grid-y row writes its F32 mean/inverse pair to statistics. Threads follow
 * the contiguous Torch mean's per-row ownership for admitted widths/row shapes.
 * There is no cross-row reduction. Qualification remains fixture-specific.
 */
#ifndef OFFICIAL_NORM_RECORDS_H
#define OFFICIAL_NORM_RECORDS_H

#include <stdint.h>

#include "anemone/model_layout.h"

#define KERNEL_FILE_OFFICIAL_NORM "kernel_official_norm"
#define KERNEL_OFFICIAL_NORM "official_norm_entry"
#define OFFICIAL_NORM_ARGUMENT_BYTES 48u
#define OFFICIAL_NORM_MOST_THREADS 512u
#define OFFICIAL_NORM_ODD_ROWS 3u
#define OFFICIAL_NORM_STATISTICS 2u

struct official_norm_arguments {
	uint64_t input_offset;
	uint64_t weight_offset;
	uint64_t output_offset;
	uint64_t statistics_offset;
	uint32_t width;
	uint32_t reserved_zero;
	float mean_factor;
	float epsilon;
};

MODEL_LAYOUT_HOLDS(sizeof(struct official_norm_arguments) ==
		   OFFICIAL_NORM_ARGUMENT_BYTES);

#endif
