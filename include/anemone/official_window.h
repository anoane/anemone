/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Optional fused composition of the retained independent BF16 norm, forward
 * rotation and group32 FP8 roundtrip. No cache layout/scatter or coefficient
 * generation occurs here. All five ranges must be disjoint and caller-owned.
 * Actual and reference rows are separately admitted at 1/3/4/8. A refusal
 * preserves launch_out. Finite input/gain/coefficients and the roundtrip's
 * finite <=65536 output domain are producer responsibilities, not host scans.
 * This building block is not a serving selector or model qualification.
 */
#ifndef OFFICIAL_WINDOW_H
#define OFFICIAL_WINDOW_H

#include "anemone/official_norm.h"
#include "anemone/official_window_records.h"

struct official_window_description {
	struct official_norm_description norm;
	struct linear_range rotation;
};

MUST_CHECK result_status
official_window_kernel_prepare(const struct backend_operations *backend,
			       struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
official_window_prepare(const struct official_window_description *description,
			uint32_t reference_rows, uint64_t arena_byte_count,
			uint32_t kernel, struct kernel_launch *launch_out);
#endif
