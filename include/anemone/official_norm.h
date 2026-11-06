/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Experimental independent RMSNorm. Widths 512/1280/5120 and rows 1/3/4/8 only.
 * Epsilon is the model's. Statistics are F32 [row][mean,inverse]; other arrays
 * are BF16. Every range is disjoint/aligned/in arena. Failures leave outputs
 * untouched. This does not replace any selected model norm or allocate memory.
 */
#ifndef OFFICIAL_NORM_H
#define OFFICIAL_NORM_H

#include "anemone/linear.h"
#include "anemone/official_norm_records.h"

struct official_norm_description {
	struct linear_range input;
	struct linear_range weight;
	struct linear_range output;
	struct linear_range statistics;
	uint32_t width;
	uint32_t row_count;
};

MUST_CHECK result_status
official_norm_kernel_prepare(const struct backend_operations *backend,
			     struct device_card *card, uint32_t *kernel_out);

MUST_CHECK result_status
official_norm_prepare(const struct official_norm_description *description,
		      uint64_t arena_byte_count, uint32_t kernel,
		      struct kernel_launch *launch_out);

/* Actual rows own the memory extents and grid. Reference rows select only the
 * reduction ownership/tree. Both admit 1/3/4/8; crossed shapes need their own
 * device qualification before a model program selects this form.
 */
MUST_CHECK result_status
official_norm_shape_prepare(const struct official_norm_description *description,
			    uint32_t reference_rows, uint64_t arena_byte_count,
			    uint32_t kernel, struct kernel_launch *launch_out);

#endif
