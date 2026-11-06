/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent BF16 rotation with explicit source-compatible FMA order and BF16
 * assignment. Finite input/coefficient domain is the producer's responsibility.
 * Rows 1..8, heads 1..64, width 64 or 512; no allocation or per-token host scan.
 * Input/output must be disjoint or the identical range; coefficients are
 * disjoint. Refusal leaves launch_out untouched. This is not a model selector
 * and does not qualify coefficient generation or any composed runtime.
 */
#ifndef OFFICIAL_ROPE_H
#define OFFICIAL_ROPE_H

#include "anemone/linear.h"
#include "anemone/official_rope_records.h"

struct official_rope_description {
	struct linear_range input;
	struct linear_range rotation;
	struct linear_range output;
	uint32_t width;
	uint32_t heads;
	uint32_t rows;
	uint32_t inverse;
};

MUST_CHECK result_status
official_rope_kernel_prepare(const struct backend_operations *backend,
			     struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
official_rope_prepare(const struct official_rope_description *description,
		      uint64_t arena_byte_count, uint32_t kernel,
		      struct kernel_launch *launch_out);
#endif
