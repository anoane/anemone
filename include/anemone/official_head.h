/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent canonical M1 head arithmetic, derived from captured native
 * outputs and a measured16-by4 GEMV geometry. BF16 input/weights, F32 output;
 * input width5120, output count a positive multiple of4 up to129280, actual
 * rows1/3/4/8. All ranges are disjoint and caller-owned. No allocation, final
 * collapse/norm, sampling, model selector or library call occurs here.
 *
 * Producer domain: finite BF16 inputs and weights, each exact product normal
 * F32 or signed zero, with finite partial sums. Then explicit FMA has the same
 * rounded chain as an exact product followed by RN addition. The pinned head
 * uses F32 weights widened exactly from these BF16 words. Full source-input
 * CPU equality does not qualify arbitrary versions, devices or runtime joins.
 * Refusal preserves launch_out. Vocabulary cuts retain the canonical full-head
 * per-output order independently of their physical placement or width.
 */
#ifndef OFFICIAL_HEAD_H
#define OFFICIAL_HEAD_H

#include "anemone/linear.h"
#include "anemone/official_head_records.h"

struct official_head_description {
	struct linear_range weights;
	struct linear_range rows;
	struct linear_range output;
	uint32_t output_size;
	uint32_t row_count;
};

MUST_CHECK result_status
official_head_kernel_prepare(const struct backend_operations *backend,
			     struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
official_head_prepare(const struct official_head_description *description,
		      uint64_t arena_byte_count, uint32_t kernel,
		      struct kernel_launch *launch_out);

/* Optional weight reuse across actual rows; each row retains the same16-owner
 * chain/fold. It is a separate one-pass form, with its own paired kernel and
 * prepare entry points. Neither form selects or replaces the other implicitly.
 */
MUST_CHECK result_status official_head_reused_kernel_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t *kernel_out);
MUST_CHECK result_status official_head_reused_prepare(
	const struct official_head_description *description,
	uint64_t arena_byte_count, uint32_t kernel,
	struct kernel_launch *launch_out);
#endif
