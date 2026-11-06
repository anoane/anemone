/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Experimental source-order mHC projection, independent of existing dense selection.
 * Only 1/3/4/8 rows of BF16 streams with FP32 weights and output are admitted:
 * 24 by 20480, row scaling only, output scale one and the model norm epsilon.
 * These orders derive from a pinned PRO library trace; device fixture equality
 * does not prove equality for every library version, shape or model state.
 * Preparation validates ranges and aliases through dense_linear_prepare and
 * leaves the caller's output untouched on failure. No inference-time allocation.
 */
#ifndef OFFICIAL_HC_H
#define OFFICIAL_HC_H

#include "anemone/dense_linear.h"
#include "anemone/official_hc_records.h"

struct official_hc_kernels {
	uint32_t dot;
	uint32_t split;
};

MUST_CHECK result_status official_hc_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct official_hc_kernels *kernels_out);

MUST_CHECK result_status official_hc_prepare(
	const struct dense_linear_description *description,
	uint64_t arena_byte_count, const struct dense_linear_kernels *dense,
	const struct official_hc_kernels *official,
	struct kernel_launch *launch_out);

/* Match a captured reference call shape independently of the engine batch.
 * Both actual and reference rows admit 1/3/4/8. Arena ranges and launch grid
 * use actual rows; the reused argument row_count records the reference shape
 * for the statistic tree. No cross-row arithmetic or padding is introduced.
 */
MUST_CHECK result_status
official_hc_shape_prepare(const struct dense_linear_description *description,
			  uint32_t reference_rows, uint64_t arena_byte_count,
			  const struct dense_linear_kernels *dense,
			  const struct official_hc_kernels *official,
			  struct kernel_launch *launch_out);

/* Optional lossless shared transpose for the multi-row projection, retaining the direct form above. */
MUST_CHECK result_status official_hc_staged_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct official_hc_kernels *kernels);

#endif
