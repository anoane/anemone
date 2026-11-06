/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent original-weight projections with no staging or allocation.
 * Quantized input codes and UE8M0 scales are read in the retained pack layout;
 * outputs are BF16. Each unscaled E4M3 K32 dot is rounded to the observed
 * SM120 instruction result, then multiplied by input scale and weight scale
 * separately, and accumulated in ascending K32 order. The native form uses
 * an unscaled SM120 E4M3 MMA; the portable form implements that instruction's
 * finite operand-exponent anchor and signed truncation.
 *
 * WoA is a different operation: packed weights first round to BF16, then each
 * K16 update includes the F32 carry in its operand-exponent anchor. It must
 * not use the quantized K32 form. Admitted operands have finite BF16 values,
 * normal or zero F32 products, and finite partials. No data scan is done.
 *
 * Preparation reuses the retained trunk descriptor/range validation, then
 * sets this form's geometry. Actual rows are 1..8; shapes stay fixed across
 * row extents. Native preparation is available only on class120 native.
 * Caller output is untouched on refusal; no execution or arena access here.
 */
#ifndef OFFICIAL_LINEAR_H
#define OFFICIAL_LINEAR_H

#include "anemone/linear.h"
#include "anemone/official_linear_records.h"

enum official_linear_form {
	OFFICIAL_LINEAR_PORTABLE = 0,
	OFFICIAL_LINEAR_NATIVE = 1
};

struct official_linear_kernels {
	uint32_t quantized;
	uint32_t quantized_native;
	uint32_t woa;
};

MUST_CHECK result_status official_linear_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct official_linear_kernels *kernels_out);

MUST_CHECK result_status official_linear_quantized_prepare(
	const struct linear_trunk_quantized_description *description,
	uint64_t arena_byte_count,
	const struct official_linear_kernels *kernels,
	enum official_linear_form form, struct kernel_launch *launch_out);

MUST_CHECK result_status official_linear_woa_prepare(
	const struct linear_trunk_bfloat16_description *description,
	uint64_t arena_byte_count,
	const struct official_linear_kernels *kernels,
	struct kernel_launch *launch_out);

#endif
