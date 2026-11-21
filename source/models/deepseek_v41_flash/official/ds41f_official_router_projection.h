/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent cached-M1 router projection: BF16 weights/rows, F32 output,
 * fixed384-by5120 matrix, actual rows1/3/4/8. Each output has32 ascending
 * strided serial FMA chains and a descending halving fold. These are the
 * captured M1 arithmetic targets, independent of actual verification rows.
 * No softplus, selection, normalization, expert work or model selector.
 *
 * Caller domain: finite BF16 inputs/weights, exact products normal F32 or
 * signed zero, finite partial sums. The original F32 router operands are exact
 * BF16 widenings on the qualified fixture. All ranges are disjoint and owned
 * by the caller; no allocation or operand scan. Refusal preserves launch_out.
 * CPU source equality is not a device, full-gate or runtime qualification.
 */
#ifndef DS41F_OFFICIAL_ROUTER_PROJECTION_H
#define DS41F_OFFICIAL_ROUTER_PROJECTION_H

#include "engine/library/block_scaled_linear/linear.h"
#include "ds41f_official_router_projection_records.h"

struct ds41f_official_router_projection_description {
	struct linear_range weights;
	struct linear_range rows;
	struct linear_range output;
	uint32_t row_count;
};

MUST_CHECK result_status ds41f_official_router_projection_kernel_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t *kernel_out);
MUST_CHECK result_status ds41f_official_router_projection_prepare(
	const struct ds41f_official_router_projection_description *description,
	uint64_t arena_byte_count, uint32_t kernel,
	struct kernel_launch *launch_out);
#endif
