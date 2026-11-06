/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent two-pass form of the pinned F1 HC projection. Actual rows 1/3/4/8
 * and every other numerical/layout restriction are official_hc.h's. It retains
 * dot47's partials and final fold, computing RMS once per row. Scratch is caller
 * owned and disjoint from all input/weight/output ranges. No runtime selector
 * is changed. Both launches and scratch stay live through pass two.
 */
#ifndef OFFICIAL_HC_PARTITIONED_H
#define OFFICIAL_HC_PARTITIONED_H

#include "anemone/official_hc.h"
#include "anemone/official_hc_partitioned_records.h"

struct official_hc_partitioned_kernels {
	uint32_t partials;
	uint32_t reduce;
};

MUST_CHECK result_status official_hc_partitioned_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct official_hc_partitioned_kernels *kernels_out);

/* Direct/dense handles validate the shared projection contract; no direct
 * launch is executed. A refusal leaves both output launch records untouched.
 */
MUST_CHECK result_status official_hc_partitioned_prepare(
	const struct dense_linear_description *description,
	const struct linear_range *partials, uint64_t arena_byte_count,
	const struct dense_linear_kernels *dense,
	const struct official_hc_kernels *official,
	const struct official_hc_partitioned_kernels *partitioned,
	struct kernel_launch launches_out[OFFICIAL_HC_PARTITIONED_LAUNCHES]);
#endif
