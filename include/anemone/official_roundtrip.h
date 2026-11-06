/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original group32 E4M3/E8M0 quantize/dequantize, returning BF16 values.
 * Caller supplies finite BF16 inputs with magnitude at most 65536. There is no
 * host scan of device data; upstream construction must preserve this domain.
 * Width is a multiple of 32 up to 32768, rows 1..8. Input/output may be identical
 * ranges or disjoint; partial overlap is refused. Preparation allocates nothing,
 * validates all extents and leaves the scalar launch untouched on failure.
 * This independent component has no selected runtime/default implication.
 */
#ifndef OFFICIAL_ROUNDTRIP_H
#define OFFICIAL_ROUNDTRIP_H

#include "anemone/linear.h"
#include "anemone/official_roundtrip_records.h"

struct official_roundtrip_description {
	struct linear_range input;
	struct linear_range output;
	uint32_t width;
	uint32_t row_count;
};

MUST_CHECK result_status official_roundtrip_kernel_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t *kernel_out);
MUST_CHECK result_status official_roundtrip_prepare(
	const struct official_roundtrip_description *description,
	uint64_t arena_byte_count, uint32_t kernel,
	struct kernel_launch *launch_out);
#endif
