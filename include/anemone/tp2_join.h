/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * TP-local completion, prepared once at load. This does not change any PP product or post kernel.
 *
 * BFLOAT16_TO_FLOAT32 publishes local BF16 product values as FP32. With routed_sums it first computes
 * add.rn.f32(routed_sum, shared_value), in the existing expert join's order, without rounding that local sum to
 * BF16. The routed sum itself must already be +0 then expert slots in ascending order. This is one local partial;
 * ordered_push sums ranks in its fixed order. FLOAT32_TO_BFLOAT16 rounds the globally reduced vector once, using
 * the original products' nearest-even/quiet-NaN rule, for the unchanged hyper post. Do not also supply that vector
 * as hyper_post.routed_sums: use either this final round or the post's combined round, never both.
 * BFLOAT16_COPY copies raw BF16 words between compact and canonical head/group rows. FLOAT32_COPY copies raw
 * FP32 words for the paired local-only collective stub. Both preserve negative zeros and NaN payloads. Width
 * elements in each row are read/written; row gaps remain untouched in every operation.
 * With a nonzero source/output_slot_stride, that base advances by (step.sequence & 1) times the byte stride.
 * Both parity slots and the complete step block are checked at prepare. Zero slot stride keeps that array static.
 *
 * TP product partials are already rounded to BF16 by the existing dot-product kernels. Local rounding followed
 * by rank reduction and one final BF16 round changes the original full-dot arithmetic: relaxed/fast only.
 * BF16 upcast alone and BF16_COPY preserve represented values/bits; they do not establish exact model TP.
 *
 * Counts: 1..LINEAR_WIDE_MOST_ROWS rows, width 1..MODEL_HEADS*MODEL_HEAD_DIMS, every element stride at
 * least width. Missing routed_sums is exactly {0,0} with zero routed stride. A routed array is allowed only for
 * BFLOAT16_TO_FLOAT32. Reserved words are zero. Ranges follow prepare_common.h: aligned, bounded, large enough
 * for the final strided row in both slots. Used row spans, both used parity spans and the step are disjoint;
 * enclosing capacities may cover unused slot gaps. Slot byte strides are aligned and at least one used span.
 * A static operation requires step exactly {0,0}. In-place copies/conversions are refused. prepare touches no card,
 * allocates nothing and leaves the complete output untouched on refusal. Mappings must outlive every launch.
 */
#ifndef TP2_JOIN_H
#define TP2_JOIN_H

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/tp2_join_records.h"

struct tp2_join_description {
	struct linear_range source;
	struct linear_range routed_sums;
	struct linear_range output;
	struct linear_range step;
	uint64_t source_slot_stride;
	uint64_t output_slot_stride;
	uint32_t width;
	uint32_t row_count;
	uint32_t source_row_stride;
	uint32_t routed_row_stride;
	uint32_t output_row_stride;
	uint32_t operation;
	uint32_t reserved_zero;
};

MUST_CHECK result_status
tp2_join_kernel_prepare(const struct backend_operations *backend,
			struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
tp2_join_prepare(const struct tp2_join_description *description,
		 uint64_t arena_byte_count, uint32_t kernel,
		 struct kernel_launch *launch_out);

#endif
