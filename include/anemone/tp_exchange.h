/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Startup-only TP exchange preparation. The underlying ordered_push.h ownership, casts and ordered sums apply.
 *
 * The positive model sequence s becomes 2*(s*layer_count+layer)+(s&1), preserving the input/output parity.
 * All ranks use the same layer in [0,layer_count), layer_count in1..MODEL_LAYERS, and increasing model sequences.
 * Within a step, layers using one shared site region execute in increasing order. Ordered push's all-rank DONE
 * barrier retires every inbox reader before return, so later layers may reuse that site's payload AND metadata.
 * Two sublayer sites use distinct regions. No concurrent flight may use a region; group scheduling must enforce it.
 * Bounds and sequence overflow fail closed. This is a scheduling change with the same active-row arithmetic.
 *
 * The prepared element_count is a whole number of HIDDEN rows. STEP.row_count selects1..that many at execution;
 * only those rows are communicated. Input/output capacities and rank inbox strides retain the prepared maximum.
 * Caller prepares local inputs before this launch and consumes its output after. The ordinary step parity still
 * selects their slot. This leaves the standalone generic ordered-push ABI and behavior unchanged.
 *
 * prepare inherits ordered_push's bounds/disjointness rules, allocates nothing, touches no card and leaves output
 * unchanged on failure. Mapping lifetime, actual-allocation route qualification and whole-group drain remain
 * caller obligations. This contract does not establish quality, speed or concurrent-flight admission.
 *
 * The wide form (a piece of more than 8 rows): the same exchange in TP_EXCHANGE_WIDE_LAUNCHES launches the
 * caller queues in order on the card's one stream, with the same arguments, checks, phase, payload, ascending-rank
 * sums and output bits: the payload pushed by TP_EXCHANGE_WIDE_BLOCKS CTAs, each writer fencing its stores
 * at system scope before its launch ends; READY published and awaited by one CTA; the ordered sums by
 * TP_EXCHANGE_WIDE_BLOCKS CTAs; DONE published and awaited by one CTA, so the all-rank DONE barrier still
 * retires every inbox reader before a later layer's payload. wide_kernels_prepare leaves kernels_out untouched on
 * failure; wide_prepare checks as prepare does and leaves launches_out untouched on failure.
 *
 * Owner sums (form TP_EXCHANGE_FORM_OWNER_SUMS, --tp_wide_exchange=2): the wide form's READY and DONE with
 * another PUBLISH and SUM.
 * Rank d owns rows d R / ranks .. (d + 1) R / ranks of the step's R: every rank pushes only that shard of its payload
 * to rank d, which sums it in ascending rank order (the push's adds) and stores the sum rounded to BF16 (the FP32-to-
 * BF16 join's rounding) into every rank's output array; the caller then copies BF16 rows, not FP32 ones. Half the
 * push's bytes, the same output bits. Direct routes only, every offset and stride a multiple of 16 bytes.
 * Owner ring sums (FORM_OWNER_RING_SUMS, the FFN site of --tp_wide_exchange=3): the input array holds BF16 rows and
 * the sums are the BF16 ring's (exact_ring_records.h) of each slab of 8 rows: rank o owns quarter o of every slab's
 * bytes as the ring does and adds the four in the ring's order from rank o + 1, each add rounded to BF16 (its
 * fma.rn.bf16x2), then stores the BF16 sums into every rank's output array. Four ranks and the BF16 wire only.
 * Copy-engine owner sums (FORM_CE_OWNER_SUMS, --tp_wide_exchange=4's attention site): the caller's copies put each
 * rank's quarter d of the prepared rows (BF16, its values as they are) into rank d's inbox of slot 0, READY passes, SUM
 * adds the live rows of the rank's quarter as the owner sums do into its own output array of slot 0, the caller's
 * copies carry that quarter to the other ranks' output arrays, DONE passes. PUBLISH is not run; the same bits.
 */
#ifndef TP_EXCHANGE_H
#define TP_EXCHANGE_H

#include "anemone/ordered_push.h"
#include "anemone/tp_exchange_records.h"

MUST_CHECK result_status
tp_exchange_kernel_prepare(const struct backend_operations *backend,
			   struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
tp_exchange_prepare(const struct ordered_push_description *description,
		    uint32_t layer, uint32_t layer_count, uint32_t kernel,
		    struct kernel_launch *launch_out);
MUST_CHECK result_status tp_exchange_wide_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t form, uint32_t kernels_out[TP_EXCHANGE_WIDE_LAUNCHES]);
MUST_CHECK result_status tp_exchange_wide_prepare(
	const struct ordered_push_description *description, uint32_t layer,
	uint32_t layer_count, const uint32_t kernels[TP_EXCHANGE_WIDE_LAUNCHES],
	uint32_t form,
	struct kernel_launch launches_out[TP_EXCHANGE_WIDE_LAUNCHES]);

#endif
