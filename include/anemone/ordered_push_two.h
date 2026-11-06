/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * Two-shot ordered push: scatter each input slice to its owner, then broadcast each owner's reduced FP32 slice.
 * Uses ordered_push.h's 96-byte arguments, peer table, full rank-indexed inboxes, numerical order, directed routes,
 * sequence/deadline and drained-lifetime contract. Reserve THREE rank_count*128-byte signal phases per parity slot;
 * READY follows scatter stores, RESULT follows reduced FP32 return stores, CREDIT follows every receiver's host
 * mirror result copies into local output. All participants publish all phases even for empty owner slices.
 * Local output contains a complete FP32 result before return. The separate two-shot plan never shares live slots
 * with a one-shot plan; changing algorithms requires complete group drain and zeroed independent preparation.
 *
 * Slice width is ceil(element_count/rank_count); the final slices can be shorter or empty. Each owner sums +0,
 * rank 0, rank 1, ... with explicit FP32 round-to-nearest. BF16 scatter rounds every input, including self, once;
 * reduced returns are FP32 so both algorithms produce the same bits under the one-shot numerical contract.
 * BF16 returns are deliberately NOT rounded again. For N ranks and P elements, an evenly split rank sends roughly
 * (N-1)/N * P * (input_wire_bytes+4) payload bytes: BF16 scatter has a larger FP32 return, unlike a BF16/BF16 model.
 * Finite FP32 inputs can overflow BF16 to infinity; required caller range/quality gates remain unchanged.
 *
 * Direct return stores target each receiver's GPU output slice; host return stores target its mapped mirror's
 * output slice. A receiver copies host sources' slices after RESULT acquire and before CREDIT publication.
 * No payload value is an arrival marker. Each operation runs one CTA, allocates nothing and has no CPU progress.
 * Preparation validates all local storage with the larger signal prefix; failure leaves launch_out untouched.
 */
#ifndef ORDERED_PUSH_TWO_H
#define ORDERED_PUSH_TWO_H

#include "anemone/ordered_push.h"
#include "anemone/ordered_push_two_records.h"

MUST_CHECK result_status
ordered_push_two_kernel_prepare(const struct backend_operations *backend,
				struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
ordered_push_two_prepare(const struct ordered_push_description *description,
			 uint32_t kernel, struct kernel_launch *launch_out);

#endif
