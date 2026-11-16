/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Masked peer row copy for an experimental head offload. No arithmetic or synchronization protocol.
 * The caller owns a peer_connect address word, the receiving arena and all source/STEP bytes until
 * actual completion. Only descriptor/STEP-bounded LOGITS rows are copied, with original row indices.
 * Every copying thread system-fences before returning. A consumer must follow an already-recorded
 * completion mark; this kernel neither polls nor publishes a ready word. Separate flights require
 * disjoint row/STEP/destination ranges and marks, or an explicit completed-use lease.
 * Local ranges and the peer destination base use LINEAR_OFFSET_ALIGNMENT_BYTES alignment; strides,
 * row bytes and column bytes use16B alignment. Destination is a distinct, attached peer arena.
 * Prepare leaves launch_out unchanged on refusal. Engine callers must enforce the group/lease contract. */
#ifndef HEAD_OFFLOAD_H
#define HEAD_OFFLOAD_H

#include "anemone/linear.h"
#include "anemone/head_offload_records.h"

struct head_offload_description {
	struct linear_range source, peer_word, step, destination;
	uint64_t destination_arena_byte_count;
	uint64_t source_stride, destination_stride;
	uint32_t row_bytes, row_count, column_bytes, reserved_zero;
};

MUST_CHECK result_status
head_offload_prepare(const struct head_offload_description *description,
		     uint64_t source_arena_byte_count, uint32_t kernel,
		     struct kernel_launch *launch_out);
#endif
