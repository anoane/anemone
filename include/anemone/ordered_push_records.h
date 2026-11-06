/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* One-shot GPU-initiated push, C11/CUDA ABI. See ordered_push.h for ownership and numerical contracts. */
#ifndef ORDERED_PUSH_RECORDS_H
#define ORDERED_PUSH_RECORDS_H

#include "anemone/ordered_collective_records.h"

#define KERNEL_FILE_ORDERED_PUSH "kernel_ordered_push"
#define KERNEL_ORDERED_PUSH "ordered_push"
#define ORDERED_PUSH_ARGUMENT_BYTES 96u
#define ORDERED_PUSH_FP32 0u
#define ORDERED_PUSH_BF16 1u
#define ORDERED_PUSH_PHASES 2u
#define ORDERED_PUSH_READY 0u
#define ORDERED_PUSH_DONE 1u

struct ordered_push_arguments {
	uint64_t peer_table_offset;
	uint64_t region_offset;
	uint64_t input_offset;
	uint64_t output_offset;
	uint64_t inbox_offset;
	uint64_t step_offset;
	uint64_t parity_stride;
	uint64_t deadline_cycles;
	uint64_t inbox_rank_stride;
	uint32_t element_count;
	uint32_t rank_count;
	uint32_t rank;
	uint32_t payload_format;
	uint32_t incoming_host_mask;
	uint32_t reserved_zero;
};

STEP_RECORD_HAS(ordered_push_arguments, ORDERED_PUSH_ARGUMENT_BYTES);
#endif
