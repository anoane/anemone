/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Integer copies of STEP_ROW_LOGITS rows into a peer's strided head array. C11/CUDA ABI. */
#ifndef HEAD_OFFLOAD_RECORDS_H
#define HEAD_OFFLOAD_RECORDS_H

#include "anemone/linear_records.h"
#include "anemone/step_records.h"

#define KERNEL_FILE_HEAD_OFFLOAD "kernel_head_offload"
#define KERNEL_HEAD_OFFLOAD_COPY "head_offload_copy_rows"
#define HEAD_OFFLOAD_ARGUMENT_BYTES 64u
#define HEAD_OFFLOAD_THREADS 256u
#define HEAD_OFFLOAD_MOST_BLOCKS 32u
#define HEAD_OFFLOAD_WORD_BYTES 16u
#define HEAD_OFFLOAD_RESERVED_WORDS 2

struct head_offload_arguments {
	uint64_t source_offset;
	uint64_t peer_word_offset;
	uint64_t destination_offset;
	uint64_t step_offset;
	uint64_t source_stride;
	uint64_t destination_stride;
	uint32_t row_bytes;
	uint32_t row_count;
	uint32_t reserved_zero[HEAD_OFFLOAD_RESERVED_WORDS];
};
LINEAR_RECORD_HAS(head_offload_arguments, HEAD_OFFLOAD_ARGUMENT_BYTES);
#endif
