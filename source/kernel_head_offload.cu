/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Exact bit copies only. Completion is an external, already-recorded mark; no device wait or counter. */
#include "anemone/head_offload_records.h"
#include <cuda_runtime.h>

extern "C" __global__ void
head_offload_copy_rows(struct head_offload_arguments arguments,
		       unsigned char *arena)
{
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	const uint32_t row = blockIdx.y;
	if (row >= arguments.row_count || row >= step->row_count ||
	    (((const struct step_row_record *)(step + 1))[row].flags &
	     STEP_ROW_LOGITS) == 0)
		return;
	const uint64_t peer =
		*(const uint64_t *)(arena + arguments.peer_word_offset);
	const uint4 *const source =
		(const uint4 *)(arena + arguments.source_offset +
				(uint64_t)row * arguments.source_stride);
	uint4 *const destination =
		(uint4 *)(peer + arguments.destination_offset +
			  (uint64_t)row * arguments.destination_stride);
	const uint32_t words = arguments.row_bytes / HEAD_OFFLOAD_WORD_BYTES;
	for (uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
	     word < words; word += gridDim.x * blockDim.x)
		destination[word] = source[word];
	__threadfence_system();
}
