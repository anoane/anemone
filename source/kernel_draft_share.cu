/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_draft_share.cu (tp4_v3_claude): the draft's rows to the other ranks (include/anemone/draft_share_records.h).
 * Device code only, built once per architecture into the module "kernel_draft_share". Here:
 *   - A copy moves 16-byte words, a thread a word at a time; every copying thread fences at system scope before the
 *     block's barrier; thread 0 takes a ticket of the receiver's counter; the block with the last ticket sets the
 *     counter back to zero and releases the receiver's word.
 *   - A step drafts where a row of its own step area, below its row count, has STEP_ROW_DRAFTS.
 */
#include <stdint.h>

#include "anemone/draft_share_records.h"

#define SHARE_WORD_BYTES 16u

static_assert(DRAFT_SHARE_STEP % SHARE_WORD_BYTES == 0 &&
		      DRAFT_SHARE_VIEWS % SHARE_WORD_BYTES == 0 &&
		      STEP_BLOCK_BYTES_VALUE + LINEAR_MAXIMUM_ROWS *
						       STEP_ROW_BYTES_VALUE <=
			      DRAFT_SHARE_PART_BYTES,
	      "16-byte parts; the step area's copy fits its part");

#define SHARE_MOST_ROWS 64u /* a step's rows: a BATCH step's most (model_seam.h) */

__device__ static __forceinline__ uint32_t share_value(const unsigned char *arena, uint64_t step_offset,
                                                       uint32_t verifier)
{
	const uint64_t sequence =
		((const struct step_block *)(arena + step_offset))->sequence;
	return (uint32_t)((sequence * DRAFT_SHARE_VERIFIERS + verifier) * 2u) +
	       1u;
}

/* True where the step's own rows hold more than `verifier` DRAFTS rows. */
__device__ static __forceinline__ bool share_step_drafts(const unsigned char *arena, uint64_t step_offset,
                                                         uint32_t verifier)
{
	const struct step_block *const block =
		(const struct step_block *)(arena + step_offset);
	const struct step_row_record *const rows =
		(const struct step_row_record *)(block + 1);
	uint32_t drafts = 0;
	for (uint32_t row = 0; row < block->row_count && row < SHARE_MOST_ROWS;
	     row++)
		drafts += (rows[row].flags & STEP_ROW_DRAFTS) != 0 ? 1u : 0u;
	return drafts > verifier;
}

extern "C" __global__ void __launch_bounds__(DRAFT_SHARE_THREADS) draft_share(
	const struct draft_share_arguments __grid_constant__ arguments,
	unsigned char *arena)
{
	const uint32_t target = blockIdx.x / DRAFT_SHARE_BLOCKS_PER_RANK;
	if (target == arguments.rank ||
	    !share_step_drafts(arena, arguments.step_offset,
			       arguments.verifier))
		return;
	const uint32_t part = blockIdx.x % DRAFT_SHARE_BLOCKS_PER_RANK;
	unsigned char *const area = (unsigned char *)((
		const uint64_t *)(arena + arguments.peer_table_offset))[target];
	const uint32_t row_words =
		arguments.positions * MODEL_HIDDEN_ROW_BYTES / SHARE_WORD_BYTES;
	const uint32_t step_words =
		(STEP_BLOCK_BYTES_VALUE +
		 arguments.positions * STEP_ROW_BYTES_VALUE) /
		SHARE_WORD_BYTES;
	const uint32_t view_words =
		arguments.positions * STEP_DRAFT_VIEW_BYTES / SHARE_WORD_BYTES;
	const uint32_t lanes = DRAFT_SHARE_BLOCKS_PER_RANK * blockDim.x;
	for (uint32_t word = part * blockDim.x + threadIdx.x;
	     word < row_words + step_words + view_words; word += lanes) {
		const bool is_row = word < row_words,
			   is_step = !is_row && word < row_words + step_words;
		const uint32_t index = is_row  ? word :
				       is_step ? word - row_words :
						 word - row_words - step_words;
		const uint4 *const source =
			(const uint4 *)(arena +
					(is_row	 ? arguments.rows_offset :
					 is_step ? arguments.draft_step_offset :
						   arguments.views_offset)) +
			index;
		uint4 *const target_word =
			(uint4 *)(area + (is_row  ? 0u :
					  is_step ? DRAFT_SHARE_STEP :
						    DRAFT_SHARE_VIEWS)) +
			index;
		*target_word = *source;
	}
    __threadfence_system();
    __syncthreads();
    uint32_t *const counter = (uint32_t *)(arena + arguments.counter_offset) + target;
    if (threadIdx.x == 0 &&
	atomicAdd(counter, 1u) + 1u == DRAFT_SHARE_BLOCKS_PER_RANK) {
	    *counter = 0;
	    __threadfence_system();
	    asm volatile("st.release.sys.global.u32 [%0], %1;\n" ::"l"(
				 area + DRAFT_SHARE_WORD),
			 "r"(share_value(arena, arguments.step_offset,
					 arguments.verifier))
			 : "memory");
    }
}

extern "C" __global__ void
__launch_bounds__(DRAFT_SHARE_WAIT_THREADS) draft_share_wait(
	const struct draft_share_wait_arguments __grid_constant__ arguments,
	unsigned char *arena)
{
    if (threadIdx.x != 0)
        return;
    unsigned char *const area = arena + arguments.area_offset;
    if (!share_step_drafts(arena, arguments.step_offset, arguments.verifier)) {
	    ((struct step_block *)(area + DRAFT_SHARE_STEP))->row_count = 0;
	    for (uint32_t position = 0; position < arguments.positions;
		 position++)
		    ((struct step_block *)(area + DRAFT_SHARE_VIEWS +
					   position * STEP_DRAFT_VIEW_BYTES))
			    ->row_count = 0;
	    return;
    }
    const uint32_t value = share_value(arena, arguments.step_offset, arguments.verifier);
    const long long start = clock64();
    uint32_t seen;
    do {
	    asm volatile("ld.acquire.sys.global.u32 %0, [%1];\n"
			 : "=r"(seen)
			 : "l"(area + DRAFT_SHARE_WORD)
			 : "memory");
	    if ((unsigned long long)(clock64() - start) >
		arguments.deadline_cycles)
		    asm volatile("trap;");
    } while (seen != value);
}
