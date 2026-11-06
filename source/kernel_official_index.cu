/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The entire source selection is visible in this admitted context domain.
 * A candidate reader sees the same chronological prefix as its owner. */
#include "anemone/official_index_records.h"

extern "C" __global__ void
official_index(struct official_index_arguments arguments, unsigned char *arena)
{
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	if (step->row_count != 1 || step->flags != 0)
		return;
	const struct step_row_record *const row =
		(const struct step_row_record *)(step + 1);
	if (row->position < 0 ||
	    (uint32_t)row->position >= arguments.context_token_limit ||
	    row->session_slot != 0)
		return;
	const uint32_t visible =
		((uint32_t)row->position + 1u) / arguments.ratio;
	int32_t *const selection =
		(int32_t *)(arena + arguments.selection_offset);
	for (uint32_t entry = threadIdx.x; entry < MODEL_SELECTED;
	     entry += blockDim.x)
		selection[entry] = entry < visible ? (int32_t)entry : -1;
	if (arguments.candidates_offset != INDEX_NO_CANDIDATES) {
		uint32_t *const candidates =
			(uint32_t *)(arena + arguments.candidates_offset);
		const uint32_t blocks = (visible + MODEL_KEYS_PER_TILE - 1u) /
					MODEL_KEYS_PER_TILE;
		for (uint32_t block = threadIdx.x;
		     block < MODEL_CANDIDATE_BLOCKS; block += blockDim.x)
			candidates[block] = block < blocks ? block :
							     MODEL_NO_CANDIDATE;
	}
}
