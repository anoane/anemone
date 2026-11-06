/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image tokens neither contribute Engram values nor update the hidden state through its gate. */
#include "anemone/image_routing_records.h"
#include "kernel_engram_gate.cuh"

extern "C" __global__ void __launch_bounds__(ENGRAM_ROWS_GATE_THREADS, 1)
	image_engram_gate(struct image_engram_arguments arguments,
			  unsigned char *arena)
{
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	const struct step_row_record *const rows =
		(const struct step_row_record *)(step + 1);
	if (blockIdx.y >= step->row_count ||
	    (rows[blockIdx.y].flags & STEP_ROW_IMAGE) != 0) {
		return;
	}
	engram_gate(arguments.base, arena);
}
