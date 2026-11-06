/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Standalone two-shot push module. The tested one-shot module and default engine modules remain unchanged. */
#include "kernel_ordered_push_two.cuh"

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	ordered_push_two(struct ordered_push_arguments arguments,
			 unsigned char *arena)
{
	push_two_run_sequence(
		arguments, arena,
		((const struct step_block *)(arena + arguments.step_offset))
			->sequence);
}
