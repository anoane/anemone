/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Standalone form; producer kernels can include the same phase helpers for fused communication. */
#include "kernel_ordered_push.cuh"

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	ordered_push(struct ordered_push_arguments arguments,
		     unsigned char *arena)
{
	push_run_sequence(
		arguments, arena,
		((const struct step_block *)(arena + arguments.step_offset))
			->sequence);
}
