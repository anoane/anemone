/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Small exact ordered collectives. The shared body also serves packet-only device-scope measurements. */
#include "kernel_ordered_collective.cuh"

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	ordered_all_reduce(struct ordered_collective_arguments arguments,
			   unsigned char *arena)
{
	ordered_run(arguments, arena, ORDERED_ALL_REDUCE);
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	ordered_two_shot(struct ordered_collective_arguments arguments,
			 unsigned char *arena)
{
	ordered_run(arguments, arena, ORDERED_TWO_SHOT);
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	ordered_reduce_scatter(struct ordered_collective_arguments arguments,
			       unsigned char *arena)
{
	ordered_run(arguments, arena, ORDERED_REDUCE_SCATTER);
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	ordered_gather(struct ordered_collective_arguments arguments,
		       unsigned char *arena)
{
	ordered_run(arguments, arena, ORDERED_GATHER);
}
