/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/ordered_push_parallel_records.h"
#include "kernel_ordered_push.cuh"

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	ordered_push_parallel_publish(struct ordered_push_arguments arguments,
				      unsigned char *arena)
{
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	if (step->sequence == 0 || arguments.reserved_zero != 0 ||
	    arguments.rank_count == 0 ||
	    arguments.rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    arguments.rank >= arguments.rank_count ||
	    arguments.element_count == 0 ||
	    arguments.element_count > ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS ||
	    arguments.payload_format > ORDERED_PUSH_BF16 ||
	    blockDim.x != ORDERED_COLLECTIVE_THREADS ||
	    gridDim.x != ORDERED_PUSH_PARALLEL_BLOCKS)
		asm volatile("trap;");
	push_publish_values(arguments, arena, step->sequence,
			    blockIdx.x * blockDim.x + threadIdx.x,
			    gridDim.x * blockDim.x);
	/* Each writer fences its own remote payload stores before the later stream-ordered READY kernel. */
	__threadfence_system();
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	ordered_push_parallel_reduce(struct ordered_push_arguments arguments,
				     unsigned char *arena)
{
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	const uint32_t generation =
		push_begin_generation(arguments, arena, step->sequence);
	push_publish_phase(arguments, arena, step->sequence, ORDERED_PUSH_READY,
			   generation);
	push_finish(arguments, arena, step->sequence, generation);
}
