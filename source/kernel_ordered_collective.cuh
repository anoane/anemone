/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Shared exact collective body. Product entries and the packet's device-scope benchmark instantiate this code. */
#ifndef KERNEL_ORDERED_COLLECTIVE_CUH
#define KERNEL_ORDERED_COLLECTIVE_CUH
#include "anemone/ordered_collective_records.h"
#include "kernel_mailbox_signal.cuh"

#define ORDERED_READY_PHASE 0u
#define ORDERED_REDUCED_PHASE 1u
#define ORDERED_DONE_PHASE 2u
#define ORDERED_WAIT_BACKOFF_CYCLES 64u
#define ORDERED_SCALAR_MAXIMUM_ELEMENTS 16u

__device__ static __forceinline__ const struct ordered_collective_peer *
ordered_peers(struct ordered_collective_arguments arguments,
	      unsigned char *arena)
{
	return (const struct ordered_collective_peer
			*)(arena + arguments.peer_table_offset);
}

__device__ static __forceinline__ unsigned char *
ordered_rank_slot(struct ordered_collective_arguments arguments,
		  unsigned char *arena, uint32_t rank, uint64_t sequence)
{
	const struct ordered_collective_peer *const peers =
		ordered_peers(arguments, arena);
	unsigned char *const region =
		rank == arguments.rank ?
			arena + arguments.region_offset :
			(unsigned char *)(uintptr_t)peers[rank].arena_address +
				peers[rank].region_offset;
	return region + (sequence & 1u) * arguments.parity_stride;
}

__device__ static __forceinline__ unsigned char *
ordered_host_slot(struct ordered_collective_arguments arguments,
		  unsigned char *arena, uint64_t sequence)
{
	const uint64_t address =
		ordered_peers(arguments, arena)[arguments.rank].host_address;
	return address == 0 ? NULL :
			      (unsigned char *)(uintptr_t)address +
				      (sequence & 1u) * arguments.parity_stride;
}

__device__ static __forceinline__ uint32_t ordered_value_load(
	struct ordered_collective_arguments arguments, unsigned char *arena,
	uint32_t rank, uint64_t sequence, uint64_t offset, uint32_t element)
{
	const uint32_t *const values =
		(const uint32_t *)(ordered_rank_slot(arguments, arena, rank,
						     sequence) +
				   offset);
	if (rank != arguments.rank &&
	    ordered_peers(arguments, arena)[rank].route ==
		    ORDERED_COLLECTIVE_HOST_ROUTE) {
		uint32_t value;
		/* Host mirrors cross noncoherent caches. Re-fetch their published words from system memory. */
		asm volatile("ld.global.cv.u32 %0, [%1];"
			     : "=r"(value)
			     : "l"(values + element)
			     : "memory");
		return value;
	}
	return __ldcg(values + element);
}

__device__ static __forceinline__ void
ordered_wait_phase(struct ordered_collective_arguments arguments,
		   unsigned char *arena, uint64_t sequence, uint32_t phase,
		   uint32_t generation)
{
	/* Independent lanes observe different ranks concurrently, then distribute their acquires at the CTA barrier. */
	if (threadIdx.x < arguments.rank_count) {
		const uint32_t rank = threadIdx.x;
		const uint64_t started = clock64();
		const unsigned char *const line =
			ordered_rank_slot(arguments, arena, rank, sequence) +
			(uint64_t)phase * ORDERED_COLLECTIVE_LINE_BYTES;
		while (mailbox_acquire32(line) != generation) {
			if (clock64() - started >= arguments.deadline_cycles) {
				asm volatile("trap;");
			}
			__nanosleep(ORDERED_WAIT_BACKOFF_CYCLES);
		}
		if (phase == ORDERED_READY_PHASE &&
		    mailbox_sequence_load(line +
					  ORDERED_COLLECTIVE_SEQUENCE_OFFSET) !=
			    sequence) {
			asm volatile("trap;");
		}
	}
	__syncthreads();
}

__device__ static __forceinline__ uint32_t
ordered_publish_phase(struct ordered_collective_arguments arguments,
		      unsigned char *arena, uint64_t sequence, uint32_t phase,
		      uint64_t source_offset, uint32_t begin, uint32_t end)
{
	unsigned char *const own_slot =
		ordered_rank_slot(arguments, arena, arguments.rank, sequence);
	unsigned char *const host_slot =
		ordered_host_slot(arguments, arena, sequence);
	if (host_slot != NULL) {
		if (phase == ORDERED_REDUCED_PHASE) {
			/* Each thread reloads its own scratch stores. First make them visible through L2. */
			__threadfence();
		}
		const uint32_t *const source =
			(const uint32_t *)(own_slot + source_offset);
		uint32_t *const destination =
			(uint32_t *)(host_slot + source_offset);
		for (uint32_t element = begin + threadIdx.x; element < end;
		     element += blockDim.x) {
			destination[element] = __ldcg(source + element);
		}
	}
	/* Every payload writer releases its stores before the block's single 32-bit commit publication. */
	__threadfence_system();
	__syncthreads();
	__shared__ uint32_t generation;
	if (threadIdx.x == 0) {
		unsigned char *const own_line =
			own_slot +
			(uint64_t)phase * ORDERED_COLLECTIVE_LINE_BYTES;
		generation = mailbox_acquire32(own_line) + 1u;
		if (phase == ORDERED_READY_PHASE) {
			mailbox_sequence_store(
				own_line + ORDERED_COLLECTIVE_SEQUENCE_OFFSET,
				sequence);
		}
		if (host_slot != NULL) {
			unsigned char *const host_line =
				host_slot +
				(uint64_t)phase * ORDERED_COLLECTIVE_LINE_BYTES;
			if (phase == ORDERED_READY_PHASE) {
				mailbox_sequence_store(
					host_line +
						ORDERED_COLLECTIVE_SEQUENCE_OFFSET,
					sequence);
			}
			mailbox_release32(host_line, generation);
		}
		mailbox_release32(own_line, generation);
	}
	__syncthreads();
	return generation;
}

__device__ static __forceinline__ void
ordered_run_sequence(struct ordered_collective_arguments arguments,
		     unsigned char *arena, uint32_t operation,
		     uint64_t sequence)
{
	if (sequence == 0 || arguments.reserved_zero != 0 ||
	    arguments.deadline_cycles == 0 || arguments.rank_count == 0 ||
	    arguments.rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    arguments.rank >= arguments.rank_count ||
	    arguments.element_count == 0 ||
	    arguments.element_count > ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS ||
	    blockDim.x != ORDERED_COLLECTIVE_THREADS || gridDim.x != 1u) {
		asm volatile("trap;");
	}
	if (threadIdx.x == 0) {
		for (uint32_t slot = 0; slot < ORDERED_COLLECTIVE_SLOTS;
		     slot++) {
			const unsigned char *const line =
				arena + arguments.region_offset +
				(uint64_t)slot * arguments.parity_stride;
			if (mailbox_sequence_load(
				    line +
				    ORDERED_COLLECTIVE_SEQUENCE_OFFSET) >=
			    sequence) {
				asm volatile("trap;");
			}
		}
		for (uint32_t rank = 0; rank < arguments.rank_count; rank++) {
			const struct ordered_collective_peer peer =
				ordered_peers(arguments, arena)[rank];
			if (peer.reserved_zero != 0 ||
			    peer.route > ORDERED_COLLECTIVE_HOST_ROUTE ||
			    (rank != arguments.rank &&
			     peer.arena_address == 0)) {
				asm volatile("trap;");
			}
		}
	}
	__syncthreads();
	unsigned char *const own_slot =
		ordered_rank_slot(arguments, arena, arguments.rank, sequence);
	float *const output = (float *)(own_slot + arguments.output_offset);
	float *const scratch = (float *)(own_slot + arguments.scratch_offset);
	const uint32_t slice =
		(arguments.element_count + arguments.rank_count - 1u) /
		arguments.rank_count;
	const uint32_t own_begin = arguments.rank * slice;
	const uint32_t own_end = own_begin + slice < arguments.element_count ?
					 own_begin + slice :
					 arguments.element_count;
	const uint32_t ready_begin = operation == ORDERED_GATHER ? own_begin :
								   0;
	const uint32_t ready_end =
		operation == ORDERED_GATHER ? own_end : arguments.element_count;
	uint32_t generation = ordered_publish_phase(arguments, arena, sequence,
						    ORDERED_READY_PHASE,
						    arguments.input_offset,
						    ready_begin, ready_end);
	ordered_wait_phase(arguments, arena, sequence, ORDERED_READY_PHASE,
			   generation);
	const bool partitioned = operation == ORDERED_TWO_SHOT ||
				 operation == ORDERED_REDUCE_SCATTER;
	const uint32_t begin = partitioned ? own_begin : 0;
	const uint32_t end = partitioned ? own_end : arguments.element_count;
	if (operation != ORDERED_GATHER) {
		for (uint32_t element = begin + threadIdx.x; element < end;
		     element += blockDim.x) {
			float sum = 0.0f;
			if (arguments.element_count <=
			    ORDERED_SCALAR_MAXIMUM_ELEMENTS) {
				/* Uniform tiny-message path retains C2's compact scalar rank loop. */
				for (uint32_t rank = 0;
				     rank < arguments.rank_count; rank++)
					sum = __fadd_rn(
						sum,
						__uint_as_float(ordered_value_load(
							arguments, arena, rank,
							sequence,
							arguments.input_offset,
							element)));
			} else {
				/* Independent loads precede the unchanged left-associated rank-order additions. */
				float inputs[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
#pragma unroll
				for (uint32_t rank = 0;
				     rank < ORDERED_COLLECTIVE_MAXIMUM_RANKS;
				     rank++) {
					if (rank < arguments.rank_count)
						inputs[rank] = __uint_as_float(
							ordered_value_load(
								arguments,
								arena, rank,
								sequence,
								arguments.input_offset,
								element));
				}
#pragma unroll
				for (uint32_t rank = 0;
				     rank < ORDERED_COLLECTIVE_MAXIMUM_RANKS;
				     rank++) {
					if (rank < arguments.rank_count)
						sum = __fadd_rn(sum,
								inputs[rank]);
				}
			}
			if (operation == ORDERED_TWO_SHOT) {
				scratch[element] = sum;
			} else {
				output[element] = sum;
			}
		}
	}
	if (operation == ORDERED_TWO_SHOT) {
		generation = ordered_publish_phase(arguments, arena, sequence,
						   ORDERED_REDUCED_PHASE,
						   arguments.scratch_offset,
						   own_begin, own_end);
		ordered_wait_phase(arguments, arena, sequence,
				   ORDERED_REDUCED_PHASE, generation);
	}
	if (operation == ORDERED_TWO_SHOT || operation == ORDERED_GATHER) {
		const uint64_t source_offset =
			operation == ORDERED_TWO_SHOT ?
				arguments.scratch_offset :
				arguments.input_offset;
		for (uint32_t element = threadIdx.x;
		     element < arguments.element_count; element += blockDim.x) {
			output[element] = __uint_as_float(ordered_value_load(
				arguments, arena, element / slice, sequence,
				source_offset, element));
		}
	}
	generation = ordered_publish_phase(arguments, arena, sequence,
					   ORDERED_DONE_PHASE, 0, 0, 0);
	ordered_wait_phase(arguments, arena, sequence, ORDERED_DONE_PHASE,
			   generation);
}

__device__ static __forceinline__ void
ordered_run(struct ordered_collective_arguments arguments, unsigned char *arena,
	    uint32_t operation)
{
	ordered_run_sequence(
		arguments, arena, operation,
		((const struct step_block *)(arena + arguments.step_offset))
			->sequence);
}
#endif
