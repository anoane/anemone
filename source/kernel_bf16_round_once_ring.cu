/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/round_once_ring_records.h"
#include "kernel_cut_through.cuh"
#include "kernel_linear_common.cuh"

__device__ static __forceinline__ uint4 *
round_once_bank(const struct exact_ring_arguments *arguments,
		unsigned char *arena,
		const struct ordered_collective_peer *peers, uint32_t rank)
{
	const uint64_t bank =
		rank == arguments->rank ?
			(uint64_t)(arena + arguments->bank_offset) :
			peers[rank].arena_address + peers[rank].region_offset +
				arguments->peer_bank_delta;
	return (uint4 *)(bank + EXACT_RING_METADATA_BYTES);
}

__device__ static __forceinline__ uint32_t *
round_once_signal(uint4 *bank, uint32_t packs, uint32_t phase, uint32_t rank)
{
	uint32_t *const flags =
		(uint32_t *)(bank +
			     ROUND_ONCE_RING_PLANES * BB_RING_RANKS * packs);
	return flags + (phase * BB_RING_RANKS + rank) * EXACT_RING_BLOCKS +
	       blockIdx.x;
}

/* All threads publish their own stores before the leader's release; its acquire reaches the other threads through
 * the trailing CTA barrier. Each CTA's packet partition is independent, so no other CTA must be scheduled to progress. */
__device__ static __forceinline__ bool
round_once_phase(const struct exact_ring_arguments *arguments,
		 unsigned char *arena,
		 const struct ordered_collective_peer *peers,
		 struct bb_cut_through channel, uint32_t phase,
		 uint32_t generation, uint64_t start)
{
	__shared__ uint32_t okay;
	__threadfence_system();
	__syncthreads();
	if (threadIdx.x == 0) {
		okay = 1;
		for (uint32_t rank = 0; rank < BB_RING_RANKS; rank++)
			ring_credit(round_once_signal(
					    round_once_bank(arguments, arena,
							    peers, rank),
					    arguments->packs_per_owner, phase,
					    arguments->rank),
				    generation);
		uint4 *const local = round_once_bank(arguments, arena, peers,
						     arguments->rank);
		for (uint32_t rank = 0; rank < BB_RING_RANKS; rank++) {
			const uint32_t *const signal = round_once_signal(
				local, arguments->packs_per_owner, phase, rank);
			while (ring_load_credit(signal) != generation) {
				if (ring_expired(channel, start)) {
					okay = 0;
					break;
				}
			}
			if (okay == 0)
				break;
		}
	}
	__syncthreads();
	return okay != 0;
}

__device__ static __forceinline__ uint32_t round_once_pair(uint32_t first,
							   uint32_t second,
							   uint32_t third,
							   uint32_t fourth)
{
	float low = 0.0f, high = 0.0f;
	low = low + linear_bfloat16_value(first);
	high = high + linear_bfloat16_value(first >> BITS_PER_HALF);
	low = low + linear_bfloat16_value(second);
	high = high + linear_bfloat16_value(second >> BITS_PER_HALF);
	low = low + linear_bfloat16_value(third);
	high = high + linear_bfloat16_value(third >> BITS_PER_HALF);
	low = low + linear_bfloat16_value(fourth);
	high = high + linear_bfloat16_value(fourth >> BITS_PER_HALF);
	return linear_bfloat16_bits(low) |
	       ((uint32_t)linear_bfloat16_bits(high) << BITS_PER_HALF);
}

__device__ static __forceinline__ void
round_once_execute(const struct exact_ring_arguments *arguments,
		   unsigned char *arena,
		   const struct ordered_collective_peer *peers,
		   struct bb_cut_through channel)
{
	__shared__ uint32_t generation;
	const uint64_t start = clock64();
	if (threadIdx.x == 0) {
		uint32_t *const counter =
			(uint32_t *)channel.generations + blockIdx.x;
		generation = *counter + 1u;
		*counter = generation;
		if (generation == 0)
			atomicExch((unsigned int *)channel.status,
				   BB_RING_GENERATION_EXHAUSTED);
	}
	__syncthreads();
	if (generation == 0)
		return;
	const uint32_t packs = arguments->packs_per_owner,
		       rank = arguments->rank;
	const uint32_t row_packs = arguments->mode ==
						   EXACT_RING_BF16_SUM_COLUMNS ?
					   packs / arguments->vectors :
					   0;
	const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x,
		       stride = gridDim.x * blockDim.x;
	const uint4 *const input =
		(const uint4 *)(arena + arguments->input_offset);
	uint4 *const output = (uint4 *)(arena + arguments->output_offset);
	for (uint32_t owner = 0; owner < BB_RING_RANKS; owner++) {
		uint4 *const incoming =
			round_once_bank(arguments, arena, peers, owner) +
			rank * packs;
		for (uint32_t pack = first; pack < packs; pack += stride)
			ring_store_pack(incoming + pack,
					input[ring_sum_place(packs, row_packs,
							     owner, pack)]);
	}
	if (!round_once_phase(arguments, arena, peers, channel,
			      ROUND_ONCE_RING_READY, generation, start))
		return;
	uint4 *const local = round_once_bank(arguments, arena, peers, rank);
	for (uint32_t pack = first; pack < packs; pack += stride) {
		const uint4 rank0 = ring_load_pack(local + pack),
			    rank1 = ring_load_pack(local + packs + pack),
			    rank2 = ring_load_pack(
				    local + ROUND_ONCE_RING_THIRD_RANK * packs +
				    pack),
			    rank3 = ring_load_pack(
				    local +
				    ROUND_ONCE_RING_FOURTH_RANK * packs + pack);
		const uint4 sum = make_uint4(
			round_once_pair(rank0.x, rank1.x, rank2.x, rank3.x),
			round_once_pair(rank0.y, rank1.y, rank2.y, rank3.y),
			round_once_pair(rank0.z, rank1.z, rank2.z, rank3.z),
			round_once_pair(rank0.w, rank1.w, rank2.w, rank3.w));
		for (uint32_t peer = 0; peer < BB_RING_RANKS; peer++) {
			uint4 *const result =
				round_once_bank(arguments, arena, peers, peer) +
				BB_RING_RANKS * packs;
			ring_store_pack(result + rank * packs + pack, sum);
		}
	}
	if (!round_once_phase(arguments, arena, peers, channel,
			      ROUND_ONCE_RING_DONE, generation, start))
		return;
	const uint4 *const result = local + BB_RING_RANKS * packs;
	for (uint32_t owner = 0; owner < BB_RING_RANKS; owner++)
		for (uint32_t pack = first; pack < packs; pack += stride)
			output[ring_sum_place(packs, row_packs, owner, pack)] =
				ring_load_pack(result + owner * packs + pack);
}

extern "C" __global__ void bf16_round_once_ring(
	const struct exact_ring_arguments __grid_constant__ arguments,
	unsigned char *arena)
{
	const struct ordered_collective_peer *const peers =
		(const struct ordered_collective_peer
			 *)(arena + arguments.peer_table_offset);
	const bool columns = arguments.mode == EXACT_RING_BF16_SUM_COLUMNS;
	if ((arguments.mode != EXACT_RING_BF16_SUM && !columns) ||
	    arguments.packs_per_owner == 0 ||
	    arguments.packs_per_owner % ROUND_ONCE_RING_PACKS != 0 ||
	    arguments.rank >= BB_RING_RANKS ||
	    (columns && (arguments.vectors == 0 ||
			 arguments.packs_per_owner % arguments.vectors != 0)) ||
	    arguments.reserved_words[0] != 0 ||
	    arguments.reserved_words[1] != 0 ||
	    blockDim.x != ROUND_ONCE_RING_THREADS ||
	    gridDim.x > EXACT_RING_BLOCKS)
		asm volatile("trap;");
	for (uint32_t rank = 0; rank < BB_RING_RANKS; rank++)
		if (rank != arguments.rank &&
		    peers[rank].route != ORDERED_COLLECTIVE_DIRECT_ROUTE)
			asm volatile("trap;");
	struct bb_cut_through channel = { 0 };
	channel.generations = (uint64_t)(arena + arguments.bank_offset);
	channel.status =
		channel.generations + EXACT_RING_BLOCKS * sizeof(uint32_t);
	channel.deadline_cycles = arguments.deadline_cycles;
	round_once_execute(&arguments, arena, peers, channel);
	__syncthreads();
	if (*(const volatile uint32_t *)channel.status != 0)
		asm volatile("trap;");
}
