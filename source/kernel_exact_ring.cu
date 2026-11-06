/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/exact_ring_records.h"
#include "kernel_cut_through.cuh"

extern "C" __global__ void
exact_ring(const struct exact_ring_arguments __grid_constant__ arguments,
	   unsigned char *arena)
{
	const struct ordered_collective_peer *const peers =
		(const struct ordered_collective_peer
			 *)(arena + arguments.peer_table_offset);
	const uint32_t next_rank = (arguments.rank + 1u) % BB_RING_RANKS;
	const bool columns = arguments.mode == EXACT_RING_BF16_SUM_COLUMNS;
	if (arguments.mode > EXACT_RING_BF16_SUM_COLUMNS ||
	    (columns && (arguments.vectors == 0 ||
			 arguments.packs_per_owner % arguments.vectors != 0)) ||
	    arguments.reserved_words[0] != 0 ||
	    arguments.reserved_words[1] != 0 ||
	    arguments.rank >= BB_RING_RANKS ||
	    peers[next_rank].route != ORDERED_COLLECTIVE_DIRECT_ROUTE)
		asm volatile("trap;");
	struct bb_cut_through ring = { 0 };
	ring.input = (uint64_t)(arena + arguments.input_offset);
	ring.output = (uint64_t)(arena + arguments.output_offset);
	ring.local_packets = (uint64_t)(arena + arguments.bank_offset +
					EXACT_RING_METADATA_BYTES);
	ring.next_packets = peers[next_rank].arena_address +
			    peers[next_rank].region_offset +
			    arguments.peer_bank_delta +
			    EXACT_RING_METADATA_BYTES;
	ring.generations = (uint64_t)(arena + arguments.bank_offset);
	ring.status = ring.generations + EXACT_RING_BLOCKS * sizeof(uint32_t);
	ring.deadline_cycles = arguments.deadline_cycles;
	ring.maximum_packs = arguments.packs_per_owner;
	ring.rank = arguments.rank;
	const bool sums = arguments.mode == EXACT_RING_BF16_SUM || columns;
	ring.mode = sums ? BB_RING_BF16_SUM : BB_RING_LOSSLESS_GATHER;
	ring.output_vectors = sums ? 0 : arguments.vectors;
	ring.sum_row_packs =
		columns ? arguments.packs_per_owner / arguments.vectors : 0;
	for (uint32_t owner = 0; owner < BB_RING_RANKS; owner++) {
		ring.first_pack[owner] = owner * arguments.packs_per_owner;
		ring.pack_count[owner] = arguments.packs_per_owner;
	}
	cut_through_execute(ring);
	__syncthreads();
	if (*(const volatile uint32_t *)ring.status != 0)
		asm volatile("trap;");
}

/* The serving entry above specializes equal owners. Reuse its packet codec and bounded waits, while this
 * independent gather schedules each hop's actual positive owner width and preserves the unequal transpose. */
__device__ static __forceinline__ void
unequal_ring_output(uint32_t first_pack, uint32_t pack_count,
		    uint32_t total_packs, uint32_t vectors, uint4 *output,
		    uint32_t pack, uint4 value)
{
	uint64_t destination = (uint64_t)first_pack + pack;
	if (vectors != 0) {
		const uint32_t width = pack_count / vectors;
		destination =
			(uint64_t)(pack / width) * (total_packs / vectors) +
			first_pack / vectors + pack % width;
	}
	output[destination] = value;
}

__device__ static __forceinline__ void
unequal_ring_execute(const struct bb_cut_through arguments,
		     const struct exact_ring_unequal_arguments *geometry)
{
	__shared__ uint32_t generation;
	const uint64_t start = clock64();
	if (threadIdx.x == 0) {
		uint32_t *const counter =
			(uint32_t *)arguments.generations + blockIdx.x;
		generation = *counter + 1u;
		*counter = generation;
		if (generation == 0)
			atomicExch((unsigned int *)arguments.status,
				   BB_RING_GENERATION_EXHAUSTED);
	}
	__syncthreads();
	if (generation == 0)
		return;
	const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t stride = gridDim.x * blockDim.x;
	const uint4 *const input = (const uint4 *)arguments.input;
	uint4 *const output = (uint4 *)arguments.output;
	const uint32_t rank = geometry->rank;
	const uint32_t total_packs = geometry->first_pack[RING_LAST_RANK] +
				     geometry->pack_count[RING_LAST_RANK];
	/* Geometry stays in immutable kernel-parameter storage; copying its dynamically indexed arrays into a
     * local record would force a per-thread stack allocation. Packet-codec fields are scalarized independently. */
	/* A uint64 loop cannot wrap at the last admitted uint32 packet index. */
	for (uint64_t packet = first; packet < geometry->pack_count[rank];
	     packet += stride) {
		const uint32_t pack = (uint32_t)packet;
		const uint4 value =
			input[(uint64_t)geometry->first_pack[rank] + pack];
		unequal_ring_output(geometry->first_pack[rank],
				    geometry->pack_count[rank], total_packs,
				    geometry->vectors, output, pack, value);
		if (!ring_send(arguments, 0, pack, generation, value, start))
			return;
	}
	for (uint32_t hop = 0; hop < RING_LAST_RANK; hop++) {
		const uint32_t owner =
			(rank + BB_RING_RANKS - 1u - hop) % BB_RING_RANKS;
		/* Every hop has a real positive producer. A narrow owner's absent tail is skipped only at this hop;
		 * wider owners at later hops still forward all of their own packets. No empty-owner proof is used. */
		for (uint64_t packet = first;
		     packet < geometry->pack_count[owner]; packet += stride) {
			const uint32_t pack = (uint32_t)packet;
			uint4 value;
			if (!ring_receive(arguments, hop, pack, generation,
					  &value, start))
				return;
			unequal_ring_output(geometry->first_pack[owner],
					    geometry->pack_count[owner],
					    total_packs, geometry->vectors,
					    output, pack, value);
			if (hop + 1u < RING_LAST_RANK &&
			    !ring_send(arguments, hop + 1u, pack, generation,
				       value, start))
				return;
		}
	}
}

extern "C" __global__ void exact_ring_unequal(
	const struct exact_ring_unequal_arguments __grid_constant__ arguments,
	unsigned char *arena)
{
	const struct ordered_collective_peer *const peers =
		(const struct ordered_collective_peer
			 *)(arena + arguments.peer_table_offset);
	bool invalid =
		arguments.rank >= EXACT_RING_RANKS ||
		arguments.reserved_zero != 0 || arguments.maximum_packs == 0 ||
		arguments.deadline_cycles == 0 ||
		arguments.deadline_cycles > EXACT_RING_MAXIMUM_DEADLINE ||
		gridDim.x > EXACT_RING_BLOCKS ||
		blockDim.x != BB_RING_THREADS || gridDim.y != 1 ||
		gridDim.z != 1 || blockDim.y != 1 || blockDim.z != 1;
	uint64_t total_packs = 0;
	uint32_t maximum_owners = 0;
	for (uint32_t owner = 0; owner < EXACT_RING_RANKS; owner++) {
		invalid =
			invalid || arguments.first_pack[owner] != total_packs ||
			arguments.pack_count[owner] == 0 ||
			arguments.pack_count[owner] > arguments.maximum_packs ||
			(arguments.vectors != 0 &&
			 arguments.pack_count[owner] % arguments.vectors !=
				 0) ||
			peers[owner].route != ORDERED_COLLECTIVE_DIRECT_ROUTE ||
			peers[owner].reserved_zero != 0;
		total_packs += arguments.pack_count[owner];
		maximum_owners +=
			arguments.pack_count[owner] == arguments.maximum_packs ?
				1u :
				0u;
	}
	if (invalid || total_packs > UINT32_MAX ||
	    maximum_owners < EXACT_RING_REQUIRED_MAXIMUM_OWNERS)
		asm volatile("trap;");
	const uint32_t next_rank = (arguments.rank + 1u) % BB_RING_RANKS;
	struct bb_cut_through ring = { 0 };
	ring.input = (uint64_t)(arena + arguments.input_offset);
	ring.output = (uint64_t)(arena + arguments.output_offset);
	ring.local_packets = (uint64_t)(arena + arguments.bank_offset +
					EXACT_RING_METADATA_BYTES);
	ring.next_packets = peers[next_rank].arena_address +
			    peers[next_rank].region_offset +
			    arguments.peer_bank_delta +
			    EXACT_RING_METADATA_BYTES;
	ring.generations = (uint64_t)(arena + arguments.bank_offset);
	ring.status = ring.generations + EXACT_RING_BLOCKS * sizeof(uint32_t);
	if (*(const volatile uint32_t *)ring.status != 0)
		asm volatile("trap;");
	ring.deadline_cycles = arguments.deadline_cycles;
	ring.maximum_packs = arguments.maximum_packs;
	ring.rank = arguments.rank;
	ring.mode = BB_RING_LOSSLESS_GATHER;
	ring.output_vectors = arguments.vectors;
	unequal_ring_execute(ring, &arguments);
	__syncthreads();
	if (*(const volatile uint32_t *)ring.status != 0)
		asm volatile("trap;");
}
