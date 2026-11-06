/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/compressed_ring_records.h"
#include "bf16_wire_codec.h"
#include "kernel_cut_through.cuh"

__device__ static __forceinline__ uint32_t pair_top(uint32_t value)
{
	return max(bf16_wire_low_exponent(value),
		   bf16_wire_high_exponent(value));
}

__device__ static __forceinline__ uint32_t pair_bottom(uint32_t value)
{
	return min(bf16_wire_low_exponent(value),
		   bf16_wire_high_exponent(value));
}

/* Leader's acquire, followed by a warp memory barrier, precedes every lane's payload reads. All returns are uniform. */
__device__ static __forceinline__ bool
compressed_wait(struct bb_cut_through ring, const uint32_t *address,
		uint32_t wanted, uint64_t start)
{
	const uint32_t lane = threadIdx.x % COMPRESSED_RING_PACKS;
	uint32_t okay = 1;
	if (lane == 0) {
		while (ring_load_credit(address) != wanted) {
			if (ring_expired(ring, start)) {
				okay = 0;
				break;
			}
		}
	}
	okay = __shfl_sync(COMPRESSED_RING_WARP_MASK, okay, 0);
	__syncwarp();
	return okay != 0;
}

__device__ static __forceinline__ uint32_t *
compressed_packet(uint64_t bank, uint32_t groups, uint32_t hop, uint32_t group)
{
	return (uint32_t *)bank +
	       ((uint64_t)hop * groups + group) * COMPRESSED_RING_PACKET_WORDS;
}

__device__ static __forceinline__ uint32_t compressed_place(uint32_t mask,
							    uint32_t lane)
{
	const uint32_t earlier = (1u << lane) - 1u;
	return COMPRESSED_RING_PAYLOAD_WORD +
	       lane * COMPRESSED_RING_COMPRESSED_WORDS +
	       (uint32_t)__popc(mask & earlier);
}

__device__ static __forceinline__ bool
compressed_send(struct bb_cut_through ring, uint32_t groups, uint32_t hop,
		uint32_t group, uint32_t generation, uint4 value,
		uint64_t start)
{
	uint32_t *const local =
		compressed_packet(ring.local_packets, groups, hop, group);
	if (!compressed_wait(ring, local + COMPRESSED_RING_ACK_WORD,
			     generation - 1u, start))
		return false;
	/* Keep the old BF16 transport's zero convention, independently of the lossless codec. */
	value = make_uint4(ring_nonzero_halves(value.x),
			   ring_nonzero_halves(value.y),
			   ring_nonzero_halves(value.z),
			   ring_nonzero_halves(value.w));
	uint32_t top = max(max(pair_top(value.x), pair_top(value.y)),
			   max(pair_top(value.z), pair_top(value.w)));
	top = __reduce_max_sync(COMPRESSED_RING_WARP_MASK, top);
	const uint32_t bottom =
		min(min(pair_bottom(value.x), pair_bottom(value.y)),
		    min(pair_bottom(value.z), pair_bottom(value.w)));
	const bool raw = top - bottom > BF16_WIRE_DELTA_MASK;
	const uint32_t mask = __ballot_sync(COMPRESSED_RING_WARP_MASK, raw);
	const uint32_t lane = threadIdx.x % COMPRESSED_RING_PACKS;
	uint32_t *const target =
		compressed_packet(ring.next_packets, groups, hop, group);
	volatile uint32_t *const payload =
		target + compressed_place(mask, lane);
	if (raw) {
		payload[0] = value.x;
		payload[1] = value.y;
		payload[RING_THIRD_WORD] = value.z;
		payload[RING_FOURTH_WORD] = value.w;
	} else {
		const uint32_t first = bf16_wire_pair_pack(value.x, top);
		const uint32_t second = bf16_wire_pair_pack(value.y, top);
		const uint32_t third = bf16_wire_pair_pack(value.z, top);
		const uint32_t fourth = bf16_wire_pair_pack(value.w, top);
		payload[0] = (first & RING_BF16_LOW_MASK) |
			     (second << BF16_WIRE_HALF_BITS);
		payload[1] = (third & RING_BF16_LOW_MASK) |
			     (fourth << BF16_WIRE_HALF_BITS);
		payload[RING_THIRD_WORD] =
			(first >> BF16_WIRE_HALF_BITS) |
			((second >> BF16_WIRE_HALF_BITS)
			 << BF16_WIRE_BYTE_BITS) |
			((third >> BF16_WIRE_HALF_BITS)
			 << BF16_WIRE_HALF_BITS) |
			((fourth >> BF16_WIRE_HALF_BITS)
			 << (BF16_WIRE_HALF_BITS + BF16_WIRE_BYTE_BITS));
	}
	if (lane == 0) {
		ring_bitmap_store(target, top);
		ring_bitmap_store(target + COMPRESSED_RING_RAW_MASK_WORD, mask);
	}
	/* Every publishing lane orders its own stores before the leader's READY, including remote writes. */
	__threadfence_system();
	__syncwarp();
	if (lane == 0)
		ring_credit(target + COMPRESSED_RING_READY_WORD, generation);
	return true;
}

__device__ static __forceinline__ bool
compressed_receive(struct bb_cut_through ring, uint64_t previous_packets,
		   uint32_t groups, uint32_t hop, uint32_t group,
		   uint32_t generation, uint4 *value, uint64_t start)
{
	uint32_t *const local =
		compressed_packet(ring.local_packets, groups, hop, group);
	if (!compressed_wait(ring, local + COMPRESSED_RING_READY_WORD,
			     generation, start))
		return false;
	const uint32_t lane = threadIdx.x % COMPRESSED_RING_PACKS;
	const uint32_t top = ring_bitmap_load(local);
	const uint32_t mask =
		ring_bitmap_load(local + COMPRESSED_RING_RAW_MASK_WORD);
	const volatile uint32_t *const payload =
		local + compressed_place(mask, lane);
	const uint32_t first = payload[0], second = payload[1],
		       third = payload[RING_THIRD_WORD];
	if ((mask & (1u << lane)) != 0) {
		*value = make_uint4(first, second, third,
				    payload[RING_FOURTH_WORD]);
	} else {
		*value = make_uint4(
			bf16_wire_pair_unpack(first, third, top),
			bf16_wire_pair_unpack(first >> BF16_WIRE_HALF_BITS,
					      third >> BF16_WIRE_BYTE_BITS,
					      top),
			bf16_wire_pair_unpack(
				second, third >> BF16_WIRE_HALF_BITS, top),
			bf16_wire_pair_unpack(second >> BF16_WIRE_HALF_BITS,
					      third >> (BF16_WIRE_HALF_BITS +
							BF16_WIRE_BYTE_BITS),
					      top));
	}
	/* No sender can reuse this packet until all lanes have read its payload. */
	__syncwarp();
	if (lane == 0)
		ring_credit(compressed_packet(previous_packets, groups, hop,
					      group) +
				    COMPRESSED_RING_ACK_WORD,
			    generation);
	return true;
}

__device__ static __forceinline__ void
compressed_execute(struct bb_cut_through ring, uint64_t previous_packets)
{
	__shared__ uint32_t generation;
	const uint64_t start = clock64();
	if (threadIdx.x == 0) {
		uint32_t *const counter =
			(uint32_t *)ring.generations + blockIdx.x;
		generation = *counter + 1u;
		*counter = generation;
		if (generation == 0)
			atomicExch((unsigned int *)ring.status,
				   BB_RING_GENERATION_EXHAUSTED);
	}
	__syncthreads();
	if (generation == 0)
		return;
	const uint32_t lane = threadIdx.x % COMPRESSED_RING_PACKS;
	const uint32_t first =
		(blockIdx.x * blockDim.x + threadIdx.x) / COMPRESSED_RING_PACKS;
	const uint32_t stride = gridDim.x * blockDim.x / COMPRESSED_RING_PACKS;
	const uint32_t groups = ring.maximum_packs / COMPRESSED_RING_PACKS;
	const uint4 *const input = (const uint4 *)ring.input;
	uint4 *const output = (uint4 *)ring.output;
	const uint32_t rank = ring.rank, row_packs = ring.sum_row_packs;
	for (uint32_t group = first; group < groups; group += stride) {
		const uint32_t pack = group * COMPRESSED_RING_PACKS + lane;
		const uint32_t owner = (rank + RING_LAST_RANK) % BB_RING_RANKS;
		if (!compressed_send(
			    ring, groups, 0, group, generation,
			    input[ring_sum_place(ring.maximum_packs, row_packs,
						 owner, pack)],
			    start))
			return;
	}
	for (uint32_t hop = 0; hop < RING_LAST_RANK; hop++) {
		const uint32_t owner =
			(rank + BB_RING_RANKS + RING_LAST_RANK - 1u - hop) %
			BB_RING_RANKS;
		for (uint32_t group = first; group < groups; group += stride) {
			const uint32_t pack =
				group * COMPRESSED_RING_PACKS + lane;
			uint4 value;
			if (!compressed_receive(ring, previous_packets, groups,
						hop, group, generation, &value,
						start))
				return;
			value = ring_bf16_sum(
				value,
				input[ring_sum_place(ring.maximum_packs,
						     row_packs, owner, pack)]);
			if (hop + 1u == RING_LAST_RANK)
				output[ring_sum_place(ring.maximum_packs,
						      row_packs, owner, pack)] =
					value;
			if (!compressed_send(ring, groups, hop + 1u, group,
					     generation, value, start))
				return;
		}
	}
	for (uint32_t hop = RING_LAST_RANK; hop < BB_RING_HOPS; hop++) {
		const uint32_t owner =
			(rank + BB_RING_RANKS + RING_LAST_RANK - 1u - hop) %
			BB_RING_RANKS;
		for (uint32_t group = first; group < groups; group += stride) {
			const uint32_t pack =
				group * COMPRESSED_RING_PACKS + lane;
			uint4 value;
			if (!compressed_receive(ring, previous_packets, groups,
						hop, group, generation, &value,
						start))
				return;
			output[ring_sum_place(ring.maximum_packs, row_packs,
					      owner, pack)] = value;
			if (hop + 1u < BB_RING_HOPS &&
			    !compressed_send(ring, groups, hop + 1u, group,
					     generation, value, start))
				return;
		}
	}
}

extern "C" __global__ void bf16_compressed_ring(
	const struct exact_ring_arguments __grid_constant__ arguments,
	unsigned char *arena)
{
	const struct ordered_collective_peer *const peers =
		(const struct ordered_collective_peer
			 *)(arena + arguments.peer_table_offset);
	const uint32_t next = (arguments.rank + 1u) % BB_RING_RANKS;
	const uint32_t previous =
		(arguments.rank + RING_LAST_RANK) % BB_RING_RANKS;
	const bool columns = arguments.mode == EXACT_RING_BF16_SUM_COLUMNS;
	if ((arguments.mode != EXACT_RING_BF16_SUM && !columns) ||
	    arguments.packs_per_owner == 0 ||
	    arguments.packs_per_owner % COMPRESSED_RING_PACKS != 0 ||
	    arguments.rank >= BB_RING_RANKS ||
	    (arguments.vectors == 0 && columns) ||
	    (columns && arguments.packs_per_owner % arguments.vectors != 0) ||
	    arguments.reserved_words[0] != 0 ||
	    arguments.reserved_words[1] != 0 ||
	    peers[next].route != ORDERED_COLLECTIVE_DIRECT_ROUTE ||
	    peers[previous].route != ORDERED_COLLECTIVE_DIRECT_ROUTE ||
	    blockDim.x != COMPRESSED_RING_THREADS ||
	    gridDim.x > EXACT_RING_BLOCKS)
		asm volatile("trap;");
	struct bb_cut_through ring = { 0 };
	ring.input = (uint64_t)(arena + arguments.input_offset);
	ring.output = (uint64_t)(arena + arguments.output_offset);
	ring.local_packets = (uint64_t)(arena + arguments.bank_offset +
					EXACT_RING_METADATA_BYTES);
	ring.next_packets =
		peers[next].arena_address + peers[next].region_offset +
		arguments.peer_bank_delta + EXACT_RING_METADATA_BYTES;
	const uint64_t previous_packets =
		peers[previous].arena_address + peers[previous].region_offset +
		arguments.peer_bank_delta + EXACT_RING_METADATA_BYTES;
	ring.generations = (uint64_t)(arena + arguments.bank_offset);
	ring.status = ring.generations + EXACT_RING_BLOCKS * sizeof(uint32_t);
	ring.deadline_cycles = arguments.deadline_cycles;
	ring.maximum_packs = arguments.packs_per_owner;
	ring.rank = arguments.rank;
	ring.sum_row_packs =
		columns ? arguments.packs_per_owner / arguments.vectors : 0;
	compressed_execute(ring, previous_packets);
	__syncthreads();
	if (*(const volatile uint32_t *)ring.status != 0)
		asm volatile("trap;");
}
