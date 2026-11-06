/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original implementation of tagged, credited cut-through communication. */
#include "kernel_cut_through_record.cuh"

#define RING_WORD_BITS 32u
#define RING_BF16_PAIR_ONES 0x3f803f80u
#define RING_THIRD_WORD 2u
#define RING_FOURTH_WORD 3u
#define RING_LAST_RANK (BB_RING_RANKS - 1u)

__device__ static __forceinline__ uint32_t
ring_load_credit(const uint32_t *address)
{
	uint32_t value;
	asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
		     : "=r"(value)
		     : "l"(address)
		     : "memory");
	return value;
}

__device__ static __forceinline__ void ring_credit(uint32_t *address,
						   uint32_t generation)
{
	asm volatile("st.release.sys.global.u32 [%0], %1;"
		     :
		     : "l"(address), "r"(generation)
		     : "memory");
}

__device__ static __forceinline__ uint64_t
ring_load_word(const uint64_t *address)
{
	uint64_t value;
	asm volatile("ld.volatile.global.u64 %0, [%1];"
		     : "=l"(value)
		     : "l"(address)
		     : "memory");
	return value;
}

__device__ static __forceinline__ void
ring_store_pair(uint64_t *address, uint64_t first, uint64_t second)
{
	asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};"
		     :
		     : "l"(address), "l"(first), "l"(second)
		     : "memory");
}

__device__ static __forceinline__ bool
ring_expired(struct bb_cut_through arguments, uint64_t start)
{
	if (clock64() - start < arguments.deadline_cycles)
		return false;
	atomicExch((unsigned int *)arguments.status, BB_RING_DEADLINE);
	return true;
}

#define RING_BF16_LOW_MASK 0xffffu
#define RING_BF16_SIGN 0x8000u
#define RING_BF16_HALF_BITS 16u
#define RING_EPOCHS 2u

__device__ static __forceinline__ uint32_t ring_nonzero_halves(uint32_t word)
{
	const uint32_t low = word & RING_BF16_LOW_MASK;
	const uint32_t high = word >> RING_BF16_HALF_BITS;
	return (low == 0 ? RING_BF16_SIGN : low) |
	       ((high == 0 ? RING_BF16_SIGN : high) << RING_BF16_HALF_BITS);
}

__device__ static __forceinline__ bool ring_has_sentinel(uint32_t word)
{
	return (word & RING_BF16_LOW_MASK) == 0 ||
	       (word >> RING_BF16_HALF_BITS) == 0;
}

__device__ static __forceinline__ uint4 ring_load_pack(const uint4 *address)
{
	uint4 value;
	asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
		     : "=r"(value.x), "=r"(value.y), "=r"(value.z),
		       "=r"(value.w)
		     : "l"(address)
		     : "memory");
	return value;
}

__device__ static __forceinline__ void ring_store_pack(uint4 *address,
						       uint4 value)
{
	asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
		     :
		     : "l"(address), "r"(value.x), "r"(value.y), "r"(value.z),
		       "r"(value.w)
		     : "memory");
}

#define RING_ZERO_PAIR_BITS 2u
#define RING_THIRD_PAIR_SHIFT 4u
#define RING_FOURTH_PAIR_SHIFT 6u

__device__ static __forceinline__ uint32_t ring_zero_mask(uint32_t word)
{
	return ((word & RING_BF16_LOW_MASK) == 0 ? 1u : 0u) |
	       (((word >> RING_BF16_HALF_BITS) == 0 ? 1u : 0u) << 1u);
}
__device__ static __forceinline__ uint32_t ring_restore_zero(uint32_t word,
							     uint32_t mask)
{
	const uint32_t low = (mask & 1u) != 0 ? RING_BF16_LOW_MASK : 0;
	const uint32_t high = (mask & RING_ZERO_PAIR_BITS) != 0 ?
				      RING_BF16_LOW_MASK
					      << RING_BF16_HALF_BITS :
				      0;
	return word & ~(low | high);
}
__device__ static __forceinline__ uint32_t
ring_bitmap_load(const uint32_t *address)
{
	uint32_t value;
	asm volatile("ld.volatile.global.u32 %0, [%1];"
		     : "=r"(value)
		     : "l"(address)
		     : "memory");
	return value;
}
__device__ static __forceinline__ void ring_bitmap_store(uint32_t *address,
							 uint32_t value)
{
	asm volatile("st.volatile.global.u32 [%0], %1;"
		     :
		     : "l"(address), "r"(value)
		     : "memory");
}

__device__ static __forceinline__ bool
ring_send(struct bb_cut_through arguments, uint32_t hop, uint32_t pack,
	  uint32_t generation, uint4 value, uint64_t start)
{
	if (arguments.mode == BB_RING_LOSSLESS_GATHER) {
		const uint64_t slot = ((uint64_t)(generation % BB_RING_EPOCHS) *
					       BB_RING_GATHER_HOPS +
				       hop) * arguments.maximum_packs +
				      pack;
		uint4 *data = (uint4 *)arguments.next_packets;
		uint32_t *bitmaps =
			(uint32_t *)(data + (uint64_t)BB_RING_EPOCHS *
						    BB_RING_GATHER_HOPS *
						    arguments.maximum_packs);
		const uint32_t bitmap =
			BB_RING_BITMAP_READY | ring_zero_mask(value.x) |
			(ring_zero_mask(value.y) << RING_ZERO_PAIR_BITS) |
			(ring_zero_mask(value.z) << RING_THIRD_PAIR_SHIFT) |
			(ring_zero_mask(value.w) << RING_FOURTH_PAIR_SHIFT);
		ring_store_pack(data + slot,
				make_uint4(ring_nonzero_halves(value.x),
					   ring_nonzero_halves(value.y),
					   ring_nonzero_halves(value.z),
					   ring_nonzero_halves(value.w)));
		ring_bitmap_store(bitmaps + slot, bitmap);
		return true;
	}
	if (arguments.mode == BB_RING_BF16_SUM) {
		const uint64_t slot =
			((uint64_t)(generation % RING_EPOCHS) * BB_RING_HOPS +
			 hop) * arguments.maximum_packs +
			pack;
		const uint4 encoded = make_uint4(ring_nonzero_halves(value.x),
						 ring_nonzero_halves(value.y),
						 ring_nonzero_halves(value.z),
						 ring_nonzero_halves(value.w));
		ring_store_pack((uint4 *)arguments.next_packets + slot,
				encoded);
		return true;
	}
	const uint64_t slot = (uint64_t)hop * arguments.maximum_packs + pack;
	uint32_t *credit = (uint32_t *)arguments.local_credits + slot;
	volatile uint32_t *sent =
		(volatile uint32_t *)arguments.local_credits +
		(uint64_t)BB_RING_HOPS * arguments.maximum_packs + slot;
	const uint32_t previous = *sent;
	while (ring_load_credit(credit) != previous)
		if (ring_expired(arguments, start))
			return false;
	*sent = generation;
	uint64_t *target =
		(uint64_t *)arguments.next_packets + slot * BB_RING_PACK_WORDS;
	const uint64_t tag = (uint64_t)generation << RING_WORD_BITS;
	ring_store_pair(target, tag | value.x, tag | value.y);
	ring_store_pair(target + RING_THIRD_WORD, tag | value.z, tag | value.w);
	return true;
}

__device__ static __forceinline__ bool
ring_receive(struct bb_cut_through arguments, uint32_t hop, uint32_t pack,
	     uint32_t generation, uint4 *value, uint64_t start)
{
	if (arguments.mode == BB_RING_LOSSLESS_GATHER) {
		const uint64_t slot = ((uint64_t)(generation % BB_RING_EPOCHS) *
					       BB_RING_GATHER_HOPS +
				       hop) * arguments.maximum_packs +
				      pack;
		uint4 *data = (uint4 *)arguments.local_packets;
		uint32_t *bitmaps =
			(uint32_t *)(data + (uint64_t)BB_RING_EPOCHS *
						    BB_RING_GATHER_HOPS *
						    arguments.maximum_packs);
		uint4 received;
		uint32_t bitmap;
		for (;;) {
			received = ring_load_pack(data + slot);
			bitmap = ring_bitmap_load(bitmaps + slot);
			if (bitmap != 0 && !ring_has_sentinel(received.x) &&
			    !ring_has_sentinel(received.y) &&
			    !ring_has_sentinel(received.z) &&
			    !ring_has_sentinel(received.w))
				break;
			if (ring_expired(arguments, start))
				return false;
		}
		ring_store_pack(data + slot, make_uint4(0, 0, 0, 0));
		ring_bitmap_store(bitmaps + slot, 0);
		*value = make_uint4(
			ring_restore_zero(received.x, bitmap),
			ring_restore_zero(received.y,
					  bitmap >> RING_ZERO_PAIR_BITS),
			ring_restore_zero(received.z,
					  bitmap >> RING_THIRD_PAIR_SHIFT),
			ring_restore_zero(received.w,
					  bitmap >> RING_FOURTH_PAIR_SHIFT));
		return true;
	}
	if (arguments.mode == BB_RING_BF16_SUM) {
		const uint64_t slot =
			((uint64_t)(generation % RING_EPOCHS) * BB_RING_HOPS +
			 hop) * arguments.maximum_packs +
			pack;
		uint4 *source = (uint4 *)arguments.local_packets + slot;
		uint4 received;
		for (;;) {
			received = ring_load_pack(source);
			if (!ring_has_sentinel(received.x) &&
			    !ring_has_sentinel(received.y) &&
			    !ring_has_sentinel(received.z) &&
			    !ring_has_sentinel(received.w))
				break;
			if (ring_expired(arguments, start))
				return false;
		}
		ring_store_pack(source, make_uint4(0, 0, 0, 0));
		*value = received;
		return true;
	}
	const uint64_t slot = (uint64_t)hop * arguments.maximum_packs + pack;
	const uint64_t *source = (const uint64_t *)arguments.local_packets +
				 slot * BB_RING_PACK_WORDS;
	uint32_t words[BB_RING_PACK_WORDS];
	for (uint32_t word = 0; word < BB_RING_PACK_WORDS; word++) {
		uint64_t tagged;
		do {
			tagged = ring_load_word(source + word);
			if ((uint32_t)(tagged >> RING_WORD_BITS) !=
				    generation &&
			    ring_expired(arguments, start))
				return false;
		} while ((uint32_t)(tagged >> RING_WORD_BITS) != generation);
		words[word] = (uint32_t)tagged;
	}
	*value = make_uint4(words[0], words[1], words[RING_THIRD_WORD],
			    words[RING_FOURTH_WORD]);
	ring_credit((uint32_t *)arguments.previous_credits + slot, generation);
	return true;
}

__device__ static __forceinline__ uint32_t ring_add_pair(uint32_t left,
							 uint32_t right)
{
	uint32_t result;
	asm("fma.rn.bf16x2 %0, %1, %2, %3;"
	    : "=r"(result)
	    : "r"(left), "r"(RING_BF16_PAIR_ONES), "r"(right));
	return result;
}

__device__ static __forceinline__ uint4 ring_bf16_sum(uint4 left, uint4 right)
{
	return make_uint4(ring_add_pair(left.x, right.x),
			  ring_add_pair(left.y, right.y),
			  ring_add_pair(left.z, right.z),
			  ring_add_pair(left.w, right.w));
}

/* Integer-only transpose of complete output-channel packs, with disjoint
 * destinations. Keep packed output available for independent layout passes. */
/* The BF16 sum's place of an owner's pack in the input and output: owner-major (owner o the o-th quarter of the
 * packs), or with sum_row_packs column owners (owner o the o-th quarter of every row: [row][owner][row's packs]). */
__device__ static __forceinline__ uint64_t ring_sum_place(uint32_t maximum_packs, uint32_t sum_row_packs,
							 uint32_t owner, uint32_t pack)
{
	if (sum_row_packs == 0)
		return (uint64_t)owner * maximum_packs + pack;
	const uint32_t row = pack / sum_row_packs;
	return ((uint64_t)row * BB_RING_RANKS + owner) * sum_row_packs +
	       (pack - row * sum_row_packs);
}

/* Serving uses four equal output-axis owners. Keeping the two scalars explicit avoids a local record copy. */
__device__ static __forceinline__ void
ring_output(uint32_t packs, uint32_t vectors, uint4 *output, uint32_t owner,
	    uint32_t pack, uint4 value)
{
	const uint32_t row_packs = packs / vectors;
	const uint32_t row = pack / row_packs;
	const uint64_t destination = (uint64_t)row * row_packs * BB_RING_RANKS +
				     owner * row_packs + pack - row * row_packs;
	output[destination] = value;
}

/* One block's threads own disjoint packet indices at every hop. Only the initial
 * generation broadcast synchronizes a block. There is no inter-rank phase barrier.
 * All blocks must be resident; prepared grids are capped below physical SM count.
 */
__device__ static __forceinline__ void
cut_through_execute(const struct bb_cut_through arguments)
{
	__shared__ uint32_t generation;
	const uint64_t start = clock64();
	if (threadIdx.x == 0) {
		uint32_t *counter =
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
	const uint4 *input = (const uint4 *)arguments.input;
	uint4 *output = (uint4 *)arguments.output;
	const uint32_t rank = arguments.rank;
	if (arguments.mode != BB_RING_BF16_SUM) {
		for (uint32_t pack = first; pack < arguments.maximum_packs;
		     pack += stride) {
			const uint4 value =
				input[(rank * arguments.maximum_packs) + pack];
			ring_output(arguments.maximum_packs,
				    arguments.output_vectors, output, rank,
				    pack, value);
			if (!ring_send(arguments, 0, pack, generation, value,
				       start))
				return;
		}
		for (uint32_t hop = 0; hop < RING_LAST_RANK; hop++) {
			const uint32_t owner =
				(rank + BB_RING_RANKS - 1u - hop) %
				BB_RING_RANKS;
			for (uint32_t pack = first;
			     pack < arguments.maximum_packs; pack += stride) {
				uint4 value;
				if (!ring_receive(arguments, hop, pack,
						  generation, &value, start))
					return;
				ring_output(arguments.maximum_packs,
					    arguments.output_vectors, output,
					    owner, pack, value);
				if (hop + 1u < RING_LAST_RANK &&
				    !ring_send(arguments, hop + 1u, pack,
					       generation, value, start))
					return;
			}
		}
		return;
	}
	const uint32_t initial_owner = (rank + RING_LAST_RANK) % BB_RING_RANKS;
	const uint32_t row_packs = arguments.sum_row_packs;
	for (uint32_t pack = first; pack < arguments.maximum_packs;
	     pack += stride)
		if (!ring_send(arguments, 0, pack, generation,
			       input[ring_sum_place(arguments.maximum_packs,
						    row_packs, initial_owner,
						    pack)],
			       start))
			return;
	for (uint32_t hop = 0; hop < RING_LAST_RANK; hop++) {
		const uint32_t owner =
			(rank + BB_RING_RANKS + RING_LAST_RANK - 1u - hop) %
			BB_RING_RANKS;
		for (uint32_t pack = first; pack < arguments.maximum_packs;
		     pack += stride) {
			uint4 value;
			if (!ring_receive(arguments, hop, pack, generation,
					  &value, start))
				return;
			value = ring_bf16_sum(
				value,
				input[ring_sum_place(arguments.maximum_packs,
						     row_packs, owner, pack)]);
			if (hop + 1u == RING_LAST_RANK)
				output[ring_sum_place(arguments.maximum_packs,
						      row_packs, owner,
						      pack)] = value;
			if (!ring_send(arguments, hop + 1u, pack, generation,
				       value, start))
				return;
		}
	}
	for (uint32_t hop = RING_LAST_RANK; hop < BB_RING_HOPS; hop++) {
		const uint32_t owner =
			(rank + BB_RING_RANKS + RING_LAST_RANK - 1u - hop) %
			BB_RING_RANKS;
		for (uint32_t pack = first; pack < arguments.maximum_packs;
		     pack += stride) {
			uint4 value;
			if (!ring_receive(arguments, hop, pack, generation,
					  &value, start))
				return;
			output[ring_sum_place(arguments.maximum_packs,
					      row_packs, owner, pack)] = value;
			if (hop + 1u < BB_RING_HOPS &&
			    !ring_send(arguments, hop + 1u, pack, generation,
				       value, start))
				return;
		}
	}
}
