/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Producer epilogues may call push_publish_input and push_finish under the same single-CTA contract. */
#ifndef KERNEL_ORDERED_PUSH_CUH
#define KERNEL_ORDERED_PUSH_CUH

#include "anemone/ordered_push_records.h"
#include "kernel_mailbox_signal.cuh"

#define PUSH_WAIT_BACKOFF_CYCLES 64u
#define PUSH_SEQUENCE_OFFSET ORDERED_COLLECTIVE_SEQUENCE_OFFSET
#define PUSH_BF16_SHIFT 16u
#define PUSH_BF16_PAIR 2u
#define PUSH_BF16_MASK 0xFFFFu
#define PUSH_VECTOR_ELEMENTS 4u
#define PUSH_VECTOR_ALIGNMENT 16u
#ifndef PUSH_MAXIMUM_ELEMENTS
#define PUSH_MAXIMUM_ELEMENTS ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS
#endif

__device__ static __forceinline__ const struct ordered_collective_peer *
push_peers(struct ordered_push_arguments arguments, unsigned char *arena)
{
	return (const struct ordered_collective_peer
			*)(arena + arguments.peer_table_offset);
}

__device__ static __forceinline__ unsigned char *
push_local_slot(struct ordered_push_arguments arguments, unsigned char *arena,
		uint64_t sequence)
{
	return arena + arguments.region_offset +
	       (sequence & 1u) * arguments.parity_stride;
}

__device__ static __forceinline__ unsigned char *
push_destination_slot(struct ordered_push_arguments arguments,
		      unsigned char *arena, uint64_t sequence,
		      uint32_t destination)
{
	if (destination == arguments.rank)
		return push_local_slot(arguments, arena, sequence);
	const struct ordered_collective_peer peer =
		push_peers(arguments, arena)[destination];
	return (unsigned char *)(uintptr_t)peer.arena_address +
	       peer.region_offset + (sequence & 1u) * arguments.parity_stride;
}

__device__ static __forceinline__ unsigned char *
push_receive_slot(struct ordered_push_arguments arguments, unsigned char *arena,
		  uint64_t sequence, uint32_t source)
{
	if ((arguments.incoming_host_mask & (1u << source)) == 0)
		return push_local_slot(arguments, arena, sequence);
	return (unsigned char *)(uintptr_t)push_peers(arguments,
						      arena)[arguments.rank]
		       .host_address +
	       (sequence & 1u) * arguments.parity_stride;
}

__device__ static __forceinline__ unsigned char *
push_line(struct ordered_push_arguments arguments, unsigned char *slot,
	  uint32_t phase, uint32_t source)
{
	return slot + ((uint64_t)phase * arguments.rank_count + source) *
			      ORDERED_COLLECTIVE_LINE_BYTES;
}

__device__ static __forceinline__ void
push_publish_phase(struct ordered_push_arguments arguments,
		   unsigned char *arena, uint64_t sequence, uint32_t phase,
		   uint32_t generation)
{
	/* Every payload writer fences its own stores before the CTA's distributed commit writers publish. */
	__threadfence_system();
	__syncthreads();
	if (threadIdx.x < arguments.rank_count) {
		unsigned char *const line =
			push_line(arguments,
				  push_destination_slot(arguments, arena,
							sequence, threadIdx.x),
				  phase, arguments.rank);
		mailbox_sequence_store(line + PUSH_SEQUENCE_OFFSET, sequence);
		mailbox_release32(line, generation);
	}
	__syncthreads();
}

__device__ static __forceinline__ void
push_wait_phase(struct ordered_push_arguments arguments, unsigned char *arena,
		uint64_t sequence, uint32_t phase, uint32_t generation)
{
	if (threadIdx.x < arguments.rank_count) {
		const uint32_t source = threadIdx.x;
		const unsigned char *const line = push_line(
			arguments,
			push_receive_slot(arguments, arena, sequence, source),
			phase, source);
		const uint64_t started = clock64();
		while (mailbox_acquire32(line) != generation) {
			if (clock64() - started >= arguments.deadline_cycles)
				asm volatile("trap;");
			__nanosleep(PUSH_WAIT_BACKOFF_CYCLES);
		}
		if (mailbox_sequence_load(line + PUSH_SEQUENCE_OFFSET) !=
		    sequence)
			asm volatile("trap;");
	}
	__syncthreads();
}

__device__ static __forceinline__ void
push_publish_values(struct ordered_push_arguments arguments,
		    unsigned char *arena, uint64_t sequence,
		    uint32_t first_element, uint32_t element_stride)
{
	const float *const input =
		(const float *)(push_local_slot(arguments, arena, sequence) +
				arguments.input_offset);
	for (uint32_t destination = 0; destination < arguments.rank_count;
	     destination++) {
		unsigned char *const inbox =
			push_destination_slot(arguments, arena, sequence,
					      destination) +
			arguments.inbox_offset +
			(uint64_t)arguments.rank * arguments.inbox_rank_stride;
		if ((((uintptr_t)input | (uintptr_t)inbox) %
		     PUSH_VECTOR_ALIGNMENT) == 0) {
			const uint32_t vectors =
				arguments.element_count / PUSH_VECTOR_ELEMENTS;
			for (uint32_t vector = first_element; vector < vectors;
			     vector += element_stride) {
				const uint32_t element =
					vector * PUSH_VECTOR_ELEMENTS;
				float first, second, third, fourth;
				asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
				    : "=f"(first), "=f"(second), "=f"(third),
				      "=f"(fourth)
				    : "l"(input + element));
				if (arguments.payload_format ==
				    ORDERED_PUSH_FP32) {
					asm volatile(
						"st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
							"l"((float *)inbox +
							    element),
						"f"(first), "f"(second),
						"f"(third), "f"(fourth)
						: "memory");
				} else {
					uint32_t first_pair, second_pair;
					asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
					    : "=r"(first_pair)
					    : "f"(second), "f"(first));
					asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
					    : "=r"(second_pair)
					    : "f"(fourth), "f"(third));
					asm volatile(
						"st.global.v2.u32 [%0], {%1, %2};" ::
							"l"((uint32_t *)inbox +
							    vector *
								    PUSH_BF16_PAIR),
						"r"(first_pair),
						"r"(second_pair)
						: "memory");
				}
			}
			for (uint32_t element = vectors * PUSH_VECTOR_ELEMENTS +
						first_element;
			     element < arguments.element_count;
			     element += element_stride) {
				if (arguments.payload_format ==
				    ORDERED_PUSH_FP32) {
					((float *)inbox)[element] =
						input[element];
				} else {
					uint16_t rounded;
					asm("cvt.rn.bf16.f32 %0, %1;"
					    : "=h"(rounded)
					    : "f"(input[element]));
					((uint16_t *)inbox)[element] = rounded;
				}
			}
			continue;
		}
		if (arguments.payload_format == ORDERED_PUSH_FP32) {
			for (uint32_t element = first_element;
			     element < arguments.element_count;
			     element += element_stride)
				((float *)inbox)[element] = input[element];
		} else {
			const uint32_t pairs =
				arguments.element_count / PUSH_BF16_PAIR;
			for (uint32_t pair = first_element; pair < pairs;
			     pair += element_stride) {
				const uint32_t element = pair * PUSH_BF16_PAIR;
				uint32_t packed;
				/* PTX puts its first operand in the high half, second in the low half. */
				asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
				    : "=r"(packed)
				    : "f"(input[element + 1u]),
				      "f"(input[element]));
				((uint32_t *)inbox)[pair] = packed;
			}
			if (first_element == 0 &&
			    (arguments.element_count % PUSH_BF16_PAIR) != 0) {
				const uint32_t element =
					arguments.element_count - 1u;
				const float value = input[element];
				uint16_t rounded;
				asm("cvt.rn.bf16.f32 %0, %1;"
				    : "=h"(rounded)
				    : "f"(value));
				((uint16_t *)inbox)[element] = rounded;
			}
		}
	}
}

__device__ static __forceinline__ void
push_publish_input(struct ordered_push_arguments arguments,
		   unsigned char *arena, uint64_t sequence, uint32_t generation)
{
	push_publish_values(arguments, arena, sequence, threadIdx.x,
			    blockDim.x);
	push_publish_phase(arguments, arena, sequence, ORDERED_PUSH_READY,
			   generation);
}

/* The ordered sums of the elements first_element, first_element + element_stride, ... of every rank's inbox. */
__device__ static __forceinline__ void
push_sum_values(struct ordered_push_arguments arguments, unsigned char *arena,
		uint64_t sequence, uint32_t first_element,
		uint32_t element_stride)
{
	float *const output =
		(float *)(push_local_slot(arguments, arena, sequence) +
			  arguments.output_offset);
	for (uint32_t element = first_element;
	     element < arguments.element_count; element += element_stride) {
		float values[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
		/* Independent rank loads precede the unchanged ordered adds; avoid a load/add dependency chain. */
#pragma unroll
		for (uint32_t source = 0;
		     source < ORDERED_COLLECTIVE_MAXIMUM_RANKS; source++) {
			if (source >= arguments.rank_count)
				continue;
			const unsigned char *const inbox =
				push_receive_slot(arguments, arena, sequence,
						  source) +
				arguments.inbox_offset +
				(uint64_t)source * arguments.inbox_rank_stride;
			const bool host = (arguments.incoming_host_mask &
					   (1u << source)) != 0;
			uint32_t bits;
			if (arguments.payload_format == ORDERED_PUSH_FP32) {
				const uint32_t *const values =
					(const uint32_t *)inbox;
				if (host)
					asm volatile(
						"ld.global.cv.u32 %0, [%1];"
						: "=r"(bits)
						: "l"(values + element)
						: "memory");
				else
					bits = __ldcg(values + element);
			} else {
				uint32_t packed;
				const uint32_t *const values =
					(const uint32_t *)inbox;
				if (host)
					asm volatile(
						"ld.global.cv.u32 %0, [%1];"
						: "=r"(packed)
						: "l"(values +
						      element / PUSH_BF16_PAIR)
						: "memory");
				else
					packed = __ldcg(values +
							element /
								PUSH_BF16_PAIR);
				const uint32_t shift =
					(element % PUSH_BF16_PAIR) *
					PUSH_BF16_SHIFT;
				bits = ((packed >> shift) & PUSH_BF16_MASK)
				       << PUSH_BF16_SHIFT;
			}
			values[source] = __uint_as_float(bits);
		}
		float sum = 0.0f;
#pragma unroll
		for (uint32_t source = 0;
		     source < ORDERED_COLLECTIVE_MAXIMUM_RANKS; source++) {
			if (source < arguments.rank_count)
				sum = __fadd_rn(sum, values[source]);
		}
		output[element] = sum;
	}
}

__device__ static __forceinline__ void
push_finish(struct ordered_push_arguments arguments, unsigned char *arena,
	    uint64_t sequence, uint32_t generation)
{
	push_wait_phase(arguments, arena, sequence, ORDERED_PUSH_READY,
			generation);
	push_sum_values(arguments, arena, sequence, threadIdx.x, blockDim.x);
	push_publish_phase(arguments, arena, sequence, ORDERED_PUSH_DONE,
			   generation);
	push_wait_phase(arguments, arena, sequence, ORDERED_PUSH_DONE,
			generation);
}

__device__ static __forceinline__ uint32_t push_begin_generation(
	struct ordered_push_arguments arguments, unsigned char *arena,
	uint64_t sequence, bool allow_empty_payload = false)
{
	if (sequence == 0 || arguments.reserved_zero != 0 ||
	    arguments.deadline_cycles == 0 || arguments.rank_count == 0 ||
	    arguments.rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    arguments.rank >= arguments.rank_count ||
	    (arguments.element_count == 0 && !allow_empty_payload) ||
	    arguments.element_count > PUSH_MAXIMUM_ELEMENTS ||
	    arguments.payload_format > ORDERED_PUSH_BF16 ||
	    blockDim.x != ORDERED_COLLECTIVE_THREADS || gridDim.x != 1u)
		asm volatile("trap;");
	__shared__ uint32_t generation;
	if (threadIdx.x == 0) {
		for (uint32_t slot = 0; slot < ORDERED_COLLECTIVE_SLOTS;
		     slot++) {
			const unsigned char *const line = push_line(
				arguments,
				arena + arguments.region_offset +
					(uint64_t)slot *
						arguments.parity_stride,
				ORDERED_PUSH_READY, arguments.rank);
			if (mailbox_sequence_load(
				    line + PUSH_SEQUENCE_OFFSET) >= sequence)
				asm volatile("trap;");
		}
		for (uint32_t destination = 0;
		     destination < arguments.rank_count; destination++) {
			const struct ordered_collective_peer peer =
				push_peers(arguments, arena)[destination];
			if (peer.reserved_zero != 0 ||
			    peer.route > ORDERED_COLLECTIVE_HOST_ROUTE ||
			    (destination != arguments.rank &&
			     peer.arena_address == 0))
				asm volatile("trap;");
		}
		if (arguments.incoming_host_mask != 0 &&
		    push_peers(arguments, arena)[arguments.rank].host_address ==
			    0)
			asm volatile("trap;");
		generation =
			mailbox_acquire32(push_line(
				arguments,
				push_local_slot(arguments, arena, sequence),
				ORDERED_PUSH_READY, arguments.rank)) +
			1u;
	}
	__syncthreads();
	return generation;
}

__device__ static __forceinline__ void
push_run_sequence(struct ordered_push_arguments arguments, unsigned char *arena,
		  uint64_t sequence, bool allow_empty_payload = false)
{
	const uint32_t generation = push_begin_generation(
		arguments, arena, sequence, allow_empty_payload);
	push_publish_input(arguments, arena, sequence, generation);
	push_finish(arguments, arena, sequence, generation);
}
#endif
