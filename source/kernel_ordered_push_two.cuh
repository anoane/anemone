/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Owner-slice scatter, fixed-order local sum and FP32 result push, using the same directed route graph. */
#ifndef KERNEL_ORDERED_PUSH_TWO_CUH
#define KERNEL_ORDERED_PUSH_TWO_CUH

#include "anemone/ordered_push_two_records.h"
#include "kernel_ordered_push.cuh"

__device__ static __forceinline__ uint32_t
push_two_slice_begin(uint32_t owner, uint32_t slice, uint32_t elements)
{
	const uint32_t begin = owner * slice;
	return begin < elements ? begin : elements;
}

__device__ static __forceinline__ void
push_two_scatter(struct ordered_push_arguments arguments, unsigned char *arena,
		 uint64_t sequence, uint32_t slice)
{
	const float *const input =
		(const float *)(push_local_slot(arguments, arena, sequence) +
				arguments.input_offset);
	for (uint32_t destination = 0; destination < arguments.rank_count;
	     destination++) {
		const uint32_t begin = push_two_slice_begin(
			destination, slice, arguments.element_count);
		const uint32_t end = push_two_slice_begin(
			destination + 1u, slice, arguments.element_count);
		unsigned char *const inbox =
			push_destination_slot(arguments, arena, sequence,
					      destination) +
			arguments.inbox_offset +
			(uint64_t)arguments.rank * arguments.inbox_rank_stride;
		const bool vectorized = (((uintptr_t)input | (uintptr_t)inbox) %
					 PUSH_VECTOR_ALIGNMENT) == 0;
		const uint32_t vector_begin =
			vectorized ? ((begin + PUSH_VECTOR_ELEMENTS - 1u) /
				      PUSH_VECTOR_ELEMENTS) *
					     PUSH_VECTOR_ELEMENTS :
				     end;
		const uint32_t vector_end =
			vectorized ? (end / PUSH_VECTOR_ELEMENTS) *
					     PUSH_VECTOR_ELEMENTS :
				     end;
		for (uint32_t element = begin + threadIdx.x;
		     element < end && element < vector_begin;
		     element += blockDim.x) {
			if (arguments.payload_format == ORDERED_PUSH_FP32) {
				((float *)inbox)[element] = input[element];
			} else {
				uint16_t rounded;
				asm("cvt.rn.bf16.f32 %0, %1;"
				    : "=h"(rounded)
				    : "f"(input[element]));
				((uint16_t *)inbox)[element] = rounded;
			}
		}
		for (uint32_t element =
			     vector_begin + threadIdx.x * PUSH_VECTOR_ELEMENTS;
		     element < vector_end;
		     element += blockDim.x * PUSH_VECTOR_ELEMENTS) {
			float first, second, third, fourth;
			asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
			    : "=f"(first), "=f"(second), "=f"(third),
			      "=f"(fourth)
			    : "l"(input + element));
			if (arguments.payload_format == ORDERED_PUSH_FP32) {
				asm volatile(
					"st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
						"l"((float *)inbox + element),
					"f"(first), "f"(second), "f"(third),
					"f"(fourth)
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
					"st.global.v2.u32 [%0], {%1, %2};" ::"l"(
						(uint16_t *)inbox + element),
					"r"(first_pair), "r"(second_pair)
					: "memory");
			}
		}
		/* If the slice is smaller than its alignment prefix, that prefix already copied every element. */
		const uint32_t tail = vector_end > vector_begin ? vector_end :
								  vector_begin;
		for (uint32_t element = tail + threadIdx.x; element < end;
		     element += blockDim.x) {
			if (arguments.payload_format == ORDERED_PUSH_FP32) {
				((float *)inbox)[element] = input[element];
			} else {
				uint16_t rounded;
				asm("cvt.rn.bf16.f32 %0, %1;"
				    : "=h"(rounded)
				    : "f"(input[element]));
				((uint16_t *)inbox)[element] = rounded;
			}
		}
	}
}

__device__ static __forceinline__ float
push_two_value(struct ordered_push_arguments arguments, unsigned char *arena,
	       uint64_t sequence, uint32_t source, uint32_t element)
{
	const unsigned char *const inbox =
		push_receive_slot(arguments, arena, sequence, source) +
		arguments.inbox_offset +
		(uint64_t)source * arguments.inbox_rank_stride;
	const bool host = (arguments.incoming_host_mask & (1u << source)) != 0;
	const uint32_t index = arguments.payload_format == ORDERED_PUSH_FP32 ?
				       element :
				       element / PUSH_BF16_PAIR;
	const uint32_t *const address = (const uint32_t *)inbox + index;
	uint32_t bits;
	if (host)
		asm volatile("ld.global.cv.u32 %0, [%1];"
			     : "=r"(bits)
			     : "l"(address)
			     : "memory");
	else
		bits = __ldcg(address);
	if (arguments.payload_format == ORDERED_PUSH_BF16) {
		const uint32_t shift =
			(element % PUSH_BF16_PAIR) * PUSH_BF16_SHIFT;
		bits = ((bits >> shift) & PUSH_BF16_MASK) << PUSH_BF16_SHIFT;
	}
	return __uint_as_float(bits);
}

__device__ static __forceinline__ void
push_two_copy_result_value(const float *source, float *destination,
			   uint32_t element, bool host_source)
{
	if (host_source) {
		uint32_t bits;
		asm volatile("ld.global.cv.u32 %0, [%1];"
			     : "=r"(bits)
			     : "l"(source + element)
			     : "memory");
		destination[element] = __uint_as_float(bits);
	} else {
		destination[element] = source[element];
	}
}

__device__ static __forceinline__ void
push_two_copy_result_slice(const float *source, float *destination,
			   uint32_t begin, uint32_t end, bool host_source)
{
	const bool vectorized = (((uintptr_t)source | (uintptr_t)destination) %
				 PUSH_VECTOR_ALIGNMENT) == 0;
	const uint32_t vector_begin =
		vectorized ? ((begin + PUSH_VECTOR_ELEMENTS - 1u) /
			      PUSH_VECTOR_ELEMENTS) *
				     PUSH_VECTOR_ELEMENTS :
			     end;
	const uint32_t vector_end = vectorized ? (end / PUSH_VECTOR_ELEMENTS) *
							 PUSH_VECTOR_ELEMENTS :
						 end;
	for (uint32_t element = begin + threadIdx.x;
	     element < end && element < vector_begin; element += blockDim.x)
		push_two_copy_result_value(source, destination, element,
					   host_source);
	for (uint32_t element =
		     vector_begin + threadIdx.x * PUSH_VECTOR_ELEMENTS;
	     element < vector_end;
	     element += blockDim.x * PUSH_VECTOR_ELEMENTS) {
		float first, second, third, fourth;
		if (host_source) {
			asm volatile(
				"ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];"
				: "=f"(first), "=f"(second), "=f"(third),
				  "=f"(fourth)
				: "l"(source + element)
				: "memory");
		} else {
			asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
				     : "=f"(first), "=f"(second), "=f"(third),
				       "=f"(fourth)
				     : "l"(source + element)
				     : "memory");
		}
		asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
				     destination + element),
			     "f"(first), "f"(second), "f"(third), "f"(fourth)
			     : "memory");
	}
	const uint32_t tail = vector_end > vector_begin ? vector_end :
							  vector_begin;
	for (uint32_t element = tail + threadIdx.x; element < end;
	     element += blockDim.x)
		push_two_copy_result_value(source, destination, element,
					   host_source);
}

__device__ static __forceinline__ void
push_two_reduce_return(struct ordered_push_arguments arguments,
		       unsigned char *arena, uint64_t sequence, uint32_t begin,
		       uint32_t end)
{
	float *const output =
		(float *)(push_local_slot(arguments, arena, sequence) +
			  arguments.output_offset);
	for (uint32_t element = begin + threadIdx.x; element < end;
	     element += blockDim.x) {
		float values[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
#pragma unroll
		for (uint32_t source = 0;
		     source < ORDERED_COLLECTIVE_MAXIMUM_RANKS; source++) {
			if (source < arguments.rank_count)
				values[source] = push_two_value(arguments,
								arena, sequence,
								source,
								element);
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
	__syncthreads();
	for (uint32_t destination = 0; destination < arguments.rank_count;
	     destination++) {
		if (destination == arguments.rank)
			continue;
		float *const remote_output =
			(float *)(push_destination_slot(arguments, arena,
							sequence, destination) +
				  arguments.output_offset);
		push_two_copy_result_slice(output, remote_output, begin, end,
					   false);
	}
}

__device__ static __forceinline__ void
push_two_copy_host_results(struct ordered_push_arguments arguments,
			   unsigned char *arena, uint64_t sequence,
			   uint32_t slice)
{
	float *const output =
		(float *)(push_local_slot(arguments, arena, sequence) +
			  arguments.output_offset);
	for (uint32_t source = 0; source < arguments.rank_count; source++) {
		if ((arguments.incoming_host_mask & (1u << source)) == 0)
			continue;
		const uint32_t begin = push_two_slice_begin(
			source, slice, arguments.element_count);
		const uint32_t end = push_two_slice_begin(
			source + 1u, slice, arguments.element_count);
		const float *const mirror_output =
			(const float *)(push_receive_slot(arguments, arena,
							  sequence, source) +
					arguments.output_offset);
		push_two_copy_result_slice(mirror_output, output, begin, end,
					   true);
	}
}

__device__ static __forceinline__ void
push_two_run_sequence_with_delay(struct ordered_push_arguments arguments,
				 unsigned char *arena, uint64_t sequence,
				 bool delay_host_copy)
{
	if (sequence == 0 || arguments.reserved_zero != 0 ||
	    arguments.deadline_cycles == 0 || arguments.rank_count == 0 ||
	    arguments.rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    arguments.rank >= arguments.rank_count ||
	    arguments.element_count == 0 ||
	    arguments.element_count > ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS ||
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
				ORDERED_PUSH_TWO_READY, arguments.rank);
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
		if (arguments.incoming_host_mask != 0) {
			/* CUDA 13.4 otherwise hoists this self address ahead of its peer-table base definition. */
			uint64_t self_entry_address;
			const uint64_t self_offset =
				arguments.peer_table_offset +
				(uint64_t)arguments.rank *
					ORDERED_COLLECTIVE_PEER_BYTES;
			asm volatile("add.u64 %0, %1, %2;"
				     : "=l"(self_entry_address)
				     : "l"((uint64_t)(uintptr_t)arena),
				       "l"(self_offset));
			const struct ordered_collective_peer *const self =
				(const struct ordered_collective_peer
					 *)(uintptr_t)self_entry_address;
			if (self->host_address == 0)
				asm volatile("trap;");
		}
		generation =
			mailbox_acquire32(push_line(
				arguments,
				push_local_slot(arguments, arena, sequence),
				ORDERED_PUSH_TWO_READY, arguments.rank)) +
			1u;
	}
	__syncthreads();
	const uint32_t slice =
		(arguments.element_count + arguments.rank_count - 1u) /
		arguments.rank_count;
	const uint32_t begin = push_two_slice_begin(arguments.rank, slice,
						    arguments.element_count);
	const uint32_t end = push_two_slice_begin(arguments.rank + 1u, slice,
						  arguments.element_count);
	push_two_scatter(arguments, arena, sequence, slice);
	push_publish_phase(arguments, arena, sequence, ORDERED_PUSH_TWO_READY,
			   generation);
	push_wait_phase(arguments, arena, sequence, ORDERED_PUSH_TWO_READY,
			generation);
	push_two_reduce_return(arguments, arena, sequence, begin, end);
	push_publish_phase(arguments, arena, sequence, ORDERED_PUSH_TWO_RESULT,
			   generation);
	push_wait_phase(arguments, arena, sequence, ORDERED_PUSH_TWO_RESULT,
			generation);
#if defined(PUSH_TWO_TEST_HOST_COPY_DELAY)
	if (delay_host_copy)
		push_two_test_host_copy_delay(arguments, arena, sequence);
#else
	(void)delay_host_copy;
#endif
	push_two_copy_host_results(arguments, arena, sequence, slice);
	push_publish_phase(arguments, arena, sequence, ORDERED_PUSH_TWO_CREDIT,
			   generation);
	push_wait_phase(arguments, arena, sequence, ORDERED_PUSH_TWO_CREDIT,
			generation);
}

__device__ static __forceinline__ void
push_two_run_sequence(struct ordered_push_arguments arguments,
		      unsigned char *arena, uint64_t sequence)
{
	push_two_run_sequence_with_delay(arguments, arena, sequence, false);
}
#endif
