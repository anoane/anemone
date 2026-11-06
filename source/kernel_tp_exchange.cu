/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Shared TP site mailboxes: preserve model parity, assign a distinct increasing full64 phase to every layer. */
#include "anemone/tp_exchange_records.h"
#define PUSH_MAXIMUM_ELEMENTS TP_EXCHANGE_MAXIMUM_ELEMENTS
#include "kernel_ordered_push.cuh"
#include "kernel_linear_common.cuh"

#define TP_EXCHANGE_PHASE_STRIDE 2ull

/* Only padded prompt programs admit an empty half. Generic pushes retain their positive-payload contract. */
__device__ static __forceinline__ bool
exchange_allows_empty(const struct tp_exchange_arguments *arguments,
		      const unsigned char *arena)
{
	const struct step_block *const step =
		(const struct step_block *)(arena +
					    arguments->collective.step_offset);
	return (step->flags & STEP_BLOCK_PADDED) != 0;
}

/* The checks, the active rows' element count and the layer's full64 sequence of every exchange launch. A step block
 * of no rows (split_overlap's second half of a short step) exchanges nothing and still passes READY and DONE. */
__device__ static __forceinline__ uint64_t
exchange_adapt(struct tp_exchange_arguments *arguments, unsigned char *arena)
{
	const struct step_block *const step =
		(const struct step_block *)(arena +
					    arguments->collective.step_offset);
	if (step->sequence == 0 || arguments->reserved_zero != 0 ||
	    arguments->layer_count == 0 ||
	    arguments->layer_count > TP_EXCHANGE_PHASES ||
	    arguments->layer >= arguments->layer_count ||
	    arguments->collective.element_count % MODEL_HIDDEN != 0 ||
	    step->row_count >
		    arguments->collective.element_count / MODEL_HIDDEN ||
	    step->sequence >
		    (UINT64_MAX / TP_EXCHANGE_PHASE_STRIDE - arguments->layer) /
			    arguments->layer_count)
		asm volatile("trap;");
	arguments->collective.element_count = step->row_count * MODEL_HIDDEN;
	const uint64_t phase =
		step->sequence * arguments->layer_count + arguments->layer;
	return phase * TP_EXCHANGE_PHASE_STRIDE + (step->sequence & 1u);
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	tp_exchange(struct tp_exchange_arguments arguments,
		    unsigned char *arena)
{
	const uint64_t sequence = exchange_adapt(&arguments, arena);
	push_run_sequence(arguments.collective, arena, sequence,
			  exchange_allows_empty(&arguments, arena));
}

/* tp_exchange.h's wide form, four launches on the card's stream (the one-CTA exchange's stores, waits and adds,
 * spread): the payload to every rank's inbox by the whole grid, each writer fencing its stores at system scope. */
extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS)
	tp_exchange_publish(struct tp_exchange_arguments arguments,
			    unsigned char *arena)
{
	const uint64_t sequence = exchange_adapt(&arguments, arena);
	const struct ordered_push_arguments *const collective =
		&arguments.collective;
	if (collective->reserved_zero != 0 || collective->rank_count == 0 ||
	    collective->rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    collective->rank >= collective->rank_count ||
	    collective->payload_format > ORDERED_PUSH_BF16 ||
	    blockDim.x != ORDERED_COLLECTIVE_THREADS)
		asm volatile("trap;");
	push_publish_values(arguments.collective, arena, sequence,
			    blockIdx.x * blockDim.x + threadIdx.x,
			    gridDim.x * blockDim.x);
	__threadfence_system();
}

/* After every payload CTA retired (stream order): READY published to every rank and every rank's awaited. */
extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	tp_exchange_ready(struct tp_exchange_arguments arguments,
			  unsigned char *arena)
{
	const uint64_t sequence = exchange_adapt(&arguments, arena);
	const uint32_t generation =
		push_begin_generation(arguments.collective, arena, sequence,
				      exchange_allows_empty(&arguments, arena));
	push_publish_phase(arguments.collective, arena, sequence,
			   ORDERED_PUSH_READY, generation);
	push_wait_phase(arguments.collective, arena, sequence,
			ORDERED_PUSH_READY, generation);
}

/* The ordered sums (ascending rank, as the one-CTA exchange adds them) by the whole grid. */
extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS)
	tp_exchange_sum(struct tp_exchange_arguments arguments,
			unsigned char *arena)
{
	const uint64_t sequence = exchange_adapt(&arguments, arena);
	if (blockDim.x != ORDERED_COLLECTIVE_THREADS)
		asm volatile("trap;");
	push_sum_values(arguments.collective, arena, sequence,
			blockIdx.x * blockDim.x + threadIdx.x,
			gridDim.x * blockDim.x);
}

/* tp_exchange.h's owner sums (--tp_wide_exchange=2): the wide form's launches with these two in place of PUBLISH and
 * SUM. Shard d (rows d R / ranks .. of the step's R) goes only to rank d's inbox; rank d sums it over every inbox in
 * push_sum_values' order, rounds once as the FP32-to-BF16 join does and stores the BF16 bits into every rank's output
 * array, which the join then copies: the push's payload and output bits, half its bytes. Direct routes only. */
__device__ static __forceinline__ uint32_t owner_first_vector(
	const struct ordered_push_arguments *collective, uint32_t shard)
{
	const uint32_t rows = collective->element_count / MODEL_HIDDEN;
	return (uint32_t)((uint64_t)rows * shard / collective->rank_count) *
	       (MODEL_HIDDEN / PUSH_VECTOR_ELEMENTS);
}

__device__ static __forceinline__ void
owner_check(const struct ordered_push_arguments *collective)
{
	if (collective->reserved_zero != 0 || collective->rank_count == 0 ||
	    collective->rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    collective->rank >= collective->rank_count ||
	    collective->payload_format > ORDERED_PUSH_BF16 ||
	    collective->incoming_host_mask != 0 ||
	    blockDim.x != ORDERED_COLLECTIVE_THREADS)
		asm volatile("trap;");
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS)
	tp_exchange_owner_publish(struct tp_exchange_arguments arguments,
				  unsigned char *arena)
{
	const uint64_t sequence = exchange_adapt(&arguments, arena);
	const struct ordered_push_arguments *const collective =
		&arguments.collective;
	owner_check(collective);
	const float4 *const input =
		(const float4 *)(push_local_slot(*collective, arena, sequence) +
				 collective->input_offset);
	/* a group of CTAs a destination, rank r's group g writing to rank r + 1 + g: every rank sends to every other at
	 * once, never all to one (the CMPs' links are each card's own) */
	const uint32_t groups = gridDim.x / collective->rank_count;
	if (groups == 0 || gridDim.x % collective->rank_count != 0)
		asm volatile("trap;");
	const uint32_t destination =
		(collective->rank + 1u + blockIdx.x % collective->rank_count) %
		collective->rank_count;
	unsigned char *const inbox =
		push_destination_slot(*collective, arena, sequence,
				      destination) +
		collective->inbox_offset +
		(uint64_t)collective->rank * collective->inbox_rank_stride;
	const uint32_t end = owner_first_vector(collective, destination + 1u);
	for (uint32_t vector = owner_first_vector(collective, destination) +
			       blockIdx.x / collective->rank_count *
				       blockDim.x +
			       threadIdx.x;
	     vector < end; vector += groups * blockDim.x) {
		const float4 value = input[vector];
		if (collective->payload_format == ORDERED_PUSH_FP32) {
			((float4 *)inbox)[vector] = value;
		} else {
			uint2 pairs;
			asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
			    : "=r"(pairs.x)
			    : "f"(value.y), "f"(value.x));
			asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
			    : "=r"(pairs.y)
			    : "f"(value.w), "f"(value.z));
			((uint2 *)inbox)[vector] = pairs;
		}
	}
	__threadfence_system();
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS)
	tp_exchange_owner_sum(struct tp_exchange_arguments arguments,
			      unsigned char *arena)
{
	const uint64_t sequence = exchange_adapt(&arguments, arena);
	const struct ordered_push_arguments *const collective =
		&arguments.collective;
	owner_check(collective);
	const unsigned char *const slot =
		push_local_slot(*collective, arena, sequence) +
		collective->inbox_offset;
	const uint32_t end =
		owner_first_vector(collective, collective->rank + 1u);
	for (uint32_t vector = owner_first_vector(collective, collective->rank) +
			       blockIdx.x * blockDim.x + threadIdx.x;
	     vector < end; vector += gridDim.x * blockDim.x) {
		float values[ORDERED_COLLECTIVE_MAXIMUM_RANKS]
			    [PUSH_VECTOR_ELEMENTS];
#pragma unroll
		for (uint32_t source = 0;
		     source < ORDERED_COLLECTIVE_MAXIMUM_RANKS; source++) {
			if (source >= collective->rank_count)
				continue;
			const unsigned char *const inbox =
				slot + (uint64_t)source *
					       collective->inbox_rank_stride;
			uint32_t bits[PUSH_VECTOR_ELEMENTS];
			if (collective->payload_format == ORDERED_PUSH_FP32) {
				const uint4 words =
					__ldcg((const uint4 *)inbox + vector);
				bits[0] = words.x, bits[1] = words.y,
				bits[2] = words.z, bits[3] = words.w;
			} else {
				const uint2 pairs =
					__ldcg((const uint2 *)inbox + vector);
				bits[0] = pairs.x << PUSH_BF16_SHIFT;
				bits[1] = pairs.x & ~PUSH_BF16_MASK;
				bits[2] = pairs.y << PUSH_BF16_SHIFT;
				bits[3] = pairs.y & ~PUSH_BF16_MASK;
			}
#pragma unroll
			for (uint32_t element = 0;
			     element < PUSH_VECTOR_ELEMENTS; element++)
				values[source][element] =
					__uint_as_float(bits[element]);
		}
		uint32_t rounded[PUSH_VECTOR_ELEMENTS];
#pragma unroll
		for (uint32_t element = 0; element < PUSH_VECTOR_ELEMENTS;
		     element++) {
			float sum = 0.0f;
#pragma unroll
			for (uint32_t source = 0;
			     source < ORDERED_COLLECTIVE_MAXIMUM_RANKS;
			     source++) {
				if (source < collective->rank_count)
					sum = __fadd_rn(sum,
							values[source][element]);
			}
			rounded[element] = linear_bfloat16_bits(sum);
		}
		const uint2 pairs = make_uint2(
			rounded[0] | rounded[1] << PUSH_BF16_SHIFT,
			rounded[2] | rounded[3] << PUSH_BF16_SHIFT);
		for (uint32_t step = 1; step <= collective->rank_count; step++)
			((uint2 *)(push_destination_slot(
					   *collective, arena, sequence,
					   (collective->rank + step) %
						   collective->rank_count) +
				   collective->output_offset))[vector] = pairs;
	}
	__threadfence_system();
}

/* tp_exchange.h's owner ring sums (the FFN site of --tp_wide_exchange=3): the BF16 ring's sums of a piece a slab of
 * 8 rows at a time (kernel_cut_through.cuh) in one exchange. Rank o owns quarter o of every slab's bytes, as in the
 * ring, and adds its elements in the ring's order, from rank o + 1 on, each add one fma.rn.bf16x2 rounding to BF16,
 * then stores the sums into every rank's output array. The input array holds the piece's BF16 rows. */
#define OWNER_RING_RANKS TP_EXCHANGE_OWNER_RING_RANKS
#define OWNER_RING_VECTOR_ELEMENTS 8u
#define OWNER_RING_ROW_VECTORS (MODEL_HIDDEN / OWNER_RING_VECTOR_ELEMENTS)
#define OWNER_RING_SLAB_VECTORS (LINEAR_MAXIMUM_ROWS * OWNER_RING_ROW_VECTORS)
#define OWNER_RING_SHARE (OWNER_RING_SLAB_VECTORS / OWNER_RING_RANKS)
#define OWNER_RING_BF16_PAIR_ONES 0x3f803f80u

/* The index-th of owner's 16-byte vectors: its quarter of each slab's, slab after slab (the last slab its rows). */
__device__ static __forceinline__ uint32_t owner_ring_vector(uint32_t rows,
							     uint32_t owner,
							     uint32_t index)
{
	const uint32_t slab = index / OWNER_RING_SHARE;
	const uint32_t slab_rows = rows - slab * LINEAR_MAXIMUM_ROWS;
	const uint32_t own =
		slab_rows < LINEAR_MAXIMUM_ROWS ?
			slab_rows * OWNER_RING_ROW_VECTORS / OWNER_RING_RANKS :
			OWNER_RING_SHARE;
	return slab * OWNER_RING_SLAB_VECTORS + owner * own + index -
	       slab * OWNER_RING_SHARE;
}

/* ring_add_pair's add: the received partial plus the rank's input, rounded once to BF16 */
__device__ static __forceinline__ uint32_t owner_ring_add(uint32_t partial,
							  uint32_t input)
{
	uint32_t result;
	asm("fma.rn.bf16x2 %0, %1, %2, %3;"
	    : "=r"(result)
	    : "r"(partial), "r"(OWNER_RING_BF16_PAIR_ONES), "r"(input));
	return result;
}

__device__ static __forceinline__ uint32_t
owner_ring_check(struct tp_exchange_arguments *arguments, unsigned char *arena,
		 uint64_t *sequence)
{
	*sequence = exchange_adapt(arguments, arena);
	owner_check(&arguments->collective);
	if (arguments->collective.rank_count != OWNER_RING_RANKS ||
	    arguments->collective.payload_format != ORDERED_PUSH_BF16)
		asm volatile("trap;");
	return arguments->collective.element_count / MODEL_HIDDEN;
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS)
	tp_exchange_owner_ring_publish(struct tp_exchange_arguments arguments,
				       unsigned char *arena)
{
	uint64_t sequence;
	const uint32_t rows = owner_ring_check(&arguments, arena, &sequence);
	const struct ordered_push_arguments *const collective =
		&arguments.collective;
	const uint32_t groups = gridDim.x / OWNER_RING_RANKS;
	if (groups == 0 || gridDim.x % OWNER_RING_RANKS != 0)
		asm volatile("trap;");
	/* as the owner sums' payload: a group of CTAs a destination, from rank + 1 on */
	const uint32_t destination =
		(collective->rank + 1u + blockIdx.x % OWNER_RING_RANKS) %
		OWNER_RING_RANKS;
	const uint4 *const input =
		(const uint4 *)(push_local_slot(*collective, arena, sequence) +
				collective->input_offset);
	uint4 *const inbox =
		(uint4 *)(push_destination_slot(*collective, arena, sequence,
						destination) +
			  collective->inbox_offset +
			  (uint64_t)collective->rank *
				  collective->inbox_rank_stride);
	const uint32_t count = rows * OWNER_RING_ROW_VECTORS / OWNER_RING_RANKS;
	for (uint32_t index = blockIdx.x / OWNER_RING_RANKS * blockDim.x +
			      threadIdx.x;
	     index < count; index += groups * blockDim.x) {
		const uint32_t vector =
			owner_ring_vector(rows, destination, index);
		inbox[vector] = input[vector];
	}
	__threadfence_system();
}

extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS)
	tp_exchange_owner_ring_sum(struct tp_exchange_arguments arguments,
				   unsigned char *arena)
{
	uint64_t sequence;
	const uint32_t rows = owner_ring_check(&arguments, arena, &sequence);
	const struct ordered_push_arguments *const collective =
		&arguments.collective;
	const uint32_t rank = collective->rank;
	const unsigned char *const slot =
		push_local_slot(*collective, arena, sequence) +
		collective->inbox_offset;
	const uint32_t count = rows * OWNER_RING_ROW_VECTORS / OWNER_RING_RANKS;
	for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	     index < count; index += gridDim.x * blockDim.x) {
		const uint32_t vector = owner_ring_vector(rows, rank, index);
		uint4 parts[OWNER_RING_RANKS]; /* in the ring's order: rank + 1, rank + 2, ..., rank */
#pragma unroll
		for (uint32_t step = 0; step < OWNER_RING_RANKS; step++)
			parts[step] = __ldcg(
				(const uint4 *)(slot +
						(uint64_t)((rank + 1u + step) %
							   OWNER_RING_RANKS) *
							collective->inbox_rank_stride) +
				vector);
		uint4 sum = parts[0];
#pragma unroll
		for (uint32_t step = 1; step < OWNER_RING_RANKS; step++) {
			const uint4 input = parts[step];
			sum = make_uint4(owner_ring_add(sum.x, input.x),
					 owner_ring_add(sum.y, input.y),
					 owner_ring_add(sum.z, input.z),
					 owner_ring_add(sum.w, input.w));
		}
		for (uint32_t step = 1; step <= OWNER_RING_RANKS; step++)
			((uint4 *)(push_destination_slot(
					   *collective, arena, sequence,
					   (rank + step) % OWNER_RING_RANKS) +
				   collective->output_offset))[vector] = sum;
	}
	__threadfence_system();
}

/* tp_exchange.h's copy-engine owner ring sums: the ring's owner-major segments (tp2_join's RING_PACK) arrived by
 * copies in slot 0's inboxes; the rank's segment of the prepared rows added in the ring's order (as the owner ring
 * sums add) into its own output array of slot 0 (copies then carry it). */
extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS)
	tp_exchange_ce_owner_ring_sum(struct tp_exchange_arguments arguments,
				      unsigned char *arena)
{
	const uint32_t prepared_rows =
		arguments.collective.element_count / MODEL_HIDDEN;
	uint64_t sequence;
	(void)owner_ring_check(&arguments, arena, &sequence);
	const struct ordered_push_arguments *const collective =
		&arguments.collective;
	const uint32_t rank = collective->rank;
	const uint32_t segment = prepared_rows * OWNER_RING_ROW_VECTORS / OWNER_RING_RANKS;
	const unsigned char *const slot = arena + collective->region_offset + collective->inbox_offset;
	uint4 *const output = (uint4 *)(arena + collective->region_offset + collective->output_offset);
	for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < segment;
	     index += gridDim.x * blockDim.x) {
		const uint32_t vector = rank * segment + index;
		uint4 parts[OWNER_RING_RANKS]; /* in the ring's order: rank + 1, rank + 2, ..., rank */
#pragma unroll
		for (uint32_t step = 0; step < OWNER_RING_RANKS; step++)
			parts[step] = __ldcg((const uint4 *)(slot + (uint64_t)((rank + 1u + step) % OWNER_RING_RANKS) *
									    collective->inbox_rank_stride) +
					     vector);
		uint4 sum = parts[0];
#pragma unroll
		for (uint32_t step = 1; step < OWNER_RING_RANKS; step++)
			sum = make_uint4(owner_ring_add(sum.x, parts[step].x), owner_ring_add(sum.y, parts[step].y),
					 owner_ring_add(sum.z, parts[step].z), owner_ring_add(sum.w, parts[step].w));
		output[vector] = sum;
	}
}

/* tp_exchange.h's copy-engine owner sums: the quarters arrived by copies in slot 0's inboxes; the rank's quarter of the
 * prepared rows, its live rows, summed as tp_exchange_owner_sum sums them, into its own output array of slot
 * 0 (copies then carry it). */
extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS)
	tp_exchange_ce_owner_sum(struct tp_exchange_arguments arguments,
				 unsigned char *arena)
{
	const uint32_t prepared_rows =
		arguments.collective.element_count / MODEL_HIDDEN;
	(void)exchange_adapt(&arguments, arena);
	const struct ordered_push_arguments *const collective =
		&arguments.collective;
	owner_check(collective);
	if (collective->payload_format != ORDERED_PUSH_BF16 ||
	    prepared_rows % collective->rank_count != 0)
		asm volatile("trap;");
	const uint32_t quarter = prepared_rows / collective->rank_count;
	const uint32_t live_rows = collective->element_count / MODEL_HIDDEN;
	const uint32_t first_row = collective->rank * quarter;
	const uint32_t end_row = first_row + quarter < live_rows ? first_row + quarter : live_rows;
	const uint32_t row_vectors = MODEL_HIDDEN / PUSH_VECTOR_ELEMENTS;
	const unsigned char *const slot = arena + collective->region_offset + collective->inbox_offset;
	uint2 *const output = (uint2 *)(arena + collective->region_offset + collective->output_offset);
	for (uint32_t vector = first_row * row_vectors + blockIdx.x * blockDim.x + threadIdx.x;
	     vector < end_row * row_vectors; vector += gridDim.x * blockDim.x) {
		float values[ORDERED_COLLECTIVE_MAXIMUM_RANKS]
			    [PUSH_VECTOR_ELEMENTS];
#pragma unroll
		for (uint32_t source = 0;
		     source < ORDERED_COLLECTIVE_MAXIMUM_RANKS; source++) {
			if (source >= collective->rank_count)
				continue;
			const uint2 pairs = __ldcg((const uint2 *)(slot + (uint64_t)source *
										   collective->inbox_rank_stride) +
						   vector);
			values[source][0] = __uint_as_float(pairs.x << PUSH_BF16_SHIFT);
			values[source][1] = __uint_as_float(pairs.x & ~PUSH_BF16_MASK);
			values[source][2] = __uint_as_float(pairs.y << PUSH_BF16_SHIFT);
			values[source][3] = __uint_as_float(pairs.y & ~PUSH_BF16_MASK);
		}
		uint32_t rounded[PUSH_VECTOR_ELEMENTS];
#pragma unroll
		for (uint32_t element = 0; element < PUSH_VECTOR_ELEMENTS; element++) {
			float sum = 0.0f;
#pragma unroll
			for (uint32_t source = 0;
			     source < ORDERED_COLLECTIVE_MAXIMUM_RANKS;
			     source++) {
				if (source < collective->rank_count)
					sum = __fadd_rn(sum, values[source][element]);
			}
			rounded[element] = linear_bfloat16_bits(sum);
		}
		output[vector] = make_uint2(rounded[0] | rounded[1] << PUSH_BF16_SHIFT,
					    rounded[2] | rounded[3] << PUSH_BF16_SHIFT);
	}
}

/* After every sum CTA retired (no inbox read left on this rank): DONE published and every rank's awaited, under the
 * generation READY published (this rank's own READY line of the slot). */
extern "C" __global__ void __launch_bounds__(ORDERED_COLLECTIVE_THREADS, 1)
	tp_exchange_done(struct tp_exchange_arguments arguments,
			 unsigned char *arena)
{
	const uint64_t sequence = exchange_adapt(&arguments, arena);
	const struct ordered_push_arguments collective = arguments.collective;
	if (gridDim.x != 1u || blockDim.x != ORDERED_COLLECTIVE_THREADS)
		asm volatile("trap;");
	__shared__ uint32_t generation;
	if (threadIdx.x == 0) {
		const unsigned char *const line =
			push_line(collective,
				  push_local_slot(collective, arena, sequence),
				  ORDERED_PUSH_READY, collective.rank);
		generation = mailbox_acquire32(line);
		if (mailbox_sequence_load(line + PUSH_SEQUENCE_OFFSET) !=
		    sequence)
			asm volatile("trap;");
	}
	__syncthreads();
	push_publish_phase(collective, arena, sequence, ORDERED_PUSH_DONE,
			   generation);
	push_wait_phase(collective, arena, sequence, ORDERED_PUSH_DONE,
			generation);
}
