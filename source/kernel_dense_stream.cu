/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Candidate SM80 exact dense product for the head: y = W x with bfloat16
 * weights and up to 8 rows, word for word kernel_dense_linear.cu's 8-row
 * bfloat16 instance (dense_linear_records.h's one order of additions):
 * thread t of 256 takes chunks t, t + 256, ... of 8 inputs; its chain of an
 * output and a row runs from +0 in ascending input order with fmaf; a warp adds
 * its lanes' chains by exchanges at distance 16 down to 1; then the warps' sums
 * are added from +0 in ascending order. What changes is the schedule:
 *
 * - Persistent blocks loop over groups of 4 outputs; the next group's weights
 *   are in flight while the current group is reduced and stored.
 * - The lane exchange is a halving one: at distance d a lane keeps half of its
 *   values and adds its partner's other half, so 32 chains end one per lane in
 *   31 exchanges instead of 160. An exchange butterfly combines the same lane
 *   pairs at every level for every lane, and float addition commutes, so each
 *   lane's sum has the bits lane 0's sum has in the butterfly.
 * - Rows from row_count on are skipped (the reference computes them from row
 *   0 and stores nothing); row squares, the same for every group, are added and
 *   reduced once per block.
 * Launch: grid = co-resident blocks (2 per multiprocessor), 256 threads, no
 * dynamic shared memory; same argument record. */
#include "kernel_linear_common.cuh"

#include "anemone/dense_linear_records.h"
#include "anemone/step_records.h"

#define STREAM_THREADS DENSE_THREADS_VALUE
#define STREAM_WARPS (STREAM_THREADS / THREADS_PER_WARP)
#define STREAM_OUTPUTS DENSE_OUTPUTS_PER_BLOCK
#define STREAM_ROWS LINEAR_MAXIMUM_ROWS
#define STREAM_VALUES (STREAM_OUTPUTS * STREAM_ROWS)
#define STREAM_CHUNK_INPUTS DENSE_CHUNK_INPUTS
#define STREAM_BFLOAT16_PER_WORD 2u
#define STREAM_FULL_WARP 0xFFFFFFFFu
#define STREAM_FIRST_DISTANCE 16u
#define STREAM_BUFFERS 2u
#define STREAM_BLOCKS_PER_MULTIPROCESSOR 2

static_assert(STREAM_VALUES == THREADS_PER_WARP,
	      "one output-row chain per lane after the halving exchange");

/* Input `index` of a 16-byte part of bfloat16 values, widened exactly. */
__device__ static __forceinline__ float stream_value(uint4 part, uint32_t index)
{
	const uint32_t word = index < 2u ? part.x :
			      index < 4u ? part.y :
			      index < 6u ? part.z :
					   part.w;
	return __uint_as_float(index % STREAM_BFLOAT16_PER_WORD == 0 ?
				       word << BITS_PER_HALF :
				       word & UPPER_HALF_MASK);
}

/* Halving exchange of 2 n values at distance n: a lane with bit n set keeps
 * the upper half. Afterwards value[i] (i < n) is the pair sum of value
 * index i + n * (lane bit set). */
template <uint32_t n>
__device__ static __forceinline__ void stream_halve(float *value, uint32_t lane)
{
	const bool upper = (lane & n) != 0;
#pragma unroll
	for (uint32_t index = 0; index < n; index++) {
		const float keep = upper ? value[index + n] : value[index];
		const float give = upper ? value[index] : value[index + n];
		value[index] =
			keep + __shfl_xor_sync(STREAM_FULL_WARP, give, n);
	}
}

/* Rows whose outputs are stored: below row_count and, for the head, flagged
 * in the step area. */
__device__ static uint32_t
stream_stored_rows(const struct dense_linear_arguments *arguments,
		   const unsigned char *arena)
{
	uint32_t mask = 0;
	const bool step_rows = (arguments->flags & DENSE_STEP_ROWS) != 0;
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments->step_offset);
	for (uint32_t row = 0; row < arguments->row_count; row++)
		if (!step_rows ||
		    (row < step->row_count &&
		     (((const struct step_row_record *)(step + 1))[row].flags &
		      STEP_ROW_LOGITS) != 0))
			mask |= 1u << row;
	return mask;
}

__device__ static __forceinline__ void stream_copy(uint4 *staged,
						   const uint4 *source)
{
	const uint32_t address = (uint32_t)__cvta_generic_to_shared(staged);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(address),
		     "l"(source)
		     : "memory");
}

__device__ static __forceinline__ void stream_commit(void)
{
	asm volatile("cp.async.commit_group;" ::: "memory");
}

template <uint32_t pending>
__device__ static __forceinline__ void stream_wait(void)
{
	asm volatile("cp.async.wait_group %0;" ::"n"(pending) : "memory");
}

/* ring_depth 0: a group's weights travel in registers, one group ahead.
 * ring_depth D: they travel through a per-thread cp.async ring in dynamic
 * shared memory, D groups ahead ([D][4 outputs][chunks] 16-byte entries; a
 * thread copies and later reads only its own entries, so no barrier). */
template <uint32_t chunks_most, uint32_t ring_depth>
__device__ static void dense_stream(struct dense_linear_arguments arguments,
				    unsigned char *arena)
{
	extern __shared__ uint4 stream_ring[];
	__shared__ float warp_sums[STREAM_BUFFERS][STREAM_WARPS][STREAM_VALUES];
	__shared__ float warp_squares[STREAM_WARPS][STREAM_ROWS];
	__shared__ float row_scale[STREAM_ROWS];
	const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP;
	const uint32_t chunk_count = arguments.input_size / STREAM_CHUNK_INPUTS;
	const uint32_t group_count = arguments.output_size / STREAM_OUTPUTS;
	const uint32_t row_count = arguments.row_count;
	const bool is_row_scaled = (arguments.flags & DENSE_ROW_SCALE) != 0;
	const uint4 *const weights =
		(const uint4 *)(arena + arguments.weights_offset);
	const uint4 *const rows =
		(const uint4 *)(arena + arguments.rows_offset);
	LINEAR_DEVICE_CHECK(blockDim.x == STREAM_THREADS && row_count != 0 &&
			    row_count <= STREAM_ROWS &&
			    chunk_count <= chunks_most * STREAM_THREADS);
	const uint32_t stored = stream_stored_rows(&arguments, arena);
	if (stored == 0)
		return; /* the head of a step without logits reads nothing */
	if (is_row_scaled) {
		float squares[STREAM_ROWS] = {};
#pragma unroll
		for (uint32_t step = 0; step < chunks_most; step++) {
			const uint32_t chunk =
				threadIdx.x + step * STREAM_THREADS;
#pragma unroll
			for (uint32_t row = 0; row < STREAM_ROWS; row++) {
				if (chunk >= chunk_count || row >= row_count)
					continue;
				const uint4 part =
					rows[(size_t)row * chunk_count + chunk];
#pragma unroll
				for (uint32_t index = 0;
				     index < STREAM_CHUNK_INPUTS; index++) {
					const float value =
						stream_value(part, index);
					squares[row] = fmaf(value, value,
							    squares[row]);
				}
			}
		}
#pragma unroll
		for (uint32_t distance = STREAM_FIRST_DISTANCE; distance != 0;
		     distance >>= 1u)
#pragma unroll
			for (uint32_t row = 0; row < STREAM_ROWS; row++)
				squares[row] = squares[row] +
					       __shfl_xor_sync(STREAM_FULL_WARP,
							       squares[row],
							       distance);
		if (lane == 0)
#pragma unroll
			for (uint32_t row = 0; row < STREAM_ROWS; row++)
				warp_squares[warp][row] = squares[row];
		__syncthreads();
		if (threadIdx.x < STREAM_ROWS) {
			float square_sum = 0.0f;
			for (uint32_t index = 0; index < STREAM_WARPS; index++)
				square_sum = square_sum +
					     warp_squares[index][threadIdx.x];
			row_scale[threadIdx.x] = rsqrtf(
				square_sum *
					(1.0f / (float)arguments.input_size) +
				arguments.epsilon);
		}
		__syncthreads();
	}
	uint4 part_weights[chunks_most][STREAM_OUTPUTS];
	auto ring_entry = [&](uint32_t slot, uint32_t output, uint32_t chunk) {
		return stream_ring +
		       ((size_t)slot * STREAM_OUTPUTS + output) * chunk_count +
		       chunk;
	};
	auto issue = [&](uint32_t group, uint32_t slot) {
#pragma unroll
		for (uint32_t step = 0; step < chunks_most; step++) {
			const uint32_t chunk =
				threadIdx.x + step * STREAM_THREADS;
#pragma unroll
			for (uint32_t output = 0; output < STREAM_OUTPUTS;
			     output++)
				if (group < group_count && chunk < chunk_count)
					stream_copy(
						ring_entry(slot, output, chunk),
						weights +
							((size_t)(group * STREAM_OUTPUTS +
								  output) *
								 chunk_count +
							 chunk));
		}
		stream_commit();
	};
	auto fetch = [&](uint32_t group) {
#pragma unroll
		for (uint32_t step = 0; step < chunks_most; step++) {
			const uint32_t chunk =
				threadIdx.x + step * STREAM_THREADS;
#pragma unroll
			for (uint32_t output = 0; output < STREAM_OUTPUTS;
			     output++)
				if (chunk < chunk_count)
					part_weights[step][output] = __ldcs(
						weights +
						((size_t)(group * STREAM_OUTPUTS +
							  output) *
							 chunk_count +
						 chunk));
		}
	};
	uint32_t buffer = 0, slot = 0;
	uint32_t group = blockIdx.x;
	if (ring_depth == 0) {
		if (group < group_count)
			fetch(group);
	} else {
		for (uint32_t ahead = 0; ahead < ring_depth; ahead++)
			issue(group + ahead * gridDim.x, ahead);
	}
	for (; group < group_count; group += gridDim.x) {
		if (ring_depth != 0)
			stream_wait<ring_depth == 0 ? 0 : ring_depth - 1>();
		float sums[STREAM_VALUES];
#pragma unroll
		for (uint32_t value = 0; value < STREAM_VALUES; value++)
			sums[value] = 0.0f;
#pragma unroll
		for (uint32_t step = 0; step < chunks_most; step++) {
			const uint32_t chunk =
				threadIdx.x + step * STREAM_THREADS;
#pragma unroll
			for (uint32_t row = 0; row < STREAM_ROWS; row++) {
				if (chunk >= chunk_count || row >= row_count)
					continue;
				const uint4 part =
					rows[(size_t)row * chunk_count + chunk];
#pragma unroll
				for (uint32_t index = 0;
				     index < STREAM_CHUNK_INPUTS; index++) {
					const float value =
						stream_value(part, index);
#pragma unroll
					for (uint32_t output = 0;
					     output < STREAM_OUTPUTS; output++)
						sums[output * STREAM_ROWS + row] = fmaf(
							stream_value(
								ring_depth == 0 ?
									part_weights
										[step]
										[output] :
									*ring_entry(
										slot,
										output,
										chunk),
								index),
							value,
							sums[output * STREAM_ROWS +
							     row]);
				}
			}
		}
		/* The next group's weights travel during the exchange. */
		if (ring_depth == 0) {
			if (group + gridDim.x < group_count)
				fetch(group + gridDim.x);
		} else {
			issue(group + ring_depth * gridDim.x, slot);
			slot = slot + 1u == ring_depth ? 0u : slot + 1u;
		}
		stream_halve<16u>(sums, lane);
		stream_halve<8u>(sums, lane);
		stream_halve<4u>(sums, lane);
		stream_halve<2u>(sums, lane);
		stream_halve<1u>(sums, lane);
		/* Lane l now holds output l / 8, row l % 8. */
		warp_sums[buffer][warp][lane] = sums[0];
		__syncthreads();
		if (threadIdx.x < STREAM_VALUES) {
			const uint32_t output = threadIdx.x / STREAM_ROWS;
			const uint32_t row = threadIdx.x % STREAM_ROWS;
			if ((stored >> row) & 1u) {
				float sum = 0.0f;
				for (uint32_t index = 0; index < STREAM_WARPS;
				     index++)
					sum = sum + warp_sums[buffer][index]
							     [threadIdx.x];
				sum = is_row_scaled ? sum * row_scale[row] :
						      sum;
				const size_t element =
					(size_t)row * arguments.output_size +
					(size_t)group * STREAM_OUTPUTS + output;
				if (arguments.output_type == DENSE_BFLOAT16)
					((uint16_t
						  *)(arena +
						     arguments
							     .output_offset))[element] =
						(uint16_t)linear_bfloat16_bits(
							linear_bfloat16_value(
								linear_bfloat16_bits(
									sum)) *
							arguments.output_scale);
				else
					((float *)(arena +
						   arguments.output_offset))
						[element] =
							sum *
							arguments.output_scale;
			}
		}
		buffer ^= 1u;
	}
}

extern "C" __global__ void __launch_bounds__(STREAM_THREADS,
					     STREAM_BLOCKS_PER_MULTIPROCESSOR)
	dense_stream_bfloat16_8(struct dense_linear_arguments arguments,
				unsigned char *arena)
{
	dense_stream<3u, 0u>(arguments, arena);
}

/* Ring forms: dynamic shared memory = depth x 4 x (input_size / 8) x 16 bytes
 * (80 KB for depth 2 at 5,120 inputs: two blocks a multiprocessor). */
extern "C" __global__ void __launch_bounds__(STREAM_THREADS, 2)
	dense_stream_ring2_bfloat16_8(struct dense_linear_arguments arguments,
				      unsigned char *arena)
{
	dense_stream<3u, 2u>(arguments, arena);
}
extern "C" __global__ void __launch_bounds__(STREAM_THREADS, 1)
	dense_stream_ring3_bfloat16_8(struct dense_linear_arguments arguments,
				      unsigned char *arena)
{
	dense_stream<3u, 3u>(arguments, arena);
}

/* Wide form: the same record with row_count 1 .. STREAM_WIDE_ROWS (a BATCH
 * step's head: several verifiers' rows in one launch). A thread loads its
 * weights of a 4-output group once and multiplies them with every 8-row group
 * (passes of `per_pass` 8-row groups); each output-row chain, exchange tree
 * and warp order is the 8-row instance's, so every row's bits equal those of a
 * launch of its 8-row group alone. With DENSE_STEP_ROWS a row is
 * stored when it is below the step block's row_count and its step row holds
 * STEP_ROW_LOGITS. Launch: grid = co-resident blocks, 256 threads. */
#define STREAM_WIDE_ROWS 64u
#define STREAM_WIDE_GROUPS (STREAM_WIDE_ROWS / STREAM_ROWS)

template <uint32_t chunks_most, uint32_t per_pass>
__device__ static void
dense_stream_wide(struct dense_linear_arguments arguments, unsigned char *arena)
{
	__shared__ float warp_sums[STREAM_BUFFERS][STREAM_WARPS]
				  [per_pass * STREAM_VALUES];
	__shared__ float warp_squares[STREAM_WARPS][STREAM_ROWS];
	__shared__ float row_scale[STREAM_WIDE_ROWS];
	__shared__ unsigned long long stored_rows;
	const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP;
	const uint32_t chunk_count = arguments.input_size / STREAM_CHUNK_INPUTS;
	const uint32_t group_count = arguments.output_size / STREAM_OUTPUTS;
	const uint32_t row_count = arguments.row_count;
	const uint32_t row_groups =
		(row_count + STREAM_ROWS - 1u) / STREAM_ROWS;
	const uint32_t passes = (row_groups + per_pass - 1u) / per_pass;
	const bool is_row_scaled = (arguments.flags & DENSE_ROW_SCALE) != 0;
	const uint4 *const weights =
		(const uint4 *)(arena + arguments.weights_offset);
	const uint4 *const rows =
		(const uint4 *)(arena + arguments.rows_offset);
	LINEAR_DEVICE_CHECK(blockDim.x == STREAM_THREADS && row_count != 0 &&
			    row_count <= STREAM_WIDE_ROWS &&
			    chunk_count <= chunks_most * STREAM_THREADS);
	unsigned long long stored = 0;
	{
		const bool step_rows = (arguments.flags & DENSE_STEP_ROWS) != 0;
		const struct step_block *const step =
			(const struct step_block *)(arena +
						    arguments.step_offset);
		const struct step_row_record *const step_row =
			(const struct step_row_record *)(step + 1);
		for (uint32_t row = 0; row < row_count; row++)
			if (!step_rows ||
			    (row < step->row_count &&
			     (step_row[row].flags & STEP_ROW_LOGITS) != 0))
				stored |= 1ull << row;
	}
	if (stored == 0)
		return; /* the head of a step without logits reads nothing */
	/* Read back after a barrier: no register holds it across the loop. */
	if (threadIdx.x == 0)
		stored_rows = stored;
	if (is_row_scaled)
		for (uint32_t row_group = 0; row_group < row_groups;
		     row_group++) {
			float squares[STREAM_ROWS] = {};
#pragma unroll
			for (uint32_t step = 0; step < chunks_most; step++) {
				const uint32_t chunk =
					threadIdx.x + step * STREAM_THREADS;
#pragma unroll
				for (uint32_t local = 0; local < STREAM_ROWS;
				     local++) {
					const uint32_t row =
						row_group * STREAM_ROWS + local;
					if (chunk >= chunk_count ||
					    row >= row_count)
						continue;
					const uint4 part =
						rows[(size_t)row * chunk_count +
						     chunk];
#pragma unroll
					for (uint32_t index = 0;
					     index < STREAM_CHUNK_INPUTS;
					     index++) {
						const float value =
							stream_value(part,
								     index);
						squares[local] =
							fmaf(value, value,
							     squares[local]);
					}
				}
			}
#pragma unroll
			for (uint32_t distance = STREAM_FIRST_DISTANCE;
			     distance != 0; distance >>= 1u)
#pragma unroll
				for (uint32_t local = 0; local < STREAM_ROWS;
				     local++)
					squares[local] =
						squares[local] +
						__shfl_xor_sync(
							STREAM_FULL_WARP,
							squares[local],
							distance);
			if (lane == 0)
#pragma unroll
				for (uint32_t local = 0; local < STREAM_ROWS;
				     local++)
					warp_squares[warp][local] =
						squares[local];
			__syncthreads();
			if (threadIdx.x < STREAM_ROWS) {
				float square_sum = 0.0f;
				for (uint32_t index = 0; index < STREAM_WARPS;
				     index++)
					square_sum =
						square_sum +
						warp_squares[index][threadIdx.x];
				row_scale[row_group * STREAM_ROWS + threadIdx.x] =
					rsqrtf(square_sum *
						       (1.0f /
							(float)arguments
								.input_size) +
					       arguments.epsilon);
			}
			__syncthreads();
		}
	uint4 part_weights[chunks_most][STREAM_OUTPUTS];
	auto fetch = [&](uint32_t group) {
#pragma unroll
		for (uint32_t step = 0; step < chunks_most; step++) {
			const uint32_t chunk =
				threadIdx.x + step * STREAM_THREADS;
#pragma unroll
			for (uint32_t output = 0; output < STREAM_OUTPUTS;
			     output++)
				if (chunk < chunk_count)
					part_weights[step][output] = __ldcs(
						weights +
						((size_t)(group * STREAM_OUTPUTS +
							  output) *
							 chunk_count +
						 chunk));
		}
	};
	uint32_t buffer = 0;
	uint32_t group = blockIdx.x;
	if (group < group_count)
		fetch(group);
	for (; group < group_count; group += gridDim.x) {
		for (uint32_t pass = 0; pass < passes; pass++) {
			float sums[per_pass][STREAM_VALUES];
#pragma unroll
			for (uint32_t part = 0; part < per_pass; part++)
#pragma unroll
				for (uint32_t value = 0; value < STREAM_VALUES;
				     value++)
					sums[part][value] = 0.0f;
#pragma unroll
			for (uint32_t step = 0; step < chunks_most; step++) {
				const uint32_t chunk =
					threadIdx.x + step * STREAM_THREADS;
#pragma unroll
				for (uint32_t part = 0; part < per_pass; part++)
#pragma unroll
					for (uint32_t local = 0;
					     local < STREAM_ROWS; local++) {
						const uint32_t row =
							(pass * per_pass +
							 part) * STREAM_ROWS +
							local;
						if (chunk >= chunk_count ||
						    row >= row_count)
							continue;
						const uint4 row_part =
							rows[(size_t)row *
								     chunk_count +
							     chunk];
#pragma unroll
						for (uint32_t index = 0;
						     index <
						     STREAM_CHUNK_INPUTS;
						     index++) {
							const float value =
								stream_value(
									row_part,
									index);
#pragma unroll
							for (uint32_t output =
								     0;
							     output <
							     STREAM_OUTPUTS;
							     output++)
								sums[part][output * STREAM_ROWS +
									   local] =
									fmaf(stream_value(
										     part_weights
											     [step]
											     [output],
										     index),
									     value,
									     sums[part]
										 [output * STREAM_ROWS +
										  local]);
						}
					}
			}
			/* After the last pass the weights are free: the next
			 * group's travel during the exchange. */
			if (pass + 1u == passes &&
			    group + gridDim.x < group_count)
				fetch(group + gridDim.x);
#pragma unroll
			for (uint32_t part = 0; part < per_pass; part++) {
				stream_halve<16u>(sums[part], lane);
				stream_halve<8u>(sums[part], lane);
				stream_halve<4u>(sums[part], lane);
				stream_halve<2u>(sums[part], lane);
				stream_halve<1u>(sums[part], lane);
				/* Lane l: output l / 8, row l % 8. */
				warp_sums[buffer][warp][part * STREAM_VALUES +
							lane] = sums[part][0];
			}
			__syncthreads();
			if (threadIdx.x < per_pass * STREAM_VALUES) {
				const uint32_t part =
					threadIdx.x / STREAM_VALUES;
				const uint32_t value =
					threadIdx.x % STREAM_VALUES;
				const uint32_t output = value / STREAM_ROWS;
				const uint32_t row =
					(pass * per_pass + part) * STREAM_ROWS +
					value % STREAM_ROWS;
				if (row < row_count &&
				    ((stored_rows >> row) & 1ull) != 0) {
					float sum = 0.0f;
					for (uint32_t index = 0;
					     index < STREAM_WARPS; index++)
						sum = sum +
						      warp_sums[buffer][index]
							       [threadIdx.x];
					sum = is_row_scaled ?
						      sum * row_scale[row] :
						      sum;
					const size_t element =
						(size_t)row *
							arguments.output_size +
						(size_t)group * STREAM_OUTPUTS +
						output;
					if (arguments.output_type ==
					    DENSE_BFLOAT16)
						((uint16_t
							  *)(arena +
							     arguments
								     .output_offset))
							[element] = (uint16_t)linear_bfloat16_bits(
								linear_bfloat16_value(
									linear_bfloat16_bits(
										sum)) *
								arguments
									.output_scale);
					else
						((float *)(arena +
							   arguments
								   .output_offset))
							[element] =
								sum *
								arguments
									.output_scale;
				}
			}
			buffer ^= 1u;
		}
	}
}

/* (tp4_v3_claude, engine copy) src37's one-group form dense_stream_bfloat16_wide spills 24 bytes at two
 * blocks a multiprocessor under the engine's flags (sm_80: no spill allowed): left out; the two-group form below
 * takes its place, the same bits. */
/* Two 8-row groups a pass (sums of 16 rows in registers, one block a
 * multiprocessor): half the barriers and passes, same bits. */
extern "C" __global__ void __launch_bounds__(STREAM_THREADS, 1)
	dense_stream_bfloat16_wide16(struct dense_linear_arguments arguments,
				     unsigned char *arena)
{
	dense_stream_wide<3u, 2u>(arguments, arena);
}
