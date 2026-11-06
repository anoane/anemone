/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_head_mma.cu: y = W x for bfloat16 weights [output_size][input_size] (the head) and 1 to 64 bfloat16 rows on
 * the tensor cores, FP32 out (--tp_head_mma, the BF16 recipe; GNmac's head is a cuBLAS BF16 GEMM: the same class of
 * arithmetic). Device code only, sm_80, module "kernel_head_mma", the dense record (dense_linear_records.h): the step
 * rows' flags gate the stores as for the head's dense product.
 *
 * A warp owns 16 outputs and every row: per 32 inputs a lane loads 16 bytes of output rows g and g + 8 (inputs
 * 8 t .. 8 t + 7 of the 32, t = lane % 4) and 8 bytes of each row group's activations; the two m16n8k16 MMAs of the 32
 * inputs take those words as they are (inside each 32 the MMA's k order is the permutation that maps its lane pairs
 * onto 8 t + 4 s + {0, 1} and {2, 3}, the same for both operands, so every product meets its partner). Each MMA starts
 * from the warp's running sums (the FP32 accumulation of the tensor core, inputs in ascending 32-blocks); the _sN forms
 * split a tile's inputs over N warps of the block and add the N sums in warp order: fixed, so deterministic. Loads run
 * HEAD_MMA_DEPTH 32-input steps ahead.
 */
#include "anemone/dense_linear_records.h"
#include "anemone/step_records.h"

#include <cuda_bf16.h>
#include <stdint.h>

#define HEAD_MMA_THREADS 256
#define HEAD_MMA_WARPS (HEAD_MMA_THREADS / 32)
#define HEAD_MMA_TILE_OUTPUTS 16
#define HEAD_MMA_BLOCK_OUTPUTS (HEAD_MMA_WARPS * HEAD_MMA_TILE_OUTPUTS)
#define HEAD_MMA_STEP_INPUTS 32
#define HEAD_MMA_DEPTH 4
#define HEAD_MMA_MOST_GROUPS 8 /* 64 rows: 8 n8 groups */

__device__ __forceinline__ uint4 head_mma_weights(const uint4 *source)
{
	uint4 value;
	asm volatile(
		"ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
		: "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
		: "l"(source));
	return value;
}

__device__ __forceinline__ void head_mma_step(float (&d)[4], uint32_t a0,
					      uint32_t a1, uint32_t a2,
					      uint32_t a3, uint32_t b0,
					      uint32_t b1)
{
	asm volatile(
		"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
		"{%0,%1,%2,%3};"
		: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
		: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* The warp's running sums of its 16 outputs over the 32-input steps [first_step, end_step), every row group. */
template <int groups>
__device__ __forceinline__ void
head_mma_sums(const struct dense_linear_arguments &arguments,
	      const unsigned char *arena, uint32_t first_output,
	      uint32_t first_step, uint32_t end_step, float (&sums)[groups][4])
{
	const int lane = threadIdx.x & 31, quad = lane & 3, column = lane >> 2;
	const uint32_t inputs = arguments.input_size;
	const uint32_t last_output = arguments.output_size - 1u;
	const uint32_t row_low = min(first_output + column, last_output),
		       row_high = min(first_output + column + 8u, last_output);
	const uint4 *weights_low =
		(const uint4 *)(arena + arguments.weights_offset +
				(uint64_t)row_low * inputs * 2u) +
		quad;
	const uint4 *weights_high =
		(const uint4 *)(arena + arguments.weights_offset +
				(uint64_t)row_high * inputs * 2u) +
		quad;
	const __nv_bfloat16 *values =
		(const __nv_bfloat16 *)(arena + arguments.rows_offset);
	const uint2 *activation[groups];
#pragma unroll
	for (int group = 0; group < groups; group++) {
		const uint32_t row = min((uint32_t)(8 * group + column),
					 arguments.row_count - 1u);
		activation[group] =
			(const uint2 *)(values + (uint64_t)row * inputs) +
			2 * quad;
	}
#pragma unroll
	for (int group = 0; group < groups; group++)
#pragma unroll
		for (int index = 0; index < 4; index++)
			sums[group][index] = 0.0f;
	if (first_step >= end_step)
		return;
	uint4 ring_low[HEAD_MMA_DEPTH], ring_high[HEAD_MMA_DEPTH];
#pragma unroll
	for (int ahead = 0; ahead < HEAD_MMA_DEPTH; ahead++) {
		const uint32_t step = min(first_step + ahead, end_step - 1u);
		ring_low[ahead] = head_mma_weights(weights_low + step * 4u);
		ring_high[ahead] = head_mma_weights(weights_high + step * 4u);
	}
#pragma unroll 1
	for (uint32_t step = first_step; step < end_step;
	     step += HEAD_MMA_DEPTH) {
#pragma unroll
		for (int slot = 0; slot < HEAD_MMA_DEPTH; slot++) {
			const uint32_t at = step + slot;
			if (at >= end_step)
				break;
			const uint4 low = ring_low[slot],
				    high = ring_high[slot];
			const uint32_t next =
				min(at + HEAD_MMA_DEPTH, end_step - 1u);
			ring_low[slot] =
				head_mma_weights(weights_low + next * 4u);
			ring_high[slot] =
				head_mma_weights(weights_high + next * 4u);
#pragma unroll
			for (int group = 0; group < groups; group++) {
				const uint2 first =
					__ldg(activation[group] + at * 8u);
				const uint2 second =
					__ldg(activation[group] + at * 8u + 1u);
				head_mma_step(sums[group], low.x, high.x, low.y,
					      high.y, first.x, first.y);
				head_mma_step(sums[group], low.z, high.z, low.w,
					      high.w, second.x, second.y);
			}
		}
	}
}

/* Rows whose outputs are stored: below row_count and, with DENSE_STEP_ROWS, flagged for logits in the step
 * area (the head's dense product's rule); 64 bits. */
__device__ __forceinline__ uint64_t
head_mma_stored_rows(const struct dense_linear_arguments &arguments,
		     const unsigned char *arena)
{
	uint64_t mask = 0;
	const bool step_rows = (arguments.flags & DENSE_STEP_ROWS) != 0;
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	for (uint32_t row = 0; row < arguments.row_count; row++)
		if (!step_rows ||
		    (row < step->row_count &&
		     (((const struct step_row_record *)(step + 1))[row].flags &
		      STEP_ROW_LOGITS) != 0))
			mask |= 1ull << row;
	return mask;
}

/* A tile of 16 outputs split over `split` warps by input steps: each sums its share, the first adds the others' in
 * warp order (fixed: deterministic) and stores. Every thread of the block reaches the barrier. */
template <int groups, int split, int most_groups>
__device__ __forceinline__ void
head_mma_tile(const struct dense_linear_arguments &arguments,
	      unsigned char *arena,
	      float (&partials)[HEAD_MMA_WARPS][4 * most_groups][32])
{
	const int lane = threadIdx.x & 31, quad = lane & 3, column = lane >> 2;
	const uint32_t warp = threadIdx.x >> 5, part = warp % split;
	const uint32_t first_output =
		(blockIdx.x * (HEAD_MMA_WARPS / split) + warp / split) *
		HEAD_MMA_TILE_OUTPUTS;
	const uint32_t steps = arguments.input_size / HEAD_MMA_STEP_INPUTS;
	float sums[groups][4];
	head_mma_sums<groups>(arguments, arena, first_output,
			      steps * part / split, steps * (part + 1u) / split,
			      sums);
	if constexpr (split > 1) {
		if (part != 0)
#pragma unroll
			for (int group = 0; group < groups; group++)
#pragma unroll
				for (int index = 0; index < 4; index++)
					partials[warp][4 * group + index][lane] =
						sums[group][index];
		__syncthreads();
		if (part != 0)
			return;
#pragma unroll
		for (uint32_t other = 1; other < split; other++)
#pragma unroll
			for (int group = 0; group < groups; group++)
#pragma unroll
				for (int index = 0; index < 4; index++)
					sums[group][index] = __fadd_rn(
						sums[group][index],
						partials[warp + other]
							[4 * group + index]
							[lane]);
	}
	if (first_output >= arguments.output_size)
		return;
	const uint64_t stored_rows = head_mma_stored_rows(arguments, arena);
	/* d0, d1: output column g, rows 8 group + 2 quad (+1); d2, d3: output column g + 8 */
	float *output = (float *)(arena + arguments.output_offset);
#pragma unroll
	for (int group = 0; group < groups; group++) {
#pragma unroll
		for (int pair = 0; pair < 2; pair++) {
			const uint32_t row = 8u * group + 2u * quad + pair;
			if (row >= arguments.row_count ||
			    ((stored_rows >> row) & 1u) == 0u)
				continue;
#pragma unroll
			for (int half = 0; half < 2; half++) {
				const uint32_t out =
					first_output + column + 8u * half;
				if (out < arguments.output_size)
					output[(uint64_t)row *
						       arguments.output_size +
					       out] =
						__fmul_rn(
							sums[group]
							    [2 * half + pair],
							arguments.output_scale);
			}
		}
	}
}

/* grid ceil(output_size / (128 / split)), 256 threads, input_size a multiple of 32: a block 8 / split tiles of 16
 * outputs, each split over `split` warps by input steps (a tile per warp leaves the last wave of the vocabulary shard
 * mostly idle); the _8 forms rows 1 to 8 (registers for one row group: the most warps resident), the _64 forms 9 to 64. */
template <int least_groups, int most_groups, int split>
__device__ __forceinline__ void
head_mma_launch(const struct dense_linear_arguments &arguments,
		unsigned char *arena)
{
	__shared__ float partials[HEAD_MMA_WARPS][4 * most_groups][32];
	const uint32_t groups = (arguments.row_count + 7u) / 8u;
	if (groups < least_groups || groups > most_groups ||
	    arguments.input_size % HEAD_MMA_STEP_INPUTS != 0)
		return; /* the same for the whole block */
	switch (groups) {
#define HEAD_MMA_GROUPS(count)                                               \
	case count:                                                          \
		if constexpr (count >= least_groups && count <= most_groups) \
			head_mma_tile<count, split, most_groups>(            \
				arguments, arena, partials);                 \
		break;
		HEAD_MMA_GROUPS(1)
		HEAD_MMA_GROUPS(2) HEAD_MMA_GROUPS(3) HEAD_MMA_GROUPS(4)
			HEAD_MMA_GROUPS(5) HEAD_MMA_GROUPS(6) HEAD_MMA_GROUPS(7)
				HEAD_MMA_GROUPS(8)
#undef HEAD_MMA_GROUPS
					default : break;
	}
}

#define HEAD_MMA_FORM(name, least, most, split)                        \
	extern "C" __global__ void __launch_bounds__(HEAD_MMA_THREADS) \
		name(const struct dense_linear_arguments arguments,    \
		     unsigned char *arena)                             \
	{                                                              \
		head_mma_launch<least, most, split>(arguments, arena); \
	}
HEAD_MMA_FORM(head_mma_8_s1, 1, 1, 1)
HEAD_MMA_FORM(head_mma_8_s2, 1, 1, 2)
HEAD_MMA_FORM(head_mma_8_s4, 1, 1, 4)
HEAD_MMA_FORM(head_mma_8_s8, 1, 1, 8)
HEAD_MMA_FORM(head_mma_64_s1, 2, HEAD_MMA_MOST_GROUPS, 1)
HEAD_MMA_FORM(head_mma_64_s2, 2, HEAD_MMA_MOST_GROUPS, 2)
HEAD_MMA_FORM(head_mma_64_s4, 2, HEAD_MMA_MOST_GROUPS, 4)

/* --tp_router_mma: the router of 9 to 64 rows by split-K (dense_linear_records.h's router_mma_arguments). A
 * warp owns one (tile of 16 outputs, split of the 32-input steps) item, ROUTER_MMA_WARPS a block, head_mma's sums over its
 * steps, its FP32 partials [split][row][output]; the reduce adds the splits from +0 in ascending order. */
#define ROUTER_MMA_WARPS 4
#define ROUTER_MMA_REDUCE_THREADS 256

/* split_rows: the rows of a split's partials (the launch's); first_row: dense's rows' first among them. */
template <int groups>
__device__ __forceinline__ void
router_mma_item(const struct router_mma_arguments &arguments,
		unsigned char *arena, uint32_t item, uint32_t split_rows = 0,
		uint32_t first_row = 0)
{
	const struct dense_linear_arguments &dense = arguments.dense;
	const uint32_t tiles = dense.output_size / HEAD_MMA_TILE_OUTPUTS;
	const uint32_t tile = item % tiles, split = item / tiles;
	const uint32_t steps = dense.input_size / HEAD_MMA_STEP_INPUTS;
	float sums[groups][4];
	head_mma_sums<groups>(dense, arena, tile * HEAD_MMA_TILE_OUTPUTS, steps * split / arguments.split_count,
			      steps * (split + 1u) / arguments.split_count, sums);
	const int lane = threadIdx.x & 31, quad = lane & 3, column = lane >> 2;
	float *const partials = (float *)(arena + arguments.partials_offset) +
				(uint64_t)split * (split_rows != 0 ? split_rows : dense.row_count) * dense.output_size +
				(uint64_t)first_row * dense.output_size;
	/* d0, d1: output g, rows 8 group + 2 quad (+ 1); d2, d3: output g + 8 */
#pragma unroll
	for (int group = 0; group < groups; group++)
#pragma unroll
		for (int pair = 0; pair < 2; pair++) {
			const uint32_t row = 8u * group + 2u * quad + pair;
			if (row >= dense.row_count)
				continue;
#pragma unroll
			for (int half = 0; half < 2; half++)
				partials[(uint64_t)row * dense.output_size + tile * HEAD_MMA_TILE_OUTPUTS + column +
					 8u * half] = sums[group][2 * half + pair];
		}
}

/* grid ceil(output_size / 16 * split_count / ROUTER_MMA_WARPS), 32 ROUTER_MMA_WARPS threads */
extern "C" __global__ void __launch_bounds__(32 * ROUTER_MMA_WARPS)
	router_mma_entry(const struct router_mma_arguments arguments,
			 unsigned char *arena)
{
	const uint32_t item = blockIdx.x * ROUTER_MMA_WARPS + (threadIdx.x >> 5);
	if (item >= arguments.dense.output_size / HEAD_MMA_TILE_OUTPUTS * arguments.split_count)
		return;
	switch ((arguments.dense.row_count + 7u) / 8u) {
#define ROUTER_MMA_GROUPS(count) \
	case count: \
		router_mma_item<count>(arguments, arena, item); \
		break;
		ROUTER_MMA_GROUPS(2)
		ROUTER_MMA_GROUPS(3)
		ROUTER_MMA_GROUPS(4)
		ROUTER_MMA_GROUPS(5)
		ROUTER_MMA_GROUPS(6)
		ROUTER_MMA_GROUPS(7)
		ROUTER_MMA_GROUPS(8)
#undef ROUTER_MMA_GROUPS
	default:
		break;
	}
}

/* --tp_router_mma=2, a wide (prefill) step's router of more than 64 rows: the 64-row form over blocks of 64 rows (grid
 * y), the partials [split][row][output] of every row; a row's sums are the 64-row form's (each row its own). */
extern "C" __global__ void __launch_bounds__(32 * ROUTER_MMA_WARPS, 1)
	router_mma_wide_entry(const struct router_mma_arguments arguments,
			      unsigned char *arena)
{
	const uint32_t item = blockIdx.x * ROUTER_MMA_WARPS + (threadIdx.x >> 5);
	const uint32_t first_row = blockIdx.y * HEAD_MMA_MOST_GROUPS * 8u, rows = arguments.dense.row_count;
	if (item >= arguments.dense.output_size / HEAD_MMA_TILE_OUTPUTS * arguments.split_count || first_row >= rows)
		return;
	const uint32_t block_rows = min(rows - first_row, (uint32_t)HEAD_MMA_MOST_GROUPS * 8u);
	struct router_mma_arguments block = arguments;
	block.dense.rows_offset += (uint64_t)first_row * arguments.dense.input_size * sizeof(uint16_t);
	block.dense.row_count = block_rows;
	switch ((block_rows + 7u) / 8u) {
#define ROUTER_MMA_GROUPS(count)                                                     \
	case count:                                                                  \
		router_mma_item<count>(block, arena, item, rows, first_row);         \
		break;
		ROUTER_MMA_GROUPS(1)
		ROUTER_MMA_GROUPS(2)
		ROUTER_MMA_GROUPS(3)
		ROUTER_MMA_GROUPS(4)
		ROUTER_MMA_GROUPS(5)
		ROUTER_MMA_GROUPS(6)
		ROUTER_MMA_GROUPS(7)
		ROUTER_MMA_GROUPS(8)
#undef ROUTER_MMA_GROUPS
	default:
		break;
	}
}

/* grid ceil(row_count * output_size / ROUTER_MMA_REDUCE_THREADS): a thread an output of a row */
extern "C" __global__ void __launch_bounds__(ROUTER_MMA_REDUCE_THREADS)
	router_mma_reduce(const struct router_mma_arguments arguments,
			  unsigned char *arena)
{
	const uint64_t results = (uint64_t)arguments.dense.row_count * arguments.dense.output_size;
	const uint64_t result = (uint64_t)blockIdx.x * ROUTER_MMA_REDUCE_THREADS + threadIdx.x;
	if (result >= results)
		return;
	const float *const partials = (const float *)(arena + arguments.partials_offset) + result;
	float sum = 0.0f;
	for (uint32_t split = 0; split < arguments.split_count; split++)
		sum += partials[(uint64_t)split * results];
	((float *)(arena + arguments.dense.output_offset))[result] = sum;
}
