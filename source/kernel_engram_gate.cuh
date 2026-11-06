/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The Engram gate arithmetic, shared by ordinary and image-masked entry points. Device code only. */
#ifndef KERNEL_ENGRAM_GATE_CUH
#define KERNEL_ENGRAM_GATE_CUH
#include "anemone/engram_rows_records.h"
#include "kernel_linear_common.cuh"

#define GATE_WARPS (ENGRAM_ROWS_GATE_THREADS / THREADS_PER_WARP)
#define VALUES_PER_THREAD (MODEL_HIDDEN / ENGRAM_ROWS_GATE_THREADS)
#define EVERY_LANE 0xFFFFFFFFu
#define INVERSE_HIDDEN \
	(1.0f / (float)MODEL_HIDDEN) /* the reference's mean multiplies by it */
#define DOT_SCALE      \
	((float)(1.0 / \
		 sqrt((double)MODEL_HIDDEN))) /* model.py:360: dim**-0.5, the same double */
#define SMALLEST_MAGNITUDE \
	((float)1.0e-6) /* model.py:341 clamp_value, as the reference's float32 */

enum gate_sum { STREAM_SQUARES, KEY_SQUARES, DOT, SUM_COUNT };

__device__ static __forceinline__ void
engram_gate(struct engram_gate_arguments arguments, unsigned char *arena)
{
	LINEAR_DEVICE_CHECK(blockDim.x == ENGRAM_ROWS_GATE_THREADS &&
			    gridDim.x == MODEL_STREAMS);
	__shared__ float warp_sums[SUM_COUNT][GATE_WARPS];
	const size_t row = blockIdx.y;
	const size_t stream_offset = (size_t)blockIdx.x * MODEL_HIDDEN;
	const uint16_t *const projection =
		(const uint16_t *)(arena + arguments.projection_offset) +
		row * ENGRAM_ROWS_PROJECTION;
	const uint16_t *const key = projection + stream_offset;
	const uint16_t *const value =
		projection + (size_t)MODEL_STREAMS * MODEL_HIDDEN;
	const uint16_t *const query_weight =
		(const uint16_t *)(arena + arguments.query_weight_offset) +
		stream_offset;
	const uint16_t *const key_weight =
		(const uint16_t *)(arena + arguments.key_weight_offset) +
		stream_offset;
	uint16_t *const stream =
		(uint16_t *)(arena + arguments.streams_offset) +
		row * MODEL_STREAMS * MODEL_HIDDEN + stream_offset;
	float values[VALUES_PER_THREAD];
	float sums[SUM_COUNT] = { 0.0f, 0.0f, 0.0f };
#pragma unroll
	for (uint32_t step = 0; step < VALUES_PER_THREAD; step++) {
		const uint32_t index =
			step * ENGRAM_ROWS_GATE_THREADS + threadIdx.x;
		const float key_value = linear_bfloat16_value(key[index]);
		const float weight =
			linear_bfloat16_value(query_weight[index]) *
			linear_bfloat16_value(key_weight[index]);
		values[step] = linear_bfloat16_value(stream[index]);
		sums[STREAM_SQUARES] += values[step] * values[step];
		sums[KEY_SQUARES] += key_value * key_value;
		sums[DOT] += values[step] * weight * key_value;
	}
#pragma unroll
	for (uint32_t sum = 0; sum < SUM_COUNT; sum++) {
		for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0;
		     distance >>= 1u) {
			sums[sum] += __shfl_xor_sync(EVERY_LANE, sums[sum],
						     (int)distance);
		}
		if (threadIdx.x % THREADS_PER_WARP == 0) {
			warp_sums[sum][threadIdx.x / THREADS_PER_WARP] =
				sums[sum];
		}
	}
	__syncthreads();
#pragma unroll
	for (uint32_t sum = 0; sum < SUM_COUNT; sum++) {
		sums[sum] = 0.0f;
		for (uint32_t warp = 0; warp < GATE_WARPS; warp++) {
			sums[sum] += warp_sums[sum][warp];
		}
	}
	const float rstd =
		rsqrtf(sums[STREAM_SQUARES] * INVERSE_HIDDEN +
		       MODEL_NORM_EPSILON) *
		rsqrtf(sums[KEY_SQUARES] * INVERSE_HIDDEN + MODEL_NORM_EPSILON);
	const float dot = sums[DOT] * rstd * DOT_SCALE;
	const float magnitude = fabsf(dot);
	const float root = sqrtf(magnitude < SMALLEST_MAGNITUDE ?
					 SMALLEST_MAGNITUDE :
					 magnitude); /* keeps a NaN */
	const float gate = 1.0f / (1.0f + expf(-copysignf(root, dot)));
#pragma unroll
	for (uint32_t step = 0; step < VALUES_PER_THREAD; step++) {
		const uint32_t index =
			step * ENGRAM_ROWS_GATE_THREADS + threadIdx.x;
		stream[index] = (uint16_t)linear_bfloat16_bits(
			values[step] +
			gate * linear_bfloat16_value(value[index]));
	}
}

#endif
