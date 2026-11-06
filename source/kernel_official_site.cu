/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Serial M1 site finish: source BF16 collapse, contiguous-vector4 RMS order and Sinkhorn. */
#include "kernel_hyper_connection.cu"

#define OFFICIAL_SITE_THREADS 512u
#define OFFICIAL_SITE_VECTOR 4u
#define OFFICIAL_SITE_ROUNDS                                                  \
	((MODEL_HIDDEN + OFFICIAL_SITE_THREADS * OFFICIAL_SITE_VECTOR - 1u) / \
	 (OFFICIAL_SITE_THREADS * OFFICIAL_SITE_VECTOR))
#define OFFICIAL_SITE_PAIR 2u
#define OFFICIAL_SITE_THIRD_CHAIN 2u
#define OFFICIAL_SITE_LAST_CHAIN 3u
#define OFFICIAL_SITE_EVERY_LANE 0xFFFFFFFFu

extern "C" __global__ void
official_site_finish(struct hyper_finish_arguments arguments,
		     unsigned char *arena)
{
	__shared__ float sums[OFFICIAL_SITE_THREADS], mean, inverse;
	const uint32_t thread = threadIdx.x, lane = thread % THREADS_PER_WARP;
	const size_t row = blockIdx.y;
	const bool is_sinkhorn = thread >= OFFICIAL_SITE_THREADS;
	float *const carried =
		(float *)(arena + arguments.pre_offset) + row * STREAMS;
	float hidden[OFFICIAL_SITE_ROUNDS][OFFICIAL_SITE_VECTOR] = {};
	float chains[OFFICIAL_SITE_VECTOR] = {}, next_pre = 0.0f;
	LINEAR_DEVICE_CHECK(blockDim.x ==
				    OFFICIAL_SITE_THREADS + THREADS_PER_WARP &&
			    gridDim.x == 1 && gridDim.z == 1);
	if (is_sinkhorn)
		next_pre = hyper_sinkhorn(&arguments, arena, row, lane);
	else {
#pragma unroll
		for (uint32_t round = 0; round < OFFICIAL_SITE_ROUNDS;
		     round++) {
			const uint32_t first =
				(round * OFFICIAL_SITE_THREADS + thread) *
				OFFICIAL_SITE_VECTOR;
			if (first < MODEL_HIDDEN) {
#pragma unroll
				for (uint32_t slot = 0;
				     slot < OFFICIAL_SITE_VECTOR; slot++) {
					const uint32_t column = first + slot;
					float value = 0.0f;
#pragma unroll
					for (uint32_t stream = 0;
					     stream < STREAMS; stream++) {
						const uint64_t place =
							(row * STREAMS +
							 stream) *
								MODEL_HIDDEN +
							column;
						const float input = linear_bfloat16_value(
							((const uint16_t
								  *)(arena +
								     arguments
									     .streams_offset))
								[place]);
						value = __fadd_rn(
							value,
							__fmul_rn(
								carried[stream],
								input));
					}
					const uint16_t bits =
						(uint16_t)linear_bfloat16_bits(
							value);
					const float rounded =
						linear_bfloat16_value(bits);
					hidden[round][slot] = rounded;
					chains[slot] = __fadd_rn(
						chains[slot],
						__fmul_rn(rounded, rounded));
				}
			}
		}
		sums[thread] =
			__fadd_rn(__fadd_rn(__fadd_rn(chains[0], chains[1]),
					    chains[OFFICIAL_SITE_THIRD_CHAIN]),
				  chains[OFFICIAL_SITE_LAST_CHAIN]);
	}
	__syncthreads();
	for (uint32_t distance = OFFICIAL_SITE_THREADS / OFFICIAL_SITE_PAIR;
	     distance >= THREADS_PER_WARP; distance /= OFFICIAL_SITE_PAIR) {
		if (thread < distance)
			sums[thread] = __fadd_rn(sums[thread],
						 sums[thread + distance]);
		__syncthreads();
	}
	if (thread < THREADS_PER_WARP) {
		float total = sums[thread];
		for (uint32_t distance = THREADS_PER_WARP / OFFICIAL_SITE_PAIR;
		     distance != 0; distance /= OFFICIAL_SITE_PAIR)
			total = __fadd_rn(
				total,
				__shfl_down_sync(OFFICIAL_SITE_EVERY_LANE,
						 total, distance));
		if (thread == 0) {
			mean = __fmul_rn(total, 1.0f / (float)MODEL_HIDDEN);
			inverse = rsqrtf(__fadd_rn(mean, MODEL_NORM_EPSILON));
		}
	}
	__syncthreads();
	if (is_sinkhorn) {
		if (lane < STREAMS)
			carried[lane] = next_pre;
		return;
	}
	const uint16_t *const weights =
		(const uint16_t *)(arena + arguments.weight_offset);
	uint16_t *const output = (uint16_t *)(arena + arguments.hidden_offset) +
				 row * MODEL_HIDDEN;
#pragma unroll
	for (uint32_t round = 0; round < OFFICIAL_SITE_ROUNDS; round++) {
		const uint32_t first =
			(round * OFFICIAL_SITE_THREADS + thread) *
			OFFICIAL_SITE_VECTOR;
		if (first < MODEL_HIDDEN) {
#pragma unroll
			for (uint32_t slot = 0; slot < OFFICIAL_SITE_VECTOR;
			     slot++) {
				const uint32_t column = first + slot;
				output[column] = (uint16_t)linear_bfloat16_bits(
					__fmul_rn(linear_bfloat16_value(
							  weights[column]),
						  __fmul_rn(hidden[round][slot],
							    inverse)));
			}
		}
	}
}
