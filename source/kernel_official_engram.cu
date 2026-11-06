/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Source cached-one Engram tensors have four reduction rows. One independent
 * CTA preserves each row's reference X=128 contiguous-vector4 ownership/tree.
 * The old512-owner gate is retained in its original module and tap control.
 */
#include "kernel_linear_common.cuh"
#include "anemone/engram_rows_records.h"
#define OFFICIAL_ENGRAM_THREADS 128u
#define OFFICIAL_ENGRAM_VECTOR 4u
#define OFFICIAL_ENGRAM_ROUNDS \
	(MODEL_HIDDEN / (OFFICIAL_ENGRAM_THREADS * OFFICIAL_ENGRAM_VECTOR))
#define OFFICIAL_ENGRAM_EVERY_LANE 0xFFFFFFFFu
#define OFFICIAL_ENGRAM_CLAMP 1.0e-6f
#define OFFICIAL_ENGRAM_SCALE ((float)(1.0 / sqrt((double)MODEL_HIDDEN)))
#define OFFICIAL_ENGRAM_PAIR 2u
#define OFFICIAL_ENGRAM_THIRD_CHAIN 2u
#define OFFICIAL_ENGRAM_LAST_CHAIN 3u
enum official_engram_statistic {
	OFFICIAL_ENGRAM_QUERY_SQUARE,
	OFFICIAL_ENGRAM_KEY_SQUARE,
	OFFICIAL_ENGRAM_WEIGHTED_DOT,
	OFFICIAL_ENGRAM_STATISTICS
};
extern "C" __global__ void
official_engram_gate(struct engram_gate_arguments arguments,
		     unsigned char *arena)
{
	__shared__ float partial[OFFICIAL_ENGRAM_STATISTICS]
				[OFFICIAL_ENGRAM_THREADS],
		shared_gate;
	const uint32_t thread = threadIdx.x, stream = blockIdx.x;
	const uint64_t row = blockIdx.y;
	const uint64_t offset = (row * MODEL_STREAMS + stream) * MODEL_HIDDEN;
	uint16_t *streams =
		(uint16_t *)(arena + arguments.streams_offset) + offset;
	const uint16_t *key =
		(const uint16_t *)(arena + arguments.projection_offset) +
		row * ENGRAM_ROWS_PROJECTION + stream * MODEL_HIDDEN;
	const uint16_t *value =
		(const uint16_t *)(arena + arguments.projection_offset) +
		row * ENGRAM_ROWS_PROJECTION + MODEL_STREAMS * MODEL_HIDDEN;
	const uint16_t *query_weight =
		(const uint16_t *)(arena + arguments.query_weight_offset) +
		stream * MODEL_HIDDEN;
	const uint16_t *key_weight =
		(const uint16_t *)(arena + arguments.key_weight_offset) +
		stream * MODEL_HIDDEN;
	float inputs[OFFICIAL_ENGRAM_ROUNDS][OFFICIAL_ENGRAM_VECTOR],
		chains[OFFICIAL_ENGRAM_STATISTICS][OFFICIAL_ENGRAM_VECTOR] = {};
#pragma unroll
	for (uint32_t round = 0; round < OFFICIAL_ENGRAM_ROUNDS; round++) {
#pragma unroll
		for (uint32_t slot = 0; slot < OFFICIAL_ENGRAM_VECTOR; slot++) {
			const uint32_t column =
				(round * OFFICIAL_ENGRAM_THREADS + thread) *
					OFFICIAL_ENGRAM_VECTOR +
				slot;
			const float query_value = linear_bfloat16_value(
					    streams[column]),
				    key_value =
					    linear_bfloat16_value(key[column]);
			const float combined_weight = __fmul_rn(
				linear_bfloat16_value(query_weight[column]),
				linear_bfloat16_value(key_weight[column]));
			inputs[round][slot] = query_value;
			chains[OFFICIAL_ENGRAM_QUERY_SQUARE][slot] = __fadd_rn(
				chains[OFFICIAL_ENGRAM_QUERY_SQUARE][slot],
				__fmul_rn(query_value, query_value));
			chains[OFFICIAL_ENGRAM_KEY_SQUARE][slot] = __fadd_rn(
				chains[OFFICIAL_ENGRAM_KEY_SQUARE][slot],
				__fmul_rn(key_value, key_value));
			chains[OFFICIAL_ENGRAM_WEIGHTED_DOT][slot] = __fadd_rn(
				chains[OFFICIAL_ENGRAM_WEIGHTED_DOT][slot],
				__fmul_rn(__fmul_rn(query_value,
						    combined_weight),
					  key_value));
		}
	}
#pragma unroll
	for (uint32_t statistic = 0; statistic < OFFICIAL_ENGRAM_STATISTICS;
	     statistic++)
		partial[statistic][thread] = __fadd_rn(
			__fadd_rn(
				__fadd_rn(chains[statistic][0],
					  chains[statistic][1]),
				chains[statistic][OFFICIAL_ENGRAM_THIRD_CHAIN]),
			chains[statistic][OFFICIAL_ENGRAM_LAST_CHAIN]);
	__syncthreads();
	for (uint32_t distance = OFFICIAL_ENGRAM_THREADS / OFFICIAL_ENGRAM_PAIR;
	     distance >= THREADS_PER_WARP; distance /= OFFICIAL_ENGRAM_PAIR) {
		if (thread < distance) {
#pragma unroll
			for (uint32_t statistic = 0;
			     statistic < OFFICIAL_ENGRAM_STATISTICS;
			     statistic++)
				partial[statistic][thread] = __fadd_rn(
					partial[statistic][thread],
					partial[statistic][thread + distance]);
		}
		__syncthreads();
	}
	if (thread < THREADS_PER_WARP) {
		float sums[OFFICIAL_ENGRAM_STATISTICS];
#pragma unroll
		for (uint32_t statistic = 0;
		     statistic < OFFICIAL_ENGRAM_STATISTICS; statistic++) {
			sums[statistic] = partial[statistic][thread];
			for (uint32_t distance =
				     THREADS_PER_WARP / OFFICIAL_ENGRAM_PAIR;
			     distance != 0; distance /= OFFICIAL_ENGRAM_PAIR)
				sums[statistic] = __fadd_rn(
					sums[statistic],
					__shfl_down_sync(
						OFFICIAL_ENGRAM_EVERY_LANE,
						sums[statistic], distance));
		}
		if (thread == 0) {
			const float query_mean = __fmul_rn(
					    sums[OFFICIAL_ENGRAM_QUERY_SQUARE],
					    1.0f / (float)MODEL_HIDDEN),
				    key_mean = __fmul_rn(
					    sums[OFFICIAL_ENGRAM_KEY_SQUARE],
					    1.0f / (float)MODEL_HIDDEN);
			const float query_inverse = rsqrtf(__fadd_rn(
					    query_mean, MODEL_NORM_EPSILON)),
				    key_inverse = rsqrtf(__fadd_rn(
					    key_mean, MODEL_NORM_EPSILON));
			const float
				combined_inverse =
					__fmul_rn(query_inverse, key_inverse),
				scaled_dot = __fmul_rn(
					__fmul_rn(
						sums[OFFICIAL_ENGRAM_WEIGHTED_DOT],
						combined_inverse),
					OFFICIAL_ENGRAM_SCALE);
			const float signed_root =
				copysignf(sqrtf(fmaxf(fabsf(scaled_dot),
						      OFFICIAL_ENGRAM_CLAMP)),
					  scaled_dot);
			shared_gate = 1.0f / (1.0f + expf(-signed_root));
		}
	}
	__syncthreads();
#pragma unroll
	for (uint32_t round = 0; round < OFFICIAL_ENGRAM_ROUNDS; round++) {
#pragma unroll
		for (uint32_t slot = 0; slot < OFFICIAL_ENGRAM_VECTOR; slot++) {
			const uint32_t column =
				(round * OFFICIAL_ENGRAM_THREADS + thread) *
					OFFICIAL_ENGRAM_VECTOR +
				slot;
			streams[column] = (uint16_t)linear_bfloat16_bits(
				__fadd_rn(inputs[round][slot],
					  __fmul_rn(shared_gate,
						    linear_bfloat16_value(
							    value[column]))));
		}
	}
}
