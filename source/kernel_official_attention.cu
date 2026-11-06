/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Release serial M1 sparse core. Explicit BF16 boundaries and source K64
 * online updates are shared across both card architectures. Each head owns its
 * private CTA stage; no intermediate is written outside the final output. */
#include "kernel_linear_common.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "anemone/official_attention_records.h"
#include <math_constants.h>
#define ATTENTION_PAIR_WIDTH 2u
#define ATTENTION_STAGE_ALIGNMENT 32u
#define ATTENTION_STATE_RESCALE 2u
#define ATTENTION_SECOND_OWNER 2u
#define ATTENTION_LAST_OWNER 3u
#define ATTENTION_OWNER_PARTS 4u
#define ATTENTION_OWNER_VALUES 8u
#define ATTENTION_HALF_TILE_KEYS 32u
#define ATTENTION_BFLOAT16_SHIFT 16u
#define ATTENTION_TILE_KEYS 64u
#define ATTENTION_PRODUCT_GROUP ATTENTION_OWNER_VALUES
#define ATTENTION_THREAD_VALUES (MODEL_HEAD_DIMS / OFFICIAL_ATTENTION_THREADS)
#define ATTENTION_PRODUCT_SHIFT 10

__device__ static __forceinline__ float
official_attention_group(float running, const uint16_t *query_words,
			 uint32_t query_stride, const uint16_t *key_words,
			 uint32_t key_stride)
{
	int32_t query_magnitudes[ATTENTION_PRODUCT_GROUP],
		query_exponents[ATTENTION_PRODUCT_GROUP],
		key_magnitudes[ATTENTION_PRODUCT_GROUP],
		key_exponents[ATTENTION_PRODUCT_GROUP];
	bool finite = true;
#pragma unroll
	for (uint32_t product_offset = 0;
	     product_offset < ATTENTION_PRODUCT_GROUP; product_offset++) {
		class_80_decode(query_words[product_offset * query_stride],
				true, &query_magnitudes[product_offset],
				&query_exponents[product_offset], &finite);
		class_80_decode(key_words[product_offset * key_stride], true,
				&key_magnitudes[product_offset],
				&key_exponents[product_offset], &finite);
	}
	return finite ? class_80_group(running, query_magnitudes,
				       query_exponents, key_magnitudes,
				       key_exponents, ATTENTION_PRODUCT_SHIFT) :
			__uint_as_float(CLASS_80_NOT_A_NUMBER_BITS);
}

__device__ static __forceinline__ uint16_t
official_attention_bfloat16(uint16_t half_word, uint32_t native_bf16)
{
	if (native_bf16 != 0)
		return half_word;
	return __bfloat16_as_ushort(
		__float2bfloat16_rn(__half2float(__ushort_as_half(half_word))));
}

/* The first serial token uses the source prefill selection: one window key.
 * Subsequent cached calls retain the full128-slot window before pool entries. */
__device__ static __forceinline__ uint32_t
official_attention_window_keys(const struct step_row_record *step_row)
{
	return step_row->position == 0 ? 1u : MODEL_WINDOW;
}

/* Physical ring positions use MODEL_RING_SLOTS. Source logical keys
 * are the 128 window positions, followed by selected compressed entries. */
__device__ static __forceinline__ const uint16_t *
official_attention_key(struct attention_core_arguments_record source,
		       const unsigned char *arena,
		       const struct step_row_record *step_row, uint32_t slot)
{
	if (slot < official_attention_window_keys(step_row)) {
		const int32_t position =
			step_row->position -
			(int32_t)(official_attention_window_keys(step_row) -
				  1u) +
			(int32_t)slot;
		return position < 0 ?
			       NULL :
			       (const uint16_t
					*)(arena + source.ring_offset +
					   (uint64_t)step_row->session_slot *
						   source.ring_session_stride) +
				       ((uint32_t)position % MODEL_RING_SLOTS) *
					       MODEL_HEAD_DIMS;
	}
	if (source.pool_ratio == 0)
		return NULL;
	const uint32_t row = (uint32_t)blockIdx.x;
	const int32_t pick =
		((const int32_t *)(arena + source.selection_offset))
			[(uint64_t)row * MODEL_SELECTED + slot -
			 official_attention_window_keys(step_row)];
	const uint32_t visible =
		((uint32_t)step_row->position + 1u) / source.pool_ratio;
	return pick < 0 || (uint32_t)pick >= visible ?
		       NULL :
		       (const uint16_t *)(arena + source.pool_offset +
					  (uint64_t)step_row->session_slot *
						  source.pool_session_stride) +
			       (uint64_t)(uint32_t)pick * MODEL_HEAD_DIMS;
}

extern "C" __global__ void
official_attention(struct official_attention_arguments arguments,
		   unsigned char *arena)
{
	const struct attention_core_arguments_record source = arguments.source;
	const uint32_t row = blockIdx.x, head = blockIdx.y,
		       thread = threadIdx.x;
	const struct step_block *step =
		(const struct step_block *)(arena + source.step_offset);
	if (row >= step->row_count || head >= MODEL_HEADS_VALUE)
		return;
	const struct step_row_record *step_row =
		(const struct step_row_record *)(step + 1) + row;
	const uint32_t visible =
		source.pool_ratio == 0 ?
			0u :
			((uint32_t)step_row->position + 1u) / source.pool_ratio;
	const uint32_t topk =
		official_attention_window_keys(step_row) +
		(visible < MODEL_SELECTED ? visible : MODEL_SELECTED);
	extern __shared__ __align__(
		ATTENTION_STAGE_ALIGNMENT) unsigned char memory[];
	uint16_t *const staged = (uint16_t *)memory;
	uint16_t *const low_prob =
		staged + ATTENTION_TILE_KEYS * MODEL_HEAD_DIMS;
	uint16_t *const query = low_prob + ATTENTION_TILE_KEYS;
	float *const score = (float *)(query + MODEL_HEAD_DIMS);
	float *const prob = score + ATTENTION_TILE_KEYS;
	float *const state = prob + ATTENTION_TILE_KEYS;
	/* The final BF16 quotient reuses the completed key stage. */
	uint16_t *const pure = staged;
	uint16_t *const output = (uint16_t *)(arena + source.output_offset);
	for (uint32_t column = thread; column < MODEL_HEAD_DIMS;
	     column += OFFICIAL_ATTENTION_THREADS)
		query[column] = official_attention_bfloat16(
			((const uint16_t *)(arena + source.queries_offset))
				[((uint64_t)row * MODEL_HEADS_VALUE + head) *
					 MODEL_HEAD_DIMS +
				 column],
			arguments.operand_types &
				OFFICIAL_ATTENTION_QUERY_BFLOAT16);
	float values[ATTENTION_THREAD_VALUES] = {};
	if (thread == 0) {
		state[0] = ATTENTION_CORE_SCORE_FLOOR;
		state[1] = 0.0f;
		state[ATTENTION_STATE_RESCALE] = 1.0f;
	}
	__syncthreads();
	for (uint32_t first = 0; first < topk; first += ATTENTION_TILE_KEYS) {
		for (uint32_t item = thread;
		     item < ATTENTION_TILE_KEYS * MODEL_HEAD_DIMS;
		     item += OFFICIAL_ATTENTION_THREADS) {
			const uint32_t slot = first + item / MODEL_HEAD_DIMS;
			const uint16_t *key_source =
				slot < topk ?
					official_attention_key(source, arena,
							       step_row, slot) :
					NULL;
			staged[item] =
				key_source == NULL ?
					0 :
					official_attention_bfloat16(
						key_source[item %
							   MODEL_HEAD_DIMS],
						arguments.operand_types &
							(slot < official_attention_window_keys(
									step_row) ?
								 OFFICIAL_ATTENTION_RING_BFLOAT16 :
								 OFFICIAL_ATTENTION_POOL_BFLOAT16));
		}
		__syncthreads();
		if (thread < ATTENTION_TILE_KEYS) {
			const uint32_t slot = first + thread;
			const bool valid = slot < topk &&
					   official_attention_key(source, arena,
								  step_row,
								  slot) != NULL;
			float sum = 0.0f;
			if (valid)
				for (uint32_t product_offset = 0;
				     product_offset < MODEL_HEAD_DIMS;
				     product_offset += ATTENTION_PRODUCT_GROUP)
					sum = official_attention_group(
						sum, query + product_offset, 1,
						staged +
							thread *
								MODEL_HEAD_DIMS +
							product_offset,
						1);
			score[thread] =
				valid ? __fmul_rn(sum, ATTENTION_CORE_SCALE) :
					-CUDART_INF_F;
		}
		__syncthreads();
		if (thread == 0) {
			float maximum = state[0];
			for (uint32_t slot = 0; slot < ATTENTION_TILE_KEYS;
			     slot++)
				maximum = fmaxf(maximum, score[slot]);
			state[ATTENTION_STATE_RESCALE] =
				expf(state[0] - maximum);
			state[0] = maximum;
		}
		__syncthreads();
		if (thread < ATTENTION_TILE_KEYS) {
			prob[thread] = expf(score[thread] - state[0]);
			low_prob[thread] = __bfloat16_as_ushort(
				__float2bfloat16_rn(prob[thread]));
		}
		__syncthreads();
		if (thread == 0) {
			float owners[ATTENTION_PAIR_WIDTH]
				    [ATTENTION_OWNER_PARTS] = {};
#pragma unroll
			for (uint32_t half = 0; half < ATTENTION_PAIR_WIDTH;
			     half++)
#pragma unroll
				for (uint32_t part = 0;
				     part < ATTENTION_OWNER_PARTS; part++)
#pragma unroll
					for (uint32_t owner_value = 0;
					     owner_value <
					     ATTENTION_OWNER_VALUES;
					     owner_value++)
						owners[half][part] = __fadd_rn(
							owners[half][part],
							prob[half * ATTENTION_HALF_TILE_KEYS +
							     part * ATTENTION_PAIR_WIDTH +
							     (owner_value %
							      ATTENTION_OWNER_PARTS) *
								     ATTENTION_OWNER_VALUES +
							     owner_value /
								     ATTENTION_OWNER_PARTS]);
			const float combined_first_owner = __fadd_rn(
					    owners[0][0], owners[1][0]),
				    combined_next_owner = __fadd_rn(
					    owners[0][1], owners[1][1]);
			const float combined_second_owner = __fadd_rn(
					    owners[0][ATTENTION_SECOND_OWNER],
					    owners[1][ATTENTION_SECOND_OWNER]),
				    combined_last_owner = __fadd_rn(
					    owners[0][ATTENTION_LAST_OWNER],
					    owners[1][ATTENTION_LAST_OWNER]);
			const float sum =
				__fadd_rn(__fadd_rn(combined_first_owner,
						    combined_second_owner),
					  __fadd_rn(combined_next_owner,
						    combined_last_owner));
			state[1] = __fmaf_rn(
				state[1], state[ATTENTION_STATE_RESCALE], sum);
		}
		__syncthreads();
#pragma unroll
		for (uint32_t item = 0; item < ATTENTION_THREAD_VALUES;
		     item++) {
			const uint32_t column =
				thread + item * OFFICIAL_ATTENTION_THREADS;
			values[item] = __fmul_rn(
				values[item], state[ATTENTION_STATE_RESCALE]);
			for (uint32_t product_offset = 0;
			     product_offset < ATTENTION_TILE_KEYS;
			     product_offset += ATTENTION_PRODUCT_GROUP)
				values[item] = official_attention_group(
					values[item], low_prob + product_offset,
					1,
					staged +
						product_offset *
							MODEL_HEAD_DIMS +
						column,
					MODEL_HEAD_DIMS);
		}
		__syncthreads();
	}
	if (thread == 0) {
		const float denominator = __fadd_rn(
			state[1],
			expf(((const float *)(arena +
					      source.sinks_offset))[head] -
			     state[0]));
		state[1] = denominator;
	}
	__syncthreads();
#pragma unroll
	for (uint32_t item = 0; item < ATTENTION_THREAD_VALUES; item++) {
		const uint64_t place =
			thread + item * OFFICIAL_ATTENTION_THREADS;
		const float result = __fdiv_rn(values[item], state[1]);
		pure[place] = __bfloat16_as_ushort(__float2bfloat16_rn(result));
	}
	__syncthreads();
	if (thread < MODEL_HEAD_DIMS / ATTENTION_PAIR_WIDTH) {
		const uint32_t column = thread * ATTENTION_PAIR_WIDTH;
		float real = __uint_as_float((uint32_t)pure[column]
					     << ATTENTION_BFLOAT16_SHIFT);
		float imaginary = __uint_as_float((uint32_t)pure[column + 1u]
						  << ATTENTION_BFLOAT16_SHIFT);
		if (column >= MODEL_FIRST_ROTATED_DIM) {
			const float *rotation =
				(const float
					 *)(arena + source.rotation_offset +
					    (uint64_t)row *
						    MODEL_ROTATION_ROW_BYTES);
			const uint32_t pair =
				(column - MODEL_FIRST_ROTATED_DIM) /
				ATTENTION_PAIR_WIDTH;
			const float
				cosine = rotation[pair * ATTENTION_PAIR_WIDTH],
				sine = rotation[pair * ATTENTION_PAIR_WIDTH +
						1u];
			const float next_real = __fmaf_rn(
				real, cosine, __fmul_rn(imaginary, sine));
			imaginary = __fmaf_rn(imaginary, cosine,
					      -__fmul_rn(real, sine));
			real = next_real;
		}
		output[((uint64_t)row * MODEL_HEADS_VALUE + head) *
			       MODEL_HEAD_DIMS +
		       column] =
			__bfloat16_as_ushort(__float2bfloat16_rn(real));
		output[((uint64_t)row * MODEL_HEADS_VALUE + head) *
			       MODEL_HEAD_DIMS +
		       column + 1u] =
			__bfloat16_as_ushort(__float2bfloat16_rn(imaginary));
	}
}
