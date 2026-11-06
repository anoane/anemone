/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Serial original source assignments: BF16 windows and index operands; finite
 * group16/E4 compressed values are carried losslessly in FP16. */
#include "kernel_activation_quantize.cuh"
#include "anemone/official_state_records.h"
#include <cuda_fp8.h>

#define OFFICIAL_VECTOR 4u
#define OFFICIAL_EVERY_LANE 0xFFFFFFFFu
#define OFFICIAL_NORM_MAXIMUM 128u
#define OFFICIAL_FP4_MAXIMUM 6.0f
#define OFFICIAL_FP4_E4_FLOOR (6.0f * 0x1p-9f)
#define OFFICIAL_FP4_E8_FLOOR (6.0f * 0x1p-126f)
#define OFFICIAL_FP4_FRACTION_BITS 23u
#define OFFICIAL_FP4_FRACTION_MASK 0x7FFFFFu
#define OFFICIAL_FP4_EXPONENT_MASK 255u
#define OFFICIAL_HALF_WARP 16u
#define OFFICIAL_INDEX_TILE_HEADS 16u
#define OFFICIAL_INDEX_HALF_TILE 8u
#define OFFICIAL_FP8_UNCARRY 0x1p-100f

#define OFFICIAL_VECTOR_THIRD 2u
#define OFFICIAL_VECTOR_LAST 3u
#define OFFICIAL_FP4_KEY_PERMUTATION 1u
#define OFFICIAL_FP4_QUERY_PERMUTATION 2u
#define OFFICIAL_FP4_ZERO_MIDPOINT 0.25f
#define OFFICIAL_FP4_HALF_MIDPOINT 0.75f
#define OFFICIAL_FP4_ONE_MIDPOINT 1.25f
#define OFFICIAL_FP4_ONE_HALF_MIDPOINT 1.75f
#define OFFICIAL_FP4_TWO_MIDPOINT 2.5f
#define OFFICIAL_FP4_THREE_MIDPOINT 3.5f
#define OFFICIAL_FP4_FOUR_MIDPOINT 5.0f
#define OFFICIAL_FP4_HALF_VALUE 0.5f
#define OFFICIAL_FP4_ONE_HALF_VALUE 1.5f
#define OFFICIAL_FP4_TWO_VALUE 2.0f
#define OFFICIAL_FP4_THREE_VALUE 3.0f
#define OFFICIAL_FP4_FOUR_VALUE 4.0f

__device__ static __forceinline__ void
official_norm_row(const uint16_t *input, const uint16_t *weight,
		  uint16_t *output, uint32_t width, float *sums)
{
	const uint32_t thread = threadIdx.x;
	const uint32_t owners = width / OFFICIAL_VECTOR;
	float chains[OFFICIAL_VECTOR] = {};
	if (thread < owners) {
		for (uint32_t first = thread * OFFICIAL_VECTOR; first < width;
		     first += owners * OFFICIAL_VECTOR) {
			for (uint32_t slot = 0; slot < OFFICIAL_VECTOR;
			     slot++) {
				const float value = linear_bfloat16_value(
					input[first + slot]);
				chains[slot] = __fadd_rn(
					chains[slot], __fmul_rn(value, value));
			}
		}
		sums[thread] =
			__fadd_rn(__fadd_rn(__fadd_rn(chains[0], chains[1]),
					    chains[OFFICIAL_VECTOR_THIRD]),
				  chains[OFFICIAL_VECTOR_LAST]);
	}
	__syncthreads();
	for (uint32_t distance = owners / MODEL_PAIR_VALUES;
	     distance >= THREADS_PER_WARP; distance /= MODEL_PAIR_VALUES) {
		if (thread < distance)
			sums[thread] = __fadd_rn(sums[thread],
						 sums[thread + distance]);
		__syncthreads();
	}
	if (thread < THREADS_PER_WARP) {
		float sum = sums[thread];
		for (uint32_t distance = THREADS_PER_WARP / MODEL_PAIR_VALUES;
		     distance != 0; distance /= MODEL_PAIR_VALUES)
			sum = __fadd_rn(sum,
					__shfl_down_sync(OFFICIAL_EVERY_LANE,
							 sum, distance));
		if (thread == 0)
			sums[0] = rsqrtf(
				__fadd_rn(__fmul_rn(sum, 1.0f / (float)width),
					  MODEL_NORM_EPSILON));
	}
	__syncthreads();
	const float inverse = sums[0];
	for (uint32_t column = thread; column < width; column += blockDim.x)
		output[column] = (uint16_t)linear_bfloat16_bits(__fmul_rn(
			linear_bfloat16_value(weight[column]),
			__fmul_rn(linear_bfloat16_value(input[column]),
				  inverse)));
	__syncthreads();
}

__device__ static __forceinline__ void
official_rotate_row(const uint16_t *input, uint16_t *output, uint32_t width,
		    const unsigned char *arena, uint64_t rotation_offset,
		    uint32_t use)
{
	const uint32_t first_rotated_pair =
		(width - MODEL_ROTATED_DIMS) / MODEL_PAIR_VALUES;
	for (uint32_t pair = threadIdx.x; pair < width / MODEL_PAIR_VALUES;
	     pair += blockDim.x) {
		const uint32_t first = pair * MODEL_PAIR_VALUES;
		const uint16_t real_bits = input[first],
			       imaginary_bits = input[first + 1u];
		if (pair < first_rotated_pair) {
			output[first] = real_bits;
			output[first + 1u] = imaginary_bits;
		} else {
			const float2 angle =
				((const float2 *)(arena + rotation_offset))
					[use * MODEL_FREQUENCIES + pair -
					 first_rotated_pair];
			const float real = linear_bfloat16_value(real_bits),
				    imaginary = linear_bfloat16_value(
					    imaginary_bits);
			output[first] = (uint16_t)linear_bfloat16_bits(fmaf(
				real, angle.x, -__fmul_rn(imaginary, angle.y)));
			output[first + 1u] = (uint16_t)linear_bfloat16_bits(
				fmaf(imaginary, angle.x,
				     __fmul_rn(real, angle.y)));
		}
	}
	__syncthreads();
}

__device__ static __forceinline__ void
official_fp4_row(const uint16_t *input, uint16_t *output, uint32_t width,
		 uint32_t group_size, uint32_t permutation, uint32_t head)
{
	for (uint32_t column = threadIdx.x; column < width;
	     column += blockDim.x) {
		const float value = linear_bfloat16_value(input[column]);
		float maximum = fabsf(value);
		for (uint32_t distance = group_size / MODEL_PAIR_VALUES;
		     distance != 0; distance /= MODEL_PAIR_VALUES)
			maximum = fmaxf(maximum,
					__shfl_xor_sync(OFFICIAL_EVERY_LANE,
							maximum, (int)distance,
							(int)group_size));
		float scale;
		if (group_size == OFFICIAL_HALF_WARP) {
			const float ratio =
				__fdiv_rn(fmaxf(maximum, OFFICIAL_FP4_E4_FLOOR),
					  OFFICIAL_FP4_MAXIMUM);
			const uint8_t code = __nv_cvt_float_to_fp8(
				ratio, __NV_SATFINITE, __NV_E4M3);
			scale = __half2float(__half(
				__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
		} else {
			const float ratio =
				__fmul_rn(fmaxf(maximum, OFFICIAL_FP4_E8_FLOOR),
					  1.0f / OFFICIAL_FP4_MAXIMUM);
			const uint32_t bits = __float_as_uint(ratio);
			const uint32_t exponent =
				((bits >> OFFICIAL_FP4_FRACTION_BITS) &
				 OFFICIAL_FP4_EXPONENT_MASK) +
				((bits & OFFICIAL_FP4_FRACTION_MASK) != 0);
			scale = __uint_as_float(exponent
						<< OFFICIAL_FP4_FRACTION_BITS);
		}
		const float scaled = fabsf(__fdiv_rn(value, scale));
		const float magnitude =
			scaled <= OFFICIAL_FP4_ZERO_MIDPOINT ?
				0.0f :
			scaled < OFFICIAL_FP4_HALF_MIDPOINT ?
				OFFICIAL_FP4_HALF_VALUE :
			scaled <= OFFICIAL_FP4_ONE_MIDPOINT ?
				1.0f :
			scaled < OFFICIAL_FP4_ONE_HALF_MIDPOINT ?
				OFFICIAL_FP4_ONE_HALF_VALUE :
			scaled <= OFFICIAL_FP4_TWO_MIDPOINT ?
				OFFICIAL_FP4_TWO_VALUE :
			scaled < OFFICIAL_FP4_THREE_MIDPOINT ?
				OFFICIAL_FP4_THREE_VALUE :
			scaled <= OFFICIAL_FP4_FOUR_MIDPOINT ?
				OFFICIAL_FP4_FOUR_VALUE :
				OFFICIAL_FP4_MAXIMUM;
		const uint16_t rounded_bits = (uint16_t)linear_bfloat16_bits(
			__fmul_rn(copysignf(magnitude, value), scale));
		const float rounded = linear_bfloat16_value(rounded_bits);
		uint32_t destination = column;
		if (permutation == OFFICIAL_FP4_KEY_PERMUTATION) {
			const uint32_t pair =
				column / MODEL_PAIR_VALUES * MODEL_PAIR_VALUES;
			destination = (pair % OFFICIAL_INDEX_HALF_TILE /
					       MODEL_PAIR_VALUES *
					       OFFICIAL_INDEX_TILE_HEADS +
				       pair / OFFICIAL_INDEX_TILE_HEADS *
					       MODEL_PAIR_VALUES +
				       pair % OFFICIAL_INDEX_TILE_HEADS /
					       OFFICIAL_INDEX_HALF_TILE) *
					      MODEL_PAIR_VALUES +
				      column % MODEL_PAIR_VALUES;
		} else if (permutation == OFFICIAL_FP4_QUERY_PERMUTATION) {
			const uint32_t paired_column = column /
						       MODEL_PAIR_VALUES *
						       MODEL_PAIR_VALUES,
				       in_step = paired_column %
						 OFFICIAL_INDEX_TILE_HEADS,
				       in_tile =
					       head % OFFICIAL_INDEX_TILE_HEADS;
			const uint32_t lane =
				in_tile % OFFICIAL_INDEX_HALF_TILE *
					OFFICIAL_VECTOR +
				in_step % OFFICIAL_INDEX_HALF_TILE /
					MODEL_PAIR_VALUES;
			const uint32_t word =
				((head / OFFICIAL_INDEX_TILE_HEADS *
					  OFFICIAL_INDEX_HALF_TILE +
				  paired_column / OFFICIAL_INDEX_TILE_HEADS) *
					 THREADS_PER_WARP +
				 lane) * OFFICIAL_VECTOR +
				in_step / OFFICIAL_INDEX_HALF_TILE *
					MODEL_PAIR_VALUES +
				in_tile / OFFICIAL_INDEX_HALF_TILE;
			destination = word * MODEL_PAIR_VALUES +
				      column % MODEL_PAIR_VALUES;
		}
		output[destination] =
			permutation == 0 ?
				__half_as_ushort(__float2half_rn(rounded)) :
				rounded_bits;
	}
}

extern "C" __global__ void
official_window_state(struct attention_prepare_arguments arguments,
		      unsigned char *arena)
{
	__shared__ uint16_t normed[MODEL_HEAD_DIMS], rotated[MODEL_HEAD_DIMS];
	__shared__ float sums[OFFICIAL_NORM_MAXIMUM];
	const struct step_block *const block =
		(const struct step_block *)(arena + arguments.step_offset);
	if (block->row_count == 0)
		return;
	const struct step_row_record *const step_row =
		(const struct step_row_record *)(arena + arguments.step_offset +
						 STEP_BLOCK_BYTES_VALUE);
	official_norm_row(
		(const uint16_t *)(arena + arguments.window_input_offset),
		(const uint16_t *)(arena + arguments.window_norm_offset),
		normed, MODEL_HEAD_DIMS, sums);
	official_rotate_row(normed, rotated, MODEL_HEAD_DIMS, arena,
			    arguments.rotation_offset, arguments.rotation_use);
	uint16_t *const output =
		(uint16_t *)(arena + arguments.ring_offset +
			     (uint64_t)step_row->session_slot *
				     arguments.ring_session_stride) +
		(uint32_t)step_row->position % MODEL_RING_SLOTS *
			MODEL_HEAD_DIMS;
	for (uint32_t column = threadIdx.x; column < MODEL_HEAD_DIMS;
	     column += blockDim.x) {
		const float value = linear_bfloat16_value(rotated[column]);
		float largest = fabsf(value);
		for (uint32_t distance = THREADS_PER_WARP / MODEL_PAIR_VALUES;
		     distance != 0; distance /= MODEL_PAIR_VALUES)
			largest = fmaxf(largest,
					__shfl_xor_sync(OFFICIAL_EVERY_LANE,
							largest, distance));
		const uint32_t exponent = quantize_exponent_code(largest);
		uint32_t operand = 0;
		(void)quantize_nearest_e4m3(
			value * quantize_inverse_scale(exponent), exponent,
			&operand);
		const float decoded =
			linear_bfloat16_value(operand) * OFFICIAL_FP8_UNCARRY;
		output[column] = (uint16_t)linear_bfloat16_bits(decoded);
	}
}

extern "C" __global__ void
official_compressed_state(struct attention_compressor_arguments arguments,
			  unsigned char *arena)
{
	__shared__ uint16_t pooled[MODEL_HEAD_DIMS], normed[MODEL_HEAD_DIMS],
		rotated[MODEL_HEAD_DIMS];
	__shared__ float sums[OFFICIAL_NORM_MAXIMUM];
	const struct step_block *const block =
		(const struct step_block *)(arena + arguments.step_offset);
	if (block->row_count == 0)
		return;
	const struct step_row_record *const step_row =
		(const struct step_row_record *)(arena + arguments.step_offset +
						 STEP_BLOCK_BYTES_VALUE);
	const uint32_t position = (uint32_t)step_row->position;
	if (arguments.ratio == MODEL_POOLED_RATIO) {
		float *const slots =
			(float *)(arena + arguments.slots_offset +
				  (uint64_t)step_row->session_slot *
					  arguments.slots_session_stride);
		for (uint32_t column = threadIdx.x; column < MODEL_HEAD_DIMS;
		     column += blockDim.x) {
			const float current_kv =
				((const float *)(arena +
						 arguments.kv_offset))[column];
			const float current_score = ((
				const float *)(arena +
					       arguments.score_offset))[column];
			if ((position + 1u) % arguments.ratio == 0) {
				const float *const previous =
					slots + ((position - 1u) %
						 ATTENTION_COMPRESSOR_SLOTS) *
							ATTENTION_SLOT_VECTORS *
							MODEL_HEAD_DIMS;
				const float first_kv = previous[column],
					    first_score =
						    previous[MODEL_HEAD_DIMS +
							     column];
				const float maximum =
					fmaxf(first_score, current_score);
				const float exp_first =
					expf(__fsub_rn(first_score, maximum));
				const float exp_current =
					expf(__fsub_rn(current_score, maximum));
				const float total =
					__fadd_rn(exp_first, exp_current);
				const float first = __fmul_rn(
					first_kv, __fdiv_rn(exp_first, total));
				const float second = __fmul_rn(
					current_kv,
					__fdiv_rn(exp_current, total));
				pooled[column] = (uint16_t)linear_bfloat16_bits(
					__fadd_rn(__fadd_rn(0.0f, first),
						  second));
			}
			if ((position + 1u) % arguments.ratio != 0) {
				const uint32_t slot =
					position % ATTENTION_COMPRESSOR_SLOTS;
				slots[(slot * ATTENTION_SLOT_VECTORS) *
					      MODEL_HEAD_DIMS +
				      column] = current_kv;
				slots[(slot * ATTENTION_SLOT_VECTORS + 1u) *
					      MODEL_HEAD_DIMS +
				      column] = current_score;
			}
		}
	} else {
		for (uint32_t column = threadIdx.x; column < MODEL_HEAD_DIMS;
		     column += blockDim.x)
			pooled[column] = ((
				const uint16_t *)(arena +
						  arguments.kv_offset))[column];
	}
	if ((position + 1u) % arguments.ratio != 0)
		return;
	__syncthreads();
	official_norm_row(pooled,
			  (const uint16_t *)(arena + arguments.norm_offset),
			  normed, MODEL_HEAD_DIMS, sums);
	for (uint32_t column = threadIdx.x; column < MODEL_HEAD_DIMS;
	     column += blockDim.x)
		((uint16_t *)(arena + arguments.latent_offset))[column] =
			normed[column];
	const uint32_t use = arguments.ratio == 1u ?
				     ATTENTION_ROTATION_POOLED :
				     ATTENTION_ROTATION_GROUP_START;
	official_rotate_row(normed, rotated, MODEL_HEAD_DIMS, arena,
			    arguments.rotation_offset, use);
	uint16_t *const output =
		(uint16_t *)(arena + arguments.pool_offset +
			     (uint64_t)step_row->session_slot *
				     arguments.pool_session_stride) +
		position / arguments.ratio * MODEL_HEAD_DIMS;
	official_fp4_row(rotated, output, MODEL_HEAD_DIMS, OFFICIAL_HALF_WARP,
			 0, 0);
}

extern "C" __global__ void
official_key_state(struct attention_key_arguments arguments,
		   unsigned char *arena)
{
	__shared__ uint16_t normed[MODEL_INDEX_HEAD_DIMS],
		rotated[MODEL_INDEX_HEAD_DIMS];
	__shared__ float sums[OFFICIAL_NORM_MAXIMUM];
	const struct step_block *const block =
		(const struct step_block *)(arena + arguments.step_offset);
	if (block->row_count == 0)
		return;
	const struct step_row_record *const step_row =
		(const struct step_row_record *)(arena + arguments.step_offset +
						 STEP_BLOCK_BYTES_VALUE);
	const uint32_t position = (uint32_t)step_row->position;
	if ((position + 1u) % arguments.ratio != 0)
		return;
	official_norm_row(
		(const uint16_t *)(arena + arguments.key_input_offset),
		(const uint16_t *)(arena + arguments.norm_offset), normed,
		MODEL_INDEX_HEAD_DIMS, sums);
	const uint32_t use = arguments.ratio == 1u ?
				     ATTENTION_ROTATION_POOLED :
				     ATTENTION_ROTATION_GROUP_START;
	official_rotate_row(normed, rotated, MODEL_INDEX_HEAD_DIMS, arena,
			    arguments.rotation_offset, use);
	uint16_t *const output =
		(uint16_t *)(arena + arguments.keys_offset +
			     (uint64_t)step_row->session_slot *
				     arguments.keys_session_stride) +
		position / arguments.ratio * MODEL_INDEX_HEAD_DIMS;
	official_fp4_row(rotated, output, MODEL_INDEX_HEAD_DIMS,
			 THREADS_PER_WARP, OFFICIAL_FP4_KEY_PERMUTATION, 0);
}

extern "C" __global__ void
official_index_queries(struct attention_queries_arguments arguments,
		       unsigned char *arena)
{
	__shared__ uint16_t rotated[MODEL_INDEX_HEAD_DIMS];
	const uint32_t head = blockIdx.x;
	official_rotate_row(
		(const uint16_t *)(arena + arguments.index_heads_offset) +
			head * MODEL_INDEX_HEAD_DIMS,
		rotated, MODEL_INDEX_HEAD_DIMS, arena,
		arguments.rotation_offset, arguments.rotation_use);
	official_fp4_row(rotated,
			 (uint16_t *)(arena + arguments.index_queries_offset),
			 MODEL_INDEX_HEAD_DIMS, THREADS_PER_WARP,
			 OFFICIAL_FP4_QUERY_PERMUTATION, head);
}
