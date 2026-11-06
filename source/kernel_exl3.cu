/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* EXL3 mul1 reconstruction; kept separate from every native module. */
#include <cuda_fp16.h>
#include "anemone/exl3_records.h"

#define MMA_PAIR 2u
#define FRAGMENT_LANE_GROUP 4u
#define FRAGMENT_VALUES_PER_LANE 8u
#define FRAGMENT_HALF_TILE 8u
#define FRAGMENT_HIGH_ROW_BIT 2u
#define FRAGMENT_HIGH_COLUMN_BIT 4u
#define EXL3_FULL_WARP_MASK 0xffffffffu
#define EXL3_BYTE_UNITS 0x01010101u

__device__ static __forceinline__ uint32_t exl3_ring_position(uint32_t input,
							      uint32_t output)
{
	const uint32_t lane =
		(output % FRAGMENT_HALF_TILE) * FRAGMENT_LANE_GROUP +
		(input % FRAGMENT_HALF_TILE) / MMA_PAIR;
	const uint32_t element =
		input % MMA_PAIR +
		(input / FRAGMENT_HALF_TILE) * FRAGMENT_HIGH_ROW_BIT +
		(output / FRAGMENT_HALF_TILE) * FRAGMENT_HIGH_COLUMN_BIT;
	return lane * FRAGMENT_VALUES_PER_LANE + element;
}

__device__ static __forceinline__ half exl3_decode(const uint32_t *tile,
						   uint32_t position,
						   uint32_t bits)
{
	const uint32_t ring_bits = EXL3_TILE_VALUES * bits;
	const uint32_t begin =
		((position + 1u) * bits + ring_bits - EXL3_STATE_BITS) %
		ring_bits;
	const uint32_t word = begin / EXL3_WORD_BITS;
	const uint32_t shift = begin % EXL3_WORD_BITS;
	uint32_t window = tile[word] << shift;
	if (shift != 0) {
		window |= tile[(word + 1u) % (ring_bits / EXL3_WORD_BITS)] >>
			  (EXL3_WORD_BITS - shift);
	}
	const uint32_t product =
		(window >> EXL3_STATE_BITS) * EXL3_MUL1_MULTIPLIER;
	const uint32_t byte_sum = __dp4a(product, EXL3_BYTE_UNITS, 0u);
	return __hfma(__ushort_as_half((uint16_t)(EXL3_MUL1_BASE + byte_sum)),
		      __ushort_as_half(EXL3_MUL1_SCALE),
		      __ushort_as_half(EXL3_MUL1_BIAS));
}

/* Every MMA lane owns eight consecutive ring symbols. Their preceding history
 * and all eight advances fit in two cyclic words for K2/K3. Decode those words
 * once per tile; no full matrix or floating-point intermediate is introduced. */
__device__ static __forceinline__ half exl3_state_value(uint32_t state)
{
	const uint32_t product = state * EXL3_MUL1_MULTIPLIER;
	const uint32_t byte_sum = __dp4a(product, EXL3_BYTE_UNITS, 0u);
	return __hfma(__ushort_as_half((uint16_t)(EXL3_MUL1_BASE + byte_sum)),
		      __ushort_as_half(EXL3_MUL1_SCALE),
		      __ushort_as_half(EXL3_MUL1_BIAS));
}

extern "C" __global__ void
exl3_reconstruct(struct exl3_reconstruct_record record, unsigned char *arena)
{
	const uint64_t element =
		(uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t count = (uint64_t)record.input_size * record.output_size;
	if (element >= count) {
		return;
	}
	const uint32_t input = (uint32_t)(element / record.output_size);
	const uint32_t output = (uint32_t)(element % record.output_size);
	const uint64_t tile_index = (uint64_t)(input / EXL3_TILE) *
					    (record.output_size / EXL3_TILE) +
				    output / EXL3_TILE;
	const uint32_t *const tile =
		(const uint32_t *)(arena + record.matrix_offset) +
		tile_index * EXL3_TILE_VALUES * record.bits_per_weight /
			EXL3_WORD_BITS;
	((half *)(arena + record.output_offset))[element] = exl3_decode(
		tile, exl3_ring_position(input % EXL3_TILE, output % EXL3_TILE),
		record.bits_per_weight);
}

#include "anemone/linear_records.h"
#include <cuda_bf16.h>

#define EXPERT_THREADS (EXL3_PRODUCT_SLICES * EXL3_WARP)
#define MMA_CHAINS 4u
#define MMA_INPUT_HALF 8u
#define MMA_LANE_GROUP 4u
#define MMA_OUTPUT_HALF 8u
#define MMA_COLUMNS 8u
#define MMA_COLUMNS_PER_LANE 2u
#define MMA_CHAIN_LOW_EVEN 0u
#define MMA_CHAIN_LOW_ODD 1u
#define MMA_CHAIN_HIGH_EVEN 2u
#define MMA_CHAIN_HIGH_ODD 3u

/* The same Sylvester butterfly for independent input and output passes. Every
 * thread reaches every barrier; job columns run serially with disjoint writes. */
__device__ static void exl3_hadamard(float *values)
{
	const uint32_t column = threadIdx.x;
	float value = values[column];
	for (uint32_t stride = 1; stride < EXL3_TRANSFORM;
	     stride *= EXL3_RADIX) {
		float other;
		if (stride < EXL3_WARP) {
			other = __shfl_xor_sync(EXL3_FULL_WARP_MASK, value,
						stride);
		} else {
			values[column] = value;
			__syncthreads();
			other = values[column ^ stride];
			__syncthreads();
		}
		value = (column & stride) == 0 ? value + other : other - value;
	}
	values[column] = value;
	__syncthreads();
}

__device__ static __forceinline__ uint64_t
exl3_codes_bytes(struct exl3_expert_record record)
{
	return (uint64_t)record.input_size * record.inner_size /
	       EXL3_BITS_PER_BYTE * record.bits_per_weight;
}

extern "C" __global__ void exl3_input(struct exl3_expert_record record,
				      unsigned char *arena)
{
	const struct expert_job_record *const job =
		(const struct expert_job_record *)(arena + record.jobs_offset) +
		blockIdx.y;
	if (job->row_count == 0) {
		return;
	}
	__shared__ float values[EXL3_TRANSFORM];
	const uint32_t channel = blockIdx.x * EXL3_TRANSFORM + threadIdx.x;
	const uint64_t matrix = blockIdx.z == 0 ? job->gate_offset :
						  job->up_offset;
	const half *const scales =
		(const half *)(arena + matrix + exl3_codes_bytes(record));
	const __nv_bfloat16 *const input =
		(const __nv_bfloat16 *)(arena + record.input_offset);
	half *const operands = (half *)(arena + record.operands_offset);
	for (uint32_t column = 0; column < job->row_count; column++) {
		const uint32_t pair = job->pair_index[column];
		const uint32_t row = pair / record.pairs_per_row;
		values[threadIdx.x] =
			__bfloat162float(
				input[(uint64_t)row * record.input_size +
				      channel]) *
			__half2float(scales[channel]);
		__syncthreads();
		exl3_hadamard(values);
		operands[((uint64_t)blockIdx.z * record.pair_count + pair) *
				 record.input_size +
			 channel] = __float2half_rn(values[threadIdx.x] *
						    EXL3_NORMALIZATION);
		__syncthreads();
	}
}

__device__ static __forceinline__ uint32_t exl3_half_pair(half low, half high)
{
	return (uint32_t)__half_as_ushort(low) |
	       ((uint32_t)__half_as_ushort(high) << EXL3_STATE_BITS);
}

/* A is 16 output channels of a decoded tile, B is up to eight job rows.
 * Four contiguous K slices accumulate independently, then reduce in fixed order.
 * Decoded weights exist only in registers; no expanded expert is materialized. */
template <uint32_t bits, bool down, uint32_t slices>
__device__ static void exl3_product(struct exl3_expert_record record,
				    unsigned char *arena)
{
	const struct expert_job_record *const job =
		(const struct expert_job_record *)(arena + record.jobs_offset) +
		blockIdx.y;
	if (job->row_count == 0) {
		return;
	}
	const uint32_t projection = blockIdx.z;
	const uint32_t inputs = down ? record.inner_size : record.input_size;
	const uint32_t outputs = down ? record.input_size : record.inner_size;
	const uint32_t lane = threadIdx.x % EXL3_WARP;
	const uint32_t slice = threadIdx.x / EXL3_WARP;
	const uint32_t lane_group = lane / MMA_LANE_GROUP;
	const uint32_t lane_place = lane % MMA_LANE_GROUP;
	const uint64_t matrix_offset = down	       ? job->down_offset :
				       projection == 0 ? job->gate_offset :
							 job->up_offset;
	const uint32_t *const matrix =
		(const uint32_t *)(arena + matrix_offset);
	const half *const operands =
		(const half *)(arena + record.operands_offset);
	float accumulators[MMA_CHAINS] = { 0 };
	const uint32_t tile_count = inputs / EXL3_TILE;
	const uint32_t first = tile_count * slice / slices;
	const uint32_t last = tile_count * (slice + 1u) / slices;
	for (uint32_t inner_tile = first; inner_tile < last; inner_tile++) {
		const uint32_t *const tile =
			matrix + ((uint64_t)inner_tile * (outputs / EXL3_TILE) +
				  blockIdx.x) *
					 EXL3_TILE_VALUES * bits /
					 EXL3_WORD_BITS;
		uint32_t weights[MMA_CHAINS];
		const uint32_t ring_bits = EXL3_TILE_VALUES * bits;
		const uint32_t begin =
			((lane * FRAGMENT_VALUES_PER_LANE + 1u) * bits +
			 ring_bits - EXL3_STATE_BITS) %
			ring_bits;
		const uint32_t word = begin / EXL3_WORD_BITS;
		const uint32_t shift = begin % EXL3_WORD_BITS;
		const uint64_t packed =
			((uint64_t)tile[word] << EXL3_WORD_BITS) |
			tile[(word + 1u) % (ring_bits / EXL3_WORD_BITS)];

		for (uint32_t fragment = 0; fragment < MMA_CHAINS; fragment++) {
			const uint32_t element =
				(fragment % MMA_PAIR) *
					FRAGMENT_HIGH_COLUMN_BIT +
				(fragment / MMA_PAIR) * FRAGMENT_HIGH_ROW_BIT;
			const uint32_t first_shift = EXL3_WORD_BITS +
						     EXL3_STATE_BITS - shift -
						     element * bits;
			const uint32_t low_state =
				(uint32_t)(packed >> first_shift) &
				EXL3_HALF_MASK;
			const uint32_t high_state =
				(uint32_t)(packed >> (first_shift - bits)) &
				EXL3_HALF_MASK;
			weights[fragment] =
				exl3_half_pair(exl3_state_value(low_state),
					       exl3_state_value(high_state));
		}
		const uint32_t pair =
			job->pair_index[lane_group < job->row_count ? lane_group :
								      0];
		const uint64_t row_offset =
			((uint64_t)projection * record.pair_count + pair) *
			inputs;
		const uint32_t input =
			inner_tile * EXL3_TILE + lane_place * MMA_PAIR;
		const uint32_t low =
			exl3_half_pair(operands[row_offset + input],
				       operands[row_offset + input + 1u]);
		const uint32_t high = exl3_half_pair(
			operands[row_offset + input + MMA_INPUT_HALF],
			operands[row_offset + input + MMA_INPUT_HALF + 1u]);
		asm volatile(
			"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
			"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
			: "+f"(accumulators[MMA_CHAIN_LOW_EVEN]),
			  "+f"(accumulators[MMA_CHAIN_LOW_ODD]),
			  "+f"(accumulators[MMA_CHAIN_HIGH_EVEN]),
			  "+f"(accumulators[MMA_CHAIN_HIGH_ODD])
			: "r"(weights[MMA_CHAIN_LOW_EVEN]),
			  "r"(weights[MMA_CHAIN_LOW_ODD]),
			  "r"(weights[MMA_CHAIN_HIGH_EVEN]),
			  "r"(weights[MMA_CHAIN_HIGH_ODD]), "r"(low),
			  "r"(high));
	}
	__shared__ float partial[slices][EXL3_WARP][MMA_CHAINS];
	for (uint32_t chain = 0; chain < MMA_CHAINS; chain++) {
		partial[slice][lane][chain] = accumulators[chain];
	}
	__syncthreads();
	if (slice == 0) {
		for (uint32_t chain = 0; chain < MMA_CHAINS; chain++) {
			float sum = 0;
			for (uint32_t part = 0; part < slices; part++) {
				sum += partial[part][lane][chain];
			}
			const uint32_t column =
				lane_place * MMA_COLUMNS_PER_LANE +
				chain % MMA_PAIR;
			const uint32_t output =
				blockIdx.x * EXL3_TILE + lane_group +
				(chain / MMA_PAIR) * MMA_OUTPUT_HALF;
			if (column < job->row_count) {
				const uint32_t pair = job->pair_index[column];
				((float *)(arena + record.products_offset))
					[((uint64_t)projection *
						  record.pair_count +
					  pair) * outputs +
					 output] = sum;
			}
		}
	}
}

#define EXL3_PRODUCT_ENTRY(name, bits, down, slices)                      \
	extern "C" __global__ void name(struct exl3_expert_record record, \
					unsigned char *arena)             \
	{                                                                 \
		exl3_product<bits, down, slices>(record, arena);          \
	}
EXL3_PRODUCT_ENTRY(exl3_gate_up_k2, EXL3_MINIMUM_BITS, false,
		   EXL3_PRODUCT_SLICES)
EXL3_PRODUCT_ENTRY(exl3_gate_up_k2_s8, EXL3_MINIMUM_BITS, false,
		   EXL3_RELAXED_SLICES)
EXL3_PRODUCT_ENTRY(exl3_gate_up_k3, EXL3_MAXIMUM_BITS, false,
		   EXL3_PRODUCT_SLICES)
EXL3_PRODUCT_ENTRY(exl3_gate_up_k3_s8, EXL3_MAXIMUM_BITS, false,
		   EXL3_RELAXED_SLICES)
EXL3_PRODUCT_ENTRY(exl3_down_k2, EXL3_MINIMUM_BITS, true, EXL3_PRODUCT_SLICES)
EXL3_PRODUCT_ENTRY(exl3_down_k2_s8, EXL3_MINIMUM_BITS, true,
		   EXL3_RELAXED_SLICES)
EXL3_PRODUCT_ENTRY(exl3_down_k3, EXL3_MAXIMUM_BITS, true, EXL3_PRODUCT_SLICES)
EXL3_PRODUCT_ENTRY(exl3_down_k3_s8, EXL3_MAXIMUM_BITS, true,
		   EXL3_RELAXED_SLICES)
#undef EXL3_PRODUCT_ENTRY

extern "C" __global__ void exl3_activate(struct exl3_expert_record record,
					 unsigned char *arena)
{
	const struct expert_job_record *const job =
		(const struct expert_job_record *)(arena + record.jobs_offset) +
		blockIdx.y;
	if (job->row_count == 0) {
		return;
	}
	__shared__ float gate[EXL3_TRANSFORM], up[EXL3_TRANSFORM];
	const uint32_t channel = blockIdx.x * EXL3_TRANSFORM + threadIdx.x;
	const uint64_t codes_bytes = exl3_codes_bytes(record);
	const half *const gate_scale =
		(const half *)(arena + job->gate_offset + codes_bytes) +
		record.input_size;
	const half *const up_scale =
		(const half *)(arena + job->up_offset + codes_bytes) +
		record.input_size;
	const half *const down_scale =
		(const half *)(arena + job->down_offset + codes_bytes);
	const float *const products =
		(const float *)(arena + record.products_offset);
	for (uint32_t column = 0; column < job->row_count; column++) {
		const uint32_t pair = job->pair_index[column];
		gate[threadIdx.x] =
			products[(uint64_t)pair * record.inner_size + channel];
		up[threadIdx.x] = products[((uint64_t)record.pair_count +
					    pair) * record.inner_size +
					   channel];
		__syncthreads();
		exl3_hadamard(gate);
		exl3_hadamard(up);
		const float gate_value = gate[threadIdx.x] *
					 EXL3_NORMALIZATION *
					 __half2float(gate_scale[channel]);
		const float up_value = up[threadIdx.x] * EXL3_NORMALIZATION *
				       __half2float(up_scale[channel]);
		const float bounded_gate =
			fminf(gate_value, record.activation_limit);
		const float bounded_up =
			fminf(fmaxf(up_value, -record.activation_limit),
			      record.activation_limit);
		const float hidden = bounded_gate /
				     (1.0f + expf(-bounded_gate)) * bounded_up;
		((__nv_bfloat16 *)(arena + record.hidden_offset))
			[(uint64_t)pair * record.inner_size + channel] =
				__float2bfloat16_rn(hidden);
		__syncthreads();
		gate[threadIdx.x] = hidden * __half2float(down_scale[channel]);
		__syncthreads();
		exl3_hadamard(gate);
		((half *)(arena +
			  record.operands_offset))[(uint64_t)pair *
							   record.inner_size +
						   channel] =
			__float2half_rn(gate[threadIdx.x] * EXL3_NORMALIZATION);
		__syncthreads();
	}
}

extern "C" __global__ void exl3_finish(struct exl3_expert_record record,
				       unsigned char *arena)
{
	const struct expert_job_record *const job =
		(const struct expert_job_record *)(arena + record.jobs_offset) +
		blockIdx.y;
	if (job->row_count == 0) {
		return;
	}
	__shared__ float values[EXL3_TRANSFORM];
	const uint32_t channel = blockIdx.x * EXL3_TRANSFORM + threadIdx.x;
	const half *const scales = (const half *)(arena + job->down_offset +
						  exl3_codes_bytes(record)) +
				   record.inner_size;
	for (uint32_t column = 0; column < job->row_count; column++) {
		const uint32_t pair = job->pair_index[column];
		values[threadIdx.x] =
			((const float *)(arena + record.products_offset))
				[(uint64_t)pair * record.input_size + channel];
		__syncthreads();
		exl3_hadamard(values);
		const float value = values[threadIdx.x] * EXL3_NORMALIZATION *
				    __half2float(scales[channel]);
		((float *)(arena + record.output_offset))
			[(uint64_t)pair * record.input_size + channel] =
				value * job->route_weight[column];
		__syncthreads();
	}
}

/* The shared expert and hyper post consume the existing FP32 routed-sum array.
 * EXL3 avoids rounding each pair before this fixed expert-order reduction. */
extern "C" __global__ void exl3_row_sum(struct expert_row_sum_arguments record,
					unsigned char *arena)
{
	const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
	if (channel >= record.width) {
		return;
	}
	const uint64_t row = blockIdx.y;
	const float *const pairs =
		(const float *)(arena + record.pair_output_offset);
	float sum = 0;
	for (uint32_t slot = 0; slot < record.pairs_per_row; slot++) {
		sum += pairs[(row * record.pairs_per_row + slot) * record.width +
			     channel];
	}
	((float *)(arena +
		   record.row_output_offset))[row * record.width + channel] =
		sum;
}

#include "anemone/exl3_tp_records.h"

extern "C" __global__ void
bb_exl3_activate_output_shard(struct bb_exl3_shard_activation_record arguments,
			      unsigned char *arena)
{
	const struct exl3_expert_record record = arguments.base;
	const struct expert_job_record *const job =
		(const struct expert_job_record *)(arena + record.jobs_offset) +
		blockIdx.y;
	if (job->row_count == 0) {
		return;
	}
	__shared__ float gate[EXL3_TRANSFORM], up[EXL3_TRANSFORM];
	const uint32_t channel = blockIdx.x * EXL3_TRANSFORM + threadIdx.x;
	const uint64_t codes_bytes = exl3_codes_bytes(record);
	const half *const gate_scale =
		(const half *)(arena + job->gate_offset + codes_bytes) +
		record.input_size;
	const half *const up_scale =
		(const half *)(arena + job->up_offset + codes_bytes) +
		record.input_size;
	const uint64_t down_codes_bytes = (uint64_t)arguments.down_input_size *
					  arguments.down_output_size /
					  EXL3_BITS_PER_BYTE *
					  record.bits_per_weight;
	const half *const down_scale =
		(const half *)(arena + job->down_offset + down_codes_bytes) +
		arguments.down_input_begin;
	const float *const products =
		(const float *)(arena + record.products_offset);
	for (uint32_t column = 0; column < job->row_count; column++) {
		const uint32_t pair = job->pair_index[column];
		gate[threadIdx.x] =
			products[(uint64_t)pair * record.inner_size + channel];
		up[threadIdx.x] = products[((uint64_t)record.pair_count +
					    pair) * record.inner_size +
					   channel];
		__syncthreads();
		exl3_hadamard(gate);
		exl3_hadamard(up);
		const float gate_value = gate[threadIdx.x] *
					 EXL3_NORMALIZATION *
					 __half2float(gate_scale[channel]);
		const float up_value = up[threadIdx.x] * EXL3_NORMALIZATION *
				       __half2float(up_scale[channel]);
		const float bounded_gate =
			fminf(gate_value, record.activation_limit);
		const float bounded_up =
			fminf(fmaxf(up_value, -record.activation_limit),
			      record.activation_limit);
		const float hidden = bounded_gate /
				     (1.0f + expf(-bounded_gate)) * bounded_up;
		((__nv_bfloat16 *)(arena + record.hidden_offset))
			[(uint64_t)pair * record.inner_size + channel] =
				__float2bfloat16_rn(hidden);
		__syncthreads();
		gate[threadIdx.x] = hidden * __half2float(down_scale[channel]);
		__syncthreads();
		exl3_hadamard(gate);
		((half *)(arena +
			  record.operands_offset))[(uint64_t)pair *
							   record.inner_size +
						   channel] =
			__float2half_rn(gate[threadIdx.x] * EXL3_NORMALIZATION);
		__syncthreads();
	}
}
