/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* kernel_vision.cu: BF16 ViT/projector passes. Float32 products, explicit reference boundaries.
 * Attention materializes one query/head's 128-key tile in block-private shared memory, never an N by N array.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/vision_records.h"
#include "anemone/step_records.h"

#define VISION_FLOAT_HALF_SHIFT 16u
#define VISION_BFLOAT16_ROUND_BIAS 0x7FFFu
#define VISION_FLOAT_EXPONENT 0x7F800000u
#define VISION_FLOAT_MANTISSA 0x007FFFFFu
#define VISION_BFLOAT16_NAN 0x7FC0u
#define VISION_HALF 0.5f
#define VISION_GELU_INVERSE_SQRT_TWO 0x1.6a09e6p-1f
#define VISION_LOG2_E 1.442695040888963407359924681001892137
#define VISION_REDUCTION_FIRST (VISION_THREADS / VISION_HALF_DIVISOR)

__device__ __forceinline__ float vision_float(uint16_t value)
{
	return __uint_as_float((uint32_t)value << VISION_FLOAT_HALF_SHIFT);
}

__device__ __forceinline__ uint16_t vision_bfloat(float value)
{
	const uint32_t bits = __float_as_uint(value);
	return (uint16_t)(((bits & VISION_FLOAT_EXPONENT) ==
				   VISION_FLOAT_EXPONENT &&
			   (bits & VISION_FLOAT_MANTISSA) != 0) ?
				  VISION_BFLOAT16_NAN :
				  (bits + VISION_BFLOAT16_ROUND_BIAS +
				   ((bits >> VISION_FLOAT_HALF_SHIFT) & 1u)) >>
					  VISION_FLOAT_HALF_SHIFT);
}

#define VISION_MMA_WARP_THREADS 32u
#define VISION_MMA_LANE_PLACES 4u
#define VISION_MMA_HALF_ROWS 8u
#define VISION_MMA_COLUMNS 8u
#define VISION_MMA_ACTIVE_WARPS 2u
#define VISION_MMA_PAIR_VALUES 2u
#define VISION_MMA_HALF_SHIFT 16u
#define VISION_MMA_RESULTS_PER_LANE 4u

__device__ static __forceinline__
	uint32_t vision_mma_pair(const uint16_t *values)
{
	return (uint32_t)values[0] |
	       ((uint32_t)values[1] << VISION_MMA_HALF_SHIFT);
}

extern "C" __global__ void vision_linear(struct vision_arguments arguments,
					 unsigned char *arena)
{
	__shared__ uint16_t input_tile[VISION_LINEAR_TILE][VISION_LINEAR_TILE];
	__shared__ uint16_t weight_tile[VISION_LINEAR_TILE][VISION_LINEAR_TILE];
	const uint32_t local_row = threadIdx.x / VISION_LINEAR_TILE;
	const uint32_t local_inner = threadIdx.x % VISION_LINEAR_TILE;
	const uint32_t row_base = blockIdx.y * VISION_LINEAR_TILE;
	const uint32_t column_base = blockIdx.x * VISION_LINEAR_TILE;
	const uint32_t warp = threadIdx.x / VISION_MMA_WARP_THREADS;
	const uint32_t lane = threadIdx.x % VISION_MMA_WARP_THREADS;
	const uint32_t lane_group = lane / VISION_MMA_LANE_PLACES;
	const uint32_t lane_place = lane % VISION_MMA_LANE_PLACES;
	const uint32_t inner_pair = lane_place * VISION_MMA_PAIR_VALUES;
	const uint16_t *const source =
		(const uint16_t *)(arena + arguments.source_offset);
	const uint16_t *const weights =
		(const uint16_t *)(arena + arguments.weights_offset);
	const uint16_t *const bias =
		(const uint16_t *)(arena + arguments.bias_offset);
	const uint32_t requested_partition =
		(arguments.flags & VISION_LINEAR_K_PARTITION_MASK) >>
		VISION_LINEAR_K_PARTITION_SHIFT;
	const uint32_t partition_width = requested_partition != 0 ?
						 requested_partition :
						 arguments.input_width;
	uint16_t carried[VISION_MMA_RESULTS_PER_LANE] = { 0 };
	for (uint32_t partition_base = 0;
	     partition_base < arguments.input_width;
	     partition_base += partition_width) {
		const uint32_t remaining =
			arguments.input_width - partition_base;
		const uint32_t partition_end =
			partition_base + (partition_width < remaining ?
						  partition_width :
						  remaining);
		float accumulators[VISION_MMA_RESULTS_PER_LANE] = { 0.0f };
		const uint32_t residue =
			(partition_end - partition_base) % VISION_LINEAR_TILE;
		for (uint32_t inner_base = partition_base;
		     inner_base < partition_end;) {
			const uint32_t tile_width =
				inner_base == partition_base && residue != 0 ?
					residue :
					VISION_LINEAR_TILE;
			const uint32_t tile_end = inner_base + tile_width;
			const uint32_t inner = inner_base + local_inner;
			const uint32_t input_row = row_base + local_row;
			const uint32_t weight_row = column_base + local_row;
			input_tile[local_row][local_inner] =
				input_row < arguments.rows && inner < tile_end ?
					source[(size_t)input_row *
						       arguments.input_width +
					       inner] :
					0;
			weight_tile[local_row][local_inner] =
				weight_row < arguments.output_width &&
						inner < tile_end ?
					weights[(size_t)weight_row *
							arguments.input_width +
						inner] :
					0;
			__syncthreads();
			if (warp < VISION_MMA_ACTIVE_WARPS) {
				/* Lane 4g+t: A rows g,g+8 at K2t,K2t+8; B column g at those K pairs. */
				const uint32_t input_low_low = vision_mma_pair(
					&input_tile[lane_group][inner_pair]);
				const uint32_t input_high_low = vision_mma_pair(
					&input_tile[lane_group +
						    VISION_MMA_HALF_ROWS]
						   [inner_pair]);
				const uint32_t input_low_high = vision_mma_pair(
					&input_tile[lane_group]
						   [inner_pair +
						    VISION_MMA_HALF_ROWS]);
				const uint32_t input_high_high = vision_mma_pair(
					&input_tile[lane_group +
						    VISION_MMA_HALF_ROWS]
						   [inner_pair +
						    VISION_MMA_HALF_ROWS]);
				const uint32_t weight_column =
					warp * VISION_MMA_COLUMNS + lane_group;
				const uint32_t weight_low = vision_mma_pair(
					&weight_tile[weight_column][inner_pair]);
				const uint32_t weight_high = vision_mma_pair(
					&weight_tile[weight_column]
						    [inner_pair +
						     VISION_MMA_HALF_ROWS]);
				if ((arguments.flags & VISION_LINEAR_MMA_K8) !=
				    0) {
					asm volatile(
						"mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
						"{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
						: "+f"(accumulators[0]),
						  "+f"(accumulators[1]),
						  "+f"(accumulators
						       [VISION_HALF_DIVISOR]),
						  "+f"(accumulators
						       [VISION_MMA_RESULTS_PER_LANE -
							1u])
						: "r"(input_low_low),
						  "r"(input_high_low),
						  "r"(weight_low));
					asm volatile(
						"mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
						"{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
						: "+f"(accumulators[0]),
						  "+f"(accumulators[1]),
						  "+f"(accumulators
						       [VISION_HALF_DIVISOR]),
						  "+f"(accumulators
						       [VISION_MMA_RESULTS_PER_LANE -
							1u])
						: "r"(input_low_high),
						  "r"(input_high_high),
						  "r"(weight_high));
				} else {
					asm volatile(
						"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
						"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
						: "+f"(accumulators[0]),
						  "+f"(accumulators[1]),
						  "+f"(accumulators
						       [VISION_HALF_DIVISOR]),
						  "+f"(accumulators
						       [VISION_MMA_RESULTS_PER_LANE -
							1u])
						: "r"(input_low_low),
						  "r"(input_high_low),
						  "r"(input_low_high),
						  "r"(input_high_high),
						  "r"(weight_low),
						  "r"(weight_high));
				}
			}
			__syncthreads();
			inner_base = tile_end;
		}
		if (warp < VISION_MMA_ACTIVE_WARPS) {
			for (uint32_t result = 0;
			     result < VISION_MMA_RESULTS_PER_LANE; result++) {
				const uint32_t column =
					column_base +
					warp * VISION_MMA_COLUMNS + inner_pair +
					result % VISION_MMA_PAIR_VALUES;
				float sum = accumulators[result];
				const bool bias_last =
					(arguments.flags &
					 VISION_LINEAR_BIAS_LAST) != 0;
				const bool bias_partition =
					bias_last ?
						partition_end ==
							arguments.input_width :
						partition_base == 0;
				if ((arguments.flags & VISION_LINEAR_BIAS) !=
					    0 &&
				    bias_partition &&
				    column < arguments.output_width) {
					sum += vision_float(bias[column]);
				}
				if (partition_base != 0) {
					sum += vision_float(carried[result]);
				}
				carried[result] = vision_bfloat(sum);
			}
		}
	}
	if (warp < VISION_MMA_ACTIVE_WARPS) {
		uint16_t *const destination =
			(uint16_t *)(arena + arguments.destination_offset);
		for (uint32_t result = 0; result < VISION_MMA_RESULTS_PER_LANE;
		     result++) {
			const uint32_t row = row_base + lane_group +
					     (result / VISION_MMA_PAIR_VALUES) *
						     VISION_MMA_HALF_ROWS;
			const uint32_t column =
				column_base + warp * VISION_MMA_COLUMNS +
				inner_pair + result % VISION_MMA_PAIR_VALUES;
			if (row < arguments.rows &&
			    column < arguments.output_width) {
				destination[(size_t)row * arguments.output_width +
					    column] = carried[result];
			}
		}
	}
}

#define VISION_RMS_REFERENCE_MAX_THREADS 512u
#define VISION_RMS_REFERENCE_WARP_THREADS 32u
#define VISION_RMS_REFERENCE_VECTOR_SIZE 4u
#define VISION_RMS_REFERENCE_VECTOR_THRESHOLD 128u

__device__ static __forceinline__
	uint32_t vision_rms_reference_floor_power(uint32_t value)
{
	uint32_t power = 1;
	while (power <= value / VISION_HALF_DIVISOR) {
		power *= VISION_HALF_DIVISOR;
	}
	return power;
}

extern "C" __global__ void vision_rms(struct vision_arguments arguments,
				      unsigned char *arena)
{
	__shared__ float partials[VISION_THREADS];
	const bool vectorized = arguments.input_width >=
				VISION_RMS_REFERENCE_VECTOR_THRESHOLD;
	const uint32_t input_vectors =
		vectorized ? arguments.input_width /
				     VISION_RMS_REFERENCE_VECTOR_SIZE :
			     arguments.input_width;
	const uint32_t dimension_x =
		vision_rms_reference_floor_power(input_vectors);
	const uint32_t dimension_y =
		arguments.rows < VISION_RMS_REFERENCE_MAX_THREADS ?
			vision_rms_reference_floor_power(arguments.rows) :
			VISION_RMS_REFERENCE_MAX_THREADS;
	const uint32_t first_width =
		dimension_x < VISION_RMS_REFERENCE_WARP_THREADS ?
			dimension_x :
			VISION_RMS_REFERENCE_WARP_THREADS;
	const uint32_t maximum_height =
		VISION_RMS_REFERENCE_MAX_THREADS / first_width;
	const uint32_t height = dimension_y < maximum_height ? dimension_y :
							       maximum_height;
	const uint32_t maximum_width =
		VISION_RMS_REFERENCE_MAX_THREADS / height;
	const uint32_t reduce_width =
		dimension_x < maximum_width ? dimension_x : maximum_width;
	const uint16_t *const source =
		(const uint16_t *)(arena + arguments.source_offset) +
		(size_t)blockIdx.x * arguments.input_width;
	float sums[VISION_RMS_REFERENCE_VECTOR_SIZE] = { 0.0f };
	if (threadIdx.x < reduce_width) {
		if (vectorized) {
			for (uint32_t vector = threadIdx.x;
			     vector < input_vectors; vector += reduce_width) {
#pragma unroll
				for (uint32_t component = 0;
				     component <
				     VISION_RMS_REFERENCE_VECTOR_SIZE;
				     component++) {
					const float value = vision_float(
						source[vector * VISION_RMS_REFERENCE_VECTOR_SIZE +
						       component]);
					const float squared = value * value;
					sums[component] += squared;
				}
			}
		} else {
			for (uint32_t first = threadIdx.x;
			     first < arguments.input_width;
			     first +=
			     reduce_width * VISION_RMS_REFERENCE_VECTOR_SIZE) {
#pragma unroll
				for (uint32_t component = 0;
				     component <
				     VISION_RMS_REFERENCE_VECTOR_SIZE;
				     component++) {
					const uint32_t channel =
						first +
						component * reduce_width;
					if (channel < arguments.input_width) {
						const float value = vision_float(
							source[channel]);
						const float squared =
							value * value;
						sums[component] += squared;
					}
				}
			}
		}
	}
	float combined = sums[0];
#pragma unroll
	for (uint32_t component = 1;
	     component < VISION_RMS_REFERENCE_VECTOR_SIZE; component++) {
		combined += sums[component];
	}
	partials[threadIdx.x] = combined;
	__syncthreads();
	for (uint32_t distance = reduce_width / VISION_HALF_DIVISOR;
	     distance != 0; distance /= VISION_HALF_DIVISOR) {
		if (threadIdx.x < distance) {
			partials[threadIdx.x] +=
				partials[threadIdx.x + distance];
		}
		__syncthreads();
	}
	const float mean = partials[0] * (1.0f / (float)arguments.input_width);
	const float inverse = rsqrtf(mean + arguments.epsilon);
	const float *const weights =
		(const float *)(arena + arguments.weights_offset);
	const uint16_t *const bfloat_weights =
		(const uint16_t *)(arena + arguments.weights_offset);
	uint16_t *const destination =
		(uint16_t *)(arena + arguments.destination_offset) +
		(size_t)blockIdx.x * arguments.input_width;
	for (uint32_t channel = threadIdx.x; channel < arguments.input_width;
	     channel += blockDim.x) {
		const float weight =
			(arguments.flags & VISION_NORM_BFLOAT16_WEIGHT) != 0 ?
				vision_float(bfloat_weights[channel]) :
				weights[channel];
		destination[channel] = vision_bfloat(
			(vision_float(source[channel]) * inverse) * weight);
	}
}

#define VISION_ROPE_MEASURED_FREQUENCIES 16u
#define VISION_ROPE_MAXIMUM_ANGLE_BITS 0x453F6000u
#define VISION_ROPE_INVERSE_HALF_PI 0x1.45F306DC9C883p-1
#define VISION_ROPE_HALF_PI_HIGH 0x1.921FB54442D18p+0
#define VISION_ROPE_HALF_PI_LOW 0x1.1A62633145C07p-54
#define VISION_ROPE_ROUND_HALF 0.5
#define VISION_ROPE_QUADRANT_MASK 3u
#define VISION_ROPE_OPPOSITE_QUADRANT 2u
#define VISION_ROPE_SINE_CUBIC (-1.0 / 6.0)
#define VISION_ROPE_SINE_FIFTH (1.0 / 120.0)
#define VISION_ROPE_SINE_SEVENTH (-1.0 / 5040.0)
#define VISION_ROPE_SINE_NINTH (1.0 / 362880.0)
#define VISION_ROPE_SINE_ELEVENTH (-1.0 / 39916800.0)
#define VISION_ROPE_SINE_THIRTEENTH (1.0 / 6227020800.0)
#define VISION_ROPE_SINE_FIFTEENTH (-1.0 / 1307674368000.0)
#define VISION_ROPE_COSINE_SQUARED (-1.0 / 2.0)
#define VISION_ROPE_COSINE_FOURTH (1.0 / 24.0)
#define VISION_ROPE_COSINE_SIXTH (-1.0 / 720.0)
#define VISION_ROPE_COSINE_EIGHTH (1.0 / 40320.0)
#define VISION_ROPE_COSINE_TENTH (-1.0 / 3628800.0)
#define VISION_ROPE_COSINE_TWELFTH (1.0 / 479001600.0)
#define VISION_ROPE_COSINE_FOURTEENTH (-1.0 / 87178291200.0)
#define VISION_ROPE_COSINE_SIXTEENTH (1.0 / 20922789888000.0)

__device__ __constant__ static const uint32_t
	vision_rope_inverse_frequency_words[VISION_ROPE_MEASURED_FREQUENCIES] = {
		0x3F800000u, 0x3F0FF59Au, 0x3EA1E89Bu, 0x3E361887u,
		0x3DCCCCCDu, 0x3D6655C2u, 0x3D0186E3u, 0x3C91AD39u,
		0x3C23D70Au, 0x3BB8449Cu, 0x3B4F3E38u, 0x3AE91528u,
		0x3A83126Fu, 0x3A136A16u, 0x39A5CB60u, 0x393A7753u
	};

/* The packed table maps this F64 helper's complete bounded domain to the pinned official GPU FP32 trig words.
 * Qualification covers both supported architectures; changing this helper or its frequencies requires new data.
 */
#define VISION_ROPE_CORRECTION_POSITIONS 3063u
#define VISION_ROPE_CORRECTION_FUNCTIONS 2u
#define VISION_ROPE_CORRECTION_BITS 3u
#define VISION_ROPE_CORRECTION_WORD_BITS 32u
#define VISION_ROPE_CORRECTION_MAGNITUDE_MASK 3u
#define VISION_ROPE_CORRECTION_SIGN_MASK 4u
#define VISION_ROPE_CORRECTION_SINE 0u
#define VISION_ROPE_CORRECTION_COSINE 1u
#define VISION_ROPE_TOY_FREQUENCIES 2u
#define VISION_ROPE_CORRECTION_WORDS                                           \
	(VISION_ROPE_CORRECTION_POSITIONS * VISION_ROPE_MEASURED_FREQUENCIES * \
	 VISION_ROPE_CORRECTION_FUNCTIONS * VISION_ROPE_CORRECTION_BITS /      \
	 VISION_ROPE_CORRECTION_WORD_BITS)

#include "vision_rope_corrections.inc"

/* Value order is [position][measured frequency][sine,cosine]. Deltas act on raw uint32 float words. */
__device__ static __forceinline__ float
vision_rope_correct(float value, uint32_t position, uint32_t frequency,
		    uint32_t function_index)
{
	const uint32_t index =
		(position * VISION_ROPE_MEASURED_FREQUENCIES + frequency) *
			VISION_ROPE_CORRECTION_FUNCTIONS +
		function_index;
	const uint32_t bit = index * VISION_ROPE_CORRECTION_BITS;
	const uint32_t word = bit / VISION_ROPE_CORRECTION_WORD_BITS;
	const uint32_t shift = bit % VISION_ROPE_CORRECTION_WORD_BITS;
	uint32_t code = vision_rope_corrections[word] >> shift;
	if (shift >
	    VISION_ROPE_CORRECTION_WORD_BITS - VISION_ROPE_CORRECTION_BITS) {
		code |= vision_rope_corrections[word + 1u]
			<< (VISION_ROPE_CORRECTION_WORD_BITS - shift);
	}
	const uint32_t magnitude = code & VISION_ROPE_CORRECTION_MAGNITUDE_MASK;
	const uint32_t bits = __float_as_uint(value);
	return __uint_as_float((code & VISION_ROPE_CORRECTION_SIGN_MASK) != 0 ?
				       bits - magnitude :
				       bits + magnitude);
}

__device__ static __forceinline__ void
vision_rope_bounded_sincos(float angle, float *sine, float *cosine)
{
	/* angle<=3062 gives a bounded integer quadrant. Split constants retain the cancelled low argument bits. */
	const uint32_t quadrant =
		(uint32_t)((double)angle * VISION_ROPE_INVERSE_HALF_PI +
			   VISION_ROPE_ROUND_HALF);
	const double turns = (double)quadrant;
	const double reduced_high =
		fma(-turns, VISION_ROPE_HALF_PI_HIGH, (double)angle);
	const double reduced =
		fma(-turns, VISION_ROPE_HALF_PI_LOW, reduced_high);
	const double squared = reduced * reduced;
	double sine_polynomial = VISION_ROPE_SINE_FIFTEENTH;
	sine_polynomial =
		fma(squared, sine_polynomial, VISION_ROPE_SINE_THIRTEENTH);
	sine_polynomial =
		fma(squared, sine_polynomial, VISION_ROPE_SINE_ELEVENTH);
	sine_polynomial = fma(squared, sine_polynomial, VISION_ROPE_SINE_NINTH);
	sine_polynomial =
		fma(squared, sine_polynomial, VISION_ROPE_SINE_SEVENTH);
	sine_polynomial = fma(squared, sine_polynomial, VISION_ROPE_SINE_FIFTH);
	sine_polynomial = fma(squared, sine_polynomial, VISION_ROPE_SINE_CUBIC);
	const float reduced_sine =
		(float)fma(reduced * squared, sine_polynomial, reduced);
	double cosine_polynomial = VISION_ROPE_COSINE_SIXTEENTH;
	cosine_polynomial =
		fma(squared, cosine_polynomial, VISION_ROPE_COSINE_FOURTEENTH);
	cosine_polynomial =
		fma(squared, cosine_polynomial, VISION_ROPE_COSINE_TWELFTH);
	cosine_polynomial =
		fma(squared, cosine_polynomial, VISION_ROPE_COSINE_TENTH);
	cosine_polynomial =
		fma(squared, cosine_polynomial, VISION_ROPE_COSINE_EIGHTH);
	cosine_polynomial =
		fma(squared, cosine_polynomial, VISION_ROPE_COSINE_SIXTH);
	cosine_polynomial =
		fma(squared, cosine_polynomial, VISION_ROPE_COSINE_FOURTH);
	cosine_polynomial =
		fma(squared, cosine_polynomial, VISION_ROPE_COSINE_SQUARED);
	const float reduced_cosine =
		(float)fma(squared, cosine_polynomial, 1.0);
	const uint32_t within_turn = quadrant & VISION_ROPE_QUADRANT_MASK;
	float sine_value = (within_turn & 1u) == 0 ? reduced_sine :
						     reduced_cosine;
	float cosine_value = (within_turn & 1u) == 0 ? reduced_cosine :
						       -reduced_sine;
	if ((within_turn & VISION_ROPE_OPPOSITE_QUADRANT) != 0) {
		sine_value = -sine_value;
		cosine_value = -cosine_value;
	}
	*sine = sine_value;
	*cosine = cosine_value;
}

extern "C" __global__ void vision_rope(struct vision_arguments arguments,
				       unsigned char *arena)
{
	const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair >= arguments.rows * arguments.output_width) {
		return;
	}
	const uint32_t half_width = arguments.input_width / VISION_HALF_DIVISOR;
	const uint32_t row = pair / arguments.input_width;
	const uint32_t within_row = pair % arguments.input_width;
	const uint32_t component = within_row / half_width;
	const uint32_t head_width = arguments.input_width / arguments.heads;
	const uint32_t half_head = head_width / VISION_HALF_DIVISOR;
	const uint32_t within_component = within_row % half_width;
	const uint32_t head = within_component / half_head;
	const uint32_t channel = within_component % half_head;
	const uint32_t axis_width = head_width / VISION_ROPE_PARTS;
	const uint32_t frequency_index = channel % axis_width;
	const uint32_t position = channel < axis_width ?
					  row / arguments.grid_width :
					  row % arguments.grid_width;
	float inverse_frequency;
	if (arguments.theta == VISION_ROPE_THETA &&
	    axis_width <= VISION_ROPE_MEASURED_FREQUENCIES &&
	    VISION_ROPE_MEASURED_FREQUENCIES % axis_width == 0) {
		inverse_frequency = __uint_as_float(
			vision_rope_inverse_frequency_words
				[frequency_index *
				 (VISION_ROPE_MEASURED_FREQUENCIES /
				  axis_width)]);
	} else {
		const float exponent =
			(float)(VISION_HALF_DIVISOR * frequency_index) /
			(float)half_head;
		inverse_frequency = exp2f(-exponent * log2f(arguments.theta));
	}
	const float raw_angle = (float)position * inverse_frequency;
	const uint32_t angle_bits = __float_as_uint(raw_angle);
	const float angle =
		__uint_as_float(angle_bits < VISION_ROPE_MAXIMUM_ANGLE_BITS ?
					angle_bits :
					VISION_ROPE_MAXIMUM_ANGLE_BITS);
	float cosine;
	float sine;
	vision_rope_bounded_sincos(angle, &sine, &cosine);
	if (arguments.theta == VISION_ROPE_THETA &&
	    position < VISION_ROPE_CORRECTION_POSITIONS &&
	    (axis_width == VISION_ROPE_MEASURED_FREQUENCIES ||
	     axis_width == VISION_ROPE_TOY_FREQUENCIES)) {
		const uint32_t measured_frequency =
			frequency_index *
			(VISION_ROPE_MEASURED_FREQUENCIES / axis_width);
		sine = vision_rope_correct(sine, position, measured_frequency,
					   VISION_ROPE_CORRECTION_SINE);
		cosine = vision_rope_correct(cosine, position,
					     measured_frequency,
					     VISION_ROPE_CORRECTION_COSINE);
	}
	uint16_t *const values =
		(uint16_t *)(arena + arguments.source_offset) +
		(size_t)row * VISION_QKV_COMPONENTS * arguments.input_width +
		(size_t)component * arguments.input_width +
		(size_t)head * head_width;
	const float first = vision_float(values[channel]);
	const float second = vision_float(values[channel + half_head]);
	values[channel] = vision_bfloat(first * cosine - second * sine);
	values[channel + half_head] =
		vision_bfloat(second * cosine + first * sine);
}

#define VISION_ATTENTION_MMA_KEY_COLUMNS 128u
#define VISION_ATTENTION_MMA_K_STEP 16u
#define VISION_ATTENTION_MMA_WARP_THREADS 32u
#define VISION_ATTENTION_MMA_LANE_PLACES 4u
#define VISION_ATTENTION_MMA_HALF_ROWS 8u
#define VISION_ATTENTION_MMA_COLUMNS 8u
#define VISION_ATTENTION_MMA_COLUMN_PARTS 2u
#define VISION_ATTENTION_MMA_GROUP_COLUMNS 64u
#define VISION_ATTENTION_MMA_OUTPUT_GROUPS 4u
#define VISION_ATTENTION_MMA_PAIR_VALUES 2u
#define VISION_ATTENTION_MMA_RESULTS 4u
#define VISION_ATTENTION_MMA_HALF_SHIFT 16u
#define VISION_ATTENTION_MMA_LOG2_E 1.442695040888963407359924681001892137

__device__ static __forceinline__
	uint32_t vision_attention_mma_pair(uint16_t first, uint16_t second)
{
	return (uint32_t)first |
	       ((uint32_t)second << VISION_ATTENTION_MMA_HALF_SHIFT);
}

__device__ static __forceinline__ void
vision_attention_mma_product(float *accumulators, uint32_t input_low_low,
			     uint32_t input_high_low, uint32_t input_low_high,
			     uint32_t input_high_high, uint32_t weight_low,
			     uint32_t weight_high)
{
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
		     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		     : "+f"(accumulators[0]), "+f"(accumulators[1]),
		       "+f"(accumulators[VISION_HALF_DIVISOR]),
		       "+f"(accumulators[VISION_ATTENTION_MMA_RESULTS - 1u])
		     : "r"(input_low_low), "r"(input_high_low),
		       "r"(input_low_high), "r"(input_high_high),
		       "r"(weight_low), "r"(weight_high));
}

extern "C" __global__ void vision_attention(struct vision_arguments arguments,
					    unsigned char *arena)
{
	__shared__ uint16_t query_chunk[VISION_ATTENTION_QUERY_TILE]
				       [VISION_ATTENTION_MMA_K_STEP];
	__shared__ uint16_t key_chunk[VISION_ATTENTION_MMA_KEY_COLUMNS]
				     [VISION_ATTENTION_MMA_K_STEP];
	__shared__ float scores[VISION_ATTENTION_QUERY_TILE]
			       [VISION_ATTENTION_MMA_KEY_COLUMNS];
	__shared__ uint16_t probabilities[VISION_ATTENTION_QUERY_TILE]
					 [VISION_ATTENTION_MMA_KEY_COLUMNS];
	__shared__ float maxima_quad[VISION_ATTENTION_QUERY_TILE]
				    [VISION_ATTENTION_SUM_LANES];
	__shared__ float denominator_quad[VISION_ATTENTION_QUERY_TILE]
					 [VISION_ATTENTION_SUM_LANES];
	__shared__ float row_maximum[VISION_ATTENTION_QUERY_TILE];
	__shared__ float row_scale[VISION_ATTENTION_QUERY_TILE];
	const uint32_t head = blockIdx.x % arguments.heads;
	const uint32_t first_query =
		blockIdx.x / arguments.heads * VISION_ATTENTION_QUERY_TILE;
	const uint32_t head_width = arguments.input_width / arguments.heads;
	const uint32_t qkv_stride =
		VISION_QKV_COMPONENTS * arguments.input_width;
	const uint32_t warp = threadIdx.x / VISION_ATTENTION_MMA_WARP_THREADS;
	const uint32_t lane = threadIdx.x % VISION_ATTENTION_MMA_WARP_THREADS;
	const uint32_t lane_group = lane / VISION_ATTENTION_MMA_LANE_PLACES;
	const uint32_t lane_place = lane % VISION_ATTENTION_MMA_LANE_PLACES;
	const uint32_t inner_pair =
		lane_place * VISION_ATTENTION_MMA_PAIR_VALUES;
	const uint16_t *const qkv =
		(const uint16_t *)(arena + arguments.source_offset);
	const float scale = (float)(1.0 / sqrt((double)head_width));
	const float scale_log2 =
		(float)((double)scale * VISION_ATTENTION_MMA_LOG2_E);
	float accumulated[VISION_ATTENTION_MMA_OUTPUT_GROUPS]
			 [VISION_ATTENTION_MMA_RESULTS] = { { 0.0f } };
	if (threadIdx.x < VISION_ATTENTION_QUERY_TILE) {
		row_maximum[threadIdx.x] = -INFINITY;
	}
	if (threadIdx.x <
	    VISION_ATTENTION_QUERY_TILE * VISION_ATTENTION_SUM_LANES) {
		denominator_quad[threadIdx.x / VISION_ATTENTION_SUM_LANES]
				[threadIdx.x % VISION_ATTENTION_SUM_LANES] =
					0.0f;
	}
	__syncthreads();
	for (uint32_t tile =
		     (arguments.rows + VISION_ATTENTION_MMA_KEY_COLUMNS - 1u) /
		     VISION_ATTENTION_MMA_KEY_COLUMNS;
	     tile != 0; tile--) {
		const uint32_t first_key =
			(tile - 1u) * VISION_ATTENTION_MMA_KEY_COLUMNS;
		float query_key[VISION_ATTENTION_MMA_COLUMN_PARTS]
			       [VISION_ATTENTION_MMA_RESULTS] = { { 0.0f } };
		for (uint32_t inner_base = 0; inner_base < head_width;
		     inner_base += VISION_ATTENTION_MMA_K_STEP) {
			const uint32_t query_row =
				threadIdx.x / VISION_ATTENTION_MMA_K_STEP;
			const uint32_t query_column =
				threadIdx.x % VISION_ATTENTION_MMA_K_STEP;
			query_chunk[query_row][query_column] =
				first_query + query_row < arguments.rows &&
						inner_base + query_column <
							head_width ?
					qkv[(size_t)(first_query + query_row) *
						    qkv_stride +
					    head * head_width + inner_base +
					    query_column] :
					0;
			for (uint32_t index = threadIdx.x;
			     index < VISION_ATTENTION_MMA_KEY_COLUMNS *
					     VISION_ATTENTION_MMA_K_STEP;
			     index += VISION_THREADS) {
				const uint32_t key_row =
					index / VISION_ATTENTION_MMA_K_STEP;
				const uint32_t key_column =
					index % VISION_ATTENTION_MMA_K_STEP;
				key_chunk[key_row][key_column] =
					first_key + key_row < arguments.rows &&
							inner_base + key_column <
								head_width ?
						qkv[(size_t)(first_key +
							     key_row) *
							    qkv_stride +
						    arguments.input_width +
						    head * head_width +
						    inner_base + key_column] :
						0;
			}
			__syncthreads();
			const uint32_t input_low_low = vision_attention_mma_pair(
				query_chunk[lane_group][inner_pair],
				query_chunk[lane_group][inner_pair + 1u]);
			const uint32_t input_high_low = vision_attention_mma_pair(
				query_chunk[lane_group +
					    VISION_ATTENTION_MMA_HALF_ROWS]
					   [inner_pair],
				query_chunk[lane_group +
					    VISION_ATTENTION_MMA_HALF_ROWS]
					   [inner_pair + 1u]);
			const uint32_t input_low_high = vision_attention_mma_pair(
				query_chunk[lane_group]
					   [inner_pair +
					    VISION_ATTENTION_MMA_HALF_ROWS],
				query_chunk[lane_group]
					   [inner_pair +
					    VISION_ATTENTION_MMA_HALF_ROWS +
					    1u]);
			const uint32_t input_high_high = vision_attention_mma_pair(
				query_chunk[lane_group +
					    VISION_ATTENTION_MMA_HALF_ROWS]
					   [inner_pair +
					    VISION_ATTENTION_MMA_HALF_ROWS],
				query_chunk[lane_group +
					    VISION_ATTENTION_MMA_HALF_ROWS]
					   [inner_pair +
					    VISION_ATTENTION_MMA_HALF_ROWS +
					    1u]);
#pragma unroll
			for (uint32_t part = 0;
			     part < VISION_ATTENTION_MMA_COLUMN_PARTS; part++) {
				const uint32_t key_column =
					warp * VISION_ATTENTION_MMA_COLUMNS *
						VISION_ATTENTION_MMA_COLUMN_PARTS +
					part * VISION_ATTENTION_MMA_COLUMNS +
					lane_group;
				const uint32_t weight_low =
					vision_attention_mma_pair(
						key_chunk[key_column]
							 [inner_pair],
						key_chunk[key_column]
							 [inner_pair + 1u]);
				const uint32_t weight_high = vision_attention_mma_pair(
					key_chunk[key_column]
						 [inner_pair +
						  VISION_ATTENTION_MMA_HALF_ROWS],
					key_chunk[key_column]
						 [inner_pair +
						  VISION_ATTENTION_MMA_HALF_ROWS +
						  1u]);
				vision_attention_mma_product(
					query_key[part], input_low_low,
					input_high_low, input_low_high,
					input_high_high, weight_low,
					weight_high);
			}
			__syncthreads();
		}
#pragma unroll
		for (uint32_t part = 0;
		     part < VISION_ATTENTION_MMA_COLUMN_PARTS; part++) {
#pragma unroll
			for (uint32_t result = 0;
			     result < VISION_ATTENTION_MMA_RESULTS; result++) {
				const uint32_t query_row =
					lane_group +
					result /
						VISION_ATTENTION_MMA_PAIR_VALUES *
						VISION_ATTENTION_MMA_HALF_ROWS;
				const uint32_t key_column =
					warp * VISION_ATTENTION_MMA_COLUMNS *
						VISION_ATTENTION_MMA_COLUMN_PARTS +
					part * VISION_ATTENTION_MMA_COLUMNS +
					inner_pair +
					result %
						VISION_ATTENTION_MMA_PAIR_VALUES;
				scores[query_row][key_column] =
					first_key + key_column <
							arguments.rows ?
						query_key[part][result] :
						-INFINITY;
			}
		}
		__syncthreads();
		if (threadIdx.x <
		    VISION_ATTENTION_QUERY_TILE * VISION_ATTENTION_SUM_LANES) {
			const uint32_t query_row =
				threadIdx.x / VISION_ATTENTION_SUM_LANES;
			const uint32_t sum_lane =
				threadIdx.x % VISION_ATTENTION_SUM_LANES;
			float maximum = -INFINITY;
			for (uint32_t column =
				     sum_lane *
				     VISION_ATTENTION_COLUMNS_PER_LANE;
			     column < VISION_ATTENTION_MMA_KEY_COLUMNS;
			     column += VISION_ATTENTION_SUM_LANES *
				       VISION_ATTENTION_COLUMNS_PER_LANE) {
				maximum = fmaxf(maximum,
						scores[query_row][column]);
				maximum = fmaxf(maximum,
						scores[query_row][column + 1u]);
			}
			maxima_quad[query_row][sum_lane] = maximum;
		}
		__syncthreads();
		if (threadIdx.x < VISION_ATTENTION_QUERY_TILE) {
			const uint32_t query_row = threadIdx.x;
			const float previous = row_maximum[query_row];
			const float maximum = fmaxf(
				previous,
				fmaxf(fmaxf(maxima_quad[query_row][0],
					    maxima_quad[query_row]
						       [VISION_HALF_DIVISOR]),
				      fmaxf(maxima_quad[query_row][1],
					    maxima_quad[query_row]
						       [VISION_ATTENTION_SUM_LANES -
							1u])));
			row_maximum[query_row] = maximum;
			row_scale[query_row] =
				previous == -INFINITY ?
					0.0f :
					exp2f((previous - maximum) *
					      scale_log2);
		}
		__syncthreads();
		if (threadIdx.x <
		    VISION_ATTENTION_QUERY_TILE * VISION_ATTENTION_SUM_LANES) {
			const uint32_t query_row =
				threadIdx.x / VISION_ATTENTION_SUM_LANES;
			const uint32_t sum_lane =
				threadIdx.x % VISION_ATTENTION_SUM_LANES;
			float sum = denominator_quad[query_row][sum_lane];
			const float maximum_scaled =
				row_maximum[query_row] * scale_log2;
			for (uint32_t column =
				     sum_lane *
				     VISION_ATTENTION_COLUMNS_PER_LANE;
			     column < VISION_ATTENTION_MMA_KEY_COLUMNS;
			     column += VISION_ATTENTION_SUM_LANES *
				       VISION_ATTENTION_COLUMNS_PER_LANE) {
				const float first = exp2f(
					scores[query_row][column] * scale_log2 -
					maximum_scaled);
				const float second =
					exp2f(scores[query_row][column + 1u] *
						      scale_log2 -
					      maximum_scaled);
				probabilities[query_row][column] =
					vision_bfloat(first);
				probabilities[query_row][column + 1u] =
					vision_bfloat(second);
				sum = column == sum_lane * VISION_ATTENTION_COLUMNS_PER_LANE ?
					      fmaf(sum, row_scale[query_row],
						   first) :
					      sum + first;
				sum += second;
			}
			denominator_quad[query_row][sum_lane] = sum;
		}
		__syncthreads();
#pragma unroll
		for (uint32_t group = 0;
		     group < VISION_ATTENTION_MMA_OUTPUT_GROUPS; group++) {
			const uint32_t output_column =
				group * VISION_ATTENTION_MMA_GROUP_COLUMNS +
				warp * VISION_ATTENTION_MMA_COLUMNS +
				lane_group;
			if (group * VISION_ATTENTION_MMA_GROUP_COLUMNS +
				    warp * VISION_ATTENTION_MMA_COLUMNS <
			    head_width) {
				accumulated[group][0] *= row_scale[lane_group];
				accumulated[group][1] *= row_scale[lane_group];
				accumulated[group][VISION_HALF_DIVISOR] *=
					row_scale[lane_group +
						  VISION_ATTENTION_MMA_HALF_ROWS];
				accumulated[group]
					   [VISION_ATTENTION_MMA_RESULTS - 1u] *=
					row_scale[lane_group +
						  VISION_ATTENTION_MMA_HALF_ROWS];
				for (uint32_t inner_base = 0;
				     inner_base <
				     VISION_ATTENTION_MMA_KEY_COLUMNS;
				     inner_base +=
				     VISION_ATTENTION_MMA_K_STEP) {
					const uint32_t input_low_low =
						vision_attention_mma_pair(
							probabilities
								[lane_group]
								[inner_base +
								 inner_pair],
							probabilities[lane_group]
								     [inner_base +
								      inner_pair +
								      1u]);
					const uint32_t input_high_low =
						vision_attention_mma_pair(
							probabilities
								[lane_group +
								 VISION_ATTENTION_MMA_HALF_ROWS]
								[inner_base +
								 inner_pair],
							probabilities
								[lane_group +
								 VISION_ATTENTION_MMA_HALF_ROWS]
								[inner_base +
								 inner_pair +
								 1u]);
					const uint32_t input_low_high = vision_attention_mma_pair(
						probabilities
							[lane_group]
							[inner_base + inner_pair +
							 VISION_ATTENTION_MMA_HALF_ROWS],
						probabilities
							[lane_group]
							[inner_base + inner_pair +
							 VISION_ATTENTION_MMA_HALF_ROWS +
							 1u]);
					const uint32_t input_high_high = vision_attention_mma_pair(
						probabilities
							[lane_group +
							 VISION_ATTENTION_MMA_HALF_ROWS]
							[inner_base + inner_pair +
							 VISION_ATTENTION_MMA_HALF_ROWS],
						probabilities
							[lane_group +
							 VISION_ATTENTION_MMA_HALF_ROWS]
							[inner_base + inner_pair +
							 VISION_ATTENTION_MMA_HALF_ROWS +
							 1u]);
					const uint32_t key_low = first_key +
								 inner_base +
								 inner_pair;
					const uint32_t key_high =
						key_low +
						VISION_ATTENTION_MMA_HALF_ROWS;
					const uint16_t weight_low_first =
						key_low < arguments.rows &&
								output_column <
									head_width ?
							qkv[(size_t)key_low *
								    qkv_stride +
							    VISION_GATE_COMPONENTS *
								    arguments.input_width +
							    head * head_width +
							    output_column] :
							0;
					const uint16_t weight_low_second =
						key_low + 1u < arguments.rows &&
								output_column <
									head_width ?
							qkv[(size_t)(key_low +
								     1u) *
								    qkv_stride +
							    VISION_GATE_COMPONENTS *
								    arguments.input_width +
							    head * head_width +
							    output_column] :
							0;
					const uint16_t weight_high_first =
						key_high < arguments.rows &&
								output_column <
									head_width ?
							qkv[(size_t)key_high *
								    qkv_stride +
							    VISION_GATE_COMPONENTS *
								    arguments.input_width +
							    head * head_width +
							    output_column] :
							0;
					const uint16_t weight_high_second =
						key_high + 1u < arguments.rows &&
								output_column <
									head_width ?
							qkv[(size_t)(key_high +
								     1u) *
								    qkv_stride +
							    VISION_GATE_COMPONENTS *
								    arguments.input_width +
							    head * head_width +
							    output_column] :
							0;
					vision_attention_mma_product(
						accumulated[group],
						input_low_low, input_high_low,
						input_low_high, input_high_high,
						vision_attention_mma_pair(
							weight_low_first,
							weight_low_second),
						vision_attention_mma_pair(
							weight_high_first,
							weight_high_second));
				}
			}
		}
		__syncthreads();
	}
	if (threadIdx.x < VISION_ATTENTION_QUERY_TILE) {
		const uint32_t query_row = threadIdx.x;
		const float total =
			(denominator_quad[query_row][0] +
			 denominator_quad[query_row][VISION_HALF_DIVISOR]) +
			(denominator_quad[query_row][1] +
			 denominator_quad[query_row]
					 [VISION_ATTENTION_SUM_LANES - 1u]);
		row_scale[query_row] = 1.0f / total;
	}
	__syncthreads();
	uint16_t *const destination =
		(uint16_t *)(arena + arguments.destination_offset);
#pragma unroll
	for (uint32_t group = 0; group < VISION_ATTENTION_MMA_OUTPUT_GROUPS;
	     group++) {
#pragma unroll
		for (uint32_t result = 0; result < VISION_ATTENTION_MMA_RESULTS;
		     result++) {
			const uint32_t query_row =
				lane_group +
				result / VISION_ATTENTION_MMA_PAIR_VALUES *
					VISION_ATTENTION_MMA_HALF_ROWS;
			const uint32_t column =
				group * VISION_ATTENTION_MMA_GROUP_COLUMNS +
				warp * VISION_ATTENTION_MMA_COLUMNS +
				inner_pair +
				result % VISION_ATTENTION_MMA_PAIR_VALUES;
			if (first_query + query_row < arguments.rows &&
			    column < head_width) {
				destination[(size_t)(first_query + query_row) *
						    arguments.input_width +
					    head * head_width + column] =
					vision_bfloat(
						accumulated[group][result] *
						row_scale[query_row]);
			}
		}
	}
}

extern "C" __global__ void vision_add(struct vision_arguments arguments,
				      unsigned char *arena)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < arguments.rows * arguments.output_width) {
		const float first = vision_float(
			((const uint16_t *)(arena +
					    arguments.source_offset))[index]);
		const float second = vision_float((
			(const uint16_t *)(arena +
					   arguments.auxiliary_offset))[index]);
		((uint16_t *)(arena + arguments.destination_offset))[index] =
			vision_bfloat(first + second);
	}
}

extern "C" __global__ void vision_silu(struct vision_arguments arguments,
				       unsigned char *arena)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < arguments.rows * arguments.output_width) {
		const uint32_t row = index / arguments.output_width;
		const uint32_t column = index % arguments.output_width;
		const uint16_t *const values =
			(const uint16_t *)(arena + arguments.source_offset) +
			(size_t)row * arguments.input_width + column;
		const float gate = vision_float(values[0]);
		const float activated = vision_float(
			vision_bfloat(gate / (1.0f + expf(-gate))));
		((uint16_t *)(arena + arguments.destination_offset))[index] =
			vision_bfloat(
				activated *
				vision_float(values[arguments.output_width]));
	}
}

extern "C" __global__ void vision_unfold(struct vision_arguments arguments,
					 unsigned char *arena)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < arguments.rows * arguments.output_width) {
		const uint32_t ratio = arguments.flags;
		const uint32_t cell_width =
			(arguments.grid_width + ratio - 1u) / ratio;
		const uint32_t cell = index / arguments.output_width;
		const uint32_t column = index % arguments.output_width;
		const uint32_t channel = column / (ratio * ratio);
		const uint32_t delta = column % (ratio * ratio);
		const uint32_t row =
			(cell / cell_width) * ratio + delta / ratio;
		const uint32_t pixel =
			(cell % cell_width) * ratio + delta % ratio;
		((uint16_t *)(arena + arguments.destination_offset))[index] =
			row < arguments.grid_height &&
					pixel < arguments.grid_width ?
				((const uint16_t *)(arena +
						    arguments.source_offset))
					[((size_t)row * arguments.grid_width +
					  pixel) *
						 arguments.input_width +
					 channel] :
				0;
	}
}

extern "C" __global__ void vision_gelu(struct vision_arguments arguments,
				       unsigned char *arena)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < arguments.rows * arguments.output_width) {
		const float value = vision_float(
			((const uint16_t *)(arena +
					    arguments.source_offset))[index]);
		((uint16_t *)(arena + arguments.destination_offset))[index] =
			vision_bfloat(
				(VISION_HALF * value) *
				(1.0f +
				 erff(value * VISION_GELU_INVERSE_SQRT_TWO)));
	}
}

extern "C" __global__ void vision_merge(struct vision_arguments arguments,
					unsigned char *arena)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < arguments.rows * arguments.output_width) {
		const uint32_t token = index / arguments.output_width;
		const uint32_t channel = index % arguments.input_width;
		const uint32_t within_row =
			token == 0 ? 0 :
				     (token - 1u) % (arguments.grid_width + 1u);
		uint64_t offset = arguments.source_offset;
		uint32_t cell = 0;
		if (token == 0) {
			offset = arguments.weights_offset;
		} else if (token + 1u == arguments.rows) {
			offset = arguments.auxiliary_offset;
		} else if (within_row == arguments.grid_width) {
			offset = arguments.bias_offset;
		} else {
			cell = ((token - 1u) / (arguments.grid_width + 1u)) *
				       arguments.grid_width +
			       within_row;
		}
		((uint16_t *)(arena + arguments.destination_offset))[index] = ((
			const uint16_t *)(arena + offset))
			[(size_t)cell * arguments.input_width + channel];
	}
}

extern "C" __global__ void vision_step_merge(struct vision_arguments arguments,
					     unsigned char *arena)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t row = index / arguments.output_width;
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.auxiliary_offset);
	if (row >= arguments.rows || row >= step->row_count) {
		return;
	}
	const struct step_row_record value =
		((const struct step_row_record *)(step + 1))[row];
	if ((value.flags & STEP_ROW_IMAGE) == 0) {
		return;
	}
	const struct vision_session *const session =
		(const struct vision_session
			 *)(arena + arguments.weights_offset +
			    (uint64_t)value.session_slot *
				    VISION_SESSION_METADATA_BYTES);
	for (uint32_t image = 0;
	     image < session->image_count && image < VISION_MAXIMUM_IMAGES;
	     image++) {
		const struct vision_span span = session->spans[image];
		if ((uint32_t)value.position >= span.token_begin &&
		    (uint32_t)value.position - span.token_begin <
			    span.token_count) {
			const uint64_t embedding_row =
				(uint64_t)value.session_slot *
					VISION_MAXIMUM_IMAGES *
					VISION_MAXIMUM_TOKENS +
				span.embedding_row + (uint32_t)value.position -
				span.token_begin;
			((uint16_t *)(arena +
				      arguments.destination_offset))[index] =
				((const uint16_t *)(arena +
						    arguments.source_offset))
					[embedding_row * arguments.input_width +
					 index % arguments.input_width];
			return;
		}
	}
}
