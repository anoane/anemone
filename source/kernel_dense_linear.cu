/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_dense_linear.cu: y = W x in float32 on the cores (no tensor-core instruction). Device code only, built once
 * per architecture into the module "kernel_dense_linear". Contract: include/anemone/dense_linear_records.h. Here a
 * block owns 4 outputs (or 1: the narrow form) of every row, so a weight is read once for all rows, with the
 * streaming hint (no launch reads it twice); the rows, which every block reads, stay in the caches. Every instance
 * takes a row through the same operations in the same order. One barrier, which every thread reaches; behind it the
 * threads below outputs times rows each finish one output of one row. A wide instance is the instance of 8 rows on the
 * group of rows of blockIdx.y, its record moved to the group's first row; it reads the weights with the default hint,
 * since every group reads them.
 */
#include "kernel_linear_common.cuh"

#include "anemone/dense_linear_records.h"
#include "anemone/model_layout.h"

#define DENSE_WARPS (DENSE_THREADS_VALUE / THREADS_PER_WARP)
#define DENSE_WORDS_PER_PART 4u                 /* a 16-byte load */
#define DENSE_BFLOAT16_PER_WORD 2u
#define DENSE_FULL_WARP 0xFFFFFFFFu
#define DENSE_FIRST_DISTANCE 16u                /* of the exchanges between lanes, halved down to 1 */
#define DENSE_BLOCKS_PER_MULTIPROCESSOR 2u      /* at least, held by the launch bound: 128 registers a thread */
#define DENSE_NARROW_CHUNKS_AHEAD 4             /* the narrow form's chunks unrolled: their loads leave together */
#define DENSE_STAGED_SUM_BUFFERS 2u             /* the staged form's warp sums: an output's while the next forms */

/* Value `index` of a 16-byte part, widened: a bfloat16 (the even one in the low half of its word) or a float32. */
template <int type>
__device__ __forceinline__ float dense_value(uint4 part, uint32_t index)
{
    const uint32_t words[DENSE_WORDS_PER_PART] = {part.x, part.y, part.z, part.w};
    const uint32_t word =
	    words[type == DENSE_FLOAT32 ? index :
					  index / DENSE_BFLOAT16_PER_WORD];
    return __uint_as_float(type == DENSE_FLOAT32 ? word :
			   index % DENSE_BFLOAT16_PER_WORD == 0 ?
						   word << BITS_PER_HALF :
						   word & UPPER_HALF_MASK);
}

template <int weight_type, uint32_t row_capacity, uint32_t outputs, bool is_wide>
__device__ __forceinline__ void
dense_linear(struct dense_linear_arguments arguments, unsigned char *arena)
{
	static_assert(
		(row_capacity == 1u || row_capacity == LINEAR_MAXIMUM_ROWS) &&
			outputs * row_capacity <= DENSE_THREADS_VALUE &&
			DENSE_WARPS * THREADS_PER_WARP == DENSE_THREADS_VALUE,
		"one row or a launch's most; a finishing thread for each output of each row; whole warps");
	__shared__ float warp_sums[DENSE_WARPS][outputs * row_capacity];
	__shared__ float warp_squares[DENSE_WARPS][row_capacity];
	const uint32_t parts =
		weight_type == DENSE_FLOAT32 ?
			(uint32_t)(sizeof(float) / sizeof(uint16_t)) :
			1u;
	const uint32_t inputs_per_part = DENSE_CHUNK_INPUTS / parts;
	const uint32_t chunk_count = arguments.input_size / DENSE_CHUNK_INPUTS;
	const bool is_row_scaled = (arguments.flags & DENSE_ROW_SCALE) != 0;
	const uint4 *weights =
		(const uint4 *)(arena + arguments.weights_offset) +
		(size_t)blockIdx.x * outputs * chunk_count * parts;
	const uint4 *rows = (const uint4 *)(arena + arguments.rows_offset);
	float sums[outputs][row_capacity] = {};
	float squares[row_capacity] = {};
	LINEAR_DEVICE_CHECK(blockDim.x == DENSE_THREADS_VALUE &&
			    gridDim.x * outputs == arguments.output_size &&
			    (is_wide || gridDim.y == 1) &&
			    arguments.row_count != 0 &&
			    arguments.row_count <= row_capacity);
	const struct step_block *step =
		(const struct step_block *)(arena + arguments.step_offset);
	if ((arguments.flags & DENSE_STEP_ROWS) !=
	    0) { /* the head: a step without a row of logits reads nothing */
		uint32_t flagged = 0;
		for (uint32_t row = 0;
		     row < step->row_count && row < arguments.row_count;
		     row++) {
			flagged |= ((const struct step_row_record *)(step +
								     1))[row]
					   .flags &
				   STEP_ROW_LOGITS;
		}
		if (flagged ==
		    0) { /* the whole block leaves, before its one barrier */
			return;
		}
	}
    /* the thin narrow form's loads leave together; a wide launch has the blocks to hide them (a pragma expands no
     * macro) */
    constexpr int chunks_unrolled = outputs == 1u && !is_wide ? DENSE_NARROW_CHUNKS_AHEAD : 1;
#pragma unroll chunks_unrolled
    for (uint32_t chunk = threadIdx.x; chunk < chunk_count;
	 chunk += DENSE_THREADS_VALUE) {
	    uint4 values[row_capacity];
	    for (uint32_t row = 0; row < row_capacity; row++) {
		    values[row] =
			    rows[(size_t)(row < arguments.row_count ? row : 0) *
					 chunk_count +
				 chunk];
	    }
	    for (uint32_t part = 0; part < parts; part++) {
		    uint4 part_weights[outputs];
		    for (uint32_t output = 0; output < outputs;
			 output++) { /* a wide launch's groups all read them */
			    const uint4 *const weight =
				    weights +
				    ((size_t)output * chunk_count + chunk) *
					    parts +
				    part;
			    part_weights[output] = is_wide ? __ldg(weight) :
							     __ldcs(weight);
		    }
		    for (uint32_t index = 0; index < inputs_per_part; index++) {
			    for (uint32_t row = 0; row < row_capacity; row++) {
				    const float value =
					    dense_value<DENSE_BFLOAT16>(
						    values[row],
						    part * inputs_per_part +
							    index);
				    squares[row] = is_row_scaled ?
							   fmaf(value, value,
								squares[row]) :
							   squares[row];
				    for (uint32_t output = 0; output < outputs;
					 output++) {
					    sums[output][row] = fmaf(
						    dense_value<weight_type>(
							    part_weights[output],
							    index),
						    value, sums[output][row]);
				    }
			    }
		    }
	    }
    }
    for (uint32_t distance = DENSE_FIRST_DISTANCE; distance != 0; distance >>= 1u) {
        for (uint32_t row = 0; row < row_capacity; row++) {
            squares[row] = is_row_scaled ? squares[row] + __shfl_xor_sync(DENSE_FULL_WARP, squares[row], distance)
                                         : squares[row];
            for (uint32_t output = 0; output < outputs; output++) {
                sums[output][row] = sums[output][row] + __shfl_xor_sync(DENSE_FULL_WARP, sums[output][row], distance);
            }
        }
    }
    for (uint32_t row = 0; threadIdx.x % THREADS_PER_WARP == 0 && row < row_capacity; row++) {
        warp_squares[threadIdx.x / THREADS_PER_WARP][row] = squares[row];
        for (uint32_t output = 0; output < outputs; output++) {
            warp_sums[threadIdx.x / THREADS_PER_WARP][output * row_capacity + row] = sums[output][row];
        }
    }
    __syncthreads();
    const uint32_t output = threadIdx.x / row_capacity, row = threadIdx.x % row_capacity;
    if (output >= outputs || row >= arguments.row_count ||
	((arguments.flags & DENSE_STEP_ROWS) != 0 &&
	 (row >= step->row_count ||
	  (((const struct step_row_record *)(step + 1))[row].flags &
	   STEP_ROW_LOGITS) == 0))) {
	    return;
    }
    float sum = 0.0f, square_sum = 0.0f;
    for (uint32_t warp = 0; warp < DENSE_WARPS; warp++) {
        sum = sum + warp_sums[warp][threadIdx.x];
        square_sum = square_sum + warp_squares[warp][row];
    }
    sum = is_row_scaled ? sum * rsqrtf(square_sum * (1.0f / (float)arguments.input_size) + arguments.epsilon) : sum;
    const size_t index = (size_t)row * arguments.output_size + (size_t)blockIdx.x * outputs + output;
    if (arguments.output_type == DENSE_BFLOAT16) {
	    ((uint16_t *)(arena + arguments.output_offset))[index] =
		    (uint16_t)linear_bfloat16_bits(
			    linear_bfloat16_value(linear_bfloat16_bits(sum)) *
			    arguments.output_scale);
    } else {
	    ((float *)(arena + arguments.output_offset))[index] =
		    sum * arguments.output_scale;
    }
}

/* Two blocks a multiprocessor: one streams weights while the other adds its sums up; the build fails on a spill. */
#define DENSE_STAMP(name, type, rows, outputs, form)                        \
	extern "C" __global__ void __launch_bounds__(                       \
		DENSE_THREADS_VALUE, DENSE_BLOCKS_PER_MULTIPROCESSOR)       \
		dense_linear_##name##_##rows##form(                         \
			struct dense_linear_arguments arguments,            \
			unsigned char *arena)                               \
	{                                                                   \
		dense_linear<type, rows, outputs, false>(arguments, arena); \
	}
DENSE_EVERY_INSTANCE
#undef DENSE_STAMP

/* A wide launch's block: the instance of 8 rows on the rows of group blockIdx.y, the record moved to the group's
 * first row, as a launch of those rows alone takes it (the host gives no step rows). */
template <int weight_type, uint32_t outputs>
__device__ __forceinline__ void
dense_linear_wide(struct dense_linear_arguments arguments, unsigned char *arena)
{
	const uint32_t first_row = blockIdx.y * LINEAR_MAXIMUM_ROWS;
	LINEAR_DEVICE_CHECK(
		arguments.row_count != 0 &&
		(arguments.flags & DENSE_STEP_ROWS) == 0 &&
		gridDim.y == (arguments.row_count + LINEAR_MAXIMUM_ROWS - 1u) /
				     LINEAR_MAXIMUM_ROWS);
	/* DENSE_CONE_ROWS: a group whose rows all skip leaves at once (every thread decides alike) */
	if ((arguments.flags & DENSE_CONE_ROWS) != 0) {
		const struct step_row_record *const step_rows =
			(const struct step_row_record
				 *)(arena + arguments.step_offset +
				    STEP_BLOCK_BYTES_VALUE) +
			first_row;
		const uint32_t decoder_layer =
			arguments.cone_layer - MODEL_DECODER_FIRST_LAYER;
		bool is_read = false;
		for (uint32_t row = 0; row < LINEAR_MAXIMUM_ROWS &&
				       first_row + row < arguments.row_count;
		     row++)
			is_read = is_read ||
				  !STEP_ROW_CONE_TRUNK_SKIPS(
					  step_rows[row].flags, decoder_layer);
		if (!is_read)
			return;
	}
    arguments.rows_offset += (uint64_t)first_row * arguments.input_size * sizeof(uint16_t);
    arguments.output_offset += (uint64_t)first_row * arguments.output_size *
			       (arguments.output_type == DENSE_BFLOAT16 ?
					sizeof(uint16_t) :
					sizeof(float));
    arguments.row_count =
	    min(arguments.row_count - first_row, LINEAR_MAXIMUM_ROWS);
    dense_linear<weight_type, LINEAR_MAXIMUM_ROWS, outputs, true>(arguments,
								  arena);
}

/* --dense_staged (dense_linear_records.h): the wide instance of S outputs a block with the group's 8 rows staged once
 * in shared memory; each output in turn by every thread: its chains (chunks t, t + 256, ... from +0, inputs in order,
 * fmaf(weight, value)), the exchanges at distance 16 down to 1 and the warps' sums from +0 in warp order, those of the
 * wide instance's block of that output alone. warp_sums alternates two buffers: one barrier an output. */
template <int weight_type, uint32_t block_outputs>
__device__ __forceinline__ void
dense_linear_staged(struct dense_linear_arguments arguments,
		    unsigned char *arena)
{
    extern __shared__ __align__(16) uint4 dense_staged_rows[];
    __shared__ float warp_sums[DENSE_STAGED_SUM_BUFFERS][DENSE_WARPS]
			      [LINEAR_MAXIMUM_ROWS];
    const uint32_t first_row = blockIdx.y * LINEAR_MAXIMUM_ROWS;
    LINEAR_DEVICE_CHECK(
	    blockDim.x == DENSE_THREADS_VALUE && arguments.row_count != 0 &&
	    (arguments.flags & (DENSE_STEP_ROWS | DENSE_ROW_SCALE)) == 0 &&
	    gridDim.x * block_outputs == arguments.output_size &&
	    arguments.input_size <= DENSE_STAGED_MOST_INPUTS &&
	    gridDim.y == (arguments.row_count + LINEAR_MAXIMUM_ROWS - 1u) /
				 LINEAR_MAXIMUM_ROWS);
    if ((arguments.flags & DENSE_CONE_ROWS) !=
	0) { /* the wide instance's group test */
	    const struct step_row_record *const step_rows =
		    (const struct step_row_record *)(arena +
						     arguments.step_offset +
						     STEP_BLOCK_BYTES_VALUE) +
		    first_row;
	    const uint32_t decoder_layer =
		    arguments.cone_layer - MODEL_DECODER_FIRST_LAYER;
	    bool is_read = false;
	    for (uint32_t row = 0; row < LINEAR_MAXIMUM_ROWS &&
				   first_row + row < arguments.row_count;
		 row++)
		    is_read = is_read ||
			      !STEP_ROW_CONE_TRUNK_SKIPS(step_rows[row].flags,
							 decoder_layer);
	    if (!is_read)
		    return;
    }
    const uint32_t row_count =
	    min(arguments.row_count - first_row, LINEAR_MAXIMUM_ROWS);
    const uint32_t parts =
	    weight_type == DENSE_FLOAT32 ?
		    (uint32_t)(sizeof(float) / sizeof(uint16_t)) :
		    1u;
    const uint32_t inputs_per_part = DENSE_CHUNK_INPUTS / parts;
    const uint32_t chunk_count = arguments.input_size / DENSE_CHUNK_INPUTS;
    const uint4 *const rows = (const uint4 *)(arena + arguments.rows_offset) + (size_t)first_row * chunk_count;
    for (uint32_t piece = threadIdx.x;
	 piece < LINEAR_MAXIMUM_ROWS * chunk_count;
	 piece +=
	 DENSE_THREADS_VALUE) { /* a column from row_count on reads the group's row 0, as the instance */
	    const uint32_t row = piece / chunk_count,
			   chunk = piece - row * chunk_count;
	    dense_staged_rows[piece] =
		    rows[(size_t)(row < row_count ? row : 0) * chunk_count +
			 chunk];
    }
    __syncthreads();
    for (uint32_t own = 0; own < block_outputs; own++) {
        const uint32_t output = blockIdx.x * block_outputs + own;
        const uint4 *const weights = (const uint4 *)(arena + arguments.weights_offset) +
                                     (size_t)output * chunk_count * parts;
	float sums[LINEAR_MAXIMUM_ROWS] = {};
	for (uint32_t chunk = threadIdx.x; chunk < chunk_count;
	     chunk += DENSE_THREADS_VALUE) {
		uint4 values[LINEAR_MAXIMUM_ROWS];
		for (uint32_t row = 0; row < LINEAR_MAXIMUM_ROWS; row++)
			values[row] =
				dense_staged_rows[row * chunk_count + chunk];
		for (uint32_t part = 0; part < parts; part++) {
			const uint4 part_weights =
				__ldg(weights + (size_t)chunk * parts + part);
			for (uint32_t index = 0; index < inputs_per_part;
			     index++)
				for (uint32_t row = 0;
				     row < LINEAR_MAXIMUM_ROWS; row++)
					sums[row] = fmaf(
						dense_value<weight_type>(
							part_weights, index),
						dense_value<DENSE_BFLOAT16>(
							values[row],
							part * inputs_per_part +
								index),
						sums[row]);
		}
	}
	for (uint32_t distance = DENSE_FIRST_DISTANCE; distance != 0; distance >>= 1u)
		for (uint32_t row = 0; row < LINEAR_MAXIMUM_ROWS; row++)
			sums[row] = sums[row] + __shfl_xor_sync(DENSE_FULL_WARP,
								sums[row],
								distance);
	float (*const buffer)[LINEAR_MAXIMUM_ROWS] =
		warp_sums[own % DENSE_STAGED_SUM_BUFFERS];
	for (uint32_t row = 0;
	     threadIdx.x % THREADS_PER_WARP == 0 && row < LINEAR_MAXIMUM_ROWS;
	     row++)
		buffer[threadIdx.x / THREADS_PER_WARP][row] = sums[row];
        __syncthreads();
        const uint32_t row = threadIdx.x;
        if (row < row_count) {
            float sum = 0.0f;
            for (uint32_t warp = 0; warp < DENSE_WARPS; warp++)
                sum = sum + buffer[warp][row];
            const size_t index = (size_t)(first_row + row) * arguments.output_size + output;
	    if (arguments.output_type == DENSE_BFLOAT16)
		    ((uint16_t *)(arena + arguments.output_offset))[index] =
			    (uint16_t)linear_bfloat16_bits(
				    linear_bfloat16_value(
					    linear_bfloat16_bits(sum)) *
				    arguments.output_scale);
            else
                ((float *)(arena + arguments.output_offset))[index] = sum * arguments.output_scale;
        }
    }
}

/* Two blocks a multiprocessor (two staged groups of 80 KiB fill one's shared memory). */
#define DENSE_STAGED_STAMP(name, type, outputs)                       \
	extern "C" __global__ void __launch_bounds__(                 \
		DENSE_THREADS_VALUE, DENSE_BLOCKS_PER_MULTIPROCESSOR) \
		dense_linear_##name##_staged_##outputs(               \
			struct dense_linear_arguments arguments,      \
			unsigned char *arena)                         \
	{                                                             \
		dense_linear_staged<type, outputs>(arguments, arena); \
	}
DENSE_EVERY_STAGED_INSTANCE
#undef DENSE_STAGED_STAMP

/* A wide instance: blocks of them a multiprocessor at least, the bound of its registers. */
#define DENSE_WIDE_STAMP(name, type, outputs, blocks)                     \
	extern "C" __global__ void __launch_bounds__(DENSE_THREADS_VALUE, \
						     blocks)              \
		dense_linear_##name##_wide_##outputs(                     \
			struct dense_linear_arguments arguments,          \
			unsigned char *arena)                             \
	{                                                                 \
		dense_linear_wide<type, outputs>(arguments, arena);       \
	}
DENSE_EVERY_WIDE_INSTANCE
#undef DENSE_WIDE_STAMP
