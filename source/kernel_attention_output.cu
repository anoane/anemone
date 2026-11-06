/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_attention_output.cu: the attention site's output span in two launches: the output groups (attn.wo_a) with
 * the quantization of what they give, and the output projection (attn.wo_b) with the site's post. Device code only,
 * built once per architecture into the module "kernel_attention_output", and for class 120 native, as the trunk's
 * products are, into "kernel_attention_output_native".
 *
 * Contract: include/anemone/attention_output_records.h: every bit is the separate chain's (kernel_trunk_linear.cu,
 * kernel_activation_quantize.cu, the post of kernel_hyper_connection.cu). Here:
 *   - A tile of 16 outputs is computed by S warps as a block of the separate product computes it: the same lanes
 *     (linear_lane_weights, the fed rows), the same chains (LINEAR_CHAINS; linear_thread_chains on class 120's
 *     emulation) and the same sum of the slices (linear_sum_of_slices), rounded once to bfloat16. A block of the
 *     groups holds its two tiles as two groups of S warps, as gate with up shares a block (kernel_expert_linear.cu):
 *     each group has its own slots, powers, scratch and round cache, and both pass the same barriers.
 *   - The quantization of a row's 32 outputs is one warp's, lane i output i: the quantizer's own warp function
 *     (activation_quantize_block of kernel_activation_quantize.cuh) for that block.
 *   - The post of one coordinate of one row is one thread's, with the operations of hyper_post in its order: the four
 *     streams read before any is written; for each target the comb products added from +0 in source order, then post
 *     times the output plus that sum, each operation rounded on its own (no fused multiply-add); then bfloat16. What
 *     the post reads besides the product (the tile's streams, post, comb) the card copies into shared memory while
 *     the product runs (cp.async); a thread waits for its copies before the barrier behind the product.
 *   - No atomic operation; nothing depends on timing. Every thread of a block passes every barrier.
 */
#include "kernel_activation_quantize.cuh"

#include "anemone/attention_output_records.h"

#define GROUP_TILES ATTENTION_GROUP_TILES
#define STREAMS MODEL_STREAMS
#define HIDDEN MODEL_HIDDEN
#define MOST_ROWS LINEAR_MAXIMUM_ROWS
#define TILE_RESULTS        \
	(THREADS_PER_WARP * \
	 ELEMENTS_PER_LANE) /* of a tile: 16 outputs by 8 columns */

/* Result `result` of a tile, element result % 4 of lane result / 4, as the product's store reads it: the column it
 * belongs to (the row that column carries) and its output of the tile. */
struct tile_result {
	uint32_t lane_index, element, row, tile_output;
};
__device__ static __forceinline__ struct tile_result
tile_result_of(uint32_t result)
{
	const uint32_t lane_index =
			       result / ELEMENTS_PER_LANE % THREADS_PER_WARP,
		       element = result % ELEMENTS_PER_LANE;
	return (struct tile_result){
		lane_index, element,
		lane_index % LANES_PER_LANE_GROUP * COLUMNS_PER_LANE +
			element % COLUMNS_PER_LANE,
		lane_index / LANES_PER_LANE_GROUP +
			element / COLUMNS_PER_LANE * OUTPUTS_PER_HALF_TILE
	};
}

/* grid (output_size / 32), block of 64 S threads, 64 bytes of shared memory per thread. */
template <uint32_t slice_count>
__device__ __forceinline__ void
attention_groups(struct attention_groups_arguments arguments,
		 unsigned char *arena)
{
	extern __shared__ float slice_chains[];
	__shared__ uint16_t block_values
		[MOST_ROWS]
		[VALUES_PER_BLOCK]; /* each row's 32 outputs of the block */
	const uint32_t group_threads = slice_count * THREADS_PER_WARP;
	const uint32_t thread = threadIdx.x; /* LINEAR_CHAINS's */
	const uint32_t group = thread / group_threads,
		       group_thread = thread % group_threads;
	const uint32_t warp = group_thread / THREADS_PER_WARP,
		       lane_index = thread % THREADS_PER_WARP;
	const uint32_t tile = blockIdx.x * GROUP_TILES + group,
		       row_count = arguments.row_count;
	const size_t segment =
		tile / (arguments.output_size / OUTPUTS_PER_TILE /
			arguments.segment_count);
	LINEAR_DEVICE_CHECK(blockDim.x == GROUP_TILES * group_threads);
	LINEAR_THREAD_CACHE(fed_cache,
			    GROUP_TILES * slice_count * THREADS_PER_WARP);
#if LINEAR_THREAD_SUMS /* the exact profile on class 120: column c carries row c */
	(void)warp;
	const size_t row_bytes = (size_t)arguments.input_size *
				 LINEAR_BFLOAT16_BYTES; /* of a segment */
	linear_thread_chains<LINEAR_WEIGHTS_E4M3, LINEAR_FORM_BFLOAT16_ROWS>(
		arena + arguments.weights_offset, arguments.output_size,
		arguments.input_size, tile,
		arena + arguments.values_offset + segment * row_bytes,
		arguments.segment_count * row_bytes, NULL, 0,
		LINEAR_THREAD_COLUMN_VECTORS_LOW,
		LINEAR_THREAD_COLUMN_VECTORS_HIGH, row_count, group_thread,
		group_threads,
		slice_chains + (size_t)blockDim.x * ELEMENTS_PER_LANE +
			(size_t)group * group_threads *
				LINEAR_THREAD_SCRATCH_FLOATS,
		fed_cache + (size_t)group * group_threads *
				    LINEAR_THREAD_CACHE_WORDS,
		slice_chains +
			(size_t)group * group_threads * ELEMENTS_PER_LANE);
#else
	(void)fed_cache;
	const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP,
		       lane_place = lane_index % LANES_PER_LANE_GROUP;
	const uint32_t load_count = arguments.input_size / VALUES_PER_LOAD;
	const size_t fed_row = lane_group < row_count ? lane_group : 0;
	struct linear_lane lane;
	lane.column_count = row_count;
	linear_lane_weights<LINEAR_WEIGHTS_E4M3>(
		&lane, arena + arguments.weights_offset, arguments.output_size,
		arguments.input_size, tile, lane_index);
	lane.fed = arena + arguments.values_offset +
		   ((fed_row * arguments.segment_count + segment) *
		    arguments.input_size) *
			   LINEAR_BFLOAT16_BYTES;
	lane.exponents[0] =
		lane.fed; /* not read: rows that are not quantized have no exponents */
	lane.exponents[1] = lane.fed;
	LINEAR_CHAINS(LINEAR_WEIGHTS_E4M3, LINEAR_FORM_BFLOAT16_ROWS, warp,
		      load_count,
		      slice_chains + (size_t)thread * ELEMENTS_PER_LANE);
#endif
	__syncthreads();
	uint16_t *const groups = (uint16_t *)(arena + arguments.groups_offset);
	for (uint32_t result = threadIdx.x; result < GROUP_TILES * TILE_RESULTS;
	     result += blockDim.x) {
		const uint32_t result_group = result / TILE_RESULTS;
		const struct tile_result of = tile_result_of(result);
		const uint32_t output = result_group * OUTPUTS_PER_TILE +
					of.tile_output; /* of the block's 32 */
		if (of.row < row_count) {
			const uint16_t bits = (uint16_t)linear_bfloat16_bits(
				linear_sum_of_slices<slice_count>(
					slice_chains,
					result_group * slice_count,
					of.lane_index, of.element));
			groups[(size_t)of.row * arguments.output_size +
			       (size_t)blockIdx.x * VALUES_PER_BLOCK + output] =
				bits;
			block_values[of.row][output] = bits;
		}
	}
	__syncthreads();
	const size_t block_count =
		arguments.output_size / VALUES_PER_BLOCK; /* of a row */
	for (uint32_t row = threadIdx.x / THREADS_PER_WARP; row < row_count;
	     row += blockDim.x / THREADS_PER_WARP) {
		activation_quantize_block(
			block_values[row][lane_index], blockIdx.x, lane_index,
			arena + arguments.codes_offset +
				(size_t)row * arguments.output_size,
			arena + arguments.exponents_offset + row * block_count,
			(uint16_t *)(arena + arguments.operands_offset) +
				(size_t)row * arguments.output_size);
	}
}

/* What the post of a tile reads of a row besides the product: the tile's 16 coordinates of the four streams, the
 * row's post and comb, copied by the card into the block's shared memory while the product runs (cp.async, 16 bytes a
 * copy: CHUNKS_PER_ROW of them a row, each member a whole number of copies). */
struct post_inputs {
	uint16_t streams[STREAMS][OUTPUTS_PER_TILE];
	float post[STREAMS];
	float comb[STREAMS * STREAMS];
};
#define ASYNC_COPY_BYTES 16u
#define STREAM_COPIES                               \
	(OUTPUTS_PER_TILE * LINEAR_BFLOAT16_BYTES / \
	 ASYNC_COPY_BYTES) /* of a stream */
#define POST_COPY (STREAMS * STREAM_COPIES) /* the row's post: one copy */
#define COMB_FIRST_COPY (POST_COPY + 1u)
#define CHUNKS_PER_ROW ((uint32_t)sizeof(struct post_inputs) / ASYNC_COPY_BYTES)
#define FLOATS_PER_COPY (ASYNC_COPY_BYTES / (uint32_t)sizeof(float))
#define VALUES_PER_COPY (ASYNC_COPY_BYTES / LINEAR_BFLOAT16_BYTES)
static_assert(
	sizeof(struct post_inputs) % ASYNC_COPY_BYTES == 0 &&
		CHUNKS_PER_ROW ==
			COMB_FIRST_COPY + STREAMS * STREAMS / FLOATS_PER_COPY,
	"the post's inputs of a row are whole copies: the streams', the post's, the comb's");

/* The copies of the post's inputs of the block's rows begun, a thread a copy (their global addresses 16-byte aligned:
 * the offsets of a record are, and so are the rows and the tile's coordinates). */
__device__ static __forceinline__ void
post_inputs_copy(const struct attention_projection_arguments *arguments,
		 const unsigned char *arena, struct post_inputs *row_inputs)
{
	for (uint32_t chunk = threadIdx.x;
	     chunk < arguments->row_count * CHUNKS_PER_ROW;
	     chunk += blockDim.x) {
		const size_t row = chunk / CHUNKS_PER_ROW;
		const uint32_t part = chunk % CHUNKS_PER_ROW;
		const unsigned char *source = NULL;
		void *destination = NULL;
		if (part < POST_COPY) { /* copy part % 2 of stream part / 2 */
			const uint32_t stream = part / STREAM_COPIES,
				       first = part % STREAM_COPIES *
					       VALUES_PER_COPY;
			source = arena + arguments->streams_offset +
				 ((row * STREAMS + stream) * HIDDEN +
				  (size_t)blockIdx.x * OUTPUTS_PER_TILE +
				  first) *
					 LINEAR_BFLOAT16_BYTES;
			destination = &row_inputs[row].streams[stream][first];
		} else if (part == POST_COPY) {
			source = arena + arguments->post_offset +
				 row * STREAMS * sizeof(float);
			destination = row_inputs[row].post;
		} else { /* copy part - COMB_FIRST_COPY of comb */
			const uint32_t first =
				(part - COMB_FIRST_COPY) * FLOATS_PER_COPY;
			source = arena + arguments->comb_offset +
				 (row * STREAMS * STREAMS + first) *
					 sizeof(float);
			destination = &row_inputs[row].comb[first];
		}
		asm volatile(
			"cp.async.ca.shared.global [%0], [%1], 16;"
			:
			: "r"((uint32_t)__cvta_generic_to_shared(destination)),
			  "l"(source)
			: "memory");
	}
}

/* grid (output_size / 16), block of 32 S threads, 64 bytes of shared memory per thread. */
template <uint32_t slice_count>
__device__ __forceinline__ void
attention_projection(struct attention_projection_arguments arguments,
		     unsigned char *arena)
{
	extern __shared__ float slice_chains[];
	__shared__ float tile_values
		[MOST_ROWS]
		[OUTPUTS_PER_TILE]; /* each row's outputs of the tile, rounded */
	__shared__ __align__(
		ASYNC_COPY_BYTES) struct post_inputs row_inputs[MOST_ROWS];
	post_inputs_copy(&arguments, arena, row_inputs);
	const uint32_t thread = threadIdx.x; /* LINEAR_CHAINS's */
	const uint32_t warp = thread / THREADS_PER_WARP,
		       lane_index = thread % THREADS_PER_WARP;
	const uint32_t tile = blockIdx.x, row_count = arguments.row_count;
	LINEAR_DEVICE_CHECK(blockDim.x == slice_count * THREADS_PER_WARP &&
			    arguments.output_size == HIDDEN);
#if LINEAR_THREAD_SUMS /* the exact profile on class 120: column c carries row c */
	(void)warp;
	(void)lane_index;
	LINEAR_THREAD_CACHE(fed_cache, slice_count * THREADS_PER_WARP);
	linear_thread_chains<LINEAR_WEIGHTS_E4M3, LINEAR_TRUNK_FORM>(
		arena + arguments.weights_offset, arguments.output_size,
		arguments.input_size, tile, arena + arguments.operands_offset,
		(size_t)arguments.input_size * LINEAR_BFLOAT16_BYTES, NULL, 0,
		LINEAR_THREAD_COLUMN_VECTORS_LOW,
		LINEAR_THREAD_COLUMN_VECTORS_HIGH, row_count, thread,
		blockDim.x,
		slice_chains + (size_t)blockDim.x * ELEMENTS_PER_LANE,
		fed_cache, slice_chains);
#else
	const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP,
		       lane_place = lane_index % LANES_PER_LANE_GROUP;
	const uint32_t load_count = arguments.input_size / VALUES_PER_LOAD;
	const uint32_t even_column = lane_place * COLUMNS_PER_LANE;
	struct linear_lane lane;
	lane.column_count = row_count;
	linear_lane_weights<LINEAR_WEIGHTS_E4M3>(
		&lane, arena + arguments.weights_offset, arguments.output_size,
		arguments.input_size, tile, lane_index);
	linear_lane_quantized<LINEAR_TRUNK_FORM>(
		&lane, arena + arguments.codes_offset,
		arena + arguments.exponents_offset,
		arena + arguments.operands_offset, arguments.input_size,
		lane_place, lane_group < row_count ? lane_group : 0,
		even_column < row_count ? even_column : 0,
		even_column + 1u < row_count ? even_column + 1u : 0);
	LINEAR_CHAINS(LINEAR_WEIGHTS_E4M3, LINEAR_TRUNK_FORM, warp, load_count,
		      slice_chains + (size_t)thread * ELEMENTS_PER_LANE);
#endif
	asm volatile(
		"cp.async.wait_all;"
		:
		:
		: "memory"); /* the thread's copies of the post's inputs, behind the barrier */
	__syncthreads();
	uint16_t *const output = (uint16_t *)(arena + arguments.output_offset);
	for (uint32_t result = threadIdx.x; result < TILE_RESULTS;
	     result += blockDim.x) {
		const struct tile_result of = tile_result_of(result);
		if (of.row < row_count) {
			const uint32_t bits = linear_bfloat16_bits(
				linear_sum_of_slices<slice_count>(
					slice_chains, 0, of.lane_index,
					of.element));
			output[(size_t)of.row * HIDDEN +
			       (size_t)tile * OUTPUTS_PER_TILE +
			       of.tile_output] = (uint16_t)bits;
			tile_values[of.row][of.tile_output] =
				linear_bfloat16_value(bits);
		}
	}
	__syncthreads();
	for (uint32_t pair = threadIdx.x; pair < row_count * OUTPUTS_PER_TILE;
	     pair += blockDim.x) {
		const size_t row = pair / OUTPUTS_PER_TILE;
		const uint32_t tile_output = pair % OUTPUTS_PER_TILE;
		const float *const post = row_inputs[row].post;
		const float *const comb = row_inputs[row].comb;
		uint16_t *const streams =
			(uint16_t *)(arena + arguments.streams_offset) +
			row * STREAMS * HIDDEN +
			(size_t)tile * OUTPUTS_PER_TILE + tile_output;
		const float value = tile_values[row][tile_output];
		float values[STREAMS];
		for (uint32_t stream = 0; stream < STREAMS; stream++) {
			values[stream] = linear_bfloat16_value(
				row_inputs[row].streams[stream][tile_output]);
		}
		for (uint32_t target = 0; target < STREAMS; target++) {
			float sum = 0.0f;
			for (uint32_t source = 0; source < STREAMS; source++) {
				sum = sum + comb[source * STREAMS + target] *
						    values[source];
			}
			streams[target * HIDDEN] =
				(uint16_t)linear_bfloat16_bits(
					post[target] * value + sum);
		}
	}
}

/* The instance of 16 slices of the groups states its block (1,024 threads) as its launch bound: without it class 80's
 * takes 72 registers, more than the register file holds for such a block. No other instance states one, as the
 * separate products state none: a bound moves the registers the compiler takes (a bound of 512 threads made class
 * 80's instance of 8 slices spill). The build checks that the registers of every instance fit its block. On class
 * 120's emulation an instance beyond the counts of the records is a stub that the host never launches. */
#define GROUPS_BOUNDS_1(threads)
#define GROUPS_BOUNDS_2(threads)
#define GROUPS_BOUNDS_4(threads)
#define GROUPS_BOUNDS_8(threads)
#define GROUPS_BOUNDS_16(threads) __launch_bounds__(threads)
#define LINEAR_SLICE_STAMP(slices)                                             \
	extern "C" __global__ void GROUPS_BOUNDS_##slices(                     \
		slices * GROUP_TILES * THREADS_PER_WARP)                       \
		attention_groups_##slices(                                     \
			struct attention_groups_arguments arguments,           \
			unsigned char *arena)                                  \
	{                                                                      \
		LINEAR_SLICED_BODY(slices, LINEAR_CLASS_120_EXACT_MOST_SLICES, \
				   attention_groups<slices>(arguments, arena)) \
	}                                                                      \
	extern "C" __global__ void attention_projection_##slices(              \
		struct attention_projection_arguments arguments,               \
		unsigned char *arena)                                          \
	{                                                                      \
		LINEAR_SLICED_BODY(slices, LINEAR_CLASS_120_EXACT_MOST_SLICES, \
				   attention_projection<slices>(arguments,     \
								arena))        \
	}
LINEAR_EVERY_SLICE_COUNT
#undef LINEAR_SLICE_STAMP
