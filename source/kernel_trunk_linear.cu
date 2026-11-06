/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_trunk_linear.cu: y = W x for a matrix of 8-bit weights (WEIGHT_FORMAT_E4M3_SCALED) and 1 to 8
 * rows (1 to 1,024 in the wide instances), with quantized rows or with bfloat16 rows, and the hidden step of an expert
 * whose gate and up are two such launches. Device code only, built once per architecture into the module
 * "kernel_trunk_linear".
 *
 * Contract: include/anemone/linear_records.h (the records, the arrays, the order of additions, the scale codes,
 * what a bfloat16 row keeps, the hidden step, the launches); kernel_linear_common.cuh (the instruction and its
 * operands). Here:
 *   - A block is one tile of 16 outputs, a warp one slice of the input width. The weights are operand A of the
 *     instruction and the rows are the columns of operand B: column c carries row c. A column from row_count on
 *     carries row 0 again and its results are stored nowhere: 1 to 8 rows cost the same instructions.
 *   - Two matrices of quantized rows in one launch: the blocks of the second follow those of the first, and a block
 *     does for its tile of its matrix exactly what it does in a launch of that matrix alone.
 *   - A ROW KEEPS ITS BITS whatever rows share its launch and whichever column it takes, by the construction
 *     kernel_expert_linear.cu states: one result is one output of one column, and the additions made for it are
 *     fixed by the matrix, the row and the slice count alone.
 *   - A bfloat16 row is read where it lies, [row][segment_count * input_size] in the order of its values, and each
 *     value becomes a 16-bit float as it is fed. A tile of segment i reads the values from i * input_size on.
 *   - The shared expert of a layer is: trunk_linear_quantized_value for its gate and up (one launch of two
 *     matrices on the same quantized rows), trunk_hidden_entry, activation_quantize_entry on the hidden
 *     vectors, and trunk_linear_quantized_value for its down.
 *   - No atomic operation. A product's block passes the barriers of its ordered slices (more than one slice: see
 *     linear_ordered_chains) and one before its results, which every thread of the block reaches.
 *   - A wide instance (up to LINEAR_WIDE_MOST_ROWS rows) runs the thin instance's body once for each group of
 *     8 rows, on a copy of the record whose arrays begin at the group's first row and whose row_count is the group's:
 *     the same lane setup, chains, barriers and stores, so every output keeps its thin bits. Its grid is the tiles
 *     times n blocks a tile, n from 1 to the groups (trunk_wide_place_of). linear.c gives every group a block (n = the
 *     groups): a tile's groups lie in adjacent blocks, which a card starts together (it hands blocks out in the order
 *     of their index), so its weights come from memory about once whatever the L2 holds, and every multiprocessor has
 *     blocks. A block of several groups passes a barrier between two, so that no warp refills its shared memory while
 *     another still adds the last group's slices, and reads its tile again for each group: from the caches only where
 *     the tiles of the blocks the card holds at once fit its L2.
 *   - The interleaved instance (one slice) turns the loops round: the loads outer, 4 groups inner, each group's chains
 *     its own registers, so that a load of weights is read and placed once for 4 groups; each chain still sees the
 *     thin kernel's operands, instructions and order (trunk_interleaved_chains).
 *   - The shared join (expert_fused_records.h) is the product of the shared expert's down, its block's tile computed
 *     and stored as a block of the product does it; then the block's threads join the tile (expert_join_values,
 *     kernel_expert_join.cuh), its values rounded from the chains as the product rounds what it stores: a block
 *     writes the streams (and the row sums, where it is given their array) at its tile's coordinates alone.
 */
#include "kernel_expert_join.cuh"
#include "kernel_linear_common.cuh"

#define TRUNK_INTERLEAVED_GROUPS 4u             /* their chains fill the 2 KB of shared memory of a one-warp block */
#define TRUNK_CHUNK_ROWS (TRUNK_INTERLEAVED_GROUPS * LINEAR_MAXIMUM_ROWS)
static_assert(
	TRUNK_INTERLEAVED_GROUPS * ELEMENTS_PER_LANE * sizeof(float) <=
		LINEAR_SHARED_BYTES_PER_THREAD,
	"a lane's chains of a chunk fit its thread's share of the dynamic shared memory (linear.c)");
static_assert(TRUNK_CHUNK_ROWS == LINEAR_INTERLEAVED_ROWS,
	      "linear.c counts the blocks of a chunk so");

/* After the barrier of a product block: warp `warp` finishes the results warp, warp + slice_count, .. of every
 * lane: the slices added in order, rounded to bfloat16, stored for the rows in use. output is values
 * [row_count][output_size] of the block's matrix, tile its tile there; thread the thread's index in the block. */
template <uint32_t slice_count>
__device__ __forceinline__ void trunk_store_results(const float *slice_chains, uint32_t row_count,
                                                    uint32_t output_size, uint32_t tile, uint16_t *output,
                                                    uint32_t thread)
{
    const uint32_t warp = thread / THREADS_PER_WARP, lane_index = thread % THREADS_PER_WARP;
    for (uint32_t element = warp; element < ELEMENTS_PER_LANE; element += slice_count) {
        const uint32_t row = lane_index % LANES_PER_LANE_GROUP * COLUMNS_PER_LANE + element % COLUMNS_PER_LANE;
        const uint32_t output_index = tile * OUTPUTS_PER_TILE + lane_index / LANES_PER_LANE_GROUP +
                                      element / COLUMNS_PER_LANE * OUTPUTS_PER_HALF_TILE;
        if (row < row_count) {
            output[(size_t)row * output_size + output_index] = (uint16_t)linear_bfloat16_bits(
                linear_sum_of_slices<slice_count>(slice_chains, 0, lane_index, element));
        }
    }
}

/* grid ((output_size[0] + output_size[1]) / 16), block of 32 S threads, 64 bytes of shared memory per thread. A
 * wide block (is_wide) hands its group the tile it computes and the thread's index, as copies the compiler cannot
 * see through (trunk_linear_quantized_wide); a thin block hands nothing: its body reads blockIdx.x and threadIdx.x
 * as it always did. The release modules hold every thin instance's instructions unchanged (cuobjdump, function by
 * function); in the checked modules of sm_120a the compiler orders some of them otherwise around the device check
 * (the thread's index read before the trap, a register zeroed by a move), no arithmetic changed. */
template <uint32_t slice_count>
__device__ __forceinline__ void
trunk_linear_quantized(struct trunk_linear_quantized_arguments arguments,
		       unsigned char *arena, bool is_wide, uint32_t wide_tile,
		       uint32_t wide_thread)
{
    extern __shared__ float slice_chains[];
    const uint32_t thread = is_wide ? wide_thread : threadIdx.x;
    const uint32_t warp = thread / THREADS_PER_WARP, lane_index = thread % THREADS_PER_WARP;
    const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP, lane_place = lane_index % LANES_PER_LANE_GROUP;
    const uint32_t load_count = arguments.input_size / VALUES_PER_LOAD;
    const uint32_t even_column = lane_place * COLUMNS_PER_LANE;
    const uint32_t first_tiles = arguments.output_size[0] / OUTPUTS_PER_TILE;
    const uint32_t block_tile = is_wide ? wide_tile : blockIdx.x;   /* of the tiles of both matrices */
    const bool is_second = block_tile >= first_tiles;           /* never where output_size[1] is 0 */
    const uint32_t tile = is_second ? block_tile - first_tiles : block_tile;
    const uint32_t output_size = is_second ? arguments.output_size[1] : arguments.output_size[0];
    LINEAR_DEVICE_CHECK(blockDim.x == slice_count * THREADS_PER_WARP);
    const unsigned char *matrix = arena + (is_second ? arguments.weights_offset[1] : arguments.weights_offset[0]);
#if LINEAR_THREAD_SUMS                          /* the exact profile on class 120: column c carries row c */
    (void)warp;
    (void)lane_group;
    (void)load_count;
    (void)even_column;
    __shared__ uint4 fed_cache[LINEAR_THREAD_CACHE_VECTORS(slice_count * THREADS_PER_WARP)];
    linear_thread_chains<LINEAR_WEIGHTS_E4M3, LINEAR_TRUNK_FORM>(
	    matrix, output_size, arguments.input_size, tile,
	    arena + arguments.operands_offset,
	    (size_t)arguments.input_size * LINEAR_BFLOAT16_BYTES, NULL, 0,
	    LINEAR_THREAD_COLUMN_VECTORS_LOW, LINEAR_THREAD_COLUMN_VECTORS_HIGH,
	    arguments.row_count, thread, blockDim.x,
	    slice_chains + (size_t)blockDim.x * ELEMENTS_PER_LANE,
	    (uint32_t *)fed_cache, slice_chains);
#else
    struct linear_lane lane;
    lane.column_count = arguments.row_count;
    linear_lane_weights<LINEAR_WEIGHTS_E4M3>(&lane, matrix, output_size, arguments.input_size, tile, lane_index);
    linear_lane_quantized<LINEAR_TRUNK_FORM>(
        &lane, arena + arguments.codes_offset, arena + arguments.exponents_offset, arena + arguments.operands_offset,
        arguments.input_size, lane_place, lane_group < arguments.row_count ? lane_group : 0,
        even_column < arguments.row_count ? even_column : 0,
        even_column + 1u < arguments.row_count ? even_column + 1u : 0);
    LINEAR_CHAINS(LINEAR_WEIGHTS_E4M3, LINEAR_TRUNK_FORM, warp, load_count,
                  slice_chains + (size_t)thread * ELEMENTS_PER_LANE);
#endif
    __syncthreads();
    trunk_store_results<slice_count>(
        slice_chains, arguments.row_count, output_size, tile,
        (uint16_t *)(arena + (is_second ? arguments.output_offset[1] : arguments.output_offset[0])), thread);
}

/* grid (output_size / 16), block of 32 S threads, 64 bytes of shared memory per thread; is_wide, wide_tile and
 * wide_thread as for trunk_linear_quantized. */
template <uint32_t slice_count>
__device__ __forceinline__ void
trunk_linear_bfloat16(struct trunk_linear_bfloat16_arguments arguments,
		      unsigned char *arena, bool is_wide, uint32_t wide_tile,
		      uint32_t wide_thread)
{
    extern __shared__ float slice_chains[];
    const uint32_t thread = is_wide ? wide_thread : threadIdx.x;
    const uint32_t warp = thread / THREADS_PER_WARP, lane_index = thread % THREADS_PER_WARP;
    const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP, lane_place = lane_index % LANES_PER_LANE_GROUP;
    const uint32_t load_count = arguments.input_size / VALUES_PER_LOAD;
    const size_t fed_row = lane_group < arguments.row_count ? lane_group : 0;
    const uint32_t tile = is_wide ? wide_tile : blockIdx.x;
    const size_t segment = tile / (arguments.output_size / OUTPUTS_PER_TILE / arguments.segment_count);
    LINEAR_DEVICE_CHECK(blockDim.x == slice_count * THREADS_PER_WARP);
#if LINEAR_THREAD_SUMS                          /* the exact profile on class 120: column c carries row c */
    (void)warp;
    (void)lane_place;
    (void)load_count;
    (void)fed_row;
    const size_t row_bytes = (size_t)arguments.input_size *
			     LINEAR_BFLOAT16_BYTES; /* of a segment */
    __shared__ uint4 fed_cache[LINEAR_THREAD_CACHE_VECTORS(slice_count * THREADS_PER_WARP)];
    linear_thread_chains<LINEAR_WEIGHTS_E4M3, LINEAR_FORM_BFLOAT16_ROWS>(
        arena + arguments.weights_offset, arguments.output_size, arguments.input_size, tile,
        arena + arguments.values_offset + segment * row_bytes, arguments.segment_count * row_bytes, NULL, 0,
        LINEAR_THREAD_COLUMN_VECTORS_LOW, LINEAR_THREAD_COLUMN_VECTORS_HIGH, arguments.row_count, thread,
        blockDim.x, slice_chains + (size_t)blockDim.x * ELEMENTS_PER_LANE, (uint32_t *)fed_cache, slice_chains);
#else
    struct linear_lane lane;
    lane.column_count = arguments.row_count;
    linear_lane_weights<LINEAR_WEIGHTS_E4M3>(&lane, arena + arguments.weights_offset, arguments.output_size,
                                             arguments.input_size, tile, lane_index);
    lane.fed = arena + arguments.values_offset +
	       ((fed_row * arguments.segment_count + segment) *
		arguments.input_size) *
		       LINEAR_BFLOAT16_BYTES;
    lane.exponents[0] = lane.fed;               /* not read: rows that are not quantized have no exponents */
    lane.exponents[1] = lane.fed;
    LINEAR_CHAINS(LINEAR_WEIGHTS_E4M3, LINEAR_FORM_BFLOAT16_ROWS, warp, load_count,
                  slice_chains + (size_t)thread * ELEMENTS_PER_LANE);
#endif
    __syncthreads();
    trunk_store_results<slice_count>(slice_chains, arguments.row_count, arguments.output_size, tile,
                                     (uint16_t *)(arena + arguments.output_offset), thread);
}

/* The rows of the group of a wide launch that begins at first_row: 8, the last group fewer. */
__device__ static __forceinline__ uint32_t trunk_group_rows(uint32_t row_count, uint32_t first_row)
{
    const uint32_t rows_left = row_count - first_row;
    return rows_left < LINEAR_MAXIMUM_ROWS ? rows_left : LINEAR_MAXIMUM_ROWS;
}

/* Where a block of a wide launch works, in units of unit_rows rows (a group of 8; a chunk of the interleaved
 * instance). Its grid is tile_count tiles times n blocks a tile, n from 1 to its units: block b computes tile b / n
 * for the units b % n, b % n + n, .. (first_row of the first, row_step to the next). n = the units, as linear.c
 * prepares the wide instance: a tile's units lie in adjacent blocks, which the card runs together; n = 1: a block
 * takes every unit of its tile in turn. */
struct trunk_wide_place {
    uint32_t tile;
    uint32_t first_row;
    uint32_t row_step;
};
__device__ static __forceinline__ struct trunk_wide_place trunk_wide_place_of(uint32_t tile_count, uint32_t row_count,
                                                                              uint32_t unit_rows)
{
    const uint32_t blocks_per_tile = gridDim.x / tile_count;
    LINEAR_DEVICE_CHECK(gridDim.x % tile_count == 0 && blocks_per_tile != 0 &&
                        (uint64_t)(blocks_per_tile - 1u) * unit_rows < row_count);
    return (struct trunk_wide_place){blockIdx.x / blocks_per_tile, blockIdx.x % blocks_per_tile * unit_rows,
                                     blocks_per_tile * unit_rows};
}

/* The block of trunk_linear_quantized, for row_count rows of up to LINEAR_WIDE_MOST_ROWS. */
template <uint32_t slice_count>
__device__ __forceinline__ void
trunk_linear_quantized_wide(struct trunk_linear_quantized_arguments arguments,
			    unsigned char *arena)
{
	const struct trunk_wide_place place = trunk_wide_place_of(
		(arguments.output_size[0] + arguments.output_size[1]) /
			OUTPUTS_PER_TILE,
		arguments.row_count, LINEAR_MAXIMUM_ROWS);
	struct trunk_linear_quantized_arguments group = arguments;
#pragma unroll 1
    for (uint32_t first_row = place.first_row; first_row < arguments.row_count; first_row += place.row_step) {
        /* opaque to the compiler: what the body derives from these it derives again for each group, as a thin launch
         * does, instead of keeping it in registers across the groups (without it the instance of one slice took 96
         * registers on class 80, the thin one 52). Not the arena: its loads would no longer be known global. */
        uint32_t tile = place.tile, thread = threadIdx.x;
        asm volatile("" : "+l"(group.weights_offset[0]), "+l"(group.weights_offset[1]), "+r"(group.output_size[0]),
                     "+r"(group.output_size[1]), "+r"(group.input_size), "+r"(tile), "+r"(thread));
        const uint64_t input_size = group.input_size;
        group.codes_offset = arguments.codes_offset + first_row * input_size;
        group.exponents_offset = arguments.exponents_offset + first_row * (input_size / VALUES_PER_BLOCK);
	group.operands_offset = arguments.operands_offset +
				first_row * input_size * LINEAR_BFLOAT16_BYTES;
	for (uint32_t matrix = 0; matrix < LINEAR_MERGED_MATRICES; matrix++) {
		const uint64_t output_row_bytes =
			(uint64_t)group.output_size[matrix] *
			LINEAR_BFLOAT16_BYTES;
		group.output_offset[matrix] = arguments.output_offset[matrix] +
					      first_row * output_row_bytes;
	}
	group.row_count = trunk_group_rows(arguments.row_count, first_row);
        if (first_row != place.first_row) {
            __syncthreads();                    /* every warp has stored the last group: its chains may go */
        }
        trunk_linear_quantized<slice_count>(group, arena, true, tile, thread);
    }
}

/* The block of trunk_linear_bfloat16, for row_count rows of up to LINEAR_WIDE_MOST_ROWS. */
template <uint32_t slice_count>
__device__ __forceinline__ void
trunk_linear_bfloat16_wide(struct trunk_linear_bfloat16_arguments arguments,
			   unsigned char *arena)
{
	const struct trunk_wide_place place =
		trunk_wide_place_of(arguments.output_size / OUTPUTS_PER_TILE,
				    arguments.row_count, LINEAR_MAXIMUM_ROWS);
	struct trunk_linear_bfloat16_arguments group = arguments;
#pragma unroll 1
    for (uint32_t first_row = place.first_row; first_row < arguments.row_count; first_row += place.row_step) {
        uint32_t tile = place.tile, thread = threadIdx.x;  /* opaque: as in trunk_linear_quantized_wide */
        asm volatile("" : "+l"(group.weights_offset), "+r"(group.output_size), "+r"(group.input_size),
                     "+r"(group.segment_count), "+r"(tile), "+r"(thread));
        /* a row: the values of its segment_count segments; an output row: output_size values */
	const uint64_t values_row_bytes = (uint64_t)group.segment_count *
					  group.input_size *
					  LINEAR_BFLOAT16_BYTES;
	group.values_offset = arguments.values_offset + first_row * values_row_bytes;
	group.output_offset = arguments.output_offset +
			      first_row * ((uint64_t)group.output_size *
					   LINEAR_BFLOAT16_BYTES);
	group.row_count = trunk_group_rows(arguments.row_count, first_row);
        if (first_row != place.first_row) {
            __syncthreads();                    /* every warp has stored the last group: its chains may go */
        }
        trunk_linear_bfloat16<slice_count>(group, arena, true, tile, thread);
    }
}

#if !LINEAR_THREAD_SUMS
/* The chains of TRUNK_INTERLEAVED_GROUPS groups of rows over every load of one slice of the whole input: each load's
 * weights are asked for once (before the load ahead of them is used, as linear_slice_chains asks) and worked on
 * against every group in turn, the same words placed the same way for each. lanes[g] is group g's lane as the thin
 * kernel sets it; chains[g] its four chains, each from +0 with one fused multiply-add per block of 32 inputs in
 * ascending order: for every group the same operands, the same instructions and the same order as linear_slice_chains
 * gives it in a thin launch of that group's rows. */
template <int form>
__device__ __forceinline__ void trunk_interleaved_chains(const struct linear_lane lanes[TRUNK_INTERLEAVED_GROUPS],
                                                         uint32_t lane_place, uint32_t load_count,
                                                         float chains[TRUNK_INTERLEAVED_GROUPS][ELEMENTS_PER_LANE])
{
    uint4 ahead_first, ahead_second;
    linear_fetch_weights<LINEAR_WEIGHTS_E4M3>(&lanes[0], 0, &ahead_first, &ahead_second);
    for (uint32_t load = 0; load < load_count; load++) {
        const uint4 first = ahead_first, second = ahead_second;
        const uint32_t weight_words[WORDS_PER_LOAD] = {first.x,  first.y,  first.z,  first.w,
                                                       second.x, second.y, second.z, second.w};
        if (load + 1u < load_count) {
            linear_fetch_weights<LINEAR_WEIGHTS_E4M3>(&lanes[0], load + 1u, &ahead_first, &ahead_second);
        }
#pragma unroll
        for (uint32_t group = 0; group < TRUNK_INTERLEAVED_GROUPS; group++) {
            const struct linear_fed fed = linear_fetch_fed<LINEAR_WEIGHTS_E4M3, form>(&lanes[group], lane_place, load);
            float sums[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE], powers[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE];
            linear_load_sums<LINEAR_WEIGHTS_E4M3, form>(&lanes[group], weight_words, &fed, sums, powers);
#pragma unroll
            for (uint32_t block = 0; block < BLOCKS_PER_LOAD; block++) {
#pragma unroll
                for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
                    chains[group][element] = fmaf(sums[block][element], powers[block][element], chains[group][element]);
                }
            }
        }
    }
}
#endif

/* The interleaved instance (one slice only, linear_records.h): a block takes its tile's groups
 * TRUNK_INTERLEAVED_GROUPS at a time, a chunk of TRUNK_CHUNK_ROWS rows, and runs each load of its weights against
 * every group of the chunk (trunk_interleaved_chains): its weights are read once a chunk. A group's lane is the thin
 * kernel's for the group's rows; a chunk past the last row gives the groups it lacks its first group's lane, computed
 * and stored nowhere. The chunk's chains then lie in the block's shared memory, 32 lanes by 4 a group (the 2 KB of a
 * block of one warp), and are stored group by group as trunk_store_results stores a thin launch's. Class 120's
 * emulation (its thread sums) takes the groups one at a time, as the wide instance does. */
template <uint32_t slice_count>
__device__ __forceinline__ void trunk_linear_quantized_interleaved(
	struct trunk_linear_quantized_arguments arguments, unsigned char *arena)
{
#if LINEAR_THREAD_SUMS
    trunk_linear_quantized_wide<slice_count>(arguments, arena);
#else
    extern __shared__ float slice_chains[];
    const uint32_t thread = threadIdx.x, lane_index = thread % THREADS_PER_WARP;
    const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP, lane_place = lane_index % LANES_PER_LANE_GROUP;
    const uint32_t even_column = lane_place * COLUMNS_PER_LANE;
    const uint32_t load_count = arguments.input_size / VALUES_PER_LOAD;
    const uint64_t input_size = arguments.input_size;
    const struct trunk_wide_place place =
        trunk_wide_place_of((arguments.output_size[0] + arguments.output_size[1]) / OUTPUTS_PER_TILE,
                            arguments.row_count, TRUNK_CHUNK_ROWS);
    const uint32_t first_tiles = arguments.output_size[0] / OUTPUTS_PER_TILE;
    const bool is_second = place.tile >= first_tiles;           /* never where output_size[1] is 0 */
    const uint32_t tile = is_second ? place.tile - first_tiles : place.tile;
    const uint32_t output_size = is_second ? arguments.output_size[1] : arguments.output_size[0];
    const unsigned char *const matrix = arena + (is_second ? arguments.weights_offset[1] : arguments.weights_offset[0]);
    uint16_t *const output =
        (uint16_t *)(arena + (is_second ? arguments.output_offset[1] : arguments.output_offset[0]));
    LINEAR_DEVICE_CHECK(slice_count == 1u && blockDim.x == THREADS_PER_WARP);
#pragma unroll 1
    for (uint32_t first_row = place.first_row; first_row < arguments.row_count; first_row += place.row_step) {
        struct linear_lane lanes[TRUNK_INTERLEAVED_GROUPS];
        float chains[TRUNK_INTERLEAVED_GROUPS][ELEMENTS_PER_LANE] = {};
#pragma unroll
        for (uint32_t group = 0; group < TRUNK_INTERLEAVED_GROUPS; group++) {
		const uint32_t own_row =
			first_row + group * LINEAR_MAXIMUM_ROWS;
		const uint32_t group_row =
			own_row < arguments.row_count ? own_row : first_row;
		const uint32_t group_rows =
			trunk_group_rows(arguments.row_count, group_row);
		lanes[group].column_count = group_rows;
		linear_lane_weights<LINEAR_WEIGHTS_E4M3>(&lanes[group], matrix,
							 output_size,
							 arguments.input_size,
							 tile, lane_index);
		linear_lane_quantized<LINEAR_TRUNK_FORM>(
			&lanes[group],
			arena + arguments.codes_offset + group_row * input_size,
			arena + arguments.exponents_offset +
				group_row * (input_size / VALUES_PER_BLOCK),
			arena + arguments.operands_offset +
				group_row * input_size * LINEAR_BFLOAT16_BYTES,
			arguments.input_size, lane_place,
			lane_group < group_rows ? lane_group : 0,
			even_column < group_rows ? even_column : 0,
			even_column + 1u < group_rows ? even_column + 1u : 0);
	}
        trunk_interleaved_chains<LINEAR_TRUNK_FORM>(lanes, lane_place, load_count, chains);
        if (first_row != place.first_row) {
            __syncthreads();                    /* the last chunk's stores have read the block's shared memory */
        }
#pragma unroll
        for (uint32_t group = 0; group < TRUNK_INTERLEAVED_GROUPS; group++) {
#pragma unroll
            for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
                slice_chains[(group * THREADS_PER_WARP + lane_index) * ELEMENTS_PER_LANE + element] =
                    chains[group][element];
            }
        }
        __syncthreads();
	for (uint32_t group = 0;
	     group < TRUNK_INTERLEAVED_GROUPS &&
	     first_row + group * LINEAR_MAXIMUM_ROWS < arguments.row_count;
	     group++) {
		const uint32_t group_row =
			first_row + group * LINEAR_MAXIMUM_ROWS;
		trunk_store_results<slice_count>(
			slice_chains + (size_t)group * THREADS_PER_WARP *
					       ELEMENTS_PER_LANE,
			trunk_group_rows(arguments.row_count, group_row),
			output_size, tile,
			output + (size_t)group_row * output_size, thread);
	}
    }
#endif
}

/* No thin instance states a launch bound (see kernel_expert_linear.cu); the build checks that the registers of each
 * instance fit its block. A wide instance keeps its thin one's share of the register file, so that a multiprocessor
 * holds as many of its blocks: the release modules give it no more units of 8 but class 120 native's of quantized
 * rows (54 to 56 registers against the thin ones' 46: a block fewer of 2 to 8 slices), which
 * TRUNK_WIDE_QUANTIZED_BOUND holds to the thin ones' 48, with no spill and the same instructions or fewer (assembler
 * reports, 25 October 2026), and class 80's of 2 slices (80 against 72), which no class 80 default takes. */
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && defined(LINEAR_CLASS_FORMS)
#define TRUNK_NATIVE_THIN_REGISTERS 48
#define TRUNK_WIDE_QUANTIZED_BOUND __maxnreg__(TRUNK_NATIVE_THIN_REGISTERS)
#else
#define TRUNK_WIDE_QUANTIZED_BOUND
#endif
#define LINEAR_SLICE_STAMP(slices)                                             \
	extern "C" __global__ void trunk_linear_quantized_##slices(            \
		struct trunk_linear_quantized_arguments arguments,             \
		unsigned char *arena)                                          \
	{                                                                      \
		LINEAR_SLICED_BODY(slices, LINEAR_CLASS_120_EXACT_MOST_SLICES, \
				   trunk_linear_quantized<slices>(             \
					   arguments, arena, false, 0u, 0u))   \
	}                                                                      \
	extern "C" __global__ void trunk_linear_bfloat16_##slices(             \
		struct trunk_linear_bfloat16_arguments arguments,              \
		unsigned char *arena)                                          \
	{                                                                      \
		LINEAR_SLICED_BODY(slices, LINEAR_CLASS_120_EXACT_MOST_SLICES, \
				   trunk_linear_bfloat16<slices>(              \
					   arguments, arena, false, 0u, 0u))   \
	}                                                                      \
	extern "C" __global__ void TRUNK_WIDE_QUANTIZED_BOUND                  \
	trunk_linear_quantized_wide_##slices(                                  \
		struct trunk_linear_quantized_arguments arguments,             \
		unsigned char *arena)                                          \
	{                                                                      \
		LINEAR_SLICED_BODY(                                            \
			slices, LINEAR_CLASS_120_EXACT_MOST_SLICES,            \
			trunk_linear_quantized_wide<slices>(arguments, arena)) \
	}                                                                      \
	extern "C" __global__ void trunk_linear_bfloat16_wide_##slices(        \
		struct trunk_linear_bfloat16_arguments arguments,              \
		unsigned char *arena)                                          \
	{                                                                      \
		LINEAR_SLICED_BODY(                                            \
			slices, LINEAR_CLASS_120_EXACT_MOST_SLICES,            \
			trunk_linear_bfloat16_wide<slices>(arguments, arena))  \
	}
LINEAR_EVERY_SLICE_COUNT
#undef LINEAR_SLICE_STAMP

/* The interleaved instance: one slice alone (linear_records.h). */
extern "C" __global__ void trunk_linear_quantized_interleaved_1(
	struct trunk_linear_quantized_arguments arguments, unsigned char *arena)
{
    trunk_linear_quantized_interleaved<1u>(arguments, arena);
}

/* The shared join: grid (MODEL_HIDDEN / 16), block of 32 S threads, 64 bytes of shared memory per thread.
 * Thread 2 r + h of the block joins values 8 h .. 8 h + 7 of the block's tile for row r. It takes them as
 * trunk_store_results rounds them, from the chains the product left in the block's shared memory behind its barrier
 * (output o of column r: lane 4 (o % 8) + r / 2, element 2 (o / 8) + r % 2): the bits it stores. */
template <uint32_t slice_count>
__device__ __forceinline__ void
expert_shared_join(struct expert_shared_join_arguments arguments,
		   unsigned char *arena)
{
    extern __shared__ float slice_chains[];
    const uint32_t row_loads = HYPER_ROW_THREADS,
		   tile_loads = OUTPUTS_PER_TILE / JOIN_VALUES;
    const struct trunk_linear_quantized_arguments product = {
	    { arguments.weights_offset, 0 },
	    { arguments.output_offset, 0 },
	    arguments.codes_offset,
	    arguments.exponents_offset,
	    arguments.operands_offset,
	    { MODEL_HIDDEN, 0 },
	    arguments.input_size,
	    arguments.row_count
    };
    trunk_linear_quantized<slice_count>(product, arena, false, 0u, 0u);
    const uint32_t row = threadIdx.x / tile_loads, half = threadIdx.x % tile_loads;
    if (row < arguments.row_count) {
        const size_t load = (size_t)blockIdx.x * tile_loads + half;
        float shared_output[JOIN_VALUES];
        for (uint32_t value = 0; value < JOIN_VALUES; value++) {
            shared_output[value] = linear_bfloat16_value(linear_bfloat16_bits(linear_sum_of_slices<slice_count>(
                slice_chains, 0, value * LANES_PER_LANE_GROUP + row / COLUMNS_PER_LANE,
                half * COLUMNS_PER_LANE + row % COLUMNS_PER_LANE)));
        }
	expert_join_values(
		(const uint4 *)(arena + arguments.pair_output_offset) +
			(size_t)row * arguments.pairs_per_row * row_loads +
			load,
		arguments.pairs_per_row, shared_output,
		(const float *)(arena + arguments.post_offset) +
			(size_t)row * JOIN_STREAMS,
		(const float *)(arena + arguments.comb_offset) +
			(size_t)row * JOIN_STREAMS * JOIN_STREAMS,
		(uint4 *)(arena + arguments.streams_offset) +
			(size_t)row * JOIN_STREAMS * row_loads + load,
		arguments.row_sums_offset == EXPERT_JOIN_NO_ROW_SUMS ?
			NULL :
			(float4 *)(arena + arguments.row_sums_offset) +
				((size_t)row * row_loads + load) *
					JOIN_SUM_STORES);
    }
}

#define LINEAR_SLICE_STAMP(slices)                                             \
	extern "C" __global__ void expert_shared_join_##slices(                \
		struct expert_shared_join_arguments arguments,                 \
		unsigned char *arena)                                          \
	{                                                                      \
		LINEAR_SLICED_BODY(slices, LINEAR_CLASS_120_EXACT_MOST_SLICES, \
				   expert_shared_join<slices>(arguments,       \
							      arena))          \
	}
LINEAR_EVERY_SLICE_COUNT
#undef LINEAR_SLICE_STAMP

/* grid (ceil(width / blockDim.x), vectors): a thread takes one value of one vector. */
extern "C" __global__ void
trunk_hidden_entry(struct trunk_hidden_arguments arguments,
		   unsigned char *arena)
{
    const size_t column = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= arguments.width) {
        return;
    }
    const size_t value_index = (size_t)blockIdx.y * arguments.width + column;
    const float gate = linear_bfloat16_value(((const uint16_t *)(arena + arguments.gate_offset))[value_index]);
    const float up = linear_bfloat16_value(((const uint16_t *)(arena + arguments.up_offset))[value_index]);
    ((uint16_t *)(arena + arguments.hidden_offset))[value_index] =
        (uint16_t)linear_hidden_bits(gate, up, arguments.activation_limit, 1.0f);
}
