/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_site_front.cu: the fronts of a block's sites in one launch each: the site front (a site's mix project, mix
 * finish and the quantization of the sublayer's input), the FFN front (the same and the router's scores and select),
 * and the site front cut in two launches (the input front, then the coefficient front). Device code only, built once
 * per architecture into the module "kernel_site_front".
 *
 * Contract: include/anemone/site_front_records.h: every output has the bits of the launches it stands for. Here a
 * block of 256 threads takes one part:
 *   - A row block (row r; the input front's four of a row each its quarter of the stores): it reads the carried pre
 *     and, where the launch finishes the coefficients, arrives at once (so the finish's overwrite follows every read
 *     of it). Then, as the mix finish's 640 threads, logical thread T on thread T % 256 (logical warp w on warp w % 8):
 *     the collapse with that pre, its square sums, the norm and the bfloat16 input, stored (and, in the FFN front of
 *     several rows, published to the score blocks). Then it quantizes the input, four logical threads (32 values) a
 *     block: each forms the largest of its eight magnitudes, two exchanges the block's (a maximum has no order:
 *     kernel_activation_quantize.cuh), and each value is quantized and stored by the quantizer's own functions.
 *   - A mix block: logical warps of one mix as the dense product's (logical thread t: chunks t, t + 256, ... of 8
 *     inputs, its chain fmaf from +0 in input order; a warp's chains by exchanges 16 down to 1): all 8 on the 8 warps
 *     (the FFN front), or 4 of them on warps w and w + 4, each warp half the rows (the site front). The warp of
 *     logical warp m of mix m (m below 8) also forms that warp's square sums of the rows, in the same chain order.
 *     Each warp sum is published; the last block to arrive finishes: each mix of each row, its warp sums and its
 *     row's square sums each from +0 in order, the sum times rsqrtf(square sum * float32(1 / 20,480) + epsilon) (the
 *     dense product's finish; its output scale 1 changes no bit: every arithmetic not-a-number of the card is the same
 *     one); then Sinkhorn of rows 2 w and 2 w + 1 on the halves of warp w (kernel_hyper_connection.cuh), which writes
 *     post, comb and the site's pre.
 *   - A score block (the FFN front's): experts 4 b .. 4 b + 3 as the dense product's wide form computes them, its
 *     weights started early (in registers at one row, an L2 request a line at several). Of one row it forms the row's
 *     input itself as a row block does (the front of one row has no row block: the score blocks also quantize its
 *     blocks of 32 values, block b and b + 96 by score block b, one a warp, with the quantizer's warp function); of
 *     several it waits until every row block has stored its input. Then each score of every row and, of each live
 *     row, the affinity the select forms from it; the last score block runs the select's own code (kernel_router.cuh)
 *     on the live rows' affinities.
 *   - A block publishes by its stores, a fence by each thread that stored, a barrier and one thread's atomic addition;
 *     a block that reads what others published fences first and reads through L2. The FFN front of several rows
 *     gives its blocks their parts by ticket: a score block waits only for row blocks, of lower tickets, which wait
 *     for nothing. The checked modules trap on a wait beyond FRONT_WAIT_POLLS polls.
 *   - Every thread of a block passes every barrier: a block's part and its being last are the same for all.
 */
#include "kernel_activation_quantize.cuh"
#include "kernel_hyper_connection.cuh"
#include "kernel_router.cuh"

#include "anemone/site_front_records.h"

#define FRONT_THREADS SITE_FRONT_THREADS
#define FRONT_WARPS (FRONT_THREADS / THREADS_PER_WARP)
#define MOST_ROWS LINEAR_MAXIMUM_ROWS
#define HIDDEN MODEL_HIDDEN
#define CHUNK_VALUES DENSE_CHUNK_INPUTS /* a 16-byte load of bfloat16 */
#define MIX_CHUNKS               \
	(SITE_FRONT_MIX_INPUTS / \
	 CHUNK_VALUES) /* of a row's streams: stream s's c at 640 s + c */
#define STREAM_CHUNKS (HIDDEN / CHUNK_VALUES)               /* of one stream; of a hidden row */
#define DENSE_WARPS SITE_FRONT_LOGICAL_WARPS /* of a dense output */
#define CHAIN_CHUNKS \
	(MIX_CHUNKS / DENSE_THREADS_VALUE) /* of one chain of the mix project */
#define ROW_THREADS \
	HYPER_ROW_THREADS /* the mix finish's logical threads of a row */
#define ROW_WARPS (ROW_THREADS / THREADS_PER_WARP)
#define ROW_PARTS ((ROW_THREADS + FRONT_THREADS - 1u) / FRONT_THREADS)
#define FLOAT32_PARTS 2u                                    /* 16-byte loads of a chunk's float32 weights */
#define PART_INPUTS (CHUNK_VALUES / FLOAT32_PARTS)
#define LOAD_WORDS 4u                                       /* of a 16-byte load */
#define BFLOAT16_PER_WORD 2u
#define FIRST_DISTANCE (THREADS_PER_WARP >> 1u)             /* of the exchanges of a warp sum, halved down to 1 */
#define BLOCK_LANES (VALUES_PER_BLOCK / HYPER_VALUES)       /* the logical threads of a block of 32 values */
#define SINKHORN_LANES HYPER_COMB_VALUES                    /* a row's lanes: half a warp */
#define WARP_HALVES (THREADS_PER_WARP / SINKHORN_LANES)
#define MIX_CHUNKS_AHEAD 2                                  /* of a chain of several rows, unrolled */
#define SCORE_EXPERTS FFN_FRONT_EXPERTS
#define SCORE_BLOCKS FFN_FRONT_SCORE_BLOCKS
#define SCORE_STEPS ((STREAM_CHUNKS + FRONT_THREADS - 1u) / FRONT_THREADS)  /* chunks of a thread of the router */
#define LINE_CHUNKS 8u                                      /* chunks of a 128-byte line: one L2 prefetch each */
#define LINE_BYTES 128u
#define TABLE_LINES                   \
	(MODEL_EXPERT_OFFSETS_BYTES / \
	 LINE_BYTES) /* of a layer's expert table */
#define FRONT_WAIT_POLLS 0x1000000u                         /* about 10 s: far beyond any row block's time */
#define FRONT_POLL_NANOSECONDS 64u
#define NO_LOAD make_uint4(0u, 0u, 0u, 0u)

static_assert(
	ROW_THREADS % (SITE_FRONT_INPUT_PARTS * THREADS_PER_WARP) == 0 &&
		THREADS_PER_WARP % BLOCK_LANES == 0,
	"a part holds whole warps (its exchanges take every lane) and so whole blocks of 32 values");
static_assert(
	FRONT_WARPS *WARP_HALVES >= MOST_ROWS && FRONT_WARPS == DENSE_WARPS &&
		ROW_WARPS <= THREADS_PER_WARP &&
		BLOCK_LANES * HYPER_VALUES == VALUES_PER_BLOCK &&
		FRONT_THREADS % THREADS_PER_WARP == 0 &&
		MIX_CHUNKS % DENSE_THREADS_VALUE == 0,
	"a warp's halves take the rows' Sinkhorn; the warps are the dense product's");

/* Value `index` of a 16-byte load, widened exactly: a bfloat16 (the even one in the low half of its word), or a
 * float32 (the dense product's dense_value). */
__device__ static __forceinline__ float front_bfloat16(uint4 load, uint32_t index)
{
    const uint32_t words[LOAD_WORDS] = {load.x, load.y, load.z, load.w};
    const uint32_t word = words[index / BFLOAT16_PER_WORD];
    return __uint_as_float(index % BFLOAT16_PER_WORD == 0 ? word << BITS_PER_HALF : word & UPPER_HALF_MASK);
}
__device__ static __forceinline__ float front_float32(uint4 load, uint32_t index)
{
    const uint32_t words[LOAD_WORDS] = {load.x, load.y, load.z, load.w};
    return __uint_as_float(words[index]);
}

/* Waits until another block's publications bring *counter to target, then fences; the checked modules trap past
 * FRONT_WAIT_POLLS polls (the release modules hold no trap: the wait ends by construction). */
__device__ static __forceinline__ void front_wait(const uint32_t *counter, uint32_t target)
{
#if CHECKING_HOOKS_ENABLED
	uint32_t polls = 0;
#endif
    while (*(volatile const uint32_t *)counter < target) {
        __nanosleep(FRONT_POLL_NANOSECONDS);
#if CHECKING_HOOKS_ENABLED
	polls++;
        LINEAR_DEVICE_CHECK(polls < FRONT_WAIT_POLLS);
#endif
    }
    __threadfence();
}

/* After a thread's stores and its fence: the block's barrier, one atomic addition, and the answer to every thread,
 * whether this block was the last of block_count to arrive. */
__device__ static __forceinline__ bool front_publish(uint32_t *counter, uint32_t block_count, uint32_t *is_last)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        *is_last = atomicAdd(counter, 1u) == block_count - 1u ? 1u : 0u;
    }
    __syncthreads();
    return *is_last != 0;
}

/* The last block to arrive at the mix counter: the mixes, Sinkhorn, post, comb and the site's pre; the counter
 * cleared. */
__device__ static __forceinline__ void
front_finish(const struct site_front_arguments *arguments, unsigned char *arena,
	     struct site_front_scratch *scratch)
{
	__shared__ float mixes[MOST_ROWS][MODEL_MIXES];
	const uint32_t thread = threadIdx.x, lane = thread % THREADS_PER_WARP,
		       row_count = arguments->select.row_count;
	__threadfence();
	for (uint32_t index = thread; index < row_count * MODEL_MIXES;
	     index += FRONT_THREADS) {
		const uint32_t row = index / MODEL_MIXES,
			       mix = index % MODEL_MIXES;
		float sum = 0.0f, square_sum = 0.0f;
		for (uint32_t warp = 0; warp < DENSE_WARPS; warp++) {
			sum = sum + __ldcg(&scratch->warp_sums[mix][warp][row]);
			square_sum = square_sum +
				     __ldcg(&scratch->square_sums[warp][row]);
		}
		const float value =
			sum *
			rsqrtf(square_sum *
				       (1.0f / (float)SITE_FRONT_MIX_INPUTS) +
			       MODEL_NORM_EPSILON);
		mixes[row][mix] = value;
		scratch->mixes[row][mix] = value;
	}
    __syncthreads();
    const uint32_t row = thread / THREADS_PER_WARP * WARP_HALVES + lane / SINKHORN_LANES;
    const uint32_t sub_lane = lane % SINKHORN_LANES;
    const float *const scale = (const float *)(arena + arguments->scale_offset);
    const float *const base = (const float *)(arena + arguments->base_offset);
    const float *const row_mixes = mixes[row < row_count ? row : 0];
    if (row < MOST_ROWS) {                      /* the warps of the eight rows: whole warps */
        const float comb = hyper_sinkhorn_comb(row_mixes, scale, base, sub_lane);
        if (row < row_count) {
            ((float *)(arena + arguments->comb_offset))[row * HYPER_COMB_VALUES + sub_lane] = comb;
            if (sub_lane < HYPER_STREAMS) {
                ((float *)(arena + arguments->post_offset))[row * HYPER_STREAMS + sub_lane] =
                    hyper_post_of(row_mixes, scale, base, sub_lane);
                ((float *)(arena + arguments->pre_offset))[row * HYPER_STREAMS + sub_lane] =
                    hyper_pre_of(row_mixes, scale, base, sub_lane);
            }
        }
    }
    if (thread == 0) {
        scratch->arrived = 0;
    }
}

/* The row's carried pre into shared memory and its streams' loads of logical thread T of the mix finish (T on thread
 * T % 256, the parts with T below 640). Each thread uses its loads only after the block's next barrier. */
__device__ static __forceinline__ void
front_input_load(const struct site_front_arguments *arguments,
		 const unsigned char *arena, uint32_t row, float *pre,
		 uint4 loads[ROW_PARTS][HYPER_STREAMS])
{
    const uint32_t thread = threadIdx.x;
    const uint4 *const streams = (const uint4 *)(arena + arguments->streams_offset) + (size_t)row * MIX_CHUNKS;
    if (thread < HYPER_STREAMS) {
        pre[thread] = ((const float *)(arena + arguments->pre_offset))[row * HYPER_STREAMS + thread];
    }
#pragma unroll
    for (uint32_t part = 0; part < ROW_PARTS; part++) {
        const uint32_t logical = thread + part * FRONT_THREADS;
        for (uint32_t stream = 0; stream < HYPER_STREAMS; stream++) {
            loads[part][stream] = logical < ROW_THREADS ? streams[stream * STREAM_CHUNKS + logical] : NO_LOAD;
        }
    }
}

/* The mix finish's collapse with pre, its square sums (one barrier of the block), the norm and the bfloat16 input of
 * the row: packed[part] the eight values of logical thread thread + 256 part, where it is below 640. */
__device__ static __forceinline__ void
front_input(const struct site_front_arguments *arguments,
	    const unsigned char *arena, const float *pre,
	    const uint4 loads[ROW_PARTS][HYPER_STREAMS], float *row_sums,
	    uint4 packed[ROW_PARTS])
{
    const uint32_t thread = threadIdx.x, lane = thread % THREADS_PER_WARP;
    float hidden[ROW_PARTS][HYPER_VALUES] = {};
#pragma unroll
    for (uint32_t part = 0; part < ROW_PARTS; part++) {     /* the collapse and its squares */
        const uint32_t logical = thread + part * FRONT_THREADS;
        if (logical < ROW_THREADS) {                        /* the same for a whole warp */
            float square_sum = 0.0f;
            for (uint32_t stream = 0; stream < HYPER_STREAMS; stream++) {
                float values[HYPER_VALUES];
                const float weight = pre[stream];
                hyper_unpack(loads[part][stream], values);
                for (uint32_t value = 0; value < HYPER_VALUES; value++) {
                    hidden[part][value] = hidden[part][value] + weight * values[value];
                }
            }
            for (uint32_t value = 0; value < HYPER_VALUES; value++) {
                hidden[part][value] = linear_bfloat16_value(linear_bfloat16_bits(hidden[part][value]));
                square_sum = square_sum + hidden[part][value] * hidden[part][value];
            }
            square_sum = hyper_warp_sum(square_sum);
            if (lane == 0) {
                row_sums[logical / THREADS_PER_WARP] = square_sum;
            }
        }
    }
    __syncthreads();
    const float factor =
	    rsqrtf(hyper_warp_sum(lane < ROW_WARPS ? row_sums[lane] : 0.0f) *
			   HYPER_MEAN_FACTOR +
		   MODEL_NORM_EPSILON);
    const uint4 *const norm_weights = (const uint4 *)(arena + arguments->norm_weight_offset);
#pragma unroll
    for (uint32_t part = 0; part < ROW_PARTS; part++) {     /* the norm and the input */
        const uint32_t logical = thread + part * FRONT_THREADS;
        packed[part] = NO_LOAD;
        if (logical < ROW_THREADS) {
            float weights[HYPER_VALUES];
            hyper_unpack(norm_weights[logical], weights);
            for (uint32_t value = 0; value < HYPER_VALUES; value++) {
                hidden[part][value] = weights[value] * (hidden[part][value] * factor);
            }
            packed[part] = hyper_pack(hidden[part]);
        }
    }
}

/* Row `row`: arrives once the carried pre is read, where the launch finishes the site's coefficients (is_arriving);
 * its input stored (and published to the score blocks that wait for it: is_published), then quantized: of the
 * logical threads part * 640 / parts .. (part + 1) * 640 / parts alone (the parts' blocks form the same input). */
__device__ static __forceinline__ void
front_row(const struct site_front_arguments *arguments, unsigned char *arena,
	  struct site_front_scratch *scratch, uint32_t row, uint32_t part,
	  uint32_t parts, uint32_t arrivals, bool is_arriving,
	  bool is_published)
{
    const uint32_t first_logical = part * (ROW_THREADS / parts), last_logical = first_logical + ROW_THREADS / parts;
    __shared__ float pre[HYPER_STREAMS], row_sums[ROW_WARPS];
    __shared__ uint32_t is_last;
    const uint32_t thread = threadIdx.x;
    uint4 loads[ROW_PARTS][HYPER_STREAMS], packed[ROW_PARTS];
    front_input_load(arguments, arena, row, pre, loads);
    if (!is_arriving) {
        __syncthreads();
    } else if (front_publish(&scratch->arrived, arrivals, &is_last)) {  /* pre read: every mix published before */
        front_finish(arguments, arena, scratch);
    }
    front_input(arguments, arena, pre, loads, row_sums, packed);
    uint4 *const hidden_row = (uint4 *)(arena + arguments->hidden_offset) + (size_t)row * STREAM_CHUNKS;
#pragma unroll
    for (uint32_t step = 0; step < ROW_PARTS; step++) {
        const uint32_t logical = thread + step * FRONT_THREADS;
        if (logical >= first_logical && logical < last_logical) {
            hidden_row[logical] = packed[step];
        }
    }
    if (is_published) {                                     /* the input of the score blocks that wait */
        __threadfence();
        __syncthreads();
        if (thread == 0) {
            atomicAdd(&scratch->rows_ready, 1u);
        }
    }
    unsigned char *const codes = arena + arguments->codes_offset + (size_t)row * HIDDEN;
    unsigned char *const exponents = arena + arguments->exponents_offset + (size_t)row * (HIDDEN / VALUES_PER_BLOCK);
    uint16_t *const operands = (uint16_t *)(arena + arguments->operands_offset) + (size_t)row * HIDDEN;
#pragma unroll
    for (uint32_t step = 0; step < ROW_PARTS; step++) {     /* its quantization */
        const uint32_t logical = thread + step * FRONT_THREADS;
        if (logical >= first_logical && logical < last_logical) {   /* whole blocks of 32 values: 4 lanes */
            float rounded[HYPER_VALUES];
            hyper_unpack(packed[step], rounded);
            float largest = fabsf(rounded[0]);
            for (uint32_t value = 1; value < HYPER_VALUES; value++) {
                largest = fmaxf(largest, fabsf(rounded[value]));
            }
            for (uint32_t distance = 1; distance < BLOCK_LANES; distance <<= 1u) {
                largest = fmaxf(largest, __shfl_xor_sync(HYPER_EVERY_LANE, largest, (int)distance));
            }
            const uint32_t exponent_code = quantize_exponent_code(largest), block = logical / BLOCK_LANES;
            const float inverse_scale = quantize_inverse_scale(exponent_code);
            for (uint32_t value = 0; value < HYPER_VALUES; value++) {
                quantize_value_store(codes, operands, block, logical % BLOCK_LANES * HYPER_VALUES + value,
                                     rounded[value], exponent_code, inverse_scale);
            }
            if (logical % BLOCK_LANES == 0) {
                exponents[block] = (unsigned char)exponent_code;
            }
        }
    }
}

/* Mix block mix_block: logical warps of one mix (block_warps of them, each on FRONT_WARPS / block_warps warps that
 * share its rows), their chains and warp sums published, and the squares of the warp whose logical warp is the mix. */
template <uint32_t row_capacity, uint32_t mix_parts>
__device__ static __forceinline__ void
front_mix(const struct site_front_arguments *arguments, unsigned char *arena,
	  struct site_front_scratch *scratch, uint32_t mix_block,
	  uint32_t arrivals)
{
    constexpr uint32_t block_warps = DENSE_WARPS / mix_parts;
    constexpr uint32_t warp_rows = row_capacity >= mix_parts ? row_capacity / mix_parts : 1u;
    constexpr int chunks_unrolled = warp_rows == 1u ? (int)CHAIN_CHUNKS : MIX_CHUNKS_AHEAD;  /* a pragma's count */
    __shared__ uint32_t is_last;
    const uint32_t warp = threadIdx.x / THREADS_PER_WARP, lane = threadIdx.x % THREADS_PER_WARP;
    const uint32_t row_count = arguments->select.row_count, mix = mix_block / mix_parts;
    const uint32_t logical_warp = mix_block % mix_parts * block_warps + warp % block_warps;
    const uint32_t first_row = warp / block_warps * warp_rows, logical = logical_warp * THREADS_PER_WARP + lane;
    const bool is_squaring = logical_warp == mix;
    const uint4 *const weights =
        (const uint4 *)(arena + arguments->mix_weights_offset) + (size_t)mix * MIX_CHUNKS * FLOAT32_PARTS;
    const uint4 *const rows = (const uint4 *)(arena + arguments->streams_offset);
    if (first_row < row_count) {                            /* the same for a whole warp */
        float sums[warp_rows] = {}, squares[warp_rows] = {};
#pragma unroll chunks_unrolled
        for (uint32_t step = 0; step < CHAIN_CHUNKS; step++) {
		const uint32_t chunk = logical + step * DENSE_THREADS_VALUE;
		uint4 values[warp_rows];
		for (uint32_t row = 0; row < warp_rows; row++) {
			const uint32_t read_row = first_row + row < row_count ?
							  first_row + row :
							  0u;
			values[row] =
				rows[(size_t)read_row * MIX_CHUNKS + chunk];
		}
            for (uint32_t part = 0; part < FLOAT32_PARTS; part++) {
                const uint4 part_weights = __ldcs(weights + (size_t)chunk * FLOAT32_PARTS + part);
#pragma unroll
                for (uint32_t index = 0; index < PART_INPUTS; index++) {
                    for (uint32_t row = 0; row < warp_rows; row++) {
                        const float value = front_bfloat16(values[row], part * PART_INPUTS + index);
                        squares[row] = is_squaring ? fmaf(value, value, squares[row]) : squares[row];
                        sums[row] = fmaf(front_float32(part_weights, index), value, sums[row]);
                    }
                }
            }
        }
        for (uint32_t distance = FIRST_DISTANCE; distance != 0; distance >>= 1u) {
            for (uint32_t row = 0; row < warp_rows; row++) {
                squares[row] = is_squaring ? squares[row] + __shfl_xor_sync(HYPER_EVERY_LANE, squares[row],
                                                                            (int)distance)
                                           : squares[row];
                sums[row] = sums[row] + __shfl_xor_sync(HYPER_EVERY_LANE, sums[row], (int)distance);
            }
        }
        if (lane == 0) {
            for (uint32_t row = 0; row < warp_rows && first_row + row < row_count; row++) {
                scratch->warp_sums[mix][logical_warp][first_row + row] = sums[row];
                if (is_squaring) {
                    scratch->square_sums[logical_warp][first_row + row] = squares[row];
                }
            }
            __threadfence();
        }
    }
    if (front_publish(&scratch->arrived, arrivals, &is_last)) {
        front_finish(arguments, arena, scratch);
    }
}

/* Of an FFN front of one row, which has no row block, score block score_block's share of the input: the blocks of 32
 * values score_block, score_block + 96, ..., one a warp, quantized by the quantizer's own warp function, each value
 * also stored in hidden. inputs: the row's input in shared memory, complete. */
__device__ static __forceinline__ void
front_quantize_share(const struct site_front_arguments *arguments,
		     unsigned char *arena, const uint4 *inputs,
		     uint32_t score_block)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
    const uint32_t block = score_block + threadIdx.x / THREADS_PER_WARP * SCORE_BLOCKS;
    if (block < HIDDEN / VALUES_PER_BLOCK) {                /* the same for a whole warp */
        const uint32_t value = (uint32_t)((const uint16_t *)inputs)[block * VALUES_PER_BLOCK + lane];
        activation_quantize_block(value, block, lane, arena + arguments->codes_offset,
                                  arena + arguments->exponents_offset,
                                  (uint16_t *)(arena + arguments->operands_offset));
        ((uint16_t *)(arena + arguments->hidden_offset))[block * VALUES_PER_BLOCK + lane] = (uint16_t)value;
    }
}

/* Score block score_block of the FFN front: its experts' scores of every row and affinities of the live rows; the last
 * one's select. Of one row it forms the row's input itself, as the row block does (arriving once the carried pre is
 * read, so that the finish's overwrite follows its read); of several it waits until every row's input is stored. */
template <uint32_t row_capacity>
__device__ static __forceinline__ void
front_scores(const struct site_front_arguments *arguments, unsigned char *arena,
	     struct site_front_scratch *scratch, uint32_t score_block,
	     uint32_t arrivals)
{
    constexpr bool is_own_input = row_capacity == 1u;
    __shared__ float warp_sums[DENSE_WARPS][SCORE_EXPERTS * row_capacity];
    __shared__ float affinities[MOST_ROWS * EXPERTS], pair_weights[MOST_PAIRS], bias[EXPERTS];
    __shared__ float pre[HYPER_STREAMS], row_sums[ROW_WARPS];
    __shared__ uint4 inputs[is_own_input ? STREAM_CHUNKS : 1u];
    __shared__ uint32_t pair_experts[MOST_PAIRS], pair_is_first[MOST_PAIRS], is_last;
    const struct router_select_arguments *const select = &arguments->select;
    const uint32_t thread = threadIdx.x, lane = thread % THREADS_PER_WARP, row_count = select->row_count;
    const uint32_t step_rows =
	    ((const struct step_block *)(arena + select->step_offset))
		    ->row_count;
    const uint32_t live_rows = step_rows < row_count ? step_rows : row_count;
    const uint32_t first_expert = score_block * SCORE_EXPERTS;
    const uint4 *const weights =
        (const uint4 *)(arena + arguments->router_weights_offset) + (size_t)first_expert * STREAM_CHUNKS;
    const uint4 *const rows = (const uint4 *)(arena + arguments->hidden_offset);
    constexpr bool is_held = row_capacity == 1u;            /* the weights wait for the rows in registers, or in L2 */
    uint4 held[is_held ? SCORE_STEPS : 1u][SCORE_EXPERTS];
    LINEAR_DEVICE_CHECK(step_rows >= 1u && step_rows <= row_count && select->reference_order <= 1u);
#pragma unroll
    for (uint32_t step = 0; step < SCORE_STEPS; step++) {
        const uint32_t chunk = thread + step * FRONT_THREADS;
        for (uint32_t expert = 0; expert < SCORE_EXPERTS && chunk < STREAM_CHUNKS; expert++) {
            const uint4 *const weight = weights + (size_t)expert * STREAM_CHUNKS + chunk;
            if (is_held) {
                held[is_held ? step : 0u][expert] = __ldcs(weight);
            } else if (chunk % LINE_CHUNKS == 0) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(weight));
            }
        }
    }
    for (uint32_t expert = thread; expert < EXPERTS; expert += FRONT_THREADS) {   /* for the select, if last */
        bias[expert] = ((const float *)(arena + select->bias_offset))[expert];
    }
    for (uint32_t line = thread; line < TABLE_LINES; line += FRONT_THREADS) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(arena + select->expert_offsets_offset + line * LINE_BYTES));
    }
    if (is_own_input) {                                     /* the row's input, as its row block forms it */
        uint4 loads[ROW_PARTS][HYPER_STREAMS], packed[ROW_PARTS];
        front_input_load(arguments, arena, 0, pre, loads);
        if (front_publish(&scratch->arrived, arrivals, &is_last)) {
            front_finish(arguments, arena, scratch);
        }
        front_input(arguments, arena, pre, loads, row_sums, packed);
#pragma unroll
        for (uint32_t part = 0; part < ROW_PARTS; part++) {
            const uint32_t logical = thread + part * FRONT_THREADS;
            if (logical < ROW_THREADS) {
                inputs[is_own_input ? logical : 0u] = packed[part];
            }
        }
    } else if (thread == 0) {
        front_wait(&scratch->rows_ready, row_count);
    }
    __syncthreads();
    float sums[SCORE_EXPERTS][row_capacity] = {};
#pragma unroll
    for (uint32_t step = 0; step < SCORE_STEPS; step++) {
        const uint32_t chunk = thread + step * FRONT_THREADS;   /* chunks t, t + 256, t + 512 as the product's */
        if (chunk < STREAM_CHUNKS) {
            uint4 values[row_capacity], expert_weights[SCORE_EXPERTS];
            for (uint32_t row = 0; row < row_capacity; row++) {
                const size_t read_row = row < row_count ? row : 0u;
                values[row] = is_own_input ? inputs[is_own_input ? chunk : 0u]
                                           : __ldcg(rows + read_row * STREAM_CHUNKS + chunk);
            }
            for (uint32_t expert = 0; expert < SCORE_EXPERTS; expert++) {
                expert_weights[expert] = is_held ? held[is_held ? step : 0u][expert]
                                                 : __ldcs(weights + (size_t)expert * STREAM_CHUNKS + chunk);
            }
#pragma unroll
            for (uint32_t index = 0; index < CHUNK_VALUES; index++) {
                for (uint32_t row = 0; row < row_capacity; row++) {
                    const float value = front_bfloat16(values[row], index);
                    for (uint32_t expert = 0; expert < SCORE_EXPERTS; expert++) {
                        sums[expert][row] =
                            fmaf(front_bfloat16(expert_weights[expert], index), value, sums[expert][row]);
                    }
                }
            }
        }
    }
    for (uint32_t distance = FIRST_DISTANCE; distance != 0; distance >>= 1u) {
        for (uint32_t row = 0; row < row_capacity; row++) {
            for (uint32_t expert = 0; expert < SCORE_EXPERTS; expert++) {
                sums[expert][row] = sums[expert][row] + __shfl_xor_sync(ALL_BITS, sums[expert][row], (int)distance);
            }
        }
    }
    for (uint32_t row = 0; lane == 0 && row < row_capacity; row++) {
        for (uint32_t expert = 0; expert < SCORE_EXPERTS; expert++) {
            warp_sums[thread / THREADS_PER_WARP][expert * row_capacity + row] = sums[expert][row];
        }
    }
    __syncthreads();
    const uint32_t expert = first_expert + thread / row_capacity, row = thread % row_capacity;
    if (thread < SCORE_EXPERTS * row_capacity && row < row_count) {
        float score = 0.0f;
        for (uint32_t warp = 0; warp < DENSE_WARPS; warp++) {
            score = score + warp_sums[warp][thread];
        }
        ((float *)(arena + select->scores_offset))[(size_t)row * EXPERTS + expert] = score;
        if (row < live_rows) {
		((float *)(arena +
			   select->affinities_offset))[(size_t)row * EXPERTS +
						       expert] =
			sqrtf(score > ROUTER_SOFTPLUS_THRESHOLD ?
				      score :
				      log1pf(expf(score)));
	}
        __threadfence();
    }
    if (!front_publish(&scratch->scored, SCORE_BLOCKS, &is_last)) {
        if (is_own_input) {
            front_quantize_share(arguments, arena, inputs, score_block);
        }
        return;
    }
    __threadfence();
    for (uint32_t index = thread; index < live_rows * EXPERTS; index += FRONT_THREADS) {
        affinities[index] = __ldcg((const float *)(arena + select->affinities_offset) + index);
    }
    __syncthreads();
    {                                           /* the select's own statements, with their names */
        const uint32_t warp = thread / THREADS_PER_WARP, pair = thread, pair_count = live_rows * PICKS;
        ROUTER_SELECT_JOBS((*select), bias, EXPERTS, PICKS);
    }
    if (is_own_input) {
        front_quantize_share(arguments, arena, inputs, score_block);
    }
    if (thread == 0) {
        scratch->scored = 0;
        scratch->rows_ready = 0;            /* where none waits for it, it is zero already */
    }
}

/* The fronts of site_front_records.h: the parts of the blocks by index, but those of an FFN front of several rows,
 * whose score blocks wait for the row blocks, by ticket. */
template <uint32_t row_capacity, uint32_t kind>
__device__ __forceinline__ void
site_front(struct site_front_arguments arguments, unsigned char *arena)
{
	constexpr bool has_rows = kind != SITE_FRONT_KIND_COEFFICIENTS;
	constexpr bool has_mixes = kind != SITE_FRONT_KIND_INPUT;
	constexpr bool has_router = kind == SITE_FRONT_KIND_FFN,
		       is_ticketed = has_router && row_capacity != 1u;
	constexpr uint32_t mix_parts = has_router ? FFN_FRONT_MIX_PARTS :
						    SITE_FRONT_MIX_PARTS;
	__shared__ uint32_t ticket;
	struct site_front_scratch *const scratch =
		(struct site_front_scratch *)(arena + arguments.scratch_offset);
	const uint32_t row_count = arguments.select.row_count;
	constexpr uint32_t row_parts =
		kind == SITE_FRONT_KIND_INPUT ? SITE_FRONT_INPUT_PARTS : 1u;
	const uint32_t row_blocks =
		has_rows && !(has_router && row_capacity == 1u) ?
			row_count * row_parts :
			0u;
	const uint32_t mix_total =
		row_blocks + (has_mixes ? MODEL_MIXES * mix_parts : 0u);
	const uint32_t arrivals =
		mix_total + (has_router && !is_ticketed ?
				     SCORE_BLOCKS :
				     0u); /* readers of the pre */
	uint32_t part = blockIdx.x;
	LINEAR_DEVICE_CHECK(blockDim.x == FRONT_THREADS &&
			    gridDim.x == SITE_FRONT_BLOCKS(kind, row_count) &&
			    row_count != 0 && row_count <= row_capacity);
	if (is_ticketed) {
		if (threadIdx.x == 0) {
			ticket = atomicAdd(&scratch->tickets, 1u);
			if (ticket ==
			    gridDim.x - 1u) { /* every block holds its ticket */
				scratch->tickets = 0;
			}
		}
		__syncthreads();
		part = ticket;
	}
    if (part < row_blocks) {
        front_row(&arguments, arena, scratch, part / row_parts, part % row_parts, row_parts, arrivals, has_mixes,
                  is_ticketed);
    } else if (part < mix_total) {
        front_mix<row_capacity, mix_parts>(&arguments, arena, scratch, part - row_blocks, arrivals);
    } else if (has_router) {
        front_scores<row_capacity>(&arguments, arena, scratch, part - mix_total, arrivals);
    }
}

/* Two blocks a multiprocessor (as the dense product's): the FFN front's blocks are all resident on a CMP at once; a
 * build that would not fit the registers that leaves spills and fails. */
#define FRONT_BLOCKS_PER_MULTIPROCESSOR 2u
#define SITE_FRONT_STAMP(name, rows, kind)                                 \
	extern "C" __global__ void __launch_bounds__(                      \
		SITE_FRONT_THREADS, FRONT_BLOCKS_PER_MULTIPROCESSOR)       \
		name##_front_##rows(struct site_front_arguments arguments, \
				    unsigned char *arena)                  \
	{                                                                  \
		site_front<rows, kind>(arguments, arena);                  \
	}
SITE_FRONT_EVERY_INSTANCE
#undef SITE_FRONT_STAMP
