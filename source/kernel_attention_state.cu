/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_attention_state.cu: the attention state at 16 bits: the rotation step, the attention prepare, its window part
 * alone for the draft's context, the queries, the compressor with its pool entry, and the index key. Device code only,
 * built once per architecture into the module "kernel_attention_state". Contract:
 * include/anemone/attention_state_records.h. Here:
 *   - A thread takes the pairs of dims 2 p, 2 p + 1, p = thread + 128 k: a word of bfloat16 or FP16 values, and the
 *     two dims a rotation turns together. A norm reads its row twice (the squares, then the values), so that no
 *     array of a thread is indexed at run time.
 *   - A block reaches its barriers as a whole: it leaves before them, if at all, by its row or its part alone.
 */
#include "kernel_linear_common.cuh"

#include "anemone/attention_state_records.h"

#define STATE_THREADS ATTENTION_STATE_THREADS
#define STATE_WARPS (ATTENTION_STATE_THREADS / THREADS_PER_WARP)
#define EVERY_LANE 0xFFFFFFFFu
#define PAIR MODEL_PAIR_VALUES
#define HEAD_DIMS MODEL_HEAD_DIMS
#define HEAD_PAIRS (MODEL_HEAD_DIMS / PAIR)
#define FIRST_ROTATED_PAIR (MODEL_FIRST_ROTATED_DIM / PAIR)
#define INDEX_PAIRS (MODEL_INDEX_HEAD_DIMS / PAIR)
#define INDEX_FIRST_ROTATED_PAIR \
	((MODEL_INDEX_HEAD_DIMS - MODEL_ROTATED_DIMS) / PAIR)
#define QUERY_RANK_PAIRS (MODEL_QUERY_RANK / PAIR)
#define WINDOW_PART 1u                          /* grid y of the prepare's window row; 0 is qr */
#define FLOAT32_LOWEST_BITS 0xFF7FFFFFu         /* where the reference's softmax maximum starts */
#define STEP_DIMS 16u                           /* of index queries and keys: one instruction's dims */
#define HALF_STEP_DIMS 8u
#define HEADS_PER_TILE 16u                      /* index queries: [tile][step][lane 4 g + t][register] */
#define INDEX_STEPS (MODEL_INDEX_HEAD_DIMS / STEP_DIMS)
#define QUERY_REGISTERS 4u
#define KEY_RUN_WORDS                             \
	(MODEL_KEY_BYTES / LANES_PER_LANE_GROUP / \
	 sizeof(uint32_t)) /* run t of a key */

__device__ static __forceinline__ const struct step_row_record *
state_step_row(const unsigned char *arena, uint64_t step_offset, uint32_t row)
{
	return (const struct step_row_record *)(arena + step_offset +
						STEP_BLOCK_BYTES_VALUE) +
	       row;
}

/* The step row of a block that writes state, or NULL beyond the step's rows; the checked modules trap on a step or
 * a row step_records.h forbids. */
__device__ static __forceinline__ const struct step_row_record *
state_writer(const unsigned char *arena, uint64_t step_offset,
	     uint32_t session_capacity, uint32_t context_token_limit)
{
	const uint32_t row_count =
		((const struct step_block *)(arena + step_offset))->row_count;
	LINEAR_DEVICE_CHECK(blockDim.x == STATE_THREADS && row_count != 0 &&
			    row_count <= gridDim.x);
	if (blockIdx.x >= row_count) {
		return NULL;
	}
	const struct step_row_record *const own =
		state_step_row(arena, step_offset, blockIdx.x);
	LINEAR_DEVICE_CHECK(own->position >= 0 &&
			    (uint32_t)own->position < context_token_limit &&
			    own->session_slot < session_capacity);
	return own;
}

/* Values 2 pair and 2 pair + 1 of a row of float32, or of bfloat16 values. */
__device__ static __forceinline__ float2 state_pair(const void *row, uint32_t pair, bool is_float32)
{
    if (is_float32) {
        return make_float2(((const float *)row)[pair * PAIR], ((const float *)row)[pair * PAIR + 1u]);
    }
    const uint32_t word = ((const uint32_t *)row)[pair];
    return make_float2(linear_bfloat16_value(word), linear_bfloat16_value(word >> BITS_PER_HALF));
}

__device__ static __forceinline__ uint32_t state_bfloat16_pair(float2 value)
{
    return linear_bfloat16_bits(value.x) | (linear_bfloat16_bits(value.y) << BITS_PER_HALF);
}

/* The checked modules trap on a finite value that becomes an infinity (risk 8). */
__device__ static __forceinline__ uint32_t state_half_pair(float2 value)
{
    const __half low = __float2half_rn(value.x), high = __float2half_rn(value.y);
    LINEAR_DEVICE_CHECK((__hisinf(low) == 0 || isinf(value.x)) && (__hisinf(high) == 0 || isinf(value.y)));
    return (uint32_t)__half_as_ushort(low) | ((uint32_t)__half_as_ushort(high) << BITS_PER_HALF);
}

/* value turned by the angle of a row and use: rotation is the step's [row][3][32][2]. */
__device__ static __forceinline__ float2 state_rotate(float2 value, const unsigned char *rotation, size_t row,
                                                      uint32_t use, uint32_t rotated_pair)
{
	const float2 angle =
		((const float2 *)(rotation + row * MODEL_ROTATION_ROW_BYTES))
			[use * MODEL_FREQUENCIES + rotated_pair];
	return make_float2(fmaf(value.x, angle.x, -(value.y * angle.y)),
			   fmaf(value.y, angle.x, value.x * angle.y));
}

/* The norm's factor f of a row of pair_count pairs; once per block. */
__device__ static __forceinline__ float state_norm_factor(const void *row, uint32_t pair_count, bool is_float32)
{
    __shared__ float warp_sums[STATE_WARPS];
    float partial = 0.0f, sum = 0.0f;
    for (uint32_t pair = threadIdx.x; pair < pair_count; pair += STATE_THREADS) {
        const float2 value = state_pair(row, pair, is_float32);
        partial = partial + value.x * value.x;
        partial = partial + value.y * value.y;
    }
    for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0; distance >>= 1u) {
        partial = partial + __shfl_xor_sync(EVERY_LANE, partial, (int)distance);
    }
    if (threadIdx.x % THREADS_PER_WARP == 0) {
        warp_sums[threadIdx.x / THREADS_PER_WARP] = partial;
    }
    __syncthreads();
    for (uint32_t warp = 0; warp < STATE_WARPS; warp++) {
        sum = sum + warp_sums[warp];
    }
    return rsqrtf(sum * (1.0f / (float)(pair_count * PAIR)) +
		  MODEL_NORM_EPSILON);
}

__device__ static __forceinline__ float2 state_normed(const void *row, uint32_t pair, bool is_float32, float factor,
                                                      const unsigned char *weights)
{
    const float2 value = state_pair(row, pair, is_float32), weight = state_pair(weights, pair, false);
    return make_float2(weight.x * (value.x * factor), weight.y * (value.y * factor));
}

/* The prepare's window part of the block's row (blockIdx.x): nothing for a row beyond the step's. */
__device__ static __forceinline__ void
state_window(struct attention_prepare_arguments arguments, unsigned char *arena,
	     size_t row)
{
	const struct step_row_record *own = state_writer(
		arena, arguments.step_offset, arguments.session_capacity,
		arguments.context_token_limit);
	if (own == NULL) {
		return;
	}
    const uint32_t *input = (const uint32_t *)(arena + arguments.window_input_offset) + row * HEAD_PAIRS;
    uint32_t *ring_row =
	    (uint32_t *)(arena + arguments.ring_offset +
			 own->session_slot * arguments.ring_session_stride) +
	    (size_t)((uint32_t)own->position % MODEL_RING_SLOTS) * HEAD_PAIRS;
    const float factor = state_norm_factor(input, HEAD_PAIRS, false);
    for (uint32_t pair = threadIdx.x; pair < HEAD_PAIRS; pair += STATE_THREADS) {
        float2 value = state_normed(input, pair, false, factor, arena + arguments.window_norm_offset);
        if (pair >= FIRST_ROTATED_PAIR) {
            value = state_rotate(value, arena + arguments.rotation_offset, row, arguments.rotation_use,
                                 pair - FIRST_ROTATED_PAIR);
        }
        ring_row[pair] = state_half_pair(value);
    }
}

extern "C" __global__ void
attention_rotation(struct attention_rotation_arguments arguments,
		   unsigned char *arena)
{
	const uint32_t use = threadIdx.x / MODEL_FREQUENCIES,
		       frequency = threadIdx.x % MODEL_FREQUENCIES;
	LINEAR_DEVICE_CHECK(blockDim.x == ATTENTION_ROTATION_THREADS);
	const int32_t position =
		state_step_row(arena, arguments.step_offset, blockIdx.x)
			->position -
		(use == ATTENTION_ROTATION_GROUP_START ? 1 : 0);
	const float angle =
		(float)position *
		((const float *)(arena + arguments.frequencies_offset))
			[(use == ATTENTION_ROTATION_WINDOW_ONLY ?
				  0 :
				  MODEL_FREQUENCIES) +
			 frequency];
	((float2 *)(arena + arguments.rotation_offset +
		    (size_t)blockIdx.x *
			    MODEL_ROTATION_ROW_BYTES))[threadIdx.x] =
		make_float2(cosf(angle), sinf(angle));
}

extern "C" __global__ void
attention_prepare_entry(struct attention_prepare_arguments arguments,
			unsigned char *arena)
{
    const size_t row = blockIdx.x;
    LINEAR_DEVICE_CHECK(blockDim.x == STATE_THREADS);
    if (blockIdx.y != WINDOW_PART) {
        const uint32_t *input = (const uint32_t *)(arena + arguments.query_rank_offset) + row * QUERY_RANK_PAIRS;
        uint32_t *output = (uint32_t *)(arena + arguments.query_normed_offset) + row * QUERY_RANK_PAIRS;
        const float factor = state_norm_factor(input, QUERY_RANK_PAIRS, false);
        for (uint32_t pair = threadIdx.x; pair < QUERY_RANK_PAIRS; pair += STATE_THREADS) {
            output[pair] = state_bfloat16_pair(state_normed(input, pair, false, factor,
                                                            arena + arguments.query_norm_offset));
        }
        return;
    }
    state_window(arguments, arena, row);
}

/* The window part alone (records: window): the prepare's record, grid (rows, 1). */
extern "C" __global__ void
attention_window(struct attention_prepare_arguments arguments,
		 unsigned char *arena)
{
    LINEAR_DEVICE_CHECK(blockDim.x == STATE_THREADS && gridDim.y == 1);
    state_window(arguments, arena, blockIdx.x);
}

extern "C" __global__ void
attention_queries_entry(struct attention_queries_arguments arguments,
			unsigned char *arena)
{
    const size_t row = blockIdx.y;
    const uint32_t head = blockIdx.x, pair = threadIdx.x;
    LINEAR_DEVICE_CHECK(
	    blockDim.x == STATE_THREADS &&
	    arguments.index_head_count <= MODEL_INDEX_HEADS &&
	    gridDim.x == arguments.head_count + arguments.index_head_count &&
	    arguments.first_head + arguments.head_count <= MODEL_HEADS_VALUE &&
	    arguments.heads_first_head + arguments.head_count <=
		    arguments.heads_row_heads);
    if (head < arguments.head_count) {
	    const size_t first_pair =
		    (row * MODEL_HEADS_VALUE + arguments.first_head + head) *
		    HEAD_PAIRS;
	    const size_t input_pair = (row * arguments.heads_row_heads +
				       arguments.heads_first_head + head) *
				      HEAD_PAIRS;
	    const uint32_t *input =
		    (const uint32_t *)(arena + arguments.heads_offset) +
		    input_pair;
	    uint32_t *output =
		    (uint32_t *)(arena + arguments.queries_offset) + first_pair;
	    for (uint32_t head_pair = pair; head_pair < HEAD_PAIRS;
		 head_pair += STATE_THREADS) {
		    float2 value = state_pair(input, head_pair, false);
		    if (head_pair >= FIRST_ROTATED_PAIR) {
			    const float2 rotated = state_rotate(
				    value, arena + arguments.rotation_offset,
				    row, arguments.rotation_use,
				    head_pair - FIRST_ROTATED_PAIR);
			    value = make_float2(
				    linear_bfloat16_value(
					    linear_bfloat16_bits(rotated.x)),
				    linear_bfloat16_value(
					    linear_bfloat16_bits(rotated.y)));
		    }
		    output[head_pair] = state_half_pair(value);
	    }
        return;
    }
    const uint32_t index_head = head - arguments.head_count, dim = pair * PAIR, in_step = dim % STEP_DIMS;
    const uint32_t in_tile = index_head % HEADS_PER_TILE;
    const uint32_t lane = in_tile % LANE_GROUPS_PER_WARP * LANES_PER_LANE_GROUP + in_step % HALF_STEP_DIMS / PAIR;
    const uint32_t word = ((index_head / HEADS_PER_TILE * INDEX_STEPS + dim / STEP_DIMS) * THREADS_PER_WARP + lane) *
                              QUERY_REGISTERS + in_step / HALF_STEP_DIMS * PAIR + in_tile / LANE_GROUPS_PER_WARP;
    if (pair < INDEX_PAIRS) {
	    float2 value = state_pair(
		    (const uint32_t *)(arena + arguments.index_heads_offset) +
			    (row * MODEL_INDEX_HEADS + index_head) *
				    INDEX_PAIRS,
		    pair, false);
	    if (pair >= INDEX_FIRST_ROTATED_PAIR) {
		    value = state_rotate(value,
					 arena + arguments.rotation_offset, row,
					 arguments.rotation_use,
					 pair - INDEX_FIRST_ROTATED_PAIR);
	    }
	    ((uint32_t *)(arena + arguments.index_queries_offset) +
	     row * MODEL_INDEX_QUERIES_ROW_BYTES / sizeof(uint32_t))[word] =
		    state_half_pair(value);
    }
}

/* One dim pooled from positions p - 1 and p, rounded to bfloat16. */
__device__ static __forceinline__ float state_pooled(float first_kv, float first_score, float kv, float score)
{
    float largest = __uint_as_float(FLOAT32_LOWEST_BITS);
    largest = largest < first_score ? first_score : largest;
    largest = largest < score ? score : largest;
    const float first_exponential = expf(first_score - largest), exponential = expf(score - largest);
    const float sum = (0.0f + first_exponential) + exponential;
    return linear_bfloat16_value(
        linear_bfloat16_bits((0.0f + first_kv * (first_exponential / sum)) + kv * (exponential / sum)));
}

extern "C" __global__ void
attention_compressor_entry(struct attention_compressor_arguments arguments,
			   unsigned char *arena)
{
    __shared__ float pooled[HEAD_DIMS];
    const size_t row = blockIdx.x;
    const struct step_row_record *own = state_writer(
	    arena, arguments.step_offset, arguments.session_capacity,
	    arguments.context_token_limit);
    LINEAR_DEVICE_CHECK(arguments.ratio == 1 ||
			arguments.ratio == MODEL_POOLED_RATIO);
    LINEAR_DEVICE_CHECK(arguments.slot_mode <= ATTENTION_COMPRESSOR_SLOTS_KEEP);
    if (own == NULL) {
        return;
    }
    const uint32_t position = (uint32_t)own->position;
    if (arguments.slot_mode == ATTENTION_COMPRESSOR_SLOTS_KEEP &&
	(arguments.ratio != MODEL_POOLED_RATIO ||
	 (position + 1u) % arguments.ratio == 0)) {
	    return;
    }
    const size_t value_bytes = arguments.ratio == 1 ? sizeof(uint16_t) : sizeof(float);    /* bfloat16, float32 */
    const float *kv = (const float *)(arena + arguments.kv_offset + row * HEAD_DIMS * value_bytes);
    const float *score = (const float *)(arena + arguments.score_offset) + row * HEAD_DIMS;
    float *slots = (float *)(arena + arguments.slots_offset + own->session_slot * arguments.slots_session_stride);
    const size_t slot_floats = ATTENTION_SLOT_VECTORS * HEAD_DIMS;
    if (arguments.ratio == 1) {
        for (uint32_t dim = threadIdx.x; dim < HEAD_DIMS; dim += STATE_THREADS) {
            pooled[dim] = linear_bfloat16_value(((const uint16_t *)kv)[dim]);
        }
    } else if ((position + 1u) % arguments.ratio != 0) {
	    /* a launch of more rows than slots (a chunk's pool, attention_pool_prepare): its rows' partners lie
         * in it, read from the arrays; only its last row keeps its kv and score for the next launch's partner */
	    const uint32_t row_count =
		    ((const struct step_block *)(arena + arguments.step_offset))
			    ->row_count;
	    if (arguments.slot_mode == ATTENTION_COMPRESSOR_SLOTS_DEFER ||
		(arguments.slot_mode == ATTENTION_COMPRESSOR_SLOTS_INLINE &&
		 row_count > ATTENTION_COMPRESSOR_SLOTS &&
		 row + 1u != row_count)) {
		    return;
	    }
	    if (arguments.slot_mode == ATTENTION_COMPRESSOR_SLOTS_KEEP) {
		    /* A BATCH prompt owner can have more than sixteen rows. Only the last writer of each slot may
             * retain it. This launch follows pool computation, so no earlier row still reads the old slot. */
		    for (uint32_t other = (uint32_t)row + 1u; other < row_count;
			 other++) {
			    const struct step_row_record *const later =
				    state_step_row(arena, arguments.step_offset,
						   other);
			    if (later->session_slot == own->session_slot &&
				(uint32_t)later->position %
						ATTENTION_COMPRESSOR_SLOTS ==
					position % ATTENTION_COMPRESSOR_SLOTS) {
				    return;
			    }
		    }
	    }
	    float *kept =
		    slots + position % ATTENTION_COMPRESSOR_SLOTS * slot_floats;
	    for (uint32_t dim = threadIdx.x; dim < HEAD_DIMS;
		 dim += STATE_THREADS) {
		    kept[dim] = kv[dim];
		    kept[HEAD_DIMS + dim] = score[dim];
	    }
        return;
    } else {
	    const float *first_kv = slots + (position - 1u) %
						    ATTENTION_COMPRESSOR_SLOTS *
						    slot_floats;
	    const float *first_score = first_kv + HEAD_DIMS;
	    const uint32_t row_count =
		    ((const struct step_block *)(arena + arguments.step_offset))
			    ->row_count;
	    /* the partner row of this launch (its kv and score, the slot's words): the row before first (a chunk's
         * rows lie in order), else the search over the step's rows */
	    const struct step_row_record *const before =
		    row != 0 ? state_step_row(arena, arguments.step_offset,
					      (uint32_t)row - 1u) :
			       own;
	    const bool is_before = before->session_slot == own->session_slot &&
				   before->position == own->position - 1;
	    for (uint32_t other = is_before ? (uint32_t)row - 1u : 0u;
		 other < (is_before ? (uint32_t)row : row_count); other++) {
		    const struct step_row_record *const candidate =
			    state_step_row(arena, arguments.step_offset, other);
		    if (candidate->session_slot == own->session_slot &&
			candidate->position == own->position - 1) {
			    first_kv = (const float *)(arena +
						       arguments.kv_offset) +
				       (size_t)other * HEAD_DIMS;
			    first_score =
				    (const float *)(arena +
						    arguments.score_offset) +
				    (size_t)other * HEAD_DIMS;
		    }
        }
        for (uint32_t dim = threadIdx.x; dim < HEAD_DIMS; dim += STATE_THREADS) {
            pooled[dim] = state_pooled(first_kv[dim], first_score[dim], kv[dim], score[dim]);
        }
    }
    __syncthreads();
    const float factor = state_norm_factor(pooled, HEAD_PAIRS, true);
    const uint32_t use = arguments.ratio == 1 ? ATTENTION_ROTATION_POOLED :
						ATTENTION_ROTATION_GROUP_START;
    uint32_t *latent = (uint32_t *)(arena + arguments.latent_offset) + row * HEAD_PAIRS;
    uint32_t *entry =
        (uint32_t *)(arena + arguments.pool_offset + own->session_slot * arguments.pool_session_stride) +
        (size_t)(position / arguments.ratio) * HEAD_PAIRS;
    for (uint32_t pair = threadIdx.x; pair < HEAD_PAIRS; pair += STATE_THREADS) {
        float2 value = state_normed(pooled, pair, true, factor, arena + arguments.norm_offset);
        latent[pair] = state_bfloat16_pair(value);
        if (pair >= FIRST_ROTATED_PAIR) {
            value = state_rotate(value, arena + arguments.rotation_offset, row, use, pair - FIRST_ROTATED_PAIR);
        }
        entry[pair] = state_half_pair(value);
    }
}

extern "C" __global__ void
attention_key_entry(struct attention_key_arguments arguments,
		    unsigned char *arena)
{
	const struct step_row_record *own = state_writer(
		arena, arguments.step_offset, arguments.session_capacity,
		arguments.context_token_limit);
	LINEAR_DEVICE_CHECK(arguments.ratio == 1 ||
			    arguments.ratio == MODEL_POOLED_RATIO);
	if (own == NULL ||
	    ((uint32_t)own->position + 1u) % arguments.ratio != 0) {
		return;
	}
    const uint32_t position = (uint32_t)own->position, pair = threadIdx.x, dim = pair * PAIR;
    const uint32_t *input = (const uint32_t *)(arena + arguments.key_input_offset) + blockIdx.x * INDEX_PAIRS;
    const float factor = state_norm_factor(input, INDEX_PAIRS, false);
    if (pair < INDEX_PAIRS) {
        float2 value = state_normed(input, pair, false, factor, arena + arguments.norm_offset);
        if (pair >= INDEX_FIRST_ROTATED_PAIR) {
		value = state_rotate(value, arena + arguments.rotation_offset,
				     blockIdx.x,
				     arguments.ratio == 1 ?
					     ATTENTION_ROTATION_POOLED :
					     ATTENTION_ROTATION_GROUP_START,
				     pair - INDEX_FIRST_ROTATED_PAIR);
	}
        uint32_t *key = (uint32_t *)(arena + arguments.keys_offset + own->session_slot *
                                                                         arguments.keys_session_stride) +
                        (size_t)(position / arguments.ratio) * INDEX_PAIRS;
        key[dim % HALF_STEP_DIMS / PAIR * KEY_RUN_WORDS + dim / STEP_DIMS * PAIR + dim % STEP_DIMS / HALF_STEP_DIMS] =
            state_half_pair(value);
    }
}
