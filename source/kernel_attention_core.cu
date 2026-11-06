/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_attention_core.cu: the attention core of one layer with FP16 state: each row's window rows and selected
 * pool entries, its sink, and its output turned back. Device code only, built once per architecture into the module
 * "kernel_attention_core". The FP16 kind of batch i11b's probe (tools/measure/attention_core_kernels.cuh) at 8 heads
 * a block and 2 stages: 0.72 of 16 heads' time on the cards of class 80 and 0.85 on class 120 at 1 and 8 rows, bit
 * for bit the same; the epilogue rounds as REF:780-781 does. The fixed order's kernel, and the reference order's two
 * (its scores, then its softmax and values), whose arithmetic on class 120 in the exact profile is
 * kernel_attention_exact.cuh's.
 *
 * Contract: include/anemone/attention_core_records.h (what is computed, the arrays, the launch). Here:
 *   - A block is one row and 8 heads; warp w takes slice w of the 512 dims. The tensor-core instruction m16n8k16 of
 *     16-bit floats (kernel_linear_common.cuh) has the keys or the dims as its 16 rows and the 8 heads as its
 *     columns: scores^T = K Q^T, then O^T += V^T P^T. Lane 4 g + t holds keys (dims) g and g + 8 of heads 2 t and
 *     2 t + 1. The queries and O stay in registers.
 *   - A group's 16 rows come into a ring of 2 stages of shared memory by 16-byte asynchronous copies, those of the
 *     next group while this one is computed; a chunk c of key k lies at chunk c ^ (k % 8), 8 keys across the banks.
 *     A key of index -1 is zero chunks, so no stale byte or not-a-number reaches P V, where its p of 0 meets it.
 *   - Shared memory: the ring, the slices' partial scores, P [head][key] (heads 4 .. 7 hold their two halves
 *     swapped, apart in the banks), the factors (the denominators at the end), the stages' indices. The epilogue
 *     stages each warp's 8 heads over the ring, which nothing reads by then.
 *   - Every thread of a block reaches every barrier, or none does: a row from row_count on leaves before the first,
 *     and so does a block with no group of keys of its row.
 *   - The reference order's scores: a block is a row, 8 heads and a tile of 4 groups; warp w takes the tile's group
 *     w through a stage of its own and forms its scores of the 4 slices, added as the fixed order adds the partials:
 *     the same bits. Its values: a block is a row, 8 heads and 128 dims; the softmax of the 8 heads from their scores
 *     by all its 8 warps (each block of the heads forms the same), then warp w's 16 dims of every group through a
 *     ring of its own,
 *     CORE_VALUE_STAGES groups on their way: O^T += V^T P^T, one instruction a group, no block barrier.
 */
#include "kernel_linear_common.cuh"

#include "anemone/attention_core_records.h"

#define CORE_HEADS ATTENTION_CORE_HEADS_PER_BLOCK
#define CORE_THREADS ATTENTION_CORE_THREADS
#define CORE_SLICES 4u /* of the dims: a warp each */
#define CORE_SLICE_DIMS 128u
#define CORE_STEP_DIMS 16u /* the inputs of one instruction */
#define CORE_STEPS 8u /* of a slice */
#define CORE_GROUP_KEYS 16u /* a softmax step */
#define CORE_HALF_GROUP 8u
#define CORE_KEY_HALVES 2u /* keys 0 .. 7 and 8 .. 15 of a group */
#define CORE_WINDOW_GROUPS (MODEL_WINDOW / CORE_GROUP_KEYS)
#define CORE_STAGES 2u /* of the fixed order's ring */
#define CORE_WARPS (CORE_THREADS / THREADS_PER_WARP)
#define CORE_INDICES_AHEAD \
	2u /* a key's index is read two groups before its group is computed */
#define CORE_ROW_BYTES MODEL_STATE_ROW_BYTES
#define CORE_CHUNK_BYTES 16u
#define CORE_CHUNK_HALVES 8u
#define CORE_ROW_CHUNKS (CORE_ROW_BYTES / CORE_CHUNK_BYTES)
#define CORE_SLICE_CHUNKS (CORE_SLICE_DIMS / CORE_CHUNK_HALVES)
#define CORE_STAGE_BYTES (CORE_GROUP_KEYS * CORE_ROW_BYTES)
#define CORE_SWIZZLE_MASK 7u
#define CORE_LINE_CHUNKS 8u /* of 128 bytes: the banks once */
#define CORE_LINE_BYTES (CORE_LINE_CHUNKS * CORE_CHUNK_BYTES)
#define CORE_THREADS_PER_KEY (CORE_THREADS / CORE_GROUP_KEYS)
#define CORE_WARP_COPY_KEYS \
	(THREADS_PER_WARP / CORE_THREADS_PER_KEY) /* of a copy instruction */
#define CORE_SOFTMAX_LANES 8u /* of a head: lane j takes keys j and j + 8 */
#define CORE_PARTIAL_PAD \
	4u /* floats behind a key's heads: keys apart in the banks */
#define CORE_PARTIAL_ROW (CORE_HEADS + CORE_PARTIAL_PAD)
#define CORE_STAGED_HALVES \
	136u /* a head's 128 dims and 8 more: heads apart in the banks */
#define CORE_PROBS_SWIZZLE_SHIFT 2u
#define CORE_MATRIX_ROWS 8u /* of an 8 x 8 matrix of ldmatrix */
#define CORE_OPERAND_B_REGISTERS 2u
#define CORE_ROTATED_FIRST_STEP \
	((MODEL_FIRST_ROTATED_DIM % CORE_SLICE_DIMS) / CORE_STEP_DIMS)
#define CORE_RING_MASK (MODEL_RING_SLOTS - 1u)
#define CORE_ROTATION_ROW_FLOATS (MODEL_ROTATION_ROW_BYTES / sizeof(float))
#define CORE_EVERY_LANE 0xFFFFFFFFu
#define CORE_PARTIALS_OFFSET \
	(CORE_STAGES * CORE_STAGE_BYTES) /* shared memory, in bytes */
#define CORE_PROBS_OFFSET       \
	(CORE_PARTIALS_OFFSET + \
	 CORE_SLICES * CORE_GROUP_KEYS * CORE_PARTIAL_ROW * sizeof(float))
#define CORE_FACTORS_OFFSET \
	(CORE_PROBS_OFFSET + CORE_HEADS * CORE_GROUP_KEYS * sizeof(uint16_t))
#define CORE_INDICES_OFFSET (CORE_FACTORS_OFFSET + CORE_HEADS * sizeof(float))
#define CORE_SHARED_END \
	(CORE_INDICES_OFFSET + CORE_STAGES * CORE_GROUP_KEYS * sizeof(int32_t))
#define CORE_KEYS ATTENTION_CORE_KEYS
/* the reference order's scores: a stage a warp, the heads' queries, the tile's rows */
#define CORE_TILE_GROUPS (ATTENTION_CORE_TILE_KEYS / CORE_GROUP_KEYS)
#define CORE_QUERY_ROW_BYTES (CORE_ROW_BYTES + CORE_CHUNK_BYTES)
#define CORE_TILE_QUERIES_OFFSET (CORE_TILE_GROUPS * CORE_STAGE_BYTES)
#define CORE_SCORES_SHARED_END(tiles)                  \
	(CORE_TILE_QUERIES_OFFSET +                    \
	 (tiles) * CORE_HEADS * CORE_QUERY_ROW_BYTES + \
	 ATTENTION_CORE_TILE_KEYS * sizeof(const unsigned char *))
/* its values: the heads' scores [head][key] (then p, then the epilogue's staging), the weights [group][head][key],
 * the row's keys' rows, the denominators, the heads' maxima, the groups' sums [head][group], a ring a warp of
 * CORE_VALUE_STAGES groups' 16 keys by its 16 dims */
#define CORE_VALUE_STAGES 8u
#define CORE_VALUE_THREADS ATTENTION_CORE_VALUE_THREADS
#define CORE_VALUE_WARPS (CORE_VALUE_THREADS / THREADS_PER_WARP)
#define CORE_WARP_DIMS (ATTENTION_CORE_TILE_DIMS / CORE_VALUE_WARPS)
#define CORE_GROUPS (CORE_KEYS / CORE_GROUP_KEYS)
#define CORE_VALUE_KEY_BYTES \
	(CORE_WARP_DIMS *    \
	 sizeof(uint16_t)) /* a key's dims in a stage: 2 chunks */
#define CORE_VALUE_SWIZZLE_KEYS \
	4u /* keys k and k + 4 of a stage would share their banks */
#define CORE_VALUE_KEY_CHUNKS (CORE_VALUE_KEY_BYTES / CORE_CHUNK_BYTES)
#define CORE_VALUE_AHEAD \
	2u /* groups of P V in registers: the one summed, the next */
#define CORE_TWO_TILES \
	2u /* tiles of 8 heads a score block: 1, these, the most */
#define CORE_FOUR_TILES 4u
#define CORE_CHUNK_FLOATS 4u
#define CORE_SOFTMAX_STRIDE \
	(THREADS_PER_WARP * CORE_CHUNK_FLOATS) /* keys: a warp's four a lane */
#define CORE_VALUE_STAGE_BYTES (CORE_GROUP_KEYS * CORE_VALUE_KEY_BYTES)
#define CORE_VALUE_PROBS_OFFSET \
	(CORE_HEADS * CORE_KEYS * sizeof(float)) /* each times the tiles */
#define CORE_VALUE_ROWS_OFFSET \
	(CORE_VALUE_PROBS_OFFSET + CORE_KEYS * CORE_HEADS * sizeof(uint16_t))
#define CORE_VALUE_ROWS_BYTES (CORE_KEYS * sizeof(const unsigned char *))
#define CORE_VALUE_HEAD_FLOATS \
	(2u +                  \
	 CORE_GROUPS) /* a head's denominator, its maximum, its groups' sums */
#define CORE_VALUES_SHARED_END(tiles)                                      \
	((tiles) * (CORE_VALUE_ROWS_OFFSET +                               \
		    CORE_HEADS * CORE_VALUE_HEAD_FLOATS * sizeof(float)) + \
	 CORE_VALUE_ROWS_BYTES +                                           \
	 CORE_VALUE_WARPS * CORE_VALUE_STAGES * CORE_VALUE_STAGE_BYTES)

static_assert(ATTENTION_CORE_SCORES_ROW_BYTES ==
		      MODEL_HEADS_VALUE * CORE_KEYS * sizeof(float),
	      "a row's scores: [head][key] in float32");
static_assert(CORE_SHARED_END == ATTENTION_CORE_SHARED_BYTES_VALUE &&
		      CORE_SCORES_SHARED_END(1u) ==
			      ATTENTION_CORE_SCORES_SHARED_BYTES(1u) &&
		      CORE_SCORES_SHARED_END(ATTENTION_CORE_SCORE_TILES_MOST) ==
			      ATTENTION_CORE_SCORES_SHARED_BYTES(
				      ATTENTION_CORE_SCORE_TILES_MOST) &&
		      CORE_VALUES_SHARED_END(1u) ==
			      ATTENTION_CORE_VALUES_SHARED_BYTES(1u) &&
		      CORE_VALUES_SHARED_END(ATTENTION_CORE_VALUE_TILES_MOST) ==
			      ATTENTION_CORE_VALUES_SHARED_BYTES(
				      ATTENTION_CORE_VALUE_TILES_MOST),
	      "the layouts are the record header's sizes");
static_assert(CORE_PROBS_OFFSET % CORE_CHUNK_BYTES == 0 &&
		      CORE_FACTORS_OFFSET % CORE_CHUNK_BYTES == 0 &&
		      CORE_INDICES_OFFSET % CORE_CHUNK_BYTES == 0 &&
		      CORE_VALUE_PROBS_OFFSET % CORE_CHUNK_BYTES == 0 &&
		      CORE_VALUE_ROWS_OFFSET % CORE_CHUNK_BYTES == 0 &&
		      CORE_QUERY_ROW_BYTES % CORE_CHUNK_BYTES == 0 &&
		      CORE_HEADS * CORE_VALUE_HEAD_FLOATS * sizeof(float) %
				      CORE_CHUNK_BYTES ==
			      0 &&
		      CORE_TILE_QUERIES_OFFSET % CORE_CHUNK_BYTES == 0,
	      "every part of shared memory begins on 16 bytes");
static_assert(
	CORE_TILE_GROUPS == CORE_WARPS &&
		CORE_KEYS % ATTENTION_CORE_TILE_KEYS == 0 &&
		CORE_WARP_DIMS == CORE_STEP_DIMS &&
		MODEL_HEAD_DIMS % ATTENTION_CORE_TILE_DIMS == 0 &&
		CORE_GROUP_KEYS * CORE_VALUE_KEY_BYTES / CORE_CHUNK_BYTES ==
			THREADS_PER_WARP,
	"a group of the tile a warp; an instruction's rows a warp; a group's copies of a warp a copy a lane");
static_assert(
	CORE_VALUE_WARPS * CORE_HEADS * CORE_WARP_DIMS * sizeof(uint16_t) <=
			CORE_VALUE_PROBS_OFFSET &&
		CORE_VALUE_WARPS * CORE_KEY_HALVES <= THREADS_PER_WARP &&
		MODEL_FIRST_ROTATED_DIM % CORE_WARP_DIMS == 0 &&
		CORE_VALUE_WARPS == CORE_HEADS,
	"the values' epilogue stages over the scores; a warp's dims are rotated or not; a warp a head's softmax");
static_assert(
	CORE_SWIZZLE_MASK == CORE_LINE_CHUNKS - 1u &&
		CORE_THREADS_PER_KEY == CORE_LINE_CHUNKS &&
		CORE_ROW_CHUNKS % CORE_LINE_CHUNKS == 0 &&
		CORE_GROUP_KEYS % CORE_WARP_COPY_KEYS == 0,
	"a key's chunks swizzled within their lines; a copy instruction whole lines of keys");
static_assert(
	CORE_KEYS % CORE_THREADS == 0 &&
		CORE_HEADS * CORE_GROUP_KEYS == CORE_THREADS,
	"the row's indices: a key of each round a thread; the reference order's softmax: a key a thread");
static_assert(CORE_THREADS / THREADS_PER_WARP * CORE_HEADS *
			      CORE_STAGED_HALVES * sizeof(uint16_t) <=
		      CORE_PARTIALS_OFFSET,
	      "the epilogue's staging lies over the ring alone");
static_assert(CORE_THREADS == CORE_SLICES * THREADS_PER_WARP &&
		      CORE_SLICES * CORE_SLICE_DIMS == MODEL_HEAD_DIMS &&
		      CORE_STEPS * CORE_STEP_DIMS == CORE_SLICE_DIMS,
	      "a warp a slice, an instruction a step");
static_assert(MODEL_FIRST_ROTATED_DIM / CORE_SLICE_DIMS == CORE_SLICES - 1u &&
		      MODEL_FIRST_ROTATED_DIM % CORE_STEP_DIMS == 0,
	      "the rotated dims are whole steps of the last slice");
static_assert(
	CORE_HEADS *CORE_SOFTMAX_LANES % THREADS_PER_WARP == 0 &&
		CORE_STAGES == CORE_KEY_HALVES &&
		MODEL_HEADS_VALUE % CORE_HEADS == 0 &&
		MODEL_SELECTED % CORE_GROUP_KEYS == 0,
	"whole warps run the softmax, the ring's two stages alternate, groups and heads divide evenly");

/* What one row reads: its session's ring and pool, its selection, and how much of the pool it sees. */
struct core_row {
	const unsigned char *ring;
	const unsigned char *pool; /* NULL without a pool */
	const int32_t *selection; /* NULL without a pool */
	int32_t position; /* the block core's: the block's last */
	int32_t block_first; /* the block core's: the block's first position */
	uint32_t visible; /* entries below (p + 1) / pool_ratio */
	uint32_t pool_entries; /* of a session's pool */
};

__device__ static __forceinline__ uint32_t core_shared_address(const void *place)
{
	return (uint32_t)__cvta_generic_to_shared(place);
}

/* 16 bytes into shared memory, past the registers and the L1: source_bytes (16 or 0) from the arena, zeros after. */
__device__ static __forceinline__ void
core_copy_chunk(void *place, const void *source, uint32_t source_bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(
			     core_shared_address(place)),
		     "l"(source), "n"((int)CORE_CHUNK_BYTES), "r"(source_bytes)
		     : "memory");
}

/* Every asynchronous copy the thread began has arrived. */
__device__ static __forceinline__ void core_wait_copies(void)
{
	asm volatile("cp.async.wait_all;\n" ::: "memory");
}

/* 8 x 8 matrices of 16-bit elements: lanes 8 m .. 8 m + 7 give the rows of matrix m; register m receives row g,
 * elements 2 t and 2 t + 1 (transposed: rows 2 t and 2 t + 1, element g). */
#define CORE_LOAD_MATRICES(function_name, instruction)                  \
	__device__ static __forceinline__ void function_name(           \
		uint32_t registers[REGISTERS_PER_OPERAND_A],            \
		const void *place)                                      \
	{                                                               \
		asm volatile(instruction " {%0, %1, %2, %3}, [%4];\n"   \
			     : "=r"(registers[LOW_OUTPUT_LOW_INPUTS]),  \
			       "=r"(registers[HIGH_OUTPUT_LOW_INPUTS]), \
			       "=r"(registers[LOW_OUTPUT_HIGH_INPUTS]), \
			       "=r"(registers[HIGH_OUTPUT_HIGH_INPUTS]) \
			     : "r"(core_shared_address(place))          \
			     : "memory");                               \
	}
CORE_LOAD_MATRICES(core_load_matrices,
		   "ldmatrix.sync.aligned.m8n8.x4.shared.b16")
CORE_LOAD_MATRICES(core_load_matrices_transposed,
		   "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")

__device__ static __forceinline__ void
core_load_two_matrices(uint32_t registers[CORE_OPERAND_B_REGISTERS],
		       const void *place)
{
	asm volatile(
		"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
		: "=r"(registers[0]), "=r"(registers[1])
		: "r"(core_shared_address(place))
		: "memory");
}

/* Where chunk `chunk` of key `key` lies in a stage. */
__device__ static __forceinline__ uint32_t core_place(uint32_t key,
						      uint32_t chunk)
{
	return key * CORE_ROW_BYTES +
	       (chunk ^ (key & CORE_SWIZZLE_MASK)) * CORE_CHUNK_BYTES;
}

/* The index of key `key` of the row: a position for the window, an entry for the pool; -1 for none. The block core's
 * (is_block): key k is position b - 128 + k of the block's first b, none below 0 or past the block's last. */
template <bool is_block = false>
__device__ static __forceinline__ int32_t
core_key_index(const struct core_row *row, uint32_t key)
{
	if (is_block) {
		const int32_t key_position =
			row->block_first - (int32_t)MODEL_WINDOW + (int32_t)key;
		return key_position < 0 || key_position > row->position ?
			       -1 :
			       key_position;
	}
	if (key < MODEL_WINDOW) {
		const int32_t key_position = row->position -
					     (int32_t)(MODEL_WINDOW - 1u) +
					     (int32_t)key;
		return key_position < 0 ? -1 : key_position;
	}
	const int32_t entry = row->selection[key - MODEL_WINDOW];
	LINEAR_DEVICE_CHECK(entry >= -1 && entry < (int32_t)row->visible);
	return entry >= 0 && (uint32_t)entry < row->pool_entries ? entry : -1;
}

/* The row of key `key` of index `index` (not -1); of the block core every key's is the ring's. */
template <bool is_block = false>
__device__ static __forceinline__ const unsigned char *
core_key_row(const struct core_row *row, uint32_t key, int32_t index)
{
	return is_block || key < MODEL_WINDOW ?
		       row->ring + (size_t)((uint32_t)index & CORE_RING_MASK) *
					   CORE_ROW_BYTES :
		       row->pool + (size_t)index * CORE_ROW_BYTES;
}

/* The copies of key `key` from its row key_row into a stage: of each line of 8 chunks (128 bytes, the banks once)
 * those from first_chunk on, chunk_step apart; zeros for a key_row of NULL, which reads nothing (stand_in, an arena
 * address, stands for its source). Unrolled, a copy is one instruction: its places in the lines apart by constants. */
__device__ static __forceinline__ void
core_copy_row(unsigned char *stage, uint32_t key, const unsigned char *key_row,
	      const unsigned char *stand_in, uint32_t first_chunk,
	      uint32_t chunk_step)
{
	const unsigned char *const source = key_row != NULL ? key_row :
							      stand_in;
	const uint32_t source_bytes = key_row != NULL ? CORE_CHUNK_BYTES : 0u;
#pragma unroll
	for (uint32_t taken = 0; taken < CORE_LINE_CHUNKS / chunk_step;
	     taken++) {
		const uint32_t chunk = first_chunk + taken * chunk_step;
		unsigned char *const place = stage + core_place(key, chunk);
#pragma unroll
		for (uint32_t line = 0;
		     line < CORE_ROW_CHUNKS / CORE_LINE_CHUNKS; line++) {
			core_copy_chunk(place + line * CORE_LINE_BYTES,
					source + chunk * CORE_CHUNK_BYTES +
						line * CORE_LINE_BYTES,
					source_bytes);
		}
	}
}

/* The output's head place (in heads from the output's start) of row row_index's head `head`: output_row_heads 0 the
 * full rows', else the rank's own rows' (head - first_head). */
__device__ static __forceinline__ size_t
core_output_head(const struct attention_core_arguments_record &arguments,
		 uint32_t row_index, uint32_t head)
{
	return arguments.output_row_heads == 0 ?
		       (size_t)row_index * MODEL_HEADS_VALUE + head :
		       (size_t)row_index * arguments.output_row_heads + head -
			       arguments.first_head;
}

/* core_copy_row of key `key` of group `group`, of index `index` (-1: none); the ring's first row the stand-in. */
template <bool is_block>
__device__ static __forceinline__ void
core_copy_key(const struct core_row *row, unsigned char *stage, uint32_t group,
	      uint32_t key, int32_t index, uint32_t first_chunk,
	      uint32_t chunk_step)
{
	core_copy_row(stage, key,
		      index >= 0 ?
			      core_key_row<is_block>(
				      row, group * CORE_GROUP_KEYS, index) :
			      NULL,
		      row->ring, first_chunk, chunk_step);
}

/* The copies of group `group` into its stage: thread i takes key i / 8, chunks i % 8, i % 8 + 8, ..; the first
 * thread of a key puts its index among the stage's. */
template <bool is_block>
__device__ static __forceinline__ void
core_issue(const struct core_row *row, unsigned char *stage,
	   int32_t *stage_indices, uint32_t group, int32_t index)
{
	const uint32_t key = threadIdx.x / CORE_THREADS_PER_KEY,
		       first_chunk = threadIdx.x % CORE_THREADS_PER_KEY;
	core_copy_key<is_block>(row, stage, group, key, index, first_chunk,
				CORE_THREADS_PER_KEY);
	if (first_chunk == 0) {
		stage_indices[key] = index;
	}
}

/* The warp's scores of one group over its slice, one chain from +0. Operand A, keys by dims, from the stage: the
 * matrices (keys 0 .. 7, 8 .. 15) by (the step's two chunks). */
__device__ static __forceinline__ void
core_scores(const unsigned char *stage, uint32_t slice, uint32_t lane,
	    const uint32_t queries[CORE_STEPS][CORE_OPERAND_B_REGISTERS],
	    float sums[ELEMENTS_PER_LANE])
{
	const uint32_t matrix = lane / CORE_MATRIX_ROWS;
	const uint32_t key = lane % CORE_MATRIX_ROWS +
			     matrix % CORE_KEY_HALVES * CORE_HALF_GROUP;
	uint32_t operands
		[CORE_STEPS]
		[REGISTERS_PER_OPERAND_A]; /* every load first: one wait, not one a step */
#pragma unroll
	for (uint32_t step = 0; step < CORE_STEPS; step++) {
		core_load_matrices(
			operands[step],
			stage + core_place(key,
					   (slice * CORE_SLICE_DIMS +
					    step * CORE_STEP_DIMS) /
							   CORE_CHUNK_HALVES +
						   matrix / CORE_KEY_HALVES));
	}
#pragma unroll
	for (uint32_t step = 0; step < CORE_STEPS; step++) {
		linear_tensor_core<LINEAR_FORM_CODES>(sums, operands[step],
						      queries[step][0],
						      queries[step][1]);
	}
}

/* The warp's scores of a group into the partials: keys g and g + 8 of heads 2 t and 2 t + 1. */
__device__ static __forceinline__ void
core_partials(const unsigned char *stage, uint32_t slice, uint32_t lane,
	      const uint32_t queries[CORE_STEPS][CORE_OPERAND_B_REGISTERS],
	      float *partials)
{
	const uint32_t lane_group = lane / LANES_PER_LANE_GROUP,
		       lane_place = lane % LANES_PER_LANE_GROUP;
	float sums[ELEMENTS_PER_LANE] = { 0.0f, 0.0f, 0.0f, 0.0f };
	core_scores(stage, slice, lane, queries, sums);
	float *slice_partials =
		partials +
		(slice * CORE_GROUP_KEYS + lane_group) * CORE_PARTIAL_ROW +
		lane_place * COLUMNS_PER_LANE;
	*(float2 *)slice_partials = make_float2(sums[LOW_OUTPUT_EVEN_COLUMN],
						sums[LOW_OUTPUT_ODD_COLUMN]);
	*(float2 *)(slice_partials + CORE_HALF_GROUP * CORE_PARTIAL_ROW) =
		make_float2(sums[HIGH_OUTPUT_EVEN_COLUMN],
			    sums[HIGH_OUTPUT_ODD_COLUMN]);
}

/* A head's scores of keys j and j + 8 of a group (lane j of its 8), and the group's largest in all 8 lanes. */
__device__ static __forceinline__ float
core_group_scores(const float *partials, const int32_t *indices, uint32_t head,
		  uint32_t head_lane, float scores[CORE_KEY_HALVES])
{
#pragma unroll
	for (uint32_t half = 0; half < CORE_KEY_HALVES; half++) {
		const uint32_t key = head_lane + half * CORE_HALF_GROUP;
		float sum = 0.0f;
#pragma unroll
		for (uint32_t slice = 0; slice < CORE_SLICES; slice++) {
			sum += partials[(slice * CORE_GROUP_KEYS + key) *
						CORE_PARTIAL_ROW +
					head];
		}
		scores[half] = indices[key] < 0 ? -INFINITY :
						  sum * ATTENTION_CORE_SCALE;
	}
	float largest = fmaxf(scores[0], scores[1]);
	for (uint32_t distance = 1u; distance < CORE_SOFTMAX_LANES;
	     distance <<= 1u) {
		largest =
			fmaxf(largest, __shfl_xor_sync(CORE_EVERY_LANE, largest,
						       (int)distance));
	}
	return largest;
}

/* A weight of P V as the order stores it: p in FP16; or p rounded to bfloat16, then times 2^14 in FP16. */
template <bool reference_order>
__device__ static __forceinline__ uint16_t core_weight(float probability)
{
	if (reference_order) {
		return __half_as_ushort(__float2half_rn(
			linear_bfloat16_value(
				linear_bfloat16_bits(probability)) *
			ATTENTION_CORE_WEIGHT_SCALE));
	}
	return __half_as_ushort(__float2half_rn(probability));
}

/* The fixed order's softmax of one group for one head by its 8 lanes. True where the head's maximum rose by more
 * than the step: O and the sum are then multiplied by factors[head]. */
__device__ static __forceinline__ bool
core_softmax(const float *partials, const int32_t *indices, uint16_t *probs,
	     float *factors, uint32_t head, uint32_t head_lane, float *maximum,
	     float *total)
{
	float scores[CORE_KEY_HALVES];
	const float largest =
		core_group_scores(partials, indices, head, head_lane, scores);
	const bool rescaled = largest > *maximum + ATTENTION_CORE_RESCALE_STEP;
	const float factor = rescaled ? expf(*maximum - largest) : 1.0f;
	*maximum = rescaled ? largest : *maximum;
	float group_sum = 0.0f;
#pragma unroll
	for (uint32_t half = 0; half < CORE_KEY_HALVES; half++) {
		const float probability = expf(scores[half] - *maximum);
		const uint32_t chunk =
			half ^ ((head >> CORE_PROBS_SWIZZLE_SHIFT) & 1u);
		probs[head * CORE_GROUP_KEYS + chunk * CORE_CHUNK_HALVES +
		      head_lane] = core_weight<false>(probability);
		group_sum += probability;
	}
	for (uint32_t distance = 1u; distance < CORE_SOFTMAX_LANES;
	     distance <<= 1u) {
		group_sum += __shfl_xor_sync(CORE_EVERY_LANE, group_sum,
					     (int)distance);
	}
	*total = *total * factor + group_sum;
	if (head_lane == 0) {
		factors[head] = factor;
	}
	return rescaled;
}

/* O^T += V^T P^T over the warp's slice: operand A (dims by keys) from the stage, transposed; operand B (keys by
 * heads) from P. */
__device__ static __forceinline__ void
core_values(const unsigned char *stage, const uint16_t *probs, uint32_t slice,
	    uint32_t lane, float values[CORE_STEPS][ELEMENTS_PER_LANE])
{
	const uint32_t matrix = lane / CORE_MATRIX_ROWS,
		       head = lane % CORE_MATRIX_ROWS;
	const uint32_t key = head + matrix / CORE_KEY_HALVES * CORE_HALF_GROUP;
	const uint32_t chunk = (matrix % CORE_KEY_HALVES) ^
			       ((head >> CORE_PROBS_SWIZZLE_SHIFT) & 1u);
	uint32_t operand_b[CORE_OPERAND_B_REGISTERS];
	core_load_two_matrices(operand_b, probs + head * CORE_GROUP_KEYS +
						  chunk * CORE_CHUNK_HALVES);
	uint32_t operands_a[CORE_STEPS][REGISTERS_PER_OPERAND_A];
#pragma unroll
	for (uint32_t step = 0; step < CORE_STEPS; step++) {
		core_load_matrices_transposed(
			operands_a[step],
			stage + core_place(key,
					   (slice * CORE_SLICE_DIMS +
					    step * CORE_STEP_DIMS) /
							   CORE_CHUNK_HALVES +
						   matrix % CORE_KEY_HALVES));
	}
#pragma unroll
	for (uint32_t step = 0; step < CORE_STEPS; step++) {
		linear_tensor_core<LINEAR_FORM_CODES>(values[step],
						      operands_a[step],
						      operand_b[0],
						      operand_b[1]);
	}
}

/* The bits of a value of O at dim `dim` of its head's 512: over the denominator (the reference order's O first times
 * 2^-14, exact), rounded to bfloat16; a rotated dim (448 .. 511) turned back from its pair's rounded values, the
 * partner partner_lanes lanes away (x1 of an even dim, x0 of an odd one), and rounded again. The lanes of a warp
 * call it together, their dims all rotated or none. */
template <bool reference_order>
__device__ static __forceinline__ uint32_t
core_output_bits(float value, float denominator, const float *rotation,
		 uint32_t dim, uint32_t partner_lanes)
{
	uint32_t bits = linear_bfloat16_bits(
		(reference_order ? value * ATTENTION_CORE_WEIGHT_UNSCALE :
				   value) /
		denominator);
	if (dim >= MODEL_FIRST_ROTATED_DIM) {
		const float rounded = linear_bfloat16_value(bits);
		const float partner = __shfl_xor_sync(CORE_EVERY_LANE, rounded,
						      (int)partner_lanes);
		const float *angle =
			rotation + (dim - MODEL_FIRST_ROTATED_DIM) /
					   MODEL_PAIR_VALUES *
					   MODEL_PAIR_VALUES;
		bits = linear_bfloat16_bits(
			dim % MODEL_PAIR_VALUES == 0 ?
				fmaf(rounded, angle[0], partner * angle[1]) :
				fmaf(rounded, angle[0], -(partner * angle[1])));
	}
	return bits;
}

/* The fixed order's epilogue of a warp (core_output_bits; a pair's partner is lane 4 (g ^ 1) + t), through the
 * warp's staging rows, 16-byte stores of its heads' rows. output: the block's first head of its row. */
__device__ static __forceinline__ void
core_finish(const float values[CORE_STEPS][ELEMENTS_PER_LANE],
	    const float *denominators, const float *rotation, uint32_t slice,
	    uint32_t lane, uint16_t *staged, uint16_t *output)
{
	const uint32_t lane_group = lane / LANES_PER_LANE_GROUP,
		       lane_place = lane % LANES_PER_LANE_GROUP;
#pragma unroll
	for (uint32_t step = 0; step < CORE_STEPS; step++) {
#pragma unroll
		for (uint32_t element = 0; element < ELEMENTS_PER_LANE;
		     element++) {
			const uint32_t head = lane_place * COLUMNS_PER_LANE +
					      element % COLUMNS_PER_LANE;
			const uint32_t dim =
				step * CORE_STEP_DIMS + lane_group +
				element / COLUMNS_PER_LANE * CORE_HALF_GROUP;
			staged[head * CORE_STAGED_HALVES + dim] =
				(uint16_t)core_output_bits<false>(
					values[step][element],
					denominators[head], rotation,
					slice * CORE_SLICE_DIMS + dim,
					LANES_PER_LANE_GROUP);
		}
	}
	__syncwarp();
	for (uint32_t chunk = lane; chunk < CORE_HEADS * CORE_SLICE_CHUNKS;
	     chunk += THREADS_PER_WARP) {
		const uint32_t head = chunk / CORE_SLICE_CHUNKS,
			       piece = chunk % CORE_SLICE_CHUNKS;
		*(uint4 *)(output + (size_t)head * MODEL_HEAD_DIMS +
			   slice * CORE_SLICE_DIMS +
			   piece * CORE_CHUNK_HALVES) =
			*(const uint4 *)(staged + head * CORE_STAGED_HALVES +
					 piece * CORE_CHUNK_HALVES);
	}
}

/* The fixed order's pass: the groups through the ring of 2 stages, the next group's copies issued and the index of
 * the one after read while this one is computed; each warp its slice's partial scores, then the softmax in the heads'
 * lanes, O and the sum rescaled where a head's maximum rose by more than the step, then P V. queries: head g's of
 * the row for lane 4 g + t, from dim 2 t. is_block: the block core's keys. */
template <bool is_block>
__device__ static __forceinline__ void
core_fixed_pass(const struct core_row *row, const uint16_t *queries,
		uint32_t first_group, uint32_t end_group, unsigned char *shared,
		float *maximum, float *total,
		float values[CORE_STEPS][ELEMENTS_PER_LANE])
{
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t lane_place = lane % LANES_PER_LANE_GROUP;
	const uint32_t softmax_head = threadIdx.x / CORE_SOFTMAX_LANES,
		       head_lane = threadIdx.x % CORE_SOFTMAX_LANES;
	const uint32_t copy_key = threadIdx.x / CORE_THREADS_PER_KEY;
	float *partials = (float *)(shared + CORE_PARTIALS_OFFSET);
	float *factors = (float *)(shared + CORE_FACTORS_OFFSET);
	uint16_t *probs = (uint16_t *)(shared + CORE_PROBS_OFFSET);
	int32_t *indices = (int32_t *)(shared + CORE_INDICES_OFFSET);
	/* operand B of each step: dims 2 t, 2 t + 1 and 2 t + 8, 2 t + 9 of the warp's slice */
	uint32_t slice_queries[CORE_STEPS][CORE_OPERAND_B_REGISTERS];
#pragma unroll
	for (uint32_t step = 0; step < CORE_STEPS; step++) {
		const uint16_t *const dims = queries + warp * CORE_SLICE_DIMS +
					     step * CORE_STEP_DIMS;
		slice_queries[step][0] = *(const uint32_t *)dims;
		slice_queries[step][1] =
			*(const uint32_t *)(dims + CORE_HALF_GROUP);
	}
	int32_t index_ahead = core_key_index<is_block>(
		row, first_group * CORE_GROUP_KEYS + copy_key);
	core_issue<is_block>(row, shared, indices, first_group, index_ahead);
	if (first_group + 1u < end_group) {
		index_ahead = core_key_index<is_block>(
			row, (first_group + 1u) * CORE_GROUP_KEYS + copy_key);
	}
	for (uint32_t group = first_group; group < end_group; group++) {
		const uint32_t stage = (group - first_group) % CORE_STAGES,
			       other_stage = stage ^ 1u;
		const unsigned char *stage_rows =
			shared + stage * CORE_STAGE_BYTES;
		core_wait_copies();
		__syncthreads(); /* this group has arrived; every warp is past the other stage */
		if (group + 1u < end_group) {
			core_issue<is_block>(
				row, shared + other_stage * CORE_STAGE_BYTES,
				indices + other_stage * CORE_GROUP_KEYS,
				group + 1u, index_ahead);
			if (group + CORE_INDICES_AHEAD < end_group) {
				index_ahead = core_key_index<is_block>(
					row, (group + CORE_INDICES_AHEAD) *
							     CORE_GROUP_KEYS +
						     copy_key);
			}
		}
		core_partials(stage_rows, warp, lane, slice_queries, partials);
		__syncthreads();
		bool rescaled = false;
		if (softmax_head < CORE_HEADS) {
			rescaled = core_softmax(
				partials, indices + stage * CORE_GROUP_KEYS,
				probs, factors, softmax_head, head_lane,
				maximum, total);
		}
		if (__syncthreads_or((int)rescaled) != 0) {
#pragma unroll
			for (uint32_t step = 0; step < CORE_STEPS; step++) {
#pragma unroll
				for (uint32_t element = 0;
				     element < ELEMENTS_PER_LANE; element++) {
					values[step][element] *= factors
						[lane_place * COLUMNS_PER_LANE +
						 element % COLUMNS_PER_LANE];
				}
			}
		}
		core_values(stage_rows, probs, warp, lane, values);
	}
}

/* What a block reads of its row (blockIdx.x): false for a row from row_count on, which leaves before any barrier.
 * first_group, end_group: the groups that hold a key, from the first window position not below 0 to the last
 * selected entry; of the block core (is_block: the row's block, its session's consecutive rows from the block's first
 * position b, no pool) those of keys 0 .. 128 + its rows - 1, positions b - 128 to the block's last. */
template <bool is_block = false>
__device__ static __forceinline__ bool
core_row_open(const struct attention_core_arguments_record arguments,
	      const unsigned char *arena, struct core_row *row,
	      uint32_t *first_group, uint32_t *end_group)
{
	const uint32_t row_index = blockIdx.x;
	const uint32_t row_count =
		((const struct step_block *)(arena + arguments.step_offset))
			->row_count;
	LINEAR_DEVICE_CHECK(row_count != 0 && row_count <= gridDim.x);
	if (row_index >= row_count) {
		return false;
	}
	const struct step_row_record *step_row =
		(const struct step_row_record *)(arena + arguments.step_offset +
						 STEP_BLOCK_BYTES_VALUE) +
		row_index;
	/* a full step's row whose output at this decoder layer nothing reads (step_records.h's cone) */
	if (arguments.cone_layer != 0 &&
	    STEP_ROW_CONE_SKIPS(step_row->flags,
				arguments.cone_layer -
					MODEL_DECODER_FIRST_LAYER)) {
		return false;
	}
	const uint32_t ratio = arguments.pool_ratio,
		       position = (uint32_t)step_row->position;
	const uint64_t session = step_row->session_slot;
	LINEAR_DEVICE_CHECK(step_row->position >= 0 &&
			    position < arguments.context_token_limit &&
			    session < arguments.session_capacity);
	row->ring = arena + arguments.ring_offset +
		    session * arguments.ring_session_stride;
	row->position = (int32_t)position;
	row->pool_entries = ratio == 0 ? 0u :
					 arguments.context_token_limit / ratio;
	row->pool = ratio == 0 ?
			    NULL :
			    arena + arguments.pool_offset +
				    session * arguments.pool_session_stride;
	row->selection =
		ratio == 0 ?
			NULL :
			(const int32_t *)(arena + arguments.selection_offset) +
				(size_t)row_index * MODEL_SELECTED;
	row->visible = ratio == 0 ? 0u : (position + 1u) / ratio;
	const uint32_t selected =
		row->visible < MODEL_SELECTED ? row->visible : MODEL_SELECTED;
	*first_group = (position + 1u >= MODEL_WINDOW ?
				0u :
				MODEL_WINDOW - 1u - position) /
		       CORE_GROUP_KEYS;
	*end_group = CORE_WINDOW_GROUPS +
		     (selected + CORE_GROUP_KEYS - 1u) / CORE_GROUP_KEYS;
	if (is_block) {
		/* the row's block: the run of the step's rows of its session at consecutive positions around it (a
		 * BATCHED draft's slab holds several verifiers' blocks; a thin draft's rows are all one block) */
		const struct step_row_record *const rows = step_row - row_index;
		uint32_t start = row_index, end = row_index;
		while (start != 0 && rows[start - 1u].session_slot == step_row->session_slot &&
		       rows[start - 1u].position + 1 == rows[start].position)
			start--;
		while (end + 1u < row_count && rows[end + 1u].session_slot == step_row->session_slot &&
		       rows[end + 1u].position == rows[end].position + 1)
			end++;
		const uint32_t first = (uint32_t)rows[start].position, block_rows = end - start + 1u;
		LINEAR_DEVICE_CHECK(ratio == 0 && position == first + (row_index - start));
		row->block_first = (int32_t)first;
		row->position = (int32_t)(first + block_rows - 1u);
		*first_group =
			(first >= MODEL_WINDOW ? 0u : MODEL_WINDOW - first) /
			CORE_GROUP_KEYS;
		*end_group =
			(MODEL_WINDOW + block_rows + CORE_GROUP_KEYS - 1u) /
			CORE_GROUP_KEYS;
	}
	return true;
}

/* The fixed order's core of a block (attention_core_records.h), or the block core's (is_block); core_shared: its
 * dynamic shared memory. */
template <bool is_block>
__device__ static __forceinline__ void
core_attention(const struct attention_core_arguments_record arguments,
	       unsigned char *arena, unsigned char *core_shared)
{
	struct core_row row;
	uint32_t first_group, end_group;
	if (!core_row_open<is_block>(arguments, arena, &row, &first_group,
				     &end_group)) {
		return;
	}
	LINEAR_DEVICE_CHECK(
		blockDim.x == CORE_THREADS &&
		arguments.first_head + gridDim.y * CORE_HEADS <=
			MODEL_HEADS_VALUE &&
		(arguments.output_row_heads == 0 ||
		 gridDim.y * CORE_HEADS <= arguments.output_row_heads));
	const uint32_t row_index = blockIdx.x;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t lane_group = lane / LANES_PER_LANE_GROUP,
		       lane_place = lane % LANES_PER_LANE_GROUP;
	const uint32_t first_head =
		arguments.first_head + blockIdx.y * CORE_HEADS;
	const uint32_t softmax_head = threadIdx.x / CORE_SOFTMAX_LANES,
		       head_lane = threadIdx.x % CORE_SOFTMAX_LANES;
	const uint16_t *head_queries =
		(const uint16_t *)(arena + arguments.queries_offset) +
		((size_t)row_index * MODEL_HEADS_VALUE + first_head +
		 lane_group) *
			MODEL_HEAD_DIMS +
		lane_place * COLUMNS_PER_LANE;
	float maximum = ATTENTION_CORE_SCORE_FLOOR, total = 0.0f;
	float values[CORE_STEPS][ELEMENTS_PER_LANE] = {};
	core_fixed_pass<is_block>(&row, head_queries, first_group, end_group,
				  core_shared, &maximum, &total, values);
	float *const factors = (float *)(core_shared + CORE_FACTORS_OFFSET);
	__syncthreads(); /* every warp is past its last P V */
	if (softmax_head < CORE_HEADS && head_lane == 0) {
		factors[softmax_head] =
			total +
			expf(((const float *)(arena + arguments.sinks_offset))
				     [first_head + softmax_head] -
			     maximum);
	}
	__syncthreads();
	core_finish(values, factors,
		    (const float *)(arena + arguments.rotation_offset) +
			    (size_t)row_index * CORE_ROTATION_ROW_FLOATS,
		    warp, lane,
		    (uint16_t *)core_shared +
			    warp * CORE_HEADS * CORE_STAGED_HALVES,
		    (uint16_t *)(arena + arguments.output_offset) +
			    core_output_head(arguments, row_index, first_head) *
				    MODEL_HEAD_DIMS);
}

#if !LINEAR_THREAD_SUMS /* class 80's arithmetic: kernel_attention_exact.cuh has class 120's */
/* A batch of the thread's asynchronous copies: those it began since its last batch. */
__device__ static __forceinline__ void core_commit_copies(void)
{
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

/* Every batch of the thread's but its last `pending` has arrived. */
template <uint32_t pending>
__device__ static __forceinline__ void core_wait_batches(void)
{
	asm volatile("cp.async.wait_group %0;\n" ::"n"((int)pending)
		     : "memory");
}

/* The rows of keys first_key .. end_key - 1 of the row into key_rows (NULL: no key), CORE_INDEX_ROUNDS of `threads`
 * threads, every read of the selection begun before the first is waited for. */
template <uint32_t threads>
__device__ static __forceinline__ void
core_key_rows(const struct core_row *row, uint32_t first_key, uint32_t end_key,
	      const unsigned char **key_rows)
{
	int32_t indices[(CORE_KEYS + threads - 1u) / threads];
#pragma unroll
	for (uint32_t round = 0; round < (CORE_KEYS + threads - 1u) / threads;
	     round++) {
		const uint32_t key = first_key + round * threads + threadIdx.x;
		indices[round] = key < end_key ? core_key_index(row, key) : -1;
	}
#pragma unroll
	for (uint32_t round = 0; round < (CORE_KEYS + threads - 1u) / threads;
	     round++) {
		const uint32_t key = first_key + round * threads + threadIdx.x;
		if (key < end_key) {
			key_rows[key] =
				indices[round] < 0 ?
					NULL :
					core_key_row(row, key, indices[round]);
		}
	}
}

/* The chains of slice `slice` of the warp's group for each of `tiles` tiles of 8 heads (core_scores's, each from
 * +0): operand A (keys by dims) from the stage once for every tile; operand B (dims by heads) of each tile and step
 * from the queries in shared memory, two steps a load. */
template <uint32_t tiles>
__device__ static __forceinline__ void
core_slice_scores(const unsigned char *stage, const unsigned char *queries,
		  uint32_t slice, uint32_t lane,
		  float chains[tiles][ELEMENTS_PER_LANE])
{
	const uint32_t matrix = lane / CORE_MATRIX_ROWS,
		       matrix_row = lane % CORE_MATRIX_ROWS;
	const uint32_t key =
		matrix_row + matrix % CORE_KEY_HALVES * CORE_HALF_GROUP;
	uint32_t operands[CORE_STEPS][REGISTERS_PER_OPERAND_A];
#pragma unroll
	for (uint32_t step = 0; step < CORE_STEPS; step++) {
		core_load_matrices(
			operands[step],
			stage + core_place(key,
					   (slice * CORE_SLICE_DIMS +
					    step * CORE_STEP_DIMS) /
							   CORE_CHUNK_HALVES +
						   matrix / CORE_KEY_HALVES));
	}
#pragma unroll
	for (uint32_t tile = 0; tile < tiles; tile++) {
		uint32_t tile_queries[CORE_STEPS][CORE_OPERAND_B_REGISTERS];
#pragma unroll
		for (uint32_t step = 0; step < CORE_STEPS;
		     step += CORE_KEY_HALVES) {
			uint32_t registers[REGISTERS_PER_OPERAND_A];
			core_load_matrices(
				registers,
				queries +
					(tile * CORE_HEADS + matrix_row) *
						CORE_QUERY_ROW_BYTES +
					(slice * CORE_SLICE_DIMS +
					 step * CORE_STEP_DIMS +
					 matrix * CORE_CHUNK_HALVES) *
						sizeof(uint16_t));
			tile_queries[step][0] = registers[0];
			tile_queries[step][1] = registers[1];
			tile_queries[step + 1u][0] =
				registers[CORE_OPERAND_B_REGISTERS];
			tile_queries[step + 1u][1] =
				registers[CORE_OPERAND_B_REGISTERS + 1u];
		}
#pragma unroll
		for (uint32_t step = 0; step < CORE_STEPS; step++) {
			linear_tensor_core<LINEAR_FORM_CODES>(
				chains[tile], operands[step],
				tile_queries[step][0], tile_queries[step][1]);
		}
	}
}

/* The reference order's scores of a block (attention_core_records.h): its row, `tiles` tiles of 8 heads and the
 * tile of groups z. The heads' queries come into shared memory (rows CORE_QUERY_ROW_BYTES apart: the 8 rows of a
 * matrix across the banks); warp w takes the tile's group w, if it holds a key, through a stage of its own (a copy
 * instruction the lines of 4 keys), read once for every head: each slice's chain from +0 (core_scores) added from +0
 * in slice order as core_group_scores adds the partials, into scores [row][head][key]. */
template <uint32_t tiles>
__device__ static __forceinline__ void
core_tile_scores(const struct attention_core_arguments_record arguments,
		 unsigned char *arena, unsigned char *shared)
{
	struct core_row row;
	uint32_t first_group, end_group;
	const uint32_t tile_group = blockIdx.z * CORE_TILE_GROUPS;
	if (!core_row_open(arguments, arena, &row, &first_group, &end_group) ||
	    tile_group >= end_group ||
	    tile_group + CORE_TILE_GROUPS <= first_group) {
		return; /* every thread: no key of the row in the tile */
	}
	LINEAR_DEVICE_CHECK(blockDim.x == CORE_THREADS &&
			    arguments.first_head +
					    gridDim.y * tiles * CORE_HEADS <=
				    MODEL_HEADS_VALUE &&
			    gridDim.z * ATTENTION_CORE_TILE_KEYS == CORE_KEYS);
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t lane_group = lane / LANES_PER_LANE_GROUP,
		       lane_place = lane % LANES_PER_LANE_GROUP;
	const uint32_t first_head =
		arguments.first_head + blockIdx.y * tiles * CORE_HEADS;
	unsigned char *const queries = shared + CORE_TILE_QUERIES_OFFSET;
	const unsigned char **const key_rows =
		(const unsigned char **)(shared + CORE_TILE_QUERIES_OFFSET +
					 tiles * CORE_HEADS *
						 CORE_QUERY_ROW_BYTES);
	const unsigned char *const block_queries =
		arena + arguments.queries_offset +
		((size_t)blockIdx.x * MODEL_HEADS_VALUE + first_head) *
			MODEL_HEAD_DIMS * sizeof(uint16_t);
	for (uint32_t chunk = threadIdx.x;
	     chunk < tiles * CORE_HEADS * CORE_ROW_CHUNKS;
	     chunk += CORE_THREADS) {
		const uint32_t head = chunk / CORE_ROW_CHUNKS,
			       piece = chunk % CORE_ROW_CHUNKS;
		core_copy_chunk(queries + head * CORE_QUERY_ROW_BYTES +
					piece * CORE_CHUNK_BYTES,
				block_queries + head * CORE_ROW_BYTES +
					piece * CORE_CHUNK_BYTES,
				CORE_CHUNK_BYTES);
	}
	const uint32_t first_key = tile_group * CORE_GROUP_KEYS;
	if (threadIdx.x < ATTENTION_CORE_TILE_KEYS) {
		const uint32_t key = first_key + threadIdx.x;
		const int32_t index =
			key >= first_group * CORE_GROUP_KEYS &&
					key < end_group * CORE_GROUP_KEYS ?
				core_key_index(&row, key) :
				-1;
		key_rows[threadIdx.x] =
			index < 0 ? NULL : core_key_row(&row, key, index);
	}
	__syncthreads(); /* the tile's rows */
	const uint32_t group = tile_group + warp;
	const bool has_keys = group >= first_group && group < end_group;
	unsigned char *const stage = shared + warp * CORE_STAGE_BYTES;
	if (has_keys) {
#pragma unroll
		for (uint32_t quarter = 0;
		     quarter < CORE_GROUP_KEYS / CORE_WARP_COPY_KEYS;
		     quarter++) {
			const uint32_t key = quarter * CORE_WARP_COPY_KEYS +
					     lane / CORE_THREADS_PER_KEY;
			core_copy_row(stage, key,
				      key_rows[warp * CORE_GROUP_KEYS + key],
				      row.ring, lane % CORE_THREADS_PER_KEY,
				      CORE_THREADS_PER_KEY);
		}
	}
	core_wait_copies();
	__syncthreads(); /* the queries, and each warp's group */
	if (!has_keys) {
		return; /* no barrier follows */
	}
	float sums[tiles][ELEMENTS_PER_LANE] = {};
#pragma unroll 1
	for (uint32_t slice = 0; slice < CORE_SLICES; slice++) {
		float chains[tiles][ELEMENTS_PER_LANE] = {};
		core_slice_scores<tiles>(stage, queries, slice, lane, chains);
#pragma unroll
		for (uint32_t tile = 0; tile < tiles; tile++) {
#pragma unroll
			for (uint32_t element = 0; element < ELEMENTS_PER_LANE;
			     element++) {
				sums[tile][element] += chains[tile][element];
			}
		}
	}
	float *const scores =
		(float *)(arena + arguments.scores_offset) +
		((size_t)blockIdx.x * MODEL_HEADS_VALUE + first_head) *
			CORE_KEYS +
		group * CORE_GROUP_KEYS;
#pragma unroll
	for (uint32_t tile = 0; tile < tiles; tile++) {
#pragma unroll
		for (uint32_t element = 0; element < ELEMENTS_PER_LANE;
		     element++) {
			const uint32_t head = tile * CORE_HEADS +
					      lane_place * COLUMNS_PER_LANE +
					      element % COLUMNS_PER_LANE;
			const uint32_t key =
				lane_group +
				element / COLUMNS_PER_LANE * CORE_HALF_GROUP;
			scores[head * CORE_KEYS + key] =
				key_rows[warp * CORE_GROUP_KEYS + key] == NULL ?
					-INFINITY :
					sums[tile][element] *
						ATTENTION_CORE_SCALE;
		}
	}
}

/* The reference order's softmax of the block's heads (8 * tiles) by every thread of the block, warp w heads w, w + 8,
 * .., from their scores in shared memory ([head][key]): m, a head's largest from the floor up; p = expf(s - m) of
 * each key in place of its score and its weight into all_probs ([group][head][key], each 8 heads as core_softmax
 * lays them); behind a barrier, each group's 16 p added as core_softmax adds them (key j with key j + 8, then the
 * eight sums 1, 2 and 4 apart: here in one thread) into group_sums. The caller adds the sums behind another barrier
 * (core_denominators). */
template <uint32_t tiles>
__device__ static __forceinline__ void
core_block_softmax(float *scores, uint16_t *all_probs, float *maxima,
		   float *group_sums, uint32_t first_group, uint32_t end_group)
{
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t first_key = first_group * CORE_GROUP_KEYS,
		       end_key = end_group * CORE_GROUP_KEYS;
#pragma unroll 1
	for (uint32_t tile = 0; tile < tiles; tile++) {
		const uint32_t head = tile * CORE_HEADS + warp;
		float *const head_scores = scores + head * CORE_KEYS;
		float largest = ATTENTION_CORE_SCORE_FLOOR;
		for (uint32_t key = first_key + lane * CORE_CHUNK_FLOATS;
		     key < end_key; key += CORE_SOFTMAX_STRIDE) {
			const float4 four =
				*(const float4 *)(head_scores + key);
			largest = fmaxf(fmaxf(largest, fmaxf(four.x, four.y)),
					fmaxf(four.z, four.w));
		}
		for (uint32_t distance = 1u; distance < THREADS_PER_WARP;
		     distance <<= 1u) {
			largest =
				fmaxf(largest,
				      __shfl_xor_sync(CORE_EVERY_LANE, largest,
						      (int)distance));
		}
		if (lane == 0) {
			maxima[head] = largest;
		}
		for (uint32_t key = first_key + lane * CORE_CHUNK_FLOATS;
		     key < end_key; key += CORE_SOFTMAX_STRIDE) {
			const float4 four =
				*(const float4 *)(head_scores + key);
			const float4 probabilities = make_float4(
				expf(four.x - largest), expf(four.y - largest),
				expf(four.z - largest), expf(four.w - largest));
			const uint32_t place =
				key %
				CORE_GROUP_KEYS; /* of the first of the four, all in one chunk */
			*(float4 *)(head_scores + key) = probabilities;
			*(uint2 *)(all_probs +
				   (key / CORE_GROUP_KEYS * tiles + tile) *
					   CORE_HEADS * CORE_GROUP_KEYS +
				   warp * CORE_GROUP_KEYS +
				   ((place / CORE_HALF_GROUP) ^
				    ((warp >> CORE_PROBS_SWIZZLE_SHIFT) & 1u)) *
					   CORE_CHUNK_HALVES +
				   place % CORE_HALF_GROUP) =
				make_uint2(core_weight<true>(probabilities.x) |
						   (uint32_t)core_weight<true>(
							   probabilities.y)
							   << BITS_PER_HALF,
					   core_weight<true>(probabilities.z) |
						   (uint32_t)core_weight<true>(
							   probabilities.w)
							   << BITS_PER_HALF);
		}
	}
	__syncthreads();
	const uint32_t group_count = end_group - first_group;
	for (uint32_t item = threadIdx.x;
	     item < tiles * CORE_HEADS * group_count;
	     item += CORE_VALUE_THREADS) {
		const uint32_t item_head = item / group_count,
			       group = first_group + item % group_count;
		const float *const probabilities = scores +
						   item_head * CORE_KEYS +
						   group * CORE_GROUP_KEYS;
		float lane_sums[CORE_SOFTMAX_LANES];
#pragma unroll
		for (uint32_t head_lane = 0; head_lane < CORE_SOFTMAX_LANES;
		     head_lane++) {
			lane_sums[head_lane] =
				(0.0f + probabilities[head_lane]) +
				probabilities[head_lane + CORE_HALF_GROUP];
		}
#pragma unroll
		for (uint32_t stride = 1; stride < CORE_SOFTMAX_LANES;
		     stride += stride) { /* ((0 + 1) + (2 + 3)) + .. */
#pragma unroll
			for (uint32_t first = 0; first < CORE_SOFTMAX_LANES;
			     first += stride + stride) {
				lane_sums[first] = lane_sums[first] +
						   lane_sums[first + stride];
			}
		}
		group_sums[item_head * CORE_GROUPS + group] = lane_sums[0];
	}
}

/* Head h's denominator by thread h (h below heads): its groups' sums added from +0 in group order, then
 * expf(sink - m). sinks: the block's first head's. */
__device__ static __forceinline__ void
core_denominators(const float *maxima, const float *group_sums,
		  float *denominators, const float *sinks, uint32_t heads,
		  uint32_t first_group, uint32_t end_group)
{
	if (threadIdx.x < heads) {
		float total = 0.0f;
		for (uint32_t group = first_group; group < end_group; group++) {
			total += group_sums[threadIdx.x * CORE_GROUPS + group];
		}
		denominators[threadIdx.x] =
			total + expf(sinks[threadIdx.x] - maxima[threadIdx.x]);
	}
}

/* The copy of group `group`'s keys by the warp's 16 dims from dim `dim` into a stage of its ring: lane l takes key
 * l / 2, chunk l % 2, placed at chunk (l % 2) ^ (key / 4 % 2): the 8 keys of a matrix across the banks; zeros for no
 * key, which reads nothing (stand_in, an arena address, stands for its source). */
__device__ static __forceinline__ void
core_value_copy(const unsigned char *const *key_rows,
		const unsigned char *stand_in, unsigned char *stage,
		uint32_t group, uint32_t dim, uint32_t lane)
{
	const uint32_t key = lane / CORE_KEY_HALVES,
		       chunk = lane % CORE_KEY_HALVES;
	const unsigned char *const key_row =
		key_rows[group * CORE_GROUP_KEYS + key];
	core_copy_chunk(stage + key * CORE_VALUE_KEY_BYTES +
				(chunk ^ (key / CORE_VALUE_SWIZZLE_KEYS %
					  CORE_VALUE_KEY_CHUNKS)) *
					CORE_CHUNK_BYTES,
			key_row != NULL ? key_row + dim * sizeof(uint16_t) +
						  chunk * CORE_CHUNK_BYTES :
					  stand_in,
			key_row != NULL ? CORE_CHUNK_BYTES : 0u);
}

/* The operands of O^T += V^T P^T of one group over the warp's 16 dims for each of `tiles` tiles of 8 heads: A (dims
 * by keys) from the warp's stage, transposed, once; B (keys by heads) of each tile from the group's weights, as
 * core_values loads it. */
template <uint32_t tiles>
__device__ static __forceinline__ void
core_dim_operands(const unsigned char *stage, const uint16_t *probs,
		  uint32_t lane,
		  uint32_t operands[REGISTERS_PER_OPERAND_A +
				    tiles * CORE_OPERAND_B_REGISTERS])
{
	const uint32_t matrix = lane / CORE_MATRIX_ROWS,
		       row_lane = lane % CORE_MATRIX_ROWS;
	const uint32_t key = row_lane +
			     matrix / CORE_KEY_HALVES * CORE_HALF_GROUP,
		       chunk = matrix % CORE_KEY_HALVES;
	core_load_matrices_transposed(
		operands, stage + key * CORE_VALUE_KEY_BYTES +
				  (chunk ^ (key / CORE_VALUE_SWIZZLE_KEYS %
					    CORE_VALUE_KEY_CHUNKS)) *
					  CORE_CHUNK_BYTES);
#pragma unroll
	for (uint32_t tile = 0; tile < tiles; tile++) {
		core_load_two_matrices(
			operands + REGISTERS_PER_OPERAND_A +
				tile * CORE_OPERAND_B_REGISTERS,
			probs +
				(tile * CORE_HEADS + row_lane) *
					CORE_GROUP_KEYS +
				(chunk ^
				 ((row_lane >> CORE_PROBS_SWIZZLE_SHIFT) &
				  1u)) * CORE_CHUNK_HALVES);
	}
}

/* The reference order's softmax and values of a block (attention_core_records.h): its row, `tiles` tiles of 8 heads
 * and the tile of dims z. The heads' scores of the row's groups from scores; the softmax (core_block_softmax); warp w's
 * 16 dims of each group's keys through its ring, the groups in order, one instruction a group and tile; the epilogue
 * (core_output_bits; a pair's partner is lane 4 (g ^ 1) + t) through the warp's staging rows, 16-byte stores. */
template <uint32_t tiles>
__device__ static __forceinline__ void
core_tile_values(const struct attention_core_arguments_record arguments,
		 unsigned char *arena, unsigned char *shared)
{
	struct core_row row;
	uint32_t first_group, end_group;
	if (!core_row_open(arguments, arena, &row, &first_group, &end_group)) {
		return;
	}
	LINEAR_DEVICE_CHECK(
		blockDim.x == CORE_VALUE_THREADS &&
		arguments.first_head + gridDim.y * tiles * CORE_HEADS <=
			MODEL_HEADS_VALUE &&
		gridDim.z * ATTENTION_CORE_TILE_DIMS == MODEL_HEAD_DIMS);
	const uint32_t heads = tiles * CORE_HEADS, row_index = blockIdx.x;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t lane_group = lane / LANES_PER_LANE_GROUP,
		       lane_place = lane % LANES_PER_LANE_GROUP;
	const uint32_t first_head = arguments.first_head + blockIdx.y * heads;
	const uint32_t warp_dim =
		blockIdx.z * ATTENTION_CORE_TILE_DIMS + warp * CORE_WARP_DIMS;
	float *const scores = (float *)
		shared; /* [head][key]: the scores, then p; then the epilogue's staging */
	uint16_t *const all_probs =
		(uint16_t *)(shared + tiles * CORE_VALUE_PROBS_OFFSET);
	const unsigned char **const key_rows =
		(const unsigned char **)(shared +
					 tiles * CORE_VALUE_ROWS_OFFSET);
	float *const denominators =
		(float *)(shared + tiles * CORE_VALUE_ROWS_OFFSET +
			  CORE_VALUE_ROWS_BYTES);
	float *const maxima = denominators + heads;
	float *const group_sums = maxima + heads;
	unsigned char *const ring =
		(unsigned char *)(group_sums + heads * CORE_GROUPS) +
		warp * CORE_VALUE_STAGES * CORE_VALUE_STAGE_BYTES;
	const uint32_t first_key = first_group * CORE_GROUP_KEYS;
	const uint32_t head_chunks =
		(end_group - first_group) * CORE_GROUP_KEYS / CORE_CHUNK_FLOATS;
	const float *const row_scores =
		(const float *)(arena + arguments.scores_offset) +
		((size_t)row_index * MODEL_HEADS_VALUE + first_head) *
			CORE_KEYS;
	for (uint32_t chunk = threadIdx.x; chunk < heads * head_chunks;
	     chunk += CORE_VALUE_THREADS) {
		const uint32_t place = chunk / head_chunks * CORE_KEYS +
				       first_key +
				       chunk % head_chunks * CORE_CHUNK_FLOATS;
		core_copy_chunk(scores + place, row_scores + place,
				CORE_CHUNK_BYTES);
	}
	core_commit_copies();
	core_key_rows<CORE_VALUE_THREADS>(
		&row, first_key, end_group * CORE_GROUP_KEYS, key_rows);
	__syncthreads(); /* the row's keys */
#pragma unroll
	for (uint32_t ahead = 0; ahead < CORE_VALUE_STAGES;
	     ahead++) { /* a batch each, none past the last */
		if (first_group + ahead < end_group) {
			core_value_copy(key_rows, row.ring,
					ring + ahead * CORE_VALUE_STAGE_BYTES,
					first_group + ahead, warp_dim, lane);
		}
		core_commit_copies();
	}
	core_wait_batches<
		CORE_VALUE_STAGES>(); /* the scores; the groups' batches behind them may still fly */
	__syncthreads();
	core_block_softmax<tiles>(scores, all_probs, maxima, group_sums,
				  first_group, end_group);
	__syncthreads(); /* the weights and the groups' sums; p is read no more */
	core_denominators(maxima, group_sums, denominators,
			  (const float *)(arena + arguments.sinks_offset) +
				  first_head,
			  heads, first_group,
			  end_group); /* read behind the barrier after P V */
	float values[tiles][ELEMENTS_PER_LANE] = {};
	constexpr uint32_t operand_words =
		REGISTERS_PER_OPERAND_A + tiles * CORE_OPERAND_B_REGISTERS;
	uint32_t operands[operand_words]; /* the group's, loaded a group ahead */
	core_wait_batches<CORE_VALUE_STAGES - 1u>();
	__syncwarp(); /* every lane's copies of the first group */
	core_dim_operands<tiles>(
		ring, all_probs + first_group * heads * CORE_GROUP_KEYS, lane,
		operands);
	for (uint32_t group = first_group; group < end_group; group++) {
		unsigned char *const stage =
			ring + (group - first_group) % CORE_VALUE_STAGES *
				       CORE_VALUE_STAGE_BYTES;
		uint32_t next[operand_words];
		if (group + 1u < end_group) {
			core_wait_batches<CORE_VALUE_STAGES -
					  CORE_VALUE_AHEAD>();
			__syncwarp(); /* every lane's copies of the next group; every lane past this one */
			core_dim_operands<tiles>(
				ring + (group + 1u - first_group) %
						CORE_VALUE_STAGES *
						CORE_VALUE_STAGE_BYTES,
				all_probs +
					(group + 1u) * heads * CORE_GROUP_KEYS,
				lane, next);
		}
#pragma unroll
		for (uint32_t tile = 0; tile < tiles; tile++) {
			linear_tensor_core<LINEAR_FORM_CODES>(
				values[tile], operands,
				operands[REGISTERS_PER_OPERAND_A +
					 tile * CORE_OPERAND_B_REGISTERS],
				operands[REGISTERS_PER_OPERAND_A +
					 tile * CORE_OPERAND_B_REGISTERS + 1u]);
		}
		if (group + CORE_VALUE_STAGES <
		    end_group) { /* the stage's operands are in registers */
			core_value_copy(key_rows, row.ring, stage,
					group + CORE_VALUE_STAGES, warp_dim,
					lane);
		}
		core_commit_copies();
#pragma unroll
		for (uint32_t word = 0; word < operand_words; word++) {
			operands[word] = next[word];
		}
	}
	__syncthreads(); /* the denominators */
	const float *const rotation =
		(const float *)(arena + arguments.rotation_offset) +
		(size_t)row_index * CORE_ROTATION_ROW_FLOATS;
	uint16_t *const staged =
		(uint16_t *)shared + warp * heads * CORE_WARP_DIMS;
#pragma unroll
	for (uint32_t tile = 0; tile < tiles; tile++) {
#pragma unroll
		for (uint32_t element = 0; element < ELEMENTS_PER_LANE;
		     element++) {
			const uint32_t head = tile * CORE_HEADS +
					      lane_place * COLUMNS_PER_LANE +
					      element % COLUMNS_PER_LANE;
			const uint32_t dim =
				lane_group +
				element / COLUMNS_PER_LANE * CORE_HALF_GROUP;
			staged[head * CORE_WARP_DIMS + dim] =
				(uint16_t)core_output_bits<true>(
					values[tile][element],
					denominators[head], rotation,
					warp_dim + dim, LANES_PER_LANE_GROUP);
		}
	}
	__syncwarp();
	for (uint32_t piece = lane; piece < heads * CORE_KEY_HALVES;
	     piece += THREADS_PER_WARP) { /* 2 a head */
		const uint32_t head = piece / CORE_KEY_HALVES,
			       half = piece % CORE_KEY_HALVES;
		*(uint4 *)((uint16_t *)(arena + arguments.output_offset) +
			   core_output_head(arguments, row_index,
					    first_head + head) *
				   MODEL_HEAD_DIMS +
			   warp_dim + half * CORE_CHUNK_HALVES) =
			*(const uint4 *)(staged + head * CORE_WARP_DIMS +
					 half * CORE_CHUNK_HALVES);
	}
}
#endif

#define KERNEL_ATTENTION_CORE_HELPERS /* the helpers above, which the exact profile on class 120 shares */
#include "kernel_attention_exact.cuh"

/* grid (rows of the program, head_count / 8), block of 128 threads, ATTENTION_CORE_SHARED_BYTES_VALUE. */
extern "C" __global__ void
attention_core_entry(struct attention_core_arguments_record arguments,
		     unsigned char *arena)
{
	extern __shared__ __align__(
		CORE_CHUNK_BYTES) unsigned char core_shared[];
	core_attention<false>(arguments, arena, core_shared);
}

/* The draft's block window (attention_core_records.h: the block core): the fixed order's launch. */
extern "C" __global__ void
attention_core_block_entry(struct attention_core_arguments_record arguments,
			   unsigned char *arena)
{
	extern __shared__ __align__(
		CORE_CHUNK_BYTES) unsigned char core_shared[];
	core_attention<true>(arguments, arena, core_shared);
}

/* The reference order's scores, then its softmax and values (attention_core_records.h: the launches); on class 120
 * in the exact profile kernel_attention_exact.cuh's. */
extern "C" __global__ void
attention_core_scores_entry(struct attention_core_arguments_record arguments,
			    unsigned char *arena)
{
	extern __shared__ __align__(
		CORE_CHUNK_BYTES) unsigned char core_shared[];
#if LINEAR_THREAD_SUMS
	exact_tile_scores(arguments, arena, core_shared);
#else
	LINEAR_DEVICE_CHECK(
		arguments.heads_per_block == CORE_HEADS ||
		arguments.heads_per_block == CORE_TWO_TILES * CORE_HEADS ||
		arguments.heads_per_block == CORE_FOUR_TILES * CORE_HEADS ||
		arguments.heads_per_block ==
			ATTENTION_CORE_SCORE_TILES_MOST * CORE_HEADS);
	switch (arguments.heads_per_block / CORE_HEADS) {
	case 1u:
		core_tile_scores<1u>(arguments, arena, core_shared);
		break;
	case CORE_TWO_TILES:
		core_tile_scores<CORE_TWO_TILES>(arguments, arena, core_shared);
		break;
	case CORE_FOUR_TILES:
		core_tile_scores<CORE_FOUR_TILES>(arguments, arena,
						  core_shared);
		break;
	default:
		core_tile_scores<ATTENTION_CORE_SCORE_TILES_MOST>(
			arguments, arena, core_shared);
		break;
	}
#endif
}

extern "C" __global__ void
attention_core_values_entry(struct attention_core_arguments_record arguments,
			    unsigned char *arena)
{
	extern __shared__ __align__(
		CORE_CHUNK_BYTES) unsigned char core_shared[];
#if LINEAR_THREAD_SUMS
	exact_tile_values(arguments, arena, core_shared);
#else
	LINEAR_DEVICE_CHECK(arguments.heads_per_block == CORE_HEADS ||
			    arguments.heads_per_block ==
				    ATTENTION_CORE_VALUE_TILES_MOST *
					    CORE_HEADS);
	if (arguments.heads_per_block == CORE_HEADS) {
		core_tile_values<1u>(arguments, arena, core_shared);
	} else {
		core_tile_values<ATTENTION_CORE_VALUE_TILES_MOST>(
			arguments, arena, core_shared);
	}
#endif
}
