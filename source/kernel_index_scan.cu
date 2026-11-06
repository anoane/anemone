/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_index_scan.cu: the index scan and the exact top 512 after it, one row a block: the HALF form (FP16 keys and
 * queries) and the one-block selection of unit 11a's probe (tools/measure/index_scan_kernels.cuh). Device code only,
 * built once per architecture into the module "kernel_index_scan".
 *
 * Contract: include/anemone/index_scan_records.h; model_layout.h (the lane orders of queries and key tiles);
 * kernel_linear_common.cuh (the instruction, a bfloat16's value, the device checks). Here:
 *   - Scan: a block is a row and a segment of tiles, warp w its tiles w, w + 4, ..; the row's queries in shared
 *     memory. A row's ceil(e / 8) tiles spread over all the grid's segments in whole turns of 4 (11a: a short row's
 *     time is its blocks). Lane 4 g + t loads its 64 bytes of key g, then one instruction per step and head tile
 *     (A: 16 heads by 16 dims of the query; B: the 8 keys), and holds heads g and g + 8 of both tiles for keys 2 t
 *     and 2 t + 1. The butterfly over lane groups (xor 4, 8, 16) gives every lane the records' tree, addition being
 *     commutative; lanes 0 .. 3 store. Batch i11a: within 2e-3 of the row's rms of its host reference, the same bits
 *     at every grid, row group and segment, on both architectures.
 *   - Selection: above 512 entries two histograms of 256 bins find the cut, the high byte of the order key where the
 *     need-th from the top falls, then its low byte; ties_needed of the keys equal to that threshold are taken, the
 *     first in position order. Each warp counts the keys above and at it over its run of whole turns of 256 entries,
 *     a prefix over the warps places its outputs, and it writes its chosen positions in order. A histogram joins
 *     the lanes of a warp that hit one bin with the match instruction; the lowest adds their count (no atomics).
 *     The candidates are the same selection over blocks of 8 scores, a block's key the largest of its 8 order keys.
 *   - Every thread of a block reaches each of its barriers: a block leaves early only as a whole.
 */
#include "kernel_linear_common.cuh"

#include "anemone/index_scan_records.h"
#include "anemone/model_layout.h"
#include "anemone/step_records.h"

#define SCAN_WARPS (INDEX_SCAN_THREADS / THREADS_PER_WARP)
#define SELECT_WARPS (INDEX_SELECT_THREADS_VALUE / THREADS_PER_WARP)
#define LOAD_BYTES 16u
#define QUERY_LOADS (MODEL_INDEX_QUERIES_ROW_BYTES / LOAD_BYTES)
#define HEAD_TILES 2u                           /* of 16 heads */
#define STEPS 8u                                /* of 16 dims */
#define KEY_LOADS 4u                            /* of a lane's 64 bytes of a key: two steps each */
#define RUN_BYTES (KEY_LOADS * LOAD_BYTES)
#define HIGH_HEAD (HIGH_OUTPUT_EVEN_COLUMN - LOW_OUTPUT_EVEN_COLUMN)   /* head g + 8 beside head g in a result */
#define KEYS_PER_TILE MODEL_KEYS_PER_TILE
#define SELECTED MODEL_SELECTED
#define BINS 256u                               /* of a byte of an order key */
#define BINS_PER_LANE (BINS / THREADS_PER_WARP)
#define KEYS_PER_LOAD 8u                        /* FP16 scores in 16 bytes */
#define HALVES_PER_WORD 2u
#define TURN (THREADS_PER_WARP * KEYS_PER_LOAD)
#define ORDER_SIGN_SHIFT 15u
#define ORDER_LOW_BITS 0x00010001u              /* bit 0 of each half */
#define ORDER_FLIP_NEGATIVE 0x7FFFu             /* with the sign set every bit flips, else the sign alone */
#define HALF_MASK 0xFFFFu
#define EVERY_LANE 0xFFFFFFFFu
#define INDEX_PINNED_KEY HALF_MASK                /* the newest block's: above every finite or infinite score's */
#define INDEX_NOTHING_WRITTEN \
	0xFFFFFFFFu /* -1 of a selection, MODEL_NO_CANDIDATE of candidates */
static_assert(INDEX_NOTHING_WRITTEN == MODEL_NO_CANDIDATE &&
		      (int32_t)INDEX_NOTHING_WRITTEN == INDEX_NO_POSITION,
	      "one fill for both outputs");

enum index_lane_head { FIRST_TILE_LOW, FIRST_TILE_HIGH, SECOND_TILE_LOW, SECOND_TILE_HIGH, LANE_HEADS };
enum index_cut_field { CUT_DIGIT, CUT_ABOVE, CUT_FIELDS };

/* The block's row in the step area (row_first rows in: the record's), or NULL at or beyond the step's row_count. The
 * checked modules trap on a position step_records.h forbids (below 0, or from limit, the context limit, on), which also
 * holds e to a session's entries. */
__device__ static __forceinline__ const struct step_row_record *
index_step_row(const unsigned char *arena, uint64_t step_offset,
	       uint32_t row_first, uint32_t limit)
{
	const struct step_block *const step =
		(const struct step_block *)(arena + step_offset);
	const uint32_t index = row_first + blockIdx.x;
	const struct step_row_record *const own =
		(const struct step_row_record *)(step + 1) + index;
	LINEAR_DEVICE_CHECK(
		index >= step->row_count ||
		(own->position >= 0 && (uint32_t)own->position < limit));
	return index < step->row_count ? own : NULL;
}

/* relu (+0 for a dot not above 0) times the head's weight. */
__device__ static __forceinline__ float index_weighted(float dot, float weight)
{
    return (dot > 0.0f ? dot : 0.0f) * weight;
}

/* The lane's (r w + r w) + (r w + r w) for key 2 t + odd, then the butterfly and the one rounding to FP16. */
__device__ static __forceinline__ uint32_t index_score_bits(const float dots[HEAD_TILES][ELEMENTS_PER_LANE],
                                                            const float weights[LANE_HEADS], uint32_t odd)
{
    const uint32_t low = LOW_OUTPUT_EVEN_COLUMN + odd;
    float sum = (index_weighted(dots[0][low], weights[FIRST_TILE_LOW]) +
                 index_weighted(dots[0][low + HIGH_HEAD], weights[FIRST_TILE_HIGH])) +
                (index_weighted(dots[1][low], weights[SECOND_TILE_LOW]) +
                 index_weighted(dots[1][low + HIGH_HEAD], weights[SECOND_TILE_HIGH]));
#pragma unroll
    for (uint32_t distance = LANES_PER_LANE_GROUP; distance < THREADS_PER_WARP; distance <<= 1u) {
        sum += __shfl_xor_sync(EVERY_LANE, sum, (int)distance);
    }
    return (uint32_t)__half_as_ushort(__float2half_rn(sum));
}

/* grid (rows, segments), 128 threads, 8,192 bytes of dynamic shared memory: the row's queries. */
extern "C" __global__ void
index_scan_entry(struct index_scan_arguments_record arguments,
		 unsigned char *arena)
{
    extern __shared__ uint4 queries[];
    const uint32_t warp = threadIdx.x / THREADS_PER_WARP, lane = threadIdx.x % THREADS_PER_WARP;
    const uint32_t lane_group = lane / LANES_PER_LANE_GROUP, lane_place = lane % LANES_PER_LANE_GROUP;
    const struct step_row_record *const row =
	    index_step_row(arena, arguments.step_offset, arguments.row_first,
			   arguments.context_token_limit);
    LINEAR_DEVICE_CHECK(
	    blockDim.x == INDEX_SCAN_THREADS &&
	    (row == NULL || row->session_slot < arguments.session_capacity));
    if (row == NULL) {
        return;
    }
    const uint32_t extent = ((uint32_t)row->position + 1u) / arguments.ratio;
    const uint32_t blocks = extent / KEYS_PER_TILE + (extent % KEYS_PER_TILE != 0 ? 1u : 0u);
    const uint32_t *const kept =
	    arguments.candidates_offset == INDEX_NO_CANDIDATES ?
		    NULL :
		    (const uint32_t *)(arena + arguments.candidates_offset) +
			    (size_t)blockIdx.x * MODEL_CANDIDATE_BLOCKS;
    const uint32_t tile_count = kept != NULL && blocks >
							MODEL_CANDIDATE_BLOCKS ?
					MODEL_CANDIDATE_BLOCKS :
					blocks;
    const uint32_t row_warps = gridDim.y * SCAN_WARPS;   /* every segment takes a share of a row, a short one too */
    const uint32_t segment = (tile_count + row_warps - 1u) / row_warps * SCAN_WARPS;
    const uint32_t first_tile = blockIdx.y * segment;
    if (first_tile >= tile_count) {
        return;
    }
    const uint32_t end_tile = tile_count - first_tile < segment ? tile_count : first_tile + segment;
    const uint4 *const source = (const uint4 *)(arena + arguments.queries_offset) + (size_t)blockIdx.x * QUERY_LOADS;
    const uint16_t *const weight_bits =
	    (const uint16_t *)(arena + arguments.weights_offset) +
	    (size_t)blockIdx.x * MODEL_INDEX_HEADS + lane_group;
    const float weights[LANE_HEADS] = {
        linear_bfloat16_value(weight_bits[0]), linear_bfloat16_value(weight_bits[OUTPUTS_PER_HALF_TILE]),
        linear_bfloat16_value(weight_bits[OUTPUTS_PER_TILE]),
        linear_bfloat16_value(weight_bits[OUTPUTS_PER_TILE + OUTPUTS_PER_HALF_TILE])};
    const unsigned char *const keys =
	    arena + arguments.keys_offset +
	    (uint64_t)row->session_slot * arguments.keys_session_stride +
	    lane_group * MODEL_KEY_BYTES + lane_place * RUN_BYTES;
    uint32_t *const score_pairs =
	    (uint32_t *)(arena + arguments.scores_offset +
			 (size_t)blockIdx.x * arguments.entry_capacity *
				 INDEX_SCORE_BYTES);
    for (uint32_t load = threadIdx.x; load < QUERY_LOADS; load += blockDim.x) {
        queries[load] = source[load];
    }
    __syncthreads();
    for (uint32_t tile = first_tile + warp; tile < end_tile; tile += SCAN_WARPS) {
        const uint32_t key_tile = kept != NULL ? kept[tile] : tile;   /* with candidates, the k-th kept block */
        LINEAR_DEVICE_CHECK(key_tile < blocks);
	const uint4 *const run =
		(const uint4 *)(keys + (size_t)key_tile * MODEL_TILE_BYTES);
	uint4 key_loads[KEY_LOADS];
        float dots[HEAD_TILES][ELEMENTS_PER_LANE] = {{0.0f}};
#pragma unroll
        for (uint32_t load = 0; load < KEY_LOADS; load++) {
            key_loads[load] = run[load];
        }
#pragma unroll
        for (uint32_t step = 0; step < STEPS; step++) {
            const uint4 pair = key_loads[step * KEY_LOADS / STEPS];
            const bool is_odd = (step & 1u) != 0;
#pragma unroll
            for (uint32_t head_tile = 0; head_tile < HEAD_TILES; head_tile++) {
                const uint4 query = queries[(head_tile * STEPS + step) * THREADS_PER_WARP + lane];
                const uint32_t operand[REGISTERS_PER_OPERAND_A] = {query.x, query.y, query.z, query.w};
                linear_tensor_core<LINEAR_FORM_CODES>(dots[head_tile], operand, is_odd ? pair.z : pair.x,
                                                      is_odd ? pair.w : pair.y);   /* the 16-bit-float form */
            }
        }
        const uint32_t entry = key_tile * KEYS_PER_TILE + lane_place * COLUMNS_PER_LANE;
        const uint32_t place = tile * KEYS_PER_TILE + lane_place * COLUMNS_PER_LANE;   /* entry, without candidates */
        const uint32_t even = index_score_bits(dots, weights, 0), odd = index_score_bits(dots, weights, 1);
        if (lane_group == 0) {
		score_pairs[place / HALVES_PER_WORD] =
			(entry < extent ? even :
					  INDEX_NEGATIVE_INFINITY_VALUE) |
			(entry + 1u < extent ? odd :
					       INDEX_NEGATIVE_INFINITY_VALUE)
				<< BITS_PER_HALF;
	}
    }
}

/* ---- the selection ---- */

/* Scores base .. base + 7 of a row (base a multiple of 8) as order keys; bit k of the answer: base + k < count. */
__device__ static __forceinline__ uint32_t index_order_keys(const uint16_t *scores, uint32_t count, uint32_t base,
                                                           uint32_t keys[KEYS_PER_LOAD])
{
    const uint4 load = base < count ? *(const uint4 *)(scores + base) : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t pairs[KEYS_PER_LOAD / HALVES_PER_WORD] = {load.x, load.y, load.z, load.w};
#pragma unroll
    for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
        const uint32_t pair = pairs[key / HALVES_PER_WORD];
        const uint32_t ordered = pair ^ (((pair >> ORDER_SIGN_SHIFT) & ORDER_LOW_BITS) * ORDER_FLIP_NEGATIVE |
                                         HALF_SIGN_MASK);
        keys[key] = (ordered >> (key % HALVES_PER_WORD * BITS_PER_HALF)) & HALF_MASK;
    }
    const uint32_t held = base >= count ? 0u : count - base < KEYS_PER_LOAD ? count - base : KEYS_PER_LOAD;
    return (1u << held) - 1u;
}

/* The keys of blocks base .. base + 7 of 8 scores each: a block's the largest of its order keys (an order key keeps
 * the order), the pinned block's above every score's; bit k of the answer: base + k < count. */
__device__ static __forceinline__ uint32_t index_block_keys(const uint16_t *scores, uint32_t count, uint32_t base,
                                                           uint32_t pinned, uint32_t keys[KEYS_PER_LOAD])
{
#pragma unroll
    for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
        uint32_t scores_of_block[KEYS_PER_LOAD], largest = 0;
        const uint32_t block = base + key;
        (void)index_order_keys(scores, block < count ? (block + 1u) * KEYS_PER_TILE : 0u, block * KEYS_PER_TILE,
                               scores_of_block);
#pragma unroll
        for (uint32_t entry = 0; entry < KEYS_PER_TILE; entry++) {
            largest = scores_of_block[entry] > largest ? scores_of_block[entry] : largest;
        }
        keys[key] = block == pinned ? INDEX_PINNED_KEY : largest;
    }
    const uint32_t held = base >= count ? 0u : count - base < KEYS_PER_LOAD ? count - base : KEYS_PER_LOAD;
    return (1u << held) - 1u;
}

/* The keys of elements base .. base + 7: blocks with is_block, else scores. */
template <bool is_block>
__device__ static __forceinline__ uint32_t index_element_keys(const uint16_t *scores, uint32_t count, uint32_t base,
                                                             uint32_t pinned, uint32_t keys[KEYS_PER_LOAD])
{
    return is_block ? index_block_keys(scores, count, base, pinned, keys) : index_order_keys(scores, count, base, keys);
}

/* An exclusive prefix over the lanes of a warp, *total of all of them; every lane calls it. */
__device__ static __forceinline__ uint32_t index_lanes_before(uint32_t value, uint32_t lane, uint32_t *total)
{
    uint32_t sum = value;
#pragma unroll
    for (uint32_t distance = 1; distance < THREADS_PER_WARP; distance <<= 1u) {
        const uint32_t below = __shfl_up_sync(EVERY_LANE, sum, distance);
        sum += lane >= distance ? below : 0u;
    }
    *total = __shfl_sync(EVERY_LANE, sum, (int)(THREADS_PER_WARP - 1u));
    return sum - value;
}

/* The shared memory of a selection's block. */
struct index_select_shared {
    uint32_t warp_bins[SELECT_WARPS][BINS], bins[BINS], cut[CUT_FIELDS], warp_greater[SELECT_WARPS];
    uint32_t warp_ties[SELECT_WARPS], outputs_before[SELECT_WARPS], ties_before[SELECT_WARPS];
};

/* bins[d]: the elements whose order key has byte d at shift (8: the high byte; 0: the low byte, of the keys whose
 * high byte is high). Every thread of the block calls it. */
template <bool is_block>
__device__ static __forceinline__ void index_histogram(const uint16_t *scores, uint32_t count, uint32_t pinned,
                                                       uint32_t shift, uint32_t high, struct index_select_shared *own)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP, warp = threadIdx.x / THREADS_PER_WARP;
    for (uint32_t bin = lane; bin < BINS; bin += THREADS_PER_WARP) {
        own->warp_bins[warp][bin] = 0;
    }
    __syncwarp();
    for (uint32_t first = 0; first < count; first += blockDim.x * KEYS_PER_LOAD) {     /* the same turns everywhere */
        uint32_t keys[KEYS_PER_LOAD];
        const uint32_t present =
            index_element_keys<is_block>(scores, count, first + threadIdx.x * KEYS_PER_LOAD, pinned, keys);
#pragma unroll
        for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
            const uint32_t digit = (keys[key] >> shift) & BYTE_MASK;
            const bool wanted = ((present >> key) & 1u) != 0 && (shift != 0 || keys[key] >> BITS_PER_BYTE == high);
            const uint32_t peers = __match_any_sync(EVERY_LANE, wanted ? digit : BINS);
            if (wanted && (peers & ((1u << lane) - 1u)) == 0) {
                own->warp_bins[warp][digit] += (uint32_t)__popc(peers);
            }
            __syncwarp();
        }
    }
    __syncthreads();
    for (uint32_t bin = threadIdx.x; bin < BINS; bin += blockDim.x) {
        uint32_t total = 0;
        for (uint32_t other = 0; other < SELECT_WARPS; other++) {
            total += own->warp_bins[other][bin];
        }
        own->bins[bin] = total;
    }
    __syncthreads();
}

/* Warp 0: the bin where `need` elements counted from the top are reached (the bins hold at least need), and the
 * count above it; the lane whose bins reach it writes them. */
__device__ static __forceinline__ void index_cut(const uint32_t *bins, uint32_t need, uint32_t cut[CUT_FIELDS])
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
    uint32_t lane_sum = 0, total = 0;
    for (uint32_t bin = 0; bin < BINS_PER_LANE; bin++) {
        lane_sum += bins[BINS - 1u - lane * BINS_PER_LANE - bin];
    }
    uint32_t above = index_lanes_before(lane_sum, lane, &total);
    for (uint32_t bin = 0; above < need && need <= above + lane_sum && bin < BINS_PER_LANE; bin++) {
        const uint32_t digit = BINS - 1u - lane * BINS_PER_LANE - bin;
        if (above + bins[digit] >= need) {
            cut[CUT_DIGIT] = digit;
            cut[CUT_ABOVE] = above;
            break;
        }
        above += bins[digit];
    }
}

/* The top `need` of `count` elements by key descending, then number ascending, written ascending as `capacity`
 * words: each element's number, or through kept (with candidates) the entry it stands for; then the fill, all ones
 * (-1, MODEL_NO_CANDIDATE). Above capacity elements two histograms find the cut, else every one is taken.
 * Every thread of the block calls it. */
template <bool is_block>
__device__ static __forceinline__ void index_top_written(const uint16_t *scores, uint32_t count, uint32_t need,
                                                         uint32_t pinned, const uint32_t *kept, uint32_t capacity,
                                                         uint32_t *written, struct index_select_shared *own)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP, warp = threadIdx.x / THREADS_PER_WARP;
    uint32_t threshold = 0, ties_needed = capacity;     /* capacity or fewer elements: every one of them */
    if (count > capacity) {
        index_histogram<is_block>(scores, count, pinned, BITS_PER_BYTE, 0, own);
        if (warp == 0) {
            index_cut(own->bins, need, own->cut);
        }
        __syncthreads();
        const uint32_t high = own->cut[CUT_DIGIT], above_high = own->cut[CUT_ABOVE];
        index_histogram<is_block>(scores, count, pinned, 0, high, own);   /* its barriers pass before cut changes */
        if (warp == 0) {
            index_cut(own->bins, need - above_high, own->cut);
        }
        __syncthreads();
        threshold = high << BITS_PER_BYTE | own->cut[CUT_DIGIT];
        ties_needed = need - above_high - own->cut[CUT_ABOVE];
    }
    const uint32_t run = (count / SELECT_WARPS / TURN + 1u) * TURN, first = warp * run;
    uint32_t greater = 0, ties = 0;
    for (uint32_t base = first + lane * KEYS_PER_LOAD; base < first + run; base += TURN) {
        uint32_t keys[KEYS_PER_LOAD];
        const uint32_t present = index_element_keys<is_block>(scores, count, base, pinned, keys);
#pragma unroll
        for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
            greater += ((present >> key) & 1u) != 0 && keys[key] > threshold ? 1u : 0u;
            ties += ((present >> key) & 1u) != 0 && keys[key] == threshold ? 1u : 0u;
        }
    }
    greater = __reduce_add_sync(EVERY_LANE, greater);
    ties = __reduce_add_sync(EVERY_LANE, ties);
    if (lane == 0) {
        own->warp_greater[warp] = greater;
        own->warp_ties[warp] = ties;
    }
    __syncthreads();
    if (warp == 0) {                                    /* a warp takes the first of the ties still needed */
        uint32_t total = 0;
        const uint32_t before = index_lanes_before(own->warp_ties[lane], lane, &total);
        const uint32_t left = ties_needed > before ? ties_needed - before : 0u;
        own->ties_before[lane] = before;
        own->outputs_before[lane] = index_lanes_before(
            own->warp_greater[lane] + (left < own->warp_ties[lane] ? left : own->warp_ties[lane]), lane, &total);
    }
    __syncthreads();
    uint32_t outputs_done = own->outputs_before[warp], ties_done = own->ties_before[warp];
    for (uint32_t base = first + lane * KEYS_PER_LOAD; base < first + run; base += TURN) {
        uint32_t keys[KEYS_PER_LOAD], lane_ties = 0, chosen = 0, total = 0;
        const uint32_t present = index_element_keys<is_block>(scores, count, base, pinned, keys);
#pragma unroll
        for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
            lane_ties += ((present >> key) & 1u) != 0 && keys[key] == threshold ? 1u : 0u;
        }
        uint32_t tie_rank = ties_done + index_lanes_before(lane_ties, lane, &total);
        ties_done += total;
#pragma unroll
        for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
            const bool there = ((present >> key) & 1u) != 0, tie = there && keys[key] == threshold;
            chosen |= (there && keys[key] > threshold) || (tie && tie_rank < ties_needed) ? 1u << key : 0u;
            tie_rank += tie ? 1u : 0u;
        }
        uint32_t output = outputs_done + index_lanes_before((uint32_t)__popc(chosen), lane, &total);
        outputs_done += total;
        for (uint32_t key = 0; chosen != 0 && key < KEYS_PER_LOAD; key++) {
            if (((chosen >> key) & 1u) != 0) {
                const uint32_t element = base + key;
                written[output++] = kept != NULL ? kept[element / KEYS_PER_TILE] * KEYS_PER_TILE +
                                                       element % KEYS_PER_TILE
                                                 : element;
            }
        }
    }
    for (uint32_t output = need + threadIdx.x; output < capacity; output += blockDim.x) {
        written[output] = INDEX_NOTHING_WRITTEN;
    }
}

/* grid (rows, 1), 1,024 threads. The bound (the threads, one block) holds a thread to the 64 registers such a block
 * leaves it: a build that would not fit fails (no spill passes it), where without it every launch would. */
extern "C" __global__ void __launch_bounds__(INDEX_SELECT_THREADS_VALUE, 1)
	index_select_entry(struct index_select_arguments_record arguments,
			   unsigned char *arena)
{
    __shared__ struct index_select_shared own;
    const struct step_row_record *const row =
	    index_step_row(arena, arguments.step_offset, arguments.row_first,
			   arguments.context_token_limit);
    LINEAR_DEVICE_CHECK(blockDim.x == INDEX_SELECT_THREADS_VALUE);
    if (row == NULL) {
        return;
    }
    const bool reads_candidates = arguments.candidates_offset !=
				  INDEX_NO_CANDIDATES;
    const uint32_t extent = ((uint32_t)row->position + 1u) / arguments.ratio;
    const uint32_t count =
	    reads_candidates && extent > MODEL_CONTEXT_WITHOUT_CANDIDATES ?
		    MODEL_CONTEXT_WITHOUT_CANDIDATES :
		    extent;
    const uint32_t row_scores =
	    reads_candidates ? MODEL_CONTEXT_WITHOUT_CANDIDATES :
			       arguments.context_token_limit / arguments.ratio;
    const uint16_t *const scores =
        (const uint16_t *)(arena + arguments.scores_offset) + (size_t)blockIdx.x * row_scores;
    const uint32_t *const kept =
	    !reads_candidates ?
		    NULL :
		    (const uint32_t *)(arena + arguments.candidates_offset) +
			    (size_t)blockIdx.x * MODEL_CANDIDATE_BLOCKS;
    uint32_t *const selected = (uint32_t *)(arena + arguments.selection_offset) + (size_t)blockIdx.x * SELECTED;
    index_top_written<false>(scores, count, count < SELECTED ? count : SELECTED, UINT32_MAX, kept, SELECTED, selected,
                             &own);
}

/* grid (rows, 1), 1,024 threads: the kept blocks of the candidate source's rows, as index_select_entry. */
extern "C" __global__ void __launch_bounds__(INDEX_SELECT_THREADS_VALUE, 1)
	index_candidates(struct index_candidates_arguments arguments,
			 unsigned char *arena)
{
    __shared__ struct index_select_shared own;
    const struct step_row_record *const row =
	    index_step_row(arena, arguments.step_offset, arguments.row_first,
			   arguments.context_token_limit);
    LINEAR_DEVICE_CHECK(blockDim.x == INDEX_SELECT_THREADS_VALUE &&
			arguments.entry_capacity % KEYS_PER_LOAD == 0);
    if (row == NULL) {
        return;
    }
    const uint32_t extent = (uint32_t)row->position + 1u;          /* the source's ratio is 1 */
    const uint32_t blocks = extent / KEYS_PER_TILE + (extent % KEYS_PER_TILE != 0 ? 1u : 0u);
    const uint16_t *const scores =
        (const uint16_t *)(arena + arguments.scores_offset) + (size_t)blockIdx.x * arguments.entry_capacity;
    uint32_t *const kept = (uint32_t *)(arena + arguments.candidates_offset) +
			   (size_t)blockIdx.x * MODEL_CANDIDATE_BLOCKS;
    index_top_written<true>(
	    scores, blocks,
	    blocks < MODEL_CANDIDATE_BLOCKS ? blocks : MODEL_CANDIDATE_BLOCKS,
	    blocks - 1u, NULL, MODEL_CANDIDATE_BLOCKS, kept, &own);
}

/* ---- selection distributed over contiguous segments ---- */
#define SPLIT_WARPS (INDEX_SPLIT_THREADS / THREADS_PER_WARP)
#define SPLIT_TURN (SPLIT_WARPS * TURN)

struct index_split_row_state {
    const uint16_t *scores;
    const uint32_t *kept;
    uint32_t *written;
    struct index_split_scratch *scratch;
    uint32_t count, capacity, pinned, run, first, end;
};

__device__ static __forceinline__ bool
index_split_row(struct index_split_arguments arguments, unsigned char *arena,
		struct index_split_row_state *state)
{
	const struct step_row_record *const row = index_step_row(
		arena, arguments.step_offset, arguments.row_first,
		arguments.context_token_limit);
	LINEAR_DEVICE_CHECK(blockDim.x == INDEX_SPLIT_THREADS &&
			    arguments.segments != 0 &&
			    arguments.segments <= INDEX_SPLIT_SEGMENTS &&
			    arguments.is_block <= 1u);
	if (row == NULL) {
		return false;
	}
	const bool reads_candidates = arguments.candidates_offset !=
				      INDEX_NO_CANDIDATES;
	const uint32_t extent =
		((uint32_t)row->position + 1u) / arguments.ratio;
	const uint32_t row_scores =
		reads_candidates ?
			MODEL_CONTEXT_WITHOUT_CANDIDATES :
			arguments.context_token_limit / arguments.ratio;
	state->count =
		arguments.is_block != 0 ?
			(extent + KEYS_PER_TILE - 1u) / KEYS_PER_TILE :
		reads_candidates && extent > MODEL_CONTEXT_WITHOUT_CANDIDATES ?
			MODEL_CONTEXT_WITHOUT_CANDIDATES :
			extent;
	state->capacity = arguments.is_block != 0 ? MODEL_CANDIDATE_BLOCKS :
						    SELECTED;
	state->pinned = arguments.is_block != 0 ? state->count - 1u :
						  UINT32_MAX;
	state->scores = (const uint16_t *)(arena + arguments.scores_offset) +
			(size_t)blockIdx.x * row_scores;
	state->kept =
		reads_candidates ?
			(const uint32_t *)(arena +
					   arguments.candidates_offset) +
				(size_t)blockIdx.x * MODEL_CANDIDATE_BLOCKS :
			NULL;
	state->written = (uint32_t *)(arena + arguments.output_offset) +
			 (size_t)blockIdx.x * state->capacity;
	state->scratch =
		(struct index_split_scratch *)(arena +
					       arguments.scratch_offset) +
		blockIdx.x;
	state->run = (state->count + arguments.segments * SPLIT_TURN - 1u) /
		     (arguments.segments * SPLIT_TURN) * SPLIT_TURN;
	state->first = blockIdx.y * state->run;
	state->end = state->first + state->run < state->count ?
			     state->first + state->run :
			     state->count;
	return true;
}

/* Each warp owns its histogram. Matching lanes elect one writer per bin, as in the legacy selector. */
template <bool is_low, bool is_block>
__device__ static __forceinline__ void index_split_histogram(struct index_split_row_state state,
                                                             uint32_t warp_bins[SPLIT_WARPS][BINS])
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP, warp = threadIdx.x / THREADS_PER_WARP;
    const uint32_t shift = is_low ? 0u : BITS_PER_BYTE;
    const uint32_t high = is_low ? state.scratch->threshold >> BITS_PER_BYTE : 0u;
    for (uint32_t bin = lane; bin < BINS; bin += THREADS_PER_WARP) {
        warp_bins[warp][bin] = 0;
    }
    __syncwarp();
    for (uint32_t first = state.first; first < state.end;
	 first += INDEX_SPLIT_THREADS * KEYS_PER_LOAD) {
	    uint32_t keys[KEYS_PER_LOAD];
	    const uint32_t present = index_element_keys<is_block>(
		    state.scores, state.end,
		    first + threadIdx.x * KEYS_PER_LOAD, state.pinned, keys);
#pragma unroll
        for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
            const uint32_t digit = (keys[key] >> shift) & BYTE_MASK;
            const bool wanted = ((present >> key) & 1u) != 0 && (!is_low || keys[key] >> BITS_PER_BYTE == high);
            const uint32_t peers = __match_any_sync(EVERY_LANE, wanted ? digit : BINS);
            if (wanted && (peers & ((1u << lane) - 1u)) == 0) {
                warp_bins[warp][digit] += (uint32_t)__popc(peers);
            }
            __syncwarp();
        }
    }
    __syncthreads();
    uint32_t total = 0;
    for (uint32_t other = 0; other < SPLIT_WARPS; other++) {
        total += warp_bins[other][threadIdx.x];
    }
    state.scratch->segment[blockIdx.y].bins[threadIdx.x] = total;
}

extern "C" __global__ void
index_split_histogram_high(struct index_split_arguments arguments,
			   unsigned char *arena)
{
    __shared__ uint32_t warp_bins[SPLIT_WARPS][BINS];
    struct index_split_row_state state;
    if (!index_split_row(arguments, arena, &state) || state.count <= state.capacity) {
        return;
    }
    if (arguments.is_block != 0) {
        index_split_histogram<false, true>(state, warp_bins);
    } else {
        index_split_histogram<false, false>(state, warp_bins);
    }
}

extern "C" __global__ void
index_split_histogram_low(struct index_split_arguments arguments,
			  unsigned char *arena)
{
    __shared__ uint32_t warp_bins[SPLIT_WARPS][BINS];
    struct index_split_row_state state;
    if (!index_split_row(arguments, arena, &state) || state.count <= state.capacity) {
        return;
    }
    if (arguments.is_block != 0) {
        index_split_histogram<true, true>(state, warp_bins);
    } else {
        index_split_histogram<true, false>(state, warp_bins);
    }
}

struct index_split_cut_shared {
    uint32_t bins[BINS], cut[CUT_FIELDS];
    uint32_t greater[INDEX_SPLIT_SEGMENTS], ties[INDEX_SPLIT_SEGMENTS];
};

/* Both reductions sum segments in increasing order. The low reduction also derives each segment's greater/tie
 * counts directly from its histogram: there is no separate global count pass. */
template <bool is_low>
__device__ static __forceinline__ void index_split_cut(struct index_split_row_state state, uint32_t segments,
                                                       struct index_split_cut_shared *own)
{
    uint32_t total = 0;
    for (uint32_t segment = 0; segment < segments; segment++) {
        total += state.scratch->segment[segment].bins[threadIdx.x];
    }
    own->bins[threadIdx.x] = total;
    __syncthreads();
    if (threadIdx.x < THREADS_PER_WARP) {
        index_cut(own->bins, state.capacity - (is_low ? state.scratch->above_high : 0u), own->cut);
    }
    __syncthreads();
    if (threadIdx.x < segments) {
	    struct index_split_segment *const partial =
		    &state.scratch->segment[threadIdx.x];
	    uint32_t greater = is_low ? partial->greater_high : 0u;
	    for (uint32_t bin = own->cut[CUT_DIGIT] + 1u; bin < BINS; bin++) {
		    greater += partial->bins[bin];
	    }
        if (is_low) {
            own->greater[threadIdx.x] = greater;
            own->ties[threadIdx.x] = partial->bins[own->cut[CUT_DIGIT]];
        } else {
            partial->greater_high = greater;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (is_low) {
            const uint32_t ties_needed = state.capacity - state.scratch->above_high - own->cut[CUT_ABOVE];
            state.scratch->threshold |= own->cut[CUT_DIGIT];
            state.scratch->ties_needed = ties_needed;
            uint32_t ties_before = 0, outputs_before = 0;
            for (uint32_t segment = 0; segment < segments; segment++) {
                state.scratch->segment[segment].ties_before = ties_before;
                state.scratch->segment[segment].outputs_before = outputs_before;
                const uint32_t left = ties_needed > ties_before ? ties_needed - ties_before : 0u;
                outputs_before += own->greater[segment] + (left < own->ties[segment] ? left : own->ties[segment]);
                ties_before += own->ties[segment];
            }
        } else {
            state.scratch->threshold = own->cut[CUT_DIGIT] << BITS_PER_BYTE;
            state.scratch->above_high = own->cut[CUT_ABOVE];
        }
    }
}

extern "C" __global__ void
index_split_cut_high(struct index_split_arguments arguments,
		     unsigned char *arena)
{
    __shared__ struct index_split_cut_shared own;
    struct index_split_row_state state;
    if (index_split_row(arguments, arena, &state) && state.count > state.capacity) {
        index_split_cut<false>(state, arguments.segments, &own);
    }
}

extern "C" __global__ void
index_split_cut_low(struct index_split_arguments arguments,
		    unsigned char *arena)
{
    __shared__ struct index_split_cut_shared own;
    struct index_split_row_state state;
    if (index_split_row(arguments, arena, &state) && state.count > state.capacity) {
        index_split_cut<true>(state, arguments.segments, &own);
    }
}

struct index_split_output_shared {
    uint32_t greater[SPLIT_WARPS], ties[SPLIT_WARPS];
    uint32_t outputs_before[SPLIT_WARPS], ties_before[SPLIT_WARPS];
};

template <bool is_block>
__device__ static __forceinline__ void index_split_write(struct index_split_row_state state,
                                                         struct index_split_output_shared *own)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP, warp = threadIdx.x / THREADS_PER_WARP;
    const uint32_t threshold = state.scratch->threshold, ties_needed = state.scratch->ties_needed;
    const uint32_t run = state.run / SPLIT_WARPS, first = state.first + warp * run;
    const struct index_split_segment *const partial =
	    &state.scratch->segment[blockIdx.y];
    uint32_t greater = 0, ties = 0;
    for (uint32_t base = first + lane * KEYS_PER_LOAD; base < first + run; base += TURN) {
        uint32_t keys[KEYS_PER_LOAD];
        const uint32_t present = index_element_keys<is_block>(state.scores, state.end, base, state.pinned, keys);
#pragma unroll
        for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
            greater += ((present >> key) & 1u) != 0 && keys[key] > threshold ? 1u : 0u;
            ties += ((present >> key) & 1u) != 0 && keys[key] == threshold ? 1u : 0u;
        }
    }
    greater = __reduce_add_sync(EVERY_LANE, greater);
    ties = __reduce_add_sync(EVERY_LANE, ties);
    if (lane == 0) {
        own->greater[warp] = greater;
        own->ties[warp] = ties;
    }
    __syncthreads();
    if (warp == 0) {
        uint32_t total = 0;
        const uint32_t lane_ties = lane < SPLIT_WARPS ? own->ties[lane] : 0u;
        const uint32_t before = partial->ties_before + index_lanes_before(lane_ties, lane, &total);
        const uint32_t left = ties_needed > before ? ties_needed - before : 0u;
        const uint32_t chosen = lane < SPLIT_WARPS ? own->greater[lane] + (left < lane_ties ? left : lane_ties) : 0u;
        const uint32_t output = partial->outputs_before + index_lanes_before(chosen, lane, &total);
        if (lane < SPLIT_WARPS) {
            own->ties_before[lane] = before;
            own->outputs_before[lane] = output;
        }
    }
    __syncthreads();
    uint32_t outputs_done = own->outputs_before[warp], ties_done = own->ties_before[warp];
    for (uint32_t base = first + lane * KEYS_PER_LOAD; base < first + run; base += TURN) {
        uint32_t keys[KEYS_PER_LOAD], lane_ties = 0, chosen = 0, total = 0;
        const uint32_t present = index_element_keys<is_block>(state.scores, state.end, base, state.pinned, keys);
#pragma unroll
        for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
            lane_ties += ((present >> key) & 1u) != 0 && keys[key] == threshold ? 1u : 0u;
        }
        uint32_t tie_rank = ties_done + index_lanes_before(lane_ties, lane, &total);
        ties_done += total;
#pragma unroll
        for (uint32_t key = 0; key < KEYS_PER_LOAD; key++) {
            const bool there = ((present >> key) & 1u) != 0, tie = there && keys[key] == threshold;
            chosen |= (there && keys[key] > threshold) || (tie && tie_rank < ties_needed) ? 1u << key : 0u;
            tie_rank += tie ? 1u : 0u;
        }
        uint32_t output = outputs_done + index_lanes_before((uint32_t)__popc(chosen), lane, &total);
        outputs_done += total;
        for (uint32_t key = 0; chosen != 0 && key < KEYS_PER_LOAD; key++) {
            if (((chosen >> key) & 1u) != 0) {
                const uint32_t element = base + key;
                state.written[output++] = state.kept != NULL ? state.kept[element / KEYS_PER_TILE] * KEYS_PER_TILE +
                                                                  element % KEYS_PER_TILE : element;
            }
        }
    }
}

extern "C" __global__ void
index_split_output(struct index_split_arguments arguments, unsigned char *arena)
{
    __shared__ struct index_split_output_shared own;
    struct index_split_row_state state;
    if (!index_split_row(arguments, arena, &state)) {
        return;
    }
    if (state.count <= state.capacity) {
        for (uint32_t element = state.first + threadIdx.x; element < state.end; element += blockDim.x) {
            state.written[element] = state.kept != NULL ? state.kept[element / KEYS_PER_TILE] * KEYS_PER_TILE +
                                                            element % KEYS_PER_TILE : element;
        }
        if (blockIdx.y == 0) {
            for (uint32_t output = state.count + threadIdx.x; output < state.capacity; output += blockDim.x) {
                state.written[output] = INDEX_NOTHING_WRITTEN;
            }
        }
    } else if (arguments.is_block != 0) {
        index_split_write<true>(state, &own);
    } else {
        index_split_write<false>(state, &own);
    }
}
