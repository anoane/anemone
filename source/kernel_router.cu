/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_router.cu: the router of one layer, and the draft's (router_select_draft: the select's template
 * stamped for 128 experts and 3 picks). Device code only, built once per architecture into the module
 * "kernel_router". Contract: include/anemone/router_records.h. The block takes the affinities of the live rows into
 * shared memory. Warp r takes row r, lane l experts l + 32 j as keys in the top-k's order (the radix order of a
 * float32, then the lower expert), and takes the warp's largest key out six times: the six in the top-k's order.
 * Lane 0 lays them out by slot; every lane forms the same sum. Thread i below 6 L takes pair i (row i / 6, slot
 * i % 6): its column is the count of earlier pairs of its expert, its job the count of lower experts some pair takes
 * first; column 0 writes the rest of the job, and all zero the empty jobs. Every thread reaches the three barriers.
 * The wide select takes its rows eight a block through the same warps. Its count and place give thread t of a block
 * row t and expert t: a warp turns its rows' picks into a mask an expert (bit l: lane l's row chose it) by a transpose
 * of bits, so a pair's place adds the pairs of lower experts (a scan of every block's counts), of its expert in lower
 * blocks (their counts), in lower warps (their masks) and in lower lanes (its mask below its lane). A block leaves
 * early only as a whole. The jobs give thread t expert t: its first job is a scan of the experts' job counts.
 */
#include "kernel_router.cuh"

#define SORT_ROWS ROUTER_SORT_ROWS
#define SORT_WARPS (SORT_ROWS / THREADS_PER_WARP)
#define NO_EXPERT EXPERTS                       /* a pick of a row at or behind the live ones: in no mask */
#define PAIRS_BITS 16u                          /* one scan adds the pairs, below 2^16, and the tiles above */
#define PAIRS_MASK 0xFFFFu
#define LOWER_HALVES 0x0000FFFFu                /* the bits of a 32-by-32 block's first exchange that stay */

MODEL_LAYOUT_HOLDS(EXPERTS % THREADS_PER_WARP == 0 &&
		   MOST_PAIRS <= ROUTER_THREADS &&
		   MODEL_WIDE_MOST_ROWS * PICKS <= PAIRS_MASK &&
		   ROUTER_WIDE_TILES(MODEL_WIDE_MOST_ROWS) <= PAIRS_MASK);
/* The select of `experts` experts and `picks` picks a row (router_records.h): the model's, and the draft's in the fixed
 * order alone. */
template <uint32_t experts, uint32_t picks>
__device__ static __forceinline__ void
router_select(struct router_select_arguments arguments, unsigned char *arena)
{
	__shared__ float affinities[LINEAR_MAXIMUM_ROWS * experts],
		pair_weights[LINEAR_MAXIMUM_ROWS * picks];
	__shared__ uint32_t pair_experts[LINEAR_MAXIMUM_ROWS * picks],
		pair_is_first[LINEAR_MAXIMUM_ROWS * picks];
	const uint32_t step_rows =
		((const struct step_block *)(arena + arguments.step_offset))
			->row_count;
	const uint32_t live_rows = step_rows < arguments.row_count ?
					   step_rows :
					   arguments.row_count;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       pair = threadIdx.x, pair_count = live_rows * picks;
	LINEAR_DEVICE_CHECK(
		blockDim.x == ROUTER_THREADS &&
		arguments.row_count <= LINEAR_MAXIMUM_ROWS && step_rows >= 1u &&
		step_rows <= arguments.row_count &&
		arguments.reference_order <= (picks == PICKS ? 1u : 0u));
	for (uint32_t index = threadIdx.x; index < live_rows * experts;
	     index += blockDim.x) {
		const float score = ((
			const float *)(arena + arguments.scores_offset))[index];
		affinities[index] = sqrtf(score > ROUTER_SOFTPLUS_THRESHOLD ?
						  score :
						  log1pf(expf(score)));
		((float *)(arena + arguments.affinities_offset))[index] =
			affinities[index];
	}
    __syncthreads();
    ROUTER_SELECT_JOBS(arguments, (const float *)(arena + arguments.bias_offset), experts, picks);
}

extern "C" __global__ void
router_select_entry(struct router_select_arguments arguments,
		    unsigned char *arena)
{
    router_select<EXPERTS, PICKS>(arguments, arena);
}

extern "C" __global__ void
router_select_draft(struct router_select_arguments arguments,
		    unsigned char *arena)
{
	router_select<MODEL_DRAFT_EXPERTS, MODEL_DRAFT_EXPERTS_PER_ROW>(
		arguments, arena);
}

/* The live rows of a wide launch: the step block's row count, at most the record's. */
__device__ static __forceinline__ uint32_t
router_wide_live_rows(const struct router_wide_arguments *arguments,
		      const unsigned char *arena)
{
	const uint32_t step_rows =
		((const struct step_block *)(arena + arguments->step_offset))
			->row_count;
	LINEAR_DEVICE_CHECK(step_rows <= arguments->row_count &&
			    arguments->row_count <= MODEL_WIDE_MOST_ROWS &&
			    arguments->reference_order <= 1u);
	return step_rows < arguments->row_count ? step_rows :
						  arguments->row_count;
}

extern "C" __global__ void
router_wide_select(struct router_wide_arguments arguments, unsigned char *arena)
{
	__shared__ float affinities[LINEAR_MAXIMUM_ROWS * EXPERTS],
		weights[MOST_PAIRS];
	__shared__ uint32_t experts[MOST_PAIRS];
	const uint32_t live_rows = router_wide_live_rows(&arguments, arena),
		       lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t first_row = blockIdx.y * LINEAR_MAXIMUM_ROWS,
		       warp = threadIdx.x / THREADS_PER_WARP;
	LINEAR_DEVICE_CHECK(blockDim.x == ROUTER_THREADS);
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	LINEAR_DEVICE_CHECK((step->flags & ~STEP_BLOCK_PADDED) == 0);
	/* Padded programs route their physical capacity. Close inactive picks, even in an empty half view. */
	if ((step->flags & STEP_BLOCK_PADDED) != 0) {
		const uint32_t row = first_row + warp;
		if (row >= live_rows && row < arguments.row_count &&
		    lane < PICKS) {
			const struct router_pick empty = { NO_EXPERT, 0.0f };
			((struct router_pick *)(arena + arguments.picks_offset))
				[row * PICKS + lane] = empty;
		}
	}
    if (first_row >= live_rows) {
        return;
    }
    const uint32_t rows = live_rows - first_row < LINEAR_MAXIMUM_ROWS ?
				  live_rows - first_row :
				  LINEAR_MAXIMUM_ROWS;
    for (uint32_t index = threadIdx.x; index < rows * EXPERTS; index += blockDim.x) {   /* as the select's */
        const float score = ((const float *)(arena + arguments.scores_offset))[first_row * EXPERTS + index];
	affinities[index] = sqrtf(score > ROUTER_SOFTPLUS_THRESHOLD ?
					  score :
					  log1pf(expf(score)));
	((float *)(arena + arguments.affinities_offset))[first_row * EXPERTS + index] = affinities[index];
    }
    __syncthreads();
    if (warp < rows) {
        router_row<EXPERTS, PICKS>(affinities + warp * EXPERTS, (const float *)(arena + arguments.bias_offset), lane,
                                   arguments.reference_order, experts + warp * PICKS, weights + warp * PICKS);
        if (lane < PICKS) {     /* lane 0 laid the experts out before router_row's barrier; lane l wrote weight l */
		const struct router_pick pick = { experts[warp * PICKS + lane],
						  weights[warp * PICKS + lane] };
		((struct router_pick *)(arena + arguments.picks_offset))
			[(first_row + warp) * PICKS + lane] = pick;
	}
    }
}

/* The picks of the thread's row (row t of the block; NO_EXPERT at or behind the live rows), then masks[w][e], bit l:
 * row 32 w + l chose expert e. A row's choices among each group of 32 experts are a word a lane, which the warp
 * transposes in five exchanges (each swaps the off-diagonal halves of every block of the 32-by-32 bits): lane b then
 * holds the mask of the group's expert b. Every thread reaches the barrier at the end. */
__device__ static __forceinline__ void
router_masks(const struct router_wide_arguments *arguments,
	     const unsigned char *arena, uint32_t live_rows,
	     uint32_t (*masks)[EXPERTS], uint32_t *experts, float *weights)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP, row = blockIdx.y * SORT_ROWS + threadIdx.x;
    const struct router_pick *const picks =
	    (const struct router_pick *)(arena + arguments->picks_offset) +
	    (size_t)row * PICKS;
#pragma unroll
    for (uint32_t slot = 0; slot < PICKS; slot++) {
        experts[slot] = row < live_rows ? picks[slot].expert : NO_EXPERT;
        weights[slot] = row < live_rows ? picks[slot].route_weight : 0.0f;
    }
    LINEAR_DEVICE_CHECK(row >= live_rows || (experts[FIRST] < experts[SECOND] && experts[SECOND] < experts[THIRD] &&
                                             experts[THIRD] < experts[FOURTH] && experts[FOURTH] < experts[FIFTH] &&
                                             experts[FIFTH] < experts[SIXTH] && experts[SIXTH] < EXPERTS));
    for (uint32_t group = 0; group < EXPERTS_PER_LANE; group++) {
        uint32_t bits = 0, lower = LOWER_HALVES;
#pragma unroll
        for (uint32_t slot = 0; slot < PICKS; slot++) {
            bits |= experts[slot] / THREADS_PER_WARP == group ? 1u << (experts[slot] % THREADS_PER_WARP) : 0u;
        }
#pragma unroll
        for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0; distance >>= 1u, lower ^= lower << distance) {
            const uint32_t other = __shfl_xor_sync(ALL_BITS, bits, (int)distance);
            bits = (lane & distance) == 0 ? (bits & lower) | ((other & lower) << distance)
                                          : (bits & ~lower) | ((other >> distance) & lower);
        }
        masks[threadIdx.x / THREADS_PER_WARP][group * THREADS_PER_WARP + lane] = bits;
    }
    __syncthreads();
}

extern "C" __global__ void
router_wide_count(struct router_wide_arguments arguments, unsigned char *arena)
{
    __shared__ uint32_t masks[SORT_WARPS][EXPERTS];
    const uint32_t live_rows = router_wide_live_rows(&arguments, arena);
    uint32_t experts[PICKS], count = 0;
    float weights[PICKS];
    LINEAR_DEVICE_CHECK(blockDim.x == SORT_ROWS);
    if (blockIdx.y * SORT_ROWS < live_rows) {
        router_masks(&arguments, arena, live_rows, masks, experts, weights);
        for (uint32_t warp = 0; warp < SORT_WARPS; warp++) {
            count += (uint32_t)__popc(masks[warp][threadIdx.x]);
        }
        ((uint32_t *)(arena + arguments.counts_offset))[blockIdx.y * EXPERTS + threadIdx.x] = count;
    }
}

/* The sum of value over the block's threads before this one, and over all of them. */
__device__ static __forceinline__ uint32_t router_sum_before(uint32_t value, uint32_t *warp_sums, uint32_t *all)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP, warp = threadIdx.x / THREADS_PER_WARP;
    uint32_t inclusive = value, below = 0, sum = 0;
    for (uint32_t distance = 1; distance < THREADS_PER_WARP; distance <<= 1u) {
        const uint32_t other = __shfl_up_sync(ALL_BITS, inclusive, distance);
        inclusive += lane >= distance ? other : 0u;
    }
    if (lane == THREADS_PER_WARP - 1u) {
        warp_sums[warp] = inclusive;
    }
    __syncthreads();
    for (uint32_t other = 0; other < SORT_WARPS; other++) {
        below += other < warp ? warp_sums[other] : 0u;
        sum += warp_sums[other];
    }
    *all = sum;
    return below + inclusive - value;
}

extern "C" __global__ void
router_wide_place(struct router_wide_arguments arguments, unsigned char *arena)
{
    __shared__ uint32_t masks[SORT_WARPS][EXPERTS], starts[SORT_WARPS][EXPERTS], warp_sums[SORT_WARPS];
    const uint32_t live_rows = router_wide_live_rows(&arguments, arena), block = blockIdx.y, expert = threadIdx.x;
    const uint32_t used_blocks = (live_rows + SORT_ROWS - 1u) / SORT_ROWS, row = block * SORT_ROWS + threadIdx.x;
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP, warp = threadIdx.x / THREADS_PER_WARP;
    const uint32_t *const counts = (const uint32_t *)(arena + arguments.counts_offset);
    uint32_t experts[PICKS], total = 0, before = 0, all = 0;
    float weights[PICKS];
    LINEAR_DEVICE_CHECK(blockDim.x == SORT_ROWS);
    if (block != 0 && block >= used_blocks) {
        return;
    }
    for (uint32_t other = 0; other < used_blocks; other++) {
        total += counts[other * EXPERTS + expert];
        before += other < block ? counts[other * EXPERTS + expert] : 0u;
    }
    const uint32_t tiles = (total + WIDE_TILE_PAIRS - 1u) / WIDE_TILE_PAIRS;
    const uint32_t first = router_sum_before(tiles << PAIRS_BITS | total, warp_sums, &all);
    const uint32_t first_pair = first & PAIRS_MASK, first_tile = first >> PAIRS_BITS;
    uint32_t start = first_pair + before;
    LINEAR_DEVICE_CHECK((all & PAIRS_MASK) == live_rows * PICKS);
    router_masks(&arguments, arena, live_rows, masks, experts, weights);
    for (uint32_t other = 0; other < SORT_WARPS; other++) {
        starts[other][expert] = start;
        start += (uint32_t)__popc(masks[other][expert]);
    }
    LINEAR_DEVICE_CHECK(block >= used_blocks || start - first_pair - before == counts[block * EXPERTS + expert]);
    __syncthreads();
#pragma unroll
    for (uint32_t slot = 0; slot < PICKS; slot++) {
        if (experts[slot] != NO_EXPERT) {
            const uint32_t place = starts[warp][experts[slot]] +
                                   (uint32_t)__popc(masks[warp][experts[slot]] & ((1u << lane) - 1u));
	    const struct wide_pair pair = { row * PICKS + slot, weights[slot] };
	    LINEAR_DEVICE_CHECK(place < arguments.row_count * PICKS);
	    ((struct wide_pair *)(arena + arguments.pairs_offset))[place] =
		    pair;
	}
    }
    if (block != 0) {
        return;
    }
    const uint64_t *const offsets =
	    (const uint64_t *)(arena + arguments.expert_offsets_offset) +
	    (size_t)expert * MODEL_EXPERT_MATRICES;
    const uint64_t gate = offsets[ROUTER_GATE], up = offsets[ROUTER_UP];
    const uint64_t down = offsets[ROUTER_DOWN];
    const uint32_t tile_count = ROUTER_WIDE_TILES(arguments.row_count);
    const struct wide_run run = { first_pair, total };
    struct wide_expert_tile *const tile_array =
	    (struct wide_expert_tile *)(arena + arguments.tiles_offset);
    LINEAR_DEVICE_CHECK(
	    tiles == 0 ||
	    ((gate | up | down) % LINEAR_OFFSET_ALIGNMENT_BYTES == 0 &&
	     gate >= arguments.weights_offset &&
	     gate <= arguments.gate_up_last_offset &&
	     up >= arguments.weights_offset &&
	     up <= arguments.gate_up_last_offset &&
	     down >= arguments.weights_offset &&
	     down <= arguments.down_last_offset &&
	     first_tile + tiles <= tile_count));
    ((struct wide_run *)(arena + arguments.runs_offset))[expert] = run;
    for (uint32_t tile = 0, taken = 0; tile < tiles;
	 tile++, taken += WIDE_TILE_PAIRS) {
	    const uint32_t pair_count = total - taken < WIDE_TILE_PAIRS ?
						total - taken :
						WIDE_TILE_PAIRS;
	    const struct wide_expert_tile filled = { gate, up, down,
						     first_pair + taken,
						     pair_count };
	    tile_array[first_tile + tile] = filled;
    }
    for (uint32_t tile = (all >> PAIRS_BITS) + threadIdx.x; tile < tile_count; tile += blockDim.x) {
	    const struct wide_expert_tile empty = { 0, 0, 0, 0, 0 };
	    tile_array[tile] = empty;
    }
}

/* The job table of the runs: thread e cuts expert e's run in jobs of 8 pairs after the jobs of the lower experts (a
 * scan of the job counts), copies each pair's index and weight, and the threads clear the jobs behind the last. */
extern "C" __global__ void
router_wide_jobs(struct router_wide_arguments arguments, unsigned char *arena)
{
    __shared__ uint32_t warp_sums[SORT_WARPS];
    const uint32_t expert = threadIdx.x,
		   capacity = ROUTER_WIDE_JOBS(arguments.row_count);
    const struct wide_run run =
	    ((const struct wide_run *)(arena + arguments.runs_offset))[expert];
    const uint32_t jobs =
	    (run.pair_count + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
    uint32_t all = 0;
    const uint32_t first_job = router_sum_before(jobs, warp_sums, &all);
    LINEAR_DEVICE_CHECK(blockDim.x == SORT_ROWS && gridDim.y == 1 && all <= capacity);
    const uint64_t *const offsets =
	    (const uint64_t *)(arena + arguments.expert_offsets_offset) +
	    (size_t)expert * MODEL_EXPERT_MATRICES;
    const struct wide_pair *const pairs =
	    (const struct wide_pair *)(arena + arguments.pairs_offset) +
	    run.first_pair;
    struct expert_job_record *const job_array =
	    (struct expert_job_record *)(arena + arguments.jobs_offset);
    for (uint32_t job = 0, taken = 0; job < jobs;
	 job++, taken += LINEAR_MAXIMUM_ROWS) {
	    struct expert_job_record filled = { offsets[ROUTER_GATE],
						offsets[ROUTER_UP],
						offsets[ROUTER_DOWN],
						0,
						expert,
						{ 0 },
						{ 0 } };
	    filled.row_count = run.pair_count - taken < LINEAR_MAXIMUM_ROWS ?
				       run.pair_count - taken :
				       LINEAR_MAXIMUM_ROWS;
	    for (uint32_t column = 0; column < filled.row_count; column++) {
		    filled.pair_index[column] =
			    pairs[taken + column].pair_index;
		    filled.route_weight[column] =
			    pairs[taken + column].route_weight;
	    }
	    job_array[first_job + job] = filled;
    }
    for (uint32_t job = all + threadIdx.x; job < capacity; job += blockDim.x) {
	    const struct expert_job_record empty = {
		    0, 0, 0, 0, 0, { 0 }, { 0 }
	    };
	    job_array[job] = empty;
    }
}

/* router_wide_small (router.h): at most S rows, one block, the count, the place and the jobs of one launch.
 * Thread t takes row t and expert t; its expert's pairs are its block's masks' count (the count kernel's sum, the
 * counts scratch unwritten), the place's scan and writes as the place's block 0 (no pairs in lower blocks), then
 * behind a barrier (every pair placed) the jobs' as theirs: the same pairs, runs, tiles and jobs. */
extern "C" __global__ void
router_wide_small(struct router_wide_arguments arguments, unsigned char *arena)
{
    __shared__ uint32_t masks[SORT_WARPS][EXPERTS], starts[SORT_WARPS][EXPERTS], warp_sums[SORT_WARPS];
    const uint32_t live_rows = router_wide_live_rows(&arguments, arena), expert = threadIdx.x, row = threadIdx.x;
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP, warp = threadIdx.x / THREADS_PER_WARP;
    uint32_t experts[PICKS], total = 0, all = 0;
    float weights[PICKS];
    LINEAR_DEVICE_CHECK(blockDim.x == SORT_ROWS && gridDim.y == 1 && arguments.row_count <= SORT_ROWS);
    router_masks(&arguments, arena, live_rows, masks, experts, weights);
    for (uint32_t other = 0; other < SORT_WARPS; other++) {
        total += (uint32_t)__popc(masks[other][expert]);
    }
    const uint32_t tiles = (total + WIDE_TILE_PAIRS - 1u) / WIDE_TILE_PAIRS;
    const uint32_t first = router_sum_before(tiles << PAIRS_BITS | total, warp_sums, &all);
    const uint32_t first_pair = first & PAIRS_MASK, first_tile = first >> PAIRS_BITS;
    uint32_t start = first_pair;
    LINEAR_DEVICE_CHECK((all & PAIRS_MASK) == live_rows * PICKS);
    for (uint32_t other = 0; other < SORT_WARPS; other++) {
        starts[other][expert] = start;
        start += (uint32_t)__popc(masks[other][expert]);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t slot = 0; slot < PICKS; slot++) {
        if (experts[slot] != NO_EXPERT) {
            const uint32_t place = starts[warp][experts[slot]] +
                                   (uint32_t)__popc(masks[warp][experts[slot]] & ((1u << lane) - 1u));
	    const struct wide_pair pair = { row * PICKS + slot, weights[slot] };
	    LINEAR_DEVICE_CHECK(place < arguments.row_count * PICKS);
	    ((struct wide_pair *)(arena + arguments.pairs_offset))[place] =
		    pair;
	}
    }
    const uint64_t *const offsets =
	    (const uint64_t *)(arena + arguments.expert_offsets_offset) +
	    (size_t)expert * MODEL_EXPERT_MATRICES;
    const uint64_t gate = offsets[ROUTER_GATE], up = offsets[ROUTER_UP];
    const uint64_t down = offsets[ROUTER_DOWN];
    const uint32_t tile_count = ROUTER_WIDE_TILES(arguments.row_count);
    const struct wide_run run = { first_pair, total };
    struct wide_expert_tile *const tile_array =
	    (struct wide_expert_tile *)(arena + arguments.tiles_offset);
    LINEAR_DEVICE_CHECK(
	    tiles == 0 ||
	    ((gate | up | down) % LINEAR_OFFSET_ALIGNMENT_BYTES == 0 &&
	     gate >= arguments.weights_offset &&
	     gate <= arguments.gate_up_last_offset &&
	     up >= arguments.weights_offset &&
	     up <= arguments.gate_up_last_offset &&
	     down >= arguments.weights_offset &&
	     down <= arguments.down_last_offset &&
	     first_tile + tiles <= tile_count));
    ((struct wide_run *)(arena + arguments.runs_offset))[expert] = run;
    for (uint32_t tile = 0, taken = 0; tile < tiles;
	 tile++, taken += WIDE_TILE_PAIRS) {
	    const uint32_t pair_count = total - taken < WIDE_TILE_PAIRS ?
						total - taken :
						WIDE_TILE_PAIRS;
	    const struct wide_expert_tile filled = { gate, up, down,
						     first_pair + taken,
						     pair_count };
	    tile_array[first_tile + tile] = filled;
    }
    for (uint32_t tile = (all >> PAIRS_BITS) + threadIdx.x; tile < tile_count; tile += blockDim.x) {
	    const struct wide_expert_tile empty = { 0, 0, 0, 0, 0 };
	    tile_array[tile] = empty;
    }
    __syncthreads(); /* every pair placed and every run written before the jobs read them (the jobs kernel's input) */
    const uint32_t capacity = ROUTER_WIDE_JOBS(arguments.row_count);
    const uint32_t jobs =
	    (run.pair_count + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
    uint32_t job_total = 0;
    const uint32_t first_job = router_sum_before(jobs, warp_sums, &job_total);
    LINEAR_DEVICE_CHECK(job_total <= capacity);
    const struct wide_pair *const pairs =
	    (const struct wide_pair *)(arena + arguments.pairs_offset) +
	    run.first_pair;
    struct expert_job_record *const job_array =
	    (struct expert_job_record *)(arena + arguments.jobs_offset);
    for (uint32_t job = 0, taken = 0; job < jobs;
	 job++, taken += LINEAR_MAXIMUM_ROWS) {
	    struct expert_job_record filled = { gate,	up,    down, 0,
						expert, { 0 }, { 0 } };
	    filled.row_count = run.pair_count - taken < LINEAR_MAXIMUM_ROWS ?
				       run.pair_count - taken :
				       LINEAR_MAXIMUM_ROWS;
	    for (uint32_t column = 0; column < filled.row_count; column++) {
		    filled.pair_index[column] =
			    pairs[taken + column].pair_index;
		    filled.route_weight[column] =
			    pairs[taken + column].route_weight;
	    }
	    job_array[first_job + job] = filled;
    }
    for (uint32_t job = job_total + threadIdx.x; job < capacity; job += blockDim.x) {
	    const struct expert_job_record empty = {
		    0, 0, 0, 0, 0, { 0 }, { 0 }
	    };
	    job_array[job] = empty;
    }
}
