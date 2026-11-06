/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_exl3_tc.cu: the routed experts of one EXL3 layer on the tensor cores from the router's job table, one TP
 * rank's intermediate slice (exl3_tc_records.h; arithmetic: kernel_exl3_tc.cuh, GNmac's `moe_out_v2_try`). Device code
 * only, sm_80, built into the module "kernel_exl3_tc". A job is a route tile: one expert, at most 8 of its pairs
 * (pair = row x 6 + slot) with their weights, the expert's three matrices by arena offset. Kernels:
 *   operands_{short,long}: block 0 counts the jobs (the table holds the routed experts ascending, the empty jobs last)
 *     and lays out each pair's down svh and weight for the combine; the others form one (pair, projection) gate/up
 *     operand row a work item;
 *   products_t{20,40,16,32}_k{3,2}: the lane-tile products of the gate/up pass (record.pass 0) or the down pass (1),
 *     k_steps 16-steps a K slice; products_t{20,16}x2_k{3,2}: the same slices two an item (exl3_tc_prepare_
 *     paired: the short forms' partials, bit for bit, half the items);
 *   activate_s{16,8}_i{640,512}: one pair a block;
 *   combine_s{2,1}: one (row, 128 outputs) a block, FP32 out (record.shared_sum: added to the shared expert's
 *     bfloat16 rows in place and rounded to bfloat16, tp2_join's BFLOAT16_SUM_TO_BFLOAT16 arithmetic).
 * --exl3_expert_rotation (record.rotation, intermediate 640): each job's expert 640 or 512 wide (job.expert,
 * exl3_tc_width); the _m forms run each job at its width in the 640 layout (operands_{short,long}_m,
 * products_t{20,40}_k{3,2}_m for gate/up, products_t20n16 / t40n32 for down: a narrow job's own K steps a slice, the
 * same slice count; activate_s{16,8}_m640), bitwise the base forms' per route.
 */
#include "anemone/exl3_tc_records.h"
#include "anemone/linear_records.h"
#include "kernel_exl3_tc.cuh"
#include "kernel_linear_common.cuh"

#define JOB_ROWS EXL3_TC_JOB_ROWS
#define PAIRS_PER_ROW EXL3_TC_PAIRS_PER_ROW
static_assert(JOB_ROWS == LINEAR_MAXIMUM_ROWS && JOB_ROWS == TC_MOE_TILE_ROWS,
	      "a job is a route tile");
static_assert(EXL3_TC_HIDDEN == TC_MOE_HIDDEN &&
		      EXL3_TC_OPERAND_THREADS == TC_MOE_ROUTE_THREADS &&
		      EXL3_TC_PRODUCT_THREADS == TC_MOE_LANE_THREADS &&
		      EXL3_TC_LONG_PAIRS == TC_MOE_LONG_ASSIGNMENTS &&
		      EXL3_TC_STASH_BYTES ==
			      tc_moe_stash_bytes<TC_MOE_COLUMN_TILES>(),
	      "one geometry");

struct tc_workspace { /* the record's workspace, its parts at their offsets (exl3_tc_records.h) */
	float *gate, *up, *down;
	__half *gate_up_halves, *down_halves;
	float *gate_up_unscale, *down_unscale;
	int32_t *count;
	struct exl3_tc_by_pair *by_pair;
	int64_t pairs;
};

__device__ __forceinline__ struct tc_workspace
tc_workspace_of(const struct exl3_tc_record &record, unsigned char *arena)
{
	const uint64_t pairs = (uint64_t)record.row_count * PAIRS_PER_ROW, inner = record.intermediate;
	unsigned char *base = arena + record.workspace_offset;
	struct tc_workspace w;
	w.gate = (float *)(base +
			   exl3_tc_part_offset(EXL3_TC_GATE, pairs, inner));
	w.up = (float *)(base + exl3_tc_part_offset(EXL3_TC_UP, pairs, inner));
	w.down = (float *)(base +
			   exl3_tc_part_offset(EXL3_TC_DOWN, pairs, inner));
	w.gate_up_halves =
		(__half *)(base + exl3_tc_part_offset(EXL3_TC_GATE_UP_HALVES,
						      pairs, inner));
	w.down_halves =
		(__half *)(base + exl3_tc_part_offset(EXL3_TC_DOWN_HALVES,
						      pairs, inner));
	w.gate_up_unscale =
		(float *)(base + exl3_tc_part_offset(EXL3_TC_GATE_UP_UNSCALE,
						     pairs, inner));
	w.down_unscale =
		(float *)(base + exl3_tc_part_offset(EXL3_TC_DOWN_UNSCALE,
						     pairs, inner));
	w.count = (int32_t *)(base +
			      exl3_tc_part_offset(EXL3_TC_COUNT, pairs, inner));
	w.by_pair = (struct exl3_tc_by_pair *)(base + exl3_tc_part_offset(
							      EXL3_TC_BY_PAIR,
							      pairs, inner));
	w.pairs = (int64_t)pairs;
	return w;
}

/* An EXL3 matrix at `offset`: its trellis, then FP16 suh[input], svh[output]. */
__device__ __forceinline__ uint64_t tc_trellis_bytes(uint64_t input, uint64_t output, uint32_t bits_x2)
{
	return input * output * bits_x2 / 16u;
}
__device__ __forceinline__ uint64_t tc_suh_offset(uint64_t matrix, uint64_t input, uint64_t output, uint32_t bits_x2)
{
	return matrix + tc_trellis_bytes(input, output, bits_x2);
}
__device__ __forceinline__ uint64_t tc_svh_offset(uint64_t matrix, uint64_t input, uint64_t output, uint32_t bits_x2)
{
	return tc_suh_offset(matrix, input, output, bits_x2) + input * sizeof(__half);
}

/* The routed pair `routed` of the table (pairs counted in job order, the jobs ascending and the empty ones last): its
 * job and row into the block's shared slot; false past the routed pairs. Warp 0 scans the jobs' row counts; every
 * thread of the block calls it (a barrier). GNmac's schedule: a block a (pair, projection) or a pair. */
__device__ __forceinline__ bool
tc_routed_entry(const struct expert_job_record *jobs, uint32_t job_capacity,
		int64_t routed, uint32_t *job_out, uint32_t *row_out)
{
	__shared__ int32_t found_job, found_row;
	if (threadIdx.x < TC_MOE_WARP) {
		const uint32_t lane = threadIdx.x;
		int64_t before = 0; /* the routed pairs of the jobs scanned so far */
		int32_t job_found = -1, row_found = 0;
		for (uint32_t first = 0; first < job_capacity && job_found < 0; first += TC_MOE_WARP) {
			const uint32_t job = first + lane;
			const int64_t rows = job < job_capacity ? (int64_t)jobs[job].row_count : 0;
			int64_t through = rows; /* inclusive scan of the warp's row counts */
			for (int distance = 1; distance < TC_MOE_WARP; distance *= 2) {
				const int64_t other = __shfl_up_sync(TC_MOE_FULL_MASK, through, distance);
				through += lane >= (uint32_t)distance ? other : 0;
			}
			const bool holds = rows != 0 && routed >= before + through - rows && routed < before + through;
			const uint32_t ballot = __ballot_sync(TC_MOE_FULL_MASK, holds);
			if (ballot != 0u) {
				const int source = __ffs((int)ballot) - 1;
				const int64_t start = __shfl_sync(TC_MOE_FULL_MASK, before + through - rows, source);
				job_found = (int32_t)(first + (uint32_t)source);
				row_found = (int32_t)(routed - start);
			}
			before += __shfl_sync(TC_MOE_FULL_MASK, through, TC_MOE_WARP - 1);
		}
		if (lane == 0) {
			found_job = job_found;
			found_row = row_found;
		}
	}
	__syncthreads();
	const bool is_routed = found_job >= 0;
	*job_out = (uint32_t)(is_routed ? found_job : 0);
	*row_out = (uint32_t)found_row;
	__syncthreads(); /* the slot's next use */
	return is_routed;
}

/* ---- operands: block 0 counts the jobs and lays out the pairs; the others a routed (pair, projection) each ---- */
/* The width of a job's expert: record.intermediate, or a mixed form's per-expert width. */
template <bool mixed>
__device__ __forceinline__ uint32_t
tc_job_width(const struct exl3_tc_record &record,
	     const struct expert_job_record &job)
{
	return mixed ? exl3_tc_width(record.intermediate, record.rotation,
				     job.expert) :
		       record.intermediate;
}

template <int scale_k, int split, bool mixed>
__device__ __forceinline__ void tc_operands(const struct exl3_tc_record &record,
					    unsigned char *arena)
{
	const struct tc_workspace w = tc_workspace_of(record, arena);
	const struct expert_job_record *jobs =
		(const struct expert_job_record *)(arena + record.jobs_offset);
	if (blockIdx.x == 0) {
		__shared__ int32_t first_empty;
		if (threadIdx.x == 0)
			first_empty = (int32_t)record.job_capacity;
		for (int64_t pair = threadIdx.x; pair < w.pairs; pair += blockDim.x)
			w.by_pair[pair].is_routed = 0u;
		__syncthreads();
		for (uint32_t job = threadIdx.x; job < record.job_capacity; job += blockDim.x) {
			const uint32_t rows = jobs[job].row_count;
			if (rows == 0u)
				atomicMin(&first_empty, (int32_t)job);
		}
		__syncthreads();
		for (uint32_t entry = threadIdx.x; entry < record.job_capacity * JOB_ROWS; entry += blockDim.x) {
			const struct expert_job_record &job =
				jobs[entry / JOB_ROWS];
			const uint32_t row = entry % JOB_ROWS;
			if (row >= job.row_count)
				continue;
			struct exl3_tc_by_pair *by_pair =
				&w.by_pair[job.pair_index[row]];
			by_pair->svh_offset = tc_svh_offset(job.down_offset, tc_job_width<mixed>(record, job),
							    TC_MOE_HIDDEN, record.bits_x2);
			by_pair->route_weight = job.route_weight[row];
			by_pair->is_routed = 1u;
		}
		if (threadIdx.x == 0)
			*w.count = first_empty;
		return;
	}
	for (int64_t work = blockIdx.x - 1; work < w.pairs * 2; work += gridDim.x - 1) {
		uint32_t job_index, row;
		if (!tc_routed_entry(jobs, record.job_capacity, work / 2, &job_index, &row))
			break; /* uniform a block: past the routed pairs */
		const struct expert_job_record &job = jobs[job_index];
		const int projection = (int)(work & 1);
		const uint32_t pair = job.pair_index[row];
		const uint64_t matrix = projection == 0 ? job.gate_offset : job.up_offset;
		tc_moe_operand_row<scale_k, split>(
			(const __nv_bfloat16 *)(arena + record.hidden_offset) +
				(uint64_t)(pair / PAIRS_PER_ROW) * TC_MOE_HIDDEN,
			(const __half *)(arena + tc_suh_offset(matrix, TC_MOE_HIDDEN, tc_job_width<mixed>(record, job),
							       record.bits_x2)),
			pair, projection, w.gate_up_halves, w.gate_up_unscale);
	}
}

#define TC_OPERANDS(scale_k, split, mixed, name)                           \
	extern "C" __global__ void __launch_bounds__(TC_MOE_ROUTE_THREADS) \
		name(struct exl3_tc_record record, unsigned char *arena)   \
	{                                                                  \
		tc_operands<scale_k, split, mixed>(record, arena);         \
	}
TC_OPERANDS(5120, 16, false, exl3_tc_operands_short)
TC_OPERANDS(1280, 2, false, exl3_tc_operands_long)
TC_OPERANDS(5120, 16, true, exl3_tc_operands_short_m)
TC_OPERANDS(1280, 2, true, exl3_tc_operands_long_m)
#undef TC_OPERANDS

/* ---- products: warp items (job, projection, K slice, column group), column groups fastest ----
 * mixed: a narrow job's gate/up column groups past its width exit and its trellis rows hold its own column tiles; its
 * down items run narrow_k_steps k steps a slice (the slice count of the 640 shape). item_slices 2: an item two
 * adjacent slices (tc_moe_item), the warps' stashes behind the rings in the dynamic shared memory. */
template <int k_steps, int narrow_k_steps, int bits_x2, bool mixed,
	  int item_slices = 1>
__device__ __forceinline__ void tc_products(const struct exl3_tc_record &record,
					    unsigned char *arena)
{
	static_assert(item_slices == 1 || !mixed, "two slices an item: the base forms");
	constexpr int wide_tiles = (int)EXL3_TC_MIXED_WIDTH / TC_MOE_TILE;
	constexpr int narrow_tiles =
		wide_tiles - (int)EXL3_TC_NARROWER / TC_MOE_TILE;
	static_assert(!mixed || k_steps == narrow_k_steps ||
			      (wide_tiles / k_steps == narrow_tiles / narrow_k_steps && narrow_tiles % narrow_k_steps == 0),
		      "a narrow down job keeps the slice count");
	extern __shared__ __align__(16) unsigned char products_rings[];
	constexpr int column_tiles = TC_MOE_COLUMN_TILES;
	const struct tc_workspace w = tc_workspace_of(record, arena);
	const struct expert_job_record *jobs =
		(const struct expert_job_record *)(arena + record.jobs_offset);
	const bool down = record.pass != 0u;
	struct tc_moe_products p;
	p.k_tiles = (int)((down ? record.intermediate : TC_MOE_HIDDEN) / TC_MOE_TILE);
	p.n_tiles = (int)((down ? TC_MOE_HIDDEN : record.intermediate) / TC_MOE_TILE);
	p.projections = down ? 1 : 2;
	p.first_projection = down ? 2 : 0;
	p.output = down ? w.down : w.gate;
	p.second_output = down ? NULL : w.up;
	p.slice_stride = w.pairs * (down ? TC_MOE_HIDDEN : (int64_t)record.intermediate);
	p.halves = down ? w.down_halves : w.gate_up_halves;
	p.unscale = down ? w.down_unscale : w.gate_up_unscale;
	const int lane = threadIdx.x & 31;
	const tc_moe_window<bits_x2> window(lane);
	unsigned char *ring = products_rings + (threadIdx.x >> 5) * tc_moe_ring_stages(column_tiles, bits_x2) *
						       tc_moe_step_bytes(column_tiles, bits_x2);
	float *const stash = (float *)(products_rings + tc_moe_ring_bytes(column_tiles, bits_x2) +
				       (threadIdx.x >> 5) * tc_moe_stash_bytes<column_tiles>());
	const int slices = p.k_tiles / k_steps / item_slices, column_groups = p.n_tiles / column_tiles;
	const int64_t per_job = (int64_t)p.projections * slices * column_groups;
	const int64_t items = (int64_t)(*w.count) * per_job;
	const int64_t warps = (int64_t)gridDim.x * (TC_MOE_LANE_THREADS / TC_MOE_WARP);
	constexpr int load = tc_moe_uses_ring(column_tiles, bits_x2) ? TC_MOE_LOAD_RING : TC_MOE_LOAD_WORDS;
	for (int64_t item = (int64_t)blockIdx.x * (TC_MOE_LANE_THREADS / TC_MOE_WARP) + (threadIdx.x >> 5);
	     item < items; item += warps) {
		const struct expert_job_record &job = jobs[item / per_job];
		int rest = (int)(item % per_job);
		const int column_group = rest % column_groups;
		rest /= column_groups;
		const int slice = rest % slices * item_slices, projection_index = rest / slices;
		const int projection = p.first_projection + projection_index;
		const uint64_t matrix = projection == 0 ? job.gate_offset : projection == 1 ? job.up_offset : job.down_offset;
		const uint16_t *trellis = (const uint16_t *)(arena + matrix);
		const int rows = (int)job.row_count;
		const int row_of = lane < rows ? (int)job.pair_index[lane] : 0;
		float *output = (projection_index == 0 ? p.output : p.second_output) + slice * p.slice_stride;
		int tiles_n = p.n_tiles;
		bool narrow = false;
		if constexpr (mixed) {
			const int width_tiles = (int)tc_job_width<mixed>(record, job) / TC_MOE_TILE;
			if (!down && column_group * column_tiles >= width_tiles)
				continue; /* a narrow expert's last column group */
			tiles_n = down ? p.n_tiles : width_tiles;
			narrow = down && width_tiles != p.k_tiles;
		}
		if (narrow) {
			if constexpr (mixed) {
				if (rows <= 4)
					tc_moe_item<1, column_tiles, narrow_k_steps, bits_x2, load>(
						p, projection_index, slice, row_of, rows, trellis, tiles_n,
						column_group * column_tiles, lane, window, ring, output);
				else
					tc_moe_item<2, column_tiles, narrow_k_steps, bits_x2, load>(
						p, projection_index, slice, row_of, rows, trellis, tiles_n,
						column_group * column_tiles, lane, window, ring, output);
			}
		} else if (rows <= 4)
			tc_moe_item<1, column_tiles, k_steps, bits_x2, load, item_slices>(
				p, projection_index, slice, row_of, rows, trellis, tiles_n, column_group * column_tiles,
				lane, window, ring, output, stash);
		else if constexpr (item_slices == 2) {
			/* two n8 groups: the column group's halves in turn, each paired through the ring (whole tiles
			 * or the register path of K2's 256-byte half steps would spill) */
			constexpr int half_tiles = column_tiles / 2;
#pragma unroll 1
			for (int half = 0; half < 2; half++)
				tc_moe_item<2, half_tiles, k_steps, bits_x2, TC_MOE_LOAD_RING, 2>(
					p, projection_index, slice, row_of, rows, trellis, tiles_n,
					column_group * column_tiles + half * half_tiles, lane, window, ring, output,
					stash);
		} else
			tc_moe_item<2, column_tiles, k_steps, bits_x2, load>(
				p, projection_index, slice, row_of, rows, trellis, tiles_n, column_group * column_tiles,
				lane, window, ring, output);
	}
}

#define TC_PRODUCTS(k_steps, narrow_k_steps, bits_x2, mixed, name)           \
	extern "C" __global__ void __launch_bounds__(TC_MOE_LANE_THREADS, 2) \
		name(struct exl3_tc_record record, unsigned char *arena)     \
	{                                                                    \
		tc_products<k_steps, narrow_k_steps, bits_x2, mixed>(record, \
								     arena); \
	}
TC_PRODUCTS(20, 20, 6, false, exl3_tc_products_t20_k3)
TC_PRODUCTS(40, 40, 6, false, exl3_tc_products_t40_k3)
TC_PRODUCTS(16, 16, 6, false, exl3_tc_products_t16_k3)
TC_PRODUCTS(32, 32, 6, false, exl3_tc_products_t32_k3)
TC_PRODUCTS(20, 20, 4, false, exl3_tc_products_t20_k2)
TC_PRODUCTS(40, 40, 4, false, exl3_tc_products_t40_k2)
TC_PRODUCTS(16, 16, 4, false, exl3_tc_products_t16_k2)
TC_PRODUCTS(32, 32, 4, false, exl3_tc_products_t32_k2)
/* --exl3_expert_rotation: gate/up (record.pass 0) at 640 with narrow jobs' column groups cut, down (pass 1) at 640
 * with narrow jobs' K steps */
TC_PRODUCTS(20, 20, 6, true, exl3_tc_products_t20_k3_m)
TC_PRODUCTS(40, 40, 6, true, exl3_tc_products_t40_k3_m)
TC_PRODUCTS(20, 16, 6, true, exl3_tc_products_t20n16_k3)
TC_PRODUCTS(40, 32, 6, true, exl3_tc_products_t40n32_k3)
TC_PRODUCTS(20, 20, 4, true, exl3_tc_products_t20_k2_m)
TC_PRODUCTS(40, 40, 4, true, exl3_tc_products_t40_k2_m)
TC_PRODUCTS(20, 16, 4, true, exl3_tc_products_t20n16_k2)
TC_PRODUCTS(40, 32, 4, true, exl3_tc_products_t40n32_k2)
#undef TC_PRODUCTS
/* two slices an item (exl3_tc_prepare_paired): gate/up t20x2, down t20x2 (640) or t16x2 (512) */
#define TC_PRODUCTS_PAIRED(k_steps, bits_x2, name)                           \
	extern "C" __global__ void __launch_bounds__(TC_MOE_LANE_THREADS, 2) \
		name(struct exl3_tc_record record, unsigned char *arena)     \
	{                                                                    \
		tc_products<k_steps, k_steps, bits_x2, false, 2>(record,     \
								 arena);     \
	}
TC_PRODUCTS_PAIRED(20, 6, exl3_tc_products_t20x2_k3)
TC_PRODUCTS_PAIRED(16, 6, exl3_tc_products_t16x2_k3)
TC_PRODUCTS_PAIRED(20, 4, exl3_tc_products_t20x2_k2)
TC_PRODUCTS_PAIRED(16, 4, exl3_tc_products_t16x2_k2)
#undef TC_PRODUCTS_PAIRED

/* ---- activation: a routed pair each, the jobs in table order ---- */
template <int slices, int row_blocks, bool mixed>
__device__ __forceinline__ void tc_activate(const struct exl3_tc_record &record,
					    unsigned char *arena)
{
	const struct tc_workspace w = tc_workspace_of(record, arena);
	const struct expert_job_record *jobs =
		(const struct expert_job_record *)(arena + record.jobs_offset);
	const uint32_t inner = record.intermediate;
	/* the short forms' 16 gate/up slices go with 2 down slices, the long forms' 8 with one (exl3_tc.c names the
	 * forms by the pairs, or the short ones at every row count: exl3_tc_prepare_short) */
	const int down_slices = slices == (int)EXL3_TC_GATE_UP_SLICES ?
					(int)EXL3_TC_DOWN_SLICES :
					1;
	for (int64_t routed = blockIdx.x; routed < w.pairs; routed += gridDim.x) {
		uint32_t job_index, row;
		if (!tc_routed_entry(jobs, record.job_capacity, routed, &job_index, &row))
			break; /* uniform a block: past the routed pairs */
		const struct expert_job_record &job = jobs[job_index];
		const uint32_t width = tc_job_width<mixed>(record, job); /* the matrices' own shape */
		tc_moe_activate_pair<slices, row_blocks, mixed>(
			job.pair_index[row], inner, w.gate, w.up, w.pairs * inner,
			(const __half *)(arena + tc_svh_offset(job.gate_offset, TC_MOE_HIDDEN, width, record.bits_x2)),
			(const __half *)(arena + tc_svh_offset(job.up_offset, TC_MOE_HIDDEN, width, record.bits_x2)),
			(const __half *)(arena + tc_suh_offset(job.down_offset, width, TC_MOE_HIDDEN, record.bits_x2)),
			w.down_halves, w.down_unscale, down_slices, (double)record.activation_limit, width);
	}
}

#define TC_ACTIVATE(slices, inner, mixed, name)                              \
	extern "C" __global__ void __launch_bounds__(                        \
		TC_MOE_WARP *(inner / TC_MOE_TRANSFORM))                     \
		name(struct exl3_tc_record record, unsigned char *arena)     \
	{                                                                    \
		tc_activate<slices, inner / TC_MOE_TRANSFORM, mixed>(record, \
								     arena); \
	}
TC_ACTIVATE(16, 640, false, exl3_tc_activate_s16_i640)
TC_ACTIVATE(8, 640, false, exl3_tc_activate_s8_i640)
TC_ACTIVATE(16, 512, false, exl3_tc_activate_s16_i512)
TC_ACTIVATE(8, 512, false, exl3_tc_activate_s8_i512)
TC_ACTIVATE(16, 640, true, exl3_tc_activate_s16_m640)
TC_ACTIVATE(8, 640, true, exl3_tc_activate_s8_m640)
#undef TC_ACTIVATE

/* ---- combine: one (row, 128 outputs) a block, a warp a route (slot), the routes' FMAs in slot order ---- */
template <int slices>
__device__ __forceinline__ void tc_combine(const struct exl3_tc_record &record,
					   unsigned char *arena)
{
	extern __shared__ __align__(16) unsigned char combine_shared[];
	const struct tc_workspace w = tc_workspace_of(record, arena);
	double *values = (double *)combine_shared; /* [routes][128] */
	float *route_weight = (float *)(values + PAIRS_PER_ROW * TC_MOE_TRANSFORM);
	int *route_live = (int *)(route_weight + PAIRS_PER_ROW);
	float *output = (float *)(arena + record.output_offset);
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int64_t blocks = TC_MOE_HIDDEN / TC_MOE_TRANSFORM, tasks = (int64_t)record.row_count * blocks;
	for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
		const int64_t row = task / blocks, first_column = (task % blocks) * TC_MOE_TRANSFORM;
		for (int route = warp; route < (int)PAIRS_PER_ROW; route += (int)(blockDim.x / TC_MOE_WARP)) {
			const struct exl3_tc_by_pair entry =
				w.by_pair[row * PAIRS_PER_ROW + route];
			if (lane == 0) {
				route_live[route] = entry.is_routed != 0u;
				route_weight[route] = entry.route_weight;
			}
			if (entry.is_routed == 0u)
				continue;
			tc_moe_route_values<slices>(w.down, w.pairs * TC_MOE_HIDDEN, row * PAIRS_PER_ROW + route, first_column,
						    (const __half *)(arena + entry.svh_offset), lane,
						    values + route * TC_MOE_TRANSFORM);
		}
		__syncthreads();
		if (threadIdx.x < TC_MOE_TRANSFORM) {
			double sum = 0.0;
			for (int route = 0; route < (int)PAIRS_PER_ROW; route++)
				if (route_live[route])
					sum = __fma_rn(values[route * TC_MOE_TRANSFORM + threadIdx.x], (double)route_weight[route], sum);
			const float value = __double2float_rn(sum);
			const int64_t place = row * TC_MOE_HIDDEN + first_column + threadIdx.x;
			if (record.shared_sum != 0u) { /* the shared expert's row plus the sum, rounded once */
				uint16_t *const shared = (uint16_t *)(arena + record.shared_offset);
				const float row = linear_bfloat16_value(shared[place]);
				shared[place] = (uint16_t)linear_bfloat16_bits(__fadd_rn(value, row));
			} else {
				output[place] = value;
			}
		}
		__syncthreads();
	}
}

extern "C" __global__ void __launch_bounds__(EXL3_TC_COMBINE_THREADS)
	exl3_tc_combine_s2(struct exl3_tc_record record, unsigned char *arena)
{
	tc_combine<2>(record, arena);
}
extern "C" __global__ void __launch_bounds__(EXL3_TC_COMBINE_THREADS)
	exl3_tc_combine_s1(struct exl3_tc_record record, unsigned char *arena)
{
	tc_combine<1>(record, arena);
}
