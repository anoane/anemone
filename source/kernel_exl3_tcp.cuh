/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_exl3_tcp.cuh: the device arithmetic of the routed experts of one DeepSeek-V4.1-Flash EXL3 layer on the tensor
 * cores for prefill-sized calls (65 to 2,048 rows on GNmac's server), one TP rank's intermediate slice: a native
 * rewrite of Morrowmake's `moe_out_prefill_try` (a0c57b5, served by GNmac's variant dsv41_moe_try; bitwise its
 * production `moe_out_prefill`, Curtis's tc-moe 88eb6e1). Routing groups the pairs by local expert into tiles of up to
 * 64 rows, the experts in decreasing row count. Arithmetic (fixed order; a row never depends on another row):
 *   1. gate/up operand: x * suh and H128 in FP32 (the shipped FP32 operations), one 2^e per (row, projection) puts the
 *      row's peak in [2^14, 2^15); hi = f16(v 2^e), lo = f16(v 2^e - hi).
 *   2. products: per k16 step the low products (MMA, C = 0) seed the high MMA; that k16 partial is summed in FP32 in
 *      increasing k; the result times 2^-e.
 *   3. down operand (FP32): H128 of gate and up, * svh, the SwiGLU limit, SiLU(gate) * up, * down suh, H128; split as
 * 1.   4. down products as 2; combine: per route H128 (FP32), * svh, the routes' weighted products summed in FP64 in
 * route      order, rounded once to FP32 (then BF16 for a BF16 output).
 * Codec and helpers: kernel_exl3_tc.cuh.
 */
#ifndef KERNEL_EXL3_TCP_CUH
#define KERNEL_EXL3_TCP_CUH

#include "kernel_exl3_tc.cuh"

#define TCP_TILE_ROWS 64          /* rows of one route tile (GEMM block) */
#define TCP_ROUTE_THREADS 1024
#define TCP_OPERAND_WARPS 8
#define TCP_DOWN_WARPS 2          /* down_operand holds a whole intermediate row a lane group */
#define TCP_GEMM_WARPS 8
#define TCP_GEMM_THREADS (32 * TCP_GEMM_WARPS)
#define TCP_K_CHUNK 64            /* K a pipeline stage */
#define TCP_K_STEPS (TCP_K_CHUNK / TC_MOE_TILE)
#define TCP_STAGES 3
#define TCP_JOINT_ROWS 1024       /* up to here gate and up share an operand launch and a GEMM launch */
#define TCP_CHUNK_ROWS 2048       /* rows of one launch sequence */
#define TCP_MOST_EXPERTS 4096
#define TCP_HALVES 2              /* split_overlap mode 7: a call's halves of rows */
#define TCP_JOINT_DOWN_ROWS 32    /* mode 7: an expert of so many pairs keeps one down tile (one pass of 4 groups) */
#define TCP_H128_SCALE 0.08838834764831845f
#define TCP_SPAN_VALUES 512       /* a warp's span: its four lane groups' H128 blocks */
#define TCP_SPANS_PER_WARP 2      /* an operand row's spans a warp */
#define TCP_GROUP_LANES 8         /* a lane group: l = lane % 8 */
#define TCP_GROUP_SHIFT 3
#define TCP_GROUP_LANE_MASK (TCP_GROUP_LANES - 1)
#define TCP_GROUP_VALUES 16       /* a lane's values of its group's H128 block: v[8 h + j] */
#define TCP_BLOCK_HALVES 2        /* h: an H128 block's halves of 64 */
#define TCP_HALF_BLOCK 64
#define TCP_VECTOR_VALUES 8       /* a lane's consecutive values of a half: one 16-byte vector */
#define TCP_VECTOR_BYTES 16
#define TCP_VECTOR_MASK (TCP_VECTOR_BYTES - 1)
#define TCP_VECTOR_WORDS 4
#define TCP_WORD_VALUES 2         /* 16-bit values a 32-bit word */
#define TCP_HIGH_HALF_MASK 0xffff0000u
/* the route table's int32 words: its counts, sorted[pairs] (16-byte aligned), int4 tiles, positions */
#define TCP_ALIGNED_WORDS 4
#define TCP_TILE_WORDS 4
enum tcp_table_count {
	TCP_TABLE_ROUTED,
	TCP_TABLE_TILES,
	TCP_TABLE_FIRST_TILES,
	TCP_TABLE_FIRST_ROUTED,
	TCP_TABLE_COUNTS
};
/* the index bit of v[8 h + j] a register stage flips: j's three, then h */
enum tcp_register_flip { TCP_FLIP_J0 = 1, TCP_FLIP_J1 = 2, TCP_FLIP_J2 = 4, TCP_FLIP_H = 8 };
/* a 16-byte vector's components */
enum tcp_component { TCP_X, TCP_Y, TCP_Z, TCP_W };
/* the GEMM: an EXL3 tile's trellis bytes, its shapes, the operand stage's rows and copies, the B fragments */
#define TCP_TILE_BYTES_A_BITS_X2 16 /* 256 weights of bits_x2 / 2 bits */
#define TCP_DOWN_PROJECTION 2     /* the projections: gate 0, up 1, down 2 */
#define TCP_WIDE_TILES 2          /* n16 tiles a warp of a wide block */
#define TCP_WIDE_BLOCK_COLUMNS 256
#define TCP_ROW_GROUP 8           /* an 8-row group: an MMA's n8 operand rows */
#define TCP_ROW_GROUP_SHIFT 3
#define TCP_ROW_GROUP_MASK (TCP_ROW_GROUP - 1)
#define TCP_PASS_GROUPS 4         /* two tiles a warp: a pass's most 8-row groups */
#define TCP_ROW_VECTORS 8         /* an operand row's 16-byte vectors of a K chunk */
#define TCP_ROW_VECTOR_SHIFT 3
#define TCP_ROW_VECTOR_MASK (TCP_ROW_VECTORS - 1)
#define TCP_COPY_ROWS 32          /* a stage copy's rows a pass: r0, then r0 + 32 */
#define TCP_COPY_PASSES 2
#define TCP_STEP_VECTORS 2        /* an operand row's 16-byte vectors of a k16 step */
#define TCP_LO_MATRIX_SHIFT 4     /* lane / 16: the lo rows' matrices */
#define TCP_B_BUFFERS 2           /* the B fragments of group g and of g + 1 */
/* a stage's operand rows: [hi, lo][TCP_TILE_ROWS][TCP_K_CHUNK] FP16 */
#define TCP_STAGE_SECOND_ROWS (TCP_COPY_ROWS * TCP_K_CHUNK * TC_MOE_HALFWORD_BYTES)
#define TCP_STAGE_LO_ROWS (TCP_TILE_ROWS * TCP_K_CHUNK * TC_MOE_HALFWORD_BYTES)
#define TCP_STAGE_LO_SECOND_ROWS (TCP_STAGE_LO_ROWS + TCP_STAGE_SECOND_ROWS)
/* an x4 ldmatrix's matrices: a k16 step's B fragments of the hi and lo rows */
enum tcp_b_matrix { TCP_B_HI_K0, TCP_B_HI_K8, TCP_B_LO_K0, TCP_B_LO_K8, TCP_B_MATRICES };
/* a route tile's 8-row groups */
enum tcp_group_count {
	TCP_ONE_GROUP = 1,
	TCP_TWO_GROUPS,
	TCP_THREE_GROUPS,
	TCP_FOUR_GROUPS,
	TCP_FIVE_GROUPS,
	TCP_SIX_GROUPS,
	TCP_SEVEN_GROUPS,
	TCP_EIGHT_GROUPS
};

template <int hidden, int intermediate, int bits_x2> struct tcp_config {
	static constexpr int input_spans = hidden / TCP_SPAN_VALUES;
	static constexpr int middle_spans = intermediate / TCP_SPAN_VALUES;
	static constexpr int middle_tail = (intermediate % TCP_SPAN_VALUES) / TC_MOE_TRANSFORM;
	static_assert(hidden % TCP_SPAN_VALUES == 0 && intermediate % TC_MOE_TRANSFORM == 0,
		      "whole spans and H128 blocks");
};

/* Route tiles a sequence of `pairs` over `experts` local experts can need. */
__host__ __device__ static inline int64_t tcp_tile_capacity(int64_t pairs, int64_t experts)
{
	const int64_t bound = pairs / TCP_TILE_ROWS + (experts < pairs ? experts : pairs) + 1;
	return bound < pairs ? bound : pairs;
}
/* The route table: [0] routed count, [1] tile count, [2] the first half's tiles and [3] its routed count (halves; else
 * [1] and [0]); sorted[pairs]; tiles[tile capacity] int4 {route expert (an expert, the second half's plus the experts),
 * first sorted position, rows, 0} (16-byte aligned); position[pairs] (the pair's sorted position, -1: not local). Int32
 * words; the layout's experts are EXL3_TCP_ROUTE_EXPERTS whatever a route's. */
__host__ __device__ static inline int64_t tcp_tiles_offset(int64_t pairs)
{
	return TCP_TABLE_COUNTS + (pairs + TCP_ALIGNED_WORDS - 1) / TCP_ALIGNED_WORDS * TCP_ALIGNED_WORDS;
}
__host__ __device__ static inline int64_t tcp_position_offset(int64_t pairs, int64_t experts)
{
	return tcp_tiles_offset(pairs) + TCP_TILE_WORDS * tcp_tile_capacity(pairs, experts);
}
__host__ __device__ static inline int64_t tcp_table_words(int64_t pairs, int64_t experts)
{
	return tcp_position_offset(pairs, experts) + pairs;
}

/* ---- the route: one block; experts by decreasing row count (ties by index), an expert's tiles adjacent ----
 * expert_of(pair): the pair's route expert (below experts) or -1. With halves 2 the experts are the first half's
 * (experts / 2 of them) then the second's: each half's ranked apart, the first half's tiles first. Shared memory:
 * 5 experts + 1 ints. */
template <typename expert_of_type>
__device__ __forceinline__ void tcp_route(int64_t pairs, int experts, int halves, expert_of_type expert_of,
					  int32_t *table)
{
	const int half_experts = experts / halves;
	extern __shared__ int tcp_route_shared[];
	int *cursor = tcp_route_shared, *row_begin = cursor + experts, *tile_begin = row_begin + experts;
	int *slot_begin = tile_begin + experts + 1, *by_rank = slot_begin + experts;
	for (int expert = threadIdx.x; expert < experts; expert += blockDim.x)
		cursor[expert] = 0;
	__syncthreads();
	for (int64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
		const int expert = expert_of(pair);
		if (expert >= 0)
			atomicAdd(&cursor[expert], 1);
	}
	__syncthreads();
	const int per_thread = (experts + blockDim.x - 1) / blockDim.x;
	const int first_expert = min(experts, (int)threadIdx.x * per_thread);
	const int end_expert = min(experts, first_expert + per_thread);
	int rows = 0, tiles = 0;
	for (int expert = first_expert; expert < end_expert; expert++) {
		rows += cursor[expert];
		tiles += (cursor[expert] + TCP_TILE_ROWS - 1) / TCP_TILE_ROWS;
	}
	int row_at, tile_at, row_total, tile_total;
	tc_moe_scan_pair(rows, tiles, row_at, tile_at, row_total, tile_total);
	for (int expert = first_expert; expert < end_expert; expert++) {
		const int count = cursor[expert];
		row_begin[expert] = row_at;
		tile_begin[expert] = tile_at;
		cursor[expert] = row_at;
		row_at += count;
		tile_at += (count + TCP_TILE_ROWS - 1) / TCP_TILE_ROWS;
	}
	if (threadIdx.x == 0) {
		tile_begin[experts] = tile_total;
		table[TCP_TABLE_ROUTED] = row_total;
		table[TCP_TABLE_TILES] = tile_total;
	}
	__syncthreads();
	if (threadIdx.x == 0) { /* the first half's tiles and rows: its experts come first in index order */
		table[TCP_TABLE_FIRST_TILES] = halves > 1 ? tile_begin[half_experts] : tile_total;
		table[TCP_TABLE_FIRST_ROUTED] = halves > 1 ? row_begin[half_experts] : row_total;
	}
	for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
		const int rows_here = (expert + 1 < experts ? row_begin[expert + 1] : row_total) - row_begin[expert];
		const int half = expert / half_experts;
		int rank = half * half_experts;
		for (int other = half * half_experts; other < (half + 1) * half_experts; other++) {
			const int rows_other = (other + 1 < experts ?
							row_begin[other + 1] :
							row_total) -
					       row_begin[other];
			rank += rows_other > rows_here || (rows_other == rows_here && other < expert);
		}
		by_rank[rank] = expert;
	}
	__syncthreads();
	int tiles_here = 0;
	for (int rank = first_expert; rank < end_expert; rank++)
		tiles_here += tile_begin[by_rank[rank] + 1] - tile_begin[by_rank[rank]];
	int slot_at, unused_at, slot_total, unused_total;
	tc_moe_scan_pair(tiles_here, 0, slot_at, unused_at, slot_total, unused_total);
	for (int rank = first_expert; rank < end_expert; rank++) {
		const int expert = by_rank[rank];
		slot_begin[expert] = slot_at;
		slot_at += tile_begin[expert + 1] - tile_begin[expert];
	}
	__syncthreads();
	int4 *tile_table = (int4 *)(table + tcp_tiles_offset(pairs));
	for (int tile = threadIdx.x; tile < tile_total; tile += blockDim.x) {
		int low = 0, high = experts;
		while (high - low > 1) {
			const int middle = (low + high) >> 1;
			if (tile_begin[middle] <= tile)
				low = middle;
			else
				high = middle;
		}
		const int first = row_begin[low] + (tile - tile_begin[low]) * TCP_TILE_ROWS;
		const int end = low + 1 < experts ? row_begin[low + 1] : row_total;
		tile_table[slot_begin[low] + tile - tile_begin[low]] =
			make_int4(low, first, min(TCP_TILE_ROWS, end - first),
				  0);
	}
	int32_t *sorted = table + TCP_TABLE_COUNTS,
		*position = table +
			    tcp_position_offset(pairs, EXL3_TCP_ROUTE_EXPERTS);
	for (int64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
		const int expert = expert_of(pair);
		int at = -1;
		if (expert >= 0) {
			at = atomicAdd(&cursor[expert], 1);
			sorted[at] = (int32_t)pair;
		}
		position[pair] = at;
	}
}

/* ---- split_overlap mode 7's route (half_rows): route expert x = 2 e + h, so expert e's first-half pairs, then its
 * second-half pairs, take adjacent sorted positions (x in index order) ----
 * retile false: the gate/up tiles, 64 rows of an expert's positions over both halves (its weights decoded once a
 * tile, as the joint call's), experts by decreasing rows; sorted and position written; table[2] and [3] the totals.
 * retile true (behind the gate/up GEMMs, the same picks): the same counts and positions, the down tiles: an expert of
 * at most TCP_JOINT_DOWN_ROWS pairs one tile over both halves in the first part (the two-tile down form runs it in
 * one pass), any other each half's tiles (route expert e + experts h: tcp_route_expert's), the first half's in the
 * first part (no more passes than its joint tiles took); each part's by decreasing rows; table[1] and [2]
 * rewritten. Shared memory: 5 routes + 2 ints. */
template <bool retile, typename expert_of_type>
__device__ __forceinline__ void tcp_route_paired(int64_t pairs, expert_of_type expert_of, int32_t *table)
{
	constexpr int routes = EXL3_TCP_ROUTE_EXPERTS,
		      experts = EXL3_TCP_EXPERTS;
	constexpr int routes_a_unit = retile ? 1 : TCP_HALVES, units = routes / routes_a_unit;
	extern __shared__ int tcp_route_shared[];
	int *cursor = tcp_route_shared, *route_begin = cursor + routes, *tile_begin = route_begin + routes + 1;
	int *slot_begin = tile_begin + units + 1, *by_rank = slot_begin + units;
	for (int route = threadIdx.x; route < routes; route += blockDim.x)
		cursor[route] = 0;
	__syncthreads();
	for (int64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
		const int route = expert_of(pair);
		if (route >= 0)
			atomicAdd(&cursor[route], 1);
	}
	__syncthreads();
	/* the positions: route experts in index order */
	const int per_thread = (routes + blockDim.x - 1) / blockDim.x;
	const int first_route = min(routes, (int)threadIdx.x * per_thread);
	const int end_route = min(routes, first_route + per_thread);
	int rows = 0;
	for (int route = first_route; route < end_route; route++)
		rows += cursor[route];
	int row_at, unused_at, row_total, unused_total;
	tc_moe_scan_pair(rows, 0, row_at, unused_at, row_total, unused_total);
	for (int route = first_route; route < end_route; route++) {
		const int count = cursor[route];
		route_begin[route] = row_at;
		cursor[route] = row_at;
		row_at += count;
	}
	if (threadIdx.x == 0)
		route_begin[routes] = row_total;
	__syncthreads();
	/* a unit's positions: an expert's (route), or (retile) a half's or a joint tile's */
	auto unit_range = [&](int unit, int &first, int &end) {
		if (!retile) {
			first = route_begin[TCP_HALVES * unit];
			end = route_begin[TCP_HALVES * (unit + 1)];
			return;
		}
		const int expert_first = route_begin[unit - unit % TCP_HALVES];
		const int expert_end = route_begin[unit - unit % TCP_HALVES + TCP_HALVES];
		const bool joint = expert_end - expert_first <= TCP_JOINT_DOWN_ROWS;
		first = joint ? (unit % TCP_HALVES == 0 ? expert_first : expert_end) : route_begin[unit];
		end = joint ? expert_end : route_begin[unit + 1];
	};
	const int unit_per_thread = (units + blockDim.x - 1) / blockDim.x;
	const int first_unit = min(units, (int)threadIdx.x * unit_per_thread);
	const int end_unit = min(units, first_unit + unit_per_thread);
	int tiles = 0;
	for (int unit = first_unit; unit < end_unit; unit++) {
		int first, end;
		unit_range(unit, first, end);
		tiles += (end - first + TCP_TILE_ROWS - 1) / TCP_TILE_ROWS;
	}
	int tile_at, tile_total;
	tc_moe_scan_pair(tiles, 0, tile_at, unused_at, tile_total, unused_total);
	for (int unit = first_unit; unit < end_unit; unit++) {
		int first, end;
		unit_range(unit, first, end);
		tile_begin[unit] = tile_at;
		tile_at += (end - first + TCP_TILE_ROWS - 1) / TCP_TILE_ROWS;
	}
	if (threadIdx.x == 0) {
		tile_begin[units] = tile_total;
		table[TCP_TABLE_TILES] = tile_total;
		if (!retile) {
			table[TCP_TABLE_ROUTED] = row_total;
			table[TCP_TABLE_FIRST_TILES] = tile_total;
			table[TCP_TABLE_FIRST_ROUTED] = row_total;
		}
	}
	__syncthreads();
	/* ranks: retile ranks each part's units apart (even units first), the first part's ranks first */
	for (int unit = threadIdx.x; unit < units; unit += blockDim.x) {
		const int part = retile ? unit % TCP_HALVES : 0, step = retile ? TCP_HALVES : 1;
		int first, end;
		unit_range(unit, first, end);
		const int rows_here = end - first;
		int rank = part * (units / TCP_HALVES);
		for (int other = part; other < units; other += step) {
			int other_first, other_end;
			unit_range(other, other_first, other_end);
			const int rows_other = other_end - other_first;
			rank += rows_other > rows_here || (rows_other == rows_here && other < unit);
		}
		by_rank[rank] = unit;
	}
	__syncthreads();
	int tiles_here = 0;
	for (int rank = first_unit; rank < end_unit; rank++)
		tiles_here += tile_begin[by_rank[rank] + 1] - tile_begin[by_rank[rank]];
	int slot_at, slot_total;
	tc_moe_scan_pair(tiles_here, 0, slot_at, unused_at, slot_total, unused_total);
	for (int rank = first_unit; rank < end_unit; rank++) {
		const int unit = by_rank[rank];
		slot_begin[unit] = slot_at;
		slot_at += tile_begin[unit + 1] - tile_begin[unit];
	}
	__syncthreads();
	if (retile && threadIdx.x == 0)
		table[TCP_TABLE_FIRST_TILES] = slot_begin[by_rank[units / TCP_HALVES]]; /* the first part's tiles */
	int4 *tile_table = (int4 *)(table + tcp_tiles_offset(pairs));
	for (int tile = threadIdx.x; tile < tile_total; tile += blockDim.x) {
		int low = 0, high = units;
		while (high - low > 1) {
			const int middle = (low + high) >> 1;
			if (tile_begin[middle] <= tile)
				low = middle;
			else
				high = middle;
		}
		int unit_first, unit_end;
		unit_range(low, unit_first, unit_end);
		const int first = unit_first + (tile - tile_begin[low]) * TCP_TILE_ROWS;
		const int expert = retile ? low / TCP_HALVES + low % TCP_HALVES * experts : low;
		tile_table[slot_begin[low] + tile - tile_begin[low]] =
			make_int4(expert, first, min(TCP_TILE_ROWS, unit_end - first), 0);
	}
	if (retile)
		return;
	int32_t *sorted = table + TCP_TABLE_COUNTS, *position = table + tcp_position_offset(pairs, routes);
	for (int64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
		const int route = expert_of(pair);
		int at = -1;
		if (route >= 0) {
			at = atomicAdd(&cursor[route], 1);
			sorted[at] = (int32_t)pair;
		}
		position[pair] = at;
	}
}

/* ---- FP32 helpers ---- */

__device__ __forceinline__ float tcp_power_of_two(int exponent)
{
	return __uint_as_float((uint32_t)(TC_MOE_FLOAT_BIAS + exponent) << TC_MOE_MANTISSA_BITS);
}
/* H128 of a lane group's 128 values (l = lane % 8: v[8 h + j] is element 64 h + 8 l + j): the per-element operations
 * and stage order of the warp form (distances 1..64 increasing, lower v + other, upper other - v, then the scale);
 * distances 1, 2, 4 and 64 in registers, 8, 16, 32 by shuffles inside the group. Every lane of the warp calls it. */
__device__ __forceinline__ void tcp_register_stage(float (&value)[TCP_GROUP_VALUES], int distance)
{
	float next[TCP_GROUP_VALUES];
#pragma unroll
	for (int index = 0; index < TCP_GROUP_VALUES; index++) {
		const float other = value[index ^ distance];
		next[index] = (index & distance) ? __fsub_rn(other, value[index]) : __fadd_rn(value[index], other);
	}
#pragma unroll
	for (int index = 0; index < TCP_GROUP_VALUES; index++)
		value[index] = next[index];
}
__device__ __forceinline__ void tcp_hadamard_group(float (&value)[TCP_GROUP_VALUES], int lane)
{
	tcp_register_stage(value, TCP_FLIP_J0);
	tcp_register_stage(value, TCP_FLIP_J1);
	tcp_register_stage(value, TCP_FLIP_J2);
#pragma unroll
	for (int distance = 1; distance < TCP_GROUP_LANES; distance <<= 1) {
		const bool upper = (lane & distance) != 0;
#pragma unroll
		for (int index = 0; index < TCP_GROUP_VALUES; index++) {
			const float other = __shfl_xor_sync(TC_MOE_FULL_MASK, value[index], distance);
			value[index] = upper ? __fsub_rn(other, value[index]) : __fadd_rn(value[index], other);
		}
	}
	tcp_register_stage(value, TCP_FLIP_H);
#pragma unroll
	for (int index = 0; index < TCP_GROUP_VALUES; index++)
		value[index] = __fmul_rn(value[index], TCP_H128_SCALE);
}
/* H128 of a warp's 128 values (lane holds lane + 32 i) in FP32. */
__device__ __forceinline__ void tcp_hadamard_warp(float (&value)[TC_MOE_LANE_VALUES], int lane)
{
#pragma unroll
	for (int distance = 1; distance < TC_MOE_WARP; distance <<= 1) {
		const bool upper = (lane & distance) != 0;
#pragma unroll
		for (int index = 0; index < TC_MOE_LANE_VALUES; index++) {
			const float other = __shfl_xor_sync(TC_MOE_FULL_MASK, value[index], distance);
			value[index] = upper ? __fsub_rn(other, value[index]) : __fadd_rn(value[index], other);
		}
	}
#pragma unroll
	for (int distance = 1; distance < TC_MOE_LANE_VALUES; distance <<= 1) {
		float next[TC_MOE_LANE_VALUES];
#pragma unroll
		for (int index = 0; index < TC_MOE_LANE_VALUES; index++) {
			const float other = value[index ^ distance];
			next[index] = (index & distance) ?
					      __fsub_rn(other, value[index]) :
					      __fadd_rn(value[index], other);
		}
#pragma unroll
		for (int index = 0; index < TC_MOE_LANE_VALUES; index++)
			value[index] = next[index];
	}
#pragma unroll
	for (int index = 0; index < TC_MOE_LANE_VALUES; index++)
		value[index] = __fmul_rn(value[index], TCP_H128_SCALE);
}
__device__ __forceinline__ bool tcp_aligned(const void *pointer)
{
	return ((uintptr_t)pointer & TCP_VECTOR_MASK) == 0;
}
__device__ __forceinline__ void tcp_load8(const __nv_bfloat16 *source, bool vector, float *values)
{
	if (vector) {
		const uint4 loaded = *(const uint4 *)source;
		const uint32_t words[TCP_VECTOR_WORDS] = { loaded.x, loaded.y, loaded.z, loaded.w };
#pragma unroll
		for (int word = 0; word < TCP_VECTOR_WORDS; word++) {
			values[TCP_WORD_VALUES * word] = __uint_as_float(words[word] << TC_MOE_HALFWORD_BITS);
			values[TCP_WORD_VALUES * word + 1] = __uint_as_float(words[word] & TCP_HIGH_HALF_MASK);
		}
	} else {
#pragma unroll
		for (int index = 0; index < TCP_VECTOR_VALUES; index++)
			values[index] = __bfloat162float(source[index]);
	}
}
__device__ __forceinline__ void tcp_load8(const __half *source, bool vector, float *values)
{
	if (vector) {
		const uint4 loaded = *(const uint4 *)source;
		const uint32_t words[TCP_VECTOR_WORDS] = { loaded.x, loaded.y, loaded.z, loaded.w };
#pragma unroll
		for (int word = 0; word < TCP_VECTOR_WORDS; word++) {
			const float2 pair = __half22float2(*(const __half2 *)&words[word]);
			values[TCP_WORD_VALUES * word] = pair.x;
			values[TCP_WORD_VALUES * word + 1] = pair.y;
		}
	} else {
#pragma unroll
		for (int index = 0; index < TCP_VECTOR_VALUES; index++)
			values[index] = __half2float(source[index]);
	}
}
__device__ __forceinline__ void tcp_load8(const float *source, float *values)
{
	const float4 first = *(const float4 *)source, second = *(const float4 *)(source + TCP_VECTOR_WORDS);
	values[TCP_X] = first.x;
	values[TCP_Y] = first.y;
	values[TCP_Z] = first.z;
	values[TCP_W] = first.w;
	values[TCP_VECTOR_WORDS + TCP_X] = second.x;
	values[TCP_VECTOR_WORDS + TCP_Y] = second.y;
	values[TCP_VECTOR_WORDS + TCP_Z] = second.z;
	values[TCP_VECTOR_WORDS + TCP_W] = second.w;
}
template <int count> __device__ __forceinline__ float tcp_lane_peak(const float (&value)[count])
{
	float peak = 0.0f;
#pragma unroll
	for (int index = 0; index < count; index++)
		peak = fmaxf(peak, fabsf(value[index]));
	return peak;
}
__device__ __forceinline__ uint32_t tcp_pack(__half first, __half second)
{
	return (uint32_t)__half_as_ushort(first) | ((uint32_t)__half_as_ushort(second) << TC_MOE_HALFWORD_BITS);
}
/* Eight consecutive values y = v 2^e: the FP16 high halves at high, the FP16-rounded residuals at low. */
__device__ __forceinline__ void tcp_store8_split(const float *value, float scale, __half *high, __half *low)
{
	uint32_t high_words[TCP_VECTOR_WORDS], low_words[TCP_VECTOR_WORDS];
#pragma unroll
	for (int word = 0; word < TCP_VECTOR_WORDS; word++) {
		const float first = __fmul_rn(value[TCP_WORD_VALUES * word], scale),
			    second = __fmul_rn(value[TCP_WORD_VALUES * word + 1], scale);
		const __half first_high = __float2half_rn(first), second_high = __float2half_rn(second);
		high_words[word] = tcp_pack(first_high, second_high);
		low_words[word] = tcp_pack(__float2half_rn(__fsub_rn(first, __half2float(first_high))),
					   __float2half_rn(__fsub_rn(second, __half2float(second_high))));
	}
	*(uint4 *)high = make_uint4(high_words[TCP_X], high_words[TCP_Y], high_words[TCP_Z], high_words[TCP_W]);
	*(uint4 *)low = make_uint4(low_words[TCP_X], low_words[TCP_Y], low_words[TCP_Z], low_words[TCP_W]);
}

/* ---- 1. gate/up operand: one block a task (sorted position, projection), warp w spans 2 w and 2 w + 1 ---- */
template <int hidden>
__device__ __forceinline__ void tcp_input_span(const __nv_bfloat16 *row, const __half *suh, bool vector, int span,
					       int lane_column, int lane, float (&value)[TCP_GROUP_VALUES])
{
#pragma unroll
	for (int half = 0; half < TCP_BLOCK_HALVES; half++) {
		const int column = TCP_SPAN_VALUES * span + TCP_HALF_BLOCK * half + lane_column;
		float inputs[TCP_VECTOR_VALUES], scales[TCP_VECTOR_VALUES];
		tcp_load8(row + column, vector, inputs);
		tcp_load8(suh + column, vector, scales);
#pragma unroll
		for (int index = 0; index < TCP_VECTOR_VALUES; index++)
			value[TCP_VECTOR_VALUES * half + index] = __fmul_rn(inputs[index], scales[index]);
	}
	tcp_hadamard_group(value, lane);
}

/* The row's split operand (K halves high, then K halves low) at operand and its 2^-e; every thread of the block calls
 * it. */
template <int hidden>
__device__ __forceinline__ void tcp_operand_row(const __nv_bfloat16 *row, const __half *suh, __half *operand,
						float *unscale)
{
	constexpr int spans_per_warp = TCP_SPANS_PER_WARP, warps = hidden / TCP_SPAN_VALUES / spans_per_warp;
	__shared__ float warp_peak[warps];
	const int lane = threadIdx.x & TC_MOE_LANE_MASK, warp = threadIdx.x >> TC_MOE_WARP_SHIFT;
	const int lane_column =
		TC_MOE_TRANSFORM * (lane >> TCP_GROUP_SHIFT) + TCP_VECTOR_VALUES * (lane & TCP_GROUP_LANE_MASK);
	const bool vector = tcp_aligned(row) && tcp_aligned(suh);
	float value[spans_per_warp][TCP_GROUP_VALUES];
	float peak = 0.0f;
#pragma unroll
	for (int span = 0; span < spans_per_warp; span++) {
		tcp_input_span<hidden>(row, suh, vector, spans_per_warp * warp + span, lane_column, lane, value[span]);
		peak = fmaxf(peak, tcp_lane_peak(value[span]));
	}
#pragma unroll
	for (int distance = TC_MOE_HALF_WARP; distance >= 1; distance >>= 1)
		peak = fmaxf(peak, __shfl_xor_sync(TC_MOE_FULL_MASK, peak, distance));
	if (lane == 0)
		warp_peak[warp] = peak;
	__syncthreads();
#pragma unroll
	for (int other = 0; other < warps; other++)
		peak = fmaxf(peak, warp_peak[other]);
	__syncthreads(); /* warp_peak is rewritten by the next task */
	const int exponent = tc_moe_exponent(peak);
	const float scale = tcp_power_of_two(exponent);
#pragma unroll
	for (int span = 0; span < spans_per_warp; span++) {
#pragma unroll
		for (int half = 0; half < TCP_BLOCK_HALVES; half++) {
			const int column =
				TCP_SPAN_VALUES * (spans_per_warp * warp + span) + TCP_HALF_BLOCK * half + lane_column;
			tcp_store8_split(&value[span][TCP_VECTOR_VALUES * half], scale, operand + column,
					 operand + hidden + column);
		}
	}
	if (threadIdx.x == 0)
		*unscale = tcp_power_of_two(-exponent);
}

/* ---- 3. down operand: one warp a sorted position ---- */
__device__ __forceinline__ float tcp_down_input(float gate, float up, float gate_svh, float up_svh, float down_suh,
						float limit)
{
	float gated = __fmul_rn(gate, gate_svh), upped = __fmul_rn(up, up_svh);
	if (limit > 0.0f) {
		gated = fminf(gated, limit);
		upped = fminf(fmaxf(upped, -limit), limit);
	}
	const float silu = gated / __fadd_rn(1.0f, expf(-gated));
	return __fmul_rn(__fmul_rn(silu, upped), down_suh);
}

/* The position's gate and up output rows (FP32) become its split down operand row (2 I halves, in place of its gate
 * row) and its 2^-e. Spans of 512 in the lane-group layout, the tail blocks in the warp layout. mixed
 * (--exl3_expert_rotation): a narrow expert (128 narrower) lacks the last tail block, neither loaded (its scales end
 * before it) nor stored, zeros in the peak (a zero never raises a peak of magnitudes): a call's at its width. */
template <int intermediate, bool mixed = false>
__device__ __forceinline__ void tcp_down_row(float *gate_row, const float *up_row, const __half *gate_svh,
					     const __half *up_svh, const __half *down_suh, float limit, float *unscale,
					     bool narrow = false)
{
	constexpr int full = intermediate / TCP_SPAN_VALUES, tail = (intermediate % TCP_SPAN_VALUES) / TC_MOE_TRANSFORM,
		      tail_base = TCP_SPAN_VALUES * full;
	static_assert(!mixed || tail > 0, "a narrow expert drops the last tail block");
	const bool short_tail = mixed && narrow; /* uniform a warp */
	const int lane = threadIdx.x & TC_MOE_LANE_MASK;
	const int lane_column =
		TC_MOE_TRANSFORM * (lane >> TCP_GROUP_SHIFT) + TCP_VECTOR_VALUES * (lane & TCP_GROUP_LANE_MASK);
	const bool vector = tcp_aligned(gate_svh) && tcp_aligned(up_svh) && tcp_aligned(down_suh);
	float value[full > 0 ? full : 1][TCP_GROUP_VALUES];
#pragma unroll
	for (int span = 0; span < full; span++) {
		float gate[TCP_GROUP_VALUES], up[TCP_GROUP_VALUES];
#pragma unroll
		for (int half = 0; half < TCP_BLOCK_HALVES; half++) {
			tcp_load8(gate_row + TCP_SPAN_VALUES * span + lane_column + TCP_HALF_BLOCK * half,
				  &gate[TCP_VECTOR_VALUES * half]);
			tcp_load8(up_row + TCP_SPAN_VALUES * span + lane_column + TCP_HALF_BLOCK * half,
				  &up[TCP_VECTOR_VALUES * half]);
		}
		tcp_hadamard_group(gate, lane);
		tcp_hadamard_group(up, lane);
#pragma unroll
		for (int half = 0; half < TCP_BLOCK_HALVES; half++) {
			const int column = TCP_SPAN_VALUES * span + lane_column + TCP_HALF_BLOCK * half;
			float gate_scales[TCP_VECTOR_VALUES], up_scales[TCP_VECTOR_VALUES],
				down_scales[TCP_VECTOR_VALUES];
			tcp_load8(gate_svh + column, vector, gate_scales);
			tcp_load8(up_svh + column, vector, up_scales);
			tcp_load8(down_suh + column, vector, down_scales);
#pragma unroll
			for (int index = 0; index < TCP_VECTOR_VALUES; index++)
				value[span][TCP_VECTOR_VALUES * half + index] =
					tcp_down_input(gate[TCP_VECTOR_VALUES * half + index],
						       up[TCP_VECTOR_VALUES * half + index], gate_scales[index],
						       up_scales[index], down_scales[index], limit);
		}
		tcp_hadamard_group(value[span], lane);
	}
	float tail_value[tail > 0 ? tail : 1][TC_MOE_LANE_VALUES];
#pragma unroll
	for (int block = 0; block < tail; block++) {
		if (short_tail && block == tail - 1) {
#pragma unroll
			for (int index = 0; index < TC_MOE_LANE_VALUES; index++)
				tail_value[block][index] = 0.0f;
			continue;
		}
		const int base = tail_base + TC_MOE_TRANSFORM * block;
		float gate[TC_MOE_LANE_VALUES], up[TC_MOE_LANE_VALUES];
#pragma unroll
		for (int index = 0; index < TC_MOE_LANE_VALUES; index++) {
			gate[index] = gate_row[base + lane + TC_MOE_WARP * index];
			up[index] = up_row[base + lane + TC_MOE_WARP * index];
		}
		tcp_hadamard_warp(gate, lane);
		tcp_hadamard_warp(up, lane);
#pragma unroll
		for (int index = 0; index < TC_MOE_LANE_VALUES; index++) {
			const int column = base + lane + TC_MOE_WARP * index;
			tail_value[block][index] = tcp_down_input(
				gate[index], up[index],
				__half2float(gate_svh[column]),
				__half2float(up_svh[column]),
				__half2float(down_suh[column]), limit);
		}
		tcp_hadamard_warp(tail_value[block], lane);
	}
	float peak = 0.0f;
#pragma unroll
	for (int span = 0; span < full; span++)
		peak = fmaxf(peak, tcp_lane_peak(value[span]));
#pragma unroll
	for (int block = 0; block < tail; block++)
		peak = fmaxf(peak, tcp_lane_peak(tail_value[block]));
#pragma unroll
	for (int distance = TC_MOE_HALF_WARP; distance >= 1; distance >>= 1)
		peak = fmaxf(peak, __shfl_xor_sync(TC_MOE_FULL_MASK, peak, distance));
	const int exponent = tc_moe_exponent(peak);
	const float scale = tcp_power_of_two(exponent);
	__half *out = (__half *)gate_row;
#pragma unroll
	for (int span = 0; span < full; span++) {
#pragma unroll
		for (int half = 0; half < TCP_BLOCK_HALVES; half++) {
			const int column = TCP_SPAN_VALUES * span + lane_column + TCP_HALF_BLOCK * half;
			tcp_store8_split(&value[span][TCP_VECTOR_VALUES * half], scale, out + column,
					 out + intermediate + column);
		}
	}
#pragma unroll
	for (int block = 0; block < tail; block++) {
		if (short_tail && block == tail - 1)
			continue;
#pragma unroll
		for (int index = 0; index < TC_MOE_LANE_VALUES; index++) {
			const int k = tail_base + TC_MOE_TRANSFORM * block + lane + TC_MOE_WARP * index;
			const float scaled = __fmul_rn(tail_value[block][index], scale);
			const __half high = __float2half_rn(scaled);
			out[k] = high;
			out[intermediate + k] = __float2half_rn(__fsub_rn(scaled, __half2float(high)));
		}
	}
	if (lane == 0)
		*unscale = tcp_power_of_two(-exponent);
}

/* ---- 2 and 4: the expert GEMM of one block item (route tile, projection, column block) ----
 * A stage: the tile's hi/lo operand rows (a row 64 halves in eight 16-byte chunks, chunk c at c ^ (row & 7)) at 0, then
 * the block's adjacent trellis tiles of each k16 step at stage_weights. mixed (--exl3_expert_rotation): a tile's expert
 * is N (gate/up) or K (down) wide or 128 narrower; a narrow gate/up expert's trellis rows hold its own column tiles and
 * its column blocks past its width exit, a narrow down expert runs its own K chunks (the operand rows keep K's layout).
 */
template <int K, int N, int projections, int first_projection, int tiles_per_warp, int bits_x2,
	  bool mixed_widths = false, bool partial_columns = false, bool two_blocks = false>
struct tcp_shape {
	static constexpr int k = K, n = N, projection_count = projections, first = first_projection;
	static constexpr int tpw = tiles_per_warp, bits = bits_x2;
	static constexpr bool mixed = mixed_widths;
	/* two_blocks (--exl3_tcp_two_blocks): the kernel at two blocks a multiprocessor (128 registers), its k16 steps
	 * of a K chunk in a loop rather than unrolled (no spill): each output's MMA and add order unchanged */
	static constexpr bool compact_steps = two_blocks;
	static constexpr int tile_bytes = TCP_TILE_BYTES_A_BITS_X2 * bits_x2;
	static constexpr int block_tiles = TCP_GEMM_WARPS * tiles_per_warp;
	static constexpr int chunks = K / TCP_K_CHUNK;
	static constexpr int column_blocks = N / (TC_MOE_TILE * block_tiles); /* partial_columns: the whole blocks */
	static constexpr int tiles_n = N / TC_MOE_TILE;
	static constexpr int64_t row_halves = (int64_t)projections * TC_MOE_HI_LO_ROWS * K;
	static constexpr int64_t weight_chunk = (int64_t)TCP_K_STEPS * tiles_n * tile_bytes;
	static constexpr uint32_t stage_weights =
		TC_MOE_HI_LO_ROWS * TCP_TILE_ROWS * TCP_K_CHUNK * TC_MOE_HALFWORD_BYTES;
	static constexpr uint32_t stage_bytes = stage_weights + TCP_K_STEPS * block_tiles * tile_bytes;
	static constexpr int shared_bytes = TCP_STAGES * stage_bytes;
	static_assert(K % TCP_K_CHUNK == 0 && (partial_columns || N % (TC_MOE_TILE * block_tiles) == 0) &&
			      chunks >= TCP_STAGES - 1,
		      "GEMM shape");
	static_assert(stage_bytes % TCP_VECTOR_BYTES == 0 && tile_bytes % TCP_VECTOR_BYTES == 0,
		      "16-byte aligned stages and tiles");
	static_assert(!mixed_widths ||
			      (first_projection < TCP_DOWN_PROJECTION ?
				       (N - TC_MOE_TRANSFORM) % (TC_MOE_TILE * block_tiles) == 0 :
						      (K - TC_MOE_TRANSFORM) % TCP_K_CHUNK == 0 &&
							      (K - TC_MOE_TRANSFORM) / TCP_K_CHUNK >= TCP_STAGES - 1),
		      "mixed-width GEMM shape");
};
/* two n16 tiles a warp where the intermediate allows (256-column blocks), one at 640 */
template <int hidden, int intermediate, int bits_x2, int projections, int first, bool mixed = false,
	  bool two_blocks = false>
using tcp_gate_up_shape = tcp_shape<hidden, intermediate, projections, first,
				    (intermediate % TCP_WIDE_BLOCK_COLUMNS == 0 ? TCP_WIDE_TILES : 1), bits_x2,
				    mixed, false, two_blocks>;
template <int hidden, int intermediate, int bits_x2, bool mixed = false, bool two_blocks = false>
using tcp_down_shape =
	tcp_shape<intermediate, hidden, 1, TCP_DOWN_PROJECTION, TCP_WIDE_TILES, bits_x2, mixed, false, two_blocks>;

struct tcp_gemm {
	const int32_t *table;
	int64_t pairs;            /* the route table's capacity */
	int64_t experts;
	int64_t first_tile, end_tile; /* the route tiles the launch takes (a part's: the route table's) */
	const __half *operand;    /* [position][projection][hi, lo][k] */
	const float *unscale;     /* [position][projection] */
	float *out;               /* [projection][position][n] */
	int64_t out_projection;   /* FP32 elements between the projections' outputs */
};

template <typename shape>
__device__ __forceinline__ uint32_t tcp_next_stage(uint32_t offset)
{
	return offset + shape::stage_bytes == TCP_STAGES * shape::stage_bytes ? 0u : offset + shape::stage_bytes;
}
__device__ __forceinline__ void tcp_ldmatrix4(uint32_t address, uint32_t (&registers)[TCP_B_MATRICES])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
		     : "=r"(registers[TCP_B_HI_K0]), "=r"(registers[TCP_B_HI_K8]), "=r"(registers[TCP_B_LO_K0]),
		       "=r"(registers[TCP_B_LO_K8])
		     : "r"(address));
}
__device__ __forceinline__ uint32_t tcp_lds32(uint32_t address)
{
	uint32_t value;
	asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address));
	return value;
}
__device__ __forceinline__ void tcp_copy16(uint32_t destination, const void *source, int source_bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(
			     destination),
		     "l"(source), "r"(source_bytes));
}
__device__ __forceinline__ void tcp_commit()
{
	asm volatile("cp.async.commit_group;");
}
template <int pending> __device__ __forceinline__ void tcp_wait()
{
	asm volatile("cp.async.wait_group %0;" ::"n"(pending));
}
__device__ __forceinline__ void tcp_mma(const uint32_t (&a)[TC_MOE_FRAGMENT_WORDS], uint32_t b0, uint32_t b1,
					const float (&c)[TC_MOE_MMA_RESULTS], float (&d)[TC_MOE_MMA_RESULTS])
{
	asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
	    "{%10, %11, %12, %13};"
	    : "=f"(d[0]), "=f"(d[1]), "=f"(d[TC_MOE_LANE_COLUMNS]), "=f"(d[TC_MOE_LANE_COLUMNS + 1])
	    : "r"(a[TC_MOE_A0]), "r"(a[TC_MOE_A1]), "r"(a[TC_MOE_A2]), "r"(a[TC_MOE_A3]), "r"(b0), "r"(b1), "f"(c[0]),
	      "f"(c[1]), "f"(c[TC_MOE_LANE_COLUMNS]), "f"(c[TC_MOE_LANE_COLUMNS + 1]));
}

/* The K loop and epilogue for `groups` 8-row groups of a tile from group first_group: the ldmatrix of group g + 1
 * before the MMAs of group g; each B fragment serves the warp's tiles_per_warp n16 tiles. `load` issues the next
 * chunk's copies. Groups are independent rows, each summed in increasing k: two passes over a tile's groups give the
 * bits of one. */
template <int groups, int first_group, typename shape, typename load_type>
__device__ __forceinline__ void tcp_run_tile(const struct tcp_gemm &p, uint32_t shared, load_type &load, int rows,
					     int first, int projection_index, int column_block, int tile_chunks)
{
	const int chunks = shape::mixed ? tile_chunks : shape::chunks; /* a mixed down tile's own K chunks */
	constexpr int tpw = shape::tpw;
	typedef tc_moe_ring<shape::bits> ring;
	const int lane = threadIdx.x & TC_MOE_LANE_MASK, warp = threadIdx.x >> TC_MOE_WARP_SHIFT;
	float sum[tpw][groups][TC_MOE_MMA_RESULTS];
#pragma unroll
	for (int j = 0; j < tpw; j++)
#pragma unroll
		for (int group = 0; group < groups; group++)
#pragma unroll
			for (int index = 0; index < TC_MOE_MMA_RESULTS; index++)
				sum[j][group][index] = 0.0f;
	const float zero[TC_MOE_MMA_RESULTS] = { 0.0f, 0.0f, 0.0f, 0.0f };
	/* ldmatrix source: matrix lane / 8 = (hi k0..7, hi k8..15, lo k0..7, lo k8..15) of rows 8 g + lane % 8; for k16
	 * step s the 16-byte vector 2 s ^ x */
	const int swizzle = ((lane >> TCP_GROUP_SHIFT) & 1) ^ (lane & TCP_ROW_GROUP_MASK);
	const uint32_t b_lane = shared +
				(lane >> TCP_LO_MATRIX_SHIFT) * (TCP_TILE_ROWS * TCP_K_CHUNK * TC_MOE_HALFWORD_BYTES) +
				(lane & TCP_ROW_GROUP_MASK) * (TCP_K_CHUNK * TC_MOE_HALFWORD_BYTES);
	uint32_t word_lane[ring::load_words];
	int shift;
	{
		const tc_moe_window<shape::bits> window(lane);
		shift = window.shift;
#pragma unroll
		for (int index = 0; index < ring::load_words; index++)
			word_lane[index] = shared + shape::stage_weights + tpw * warp * shape::tile_bytes +
					   TC_MOE_WORD_BYTES * window.word[index];
	}
#pragma unroll
	for (int stage = 0; stage < TCP_STAGES - 1; stage++) {
		load();
		tcp_commit();
	}
	uint32_t read = 0;
	for (int chunk = 0; chunk < chunks; chunk++) {
		tcp_wait<TCP_STAGES - TC_MOE_RING_OPEN_STAGES>();
		__syncthreads();
		if (chunk + TCP_STAGES - 1 < chunks)
			load();
		tcp_commit();
		auto k16_step = [&](int step) {
			uint32_t a[tpw][TC_MOE_FRAGMENT_WORDS];
#pragma unroll
			for (int j = 0; j < tpw; j++) {
				uint32_t words[ring::load_words];
#pragma unroll
				for (int index = 0; index < ring::load_words; index++)
					words[index] = tcp_lds32(word_lane[index] + read +
								 step * (shape::block_tiles * shape::tile_bytes) +
								 j * shape::tile_bytes);
				tc_moe_fragment<shape::bits>(words, shift, a[j]);
			}
			const uint32_t b_source =
				b_lane + read + ((swizzle ^ (TCP_STEP_VECTORS * step)) * TCP_VECTOR_BYTES) +
				first_group * TCP_ROW_GROUP * TCP_K_CHUNK * TC_MOE_HALFWORD_BYTES;
			uint32_t b[TCP_B_BUFFERS][TCP_B_MATRICES];
			tcp_ldmatrix4(b_source, b[0]);
#pragma unroll
			for (int group = 0; group < groups; group++) {
				if (group + 1 < groups)
					tcp_ldmatrix4(b_source + (group + 1) * TCP_ROW_GROUP * TCP_K_CHUNK *
									 TC_MOE_HALFWORD_BYTES,
						      b[(group + 1) & 1]);
				const uint32_t(&fragment)[TCP_B_MATRICES] = b[group & 1];
#pragma unroll
				for (int j = 0; j < tpw; j++) {
					float d[TC_MOE_MMA_RESULTS];
					/* the low products seed the high MMA */
					tcp_mma(a[j], fragment[TCP_B_LO_K0], fragment[TCP_B_LO_K8], zero, d);
					tcp_mma(a[j], fragment[TCP_B_HI_K0], fragment[TCP_B_HI_K8], d, d);
#pragma unroll
					for (int index = 0; index < TC_MOE_MMA_RESULTS; index++)
						sum[j][group][index] = __fadd_rn(sum[j][group][index], d[index]);
				}
			}
		};
		if constexpr (shape::compact_steps) {
#pragma unroll 1
			for (int step = 0; step < TCP_K_STEPS; step++)
				k16_step(step);
		} else {
#pragma unroll
			for (int step = 0; step < TCP_K_STEPS; step++)
				k16_step(step);
		}
		read = tcp_next_stage<shape>(read);
	}
	/* lane holds output columns 16 tile + lane / 4 (+ 8) of rows 8 g + 2 (lane % 4) (+ 1) */
	const int quad = lane & TC_MOE_QUAD_MASK;
#pragma unroll
	for (int j = 0; j < tpw; j++) {
		const int column = (column_block * shape::block_tiles + tpw * warp + j) * TC_MOE_TILE +
				   (lane >> TC_MOE_COLUMN_SHIFT);
#pragma unroll
		for (int group = 0; group < groups; group++) {
#pragma unroll
			for (int pair = 0; pair < TC_MOE_LANE_COLUMNS; pair++) {
				const int row =
					TCP_ROW_GROUP * (first_group + group) + TC_MOE_LANE_COLUMNS * quad + pair;
				if (row < rows) {
					const int64_t position = first + row;
					const float scale =
						p.unscale[position *
								  shape::projection_count +
							  projection_index];
					float *out = p.out +
						     projection_index *
							     p.out_projection +
						     position * shape::n +
						     column;
					out[0] = __fmul_rn(sum[j][group][pair], scale);
					out[TC_MOE_HALF_OUTPUTS] =
						__fmul_rn(sum[j][group][TC_MOE_LANE_COLUMNS + pair], scale);
				}
			}
		}
	}
}

/* One block item: (route tile, projection, column block); trellis(expert, projection) gives the matrix's trellis,
 * width_of(expert) a mixed shape's expert width. */
template <typename shape, typename trellis_type, typename width_type>
__device__ __forceinline__ void tcp_gemm_body(const struct tcp_gemm &p, int64_t tile_index, int projection_index,
					      int column_block, trellis_type trellis_of, width_type width_of);
template <typename shape, typename trellis_type, typename width_type>
__device__ __forceinline__ void tcp_gemm_item(const struct tcp_gemm &p, trellis_type trellis_of, width_type width_of)
{
	const int64_t items = (p.end_tile - p.first_tile) * shape::projection_count * shape::column_blocks;
	const int64_t item = blockIdx.x;
	if (item >= items)
		return;
	const int64_t tile_index = p.first_tile + item / (shape::projection_count * shape::column_blocks);
	const int rest = (int)(item % (shape::projection_count * shape::column_blocks));
	tcp_gemm_body<shape>(p, tile_index, rest / shape::column_blocks, rest % shape::column_blocks, trellis_of,
			     width_of);
}
/* The hybrid form of a width that is no multiple of the two-tile blocks (640): its whole blocks of wide (two tiles a
 * warp, 256 columns) and the rest as one block of narrow (a tile a warp, 128 columns), items (tile, projection, block
 * of the wide_shape's then the narrow one). Each output's K loop is the base form's: the same bits; a tile's operand
 * rows are read three times, not five. */
template <typename wide, typename narrow, typename trellis_type, typename width_type>
__device__ __forceinline__ void tcp_gemm_hybrid_item(const struct tcp_gemm &p, trellis_type trellis_of,
						     width_type width_of)
{
	constexpr int blocks = wide::column_blocks + 1;
	static_assert(wide::n == narrow::n && wide::n - wide::column_blocks * TC_MOE_TILE * wide::block_tiles ==
							   TC_MOE_TILE * narrow::block_tiles,
		      "the narrow block takes the columns behind the wide blocks");
	const int64_t items = (p.end_tile - p.first_tile) * wide::projection_count * blocks;
	const int64_t item = blockIdx.x;
	if (item >= items)
		return;
	const int64_t tile_index = p.first_tile + item / (wide::projection_count * blocks);
	const int rest = (int)(item % (wide::projection_count * blocks));
	const int projection_index = rest / blocks, block = rest % blocks;
	if (block < wide::column_blocks) /* block-uniform: every thread reaches the same barriers */
		tcp_gemm_body<wide>(p, tile_index, projection_index, block, trellis_of, width_of);
	else
		tcp_gemm_body<narrow>(p, tile_index, projection_index,
				      wide::column_blocks * wide::block_tiles / narrow::block_tiles, trellis_of,
				      width_of);
}
template <typename shape, typename trellis_type, typename width_type>
__device__ __forceinline__ void tcp_gemm_body(const struct tcp_gemm &p, int64_t tile_index, int projection_index,
					      int column_block, trellis_type trellis_of, width_type width_of)
{
	extern __shared__ __align__(TCP_VECTOR_BYTES) unsigned char tcp_gemm_shared[];
	const uint32_t shared = (uint32_t)__cvta_generic_to_shared(tcp_gemm_shared);
	const int4 tile = ((const int4 *)(p.table + tcp_tiles_offset(p.pairs)))[tile_index];
	const int rows = tile.z;
	/* mixed: block-uniform exits before any copy or barrier */
	int tiles_n = shape::tiles_n, tile_chunks = shape::chunks;
	if constexpr (shape::mixed) {
		const int width = width_of(tile.x);
		if constexpr (shape::first < TCP_DOWN_PROJECTION) {
			if (column_block * TC_MOE_TILE * shape::block_tiles >= width)
				return;
			tiles_n = width / TC_MOE_TILE;
		} else {
			tile_chunks = width / TCP_K_CHUNK;
		}
	}
	const int64_t weight_chunk = shape::mixed ? (int64_t)TCP_K_STEPS * tiles_n * shape::tile_bytes :
						    shape::weight_chunk;
	const unsigned char *trellis = trellis_of(tile.x, shape::first + projection_index);
	const __half *operand = p.operand + (int64_t)tile.y * shape::row_halves +
				(int64_t)projection_index * TC_MOE_HI_LO_ROWS * shape::k;
	/* a stage's cp.async work: operand vectors tid + 256 i (rows r0 and r0 + 32 of both parts), trellis vectors
	 * tid + 256 j of the block's tiles; rows past the tile inside its 8-row groups are zero-filled from row 0 */
	const int tid = threadIdx.x;
	static_assert(TCP_GEMM_THREADS == TCP_COPY_ROWS * TCP_ROW_VECTORS &&
			      TCP_TILE_ROWS == TCP_COPY_PASSES * TCP_COPY_ROWS &&
			      TCP_K_CHUNK == TCP_ROW_VECTORS * TCP_VECTOR_VALUES,
		      "operand copy layout");
	constexpr int step_vectors = shape::block_tiles * shape::tile_bytes / TCP_VECTOR_BYTES;
	constexpr int weight_vectors = TCP_K_STEPS * step_vectors;
	constexpr int weight_copies = (weight_vectors + TCP_GEMM_THREADS - 1) / TCP_GEMM_THREADS;
	const int r0 = tid >> TCP_ROW_VECTOR_SHIFT, chunk_index = tid & TCP_ROW_VECTOR_MASK;
	const int padded_rows = (rows + TCP_ROW_GROUP_MASK) & ~TCP_ROW_GROUP_MASK;
	const bool live0 = r0 < padded_rows, live1 = r0 + TCP_COPY_ROWS < padded_rows;
	const int size0 = r0 < rows ? TCP_VECTOR_BYTES : 0, size1 = r0 + TCP_COPY_ROWS < rows ? TCP_VECTOR_BYTES : 0;
	const __half *source0 =
		operand + (int64_t)(r0 < rows ? r0 : 0) * shape::row_halves + chunk_index * TCP_VECTOR_VALUES;
	const __half *source1 = operand +
				 (int64_t)(r0 + TCP_COPY_ROWS < rows ? r0 + TCP_COPY_ROWS : 0) * shape::row_halves +
				 chunk_index * TCP_VECTOR_VALUES;
	const uint32_t b_destination = shared + r0 * (TCP_K_CHUNK * TC_MOE_HALFWORD_BYTES) +
				       (chunk_index ^ (r0 & TCP_ROW_GROUP_MASK)) * TCP_VECTOR_BYTES;
	const bool weight_last_live = tid + (weight_copies - 1) * TCP_GEMM_THREADS < weight_vectors;
	const unsigned char *weight_source[weight_copies];
#pragma unroll
	for (int copy = 0; copy < weight_copies; copy++) {
		const int vector = copy + 1 < weight_copies || weight_last_live ? tid + copy * TCP_GEMM_THREADS : tid;
		weight_source[copy] =
			trellis +
			((int64_t)(vector / step_vectors) * tiles_n + (int64_t)column_block * shape::block_tiles) *
				shape::tile_bytes +
			(vector % step_vectors) * TCP_VECTOR_BYTES;
	}
	const uint32_t weight_destination = shared + shape::stage_weights + tid * TCP_VECTOR_BYTES;
	uint32_t stage = 0;
	auto load = [&]() {
		if (live0) {
			tcp_copy16(b_destination + stage, source0, size0);
			tcp_copy16(b_destination + stage + TCP_STAGE_LO_ROWS, source0 + shape::k, size0);
		}
		if (live1) {
			tcp_copy16(b_destination + stage + TCP_STAGE_SECOND_ROWS, source1, size1);
			tcp_copy16(b_destination + stage + TCP_STAGE_LO_SECOND_ROWS, source1 + shape::k, size1);
		}
#pragma unroll
		for (int copy = 0; copy < weight_copies; copy++) {
			if (copy + 1 < weight_copies || weight_last_live)
				tcp_copy16(weight_destination + stage + copy * TCP_GEMM_THREADS * TCP_VECTOR_BYTES,
					   weight_source[copy], TCP_VECTOR_BYTES);
			weight_source[copy] += weight_chunk;
		}
		source0 += TCP_K_CHUNK;
		source1 += TCP_K_CHUNK;
		stage = tcp_next_stage<shape>(stage);
	};
	const int groups = (rows + TCP_ROW_GROUP_MASK) >> TCP_ROW_GROUP_SHIFT;
	if constexpr (shape::tpw == TCP_WIDE_TILES) {
		/* two tiles a warp keep sum[2][8][4]: a tile of more than 4 groups runs as two passes of at most 4, so
		 * no register spills; the rewind restarts the K pipeline from the tile's first chunk */
		const __half *const source0_first = source0, *const source1_first = source1;
		const unsigned char *weight_first[weight_copies];
#pragma unroll
		for (int copy = 0; copy < weight_copies; copy++)
			weight_first[copy] = weight_source[copy];
		auto rewind = [&]() {
			tcp_wait<0>();
			__syncthreads();
			source0 = source0_first;
			source1 = source1_first;
#pragma unroll
			for (int copy = 0; copy < weight_copies; copy++)
				weight_source[copy] = weight_first[copy];
			stage = 0;
		};
		switch (groups) {
#define TCP_GROUPS(count)                                                                                    \
	case count:                                                                                          \
		tcp_run_tile<count, 0, shape>(p, shared, load, rows, tile.y, projection_index, column_block, \
					      tile_chunks);                                                  \
		break;
			TCP_GROUPS(TCP_ONE_GROUP) TCP_GROUPS(TCP_TWO_GROUPS) TCP_GROUPS(TCP_THREE_GROUPS)
			TCP_GROUPS(TCP_FOUR_GROUPS)
#undef TCP_GROUPS
#define TCP_GROUPS(count)                                                                                              \
	case count:                                                                                                    \
		tcp_run_tile<TCP_PASS_GROUPS, 0, shape>(p, shared, load, rows, tile.y, projection_index, column_block, \
							tile_chunks);                                                  \
		rewind();                                                                                              \
		tcp_run_tile<count - TCP_PASS_GROUPS, TCP_PASS_GROUPS, shape>(p, shared, load, rows, tile.y,           \
							projection_index, column_block, tile_chunks);                  \
		break;
			TCP_GROUPS(TCP_FIVE_GROUPS) TCP_GROUPS(TCP_SIX_GROUPS) TCP_GROUPS(TCP_SEVEN_GROUPS)
			TCP_GROUPS(TCP_EIGHT_GROUPS)
#undef TCP_GROUPS
		default:
			break;
		}
	} else {
		switch (groups) {
#define TCP_GROUPS(count)                                                                                    \
	case count:                                                                                          \
		tcp_run_tile<count, 0, shape>(p, shared, load, rows, tile.y, projection_index, column_block, \
					      tile_chunks);                                                  \
		break;
			TCP_GROUPS(TCP_ONE_GROUP) TCP_GROUPS(TCP_TWO_GROUPS) TCP_GROUPS(TCP_THREE_GROUPS)
			TCP_GROUPS(TCP_FOUR_GROUPS) TCP_GROUPS(TCP_FIVE_GROUPS) TCP_GROUPS(TCP_SIX_GROUPS)
			TCP_GROUPS(TCP_SEVEN_GROUPS) TCP_GROUPS(TCP_EIGHT_GROUPS)
#undef TCP_GROUPS
		default:
			break;
		}
	}
}

/* ---- combine: one warp a (row, 512-column span); the routes' weighted products in FP64, route order ---- */
__device__ __forceinline__ void tcp_store8(__nv_bfloat16 *out, bool vector, const float *sum)
{
	if (vector) {
		uint32_t words[TCP_VECTOR_WORDS];
#pragma unroll
		for (int word = 0; word < TCP_VECTOR_WORDS; word++)
			words[word] =
				(uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(sum[TCP_WORD_VALUES * word])) |
				((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(sum[TCP_WORD_VALUES * word + 1]))
				 << TC_MOE_HALFWORD_BITS);
		*(uint4 *)out = make_uint4(words[TCP_X], words[TCP_Y], words[TCP_Z], words[TCP_W]);
	} else {
#pragma unroll
		for (int index = 0; index < TCP_VECTOR_VALUES; index++)
			out[index] = __float2bfloat16_rn(sum[index]);
	}
}
__device__ __forceinline__ void tcp_store8(float *out, bool vector, const float *sum)
{
	if (vector) {
		*(float4 *)out = make_float4(sum[TCP_X], sum[TCP_Y], sum[TCP_Z], sum[TCP_W]);
		*(float4 *)(out + TCP_VECTOR_WORDS) =
			make_float4(sum[TCP_VECTOR_WORDS + TCP_X], sum[TCP_VECTOR_WORDS + TCP_Y],
				    sum[TCP_VECTOR_WORDS + TCP_Z], sum[TCP_VECTOR_WORDS + TCP_W]);
	} else {
#pragma unroll
		for (int index = 0; index < TCP_VECTOR_VALUES; index++)
			out[index] = sum[index];
	}
}
/* route_of(row, route, &position, &svh, &weight): the route's sorted position (-1: none), down svh and weight. */
template <int hidden, typename out_type, typename route_type>
__device__ __forceinline__ void tcp_combine_task(const float *down, int64_t rows, int routes, route_type route_of,
						 out_type *out)
{
	constexpr int spans = hidden / TCP_SPAN_VALUES;
	const int lane = threadIdx.x & TC_MOE_LANE_MASK;
	const int lane_column =
		TC_MOE_TRANSFORM * (lane >> TCP_GROUP_SHIFT) + TCP_VECTOR_VALUES * (lane & TCP_GROUP_LANE_MASK);
	const bool out_vector = tcp_aligned(out);
	const int64_t task = (int64_t)blockIdx.x * TCP_OPERAND_WARPS + (threadIdx.x >> TC_MOE_WARP_SHIFT);
	if (task >= rows * spans)
		return;
	const int64_t row = task / spans;
	const int base = (int)(task % spans) * TCP_SPAN_VALUES + lane_column;
	double sum[TCP_GROUP_VALUES];
#pragma unroll
	for (int index = 0; index < TCP_GROUP_VALUES; index++)
		sum[index] = 0.0;
	for (int route = 0; route < routes; route++) {
		int position;
		const __half *svh;
		double weight;
		if (!route_of(row, route, &position, &svh, &weight))
			continue;
		const bool vector = tcp_aligned(svh);
		const float *source = down + (int64_t)position * hidden + base;
		float value[TCP_GROUP_VALUES];
		tcp_load8(source, value);
		tcp_load8(source + TCP_HALF_BLOCK, value + TCP_VECTOR_VALUES);
		tcp_hadamard_group(value, lane);
#pragma unroll
		for (int half = 0; half < TCP_BLOCK_HALVES; half++) {
			float scale[TCP_VECTOR_VALUES];
			tcp_load8(svh + base + TCP_HALF_BLOCK * half, vector, scale);
#pragma unroll
			for (int index = 0; index < TCP_VECTOR_VALUES; index++) {
				const double product = __fmul_rn(value[TCP_VECTOR_VALUES * half + index], scale[index]);
				sum[TCP_VECTOR_VALUES * half + index] =
					fma(product, weight, sum[TCP_VECTOR_VALUES * half + index]);
			}
		}
	}
	float rounded[TCP_GROUP_VALUES];
#pragma unroll
	for (int index = 0; index < TCP_GROUP_VALUES; index++)
		rounded[index] = __double2float_rn(sum[index]);
	out_type *destination = out + row * hidden + base;
	tcp_store8(destination, out_vector, rounded);
	tcp_store8(destination + TCP_HALF_BLOCK, out_vector, rounded + TCP_VECTOR_VALUES);
}

#endif
