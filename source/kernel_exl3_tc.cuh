/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_exl3_tc.cuh: the device arithmetic of the routed experts of one DeepSeek-V4.1-Flash EXL3 layer on the tensor
 * cores, one TP rank's intermediate slice: a native rewrite of GNmac's MoE (Morrowmake's `moe_out_v2_try`, a0c57b5,
 * served by GNmac's variant dsv41_moe_v2try for every call of 1 to 64 rows; its arithmetic is Curtis's tc-moe
 * `moe_out_v2`, 88eb6e1), bitwise the original's (gnmac_port/kernels: tcmoe01/02). Two kernel sets include it: the
 * engine's (kernel_exl3_tc.cu, from the router's job table, exl3_tc_records.h) and the original-API harness's
 * (gnmac_port/kernels/kernel_exl3_tc_moe.cuh, which also routes).
 *
 * Shapes: hidden 5,120; rank intermediate 640 or 512 (the 640/640/512/512 split of 2,304); K3 (bits_x2 6) or K2
 * (bits_x2 4); the mul1 codebook. A matrix is EXL3: tiles [input/16][output/16][8 bits_x2] 16-bit words, FP16
 * suh[input] and svh[output]; y = svh * H128(W^T H128(suh * x)).
 *
 * Arithmetic (the order is the contract; a row never depends on another row):
 *   1. gate/up operand: x * suh and H128 in FP64, rounded once to FP32; one power of two 2^e per row and scale group
 *      puts the group's peak in [2^14, 2^15); hi = f16(v 2^e), lo = f16(v 2^e - hi), stored in the MMA B lane order.
 *   2. products: m16n8k16 FP16 MMAs, A = decoded weights (exact FP16), B = 4 rows' hi and 4 rows' lo; each k16 MMA
 *      starts from C = 0 and is added in FP32 in increasing k inside a K slice; a slice's (hi + lo) 2^-e is stored.
 *   3. activation (FP64): gate and up slices summed in slice order (slice 0 taken as it is), H128, * svh, the SwiGLU
 *      limit, SiLU(gate) * up, * down suh, H128; the down operand as in 1 with one 2^e per row.
 *   4. down products as 2; combine (FP64): slices summed, H128, * svh, one FMA a route in route order, then FP32.
 * Schedule (moe_out_v2_try's): 8-tile column groups, each warp's trellis words through its own cp.async shared ring
 * (16-byte aligned trellis), two product blocks a multiprocessor, the activation's partials 8 slices at a time, one
 * operand block per (pair, projection).
 */
#ifndef KERNEL_EXL3_TC_CUH
#define KERNEL_EXL3_TC_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define TC_MOE_HIDDEN 5120
#define TC_MOE_TRANSFORM 128                 /* the H128 block */
#define TC_MOE_TILE 16                       /* an EXL3 tile side, the MMA k step */
#define TC_MOE_TILE_ROWS 8                   /* rows of one route tile: two n8 groups of 4 rows (hi and lo) */
#define TC_MOE_MOST_ASSIGNMENTS 1024
#define TC_MOE_MOST_EXPERTS 4096
#define TC_MOE_PROJECTIONS 3                 /* gate, up, down */
#define TC_MOE_ROUTE_THREADS 640             /* 20 warps: two H128 blocks of the hidden row each */
#define TC_MOE_LANE_THREADS 256
#define TC_MOE_COLUMN_TILES 8                /* n16 output tiles of one warp item */
#define TC_MOE_PREFETCH_STEPS 3              /* k steps prefetched to L2 ahead of the decode (register path) */
#define TC_MOE_RING_STAGES 4                 /* k steps of a warp's ring: 3 in flight while one decodes */
#define TC_MOE_RING_LEAST_STEP_BYTES 384     /* a k step of fewer trellis bytes keeps the register path */
#define TC_MOE_RING_MOST_BLOCK_BYTES 65536   /* stages halve while a block's rings exceed this */
#define TC_MOE_ACTIVATION_CHUNK 8            /* gate/up slice partials in flight at once in the activation */
#define TC_MOE_LONG_ASSIGNMENTS 160          /* from here: long gate/up slices and one down slice */
#define TC_MOE_COMBINE_MOST_THREADS 1024
#define TC_MOE_WARP 32
#define TC_MOE_FULL_MASK 0xffffffffu

/* Per local expert and projection: the trellis, suh and svh device addresses (uint64 [experts][3][3]) and the
 * shape words {input, output, bits_x2, codebook} (int64 [experts][3][4]), the original op's packed tables. */
/* ---- the codec: a lane's eight trellis states of one 16x16 tile (its ring window), the mul1 values ---- */

template <int bits_x2> struct tc_moe_ring {
	static constexpr int whole = bits_x2 / 2;
	static constexpr bool odd = (bits_x2 & 1) != 0;
	static constexpr int ring_bits = 128 * bits_x2;
	static constexpr int words = 4 * bits_x2;
	static constexpr int tile_halfwords = 8 * bits_x2;
	__host__ __device__ static constexpr int relative(int state) { return state * whole + (odd ? (state + 1) / 2 : 0); }
	static constexpr int window_bits = relative(7) + 16;
	static constexpr int window_words = (window_bits + 31) / 32;
	/* the largest in-word start over the 32 lanes' fixed window starts (bits_x2 6: 2 words, not 3) */
	__host__ __device__ static constexpr int most_lane_shift()
	{
		int peak = 0;
		for (int lane = 0; lane < 32; lane++) {
			int start = (8 * lane + 1) * whole + (odd ? 4 * lane : 0) - 16;
			if (start < 0)
				start += ring_bits;
			peak = (start & 31) > peak ? (start & 31) : peak;
		}
		return peak;
	}
	static constexpr int load_words = (most_lane_shift() + window_bits + 31) / 32;
	static_assert(load_words <= words, "the window fits the ring");
};

template <int bits_x2> struct tc_moe_window {
	int word[tc_moe_ring<bits_x2>::load_words];
	int shift;
	__device__ __forceinline__ explicit tc_moe_window(int lane)
	{
		typedef tc_moe_ring<bits_x2> ring;
		int start = (8 * lane + 1) * ring::whole + (ring::odd ? 4 * lane : 0) - 16;
		if (start < 0)
			start += ring::ring_bits;
		shift = start & 31;
		int index = start >> 5;
#pragma unroll
		for (int load = 0; load < ring::load_words; load++) {
			word[load] = index;
			index = index + 1 == ring::words ? 0 : index + 1;
		}
	}
};

template <int offset, int window_words> __device__ __forceinline__ uint32_t tc_moe_state(const uint32_t (&window)[window_words])
{
	constexpr int word = offset >> 5, bit = offset & 31;
	if constexpr (bit <= 16)
		return (window[word] >> (16 - bit)) & 0xffffu;
	else
		return __funnelshift_l(window[word + 1], window[word], bit) >> 16;
}

/* Two mul1 values as an FP16 pair (low half the first): the byte sum of the hash in the mantissa of 1024 (the dp4a
 * accumulator 0x6400, exact), then one FP16 FMA, round to nearest. */
__device__ __forceinline__ uint32_t tc_moe_mul1_pair(uint32_t first, uint32_t second)
{
	const uint32_t sum_first = __dp4a(first * 0x83DCD12Du, 0x01010101u, 0x6400u);
	const uint32_t sum_second = __dp4a(second * 0x83DCD12Du, 0x01010101u, 0x6400u);
	const uint32_t packed = __byte_perm(sum_first, sum_second, 0x5410);
	const uint32_t scale = 0x1eee1eeeu, bias = 0xc931c931u;
	const __half2 value = __hfma2(*(const __half2 *)&packed, *(const __half2 *)&scale, *(const __half2 *)&bias);
	return *(const uint32_t *)&value;
}

/* A lane's eight values as the m16n8k16 A fragment of W^T (fragment row = tile column = output, fragment column =
 * tile row = k): value j sits at tile row 2 (lane % 4) + j % 2 + 8 ((j / 2) % 2), column lane / 4 + 8 (j / 4), so
 * the fragment's register pairs are values (0,1), (4,5), (2,3), (6,7). */
template <int bits_x2>
__device__ __forceinline__ void tc_moe_fragment(const uint32_t (&words)[tc_moe_ring<bits_x2>::load_words], int shift,
						uint32_t (&fragment)[4])
{
	typedef tc_moe_ring<bits_x2> ring;
	uint32_t window[ring::window_words];
#pragma unroll
	for (int index = 0; index < ring::window_words; index++) {
		const uint32_t next = index + 1 < ring::load_words ? words[index + 1] : 0u;
		window[index] = __funnelshift_l(next, words[index], shift);
	}
	uint32_t state[8];
	state[0] = tc_moe_state<ring::relative(0), ring::window_words>(window);
	state[1] = tc_moe_state<ring::relative(1), ring::window_words>(window);
	state[2] = tc_moe_state<ring::relative(2), ring::window_words>(window);
	state[3] = tc_moe_state<ring::relative(3), ring::window_words>(window);
	state[4] = tc_moe_state<ring::relative(4), ring::window_words>(window);
	state[5] = tc_moe_state<ring::relative(5), ring::window_words>(window);
	state[6] = tc_moe_state<ring::relative(6), ring::window_words>(window);
	state[7] = tc_moe_state<ring::relative(7), ring::window_words>(window);
	fragment[0] = tc_moe_mul1_pair(state[0], state[1]);
	fragment[1] = tc_moe_mul1_pair(state[4], state[5]);
	fragment[2] = tc_moe_mul1_pair(state[2], state[3]);
	fragment[3] = tc_moe_mul1_pair(state[6], state[7]);
}

/* H128 of a warp's 128 values (lane holds lane + 32 i) in FP64: butterflies over the lanes, then over i, then the
 * scale 1/sqrt(128) rounded to FP64. */
__device__ __forceinline__ void tc_moe_hadamard(double (&value)[4], int lane)
{
#pragma unroll
	for (int distance = 1; distance < TC_MOE_WARP; distance <<= 1) {
		const bool upper = (lane & distance) != 0;
#pragma unroll
		for (int index = 0; index < 4; index++) {
			const double other = __shfl_xor_sync(TC_MOE_FULL_MASK, value[index], distance);
			value[index] = upper ? __dsub_rn(other, value[index]) : __dadd_rn(value[index], other);
		}
	}
#pragma unroll
	for (int distance = 1; distance < 4; distance <<= 1) {
		double next[4];
#pragma unroll
		for (int index = 0; index < 4; index++) {
			const double other = value[index ^ distance];
			next[index] = (index & distance) ? __dsub_rn(other, value[index]) : __dadd_rn(value[index], other);
		}
#pragma unroll
		for (int index = 0; index < 4; index++)
			value[index] = next[index];
	}
#pragma unroll
	for (int index = 0; index < 4; index++)
		value[index] = __dmul_rn(value[index], 0.08838834764831844055);
}

/* ---- B operands: FP16 hi and lo rows in MMA lane order, one 2^-e per (row, slice) ---- */

/* Lane order of a K-long row: k step k / 16, then lane t's k values 2t, 2t + 1, 2t + 8, 2t + 9. */
__device__ __forceinline__ int64_t tc_moe_lane_position(int64_t k)
{
	const int within = (int)(k & 15);
	return (k & ~(int64_t)15) + ((within & 7) >> 1) * 4 + ((within >> 3) << 1) + (within & 1);
}

template <int count> __device__ __forceinline__ float tc_moe_warp_peak(const float (&value)[count])
{
	float peak = 0.0f;
#pragma unroll
	for (int index = 0; index < count; index++)
		peak = fmaxf(peak, fabsf(value[index]));
#pragma unroll
	for (int distance = 16; distance >= 1; distance >>= 1)
		peak = fmaxf(peak, __shfl_xor_sync(TC_MOE_FULL_MASK, peak, distance));
	return peak;
}

/* The exponent e that puts peak in [2^14, 2^15) (0 for a zero peak, at most 126 so 2^e and 2^-e stay normal). */
__device__ __forceinline__ int tc_moe_exponent(float peak)
{
	const int biased = (int)(__float_as_uint(peak) >> 23);
	return peak == 0.0f ? 0 : min(141 - biased, 126);
}

template <int count>
__device__ __forceinline__ void tc_moe_store_halves(const float (&value)[count], int lane, int first, int exponent,
						    __half *high, __half *low)
{
	const float scale = __uint_as_float((uint32_t)(127 + exponent) << 23);
#pragma unroll
	for (int index = 0; index < count; index++) {
		const int64_t at = tc_moe_lane_position(first + lane + TC_MOE_WARP * index);
		const float scaled = __fmul_rn(value[index], scale);
		const __half rounded = __float2half_rn(scaled);
		high[at] = rounded;
		low[at] = __float2half_rn(__fsub_rn(scaled, __half2float(rounded)));
	}
}

/* ---- route helpers ---- */

/* Exclusive block scan of two counters, with their totals. */
__device__ __forceinline__ void tc_moe_scan_pair(int first, int second, int &first_before, int &second_before,
						 int &first_total, int &second_total)
{
	__shared__ int warp_first[TC_MOE_WARP], warp_second[TC_MOE_WARP];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
	int running_first = first, running_second = second;
#pragma unroll
	for (int distance = 1; distance < TC_MOE_WARP; distance <<= 1) {
		const int up_first = __shfl_up_sync(TC_MOE_FULL_MASK, running_first, distance);
		const int up_second = __shfl_up_sync(TC_MOE_FULL_MASK, running_second, distance);
		if (lane >= distance) {
			running_first += up_first;
			running_second += up_second;
		}
	}
	if (lane == 31) {
		warp_first[warp] = running_first;
		warp_second[warp] = running_second;
	}
	__syncthreads();
	if (warp == 0) {
		int sum_first = lane < warps ? warp_first[lane] : 0, sum_second = lane < warps ? warp_second[lane] : 0;
#pragma unroll
		for (int distance = 1; distance < TC_MOE_WARP; distance <<= 1) {
			const int up_first = __shfl_up_sync(TC_MOE_FULL_MASK, sum_first, distance);
			const int up_second = __shfl_up_sync(TC_MOE_FULL_MASK, sum_second, distance);
			if (lane >= distance) {
				sum_first += up_first;
				sum_second += up_second;
			}
		}
		warp_first[lane] = sum_first;
		warp_second[lane] = sum_second;
	}
	__syncthreads();
	first_before = running_first - first + (warp > 0 ? warp_first[warp - 1] : 0);
	second_before = running_second - second + (warp > 0 ? warp_second[warp - 1] : 0);
	first_total = warp_first[warps - 1];
	second_total = warp_second[warps - 1];
}

/* ---- 1. one (pair, projection) operand row: every thread of a TC_MOE_ROUTE_THREADS block calls it ----
 * Warp w owns H128 blocks 2w and 2w + 1 of the hidden row. scale_k: columns of one 2^e (5,120 short, 1,280 long);
 * split: K slices of one scale group. Operand rows [pair][projection][hi | lo][5,120], 2^-e [pair][projection][slice]. */
template <int scale_k, int split>
__device__ __forceinline__ void tc_moe_operand_row(const __nv_bfloat16 *row, const __half *suh, int64_t pair,
						   int projection, __half *halves, float *unscale)
{
	constexpr int blocks_per_warp = 2, group_blocks = scale_k / TC_MOE_TRANSFORM;
	constexpr int warps = TC_MOE_ROUTE_THREADS / TC_MOE_WARP, warps_per_group = group_blocks / blocks_per_warp;
	static_assert(warps * blocks_per_warp * TC_MOE_TRANSFORM == TC_MOE_HIDDEN, "the warps cover the hidden row");
	static_assert(group_blocks % blocks_per_warp == 0 && warps % warps_per_group == 0 && split <= 32,
		      "the warps tile whole scale groups");
	__shared__ float block_peaks[warps][blocks_per_warp];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int group = warp / warps_per_group, member = warp % warps_per_group;
	const int64_t slices = TC_MOE_HIDDEN / scale_k * split;
	float value[blocks_per_warp][4];
#pragma unroll
	for (int sub = 0; sub < blocks_per_warp; sub++) {
		const int64_t first = ((int64_t)warp * blocks_per_warp + sub) * TC_MOE_TRANSFORM;
		double block[4];
#pragma unroll
		for (int index = 0; index < 4; index++) {
			const int64_t column = first + lane + TC_MOE_WARP * index;
			block[index] = __dmul_rn((double)__bfloat162float(row[column]), (double)__half2float(suh[column]));
		}
		tc_moe_hadamard(block, lane);
#pragma unroll
		for (int index = 0; index < 4; index++)
			value[sub][index] = __double2float_rn(block[index]);
		const float peak = tc_moe_warp_peak(value[sub]);
		if (lane == 0)
			block_peaks[warp][sub] = peak;
	}
	__syncthreads();
	float group_peak = 0.0f;
#pragma unroll
	for (int other = 0; other < warps_per_group; other++)
#pragma unroll
		for (int sub = 0; sub < blocks_per_warp; sub++)
			group_peak = fmaxf(group_peak, block_peaks[group * warps_per_group + other][sub]);
	const int exponent = tc_moe_exponent(group_peak);
	__half *high = halves + (pair * 2 + projection) * 2 * TC_MOE_HIDDEN + (int64_t)group * scale_k;
#pragma unroll
	for (int sub = 0; sub < blocks_per_warp; sub++)
		tc_moe_store_halves(value[sub], lane, (member * blocks_per_warp + sub) * TC_MOE_TRANSFORM, exponent, high,
				    high + TC_MOE_HIDDEN);
	if (member == 0 && lane < split)
		unscale[(pair * 2 + projection) * slices + group * split + lane] =
			__uint_as_float((uint32_t)(127 - exponent) << 23);
	__syncthreads();
}

/* ---- 3 and 5: the lane-tile products ---- */

struct tc_moe_products {
	int k_tiles, n_tiles;    /* input / 16, output / 16 */
	int projections;         /* 2: gate (output) and up (second_output); 1: down (output) */
	int first_projection;
	float *output, *second_output; /* [slice][assignment][output] partials */
	int64_t slice_stride;
	const __half *halves;    /* [assignment][projection][hi | lo][input] */
	const float *unscale;    /* [assignment][projection][slice] */
};

/* How a warp gets its trellis words: its shared ring (cp.async) or per-lane 32-bit loads (a k step of fewer than
 * TC_MOE_RING_LEAST_STEP_BYTES); halfword loads are kept for an unaligned caller's variant (not built). */
enum tc_moe_load { TC_MOE_LOAD_HALVES, TC_MOE_LOAD_WORDS, TC_MOE_LOAD_RING };
__host__ __device__ constexpr int tc_moe_step_bytes(int column_tiles, int bits_x2)
{
	return column_tiles * TC_MOE_TILE * bits_x2;
}
__host__ __device__ constexpr bool tc_moe_uses_ring(int column_tiles, int bits_x2)
{
	return tc_moe_step_bytes(column_tiles, bits_x2) >= TC_MOE_RING_LEAST_STEP_BYTES;
}
__host__ __device__ constexpr int tc_moe_ring_stages(int column_tiles, int bits_x2)
{
	int stages = TC_MOE_RING_STAGES;
	while (stages > 2 && tc_moe_step_bytes(column_tiles, bits_x2) * stages * (TC_MOE_LANE_THREADS / TC_MOE_WARP) >
				     TC_MOE_RING_MOST_BLOCK_BYTES)
		stages /= 2;
	return stages;
}
__host__ __device__ constexpr int tc_moe_ring_bytes(int column_tiles, int bits_x2)
{
	return tc_moe_uses_ring(column_tiles, bits_x2) ? tc_moe_ring_stages(column_tiles, bits_x2) *
								  tc_moe_step_bytes(column_tiles, bits_x2) *
								  (TC_MOE_LANE_THREADS / TC_MOE_WARP) :
							  0;
}

__device__ __forceinline__ void tc_moe_mma(const uint32_t (&a)[4], uint2 b, float (&d)[4])
{
	asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
	    "{%10, %10, %10, %10};"
	    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
	    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "f"(0.0f));
}

/* One warp item: column_tiles n16 output tiles x k_steps k16 steps (one K slice) of one route tile with `groups` n8
 * groups of 4 rows (lane r < rows holds the tile's row r's pair in row_of). The column tiles of a k step lie
 * adjacent in the trellis (16-byte aligned: the route kernel traps on any other). Ring: each k step's trellis bytes
 * go through the warp's shared ring, stages - 1 steps
 * ahead of the decode, the window words of tile j + 1 read while tile j decodes. Otherwise: per-lane loads one step
 * ahead in registers with an L2 prefetch TC_MOE_PREFETCH_STEPS ahead. The next step's B fragments load meanwhile.
 * item_slices 2: slices `slice` and `slice + 1` in one pass of the ring (2 k_steps steps); each slice's sums start
 * at 0 and end at its last step, so each partial is the one-slice item's: at the first slice's end its hi + lo sums
 * go to the warp's shared stash (TC_MOE_STASH_BYTES, the lanes quad < 2), at the item's end out with its 2^-e. */
#define TC_MOE_STASH_LANES 16 /* the lanes that store sums: quad < 2 */
template <int column_tiles> __host__ __device__ constexpr int tc_moe_stash_bytes()
{
	return column_tiles * 2 * 4 * TC_MOE_STASH_LANES * (int)sizeof(float); /* two n8 groups of 4 values */
}
/* A first slice's end in a paired item: each sum plus its lo partner's (lane + 2) into the stash by the lanes
 * quad < 2 (value v of lane 4 c + quad at v x 16 + 2 c + quad), every sum from 0 again. */
template <int groups, int column_tiles>
__device__ __forceinline__ void tc_moe_stash_sums(float (&sums)[column_tiles][groups][4], float *stash, int quad,
						  int column)
{
	float *const own_stash = stash + 2 * column + quad;
#pragma unroll
	for (int j = 0; j < column_tiles; j++)
#pragma unroll
		for (int group = 0; group < groups; group++)
#pragma unroll
			for (int index = 0; index < 4; index++) {
				const float low = __shfl_xor_sync(TC_MOE_FULL_MASK, sums[j][group][index], 2);
				if (quad < 2)
					own_stash[((j * groups + group) * 4 + index) * TC_MOE_STASH_LANES] =
						__fadd_rn(sums[j][group][index], low);
				sums[j][group][index] = 0.0f;
			}
}

template <int groups, int column_tiles, int k_steps, int bits_x2, int load, int item_slices = 1>
__device__ __forceinline__ void tc_moe_item(const struct tc_moe_products &p, int projection_index, int slice,
					    int row_of, int rows, const uint16_t *trellis, int64_t tiles_n, int first_tile,
					    int lane, const tc_moe_window<bits_x2> &window, unsigned char *ring,
					    float *output, float *stash = nullptr)
{
	static_assert(item_slices == 1 || item_slices == 2, "one or two slices an item");
	constexpr int steps = item_slices * k_steps;
	typedef tc_moe_ring<bits_x2> ring_shape;
	constexpr int tile_words = ring_shape::tile_halfwords / 2;
	constexpr int step_bytes = tc_moe_step_bytes(column_tiles, bits_x2);
	constexpr int stages = tc_moe_ring_stages(column_tiles, bits_x2);
	const int first_step = slice * k_steps, slices = p.k_tiles / k_steps;
	const int64_t row_halves = 2 * (int64_t)p.k_tiles * TC_MOE_TILE;
	const int quad = lane & 3, column = lane >> 2;
	/* B: column `column` of group g is the hi (column < 4) or lo row of tile row 4 g + column % 4 */
	const uint2 *b_source[groups];
	bool b_present[groups];
#pragma unroll
	for (int group = 0; group < groups; group++) {
		const int row = 4 * group + (column & 3);
		b_present[group] = row < rows;
		const int64_t assignment = __shfl_sync(TC_MOE_FULL_MASK, row_of, b_present[group] ? row : 0);
		b_source[group] = (const uint2 *)(p.halves + (assignment * p.projections + projection_index) * row_halves +
						  (column >> 2) * (row_halves / 2) + first_step * TC_MOE_TILE) +
				  quad;
	}
	float unscale[groups][2]; /* 2^-e of the rows this lane stores: 4 g + 2 (quad % 2) + c (paired: at the end) */
	if constexpr (item_slices == 1) {
#pragma unroll
		for (int group = 0; group < groups; group++) {
#pragma unroll
			for (int pair = 0; pair < 2; pair++) {
				const int row = 4 * group + 2 * (quad & 1) + pair;
				const int64_t assignment = __shfl_sync(TC_MOE_FULL_MASK, row_of, row < rows ? row : 0);
				const int64_t own = (assignment * p.projections + projection_index) * slices;
				unscale[group][pair] = __ldg(p.unscale + own + slice);
			}
		}
	}
	const uint16_t *step = trellis + ((int64_t)first_step * tiles_n + first_tile) * ring_shape::tile_halfwords;
	const int64_t stride = tiles_n * ring_shape::tile_halfwords;
	uint32_t words[column_tiles][ring_shape::load_words];
	/* ring: a stage is a step's trellis bytes in 16-byte chunks, lane-strided; fixed per-lane shared read bases */
	const char *fill = (const char *)step + lane * 16;
	const uint32_t ring_base = load == TC_MOE_LOAD_RING ? (uint32_t)__cvta_generic_to_shared(ring) : 0u;
	uint32_t ring_read[ring_shape::load_words];
#pragma unroll
	for (int index = 0; index < ring_shape::load_words; index++)
		ring_read[index] = ring_base + 4 * window.word[index];
	uint32_t slot = 0; /* the current step's stage offset */
	const uint32_t *word_pointer[ring_shape::load_words];
#pragma unroll
	for (int index = 0; index < ring_shape::load_words; index++)
		word_pointer[index] = (const uint32_t *)step + window.word[index];
#define TC_MOE_ISSUE(s)                                                                                              \
	do {                                                                                                         \
		if ((s) < steps) {                                                                                   \
			const uint32_t stage = ring_base + lane * 16 + ((s) & (stages - 1)) * step_bytes;            \
			_Pragma("unroll") for (int chunk = 0; chunk < step_bytes / 16; chunk += 32) if (chunk + lane < \
													step_bytes / 16) \
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(stage + chunk * 16),   \
					     "l"(fill + chunk * 16));                                                \
		}                                                                                                    \
		asm volatile("cp.async.commit_group;");                                                              \
		fill += stride * 2;                                                                                  \
	} while (0)
#define TC_MOE_RING_WORDS(j)                                                                                         \
	do {                                                                                                         \
		_Pragma("unroll") for (int index = 0; index < ring_shape::load_words; index++) asm volatile(        \
			"ld.shared.u32 %0, [%1];" : "=r"(words[j][index]) : "r"(ring_read[index] + slot + (j) * tile_words * 4)); \
	} while (0)
#define TC_MOE_FETCH(j)                                                                                              \
	do {                                                                                                         \
		if constexpr (load == TC_MOE_LOAD_WORDS) {                                                           \
			_Pragma("unroll") for (int index = 0; index < ring_shape::load_words; index++) words[j][index] = \
				__ldg(word_pointer[index] + (j) * tile_words);                                      \
		} else {                                                                                             \
			_Pragma("unroll") for (int index = 0; index < ring_shape::load_words; index++) {            \
				const uint16_t *tile = step + (j) * ring_shape::tile_halfwords;                     \
				words[j][index] = (uint32_t)__ldg(tile + 2 * window.word[index]) |                 \
						  ((uint32_t)__ldg(tile + 2 * window.word[index] + 1) << 16);      \
			}                                                                                    \
		}                                                                                            \
	} while (0)
	if constexpr (load == TC_MOE_LOAD_RING) {
#pragma unroll
		for (int s = 0; s < stages - 1; s++)
			TC_MOE_ISSUE(s);
	} else {
#pragma unroll
		for (int j = 0; j < column_tiles; j++)
			TC_MOE_FETCH(j);
	}
	uint2 b_next[groups];
#pragma unroll
	for (int group = 0; group < groups; group++)
		b_next[group] = b_present[group] ? __ldg(b_source[group]) : make_uint2(0u, 0u);
	float sums[column_tiles][groups][4];
#pragma unroll
	for (int j = 0; j < column_tiles; j++)
#pragma unroll
		for (int group = 0; group < groups; group++)
#pragma unroll
			for (int index = 0; index < 4; index++)
				sums[j][group][index] = 0.0f;
	const char *ahead =
		(const char *)(step + TC_MOE_PREFETCH_STEPS * stride) + min(lane * 128, step_bytes - 1);
#pragma unroll 1
	for (int k_step = 0; k_step < steps; k_step++) {
		const bool more = k_step + 1 < steps;
		uint2 b[groups];
#pragma unroll
		for (int group = 0; group < groups; group++) {
			b[group] = b_next[group];
			if (more && b_present[group])
				b_next[group] = __ldg(b_source[group] + 4 * (k_step + 1));
		}
		if constexpr (load == TC_MOE_LOAD_RING) {
			asm volatile("cp.async.wait_group %0;" ::"n"(stages - 2));
			asm volatile("bar.warp.sync -1;" ::: "memory");
			slot = (k_step & (stages - 1)) * step_bytes;
			TC_MOE_RING_WORDS(0);
			TC_MOE_ISSUE(k_step + stages - 1); /* every lane read the step before's stage before the barrier */
		} else {
			if (lane * 128 < step_bytes + 127 && k_step + TC_MOE_PREFETCH_STEPS < steps)
				asm volatile("prefetch.global.L2 [%0];" ::"l"(ahead));
			ahead += stride * 2;
			if constexpr (load == TC_MOE_LOAD_WORDS) {
#pragma unroll
				for (int index = 0; index < ring_shape::load_words; index++)
					word_pointer[index] += stride / 2;
			} else {
				step += stride;
			}
		}
#pragma unroll
		for (int j = 0; j < column_tiles; j++) {
			if constexpr (load == TC_MOE_LOAD_RING) {
				if (j + 1 < column_tiles)
					TC_MOE_RING_WORDS(j + 1);
			}
			uint32_t a[4];
			tc_moe_fragment<bits_x2>(words[j], window.shift, a);
			if constexpr (load != TC_MOE_LOAD_RING) {
				if (more)
					TC_MOE_FETCH(j);
			}
#pragma unroll
			for (int group = 0; group < groups; group++) {
				float d[4];
				tc_moe_mma(a, b[group], d);
#pragma unroll
				for (int index = 0; index < 4; index++)
					sums[j][group][index] = __fadd_rn(sums[j][group][index], d[index]);
			}
		}
		if constexpr (item_slices == 2) {
			if (k_step == k_steps - 1) /* the first slice's end: hi + lo into the stash, the sums from 0 */
				tc_moe_stash_sums<groups, column_tiles>(sums, stash, quad, column);
		}
	}
#undef TC_MOE_ISSUE
#undef TC_MOE_RING_WORDS
#undef TC_MOE_FETCH
	if constexpr (load == TC_MOE_LOAD_RING) {
		asm volatile("cp.async.wait_all;"); /* the next item refills the ring: every lane is past its reads */
		__syncwarp();
	}
	/* lanes quad < 2 hold the hi sums of rows 4 g + 2 quad + {0, 1}, lane quad + 2 the lo sums of the same */
	const int64_t output_stride = (int64_t)p.n_tiles * TC_MOE_TILE;
	if constexpr (item_slices == 2) { /* the second slice's 2^-e */
#pragma unroll
		for (int group = 0; group < groups; group++) {
#pragma unroll
			for (int pair = 0; pair < 2; pair++) {
				const int row = 4 * group + 2 * (quad & 1) + pair;
				const int64_t assignment = __shfl_sync(TC_MOE_FULL_MASK, row_of, row < rows ? row : 0);
				const int64_t own = (assignment * p.projections + projection_index) * slices;
				unscale[group][pair] = __ldg(p.unscale + own + slice + 1);
			}
		}
	}
	float *const last_output = output + (item_slices - 1) * p.slice_stride + first_tile * TC_MOE_TILE + column;
#pragma unroll
	for (int group = 0; group < groups; group++) {
#pragma unroll
		for (int pair = 0; pair < 2; pair++) {
			const int row = 4 * group + 2 * (quad & 1) + pair;
			const int64_t assignment = __shfl_sync(TC_MOE_FULL_MASK, row_of, row < rows ? row : 0);
			float *const row_output = last_output + assignment * output_stride;
#pragma unroll
			for (int j = 0; j < column_tiles; j++) {
#pragma unroll
				for (int half = 0; half < 2; half++) {
					const int index = 2 * half + pair;
					const float low = __shfl_xor_sync(TC_MOE_FULL_MASK, sums[j][group][index], 2);
					if (quad < 2 && row < rows)
						row_output[j * TC_MOE_TILE + 8 * half] =
							__fmul_rn(__fadd_rn(sums[j][group][index], low), unscale[group][pair]);
				}
			}
		}
	}
	if constexpr (item_slices == 2) { /* the first slice from the stash, with its own 2^-e (each lane its writes) */
		const float *const own_stash = stash + 2 * column + quad;
#pragma unroll
		for (int group = 0; group < groups; group++) {
#pragma unroll
			for (int pair = 0; pair < 2; pair++) {
				const int row = 4 * group + 2 * (quad & 1) + pair;
				const int64_t assignment = __shfl_sync(TC_MOE_FULL_MASK, row_of, row < rows ? row : 0);
				const float first_unscale = __ldg(
					p.unscale + (assignment * p.projections + projection_index) * slices + slice);
				float *const row_output =
					output + assignment * output_stride + first_tile * TC_MOE_TILE + column;
#pragma unroll
				for (int j = 0; j < column_tiles; j++) {
#pragma unroll
					for (int half = 0; half < 2; half++) {
						const int value = (j * groups + group) * 4 + 2 * half + pair;
						if (quad < 2 && row < rows)
							row_output[j * TC_MOE_TILE + 8 * half] = __fmul_rn(
								own_stash[value * TC_MOE_STASH_LANES], first_unscale);
					}
				}
			}
		}
	}
}

/* ---- 3. activation of one sorted pair: one warp per H128 block of the intermediate, a block per call ----
 * The partials load TC_MOE_ACTIVATION_CHUNK slices at a time (a chunk's loads in flight together, summed after it
 * lands: the same additions in the same order; slice 0 is taken as it is, the later ones added). mixed
 * (--exl3_expert_rotation): the expert is `width` wide (at most intermediate, the row stride); a block past it reads
 * nothing and gives zeros, so the row peak and the values of its own blocks are a call's at its width. */
__device__ __forceinline__ double tc_moe_silu(double gate)
{
	return gate / __dadd_rn(1.0, exp(-gate));
}

template <int slices, int row_blocks, bool mixed = false>
__device__ __forceinline__ void
tc_moe_activate_pair(int64_t pair, int64_t intermediate, const float *gate, const float *up, int64_t slice_stride,
		     const __half *gate_svh, const __half *up_svh, const __half *down_suh, __half *down_halves,
		     float *down_unscale, int down_slices, double limit, int64_t width = 0)
{
	constexpr int chunk = slices < TC_MOE_ACTIVATION_CHUNK ? slices : TC_MOE_ACTIVATION_CHUNK;
	static_assert(slices % chunk == 0, "whole chunks of slices");
	__shared__ float warp_peaks[row_blocks];
	const int lane = threadIdx.x & 31, block = threadIdx.x >> 5;
	const int64_t base = (int64_t)block * TC_MOE_TRANSFORM + lane;
	double value[4];
	if (!mixed || (int64_t)block * TC_MOE_TRANSFORM < width) {
		float gate_scale[4], up_scale[4], down_scale[4];
#pragma unroll
		for (int index = 0; index < 4; index++) {
			gate_scale[index] = __half2float(gate_svh[base + TC_MOE_WARP * index]);
			up_scale[index] = __half2float(up_svh[base + TC_MOE_WARP * index]);
			down_scale[index] = __half2float(down_suh[base + TC_MOE_WARP * index]);
		}
		double gate_value[4], up_value[4];
#pragma unroll
		for (int first = 0; first < slices; first += chunk) {
			float gate_part[chunk][4], up_part[chunk][4];
#pragma unroll
			for (int slice = 0; slice < chunk; slice++) {
#pragma unroll
				for (int index = 0; index < 4; index++) {
					const int64_t at = (first + slice) * slice_stride + pair * intermediate + base +
							   TC_MOE_WARP * index;
					gate_part[slice][index] = __ldg(gate + at);
					up_part[slice][index] = __ldg(up + at);
				}
			}
#pragma unroll
			for (int index = 0; index < 4; index++) {
#pragma unroll
				for (int slice = 0; slice < chunk; slice++) {
					if (first + slice == 0) {
						gate_value[index] = gate_part[0][index];
						up_value[index] = up_part[0][index];
					} else {
						gate_value[index] =
							__dadd_rn(gate_value[index], (double)gate_part[slice][index]);
						up_value[index] =
							__dadd_rn(up_value[index], (double)up_part[slice][index]);
					}
				}
			}
		}
		tc_moe_hadamard(gate_value, lane);
		tc_moe_hadamard(up_value, lane);
#pragma unroll
		for (int index = 0; index < 4; index++) {
			double gated = __dmul_rn(gate_value[index], (double)gate_scale[index]);
			double upped = __dmul_rn(up_value[index], (double)up_scale[index]);
			value[index] = 0.0;
			if (base + TC_MOE_WARP * index < intermediate) {
				if (limit > 0.0) {
					gated = fmin(gated, limit);
					upped = fmin(fmax(upped, -limit), limit);
				}
				value[index] =
					__dmul_rn(__dmul_rn(tc_moe_silu(gated), upped), (double)down_scale[index]);
			}
		}
		tc_moe_hadamard(value, lane);
	} else {
#pragma unroll
		for (int index = 0; index < 4; index++)
			value[index] = 0.0; /* a narrow expert's last block: H128 of zeros */
	}
	float rounded[4];
#pragma unroll
	for (int index = 0; index < 4; index++)
		rounded[index] = __double2float_rn(value[index]);
	const float peak = tc_moe_warp_peak(rounded);
	if (lane == 0)
		warp_peaks[block] = peak;
	__syncthreads();
	float row_peak = 0.0f;
#pragma unroll
	for (int warp = 0; warp < row_blocks; warp++)
		row_peak = fmaxf(row_peak, warp_peaks[warp]);
	const int exponent = tc_moe_exponent(row_peak);
	__half *high = down_halves + pair * 2 * intermediate;
	tc_moe_store_halves(rounded, lane, block * TC_MOE_TRANSFORM, exponent, high, high + intermediate);
	if (block == 0 && lane < down_slices)
		down_unscale[pair * down_slices + lane] = __uint_as_float((uint32_t)(127 - exponent) << 23);
	__syncthreads();
}

/* ---- 4. combine: a route's transformed down output of one H128 output block (one warp) ----
 * The down slice partials summed in slice order (FP64), H128, * svh: the value the route's FMA takes. */
template <int slices>
__device__ __forceinline__ void tc_moe_route_values(const float *down, int64_t slice_stride, int64_t pair,
						    int64_t first_column, const __half *svh, int lane, double *values)
{
	float partial[slices][4];
#pragma unroll
	for (int slice = 0; slice < slices; slice++)
#pragma unroll
		for (int index = 0; index < 4; index++)
			partial[slice][index] =
				down[slice * slice_stride + pair * TC_MOE_HIDDEN + first_column + lane + TC_MOE_WARP * index];
	float scale[4];
#pragma unroll
	for (int index = 0; index < 4; index++)
		scale[index] = __half2float(svh[first_column + lane + TC_MOE_WARP * index]);
	double value[4];
#pragma unroll
	for (int index = 0; index < 4; index++) {
		value[index] = partial[0][index];
#pragma unroll
		for (int slice = 1; slice < slices; slice++)
			value[index] = __dadd_rn(value[index], (double)partial[slice][index]);
	}
	tc_moe_hadamard(value, lane);
#pragma unroll
	for (int index = 0; index < 4; index++)
		values[lane + TC_MOE_WARP * index] = __dmul_rn(value[index], (double)scale[index]);
}

static inline size_t tc_moe_combine_shared_bytes(int routes)
{
	return (size_t)routes * (TC_MOE_TRANSFORM * sizeof(double) + sizeof(float) + sizeof(int));
}

#endif
