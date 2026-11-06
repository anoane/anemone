/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_trunk_mma.cu: the trunk product of bfloat16 rows, y = W x, W WEIGHT_FORMAT_E4M3_SCALED, 1 to 8 rows, on
 * the tensor cores in GNmac's arithmetic (dsv41_fp8w gemv_small: --tp_trunk_mma, the BF16 recipe). Device code only,
 * sm_80, module "kernel_trunk_mma", the record of trunk_linear_bfloat16_<S> (linear_records.h): the same reads,
 * writes and segments, not its order of additions.
 *
 * A weight becomes the bfloat16 of its exact value: the E4M3 byte placed in a bfloat16 (sign, then exponent and
 * mantissa in bits 4 to 10: the value times 2^-120), times 2^(w - 7) for its block's scale code w in one bfloat16
 * multiply (exact for scale codes 7 to 134; a trained matrix's lie near 120, and a larger code is taken as 134). The
 * store's 16 bytes of a lane are the A operands of the two m16n8k16 instructions of 32 inputs as they stand (bytes 0
 * and 2 of a word: output g, inputs 2 t and 2 t + 1; bytes 1 and 3: output g + 8); the B operand of row g is its
 * bfloat16 pairs at inputs 2 t, 2 t + 8, 2 t + 16 and 2 t + 24.
 * A warp owns `tiles` tiles of 16 outputs and a share of the input groups (the _sN forms split them over N warps);
 * each instruction starts from the warp's running float32 sums (inputs in ascending groups), the first warp of a split
 * adds the others' sums in warp order, and the sum is rounded once to bfloat16 (nearest, ties to even): fixed, so
 * deterministic, and a row's outputs do not depend on the other rows. Weight loads run TRUNK_MMA_DEPTH groups ahead.
 *
 * The grouped forms trunk_mma_<16|24|32|40|48>_<form> (batch_thin_bits, track G): the 1-8-row forms over 2
 * to 6 row groups of 8 (9 to 48 rows), each warp's decoded weights fed to every group's B operands: each row's
 * instructions, splits and warp-order additions are the 8-row form's, so its bits are a thin step's; the weights are
 * read and decoded once for all the rows. Row sets (--batch_row_sets, track J): a launch of a grouped form (or the
 * 1-8-row form) with gridDim.y above 1 takes the step's rows in sets of its row groups, set y from row 8 G y on: the
 * same bits, the weights decoded once a set, gridDim.y times the blocks.
 *
 * The _64 form (GNmac's gemm_mid), 9 to 64 rows, the record of trunk_relaxed_bfloat16_arguments (its
 * fold_reference unread): a block's 8 warps take 8 tiles (128 outputs of one segment) and one of k_split_count runs of
 * the input groups; the rows' values of TRUNK_MMA_PHASE groups at a time are staged in shared memory (two buffers,
 * cp.async) and read by every warp; a warp sums its tile for every row group in ascending groups. With one run the
 * block rounds its sums to bfloat16 (linear_bfloat16_bits' rounding); with more it writes them as float32 partials
 * [split][row][output] and trunk_relaxed_reduce adds the runs from +0 in ascending order: deterministic.
 * Row sets (--prefill_tail_sets, a tail program's 64-row groups): gridDim.y sets of row_count rows each, set s from row
 * row_count s on, the blocks of one (block, split) of every set adjacent in launch order (their weights read once
 * from memory); each set's rows take a one-set launch's instructions, splits and order: its bits. The partials are
 * then [split][every set's rows][output] (one reduce over all of them); the fused form takes one set alone.
 */
#include "anemone/linear_records.h"
#include "anemone/relaxed_linear_records.h"

#include <cuda_bf16.h>
#include <stdint.h>

#define TRUNK_MMA_THREADS 256
#define TRUNK_MMA_WARPS (TRUNK_MMA_THREADS / 32)
#define TRUNK_MMA_TILE_OUTPUTS 16
#define TRUNK_MMA_GROUP_INPUTS 32
#define TRUNK_MMA_SCALE_TILES 2 /* a scale block of 32 outputs */
#define TRUNK_MMA_DEPTH 4
#define TRUNK_MMA_MOST_ROWS 8
#define TRUNK_MMA_SCALE_SHIFT \
	120u /* a scale code w becomes the bfloat16 exponent field of 2^(w - 7) */
#define TRUNK_MMA_LARGEST_SCALE \
	134u /* its field 254: the largest finite power */
#define TRUNK_MMA_PLACED_MASK 0x7F007F00u
#define TRUNK_MMA_SIGN_MASK 0x80008000u
#define TRUNK_MMA_HALF_EXPONENT \
	0x00800080u /* times an exponent field: that field in both bfloat16 halves */

__device__ __forceinline__ uint4 trunk_mma_weights(const uint4 *source)
{
	uint4 value;
	asm volatile(
		"ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
		: "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
		: "l"(source));
	return value;
}

__device__ __forceinline__ uint32_t trunk_mma_multiply(uint32_t a, uint32_t b)
{
	const __nv_bfloat162 product =
		__hmul2(*reinterpret_cast<const __nv_bfloat162 *>(&a),
			*reinterpret_cast<const __nv_bfloat162 *>(&b));
	return *reinterpret_cast<const uint32_t *>(&product);
}

/* A stored word (bytes: output g input k, output g + 8 input k, output g input k + 1, output g + 8 input k + 1) becomes
 * the bfloat16 pairs of output g (low) and output g + 8 (high), each times scale. */
__device__ __forceinline__ void trunk_mma_decode(uint32_t word, uint32_t scale,
						 uint32_t &low, uint32_t &high)
{
	const uint32_t odd = ((word & TRUNK_MMA_PLACED_MASK) >> 4) |
			     (word & TRUNK_MMA_SIGN_MASK);
	const uint32_t shifted = word << 8;
	const uint32_t even = ((shifted & TRUNK_MMA_PLACED_MASK) >> 4) |
			      (shifted & TRUNK_MMA_SIGN_MASK);
	low = trunk_mma_multiply(even, scale);
	high = trunk_mma_multiply(odd, scale);
}

__device__ __forceinline__ void trunk_mma_step(float (&d)[4], uint32_t a0,
					       uint32_t a1, uint32_t a2,
					       uint32_t a3, uint32_t b0,
					       uint32_t b1)
{
	asm volatile(
		"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
		"{%0,%1,%2,%3};"
		: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
		: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* The warp's sums of its tiles (first_tile ..) over the input groups [first_group, end_group), for each row group of
 * 8 rows (row group r: rows 8 r .. 8 r + 7, its B operands from row 8 r + column). */
template <int tiles, int row_groups>
__device__ __forceinline__ void
trunk_mma_sums(const struct trunk_linear_bfloat16_arguments &arguments,
	       const unsigned char *arena, uint32_t first_tile,
	       uint32_t first_group, uint32_t end_group,
	       float (&sums)[tiles][row_groups][4], uint32_t first_row)
{
	const int lane = threadIdx.x & 31, quad = lane & 3, column = lane >> 2;
	const uint32_t groups = arguments.input_size / TRUNK_MMA_GROUP_INPUTS;
	const uint32_t tile_count =
		arguments.output_size / TRUNK_MMA_TILE_OUTPUTS;
	const uint32_t segment_tiles = tile_count / arguments.segment_count;
	const unsigned char *const matrix = arena + arguments.weights_offset;
	const unsigned char *const scale_codes =
		matrix + (uint64_t)arguments.output_size * arguments.input_size;
	const uint4 *weights[tiles];
	const unsigned char *scales[tiles];
	const uint32_t *values[tiles][row_groups];
#pragma unroll
	for (int tile = 0; tile < tiles; tile++) {
		const uint32_t at = min(first_tile + tile, tile_count - 1u);
		weights[tile] = (const uint4 *)matrix +
				(uint64_t)at * groups * 32u + lane;
		scales[tile] = scale_codes +
			       (uint64_t)(at / TRUNK_MMA_SCALE_TILES) * groups;
#pragma unroll
		for (int group = 0; group < row_groups; group++) {
			const uint32_t row =
				min(first_row + (uint32_t)(TRUNK_MMA_MOST_ROWS *
								   group +
							   column),
				    arguments.row_count - 1u);
			values[tile][group] =
				(const uint32_t *)(arena +
						   arguments.values_offset +
						   ((uint64_t)row *
							    arguments
								    .segment_count +
						    at / segment_tiles) *
							   arguments.input_size *
							   2u) +
				quad;
		}
	}
#pragma unroll
	for (int tile = 0; tile < tiles; tile++)
#pragma unroll
		for (int group = 0; group < row_groups; group++)
#pragma unroll
			for (int index = 0; index < 4; index++)
				sums[tile][group][index] = 0.0f;
	if (first_group >= end_group)
		return;
	uint4 ring[TRUNK_MMA_DEPTH][tiles];
#pragma unroll
	for (int ahead = 0; ahead < TRUNK_MMA_DEPTH; ahead++) {
		const uint32_t group = min(first_group + ahead, end_group - 1u);
#pragma unroll
		for (int tile = 0; tile < tiles; tile++)
			ring[ahead][tile] =
				trunk_mma_weights(weights[tile] + group * 32u);
	}
#pragma unroll 1
	for (uint32_t group = first_group; group < end_group;
	     group += TRUNK_MMA_DEPTH) {
#pragma unroll
		for (int slot = 0; slot < TRUNK_MMA_DEPTH; slot++) {
			const uint32_t at = group + slot;
			if (at >= end_group)
				break;
			uint4 stored[tiles];
			const uint32_t next =
				min(at + TRUNK_MMA_DEPTH, end_group - 1u);
#pragma unroll
			for (int tile = 0; tile < tiles; tile++) {
				stored[tile] = ring[slot][tile];
				ring[slot][tile] = trunk_mma_weights(
					weights[tile] + next * 32u);
			}
#pragma unroll
			for (int tile = 0; tile < tiles; tile++) {
				const uint32_t code =
					min((uint32_t)__ldg(scales[tile] + at),
					    TRUNK_MMA_LARGEST_SCALE);
				const uint32_t scale =
					(code + TRUNK_MMA_SCALE_SHIFT) *
					TRUNK_MMA_HALF_EXPONENT;
				uint32_t a[8];
				trunk_mma_decode(stored[tile].x, scale, a[0],
						 a[1]);
				trunk_mma_decode(stored[tile].y, scale, a[2],
						 a[3]);
				trunk_mma_decode(stored[tile].z, scale, a[4],
						 a[5]);
				trunk_mma_decode(stored[tile].w, scale, a[6],
						 a[7]);
#pragma unroll
				for (int group = 0; group < row_groups;
				     group++) {
					/* a tile's rows share a segment only within it: a tile reads its own values */
					const uint32_t *const fed =
						values[tile][group] +
						at * (TRUNK_MMA_GROUP_INPUTS /
						      2u);
					const uint32_t b0 = __ldg(fed),
						       b1 = __ldg(fed + 4),
						       b2 = __ldg(fed + 8),
						       b3 = __ldg(fed + 12);
					trunk_mma_step(sums[tile][group], a[0],
						       a[1], a[2], a[3], b0,
						       b1);
					trunk_mma_step(sums[tile][group], a[4],
						       a[5], a[6], a[7], b2,
						       b3);
				}
			}
		}
	}
}

/* A block: warps / split groups of `split` warps, each group `tiles` tiles of every row group from first_row on;
 * every thread reaches the barrier. */
template <int tiles, int split, int warps, int row_groups>
__device__ __forceinline__ void
trunk_mma_launch(const struct trunk_linear_bfloat16_arguments &arguments,
		 unsigned char *arena, uint32_t block, uint32_t first_row = 0)
{
	__shared__ float partials[warps][4 * tiles * row_groups][32];
	const int lane = threadIdx.x & 31, quad = lane & 3, column = lane >> 2;
	const uint32_t warp = threadIdx.x >> 5, part = warp % split;
	const uint32_t first_tile =
		(block * (warps / split) + warp / split) * tiles;
	const uint32_t groups = arguments.input_size / TRUNK_MMA_GROUP_INPUTS;
	float sums[tiles][row_groups][4];
	trunk_mma_sums<tiles, row_groups>(arguments, arena, first_tile,
					  groups * part / split,
					  groups * (part + 1u) / split, sums,
					  first_row);
	if constexpr (split > 1) {
		if (part != 0)
#pragma unroll
			for (int tile = 0; tile < tiles; tile++)
#pragma unroll
				for (int group = 0; group < row_groups; group++)
#pragma unroll
					for (int index = 0; index < 4; index++)
						partials[warp]
							[4 * (tile * row_groups +
							      group) +
							 index][lane] =
								sums[tile][group]
								    [index];
		__syncthreads();
		if (part != 0)
			return;
#pragma unroll
		for (uint32_t other = 1; other < split; other++)
#pragma unroll
			for (int tile = 0; tile < tiles; tile++)
#pragma unroll
				for (int group = 0; group < row_groups; group++)
#pragma unroll
					for (int index = 0; index < 4; index++)
						sums[tile][group][index] = __fadd_rn(
							sums[tile][group][index],
							partials[warp + other]
								[4 * (tile * row_groups +
								      group) +
								 index][lane]);
	}
	/* d0, d1: output g, rows 2 quad and 2 quad + 1; d2, d3: output g + 8 */
	__nv_bfloat16 *const output =
		(__nv_bfloat16 *)(arena + arguments.output_offset);
	const uint32_t tile_count =
		arguments.output_size / TRUNK_MMA_TILE_OUTPUTS;
#pragma unroll
	for (int tile = 0; tile < tiles; tile++) {
		if (first_tile + tile >= tile_count)
			break;
#pragma unroll
		for (int group = 0; group < row_groups; group++)
#pragma unroll
			for (int pair = 0; pair < 2; pair++) {
				const uint32_t row = first_row +
						     TRUNK_MMA_MOST_ROWS * group +
						     2u * quad + pair;
				if (row >= arguments.row_count)
					continue;
#pragma unroll
				for (int half = 0; half < 2; half++)
					output[(uint64_t)row *
						       arguments.output_size +
					       (first_tile + tile) *
						       TRUNK_MMA_TILE_OUTPUTS +
					       column + 8u * half] =
						__float2bfloat16_rn(
							sums[tile][group]
							    [2 * half + pair]);
			}
	}
}

/* grid ceil(output_size / 16 / (tiles * warps / split)), 32 warps threads; rows 1 to 8, input_size a multiple of 32.
 * Forms of 8 warps: trunk_mma_8_t<T>_s<S>; of 16: _t1_s16, _t1_s8_w16, _t1_s4_w16. Row sets: gridDim.y sets
 * of 8 x row_groups rows (a set past the rows returns at once, a whole block). */
#define TRUNK_MMA_ROWS_FORM(name, tiles, split, warps, row_groups)           \
	extern "C" __global__ void __launch_bounds__(32 * (warps))           \
		name(const struct trunk_linear_bfloat16_arguments arguments, \
		     unsigned char *arena)                                   \
	{                                                                    \
		const uint32_t first_row =                                   \
			blockIdx.y * TRUNK_MMA_MOST_ROWS * (row_groups);     \
		if (arguments.row_count == 0 ||                              \
		    first_row >= arguments.row_count ||                      \
		    (gridDim.y == 1u &&                                      \
		     arguments.row_count >                                   \
			     TRUNK_MMA_MOST_ROWS * (row_groups)))            \
			return;                                              \
		trunk_mma_launch<tiles, split, warps, row_groups>(           \
			arguments, arena, blockIdx.x, first_row);            \
	}
#define TRUNK_MMA_FORM(name, tiles, split, warps) \
	TRUNK_MMA_ROWS_FORM(name, tiles, split, warps, 1)
TRUNK_MMA_FORM(trunk_mma_8_t1_s1, 1, 1, 8)
TRUNK_MMA_FORM(trunk_mma_8_t1_s2, 1, 2, 8)
TRUNK_MMA_FORM(trunk_mma_8_t1_s4, 1, 4, 8)
TRUNK_MMA_FORM(trunk_mma_8_t1_s8, 1, 8, 8)
TRUNK_MMA_FORM(trunk_mma_8_t2_s1, 2, 1, 8)
TRUNK_MMA_FORM(trunk_mma_8_t2_s2, 2, 2, 8)
TRUNK_MMA_FORM(trunk_mma_8_t2_s4, 2, 4, 8)
TRUNK_MMA_FORM(trunk_mma_8_t2_s8, 2, 8, 8)
TRUNK_MMA_FORM(trunk_mma_8_t1_s16, 1, 16, 16)
TRUNK_MMA_FORM(trunk_mma_8_t1_s8_w16, 1, 8, 16)
TRUNK_MMA_FORM(trunk_mma_8_t1_s4_w16, 1, 4, 16)
/* the grouped forms: the 1-8-row forms' geometry over 2 to 6 row groups (9 to 48 rows; batch_thin_bits) */
#define TRUNK_MMA_GROUPED_FORMS(rows, row_groups)                             \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t1_s1, 1, 1, 8, row_groups)    \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t1_s2, 1, 2, 8, row_groups)    \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t1_s4, 1, 4, 8, row_groups)    \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t1_s8, 1, 8, 8, row_groups)    \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t2_s1, 2, 1, 8, row_groups)    \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t2_s2, 2, 2, 8, row_groups)    \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t2_s4, 2, 4, 8, row_groups)    \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t2_s8, 2, 8, 8, row_groups)    \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t1_s16, 1, 16, 16, row_groups) \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t1_s8_w16, 1, 8, 16,           \
			    row_groups)                                       \
	TRUNK_MMA_ROWS_FORM(trunk_mma_##rows##_t1_s4_w16, 1, 4, 16, row_groups)
TRUNK_MMA_GROUPED_FORMS(16, 2)
TRUNK_MMA_GROUPED_FORMS(24, 3)
TRUNK_MMA_GROUPED_FORMS(32, 4)
TRUNK_MMA_GROUPED_FORMS(40, 5)
TRUNK_MMA_GROUPED_FORMS(48, 6)
/* The pair forms (--tp_launch_trims): two products of a form in one launch, trunk_mma_pair_arguments (a
 * layer's wq_a and wkv, its shared expert's gate and up): a block runs its product's record as that product's own
 * launch runs its block (the record's fields selected one by one, so no local copy) */
#define TRUNK_MMA_PAIR_FORM(name, tiles, split, warps, row_groups)                                    \
	extern "C" __global__ void __launch_bounds__(32 * (warps))                                    \
		name(const struct trunk_mma_pair_arguments arguments,                                 \
		     unsigned char *arena)                                                            \
	{                                                                                             \
		const bool second = blockIdx.x >= arguments.first_blocks;                             \
		const struct trunk_linear_bfloat16_arguments own = {                                  \
			second ? arguments.second.weights_offset :                                    \
				 arguments.first.weights_offset,                                      \
			second ? arguments.second.values_offset :                                     \
				 arguments.first.values_offset,                                       \
			second ? arguments.second.output_offset :                                     \
				 arguments.first.output_offset,                                       \
			second ? arguments.second.output_size :                                       \
				 arguments.first.output_size,                                         \
			second ? arguments.second.input_size :                                        \
				 arguments.first.input_size,                                          \
			second ? arguments.second.row_count :                                         \
				 arguments.first.row_count,                                           \
			second ? arguments.second.segment_count :                                     \
				 arguments.first.segment_count                                        \
		};                                                                                    \
		/* a row-set pair (gridDim.y above 1): set y of the rows, as TRUNK_MMA_ROWS_FORM's */ \
		const uint32_t first_row =                                                            \
			blockIdx.y * TRUNK_MMA_MOST_ROWS * (row_groups);                              \
		if (own.row_count == 0 || first_row >= own.row_count ||                               \
		    (gridDim.y == 1u &&                                                               \
		     own.row_count > TRUNK_MMA_MOST_ROWS * (row_groups)))                             \
			return;                                                                       \
		trunk_mma_launch<tiles, split, warps, row_groups>(                                    \
			own, arena,                                                                   \
			second ? blockIdx.x - arguments.first_blocks :                                \
				 blockIdx.x,                                                          \
			first_row);                                                                   \
	}
TRUNK_MMA_PAIR_FORM(trunk_mma_8_t1_s16_pair, 1, 16, 16, 1)
TRUNK_MMA_PAIR_FORM(trunk_mma_16_t1_s16_pair, 1, 16, 16, 2)
TRUNK_MMA_PAIR_FORM(trunk_mma_24_t1_s16_pair, 1, 16, 16, 3)
TRUNK_MMA_PAIR_FORM(trunk_mma_32_t1_s16_pair, 1, 16, 16, 4)
TRUNK_MMA_PAIR_FORM(trunk_mma_40_t1_s16_pair, 1, 16, 16, 5)
TRUNK_MMA_PAIR_FORM(trunk_mma_48_t1_s16_pair, 1, 16, 16, 6)

/* ------------------------------------------------------------------------------------------- 9 to 64 rows (_64) */

#define TRUNK_MMA_PHASE \
	4 /* input groups a staged phase: also the weight ring of a warp */
#define TRUNK_MMA_PITCH         \
	(TRUNK_MMA_PHASE * 16 + \
	 4) /* 32-bit words a staged row: rows g fall on banks 4 g + t */
#define TRUNK_MMA_WIDE_MOST_ROWS 64
#define TRUNK_MMA_STAGE_WORDS (2 * TRUNK_MMA_WIDE_MOST_ROWS * TRUNK_MMA_PITCH)
#define TRUNK_MMA_ROUNDING_BIAS 0x7FFFu
#define TRUNK_MMA_MAGNITUDE_MASK 0x7FFFFFFFu
#define TRUNK_MMA_INFINITY_BITS 0x7F800000u
#define TRUNK_MMA_NOT_A_NUMBER 0x7FC0u

__device__ __forceinline__ uint16_t
trunk_mma_bfloat16_bits(float value) /* linear_bfloat16_bits */
{
	const uint32_t bits = __float_as_uint(value);
	const uint32_t rounded =
		(bits + TRUNK_MMA_ROUNDING_BIAS + ((bits >> 16) & 1u)) >> 16;
	return (uint16_t)((bits & TRUNK_MMA_MAGNITUDE_MASK) >
					  TRUNK_MMA_INFINITY_BITS ?
				  TRUNK_MMA_NOT_A_NUMBER :
				  rounded);
}

__device__ __forceinline__ void trunk_mma_copy(uint32_t *shared,
					       const void *global)
{
	const uint32_t address = (uint32_t)__cvta_generic_to_shared(shared);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(address),
		     "l"(global));
}

/* The phase of input groups [first, first + count) of rows 0 .. row_count - 1 into a buffer: 16-byte copies. */
__device__ __forceinline__ void
trunk_mma_stage(const struct trunk_relaxed_bfloat16_arguments &arguments,
		const unsigned char *values, uint32_t first, uint32_t count,
		uint32_t *buffer)
{
	const uint32_t per_row = count * 4u; /* 16-byte pieces: 4 a group */
	const uint64_t row_bytes =
		(uint64_t)arguments.segment_count * arguments.input_size * 2u;
	for (uint32_t piece = threadIdx.x;
	     piece < arguments.row_count * per_row;
	     piece += TRUNK_MMA_THREADS) {
		const uint32_t row = piece / per_row, at = piece % per_row;
		trunk_mma_copy(buffer + row * TRUNK_MMA_PITCH + at * 4u,
			       values + row * row_bytes +
				       (uint64_t)first *
					       TRUNK_MMA_GROUP_INPUTS * 2u +
				       at * 16u);
	}
	asm volatile("cp.async.commit_group;");
}

template <int row_groups, bool fused>
__device__ __forceinline__ void
trunk_mma_wide(const struct trunk_relaxed_bfloat16_arguments &arguments,
	       unsigned char *arena, uint32_t *stage, uint64_t counters_offset)
{
	const int lane = threadIdx.x & 31, quad = lane & 3, column = lane >> 2;
	const uint32_t warp = threadIdx.x >> 5;
	const uint32_t tile_count =
		arguments.output_size / TRUNK_MMA_TILE_OUTPUTS;
	const uint32_t segment_tiles = tile_count / arguments.segment_count;
	const uint32_t groups = arguments.input_size / TRUNK_MMA_GROUP_INPUTS,
		       splits = arguments.k_split_count;
	/* row sets (above): the launch's linear block, the sets innermost */
	const uint32_t sets = gridDim.y, linear = blockIdx.y * gridDim.x + blockIdx.x;
	const uint32_t set = linear % sets, item = linear / sets;
	const uint32_t block = item / splits, split = item % splits;
	const uint64_t set_rows = (uint64_t)set * arguments.row_count,
		       all_rows = (uint64_t)sets * arguments.row_count;
	const uint32_t tile = block * TRUNK_MMA_WARPS + warp,
		       at_tile = min(tile, tile_count - 1u);
	const uint32_t first_group = groups * split / splits,
		       end_group = groups * (split + 1u) / splits;
	const unsigned char *const matrix = arena + arguments.weights_offset;
	const uint4 *const weights =
		(const uint4 *)matrix + (uint64_t)at_tile * groups * 32u + lane;
	const unsigned char *const scales =
		matrix +
		(uint64_t)arguments.output_size * arguments.input_size +
		(uint64_t)(at_tile / TRUNK_MMA_SCALE_TILES) * groups;
	const unsigned char *const values =
		arena + arguments.values_offset +
		set_rows * arguments.segment_count * arguments.input_size * 2u +
		(uint64_t)(block * TRUNK_MMA_WARPS / segment_tiles) *
			arguments.input_size * 2u;
	float sums[row_groups][4];
#pragma unroll
	for (int group = 0; group < row_groups; group++)
#pragma unroll
		for (int index = 0; index < 4; index++)
			sums[group][index] = 0.0f;
	uint4 ring[TRUNK_MMA_PHASE];
#pragma unroll
	for (int ahead = 0; ahead < TRUNK_MMA_PHASE; ahead++)
		ring[ahead] = trunk_mma_weights(
			weights +
			min(first_group + ahead, end_group - 1u) * 32u);
	trunk_mma_stage(arguments, values, first_group,
			min((uint32_t)TRUNK_MMA_PHASE, end_group - first_group),
			stage);
	uint32_t phase = 0;
	for (uint32_t start = first_group; start < end_group;
	     start += TRUNK_MMA_PHASE, phase++) {
		const uint32_t next = start + TRUNK_MMA_PHASE;
		if (next < end_group) {
			trunk_mma_stage(
				arguments, values, next,
				min((uint32_t)TRUNK_MMA_PHASE,
				    end_group - next),
				stage + ((phase + 1u) & 1u) *
						TRUNK_MMA_WIDE_MOST_ROWS *
						TRUNK_MMA_PITCH);
			asm volatile("cp.async.wait_group 1;");
		} else {
			asm volatile("cp.async.wait_group 0;");
		}
		__syncthreads();
		const uint32_t *const buffer =
			stage + (phase & 1u) * TRUNK_MMA_WIDE_MOST_ROWS *
					TRUNK_MMA_PITCH;
#pragma unroll
		for (int slot = 0; slot < TRUNK_MMA_PHASE; slot++) {
			const uint32_t at = start + slot;
			if (at >= end_group)
				break;
			const uint4 stored = ring[slot];
			ring[slot] = trunk_mma_weights(
				weights +
				min(at + TRUNK_MMA_PHASE, end_group - 1u) *
					32u);
			const uint32_t code = min((uint32_t)__ldg(scales + at),
						  TRUNK_MMA_LARGEST_SCALE);
			const uint32_t scale = (code + TRUNK_MMA_SCALE_SHIFT) *
					       TRUNK_MMA_HALF_EXPONENT;
			uint32_t a[8];
			trunk_mma_decode(stored.x, scale, a[0], a[1]);
			trunk_mma_decode(stored.y, scale, a[2], a[3]);
			trunk_mma_decode(stored.z, scale, a[4], a[5]);
			trunk_mma_decode(stored.w, scale, a[6], a[7]);
#pragma unroll
			for (int group = 0; group < row_groups; group++) {
				const uint32_t *const fed =
					buffer +
					(8u * group + column) *
						TRUNK_MMA_PITCH +
					slot * 16 + quad;
				trunk_mma_step(sums[group], a[0], a[1], a[2],
					       a[3], fed[0], fed[4]);
				trunk_mma_step(sums[group], a[4], a[5], a[6],
					       a[7], fed[8], fed[12]);
			}
		}
		__syncthreads(); /* the buffer is staged again two phases on */
	}
	/* d0, d1: output g, rows 8 group + 2 quad (+ 1); d2, d3: output g + 8 */
#pragma unroll
	for (int group = 0; group < row_groups && tile < tile_count; group++)
#pragma unroll
		for (int pair = 0; pair < 2; pair++) {
			const uint32_t row = 8u * group + 2u * quad + pair;
			if (row >= arguments.row_count)
				continue;
#pragma unroll
			for (int half = 0; half < 2; half++) {
				const uint64_t at =
					(set_rows + row) * arguments.output_size +
					tile * TRUNK_MMA_TILE_OUTPUTS + column +
					8u * half;
				const float sum = sums[group][2 * half + pair];
				if (splits == 1u)
					((uint16_t *)(arena +
						      arguments.output_offset))
						[at] = trunk_mma_bfloat16_bits(
							sum);
				else
					((float *)(arena +
						   arguments.partials_offset))
						[(uint64_t)split * all_rows *
							 arguments.output_size +
						 at] = sum;
			}
		}
	if (!fused || splits == 1u || sets != 1u)
		return;
	/* fused: the block's last split adds its 128 outputs' partials from +0 in ascending split order (the reduce's) */
	__shared__ uint32_t arrived;
	__threadfence();
	__syncthreads();
	uint32_t *const counter = (uint32_t *)(arena + counters_offset) + block;
	if (threadIdx.x == 0)
		arrived = atomicAdd(counter, 1u);
	__syncthreads();
	if (arrived + 1u != splits)
		return;
	__threadfence();
	const uint32_t first_output = block * TRUNK_MMA_WARPS * TRUNK_MMA_TILE_OUTPUTS;
	const uint32_t outputs = min((uint32_t)(TRUNK_MMA_WARPS * TRUNK_MMA_TILE_OUTPUTS),
				     arguments.output_size - first_output);
	const uint64_t result_count = (uint64_t)arguments.row_count * arguments.output_size;
	const float *const partials = (const float *)(arena + arguments.partials_offset);
	uint16_t *const output = (uint16_t *)(arena + arguments.output_offset);
	for (uint32_t item = threadIdx.x; item < arguments.row_count * outputs; item += TRUNK_MMA_THREADS) {
		const uint64_t at = (uint64_t)(item / outputs) * arguments.output_size + first_output + item % outputs;
		float sum = 0.0f;
		for (uint32_t split = 0; split < splits; split++)
			sum += __ldcg(partials + (uint64_t)split * result_count + at);
		output[at] = trunk_mma_bfloat16_bits(sum);
	}
	if (threadIdx.x == 0)
		*counter = 0u; /* zero again for the next launch */
}

/* grid ceil(output_size / 128) * k_split_count, 256 threads, rows 9 to 64; a segment a whole number of blocks of 128
 * outputs, input_size a multiple of 32, values and partials 16-byte aligned. */
extern "C" __global__ void __launch_bounds__(TRUNK_MMA_THREADS, 2)
	trunk_mma_64(const struct trunk_relaxed_bfloat16_arguments arguments,
		     unsigned char *arena)
{
	__shared__ __align__(16) uint32_t stage[TRUNK_MMA_STAGE_WORDS];
	switch ((arguments.row_count + 7u) / 8u) {
#define TRUNK_MMA_ROW_GROUPS(count)                             \
	case count:                                             \
		trunk_mma_wide<count, false>(arguments, arena, stage, 0); \
		break;
		TRUNK_MMA_ROW_GROUPS(2)
		TRUNK_MMA_ROW_GROUPS(3)
		TRUNK_MMA_ROW_GROUPS(4) TRUNK_MMA_ROW_GROUPS(5)
			TRUNK_MMA_ROW_GROUPS(6) TRUNK_MMA_ROW_GROUPS(7)
				TRUNK_MMA_ROW_GROUPS(8)
#undef TRUNK_MMA_ROW_GROUPS
					default : break;
	}
}

/* the _64 form with its own reduce (relaxed_linear_records.h's trunk_mma_fused_arguments): the same grid */
extern "C" __global__ void __launch_bounds__(TRUNK_MMA_THREADS, 2)
	trunk_mma_64_fused(const struct trunk_mma_fused_arguments arguments,
			   unsigned char *arena)
{
	__shared__ __align__(16) uint32_t stage[TRUNK_MMA_STAGE_WORDS];
	switch ((arguments.wide.row_count + 7u) / 8u) {
#define TRUNK_MMA_ROW_GROUPS(count) \
	case count: \
		trunk_mma_wide<count, true>(arguments.wide, arena, stage, arguments.counters_offset); \
		break;
		TRUNK_MMA_ROW_GROUPS(2)
		TRUNK_MMA_ROW_GROUPS(3)
		TRUNK_MMA_ROW_GROUPS(4)
		TRUNK_MMA_ROW_GROUPS(5)
		TRUNK_MMA_ROW_GROUPS(6)
		TRUNK_MMA_ROW_GROUPS(7)
		TRUNK_MMA_ROW_GROUPS(8)
#undef TRUNK_MMA_ROW_GROUPS
	default:
		break;
	}
}
