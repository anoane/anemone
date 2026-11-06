/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_trunk_relaxed.cu: the trunk products of class 80 (WEIGHT_FORMAT_E4M3_SCALED weights) in the
 * relaxed order, in a wide form (tiles of 64 or 128 rows, operands staged in shared memory) and a decode form (1 to 8
 * rows, the input width cut into slices whose sums are added in a fixed order); and the reference order in the wide
 * form's tiles (the ordered instance). Device code only, built for sm_80 into the module "kernel_trunk_relaxed".
 *
 * Contract: include/anemone/relaxed_linear_records.h (the order of additions, the fold, the records, the launches);
 * kernel_linear_common.cuh (the instruction, the placement of a code, the rounding). Here:
 *   - Wide form. A block computes 128 outputs (8 tiles of 16, in one matrix) by 64 or 128 rows; warp w takes tiles
 *     2 (w % 4) and 2 (w % 4) + 1, one block of 32 outputs and so one scale code a block of 32 inputs, by the 64
 *     rows w / 4. The weights are operand A as stored (a lane's 16 bytes of a tile's block of 32 inputs are its part
 *     of two instructions), the rows the columns of operand B. Each load of 64 inputs is one stage: cp.async copies
 *     the 8 tiles' 1,024 bytes each and every row's 64 operands (rows past row_count are filled with zeros, computed
 *     and stored nowhere) into one of RELAXED_WIDE_STAGES buffers, RELAXED_WIDE_STAGES - 1 loads
 *     ahead of the one worked on. A row's 128 bytes lie RELAXED_STAGE_ROW_PITCH apart, so the 16-byte reads
 *     of 8 rows by 4 lane places fall in distinct banks. The scale codes of the block's 4 blocks of 32 outputs are
 *     read into shared memory once, before the first stage.
 *   - Each warp places its tiles' codes of a block of 32 inputs once (16 registers), folds their scale (relaxed:
 *     one fma.rn.bf16x2 or .f16x2 by a power of two and -0 a register, exact by relaxed_linear_records.h's check),
 *     and runs them against its 8 groups of 8 rows. Relaxed: two instructions a tile and group add to the group's
 *     accumulators. Ordered: they sum on a zero accumulator and the block's term is added to the chain with its
 *     power, chain = fma(p, 2^e, chain), class 80's carried form exactly as linear_slice_chains has it.
 *   - Rows that are not quantized are copied as bfloat16, turned into 16-bit floats in shared memory by the block
 *     (linear_place_bfloat16: the reference's conversion) between the arrival of a stage and its use, and read as
 *     operand B by ldmatrix.
 *   - Decode form. A block is one tile of 16 outputs and S warps, warp s its slice of the loads (linear_slice_start);
 *     a lane asks for its tile's 32 bytes of weights, its scale codes and its column's 32 bytes of operands one load
 *     ahead, folds and accumulates as the wide form does; the S accumulators of a result are added from +0 in
 *     ascending slice order through shared memory. Column c carries row c, a column from row_count on row 0 again
 *     (stored nowhere): 1 to 8 rows cost the same instructions.
 *   - No atomic operation; every thread of a block passes the same barriers.
 */
#include "anemone/model_layout.h"
#include "anemone/relaxed_linear_records.h"
#include "anemone/step_records.h"
#include "kernel_linear_common.cuh"

#define WIDE_TILES (RELAXED_WIDE_OUTPUTS / OUTPUTS_PER_TILE)
#define WARPS_ALONG_OUTPUTS RELAXED_WIDE_WARPS_ALONG_OUTPUTS
#define WARP_TILES    \
	(WIDE_TILES / \
	 WARPS_ALONG_OUTPUTS) /* one block of 32 outputs: one scale code */
#define GROUP_ROWS LINEAR_MAXIMUM_ROWS /* the columns of an instruction */
#define WARP_ROW_GROUPS (RELAXED_WIDE_WARP_ROWS / GROUP_ROWS)
#define WARPS_ALONG_ROWS_SMALL \
	(RELAXED_WIDE_ROWS_SMALL / RELAXED_WIDE_WARP_ROWS)
#define WARPS_ALONG_ROWS_LARGE \
	(RELAXED_WIDE_ROWS_LARGE / RELAXED_WIDE_WARP_ROWS)
#define WIDE_THREADS_SMALL \
	(WARPS_ALONG_OUTPUTS * WARPS_ALONG_ROWS_SMALL * THREADS_PER_WARP)
#define WIDE_THREADS_LARGE \
	(WARPS_ALONG_OUTPUTS * WARPS_ALONG_ROWS_LARGE * THREADS_PER_WARP)
/* The launch bounds: three blocks of 64 rows a multiprocessor (what its shared memory holds; 168 registers), one of
 * 128 rows (in 128 registers, two of them would spill: assembler reports, 26 October). */
#define ROW_TILE_CHUNK \
	16u /* tiles of rows of a chunk of the block order (relaxed_block_place_of) */
#define WIDE_BLOCKS_SMALL 3u
#define WIDE_BLOCKS_LARGE 1u
#define WIDE_BLOCKS_ORDERED_SMALL \
	2u /* the ordered chains: about 215 registers */
/* the groups of copies still in flight when a stage is used: the ones after it (RELAXED_WIDE_STAGES - 2) */
#define COPIES_STILL_PENDING (RELAXED_WIDE_STAGES - 2u)
#define CHUNK_BYTES 16u /* one cp.async */
#define GROUP_BYTES \
	(THREADS_PER_WARP * CHUNK_BYTES) /* a tile's block of 32 inputs */
#define TILE_CHUNKS (RELAXED_STAGE_TILE_BYTES / CHUNK_BYTES)
#define LOAD_ROW_BYTES \
	(VALUES_PER_LOAD * LINEAR_BFLOAT16_BYTES) /* a row's 64 inputs */
#define ROW_CHUNKS (LOAD_ROW_BYTES / CHUNK_BYTES)
#define STAGE_WEIGHT_BYTES (WIDE_TILES * RELAXED_STAGE_TILE_BYTES)
#define LANE_PLACE_FED_BYTES \
	(OPERAND_LOADS *     \
	 CHUNK_BYTES) /* a lane place's carried operands of one load */
#define MATRIX_ROWS 8u /* of an 8-by-8 matrix of ldmatrix */
#define MATRIX_ROW_HALVES 8u
#define BFLOAT16_FIELD_SHIFT 7u
#define HALF_FIELD_SHIFT 10u
#define BFLOAT16_BIAS 127
#define HALF_BIAS 15
#define FLOAT32_BIAS 127
#define E4M3_BFLOAT16_PLACEMENT \
	120 /* a placed E4M3 byte is its value times 2^-120 as a bfloat16 */
#define CARRIED_OPERAND_EXPONENT \
	100 /* a carried operand is its value times 2^100 */
#define NEGATIVE_ZEROS \
	0x80008000u /* -0 in both halves: what the fold adds, which changes no product */
#define BOTH_HALVES 0x00010001u
/* The fold of quantized rows: the bfloat16 field of 2^(w - reference + 120), and the float32 field of the power
 * that undoes the reference and the operands' 2^100. Of bfloat16 rows: the 16-bit float field of 2^(w - reference +
 * 8) is w less the reference plus this; the output's float32 field is the reference itself. */
#define FOLD_FIELD_QUANTIZED \
	(E4M3_BFLOAT16_PLACEMENT + BFLOAT16_BIAS - RELAXED_FOLD_REFERENCE)
#define OUTPUT_FIELD_QUANTIZED \
	(RELAXED_FOLD_REFERENCE - CARRIED_OPERAND_EXPONENT)
#define FOLD_FIELD_HALF_ROWS (E4M3_PLACEMENT_EXPONENT + HALF_BIAS)

enum relaxed_order { RELAXED_ORDER = 0, REFERENCE_ORDER = 1 };
enum relaxed_rows { QUANTIZED_ROWS = 0, BFLOAT16_ROWS = 1 };

/* What a block of the wide form works on, from either record. */
struct relaxed_wide_plan {
	const unsigned char *matrix; /* of the block's tiles */
	const unsigned char *
		rows; /* row 0's first input of the block: operands, or values of its segment */
	uint16_t *output; /* values [row_count][output_size] of the block's matrix */
	float *partials; /* [k_split_count][row_count][partial_outputs] */
	size_t row_bytes; /* from one row of `rows` to the next */
	uint32_t output_size;
	uint32_t matrix_tile; /* the block's first tile in its matrix */
	uint32_t partial_first_output; /* the block's first output among all outputs of the launch */
	uint32_t partial_outputs;
	uint32_t input_size;
	uint32_t row_count;
	uint32_t k_split_count;
	int32_t fold_field; /* the field of the fold's power of two less the scale code */
	float output_scale;
};

/* cp.async of 16 bytes into shared memory, of which source_bytes (16, or 0 for zeros) are read. */
__device__ static __forceinline__ void
relaxed_copy_chunk(void *shared_destination, const void *source,
		   uint32_t source_bytes)
{
	const uint32_t destination =
		(uint32_t)__cvta_generic_to_shared(shared_destination);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
			     destination),
		     "l"(source), "r"(source_bytes)
		     : "memory");
}

__device__ static __forceinline__ void relaxed_copies_commit(void)
{
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

/* Waits until at most `pending` committed groups of this thread's copies are still in flight. */
template <uint32_t pending>
__device__ __forceinline__ void relaxed_copies_wait(void)
{
	asm volatile("cp.async.wait_group %0;\n" ::"n"(pending) : "memory");
}

/* Both halves of a register times both halves of `power` plus -0: exact for a power of two (the fold). */
template <int rows>
__device__ __forceinline__ uint32_t relaxed_fold(uint32_t placed,
						 uint32_t power)
{
	uint32_t folded = 0;
	if (rows == QUANTIZED_ROWS) {
		asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
		    : "=r"(folded)
		    : "r"(placed), "r"(power), "r"(NEGATIVE_ZEROS));
	} else {
		asm("fma.rn.f16x2 %0, %1, %2, %3;\n"
		    : "=r"(folded)
		    : "r"(placed), "r"(power), "r"(NEGATIVE_ZEROS));
	}
	return folded;
}

/* The fold's power of two of scale code `code` in both halves: bfloat16 or 16-bit float. */
template <int rows>
__device__ __forceinline__ uint32_t relaxed_fold_power(uint32_t code,
						       int32_t fold_field)
{
	const uint32_t field = (uint32_t)((int32_t)code + fold_field);
	return (field << (rows == QUANTIZED_ROWS ? BFLOAT16_FIELD_SHIFT :
						   HALF_FIELD_SHIFT)) *
	       BOTH_HALVES;
}

/* A lane's 16 bytes of a tile's block of 32 inputs as the registers of operand A of its two instructions, placed for
 * the rows' form, and in the relaxed order folded by `power`. */
template <int rows, int order>
__device__ __forceinline__ void
relaxed_place_block(uint4 stored, uint32_t power,
		    uint32_t weights[STEPS_PER_BLOCK][REGISTERS_PER_OPERAND_A])
{
	const uint32_t words[WORDS_PER_PART] = { stored.x, stored.y, stored.z,
						 stored.w };
	const uint32_t bit = rows == QUANTIZED_ROWS ? E4M3_BFLOAT16_BIT :
						      E4M3_HALF_BIT;
#pragma unroll
	for (uint32_t step = 0; step < STEPS_PER_BLOCK; step++) {
		linear_place_e4m3(words[step * WORDS_PER_STEP], bit,
				  &weights[step][LOW_OUTPUT_LOW_INPUTS],
				  &weights[step][HIGH_OUTPUT_LOW_INPUTS]);
		linear_place_e4m3(words[step * WORDS_PER_STEP + 1u], bit,
				  &weights[step][LOW_OUTPUT_HIGH_INPUTS],
				  &weights[step][HIGH_OUTPUT_HIGH_INPUTS]);
#pragma unroll
		for (uint32_t operand_register = 0;
		     order == RELAXED_ORDER &&
		     operand_register < REGISTERS_PER_OPERAND_A;
		     operand_register++) {
			weights[step][operand_register] = relaxed_fold<rows>(
				weights[step][operand_register], power);
		}
	}
}

/* One stage of the wide form: the block's 8 tiles' weights of load `load` and its rows' 64 inputs there; a row past
 * row_count is filled with zeros. Every thread copies whole passes of chunks: no bound is tested per chunk. A thread
 * takes the chunk of its index and those threads further: threads / 64 tiles, threads / 8 rows. */
template <uint32_t row_tile, uint32_t threads>
__device__ __forceinline__ void
relaxed_stage_copy(const struct relaxed_wide_plan *plan, uint32_t first_row,
		   uint32_t load, unsigned char *stage)
{
	constexpr uint32_t weight_passes = WIDE_TILES * TILE_CHUNKS / threads;
	constexpr uint32_t row_passes = row_tile * ROW_CHUNKS / threads;
	constexpr uint32_t pass_tiles = threads / TILE_CHUNKS;
	constexpr uint32_t pass_rows = threads / ROW_CHUNKS;
	static_assert(weight_passes * threads == WIDE_TILES * TILE_CHUNKS &&
			      row_passes * threads == row_tile * ROW_CHUNKS &&
			      pass_tiles * TILE_CHUNKS == threads,
		      "every thread copies whole passes");
	const size_t group_count = plan->input_size / VALUES_PER_BLOCK;
	const uint32_t tile = threadIdx.x / TILE_CHUNKS,
		       part = threadIdx.x % TILE_CHUNKS;
	const unsigned char *const weights =
		plan->matrix +
		((size_t)(plan->matrix_tile + tile) * group_count +
		 (size_t)load * BLOCKS_PER_LOAD) *
			GROUP_BYTES +
		(size_t)part * CHUNK_BYTES;
#pragma unroll
	for (uint32_t pass = 0; pass < weight_passes; pass++) {
		relaxed_copy_chunk(stage +
					   (tile + pass * pass_tiles) *
						   RELAXED_STAGE_TILE_BYTES +
					   part * CHUNK_BYTES,
				   weights + pass * pass_tiles * group_count *
						     GROUP_BYTES,
				   CHUNK_BYTES);
	}
	const uint32_t row = threadIdx.x / ROW_CHUNKS,
		       row_part = threadIdx.x % ROW_CHUNKS;
	const unsigned char *const rows = plan->rows +
					  (size_t)load * LOAD_ROW_BYTES +
					  (size_t)row_part * CHUNK_BYTES;
#pragma unroll
	for (uint32_t pass = 0; pass < row_passes; pass++) {
		const uint32_t tile_row = row + pass * pass_rows;
		const bool is_live = first_row + tile_row < plan->row_count;
		relaxed_copy_chunk(
			stage + STAGE_WEIGHT_BYTES +
				tile_row * RELAXED_STAGE_ROW_PITCH +
				row_part * CHUNK_BYTES,
			rows + (is_live ? (size_t)(first_row + tile_row) : 0) *
					plan->row_bytes,
			is_live ? CHUNK_BYTES : 0u);
	}
}

/* The rows of a stage of bfloat16 values as 16-bit floats, in place: the reference's conversion of each value. */
template <uint32_t row_tile, uint32_t threads>
__device__ __forceinline__ void relaxed_stage_to_half(unsigned char *stage)
{
	constexpr uint32_t row_passes = row_tile * ROW_CHUNKS / threads;
	unsigned char *const stage_rows = stage + STAGE_WEIGHT_BYTES;
#pragma unroll
	for (uint32_t pass = 0; pass < row_passes; pass++) {
		const uint32_t chunk = threadIdx.x + pass * threads;
		uint4 *const place =
			(uint4 *)(stage_rows +
				  (size_t)(chunk / ROW_CHUNKS) *
					  RELAXED_STAGE_ROW_PITCH +
				  (size_t)(chunk % ROW_CHUNKS) * CHUNK_BYTES);
		const uint4 pairs = *place;
		*place = make_uint4(linear_place_bfloat16(pairs.x),
				    linear_place_bfloat16(pairs.y),
				    linear_place_bfloat16(pairs.z),
				    linear_place_bfloat16(pairs.w));
	}
}

/* Operand B of a group of 8 rows for the two instructions of block `block` of a stage: the carried operands as
 * stored (a lane place's 16 bytes), or the 16-bit floats by ldmatrix (four 8-by-8 matrices, inputs 0-7, 8-15, 16-23
 * and 24-31 of the block). */
template <int rows>
__device__ __forceinline__ uint4
relaxed_group_fed(const unsigned char *group_rows, uint32_t block)
{
	const uint32_t lane_index = threadIdx.x % THREADS_PER_WARP;
	if (rows == QUANTIZED_ROWS) {
		return *(const uint4 *)(group_rows +
					(size_t)(lane_index /
						 LANES_PER_LANE_GROUP) *
						RELAXED_STAGE_ROW_PITCH +
					(size_t)(lane_index %
						 LANES_PER_LANE_GROUP) *
						LANE_PLACE_FED_BYTES +
					(size_t)block * CHUNK_BYTES);
	}
	uint4 fed;
	const uint32_t address = (uint32_t)__cvta_generic_to_shared(
		group_rows +
		(size_t)(lane_index % MATRIX_ROWS) * RELAXED_STAGE_ROW_PITCH +
		((size_t)block * VALUES_PER_BLOCK +
		 (size_t)(lane_index / MATRIX_ROWS) * MATRIX_ROW_HALVES) *
			LINEAR_BFLOAT16_BYTES);
	asm volatile(
		"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
		: "=r"(fed.x), "=r"(fed.y), "=r"(fed.z), "=r"(fed.w)
		: "r"(address));
	return fed;
}

/* Where a block of the wide form works: its tile of rows and its block of 128 outputs. The card starts blocks in
 * the order of their index (x first); a chunk of ROW_TILE_CHUNK tiles of rows takes every block of outputs in turn,
 * its tiles of rows adjacent, so that the rows of a chunk (1,024 or fewer) stay in the L2 while the weights stream
 * once a chunk. */
struct relaxed_block_place {
	uint32_t row_tile;
	uint32_t output_block;
};
__device__ static __forceinline__ struct relaxed_block_place
relaxed_block_place_of(void)
{
	const uint32_t row_tiles = gridDim.x, output_blocks = gridDim.y;
	if (row_tiles <= ROW_TILE_CHUNK) { /* one chunk: the grid's own order */
		return (struct relaxed_block_place){ blockIdx.x, blockIdx.y };
	}
	const uint32_t linear = blockIdx.y * row_tiles + blockIdx.x;
	const uint32_t chunk = linear / (ROW_TILE_CHUNK * output_blocks);
	const uint32_t in_chunk =
		linear - chunk * ROW_TILE_CHUNK * output_blocks;
	const uint32_t chunk_tiles =
		min(ROW_TILE_CHUNK, row_tiles - chunk * ROW_TILE_CHUNK);
	return (struct relaxed_block_place){ chunk * ROW_TILE_CHUNK +
						     in_chunk % chunk_tiles,
					     in_chunk / chunk_tiles };
}

/* A block of the wide form: see the top. */
template <uint32_t warps_along_rows, uint32_t warp_row_groups, int order,
	  int rows>
__device__ __forceinline__ void
relaxed_wide_block(const struct relaxed_wide_plan *plan,
		   uint32_t row_tile_index)
{
	constexpr uint32_t row_tile =
		warps_along_rows * warp_row_groups * GROUP_ROWS;
	constexpr uint32_t threads =
		WARPS_ALONG_OUTPUTS * warps_along_rows * THREADS_PER_WARP;
	constexpr uint32_t stage_bytes =
		STAGE_WEIGHT_BYTES + row_tile * RELAXED_STAGE_ROW_PITCH;
	constexpr int form = rows == QUANTIZED_ROWS ? LINEAR_FORM_CARRIED :
						      LINEAR_FORM_BFLOAT16_ROWS;
	extern __shared__ uint4 relaxed_shared[];
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane_index = threadIdx.x % THREADS_PER_WARP;
	const uint32_t warp_tiles = warp % WARPS_ALONG_OUTPUTS * WARP_TILES,
		       warp_rows = warp / WARPS_ALONG_OUTPUTS;
	const uint32_t group_count = plan->input_size / VALUES_PER_BLOCK,
		       load_count = plan->input_size / VALUES_PER_LOAD;
	const uint32_t first_row = row_tile_index * row_tile,
		       split = blockIdx.z;
	const uint32_t first_load = split * load_count / plan->k_split_count;
	const uint32_t end_load =
		(split + 1u) * load_count / plan->k_split_count;
	unsigned char *const scales = (unsigned char *)relaxed_shared;
	unsigned char *const stages =
		scales + (RELAXED_SCALE_BLOCKS * group_count +
			  RELAXED_SCALE_ALIGNMENT - 1u) /
				 RELAXED_SCALE_ALIGNMENT *
				 RELAXED_SCALE_ALIGNMENT;
	const unsigned char *const block_scales =
		plan->matrix + (size_t)plan->output_size * plan->input_size +
		(size_t)(plan->matrix_tile / TILES_PER_SCALE_BLOCK) *
			group_count;
	for (uint32_t index = threadIdx.x;
	     index < RELAXED_SCALE_BLOCKS * group_count; index += threads) {
		scales[index] = block_scales[index];
	}
	const bool is_full = first_row + row_tile <= plan->row_count;
#pragma unroll
	for (uint32_t ahead = 0; ahead + 1u < RELAXED_WIDE_STAGES; ahead++) {
		if (first_load + ahead < end_load) {
			relaxed_stage_copy<row_tile, threads>(
				plan, first_row, first_load + ahead,
				stages + ahead * stage_bytes);
		}
		relaxed_copies_commit();
	}
	float sums[WARP_TILES][warp_row_groups][ELEMENTS_PER_LANE] = {};
	const unsigned char *const warp_scales =
		scales +
		(size_t)(warp_tiles / TILES_PER_SCALE_BLOCK) * group_count;
	for (uint32_t load = first_load; load < end_load; load++) {
		const uint32_t stage_index =
			(load - first_load) % RELAXED_WIDE_STAGES;
		unsigned char *const stage = stages + stage_index * stage_bytes;
		relaxed_copies_wait<COPIES_STILL_PENDING>();
		__syncthreads(); /* the stage is here for every thread; the one copied next is no longer read */
		const uint32_t next = load + RELAXED_WIDE_STAGES - 1u;
		if (next < end_load) {
			relaxed_stage_copy<row_tile, threads>(
				plan, first_row, next,
				stages + (next - first_load) %
						 RELAXED_WIDE_STAGES *
						 stage_bytes);
		}
		relaxed_copies_commit();
		if (rows == BFLOAT16_ROWS) {
			relaxed_stage_to_half<row_tile, threads>(stage);
			__syncthreads();
		}
		const unsigned char *const group_rows =
			stage + STAGE_WEIGHT_BYTES +
			(size_t)warp_rows * warp_row_groups * GROUP_ROWS *
				RELAXED_STAGE_ROW_PITCH;
#pragma unroll
		for (uint32_t block = 0; block < BLOCKS_PER_LOAD; block++) {
			const uint32_t code =
				warp_scales[load * BLOCKS_PER_LOAD + block];
			const uint32_t power = relaxed_fold_power<rows>(
				code, plan->fold_field);
			/* the reference order's power of the block (linear_block_power): carried, or bfloat16 rows */
			const float block_power =
				rows == QUANTIZED_ROWS ?
					__uint_as_float(
						(uint32_t)min(
							(int32_t)code +
								E4M3_CARRIED_FIELD,
							FLOAT32_LARGEST_EXPONENT_FIELD)
						<< FLOAT32_EXPONENT_SHIFT) :
					linear_power_of_two(
						(int32_t)code +
						E4M3_PLACEMENT_EXPONENT);
			uint32_t weights[WARP_TILES][STEPS_PER_BLOCK]
					[REGISTERS_PER_OPERAND_A];
#pragma unroll
			for (uint32_t tile = 0; tile < WARP_TILES; tile++) {
				const uint4 stored = ((
					const uint4
						*)(stage +
						   (size_t)(warp_tiles + tile) *
							   RELAXED_STAGE_TILE_BYTES +
						   (size_t)block * GROUP_BYTES))
					[lane_index];
				relaxed_place_block<rows, order>(stored, power,
								 weights[tile]);
			}
#pragma unroll
			for (uint32_t group = 0; group < warp_row_groups;
			     group++) {
				const uint4 fed = relaxed_group_fed<rows>(
					group_rows +
						(size_t)group * GROUP_ROWS *
							RELAXED_STAGE_ROW_PITCH,
					block);
#pragma unroll
				for (uint32_t tile = 0; tile < WARP_TILES;
				     tile++) {
					if (order == RELAXED_ORDER) {
						linear_tensor_core<form>(
							sums[tile][group],
							weights[tile][0], fed.x,
							fed.y);
						linear_tensor_core<form>(
							sums[tile][group],
							weights[tile][1], fed.z,
							fed.w);
					} else {
						float term[ELEMENTS_PER_LANE] = {
							0.0f, 0.0f, 0.0f, 0.0f
						};
						linear_tensor_core<form>(
							term, weights[tile][0],
							fed.x, fed.y);
						linear_tensor_core<form>(
							term, weights[tile][1],
							fed.z, fed.w);
#pragma unroll
						for (uint32_t element = 0;
						     element <
						     ELEMENTS_PER_LANE;
						     element++) {
							sums[tile][group]
							    [element] = fmaf(
								    term[element],
								    block_power,
								    sums[tile]
									[group]
									[element]);
						}
					}
				}
			}
		}
	}
	/* the results: lane 4 g + t holds outputs g and g + 8 of each tile for rows 2 t and 2 t + 1 of each group */
#pragma unroll
	for (uint32_t tile = 0; tile < WARP_TILES; tile++) {
#pragma unroll
		for (uint32_t group = 0; group < warp_row_groups; group++) {
#pragma unroll
			for (uint32_t element = 0; element < ELEMENTS_PER_LANE;
			     element++) {
				const uint32_t row =
					first_row +
					warp_rows * warp_row_groups *
						GROUP_ROWS +
					group * GROUP_ROWS +
					lane_index % LANES_PER_LANE_GROUP *
						COLUMNS_PER_LANE +
					element % COLUMNS_PER_LANE;
				const uint32_t tile_output =
					(warp_tiles + tile) * OUTPUTS_PER_TILE +
					lane_index / LANES_PER_LANE_GROUP +
					element / COLUMNS_PER_LANE *
						OUTPUTS_PER_HALF_TILE;
				const float sum = sums[tile][group][element];
				if (!is_full && row >= plan->row_count) {
					continue;
				}
				if (order == REFERENCE_ORDER) {
					plan->output[(size_t)row *
							     plan->output_size +
						     plan->matrix_tile *
							     OUTPUTS_PER_TILE +
						     tile_output] =
						(uint16_t)linear_bfloat16_bits(
							0.0f + sum);
				} else if (plan->k_split_count == 1u) {
					plan->output[(size_t)row *
							     plan->output_size +
						     plan->matrix_tile *
							     OUTPUTS_PER_TILE +
						     tile_output] =
						(uint16_t)linear_bfloat16_bits(
							sum *
							plan->output_scale);
				} else {
					plan->partials
						[((size_t)split *
							  plan->row_count +
						  row) * plan->partial_outputs +
						 plan->partial_first_output +
						 tile_output] = sum;
				}
			}
		}
	}
}

/* The plan of a block of quantized rows: the block's 8 tiles lie in matrix 0 or 1. */
__device__ static __forceinline__ struct relaxed_wide_plan
relaxed_quantized_plan(const struct trunk_relaxed_quantized_arguments *arguments,
		       unsigned char *arena, uint32_t first_tile)
{
	const uint32_t first_tiles =
		arguments->output_size[0] / OUTPUTS_PER_TILE;
	const bool is_second =
		first_tile >=
		first_tiles; /* selected, not indexed: the record stays in registers */
	struct relaxed_wide_plan plan;
	plan.matrix = arena + (is_second ? arguments->weights_offset[1] :
					   arguments->weights_offset[0]);
	plan.rows = arena + arguments->operands_offset;
	plan.output =
		(uint16_t *)(arena + (is_second ? arguments->output_offset[1] :
						  arguments->output_offset[0]));
	plan.partials = (float *)(arena + arguments->partials_offset);
	plan.row_bytes = (size_t)arguments->input_size * LINEAR_BFLOAT16_BYTES;
	plan.output_size = is_second ? arguments->output_size[1] :
				       arguments->output_size[0];
	plan.matrix_tile = is_second ? first_tile - first_tiles : first_tile;
	plan.partial_first_output = first_tile * OUTPUTS_PER_TILE;
	plan.partial_outputs =
		arguments->output_size[0] + arguments->output_size[1];
	plan.input_size = arguments->input_size;
	plan.row_count = arguments->row_count;
	plan.k_split_count = arguments->k_split_count;
	plan.fold_field = FOLD_FIELD_QUANTIZED;
	plan.output_scale = __uint_as_float((uint32_t)OUTPUT_FIELD_QUANTIZED
					    << FLOAT32_EXPONENT_SHIFT);
	return plan;
}

/* The plan of a block of bfloat16 rows: its segment's values. */
__device__ static __forceinline__ struct relaxed_wide_plan
relaxed_bfloat16_plan(const struct trunk_relaxed_bfloat16_arguments *arguments,
		      unsigned char *arena, uint32_t first_tile)
{
	const uint32_t segment_tiles = arguments->output_size /
				       OUTPUTS_PER_TILE /
				       arguments->segment_count;
	struct relaxed_wide_plan plan;
	plan.matrix = arena + arguments->weights_offset;
	plan.rows = arena + arguments->values_offset +
		    (size_t)(first_tile / segment_tiles) *
			    arguments->input_size * LINEAR_BFLOAT16_BYTES;
	plan.output = (uint16_t *)(arena + arguments->output_offset);
	plan.partials = (float *)(arena + arguments->partials_offset);
	plan.row_bytes = (size_t)arguments->segment_count *
			 arguments->input_size * LINEAR_BFLOAT16_BYTES;
	plan.output_size = arguments->output_size;
	plan.matrix_tile = first_tile;
	plan.partial_first_output = first_tile * OUTPUTS_PER_TILE;
	plan.partial_outputs = arguments->output_size;
	plan.input_size = arguments->input_size;
	plan.row_count = arguments->row_count;
	plan.k_split_count = arguments->k_split_count;
	plan.fold_field = FOLD_FIELD_HALF_ROWS - arguments->fold_reference;
	plan.output_scale = __uint_as_float((uint32_t)arguments->fold_reference
					    << FLOAT32_EXPONENT_SHIFT);
	return plan;
}

/* cone_layer (relaxed_linear_records.h): true where every row of the block's tile of rows skips at that decoder layer
 * (each thread reads one row's flags; the whole block returns together, before any of its barriers) */
template <uint32_t warps_along_rows>
__device__ static __forceinline__ bool relaxed_cone_skips(uint64_t step_offset, uint32_t cone_layer,
							  const unsigned char *arena, uint32_t row_tile_index,
							  uint32_t row_count)
{
	constexpr uint32_t row_tile = warps_along_rows * WARP_ROW_GROUPS * GROUP_ROWS;
	static_assert(WARPS_ALONG_OUTPUTS * warps_along_rows * THREADS_PER_WARP >= row_tile, "a thread a row");
	if (cone_layer == 0)
		return false;
	const struct step_row_record *const rows =
		(const struct step_row_record *)(arena + step_offset +
						 STEP_BLOCK_BYTES_VALUE);
	const uint32_t row = row_tile_index * row_tile + threadIdx.x;
	const uint32_t decoder_layer = cone_layer - MODEL_DECODER_FIRST_LAYER;
	const bool is_read =
		threadIdx.x < row_tile && row < row_count &&
		!STEP_ROW_CONE_TRUNK_SKIPS(rows[row].flags, decoder_layer);
	return __syncthreads_or(is_read) == 0;
}

#define RELAXED_WIDE_INSTANCES(rows_name, warps_along_rows, threads, blocks,  \
			       other_blocks)                                  \
	extern "C" __global__ void __launch_bounds__(threads, blocks)         \
		trunk_relaxed_wide_quantized##rows_name(                      \
			struct trunk_relaxed_quantized_arguments arguments,   \
			unsigned char *arena)                                 \
	{                                                                     \
		const struct relaxed_block_place place =                      \
			relaxed_block_place_of();                             \
		if (relaxed_cone_skips<warps_along_rows>(                     \
			    arguments.step_offset, arguments.cone_layer,      \
			    arena, place.row_tile, arguments.row_count))      \
			return;                                               \
		const struct relaxed_wide_plan plan = relaxed_quantized_plan( \
			&arguments, arena, place.output_block * WIDE_TILES);  \
		relaxed_wide_block<warps_along_rows, WARP_ROW_GROUPS,         \
				   RELAXED_ORDER, QUANTIZED_ROWS>(            \
			&plan, place.row_tile);                               \
	}                                                                     \
	extern "C" __global__ void __launch_bounds__(threads, other_blocks)   \
		trunk_ordered_wide_quantized##rows_name(                      \
			struct trunk_relaxed_quantized_arguments arguments,   \
			unsigned char *arena)                                 \
	{                                                                     \
		const struct relaxed_block_place place =                      \
			relaxed_block_place_of();                             \
		if (relaxed_cone_skips<warps_along_rows>(                     \
			    arguments.step_offset, arguments.cone_layer,      \
			    arena, place.row_tile, arguments.row_count))      \
			return;                                               \
		const struct relaxed_wide_plan plan = relaxed_quantized_plan( \
			&arguments, arena, place.output_block * WIDE_TILES);  \
		relaxed_wide_block<warps_along_rows, WARP_ROW_GROUPS,         \
				   REFERENCE_ORDER, QUANTIZED_ROWS>(          \
			&plan, place.row_tile);                               \
	}                                                                     \
	extern "C" __global__ void __launch_bounds__(threads, other_blocks)   \
		trunk_relaxed_wide_bfloat16##rows_name(                       \
			struct trunk_relaxed_bfloat16_arguments arguments,    \
			unsigned char *arena)                                 \
	{                                                                     \
		const struct relaxed_block_place place =                      \
			relaxed_block_place_of();                             \
		if (relaxed_cone_skips<warps_along_rows>(                     \
			    arguments.step_offset, arguments.cone_layer,      \
			    arena, place.row_tile, arguments.row_count))      \
			return;                                               \
		const struct relaxed_wide_plan plan = relaxed_bfloat16_plan(  \
			&arguments, arena, place.output_block * WIDE_TILES);  \
		relaxed_wide_block<warps_along_rows, WARP_ROW_GROUPS,         \
				   RELAXED_ORDER, BFLOAT16_ROWS>(             \
			&plan, place.row_tile);                               \
	}                                                                     \
	extern "C" __global__ void __launch_bounds__(threads, other_blocks)   \
		trunk_ordered_wide_bfloat16##rows_name(                       \
			struct trunk_relaxed_bfloat16_arguments arguments,    \
			unsigned char *arena)                                 \
	{                                                                     \
		const struct relaxed_block_place place =                      \
			relaxed_block_place_of();                             \
		if (relaxed_cone_skips<warps_along_rows>(                     \
			    arguments.step_offset, arguments.cone_layer,      \
			    arena, place.row_tile, arguments.row_count))      \
			return;                                               \
		const struct relaxed_wide_plan plan = relaxed_bfloat16_plan(  \
			&arguments, arena, place.output_block * WIDE_TILES);  \
		relaxed_wide_block<warps_along_rows, WARP_ROW_GROUPS,         \
				   REFERENCE_ORDER, BFLOAT16_ROWS>(           \
			&plan, place.row_tile);                               \
	}
RELAXED_WIDE_INSTANCES(_64, WARPS_ALONG_ROWS_SMALL, WIDE_THREADS_SMALL,
		       WIDE_BLOCKS_SMALL, WIDE_BLOCKS_ORDERED_SMALL)
RELAXED_WIDE_INSTANCES(_128, WARPS_ALONG_ROWS_LARGE, WIDE_THREADS_LARGE,
		       WIDE_BLOCKS_LARGE, WIDE_BLOCKS_LARGE)
#undef RELAXED_WIDE_INSTANCES

/* ---- the decode form ---- */

/* What a lane reads for one load: its tile's 16 bytes of each block of 32 inputs, the two scale codes, its column's
 * 16 bytes of each block (carried operands: a lane place's; bfloat16 values: inputs 2 t, 2 t + 1, 2 t + 8 and 2 t + 9
 * of each of the 4 instructions, as 4 pairs). */
struct relaxed_decode_load {
	uint4 weights[BLOCKS_PER_LOAD];
	uint4 fed[BLOCKS_PER_LOAD];
	uint32_t codes;
};

/* What a lane of the decode form reads, as pointers at load 0. */
struct relaxed_decode_lane {
	const uint4 *
		weights; /* its 16 bytes of the tile's first block of 32 inputs */
	const unsigned char *scales; /* the scale code of that block */
	const unsigned char *
		fed; /* its column's first input: a lane place's operands, or the value 2 t */
};

template <int rows>
__device__ __forceinline__ struct relaxed_decode_load
relaxed_decode_fetch(const struct relaxed_decode_lane *lane, uint32_t load)
{
	struct relaxed_decode_load fetched;
#pragma unroll
	for (uint32_t block = 0; block < BLOCKS_PER_LOAD; block++) {
		fetched.weights[block] =
			lane->weights[((size_t)load * BLOCKS_PER_LOAD + block) *
				      THREADS_PER_WARP];
	}
	fetched.codes = ((const uint16_t *)lane->scales)[load];
	if (rows == QUANTIZED_ROWS) {
		const uint4 *const parts =
			(const uint4 *)(lane->fed +
					(size_t)load * LOAD_ROW_BYTES);
		fetched.fed[0] = parts[0];
		fetched.fed[1] = parts[1];
	} else {
		const uint32_t *const pairs =
			(const uint32_t *)(lane->fed +
					   (size_t)load * LOAD_ROW_BYTES);
		/* the 8 registers, as the 4 instructions of the load take them */
		uint32_t *const words = (uint32_t *)fetched.fed;
#pragma unroll
		for (uint32_t step = 0; step < STEPS_PER_LOAD; step++) {
			words[step * WORDS_PER_STEP] =
				pairs[step * PAIRS_PER_STEP];
			words[step * WORDS_PER_STEP + 1u] =
				pairs[step * PAIRS_PER_STEP +
				      PAIRS_PER_HALF_STEP];
		}
	}
	return fetched;
}

/* A block of the decode form for one tile: see the top. */
template <int rows>
__device__ __forceinline__ void
relaxed_decode_block(const struct relaxed_wide_plan *plan,
		     const struct relaxed_decode_lane *lane)
{
	constexpr int form = rows == QUANTIZED_ROWS ? LINEAR_FORM_CARRIED :
						      LINEAR_FORM_BFLOAT16_ROWS;
	extern __shared__ float decode_sums[]; /* [slice][lane][element] */
	const uint32_t slice_count = blockDim.x / THREADS_PER_WARP,
		       slice = threadIdx.x / THREADS_PER_WARP;
	const uint32_t load_count = plan->input_size / VALUES_PER_LOAD;
	const uint32_t first_load =
		linear_slice_start(slice, slice_count, load_count);
	const uint32_t end_load =
		linear_slice_start(slice + 1u, slice_count, load_count);
	float sums[ELEMENTS_PER_LANE] = { 0.0f, 0.0f, 0.0f, 0.0f };
	struct relaxed_decode_load ahead = {};
	if (first_load < end_load) {
		ahead = relaxed_decode_fetch<rows>(lane, first_load);
	}
	for (uint32_t load = first_load; load < end_load; load++) {
		const struct relaxed_decode_load current = ahead;
		if (load + 1u < end_load) {
			ahead = relaxed_decode_fetch<rows>(lane, load + 1u);
		}
#pragma unroll
		for (uint32_t block = 0; block < BLOCKS_PER_LOAD; block++) {
			uint32_t weights[STEPS_PER_BLOCK]
					[REGISTERS_PER_OPERAND_A];
			const uint32_t code =
				(current.codes >> (block * BITS_PER_BYTE)) &
				BYTE_MASK;
			relaxed_place_block<rows, RELAXED_ORDER>(
				current.weights[block],
				relaxed_fold_power<rows>(code,
							 plan->fold_field),
				weights);
			const uint4 stored =
				current.fed[block]; /* bfloat16 values become 16-bit floats where used */
			const uint4 fed =
				rows == QUANTIZED_ROWS ?
					stored :
					make_uint4(
						linear_place_bfloat16(stored.x),
						linear_place_bfloat16(stored.y),
						linear_place_bfloat16(stored.z),
						linear_place_bfloat16(
							stored.w));
			linear_tensor_core<form>(sums, weights[0], fed.x,
						 fed.y);
			linear_tensor_core<form>(sums, weights[1], fed.z,
						 fed.w);
		}
	}
#pragma unroll
	for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
		decode_sums[threadIdx.x * ELEMENTS_PER_LANE + element] =
			sums[element];
	}
	__syncthreads();
	for (uint32_t result = threadIdx.x;
	     result < THREADS_PER_WARP * ELEMENTS_PER_LANE;
	     result += blockDim.x) {
		const uint32_t result_lane = result / ELEMENTS_PER_LANE,
			       element = result % ELEMENTS_PER_LANE;
		const uint32_t row =
			result_lane % LANES_PER_LANE_GROUP * COLUMNS_PER_LANE +
			element % COLUMNS_PER_LANE;
		float sum = 0.0f;
		for (uint32_t each = 0; each < slice_count; each++) {
			sum += decode_sums[(each * THREADS_PER_WARP +
					    result_lane) *
						   ELEMENTS_PER_LANE +
					   element];
		}
		if (row < plan->row_count) {
			plan->output[(size_t)row * plan->output_size +
				     plan->matrix_tile * OUTPUTS_PER_TILE +
				     result_lane / LANES_PER_LANE_GROUP +
				     element / COLUMNS_PER_LANE *
					     OUTPUTS_PER_HALF_TILE] =
				(uint16_t)linear_bfloat16_bits(
					sum * plan->output_scale);
		}
	}
}

/* The lane of a thread of the decode form: its tile's weights and scale codes, its column's row. */
__device__ static __forceinline__ struct relaxed_decode_lane
relaxed_decode_lane_of(const struct relaxed_wide_plan *plan, int rows)
{
	const uint32_t lane_index = threadIdx.x % THREADS_PER_WARP;
	const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP,
		       lane_place = lane_index % LANES_PER_LANE_GROUP;
	const uint32_t group_count = plan->input_size / VALUES_PER_BLOCK;
	const size_t fed_row = lane_group < plan->row_count ? lane_group : 0;
	struct relaxed_decode_lane lane;
	lane.weights =
		(const uint4 *)plan->matrix +
		(size_t)plan->matrix_tile * group_count * THREADS_PER_WARP +
		lane_index;
	lane.scales = plan->matrix +
		      (size_t)plan->output_size * plan->input_size +
		      (size_t)(plan->matrix_tile / TILES_PER_SCALE_BLOCK) *
			      group_count;
	lane.fed = plan->rows + fed_row * plan->row_bytes +
		   (rows == QUANTIZED_ROWS ?
			    (size_t)lane_place * LANE_PLACE_FED_BYTES :
			    (size_t)lane_place * COLUMNS_PER_LANE *
				    LINEAR_BFLOAT16_BYTES);
	return lane;
}

extern "C" __global__ void trunk_relaxed_decode_quantized(
	struct trunk_relaxed_quantized_arguments arguments,
	unsigned char *arena)
{
	const struct relaxed_wide_plan plan =
		relaxed_quantized_plan(&arguments, arena, blockIdx.x);
	const struct relaxed_decode_lane lane =
		relaxed_decode_lane_of(&plan, QUANTIZED_ROWS);
	relaxed_decode_block<QUANTIZED_ROWS>(&plan, &lane);
}

extern "C" __global__ void
trunk_relaxed_decode_bfloat16(struct trunk_relaxed_bfloat16_arguments arguments,
			      unsigned char *arena)
{
	const struct relaxed_wide_plan plan =
		relaxed_bfloat16_plan(&arguments, arena, blockIdx.x);
	const struct relaxed_decode_lane lane =
		relaxed_decode_lane_of(&plan, BFLOAT16_ROWS);
	relaxed_decode_block<BFLOAT16_ROWS>(&plan, &lane);
}

/* grid (ceil(row_count * outputs / 256)): a thread takes one output of one row. */
extern "C" __global__ void
trunk_relaxed_reduce(struct trunk_relaxed_reduce_arguments arguments,
		     unsigned char *arena)
{
	const uint32_t output_total =
		arguments.output_size[0] + arguments.output_size[1];
	const size_t result = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	const size_t result_count = (size_t)arguments.row_count * output_total;
	if (result >= result_count) {
		return;
	}
	const float *const partials =
		(const float *)(arena + arguments.partials_offset) + result;
	float sum = 0.0f;
	for (uint32_t split = 0; split < arguments.k_split_count; split++) {
		sum += partials[(size_t)split * result_count];
	}
	const size_t row = result / output_total;
	const uint32_t output = (uint32_t)(result % output_total);
	const bool is_second = output >= arguments.output_size[0];
	uint16_t *const values =
		(uint16_t *)(arena + (is_second ? arguments.output_offset[1] :
						  arguments.output_offset[0]));
	values[row * (is_second ? arguments.output_size[1] :
				  arguments.output_size[0]) +
	       (is_second ? output - arguments.output_size[0] : output)] =
		(uint16_t)linear_bfloat16_bits(sum * arguments.output_scale);
}
