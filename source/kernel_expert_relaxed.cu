/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_expert_relaxed.cu: the routed experts of one layer in the relaxed order of class 80 (decode_accuracy): gate
 * and up with the hidden step, and down, from the carried operands. Device code only, built for sm_80 into the module
 * "kernel_expert_relaxed". Between gate-up and down the hidden vectors are quantized by activation_quantize_entry;
 * expert_row_sum_entry adds a row's pairs.
 *
 * Contract: include/anemone/expert_relaxed_records.h (the order, the records, the launches); linear_records.h (the
 * arrays, the job table, the operands, the hidden step); wide_linear_records.h (tiles, pairs). Here:
 *   - A block is eight warps. A warp owns a unit of two tiles of 16 outputs: gate's and up's tile u, or down's tiles
 *     2 u and 2 u + 1; its weights are operand A of the instruction, its columns of eight pairs operand B.
 *   - narrow (the job table): the block's warps share one unit (grid[0]) and one job (grid[1]); warp s takes slice s
 *     of the loads (linear_slice_start). A warp is a pipeline of its own: NARROW_STAGES loads in its part of shared
 *     memory, copied by cp.async ahead of the one it works on (a lane copies 32 bytes of one tile's weights and the 32
 *     bytes of operands its column feeds, four lanes the scale words), behind a wait for the lane's own copies and a
 *     __syncwarp; no barrier of the block until its last load. Its sums are linear_load_sums' carried blocks and
 *     linear_slice_chains' fused multiply-adds: a slice's chain is the class forms' bit for bit. The block adds its
 *     warps' chains in slice order once every warp is done.
 *   - wide (the tiles), a weight: the placed code (relaxed_place_e2m1: linear_place_e2m1's bfloat16, the code's value
 *     times 2^-126) times the bfloat16 pair whose field is the scale code plus 26, 254 at most (fma.rn.bf16x2 with -0
 *     added: one rounding, exact in the records header's range): 2^-100 times the true weight against the operands'
 *     2^100. Each step of a load is one instruction on one accumulator a tile and column group, in ascending input
 *     order from +0, in both wide kernels: an output's instructions are the same in either, so are its bits.
 *   - wide: a block is one tile's pairs; warp w takes unit 8 grid[0] + w over every load. Its weights flow through a
 *     pipeline of its own (STAGES deep: cp.async, a wait and a __syncwarp); the pairs' rows of operands, which every
 *     warp reads, through FED_STAGES chunks of FED_CHUNK loads in the block's memory, copied by every thread, behind a
 *     barrier at each chunk's first load (which also frees the chunk before the last: STAGES - 1 <= FED_CHUNK). Rows
 *     lie a chunk's bytes and 16 apart, 4 banks, so no two lanes of a phase of a 16-byte load meet a bank.
 *   - Two shapes of it (stages, chunk, rows a chunk, blocks an SM): small (3, 2, 32, 3: a tile of at most
 *     SMALL_TILE_PAIRS pairs, its expert's only one) and large (3, 2, 64, 2: every other tile; a ragged tile so runs
 *     beside its expert's others, whose weights it then finds in the L2).
 *   - A column from pair_count on is stored nowhere (narrow: it carries the vector of column 0; wide: its rows hold
 *     zeros). A wide block runs the instance of its group count (relaxed_groups): a group from pair_count on is
 *     neither copied nor computed, and no instruction of it is issued.
 *   - No atomic operation. Every warp of a block passes the same barriers; a block of no pair, or of a tile of the
 *     other wide kernel, leaves before any.
 */
#include "kernel_linear_common.cuh"

#include "anemone/expert_relaxed_records.h"

#define WARPS EXPERT_RELAXED_WARPS
#define BLOCK_THREADS EXPERT_RELAXED_THREADS
#define MULTIPROCESSOR_SHARED_BYTES \
	167936u /* sm_80: what the blocks of an SM share */
#define BLOCK_SHARED_RESERVE_BYTES \
	1792u /* a block's beside its dynamic memory: the system's, its pairs */
#define UNIT_TILES 2u /* a warp's tiles: of gate and of up, or two of down */
#define GROUP_COLUMNS 8u /* the columns of one instruction */
#define ROLLED_GROUPS \
	7u /* from this many groups on, one step of a load at a time */
#define FED_STAGES 2u /* chunks of rows a block holds */
#define LEAST_STAGES 2u /* the load worked on, and the one copied meanwhile */
#define PAIR_CAPACITY WIDE_TILE_PAIRS /* the most pairs of a block: a tile's */
#define SMALL_TILE_PAIRS EXPERT_RELAXED_SMALL_TILE_PAIRS
#define NARROW_SLICES EXPERT_RELAXED_NARROW_SLICES
#define NARROW_STAGES \
	EXPERT_RELAXED_NARROW_STAGES /* loads a narrow warp holds: one fewer in flight */
#define NARROW_BLOCKS_PER_MULTIPROCESSOR \
	4 /* four blocks fill an SM's shared memory: 64 registers a thread */
#define GATE_UP_RESULTS     \
	(OUTPUTS_PER_TILE * \
	 GROUP_COLUMNS) /* narrow: a tile's sums, a thread each at the end */
/* the wide shapes: stages, loads a chunk, rows a chunk, the most groups, blocks an SM (their registers) */
#define SMALL_STAGES 3u
#define SMALL_FED_CHUNK 2u
#define SMALL_FED_ROWS 32u /* its groups' rows, two parts a thread */
#define SMALL_GROUPS (SMALL_TILE_PAIRS / GROUP_COLUMNS)
#define SMALL_BLOCKS_PER_MULTIPROCESSOR 3
#define LARGE_STAGES 3u
#define LARGE_FED_CHUNK 2u
#define LARGE_FED_ROWS PAIR_CAPACITY
#define LARGE_GROUPS (PAIR_CAPACITY / GROUP_COLUMNS)
#define LARGE_BLOCKS_PER_MULTIPROCESSOR 2
#define ORDERED_LARGE_BLOCKS_PER_MULTIPROCESSOR 1
#define ORDERED_SMALL_BLOCKS_PER_MULTIPROCESSOR 3
#define PART_BYTES 16u /* one cp.async */
#define STEP_OPERAND_BYTES \
	(PART_BYTES /      \
	 STEPS_PER_BLOCK) /* a lane's operand B of one step: two registers */
#define LOAD_OPERAND_BYTES (VALUES_PER_LOAD * LINEAR_BFLOAT16_BYTES)
#define LANE_PLACE_BYTES (OPERAND_LOADS * PART_BYTES)
#define WEIGHT_LOAD_BYTES \
	(THREADS_PER_WARP * PART_BYTES) /* a tile's weights of one load */
#define SCALE_LOAD_BYTES (LANE_GROUPS_PER_WARP * E2M1_WORD_BYTES)
#define SCALE_PARTS         \
	(SCALE_LOAD_BYTES / \
	 PART_BYTES) /* 16-byte parts of a tile's scale words of a load */
#define TILE_COPY_LANES     \
	(THREADS_PER_WARP / \
	 UNIT_TILES) /* the lanes that copy a tile's weights of a load */
#define SCALE_FIELD_OFFSET \
	E2M1_CARRIED_FIELD /* 126 of the placement less the operands' 100 */
#define PAIR_FIELD \
	0x00800080u /* a field times this: its power of two in both halves of a bfloat16 pair */
#define NEGATIVE_ZERO_PAIR \
	0x80008000u /* x * y + -0 is x * y, a zero keeping its sign */
#define BYTES_OF_ONE 0x01010101u /* 1 in every byte of a word */
#define LARGE_CODE_BITS \
	0x80808080u /* bit 7 of every byte: a scale code from 128 on */
#define PAIR_SELECTOR \
	0x4040u /* __byte_perm: byte 0 of the first word to bytes 0 and 2, zeros to 1 and 3 */
#define PAIR_SELECTOR_STEP 0x0101u /* the selector of the next byte */
#define BFLOAT16_MANTISSA_BITS 7u
#define UPPER_SHIFT_FACTOR      \
	(1u << (BITS_PER_WORD - \
		E2M1_UPPER_SHIFT)) /* the high word of x times it: x >> 8 */
#define BITS_PER_WORD 32u
#define FULL_WARP 0xFFFFFFFFu

/* A tile's weights of one load: its lanes' 16 bytes, its lane groups' scale words. A warp holds its two tiles' a load
 * of its ring. */
struct relaxed_tile_load {
	uint4 weights[THREADS_PER_WARP];
	uint32_t scales[LANE_GROUPS_PER_WARP];
};
/* A narrow warp's stage: its unit's weights of one load, the 32 bytes of operands each lane's column feeds,
 * [part][lane]. */
struct relaxed_narrow_stage {
	struct relaxed_tile_load load[UNIT_TILES];
	uint4 fed[OPERAND_LOADS][THREADS_PER_WARP];
};

/* The shared memory of a shape: its warps' rings of weights, then its chunks of rows. */
#define SHAPE_SHARED_BYTES(stages, fed_chunk, fed_rows)                     \
	(sizeof(struct relaxed_tile_load) * UNIT_TILES * (stages) * WARPS + \
	 FED_STAGES * (fed_rows) *                                          \
		 ((fed_chunk) * LOAD_OPERAND_BYTES + PART_BYTES))
/* A shape's copies of rows: every thread the same count of adjacent parts of one row. */
#define SHAPE_COPIES_EVEN(fed_chunk, fed_rows)           \
	(BLOCK_THREADS % (fed_rows) == 0 &&              \
	 (fed_chunk) * LOAD_OPERAND_BYTES / PART_BYTES % \
			 (BLOCK_THREADS / (fed_rows)) == \
		 0)
static_assert(
	sizeof(struct relaxed_narrow_stage) ==
			EXPERT_RELAXED_NARROW_STAGE_BYTES &&
		sizeof(struct relaxed_narrow_stage) * NARROW_STAGES * WARPS ==
			EXPERT_RELAXED_NARROW_SHARED_BYTES &&
		NARROW_SLICES == WARPS &&
		UNIT_TILES * GATE_UP_RESULTS <= BLOCK_THREADS &&
		NARROW_SLICES * UNIT_TILES * THREADS_PER_WARP *
				sizeof(float4) <=
			EXPERT_RELAXED_NARROW_SHARED_BYTES &&
		NARROW_STAGES >= LEAST_STAGES &&
		SHAPE_SHARED_BYTES(SMALL_STAGES, SMALL_FED_CHUNK,
				   SMALL_FED_ROWS) ==
			EXPERT_RELAXED_WIDE_SMALL_SHARED_BYTES &&
		SHAPE_SHARED_BYTES(LARGE_STAGES, LARGE_FED_CHUNK,
				   LARGE_FED_ROWS) ==
			EXPERT_RELAXED_WIDE_SHARED_BYTES &&
		SHAPE_COPIES_EVEN(SMALL_FED_CHUNK, SMALL_FED_ROWS) &&
		SHAPE_COPIES_EVEN(LARGE_FED_CHUNK, LARGE_FED_ROWS) &&
		SMALL_STAGES - 1u <= SMALL_FED_CHUNK &&
		LARGE_STAGES - 1u <= LARGE_FED_CHUNK &&
		EXPERT_RELAXED_NARROW_INPUTS == VALUES_PER_LOAD &&
		EXPERT_RELAXED_WIDE_INPUTS ==
			SMALL_FED_CHUNK * VALUES_PER_LOAD &&
		EXPERT_RELAXED_WIDE_INPUTS ==
			LARGE_FED_CHUNK * VALUES_PER_LOAD &&
		SMALL_GROUPS * GROUP_COLUMNS == SMALL_TILE_PAIRS &&
		SMALL_TILE_PAIRS <= SMALL_FED_ROWS &&
		GROUP_COLUMNS == LINEAR_MAXIMUM_ROWS &&
		UNIT_TILES * SCALE_PARTS <= THREADS_PER_WARP &&
		EXPERT_RELAXED_THREADS == WARPS * THREADS_PER_WARP &&
		EXPERT_RELAXED_NARROW_GATE_UP_OUTPUTS == OUTPUTS_PER_TILE &&
		EXPERT_RELAXED_NARROW_DOWN_OUTPUTS ==
			UNIT_TILES * OUTPUTS_PER_TILE &&
		EXPERT_RELAXED_WIDE_GATE_UP_OUTPUTS ==
			WARPS * OUTPUTS_PER_TILE &&
		EXPERT_RELAXED_WIDE_DOWN_OUTPUTS ==
			WARPS * UNIT_TILES * OUTPUTS_PER_TILE,
	"the shapes, their copies and their launches as the records header counts them");
/* The blocks an SM holds (the launch bounds, which set the registers a thread may take) fit its shared memory. */
#define BLOCKS_FIT(blocks, shared_bytes)                             \
	((blocks) * ((shared_bytes) + BLOCK_SHARED_RESERVE_BYTES) <= \
	 MULTIPROCESSOR_SHARED_BYTES)
static_assert(BLOCKS_FIT(NARROW_BLOCKS_PER_MULTIPROCESSOR,
			 EXPERT_RELAXED_NARROW_SHARED_BYTES) &&
		      BLOCKS_FIT(SMALL_BLOCKS_PER_MULTIPROCESSOR,
				 EXPERT_RELAXED_WIDE_SMALL_SHARED_BYTES) &&
		      BLOCKS_FIT(LARGE_BLOCKS_PER_MULTIPROCESSOR,
				 EXPERT_RELAXED_WIDE_SHARED_BYTES),
	      "the blocks an SM holds fit its shared memory");

/* What every thread knows of its block's pairs: set by the first threads before the first barrier. Column c holds
 * pair_index[c], its route weight and its fed vector (the pair's row for gate and up, the pair for down); from
 * pair_count on, column 0's. */
struct relaxed_pairs {
	uint32_t pair_index[PAIR_CAPACITY];
	uint32_t vector[PAIR_CAPACITY];
	float route_weight[PAIR_CAPACITY];
};

__device__ static __forceinline__ uint32_t
relaxed_shared_address(const void *pointer)
{
	return (uint32_t)__cvta_generic_to_shared(pointer);
}

/* 16 bytes from the card's memory to shared memory, behind the L2 only (weights: read once) or through L1 too. */
__device__ static __forceinline__ void relaxed_copy_streamed(void *shared,
							     const void *global)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
			     relaxed_shared_address(shared)),
		     "l"(global)
		     : "memory");
}
__device__ static __forceinline__ void relaxed_copy_cached(void *shared,
							   const void *global)
{
	asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
			     relaxed_shared_address(shared)),
		     "l"(global)
		     : "memory");
}
__device__ static __forceinline__ void relaxed_copies_commit(void)
{
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}
/* Waits until at most `pending` of the thread's latest groups of copies are unfinished. */
template <int pending> __device__ __forceinline__ void relaxed_copies_wait(void)
{
	asm volatile("cp.async.wait_group %0;\n" ::"n"(pending) : "memory");
}

/* The bfloat16 pair that scales the weights of output half `half` of block `block` of a load: byte 2 block + half of
 * the lane group's scale word (weight_format_seam.h), its field c + 26, 254 at most. */
__device__ static __forceinline__ uint32_t
relaxed_scale_pair(uint32_t scale_word, uint32_t block, uint32_t half)
{
	const uint32_t code = (scale_word >> ((block * OUTPUT_HALVES + half) *
					      BITS_PER_BYTE)) &
			      BYTE_MASK;
	const uint32_t field = code + SCALE_FIELD_OFFSET;
	return (field > FLOAT32_LARGEST_EXPONENT_FIELD ?
			(uint32_t)FLOAT32_LARGEST_EXPONENT_FIELD :
			field) *
	       PAIR_FIELD;
}

/* relaxed_scale_pair of byte `byte` (2 block + half) of a scale word whose four codes are all below 128, from `fields`:
 * the word plus 26 in every byte (no byte carries, no field needs the clamp). Two instructions: the byte to both
 * halves, shifted into the exponent. */
__device__ static __forceinline__ uint32_t
relaxed_small_scale_pair(uint32_t fields, uint32_t byte)
{
	return __byte_perm(fields, 0, PAIR_SELECTOR + byte * PAIR_SELECTOR_STEP)
	       << BFLOAT16_MANTISSA_BITS;
}

/* linear_place_e2m1's bfloat16 placement, the same registers bit for bit; the shift of the upper codes is the high
 * word of a multiplication (the multiply-add pipe: the masks keep the integer pipe busy). */
__device__ static __forceinline__ void
relaxed_place_e2m1(uint32_t word, uint32_t weights[REGISTERS_PER_OPERAND_A])
{
	const uint32_t mask = (E2M1_MAGNITUDES << E2M1_BFLOAT16_BIT) |
			      HALF_SIGN_MASK;
	const uint32_t shifts = (1u << E2M1_BFLOAT16_BIT) |
				(1u << E2M1_SIGN_SHIFT);
	uint32_t upper;
	asm("mul.hi.u32 %0, %1, %2;\n"
	    : "=r"(upper)
	    : "r"(word), "n"(UPPER_SHIFT_FACTOR));
	weights[LOW_OUTPUT_LOW_INPUTS] = ((word & E2M1_PAIR_MASK) * shifts) &
					 mask;
	weights[HIGH_OUTPUT_LOW_INPUTS] =
		((word & (E2M1_PAIR_MASK << NIBBLE_BITS)) *
		 (shifts >> NIBBLE_BITS)) &
		mask;
	weights[LOW_OUTPUT_HIGH_INPUTS] = ((upper & E2M1_PAIR_MASK) * shifts) &
					  mask;
	weights[HIGH_OUTPUT_HIGH_INPUTS] =
		((upper & (E2M1_PAIR_MASK << NIBBLE_BITS)) *
		 (shifts >> NIBBLE_BITS)) &
		mask;
}

/* placed * scale for both halves of a bfloat16 pair, one rounding each. */
__device__ static __forceinline__ uint32_t relaxed_scaled(uint32_t placed,
							  uint32_t scale)
{
	uint32_t scaled;
	asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
	    : "=r"(scaled)
	    : "r"(placed), "r"(scale), "r"(NEGATIVE_ZERO_PAIR));
	return scaled;
}

/* accumulator = A B + accumulator, bfloat16 operands: the instruction, free to be scheduled among the others. */
__device__ static __forceinline__ void
relaxed_tensor_core(float accumulator[ELEMENTS_PER_LANE],
		    const uint32_t weights[REGISTERS_PER_OPERAND_A],
		    uint32_t fed_low_inputs, uint32_t fed_high_inputs)
{
	asm(LINEAR_MMA("bf16") : LINEAR_MMA_OPERANDS);
}

/* The copies of a warp's weights of one load into its ring entry (its two tiles): lane l parts l % 16 and l % 16 + 16
 * of tile l / 16 (an instruction reads 256 bytes of a tile in a row); lanes below 2 SCALE_PARTS part l % SCALE_PARTS
 * of the scale words of tile l / SCALE_PARTS. weight_source and scale_source are the lane's at load 0 of its run. */
__device__ static __forceinline__ void
relaxed_weights_copy(struct relaxed_tile_load *entry, uint32_t lane_index,
		     const unsigned char *weight_source,
		     const unsigned char *scale_source, uint32_t load)
{
	uint4 *const parts = &entry[lane_index / TILE_COPY_LANES]
				      .weights[lane_index % TILE_COPY_LANES];
#pragma unroll
	for (uint32_t copy = 0; copy < UNIT_TILES; copy++) {
		relaxed_copy_streamed(
			parts + copy * TILE_COPY_LANES,
			weight_source + (size_t)load * WEIGHT_LOAD_BYTES +
				copy * TILE_COPY_LANES * PART_BYTES);
	}
	if (lane_index < UNIT_TILES * SCALE_PARTS) {
		relaxed_copy_streamed(
			&entry[lane_index / SCALE_PARTS]
				 .scales[lane_index % SCALE_PARTS *
					 WORDS_PER_PART],
			scale_source + (size_t)load * SCALE_LOAD_BYTES);
	}
}

/* Where a lane's weight copies begin at first_load: its first part of its tile, and (lanes below 2 SCALE_PARTS) its
 * part of a tile's scale words, returned. */
__device__ static __forceinline__ const unsigned char *
relaxed_weight_sources(const unsigned char *const matrices[UNIT_TILES],
		       const uint32_t tiles[UNIT_TILES], uint32_t output_size,
		       uint32_t input_size, uint32_t first_load,
		       uint32_t lane_index, const unsigned char **weight_source)
{
	const uint32_t load_count = input_size / VALUES_PER_LOAD;
	const bool is_second_copy_tile = lane_index >= TILE_COPY_LANES;
	const bool is_second_scale_tile =
		lane_index / SCALE_PARTS % UNIT_TILES != 0;
	*weight_source = (is_second_copy_tile ? matrices[1] : matrices[0]) +
			 (((size_t)(is_second_copy_tile ? tiles[1] : tiles[0]) *
				   load_count +
			   first_load) *
				  THREADS_PER_WARP +
			  lane_index % TILE_COPY_LANES) *
				 PART_BYTES;
	const size_t weight_bytes =
		(size_t)output_size * input_size / E2M1_WEIGHTS_PER_BYTE;
	return (is_second_scale_tile ? matrices[1] : matrices[0]) +
	       weight_bytes +
	       ((size_t)(is_second_scale_tile ? tiles[1] : tiles[0]) *
			load_count +
		first_load) *
		       SCALE_LOAD_BYTES +
	       lane_index % SCALE_PARTS * PART_BYTES;
}

/* The hidden value of one output of a column from its gate and up sums (each rounded to bfloat16 first, as the
 * reference's linear returns them), or one output of down rounded. */
__device__ static __forceinline__ void
relaxed_store_hidden(unsigned char *outputs, uint32_t output_size,
		     const struct relaxed_pairs *pairs, uint32_t column,
		     uint32_t output, float gate, float up,
		     float activation_limit)
{
	uint16_t *const row = (uint16_t *)outputs +
			      (size_t)pairs->pair_index[column] * output_size;
	row[output] = (uint16_t)linear_hidden_bits(
		linear_bfloat16_value(linear_bfloat16_bits(gate)),
		linear_bfloat16_value(linear_bfloat16_bits(up)),
		activation_limit, pairs->route_weight[column]);
}
__device__ static __forceinline__ void
relaxed_store_output(unsigned char *outputs, uint32_t output_size,
		     const struct relaxed_pairs *pairs, uint32_t column,
		     uint32_t output, float sum)
{
	((uint16_t *)outputs +
	 (size_t)pairs->pair_index[column] * output_size)[output] =
		(uint16_t)linear_bfloat16_bits(sum);
}

/* The four registers of operand A of one step of block `block` of a load: the step's word placed and scaled by the
 * block's pairs of the lane group's scale word (relaxed_small_scale_pair's two instructions when are_codes_small). */
template <bool are_codes_small>
__device__ __forceinline__ void
relaxed_step_weights(uint32_t word, uint32_t scale_word, uint32_t block,
		     uint32_t weights[REGISTERS_PER_OPERAND_A])
{
	const uint32_t fields = scale_word + SCALE_FIELD_OFFSET * BYTES_OF_ONE;
	const uint32_t low_scale =
		are_codes_small ?
			relaxed_small_scale_pair(fields,
						 block * OUTPUT_HALVES) :
			relaxed_scale_pair(scale_word, block, 0);
	const uint32_t high_scale =
		are_codes_small ?
			relaxed_small_scale_pair(fields,
						 block * OUTPUT_HALVES + 1u) :
			relaxed_scale_pair(scale_word, block, 1);
	uint32_t placed[REGISTERS_PER_OPERAND_A];
	relaxed_place_e2m1(word, placed);
	weights[LOW_OUTPUT_LOW_INPUTS] =
		relaxed_scaled(placed[LOW_OUTPUT_LOW_INPUTS], low_scale);
	weights[HIGH_OUTPUT_LOW_INPUTS] =
		relaxed_scaled(placed[HIGH_OUTPUT_LOW_INPUTS], high_scale);
	weights[LOW_OUTPUT_HIGH_INPUTS] =
		relaxed_scaled(placed[LOW_OUTPUT_HIGH_INPUTS], low_scale);
	weights[HIGH_OUTPUT_HIGH_INPUTS] =
		relaxed_scaled(placed[HIGH_OUTPUT_HIGH_INPUTS], high_scale);
}

/* The two matrices and tiles of a warp's unit u: gate's and up's tile u, or down's tiles 2 u and 2 u + 1. */
template <bool is_gate_up>
__device__ __forceinline__ void
relaxed_unit(const unsigned char *arena, uint64_t first_offset,
	     uint64_t second_offset, uint32_t unit,
	     const unsigned char *matrices[UNIT_TILES],
	     uint32_t tiles[UNIT_TILES])
{
	matrices[0] = arena + first_offset;
	matrices[1] = arena + (is_gate_up ? second_offset : first_offset);
	tiles[0] = is_gate_up ? unit : unit * UNIT_TILES;
	tiles[1] = is_gate_up ? unit : unit * UNIT_TILES + 1u;
}

/* ---- narrow ---- */

/* grid (output_size / 16 or / 32, jobs): see the top. pairs: the block's, in shared memory; pair_count 1 to 8. */
template <bool is_gate_up>
__device__ __forceinline__ void
relaxed_narrow_product(const unsigned char *const matrices[UNIT_TILES],
		       const uint32_t tiles[UNIT_TILES], uint32_t output_size,
		       uint32_t input_size, const unsigned char *operands,
		       const struct relaxed_pairs *pairs, uint32_t pair_count,
		       unsigned char *outputs, float activation_limit)
{
	extern __shared__ uint4 relaxed_shared[];
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane_index = threadIdx.x % THREADS_PER_WARP;
	const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP,
		       lane_place = lane_index % LANES_PER_LANE_GROUP;
	struct relaxed_narrow_stage *const ring =
		(struct relaxed_narrow_stage *)relaxed_shared +
		warp * NARROW_STAGES;
	const uint32_t load_count = input_size / VALUES_PER_LOAD;
	const uint32_t first_load =
		linear_slice_start(warp, NARROW_SLICES, load_count);
	const uint32_t own_loads =
		linear_slice_start(warp + 1u, NARROW_SLICES, load_count) -
		first_load;
	const unsigned char *weight_source;
	const unsigned char *const scale_source =
		relaxed_weight_sources(matrices, tiles, output_size, input_size,
				       first_load, lane_index, &weight_source);
	/* the 32 bytes of operands the lane's column feeds, at the slice's first load */
	const unsigned char *const fed_source =
		operands +
		(size_t)pairs->vector[lane_group] * input_size *
			LINEAR_BFLOAT16_BYTES +
		(size_t)first_load * LOAD_OPERAND_BYTES +
		lane_place * LANE_PLACE_BYTES;
	struct linear_lane
		columns = {}; /* linear_load_sums reads column_count alone */
	columns.column_count = LINEAR_MAXIMUM_ROWS;
	float chains[UNIT_TILES][ELEMENTS_PER_LANE] = {};
	for (uint32_t copied = 0; copied + 1u < NARROW_STAGES; copied++) {
		if (copied < own_loads) {
			relaxed_weights_copy(ring[copied].load, lane_index,
					     weight_source, scale_source,
					     copied);
#pragma unroll
			for (uint32_t part = 0; part < OPERAND_LOADS; part++) {
				relaxed_copy_cached(
					&ring[copied].fed[part][lane_index],
					fed_source +
						(size_t)copied *
							LOAD_OPERAND_BYTES +
						part * PART_BYTES);
			}
		}
		relaxed_copies_commit();
	}
	for (uint32_t worked = 0; worked < own_loads; worked++) {
		relaxed_copies_wait<(int)(NARROW_STAGES - LEAST_STAGES)>();
		__syncwarp(); /* the scale words of other lanes; the stage refilled below is free */
		const uint32_t copied = worked + NARROW_STAGES - 1u;
		if (copied < own_loads) {
			struct relaxed_narrow_stage *const stage =
				&ring[copied % NARROW_STAGES];
			relaxed_weights_copy(stage->load, lane_index,
					     weight_source, scale_source,
					     copied);
#pragma unroll
			for (uint32_t part = 0; part < OPERAND_LOADS; part++) {
				relaxed_copy_cached(
					&stage->fed[part][lane_index],
					fed_source +
						(size_t)copied *
							LOAD_OPERAND_BYTES +
						part * PART_BYTES);
			}
		}
		relaxed_copies_commit();
		const struct relaxed_narrow_stage *const stage =
			&ring[worked % NARROW_STAGES];
		struct linear_fed fed = {};
#pragma unroll
		for (uint32_t part = 0; part < OPERAND_LOADS; part++) {
			const uint4 loaded = stage->fed[part][lane_index];
			const uint32_t words[WORDS_PER_PART] = {
				loaded.x, loaded.y, loaded.z, loaded.w
			};
#pragma unroll
			for (uint32_t word = 0; word < WORDS_PER_PART; word++) {
				fed.words[part * WORDS_PER_PART + word] =
					words[word];
			}
		}
#pragma unroll
		for (uint32_t tile = 0; tile < UNIT_TILES; tile++) {
			const uint4 part =
				stage->load[tile].weights[lane_index];
			const uint32_t weight_words[WORDS_PER_LOAD] = {
				part.x, part.y, part.z, part.w, 0, 0, 0, 0
			};
			float sums[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE],
				powers[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE];
			fed.weight_codes = stage->load[tile].scales[lane_group];
			linear_load_sums<LINEAR_WEIGHTS_E2M1,
					 LINEAR_FORM_CARRIED>(
				&columns, weight_words, &fed, sums, powers);
#pragma unroll
			for (uint32_t block = 0; block < BLOCKS_PER_LOAD;
			     block++) {
#pragma unroll
				for (uint32_t element = 0;
				     element < ELEMENTS_PER_LANE; element++) {
					chains[tile][element] =
						fmaf(sums[block][element],
						     powers[block][element],
						     chains[tile][element]);
				}
			}
		}
	}
	/* the slices' chains through shared memory, [slice][tile][lane] four sums each, added in slice order */
	relaxed_copies_wait<0>();
	__syncthreads();
	float4 *const slice_chains = (float4 *)relaxed_shared;
#pragma unroll
	for (uint32_t tile = 0; tile < UNIT_TILES; tile++) {
		slice_chains[(warp * UNIT_TILES + tile) * THREADS_PER_WARP +
			     lane_index] =
			make_float4(chains[tile][LOW_OUTPUT_EVEN_COLUMN],
				    chains[tile][LOW_OUTPUT_ODD_COLUMN],
				    chains[tile][HIGH_OUTPUT_EVEN_COLUMN],
				    chains[tile][HIGH_OUTPUT_ODD_COLUMN]);
	}
	__syncthreads();
	const float *const sums = (const float *)relaxed_shared;
	const bool is_second_tile =
		!is_gate_up &&
		threadIdx.x >= GATE_UP_RESULTS; /* down: its tile 1 */
	const uint32_t result =
		threadIdx.x %
		GATE_UP_RESULTS; /* lane 4 g + t, element: [lane][element] */
	const uint32_t result_lane = result / ELEMENTS_PER_LANE,
		       element = result % ELEMENTS_PER_LANE;
	const uint32_t column =
		result_lane % LANES_PER_LANE_GROUP * COLUMNS_PER_LANE +
		element % COLUMNS_PER_LANE;
	const uint32_t in_tile =
		result_lane / LANES_PER_LANE_GROUP +
		element / COLUMNS_PER_LANE * OUTPUTS_PER_HALF_TILE;
	if ((is_gate_up && threadIdx.x >= GATE_UP_RESULTS) ||
	    column >= pair_count) {
		return;
	}
	const uint32_t first_tile = is_second_tile ? 1u : 0u;
	float first_sum = 0.0f,
	      second_sum =
		      0.0f; /* gate-up: gate's and up's; down: its tile's */
#pragma unroll
	for (uint32_t slice = 0; slice < NARROW_SLICES; slice++) {
		first_sum += sums[(slice * UNIT_TILES + first_tile) *
					  THREADS_PER_WARP * ELEMENTS_PER_LANE +
				  result];
		if (is_gate_up) {
			second_sum += sums[(slice * UNIT_TILES + 1u) *
						   THREADS_PER_WARP *
						   ELEMENTS_PER_LANE +
					   result];
		}
	}
	if (is_gate_up) {
		relaxed_store_hidden(outputs, output_size, pairs, column,
				     tiles[0] * OUTPUTS_PER_TILE + in_tile,
				     first_sum, second_sum, activation_limit);
	} else {
		relaxed_store_output(outputs, output_size, pairs, column,
				     (is_second_tile ? tiles[1] : tiles[0]) *
						     OUTPUTS_PER_TILE +
					     in_tile,
				     first_sum);
	}
}

/* ---- wide ---- */

/* Keep each term's two MMAs and four fmaf operations together. This bounds live term registers independently of
 * the number of column groups; separating them let the compiler retain all groups' terms and spill. */
__device__ static __forceinline__ void
ordered_block_chain(float chain[ELEMENTS_PER_LANE],
		    const uint32_t first[REGISTERS_PER_OPERAND_A],
		    const uint32_t second[REGISTERS_PER_OPERAND_A],
		    uint4 operands, float low_power, float high_power)
{
	asm volatile(
		"{\n"
		".reg .f32 term0, term1, term2, term3;\n"
		"mov.f32 term0, 0f00000000; mov.f32 term1, 0f00000000;\n"
		"mov.f32 term2, 0f00000000; mov.f32 term3, 0f00000000;\n"
		"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
		"{term0,term1,term2,term3}, {%4,%5,%6,%7}, {%12,%13}, {term0,term1,term2,term3};\n"
		"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
		"{term0,term1,term2,term3}, {%8,%9,%10,%11}, {%14,%15}, {term0,term1,term2,term3};\n"
		"fma.rn.f32 %0, term0, %16, %0;\n"
		"fma.rn.f32 %1, term1, %16, %1;\n"
		"fma.rn.f32 %2, term2, %17, %2;\n"
		"fma.rn.f32 %3, term3, %17, %3;\n"
		"}\n"
		: "+f"(chain[LOW_OUTPUT_EVEN_COLUMN]),
		  "+f"(chain[LOW_OUTPUT_ODD_COLUMN]),
		  "+f"(chain[HIGH_OUTPUT_EVEN_COLUMN]),
		  "+f"(chain[HIGH_OUTPUT_ODD_COLUMN])
		: "r"(first[LOW_OUTPUT_LOW_INPUTS]),
		  "r"(first[HIGH_OUTPUT_LOW_INPUTS]),
		  "r"(first[LOW_OUTPUT_HIGH_INPUTS]),
		  "r"(first[HIGH_OUTPUT_HIGH_INPUTS]),
		  "r"(second[LOW_OUTPUT_LOW_INPUTS]),
		  "r"(second[HIGH_OUTPUT_LOW_INPUTS]),
		  "r"(second[LOW_OUTPUT_HIGH_INPUTS]),
		  "r"(second[HIGH_OUTPUT_HIGH_INPUTS]), "r"(operands.x),
		  "r"(operands.y), "r"(operands.z), "r"(operands.w),
		  "f"(low_power), "f"(high_power)
		: "memory");
}

/* The reference chain on the wide pipeline. Decode each K32 block once per tile and reuse it across all column
 * groups. Two MMAs start at +0; only their completed term is scaled into the persistent chain. The powers and
 * their clamp are exactly linear_load_sums' carried powers, including codes outside relaxed fold admission. */
template <uint32_t groups>
__device__ __forceinline__ void
ordered_load_products(const struct relaxed_tile_load *entry,
		      uint32_t lane_index, uint32_t lane_group,
		      const unsigned char *fed, uint32_t fed_group_bytes,
		      float accumulators[UNIT_TILES][groups][ELEMENTS_PER_LANE])
{
#pragma unroll 1
	for (uint32_t block = 0; block < BLOCKS_PER_LOAD; block++) {
#pragma unroll
		for (uint32_t tile = 0; tile < UNIT_TILES; tile++) {
			uint32_t weights[STEPS_PER_BLOCK]
					[REGISTERS_PER_OPERAND_A];
#pragma unroll
			for (uint32_t step = 0; step < STEPS_PER_BLOCK;
			     step++) {
				relaxed_place_e2m1(
					((const uint32_t *)&entry[tile]
						 .weights[lane_index])
						[block * STEPS_PER_BLOCK + step],
					weights[step]);
			}
			const uint32_t scale_codes =
				entry[tile].scales[lane_group];
			const float low_power =
				linear_block_power<LINEAR_WEIGHTS_E2M1,
						   LINEAR_FORM_CARRIED>(
					scale_codes, block, 0, 0);
			const float high_power =
				linear_block_power<LINEAR_WEIGHTS_E2M1,
						   LINEAR_FORM_CARRIED>(
					scale_codes, block, 1, 0);
#pragma unroll
			for (uint32_t group = 0; group < groups; group++) {
				const uint4 operands = *(
					const uint4 *)(fed +
						       group * fed_group_bytes +
						       block * PART_BYTES);
				ordered_block_chain(accumulators[tile][group],
						    weights[0], weights[1],
						    operands, low_power,
						    high_power);
			}
		}
	}
}

/* The products of one load of a wide warp: its two tiles (its weights in `ring`: the lane's words, its lane group's
 * scale words) by the first `groups` column groups, whose rows lie fed_group_bytes apart from the lane's first. Step
 * by step: the step's weights of both tiles scaled, then each group's two instructions. From ROLLED_GROUPS groups on
 * the steps take turns (a loop kept rolled): more weights at once would not fit beside the accumulators; their
 * operands are then 8 bytes a step, else 16 a block (a load of every lane of a phase in banks of its own).
 * are_codes_small: every scale code of the load's scale words below 128 (relaxed_small_scale_pair). */
template <uint32_t groups, bool are_codes_small>
__device__ __forceinline__ void
relaxed_load_steps(const struct relaxed_tile_load *entry, uint32_t lane_index,
		   uint32_t lane_group, const unsigned char *fed,
		   uint32_t fed_group_bytes,
		   float accumulators[UNIT_TILES][groups][ELEMENTS_PER_LANE])
{
	constexpr bool is_rolled = groups >= ROLLED_GROUPS;
	constexpr int steps_unrolled =
		is_rolled ? 1 :
			    (int)STEPS_PER_LOAD; /* a pragma expands no macro */
#pragma unroll steps_unrolled
	for (uint32_t step = 0; step < STEPS_PER_LOAD; step++) {
		const uint32_t block = step / STEPS_PER_BLOCK,
			       is_second_step = step % STEPS_PER_BLOCK;
		uint32_t weights[UNIT_TILES][REGISTERS_PER_OPERAND_A];
#pragma unroll
		for (uint32_t tile = 0; tile < UNIT_TILES; tile++) {
			relaxed_step_weights<are_codes_small>(
				((const uint32_t *)&entry[tile]
					 .weights[lane_index])[step],
				entry[tile].scales[lane_group], block,
				weights[tile]);
		}
#pragma unroll
		for (uint32_t group = 0; group < groups; group++) {
			const unsigned char *const group_fed =
				fed + group * fed_group_bytes;
			uint32_t low_inputs, high_inputs;
			if (is_rolled) {
				const uint2 operands =
					*(const uint2
						  *)(group_fed +
						     step * STEP_OPERAND_BYTES);
				low_inputs = operands.x;
				high_inputs = operands.y;
			} else {
				const uint4 operands =
					*(const uint4 *)(group_fed +
							 block * PART_BYTES);
				low_inputs = is_second_step != 0 ? operands.z :
								   operands.x;
				high_inputs = is_second_step != 0 ? operands.w :
								    operands.y;
			}
#pragma unroll
			for (uint32_t tile = 0; tile < UNIT_TILES; tile++) {
				relaxed_tensor_core(accumulators[tile][group],
						    weights[tile], low_inputs,
						    high_inputs);
			}
		}
	}
}

/* relaxed_load_steps, the small codes' way when every lane's scale words have them (a vote: one way a warp). */
template <uint32_t groups>
__device__ __forceinline__ void
relaxed_load_products(const struct relaxed_tile_load *entry,
		      uint32_t lane_index, uint32_t lane_group,
		      const unsigned char *fed, uint32_t fed_group_bytes,
		      float accumulators[UNIT_TILES][groups][ELEMENTS_PER_LANE])
{
	if (__all_sync(FULL_WARP, ((entry[0].scales[lane_group] |
				    entry[1].scales[lane_group]) &
				   LARGE_CODE_BITS) == 0)) {
		relaxed_load_steps<groups, true>(entry, lane_index, lane_group,
						 fed, fed_group_bytes,
						 accumulators);
	} else {
		relaxed_load_steps<groups, false>(entry, lane_index, lane_group,
						  fed, fed_group_bytes,
						  accumulators);
	}
}

/* A wide block's products: see the top. grid (output_size / 128 or / 256, tile_count); pairs: the block's, in shared
 * memory, pair_count from 8 (groups - 1) + 1 to 8 groups: the instance computes `groups` groups, no more. The shape:
 * stages, fed_chunk, fed_rows. */
template <bool is_gate_up, bool is_ordered, uint32_t groups, uint32_t stages,
	  uint32_t fed_chunk, uint32_t fed_rows>
__device__ __forceinline__ void
relaxed_product(const unsigned char *const matrices[UNIT_TILES],
		const uint32_t tiles[UNIT_TILES], uint32_t output_size,
		uint32_t input_size, const unsigned char *operands,
		const struct relaxed_pairs *pairs, uint32_t pair_count,
		unsigned char *outputs, float activation_limit)
{
	constexpr uint32_t fed_row_bytes = fed_chunk * LOAD_OPERAND_BYTES;
	constexpr uint32_t fed_row_stride = fed_row_bytes + PART_BYTES;
	constexpr uint32_t threads_per_row = BLOCK_THREADS / fed_rows;
	constexpr uint32_t parts_per_thread =
		fed_row_bytes / PART_BYTES / threads_per_row;
	extern __shared__ uint4 relaxed_shared[];
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane_index = threadIdx.x % THREADS_PER_WARP;
	const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP,
		       lane_place = lane_index % LANES_PER_LANE_GROUP;
	struct relaxed_tile_load *const rings = (struct relaxed_tile_load *)
		relaxed_shared; /* [stage][warp][tile] */
	unsigned char *const fed_stages =
		(unsigned char *)(rings +
				  stages * WARPS *
					  UNIT_TILES); /* [chunk][row] */
	const uint32_t load_count = input_size / VALUES_PER_LOAD;
	const unsigned char *weight_source;
	const unsigned char *const scale_source =
		relaxed_weight_sources(matrices, tiles, output_size, input_size,
				       0, lane_index, &weight_source);
	/* the thread's adjacent parts of a row of every chunk: copied if its column is in use, zeros once if its column
     * is not but its group is */
	const uint32_t copy_row = threadIdx.x / threads_per_row;
	const uint32_t copy_part =
		threadIdx.x % threads_per_row * parts_per_thread;
	const bool is_fed_copied = copy_row < pair_count;
	const unsigned char *const fed_source =
		operands +
		(size_t)pairs->vector[copy_row] * input_size *
			LINEAR_BFLOAT16_BYTES +
		copy_part * PART_BYTES;
	unsigned char *const fed_place =
		fed_stages + copy_row * fed_row_stride + copy_part * PART_BYTES;
	if (!is_fed_copied && copy_row < groups * GROUP_COLUMNS) {
#pragma unroll
		for (uint32_t chunk = 0; chunk < FED_STAGES; chunk++) {
#pragma unroll
			for (uint32_t part = 0; part < parts_per_thread;
			     part++) {
				*(uint4 *)(fed_place +
					   (size_t)chunk * fed_rows *
						   fed_row_stride +
					   part * PART_BYTES) =
					make_uint4(0u, 0u, 0u, 0u);
			}
		}
	}
	float accumulators[UNIT_TILES][groups][ELEMENTS_PER_LANE] = {};
	/* step `copied`: the warp's weights of that load; at a chunk's first load the block's rows of that chunk */
	for (uint32_t copied = 0; copied + 1u < stages; copied++) {
		if (copied < load_count) {
			relaxed_weights_copy(
				&rings[(copied * WARPS + warp) * UNIT_TILES],
				lane_index, weight_source, scale_source,
				copied);
			if (copied % fed_chunk == 0 && is_fed_copied) {
#pragma unroll
				for (uint32_t part = 0; part < parts_per_thread;
				     part++) {
					relaxed_copy_cached(
						fed_place +
							(size_t)(copied /
								 fed_chunk %
								 FED_STAGES) *
								fed_rows *
								fed_row_stride +
							part * PART_BYTES,
						fed_source +
							(size_t)copied *
								LOAD_OPERAND_BYTES +
							part * PART_BYTES);
				}
			}
		}
		relaxed_copies_commit();
	}
	for (uint32_t worked = 0; worked < load_count; worked++) {
		relaxed_copies_wait<(int)(stages - LEAST_STAGES)>();
		if (worked % fed_chunk == 0) {
			__syncthreads(); /* the chunk's rows from every thread; the chunk before is free */
		} else {
			__syncwarp(); /* the scale words of other lanes; the weights refilled are free */
		}
		const uint32_t copied = worked + stages - 1u;
		if (copied < load_count) {
			relaxed_weights_copy(
				&rings[(copied % stages * WARPS + warp) *
				       UNIT_TILES],
				lane_index, weight_source, scale_source,
				copied);
			if (copied % fed_chunk == 0 && is_fed_copied) {
#pragma unroll
				for (uint32_t part = 0; part < parts_per_thread;
				     part++) {
					relaxed_copy_cached(
						fed_place +
							(size_t)(copied /
								 fed_chunk %
								 FED_STAGES) *
								fed_rows *
								fed_row_stride +
							part * PART_BYTES,
						fed_source +
							(size_t)copied *
								LOAD_OPERAND_BYTES +
							part * PART_BYTES);
				}
			}
		}
		relaxed_copies_commit();
		const struct relaxed_tile_load *const entry =
			&rings[(worked % stages * WARPS + warp) * UNIT_TILES];
		const unsigned char *const fed =
			fed_stages +
			(size_t)(worked / fed_chunk % FED_STAGES) * fed_rows *
				fed_row_stride +
			worked % fed_chunk * LOAD_OPERAND_BYTES +
			lane_group * fed_row_stride +
			lane_place * LANE_PLACE_BYTES;
		if constexpr (is_ordered) {
			ordered_load_products<groups>(
				entry, lane_index, lane_group, fed,
				GROUP_COLUMNS * fed_row_stride, accumulators);
		} else {
			relaxed_load_products<groups>(
				entry, lane_index, lane_group, fed,
				GROUP_COLUMNS * fed_row_stride, accumulators);
		}
	}
#pragma unroll
	for (uint32_t group = 0; group < groups; group++) {
#pragma unroll
		for (uint32_t element = 0; element < ELEMENTS_PER_LANE;
		     element++) {
			const uint32_t column = group * GROUP_COLUMNS +
						lane_place * COLUMNS_PER_LANE +
						element % COLUMNS_PER_LANE;
			const uint32_t in_tile =
				lane_group + element / COLUMNS_PER_LANE *
						     OUTPUTS_PER_HALF_TILE;
			if (column >= pair_count) {
				continue;
			}
			if constexpr (is_ordered) {
				/* Add the original slice fold's +0, preserving signed zero. */
#pragma unroll
				for (uint32_t tile = 0; tile < UNIT_TILES;
				     tile++) {
					accumulators[tile][group][element] =
						0.0f + accumulators[tile][group]
								   [element];
				}
			}
			if (is_gate_up) {
				relaxed_store_hidden(
					outputs, output_size, pairs, column,
					tiles[0] * OUTPUTS_PER_TILE + in_tile,
					accumulators[0][group][element],
					accumulators[1][group][element],
					activation_limit);
			} else {
#pragma unroll
				for (uint32_t tile = 0; tile < UNIT_TILES;
				     tile++) {
					relaxed_store_output(
						outputs, output_size, pairs,
						column,
						tiles[tile] * OUTPUTS_PER_TILE +
							in_tile,
						accumulators[tile][group]
							    [element]);
				}
			}
		}
	}
}

/* relaxed_product stamped at the block's group count, from least_groups to `groups`: no instruction of a group not in
 * use is issued (a predicated one would be). pair_count from 8 (least_groups - 1) + 1 to 8 groups. */
template <bool is_gate_up, bool is_ordered, uint32_t groups,
	  uint32_t least_groups, uint32_t stages, uint32_t fed_chunk,
	  uint32_t fed_rows>
__device__ __forceinline__ void
relaxed_groups(const unsigned char *const matrices[UNIT_TILES],
	       const uint32_t tiles[UNIT_TILES], uint32_t output_size,
	       uint32_t input_size, const unsigned char *operands,
	       const struct relaxed_pairs *pairs, uint32_t pair_count,
	       unsigned char *outputs, float activation_limit)
{
	if constexpr (groups > least_groups) {
		if (pair_count <= (groups - 1u) * GROUP_COLUMNS) {
			relaxed_groups<is_gate_up, is_ordered, groups - 1u,
				       least_groups, stages, fed_chunk,
				       fed_rows>(matrices, tiles, output_size,
						 input_size, operands, pairs,
						 pair_count, outputs,
						 activation_limit);
			return;
		}
	}
	relaxed_product<is_gate_up, is_ordered, groups, stages, fed_chunk,
			fed_rows>(matrices, tiles, output_size, input_size,
				  operands, pairs, pair_count, outputs,
				  activation_limit);
}

/* ---- the kernels ---- */

/* Column c of a block's pairs from a job (narrow) or the sorted pairs of a tile (wide), for the first PAIR_CAPACITY
 * threads: from pair_count on, column 0's. vector_divisor: pairs_per_row (gate and up read the pair's row) or 1. */
__device__ static __forceinline__ void
relaxed_pair_set(struct relaxed_pairs *pairs, uint32_t column,
		 uint32_t pair_index, float route_weight,
		 uint32_t vector_divisor, uint32_t record_pair_count)
{
	LINEAR_DEVICE_CHECK(pair_index < record_pair_count);
	pairs->pair_index[column] = pair_index;
	pairs->vector[column] = pair_index / vector_divisor;
	pairs->route_weight[column] = route_weight;
}

template <bool is_gate_up>
__device__ __forceinline__ void
relaxed_narrow(const struct expert_job_record *job, unsigned char *arena,
	       uint32_t output_size, uint32_t input_size,
	       uint64_t operands_offset, uint64_t output_offset,
	       uint32_t record_pair_count, uint32_t pairs_per_row,
	       float activation_limit)
{
	__shared__ struct relaxed_pairs pairs;
	LINEAR_DEVICE_CHECK(blockDim.x == BLOCK_THREADS &&
			    job->row_count <= LINEAR_MAXIMUM_ROWS);
	const uint32_t pair_count = job->row_count;
	if (pair_count == 0) {
		return; /* the whole block, before any barrier */
	}
	if (threadIdx.x < GROUP_COLUMNS) {
		const uint32_t column = threadIdx.x < pair_count ? threadIdx.x :
								   0u;
		relaxed_pair_set(&pairs, threadIdx.x, job->pair_index[column],
				 job->route_weight[column],
				 is_gate_up ? pairs_per_row : 1u,
				 record_pair_count);
	}
	__syncthreads();
	const unsigned char *matrices[UNIT_TILES];
	uint32_t tiles[UNIT_TILES];
	relaxed_unit<is_gate_up>(
		arena, is_gate_up ? job->gate_offset : job->down_offset,
		job->up_offset, blockIdx.x, matrices, tiles);
	relaxed_narrow_product<is_gate_up>(matrices, tiles, output_size,
					   input_size, arena + operands_offset,
					   &pairs, pair_count,
					   arena + output_offset,
					   activation_limit);
}

/* is_small: the small kernels' tiles (of 1 to SMALL_TILE_PAIRS pairs, their expert's only one: the tile before it
 * names other matrices, or there is none), else the large kernels' (every other tile of pairs). */
template <bool is_gate_up, bool is_small, bool is_ordered = false>
__device__ __forceinline__ void
relaxed_wide(unsigned char *arena, uint64_t tiles_offset, uint64_t pairs_offset,
	     uint32_t tile_count, uint32_t output_size, uint32_t input_size,
	     uint64_t operands_offset, uint64_t output_offset,
	     uint32_t record_pair_count, uint32_t pairs_per_row,
	     float activation_limit)
{
	__shared__ struct relaxed_pairs pairs;
	const struct wide_expert_tile *const tile =
		(const struct wide_expert_tile *)(arena + tiles_offset) +
		blockIdx.y;
	LINEAR_DEVICE_CHECK(
		blockDim.x == BLOCK_THREADS && blockIdx.y < tile_count &&
		tile->pair_count <= WIDE_TILE_PAIRS &&
		tile->first_pair + tile->pair_count <= record_pair_count);
	const uint32_t pair_count = tile->pair_count;
	const uint64_t matrix_offset = is_gate_up ? tile->gate_offset :
						    tile->down_offset;
	const bool is_alone = blockIdx.y == 0 ||
			      (is_gate_up ? (tile - 1)->gate_offset :
					    (tile - 1)->down_offset) !=
				      matrix_offset;
	if (pair_count == 0 ||
	    (pair_count <= SMALL_TILE_PAIRS && is_alone) != is_small) {
		return; /* the whole block, before any barrier: the other kernels' */
	}
	if (threadIdx.x < PAIR_CAPACITY) {
		const struct wide_pair *const sorted =
			(const struct wide_pair *)(arena + pairs_offset);
		const struct wide_pair pair =
			sorted[tile->first_pair +
			       (threadIdx.x < pair_count ? threadIdx.x : 0u)];
		relaxed_pair_set(&pairs, threadIdx.x, pair.pair_index,
				 pair.route_weight,
				 is_gate_up ? pairs_per_row : 1u,
				 record_pair_count);
	}
	__syncthreads();
	const unsigned char *matrices[UNIT_TILES];
	uint32_t tiles[UNIT_TILES];
	relaxed_unit<is_gate_up>(arena, matrix_offset, tile->up_offset,
				 blockIdx.x * WARPS +
					 threadIdx.x / THREADS_PER_WARP,
				 matrices, tiles);
	if constexpr (is_small) {
		relaxed_groups<is_gate_up, is_ordered, SMALL_GROUPS, 1u,
			       SMALL_STAGES, SMALL_FED_CHUNK, SMALL_FED_ROWS>(
			matrices, tiles, output_size, input_size,
			arena + operands_offset, &pairs, pair_count,
			arena + output_offset, activation_limit);
	} else {
		relaxed_groups<is_gate_up, is_ordered, LARGE_GROUPS, 1u,
			       LARGE_STAGES, LARGE_FED_CHUNK, LARGE_FED_ROWS>(
			matrices, tiles, output_size, input_size,
			arena + operands_offset, &pairs, pair_count,
			arena + output_offset, activation_limit);
	}
}

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS,
					     NARROW_BLOCKS_PER_MULTIPROCESSOR)
	expert_relaxed_gate_up(struct expert_gate_up_arguments arguments,
			       unsigned char *arena)
{
	relaxed_narrow<true>(
		(const struct expert_job_record *)(arena +
						   arguments.jobs_offset) +
			blockIdx.y,
		arena, arguments.output_size, arguments.input_size,
		arguments.operands_offset, arguments.hidden_offset,
		arguments.pair_count, arguments.pairs_per_row,
		arguments.activation_limit);
}

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS,
					     NARROW_BLOCKS_PER_MULTIPROCESSOR)
	expert_relaxed_down(struct expert_down_arguments arguments,
			    unsigned char *arena)
{
	relaxed_narrow<false>(
		(const struct expert_job_record *)(arena +
						   arguments.jobs_offset) +
			blockIdx.y,
		arena, arguments.output_size, arguments.input_size,
		arguments.operands_offset, arguments.output_offset,
		arguments.pair_count, 1u, 0.0f);
}

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS,
					     LARGE_BLOCKS_PER_MULTIPROCESSOR)
	expert_relaxed_wide_gate_up_entry(
		struct expert_relaxed_wide_gate_up_arguments arguments,
		unsigned char *arena)
{
	relaxed_wide<true, false>(arena, arguments.tiles_offset,
				  arguments.pairs_offset, arguments.tile_count,
				  arguments.output_size, arguments.input_size,
				  arguments.operands_offset,
				  arguments.hidden_offset, arguments.pair_count,
				  arguments.pairs_per_row,
				  arguments.activation_limit);
}

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS,
					     LARGE_BLOCKS_PER_MULTIPROCESSOR)
	expert_relaxed_wide_down(
		struct expert_relaxed_wide_down_arguments arguments,
		unsigned char *arena)
{
	relaxed_wide<false, false>(arena, arguments.tiles_offset,
				   arguments.pairs_offset, arguments.tile_count,
				   arguments.output_size, arguments.input_size,
				   arguments.operands_offset,
				   arguments.output_offset,
				   arguments.pair_count, 1u, 0.0f);
}

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS,
					     SMALL_BLOCKS_PER_MULTIPROCESSOR)
	expert_relaxed_wide_small_gate_up(
		struct expert_relaxed_wide_gate_up_arguments arguments,
		unsigned char *arena)
{
	relaxed_wide<true, true>(arena, arguments.tiles_offset,
				 arguments.pairs_offset, arguments.tile_count,
				 arguments.output_size, arguments.input_size,
				 arguments.operands_offset,
				 arguments.hidden_offset, arguments.pair_count,
				 arguments.pairs_per_row,
				 arguments.activation_limit);
}

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS,
					     SMALL_BLOCKS_PER_MULTIPROCESSOR)
	expert_relaxed_wide_small_down(
		struct expert_relaxed_wide_down_arguments arguments,
		unsigned char *arena)
{
	relaxed_wide<false, true>(arena, arguments.tiles_offset,
				  arguments.pairs_offset, arguments.tile_count,
				  arguments.output_size, arguments.input_size,
				  arguments.operands_offset,
				  arguments.output_offset, arguments.pair_count,
				  1u, 0.0f);
}

extern "C" __global__ void
__launch_bounds__(BLOCK_THREADS, ORDERED_LARGE_BLOCKS_PER_MULTIPROCESSOR)
	expert_ordered_wide_gate_up(
		struct expert_relaxed_wide_gate_up_arguments arguments,
		unsigned char *arena)
{
	relaxed_wide<true, false, true>(
		arena, arguments.tiles_offset, arguments.pairs_offset,
		arguments.tile_count, arguments.output_size,
		arguments.input_size, arguments.operands_offset,
		arguments.hidden_offset, arguments.pair_count,
		arguments.pairs_per_row, arguments.activation_limit);
}

extern "C" __global__ void
__launch_bounds__(BLOCK_THREADS, ORDERED_LARGE_BLOCKS_PER_MULTIPROCESSOR)
	expert_ordered_wide_down(
		struct expert_relaxed_wide_down_arguments arguments,
		unsigned char *arena)
{
	relaxed_wide<false, false, true>(
		arena, arguments.tiles_offset, arguments.pairs_offset,
		arguments.tile_count, arguments.output_size,
		arguments.input_size, arguments.operands_offset,
		arguments.output_offset, arguments.pair_count, 1u, 0.0f);
}

extern "C" __global__ void
__launch_bounds__(BLOCK_THREADS, ORDERED_SMALL_BLOCKS_PER_MULTIPROCESSOR)
	expert_ordered_wide_small_gate_up(
		struct expert_relaxed_wide_gate_up_arguments arguments,
		unsigned char *arena)
{
	relaxed_wide<true, true, true>(
		arena, arguments.tiles_offset, arguments.pairs_offset,
		arguments.tile_count, arguments.output_size,
		arguments.input_size, arguments.operands_offset,
		arguments.hidden_offset, arguments.pair_count,
		arguments.pairs_per_row, arguments.activation_limit);
}

extern "C" __global__ void
__launch_bounds__(BLOCK_THREADS, ORDERED_SMALL_BLOCKS_PER_MULTIPROCESSOR)
	expert_ordered_wide_small_down(
		struct expert_relaxed_wide_down_arguments arguments,
		unsigned char *arena)
{
	relaxed_wide<false, true, true>(
		arena, arguments.tiles_offset, arguments.pairs_offset,
		arguments.tile_count, arguments.output_size,
		arguments.input_size, arguments.operands_offset,
		arguments.output_offset, arguments.pair_count, 1u, 0.0f);
}
