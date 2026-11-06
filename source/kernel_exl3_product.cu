/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Candidate SM80 EXL3 routed-expert products with fewer decode instructions per
 * weight. They keep kernel_exl3.cu's record ABI, launch axes y/z, slice
 * partition and FP32 association exactly: slice s runs one in-order m16n8k16
 * chain over its contiguous K tiles from +0, and the partials are added in slice
 * order from +0. Each warp decodes several adjacent output tiles of one K tile,
 * so the activation fragment and the loop bookkeeping are shared between them.
 *
 * Grouped form (prefill): the wide router writes each expert's pairs as a run
 * of consecutive 8-row jobs. The CTA of every groups-th job of a run decodes
 * each tile once and multiplies it with up to groups jobs' activations; the CTAs
 * of the other jobs of that run exit. Every output keeps its own chain, so the
 * grouped and ungrouped forms write identical words.
 *
 * Decode identity with exl3_state_value: dp4a with accumulator 0x6400 equals
 * dp4a with zero plus 0x6400 (byte sums are at most 1020, so neither wraps), and
 * the half2 FMA of two packed values is the same per-lane IEEE FP16 FMA that the
 * compiler already emits for the retained kernel. No arithmetic changes. */
#include <cuda_fp16.h>
#include "anemone/exl3_records.h"
#include "anemone/linear_records.h"

#define DECODE_C_WARP 32u
#define DECODE_C_LANE_GROUP 4u
#define DECODE_C_PAIR 2u
#define DECODE_C_CHAINS 4u
#define DECODE_C_INPUT_HALF 8u
#define DECODE_C_OUTPUT_HALF 8u
#define DECODE_C_VALUES_PER_LANE 8u
#define DECODE_C_HIGH_COLUMN_BIT 4u
#define DECODE_C_HIGH_ROW_BIT 2u
#define DECODE_C_BYTE_UNITS 0x01010101u
#define DECODE_C_HALF_PAIR_SELECT 0x5410u
#define DECODE_C_SCALE_PAIR 0x1EEE1EEEu
#define DECODE_C_BIAS_PAIR 0xC931C931u
#define DECODE_C_COLUMN_PAIR 2u
#define DECODE_C_FULL_MASK 0xffffffffu
/* A run is at most 1024 rows / 8 rows per job; the backward scan covers it. */
#define DECODE_C_RUN_SCAN 128u
#define DECODE_C_CHAIN_HIGH_EVEN 2u
#define DECODE_C_CHAIN_HIGH_ODD 3u
#define DECODE_C_PREFETCH_STEPS 2u
#define DECODE_C_TWO_OUTPUT_TILES 2
#define DECODE_C_FOUR_OUTPUT_TILES 4

template <uint32_t bits>
__device__ static __forceinline__ uint32_t decode_c_pair(uint64_t window,
							 uint32_t low_shift,
							 uint32_t high_shift)
{
	const uint32_t low_state = (uint32_t)(window >> low_shift) &
				   EXL3_HALF_MASK;
	const uint32_t high_state = (uint32_t)(window >> high_shift) &
				    EXL3_HALF_MASK;
	const uint32_t low_sum = __dp4a(low_state * EXL3_MUL1_MULTIPLIER,
					DECODE_C_BYTE_UNITS, EXL3_MUL1_BASE);
	const uint32_t high_sum = __dp4a(high_state * EXL3_MUL1_MULTIPLIER,
					 DECODE_C_BYTE_UNITS, EXL3_MUL1_BASE);
	const uint32_t packed =
		__byte_perm(low_sum, high_sum, DECODE_C_HALF_PAIR_SELECT);
	const uint32_t scale = DECODE_C_SCALE_PAIR;
	const uint32_t bias = DECODE_C_BIAS_PAIR;
	const __half2 value = __hfma2(*(const __half2 *)&packed,
				      *(const __half2 *)&scale,
				      *(const __half2 *)&bias);
	return *(const uint32_t *)&value;
}

/* One K step of the warp: trellis word pairs of its output tiles and the
 * activation fragments of its job groups. */
template <uint32_t tiles, uint32_t groups> struct decode_c_stage {
	uint32_t high[tiles];
	uint32_t low[tiles];
	uint32_t operand_low[groups];
	uint32_t operand_high[groups];
};

/* Operands written earlier in the same launch (fused form) must bypass the
 * non-coherent read-only path; trellis words are immutable and use it. */
template <bool coherent>
__device__ static __forceinline__ uint32_t
decode_c_operand_load(const uint32_t *address)
{
	return coherent ? __ldcg(address) : __ldg(address);
}

template <uint32_t tiles, uint32_t groups, bool coherent>
__device__ static __forceinline__ void
decode_c_load(struct decode_c_stage<tiles, groups> *stage,
	      const uint32_t *word_pointer, const uint32_t *successor_pointer,
	      const uint32_t *const *operand_pointer, uint32_t ring_words,
	      bool odd_step)
{
	for (uint32_t tile = 0; tile < tiles; tile++) {
		stage->high[tile] = __ldg(word_pointer + tile * ring_words);
		stage->low[tile] = __ldg(successor_pointer + tile * ring_words);
	}
	/* Operand pointers sit at the even step; the odd step is one K tile
	 * (eight words) further, an immediate offset. */
	const uint32_t step_words = odd_step ? EXL3_TILE / DECODE_C_PAIR : 0u;
	for (uint32_t group = 0; group < groups; group++) {
		stage->operand_low[group] = decode_c_operand_load<coherent>(
			operand_pointer[group] + step_words);
		stage->operand_high[group] = decode_c_operand_load<coherent>(
			operand_pointer[group] + step_words +
			DECODE_C_INPUT_HALF / DECODE_C_PAIR);
	}
}

template <uint32_t bits, uint32_t tiles, uint32_t groups>
__device__ static __forceinline__ void
decode_c_compute(const struct decode_c_stage<tiles, groups> *stage,
		 const uint32_t *symbol_shift,
		 float (*accumulators)[groups][DECODE_C_CHAINS])
{
	for (uint32_t tile = 0; tile < tiles; tile++) {
		const uint64_t window =
			((uint64_t)stage->high[tile] << EXL3_WORD_BITS) |
			stage->low[tile];
		uint32_t weights[DECODE_C_CHAINS];
		for (uint32_t fragment = 0; fragment < DECODE_C_CHAINS;
		     fragment++) {
			const uint32_t element =
				(fragment % DECODE_C_PAIR) *
					DECODE_C_HIGH_COLUMN_BIT +
				(fragment / DECODE_C_PAIR) *
					DECODE_C_HIGH_ROW_BIT;
			weights[fragment] = decode_c_pair<bits>(
				window, symbol_shift[element],
				symbol_shift[element + 1u]);
		}
		for (uint32_t group = 0; group < groups; group++)
			asm volatile(
				"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
				"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
				"{%0,%1,%2,%3};"
				: "+f"(accumulators[tile][group][0]),
				  "+f"(accumulators[tile][group][1]),
				  "+f"(accumulators[tile][group]
				       [DECODE_C_CHAIN_HIGH_EVEN]),
				  "+f"(accumulators[tile][group]
				       [DECODE_C_CHAIN_HIGH_ODD])
				: "r"(weights[0]), "r"(weights[1]),
				  "r"(weights[DECODE_C_CHAIN_HIGH_EVEN]),
				  "r"(weights[DECODE_C_CHAIN_HIGH_ODD]),
				  "r"(stage->operand_low[group]),
				  "r"(stage->operand_high[group]));
	}
}

/* Number of jobs this CTA owns from its own job: 0 when another CTA of the
 * same run owns it. Runs share gate_offset and are consecutive. */
template <uint32_t groups>
__device__ static uint32_t
decode_c_owned_jobs(const struct expert_job_record *jobs, uint32_t job_index,
		    uint32_t job_count)
{
	if (groups == 1)
		return 1;
	__shared__ uint32_t run_offset[DECODE_C_RUN_SCAN / DECODE_C_WARP];
	__shared__ uint32_t owned;
	const uint64_t expert = jobs[job_index].gate_offset;
	const uint32_t distance = threadIdx.x + 1u;
	/* A thread votes "other" at the first job before the run. */
	bool other = true;
	if (threadIdx.x < DECODE_C_RUN_SCAN && distance <= job_index)
		other = jobs[job_index - distance].gate_offset != expert ||
			jobs[job_index - distance].row_count == 0;
	const uint32_t warp = threadIdx.x / DECODE_C_WARP;
	const uint32_t votes = __ballot_sync(DECODE_C_FULL_MASK, other);
	if (threadIdx.x % DECODE_C_WARP == 0 &&
	    warp < DECODE_C_RUN_SCAN / DECODE_C_WARP)
		run_offset[warp] =
			votes == 0 ? DECODE_C_RUN_SCAN :
				     warp * DECODE_C_WARP + __ffs(votes) - 1u;
	__syncthreads();
	if (threadIdx.x == 0) {
		uint32_t position = DECODE_C_RUN_SCAN;
		for (uint32_t index = 0;
		     index < DECODE_C_RUN_SCAN / DECODE_C_WARP; index++)
			if (run_offset[index] < position) {
				position = run_offset[index];
				break;
			}
		/* position = jobs of this run before job_index. */
		uint32_t count = 0;
		if (position % groups == 0)
			while (count < groups &&
			       job_index + count < job_count &&
			       jobs[job_index + count].gate_offset == expert &&
			       jobs[job_index + count].row_count != 0)
				count++;
		owned = count;
	}
	__syncthreads();
	return owned;
}

/* Where one CTA's product work lies: its first job and owned job count, its
 * output tile group and projection, the FP16 operand array [projection][pair]
 * [inputs] and FP32 product array [projection][pair][outputs] of that pass. */
struct decode_c_place {
	const struct expert_job_record *job;
	uint64_t matrix_offset;
	uint64_t operand_offset;
	uint64_t product_offset;
	uint32_t owned_jobs;
	uint32_t tile_group;
	uint32_t projection;
	uint32_t inputs;
	uint32_t outputs;
	uint32_t pair_count;
};

template <uint32_t bits, uint32_t slices, uint32_t tiles, uint32_t groups,
	  bool coherent>
__device__ static void decode_c_product_core(const struct decode_c_place *place,
					     unsigned char *arena)
{
	const struct expert_job_record *const job = place->job;
	const uint32_t owned_jobs = place->owned_jobs;
	const uint32_t projection = place->projection;
	const uint32_t inputs = place->inputs;
	const uint32_t outputs = place->outputs;
	const uint32_t lane = threadIdx.x % DECODE_C_WARP;
	const uint32_t slice = threadIdx.x / DECODE_C_WARP;
	const uint32_t lane_group = lane / DECODE_C_LANE_GROUP;
	const uint32_t lane_place = lane % DECODE_C_LANE_GROUP;
	const uint32_t *const matrix =
		(const uint32_t *)(arena + place->matrix_offset);
	const uint32_t tile_count = inputs / EXL3_TILE;
	const uint32_t first = tile_count * slice / slices;
	const uint32_t last = tile_count * (slice + 1u) / slices;
	const uint32_t operand_words_per_tile = EXL3_TILE / DECODE_C_PAIR;
	/* FP16 pairs of each group's activation fragment, as 32-bit words.
	 * Groups past the owned jobs repeat group zero; their sums are unused. */
	const uint32_t *operand_pointer[groups];
	for (uint32_t group = 0; group < groups; group++) {
		const struct expert_job_record *const group_job =
			job + (group < owned_jobs ? group : 0);
		const uint32_t operand_pair =
			group_job->pair_index[lane_group < group_job->row_count ?
						      lane_group :
						      0];
		operand_pointer[group] =
			(const uint32_t
				 *)((const half *)(arena +
						   place->operand_offset) +
				    ((uint64_t)projection * place->pair_count +
				     operand_pair) *
					    inputs) +
			lane_place + (uint64_t)first * operand_words_per_tile;
	}
	const uint32_t ring_bits = EXL3_TILE_VALUES * bits;
	const uint32_t ring_words = ring_bits / EXL3_WORD_BITS;
	const uint32_t begin = ((lane * DECODE_C_VALUES_PER_LANE + 1u) * bits +
				ring_bits - EXL3_STATE_BITS) %
			       ring_bits;
	const uint32_t word = begin / EXL3_WORD_BITS;
	const uint32_t next_word = (word + 1u) % ring_words;
	const uint32_t shift = begin % EXL3_WORD_BITS;
	uint32_t symbol_shift[DECODE_C_VALUES_PER_LANE];
	for (uint32_t symbol = 0; symbol < DECODE_C_VALUES_PER_LANE; symbol++)
		symbol_shift[symbol] = EXL3_WORD_BITS + EXL3_STATE_BITS -
				       shift - symbol * bits;
	const uint32_t output_tiles = outputs / EXL3_TILE;
	/* Words between consecutive K tiles of one output tile. */
	const uint64_t tile_stride = (uint64_t)output_tiles * ring_words;
	/* Strength-reduced addresses: one pointer per word role, advanced by two
	 * K tiles per iteration; stage B adds one stride. The lane's second word
	 * is its cyclic successor inside the ring. */
	const uint32_t *word_pointer =
		matrix + (uint64_t)first * tile_stride +
		(uint64_t)place->tile_group * tiles * ring_words + word;
	const uint32_t *successor_pointer = word_pointer + next_word - word;
	float accumulators[tiles][groups][DECODE_C_CHAINS];
	for (uint32_t tile = 0; tile < tiles; tile++)
		for (uint32_t group = 0; group < groups; group++)
			for (uint32_t chain = 0; chain < DECODE_C_CHAINS;
			     chain++)
				accumulators[tile][group][chain] = 0;
	/* Two register stages: A holds even steps of the slice, B odd steps.
	 * Neither stage is rotated, so no register moves are needed. */
	struct decode_c_stage<tiles, groups> stage_a, stage_b;
	const uint32_t steps = last - first;
	if (steps > 0)
		decode_c_load<tiles, groups, coherent>(&stage_a, word_pointer,
						       successor_pointer,
						       operand_pointer,
						       ring_words, false);
	uint32_t step = 0;
	for (; step + 1u < steps; step += DECODE_C_PREFETCH_STEPS) {
		decode_c_load<tiles, groups, coherent>(
			&stage_b, word_pointer + tile_stride,
			successor_pointer + tile_stride, operand_pointer,
			ring_words, true);
		decode_c_compute<bits, tiles, groups>(&stage_a, symbol_shift,
						      accumulators);
		word_pointer += DECODE_C_PREFETCH_STEPS * tile_stride;
		successor_pointer += DECODE_C_PREFETCH_STEPS * tile_stride;
		for (uint32_t group = 0; group < groups; group++)
			operand_pointer[group] += DECODE_C_PREFETCH_STEPS *
						  operand_words_per_tile;
		if (step + DECODE_C_PREFETCH_STEPS < steps)
			decode_c_load<tiles, groups, coherent>(
				&stage_a, word_pointer, successor_pointer,
				operand_pointer, ring_words, false);
		decode_c_compute<bits, tiles, groups>(&stage_b, symbol_shift,
						      accumulators);
	}
	if (step < steps)
		decode_c_compute<bits, tiles, groups>(&stage_a, symbol_shift,
						      accumulators);
	__shared__ float partial[slices][tiles][groups][DECODE_C_WARP]
				[DECODE_C_CHAINS];
	for (uint32_t tile = 0; tile < tiles; tile++)
		for (uint32_t group = 0; group < groups; group++)
			for (uint32_t chain = 0; chain < DECODE_C_CHAINS;
			     chain++)
				partial[slice][tile][group][lane][chain] =
					accumulators[tile][group][chain];
	__syncthreads();
	/* Warps share the epilogue by (tile, group); every sum keeps the
	 * retained slice order from +0. */
	float *const products = (float *)(arena + place->product_offset);
	for (uint32_t item = slice; item < tiles * groups; item += slices) {
		const uint32_t tile = item / groups;
		const uint32_t group = item % groups;
		if (group >= owned_jobs)
			continue;
		const struct expert_job_record *const group_job = job + group;
		for (uint32_t chain = 0; chain < DECODE_C_CHAINS; chain++) {
			float sum = 0;
			for (uint32_t part = 0; part < slices; part++)
				sum += partial[part][tile][group][lane][chain];
			const uint32_t column =
				lane_place * DECODE_C_COLUMN_PAIR +
				chain % DECODE_C_PAIR;
			const uint32_t output =
				(place->tile_group * tiles + tile) * EXL3_TILE +
				lane_group +
				(chain / DECODE_C_PAIR) * DECODE_C_OUTPUT_HALF;
			if (column < group_job->row_count)
				products[((uint64_t)projection *
						  place->pair_count +
					  group_job->pair_index[column]) *
						 outputs +
					 output] = sum;
		}
	}
}

template <uint32_t bits, bool down, uint32_t slices, uint32_t tiles,
	  uint32_t groups>
__device__ static void exl3_decode_c_product(struct exl3_expert_record record,
					     unsigned char *arena)
{
	const struct expert_job_record *const jobs =
		(const struct expert_job_record *)(arena + record.jobs_offset);
	const struct expert_job_record *const job = jobs + blockIdx.y;
	if (job->row_count == 0)
		return;
	struct decode_c_place place;
	place.owned_jobs =
		decode_c_owned_jobs<groups>(jobs, blockIdx.y, record.job_count);
	if (place.owned_jobs == 0)
		return;
	place.job = job;
	place.tile_group = blockIdx.x;
	place.projection = blockIdx.z;
	place.inputs = down ? record.inner_size : record.input_size;
	place.outputs = down ? record.input_size : record.inner_size;
	place.pair_count = record.pair_count;
	place.matrix_offset =
		down ? job->down_offset :
		       (blockIdx.z == 0 ? job->gate_offset : job->up_offset);
	place.operand_offset = record.operands_offset;
	place.product_offset = record.products_offset;
	decode_c_product_core<bits, slices, tiles, groups, false>(&place,
								  arena);
}

#define EXL3_DECODE_C_ENTRY(name, bits, down, slices, tiles, groups)      \
	extern "C" __global__ void name(struct exl3_expert_record record, \
					unsigned char *arena)             \
	{                                                                 \
		exl3_decode_c_product<bits, down, slices, tiles, groups>( \
			record, arena);                                   \
	}
#define EXL3_DECODE_C_FAMILY_STAMP(tiles, groups)                          \
	EXL3_DECODE_C_ENTRY(exl3_c_gate_up_k2_t##tiles##_g##groups,        \
			    EXL3_MINIMUM_BITS, false, EXL3_PRODUCT_SLICES, \
			    tiles, groups)                                 \
	EXL3_DECODE_C_ENTRY(exl3_c_gate_up_k3_t##tiles##_g##groups,        \
			    EXL3_MAXIMUM_BITS, false, EXL3_PRODUCT_SLICES, \
			    tiles, groups)                                 \
	EXL3_DECODE_C_ENTRY(exl3_c_down_k2_t##tiles##_g##groups,           \
			    EXL3_MINIMUM_BITS, true, EXL3_PRODUCT_SLICES,  \
			    tiles, groups)                                 \
	EXL3_DECODE_C_ENTRY(exl3_c_down_k3_t##tiles##_g##groups,           \
			    EXL3_MAXIMUM_BITS, true, EXL3_PRODUCT_SLICES,  \
			    tiles, groups)
/* The explicit eight-partial alternative (relaxed preparation only). */
#define EXL3_DECODE_C_RELAXED_STAMP(tiles, groups)                         \
	EXL3_DECODE_C_ENTRY(exl3_c_gate_up_k2_s8_t##tiles##_g##groups,     \
			    EXL3_MINIMUM_BITS, false, EXL3_RELAXED_SLICES, \
			    tiles, groups)                                 \
	EXL3_DECODE_C_ENTRY(exl3_c_gate_up_k3_s8_t##tiles##_g##groups,     \
			    EXL3_MAXIMUM_BITS, false, EXL3_RELAXED_SLICES, \
			    tiles, groups)                                 \
	EXL3_DECODE_C_ENTRY(exl3_c_down_k2_s8_t##tiles##_g##groups,        \
			    EXL3_MINIMUM_BITS, true, EXL3_RELAXED_SLICES,  \
			    tiles, groups)                                 \
	EXL3_DECODE_C_ENTRY(exl3_c_down_k3_s8_t##tiles##_g##groups,        \
			    EXL3_MAXIMUM_BITS, true, EXL3_RELAXED_SLICES,  \
			    tiles, groups)
#define EXL3_DECODE_C_FAMILY(tiles, groups) \
	EXL3_DECODE_C_FAMILY_STAMP(tiles, groups)
#define EXL3_DECODE_C_RELAXED(tiles, groups) \
	EXL3_DECODE_C_RELAXED_STAMP(tiles, groups)
EXL3_DECODE_C_FAMILY(1, 1)
EXL3_DECODE_C_FAMILY(DECODE_C_TWO_OUTPUT_TILES, 1)
EXL3_DECODE_C_FAMILY(DECODE_C_FOUR_OUTPUT_TILES, 1)

EXL3_DECODE_C_RELAXED(DECODE_C_TWO_OUTPUT_TILES, 1)
EXL3_DECODE_C_RELAXED(DECODE_C_FOUR_OUTPUT_TILES, 1)
#undef EXL3_DECODE_C_RELAXED_STAMP
#undef EXL3_DECODE_C_FAMILY_STAMP
#undef EXL3_DECODE_C_RELAXED
#undef EXL3_DECODE_C_FAMILY
#undef EXL3_DECODE_C_ENTRY
