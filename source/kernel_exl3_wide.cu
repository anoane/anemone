/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_exl3_wide.cu: EXL3 routed-expert products and transforms for the TP passes' wide (chunked) and thin calls,
 * by exl3_v3_claude (R&D epoch02/src42 sources, unchanged arithmetic). Device code only, one module
 * "kernel_exl3_wide". Contract: anemone/exl3_records.h (the 80-byte exl3_expert_record) and kernel_exl3.cu's
 * launches: every product writes the words of kernel_exl3.cu's exl3_{gate_up,down}_k{2,3} (slice s an
 * in-order m16n8k16 chain over its K tiles from +0, the four slices' partials added in order from +0), every
 * transform those of exl3_{input,activate,finish}. Only the work mapping differs:
 *   - exl3_c_{gate_up,down}_k{2,3}_t8_g1: a warp decodes 8 adjacent output tiles of a K tile; grid x = width
 *     / 128 (width / 16 / 8), y = job_count, z = 2 (gate/up) or 1; block 128 (4 slices).
 *   - exl3_c_{gate_up,down}_k{2,3}_t4_g4p (wide calls: the wide select's runs of 8-row jobs): the CTA of
 *     every 4th job of a run decodes each tile once for up to 4 jobs of that run; CTAs of the other jobs exit;
 *     groups past the run issue no load and no MMA. grid x = width / 64, y = job_count, z = 2 / 1; block 128.
 *   - exl3_w_{input,activate,finish}: one warp a 128-channel block; grid x = (blocks + 3) / 4, y =
 *     job_count, z = 2 (input) / 1; block 128; no shared memory.
 * Built like every module with --fmad=false.
 */
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

#define DECODE_C_MAXIMUM_TILES 8u
#define DECODE_C_MAXIMUM_GROUPS 4u
#define DECODE_C_WIDE_GROUPS 8u

/* One K step of the warp: trellis word pairs of its output tiles and the
 * activation fragments of its job groups. Entries past the stamped tile and
 * group counts are never touched. */
template <uint32_t operand_groups> struct decode_c_stage_of {
	uint32_t high[DECODE_C_MAXIMUM_TILES];
	uint32_t low[DECODE_C_MAXIMUM_TILES];
	uint32_t operand_low[operand_groups];
	uint32_t operand_high[operand_groups];
};
/* Capacity for a launch's groups: the four-group layout up to four. */
template <uint32_t groups>
using decode_c_stage = decode_c_stage_of<(groups > DECODE_C_MAXIMUM_GROUPS ?
						  DECODE_C_WIDE_GROUPS :
						  DECODE_C_MAXIMUM_GROUPS)>;

/* Operands written earlier in the same launch (fused form) must bypass the
 * non-coherent read-only path; trellis words are immutable and use it. */
template <bool coherent>
__device__ static __forceinline__ uint32_t
decode_c_operand_load(const uint32_t *address)
{
	return coherent ? __ldcg(address) : __ldg(address);
}

/* predicated: groups from live_groups on (past the owned jobs) load nothing
 * and multiply nothing; their sums are never stored. */
template <uint32_t tiles, uint32_t groups, bool coherent,
	  bool predicated = false>
__device__ static __forceinline__ void
decode_c_load(decode_c_stage<groups> *stage, const uint32_t *word_pointer,
	      const uint32_t *successor_pointer,
	      const uint32_t *const *operand_pointer,
	      const uint32_t *operand_mask, uint32_t ring_words, bool odd_step,
	      uint32_t live_groups = groups)
{
	for (uint32_t tile = 0; tile < tiles; tile++) {
		stage->high[tile] = __ldg(word_pointer + tile * ring_words);
		stage->low[tile] = __ldg(successor_pointer + tile * ring_words);
	}
	/* Operand pointers sit at the even step; the odd step is one K tile
	 * (eight words) further, an immediate offset. */
	const uint32_t step_words = odd_step ? EXL3_TILE / DECODE_C_PAIR : 0u;
	for (uint32_t group = 0; group < groups; group++) {
		if (predicated && group >= live_groups)
			continue;
		stage->operand_low[group] =
			decode_c_operand_load<coherent>(operand_pointer[group] +
							step_words) &
			operand_mask[group];
		stage->operand_high[group] =
			decode_c_operand_load<coherent>(
				operand_pointer[group] + step_words +
				DECODE_C_INPUT_HALF / DECODE_C_PAIR) &
			operand_mask[group];
	}
}

/* Tensor-core byte sums (candidate): an m16n8k32 u8 product whose B selects
 * chunk DECODE_C_SUM_CHUNK(n) for column n returns, in every lane, the byte
 * sums of the lane's own four A words: c0 = a0, c1 = a2, c2 = a1, c3 = a3.
 * Integer sums of bytes times 0 or 1 are exact; the accumulator adds 0x6400. */
#define DECODE_C_SUM_CHUNK(column)  \
	((column) / DECODE_C_PAIR + \
	 DECODE_C_LANE_GROUP * ((column) % DECODE_C_PAIR))
#define DECODE_C_SUM_HIGH_CHUNKS 4u

__device__ static __forceinline__ void
decode_c_tensor_sums(uint32_t low_first, uint32_t high_first,
		     uint32_t low_second, uint32_t high_second,
		     uint32_t select_low, uint32_t select_high,
		     uint32_t *first_pair, uint32_t *second_pair)
{
	uint32_t sums[DECODE_C_CHAINS] = { EXL3_MUL1_BASE, EXL3_MUL1_BASE,
					   EXL3_MUL1_BASE, EXL3_MUL1_BASE };
	asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
		     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
		     : "+r"(sums[0]), "+r"(sums[1]), "+r"(sums[2]),
		       "+r"(sums[3])
		     : "r"(low_first), "r"(low_second), "r"(high_first),
		       "r"(high_second), "r"(select_low), "r"(select_high));
	*first_pair = __byte_perm(sums[0], sums[1], DECODE_C_HALF_PAIR_SELECT);
	*second_pair = __byte_perm(sums[2], sums[3], DECODE_C_HALF_PAIR_SELECT);
}

__device__ static __forceinline__ uint32_t decode_c_scale_pair(uint32_t packed)
{
	const uint32_t scale = DECODE_C_SCALE_PAIR;
	const uint32_t bias = DECODE_C_BIAS_PAIR;
	const __half2 value = __hfma2(*(const __half2 *)&packed,
				      *(const __half2 *)&scale,
				      *(const __half2 *)&bias);
	return *(const uint32_t *)&value;
}

template <uint32_t bits, uint32_t tiles, uint32_t groups, bool tensor_sums,
	  bool predicated = false>
__device__ static __forceinline__ void
decode_c_compute(const decode_c_stage<groups> *stage,
		 const uint32_t *symbol_shift, uint32_t select_low,
		 uint32_t select_high,
		 float (*accumulators)[groups][DECODE_C_CHAINS],
		 uint32_t live_groups = groups)
{
	for (uint32_t tile = 0; tile < tiles; tile++) {
		const uint64_t window =
			((uint64_t)stage->high[tile] << EXL3_WORD_BITS) |
			stage->low[tile];
		uint32_t weights[DECODE_C_CHAINS];
		if (tensor_sums) {
			uint32_t product[DECODE_C_VALUES_PER_LANE];
			for (uint32_t symbol = 0;
			     symbol < DECODE_C_VALUES_PER_LANE; symbol++)
				product[symbol] =
					((uint32_t)(window >>
						    symbol_shift[symbol]) &
					 EXL3_HALF_MASK) *
					EXL3_MUL1_MULTIPLIER;
			/* Fragments 0/1 hold symbols (0,1)/(4,5), 2/3 hold
			 * (2,3)/(6,7). */
			decode_c_tensor_sums(product[0], product[1], product[4],
					     product[5], select_low,
					     select_high, &weights[0],
					     &weights[1]);
			decode_c_tensor_sums(product[2], product[3], product[6],
					     product[7], select_low,
					     select_high, &weights[2],
					     &weights[3]);
			for (uint32_t fragment = 0; fragment < DECODE_C_CHAINS;
			     fragment++)
				weights[fragment] =
					decode_c_scale_pair(weights[fragment]);
		} else {
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
		}
		for (uint32_t group = 0; group < groups; group++)
			if (!predicated || group < live_groups)
				asm volatile(
					"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
					"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
					"{%0,%1,%2,%3};"
					: "+f"(accumulators[tile][group][0]),
					  "+f"(accumulators[tile][group][1]),
					  "+f"(accumulators[tile][group][2]),
					  "+f"(accumulators[tile][group][3])
					: "r"(weights[0]), "r"(weights[1]),
					  "r"(weights[2]), "r"(weights[3]),
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

/* Split form (split = true): the CTA's warps run slices slice_base + warp of
 * the slices-part order and store their raw chains into split_partials
 * [slice][projection][pair][outputs] (FP32, slice stride in floats); a later
 * pass adds the slices in order from +0, exactly as the slice-zero epilogue. */
template <uint32_t bits, uint32_t slices, uint32_t tiles, uint32_t groups,
	  bool coherent, bool zero_absent, bool tensor_sums, bool split = false,
	  bool predicated = false>
__device__ static void
decode_c_product_core(const struct decode_c_place *place, unsigned char *arena,
		      uint32_t slice_base = 0, float *split_partials = nullptr,
		      uint64_t slice_stride = 0)
{
	const struct expert_job_record *const job = place->job;
	const uint32_t owned_jobs = place->owned_jobs;
	const uint32_t projection = place->projection;
	const uint32_t inputs = place->inputs;
	const uint32_t outputs = place->outputs;
	const uint32_t lane = threadIdx.x % DECODE_C_WARP;
	const uint32_t slice =
		(split ? slice_base : 0u) + threadIdx.x / DECODE_C_WARP;
	const uint32_t lane_group = lane / DECODE_C_LANE_GROUP;
	const uint32_t lane_place = lane % DECODE_C_LANE_GROUP;
	const uint32_t *const matrix =
		(const uint32_t *)(arena + place->matrix_offset);
	const uint32_t tile_count = inputs / EXL3_TILE;
	const uint32_t first = tile_count * slice / slices;
	const uint32_t last = tile_count * (slice + 1u) / slices;
	const uint32_t operand_words_per_tile = EXL3_TILE / DECODE_C_PAIR;
	/* FP16 pairs of each group's activation fragment, as 32-bit words.
	 * Columns without a row (and groups past the owned jobs) read row zero
	 * again, or with zero_absent multiply +0; their sums are never stored
	 * and the stored columns are independent of them. Measured: the zero
	 * form saves HMMA energy about as much as its masks cost. */
	const uint32_t *operand_pointer[groups];
	uint32_t operand_mask[groups];
	for (uint32_t group = 0; group < groups; group++) {
		const struct expert_job_record *const group_job =
			job + (group < owned_jobs ? group : 0);
		const bool present = group < owned_jobs &&
				     lane_group < group_job->row_count;
		operand_mask[group] =
			present || !zero_absent ? DECODE_C_FULL_MASK : 0u;
		const uint32_t operand_pair =
			group_job->pair_index[present ? lane_group : 0];
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
	/* B of the tensor-core byte sums: lane (column g, place t) holds
	 * chunks t and t + 4 of column g. */
	const uint32_t sum_chunk = DECODE_C_SUM_CHUNK(lane_group);
	const uint32_t select_low =
		sum_chunk == lane_place ? DECODE_C_BYTE_UNITS : 0u;
	const uint32_t select_high =
		sum_chunk == lane_place + DECODE_C_SUM_HIGH_CHUNKS ?
			DECODE_C_BYTE_UNITS :
			0u;
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
	decode_c_stage<groups> stage_a, stage_b;
	const uint32_t steps = last - first;
	if (steps > 0)
		decode_c_load<tiles, groups, coherent, predicated>(
			&stage_a, word_pointer, successor_pointer,
			operand_pointer, operand_mask, ring_words, false,
			owned_jobs);
	uint32_t step = 0;
	for (; step + 1u < steps; step += 2u) {
		decode_c_load<tiles, groups, coherent, predicated>(
			&stage_b, word_pointer + tile_stride,
			successor_pointer + tile_stride, operand_pointer,
			operand_mask, ring_words, true, owned_jobs);
		decode_c_compute<bits, tiles, groups, tensor_sums, predicated>(
			&stage_a, symbol_shift, select_low, select_high,
			accumulators, owned_jobs);
		word_pointer += 2u * tile_stride;
		successor_pointer += 2u * tile_stride;
		for (uint32_t group = 0; group < groups; group++)
			operand_pointer[group] += 2u * operand_words_per_tile;
		if (step + 2u < steps)
			decode_c_load<tiles, groups, coherent, predicated>(
				&stage_a, word_pointer, successor_pointer,
				operand_pointer, operand_mask, ring_words,
				false, owned_jobs);
		decode_c_compute<bits, tiles, groups, tensor_sums, predicated>(
			&stage_b, symbol_shift, select_low, select_high,
			accumulators, owned_jobs);
	}
	if (step < steps)
		decode_c_compute<bits, tiles, groups, tensor_sums, predicated>(
			&stage_a, symbol_shift, select_low, select_high,
			accumulators, owned_jobs);
	if (split) {
		float *const out =
			split_partials + (uint64_t)slice * slice_stride;
		for (uint32_t tile = 0; tile < tiles; tile++)
			for (uint32_t group = 0;
			     group < owned_jobs && group < groups; group++) {
				const struct expert_job_record *const group_job =
					job + group;
				for (uint32_t chain = 0;
				     chain < DECODE_C_CHAINS; chain++) {
					const uint32_t column =
						lane_place *
							DECODE_C_COLUMN_PAIR +
						chain % DECODE_C_PAIR;
					const uint32_t output =
						(place->tile_group * tiles +
						 tile) * EXL3_TILE +
						lane_group +
						(chain / DECODE_C_PAIR) *
							DECODE_C_OUTPUT_HALF;
					if (column < group_job->row_count)
						out[((uint64_t)projection *
							     place->pair_count +
						     group_job->pair_index
							     [column]) *
							    outputs +
						    output] =
							accumulators[tile][group]
								    [chain];
				}
			}
		return;
	}
	/* The split form never reaches the slice exchange; size it for one. */
	__shared__ float partial[split ? 1u : slices][tiles][groups]
				[DECODE_C_WARP][DECODE_C_CHAINS];
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
	  uint32_t groups, bool zero_absent, bool tensor_sums,
	  bool predicated = false>
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
	decode_c_product_core<bits, slices, tiles, groups, false, zero_absent,
			      tensor_sums, false, predicated>(&place, arena);
}

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "anemone/exl3_records.h"
#include "anemone/linear_records.h"

#define TRANSFORM_W_WARP 32u
#define TRANSFORM_W_VALUES 4u
#define TRANSFORM_W_LANE_STRIDE 32u
#define TRANSFORM_W_FULL_MASK 0xffffffffu
#define TRANSFORM_W_HIGH_PAIR 2u

__device__ static __forceinline__ void transform_w_hadamard(float *value)
{
	const uint32_t lane = threadIdx.x % TRANSFORM_W_WARP;
	for (uint32_t stride = 1; stride < TRANSFORM_W_LANE_STRIDE;
	     stride *= EXL3_RADIX) {
		for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++) {
			const float other = __shfl_xor_sync(
				TRANSFORM_W_FULL_MASK, value[index], stride);
			value[index] = (lane & stride) == 0 ?
					       value[index] + other :
					       other - value[index];
		}
	}
	/* Stride 32: (0,1) and (2,3); stride 64: (0,2) and (1,3). */
	for (uint32_t low = 0; low < TRANSFORM_W_VALUES;
	     low += TRANSFORM_W_HIGH_PAIR) {
		const float a = value[low], b = value[low + 1u];
		value[low] = a + b;
		value[low + 1u] = a - b;
	}
	for (uint32_t low = 0; low < TRANSFORM_W_HIGH_PAIR; low++) {
		const float a = value[low],
			    b = value[low + TRANSFORM_W_HIGH_PAIR];
		value[low] = a + b;
		value[low + TRANSFORM_W_HIGH_PAIR] = a - b;
	}
}

__device__ static __forceinline__ uint64_t
transform_w_codes_bytes(struct exl3_expert_record record)
{
	return (uint64_t)record.input_size * record.inner_size /
	       EXL3_BITS_PER_BYTE * record.bits_per_weight;
}

template <uint32_t warps>
__device__ static void transform_w_input(struct exl3_expert_record record,
					 unsigned char *arena)
{
	const struct expert_job_record *const job =
		(const struct expert_job_record *)(arena + record.jobs_offset) +
		blockIdx.y;
	const uint32_t block =
		blockIdx.x * warps + threadIdx.x / TRANSFORM_W_WARP;
	if (job->row_count == 0 || block >= record.input_size / EXL3_TRANSFORM)
		return;
	const uint32_t lane = threadIdx.x % TRANSFORM_W_WARP;
	const uint64_t matrix = blockIdx.z == 0 ? job->gate_offset :
						  job->up_offset;
	const half *const scales =
		(const half *)(arena + matrix +
			       transform_w_codes_bytes(record));
	const __nv_bfloat16 *const input =
		(const __nv_bfloat16 *)(arena + record.input_offset);
	half *const operands = (half *)(arena + record.operands_offset);
	const uint32_t first = block * EXL3_TRANSFORM + lane;
	float scale[TRANSFORM_W_VALUES];
	for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++)
		scale[index] = __half2float(
			scales[first + index * TRANSFORM_W_LANE_STRIDE]);
	for (uint32_t column = 0; column < job->row_count; column++) {
		const uint32_t pair = job->pair_index[column];
		const uint32_t row = pair / record.pairs_per_row;
		float value[TRANSFORM_W_VALUES];
		for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++)
			value[index] =
				__bfloat162float(
					input[(uint64_t)row * record.input_size +
					      first +
					      index * TRANSFORM_W_LANE_STRIDE]) *
				scale[index];
		transform_w_hadamard(value);
		half *const out = operands +
				  ((uint64_t)blockIdx.z * record.pair_count +
				   pair) * record.input_size +
				  first;
		for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++)
			out[index * TRANSFORM_W_LANE_STRIDE] = __float2half_rn(
				value[index] * EXL3_NORMALIZATION);
	}
}

template <uint32_t warps>
__device__ static void transform_w_activate(struct exl3_expert_record record,
					    unsigned char *arena)
{
	const struct expert_job_record *const job =
		(const struct expert_job_record *)(arena + record.jobs_offset) +
		blockIdx.y;
	const uint32_t block =
		blockIdx.x * warps + threadIdx.x / TRANSFORM_W_WARP;
	if (job->row_count == 0 || block >= record.inner_size / EXL3_TRANSFORM)
		return;
	const uint32_t lane = threadIdx.x % TRANSFORM_W_WARP;
	const uint32_t first = block * EXL3_TRANSFORM + lane;
	const uint64_t codes_bytes = transform_w_codes_bytes(record);
	const half *const gate_scale =
		(const half *)(arena + job->gate_offset + codes_bytes) +
		record.input_size;
	const half *const up_scale =
		(const half *)(arena + job->up_offset + codes_bytes) +
		record.input_size;
	const half *const down_scale =
		(const half *)(arena + job->down_offset + codes_bytes);
	const float *const products =
		(const float *)(arena + record.products_offset);
	for (uint32_t column = 0; column < job->row_count; column++) {
		const uint32_t pair = job->pair_index[column];
		float gate[TRANSFORM_W_VALUES], up[TRANSFORM_W_VALUES];
		for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++) {
			const uint32_t channel =
				first + index * TRANSFORM_W_LANE_STRIDE;
			gate[index] =
				products[(uint64_t)pair * record.inner_size +
					 channel];
			up[index] = products[((uint64_t)record.pair_count +
					      pair) * record.inner_size +
					     channel];
		}
		transform_w_hadamard(gate);
		transform_w_hadamard(up);
		float down[TRANSFORM_W_VALUES];
		for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++) {
			const uint32_t channel =
				first + index * TRANSFORM_W_LANE_STRIDE;
			const float gate_value =
				gate[index] * EXL3_NORMALIZATION *
				__half2float(gate_scale[channel]);
			const float up_value = up[index] * EXL3_NORMALIZATION *
					       __half2float(up_scale[channel]);
			const float bounded_gate =
				fminf(gate_value, record.activation_limit);
			const float bounded_up =
				fminf(fmaxf(up_value, -record.activation_limit),
				      record.activation_limit);
			const float hidden = bounded_gate /
					     (1.0f + expf(-bounded_gate)) *
					     bounded_up;
			((__nv_bfloat16 *)(arena + record.hidden_offset))
				[(uint64_t)pair * record.inner_size + channel] =
					__float2bfloat16_rn(hidden);
			down[index] =
				hidden * __half2float(down_scale[channel]);
		}
		transform_w_hadamard(down);
		half *const out = (half *)(arena + record.operands_offset) +
				  (uint64_t)pair * record.inner_size + first;
		for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++)
			out[index * TRANSFORM_W_LANE_STRIDE] = __float2half_rn(
				down[index] * EXL3_NORMALIZATION);
	}
}

template <uint32_t warps>
__device__ static void transform_w_finish(struct exl3_expert_record record,
					  unsigned char *arena)
{
	const struct expert_job_record *const job =
		(const struct expert_job_record *)(arena + record.jobs_offset) +
		blockIdx.y;
	const uint32_t block =
		blockIdx.x * warps + threadIdx.x / TRANSFORM_W_WARP;
	if (job->row_count == 0 || block >= record.input_size / EXL3_TRANSFORM)
		return;
	const uint32_t lane = threadIdx.x % TRANSFORM_W_WARP;
	const uint32_t first = block * EXL3_TRANSFORM + lane;
	const half *const scales =
		(const half *)(arena + job->down_offset +
			       transform_w_codes_bytes(record)) +
		record.inner_size;
	float scale[TRANSFORM_W_VALUES];
	for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++)
		scale[index] = __half2float(
			scales[first + index * TRANSFORM_W_LANE_STRIDE]);
	float *const values = (float *)(arena + record.products_offset);
	for (uint32_t column = 0; column < job->row_count; column++) {
		const uint32_t pair = job->pair_index[column];
		float *const row =
			values + (uint64_t)pair * record.input_size + first;
		float value[TRANSFORM_W_VALUES];
		for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++)
			value[index] = row[index * TRANSFORM_W_LANE_STRIDE];
		transform_w_hadamard(value);
		for (uint32_t index = 0; index < TRANSFORM_W_VALUES; index++)
			row[index * TRANSFORM_W_LANE_STRIDE] =
				value[index] * EXL3_NORMALIZATION *
				scale[index] * job->route_weight[column];
	}
}

#define TRANSFORM_W_WARPS 4u
#define EXL3_TRANSFORM_W_ENTRY(name, body)                                   \
	extern "C" __global__ void __launch_bounds__(                        \
		TRANSFORM_W_WARPS *TRANSFORM_W_WARP)                         \
		name(struct exl3_expert_record record, unsigned char *arena) \
	{                                                                    \
		body<TRANSFORM_W_WARPS>(record, arena);                      \
	}
EXL3_TRANSFORM_W_ENTRY(exl3_w_input, transform_w_input)
EXL3_TRANSFORM_W_ENTRY(exl3_w_activate, transform_w_activate)
EXL3_TRANSFORM_W_ENTRY(exl3_w_finish, transform_w_finish)
#undef EXL3_TRANSFORM_W_ENTRY

#define EXL3_DECODE_C_ENTRY(name, bits, down, slices, tiles, groups,      \
			    predicated)                                   \
	extern "C" __global__ void name(struct exl3_expert_record record, \
					unsigned char *arena)             \
	{                                                                 \
		exl3_decode_c_product<bits, down, slices, tiles, groups,  \
				      false, false, predicated>(record,   \
								arena);   \
	}
#define EXL3_DECODE_C_FAMILY(tiles, groups, suffix, predicated)            \
	EXL3_DECODE_C_ENTRY(exl3_c_gate_up_k2_t##tiles##_g##suffix,        \
			    EXL3_MINIMUM_BITS, false, EXL3_PRODUCT_SLICES, \
			    tiles, groups, predicated)                     \
	EXL3_DECODE_C_ENTRY(exl3_c_gate_up_k3_t##tiles##_g##suffix,        \
			    EXL3_MAXIMUM_BITS, false, EXL3_PRODUCT_SLICES, \
			    tiles, groups, predicated)                     \
	EXL3_DECODE_C_ENTRY(exl3_c_down_k2_t##tiles##_g##suffix,           \
			    EXL3_MINIMUM_BITS, true, EXL3_PRODUCT_SLICES,  \
			    tiles, groups, predicated)                     \
	EXL3_DECODE_C_ENTRY(exl3_c_down_k3_t##tiles##_g##suffix,           \
			    EXL3_MAXIMUM_BITS, true, EXL3_PRODUCT_SLICES,  \
			    tiles, groups, predicated)
EXL3_DECODE_C_FAMILY(8, 1, 1, false)
EXL3_DECODE_C_FAMILY(4, 4, 4p, true)
#undef EXL3_DECODE_C_FAMILY
#undef EXL3_DECODE_C_ENTRY
