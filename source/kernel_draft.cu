/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_draft.cu: the draft's own kernels (dspark_v1/DESIGN.md 4.2): the verify kernel of a round, and the Markov
 * bias and the confidence of the draft's head. Device code only, built once per architecture into the module
 * "kernel_draft". Contract: include/anemone/draft_records.h. Here:
 *   - The verify kernel is one warp: lane i (1 .. 7) finds the rows of positions P + i - 1 and P + i of the verifying
 *     session by a scan of the step's live rows (at most the record's, at most 8), the ballot of the matches gives a,
 *     and lane e hands y_e to every lane. Lane j writes row j of the draft's step area and view j; lane 0 the round
 *     and the block.
 *   - The Markov bias: lane l holds E_(8 l) .. E_(8 l + 7) in registers for every entry its warp takes; a warp reads an
 *     entry's 512 bytes of W2 as one 16-byte load a lane.
 *   - The confidence: block r, one row; thread t's chain, then the lanes, then warp 0 the warps' sums.
 *   - No atomic operation; nothing depends on timing.
 */
#include "kernel_linear_common.cuh"

#include "anemone/draft_records.h"

#define EVERY_LANE 0xFFFFFFFFu
#define VALUES_PER_LOAD_OF_SIXTEEN 8u /* bfloat16 values in a 16-byte load */
#define VALUES_PER_WORD 2u
#define LOW_HALF_MASK 0xFFFFu
#define NO_BATCH_ROW DRAFT_BATCH_MOST_ROWS /* no row of a step of any program */
#define CONFIDENCE_WARPS (DRAFT_CONFIDENCE_THREADS / THREADS_PER_WARP)
#define MARKOV_LOADS \
	(DRAFT_MARKOV_ROW_BYTES / sizeof(uint4)) /* of a row: a load a lane */

static_assert(MARKOV_LOADS == THREADS_PER_WARP &&
		      CONFIDENCE_WARPS <= THREADS_PER_WARP,
	      "a Markov row is a warp's loads; warp 0 adds the warps' sums");

/* The eight bfloat16 values of a 16-byte load as float32, widened exactly. */
__device__ static __forceinline__ void
draft_unpack(uint4 load, float values[VALUES_PER_LOAD_OF_SIXTEEN])
{
	const uint32_t words[] = { load.x, load.y, load.z, load.w };
	for (uint32_t value = 0; value < VALUES_PER_LOAD_OF_SIXTEEN; value++) {
		const uint32_t word = words[value / VALUES_PER_WORD];
		values[value] = linear_bfloat16_value(
			value % VALUES_PER_WORD == 0 ? word & LOW_HALF_MASK :
						       word >> BITS_PER_HALF);
	}
}

/* The sum of a value of every lane of a warp, by exchanges at distance 16, 8, 4, 2 and 1: the same on every lane. */
__device__ static __forceinline__ float draft_warp_sum(float value)
{
	for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0;
	     distance >>= 1u) {
		value = value +
			__shfl_xor_sync(EVERY_LANE, value, (int)distance);
	}
	return value;
}

/* The token of a draft's result area (its one row's), area `area` of those at results_offset. */
__device__ static __forceinline__ int32_t draft_result_token(
	const unsigned char *arena, uint64_t results_offset, uint32_t area)
{
	return ((const struct step_row_result
			 *)(arena + results_offset +
			    (size_t)area * STEP_DRAFT_RESULT_BYTES +
			    STEP_RESULT_BYTES))
		->token_id;
}

/* The result of a step's row: in its one area, or with SLAB_RESULTS in its slab's (draft_records.h). */
__device__ static __forceinline__ const struct step_row_result *
verify_result(const unsigned char *arena, uint64_t result_offset, bool slabs,
	      uint32_t row)
{
	const uint32_t slab = slabs ? row / DRAFT_SLAB_ROWS : 0u;
	return (const struct step_row_result *)(arena + result_offset +
						(size_t)slab *
							DRAFT_SLAB_RESULT_BYTES +
						STEP_RESULT_BYTES) +
	       (row - slab * DRAFT_SLAB_ROWS);
}

extern "C" __global__ void __launch_bounds__(VERIFY_MATCHES_THREADS, 1)
	verify_matches_entry(struct verify_matches_arguments arguments,
			     unsigned char *arena)
{
	const struct step_block block =
		*(const struct step_block *)(arena + arguments.step_offset);
	const struct step_row_record *const rows =
		(const struct step_row_record *)(arena + arguments.step_offset +
						 STEP_BLOCK_BYTES_VALUE);
	const bool table = (arguments.flags & DRAFT_ROUND_TABLE) != 0;
	const bool slabs = (arguments.flags & DRAFT_SLAB_RESULTS) != 0;
	const bool batched = (arguments.flags & DRAFT_BATCHED) != 0;
	const uint32_t lane = threadIdx.x,
		       row_count = block.row_count < arguments.row_count ?
					   block.row_count :
					   arguments.row_count;
	uint32_t anchor_row = NO_BATCH_ROW, drafts_rows = 0;
	for (uint32_t row = 0; row < row_count; row++) {
		const bool drafts = (rows[row].flags & STEP_ROW_DRAFTS) != 0;
		anchor_row = drafts && drafts_rows == arguments.verifier ?
				     row :
				     anchor_row;
		drafts_rows += drafts ? 1u : 0u;
	}
	LINEAR_DEVICE_CHECK(
		blockDim.x == VERIFY_MATCHES_THREADS &&
		arguments.row_count <=
			(table ? DRAFT_BATCH_MOST_ROWS : LINEAR_MAXIMUM_ROWS) &&
		block.row_count <= arguments.row_count &&
		(table || drafts_rows == 1u) &&
		(!batched || (drafts_rows <= DRAFT_BATCH_SLOTS &&
			      arguments.verifier < DRAFT_BATCH_SLOTS)) &&
		arguments.draft_positions >= 1u &&
		arguments.draft_positions <= STEP_DRAFT_MOST);
	struct step_round *const round =
		(struct step_round *)(arena + arguments.round_offset) +
		(table && anchor_row != NO_BATCH_ROW ?
			 rows[anchor_row].session_slot :
			 0u);
	LINEAR_DEVICE_CHECK(!table || anchor_row == NO_BATCH_ROW ||
			    rows[anchor_row].session_slot <
				    arguments.round_capacity);
	struct step_block *const draft_block =
		(struct step_block *)(arena + arguments.draft_step_offset);
	/* BATCHED: verifier 0 heads every slab block (the DRAFTS rows' count x R rows in all) and every position's view
	 * (the DRAFTS rows' count D); verifier v's rows from v R on, its row j slot v of view j (draft_records.h) */
	const uint32_t base = batched ? arguments.verifier * arguments.draft_positions : 0u;
	if (batched && arguments.verifier == 0 && lane < arguments.draft_slabs) {
		const uint32_t total = drafts_rows * arguments.draft_positions,
			       first = lane * DRAFT_SLAB_ROWS;
		const struct step_block header = {
			block.sequence,
			total <= first			? 0u :
			total - first < DRAFT_SLAB_ROWS ? total - first :
							  DRAFT_SLAB_ROWS,
			0
		};
		*(struct step_block *)(arena + arguments.draft_step_offset +
				       (size_t)lane * DRAFT_SLAB_STEP_BYTES) =
			header;
	}
	/* WIDE_STEP: verifier 0 heads the step block of every verifier's rows behind the slab blocks too */
	const bool wide_step = (arguments.flags & DRAFT_WIDE_STEP) != 0;
	unsigned char *const wide_block =
		arena + arguments.draft_step_offset +
		(size_t)arguments.draft_slabs * DRAFT_SLAB_STEP_BYTES;
	if (wide_step && arguments.verifier == 0 && lane == 0) {
		const struct step_block header = {
			block.sequence, drafts_rows * arguments.draft_positions,
			0
		};
		*(struct step_block *)wide_block = header;
	}
	if (batched && arguments.verifier == 0 && lane < arguments.draft_positions) {
		const struct step_block header = { block.sequence, drafts_rows,
						   0 };
		*(struct step_block *)(arena + arguments.draft_views_offset +
				       (size_t)lane * DRAFT_SLAB_STEP_BYTES) =
			header;
	}
	unsigned char *const views = arena + arguments.draft_views_offset;
	if (anchor_row ==
	    NO_BATCH_ROW) { /* no such verifier, or the driver's promise broken: nothing for the draft to run */
		const struct step_block empty = { block.sequence, 0, 0 };
		if (lane == 0 && table && !batched)
			*draft_block = empty;
		if (lane == 0 && !table) {
			round->sequence = block.sequence;
			round->session_slot = STEP_NO_SESSION;
			round->emitted_count = 0;
			round->anchor_position = 0;
			round->anchor_token = 0;
			round->proposal_count = 0;
			round->reserved_zero = 0;
			*draft_block = empty;
		}
		if (lane < arguments.draft_positions && !batched) {
			*(struct step_block *)(views +
					       (size_t)lane *
						       STEP_DRAFT_VIEW_BYTES) =
				empty;
		}
		return;
	}
	const struct step_row_record anchor = rows[anchor_row];
	int32_t drawn = STEP_NO_TOKEN, proposal = STEP_NO_TOKEN;
	bool has_proposal = false;
	for (uint32_t row = 0; row < row_count; row++) {
		const struct step_row_record candidate = rows[row];
		const bool is_session = candidate.session_slot ==
					anchor.session_slot;
		drawn = is_session && candidate.position ==
						anchor.position +
							(int32_t)lane - 1 ?
				verify_result(arena, arguments.result_offset,
					      slabs, row)
					->token_id :
				drawn;
		const bool is_proposal =
			is_session &&
			candidate.position == anchor.position + (int32_t)lane &&
			(candidate.flags & STEP_ROW_PROPOSED) != 0;
		proposal = is_proposal ? candidate.token_id : proposal;
		has_proposal = has_proposal || is_proposal;
	}
	const bool matches = lane != 0 && has_proposal && proposal == drawn &&
			     drawn != (int32_t)arguments.end_token_id;
	const uint32_t accepted =
		(uint32_t)__ffs(
			(int)~(__ballot_sync(EVERY_LANE, matches) >> 1u)) -
		1u;
	const uint32_t emitted = accepted + 1u;
	const int32_t anchor_token =
		__shfl_sync(EVERY_LANE, drawn, (int)emitted);
	const int32_t anchor_position = anchor.position + (int32_t)emitted;
	LINEAR_DEVICE_CHECK(anchor_token >= 0);
	if (lane == 0) {
		const struct step_block draft = { block.sequence,
						  arguments.draft_positions,
						  0 };
		round->sequence = block.sequence;
		round->session_slot = anchor.session_slot;
		round->emitted_count = emitted;
		round->anchor_position = anchor_position;
		round->anchor_token = anchor_token;
		round->proposal_count = arguments.draft_positions;
		round->reserved_zero = 0;
		if (!batched)
			*draft_block = draft;
	}
	if (lane < arguments.draft_positions) {
		const struct step_row_record row = {
			anchor_position + (int32_t)lane,
			lane == 0 ? anchor_token :
				    (int32_t)arguments.noise_token_id,
			anchor.session_slot, STEP_ROW_LOGITS
		};
		const uint32_t index = base + lane;
		struct step_row_record *const draft_row =
			batched ?
				(struct step_row_record
					 *)(arena +
					    arguments.draft_step_offset +
					    (size_t)(index / DRAFT_SLAB_ROWS) *
						    DRAFT_SLAB_STEP_BYTES +
					    STEP_BLOCK_BYTES_VALUE) +
					index % DRAFT_SLAB_ROWS :
				(struct step_row_record
					 *)(arena +
					    arguments.draft_step_offset +
					    STEP_BLOCK_BYTES_VALUE) +
					lane;
		*draft_row = row;
		if (wide_step)
			((struct step_row_record
				  *)(wide_block +
				     STEP_BLOCK_BYTES_VALUE))[index] = row;
		if (batched) {
			((struct step_row_record
				  *)(views +
				     (size_t)lane * DRAFT_SLAB_STEP_BYTES +
				     STEP_BLOCK_BYTES_VALUE))[arguments.verifier] =
				row;
		} else {
			const struct step_block view = { block.sequence, 1, 0 };
			unsigned char *const view_place =
				views + (size_t)lane * STEP_DRAFT_VIEW_BYTES;
			*(struct step_block *)view_place = view;
			*(struct step_row_record *)(view_place +
						    STEP_BLOCK_BYTES_VALUE) =
				row;
		}
	}
}

__device__ static __forceinline__ void markov_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
						  uint32_t b0, uint32_t b1)
{
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
		     "{%0,%1,%2,%3};"
		     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
		     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* BATCHED: position j's Markov bias of every slot below view 0's row count on the tensor cores. Warp w of block b takes
 * the entries 16 (8 b + w) .. + 15: per 32 of the 256 values a lane loads 16 bytes of W2's rows g and g + 8 (values
 * 8 t .. 8 t + 7) and 8 bytes of each of slot g's embedding halves; the two instructions take them as kernel_head_mma
 * does (the same k order for both). d0, d1: entry g, slots 2 t, 2 t + 1; d2, d3: entry g + 8. Block 0 copies each
 * slot's W1 row to its Markov row. */
__device__ static void
markov_bias_slots_mma(const struct draft_markov_bias_arguments &arguments,
		      unsigned char *arena)
{
	const struct step_block *const view =
		(const struct step_block *)(arena +
					    arguments.draft_step_offset);
	const uint32_t slots = view->row_count < DRAFT_BATCH_SLOTS ?
				       view->row_count :
				       DRAFT_BATCH_SLOTS;
	const uint32_t lane = threadIdx.x % THREADS_PER_WARP, quad = lane & 3u, column = lane >> 2;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP;
	/* each lane's slot (its column g) and that slot's token */
	const uint32_t slot = column < slots ? column : 0u;
	const int32_t token =
		arguments.position == 0 ?
			((const struct step_row_record *)(view + 1))[slot]
				.token_id :
			((const struct step_row_result
				  *)(arena + arguments.results_offset +
				     (size_t)(arguments.position - 1u) *
					     DRAFT_SLAB_RESULT_BYTES +
				     STEP_RESULT_BYTES))[slot]
				.token_id;
	LINEAR_DEVICE_CHECK(blockDim.x == DRAFT_MARKOV_THREADS && token >= 0 &&
			    (uint32_t)token < arguments.embedding_rows);
	if (blockIdx.x == 0) /* warp w copies slot w's W1 row (a uint4 a lane) */
		for (uint32_t copied = warp; copied < slots;
		     copied += DRAFT_MARKOV_THREADS / THREADS_PER_WARP) {
			const int32_t copied_token =
				arguments.position == 0 ?
					((const struct step_row_record
						  *)(view + 1))[copied]
						.token_id :
					((const struct step_row_result
						  *)(arena +
						     arguments.results_offset +
						     (size_t)(arguments.position -
							      1u) *
							     DRAFT_SLAB_RESULT_BYTES +
						     STEP_RESULT_BYTES))[copied]
						.token_id;
			((uint4 *)(arena + arguments.markov_offset +
				   (size_t)copied *
					   DRAFT_MARKOV_ROW_BYTES))[lane] =
				((const uint4 *)(arena +
						 arguments.embedding_offset))
					[(size_t)copied_token * MARKOV_LOADS +
					 lane];
		}
	const uint32_t first =
		(blockIdx.x * (DRAFT_MARKOV_THREADS / THREADS_PER_WARP) +
		 warp) *
		16u;
	if (first >= arguments.vocabulary_size)
		return;
	const uint32_t last = arguments.vocabulary_size - 1u;
	const uint32_t low = min(first + column, last), high = min(first + column + 8u, last);
	const uint4 *const head_low = (const uint4 *)(arena + arguments.head_offset) + (size_t)low * MARKOV_LOADS + quad;
	const uint4 *const head_high = (const uint4 *)(arena + arguments.head_offset) + (size_t)high * MARKOV_LOADS + quad;
	const uint2 *const embedded =
		(const uint2 *)((const uint4 *)(arena + arguments.embedding_offset) + (size_t)token * MARKOV_LOADS) +
		2u * quad;
	float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
	for (uint32_t step = 0; step < MARKOV_LOADS / 4u; step++) {
		const uint4 a_low = head_low[4u * step], a_high = head_high[4u * step];
		const uint2 b_first = embedded[8u * step], b_second = embedded[8u * step + 1u];
		markov_mma(sums, a_low.x, a_high.x, a_low.y, a_high.y, b_first.x, b_first.y);
		markov_mma(sums, a_low.z, a_high.z, a_low.w, a_high.w, b_second.x, b_second.y);
	}
#pragma unroll
	for (uint32_t pair = 0; pair < 2u; pair++) {
		const uint32_t row = 2u * quad + pair;
		if (row >= slots)
			continue;
#pragma unroll
		for (uint32_t half = 0; half < 2u; half++) {
			const uint32_t entry = first + column + 8u * half;
			if (entry < arguments.vocabulary_size) {
				float *const logit = (float *)(arena + arguments.logits_offset) +
						     (size_t)row * arguments.vocabulary_size + entry;
				*logit = *logit + sums[2u * half + pair];
			}
		}
	}
}

extern "C" __global__ void __launch_bounds__(DRAFT_MARKOV_THREADS, 1)
	draft_markov_bias(struct draft_markov_bias_arguments arguments,
			  unsigned char *arena)
{
	const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t entry = blockIdx.x * DRAFT_MARKOV_ENTRIES_PER_BLOCK +
			       threadIdx.x / THREADS_PER_WARP;
	if ((arguments.flags & DRAFT_BATCHED) != 0) {
		markov_bias_slots_mma(arguments, arena);
		return;
	}
	const int32_t token =
		arguments.position == 0 ?
			((const struct step_row_record
				  *)(arena + arguments.draft_step_offset +
				     STEP_BLOCK_BYTES_VALUE))
				->token_id :
			draft_result_token(arena, arguments.results_offset,
					   arguments.position - 1u);
	LINEAR_DEVICE_CHECK(blockDim.x == DRAFT_MARKOV_THREADS && token >= 0 &&
			    (uint32_t)token < arguments.embedding_rows);
	const uint4 embedded =
		((const uint4 *)(arena + arguments.embedding_offset))
			[(size_t)token * MARKOV_LOADS + lane];
	if (blockIdx.x == 0 && threadIdx.x < MARKOV_LOADS) {
		((uint4 *)(arena + arguments.markov_offset))[lane] = embedded;
	}
	if (entry >= arguments.vocabulary_size) {
		return;
	}
	float embedding[VALUES_PER_LOAD_OF_SIXTEEN],
		head[VALUES_PER_LOAD_OF_SIXTEEN], partial = 0.0f;
	draft_unpack(embedded, embedding);
	draft_unpack(((const uint4 *)(arena + arguments.head_offset))
			     [(size_t)entry * MARKOV_LOADS + lane],
		     head);
	for (uint32_t value = 0; value < VALUES_PER_LOAD_OF_SIXTEEN; value++) {
		partial = fmaf(head[value], embedding[value], partial);
	}
	partial = draft_warp_sum(partial);
	if (lane == 0) {
		float *const logit =
			(float *)(arena + arguments.logits_offset) + entry;
		*logit = *logit + partial;
	}
}

extern "C" __global__ void __launch_bounds__(DRAFT_CONFIDENCE_THREADS, 1)
	draft_confidence_entry(struct draft_confidence_arguments arguments,
			       unsigned char *arena)
{
	__shared__ float warp_sums[CONFIDENCE_WARPS];
	const uint32_t position = blockIdx.y, thread = threadIdx.x,
		       lane = thread % THREADS_PER_WARP;
	const uint32_t step_rows =
		((const struct step_block *)(arena +
					     arguments.draft_step_offset))
			->row_count;
	const bool table = (arguments.flags & DRAFT_ROUND_TABLE) != 0;
	const bool batched = (arguments.flags & DRAFT_BATCHED) != 0;
	/* ROUND_TABLE: a draft step of any row stands for its draft_positions rows; BATCHED: view 0, the verifier's slot
	 * below its row count (draft_records.h) */
	const bool drafts = batched ? arguments.verifier < step_rows : step_rows != 0;
	const uint32_t rows = table ? (drafts ? arguments.draft_positions : 0u) :
			      step_rows < arguments.draft_positions ? step_rows :
								     arguments.draft_positions;
	if (table && !drafts)
		return; /* a verifier without a DRAFTS row: no round */
	const uint32_t slot = batched ? arguments.verifier : 0u;
	struct step_round *const round =
		(struct step_round *)(arena + arguments.round_offset) +
		(table ? ((const struct step_row_record
				   *)(arena + arguments.draft_step_offset +
				      STEP_BLOCK_BYTES_VALUE))[slot]
				 .session_slot :
			 0u);
	LINEAR_DEVICE_CHECK(blockDim.x == DRAFT_CONFIDENCE_THREADS &&
			    gridDim.y == STEP_DRAFT_MOST &&
			    step_rows <= (batched ?
						  DRAFT_BATCH_SLOTS :
						  arguments.draft_positions) &&
			    arguments.draft_positions <= STEP_DRAFT_MOST);
	if (position >= rows) {
		if (thread == 0) {
			round->proposals[position] = 0;
			round->confidence_logits[position] = 0.0f;
		}
		return;
	}
	const uint16_t *const collapse =
		(const uint16_t *)(arena + arguments.collapse_offset) +
		(size_t)position * MODEL_HIDDEN;
	const uint16_t *const markov =
		(const uint16_t *)(arena + arguments.markov_offset) +
		((size_t)position * (batched ? DRAFT_BATCH_SLOTS : 1u) + slot) *
			MODEL_MARKOV_RANK;
	const uint16_t *const weights =
		(const uint16_t *)(arena + arguments.weight_offset);
	float chain = 0.0f;
	for (uint32_t input = thread; input < MODEL_DRAFT_CONFIDENCE_INPUTS;
	     input += DRAFT_CONFIDENCE_THREADS) {
		const uint32_t value_bits =
			input < MODEL_HIDDEN ? collapse[input] :
					       markov[input - MODEL_HIDDEN];
		chain = fmaf(linear_bfloat16_value(weights[input]),
			     linear_bfloat16_value(value_bits), chain);
	}
	chain = draft_warp_sum(chain);
	if (lane == 0) {
		warp_sums[thread / THREADS_PER_WARP] = chain;
	}
	__syncthreads();
	if (thread == 0) {
		float score = 0.0f;
		for (uint32_t warp = 0; warp < CONFIDENCE_WARPS; warp++) {
			score = score + warp_sums[warp];
		}
		round->proposals[position] =
			batched ?
				((const struct step_row_result
					  *)(arena + arguments.results_offset +
					     (size_t)position *
						     DRAFT_SLAB_RESULT_BYTES +
					     STEP_RESULT_BYTES))[slot]
					.token_id :
				draft_result_token(arena,
						   arguments.results_offset,
						   position);
		round->confidence_logits[position] = score;
	}
}
