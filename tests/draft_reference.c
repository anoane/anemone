/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft_reference.c: the host reference of the draft's own kernels. Contract: draft_reference.h.
 */
#include "draft_reference.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#define LANES 32u /* of a warp: the exchanges at distance 16 .. 1 */
#define MARKOV_VALUES_PER_LANE \
	8u /* of the Markov rank's 256: lane l takes 8 l .. 8 l + 7 */
#define CONFIDENCE_THREADS DRAFT_CONFIDENCE_THREADS
#define CONFIDENCE_WARPS (CONFIDENCE_THREADS / LANES)
#define BFLOAT16_SHIFT 16u
#define NO_ROW UINT32_MAX

static float value_of(uint16_t bits)
{
	const uint32_t widened = (uint32_t)bits << BFLOAT16_SHIFT;
	float value = 0.0f;
	memcpy(&value, &widened, sizeof(value));
	return value;
}

/* The value of lane 0 after lane l adds lane l ^ d's at d = 16, 8, 4, 2 and 1. */
static float warp_sum(float lanes[LANES])
{
	for (uint32_t distance = LANES >> 1u; distance != 0; distance >>= 1u) {
		float next[LANES];
		for (uint32_t lane = 0; lane < LANES; lane++) {
			next[lane] = lanes[lane] + lanes[lane ^ distance];
		}
		memcpy(lanes, next, sizeof(next));
	}
	return lanes[0];
}

/* The row of the session at a position, or NO_ROW. */
static uint32_t row_at(const struct step_row_record *rows, uint32_t row_count,
		       uint32_t session_slot, int32_t position)
{
	uint32_t found = NO_ROW;
	for (uint32_t row = 0; row < row_count; row++) {
		found = found == NO_ROW &&
					rows[row].session_slot ==
						session_slot &&
					rows[row].position == position ?
				row :
				found;
	}
	return found;
}

void test_draft_verify(const struct step_block *block,
		       const struct step_row_record *rows,
		       const struct step_row_result *results,
		       uint32_t record_row_count, uint32_t draft_positions,
		       uint32_t end_token_id, uint32_t noise_token_id,
		       struct step_round *round, struct step_block *draft_block,
		       struct step_row_record *draft_rows, unsigned char *views)
{
	const uint32_t row_count = block->row_count < record_row_count ?
					   block->row_count :
					   record_row_count;
	uint32_t anchor_row = NO_ROW;
	for (uint32_t row = 0; row < row_count && anchor_row == NO_ROW; row++) {
		anchor_row = (rows[row].flags & STEP_ROW_DRAFTS) != 0 ? row :
									NO_ROW;
	}
	if (anchor_row == NO_ROW) {
		const struct step_block empty = { block->sequence, 0, 0 };
		round->sequence = block->sequence;
		round->session_slot = STEP_NO_SESSION;
		round->emitted_count = 0;
		round->anchor_position = 0;
		round->anchor_token = 0;
		round->proposal_count = 0;
		round->reserved_zero = 0;
		*draft_block = empty;
		for (uint32_t view = 0; view < draft_positions; view++) {
			memcpy(views + (size_t)view * STEP_DRAFT_VIEW_BYTES,
			       &empty, sizeof(empty));
		}
		return;
	}
	const struct step_row_record anchor = rows[anchor_row];
	uint32_t emitted = 1; /* y_1 always; one more for each leading match */
	for (uint32_t index = 1;; index++) {
		const uint32_t drawn_row =
			row_at(rows, row_count, anchor.session_slot,
			       anchor.position + (int32_t)index - 1);
		const uint32_t proposal_row =
			row_at(rows, row_count, anchor.session_slot,
			       anchor.position + (int32_t)index);
		if (drawn_row == NO_ROW || proposal_row == NO_ROW ||
		    (rows[proposal_row].flags & STEP_ROW_PROPOSED) == 0 ||
		    rows[proposal_row].token_id !=
			    results[drawn_row].token_id ||
		    results[drawn_row].token_id == (int32_t)end_token_id) {
			break;
		}
		emitted++;
	}
	const int32_t anchor_token =
		results[row_at(rows, row_count, anchor.session_slot,
			       anchor.position + (int32_t)emitted - 1)]
			.token_id;
	const int32_t anchor_position = anchor.position + (int32_t)emitted;
	round->sequence = block->sequence;
	round->session_slot = anchor.session_slot;
	round->emitted_count = emitted;
	round->anchor_position = anchor_position;
	round->anchor_token = anchor_token;
	round->proposal_count = draft_positions;
	round->reserved_zero = 0;
	*draft_block =
		(struct step_block){ block->sequence, draft_positions, 0 };
	for (uint32_t position = 0; position < draft_positions; position++) {
		const struct step_block view = { block->sequence, 1, 0 };
		draft_rows[position] = (struct step_row_record){
			anchor_position + (int32_t)position,
			position == 0 ? anchor_token : (int32_t)noise_token_id,
			anchor.session_slot, STEP_ROW_LOGITS
		};
		memcpy(views + (size_t)position * STEP_DRAFT_VIEW_BYTES, &view,
		       sizeof(view));
		memcpy(views + (size_t)position * STEP_DRAFT_VIEW_BYTES +
			       STEP_BLOCK_BYTES_VALUE,
		       &draft_rows[position], sizeof(draft_rows[position]));
	}
}

void test_draft_markov_bias(const uint16_t *embedding, const uint16_t *head,
			    uint32_t vocabulary_size, uint32_t token,
			    float *logits_row, uint16_t *markov_row)
{
	const uint16_t *const embedded =
		embedding + (size_t)token * MODEL_MARKOV_RANK;
	memcpy(markov_row, embedded, DRAFT_MARKOV_ROW_BYTES);
	for (uint32_t entry = 0; entry < vocabulary_size; entry++) {
		const uint16_t *const head_row =
			head + (size_t)entry * MODEL_MARKOV_RANK;
		float lanes[LANES];
		for (uint32_t lane = 0; lane < LANES; lane++) {
			float partial = 0.0f;
			for (uint32_t value = 0; value < MARKOV_VALUES_PER_LANE;
			     value++) {
				const uint32_t rank =
					lane * MARKOV_VALUES_PER_LANE + value;
				partial = fmaf(value_of(head_row[rank]),
					       value_of(embedded[rank]),
					       partial);
			}
			lanes[lane] = partial;
		}
		logits_row[entry] = logits_row[entry] + warp_sum(lanes);
	}
}

double test_draft_markov_bias_exact(const uint16_t *embedding,
				    const uint16_t *head, uint32_t token,
				    uint32_t entry)
{
	double sum = 0.0;
	for (uint32_t rank = 0; rank < MODEL_MARKOV_RANK; rank++) {
		sum += (double)value_of(
			       head[(size_t)entry * MODEL_MARKOV_RANK + rank]) *
		       (double)value_of(
			       embedding[(size_t)token * MODEL_MARKOV_RANK +
					 rank]);
	}
	return sum;
}

/* Input i of the confidence: the collapse's values, then the Markov row's. */
static float confidence_input(const uint16_t *collapse_row,
			      const uint16_t *markov_row, uint32_t input)
{
	return value_of(input < MODEL_HIDDEN ?
				collapse_row[input] :
				markov_row[input - MODEL_HIDDEN]);
}

float test_draft_confidence(const uint16_t *collapse_row,
			    const uint16_t *markov_row, const uint16_t *weights)
{
	float warp_sums[CONFIDENCE_WARPS];
	for (uint32_t warp = 0; warp < CONFIDENCE_WARPS; warp++) {
		float lanes[LANES];
		for (uint32_t lane = 0; lane < LANES; lane++) {
			float chain = 0.0f;
			for (uint32_t input = warp * LANES + lane;
			     input < MODEL_DRAFT_CONFIDENCE_INPUTS;
			     input += CONFIDENCE_THREADS) {
				chain = fmaf(value_of(weights[input]),
					     confidence_input(collapse_row,
							      markov_row,
							      input),
					     chain);
			}
			lanes[lane] = chain;
		}
		warp_sums[warp] = warp_sum(lanes);
	}
	float score = 0.0f;
	for (uint32_t warp = 0; warp < CONFIDENCE_WARPS; warp++) {
		score = score + warp_sums[warp];
	}
	return score;
}

double test_draft_confidence_exact(const uint16_t *collapse_row,
				   const uint16_t *markov_row,
				   const uint16_t *weights)
{
	double sum = 0.0;
	for (uint32_t input = 0; input < MODEL_DRAFT_CONFIDENCE_INPUTS;
	     input++) {
		sum += (double)value_of(weights[input]) *
		       (double)confidence_input(collapse_row, markov_row,
						input);
	}
	return sum;
}
