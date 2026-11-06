/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Shared immutable program boundaries for eager submission and startup graph preparation. */
#ifndef STEP_RUNS_H
#define STEP_RUNS_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "anemone/model_seam.h"
#include "anemone/step.h"
#include "anemone/step_records.h"

/* Whether a step is a front step (model_seam.h): it samples nothing, whatever its program's segments. Rows checked. */
static inline bool step_is_front(const struct step_row_record *rows,
				 uint32_t row_count)
{
	return row_count != 0 && (rows[0].flags & STEP_ROW_FRONT) != 0;
}

/* Whether a step samples nothing: a front step, or a step of a full program, whose chunks carry no row of logits in
 * this build (model_v41_program.c). Rows checked, program the step's. */
static inline bool step_is_headless(const struct execution_program *program,
				    const struct step_row_record *rows,
				    uint32_t row_count)
{
	return step_is_front(rows, row_count) || program->kind == PROGRAM_FULL;
}

/* Whether the admitted step runs its device drafter. */
static inline bool step_drafts(const struct step_row_record *rows,
			       uint32_t row_count)
{
	for (uint32_t row = 0; row < row_count; row++) {
		if ((rows[row].flags & STEP_ROW_DRAFTS) != 0) {
			return true;
		}
	}
	return false;
}

/* Kind is the unchanged round-record ABI discriminator: BATCH owns one record per session slot. */
static inline uint64_t step_round_bytes(const struct execution_program *program)
{
	return (uint64_t)STEP_ROUND_BYTES *
	       (program->kind == PROGRAM_BATCH ?
			program->limits.session_capacity :
			1u);
}

/* BATCH samplers write one80-byte area per prepared8-row slab, including empty slab headers.
 * Ordinary THIN retains one flat result area. The actual prepared program selects the slab count. */
static inline uint64_t
step_result_bytes(const struct execution_program *program, uint32_t row_count)
{
	return program->kind == PROGRAM_BATCH ?
		       (uint64_t)((program->row_count + STEP_BATCH_SLAB_ROWS -
				   1u) /
				  STEP_BATCH_SLAB_ROWS) *
			       STEP_BATCH_SLAB_RESULT_BYTES :
		       STEP_RESULT_BYTES +
			       (uint64_t)row_count * STEP_ROW_RESULT_BYTES;
}

/* The admitted step's one result download, including its round table only when it drafts. */
static inline uint64_t step_result_span(const struct execution_program *program,
					const struct step_row_record *rows,
					uint32_t row_count)
{
	const struct program_card_record *const card =
		&program->cards[program->result_card];
	return step_drafts(rows, row_count) ?
		       card->round_offset + step_round_bytes(program) -
			       card->result_offset :
		       step_result_bytes(program, row_count);
}

/* Front steps run the front; DRAFTS steps run every segment; the others stop before the draft tail. */
static inline uint32_t
step_segment_count(const struct execution_program *program,
		   const struct step_row_record *rows, uint32_t row_count)
{
	return step_is_front(rows, row_count) ? program->front_segment_count :
	       step_drafts(rows, row_count)   ? program->segment_count :
						program->draft_segment_first;
}

/* hand_off is the first not yet consumed. Metadata-only hand-offs also split runs: the driver still owns them, and
 * the front's end does (model_seam.h). The step's first run ends with its first layer: a short graph the card starts
 * while the host still launches the next one (a graph launch costs the host about 1 us a node, on the critical path
 * at the step's start). */
static inline uint32_t step_run_last(const struct execution_program *program,
				     uint32_t segment, uint32_t hand_off)
{
	const uint32_t card = program->segments[segment].card;
	uint32_t last = segment;
	while (last + 1u < program->segment_count &&
	       last + 1u != program->front_segment_count &&
	       last + 1u != program->draft_segment_first &&
	       program->segments[last + 1u].card == card &&
	       program->segments[last].pause == 0 &&
	       (segment != 0 ||
		program->segments[last].layer == SEGMENT_NO_LAYER) &&
	       (hand_off == program->hand_off_count ||
		program->hand_offs[hand_off].after_segment > last)) {
		last++;
	}
	return last;
}

/* Create validated consecutive launch ranges on this card, including every launch of the last segment. */
static inline uint32_t
step_run_launch_count(const struct execution_program *program, uint32_t segment,
		      uint32_t last)
{
	return program->segments[last].first_launch +
	       program->segments[last].launch_count -
	       program->segments[segment].first_launch;
}

/* A companion member is a separate local run; an ordinary run may not absorb a following member. The map was
 * validated/copied at startup and is shared by all program prefixes. NULL retains the original PP boundaries. */
static inline uint32_t
step_run_last_companions(const struct execution_program *program,
			 uint32_t segment, uint32_t hand_off,
			 const uint32_t *companions)
{
	if (companions == NULL)
		return step_run_last(program, segment, hand_off);
	if (companions[segment] != STEP_NO_COMPANION)
		return segment;
	const uint32_t last = step_run_last(program, segment, hand_off);
	for (uint32_t next = segment + 1u; next <= last; next++) {
		if (companions[next] != STEP_NO_COMPANION)
			return next - 1u;
	}
	return last;
}

#endif
