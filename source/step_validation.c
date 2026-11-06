/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Immutable program/layout admission at driver creation. No recurrent execution or new policy lives here. */
#include <stddef.h>
#include <string.h>

#include "anemone/step_records.h"
#include "step_validation.h"
#include "step_runs.h"

#define NOWHERE PROGRAM_NOWHERE
#define TAG_BYTES LINEAR_OFFSET_ALIGNMENT_BYTES
#define CARD_AREA_BYTES                                       \
	(offsetof(struct program_card_record, launch_count) - \
	 offsetof(struct program_card_record, arena_byte_count))

/* [offset, offset + byte_count) and a range of the same arena found inside it share no byte; a range of no bytes or
 * at NOWHERE shares none. */
static bool apart(uint64_t offset, uint64_t byte_count, uint64_t other,
		  uint64_t other_byte_count)
{
	return other == NOWHERE || other_byte_count == 0 ||
	       other >= offset + byte_count ||
	       offset >= other + other_byte_count;
}

/* Whether a pause's slots and jobs, found inside its card's arena, share no byte with each other or the slots with
 * what else the driver moves on that card: the step area, the n-gram rows, the session settings and the result area,
 * the hand-offs. */
static bool slots_are_apart(const struct execution_program *program,
			    const struct expert_pause *pause,
			    const struct engram_description *tables)
{
	const struct program_card_record *const area =
		&program->cards[pause->card];
	const bool samples = pause->card == program->result_card;
	const uint64_t rows = program->limits.row_capacity,
		       sessions = program->limits.session_capacity;
	const uint64_t at = pause->slots_offset,
		       bytes = (uint64_t)pause->slot_count *
			       pause->slot_byte_count;
	const uint64_t engram_bytes =
		tables == NULL ?
			0 :
			(uint64_t)__builtin_popcount(area->engram_tables) *
				rows * tables->rows_per_token *
				ENGRAM_ROW_BYTES;
	bool is_apart =
		apart(at, bytes, pause->jobs_offset,
		      (uint64_t)pause->jobs_per_row * program->row_count *
			      EXPERT_JOB_BYTES) &&
		apart(at, bytes, area->step_offset,
		      STEP_BLOCK_BYTES_VALUE + rows * STEP_ROW_BYTES_VALUE) &&
		apart(at, bytes, area->engram_rows_offset, engram_bytes) &&
		apart(at, bytes, samples ? area->session_offset : NOWHERE,
		      sessions * STEP_SESSION_BYTES) &&
		apart(at, bytes, samples ? area->result_offset : NOWHERE,
		      step_result_bytes(program,
					program->limits.row_capacity)) &&
		apart(at, bytes, samples ? area->round_offset : NOWHERE,
		      step_round_bytes(program));
	for (uint32_t index = 0; index < program->hand_off_count && is_apart;
	     index++) {
		const struct hand_off_record *const hand_off =
			&program->hand_offs[index];
		const uint64_t byte_count =
			hand_off->entry_ratio == 0 ?
				rows * hand_off->byte_count :
				sessions * hand_off->session_stride;
		is_apart = apart(at, bytes,
				 hand_off->from_card == pause->card ?
					 hand_off->source_offset :
					 NOWHERE,
				 byte_count) &&
			   apart(at, bytes,
				 hand_off->to_card == pause->card ?
					 hand_off->destination_offset :
					 NOWHERE,
				 byte_count);
	}
	return is_apart;
}

/* What breaks create's rules in a program's expert pauses (paused: its segments that end with one), measured against
 * the first program, or NULL. The pauses name the program's host experts in order, each once. */
static const char *pauses_problem(const struct execution_program *program,
				  const struct execution_program *first,
				  uint32_t paused,
				  const struct engram_description *tables)
{
	const struct expert_pause *const pauses = program->expert_pauses;
	uint64_t host_experts = 0; /* those the pauses before name */
	if (program->expert_pause_count != paused ||
	    program->expert_pause_count != first->expert_pause_count ||
	    program->host_expert_count != first->host_expert_count ||
	    (paused != 0 && pauses == NULL) ||
	    (program->host_expert_count != 0 &&
	     program->host_expert_offsets == NULL) ||
	    (paused != 0 && memcmp(pauses, first->expert_pauses,
				   paused * sizeof(*pauses)) != 0) ||
	    (program->host_expert_count != 0 &&
	     memcmp(program->host_expert_offsets, first->host_expert_offsets,
		    program->host_expert_count *
			    sizeof(*program->host_expert_offsets)) != 0)) {
		return "the expert pauses are not those of its paused segments, or not those of the first program";
	}
	for (uint32_t index = 0; index < paused; index++) {
		const struct expert_pause *const pause = &pauses[index],
						 *card_first = pause;
		for (uint32_t earlier = 0;
		     earlier < index && card_first == pause; earlier++) {
			card_first = pauses[earlier].card == pause->card ?
					     &pauses[earlier] :
					     card_first;
		}
		const uint64_t arena =
			pause->card < program->card_count ?
				program->cards[pause->card].arena_byte_count :
				0;
		const uint64_t job_count =
			(uint64_t)pause->jobs_per_row * program->row_count;
		bool is_valid =
			pause->after_segment < program->segment_count &&
			(index == 0 ||
			 pause->after_segment >
				 pauses[index - 1u].after_segment) &&
			program->segments[pause->after_segment].pause ==
				SEGMENT_PAUSE_EXPERT_UPLOADS &&
			program->segments[pause->after_segment].card ==
				pause->card &&
			job_count != 0 && pause->host_expert_count != 0 &&
			pause->slot_count >= job_count &&
			pause->expert_byte_count != 0 &&
			pause->expert_byte_count <= pause->slot_byte_count &&
			(pause->jobs_offset | pause->slots_offset |
			 pause->slot_byte_count) %
					TAG_BYTES ==
				0 &&
			pause->first_host_expert == host_experts &&
			step_range_fits(pause->jobs_offset, job_count,
					EXPERT_JOB_BYTES, arena) &&
			step_range_fits(pause->slots_offset, pause->slot_count,
					pause->slot_byte_count, arena) &&
			pause->slots_offset == card_first->slots_offset &&
			pause->slot_count == card_first->slot_count &&
			pause->slot_byte_count == card_first->slot_byte_count;
		for (uint32_t matrix = 0; matrix < EXPERT_PAUSE_MATRICES;
		     matrix++) {
			is_valid = is_valid &&
				   pause->matrix_offsets[matrix] <
					   pause->expert_byte_count &&
				   pause->matrix_offsets[matrix] % TAG_BYTES ==
					   0;
		}
		if (!is_valid || !slots_are_apart(program, pause, tables)) {
			return "an expert pause names no paused segment of its card, its jobs, slots or "
			       "experts do not fit, or "
			       "its slots share bytes with what the driver moves on the card";
		}
		host_experts += pause->host_expert_count;
	}
	return host_experts == program->host_expert_count ?
		       NULL :
		       "the expert pauses do not name every host expert";
}

/* The hand-offs of a program behind its first segment_count segments, which come first: of its front (model_seam.h)
 * with front_segment_count; of all but its head with one segment fewer than it has, where without_head leaves out
 * those of no entry ratio behind the last of them: rows of the step that only the head reads. */
static uint32_t hand_offs_before(const struct execution_program *program,
				 uint32_t segment_count, bool without_head)
{
	uint32_t count = 0;
	while (count < program->hand_off_count &&
	       program->hand_offs[count].after_segment < segment_count) {
		count++;
	}
	while (without_head && count != 0 &&
	       program->hand_offs[count - 1u].after_segment + 1u ==
		       segment_count &&
	       program->hand_offs[count - 1u].entry_ratio == 0) {
		count--;
	}
	return count;
}

/* True where the thin first program's segments from `from` up to its draft tail are its head's, at least one, none
 * of a layer: one segment, or one a card of a head cut over several cards (tp4_v3_claude). */
static bool head_segments_only(const struct execution_program *first,
			       uint32_t from)
{
	for (uint32_t index = from; index < first->draft_segment_first;
	     index++) {
		if (first->segments[index].layer != SEGMENT_NO_LAYER)
			return false;
	}
	return from < first->draft_segment_first;
}

/* What breaks create's rules in a program's views (model_seam.h), or NULL. A thin program has none. Each card of a
 * wide program that runs launches has the same views as every other: slabs one after another from row 0 that hold its
 * rows, each of 1 to 8 rows and beginning on one, their blocks one behind the other inside the card's arena; a wide
 * or full program may hold its rows again behind them, in views of up to LINEAR_WIDE_MOST_ROWS rows (a chunk's
 * halves, model_v41_tp2_stage.h's split_overlap). A full program's cards are a wide one's. */
static const char *views_problem(const struct execution_program *program)
{
	const bool is_wide = program->kind == PROGRAM_WIDE ||
			     program->kind == PROGRAM_FULL ||
			     (program->kind == PROGRAM_BATCH &&
			      program->row_count > LINEAR_MAXIMUM_ROWS);
	const struct program_card_record *same =
		NULL; /* the first with views */
	for (uint32_t index = 0; index < program->card_count; index++) {
		const struct program_card_record *const card =
			&program->cards[index];
		uint64_t block = card->view_count != 0 && card->views != NULL ?
					 card->views[0].block_offset :
					 0;
		uint32_t first_row = 0, tiling = 0; /* 0: the slabs; 1: the rows again (halves) */
		bool is_valid =
			card->view_count == 0 ?
				!is_wide || card->launch_count == 0 :
				is_wide && card->views != NULL &&
					block % LINEAR_OFFSET_ALIGNMENT_BYTES ==
						0 &&
					(same == NULL ||
					 card->view_count == same->view_count);
		for (uint32_t view = 0;
		     view < card->view_count && is_valid; view++) {
			const struct step_view *const own = &card->views[view];
			const uint64_t bytes =
				STEP_BLOCK_BYTES_VALUE +
				(uint64_t)own->row_limit * STEP_ROW_BYTES_VALUE;
			is_valid =
				(program->kind != PROGRAM_BATCH ||
				 (own->first_row ==
					  view * STEP_BATCH_SLAB_ROWS &&
				  (view + 1u == card->view_count ||
				   own->row_limit == STEP_BATCH_SLAB_ROWS) &&
				  card->view_count ==
					  (program->row_count +
					   STEP_BATCH_SLAB_ROWS - 1u) /
						  STEP_BATCH_SLAB_ROWS)) &&
				own->block_offset == block &&
				own->first_row == first_row &&
				own->first_row < program->row_count &&
				tiling < 2u &&
				(tiling == 0 ||
				 program->kind != PROGRAM_BATCH) &&
				own->row_limit - 1u <
					(tiling == 0 ? LINEAR_MAXIMUM_ROWS :
						       LINEAR_WIDE_MOST_ROWS) &&
				step_range_fits(own->block_offset, 1, bytes,
						card->arena_byte_count) &&
				(same == NULL || (same->views[view].first_row ==
							  own->first_row &&
						  same->views[view].row_limit ==
							  own->row_limit));
			block += bytes;
			first_row += own->row_limit;
			if (first_row >= program->row_count) {
				first_row = 0;
				tiling++;
			}
		}
		if (!is_valid ||
		    (card->view_count != 0 && (tiling == 0 || first_row != 0))) {
			return "a thin program gives a card views, or a wide program's card that runs launches has "
			       "none, or other views than another's, or views that are no slabs of its rows one "
			       "behind the other in its arena";
		}
		same = card->view_count != 0 && same == NULL ? card : same;
	}
	return NULL;
}

/* What breaks create's rules in a program, measured against the first, or NULL. A thin program has the first's
 * segments, in card, stage, layer and pause, and front (the pipeline runs every program by the first's visits), and
 * its hand-offs, byte for byte. A wide program is the front of the first, a thin one (model_seam.h): its segments, in
 * card, stage and layer, and its hand-offs, byte for byte. A full program is all of the first but its last segment,
 * the head (this build's full programs have none): those segments, the same front, and the hand-offs behind them,
 * byte for byte, but those of rows behind its last segment, which only the head reads. */
const char *step_program_problem(const struct execution_program *program,
				 const struct execution_program *first,
				 struct device_card *const *cards,
				 const struct engram_record *engram,
				 bool *is_unsupported_out)
{
	const struct engram_description *const tables =
		engram != NULL ? engram_description_of(engram) : NULL;
	const struct model_limits *const limits =
		program != NULL ? &program->limits : NULL;
	uint32_t next_launch[BACKEND_MAXIMUM_CARDS] = { 0 };
	uint32_t paused = 0;
	if (limits == NULL || program->cards == NULL ||
	    program->segments == NULL || program->segment_count == 0 ||
	    program->draft_segment_first > program->segment_count ||
	    program->front_segment_count > program->draft_segment_first ||
	    (program->hand_offs == NULL && program->hand_off_count != 0) ||
	    (program->session_ranges == NULL &&
	     program->session_range_count != 0) ||
	    (program->checkpoints == NULL && program->checkpoint_count != 0) ||
	    (program->leases == NULL && program->lease_count != 0) ||
	    program->card_count - 1u >= BACKEND_MAXIMUM_CARDS ||
	    program->result_card >= program->card_count ||
	    program->row_count - 1u >= limits->row_capacity ||
	    limits->session_capacity == 0 || limits->context_token_limit == 0 ||
	    limits->draft_positions > STEP_DRAFT_MOST) {
		return "a program or a part of it is missing, or a count or limit is out of range";
	}
	const bool is_wide = program->kind == PROGRAM_WIDE,
		   is_full = program->kind == PROGRAM_FULL;
	const bool is_chunked = is_wide || is_full;
	const bool is_batch = program->kind == PROGRAM_BATCH;
	const bool has_draft = program->draft_segment_first <
			       program->segment_count;
	if ((program->kind != PROGRAM_THIN && !is_batch && !is_chunked) ||
	    (!is_chunked && is_batch != (first->kind == PROGRAM_BATCH)) ||
	    (is_batch &&
	     (program->row_count > STEP_BATCH_MAXIMUM_ROWS ||
	      limits->session_capacity > STEP_BATCH_MAXIMUM_SESSIONS)) ||
	    (has_draft && (is_chunked || limits->draft_positions == 0)) ||
	    (!is_chunked && !has_draft && limits->draft_positions != 0) ||
	    (!is_chunked && !is_batch &&
	     program->row_count > LINEAR_MAXIMUM_ROWS) ||
	    (is_chunked && first->kind != PROGRAM_THIN &&
	     first->kind != PROGRAM_BATCH) ||
	    (is_wide &&
	     (program->segment_count != first->front_segment_count ||
	      program->front_segment_count != program->segment_count)) ||
	    (is_full &&
	     (program->front_segment_count != first->front_segment_count ||
	      !head_segments_only(first, program->segment_count)))) {
		*is_unsupported_out = program->kind == PROGRAM_CONE;
		return "a program of no kind or of a kind this driver does not run yet, a thin one of too many "
		       "rows, a wide one that is not the front of a thin first program, or a full one that is "
		       "not all of it but its head";
	}
	const uint32_t hand_off_count =
		is_wide ?
			hand_offs_before(first, first->front_segment_count,
					 false) :
		is_full ?
			hand_offs_before(first, program->segment_count, true) :
			first->hand_off_count;
	bool is_same =
		is_chunked ||
		(program->segment_count == first->segment_count &&
		 program->front_segment_count == first->front_segment_count &&
		 program->draft_segment_first == first->draft_segment_first);
	for (uint32_t index = 0; index < program->segment_count && is_same;
	     index++) {
		const struct segment_record
			*const own = &program->segments[index],
			       *const thin = &first->segments[index];
		is_same = own->card == thin->card &&
			  own->stage == thin->stage &&
			  own->layer == thin->layer &&
			  own->pause == (is_chunked ? 0u : thin->pause);
	}
	if (!is_same) {
		return "a program's segments are not those of the thin first program, as far as they reach";
	}
	if (program->card_count != first->card_count ||
	    program->result_card != first->result_card ||
	    memcmp(limits, &first->limits, sizeof(*limits)) != 0 ||
	    program->hand_off_count != hand_off_count ||
	    (program->hand_off_count != 0 &&
	     memcmp(program->hand_offs, first->hand_offs,
		    program->hand_off_count * sizeof(*program->hand_offs)) !=
		     0) ||
	    program->session_range_count != first->session_range_count ||
	    (program->session_range_count != 0 &&
	     memcmp(program->session_ranges, first->session_ranges,
		    program->session_range_count *
			    sizeof(*program->session_ranges)) != 0) ||
	    program->checkpoint_count != first->checkpoint_count ||
	    (program->checkpoint_count != 0 &&
	     memcmp(program->checkpoints, first->checkpoints,
		    program->checkpoint_count * sizeof(*program->checkpoints)) != 0)) {
		return "the programs are not of one plan";
	}
	for (uint32_t index = 0; index < program->lease_count; index++) {
		const struct visit_lease *const lease = &program->leases[index];
		if (lease->first_segment >= lease->last_segment ||
		    lease->last_segment >= program->segment_count ||
		    program->segments[lease->first_segment].card !=
			    lease->card ||
		    program->segments[lease->last_segment].card !=
			    lease->card) {
			return "a lease names no two segments of its card in order";
		}
	}
	for (uint32_t index = 0; index < program->segment_count; index++) {
		const struct segment_record *const segment =
			&program->segments[index];
		const bool is_valid =
			segment->card < program->card_count &&
			segment->first_launch == next_launch[segment->card] &&
			segment->launch_count - 1u <
				program->cards[segment->card].launch_count -
					segment->first_launch;
		if (!is_valid ||
		    (index >= program->draft_segment_first &&
		     (segment->layer != SEGMENT_NO_LAYER ||
		      segment->pause != 0)) ||
		    (segment->pause != 0 &&
		     segment->pause != SEGMENT_PAUSE_EXPERT_UPLOADS)) {
			*is_unsupported_out =
				is_valid &&
				segment->pause == SEGMENT_PAUSE_PAIR_OUTPUTS;
			return "a segment names no card, does not follow its card's one before, or pauses for "
			       "pair outputs";
		}
		next_launch[segment->card] += segment->launch_count;
		paused += segment->pause != 0 ? 1u : 0u;
	}
	for (uint32_t index = 0; index < program->card_count; index++) {
		const struct program_card_record *const card =
			&program->cards[index];
		const uint64_t table_bytes =
			tables == NULL ? 0 :
					 (uint64_t)__builtin_popcount(
						 card->engram_tables) *
						 tables->rows_per_token *
						 ENGRAM_ROW_BYTES;
		if (next_launch[index] != card->launch_count ||
		    ((!has_draft || index != program->result_card) &&
		     card->round_offset != NOWHERE) ||
		    (!is_chunked &&
		     card->round_offset != first->cards[index].round_offset) ||
		    (card->launches == NULL && card->launch_count != 0) ||
		    (card->launch_count != 0 && card->step_offset == NOWHERE) ||
		    (cards[index] == NULL &&
		     ((card->launch_count | card->engram_tables) != 0 ||
		      card->step_offset != NOWHERE ||
		      index == program->result_card)) ||
		    memcmp(&card->arena_byte_count,
			   &first->cards[index].arena_byte_count,
			   CARD_AREA_BYTES) != 0 ||
		    (card->step_offset != NOWHERE &&
		     !step_range_fits(card->step_offset, 1,
				      STEP_BLOCK_BYTES_VALUE +
					      (uint64_t)limits->row_capacity *
						      STEP_ROW_BYTES_VALUE,
				      card->arena_byte_count)) ||
		    (card->engram_tables != 0 &&
		     (tables == NULL ||
		      (card->engram_tables >> tables->table_count) != 0 ||
		      !step_range_fits(card->engram_rows_offset,
				       limits->row_capacity, table_bytes,
				       card->arena_byte_count)))) {
			return "a card's launches do not all run, a card that works is missing, or its areas are not "
			       "all there";
		}
	}
	const struct program_card_record *const result =
		&program->cards[program->result_card];
	if (!step_range_fits(result->result_offset, 1,
			     step_result_bytes(program, limits->row_capacity),
			     result->arena_byte_count) ||
	    !step_range_fits(result->session_offset, limits->session_capacity,
			     STEP_SESSION_BYTES, result->arena_byte_count)) {
		return "the result area or the session settings lie outside the arena of the card that samples";
	}
	if (has_draft) {
		const uint64_t round = result->round_offset;
		const uint64_t round_bytes = step_round_bytes(program);
		const uint64_t span_most = is_batch ?
						   STEP_BATCH_RESULT_SPAN_MOST :
						   STEP_RESULT_SPAN_MOST;
		const uint64_t thin_results =
			is_batch ?
				step_result_bytes(program, program->row_count) :
				STEP_RESULT_BYTES +
					(uint64_t)LINEAR_MAXIMUM_ROWS *
						STEP_ROW_RESULT_BYTES;
		if (round % TAG_BYTES != 0 || round < result->result_offset ||
		    round - result->result_offset < thin_results ||
		    round - result->result_offset > span_most - round_bytes ||
		    !step_range_fits(round, 1, round_bytes,
				     result->arena_byte_count) ||
		    !apart(round, round_bytes, result->step_offset,
			   STEP_BLOCK_BYTES_VALUE +
				   (uint64_t)limits->row_capacity *
					   STEP_ROW_BYTES_VALUE) ||
		    !apart(round, round_bytes, result->session_offset,
			   (uint64_t)limits->session_capacity *
				   STEP_SESSION_BYTES)) {
			return "the draft round overlaps step areas or settings, lies outside the result span, "
			       "or is unaligned";
		}
	}
	for (uint32_t index = 0; index < program->hand_off_count; index++) {
		const struct hand_off_record *const hand_off =
			&program->hand_offs[index];
		const uint32_t ratio = hand_off->entry_ratio;
		const uint64_t count = ratio == 0 ? limits->row_capacity :
						    limits->session_capacity;
		const uint64_t unit = ratio == 0 ? hand_off->byte_count :
						   hand_off->session_stride;
		if (hand_off->after_segment >= program->draft_segment_first ||
		    (index != 0 &&
		     hand_off->after_segment <
			     program->hand_offs[index - 1u].after_segment) ||
		    hand_off->from_card !=
			    program->segments[hand_off->after_segment].card ||
		    hand_off->to_card >= program->card_count ||
		    hand_off->to_card == hand_off->from_card ||
		    cards[hand_off->to_card] == NULL ||
		    hand_off->byte_count == 0 || hand_off->view != 0 ||
		    (ratio == 0 ?
			     hand_off->session_stride != 0 :
			     (uint64_t)(limits->context_token_limit / ratio) *
					     hand_off->byte_count >
				     unit) ||
		    !step_range_fits(hand_off->source_offset, count, unit,
				     program->cards[hand_off->from_card]
					     .arena_byte_count) ||
		    !step_range_fits(hand_off->destination_offset, count, unit,
				     program->cards[hand_off->to_card]
					     .arena_byte_count)) {
			return "a hand-off names no segment, card or view, "
			       "leaves no card of its segment, or lies outside "
			       "an "
			       "arena";
		}
	}
	for (uint32_t index = 0; index < program->session_range_count; index++) {
		const struct session_range *const range =
			&program->session_ranges[index];
		if (range->card >= program->card_count ||
		    cards[range->card] == NULL || range->unit_bytes == 0 ||
		    range->unit_count == 0 ||
		    (index != 0 &&
		     range->identity <
			     program->session_ranges[index - 1u].identity) ||
		    range->unit_count >
			    range->session_stride / range->unit_bytes ||
		    !step_range_fits(
			    range->offset, limits->session_capacity,
			    range->session_stride,
			    program->cards[range->card].arena_byte_count)) {
			return "a session range names no card, holds no unit, is out of order or lies outside an arena";
		}
	}
	for (uint32_t index = 0; index < program->checkpoint_count; index++) {
		const struct window_checkpoint *const range =
			&program->checkpoints[index];
		if (range->card >= program->card_count ||
		    cards[range->card] == NULL || range->row_bytes == 0 ||
		    range->window_rows == 0 ||
		    range->window_rows >= range->ring_rows) {
			return "a window checkpoint names no card, no row bytes or an invalid window";
		}
		const uint64_t ring_bytes =
			(uint64_t)range->ring_rows * range->row_bytes;
		const uint64_t saved_bytes =
			(uint64_t)range->window_rows * range->row_bytes;
		const uint64_t arena =
			program->cards[range->card].arena_byte_count;
		if (ring_bytes > range->ring_session_stride ||
		    !step_range_fits(range->ring_offset,
				     limits->session_capacity,
				     range->ring_session_stride, arena) ||
		    !step_range_fits(range->checkpoint_offset,
				     limits->session_capacity, saved_bytes,
				     arena)) {
			return "a window checkpoint lies outside its arena";
		}
		const uint64_t total = saved_bytes * limits->session_capacity;
		const struct program_card_record *const area =
			&program->cards[range->card];
		if (!apart(range->checkpoint_offset, total, area->step_offset,
			   STEP_BLOCK_BYTES_VALUE +
				   (uint64_t)limits->row_capacity *
					   STEP_ROW_BYTES_VALUE) ||
		    !apart(range->checkpoint_offset, total,
			   area->session_offset,
			   (uint64_t)limits->session_capacity *
				   STEP_SESSION_BYTES) ||
		    !apart(range->checkpoint_offset, total, area->result_offset,
			   STEP_RESULT_BYTES + (uint64_t)limits->row_capacity *
						       STEP_ROW_RESULT_BYTES)) {
			return "a window checkpoint overlaps a step, settings or result area";
		}
		if (!apart(range->checkpoint_offset, total, range->ring_offset,
			   range->ring_session_stride *
				   limits->session_capacity)) {
			return "a window checkpoint overlaps its ring";
		}
		for (uint32_t other = 0; other < program->session_range_count;
		     other++) {
			const struct session_range *const session =
				&program->session_ranges[other];
			if (session->card == range->card &&
			    !apart(range->checkpoint_offset, total,
				   session->offset,
				   session->session_stride *
					   limits->session_capacity)) {
				return "a window checkpoint overlaps session state";
			}
		}
		for (uint32_t other = 0; other < index; other++) {
			const struct window_checkpoint *const previous =
				&program->checkpoints[other];
			if (previous->card == range->card &&
			    (!apart(range->checkpoint_offset, total,
				    previous->checkpoint_offset,
				    (uint64_t)previous->window_rows *
					    previous->row_bytes *
					    limits->session_capacity) ||
			     !apart(range->checkpoint_offset, total,
				    previous->ring_offset,
				    previous->ring_session_stride *
					    limits->session_capacity) ||
			     !apart(previous->checkpoint_offset,
				    (uint64_t)previous->window_rows *
					    previous->row_bytes *
					    limits->session_capacity,
				    range->ring_offset,
				    range->ring_session_stride *
					    limits->session_capacity))) {
				return "window checkpoints overlap each other or another ring";
			}
		}
	}
	const char *const views = views_problem(program);
	return views != NULL ? views :
			       pauses_problem(program, first, paused, tables);
}
