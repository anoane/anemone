/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step.c: the step driver of step.h: eager by default, with optional prepared graph replay of each run.
 * Every copy uses one block of page-locked staging memory, which holds the step area, the session
 * settings, the result area, each card's n-gram rows and each hand-off's bytes at their largest, apart. A step ends
 * when waits have seen every card's copies and launches finish, so staging may be reused. A wait
 * on a card's last ticket covers its earlier ones, and comes before a copy would pass the backend's limit. The
 * hand-offs after a segment leave its card: downloads, one wait, uploads; with an entry ratio, the rows of a session
 * that stand together emit entries that follow one another: one copy. Tokens are held to the n-gram vocabulary. An
 * expert pause is step_experts.c's.
 */
#include "anemone/step.h"

#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/step_images.h"
#include "anemone/image.h"
#include "anemone/hooks.h"
#include "step_driver.h"
#include "step_runs.h"
#include "step_validation.h"

#define OK STATUS_OK
#define THEN(call)        \
	(status != OK ?   \
		 status : \
		 (call)) /* the next call while every one before passed */
#define NOWHERE PROGRAM_NOWHERE
/* per session behind the histories: its token count, next row and mirror range (from, to), a mirror sync */
#define SESSION_COUNTERS (4u + STEP_SYNC_FIELDS)
#define NO_ROW_YET \
	UINT32_MAX /* the next position of a session no row of the step continued */

/* Begins one copy between the card's arena and staging memory: a download when is_download. */
result_status step_copy_begin(struct step_driver *driver, uint32_t card,
			      uint64_t arena, unsigned char *staging,
			      uint64_t byte_count, bool is_download)
{
	result_status status =
		driver->unconfirmed[card] < BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT ?
			OK :
			card_wait(driver, card);
	status = THEN(is_download ?
			      driver->backend->download_begin(
				      driver->cards[card], arena, staging,
				      byte_count, &driver->last_ticket[card]) :
			      driver->backend->upload_begin(
				      driver->cards[card], arena, staging,
				      byte_count, &driver->last_ticket[card]));
	driver->unconfirmed[card] += status == OK ? 1u : 0u;
	/* A successful ticket covers earlier launches. A failed begin may have queued a copy
     * before its ticket/event failed: preserve ownership until completion or card close. */
	driver->uncovered_work[card] = status != OK;
	return status;
}

/* The copies of one hand-off for the step's rows, all downloads or all uploads, through the same bytes of block. */
result_status step_hand_off_copies(struct step_driver *driver,
				   unsigned char *block, uint32_t index,
				   const struct step_row_record *rows,
				   uint32_t row_count, bool is_download)
{
	const struct hand_off_record *const hand_off =
		&driver->programs[0]->hand_offs[index];
	const uint32_t card = is_download ? hand_off->from_card :
					    hand_off->to_card;
	const uint64_t offset = is_download ? hand_off->source_offset :
					      hand_off->destination_offset;
	const uint32_t ratio = hand_off->entry_ratio;
	uint64_t staging = driver->hand_off_staging[index];
	result_status status = OK;
	if (ratio == 0) {
		return step_copy_begin(driver, card, offset, block + staging,
				       (uint64_t)row_count *
					       hand_off->byte_count,
				       is_download);
	}
	for (uint32_t first = 0, last = 0; first < row_count && status == OK;
	     first = ++last) {
		while (last + 1u < row_count &&
		       rows[last + 1u].session_slot ==
			       rows[first].session_slot) {
			last++;
		}
		/* positions p0 .. p1 emit the entries from p0 / ratio up to (p1 + 1) / ratio */
		const uint64_t first_entry =
			(uint64_t)rows[first].position / ratio;
		const uint64_t entry_count =
			((uint64_t)rows[last].position + 1u) / ratio -
			first_entry;
		if (entry_count != 0) {
			status = step_copy_begin(
				driver, card,
				offset +
					rows[first].session_slot *
						hand_off->session_stride +
					first_entry * hand_off->byte_count,
				block + staging,
				entry_count * hand_off->byte_count,
				is_download);
			staging += entry_count * hand_off->byte_count;
		}
	}
	return status;
}

static uint32_t batch_speculative_wrong_row(const struct step_driver *driver,
					    const struct step_row_record *rows,
					    uint32_t row_count)
{
	const struct execution_program *const first = driver->programs[0];
	if (row_count > STEP_BATCH_MAXIMUM_ROWS)
		return 0;
	for (uint32_t session = 0; session < first->limits.session_capacity;
	     session++) {
		uint32_t anchor = row_count, drafts = row_count, proposed = 0;
		for (uint32_t row = 0; row < row_count; row++) {
			if (rows[row].session_slot != session)
				continue;
			const uint32_t flags = rows[row].flags;
			if ((flags & (STEP_ROW_PROPOSED | STEP_ROW_DRAFTS)) !=
				    0 &&
			    ((flags & STEP_ROW_LOGITS) == 0 ||
			     (flags & STEP_ROW_FRONT) != 0))
				return row;
			if ((flags & STEP_ROW_DRAFTS) != 0) {
				if (drafts != row_count ||
				    first->draft_segment_first ==
					    first->segment_count ||
				    first->limits.draft_positions == 0)
					return row;
				drafts = row;
			}
			if ((flags & STEP_ROW_PROPOSED) != 0) {
				if (anchor == row_count ||
				    (rows[anchor].flags & STEP_ROW_LOGITS) ==
					    0 ||
				    proposed >= STEP_DRAFT_MOST)
					return row;
				proposed++;
			} else {
				if (proposed != 0)
					return row;
				anchor = row;
			}
		}
		if (drafts != row_count &&
		    (drafts != anchor ||
		     proposed > first->limits.draft_positions ||
		     (uint64_t)(uint32_t)rows[anchor].position + proposed + 1u +
				     first->limits.draft_positions >
			     first->limits.context_token_limit))
			return drafts;
	}
	return row_count;
}

/* The speculative row rules, after ordinary positions, tokens and flags were checked. The anchor can follow prompt
 * rows of its session, and other sessions may be interleaved, so find it in session order. */
static uint32_t speculative_wrong_row(const struct step_driver *driver,
				      const struct step_row_record *rows,
				      uint32_t row_count)
{
	const struct execution_program *const first = driver->programs[0];
	if (step_rounds_is_batch(driver))
		return batch_speculative_wrong_row(driver, rows, row_count);
	uint32_t session = STEP_NO_SESSION, anchor = row_count,
		 drafts = row_count, proposed = 0;
	for (uint32_t row = 0; row < row_count; row++) {
		const uint32_t flags = rows[row].flags;
		if ((flags & (STEP_ROW_PROPOSED | STEP_ROW_DRAFTS)) == 0) {
			continue;
		}
		if (row_count > LINEAR_MAXIMUM_ROWS ||
		    (flags & STEP_ROW_LOGITS) == 0 ||
		    (flags & STEP_ROW_FRONT) != 0 ||
		    (session != STEP_NO_SESSION &&
		     session != rows[row].session_slot)) {
			return row;
		}
		session = rows[row].session_slot;
		if ((flags & STEP_ROW_DRAFTS) != 0) {
			if (drafts != row_count ||
			    first->draft_segment_first ==
				    first->segment_count ||
			    first->limits.draft_positions == 0) {
				return row;
			}
			drafts = row;
		}
	}
	for (uint32_t row = 0; row < row_count && session != STEP_NO_SESSION;
	     row++) {
		if (rows[row].session_slot != session) {
			continue;
		}
		if ((rows[row].flags & STEP_ROW_PROPOSED) != 0) {
			if (anchor == row_count ||
			    (rows[anchor].flags & STEP_ROW_LOGITS) == 0) {
				return row;
			}
			proposed++;
		} else {
			if (proposed != 0) {
				return row;
			}
			anchor = row;
		}
	}
	if (drafts != row_count &&
	    (drafts != anchor || (uint64_t)(uint32_t)rows[anchor].position +
						 proposed + 1u +
						 first->limits.draft_positions >
					 first->limits.context_token_limit)) {
		return drafts;
	}
	return row_count;
}

/* The first row that breaks step.h's rules, or row_count; 0 for a count above every program's. Changes nothing. */
static uint32_t first_wrong_row(struct step_driver *driver,
				const struct step_row_record *rows,
				uint32_t row_count)
{
	const struct model_limits *const limits = &driver->programs[0]->limits;
	/* a front step's rows: all front, none with logits, a program with a front */
	const uint32_t front = row_count != 0 ? rows[0].flags & STEP_ROW_FRONT :
						0;
	const bool has_front = driver->programs[0]->front_segment_count != 0;
	const uint32_t padded =
		row_count != 0 ? rows[0].flags & STEP_ROW_PADDED : 0;
	uint32_t row = 0, speculative = 0;
	for (uint32_t slot = 0; slot < limits->session_capacity; slot++) {
		driver->next_positions[slot] = NO_ROW_YET;
	}
	for (; row < row_count && row_count <= driver->largest_row_count;
	     row++) {
		const struct step_row_record *const step_row = &rows[row];
		const uint32_t slot = step_row->session_slot,
			       position = (uint32_t)step_row->position;
		if (slot >= limits->session_capacity ||
		    step_row->position < 0 ||
		    position >= limits->context_token_limit ||
		    step_row->token_id < 0 ||
		    (step_row->flags & ~(uint32_t)STEP_ROW_FLAGS) != 0 ||
		    (((step_row->flags & STEP_ROW_IMAGE) != 0) !=
		     step_image_position(driver, slot, position)) ||
		    ((step_row->flags & STEP_ROW_IMAGE) != 0 &&
		     (uint32_t)step_row->token_id != IMAGE_TOKEN_ID) ||
		    (step_row->flags & STEP_ROW_FRONT) != front ||
		    (step_row->flags & STEP_ROW_PADDED) != padded ||
		    (padded != 0 &&
		     (slot != rows[0].session_slot ||
		      (step_row->flags & (STEP_ROW_LOGITS | STEP_ROW_PROPOSED |
					  STEP_ROW_DRAFTS | STEP_ROW_IMAGE)) !=
			      0)) ||
		    (front != 0 && ((step_row->flags & STEP_ROW_LOGITS) != 0 ||
				    !has_front)) ||
		    (driver->vocabulary_size != 0 &&
		     (uint32_t)step_row->token_id >= driver->vocabulary_size) ||
		    (driver->next_positions[slot] == NO_ROW_YET ?
			     position > driver->token_counts[slot] :
			     position != driver->next_positions[slot])) {
			break;
		}
		driver->next_positions[slot] = position + 1u;
		speculative |= step_row->flags &
			       (STEP_ROW_PROPOSED | STEP_ROW_DRAFTS);
	}
	return row == row_count && speculative != 0 ?
		       speculative_wrong_row(driver, rows, row_count) :
		       row;
}

/* first_wrong_row already admitted this thin step: at most eight rows, each
 * session consecutive, its first row continuing or going back to a committed position.
 * Borrow the committed prefix and overlay only earlier rows of THIS session;
 * the current token is local. No committed history/count is written here. */
MUST_CHECK static result_status
next_engram_row_ids(const struct step_driver *driver,
		    const struct step_row_record *rows, uint32_t row_index,
		    uint64_t *row_ids_out)
{
	const struct step_row_record *const current = rows + row_index;
	const struct engram_description *const tables =
		engram_description_of(driver->engram);
	const uint64_t context =
		driver->programs[0]->limits.context_token_limit;
	const uint32_t position = (uint32_t)current->position;
	const uint32_t older_limit = tables->lookback - 1u;
	const uint32_t older = position < older_limit ? position : older_limit;
	uint32_t recent[ENGRAM_MAXIMUM_LOOKBACK];
	uint8_t dead[ENGRAM_MAXIMUM_LOOKBACK];
	uint32_t pending = 0;
	for (uint32_t prior = row_index; prior != 0 && pending < older;) {
		prior--;
		if (rows[prior].session_slot == current->session_slot) {
			pending++;
			recent[older - pending] =
				(uint32_t)rows[prior].token_id;
		}
	}
	const uint32_t committed = older - pending;
	if (committed != 0) {
		const uint64_t first =
			current->session_slot * context + position - older;
		memcpy(recent, driver->histories + first,
		       committed * sizeof(*recent));
	}
	recent[older] = (uint32_t)current->token_id;
    for (uint32_t token = 0; token <= older; token++) {
	    dead[token] = step_image_position(driver, current->session_slot,
					      position - older + token) ?
				  1u :
				  0u;
    }
    return engram_row_ids_entry(driver->engram, recent, dead, older,
				row_ids_out);
}
result_status step_row_ids(const struct step_driver *driver,
			   const struct step_row_record *rows,
			   uint32_t row_count, uint64_t *row_ids_out)
{
	const uint64_t token_rows =
		(uint64_t)driver->table_count * driver->rows_per_token;
	result_status status = OK;
	for (uint32_t row = 0; row < row_count && driver->engram != NULL;
	     row++) {
		status = THEN(next_engram_row_ids(
			driver, rows, row, &row_ids_out[row * token_rows]));
	}
	return status;
}

result_status step_engram_share(struct step_driver *driver)
{
	if (driver != NULL && driver->failure != OK)
		return driver->failure;
	if (driver == NULL || driver->companion_segments == NULL ||
	    driver->engram == NULL || driver->engram_shared ||
	    driver->sequence != 0 || driver->pipeline != NULL ||
	    driver->graphs_attempted)
		return STATUS_INVALID_ARGUMENT;
	driver->engram_shared = true;
	return OK;
}

/* A card's n-gram rows of every table it reads asked for first (their cache and TLB misses overlap); with gathers,
 * table by table gathered into its staging in block, each table's then going up when goes_up. */
MUST_CHECK static result_status card_tables(struct step_driver *driver,
					    unsigned char *block,
					    const uint64_t *row_ids,
					    uint32_t row_count, uint32_t card,
					    bool gathers, bool goes_up)
{
	const struct program_card_record *const area =
		&driver->programs[0]->cards[card];
	const uint64_t token_rows =
		(uint64_t)driver->table_count * driver->rows_per_token;
	const uint64_t table_bytes =
		(uint64_t)driver->rows_per_token * ENGRAM_ROW_BYTES;
	uint32_t source_card = card;
	for (uint32_t prior = 0; driver->engram_shared && prior < card;
	     prior++) {
		if (driver->programs[0]->cards[prior].engram_tables ==
		    area->engram_tables) {
			source_card = prior;
			break;
		}
	}
	/* Only serialized TP card visits may share. The source remains immutable until every card retires. */
	gathers = gathers && source_card == card;
	uint64_t staging = driver->engram_staging[source_card],
		 arena = area->engram_rows_offset;
	result_status status = OK;
	for (uint32_t table = 0; gathers && table < driver->table_count;
	     table++) {
		for (uint32_t row = 0;
		     ((area->engram_tables >> table) & 1u) != 0 &&
		     row < row_count;
		     row++)
			engram_prefetch_entry(
				driver->engram, table,
				&row_ids[row * token_rows +
					 (uint64_t)table *
						 driver->rows_per_token],
				driver->rows_per_token);
	}
	for (uint32_t table = 0; table < driver->table_count; table++) {
		const bool is_read = ((area->engram_tables >> table) & 1u) != 0;
		for (uint32_t row = 0; is_read && gathers && row < row_count;
		     row++) {
			status = THEN(engram_gather_entry(
				driver->engram, table,
				&row_ids[row * token_rows +
					 (uint64_t)table *
						 driver->rows_per_token],
				driver->rows_per_token,
				block + staging + row * table_bytes));
		}
		status = is_read && goes_up ?
				 THEN(step_copy_begin(
					 driver, card, arena, block + staging,
					 row_count * table_bytes, false)) :
				 status;
		staging += is_read ? driver->largest_row_count * table_bytes :
				     0;
		arena += is_read ? driver->programs[0]->limits.row_capacity *
					   table_bytes :
				   0;
	}
	return status;
}

result_status step_card_gather(struct step_driver *driver, unsigned char *block,
			       const uint64_t *row_ids, uint32_t row_count,
			       uint32_t card)
{
	return card_tables(driver, block, row_ids, row_count, card, true,
			   false);
}

result_status step_card_begin(struct step_driver *driver, unsigned char *block,
			      const uint64_t *row_ids, uint32_t row_count,
			      uint32_t card, bool gathers)
{
	const struct program_card_record *const area =
		&driver->programs[0]->cards[card];
	const result_status status =
		area->step_offset == NOWHERE ?
			OK :
			step_copy_begin(driver, card, area->step_offset, block,
					STEP_BLOCK_BYTES_VALUE +
						(uint64_t)row_count *
							STEP_ROW_BYTES_VALUE,
					false);
	return THEN(card_tables(driver, block, row_ids, row_count, card,
				gathers, true));
}

void step_block_write(unsigned char *block, uint64_t sequence,
		      const struct step_row_record *rows, uint32_t row_count)
{
	const struct step_block header = {
		.sequence = sequence,
		.row_count = row_count,
		.flags = row_count != 0 && (rows[0].flags & STEP_ROW_PADDED) !=
						   0 ?
				 STEP_BLOCK_PADDED :
				 0u
	};
	memcpy(block, &header, sizeof(header));
	memcpy(block + STEP_BLOCK_BYTES_VALUE, rows, row_count * sizeof(*rows));
}

/* The first card of a program that has views: every such card's are the same (step_validation.c); NULL for none. */
static const struct program_card_record *
viewing_card(const struct execution_program *program)
{
	for (uint32_t card = 0; card < program->card_count; card++) {
		if (program->cards[card].view_count != 0) {
			return &program->cards[card];
		}
	}
	return NULL;
}

/* The bytes of a card's views, one behind the other. */
static uint64_t views_bytes(const struct program_card_record *card)
{
	uint64_t byte_count = 0;
	for (uint32_t view = 0; card != NULL && view < card->view_count;
	     view++) {
		byte_count += STEP_BLOCK_BYTES_VALUE +
			      (uint64_t)card->views[view].row_limit *
				      STEP_ROW_BYTES_VALUE;
	}
	return byte_count;
}

void step_views_write(const struct step_driver *driver,
		      const struct execution_program *program,
		      unsigned char *block, uint64_t sequence,
		      const struct step_row_record *rows, uint32_t row_count)
{
	const struct program_card_record *const card = viewing_card(program);
	const uint32_t flags =
		row_count != 0 && (rows[0].flags & STEP_ROW_PADDED) != 0 ?
			STEP_BLOCK_PADDED :
			0u;
	uint64_t at = driver->view_staging;
	for (uint32_t view = 0; card != NULL && view < card->view_count;
	     view++) {
		const struct step_view *const own = &card->views[view];
		const uint32_t left = own->first_row < row_count ?
					      row_count - own->first_row :
					      0;
		step_block_write(block + at, sequence,
				 left == 0 ? rows : rows + own->first_row,
				 left < own->row_limit ? left : own->row_limit);
		if (flags != 0)
			memcpy(block + at + offsetof(struct step_block, flags),
			       &flags, sizeof(flags));
		at += STEP_BLOCK_BYTES_VALUE +
		      (uint64_t)own->row_limit * STEP_ROW_BYTES_VALUE;
	}
}

result_status step_views_begin(struct step_driver *driver,
			       const struct execution_program *program,
			       unsigned char *block, uint32_t card)
{
	const struct program_card_record *const own = &program->cards[card];
	return own->view_count == 0 ?
		       OK :
		       step_copy_begin(driver, card, own->views[0].block_offset,
				       block + driver->view_staging,
				       views_bytes(own), false);
}

/* Prepare rows from committed history plus this step; upload settings, step, views and n-gram bytes. */
MUST_CHECK static result_status
step_begin(struct step_driver *driver, const struct execution_program *program,
	   const struct step_row_record *rows, uint32_t row_count)
{
	const struct execution_program *const first = driver->programs[0];
	step_block_write(driver->staging, ++driver->sequence, rows, row_count);
	step_views_write(driver, program, driver->staging, driver->sequence,
			 rows, row_count);
	result_status status =
		step_row_ids(driver, rows, row_count, driver->row_ids);
	if (driver->sessions_changed) {
		status = THEN(step_copy_begin(
			driver, first->result_card,
			first->cards[first->result_card].session_offset,
			driver->staging + driver->session_staging,
			(uint64_t)first->limits.session_capacity *
				STEP_SESSION_BYTES,
			false));
		driver->sessions_changed =
			false; /* a failed step fails the driver: nothing is sent again */
	}
	for (uint32_t card = 0; card < first->card_count; card++) {
		status = THEN(step_card_begin(driver, driver->staging,
					      driver->row_ids, row_count, card,
					      true));
		status = THEN(step_views_begin(driver, program, driver->staging,
					       card));
	}
	return status;
}

/* The program's segments in order (a front step's: its front's, model_seam.h), each run of one card's that no
 * hand-off or expert pause divides as one launch_run, the hand-offs and the pauses; then the result area of the card
 * that samples, taken only if it answers this step (a front step's rows: no token, model_seam.h). */
MUST_CHECK static result_status
segments_run(struct step_driver *driver,
	     const struct execution_program *program,
	     const uint32_t *run_graphs, const struct step_row_record *rows,
	     uint32_t row_count, struct step_row_result *results_out)
{
	if (driver->companion_segments != NULL)
		return step_companions_run(driver, program, run_graphs, rows,
					   row_count, results_out);
	const struct hand_off_record *const hand_offs = program->hand_offs;
	const uint32_t result_card = program->result_card;
	const uint32_t segment_count =
		step_segment_count(program, rows, row_count);
	const bool is_front = step_is_front(rows, row_count);
	uint32_t hand_off = 0, pause = 0;
	result_status status = OK;
	for (uint32_t segment = 0, last = 0;
	     segment < segment_count && status == OK; segment = ++last) {
		const struct segment_record *const first =
			&program->segments[segment];
		last = step_run_last(program, segment, hand_off);
		const uint32_t graph =
			run_graphs != NULL ? run_graphs[segment] : 0;
		/* Both submissions can leave queued work on failure; only a later ticket or wait proves completion. */
		driver->uncovered_work[first->card] = true;
		status =
			graph != 0 ?
				driver->backend->graph_run(
					driver->cards[first->card],
					graph - 1u) :
				driver->backend->launch_run(
					driver->cards[first->card],
					&program->cards[first->card]
						 .launches[first->first_launch],
					step_run_launch_count(program, segment,
							      last));
		COUNTER_ADD(step_graph_runs, graph != 0 ? 1u : 0u);
		/* a front step leaves its deferred hand-offs to the mirror sync (step.h) */
		const uint32_t first_hand_off = hand_off;
		bool is_sent = false;
		for (; hand_off < program->hand_off_count &&
		       hand_offs[hand_off].after_segment == last;
		     hand_off++) {
			const bool is_kept = !is_front ||
					     driver->deferred[hand_off] == 0;
			status = is_kept ? THEN(step_hand_off_copies(
						   driver, driver->staging,
						   hand_off, rows, row_count,
						   true)) :
					   status;
			is_sent = is_sent || is_kept;
		}
		status = is_sent ? THEN(card_wait(driver, first->card)) :
				   status;
		for (uint32_t index = first_hand_off; index < hand_off;
		     index++) {
			status = !is_front || driver->deferred[index] == 0 ?
					 THEN(step_hand_off_copies(
						 driver, driver->staging, index,
						 rows, row_count, false)) :
					 status;
		}
		status = program->segments[last].pause == 0 ?
				 status :
				 THEN(step_experts_pause(driver, pause++,
							 row_count));
	}
	if (step_is_headless(program, rows, row_count)) {
		step_front_results(row_count, results_out);
		return status;
	}
	status = THEN(step_copy_begin(
		driver, result_card, program->cards[result_card].result_offset,
		driver->staging + driver->result_staging,
		step_result_span(program, rows, row_count), true));
	status = THEN(card_wait(driver, result_card));
	return THEN(step_result_take(driver, driver->staging, driver->sequence,
				     rows, row_count, results_out));
}

void step_front_results(uint32_t row_count, struct step_row_result *results_out)
{
	for (uint32_t row = 0; row < row_count && results_out != NULL; row++) {
		results_out[row] =
			(struct step_row_result){ STEP_NO_TOKEN, 0.0f };
	}
}

result_status step_result_take(struct step_driver *driver,
			       const unsigned char *block, uint64_t sequence,
			       const struct step_row_record *rows,
			       uint32_t row_count,
			       struct step_row_result *results_out)
{
	const unsigned char *const result_bytes =
		block + driver->result_staging;
	if (step_rounds_is_batch(driver))
		return step_batch_result_take(driver, result_bytes, sequence,
					      rows, row_count, results_out);
	struct step_result result = { 0 };
	memcpy(&result, result_bytes, sizeof(result));
	memcpy(results_out, result_bytes + STEP_RESULT_BYTES,
	       row_count * sizeof(*results_out));
	bool answers = result.sequence == sequence &&
		       result.row_count == row_count;
	for (uint32_t row = 0; row < row_count; row++) {
		const int32_t token = results_out[row].token_id;
		answers = answers &&
			  ((rows[row].flags & STEP_ROW_LOGITS) == 0 ?
				   token == STEP_NO_TOKEN :
				   token >= 0 &&
					   (driver->vocabulary_size == 0 ||
					    (uint32_t)token <
						    driver->vocabulary_size));
	}
	struct step_round round = { .session_slot = STEP_NO_SESSION };
	if (step_drafts(rows, row_count)) {
		const struct execution_program *const first =
			driver->programs[0];
		const struct program_card_record *const card =
			&first->cards[first->result_card];
		memcpy(&round,
		       result_bytes + card->round_offset - card->result_offset,
		       sizeof(round));
		uint32_t anchor = 0, proposed = 0, last = row_count;
		for (uint32_t row = 0; row < row_count; row++) {
			anchor = (rows[row].flags & STEP_ROW_DRAFTS) != 0 ?
					 row :
					 anchor;
			proposed += (rows[row].flags & STEP_ROW_PROPOSED) != 0 ?
					    1u :
					    0u;
		}
		for (uint32_t row = anchor; row < row_count; row++) {
			last = rows[row].session_slot ==
						       rows[anchor]
							       .session_slot &&
					       (int64_t)rows[row].position +
							       1 ==
						       round.anchor_position ?
				       row :
				       last;
		}
		answers = answers && round.sequence == sequence &&
			  round.reserved_zero == 0 &&
			  round.session_slot == rows[anchor].session_slot &&
			  round.emitted_count != 0 &&
			  round.emitted_count <= proposed + 1u &&
			  (int64_t)round.anchor_position ==
				  (int64_t)rows[anchor].position +
					  round.emitted_count &&
			  last != row_count &&
			  round.anchor_token == results_out[last].token_id &&
			  round.proposal_count == first->limits.draft_positions;
		for (uint32_t position = 0; position < STEP_DRAFT_MOST;
		     position++) {
			answers =
				answers &&
				(position < round.proposal_count ?
					 round.proposals[position] >= 0 &&
						 (driver->vocabulary_size ==
							  0 ||
						  (uint32_t)round.proposals
								  [position] <
							  driver->vocabulary_size) &&
						 isfinite(
							 round.confidence_logits
								 [position]) :
					 round.proposals[position] == 0 &&
						 round.confidence_logits
								 [position] ==
							 0.0f);
		}
	}
	if (answers) {
		driver->last_round = round;
	}
	return answers ? OK : STATUS_DEVICE_ERROR;
}

static void driver_free(struct step_driver *driver)
{
	free((void *)driver->programs);
	free((void *)driver->image_programs);
	free(driver->histories);
	free(driver->run_graphs);
	free(driver->companion_segments);
	free(driver->hand_off_staging);
	free(driver->deferred);
	free(driver->batch_verifiers);
	free(driver);
}

/* step.h's deferred hand-offs of programs[0]: of an entry ratio, behind a segment of the front, to a card that runs a
 * segment of the front before it and none behind it there, of an entry a piece of the session block takes; all from one
 * card behind one segment, else none (a pipeline's sync stands before the visit of that segment: older steps have
 * written by then all it moves). */
static void mirrors_find(struct step_driver *driver)
{
	const struct execution_program *const first = driver->programs[0];
	const uint32_t front = first->front_segment_count;
	bool is_one_source = true;
	driver->mirror_source = BACKEND_MAXIMUM_CARDS;
	driver->mirror_segment = UINT32_MAX;
	for (uint32_t index = 0; index < first->hand_off_count; index++) {
		const struct hand_off_record *const hand_off =
			&first->hand_offs[index];
		uint32_t before = 0, behind = 0;
		for (uint32_t segment = 0; segment < front; segment++) {
			const bool is_to = first->segments[segment].card ==
					   hand_off->to_card;
			before += is_to && segment < hand_off->after_segment ?
					  1u :
					  0u;
			behind += is_to && segment > hand_off->after_segment ?
					  1u :
					  0u;
		}
		const bool is_deferred = hand_off->entry_ratio != 0 &&
					 hand_off->after_segment < front &&
					 before != 0 && behind == 0 &&
					 hand_off->byte_count <=
						 STEP_SESSION_STAGING_BYTES;
		is_one_source =
			is_one_source &&
			(!is_deferred ||
			 driver->mirror_source == BACKEND_MAXIMUM_CARDS ||
			 (driver->mirror_source == hand_off->from_card &&
			  driver->mirror_segment == hand_off->after_segment));
		driver->deferred[index] = is_deferred ? 1u : 0u;
		driver->mirror_source = is_deferred ? hand_off->from_card :
						      driver->mirror_source;
		driver->mirror_cards |= is_deferred ? 1u << hand_off->to_card :
						      0u;
		driver->mirror_segment = is_deferred ? hand_off->after_segment :
						       driver->mirror_segment;
	}
	if (!is_one_source) {
		memset(driver->deferred, 0, first->hand_off_count);
		driver->mirror_source = BACKEND_MAXIMUM_CARDS;
		driver->mirror_cards = 0;
		driver->mirror_segment = UINT32_MAX;
	}
}

uint32_t step_mirror_ranges(struct step_driver *driver,
			    const struct step_row_record *rows,
			    uint32_t row_count, uint32_t *syncs_out)
{
	const bool is_front = step_is_front(rows, row_count);
	uint32_t count = 0;
	for (uint32_t row = 0; row < row_count && driver->mirror_cards != 0;
	     row++) {
		const uint32_t slot = rows[row].session_slot,
			       position = (uint32_t)rows[row].position;
		uint32_t *const from = &driver->mirror_from[slot],
				*const to = &driver->mirror_to[slot];
		const bool is_empty = *from >= *to;
		/* the session's first row of a step of other rows: its rows write from there on */
		const uint32_t end = position < *to ? position : *to;
		if (!is_front && !is_empty && *from < end) {
			uint32_t *const sync =
				&syncs_out[(uint64_t)count * STEP_SYNC_FIELDS];
			count++;
			sync[0] = slot;
			sync[1] = *from;
			sync[STEP_SYNC_FIELDS - 1u] = end;
		}
		*to = !is_front			      ? 0 :
		      is_empty || position + 1u > *to ? position + 1u :
							*to;
		*from = !is_front		     ? 0 :
			is_empty || position < *from ? position :
						       *from;
	}
	return count;
}

result_status
step_driver_create(const struct backend_operations *backend,
		   struct device_card *const *cards,
		   const struct execution_program *const *programs,
		   uint32_t program_count, const struct engram_record *engram,
		   uint64_t wait_limit_nanoseconds,
		   struct step_driver **driver_out)
{
	const char *problem =
		backend == NULL || backend->staging_allocate == NULL ||
				backend->staging_free == NULL ||
				backend->upload_begin == NULL ||
				backend->upload_wait == NULL ||
				backend->download_begin == NULL ||
				backend->launch_run == NULL ||
				backend->launch_wait == NULL || cards == NULL ||
				programs == NULL || program_count == 0 ||
				wait_limit_nanoseconds == 0 ||
				driver_out == NULL ?
			"a missing argument or function of the backend, no program or no wait" :
			NULL;
	bool is_unsupported = false;
	uint32_t largest = 0;
	uint64_t view_bytes = 0; /* the most a program's views take on a card */
	for (uint32_t index = 0; index < program_count && problem == NULL;
	     index++) {
		problem = step_program_problem(programs[index], programs[0],
					       cards, engram, &is_unsupported);
		/* a step's rows find one program: a wide program takes front rows, a full one other rows of its
		 * exact count, the thin ones the rest, so two programs of one kind and row count would compete; BATCH
		 * programs of one row count are told apart by their verifiers (step_batch_verifiers_attach) */
		for (uint32_t other = 0; other < index && problem == NULL;
		     other++) {
			problem =
				programs[other]->row_count ==
							programs[index]
								->row_count &&
						programs[other]->kind ==
							programs[index]->kind &&
						programs[index]->kind !=
							PROGRAM_BATCH ?
					"two programs of one kind and row count" :
					NULL;
		}
		largest = problem == NULL &&
					  programs[index]->row_count > largest ?
				  programs[index]->row_count :
				  largest;
		const uint64_t own_view_bytes =
			problem == NULL ?
				views_bytes(viewing_card(programs[index])) :
				0;
		view_bytes = own_view_bytes > view_bytes ? own_view_bytes :
							   view_bytes;
	}
	if (problem != NULL) {
		LOG_ERROR("step driver: refused: %s", problem);
		return is_unsupported ? STATUS_UNSUPPORTED :
					STATUS_INVALID_ARGUMENT;
	}
	const struct execution_program *const first = programs[0];
	const struct model_limits *const limits = &first->limits;
	const struct engram_description *const tables =
		engram != NULL ? engram_description_of(engram) : NULL;
	const uint64_t token_rows =
		tables != NULL ?
			(uint64_t)tables->table_count * tables->rows_per_token :
			0;
	const uint64_t histories = (uint64_t)limits->session_capacity *
				   limits->context_token_limit;
	const uint64_t pause_words =
		(uint64_t)STEP_PAUSE_REGIONS * first->expert_pause_count;
	uint64_t slot_count = 0;
	uint32_t first_slot[BACKEND_MAXIMUM_CARDS] = { 0 },
		 cards_with_slots = 0;
	for (uint32_t index = 0; index < first->expert_pause_count;
	     index++) { /* the slots of a card: in its first */
		const struct expert_pause *const pause =
			&first->expert_pauses[index];
		first_slot[pause->card] =
			(cards_with_slots >> pause->card & 1u) != 0 ?
				first_slot[pause->card] :
				(uint32_t)slot_count;
		slot_count += (cards_with_slots >> pause->card & 1u) != 0 ?
				      0 :
				      pause->slot_count;
		cards_with_slots |= 1u << pause->card;
	}
	struct step_driver *const driver = FAULT_POINT(step_driver_memory) ?
						   NULL :
						   calloc(1, sizeof(*driver));
	if (driver == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	*driver = (struct step_driver){
		.backend = backend,
		.engram = engram,
		.programs = (const struct execution_program **)calloc(
			program_count, sizeof(*driver->programs)),
		.histories =
			FAULT_POINT(step_driver_memory) ?
				NULL /* the fault point's second reach */
				:
				calloc(histories +
					       (uint64_t)limits->session_capacity *
						       SESSION_COUNTERS,
				       sizeof(uint32_t)),
		.hand_off_staging =
			slot_count > UINT32_MAX ?
				NULL :
				calloc(first->hand_off_count + pause_words +
					       largest * token_rows +
					       slot_count * STEP_SLOT_FIELDS +
					       limits->session_capacity + 1u,
				       sizeof(uint64_t)),
		.deferred = calloc(first->hand_off_count + 1u, sizeof(uint8_t)),
		.wait_limit_nanoseconds = wait_limit_nanoseconds,
		.program_count = program_count,
		.largest_row_count = largest,
		.table_count = tables != NULL ? tables->table_count : 0,
		.sessions_changed = true,
		.rows_per_token = tables != NULL ? tables->rows_per_token : 0,
		.vocabulary_size = tables != NULL ? tables->vocabulary_size : 0
	};
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
		atomic_init(&driver->live_wait_sequence[card], 0);
		atomic_init(&driver->live_wait_completed_ns[card], 0);
		atomic_init(&driver->live_wait_started_ns[card], 0);
	}
	/* the regions of the block: step area, settings, result, each card's n-gram rows, each hand-off's and expert
     * pause's bytes */
	memcpy(driver->first_slot, first_slot, sizeof(first_slot));
	driver->pause_staging =
		driver->hand_off_staging != NULL ?
			driver->hand_off_staging + first->hand_off_count :
			NULL;
	driver->session_staging = STEP_BLOCK_BYTES_VALUE +
				  (uint64_t)largest * STEP_ROW_BYTES_VALUE;
	driver->result_staging =
		driver->session_staging +
		limits->session_capacity * (uint64_t)STEP_SESSION_BYTES;
	uint64_t result_bytes =
		STEP_RESULT_BYTES + (uint64_t)largest * STEP_ROW_RESULT_BYTES;
	/* The program pointer table is populated below; use the validated caller's first program here. */
	if (first->kind == PROGRAM_BATCH) {
		for (uint32_t index = 0; index < program_count; index++) {
			const uint64_t bytes = step_result_bytes(
				programs[index], programs[index]->row_count);
			result_bytes = bytes > result_bytes ? bytes :
							      result_bytes;
		}
	}
	const struct program_card_record *const result_card =
		&first->cards[first->result_card];
	const uint64_t round_span =
		first->draft_segment_first < first->segment_count ?
			result_card->round_offset + step_round_bytes(first) -
				result_card->result_offset :
			0;
	driver->view_staging =
		driver->result_staging +
		(round_span > result_bytes ? round_span : result_bytes);
	uint64_t end = driver->view_staging + view_bytes;
	for (uint32_t card = 0; card < first->card_count; card++) {
		driver->cards[card] = cards[card];
		driver->engram_staging[card] = end;
		end += (uint64_t)__builtin_popcount(
			       first->cards[card].engram_tables) *
		       largest * driver->rows_per_token * ENGRAM_ROW_BYTES;
	}
	for (uint32_t index = 0;
	     index < first->hand_off_count && driver->hand_off_staging != NULL;
	     index++) {
		uint32_t rows = 0; /* the most a program that runs it carries: a wide one runs the front's alone */
		for (uint32_t program = 0; program < program_count; program++) {
			rows = index < programs[program]->hand_off_count &&
					       programs[program]->row_count > rows ?
				       programs[program]->row_count :
				       rows;
		}
		const uint64_t byte_count =
			(uint64_t)rows * first->hand_offs[index].byte_count;
		driver->hand_off_staging[index] = end;
		end = end > UINT64_MAX - byte_count ? UINT64_MAX :
						      end + byte_count;
	}
	for (uint32_t index = 0; index < first->expert_pause_count &&
				 driver->hand_off_staging != NULL;
	     index++) {
		const uint64_t byte_count =
			(uint64_t)largest *
			first->expert_pauses[index].jobs_per_row *
			EXPERT_JOB_BYTES;
		driver->pause_staging[index] = end;
		end = end > UINT64_MAX - byte_count ? UINT64_MAX :
						      end + byte_count;
	}
	result_status status =
		driver->programs == NULL || driver->histories == NULL ||
				driver->hand_off_staging == NULL ||
				driver->deferred == NULL || end == UINT64_MAX ?
			STATUS_OUT_OF_MEMORY :
			backend->staging_allocate(end,
						  (void **)&driver->staging);
	if (status == OK) {
		memcpy((void *)driver->programs, (const void *)programs,
		       program_count * sizeof(*driver->programs));
		mirrors_find(driver);
	}
	/* the mirror sync's pieces, taken now so that no step allocates */
	status = status == OK && driver->mirror_cards != 0 ?
			 backend->staging_allocate(
				 STEP_SESSION_STAGING_BYTES,
				 (void **)&driver->session_block) :
			 status;
	if (status != OK) {
		backend->staging_free(driver->staging);
		driver_free(driver);
		return status;
	}
	driver->staging_bytes = end;
	for (uint64_t slot = 0; slot < limits->session_capacity; slot++) {
		const struct step_session settings = { .seed = 0,
						       .temperature = 0.0f,
						       .top_p = 1.0f };
		memcpy(driver->staging + driver->session_staging +
			       slot * STEP_SESSION_BYTES,
		       &settings, sizeof(settings));
	}
	driver->token_counts = driver->histories + histories;
	driver->next_positions =
		driver->token_counts + limits->session_capacity;
	driver->mirror_from = driver->next_positions + limits->session_capacity;
	driver->mirror_to = driver->mirror_from + limits->session_capacity;
	driver->mirror_syncs = driver->mirror_to + limits->session_capacity;
	driver->pause_copies =
		driver->pause_staging + first->expert_pause_count;
	driver->row_ids = driver->pause_copies + first->expert_pause_count;
	driver->slots = driver->row_ids + largest * token_rows;
	driver->session_sequences =
		driver->slots + slot_count * STEP_SLOT_FIELDS;
	*driver_out = driver;
	return OK;
}

void step_driver_destroy(struct step_driver *driver)
{
	bool staging_is_read = false;
	if (driver == NULL) {
		return;
	}
	step_pipeline_stop(driver); /* the cards are this thread's again */
	step_launchers_stop(driver);
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
		staging_is_read = card_wait(driver, card) != OK ||
				  staging_is_read;
	}
	step_pipeline_free(driver, staging_is_read);
	step_images_free(driver, staging_is_read);
	if (staging_is_read && driver->backend->staging_retire != NULL) {
		driver->backend->staging_retire(driver->staging);
		driver->backend->staging_retire(driver->expert_copies);
		driver->backend->staging_retire(driver->session_block);
	} else if (staging_is_read) {
		LOG_ERROR(
			"step driver: its staging memory stays allocated: work may still be in flight");
	} else {
		driver->backend->staging_free(driver->staging);
		driver->backend->staging_free(driver->expert_copies);
		driver->backend->staging_free(driver->session_block);
	}
	driver_free(driver);
}

result_status step_session_set(struct step_driver *driver,
			       uint32_t session_slot,
			       const struct step_session *settings)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	if (driver == NULL || settings == NULL ||
	    session_slot >= driver->programs[0]->limits.session_capacity ||
	    !isfinite(settings->temperature) || settings->temperature < 0.0f ||
	    !(settings->top_p > 0.0f && settings->top_p <= 1.0f)) {
		LOG_ERROR(
			"step: session settings refused: no driver or settings, no such slot, or out of range");
		return STATUS_INVALID_ARGUMENT;
	}
	memcpy(driver->staging + driver->session_staging +
		       (uint64_t)session_slot * STEP_SESSION_BYTES,
	       settings, sizeof(*settings));
	driver->sessions_changed = true;
	return OK;
}

result_status step_rows_check(struct step_driver *driver,
			      const struct step_row_record *rows,
			      uint32_t row_count, bool has_results)
{
	const uint32_t wrong_row =
		driver != NULL && rows != NULL ?
			first_wrong_row(driver, rows, row_count) :
			0;
	const uint32_t *run_graphs = NULL;
	const struct execution_program *const program =
		driver != NULL && rows != NULL && row_count != 0 &&
				wrong_row == row_count ?
			step_program_for_rows(driver, rows, row_count,
					      &run_graphs) :
			NULL;
	uint32_t logits_rows = 0,
		 speculative_rows =
			 0; /* full programs have no head or speculative rows */
	for (uint32_t row = 0; program != NULL && row < row_count; row++) {
		logits_rows += (rows[row].flags & STEP_ROW_LOGITS) != 0 ? 1u :
									  0u;
		speculative_rows += (rows[row].flags & (STEP_ROW_PROPOSED |
							STEP_ROW_DRAFTS)) != 0 ?
					    1u :
					    0u;
	}
	if (program == NULL || !has_results ||
	    (step_rounds_is_batch(driver) && !driver->round_domain_attached) ||
	    (program->kind == PROGRAM_FULL && logits_rows != 0) ||
	    (program->kind != PROGRAM_THIN && program->kind != PROGRAM_BATCH &&
	     speculative_rows != 0) ||
	    (driver->programs[0]->host_expert_count != 0 &&
	     driver->expert_copies == NULL)) {
		LOG_ERROR(
			"step: refused: no driver, rows, results or attached host experts, a step no program runs, "
			"or row %" PRIu32 " of %" PRIu32
			" breaks step_records.h or step.h",
			wrong_row, row_count);
		return STATUS_INVALID_ARGUMENT;
	}
	return OK;
}

const struct execution_program *
step_program_of(const struct step_driver *driver, uint32_t row_count,
		bool is_front, const uint32_t **run_graphs_out)
{
	const struct execution_program *program = NULL;
	size_t graph_base = 0;
	*run_graphs_out = NULL;
	for (uint32_t index = 0; index < driver->program_count; index++) {
		const struct execution_program *const candidate =
			driver->programs[index];
		const bool holds =
			candidate->kind == PROGRAM_WIDE ?
				is_front && candidate->row_count == row_count :
			candidate->kind == PROGRAM_FULL ?
				!is_front && candidate->row_count == row_count :
				candidate->row_count >= row_count;
		const bool is_fewer =
			holds && (program == NULL ||
				  candidate->row_count < program->row_count);
		program = is_fewer ? candidate : program;
		if (driver->run_graphs != NULL) {
			*run_graphs_out =
				is_fewer ? driver->run_graphs + graph_base :
					   *run_graphs_out;
			graph_base += candidate->segment_count;
		}
	}
	return program;
}

uint32_t step_wide_rows(const struct step_driver *driver)
{
	return step_chunk_rows(driver, PROGRAM_WIDE, UINT32_MAX);
}

uint32_t step_chunk_rows(const struct step_driver *driver,
			 enum program_kind kind, uint32_t most_rows)
{
	uint32_t rows = 0;
	for (uint32_t index = 0; index < driver->program_count; index++) {
		const struct execution_program *const program =
			driver->programs[index];
		rows = program->kind == kind &&
				       program->row_count <= most_rows &&
				       program->row_count > rows ?
			       program->row_count :
			       rows;
	}
	return rows;
}

uint32_t step_cone_rows(const struct step_driver *driver)
{
	return driver->programs[0]->front_segment_count != 0 ?
		       driver->programs[0]->cone_rows :
		       0;
}

uint32_t step_history_token(const struct step_driver *driver,
			    uint32_t session_slot, uint32_t position)
{
	const uint64_t context =
		driver->programs[0]->limits.context_token_limit;
	return driver->histories[session_slot * context + position];
}

uint32_t step_draft_positions(const struct step_driver *driver)
{
	return driver != NULL ? driver->programs[0]->limits.draft_positions : 0;
}

result_status step_round_take(const struct step_driver *driver,
			      struct step_round *round_out)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	if (driver == NULL || round_out == NULL || !driver->has_finished) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (step_rounds_is_batch(driver)) {
		const uint32_t mask = driver->last_rounds.valid_mask;
		if ((mask & (mask - 1u)) != 0)
			return STATUS_INVALID_ARGUMENT;
		if (mask != 0) {
			*round_out =
				driver->last_rounds
					.rounds[(uint32_t)__builtin_ctz(mask)];
			return OK;
		}
	}
	*round_out = driver->last_round;
	return OK;
}

result_status step_round_take_for_session(const struct step_driver *driver,
					  uint32_t session_slot,
					  struct step_round *round_out)
{
	if (driver != NULL && driver->failure != OK)
		return driver->failure;
	if (driver == NULL || round_out == NULL || !driver->has_finished ||
	    session_slot >= driver->programs[0]->limits.session_capacity)
		return STATUS_INVALID_ARGUMENT;
	if (step_rounds_is_batch(driver)) {
		if ((driver->last_rounds.valid_mask & (1u << session_slot)) ==
		    0)
			return STATUS_INVALID_ARGUMENT;
		*round_out = driver->last_rounds.rounds[session_slot];
	} else {
		if (driver->last_round.session_slot != session_slot)
			return STATUS_INVALID_ARGUMENT;
		*round_out = driver->last_round;
	}
	return OK;
}

void step_history_commit(struct step_driver *driver,
			 const struct step_row_record *rows, uint32_t row_count)
{
	const uint64_t context =
		driver->programs[0]->limits.context_token_limit;
	for (uint32_t row = 0; row < row_count; row++) {
		const uint32_t slot = rows[row].session_slot;
		driver->histories[slot * context + (uint32_t)rows[row].position] =
			(uint32_t)rows[row].token_id;
		driver->token_counts[slot] = (uint32_t)rows[row].position + 1u;
		driver->session_sequences[slot] = driver->sequence;
	}
}

result_status step_history_truncate(struct step_driver *driver,
				    uint32_t session_slot, uint32_t token_count)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	if (driver == NULL ||
	    session_slot >= driver->programs[0]->limits.session_capacity ||
	    token_count > driver->token_counts[session_slot] ||
	    step_pipeline_holds(driver,
				driver->session_sequences[session_slot])) {
		LOG_ERROR(
			"step: a cut refused: no driver, no such slot, more tokens than the session holds, "
			"or a step of it in flight");
		return STATUS_INVALID_ARGUMENT;
	}
	driver->token_counts[session_slot] = token_count;
	return OK;
}

result_status step_run_entry(struct step_driver *driver,
			     const struct step_row_record *rows,
			     uint32_t row_count,
			     struct step_row_result *results_out)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	if (driver != NULL && driver->pipeline != NULL) {
		result_status status =
			results_out == NULL ?
				step_rows_check(driver, rows, row_count,
						false) :
				step_submit(driver, rows, row_count);
		return THEN(step_finish(driver, results_out));
	}
	result_status status =
		step_rows_check(driver, rows, row_count, results_out != NULL);
	if (status != OK) {
		return status;
	}
	TIMER_BEGIN(step_run);
	const uint32_t *run_graphs = NULL;
	const struct execution_program *const program =
		step_program_for_rows(driver, rows, row_count, &run_graphs);
	/* the mirrors of the step's sessions take what earlier front steps left them, before the step reads them */
	const uint32_t sync_count = step_mirror_ranges(driver, rows, row_count,
						       driver->mirror_syncs);
	for (uint32_t sync = 0; sync < sync_count; sync++) {
		const uint32_t *const own =
			&driver->mirror_syncs[(uint64_t)sync * STEP_SYNC_FIELDS];
		status = THEN(step_mirror_sync(driver, own[0], own[1],
					       own[STEP_SYNC_FIELDS - 1u]));
	}
	status = program == NULL ?
			 STATUS_INVALID_ARGUMENT :
			 THEN(step_begin(driver, program, rows, row_count));
	status = THEN(segments_run(driver, program, run_graphs, rows, row_count,
				   results_out));
	for (uint32_t card = 0; card < driver->programs[0]->card_count;
	     card++) {
		if (driver->companion_segments != NULL) {
			const result_status waited = card_wait(driver, card);
			status = status == OK ? waited : status;
		} else {
			status = THEN(card_wait(driver, card));
		}
	}
	if (status == OK) {
		/* Validated result and all copy/launch completions succeeded. Host history only;
         * device state has no rollback here and any failure still poisons the driver. */
		uint32_t seen_cards = 0;
		const uint32_t segment_count =
			step_segment_count(program, rows, row_count);
		for (uint32_t segment = 0; segment < segment_count; segment++) {
			const uint32_t card = program->segments[segment].card;
			driver->live_visits[card] +=
				segment == 0 || program->segments[segment - 1u]
								.card != card ?
					1u :
					0u;
			driver->live_steps[card] +=
				(seen_cards & (1u << card)) == 0 ? 1u : 0u;
			seen_cards |= 1u << card;
		}
		step_history_commit(driver, rows, row_count);
		if (!step_drafts(rows, row_count)) {
			driver->last_rounds = (struct step_round_table){ 0 };
			driver->last_round =
				(struct step_round){ .session_slot =
							     STEP_NO_SESSION };
		}
		driver->has_finished = true;
	}
	if (status != OK) {
		LOG_ERROR("step %" PRIu64
			  " failed (%s): the driver answers so from now on",
			  driver->sequence, status_text(status));
	}
	driver->failure = status;
	TIMER_END(step_run);
	return status;
}

/* Optional test observation only. The release translation unit contains neither this function nor its copies. */
#if CHECKING_HOOKS_ENABLED
result_status step_committed_history_read(const struct step_driver *driver,
					  uint32_t session_slot,
					  uint32_t *history_out,
					  uint64_t history_capacity,
					  uint32_t *token_count_out)
{
	if (driver == NULL || history_out == NULL || token_count_out == NULL ||
	    session_slot >= driver->programs[0]->limits.session_capacity ||
	    history_capacity <
		    driver->programs[0]->limits.context_token_limit) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t context =
		driver->programs[0]->limits.context_token_limit;
	memcpy(history_out, driver->histories + session_slot * context,
	       (size_t)context * sizeof(*history_out));
	*token_count_out = driver->token_counts[session_slot];
	return OK;
}
#endif
