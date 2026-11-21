/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "step_rounds.h"
#include "step_driver.h"
#include "step_runs.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#define NO_ROW UINT32_MAX

_Static_assert(sizeof(struct step_round) == STEP_ROUND_BYTES,
	       "the existing round ABI stays unchanged");

struct session_rows {
	uint32_t count, first_position, last_position, first_proposed, anchor;
};

static bool separate(const void *first, uint64_t first_bytes,
		     const void *second, uint64_t second_bytes)
{
	const uintptr_t first_address = (uintptr_t)first,
			second_address = (uintptr_t)second;
	return first_address <= UINTPTR_MAX - first_bytes &&
	       second_address <= UINTPTR_MAX - second_bytes &&
	       (first_address + first_bytes <= second_address ||
		second_address + second_bytes <= first_address);
}

static uint32_t row_at(const struct step_round_description *description,
		       uint32_t session, uint32_t position)
{
	for (uint32_t row = 0; row < description->row_count; row++)
		if (description->rows[row].session_slot == session &&
		    (uint32_t)description->rows[row].position == position)
			return row;
	return NO_ROW;
}

result_status
step_rounds_validate(const struct step_round_description *description,
		     struct step_round_table *output)
{
	if (description == NULL || output == NULL ||
	    description->rows == NULL || description->results == NULL ||
	    description->result_header == NULL || description->sequence == 0 ||
	    description->reserved_zero != 0 || description->row_count == 0 ||
	    description->row_count > STEP_BATCH_MAXIMUM_ROWS ||
	    description->context_token_limit == 0 ||
	    description->context_token_limit > INT32_MAX ||
	    description->session_capacity == 0 ||
	    description->session_capacity > STEP_BATCH_MAXIMUM_SESSIONS ||
	    description->vocabulary_size == 0 ||
	    description->vocabulary_size > INT32_MAX ||
	    description->draft_positions > STEP_DRAFT_MOST ||
	    description->end_token_id >= description->vocabulary_size ||
	    (description->round_capacity > 1u &&
	     description->round_capacity != description->session_capacity))
		return STATUS_INVALID_ARGUMENT;
	const uint32_t capacity = description->round_capacity == 0 ?
					  1u :
					  description->round_capacity;
	if (!separate(output, sizeof(*output), description,
		      sizeof(*description)) ||
	    !separate(output, sizeof(*output), description->rows,
		      (uint64_t)description->row_count *
			      sizeof(*description->rows)) ||
	    !separate(output, sizeof(*output), description->results,
		      (uint64_t)description->row_count *
			      sizeof(*description->results)) ||
	    !separate(output, sizeof(*output), description->result_header,
		      sizeof(*description->result_header)) ||
	    (description->rounds != NULL &&
	     !separate(output, sizeof(*output), description->rounds,
		       (uint64_t)capacity * sizeof(*description->rounds))))
		return STATUS_INVALID_ARGUMENT;
	const struct step_result *const header = description->result_header;
	if (header->sequence != description->sequence ||
	    header->row_count != description->row_count ||
	    header->reserved_zero != 0)
		return STATUS_DEVICE_ERROR;
	struct session_rows sessions[STEP_BATCH_MAXIMUM_SESSIONS] = { 0 };
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		sessions[slot].first_position = UINT32_MAX;
		sessions[slot].first_proposed = UINT32_MAX;
		sessions[slot].anchor = NO_ROW;
	}
	uint32_t verifier_count = 0;
	const bool front_step = (description->rows[0].flags & STEP_ROW_FRONT) !=
				0;
	for (uint32_t index = 0; index < description->row_count; index++) {
		const struct step_row_record *const row =
			&description->rows[index];
		const bool logits = (row->flags & STEP_ROW_LOGITS) != 0;
		const bool front = (row->flags & STEP_ROW_FRONT) != 0;
		const bool proposed = (row->flags & STEP_ROW_PROPOSED) != 0;
		const bool drafts = (row->flags & STEP_ROW_DRAFTS) != 0;
		const int32_t drawn = description->results[index].token_id;
		if (row->session_slot >= description->session_capacity ||
		    row->position < 0 ||
		    (uint32_t)row->position >=
			    description->context_token_limit ||
		    row->token_id < 0 ||
		    (uint32_t)row->token_id >= description->vocabulary_size ||
		    (row->flags & ~(uint32_t)STEP_ROW_FLAGS) != 0 ||
		    (front != front_step) || (front && logits) ||
		    ((proposed || drafts) && (!logits || front)) ||
		    (proposed && drafts) ||
		    (logits ?
			     drawn < 0 || (uint32_t)drawn >=
						  description->vocabulary_size :
			     drawn != STEP_NO_TOKEN))
			return STATUS_DEVICE_ERROR;
		for (uint32_t prior = 0; prior < index; prior++)
			if (description->rows[prior].session_slot ==
				    row->session_slot &&
			    description->rows[prior].position == row->position)
				return STATUS_DEVICE_ERROR;
		struct session_rows *const own = &sessions[row->session_slot];
		const uint32_t position = (uint32_t)row->position;
		if (own->count != 0 && position != own->last_position + 1u)
			return STATUS_DEVICE_ERROR;
		own->count++;
		if (position < own->first_position)
			own->first_position = position;
		if (position > own->last_position)
			own->last_position = position;
		if (proposed && position < own->first_proposed)
			own->first_proposed = position;
		if (drafts) {
			if (own->anchor != NO_ROW ||
			    description->draft_positions == 0)
				return STATUS_DEVICE_ERROR;
			own->anchor = index;
			verifier_count++;
		}
	}
	if (verifier_count != 0 && (description->rounds == NULL ||
				    (capacity == 1u && verifier_count > 1u)))
		return STATUS_INVALID_ARGUMENT;
	struct step_round_table pending;
	memset(&pending, 0, sizeof(pending));
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++)
		pending.rounds[slot].session_slot = STEP_NO_SESSION;
	for (uint32_t slot = 0; slot < description->session_capacity; slot++) {
		const struct session_rows *const own = &sessions[slot];
		if (own->count == 0)
			continue;
		if ((uint64_t)own->last_position - own->first_position + 1u !=
		    own->count)
			return STATUS_DEVICE_ERROR;
		if (own->first_proposed != UINT32_MAX) {
			const uint32_t prior =
				own->first_proposed > own->first_position ?
					row_at(description, slot,
					       own->first_proposed - 1u) :
					NO_ROW;
			if (prior == NO_ROW ||
			    (description->rows[prior].flags &
			     STEP_ROW_LOGITS) == 0 ||
			    own->last_position - own->first_proposed + 1u >
				    STEP_DRAFT_MOST)
				return STATUS_DEVICE_ERROR;
			for (uint32_t row = 0; row < description->row_count;
			     row++)
				if (description->rows[row].session_slot ==
					    slot &&
				    (((description->rows[row].flags &
				       STEP_ROW_PROPOSED) != 0) !=
				     ((uint32_t)description->rows[row].position >=
				      own->first_proposed)))
					return STATUS_DEVICE_ERROR;
		}
		if (own->anchor == NO_ROW)
			continue;
		const uint32_t anchor =
			(uint32_t)description->rows[own->anchor].position;
		const uint32_t proposed = own->last_position - anchor;
		if (proposed > description->draft_positions ||
		    (uint64_t)anchor + proposed + 1u +
				    description->draft_positions >
			    description->context_token_limit)
			return STATUS_DEVICE_ERROR;
		for (uint32_t row = 0; row < description->row_count; row++) {
			if (description->rows[row].session_slot != slot)
				continue;
			const bool after =
				(uint32_t)description->rows[row].position >
				anchor;
			if (((description->rows[row].flags &
			      STEP_ROW_PROPOSED) != 0) != after)
				return STATUS_DEVICE_ERROR;
		}
		uint32_t emitted = 1;
		while (emitted <= proposed) {
			const uint32_t drawn_row = row_at(
				description, slot, anchor + emitted - 1u);
			const uint32_t proposal_row =
				row_at(description, slot, anchor + emitted);
			if (drawn_row == NO_ROW || proposal_row == NO_ROW)
				return STATUS_DEVICE_ERROR;
			const int32_t drawn =
				description->results[drawn_row].token_id;
			if ((uint32_t)drawn == description->end_token_id ||
			    drawn != description->rows[proposal_row].token_id)
				break;
			emitted++;
		}
		const uint32_t last =
			row_at(description, slot, anchor + emitted - 1u);
		const struct step_round *const round =
			&description->rounds[capacity == 1u ? 0 : slot];
		if (last == NO_ROW ||
		    round->sequence != description->sequence ||
		    round->session_slot != slot || round->reserved_zero != 0 ||
		    round->emitted_count != emitted ||
		    (int64_t)round->anchor_position !=
			    (int64_t)anchor + emitted ||
		    round->anchor_token !=
			    description->results[last].token_id ||
		    round->proposal_count != description->draft_positions)
			return STATUS_DEVICE_ERROR;
		for (uint32_t position = 0; position < STEP_DRAFT_MOST;
		     position++) {
			if (position < round->proposal_count ?
				    (round->proposals[position] < 0 ||
				     (uint32_t)round->proposals[position] >=
					     description->vocabulary_size ||
				     !isfinite(
					     round->confidence_logits[position])) :
				    (round->proposals[position] != 0 ||
				     round->confidence_logits[position] !=
					     0.0f))
				return STATUS_DEVICE_ERROR;
		}
		pending.rounds[slot] = *round;
		pending.valid_mask |= 1u << slot;
	}
	memcpy(output, &pending, sizeof(pending));
	return STATUS_OK;
}

bool step_rounds_is_batch(const struct step_driver *driver)
{
	return driver != NULL && driver->program_count != 0 &&
	       driver->programs != NULL && driver->programs[0] != NULL &&
	       driver->programs[0]->kind == PROGRAM_BATCH;
}

result_status step_rounds_domain_attach(struct step_driver *driver,
					uint32_t vocabulary_size,
					uint32_t end_token_id)
{
	if (!step_rounds_is_batch(driver) || driver->sequence != 0 ||
	    driver->has_finished || driver->failure != STATUS_OK ||
	    vocabulary_size == 0 || vocabulary_size > INT32_MAX ||
	    end_token_id >= vocabulary_size ||
	    (driver->vocabulary_size != 0 &&
	     driver->vocabulary_size != vocabulary_size))
		return STATUS_INVALID_ARGUMENT;
	if (driver->round_domain_attached &&
	    (driver->vocabulary_size != vocabulary_size ||
	     driver->round_end_token_id != end_token_id))
		return STATUS_INVALID_ARGUMENT;
	driver->vocabulary_size = vocabulary_size;
	driver->round_end_token_id = end_token_id;
	driver->round_domain_attached = true;
	return STATUS_OK;
}

result_status step_batch_result_take(struct step_driver *driver,
				     const unsigned char *result_bytes,
				     uint64_t sequence,
				     const struct step_row_record *rows,
				     uint32_t row_count,
				     struct step_row_result *results_out)
{
	if (!step_rounds_is_batch(driver) || !driver->round_domain_attached ||
	    result_bytes == NULL || rows == NULL || results_out == NULL ||
	    row_count == 0 || row_count > STEP_BATCH_MAXIMUM_ROWS ||
	    !separate(results_out, (uint64_t)row_count * sizeof(*results_out),
		      rows, (uint64_t)row_count * sizeof(*rows)))
		return STATUS_INVALID_ARGUMENT;
	const uint32_t *run_graphs = NULL;
	const struct execution_program *const program =
		step_program_for_rows(driver, rows, row_count, &run_graphs);
	if (program == NULL || program->kind != PROGRAM_BATCH)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t capacity = program->limits.session_capacity;
	struct step_result header;
	struct step_row_result results[STEP_BATCH_MAXIMUM_ROWS];
	struct step_round rounds[STEP_BATCH_MAXIMUM_SESSIONS];
	if (capacity == 0 || capacity > STEP_BATCH_MAXIMUM_SESSIONS)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t slab_count =
		(program->row_count + STEP_BATCH_SLAB_ROWS - 1u) /
		STEP_BATCH_SLAB_ROWS;
	uint32_t total = 0;
	for (uint32_t slab = 0; slab < slab_count; slab++) {
		struct step_result part;
		const unsigned char *const area =
			result_bytes +
			(size_t)slab * STEP_BATCH_SLAB_RESULT_BYTES;
		memcpy(&part, area, sizeof(part));
		const uint32_t first = slab * STEP_BATCH_SLAB_ROWS;
		const uint32_t remaining =
			row_count > first ? row_count - first : 0u;
		const uint32_t count = remaining < STEP_BATCH_SLAB_ROWS ?
					       remaining :
					       STEP_BATCH_SLAB_ROWS;
		if (part.sequence != sequence || part.row_count != count ||
		    part.reserved_zero != 0)
			return STATUS_DEVICE_ERROR;
		if (count != 0)
			memcpy(&results[first], area + STEP_RESULT_BYTES,
			       (size_t)count * sizeof(results[0]));
		total += count;
	}
	if (total != row_count)
		return STATUS_DEVICE_ERROR;
	header = (struct step_result){ sequence, total, 0 };
	const bool drafts = step_drafts(rows, row_count);
	if (drafts) {
		const struct program_card_record *const card =
			&program->cards[program->result_card];
		memcpy(rounds,
		       result_bytes + card->round_offset - card->result_offset,
		       (size_t)capacity * sizeof(*rounds));
	}
	const struct step_round_description description = {
		.rows = rows,
		.results = results,
		.result_header = &header,
		.rounds = drafts ? rounds : NULL,
		.sequence = sequence,
		.row_count = row_count,
		.context_token_limit = program->limits.context_token_limit,
		.session_capacity = capacity,
		.vocabulary_size = driver->vocabulary_size,
		.draft_positions = program->limits.draft_positions,
		.end_token_id = driver->round_end_token_id,
		.round_capacity = capacity
	};
	struct step_round_table pending;
	const result_status status =
		step_rounds_validate(&description, &pending);
	if (status != STATUS_OK)
		return status;
	memcpy(results_out, results, (size_t)row_count * sizeof(*results_out));
	driver->last_rounds = pending;
	driver->last_round =
		(struct step_round){ .session_slot = STEP_NO_SESSION };
	if (pending.valid_mask != 0 &&
	    (pending.valid_mask & (pending.valid_mask - 1u)) == 0)
		driver->last_round = pending.rounds[(uint32_t)__builtin_ctz(
			pending.valid_mask)];
	return STATUS_OK;
}
