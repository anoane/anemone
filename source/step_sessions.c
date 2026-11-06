/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step_sessions.c: step.h's snapshots of a session, and its mirror syncs. A snapshot is the program's session ranges
 * (model_seam.h) for a session's first tokens, each identity once from its first range, in identity order, moved
 * through a staging block of the driver's own (the session block, taken at the first snapshot, or at create where
 * front steps defer hand-offs) in pieces: a save takes a piece down, waits for its card and hands it to the caller; a
 * load takes it from the caller, puts it into every range of its identity and waits for those cards, so the block is
 * free again for the next piece. A save first syncs the session's mirrors (its first range may be a mirror), and a
 * load leaves them synced. A mirror sync moves the entries of the deferred hand-offs (step.h) of a range of positions
 * the same way: a piece down from the card they leave, once for every hand-off of that source, then up to each card.
 */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/exchange_records.h"
#include "anemone/hooks.h"
#include "anemone/model_layout.h"
#include "step_driver.h"

#define OK STATUS_OK
#define THEN(call)        \
	(status != OK ?   \
		 status : \
		 (call)) /* the next call while every one before passed */
#define IDENTITY_OFFSET 0xCBF29CE484222325ull /* FNV-1a, 64 bits */
#define IDENTITY_PRIME 0x100000001B3ull
#define IDENTITY_RELAXED_EXPERTS \
	0x4558505F52454C31ull /* EXP_REL1: routed experts now relax with the trunk */
#define IDENTITY_OFFICIAL_HC \
	0x48435F4F464631ull /* HC_OFF1: pinned HC reference shape */

/* The units of a range for token_count tokens. */
static uint64_t range_units(const struct session_range *range,
			    uint32_t token_count)
{
	const uint64_t wanted =
		range->unit_tokens == 0 ?
			range->unit_count :
			((uint64_t)token_count + range->unit_tokens - 1u) /
				range->unit_tokens;
	return wanted < range->unit_count ? wanted : range->unit_count;
}

uint64_t step_session_bytes(const struct step_driver *driver,
			    uint32_t token_count)
{
	const struct execution_program *const program = driver->programs[0];
	uint64_t byte_count = 0;
	for (uint32_t index = 0; index < program->session_range_count;
	     index++) {
		const struct session_range *const range =
			&program->session_ranges[index];
		byte_count += index == 0 || range[-1].identity !=
						      range->identity ?
				      range_units(range, token_count) *
					      range->unit_bytes :
				      0;
	}
	return byte_count;
}

/* A snapshot of token_count tokens of slot, through move: a load when is_load (see the top). */
MUST_CHECK static result_status
snapshot_move(struct step_driver *driver, uint32_t slot, uint32_t token_count,
	      step_bytes_function move, void *context, bool is_load)
{
	const struct execution_program *const program = driver->programs[0];
	const struct session_range *const ranges = program->session_ranges;
	result_status status = driver->session_block != NULL ?
				       OK :
				       driver->backend->staging_allocate(
					       STEP_SESSION_STAGING_BYTES,
					       (void **)&driver->session_block);
	for (uint32_t first = 0, last = 0;
	     first < program->session_range_count && status == OK;
	     first = last) {
		for (last = first + 1u;
		     last < program->session_range_count &&
		     ranges[last].identity == ranges[first].identity;
		     last++) {
		}
		const uint64_t total =
			range_units(&ranges[first], token_count) *
			ranges[first].unit_bytes;
		for (uint64_t done = 0, piece = 0; done < total && status == OK;
		     done += piece) {
			piece = total - done < STEP_SESSION_STAGING_BYTES ?
					total - done :
					STEP_SESSION_STAGING_BYTES;
			status = is_load && !move(context,
						  driver->session_block,
						  piece) ?
					 STATUS_INPUT_OUTPUT_ERROR :
					 OK;
			/* a load fills every range of the identity, a save reads its first */
			const uint32_t end = is_load ? last : first + 1u;
			for (uint32_t index = first; index < end; index++) {
				status = THEN(step_copy_begin(
					driver, ranges[index].card,
					ranges[index].offset +
						slot * ranges[index]
								.session_stride +
						done,
					driver->session_block, piece,
					!is_load));
			}
			for (uint32_t index = first; index < end; index++) {
				status = THEN(
					card_wait(driver, ranges[index].card));
			}
			status = !is_load && status == OK &&
						 !move(context,
						       driver->session_block,
						       piece) ?
					 STATUS_INPUT_OUTPUT_ERROR :
					 status;
		}
	}
	return status;
}

/* Whether two hand-offs carry the same entries of the same card. */
static bool is_same_source(const struct hand_off_record *hand_off,
			   const struct hand_off_record *other)
{
	return hand_off->from_card == other->from_card &&
	       hand_off->source_offset == other->source_offset &&
	       hand_off->session_stride == other->session_stride &&
	       hand_off->byte_count == other->byte_count &&
	       hand_off->entry_ratio == other->entry_ratio;
}

/* A piece of the entries of the deferred hand-offs of index's source, in the session block: up to each card they go to,
 * byte_count bytes at into of the session's range there, then their waits. */
MUST_CHECK static result_status piece_up(struct step_driver *driver,
					 uint32_t index, uint32_t session_slot,
					 uint64_t into, uint64_t byte_count)
{
	const struct execution_program *const program = driver->programs[0];
	const struct hand_off_record *const own = &program->hand_offs[index];
	result_status status = OK;
	for (uint32_t pass = 0; pass <= 1u; pass++) {
		for (uint32_t other = index;
		     other < program->hand_off_count && status == OK; other++) {
			const struct hand_off_record *const copy =
				&program->hand_offs[other];
			if (driver->deferred[other] == 0 ||
			    !is_same_source(copy, own)) {
				continue;
			}
			status =
				pass == 0 ?
					step_copy_begin(
						driver, copy->to_card,
						copy->destination_offset +
							session_slot *
								copy->session_stride +
							into,
						driver->session_block,
						byte_count, false) :
					card_wait(driver, copy->to_card);
		}
	}
	return status;
}

result_status step_mirror_sync(struct step_driver *driver,
			       uint32_t session_slot, uint32_t from,
			       uint32_t to)
{
	const struct execution_program *const program = driver->programs[0];
	result_status status = OK;
	for (uint32_t index = 0; index < program->hand_off_count; index++) {
		const struct hand_off_record *const own =
			&program->hand_offs[index];
		bool is_first = driver->deferred[index] != 0;
		for (uint32_t other = 0; other < index && is_first; other++) {
			is_first = driver->deferred[other] == 0 ||
				   !is_same_source(&program->hand_offs[other],
						   own);
		}
		if (!is_first) {
			continue;
		}
		/* positions p .. q emit the entries from p / ratio up to (q + 1) / ratio (model_seam.h) */
		const uint64_t piece =
			STEP_SESSION_STAGING_BYTES / own->byte_count;
		const uint64_t end = to / own->entry_ratio;
		for (uint64_t entry = from / own->entry_ratio;
		     entry < end && status == OK; entry += piece) {
			const uint64_t into = entry * own->byte_count,
				       byte_count = (end - entry < piece ?
							     end - entry :
							     piece) *
						    own->byte_count;
			status = step_copy_begin(
				driver, own->from_card,
				own->source_offset +
					session_slot * own->session_stride +
					into,
				driver->session_block, byte_count, true);
			status = THEN(card_wait(driver, own->from_card));
			status = THEN(piece_up(driver, index, session_slot,
					       into, byte_count));
		}
	}
	return status;
}

/* step.h's refusals of a snapshot; logged. */
MUST_CHECK static result_status snapshot_check(const struct step_driver *driver,
					       uint32_t slot,
					       uint32_t token_count,
					       step_bytes_function move)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	if (driver == NULL || move == NULL ||
	    slot >= driver->programs[0]->limits.session_capacity ||
	    token_count > driver->programs[0]->limits.context_token_limit ||
	    !step_pipeline_is_idle(driver)) {
		LOG_ERROR(
			"step: a snapshot refused: no driver or function, no such slot, a count beyond the context, "
			"or steps in flight");
		return STATUS_INVALID_ARGUMENT;
	}
	return OK;
}

result_status step_session_save(struct step_driver *driver,
				uint32_t session_slot, uint32_t token_count,
				step_bytes_function write, void *context)
{
	result_status status =
		snapshot_check(driver, session_slot, token_count, write);
	if (status == OK && token_count > driver->token_counts[session_slot]) {
		LOG_ERROR("step: a snapshot of %" PRIu32
			  " tokens refused: the slot holds fewer",
			  token_count);
		status = STATUS_INVALID_ARGUMENT;
	}
	if (status != OK) {
		return status;
	}
	/* the first range of an identity may be a mirror: the session's syncs first */
	const uint32_t from = driver->mirror_from[session_slot],
		       to = driver->mirror_to[session_slot];
	status = from < to ? step_mirror_sync(driver, session_slot, from, to) :
			     OK;
	if (status == OK) {
		driver->mirror_from[session_slot] = 0;
		driver->mirror_to[session_slot] = 0;
	}
	status = THEN(snapshot_move(driver, session_slot, token_count, write,
				    context, false));
	driver->failure = status == STATUS_DEVICE_ERROR ||
					  status == STATUS_TIMED_OUT ?
				  status :
				  driver->failure;
	return status;
}

result_status step_session_load(struct step_driver *driver,
				uint32_t session_slot,
				const uint32_t *token_ids, uint32_t token_count,
				step_bytes_function read, void *context)
{
	result_status status =
		token_ids == NULL && token_count != 0 ?
			STATUS_INVALID_ARGUMENT :
			snapshot_check(driver, session_slot, token_count, read);
	if (status == STATUS_INVALID_ARGUMENT && driver != NULL &&
	    token_ids == NULL) {
		LOG_ERROR("step: a snapshot refused: no tokens");
	}
	if (status != OK) {
		return status;
	}
	const uint64_t context_limit =
		driver->programs[0]->limits.context_token_limit;
	driver->token_counts[session_slot] =
		0; /* until the whole snapshot is in: a failed load keeps nothing */
	driver->mirror_from[session_slot] =
		0; /* a load puts every range of an identity: the mirrors lack nothing */
	driver->mirror_to[session_slot] = 0;
	status = snapshot_move(driver, session_slot, token_count, read, context,
			       true);
	if (status == OK && token_count != 0) {
		memcpy(driver->histories + session_slot * context_limit,
		       token_ids, token_count * sizeof(*token_ids));
		driver->token_counts[session_slot] = token_count;
	}
	driver->failure = status == STATUS_DEVICE_ERROR ||
					  status == STATUS_TIMED_OUT ?
				  status :
				  driver->failure;
	return status;
}

result_status step_session_discard(struct step_driver *driver,
				   uint32_t session_slot)
{
	if (driver == NULL ||
	    session_slot >= driver->programs[0]->limits.session_capacity ||
	    !step_pipeline_is_idle(driver)) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (driver->failure != OK) {
		return driver->failure;
	}
	/* A new cold row starts at zero; never advertise an earlier prefix of partially overwritten state. */
	driver->token_counts[session_slot] = 0;
	driver->next_positions[session_slot] = 0;
	driver->mirror_from[session_slot] = 0;
	driver->mirror_to[session_slot] = 0;
	driver->session_sequences[session_slot] = 0;
	if (session_slot < STEP_BATCH_MAXIMUM_SESSIONS) {
		driver->last_rounds.valid_mask &= ~(1u << session_slot);
		driver->last_rounds.rounds[session_slot] =
			(struct step_round){ .session_slot = STEP_NO_SESSION };
	}
	if (driver->last_round.session_slot == session_slot) {
		driver->last_round =
			(struct step_round){ .session_slot = STEP_NO_SESSION };
	}
	return OK;
}

MUST_CHECK static result_status clone_range(struct step_driver *driver,
					    uint32_t card, uint64_t source,
					    uint64_t destination,
					    uint64_t bytes)
{
	if (bytes == 0) {
		return OK;
	}
	const struct exchange_copy_arguments copy = {
		{ EXCHANGE_OWN_ARENA, source },
		{ EXCHANGE_OWN_ARENA, destination },
		bytes
	};
	struct kernel_launch launch = { .kernel = BACKEND_COPY,
					.argument_byte_count = sizeof(copy) };
	memcpy(launch.argument, &copy, sizeof(copy));
	driver->uncovered_work[card] = true;
	return driver->backend->launch_run(driver->cards[card], &launch, 1u);
}

result_status step_session_clone(struct step_driver *driver,
				 uint32_t source_slot,
				 uint32_t destination_slot,
				 uint32_t token_count, bool checkpoint,
				 uint64_t *byte_count_out)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	if (driver == NULL || byte_count_out == NULL ||
	    source_slot == destination_slot || token_count == 0 ||
	    source_slot >= driver->programs[0]->limits.session_capacity ||
	    destination_slot >= driver->programs[0]->limits.session_capacity ||
	    token_count > driver->programs[0]->limits.context_token_limit ||
	    token_count > driver->token_counts[source_slot] ||
	    token_count % MODEL_POOLED_RATIO != 0 ||
	    !step_pipeline_is_idle(driver) ||
	    (checkpoint && !step_checkpoint_supported(driver))) {
		return STATUS_INVALID_ARGUMENT;
	}
	/* Synchronizing the donor would alter its bytes/markers. A completed ordinary generation normally has no
	 * deferred mirror work; until a destination-only mirror repair is qualified, pending mirrors cold-miss. */
	if (driver->mirror_from[source_slot] < driver->mirror_to[source_slot]) {
		return STATUS_BUSY_VALUE;
	}
	const struct execution_program *const program = driver->programs[0];
	uint64_t bytes = 0;
	for (uint32_t index = 0; index < program->session_range_count;
	     index++) {
		const struct session_range *const range =
			&program->session_ranges[index];
		const uint64_t count =
			range_units(range, token_count) * range->unit_bytes;
		if (count > UINT64_MAX - bytes) {
			return STATUS_UNSUPPORTED;
		}
		bytes += count;
	}
	for (uint32_t index = 0;
	     checkpoint && index < program->checkpoint_count; index++) {
		const struct window_checkpoint *const range =
			&program->checkpoints[index];
		const uint64_t rows = token_count < range->window_rows ?
					      token_count :
					      range->window_rows;
		const uint64_t count = rows * range->row_bytes;
		if (count > UINT64_MAX - bytes) {
			return STATUS_UNSUPPORTED;
		}
		bytes += count;
	}
	result_status status = step_session_discard(driver, destination_slot);
	for (uint32_t index = 0;
	     index < program->session_range_count && status == OK; index++) {
		const struct session_range *const range =
			&program->session_ranges[index];
		status = clone_range(
			driver, range->card,
			range->offset +
				(uint64_t)source_slot * range->session_stride,
			range->offset + (uint64_t)destination_slot *
						range->session_stride,
			range_units(range, token_count) * range->unit_bytes);
	}
	for (uint32_t index = 0;
	     checkpoint && index < program->checkpoint_count && status == OK;
	     index++) {
		const struct window_checkpoint *const range =
			&program->checkpoints[index];
		const uint64_t rows = token_count < range->window_rows ?
					      token_count :
					      range->window_rows;
		status = clone_range(
			driver, range->card,
			range->checkpoint_offset + (uint64_t)source_slot *
							   range->window_rows *
							   range->row_bytes,
			range->checkpoint_offset + (uint64_t)destination_slot *
							   range->window_rows *
							   range->row_bytes,
			rows * range->row_bytes);
	}
	/* Completion is global even if a later launch fails after earlier cards queued their copies. */
	for (uint32_t card = 0; card < program->card_count; card++) {
		const result_status waited = card_wait(driver, card);
		/* A failed drain outranks an earlier logical refusal: the device cannot safely run a cold fallback. */
		if (status == OK || waited == STATUS_DEVICE_ERROR ||
		    waited == STATUS_TIMED_OUT) {
			status = waited;
		}
	}
	if (status == OK) {
		const size_t context =
			(size_t)program->limits.context_token_limit;
		memcpy(driver->histories + (size_t)destination_slot * context,
		       driver->histories + (size_t)source_slot * context,
		       (size_t)token_count * sizeof(*driver->histories));
		driver->token_counts[destination_slot] = token_count;
		driver->next_positions[destination_slot] = token_count;
		*byte_count_out = bytes;
	}
	driver->failure = status == STATUS_DEVICE_ERROR ||
					  status == STATUS_TIMED_OUT ?
				  status :
				  driver->failure;
	return status;
}

uint64_t step_session_identity(const struct step_driver *driver)
{
	const struct execution_program *const program = driver->programs[0];
	const uint32_t limits[] = { (uint32_t)program->limits.state_width,
				    (uint32_t)program->limits.linear_profile,
				    (uint32_t)program->limits.linear_order };
	uint64_t hash = IDENTITY_OFFSET;
	for (uint32_t index = 0; index < sizeof(limits) / sizeof(limits[0]);
	     index++) {
		hash = (hash ^ limits[index]) * IDENTITY_PRIME;
	}
	/* Keep existing reference snapshots usable; relaxed CMP bits have a distinct identity. */
	if (program->limits.cmp_order != CMP_ORDER_REFERENCE) {
		hash = (hash ^ (uint32_t)program->limits.cmp_order) *
		       IDENTITY_PRIME;
		hash = (hash ^ program->limits.row_capacity) * IDENTITY_PRIME;
	}
	/* Preserve every default0 identity; source HC shapes cannot reuse another recipe's state. */
	if (program->limits.official_hc_rows != 0) {
		hash = (hash ^ IDENTITY_OFFICIAL_HC) * IDENTITY_PRIME;
		hash = (hash ^ program->limits.official_hc_rows) *
		       IDENTITY_PRIME;
	}
	if (program->limits.official_reference != 0)
		hash = (hash ^ UINT32_MAX) * IDENTITY_PRIME;
	for (uint32_t index = 0; index < program->session_range_count;
	     index++) {
		const struct session_range *const range =
			&program->session_ranges[index];
		if (index == 0 || range[-1].identity != range->identity) {
			hash = (hash ^ range->identity) * IDENTITY_PRIME;
			hash = (hash ^ range->unit_bytes) * IDENTITY_PRIME;
			hash = (hash ^ range->unit_tokens) * IDENTITY_PRIME;
		}
	}
	/* Reject relaxed snapshots from the earlier trunk-only arithmetic. */
	hash = program->limits.cmp_order == CMP_ORDER_REFERENCE ?
		       hash :
		       (hash ^ IDENTITY_RELAXED_EXPERTS) * IDENTITY_PRIME;
	return driver->companion_segments == NULL ?
		       hash :
		       (hash ^ driver->companion_identity) * IDENTITY_PRIME;
}
