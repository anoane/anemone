/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step_experts.c: the step driver's host experts (step.h): their copies in page-locked memory, and a step's expert
 * pauses, which put the experts its jobs name into the cards' slots (model_seam.h). At an expert pause the job table
 * comes back into its own region of the block and goes up again from there, its experts from the second staging block,
 * which holds every host expert once; each slot remembers the expert it holds and the pause that took it last, and the
 * next slot to take on a card goes round them in turn.
 */
#include "anemone/step.h"

#include <inttypes.h>
#include <string.h>

#include "anemone/hooks.h"
#include "step_driver.h"
#include "step_validation.h"

#define OK STATUS_OK
#define THEN(call)        \
	(status != OK ?   \
		 status : \
		 (call)) /* the next call while every one before passed */
#define SLOT_PASSES 2u /* the slots that hold their expert; then the others' */
#define TAG_BYTES LINEAR_OFFSET_ALIGNMENT_BYTES

/* The slot of the card's slot_count slots that holds this host expert, or slot_count. */
static uint32_t slot_holding(const uint64_t *slots, uint32_t slot_count,
			     uint64_t expert)
{
	uint32_t slot = 0;
	while (slot < slot_count &&
	       slots[(uint64_t)slot * STEP_SLOT_FIELDS] != expert + 1u) {
		slot++;
	}
	return slot;
}

/* An expert pause (model_seam.h): the step's jobs come back; a first pass marks each slot that holds an expert a tag
 * names, the second gives each job of a tag that slot or else the next in turn that no job of the pause marked, into
 * which its expert goes up, and points the job there; then the jobs go up again. A job of a tag marks one slot at
 * most and there are no more jobs than slots, so an unmarked one is always left. */
MUST_CHECK result_status step_experts_pause(struct step_driver *driver,
					    uint32_t index, uint32_t row_count)
{
	const struct expert_pause *const pause =
		&driver->programs[0]->expert_pauses[index];
	const uint32_t card = pause->card, slot_count = pause->slot_count;
	const uint64_t job_count = (uint64_t)pause->jobs_per_row * row_count,
		       mark = ++driver->pause_mark;
	unsigned char *const jobs =
		driver->staging + driver->pause_staging[index];
	uint64_t *const slots =
		driver->slots +
		(uint64_t)driver->first_slot[card] * STEP_SLOT_FIELDS;
	uint32_t tags = 0;
	TIMER_BEGIN(step_expert_pause);
	result_status status =
		step_copy_begin(driver, card, pause->jobs_offset, jobs,
				job_count * EXPERT_JOB_BYTES, true);
	status = THEN(card_wait(driver, card));
	for (uint32_t pass = 0; pass < SLOT_PASSES && status == OK; pass++) {
		for (uint64_t job_index = 0;
		     job_index < job_count && status == OK; job_index++) {
			struct expert_job_record job;
			memcpy(&job, jobs + job_index * EXPERT_JOB_BYTES,
			       sizeof(job));
			const uint64_t tag =
				(job.gate_offset - pause->slots_offset) /
				TAG_BYTES; /* where it is one */
			if (job.row_count == 0 ||
			    (job.gate_offset < pause->slots_offset &&
			     job.up_offset < pause->slots_offset &&
			     job.down_offset < pause->slots_offset)) {
				continue; /* empty, or on the card */
			}
			if (job.gate_offset < pause->slots_offset ||
			    (job.gate_offset - pause->slots_offset) %
					    TAG_BYTES !=
				    0 ||
			    tag >= pause->host_expert_count ||
			    job.up_offset != job.gate_offset ||
			    job.down_offset != job.gate_offset) {
				LOG_ERROR("step %" PRIu64 ": job %" PRIu64
					  " of expert pause %" PRIu32
					  " names neither an "
					  "expert on card %" PRIu32
					  " nor one of the %" PRIu32
					  " in host memory",
					  driver->sequence, job_index, index,
					  card, pause->host_expert_count);
				status = STATUS_DEVICE_ERROR;
				break;
			}
			uint32_t slot =
				slot_holding(slots, slot_count,
					     pause->first_host_expert + tag);
			if (slot <
			    slot_count) { /* the first pass marks it, the second finds it marked */
				slots[(uint64_t)slot * STEP_SLOT_FIELDS + 1u] =
					mark;
				COUNTER_ADD(step_expert_slot_hits,
					    pass != 0 ? 1u : 0u);
			}
			if (pass == 0) {
				continue;
			}
			if (slot == slot_count) {
				slot = driver->next_slot[card];
				for (uint32_t tried = 1;
				     tried < slot_count &&
				     slots[(uint64_t)slot * STEP_SLOT_FIELDS +
					   1u] == mark;
				     tried++) {
					slot = (slot + 1u) % slot_count;
				}
				driver->next_slot[card] =
					(slot + 1u) % slot_count;
				if (slots[(uint64_t)slot * STEP_SLOT_FIELDS +
					  1u] ==
				    mark) { /* what create's rules rule out */
					LOG_ERROR("step %" PRIu64
						  ": expert pause %" PRIu32
						  " found none of the %" PRIu32
						  " slots "
						  "of card %" PRIu32 " free",
						  driver->sequence, index,
						  slot_count, card);
					status = STATUS_DEVICE_ERROR;
					break;
				}
				status = step_copy_begin(
					driver, card,
					pause->slots_offset +
						slot * pause->slot_byte_count,
					driver->expert_copies +
						driver->pause_copies[index] +
						tag * pause->expert_byte_count,
					pause->expert_byte_count, false);
				slots[(uint64_t)slot * STEP_SLOT_FIELDS] =
					status == OK ?
						pause->first_host_expert + tag +
							1u :
						0;
				slots[(uint64_t)slot * STEP_SLOT_FIELDS + 1u] =
					mark;
				COUNTER_ADD(step_expert_uploads,
					    status == OK ? 1u : 0u);
			}
			const uint64_t at = pause->slots_offset +
					    slot * pause->slot_byte_count;
			job.gate_offset = at + pause->matrix_offsets[0];
			job.up_offset = at + pause->matrix_offsets[1];
			job.down_offset =
				at +
				pause->matrix_offsets[EXPERT_PAUSE_MATRICES -
						      1u];
			memcpy(jobs + job_index * EXPERT_JOB_BYTES, &job,
			       sizeof(job));
			tags++;
		}
	}
	status = tags == 0 ? status :
			     THEN(step_copy_begin(
				     driver, card, pause->jobs_offset, jobs,
				     job_count * EXPERT_JOB_BYTES, false));
	TIMER_END(step_expert_pause);
	return status;
}

result_status step_host_experts_attach(struct step_driver *driver,
				       const void *host_arena,
				       uint64_t host_byte_count)
{
	const struct execution_program *const program =
		driver != NULL ? driver->programs[0] : NULL;
	uint64_t byte_count = 0;
	unsigned char *copies = NULL;
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	bool is_valid = program != NULL && driver->expert_copies == NULL &&
			(host_arena != NULL || program->host_expert_count == 0);
	for (uint32_t index = 0;
	     is_valid && index < program->expert_pause_count; index++) {
		const struct expert_pause *const pause =
			&program->expert_pauses[index];
		driver->pause_copies[index] = byte_count;
		for (uint32_t expert = 0;
		     is_valid && expert < pause->host_expert_count; expert++) {
			is_valid = step_range_fits(
				program->host_expert_offsets
					[pause->first_host_expert + expert],
				1, pause->expert_byte_count, host_byte_count);
		}
		is_valid = is_valid && pause->host_expert_count != 0 &&
			   pause->expert_byte_count <=
				   (UINT64_MAX - byte_count) /
					   pause->host_expert_count;
		byte_count =
			is_valid ?
				byte_count + pause->expert_byte_count *
						     pause->host_expert_count :
				byte_count;
	}
	if (!is_valid || byte_count > STEP_MOST_PINNED_EXPERT_BYTES) {
		LOG_ERROR(
			"step: host experts refused: no driver, attached before, an expert outside the host arena "
			"of %" PRIu64 " bytes, or more than %llu bytes of them",
			host_byte_count, STEP_MOST_PINNED_EXPERT_BYTES);
		return is_valid ? STATUS_OUT_OF_MEMORY :
				  STATUS_INVALID_ARGUMENT;
	}
	const result_status status =
		byte_count == 0 ?
			OK :
			driver->backend->staging_allocate(byte_count,
							  (void **)&copies);
	for (uint32_t index = 0;
	     copies != NULL && index < program->expert_pause_count; index++) {
		const struct expert_pause *const pause =
			&program->expert_pauses[index];
		for (uint32_t expert = 0; expert < pause->host_expert_count;
		     expert++) {
			memcpy(copies + driver->pause_copies[index] +
				       (uint64_t)expert *
					       pause->expert_byte_count,
			       (const unsigned char *)host_arena +
				       program->host_expert_offsets
					       [pause->first_host_expert +
						expert],
			       (size_t)pause->expert_byte_count);
		}
	}
	driver->expert_copies = status == OK ? copies : NULL;
	return status;
}
