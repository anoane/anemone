/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Prompt-end windows: same-card copies of the rows preceding an even kept boundary. No host staging or replay. */
#include <string.h>

#include "anemone/exchange_records.h"
#include "anemone/model_layout.h"
#include "step_driver.h"

#define OK STATUS_OK
#define WRAPPED_PIECES 2u

bool step_checkpoint_supported(const struct step_driver *driver)
{
	return driver != NULL && driver->programs[0]->checkpoint_count != 0;
}

MUST_CHECK static result_status checkpoint_move(struct step_driver *driver,
						uint32_t slot,
						uint32_t boundary,
						bool is_restore)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	/* Finish retires every card's last task, including upload-only closing visits, after its bounded wait.
	 * Global idle therefore gives the submitter all cards and their copy bookkeeping until it submits again. */
	if (!step_checkpoint_supported(driver) ||
	    slot >= driver->programs[0]->limits.session_capacity ||
	    boundary > driver->programs[0]->limits.context_token_limit ||
	    boundary > driver->token_counts[slot] ||
	    boundary % MODEL_POOLED_RATIO != 0 ||
	    !step_pipeline_is_idle(driver)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct execution_program *const program = driver->programs[0];
	result_status status = OK;
	for (uint32_t index = 0;
	     index < program->checkpoint_count && status == OK; index++) {
		const struct window_checkpoint *const range =
			&program->checkpoints[index];
		const uint32_t rows = boundary < range->window_rows ?
					      boundary :
					      range->window_rows;
		const uint32_t first = (boundary - rows) % range->ring_rows;
		const uint32_t until_wrap = range->ring_rows - first;
		const uint32_t first_rows = rows < until_wrap ? rows :
								until_wrap;
		struct kernel_launch launches[WRAPPED_PIECES] = { 0 };
		uint32_t count = 0;
		for (uint32_t done = 0;
		     done < rows && count < WRAPPED_PIECES;) {
			const uint32_t piece = done == 0 ? first_rows :
							   rows - done;
			const uint64_t ring =
				range->ring_offset +
				(uint64_t)slot * range->ring_session_stride +
				(uint64_t)(done == 0 ? first : 0u) *
					range->row_bytes;
			const uint64_t saved =
				range->checkpoint_offset +
				((uint64_t)slot * range->window_rows + done) *
					range->row_bytes;
			const struct exchange_copy_arguments copy = {
				{ EXCHANGE_OWN_ARENA,
				  is_restore ? saved : ring },
				{ EXCHANGE_OWN_ARENA,
				  is_restore ? ring : saved },
				(uint64_t)piece * range->row_bytes
			};
			struct kernel_launch *const launch = &launches[count++];
			launch->kernel = BACKEND_COPY;
			launch->argument_byte_count = sizeof(copy);
			memcpy(launch->argument, &copy, sizeof(copy));
			done += piece;
		}
		if (count != 0) {
			driver->uncovered_work[range->card] = true;
			status = driver->backend->launch_run(
				driver->cards[range->card], launches, count);
		}
	}
	/* Wait every card even after one fails: every successful copy must have retired before returning. */
	for (uint32_t card = 0; card < program->card_count; card++) {
		const result_status waited = card_wait(driver, card);
		status = status == OK ? waited : status;
	}
	driver->failure = status == STATUS_DEVICE_ERROR ||
					  status == STATUS_TIMED_OUT ?
				  status :
				  driver->failure;
	return status;
}

result_status step_checkpoint_save(struct step_driver *driver,
				   uint32_t session_slot, uint32_t boundary)
{
	return checkpoint_move(driver, session_slot, boundary, false);
}

result_status step_checkpoint_restore(struct step_driver *driver,
				      uint32_t session_slot, uint32_t boundary)
{
	return checkpoint_move(driver, session_slot, boundary, true);
}
