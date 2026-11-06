/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image programs share the base driver's state and staging, with immutable per-step selection. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "step_driver.h"
#include "step_runs.h"
#include "anemone/step_images.h"
#include "step_validation.h"

static bool program_views_match(const struct execution_program *image,
				const struct execution_program *text)
{
	if (image->lease_count != text->lease_count ||
	    (text->lease_count != 0 &&
	     memcmp(image->leases, text->leases,
		    (size_t)text->lease_count * sizeof(*text->leases)) != 0)) {
		return false;
	}
	for (uint32_t card = 0; card < text->card_count; card++) {
		const struct program_card_record *const base =
			&text->cards[card];
		const struct program_card_record *const alternative =
			&image->cards[card];
		if (alternative->view_count != base->view_count ||
		    (base->view_count != 0 &&
		     memcmp(alternative->views, base->views,
			    (size_t)base->view_count * sizeof(*base->views)) !=
			     0)) {
			return false;
		}
	}
	return true;
}

result_status
step_image_programs_attach(struct step_driver *driver,
			   const struct execution_program *const *programs,
			   uint32_t program_count)
{
	if (driver == NULL || programs == NULL || program_count == 0 ||
	    program_count != driver->program_count ||
	    driver->image_programs != NULL || driver->sequence != 0 ||
	    driver->pipeline != NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (driver->failure != STATUS_OK) {
		return driver->failure;
	}
	if (driver->companion_segments != NULL)
		return STATUS_UNSUPPORTED;
	bool unsupported = false;
	for (uint32_t index = 0; index < program_count; index++) {
		const struct execution_program *const image = programs[index];
		const struct execution_program *const text =
			driver->programs[index];
		const char *const problem = step_program_problem(
			image, driver->programs[0], driver->cards,
			driver->engram, &unsupported);
		if (problem != NULL || image->row_count != text->row_count ||
		    image->kind != text->kind ||
		    image->cone_rows != text->cone_rows ||
		    !program_views_match(image, text)) {
			LOG_ERROR(
				"image step programs: incompatible base and image program metadata");
			return unsupported ? STATUS_UNSUPPORTED :
					     STATUS_INVALID_ARGUMENT;
		}
	}
	const struct execution_program **const prepared =
		FAULT_POINT(step_driver_memory) ?
			NULL :
			(const struct execution_program **)calloc(
				program_count, sizeof(*prepared));
	if (prepared == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	memcpy((void *)prepared, (const void *)programs,
	       (size_t)program_count * sizeof(*prepared));
	driver->image_programs = prepared;
	return STATUS_OK;
}

const struct execution_program *
step_program_for_rows(const struct step_driver *driver,
		      const struct step_row_record *rows, uint32_t row_count,
		      const uint32_t **run_graphs_out)
{
	if (driver == NULL || rows == NULL || row_count == 0 ||
	    row_count > driver->largest_row_count || run_graphs_out == NULL) {
		return NULL;
	}
	const bool front = step_is_front(rows, row_count);
	const bool padded = (rows[0].flags & STEP_ROW_PADDED) != 0;
	const uint32_t capacity =
		padded ? step_chunk_rows(driver,
					 front ? PROGRAM_WIDE : PROGRAM_FULL,
					 UINT32_MAX) :
			 row_count;
	if (capacity == 0 || row_count > capacity) {
		*run_graphs_out = NULL;
		return NULL;
	}
	const struct execution_program *base =
		step_program_of(driver, capacity, front, run_graphs_out);
	/* A FULL program omits the head and cannot verify a BATCH owner. Keep legacy selection unchanged. */
	bool draws = false;
	uint32_t drafts = 0;
	for (uint32_t row = 0; row < row_count; row++) {
		draws = draws || (rows[row].flags & STEP_ROW_LOGITS) != 0;
		drafts += (rows[row].flags & STEP_ROW_DRAFTS) != 0 ? 1u : 0u;
	}
	if (draws && step_rounds_is_batch(driver)) {
		base = NULL;
		*run_graphs_out = NULL;
		const uint32_t verifier_rows =
			drafts *
			(driver->programs[0]->limits.draft_positions + 1u);
		/* attached verifiers: the rows alone, the program's verifiers the drafting sessions */
		const uint32_t needed_rows =
			verifier_rows > row_count && driver->batch_verifiers == NULL ? verifier_rows : row_count;
		size_t graph_base = 0;
		uint32_t base_verifiers = 0; /* of programs of equal rows, the fewest verifiers holding the drafts */
		for (uint32_t index = 0; index < driver->program_count;
		     index++) {
			const struct execution_program *const candidate =
				driver->programs[index];
			const uint32_t verifiers = driver->batch_verifiers != NULL ? driver->batch_verifiers[index] : 0;
			if (candidate->kind == PROGRAM_BATCH &&
			    candidate->row_count >= needed_rows &&
			    (driver->batch_verifiers == NULL ||
			     verifiers >= drafts) &&
			    (base == NULL ||
			     candidate->row_count < base->row_count ||
			     (candidate->row_count == base->row_count &&
			      verifiers < base_verifiers))) {
				base = candidate;
				base_verifiers = verifiers;
				*run_graphs_out = driver->run_graphs != NULL ?
							  driver->run_graphs +
								  graph_base :
							  NULL;
			}
			graph_base += candidate->segment_count;
		}
	}
	/* A headless exact-size FULL chunk avoids preparing unused BATCH heads. */
	if (!draws && !step_is_front(rows, row_count) &&
	    step_rounds_is_batch(driver)) {
		size_t graph_base = 0;
		for (uint32_t index = 0; index < driver->program_count;
		     index++) {
			const struct execution_program *const candidate =
				driver->programs[index];
			if (candidate->kind == PROGRAM_FULL &&
			    candidate->row_count == row_count) {
				base = candidate;
				*run_graphs_out = driver->run_graphs != NULL ?
							  driver->run_graphs +
								  graph_base :
							  NULL;
				break;
			}
			graph_base += candidate->segment_count;
		}
	}
	bool has_images = false;
	for (uint32_t row = 0; row < row_count; row++) {
		has_images = has_images ||
			     (rows[row].flags & STEP_ROW_IMAGE) != 0;
	}
	if (!has_images || base == NULL) {
		return base;
	}
	/* Graph storage belongs solely to base programs. Image variants use their checked eager launch list. */
	*run_graphs_out = NULL;
	for (uint32_t index = 0;
	     driver->image_programs != NULL && index < driver->program_count;
	     index++) {
		if (driver->programs[index] == base) {
			return driver->image_programs[index];
		}
	}
	return NULL;
}

result_status step_batch_verifiers_attach(struct step_driver *driver,
					  const uint32_t *verifiers,
					  uint32_t program_count)
{
	if (driver == NULL || verifiers == NULL ||
	    driver->batch_verifiers != NULL ||
	    program_count != driver->program_count || driver->sequence != 0 ||
	    !step_rounds_is_batch(driver))
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t index = 0; index < program_count; index++) {
		const struct execution_program *const program =
			driver->programs[index];
		if ((program->kind == PROGRAM_BATCH) !=
			    (verifiers[index] != 0) ||
		    verifiers[index] > program->limits.session_capacity)
			return STATUS_INVALID_ARGUMENT;
		/* BATCH programs of one row count: other verifiers */
		for (uint32_t other = 0; other < index; other++) {
			const struct execution_program *const earlier =
				driver->programs[other];
			if (program->kind == PROGRAM_BATCH &&
			    earlier->kind == PROGRAM_BATCH &&
			    earlier->row_count == program->row_count &&
			    verifiers[other] == verifiers[index])
				return STATUS_INVALID_ARGUMENT;
		}
	}
	uint32_t *const copy = calloc(program_count, sizeof(*copy));
	if (copy == NULL)
		return STATUS_OUT_OF_MEMORY;
	memcpy(copy, verifiers, program_count * sizeof(*copy));
	driver->batch_verifiers = copy;
	return STATUS_OK;
}

/* A BATCH program's drafting sessions: the attached count, else the rule's (rows / 6). */
static uint32_t verifiers_of(const struct step_driver *driver, uint32_t index)
{
	const struct execution_program *const program = driver->programs[index];
	return driver->batch_verifiers != NULL ? driver->batch_verifiers[index] :
						 program->row_count / (program->limits.draft_positions + 1u);
}

uint32_t step_batch_verifiers(const struct step_driver *driver, uint32_t rows)
{
	uint32_t fewest = 0;
	for (uint32_t index = 0; driver != NULL && index < driver->program_count; index++) {
		const struct execution_program *const program =
			driver->programs[index];
		if (program->kind == PROGRAM_BATCH &&
		    program->row_count == rows &&
		    (fewest == 0 || verifiers_of(driver, index) < fewest))
			fewest = verifiers_of(driver, index);
	}
	return fewest;
}

uint32_t step_batch_layout(const struct step_driver *driver, uint32_t *rows_out,
			   uint32_t *verifiers_out, uint32_t most)
{
	uint32_t count = 0;
	for (uint32_t index = 0; driver != NULL && rows_out != NULL && verifiers_out != NULL &&
				 index < driver->program_count;
	     index++) {
		const struct execution_program *const program =
			driver->programs[index];
		if (program->kind != PROGRAM_BATCH)
			continue;
		if (count == most)
			return 0;
		const uint32_t rows = program->row_count, verifiers = verifiers_of(driver, index);
		uint32_t at = count++; /* insertion: fewest rows, then fewest verifiers, first */
		for (; at > 0 && (rows_out[at - 1u] > rows ||
				  (rows_out[at - 1u] == rows && verifiers_out[at - 1u] > verifiers));
		     at--) {
			rows_out[at] = rows_out[at - 1u];
			verifiers_out[at] = verifiers_out[at - 1u];
		}
		rows_out[at] = rows;
		verifiers_out[at] = verifiers;
	}
	return count;
}
