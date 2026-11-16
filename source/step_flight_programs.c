/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Immutable transient-plane alternatives selected by the pipeline's owned ring slot. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "step_driver.h"
#include "step_validation.h"

/* the copied table holds the programs, then their image programs */
#define PROGRAM_FORMS 2u

static bool same_program(const struct execution_program *candidate,
			 const struct execution_program *base)
{
	if (candidate == NULL || candidate->kind != base->kind ||
	    candidate->row_count != base->row_count ||
	    candidate->cone_rows != base->cone_rows ||
	    candidate->lease_count != base->lease_count ||
	    (base->lease_count != 0 &&
	     memcmp(candidate->leases, base->leases,
		    (size_t)base->lease_count * sizeof(*base->leases)) != 0) ||
	    candidate->segment_count != base->segment_count ||
	    memcmp(candidate->segments, base->segments,
		   (size_t)base->segment_count * sizeof(*base->segments)) != 0)
		return false;
	for (uint32_t card = 0; card < base->card_count; card++) {
		const struct program_card_record *const own =
			&candidate->cards[card];
		const struct program_card_record *const original =
			&base->cards[card];
		if (own->launch_count != original->launch_count ||
		    own->view_count != original->view_count ||
		    (own->view_count != 0 &&
		     memcmp(own->views, original->views,
			    (size_t)own->view_count * sizeof(*own->views)) !=
			     0))
			return false;
	}
	return true;
}

result_status
step_flight_programs_attach(struct step_driver *driver,
			    const struct step_flight_programs *banks,
			    uint32_t bank_count, uint32_t program_count,
			    const struct visit_lease *owner_guard)
{
	if (driver == NULL || banks == NULL || owner_guard == NULL ||
	    bank_count == 0 || bank_count > STEP_PIPELINE_MOST ||
	    program_count == 0 || program_count != driver->program_count ||
	    driver->sequence != 0 || driver->pipeline != NULL ||
	    driver->graphs_attempted || driver->flight_bank_count != 0)
		return STATUS_INVALID_ARGUMENT;
	if (driver->failure != STATUS_OK)
		return driver->failure;
	if (driver->companion_segments != NULL)
		return STATUS_UNSUPPORTED;
	const struct execution_program *const first = driver->programs[0];
	if (owner_guard->card >= first->card_count ||
	    owner_guard->first_segment >= owner_guard->last_segment ||
	    owner_guard->last_segment >= first->segment_count ||
	    first->segments[owner_guard->first_segment].card !=
		    owner_guard->card ||
	    first->segments[owner_guard->last_segment].card !=
		    owner_guard->card)
		return STATUS_INVALID_ARGUMENT;
	bool leaves_card = false;
	for (uint32_t segment = owner_guard->first_segment;
	     segment < owner_guard->last_segment; segment++)
		leaves_card = leaves_card || first->segments[segment].card !=
						     owner_guard->card;
	if (!leaves_card)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t session = 0; session < first->limits.session_capacity;
	     session++)
		if (driver->token_counts[session] != 0 ||
		    driver->next_positions[session] != 0)
			return STATUS_INVALID_ARGUMENT;
	bool unsupported = false;
	for (uint32_t bank = 0; bank < bank_count; bank++) {
		if (banks[bank].programs == NULL ||
		    ((banks[bank].image_programs != NULL) !=
		     (driver->image_programs != NULL)))
			return STATUS_INVALID_ARGUMENT;
		for (uint32_t index = 0; index < program_count; index++) {
			const struct execution_program *const candidate =
				banks[bank].programs[index];
			const struct execution_program *const base =
				driver->programs[index];
			if (base->kind == PROGRAM_BATCH ||
			    base->expert_pause_count != 0)
				return STATUS_UNSUPPORTED;
			if ((bank == 0 && candidate != base) ||
			    step_program_problem(candidate, first,
						 driver->cards, driver->engram,
						 &unsupported) != NULL ||
			    !same_program(candidate, base))
				return unsupported ? STATUS_UNSUPPORTED :
						     STATUS_INVALID_ARGUMENT;
			/* both image tables or neither (refused above) */
			if (banks[bank].image_programs != NULL &&
			    driver->image_programs != NULL) {
				const struct execution_program *const image =
					banks[bank].image_programs[index];
				const struct execution_program *const original =
					driver->image_programs[index];
				if ((bank == 0 && image != original) ||
				    step_program_problem(
					    image, first, driver->cards,
					    driver->engram,
					    &unsupported) != NULL ||
				    !same_program(image, original))
					return unsupported ?
						       STATUS_UNSUPPORTED :
						       STATUS_INVALID_ARGUMENT;
			}
		}
	}
	const size_t count = (size_t)bank_count * program_count;
	if (program_count != 0 && count / program_count != bank_count)
		return STATUS_OUT_OF_MEMORY;
	if (count >
	    SIZE_MAX / (PROGRAM_FORMS * sizeof(*driver->flight_programs)))
		return STATUS_OUT_OF_MEMORY;
	const struct execution_program **const copied =
		FAULT_POINT(step_driver_memory) ?
			NULL :
			(const struct execution_program **)calloc(
				count * PROGRAM_FORMS, sizeof(*copied));
	if (copied == NULL)
		return STATUS_OUT_OF_MEMORY;
	for (uint32_t bank = 0; bank < bank_count; bank++) {
		memcpy((void *)(copied + (size_t)bank * program_count),
		       (const void *)banks[bank].programs,
		       (size_t)program_count * sizeof(*copied));
		if (banks[bank].image_programs != NULL &&
		    driver->image_programs != NULL) /* both or neither */
			memcpy((void *)(copied + count +
					(size_t)bank * program_count),
			       (const void *)banks[bank].image_programs,
			       (size_t)program_count * sizeof(*copied));
	}
	driver->flight_programs = copied;
	driver->flight_image_programs =
		driver->image_programs != NULL ? copied + count : NULL;
	driver->flight_owner_guard = *owner_guard;
	driver->flight_bank_count = bank_count;
	return STATUS_OK;
}

const struct execution_program *
step_flight_program(const struct step_driver *driver,
		    const struct execution_program *base, uint32_t bank,
		    const uint32_t **run_graphs_out)
{
	if (driver->flight_bank_count == 0 || base == NULL)
		return base;
	if (bank >= driver->flight_bank_count)
		return NULL;
	size_t graph_base = (size_t)bank * driver->flight_graph_stride;
	for (uint32_t index = 0; index < driver->program_count; index++) {
		const size_t at = (size_t)bank * driver->program_count + index;
		if (base == driver->programs[index]) {
			*run_graphs_out =
				driver->run_graphs != NULL ?
					driver->run_graphs + graph_base :
					NULL;
			return driver->flight_programs[at];
		}
		if (driver->image_programs != NULL &&
		    base == driver->image_programs[index]) {
			*run_graphs_out = NULL;
			return driver->flight_image_programs[at];
		}
		graph_base += driver->programs[index]->segment_count;
	}
	return NULL;
}
