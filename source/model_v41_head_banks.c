/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Immutable startup copies of the target-head records; bank0 and non-THIN launch tables are shared. */
#include "model_v41.h"

#include <stdlib.h>
#include <string.h>

#include "anemone/head_offload.h"
#include "anemone/hooks.h"
#include "anemone/step.h"

#define HEAD_PATCH_RECORDS (V41_HEAD_SAMPLE * V41_HEAD_RANKS + 1u)
#define OK STATUS_OK
#define THEN(call) (status != OK ? status : (call))
/* the tables hold the programs, then their image programs */
#define PROGRAM_FORMS 2u

struct bank_program {
	struct execution_program program;
	struct program_card_record cards[V41_HEAD_CARDS];
	struct kernel_launch *launches[V41_HEAD_CARDS];
};

/* Peak retained heap: sizeof(this) + 2 * entries * sizeof(pointer) +, for each nonzero bank and THIN
 * ordinary/image program, sizeof(bank_program) + sum(card.launch_count * sizeof(kernel_launch)).
 * Bank0 and every lease/segment table are borrowed. The driver separately owns 2*B*P pointers, B*S graph
 * handles and B*ceil(session_capacity/64) owner words; no allocation occurs on the token path. */
struct v41_head_program_banks {
	struct step_flight_programs tables[V41_HEAD_BANKS];
	struct visit_lease owner_guard;
	const struct execution_program **pointers;
	struct bank_program **owned;
	size_t entries;
};

static void program_free(struct bank_program *made)
{
	if (made == NULL)
		return;
	for (uint32_t card = 0; card < V41_HEAD_CARDS; card++)
		free(made->launches[card]);
	free(made);
}

void v41_original_head_banks_destroy(struct v41_head_program_banks *banks)
{
	if (banks == NULL)
		return;
	for (size_t index = 0; banks->owned != NULL && index < banks->entries;
	     index++)
		program_free(banks->owned[index]);
	free((void *)banks->owned);
	free((void *)banks->pointers);
	free(banks);
}

struct v41_head_bank_memory
v41_original_head_banks_memory(const struct v41_head_program_banks *banks)
{
	struct v41_head_bank_memory memory = { 0 };
	if (banks == NULL)
		return memory;
	memory.program_metadata_bytes = sizeof(struct bank_program);
	memory.heap_bytes =
		sizeof(*banks) + banks->entries * (sizeof(*banks->pointers) +
						   sizeof(*banks->owned));
	while (memory.bank_count < V41_HEAD_BANKS &&
	       banks->tables[memory.bank_count].programs != NULL)
		memory.bank_count++;
	if (memory.bank_count == 0)
		return (struct v41_head_bank_memory){ 0 };
	memory.program_count =
		(uint32_t)(banks->entries /
			   (size_t)(PROGRAM_FORMS * memory.bank_count));
	for (size_t index = 0; index < banks->entries; index++) {
		const struct bank_program *const program = banks->owned[index];
		if (program == NULL)
			continue;
		memory.heap_bytes += sizeof(*program);
		for (uint32_t card = 0; card < V41_HEAD_CARDS; card++)
			memory.heap_bytes +=
				(size_t)program->cards[card].launch_count *
				sizeof(struct kernel_launch);
	}
	for (uint32_t image = 0;
	     image <
	     (banks->tables[0].image_programs != NULL ? PROGRAM_FORMS : 1u);
	     image++) {
		const struct execution_program *const *programs =
			image == 0 ? banks->tables[0].programs :
				     banks->tables[0].image_programs;
		for (uint32_t index = 0; index < memory.program_count;
		     index++) {
			const struct execution_program *const program =
				programs[index];
			if (program->kind != PROGRAM_THIN)
				continue;
			memory.thin_programs++;
			memory.source_lease_count += program->lease_count;
			for (uint32_t card = 0; card < V41_HEAD_CARDS; card++)
				memory.source_launch_bytes +=
					(size_t)program->cards[card]
						.launch_count *
					sizeof(struct kernel_launch);
		}
	}
	return memory;
}

const struct step_flight_programs *
v41_original_head_bank_tables(const struct v41_head_program_banks *banks)
{
	return banks != NULL ? banks->tables : NULL;
}

const struct visit_lease *
v41_original_head_owner_guard(const struct v41_head_program_banks *banks)
{
	return banks != NULL ? &banks->owner_guard : NULL;
}

/* One full launch match, confined to the target head's NO_LAYER segments. */
MUST_CHECK static result_status
replace_one(struct bank_program *made, uint32_t card,
	    const struct kernel_launch *old,
	    const struct kernel_launch *replacement)
{
	uint32_t matches = 0;
	for (uint32_t index = 0; index < made->cards[card].launch_count;
	     index++) {
		if (memcmp(&made->launches[card][index], old, sizeof(*old)) !=
		    0)
			continue;
		bool in_head = false;
		for (uint32_t segment = 0;
		     segment < made->program.draft_segment_first; segment++) {
			const struct segment_record *const part =
				&made->program.segments[segment];
			in_head = in_head || (part->card == card &&
					      part->layer == SEGMENT_NO_LAYER &&
					      index >= part->first_launch &&
					      index - part->first_launch <
						      part->launch_count);
		}
		if (!in_head)
			return STATUS_INVALID_ARGUMENT;
		made->launches[card][index] = *replacement;
		matches++;
	}
	return matches == 1u ? OK : STATUS_INVALID_ARGUMENT;
}

MUST_CHECK static result_status
program_make(const struct model_plan_record *plan,
	     const struct backend_operations *backend,
	     struct device_card *const *cards,
	     const struct execution_program *source, uint32_t bank,
	     struct bank_program **out)
{
	struct bank_program *const made =
		FAULT_POINT(model_v41_program_memory) ?
			NULL :
			calloc(1, sizeof(*made));
	if (made == NULL)
		return STATUS_OUT_OF_MEMORY;
	made->program = *source;
	made->program.cards = made->cards;
	result_status status = OK;
	for (uint32_t card = 0; card < V41_HEAD_CARDS && status == OK; card++) {
		made->cards[card] = source->cards[card];
		made->launches[card] =
			FAULT_POINT(model_v41_program_memory) ?
				NULL :
				calloc(source->cards[card].launch_count,
				       sizeof(struct kernel_launch));
		if (made->launches[card] == NULL) {
			status = STATUS_OUT_OF_MEMORY;
			break;
		}
		memcpy(made->launches[card], source->cards[card].launches,
		       (size_t)source->cards[card].launch_count *
			       sizeof(struct kernel_launch));
		made->cards[card].launches = made->launches[card];
	}
	uint32_t patched = 0;
	for (uint32_t card = 0; card < V41_HEAD_CARDS && status == OK; card++) {
		uint32_t copy_kernel = BACKEND_NO_KERNEL, copies = 0;
		status = backend->kernel_prepare(cards[card],
						 KERNEL_FILE_HEAD_OFFLOAD,
						 KERNEL_HEAD_OFFLOAD_COPY,
						 HEAD_OFFLOAD_ARGUMENT_BYTES,
						 &copy_kernel);
		for (uint32_t index = 0;
		     index < made->cards[card].launch_count && status == OK;
		     index++)
			copies += made->launches[card][index].kernel ==
						  copy_kernel ?
					  1u :
					  0u;
		if (status == OK &&
		    copies != (card == V41_HEAD_RANKS ? V41_HEAD_RANKS : 1u))
			status = STATUS_INVALID_ARGUMENT;
	}
	for (uint32_t rank = 0; rank < V41_HEAD_RANKS && status == OK; rank++) {
		struct kernel_launch old, replacement;
		for (enum v41_head_launch_kind kind = V41_HEAD_BROADCAST;
		     kind < V41_HEAD_SAMPLE && status == OK; kind++) {
			const uint32_t card = kind == V41_HEAD_BROADCAST ?
						      V41_HEAD_RANKS :
						      rank;
			status = v41_original_head_launch(plan, backend, cards,
							  kind, rank, 0,
							  source->row_count,
							  &old);
			status = THEN(v41_original_head_launch(
				plan, backend, cards, kind, rank, bank,
				source->row_count, &replacement));
			status = THEN(
				replace_one(made, card, &old, &replacement));
			patched += status == OK ? 1u : 0u;
		}
	}
	struct kernel_launch old, replacement;
	status = THEN(v41_original_head_launch(plan, backend, cards,
					       V41_HEAD_SAMPLE, 0, 0,
					       source->row_count, &old));
	status = THEN(v41_original_head_launch(
		plan, backend, cards, V41_HEAD_SAMPLE, 0, bank,
		source->row_count, &replacement));
	status = THEN(replace_one(made, V41_HEAD_RANKS, &old, &replacement));
	patched += status == OK ? 1u : 0u;
	if (status != OK || patched != HEAD_PATCH_RECORDS) {
		program_free(made);
		return status != OK ? status : STATUS_INVALID_ARGUMENT;
	}
	*out = made;
	return OK;
}

result_status v41_original_head_banks_create(
	const struct model_plan_record *plan,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct execution_program *const *programs,
	const struct execution_program *const *image_programs,
	uint32_t program_count, struct v41_head_program_banks **banks_out)
{
	if (plan == NULL || backend == NULL ||
	    backend->kernel_prepare == NULL || cards == NULL ||
	    programs == NULL || program_count == 0 || banks_out == NULL ||
	    !plan->original_head.enabled ||
	    plan->original_head.bank_count == 0 ||
	    plan->original_head.bank_count > V41_HEAD_BANKS ||
	    programs[0] == NULL || programs[0]->kind != PROGRAM_THIN ||
	    programs[0]->card_count != V41_HEAD_CARDS ||
	    programs[0]->segments == NULL || programs[0]->segment_count == 0)
		return STATUS_INVALID_ARGUMENT;
	struct v41_head_program_banks *const made =
		FAULT_POINT(model_v41_program_memory) ?
			NULL :
			calloc(1, sizeof(*made));
	if (made == NULL)
		return STATUS_OUT_OF_MEMORY;
	const size_t count =
		(size_t)plan->original_head.bank_count * program_count;
	if (count / program_count != plan->original_head.bank_count ||
	    count > SIZE_MAX / (PROGRAM_FORMS * sizeof(*made->pointers))) {
		free(made);
		return STATUS_OUT_OF_MEMORY;
	}
	made->entries = count * PROGRAM_FORMS;
	made->pointers =
		FAULT_POINT(model_v41_program_memory) ?
			NULL :
			(const struct execution_program **)calloc(
				made->entries, sizeof(*made->pointers));
	made->owned = FAULT_POINT(model_v41_program_memory) ?
			      NULL :
			      (struct bank_program **)calloc(
				      made->entries, sizeof(*made->owned));
	result_status status = made->pointers != NULL && made->owned != NULL ?
				       OK :
				       STATUS_OUT_OF_MEMORY;
	uint32_t first = 0;
	while (first < programs[0]->segment_count &&
	       programs[0]->segments[first].card != V41_HEAD_RANKS)
		first++;
	made->owner_guard =
		(struct visit_lease){ V41_HEAD_RANKS, first,
				      programs[0]->segment_count - 1u };
	for (uint32_t bank = 0;
	     bank < plan->original_head.bank_count && status == OK; bank++) {
		made->tables[bank].programs =
			made->pointers + (size_t)bank * program_count;
		made->tables[bank].image_programs =
			image_programs != NULL ?
				made->pointers + count +
					(size_t)bank * program_count :
				NULL;
		for (uint32_t image = 0;
		     image < (image_programs != NULL ? PROGRAM_FORMS : 1u) &&
		     status == OK;
		     image++) {
			for (uint32_t index = 0;
			     index < program_count && status == OK; index++) {
				const size_t at = (size_t)image * count +
						  (size_t)bank * program_count +
						  index;
				const struct execution_program *const source =
					image != 0 ? image_programs[index] :
						     programs[index];
				if (source == NULL || source->cards == NULL ||
				    source->segments == NULL ||
				    source->card_count != V41_HEAD_CARDS ||
				    source->segment_count == 0 ||
				    source->kind == PROGRAM_BATCH ||
				    (source->lease_count != 0 &&
				     source->leases == NULL)) {
					status = STATUS_INVALID_ARGUMENT;
					break;
				}
				if (source->result_card != V41_HEAD_RANKS ||
				    memcmp(&source->limits, &plan->limits,
					   sizeof(plan->limits)) != 0) {
					status = STATUS_INVALID_ARGUMENT;
					break;
				}
				for (uint32_t card = 0;
				     card < V41_HEAD_CARDS && status == OK;
				     card++)
					if (source->cards[card]
							    .arena_byte_count !=
						    plan->cards[card]
							    .arena_byte_count ||
					    (source->cards[card].launch_count !=
						     0 &&
					     source->cards[card].launches ==
						     NULL))
						status =
							STATUS_INVALID_ARGUMENT;
				if (status != OK)
					break;
				if (source->kind == PROGRAM_THIN && bank != 0) {
					status = program_make(plan, backend,
							      cards, source,
							      bank,
							      &made->owned[at]);
					made->pointers[at] =
						status == OK ?
							&made->owned[at]
								 ->program :
							NULL;
				} else {
					made->pointers[at] = source;
				}
			}
		}
	}
	if (status != OK) {
		v41_original_head_banks_destroy(made);
		return status;
	}
	*banks_out = made;
	return OK;
}
