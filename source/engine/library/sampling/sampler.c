/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler.c: the launch of the sampler. Contract: sampler.h; the kernel: sampler_records.h.
 */
#include "sampler.h"

#include <inttypes.h>
#include <string.h>

#include "common/hooks.h"
#include "engine/program/model_seam.h"
#include "sampler_shard_records.h"
#include "engine/library/prepare_checks.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

result_status
sampler_kernels_prepare(const struct backend_operations *backend,
			struct device_card *card,
			struct sampler_kernels_record *kernels_out)
{
    /* a backend that answers OK and gives no handle leaves this one, which the prepare function refuses */
    uint32_t handle = BACKEND_NO_KERNEL;
    if (backend == NULL || backend->kernel_prepare == NULL || card == NULL || kernels_out == NULL) {
	    LOG_ERROR("the kernel of the sampler: a missing argument");
	    return STATUS_INVALID_ARGUMENT;
    }
    const result_status status =
	    backend->kernel_prepare(card, KERNEL_FILE_SAMPLER, KERNEL_SAMPLER,
				    SAMPLER_ARGUMENT_BYTES, &handle);
    if (status != STATUS_OK) {
	    LOG_ERROR("the kernel of the sampler: %s is not prepared: %s",
		      KERNEL_SAMPLER, status_text(status));
	    return status;
    }
    kernels_out->sampler = handle;
    return STATUS_OK;
}

result_status
sampler_prepare_entry(const struct sampler_description *description,
		      uint64_t arena_byte_count,
		      const struct sampler_kernels_record *kernels,
		      struct kernel_launch *launch_out)
{
    static const char what[] = "the sampler";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out) ||
	!prepare_kernel_is_prepared(what, kernels->sampler, KERNEL_SAMPLER) ||
	!prepare_reserved_is_zero(what, description->reserved_zero)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t row_count = description->row_count;
    const uint32_t vocabulary_size = description->vocabulary_size;
    if (row_count == 0 || row_count > LINEAR_MAXIMUM_ROWS ||
	vocabulary_size == 0 || vocabulary_size > SAMPLER_MOST_VOCABULARY ||
	vocabulary_size % SAMPLER_TOKENS_PER_LOAD_VALUE != 0 ||
	description->session_capacity == 0) {
	    LOG_ERROR("%s: %" PRIu32 " rows, %" PRIu32 " tokens, %" PRIu32
		      " sessions: 1 to %u rows, %u to %u "
		      "tokens in %us, a session or more",
		      what, row_count, vocabulary_size,
		      description->session_capacity, LINEAR_MAXIMUM_ROWS,
		      SAMPLER_TOKENS_PER_LOAD_VALUE, SAMPLER_MOST_VOCABULARY,
		      SAMPLER_TOKENS_PER_LOAD_VALUE);
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t row_bytes =
	    linear_sums_byte_count(row_count, vocabulary_size);
    const struct checked_range ranges[] = {
	    { "the step area", &description->step,
	      STEP_BLOCK_BYTES_VALUE +
		      (uint64_t)row_count * STEP_ROW_BYTES_VALUE,
	      false },
	    { "the sessions", &description->sessions,
	      (uint64_t)description->session_capacity * STEP_SESSION_BYTES,
	      false },
	    { "the logits", &description->logits, row_bytes, false },
	    { "the weights", &description->weights, row_bytes, false },
	    { "the result area", &description->result,
	      STEP_RESULT_BYTES + (uint64_t)row_count * STEP_ROW_RESULT_BYTES,
	      false },
    };
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct sampler_arguments record = {
	    .step_offset = description->step.byte_offset,
	    .session_offset = description->sessions.byte_offset,
	    .logits_offset = description->logits.byte_offset,
	    .weights_offset = description->weights.byte_offset,
	    .result_offset = description->result.byte_offset,
	    .vocabulary_size = vocabulary_size,
	    .session_capacity = description->session_capacity,
    };
    prepare_launch_fill(launch_out, kernels->sampler, 1, row_count,
			SAMPLER_THREADS, 0, &record, SAMPLER_ARGUMENT_BYTES);
    return STATUS_OK;
}

result_status sampler_logits_parts(const struct execution_program *program,
				   uint32_t vocabulary_size,
				   struct sampler_logits_part *parts_out,
				   uint32_t most, uint32_t *count_out)
{
	struct sampler_logits_part parts[BACKEND_MAXIMUM_CARDS];
	uint32_t count = 0, covered = 0;
	bool tiles = program != NULL && parts_out != NULL &&
		     count_out != NULL && program->kind == PROGRAM_THIN &&
		     program->card_count <= BACKEND_MAXIMUM_CARDS;
	for (uint32_t card = 0; tiles && card < program->card_count; card++) {
		const struct program_card_record *const own =
			&program->cards[card];
		bool found = false;
		for (uint32_t index = 0;
		     own->logits_offset != PROGRAM_NOWHERE &&
		     index < own->launch_count && !found;
		     index++) {
			const struct kernel_launch *const launch =
				&own->launches[index];
			struct sampler_arguments whole;
			struct sampler_shard_arguments cut;
			memcpy(&whole, launch->argument, sizeof(whole));
			memcpy(&cut, launch->argument, sizeof(cut));
			if (launch->argument_byte_count ==
				    SAMPLER_ARGUMENT_BYTES &&
			    whole.step_offset == own->step_offset &&
			    whole.logits_offset == own->logits_offset &&
			    whole.result_offset == own->result_offset)
				parts[count] = (struct sampler_logits_part){
					own->logits_offset, card, 0,
					whole.vocabulary_size, 0
				};
			else if (launch->argument_byte_count ==
					 SAMPLER_SHARD_ARGUMENT_BYTES &&
				 cut.step_offset == own->step_offset &&
				 cut.logits_offset == own->logits_offset &&
				 cut.result_offset == own->result_offset)
				parts[count] = (struct sampler_logits_part){
					own->logits_offset, card,
					cut.first_token, cut.shard_size, 0
				};
			else
				continue;
			found = true;
			covered += parts[count].token_count;
			count++;
		}
	}
	/* the parts tile the vocabulary: no two overlap and together they cover it */
	for (uint32_t index = 0; tiles && index < count; index++) {
		tiles = parts[index].token_count != 0 &&
			parts[index].first_token <= vocabulary_size &&
			parts[index].token_count <=
				vocabulary_size - parts[index].first_token;
		for (uint32_t other = 0; tiles && other < index; other++)
			tiles = parts[other].first_token +
						parts[other].token_count <=
					parts[index].first_token ||
				parts[index].first_token +
						parts[index].token_count <=
					parts[other].first_token;
	}
	if (!tiles || count == 0 || count > most ||
	    covered != vocabulary_size) {
		LOG_ERROR(
			"sampler logits: refused: no thin program, no sampler of the head, parts that do not tile the "
			"vocabulary or more than the room");
		return STATUS_INVALID_ARGUMENT;
	}
	memcpy(parts_out, parts, count * sizeof(*parts));
	*count_out = count;
	return STATUS_OK;
}
