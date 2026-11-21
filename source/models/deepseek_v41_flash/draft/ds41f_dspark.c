/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_dspark.c: the draft's kernels and the launches of ds41f_dspark's. Contract: ds41f_dspark.h; the kernels, their
 * records and launches: ds41f_dspark_records.h. As linear.c: the counts, then every range against what the launch
 * touches and the arena, then the launch; nothing is written before every check has passed.
 */
#include "ds41f_dspark.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#include "common/hooks.h"
#include "engine/library/prepare_checks.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

/* Every kernel the draft adds to its card, in the order they are prepared. */
enum draft_kernel {
	DRAFT_INPUTS_KERNEL,
	FINAL_KEPT_KERNEL,
	SELECT_DRAFT_KERNEL,
	WINDOW_KERNEL,
	CORE_BLOCK_KERNEL,
	VERIFY_KERNEL,
	MARKOV_KERNEL,
	CONFIDENCE_KERNEL,
	DRAFT_KERNEL_COUNT
};

result_status ds41f_draft_kernels_prepare(const struct backend_operations *backend,
				    struct device_card *card,
				    uint32_t arithmetic_class,
				    struct ds41f_hyper_kernels_record *hyper,
				    struct router_kernels_record *router,
				    struct ds41f_attention_state_kernels *state,
				    struct ds41f_attention_core_kernels *core,
				    struct ds41f_draft_kernels_record *kernels_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || hyper == NULL || router == NULL || state == NULL ||
	    core == NULL || kernels_out == NULL) {
		LOG_ERROR("the kernels of the draft: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	const char *const core_file =
		arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			DS41F_MODULE_ATTENTION_CORE_NATIVE :
			DS41F_MODULE_ATTENTION_CORE;
	const struct {
		const char *file_name;
		const char *function_name;
		uint32_t argument_byte_count;
	} kernels[DRAFT_KERNEL_COUNT] = {
		[DRAFT_INPUTS_KERNEL] = { DS41F_MODULE_HYPER_CONNECTION,
					  DS41F_KERNEL_HYPER_DRAFT_INPUTS,
					  DS41F_HYPER_DRAFT_INPUTS_ARGUMENT_BYTES },
		[FINAL_KEPT_KERNEL] = { DS41F_MODULE_HYPER_CONNECTION,
					DS41F_KERNEL_HYPER_FINAL_KEPT,
					DS41F_HYPER_FINISH_ARGUMENT_BYTES },
		[SELECT_DRAFT_KERNEL] = { DS41F_MODULE_ROUTER,
					  DS41F_KERNEL_ROUTER_SELECT_DRAFT,
					  DS41F_ROUTER_SELECT_ARGUMENT_BYTES },
		[WINDOW_KERNEL] = { DS41F_MODULE_ATTENTION_STATE,
				    DS41F_KERNEL_ATTENTION_WINDOW,
				    DS41F_ATTENTION_PREPARE_ARGUMENT_BYTES },
		[CORE_BLOCK_KERNEL] = { core_file, DS41F_KERNEL_ATTENTION_CORE_BLOCK,
					DS41F_ATTENTION_CORE_ARGUMENT_BYTES },
		[VERIFY_KERNEL] = { DS41F_MODULE_DSPARK, DS41F_KERNEL_VERIFY_MATCHES,
				    DS41F_VERIFY_MATCHES_ARGUMENT_BYTES },
		[MARKOV_KERNEL] = { DS41F_MODULE_DSPARK, DS41F_KERNEL_DRAFT_MARKOV_BIAS,
				    DS41F_DRAFT_MARKOV_BIAS_ARGUMENT_BYTES },
		[CONFIDENCE_KERNEL] = { DS41F_MODULE_DSPARK,
					DS41F_KERNEL_DRAFT_CONFIDENCE,
					DS41F_DRAFT_CONFIDENCE_ARGUMENT_BYTES },
	};
	uint32_t handles[DRAFT_KERNEL_COUNT];
	for (uint32_t kernel = 0; kernel < DRAFT_KERNEL_COUNT; kernel++) {
		handles[kernel] =
			BACKEND_NO_KERNEL; /* a backend that answers OK and gives no handle */
		const result_status status = backend->kernel_prepare(
			card, kernels[kernel].file_name,
			kernels[kernel].function_name,
			kernels[kernel].argument_byte_count, &handles[kernel]);
		if (status != STATUS_OK) {
			LOG_ERROR(
				"the kernels of the draft: %s of %s is not prepared: %s",
				kernels[kernel].function_name,
				kernels[kernel].file_name, status_text(status));
			return status;
		}
	}
	hyper->draft_inputs = handles[DRAFT_INPUTS_KERNEL];
	hyper->final_kept = handles[FINAL_KEPT_KERNEL];
	router->select_draft = handles[SELECT_DRAFT_KERNEL];
	state->window = handles[WINDOW_KERNEL];
	core->attention_core_block = handles[CORE_BLOCK_KERNEL];
	*kernels_out =
		(struct ds41f_draft_kernels_record){ handles[VERIFY_KERNEL],
					       handles[MARKOV_KERNEL],
					       handles[CONFIDENCE_KERNEL] };
	return STATUS_OK;
}

/* draft_positions of 1 to STEP_DRAFT_MOST: true, or false behind the one error line of the refusal. */
static bool positions_are_valid(const char *what, uint32_t draft_positions)
{
	if (draft_positions == 0 || draft_positions > STEP_DRAFT_MOST) {
		LOG_ERROR("%s: %" PRIu32 " draft positions: 1 to %u", what,
			  draft_positions, STEP_DRAFT_MOST);
		return false;
	}
	return true;
}

result_status
ds41f_verify_matches_prepare(const struct ds41f_verify_matches_description *description,
		       uint64_t arena_byte_count,
		       const struct ds41f_draft_kernels_record *kernels,
		       struct kernel_launch *launch_out)
{
	static const char what[] = "the verify kernel";
	if (!prepare_arguments_are_given(what, description, kernels,
					 launch_out) ||
	    !prepare_kernel_is_prepared(what, kernels->verify_matches,
					DS41F_KERNEL_VERIFY_MATCHES) ||
	    !positions_are_valid(what, description->draft_positions)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const bool table = (description->flags & DS41F_DRAFT_ROUND_TABLE) != 0;
	const bool slabs = (description->flags & DS41F_DRAFT_SLAB_RESULTS) != 0;
	const bool batched = (description->flags & DS41F_DRAFT_BATCHED) != 0;
	const bool wide_step = (description->flags & DS41F_DRAFT_WIDE_STEP) != 0;
	const uint64_t batched_rows = ((uint64_t)description->verifier + 1u) *
				      description->draft_positions;
	if ((description->flags & ~(DS41F_DRAFT_ROUND_TABLE | DS41F_DRAFT_SLAB_RESULTS |
				    DS41F_DRAFT_BATCHED | DS41F_DRAFT_WIDE_STEP)) != 0 ||
	    (wide_step && !batched) || (description->flags != 0 && !table) ||
	    (table ? description->round_capacity == 0 :
		     description->round_capacity != 0 ||
			     description->verifier != 0) ||
	    (batched ? batched_rows > (uint64_t)description->draft_slabs *
					       DS41F_DRAFT_SLAB_ROWS ||
			       description->verifier >= DS41F_DRAFT_BATCH_SLOTS :
		       description->draft_slabs != 0)) {
		LOG_ERROR("%s: flags 0x%" PRIx32 ", verifier %" PRIu32
			  ", %" PRIu32
			  " rounds: a round table with slab results "
			  "or alone, rounds and a verifier only with a table",
			  what, description->flags, description->verifier,
			  description->round_capacity);
		return STATUS_INVALID_ARGUMENT;
	}
	if (description->row_count == 0 ||
	    description->row_count >
		    (table ? DS41F_DRAFT_BATCH_MOST_ROWS : LINEAR_MAXIMUM_ROWS) ||
	    description->end_token_id >= description->vocabulary_size ||
	    DS41F_DRAFT_NOISE_TOKEN >= description->vocabulary_size) {
		LOG_ERROR("%s: %" PRIu32 " rows, end token %" PRIu32
			  " of %" PRIu32 " tokens: 1 to %u rows, the end "
			  "token and the noise token %u below the vocabulary",
			  what, description->row_count,
			  description->end_token_id,
			  description->vocabulary_size, LINEAR_MAXIMUM_ROWS,
			  DS41F_DRAFT_NOISE_TOKEN);
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t rows = description->row_count,
		       positions = description->draft_positions;
	const struct checked_range ranges[] = {
		{ "the step area", &description->step,
		  STEP_BLOCK_BYTES_VALUE + rows * STEP_ROW_BYTES_VALUE, false },
		{ "the result area", &description->result,
		  slabs ? (rows + DS41F_DRAFT_SLAB_ROWS - 1u) / DS41F_DRAFT_SLAB_ROWS *
				  DS41F_DRAFT_SLAB_RESULT_BYTES :
			  STEP_RESULT_BYTES + rows * STEP_ROW_RESULT_BYTES,
		  false },
		{ "the round", &description->round,
		  (table ? description->round_capacity : 1u) *
			  (uint64_t)STEP_ROUND_BYTES,
		  false },
		{ "the draft's step area", &description->draft_step,
		  batched ?
			  (uint64_t)description->draft_slabs *
					  DS41F_DRAFT_SLAB_STEP_BYTES +
				  (wide_step ?
					   STEP_BLOCK_BYTES_VALUE +
						   (uint64_t)description
								   ->draft_slabs *
							   DS41F_DRAFT_SLAB_ROWS *
							   STEP_ROW_BYTES_VALUE :
					   0u) :
			  STEP_BLOCK_BYTES_VALUE +
				  positions * STEP_ROW_BYTES_VALUE,
		  false },
		{ "the draft's views", &description->draft_views,
		  positions * (batched ? DS41F_DRAFT_SLAB_STEP_BYTES :
					 STEP_DRAFT_VIEW_BYTES),
		  false },
	};
	if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct ds41f_verify_matches_arguments record = {
		.step_offset = description->step.byte_offset,
		.result_offset = description->result.byte_offset,
		.round_offset = description->round.byte_offset,
		.draft_step_offset = description->draft_step.byte_offset,
		.draft_views_offset = description->draft_views.byte_offset,
		.row_count = description->row_count,
		.draft_positions = description->draft_positions,
		.end_token_id = description->end_token_id,
		.noise_token_id = DS41F_DRAFT_NOISE_TOKEN,
		.verifier = description->verifier,
		.flags = description->flags,
		.round_capacity = description->round_capacity,
		.draft_slabs = description->draft_slabs
	};
	prepare_launch_fill(launch_out, kernels->verify_matches, 1, 1,
			    DS41F_VERIFY_MATCHES_THREADS, 0, &record, sizeof(record));
	return STATUS_OK;
}

result_status ds41f_draft_markov_bias_prepare(
	const struct ds41f_draft_markov_bias_description *description,
	uint64_t arena_byte_count, const struct ds41f_draft_kernels_record *kernels,
	struct kernel_launch *launch_out)
{
	static const char what[] = "the Markov bias";
	if (!prepare_arguments_are_given(what, description, kernels,
					 launch_out) ||
	    !prepare_kernel_is_prepared(what, kernels->markov_bias,
					DS41F_KERNEL_DRAFT_MARKOV_BIAS) ||
	    !positions_are_valid(what, description->draft_positions)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t vocabulary = description->vocabulary_size;
	const uint32_t embedding_rows = description->embedding_rows != 0 ?
						description->embedding_rows :
						vocabulary;
	const bool batched = description->flags == DS41F_DRAFT_BATCHED;
	if (!prepare_reserved_is_zero(what,
				      description->flags & ~DS41F_DRAFT_BATCHED)) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (description->position >= description->draft_positions ||
	    vocabulary == 0 || embedding_rows < vocabulary) {
		LOG_ERROR("%s: position %" PRIu32 " of %" PRIu32 ", %" PRIu32
			  " tokens, %" PRIu32
			  " embedding rows: a position below the draft's, "
			  "a token or more, at least as many rows",
			  what, description->position,
			  description->draft_positions, vocabulary,
			  embedding_rows);
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t positions = description->draft_positions;
	const uint64_t matrix_bytes =
		(uint64_t)vocabulary * DS41F_DRAFT_MARKOV_ROW_BYTES;
	/* BATCHED: a position's slots of logits and Markov rows (ds41f_dspark_records.h) */
	const uint64_t slots = batched ? DS41F_DRAFT_BATCH_SLOTS : 1u;
	const uint64_t logits_row_bytes = (uint64_t)vocabulary * sizeof(float);
	const struct checked_range ranges[] = {
		{ "the draft's step area", &description->draft_step,
		  batched ? DS41F_DRAFT_SLAB_STEP_BYTES :
			    STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE,
		  false },
		{ "the draft's result areas", &description->results,
		  positions * (batched ? DS41F_DRAFT_SLAB_RESULT_BYTES :
					 STEP_DRAFT_RESULT_BYTES),
		  false },
		{ "markov_head.embed.weight", &description->embedding,
		  (uint64_t)embedding_rows * DS41F_DRAFT_MARKOV_ROW_BYTES, true },
		{ "markov_head.head.weight", &description->head, matrix_bytes,
		  true },
		{ "the draft's logits", &description->logits,
		  positions * slots * logits_row_bytes, false },
		{ "the Markov rows", &description->markov,
		  positions * slots * DS41F_DRAFT_MARKOV_ROW_BYTES, false },
	};
	if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct ds41f_draft_markov_bias_arguments record = {
		.draft_step_offset = description->draft_step.byte_offset,
		.results_offset = description->results.byte_offset,
		.embedding_offset = description->embedding.byte_offset,
		.head_offset = description->head.byte_offset,
		.logits_offset =
			description->logits.byte_offset +
			description->position * slots * logits_row_bytes,
		.markov_offset = description->markov.byte_offset +
				 (uint64_t)description->position * slots *
					 DS41F_DRAFT_MARKOV_ROW_BYTES,
		.position = description->position,
		.vocabulary_size = vocabulary,
		.embedding_rows = embedding_rows,
		.flags = description->flags
	};
	const uint64_t block_entries =
		(description->flags & DS41F_DRAFT_BATCHED) != 0 ?
			DS41F_DRAFT_MARKOV_BATCHED_ENTRIES_PER_BLOCK :
			DS41F_DRAFT_MARKOV_ENTRIES_PER_BLOCK;
	prepare_launch_fill(
		launch_out, kernels->markov_bias,
		(uint32_t)(((uint64_t)vocabulary + block_entries - 1u) /
			   block_entries),
		1, DS41F_DRAFT_MARKOV_THREADS, 0, &record, sizeof(record));
	return STATUS_OK;
}

result_status
ds41f_draft_confidence_prepare(const struct ds41f_draft_confidence_description *description,
			 uint64_t arena_byte_count,
			 const struct ds41f_draft_kernels_record *kernels,
			 struct kernel_launch *launch_out)
{
	static const char what[] = "the confidence";
	if (!prepare_arguments_are_given(what, description, kernels,
					 launch_out) ||
	    !prepare_kernel_is_prepared(what, kernels->confidence,
					DS41F_KERNEL_DRAFT_CONFIDENCE) ||
	    !prepare_reserved_is_zero(what,
				      description->flags & ~(DS41F_DRAFT_ROUND_TABLE |
							     DS41F_DRAFT_BATCHED)) ||
	    !positions_are_valid(what, description->draft_positions)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const bool batched = (description->flags & DS41F_DRAFT_BATCHED) != 0;
	if (batched ? (description->flags & DS41F_DRAFT_ROUND_TABLE) == 0 ||
			      description->verifier >= DS41F_DRAFT_BATCH_SLOTS :
		      description->verifier != 0) {
		LOG_ERROR("%s: flags 0x%" PRIx32 ", verifier %" PRIu32
			  ": a batched draft's slot, with a round table",
			  what, description->flags, description->verifier);
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t positions = description->draft_positions;
	const uint64_t slots = batched ? DS41F_DRAFT_BATCH_SLOTS : 1u;
	const struct checked_range ranges[] = {
		{ "the draft's step area", &description->draft_step,
		  batched ? DS41F_DRAFT_SLAB_STEP_BYTES : STEP_BLOCK_BYTES_VALUE,
		  false },
		{ "the collapse rows", &description->collapse,
		  positions * DS41F_DRAFT_COLLAPSE_ROW_BYTES, false },
		{ "the Markov rows", &description->markov,
		  positions * slots * DS41F_DRAFT_MARKOV_ROW_BYTES, false },
		{ "confidence_head.proj.weight", &description->weight,
		  RECORD_HALVES(DS41F_DRAFT_CONFIDENCE_INPUTS), true },
		{ "the draft's result areas", &description->results,
		  positions * (batched ? DS41F_DRAFT_SLAB_RESULT_BYTES :
					 STEP_DRAFT_RESULT_BYTES),
		  false },
		{ "the round", &description->round, STEP_ROUND_BYTES, false },
	};
	if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct ds41f_draft_confidence_arguments record = {
		.draft_step_offset = description->draft_step.byte_offset,
		.collapse_offset = description->collapse.byte_offset,
		.markov_offset = description->markov.byte_offset,
		.weight_offset = description->weight.byte_offset,
		.results_offset = description->results.byte_offset,
		.round_offset = description->round.byte_offset,
		.draft_positions = description->draft_positions,
		.flags = description->flags,
		.verifier = description->verifier,
		.reserved_zero = 0
	};
	prepare_launch_fill(launch_out, kernels->confidence, 1, STEP_DRAFT_MOST,
			    DS41F_DRAFT_CONFIDENCE_THREADS, 0, &record,
			    sizeof(record));
	return STATUS_OK;
}
