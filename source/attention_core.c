/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core.c: the launch of the attention core, prepared from a description. Contract: attention_core.h; the
 * kernel, its record and its launch: attention_core_records.h.
 *
 *   - The counts are checked first, then the ranges (prepare_common.h), then the launch is filled. Nothing is written
 *     before every check has passed, and each check that fails writes the one error line of the refusal.
 *   - The order picks the kernels: the fixed order's one launch, the reference order's two (its scores, then its
 *     softmax and values), every launch of the same record but heads_per_block (attention_core_records.h).
 *   - A byte count here cannot pass 64 bits: rows are at most 8, a context at most 16,384 tokens, sessions fewer
 *     than 2^32 (asserted below).
 */
#include "anemone/attention_core.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#include "anemone/hooks.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define RANGES_OF_THE_POOL 2u /* the pool and the selection */
#define RANGES_OF_EVERY_LAUNCH 6u
#define HEADS_STEP 2u /* the heads of a block go by powers of 2 */
/* the block core's rows a launch: a BATCH step's (a batched draft's whole rows, --batch_draft_wide=3); each block, a
 * session's run of consecutive positions, a draft's R rows (STEP_DRAFT_MOST at most) */
#define BLOCK_MOST_ROWS 64u
#define VALUE_TILES_FROM_ROWS 4u /* class 80's value blocks of 16 heads */
#define CLASS_120_SCORE_TILES_MOST \
	2u /* in the forms of class 120: its 99 KiB a block */
#define RING_SESSION_BYTES ((uint64_t)MODEL_RING_SLOTS * MODEL_STATE_ROW_BYTES)
#define SINKS_BYTES ((uint64_t)MODEL_HEADS_VALUE * sizeof(float))
#define USE_BYTES                             \
	((uint64_t)MODEL_ROTATION_ROW_BYTES / \
	 MODEL_ROTATION_USES) /* of one use */

_Static_assert(
	(uint64_t)MODEL_CONTEXT_MOST *MODEL_STATE_ROW_BYTES <= UINT32_MAX &&
		RING_SESSION_BYTES <= UINT32_MAX,
	"a session's pool and ring times 2^32 sessions stay below 2^64 bytes");
_Static_assert(ATTENTION_CORE_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES,
	       "the record fits the argument of a launch");

/* The heads of class 120's value blocks in the exact profile by the program's rows (measured: 1 at a row, then the
 * fewest blocks that keep the PRO's 188 multiprocessors fed). */
static const uint32_t exact_value_heads[LINEAR_MAXIMUM_ROWS + 1u] = {
	1u, 1u, 2u, 4u, 4u, 2u, 2u, 2u, 2u
};

/* The most heads, at most wanted and halving from it, that divide head_count (a multiple of 8). */
static uint32_t most_heads_dividing(uint32_t wanted, uint32_t head_count)
{
	uint32_t heads = wanted;
	while (head_count % heads != 0) {
		heads /= HEADS_STEP;
	}
	return heads;
}

/* One kernel of the file into handle_out, or its status behind the error line that names it. */
MUST_CHECK static result_status
kernel_prepare(const struct backend_operations *backend,
	       struct device_card *card, const char *file_name,
	       const char *function_name, uint32_t *handle_out)
{
	const result_status status = backend->kernel_prepare(
		card, file_name, function_name, ATTENTION_CORE_ARGUMENT_BYTES,
		handle_out);
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the kernels of the attention core: %s of %s is not prepared: %s",
			function_name, file_name, status_text(status));
	}
	return status;
}

result_status attention_core_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct attention_core_kernels *kernels_out)
{
	const char *const file_name =
		arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			KERNEL_FILE_ATTENTION_CORE_NATIVE :
			KERNEL_FILE_ATTENTION_CORE;
	struct attention_core_kernels kernels = { BACKEND_NO_KERNEL,
						  BACKEND_NO_KERNEL,
						  BACKEND_NO_KERNEL,
						  BACKEND_NO_KERNEL };
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR(
			"the kernels of the attention core: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = kernel_prepare(backend, card, file_name,
					      KERNEL_ATTENTION_CORE,
					      &kernels.attention_core);
	if (status == STATUS_OK) {
		status = kernel_prepare(backend, card, file_name,
					KERNEL_ATTENTION_CORE_SCORES,
					&kernels.attention_core_scores);
	}
	if (status == STATUS_OK) {
		status = kernel_prepare(backend, card, file_name,
					KERNEL_ATTENTION_CORE_VALUES,
					&kernels.attention_core_values);
	}
	if (status == STATUS_OK) {
		*kernels_out = kernels;
	}
	return status;
}

/* The counts of attention_core.h, refused together by one error line that states them all. */
static bool
counts_are_valid(const char *what,
		 const struct attention_core_description *description)
{
	const uint32_t limit = description->context_token_limit,
		       ratio = description->pool_ratio;
	const uint32_t first_head = description->first_head,
		       head_count = description->head_count;
	const bool is_block = description->window_mode ==
			      ATTENTION_CORE_WINDOW_BLOCK;
	if ((uint32_t)description->order > LINEAR_ORDER_REFERENCE ||
	    description->window_mode > ATTENTION_CORE_WINDOW_BLOCK ||
	    (is_block &&
	     (ratio != 0 || description->order != LINEAR_ORDER_FIXED)) ||
	    description->row_count == 0 ||
	    description->row_count >
		    (is_block ? BLOCK_MOST_ROWS : LINEAR_WIDE_MOST_ROWS) ||
	    description->session_capacity == 0 || limit == 0 ||
	    limit % MODEL_CONTEXT_GRANULE != 0 || limit > MODEL_CONTEXT_MOST ||
	    (ratio != 0 && ratio != MODEL_DIRECT_RATIO &&
	     ratio != MODEL_POOLED_RATIO) ||
	    head_count == 0 ||
	    description->most_heads_per_block %
			    ATTENTION_CORE_HEADS_PER_BLOCK !=
		    0 ||
	    first_head % ATTENTION_CORE_HEADS_PER_BLOCK != 0 ||
	    head_count % ATTENTION_CORE_HEADS_PER_BLOCK != 0 ||
	    first_head > MODEL_HEADS_VALUE ||
	    head_count > MODEL_HEADS_VALUE - first_head) {
		LOG_ERROR(
			"%s: order %" PRIu32 ", window %" PRIu32 ", %" PRIu32
			" rows, %" PRIu32 " sessions, %" PRIu32
			" tokens, ratio %" PRIu32 ", %" PRIu32
			" heads from %" PRIu32 ": order %" PRIu32 " or %" PRIu32
			", window %u or %u (the block: order %" PRIu32
			", ratio 0); 1 to %u "
			"rows; a session or more; whole granules of %u tokens, at most %u; ratio 0, %u or %u; "
			"whole blocks of %u heads, at least one, among the %u",
			what, (uint32_t)description->order,
			description->window_mode, description->row_count,
			description->session_capacity, limit, ratio, head_count,
			first_head, (uint32_t)LINEAR_ORDER_FIXED,
			(uint32_t)LINEAR_ORDER_REFERENCE,
			ATTENTION_CORE_WINDOW_CAUSAL,
			ATTENTION_CORE_WINDOW_BLOCK,
			(uint32_t)LINEAR_ORDER_FIXED, BLOCK_MOST_ROWS,
			MODEL_CONTEXT_GRANULE, MODEL_CONTEXT_MOST,
			MODEL_DIRECT_RATIO, MODEL_POOLED_RATIO,
			ATTENTION_CORE_HEADS_PER_BLOCK, MODEL_HEADS_VALUE);
		return false;
	}
	return true;
}

result_status attention_core_prepare(
	const struct attention_core_description *description,
	uint64_t arena_byte_count, const struct attention_core_kernels *kernels,
	struct kernel_launch launches_out[ATTENTION_CORE_MOST_LAUNCHES],
	uint32_t *launch_count_out)
{
	static const char what[] = "the attention core";
	if (!prepare_arguments_are_given(what, description, kernels,
					 launches_out) ||
	    !prepare_arguments_are_given(what, description, kernels,
					 launch_count_out) ||
	    !counts_are_valid(what, description)) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (description->output_compact > 1u ||
	    (description->cone_layer != 0 &&
	     (description->cone_layer < MODEL_DECODER_FIRST_LAYER ||
	      description->cone_layer >= MODEL_LAYERS))) {
		LOG_ERROR("%s: output_compact %" PRIu32
			  ": 0 or 1; cone_layer %" PRIu32
			  ": 0 or a decoder layer",
			  what, description->output_compact,
			  description->cone_layer);
		return STATUS_INVALID_ARGUMENT;
	}
	const bool is_reference = description->order == LINEAR_ORDER_REFERENCE;
	const bool is_block = description->window_mode ==
			      ATTENTION_CORE_WINDOW_BLOCK;
	const uint32_t fixed_kernel = is_block ? kernels->attention_core_block :
						 kernels->attention_core;
	if (is_reference ?
		    !prepare_kernel_is_prepared(what,
						kernels->attention_core_scores,
						KERNEL_ATTENTION_CORE_SCORES) ||
			    !prepare_kernel_is_prepared(
				    what, kernels->attention_core_values,
				    KERNEL_ATTENTION_CORE_VALUES) :
		    !prepare_kernel_is_prepared(
			    what, fixed_kernel,
			    is_block ? KERNEL_ATTENTION_CORE_BLOCK :
				       KERNEL_ATTENTION_CORE)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t rows = description->row_count;
	const uint64_t sessions = description->session_capacity;
	const bool has_pool = description->pool_ratio != 0;
	const uint64_t pool_stride = has_pool ?
					     description->context_token_limit /
						     description->pool_ratio *
						     MODEL_STATE_ROW_BYTES :
					     0;
	struct checked_range ranges[] = {
		{ "the step area", &description->step,
		  STEP_BLOCK_BYTES_VALUE + rows * STEP_ROW_BYTES_VALUE, false },
		{ "the queries", &description->queries,
		  rows * MODEL_HEADS_ROW_BYTES, false },
		{ "the ring", &description->ring, sessions * RING_SESSION_BYTES,
		  false },
		{ "the sinks", &description->sinks, SINKS_BYTES, false },
		{ "the rotation", &description->rotation,
		  rows * MODEL_ROTATION_ROW_BYTES, false },
		{ "the output", &description->output,
		  description->output_compact != 0 ?
			  rows * description->head_count *
				  MODEL_STATE_ROW_BYTES :
			  rows * MODEL_HEADS_ROW_BYTES,
		  false },
		{ "the pool", &description->pool, sessions * pool_stride,
		  false },
		{ "the selection", &description->selection,
		  rows * MODEL_SELECTION_ROW_BYTES, false },
		{ "the scores", &description->scores,
		  rows * ATTENTION_CORE_SCORES_ROW_BYTES, false },
	};
	uint32_t range_count =
		RANGES_OF_EVERY_LAUNCH + (has_pool ? RANGES_OF_THE_POOL : 0u);
	if (is_reference) { /* the scores, behind the pool's or in their place */
		ranges[range_count++] = ranges[COUNT_OF(ranges) - 1u];
	}
	if (!prepare_ranges_are_valid(what, ranges, range_count,
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct attention_core_arguments_record record = {
		.step_offset = description->step.byte_offset,
		.queries_offset = description->queries.byte_offset,
		.ring_offset = description->ring.byte_offset,
		.ring_session_stride = RING_SESSION_BYTES,
		.pool_offset = has_pool ? description->pool.byte_offset : 0,
		.pool_session_stride = pool_stride,
		.selection_offset =
			has_pool ? description->selection.byte_offset : 0,
		.sinks_offset = description->sinks.byte_offset,
		.rotation_offset =
			description->rotation.byte_offset +
			(has_pool ? USE_BYTES : 0), /* theta 1 with a pool */
		.output_offset = description->output.byte_offset,
		.scores_offset =
			is_reference ? description->scores.byte_offset : 0,
		.context_token_limit = description->context_token_limit,
		.session_capacity = description->session_capacity,
		.pool_ratio = description->pool_ratio,
		.first_head = description->first_head,
		.heads_per_block = ATTENTION_CORE_HEADS_PER_BLOCK,
		.output_row_heads = description->output_compact != 0 ?
					    description->head_count :
					    0u,
		.cone_layer = description->cone_layer,
	};
	const uint32_t heads = description->head_count;
	if (!is_reference) {
		prepare_launch_fill(&launches_out[0], fixed_kernel,
				    description->row_count,
				    heads / ATTENTION_CORE_HEADS_PER_BLOCK,
				    ATTENTION_CORE_THREADS,
				    ATTENTION_CORE_SHARED_BYTES_VALUE, &record,
				    ATTENTION_CORE_ARGUMENT_BYTES);
		*launch_count_out = 1;
		return STATUS_OK;
	}
#if defined(LINEAR_CLASS_FORMS)
	const bool is_emulated = false;
#else /* the reference order on class 120 sums as class 80 without its tensor cores; 120 native uses its own */
	const bool is_emulated = description->arithmetic_class ==
				 ARITHMETIC_CLASS_SM_120;
#endif
	const bool is_class_120 =
		description->arithmetic_class == ARITHMETIC_CLASS_SM_120 ||
		description->arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE;
	/* a launch of more than 8 rows (a chunk's core, model_v41_tp2_stage.h's chunk_core): an 8-row launch's blocks,
	 * one row each, over every row */
	const uint32_t row_count = description->row_count <
						   LINEAR_MAXIMUM_ROWS ?
					   description->row_count :
					   LINEAR_MAXIMUM_ROWS;
	uint32_t score_heads = ATTENTION_CORE_EXACT_SCORE_HEADS,
		 value_heads = exact_value_heads[row_count];
	if (!is_emulated) {
		uint32_t score_tiles = 1,
			 value_tiles = row_count >= VALUE_TILES_FROM_ROWS ?
					       ATTENTION_CORE_VALUE_TILES_MOST :
					       1u;
		while (score_tiles * HEADS_STEP <= row_count &&
		       score_tiles * HEADS_STEP <=
			       (is_class_120 ?
					CLASS_120_SCORE_TILES_MOST :
					ATTENTION_CORE_SCORE_TILES_MOST)) {
			score_tiles *= HEADS_STEP;
		}
		score_heads = most_heads_dividing(
			score_tiles * ATTENTION_CORE_HEADS_PER_BLOCK, heads);
		value_heads = most_heads_dividing(
			(is_class_120 ? 1u : value_tiles) *
				ATTENTION_CORE_HEADS_PER_BLOCK,
			heads);
	}
	value_heads = most_heads_dividing(value_heads, heads);
	if (!is_emulated && description->most_heads_per_block != 0) { /* a cap: more blocks, each head's sums its own */
		score_heads = most_heads_dividing(score_heads < description->most_heads_per_block ?
							  score_heads : description->most_heads_per_block, heads);
		value_heads = most_heads_dividing(value_heads < description->most_heads_per_block ?
							  value_heads : description->most_heads_per_block, heads);
	}
	record.heads_per_block = score_heads;
	prepare_launch_fill(
		&launches_out[0], kernels->attention_core_scores,
		description->row_count, heads / score_heads,
		ATTENTION_CORE_THREADS,
		is_emulated ?
			ATTENTION_CORE_EXACT_SCORES_SHARED_BYTES :
			ATTENTION_CORE_SCORES_SHARED_BYTES(
				score_heads / ATTENTION_CORE_HEADS_PER_BLOCK),
		&record, ATTENTION_CORE_ARGUMENT_BYTES);
	launches_out[0].grid[LAUNCH_DIMENSIONS - 1] =
		ATTENTION_CORE_KEYS / (is_emulated ?
					       ATTENTION_CORE_EXACT_TILE_KEYS :
					       ATTENTION_CORE_TILE_KEYS);
	record.heads_per_block = value_heads;
	prepare_launch_fill(
		&launches_out[1], kernels->attention_core_values,
		description->row_count, heads / value_heads,
		is_emulated ? ATTENTION_CORE_THREADS :
			      ATTENTION_CORE_VALUE_THREADS,
		is_emulated ?
			ATTENTION_CORE_EXACT_VALUES_SHARED_BYTES :
			ATTENTION_CORE_VALUES_SHARED_BYTES(
				value_heads / ATTENTION_CORE_HEADS_PER_BLOCK),
		&record, ATTENTION_CORE_ARGUMENT_BYTES);
	launches_out[1].grid[LAUNCH_DIMENSIONS - 1] =
		MODEL_HEAD_DIMS / ATTENTION_CORE_TILE_DIMS;
	*launch_count_out = ATTENTION_CORE_MOST_LAUNCHES;
	return STATUS_OK;
}
