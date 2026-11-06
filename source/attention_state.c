/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state.c: the launches of the attention state, prepared from descriptions. Contract: attention_state.h;
 * the kernels, their records and launches: attention_state_records.h.
 *
 *   - As linear.c: the counts, then every range against what the launches touch and the arena, then the launches;
 *     nothing is written before every check has passed. A byte count is a product in 64 bits that stops at
 *     UINT64_MAX (linear.h's count functions), which no range holds.
 *   - A layer's ranges are one list, so that no two of its arrays share a byte. A pool implies an indexer and ratio
 *     2 a pool, so the ranges a layer has are a prefix of the list: the window, the indexer, the pool, ratio 2's.
 */
#include "anemone/attention_state.h"

#include <inttypes.h>
#include <stdbool.h>
#include <string.h>

#include "anemone/hooks.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
enum { WINDOW_RANGES = 10, INDEX_RANGES = 12, RATIO_ONE_RANGES = 19, RANGES = 21 };  /* each kind's prefix */
enum {
	POOL_ONE_RANGES = 9
}; /* attention_pool_prepare's ranges of ratio 1 (no score, no slots) */
enum { QUERY_RANGES_WITHOUT_INDEX = 3 };                /* the queries alone: the index ranges are the last two */
#define ROTATION_FLOATS ((uint64_t)MODEL_ROTATION_USES * MODEL_ROTATED_DIMS)
#define SLOT_FLOATS ((uint64_t)ATTENTION_SLOT_VECTORS * MODEL_HEAD_DIMS)
#define TWO_TO_ONE MODEL_POOLED_RATIO
#define HEAD_DIMS MODEL_HEAD_DIMS /* short names for the list of ranges */
#define INDEX_DIMS MODEL_INDEX_HEAD_DIMS
#define DRAFT_KERNELS 1u                        /* window, behind the table's: draft.h prepares it */

struct kernel_name {
    const char *function_name;
    uint32_t argument_byte_count;
};
static const struct kernel_name kernel_names[] = {
	/* the order of struct attention_state_kernels */
	{ KERNEL_ATTENTION_ROTATION, ATTENTION_ROTATION_ARGUMENT_BYTES },
	{ KERNEL_ATTENTION_PREPARE, ATTENTION_PREPARE_ARGUMENT_BYTES },
	{ KERNEL_ATTENTION_QUERIES, ATTENTION_QUERIES_ARGUMENT_BYTES },
	{ KERNEL_ATTENTION_COMPRESSOR, ATTENTION_COMPRESSOR_ARGUMENT_BYTES },
	{ KERNEL_ATTENTION_KEY, ATTENTION_KEY_ARGUMENT_BYTES },
};
_Static_assert(
	sizeof(struct attention_state_kernels) ==
		(COUNT_OF(kernel_names) + DRAFT_KERNELS) * sizeof(uint32_t),
	"one handle per kernel, in the order of the table, then the draft's");

result_status
attention_state_kernels_prepare(const struct backend_operations *backend,
				struct device_card *card,
				struct attention_state_kernels *kernels_out)
{
    uint32_t handles[COUNT_OF(kernel_names) + DRAFT_KERNELS];
    if (backend == NULL || backend->kernel_prepare == NULL || card == NULL || kernels_out == NULL) {
	    LOG_ERROR("the kernels of the attention state: a missing argument");
	    return STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t draft = 0; draft < DRAFT_KERNELS; draft++) {
	    handles[COUNT_OF(kernel_names) + draft] = BACKEND_NO_KERNEL;
    }
    for (uint32_t kernel_index = 0; kernel_index < COUNT_OF(kernel_names); kernel_index++) {
        const struct kernel_name *const kernel = &kernel_names[kernel_index];
	handles[kernel_index] =
		BACKEND_NO_KERNEL; /* a backend that answers OK and gives no handle */
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_ATTENTION_STATE, kernel->function_name,
		kernel->argument_byte_count, &handles[kernel_index]);
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the kernels of the attention state: %s is not prepared: %s",
			kernel->function_name, status_text(status));
		return status;
	}
    }
    memcpy(kernels_out, handles, sizeof(*kernels_out));
    return STATUS_OK;
}

/* most: LINEAR_MAXIMUM_ROWS, a step's (of qr alone and the queries alone, a chunk's:
 * MODEL_WIDE_MOST_ROWS). */
static bool rows_are_valid(const char *what, uint32_t row_count, uint32_t most)
{
    if (row_count == 0 || row_count > most) {
	    LOG_ERROR("%s: %" PRIu32 " rows: a launch takes 1 to %" PRIu32,
		      what, row_count, most);
	    return false;
    }
    return true;
}

static uint64_t step_byte_count(uint32_t row_count)
{
	return STEP_BLOCK_BYTES_VALUE +
	       (uint64_t)row_count * STEP_ROW_BYTES_VALUE;
}

result_status attention_rotation_prepare(
	const struct attention_rotation_description *description,
	uint64_t arena_byte_count,
	const struct attention_state_kernels *kernels,
	struct kernel_launch *launch_out)
{
    static const char what[] = "the rotation step";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t row_count = description->row_count;
    const struct checked_range ranges[] = {
	    { "the step area", &description->step, step_byte_count(row_count),
	      false },
	    { "the frequencies", &description->frequencies,
	      linear_sums_byte_count(MODEL_THETAS, MODEL_FREQUENCIES), false },
	    { "the rotation", &description->rotation,
	      linear_sums_byte_count(row_count, ROTATION_FLOATS), false },
    };
    if (!prepare_kernel_is_prepared(what, kernels->rotation,
				    KERNEL_ATTENTION_ROTATION) ||
	!prepare_reserved_is_zero(what, description->reserved_zero) ||
	!rows_are_valid(what, row_count, LINEAR_MAXIMUM_ROWS) ||
	!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct attention_rotation_arguments record = {
	    .step_offset = description->step.byte_offset,
	    .frequencies_offset = description->frequencies.byte_offset,
	    .rotation_offset = description->rotation.byte_offset
    };
    prepare_launch_fill(launch_out, kernels->rotation, row_count, 1,
			ATTENTION_ROTATION_THREADS, 0, &record, sizeof(record));
    return STATUS_OK;
}

/* The queries' launch of a description (attention_state_records.h), the state's and the queries alone's. */
static void
queries_launch_fill(struct kernel_launch *launch, uint32_t kernel,
		    const struct attention_queries_description *queries)
{
    const bool has_index = queries->has_index != 0, compact = queries->heads_compact != 0;
    const struct attention_queries_arguments record = {
	    .rotation_offset = queries->rotation.byte_offset,
	    .heads_offset = queries->heads.byte_offset,
	    .queries_offset = queries->queries.byte_offset,
	    .index_heads_offset = has_index ? queries->index_heads.byte_offset :
					      0,
	    .index_queries_offset =
		    has_index ? queries->index_queries.byte_offset : 0,
	    .first_head = queries->first_head,
	    .head_count = queries->head_count,
	    .index_head_count = has_index ? MODEL_INDEX_HEADS : 0,
	    .rotation_use = queries->theta == 0 ?
				    ATTENTION_ROTATION_WINDOW_ONLY :
				    ATTENTION_ROTATION_POOLED,
	    .heads_row_heads = compact ? queries->head_count :
					 MODEL_HEADS_VALUE,
	    .heads_first_head = compact ? 0 : queries->first_head
    };
    prepare_launch_fill(launch, kernel,
			queries->head_count + record.index_head_count,
			queries->row_count, ATTENTION_STATE_THREADS, 0, &record,
			sizeof(record));
}

/* The counts and choices of a layer, each refusal one line. */
static bool layer_is_valid(const char *what,
			   const struct attention_state_description *layer,
			   const struct attention_state_kernels *kernels)
{
    const uint32_t ratio = layer->pool_ratio;
    if (!rows_are_valid(what, layer->row_count, LINEAR_MAXIMUM_ROWS) ||
	!prepare_reserved_is_zero(what, layer->reserved_zero) ||
	!prepare_kernel_is_prepared(what, kernels->prepare,
				    KERNEL_ATTENTION_PREPARE) ||
	!prepare_kernel_is_prepared(what, kernels->queries,
				    KERNEL_ATTENTION_QUERIES) ||
	(ratio != 0 &&
	 !prepare_kernel_is_prepared(what, kernels->compressor,
				     KERNEL_ATTENTION_COMPRESSOR)) ||
	(ratio != 0 && !prepare_kernel_is_prepared(what, kernels->key,
						   KERNEL_ATTENTION_KEY))) {
	    return false;
    }
    if (layer->session_capacity == 0 || layer->context_token_limit == 0 ||
	layer->context_token_limit % MODEL_CONTEXT_GRANULE != 0 ||
	layer->head_count == 0 || layer->head_count > MODEL_HEADS_VALUE ||
	layer->first_head > MODEL_HEADS_VALUE - layer->head_count ||
	layer->theta > 1 || layer->has_index > 1 || ratio > TWO_TO_ONE ||
	(ratio != 0 && layer->has_index == 0) ||
	(layer->theta == 0 && (ratio != 0 || layer->has_index != 0))) {
	    LOG_ERROR(
		    "%s: %" PRIu32 " sessions of %" PRIu32 " tokens, %" PRIu32
		    " heads from head %" PRIu32 ", theta %" PRIu32
		    ", indexer %" PRIu32 ", pool ratio %" PRIu32
		    ": a session or more, tokens "
		    "a multiple of %u, heads 1 or more, all below %u; theta and indexer 0 or 1, a ratio up to "
		    "%u, a pool only with an indexer, neither of theta 0",
		    what, layer->session_capacity, layer->context_token_limit,
		    layer->head_count, layer->first_head, layer->theta,
		    layer->has_index, ratio, MODEL_CONTEXT_GRANULE,
		    MODEL_HEADS_VALUE, TWO_TO_ONE);
	    return false;
    }
    return true;
}

/* The compressor and key launches of a layer with a pool (its rows: one block each). */
static void pool_launches_fill(const struct attention_state_description *layer,
			       const struct attention_state_kernels *kernels,
			       struct kernel_launch *launches_out)
{
    const uint64_t ratio = layer->pool_ratio, entries = layer->context_token_limit / ratio;
    const struct attention_compressor_arguments compressor = {
	    .step_offset = layer->step.byte_offset,
	    .rotation_offset = layer->rotation.byte_offset,
	    .kv_offset = layer->pool_input.byte_offset,
	    .score_offset =
		    ratio == TWO_TO_ONE ? layer->pool_score.byte_offset : 0,
	    .slots_offset = ratio == TWO_TO_ONE ? layer->slots.byte_offset : 0,
	    .slots_session_stride = ATTENTION_SLOTS_SESSION_BYTES,
	    .norm_offset = layer->pool_norm.byte_offset,
	    .latent_offset = layer->latent.byte_offset,
	    .pool_offset = layer->pool.byte_offset,
	    .pool_session_stride = linear_values_byte_count(entries, HEAD_DIMS),
	    .ratio = layer->pool_ratio,
	    .session_capacity = layer->session_capacity,
	    .context_token_limit = layer->context_token_limit
    };
    const struct attention_key_arguments key = {
	    .step_offset = layer->step.byte_offset,
	    .rotation_offset = layer->rotation.byte_offset,
	    .key_input_offset = layer->key_input.byte_offset,
	    .norm_offset = layer->key_norm.byte_offset,
	    .keys_offset = layer->keys.byte_offset,
	    .keys_session_stride =
		    linear_values_byte_count(entries, INDEX_DIMS),
	    .ratio = layer->pool_ratio,
	    .session_capacity = layer->session_capacity,
	    .context_token_limit = layer->context_token_limit
    };
    prepare_launch_fill(&launches_out[ATTENTION_STATE_COMPRESSOR],
			kernels->compressor, layer->row_count, 1,
			ATTENTION_STATE_THREADS, 0, &compressor,
			sizeof(compressor));
    prepare_launch_fill(&launches_out[ATTENTION_STATE_KEY], kernels->key,
			layer->row_count, 1, ATTENTION_STATE_THREADS, 0, &key,
			sizeof(key));
}

result_status
attention_state_prepare(const struct attention_state_description *description,
			uint64_t arena_byte_count,
			const struct attention_state_kernels *kernels,
			struct kernel_launch *launches_out)
{
    static const char what[] = "the attention state of a layer";
    if (!prepare_arguments_are_given(what, description, kernels,
				     launches_out) ||
	!layer_is_valid(what, description, kernels)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct attention_state_description *const layer = description;
    const uint64_t rows = layer->row_count, sessions = layer->session_capacity, ratio = layer->pool_ratio;
    const uint64_t entries = ratio == 0 ? 0 : layer->context_token_limit / ratio;     /* of a session's pool */
    const uint64_t heads = (uint64_t)MODEL_HEADS_VALUE *
			   HEAD_DIMS; /* a row of the queries: every head */
    const uint64_t index_bytes = rows * MODEL_INDEX_QUERIES_ROW_BYTES;
    const struct checked_range ranges[] = {
	    { "the step area", &layer->step, step_byte_count(layer->row_count),
	      false },
	    { "the rotation", &layer->rotation,
	      linear_sums_byte_count(rows, ROTATION_FLOATS), false },
	    { "wq_a's output", &layer->query_rank,
	      linear_values_byte_count(rows, MODEL_QUERY_RANK), false },
	    { "q_norm", &layer->query_norm, MODEL_HALVES(MODEL_QUERY_RANK),
	      true },
	    { "qr", &layer->query_normed,
	      linear_values_byte_count(rows, MODEL_QUERY_RANK), false },
	    { "wkv's output", &layer->window_input,
	      linear_values_byte_count(rows, HEAD_DIMS), false },
	    { "kv_norm", &layer->window_norm, MODEL_HALVES(HEAD_DIMS), true },
	    { "the ring", &layer->ring,
	      linear_values_byte_count(sessions * MODEL_RING_SLOTS, HEAD_DIMS),
	      false },
	    { "wq_b's output", &layer->heads,
	      linear_values_byte_count(rows, heads), false },
	    { "the queries", &layer->queries,
	      linear_values_byte_count(rows, heads), false },
	    { "the indexer's wq_b output", &layer->index_heads, index_bytes,
	      false },
	    { "the index queries", &layer->index_queries, index_bytes, false },
	    { "the compressor's norm", &layer->pool_norm,
	      MODEL_HALVES(HEAD_DIMS), true },
	    { "the latent", &layer->latent,
	      linear_values_byte_count(rows, HEAD_DIMS), false },
	    { "the pool", &layer->pool,
	      linear_values_byte_count(sessions * entries, HEAD_DIMS), false },
	    { "wk's output", &layer->key_input,
	      linear_values_byte_count(rows, INDEX_DIMS), false },
	    { "k_norm", &layer->key_norm, MODEL_HALVES(INDEX_DIMS), true },
	    { "the keys", &layer->keys,
	      linear_values_byte_count(sessions * entries, INDEX_DIMS), false },
	    { "the pooled product", &layer->pool_input,
	      ratio == TWO_TO_ONE ? linear_sums_byte_count(rows, HEAD_DIMS) :
				    linear_values_byte_count(rows, HEAD_DIMS),
	      false },
	    { "the pooled score", &layer->pool_score,
	      linear_sums_byte_count(rows, HEAD_DIMS), false },
	    { "the compressor's slots", &layer->slots,
	      linear_sums_byte_count(sessions * ATTENTION_COMPRESSOR_SLOTS,
				     SLOT_FLOATS),
	      false },
    };
    _Static_assert(COUNT_OF(ranges) == RANGES, "the ranges of every kind of layer are a prefix of the list");
    const uint32_t range_count = (uint32_t)(ratio == TWO_TO_ONE ? RANGES : ratio != 0 ? RATIO_ONE_RANGES
                                            : layer->has_index != 0 ? INDEX_RANGES : WINDOW_RANGES);
    if (!prepare_ranges_are_valid(what, ranges, range_count,
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t use = layer->theta == 0 ? ATTENTION_ROTATION_WINDOW_ONLY :
					     ATTENTION_ROTATION_POOLED;
    const struct attention_prepare_arguments prepare = {
	    .step_offset = layer->step.byte_offset,
	    .rotation_offset = layer->rotation.byte_offset,
	    .query_rank_offset = layer->query_rank.byte_offset,
	    .query_norm_offset = layer->query_norm.byte_offset,
	    .query_normed_offset = layer->query_normed.byte_offset,
	    .window_input_offset = layer->window_input.byte_offset,
	    .window_norm_offset = layer->window_norm.byte_offset,
	    .ring_offset = layer->ring.byte_offset,
	    .ring_session_stride =
		    linear_values_byte_count(MODEL_RING_SLOTS, HEAD_DIMS),
	    .rotation_use = use,
	    .session_capacity = layer->session_capacity,
	    .context_token_limit = layer->context_token_limit
    };
    const struct attention_queries_description queries = { layer->rotation,
							   layer->heads,
							   layer->queries,
							   layer->index_heads,
							   layer->index_queries,
							   layer->row_count,
							   layer->theta,
							   layer->first_head,
							   layer->head_count,
							   layer->has_index,
							   0 };
    memset(launches_out, 0,
	   ATTENTION_STATE_LAUNCH_COUNT * sizeof(*launches_out));
    prepare_launch_fill(&launches_out[ATTENTION_STATE_PREPARE],
			kernels->prepare, layer->row_count,
			ATTENTION_PREPARE_PARTS, ATTENTION_STATE_THREADS, 0,
			&prepare, sizeof(prepare));
    queries_launch_fill(&launches_out[ATTENTION_STATE_QUERIES],
			kernels->queries, &queries);
    if (ratio != 0)
        pool_launches_fill(layer, kernels, launches_out);
    return STATUS_OK;
}

result_status
attention_pool_prepare(const struct attention_state_description *description,
		       uint64_t arena_byte_count,
		       const struct attention_state_kernels *kernels,
		       struct kernel_launch *launches_out)
{
    static const char what[] = "the pool entries and keys of a chunk";
    if (!prepare_arguments_are_given(what, description, kernels,
				     launches_out) ||
	!rows_are_valid(what, description->row_count, MODEL_WIDE_MOST_ROWS) ||
	!prepare_reserved_is_zero(what, description->reserved_zero) ||
	!prepare_kernel_is_prepared(what, kernels->compressor,
				    KERNEL_ATTENTION_COMPRESSOR) ||
	!prepare_kernel_is_prepared(what, kernels->key, KERNEL_ATTENTION_KEY)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct attention_state_description *const layer = description;
    const uint64_t rows = layer->row_count, sessions = layer->session_capacity, ratio = layer->pool_ratio;
    if (sessions == 0 || layer->context_token_limit == 0 ||
	layer->context_token_limit % MODEL_CONTEXT_GRANULE != 0 || ratio == 0 ||
	ratio > TWO_TO_ONE || layer->theta != 1 || layer->has_index != 1) {
	    LOG_ERROR(
		    "%s: %" PRIu64 " sessions of %" PRIu32
		    " tokens, pool ratio %" PRIu64 ", theta %" PRIu32
		    ", indexer %" PRIu32
		    ": a session or more, tokens a multiple of %u, a ratio of 1 or 2, theta "
		    "and indexer 1",
		    what, sessions, layer->context_token_limit, ratio,
		    layer->theta, layer->has_index, MODEL_CONTEXT_GRANULE);
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t entries = layer->context_token_limit / ratio;
    const struct checked_range ranges[] = {
	    { "the step area", &layer->step, step_byte_count(layer->row_count),
	      false },
	    { "the rotation", &layer->rotation,
	      linear_sums_byte_count(rows, ROTATION_FLOATS), false },
	    { "the compressor's norm", &layer->pool_norm,
	      MODEL_HALVES(HEAD_DIMS), true },
	    { "the latent", &layer->latent,
	      linear_values_byte_count(rows, HEAD_DIMS), false },
	    { "the pool", &layer->pool,
	      linear_values_byte_count(sessions * entries, HEAD_DIMS), false },
	    { "wk's output", &layer->key_input,
	      linear_values_byte_count(rows, INDEX_DIMS), false },
	    { "k_norm", &layer->key_norm, MODEL_HALVES(INDEX_DIMS), true },
	    { "the keys", &layer->keys,
	      linear_values_byte_count(sessions * entries, INDEX_DIMS), false },
	    { "the pooled product", &layer->pool_input,
	      ratio == TWO_TO_ONE ? linear_sums_byte_count(rows, HEAD_DIMS) :
				    linear_values_byte_count(rows, HEAD_DIMS),
	      false },
	    { "the pooled score", &layer->pool_score,
	      linear_sums_byte_count(rows, HEAD_DIMS), false },
	    { "the compressor's slots", &layer->slots,
	      linear_sums_byte_count(sessions * ATTENTION_COMPRESSOR_SLOTS,
				     SLOT_FLOATS),
	      false },
    };
    if (!prepare_ranges_are_valid(what, ranges,
				  ratio == TWO_TO_ONE ? COUNT_OF(ranges) :
							POOL_ONE_RANGES,
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    memset(launches_out, 0,
	   ATTENTION_STATE_LAUNCH_COUNT * sizeof(*launches_out));
    pool_launches_fill(layer, kernels, launches_out);
    return STATUS_OK;
}

result_status attention_queries_prepare(
	const struct attention_queries_description *description,
	uint64_t arena_byte_count,
	const struct attention_state_kernels *kernels,
	struct kernel_launch *launch_out)
{
    static const char what[] = "the queries alone";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = description->row_count,
		   heads = (uint64_t)MODEL_HEADS_VALUE * HEAD_DIMS;
    const uint64_t read_heads = description->heads_compact != 0 ?
					description->head_count :
					MODEL_HEADS_VALUE;
    const struct checked_range ranges[] = {
	    { "the rotation", &description->rotation,
	      linear_sums_byte_count(rows, ROTATION_FLOATS), false },
	    { "wq_b's output", &description->heads,
	      linear_values_byte_count(rows, read_heads * HEAD_DIMS), false },
	    { "the queries", &description->queries,
	      linear_values_byte_count(rows, heads), false },
	    { "the indexer's wq_b output", &description->index_heads,
	      rows * MODEL_INDEX_QUERIES_ROW_BYTES, false },
	    { "the index queries", &description->index_queries,
	      rows * MODEL_INDEX_QUERIES_ROW_BYTES, false },
    };
    if (!prepare_kernel_is_prepared(what, kernels->queries,
				    KERNEL_ATTENTION_QUERIES) ||
	!rows_are_valid(what, description->row_count, MODEL_WIDE_MOST_ROWS)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (description->head_count == 0 ||
	description->head_count > MODEL_HEADS_VALUE ||
	description->first_head > MODEL_HEADS_VALUE - description->head_count ||
	description->theta > 1 || description->has_index > 1 ||
	(description->theta == 0 && description->has_index != 0) ||
	description->heads_compact > 1) {
	    LOG_ERROR(
		    "%s: %" PRIu32 " heads from head %" PRIu32
		    ", theta %" PRIu32 ", indexer %" PRIu32 ", compact %" PRIu32
		    ": heads 1 or more, all below %u; theta, indexer and compact 0 or 1, "
		    "no indexer of theta 0",
		    what, description->head_count, description->first_head,
		    description->theta, description->has_index,
		    description->heads_compact, MODEL_HEADS_VALUE);
	    return STATUS_INVALID_ARGUMENT;
    }
    if (!prepare_ranges_are_valid(what, ranges,
				  description->has_index != 0 ?
					  COUNT_OF(ranges) :
					  QUERY_RANGES_WITHOUT_INDEX,
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    queries_launch_fill(launch_out, kernels->queries, description);
    return STATUS_OK;
}

result_status attention_query_norm_prepare(
	const struct attention_query_norm_description *description,
	uint64_t arena_byte_count,
	const struct attention_state_kernels *kernels,
	struct kernel_launch *launch_out)
{
    static const char what[] = "qr alone";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t row_count = description->row_count;
    const uint64_t rank_bytes =
	    linear_values_byte_count(row_count, MODEL_QUERY_RANK);
    const struct checked_range ranges[] = {
	    { "wq_a's output", &description->query_rank, rank_bytes, false },
	    { "q_norm", &description->query_norm,
	      MODEL_HALVES(MODEL_QUERY_RANK), true },
	    { "qr", &description->query_normed, rank_bytes, false },
    };
    if (!prepare_kernel_is_prepared(what, kernels->prepare,
				    KERNEL_ATTENTION_PREPARE) ||
	!prepare_reserved_is_zero(what, description->reserved_zero) ||
	!rows_are_valid(what, row_count, MODEL_WIDE_MOST_ROWS) ||
	!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    /* the prepare's grid y 0 alone: its other offsets are not read there */
    const struct attention_prepare_arguments record = {
	    .query_rank_offset = description->query_rank.byte_offset,
	    .query_norm_offset = description->query_norm.byte_offset,
	    .query_normed_offset = description->query_normed.byte_offset
    };
    prepare_launch_fill(launch_out, kernels->prepare, row_count, 1,
			ATTENTION_STATE_THREADS, 0, &record, sizeof(record));
    return STATUS_OK;
}

result_status
attention_window_prepare(const struct attention_window_description *description,
			 uint64_t arena_byte_count,
			 const struct attention_state_kernels *kernels,
			 struct kernel_launch *launch_out)
{
    static const char what[] = "the window alone";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t row_count = description->row_count;
    const uint64_t rows = row_count, sessions = description->session_capacity;
    const struct checked_range ranges[] = {
	    { "the step area", &description->step, step_byte_count(row_count),
	      false },
	    { "the rotation", &description->rotation,
	      linear_sums_byte_count(rows, ROTATION_FLOATS), false },
	    { "wkv's output", &description->window_input,
	      linear_values_byte_count(rows, HEAD_DIMS), false },
	    { "kv_norm", &description->window_norm, MODEL_HALVES(HEAD_DIMS),
	      true },
	    { "the ring", &description->ring,
	      linear_values_byte_count(sessions * MODEL_RING_SLOTS, HEAD_DIMS),
	      false },
    };
    if (!prepare_kernel_is_prepared(what, kernels->window,
				    KERNEL_ATTENTION_WINDOW) ||
	!prepare_reserved_is_zero(what, description->reserved_zero) ||
	!rows_are_valid(what, row_count, MODEL_WIDE_MOST_ROWS)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (description->session_capacity == 0 ||
	description->context_token_limit == 0 ||
	description->context_token_limit % MODEL_CONTEXT_GRANULE != 0 ||
	description->rotation_use > ATTENTION_ROTATION_GROUP_START) {
	    LOG_ERROR(
		    "%s: %" PRIu32 " sessions of %" PRIu32
		    " tokens, rotation use %" PRIu32
		    ": a session or more, tokens a multiple of %u, a use up to %u",
		    what, description->session_capacity,
		    description->context_token_limit, description->rotation_use,
		    MODEL_CONTEXT_GRANULE,
		    (uint32_t)ATTENTION_ROTATION_GROUP_START);
	    return STATUS_INVALID_ARGUMENT;
    }
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    /* the prepare's window part alone: its qr offsets are not read there */
    const struct attention_prepare_arguments record = {
	    .step_offset = description->step.byte_offset,
	    .rotation_offset = description->rotation.byte_offset,
	    .window_input_offset = description->window_input.byte_offset,
	    .window_norm_offset = description->window_norm.byte_offset,
	    .ring_offset = description->ring.byte_offset,
	    .ring_session_stride =
		    linear_values_byte_count(MODEL_RING_SLOTS, HEAD_DIMS),
	    .rotation_use = description->rotation_use,
	    .session_capacity = description->session_capacity,
	    .context_token_limit = description->context_token_limit
    };
    prepare_launch_fill(launch_out, kernels->window, row_count, 1,
			ATTENTION_STATE_THREADS, 0, &record, sizeof(record));
    return STATUS_OK;
}
