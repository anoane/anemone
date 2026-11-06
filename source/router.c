/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router.c: the launches of the router. Contract: router.h; the kernels and their records: router_records.h. As
 * linear.c: the counts first, then the ranges against the arena, then the table inside the experts' range (once the
 * ranges passed, that range holds the larger matrix: the last offsets do not wrap), then the launches; nothing is
 * written before every check has passed.
 */
#include "anemone/router.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/weight_format_seam.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define PICKS MODEL_EXPERTS_PER_ROW
#define DRAFT_KERNELS 1u                        /* select_draft, behind the table's: draft.h prepares it */

struct kernel_name {
    const char *function_name;
    uint32_t argument_byte_count;
};
static const struct kernel_name kernel_names[] = {
	/* the order of struct router_kernels_record */
	{ KERNEL_ROUTER_SELECT, ROUTER_SELECT_ARGUMENT_BYTES },
	{ KERNEL_ROUTER_WIDE_SELECT, ROUTER_WIDE_ARGUMENT_BYTES },
	{ KERNEL_ROUTER_WIDE_COUNT, ROUTER_WIDE_ARGUMENT_BYTES },
	{ KERNEL_ROUTER_WIDE_PLACE, ROUTER_WIDE_ARGUMENT_BYTES },
	{ KERNEL_ROUTER_WIDE_JOBS, ROUTER_WIDE_ARGUMENT_BYTES },
	{ KERNEL_ROUTER_WIDE_SMALL, ROUTER_WIDE_ARGUMENT_BYTES },
};
_Static_assert(
	sizeof(struct router_kernels_record) ==
		(COUNT_OF(kernel_names) + DRAFT_KERNELS) * sizeof(uint32_t),
	"one handle per kernel, in the order of the table, then the draft's");

result_status router_kernels_prepare(const struct backend_operations *backend,
				     struct device_card *card,
				     struct router_kernels_record *kernels_out)
{
    uint32_t handles[COUNT_OF(kernel_names) + DRAFT_KERNELS];
    if (backend == NULL || backend->kernel_prepare == NULL || card == NULL || kernels_out == NULL) {
	    LOG_ERROR("the kernels of the router: a missing argument");
	    return STATUS_INVALID_ARGUMENT;
    }
    handles[COUNT_OF(kernel_names)] = BACKEND_NO_KERNEL;
    for (uint32_t index = 0; index < COUNT_OF(kernel_names); index++) {
	    handles[index] =
		    BACKEND_NO_KERNEL; /* a backend that answers OK with no handle leaves this one */
	    const result_status status = backend->kernel_prepare(
		    card, KERNEL_FILE_ROUTER, kernel_names[index].function_name,
		    kernel_names[index].argument_byte_count, &handles[index]);
	    if (status != STATUS_OK) {
		    LOG_ERROR(
			    "the kernels of the router: %s is not prepared: %s",
			    kernel_names[index].function_name,
			    status_text(status));
		    return status;
	    }
    }
    memcpy(kernels_out, handles, sizeof(*kernels_out));
    return STATUS_OK;
}

/* The rows (of most_rows at most, of pick_count pairs), format, shapes and order; and the bytes of a gate (or up) and
 * of a down matrix. most_order: the last order taken (the draft's router takes the fixed one alone). */
static bool counts_are_valid(const char *what, bool rows_are_valid,
			     uint32_t most_rows, uint32_t pick_count,
			     enum weight_format_code weight_format,
			     const struct matrix_shape_record *gate_up_shape,
			     const struct matrix_shape_record *down_shape,
			     enum linear_order_code order,
			     enum linear_order_code most_order,
			     uint64_t *gate_up_byte_count,
			     uint64_t *down_byte_count)
{
	const struct weight_format_operations *const format =
		weight_format_operations_of(weight_format);
	if (!rows_are_valid ||
	    (weight_format != WEIGHT_FORMAT_E2M1_SCALED &&
	     weight_format != WEIGHT_FORMAT_EXL3) ||
	    format == NULL ||
	    format->stored_byte_count(gate_up_shape, gate_up_byte_count) !=
		    STATUS_OK ||
	    format->stored_byte_count(down_shape, down_byte_count) !=
		    STATUS_OK ||
	    (uint32_t)order > most_order) {
		LOG_ERROR(
			"%s: the router writes 1 to %u rows of %u pairs (the select a job a pair), for matrices of "
			"format %u in shapes it stores, in order 0 to %u",
			what, most_rows, pick_count,
			(uint32_t)WEIGHT_FORMAT_E2M1_SCALED,
			(uint32_t)most_order);
		return false;
	}
    return true;
}

/* Every matrix of the table's first expert_count experts on a multiple of the alignment, from the experts' first byte
 * to the last it may begin at (the kernels' checked modules hold the arena's copy to the same). */
static bool offsets_are_valid(const char *what, const uint64_t *expert_offsets, uint32_t expert_count,
                              uint64_t weights_offset, uint64_t gate_up_last_offset, uint64_t down_last_offset)
{
	for (uint32_t index = 0; index < expert_count * MODEL_EXPERT_MATRICES;
	     index++) {
		const uint64_t offset = expert_offsets[index];
		const bool is_down = index % MODEL_EXPERT_MATRICES ==
				     ROUTER_DOWN;
		const uint64_t last = is_down ? down_last_offset :
						gate_up_last_offset;
		if (offset % LINEAR_OFFSET_ALIGNMENT_BYTES != 0 ||
		    offset < weights_offset || offset > last) {
			LOG_ERROR(
				"%s: entry %" PRIu32
				" of the table, byte %" PRIu64
				", is no multiple of %u whose "
				"matrix lies wholly inside the range of the experts' matrices",
				what, index, offset,
				LINEAR_OFFSET_ALIGNMENT_BYTES);
			return false;
		}
	}
    return true;
}

result_status
router_select_prepare(const struct router_description *description,
		      uint64_t arena_byte_count,
		      const struct router_kernels_record *kernels,
		      struct kernel_launch *launch_out)
{
    static const char what[] = "the router";
    uint64_t gate_up_byte_count = 0;
    uint64_t down_byte_count = 0;
    if (!prepare_arguments_are_given(what, description, kernels, launch_out) ||
	!prepare_arguments_are_given(what, description->experts,
				     description->expert_offsets, launch_out) ||
	!prepare_reserved_is_zero(what, description->experts->reserved_zero)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t expert_count = description->expert_count, pick_count = description->pick_count;
    const bool is_draft = expert_count == MODEL_DRAFT_EXPERTS &&
			  pick_count == MODEL_DRAFT_EXPERTS_PER_ROW;
    if (!is_draft && (expert_count != MODEL_EXPERTS || pick_count != PICKS)) {
	    LOG_ERROR("%s: %" PRIu32 " experts of %" PRIu32
		      " picks a row: the model's %u of %u or the draft's %u "
		      "of %u",
		      what, expert_count, pick_count, MODEL_EXPERTS, PICKS,
		      MODEL_DRAFT_EXPERTS, MODEL_DRAFT_EXPERTS_PER_ROW);
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t kernel = is_draft ? kernels->select_draft : kernels->select;
    const struct linear_routed_experts_description *const experts =
	    description->experts;
    if (!prepare_kernel_is_prepared(what, kernel,
				    is_draft ? KERNEL_ROUTER_SELECT_DRAFT :
					       KERNEL_ROUTER_SELECT) ||
	!counts_are_valid(
		what,
		experts->row_count != 0 &&
			experts->row_count <= LINEAR_MAXIMUM_ROWS &&
			experts->pairs_per_row == pick_count &&
			experts->job_count == experts->row_count * pick_count,
		LINEAR_MAXIMUM_ROWS, pick_count, experts->weight_format,
		&experts->gate_up_shape, &experts->down_shape,
		description->order,
		is_draft ? LINEAR_ORDER_FIXED : LINEAR_ORDER_REFERENCE,
		&gate_up_byte_count, &down_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t expert_bytes = (uint64_t)expert_count * sizeof(float);
    const uint64_t rows_byte_count = (uint64_t)experts->row_count * expert_bytes;
    const struct checked_range ranges[] = {
	    { "the step block", &description->step, STEP_BLOCK_BYTES_VALUE,
	      false },
	    { "the router scores", &description->scores, rows_byte_count,
	      false },
	    { "the bias", &description->bias, expert_bytes, false },
	    { "the expert offsets", &description->offsets,
	      (uint64_t)expert_count * MODEL_EXPERT_MATRICES * sizeof(uint64_t),
	      false },
	    { "the affinities", &description->affinities, rows_byte_count,
	      false },
	    { "the job table", &experts->jobs,
	      linear_jobs_byte_count(experts->job_count), false },
	    { "the experts' matrices", &experts->expert_weights,
	      gate_up_byte_count > down_byte_count ? gate_up_byte_count :
						     down_byte_count,
	      false },
	    { "the rows' codes", &experts->codes, 0,
	      false }, /* the other nine: no byte shared */
	    { "the rows' exponents", &experts->exponents, 0, false },
	    { "the rows' operands", &experts->operands, 0, false },
	    { "the hidden values", &experts->hidden, 0, false },
	    { "the hidden codes", &experts->hidden_codes, 0, false },
	    { "the hidden exponents", &experts->hidden_exponents, 0, false },
	    { "the hidden operands", &experts->hidden_operands, 0, false },
	    { "the pair outputs", &experts->pair_output, 0, false },
	    { "the row outputs", &experts->row_output, 0, false },
    };
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t weights_end = experts->expert_weights.byte_offset + experts->expert_weights.byte_count;
    const struct router_select_arguments record = {
	    .step_offset = description->step.byte_offset,
	    .scores_offset = description->scores.byte_offset,
	    .bias_offset = description->bias.byte_offset,
	    .expert_offsets_offset = description->offsets.byte_offset,
	    .affinities_offset = description->affinities.byte_offset,
	    .jobs_offset = experts->jobs.byte_offset,
	    .weights_offset = experts->expert_weights.byte_offset,
	    .gate_up_last_offset = weights_end - gate_up_byte_count,
	    .down_last_offset = weights_end - down_byte_count,
	    .row_count = experts->row_count,
	    .reference_order =
		    description->order == LINEAR_ORDER_REFERENCE ? 1u : 0u,
    };
    if (!offsets_are_valid(what, description->expert_offsets, expert_count, record.weights_offset,
                           record.gate_up_last_offset, record.down_last_offset)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    prepare_launch_fill(launch_out, kernel, 1, 1, ROUTER_THREADS, 0, &record,
			ROUTER_SELECT_ARGUMENT_BYTES);
    return STATUS_OK;
}

/* The wide select's checks and record (both preparations'); false after an error line. */
static bool wide_record_of(const struct router_wide_description *description,
			   uint64_t arena_byte_count,
			   const struct router_kernels_record *kernels,
			   struct kernel_launch *launches_out,
			   struct router_wide_arguments *record_out)
{
    static const char what[] = "the router's wide select";
    uint64_t gate_up_byte_count = 0;
    uint64_t down_byte_count = 0;
    if (!prepare_arguments_are_given(what, description, kernels,
				     launches_out) ||
	!prepare_arguments_are_given(what, description->expert_offsets, kernels,
				     launches_out) ||
	!prepare_kernel_is_prepared(what, kernels->wide_select,
				    KERNEL_ROUTER_WIDE_SELECT) ||
	!prepare_kernel_is_prepared(what, kernels->wide_count,
				    KERNEL_ROUTER_WIDE_COUNT) ||
	!prepare_kernel_is_prepared(what, kernels->wide_place,
				    KERNEL_ROUTER_WIDE_PLACE) ||
	!prepare_kernel_is_prepared(what, kernels->wide_jobs,
				    KERNEL_ROUTER_WIDE_JOBS) ||
	!prepare_reserved_is_zero(what, description->reserved_zero) ||
	!counts_are_valid(
		what,
		description->row_count != 0 &&
			description->row_count <= MODEL_WIDE_MOST_ROWS,
		MODEL_WIDE_MOST_ROWS, PICKS, description->weight_format,
		&description->gate_up_shape, &description->down_shape,
		description->order, LINEAR_ORDER_REFERENCE, &gate_up_byte_count,
		&down_byte_count)) {
	    return false;
    }
    const uint32_t row_count = description->row_count;
    const uint32_t sort_blocks =
	    (row_count + ROUTER_SORT_ROWS - 1u) / ROUTER_SORT_ROWS;
    const uint64_t rows_byte_count =
	    (uint64_t)row_count * MODEL_ROUTER_ROW_BYTES;
    const uint64_t pair_count = (uint64_t)row_count * PICKS;
    const struct checked_range ranges[] = {
	    { "the step block", &description->step, STEP_BLOCK_BYTES_VALUE,
	      false },
	    { "the router scores", &description->scores, rows_byte_count,
	      false },
	    { "the bias", &description->bias, MODEL_ROUTER_ROW_BYTES, false },
	    { "the expert offsets", &description->offsets,
	      MODEL_EXPERT_OFFSETS_BYTES, false },
	    { "the experts' matrices", &description->expert_weights,
	      gate_up_byte_count > down_byte_count ? gate_up_byte_count :
						     down_byte_count,
	      false },
	    { "the affinities", &description->affinities, rows_byte_count,
	      false },
	    { "the picks", &description->picks, pair_count * ROUTER_PICK_BYTES,
	      false },
	    { "the counts", &description->counts,
	      (uint64_t)sort_blocks * MODEL_EXPERTS * sizeof(uint32_t), false },
	    { "the sorted pairs", &description->pairs,
	      pair_count * WIDE_PAIR_BYTES, false },
	    { "the runs", &description->runs,
	      (uint64_t)MODEL_EXPERTS * WIDE_RUN_BYTES, false },
	    { "the expert tiles", &description->tiles,
	      (uint64_t)ROUTER_WIDE_TILES(row_count) * WIDE_EXPERT_TILE_BYTES,
	      false },
	    { "the jobs", &description->jobs,
	      (uint64_t)ROUTER_WIDE_JOBS(row_count) * EXPERT_JOB_BYTES, false },
    };
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return false;
    }
    const uint64_t weights_end = description->expert_weights.byte_offset + description->expert_weights.byte_count;
    const struct router_wide_arguments record = {
	    .step_offset = description->step.byte_offset,
	    .scores_offset = description->scores.byte_offset,
	    .bias_offset = description->bias.byte_offset,
	    .expert_offsets_offset = description->offsets.byte_offset,
	    .affinities_offset = description->affinities.byte_offset,
	    .picks_offset = description->picks.byte_offset,
	    .counts_offset = description->counts.byte_offset,
	    .pairs_offset = description->pairs.byte_offset,
	    .runs_offset = description->runs.byte_offset,
	    .tiles_offset = description->tiles.byte_offset,
	    .jobs_offset = description->jobs.byte_offset,
	    .weights_offset = description->expert_weights.byte_offset,
	    .gate_up_last_offset = weights_end - gate_up_byte_count,
	    .down_last_offset = weights_end - down_byte_count,
	    .row_count = row_count,
	    .reference_order =
		    description->order == LINEAR_ORDER_REFERENCE ? 1u : 0u,
    };
    *record_out = record;
    return offsets_are_valid(what, description->expert_offsets, MODEL_EXPERTS,
			     record.weights_offset, record.gate_up_last_offset,
			     record.down_last_offset);
}

result_status
router_wide_prepare(const struct router_wide_description *description,
		    uint64_t arena_byte_count,
		    const struct router_kernels_record *kernels,
		    struct kernel_launch *launches_out)
{
	struct router_wide_arguments record;
	if (!wide_record_of(description, arena_byte_count, kernels,
			    launches_out, &record))
		return STATUS_INVALID_ARGUMENT;
	const uint32_t row_count = description->row_count;
	const uint32_t sort_blocks =
		(row_count + ROUTER_SORT_ROWS - 1u) / ROUTER_SORT_ROWS;
	prepare_launch_fill(
		&launches_out[ROUTER_WIDE_SELECT_LAUNCH], kernels->wide_select,
		1, (row_count + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS,
		ROUTER_THREADS, 0, &record, ROUTER_WIDE_ARGUMENT_BYTES);
	prepare_launch_fill(&launches_out[ROUTER_WIDE_COUNT_LAUNCH],
			    kernels->wide_count, 1, sort_blocks,
			    ROUTER_SORT_ROWS, 0, &record,
			    ROUTER_WIDE_ARGUMENT_BYTES);
	prepare_launch_fill(&launches_out[ROUTER_WIDE_PLACE_LAUNCH],
			    kernels->wide_place, 1, sort_blocks,
			    ROUTER_SORT_ROWS, 0, &record,
			    ROUTER_WIDE_ARGUMENT_BYTES);
	prepare_launch_fill(&launches_out[ROUTER_WIDE_JOBS_LAUNCH],
			    kernels->wide_jobs, 1, 1, ROUTER_SORT_ROWS, 0,
			    &record, ROUTER_WIDE_ARGUMENT_BYTES);
	return STATUS_OK;
}

result_status
router_wide_small_prepare(const struct router_wide_description *description,
			  uint64_t arena_byte_count,
			  const struct router_kernels_record *kernels,
			  struct kernel_launch *launches_out)
{
	struct router_wide_arguments record;
	if (description == NULL || description->row_count > ROUTER_SORT_ROWS ||
	    kernels == NULL ||
	    !prepare_kernel_is_prepared("the router's small wide select",
					kernels->wide_small,
					KERNEL_ROUTER_WIDE_SMALL) ||
	    !wide_record_of(description, arena_byte_count, kernels,
			    launches_out, &record))
		return STATUS_INVALID_ARGUMENT;
	prepare_launch_fill(&launches_out[0], kernels->wide_select, 1,
			    (description->row_count + LINEAR_MAXIMUM_ROWS -
			     1u) / LINEAR_MAXIMUM_ROWS,
			    ROUTER_THREADS, 0, &record,
			    ROUTER_WIDE_ARGUMENT_BYTES);
	prepare_launch_fill(&launches_out[1], kernels->wide_small, 1, 1,
			    ROUTER_SORT_ROWS, 0, &record,
			    ROUTER_WIDE_ARGUMENT_BYTES);
	return STATUS_OK;
}
