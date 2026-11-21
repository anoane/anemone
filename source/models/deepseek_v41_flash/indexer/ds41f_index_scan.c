/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_index_scan.c: the launches of the index scan and its selection, prepared from a description. Contract:
 * ds41f_index_scan.h; the kernels' rules: ds41f_index_scan_records.h. As linear.c: the counts, then the ranges against
 * what the launches touch and the arena (prepare_checks.h), then the launches; nothing is written before every check
 * passed. Once the counts pass, every byte count fits 64 bits: at most 4,096 rows of 2^20 entries, sessions a 32-bit
 * count.
 */
#include "ds41f_index_scan.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#include "common/hooks.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "engine/program/step_records.h"
#include "engine/library/prepare_checks.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define SOURCE_SELECTIONS 2u

/* The kernels in the order they are prepared: the order of the members of struct ds41f_index_scan_kernels. */
static const struct {
    const char *function_name;
    uint32_t argument_byte_count;
} kernel_names[] = {
	{ DS41F_KERNEL_INDEX_SCAN, DS41F_INDEX_SCAN_ARGUMENT_BYTES },
	{ DS41F_KERNEL_INDEX_SELECT, DS41F_INDEX_SELECT_ARGUMENT_BYTES },
	{ DS41F_KERNEL_INDEX_CANDIDATES, DS41F_INDEX_CANDIDATES_ARGUMENT_BYTES },
	{ DS41F_KERNEL_INDEX_SPLIT_HISTOGRAM_HIGH, DS41F_INDEX_SPLIT_ARGUMENT_BYTES },
	{ DS41F_KERNEL_INDEX_SPLIT_CUT_HIGH, DS41F_INDEX_SPLIT_ARGUMENT_BYTES },
	{ DS41F_KERNEL_INDEX_SPLIT_HISTOGRAM_LOW, DS41F_INDEX_SPLIT_ARGUMENT_BYTES },
	{ DS41F_KERNEL_INDEX_SPLIT_CUT_LOW, DS41F_INDEX_SPLIT_ARGUMENT_BYTES },
	{ DS41F_KERNEL_INDEX_SPLIT_OUTPUT, DS41F_INDEX_SPLIT_ARGUMENT_BYTES }
};

_Static_assert(
	sizeof(struct ds41f_index_scan_kernels) ==
			COUNT_OF(kernel_names) * sizeof(uint32_t) &&
		DS41F_INDEX_SCAN_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES,
	"a handle for each kernel, in their order; the records fit a launch");

result_status
ds41f_index_scan_kernels_prepare(const struct backend_operations *backend,
			   struct device_card *card, uint32_t arithmetic_class,
			   struct ds41f_index_scan_kernels *kernels_out)
{
	const char *const file_name =
		arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			DS41F_MODULE_INDEX_SCAN_NATIVE :
			DS41F_MODULE_INDEX_SCAN;
	uint32_t handles[COUNT_OF(kernel_names)];
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR("the kernels of the index scan: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
    for (uint32_t kernel = 0; kernel < COUNT_OF(kernel_names); kernel++) {
        /* a backend that answers OK and gives no handle leaves NO_KERNEL, which the prepare function refuses */
	handles[kernel] = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, file_name, kernel_names[kernel].function_name,
		kernel_names[kernel].argument_byte_count, &handles[kernel]);
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the kernels of the index scan: %s is not prepared: %s",
			kernel_names[kernel].function_name,
			status_text(status));
		return status;
	}
    }
    struct ds41f_index_scan_kernels prepared = {
	    .scan = handles[DS41F_INDEX_SCAN_LAUNCH_SCAN],
	    .select = handles[DS41F_INDEX_SCAN_LAUNCH_SELECT],
	    .candidates = handles[DS41F_INDEX_SCAN_LAUNCH_CANDIDATES]
    };
    for (uint32_t stage = 0; stage < DS41F_INDEX_SPLIT_LAUNCH_COUNT; stage++) {
	    prepared.split[stage] =
		    handles[DS41F_INDEX_SCAN_LEGACY_LAUNCH_COUNT + stage];
    }
    *kernels_out = prepared;
    return STATUS_OK;
}

/* The counts of ds41f_index_scan.h, refused together by one error line that states them all. */
static bool counts_are_valid(const char *what,
			     const struct ds41f_index_scan_description *description)
{
    const uint32_t ratio = description->ratio, context = description->context_token_limit;
    const uint32_t role = description->candidate_role;
    if (description->row_count == 0 ||
	description->row_count > STEP_WIDE_MOST_ROWS ||
	description->step_row_first >
		STEP_WIDE_MOST_ROWS - description->row_count ||
	description->split_rows > STEP_WIDE_MOST_ROWS ||
	(ratio != DS41F_DIRECT_RATIO && ratio != DS41F_POOLED_RATIO) ||
	context == 0 || context % DS41F_CONTEXT_GRANULE != 0 ||
	context > DS41F_CONTEXT_MOST || description->session_capacity == 0 ||
	role > DS41F_INDEX_CANDIDATES_READER ||
	(role != DS41F_INDEX_CANDIDATES_NONE && ratio != DS41F_DIRECT_RATIO)) {
	    LOG_ERROR(
		    "%s: %" PRIu32 " rows from row %" PRIu32
		    " (split shape %" PRIu32 "), ratio %" PRIu32 ", %" PRIu32
		    " tokens, %" PRIu32 " sessions, candidate role %" PRIu32
		    ": 1 to %u rows; ratio"
		    " %u or %u; whole granules of %u tokens, at most %u; one session or more; a candidate role"
		    " of ratio %u; the rows from the first and the split shape within the most",
		    what, description->row_count, description->step_row_first,
		    description->split_rows, ratio, context,
		    description->session_capacity, role, STEP_WIDE_MOST_ROWS,
		    DS41F_DIRECT_RATIO, DS41F_POOLED_RATIO,
		    DS41F_CONTEXT_GRANULE, DS41F_CONTEXT_MOST,
		    DS41F_DIRECT_RATIO);
	    return false;
    }
    return true;
}

result_status ds41f_index_scan_prepare(
	const struct ds41f_index_scan_description *description,
	uint64_t arena_byte_count, const struct ds41f_index_scan_kernels *kernels,
	struct kernel_launch *launches_out, uint32_t *launch_count_out)
{
    static const char what[] = "the index scan";
    if (!prepare_arguments_are_given(what, description, kernels,
				     launches_out) ||
	launch_count_out == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (!prepare_kernel_is_prepared(what, kernels->scan, DS41F_KERNEL_INDEX_SCAN) ||
	!prepare_kernel_is_prepared(what, kernels->select,
				    DS41F_KERNEL_INDEX_SELECT) ||
	!prepare_kernel_is_prepared(what, kernels->candidates,
				    DS41F_KERNEL_INDEX_CANDIDATES) ||
	!counts_are_valid(what, description)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t rows = description->row_count, ratio = description->ratio, role = description->candidate_role;
    const uint32_t context = description->context_token_limit, entries = context / ratio;
    const uint32_t row_scores = role == DS41F_INDEX_CANDIDATES_READER ?
					DS41F_CONTEXT_WITHOUT_CANDIDATES :
					entries;
    const uint64_t keys_session_stride = (uint64_t)entries * DS41F_KEY_BYTES;
    const uint64_t candidates = role == DS41F_INDEX_CANDIDATES_READER ?
					description->candidates.byte_offset :
					DS41F_INDEX_NO_CANDIDATES;
    const uint32_t shape_rows = description->split_rows != 0 ? description->split_rows : rows;
    const uint32_t first = description->step_row_first;
    const bool split_rows = shape_rows <= DS41F_INDEX_SPLIT_MAX_ROWS ||
			    (shape_rows <= DS41F_INDEX_SPLIT_DEEP_MAX_ROWS &&
			     row_scores >= DS41F_INDEX_SPLIT_DEEP_MIN_ENTRIES);
    const bool split = split_rows && row_scores >= DS41F_INDEX_SPLIT_MIN_ENTRIES &&
		       role != DS41F_INDEX_CANDIDATES_READER;
    struct checked_range ranges[] = {
	    { "the step area", &description->step,
	      STEP_BLOCK_BYTES_VALUE +
		      (uint64_t)(first + rows) * STEP_ROW_BYTES_VALUE,
	      false },
	    { "the key tiles", &description->keys,
	      description->session_capacity * keys_session_stride, false },
	    { "the index queries", &description->queries,
	      (uint64_t)rows * DS41F_INDEX_QUERIES_ROW_BYTES, false },
	    { "the index weights", &description->weights,
	      (uint64_t)rows * DS41F_INDEX_WEIGHTS_ROW_BYTES, false },
	    { "the scores", &description->scores,
	      (uint64_t)rows * row_scores * DS41F_INDEX_SCORE_BYTES, false },
	    { "the selection", &description->selection,
	      (uint64_t)rows * DS41F_SELECTION_ROW_BYTES, false },
	    { "the candidates", &description->candidates,
	      (uint64_t)rows * DS41F_CANDIDATES_ROW_BYTES, false },
	    { 0 },
    };
    uint32_t range_count =
	    COUNT_OF(ranges) - 1u - (role == DS41F_INDEX_CANDIDATES_NONE ? 1u : 0u);
    if (!prepare_ranges_are_valid(what, ranges, range_count,
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    /* The payload range passed above proves this addition cannot wrap. The annex is reserved by the planner;
     * the builder continues to name only the score payload, so its existing descriptions need no change. */
    const struct linear_range scratch = {
	    description->scores.byte_offset + description->scores.byte_count,
	    (uint64_t)rows * DS41F_INDEX_SPLIT_ROW_BYTES
    };
    if (split) {
	    for (uint32_t stage = 0; stage < DS41F_INDEX_SPLIT_LAUNCH_COUNT;
		 stage++) {
		    if (!prepare_kernel_is_prepared(
				what, kernels->split[stage],
				kernel_names[DS41F_INDEX_SCAN_LEGACY_LAUNCH_COUNT +
					     stage]
					.function_name)) {
			    return STATUS_INVALID_ARGUMENT;
		    }
	    }
	ranges[range_count++] = (struct checked_range){"the split selection scratch", &scratch,
                                                        scratch.byte_count, false};
	if (!prepare_ranges_are_valid(what, ranges, range_count,
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
    }
    const struct ds41f_index_scan_arguments_record scan_record = {
	    .step_offset = description->step.byte_offset,
	    .keys_offset = description->keys.byte_offset,
	    .keys_session_stride = keys_session_stride,
	    .queries_offset = description->queries.byte_offset,
	    .weights_offset = description->weights.byte_offset,
	    .scores_offset = description->scores.byte_offset,
	    .candidates_offset = candidates,
	    .entry_capacity = row_scores,
	    .ratio = ratio,
	    .context_token_limit = context,
	    .session_capacity = description->session_capacity,
	    .row_first = first
    };
    const struct ds41f_index_select_arguments_record select_record = {
	    .step_offset = description->step.byte_offset,
	    .scores_offset = description->scores.byte_offset,
	    .selection_offset = description->selection.byte_offset,
	    .candidates_offset = candidates,
	    .context_token_limit = context,
	    .ratio = ratio,
	    .row_first = first
    };
    const struct ds41f_index_candidates_arguments candidates_record = {
	    .step_offset = description->step.byte_offset,
	    .scores_offset = description->scores.byte_offset,
	    .candidates_offset = description->candidates.byte_offset,
	    .entry_capacity = entries,
	    .context_token_limit = context,
	    .row_first = first
    };
    prepare_launch_fill(
	    &launches_out[DS41F_INDEX_SCAN_LAUNCH_SCAN], kernels->scan, rows,
	    (row_scores / DS41F_KEYS_PER_TILE + DS41F_INDEX_SEGMENT_TILES - 1u) /
		    DS41F_INDEX_SEGMENT_TILES,
	    DS41F_INDEX_SCAN_THREADS, DS41F_INDEX_QUERIES_ROW_BYTES, &scan_record,
	    DS41F_INDEX_SCAN_ARGUMENT_BYTES);
    if (split) {
        uint32_t count = 1;
	const uint32_t selections =
		role == DS41F_INDEX_CANDIDATES_SOURCE ? SOURCE_SELECTIONS : 1u;
	for (uint32_t selection = 0; selection < selections; selection++) {
		if (selection != 0 &&
		    entries < DS41F_INDEX_SPLIT_MIN_CANDIDATE_ENTRIES) {
			prepare_launch_fill(&launches_out[count++],
					    kernels->candidates, rows, 1,
					    DS41F_INDEX_SELECT_THREADS_VALUE, 0,
					    &candidates_record,
					    DS41F_INDEX_CANDIDATES_ARGUMENT_BYTES);
			continue;
		}
		const struct ds41f_index_split_arguments record = {
			.step_offset = description->step.byte_offset,
			.scores_offset = description->scores.byte_offset,
			.output_offset =
				selection == 0 ?
					description->selection.byte_offset :
					description->candidates.byte_offset,
			.candidates_offset = DS41F_INDEX_NO_CANDIDATES,
			.scratch_offset = scratch.byte_offset,
			.context_token_limit = context,
			.ratio = ratio,
			.segments = DS41F_INDEX_SPLIT_SEGMENTS,
			.is_block = selection,
			.row_first = first
		};
		for (uint32_t stage = 0; stage < DS41F_INDEX_SPLIT_LAUNCH_COUNT;
		     stage++) {
			const bool cut = stage == DS41F_INDEX_SPLIT_CUT_HIGH ||
					 stage == DS41F_INDEX_SPLIT_CUT_LOW;
			prepare_launch_fill(&launches_out[count++],
					    kernels->split[stage], rows,
					    cut ? 1u : DS41F_INDEX_SPLIT_SEGMENTS,
					    DS41F_INDEX_SPLIT_THREADS, 0, &record,
					    DS41F_INDEX_SPLIT_ARGUMENT_BYTES);
		}
	}
        *launch_count_out = count;
	return STATUS_OK;
    }
    prepare_launch_fill(&launches_out[DS41F_INDEX_SCAN_LAUNCH_SELECT],
			kernels->select, rows, 1, DS41F_INDEX_SELECT_THREADS_VALUE, 0,
			&select_record, DS41F_INDEX_SELECT_ARGUMENT_BYTES);
    if (role == DS41F_INDEX_CANDIDATES_SOURCE) {
	    prepare_launch_fill(&launches_out[DS41F_INDEX_SCAN_LAUNCH_CANDIDATES],
				kernels->candidates, rows, 1,
				DS41F_INDEX_SELECT_THREADS_VALUE, 0,
				&candidates_record,
				DS41F_INDEX_CANDIDATES_ARGUMENT_BYTES);
    }
    *launch_count_out = role == DS41F_INDEX_CANDIDATES_SOURCE ?
				DS41F_INDEX_SCAN_LEGACY_LAUNCH_COUNT :
				DS41F_INDEX_SCAN_LAUNCH_CANDIDATES;
    return STATUS_OK;
}
