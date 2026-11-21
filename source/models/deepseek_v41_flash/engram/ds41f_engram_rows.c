/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_engram_rows.c: the launches of an n-gram layer's own. Contract: ds41f_engram_rows.h; the kernels:
 * ds41f_engram_rows_records.h.
 */
#include "ds41f_engram_rows.h"

#include <inttypes.h>

#include "engine/engram/engram.h"
#include "common/hooks.h"
#include "engine/library/prepare_checks.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define WEIGHT_BYTES               \
	((uint64_t)DS41F_STREAMS * \
	 DS41F_HIDDEN_ROW_BYTES) /* q_weight; k_weight */

_Static_assert(DS41F_ENGRAM_ROWS_ROW_BYTES == ENGRAM_ROW_BYTES &&
		       DS41F_ENGRAM_ROWS_VALUES == ENGRAM_VALUE_BYTES_VALUE,
	       "the kernel reads the rows anemone_engram_gather gives");

result_status
ds41f_engram_rows_kernels_prepare(const struct backend_operations *backend,
			    struct device_card *card,
			    struct ds41f_engram_rows_kernels *kernels_out)
{
	/* a backend that answers OK and gives no handle leaves this one, which ds41f_engram_rows_prepare refuses */
	struct ds41f_engram_rows_kernels kernels = { BACKEND_NO_KERNEL,
					       BACKEND_NO_KERNEL };
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR("the kernels of the n-gram rows: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	const char *function_name = DS41F_KERNEL_ENGRAM_VALUES;
	result_status status = backend->kernel_prepare(
		card, DS41F_MODULE_ENGRAM_ROWS, function_name,
		DS41F_ENGRAM_VALUES_ARGUMENT_BYTES, &kernels.values);
	if (status == STATUS_OK) {
		function_name = DS41F_KERNEL_ENGRAM_GATE;
		status = backend->kernel_prepare(card, DS41F_MODULE_ENGRAM_ROWS,
						 function_name,
						 DS41F_ENGRAM_GATE_ARGUMENT_BYTES,
						 &kernels.gate);
	}
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the kernels of the n-gram rows: %s is not prepared: %s",
			function_name, status_text(status));
		return status;
	}
    *kernels_out = kernels;
    return STATUS_OK;
}

result_status
ds41f_engram_rows_prepare(const struct ds41f_engram_rows_description *description,
		    uint64_t arena_byte_count,
		    const struct ds41f_engram_rows_kernels *kernels,
		    struct kernel_launch *launches_out)
{
    static const char what[] = "the n-gram rows";
    if (!prepare_arguments_are_given(what, description, kernels,
				     launches_out) ||
	!prepare_kernel_is_prepared(what, kernels->values,
				    DS41F_KERNEL_ENGRAM_VALUES) ||
	!prepare_kernel_is_prepared(what, kernels->gate, DS41F_KERNEL_ENGRAM_GATE)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t rows_per_token = description->rows_per_token;
    const uint32_t row_count = description->row_count;
    if (rows_per_token == 0 || rows_per_token > ENGRAM_MAXIMUM_ROWS_PER_TOKEN ||
	row_count == 0 || row_count > STEP_WIDE_MOST_ROWS) {
	    LOG_ERROR("%s: %" PRIu32 " rows of %" PRIu32
		      " table rows: a launch takes 1 to %u of 1 to %d",
		      what, row_count, rows_per_token, STEP_WIDE_MOST_ROWS,
		      ENGRAM_MAXIMUM_ROWS_PER_TOKEN);
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t table_rows = (uint64_t)row_count * rows_per_token;
    const struct checked_range ranges[] = {
	    { "the rows", &description->rows,
	      table_rows * DS41F_ENGRAM_ROWS_ROW_BYTES, false },
	    { "the values", &description->values,
	      linear_values_byte_count(table_rows, DS41F_ENGRAM_ROWS_VALUES), false },
	    { "the projection", &description->projection,
	      linear_values_byte_count(row_count, DS41F_ENGRAM_ROWS_PROJECTION),
	      false },
	    { "q_weight", &description->query_weight, WEIGHT_BYTES, false },
	    { "k_weight", &description->key_weight, WEIGHT_BYTES, false },
	    { "the streams", &description->streams,
	      (uint64_t)row_count * DS41F_STREAMS_ROW_BYTES, false },
    };
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct ds41f_engram_values_arguments values_record = {
	    description->rows.byte_offset, description->values.byte_offset
    };
    const struct ds41f_engram_gate_arguments gate_record = {
	    description->projection.byte_offset,
	    description->query_weight.byte_offset,
	    description->key_weight.byte_offset,
	    description->streams.byte_offset
    };
    prepare_launch_fill(&launches_out[DS41F_ENGRAM_ROWS_VALUES_LAUNCH],
			kernels->values, rows_per_token, row_count,
			DS41F_ENGRAM_ROWS_VALUES, 0, &values_record,
			DS41F_ENGRAM_VALUES_ARGUMENT_BYTES);
    prepare_launch_fill(&launches_out[DS41F_ENGRAM_ROWS_GATE_LAUNCH], kernels->gate,
			DS41F_STREAMS, row_count, DS41F_ENGRAM_ROWS_GATE_THREADS, 0,
			&gate_record, DS41F_ENGRAM_GATE_ARGUMENT_BYTES);
    return STATUS_OK;
}
