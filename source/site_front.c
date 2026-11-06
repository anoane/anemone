/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * site_front.c: the launches of the fused fronts of a block's sites. Contract: site_front.h; the kernels:
 * site_front_records.h. Byte counts are the linear's, which stop at UINT64_MAX: no product of sizes wraps, and such a
 * count fits no range.
 */
#include "anemone/site_front.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "anemone/hooks.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define MIX_WEIGHTS_BYTES \
	((uint64_t)MODEL_MIXES * MODEL_STREAMS * MODEL_HIDDEN * sizeof(float))
#define SCALE_BYTES ((uint64_t)HYPER_SCALES * sizeof(float))
#define ROUTER_WEIGHTS_BYTES ((uint64_t)MODEL_EXPERTS * MODEL_HIDDEN_ROW_BYTES)
#define SITE_RANGES 13u                         /* the ranges every front has; the FFN front's follow */

#define SITE_FRONT_STAMP(name, rows, kind) "" #name "_front_" #rows,
static const char *const function_names[] = { SITE_FRONT_EVERY_INSTANCE };
#undef SITE_FRONT_STAMP
_Static_assert(
	COUNT_OF(function_names) == SITE_FRONT_INSTANCES &&
		sizeof(struct site_front_kernels) ==
			sizeof(uint32_t) * COUNT_OF(function_names) &&
		SITE_FRONT_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES,
	"a handle for each function, in their order; the record in a launch");

result_status
site_front_kernels_prepare(const struct backend_operations *backend,
			   struct device_card *card,
			   struct site_front_kernels *kernels_out)
{
	struct site_front_kernels kernels;
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR("the kernels of the site fronts: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
    for (uint32_t function = 0; function < COUNT_OF(function_names); function++) {
	    kernels.fronts[function] =
		    BACKEND_NO_KERNEL; /* what a backend that gives no handle leaves */
	    const result_status status = backend->kernel_prepare(
		    card, KERNEL_FILE_SITE_FRONT, function_names[function],
		    SITE_FRONT_ARGUMENT_BYTES, &kernels.fronts[function]);
	    if (status != STATUS_OK) {
		    LOG_ERROR(
			    "the kernels of the site fronts: %s is not prepared: %s",
			    function_names[function], status_text(status));
		    return status;
	    }
    }
    *kernels_out = kernels;
    return STATUS_OK;
}

result_status
site_front_prepare(const struct site_front_description *description,
		   uint64_t arena_byte_count,
		   const struct router_kernels_record *router_kernels,
		   const struct site_front_kernels *kernels,
		   struct kernel_launch *launch_out)
{
    static const char what[] = "a site front";
    struct kernel_launch select;
    if (!prepare_arguments_are_given(what, description, kernels, launch_out) ||
	!prepare_reserved_is_zero(what, description->reserved_zero)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct router_description *const router = description->router;
    const uint32_t row_count = description->row_count, kind = description->kind;
    const bool is_ffn = kind == SITE_FRONT_KIND_FFN;
    if (row_count == 0 || row_count > LINEAR_MAXIMUM_ROWS ||
	kind >= SITE_FRONT_KINDS || (router != NULL) != is_ffn ||
	(router_kernels != NULL) != is_ffn ||
	(router != NULL &&
	 (router->experts == NULL || router->experts->row_count != row_count ||
	  router->expert_count != MODEL_EXPERTS ||
	  router->pick_count != MODEL_EXPERTS_PER_ROW))) {
	    LOG_ERROR(
		    "%s: kind %" PRIu32 ", %" PRIu32
		    " rows: a known kind, 1 to %u rows (the routed experts' "
		    "with a router), a router with its kernels exactly for the FFN front",
		    what, kind, row_count, LINEAR_MAXIMUM_ROWS);
	    return STATUS_INVALID_ARGUMENT;
    }
    if (router != NULL) {
	    const result_status status = router_select_prepare(
		    router, arena_byte_count, router_kernels, &select);
	    if (status != STATUS_OK) {
		    return status;
	    }
    }
    const uint64_t rows = row_count;
    const uint32_t instance =
	    kind * SITE_FRONT_ROW_FORMS + (row_count > 1u ? 1u : 0u);
    const struct linear_range none = { 0, 0 };
    const struct checked_range ranges[] = {
	    { "the streams", &description->streams,
	      rows * MODEL_STREAMS_ROW_BYTES, false },
	    { "the carried pre", &description->carried_pre,
	      rows * MODEL_SITE_ROW_BYTES, false },
	    { "the mix weights", &description->mix_weights, MIX_WEIGHTS_BYTES,
	      true },
	    { "the scale", &description->scale, SCALE_BYTES, true },
	    { "the base", &description->base, MODEL_MIXES_ROW_BYTES, true },
	    { "the norm's weight", &description->norm_weight,
	      MODEL_HIDDEN_ROW_BYTES, true },
	    { "the post", &description->post, rows * MODEL_SITE_ROW_BYTES,
	      false },
	    { "the comb", &description->comb, rows * MODEL_COMB_ROW_BYTES,
	      false },
	    { "the sublayer's input", &description->hidden,
	      rows * MODEL_HIDDEN_ROW_BYTES, false },
	    { "the codes", &description->codes,
	      linear_codes_byte_count(rows, MODEL_HIDDEN), false },
	    { "the exponents", &description->exponents,
	      linear_exponents_byte_count(rows, MODEL_HIDDEN), false },
	    { "the operands", &description->operands,
	      linear_values_byte_count(rows, MODEL_HIDDEN), false },
	    { "the scratch", &description->scratch, SITE_FRONT_SCRATCH_BYTES,
	      false },
	    /* the FFN front's: its weights; the select's ranges, checked by its prepare, here for sharing no byte */
	    { "the router's weights", &description->router_weights,
	      ROUTER_WEIGHTS_BYTES, true },
	    { "the step block", router != NULL ? &router->step : &none, 0,
	      false },
	    { "the router scores", router != NULL ? &router->scores : &none, 0,
	      false },
	    { "the bias", router != NULL ? &router->bias : &none, 0, false },
	    { "the expert offsets", router != NULL ? &router->offsets : &none,
	      0, false },
	    { "the affinities", router != NULL ? &router->affinities : &none, 0,
	      false },
	    { "the job table", router != NULL ? &router->experts->jobs : &none,
	      0, false },
	    { "the experts' matrices",
	      router != NULL ? &router->experts->expert_weights : &none, 0,
	      false },
    };
    if (!prepare_kernel_is_prepared(what, kernels->fronts[instance],
				    function_names[instance]) ||
	!prepare_ranges_are_valid(
		what, ranges, router != NULL ? COUNT_OF(ranges) : SITE_RANGES,
		arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    struct site_front_arguments record = {
	    .streams_offset = description->streams.byte_offset,
	    .pre_offset = description->carried_pre.byte_offset,
	    .scale_offset = description->scale.byte_offset,
	    .base_offset = description->base.byte_offset,
	    .post_offset = description->post.byte_offset,
	    .comb_offset = description->comb.byte_offset,
	    .norm_weight_offset = description->norm_weight.byte_offset,
	    .hidden_offset = description->hidden.byte_offset,
	    .mix_weights_offset = description->mix_weights.byte_offset,
	    .codes_offset = description->codes.byte_offset,
	    .exponents_offset = description->exponents.byte_offset,
	    .operands_offset = description->operands.byte_offset,
	    .scratch_offset = description->scratch.byte_offset,
	    .router_weights_offset =
		    router != NULL ? description->router_weights.byte_offset : 0
    };
    if (router != NULL) {
        memcpy(&record.select, select.argument, sizeof(record.select));
    }
    record.select.row_count = row_count;
    prepare_launch_fill(launch_out, kernels->fronts[instance],
			SITE_FRONT_BLOCKS(kind, row_count), 1,
			SITE_FRONT_THREADS, 0, &record,
			SITE_FRONT_ARGUMENT_BYTES);
    return STATUS_OK;
}
