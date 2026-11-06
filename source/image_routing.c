/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image-aware prepare records; numerical work is in kernel_image_routing and kernel_image_engram. */
#include "anemone/image_routing.h"
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define EXTRA_RANGE_PAIR 2u

result_status
image_routing_kernels_prepare(const struct backend_operations *backend,
			      struct device_card *card,
			      struct image_routing_kernels *kernels_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	static const struct {
		const char *file, *function;
		uint32_t bytes;
	} names[] = { { KERNEL_FILE_IMAGE_ROUTING, KERNEL_IMAGE_ROUTER_SELECT,
			IMAGE_ROUTER_ARGUMENT_BYTES },
		      { KERNEL_FILE_IMAGE_ROUTING,
			KERNEL_IMAGE_ROUTER_WIDE_SELECT,
			IMAGE_ROUTER_WIDE_ARGUMENT_BYTES },
		      { KERNEL_FILE_IMAGE_ENGRAM, KERNEL_IMAGE_ENGRAM_GATE,
			IMAGE_ENGRAM_ARGUMENT_BYTES } };
	uint32_t handles[COUNT_OF(names)] = { 0 };
	for (uint32_t index = 0; index < COUNT_OF(names); index++) {
		handles[index] = BACKEND_NO_KERNEL;
		const result_status status = backend->kernel_prepare(
			card, names[index].file, names[index].function,
			names[index].bytes, &handles[index]);
		if (status != STATUS_OK) {
			return status;
		}
	}
	_Static_assert(sizeof(handles) == sizeof(*kernels_out),
		       "one prepared handle per variant");
	memcpy(kernels_out, handles, sizeof(handles));
	return STATUS_OK;
}

/* The base prepare checked its ranges; this also proves the new range shares no byte with any of them. */
static bool extra_range_is_valid(struct linear_range extra, uint64_t required,
				 const struct linear_range *const *base,
				 uint32_t count, uint64_t arena_byte_count)
{
	for (uint32_t index = 0; index < count; index++) {
		const struct checked_range ranges[EXTRA_RANGE_PAIR] = {
			{ "the image range", &extra, required, false },
			{ "the base range", base[index], 0, false }
		};
		if (!prepare_ranges_are_valid("image-aware routing", ranges,
					      COUNT_OF(ranges),
					      arena_byte_count)) {
			return false;
		}
	}
	return true;
}

static bool step_rows_fit(struct linear_range step, uint32_t rows)
{
	return step.byte_count >=
	       STEP_BLOCK_BYTES_VALUE + (uint64_t)rows * STEP_ROW_BYTES_VALUE;
}

result_status
image_router_select_prepare(const struct router_description *description,
			    struct linear_range vision_bias,
			    uint64_t arena_byte_count,
			    const struct router_kernels_record *base_kernels,
			    const struct image_routing_kernels *kernels,
			    struct kernel_launch *launch_out)
{
	struct kernel_launch launch;
	if (description == NULL || kernels == NULL || launch_out == NULL ||
	    description->expert_count != MODEL_EXPERTS ||
	    !prepare_kernel_is_prepared("image router", kernels->select,
					KERNEL_IMAGE_ROUTER_SELECT)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const result_status status = router_select_prepare(
		description, arena_byte_count, base_kernels, &launch);
	if (status != STATUS_OK) {
		return status;
	}
	const struct linear_routed_experts_description *const experts =
		description->experts;
	const struct linear_range *const ranges[] = {
		&description->step,	    &description->scores,
		&description->bias,	    &description->offsets,
		&description->affinities,   &experts->jobs,
		&experts->expert_weights,   &experts->codes,
		&experts->exponents,	    &experts->operands,
		&experts->hidden,	    &experts->hidden_codes,
		&experts->hidden_exponents, &experts->hidden_operands,
		&experts->pair_output,	    &experts->row_output
	};
	if (!step_rows_fit(description->step, experts->row_count) ||
	    !extra_range_is_valid(vision_bias, MODEL_ROUTER_ROW_BYTES, ranges,
				  COUNT_OF(ranges), arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct image_router_arguments arguments = {
		.vision_bias_offset = vision_bias.byte_offset
	};
	memcpy(&arguments.base, launch.argument, sizeof(arguments.base));
	prepare_launch_fill(launch_out, kernels->select, launch.grid[0],
			    launch.grid[1], launch.block[0],
			    launch.dynamic_shared_byte_count, &arguments,
			    IMAGE_ROUTER_ARGUMENT_BYTES);
	return STATUS_OK;
}

result_status
image_router_wide_prepare(const struct router_wide_description *description,
			  struct linear_range vision_bias,
			  uint64_t arena_byte_count,
			  const struct router_kernels_record *base_kernels,
			  const struct image_routing_kernels *kernels,
			  struct kernel_launch *launches_out)
{
	struct kernel_launch launches[ROUTER_WIDE_LAUNCH_COUNT];
	if (description == NULL || kernels == NULL || launches_out == NULL ||
	    !prepare_kernel_is_prepared("image router", kernels->wide_select,
					KERNEL_IMAGE_ROUTER_WIDE_SELECT)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const result_status status = router_wide_prepare(
		description, arena_byte_count, base_kernels, launches);
	if (status != STATUS_OK) {
		return status;
	}
	const struct linear_range *const ranges[] = {
		&description->step,	      &description->scores,
		&description->bias,	      &description->offsets,
		&description->expert_weights, &description->affinities,
		&description->picks,	      &description->counts,
		&description->pairs,	      &description->runs,
		&description->tiles,	      &description->jobs
	};
	if (!step_rows_fit(description->step, description->row_count) ||
	    !extra_range_is_valid(vision_bias, MODEL_ROUTER_ROW_BYTES, ranges,
				  COUNT_OF(ranges), arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct image_router_wide_arguments arguments = {
		.vision_bias_offset = vision_bias.byte_offset
	};
	struct kernel_launch *const select =
		&launches[ROUTER_WIDE_SELECT_LAUNCH];
	memcpy(&arguments.base, select->argument, sizeof(arguments.base));
	prepare_launch_fill(select, kernels->wide_select, select->grid[0],
			    select->grid[1], select->block[0],
			    select->dynamic_shared_byte_count, &arguments,
			    IMAGE_ROUTER_WIDE_ARGUMENT_BYTES);
	memcpy(launches_out, launches, sizeof(launches));
	return STATUS_OK;
}

result_status
image_engram_rows_prepare(const struct engram_rows_description *description,
			  struct linear_range step, uint64_t arena_byte_count,
			  const struct engram_rows_kernels *base_kernels,
			  const struct image_routing_kernels *kernels,
			  struct kernel_launch *launches_out)
{
	struct kernel_launch launches[ENGRAM_ROWS_LAUNCH_COUNT];
	if (description == NULL || kernels == NULL || launches_out == NULL ||
	    !prepare_kernel_is_prepared("image Engram", kernels->engram_gate,
					KERNEL_IMAGE_ENGRAM_GATE)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const result_status status = engram_rows_prepare(
		description, arena_byte_count, base_kernels, launches);
	if (status != STATUS_OK) {
		return status;
	}
	const struct linear_range *const ranges[] = {
		&description->rows,	  &description->values,
		&description->projection, &description->query_weight,
		&description->key_weight, &description->streams
	};
	const uint64_t step_bytes =
		STEP_BLOCK_BYTES_VALUE +
		(uint64_t)description->row_count * STEP_ROW_BYTES_VALUE;
	if (!extra_range_is_valid(step, step_bytes, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct image_engram_arguments arguments = { .step_offset =
							    step.byte_offset };
	struct kernel_launch *const gate = &launches[ENGRAM_ROWS_GATE_LAUNCH];
	memcpy(&arguments.base, gate->argument, sizeof(arguments.base));
	prepare_launch_fill(gate, kernels->engram_gate, gate->grid[0],
			    gate->grid[1], gate->block[0],
			    gate->dynamic_shared_byte_count, &arguments,
			    IMAGE_ENGRAM_ARGUMENT_BYTES);
	memcpy(launches_out, launches, sizeof(launches));
	return STATUS_OK;
}
