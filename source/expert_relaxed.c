/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_relaxed.c: the launches of the routed experts in the relaxed order of class 80. Contract: expert_relaxed.h;
 * the kernels, their order and launches: expert_relaxed_records.h.
 *
 *   - The chain is linear_routed_experts_prepare's, which checks the description and gives the quantization
 *     and the row sum; its gate-up and down launches are replaced by the relaxed kernels': the narrow ones take the
 *     same records, the wide ones their own (each product's two kernels the same record). Nothing is written before
 *     every check has passed.
 */
#include "anemone/expert_relaxed.h"

#include <inttypes.h>
#include <stdbool.h>
#include <string.h>

#include "anemone/hooks.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define NARROW_KERNEL_COUNT \
	2u /* the first of the table: gate-up and down over the job table */
#define RELAXED_KERNEL_COUNT 6u

/* The kernels in the order of the members of struct expert_relaxed_kernels. */
struct relaxed_kernel {
	const char *function_name;
	uint32_t argument_byte_count;
};
static const struct relaxed_kernel relaxed_kernels[] = {
	{ KERNEL_EXPERT_RELAXED_GATE_UP, EXPERT_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_RELAXED_DOWN, EXPERT_DOWN_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_RELAXED_WIDE_GATE_UP,
	  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_RELAXED_WIDE_DOWN,
	  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_RELAXED_WIDE_SMALL_GATE_UP,
	  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_RELAXED_WIDE_SMALL_DOWN,
	  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_ORDERED_WIDE_GATE_UP,
	  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_ORDERED_WIDE_DOWN,
	  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_ORDERED_WIDE_SMALL_GATE_UP,
	  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_EXPERT_ORDERED_WIDE_SMALL_DOWN,
	  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES },
};
_Static_assert(sizeof(struct expert_relaxed_kernels) ==
		       COUNT_OF(relaxed_kernels) * sizeof(uint32_t),
	       "a handle for each relaxed kernel");

result_status expert_relaxed_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct expert_relaxed_kernels *kernels_out)
{
	uint32_t handles[COUNT_OF(relaxed_kernels)] = { 0 };
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR("the relaxed experts' kernels: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	if (arithmetic_class != ARITHMETIC_CLASS_SM_80) {
		LOG_ERROR(
			"the relaxed experts' kernels: a card of class %" PRIu32
			": the relaxed order is class %u's",
			arithmetic_class, (unsigned int)ARITHMETIC_CLASS_SM_80);
		return STATUS_UNSUPPORTED;
	}
	for (uint32_t kernel = 0; kernel < COUNT_OF(relaxed_kernels);
	     kernel++) {
		handles[kernel] = BACKEND_NO_KERNEL;
		const result_status status = backend->kernel_prepare(
			card, KERNEL_FILE_EXPERT_RELAXED,
			relaxed_kernels[kernel].function_name,
			relaxed_kernels[kernel].argument_byte_count,
			&handles[kernel]);
		if (status != STATUS_OK) {
			LOG_ERROR(
				"the relaxed experts' kernels: %s is not prepared: %s",
				relaxed_kernels[kernel].function_name,
				status_text(status));
			return status;
		}
	}
	memcpy(kernels_out, handles, sizeof(*kernels_out));
	return STATUS_OK;
}

result_status expert_relaxed_fold_check(const uint8_t *scale_codes,
					uint64_t scale_count)
{
	uint32_t least = UINT8_MAX, largest = 0;
	if (scale_codes == NULL || scale_count == 0) {
		LOG_ERROR(
			"the relaxed experts' fold check: a missing argument or no scale code");
		return STATUS_INVALID_ARGUMENT;
	}
	for (uint64_t index = 0; index < scale_count; index++) {
		least = scale_codes[index] < least ? scale_codes[index] : least;
		largest = scale_codes[index] > largest ? scale_codes[index] :
							 largest;
	}
	if (least < EXPERT_RELAXED_SCALE_SMALLEST ||
	    largest > EXPERT_RELAXED_SCALE_LARGEST) {
		LOG_ERROR(
			"the relaxed experts' fold check: scale codes %" PRIu32
			" to %" PRIu32 ": the fold is exact "
			"from %u to %u",
			least, largest, EXPERT_RELAXED_SCALE_SMALLEST,
			EXPERT_RELAXED_SCALE_LARGEST);
		return STATUS_INVALID_ARGUMENT;
	}
	return STATUS_OK;
}

/* The ranges of a wide description: the tiles and pairs with what the kernels read there, the experts' (checked by
 * linear_routed_experts_prepare) for overlap alone. */
static bool
wide_ranges_are_valid(const char *what,
		      const struct expert_relaxed_description *description,
		      uint64_t arena_byte_count)
{
	const struct linear_routed_experts_description *const experts =
		&description->experts;
	const uint64_t pair_count =
		(uint64_t)experts->row_count * experts->pairs_per_row;
	const struct checked_range ranges[] = {
		{ "the tiles", &description->tiles,
		  (uint64_t)description->tile_count * WIDE_EXPERT_TILE_BYTES,
		  false },
		{ "the sorted pairs", &description->pairs,
		  pair_count * WIDE_PAIR_BYTES, false },
		{ "the job table", &experts->jobs, 0, false },
		{ "the experts' matrices", &experts->expert_weights, 0, false },
		{ "the rows' codes", &experts->codes, 0, false },
		{ "the rows' exponents", &experts->exponents, 0, false },
		{ "the rows' operands", &experts->operands, 0, false },
		{ "the hidden values", &experts->hidden, 0, false },
		{ "the hidden codes", &experts->hidden_codes, 0, false },
		{ "the hidden exponents", &experts->hidden_exponents, 0,
		  false },
		{ "the hidden operands", &experts->hidden_operands, 0, false },
		{ "the pair outputs", &experts->pair_output, 0, false },
		{ "the row outputs", &experts->row_output, 0, false },
	};
	return prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
					arena_byte_count);
}

MUST_CHECK static result_status
expert_prepare(const struct expert_relaxed_description *description,
	       uint64_t arena_byte_count,
	       const struct linear_kernels_record *linear_kernels,
	       const struct expert_relaxed_kernels *kernels,
	       struct kernel_launch *launches_out, bool is_ordered)
{
	static const char what[] = "the relaxed routed experts";
	struct kernel_launch launches[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	if (!prepare_arguments_are_given(what, description, kernels,
					 launches_out) ||
	    !prepare_arguments_are_given(what, linear_kernels, kernels,
					 launches_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct linear_routed_experts_description *const experts =
		&description->experts;
	const bool is_wide = description->tile_count != 0;
	if (description->tile_count > EXPERT_WIDE_MOST_TILES) {
		LOG_ERROR("the wide experts: too many tiles for grid Y");
		return STATUS_INVALID_ARGUMENT;
	}
	if (is_ordered && !is_wide) {
		LOG_ERROR("the ordered wide experts: no tiles");
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t gate_up_divisor =
		is_wide ? EXPERT_RELAXED_WIDE_GATE_UP_OUTPUTS :
			  EXPERT_RELAXED_NARROW_GATE_UP_OUTPUTS;
	const uint32_t down_divisor =
		is_wide ? EXPERT_RELAXED_WIDE_DOWN_OUTPUTS :
			  EXPERT_RELAXED_NARROW_DOWN_OUTPUTS;
	const uint32_t gate_up_outputs = experts->gate_up_shape.output_size;
	const uint32_t down_outputs = experts->down_shape.output_size;
	uint32_t handles[COUNT_OF(relaxed_kernels)];
	memcpy(handles, kernels, sizeof(handles));
	bool are_prepared =
		true; /* the form's kernels: the narrow ones first in the table, then the wide ones */
	for (uint32_t kernel = is_ordered ? RELAXED_KERNEL_COUNT :
					    (is_wide ? NARROW_KERNEL_COUNT : 0);
	     kernel < (is_ordered ? COUNT_OF(relaxed_kernels) :
				    (is_wide ? RELAXED_KERNEL_COUNT :
					       NARROW_KERNEL_COUNT));
	     kernel++) {
		are_prepared = are_prepared &&
			       prepare_kernel_is_prepared(
				       what, handles[kernel],
				       relaxed_kernels[kernel].function_name);
	}
	if (!prepare_reserved_is_zero(what, description->reserved_zero) ||
	    !are_prepared ||
	    linear_routed_experts_prepare(experts, arena_byte_count,
					  linear_kernels,
					  launches) != STATUS_OK) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t input_divisor = is_wide ? EXPERT_RELAXED_WIDE_INPUTS :
						 EXPERT_RELAXED_NARROW_INPUTS;
	const uint32_t most_inputs = is_wide && !is_ordered ?
					     EXPERT_RELAXED_WIDE_MOST_INPUTS :
					     UINT32_MAX;
	if (gate_up_outputs % gate_up_divisor != 0 ||
	    down_outputs % down_divisor != 0 ||
	    experts->gate_up_shape.input_size % input_divisor != 0 ||
	    experts->down_shape.input_size % input_divisor != 0 ||
	    experts->gate_up_shape.input_size > most_inputs ||
	    experts->down_shape.input_size > most_inputs) {
		LOG_ERROR("%s: gate and up of %" PRIu32
			  " outputs, down of %" PRIu32 ", inputs %" PRIu32
			  " and %" PRIu32
			  ": the %s form takes outputs in multiples of %" PRIu32
			  " and %" PRIu32 ", inputs of %" PRIu32 ", %" PRIu32
			  " at most",
			  what, gate_up_outputs, down_outputs,
			  experts->gate_up_shape.input_size,
			  experts->down_shape.input_size,
			  is_wide ? "wide" : "narrow", gate_up_divisor,
			  down_divisor, input_divisor, most_inputs);
		return STATUS_INVALID_ARGUMENT;
	}
	if (!is_wide) { /* the reference chain's records, the relaxed kernels */
		struct expert_gate_up_arguments narrow_gate_up;
		struct expert_down_arguments narrow_down;
		memcpy(&narrow_gate_up,
		       launches[LINEAR_ROUTED_EXPERTS_GATE_UP].argument,
		       sizeof(narrow_gate_up));
		memcpy(&narrow_down,
		       launches[LINEAR_ROUTED_EXPERTS_DOWN].argument,
		       sizeof(narrow_down));
		prepare_launch_fill(&launches[LINEAR_ROUTED_EXPERTS_GATE_UP],
				    kernels->gate_up,
				    gate_up_outputs / gate_up_divisor,
				    experts->job_count, EXPERT_RELAXED_THREADS,
				    EXPERT_RELAXED_NARROW_SHARED_BYTES,
				    &narrow_gate_up,
				    EXPERT_GATE_UP_ARGUMENT_BYTES);
		prepare_launch_fill(&launches[LINEAR_ROUTED_EXPERTS_DOWN],
				    kernels->down, down_outputs / down_divisor,
				    experts->job_count, EXPERT_RELAXED_THREADS,
				    EXPERT_RELAXED_NARROW_SHARED_BYTES,
				    &narrow_down, EXPERT_DOWN_ARGUMENT_BYTES);
		memcpy(launches_out, launches,
		       LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT *
			       sizeof(launches[0]));
		return STATUS_OK;
	}
	if (!wide_ranges_are_valid(what, description, arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t pair_count =
		experts->row_count *
		experts->pairs_per_row; /* checked to fit 32 bits */
	const struct expert_relaxed_wide_gate_up_arguments wide_gate_up = {
		.tiles_offset = description->tiles.byte_offset,
		.pairs_offset = description->pairs.byte_offset,
		.operands_offset = experts->operands.byte_offset,
		.hidden_offset = experts->hidden.byte_offset,
		.output_size = gate_up_outputs,
		.input_size = experts->gate_up_shape.input_size,
		.tile_count = description->tile_count,
		.pair_count = pair_count,
		.pairs_per_row = experts->pairs_per_row,
		.activation_limit = experts->activation_limit,
	};
	const struct expert_relaxed_wide_down_arguments wide_down = {
		.tiles_offset = description->tiles.byte_offset,
		.pairs_offset = description->pairs.byte_offset,
		.operands_offset = experts->hidden_operands.byte_offset,
		.output_offset = experts->pair_output.byte_offset,
		.output_size = down_outputs,
		.input_size = experts->down_shape.input_size,
		.tile_count = description->tile_count,
		.pair_count = pair_count,
	};
	struct kernel_launch wide[EXPERT_RELAXED_WIDE_LAUNCH_COUNT];
	wide[EXPERT_RELAXED_WIDE_QUANTIZE] =
		launches[LINEAR_ROUTED_EXPERTS_QUANTIZE];
	wide[EXPERT_RELAXED_WIDE_ROW_SUM] =
		launches[LINEAR_ROUTED_EXPERTS_ROW_SUM];
	const struct {
		uint32_t launch, kernel, grid_columns, shared_bytes,
			record_bytes;
		const void *record;
	} products[] = {
		{ EXPERT_RELAXED_WIDE_GATE_UP,
		  is_ordered ? kernels->ordered_wide_gate_up :
			       kernels->wide_gate_up,
		  gate_up_outputs / gate_up_divisor,
		  EXPERT_RELAXED_WIDE_SHARED_BYTES,
		  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES, &wide_gate_up },
		{ EXPERT_RELAXED_WIDE_SMALL_GATE_UP,
		  is_ordered ? kernels->ordered_wide_small_gate_up :
			       kernels->wide_small_gate_up,
		  gate_up_outputs / gate_up_divisor,
		  EXPERT_RELAXED_WIDE_SMALL_SHARED_BYTES,
		  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES, &wide_gate_up },
		{ EXPERT_RELAXED_WIDE_DOWN,
		  is_ordered ? kernels->ordered_wide_down : kernels->wide_down,
		  down_outputs / down_divisor, EXPERT_RELAXED_WIDE_SHARED_BYTES,
		  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES, &wide_down },
		{ EXPERT_RELAXED_WIDE_SMALL_DOWN,
		  is_ordered ? kernels->ordered_wide_small_down :
			       kernels->wide_small_down,
		  down_outputs / down_divisor,
		  EXPERT_RELAXED_WIDE_SMALL_SHARED_BYTES,
		  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES, &wide_down },
	};
	for (uint32_t product = 0; product < COUNT_OF(products); product++) {
		prepare_launch_fill(&wide[products[product].launch],
				    products[product].kernel,
				    products[product].grid_columns,
				    description->tile_count,
				    EXPERT_RELAXED_THREADS,
				    products[product].shared_bytes,
				    products[product].record,
				    products[product].record_bytes);
	}
	memcpy(launches_out, wide, sizeof(wide));
	return STATUS_OK;
}

result_status
expert_relaxed_prepare(const struct expert_relaxed_description *description,
		       uint64_t arena_byte_count,
		       const struct linear_kernels_record *linear_kernels,
		       const struct expert_relaxed_kernels *kernels,
		       struct kernel_launch *launches_out)
{
	return expert_prepare(description, arena_byte_count, linear_kernels,
			      kernels, launches_out, false);
}

result_status
expert_ordered_prepare(const struct expert_relaxed_description *description,
		       uint64_t arena_byte_count,
		       const struct linear_kernels_record *linear_kernels,
		       const struct expert_relaxed_kernels *kernels,
		       struct kernel_launch *launches_out)
{
	return expert_prepare(description, arena_byte_count, linear_kernels,
			      kernels, launches_out, true);
}

result_status
expert_select_prepare(const struct expert_relaxed_description *description,
		      uint64_t arena_byte_count,
		      const struct linear_kernels_record *linear_kernels,
		      const struct expert_relaxed_kernels *kernels,
		      uint32_t arithmetic_class, enum cmp_order_code cmp_order,
		      bool folds_admitted, struct kernel_launch *launches_out,
		      uint32_t *launch_count_out)
{
	static const char what[] = "the routed experts' CMP selection";
	struct kernel_launch launches[EXPERT_RELAXED_WIDE_LAUNCH_COUNT];
	if (description == NULL || launches_out == NULL ||
	    launch_count_out == NULL ||
	    (uint32_t)cmp_order > CMP_ORDER_RELAXED ||
	    (arithmetic_class != ARITHMETIC_CLASS_SM_80 &&
	     arithmetic_class != ARITHMETIC_CLASS_SM_120 &&
	     arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE) ||
	    description->reserved_zero != 0) {
		LOG_ERROR(
			"%s: missing argument, invalid order, class or reserved field",
			what);
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = linear_routed_experts_prepare(
		&description->experts, arena_byte_count, linear_kernels,
		launches);
	if (status != STATUS_OK) {
		return status;
	}
	const bool is_cmp = arithmetic_class == ARITHMETIC_CLASS_SM_80;
	const bool is_wide = description->experts.row_count >
			     LINEAR_MAXIMUM_ROWS;
	const bool is_relaxed = is_cmp && cmp_order == CMP_ORDER_RELAXED &&
				folds_admitted;
	const bool is_ordered = is_cmp && description->experts.row_count >=
						  EXPERT_ORDERED_MINIMUM_ROWS;
	const bool selected_wide = is_ordered || (is_relaxed && is_wide);
	uint32_t launch_count = LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT;
	if (selected_wide && description->tile_count == 0) {
		LOG_ERROR("%s: wide rows need the router's tiles", what);
		return STATUS_INVALID_ARGUMENT;
	}
	if (is_ordered || is_relaxed) {
		struct expert_relaxed_description selected = *description;
		selected.tile_count = is_wide ? description->tile_count : 0;
		status = expert_prepare(&selected, arena_byte_count,
					linear_kernels, kernels, launches,
					!is_relaxed);
		launch_count = is_wide ? EXPERT_RELAXED_WIDE_LAUNCH_COUNT :
					 launch_count;
	}
	if (status == STATUS_OK) {
		memcpy(launches_out, launches,
		       (size_t)launch_count * sizeof(launches[0]));
		*launch_count_out = launch_count;
	}
	return status;
}
