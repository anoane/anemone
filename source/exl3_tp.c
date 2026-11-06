/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Stage admission reuses the original 80-byte records and product entry names. */
#include "anemone/exl3_tp.h"

#include "anemone/linear_records.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define BB_EXL3_MAXIMUM_GRID_ROWS 65535u
#define BB_EXL3_FP32_BYTES 4u
#define BB_EXL3_MAXIMUM_PASS_REGIONS 5u
#define BB_EXL3_PRODUCT_NAME_BYTES 64u
#define BB_EXL3_FAST_WARPS 4u

_Static_assert(BB_EXL3_SHARD_ACTIVATION_BYTES <= LAUNCH_ARGUMENT_BYTES,
	       "EXL3 activation fits the backend launch record");

struct exl3_pass_region {
	uint64_t offset;
	uint64_t bytes;
};

static bool pass_regions_valid(const struct exl3_pass_region *regions,
			       uint32_t count, uint64_t arena_bytes)
{
	for (uint32_t region = 0; region < count; region++) {
		if (regions[region].offset % LINEAR_OFFSET_ALIGNMENT_BYTES !=
			    0 ||
		    regions[region].offset > arena_bytes ||
		    regions[region].bytes == 0 ||
		    regions[region].bytes >
			    arena_bytes - regions[region].offset) {
			return false;
		}
		for (uint32_t earlier = 0; earlier < region; earlier++) {
			if (regions[region].offset <
				    regions[earlier].offset +
					    regions[earlier].bytes &&
			    regions[earlier].offset <
				    regions[region].offset +
					    regions[region].bytes) {
				return false;
			}
		}
	}
	return true;
}

result_status
bb_exl3_pass_prepare(const struct bb_exl3_pass_description *description,
		     const struct backend_operations *backend,
		     struct device_card *card, struct kernel_launch *launch_out)
{
	if (description == NULL || description->plan == NULL ||
	    backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || launch_out == NULL ||
	    (uint32_t)description->pass >= BB_EXL3_PASSES) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct bb_exl3_output_shard_plan plan = *description->plan;
	struct bb_exl3_output_shard_plan canonical;
	if (bb_exl3_output_shard_prepare(
		    plan.gate_up.input_size, plan.down.input_size,
		    plan.gate_up.bits_per_weight, plan.rank_count, plan.rank,
		    plan.order, &canonical) != STATUS_OK ||
	    memcmp(&plan, &canonical, sizeof(plan)) != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t tiles = description->product_tiles;
	if ((tiles != 0 && tiles != 1u && tiles != BB_EXL3_PRODUCT_TWO_TILES &&
	     tiles != BB_EXL3_PRODUCT_FOUR_TILES &&
	     tiles != BB_EXL3_PRODUCT_EIGHT_TILES) ||
	    ((tiles == 1u || tiles == BB_EXL3_PRODUCT_EIGHT_TILES) &&
	     plan.order == BB_EXL3_EIGHT_PART_ORDER))
		return STATUS_UNSUPPORTED;
	const struct exl3_expert_record record = description->record;
	const enum bb_exl3_pass pass = description->pass;
	const bool down_shape = pass >= BB_EXL3_DOWN_PASS;
	if (record.reserved_zero != 0 || record.pair_count == 0 ||
	    record.pairs_per_row == 0 ||
	    record.pair_count % record.pairs_per_row != 0 ||
	    record.job_count == 0 ||
	    record.job_count > BB_EXL3_MAXIMUM_GRID_ROWS ||
	    description->weight_offset % WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES !=
		    0 ||
	    record.bits_per_weight != plan.gate_up.bits_per_weight ||
	    record.input_size != (down_shape ? plan.down.output_count :
					       plan.gate_up.input_count) ||
	    record.inner_size != (down_shape ? plan.down.input_count :
					       plan.gate_up.output_count) ||
	    !isfinite(record.activation_limit) ||
	    record.activation_limit <= 0 ||
	    (pass == BB_EXL3_FINISH_PASS &&
	     record.output_offset != record.products_offset)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t outer_values =
		(uint64_t)record.pair_count * record.input_size;
	const uint64_t inner_values =
		(uint64_t)record.pair_count * record.inner_size;
	/* This bound covers both projections and the wider FP32 product plane. */
	if (outer_values > UINT64_MAX / EXL3_PROJECTIONS / BB_EXL3_FP32_BYTES ||
	    inner_values > UINT64_MAX / EXL3_PROJECTIONS / BB_EXL3_FP32_BYTES) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct exl3_pass_region regions[BB_EXL3_MAXIMUM_PASS_REGIONS] = {
		{ description->weight_offset, description->weight_bytes },
		{ record.jobs_offset, (uint64_t)record.job_count *
					      sizeof(struct expert_job_record) }
	};
	uint32_t region_count = EXL3_PROJECTIONS;
	if (pass == BB_EXL3_INPUT_PASS) {
		regions[region_count++] = (struct exl3_pass_region){
			record.input_offset,
			outer_values / record.pairs_per_row * EXL3_HALF_BYTES
		};
	}
	if (pass <= BB_EXL3_DOWN_PASS) {
		const uint64_t operand_values =
			pass <= BB_EXL3_GATE_UP_PASS ?
				outer_values * EXL3_PROJECTIONS :
				inner_values;
		regions[region_count++] = (struct exl3_pass_region){
			record.operands_offset, operand_values * EXL3_HALF_BYTES
		};
	}
	if (pass >= BB_EXL3_GATE_UP_PASS) {
		const uint64_t product_values =
			pass <= BB_EXL3_ACTIVATE_PASS ?
				inner_values * EXL3_PROJECTIONS :
				outer_values;
		regions[region_count++] =
			(struct exl3_pass_region){ record.products_offset,
						   product_values *
							   BB_EXL3_FP32_BYTES };
	}
	if (pass == BB_EXL3_ACTIVATE_PASS) {
		regions[region_count++] = (struct exl3_pass_region){
			record.hidden_offset, inner_values * EXL3_HALF_BYTES
		};
	}
	if (!pass_regions_valid(regions, region_count,
				description->arena_bytes)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const bool optional_order = plan.order == BB_EXL3_EIGHT_PART_ORDER;
	const bool k2 = record.bits_per_weight == EXL3_MINIMUM_BITS;
	const char *const names[BB_EXL3_PASSES] = {
		"exl3_input",
		optional_order ?
			(k2 ? "exl3_gate_up_k2_s8" : "exl3_gate_up_k3_s8") :
			(k2 ? "exl3_gate_up_k2" : "exl3_gate_up_k3"),
		BB_EXL3_SHARD_ACTIVATION_KERNEL,
		optional_order ? (k2 ? "exl3_down_k2_s8" : "exl3_down_k3_s8") :
				 (k2 ? "exl3_down_k2" : "exl3_down_k3"),
		"exl3_finish"
	};
	const bool activation = pass == BB_EXL3_ACTIVATE_PASS;
	const bool product = pass == BB_EXL3_GATE_UP_PASS ||
			     pass == BB_EXL3_DOWN_PASS;
	/* The fast transforms and the wide products (kernel_exl3_wide): with any nonzero tiles, the input and finish
	 * passes take w_input/w_finish; a launch of more than 8 rows (reference order) takes _t4_g4p. */
	const bool fast_transform = tiles != 0 && (pass == BB_EXL3_INPUT_PASS ||
						   pass == BB_EXL3_FINISH_PASS);
	const bool wide = product && tiles != 0 && !optional_order &&
			  record.pair_count / record.pairs_per_row >
				  LINEAR_MAXIMUM_ROWS;
	const uint32_t product_tiles = wide ? BB_EXL3_PRODUCT_FOUR_TILES : tiles;
	const bool wide_module = wide || fast_transform ||
				 (product && tiles == BB_EXL3_PRODUCT_EIGHT_TILES);
	char product_name[BB_EXL3_PRODUCT_NAME_BYTES];
	const char *name = fast_transform ? (pass == BB_EXL3_INPUT_PASS ?
						     "exl3_w_input" :
						     "exl3_w_finish") :
					    names[pass];
	if (product && tiles != 0) {
		const uint32_t width = pass == BB_EXL3_GATE_UP_PASS ?
					       record.inner_size :
					       record.input_size;
		const int length = snprintf(
			product_name, sizeof(product_name),
			"exl3_c_%s_k%u%s_t%u_g%s",
			pass == BB_EXL3_GATE_UP_PASS ? "gate_up" : "down",
			record.bits_per_weight, optional_order ? "_s8" : "",
			product_tiles, wide ? "4p" : "1");
		if (length <= 0 || (size_t)length >= sizeof(product_name) ||
		    width / EXL3_TILE % product_tiles != 0)
			return STATUS_INVALID_ARGUMENT;
		name = product_name;
	}
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card,
		wide_module	      ? BB_EXL3_WIDE_KERNEL_FILE :
		product && tiles != 0 ? BB_EXL3_PRODUCT_KERNEL_FILE :
					EXL3_KERNEL_FILE,
		name,
		activation ? BB_EXL3_SHARD_ACTIVATION_BYTES : sizeof(record),
		&kernel);
	if (status != STATUS_OK) {
		return status;
	}
	if (kernel == BACKEND_NO_KERNEL) {
		return STATUS_UNSUPPORTED;
	}

	const uint32_t width = pass == BB_EXL3_GATE_UP_PASS || activation ?
				       record.inner_size :
				       record.input_size;
	struct kernel_launch launch = {
		.kernel = kernel,
		.grid = { width / (product ? EXL3_TILE : EXL3_TRANSFORM),
			  record.job_count,
			  pass <= BB_EXL3_GATE_UP_PASS ? EXL3_PROJECTIONS : 1 },
		.block = { product ? plan.product_slices * EXL3_WARP :
				     EXL3_TRANSFORM,
			   1, 1 },
		.argument_byte_count = activation ?
					       BB_EXL3_SHARD_ACTIVATION_BYTES :
					       sizeof(record)
	};
	if (product && tiles != 0)
		launch.grid[0] /= product_tiles;
	if (fast_transform) /* one warp a 128-channel block, four a CTA */
		launch.grid[0] = (launch.grid[0] + BB_EXL3_FAST_WARPS - 1u) /
				 BB_EXL3_FAST_WARPS;
	if (activation) {
		const struct bb_exl3_shard_activation_record arguments = {
			.base = record,
			.down_input_size = plan.down.input_count,
			.down_output_size = plan.down.output_count,
			.down_input_begin = plan.gate_up.output_begin
		};
		memcpy(launch.argument, &arguments, sizeof(arguments));
	} else {
		memcpy(launch.argument, &record, sizeof(record));
	}
	*launch_out = launch;
	return STATUS_OK;
}

result_status bb_exl3_row_sum_prepare(
	const struct expert_row_sum_arguments *record, uint32_t rows,
	uint64_t arena_bytes, const struct backend_operations *backend,
	struct device_card *card, struct kernel_launch *launch_out)
{
	if (record == NULL || rows == 0 || rows > BB_EXL3_MAXIMUM_GRID_ROWS ||
	    backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || launch_out == NULL || record->width == 0 ||
	    record->width % EXL3_TRANSFORM != 0 || record->pairs_per_row == 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t row_values = (uint64_t)rows * record->width;
	if (row_values >
	    UINT64_MAX / BB_EXL3_FP32_BYTES / record->pairs_per_row) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct exl3_pass_region regions[] = {
		{ record->pair_output_offset,
		  row_values * record->pairs_per_row * BB_EXL3_FP32_BYTES },
		{ record->row_output_offset, row_values * BB_EXL3_FP32_BYTES }
	};
	if (!pass_regions_valid(regions, EXL3_PROJECTIONS, arena_bytes)) {
		return STATUS_INVALID_ARGUMENT;
	}
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, EXL3_KERNEL_FILE, EXL3_ROW_SUM, sizeof(*record), &kernel);
	if (status != STATUS_OK) {
		return status;
	}
	if (kernel == BACKEND_NO_KERNEL) {
		return STATUS_UNSUPPORTED;
	}
	struct kernel_launch launch = {
		.kernel = kernel,
		.grid = { record->width / EXL3_TRANSFORM, rows, 1 },
		.block = { EXL3_TRANSFORM, 1, 1 },
		.argument_byte_count = sizeof(*record)
	};
	memcpy(launch.argument, record, sizeof(*record));
	*launch_out = launch;
	return STATUS_OK;
}
