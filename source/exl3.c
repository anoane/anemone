/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* EXL3 format sizes and independent reconstruction launch preparation. */
#include "anemone/exl3.h"

#include <stddef.h>
#include <string.h>

#define RECONSTRUCT_THREADS 256u

MUST_CHECK static result_status
exl3_stored_byte_count(const struct matrix_shape_record *shape,
		       uint64_t *byte_count_out)
{
	if (shape == NULL || byte_count_out == NULL ||
	    shape->reserved_must_be_zero != 0 || shape->input_size == 0 ||
	    shape->output_size == 0 ||
	    shape->input_size % EXL3_TRANSFORM != 0 ||
	    shape->output_size % EXL3_TRANSFORM != 0 ||
	    shape->bits_per_weight < EXL3_MINIMUM_BITS ||
	    shape->bits_per_weight > EXL3_MAXIMUM_BITS) {
		return STATUS_INVALID_ARGUMENT;
	}
	/* Divide before multiplying: even the largest uint32 shapes cannot overflow. */
	const uint64_t codes = (uint64_t)shape->input_size *
			       shape->output_size / EXL3_BITS_PER_BYTE *
			       shape->bits_per_weight;
	const uint64_t vectors =
		((uint64_t)shape->input_size + shape->output_size) *
		EXL3_HALF_BYTES;
	if (codes > UINT64_MAX - vectors) {
		return STATUS_INVALID_ARGUMENT;
	}
	*byte_count_out = codes + vectors;
	return STATUS_OK;
}

const struct weight_format_operations exl3_weight_operations = {
	.name = "exl3_mul1",
	.stored_byte_count = exl3_stored_byte_count
};

result_status
exl3_reconstruct_prepare(const struct exl3_reconstruct_record *record,
			 uint64_t arena_byte_count, uint32_t kernel,
			 struct kernel_launch *launch_out)
{
	uint64_t matrix_bytes = 0;
	if (record == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL || record->reserved_zero != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct matrix_shape_record shape = { record->output_size,
						   record->input_size,
						   record->bits_per_weight, 0 };
	const uint64_t values = (uint64_t)shape.input_size * shape.output_size;
	if (exl3_stored_byte_count(&shape, &matrix_bytes) != STATUS_OK ||
	    values > UINT64_MAX / EXL3_HALF_BYTES) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t output_bytes = values * EXL3_HALF_BYTES;
	if (record->matrix_offset % WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES != 0 ||
	    record->output_offset % WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES != 0 ||
	    record->matrix_offset > arena_byte_count ||
	    matrix_bytes > arena_byte_count - record->matrix_offset ||
	    record->output_offset > arena_byte_count ||
	    output_bytes > arena_byte_count - record->output_offset ||
	    (record->matrix_offset < record->output_offset + output_bytes &&
	     record->output_offset < record->matrix_offset + matrix_bytes) ||
	    values / RECONSTRUCT_THREADS > UINT32_MAX) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct kernel_launch launch = {
		.kernel = kernel,
		.grid = { (uint32_t)((values + RECONSTRUCT_THREADS - 1u) /
				     RECONSTRUCT_THREADS),
			  1, 1 },
		.block = { RECONSTRUCT_THREADS, 1, 1 },
		.argument_byte_count = sizeof(*record)
	};
	memcpy(launch.argument, record, sizeof(*record));
	*launch_out = launch;
	return STATUS_OK;
}

#include <math.h>
#include <stdio.h>
#include "prepare_common.h"

#define EXL3_OPERAND_BYTES_PER_INPUT 4u
#define EXL3_PRODUCT_BYTES_PER_INPUT 8u
#define EXL3_SCRATCH_BYTES_PER_INPUT \
	(EXL3_OPERAND_BYTES_PER_INPUT + EXL3_PRODUCT_BYTES_PER_INPUT)
#define EXL3_SCALAR_BYTES 4u
#define EXL3_MAXIMUM_GRID_ROWS 65535u
#define EXL3_PROJECTION_AXIS 2u
#define EXL3_FAST_WARPS 4u		/* blocks of the w_* transforms: one warp a 128-channel block */
#define EXL3_FAST_THIN_TILES 8u		/* output tiles a warp, thin products (_t8_g1) */
#define EXL3_FAST_WIDE_TILES 4u		/* output tiles a warp, wide products (_t4_g4p) */
#define EXL3_KERNEL_NAME_BYTES 64u

enum exl3_stage {
	EXL3_INPUT = 0,
	EXL3_GATE_UP = 1,
	EXL3_ACTIVATE = 2,
	EXL3_DOWN = 3,
	EXL3_FINISH = 4
};
enum exl3_range {
	EXL3_JOBS_RANGE = 0,
	EXL3_INPUT_RANGE = 1,
	EXL3_OPERANDS_RANGE = 2,
	EXL3_PRODUCTS_RANGE = 3,
	EXL3_HIDDEN_RANGE = 4
};

uint64_t exl3_scratch_byte_count(uint32_t pair_count, uint32_t input_size)
{
	const uint64_t values = (uint64_t)pair_count * input_size;
	return values > UINT64_MAX / EXL3_SCRATCH_BYTES_PER_INPUT ?
		       UINT64_MAX :
		       values * EXL3_SCRATCH_BYTES_PER_INPUT;
}

enum exl3_products { EXL3_RETAINED, EXL3_RELAXED, EXL3_FAST };

MUST_CHECK static result_status experts_prepare(
	const struct exl3_expert_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch *launches_out, enum exl3_products products)
{
	const bool relaxed = products == EXL3_RELAXED,
		   fast = products == EXL3_FAST;
	if (record == NULL || backend == NULL || card == NULL ||
	    launches_out == NULL || backend->kernel_prepare == NULL ||
	    record->reserved_zero != 0 || record->pair_count == 0 ||
	    record->pairs_per_row == 0 ||
	    record->pair_count % record->pairs_per_row != 0 ||
	    record->output_offset != record->products_offset ||
	    record->job_count == 0 ||
	    record->job_count > EXL3_MAXIMUM_GRID_ROWS ||
	    record->inner_size > record->input_size ||
	    !isfinite(record->activation_limit) ||
	    record->activation_limit <= 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct matrix_shape_record shape = { record->inner_size,
						   record->input_size,
						   record->bits_per_weight, 0 };
	uint64_t bytes = 0;
	if (exl3_stored_byte_count(&shape, &bytes) != STATUS_OK) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t inputs =
		(uint64_t)record->pair_count * record->input_size;
	if (inputs > UINT64_MAX / EXL3_SCRATCH_BYTES_PER_INPUT) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct linear_range ranges[] = {
		{ record->jobs_offset,
		  (uint64_t)record->job_count *
			  sizeof(struct expert_job_record) },
		{ record->input_offset,
		  inputs / record->pairs_per_row * EXL3_HALF_BYTES },
		{ record->operands_offset,
		  inputs * EXL3_OPERAND_BYTES_PER_INPUT },
		{ record->products_offset,
		  inputs * EXL3_PRODUCT_BYTES_PER_INPUT },
		{ record->hidden_offset, (uint64_t)record->pair_count *
						 record->inner_size *
						 EXL3_HALF_BYTES }
	};
	const struct checked_range checks[] = {
		{ "EXL3 jobs", &ranges[EXL3_JOBS_RANGE],
		  ranges[EXL3_JOBS_RANGE].byte_count, false },
		{ "EXL3 input", &ranges[EXL3_INPUT_RANGE],
		  ranges[EXL3_INPUT_RANGE].byte_count, false },
		{ "EXL3 operands", &ranges[EXL3_OPERANDS_RANGE],
		  ranges[EXL3_OPERANDS_RANGE].byte_count, false },
		{ "EXL3 products", &ranges[EXL3_PRODUCTS_RANGE],
		  ranges[EXL3_PRODUCTS_RANGE].byte_count, false },
		{ "EXL3 hidden", &ranges[EXL3_HIDDEN_RANGE],
		  ranges[EXL3_HIDDEN_RANGE].byte_count, false }
	};
	if (!prepare_ranges_are_valid(
		    "EXL3 experts", checks,
		    (uint32_t)(sizeof(checks) / sizeof(checks[0])),
		    arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const bool k2 = record->bits_per_weight == EXL3_MINIMUM_BITS;
	const char *const names[EXL3_EXPERT_LAUNCHES] = {
		"exl3_input",
		relaxed ? (k2 ? "exl3_gate_up_k2_s8" : "exl3_gate_up_k3_s8") :
			  (k2 ? "exl3_gate_up_k2" : "exl3_gate_up_k3"),
		"exl3_activate",
		relaxed ? (k2 ? "exl3_down_k2_s8" : "exl3_down_k3_s8") :
			  (k2 ? "exl3_down_k2" : "exl3_down_k3"),
		"exl3_finish"
	};
	/* The fast form: thin launches (at most 8 rows) take _t8_g1, wide ones _t4_g4p. */
	const bool wide = record->pair_count / record->pairs_per_row >
			  LINEAR_MAXIMUM_ROWS;
	const uint32_t fast_tiles =
		wide ? EXL3_FAST_WIDE_TILES : EXL3_FAST_THIN_TILES;
	if (fast && (record->inner_size % (EXL3_TILE * fast_tiles) != 0 ||
		     record->input_size % (EXL3_TILE * fast_tiles) != 0)) {
		return STATUS_UNSUPPORTED;
	}
	char gate_up_name[EXL3_KERNEL_NAME_BYTES], down_name[EXL3_KERNEL_NAME_BYTES];
	const char *const product_suffix = wide ? "t4_g4p" : "t8_g1";
	if (snprintf(gate_up_name, sizeof(gate_up_name),
		     "exl3_c_gate_up_k%u_%s", record->bits_per_weight,
		     product_suffix) <= 0 ||
	    snprintf(down_name, sizeof(down_name), "exl3_c_down_k%u_%s",
		     record->bits_per_weight, product_suffix) <= 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const char *const fast_names[EXL3_EXPERT_LAUNCHES] = {
		"exl3_w_input", gate_up_name, "exl3_w_activate", down_name,
		"exl3_w_finish"
	};
	struct kernel_launch launches[EXL3_EXPERT_LAUNCHES];
	for (uint32_t index = 0; index < EXL3_EXPERT_LAUNCHES; index++) {
		uint32_t kernel = BACKEND_NO_KERNEL;
		const result_status status = backend->kernel_prepare(
			card, fast ? EXL3_WIDE_KERNEL_FILE : EXL3_KERNEL_FILE,
			fast ? fast_names[index] : names[index],
			sizeof(*record), &kernel);
		if (status != STATUS_OK) {
			return status;
		}
		if (kernel == BACKEND_NO_KERNEL) {
			return STATUS_UNSUPPORTED;
		}
		const bool product = index == 1u || index == EXL3_MAXIMUM_BITS;
		const bool inner = index == 1u || index == EXL3_MINIMUM_BITS;
		const uint32_t width = inner ? record->inner_size :
					       record->input_size;
		const uint32_t blocks = width / EXL3_TRANSFORM;
		const uint32_t grid_x =
			!fast ? width / (product ? EXL3_TILE : EXL3_TRANSFORM) :
			product ? width / EXL3_TILE / fast_tiles :
				  (blocks + EXL3_FAST_WARPS - 1u) /
					  EXL3_FAST_WARPS;
		prepare_launch_fill(&launches[index], kernel, grid_x,
				    record->job_count,
				    product && relaxed ?
					    EXL3_RELAXED_SLICES * EXL3_WARP :
					    EXL3_TRANSFORM,
				    0, record, sizeof(*record));
		launches[index].grid[EXL3_PROJECTION_AXIS] =
			index <= EXL3_GATE_UP ? EXL3_PROJECTIONS : 1u;
	}
	memcpy(launches_out, launches, sizeof(launches));
	return STATUS_OK;
}

result_status exl3_experts_prepare(const struct exl3_expert_record *record,
				   uint64_t arena_byte_count,
				   const struct backend_operations *backend,
				   struct device_card *card,
				   struct kernel_launch *launches_out)
{
	return experts_prepare(record, arena_byte_count, backend, card,
			       launches_out, EXL3_RETAINED);
}

result_status exl3_experts_fast_prepare(
	const struct exl3_expert_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch *launches_out)
{
	return experts_prepare(record, arena_byte_count, backend, card,
			       launches_out, EXL3_FAST);
}

result_status exl3_experts_relaxed_prepare(
	const struct exl3_expert_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch *launches_out)
{
	return experts_prepare(record, arena_byte_count, backend, card,
			       launches_out, EXL3_RELAXED);
}
