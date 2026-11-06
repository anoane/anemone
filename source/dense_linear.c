/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_linear.c: the launches of the dense products. Contract: dense_linear.h; the kernel's: dense_linear_records.h.
 * Byte counts are the linear's, which stop at UINT64_MAX: no product of sizes wraps, and such a count fits no range.
 * The wide instances' names, outputs a block and types come from DENSE_EVERY_WIDE_INSTANCE, the one list of
 * them; each class's choice of them is the table wide_block_outputs_of.
 */
#include "anemone/dense_linear.h"

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/model_layout.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define RANGES_WITHOUT_STEP 3u                  /* the step area is the last range */

#define DENSE_STAMP(name, type, rows, outputs, form) \
	KERNEL_DENSE_LINEAR_STEM #name "_" #rows #form,
static const char *const function_names[] = { DENSE_EVERY_INSTANCE };
#undef DENSE_STAMP
#define DENSE_WIDE_STAMP(name, type, outputs, blocks) \
	KERNEL_DENSE_LINEAR_STEM #name "_wide_" #outputs,
static const char *const wide_function_names[] = { DENSE_EVERY_WIDE_INSTANCE };
#undef DENSE_WIDE_STAMP
#define DENSE_STAGED_STAMP(name, type, outputs) \
	[outputs][type] = KERNEL_DENSE_LINEAR_STEM #name "_staged_" #outputs,
static const char *const staged_function_names[DENSE_STAGED_MOST_OUTPUTS +
					       1u][DENSE_TYPE_COUNT] = {
	DENSE_EVERY_STAGED_INSTANCE
};
#undef DENSE_STAGED_STAMP
#define DENSE_WIDE_STAMP(name, type, outputs, blocks) { type, outputs },
static const struct wide_form {
    uint32_t weight_type, block_outputs;
} wide_forms[] = { DENSE_EVERY_WIDE_INSTANCE };
#undef DENSE_WIDE_STAMP
/* The name a refusal gives a wide form of a weight type that is not prepared. */
static const char *const wide_type_names[DENSE_TYPE_COUNT] = {
	[DENSE_BFLOAT16] = KERNEL_DENSE_LINEAR_STEM "bfloat16_wide",
	[DENSE_FLOAT32] = KERNEL_DENSE_LINEAR_STEM "float32_wide"
};
/* The outputs a block of each weight type's wide form, on class 80 and on class 120 (either form): dense_linear.h. */
enum { WIDE_CLASS_80, WIDE_CLASS_120, WIDE_CLASSES };
static const uint32_t wide_block_outputs_of[WIDE_CLASSES][DENSE_TYPE_COUNT] = {
	[WIDE_CLASS_80] = { [DENSE_BFLOAT16] = 1, [DENSE_FLOAT32] = 8 },
	[WIDE_CLASS_120] = { [DENSE_BFLOAT16] = 8, [DENSE_FLOAT32] = 4 }
};
/* A skinny matrix's (DENSE_SKINNY_MOST_OUTPUTS_VALUE outputs at most: the routers, the hyper mixes) on class 80, to
 * DENSE_SKINNY_MANY_ROWS rows and above; 0 keeps the wide form. dense01: the routers are fastest at the wide
 * form's 1 (48 rows 58.4 us; 8: 80.9); the hyper mixes at 1 (12 rows 16.4 us, 24 18.4, 48 28.7; the wide form's 8:
 * 37.9) and at 4 above 48 rows (64: 29.7 against 39.9). */
static const uint32_t
	skinny_block_outputs_of[DENSE_TYPE_COUNT][DENSE_SKINNY_REGIMES] = {
		[DENSE_BFLOAT16] = { 0, 0 },
		[DENSE_FLOAT32] = { 1, 4 }
	};

_Static_assert(
	COUNT_OF(function_names) == DENSE_INSTANCES &&
		DENSE_FORM_INSTANCES ==
			DENSE_TYPE_COUNT * DENSE_ROW_INSTANCES &&
		DENSE_INSTANCES == DENSE_FORMS * DENSE_FORM_INSTANCES &&
		COUNT_OF(wide_function_names) == DENSE_WIDE_INSTANCES &&
		COUNT_OF(wide_forms) == DENSE_WIDE_INSTANCES &&
		DENSE_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES,
	"a function for each type and row count, and of each wide form; the record in a launch");

/* The bytes of [vector_count][width] of a type, by enum dense_type_code: UINT64_MAX beyond 64 bits. */
static uint64_t (*const byte_count_of[DENSE_TYPE_COUNT])(uint64_t, uint64_t) = {
	[DENSE_BFLOAT16] = linear_values_byte_count,
	[DENSE_FLOAT32] = linear_sums_byte_count
};

result_status
dense_linear_kernels_prepare(const struct backend_operations *backend,
			     struct device_card *card,
			     struct dense_linear_kernels *kernels_out)
{
	struct dense_linear_kernels kernels;
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR(
			"the kernels of the dense products: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
    kernels.staged_block_outputs = 0;
    for (uint32_t type = 0; type < DENSE_TYPE_COUNT;
	 type++) { /* wide_kernels_prepare's */
	    kernels.wide_products[type] = BACKEND_NO_KERNEL;
	    kernels.wide_block_outputs[type] = 0;
	    kernels.staged_products[type] = BACKEND_NO_KERNEL;
	    for (uint32_t regime = 0; regime < DENSE_SKINNY_REGIMES; regime++) {
		    kernels.skinny_products[type][regime] = BACKEND_NO_KERNEL;
		    kernels.skinny_block_outputs[type][regime] = 0;
	    }
    }
    for (uint32_t instance = 0; instance < DENSE_INSTANCES; instance++) {
	    kernels.products[instance] =
		    BACKEND_NO_KERNEL; /* what a backend that gives no handle leaves */
	    const result_status status = backend->kernel_prepare(
		    card, KERNEL_FILE_DENSE_LINEAR, function_names[instance],
		    DENSE_ARGUMENT_BYTES, &kernels.products[instance]);
	    if (status != STATUS_OK) {
		    LOG_ERROR(
			    "the kernels of the dense products: %s is not prepared: %s",
			    function_names[instance], status_text(status));
		    return status;
	    }
    }
    *kernels_out = kernels;
    return STATUS_OK;
}

result_status dense_linear_wide_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct dense_linear_kernels *kernels)
{
	const bool is_class_120 = arithmetic_class == ARITHMETIC_CLASS_SM_120 ||
				  arithmetic_class ==
					  ARITHMETIC_CLASS_SM_120_NATIVE;
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels == NULL ||
	    (arithmetic_class != ARITHMETIC_CLASS_SM_80 && !is_class_120)) {
		LOG_ERROR(
			"the wide kernels of the dense products: a missing argument, or class %" PRIu32
			" (classes 80, 120 and 120 native take them)",
			arithmetic_class);
		return STATUS_INVALID_ARGUMENT;
	}
    const uint32_t *const class_outputs = wide_block_outputs_of[is_class_120 ? WIDE_CLASS_120 : WIDE_CLASS_80];
    struct dense_linear_kernels prepared = *kernels;
    for (uint32_t form = 0; form < DENSE_WIDE_INSTANCES; form++) {
	    const uint32_t type = wide_forms[form].weight_type;
	    for (uint32_t regime = 0;
		 regime < DENSE_SKINNY_REGIMES && !is_class_120; regime++) {
		    if (skinny_block_outputs_of[type][regime] == 0 ||
			wide_forms[form].block_outputs !=
				skinny_block_outputs_of[type][regime])
			    continue;
		    const result_status status = backend->kernel_prepare(
			    card, KERNEL_FILE_DENSE_LINEAR,
			    wide_function_names[form], DENSE_ARGUMENT_BYTES,
			    &prepared.skinny_products[type][regime]);
		    if (status != STATUS_OK) {
			    LOG_ERROR(
				    "the wide kernels of the dense products: %s is not prepared: %s",
				    wide_function_names[form],
				    status_text(status));
			    return status;
		    }
		    prepared.skinny_block_outputs[type][regime] =
			    wide_forms[form].block_outputs;
	    }
	    if (wide_forms[form].block_outputs != class_outputs[type]) {
		    continue; /* another class's form: stamped for the card tests */
	    }
	    prepared.wide_products[type] = BACKEND_NO_KERNEL;
	    const result_status status = backend->kernel_prepare(
		    card, KERNEL_FILE_DENSE_LINEAR, wide_function_names[form],
		    DENSE_ARGUMENT_BYTES, &prepared.wide_products[type]);
	    if (status != STATUS_OK) {
		    LOG_ERROR(
			    "the wide kernels of the dense products: %s is not prepared: %s",
			    wide_function_names[form], status_text(status));
		    return status;
	    }
	    prepared.wide_block_outputs[type] = wide_forms[form].block_outputs;
    }
    *kernels = prepared;
    return STATUS_OK;
}

result_status
dense_linear_staged_kernels_prepare(const struct backend_operations *backend,
				    struct device_card *card, uint32_t staged,
				    struct dense_linear_kernels *kernels)
{
    if (staged == 0)
	    return STATUS_OK;
    if (backend == NULL || backend->kernel_prepare == NULL || card == NULL ||
	kernels == NULL || staged > DENSE_STAGED_MOST_OUTPUTS ||
	staged_function_names[staged][0] == NULL) {
	    LOG_ERROR("the staged dense forms: a missing argument, or %" PRIu32
		      " outputs a block (8,16,32,64)",
		      staged);
	    return STATUS_INVALID_ARGUMENT;
    }
    uint32_t handles[DENSE_TYPE_COUNT];
    for (uint32_t type = 0; type < DENSE_TYPE_COUNT; type++) {
	    handles[type] = BACKEND_NO_KERNEL;
	    const result_status status = backend->kernel_prepare(
		    card, KERNEL_FILE_DENSE_LINEAR,
		    staged_function_names[staged][type], DENSE_ARGUMENT_BYTES,
		    &handles[type]);
	    if (status != STATUS_OK)
		    return status;
    }
    memcpy(kernels->staged_products, handles, sizeof(handles));
    kernels->staged_block_outputs = staged;
    return STATUS_OK;
}

result_status
dense_linear_skinny_kernel_prepare(const struct backend_operations *backend,
				   struct device_card *card, uint32_t type,
				   uint32_t block_outputs,
				   struct dense_linear_kernels *kernels)
{
	const uint32_t regime = DENSE_SKINNY_REGIMES -
				1u; /* above DENSE_SKINNY_MANY_ROWS rows */
	if (block_outputs == 0)
		return STATUS_OK;
	for (uint32_t form = 0;
	     backend != NULL && card != NULL && kernels != NULL &&
	     type < DENSE_TYPE_COUNT && form < DENSE_WIDE_INSTANCES;
	     form++) {
		if (wide_forms[form].weight_type != type ||
		    wide_forms[form].block_outputs != block_outputs)
			continue;
		uint32_t handle = BACKEND_NO_KERNEL;
		const result_status status =
			backend->kernel_prepare(card, KERNEL_FILE_DENSE_LINEAR,
						wide_function_names[form],
						DENSE_ARGUMENT_BYTES, &handle);
		if (status != STATUS_OK)
			return status;
		kernels->skinny_products[type][regime] = handle;
		kernels->skinny_block_outputs[type][regime] = block_outputs;
		return STATUS_OK;
	}
	LOG_ERROR("the skinny dense form of type %" PRIu32 " at %" PRIu32
		  " outputs a block: a missing argument, or"
		  " no wide form of them",
		  type, block_outputs);
	return STATUS_INVALID_ARGUMENT;
}

/* The types, flags, counts and scales of a description: true, or false behind the one error line of the rule. */
static bool
numbers_are_valid(const char *what,
		  const struct dense_linear_description *description)
{
    /* a chunk's rows in a wide launch; the head, which reads the step rows, is never one */
    const uint32_t most_rows = (description->flags & DENSE_STEP_ROWS) != 0 ?
				       LINEAR_MAXIMUM_ROWS :
				       MODEL_WIDE_MOST_ROWS;
    /* the cone's rows: a wide launch's, of a decoder layer, never with the head's step rows */
    const bool is_cone = (description->flags & DENSE_CONE_ROWS) != 0;
    const bool cone_is_valid =
	    is_cone ? description->row_count > LINEAR_MAXIMUM_ROWS &&
			      (description->flags & DENSE_STEP_ROWS) == 0 &&
			      description->cone_layer >=
				      MODEL_DECODER_FIRST_LAYER &&
			      description->cone_layer < MODEL_LAYERS :
		      description->cone_layer == 0;
    const bool is_valid =
	    cone_is_valid && description->weight_type < DENSE_TYPE_COUNT &&
	    description->output_type < DENSE_TYPE_COUNT &&
	    (description->flags & ~DENSE_FLAGS) == 0 &&
	    description->output_size != 0 &&
	    description->output_size % DENSE_OUTPUTS_PER_BLOCK == 0 &&
	    description->input_size != 0 &&
	    description->input_size % DENSE_CHUNK_INPUTS == 0 &&
	    description->row_count != 0 &&
	    description->row_count <= most_rows &&
	    isfinite(description->output_scale) &&
	    description->output_scale > 0.0f &&
	    isfinite(description->epsilon) && description->epsilon >= 0.0f;
    if (!is_valid) {
	    LOG_ERROR(
		    "%s: types %" PRIu32 " and %" PRIu32 ", flags 0x%" PRIx32
		    ", %" PRIu32 " by %" PRIu32 ", %" PRIu32
		    " rows, scale %g, epsilon %g: known types and flags, outputs in blocks of %u, "
		    "inputs in chunks of %u, 1 to %u rows (with the step rows to %u), a finite scale above zero "
		    "and epsilon not below",
		    what, description->weight_type, description->output_type,
		    description->flags, description->output_size,
		    description->input_size, description->row_count,
		    (double)description->output_scale,
		    (double)description->epsilon, DENSE_OUTPUTS_PER_BLOCK,
		    DENSE_CHUNK_INPUTS, MODEL_WIDE_MOST_ROWS,
		    LINEAR_MAXIMUM_ROWS);
    }
    return is_valid;
}

result_status
dense_linear_prepare(const struct dense_linear_description *description,
		     uint64_t arena_byte_count,
		     const struct dense_linear_kernels *kernels,
		     struct kernel_launch *launch_out)
{
    static const char what[] = "a dense product";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out) ||
	!numbers_are_valid(what, description)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t row_count = description->row_count;
    const bool is_narrow = description->output_size / DENSE_OUTPUTS_PER_BLOCK <
			   DENSE_NARROW_BELOW_BLOCKS;
    const bool is_wide = row_count > LINEAR_MAXIMUM_ROWS;
    const uint32_t groups =
	    (row_count + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
    const uint32_t instance = (is_narrow ? DENSE_FORM_INSTANCES : 0u) +
			      description->weight_type * DENSE_ROW_INSTANCES +
			      (row_count > 1 ? 1u : 0u);
    const uint32_t type = description->weight_type;
    const uint32_t regime = row_count > DENSE_SKINNY_MANY_ROWS ? 1u : 0u;
    const bool is_skinny =
	    is_wide &&
	    description->output_size <= DENSE_SKINNY_MOST_OUTPUTS_VALUE &&
	    kernels->skinny_block_outputs[type][regime] != 0 &&
	    description->output_size %
			    kernels->skinny_block_outputs[type][regime] ==
		    0;
    /* --dense_staged: a wide product of bfloat16 weights (rk06_card: float32 ones 2-4% slower staged), a chunk's rows,
     * few enough inputs, outputs in whole staged blocks and no row scale */
    const bool is_staged =
	    row_count >= DENSE_STAGED_LEAST_ROWS && type == DENSE_BFLOAT16 &&
	    kernels->staged_products[type] != BACKEND_NO_KERNEL &&
	    kernels->staged_block_outputs != 0 &&
	    (description->flags & DENSE_ROW_SCALE) == 0 &&
	    description->input_size <= DENSE_STAGED_MOST_INPUTS &&
	    description->output_size % kernels->staged_block_outputs == 0;
    const uint32_t block_outputs =
	    is_staged ? kernels->staged_block_outputs :
	    is_skinny ? kernels->skinny_block_outputs[type][regime] :
	    is_wide   ? kernels->wide_block_outputs[type] :
	    is_narrow ? 1u :
			DENSE_OUTPUTS_PER_BLOCK;
    const uint32_t kernel = is_staged ? kernels->staged_products[type] :
                            is_skinny ? kernels->skinny_products[type][regime] :
                            is_wide   ? kernels->wide_products[type] : kernels->products[instance];
    const uint32_t staged_bytes =
	    is_staged ?
		    description->input_size *
			    (uint32_t)(LINEAR_MAXIMUM_ROWS * sizeof(uint16_t)) :
		    0u;
    const char *const function_name = is_wide ? wide_type_names[description->weight_type] : function_names[instance];
    const bool reads_step =
	    (description->flags & (DENSE_STEP_ROWS | DENSE_CONE_ROWS)) != 0;
    const struct checked_range ranges[] = {
	    { "the weights", &description->weights,
	      byte_count_of[description->weight_type](description->output_size,
						      description->input_size),
	      true },
	    { "the rows", &description->rows,
	      linear_values_byte_count(row_count, description->input_size),
	      false },
	    { "the output", &description->output,
	      byte_count_of[description->output_type](row_count,
						      description->output_size),
	      false },
	    { "the step area", &description->step,
	      STEP_BLOCK_BYTES_VALUE +
		      (uint64_t)row_count * STEP_ROW_BYTES_VALUE,
	      false },
    };
    if (!prepare_kernel_is_prepared(what, kernel, function_name) ||
	!prepare_ranges_are_valid(what, ranges,
				  reads_step ? COUNT_OF(ranges) :
					       RANGES_WITHOUT_STEP,
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (block_outputs == 0 || description->output_size % block_outputs != 0) {
	    LOG_ERROR("%s: %" PRIu32
		      " outputs: the wide form of its weights takes %" PRIu32
		      " outputs a block, "
		      "which must divide them",
		      what, description->output_size, block_outputs);
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct dense_linear_arguments record = {
	    .weights_offset = description->weights.byte_offset,
	    .rows_offset = description->rows.byte_offset,
	    .output_offset = description->output.byte_offset,
	    .step_offset = reads_step ? description->step.byte_offset : 0,
	    .output_size = description->output_size,
	    .input_size = description->input_size,
	    .row_count = row_count,
	    .output_type = description->output_type,
	    .flags = description->flags,
	    .output_scale = description->output_scale,
	    .epsilon = description->epsilon,
	    .cone_layer = description->cone_layer
    };
    prepare_launch_fill(launch_out, kernel,
			description->output_size / block_outputs,
			is_wide ? groups : 1u, DENSE_THREADS_VALUE,
			staged_bytes, &record, DENSE_ARGUMENT_BYTES);
    return STATUS_OK;
}
