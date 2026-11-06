/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear.c: the launches of the linear, prepared from descriptions. Contract: linear.h; the rules of the kernels,
 * their records and their launches: linear_records.h.
 *
 *   - A prepare function runs in three parts: the counts and matrices of its description, then its ranges against
 *     what the launches touch and against the arena, then the launches. Nothing is written before every check has
 *     passed, and each check that fails writes the one error line of the refusal.
 *   - Every byte count is a product in 64 bits that stops at UINT64_MAX, and a range that is not a matrix refuses
 *     that count whatever its size. The checks every family shares, those of the ranges among them, and the filling
 *     of a launch are prepare_common.h's.
 *   - The bytes of a matrix are asked of its weight format; nothing here computes with a weight.
 *   - A record is built whole, its reserved member zero, and copied into a launch whose other bytes are zero.
 */
#include "anemone/linear.h"

#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/model_layout.h"
#include "anemone/relaxed_linear.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

/* The kernels of the linear in the order they are prepared, the order of the members of struct
 * linear_kernels_record: each family has one kernel, or one for each slice count (a column of slice_instances). The
 * wide trunk families come last: linear_wide_kernels_prepare's, from FAMILY_WIDE_FIRST on (those a class's
 * wide forms take). */
enum linear_family {
    FAMILY_ACTIVATION_QUANTIZE,
    FAMILY_EXPERT_GATE_UP,
    FAMILY_EXPERT_DOWN,
    FAMILY_EXPERT_ROW_SUM,
    FAMILY_TRUNK_LINEAR_QUANTIZED,
    FAMILY_TRUNK_LINEAR_BFLOAT16,
    FAMILY_TRUNK_HIDDEN,
    FAMILY_TRUNK_LINEAR_QUANTIZED_WIDE,
    FAMILY_TRUNK_LINEAR_BFLOAT16_WIDE,
    FAMILY_TRUNK_LINEAR_QUANTIZED_INTERLEAVED,  /* of one slice, in place of a wide launch (its wide form) */
    FAMILY_COUNT
};
#define FAMILY_WIDE_FIRST FAMILY_TRUNK_LINEAR_QUANTIZED_WIDE
enum stamped_column {
    GATE_UP_COLUMN,
    DOWN_COLUMN,
    TRUNK_QUANTIZED_COLUMN,
    TRUNK_BFLOAT16_COLUMN,
    TRUNK_QUANTIZED_WIDE_COLUMN,
    TRUNK_BFLOAT16_WIDE_COLUMN,
    STAMPED_COLUMNS
};
#define NOT_STAMPED STAMPED_COLUMNS
#define INSTANCES LINEAR_SLICE_INSTANCES
#define KERNEL_COUNT (FAMILY_COUNT - STAMPED_COLUMNS + STAMPED_COLUMNS * INSTANCES)
#define RANGES_OF_A_MATRIX 2u                   /* of a trunk launch: the matrix and its output */

struct kernel_family {
    const char *kernel_file_name;
    const char *native_file_name;               /* class 120 native: the build in the PRO's own forms */
    const char *function_name;                  /* of a family of one kernel */
    uint32_t argument_byte_count;
    uint32_t stamped_column;                    /* or NOT_STAMPED */
};

#define EXPERT_FILES KERNEL_FILE_EXPERT_LINEAR, KERNEL_FILE_EXPERT_LINEAR_NATIVE
#define TRUNK_FILES KERNEL_FILE_TRUNK_LINEAR, KERNEL_FILE_TRUNK_LINEAR_NATIVE
static const struct kernel_family kernel_families[FAMILY_COUNT] = {
	[FAMILY_ACTIVATION_QUANTIZE] = { KERNEL_FILE_ACTIVATION_QUANTIZE,
					 KERNEL_FILE_ACTIVATION_QUANTIZE,
					 KERNEL_ACTIVATION_QUANTIZE,
					 ACTIVATION_QUANTIZE_ARGUMENT_BYTES,
					 NOT_STAMPED },
	[FAMILY_EXPERT_GATE_UP] = { EXPERT_FILES, NULL,
				    EXPERT_GATE_UP_ARGUMENT_BYTES,
				    GATE_UP_COLUMN },
	[FAMILY_EXPERT_DOWN] = { EXPERT_FILES, NULL, EXPERT_DOWN_ARGUMENT_BYTES,
				 DOWN_COLUMN },
	[FAMILY_EXPERT_ROW_SUM] = { EXPERT_FILES, KERNEL_EXPERT_ROW_SUM,
				    EXPERT_ROW_SUM_ARGUMENT_BYTES,
				    NOT_STAMPED },
	[FAMILY_TRUNK_LINEAR_QUANTIZED] = { TRUNK_FILES, NULL,
					    TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES,
					    TRUNK_QUANTIZED_COLUMN },
	[FAMILY_TRUNK_LINEAR_BFLOAT16] = { TRUNK_FILES, NULL,
					   TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES,
					   TRUNK_BFLOAT16_COLUMN },
	[FAMILY_TRUNK_HIDDEN] = { TRUNK_FILES, KERNEL_TRUNK_HIDDEN_VALUE,
				  TRUNK_HIDDEN_ARGUMENT_BYTES, NOT_STAMPED },
	[FAMILY_TRUNK_LINEAR_QUANTIZED_WIDE] = { TRUNK_FILES, NULL,
						 TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES,
						 TRUNK_QUANTIZED_WIDE_COLUMN },
	[FAMILY_TRUNK_LINEAR_BFLOAT16_WIDE] = { TRUNK_FILES, NULL,
						TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES,
						TRUNK_BFLOAT16_WIDE_COLUMN },
	[FAMILY_TRUNK_LINEAR_QUANTIZED_INTERLEAVED] = { TRUNK_FILES,
							KERNEL_TRUNK_LINEAR_QUANTIZED_INTERLEAVED,
							TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES,
							NOT_STAMPED },
};
#undef EXPERT_FILES
#undef TRUNK_FILES

/* Each count the product kernels are stamped for, with the function of each stamped family for it. */
struct slice_instance {
    uint32_t slice_count;
    const char *function_names[STAMPED_COLUMNS];
};
#define LINEAR_SLICE_STAMP(slices)                           \
	{ (slices),                                          \
	  { KERNEL_EXPERT_GATE_UP_STEM #slices,              \
	    KERNEL_EXPERT_DOWN_STEM #slices,                 \
	    KERNEL_TRUNK_LINEAR_QUANTIZED_STEM #slices,      \
	    KERNEL_TRUNK_LINEAR_BFLOAT16_STEM #slices,       \
	    KERNEL_TRUNK_LINEAR_QUANTIZED_WIDE_STEM #slices, \
	    KERNEL_TRUNK_LINEAR_BFLOAT16_WIDE_STEM #slices } },
static const struct slice_instance slice_instances[] = {
	LINEAR_EVERY_SLICE_COUNT
};
#undef LINEAR_SLICE_STAMP
#define LINEAR_SLICE_STAMP(slices) " " #slices
static const char stamped_counts_text[] =
	LINEAR_EVERY_SLICE_COUNT; /* " 1 2 ..": for an error line */
#undef LINEAR_SLICE_STAMP

/* The default slice counts: a row per kernel family, shape and class of card, a count per profile. From batches i01
 * and i04; the experts from the sweep of 25 October 2026: on class 80 one slice, a warp's own chain (gate/up 136 ->
 * 84 us at a row, 1,019 -> 693 at 8; down 67 -> 48 and 488 -> 295); on class 120, with its whole-number groups,
 * gate/up 2 (131 us at a row, 1,020 at 8). Every other row confirmed by it.
 * A trunk row then gives a wide launch (linear_default_wide_form) its count (0: the profile's) and form: the
 * fastest of the trunk wide card test at 1,024 rows after a flush of the L2 (26 October 2026, wide_wiring_v1's
 * CARD_TIMES.txt; counts other than the default measured for attn.wq_a with attn.wkv and attn.wo_a alone). Thin
 * launches of 8 rows -> the form, us: class 80 (cards 1 to 4) wq_a+wkv 2,167-2,239 -> 486-525 (16 slices a block a
 * group: 1,330-1,347), wq_b 4,157-4,237 -> 2,463-2,838, the indexer 1,144-1,223 -> 675-713, wo_b 5,791-6,058 ->
 * 4,687-4,836, engram.wkv 18,458-20,087 -> 10,461-10,908, shared gate/up 2,863-3,092 -> 2,389-2,514, down
 * 1,820-2,010 -> 1,123-1,198, wo_a 5,641-5,863 -> 3,682-3,762 (4 slices: 5,061-5,238); class 120 wq_a+wkv 14,908 ->
 * 5,067 (8 slices: 5,221), wo_a 27,778 -> 19,009 (8: 21,207), wq_b 25,234 -> 28,376 (its thin launches are faster at
 * 1,024 rows, the wide at 100), the others 8 to 54 % below thin; class 120 native wo_a 1,533 -> 1,269 a block a tile
 * (a group: 1,419), the others 8 to 65 % below thin, a block a group. The draft's mtp.0.main_proj (26 October 2026,
 * dspark_parts_v1 K8, under the shared timing lock, production serving beside it): engine_linear_timing_probe on cold
 * weights, us at 1 / 8 rows, class 120 native 1 slice 119.9 / 120.8, 2: 68.8 / 70.1, 4: 56.1 / 56.5, 8: 54.4 / 54.7,
 * 16: 54.3 / 54.6; class 80 4 slices 78.8 / 93.2, 8: 85.0 / 103.4, 16: 78.8 / 101.4 (a second run the same order);
 * the trunk wide card test on class 80 at 1,024 rows: thin 11,756 -> 4,510 interleaved in one slice (4 slices a
 * group: 13,213). Its class 120 native wide form is not measured yet (the test's arena did not fit beside production
 * on the PRO): a block a group, the form every native shape of quantized rows measured fastest. */
enum { PROFILE_COUNT = LINEAR_PROFILE_SPEED + 1 };
struct slice_default {
	enum linear_family_code family;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t arithmetic_class;
	uint32_t slice_counts[PROFILE_COUNT];
	uint32_t wide_slice_count; /* of a wide trunk launch; 0: the profile's count */
	enum linear_wide_form wide_form;
};
#define SM_80 \
	ARITHMETIC_CLASS_SM_80 /* short names for the table, undefined behind it */
#define SM_120 ARITHMETIC_CLASS_SM_120
#define SM_120_NATIVE ARITHMETIC_CLASS_SM_120_NATIVE
#define EXPERT_GATE_UP LINEAR_FAMILY_EXPERT_GATE_UP
#define EXPERT_DOWN LINEAR_FAMILY_EXPERT_DOWN
#define TRUNK LINEAR_FAMILY_TRUNK_QUANTIZED
#define TRUNK_BFLOAT16 LINEAR_FAMILY_TRUNK_BFLOAT16
#define GROUPS LINEAR_WIDE_GROUP_BLOCKS
#define TILES LINEAR_WIDE_TILE_BLOCKS
#define INTERLEAVED_CHUNKS LINEAR_INTERLEAVED_CHUNK_BLOCKS
#define INTERLEAVED_TILES LINEAR_INTERLEAVED_TILE_BLOCKS
static const struct slice_default slice_defaults[] = {
    {EXPERT_GATE_UP, 2304, 5120, SM_80, {1, 1}, 0, GROUPS},  {EXPERT_GATE_UP, 2304, 5120, SM_120, {2, 2}, 0, GROUPS},
    {EXPERT_DOWN, 5120, 2304, SM_80, {1, 1}, 0, GROUPS},     {EXPERT_DOWN, 5120, 2304, SM_120, {4, 4}, 0, GROUPS},
    {TRUNK, 1280, 5120, SM_80, {16, 16}, 1, INTERLEAVED_CHUNKS},  /* attn.wq_a */
    {TRUNK, 512, 5120, SM_80, {16, 16}, 1, INTERLEAVED_CHUNKS},   /* attn.wkv */
    {TRUNK, 32768, 1280, SM_80, {1, 1}, 0, INTERLEAVED_TILES},    /* attn.wq_b */
    {TRUNK, 4096, 1280, SM_80, {8, 8}, 0, GROUPS},                /* attn.indexer.wq_b */
    {TRUNK, 5120, 8192, SM_80, {4, 4}, 0, GROUPS},                /* attn.wo_b */
    {TRUNK, 25600, 6144, SM_80, {1, 1}, 0, INTERLEAVED_CHUNKS},   /* engram.wkv */
    {TRUNK, 2304, 5120, SM_80, {8, 8}, 0, GROUPS},                /* shared gate, up */
    {TRUNK, 5120, 2304, SM_80, {4, 4}, 0, GROUPS},                /* shared down */
    {TRUNK, 5120, 15360, SM_80, {4, 4}, 1, INTERLEAVED_CHUNKS},   /* mtp.0.main_proj, the draft's */
    {TRUNK_BFLOAT16, 8192, 4096, SM_80, {4, 4}, 1, GROUPS},       /* attn.wo_a */
    {TRUNK, 1280, 5120, SM_120, {16, 16}, 4, GROUPS},             {TRUNK, 512, 5120, SM_120, {16, 16}, 4, GROUPS},
    {TRUNK, 32768, 1280, SM_120, {4, 4}, 0, GROUPS},              {TRUNK, 4096, 1280, SM_120, {4, 4}, 0, GROUPS},
    {TRUNK, 5120, 8192, SM_120, {16, 16}, 0, GROUPS},             {TRUNK, 25600, 6144, SM_120, {4, 4}, 0, GROUPS},
    {TRUNK, 2304, 5120, SM_120, {16, 16}, 0, GROUPS},             {TRUNK, 5120, 2304, SM_120, {4, 4}, 0, GROUPS},
    {TRUNK_BFLOAT16, 8192, 4096, SM_120, {8, 8}, 1, GROUPS},
    /* class 120 native, the PRO in its own forms (the default), from the native sweep of 25 October 2026, 21:00
     * (tmp_trunk/native_sweep.sh, every check passed), us at 1 / 8 rows: experts 79.7 / 562.8 (8 x 4), wq_a+wkv
     * 10.7 / 10.5, wq_b 31.2 / 31.2, wo_b 31.2 / 31.1, shared expert 29.0 / 29.8 (16 x 4), wo_a 24.0 / 24.0,
     * indexer 8.6 / 8.4, engram.wkv 101.3 / 103.1; a layer's linears about 207 us at a row against 453 emulated */
    {EXPERT_GATE_UP, 2304, 5120, SM_120_NATIVE, {8, 8}, 0, GROUPS},
    {EXPERT_DOWN, 5120, 2304, SM_120_NATIVE, {4, 4}, 0, GROUPS},
    {TRUNK, 1280, 5120, SM_120_NATIVE, {16, 16}, 0, GROUPS}, {TRUNK, 512, 5120, SM_120_NATIVE, {16, 16}, 0, GROUPS},
    {TRUNK, 32768, 1280, SM_120_NATIVE, {2, 2}, 0, GROUPS},  {TRUNK, 4096, 1280, SM_120_NATIVE, {8, 8}, 0, GROUPS},
    {TRUNK, 5120, 8192, SM_120_NATIVE, {8, 8}, 0, GROUPS},   {TRUNK, 25600, 6144, SM_120_NATIVE, {16, 16}, 0, GROUPS},
    {TRUNK, 2304, 5120, SM_120_NATIVE, {16, 16}, 0, GROUPS}, {TRUNK, 5120, 2304, SM_120_NATIVE, {4, 4}, 0, GROUPS},
    {TRUNK, 5120, 15360, SM_120_NATIVE, {16, 16}, 0, GROUPS},     /* mtp.0.main_proj, the draft's */
    {TRUNK_BFLOAT16, 8192, 4096, SM_120_NATIVE, {8, 8}, 0, TILES},
};
#undef SM_80
#undef SM_120
#undef SM_120_NATIVE
#undef EXPERT_GATE_UP
#undef EXPERT_DOWN
#undef TRUNK
#undef TRUNK_BFLOAT16
#undef GROUPS
#undef TILES
#undef INTERLEAVED_CHUNKS
#undef INTERLEAVED_TILES

_Static_assert(COUNT_OF(slice_instances) == INSTANCES, "ANEMONE_LINEAR_SLICE_INSTANCES counts the stamped counts");
_Static_assert(
	sizeof(struct linear_kernels_record) ==
			KERNEL_COUNT * sizeof(uint32_t) &&
		KERNEL_COUNT <= BACKEND_MAXIMUM_KERNELS &&
		KERNEL_COUNT <= sizeof(uint64_t) * CHAR_BIT,
	"the handles are the functions in their order, as many as a card can prepare, a bit each in 64");
_Static_assert(
	EXPERT_GATE_UP_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES &&
		TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES,
	"the two largest records of linear_records.h fit the argument of a launch");

/* The functions of a family: 1, or one per stamped count; and the name of one of them. */
static uint32_t instance_count_of(uint32_t family)
{
    return kernel_families[family].stamped_column == NOT_STAMPED ? 1 : INSTANCES;
}
static const char *function_name_of(uint32_t family, uint32_t instance)
{
    const struct kernel_family *const of = &kernel_families[family];
    return of->stamped_column == NOT_STAMPED ? of->function_name
                                             : slice_instances[instance].function_names[of->stamped_column];
}

/* The product, or UINT64_MAX when it is beyond 64 bits. */
static uint64_t product_or_most(uint64_t first, uint64_t second)
{
    return first != 0 && second > UINT64_MAX / first ? UINT64_MAX : first * second;
}

/* A row's count in a profile and order (the table's comment); of a wide trunk launch, its own where it has one. */
static uint32_t slices_of_row(const struct slice_default *known,
			      enum linear_profile_code profile,
			      enum linear_order_code order)
{
#if defined(LINEAR_CLASS_FORMS)
	return order == LINEAR_ORDER_REFERENCE ? 1 :
						 known->slice_counts[profile];
#else   /* ordered slices keep the reference order at any count; class 120 fits its emulation; 120 native its own */
    (void)order;
    const uint32_t slices = known->slice_counts[profile];
    const uint32_t most = known->family == LINEAR_FAMILY_EXPERT_GATE_UP ?
				  LINEAR_CLASS_120_EXACT_MOST_GATE_UP_SLICES :
				  LINEAR_CLASS_120_EXACT_MOST_SLICES;
    return known->arithmetic_class == ARITHMETIC_CLASS_SM_120 && slices > most ?
		   most :
		   slices;
#endif
}
static uint32_t wide_slices_of_row(const struct slice_default *known,
				   enum linear_profile_code profile,
				   enum linear_order_code order)
{
    return known->wide_slice_count != 0 ? known->wide_slice_count : slices_of_row(known, profile, order);
}

/* The row of a matrix of family and shape on a card of arithmetic_class; NULL, behind the error line, for none. */
static const struct slice_default *
slice_default_of(enum linear_family_code family,
		 const struct matrix_shape_record *shape,
		 uint32_t arithmetic_class)
{
    for (uint32_t row = 0; row < COUNT_OF(slice_defaults); row++) {
        const struct slice_default *const known = &slice_defaults[row];
        if (known->family == family && known->output_size == shape->output_size &&
            known->input_size == shape->input_size && known->arithmetic_class == arithmetic_class) {
            return known;
        }
    }
    LOG_ERROR("the default slice count: no row for family %" PRIu32 ", %" PRIu32
	      " by %" PRIu32 " on cards of class %" PRIu32
	      ": a shape is measured before it is served",
	      (uint32_t)family, shape->output_size, shape->input_size,
	      arithmetic_class);
    return NULL;
}

result_status linear_default_slice_count(
	enum linear_profile_code profile, enum linear_order_code order,
	enum linear_family_code family, const struct matrix_shape_record *shape,
	uint32_t arithmetic_class, uint32_t *slice_count_out)
{
	if (shape == NULL || slice_count_out == NULL ||
	    (uint32_t)profile >= PROFILE_COUNT ||
	    (uint32_t)order > LINEAR_ORDER_REFERENCE ||
	    (uint32_t)family > LINEAR_FAMILY_TRUNK_BFLOAT16) {
		LOG_ERROR(
			"the default slice count: a missing argument, or a profile, order or family not known");
		return STATUS_INVALID_ARGUMENT;
	}
    const struct slice_default *const known = slice_default_of(family, shape, arithmetic_class);
    if (known == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    *slice_count_out = slices_of_row(known, profile, order);
    return STATUS_OK;
}

/* The selector changes only a product launch. Validation through the existing prepare also preserves its checks
 * for codes/exponents, reserved fields and all shared-expert scratch, even where the new kernel reads less. */
static bool cmp_selection_is_valid(uint32_t arithmetic_class,
				   enum cmp_order_code cmp_order)
{
	if ((uint32_t)cmp_order > CMP_ORDER_RELAXED ||
	    (arithmetic_class != ARITHMETIC_CLASS_SM_80 &&
	     arithmetic_class != ARITHMETIC_CLASS_SM_120 &&
	     arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE)) {
		LOG_ERROR(
			"the CMP trunk selection: an unknown order or arithmetic class");
		return false;
	}
    return true;
}

/* The relaxed decode count is fixed by shape (26 October trunk card report), so rows 1 through 8 have one order.
 * Most match the class-80 thin count. The indexer and engram take four slices instead of eight and one. */
#define RELAXED_INDEXER_OUTPUTS 4096u
#define RELAXED_ENGRAM_OUTPUTS 25600u
#define RELAXED_FOUR_SLICES 4u
MUST_CHECK static result_status
relaxed_decode_slices(enum linear_family_code family,
		      const struct matrix_shape_record *shape,
		      uint32_t *slices_out)
{
    uint32_t slices = 0;
    const result_status status = linear_default_slice_count(
	    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED, family, shape,
	    ARITHMETIC_CLASS_SM_80, &slices);
    if (status == STATUS_OK) {
	    *slices_out = family == LINEAR_FAMILY_TRUNK_QUANTIZED &&
					  (shape->output_size ==
						   RELAXED_INDEXER_OUTPUTS ||
					   shape->output_size ==
						   RELAXED_ENGRAM_OUTPUTS) ?
				  RELAXED_FOUR_SLICES :
				  slices;
    }
    return status;
}

result_status linear_trunk_quantized_select_prepare(
	const struct linear_trunk_quantized_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	const struct relaxed_linear_kernels *relaxed_kernels,
	uint32_t arithmetic_class, enum cmp_order_code cmp_order,
	const int32_t *fold_references, struct kernel_launch *launch_out)
{
	struct kernel_launch launches[RELAXED_MOST_LAUNCHES] = { 0 };
	if (!cmp_selection_is_valid(arithmetic_class, cmp_order) ||
	    !prepare_arguments_are_given("the CMP trunk selection", description,
					 kernels, launch_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = linear_trunk_quantized_prepare(
		description, arena_byte_count, kernels, &launches[0]);
	if (status != STATUS_OK) {
		return status;
	}
	const bool check_folds = arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
				 cmp_order == CMP_ORDER_RELAXED &&
				 fold_references != NULL;
	bool is_relaxed = check_folds;
	for (uint32_t matrix = 0;
	     matrix < description->matrix_count && check_folds; matrix++) {
		if (fold_references[matrix] != 0 &&
		    fold_references[matrix] != RELAXED_FOLD_REFERENCE) {
			LOG_ERROR(
				"the CMP trunk selection: a quantized matrix has an invalid fold reference");
			return STATUS_INVALID_ARGUMENT;
		}
		is_relaxed = is_relaxed && fold_references[matrix] != 0;
	}
	if (arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
	    (description->row_count > LINEAR_MAXIMUM_ROWS || is_relaxed)) {
		struct relaxed_trunk_quantized_description selected = {
			.operands = description->operands,
			.matrix_count = description->matrix_count,
			.row_count = description->row_count,
			.k_split_count = 1,
			.is_ordered = is_relaxed ? 0u : 1u,
		};
		memcpy(selected.matrices, description->matrices,
		       sizeof(selected.matrices));
		memcpy(selected.outputs, description->outputs,
		       sizeof(selected.outputs));
		for (uint32_t matrix = 0;
		     matrix < description->matrix_count && is_relaxed;
		     matrix++) {
			if (description->row_count <= LINEAR_MAXIMUM_ROWS) {
				status = relaxed_decode_slices(
					LINEAR_FAMILY_TRUNK_QUANTIZED,
					&selected.matrices[matrix].shape,
					&selected.matrices[matrix].slice_count);
				if (status != STATUS_OK) {
					return status;
				}
			}
		}
		uint32_t launch_count = 0;
		status = relaxed_trunk_quantized_prepare(
			&selected, arena_byte_count, relaxed_kernels, launches,
			&launch_count);
		if (status != STATUS_OK) {
			return status;
		}
		ASSERT(launch_count == 1u);
	}
    *launch_out = launches[0];
    return STATUS_OK;
}

result_status linear_trunk_bfloat16_select_prepare(
	const struct linear_trunk_bfloat16_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	const struct relaxed_linear_kernels *relaxed_kernels,
	uint32_t arithmetic_class, enum cmp_order_code cmp_order,
	int32_t fold_reference, struct kernel_launch *launch_out)
{
	struct kernel_launch launches[RELAXED_MOST_LAUNCHES] = { 0 };
	if (!cmp_selection_is_valid(arithmetic_class, cmp_order) ||
	    !prepare_arguments_are_given("the CMP trunk selection", description,
					 kernels, launch_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = linear_trunk_bfloat16_prepare(
		description, arena_byte_count, kernels, &launches[0]);
	if (status != STATUS_OK) {
		return status;
	}
	const bool is_relaxed = cmp_order == CMP_ORDER_RELAXED &&
				fold_reference != 0;
	if (arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
	    (description->row_count > LINEAR_MAXIMUM_ROWS || is_relaxed)) {
		struct relaxed_trunk_bfloat16_description selected = {
			.matrix = description->matrix,
			.values = description->values,
			.output = description->output,
			.row_count = description->row_count,
			.segment_count = description->segment_count,
			.k_split_count = 1,
			.fold_reference = is_relaxed ? fold_reference :
						       RELAXED_FOLD_REFERENCE,
			.is_ordered = is_relaxed ? 0u : 1u,
		};
		if (is_relaxed &&
		    description->row_count <= LINEAR_MAXIMUM_ROWS) {
			status = relaxed_decode_slices(
				LINEAR_FAMILY_TRUNK_BFLOAT16,
				&selected.matrix.shape,
				&selected.matrix.slice_count);
			if (status != STATUS_OK) {
				return status;
			}
		}
		uint32_t launch_count = 0;
		status = relaxed_trunk_bfloat16_prepare(
			&selected, arena_byte_count, relaxed_kernels, launches,
			&launch_count);
		if (status != STATUS_OK) {
			return status;
		}
		ASSERT(launch_count == 1u);
	}
    *launch_out = launches[0];
    return STATUS_OK;
}

result_status linear_shared_expert_select_prepare(
	const struct linear_shared_expert_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	const struct relaxed_linear_kernels *relaxed_kernels,
	uint32_t arithmetic_class, enum cmp_order_code cmp_order,
	const int32_t *fold_references, struct kernel_launch *launches_out)
{
	struct kernel_launch launches[LINEAR_SHARED_EXPERT_LAUNCH_COUNT] = { 0 };
	if (!cmp_selection_is_valid(arithmetic_class, cmp_order) ||
	    !prepare_arguments_are_given("the CMP trunk selection", description,
					 kernels, launches_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = linear_shared_expert_prepare(
		description, arena_byte_count, kernels, launches);
	if (status != STATUS_OK) {
		return status;
	}
	const struct linear_trunk_quantized_description gate_up = {
		.matrices = { description->gate, description->up },
		.codes = description->codes,
		.exponents = description->exponents,
		.operands = description->operands,
		.outputs = { description->gate_output, description->up_output },
		.matrix_count = LINEAR_MERGED_MATRICES,
		.row_count = description->row_count,
	};
	const struct linear_trunk_quantized_description down = {
		.matrices = { description->down },
		.codes = description->hidden_codes,
		.exponents = description->hidden_exponents,
		.operands = description->hidden_operands,
		.outputs = { description->output },
		.matrix_count = 1,
		.row_count = description->row_count,
	};
	status = linear_trunk_quantized_select_prepare(
		&gate_up, arena_byte_count, kernels, relaxed_kernels,
		arithmetic_class, cmp_order, fold_references,
		&launches[LINEAR_SHARED_EXPERT_GATE_UP]);
	if (status != STATUS_OK) {
		return status;
	}
	status = linear_trunk_quantized_select_prepare(
		&down, arena_byte_count, kernels, relaxed_kernels,
		arithmetic_class, cmp_order,
		fold_references == NULL ?
			NULL :
			&fold_references[LINEAR_MERGED_MATRICES],
		&launches[LINEAR_SHARED_EXPERT_DOWN]);
	if (status == STATUS_OK) {
		memcpy(launches_out, launches, sizeof(launches));
	}
    return status;
}

result_status linear_default_wide_form(enum linear_profile_code profile,
				       enum linear_order_code order,
				       enum linear_family_code family,
				       const struct matrix_shape_record *shape,
				       uint32_t arithmetic_class,
				       uint32_t *slice_count_out,
				       enum linear_wide_form *form_out)
{
    uint32_t slice_count = 0;
    if (slice_count_out == NULL || form_out == NULL ||
	(family != LINEAR_FAMILY_TRUNK_QUANTIZED &&
	 family != LINEAR_FAMILY_TRUNK_BFLOAT16)) {
	    LOG_ERROR(
		    "the wide form: a missing argument, or a family other than the trunk's");
	    return STATUS_INVALID_ARGUMENT;
    }
    const result_status status = linear_default_slice_count(
	    profile, order, family, shape, arithmetic_class, &slice_count);
    const struct slice_default *const known =
	    status == STATUS_OK ?
		    slice_default_of(family, shape, arithmetic_class) :
		    NULL;
    if (known != NULL) {                        /* found again: the count's own row */
        *slice_count_out = wide_slices_of_row(known, profile, order);
        *form_out = known->wide_form;
    }
    return status;
}

uint64_t linear_values_byte_count(uint64_t vector_count, uint64_t width)
{
	return product_or_most(product_or_most(vector_count, width),
			       LINEAR_BFLOAT16_BYTES);
}

uint64_t linear_codes_byte_count(uint64_t vector_count, uint64_t width)
{
    return product_or_most(vector_count, width);
}

uint64_t linear_exponents_byte_count(uint64_t vector_count, uint64_t width)
{
	return product_or_most(vector_count, width / LINEAR_VALUES_PER_BLOCK);
}

uint64_t linear_sums_byte_count(uint64_t row_count, uint64_t width)
{
	return product_or_most(product_or_most(row_count, width),
			       LINEAR_SUM_BYTES);
}

uint64_t linear_jobs_byte_count(uint64_t job_count)
{
	return product_or_most(job_count, EXPERT_JOB_BYTES);
}

/* The place of the first handle of a family among the members of struct linear_kernels_record. */
static uint32_t first_kernel_of(uint32_t family)
{
    uint32_t kernel_index = 0;
    for (uint32_t earlier = 0; earlier < family; earlier++) {
        kernel_index += instance_count_of(earlier);
    }
    return kernel_index;
}

/* The place of a count in slice_instances: INSTANCES for a count no kernel is stamped for. */
static uint32_t instance_of(uint32_t slice_count)
{
    uint32_t instance = 0;
    while (instance < INSTANCES && slice_instances[instance].slice_count != slice_count) {
        instance++;
    }
    return instance;
}

/* The family of the kernel of a trunk launch of row_count rows of thin_family's rows whose matrix has wide_form: the
 * thin one up to LINEAR_MAXIMUM_ROWS, above it the wide one, or the interleaved one of an interleaved form. */
static enum linear_family trunk_family_of(enum linear_family thin_family,
					  uint32_t row_count,
					  enum linear_wide_form wide_form)
{
    const bool is_quantized = thin_family == FAMILY_TRUNK_LINEAR_QUANTIZED;
    return row_count <= LINEAR_MAXIMUM_ROWS ?
		   thin_family :
	   wide_form >= LINEAR_INTERLEAVED_CHUNK_BLOCKS ?
		   FAMILY_TRUNK_LINEAR_QUANTIZED_INTERLEAVED :
	   is_quantized ? FAMILY_TRUNK_LINEAR_QUANTIZED_WIDE :
			  FAMILY_TRUNK_LINEAR_BFLOAT16_WIDE;
}

/* The place of a kernel among the members of struct linear_kernels_record: of a family, at the instance of a slice
 * count where it is stamped per count. */
static uint32_t kernel_place_of(enum linear_family family, uint32_t instance)
{
    return first_kernel_of(family) + (kernel_families[family].stamped_column == NOT_STAMPED ? 0 : instance);
}

/* The wide trunk kernels the matrices of a class take in a profile and order (linear_default_wide_form): a bit
 * for each, at its place, as kernels_ask takes them. */
static uint64_t wide_kernels_of(enum linear_profile_code profile,
				enum linear_order_code order,
				uint32_t arithmetic_class)
{
    uint64_t kernels = 0;
    for (uint32_t row = 0; row < COUNT_OF(slice_defaults); row++) {
        const struct slice_default *const known = &slice_defaults[row];
	const bool is_quantized = known->family ==
				  LINEAR_FAMILY_TRUNK_QUANTIZED;
	if (known->arithmetic_class == arithmetic_class &&
	    (is_quantized || known->family == LINEAR_FAMILY_TRUNK_BFLOAT16)) {
		const uint32_t instance =
			instance_of(wide_slices_of_row(known, profile, order));
		ASSERT(instance < INSTANCES);
		kernels |=
			1ull << kernel_place_of(
				trunk_family_of(
					is_quantized ?
						FAMILY_TRUNK_LINEAR_QUANTIZED :
						FAMILY_TRUNK_LINEAR_BFLOAT16,
					LINEAR_WIDE_MOST_ROWS,
					known->wide_form),
				instance);
	}
    }
    return kernels;
}

/* Asks a card for the kernels whose bits kernels_asked holds (bit k: the member k of struct linear_kernels_record)
 * and, once all are prepared, writes the table kernels_out: those, and every other member as from has it
 * (BACKEND_NO_KERNEL where from is NULL). The first that fails ends it behind its error line, kernels_out
 * untouched. */
MUST_CHECK static result_status
kernels_ask(const char *what, const struct backend_operations *backend,
	    struct device_card *card, uint32_t arithmetic_class,
	    uint64_t kernels_asked, const struct linear_kernels_record *from,
	    struct linear_kernels_record *kernels_out)
{
	const bool is_native = arithmetic_class ==
			       ARITHMETIC_CLASS_SM_120_NATIVE;
	uint32_t handles[KERNEL_COUNT];
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR("%s: a missing argument", what);
		return STATUS_INVALID_ARGUMENT;
	}
    for (uint32_t kernel_index = 0; kernel_index < KERNEL_COUNT; kernel_index++) {
	    handles[kernel_index] = BACKEND_NO_KERNEL;
    }
    if (from != NULL) {
        memcpy(handles, from, sizeof(handles));
    }
    for (uint32_t family_index = 0; family_index < FAMILY_COUNT; family_index++) {
        const struct kernel_family *const family = &kernel_families[family_index];
        const char *const file_name = is_native ? family->native_file_name : family->kernel_file_name;
        for (uint32_t instance = 0; instance < instance_count_of(family_index); instance++) {
            const uint32_t kernel_index = first_kernel_of(family_index) + instance;
            if ((kernels_asked >> kernel_index & 1u) == 0) {
                continue;
            }
            const char *const function_name = function_name_of(family_index, instance);
            /* a backend that answers OK and gives no handle leaves this one, which every prepare function refuses */
	    handles[kernel_index] = BACKEND_NO_KERNEL;
	    const result_status status = backend->kernel_prepare(
		    card, file_name, function_name, family->argument_byte_count,
		    &handles[kernel_index]);
	    if (status != STATUS_OK) {
		    LOG_ERROR("%s: %s of %s is not prepared: %s", what,
			      function_name, file_name, status_text(status));
		    return status;
	    }
	}
    }
    ASSERT(first_kernel_of(FAMILY_COUNT) == KERNEL_COUNT);
    memcpy(kernels_out, handles, sizeof(*kernels_out));
    return STATUS_OK;
}

result_status linear_kernels_prepare(const struct backend_operations *backend,
				     struct device_card *card,
				     uint32_t arithmetic_class,
				     struct linear_kernels_record *kernels_out)
{
    return kernels_ask("the kernels of the linear", backend, card, arithmetic_class,
                       (1ull << first_kernel_of(FAMILY_WIDE_FIRST)) - 1u, NULL, kernels_out);
}

result_status linear_wide_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	enum linear_profile_code profile, enum linear_order_code order,
	uint32_t arithmetic_class, struct linear_kernels_record *kernels)
{
	if ((uint32_t)profile >= PROFILE_COUNT ||
	    (uint32_t)order > LINEAR_ORDER_REFERENCE) {
		LOG_ERROR(
			"the wide kernels of the linear: a profile or order not known");
		return STATUS_INVALID_ARGUMENT;
	}
    return kernels_ask("the wide kernels of the linear", backend, card, arithmetic_class,
                       wide_kernels_of(profile, order, arithmetic_class), kernels, kernels);
}

result_status linear_wide_reference_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct linear_kernels_record *kernels)
{
    const uint64_t below = (1ull << first_kernel_of(FAMILY_WIDE_FIRST)) - 1u;
    const uint64_t through = (1ull << first_kernel_of(FAMILY_TRUNK_LINEAR_QUANTIZED_INTERLEAVED)) - 1u;
    return kernels_ask("the explicit reference-wide linear kernels", backend, card, arithmetic_class,
        through & ~below, kernels, kernels);
}

/* ---- the checks of the linear alone: each is true, or false behind the one error line of the refusal ---- */

/* The handle of a family's kernel; instance: of a family stamped per slice count, the place of the count in
 * slice_instances, which a family of one kernel ignores. */
static uint32_t handle_of(const struct linear_kernels_record *kernels,
			  enum linear_family family, uint32_t instance)
{
    uint32_t handles[KERNEL_COUNT];
    memcpy(handles, kernels, sizeof(handles));
    return handles[kernel_place_of(family, instance)];
}
static bool kernel_is_prepared(const char *what,
			       const struct linear_kernels_record *kernels,
			       enum linear_family family, uint32_t instance)
{
	return prepare_kernel_is_prepared(what,
					  handle_of(kernels, family, instance),
					  function_name_of(family, instance));
}

/* The wide form of a matrix in a trunk launch of row_count rows (of quantized rows, or of bfloat16 rows): one of
 * linear.h, and above LINEAR_MAXIMUM_ROWS one the launch takes: an interleaved one of quantized rows in one
 * slice alone. */
static bool wide_form_is_valid(const char *what,
			       const struct linear_matrix *matrix,
			       bool is_quantized, uint32_t row_count)
{
	const bool is_interleaved = matrix->wide_form >=
				    LINEAR_INTERLEAVED_CHUNK_BLOCKS;
	if ((uint32_t)matrix->wide_form >= LINEAR_WIDE_FORM_COUNT ||
	    (row_count > LINEAR_MAXIMUM_ROWS && is_interleaved &&
	     (!is_quantized || matrix->slice_count != 1u))) {
		LOG_ERROR(
			"%s: a matrix of wide form %" PRIu32 " in %" PRIu32
			" slices: the forms are linear.h's, and "
			"an interleaved one takes quantized rows in one slice",
			what, (uint32_t)matrix->wide_form, matrix->slice_count);
		return false;
	}
    return true;
}

static bool width_is_valid(const char *what, const char *name, uint32_t width)
{
	if (width == 0 || width % LINEAR_VALUES_PER_LOAD != 0) {
		LOG_ERROR(
			"%s: %s is %" PRIu32
			" values wide: a width is whole loads of %u values, at least one",
			what, name, width, LINEAR_VALUES_PER_LOAD);
		return false;
	}
    return true;
}

/* The rows of a trunk launch (and of the shared expert's chain of them): its wide instance above
 * LINEAR_MAXIMUM_ROWS. */
static bool trunk_rows_are_valid(const char *what, uint32_t row_count)
{
	if (row_count == 0 || row_count > LINEAR_WIDE_MOST_ROWS) {
		LOG_ERROR("%s: %" PRIu32 " rows: a trunk launch takes 1 to %u",
			  what, row_count, LINEAR_WIDE_MOST_ROWS);
		return false;
	}
    return true;
}

/* The rows of the routed experts and of a quantization: a wide step's chunk too (linear.h). */
static bool chunk_rows_are_valid(const char *what, const char *who, uint32_t row_count)
{
	if (row_count == 0 || row_count > MODEL_WIDE_MOST_ROWS) {
		LOG_ERROR("%s: %" PRIu32 " rows: %s take 1 to %u", what,
			  row_count, who, MODEL_WIDE_MOST_ROWS);
		return false;
	}
    return true;
}
static bool routed_rows_are_valid(const char *what, uint32_t row_count)
{
    return chunk_rows_are_valid(what, "the routed experts", row_count);
}
static bool quantized_rows_are_valid(const char *what, uint32_t row_count)
{
    return chunk_rows_are_valid(what, "quantizations", row_count);
}

static bool limit_is_valid(const char *what, float activation_limit)
{
    if (!(activation_limit > 0.0f)) {           /* written so that a not-a-number is refused */
	    LOG_ERROR(
		    "%s: an activation limit of %g is not above zero (no limit is +infinity)",
		    what, (double)activation_limit);
	    return false;
    }
    return true;
}

/* A matrix a product kernel reads: its format is the kernel's and stores its shape, its inputs are whole loads and
 * its slices are as many as linear_records.h allows, a count a kernel is stamped for. Gives the bytes the matrix
 * occupies and the instance of its slice count. */
static bool matrix_is_read(const char *what, const char *name,
			   enum weight_format_code weight_format,
			   enum weight_format_code format_of_kernel,
			   const struct matrix_shape_record *shape,
			   uint32_t slice_count,
			   uint64_t *stored_byte_count_out,
			   uint32_t *instance_out)
{
	const struct weight_format_operations *const operations =
		weight_format_operations_of(weight_format);
	const uint32_t load_count = shape->input_size / LINEAR_VALUES_PER_LOAD;
	uint64_t stored_byte_count = 0;
	if (operations == NULL || weight_format != format_of_kernel) {
		LOG_ERROR("%s: %s is of weight format %" PRIu32
			  ": its kernel reads format %" PRIu32 " only",
			  what, name, (uint32_t)weight_format,
			  (uint32_t)format_of_kernel);
		return false;
	}
    if (shape->reserved_must_be_zero != 0) {
	    LOG_ERROR("%s: a reserved field of the shape of %s is not zero",
		      what, name);
	    return false;
    }
    if (operations->stored_byte_count(shape, &stored_byte_count) != STATUS_OK) {
	    LOG_ERROR("%s: the format of %s stores no matrix of %" PRIu32
		      " outputs by %" PRIu32 " inputs at %" PRIu32
		      " bits per weight",
		      what, name, shape->output_size, shape->input_size,
		      shape->bits_per_weight);
	    return false;
    }
    if (shape->input_size % LINEAR_VALUES_PER_LOAD != 0) {
	    LOG_ERROR("%s: %s has %" PRIu32
		      " inputs: a product reads whole loads of %u inputs",
		      what, name, shape->input_size, LINEAR_VALUES_PER_LOAD);
	    return false;
    }
    const uint32_t instance = instance_of(slice_count);
    if (instance == INSTANCES || slice_count > load_count) {
	    LOG_ERROR(
		    "%s: %s is cut into %" PRIu32
		    " slices: a product kernel is stamped for%s slices, and its %" PRIu32
		    " loads take no more",
		    what, name, slice_count, stamped_counts_text, load_count);
	    return false;
    }
    /* what a format stores is whole tiles of outputs: the grid of a product counts them */
    ASSERT_TEXT(
	    shape->output_size != 0 &&
		    shape->output_size % LINEAR_OUTPUTS_PER_TILE == 0,
	    "shape->output_size != 0 && shape->output_size % ANEMONE_LINEAR_OUTPUTS_PER_TILE == 0");
    *stored_byte_count_out = stored_byte_count;
    *instance_out = instance;
    return true;
}

static bool trunk_matrix_is_read(const char *what, const char *name,
				 const struct linear_matrix *matrix,
				 uint64_t *stored_byte_count_out,
				 uint32_t *instance_out)
{
	return prepare_reserved_is_zero(what, matrix->reserved_zero) &&
	       matrix_is_read(what, name, matrix->weight_format,
			      WEIGHT_FORMAT_E4M3_SCALED, &matrix->shape,
			      matrix->slice_count, stored_byte_count_out,
			      instance_out);
}

/* The matrices of one trunk launch: one, or two that read the same rows in the same slices and wide form. Gives the
 * bytes of each and the instance of their slice count. */
static bool trunk_matrices_are_read(const char *what,
				    const struct linear_matrix *matrices,
				    uint32_t matrix_count,
				    uint64_t *stored_byte_counts_out,
				    uint32_t *instance_out)
{
	static const char *const names[LINEAR_MERGED_MATRICES] = {
		"the first matrix", "the second matrix"
	};
	if (matrix_count == 0 || matrix_count > LINEAR_MERGED_MATRICES) {
		LOG_ERROR("%s: %" PRIu32 " matrices: a launch reads 1 or %u",
			  what, matrix_count, LINEAR_MERGED_MATRICES);
		return false;
	}
    for (uint32_t matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
        if (!trunk_matrix_is_read(what, names[matrix_index], &matrices[matrix_index],
                                  &stored_byte_counts_out[matrix_index], instance_out)) {
            return false;
        }
    }
    if (matrix_count == LINEAR_MERGED_MATRICES &&
	(matrices[1].shape.input_size != matrices[0].shape.input_size ||
	 matrices[1].slice_count != matrices[0].slice_count ||
	 matrices[1].weight_format != matrices[0].weight_format ||
	 matrices[1].wide_form != matrices[0].wide_form ||
	 (uint64_t)matrices[0].shape.output_size +
			 matrices[1].shape.output_size >
		 UINT32_MAX)) {
	    LOG_ERROR(
		    "%s: the two matrices of one launch differ in input_size, slice_count, weight format or wide "
		    "form, or have more outputs than 32 bits count",
		    what);
	    return false;
    }
    return true;
}

/* The hidden vectors of an expert are what its down matrix reads. */
static bool down_reads_hidden(const char *what,
			      const struct matrix_shape_record *gate_shape,
			      const struct matrix_shape_record *down_shape)
{
    if (down_shape->input_size != gate_shape->output_size) {
	    LOG_ERROR("%s: down reads %" PRIu32
		      " inputs: gate and up give %" PRIu32 " outputs",
		      what, down_shape->input_size, gate_shape->output_size);
	    return false;
    }
    return true;
}

/* ---- the launches: each is built from a description every check has passed ---- */

/* A launch of a product: tile_blocks blocks (1, or a wide trunk launch's: trunk_tile_blocks) for each tile of 16
 * outputs of each of grid_rows jobs (1 for a trunk launch), of warp_count warps: the slices of its matrix, or of its
 * two. Each thread has its bytes of shared memory. */
static void product_launch_fill(struct kernel_launch *launch, uint32_t kernel,
				uint32_t output_size, uint32_t tile_blocks,
				uint32_t grid_rows, uint32_t warp_count,
				const void *record, uint32_t record_byte_count)
{
	const uint32_t block_threads = warp_count * LINEAR_THREADS_PER_WARP;
	prepare_launch_fill(launch, kernel,
			    output_size / LINEAR_OUTPUTS_PER_TILE * tile_blocks,
			    grid_rows, block_threads,
			    block_threads * LINEAR_SHARED_BYTES_PER_THREAD,
			    record, record_byte_count);
}

/* The blocks a tile of a trunk launch of row_count rows takes (linear_records.h): one; in a wide form one for each
 * unit of its rows (a group of 8, of the interleaved instance a chunk of LINEAR_INTERLEAVED_ROWS) or one for
 * all. A tile's units then lie in adjacent blocks, which a card starts together (it hands blocks out in the order of
 * their index), so its weights come from memory about once whatever its L2 holds, and every multiprocessor has
 * blocks; one block taking every unit of its tile reads it again for each, which a few shapes measured faster (the
 * table of linear_default_wide_form). Fewer only where the grid would pass 32 bits: any count from 1 to the
 * units gives the same bits. */
static uint32_t trunk_tile_blocks(uint32_t tile_count, uint32_t row_count,
				  enum linear_wide_form wide_form)
{
	const bool is_interleaved = wide_form >=
				    LINEAR_INTERLEAVED_CHUNK_BLOCKS;
	const uint32_t unit_rows = is_interleaved ? LINEAR_INTERLEAVED_ROWS :
						    LINEAR_MAXIMUM_ROWS;
	const uint32_t unit_count = (row_count + unit_rows - 1u) / unit_rows;
	const uint32_t most_blocks =
		UINT32_MAX / tile_count; /* a matrix has a tile at least */
	const bool takes_units = wide_form == LINEAR_WIDE_GROUP_BLOCKS ||
				 wide_form == LINEAR_INTERLEAVED_CHUNK_BLOCKS;
	return row_count <= LINEAR_MAXIMUM_ROWS || !takes_units ? 1u :
	       unit_count < most_blocks				? unit_count :
								  most_blocks;
}

/* A launch in which a thread takes one value: blocks of LINEAR_ELEMENT_THREADS threads that cover the
 * width, for each of grid_rows vectors. */
static void element_launch_fill(struct kernel_launch *launch, uint32_t kernel,
				uint32_t width, uint32_t grid_rows,
				const void *record, uint32_t record_byte_count)
{
	const uint32_t block_count =
		width / LINEAR_ELEMENT_THREADS +
		(width % LINEAR_ELEMENT_THREADS != 0 ? 1u : 0u);
	prepare_launch_fill(launch, kernel, block_count, grid_rows,
			    LINEAR_ELEMENT_THREADS, 0, record,
			    record_byte_count);
}

static void quantize_launch_fill(struct kernel_launch *launch, uint32_t kernel,
				 const struct linear_range *values,
				 const struct linear_range *codes,
				 const struct linear_range *exponents,
				 const struct linear_range *operands,
				 uint32_t width, uint32_t vector_count)
{
	const struct activation_quantize_arguments record = {
		.values_offset = values->byte_offset,
		.codes_offset = codes->byte_offset,
		.exponents_offset = exponents->byte_offset,
		.operands_offset = operands->byte_offset,
		.width = width,
		.reserved_zero = 0,
	};
	element_launch_fill(launch, kernel, width, vector_count, &record,
			    ACTIVATION_QUANTIZE_ARGUMENT_BYTES);
}

/* A trunk launch of quantized rows: matrix_count matrices, each with its output; above LINEAR_MAXIMUM_ROWS
 * rows of the instance of their wide form, with trunk_tile_blocks blocks a tile. */
static void trunk_quantized_launch_fill(
	struct kernel_launch *launch,
	const struct linear_kernels_record *kernels, uint32_t instance,
	const struct linear_matrix *matrices, uint32_t matrix_count,
	const struct linear_range *codes, const struct linear_range *exponents,
	const struct linear_range *operands, const struct linear_range *outputs,
	uint32_t row_count)
{
	struct trunk_linear_quantized_arguments record = {
		.codes_offset = codes->byte_offset,
		.exponents_offset = exponents->byte_offset,
		.operands_offset = operands->byte_offset,
		.input_size = matrices[0].shape.input_size,
		.row_count = row_count,
	};
	for (uint32_t matrix_index = 0; matrix_index < matrix_count;
	     matrix_index++) {
		record.weights_offset[matrix_index] =
			matrices[matrix_index].stored.byte_offset;
		record.output_offset[matrix_index] =
			outputs[matrix_index].byte_offset;
		record.output_size[matrix_index] =
			matrices[matrix_index].shape.output_size;
	}
    const uint32_t output_size = record.output_size[0] + record.output_size[1];
    const enum linear_wide_form wide_form = matrices[0].wide_form;
    product_launch_fill(launch,
			handle_of(kernels,
				  trunk_family_of(FAMILY_TRUNK_LINEAR_QUANTIZED,
						  row_count, wide_form),
				  instance),
			output_size,
			trunk_tile_blocks(output_size / LINEAR_OUTPUTS_PER_TILE,
					  row_count, wide_form),
			1, matrices[0].slice_count, &record,
			TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES);
}

/* ---- the prepare functions ---- */

result_status
linear_quantize_prepare(const struct linear_quantize_description *description,
			uint64_t arena_byte_count,
			const struct linear_kernels_record *kernels,
			struct kernel_launch *launch_out)
{
    static const char what[] = "quantized rows";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t width = description->width;
    const uint32_t row_count = description->row_count;
    const struct checked_range ranges[] = {
	    { "the values", &description->values,
	      linear_values_byte_count(row_count, width), false },
	    { "the codes", &description->codes,
	      linear_codes_byte_count(row_count, width), false },
	    { "the exponents", &description->exponents,
	      linear_exponents_byte_count(row_count, width), false },
	    { "the operands", &description->operands,
	      linear_values_byte_count(row_count, width), false },
    };
    if (!kernel_is_prepared(what, kernels, FAMILY_ACTIVATION_QUANTIZE, 0) ||
	!width_is_valid(what, "a row", width) ||
	!quantized_rows_are_valid(what, row_count) ||
	!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    quantize_launch_fill(launch_out, kernels->activation_quantize, &description->values, &description->codes,
                         &description->exponents, &description->operands, width, row_count);
    return STATUS_OK;
}

result_status linear_trunk_quantized_prepare(
	const struct linear_trunk_quantized_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launch_out)
{
    static const char what[] = "a trunk linear of quantized rows";
    uint64_t matrix_byte_counts[LINEAR_MERGED_MATRICES] = { 0, 0 };
    uint32_t instance = 0;
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct linear_matrix *const matrices = description->matrices;
    const uint32_t matrix_count = description->matrix_count;
    const uint32_t row_count = description->row_count;
    const uint32_t input_size = matrices[0].shape.input_size;
    if (!trunk_matrices_are_read(what, matrices, matrix_count, matrix_byte_counts, &instance) ||
        !wide_form_is_valid(what, &matrices[0], true, row_count) ||
        !kernel_is_prepared(what, kernels,
                            trunk_family_of(FAMILY_TRUNK_LINEAR_QUANTIZED, row_count, matrices[0].wide_form),
                            instance) ||
        !trunk_rows_are_valid(what, row_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const bool is_merged = matrix_count == LINEAR_MERGED_MATRICES;
    const struct checked_range ranges[] = {
	    { "the codes", &description->codes,
	      linear_codes_byte_count(row_count, input_size), false },
	    { "the exponents", &description->exponents,
	      linear_exponents_byte_count(row_count, input_size), false },
	    { "the operands", &description->operands,
	      linear_values_byte_count(row_count, input_size), false },
	    { "the first matrix", &matrices[0].stored, matrix_byte_counts[0],
	      true },
	    { "the first output", &description->outputs[0],
	      linear_values_byte_count(row_count,
				       matrices[0].shape.output_size),
	      false },
	    { "the second matrix", &matrices[1].stored, matrix_byte_counts[1],
	      true },
	    { "the second output", &description->outputs[1],
	      linear_values_byte_count(row_count,
				       matrices[1].shape.output_size),
	      false },
    };
    if (!prepare_ranges_are_valid(what, ranges,
				  is_merged ?
					  COUNT_OF(ranges) :
					  COUNT_OF(ranges) - RANGES_OF_A_MATRIX,
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    trunk_quantized_launch_fill(launch_out, kernels, instance, matrices, matrix_count, &description->codes,
                                &description->exponents, &description->operands, description->outputs, row_count);
    return STATUS_OK;
}

result_status linear_trunk_bfloat16_prepare(
	const struct linear_trunk_bfloat16_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launch_out)
{
    static const char what[] = "a trunk linear of bfloat16 rows";
    uint64_t matrix_byte_count = 0;
    uint32_t instance = 0;
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct matrix_shape_record *const shape = &description->matrix.shape;
    const uint32_t row_count = description->row_count;
    const uint32_t segment_count = description->segment_count;
    const enum linear_wide_form wide_form = description->matrix.wide_form;
    if (!trunk_matrix_is_read(what, "the matrix", &description->matrix, &matrix_byte_count, &instance) ||
        !wide_form_is_valid(what, &description->matrix, false, row_count) ||
        !kernel_is_prepared(what, kernels, trunk_family_of(FAMILY_TRUNK_LINEAR_BFLOAT16, row_count, wide_form),
                            instance) ||
        !trunk_rows_are_valid(what, row_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (segment_count == 0 ||
	shape->output_size / LINEAR_OUTPUTS_PER_TILE % segment_count != 0) {
	    LOG_ERROR(
		    "%s: %" PRIu32
		    " segments: at least one, each a whole number of the %" PRIu32
		    " tiles",
		    what, segment_count,
		    shape->output_size / LINEAR_OUTPUTS_PER_TILE);
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct checked_range ranges[] = {
	    { "the matrix", &description->matrix.stored, matrix_byte_count,
	      true },
	    { "the values", &description->values,
	      linear_values_byte_count(row_count, (uint64_t)segment_count *
							  shape->input_size),
	      false },
	    { "the output", &description->output,
	      linear_values_byte_count(row_count, shape->output_size), false },
    };
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct trunk_linear_bfloat16_arguments record = {
	    .weights_offset = description->matrix.stored.byte_offset,
	    .values_offset = description->values.byte_offset,
	    .output_offset = description->output.byte_offset,
	    .output_size = shape->output_size,
	    .input_size = shape->input_size,
	    .row_count = row_count,
	    .segment_count = segment_count,
    };
    const uint32_t tile_count = shape->output_size / LINEAR_OUTPUTS_PER_TILE;
    product_launch_fill(launch_out,
			handle_of(kernels,
				  trunk_family_of(FAMILY_TRUNK_LINEAR_BFLOAT16,
						  row_count, wide_form),
				  instance),
			shape->output_size,
			trunk_tile_blocks(tile_count, row_count, wide_form), 1,
			description->matrix.slice_count, &record,
			TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES);
    return STATUS_OK;
}

/* The counts of the routed experts that no other chain has: at least one job, and pairs that 32 bits count. */
static bool jobs_and_pairs_are_valid(const char *what, uint32_t job_count, uint32_t row_count, uint32_t pairs_per_row)
{
    if (job_count == 0) {
	    LOG_ERROR("%s: a job table of no entry", what);
	    return false;
    }
    if (pairs_per_row == 0 || (uint64_t)row_count * pairs_per_row > UINT32_MAX) {
	    LOG_ERROR("%s: %" PRIu32
		      " pairs per row: at least one, and with %" PRIu32
		      " rows no more pairs than 32 bits count",
		      what, pairs_per_row, row_count);
	    return false;
    }
    return true;
}

result_status linear_routed_experts_prepare(
	const struct linear_routed_experts_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launches_out)
{
    static const char what[] = "the routed experts";
    uint64_t gate_up_byte_count = 0;
    uint64_t down_byte_count = 0;
    uint32_t gate_up_instance = 0;
    uint32_t down_instance = 0;
    if (!prepare_arguments_are_given(what, description, kernels,
				     launches_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct matrix_shape_record *const gate_up =
	    &description->gate_up_shape;
    const struct matrix_shape_record *const down = &description->down_shape;
    const uint32_t row_count = description->row_count;
    const uint64_t pair_count = (uint64_t)row_count * description->pairs_per_row;
    if (!prepare_reserved_is_zero(what, description->reserved_zero) ||
	!matrix_is_read(what, "gate and up", description->weight_format,
			WEIGHT_FORMAT_E2M1_SCALED, gate_up,
			description->gate_up_slice_count, &gate_up_byte_count,
			&gate_up_instance) ||
	!matrix_is_read(what, "down", description->weight_format,
			WEIGHT_FORMAT_E2M1_SCALED, down,
			description->down_slice_count, &down_byte_count,
			&down_instance) ||
	!kernel_is_prepared(what, kernels, FAMILY_EXPERT_GATE_UP,
			    gate_up_instance) ||
	!kernel_is_prepared(what, kernels, FAMILY_ACTIVATION_QUANTIZE, 0) ||
	!kernel_is_prepared(what, kernels, FAMILY_EXPERT_DOWN, down_instance) ||
	!kernel_is_prepared(what, kernels, FAMILY_EXPERT_ROW_SUM, 0) ||
	!down_reads_hidden(what, gate_up, down) ||
	!width_is_valid(what, "a row sum", down->output_size) ||
	!routed_rows_are_valid(what, row_count) ||
	!jobs_and_pairs_are_valid(what, description->job_count, row_count,
				  description->pairs_per_row) ||
	!limit_is_valid(what, description->activation_limit)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct checked_range ranges[] = {
	    { "the job table", &description->jobs,
	      linear_jobs_byte_count(description->job_count), false },
	    { "the experts' matrices", &description->expert_weights,
	      gate_up_byte_count > down_byte_count ? gate_up_byte_count :
						     down_byte_count,
	      false },
	    { "the rows' codes", &description->codes,
	      linear_codes_byte_count(row_count, gate_up->input_size), false },
	    { "the rows' exponents", &description->exponents,
	      linear_exponents_byte_count(row_count, gate_up->input_size),
	      false },
	    { "the rows' operands", &description->operands,
	      linear_values_byte_count(row_count, gate_up->input_size), false },
	    { "the hidden values", &description->hidden,
	      linear_values_byte_count(pair_count, gate_up->output_size),
	      false },
	    { "the hidden codes", &description->hidden_codes,
	      linear_codes_byte_count(pair_count, gate_up->output_size),
	      false },
	    { "the hidden exponents", &description->hidden_exponents,
	      linear_exponents_byte_count(pair_count, gate_up->output_size),
	      false },
	    { "the hidden operands", &description->hidden_operands,
	      linear_values_byte_count(pair_count, gate_up->output_size),
	      false },
	    { "the pair outputs", &description->pair_output,
	      linear_values_byte_count(pair_count, down->output_size), false },
	    { "the row outputs", &description->row_output,
	      linear_sums_byte_count(row_count, down->output_size), false },
    };
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct expert_gate_up_arguments gate_up_record = {
	    .jobs_offset = description->jobs.byte_offset,
	    .codes_offset = description->codes.byte_offset,
	    .exponents_offset = description->exponents.byte_offset,
	    .operands_offset = description->operands.byte_offset,
	    .hidden_offset = description->hidden.byte_offset,
	    .output_size = gate_up->output_size,
	    .input_size = gate_up->input_size,
	    .pair_count = (uint32_t)pair_count,
	    .pairs_per_row = description->pairs_per_row,
	    .activation_limit = description->activation_limit,
	    .reserved_zero = 0,
    };
    const struct expert_down_arguments down_record = {
	    .jobs_offset = description->jobs.byte_offset,
	    .codes_offset = description->hidden_codes.byte_offset,
	    .exponents_offset = description->hidden_exponents.byte_offset,
	    .operands_offset = description->hidden_operands.byte_offset,
	    .output_offset = description->pair_output.byte_offset,
	    .output_size = down->output_size,
	    .input_size = down->input_size,
	    .pair_count = (uint32_t)pair_count,
	    .reserved_zero = 0,
    };
    const struct expert_row_sum_arguments row_sum_record = {
	    .pair_output_offset = description->pair_output.byte_offset,
	    .row_output_offset = description->row_output.byte_offset,
	    .width = down->output_size,
	    .pairs_per_row = description->pairs_per_row,
    };
    product_launch_fill(&launches_out[LINEAR_ROUTED_EXPERTS_GATE_UP],
			kernels->expert_gate_up[gate_up_instance],
			gate_up->output_size, 1, description->job_count,
			description->gate_up_slice_count *
				EXPERT_GATE_UP_MATRICES,
			&gate_up_record, EXPERT_GATE_UP_ARGUMENT_BYTES);
    quantize_launch_fill(&launches_out[LINEAR_ROUTED_EXPERTS_QUANTIZE],
			 kernels->activation_quantize, &description->hidden,
			 &description->hidden_codes,
			 &description->hidden_exponents,
			 &description->hidden_operands, gate_up->output_size,
			 (uint32_t)pair_count);
    product_launch_fill(&launches_out[LINEAR_ROUTED_EXPERTS_DOWN],
			kernels->expert_down[down_instance], down->output_size,
			1, description->job_count,
			description->down_slice_count, &down_record,
			EXPERT_DOWN_ARGUMENT_BYTES);
    element_launch_fill(&launches_out[LINEAR_ROUTED_EXPERTS_ROW_SUM],
			kernels->expert_row_sum, down->output_size, row_count,
			&row_sum_record, EXPERT_ROW_SUM_ARGUMENT_BYTES);
    return STATUS_OK;
}

result_status linear_shared_expert_prepare(
	const struct linear_shared_expert_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launches_out)
{
    static const char what[] = "the shared expert";
    uint64_t gate_up_byte_counts[LINEAR_MERGED_MATRICES] = { 0, 0 };
    uint64_t down_byte_count = 0;
    uint32_t gate_up_instance = 0;
    uint32_t down_instance = 0;
    if (!prepare_arguments_are_given(what, description, kernels,
				     launches_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct linear_matrix gate_up[LINEAR_MERGED_MATRICES] = {
	    description->gate, description->up
    };
    const struct linear_range gate_up_outputs[LINEAR_MERGED_MATRICES] = {
	    description->gate_output, description->up_output
    };
    const struct matrix_shape_record *const gate = &description->gate.shape;
    const struct matrix_shape_record *const up = &description->up.shape;
    const struct matrix_shape_record *const down = &description->down.shape;
    const uint32_t row_count = description->row_count;
    if (!trunk_matrices_are_read(what, gate_up, LINEAR_MERGED_MATRICES,
				 gate_up_byte_counts, &gate_up_instance) ||
	!trunk_matrix_is_read(what, "down", &description->down,
			      &down_byte_count, &down_instance) ||
	!wide_form_is_valid(what, &gate_up[0], true, row_count) ||
	!wide_form_is_valid(what, &description->down, true, row_count) ||
	!kernel_is_prepared(what, kernels,
			    trunk_family_of(FAMILY_TRUNK_LINEAR_QUANTIZED,
					    row_count, gate_up[0].wide_form),
			    gate_up_instance) ||
	!kernel_is_prepared(what, kernels,
			    trunk_family_of(FAMILY_TRUNK_LINEAR_QUANTIZED,
					    row_count,
					    description->down.wide_form),
			    down_instance) ||
	!kernel_is_prepared(what, kernels, FAMILY_TRUNK_HIDDEN, 0) ||
	!kernel_is_prepared(what, kernels, FAMILY_ACTIVATION_QUANTIZE, 0)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (up->output_size != gate->output_size || up->input_size != gate->input_size) {
	    LOG_ERROR("%s: up is %" PRIu32 " by %" PRIu32 " and gate %" PRIu32
		      " by %" PRIu32 ": the two are of one shape",
		      what, up->output_size, up->input_size, gate->output_size,
		      gate->input_size);
	    return STATUS_INVALID_ARGUMENT;
    }
    if (!down_reads_hidden(what, gate, down) || !trunk_rows_are_valid(what, row_count) ||
        !limit_is_valid(what, description->activation_limit)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t hidden_byte_count =
	    linear_values_byte_count(row_count, gate->output_size);
    const struct checked_range ranges[] = {
	    { "gate", &description->gate.stored, gate_up_byte_counts[0], true },
	    { "up", &description->up.stored, gate_up_byte_counts[1], true },
	    { "down", &description->down.stored, down_byte_count, true },
	    { "the rows' codes", &description->codes,
	      linear_codes_byte_count(row_count, gate->input_size), false },
	    { "the rows' exponents", &description->exponents,
	      linear_exponents_byte_count(row_count, gate->input_size), false },
	    { "the rows' operands", &description->operands,
	      linear_values_byte_count(row_count, gate->input_size), false },
	    { "the output of gate", &description->gate_output,
	      hidden_byte_count, false },
	    { "the output of up", &description->up_output, hidden_byte_count,
	      false },
	    { "the hidden values", &description->hidden, hidden_byte_count,
	      false },
	    { "the hidden codes", &description->hidden_codes,
	      linear_codes_byte_count(row_count, gate->output_size), false },
	    { "the hidden exponents", &description->hidden_exponents,
	      linear_exponents_byte_count(row_count, gate->output_size),
	      false },
	    { "the hidden operands", &description->hidden_operands,
	      hidden_byte_count, false },
	    { "the output", &description->output,
	      linear_values_byte_count(row_count, down->output_size), false },
    };
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct trunk_hidden_arguments hidden_record = {
	    .gate_offset = description->gate_output.byte_offset,
	    .up_offset = description->up_output.byte_offset,
	    .hidden_offset = description->hidden.byte_offset,
	    .width = gate->output_size,
	    .activation_limit = description->activation_limit,
    };
    trunk_quantized_launch_fill(&launches_out[LINEAR_SHARED_EXPERT_GATE_UP],
				kernels, gate_up_instance, gate_up,
				LINEAR_MERGED_MATRICES, &description->codes,
				&description->exponents, &description->operands,
				gate_up_outputs, row_count);
    element_launch_fill(&launches_out[LINEAR_SHARED_EXPERT_HIDDEN],
			kernels->trunk_hidden, gate->output_size, row_count,
			&hidden_record, TRUNK_HIDDEN_ARGUMENT_BYTES);
    quantize_launch_fill(&launches_out[LINEAR_SHARED_EXPERT_QUANTIZE],
			 kernels->activation_quantize, &description->hidden,
			 &description->hidden_codes,
			 &description->hidden_exponents,
			 &description->hidden_operands, gate->output_size,
			 row_count);
    trunk_quantized_launch_fill(&launches_out[LINEAR_SHARED_EXPERT_DOWN],
				kernels, down_instance, &description->down, 1,
				&description->hidden_codes,
				&description->hidden_exponents,
				&description->hidden_operands,
				&description->output, row_count);
    return STATUS_OK;
}
