/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_relaxed_test: what source/expert_relaxed.c does, without a card: a prepare function is arithmetic on a
 * description, the arena is a number, the kernel handles are made up.
 *
 *   - expert_relaxed_kernels_prepare against a table that records what it is asked: the ten functions once, in
 *     the order of the members, of kernel_expert_relaxed, each with the size of its record; a card of class 120 (or
 *     its native class) is STATUS_UNSUPPORTED and asks nothing; the first kernel_prepare that fails ends it
 *     with its status and leaves the handles untouched; a missing argument is STATUS_INVALID_ARGUMENT.
 *   - expert_relaxed_prepare, narrow (no tiles: four launches) and wide (tiles: six): the quantization and the
 *     row sum are linear_routed_experts_prepare's launches byte for byte; gate-up and down launch the relaxed
 *     kernels (wide: the plain and the small kernel of each, one record) with the grids, blocks and shared memory of
 *     expert_relaxed_records.h; narrow records are the reference chain's, wide records name the tiles, the pairs and
 *     the chain's arrays. Refused (STATUS_INVALID_ARGUMENT, the launches
 *     untouched): a missing argument; a reserved field that is not zero; a relaxed kernel not prepared; what
 *     linear_routed_experts_prepare refuses; gate and up of an output_size the wide form does not divide, and
 *     a down input_size it does not (the narrow form takes both); a hidden size above the wide form's
 *     EXPERT_RELAXED_WIDE_MOST_INPUTS (the narrow form takes it); tiles or pairs too small for their counts,
 *     or sharing a byte with each other or with an array of the chain.
 *   - expert_relaxed_fold_check: scale codes from EXPERT_RELAXED_SCALE_SMALLEST to
 *     EXPERT_RELAXED_SCALE_LARGEST pass, one code beyond either end (wherever it stands) fails, as do a
 *     missing argument and no code.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_seam.h"
#include "anemone/expert_relaxed.h"
#include "anemone/linear.h"
#include "anemone/status.h"
#include "host_memory_backend.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define RELAXED_KERNEL_COUNT 10u
#define FIRST_RECORDED_HANDLE 40u
#define FIRST_MADE_UP_HANDLE 11u
#define FIRST_RELAXED_HANDLE 60u
#define UNTOUCHED_BYTE 0xA5
#define HIDDEN 5120u /* the model's shapes */
#define INNER 2304u
#define OTHER_INNER 2368u /* a multiple of 64, not of 128 */
#define OTHER_HIDDEN 5184u /* a multiple of 64 and 32, not of 128 or 256 */
#define WIDE_ONLY_INNER 2176u /* a multiple of 128, not of 256 */
#define TOO_WIDE_HIDDEN \
	16640u /* a multiple of 256 above the wide form's 16,384 inputs */
#define PAIRS_PER_ROW 6u
#define ROWS 8u
#define JOBS 48u
#define TILES 49u
#define GATE_UP_SLICES 1u
#define DOWN_SLICES 1u
#define E2M1_BITS_PER_WEIGHT 4u
#define ACTIVATION_LIMIT 10.0f
#define ALIGNMENT LINEAR_OFFSET_ALIGNMENT_BYTES
#define ARENA_BYTES (1ull << 40u) /* far above every description */
#define WEIGHTS_BYTES (1ull << 30u)
#define TOO_FEW 1u /* bytes a range lacks */
#define CARD_MEMORY_BYTES \
	4096u /* of the stand-in card the kernels are asked of */
#define FOLD_CODES 300u /* the exact range, and the first ones again */
#define FOLD_PLACE_STEP 37u /* where a code beyond the range is put */
#define WIDE_ROWS EXPERT_ORDERED_MINIMUM_ROWS
#define UNCHANGED_COUNT UINT32_MAX
#define INVALID_CLASS 79u
#define INVALID_CMP_ORDER 2u

enum launch_dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };

struct kernel_asked {
	const char *kernel_file_name;
	const char *function_name;
	uint32_t argument_byte_count;
};

static const struct kernel_asked kernels_expected[RELAXED_KERNEL_COUNT] = {
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_RELAXED_GATE_UP,
	  EXPERT_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_RELAXED_DOWN,
	  EXPERT_DOWN_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_RELAXED_WIDE_GATE_UP,
	  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_RELAXED_WIDE_DOWN,
	  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_RELAXED_WIDE_SMALL_GATE_UP,
	  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_RELAXED_WIDE_SMALL_DOWN,
	  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_ORDERED_WIDE_GATE_UP,
	  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_ORDERED_WIDE_DOWN,
	  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_ORDERED_WIDE_SMALL_GATE_UP,
	  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES },
	{ KERNEL_FILE_EXPERT_RELAXED, KERNEL_EXPERT_ORDERED_WIDE_SMALL_DOWN,
	  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES },
};
static struct kernel_asked kernels_asked[RELAXED_KERNEL_COUNT];
static uint32_t asked_count = 0;
static uint32_t failing_call_number = 0; /* counted from 1; 0: no call fails */
static int failures = 0;
static const char *context_text = "";

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)printf("FAILED: %s: %s\n", context_text, what);
		failures++;
	}
}

MUST_CHECK static result_status
recording_kernel_prepare(struct device_card *card, const char *kernel_file_name,
			 const char *function_name,
			 uint32_t argument_byte_count, uint32_t *kernel_out)
{
	(void)card;
	if (asked_count >= RELAXED_KERNEL_COUNT) {
		return STATUS_UNSUPPORTED;
	}
	kernels_asked[asked_count] =
		(struct kernel_asked){ kernel_file_name, function_name,
				       argument_byte_count };
	asked_count++;
	if (asked_count == failing_call_number) {
		return STATUS_DEVICE_ERROR;
	}
	*kernel_out = FIRST_RECORDED_HANDLE + asked_count - 1u;
	return STATUS_OK;
}

static void test_kernels_prepare(void)
{
	static const struct backend_operations recording_backend = {
		.name = "recording", .kernel_prepare = recording_kernel_prepare
	};
	static const struct backend_operations backend_without_kernels = {
		.name = "no kernels"
	};
	struct device_card *card =
		NULL; /* of the host memory backend: the recording table looks into none */
	struct expert_relaxed_kernels kernels, untouched;
	uint32_t handles[RELAXED_KERNEL_COUNT];
	context_text = "the relaxed kernels of a card";
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	memset(&kernels, UNTOUCHED_BYTE, sizeof(kernels));
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	expect(expert_relaxed_kernels_prepare(
		       NULL, card, ARITHMETIC_CLASS_SM_80, &kernels) ==
		       STATUS_INVALID_ARGUMENT,
	       "no table");
	expect(expert_relaxed_kernels_prepare(
		       &backend_without_kernels, card, ARITHMETIC_CLASS_SM_80,
		       &kernels) == STATUS_INVALID_ARGUMENT,
	       "a table without kernel_prepare");
	expect(expert_relaxed_kernels_prepare(
		       &recording_backend, NULL, ARITHMETIC_CLASS_SM_80,
		       &kernels) == STATUS_INVALID_ARGUMENT,
	       "no card");
	expect(expert_relaxed_kernels_prepare(&recording_backend, card,
					      ARITHMETIC_CLASS_SM_80,
					      NULL) == STATUS_INVALID_ARGUMENT,
	       "nowhere to put the handles");
	expect(expert_relaxed_kernels_prepare(&recording_backend, card,
					      ARITHMETIC_CLASS_SM_120,
					      &kernels) == STATUS_UNSUPPORTED &&
		       expert_relaxed_kernels_prepare(
			       &recording_backend, card,
			       ARITHMETIC_CLASS_SM_120_NATIVE,
			       &kernels) == STATUS_UNSUPPORTED &&
		       asked_count == 0,
	       "a card of class 120 has no relaxed kernels and is asked nothing");
	for (uint32_t call_number = 1; call_number <= RELAXED_KERNEL_COUNT;
	     call_number++) {
		failing_call_number = call_number;
		asked_count = 0;
		expect(expert_relaxed_kernels_prepare(
			       &recording_backend, card, ARITHMETIC_CLASS_SM_80,
			       &kernels) == STATUS_DEVICE_ERROR &&
			       asked_count == call_number &&
			       memcmp(&kernels, &untouched, sizeof(kernels)) ==
				       0,
		       "the first failure ends the preparation with its status, the handles untouched");
	}
	failing_call_number = 0;
	asked_count = 0;
	expect(expert_relaxed_kernels_prepare(&recording_backend, card,
					      ARITHMETIC_CLASS_SM_80,
					      &kernels) == STATUS_OK &&
		       asked_count == RELAXED_KERNEL_COUNT,
	       "the ten kernels are prepared");
	memcpy(handles, &kernels, sizeof(handles));
	for (uint32_t kernel_index = 0; kernel_index < RELAXED_KERNEL_COUNT;
	     kernel_index++) {
		const struct kernel_asked *const asked =
			&kernels_asked[kernel_index];
		const struct kernel_asked *const expected =
			&kernels_expected[kernel_index];
		expect(strcmp(asked->kernel_file_name,
			      expected->kernel_file_name) == 0 &&
			       strcmp(asked->function_name,
				      expected->function_name) == 0 &&
			       asked->argument_byte_count ==
				       expected->argument_byte_count &&
			       handles[kernel_index] ==
				       FIRST_RECORDED_HANDLE + kernel_index,
		       "each is asked once, in the order of the members, by its file, its name and its record's size");
	}
	test_host_memory_backend.card_close(card, 0);
}

/* ---- descriptions laid out one range behind another ---- */

static struct linear_range next_range(uint64_t *cursor, uint64_t byte_count)
{
	const struct linear_range range = { *cursor, byte_count };
	*cursor += (byte_count + ALIGNMENT - 1u) / ALIGNMENT * ALIGNMENT;
	return range;
}

static struct expert_relaxed_description
description_for_rows(uint32_t inner, uint32_t hidden, uint32_t tile_count,
		     uint32_t row_count)
{
	struct expert_relaxed_description description;
	struct linear_routed_experts_description *const experts =
		&description.experts;
	const uint64_t pairs = (uint64_t)row_count * PAIRS_PER_ROW;
	uint64_t cursor = 0;
	memset(&description, 0, sizeof(description));
	experts->jobs = next_range(&cursor, linear_jobs_byte_count(JOBS));
	experts->expert_weights = next_range(&cursor, WEIGHTS_BYTES);
	experts->codes =
		next_range(&cursor, linear_codes_byte_count(row_count, hidden));
	experts->exponents = next_range(
		&cursor, linear_exponents_byte_count(row_count, hidden));
	experts->operands = next_range(
		&cursor, linear_values_byte_count(row_count, hidden));
	experts->hidden =
		next_range(&cursor, linear_values_byte_count(pairs, inner));
	experts->hidden_codes =
		next_range(&cursor, linear_codes_byte_count(pairs, inner));
	experts->hidden_exponents =
		next_range(&cursor, linear_exponents_byte_count(pairs, inner));
	experts->hidden_operands =
		next_range(&cursor, linear_values_byte_count(pairs, inner));
	experts->pair_output =
		next_range(&cursor, linear_values_byte_count(pairs, hidden));
	experts->row_output =
		next_range(&cursor, linear_sums_byte_count(row_count, hidden));
	experts->gate_up_shape =
		(struct matrix_shape_record){ inner, hidden,
					      E2M1_BITS_PER_WEIGHT, 0 };
	experts->down_shape =
		(struct matrix_shape_record){ hidden, inner,
					      E2M1_BITS_PER_WEIGHT, 0 };
	experts->weight_format = WEIGHT_FORMAT_E2M1_SCALED;
	experts->job_count = JOBS;
	experts->row_count = row_count;
	experts->pairs_per_row = PAIRS_PER_ROW;
	experts->gate_up_slice_count = GATE_UP_SLICES;
	experts->down_slice_count = DOWN_SLICES;
	experts->activation_limit = ACTIVATION_LIMIT;
	description.tiles = next_range(
		&cursor, (uint64_t)TILES * sizeof(struct wide_expert_tile));
	description.pairs =
		next_range(&cursor, pairs * sizeof(struct wide_pair));
	description.tile_count = tile_count;
	return description;
}

static struct expert_relaxed_description
description_of(uint32_t inner, uint32_t hidden, uint32_t tile_count)
{
	return description_for_rows(inner, hidden, tile_count, ROWS);
}

/* Relocate the tile table past the final pair range so a large table still has a valid, disjoint extent. */
static struct expert_relaxed_description
with_tile_capacity(struct expert_relaxed_description description,
		   uint32_t tile_count)
{
	uint64_t cursor = (description.pairs.byte_offset +
			   description.pairs.byte_count + ALIGNMENT - 1u) /
			  ALIGNMENT * ALIGNMENT;
	description.tile_count = tile_count;
	description.tiles =
		next_range(&cursor, (uint64_t)tile_count *
					    sizeof(struct wide_expert_tile));
	return description;
}

static struct linear_kernels_record made_up_linear(void)
{
	struct linear_kernels_record kernels;
	uint32_t handles[sizeof(kernels) / sizeof(uint32_t)];
	for (uint32_t index = 0; index < COUNT_OF(handles); index++) {
		handles[index] = FIRST_MADE_UP_HANDLE + index;
	}
	memcpy(&kernels, handles, sizeof(kernels));
	return kernels;
}

static const struct expert_relaxed_kernels made_up_relaxed = {
	FIRST_RELAXED_HANDLE,	   FIRST_RELAXED_HANDLE + 1u,
	FIRST_RELAXED_HANDLE + 2u, FIRST_RELAXED_HANDLE + 3u,
	FIRST_RELAXED_HANDLE + 4u, FIRST_RELAXED_HANDLE + 5u,
	FIRST_RELAXED_HANDLE + 6u, FIRST_RELAXED_HANDLE + 7u,
	FIRST_RELAXED_HANDLE + 8u, FIRST_RELAXED_HANDLE + 9u
};

/* threads and shared_bytes: the form's block and shared memory (expert_relaxed_records.h). */
static bool launch_is(const struct kernel_launch *launch, uint32_t kernel,
		      uint32_t grid_columns, uint32_t grid_rows,
		      uint32_t threads, uint32_t shared_bytes,
		      uint32_t record_bytes)
{
	return launch->kernel == kernel &&
	       launch->grid[DIMENSION_X] == grid_columns &&
	       launch->grid[DIMENSION_Y] == grid_rows &&
	       launch->grid[DIMENSION_Z] == 1u &&
	       launch->block[DIMENSION_X] == threads &&
	       launch->block[DIMENSION_Y] == 1u &&
	       launch->block[DIMENSION_Z] == 1u &&
	       launch->dynamic_shared_byte_count == shared_bytes &&
	       launch->argument_byte_count == record_bytes;
}

static void expect_refused(const struct expert_relaxed_description *description,
			   const struct linear_kernels_record *linear,
			   const struct expert_relaxed_kernels *kernels,
			   const char *what)
{
	struct kernel_launch launches[EXPERT_RELAXED_WIDE_LAUNCH_COUNT],
		untouched[COUNT_OF(launches)];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	expect(expert_relaxed_prepare(description, ARENA_BYTES, linear, kernels,
				      launches) == STATUS_INVALID_ARGUMENT &&
		       memcmp(launches, untouched, sizeof(launches)) == 0,
	       what);
}

static void test_prepare(void)
{
	const struct linear_kernels_record linear = made_up_linear();
	struct kernel_launch relaxed[EXPERT_RELAXED_WIDE_LAUNCH_COUNT];
	struct kernel_launch chain[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	struct expert_relaxed_description narrow =
		description_of(INNER, HIDDEN, 0);
	struct expert_relaxed_description wide =
		description_of(INNER, HIDDEN, TILES);
	context_text = "the narrow form";
	expect(expert_relaxed_prepare(&narrow, ARENA_BYTES, &linear,
				      &made_up_relaxed, relaxed) == STATUS_OK &&
		       linear_routed_experts_prepare(&narrow.experts,
						     ARENA_BYTES, &linear,
						     chain) == STATUS_OK,
	       "a narrow description is prepared");
	expect(memcmp(&relaxed[LINEAR_ROUTED_EXPERTS_QUANTIZE],
		      &chain[LINEAR_ROUTED_EXPERTS_QUANTIZE],
		      sizeof(chain[0])) == 0 &&
		       memcmp(&relaxed[LINEAR_ROUTED_EXPERTS_ROW_SUM],
			      &chain[LINEAR_ROUTED_EXPERTS_ROW_SUM],
			      sizeof(chain[0])) == 0,
	       "the quantization and the row sum are the reference chain's");
	expect(launch_is(&relaxed[LINEAR_ROUTED_EXPERTS_GATE_UP],
			 made_up_relaxed.gate_up,
			 INNER / EXPERT_RELAXED_NARROW_GATE_UP_OUTPUTS, JOBS,
			 EXPERT_RELAXED_THREADS,
			 EXPERT_RELAXED_NARROW_SHARED_BYTES,
			 EXPERT_GATE_UP_ARGUMENT_BYTES) &&
		       memcmp(relaxed[LINEAR_ROUTED_EXPERTS_GATE_UP].argument,
			      chain[LINEAR_ROUTED_EXPERTS_GATE_UP].argument,
			      LAUNCH_ARGUMENT_BYTES) == 0 &&
		       launch_is(&relaxed[LINEAR_ROUTED_EXPERTS_DOWN],
				 made_up_relaxed.down,
				 HIDDEN / EXPERT_RELAXED_NARROW_DOWN_OUTPUTS,
				 JOBS, EXPERT_RELAXED_THREADS,
				 EXPERT_RELAXED_NARROW_SHARED_BYTES,
				 EXPERT_DOWN_ARGUMENT_BYTES) &&
		       memcmp(relaxed[LINEAR_ROUTED_EXPERTS_DOWN].argument,
			      chain[LINEAR_ROUTED_EXPERTS_DOWN].argument,
			      LAUNCH_ARGUMENT_BYTES) == 0,
	       "gate-up and down launch the narrow kernels over the jobs with the reference chain's records");
	context_text = "the wide form";
	expect(expert_relaxed_prepare(&wide, ARENA_BYTES, &linear,
				      &made_up_relaxed, relaxed) == STATUS_OK,
	       "a wide description is prepared");
	expect(memcmp(&relaxed[EXPERT_RELAXED_WIDE_QUANTIZE],
		      &chain[LINEAR_ROUTED_EXPERTS_QUANTIZE],
		      sizeof(chain[0])) == 0 &&
		       memcmp(&relaxed[EXPERT_RELAXED_WIDE_ROW_SUM],
			      &chain[LINEAR_ROUTED_EXPERTS_ROW_SUM],
			      sizeof(chain[0])) == 0,
	       "the quantization and the row sum are the reference chain's");
	struct expert_relaxed_wide_gate_up_arguments gate_up;
	struct expert_relaxed_wide_down_arguments down;
	memcpy(&gate_up, relaxed[EXPERT_RELAXED_WIDE_GATE_UP].argument,
	       sizeof(gate_up));
	memcpy(&down, relaxed[EXPERT_RELAXED_WIDE_DOWN].argument, sizeof(down));
	expect(launch_is(&relaxed[EXPERT_RELAXED_WIDE_GATE_UP],
			 made_up_relaxed.wide_gate_up,
			 INNER / EXPERT_RELAXED_WIDE_GATE_UP_OUTPUTS, TILES,
			 EXPERT_RELAXED_THREADS,
			 EXPERT_RELAXED_WIDE_SHARED_BYTES,
			 EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES) &&
		       launch_is(&relaxed[EXPERT_RELAXED_WIDE_SMALL_GATE_UP],
				 made_up_relaxed.wide_small_gate_up,
				 INNER / EXPERT_RELAXED_WIDE_GATE_UP_OUTPUTS,
				 TILES, EXPERT_RELAXED_THREADS,
				 EXPERT_RELAXED_WIDE_SMALL_SHARED_BYTES,
				 EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES) &&
		       memcmp(relaxed[EXPERT_RELAXED_WIDE_GATE_UP].argument,
			      relaxed[EXPERT_RELAXED_WIDE_SMALL_GATE_UP]
				      .argument,
			      LAUNCH_ARGUMENT_BYTES) == 0 &&
		       gate_up.tiles_offset == wide.tiles.byte_offset &&
		       gate_up.pairs_offset == wide.pairs.byte_offset &&
		       gate_up.operands_offset ==
			       wide.experts.operands.byte_offset &&
		       gate_up.hidden_offset ==
			       wide.experts.hidden.byte_offset &&
		       gate_up.output_size == INNER &&
		       gate_up.input_size == HIDDEN &&
		       gate_up.tile_count == TILES &&
		       gate_up.pair_count == ROWS * PAIRS_PER_ROW &&
		       gate_up.pairs_per_row == PAIRS_PER_ROW &&
		       gate_up.activation_limit == ACTIVATION_LIMIT,
	       "gate-up launches the two wide kernels over the tiles with one record");
	expect(launch_is(&relaxed[EXPERT_RELAXED_WIDE_DOWN],
			 made_up_relaxed.wide_down,
			 HIDDEN / EXPERT_RELAXED_WIDE_DOWN_OUTPUTS, TILES,
			 EXPERT_RELAXED_THREADS,
			 EXPERT_RELAXED_WIDE_SHARED_BYTES,
			 EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES) &&
		       launch_is(&relaxed[EXPERT_RELAXED_WIDE_SMALL_DOWN],
				 made_up_relaxed.wide_small_down,
				 HIDDEN / EXPERT_RELAXED_WIDE_DOWN_OUTPUTS,
				 TILES, EXPERT_RELAXED_THREADS,
				 EXPERT_RELAXED_WIDE_SMALL_SHARED_BYTES,
				 EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES) &&
		       memcmp(relaxed[EXPERT_RELAXED_WIDE_DOWN].argument,
			      relaxed[EXPERT_RELAXED_WIDE_SMALL_DOWN].argument,
			      LAUNCH_ARGUMENT_BYTES) == 0 &&
		       down.tiles_offset == wide.tiles.byte_offset &&
		       down.pairs_offset == wide.pairs.byte_offset &&
		       down.operands_offset ==
			       wide.experts.hidden_operands.byte_offset &&
		       down.output_offset ==
			       wide.experts.pair_output.byte_offset &&
		       down.output_size == HIDDEN && down.input_size == INNER &&
		       down.tile_count == TILES &&
		       down.pair_count == ROWS * PAIRS_PER_ROW,
	       "down launches the two wide kernels over the tiles with one record");
	context_text = "refusals";
	struct kernel_launch launches[COUNT_OF(relaxed)];
	expect(expert_relaxed_prepare(NULL, ARENA_BYTES, &linear,
				      &made_up_relaxed,
				      launches) == STATUS_INVALID_ARGUMENT &&
		       expert_relaxed_prepare(&narrow, ARENA_BYTES, NULL,
					      &made_up_relaxed, launches) ==
			       STATUS_INVALID_ARGUMENT &&
		       expert_relaxed_prepare(&narrow, ARENA_BYTES, &linear,
					      NULL, launches) ==
			       STATUS_INVALID_ARGUMENT &&
		       expert_relaxed_prepare(&narrow, ARENA_BYTES, &linear,
					      &made_up_relaxed,
					      NULL) == STATUS_INVALID_ARGUMENT,
	       "a missing argument");
	struct expert_relaxed_description changed = narrow;
	changed.reserved_zero = 1;
	expect_refused(&changed, &linear, &made_up_relaxed,
		       "a reserved field that is not zero");
	struct expert_relaxed_kernels unprepared = made_up_relaxed;
	unprepared.down = BACKEND_NO_KERNEL;
	expect_refused(&narrow, &linear, &unprepared,
		       "the narrow down not prepared");
	unprepared = made_up_relaxed;
	unprepared.wide_gate_up = BACKEND_NO_KERNEL;
	expect_refused(&wide, &linear, &unprepared,
		       "the wide gate-up not prepared");
	unprepared = made_up_relaxed;
	unprepared.wide_small_down = BACKEND_NO_KERNEL;
	expect_refused(&wide, &linear, &unprepared,
		       "the wide small down not prepared");
	expect(expert_relaxed_prepare(&narrow, ARENA_BYTES, &linear,
				      &unprepared, launches) == STATUS_OK,
	       "the narrow form asks for no wide kernel");
	changed = narrow;
	changed.experts.row_count = 0;
	expect_refused(&changed, &linear, &made_up_relaxed,
		       "what the reference chain refuses: no row");
	struct expert_relaxed_description other =
		description_of(OTHER_INNER, HIDDEN, 0);
	struct kernel_launch accepted[COUNT_OF(relaxed)];
	expect(expert_relaxed_prepare(&other, ARENA_BYTES, &linear,
				      &made_up_relaxed, accepted) == STATUS_OK,
	       "the narrow form takes gate and up of 2,368 outputs");
	other.tile_count = TILES;
	expect_refused(&other, &linear, &made_up_relaxed,
		       "the wide form refuses gate and up of 2,368 outputs");
	other = description_of(INNER, OTHER_HIDDEN, 0);
	expect(expert_relaxed_prepare(&other, ARENA_BYTES, &linear,
				      &made_up_relaxed, accepted) == STATUS_OK,
	       "the narrow form takes a hidden size of 5,184");
	other.tile_count = TILES;
	expect_refused(&other, &linear, &made_up_relaxed,
		       "the wide form refuses a hidden size of 5,184");
	other = description_of(INNER, TOO_WIDE_HIDDEN, 0);
	expect(expert_relaxed_prepare(&other, ARENA_BYTES, &linear,
				      &made_up_relaxed, accepted) == STATUS_OK,
	       "the narrow form takes a hidden size of 16,640");
	other.tile_count = TILES;
	expect_refused(&other, &linear, &made_up_relaxed,
		       "the wide form refuses gate and up of 16,640 inputs");
	other = description_of(WIDE_ONLY_INNER, HIDDEN, TILES);
	expect(expert_relaxed_prepare(&other, ARENA_BYTES, &linear,
				      &made_up_relaxed, accepted) == STATUS_OK,
	       "the wide form takes down of 2,176 inputs, 17 chunks of 128");
	changed = wide;
	changed.tiles.byte_count -= TOO_FEW;
	expect_refused(&changed, &linear, &made_up_relaxed,
		       "tiles too few for tile_count");
	changed = wide;
	changed.pairs.byte_count -= TOO_FEW;
	expect_refused(&changed, &linear, &made_up_relaxed,
		       "pairs too few for the rows");
	changed = wide;
	changed.pairs.byte_offset = changed.tiles.byte_offset;
	expect_refused(&changed, &linear, &made_up_relaxed,
		       "pairs on the tiles");
	changed = wide;
	changed.tiles.byte_offset = changed.experts.hidden.byte_offset;
	expect_refused(&changed, &linear, &made_up_relaxed,
		       "tiles on the hidden values");
	changed = narrow;
	changed.tiles.byte_offset =
		changed.experts.hidden
			.byte_offset; /* the narrow form reads no tiles */
	expect(expert_relaxed_prepare(&changed, ARENA_BYTES, &linear,
				      &made_up_relaxed, accepted) == STATUS_OK,
	       "the narrow form looks at no tiles");
}

static void test_fold_check(void)
{
	uint8_t codes[FOLD_CODES];
	context_text = "the fold check";
	for (uint32_t index = 0; index < FOLD_CODES;
	     index++) { /* every code of the exact range, then its ends */
		const uint32_t span = EXPERT_RELAXED_SCALE_LARGEST -
				      EXPERT_RELAXED_SCALE_SMALLEST + 1u;
		codes[index] =
			(uint8_t)(EXPERT_RELAXED_SCALE_SMALLEST + index % span);
	}
	expect(expert_relaxed_fold_check(codes, FOLD_CODES) == STATUS_OK,
	       "the exact range passes");
	for (uint32_t place = 0; place < FOLD_CODES; place += FOLD_PLACE_STEP) {
		const uint8_t kept = codes[place];
		codes[place] = (uint8_t)(EXPERT_RELAXED_SCALE_SMALLEST - 1u);
		expect(expert_relaxed_fold_check(codes, FOLD_CODES) ==
			       STATUS_INVALID_ARGUMENT,
		       "a code below the range fails");
		codes[place] = (uint8_t)(EXPERT_RELAXED_SCALE_LARGEST + 1u);
		expect(expert_relaxed_fold_check(codes, FOLD_CODES) ==
			       STATUS_INVALID_ARGUMENT,
		       "a code above the range fails");
		codes[place] = kept;
	}
	expect(expert_relaxed_fold_check(NULL, FOLD_CODES) ==
			       STATUS_INVALID_ARGUMENT &&
		       expert_relaxed_fold_check(codes, 0) ==
			       STATUS_INVALID_ARGUMENT,
	       "a missing argument, no code");
}

static void
expect_ordered_refused(const struct expert_relaxed_description *description,
		       const struct linear_kernels_record *linear,
		       const struct expert_relaxed_kernels *kernels,
		       const char *what)
{
	struct kernel_launch launches[EXPERT_RELAXED_WIDE_LAUNCH_COUNT],
		untouched[COUNT_OF(launches)];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	expect(expert_ordered_prepare(description, ARENA_BYTES, linear, kernels,
				      launches) == STATUS_INVALID_ARGUMENT &&
		       memcmp(launches, untouched, sizeof(launches)) == 0,
	       what);
}

static void test_ordered_prepare(void)
{
	const struct linear_kernels_record linear = made_up_linear();
	const struct expert_relaxed_description wide =
		description_of(INNER, HIDDEN, TILES);
	struct kernel_launch ordered[EXPERT_RELAXED_WIDE_LAUNCH_COUNT],
		relaxed[COUNT_OF(ordered)];
	memset(ordered, UNTOUCHED_BYTE, sizeof(ordered));
	memset(relaxed, UNTOUCHED_BYTE, sizeof(relaxed));
	context_text = "the ordered wide form";
	expect(expert_ordered_prepare(&wide, ARENA_BYTES, &linear,
				      &made_up_relaxed, ordered) == STATUS_OK &&
		       expert_relaxed_prepare(&wide, ARENA_BYTES, &linear,
					      &made_up_relaxed,
					      relaxed) == STATUS_OK,
	       "both wide orders prepare");
	relaxed[EXPERT_RELAXED_WIDE_GATE_UP].kernel =
		made_up_relaxed.ordered_wide_gate_up;
	relaxed[EXPERT_RELAXED_WIDE_SMALL_GATE_UP].kernel =
		made_up_relaxed.ordered_wide_small_gate_up;
	relaxed[EXPERT_RELAXED_WIDE_DOWN].kernel =
		made_up_relaxed.ordered_wide_down;
	relaxed[EXPERT_RELAXED_WIDE_SMALL_DOWN].kernel =
		made_up_relaxed.ordered_wide_small_down;
	expect(memcmp(ordered, relaxed, sizeof(ordered)) == 0,
	       "only product handles differ: records, grids, quantization and row sum keep their bytes");
	struct expert_relaxed_kernels unprepared = made_up_relaxed;
	uint32_t *const ordered_handles[] = {
		&unprepared.ordered_wide_gate_up, &unprepared.ordered_wide_down,
		&unprepared.ordered_wide_small_gate_up,
		&unprepared.ordered_wide_small_down
	};
	for (uint32_t index = 0; index < COUNT_OF(ordered_handles); index++) {
		unprepared = made_up_relaxed;
		*ordered_handles[index] = BACKEND_NO_KERNEL;
		expect_ordered_refused(
			&wide, &linear, &unprepared,
			"every selected ordered handle is required");
		expect(expert_relaxed_prepare(&wide, ARENA_BYTES, &linear,
					      &unprepared,
					      relaxed) == STATUS_OK,
		       "relaxed wide requires no ordered handle");
	}
	unprepared = made_up_relaxed;
	unprepared.gate_up = BACKEND_NO_KERNEL;
	unprepared.down = BACKEND_NO_KERNEL;
	unprepared.wide_gate_up = BACKEND_NO_KERNEL;
	unprepared.wide_down = BACKEND_NO_KERNEL;
	unprepared.wide_small_gate_up = BACKEND_NO_KERNEL;
	unprepared.wide_small_down = BACKEND_NO_KERNEL;
	expect(expert_ordered_prepare(&wide, ARENA_BYTES, &linear, &unprepared,
				      ordered) == STATUS_OK,
	       "ordered wide requires no relaxed handle");
	expect_ordered_refused(NULL, &linear, &made_up_relaxed,
			       "a missing description");
	expect_ordered_refused(&wide, NULL, &made_up_relaxed,
			       "missing linear handles");
	expect_ordered_refused(&wide, &linear, NULL, "missing expert handles");
	expect(expert_ordered_prepare(&wide, ARENA_BYTES, &linear,
				      &made_up_relaxed,
				      NULL) == STATUS_INVALID_ARGUMENT,
	       "nowhere to put ordered launches");
	struct expert_relaxed_description changed = wide;
	changed.tile_count = 0;
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "ordered has no narrow form");
	changed = wide;
	changed.reserved_zero = 1;
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "the outer reserved field is checked");
	changed = wide;
	changed.experts.reserved_zero = 1;
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "the reference description is checked");
	changed = wide;
	changed.experts.gate_up_slice_count = 0;
	expect_ordered_refused(
		&changed, &linear, &made_up_relaxed,
		"unused slice counts still obey the reference contract");
	changed = wide;
	changed.tiles.byte_count -= TOO_FEW;
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "tile extent is checked");
	changed = wide;
	changed.pairs.byte_count -= TOO_FEW;
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "pair extent is checked");
	changed = wide;
	changed.tiles.byte_offset = changed.experts.hidden.byte_offset;
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "tiles may not overlap hidden values");
	changed = wide;
	changed.pairs.byte_offset = changed.tiles.byte_offset;
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "pairs may not overlap tiles");
	changed = description_of(OTHER_INNER, HIDDEN, TILES);
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "ordered gate/up output tiling is checked");
	changed = description_of(INNER, OTHER_HIDDEN, TILES);
	expect_ordered_refused(&changed, &linear, &made_up_relaxed,
			       "ordered down output tiling is checked");
	changed = description_of(INNER, TOO_WIDE_HIDDEN, TILES);
	expect(expert_ordered_prepare(&changed, ARENA_BYTES, &linear,
				      &made_up_relaxed, ordered) == STATUS_OK,
	       "ordered needs no relaxed sum-range bound");
	changed = with_tile_capacity(wide, EXPERT_WIDE_MOST_TILES);
	expect(expert_ordered_prepare(&changed, ARENA_BYTES, &linear,
				      &made_up_relaxed, ordered) == STATUS_OK &&
		       expert_relaxed_prepare(&changed, ARENA_BYTES, &linear,
					      &made_up_relaxed,
					      relaxed) == STATUS_OK,
	       "the largest grid Y dimension is accepted in both wide forms");
	changed = with_tile_capacity(wide, EXPERT_WIDE_MOST_TILES + 1u);
	expect_ordered_refused(
		&changed, &linear, &made_up_relaxed,
		"ordered rejects an excessive grid Y dimension with sufficient tile bytes");
	expect_refused(
		&changed, &linear, &made_up_relaxed,
		"relaxed rejects an excessive grid Y dimension with sufficient tile bytes");
}

enum selection_form {
	SELECTION_REFERENCE,
	SELECTION_ORDERED,
	SELECTION_RELAXED_NARROW,
	SELECTION_RELAXED_WIDE
};

static void
expect_selection(const struct expert_relaxed_description *description,
		 const struct linear_kernels_record *linear,
		 const struct expert_relaxed_kernels *kernels,
		 uint32_t arithmetic_class, enum cmp_order_code cmp_order,
		 bool folds_admitted, enum selection_form form)
{
	struct kernel_launch expected[EXPERT_RELAXED_WIDE_LAUNCH_COUNT],
		selected[COUNT_OF(expected)];
	struct expert_relaxed_description direct = *description;
	uint32_t selected_count = UNCHANGED_COUNT;
	uint32_t expected_count = LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT;
	memset(expected, UNTOUCHED_BYTE, sizeof(expected));
	memset(selected, UNTOUCHED_BYTE, sizeof(selected));
	result_status status;
	if (form == SELECTION_REFERENCE) {
		status = linear_routed_experts_prepare(
			&description->experts, ARENA_BYTES, linear, expected);
	} else if (form == SELECTION_ORDERED) {
		status = expert_ordered_prepare(description, ARENA_BYTES,
						linear, kernels, expected);
		expected_count = EXPERT_RELAXED_WIDE_LAUNCH_COUNT;
	} else {
		direct.tile_count = form == SELECTION_RELAXED_NARROW ?
					    0 :
					    direct.tile_count;
		status = expert_relaxed_prepare(&direct, ARENA_BYTES, linear,
						kernels, expected);
		expected_count = form == SELECTION_RELAXED_NARROW ?
					 expected_count :
					 EXPERT_RELAXED_WIDE_LAUNCH_COUNT;
	}
	expect(status == STATUS_OK, "the independent expected form prepares");
	expect(expert_select_prepare(description, ARENA_BYTES, linear, kernels,
				     arithmetic_class, cmp_order,
				     folds_admitted, selected,
				     &selected_count) == STATUS_OK &&
		       selected_count == expected_count &&
		       memcmp(selected, expected, sizeof(selected)) == 0,
	       "selection gives the requested form byte for byte and leaves unused launch entries untouched");
}

static void expect_selection_refused(
	const struct expert_relaxed_description *description,
	const struct linear_kernels_record *linear,
	const struct expert_relaxed_kernels *kernels, uint32_t arithmetic_class,
	enum cmp_order_code cmp_order, bool folds_admitted, const char *what)
{
	struct kernel_launch launches[EXPERT_RELAXED_WIDE_LAUNCH_COUNT],
		untouched[COUNT_OF(launches)];
	uint32_t count = UNCHANGED_COUNT;
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	expect(expert_select_prepare(description, ARENA_BYTES, linear, kernels,
				     arithmetic_class, cmp_order,
				     folds_admitted, launches,
				     &count) == STATUS_INVALID_ARGUMENT &&
		       count == UNCHANGED_COUNT &&
		       memcmp(launches, untouched, sizeof(launches)) == 0,
	       what);
}

static void test_selection(void)
{
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120,
					    ARITHMETIC_CLASS_SM_120_NATIVE };
	static const uint32_t wide_rows[] = { 64,
					      256,
					      EXPERT_ORDERED_MINIMUM_ROWS - 1u,
					      EXPERT_ORDERED_MINIMUM_ROWS,
					      EXPERT_ORDERED_MINIMUM_ROWS + 1u,
					      1024,
					      4096 };
	const struct linear_kernels_record linear = made_up_linear();
	context_text = "thin selection";
	for (uint32_t class_index = 0; class_index < COUNT_OF(classes);
	     class_index++) {
		for (uint32_t rows = 1; rows <= LINEAR_MAXIMUM_ROWS; rows++) {
			const struct expert_relaxed_description thin =
				description_for_rows(INNER, HIDDEN, 0, rows);
			expect_selection(&thin, &linear, NULL,
					 classes[class_index],
					 CMP_ORDER_REFERENCE, true,
					 SELECTION_REFERENCE);
			expect_selection(&thin, &linear, NULL,
					 classes[class_index],
					 CMP_ORDER_RELAXED, false,
					 SELECTION_REFERENCE);
			const bool is_cmp = classes[class_index] ==
					    ARITHMETIC_CLASS_SM_80;
			expect_selection(&thin, &linear,
					 is_cmp ? &made_up_relaxed : NULL,
					 classes[class_index],
					 CMP_ORDER_RELAXED, true,
					 is_cmp ? SELECTION_RELAXED_NARROW :
						  SELECTION_REFERENCE);
		}
	}
	context_text = "wide selection";
	for (uint32_t class_index = 0; class_index < COUNT_OF(classes);
	     class_index++) {
		for (uint32_t size = 0; size < COUNT_OF(wide_rows); size++) {
			const struct expert_relaxed_description wide =
				description_for_rows(INNER, HIDDEN, TILES,
						     wide_rows[size]);
			const bool is_cmp = classes[class_index] ==
					    ARITHMETIC_CLASS_SM_80;
			const bool is_ordered =
				is_cmp &&
				wide_rows[size] >= EXPERT_ORDERED_MINIMUM_ROWS;
			const struct expert_relaxed_kernels *const kernels =
				is_cmp ? &made_up_relaxed : NULL;
			expect_selection(&wide, &linear, kernels,
					 classes[class_index],
					 CMP_ORDER_REFERENCE, false,
					 is_ordered ? SELECTION_ORDERED :
						      SELECTION_REFERENCE);
			expect_selection(&wide, &linear, kernels,
					 classes[class_index],
					 CMP_ORDER_RELAXED, false,
					 is_ordered ? SELECTION_ORDERED :
						      SELECTION_REFERENCE);
			expect_selection(&wide, &linear, kernels,
					 classes[class_index],
					 CMP_ORDER_RELAXED, true,
					 is_cmp ? SELECTION_RELAXED_WIDE :
						  SELECTION_REFERENCE);
		}
	}
	context_text = "selection refusals";
	const struct expert_relaxed_description wide =
		description_for_rows(INNER, HIDDEN, TILES, WIDE_ROWS);
	struct expert_relaxed_description changed = wide;
	const uint32_t cmp = ARITHMETIC_CLASS_SM_80;
	expect_selection_refused(NULL, &linear, &made_up_relaxed, cmp,
				 CMP_ORDER_REFERENCE, false,
				 "missing description preserves both outputs");
	expect_selection_refused(&wide, NULL, &made_up_relaxed, cmp,
				 CMP_ORDER_REFERENCE, false,
				 "missing linear table preserves both outputs");
	expect_selection_refused(
		&wide, &linear, NULL, cmp, CMP_ORDER_REFERENCE, false,
		"missing selected expert table preserves both outputs");
	expect_selection_refused(
		&wide, &linear, &made_up_relaxed, INVALID_CLASS,
		CMP_ORDER_REFERENCE, false,
		"unknown arithmetic class preserves both outputs");
	expect_selection_refused(&wide, &linear, &made_up_relaxed, cmp,
				 (enum cmp_order_code)INVALID_CMP_ORDER, false,
				 "unknown CMP order preserves both outputs");
	expect_selection_refused(&wide, &linear, &made_up_relaxed, cmp,
				 (enum cmp_order_code) - 1, false,
				 "negative CMP order preserves both outputs");
	changed.tile_count = 0;
	expect_selection_refused(&changed, &linear, &made_up_relaxed, cmp,
				 CMP_ORDER_REFERENCE, false,
				 "wide CMP reference needs router tiles");
	expect_selection_refused(&changed, &linear, &made_up_relaxed, cmp,
				 CMP_ORDER_RELAXED, true,
				 "wide CMP relaxed needs router tiles");
	expect_selection(&changed, &linear, NULL, ARITHMETIC_CLASS_SM_120,
			 CMP_ORDER_REFERENCE, false, SELECTION_REFERENCE);
	expect_selection(&changed, &linear, NULL,
			 ARITHMETIC_CLASS_SM_120_NATIVE, CMP_ORDER_RELAXED,
			 true, SELECTION_REFERENCE);
	changed = wide;
	changed.reserved_zero = 1;
	expect_selection_refused(&changed, &linear, &made_up_relaxed, cmp,
				 CMP_ORDER_REFERENCE, false,
				 "outer reserved field preserves both outputs");
	changed = wide;
	changed.experts.hidden.byte_count -= TOO_FEW;
	expect_selection_refused(
		&changed, &linear, &made_up_relaxed, cmp, CMP_ORDER_REFERENCE,
		false, "reference extent refusal preserves both outputs");
	changed = wide;
	changed.tiles.byte_count -= TOO_FEW;
	expect_selection_refused(
		&changed, &linear, &made_up_relaxed, cmp, CMP_ORDER_REFERENCE,
		false, "selected wide extent refusal preserves both outputs");
	changed = with_tile_capacity(wide, EXPERT_WIDE_MOST_TILES + 1u);
	expect_selection_refused(
		&changed, &linear, &made_up_relaxed, cmp, CMP_ORDER_REFERENCE,
		false,
		"excessive ordered grid Y dimension preserves both outputs");
	expect_selection_refused(
		&changed, &linear, &made_up_relaxed, cmp, CMP_ORDER_RELAXED,
		true,
		"excessive relaxed grid Y dimension preserves both outputs");
	struct expert_relaxed_kernels unprepared = made_up_relaxed;
	unprepared.ordered_wide_small_down = BACKEND_NO_KERNEL;
	expect_selection_refused(
		&wide, &linear, &unprepared, cmp, CMP_ORDER_REFERENCE, false,
		"missing selected ordered handle preserves both outputs");
	expect_selection(&wide, &linear, &unprepared, cmp, CMP_ORDER_RELAXED,
			 true, SELECTION_RELAXED_WIDE);
	unprepared = made_up_relaxed;
	unprepared.wide_small_down = BACKEND_NO_KERNEL;
	expect_selection_refused(
		&wide, &linear, &unprepared, cmp, CMP_ORDER_RELAXED, true,
		"missing selected relaxed handle preserves both outputs");
	expect_selection(&wide, &linear, &unprepared, cmp, CMP_ORDER_REFERENCE,
			 false, SELECTION_ORDERED);
	changed = description_of(INNER, HIDDEN, TILES);
	changed.tiles.byte_offset = changed.experts.hidden.byte_offset;
	expect_selection(&changed, &linear, &made_up_relaxed, cmp,
			 CMP_ORDER_RELAXED, true, SELECTION_RELAXED_NARROW);
	changed.tile_count = 0;
	unprepared = made_up_relaxed;
	unprepared.down = BACKEND_NO_KERNEL;
	expect_selection_refused(
		&changed, &linear, &unprepared, cmp, CMP_ORDER_RELAXED, true,
		"missing selected narrow handle preserves both outputs");
	uint32_t count = UNCHANGED_COUNT;
	struct kernel_launch launches[EXPERT_RELAXED_WIDE_LAUNCH_COUNT],
		untouched[COUNT_OF(launches)];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	expect(expert_select_prepare(&wide, ARENA_BYTES, &linear,
				     &made_up_relaxed, cmp, CMP_ORDER_REFERENCE,
				     false, NULL,
				     &count) == STATUS_INVALID_ARGUMENT &&
		       count == UNCHANGED_COUNT,
	       "missing launch output preserves count");
	expect(expert_select_prepare(&wide, ARENA_BYTES, &linear,
				     &made_up_relaxed, cmp, CMP_ORDER_REFERENCE,
				     false, launches,
				     NULL) == STATUS_INVALID_ARGUMENT &&
		       memcmp(launches, untouched, sizeof(launches)) == 0,
	       "missing count output preserves launches");
}

int main(void)
{
	test_kernels_prepare();
	test_prepare();
	test_fold_check();
	test_ordered_prepare();
	test_selection();
	(void)printf("expert_relaxed_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
