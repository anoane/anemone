/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trunk_chunk_card_test: the eight trunk product sites of a CMP layer at the production chunk sizes, using the
 * activation quantizer and grouped BF16 layouts of the chunk program. The base launch selection is retained by
 * linear.h's direct prepare functions; the selected reference order must match every output byte at 64, 256 and
 * 1,024 rows. Joined projections and the complete shared expert also compare every intermediate, including the
 * unchanged quantizer's codes, exponents and operands. Attention, routed experts and recurrent state are outside
 * this trunk fixture; the host program test holds their slab order and data flow. --time adds base/ordered/relaxed
 * products and one-row decode, after an L2 flush, with trace markers for GPU timing (the host wait has a polling
 * floor).
 * The process must see exactly one class-80 card at the supplied PCI address before opening any arena.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/relaxed_linear.h"
#include "linear_card_fixture.h"

#define WORK_CAPACITY (512ull << 20u)
#define WEIGHTS_CAPACITY (160ull << 20u)
#define MOST_ROWS 1024u
#define MOST_INPUTS 32768u
#define MATRICES LINEAR_MERGED_MATRICES
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define VALUE_BYTES LINEAR_BFLOAT16_BYTES
#define SCALE_BASE 118u
#define SCALE_SPREAD 3u
#define WEIGHT_BITS 8u
#define STREAM 0x15000000ull
#define MATRIX_STREAM 0x100ull
#define ROW_STREAM 0x10000ull
#define RUBBISH_BASE 0xA0u
#define FILL_BYTES 65536u
#define TIMED_RUNS 5u
#define MEDIAN_RUN 2u /* the middle of five sorted runs */
#define MARKER_BASE 61000u
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0
#define TIME_LIMIT_SECONDS 1800
#define MOST_ARGUMENTS 3
#define BUS_OPTION "--bus="
#define TIME_OPTION "--time"
#define MEBIBYTE_SHIFT 20u
#define JOIN_PRODUCTS 3u
#define JOIN_LAUNCHES (JOIN_PRODUCTS + 1u)
#define SHARED_ACTIVATION_LIMIT 10.0f

/* The measured base selection of this source epoch, independent of the candidate selector. */
struct trunk_shape {
	const char *name;
	uint32_t outputs[MATRICES];
	uint32_t inputs, segments, thin_slices, wide_slices;
	enum linear_wide_form wide_form;
	bool is_bfloat16;
};
#define GROUPS LINEAR_WIDE_GROUP_BLOCKS
#define INTERLEAVED_CHUNKS LINEAR_INTERLEAVED_CHUNK_BLOCKS
#define INTERLEAVED_TILES LINEAR_INTERLEAVED_TILE_BLOCKS
static const struct trunk_shape shapes[] = {
	{ "wq_a+wkv", { 1280, 512 }, 5120, 1, 16, 1, INTERLEAVED_CHUNKS, false },
	{ "wq_b", { 32768, 0 }, 1280, 1, 1, 1, INTERLEAVED_TILES, false },
	{ "indexer.wq_b", { 4096, 0 }, 1280, 1, 8, 8, GROUPS, false },
	{ "wo_a", { 8192, 0 }, 4096, 8, 4, 1, GROUPS, true },
	{ "wo_b", { 5120, 0 }, 8192, 1, 4, 4, GROUPS, false },
	{ "shared.gate+up", { 2304, 2304 }, 5120, 1, 8, 8, GROUPS, false },
	{ "shared.down", { 5120, 0 }, 2304, 1, 4, 4, GROUPS, false },
	{ "engram.wkv", { 25600, 0 }, 6144, 1, 1, 1, INTERLEAVED_CHUNKS, false },
};
#undef GROUPS
#undef INTERLEAVED_CHUNKS
#undef INTERLEAVED_TILES
static const uint32_t row_counts[] = { 1, 64, 256, MOST_ROWS };
enum variant { BASE, ORDERED, RELAXED, VARIANTS };
enum shape_index {
	SHAPE_QUERY_A,
	SHAPE_QUERY_B,
	SHAPE_INDEXER,
	SHAPE_OUTPUT_A,
	SHAPE_OUTPUT_B,
	SHAPE_SHARED_GATE_UP,
	SHAPE_SHARED_DOWN,
	SHAPE_ENGRAM
};
#define REFERENCE_VARIANTS (ORDERED + 1u)
static const char *const variant_names[] = { "base", "ordered", "relaxed" };

struct shape_case {
	const struct trunk_shape *shape;
	uint32_t matrix_count;
	struct linear_matrix matrices[MATRICES];
	int32_t fold_references[MATRICES];
	struct linear_range values, codes, exponents, operands;
	struct linear_range outputs[VARIANTS][MATRICES];
};
static struct relaxed_linear_kernels relaxed_kernels;
static uint16_t row_values[MOST_INPUTS];
static uint32_t time_index;

static uint64_t clock_nanoseconds(void)
{
	struct timespec now = { 0 };
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0;
}

static struct linear_range first_rows(struct linear_range range,
				      uint64_t bytes_per_row, uint32_t rows)
{
	range.byte_count = bytes_per_row * rows;
	return range;
}

static bool fill(struct test_linear_card *card, struct linear_range range,
		 unsigned char byte)
{
	static unsigned char piece[FILL_BYTES];
	memset(piece, byte, sizeof(piece));
	for (uint64_t done = 0; done < range.byte_count;
	     done += sizeof(piece)) {
		const uint64_t left = range.byte_count - done;
		if (!test_linear_card_put(
			    card, range, done, piece,
			    left < sizeof(piece) ? left : sizeof(piece))) {
			return false;
		}
	}
	return true;
}

static bool shape_lay_out(struct test_linear_card *card,
			  struct shape_case *made, uint32_t shape_index)
{
	const struct trunk_shape *const shape = made->shape;
	const uint64_t row_width = (uint64_t)shape->segments * shape->inputs;
	made->matrix_count = shape->outputs[1] == 0 ? 1u : MATRICES;
	bool is_ready = true;
	for (uint32_t matrix = 0; matrix < made->matrix_count && is_ready;
	     matrix++) {
		const struct test_linear_matrix_recipe recipe = {
			.format = WEIGHT_FORMAT_E4M3_SCALED,
			.shape = { shape->outputs[matrix], shape->inputs,
				   WEIGHT_BITS, 0 },
			.stream = STREAM +
				  MATRIX_STREAM * (shape_index * MATRICES +
						   matrix + 1u),
			.code = test_linear_e4m3_code,
			.scale_code_base = SCALE_BASE,
			.scale_code_spread = SCALE_SPREAD
		};
		made->matrices[matrix] =
			(struct linear_matrix){ .shape = recipe.shape,
						.weight_format = recipe.format,
						.slice_count =
							shape->wide_slices,
						.wide_form = shape->wide_form };
		is_ready = test_linear_card_matrix(
			card, &recipe, &made->matrices[matrix].stored);
		if (is_ready) {
			const uint64_t values =
				(uint64_t)shape->outputs[matrix] *
				shape->inputs;
			is_ready = test_linear_expect_status(
				relaxed_fold_check(
					shape->is_bfloat16 ?
						RELAXED_BFLOAT16_ROWS :
						RELAXED_QUANTIZED_ROWS,
					test_linear_card_stored(
						card,
						made->matrices[matrix].stored) +
						values,
					made->matrices[matrix].stored.byte_count -
						values,
					&made->fold_references[matrix]),
				STATUS_OK,
				"the fixture's matrix has an exact scale fold");
		}
	}
	made->values = test_linear_card_work(card, MOST_ROWS * row_width *
							   VALUE_BYTES);
	if (!shape->is_bfloat16) {
		made->codes = test_linear_card_work(
			card, (uint64_t)MOST_ROWS * shape->inputs);
		made->exponents = test_linear_card_work(
			card,
			linear_exponents_byte_count(MOST_ROWS, shape->inputs));
		made->operands = test_linear_card_work(
			card,
			linear_values_byte_count(MOST_ROWS, shape->inputs));
	}
	for (uint32_t variant = 0; variant < VARIANTS; variant++) {
		for (uint32_t matrix = 0;
		     matrix < made->matrix_count && matrix < MATRICES;
		     matrix++) {
			made->outputs[variant][matrix] = test_linear_card_work(
				card,
				linear_values_byte_count(
					MOST_ROWS, shape->outputs[matrix]));
			is_ready =
				fill(card, made->outputs[variant][matrix],
				     (unsigned char)(RUBBISH_BASE + variant)) &&
				is_ready;
		}
	}
	is_ready = test_linear_card_has_room(card) && is_ready;
	for (uint32_t row = 0; row < MOST_ROWS && is_ready; row++) {
		test_linear_values(
			row_values, (uint32_t)row_width,
			STREAM + ROW_STREAM * (shape_index + 1u) + row,
			row == 0	   ? TEST_LINEAR_VALUES_PLAIN :
			shape->is_bfloat16 ? TEST_LINEAR_VALUES_SMALL :
					     TEST_LINEAR_VALUES_EDGES);
		is_ready = test_linear_card_put(card, made->values,
						row * row_width * VALUE_BYTES,
						row_values,
						row_width * VALUE_BYTES);
	}
	return is_ready;
}

static bool launch_prepare(const struct test_linear_card *card,
			   const struct shape_case *made, uint32_t rows,
			   enum variant variant, struct kernel_launch *launch)
{
	const struct trunk_shape *const shape = made->shape;
	const enum cmp_order_code order =
		variant == RELAXED ? CMP_ORDER_RELAXED : CMP_ORDER_REFERENCE;
	struct linear_matrix matrices[MATRICES] = { made->matrices[0],
						    made->matrices[1] };
	for (uint32_t matrix = 0; matrix < made->matrix_count; matrix++) {
		matrices[matrix].slice_count = rows <= LINEAR_MAXIMUM_ROWS ?
						       shape->thin_slices :
						       shape->wide_slices;
	}
	result_status status;
	if (shape->is_bfloat16) {
		const struct linear_trunk_bfloat16_description description = {
			.matrix = matrices[0],
			.values =
				first_rows(made->values,
					   (uint64_t)shape->segments *
						   shape->inputs * VALUE_BYTES,
					   rows),
			.output = first_rows(made->outputs[variant][0],
					     (uint64_t)shape->outputs[0] *
						     VALUE_BYTES,
					     rows),
			.row_count = rows,
			.segment_count = shape->segments
		};
		status = variant == BASE ?
				 linear_trunk_bfloat16_prepare(
					 &description, card->arena_byte_count,
					 &card->kernels, launch) :
				 linear_trunk_bfloat16_select_prepare(
					 &description, card->arena_byte_count,
					 &card->kernels, &relaxed_kernels,
					 card->arithmetic_class, order,
					 made->fold_references[0], launch);
	} else {
		const struct linear_trunk_quantized_description description = {
			.matrices = { matrices[0], matrices[1] },
			.codes = first_rows(made->codes, shape->inputs, rows),
			.exponents = first_rows(
				made->exponents,
				shape->inputs / TEST_LINEAR_VALUES_PER_BLOCK,
				rows),
			.operands = first_rows(
				made->operands,
				(uint64_t)shape->inputs * VALUE_BYTES, rows),
			.outputs = { first_rows(made->outputs[variant][0],
						(uint64_t)shape->outputs[0] *
							VALUE_BYTES,
						rows),
				     first_rows(made->outputs[variant][1],
						(uint64_t)shape->outputs[1] *
							VALUE_BYTES,
						rows) },
			.matrix_count = made->matrix_count,
			.row_count = rows
		};
		status = variant == BASE ?
				 linear_trunk_quantized_prepare(
					 &description, card->arena_byte_count,
					 &card->kernels, launch) :
				 linear_trunk_quantized_select_prepare(
					 &description, card->arena_byte_count,
					 &card->kernels, &relaxed_kernels,
					 card->arithmetic_class, order,
					 made->fold_references, launch);
	}
	return test_linear_expect_status(status, STATUS_OK,
					 "a layer trunk product is prepared");
}

static bool l2_flush(struct test_linear_card *card)
{
	uint64_t unused_sum = 0;
	return test_linear_expect_status(
		backend_cuda_value.digest(card->card, 0, card->work_capacity,
					  TEST_LINEAR_WAIT_LIMIT_NANOSECONDS,
					  &unused_sum),
		STATUS_OK, "the work region is read to empty the L2");
}

static double median_microseconds(struct test_linear_card *card,
				  const struct kernel_launch *launch)
{
	uint64_t durations[TIMED_RUNS];
	bool is_timed = test_linear_card_run(card, launch, 1);
	for (uint32_t run = 0; run < TIMED_RUNS; run++) {
		is_timed = l2_flush(card) && is_timed;
		const uint64_t started = clock_nanoseconds();
		is_timed = test_linear_card_run(card, launch, 1) && is_timed;
		durations[run] = clock_nanoseconds() - started;
	}
	for (uint32_t index = 1; index < TIMED_RUNS; index++) {
		const uint64_t duration = durations[index];
		uint32_t place = index;
		while (place > 0 && durations[place - 1u] > duration) {
			durations[place] = durations[place - 1u];
			place--;
		}
		durations[place] = duration;
	}
	return is_timed ? (double)durations[MEDIAN_RUN] /
				  NANOSECONDS_PER_MICROSECOND :
			  0.0;
}

static void shape_run(struct test_linear_card *card, struct shape_case *made,
		      bool is_timed)
{
	const struct trunk_shape *const shape = made->shape;
	struct kernel_launch quantize;
	bool is_ready = test_linear_card_load(card);
	if (!shape->is_bfloat16) {
		const struct linear_quantize_description description = {
			made->values,	made->codes,   made->exponents,
			made->operands, shape->inputs, MOST_ROWS
		};
		is_ready =
			is_ready &&
			test_linear_expect_status(
				linear_quantize_prepare(
					&description, card->arena_byte_count,
					&card->kernels, &quantize),
				STATUS_OK,
				"the unchanged activation quantizer is prepared") &&
			test_linear_card_run(card, &quantize, 1);
	}
	for (uint32_t row_index = 0;
	     row_index < COUNT_OF(row_counts) && is_ready; row_index++) {
		const uint32_t rows = row_counts[row_index];
		struct kernel_launch products[VARIANTS];
		for (uint32_t variant = 0; variant < VARIANTS && is_ready;
		     variant++) {
			is_ready = launch_prepare(card, made, rows,
						  (enum variant)variant,
						  &products[variant]);
		}
		is_ready = is_ready &&
			   test_linear_card_run(card, products, VARIANTS) &&
			   test_linear_card_fetch(card);
		if (!is_ready) {
			break;
		}
		for (uint32_t matrix = 0; matrix < made->matrix_count;
		     matrix++) {
			(void)test_linear_card_expect_same(
				card, made->outputs[BASE][matrix], 0,
				made->outputs[ORDERED][matrix], 0,
				(uint64_t)rows * shape->outputs[matrix] *
					VALUE_BYTES,
				"the selected reference order is the base's every output byte");
		}
		(void)printf(
			"CHUNK %s rows=%" PRIu32
			": selected reference outputs equal base, every byte\n",
			shape->name, rows);
		if (is_timed) {
			for (uint32_t variant = 0; variant < VARIANTS;
			     variant++) {
				const struct kernel_launch marker = {
					.kernel = relaxed_kernels.reduce,
					.grid = { MARKER_BASE + time_index, 1,
						  1 },
					.block = { LINEAR_THREADS_PER_WARP, 1,
						   1 },
					.argument_byte_count =
						TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES
				};
				(void)test_linear_card_run(card, &marker, 1);
				const double host_microseconds =
					median_microseconds(card,
							    &products[variant]);
				(void)printf("TIME %" PRIu32
					     " %s %s rows=%" PRIu32
					     " host_us=%.1f\n",
					     time_index++, shape->name,
					     variant_names[variant], rows,
					     host_microseconds);
				(void)fflush(stdout);
			}
		}
	}
	if (is_ready) {
		(void)test_linear_card_fetch(card);
		(void)test_linear_card_expect_untouched(
			card, made->values,
			"the layer's input values are only read");
		(void)test_linear_card_expect_weights_untouched(card);
	}
}

/* A producer's BF16 rows and the unchanged quantizer's three arrays, kept separate for the two chains. */
struct joined_rows {
	struct linear_range values, codes, exponents, operands;
};

static bool joined_rows_lay_out(struct test_linear_card *card,
				struct joined_rows *made,
				struct linear_range values, uint32_t width,
				uint32_t variant)
{
	made->values =
		values.byte_count != 0 ?
			values :
			test_linear_card_work(card, linear_values_byte_count(
							    MOST_ROWS, width));
	made->codes = test_linear_card_work(
		card, linear_codes_byte_count(MOST_ROWS, width));
	made->exponents = test_linear_card_work(
		card, linear_exponents_byte_count(MOST_ROWS, width));
	made->operands = test_linear_card_work(
		card, linear_values_byte_count(MOST_ROWS, width));
	const struct linear_range written[] = { made->values, made->codes,
						made->exponents,
						made->operands };
	bool is_ready = test_linear_card_has_room(card);
	for (uint32_t index = 0; index < COUNT_OF(written) && is_ready;
	     index++) {
		is_ready = fill(card, written[index],
				(unsigned char)(RUBBISH_BASE + variant));
	}
	return is_ready;
}

static bool joined_quantize_prepare(const struct test_linear_card *card,
				    const struct joined_rows *made,
				    uint32_t width, uint32_t rows,
				    struct kernel_launch *launch)
{
	const struct linear_quantize_description description = {
		.values = first_rows(made->values,
				     (uint64_t)width * VALUE_BYTES, rows),
		.codes = first_rows(made->codes, width, rows),
		.exponents = first_rows(made->exponents,
					width / TEST_LINEAR_VALUES_PER_BLOCK,
					rows),
		.operands = first_rows(made->operands,
				       (uint64_t)width * VALUE_BYTES, rows),
		.width = width,
		.row_count = rows
	};
	return test_linear_expect_status(
		linear_quantize_prepare(&description, card->arena_byte_count,
					&card->kernels, launch),
		STATUS_OK, "the joined activation quantizer is prepared");
}

static bool joined_rows_compare(const struct test_linear_card *card,
				const struct joined_rows *base,
				const struct joined_rows *ordered,
				uint32_t width, uint32_t rows)
{
	return test_linear_card_expect_same(
		       card, base->values, 0, ordered->values, 0,
		       linear_values_byte_count(rows, width),
		       "the joined BF16 values match") &&
	       test_linear_card_expect_same(
		       card, base->codes, 0, ordered->codes, 0,
		       linear_codes_byte_count(rows, width),
		       "the joined activation codes match") &&
	       test_linear_card_expect_same(
		       card, base->exponents, 0, ordered->exponents, 0,
		       linear_exponents_byte_count(rows, width),
		       "the joined activation exponents match") &&
	       test_linear_card_expect_same(
		       card, base->operands, 0, ordered->operands, 0,
		       linear_values_byte_count(rows, width),
		       "the joined activation operands match");
}

static bool joined_outputs_compare(const struct test_linear_card *card,
				   const struct shape_case *made, uint32_t rows)
{
	bool is_same = true;
	for (uint32_t matrix = 0;
	     matrix < made->matrix_count && matrix < MATRICES; matrix++) {
		is_same =
			test_linear_card_expect_same(
				card, made->outputs[BASE][matrix], 0,
				made->outputs[ORDERED][matrix], 0,
				linear_values_byte_count(
					rows, made->shape->outputs[matrix]),
				"the joined projection output matches every byte") &&
			is_same;
	}
	return is_same;
}

/* The query fan-out and the grouped output projection: no host round trip between their product and quantizer. */
static void joined_projections_run(struct test_linear_card *card,
				   const uint32_t *indices, uint32_t count,
				   const char *name)
{
	struct shape_case products[JOIN_PRODUCTS] = { 0 };
	struct joined_rows joined[REFERENCE_VARIANTS] = { 0 };
	if (count <= 1u || count > JOIN_PRODUCTS) {
		(void)test_linear_expect(
			false, "the join has a producer and consumers");
		return;
	}
	test_linear_context("%s", name);
	test_linear_card_clear(card);
	bool is_ready = true;
	for (uint32_t index = 0; index < count && is_ready; index++) {
		products[index].shape = &shapes[indices[index]];
		is_ready =
			shape_lay_out(card, &products[index], indices[index]);
	}
	const uint32_t width = products[0].shape->outputs[0];
	for (uint32_t variant = 0; variant < REFERENCE_VARIANTS && is_ready;
	     variant++) {
		is_ready = joined_rows_lay_out(card, &joined[variant],
					       products[0].outputs[variant][0],
					       width, variant);
	}
	for (uint32_t row_index = 1;
	     row_index < COUNT_OF(row_counts) && is_ready; row_index++) {
		const uint32_t rows = row_counts[row_index];
		is_ready = test_linear_card_load(card);
		if (!products[0].shape->is_bfloat16) {
			const struct joined_rows input = {
				products[0].values, products[0].codes,
				products[0].exponents, products[0].operands
			};
			struct kernel_launch quantize = { 0 };
			is_ready = is_ready &&
				   joined_quantize_prepare(
					   card, &input,
					   products[0].shape->inputs, rows,
					   &quantize) &&
				   test_linear_card_run(card, &quantize, 1);
		}
		for (uint32_t variant = 0;
		     variant < REFERENCE_VARIANTS && is_ready; variant++) {
			struct kernel_launch launches[JOIN_LAUNCHES] = { 0 };
			is_ready = launch_prepare(card, &products[0], rows,
						  (enum variant)variant,
						  &launches[0]) &&
				   joined_quantize_prepare(card,
							   &joined[variant],
							   width, rows,
							   &launches[1]);
			for (uint32_t index = 1; index < count && is_ready;
			     index++) {
				struct shape_case consumer = products[index];
				consumer.values = joined[variant].values;
				consumer.codes = joined[variant].codes;
				consumer.exponents = joined[variant].exponents;
				consumer.operands = joined[variant].operands;
				is_ready =
					launch_prepare(card, &consumer, rows,
						       (enum variant)variant,
						       &launches[index + 1u]);
			}
			is_ready = is_ready &&
				   test_linear_card_run(card, launches,
							count + 1u);
		}
		is_ready = is_ready && test_linear_card_fetch(card) &&
			   joined_rows_compare(card, &joined[BASE],
					       &joined[ORDERED], width, rows);
		for (uint32_t index = 0; index < count && is_ready; index++) {
			is_ready = joined_outputs_compare(
				card, &products[index], rows);
		}
		if (is_ready) {
			(void)printf(
				"JOIN %s rows=%" PRIu32
				": every intermediate and output byte equals base\n",
				name, rows);
		}
	}
	if (is_ready) {
		(void)test_linear_card_expect_untouched(
			card, products[0].values,
			"the join's source rows are only read");
		(void)test_linear_card_expect_weights_untouched(card);
	}
}

/* The actual four-launch shared expert preparation, including the hidden rounding and activation quantization. */
static void joined_shared_run(struct test_linear_card *card)
{
	struct shape_case gate_up = { .shape = &shapes[SHAPE_SHARED_GATE_UP] };
	struct shape_case down = { .shape = &shapes[SHAPE_SHARED_DOWN] };
	struct joined_rows hidden[REFERENCE_VARIANTS] = { 0 };
	test_linear_context("shared gate/up, hidden, quantizer, down");
	test_linear_card_clear(card);
	bool is_ready = shape_lay_out(card, &gate_up, SHAPE_SHARED_GATE_UP) &&
			shape_lay_out(card, &down, SHAPE_SHARED_DOWN);
	const uint32_t width = gate_up.shape->outputs[0];
	for (uint32_t variant = 0; variant < REFERENCE_VARIANTS && is_ready;
	     variant++) {
		is_ready = joined_rows_lay_out(card, &hidden[variant],
					       (struct linear_range){ 0 },
					       width, variant);
	}
	for (uint32_t row_index = 1;
	     row_index < COUNT_OF(row_counts) && is_ready; row_index++) {
		const uint32_t rows = row_counts[row_index];
		const struct joined_rows input = { gate_up.values,
						   gate_up.codes,
						   gate_up.exponents,
						   gate_up.operands };
		struct kernel_launch quantize = { 0 };
		is_ready = test_linear_card_load(card) &&
			   joined_quantize_prepare(card, &input,
						   gate_up.shape->inputs, rows,
						   &quantize) &&
			   test_linear_card_run(card, &quantize, 1);
		for (uint32_t variant = 0;
		     variant < REFERENCE_VARIANTS && is_ready; variant++) {
			const struct linear_shared_expert_description description = {
				.gate = gate_up.matrices[0],
				.up = gate_up.matrices[1],
				.down = down.matrices[0],
				.codes = first_rows(input.codes,
						    gate_up.shape->inputs,
						    rows),
				.exponents = first_rows(
					input.exponents,
					gate_up.shape->inputs /
						TEST_LINEAR_VALUES_PER_BLOCK,
					rows),
				.operands = first_rows(
					input.operands,
					(uint64_t)gate_up.shape->inputs *
						VALUE_BYTES,
					rows),
				.gate_output = first_rows(
					gate_up.outputs[variant][0],
					(uint64_t)width * VALUE_BYTES, rows),
				.up_output = first_rows(
					gate_up.outputs[variant][1],
					(uint64_t)width * VALUE_BYTES, rows),
				.hidden = first_rows(
					hidden[variant].values,
					(uint64_t)width * VALUE_BYTES, rows),
				.hidden_codes = first_rows(
					hidden[variant].codes, width, rows),
				.hidden_exponents = first_rows(
					hidden[variant].exponents,
					width / TEST_LINEAR_VALUES_PER_BLOCK,
					rows),
				.hidden_operands = first_rows(
					hidden[variant].operands,
					(uint64_t)width * VALUE_BYTES, rows),
				.output = first_rows(
					down.outputs[variant][0],
					(uint64_t)down.shape->outputs[0] *
						VALUE_BYTES,
					rows),
				.row_count = rows,
				.activation_limit = SHARED_ACTIVATION_LIMIT
			};
			struct kernel_launch
				launches[LINEAR_SHARED_EXPERT_LAUNCH_COUNT] = {
					0
				};
			const result_status status =
				variant == BASE ?
					linear_shared_expert_prepare(
						&description,
						card->arena_byte_count,
						&card->kernels, launches) :
					linear_shared_expert_select_prepare(
						&description,
						card->arena_byte_count,
						&card->kernels,
						&relaxed_kernels,
						card->arithmetic_class,
						CMP_ORDER_REFERENCE, NULL,
						launches);
			is_ready =
				test_linear_expect_status(
					status, STATUS_OK,
					"the complete shared expert is prepared") &&
				test_linear_card_run(
					card, launches,
					LINEAR_SHARED_EXPERT_LAUNCH_COUNT);
		}
		is_ready = is_ready && test_linear_card_fetch(card) &&
			   joined_outputs_compare(card, &gate_up, rows) &&
			   joined_rows_compare(card, &hidden[BASE],
					       &hidden[ORDERED], width, rows) &&
			   joined_outputs_compare(card, &down, rows);
		if (is_ready) {
			(void)printf(
				"JOIN shared expert rows=%" PRIu32
				": every intermediate and output byte equals base\n",
				rows);
		}
	}
	if (is_ready) {
		(void)test_linear_card_expect_untouched(
			card, gate_up.values,
			"the shared expert's source rows are only read");
		(void)test_linear_card_expect_weights_untouched(card);
	}
}

int main(int argument_count, char **arguments)
{
	static struct test_linear_card card;
	const char *bus_address = NULL;
	bool is_timed = false;
	for (int index = 1; index < argument_count; index++) {
		if (strncmp(arguments[index], BUS_OPTION, strlen(BUS_OPTION)) ==
		    0) {
			bus_address = arguments[index] + strlen(BUS_OPTION);
		} else if (strcmp(arguments[index], TIME_OPTION) == 0) {
			is_timed = true;
		} else {
			bus_address = NULL;
			break;
		}
	}
	if (bus_address == NULL || argument_count > MOST_ARGUMENTS) {
		(void)fprintf(
			stderr,
			"usage: trunk_chunk_card_test --bus=<PCI address> [--time]\n");
		return EXIT_FAILURE;
	}
	(void)alarm(TIME_LIMIT_SECONDS);
	uint32_t card_count = 0;
	struct card_description_record description;
	if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				      STATUS_OK, "the cards are counted") &&
	    test_linear_expect(card_count == 1u,
			       "exactly one card is visible") &&
	    test_linear_expect_status(
		    backend_cuda_value.card_describe(0, &description),
		    STATUS_OK, "the card is described") &&
	    test_linear_expect(
		    description.arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
			    strcmp(description.bus_address, bus_address) == 0,
		    "the class-80 card has the required PCI address") &&
	    test_linear_card_open(&card, 0, WORK_CAPACITY, WEIGHTS_CAPACITY) &&
	    test_linear_expect_status(
		    linear_wide_kernels_prepare(
			    &backend_cuda_value, card.card,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    card.arithmetic_class, &card.kernels),
		    STATUS_OK,
		    "the base's production wide kernels are prepared") &&
	    test_linear_expect_status(
		    relaxed_linear_kernels_prepare(
			    &backend_cuda_value, card.card,
			    card.arithmetic_class, &relaxed_kernels),
		    STATUS_OK,
		    "the ordered and relaxed kernels are prepared")) {
		(void)printf("card %s (%s), arena %" PRIu64 " MiB\n",
			     description.name, description.bus_address,
			     card.arena_byte_count >> MEBIBYTE_SHIFT);
		static const uint32_t query_join[] = { SHAPE_QUERY_A,
						       SHAPE_QUERY_B,
						       SHAPE_INDEXER };
		static const uint32_t output_join[] = { SHAPE_OUTPUT_A,
							SHAPE_OUTPUT_B };
		joined_projections_run(&card, query_join, COUNT_OF(query_join),
				       "query and indexer projections");
		joined_projections_run(&card, output_join,
				       COUNT_OF(output_join),
				       "grouped output projections");
		joined_shared_run(&card);
		for (uint32_t shape_index = 0; shape_index < COUNT_OF(shapes);
		     shape_index++) {
			struct shape_case made = {
				.shape = &shapes[shape_index]
			};
			test_linear_context("%s", made.shape->name);
			test_linear_card_clear(&card);
			if (shape_lay_out(&card, &made, shape_index)) {
				shape_run(&card, &made, is_timed);
			}
		}
	}
	test_linear_card_close(&card);
	if (test_linear_failures() != 0) {
		(void)fprintf(stderr,
			      "trunk_chunk_card_test FAILED: %d failures\n",
			      test_linear_failures());
		return EXIT_FAILURE;
	}
	(void)printf("trunk_chunk_card_test passed\n");
	return EXIT_SUCCESS;
}
