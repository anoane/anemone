/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_tcp_forms_card_test.c: the prefill tensor-core MoE's hybrid 640-column gate/up form (h640,
 * exl3_tcp_prepare_hybrid: two 256-column blocks of two tiles a warp, then 128 columns of one) against the i640
 * form (exl3_tcp_prepare_base: five 128-column blocks of one tile a warp) on one card, bit for bit, and times;
 * and the default call with the halves' route (half_rows: split_overlap mode 7's, the down operands, down GEMM and
 * combine a half each) against the base call, bit for bit, and its time against the default call's.
 * Synthetic EXL3 experts of 640 columns (EXPERT_SETS distinct sets of random mul1 trellis words and scales of both
 * signs, expert e taking set e % EXPERT_SETS, so a call streams most of its weights from memory, not from L2), each row
 * six distinct experts of the 384 with weights. A case runs both forms' whole call (route, operands, gate/up GEMMs,
 * down, combine) on the same rows and picks: every output row of the hybrid call must equal the base call's. Then each
 * form's call as one graph, the two sampled in turn on the card's clock (card_timing.h): microseconds a call. Rows
 * 512 and 1,024 (gate and up in one launch), 2,048 (gate, then up); K3, and K2 at 1,024. It needs the cards free
 * (gpu_run).
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_seam.h"
#include "anemone/exl3_tc.h"
#include "anemone/exl3_tc_records.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/router_records.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define HIDDEN EXL3_TC_HIDDEN
#define EXPERTS EXL3_TCP_EXPERTS
#define PAIRS EXL3_TC_PAIRS_PER_ROW
#define WIDTH EXL3_TC_MIXED_WIDTH /* the 640-column rank's intermediate */
#define EXPERT_SETS 96u
#define ACTIVATION_LIMIT 10.0f
#define WORK_CAPACITY (768ull << 20)
#define WEIGHTS_CAPACITY (384ull << 20)
#define HALF_ONE 0x3C00u
#define HALF_SIGN 0x8000u
#define HALF_MANTISSA_MASK 0x03FFu
#define BITS_PER_BYTE 8u
#define HALF_BYTES 2u
#define FLOAT_BYTES 4u
#define WEIGHT_BASE 0.05f
#define WEIGHT_STEPS 1000u
#define WEIGHT_SCALE 2500.0f
#define SCALE_STREAM 0x5u
#define WEIGHT_STREAM 0x9u
#define STREAM_STRIDE 131u
#define ROW_STREAM_STRIDE 977u
#define DRAW_STRIDE 1000u
#define FORMS 4u /* the base i640 call, the hybrid h640 call, the default call with the halves' route, the default
		    call with the two-blocks GEMMs */
#define HALVES_FORM 2u
#define TWO_BLOCKS_FORM 3u
#define WARM_RUNS 3u
#define TIME_SAMPLES 9u

struct test_case {
	uint32_t rows, bits;
};
static const struct test_case cases[] = { { 512, 3 }, { 1024, 3 }, { 2048, 3 }, { 1024, 2 } };

static uint64_t matrix_bytes(uint64_t input, uint64_t output, uint32_t bits)
{
	return input * output * bits / BITS_PER_BYTE + (input + output) * HALF_BYTES;
}

/* random trellis words; suh and svh FP16 of magnitude 1 to 2 and either sign */
static void matrix_fill(unsigned char *stored, uint64_t input, uint64_t output, uint32_t bits, uint64_t stream)
{
	const uint64_t trellis = input * output * bits / BITS_PER_BYTE;
	for (uint64_t word = 0; word < trellis / HALF_BYTES; word++) {
		const uint16_t value = (uint16_t)test_linear_hash(stream, word);
		memcpy(stored + HALF_BYTES * word, &value, HALF_BYTES);
	}
	for (uint64_t value = 0; value < input + output; value++) {
		const uint64_t hash = test_linear_hash(stream ^ SCALE_STREAM, value);
		const uint16_t half =
			(uint16_t)(HALF_ONE | (hash & HALF_MANTISSA_MASK) | ((hash >> 16) & 1u ? HALF_SIGN : 0u));
		memcpy(stored + trellis + HALF_BYTES * value, &half, HALF_BYTES);
	}
}

/* each row's six distinct experts (ascending) and their weights */
static void picks_fill(struct router_pick *picks, uint32_t rows,
		       uint64_t stream)
{
	for (uint32_t row = 0; row < rows; row++) {
		uint32_t count = 0;
		for (uint64_t draw = 0; count < PAIRS; draw++) {
			const uint32_t expert =
				(uint32_t)(test_linear_hash(stream, row * DRAW_STRIDE + draw) % EXPERTS);
			bool taken = false;
			for (uint32_t slot = 0; slot < count; slot++)
				taken = taken || picks[row * PAIRS + slot].expert == expert;
			if (!taken)
				picks[row * PAIRS + count++].expert = expert;
		}
		for (uint32_t first = 0; first < PAIRS; first++)
			for (uint32_t second = first + 1u; second < PAIRS; second++)
				if (picks[row * PAIRS + second].expert < picks[row * PAIRS + first].expert) {
					const uint32_t swap = picks[row * PAIRS + first].expert;
					picks[row * PAIRS + first].expert = picks[row * PAIRS + second].expert;
					picks[row * PAIRS + second].expert = swap;
				}
		for (uint32_t slot = 0; slot < PAIRS; slot++)
			picks[row * PAIRS + slot].route_weight =
				WEIGHT_BASE +
				(float)(test_linear_hash(stream ^ WEIGHT_STREAM, row * PAIRS + slot) % WEIGHT_STEPS) /
					WEIGHT_SCALE;
	}
}

static bool case_run(struct test_linear_card *card, const struct test_case *test, uint64_t stream)
{
	test_linear_context("rows %u, K%u", test->rows, test->bits);
	test_linear_card_clear(card);
	const uint64_t gate_up = matrix_bytes(HIDDEN, WIDTH, test->bits);
	const uint64_t expert = 2u * gate_up + matrix_bytes(WIDTH, HIDDEN, test->bits);
	static uint64_t offsets[EXPERTS][MODEL_EXPERT_MATRICES];
	uint64_t sets[EXPERT_SETS];
	for (uint32_t set = 0; set < EXPERT_SETS; set++) {
		const struct linear_range range =
			test_linear_card_weights(card, expert);
		unsigned char *const stored = test_linear_card_writable(card, range);
		if (stored == NULL)
			return test_linear_expect(false, "room for the experts");
		sets[set] = range.byte_offset;
		const uint64_t base =
			stream * STREAM_STRIDE + set * MODEL_EXPERT_MATRICES;
		matrix_fill(stored, HIDDEN, WIDTH, test->bits, base);
		matrix_fill(stored + gate_up, HIDDEN, WIDTH, test->bits, base + 1u);
		matrix_fill(stored + 2u * gate_up, WIDTH, HIDDEN, test->bits, base + 2u);
	}
	for (uint32_t index = 0; index < EXPERTS; index++) {
		offsets[index][0] = sets[index % EXPERT_SETS];
		offsets[index][1] = sets[index % EXPERT_SETS] + gate_up;
		offsets[index][2] = sets[index % EXPERT_SETS] + 2u * gate_up;
	}
	const uint32_t pairs = test->rows * PAIRS;
	struct router_pick *const picks = malloc(sizeof(*picks) * pairs);
	uint16_t *const rows = malloc((size_t)test->rows * HIDDEN * HALF_BYTES);
	float *const outputs = malloc((size_t)test->rows * HIDDEN * FLOAT_BYTES * FORMS);
	bool passed = test_linear_expect(picks != NULL && rows != NULL && outputs != NULL, "host memory");
	if (passed) {
		picks_fill(picks, test->rows, stream);
		for (uint32_t row = 0; row < test->rows; row++)
			test_linear_values(rows + (size_t)row * HIDDEN, HIDDEN, stream * ROW_STREAM_STRIDE + row,
					   TEST_LINEAR_VALUES_PLAIN);
	}
	const struct linear_range pick_range =
		test_linear_card_work(card, sizeof(*picks) * pairs);
	const struct linear_range offset_range =
		test_linear_card_work(card, sizeof(offsets));
	const struct linear_range hidden = test_linear_card_work(
		card, (uint64_t)test->rows * HIDDEN * HALF_BYTES);
	const struct linear_range workspace = test_linear_card_work(
		card, exl3_tcp_workspace_bytes(test->rows, WIDTH));
	struct linear_range output_ranges[FORMS];
	for (uint32_t form = 0; form < FORMS; form++)
		output_ranges[form] = test_linear_card_work(card, (uint64_t)test->rows * HIDDEN * FLOAT_BYTES);
	passed = passed && test_linear_card_has_room(card) &&
		 test_linear_card_put(card, hidden, 0, rows, (uint64_t)test->rows * HIDDEN * HALF_BYTES) &&
		 test_linear_card_put(card, offset_range, 0, offsets, sizeof(offsets)) &&
		 test_linear_card_put(card, pick_range, 0, picks, sizeof(*picks) * pairs) &&
		 test_linear_card_load(card);
	struct kernel_launch launches[FORMS][EXL3_TCP_MOST_LAUNCHES];
	uint32_t counts[FORMS] = { 0 };
	for (uint32_t form = 0; form < FORMS && passed; form++) {
		const struct exl3_tcp_record tcp = {
			.picks_offset = pick_range.byte_offset,
			.expert_offsets_offset = offset_range.byte_offset,
			.hidden_offset = hidden.byte_offset,
			.workspace_offset = workspace.byte_offset,
			.output_offset = output_ranges[form].byte_offset,
			.row_count = test->rows,
			.intermediate = WIDTH,
			.bits_x2 = 2u * test->bits,
			.activation_limit = ACTIVATION_LIMIT,
			.rotation = 0,
			.half_rows = form == HALVES_FORM ? test->rows / 2u : 0u,
			.two_blocks = form == TWO_BLOCKS_FORM ? 1u : 0u
		};
		passed =
			test_linear_expect_status(
				form == 0 ?
					exl3_tcp_prepare_base(
						&tcp, card->arena_byte_count,
						&backend_cuda_value, card->card,
						launches[form], &counts[form]) :
				form == 1 ?
					exl3_tcp_prepare_hybrid(
						&tcp, card->arena_byte_count,
						&backend_cuda_value, card->card,
						launches[form], &counts[form]) :
					exl3_tcp_prepare(
						&tcp, card->arena_byte_count,
						&backend_cuda_value, card->card,
						launches[form], &counts[form]),
				STATUS_OK,
				"the prefill tensor-core MoE prepares") &&
			test_linear_card_run(card, launches[form],
					     counts[form]);
	}
	passed = passed && test_linear_card_fetch(card);
	for (uint32_t form = 0; form < FORMS && passed; form++)
		passed = test_linear_card_get(card, output_ranges[form], 0, outputs + (size_t)form * test->rows * HIDDEN,
					      (uint64_t)test->rows * HIDDEN * FLOAT_BYTES);
	for (uint32_t form = 1; form < FORMS && passed; form++) {
		uint32_t differing_rows = 0, first_differing = UINT32_MAX;
		for (uint32_t row = 0; row < test->rows; row++) {
			const float *const base = outputs + (size_t)row * HIDDEN;
			const float *const other = outputs + ((size_t)form * test->rows + row) * HIDDEN;
			if (memcmp(base, other, HIDDEN * FLOAT_BYTES) != 0) {
				differing_rows++;
				first_differing = first_differing == UINT32_MAX ? row : first_differing;
			}
		}
		(void)printf("rows %u K%u: %u rows of the %s call differ from the i640 call (first %d)\n", test->rows,
			     test->bits, differing_rows,
			     form == HALVES_FORM ? "halves" : form == TWO_BLOCKS_FORM ? "two-blocks" : "hybrid",
			     first_differing == UINT32_MAX ? -1 : (int)first_differing);
		passed = test_linear_expect(differing_rows == 0, "every row is the i640 call's, bit for bit");
	}
	uint32_t graphs[FORMS] = { 0 };
	for (uint32_t form = 0; form < FORMS && passed; form++)
		passed = test_linear_expect_status(
			backend_cuda_value.graph_capture(
				card->card, launches[form], counts[form],
				&graphs[form]),
			STATUS_OK, "a call is captured as a graph");
	static double samples[2u * TIME_SAMPLES], savings[TIME_SAMPLES];
	const struct test_card_timing_form timed[2] = { { &graphs[0], 1 }, { &graphs[1], 1 } };
	if (passed && test_card_timing_pair(card->card, timed, WARM_RUNS, 1, TIME_SAMPLES, samples, savings)) {
		(void)printf("TIME rows=%u K%u", test->rows, test->bits);
		test_card_timing_print(samples, TIME_SAMPLES, 1.0, "i640");
		test_card_timing_print(samples + TIME_SAMPLES, TIME_SAMPLES, 1.0, "h640");
		test_card_timing_print(savings, TIME_SAMPLES, 1.0, "saving");
		(void)printf("\n");
	}
	/* the halves' call against the default call (h640 up to the joint rows, else i640): what the second launch of
	 * the down GEMM and its tiles a half cost */
	const struct test_card_timing_form halves_timed[2] = {
		{ &graphs[test->rows <= EXL3_TCP_JOINT_ROWS ? 1u : 0u], 1 },
		{ &graphs[HALVES_FORM], 1 }
	};
	if (passed && test_card_timing_pair(card->card, halves_timed, WARM_RUNS, 1, TIME_SAMPLES, samples, savings)) {
		(void)printf("TIME rows=%u K%u", test->rows, test->bits);
		test_card_timing_print(samples, TIME_SAMPLES, 1.0, "default");
		test_card_timing_print(samples + TIME_SAMPLES, TIME_SAMPLES, 1.0, "halves");
		test_card_timing_print(savings, TIME_SAMPLES, 1.0, "saving");
		(void)printf("\n");
	}
	/* the two-blocks GEMMs' call against the default call: two blocks a multiprocessor */
	const struct test_card_timing_form two_timed[2] = {
		{ &graphs[test->rows <= EXL3_TCP_JOINT_ROWS ? 1u : 0u], 1 },
		{ &graphs[TWO_BLOCKS_FORM], 1 }
	};
	if (passed && test_card_timing_pair(card->card, two_timed, WARM_RUNS, 1, TIME_SAMPLES, samples, savings)) {
		(void)printf("TIME rows=%u K%u", test->rows, test->bits);
		test_card_timing_print(samples, TIME_SAMPLES, 1.0, "default");
		test_card_timing_print(samples + TIME_SAMPLES, TIME_SAMPLES, 1.0, "two_blocks");
		test_card_timing_print(savings, TIME_SAMPLES, 1.0, "saving");
		(void)printf("\n");
	}
	free(picks);
	free(rows);
	free(outputs);
	return passed;
}

int main(void)
{
	struct test_linear_card card;
	if (!test_linear_card_open(&card, 0, WORK_CAPACITY, WEIGHTS_CAPACITY))
		return EXIT_FAILURE;
	for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++)
		(void)case_run(&card, &cases[index], 41u + index);
	test_linear_card_close(&card);
	(void)printf("exl3_tcp_forms_card_test: %d failures\n", test_linear_failures());
	return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
