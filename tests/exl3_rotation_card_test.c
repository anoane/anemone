/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_rotation_card_test.c: --exl3_expert_rotation's mixed-width tensor-core MoE forms (exl3_tc_records.h) against
 * the base forms, bit for bit, on one card. Rotation 2: the even experts are 512 columns wide, the odd ones 640. The
 * even rows pick six even (narrow) experts, the odd rows six odd (wide) ones. The mixed call (intermediate 640,
 * rotation 2; then rotation 3, the three-way form, the experts e % 3 == 0 narrow) takes every row; two base calls (i512 with the narrow rows routed, i640 with the wide rows; the other
 * rows unrouted) give each row's reference: a row never depends on another, so every row of the mixed call must equal
 * its base call's bit for bit. Decode form (exl3_tc, from job tables): rows 8 (6 live), 48 and 64; prefill form
 * (exl3_tcp, from the router's picks): rows 65 and 1,024 (gate and up in one launch), 1,500 and 2,048 (gate, then up);
 * K3, and K2 at 64 and 1,024 rows. Synthetic EXL3 matrices: random mul1 trellis words, suh and svh of both signs. It
 * needs the cards free (gpu_run).
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
#include "linear_card_fixture.h"

#define HIDDEN EXL3_TC_HIDDEN
#define EXPERTS EXL3_TCP_EXPERTS
#define PAIRS EXL3_TC_PAIRS_PER_ROW
#define JOB_ROWS LINEAR_MAXIMUM_ROWS
#define WIDE EXL3_TC_MIXED_WIDTH
#define NARROW (EXL3_TC_MIXED_WIDTH - EXL3_TC_NARROWER)
#define ROTATION 2u /* the even experts narrow (exl3_tc_width) */
#define THREE_WAY_ROTATION \
	EXL3_TC_THREE_WAY_FIRST /* the experts e % 3 == 0 narrow */
#define DISTINCT 8u /* distinct matrix sets a width; expert e takes set (e / 2) % DISTINCT of its width */
#define CLASSES 2u /* a row's experts: all narrow (even rows) or all wide (odd rows) */
#define ACTIVATION_LIMIT 10.0f
#define WORK_CAPACITY (768ull << 20)
#define WEIGHTS_CAPACITY (128ull << 20)
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
#define CALLS 3u /* mixed, narrow base, wide base */
#define NO_EXPERT EXPERTS

enum test_form { FORM_DECODE, FORM_PREFILL };
struct test_case {
	enum test_form form;
	uint32_t rows, live, bits;
};
static const struct test_case cases[] = { { FORM_DECODE, 8, 6, 3 },	   { FORM_DECODE, 48, 48, 3 },
					  { FORM_DECODE, 64, 64, 3 },	   { FORM_DECODE, 64, 64, 2 },
					  { FORM_PREFILL, 65, 65, 3 },	   { FORM_PREFILL, 1024, 1024, 3 },
					  { FORM_PREFILL, 1024, 1024, 2 }, { FORM_PREFILL, 1500, 1500, 3 },
					  { FORM_PREFILL, 2048, 2048, 3 } };

static uint64_t matrix_bytes(uint64_t input, uint64_t output, uint32_t bits)
{
	return input * output * bits / BITS_PER_BYTE + (input + output) * HALF_BYTES;
}
static uint64_t expert_bytes(uint32_t width, uint32_t bits)
{
	return 2u * matrix_bytes(HIDDEN, width, bits) + matrix_bytes(width, HIDDEN, bits);
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

/* an expert's class under rotation: 0 narrow, 1 wide */
static uint32_t class_of(uint32_t rotation, uint32_t expert)
{
	return exl3_tc_width(WIDE, rotation, expert) == WIDE ? 1u : 0u;
}

/* each row's six distinct experts of its class (row % 2), ascending, and their weights */
static void picks_fill(struct router_pick *picks, uint32_t rows, uint32_t live,
		       uint32_t rotation, uint64_t stream)
{
	for (uint32_t row = 0; row < rows; row++) {
		uint32_t count = 0;
		for (uint64_t draw = 0; row < live && count < PAIRS; draw++) {
			const uint32_t expert =
				(uint32_t)(test_linear_hash(stream, row * DRAW_STRIDE + draw) % EXPERTS);
			if (class_of(rotation, expert) != row % CLASSES)
				continue;
			bool taken = false;
			for (uint32_t slot = 0; slot < count; slot++)
				taken = taken || picks[row * PAIRS + slot].expert == expert;
			if (!taken)
				picks[row * PAIRS + count++].expert = expert;
		}
		for (uint32_t slot = count; slot < PAIRS; slot++)
			picks[row * PAIRS + slot].expert = NO_EXPERT;
		for (uint32_t first = 0; first < PAIRS; first++) /* ascending */
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

/* the router's job table of the picks (the routed experts ascending, runs cut in jobs of 8, then empty jobs) */
static uint32_t jobs_fill(struct expert_job_record *jobs, uint32_t capacity,
			  const struct router_pick *picks, uint32_t rows,
			  uint64_t (*offsets)[MODEL_EXPERT_MATRICES])
{
	memset(jobs, 0, sizeof(*jobs) * capacity);
	uint32_t job = 0;
	for (uint32_t expert = 0; expert < EXPERTS; expert++) {
		for (uint32_t pair = 0; pair < rows * PAIRS; pair++) {
			if (picks[pair].expert != expert)
				continue;
			if (job < capacity && jobs[job].row_count == JOB_ROWS)
				job++;
			if (job >= capacity)
				return job;
			struct expert_job_record *const at = &jobs[job];
			at->gate_offset = offsets[expert][0];
			at->up_offset = offsets[expert][1];
			at->down_offset = offsets[expert][2];
			at->expert = expert;
			at->pair_index[at->row_count] = pair;
			at->route_weight[at->row_count] = picks[pair].route_weight;
			at->row_count++;
		}
		if (job < capacity && jobs[job].row_count != 0)
			job++;
	}
	return job;
}

static bool case_run(struct test_linear_card *card, const struct test_case *test, uint32_t rotation, uint64_t stream)
{
	test_linear_context("%s rows %u (live %u), K%u, rotation %u", test->form == FORM_DECODE ? "decode" : "prefill",
			    test->rows, test->live, test->bits, rotation);
	test_linear_card_clear(card);
	const uint32_t widths[CLASSES] = { NARROW, WIDE };
	uint64_t sets[CLASSES][DISTINCT];
	for (uint32_t width = 0; width < CLASSES; width++) {
		for (uint32_t set = 0; set < DISTINCT; set++) {
			const uint32_t columns = widths[width];
			const struct linear_range range =
				test_linear_card_weights(
					card,
					expert_bytes(columns, test->bits));
			unsigned char *const stored = test_linear_card_writable(card, range);
			if (stored == NULL)
				return test_linear_expect(false, "room for the experts");
			sets[width][set] = range.byte_offset;
			const uint64_t base = stream * STREAM_STRIDE +
					      (width * DISTINCT + set) *
						      MODEL_EXPERT_MATRICES;
			const uint64_t gate_up = matrix_bytes(HIDDEN, columns, test->bits);
			matrix_fill(stored, HIDDEN, columns, test->bits, base);
			matrix_fill(stored + gate_up, HIDDEN, columns, test->bits, base + 1u);
			matrix_fill(stored + 2u * gate_up, columns, HIDDEN, test->bits, base + 2u);
		}
	}
	/* the calls' expert tables: [expert][gate, up, down]; the mixed one each expert at its width, the base ones every
	 * expert at the call's width (the other class is never routed there) */
	static uint64_t offsets[CALLS][EXPERTS][MODEL_EXPERT_MATRICES];
	for (uint32_t call = 0; call < CALLS; call++)
		for (uint32_t expert = 0; expert < EXPERTS; expert++) {
			const uint32_t width = call == 0 ? class_of(rotation, expert) : call - 1u;
			const uint64_t gate_up = matrix_bytes(HIDDEN, widths[width], test->bits);
			const uint64_t at = sets[width][expert % DISTINCT];
			offsets[call][expert][0] = at;
			offsets[call][expert][1] = at + gate_up;
			offsets[call][expert][2] = at + 2u * gate_up;
		}
	const uint32_t pairs = test->rows * PAIRS;
	struct router_pick *const picks =
		malloc(sizeof(*picks) * pairs * CALLS);
	struct expert_job_record *const tables =
		malloc(sizeof(*tables) * pairs * CALLS);
	uint16_t *const rows = malloc((size_t)test->rows * HIDDEN * HALF_BYTES);
	float *const outputs = malloc((size_t)test->rows * HIDDEN * FLOAT_BYTES * CALLS);
	bool passed =
		test_linear_expect(picks != NULL && tables != NULL && rows != NULL && outputs != NULL, "host memory");
	if (passed) {
		picks_fill(picks, test->rows, test->live, rotation, stream);
		for (uint32_t call = 1; call < CALLS; call++) /* base call c routes the rows of class c - 1 */
			for (uint32_t pair = 0; pair < pairs; pair++) {
				picks[call * pairs + pair] = picks[pair];
				if (pair / PAIRS % CLASSES != call - 1u)
					picks[call * pairs + pair].expert = NO_EXPERT;
			}
		for (uint32_t row = 0; row < test->rows; row++)
			test_linear_values(rows + (size_t)row * HIDDEN, HIDDEN, stream * ROW_STREAM_STRIDE + row,
					   TEST_LINEAR_VALUES_PLAIN);
	}
	struct linear_range table_ranges[CALLS], pick_ranges[CALLS],
		output_ranges[CALLS];
	for (uint32_t call = 0; call < CALLS; call++) {
		table_ranges[call] = test_linear_card_work(card, sizeof(*tables) * pairs);
		pick_ranges[call] = test_linear_card_work(card, sizeof(*picks) * pairs);
		output_ranges[call] = test_linear_card_work(card, (uint64_t)test->rows * HIDDEN * FLOAT_BYTES);
	}
	const struct linear_range offset_range =
		test_linear_card_work(card, sizeof(offsets));
	const struct linear_range hidden = test_linear_card_work(
		card, (uint64_t)test->rows * HIDDEN * HALF_BYTES);
	const uint64_t workspace_bytes =
		test->form == FORM_DECODE ?
			exl3_tc_workspace_bytes(pairs, WIDE) :
			exl3_tcp_workspace_bytes(test->rows, WIDE);
	const struct linear_range workspace =
		test_linear_card_work(card, workspace_bytes);
	passed = passed && test_linear_card_has_room(card) &&
		 test_linear_card_put(card, hidden, 0, rows, (uint64_t)test->rows * HIDDEN * HALF_BYTES) &&
		 test_linear_card_put(card, offset_range, 0, offsets, sizeof(offsets));
	for (uint32_t call = 0; call < CALLS && passed; call++) {
		const uint32_t job_count =
			jobs_fill(tables + call * pairs, pairs, picks + call * pairs, test->rows, offsets[call]);
		passed = test_linear_expect(job_count < pairs || call != 0, "the table holds the jobs") &&
			 test_linear_card_put(card, table_ranges[call], 0, tables + call * pairs,
					      sizeof(*tables) * pairs) &&
			 test_linear_card_put(card, pick_ranges[call], 0, picks + call * pairs, sizeof(*picks) * pairs);
	}
	passed = passed && test_linear_card_load(card);
	for (uint32_t call = 0; call < CALLS && passed; call++) {
		const uint32_t intermediate = call == 0 ? WIDE : widths[call - 1u];
		const uint32_t record_rotation = call == 0 ? rotation : 0u;
		struct kernel_launch launches[EXL3_TCP_MOST_LAUNCHES];
		uint32_t count = 0;
		if (test->form == FORM_DECODE) {
			const struct exl3_tc_record tc = {
				.jobs_offset = table_ranges[call].byte_offset,
				.hidden_offset = hidden.byte_offset,
				.workspace_offset = workspace.byte_offset,
				.output_offset =
					output_ranges[call].byte_offset,
				.row_count = test->rows,
				.job_capacity = pairs,
				.intermediate = intermediate,
				.bits_x2 = 2u * test->bits,
				.activation_limit = ACTIVATION_LIMIT,
				.rotation = record_rotation
			};
			count = EXL3_TC_LAUNCHES;
			passed = test_linear_expect_status(
				exl3_tc_prepare(&tc, card->arena_byte_count,
						&backend_cuda_value, card->card,
						launches),
				STATUS_OK, "the tensor-core MoE prepares");
		} else {
			const struct exl3_tcp_record tcp = {
				.picks_offset = pick_ranges[call].byte_offset,
				.expert_offsets_offset =
					offset_range.byte_offset +
					call * sizeof(offsets[0]),
				.hidden_offset = hidden.byte_offset,
				.workspace_offset = workspace.byte_offset,
				.output_offset =
					output_ranges[call].byte_offset,
				.row_count = test->rows,
				.intermediate = intermediate,
				.bits_x2 = 2u * test->bits,
				.activation_limit = ACTIVATION_LIMIT,
				.rotation = record_rotation
			};
			passed = test_linear_expect_status(
				exl3_tcp_prepare(&tcp, card->arena_byte_count,
						 &backend_cuda_value,
						 card->card, launches, &count),
				STATUS_OK,
				"the prefill tensor-core MoE prepares");
		}
		passed = passed && test_linear_card_run(card, launches, count);
	}
	passed = passed && test_linear_card_fetch(card);
	for (uint32_t call = 0; call < CALLS && passed; call++)
		passed = test_linear_card_get(card, output_ranges[call], 0,
					      outputs + (size_t)call * test->rows * HIDDEN,
					      (uint64_t)test->rows * HIDDEN * FLOAT_BYTES);
	if (passed) {
		uint32_t differing_rows = 0, first_differing = UINT32_MAX;
		for (uint32_t row = 0; row < test->rows; row++) {
			const float *const mixed = outputs + (size_t)row * HIDDEN;
			const float *const base = outputs + ((size_t)(1u + row % CLASSES) * test->rows + row) * HIDDEN;
			if (memcmp(mixed, base, HIDDEN * FLOAT_BYTES) != 0) {
				differing_rows++;
				first_differing = first_differing == UINT32_MAX ? row : first_differing;
			}
		}
		(void)printf("%s rows %u live %u K%u rotation %u: %u rows differ from their base call (first %d)\n",
			     test->form == FORM_DECODE ? "decode" : "prefill", test->rows, test->live, test->bits,
			     rotation, differing_rows, first_differing == UINT32_MAX ? -1 : (int)first_differing);
		passed = test_linear_expect(differing_rows == 0, "every row is its base call's, bit for bit");
	}
	free(picks);
	free(tables);
	free(rows);
	free(outputs);
	return passed;
}

int main(void)
{
	struct test_linear_card card;
	if (!test_linear_card_open(&card, 0, WORK_CAPACITY, WEIGHTS_CAPACITY))
		return EXIT_FAILURE;
	static const uint32_t rotations[] = { ROTATION, THREE_WAY_ROTATION };
	for (uint32_t form = 0; form < sizeof(rotations) / sizeof(rotations[0]); form++)
		for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++)
			(void)case_run(&card, &cases[index], rotations[form], 23u + index);
	test_linear_card_close(&card);
	(void)printf("exl3_rotation_card_test: %d failures\n", test_linear_failures());
	return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
