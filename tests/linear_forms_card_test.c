/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_forms_card_test: on the real cards, the forms of linear_records.h at the edge of the carried form, and what
 * the kernels of the linear do where a promise of the host is broken (kernel_linear_common.cuh,
 * LINEAR_DEVICE_CHECK).
 *
 *     linear_forms_card_test [--trap <case>]
 *
 * It needs the cards free (`make card_tests`): a card without room for the arena FAILS the test, it is never skipped.
 * On every card: eight rows quantized on the card and fed to a trunk launch, three times. Each time every code,
 * exponent and operand must be the host reference's (test_linear_quantize, test_linear_operands_lay_out). The second
 * time the first block of row 3 holds the largest bfloat16 the reference quantizes to the exponent
 * ACTIVATION_CARRIED_EXPONENT_LARGEST (found by search: float32(1 / 448) decides where the exponents change)
 * and a value that rounds to 16 units of the stretch below the largest, whose operand is 2^127: every output must be
 * within the tolerance of the reference's products, on every card. The third time that block holds the next bfloat16,
 * of the exponent above: on a card whose trunk takes the carried form (class 80, and every class in the exact
 * profile), every output of row 3 must then be no finite number; on any other card within the tolerance. Every
 * other row keeps the bits of the first launch.
 * With --trap, in the builds that link the checked kernel modules (debug and thread: checking hooks 1), the one case
 * <case>, of card <case> / 3 and of promise <case> % 3, alone: a trap ends what a process can do with any card (the
 * driver's error stays), so each case is a process of its own. The launch as prepared runs and is waited for; then
 * the same launch with one promise broken must end with STATUS_DEVICE_ERROR, and the card then begins no new
 * work. Each promise is broken so that a kernel without its check would run to its end without a fault and give
 * wrong outputs: a block of its instance's size (a trunk launch of 8 slices given the block of 4, which would add
 * chains no warp wrote), a job of at most 8 rows (gate and up of a job that says 9), and pairs below the launch's pair
 * count (down of a job whose column carries the first pair beyond it). A trap is a fault the driver logs, so a batch
 * runs these cases as steps of their own, and `make card_tests`, whose steps the driver's log judges, runs the test
 * without them. With the release modules no case is run: their kernels hold no check, and `make tests` shows that
 * they hold no trap.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "linear_card_fixture.h"

#define TEST_TIME_LIMIT_SECONDS 600             /* the alarm ends the test here, whatever the driver does */
#define WORK_CAPACITY (1ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 20u)
#define ROWS 2u
#define WIDTH 256u                              /* inputs and outputs of every matrix of the test */
#define E2M1_BITS_PER_WEIGHT 4u
#define E4M3_BITS_PER_WEIGHT 8u
#define SCALE_CODE_BASE 118u
#define SLICES 4u
#define TRAP_BLOCK_SLICES 8u                    /* the trunk launch of the trap case, given the block of SLICES */
#define TRAP_TRUNK_INPUTS 512u                  /* its matrix: 8 loads, room for 8 slices */
#define ACTIVATION_LIMIT 10.0f
#define CASE_STREAM 0x10000000ull
#define WAIT_LIMIT_NANOSECONDS TEST_LINEAR_WAIT_LIMIT_NANOSECONDS
#define TRAP_FLAG "--trap"
#define ARGUMENTS_WITH_A_TRAP 3                 /* the program's name, the flag and the case */
#define CASE_ARGUMENT 2
#if defined(LINEAR_CLASS_FORMS)
#define EVERY_CLASS_CARRIES false               /* each class its own forms: class 80 alone takes the carried one */
#else
#define EVERY_CLASS_CARRIES true                /* the exact profile: every class takes class 80's forms */
#endif
#define DECIMAL 10
#define EDGE_ROW 3u                             /* the row whose first block is at an edge of the carried form */
#define TURNS 3u                                /* plain rows; the largest carried exponent; the one above */
#define TURN_AT_THE_EDGE 1u
#define TURN_ABOVE_THE_EDGE 2u
#define SEARCH_FIRST 0x4D00u                    /* 2^27: its block's exponent is 146 */
#define SEARCH_END 0x4D80u                      /* 2^28: 147 */
#define BFLOAT16_OF_16_UNITS 0x4CF8u            /* 248 * 2^19: in a block of exponent 146, 15.5 units of 16, a tie to
                                                   the even code, 256: an operand of 2^127 */
#define VALUES_STREAM 0x10000100ull

enum broken_promise { BROKEN_BLOCK, BROKEN_ROWS, BROKEN_PAIR, BROKEN_COUNT };

static const char *const promise_names[BROKEN_COUNT] = {
    [BROKEN_BLOCK] = "a block of its instance's size", [BROKEN_ROWS] = "a job of at most 8 rows",
    [BROKEN_PAIR] = "pairs below the launch's pair count"};

/* The launch of a case, as prepared, into launch_out: a trunk linear, or the routed experts' gate with up or down
 * of one job of ROWS rows whose table lies in jobs. */
static bool case_prepare(struct test_linear_card *card,
			 enum broken_promise promise,
			 struct linear_range *jobs_out,
			 struct kernel_launch *launch_out)
{
    const bool is_trunk = promise == BROKEN_BLOCK;
    const uint32_t slice_count = is_trunk ? TRAP_BLOCK_SLICES : SLICES;
    const uint32_t input_size = is_trunk ? TRAP_TRUNK_INPUTS : WIDTH;
    const struct test_linear_matrix_recipe recipe = {
	    .format = is_trunk ? WEIGHT_FORMAT_E4M3_SCALED :
				 WEIGHT_FORMAT_E2M1_SCALED,
	    .shape = { WIDTH, input_size,
		       is_trunk ? E4M3_BITS_PER_WEIGHT : E2M1_BITS_PER_WEIGHT,
		       0 },
	    .stream = CASE_STREAM + (uint64_t)promise,
	    .code = is_trunk ? test_linear_e4m3_code : test_linear_e2m1_code,
	    .scale_code_base = SCALE_CODE_BASE,
	    .scale_code_spread = 0
    };
    struct linear_range matrix = { 0, 0 };
    struct kernel_launch chain[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
    struct expert_job_record job = { .row_count = ROWS };
    test_linear_card_clear(card);
    const struct linear_range codes =
	    test_linear_card_work(card, (uint64_t)ROWS * input_size);
    const struct linear_range exponents = test_linear_card_work(
	    card, (uint64_t)ROWS * input_size / LINEAR_VALUES_PER_BLOCK);
    const struct linear_range operands = test_linear_card_work(
	    card, (uint64_t)ROWS * input_size * LINEAR_BFLOAT16_BYTES);
    const struct linear_range output = test_linear_card_work(
	    card, (uint64_t)ROWS * WIDTH * LINEAR_BFLOAT16_BYTES);
    *jobs_out = test_linear_card_work(card, sizeof(job));
    if (!test_linear_card_matrix(card, &recipe, &matrix)) {
        return false;
    }
    if (is_trunk) {
	    const struct linear_trunk_quantized_description description = {
		    .matrices = { { .stored = matrix,
				    .shape = recipe.shape,
				    .weight_format = recipe.format,
				    .slice_count = slice_count } },
		    .codes = codes,
		    .exponents = exponents,
		    .operands = operands,
		    .outputs = { output },
		    .matrix_count = 1,
		    .row_count = ROWS
	    };
	    return test_linear_expect_status(
		    linear_trunk_quantized_prepare(&description,
						   card->arena_byte_count,
						   &card->kernels, launch_out),
		    STATUS_OK, "the trunk launch is prepared");
    }
    job.gate_offset = job.up_offset = job.down_offset = matrix.byte_offset;
    for (uint32_t column = 0; column < ROWS; column++) {
        job.pair_index[column] = column;
        job.route_weight[column] = 1.0f;
    }
    const struct linear_range hidden = test_linear_card_work(
	    card, (uint64_t)ROWS * WIDTH * LINEAR_BFLOAT16_BYTES);
    const struct linear_routed_experts_description description = {
	    .jobs = *jobs_out,
	    .expert_weights = matrix,
	    .codes = codes,
	    .exponents = exponents,
	    .operands = operands,
	    .hidden = hidden,
	    .hidden_codes = test_linear_card_work(card, (uint64_t)ROWS * WIDTH),
	    .hidden_exponents = test_linear_card_work(
		    card, (uint64_t)ROWS * WIDTH / LINEAR_VALUES_PER_BLOCK),
	    .hidden_operands = test_linear_card_work(
		    card, (uint64_t)ROWS * WIDTH * LINEAR_BFLOAT16_BYTES),
	    .pair_output = output,
	    .row_output = test_linear_card_work(card, (uint64_t)ROWS * WIDTH *
							      LINEAR_SUM_BYTES),
	    .gate_up_shape = recipe.shape,
	    .down_shape = recipe.shape,
	    .weight_format = recipe.format,
	    .job_count = 1,
	    .row_count = ROWS,
	    .pairs_per_row = 1,
	    .gate_up_slice_count = slice_count,
	    .down_slice_count = slice_count,
	    .activation_limit = ACTIVATION_LIMIT
    };
    if (!test_linear_card_put(card, *jobs_out, 0, &job, sizeof(job)) ||
	!test_linear_expect_status(linear_routed_experts_prepare(
					   &description, card->arena_byte_count,
					   &card->kernels, chain),
				   STATUS_OK,
				   "the routed experts are prepared")) {
	    return false;
    }
    *launch_out = chain[promise == BROKEN_ROWS ? LINEAR_ROUTED_EXPERTS_GATE_UP :
						 LINEAR_ROUTED_EXPERTS_DOWN];
    return true;
}

/* The largest bfloat16 that, as the largest magnitude of a block, the reference quantizes to the largest exponent of
 * the carried form; 0 behind a failure line where the search finds none. */
static uint16_t carried_edge_value(void)
{
    uint16_t block[TEST_LINEAR_VALUES_PER_BLOCK] = {0};
    uint8_t codes[TEST_LINEAR_VALUES_PER_BLOCK];
    uint8_t exponent = 0;
    for (uint32_t bits = SEARCH_FIRST; bits < SEARCH_END; bits++) {
        block[0] = (uint16_t)(bits + 1u);
	if (test_linear_quantize(block, TEST_LINEAR_VALUES_PER_BLOCK, codes,
				 &exponent) != STATUS_OK) {
		break;
	}
	if (exponent > ACTIVATION_CARRIED_EXPONENT_LARGEST) {
		block[0] = (uint16_t)bits;
		return test_linear_quantize(block, TEST_LINEAR_VALUES_PER_BLOCK,
					    codes, &exponent) == STATUS_OK &&
				       exponent ==
					       ACTIVATION_CARRIED_EXPONENT_LARGEST ?
			       (uint16_t)bits :
			       0;
	}
    }
    test_linear_fail("no bfloat16 from 2^27 to 2^28 ends the largest exponent of the carried form");
    return 0;
}

/* The rows of a turn: plain, and at or above the edge of the carried form in the first block of EDGE_ROW. */
static void edge_rows(uint16_t *values, uint32_t turn, uint16_t edge_value)
{
	for (uint32_t row = 0; row < LINEAR_MAXIMUM_ROWS; row++) {
		test_linear_values(values + (size_t)row * WIDTH, WIDTH,
				   VALUES_STREAM + row,
				   TEST_LINEAR_VALUES_PLAIN);
	}
    if (turn != 0) {
        values[(size_t)EDGE_ROW * WIDTH] = turn == TURN_AT_THE_EDGE ? edge_value : (uint16_t)(edge_value + 1u);
        values[(size_t)EDGE_ROW * WIDTH + 1u] = BFLOAT16_OF_16_UNITS;
    }
}

/* What the card wrote for the rows of a turn, against the host: the codes, exponents and operands of every row; the
 * outputs: see the top. first_outputs: those of the plain rows. */
static void expect_turn(const struct test_linear_card *card, uint32_t turn,
			const uint16_t *values,
			const struct linear_quantize_description *quantize,
			struct linear_range output,
			const struct linear_matrix *matrix,
			const uint16_t *first_outputs)
{
	static uint8_t codes[LINEAR_MAXIMUM_ROWS * WIDTH];
	static uint8_t exponents[LINEAR_MAXIMUM_ROWS * WIDTH /
				 TEST_LINEAR_VALUES_PER_BLOCK];
	static double fed_values[LINEAR_MAXIMUM_ROWS * WIDTH];
	static struct test_linear_product products[LINEAR_MAXIMUM_ROWS * WIDTH];
	static uint16_t outputs[LINEAR_MAXIMUM_ROWS * WIDTH];
	const uint32_t blocks_per_row = WIDTH / TEST_LINEAR_VALUES_PER_BLOCK;
	const bool is_carried = EVERY_CLASS_CARRIES ||
				card->arithmetic_class ==
					ARITHMETIC_CLASS_SM_80;
	const bool is_beyond = turn == TURN_ABOVE_THE_EDGE && is_carried;
	bool is_known = true;
	for (uint32_t row = 0; is_known && row < LINEAR_MAXIMUM_ROWS; row++) {
		uint8_t *const row_codes = codes + (size_t)row * WIDTH;
		uint8_t *const row_exponents =
			exponents + (size_t)row * blocks_per_row;
		is_known = test_linear_quantize(values + (size_t)row * WIDTH,
						WIDTH, row_codes,
						row_exponents) == STATUS_OK &&
			   test_linear_fed_values_of_codes(
				   row_codes, WIDTH,
				   fed_values + (size_t)row * WIDTH) ==
				   STATUS_OK;
		is_known =
			is_known &&
			test_linear_card_expect_codes(
				card, quantize->codes, (uint64_t)row * WIDTH,
				row_codes, WIDTH,
				"the codes are the reference's") &&
			test_linear_card_expect_operands(
				card, quantize->operands,
				(uint64_t)row * WIDTH * LINEAR_BFLOAT16_BYTES,
				row_codes, row_exponents, WIDTH,
				"the operands are the carry of the codes");
	}
	is_known =
		is_known &&
		test_linear_card_expect_bytes(
			card, quantize->exponents, 0, exponents,
			sizeof(exponents),
			"the exponents are the reference's") &&
		test_linear_expect(
			turn == 0 ||
				exponents[(size_t)EDGE_ROW * blocks_per_row] ==
					ACTIVATION_CARRIED_EXPONENT_LARGEST +
						turn - 1u,
			"the edge block has the exponent of its turn") &&
		test_linear_card_get(card, output, 0, outputs,
				     sizeof(outputs)) &&
		test_linear_expect_status(
			test_linear_products(
				matrix->weight_format, &matrix->shape,
				test_linear_card_stored(card, matrix->stored),
				matrix->stored.byte_count, 0, WIDTH, fed_values,
				exponents, LINEAR_MAXIMUM_ROWS, 1, products),
			STATUS_OK, "the reference computes the products");
	if (!is_known) {
		return;
	}
    uint32_t wrong_count = 0;
    for (size_t index = 0; index < (size_t)LINEAR_MAXIMUM_ROWS * WIDTH;
	 index++) {
	    const bool is_edge_row = index / WIDTH == EDGE_ROW;
	    wrong_count +=
		    (is_edge_row && is_beyond &&
		     test_linear_bfloat16_is_finite(outputs[index])) ||
				    (!is_edge_row && turn != 0 &&
				     outputs[index] != first_outputs[index]) ?
			    1u :
			    0u;
    }
    (void)test_linear_expect(wrong_count == 0, "row 3 is no finite number above the edge where the trunk takes the "
                                               "carried form, and every other row keeps its bits");
    if (!is_beyond) {
	    test_linear_expect_products(turn == 0 ? "plain rows" :
					turn == TURN_AT_THE_EDGE ?
						    "at the edge" :
						    "above it",
					outputs, products,
					(uint64_t)LINEAR_MAXIMUM_ROWS * WIDTH,
					WIDTH, SLICES, NULL);
    }
    (void)printf("card %" PRIu32 " (class %" PRIu32 "), turn %" PRIu32 ": %" PRIu32 " outputs are not as they must "
                 "be\n", card->card_index, card->arithmetic_class, turn, wrong_count);
}

/* The rows of the three turns, quantized and fed to one trunk launch each: see the top. */
static void test_carried_edge(uint32_t card_index)
{
    static struct test_linear_card card;
    static uint16_t values[TURNS][LINEAR_MAXIMUM_ROWS * WIDTH];
    static uint16_t first_outputs[LINEAR_MAXIMUM_ROWS * WIDTH];
    struct kernel_launch launches[TURNS + TURNS];
    struct linear_quantize_description quantize[TURNS];
    struct linear_range outputs[TURNS];
    struct linear_matrix matrix = { .shape = { WIDTH, WIDTH,
					       E4M3_BITS_PER_WEIGHT, 0 },
				    .weight_format = WEIGHT_FORMAT_E4M3_SCALED,
				    .slice_count = SLICES };
    const uint64_t vectors = LINEAR_MAXIMUM_ROWS;
    const struct test_linear_matrix_recipe recipe = {
        .format = matrix.weight_format, .shape = matrix.shape, .stream = VALUES_STREAM,
        .code = test_linear_e4m3_code, .scale_code_base = SCALE_CODE_BASE, .scale_code_spread = 0};
    const uint16_t edge_value = carried_edge_value();
    bool is_ready = edge_value != 0 && test_linear_card_open(&card, card_index, WORK_CAPACITY, WEIGHTS_CAPACITY) &&
                    test_linear_card_matrix(&card, &recipe, &matrix.stored);
    test_linear_context("card %" PRIu32 ", the edge of the carried form", card_index);
    for (uint32_t turn = 0; is_ready && turn < TURNS; turn++) {
	    quantize[turn] = (struct linear_quantize_description){
		    .values = test_linear_card_work(
			    &card, vectors * WIDTH * LINEAR_BFLOAT16_BYTES),
		    .codes = test_linear_card_work(&card, vectors * WIDTH),
		    .exponents = test_linear_card_work(
			    &card, vectors * WIDTH / LINEAR_VALUES_PER_BLOCK),
		    .operands = test_linear_card_work(
			    &card, vectors * WIDTH * LINEAR_BFLOAT16_BYTES),
		    .width = WIDTH,
		    .row_count = LINEAR_MAXIMUM_ROWS
	    };
	    outputs[turn] = test_linear_card_work(
		    &card, vectors * WIDTH * LINEAR_BFLOAT16_BYTES);
	    const struct linear_trunk_quantized_description trunk = {
		    .matrices = { matrix },
		    .codes = quantize[turn].codes,
		    .exponents = quantize[turn].exponents,
		    .operands = quantize[turn].operands,
		    .outputs = { outputs[turn] },
		    .matrix_count = 1,
		    .row_count = LINEAR_MAXIMUM_ROWS
	    };
	    edge_rows(values[turn], turn, edge_value);
	    is_ready =
		    test_linear_card_has_room(&card) &&
		    test_linear_card_put(&card, quantize[turn].values, 0,
					 values[turn], sizeof(values[turn])) &&
		    test_linear_expect_status(
			    linear_quantize_prepare(
				    &quantize[turn], card.arena_byte_count,
				    &card.kernels, &launches[turn]),
			    STATUS_OK, "the rows' quantization is prepared") &&
		    test_linear_expect_status(
			    linear_trunk_quantized_prepare(
				    &trunk, card.arena_byte_count,
				    &card.kernels, &launches[TURNS + turn]),
			    STATUS_OK, "the trunk launch is prepared");
    }
    if (is_ready && test_linear_card_load(&card) && test_linear_card_run(&card, launches, TURNS + TURNS) &&
        test_linear_card_fetch(&card) &&
        test_linear_card_get(&card, outputs[0], 0, first_outputs, sizeof(first_outputs))) {
        for (uint32_t turn = 0; turn < TURNS; turn++) {
            expect_turn(&card, turn, values[turn], &quantize[turn], outputs[turn], &matrix, first_outputs);
        }
    }
    test_linear_card_close(&card);
}

/* One case on a card opened for it: see the top. */
static void test_broken_promise(uint32_t card_index, enum broken_promise promise)
{
    static struct test_linear_card card;
    struct linear_range jobs = { 0, 0 };
    struct kernel_launch launch;
    struct expert_job_record broken_job;
    if (!test_linear_card_open(&card, card_index, WORK_CAPACITY, WEIGHTS_CAPACITY)) {
        test_linear_card_close(&card);
        return;
    }
    test_linear_context("card %" PRIu32 ", %s", card_index, promise_names[promise]);
    if (case_prepare(&card, promise, &jobs, &launch) && test_linear_card_load(&card) &&
        test_linear_card_run(&card, &launch, 1) && test_linear_card_stored(&card, jobs) != NULL) {
        memcpy(&broken_job, test_linear_card_stored(&card, jobs), sizeof(broken_job));
        if (promise == BROKEN_BLOCK) {
		launch.block[0] =
			SLICES *
			LINEAR_THREADS_PER_WARP; /* its shared memory stays that of 8 */
	} else if (promise == BROKEN_ROWS) {
		broken_job.row_count = LINEAR_MAXIMUM_ROWS + 1u;
	} else {
            broken_job.pair_index[ROWS - 1u] = ROWS;    /* the launch's pair count: the first pair beyond it */
        }
        const bool is_put = promise == BROKEN_BLOCK ||
                            (test_linear_card_put(&card, jobs, 0, &broken_job, sizeof(broken_job)) &&
                             test_linear_card_upload(&card, jobs, 0, sizeof(broken_job)));
	const result_status run_status =
		is_put ? backend_cuda_value.launch_run(card.card, &launch, 1) :
			 STATUS_INVALID_ARGUMENT;
	const result_status wait_status =
		run_status == STATUS_OK ?
			backend_cuda_value.launch_wait(card.card,
						       WAIT_LIMIT_NANOSECONDS) :
			run_status;
	(void)test_linear_expect_status(
		wait_status, STATUS_DEVICE_ERROR,
		"a launch that breaks a promise of the host fails its card");
	(void)test_linear_expect_status(
		backend_cuda_value.launch_run(card.card, &launch, 1),
		STATUS_DEVICE_ERROR, "the card then begins no new work");
	(void)printf("card %" PRIu32 ", %s: the launch as prepared ran; with the promise broken it failed its card\n",
                     card_index, promise_names[promise]);
    }
    test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
    uint32_t card_count = 0;
    char *case_end = NULL;
    const bool shows_a_trap = argument_count == ARGUMENTS_WITH_A_TRAP && strcmp(arguments[1], TRAP_FLAG) == 0;
    const unsigned long trap_case = shows_a_trap ? strtoul(arguments[CASE_ARGUMENT], &case_end, DECIMAL) : 0;
    if (argument_count != 1 && (!shows_a_trap || case_end == arguments[CASE_ARGUMENT] || *case_end != '\0')) {
        (void)fprintf(stderr, "usage: linear_forms_card_test [%s <case>]\n", TRAP_FLAG);
        return EXIT_FAILURE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    (void)test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				    STATUS_OK, "the cards are counted");
    if (!test_linear_expect(
		card_count > 0 && card_count <= BACKEND_MAXIMUM_CARDS &&
			trap_case < (unsigned long)card_count * BROKEN_COUNT,
		"the machine has cards for the test, and the case asked for")) {
	    return EXIT_FAILURE;
    }
    for (uint32_t card_index = 0; !shows_a_trap && card_index < card_count; card_index++) {
        test_carried_edge(card_index);
    }
    if (shows_a_trap && CHECKING_HOOKS_ENABLED) {
	    test_broken_promise(
		    (uint32_t)(trap_case / BROKEN_COUNT),
		    (enum broken_promise)(trap_case % BROKEN_COUNT));
    }
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "linear_forms_card_test FAILED: %d failures\n", test_linear_failures());
        return EXIT_FAILURE;
    }
    (void)printf("linear_forms_card_test passed on %" PRIu32 " cards%s\n",
		 card_count,
		 !shows_a_trap ? " (no trap shown: that takes " TRAP_FLAG ")" :
		 CHECKING_HOOKS_ENABLED ?
				 "" :
				 " (release modules: no device check to show)");
    return EXIT_SUCCESS;
}
