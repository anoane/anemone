/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_program_refusal_test: what the programs of DeepSeek-V4.1-Flash (source/model_v41_program.c) refuse, and
 * the model's table of the seam, from the plan of first_token_plan.md 1.3 made from the model's pack held as tables
 * only (model_v41_pack.h), on the backend in host memory (model_v41_program_test makes the programs).
 *
 *   - Each refusal answers its status with one error line and leaves the output untouched: a missing argument, 0 rows
 *     or more than the plan's, a kind outside the enumeration, a wide, cone or full program (unsupported), a card not
 *     given, without a build of the kernels or of a class round 2's table lacks; a plan that gives a card no room for
 *     an array, puts a layer, the embedding or the head on no card of its own (host memory among them), gives a card
 *     other layers or n-gram tables than it puts there, or names an expert table outside its card's image or off a word
 *     of it; both faults of a program's memory.
 *   - The model's name finds its table and no other name does, nor 64 bytes without a NUL.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "log_expectation.h"
#include "anemone/loader.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"

#define CARDS TEST_MODEL_V41_CARDS
#define CONTEXT 16384u
#define ROWS 8u
#define SESSIONS 3u
#define PRO_CARD 4u
#define HOST_CARD_BYTES 4096u                      /* an arena no launch reads: the programs only name offsets */
#define CAPTURE_CAPACITY 4096u
#define UNKNOWN_CLASS 75u                          /* a class with no row in round 2's table */
#define REFUSED_CARD 3u                            /* a 170HX of layers 24 to 31 */
#define REFUSED_LAYER 8u                           /* the first of card 1 */
#define ENGRAM_CARD 1u                             /* the card of layer 14 */
#define LATER_TABLE 1u                             /* the n-gram table of layer 14 */
#define MEMORY_FAULTS 2u                           /* the builder, then the program */
#define BEYOND_THE_LAYERS 40u                      /* a layer bit no layer has */
#define OFF_THE_PLAN "none of its 5 cards"         /* the refusal of a plan's cards */

enum missing_argument { NO_PLAN, NO_BACKEND, NO_CARDS, NO_DESCRIPTIONS, NO_OUTPUT, MISSING_ARGUMENTS };

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_model_v41_program_refusal_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected,
			  const char *what)
{
    if (status != expected) {
	    (void)printf("FAILED: %s: %s: %s, expected %s\n", context_text,
			 what, status_text(status), status_text(expected));
	    failures++;
    }
}

static void capture_begin(void)
{
    expect(fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is pointed at the temporary file");
}

/* What was logged since capture_begin is one error line that holds words. */
static void capture_end_expecting(const char *words, const char *what)
{
    char text[CAPTURE_CAPACITY];
    const bool is_flushed = fflush(stderr) == 0;
    const ssize_t byte_count = is_flushed ? pread(capture_descriptor, text, sizeof(text) - 1, 0) : -1;
    expect(dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: %s: expected one error line with \"%s\", got \"%s\"\n", context_text, what, words,
                     text);
        failures++;
    }
}

/* The plan of 1.3: layer l on card l / 8, the embedding on card 0, the head on card 4. */
static struct model_plan_record *plan_of_1_3(const struct pack_record *pack)
{
    static uint32_t card_of_unit[TEST_MODEL_V41_UNIT_COUNT];
    struct load_layout *layout = NULL;
    struct model_plan_record *plan = NULL;
    const struct model_limits limits = { CONTEXT,
					 ROWS,
					 SESSIONS,
					 STATE_WIDTH_16,
					 LINEAR_PROFILE_PRECISION,
					 LINEAR_ORDER_FIXED,
					 CMP_ORDER_REFERENCE,
					 0,
					 0,
					 0 };
    test_model_v41_five_cards(card_of_unit, 0, 0);
    expect_status(load_layout_create(pack, card_of_unit,
				     TEST_MODEL_V41_UNIT_COUNT, CARDS, &layout),
		  STATUS_OK, "the placement is laid out");
    expect_status(layout == NULL ?
			  STATUS_INVALID_ARGUMENT :
			  v41_plan_create(pack, layout, &limits, &plan),
		  STATUS_OK, "the plan is made");
    load_layout_destroy(layout);
    return plan;
}

/* A refusal of a program of that kind: that status, nothing made, and one error line with words. */
static void
expect_kind_refused(const struct model_operations *model,
		    const struct model_plan_record *plan,
		    struct device_card *const *cards,
		    const struct card_description_record *descriptions,
		    enum program_kind kind, uint32_t row_count,
		    result_status expected, const char *words, const char *what)
{
	struct execution_program *const untouched =
		(struct execution_program *)(uintptr_t)UINT64_MAX;
	struct execution_program *program = untouched;
	context_text = what;
	capture_begin();
	expect_status(model->program_prepare(plan, &test_host_memory_backend,
					     cards, descriptions, kind,
					     row_count, &program),
		      expected, "refused");
	capture_end_expecting(words, "one error line");
	expect(program == untouched,
	       "a refused program_prepare leaves its output untouched");
	if (program != untouched) {
		model->program_destroy(program);
	}
}

/* A refusal of a thin program. */
static void expect_refused(const struct model_operations *model,
			   const struct model_plan_record *plan,
			   struct device_card *const *cards,
			   const struct card_description_record *descriptions,
			   uint32_t row_count, result_status expected,
			   const char *words, const char *what)
{
	expect_kind_refused(model, plan, cards, descriptions, PROGRAM_THIN,
			    row_count, expected, words, what);
}

/* broken, a copy of plan for one change. */
static struct model_plan_record *copy_of(struct model_plan_record *broken,
					 const struct model_plan_record *plan)
{
    memcpy(broken, plan, sizeof(*broken));
    return broken;
}

static void test_refusals(const struct model_operations *model,
			  const struct pack_record *pack,
			  struct device_card **cards,
			  struct card_description_record *descriptions)
{
	struct model_plan_record *const plan = plan_of_1_3(pack);
	struct model_plan_record *const broken = calloc(1, sizeof(*broken));
	if (plan == NULL || broken == NULL) {
		expect(false, "a plan and its copy are made");
		free(broken);
		model->plan_destroy(plan);
		return;
	}
    context_text = "a missing argument";
    for (uint32_t missing = 0; missing < MISSING_ARGUMENTS; missing++) {
	    struct execution_program *program = NULL;
	    capture_begin();
	    expect_status(
		    model->program_prepare(
			    missing == NO_PLAN ? NULL : plan,
			    missing == NO_BACKEND ? NULL :
						    &test_host_memory_backend,
			    missing == NO_CARDS ? NULL : cards,
			    missing == NO_DESCRIPTIONS ? NULL : descriptions,
			    PROGRAM_THIN, 1,
			    missing == NO_OUTPUT ? NULL : &program),
		    STATUS_INVALID_ARGUMENT, "refused");
	    capture_end_expecting("a missing argument", "one error line");
	    expect(program == NULL, "no program from a refusal");
    }
    expect_refused(model, plan, cards, descriptions, 0, STATUS_INVALID_ARGUMENT,
		   "0 rows", "no rows");
    expect_refused(model, plan, cards, descriptions, ROWS + 1u,
		   STATUS_INVALID_ARGUMENT, "9 rows",
		   "more rows than the plan's");
    expect_kind_refused(model, plan, cards, descriptions,
			(enum program_kind)PROGRAM_KIND_COUNT, 1,
			STATUS_INVALID_ARGUMENT, "or kind 5",
			"a kind outside the enumeration");
    expect_kind_refused(
	    model, plan, cards, descriptions, PROGRAM_WIDE, ROWS,
	    STATUS_INVALID_ARGUMENT, "a chunked one at least 64",
	    "a wide program of fewer rows than a wide step's least");
    expect_kind_refused(
	    model, plan, cards, descriptions, PROGRAM_CONE, ROWS,
	    STATUS_UNSUPPORTED,
	    "kind 2: this build makes thin, wide and full programs alone",
	    "a cone program");
    expect_kind_refused(
	    model, plan, cards, descriptions, PROGRAM_FULL, ROWS,
	    STATUS_INVALID_ARGUMENT, "a chunked one at least 64",
	    "a full program of fewer rows than a wide step's least");
    struct device_card *const kept = cards[REFUSED_CARD];
    cards[REFUSED_CARD] = NULL;
    expect_refused(model, plan, cards, descriptions, 1, STATUS_INVALID_ARGUMENT,
		   "card 3 of 5 runs part",
		   "a card that runs layers is not given");
    cards[REFUSED_CARD] = kept;
    test_host_memory_backend_set_arithmetic_class(REFUSED_CARD, 0);
    expect_refused(model, plan, cards, descriptions, 1, STATUS_UNSUPPORTED,
		   "not prepared", "a card with no build of the kernels");
    test_host_memory_backend_set_arithmetic_class(REFUSED_CARD,
						  ARITHMETIC_CLASS_SM_80);
    descriptions[REFUSED_CARD].arithmetic_class = UNKNOWN_CLASS;
    expect_refused(model, plan, cards, descriptions, 1, STATUS_INVALID_ARGUMENT,
		   "no row for family",
		   "a class round 2's table has no slice count for");
    descriptions[REFUSED_CARD].arithmetic_class = ARITHMETIC_CLASS_SM_80;
    copy_of(broken, plan)->cards[1].array_offset[V41_DATA_ARRAY_QUERIES] =
	    PROGRAM_NOWHERE;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, "the queries",
		   "a plan that gives a card no room for an array");
    copy_of(broken, plan)->layer_card[REFUSED_LAYER] = CARDS;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, OFF_THE_PLAN,
		   "a plan that runs a layer on a card it does not have");
    copy_of(broken, plan)->layer_card[REFUSED_LAYER] = PLACEMENT_IN_HOST_MEMORY;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, OFF_THE_PLAN,
		   "a plan that runs a layer in host memory");
    copy_of(broken, plan)->embedding_card = PLACEMENT_IN_HOST_MEMORY;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, OFF_THE_PLAN,
		   "a plan that embeds in host memory");
    copy_of(broken, plan)->result_card = CARDS;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, OFF_THE_PLAN,
		   "a plan that samples on a card it does not have");
    copy_of(broken, plan)->layer_card[REFUSED_LAYER] = 0;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, OFF_THE_PLAN,
		   "a plan that runs a layer on a card whose layers lack it");
    copy_of(broken, plan)->cards[0].layers |= 1ull << REFUSED_LAYER;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, OFF_THE_PLAN,
		   "a plan that gives one layer to two cards");
    copy_of(broken, plan)->cards[1].layers |= 1ull << BEYOND_THE_LAYERS;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, OFF_THE_PLAN,
		   "a plan that gives a card a layer the model has not");
    copy_of(broken, plan)->cards[ENGRAM_CARD].engram_tables = 0;
    expect_refused(
	    model, broken, cards, descriptions, 1, STATUS_INVALID_ARGUMENT,
	    OFF_THE_PLAN,
	    "a plan whose card of layer 14 does not read its n-gram table");
    copy_of(broken, plan)->cards[0].engram_tables |= 1u << LATER_TABLE;
    expect_refused(
	    model, broken, cards, descriptions, 1, STATUS_INVALID_ARGUMENT,
	    OFF_THE_PLAN,
	    "a plan whose card of layer 1 reads the n-gram table of layer 14 too");
    copy_of(broken, plan)->cards[1].expert_table_offset[REFUSED_LAYER] = plan->cards[1].state_offset +
                                                                         plan->cards[1].image_byte_count;
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, "no expert table of layer 8",
		   "an expert table outside the card's image");
    copy_of(broken, plan)->cards[1].expert_table_offset[REFUSED_LAYER] += sizeof(uint32_t);
    expect_refused(model, broken, cards, descriptions, 1,
		   STATUS_INVALID_ARGUMENT, "no expert table of layer 8",
		   "an expert table off a word of the card's image");
#if CHECKING_HOOKS_ENABLED
    for (uint32_t call = 1; call <= MEMORY_FAULTS; call++) {
	    expect(fault_arm("model_v41_program_memory", call),
		   "the fault of a program's memory is armed");
	    expect_refused(model, plan, cards, descriptions, 1,
			   STATUS_OUT_OF_MEMORY, "no memory",
			   call == 1 ? "no memory for the builder" :
				       "no memory for the program");
	    expect(fault_disarm() == call,
		   "the fault point of a program's memory was reached");
    }
#endif
    model->program_destroy(NULL);
    free(broken);                                 /* its images are the plan's */
    model->plan_destroy(plan);
}

static void test_table(void)
{
    static const char other[] = "DeepSeek-V4.1-Lite", longer[] = "DeepSeek-V4.1-FlashX";
    char *const unterminated = malloc(MODEL_NAME_CAPACITY);
    const struct model_operations *const model =
	    model_operations_of(V41_DATA_MODEL_NAME);
    context_text = "the table of the seam";
    expect(model != NULL && strcmp(model->name, V41_DATA_MODEL_NAME) == 0 &&
		   model->describe == v41_describe &&
		   model->state_byte_count == v41_state_byte_count &&
		   model->plan_create == v41_plan_create &&
		   model->plan_destroy == v41_plan_destroy &&
		   model->arena_byte_count == v41_arena_byte_count &&
		   model->arena_initialise == v41_arena_initialise &&
		   model->program_prepare != NULL &&
		   model->program_destroy != NULL,
	   "the model's name finds its table, of model_v41.h's functions and the programs'");
    if (unterminated != NULL) {
	    memset(unterminated, 'D', MODEL_NAME_CAPACITY);
    }
    expect(model_operations_of(NULL) == NULL &&
		   model_operations_of(other) == NULL &&
		   model_operations_of(longer) == NULL &&
		   (unterminated == NULL ||
		    model_operations_of(unterminated) == NULL),
	   "no name, another name, a longer one and one without its NUL in 64 bytes find no table");
    free(unterminated);
}

int main(void)
{
	struct device_card *cards[CARDS] = { NULL };
	struct card_description_record descriptions[CARDS];
	struct pack_record *pack = NULL;
	error_stream_descriptor = dup(STDERR_FILENO);
	capture_descriptor = mkstemp(capture_path);
	if (capture_descriptor >= 0 &&
	    unlink(capture_path) !=
		    0) { /* nothing left behind, however it ends */
		(void)close(capture_descriptor);
		capture_descriptor = -1;
	}
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file could not be made\n");
        return EXIT_FAILURE;
    }
    test_table();
    const struct model_operations *const model =
	    model_operations_of(V41_DATA_MODEL_NAME);
    context_text = "the model's pack and the cards";
    expect_status(test_model_v41_pack_open(
			  (struct test_model_v41_change){
				  TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			  &pack),
		  STATUS_OK, "the model's pack opens from its tables");
    test_host_memory_backend_reset(CARDS, HOST_CARD_BYTES);
    for (uint32_t card = 0; card < CARDS; card++) {
	    test_host_memory_backend_set_arithmetic_class(
		    card, card == PRO_CARD ? ARITHMETIC_CLASS_SM_120 :
					     ARITHMETIC_CLASS_SM_80);
	    expect_status(test_host_memory_backend.card_describe(
				  card, &descriptions[card]),
			  STATUS_OK, "a card is described");
	    expect_status(test_host_memory_backend.card_open(
				  card, HOST_CARD_BYTES, &cards[card]),
			  STATUS_OK, "a card is opened");
    }
    if (pack != NULL && model != NULL && failures == 0) {
        test_refusals(model, pack, cards, descriptions);
    }
    for (uint32_t card = 0; card < CARDS; card++) {
        test_host_memory_backend.card_close(cards[card], 0);
    }
    pack_close(pack);
    test_memory_storage_reset();
    (void)close(capture_descriptor);
    if (failures != 0) {
        (void)printf("model_v41_program_refusal_test FAILED: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("model_v41_program_refusal_test passed\n");
    return EXIT_SUCCESS;
}
