/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_placement_test: the placement anemone generate makes of a DeepSeek-V4.1-Flash pack when it is given none,
 * on cards of a stated free memory, and whether it keeps every routed expert on a card; beside it the teacher-forced
 * card test's placement of all 40 layers (subset_placement.h). Host only: no card is opened or asked; the programs
 * are prepared on the backend in host memory, whose arenas only name offsets.
 *
 *     model_v41_placement_test --card_free_mib=64507,64507,64507,64507,96608 [--card_memory_kept_free_mib=1024]
 *         [--rows=1] [--wide_rows=0] [--context_tokens=16384] [--session_slots=1] [--linear_order=0]
 *         [--draft_positions=0] [--layout=<layout text>] <pack copy> [<pack copy> ...]
 *
 *   - --card_free_mib: each card's free memory as the CUDA backend finds it, in MiB, in the order the stages fill
 *     them (main.c's --card_order); the last card is the RTX PRO 6000 (sm_120), every other a CMP 170HX (sm_80), as
 *     the batches order them. --card_memory_kept_free_mib, --rows, --wide_rows, --context_tokens, --session_slots and
 *     --linear_order are main.c's (the kept free memory also the teacher-forced card test's, which holds its state);
 *     --draft_positions is the plan's draft (model_seam.h: MODEL_DRAFT_BLOCK, or 0 for none).
 *   - As main.c places: a card's budget is its free memory less the backend's reserve and what is kept free, then
 *     the passes of placement_fill_in_order, each in the budgets less the state the pass before gives each
 *     card, for main.c's limits (the context, the rows of the wider of --rows and --wide_rows, the sessions, 16-bit
 *     state, the precision profile, the order, the draft). The reserve and the count of passes are the constants
 *     backend_cuda.c and main.c use
 *     (BACKEND_CUDA_MEMORY_RESERVED_BYTES, PLACEMENT_STATE_PASSES); the loop is main.c's, copied.
 *     The teacher-forced placement is the card test's: every layer, each card's room its free memory less the
 *     reserve, its budget the room less what is kept free.
 *   - --layout: the placement a layout text states (placement.h) in place of the fill, its budgets checked by the same
 *     passes (a text places the same units in each, so the later passes check it against the budgets less the state).
 *   - Printed: per card its free memory, budget, weights, state, arena and what is left, its layers; host memory and
 *     the routed experts in it; the on-demand units each stage unit keeps (the fill's); the segments and expert pauses
 *     of the program of one row and, above one, of --rows rows; per card the teacher-forced arena against its room; a
 *     VERDICT.
 *   - Exit 0 only where every routed expert is on a card (no on-demand unit in host memory, no page-locked expert
 *     bytes, no layer of the plan with experts in host memory, no expert pause or paused segment in either program,
 *     every arena within its budget) and the teacher-forced placement of every layer fits; with --layout, where the
 *     text's placement, its plan and both programs are made, every arena is within its budget and at most
 *     STEP_MOST_PINNED_EXPERT_BYTES of routed experts stay in host memory, as main.c refuses more. Else 1,
 *     and 2 with the usage for a command line it does not describe.
 *   - Without arguments, as `make check` runs every test, the same on the pack of the model's tables
 *     (model_v41_pack.h) with five cards of the free memory batch p2p_gate measured (2026-10-18): every expert on a
 *     card; and with four of them, where experts stay in host memory, both programs pause and the 40 layers do not
 *     fit: the check must say so. Then the layout texts of wide_prefill.md (that pack has neither vision nor draft
 *     head, so they name neither): L5 on the five cards, every layer, the embedding and the head on the card of 5.1's
 *     table and every expert on a card; L4 on the four, the layers of 6.3 and 192 experts of each of the PRO's decoder
 *     layers (20-23, 36-39) in host memory, none of any other; L4 keeping none of the PRO's 16 layers' experts, every
 *     arena in budget but more experts in host memory than may be page-locked: the text must not fit; the text of the
 *     fill's cuts, the fill's placement unit for unit. Then the draft (dspark_v1/DESIGN.md 9.1) on the pack with a
 *     draft head, under production's budgets and limits (four sessions of 1,048,576 tokens, 1,024 rows, the reference
 *     order, no memory kept free): the fill in card order with the PRO last places layers 37 to 39, the head and the
 *     draft head on the PRO, every expert on a card and every arena in budget, unit for unit as without the draft;
 *     layout D (the PRO third, the draft head on the first card, the head on the last) is refused with the draft, in
 *     one line that names --layout and --card_order, and fits without it. Then the refusals of the options.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/loader.h"
#include "log_expectation.h"
#include "anemone/pack.h"
#include "anemone/placement.h"
#include "anemone/step.h"
#include "anemone/storage_seam.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"
#include "subset_placement.h"

#define DEFAULT_KEPT_FREE_MIB 1024u                 /* first_token_plan.md 1.3, as the batches keep */
#define MOST_MIB 1048576u                           /* a TiB: above any card */
#define BYTES_PER_MIB 1048576ull
#define BYTES_PER_GIB 1073741824.0
#define PROGRAMS 2u                                 /* of one row, and of --rows rows */
#define HOST_CARD_BYTES 4096u                       /* an arena no launch reads: the programs only name offsets */
#define DECIMAL_BASE 10
#define MEASURED_CMP_FREE_MIB 64507u                /* batch p2p_gate: the least of the three CMP 170HX */
#define MEASURED_PRO_FREE_MIB 96608u
#define FIVE_CARDS 5u
#define FOUR_CARDS 4u
#define EVERY_LAYER ((1ull << MODEL_LAYERS) - 1u)
#define CAPTURE_CAPACITY 4096u
#define MOST_ARGUMENTS 12u                          /* of a refusal case, its program name included */
#define L4_PRO_DECODER_EXPERTS_ON_CARD 192u         /* wide_prefill.md 6.3: of each of the PRO's decoder layers */
#define L4_HOST_EXPERT_LAYERS 0xF000F00000ull       /* bit l: layers 20-23 and 36-39, the PRO's decoder layers */
#define LAYOUT_EMBEDDING_CARD 0u                    /* of L5 and L4 alike */
#define LAYOUT_RESULT_CARD 3u
#define PRODUCTION_CMP_FREE_MIB 64512u              /* a CMP 170HX's free memory: production's budget 62.75 GiB */
#define PRODUCTION_PRO_FREE_MIB 96608u              /* the PRO's: 94.09 GiB (PLAN_0.2_ALPHA.md, 01:15) */
#define PRODUCTION_ROWS 8u                          /* --rows */
#define PRODUCTION_WIDE_ROWS 1024u                  /* --wide_rows */
#define PRODUCTION_SESSIONS 4u
#define LAYOUT_D_PRO_CARD 2u                        /* --card_order=1,2,0,3,4: the PRO third */
#define MOST_SESSIONS 64u                           /* of --session_slots: far above a serving engine's */
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };

struct options {
	uint64_t free_bytes[BACKEND_MAXIMUM_CARDS];
	uint32_t card_count;
	uint32_t kept_free_mib;
	uint32_t rows;
	uint32_t wide_rows; /* 0: none; the plan's row capacity is the wider */
	uint32_t context_tokens;
	uint32_t sessions;
	uint32_t order; /* enum linear_order_code */
	uint32_t draft_positions;
	uint32_t copy_count;
	const char *const *copy_paths;
	const char *layout; /* NULL: the fill */
};

/* The layout texts of wide_prefill.md 5.1 (L5) and 6.3 (L4) without vision and draft head, and the fill's five-card
 * cuts (first_token_plan.md 1.3); the card of each layer as those tables give it, the embedding's and the head's. */
static const char *const l5_text = "embedding: card 0; layers 0..1, 22..27: card 0; layers 2..4, 28..31: card 1; "
                                   "layers 5..7, 32..35: card 2; layers 8..10, 36..39: card 3; head: card 3; "
                                   "layers 11..21: card 4";
static const char *const l4_text = "embedding: card 0; layers 0..3, 24..27: card 0; layers 4..7, 28..31: card 1; "
                                   "layers 8..11, 32..35: card 2; layers 12..23, 36..39: card 3; head: card 3; "
                                   "experts kept: layers 20..23, 36..39: 192";
static const char *const l4_none_kept_text = "embedding: card 0; layers 0..3, 24..27: card 0; layers 4..7, 28..31: "
                                             "card 1; layers 8..11, 32..35: card 2; layers 12..23, 36..39: card 3; "
                                             "head: card 3; experts kept: layers 12..23, 36..39: 0";
static const char *const fill_text = "embedding: card 0; layers 0..7: card 0; layers 8..15: card 1; "
                                     "layers 16..23: card 2; layers 24..31: card 3; layers 32..39: card 4; "
                                     "head: card 4";
/* Production's layout D (PLAN_0.2_ALPHA.md, 01:26) without vision, which the pack of the tables lacks. */
static const char *const layout_d_text = "embedding: card 0; layers 0..5: card 0; draft head: card 0; layers 6..11: "
                                         "card 1; layers 12..23: card 2; layers 24..31: card 3; layers 32..39: "
                                         "card 4; head: card 4";
static const uint32_t l5_layer_cards[MODEL_LAYERS] = {
	0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
};
static const uint32_t l4_layer_cards[MODEL_LAYERS] = {
	0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
};

/* What the check found. */
struct findings {
	uint64_t budget
		[BACKEND_MAXIMUM_CARDS]; /* free memory less the reserve and what is kept free */
	uint64_t room[BACKEND_MAXIMUM_CARDS]; /* free memory less the reserve */
	uint64_t arena[BACKEND_MAXIMUM_CARDS];
	uint64_t pinned_bytes; /* state_byte_count of host memory: the host experts */
	uint32_t kept_units; /* on-demand units each stage unit keeps on its card */
	uint32_t most_owned_units; /* the most one stage unit owns */
	uint32_t host_on_demand_units;
	uint32_t paused_layers; /* layers of the plan with experts in host memory */
	uint32_t pauses; /* expert pauses and paused segments of both programs */
	uint32_t over_budget; /* cards whose arena passes their budget */
	uint32_t teacher_forced_over_room;
	uint32_t layer_card[MODEL_LAYERS]; /* the plan's */
	uint32_t host_experts
		[MODEL_LAYERS]; /* routed experts the plan keeps in host memory */
	uint32_t embedding_card, result_card;
	uint32_t draft_card; /* of the draft's stages; the result card's holds it */
	bool holds_draft;
	bool is_placed; /* the placement, its plan and both programs were made */
	bool is_teacher_forced_placed;
};

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_model_v41_placement_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is captured");
}

/* One error line was logged since capture_begin, and it holds words. */
static void capture_end_expecting(const char *words)
{
    char text[CAPTURE_CAPACITY];
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, sizeof(text) - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the error stream is restored");
    text[byte_count > 0 ? (size_t)byte_count : 0] = '\0';
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: expected one error line with \"%s\", got \"%s\"\n", context_text, words, text);
        failures++;
    }
}

static double gib_of(uint64_t byte_count)
{
    return (double)byte_count / BYTES_PER_GIB;
}

/* A decimal number from least to most at text, a digit first (strtoull alone would take blanks and a sign). */
static bool number_read(const char *text, uint32_t least, uint32_t most, const char **end_out, uint32_t *value_out)
{
    char *end = NULL;
    if (*text < '0' || *text > '9') {
        return false;
    }
    errno = 0;
    const unsigned long long number = strtoull(text, &end, DECIMAL_BASE);
    if (errno != 0 || number < least || number > most) {
        return false;
    }
    *end_out = end;
    *value_out = (uint32_t)number;
    return true;
}

/* The option of this name with one number, given at most once; false for any other argument. */
static bool option_take(const char *argument, const char *name, uint32_t least, uint32_t most, bool *is_given,
                        uint32_t *value_out)
{
    const char *end = NULL;
    const size_t length = strlen(name);
    if (*is_given || strncmp(argument, name, length) != 0 ||
        !number_read(argument + length, least, most, &end, value_out) || *end != '\0') {
        return false;
    }
    *is_given = true;
    return true;
}

/* "a,b,c": every card's free memory, 1 MiB or more each, at most BACKEND_MAXIMUM_CARDS of them. */
static bool card_list_take(const char *argument, struct options *options)
{
    const char *const name = "--card_free_mib=";
    const size_t length = strlen(name);
    uint32_t count = 0;
    if (options->card_count != 0 || strncmp(argument, name, length) != 0) {
        return false;
    }
    const char *text = argument + length;                /* past the name, now known to be there */
    while (count < BACKEND_MAXIMUM_CARDS) {
	    const char *end = NULL;
	    uint32_t mib = 0;
	    if (!number_read(text, 1, MOST_MIB, &end, &mib)) {
		    return false;
	    }
	    options->free_bytes[count++] = (uint64_t)mib * BYTES_PER_MIB;
	    if (*end != ',') {
		    options->card_count = *end == '\0' ? count : 0;
		    return *end == '\0';
	    }
	    text = end + 1;
    }
    return false;
}

/* "--layout=<text>", the text not empty, given at most once. */
static bool layout_take(const char *argument, struct options *options)
{
    const char *const name = "--layout=";
    const size_t length = strlen(name);
    if (options->layout != NULL || strncmp(argument, name, length) != 0 || argument[length] == '\0') {
        return false;
    }
    options->layout = argument + length;
    return true;
}

/* The options of a command line that names only the cards and the pack. */
static struct options options_default(void)
{
	return (struct options){ .kept_free_mib = DEFAULT_KEPT_FREE_MIB,
				 .rows = 1,
				 .context_tokens =
					 MODEL_CONTEXT_WITHOUT_CANDIDATES,
				 .sessions = 1,
				 .order = LINEAR_ORDER_FIXED };
}

/* False for a command line the usage does not describe; options is then unspecified. */
static bool options_parse(int argument_count, const char *const *arguments, struct options *options)
{
    bool is_kept_given = false, is_rows_given = false, is_wide_given = false, is_context_given = false;
    bool is_sessions_given = false, is_order_given = false, is_draft_given = false;
    int index = 1;
    *options = options_default();
    for (; index < argument_count && strncmp(arguments[index], "--", strlen("--")) == 0; index++) {
        const char *const argument = arguments[index];
	if (!card_list_take(argument, options) &&
	    !layout_take(argument, options) &&
	    !option_take(argument, "--card_memory_kept_free_mib=", 0, MOST_MIB,
			 &is_kept_given, &options->kept_free_mib) &&
	    !option_take(argument, "--rows=", 1, LINEAR_MAXIMUM_ROWS,
			 &is_rows_given, &options->rows) &&
	    !option_take(argument, "--wide_rows=", 0, MODEL_WIDE_MOST_ROWS,
			 &is_wide_given, &options->wide_rows) &&
	    !option_take(argument, "--context_tokens=", 1, MODEL_CONTEXT_MOST,
			 &is_context_given, &options->context_tokens) &&
	    !option_take(argument, "--session_slots=", 1, MOST_SESSIONS,
			 &is_sessions_given, &options->sessions) &&
	    !option_take(argument, "--linear_order=", 0, LINEAR_ORDER_REFERENCE,
			 &is_order_given, &options->order) &&
	    !option_take(argument, "--draft_positions=", 0, STEP_DRAFT_MOST,
			 &is_draft_given, &options->draft_positions)) {
		return false;
	}
    }
    options->copy_paths = &arguments[index];
    options->copy_count = index < argument_count ? (uint32_t)(argument_count - index) : 0;
    return options->card_count != 0 && options->copy_count >= 1 &&
	   options->copy_count <= STORAGE_MAXIMUM_COPIES;
}

/* main.c's placement, or the layout text's: the budgets less what each card's state takes under the placement of the
 * pass before. */
MUST_CHECK static result_status default_place(const struct pack_record *pack,
					      const struct model_limits *limits,
					      const struct options *options,
					      struct findings *found,
					      uint32_t *card_of_unit)
{
	const uint32_t unit_count = pack_header_of(pack)->unit_count;
	const struct placement_record placement = { card_of_unit, unit_count,
						    options->card_count };
	const uint64_t reserved = BACKEND_CUDA_MEMORY_RESERVED_BYTES;
	const uint64_t kept = (uint64_t)options->kept_free_mib * BYTES_PER_MIB;
	uint64_t weight_budget[BACKEND_MAXIMUM_CARDS] = { 0 };
	result_status status = STATUS_OK;
	for (uint32_t card = 0; card < options->card_count; card++) {
		found->room[card] =
			options->free_bytes[card] > reserved ?
				options->free_bytes[card] - reserved :
				0;
		found->budget[card] =
			found->room[card] > kept ? found->room[card] - kept : 0;
		weight_budget[card] = found->budget[card];
	}
	for (uint32_t pass = 0;
	     pass < PLACEMENT_STATE_PASSES && status == STATUS_OK; pass++) {
		status = options->layout != NULL ?
				 placement_from_layout_text(
					 pack, options->layout, weight_budget,
					 options->card_count, unit_count,
					 card_of_unit) :
				 placement_fill_in_order(pack, weight_budget,
							 options->card_count,
							 unit_count,
							 card_of_unit,
							 &found->kept_units);
		for (uint32_t card = 0;
		     card < options->card_count && status == STATUS_OK;
		     card++) {
			uint64_t state_bytes = 0;
			status = v41_state_byte_count(pack, &placement, limits,
						      card, &state_bytes);
			weight_budget[card] =
				found->budget[card] > state_bytes ?
					found->budget[card] - state_bytes :
					0;
		}
	}
	return status == STATUS_OK ?
		       v41_state_byte_count(pack, &placement, limits,
					    PLACEMENT_IN_HOST_MEMORY,
					    &found->pinned_bytes) :
		       status;
}

/* The on-demand units: the most one stage unit owns, and those in host memory. */
static void units_count(const struct pack_record *pack,
			const uint32_t *card_of_unit, struct findings *found)
{
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	uint32_t owned = 0;
	for (uint32_t unit = 0; unit < pack_header_of(pack)->unit_count;
	     unit++) {
		const bool is_on_demand = units[unit].unit_kind ==
					  PLACEMENT_UNIT_ON_DEMAND;
		owned = units[unit].unit_kind == PLACEMENT_UNIT_STAGE ?
				0 :
				owned + (is_on_demand ? 1u : 0u);
		found->most_owned_units = owned > found->most_owned_units ?
						  owned :
						  found->most_owned_units;
		found->host_on_demand_units +=
			is_on_demand && card_of_unit[unit] ==
						PLACEMENT_IN_HOST_MEMORY ?
				1u :
				0u;
	}
}

/* The programs of one row and, above one, of --rows rows (as main.c makes them) on the backend in host memory: their
 * expert pauses and paused segments, and with is_printing their lines. */
MUST_CHECK static result_status
programs_check(const struct model_plan_record *plan,
	       const struct options *options, bool is_printing,
	       struct findings *found)
{
	const struct model_operations *const model =
		model_operations_of(V41_DATA_MODEL_NAME);
	const uint32_t row_counts[PROGRAMS] = { 1, options->rows };
	const uint32_t program_count = options->rows > 1 ? PROGRAMS : 1;
	struct device_card *cards[BACKEND_MAXIMUM_CARDS] = { NULL };
	struct card_description_record descriptions[BACKEND_MAXIMUM_CARDS];
	result_status status = model == NULL ? STATUS_UNSUPPORTED : STATUS_OK;
	test_host_memory_backend_reset(options->card_count, HOST_CARD_BYTES);
	for (uint32_t card = 0;
	     card < options->card_count && status == STATUS_OK; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, card + 1u == options->card_count ?
				      ARITHMETIC_CLASS_SM_120 :
				      ARITHMETIC_CLASS_SM_80);
		status = test_host_memory_backend.card_describe(
			card, &descriptions[card]);
		status = status == STATUS_OK ?
				 test_host_memory_backend.card_open(
					 card, HOST_CARD_BYTES, &cards[card]) :
				 status;
	}
	for (uint32_t index = 0; index < program_count && status == STATUS_OK;
	     index++) {
		struct execution_program *program = NULL;
		status = model->program_prepare(plan, &test_host_memory_backend,
						cards, descriptions,
						PROGRAM_THIN, row_counts[index],
						&program);
		uint32_t paused_segments = 0;
		for (uint32_t segment = 0;
		     program != NULL && segment < program->segment_count;
		     segment++) {
			paused_segments +=
				program->segments[segment].pause != 0 ? 1u : 0u;
		}
		if (program != NULL) {
			found->pauses += program->expert_pause_count +
					 paused_segments +
					 program->host_expert_count;
			if (is_printing) {
				(void)printf("PROGRAM rows=%" PRIu32
					     " segments=%" PRIu32
					     " expert_pauses=%" PRIu32
					     " paused_segments="
					     "%" PRIu32 " host_experts=%" PRIu32
					     " hand_offs=%" PRIu32 "\n",
					     row_counts[index],
					     program->segment_count,
					     program->expert_pause_count,
					     paused_segments,
					     program->host_expert_count,
					     program->hand_off_count);
			}
		}
		model->program_destroy(program);
	}
    for (uint32_t card = 0; card < options->card_count; card++) {
        test_host_memory_backend.card_close(cards[card], 0);
    }
    return status;
}

/* The line of a card: what it has, what the placement gives it and its layers as ranges. */
static void card_print(const struct model_plan_record *plan,
		       const struct load_layout *layout,
		       const struct options *options,
		       const struct findings *found, uint32_t card)
{
    const uint64_t arena = found->arena[card], weights = layout->card_byte_count[card];
    (void)printf("CARD index=%" PRIu32 " class=%s free_gib=%.2f budget_gib=%.2f weights_gib=%.2f state_gib=%.3f "
                 "arena_gib=%.2f left_of_budget_gib=%.2f%s%s layers=", card,
                 card + 1u == options->card_count ? "sm_120" : "sm_80", gib_of(options->free_bytes[card]),
                 gib_of(found->budget[card]), gib_of(weights), gib_of(arena > weights ? arena - weights : 0),
                 gib_of(arena), gib_of(found->budget[card]) - gib_of(arena),
                 plan->embedding_card == card ? " embedding=yes" : "", plan->result_card == card ? " head=yes" : "");
    const char *separator = "";
    for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
	    const bool is_here = plan->layer_card[layer] == card;
	    const bool is_first =
		    is_here &&
		    (layer == 0 || plan->layer_card[layer - 1u] != card);
	    const bool is_last = is_here &&
				 (layer + 1u == MODEL_LAYERS ||
				  plan->layer_card[layer + 1u] != card);
	    if (is_first) {
		    (void)printf("%s%" PRIu32, separator, layer);
		    separator = ",";
	    }
	    if (is_last && !is_first) {
		    (void)printf("-%" PRIu32, layer);
	    }
    }
    (void)printf("%s\n", *separator == '\0' ? "none" : "");
}

/* The teacher-forced card test's placement of every layer: each card's arena against its room. */
MUST_CHECK static result_status
teacher_forced_check(const struct pack_record *pack,
		     const struct options *options, bool is_printing,
		     struct findings *found)
{
	const struct model_limits limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES,
					     1,
					     1,
					     STATE_WIDTH_16,
					     LINEAR_PROFILE_PRECISION,
					     LINEAR_ORDER_FIXED,
					     CMP_ORDER_REFERENCE,
					     0,
					     0,
					     0 };
	struct test_subset_placement subset = { .card_of_unit = NULL };
	struct load_layout *layout = NULL;
	struct model_plan_record *plan = NULL;
	result_status status = test_subset_place(
		pack, found->budget, options->card_count,
		test_subset_layers_with_sources(EVERY_LAYER), &subset);
	status = status != STATUS_OK ?
			 status :
			 load_layout_create(pack, subset.card_of_unit,
					    pack_header_of(pack)->unit_count,
					    options->card_count, &layout);
	status = status != STATUS_OK ?
			 status :
			 test_subset_plan_layout(pack, layout, &subset);
	status = status != STATUS_OK ?
			 status :
			 v41_plan_create(pack, &subset.plan_layout, &limits,
					 &plan);
	for (uint32_t card = 0;
	     card < options->card_count && status == STATUS_OK; card++) {
		uint64_t arena = 0;
		status = v41_arena_byte_count(plan, card, &arena);
		found->teacher_forced_over_room +=
			status == STATUS_OK && arena > found->room[card] ? 1u :
									   0u;
		if (status == STATUS_OK && is_printing) {
			(void)printf(
				"TEACHER_FORCED card=%" PRIu32
				" layers=0-%u arena_gib=%.2f room_gib=%.2f fits=%s\n",
				card, MODEL_LAYERS - 1u, gib_of(arena),
				gib_of(found->room[card]),
				arena <= found->room[card] ? "yes" : "no");
		}
	}
	found->is_teacher_forced_placed = status == STATUS_OK;
	v41_plan_destroy(plan);
	load_layout_destroy(layout);
	test_subset_placement_release(&subset);
	return status;
}

/* The whole check of one pack and one set of cards; with is_printing its lines (the header comment). */
MUST_CHECK static result_status placement_check(const struct pack_record *pack,
						const struct options *options,
						bool is_printing,
						struct findings *found)
{
	const struct model_limits limits = {
		.context_token_limit = options->context_tokens,
		.row_capacity = options->wide_rows > options->rows ?
					options->wide_rows :
					options->rows,
		.session_capacity = options->sessions,
		.state_width = STATE_WIDTH_16,
		.linear_profile = LINEAR_PROFILE_PRECISION,
		.linear_order = (enum linear_order_code)options->order,
		.cmp_order = CMP_ORDER_REFERENCE,
		.draft_positions = options->draft_positions
	};
	const uint32_t unit_count = pack_header_of(pack)->unit_count;
	uint32_t *const card_of_unit =
		calloc(unit_count, sizeof(*card_of_unit));
	struct load_layout *layout = NULL;
	struct model_plan_record *plan = NULL;
	*found = (struct findings){ .pinned_bytes = 0 };
	result_status status = card_of_unit == NULL ?
				       STATUS_OUT_OF_MEMORY :
				       default_place(pack, &limits, options,
						     found, card_of_unit);
	status = status != STATUS_OK ?
			 status :
			 load_layout_create(pack, card_of_unit, unit_count,
					    options->card_count, &layout);
	status = status != STATUS_OK ?
			 status :
			 v41_plan_create(pack, layout, &limits, &plan);
	for (uint32_t card = 0;
	     card < options->card_count && status == STATUS_OK; card++) {
		status = v41_arena_byte_count(plan, card, &found->arena[card]);
		found->over_budget +=
			status == STATUS_OK && found->arena[card] >
						       found->budget[card] ?
				1u :
				0u;
	}
	for (uint32_t layer = 0; layer < MODEL_LAYERS && status == STATUS_OK;
	     layer++) {
		found->paused_layers +=
			plan->host_expert_count[layer] != 0 ? 1u : 0u;
		found->layer_card[layer] = plan->layer_card[layer];
		found->host_experts[layer] = plan->host_expert_count[layer];
	}
	found->embedding_card = status == STATUS_OK ? plan->embedding_card :
						      found->embedding_card;
	found->result_card = status == STATUS_OK ? plan->result_card :
						   found->result_card;
	found->draft_card = status == STATUS_OK ?
				    plan->layer_card[MODEL_DRAFT_FIRST_LAYER] :
				    0;
	found->holds_draft =
		status == STATUS_OK &&
		(plan->cards[plan->result_card].holders >> V41_DRAFT & 1u) != 0;
	if (status == STATUS_OK) {
		units_count(pack, card_of_unit, found);
		for (uint32_t card = 0;
		     card < options->card_count && is_printing; card++) {
			card_print(plan, layout, options, found, card);
		}
		if (is_printing) {
			(void)printf(
				"HOST bytes_gib=%.2f on_demand_units_in_host=%" PRIu32
				" page_locked_expert_gib=%.2f "
				"layers_with_host_experts=%" PRIu32
				"\nPLACEMENT cards=%" PRIu32
				" kept_free_mib=%" PRIu32 " rows=%" PRIu32
				" on_demand_units_kept_per_stage_unit=%" PRIu32
				" of=%" PRIu32
				" placed_by=%s row_capacity=%" PRIu32
				" sessions=%" PRIu32 " context=%" PRIu32
				" draft_positions=%" PRIu32 "%s\n",
				gib_of(layout->host_byte_count),
				found->host_on_demand_units,
				gib_of(found->pinned_bytes),
				found->paused_layers, options->card_count,
				options->kept_free_mib, options->rows,
				found->kept_units, found->most_owned_units,
				options->layout != NULL ? "text" : "fill",
				limits.row_capacity, limits.session_capacity,
				limits.context_token_limit,
				limits.draft_positions,
				found->holds_draft ?
					" draft=on_the_result_card" :
					"");
		}
		status = programs_check(plan, options, is_printing, found);
	}
	found->is_placed = status == STATUS_OK;
	v41_plan_destroy(plan);
	load_layout_destroy(layout);
	free(card_of_unit);
	return status == STATUS_OK ?
		       teacher_forced_check(pack, options, is_printing, found) :
		       status;
}

static bool every_expert_is_on_a_card(const struct findings *found)
{
    return found->is_placed && found->host_on_demand_units == 0 && found->pinned_bytes == 0 &&
           found->paused_layers == 0 && found->pauses == 0 && found->over_budget == 0;
}

static bool teacher_forced_fits(const struct findings *found)
{
    return found->is_teacher_forced_placed && found->teacher_forced_over_room == 0;
}

/* A layout text's placement as generate takes it: made, in budget, and no more host experts than main.c may lock. */
static bool layout_fits(const struct findings *found)
{
	return found->is_placed && found->over_budget == 0 &&
	       found->pinned_bytes <= STEP_MOST_PINNED_EXPERT_BYTES;
}

/* With a pack: the check of the header comment on it. */
static int pack_check(const struct options *options)
{
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct findings found = { .pinned_bytes = 0 };
	result_status status = storage_direct.reader_open(
		options->copy_paths, options->copy_count, &reader);
	status = status != STATUS_OK ?
			 status :
			 pack_open_entry(&storage_direct, reader, &pack);
	status = status != STATUS_OK ?
			 status :
			 placement_check(pack, options, true, &found);
	const bool is_on_cards = status == STATUS_OK &&
				 every_expert_is_on_a_card(&found);
	const bool is_fitting = status == STATUS_OK &&
				teacher_forced_fits(&found);
	const bool is_layout_fitting = options->layout != NULL &&
				       layout_fits(&found);
	(void)printf(
		"VERDICT every routed expert on a card: %s; the teacher-forced 40 layers fit: %s%s%s (%s)\n",
		is_on_cards ? "yes" : "no", is_fitting ? "yes" : "no",
		options->layout != NULL ? "; the layout text fits: " : "",
		options->layout == NULL ? "" :
		is_layout_fitting	? "yes" :
					  "no",
		status_text(status));
	pack_close(pack);
	storage_direct.reader_close(reader);
	return (options->layout != NULL ? is_layout_fitting :
					  is_on_cards && is_fitting) ?
		       TEST_PASSED :
		       TEST_FAILED;
}

/* Without arguments: the model's tables on the measured cards, five and four. */
static void test_measured_cards(const struct pack_record *pack)
{
    struct options options = options_default();
    struct findings found;
    options.card_count = FIVE_CARDS;
    options.rows = LINEAR_MAXIMUM_ROWS;
    for (uint32_t card = 0; card < FIVE_CARDS; card++) {
        options.free_bytes[card] = (uint64_t)(card + 1u == FIVE_CARDS ? MEASURED_PRO_FREE_MIB : MEASURED_CMP_FREE_MIB) *
                                   BYTES_PER_MIB;
    }
    context_text = "five cards";
    expect(placement_check(pack, &options, false, &found) == STATUS_OK,
	   "the check runs");
    expect(every_expert_is_on_a_card(&found) &&
		   found.kept_units == MODEL_EXPERTS &&
		   found.most_owned_units == MODEL_EXPERTS,
	   "every routed expert is on a card");
    expect(teacher_forced_fits(&found), "the teacher-forced 40 layers fit");
    context_text = "four cards";
    options.card_count = FOUR_CARDS;
    options.free_bytes[FOUR_CARDS - 1u] = (uint64_t)MEASURED_PRO_FREE_MIB * BYTES_PER_MIB;
    capture_begin();
    const result_status status = placement_check(pack, &options, false, &found);
    capture_end_expecting("fit no card left");
    expect(status == STATUS_OUT_OF_MEMORY && found.is_placed &&
		   !found.is_teacher_forced_placed,
	   "the placement is made and the teacher-forced layers are refused");
    expect(!every_expert_is_on_a_card(&found) &&
		   found.host_on_demand_units != 0 && found.pinned_bytes != 0 &&
		   found.paused_layers != 0 && found.pauses != 0 &&
		   found.over_budget == 0 && found.kept_units < MODEL_EXPERTS,
	   "the check says that experts stay in host memory");
}

/* The plan of a layout text holds every layer, the embedding and the head on the cards of its table, and the routed
 * experts in host memory that the text keeps there: those of the layers of host_layers but expected_on_card. */
static void layout_expect(const struct findings *found, const uint32_t *layer_cards, uint64_t host_layers,
                          uint32_t expected_on_card)
{
    bool is_as_stated = found->embedding_card == LAYOUT_EMBEDDING_CARD && found->result_card == LAYOUT_RESULT_CARD;
    uint32_t host_units = 0;
    for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
	    const uint32_t host_experts =
		    (host_layers >> layer & 1u) != 0 ?
			    MODEL_EXPERTS - expected_on_card :
			    0;
	    is_as_stated = is_as_stated &&
			   found->layer_card[layer] == layer_cards[layer] &&
			   found->host_experts[layer] == host_experts;
	    host_units += host_experts;
    }
    expect(layout_fits(found), "the text's placement, plan and programs are made in budget, its host experts lockable");
    expect(is_as_stated, "every layer, the embedding and the head on the card the text states");
    expect(found->host_on_demand_units == host_units, "the routed experts in host memory are the text's alone");
}

/* The layout texts of wide_prefill.md on the measured cards, and the text of the fill's cuts against the fill. */
static void test_layout_texts(const struct pack_record *pack)
{
    struct options options = options_default();
    struct findings found;
    options.card_count = FIVE_CARDS;
    options.rows = LINEAR_MAXIMUM_ROWS;
    options.layout = l5_text;
    for (uint32_t card = 0; card < FIVE_CARDS; card++) {
        options.free_bytes[card] = (uint64_t)(card + 1u == FIVE_CARDS ? MEASURED_PRO_FREE_MIB : MEASURED_CMP_FREE_MIB) *
                                   BYTES_PER_MIB;
    }
    context_text = "layout L5 on five cards";
    expect(placement_check(pack, &options, false, &found) == STATUS_OK,
	   "the check runs");
    layout_expect(&found, l5_layer_cards, 0, MODEL_EXPERTS);
    expect(every_expert_is_on_a_card(&found), "every routed expert is on a card");

    context_text = "the fill's cuts as a text";
    const uint32_t unit_count = pack_header_of(pack)->unit_count;
    uint32_t *const filled = calloc(unit_count, sizeof(*filled)), *const stated = calloc(unit_count, sizeof(*stated));
    const struct model_limits limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES,
					 options.rows,
					 1,
					 STATE_WIDTH_16,
					 LINEAR_PROFILE_PRECISION,
					 LINEAR_ORDER_FIXED,
					 CMP_ORDER_REFERENCE,
					 0,
					 0,
					 0 };
    options.layout = NULL;
    result_status status =
	    filled == NULL || stated == NULL ?
		    STATUS_OUT_OF_MEMORY :
		    default_place(pack, &limits, &options, &found, filled);
    options.layout = fill_text;
    status = status != STATUS_OK ?
		     status :
		     default_place(pack, &limits, &options, &found, stated);
    expect(filled != NULL && stated != NULL && status == STATUS_OK &&
		   memcmp(filled, stated, unit_count * sizeof(*filled)) == 0,
	   "the text of the fill's cuts places every unit where the fill does");
    free(filled);
    free(stated);

    context_text = "layout L4 on four cards";
    options.card_count = FOUR_CARDS;
    options.free_bytes[FOUR_CARDS - 1u] = (uint64_t)MEASURED_PRO_FREE_MIB * BYTES_PER_MIB;
    options.layout = l4_text;
    capture_begin();
    status = placement_check(pack, &options, false, &found);
    capture_end_expecting("fit no card left");                 /* the teacher-forced 40 layers, as with the fill */
    expect(status == STATUS_OUT_OF_MEMORY && !found.is_teacher_forced_placed,
	   "the check runs");
    layout_expect(&found, l4_layer_cards, L4_HOST_EXPERT_LAYERS, L4_PRO_DECODER_EXPERTS_ON_CARD);

    context_text = "layout L4 keeping none of the PRO's experts";
    options.layout = l4_none_kept_text;
    capture_begin();
    status = placement_check(pack, &options, false, &found);
    capture_end_expecting("fit no card left");
    expect(status == STATUS_OUT_OF_MEMORY && found.is_placed &&
		   found.over_budget == 0,
	   "the check runs");
    expect(found.pinned_bytes > STEP_MOST_PINNED_EXPERT_BYTES &&
		   !layout_fits(&found),
	   "more experts in host memory than may be page-locked: the text does not fit");
}

/* Production's budgets and limits on five cards in stage order, the PRO at pro_card: a CMP 170HX's free memory and
 * the PRO's, no memory kept free, --rows 8 and --wide_rows 1,024, four sessions of the most context, the reference
 * order (main.c's default), the draft and the layout text given. */
static struct options production_options(uint32_t pro_card, uint32_t draft_positions, const char *layout)
{
    struct options options = options_default();
    options.card_count = FIVE_CARDS;
    options.kept_free_mib = 0;
    options.rows = PRODUCTION_ROWS;
    options.wide_rows = PRODUCTION_WIDE_ROWS;
    options.context_tokens = MODEL_CONTEXT_MOST;
    options.sessions = PRODUCTION_SESSIONS;
    options.order = LINEAR_ORDER_REFERENCE;
    options.draft_positions = draft_positions;
    options.layout = layout;
    for (uint32_t card = 0; card < FIVE_CARDS; card++) {
        options.free_bytes[card] =
            (uint64_t)(card == pro_card ? PRODUCTION_PRO_FREE_MIB : PRODUCTION_CMP_FREE_MIB) * BYTES_PER_MIB;
    }
    return options;
}

/* The draft on production's budgets, on the pack with a draft head (the header comment). */
static void test_draft_placement(const struct pack_record *pack)
{
    const uint32_t pro_card = FIVE_CARDS - 1u;
    struct options options =
	    production_options(pro_card, MODEL_DRAFT_BLOCK, NULL);
    struct findings found, plain;
    context_text = "the draft on production's budgets, the fill with the PRO last";
    expect(placement_check(pack, &options, false, &found) == STATUS_OK,
	   "the check runs");
    bool is_on_the_pro = found.result_card == pro_card && found.holds_draft && found.draft_card == pro_card;
    for (uint32_t layer = MODEL_DRAFT_FIRST_TAP_LAYER; layer < MODEL_LAYERS;
	 layer++) {
	    is_on_the_pro = is_on_the_pro &&
			    found.layer_card[layer] == pro_card;
    }
    expect(is_on_the_pro, "layers 37 to 39, the head and the draft head on the PRO");
    expect(every_expert_is_on_a_card(&found), "every routed expert on a card, every arena in budget");
    options.draft_positions = 0;
    expect(placement_check(pack, &options, false, &plain) == STATUS_OK &&
		   !plain.holds_draft &&
		   memcmp(plain.layer_card, found.layer_card,
			  sizeof(found.layer_card)) == 0 &&
		   plain.result_card == found.result_card &&
		   found.arena[pro_card] > plain.arena[pro_card],
	   "without the draft the fill places every layer alike, and the PRO's arena is the smaller");
    for (uint32_t card = 0; card < pro_card; card++) {
        expect(found.arena[card] == plain.arena[card], "a card without the draft keeps its arena");
    }
    context_text = "layout D with the draft";
    options = production_options(LAYOUT_D_PRO_CARD, MODEL_DRAFT_BLOCK,
				 layout_d_text);
    capture_begin();
    const result_status status = placement_check(pack, &options, false, &found);
    capture_end_expecting("the draft needs layers 37 to 39, the head and the draft head on one card, but "
                          "mtp.0.main_proj.weight lies on card 0 and the head on card 4: place them together "
                          "(--layout, or --card_order for the fill)");
    expect(status == STATUS_UNSUPPORTED && !found.is_placed,
	   "the draft is refused on layout D");
    context_text = "layout D without the draft";
    options.draft_positions = 0;
    expect(placement_check(pack, &options, false, &found) == STATUS_OK &&
		   layout_fits(&found) &&
		   found.result_card == FIVE_CARDS - 1u &&
		   found.embedding_card == 0,
	   "layout D fits without the draft");
}

/* Every refusal of the options, and the command lines it takes. */
static void test_options(void)
{
    static const struct {
        const char *arguments[MOST_ARGUMENTS];
        bool is_taken;
        const char *what;
    } cases[] = {
        {{"t", "--card_free_mib=64507,96608", "pack"}, true, "two cards and a pack"},
        {{"t", "--rows=8", "--card_memory_kept_free_mib=0", "--card_free_mib=1", "a", "b"}, true, "every option"},
        {{"t", "pack"}, false, "no cards"},
        {{"t", "--card_free_mib=64507"}, false, "no pack"},
        {{"t", "--card_free_mib=0", "pack"}, false, "a card of no memory"},
        {{"t", "--card_free_mib=1048577", "pack"}, false, "a card above a TiB"},
        {{"t", "--card_free_mib=1,,2", "pack"}, false, "an empty figure"},
        {{"t", "--card_free_mib=1,", "pack"}, false, "a comma last"},
        {{"t", "--card_free_mib=", "pack"}, false, "no figure"},
        {{"t", "--card_free_mib= 1", "pack"}, false, "a blank"},
        {{"t", "--card_free_mib=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", "pack"}, false, "17 cards"},
        {{"t", "--card_free_mib=1", "--card_free_mib=1", "pack"}, false, "the cards twice"},
        {{"t", "--card_free_mib=1", "--rows=0", "pack"}, false, "no rows"},
        {{"t", "--card_free_mib=1", "--rows=9", "pack"}, false, "more rows than a launch"},
        {{"t", "--card_free_mib=1", "--rows=1", "--rows=1", "pack"}, false, "the rows twice"},
        {{"t", "--card_free_mib=1", "--card_memory_kept_free_mib=-1", "pack"}, false, "a sign"},
        {{"t", "--card_free_mib=1", "--card_memory_kept_free_mib=1x", "pack"}, false, "a figure and a letter"},
        {{"t", "--card_free_mib=1", "--cards=1", "pack"}, false, "an unknown option"},
        {{"t", "--card_free_mib=1", "--layout=head: card 0", "pack"}, true, "a layout"},
        {{"t", "--card_free_mib=1", "--layout=", "pack"}, false, "an empty layout"},
        {{"t", "--card_free_mib=1", "--layout=a", "--layout=a", "pack"}, false, "the layout twice"},
        {{"t", "--card_free_mib=1", "--wide_rows=1024", "--context_tokens=1048576", "--session_slots=4",
          "--linear_order=1", "--draft_positions=5", "pack"}, true, "production's limits"},
        {{"t", "--card_free_mib=1", "--wide_rows=4097", "pack"}, false, "a chunk above the most"},
        {{"t", "--card_free_mib=1", "--context_tokens=0", "pack"}, false, "no context"},
        {{"t", "--card_free_mib=1", "--session_slots=0", "pack"}, false, "no session"},
        {{"t", "--card_free_mib=1", "--linear_order=2", "pack"}, false, "an order past the reference's"},
        {{"t", "--card_free_mib=1", "--draft_positions=8", "pack"}, false, "more draft positions than a round's"},
        {{"t", "--card_free_mib=1", "--draft_positions=5", "--draft_positions=5", "pack"}, false, "the draft twice"},
        {{"t", "--card_free_mib=1", "1", "2", "3", "4", "5", "6", "7", "8", "9"}, false, "nine copies"}};
    for (uint32_t index = 0; index < COUNT_OF(cases); index++) {
        int argument_count = 0;
        struct options options;
        while (argument_count < (int)MOST_ARGUMENTS && cases[index].arguments[argument_count] != NULL) {
            argument_count++;
        }
        context_text = cases[index].what;
        expect(options_parse(argument_count, cases[index].arguments, &options) == cases[index].is_taken,
               cases[index].is_taken ? "the command line is taken" : "the command line is refused");
    }
}

int main(int argument_count, char **arguments)
{
	struct pack_record *pack = NULL;
	struct options options;
	(void)setvbuf(stdout, NULL, _IOLBF,
		      0); /* a failure's line survives a sanitizer's stop */
	if (argument_count > 1) {
		if (!options_parse(argument_count,
				   (const char *const *)arguments, &options)) {
			(void)fprintf(
				stderr,
				"usage: model_v41_placement_test --card_free_mib=N,N,... [--card_memory_kept_free_mib"
				"=N] [--rows=1..%u] [--wide_rows=0..%u] [--context_tokens=N] [--session_slots=N] "
				"[--linear_order=0|1] [--draft_positions=0..%u] [--layout=<layout text>] <pack copy> "
				"[... up to %d copies]\n",
				LINEAR_MAXIMUM_ROWS, MODEL_WIDE_MOST_ROWS,
				STEP_DRAFT_MOST, STORAGE_MAXIMUM_COPIES);
			return TEST_WRONG_USAGE;
		}
		return pack_check(&options);
	}
    capture_descriptor = mkstemp(capture_path);
    error_stream_descriptor = dup(STDERR_FILENO);
    if (capture_descriptor < 0 || error_stream_descriptor < 0 || unlink(capture_path) != 0) {
        (void)printf("FAILED: the temporary file for the error stream\n");
        return TEST_FAILED;
    }
    context_text = "the model's pack";
    expect(test_model_v41_pack_open(
		   (struct test_model_v41_change){ TEST_MODEL_V41_CHANGE_NONE,
						   NULL, 0 },
		   &pack) == STATUS_OK,
	   "the model's pack opens from its tables");
    if (pack != NULL) {
        test_measured_cards(pack);
        test_layout_texts(pack);
    }
    pack_close(pack);
    pack = NULL;
    context_text = "the pack with a draft head";
    expect(test_model_v41_draft_pack_open(
		   (struct test_model_v41_change){ TEST_MODEL_V41_CHANGE_NONE,
						   NULL, 0 },
		   &pack) == STATUS_OK,
	   "the pack with a draft head opens from its tables");
    if (pack != NULL) {
        test_draft_placement(pack);
    }
    test_options();
    pack_close(pack);
    test_memory_storage_reset();
    (void)close(capture_descriptor);
    (void)close(error_stream_descriptor);
    if (failures != 0) {
        (void)printf("model_v41_placement_test FAILED: %d failures\n", failures);
        return TEST_FAILED;
    }
    (void)printf("model_v41_placement_test passed\n");
    return TEST_PASSED;
}
