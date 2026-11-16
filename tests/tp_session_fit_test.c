/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * tp_session_fit_test: serve's --session_slots=max on the TP splits (TP4_PRO, TP4_EXL3), through commands.c's own
 * parse, pack open and tensor_plan_prepare / tensor_plan_fit, host only: no card is opened (CUDA_VISIBLE_DEVICES may be
 * empty).
 * Each card's arena is split into what grows with the sessions and what does not, at N = 1..8 sessions of the
 * no-axis serve's context (1,048,576 tokens), on cards of a stated free memory.
 *
 *     tp_session_fit_test --placement=tp4_pro|tp4_exl3 --card_free_mib=N,N,... <tables> <pack or EXL3 map>
 *         [--knob=value ...]
 *
 *   - The command is the split's no-axis serve (its all-balanced row: the base, 1M context, --session_slots=max),
 *     with the given knobs (at most FIT_MOST_KNOBS) beside it, for example --tp_attention_owner=1.
 *   - Per card: its free memory and budget (less the backend's reserve and the kept-free memory); the weights and slots
 *     (the arena before its first array); the row arrays; the per-session attention state (rings, checkpoints,
 *     compressors, pools, key tiles: one session's bytes); the row-sized selections; the stage's named areas and its
 *     communication; then the arena at each N and what is left. A line "MAX" gives what tensor_plan_fit resolves.
 *   - "MOST": the most sessions with no serving cap (unbounded sessions, U0), from the parts: the arena is linear in
 *     the sessions (checked), so a card holds (budget - fixed) / growth. tensor_plan_fit from one more, on the
 *     cards' budgets, must resolve to it. A knob --context_tokens=N gives the fit at that context.
 *   - Without arguments, as `make check` runs it: the same on the model's tables (model_v41_pack.h) at production's
 *     free memory, where each arena must be its fixed part plus N times its per-session growth, and that growth must
 *     be at least the per-session attention state the plan names; TP4_PRO also with --tp_attention_owner=1, where
 *     each CMP keeps only its attention group's state.
 */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "model_v41_pack.h"
#include "anemone/storage_exl3.h"
#include "serving_cursor.h"
#include "model_v41_tp2_model.h"
#include "commands.h"

#define FIT_MOST_SESSIONS 8u /* the arena table's counts */
/* the uncapped count's bound: above any fit (the rings alone, 4 MiB a layer a session, keep every split below it) */
#define FIT_UNCAPPED_MOST SERVING_SLOTS_MOST
#define FIT_ROOMY_BYTES \
	(1ull << 42u) /* a card that holds every count, to read each arena */
#define FIT_CMP_FREE_MIB \
	64512u /* production's free memory (model_v41_placement_test.c) */
#define FIT_PRO_FREE_MIB 96608u
#define FIT_OPTION_BYTES 128u
#define FIT_MOST_KNOBS 8u /* knobs a real-pack run takes beside the serve */
#define FIT_FIXED_WORDS \
	5u /* program, command, placement, tables, pack: around the knobs */
#define FIT_REAL_ARGUMENTS \
	5 /* the test, placement, free memory, tables, pack */
#define FIT_WRONG_USAGE 2
#define FIT_DECIMAL_BASE 10
/* the words of a serve around its knobs: program, command, placement (the fixed head), then tables and pack */
enum fit_word { FIT_PROGRAM, FIT_COMMAND, FIT_PLACEMENT, FIT_TABLES };
#define FIT_HEAD_WORDS FIT_TABLES
/* a real-pack run's arguments: the test, placement, free memory, tables, pack, then knobs */
enum fit_argument {
	FIT_ARGUMENT_PLACEMENT = 1,
	FIT_ARGUMENT_FREE,
	FIT_ARGUMENT_TABLES,
	FIT_ARGUMENT_PACK
};
/* the arena of one and of two sessions: their difference is a session's growth */
enum fit_sessions { FIT_ONE_SESSION = 1, FIT_TWO_SESSIONS };
#define FIT_MIB(bytes) ((double)(bytes) / (double)BYTES_PER_MIB)

static uint64_t free_bytes_of_card[BACKEND_MAXIMUM_CARDS];
static bool pro_last;

static void cards_describe(const uint64_t *free_bytes)
{
	for (uint32_t card = 0; card < card_count; card++)
		descriptions[card] = (struct card_description_record){
			.arithmetic_class =
				pro_last && card + 1u == card_count ?
					ARITHMETIC_CLASS_SM_120_NATIVE :
					ARITHMETIC_CLASS_SM_80,
			.memory_total_bytes = FIT_ROOMY_BYTES,
			.memory_free_bytes = free_bytes[card],
			.memory_reserved_bytes =
				BACKEND_CUDA_MEMORY_RESERVED_BYTES
		};
}

static void plan_release(void)
{
	v41_tp2_model_destroy(tensor_model);
	tensor_model = NULL;
	plan = NULL;
}

/* One card's arena by kind at the plan's session count (one session's attention state). */
struct card_parts {
	uint64_t arena, weights_and_slots, arrays, session_state, selections,
		named, communication, largest_named;
	const char *largest_name;
};

static struct card_parts card_parts_of(uint32_t card)
{
	const struct v41_card *const planned =
		&tensor_model->views[card]->cards[card];
	const struct v41_tp2_stage_plan *const stage =
		tensor_model->stages[card];
	const struct model_plan_record *const view = tensor_model->views[card];
	struct card_parts parts = { .arena = planned->arena_byte_count,
				    .weights_and_slots = UINT64_MAX,
				    .communication =
					    stage->communication_byte_count };
	for (uint32_t array = 0; array < V41_DATA_ARRAY_COUNT; array++)
		if (planned->array_offset[array] != PROGRAM_NOWHERE &&
		    planned->array_offset[array] < parts.weights_and_slots)
			parts.weights_and_slots = planned->array_offset[array];
	uint64_t state_first = UINT64_MAX;
	for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
		const uint64_t offsets[] = { planned->selection_offset[layer],
					     planned->ring_offset[layer],
					     planned->checkpoint_offset[layer],
					     planned->compressor_offset[layer],
					     planned->pool_offset[layer],
					     planned->key_tiles_offset[layer] };
		for (size_t kind = 0;
		     kind < sizeof(offsets) / sizeof(offsets[0]); kind++)
			state_first = offsets[kind] < state_first ?
					      offsets[kind] :
					      state_first;
		parts.selections +=
			planned->selection_offset[layer] != PROGRAM_NOWHERE ?
				(uint64_t)view->limits.row_capacity *
					MODEL_SELECTION_ROW_BYTES :
				0u;
		parts.session_state +=
			(planned->ring_offset[layer] != PROGRAM_NOWHERE ?
				 V41_DATA_RING_SESSION_BYTES :
				 0u) +
			(planned->checkpoint_offset[layer] != PROGRAM_NOWHERE ?
				 V41_DATA_CHECKPOINT_SESSION_BYTES :
				 0u) +
			(planned->compressor_offset[layer] != PROGRAM_NOWHERE ?
				 V41_DATA_COMPRESSOR_SESSION_BYTES :
				 0u) +
			(planned->pool_offset[layer] != PROGRAM_NOWHERE ?
				 view->pool_session_bytes[layer] :
				 0u) +
			(planned->key_tiles_offset[layer] != PROGRAM_NOWHERE ?
				 view->key_session_bytes[layer] :
				 0u);
	}
	if (parts.weights_and_slots == UINT64_MAX)
		parts.weights_and_slots = 0;
	parts.arrays = state_first != UINT64_MAX &&
				       state_first > parts.weights_and_slots ?
			       state_first - parts.weights_and_slots :
			       0u;
	const struct {
		const char *name;
		struct linear_range range;
	} named[] = {
		{ "compact", stage->compact },
		{ "exact_hidden", stage->exact_hidden },
		{ "exact_sums", stage->exact_sums },
		{ "dense_input", stage->dense_input },
		{ "exl3_tc_workspace", stage->exl3_tc_workspace },
		{ "exl3_tcp_workspace", stage->exl3_tcp_workspace },
		{ "trunk_mma_partials", stage->trunk_mma_partials },
		{ "router_partials", stage->router_partials },
		{ "batch_results", stage->batch_results },
		{ "batch_logits", stage->batch_logits },
		{ "batch_rounds", stage->batch_rounds },
		{ "batch_drafts", stage->batch_drafts },
		{ "batch_rows", stage->batch_rows },
		{ "draft_area", stage->draft_area },
		{ "sampler_exchange", stage->sampler_exchange },
	};
	parts.largest_name = "-";
	for (size_t index = 0; index < sizeof(named) / sizeof(named[0]);
	     index++) {
		parts.named += named[index].range.byte_count;
		if (named[index].range.byte_count > parts.largest_named) {
			parts.largest_named = named[index].range.byte_count;
			parts.largest_name = named[index].name;
		}
	}
	return parts;
}

/* The plan at a session count on roomy cards; each card's parts, or false where the plan is not made. */
static bool parts_at(uint32_t sessions, uint32_t *card_of_unit,
		     struct card_parts *parts)
{
	uint64_t roomy[BACKEND_MAXIMUM_CARDS];
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++)
		roomy[card] = FIT_ROOMY_BYTES;
	cards_describe(roomy);
	struct model_limits limits = options_limits();
	limits.session_capacity = sessions;
	const bool made =
		tensor_plan_prepare(&limits,
				    (uint64_t)card_memory_kept_free_mib *
					    BYTES_PER_MIB,
				    card_of_unit) == STATUS_OK;
	for (uint32_t card = 0; made && card < card_count; card++)
		parts[card] = card_parts_of(card);
	plan_release();
	return made;
}

/* The split's no-axis serve, parsed and opened as commands.c does; its pack is commands.c's pack. */
static bool serve_open(const char *placement, const char *tables,
		       const char *pack_path, uint32_t knob_count,
		       char *const *knobs)
{
	static char words[FIT_TABLES + 1][FIT_OPTION_BYTES];
	(void)snprintf(words[FIT_PROGRAM], sizeof(words[FIT_PROGRAM]),
		       "anemone");
	(void)snprintf(words[FIT_COMMAND], sizeof(words[FIT_COMMAND]), "serve");
	(void)snprintf(words[FIT_PLACEMENT], sizeof(words[FIT_PLACEMENT]), "%s",
		       placement);
	(void)snprintf(words[FIT_TABLES], sizeof(words[FIT_TABLES]), "%s",
		       tables);
	static char pack_word[PATH_MAX];
	(void)snprintf(pack_word, sizeof(pack_word), "%s", pack_path);
	if (knob_count > FIT_MOST_KNOBS)
		return false;
	char *arguments[FIT_FIXED_WORDS + FIT_MOST_KNOBS] = {
		words[FIT_PROGRAM], words[FIT_COMMAND], words[FIT_PLACEMENT]
	};
	uint32_t count = FIT_HEAD_WORDS;
	for (uint32_t knob = 0; knob < knob_count; knob++)
		arguments[count++] = knobs[knob];
	arguments[count++] = words[FIT_TABLES];
	arguments[count++] = pack_word;
	static struct axis_arguments_record expanded;
	struct storage_reader *opened = NULL;
	if (!axis_expand((int)count, arguments, &expanded) ||
	    !parse_arguments(expanded.count, expanded.values))
		return false;
	storage_exl3_read_mode(EXL3_READS_DROPPED);
	const struct storage_operations_record *const storage = pack_storage();
	return storage->reader_open(copy_paths, copy_count, &opened) ==
		       STATUS_OK &&
	       pack_open_entry(storage, opened, &pack) == STATUS_OK;
}

static bool fit_report(const char *label, uint32_t split_cards)
{
	card_count = split_cards;
	const uint32_t unit_count = pack_header_of(pack)->unit_count;
	uint32_t *const card_of_unit =
		calloc(unit_count, sizeof(*card_of_unit));
	struct card_parts parts[FIT_MOST_SESSIONS + 1u][BACKEND_MAXIMUM_CARDS];
	bool passed = card_of_unit != NULL;
	for (uint32_t sessions = 1; passed && sessions <= FIT_MOST_SESSIONS;
	     sessions++)
		passed = parts_at(sessions, card_of_unit, parts[sessions]);
	const uint64_t kept =
		(uint64_t)card_memory_kept_free_mib * BYTES_PER_MIB;
	(void)printf("%s: context %" PRIu32 " tokens, rows %" PRIu32
		     ", draft %" PRIu32 ", pipeline %" PRIu32
		     ", kept free %" PRIu32 " MiB\n",
		     label, context_tokens, effective_row_capacity(),
		     options_limits().draft_positions, pipeline_steps,
		     card_memory_kept_free_mib);
	for (uint32_t card = 0; passed && card < card_count; card++) {
		const struct card_parts *const one =
			&parts[FIT_ONE_SESSION][card];
		const uint64_t growth =
			parts[FIT_TWO_SESSIONS][card].arena - one->arena;
		const uint64_t budget =
			free_bytes_of_card[card] >
					BACKEND_CUDA_MEMORY_RESERVED_BYTES +
						kept ?
				free_bytes_of_card[card] -
					BACKEND_CUDA_MEMORY_RESERVED_BYTES -
					kept :
				0u;
		(void)printf(
			"CARD %" PRIu32
			" free %.0f budget %.0f MiB | weights+slots %.0f arrays %.0f selections %.0f named %.0f"
			" (largest %s %.0f) communication %.0f | per session: grows %.0f, attention state %.0f |"
			" fixed %.0f\n",
			card, FIT_MIB(free_bytes_of_card[card]),
			FIT_MIB(budget), FIT_MIB(one->weights_and_slots),
			FIT_MIB(one->arrays), FIT_MIB(one->selections),
			FIT_MIB(one->named), one->largest_name,
			FIT_MIB(one->largest_named),
			FIT_MIB(one->communication), FIT_MIB(growth),
			FIT_MIB(one->session_state),
			FIT_MIB(one->arena - growth));
		(void)printf("CARD %" PRIu32 " arena MiB by sessions:", card);
		for (uint32_t sessions = 1; sessions <= FIT_MOST_SESSIONS;
		     sessions++) {
			const uint64_t arena = parts[sessions][card].arena;
			(void)printf(" %" PRIu32 ":%.0f(%s%.0f)", sessions,
				     FIT_MIB(arena),
				     arena <= budget ? "+" : "-",
				     FIT_MIB(arena <= budget ? budget - arena :
							       arena - budget));
			/* linear in the sessions, and growing at least by the state the plan names */
			passed = passed &&
				 arena == one->arena - growth +
						  (uint64_t)sessions * growth &&
				 growth >= one->session_state;
		}
		(void)putchar('\n');
	}
	cards_describe(free_bytes_of_card);
	struct model_limits limits = options_limits();
	limits.session_capacity = FIT_MOST_SESSIONS;
	session_slots_max = true;
	const result_status fitted =
		passed ? tensor_plan_fit(&limits, kept, card_of_unit) :
			 STATUS_INVALID_ARGUMENT;
	plan_release();
	(void)printf("MAX %s: %s, max -> %" PRIu32 "\n", label,
		     status_text(fitted),
		     fitted == STATUS_OK ? limits.session_capacity : 0u);
	uint64_t most = FIT_UNCAPPED_MOST;
	for (uint32_t card = 0; passed && card < card_count; card++) {
		const struct card_parts *const one =
			&parts[FIT_ONE_SESSION][card];
		const uint64_t growth =
			parts[FIT_TWO_SESSIONS][card].arena - one->arena;
		const uint64_t fixed = one->arena - growth;
		const uint64_t budget =
			free_bytes_of_card[card] >
					BACKEND_CUDA_MEMORY_RESERVED_BYTES +
						kept ?
				free_bytes_of_card[card] -
					BACKEND_CUDA_MEMORY_RESERVED_BYTES -
					kept :
				0u;
		const uint64_t holds = budget <= fixed ? 0u :
				       growth == 0 ? FIT_UNCAPPED_MOST :
						     (budget - fixed) / growth;
		most = holds < most ? holds : most;
	}
	limits = options_limits();
	limits.session_capacity = (uint32_t)most + 1u;
	const result_status uncapped =
		passed && most != 0 && most < FIT_UNCAPPED_MOST ?
			tensor_plan_fit(&limits, kept, card_of_unit) :
			STATUS_INVALID_ARGUMENT;
	plan_release();
	(void)printf(
		"MOST %s: %" PRIu64 " sessions of %" PRIu32
		" tokens with no serving cap (the parts); the plan from %" PRIu64
		": %s, -> %" PRIu32 "\n",
		label, most, context_tokens, most + 1u, status_text(uncapped),
		uncapped == STATUS_OK ? limits.session_capacity : 0u);
	passed = passed && uncapped == STATUS_OK &&
		 limits.session_capacity == most;
	free(card_of_unit);
	(void)fflush(stdout);
	return passed;
}

static bool free_memory_parse(const char *text)
{
	uint32_t count = 0;
	for (const char *cursor = text; count < BACKEND_MAXIMUM_CARDS;) {
		char *end = NULL;
		const unsigned long mib =
			strtoul(cursor, &end, FIT_DECIMAL_BASE);
		if (end == cursor || mib == 0)
			return false;
		free_bytes_of_card[count++] = (uint64_t)mib * BYTES_PER_MIB;
		if (*end == '\0')
			return count == card_count;
		if (*end != ',')
			return false;
		cursor = end + 1;
	}
	return false;
}

/* The model's tables at production's free memory, in a child per split (pristine commands.c state); knob, when not
 * NULL, beside the serve. */
static bool tables_case(const char *placement, const char *knob,
			uint32_t split_cards, bool pro,
			enum test_model_v41_change_kind kind)
{
	const pid_t child = fork();
	if (child == 0) {
		char program[] = "anemone", serve[] = "serve",
		     option[FIT_OPTION_BYTES], knob_word[FIT_OPTION_BYTES],
		     tables[] = "tables", weights[] = "weights";
		(void)snprintf(option, sizeof(option), "%s", placement);
		(void)snprintf(knob_word, sizeof(knob_word), "%s",
			       knob != NULL ? knob : "");
		char *arguments[FIT_FIXED_WORDS + 1] = { program, serve,
							 option };
		int count = FIT_HEAD_WORDS;
		if (knob != NULL) /* the knob before the tables and pack */
			arguments[count++] = knob_word;
		arguments[count++] = tables;
		arguments[count++] = weights;
		struct axis_arguments_record expanded = { 0 };
		pro_last = pro;
		card_count = split_cards;
		for (uint32_t card = 0; card < split_cards; card++)
			free_bytes_of_card[card] =
				(uint64_t)(pro && card + 1u == split_cards ?
						   FIT_PRO_FREE_MIB :
						   FIT_CMP_FREE_MIB) *
				BYTES_PER_MIB;
		const struct test_model_v41_change change = { .kind = kind };
		const bool passed =
			axis_expand(count, arguments, &expanded) &&
			parse_arguments(expanded.count, expanded.values) &&
			test_model_v41_draft_pack_open(change, &pack) ==
				STATUS_OK &&
			fit_report(placement, split_cards);
		_exit(passed ? 0 : 1);
	}
	int status = 0;
	return child > 0 && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main(int argument_count, char **arguments)
{
	(void)setvbuf(stdout, NULL, _IOLBF, 0);
	if (argument_count == 1) {
		const bool passed =
			tables_case("--placement=tp4_pro", NULL,
				    V41_DATA_TP2_PROGRAM_CARDS, true,
				    TEST_MODEL_V41_CHANGE_NONE) &&
			tables_case("--placement=tp4_pro",
				    "--tp_attention_owner=1",
				    V41_DATA_TP2_PROGRAM_CARDS, true,
				    TEST_MODEL_V41_CHANGE_NONE) &&
			tables_case("--placement=tp4_exl3", NULL, TP4_RANKS,
				    false, TEST_MODEL_V41_CHANGE_EXL3);
		(void)printf("tp_session_fit_test %s\n",
			     passed ? "passed" : "FAILED");
		return passed ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	const char *const prefix = "--card_free_mib=";
	if (argument_count < FIT_REAL_ARGUMENTS ||
	    argument_count > FIT_REAL_ARGUMENTS + (int)FIT_MOST_KNOBS ||
	    (strcmp(arguments[FIT_ARGUMENT_PLACEMENT], "--placement=tp4_pro") !=
		     0 &&
	     strcmp(arguments[FIT_ARGUMENT_PLACEMENT],
		    "--placement=tp4_exl3") != 0) ||
	    strncmp(arguments[FIT_ARGUMENT_FREE], prefix, strlen(prefix)) !=
		    0) {
		(void)fprintf(
			stderr,
			"usage: tp_session_fit_test --placement=tp4_pro|tp4_exl3 "
			"--card_free_mib=N,N,... <tables> <pack> [--knob=value ...]\n");
		return FIT_WRONG_USAGE;
	}
	pro_last = strcmp(arguments[FIT_ARGUMENT_PLACEMENT],
			  "--placement=tp4_pro") == 0;
	const uint32_t split_cards = pro_last ? V41_DATA_TP2_PROGRAM_CARDS :
						TP4_RANKS;
	card_count = split_cards;
	if (!free_memory_parse(arguments[FIT_ARGUMENT_FREE] + strlen(prefix)) ||
	    !serve_open(arguments[FIT_ARGUMENT_PLACEMENT],
			arguments[FIT_ARGUMENT_TABLES],
			arguments[FIT_ARGUMENT_PACK],
			(uint32_t)(argument_count - FIT_REAL_ARGUMENTS),
			arguments + FIT_REAL_ARGUMENTS)) {
		(void)fprintf(
			stderr,
			"tp_session_fit_test: the serve or its pack does not open\n");
		return EXIT_FAILURE;
	}
	return fit_report(arguments[FIT_ARGUMENT_PLACEMENT], split_cards) ?
		       EXIT_SUCCESS :
		       EXIT_FAILURE;
}
