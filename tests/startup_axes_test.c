/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Exercise real CLI parsing and admission without opening a model or a card.
 * Each child starts from pristine main.c defaults. */
#include <sys/wait.h>
#include "step_driver.h"

int axis_embedded_main(int argument_count, char **arguments);
#define main axis_embedded_main
#include "../source/main.c"
#undef main

#define TEST_ARGUMENT_CAPACITY 96u
#define TEST_OPTION_BYTES 96u
#define TEST_LOG_CAPACITY 16384u
#define TEST_OPTION_START 2u
#define TEST_EXTRA_CAPACITY 3u
#define TEST_WEIGHT_FAMILIES 2u
#define TEST_OPTION_ORDERS 2u
#define TEST_REFERENCE_SELECTOR_OPTIONS 2u
#define TEST_REFERENCE_CONTEXT_TOKENS 512u
#define TEST_REFERENCE_VOCABULARY 129280u
#define TEST_REFERENCE_INPUT_CAPACITY 9u
#define TEST_THIN_PIPELINE_STEPS 2u
#define TEST_THIN_WIDE_ROWS 2048u
#define TEST_BATCH_WIDE_ROWS 1024u
#define TEST_BATCH_WINDOW_CORE 2u
#define TEST_BATCH_POOL_DEFERRED_CORE 3u
#define TEST_FULL_DRAFT_DEPTH 5u
#define TEST_OVERRIDE_DRAFT_DEPTH 6u
#define TEST_OVERRIDE_ROUNDS 4u
#define TEST_ORIGINAL_CARD_COUNT 5u
#define TEST_ORIGINAL_CMP_CARDS 4u
#define TEST_ORIGINAL_DECODE_PIPELINE 8u

#define TEST_ORIGINAL_LAYOUT                                                   \
	"vision: card 0; embedding: card 0; layers 0..7: card 0; "             \
	"layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; " \
	"layers 29..39: card 4; head: card 4; draft head: card 4"
#define TEST_LAYOUT_OVERRIDE                                                   \
	"embedding: card 0; vision: card 0; layers 0..7: card 0; "             \
	"layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; " \
	"layers 29..39: card 4; head: card 4; draft head: card 4"

enum test_axis {
	TEST_CATEGORY,
	TEST_EXACTNESS,
	TEST_CONCURRENCY,
	TEST_FOCUS,
	TEST_AXES
};

static const char *const category_values[] = { "code", "prose", "structured",
					       "vision", "balanced" };
static const char *const exactness_values[] = { "exact", "balanced", "fast" };
static const char *const concurrency_values[] = { "single", "multiple",
						  "balanced" };
static const char *const focus_values[] = { "balanced", "ttft", "decode",
					    "prefill" };
static const char *const axis_names[TEST_AXES] = { "category", "exactness",
						   "concurrency", "focus" };
static unsigned axis_cases_run;

enum test_destination {
	TEST_UNCHANGED,
	TEST_WIDE_ROWS,
	TEST_DRAFT_POLICY,
	TEST_DRAFT_DEPTH,
	TEST_SPECULATIVE_LOOKUP,
	TEST_SPECULATIVE_OFF,
	TEST_DECODE_ROWS,
	TEST_DECODE_ROUNDS,
	TEST_LAYOUT,
	TEST_BATCH_MENU_OFF
};

struct axis_test_case {
	const struct axis_choice *choice;
	const char *command;
	const char *extra[TEST_EXTRA_CAPACITY];
	bool extras_first, expand_expected, parse_expected;
	bool placement_expected, batch_expected;
	enum test_destination destination;
};

/* This is the production late-startup verifier gate with host-only program
 * descriptions. In particular, eight-verifier recipes cannot bind one slot. */
static bool batch_axes_admitted(void)
{
	if (batch_rows[0] == 0)
		return true;
	struct execution_program axis_programs[MOST_BATCH_PROGRAMS] = { 0 };
	const struct execution_program *bindings[MOST_BATCH_PROGRAMS] = { 0 };
	uint32_t count = 0;
	for (; count < MOST_BATCH_PROGRAMS && batch_rows[count] != 0; count++) {
		axis_programs[count].kind = PROGRAM_BATCH;
		axis_programs[count].row_count = batch_rows[count];
		axis_programs[count].limits.session_capacity = session_slots;
		bindings[count] = &axis_programs[count];
	}
	struct step_driver axis_driver = { .programs = bindings,
					   .program_count = count };
	const result_status status = step_batch_verifiers_attach(
		&axis_driver, batch_verifiers, count);
	free(axis_driver.batch_verifiers);
	return status == STATUS_OK;
}

static size_t option_name(const char *option, const char **name)
{
	size_t bytes = strcspn(option, "=");
	*name = option;
	if ((bytes == strlen("--draft") &&
	     strncmp(option, "--draft", bytes) == 0) ||
	    (bytes == strlen("--speculative") &&
	     strncmp(option, "--speculative", bytes) == 0)) {
		*name = "--speculative";
		bytes = strlen(*name);
	} else if (bytes == strlen("--speculative_depth") &&
		   strncmp(option, "--speculative_depth", bytes) == 0) {
		*name = "--draft_positions";
		bytes = strlen(*name);
	}
	return bytes;
}

static bool explicitly_selected(const char *setting, int count,
				char *const *arguments)
{
	const char *name;
	const size_t bytes = option_name(setting, &name);
	for (int index = TEST_OPTION_START;
	     index < count &&
	     strncmp(arguments[index], "--", strlen("--")) == 0;
	     index++) {
		const char *candidate;
		if (option_name(arguments[index], &candidate) == bytes &&
		    strncmp(name, candidate, bytes) == 0)
			return true;
	}
	return false;
}

static bool axis_argument(const char *argument)
{
	for (size_t axis = 0; axis < TEST_AXES; axis++) {
		char prefix[TEST_OPTION_BYTES];
		const int written = snprintf(prefix, sizeof(prefix),
					     "--%s=", axis_names[axis]);
		if (written < 0 || (size_t)written >= sizeof(prefix))
			return false;
		if (strncmp(argument, prefix, (size_t)written) == 0)
			return true;
	}
	return false;
}

static unsigned exact_occurrences(const char *text, const char *token)
{
	unsigned count = 0;
	const size_t bytes = strlen(token);
	for (const char *found = strstr(text, token); found != NULL;
	     found = strstr(found + bytes, token)) {
		const char after = found[bytes];
		if (after == '\0' || after == ' ' || after == '\n' ||
		    after == '\r' || after == '\t')
			count++;
	}
	return count;
}

/* Compare the table's complete expansion before main's list parser mutates
 * comma-separated arguments. Every inserted setting must also be logged once. */
static bool
declared_settings_match(const struct axis_test_case *test,
			const struct axis_arguments_record *expanded,
			int original_count, char *const *original_arguments)
{
	char *log_text = NULL;
	size_t log_bytes = 0;
	FILE *log = open_memstream(&log_text, &log_bytes);
	if (log == NULL)
		return false;
	axis_log(expanded, log);
	bool passed = fclose(log) == 0;
	for (size_t axis = 0; axis < TEST_AXES; axis++) {
		char selected[TEST_OPTION_BYTES];
		const int written = snprintf(selected, sizeof(selected),
					     "%s=%s", axis_names[axis],
					     expanded->selected[axis] != NULL ?
						     test->choice->value[axis] :
						     "unset");
		passed = passed && written > 0 &&
			 (size_t)written < sizeof(selected) &&
			 strstr(log_text, selected) != NULL;
		if (expanded->selected[axis] != NULL)
			passed = passed &&
				 strcmp(expanded->selected[axis],
					test->choice->value[axis]) == 0;
	}
	const bool serving = strcmp(test->command, "serve") == 0;
	const char *base = !serving && test->choice->generate_settings[0] !=
						   '\0' ?
				   test->choice->generate_settings :
				   test->choice->settings;
	char *groups[] = { strdup(base),
			   strdup(serving ? test->choice->serve_settings :
					    "") };
	if (groups[0] == NULL || groups[1] == NULL)
		passed = false;
	unsigned inserted = 0;
	for (size_t group = 0; group < sizeof(groups) / sizeof(groups[0]);
	     group++) {
		char *save = NULL;
		for (char *setting =
			     groups[group] != NULL ?
				     strtok_r(groups[group], "\n", &save) :
				     NULL;
		     setting != NULL; setting = strtok_r(NULL, "\n", &save)) {
			if (explicitly_selected(setting, original_count,
						original_arguments))
				continue;
			const unsigned index = TEST_OPTION_START + inserted;
			passed = passed && index < (unsigned)expanded->count &&
				 strcmp(expanded->values[index], setting) ==
					 0 &&
				 exact_occurrences(log_text, setting) == 1u;
			inserted++;
		}
		free(groups[group]);
	}
	unsigned consumed = 0;
	int option_end = TEST_OPTION_START;
	for (; option_end < original_count &&
	       strncmp(original_arguments[option_end], "--", strlen("--")) == 0;
	     option_end++)
		consumed += axis_argument(original_arguments[option_end]) ? 1u :
									    0u;
	passed = passed && expanded->selected_count == consumed &&
		 expanded->inserted_count == inserted &&
		 expanded->count ==
			 original_count - (int)consumed + (int)inserted;
	int remaining = (int)(TEST_OPTION_START + inserted);
	for (int index = TEST_OPTION_START; index < original_count; index++) {
		if (index < option_end &&
		    axis_argument(original_arguments[index]))
			continue;
		passed = passed && remaining < expanded->count &&
			 strcmp(expanded->values[remaining],
				original_arguments[index]) == 0;
		remaining++;
	}
	passed = passed && remaining == expanded->count;
	free(log_text);
	return passed;
}

/* This current-knob candidate is opt-in for its qualification window; it is
 * not an owner-selected default or a claimed measured compromise. */
static bool all_balanced_candidate(const struct axis_choice *choice)
{
	if (strcmp(choice->weight_family, "exl3") != 0)
		return false;
	for (size_t axis = 0; axis < TEST_AXES; axis++)
		if (strcmp(choice->value[axis], "balanced") != 0)
			return false;
	return true;
}

static bool original_decode_candidate(const struct axis_choice *choice)
{
	const char *category = choice->value[TEST_CATEGORY];
	const char *concurrency = choice->value[TEST_CONCURRENCY];
	return strcmp(choice->weight_family, "original") == 0 &&
	       strcmp(choice->value[TEST_EXACTNESS], "balanced") == 0 &&
	       strcmp(choice->value[TEST_FOCUS], "decode") == 0 &&
	       (strcmp(category, "code") == 0 ||
		strcmp(category, "prose") == 0 ||
		strcmp(category, "structured") == 0) &&
	       (strcmp(concurrency, "single") == 0 ||
		strcmp(concurrency, "multiple") == 0);
}

static bool original_reference_candidate(const struct axis_choice *choice)
{
	return strcmp(choice->weight_family, "original") == 0 &&
	       strcmp(choice->value[TEST_EXACTNESS], "exact") == 0 &&
	       strcmp(choice->value[TEST_CATEGORY], "vision") != 0 &&
	       strcmp(choice->value[TEST_CONCURRENCY], "single") == 0;
}

/* Host descriptions model the intended ordinal map, without opening cards.
 * The raw strategy order must put the four CMPs before the native PRO. */
static bool original_card_classes_match(void)
{
	static const uint32_t expected_order[TEST_ORIGINAL_CARD_COUNT] = {
		0u, 1u, 3u, 4u, 2u
	};
	static const uint32_t ordinal_classes[TEST_ORIGINAL_CARD_COUNT] = {
		ARITHMETIC_CLASS_SM_80, ARITHMETIC_CLASS_SM_80,
		ARITHMETIC_CLASS_SM_120_NATIVE, ARITHMETIC_CLASS_SM_80,
		ARITHMETIC_CLASS_SM_80
	};
	if (card_count != TEST_ORIGINAL_CARD_COUNT)
		return false;
	uint32_t used = 0;
	for (uint32_t slot = 0; slot < TEST_ORIGINAL_CARD_COUNT; slot++) {
		const uint32_t ordinal = card_order[slot];
		if (ordinal >= TEST_ORIGINAL_CARD_COUNT ||
		    ordinal != expected_order[slot] ||
		    (used & (1u << ordinal)) != 0)
			return false;
		const uint32_t expected_class =
			slot < TEST_ORIGINAL_CMP_CARDS ?
				ARITHMETIC_CLASS_SM_80 :
				ARITHMETIC_CLASS_SM_120_NATIVE;
		if (ordinal_classes[ordinal] != expected_class)
			return false;
		used |= 1u << ordinal;
	}
	return used == (1u << TEST_ORIGINAL_CARD_COUNT) - 1u;
}

static bool original_tuple_geometry(const struct axis_choice *choice,
				    const char *command)
{
	const bool serving = strcmp(command, "serve") == 0;
	if (original_reference_candidate(choice))
		return execution_placement == EXECUTION_PP5 &&
		       numerics == NUMERICS_EXACT &&
		       cmp_order == CMP_ORDER_REFERENCE &&
		       linear_order == LINEAR_ORDER_REFERENCE &&
		       linear_profile == LINEAR_PROFILE_PRECISION &&
		       pro_emulation == 0u && official_reference == 1u &&
		       official_hc_rows == 1u && step_rows == 1u &&
		       context_tokens == TEST_REFERENCE_CONTEXT_TOKENS &&
		       session_slots == 1u && wide_rows == 0u &&
		       full_steps == 0u && speculative == SPECULATIVE_OFF &&
		       pipeline_steps == 1u && chunks_in_flight == 1u &&
		       step_graphs == 1u && batch_rows[0] == 0u &&
		       batch_verifiers[0] == 0u && original_batch == 0u &&
		       original_chunk_core == 0u &&
		       original_dense_staged == 0u &&
		       original_wide_experts == 0u &&
		       tp_pipeline_overlap == 0u && layout_text != NULL &&
		       strcmp(layout_text, TEST_ORIGINAL_LAYOUT) == 0 &&
		       original_card_classes_match() &&
		       (!serving ||
			(prefix_reuse == 0u && prefill_order == 0u));
	const bool multiple =
		strcmp(choice->value[TEST_CONCURRENCY], "multiple") == 0;
	const uint32_t depth = !serving || !multiple ?
				       TEST_THIN_PIPELINE_STEPS :
				       TEST_ORIGINAL_DECODE_PIPELINE;
	bool passed =
		original_decode_candidate(choice) &&
		execution_placement ==
			(multiple ? EXECUTION_PP5 : EXECUTION_TP4_PRO) &&
		numerics == NUMERICS_EXACT &&
		cmp_order == CMP_ORDER_REFERENCE &&
		linear_order == LINEAR_ORDER_REFERENCE &&
		linear_profile == LINEAR_PROFILE_PRECISION &&
		pro_emulation == 0u && tp_wire_bf16 == 0u &&
		official_reference == 0u && official_hc_rows == 0u &&
		speculative == SPECULATIVE_DEVICE &&
		step_rows == LINEAR_MAXIMUM_ROWS &&
		wide_rows == TEST_BATCH_WIDE_ROWS && full_steps == 1u &&
		prefill_cone == 1u && chunks_in_flight == 1u &&
		step_graphs == 1u && draft_block == TEST_FULL_DRAFT_DEPTH &&
		draft_positions == TEST_FULL_DRAFT_DEPTH &&
		draft_policy.depth_policy == DRAFT_DEPTH_FIXED &&
		batch_rows[0] == 0u && batch_verifiers[0] == 0u &&
		batch_step_us[0] == 0u && pipeline_steps == depth &&
		session_slots ==
			(serving && multiple ? SERVING_SESSION_CAPACITY : 1u) &&
		tp_head_shards == 0u && tp_head_stream == 0u &&
		tp_draft_shards == 0u && tp_draft_layers == 0u &&
		tp_code_wire == 0u && tp_attention_owner == 0u &&
		prefill_decode_rows == 0u && prefill_order == 0u &&
		original_card_classes_match();
	if (serving)
		passed = passed && prefix_reuse == 0u;
	if (multiple)
		passed = passed && layout_text != NULL &&
			 strcmp(layout_text, TEST_ORIGINAL_LAYOUT) == 0;
	else
		passed = passed && layout_text == NULL;
	return passed;
}

static bool ordinary_tuple_geometry(const struct axis_choice *choice,
				    const char *command)
{
	if (strcmp(choice->weight_family, "original") == 0)
		return original_tuple_geometry(choice, command);
	const bool compromise = all_balanced_candidate(choice);
	const bool multiple =
		compromise ||
		strcmp(choice->value[TEST_CONCURRENCY], "multiple") == 0;
	const bool serving = strcmp(command, "serve") == 0;
	const bool full_layer_draft =
		multiple ||
		(strcmp(choice->value[TEST_CATEGORY], "code") == 0 &&
		 strcmp(choice->value[TEST_FOCUS], "decode") == 0);
	bool passed = execution_placement == EXECUTION_TP4_EXL3 &&
		      numerics == NUMERICS_RELAXED &&
		      cmp_order == CMP_ORDER_REFERENCE && tp_wire_bf16 == 1u &&
		      speculative == SPECULATIVE_DEVICE &&
		      draft_positions == TEST_FULL_DRAFT_DEPTH &&
		      tp_head_shards == 1u && tp_head_stream == 1u &&
		      tp_draft_shards == 1u &&
		      tp_draft_layers == (full_layer_draft ? 1u : 0u);
	if (multiple) {
		const uint32_t expected_core =
			strcmp(choice->value[TEST_CATEGORY], "code") == 0 &&
					strcmp(choice->value[TEST_FOCUS],
					       "decode") == 0 ?
				TEST_BATCH_POOL_DEFERRED_CORE :
				TEST_BATCH_WINDOW_CORE;
		passed = passed && batch_core_once == expected_core &&
			 wide_rows == (compromise ? TEST_THIN_WIDE_ROWS :
						    TEST_BATCH_WIDE_ROWS) &&
			 pipeline_steps == 1u &&
			 batch_rows[0] == LINEAR_MAXIMUM_ROWS &&
			 session_slots ==
				 (serving ? SERVING_SESSION_CAPACITY : 1u);
		if (!serving)
			passed = passed && batch_rows[1] == 0u &&
				 batch_verifiers[0] == 1u &&
				 batch_verifiers[1] == 0u;
		passed = passed &&
			 prefill_decode_rows ==
				 (serving ? PREFILL_DECODE_DEFAULT_ROWS : 0u) &&
			 prefill_decode_rounds ==
				 PREFILL_DECODE_DEFAULT_ROUNDS &&
			 prefill_decode_waiters ==
				 PREFILL_DECODE_DEFAULT_WAITERS &&
			 prefill_order ==
				 (serving ? PREFILL_ORDER_SHORTEST : 0u);
	} else {
		passed = passed && wide_rows == TEST_THIN_WIDE_ROWS &&
			 pipeline_steps == TEST_THIN_PIPELINE_STEPS &&
			 batch_rows[0] == 0u && session_slots == 1u &&
			 prefill_decode_rows == 0u;
	}
	if (!multiple || (strcmp(choice->value[TEST_CATEGORY], "code") == 0 &&
			  strcmp(choice->value[TEST_FOCUS], "decode") == 0))
		passed = passed &&
			 draft_policy.depth_policy == DRAFT_DEPTH_FIXED &&
			 batch_step_us[0] == 0u;
	return passed;
}

static bool destination_matches(enum test_destination destination)
{
	switch (destination) {
	case TEST_UNCHANGED:
		return true;
	case TEST_WIDE_ROWS:
		return wide_rows == TEST_BATCH_WIDE_ROWS;
	case TEST_DRAFT_POLICY:
		return draft_policy.depth_policy == DRAFT_DEPTH_COST;
	case TEST_DRAFT_DEPTH:
		return draft_positions == TEST_OVERRIDE_DRAFT_DEPTH;
	case TEST_SPECULATIVE_LOOKUP:
		return speculative == SPECULATIVE_LOOKUP;
	case TEST_SPECULATIVE_OFF:
		return speculative == SPECULATIVE_OFF;
	case TEST_DECODE_ROWS:
		return prefill_decode_rows == 0u;
	case TEST_DECODE_ROUNDS:
		return prefill_decode_rounds == TEST_OVERRIDE_ROUNDS;
	case TEST_LAYOUT:
		return layout_text != NULL &&
		       strcmp(layout_text, TEST_LAYOUT_OVERRIDE) == 0;
	case TEST_BATCH_MENU_OFF:
		for (size_t cost = 0;
		     cost < sizeof(batch_step_us) / sizeof(batch_step_us[0]);
		     cost++)
			if (batch_step_us[cost] != 0u)
				return false;
		return true;
	}
	return false;
}

static bool parse_tuple_case(const struct axis_test_case *test)
{
	char axis_options[TEST_AXES][TEST_OPTION_BYTES];
	for (size_t axis = 0; axis < TEST_AXES; axis++) {
		const int written = snprintf(axis_options[axis],
					     sizeof(axis_options[axis]),
					     "--%s=%s", axis_names[axis],
					     test->choice->value[axis]);
		if (written < 0 ||
		    (size_t)written >= sizeof(axis_options[axis]))
			return false;
	}
	char program[] = "anemone", generate[] = "generate", serve[] = "serve",
	     load[] = "load", messages[] = "messages", tables[] = "tables",
	     weights[] = "weights", exl3[] = "--placement=tp4_exl3";
	char *arguments[TEST_ARGUMENT_CAPACITY] = {
		program, strcmp(test->command, "serve") == 0 ? serve :
			 strcmp(test->command, "load") == 0  ? load :
							       generate
	};
	char *extras[TEST_EXTRA_CAPACITY] = { 0 };
	for (size_t index = 0; index < TEST_EXTRA_CAPACITY; index++) {
		if (test->extra[index] == NULL)
			continue;
		extras[index] = strdup(test->extra[index]);
		if (extras[index] == NULL) {
			for (size_t previous = 0; previous < index; previous++)
				free(extras[previous]);
			return false;
		}
	}
	int count = TEST_OPTION_START;
	if (test->extras_first)
		for (size_t index = 0; index < TEST_EXTRA_CAPACITY; index++)
			if (extras[index] != NULL)
				arguments[count++] = extras[index];
	/* Original tuples must supply their own TP4_PRO or PP5 placement. An
	 * unconditional raw PP5 option would override the single-user strategy. */
	if (strcmp(test->choice->weight_family, "exl3") == 0)
		arguments[count++] = exl3;
	for (size_t axis = 0; axis < TEST_AXES; axis++)
		arguments[count++] = axis_options[axis];
	if (!test->extras_first)
		for (size_t index = 0; index < TEST_EXTRA_CAPACITY; index++)
			if (extras[index] != NULL)
				arguments[count++] = extras[index];
	if (strcmp(test->command, "generate") == 0)
		arguments[count++] = messages;
	if (strcmp(test->command, "load") != 0)
		arguments[count++] = tables;
	arguments[count++] = weights;
	FILE *diagnostic = tmpfile();
	const int saved_error = dup(STDERR_FILENO);
	if (diagnostic == NULL || saved_error < 0 ||
	    dup2(fileno(diagnostic), STDERR_FILENO) != STDERR_FILENO) {
		if (diagnostic != NULL)
			(void)fclose(diagnostic);
		if (saved_error >= 0)
			(void)close(saved_error);
		for (size_t index = 0; index < TEST_EXTRA_CAPACITY; index++)
			free(extras[index]);
		return false;
	}
	struct axis_arguments_record expanded = { 0 };
	const bool expanded_ok = axis_expand(count, arguments, &expanded);
	bool passed = expanded_ok == test->expand_expected;
	if (expanded_ok) {
		passed = passed && expanded.choice == test->choice &&
			 strcmp(expanded.weight_family,
				test->choice->weight_family) == 0 &&
			 declared_settings_match(test, &expanded, count,
						 arguments);
		const bool parsed =
			parse_arguments(expanded.count, expanded.values);
		passed = passed && parsed == test->parse_expected;
		if (parsed) {
			passed = passed &&
				 (startup_placement_status() ==
				  PROGRAM_SUCCEEDED) ==
					 test->placement_expected &&
				 batch_axes_admitted() ==
					 test->batch_expected &&
				 destination_matches(test->destination);
			if (test->extra[0] == NULL)
				passed = passed &&
					 ordinary_tuple_geometry(test->choice,
								 test->command);
		}
	}
	(void)fflush(stderr);
	passed = (dup2(saved_error, STDERR_FILENO) == STDERR_FILENO) && passed;
	(void)close(saved_error);
	(void)rewind(diagnostic);
	char diagnostic_text[TEST_LOG_CAPACITY];
	const size_t diagnostic_bytes = fread(
		diagnostic_text, 1u, sizeof(diagnostic_text) - 1u, diagnostic);
	diagnostic_text[diagnostic_bytes] = '\0';
	passed = passed && ferror(diagnostic) == 0 && fgetc(diagnostic) == EOF;
	if (!test->expand_expected && test->choice->hold[0] != '\0')
		passed = passed &&
			 strstr(diagnostic_text, test->choice->hold) != NULL;
	if (!passed)
		(void)fputs(diagnostic_text, stderr);
	(void)fclose(diagnostic);
	axis_arguments_free(&expanded);
	passed = passed && expanded.values == NULL &&
		 expanded.storage == NULL && expanded.count == 0 &&
		 expanded.selected_count == 0u && expanded.inserted_count == 0u;
	for (size_t index = 0; index < TEST_EXTRA_CAPACITY; index++)
		free(extras[index]);
	return passed;
}

static bool isolated_tuple_case(const struct axis_test_case *test)
{
	axis_cases_run++;
	const pid_t child = fork();
	if (child == 0)
		_exit(parse_tuple_case(test) ? 0 : 1);
	int status = 0;
	const bool passed = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) && WEXITSTATUS(status) == 0;
	if (!passed)
		(void)fprintf(stderr, "failed axes: %s/%s/%s/%s %s %s %s %s\n",
			      test->choice->value[TEST_CATEGORY],
			      test->choice->value[TEST_EXACTNESS],
			      test->choice->value[TEST_CONCURRENCY],
			      test->choice->value[TEST_FOCUS],
			      test->choice->weight_family, test->command,
			      test->extra[0] != NULL ? test->extra[0] : "",
			      test->extra[1] != NULL ? test->extra[1] : "");
	return passed;
}

static const struct axis_choice *
find_choice(const struct axis_choice *choices, size_t count,
	    const char *category, const char *exactness,
	    const char *concurrency, const char *focus, const char *family)
{
	for (size_t index = 0; index < count; index++)
		if (strcmp(choices[index].value[TEST_CATEGORY], category) ==
			    0 &&
		    strcmp(choices[index].value[TEST_EXACTNESS], exactness) ==
			    0 &&
		    strcmp(choices[index].value[TEST_CONCURRENCY],
			   concurrency) == 0 &&
		    strcmp(choices[index].value[TEST_FOCUS], focus) == 0 &&
		    strcmp(choices[index].weight_family, family) == 0)
			return &choices[index];
	return NULL;
}

static bool raw_arguments_case(const char *const *options, size_t option_count,
			       bool expansion_expected, bool parse_expected,
			       bool untouched)
{
	char program[] = "anemone", generate[] = "generate",
	     messages[] = "messages", tables[] = "tables",
	     weights[] = "weights";
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, generate };
	char *copies[TEST_ARGUMENT_CAPACITY] = { 0 };
	int count = TEST_OPTION_START;
	for (size_t index = 0; index < option_count; index++) {
		copies[index] = strdup(options[index]);
		if (copies[index] == NULL) {
			for (size_t previous = 0; previous < index; previous++)
				free(copies[previous]);
			return false;
		}
		arguments[count++] = copies[index];
	}
	arguments[count++] = messages;
	arguments[count++] = tables;
	arguments[count++] = weights;
	struct axis_arguments_record expanded = { 0 };
	const bool expanded_ok = axis_expand(count, arguments, &expanded);
	bool passed = expanded_ok == expansion_expected;
	if (expanded_ok) {
		if (untouched) {
			passed = passed && expanded.count == count &&
				 expanded.values == arguments &&
				 expanded.storage == NULL &&
				 expanded.choice == NULL &&
				 expanded.explicit_count == count &&
				 expanded.explicit_values == arguments &&
				 expanded.selected_count == 0u &&
				 expanded.inserted_count == 0u;
			for (size_t axis = 0; axis < TEST_AXES; axis++)
				passed = passed &&
					 expanded.selected[axis] == NULL;
			char *log_text = NULL;
			size_t log_bytes = 0;
			FILE *log = open_memstream(&log_text, &log_bytes);
			if (log == NULL) {
				passed = false;
			} else {
				axis_log(&expanded, log);
				passed = (fclose(log) == 0) &&
					 log_bytes == 0u && passed;
			}
			free(log_text);
		}
		passed = (parse_arguments(expanded.count, expanded.values) ==
			  parse_expected) &&
			 passed;
	}
	axis_arguments_free(&expanded);
	for (size_t index = 0; index < option_count; index++)
		free(copies[index]);
	return passed;
}

static bool isolated_raw_case(const char *const *options, size_t option_count,
			      bool expansion_expected, bool parse_expected,
			      bool untouched)
{
	axis_cases_run++;
	const pid_t child = fork();
	if (child == 0)
		_exit(raw_arguments_case(options, option_count,
					 expansion_expected, parse_expected,
					 untouched) ?
			      0 :
			      1);
	int status = 0;
	const bool passed = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) && WEXITSTATUS(status) == 0;
	if (!passed)
		(void)fprintf(stderr, "failed raw axes case: %s\n",
			      option_count > 0 ? options[0] : "no axes");
	return passed;
}

/* Raw flags exercise the real parser while exact still awaits serving gates. */
static bool reference_raw_recipe_cases(const struct axis_choice *reference)
{
	if (reference == NULL)
		return false;
	char *storage = strdup(reference->generate_settings);
	if (storage == NULL)
		return false;
	const char *options[TEST_ARGUMENT_CAPACITY] = { 0 };
	size_t count = 0;
	char *save = NULL;
	for (char *setting = strtok_r(storage, "\n", &save); setting != NULL;
	     setting = strtok_r(NULL, "\n", &save)) {
		if (count == TEST_ARGUMENT_CAPACITY) {
			free(storage);
			return false;
		}
		options[count++] = setting;
	}
	bool passed = isolated_raw_case(options, count, true, true, true);
	const char *const unsupported[] = {
		"--rows=8",	      "--wide_rows=1024",
		"--full_steps=1",     "--context_tokens=513",
		"--speculative=2",    "--official_hc_rows=0",
		"--pro_emulation=1",  "--linear_order=0",
		"--linear_profile=1", "--placement=tp4_exl3",
		"--pipeline_steps=2"
	};
	for (size_t replacement = 0;
	     replacement < sizeof(unsupported) / sizeof(unsupported[0]);
	     replacement++) {
		const size_t bytes = strcspn(unsupported[replacement], "=");
		bool found = false;
		for (size_t index = 0; index < count && !found; index++) {
			if (strcspn(options[index], "=") != bytes ||
			    strncmp(options[index], unsupported[replacement],
				    bytes) != 0)
				continue;
			const char *previous = options[index];
			options[index] = unsupported[replacement];
			passed = isolated_raw_case(options, count, true, false,
						   true) &&
				 passed;
			options[index] = previous;
			found = true;
		}
		passed = found && passed;
	}
	if (count == TEST_ARGUMENT_CAPACITY) {
		free(storage);
		return false;
	}
	const char *const temperatures[] = { "--temperature=0",
					     "--temperature=-0",
					     "--temperature=0.7" };
	for (size_t index = 0;
	     index < sizeof(temperatures) / sizeof(temperatures[0]); index++) {
		options[count] = temperatures[index];
		/* The CLI's existing number grammar rejects a leading sign. */
		passed = isolated_raw_case(options, count + 1u, true,
					   strcmp(temperatures[index],
						  "--temperature=-0") != 0,
					   true) &&
			 passed;
	}
	free(storage);
	return passed;
}

/* Fresh CLI state in each child: neither the exact axis nor command may override sampling. */
static bool sampling_parser_cases(void)
{
	static const struct {
		const char *temperature_option, *top_p_option, *seed_option;
		float expected_temperature, expected_top_p;
		uint32_t expected_seed;
	} cases[] = { { NULL, NULL, NULL, 0.0f, 1.0f, 0 },
		      { "--temperature=0", "--top_p=1", "--seed=0", 0.0f, 1.0f,
			0 },
		      { "--temperature=0.7", "--top_p=0.4", "--seed=4294967295",
			0.7f, 0.4f, UINT32_MAX },
		      { "--temperature=2", "--top_p=0.001", "--seed=1", 2.0f,
			0.001f, 1 } };
	bool passed = true;
	for (uint32_t exact = 0; exact <= 1; exact++) {
		for (uint32_t serve = 0; serve <= 1; serve++) {
			for (size_t index = 0;
			     index < sizeof(cases) / sizeof(cases[0]);
			     index++) {
				axis_cases_run++;
				const pid_t child = fork();
				if (child == 0) {
					char program[] = "anemone",
					     generating[] = "generate",
					     serving[] = "serve";
					char messages[] = "messages",
					     tables[] = "tables",
					     weight_path[] = "weights";
					char reference[] = "--exactness=exact";
					char *arguments[TEST_ARGUMENT_CAPACITY] = {
						program, serve != 0 ? serving :
								      generating
					};
					int arguments_count = TEST_OPTION_START;
					if (exact != 0)
						arguments[arguments_count++] =
							reference;
					if (cases[index].temperature_option !=
					    NULL) {
						arguments[arguments_count++] = strdup(
							cases[index]
								.temperature_option);
						arguments[arguments_count++] = strdup(
							cases[index]
								.top_p_option);
						arguments[arguments_count++] = strdup(
							cases[index]
								.seed_option);
					}
					if (serve == 0)
						arguments[arguments_count++] =
							messages;
					arguments[arguments_count++] = tables;
					arguments[arguments_count++] =
						weight_path;
					struct axis_arguments_record expanded = {
						0
					};
					const bool accepted =
						axis_expand(arguments_count,
							    arguments,
							    &expanded) &&
						parse_arguments(
							expanded.count,
							expanded.values) &&
						temperature ==
							cases[index]
								.expected_temperature &&
						top_p ==
							cases[index]
								.expected_top_p &&
						seed == cases[index]
								.expected_seed &&
						is_seed_given ==
							(cases[index]
								 .seed_option !=
							 NULL);
					axis_arguments_free(&expanded);
					_exit(accepted ? 0 : 1);
				}
				int status = 0;
				const bool accepted =
					child > 0 &&
					waitpid(child, &status, 0) == child &&
					WIFEXITED(status) &&
					WEXITSTATUS(status) == 0;
				if (!accepted)
					(void)fprintf(
						stderr,
						"sampling parser failed: exact=%u serve=%u case=%zu\n",
						exact, serve, index);
				passed = accepted && passed;
			}
		}
	}
	return passed;
}

enum test_selection_boundary {
	TEST_COMMON_PLAN,
	TEST_LATE_WITH_AXES,
	TEST_LATE_WITHOUT_AXES,
	TEST_SELECTION_BOUNDARIES
};

static bool selection_boundary_case(enum test_selection_boundary boundary)
{
	char program[] = "anemone", generate[] = "generate",
	     placement[] = "--placement=tp4_exl3", code[] = "--category=code",
	     prose[] = "--category=prose", exactness[] = "--exactness=balanced",
	     concurrency[] = "--concurrency=single", focus[] = "--focus=decode",
	     messages[] = "messages", tables[] = "tables",
	     weights[] = "weights", late[] = "--category=vision";
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, generate };
	int count = TEST_OPTION_START;
	if (boundary != TEST_LATE_WITHOUT_AXES) {
		arguments[count++] = placement;
		arguments[count++] = boundary == TEST_COMMON_PLAN ? prose :
								    code;
		arguments[count++] = exactness;
		arguments[count++] = concurrency;
		if (boundary != TEST_COMMON_PLAN)
			arguments[count++] = focus;
	}
	arguments[count++] = messages;
	arguments[count++] = tables;
	arguments[count++] = weights;
	if (boundary != TEST_COMMON_PLAN)
		arguments[count++] = late;
	struct axis_arguments_record expanded = { 0 };
	bool passed = axis_expand(count, arguments, &expanded);
	if (passed && boundary == TEST_LATE_WITHOUT_AXES) {
		passed = expanded.values == arguments &&
			 expanded.count == count && expanded.storage == NULL &&
			 expanded.selected_count == 0u &&
			 expanded.values[expanded.count - 1] == late;
	} else if (passed) {
		struct axis_test_case test = { .choice = expanded.choice,
					       .command = "generate" };
		passed = expanded.choice != NULL &&
			 declared_settings_match(&test, &expanded, count,
						 arguments);
		if (boundary == TEST_COMMON_PLAN) {
			/* Compatible full rows may share a command plan without
			 * assigning a value to the omitted focus axis. */
			passed = passed &&
				 expanded.selected_count == TEST_AXES - 1u &&
				 expanded.selected[TEST_FOCUS] == NULL &&
				 parse_arguments(expanded.count,
						 expanded.values) &&
				 startup_placement_status() ==
					 PROGRAM_SUCCEEDED &&
				 batch_axes_admitted() &&
				 ordinary_tuple_geometry(expanded.choice,
							 "generate");
		} else {
			/* A flag-shaped pack-copy path remains a positional argument;
			 * it neither replaces the earlier category nor disappears. */
			passed = passed &&
				 expanded.selected_count == TEST_AXES &&
				 strcmp(expanded.selected[TEST_CATEGORY],
					"code") == 0 &&
				 expanded.values[expanded.count - 1] == late &&
				 strcmp(expanded.values[expanded.count - 1],
					late) == 0;
		}
	}
	axis_arguments_free(&expanded);
	return passed;
}

static bool isolated_boundary_case(enum test_selection_boundary boundary)
{
	axis_cases_run++;
	const pid_t child = fork();
	if (child == 0)
		_exit(selection_boundary_case(boundary) ? 0 : 1);
	int status = 0;
	const bool passed = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) && WEXITSTATUS(status) == 0;
	if (!passed)
		(void)fprintf(stderr, "failed axis selection boundary %u\n",
			      (unsigned)boundary);
	return passed;
}

/* A shorter evidence string can be a prefix of a scoped one. Match complete
 * help lines so printing only the longer, first tuple's caveat cannot pass. */
static bool scoped_help_case(const struct axis_choice *choices, size_t count)
{
	axis_cases_run++;
	char *text = NULL;
	size_t bytes = 0;
	FILE *stream = open_memstream(&text, &bytes);
	if (stream == NULL)
		return false;
	bool passed = help_axes(stream) == 0;
	passed = fclose(stream) == 0 && passed;
	if (text == NULL || bytes == 0u) {
		free(text);
		return false;
	}
	const char *const category_meaning =
		"--category=code|prose|structured|vision|balanced\n"
		"    Workload kind; balanced is a mixed code/prose/structured measurement, with code winning ties.\n";
	const char *const concurrency_meaning =
		"--concurrency=single|multiple|balanced\n"
		"    C1/THIN, C2-C8/BATCH, or one measured compromise across both.\n";
	passed = passed && bytes > 0u &&
		 strstr(text, category_meaning) != NULL &&
		 strstr(text, concurrency_meaning) != NULL &&
		 strstr(text, "--exactness=") != NULL &&
		 strstr(text, "--focus=") != NULL &&
		 strstr(text, "--profile=") == NULL;
	size_t distinct = 0;
	for (size_t index = 0; index < count; index++) {
		bool repeated = false;
		for (size_t prior = 0; !repeated && prior < index; prior++)
			repeated = strcmp(choices[index].evidence,
					  choices[prior].evidence) == 0;
		if (repeated)
			continue;
		distinct++;
		const size_t capacity = strlen(choices[index].evidence) +
					sizeof("  evidence: \n");
		char *expected = malloc(capacity);
		if (expected == NULL) {
			passed = false;
			break;
		}
		const int written = snprintf(expected, capacity,
					     "  evidence: %s\n",
					     choices[index].evidence);
		const bool present = written >= 0 &&
				     (size_t)written < capacity &&
				     strstr(text, expected) != NULL;
		if (!present)
			(void)fprintf(
				stderr,
				"help omitted scoped evidence for %s/%s/%s/%s (%s)\n",
				choices[index].value[TEST_CATEGORY],
				choices[index].value[TEST_EXACTNESS],
				choices[index].value[TEST_CONCURRENCY],
				choices[index].value[TEST_FOCUS],
				choices[index].weight_family);
		passed = present && passed;
		free(expected);
	}
	passed = distinct > 1u && passed;
	free(text);
	return passed;
}

static bool prompt_token_ids_cases(void)
{
	static const uint32_t expected[] = { 0u,   128803u, 19417u, 566u,
					     345u, 11592u,  17437u, 16u };
	const char valid[] = "[0,128803,19417,566,345,11592,17437,16]";
	uint32_t tokens[TEST_REFERENCE_INPUT_CAPACITY] = { 0 };
	uint32_t count = UINT32_MAX;
	bool passed = prompt_token_ids_parse(valid, strlen(valid),
					     TEST_REFERENCE_VOCABULARY, tokens,
					     TEST_REFERENCE_INPUT_CAPACITY,
					     &count) == STATUS_OK &&
		      count == sizeof(expected) / sizeof(expected[0]) &&
		      memcmp(tokens, expected, sizeof(expected)) == 0;
	const char *const invalid[] = {
		"",	   "[]",       "{}",	       "null",
		"[false]", "[null]",   "[[0]]",	       "[\"0\"]",
		"[-1]",	   "[-0]",     "[0.0]",	       "[1e0]",
		"[NaN]",   "[129280]", "[4294967295]", "[4294967296]",
		"[0,]",	   "[0] [1]"
	};
	for (size_t index = 0; index < sizeof(invalid) / sizeof(invalid[0]);
	     index++) {
		count = UINT32_MAX;
		passed = prompt_token_ids_parse(invalid[index],
						strlen(invalid[index]),
						TEST_REFERENCE_VOCABULARY,
						tokens,
						TEST_REFERENCE_INPUT_CAPACITY,
						&count) != STATUS_OK &&
			 count == UINT32_MAX && passed;
	}
	count = UINT32_MAX;
	passed = prompt_token_ids_parse(valid, strlen(valid),
					TEST_REFERENCE_VOCABULARY, tokens,
					TEST_REFERENCE_INPUT_CAPACITY - 1u,
					&count) == STATUS_UNSUPPORTED &&
		 count == UINT32_MAX && passed;
	passed = prompt_token_ids_parse(valid, strlen(valid),
					TEST_REFERENCE_VOCABULARY, tokens, 1u,
					&count) == STATUS_INVALID_ARGUMENT &&
		 passed;
	passed = prompt_token_ids_parse(valid, strlen(valid),
					TEST_REFERENCE_VOCABULARY, NULL,
					TEST_REFERENCE_INPUT_CAPACITY,
					&count) == STATUS_INVALID_ARGUMENT &&
		 passed;
	return passed;
}

enum raw_token_cli_case {
	RAW_TOKEN_VALID,
	RAW_TOKEN_EXTRA_MESSAGES,
	RAW_TOKEN_MISSING_PACK,
	RAW_TOKEN_DUPLICATE,
	RAW_TOKEN_SERVE,
	RAW_TOKEN_EMPTY,
	RAW_TOKEN_IGNORE_END,
	RAW_TOKEN_IGNORE_END_INVALID,
	RAW_TOKEN_CLI_CASES
};

static bool raw_token_cli_case(enum raw_token_cli_case test)
{
	char program[] = "anemone", generate[] = "generate", serve[] = "serve",
	     input[] = "--prompt_token_ids=fixture.json",
	     empty[] = "--prompt_token_ids=", ignore[] = "--ignore_eos=1",
	     invalid_ignore[] = "--ignore_eos=2", messages[] = "messages",
	     tables[] = "tables", weights[] = "weights";
	char *arguments[TEST_ARGUMENT_CAPACITY] = {
		program, test == RAW_TOKEN_SERVE ? serve : generate
	};
	int count = TEST_OPTION_START;
	arguments[count++] = test == RAW_TOKEN_EMPTY ? empty : input;
	if (test == RAW_TOKEN_DUPLICATE)
		arguments[count++] = input;
	if (test == RAW_TOKEN_IGNORE_END ||
	    test == RAW_TOKEN_IGNORE_END_INVALID)
		arguments[count++] =
			test == RAW_TOKEN_IGNORE_END ? ignore : invalid_ignore;
	if (test == RAW_TOKEN_EXTRA_MESSAGES)
		arguments[count++] = messages;
	arguments[count++] = tables;
	if (test != RAW_TOKEN_MISSING_PACK)
		arguments[count++] = weights;
	const bool expected = test == RAW_TOKEN_VALID ||
			      test == RAW_TOKEN_IGNORE_END;
	const bool parsed = parse_arguments(count, arguments);
	return parsed == expected &&
	       (!parsed ||
		(prompt_token_ids_path != NULL && copy_count == 1u &&
		 strcmp(front_end_paths[MESSAGES_FILE], "fixture.json") == 0 &&
		 strcmp(front_end_paths[TABLES_FILE], "tables") == 0 &&
		 strcmp(copy_paths[0], "weights") == 0 &&
		 ignore_eos == (test == RAW_TOKEN_IGNORE_END ? 1u : 0u)));
}

static bool isolated_raw_token_cli_case(enum raw_token_cli_case test)
{
	axis_cases_run++;
	const pid_t child = fork();
	if (child == 0)
		_exit(raw_token_cli_case(test) ? 0 : 1);
	int status = 0;
	return child > 0 && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/* The matrix job42 recipe is kept raw: no axis tuple supplies an explicit draft policy or the new guard. Dummy
 * positional files are never opened by these parser/startup checks. Each scenario forks before resolving defaults. */
static const char *const prompt_chunks_job42_options[] = {
	"--placement=tp4_exl3",
	"--numerics=relaxed",
	"--tp_wire_bf16=1",
	"--cmp_order=reference",
	"--linear_order=1",
	"--linear_profile=0",
	"--pro_emulation=0",
	"--exl3_expert_parts=4",
	"--exl3_tc_moe=1",
	"--tp_head_mma=1",
	"--tp_trunk_mma=1",
	"--tp_ordered_wide=1",
	"--tp_head_shards=1",
	"--tp_head_stream=1",
	"--tp_draft_shards=1",
	"--rows=8",
	"--full_steps=1",
	"--prefill_cone=1",
	"--chunks_in_flight=1",
	"--step_graphs=1",
	"--speculative=2",
	"--draft=dspark",
	"--draft_block=5",
	"--draft_positions=5",
	"--card_memory_kept_free_mib=0",
	"--wide_rows=1024",
	"--batch_rows=8,8,12,12,16,24,32,48",
	"--batch_verifiers=1,2,2,4,4,4,8,8",
	"--batch_wide_head=1",
	"--batch_draft_rows=1",
	"--tp_draft_layers=1",
	"--session_slots=8",
	"--context_tokens=65536",
	"--prefix_reuse=0",
	"--prefill_first=0"
};

enum prompt_chunks_recipe {
	PROMPT_CHUNKS_BATCH,
	PROMPT_CHUNKS_THIN,
	PROMPT_CHUNKS_PP5
};
struct prompt_chunks_case {
	const char *name;
	const char *command;
	enum prompt_chunks_recipe recipe;
	const char *replace[TEST_EXTRA_CAPACITY];
	bool parse_expected;
	uint32_t selected;
};

static bool prompt_chunks_option_matches(const char *first, const char *second)
{
	const size_t bytes = strcspn(first, "=");
	return bytes == strcspn(second, "=") &&
	       strncmp(first, second, bytes) == 0;
}

static bool parse_prompt_chunks_case(const struct prompt_chunks_case *test,
				     bool reverse)
{
	char program[] = "anemone", serve[] = "serve", generate[] = "generate",
	     messages[] = "messages", tables[] = "tables",
	     weights[] = "weights", original[] = "--placement=pp5",
	     single[] = "--session_slots=1";
	char *arguments[TEST_ARGUMENT_CAPACITY] = {
		program, strcmp(test->command, "serve") == 0 ? serve : generate
	};
	char *copies[TEST_ARGUMENT_CAPACITY] = { 0 };
	int count = TEST_OPTION_START;
	if (test->recipe == PROMPT_CHUNKS_PP5) {
		arguments[count++] = original;
	} else {
		const size_t options = sizeof(prompt_chunks_job42_options) /
				       sizeof(prompt_chunks_job42_options[0]);
		for (size_t turn = 0; turn < options; turn++) {
			const size_t index = reverse ? options - 1u - turn :
						       turn;
			const char *option = prompt_chunks_job42_options[index];
			bool replaced = false;
			for (size_t extra = 0; extra < TEST_EXTRA_CAPACITY;
			     extra++)
				replaced =
					replaced ||
					(test->replace[extra] != NULL &&
					 prompt_chunks_option_matches(
						 option, test->replace[extra]));
			if (test->recipe == PROMPT_CHUNKS_THIN &&
			    (strncmp(option, "--batch_", strlen("--batch_")) ==
				     0 ||
			     strncmp(option, "--session_slots=",
				     strlen("--session_slots=")) == 0))
				replaced = true;
			/* Generate inherits its single session and has no serving/cache scheduler options. */
			if (strcmp(test->command, "generate") == 0 &&
			    (prompt_chunks_option_matches(option,
							  "--prefix_reuse=") ||
			     prompt_chunks_option_matches(option,
							  "--prefill_first=") ||
			     prompt_chunks_option_matches(option,
							  "--session_slots=")))
				replaced = true;
			if (replaced)
				continue;
			copies[count] = strdup(option);
			if (copies[count] == NULL)
				goto allocation_failed;
			arguments[count] = copies[count];
			count++;
		}
		if (test->recipe == PROMPT_CHUNKS_THIN &&
		    strcmp(test->command, "serve") == 0)
			arguments[count++] = single;
	}
	for (size_t extra = 0; extra < TEST_EXTRA_CAPACITY; extra++) {
		if (test->replace[extra] == NULL)
			continue;
		copies[count] = strdup(test->replace[extra]);
		if (copies[count] == NULL)
			goto allocation_failed;
		arguments[count] = copies[count];
		count++;
	}
	if (strcmp(test->command, "generate") == 0)
		arguments[count++] = messages;
	arguments[count++] = tables;
	arguments[count++] = weights;
	struct axis_arguments_record expanded = { 0 };
	bool passed = axis_expand(count, arguments, &expanded);
	if (passed) {
		passed = expanded.choice == NULL &&
			 expanded.values == arguments;
		const bool parsed =
			parse_arguments(expanded.count, expanded.values);
		passed = passed && parsed == test->parse_expected;
		if (parsed) {
			passed = passed &&
				 prefill_prompt_chunks == test->selected &&
				 startup_placement_status() ==
					 PROGRAM_SUCCEEDED &&
				 batch_axes_admitted();
			if (test->selected != 0 && test->replace[0] == NULL)
				passed =
					passed && prefill_first == 0 &&
					prefill_order == 0 &&
					prefill_decode_rows == 0 &&
					batch_core_once ==
						TEST_BATCH_POOL_DEFERRED_CORE &&
					draft_policy.depth_policy ==
						DRAFT_DEPTH_RATE &&
					memcmp(batch_step_us,
					       default_batch_step_us,
					       sizeof(batch_step_us)) == 0;
		}
	}
	axis_arguments_free(&expanded);
	for (size_t index = 0; index < TEST_ARGUMENT_CAPACITY; index++)
		free(copies[index]);
	return passed;
allocation_failed:
	for (size_t index = 0; index < TEST_ARGUMENT_CAPACITY; index++)
		free(copies[index]);
	return false;
}

static bool isolated_prompt_chunks_cases(void)
{
	static const struct prompt_chunks_case cases[] = {
		{ "job42 omitted guard",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { NULL },
		  true,
		  0 },
		{ "explicit mixing fallback",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_prompt_chunks=0" },
		  true,
		  0 },
		{ "explicit guard",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_prompt_chunks=1" },
		  true,
		  1 },
		{ "reserved scheduler",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_first=1" },
		  true,
		  0 },
		{ "value outside public range",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_prompt_chunks=2" },
		  false,
		  0 },
		{ "duplicate guard",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_prompt_chunks=0", "--prefill_prompt_chunks=1" },
		  false,
		  0 },
		{ "old core",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--batch_core_once=2" },
		  true,
		  0 },
		{ "alternate menu",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--batch_rows=48", "--batch_verifiers=8" },
		  true,
		  0 },
		{ "THIN unaffected",
		  "serve",
		  PROMPT_CHUNKS_THIN,
		  { NULL },
		  true,
		  0 },
		{ "PP5 unaffected",
		  "serve",
		  PROMPT_CHUNKS_PP5,
		  { NULL },
		  true,
		  0 },
		{ "generate unaffected",
		  "generate",
		  PROMPT_CHUNKS_THIN,
		  { NULL },
		  true,
		  0 },
		{ "explicit RATE policy",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--draft_policy=2" },
		  true,
		  0 },
		{ "row-owned sums",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--tp_ring_column_owners=0" },
		  true,
		  0 },
		{ "prefill cap",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_chunk_rows=256" },
		  true,
		  0 },
		{ "decode cap",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_decode_rows=256" },
		  true,
		  0 },
		{ "tail disabled",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_tail_rows=0" },
		  true,
		  0 },
		{ "front tail disabled",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_front_tail=0" },
		  true,
		  0 },
		{ "thin tail enabled",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_thin_tail=1" },
		  true,
		  0 },
		{ "ordered prompts",
		  "serve",
		  PROMPT_CHUNKS_BATCH,
		  { "--prefill_order=2" },
		  true,
		  0 }
	};
	bool passed = true;
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		for (unsigned order = 0; order < TEST_OPTION_ORDERS; order++) {
			axis_cases_run++;
			const pid_t child = fork();
			if (child == 0)
				_exit(parse_prompt_chunks_case(cases + index,
							       order != 0) ?
					      0 :
					      1);
			int status = 0;
			const bool child_passed =
				child > 0 &&
				waitpid(child, &status, 0) == child &&
				WIFEXITED(status) && WEXITSTATUS(status) == 0;
			if (!child_passed)
				(void)fprintf(
					stderr,
					"failed prompt chunk defaults: %s (reverse%u)\n",
					cases[index].name, order);
			passed = child_passed && passed;
		}
	}
	return passed;
}

int main(void)
{
	size_t count = 0;
	const struct axis_choice *choices = axis_choices_entry(&count);
	const size_t categories =
		sizeof(category_values) / sizeof(category_values[0]);
	const size_t exactnesses =
		sizeof(exactness_values) / sizeof(exactness_values[0]);
	const size_t concurrencies =
		sizeof(concurrency_values) / sizeof(concurrency_values[0]);
	const size_t focuses = sizeof(focus_values) / sizeof(focus_values[0]);
	bool passed = prompt_token_ids_cases() &&
		      count == TEST_WEIGHT_FAMILIES * categories * exactnesses *
				       concurrencies * focuses;
	for (size_t index = 0; index < count; index++) {
		const struct axis_choice *choice = &choices[index];
		const bool available = choice->hold[0] == '\0';
		/* A selector is never a substitute for a missing numerical/runtime
		 * dependency or an unmeasured lossy-error budget. */
		if ((strcmp(choice->weight_family, "original") == 0 &&
		     !original_decode_candidate(choice) &&
		     !original_reference_candidate(choice)) ||
		    (strcmp(choice->value[TEST_EXACTNESS], "balanced") != 0 &&
		     !original_reference_candidate(choice)) ||
		    strcmp(choice->value[TEST_CATEGORY], "vision") == 0 ||
		    (strcmp(choice->value[TEST_CONCURRENCY], "balanced") == 0 &&
		     !all_balanced_candidate(choice)))
			passed = passed && !available;
		struct axis_test_case test = { .choice = choice,
					       .command = "generate",
					       .expand_expected = available,
					       .parse_expected = available,
					       .placement_expected = available,
					       .batch_expected = available };
		passed = isolated_tuple_case(&test) && passed;
		test.command = "serve";
		test.extras_first = true;
		passed = isolated_tuple_case(&test) && passed;
	}
	const char *const families[] = { "original", "exl3" };
	for (size_t family = 0; family < TEST_WEIGHT_FAMILIES; family++)
		for (size_t category = 0; category < categories; category++)
			for (size_t exactness = 0; exactness < exactnesses;
			     exactness++)
				for (size_t concurrency = 0;
				     concurrency < concurrencies; concurrency++)
					for (size_t focus = 0; focus < focuses;
					     focus++) {
						unsigned matches = 0;
						for (size_t index = 0;
						     index < count; index++)
							matches +=
								strcmp(choices[index]
									       .weight_family,
								       families[family]) ==
											0 &&
										strcmp(choices[index]
											       .value[TEST_CATEGORY],
										       category_values
											       [category]) ==
											0 &&
										strcmp(choices[index]
											       .value[TEST_EXACTNESS],
										       exactness_values
											       [exactness]) ==
											0 &&
										strcmp(choices[index]
											       .value[TEST_CONCURRENCY],
										       concurrency_values
											       [concurrency]) ==
											0 &&
										strcmp(choices[index]
											       .value[TEST_FOCUS],
										       focus_values
											       [focus]) ==
											0 ?
									1u :
									0u;
						passed = passed &&
							 matches == 1u;
					}
	const struct axis_choice *single = find_choice(
		choices, count, "code", "balanced", "single", "decode", "exl3");
	const struct axis_choice *multiple = find_choice(choices, count, "code",
							 "balanced", "multiple",
							 "decode", "exl3");
	const struct axis_choice *adaptive =
		find_choice(choices, count, "structured", "balanced",
			    "multiple", "prefill", "exl3");
	if (single == NULL || multiple == NULL || adaptive == NULL) {
		passed = false;
	} else {
		const struct axis_test_case overrides[] = {
			{ adaptive,
			  "serve",
			  { "--batch_step_us=0" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_BATCH_MENU_OFF },
			{ adaptive,
			  "generate",
			  { "--batch_step_us=0" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_BATCH_MENU_OFF },
			{ adaptive,
			  "serve",
			  { "--batch_step_us=19500,0" },
			  false,
			  true,
			  false,
			  true,
			  true,
			  TEST_UNCHANGED },
			{ single,
			  "serve",
			  { "--wide_rows=1024" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_WIDE_ROWS },
			{ single,
			  "generate",
			  { "--draft_policy=1" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_DRAFT_POLICY },
			{ single,
			  "generate",
			  { "--speculative_depth=6", "--draft_block=7" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_DRAFT_DEPTH },
			{ single,
			  "generate",
			  { "--draft=lookup" },
			  false,
			  true,
			  true,
			  false,
			  true,
			  TEST_SPECULATIVE_LOOKUP },
			{ single,
			  "generate",
			  { "--speculative=0" },
			  false,
			  true,
			  true,
			  false,
			  true,
			  TEST_SPECULATIVE_OFF },
			{ single,
			  "serve",
			  { "--numerics=exact" },
			  false,
			  true,
			  true,
			  false,
			  true,
			  TEST_UNCHANGED },
			{ single,
			  "serve",
			  { "--pro_emulation=1" },
			  false,
			  true,
			  true,
			  false,
			  true,
			  TEST_UNCHANGED },
			{ single,
			  "serve",
			  { "--wide_rows=1024", "--wide_rows=2048" },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED },
			{ multiple,
			  "serve",
			  { "--pipeline_steps=2" },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED },
			{ multiple,
			  "serve",
			  { "--batch_draft_rows=0" },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED },
			{ multiple,
			  "serve",
			  { "--session_slots=1" },
			  false,
			  true,
			  true,
			  true,
			  false,
			  TEST_UNCHANGED },
			{ multiple,
			  "generate",
			  { "--batch_verifiers=8" },
			  false,
			  true,
			  true,
			  true,
			  false,
			  TEST_UNCHANGED },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rows=0" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_DECODE_ROWS },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rounds=4" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_DECODE_ROUNDS },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rows=128" },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rows=1024",
			    "--chunks_in_flight=2" },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rows=1024",
			    "--prefill_fast_plan=1" },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED },
			{ single,
			  "load",
			  { NULL },
			  false,
			  false,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED }
		};
		for (size_t index = 0;
		     index < sizeof(overrides) / sizeof(overrides[0]); index++)
			for (unsigned order = 0; order < TEST_OPTION_ORDERS;
			     order++) {
				struct axis_test_case test = overrides[index];
				test.extras_first = order != 0u;
				passed = isolated_tuple_case(&test) && passed;
			}
	}
	const struct axis_choice *original_single_choice =
		find_choice(choices, count, "code", "balanced", "single",
			    "decode", "original");
	const struct axis_choice *original_multiple_choice =
		find_choice(choices, count, "code", "balanced", "multiple",
			    "decode", "original");
	if (original_single_choice == NULL ||
	    original_multiple_choice == NULL) {
		passed = false;
	} else {
		const struct axis_test_case layout_cases[] = {
			{ original_multiple_choice,
			  "serve",
			  { "--layout=" TEST_LAYOUT_OVERRIDE },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_LAYOUT },
			{ original_multiple_choice,
			  "generate",
			  { "--layout=" TEST_LAYOUT_OVERRIDE },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_LAYOUT },
			{ original_multiple_choice,
			  "serve",
			  { "--placement=tp4_pro" },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED },
			{ original_single_choice,
			  "serve",
			  { "--layout=" TEST_LAYOUT_OVERRIDE },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED }
		};
		for (size_t index = 0;
		     index < sizeof(layout_cases) / sizeof(layout_cases[0]);
		     index++)
			for (unsigned order = 0; order < TEST_OPTION_ORDERS;
			     order++) {
				struct axis_test_case test =
					layout_cases[index];
				test.extras_first = order != 0u;
				passed = isolated_tuple_case(&test) && passed;
			}
	}
	passed = isolated_raw_case(NULL, 0u, true, true, true) && passed;
	for (enum raw_token_cli_case test = RAW_TOKEN_VALID;
	     test < RAW_TOKEN_CLI_CASES; test++)
		passed = isolated_raw_token_cli_case(test) && passed;
	const char *trace_zero[] = { "--draft_trace=0" };
	const char *trace_one[] = { "--draft_trace=1" };
	const char *trace_two[] = { "--draft_trace=2" };
	passed = isolated_raw_case(trace_zero, 1u, true, true, true) && passed;
	passed = isolated_raw_case(trace_one, 1u, true,
				   TIMING_HOOKS_ENABLED != 0, true) &&
		 passed;
	passed = isolated_raw_case(trace_two, 1u, true, false, true) && passed;
	const char *const removed[] = { "--profile=balanced" };
	passed = isolated_raw_case(removed, 1u, true, false, true) && passed;
	const char *const malformed[] = {
		"--category=",	      "--category=CODE",
		"--category=unknown", "--exactness=reference",
		"--exactness=",	      "--concurrency=c8",
		"--concurrency=",     "--focus=fast",
		"--focus=Decode",     "--focus="
	};
	for (size_t index = 0; index < sizeof(malformed) / sizeof(malformed[0]);
	     index++)
		passed = isolated_raw_case(&malformed[index], 1u, false, false,
					   false) &&
			 passed;
	const char *const duplicate_same[] = { "--category=code",
					       "--category=code" };
	const char *const duplicate_different[] = { "--exactness=balanced",
						    "--exactness=fast" };
	passed = isolated_raw_case(duplicate_same,
				   sizeof(duplicate_same) /
					   sizeof(duplicate_same[0]),
				   false, false, false) &&
		 passed;
	passed = isolated_raw_case(duplicate_different,
				   sizeof(duplicate_different) /
					   sizeof(duplicate_different[0]),
				   false, false, false) &&
		 passed;
	const char *const exact_partial[] = { "--exactness=exact" };
	const char *const fast_partial[] = { "--exactness=fast" };
	const struct axis_choice *reference =
		find_choice(choices, count, "code", "exact", "single", "decode",
			    "original");
	passed = reference_raw_recipe_cases(reference) && passed;
	const bool reference_available = reference != NULL &&
					 reference->hold[0] == '\0';
	if (reference != NULL) {
		const char *const unsupported[] = { "--prefix_reuse=1",
						    "--prefix_reuse_exact=1",
						    "--session_fork=1" };
		for (size_t index = 0;
		     index < sizeof(unsupported) / sizeof(unsupported[0]);
		     index++) {
			const struct axis_test_case test = {
				.choice = reference,
				.command = "serve",
				.extra = { unsupported[index] }
			};
			passed = isolated_tuple_case(&test) && passed;
		}
	}
	passed = isolated_raw_case(exact_partial, 1u, reference_available,
				   reference_available, false) &&
		 passed;
	const char *const unsupported_reference[][TEST_EXTRA_CAPACITY] = {
		{ "--exactness=exact", "--concurrency=multiple" },
		{ "--exactness=exact", "--concurrency=balanced" },
		{ "--exactness=exact", "--category=vision" },
		{ "--exactness=exact", "--placement=tp4_exl3" },
		{ "--exactness=exact", "--official_reference=0" },
		{ "--exactness=exact", "--official_hc_rows=0" },
		{ "--exactness=exact", "--rows=8" },
		{ "--exactness=exact", "--wide_rows=1024" },
		{ "--exactness=exact", "--placement=tp4_pro" },
		{ "--exactness=exact", "--speculative=2" },
		{ "--exactness=exact", "--draft=2" },
		{ "--exactness=exact", "--pro_emulation=1" }
	};
	for (size_t index = 0; index < sizeof(unsupported_reference) /
					       sizeof(unsupported_reference[0]);
	     index++)
		passed = isolated_raw_case(unsupported_reference[index],
					   TEST_REFERENCE_SELECTOR_OPTIONS,
					   false, false, false) &&
			 passed;
	const char *const matching_reference[] = { "--exactness=exact",
						   "--official_reference=1",
						   "--rows=1" };
	passed = isolated_raw_case(matching_reference,
				   sizeof(matching_reference) /
					   sizeof(matching_reference[0]),
				   reference_available, reference_available,
				   false) &&
		 passed;
	const char *const sampled_reference[] = { "--exactness=exact",
						  "--temperature=0.7" };
	passed = isolated_raw_case(sampled_reference,
				   sizeof(sampled_reference) /
					   sizeof(sampled_reference[0]),
				   reference_available, reference_available,
				   false) &&
		 passed;
	const char *const greedy_reference[] = { "--exactness=exact",
						 "--temperature=0" };
	passed = isolated_raw_case(
			 greedy_reference,
			 sizeof(greedy_reference) / sizeof(greedy_reference[0]),
			 reference_available, reference_available, false) &&
		 passed;
	const char *const ordinary_sampling[] = { "--temperature=0.7" };
	passed = isolated_raw_case(ordinary_sampling, 1u, true, true, true) &&
		 passed;
	const char *const bad_reference[] = { "--official_reference=2" };
	passed = isolated_raw_case(bad_reference, 1u, true, false, true) &&
		 passed;
	passed = isolated_raw_case(fast_partial, 1u, false, false, false) &&
		 passed;
	const char *const inherited_original[] = { "--category=code",
						   "--exactness=balanced",
						   "--concurrency=single",
						   "--focus=decode" };
	const char *const original_tp[] = {
		"--category=code", "--exactness=balanced",
		"--concurrency=single", "--focus=decode", "--placement=tp4_pro"
	};
	passed = isolated_raw_case(inherited_original,
				   sizeof(inherited_original) /
					   sizeof(inherited_original[0]),
				   true, true, false) &&
		 passed;
	passed = isolated_raw_case(original_tp,
				   sizeof(original_tp) / sizeof(original_tp[0]),
				   true, true, false) &&
		 passed;
	const char *const ambiguous_partial[] = { "--category=code",
						  "--placement=tp4_exl3" };
	passed = isolated_raw_case(ambiguous_partial,
				   sizeof(ambiguous_partial) /
					   sizeof(ambiguous_partial[0]),
				   false, false, false) &&
		 passed;
	for (enum test_selection_boundary boundary = TEST_COMMON_PLAN;
	     boundary < TEST_SELECTION_BOUNDARIES; boundary++)
		passed = isolated_boundary_case(boundary) && passed;

	passed = sampling_parser_cases() && passed;
	passed = isolated_prompt_chunks_cases() && passed;
	passed = scoped_help_case(choices, count) && passed;
	(void)printf(
		"startup axes: %s (%zu table rows, %u cases, real parser/startup/verifier gates, no cards)\n",
		passed ? "PASS" : "FAIL", count, axis_cases_run);
	return passed ? 0 : 1;
}
