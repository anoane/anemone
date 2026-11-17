/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Exercise real CLI parsing and admission without opening a model or a card.
 * Each child starts from pristine commands.c defaults. */
#include <fcntl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/wait.h>
#include "anemone/head_offload_records.h"
#include "step_driver.h"
#include "backend_cuda_kernel_images.h"
#include "engram_pack.h"
#include "model_v41_pack.h"
#include "anemone/hooks.h"
#include "serving_cursor.h"
#include "engram_step.h"
#include "model_v41_tp2_model.h"
#include "commands.h"

#define TEST_ARGUMENT_CAPACITY 96u
#define TEST_OPTION_BYTES 96u
#define TEST_LOG_CAPACITY 16384u
#define TEST_OPTION_START 2u
#define TEST_EXTRA_CAPACITY 3u
#define TEST_SPLITS 3u
#define TEST_SPLIT_NAMES \
	"pp5, tp4_pro, tp4_exl3" /* the tables, as a refusal names them */
#define TEST_PLACEMENT_BYTES 64u
#define TEST_BATCH_FIRST_STEP_US 19500u
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
#define TEST_BANKED_PIPELINE_STEPS 5u
#define TEST_BANKED_SESSION_ACCEPTANCE 6u
/* Stage 2: category=code's draft depth and focus=decode's arm 1 and rows
 * (TP4_PRO) on the Original splits. */
#define TEST_CODE_DRAFT_DEPTH 7u
#define TEST_DECODE_SESSION_ACCEPTANCE 6u
#define TEST_TP_DECODE_ROWS 6u
/* Stage 3: PP5 concurrency=single's rows (r7c g3). Unbounded sessions: the owner's rule is the raw defaults, 1M context
 * per session on every row an exact recipe does not pin, and on serve the most slots that fit
 * (SESSION_SLOTS_MAX_PARSED at parse, the placement fit at planning); no axis set pins either. */
#define TEST_PP5_SINGLE_ROWS 6u
#define TEST_OWNER_CONTEXT_TOKENS 1048576u
/* TP4_EXL3 concurrency=multiple's BATCH set on serve: only at exactly 8 slots, as measured at 64K. */
#define TEST_BATCH_SESSIONS 8u
#define TEST_BATCH_CONTEXT_TOKENS 65536u
#define TEST_BATCH_CONTEXT \
	"--context_tokens=65536" /* the BATCH row as measured */
#define TEST_PIPE_ENDS 2u /* a pipe's two descriptors */
#define TEST_FIT_LINE_MOST \
	11u /* an explicit count's fit line: the most that fit, above the 8 it serves */
#define TEST_SLOTS_OPTION "--session_slots="
#define TEST_BATCH_SLOTS "--session_slots=8"
/* Stage 3's raw-default switch (nothing inserted), the option that names a split, and the split without it. */
#define TEST_AXES_OFF "--axes=off"
#define TEST_PLACEMENT_PREFIX "--placement="
#define TEST_DEFAULT_SPLIT "pp5"

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
static const char *const split_names[TEST_SPLITS] = { "pp5", "tp4_pro",
						      "tp4_exl3" };
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
	TEST_BATCH_MENU_OFF,
	TEST_HEAD_OFF,
	TEST_TP_METRICS_ON,
	TEST_SESSION_OVERRIDE,
	TEST_BATCH_AT_EIGHT,
	TEST_BATCH_GEOMETRY /* TEST_BATCH_AT_EIGHT, and the row's whole geometry (ordinary_tuple_geometry) */
};

struct axis_test_case {
	const struct axis_choice *choice;
	const char *command;
	const char *extra[TEST_EXTRA_CAPACITY];
	bool extras_first, expand_expected, parse_expected;
	bool placement_expected, batch_expected;
	enum test_destination destination;
	const char *placement; /* NULL: the default GPU split */
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

/* The row whose settings a case inserts: on serve, a row whose set needs one slot count (TP4_EXL3 BATCH's 8) runs its
 * concurrency=balanced row unless the case gives exactly that --session_slots. */
static const struct axis_choice *
inserted_choice(const struct axis_test_case *test)
{
	const struct axis_choice *const choice = test->choice;
	if (strcmp(test->command, "serve") != 0 ||
	    choice->serve_slots_exactly == 0u)
		return choice;
	char exact[TEST_OPTION_BYTES];
	(void)snprintf(exact, sizeof(exact), TEST_SLOTS_OPTION "%u",
		       choice->serve_slots_exactly);
	for (size_t index = 0; index < TEST_EXTRA_CAPACITY; index++)
		if (test->extra[index] != NULL &&
		    strcmp(test->extra[index], exact) == 0)
			return choice;
	size_t count = 0;
	const struct axis_choice *const choices = axis_choices_entry(&count);
	for (size_t index = 0; index < count; index++) {
		bool same = strcmp(choices[index].gpu_split,
				   choice->gpu_split) == 0 &&
			    strcmp(choices[index].value[TEST_CONCURRENCY],
				   "balanced") == 0;
		for (size_t axis = 0; same && axis < TEST_AXES; axis++)
			same = axis == TEST_CONCURRENCY ||
			       strcmp(choices[index].value[axis],
				      choice->value[axis]) == 0;
		if (same)
			return &choices[index];
	}
	return NULL;
}

/* The log's "axes set" lines that carry exactly this setting (a clamp note
 * may name the bounding setting again). */
static unsigned set_lines(const char *text, const char *setting)
{
	static const char prefix[] = "axes set ";
	const size_t prefix_bytes = strlen(prefix), bytes = strlen(setting);
	unsigned count = 0;
	for (const char *found = strstr(text, setting); found != NULL;
	     found = strstr(found + bytes, setting))
		if ((size_t)(found - text) >= prefix_bytes &&
		    strncmp(found - prefix_bytes, prefix, prefix_bytes) == 0 &&
		    found[bytes] == '\n')
			count++;
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
	const struct axis_choice *const row = inserted_choice(test);
	if (row == NULL) {
		free(log_text);
		return false;
	}
	char split_line[TEST_OPTION_BYTES];
	const int split_written = snprintf(
		split_line, sizeof(split_line),
		"axes gpu_split=%s tuple=%s/%s/%s/%s\n", row->gpu_split,
		row->value[TEST_CATEGORY], row->value[TEST_EXACTNESS],
		row->value[TEST_CONCURRENCY], row->value[TEST_FOCUS]);
	passed = passed && log_text != NULL && split_written > 0 &&
		 (size_t)split_written < sizeof(split_line) &&
		 strstr(log_text, split_line) != NULL;
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
	const char *base = !serving && row->generate_settings[0] != '\0' ?
				   row->generate_settings :
				   row->settings;
	char *groups[] = { strdup(base),
			   strdup(serving ? row->serve_settings : "") };
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
				 set_lines(log_text, setting) == 1u;
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

static bool row_value(const struct axis_choice *choice, enum test_axis axis,
		      const char *value)
{
	return strcmp(choice->value[axis], value) == 0;
}

/* Which rows the registry must hold: vision everywhere, exact outside the PP5
 * reference's one session and multiple on the one-slot TP4_PRO split. */
static bool row_held(const struct axis_choice *choice)
{
	const bool exact = row_value(choice, TEST_EXACTNESS, "exact");
	/* exact runs PP5's reference and TP4_EXL3's exact shaping, one session each */
	return row_value(choice, TEST_CATEGORY, "vision") ||
	       (exact && (strcmp(choice->gpu_split, "tp4_pro") == 0 ||
			  !row_value(choice, TEST_CONCURRENCY, "single"))) ||
	       (strcmp(choice->gpu_split, "tp4_pro") == 0 &&
		row_value(choice, TEST_CONCURRENCY, "multiple"));
}

/* An Original row on a measured recipe's parsed configuration: PP5 multiple's recipe, or TP4_PRO single and balanced,
 * whose base is original_tp_serial2's (raw-default pins omitted, which raw_default_cases proves parse the same). */
static bool original_recipe_row(const struct axis_choice *choice)
{
	return !row_value(choice, TEST_EXACTNESS, "exact") &&
	       ((strcmp(choice->gpu_split, "pp5") == 0 &&
		 row_value(choice, TEST_CONCURRENCY, "multiple")) ||
		(strcmp(choice->gpu_split, "tp4_pro") == 0 &&
		 !row_value(choice, TEST_CONCURRENCY, "multiple")));
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

/* The native head qualifier prepares this module on both card classes before
 * readiness. Check the linked table, not just the generated C or loose cubins. */
static bool banked_head_kernels_present(void)
{
	static const char *const architectures[] = { "sm_80", "sm_120a" };
	for (size_t architecture = 0;
	     architecture < sizeof(architectures) / sizeof(architectures[0]);
	     architecture++) {
		bool found = false;
		for (uint32_t image = 0; image < cuda_kernel_image_count;
		     image++) {
			const struct cuda_kernel_image *const module =
				&cuda_kernel_images[image];
			found = found ||
				(module->bytes != NULL &&
				 module->byte_count != 0 &&
				 strcmp(module->kernel_name,
					KERNEL_FILE_HEAD_OFFLOAD) == 0 &&
				 strcmp(module->architecture,
					architectures[architecture]) == 0);
		}
		if (!found)
			return false;
	}
	return true;
}

/* The rows and draft knobs of an Original row: TP4_PRO's focus=decode sets
 * 6 rows (otherwise the route's own; 0: no source sets rows, raw PP5's 1),
 * category=code depth 7, clamped to rows less 1 where a source sets rows,
 * and focus=decode the cost-priced depth with session acceptance 6. Under
 * DSpark an unset depth follows the block. Elsewhere a recipe row keeps its
 * measured fixed depth and a raw row the Original default, the cost-priced
 * depth. */
static bool original_draft_geometry(const struct axis_choice *choice,
				    uint32_t route_rows)
{
	const bool decode = row_value(choice, TEST_FOCUS, "decode");
	const bool code = row_value(choice, TEST_CATEGORY, "code");
	const bool tp = strcmp(choice->gpu_split, "tp4_pro") == 0;
	const uint32_t rows = decode && tp ? TEST_TP_DECODE_ROWS : route_rows;
	const bool clamped = code && rows != 0u &&
			     TEST_CODE_DRAFT_DEPTH >= rows;
	const uint32_t block = clamped ? rows - 1u :
			       code    ? TEST_CODE_DRAFT_DEPTH :
					 MODEL_DRAFT_BLOCK;
	return step_rows == (rows != 0u ? rows : 1u) && draft_block == block &&
	       draft_positions == (code || speculative >= SPECULATIVE_DEVICE ?
					   block :
					   SERVING_DRAFT_MOST) &&
	       draft_policy.depth_policy ==
		       (decode || !original_recipe_row(choice) ?
				DRAFT_DEPTH_COST :
				DRAFT_DEPTH_FIXED) &&
	       draft_session_acceptance ==
		       (decode ? TEST_DECODE_SESSION_ACCEPTANCE : 0u);
}

static bool original_tuple_geometry(const struct axis_choice *choice,
				    const char *command)
{
	const bool serving = strcmp(command, "serve") == 0;
	if (original_head_offload == V41_HEAD_MODE_BANKED &&
	    strcmp(choice->value[TEST_EXACTNESS], "exact") != 0)
		return execution_placement == EXECUTION_PP5 &&
		       original_head_offload == V41_HEAD_MODE_BANKED &&
		       pro_emulation == 0u && official_reference == 0u &&
		       official_hc_rows == 0u &&
		       step_rows == LINEAR_MAXIMUM_ROWS &&
		       wide_rows == TEST_BATCH_WIDE_ROWS &&
		       draft_policy.depth_policy == DRAFT_DEPTH_COST &&
		       draft_policy.base_us == DRAFT_DEFAULT_BASE_US &&
		       draft_policy.tail_us == DRAFT_DEFAULT_TAIL_US &&
		       draft_policy.row_us == DRAFT_DEFAULT_ROW_US &&
		       draft_learned_acceptance == 0 &&
		       draft_session_acceptance ==
			       TEST_BANKED_SESSION_ACCEPTANCE &&
		       batch_plan_ties == 0 && tp_pipeline_overlap == 0 &&
		       original_chunk_core == 0 && original_dense_staged == 0 &&
		       original_wide_experts == 0 && original_batch == 0 &&
		       (!serving ||
			(prefix_reuse == 1 && prefill_order == 0)) &&
		       speculative == SPECULATIVE_DEVICE && is_draft_given &&
		       draft_positions == TEST_FULL_DRAFT_DEPTH &&
		       pipeline_steps ==
			       (serving ? TEST_BANKED_PIPELINE_STEPS : 1u) &&
		       session_slots ==
			       (serving ? SESSION_SLOTS_MAX_PARSED : 1u) &&
		       layout_text != NULL &&
		       strcmp(layout_text, TEST_ORIGINAL_LAYOUT) == 0 &&
		       original_card_classes_match() &&
		       banked_head_kernels_present();
	if (row_value(choice, TEST_EXACTNESS, "exact"))
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
	/* Unbounded sessions: no set pins the context or the slots, so every row has 1M context and every serve
	 * parses --session_slots=max. PP5 single and balanced run the split's base: DSpark at eight
	 * rows (six for single), the 8/7/7/7/11 layout, the CLI's draft policy and pipeline depth (five flights for
	 * a multi-session serve). */
	if (!original_recipe_row(choice))
		return execution_placement == EXECUTION_PP5 &&
		       numerics == NUMERICS_EXACT &&
		       cmp_order == CMP_ORDER_REFERENCE &&
		       linear_order == LINEAR_ORDER_REFERENCE &&
		       linear_profile == LINEAR_PROFILE_PRECISION &&
		       pro_emulation == 0u && tp_wire_bf16 == 0u &&
		       official_reference == 0u && official_hc_rows == 0u &&
		       speculative == SPECULATIVE_DEVICE &&
		       original_draft_geometry(
			       choice,
			       row_value(choice, TEST_CONCURRENCY, "single") ?
				       TEST_PP5_SINGLE_ROWS :
				       LINEAR_MAXIMUM_ROWS) &&
		       wide_rows == TEST_BATCH_WIDE_ROWS && full_steps == 1u &&
		       prefill_cone == 1u && chunks_in_flight == 1u &&
		       step_graphs == 1u && batch_rows[0] == 0u &&
		       batch_verifiers[0] == 0u && batch_step_us[0] == 0u &&
		       pipeline_steps ==
			       (serving ? TEST_BANKED_PIPELINE_STEPS : 1u) &&
		       session_slots ==
			       (serving ? SESSION_SLOTS_MAX_PARSED : 1u) &&
		       session_slots_max == serving &&
		       context_tokens == TEST_OWNER_CONTEXT_TOKENS &&
		       tp_head_shards == 0u && tp_draft_shards == 0u &&
		       (!serving ||
			(prefix_reuse == 1u && prefill_order == 0u)) &&
		       layout_text != NULL &&
		       strcmp(layout_text, TEST_ORIGINAL_LAYOUT) == 0 &&
		       original_card_classes_match();
	const uint32_t depth = !serving || !multiple ?
				       TEST_THIN_PIPELINE_STEPS :
				       TEST_ORIGINAL_DECODE_PIPELINE;
	bool passed = execution_placement ==
			      (multiple ? EXECUTION_PP5 : EXECUTION_TP4_PRO) &&
		      numerics == NUMERICS_EXACT &&
		      cmp_order == CMP_ORDER_REFERENCE &&
		      linear_order == LINEAR_ORDER_REFERENCE &&
		      linear_profile == LINEAR_PROFILE_PRECISION &&
		      pro_emulation == 0u && tp_wire_bf16 == 0u &&
		      official_reference == 0u && official_hc_rows == 0u &&
		      speculative == SPECULATIVE_DEVICE &&
		      original_draft_geometry(choice, LINEAR_MAXIMUM_ROWS) &&
		      wide_rows == TEST_BATCH_WIDE_ROWS && full_steps == 1u &&
		      prefill_cone == 1u && chunks_in_flight == 1u &&
		      step_graphs == 1u && batch_rows[0] == 0u &&
		      batch_verifiers[0] == 0u && batch_step_us[0] == 0u &&
		      pipeline_steps == depth &&
		      session_slots ==
			      (serving ? SESSION_SLOTS_MAX_PARSED : 1u) &&
		      session_slots_max == serving &&
		      context_tokens == TEST_OWNER_CONTEXT_TOKENS &&
		      tp_head_shards == 0u && tp_head_stream == 0u &&
		      tp_draft_shards == 0u && tp_draft_layers == 0u &&
		      tp_code_wire == 0u && tp_attention_owner == 0u &&
		      prefill_decode_rows == 0u && prefill_order == 0u &&
		      original_card_classes_match();
	/* The recipes' --prefix_reuse=0 pinned their measurement protocol and is
	 * not inserted, so serving keeps the prefix cache. */
	if (serving)
		passed = passed && prefix_reuse == 1u;
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
	if (metrics != 0u)
		return false;
	if (strcmp(choice->weight_family, "original") == 0)
		return original_tuple_geometry(choice, command);
	/* Exact is the exact-shaping set (q4's exact-FP32 arm), none of the THIN base: exact numerics, the reference
	 * order, the FP32 wire, DSpark at eight rows, pipeline depth 1, one session. */
	if (row_value(choice, TEST_EXACTNESS, "exact"))
		return execution_placement == EXECUTION_TP4_EXL3 &&
		       numerics == NUMERICS_EXACT &&
		       cmp_order == CMP_ORDER_REFERENCE && tp_wire_bf16 == 0u &&
		       speculative == SPECULATIVE_DEVICE &&
		       step_rows == LINEAR_MAXIMUM_ROWS &&
		       pipeline_steps == 1u && session_slots == 1u &&
		       batch_rows[0] == 0u;
	/* Concurrency single and balanced run the THIN base, multiple BATCH (a serve without exactly 8 slots runs the
	 * balanced row, so the row here is the one inserted). Every row has 1M context and every serve parses
	 * --session_slots=max as several sessions; at 1M the TP4_EXL3 fit is one session (U0,
	 * tp_session_fit_test), which takes --session_slots=1's defaults (single_session_defaults, stage 3
	 * review D1): the stage-2 single rows' configuration, checked below. */
	const bool multiple = row_value(choice, TEST_CONCURRENCY, "multiple");
	const bool serving = strcmp(command, "serve") == 0;
	const bool max_parsed = session_slots_max &&
				session_slots == SESSION_SLOTS_MAX_PARSED;
	if (serving && !multiple && max_parsed &&
	    single_session_defaults(NULL, NULL) == NULL)
		session_slots = 1u; /* what the fit sets */
	/* Category code fixes the draft prices; otherwise the CLI keeps fixed
	 * prices on one-slot THIN and the rate objective on BATCH. Focus decode
	 * owns layer drafting; BATCH's sharded draft rows raise it to 1. */
	bool passed =
		execution_placement == EXECUTION_TP4_EXL3 &&
		max_parsed == (serving && !multiple) &&
		draft_policy.depth_policy ==
			(multiple && !row_value(choice, TEST_CATEGORY, "code") ?
				 DRAFT_DEPTH_RATE :
				 DRAFT_DEPTH_FIXED) &&
		tp_draft_layers ==
			(row_value(choice, TEST_FOCUS, "decode") || multiple ?
				 1u :
				 0u);
	/* The recipes' --prefix_reuse=0 pinned their measurement protocol. */
	if (serving)
		passed = passed && prefix_reuse == 1u;
	passed = passed && numerics == NUMERICS_RELAXED &&
		 cmp_order == CMP_ORDER_REFERENCE && tp_wire_bf16 == 1u &&
		 speculative == SPECULATIVE_DEVICE &&
		 draft_positions == TEST_FULL_DRAFT_DEPTH &&
		 tp_head_shards == 1u && tp_head_stream == 1u &&
		 tp_draft_shards == 1u;
	if (multiple) {
		/* Step prices: prose and structured insert the menu; code's
		 * fixed prices keep none; the balanced category leaves it to
		 * the CLI, which prices a measured BATCH plan (serve) itself. */
		const bool code = row_value(choice, TEST_CATEGORY, "code");
		const bool priced =
			!code && (serving || !row_value(choice, TEST_CATEGORY,
							"balanced"));
		passed = passed &&
			 batch_core_once == TEST_BATCH_POOL_DEFERRED_CORE &&
			 wide_rows == TEST_BATCH_WIDE_ROWS &&
			 pipeline_steps == 1u &&
			 batch_rows[0] == LINEAR_MAXIMUM_ROWS &&
			 batch_step_us[0] ==
				 (priced ? TEST_BATCH_FIRST_STEP_US : 0u) &&
			 session_slots == (serving ? TEST_BATCH_SESSIONS : 1u);
		if (!serving)
			passed = passed && batch_rows[1] == 0u &&
				 batch_verifiers[0] == 1u &&
				 batch_verifiers[1] == 0u;
		passed = passed &&
			 prefill_decode_rows ==
				 (serving ? PREFILL_DECODE_DEFAULT_ROWS : 0u) &&
			 prefill_decode_rounds ==
				 prefill_decode_default_rounds &&
			 prefill_decode_waiters ==
				 prefill_decode_default_waiters &&
			 prefill_order ==
				 (serving ? PREFILL_ORDER_SHORTEST : 0u);
	} else {
		passed = passed && wide_rows == TEST_THIN_WIDE_ROWS &&
			 pipeline_steps == TEST_THIN_PIPELINE_STEPS &&
			 batch_rows[0] == 0u && batch_step_us[0] == 0u &&
			 session_slots == 1u &&
			 context_tokens == TEST_OWNER_CONTEXT_TOKENS &&
			 prefill_decode_rows == 0u;
	}
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
	case TEST_HEAD_OFF:
		return original_head_offload == 0u;
	case TEST_SESSION_OVERRIDE:
		return draft_session_acceptance ==
		       TEST_BANKED_SESSION_ACCEPTANCE;
	case TEST_BATCH_AT_EIGHT:
	case TEST_BATCH_GEOMETRY:
		return batch_rows[0] == LINEAR_MAXIMUM_ROWS &&
		       pipeline_steps == 1u && !session_slots_max &&
		       session_slots == TEST_BATCH_SESSIONS &&
		       context_tokens == TEST_BATCH_CONTEXT_TOKENS;
	case TEST_TP_METRICS_ON:
		return metrics == 1u &&
		       execution_placement == EXECUTION_TP4_PRO &&
		       original_head_offload == 0u &&
		       pipeline_steps == TEST_THIN_PIPELINE_STEPS;
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
	     weights[] = "weights", placement[TEST_PLACEMENT_BYTES];
	if (test->placement != NULL) {
		const int written = snprintf(placement, sizeof(placement),
					     "--placement=%s", test->placement);
		if (written < 0 || (size_t)written >= sizeof(placement))
			return false;
	}
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
	/* An explicit placement pins the GPU split; without one, PP5. */
	if (test->placement != NULL)
		arguments[count++] = placement;
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
	const struct axis_choice *const row = inserted_choice(test);
	if (expanded_ok) {
		passed = passed && row != NULL && expanded.choice == row &&
			 expanded.slots_unmet ==
				 (row != test->choice ? test->choice : NULL) &&
			 strcmp(expanded.weight_family,
				test->choice->weight_family) == 0 &&
			 strcmp(expanded.gpu_split, test->choice->gpu_split) ==
				 0 &&
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
			if (test->extra[0] == NULL ||
			    test->destination == TEST_BATCH_GEOMETRY)
				passed = passed && ordinary_tuple_geometry(
							   row, test->command);
		}
	}
	(void)fflush(stderr);
	passed = (dup2(saved_error, STDERR_FILENO) == STDERR_FILENO) && passed;
	(void)close(saved_error);
	passed = fseek(diagnostic, 0L, SEEK_SET) == 0 && passed;
	clearerr(diagnostic);
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
		(void)fprintf(
			stderr,
			"failed axes: %s/%s/%s/%s %s (placement %s) %s %s %s\n",
			test->choice->value[TEST_CATEGORY],
			test->choice->value[TEST_EXACTNESS],
			test->choice->value[TEST_CONCURRENCY],
			test->choice->value[TEST_FOCUS],
			test->choice->gpu_split,
			test->placement != NULL ? test->placement : "default",
			test->command,
			test->extra[0] != NULL ? test->extra[0] : "",
			test->extra[1] != NULL ? test->extra[1] : "");
	return passed;
}

static const struct axis_choice *
find_choice(const struct axis_choice *choices, size_t count,
	    const char *category, const char *exactness,
	    const char *concurrency, const char *focus, const char *split)
{
	for (size_t index = 0; index < count; index++)
		if (strcmp(choices[index].value[TEST_CATEGORY], category) ==
			    0 &&
		    strcmp(choices[index].value[TEST_EXACTNESS], exactness) ==
			    0 &&
		    strcmp(choices[index].value[TEST_CONCURRENCY],
			   concurrency) == 0 &&
		    strcmp(choices[index].value[TEST_FOCUS], focus) == 0 &&
		    strcmp(choices[index].gpu_split, split) == 0)
			return &choices[index];
	return NULL;
}

/* Every row names a tabled GPU split and is the only row of its tuple there. */
static bool rows_unique(const struct axis_choice *choices, size_t count)
{
	bool passed = true;
	for (size_t index = 0; index < count; index++) {
		const struct axis_choice *row = &choices[index];
		bool tabled = false;
		for (size_t split = 0; split < TEST_SPLITS; split++)
			tabled = tabled || strcmp(row->gpu_split,
						  split_names[split]) == 0;
		passed = passed && tabled &&
			 find_choice(choices, count, row->value[TEST_CATEGORY],
				     row->value[TEST_EXACTNESS],
				     row->value[TEST_CONCURRENCY],
				     row->value[TEST_FOCUS],
				     row->gpu_split) == row;
	}
	return passed;
}

/* Stage 3: a case without an axis flag runs in two forms. RAW_AXES_OFF puts --axes=off first: nothing is inserted
 * and the parser sees the raw engine defaults, as these cases expected before stage 3. RAW_NO_AXIS runs the argv as
 * given, which takes its split's all-balanced row; that row's parse is the tuple cases'. */
enum raw_form { RAW_NO_AXIS, RAW_AXES_OFF };

/* The all-balanced row a run without an axis flag takes: its --placement's split, else the default (NULL: no table). */
static const struct axis_choice *no_axis_row(const char *const *options,
					     size_t option_count)
{
	const char *split = TEST_DEFAULT_SPLIT;
	for (size_t index = 0; index < option_count; index++)
		if (strncmp(options[index], TEST_PLACEMENT_PREFIX,
			    strlen(TEST_PLACEMENT_PREFIX)) == 0)
			split = options[index] + strlen(TEST_PLACEMENT_PREFIX);
	size_t rows = 0;
	const struct axis_choice *const all = axis_choices_entry(&rows);
	return find_choice(all, rows, "balanced", "balanced", "balanced",
			   "balanced", split);
}

/* Stage 3, "no axis = balanced everywhere": a run without an axis flag (arguments) expands to the argv the same run
 * gives with the four axes at balanced, bound for bound, on its split's all-balanced row (expected). */
static bool all_balanced_expansion(const struct axis_arguments_record *expanded,
				   int count, char *const *arguments,
				   const struct axis_choice *expected)
{
	char category[] = "--category=balanced",
	     exactness[] = "--exactness=balanced",
	     concurrency[] = "--concurrency=balanced",
	     focus[] = "--focus=balanced";
	char *balanced[TEST_ARGUMENT_CAPACITY] = { arguments[0], arguments[1],
						   category,	 exactness,
						   concurrency,	 focus };
	int balanced_count = TEST_OPTION_START + (int)TEST_AXES;
	if (count + (int)TEST_AXES > (int)TEST_ARGUMENT_CAPACITY)
		return false;
	for (int index = TEST_OPTION_START; index < count; index++)
		balanced[balanced_count++] = arguments[index];
	struct axis_arguments_record selected = { 0 };
	bool passed = axis_expand(balanced_count, balanced, &selected) &&
		      expected != NULL && expanded->choice == expected &&
		      selected.choice == expected &&
		      expanded->selected_count == 0u &&
		      selected.selected_count == TEST_AXES &&
		      expanded->inserted_count == selected.inserted_count &&
		      expanded->count == selected.count;
	for (int index = 0; passed && index < expanded->count; index++)
		passed = strcmp(expanded->values[index],
				selected.values[index]) == 0;
	char split_line[TEST_OPTION_BYTES];
	const int written =
		expected != NULL ?
			snprintf(split_line, sizeof(split_line),
				 "axes gpu_split=%s tuple=balanced/balanced/"
				 "balanced/balanced\n",
				 expected->gpu_split) :
			0;
	char *log_text = NULL;
	size_t log_bytes = 0;
	FILE *log = passed ? open_memstream(&log_text, &log_bytes) : NULL;
	if (log != NULL) {
		axis_log(expanded, log);
		passed = fclose(log) == 0 && log_text != NULL && written > 0 &&
			 (size_t)written < sizeof(split_line) &&
			 strstr(log_text, split_line) != NULL;
	} else {
		passed = false;
	}
	free(log_text);
	axis_arguments_free(&selected);
	return passed;
}

/* --axes=off: the argv less that word, nothing selected, inserted or logged. */
static bool axes_off_untouched(const struct axis_arguments_record *expanded,
			       int count, char *const *arguments)
{
	bool passed = expanded->count == count - 1 &&
		      expanded->choice == NULL &&
		      expanded->explicit_count == count &&
		      expanded->explicit_values == arguments &&
		      expanded->selected_count == 0u &&
		      expanded->inserted_count == 0u;
	for (int index = 0; passed && index < expanded->count; index++)
		passed = expanded->values[index] ==
			 arguments[index < (int)TEST_OPTION_START ? index :
								    index + 1];
	for (size_t axis = 0; axis < TEST_AXES; axis++)
		passed = passed && expanded->selected[axis] == NULL;
	char *log_text = NULL;
	size_t log_bytes = 0;
	FILE *log = open_memstream(&log_text, &log_bytes);
	if (log == NULL) {
		passed = false;
	} else {
		axis_log(expanded, log);
		passed = (fclose(log) == 0) && log_bytes == 0u && passed;
	}
	free(log_text);
	return passed;
}

static bool raw_arguments_case(const char *const *options, size_t option_count,
			       bool expansion_expected, bool parse_expected,
			       bool untouched, enum raw_form form)
{
	char program[] = "anemone", generate[] = "generate",
	     messages[] = "messages", tables[] = "tables",
	     weights[] = "weights", axes_off[] = TEST_AXES_OFF;
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, generate };
	char *copies[TEST_ARGUMENT_CAPACITY] = { 0 };
	int count = TEST_OPTION_START;
	const bool raw = untouched && form == RAW_AXES_OFF;
	if (raw)
		arguments[count++] = axes_off;
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
	const struct axis_choice *const balanced =
		untouched && !raw ? no_axis_row(options, option_count) : NULL;
	if (expanded_ok && raw) {
		passed = passed &&
			 axes_off_untouched(&expanded, count, arguments);
	} else if (expanded_ok && balanced != NULL) {
		passed = passed && all_balanced_expansion(&expanded, count,
							  arguments, balanced);
	} else if (expanded_ok && untouched) {
		/* a split without a table keeps its raw defaults */
		passed = passed && expanded.count == count &&
			 expanded.values == arguments &&
			 expanded.storage == NULL && expanded.choice == NULL &&
			 expanded.inserted_count == 0u;
	}
	/* Stage 3 review D5: the no-axis form parses as its all-balanced row expects, as the --axes=off form and
	 * a flagged run do (a raw --rows=1 --speculative=0 caps the base's inert draft knobs at their parser least). */
	if (expanded_ok)
		passed = (parse_arguments(expanded.count, expanded.values) ==
			  parse_expected) &&
			 passed;
	axis_arguments_free(&expanded);
	for (size_t index = 0; index < option_count; index++)
		free(copies[index]);
	return passed;
}

static bool isolated_raw_form(const char *const *options, size_t option_count,
			      bool expansion_expected, bool parse_expected,
			      bool untouched, enum raw_form form)
{
	axis_cases_run++;
	const pid_t child = fork();
	if (child == 0)
		_exit(raw_arguments_case(options, option_count,
					 expansion_expected, parse_expected,
					 untouched, form) ?
			      0 :
			      1);
	int status = 0;
	const bool passed = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) && WEXITSTATUS(status) == 0;
	if (!passed)
		(void)fprintf(stderr, "failed raw axes case%s: %s\n",
			      !untouched ? "" :
			      form == RAW_AXES_OFF ?
					   " (" TEST_AXES_OFF ")" :
					   " (no axis: all-balanced row)",
			      option_count > 0 ? options[0] : "no axes");
	return passed;
}

/* untouched: the options hold no axis flag, so both forms run (raw_form). */
static bool isolated_raw_case(const char *const *options, size_t option_count,
			      bool expansion_expected, bool parse_expected,
			      bool untouched)
{
	bool passed = isolated_raw_form(options, option_count,
					expansion_expected, parse_expected,
					untouched, RAW_NO_AXIS);
	if (untouched)
		passed = isolated_raw_form(options, option_count,
					   expansion_expected, parse_expected,
					   untouched, RAW_AXES_OFF) &&
			 passed;
	return passed;
}

/* A partial tuple resolves to the row with balanced on each unset axis (and,
 * under exact, the reference recipe's single session): expected is that row.
 * A refusal (expected NULL) prints the given reason. */
static bool partial_row_case(const char *const *options, size_t option_count,
			     const struct axis_choice *expected,
			     const char *refusal)
{
	char program[] = "anemone", generate[] = "generate",
	     messages[] = "messages", tables[] = "tables",
	     weights[] = "weights";
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, generate };
	char *copies[TEST_ARGUMENT_CAPACITY] = { 0 };
	int count = TEST_OPTION_START;
	bool passed = option_count + TEST_EXTRA_CAPACITY + TEST_OPTION_START <=
		      TEST_ARGUMENT_CAPACITY;
	for (size_t index = 0; passed && index < option_count; index++) {
		copies[index] = strdup(options[index]);
		passed = copies[index] != NULL;
		arguments[count++] = copies[index];
	}
	arguments[count++] = messages;
	arguments[count++] = tables;
	arguments[count++] = weights;
	FILE *diagnostic = tmpfile();
	const int saved_error = dup(STDERR_FILENO);
	passed = passed && diagnostic != NULL && saved_error >= 0 &&
		 dup2(fileno(diagnostic), STDERR_FILENO) == STDERR_FILENO;
	struct axis_arguments_record expanded = { 0 };
	const bool expanded_ok = passed &&
				 axis_expand(count, arguments, &expanded);
	passed = passed && expanded_ok == (expected != NULL) &&
		 (expected == NULL || expanded.choice == expected);
	(void)fflush(stderr);
	if (saved_error >= 0) {
		passed = dup2(saved_error, STDERR_FILENO) == STDERR_FILENO &&
			 passed;
		(void)close(saved_error);
	}
	char text[TEST_LOG_CAPACITY] = "";
	if (diagnostic != NULL) {
		passed = fseek(diagnostic, 0L, SEEK_SET) == 0 && passed;
		clearerr(diagnostic);
		const size_t bytes =
			fread(text, 1u, sizeof(text) - 1u, diagnostic);
		text[bytes] = '\0';
		(void)fclose(diagnostic);
	}
	if (refusal != NULL)
		passed = passed && strstr(text, refusal) != NULL;
	axis_arguments_free(&expanded);
	for (size_t index = 0; index < option_count; index++)
		free(copies[index]);
	return passed;
}

static bool isolated_partial_case(const char *const *options,
				  size_t option_count,
				  const struct axis_choice *expected,
				  const char *refusal)
{
	axis_cases_run++;
	const pid_t child = fork();
	if (child == 0)
		_exit(partial_row_case(options, option_count, expected,
				       refusal) ?
			      0 :
			      1);
	int status = 0;
	const bool passed = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) && WEXITSTATUS(status) == 0;
	if (!passed)
		(void)fprintf(stderr, "failed partial row case: %s %s\n",
			      options[0], option_count > 1u ? options[1] : "");
	return passed;
}

/* Raw flags exercise the real parser while exact still awaits serving gates. */
static bool reference_raw_recipe_cases(const struct axis_choice *reference)
{
	if (reference == NULL)
		return false;
	/* An empty generate base is the serve base (the table stores it once). */
	char *storage = strdup(reference->generate_settings[0] != '\0' ?
				       reference->generate_settings :
				       reference->settings);
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
	if (!passed)
		(void)fputs("failed exact reference raw recipe cases\n",
			    stderr);
	return passed;
}

/* Fresh CLI state in each child: neither the exact axis nor command may override sampling. */
/* the parsed values of --temperature=0.7, --top_p=0.4, --temperature=2 and --top_p=0.001 */
#define CASE_TEMPERATURE 0.7f
#define CASE_TOP_P 0.4f
#define HOT_TEMPERATURE 2.0f
#define NARROW_TOP_P 0.001f
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
			CASE_TEMPERATURE, CASE_TOP_P, UINT32_MAX },
		      { "--temperature=2", "--top_p=0.001", "--seed=1",
			HOT_TEMPERATURE, NARROW_TOP_P, 1 } };
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
	TEST_LATE_WITHOUT_AXES, /* stage 3: the all-balanced row */
	TEST_LATE_AXES_OFF, /* --axes=off: nothing inserted */
	TEST_SELECTION_BOUNDARIES
};

static bool selection_boundary_case(enum test_selection_boundary boundary)
{
	char program[] = "anemone", generate[] = "generate",
	     placement[] = "--placement=tp4_exl3", code[] = "--category=code",
	     prose[] = "--category=prose", exactness[] = "--exactness=balanced",
	     concurrency[] = "--concurrency=single", focus[] = "--focus=decode",
	     messages[] = "messages", tables[] = "tables",
	     weights[] = "weights", late[] = "--category=vision",
	     axes_off[] = TEST_AXES_OFF;
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, generate };
	int count = TEST_OPTION_START;
	const bool without_axes = boundary == TEST_LATE_WITHOUT_AXES ||
				  boundary == TEST_LATE_AXES_OFF;
	if (boundary == TEST_LATE_AXES_OFF)
		arguments[count++] = axes_off;
	if (!without_axes) {
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
	/* A flag-shaped pack-copy path after the positionals stays the last positional, never an axis. */
	if (passed && boundary == TEST_LATE_AXES_OFF) {
		passed = axes_off_untouched(&expanded, count, arguments) &&
			 expanded.values[expanded.count - 1] == late;
	} else if (passed && boundary == TEST_LATE_WITHOUT_AXES) {
		passed = all_balanced_expansion(&expanded, count, arguments,
						no_axis_row(NULL, 0u)) &&
			 expanded.values[expanded.count - 1] == late;
	} else if (passed) {
		size_t rows = 0;
		const struct axis_choice *const all = axis_choices_entry(&rows);
		const struct axis_choice *const common =
			find_choice(all, rows, "prose", "balanced", "single",
				    "balanced", "tp4_exl3");
		struct axis_test_case test = { .choice = expanded.choice,
					       .command = "generate" };
		passed = expanded.choice != NULL &&
			 (boundary != TEST_COMMON_PLAN ||
			  expanded.choice == common) &&
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
#define TEST_CLAMP_OPTIONS 8u
#define TEST_CLAMP_LINES 3u
#define TEST_CODE_DECODE_TUPLE                                             \
	"--category=code", "--exactness=balanced", "--concurrency=single", \
		"--focus=decode"
#define TEST_CLAMP_NOTE(knob) \
	" " knob " 7 -> 5 (bounded by --rows=6 from focus=decode)"
#define TEST_USER_CLAMP_NOTE(knob) \
	" " knob " 7 -> 5 (bounded by --rows=6 from the user)"
#define TEST_SINGLE_CLAMP_NOTE(knob) \
	" " knob " 7 -> 5 (bounded by --rows=6 from concurrency=single)"
#define TEST_INACTIVE_NOTE " (inactive: no DSpark on this route)"
#define TEST_EXL3_TTFT_TUPLE                                                  \
	"--category=prose", "--exactness=balanced", "--concurrency=multiple", \
		"--focus=ttft"
#define TEST_FLOOR_NOTE                                                    \
	" --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from " \
	"concurrency=multiple)"
#define TEST_USER_FLOOR_NOTE \
	" --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from the user)"

enum test_parse { TEST_PARSES, TEST_REFUSES, TEST_UNPARSED };

/* A declared cross-axis bound on the final values, and the inactive-draft
 * note: the resolver's argv, its axis log and the ordinary parser. */
struct clamp_test {
	const char *options[TEST_CLAMP_OPTIONS];
	enum test_parse parse;
	uint32_t rows, block, positions;
	const char *lines[TEST_CLAMP_LINES], *absent;
	uint32_t layers; /* --tp_draft_layers once parsed */
	/* TP4_EXL3 BATCH: serve gives --session_slots=8, without which the row runs its concurrency=balanced row */
	bool batch_serve;
};

static bool clamp_case(const struct clamp_test *test, bool serving)
{
	char program[] = "anemone", generate[] = "generate", serve[] = "serve",
	     messages[] = "messages", tables[] = "tables",
	     weights[] = "weights";
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, serving ?
								     serve :
								     generate };
	char *copies[TEST_CLAMP_OPTIONS] = { 0 };
	int count = TEST_OPTION_START;
	bool passed = true;
	for (size_t index = 0;
	     index < TEST_CLAMP_OPTIONS && test->options[index] != NULL;
	     index++) {
		copies[index] = strdup(test->options[index]);
		passed = passed && copies[index] != NULL;
		arguments[count++] = copies[index];
	}
	char batch_slots[] = TEST_BATCH_SLOTS;
	if (serving && test->batch_serve)
		arguments[count++] = batch_slots;
	if (!serving)
		arguments[count++] = messages;
	arguments[count++] = tables;
	arguments[count++] = weights;
	struct axis_arguments_record expanded = { 0 };
	passed = passed && axis_expand(count, arguments, &expanded);
	char *log_text = NULL;
	size_t log_bytes = 0;
	FILE *log = passed ? open_memstream(&log_text, &log_bytes) : NULL;
	if (log != NULL) {
		axis_log(&expanded, log);
		passed = fclose(log) == 0 && log_text != NULL;
	} else {
		passed = false;
	}
	for (size_t line = 0; passed && line < TEST_CLAMP_LINES; line++)
		passed = test->lines[line] == NULL ||
			 strstr(log_text, test->lines[line]) != NULL;
	passed = passed && (test->absent == NULL ||
			    strstr(log_text, test->absent) == NULL);
	if (passed && test->parse != TEST_UNPARSED) {
		const bool parsed =
			parse_arguments(expanded.count, expanded.values);
		passed = parsed == (test->parse == TEST_PARSES) &&
			 (!parsed || (step_rows == test->rows &&
				      draft_block == test->block &&
				      draft_positions == test->positions &&
				      tp_draft_layers == test->layers));
	}
	free(log_text);
	axis_arguments_free(&expanded);
	for (size_t index = 0; index < TEST_CLAMP_OPTIONS; index++)
		free(copies[index]);
	return passed;
}

static bool clamp_cases(void)
{
	static const struct clamp_test tests[] = {
		/* TP4_PRO code/decode: depth 7 on rows 6 is clamped to 5. */
		{ { "--placement=tp4_pro", TEST_CODE_DECODE_TUPLE },
		  TEST_PARSES,
		  TEST_TP_DECODE_ROWS,
		  MODEL_DRAFT_BLOCK,
		  MODEL_DRAFT_BLOCK,
		  { "axes from category=code:" TEST_CLAMP_NOTE("--draft_block")
			    TEST_CLAMP_NOTE("--draft_positions") "\n",
		    "axes set --draft_block=5\naxes set --draft_positions=5\n",
		    "axes from focus=decode: --draft_policy "
		    "--draft_session_acceptance --rows\n" },
		  "axes set --draft_block=7",
		  0u,
		  false },
		/* An explicit depth wins and is validated as usual. */
		{ { "--placement=tp4_pro", TEST_CODE_DECODE_TUPLE,
		    "--draft_positions=4" },
		  TEST_PARSES,
		  TEST_TP_DECODE_ROWS,
		  MODEL_DRAFT_BLOCK,
		  TEST_FULL_DRAFT_DEPTH - 1u,
		  { "axes from category=code:" TEST_CLAMP_NOTE(
			    "--draft_block") "\n",
		    NULL, NULL },
		  "axes set --draft_positions",
		  0u,
		  false },
		{ { "--placement=tp4_pro", TEST_CODE_DECODE_TUPLE,
		    "--draft_positions=7" },
		  TEST_REFUSES,
		  0u,
		  0u,
		  0u,
		  { NULL, NULL, NULL },
		  "axes set --draft_positions",
		  0u,
		  false },
		/* Bounds act on the final rows (stage 3 review D1): an explicit --rows=8
		 * lifts the clamp... */
		{ { "--placement=tp4_pro", TEST_CODE_DECODE_TUPLE, "--rows=8" },
		  TEST_PARSES,
		  LINEAR_MAXIMUM_ROWS,
		  TEST_CODE_DRAFT_DEPTH,
		  TEST_CODE_DRAFT_DEPTH,
		  { "axes from category=code: --draft_block --draft_positions\n",
		    "axes set --draft_block=7\naxes set --draft_positions=7\n",
		    "axes from focus=decode: --draft_policy "
		    "--draft_session_acceptance\n" },
		  "bounded by",
		  0u,
		  false },
		/* ...and an explicit --rows=6 keeps it (no r5c stack), naming
		 * the user. */
		{ { "--placement=tp4_pro", TEST_CODE_DECODE_TUPLE, "--rows=6" },
		  TEST_PARSES,
		  TEST_TP_DECODE_ROWS,
		  MODEL_DRAFT_BLOCK,
		  MODEL_DRAFT_BLOCK,
		  { "axes from category=code:" TEST_USER_CLAMP_NOTE(
			    "--draft_block")
			    TEST_USER_CLAMP_NOTE("--draft_positions") "\n",
		    "axes set --draft_block=5\naxes set --draft_positions=5\n",
		    "axes from focus=decode: --draft_policy "
		    "--draft_session_acceptance\n" },
		  "from focus=decode)",
		  0u,
		  false },
		/* On PP5 an explicit --rows bounds the inserted depth too. */
		{ { TEST_CODE_DECODE_TUPLE, "--speculative=2", "--rows=6" },
		  TEST_UNPARSED,
		  0u,
		  0u,
		  0u,
		  { "axes from category=code:" TEST_USER_CLAMP_NOTE(
			    "--draft_block")
			    TEST_USER_CLAMP_NOTE("--draft_positions") "\n",
		    "axes set --draft_block=5\naxes set --draft_positions=5\n",
		    NULL },
		  "inactive",
		  0u,
		  false },
		/* Stage 3: PP5's base runs DSpark, and concurrency=single's rows 6
		 * (r7c g3) clamp code's depth 7 to 5; with speculation on, no
		 * draft line is noted inactive. */
		{ { TEST_CODE_DECODE_TUPLE },
		  TEST_PARSES,
		  TEST_PP5_SINGLE_ROWS,
		  MODEL_DRAFT_BLOCK,
		  MODEL_DRAFT_BLOCK,
		  { "axes from category=code:" TEST_SINGLE_CLAMP_NOTE(
			    "--draft_block")
			    TEST_SINGLE_CLAMP_NOTE("--draft_positions") "\n",
		    "axes set --draft_block=5\naxes set --draft_positions=5\n",
		    "axes from focus=decode: --draft_policy "
		    "--draft_session_acceptance\n" },
		  "inactive",
		  0u,
		  false },
		/* An explicit --rows=8 lifts the clamp on PP5 too... */
		{ { TEST_CODE_DECODE_TUPLE, "--speculative=2", "--rows=8" },
		  TEST_UNPARSED,
		  0u,
		  0u,
		  0u,
		  { "axes from category=code: --draft_block --draft_positions\n",
		    NULL, NULL },
		  "inactive",
		  0u,
		  false },
		/* ...and an explicit --speculative=0 on the stage-3 PP5 row
		 * keeps the sets and their clamp, which then have no DSpark to
		 * act on: each draft line is noted inactive (stage 3 review D2). */
		{ { TEST_CODE_DECODE_TUPLE, "--speculative=0" },
		  TEST_UNPARSED,
		  0u,
		  0u,
		  0u,
		  { "axes from category=code:" TEST_SINGLE_CLAMP_NOTE(
			    "--draft_block")
			    TEST_SINGLE_CLAMP_NOTE("--draft_positions")
				    TEST_INACTIVE_NOTE "\n",
		    "axes set --draft_block=5\naxes set --draft_positions=5\n",
		    "axes from focus=decode: --draft_policy "
		    "--draft_session_acceptance" TEST_INACTIVE_NOTE "\n" },
		  "from the user",
		  0u,
		  false },
		/* TP4_EXL3 BATCH: sharded draft rows need draft layers, so a
		 * focus without layer drafting is raised to 1. */
		{ { "--placement=tp4_exl3", TEST_EXL3_TTFT_TUPLE },
		  TEST_PARSES,
		  LINEAR_MAXIMUM_ROWS,
		  MODEL_DRAFT_BLOCK,
		  MODEL_DRAFT_BLOCK,
		  { TEST_FLOOR_NOTE, "axes set --tp_draft_layers=1\n", NULL },
		  "axes from focus=",
		  1u,
		  true },
		/* An explicit --tp_draft_layers wins (the engine then refuses
		 * the sharded BATCH plan at startup, as usual). */
		{ { "--placement=tp4_exl3", TEST_EXL3_TTFT_TUPLE,
		    "--tp_draft_layers=0" },
		  TEST_PARSES,
		  LINEAR_MAXIMUM_ROWS,
		  MODEL_DRAFT_BLOCK,
		  MODEL_DRAFT_BLOCK,
		  { NULL, NULL, NULL },
		  "required by",
		  0u,
		  true },
		/* Bounds act on the final values (stage 3 review D1): an explicit
		 * --tp_draft_shards=0 removes the raise... */
		{ { "--placement=tp4_exl3", TEST_EXL3_TTFT_TUPLE,
		    "--tp_draft_shards=0" },
		  TEST_PARSES,
		  LINEAR_MAXIMUM_ROWS,
		  MODEL_DRAFT_BLOCK,
		  MODEL_DRAFT_BLOCK,
		  { "axes set --tp_draft_layers=0\n", NULL, NULL },
		  "required by",
		  0u,
		  true },
		/* ...and an explicit --tp_draft_shards=1 keeps it, naming the
		 * user (on tcb88 the same command started with layers 1). */
		{ { "--placement=tp4_exl3", TEST_EXL3_TTFT_TUPLE,
		    "--tp_draft_shards=1" },
		  TEST_PARSES,
		  LINEAR_MAXIMUM_ROWS,
		  MODEL_DRAFT_BLOCK,
		  MODEL_DRAFT_BLOCK,
		  { "axes set --tp_draft_layers=1\n", TEST_USER_FLOOR_NOTE,
		    NULL },
		  "from concurrency=multiple)",
		  1u,
		  true },
		/* An explicit --batch_draft_rows=0 leaves no sharded draft
		 * rows, so no raise. */
		{ { "--placement=tp4_exl3", TEST_EXL3_TTFT_TUPLE,
		    "--batch_draft_rows=0" },
		  TEST_UNPARSED,
		  0u,
		  0u,
		  0u,
		  { "axes set --tp_draft_layers=0\n", NULL, NULL },
		  "required by",
		  0u,
		  true }
	};
	bool passed = true;
	for (size_t index = 0; index < sizeof(tests) / sizeof(tests[0]);
	     index++)
		for (unsigned serving = 0; serving < TEST_OPTION_ORDERS;
		     serving++) {
			axis_cases_run++;
			const pid_t child = fork();
			if (child == 0)
				_exit(clamp_case(&tests[index], serving != 0) ?
					      0 :
					      1);
			int status = 0;
			const bool accepted =
				child > 0 &&
				waitpid(child, &status, 0) == child &&
				WIFEXITED(status) && WEXITSTATUS(status) == 0;
			if (!accepted)
				(void)fprintf(
					stderr, "failed clamp case %zu (%s)\n",
					index,
					serving != 0 ? "serve" : "generate");
			passed = accepted && passed;
		}
	return passed;
}

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
		"    Workload kind; balanced is beneficial across the applicable measured workloads. "
		"Scope and missing mixed-user gates remain explicit per strategy.\n";
	const char *const concurrency_meaning =
		"--concurrency=single|multiple|balanced\n"
		"    C1 workload, separately qualified C2-C8 workloads, or a measured compromise across both; "
		"this selector is not session_slots or a numeric batch size.\n";
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
				choices[index].gpu_split);
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
	     single[] = "--session_slots=1", axes_off[] = TEST_AXES_OFF;
	/* The engine's raw defaults under these recipes: --axes=off (stage 3), so no base is inserted. */
	char *arguments[TEST_ARGUMENT_CAPACITY] = {
		program, strcmp(test->command, "serve") == 0 ? serve : generate,
		axes_off
	};
	char *copies[TEST_ARGUMENT_CAPACITY] = { 0 };
	int count = TEST_OPTION_START + 1;
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
		passed = axes_off_untouched(&expanded, count, arguments);
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

static bool compatible_fast_plans(const struct axis_choice *choices,
				  size_t count)
{
	bool passed = true;
	for (size_t index = 0; index < count; index++) {
		const struct axis_choice *fast = &choices[index];
		if (strcmp(fast->value[TEST_EXACTNESS], "fast") != 0)
			continue;
		const struct axis_choice *balanced =
			find_choice(choices, count, fast->value[TEST_CATEGORY],
				    "balanced", fast->value[TEST_CONCURRENCY],
				    fast->value[TEST_FOCUS], fast->gpu_split);
		passed = passed && balanced != NULL;
		if (balanced == NULL)
			continue;
		if (balanced->hold[0] != '\0') {
			passed = passed && fast->hold[0] != '\0';
			continue;
		}
		passed = passed && fast->hold[0] == '\0' &&
			 strcmp(fast->settings, balanced->settings) == 0 &&
			 strcmp(fast->serve_settings,
				balanced->serve_settings) == 0 &&
			 strcmp(fast->generate_settings,
				balanced->generate_settings) == 0 &&
			 strcmp(fast->error_class, balanced->error_class) == 0;
		if (strcmp(fast->weight_family, "exl3") == 0 &&
		    strcmp(fast->value[TEST_CATEGORY], "code") == 0 &&
		    strcmp(fast->value[TEST_CONCURRENCY], "single") == 0 &&
		    strcmp(fast->value[TEST_FOCUS], "decode") == 0) {
			struct axis_test_case override = {
				.choice = fast,
				.command = "serve",
				.placement = fast->gpu_split,
				.extra = { "--wide_rows=1024" },
				.expand_expected = true,
				.parse_expected = true,
				.placement_expected = true,
				.batch_expected = true,
				.destination = TEST_WIDE_ROWS
			};
			passed = isolated_tuple_case(&override) && passed;
			override.extras_first = true;
			passed = isolated_tuple_case(&override) && passed;
		}
	}
	size_t retained_count = 0;
	const struct axis_retained_route *retained =
		axis_retained_routes_entry(&retained_count);
	passed = passed && retained_count > 0;
	for (size_t index = 0; index < retained_count; index++)
		passed = passed && retained[index].hold[0] != '\0' &&
			 retained[index].fallback[0] != '\0' &&
			 retained[index].constraints[0] != '\0';
	return passed;
}

/* Retain the banked raw qualification and metrics interaction when no axis
 * selects that route. Real parser, geometry and linked-module checks only. */
static bool raw_banked_qualification(const bool with_metrics)
{
	static const char *const options[] = {
		"serve",
		"--placement=pp5",
		"--numerics=exact",
		"--cmp_order=reference",
		"--linear_order=1",
		"--linear_profile=0",
		"--pro_emulation=0",
		"--card_order=0,1,3,4,2",
		("--layout=" TEST_ORIGINAL_LAYOUT),
		"--wide_rows=1024",
		"--rows=8",
		"--full_steps=1",
		"--prefill_cone=1",
		"--chunks_in_flight=1",
		"--step_graphs=1",
		"--draft=dspark",
		"--draft_block=5",
		"--draft_positions=5",
		"--card_memory_kept_free_mib=0",
		"--original_head_offload=2",
		"--pipeline_steps=5",
		"--official_hc_rows=0",
		"--official_reference=0",
		"--context_tokens=65536",
		"--draft_policy=1",
		"--draft_base_us=12700",
		"--draft_tail_us=2200",
		"--draft_row_us=2700",
		"--draft_learned_acceptance=0",
		"--draft_session_acceptance=6",
		"--batch_plan_ties=0",
		"--tp_pipeline_overlap=0",
		"--original_chunk_core=0",
		"--original_dense_staged=0",
		"--original_wide_experts=0",
		"--session_slots=8",
		"--prefix_reuse=1",
		"--prefill_order=0",
		"--original_batch=0",
	};
	const pid_t child = fork();
	if (child == 0) {
		char program[] = "raw-banked", tables[] = "front_end.tables",
		     pack_path[] = "original.anemone";
		char metrics_on[] = "--metrics=1",
		     metrics_off[] = "--metrics=0";
		char *arguments[TEST_ARGUMENT_CAPACITY] = { program };
		char *copies[sizeof(options) / sizeof(options[0])] = { 0 };
		int count = 1;
		bool good = true;
		for (size_t index = 0;
		     index < sizeof(options) / sizeof(options[0]); index++) {
			copies[index] = strdup(options[index]);
			if (copies[index] == NULL) {
				good = false;
				break;
			}
			arguments[count] = copies[index];
			count++;
		}
		arguments[count] = with_metrics ? metrics_on : metrics_off;
		count++;
		arguments[count] = tables;
		count++;
		arguments[count] = pack_path;
		count++;
		if (good)
			good = parse_arguments(count, arguments);
		if (good)
			good = startup_placement_status() == STATUS_OK;
		static const struct axis_choice fixture = {
			.value = { "balanced", "balanced", "single", "decode" },
			.weight_family = "original",
			.gpu_split = "pp5"
		};
		if (good)
			good = original_tuple_geometry(&fixture, "serve") &&
			       metrics == (with_metrics ? 1u : 0u);
		for (size_t index = 0;
		     index < sizeof(copies) / sizeof(copies[0]); index++)
			free(copies[index]);
		_exit(good ? 0 : 1);
	}
	int status = 0;
	bool passed = false;
	if (child > 0) {
		const pid_t waited = waitpid(child, &status, 0);
		passed = waited == child && WIFEXITED(status) &&
			 WEXITSTATUS(status) == 0;
	}
	axis_cases_run++;
	return passed;
}

#define TEST_ENGRAM_OPTIONS 5u
#define TEST_ENGRAM_CACHE_MIB 256u
#define TEST_ENGRAM_CACHE_MOST_MIB 32768u
#define TEST_ENGRAM_HALF 50u
#define TEST_ENGRAM_HOT_SHA256 \
	"--engram_hot_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
/* An engine argument vector: the command, the options, then its files. Returns the count. */
static int
engram_arguments(const char *const *options, bool generate,
		 char values[TEST_ARGUMENT_CAPACITY][TEST_OPTION_BYTES],
		 char *arguments[TEST_ARGUMENT_CAPACITY])
{
	int count = 0;
	const char *first[] = { "engine", generate ? "generate" : "serve" };
	for (size_t word = 0; word < sizeof(first) / sizeof(first[0]); word++) {
		(void)snprintf(values[count], sizeof(values[count]), "%s",
			       first[word]);
		arguments[count] = values[count];
		count++;
	}
	for (uint32_t option = 0;
	     option < TEST_ENGRAM_OPTIONS && options[option] != NULL;
	     option++) {
		(void)snprintf(values[count], sizeof(values[count]), "%s",
			       options[option]);
		arguments[count] = values[count];
		count++;
	}
	const char *files[] = { "prompt.json", "tokenizer.bin",
				"model.anemone" };
	for (size_t file = generate ? 0u : 1u;
	     file < sizeof(files) / sizeof(files[0]); file++) {
		(void)snprintf(values[count], sizeof(values[count]), "%s",
			       files[file]);
		arguments[count] = values[count];
		count++;
	}
	arguments[count] = NULL;
	return count;
}

/* parses: the parse and the settings after the storage profile (engram_settings_resolve) both accept. */
static bool engram_parser_cases(void)
{
	static const struct {
		const char *options[TEST_ENGRAM_OPTIONS];
		bool generate, parses;
		uint32_t percent, io, cache, turbo;
		uint32_t step_io; /* step I/O is the default serving PP5 or TP4_PRO from disk tables */
		const char *sources;
	} cases[] = {
		{ { NULL }, false, true, ENGRAM_RAM_PERCENT, 0, 0, 0, 0, NULL },
		{ { "--engram_ram_percent=50", "--engram_io=1",
		    "--engram_cache_mib=256", "--turbo-nvme-engrams=1",
		    "--engram_sources=/table-a,/table-b" },
		  false,
		  true,
		  TEST_ENGRAM_HALF,
		  1,
		  TEST_ENGRAM_CACHE_MIB,
		  1,
		  1,
		  "/table-a,/table-b" },
		{ { "--engram_ram_percent=0" }, false, true, 0, 0, 0, 0, 1, NULL },
		{ { "--placement=tp4_pro", "--engram_ram_percent=0" },
		  false,
		  true,
		  0,
		  0,
		  0,
		  0,
		  1,
		  NULL },
		/* the other GPU splits keep the per-row gathers until measured there */
		{ { "--placement=tp4_exl3", "--engram_ram_percent=0" },
		  false,
		  true,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		/* an explicit 0 keeps the per-row gathers */
		{ { "--engram_ram_percent=0", "--engram_step_io=0" },
		  false,
		  true,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_ram_percent=0", "--engram_io=1",
		    "--engram_sources=/table-a" },
		  true,
		  true,
		  0,
		  1,
		  0,
		  0,
		  0,
		  "/table-a" },
		{ { "--engram_ram_percent=101" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_ram_percent=0", "--engram_cache_mib=32768" },
		  false,
		  true,
		  0,
		  0,
		  TEST_ENGRAM_CACHE_MOST_MIB,
		  0,
		  1,
		  NULL },
		{ { "--engram_ram_percent=0", "--engram_cache_mib=32769" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_ram_percent=0", "--engram_io=2" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_ram_percent=0", "--turbo-nvme-engrams=2" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_ram_percent=0", "--engram_ram_percent=50" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_ram_percent=0", "--engram_sources=" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_cache_mib=1" }, false, false, 0, 0, 0, 0, 0, NULL },
		{ { "--engram_io=1" }, false, false, 0, 0, 0, 0, 0, NULL },
		{ { "--turbo-nvme-engrams=1" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_sources=/table-a" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		/* generate takes the hot set beside its sources; a hot set needs its SHA-256 and a disk store */
		{ { "--engram_ram_percent=0", "--engram_io=1",
		    "--engram_sources=/table-a", "--engram_hot_set=/hot",
		    TEST_ENGRAM_HOT_SHA256 },
		  true,
		  true,
		  0,
		  1,
		  0,
		  0,
		  0,
		  "/table-a" },
		{ { "--engram_ram_percent=0", "--engram_hot_set=/hot" },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL },
		{ { "--engram_hot_set=/hot", TEST_ENGRAM_HOT_SHA256 },
		  false,
		  false,
		  0,
		  0,
		  0,
		  0,
		  0,
		  NULL }
	};
	bool passed = true;
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		const pid_t child = fork();
		if (child == 0) {
			char values[TEST_ARGUMENT_CAPACITY][TEST_OPTION_BYTES];
			char *arguments[TEST_ARGUMENT_CAPACITY];
			const int count = engram_arguments(
				cases[index].options, cases[index].generate,
				values, arguments);
			const bool parsed = parse_arguments(count, arguments) &&
					    engram_settings_resolve();
			bool matches = parsed == cases[index].parses;
			if (parsed)
				matches = matches &&
					  engram_ram_percent ==
						  cases[index].percent &&
					  engram_io == cases[index].io &&
					  engram_cache_mib ==
						  cases[index].cache &&
					  engram_step_io ==
						  cases[index].step_io &&
					  turbo_nvme_engrams ==
						  cases[index].turbo &&
					  ((engram_sources_text == NULL &&
					    cases[index].sources == NULL) ||
					   (engram_sources_text != NULL &&
					    cases[index].sources != NULL &&
					    strcmp(engram_sources_text,
						   cases[index].sources) == 0));
			_exit(matches ? 0 : 1);
		}
		int status = 0;
		passed = child > 0 && waitpid(child, &status, 0) == child &&
			 WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
			 passed;
		axis_cases_run++;
	}
	return passed;
}

#define TEST_READERS_DEFAULT \
	4u /* the PP loader's readers a copy since the real PP5 serve loads of 15 Nov 2026 */
#define TEST_READERS_EARLIER \
	2u /* the default before it, which the knob still reaches */
/* The PP loader's readers a copy as the parser leaves them, for generate and serve: the default without the knob,
 * and --reader_threads_per_copy=2 when it is given. */
static bool loader_readers_cases(void)
{
	static const struct {
		const char *options[TEST_ENGRAM_OPTIONS];
		uint32_t readers;
	} cases[] = { { { NULL }, TEST_READERS_DEFAULT },
		      { { "--reader_threads_per_copy=2" },
			TEST_READERS_EARLIER } };
	bool passed = LOADER_DEFAULT_READER_THREADS_PER_COPY ==
		      TEST_READERS_DEFAULT;
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++)
		for (unsigned generate = 0; generate < TEST_OPTION_ORDERS;
		     generate++) {
			const uint32_t readers = cases[index].readers;
			const pid_t child = fork();
			if (child == 0) {
				char values[TEST_ARGUMENT_CAPACITY]
					   [TEST_OPTION_BYTES];
				char *arguments[TEST_ARGUMENT_CAPACITY];
				const int count = engram_arguments(
					cases[index].options, generate != 0,
					values, arguments);
				const bool reached =
					parse_arguments(count, arguments) &&
					reader_threads_per_copy == readers;
				_exit(reached ? 0 : 1);
			}
			int status = 0;
			const bool matched =
				child > 0 &&
				waitpid(child, &status, 0) == child &&
				WIFEXITED(status) && WEXITSTATUS(status) == 0;
			if (!matched)
				(void)fprintf(
					stderr,
					"readers a copy: case %zu (%s) is not %u\n",
					index, generate ? "generate" : "serve",
					readers);
			passed = matched && passed;
			axis_cases_run++;
		}
	return passed;
}

#define TEST_STEP_COPIES 3u
#define TEST_PATH_BYTES 512u
#define TEST_DIRECTORY_BYTES 256u
#define TEST_STEP_WAIT_SHARE \
	2u /* a step's whole wait (its read deadline times copies + 1) in half the cards' wait limit */
#define TEST_PROFILE_RAM_PERCENT \
	0u /* what storage_profile_apply leaves from a tuned turbo_engram: the tuner writes ram_percent 0 */
/* A storage profile puts the tables on disk after the parse (review findings 3 and 11): step I/O's default and its
 * guards follow the final ram percent, so profile-driven PP5 and TP4_PRO serving gets the default and takes the
 * explicit knobs, which the parse refused before. */
static bool engram_profile_cases(void)
{
	static const struct {
		const char *options[TEST_ENGRAM_OPTIONS];
		bool generate, accepted;
		uint32_t step_io;
	} cases[] = {
		{ { NULL }, false, true, 1 },
		{ { "--placement=tp4_pro" }, false, true, 1 },
		{ { "--placement=tp4_exl3" }, false, true, 0 },
		{ { "--engram_step_io=0" }, false, true, 0 },
		{ { "--engram_step_io=1" }, false, true, 1 },
		{ { "--engram_hot_set=/hot", TEST_ENGRAM_HOT_SHA256 },
		  false,
		  true,
		  1 },
		{ { NULL }, true, true, 0 },
	};
	bool passed = true;
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		const pid_t child = fork();
		if (child == 0) {
			char values[TEST_ARGUMENT_CAPACITY][TEST_OPTION_BYTES];
			char *arguments[TEST_ARGUMENT_CAPACITY];
			const int count = engram_arguments(
				cases[index].options, cases[index].generate,
				values, arguments);
			const bool parsed = parse_arguments(count, arguments);
			engram_ram_percent = TEST_PROFILE_RAM_PERCENT;
			const bool accepted = parsed &&
					      engram_settings_resolve();
			_exit(accepted == cases[index].accepted &&
					      (!accepted ||
					       engram_step_io ==
						       cases[index].step_io) ?
				      0 :
				      1);
		}
		int status = 0;
		passed = child > 0 && waitpid(child, &status, 0) == child &&
			 WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
			 passed;
		axis_cases_run++;
	}
	return passed;
}

/* Stage 3: the knobs a base omits as its split's raw engine defaults, read as the parser leaves them. */
static const struct {
	const char *key;
	const uint32_t *value;
} raw_default_numbers[] = {
	{ "--linear_profile", &linear_profile },
	{ "--linear_order", &linear_order },
	{ "--pro_emulation", &pro_emulation },
	{ "--full_steps", &full_steps },
	{ "--chunks_in_flight", &chunks_in_flight },
	{ "--draft_block", &draft_block },
	{ "--card_memory_kept_free_mib", &card_memory_kept_free_mib },
	{ "--official_hc_rows", &official_hc_rows },
	{ "--tp_pipeline_overlap", &tp_pipeline_overlap },
	{ "--original_chunk_core", &original_chunk_core },
	{ "--original_dense_staged", &original_dense_staged },
	{ "--batch_plan_ties", &batch_plan_ties },
	{ "--draft_learned_acceptance", &draft_learned_acceptance },
	{ "--draft_session_acceptance", &draft_session_acceptance },
	{ "--draft_base_us", &draft_policy.base_us },
	{ "--draft_tail_us", &draft_policy.tail_us },
	{ "--draft_row_us", &draft_policy.row_us }
};
/* the numbers above, then numerics and cmp_order */
#define RAW_DEFAULT_ENUMS 2u
#define RAW_DEFAULT_VALUES                                              \
	(sizeof(raw_default_numbers) / sizeof(raw_default_numbers[0]) + \
	 RAW_DEFAULT_ENUMS)

/* One parse of a split's command, with the pin or without it, in a child: every defaulted value, through a pipe. */
static bool raw_default_parse(const char *split, const char *pin, bool serving,
			      uint64_t values[RAW_DEFAULT_VALUES])
{
	int channel[TEST_PIPE_ENDS];
	if (pipe(channel) != 0)
		return false;
	const pid_t child = fork();
	if (child == 0) {
		(void)close(channel[0]);
		char program[] = "anemone", generate[] = "generate",
		     serve[] = "serve", messages[] = "messages",
		     tables[] = "tables", weights[] = "weights",
		     placement[TEST_PLACEMENT_BYTES], pinned[TEST_OPTION_BYTES];
		(void)snprintf(placement, sizeof(placement), "--placement=%s",
			       split);
		(void)snprintf(pinned, sizeof(pinned), "%s",
			       pin != NULL ? pin : "");
		char *arguments[TEST_ARGUMENT_CAPACITY] = {
			program, serving ? serve : generate, placement
		};
		int count = TEST_OPTION_START + 1;
		if (pin != NULL)
			arguments[count++] = pinned;
		if (!serving)
			arguments[count++] = messages;
		arguments[count++] = tables;
		arguments[count++] = weights;
		uint64_t parsed[RAW_DEFAULT_VALUES] = { 0 };
		const bool good = parse_arguments(count, arguments);
		for (size_t index = 0;
		     index + RAW_DEFAULT_ENUMS < RAW_DEFAULT_VALUES; index++)
			parsed[index] = *raw_default_numbers[index].value;
		parsed[RAW_DEFAULT_VALUES - RAW_DEFAULT_ENUMS] =
			(uint64_t)numerics;
		parsed[RAW_DEFAULT_VALUES - 1u] = (uint64_t)cmp_order;
		_exit(good && write(channel[1], parsed, sizeof(parsed)) ==
					      (ssize_t)sizeof(parsed) ?
			      0 :
			      1);
	}
	(void)close(channel[1]);
	const bool read_all =
		child > 0 &&
		read(channel[0], values,
		     RAW_DEFAULT_VALUES * sizeof(values[0])) ==
			(ssize_t)(RAW_DEFAULT_VALUES * sizeof(values[0]));
	(void)close(channel[0]);
	int status = 0;
	return read_all && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/* Each pin a split's base omits as its raw default changes no parsed value, for generate and serve; every omitted
 * key is one this test reads. */
static bool raw_default_cases(void)
{
	size_t count = 0;
	const struct axis_base *const bases = axis_bases_entry(&count);
	bool passed = count == TEST_SPLITS;
	for (size_t entry = 0; entry < count; entry++) {
		char *const pins = strdup(bases[entry].raw_defaults);
		char *save = NULL;
		passed = passed && pins != NULL;
		for (char *pin = pins == NULL ? NULL :
						strtok_r(pins, "\n", &save);
		     pin != NULL; pin = strtok_r(NULL, "\n", &save)) {
			const size_t key_bytes = strcspn(pin, "=");
			bool known =
				strncmp(pin, "--numerics", key_bytes) == 0 ||
				strncmp(pin, "--cmp_order", key_bytes) == 0;
			for (size_t index = 0;
			     !known &&
			     index + RAW_DEFAULT_ENUMS < RAW_DEFAULT_VALUES;
			     index++)
				known = strlen(raw_default_numbers[index].key) ==
						key_bytes &&
					strncmp(pin,
						raw_default_numbers[index].key,
						key_bytes) == 0;
			for (unsigned serving = 0; serving < TEST_OPTION_ORDERS;
			     serving++) {
				uint64_t raw[RAW_DEFAULT_VALUES],
					pinned[RAW_DEFAULT_VALUES];
				const bool same =
					known &&
					raw_default_parse(
						bases[entry].gpu_split, NULL,
						serving != 0, raw) &&
					raw_default_parse(
						bases[entry].gpu_split, pin,
						serving != 0, pinned) &&
					memcmp(raw, pinned, sizeof(raw)) == 0;
				axis_cases_run++;
				if (!same)
					(void)fprintf(
						stderr,
						"raw default %s changes the parse on %s (%s)\n",
						pin, bases[entry].gpu_split,
						serving ? "serve" : "generate");
				passed = passed && same;
			}
		}
		free(pins);
	}
	return passed;
}

/* Default step I/O falls back to per-row gathers when its flight cannot start; an explicit one fails the start
 * (review finding 2). Without a disk store engram_steps_start refuses, as a host that refuses io_uring does. */
static bool engram_fallback_cases(void)
{
	const pid_t child = fork();
	if (child == 0) {
		engram_disk = NULL;
		engram_step_io = 1;
		engram_step_io_default = true;
		const bool fell_back =
			engram_steps_start_or_fall_back() == STATUS_OK &&
			engram_step_io == 0 && engram_steps == NULL;
		engram_step_io = 1;
		engram_step_io_default = false;
		const bool refused = engram_steps_start_or_fall_back() ==
					     STATUS_INVALID_ARGUMENT &&
				     engram_step_io == 1;
		_exit(fell_back && refused ? 0 : 1);
	}
	int status = 0;
	axis_cases_run++;
	return child > 0 && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/* The pack's header at the start of its bytes (the pack writer aligns it). */
static const void *pack_bytes_header(const unsigned char *bytes)
{
	return bytes;
}

/* In a child: the test pack in three files, opened as commands.c opens a disk store, with its engram attached. The
 * files are in a folder of their own under TMPDIR, else /var/tmp (a disk where /tmp is a tmpfs, as on Debian),
 * whatever the working directory is; the step I/O does not start on a tmpfs, so a store there is refused by name. */
static bool engram_store_open(char paths[TEST_STEP_COPIES][TEST_PATH_BYTES],
			      char *directory, size_t directory_bytes)
{
	unsigned char *bytes = NULL;
	struct engram_record *memory_engram = NULL;
	if (test_engram_pack_open(&bytes, &memory_engram) != STATUS_OK)
		return false;
	const uint64_t pack_bytes =
		((const struct pack_header *)pack_bytes_header(bytes))
			->file_byte_count;
	const char *const temporary = getenv("TMPDIR");
	const int length = snprintf(directory, directory_bytes,
				    "%s/anemone-startup-axes-XXXXXX",
				    temporary != NULL ? temporary : "/var/tmp");
	struct statfs file_system;
	const bool made = length > 0 && (size_t)length < directory_bytes &&
			  mkdtemp(directory) != NULL;
	if (!made)
		directory[0] = '\0'; /* no folder of its own: none to remove */
	bool written = made && statfs(directory, &file_system) == 0;
	if (written && file_system.f_type == TMPFS_MAGIC) {
		(void)fprintf(
			stderr,
			"startup_axes_test: the Engram store %s is on a tmpfs, where the step I/O does not start: set "
			"TMPDIR to a folder on a disk\n",
			directory);
		written = false;
	}
	const char *names[TEST_STEP_COPIES];
	for (uint32_t copy = 0; written && copy < TEST_STEP_COPIES; copy++) {
		(void)snprintf(paths[copy], TEST_PATH_BYTES,
			       "%s/copy%u.anemone", directory, copy);
		names[copy] = paths[copy];
		const int descriptor = open(
			paths[copy], O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			S_IRUSR | S_IWUSR);
		written = descriptor >= 0 &&
			  write(descriptor, bytes, (size_t)pack_bytes) ==
				  (ssize_t)pack_bytes;
		if (descriptor >= 0)
			(void)close(descriptor);
		else
			paths[copy][0] = '\0'; /* not made: not removed */
	}
	engram_close(memory_engram);
	free(bytes);
	result_status status =
		written ? storage_direct.reader_open(names, TEST_STEP_COPIES,
						     &reader) :
			  STATUS_INPUT_OUTPUT_ERROR;
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, reader, &pack) :
			 status;
	status = status == STATUS_OK ?
			 engram_open(&storage_direct, reader, pack, &engram) :
			 status;
	status = status == STATUS_OK ?
			 engram_disk_open(engram_description_of(engram), pack,
					  &storage_direct, reader, 0, 0, 1, 1,
					  &engram_disk) :
			 status;
	status = status == STATUS_OK ? engram_attach_disk(engram, engram_disk) :
				       status;
	return status == STATUS_OK;
}

/* Step I/O as serving starts it (engram_steps_start): a read's deadline puts its re-reads on every other copy inside
 * half the cards' wait limit (review N1: before, the storage limit, 180 s a read against the 10 s a pipeline waits
 * for progress). With checking hooks, a flight the host refuses (its ring) makes the default fall back to per-row
 * gathers and an explicit --engram_step_io=1 fail, through the real open (review finding 2). */
static bool engram_steps_start_cases(void)
{
	const pid_t child = fork();
	if (child == 0) {
		/* empty until made: the cleanup below removes only what was made */
		char paths[TEST_STEP_COPIES][TEST_PATH_BYTES] = { { 0 } },
		     directory[TEST_DIRECTORY_BYTES] = { 0 };
		bool passed =
			engram_store_open(paths, directory, sizeof(directory));
		engram_io_threads = 1;
		engram_io_pin = 0;
		engram_step_io = 1;
#if CHECKING_HOOKS_ENABLED
		engram_step_io_default = true;
		passed = passed && fault_arm("storage_flight_ring", 1) &&
			 engram_steps_start_or_fall_back() == STATUS_OK &&
			 engram_step_io == 0 && engram_steps == NULL &&
			 fault_disarm() != 0;
		engram_step_io = 1;
		engram_step_io_default = false;
		passed = passed && fault_arm("storage_flight_ring", 1) &&
			 engram_steps_start_or_fall_back() ==
				 STATUS_UNSUPPORTED &&
			 engram_step_io == 1 && fault_disarm() != 0;
		engram_steps_close(engram_steps);
		engram_steps = NULL;
#endif
		passed = passed && engram_steps_start() == STATUS_OK;
		const uint64_t limit =
			passed ? storage_flight_read_limit(
					 engram_steps_flight(engram_steps)) :
				 UINT64_MAX;
		passed = passed &&
			 limit <= CARD_WAIT_LIMIT_NANOSECONDS /
					  (TEST_STEP_WAIT_SHARE *
					   ((uint64_t)TEST_STEP_COPIES + 1u)) &&
			 limit != 0;
		engram_close(engram);
		engram_steps_close(engram_steps);
		engram_disk_close(engram_disk);
		pack_close(pack);
		storage_direct.reader_close(reader);
		for (uint32_t copy = 0; copy < TEST_STEP_COPIES; copy++)
			if (paths[copy][0] != '\0')
				(void)unlink(paths[copy]);
		if (directory[0] != '\0')
			(void)rmdir(directory);
		_exit(passed ? 0 : 1);
	}
	int status = 0;
	axis_cases_run++;
	return child > 0 && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/* Exact on TP4_EXL3 (owner, 6 Nov: the same shaping and strictness, not official-exact): the exact-shaping set is
 * inserted, never the THIN base's relaxed numerics or BF16 wire; startup names it; a conflicting override refuses. */
#define TEST_EXL3_EXACT_CLASS \
	"axes class=exact shaping on quantized weights (not official-exact)\n"

static bool exl3_exact_case(bool serving, const char *override)
{
	char program[] = "anemone", generate[] = "generate", serve[] = "serve",
	     placement[] = "--placement=tp4_exl3",
	     exactness[] = "--exactness=exact", messages[] = "messages",
	     tables[] = "tables", weights[] = "weights",
	     extra[TEST_OPTION_BYTES];
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, serving ?
								     serve :
								     generate };
	int count = TEST_OPTION_START;
	arguments[count++] = placement;
	arguments[count++] = exactness;
	if (override != NULL) {
		(void)snprintf(extra, sizeof(extra), "%s", override);
		arguments[count++] = extra;
	}
	if (!serving)
		arguments[count++] = messages;
	arguments[count++] = tables;
	arguments[count++] = weights;
	size_t rows = 0;
	const struct axis_choice *const all = axis_choices_entry(&rows);
	const struct axis_test_case test = {
		.choice = find_choice(all, rows, "balanced", "exact", "single",
				      "balanced", "tp4_exl3"),
		.command = serving ? "serve" : "generate"
	};
	struct axis_arguments_record expanded = { 0 };
	const bool expanded_ok = test.choice != NULL &&
				 axis_expand(count, arguments, &expanded);
	bool passed = expanded_ok == (override == NULL);
	if (expanded_ok) {
		char *log_text = NULL;
		size_t log_bytes = 0;
		FILE *log = open_memstream(&log_text, &log_bytes);
		if (log != NULL)
			axis_log(&expanded, log);
		passed = passed && log != NULL && fclose(log) == 0 &&
			 log_text != NULL &&
			 strstr(log_text, TEST_EXL3_EXACT_CLASS) != NULL &&
			 strstr(log_text, "relaxed") == NULL &&
			 expanded.choice == test.choice &&
			 declared_settings_match(&test, &expanded, count,
						 arguments) &&
			 parse_arguments(expanded.count, expanded.values) &&
			 ordinary_tuple_geometry(test.choice, test.command);
		free(log_text);
	}
	axis_arguments_free(&expanded);
	return passed;
}

static bool exl3_exact_cases(void)
{
	static const char *const overrides[] = { NULL, "--numerics=relaxed",
						 "--tp_wire_bf16=1" };
	bool passed = true;
	for (size_t index = 0; index < sizeof(overrides) / sizeof(overrides[0]);
	     index++)
		for (unsigned serving = 0; serving < TEST_OPTION_ORDERS;
		     serving++) {
			axis_cases_run++;
			const pid_t child = fork();
			if (child == 0)
				_exit(exl3_exact_case(serving != 0,
						      overrides[index]) ?
					      0 :
					      1);
			int status = 0;
			const bool accepted =
				child > 0 &&
				waitpid(child, &status, 0) == child &&
				WIFEXITED(status) && WEXITSTATUS(status) == 0;
			if (!accepted)
				(void)fprintf(
					stderr,
					"failed EXL3 exact case %s (%s)\n",
					overrides[index] != NULL ?
						overrides[index] :
						"the set",
					serving != 0 ? "serve" : "generate");
			passed = accepted && passed;
		}
	return passed;
}

/* Stage 3, the owner's balanced rule on the TP splits (tensor_plan_fit in commands.c): each split's serve without an
 * axis flag parses --session_slots=max at SESSION_SLOTS_MAX_PARSED with 1M context, and its TP plan on the tables pack
 * (model_v41_pack.h) is made at the most sessions the cards hold: cards with room for TEST_TP_FIT_SESSIONS sessions'
 * arenas and no more resolve to that count, cards of TEST_TP_FIT_ROOMY_BYTES to SERVING_SLOTS_MOST, cards a
 * byte short of one session refuse at one with STATUS_OUT_OF_MEMORY, and an explicit count is never retried. */
#define TEST_TP_FIT_SESSIONS 3u
#define TEST_TP_FIT_ROOMY_BYTES (1ull << 42u)
/* production's free memory per card (model_v41_placement_test.c): a CMP 170HX, the RTX PRO 6000 */
#define TEST_PRODUCTION_CMP_FREE_MIB 64512u
#define TEST_PRODUCTION_PRO_FREE_MIB 96608u

struct tensor_fit_split {
	const char *placement;
	enum test_model_v41_change_kind pack_kind;
	uint32_t cards;
	bool pro_last; /* the last card is the native RTX PRO (TP4_PRO); else every card is a CMP */
};

/* Every card's room: free bytes, with the backend's reserve and the kept-free bytes beside them. */
static void tensor_fit_room(const struct tensor_fit_split *split,
			    const uint64_t *free_bytes)
{
	for (uint32_t card = 0; card < split->cards; card++) {
		descriptions[card] = (struct card_description_record){
			.arithmetic_class =
				split->pro_last && card + 1u == split->cards ?
					ARITHMETIC_CLASS_SM_120_NATIVE :
					ARITHMETIC_CLASS_SM_80,
			.memory_total_bytes = TEST_TP_FIT_ROOMY_BYTES,
			.memory_free_bytes = free_bytes[card],
			.memory_reserved_bytes =
				BACKEND_CUDA_MEMORY_RESERVED_BYTES
		};
	}
}

static void tensor_fit_release(void)
{
	v41_tp2_model_destroy(tensor_model);
	tensor_model = NULL;
	plan = NULL;
}

/* One fit at the given rooms from the given sessions: its status, the count it ended at, each card's arena. */
MUST_CHECK static result_status tensor_fit(const struct tensor_fit_split *split,
					   const uint64_t *free_bytes,
					   uint32_t sessions,
					   uint32_t *card_of_unit,
					   uint32_t *ended, uint64_t *arenas)
{
	tensor_fit_room(split, free_bytes);
	struct model_limits limits = options_limits();
	limits.session_capacity = sessions;
	const result_status status = tensor_plan_fit(
		&limits, (uint64_t)card_memory_kept_free_mib * BYTES_PER_MIB,
		card_of_unit);
	*ended = limits.session_capacity;
	for (uint32_t card = 0; card < split->cards; card++)
		arenas[card] = status == STATUS_OK ?
				       tensor_model->stages[card]
					       ->model->cards[card]
					       .arena_byte_count :
				       0u;
	tensor_fit_release();
	return status;
}

static bool tensor_fit_case(const struct tensor_fit_split *split)
{
	char program[] = "anemone", serve[] = "serve",
	     placement[TEST_PLACEMENT_BYTES], tables[] = "tables",
	     weights[] = "weights";
	(void)snprintf(placement, sizeof(placement), "%s", split->placement);
	char *arguments[] = { program, serve, placement, tables, weights };
	struct axis_arguments_record expanded = { 0 };
	bool passed =
		axis_expand((int)(sizeof(arguments) / sizeof(arguments[0])),
			    arguments, &expanded) &&
		parse_arguments(expanded.count, expanded.values) &&
		session_slots_max &&
		session_slots == SESSION_SLOTS_MAX_PARSED &&
		context_tokens == TEST_OWNER_CONTEXT_TOKENS;
	card_count = split->cards;
	const struct test_model_v41_change change = { .kind = split->pack_kind };
	passed = passed &&
		 test_model_v41_draft_pack_open(change, &pack) == STATUS_OK;
	uint32_t *const card_of_unit =
		passed ? calloc(pack_header_of(pack)->unit_count,
				sizeof(*card_of_unit)) :
			 NULL;
	const uint64_t kept =
		(uint64_t)card_memory_kept_free_mib * BYTES_PER_MIB;
	uint64_t roomy[BACKEND_MAXIMUM_CARDS],
		room[BACKEND_MAXIMUM_CARDS] = { 0 }, one[BACKEND_MAXIMUM_CARDS],
		fitted[BACKEND_MAXIMUM_CARDS], more[BACKEND_MAXIMUM_CARDS],
		arenas[BACKEND_MAXIMUM_CARDS];
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++)
		roomy[card] = TEST_TP_FIT_ROOMY_BYTES;
	uint32_t ended = 0;
	passed = passed && card_of_unit != NULL;
	/* Each card's arena at one session, at the fixture's count and at one more, on cards that hold them all. */
	session_slots_max = false;
	passed = passed &&
		 tensor_fit(split, roomy, 1u, card_of_unit, &ended, one) ==
			 STATUS_OK &&
		 tensor_fit(split, roomy, TEST_TP_FIT_SESSIONS, card_of_unit,
			    &ended, fitted) == STATUS_OK &&
		 tensor_fit(split, roomy, TEST_TP_FIT_SESSIONS + 1u,
			    card_of_unit, &ended, more) == STATUS_OK;
	bool grows = false;
	for (uint32_t card = 0; passed && card < split->cards; card++) {
		grows = grows || more[card] > fitted[card];
		room[card] = fitted[card] + BACKEND_CUDA_MEMORY_RESERVED_BYTES +
			     kept;
	}
	passed = passed && grows;
	/* An explicit count is made once: one session more than the room holds is refused as it is. */
	passed = passed &&
		 tensor_fit(split, room, TEST_TP_FIT_SESSIONS + 1u,
			    card_of_unit, &ended,
			    arenas) == STATUS_OUT_OF_MEMORY &&
		 ended == TEST_TP_FIT_SESSIONS + 1u;
	/* max: the room's count, every session on cards that hold them all, and none a byte short of one session. */
	session_slots_max = true;
	passed = passed &&
		 tensor_fit(split, room, SERVING_SLOTS_MOST, card_of_unit,
			    &ended, arenas) == STATUS_OK &&
		 ended == TEST_TP_FIT_SESSIONS &&
		 memcmp(arenas, fitted, split->cards * sizeof(arenas[0])) == 0;
	passed = passed &&
		 tensor_fit(split, roomy, SERVING_SLOTS_MOST, card_of_unit,
			    &ended, arenas) == STATUS_OK &&
		 ended == SERVING_SLOTS_MOST;
	for (uint32_t card = 0; passed && card < split->cards; card++)
		room[card] =
			card == 0u ?
				one[card] + BACKEND_CUDA_MEMORY_RESERVED_BYTES +
					kept - 1u :
				TEST_TP_FIT_ROOMY_BYTES;
	passed = passed &&
		 tensor_fit(split, room, SERVING_SLOTS_MOST, card_of_unit,
			    &ended, arenas) == STATUS_OUT_OF_MEMORY &&
		 ended == 1u;
	/* What max resolves to at production's free memory: recorded, not asserted (request 5 boots it). */
	for (uint32_t card = 0; passed && card < split->cards; card++)
		room[card] =
			(uint64_t)(split->pro_last &&
						   card + 1u == split->cards ?
					   TEST_PRODUCTION_PRO_FREE_MIB :
					   TEST_PRODUCTION_CMP_FREE_MIB) *
			BYTES_PER_MIB;
	const result_status production =
		passed ? tensor_fit(split, room, SERVING_SLOTS_MOST,
				    card_of_unit, &ended, arenas) :
			 STATUS_INVALID_ARGUMENT;
	if (passed)
		(void)printf("TP session fit: %s serve at production's free "
			     "memory: %s, max -> %" PRIu32 " of %" PRIu32
			     " tokens\n",
			     split->placement, status_text(production),
			     production == STATUS_OK ? ended : 0u,
			     context_tokens);
	(void)fflush(stdout);
	free(card_of_unit);
	pack_close(pack);
	pack = NULL;
	axis_arguments_free(&expanded);
	return passed;
}

static bool tensor_fit_cases(void)
{
	static const struct tensor_fit_split splits[] = {
		{ "--placement=tp4_pro", TEST_MODEL_V41_CHANGE_NONE,
		  V41_DATA_TP2_PROGRAM_CARDS, true },
		{ "--placement=tp4_exl3", TEST_MODEL_V41_CHANGE_EXL3, TP4_RANKS,
		  false }
	};
	bool passed = true;
	for (size_t index = 0; index < sizeof(splits) / sizeof(splits[0]);
	     index++) {
		axis_cases_run++;
		const pid_t child = fork();
		if (child == 0)
			_exit(tensor_fit_case(&splits[index]) ? 0 : 1);
		int status = 0;
		const bool fitted =
			child > 0 && waitpid(child, &status, 0) == child &&
			WIFEXITED(status) && WEXITSTATUS(status) == 0;
		if (!fitted)
			(void)fprintf(stderr,
				      "failed TP session fit: %s serve\n",
				      splits[index].placement);
		passed = fitted && passed;
	}
	return passed;
}

/* A raw TP4_PRO serve of several slots (serve's default max) with the device draft and no depth takes
 * depth 2, so a default never turns a working serve into a refusal; an explicit depth 1 keeps today's refusal, one
 * slot keeps depth 1, and an explicit depth stays. */
struct raw_tp_depth_test {
	const char *given; /* NULL: neither depth nor slots given */
	bool parse_expected, defaulted;
	uint32_t depth;
};

static bool raw_tp_depth_case(const struct raw_tp_depth_test *test)
{
	char program[] = "anemone", serve[] = "serve", off[] = TEST_AXES_OFF,
	     placement[] = "--placement=tp4_pro", draft[] = "--speculative=2",
	     rows[] = "--rows=8", tables[] = "tables", weights[] = "weights",
	     given[TEST_OPTION_BYTES] = "";
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, serve };
	int count = TEST_OPTION_START;
	char *const raw_words[] = { off, placement, draft, rows };
	for (size_t index = 0; index < sizeof(raw_words) / sizeof(raw_words[0]);
	     index++)
		arguments[count++] = raw_words[index];
	if (test->given != NULL) {
		(void)snprintf(given, sizeof(given), "%s", test->given);
		arguments[count++] = given;
	}
	arguments[count++] = tables;
	arguments[count++] = weights;
	struct axis_arguments_record expanded = { 0 };
	bool passed = axis_expand(count, arguments, &expanded);
	const bool parsed = passed &&
			    parse_arguments(expanded.count, expanded.values);
	passed = passed && parsed == test->parse_expected &&
		 (!parsed || (execution_placement == EXECUTION_TP4_PRO &&
			      pipeline_steps == test->depth &&
			      pipeline_steps_defaulted == test->defaulted &&
			      (test->given != NULL || session_slots_max)));
	axis_arguments_free(&expanded);
	return passed;
}

static bool raw_tp_depth_cases(void)
{
	static const struct raw_tp_depth_test tests[] = {
		{ NULL, true, true, TP_MULTI_PIPELINE_STEPS },
		{ "--pipeline_steps=1", false, false, 1u },
		{ "--session_slots=1", true, false, 1u },
		{ "--pipeline_steps=3", true, false, 3u }
	};
	bool passed = true;
	for (size_t index = 0; index < sizeof(tests) / sizeof(tests[0]);
	     index++) {
		axis_cases_run++;
		const pid_t child = fork();
		if (child == 0)
			_exit(raw_tp_depth_case(&tests[index]) ? 0 : 1);
		int status = 0;
		const bool case_passed =
			child > 0 && waitpid(child, &status, 0) == child &&
			WIFEXITED(status) && WEXITSTATUS(status) == 0;
		if (!case_passed)
			(void)fprintf(stderr, "failed raw TP depth: %s\n",
				      tests[index].given != NULL ?
					      tests[index].given :
					      "(no depth, no slots)");
		passed = case_passed && passed;
	}
	return passed;
}

/* Stage 3 review D1: a serve whose --session_slots=max resolves to one session (TP4_EXL3 at the default 1M)
 * runs what an explicit --session_slots=1 runs. Each split's no-axis serve and its --axes=off serve with the device
 * draft parse at max, take single_session_defaults as the fit does, and give every value these name as the same
 * command with --session_slots=1 does (each parse in a child, through a pipe). */
#define TEST_SESSION_VALUES 24u
static bool session_values_parse(const char *split, bool axes_off,
				 bool slots_one,
				 uint64_t values[TEST_SESSION_VALUES])
{
	int channel[TEST_PIPE_ENDS];
	if (pipe(channel) != 0)
		return false;
	const pid_t child = fork();
	if (child == 0) {
		(void)close(channel[0]);
		char program[] = "anemone", serve[] = "serve",
		     off[] = TEST_AXES_OFF, draft[] = "--speculative=2",
		     rows[] = "--rows=8", one[] = "--session_slots=1",
		     tables[] = "tables", weights[] = "weights",
		     placement[TEST_PLACEMENT_BYTES];
		(void)snprintf(placement, sizeof(placement), "--placement=%s",
			       split);
		char *arguments[TEST_ARGUMENT_CAPACITY] = { program, serve };
		int count = TEST_OPTION_START;
		if (axes_off)
			arguments[count++] = off;
		arguments[count++] = placement;
		if (axes_off) {
			arguments[count++] = draft;
			arguments[count++] = rows;
		}
		if (slots_one)
			arguments[count++] = one;
		arguments[count++] = tables;
		arguments[count++] = weights;
		struct axis_arguments_record expanded = { 0 };
		bool good = axis_expand(count, arguments, &expanded) &&
			    parse_arguments(expanded.count, expanded.values) &&
			    session_slots_max == !slots_one &&
			    session_slots ==
				    (slots_one ? 1u : SESSION_SLOTS_MAX_PARSED);
		if (good && !slots_one) {
			good = single_session_defaults(NULL, NULL) == NULL;
			session_slots = 1u; /* what the fit sets */
		}
		const uint64_t parsed[TEST_SESSION_VALUES] = {
			session_slots,
			pipeline_steps,
			pipeline_steps_defaulted,
			draft_policy.depth_policy,
			draft_policy.base_us,
			draft_policy.tail_us,
			draft_policy.row_us,
			draft_learned_acceptance,
			draft_session_acceptance,
			context_tokens,
			speculative,
			draft_positions,
			draft_block,
			step_rows,
			wide_rows,
			batch_rows[0],
			batch_step_us[0],
			prefill_decode_rows,
			prefill_decode_rounds,
			prefill_decode_waiters,
			prefill_order,
			prefill_first,
			(uint64_t)numerics,
			(uint64_t)cmp_order
		};
		_exit(good && write(channel[1], parsed, sizeof(parsed)) ==
					      (ssize_t)sizeof(parsed) ?
			      0 :
			      1);
	}
	(void)close(channel[1]);
	const bool read_all =
		child > 0 &&
		read(channel[0], values,
		     TEST_SESSION_VALUES * sizeof(values[0])) ==
			(ssize_t)(TEST_SESSION_VALUES * sizeof(values[0]));
	(void)close(channel[0]);
	int status = 0;
	return read_all && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static bool single_session_cases(void)
{
	bool passed = true;
	for (size_t split = 0; split < TEST_SPLITS; split++)
		for (unsigned off = 0; off < TEST_OPTION_ORDERS; off++) {
			uint64_t fitted[TEST_SESSION_VALUES],
				given[TEST_SESSION_VALUES];
			const bool same =
				session_values_parse(split_names[split],
						     off != 0, false, fitted) &&
				session_values_parse(split_names[split],
						     off != 0, true, given) &&
				memcmp(fitted, given, sizeof(fitted)) == 0;
			axis_cases_run++;
			if (!same)
				(void)fprintf(
					stderr,
					"failed single session: %s serve%s at max -> 1 is not --session_slots=1\n",
					split_names[split],
					off != 0 ? " " TEST_AXES_OFF : "");
			passed = same && passed;
		}
	return passed;
}

/* Stage 3 review D4: a default never turns a working serve into a refusal. PP5's depth default takes any device
 * draft (--draft=hybrid: five flights at max); a knob that serves one session (one_session_knob) given without
 * --session_slots takes one session; a given count above one, or max, is refused. Each parse in a child. */
#define TEST_ONE_SESSION_WORDS 40u
#define TEST_LONG_OPTION_BYTES \
	512u /* a word of the reference recipe (its --layout) */
enum one_session_expect {
	TEST_REFUSED,
	TEST_PARSED,
	TEST_ONE_SESSION,
	TEST_HYBRID_FLIGHTS
};
struct one_session_test {
	const char *words[TEST_ONE_SESSION_WORDS];
	enum one_session_expect expect;
};

static bool one_session_case(const struct one_session_test *test);

static bool isolated_serve_case(const struct one_session_test *test)
{
	axis_cases_run++;
	const pid_t child = fork();
	if (child == 0)
		_exit(one_session_case(test) ? 0 : 1);
	int status = 0;
	const bool passed = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) && WEXITSTATUS(status) == 0;
	if (!passed)
		(void)fprintf(stderr, "failed serve case: %s %s\n",
			      test->words[0],
			      test->words[1] != NULL ? test->words[1] : "");
	return passed;
}

static bool one_session_case(const struct one_session_test *test)
{
	char program[] = "anemone", serve[] = "serve", tables[] = "tables",
	     weights[] = "weights";
	char words[TEST_ONE_SESSION_WORDS][TEST_LONG_OPTION_BYTES];
	char *arguments[TEST_ARGUMENT_CAPACITY] = { program, serve };
	int count = TEST_OPTION_START;
	for (size_t index = 0;
	     index < TEST_ONE_SESSION_WORDS && test->words[index] != NULL;
	     index++) {
		(void)snprintf(words[index], sizeof(words[index]), "%s",
			       test->words[index]);
		arguments[count++] = words[index];
	}
	arguments[count++] = tables;
	arguments[count++] = weights;
	struct axis_arguments_record expanded = { 0 };
	const bool parsed = axis_expand(count, arguments, &expanded) &&
			    parse_arguments(expanded.count, expanded.values);
	axis_arguments_free(&expanded);
	switch (test->expect) {
	case TEST_REFUSED:
		return !parsed;
	case TEST_PARSED:
		return parsed;
	case TEST_ONE_SESSION:
		return parsed && session_slots == 1u && !session_slots_max &&
		       (official_reference != 0 || prefill_prefix_stable != 0 ||
			prefix_cold_plan_is_qualified());
	case TEST_HYBRID_FLIGHTS:
		return parsed && speculative == SPECULATIVE_HYBRID &&
		       session_slots_max &&
		       pipeline_steps == ORIGINAL_PP5_MULTI_PIPELINE_STEPS;
	}
	return false;
}

static bool one_session_cases(const struct axis_choice *reference)
{
	struct one_session_test tests[] = {
		{ { "--draft=hybrid" }, TEST_HYBRID_FLIGHTS },
		{ { "--placement=tp4_exl3", "--prefix_reuse_exact=1" },
		  TEST_ONE_SESSION },
		{ { "--placement=tp4_exl3", "--prefix_reuse_exact=1",
		    "--session_slots=4" },
		  TEST_REFUSED },
		{ { "--placement=tp4_exl3", "--prefix_reuse_exact=1",
		    "--session_slots=max" },
		  TEST_REFUSED },
		{ { "--placement=tp4_exl3", "--session_fork=1" },
		  TEST_ONE_SESSION },
		{ { "--placement=tp4_exl3", "--session_fork=1",
		    "--session_slots=2" },
		  TEST_REFUSED },
		/* the PP5 reference recipe's serve words, without its --session_slots=1 (below) */
		{ { TEST_AXES_OFF }, TEST_ONE_SESSION },
		{ { TEST_AXES_OFF, "--session_slots=2" }, TEST_REFUSED }
	};
	const size_t test_count = sizeof(tests) / sizeof(tests[0]);
	const size_t reference_first = test_count - TEST_OPTION_ORDERS;
	const size_t bytes = reference == NULL ?
				     0 :
				     strlen(reference->settings) +
					     strlen(reference->serve_settings) +
					     sizeof("\n");
	char *const storage = bytes != 0 ? malloc(bytes) : NULL;
	bool passed = storage != NULL;
	if (passed)
		(void)snprintf(storage, bytes, "%s\n%s", reference->settings,
			       reference->serve_settings);
	char *save = NULL;
	for (char *setting = storage == NULL ? NULL :
					       strtok_r(storage, "\n", &save);
	     setting != NULL; setting = strtok_r(NULL, "\n", &save)) {
		if (strncmp(setting, "--session_slots=",
			    strlen("--session_slots=")) == 0)
			continue;
		for (size_t index = reference_first; index < test_count;
		     index++) {
			size_t used = 0;
			while (used < TEST_ONE_SESSION_WORDS &&
			       tests[index].words[used] != NULL)
				used++;
			passed = used + 1u < TEST_ONE_SESSION_WORDS && passed;
			if (used + 1u < TEST_ONE_SESSION_WORDS)
				tests[index].words[used] = setting;
		}
	}
	for (size_t index = 0; index < test_count && passed; index++)
		passed = isolated_serve_case(&tests[index]) && passed;
	free(storage);
	return passed;
}

/* Stage 3 review D2: a context off the model's granule is refused at parse, by name, on both commands; a
 * multiple of it parses. */
static bool context_granule_cases(void)
{
	static const struct one_session_test serves[] = {
		{ { "--context_tokens=131000" }, TEST_REFUSED },
		{ { "--context_tokens=131072" }, TEST_PARSED }
	};
	const char *const off_granule[] = { "--context_tokens=131000" };
	const char *const on_granule[] = { "--context_tokens=131072" };
	bool passed = isolated_raw_case(off_granule, 1u, true, false, true);
	passed = isolated_raw_case(on_granule, 1u, true, true, true) && passed;
	for (size_t index = 0; index < sizeof(serves) / sizeof(serves[0]);
	     index++)
		passed = isolated_serve_case(&serves[index]) && passed;
	return passed;
}

/* Stage 3 review D2 (F3): a BATCH serve's fit tries only its eight sessions (no bisection over counts BATCH
 * refuses), and its refusal gives the longest context of eight but never the --session_slots=max advice: TP4_EXL3's
 * measured BATCH serve (multiple, 64K, eight slots) on cards that hold everything fits eight; on cards of a MiB it
 * fits none, for memory. */
#define TEST_REFUSAL_BYTES 1024u
static bool batch_fit_case(void)
{
	char program[] = "anemone", serve[] = "serve",
	     placement[] = "--placement=tp4_exl3",
	     multiple[] = "--concurrency=multiple", slots[] = TEST_BATCH_SLOTS,
	     context[] = "--context_tokens=65536", tables[] = "tables",
	     weights[] = "weights";
	char *arguments[] = { program, serve,	placement, multiple,
			      slots,   context, tables,	   weights };
	static const struct tensor_fit_split split = {
		"--placement=tp4_exl3", TEST_MODEL_V41_CHANGE_EXL3, TP4_RANKS,
		false
	};
	struct axis_arguments_record expanded = { 0 };
	bool passed =
		axis_expand((int)(sizeof(arguments) / sizeof(arguments[0])),
			    arguments, &expanded) &&
		parse_arguments(expanded.count, expanded.values) &&
		batch_rows[0] != 0u && batch_serving() &&
		session_slots == TEST_BATCH_SESSIONS;
	card_count = split.cards;
	const struct test_model_v41_change change = { .kind = split.pack_kind };
	passed = passed &&
		 test_model_v41_draft_pack_open(change, &pack) == STATUS_OK;
	uint32_t *const card_of_unit =
		passed ? calloc(pack_header_of(pack)->unit_count,
				sizeof(*card_of_unit)) :
			 NULL;
	const struct slot_fit fit = { NULL,
				      (uint64_t)card_memory_kept_free_mib *
					      BYTES_PER_MIB,
				      card_of_unit };
	uint64_t roomy[BACKEND_MAXIMUM_CARDS], tiny[BACKEND_MAXIMUM_CARDS];
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
		roomy[card] = TEST_TP_FIT_ROOMY_BYTES;
		tiny[card] = BYTES_PER_MIB;
	}
	const struct model_limits limits = options_limits();
	result_status refusal = STATUS_INVALID_ARGUMENT;
	passed = passed && card_of_unit != NULL;
	if (passed)
		tensor_fit_room(&split, roomy);
	const uint32_t roomy_most =
		passed ? slots_most(&fit, &limits, &refusal) : 0u;
	passed = passed && roomy_most == TEST_BATCH_SESSIONS &&
		 refusal == STATUS_OK;
	if (passed)
		tensor_fit_room(&split, tiny);
	passed = passed && slots_most(&fit, &limits, &refusal) == 0u &&
		 refusal == STATUS_OK;
	FILE *const capture = tmpfile();
	(void)fflush(stderr);
	const int saved = dup(STDERR_FILENO);
	passed = passed && capture != NULL && saved >= 0 &&
		 dup2(fileno(capture), STDERR_FILENO) == STDERR_FILENO;
	if (passed)
		(void)sessions_refused(&fit, &limits, tiny);
	(void)fflush(stderr);
	passed = saved >= 0 && dup2(saved, STDERR_FILENO) == STDERR_FILENO &&
		 passed;
	char text[TEST_REFUSAL_BYTES] = { 0 };
	if (capture != NULL) {
		passed = fseek(capture, 0L, SEEK_SET) == 0 && passed;
		clearerr(capture);
		const size_t bytes =
			fread(text, 1u, sizeof(text) - 1u, capture);
		text[bytes] = '\0';
		(void)fclose(capture);
	}
	if (saved >= 0)
		(void)close(saved);
	passed = passed &&
		 strstr(text,
			"session slots: 8 sessions of 65536 tokens do not "
			"fit the cards; nothing is shrunk") != NULL &&
		 strstr(text, SESSION_SLOTS_MAX_OPTION) == NULL;
	if (!passed)
		(void)fprintf(stderr, "BATCH fit: most %" PRIu32 " (%s); %s",
			      roomy_most, status_text(refusal), text);
	free(card_of_unit);
	if (pack != NULL)
		pack_close(pack);
	pack = NULL;
	axis_arguments_free(&expanded);
	return passed;
}

/* The fit line (sessions_fit_report) as a serve prints it on stdout, against expected. */
#define TEST_FIT_LINE_BYTES 512u
static bool fit_line_matches(uint32_t most, const struct model_limits *limits,
			     const struct axis_choice *slots_unmet,
			     const char *expected)
{
	FILE *const capture = tmpfile();
	(void)fflush(stdout);
	const int saved = dup(STDOUT_FILENO);
	bool passed = capture != NULL && saved >= 0 &&
		      dup2(fileno(capture), STDOUT_FILENO) == STDOUT_FILENO;
	if (passed)
		sessions_fit_report(most, limits, slots_unmet);
	(void)fflush(stdout);
	passed = saved >= 0 && dup2(saved, STDOUT_FILENO) == STDOUT_FILENO &&
		 passed;
	char text[TEST_FIT_LINE_BYTES] = { 0 };
	if (capture != NULL) {
		passed = fseek(capture, 0L, SEEK_SET) == 0 && passed;
		clearerr(capture);
		const size_t bytes =
			fread(text, 1u, sizeof(text) - 1u, capture);
		text[bytes] = '\0';
		(void)fclose(capture);
	}
	if (saved >= 0)
		(void)close(saved);
	if (passed && strcmp(text, expected) != 0)
		(void)fprintf(stderr, "fit line: %s", text);
	return passed && strcmp(text, expected) == 0;
}

/* TP4_EXL3 concurrency=multiple on serve at the 1M default runs its concurrency=balanced row (thin steps,
 * no BATCH knob) at the most sessions that fit, here cards with room for one session's arenas, and the fit line names
 * the row, the count, the context and BATCH's 8. An explicit count's line names the count it serves. */
static bool batch_slots_note_case(void)
{
	char program[] = "anemone", serve[] = "serve",
	     placement[] = "--placement=tp4_exl3",
	     multiple[] = "--concurrency=multiple", tables[] = "tables",
	     weights[] = "weights";
	char *arguments[] = { program,	serve,	placement,
			      multiple, tables, weights };
	static const struct tensor_fit_split split = {
		"--placement=tp4_exl3", TEST_MODEL_V41_CHANGE_EXL3, TP4_RANKS,
		false
	};
	size_t rows = 0;
	const struct axis_choice *const choices = axis_choices_entry(&rows);
	const struct axis_choice *const batch =
		find_choice(choices, rows, "balanced", "balanced", "multiple",
			    "balanced", "tp4_exl3");
	const struct axis_choice *const thin =
		find_choice(choices, rows, "balanced", "balanced", "balanced",
			    "balanced", "tp4_exl3");
	struct axis_arguments_record expanded = { 0 };
	bool passed =
		batch != NULL && thin != NULL &&
		batch->serve_slots_exactly == TEST_BATCH_SESSIONS &&
		axis_expand((int)(sizeof(arguments) / sizeof(arguments[0])),
			    arguments, &expanded) &&
		expanded.choice == thin && expanded.slots_unmet == batch &&
		parse_arguments(expanded.count, expanded.values) &&
		batch_rows[0] == 0u && session_slots_max &&
		context_tokens == TEST_OWNER_CONTEXT_TOKENS;
	card_count = split.cards;
	const struct test_model_v41_change change = { .kind = split.pack_kind };
	passed = passed &&
		 test_model_v41_draft_pack_open(change, &pack) == STATUS_OK;
	uint32_t *const card_of_unit =
		passed ? calloc(pack_header_of(pack)->unit_count,
				sizeof(*card_of_unit)) :
			 NULL;
	const uint64_t kept =
		(uint64_t)card_memory_kept_free_mib * BYTES_PER_MIB;
	uint64_t roomy[BACKEND_MAXIMUM_CARDS], one[BACKEND_MAXIMUM_CARDS],
		room[BACKEND_MAXIMUM_CARDS] = { 0 },
		arenas[BACKEND_MAXIMUM_CARDS];
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++)
		roomy[card] = TEST_TP_FIT_ROOMY_BYTES;
	uint32_t ended = 0;
	session_slots_max = false;
	passed = passed && card_of_unit != NULL &&
		 tensor_fit(&split, roomy, 1u, card_of_unit, &ended, one) ==
			 STATUS_OK;
	for (uint32_t card = 0; passed && card < split.cards; card++)
		room[card] =
			one[card] + BACKEND_CUDA_MEMORY_RESERVED_BYTES + kept;
	session_slots_max = true;
	passed = passed &&
		 tensor_fit(&split, room, SERVING_SLOTS_MOST, card_of_unit,
			    &ended, arenas) == STATUS_OK &&
		 ended == 1u;
	struct model_limits limits = options_limits();
	limits.session_capacity = ended;
	passed =
		passed &&
		fit_line_matches(
			ended, &limits, expanded.slots_unmet,
			"concurrency=multiple: 1 session(s) fit at 1048576 tokens of context on this split; a "
			"shorter --context_tokens gives more; BATCH needs exactly 8\n");
	/* an explicit count (the BATCH row as measured: 64K, 8 slots): the line names what it serves */
	session_slots_max = false;
	limits.context_token_limit = TEST_BATCH_CONTEXT_TOKENS;
	limits.session_capacity = TEST_BATCH_SESSIONS;
	passed =
		passed &&
		fit_line_matches(
			TEST_FIT_LINE_MOST, &limits, NULL,
			"11 session(s) fit at 65536 tokens of context on this split; serving 8 (--session_slots); "
			"a shorter --context_tokens gives more\n");
	free(card_of_unit);
	pack_close(pack);
	pack = NULL;
	axis_arguments_free(&expanded);
	return passed;
}

static bool batch_slots_note_cases(void)
{
	axis_cases_run++;
	const pid_t child = fork();
	if (child == 0)
		_exit(batch_slots_note_case() ? 0 : 1);
	int status = 0;
	const bool passed = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) && WEXITSTATUS(status) == 0;
	if (!passed)
		(void)fputs("failed BATCH slot note: tp4_exl3 serve\n", stderr);
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
		      compatible_fast_plans(choices, count) &&
		      count == TEST_SPLITS * categories * exactnesses *
				       concurrencies * focuses;
	if (!passed)
		(void)fputs("failed table shape, fast plans or prompt ids\n",
			    stderr);
	for (size_t index = 0; index < count; index++) {
		const struct axis_choice *choice = &choices[index];
		const bool available = choice->hold[0] == '\0';
		/* A selector is never a substitute for a missing numerical/runtime
		 * dependency: exactly these rows are held. */
		if (available == row_held(choice)) {
			(void)fprintf(stderr, "failed hold: %s/%s/%s/%s %s\n",
				      choice->value[TEST_CATEGORY],
				      choice->value[TEST_EXACTNESS],
				      choice->value[TEST_CONCURRENCY],
				      choice->value[TEST_FOCUS],
				      choice->gpu_split);
			passed = false;
		}
		/* PP5 rows run as the default split and with explicit placement. */
		const bool default_split = strcmp(choice->gpu_split, "pp5") ==
					   0;
		for (unsigned explicit_split = default_split ? 0u : 1u;
		     explicit_split <= 1u; explicit_split++) {
			struct axis_test_case test = {
				.choice = choice,
				.command = "generate",
				.placement = explicit_split != 0 ?
						     choice->gpu_split :
						     NULL,
				.expand_expected = available,
				.parse_expected = available,
				.placement_expected = available,
				.batch_expected = available
			};
			passed = isolated_tuple_case(&test) && passed;
			test.command = "serve";
			test.extras_first = true;
			passed = isolated_tuple_case(&test) && passed;
			/* Stage 3 review D7: a row whose set applies on serve at one count (TP4_EXL3 multiple's
			 * BATCH at 8) runs it there as measured, with its whole geometry checked */
			if (available && choice->serve_slots_exactly != 0u) {
				struct axis_test_case batch = test;
				batch.extra[0] = TEST_BATCH_SLOTS;
				batch.extra[1] = TEST_BATCH_CONTEXT;
				batch.destination = TEST_BATCH_GEOMETRY;
				passed = isolated_tuple_case(&batch) && passed;
			}
			if (!available || !original_recipe_row(choice) ||
			    !row_value(choice, TEST_FOCUS, "decode"))
				continue;
			for (unsigned order = 0; order < TEST_OPTION_ORDERS;
			     order++) {
				for (unsigned command = 0;
				     command < TEST_OPTION_ORDERS; command++) {
					test.command = command == 0 ?
							       "generate" :
							       "serve";
					test.extras_first = order != 0;
					test.extra[0] =
						"--draft_session_acceptance=6";
					test.destination =
						TEST_SESSION_OVERRIDE;
					passed = isolated_tuple_case(&test) &&
						 passed;
				}
			}
		}
	}
	if (!rows_unique(choices, count)) {
		(void)fputs("failed row uniqueness\n", stderr);
		passed = false;
	}
	const struct axis_choice *single = find_choice(choices, count, "code",
						       "balanced", "single",
						       "decode", "tp4_exl3");
	const struct axis_choice *multiple = find_choice(choices, count, "code",
							 "balanced", "multiple",
							 "decode", "tp4_exl3");
	const struct axis_choice *adaptive =
		find_choice(choices, count, "structured", "balanced",
			    "multiple", "prefill", "tp4_exl3");
	if (single == NULL || multiple == NULL || adaptive == NULL) {
		passed = false;
	} else {
		const struct axis_test_case overrides[] = {
			{ adaptive,
			  "serve",
			  { "--batch_step_us=0", TEST_BATCH_SLOTS },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_BATCH_MENU_OFF,
			  "tp4_exl3" },
			{ adaptive,
			  "generate",
			  { "--batch_step_us=0" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_BATCH_MENU_OFF,
			  "tp4_exl3" },
			{ adaptive,
			  "serve",
			  { "--batch_step_us=19500,0", TEST_BATCH_SLOTS },
			  false,
			  true,
			  false,
			  true,
			  true,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ single,
			  "serve",
			  { "--wide_rows=1024" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_WIDE_ROWS,
			  "tp4_exl3" },
			{ single,
			  "generate",
			  { "--draft_policy=1" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_DRAFT_POLICY,
			  "tp4_exl3" },
			{ single,
			  "generate",
			  { "--speculative_depth=6", "--draft_block=7" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_DRAFT_DEPTH,
			  "tp4_exl3" },
			{ single,
			  "generate",
			  { "--draft=lookup" },
			  false,
			  true,
			  true,
			  false,
			  true,
			  TEST_SPECULATIVE_LOOKUP,
			  "tp4_exl3" },
			{ single,
			  "generate",
			  { "--speculative=0" },
			  false,
			  true,
			  true,
			  false,
			  true,
			  TEST_SPECULATIVE_OFF,
			  "tp4_exl3" },
			{ single,
			  "serve",
			  { "--numerics=exact" },
			  false,
			  true,
			  true,
			  false,
			  true,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ single,
			  "serve",
			  { "--pro_emulation=1" },
			  false,
			  true,
			  true,
			  false,
			  true,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ single,
			  "serve",
			  { "--wide_rows=1024", "--wide_rows=2048" },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ multiple,
			  "serve",
			  { "--pipeline_steps=2", TEST_BATCH_SLOTS },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ multiple,
			  "serve",
			  { "--batch_draft_rows=0", TEST_BATCH_SLOTS },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			/* BATCH needs exactly 8 sessions: at one slot the serve runs the concurrency=balanced row */
			{ multiple,
			  "serve",
			  { "--session_slots=1" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ multiple,
			  "serve",
			  { "--session_slots=max" },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			/* as measured: 64K context and exactly 8 slots run BATCH */
			{ multiple,
			  "serve",
			  { "--context_tokens=65536", TEST_BATCH_SLOTS },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_BATCH_AT_EIGHT,
			  "tp4_exl3" },
			{ multiple,
			  "generate",
			  { "--batch_verifiers=8" },
			  false,
			  true,
			  true,
			  true,
			  false,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rows=0", TEST_BATCH_SLOTS },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_DECODE_ROWS,
			  "tp4_exl3" },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rounds=4", TEST_BATCH_SLOTS },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_DECODE_ROUNDS,
			  "tp4_exl3" },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rows=128", TEST_BATCH_SLOTS },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rows=1024",
			    "--chunks_in_flight=2", TEST_BATCH_SLOTS },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ multiple,
			  "serve",
			  { "--prefill_decode_rows=1024",
			    "--prefill_fast_plan=1", TEST_BATCH_SLOTS },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  "tp4_exl3" },
			{ single,
			  "load",
			  { NULL },
			  false,
			  false,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  "tp4_exl3" }
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
			    "decode", "tp4_pro");
	const struct axis_choice *original_multiple_choice =
		find_choice(choices, count, "code", "balanced", "multiple",
			    "decode", "pp5");
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
			  TEST_LAYOUT,
			  NULL },
			{ original_multiple_choice,
			  "generate",
			  { "--layout=" TEST_LAYOUT_OVERRIDE },
			  false,
			  true,
			  true,
			  true,
			  true,
			  TEST_LAYOUT,
			  NULL },
			{ original_multiple_choice,
			  "serve",
			  { "--placement=tp4_pro" },
			  false,
			  false,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  NULL },
			{ original_single_choice,
			  "serve",
			  { "--layout=" TEST_LAYOUT_OVERRIDE },
			  false,
			  true,
			  false,
			  false,
			  false,
			  TEST_UNCHANGED,
			  "tp4_pro" }
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
	const struct axis_choice *reference = find_choice(
		choices, count, "code", "exact", "single", "decode", "pp5");
	passed = reference_raw_recipe_cases(reference) && passed;
	passed = raw_default_cases() && passed;
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
	passed = isolated_raw_case(fast_partial, 1u, true, true, false) &&
		 passed;
	const char *const common_fast[] = { "--placement=tp4_exl3",
					    "--category=structured",
					    "--exactness=fast",
					    "--concurrency=single" };
	passed = isolated_raw_case(common_fast,
				   sizeof(common_fast) / sizeof(common_fast[0]),
				   true, true, false) &&
		 passed;
	/* Stage 3 review D5: a cap stops at the knob's parser least: a raw --rows=1 --speculative=0 caps the
	 * base's inert --draft_positions to 1, not 0; code's --draft_block 7 at --rows=4 to 5, not 3 */
	const char *const raw_serial[] = { "--rows=1", "--speculative=0" };
	passed = isolated_raw_case(raw_serial,
				   sizeof(raw_serial) / sizeof(raw_serial[0]),
				   true, true, true) &&
		 passed;
	const char *const code_four_rows[] = { "--category=code", "--rows=4" };
	passed = isolated_raw_case(code_four_rows,
				   sizeof(code_four_rows) /
					   sizeof(code_four_rows[0]),
				   true, true, false) &&
		 passed;
	const char *const raw_head[] = { "--original_head_offload=2" };
	passed = isolated_raw_case(raw_head, 1u, true, true, true) && passed;
	const struct axis_choice *banked =
		find_choice(choices, count, "balanced", "balanced", "single",
			    "decode", "tp4_pro");
	if (banked == NULL) {
		passed = false;
	} else {
		for (unsigned order = 0; order < TEST_OPTION_ORDERS; order++) {
			struct axis_test_case fallback = {
				.choice = banked,
				.command = "serve",
				.extra = { "--original_head_offload=0" },
				.extras_first = order != 0,
				.expand_expected = true,
				.parse_expected = true,
				.placement_expected = true,
				.batch_expected = true,
				.destination = TEST_HEAD_OFF,
				.placement = "tp4_pro"
			};
			passed = isolated_tuple_case(&fallback) && passed;
			fallback.extra[0] = "--metrics=1";
			fallback.parse_expected = true;
			fallback.destination = TEST_TP_METRICS_ON;
			passed = isolated_tuple_case(&fallback) && passed;
		}
	}
	const char *const banked_partial[] = { "--category=balanced",
					       "--exactness=balanced",
					       "--focus=decode" };
	passed = isolated_raw_case(banked_partial,
				   sizeof(banked_partial) /
					   sizeof(banked_partial[0]),
				   true, true, false) &&
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
	const char *const exl3_code_partial[] = { "--category=code",
						  "--placement=tp4_exl3" };
	passed = isolated_raw_case(exl3_code_partial,
				   sizeof(exl3_code_partial) /
					   sizeof(exl3_code_partial[0]),
				   true, true, false) &&
		 passed;
	/* Refusals: a split without a table and a value its table holds. */
	const char *const refused_partials[][TEST_EXTRA_CAPACITY] = {
		{ "--placement=tp5", "--category=code" },
		{ "--placement=tp2_pp2_pro", "--focus=decode" },
		{ "--placement=tp2_single_single_pro", "--category=prose" },
		{ "--placement=tp4_pro", "--concurrency=multiple" },
		{ "--placement=tp4_pro", "--exactness=exact" }
	};
	for (size_t index = 0;
	     index < sizeof(refused_partials) / sizeof(refused_partials[0]);
	     index++)
		passed = isolated_raw_case(refused_partials[index],
					   TEST_REFERENCE_SELECTOR_OPTIONS,
					   false, false, false) &&
			 passed;
	/* Each partial tuple lands on the row with balanced on its unset axes. */
	const struct {
		const char *options[TEST_EXTRA_CAPACITY];
		const char *value[TEST_AXES], *split;
	} partial_rows[] = {
		{ { "--exactness=exact" },
		  { "balanced", "exact", "single", "balanced" },
		  "pp5" },
		{ { "--exactness=fast" },
		  { "balanced", "fast", "balanced", "balanced" },
		  "pp5" },
		{ { "--category=balanced", "--focus=decode" },
		  { "balanced", "balanced", "balanced", "decode" },
		  "pp5" },
		{ { "--placement=tp4_pro", "--category=code" },
		  { "code", "balanced", "balanced", "balanced" },
		  "tp4_pro" },
		{ { "--placement=tp4_exl3", "--category=code" },
		  { "code", "balanced", "balanced", "balanced" },
		  "tp4_exl3" },
		{ { "--placement=tp4_exl3", "--category=prose",
		    "--concurrency=single" },
		  { "prose", "balanced", "single", "balanced" },
		  "tp4_exl3" },
		{ { "--placement=tp4_exl3", "--concurrency=multiple" },
		  { "balanced", "balanced", "multiple", "balanced" },
		  "tp4_exl3" },
		/* layer drafting on concurrency=balanced, which repeats THIN */
		{ { "--placement=tp4_exl3", "--focus=decode" },
		  { "balanced", "balanced", "balanced", "decode" },
		  "tp4_exl3" },
		/* the table is keyed on the family too: --weights=exl3 alone takes TP4_EXL3's rows, each axis flag */
		{ { "--weights=exl3", "--category=code" },
		  { "code", "balanced", "balanced", "balanced" },
		  "tp4_exl3" },
		{ { "--weights=exl3", "--exactness=fast" },
		  { "balanced", "fast", "balanced", "balanced" },
		  "tp4_exl3" },
		{ { "--weights=exl3", "--concurrency=multiple" },
		  { "balanced", "balanced", "multiple", "balanced" },
		  "tp4_exl3" },
		{ { "--weights=exl3", "--focus=decode" },
		  { "balanced", "balanced", "balanced", "decode" },
		  "tp4_exl3" },
		{ { "--weights=original", "--focus=decode" },
		  { "balanced", "balanced", "balanced", "decode" },
		  "pp5" }
	};
	for (size_t index = 0;
	     index < sizeof(partial_rows) / sizeof(partial_rows[0]); index++) {
		const struct axis_choice *const row =
			find_choice(choices, count,
				    partial_rows[index].value[TEST_CATEGORY],
				    partial_rows[index].value[TEST_EXACTNESS],
				    partial_rows[index].value[TEST_CONCURRENCY],
				    partial_rows[index].value[TEST_FOCUS],
				    partial_rows[index].split);
		size_t options = 0;
		while (options < TEST_EXTRA_CAPACITY &&
		       partial_rows[index].options[options] != NULL)
			options++;
		passed = row != NULL &&
			 isolated_partial_case(partial_rows[index].options,
					       options, row, NULL) &&
			 passed;
	}
	/* A family the split does not run refuses, naming both (never the other family's table). */
	static const struct {
		const char *options[TEST_EXTRA_CAPACITY], *refusal;
	} family_refusals[] = {
		{ { "--weights=original", "--placement=tp4_exl3",
		    "--category=code" },
		  "--weights=original conflicts with GPU split tp4_exl3, whose table runs exl3 weights" },
		{ { "--weights=exl3", "--placement=pp5", "--category=code" },
		  "--weights=exl3 conflicts with GPU split pp5, whose table runs original weights" },
		/* a value that names no family is refused as such, not as a conflict */
		{ { "--weights=EXL3", "--focus=decode", "--category=code" },
		  "axes refused: --weights=EXL3 is not original or exl3" },
		{ { "--weights=", "--focus=decode", "--category=code" },
		  "axes refused: --weights= is not original or exl3" }
	};
	for (size_t index = 0;
	     index < sizeof(family_refusals) / sizeof(family_refusals[0]);
	     index++)
		passed =
			isolated_partial_case(family_refusals[index].options,
					      TEST_EXTRA_CAPACITY, NULL,
					      family_refusals[index].refusal) &&
			passed;
	static const char *const untabled[] = { "tp5", "tp2_pp2_pro",
						"tp2_single_single_pro" };
	for (size_t index = 0; index < sizeof(untabled) / sizeof(untabled[0]);
	     index++) {
		char placement[TEST_PLACEMENT_BYTES], reason[TEST_LOG_CAPACITY];
		const int placement_bytes =
			snprintf(placement, sizeof(placement), "--placement=%s",
				 untabled[index]);
		const int reason_bytes = snprintf(
			reason, sizeof(reason),
			"axes refused: GPU split %s has no axis table; the tables are %s.",
			untabled[index], TEST_SPLIT_NAMES);
		const char *const options[] = { placement, "--category=code" };
		passed = placement_bytes > 0 &&
			 (size_t)placement_bytes < sizeof(placement) &&
			 reason_bytes > 0 &&
			 (size_t)reason_bytes < sizeof(reason) &&
			 isolated_partial_case(
				 options, sizeof(options) / sizeof(options[0]),
				 NULL, reason) &&
			 passed;
	}
	for (enum test_selection_boundary boundary = TEST_COMMON_PLAN;
	     boundary < TEST_SELECTION_BOUNDARIES; boundary++)
		passed = isolated_boundary_case(boundary) && passed;

	passed = engram_parser_cases() && passed;
	passed = loader_readers_cases() && passed;
	passed = engram_profile_cases() && passed;
	passed = engram_fallback_cases() && passed;
	passed = engram_steps_start_cases() && passed;
	passed = sampling_parser_cases() && passed;
	passed = isolated_prompt_chunks_cases() && passed;
	passed = raw_banked_qualification(false) && passed;
	passed = raw_banked_qualification(true) && passed;
	passed = scoped_help_case(choices, count) && passed;
	passed = clamp_cases() && passed;
	passed = tensor_fit_cases() && passed;
	passed = batch_slots_note_cases() && passed;
	passed = raw_tp_depth_cases() && passed;
	passed = single_session_cases() && passed;
	passed = one_session_cases(find_choice(choices, count, "code", "exact",
					       "single", "decode", "pp5")) &&
		 passed;
	passed = context_granule_cases() && passed;
	axis_cases_run++;
	const pid_t batch_fit_child = fork();
	if (batch_fit_child == 0)
		_exit(batch_fit_case() ? 0 : 1);
	int batch_fit_status = 0;
	const bool batch_fit_passed =
		batch_fit_child > 0 &&
		waitpid(batch_fit_child, &batch_fit_status, 0) ==
			batch_fit_child &&
		WIFEXITED(batch_fit_status) &&
		WEXITSTATUS(batch_fit_status) == 0;
	if (!batch_fit_passed)
		(void)fputs("failed BATCH fit: tp4_exl3 serve\n", stderr);
	passed = batch_fit_passed && passed;
	passed = exl3_exact_cases() && passed;
	(void)printf(
		"startup axes: %s (%zu table rows, %u cases, real parser/startup/verifier gates, no cards)\n",
		passed ? "PASS" : "FAIL", count, axis_cases_run);
	return passed ? 0 : 1;
}
