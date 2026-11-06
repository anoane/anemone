/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "startup_axes.h"
#include "cli_axes_data.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { AXIS_OPTION_START = 2 };
enum axis_index { AXIS_CATEGORY, AXIS_EXACTNESS, AXIS_CONCURRENCY, AXIS_FOCUS };

_Static_assert(sizeof(axis_domains) / sizeof(axis_domains[0]) == AXIS_COUNT,
	       "axis registry and parser must have the same dimensions");

const struct axis_choice *axis_choices_entry(size_t *count_out)
{
	*count_out = sizeof(axis_choices) / sizeof(axis_choices[0]);
	return axis_choices;
}

/* Existing low-level aliases retain their CLI destination and conflict rules. */
static size_t option_key(const char *option, const char **key_out)
{
	const char *end = strchr(option, '=');
	size_t bytes = end != NULL ? (size_t)(end - option) : strlen(option);
	*key_out = option;
	if ((bytes == strlen("--draft") &&
	     strncmp(option, "--draft", bytes) == 0) ||
	    (bytes == strlen("--speculative") &&
	     strncmp(option, "--speculative", bytes) == 0)) {
		*key_out = "--speculative";
		bytes = strlen(*key_out);
	} else if (bytes == strlen("--speculative_depth") &&
		   strncmp(option, "--speculative_depth", bytes) == 0) {
		*key_out = "--draft_positions";
		bytes = strlen(*key_out);
	}
	return bytes;
}

static bool explicit_key(const char *option, int count, char *const *values)
{
	const char *key;
	const size_t bytes = option_key(option, &key);
	for (int index = AXIS_OPTION_START;
	     index < count && strncmp(values[index], "--", strlen("--")) == 0;
	     index++) {
		const char *candidate;
		if (option_key(values[index], &candidate) == bytes &&
		    strncmp(key, candidate, bytes) == 0)
			return true;
	}
	return false;
}

static int axis_option(const char *option)
{
	for (unsigned axis = 0; axis < AXIS_COUNT; axis++)
		if (strncmp(option, axis_domains[axis].prefix,
			    strlen(axis_domains[axis].prefix)) == 0)
			return (int)axis;
	return -1;
}

static void selection_log(const struct axis_arguments_record *arguments,
			  FILE *stream)
{
	(void)fputs("axes", stream);
	for (unsigned axis = 0; axis < AXIS_COUNT; axis++)
		(void)fprintf(stream, " %s=%s", axis_domains[axis].name,
			      arguments->selected[axis] != NULL ?
				      arguments->selected[axis] :
				      "unset");
	(void)fprintf(stream, " weight_family=%s\n", arguments->weight_family);
}

static const char *command_settings(const struct axis_choice *choice,
				    bool serving)
{
	return !serving && choice->generate_settings[0] != '\0' ?
		       choice->generate_settings :
		       choice->settings;
}

/* A partial tuple may apply a plan only when every compatible full tuple has
 * the same plan. Missing axes never acquire an implicit preset default. */
static bool same_plan(const struct axis_choice *left,
		      const struct axis_choice *right, bool serving)
{
	return strcmp(command_settings(left, serving),
		      command_settings(right, serving)) == 0 &&
	       (!serving ||
		strcmp(left->serve_settings, right->serve_settings) == 0) &&
	       strcmp(left->error_class, right->error_class) == 0 &&
	       strcmp(left->hold, right->hold) == 0;
}

/* Exact has one qualified execution scope. An omitted workload label selects
 * that text/C1 scope; explicit unsupported labels still retain their refusal. */
static bool
original_reference_scope(const struct axis_arguments_record *arguments,
			 const struct axis_choice *choice)
{
	if (arguments->selected[AXIS_EXACTNESS] == NULL ||
	    strcmp(arguments->selected[AXIS_EXACTNESS], "exact") != 0 ||
	    strcmp(arguments->weight_family, "original") != 0)
		return true;
	return (arguments->selected[AXIS_CATEGORY] != NULL ||
		strcmp(choice->value[AXIS_CATEGORY], "code") == 0) &&
	       (arguments->selected[AXIS_CONCURRENCY] != NULL ||
		strcmp(choice->value[AXIS_CONCURRENCY], "single") == 0);
}

static bool reference_setting_conflicts(const char *setting, int count,
					char *const *values)
{
	const char *key;
	const size_t bytes = option_key(setting, &key);
	for (int index = AXIS_OPTION_START;
	     index < count && strncmp(values[index], "--", strlen("--")) == 0;
	     index++) {
		const char *candidate;
		const char *value = strchr(values[index], '=');
		if (option_key(values[index], &candidate) == bytes &&
		    strncmp(key, candidate, bytes) == 0 &&
		    (value == NULL ||
		     strcmp(strchr(setting, '='), value) != 0)) {
			(void)fprintf(
				stderr,
				"axes refused: exact reference requires %s; conflicting override %s\n",
				setting, values[index]);
			return true;
		}
	}
	return false;
}

bool axis_expand(int count, char **values, struct axis_arguments_record *result)
{
	if (result == NULL || count < 1 || values == NULL)
		return false;
	*result = (struct axis_arguments_record){ .count = count,
						  .values = values,
						  .explicit_count = count,
						  .explicit_values = values,
						  .weight_family = "original" };
	for (int index = AXIS_OPTION_START;
	     index < count && strncmp(values[index], "--", strlen("--")) == 0;
	     index++) {
		const int axis = axis_option(values[index]);
		if (axis < 0)
			continue;
		if (result->selected[axis] != NULL) {
			(void)fprintf(stderr, "axes: duplicate --%s\n",
				      axis_domains[axis].name);
			return false;
		}
		const char *value =
			values[index] + strlen(axis_domains[axis].prefix);
		for (size_t entry = 0; entry < axis_domains[axis].count;
		     entry++)
			if (strcmp(value, axis_domains[axis].values[entry]) ==
			    0)
				result->selected[axis] =
					axis_domains[axis].values[entry];
		if (result->selected[axis] == NULL) {
			(void)fprintf(
				stderr,
				"axes: invalid --%s value; see anemone help axes\n",
				axis_domains[axis].name);
			return false;
		}
		result->selected_count++;
	}
	if (result->selected_count == 0)
		return true;
	if (count < AXIS_OPTION_START || (strcmp(values[1], "generate") != 0 &&
					  strcmp(values[1], "serve") != 0))
		return false;
	int option_end = AXIS_OPTION_START;
	for (; option_end < count &&
	       strncmp(values[option_end], "--", strlen("--")) == 0;
	     option_end++)
		if (strncmp(values[option_end],
			    "--placement=", strlen("--placement=")) == 0)
			result->weight_family =
				strcmp(values[option_end] +
					       strlen("--placement="),
				       "tp4_exl3") == 0 ?
					"exl3" :
					"original";
	const bool serving = strcmp(values[1], "serve") == 0;
	for (size_t entry = 0;
	     entry < sizeof(axis_choices) / sizeof(axis_choices[0]); entry++) {
		const struct axis_choice *choice = &axis_choices[entry];
		bool matches = strcmp(choice->weight_family,
				      result->weight_family) == 0;
		for (unsigned axis = 0; matches && axis < AXIS_COUNT; axis++)
			matches = result->selected[axis] == NULL ||
				  strcmp(result->selected[axis],
					 choice->value[axis]) == 0;
		if (!matches || !original_reference_scope(result, choice))
			continue;
		if (result->choice != NULL &&
		    !same_plan(result->choice, choice, serving)) {
			selection_log(result, stderr);
			(void)fputs(
				"axes refused: selected values need different settings for the unset axes; "
				"supply those axes. No axis defaults have been selected.\n",
				stderr);
			return false;
		}
		result->choice = choice;
	}
	if (result->choice == NULL)
		return false;
	if (result->choice->hold[0] != '\0') {
		selection_log(result, stderr);
		(void)fprintf(stderr, "axes refused: %s\n",
			      result->choice->hold);
		return false;
	}
	const char *settings = command_settings(result->choice, serving);
	const char *serve = serving ? result->choice->serve_settings : "";
	const size_t bytes = strlen(settings) + strlen(serve) + sizeof("\n");
	if (bytes > (size_t)INT_MAX || count > INT_MAX - (int)bytes)
		return false;
	char *storage = malloc(bytes);
	char **expanded = calloc((size_t)count + bytes + 1u, sizeof(*expanded));
	if (storage == NULL || expanded == NULL) {
		free(storage);
		free(expanded);
		return false;
	}
	(void)strcpy(storage, settings);
	(void)strcat(storage, "\n");
	(void)strcat(storage, serve);
	int expanded_count = AXIS_OPTION_START;
	expanded[0] = values[0];
	expanded[1] = values[1];
	char *save = NULL;
	for (char *setting = strtok_r(storage, "\n", &save); setting != NULL;
	     setting = strtok_r(NULL, "\n", &save)) {
		if (result->selected[AXIS_EXACTNESS] != NULL &&
		    strcmp(result->selected[AXIS_EXACTNESS], "exact") == 0 &&
		    reference_setting_conflicts(setting, count, values)) {
			free(expanded);
			free(storage);
			return false;
		}
		if (!explicit_key(setting, count, values)) {
			expanded[expanded_count++] = setting;
			result->inserted_count++;
		}
	}
	for (int index = AXIS_OPTION_START; index < count; index++)
		if (index >= option_end || axis_option(values[index]) < 0)
			expanded[expanded_count++] = values[index];
	result->values = expanded;
	result->count = expanded_count;
	result->storage = storage;
	return true;
}

void axis_arguments_free(struct axis_arguments_record *arguments)
{
	if (arguments->storage != NULL) {
		free(arguments->values);
		free(arguments->storage);
	}
	*arguments = (struct axis_arguments_record){ 0 };
}

void axis_log(const struct axis_arguments_record *arguments, FILE *stream)
{
	if (arguments->selected_count == 0)
		return;
	selection_log(arguments, stream);
	(void)fprintf(stream, "axes class=%s\n",
		      arguments->choice->error_class);
	for (unsigned index = 0; index < arguments->inserted_count; index++)
		(void)fprintf(stream, "axes set %s\n",
			      arguments->values[AXIS_OPTION_START + index]);
	(void)fputs(
		arguments->selected[AXIS_EXACTNESS] != NULL &&
				strcmp(arguments->selected[AXIS_EXACTNESS],
				       "exact") == 0 ?
			"axes: exact reference recipe values are required; unsupported overrides refuse.\n" :
			"axes: explicit low-level knobs override these settings; "
			"measurements apply to the recorded recipe.\n",
		stream);
}

static void settings_help(FILE *stream, const char *label, const char *settings)
{
	(void)fprintf(stream, "  %s:\n", label);
	if (settings[0] == '\0') {
		(void)fputs("    (none)\n", stream);
		return;
	}
	for (const char *argument = settings; argument[0] != '\0';) {
		const char *end = strchr(argument, '\n');
		const size_t bytes = end != NULL ? (size_t)(end - argument) :
						   strlen(argument);
		(void)fputs("    ", stream);
		(void)fwrite(argument, 1u, bytes, stream);
		(void)fputc('\n', stream);
		if (end == NULL)
			break;
		argument = end + 1;
	}
}

/* Partial-axis resolution depends on the concrete plan. Help also preserves
 * every distinct evidence scope attached to that plan. */
static bool same_help(const struct axis_choice *left,
		      const struct axis_choice *right)
{
	return strcmp(left->weight_family, right->weight_family) == 0 &&
	       same_plan(left, right, true) && same_plan(left, right, false) &&
	       strcmp(left->evidence, right->evidence) == 0;
}

static void choices_help(FILE *stream, bool available)
{
	for (size_t entry = 0;
	     entry < sizeof(axis_choices) / sizeof(axis_choices[0]); entry++) {
		const struct axis_choice *choice = &axis_choices[entry];
		if ((choice->hold[0] == '\0') != available)
			continue;
		bool shown = false;
		for (size_t prior = 0; !shown && prior < entry; prior++)
			shown = same_help(choice, &axis_choices[prior]);
		if (shown)
			continue;
		(void)fprintf(stream, "\nweight_family=%s\n",
			      choice->weight_family);
		for (size_t other = entry;
		     other < sizeof(axis_choices) / sizeof(axis_choices[0]);
		     other++) {
			const struct axis_choice *candidate =
				&axis_choices[other];
			if (same_help(choice, candidate))
				(void)fprintf(
					stream, "  %s / %s / %s / %s [%s]\n",
					candidate->value[AXIS_CATEGORY],
					candidate->value[AXIS_EXACTNESS],
					candidate->value[AXIS_CONCURRENCY],
					candidate->value[AXIS_FOCUS],
					candidate->buckets);
		}
		(void)fprintf(stream, "  class: %s\n", choice->error_class);
		if (!available) {
			(void)fprintf(stream,
				      "  unavailable: %s\n  evidence: %s\n",
				      choice->hold, choice->evidence);
			(void)fputs(
				"  No settings are applied for a held tuple.\n",
				stream);
			continue;
		}
		(void)fputs(
			"  Settings: one argv token per indented line; spaces stay in the token.\n",
			stream);
		settings_help(stream, "serve base", choice->settings);
		settings_help(stream, "generate base",
			      command_settings(choice, false));
		settings_help(stream, "serve additions",
			      choice->serve_settings);
		(void)fprintf(stream, "  evidence: %s\n", choice->evidence);
	}
}

int help_axes(FILE *stream)
{
	(void)fputs(
		"Axes on generate/serve; no axis is selected by default. With no axes, the CLI is unchanged.\n"
		"Balanced/fast low-level knobs win in either order; ordinary admission still applies.\n"
		"Unset axes have no implicit preset default. A partial tuple requires common settings across\n"
		"every compatible full tuple; otherwise it refuses and asks for the missing axes.\n"
		"Original exact alone selects its supported text/C1 serial M1 scope; conflicting recipe knobs refuse.\n"
		"Weight family follows explicit --placement; omitted placement selects the original-weight family.\n"
		"Within that family the tuple chooses a placement; exact requires its recorded PP placement.\n"
		"Use --placement=tp4_exl3 with EXL3. Named presets and preset aliases are removed.\n"
		"\nExamples (replace the uppercase message/table/weight paths):\n"
		"  anemone generate --placement=tp4_exl3 --category=code "
		"--exactness=balanced --concurrency=single --focus=decode "
		"MESSAGES TABLES WEIGHTS\n"
		"  anemone serve --placement=tp4_exl3 --category=code "
		"--exactness=balanced --concurrency=multiple --focus=decode TABLES WEIGHTS\n"
		"  anemone help knobs draft_\n",
		stream);
	for (unsigned axis = 0; axis < AXIS_COUNT; axis++) {
		(void)fprintf(stream, "  --%s=", axis_domains[axis].name);
		for (size_t value = 0; value < axis_domains[axis].count;
		     value++)
			(void)fprintf(stream, "%s%s", value == 0 ? "" : "|",
				      axis_domains[axis].values[value]);
		(void)fprintf(stream, "\n    %s\n", axis_domains[axis].meaning);
	}
	for (size_t tier = 0; tier < sizeof(axis_tiers) / sizeof(axis_tiers[0]);
	     tier++)
		(void)fprintf(stream, "\nexactness=%s: %s\n%s%s\n",
			      axis_tiers[tier].name,
			      axis_tiers[tier].error_class,
			      axis_tiers[tier].hold[0] != '\0' ? "HOLD: " : "",
			      axis_tiers[tier].hold);
	(void)fputs("\nFast error-budget proposal:\n", stream);
	(void)fputs(axis_quality_budget_text, stream);
	(void)fputs(
		"\nAll-balanced is the intended general-purpose recommendation; its admission and\n"
		"mixed-user/C1-C8 qualification are stated below. Current defaults are unchanged.\n"
		"Tuple order below: category / exactness / concurrency / focus.\n"
		"Equal settings with the same scoped evidence are shown once.\n",
		stream);
	(void)fputs("\nRunnable axis choices:\n", stream);
	choices_help(stream, true);
	(void)fputs("\nHeld axis choices:\n", stream);
	choices_help(stream, false);
	return ferror(stream) != 0 ? 1 : 0;
}
