/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "startup_axes.h"
#include "cli_axes_data.h"

#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define AXIS_DECIMAL_BASE 10 /* knob values are decimal */

/* AXIS_BOUND_WORD_BYTES: a knob word (key=value) a bound writes; AXIS_BOUND_WORDS_MOST: the bounded knobs one row
 * inserts at most. */
enum {
	AXIS_OPTION_START = 2,
	AXIS_SOURCE_KEY_BYTES = 64,
	AXIS_BOUND_WORD_BYTES = 64,
	AXIS_BOUND_WORDS_MOST = 8
};
enum axis_index { AXIS_CATEGORY, AXIS_EXACTNESS, AXIS_CONCURRENCY, AXIS_FOCUS };
#define AXIS_PLACEMENT_OPTION "--placement="
#define AXIS_WEIGHTS_OPTION "--weights="
#define AXIS_EXL3_SPLIT "tp4_exl3" /* --weights=exl3's default GPU split */

_Static_assert(sizeof(axis_domains) / sizeof(axis_domains[0]) == AXIS_COUNT,
	       "axis registry and parser must have the same dimensions");

const struct axis_choice *axis_choices_entry(size_t *count_out)
{
	*count_out = sizeof(axis_choices) / sizeof(axis_choices[0]);
	return axis_choices;
}

const struct axis_base *axis_bases_entry(size_t *count_out)
{
	*count_out = sizeof(axis_bases) / sizeof(axis_bases[0]);
	return axis_bases;
}

const struct axis_retained_route *axis_retained_routes_entry(size_t *count_out)
{
	*count_out =
		sizeof(axis_retained_routes) / sizeof(axis_retained_routes[0]);
	return axis_retained_routes;
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

/* A clamp line's next field, and a field's bytes (fields end at a space or
 * at the end of the line). */
static const char *clamp_field(const char *field)
{
	const char *const space = strchr(field, ' ');
	return space != NULL ? space + 1 : field + strlen(field);
}

static int field_bytes(const char *field)
{
	return (int)strcspn(field, " \n");
}

/* The clamp line of the knob with this key, or NULL. */
static const char *clamp_line(const char *clamps, const char *option)
{
	const char *key;
	const size_t bytes = option_key(option, &key);
	for (const char *line = clamps; line[0] != '\0';) {
		const char *candidate;
		if (option_key(line, &candidate) == bytes &&
		    strncmp(candidate, key, bytes) == 0)
			return line;
		const char *const end = strchr(line, '\n');
		if (end == NULL)
			break;
		line = end + 1;
	}
	return NULL;
}

static const char *command_settings(const struct axis_choice *choice,
				    bool serving);

/* The value the command gives key: the user's explicit argument (by_user),
 * else the row's composed setting, else NULL. A composed value ends at its
 * newline. */
static const char *final_value(const char *option,
			       const struct axis_choice *choice, bool serving,
			       int count, char *const *values, bool *by_user)
{
	const char *key;
	const size_t bytes = option_key(option, &key);
	*by_user = false;
	for (int index = AXIS_OPTION_START;
	     index < count && strncmp(values[index], "--", strlen("--")) == 0;
	     index++) {
		const char *candidate;
		if (option_key(values[index], &candidate) == bytes &&
		    strncmp(key, candidate, bytes) == 0) {
			const char *const equals = strchr(values[index], '=');
			*by_user = true;
			return equals != NULL ? equals + 1 : "";
		}
	}
	const char *const composed[] = { command_settings(choice, serving),
					 serving ? choice->serve_settings :
						   "" };
	for (size_t text = 0; text < sizeof(composed) / sizeof(composed[0]);
	     text++)
		for (const char *line = composed[text]; line[0] != '\0';) {
			const char *candidate;
			if (option_key(line, &candidate) == bytes &&
			    strncmp(key, candidate, bytes) == 0)
				return strchr(line, '=') + 1;
			const char *const end = strchr(line, '\n');
			if (end == NULL)
				break;
			line = end + 1;
		}
	return NULL;
}

/* Whether a bound's space-separated key list names key. */
static bool key_listed(const char *list, const char *key, size_t bytes)
{
	for (const char *cursor = list; cursor[0] != '\0';) {
		const size_t length = strcspn(cursor, " ");
		if (length == bytes && strncmp(cursor, key, bytes) == 0)
			return true;
		cursor += length + (cursor[length] == ' ' ? 1u : 0u);
	}
	return false;
}

/* What the declared bounds make of one inserted setting (stage 3 review D1): they act on
 * the final values, explicit ones included, and change only inserted knobs. A
 * cap lowers the setting to the bounding knob's value less the offset, never
 * below the knob's parser least (stage 3 review D5), also where the
 * final speculation is off (review D2: an inactive draft set keeps its clamp,
 * noted inactive); a floor raises it to its least where the bounding knob
 * holds the floor's value and the under knob is not 0. The set value is the
 * composition's unclamped word where it applied the bound, else the setting's
 * own. */
struct bound_outcome {
	char word[AXIS_BOUND_WORD_BYTES];
	const struct axis_bound *bound; /* the bound that changed it, or NULL */
	unsigned long set, value, bounding;
	bool by_user; /* the user set the bounding knob */
	const char *source; /* else the composed bounding knob's source */
	int source_bytes;
};

static bool bound_setting(const struct axis_choice *choice, bool serving,
			  const char *setting, int count, char *const *values,
			  struct bound_outcome *out)
{
	const char *key;
	const size_t key_bytes = option_key(setting, &key);
	const char *const line = clamp_line(
		serving ? choice->clamps : choice->generate_clamps, setting);
	const char *const set_word = line != NULL ? clamp_field(line) : setting;
	const char *const equals =
		memchr(set_word, '=', (size_t)field_bytes(set_word));
	for (size_t index = 0;
	     equals != NULL && isdigit((unsigned char)equals[1]) &&
	     index < sizeof(axis_bounds) / sizeof(axis_bounds[0]);
	     index++) {
		const struct axis_bound *const bound = &axis_bounds[index];
		if (!key_listed(bound->bounded, key, key_bytes))
			continue;
		bool by_user = false, under_by_user = false;
		const char *const bounding = final_value(
			bound->bound, choice, serving, count, values, &by_user);
		const char *const under =
			bound->floor ?
				final_value(bound->under, choice, serving,
					    count, values, &under_by_user) :
				NULL;
		*out = (struct bound_outcome){
			.set = strtoul(equals + 1, NULL, AXIS_DECIMAL_BASE)
		};
		out->value = out->set;
		if (bounding != NULL && isdigit((unsigned char)bounding[0])) {
			out->bounding =
				strtoul(bounding, NULL, AXIS_DECIMAL_BASE);
			const unsigned long capped =
				out->bounding >= bound->offset ?
					out->bounding - bound->offset :
					0u;
			const unsigned long lowest =
				capped > bound->least ? capped : bound->least;
			if (!bound->floor && out->set > lowest)
				out->value = lowest;
			if (bound->floor &&
			    out->bounding == strtoul(bound->value, NULL,
						     AXIS_DECIMAL_BASE) &&
			    under != NULL &&
			    strtoul(under, NULL, AXIS_DECIMAL_BASE) != 0 &&
			    out->set < bound->least)
				out->value = bound->least;
		}
		if (out->value != out->set) {
			out->bound = bound;
			out->by_user = by_user;
			out->source = line != NULL ?
					      clamp_field(clamp_field(
						      clamp_field(line))) :
					      "the axes";
			out->source_bytes = field_bytes(out->source);
		}
		const int written = snprintf(out->word, sizeof(out->word),
					     "%.*s=%lu",
					     (int)(equals - set_word), set_word,
					     out->value);
		return written > 0 && (size_t)written < sizeof(out->word);
	}
	return false;
}

/* After a bounded key: " 7 -> 5 (bounded by --rows=6 from focus=decode)" or
 * " 0 -> 1 (required by --tp_draft_shards=1 from the user)". */
static void bound_log(const struct bound_outcome *bound, FILE *stream)
{
	(void)fprintf(
		stream, " %lu -> %lu (%s by %s=%lu from %.*s)", bound->set,
		bound->value, bound->bound->floor ? "required" : "bounded",
		bound->bound->bound, bound->bounding,
		bound->by_user ? (int)strlen("the user") : bound->source_bytes,
		bound->by_user ? "the user" : bound->source);
}

/* The row's composed setting with this key, for the command, or NULL. */
static const char *composed_setting(const struct axis_choice *choice,
				    bool serving, const char *option)
{
	const char *key;
	const size_t bytes = option_key(option, &key);
	const char *const composed[] = { command_settings(choice, serving),
					 serving ? choice->serve_settings :
						   "" };
	for (size_t text = 0; text < sizeof(composed) / sizeof(composed[0]);
	     text++)
		for (const char *line = composed[text]; line[0] != '\0';) {
			const char *candidate;
			if (option_key(line, &candidate) == bytes &&
			    strncmp(key, candidate, bytes) == 0)
				return line;
			const char *const end = strchr(line, '\n');
			if (end == NULL)
				break;
			line = end + 1;
		}
	return NULL;
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

/* The knobs each axis value inserted: a source line's keys without the ones
 * the user gave explicitly, each applied bound after its key, then the line's
 * note (an inactive-draft note: dropped when the final speculation is on). A
 * line left without keys is not printed. */
static void sources_log(const struct axis_arguments_record *arguments,
			FILE *stream)
{
	const bool serving = strcmp(arguments->values[1], "serve") == 0;
	const char *line = serving ? arguments->choice->sources :
				     arguments->choice->generate_sources;
	while (line[0] != '\0') {
		const char *end = strchr(line, '\n');
		const char *last = end != NULL ? end : line + strlen(line);
		const char *key = strstr(line, ": ");
		const char *note =
			key != NULL && key < last ? strstr(key, " (") : NULL;
		note = note != NULL && note < last ? note : last;
		bool printed = false;
		for (const char *cursor = key != NULL && key < last ?
						  key + strlen(": ") :
						  last;
		     cursor < note;) {
			const char *space =
				memchr(cursor, ' ', (size_t)(note - cursor));
			const char *stop = space != NULL ? space : note;
			char option[AXIS_SOURCE_KEY_BYTES];
			const size_t bytes = (size_t)(stop - cursor);
			if (bytes < sizeof(option)) {
				memcpy(option, cursor, bytes);
				option[bytes] = '\0';
				if (!explicit_key(option,
						  arguments->explicit_count,
						  arguments->explicit_values)) {
					if (!printed)
						(void)fprintf(stream,
							      "axes from %.*s:",
							      (int)(key - line),
							      line);
					(void)fprintf(stream, " %s", option);
					const char *const inserted =
						composed_setting(
							arguments->choice,
							serving, option);
					struct bound_outcome bound;
					if (inserted != NULL &&
					    bound_setting(
						    arguments->choice, serving,
						    inserted,
						    arguments->explicit_count,
						    arguments->explicit_values,
						    &bound) &&
					    bound.bound != NULL)
						bound_log(&bound, stream);
					printed = true;
				}
			}
			cursor = space != NULL ? space + 1 : note;
		}
		/* Stage 3 review D2: the inactive-draft note goes only when the final
		 * speculation is on (an explicit --speculative=0 keeps it). */
		bool by_user = false;
		const char *const speculation =
			final_value("--speculative", arguments->choice, serving,
				    arguments->explicit_count,
				    arguments->explicit_values, &by_user);
		const bool speculating =
			speculation != NULL &&
			!(speculation[0] == '0' &&
			  (speculation[1] == '\0' || speculation[1] == '\n'));
		if (printed && note < last && !speculating)
			(void)fprintf(stream, "%.*s", (int)(last - note), note);
		if (printed)
			(void)fputc('\n', stream);
		line = end != NULL ? end + 1 : last;
	}
}

static const char *command_settings(const struct axis_choice *choice,
				    bool serving)
{
	return !serving && choice->generate_settings[0] != '\0' ?
		       choice->generate_settings :
		       choice->settings;
}

/* Two rows insert the same argv, class and hold for the command. */
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

/* The row of the GPU split's table for the selected values. An unset axis
 * takes balanced; under exact, an unset concurrency takes the reference
 * recipe's single session. NULL: the split has no table. */
static const struct axis_choice *
split_row(const struct axis_arguments_record *arguments, const char *split)
{
	const bool exact =
		arguments->selected[AXIS_EXACTNESS] != NULL &&
		strcmp(arguments->selected[AXIS_EXACTNESS], "exact") == 0;
	const char *value[AXIS_COUNT];
	for (unsigned axis = 0; axis < AXIS_COUNT; axis++)
		value[axis] = arguments->selected[axis] != NULL ?
				      arguments->selected[axis] :
			      exact && axis == AXIS_CONCURRENCY ? "single" :
								  "balanced";
	for (size_t entry = 0;
	     entry < sizeof(axis_choices) / sizeof(axis_choices[0]); entry++) {
		const struct axis_choice *choice = &axis_choices[entry];
		bool matches = strcmp(choice->gpu_split, split) == 0;
		for (unsigned axis = 0; matches && axis < AXIS_COUNT; axis++)
			matches = strcmp(choice->value[axis], value[axis]) == 0;
		if (matches)
			return choice;
	}
	return NULL;
}

/* The row of choice's split and values with concurrency=balanced: where a set applies only at one serving slot count,
 * the row a serve at any other count runs (the generator keeps it unheld). NULL: none. */
static const struct axis_choice *balanced_row(const struct axis_choice *choice)
{
	for (size_t entry = 0;
	     entry < sizeof(axis_choices) / sizeof(axis_choices[0]); entry++) {
		const struct axis_choice *candidate = &axis_choices[entry];
		bool matches =
			strcmp(candidate->gpu_split, choice->gpu_split) == 0 &&
			strcmp(candidate->value[AXIS_CONCURRENCY],
			       "balanced") == 0;
		for (unsigned axis = 0; matches && axis < AXIS_COUNT; axis++)
			matches = axis == AXIS_CONCURRENCY ||
				  strcmp(candidate->value[axis],
					 choice->value[axis]) == 0;
		if (matches)
			return candidate;
	}
	return NULL;
}

/* Whether the user's --session_slots is exactly count (the default, max, is not a count). */
static bool slots_given_exactly(unsigned count, int option_count,
				char *const *values)
{
	static const char option[] = "--session_slots=";
	for (int index = AXIS_OPTION_START;
	     index < option_count &&
	     strncmp(values[index], "--", strlen("--")) == 0;
	     index++) {
		if (strncmp(values[index], option, strlen(option)) != 0)
			continue;
		const char *const digits = values[index] + strlen(option);
		char *end = NULL;
		const unsigned long given =
			strtoul(digits, &end, AXIS_DECIMAL_BASE);
		return isdigit((unsigned char)digits[0]) && end != NULL &&
		       end[0] == '\0' && given == count;
	}
	return false;
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

/* --axes=off: the raw engine defaults with nothing inserted (stage 3's gates
 * and references); the argv without that word. */
#define AXES_OPTION "--axes="
#define AXES_OFF "--axes=off"

static bool axes_off_expand(int count, char **values, int off_index,
			    struct axis_arguments_record *result)
{
	char **const expanded =
		(char **)calloc((size_t)count, sizeof(*expanded));
	char *const storage = malloc(1u);
	if (expanded == NULL || storage == NULL) {
		free((void *)expanded);
		free(storage);
		return false;
	}
	int kept = 0;
	for (int index = 0; index < count; index++)
		if (index != off_index)
			expanded[kept++] = values[index];
	result->values = expanded;
	result->count = kept;
	result->storage = storage;
	return true;
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
	int off_index = 0;
	for (int index = AXIS_OPTION_START;
	     index < count && strncmp(values[index], "--", strlen("--")) == 0;
	     index++) {
		if (strncmp(values[index], AXES_OPTION, strlen(AXES_OPTION)) ==
		    0) {
			if (strcmp(values[index], AXES_OFF) != 0 ||
			    off_index != 0) {
				(void)fputs(
					"axes: --axes takes only off, once\n",
					stderr);
				return false;
			}
			off_index = index;
			continue;
		}
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
	if (off_index != 0 && result->selected_count != 0) {
		(void)fputs("axes refused: --axes=off takes no axis flag\n",
			    stderr);
		return false;
	}
	if (off_index != 0)
		return axes_off_expand(count, values, off_index, result);
	/* Stage 3: without an axis flag, generate and serve take the split's
	 * all-balanced row (its base); other commands insert nothing. */
	if (count < AXIS_OPTION_START || (strcmp(values[1], "generate") != 0 &&
					  strcmp(values[1], "serve") != 0))
		return result->selected_count == 0;
	/* The table is keyed on (split, weight family): an explicit --placement names the split, --weights alone its
	 * family's default split (EXL3: TP4_EXL3), as the parser resolves them later. */
	int option_end = AXIS_OPTION_START;
	const char *split = NULL, *weights = NULL;
	for (; option_end < count &&
	       strncmp(values[option_end], "--", strlen("--")) == 0;
	     option_end++) {
		if (strncmp(values[option_end], AXIS_PLACEMENT_OPTION,
			    strlen(AXIS_PLACEMENT_OPTION)) == 0)
			split = values[option_end] +
				strlen(AXIS_PLACEMENT_OPTION);
		if (strncmp(values[option_end], AXIS_WEIGHTS_OPTION,
			    strlen(AXIS_WEIGHTS_OPTION)) == 0)
			weights = values[option_end] +
				  strlen(AXIS_WEIGHTS_OPTION);
	}
	/* A value that names no family is refused as such, before any family check. */
	if (weights != NULL && strcmp(weights, "original") != 0 &&
	    strcmp(weights, "exl3") != 0) {
		(void)fprintf(
			stderr,
			"axes refused: --weights=%s is not original or exl3\n",
			weights);
		return false;
	}
	if (split == NULL)
		split = weights != NULL && strcmp(weights, "exl3") == 0 ?
				AXIS_EXL3_SPLIT :
				axis_default_split;
	const bool serving = strcmp(values[1], "serve") == 0;
	result->choice = split_row(result, split);
	if (result->choice == NULL && result->selected_count == 0)
		return true; /* a split without a table keeps its raw defaults */
	if (result->choice == NULL) {
		selection_log(result, stderr);
		(void)fprintf(
			stderr,
			"axes refused: GPU split %s has no axis table; the tables are %s.\n",
			split, axis_split_names);
		return false;
	}
	if (weights != NULL &&
	    strcmp(weights, result->choice->weight_family) != 0) {
		selection_log(result, stderr);
		(void)fprintf(
			stderr,
			"axes refused: --weights=%s conflicts with GPU split %s, whose table runs %s weights\n",
			weights, split, result->choice->weight_family);
		return false;
	}
	result->weight_family = result->choice->weight_family;
	result->gpu_split = result->choice->gpu_split;
	if (result->choice->hold[0] != '\0') {
		selection_log(result, stderr);
		(void)fprintf(stderr, "axes refused: %s\n",
			      result->choice->hold);
		return false;
	}
	/* A set that needs one session count (TP4_EXL3 BATCH's 8) applies only where the user's
	 * --session_slots is exactly that count; at any other, the default max included, the serve runs the row's
	 * concurrency=balanced row, thin steps for as many sessions as fit, and startup says how many fit. */
	if (serving && result->choice->serve_slots_exactly != 0u &&
	    !slots_given_exactly(result->choice->serve_slots_exactly, count,
				 values)) {
		result->slots_unmet = result->choice;
		result->choice = balanced_row(result->choice);
		if (result->choice == NULL || result->choice->hold[0] != '\0')
			return false;
	}
	const char *settings = command_settings(result->choice, serving);
	const char *serve = serving ? result->choice->serve_settings : "";
	const size_t settings_bytes = strlen(settings),
		     serve_bytes = strlen(serve);
	/* "settings\nserve\0", then the words the bounds write. */
	const size_t words_bytes =
		(size_t)AXIS_BOUND_WORDS_MOST * AXIS_BOUND_WORD_BYTES;
	const size_t separators = sizeof("\n") + words_bytes;
	if (settings_bytes > (size_t)INT_MAX - separators ||
	    serve_bytes > (size_t)INT_MAX - separators - settings_bytes) {
		return false;
	}
	const size_t bytes = settings_bytes + serve_bytes + separators;
	if (count > INT_MAX - (int)bytes) {
		return false;
	}
	char *storage = malloc(bytes);
	char **expanded =
		(char **)calloc((size_t)count + bytes + 1u, sizeof(*expanded));
	if (storage == NULL || expanded == NULL) {
		free(storage);
		free((void *)expanded);
		return false;
	}
	memcpy(storage, settings, settings_bytes);
	storage[settings_bytes] = '\n';
	memcpy(storage + settings_bytes + 1u, serve, serve_bytes);
	storage[settings_bytes + 1u + serve_bytes] = '\0';
	char *const words = storage + settings_bytes + 1u + serve_bytes + 1u;
	unsigned words_used = 0;
	int expanded_count = AXIS_OPTION_START;
	expanded[0] = values[0];
	expanded[1] = values[1];
	char *save = NULL;
	for (char *setting = strtok_r(storage, "\n", &save); setting != NULL;
	     setting = strtok_r(NULL, "\n", &save)) {
		if (result->selected[AXIS_EXACTNESS] != NULL &&
		    strcmp(result->selected[AXIS_EXACTNESS], "exact") == 0 &&
		    reference_setting_conflicts(setting, count, values)) {
			free((void *)expanded);
			free(storage);
			return false;
		}
		/* Each declared bound on the final values (bound_setting). */
		struct bound_outcome bound;
		if (bound_setting(result->choice, serving, setting, count,
				  values, &bound) &&
		    strcmp(bound.word, setting) != 0) {
			if (words_used == AXIS_BOUND_WORDS_MOST) {
				free((void *)expanded);
				free(storage);
				return false;
			}
			setting =
				memcpy(words + (size_t)(words_used++ *
							AXIS_BOUND_WORD_BYTES),
				       bound.word, strlen(bound.word) + 1u);
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
		free((void *)arguments->values);
		free(arguments->storage);
	}
	*arguments = (struct axis_arguments_record){ 0 };
}

void axis_log(const struct axis_arguments_record *arguments, FILE *stream)
{
	if (arguments->choice == NULL)
		return;
	selection_log(arguments, stream);
	const struct axis_choice *choice = arguments->choice;
	(void)fprintf(stream, "axes gpu_split=%s tuple=%s/%s/%s/%s\n",
		      choice->gpu_split, choice->value[AXIS_CATEGORY],
		      choice->value[AXIS_EXACTNESS],
		      choice->value[AXIS_CONCURRENCY],
		      choice->value[AXIS_FOCUS]);
	(void)fprintf(stream, "axes class=%s\n", choice->error_class);
	for (unsigned index = 0; index < arguments->inserted_count; index++)
		(void)fprintf(stream, "axes set %s\n",
			      arguments->values[AXIS_OPTION_START + index]);
	sources_log(arguments, stream);
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

/* Help shows rows of one split with one plan together, keeping every distinct
 * evidence scope attached to that plan. */
static bool same_help(const struct axis_choice *left,
		      const struct axis_choice *right)
{
	return strcmp(left->gpu_split, right->gpu_split) == 0 &&
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
		(void)fprintf(stream, "\ngpu_split=%s weight_family=%s\n",
			      choice->gpu_split, choice->weight_family);
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
		"Axes on generate/serve. --placement picks the GPU split (default pp5); each split has its own table.\n"
		"Each selected axis value adds its knob set to the split's base (below); each knob belongs to one\n"
		"axis. An unset axis takes balanced, so a generate or serve without axis flags runs the split's\n"
		"all-balanced row. No set pins the context or the slots: by default every session has 1M\n"
		"context and serve takes as many sessions as fit (--session_slots=max, no cap). TP4_EXL3's\n"
		"concurrency=multiple runs BATCH only at --session_slots=8 (else its concurrency=balanced row).\n"
		"--axes=off runs the raw engine defaults with nothing inserted.\n"
		"Explicit low-level knobs win in either order; ordinary admission still applies.\n"
		"Exact selects the split's reference recipe, one session (PP5: text/C1 serial M1; TP4_EXL3: exact\n"
		"shaping on quantized weights, not official-exact); conflicting recipe knobs refuse. Use\n"
		"--placement=tp4_exl3 with EXL3. Original single-user decode is\n"
		"fastest on --placement=tp4_pro. Named presets and preset aliases are removed.\n"
		"\nExamples (replace the uppercase message/table/weight paths):\n"
		"  anemone generate --placement=tp4_exl3 --category=code "
		"--exactness=balanced --concurrency=single --focus=decode "
		"MESSAGES TABLES WEIGHTS\n"
		"  anemone serve --placement=tp4_exl3 --category=code "
		"--exactness=balanced --concurrency=multiple --focus=decode --context_tokens=65536\n"
		"    --session_slots=8 TABLES WEIGHTS\n"
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
		"mixed-user/C1-C8 qualification are stated below. A run without an axis flag is\n"
		"all-balanced: its split's base below (--axes=off runs the raw engine defaults).\n"
		"Tuple order below: category / exactness / concurrency / focus.\n"
		"Equal settings with the same scoped evidence are shown once.\n",
		stream);
	(void)fputs(
		"\nEach split's base (no axis flag, or a knob no set gives):\n",
		stream);
	for (size_t entry = 0;
	     entry < sizeof(axis_bases) / sizeof(axis_bases[0]); entry++) {
		const struct axis_base *const base = &axis_bases[entry];
		(void)fprintf(
			stream,
			"\n%s, from %s:\n%s\n  over the raw defaults: %s\n",
			base->gpu_split, base->route, base->settings,
			base->raw_defaults);
	}
	(void)fputs("\nRunnable axis choices:\n", stream);
	choices_help(stream, true);
	(void)fputs("\nHeld axis choices:\n", stream);
	choices_help(stream, false);
	(void)fputs(
		"\nRetained route proposals (held metadata; never inserted settings):\n",
		stream);
	for (size_t entry = 0; entry < sizeof(axis_retained_routes) /
					       sizeof(axis_retained_routes[0]);
	     entry++) {
		const struct axis_retained_route *route =
			&axis_retained_routes[entry];
		(void)fprintf(stream,
			      "\n%s weight_family=%s\n  proposed: %s\n"
			      "  branch control: %s\n  old fallback: %s\n"
			      "  constraints: %s\n  HOLD: %s\n",
			      route->name, route->weight_family,
			      route->proposed_axes, route->knob,
			      route->fallback, route->constraints, route->hold);
	}
	return ferror(stream) != 0 ? 1 : 0;
}
