/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Startup-only axis selection. Ordinary CLI and model admission stay authoritative. */
#ifndef STARTUP_AXES_H
#define STARTUP_AXES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

enum {
	AXIS_CATEGORY_VALUE,
	AXIS_EXACTNESS_VALUE,
	AXIS_CONCURRENCY_VALUE,
	AXIS_FOCUS_VALUE,
	AXIS_COUNT
};

struct axis_domain {
	const char *name, *prefix, *meaning;
	const char *const *values;
	size_t count;
};

struct axis_tier {
	const char *name, *error_class, *hold;
};

struct axis_choice {
	const char *value[AXIS_COUNT];
	const char *weight_family;
	/* Trusted generated argv: one complete --key=value token per newline.
	 * Tokens contain no NUL/newline; spaces in values are preserved. */
	const char *settings, *serve_settings, *generate_settings;
	const char *buckets, *error_class, *evidence, *hold;
};

struct axis_arguments_record {
	int count;
	char **values;
	char *storage;
	int explicit_count;
	char *const *explicit_values;
	const struct axis_choice *choice;
	const char *selected[AXIS_COUNT];
	unsigned selected_count, inserted_count;
	const char *weight_family;
};

const struct axis_choice *axis_choices_entry(size_t *count_out);
bool axis_expand(int count, char **values,
		 struct axis_arguments_record *result);
void axis_arguments_free(struct axis_arguments_record *arguments);
void axis_log(const struct axis_arguments_record *arguments, FILE *stream);
int help_axes(FILE *stream);
int help_knobs(const char *prefix, FILE *stream);

#endif
