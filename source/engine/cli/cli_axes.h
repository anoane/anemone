/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Startup-only axis selection. Ordinary CLI and model admission stay authoritative. */
#ifndef CLI_AXES_H
#define CLI_AXES_H

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

/* One row of a GPU split's table (gpu_split is the --placement text). */
struct axis_choice {
	const char *value[AXIS_COUNT];
	const char *weight_family, *gpu_split;
	/* Trusted generated argv: one complete --key=value token per newline.
	 * Tokens contain no NUL/newline; spaces in values are preserved. */
	const char *settings, *serve_settings, *generate_settings;
	const char *buckets, *error_class, *evidence, *hold;
	/* Per command, one line per inserting axis value (or the split's
	 * baseline): "axis=value: --key --key", optionally followed by
	 * " (inactive: no DSpark on this route)". */
	const char *sources, *generate_sources;
	/* Per command, one line per knob a declared cross-axis bound changed
	 * at composition: "new old bounding source verb" (verb: bounded by a
	 * cap, required by a floor). Startup evaluates every bound again on the
	 * final values, explicit ones included (old is the set value). */
	const char *clamps, *generate_clamps;
	/* On serve the row applies only where the final --session_slots is exactly this count (BATCH's 8); elsewhere
	 * its concurrency=balanced row runs (thin steps for as many sessions as fit). 0: at any count. */
	unsigned serve_slots_exactly;
};

/* One declared cross-axis bound (axes.json cross_axis_bounds). A cap holds
 * each bounded knob at most the bounding knob's value less offset, and never
 * below least, the knob's parser least (one entry per bounded knob); a floor
 * raises each to at least least where the bounding knob holds value and the
 * under knob is not 0. bounded: the keys, space separated. */
struct axis_bound {
	bool floor;
	const char *bound, *value, *under, *bounded;
	unsigned offset, least;
};

/* A GPU split's base (stage 3): its route's knobs over the raw engine defaults, one --key=value per line, and the
 * route's pins equal to a raw default, omitted (the host test pins each and expects no parsed change). */
struct axis_base {
	const char *gpu_split, *route, *settings, *raw_defaults;
};

/* Proposals have no executable settings and cannot participate in resolution. */
struct axis_retained_route {
	const char *name, *weight_family, *proposed_axes;
	const char *knob, *fallback, *constraints, *hold;
};

struct axis_arguments_record {
	int count;
	char **values;
	char *storage;
	int explicit_count;
	char *const *explicit_values;
	/* choice: the row whose settings were inserted. slots_unmet: the selected row when it did not apply at the
	 * serve's slot count (choice is then its concurrency=balanced row), else NULL. */
	const struct axis_choice *choice, *slots_unmet;
	const char *selected[AXIS_COUNT];
	unsigned selected_count, inserted_count;
	const char *weight_family, *gpu_split;
};

const struct axis_choice *axis_choices_entry(size_t *count_out);
const struct axis_retained_route *axis_retained_routes_entry(size_t *count_out);
const struct axis_base *axis_bases_entry(size_t *count_out);
bool axis_expand(int count, char **values,
		 struct axis_arguments_record *result);
void axis_arguments_free(struct axis_arguments_record *arguments);
void axis_log(const struct axis_arguments_record *arguments, FILE *stream);
int help_axes(FILE *stream);
int help_knobs(const char *prefix, FILE *stream);
/* Whether the knob catalogue has a row for the option of argument (its name, up to any "="). */
bool knob_catalogued(const char *argument);

#endif
