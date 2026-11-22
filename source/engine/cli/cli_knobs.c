/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "cli_axes.h"

#include <string.h>

#include "engine/cli/cli_knobs.inc"
#include "models/deepseek_v41_flash/options/ds41f_option_texts.h"

int help_knobs(const char *prefix, FILE *stream)
{
	if (strncmp(prefix, "--", strlen("--")) == 0)
		prefix += strlen("--");
	const size_t bytes = strlen(prefix);
	bool found = false;
	(void)fputs(
		"Knob catalog: current parser; track-only rows are labelled unavailable.\n"
		"G0: --name=value before positional files; command/range/arity/duplicate and runtime gates apply.\n"
		"Numeric lists require distinct elements except batch_rows/batch_verifiers; replicate is last-wins.\n",
		stream);
	(void)fputs(ds41f_knob_help_preface, stream);
	for (size_t index = 0; index < sizeof(cli_knobs) / sizeof(cli_knobs[0]);
	     index++) {
		const char *name = cli_knobs[index].name;
		if (strncmp(name, "--", strlen("--")) == 0)
			name += strlen("--");
		if (strncmp(name, prefix, bytes) != 0)
			continue;
		found = true;
		(void)fprintf(stream, "\n%s\n%s", cli_knobs[index].name,
			      cli_knobs[index].detail);
	}
	if (!found)
		(void)fprintf(stream, "No knobs match prefix '%s'.\n", prefix);
	return ferror(stream) != 0 || !found ? 1 : 0;
}

bool knob_catalogued(const char *argument)
{
	const size_t bytes = strcspn(argument, "=");
	for (size_t index = 0; index < sizeof(cli_knobs) / sizeof(cli_knobs[0]);
	     index++)
		if (strlen(cli_knobs[index].name) == bytes &&
		    strncmp(cli_knobs[index].name, argument, bytes) == 0)
			return true;
	return false;
}
