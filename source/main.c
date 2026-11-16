/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* main.c: the program's entry (commands.c runs the command): the help topics, then the axes expanded into the
 * arguments run_program reads. */
#include <stdio.h>
#include <string.h>

#include "startup_axes.h"
#include "commands.h"

/* anemone help TOPIC [PREFIX]: the arguments' places and counts */
#define HELP_TOPIC 2
#define HELP_PREFIX 3
#define HELP_TOPIC_ARGUMENTS 3
#define HELP_PREFIX_ARGUMENTS 4

/* Keep expanded storage alive through shutdown: text options may refer to its bytes. */
int main(int argument_count, char **arguments)
{
	if (argument_count > 1 && strcmp(arguments[1], "help") == 0) {
		if (argument_count == HELP_TOPIC_ARGUMENTS &&
		    strcmp(arguments[HELP_TOPIC], "axes") == 0)
			return help_axes(stdout);
		if ((argument_count == HELP_TOPIC_ARGUMENTS ||
		     argument_count == HELP_PREFIX_ARGUMENTS) &&
		    strcmp(arguments[HELP_TOPIC], "knobs") == 0)
			return help_knobs(
				argument_count == HELP_PREFIX_ARGUMENTS ?
					arguments[HELP_PREFIX] :
					"",
				stdout);
		(void)fputs(
			"usage: anemone help axes | anemone help knobs [prefix]\n",
			stderr);
		return PROGRAM_WRONG_USAGE;
	}
	struct axis_arguments_record expanded = { 0 };
	if (!axis_expand(argument_count, arguments, &expanded)) {
		axis_arguments_free(&expanded);
		(void)fputs("axes: invalid selection; see anemone help axes\n",
			    stderr);
		return PROGRAM_WRONG_USAGE;
	}
	const int status =
		run_program(expanded.count, expanded.values, &expanded);
	axis_arguments_free(&expanded);
	return status;
}
