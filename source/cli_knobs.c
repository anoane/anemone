/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "startup_axes.h"

#include <string.h>

#include "cli_knobs_data.h"

int help_knobs(const char *prefix, FILE *stream)
{
	if (strncmp(prefix, "--", strlen("--")) == 0)
		prefix += strlen("--");
	const size_t bytes = strlen(prefix);
	bool found = false;
	(void)fputs(
		"Knob catalog: current parser; track-only rows are labelled unavailable.\n"
		"G0: --name=value before positional files; command/range/arity/duplicate and runtime gates apply.\n"
		"Numeric lists require distinct elements except batch_rows/batch_verifiers; replicate is last-wins.\n"
		"T: TP requires generate/serve, DSpark/hybrid, pro_emulation=0, reference CMP order, context>=16384\n"
		"and ordinary rows<=8. Context must be a multiple of 16; allocated draft_block is 5..7.\n"
		"TP load refuses; TP2 needs relaxed/fast numerics. TP4_EXL3 needs four class-80 CMPs and the map.\n"
		"Overall TP row capacity is <=8 or >=64, at most 2048 with split overlap or 1024 without.\n"
		"B: BATCH needs TP4_EXL3, pipeline_steps=1, overall capacity>=64 and valid program/verifier lists.\n"
		"Used verifier counts must fit sessions; row/verifier pairs must be unique. "
		"Costs match the sorted plan.\n"
		"F: nonzero tail programs need TP4_EXL3, one chunk in flight, exact planning, full steps, wide>=1024,\n"
		"TC MoE and trunk MMA. Front/thin tails need a tail; unrelated prefill knobs need not enable tails.\n"
		"Same-bits candidates are relative to their recipe, not original "
		"DeepSeek equivalence.\n"
		"Measured epochs and joint recipes remain scoped; unmeasured cells are "
		"explicit.\n",
		stream);
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
