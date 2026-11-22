/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_splits.h: the GPU splits DeepSeek-V4.1-Flash runs, its policy on each and its measured schedules (P5d-lite;
 * DESIGN_v2 section 4.8's struct model_split_support): the cli reads them.
 */
#ifndef DS41F_SPLITS_H
#define DS41F_SPLITS_H

#include <stdbool.h>
#include <stdint.h>

#include "engine/cli/cli_policy.h"

/* The engine-start ladder's and the tail programs' measured rows (the options' bounds and the cli's array sizes) */
#define DS41F_CHUNK_LADDER_TOP \
	1024u /* above it, wide_rows then 512, 256 and 64 rows (scheduler.c's exact plan: whole multiples of 256) */
/* --prefill_tail_rows: a full program of two of the exact plan's 64-row tail chunks, beside the ladder's (BATCH
 * programs take the thin ones' place, so MOST_PROGRAMS holds it), with a ladder of no 128-row program of its own */
#define DS41F_TAIL_PROGRAM_ROWS 128u
#define DS41F_TAIL_LONG_ROWS \
	320u /* --prefill_tail_rows=320: a full program of five tail chunks beside the 128-row one */
#define DS41F_TAIL_WIDEST_ROWS \
	448u /* --prefill_tail_rows=448: and one of seven beside the 320-row one */
#define DS41F_LADDER_1536_ROWS \
	1536u /* --prefill_ladder_1536: six whole granules of the exact plan (scheduler.c) */
#define DS41F_LADDER_FINE_COUNT \
	3u /* --prefill_ladder_fine: 1,792, 1,280 and 768 rows (7, 5 and 3 whole granules) */

/* A split's policy (startup_policy_prepare and startup_policy_admit): the row of its code in ds41f_splits[] */
struct ds41f_split_support {
	bool admitted; /* admitted at either CMP order (PP5) */
	bool admitted_at_reference; /* admitted at the reference CMP order (the TP splits) */
	bool admits_exact; /* NUMERICS_EXACT is served here */
	enum cmp_order_code
		cmp_order_default; /* without --cmp_order, unless the numerics are exact (then reference) */
};

#define DS41F_SPLITS          \
	(GPU_SPLIT_TP4_EXL3 + \
	 1) /* a row for each split code, in the codes' order */
extern const struct ds41f_split_support ds41f_splits[DS41F_SPLITS];
/* --prefill_ladder_fine's programs, in the ladder's order */
extern const uint32_t ds41f_ladder_fine_rows[DS41F_LADDER_FINE_COUNT];

#endif
