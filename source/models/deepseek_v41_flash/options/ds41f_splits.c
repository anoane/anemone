/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_splits.c: the GPU splits DeepSeek-V4.1-Flash runs, its policy on each and its measured schedules (P5d-lite).
 */
#include "ds41f_splits.h"

#include "engine/cli/commands.h"

/* startup_policy_prepare and startup_policy_admit: PP5 is admitted at either CMP order, the TP splits at the
 * reference order (TP5 is not admitted); exact numerics are served on PP5, TP4_PRO and TP4_EXL3; without
 * --cmp_order the TP splits take the reference order, PP5 and TP5 the relaxed one. */
const struct ds41f_split_support ds41f_splits[DS41F_SPLITS] = {
	[GPU_SPLIT_PP5] = { .admitted = true,
			    .admits_exact = true,
			    .cmp_order_default = CMP_ORDER_RELAXED },
	[GPU_SPLIT_TP5] = { .cmp_order_default = CMP_ORDER_RELAXED },
	[GPU_SPLIT_TP4_PRO] = { .admitted_at_reference = true,
				.admits_exact = true,
				.cmp_order_default = CMP_ORDER_REFERENCE },
	[GPU_SPLIT_TP2_PP2_PRO] = { .admitted_at_reference = true,
				    .cmp_order_default = CMP_ORDER_REFERENCE },
	[GPU_SPLIT_TP2_SINGLE_SINGLE_PRO] = { .admitted_at_reference = true,
					      .cmp_order_default =
						      CMP_ORDER_REFERENCE },
	[GPU_SPLIT_TP4_EXL3] = { .admitted_at_reference = true,
				 .admits_exact = true,
				 .cmp_order_default = CMP_ORDER_REFERENCE },
};

const uint32_t ds41f_ladder_fine_rows[DS41F_LADDER_FINE_COUNT] = { 1792u, 1280u,
								   768u };

/* Measured TP4 BF16 programs, sorted by rows then verifiers. Other layouts need explicit prices. */
const uint32_t ds41f_default_batch_rows[MOST_BATCH_PROGRAMS] = {
	8, 8, 12, 12, 16, 24, 32, 48
};
const uint32_t ds41f_default_batch_verifiers[MOST_BATCH_PROGRAMS] = { 1, 2, 2,
								      4, 4, 4,
								      8, 8 };
const uint32_t ds41f_default_batch_step_us[MOST_BATCH_PROGRAMS] = {
	19500, 21500, 28600, 30000, 32000, 39000, 46000, 58000
};
