/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_option_texts.h: DeepSeek-V4.1-Flash's texts in the cli's help (P5d-lite; DESIGN_v2 section 4.8's
 * knob_help_preface and usage_lines): the cli prints each in its place, byte for byte as before.
 */
#ifndef DS41F_OPTION_TEXTS_H
#define DS41F_OPTION_TEXTS_H

/* help knobs' T, B and F paragraphs, after the engine's G0 lines (cli_knobs.c) */
extern const char ds41f_knob_help_preface[];

/* The usage text's model lines, by the first knob of each: cli_run.c prints each among the engine's lines */
enum ds41f_usage_line {
	DS41F_USAGE_EXL3_EXPERT_PARTS,
	DS41F_USAGE_TP_QUERIES_PIECE,
	DS41F_USAGE_TP_GRAPH_PARTS,
	DS41F_USAGE_ORIGINAL_BATCH,
	DS41F_USAGE_EXL3_TC_PRODUCT_BLOCKS,
	DS41F_USAGE_OFFICIAL_HC_ROWS,
	DS41F_USAGE_PREFILL_TAIL_OVERLAY,
	DS41F_USAGE_DENSE_STAGED,
	DS41F_USAGE_LINES
};
extern const char *const ds41f_usage_lines[DS41F_USAGE_LINES];

#endif
