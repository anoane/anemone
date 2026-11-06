/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * teacher_forced.h: the teacher-forced harness of the model's program (first_token_plan.md 3.3 and 4.2, unit U14).
 * It drives a program's segments through a backend one position a step, puts before each layer the streams and the
 * carried pre that a stepwise capture of the reference (tools/oracle/stepwise_capture.py) holds as leaving the layer
 * before it, and writes what the engine computed as a capture folder of the same names, shapes and dtypes, which
 * tools/oracle/compare_engine_capture.py holds to the reference's. Test support: no part of the engine.
 *
 * Contract:
 *   - The program is a thin one of program_prepare (model_seam.h) for steps of one row, its cards open and their state
 *     set by arena_initialise. Its layers are model_layout.h's. The segment of a layer holds, in this order, the mix
 *     finish and the post of its attention site, the mix finish of its FFN site, one router select, the post of its
 *     FFN site with the combine (hyper_connection.h, router.h), or the fused launches that stand for them
 *     (program_view.h), any other launches between them; their records read and write the card's streams and carried
 *     pre (program_card_record). A layer split at the front boundary executes both physical fragments once with
 *     hand-offs at each; its input is injected before the first fragment, capture moments are written by the
 *     fragment that ends them and final output/routing only by the last. The segment without a layer that holds
 *     the embed launch is the embedding's; the one that holds the final collapse is the head's, on the result card,
 *     which has a logits and a result area. Every other segment without a layer runs as it is at every step. A
 *     program that breaks this or program_view.h's rules, a range or a hand-off that does not fit the harness's
 *     staging, a card the program uses that is not open, or a run layer whose n-gram gate is not that of a table the
 *     cards read (when they read any), is refused before any step runs.
 *   - Before the first step: the session settings of slot 0 (seed 0, temperature 0, top_p 1) go to the result card,
 *     and each run layer's expert table and router bias come back from its card. A step at position p from 0: the
 *     step area (one row: p, the capture's token at p, slot 0, the logits flag) goes to every card that has one, and
 *     to each card whose program names n-gram tables the rows of the token for each of them; then the segments in
 *     the program's order, skipping a layer not in run_layers with its hand-offs and, without a head to run, the
 *     head's; after each segment its hand-offs as step.h moves them, through staging. A head runs in free mode, and
 *     in injected mode where the capture holds the streams leaving the last layer.
 *   - Injected (TEST_TEACHER_FORCED_INJECTED): before a layer L of 1 or more, the capture's layer_<L-1>.block.output
 *     and .output_collapse_weights at p go into the card's streams and carried pre; before the head, those of the
 *     last layer. Layer 0 takes the engine's own embedding. At break_layer the carried pre goes in as zeros, every
 *     step: the layer gate must fail at that layer alone. Free (TEST_TEACHER_FORCED_FREE): nothing is put in; every
 *     layer must run.
 *   - Written into folder (which must not exist), each record as the capture names it: input.token_ids (the
 *     capture's, the run's input) and, per position, input.embeddings (stream 0 of the embedding segment's output,
 *     whose four streams must be equal and whose pre (1, 0, 0, 0), or the run fails); for each run layer
 *     layer_L.block.output, .output_collapse_weights, layer_L.mlp.routed_expert_ids, .route_weights and
 *     .biased_scores, the experts in the gate's order (the job table of the router decoded through the layer's
 *     expert table; their affinity plus the bias descending, the lower expert first among equals); for a run layer
 *     with an n-gram gate layer_L.ngram.output (the streams once the gate added the rows) and, where the cards read
 *     its table, layer_L.ngram.row_ids; with logits, logits.all_rows. At each capture point of a run layer (the
 *     capture holds layer_L.position_P.attn_site.normed), the point records of teacher_forced.c's table. bfloat16
 *     values are stored as float32 and say so (capture_writer.h); manifest.json is written last, with conditions (a
 *     JSON object the caller gives) and results, and says complete only when every step and every write succeeded.
 *   - Every wait has wait_limit_nanoseconds. A failure is one error line and the status; the folder then holds a
 *     manifest that says incomplete (when it could be written). Staging that a copy not seen finished may still read
 *     stays allocated, with one error line: only closing the card makes it safe to free.
 *   - A capture of the reference refused by capture_reader.h, a record the run needs that it lacks or holds in
 *     another type or shape, and positions beyond its tokens are STATUS_INVALID_ARGUMENT, as is a NULL member,
 *     a layer count of 0 or above TEST_TEACHER_FORCED_MOST_LAYERS, no position, run layers beyond the layer count, a
 *     break layer that is not a run layer of 1 or more in injected mode, and a free mode with a layer not run.
 */
#ifndef TEST_TEACHER_FORCED_H
#define TEST_TEACHER_FORCED_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/engram.h"
#include "anemone/model_seam.h"
#include "anemone/status.h"
#include "capture_reader.h"

#define TEST_TEACHER_FORCED_MOST_LAYERS 64u            /* bits of run_layers */
#define TEST_TEACHER_FORCED_NO_BREAK UINT32_MAX

enum test_teacher_forced_mode { TEST_TEACHER_FORCED_INJECTED, TEST_TEACHER_FORCED_FREE };

struct test_teacher_forced_run {
	const struct backend_operations *backend;
	struct device_card *const *
		cards; /* the program's card_count: NULL for one it never uses */
	const struct card_description_record *
		card_descriptions; /* cards' selected classes; NULL: reference order */
	const struct execution_program *program; /* for steps of one row */
	const struct test_capture
		*capture; /* the stepwise capture of the reference */
	const struct engram_record
		*engram; /* NULL: no card of the program reads n-gram rows */
	const char *folder; /* written; must not exist */
	const char *conditions; /* a JSON object, as manifest.json holds it */
	uint64_t run_layers; /* bit l: layer l runs and is written */
	uint64_t wait_limit_nanoseconds;
	uint32_t layer_count; /* the model's */
	uint32_t engram_layers
		[ENGRAM_MAXIMUM_TABLES]; /* the layer of each n-gram table */
	uint32_t positions; /* the first positions of the capture */
	uint32_t break_layer; /* or TEST_TEACHER_FORCED_NO_BREAK */
	enum test_teacher_forced_mode mode;
	bool logits; /* write logits.all_rows (a head must run) */
};

MUST_CHECK result_status
test_teacher_forced_write(const struct test_teacher_forced_run *run);

#endif
