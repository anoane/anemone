/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * program_view.h: what a test reads of a model's program (model_seam.h) to run it a segment at a time: the kind of
 * each segment and, of a layer's segment, where its sites, its n-gram gate and its router lie and what their records
 * hold. First user: the teacher-forced harness (teacher_forced.h). Test support: no part of the engine.
 *
 * Contract:
 *   - The program's numbers are checked first: at least one segment; each segment and hand-off names a card of the
 *     program, a range of its card's launches (not empty) or a segment that exist; a card that a segment, a hand-off
 *     or a step area names is open (its entry of cards is not NULL); the result card is one of the program's.
 *   - A segment's kind comes from the kernels of its launches, whose handles hyper_connection.h's, router.h's and
 *     engram_rows.h's prepare functions give again on a card that prepared them (backend_seam.h). A segment with a
 *     layer is that layer's, one a layer, below layer_count. Of the segments without one, the one that holds an
 *     embed launch is the embedding's (exactly one such segment, one such launch); the one that holds a final
 *     collapse is the head's (at most one, on the result card, which has a logits and a result area); any other is
 *     TEST_PROGRAM_SEGMENT_OTHER.
 *   - Each layer of layers (bit l) has its segment, which holds in this order: its attention site's mix finish and post
 *     (a layer whose segment the program's front ends, model_seam.h, goes on in the next segment of its card: the
 *     two are read as one, first_segment names the first and segment the second)
 *     (without the combine), its FFN site's mix finish, one router select of one row, its FFN site's post with the
 *     combine; any other launches between them, none an embed or a final collapse. The records of the sites read
 *     and write the card's streams and carried pre (program_card_record). An n-gram layer's gate (engram_rows.h),
 *     which adds the rows into the card's streams, lies in its segment before the attention site's mix finish, one at
 *     most; no segment of no layer holds one.
 *   - A fused launch stands for the launches it fuses, its record read as theirs: a site front, or the coefficient
 *     front of a front cut in two (site_front.h), for a mix finish whose mixes lie in the front's scratch; the FFN
 *     front for the FFN site's mix finish and its router select, one launch; the attention's output projection
 *     (attention_output.h) for its site's post, its output the projection's; the join or the shared join
 *     (expert_fused.h) for the FFN site's post, its output the shared expert's and its sums the row sums it stores (a
 *     join that stores none is refused: a capture reads them). card_descriptions, when given, names each logical
 *     card's selected arithmetic class, in the same order as cards; NULL means reference-order modules. Native
 *     modules are read too: reading a program does not establish its agreement with a reference capture.
 *   - Refused with STATUS_INVALID_ARGUMENT and one error line: a NULL argument, a layer_count of 0 or above
 *     TEST_PROGRAM_VIEW_MOST_LAYERS, a layer of layers at or above it, and a program that breaks a rule above; a
 *     launch whose record is not its kernel's size; STATUS_OUT_OF_MEMORY with one error line; the status of
 *     a prepare function. On failure view_out is untouched.
 */
#ifndef TEST_PROGRAM_VIEW_H
#define TEST_PROGRAM_VIEW_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/engram_rows.h"
#include "anemone/hyper_connection.h"
#include "anemone/model_seam.h"
#include "anemone/router.h"
#include "anemone/status.h"

#define TEST_PROGRAM_VIEW_MOST_LAYERS 64u
#define TEST_PROGRAM_VIEW_NONE UINT32_MAX

enum test_program_segment_kind { TEST_PROGRAM_SEGMENT_OTHER, TEST_PROGRAM_SEGMENT_EMBED, TEST_PROGRAM_SEGMENT_LAYER,
                                 TEST_PROGRAM_SEGMENT_HEAD };
enum test_program_site { TEST_PROGRAM_ATTENTION_SITE, TEST_PROGRAM_FFN_SITE, TEST_PROGRAM_SITES };

/* A layer of layers; of another, segment alone (or TEST_PROGRAM_VIEW_NONE where the program has none). */
struct test_program_layer {
	struct hyper_finish_arguments finish[TEST_PROGRAM_SITES];
	struct hyper_post_arguments post[TEST_PROGRAM_SITES];
	struct router_select_arguments router;
	uint32_t finish_launch[TEST_PROGRAM_SITES]; /* in its card's launches */
	uint32_t gate_launch; /* its n-gram gate's, or TEST_PROGRAM_VIEW_NONE */
	uint32_t first_segment; /* the first fragment; segment below is the last */
	uint32_t segment;
};

struct test_program_view {
    uint8_t *segment_kinds;                         /* the program's segment_count: enum test_program_segment_kind */
    struct test_program_layer *layers;              /* layer_count */
    struct hyper_embed_arguments embed; /* the embedding segment's record */
    struct hyper_finish_arguments
	    final; /* the head's final collapse's record, if there is a head */
    uint32_t embed_segment;
    uint32_t head_segment;                          /* or TEST_PROGRAM_VIEW_NONE */
};

MUST_CHECK result_status test_program_view_create(
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *card_descriptions,
	const struct execution_program *program, uint32_t layer_count,
	uint64_t layers, struct test_program_view **view_out);
void test_program_view_destroy(struct test_program_view *view);   /* accepts NULL */

#endif
