/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * teacher_forced_synthetic.h: the synthetic model teacher_forced_test holds the harness (teacher_forced.h) to: a
 * program of four layers on two cards of the backend in host memory whose kernels are stand-ins, what those compute
 * on host arrays, and the streams of the reference it is teacher-forced on. Test support: no part of the engine.
 *
 * Contract:
 *   - test_synthetic_cards_set resets the backend (no card open, no staging alive) and opens
 *     TEST_SYNTHETIC_CARDS cards of class SM_80, each with the kernels the program launches (the same handles on
 *     both), every layer's expert table and router bias, a pool of rubbish (a hand-off at a position that writes no
 *     entry would carry it to card 1), copies that stay in flight until a wait asks, and the stand-ins; false when
 *     any of it was refused. calls counts the runs of each card's segment of its own.
 *   - test_synthetic_program_build: card 0 the embedding and layers 0 and 1 (layer TEST_SYNTHETIC_GATE_LAYER with
 *     its two n-gram launches before its sites and a pool writer between them); card 1 a segment of its own, layers
 *     2 and 3 and the head; behind layer 1 the streams, the carried pre and, at every second position, a pool entry go
 *     to card 1. Both sites of a layer use the same arrays and the FFN site's post writes its output where wo_b's
 *     lies, so that a record that comes back at another moment holds another part's values. With is_fused a layer's
 *     sites are the fused launches that stand for them (site_front.h, attention_output.h, expert_fused.h): the site
 *     front, the output projection, the FFN front and the join, which stores its row sums (the mixes in the fronts'
 *     scratch); their stand-ins compute what the others' do, so a run gives the same records.
 *   - The stand-ins compute what test_synthetic_layer_compute and test_synthetic_logit_of compute: the n-gram gate
 *     (it adds to every stream value; the rows-to-values launch has none), a site's finish, the posts (which write
 *     their output, and with the combine the routed sums, first), a router over the site's normed input with a bias,
 *     the head's logits of its token and of the final collapse's output, and its result.
 *   - test_synthetic_capture_row: the streams and pre of the reference leaving a layer at a position;
 *     test_synthetic_embedding_row: the embedding segment's output. test_synthetic_mirror_holds: card 1 holds the
 *     pool entries of positions 1 and 3 in its mirror and zeros behind them (false also when a copy was refused).
 */
#ifndef TEST_TEACHER_FORCED_SYNTHETIC_H
#define TEST_TEACHER_FORCED_SYNTHETIC_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/attention_output.h"
#include "anemone/backend_seam.h"
#include "anemone/engram_rows.h"
#include "anemone/expert_fused.h"
#include "anemone/hyper_connection.h"
#include "anemone/model_layout.h"
#include "anemone/model_seam.h"
#include "anemone/router.h"
#include "anemone/site_front.h"

#define TEST_SYNTHETIC_CARDS 2u
#define TEST_SYNTHETIC_LAYERS 4u
#define TEST_SYNTHETIC_POSITIONS 5u
#define TEST_SYNTHETIC_GATE_LAYER 1u                /* also the layer with the pool writer */
#define TEST_SYNTHETIC_HEAD_INPUT_STREAM 1u         /* the stream the final collapse's stand-in passes on */
#define TEST_SYNTHETIC_SEGMENTS 7u
#define TEST_SYNTHETIC_HAND_OFFS 3u
#define TEST_SYNTHETIC_MOST_LAUNCHES 16u            /* of a card */
#define TEST_SYNTHETIC_SITES 2u                     /* attention, FFN */
#define TEST_SYNTHETIC_ROUTER_FROM_THE_END 2u       /* in a layer's segment: the router, then the FFN site's post */

struct test_synthetic_row {
	uint16_t streams[MODEL_STREAMS][MODEL_HIDDEN];
	float pre[MODEL_STREAMS];
};

/* What a layer leaves, and what its n-gram output and a capture point hold of it; a site's arrays [site]. */
struct test_synthetic_layer {
    struct test_synthetic_row output;
    uint16_t ngram[MODEL_STREAMS][MODEL_HIDDEN]; /* the streams after the gate */
    uint16_t ffn_streams[MODEL_STREAMS][MODEL_HIDDEN];
    uint16_t hidden[TEST_SYNTHETIC_SITES][MODEL_HIDDEN], wo_b[MODEL_HIDDEN];
    uint16_t shared[MODEL_HIDDEN];
    float pre[TEST_SYNTHETIC_SITES][MODEL_STREAMS],
	    post[TEST_SYNTHETIC_SITES][MODEL_STREAMS];
    float mixes[TEST_SYNTHETIC_SITES][MODEL_MIXES];
    float comb[TEST_SYNTHETIC_SITES][MODEL_STREAMS * MODEL_STREAMS];
    float sums[MODEL_HIDDEN], affinities[MODEL_EXPERTS], biased[MODEL_EXPERTS];
    float weights[MODEL_EXPERTS_PER_ROW]; /* of picks */
    uint32_t picks[MODEL_EXPERTS_PER_ROW]; /* ascending */
    int64_t gate_ids[MODEL_EXPERTS_PER_ROW]; /* in the gate's order */
    float gate_weights[MODEL_EXPERTS_PER_ROW];
};

struct test_synthetic_kernels {
	struct hyper_kernels_record hyper;
	struct router_kernels_record router;
	struct engram_rows_kernels engram;
	struct site_front_kernels fronts;
	struct attention_output_kernels outputs;
	struct expert_fused_kernels joins;
	uint32_t fill;
	uint32_t sampler;
};

struct test_synthetic_calls {
    uint32_t others[TEST_SYNTHETIC_CARDS];          /* runs of the segment of its own of each card */
};

struct test_synthetic_program {
	struct kernel_launch launches[TEST_SYNTHETIC_CARDS]
				     [TEST_SYNTHETIC_MOST_LAUNCHES];
	struct program_card_record cards[TEST_SYNTHETIC_CARDS];
	struct segment_record segments[TEST_SYNTHETIC_SEGMENTS];
	struct hand_off_record hand_offs[TEST_SYNTHETIC_HAND_OFFS];
	struct execution_program program;
};

int32_t test_synthetic_token_at(uint32_t position);
void test_synthetic_capture_row(uint32_t layer, uint32_t position, struct test_synthetic_row *row);
void test_synthetic_embedding_row(uint32_t position, struct test_synthetic_row *row);
void test_synthetic_layer_compute(uint32_t layer, const struct test_synthetic_row *input,
                                  struct test_synthetic_layer *values);
float test_synthetic_logit_of(int32_t token, uint32_t vocabulary_index, const uint16_t *head_input);
bool test_synthetic_cards_set(struct device_card **cards,
			      struct test_synthetic_kernels *kernels,
			      struct test_synthetic_calls *calls);
void test_synthetic_program_build(struct test_synthetic_program *made, const struct test_synthetic_kernels *kernels,
                                  bool is_fused);
bool test_synthetic_mirror_holds(struct device_card *card);

#endif
