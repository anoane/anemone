/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_teacher_forced_synthetic.h: the synthetic model ds41f_teacher_forced_test holds the harness
 * (ds41f_teacher_forced.h) to: a program of four layers on two cards of the backend in host memory whose kernels are
 * stand-ins, what those compute on host arrays, and the streams of the reference it is teacher-forced on. Test support:
 * no part of the engine.
 *
 * Contract:
 *   - ds41f_test_synthetic_cards_set resets the backend (no card open, no staging alive) and opens
 *     DS41F_TEST_SYNTHETIC_CARDS cards of class SM_80, each with the kernels the program launches (the same handles on
 *     both), every layer's expert table and router bias, a pool of rubbish (a hand-off at a position that writes no
 *     entry would carry it to card 1), copies that stay in flight until a wait asks, and the stand-ins; false when
 *     any of it was refused. calls counts the runs of each card's segment of its own.
 *   - ds41f_test_synthetic_program_build: card 0 the embedding and layers 0 and 1 (layer
 *     DS41F_TEST_SYNTHETIC_GATE_LAYER with its two n-gram launches before its sites and a pool writer between them);
 *     card 1 a segment of its own, layers 2 and 3 and the head; behind layer 1 the streams, the carried pre and, at
 *     every second position, a pool entry go to card 1. Both sites of a layer use the same arrays and the FFN site's
 *     post writes its output where wo_b's lies, so that a record that comes back at another moment holds another part's
 *     values. With is_fused a layer's sites are the fused launches that stand for them (ds41f_site_front.h,
 *     ds41f_attention_output.h, expert_producer.h): the site front, the output projection, the FFN front and the join,
 *     which stores its row sums (the mixes in the fronts' scratch); their stand-ins compute what the others' do, so a
 *     run gives the same records.
 *   - The stand-ins compute what ds41f_test_synthetic_layer_compute and ds41f_test_synthetic_logit_of compute: the
 *     n-gram gate (it adds to every stream value; the rows-to-values launch has none), a site's finish, the posts
 *     (which write their output, and with the combine the routed sums, first), a router over the site's normed input
 *     with a bias, the head's logits of its token and of the final collapse's output, and its result.
 *   - ds41f_test_synthetic_capture_row: the streams and pre of the reference leaving a layer at a position;
 *     ds41f_test_synthetic_embedding_row: the embedding segment's output. ds41f_test_synthetic_mirror_holds: card 1
 *     holds the pool entries of positions 1 and 3 in its mirror and zeros behind them (false also when a copy was
 *     refused).
 */
#ifndef DS41F_TEACHER_FORCED_SYNTHETIC_H
#define DS41F_TEACHER_FORCED_SYNTHETIC_H

#include <stdbool.h>
#include <stdint.h>

#include "models/deepseek_v41_flash/attention/ds41f_attention_output.h"
#include "engine/backend/backend_seam.h"
#include "models/deepseek_v41_flash/engram/ds41f_engram_rows.h"
#include "engine/library/moe/expert_producer.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_hyper_connection.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "engine/program/model_seam.h"
#include "models/deepseek_v41_flash/moe/ds41f_router.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_site_front.h"

#define DS41F_TEST_SYNTHETIC_CARDS 2u
#define DS41F_TEST_SYNTHETIC_LAYERS 4u
#define DS41F_TEST_SYNTHETIC_POSITIONS 5u
#define DS41F_TEST_SYNTHETIC_GATE_LAYER 1u                /* also the layer with the pool writer */
#define DS41F_TEST_SYNTHETIC_HEAD_INPUT_STREAM 1u         /* the stream the final collapse's stand-in passes on */
#define DS41F_TEST_SYNTHETIC_SEGMENTS 7u
#define DS41F_TEST_SYNTHETIC_HAND_OFFS 3u
#define DS41F_TEST_SYNTHETIC_MOST_LAUNCHES 16u            /* of a card */
#define DS41F_TEST_SYNTHETIC_SITES 2u                     /* attention, FFN */
#define DS41F_TEST_SYNTHETIC_ROUTER_FROM_THE_END 2u       /* in a layer's segment: the router, then the FFN site's
                                                           * post */

struct ds41f_test_synthetic_row {
	uint16_t streams[DS41F_STREAMS][DS41F_HIDDEN];
	float pre[DS41F_STREAMS];
};

/* What a layer leaves, and what its n-gram output and a capture point hold of it; a site's arrays [site]. */
struct ds41f_test_synthetic_layer {
    struct ds41f_test_synthetic_row output;
    uint16_t ngram[DS41F_STREAMS][DS41F_HIDDEN]; /* the streams after the gate */
    uint16_t ffn_streams[DS41F_STREAMS][DS41F_HIDDEN];
    uint16_t hidden[DS41F_TEST_SYNTHETIC_SITES][DS41F_HIDDEN], wo_b[DS41F_HIDDEN];
    uint16_t shared[DS41F_HIDDEN];
    float pre[DS41F_TEST_SYNTHETIC_SITES][DS41F_STREAMS],
	    post[DS41F_TEST_SYNTHETIC_SITES][DS41F_STREAMS];
    float mixes[DS41F_TEST_SYNTHETIC_SITES][DS41F_MIXES];
    float comb[DS41F_TEST_SYNTHETIC_SITES][DS41F_STREAMS * DS41F_STREAMS];
    float sums[DS41F_HIDDEN], affinities[DS41F_EXPERTS], biased[DS41F_EXPERTS];
    float weights[DS41F_EXPERTS_PER_ROW]; /* of picks */
    uint32_t picks[DS41F_EXPERTS_PER_ROW]; /* ascending */
    int64_t gate_ids[DS41F_EXPERTS_PER_ROW]; /* in the gate's order */
    float gate_weights[DS41F_EXPERTS_PER_ROW];
};

struct ds41f_test_synthetic_kernels {
	struct ds41f_hyper_kernels_record hyper;
	struct router_kernels_record router;
	struct ds41f_engram_rows_kernels engram;
	struct ds41f_site_front_kernels fronts;
	struct ds41f_attention_output_kernels outputs;
	struct expert_fused_kernels joins;
	uint32_t fill;
	uint32_t sampler;
};

struct ds41f_test_synthetic_calls {
    uint32_t others[DS41F_TEST_SYNTHETIC_CARDS];          /* runs of the segment of its own of each card */
};

struct ds41f_test_synthetic_program {
	struct kernel_launch launches[DS41F_TEST_SYNTHETIC_CARDS]
				     [DS41F_TEST_SYNTHETIC_MOST_LAUNCHES];
	struct program_card_record cards[DS41F_TEST_SYNTHETIC_CARDS];
	struct segment_record segments[DS41F_TEST_SYNTHETIC_SEGMENTS];
	struct hand_off_record hand_offs[DS41F_TEST_SYNTHETIC_HAND_OFFS];
	struct execution_program program;
};

int32_t ds41f_test_synthetic_token_at(uint32_t position);
void ds41f_test_synthetic_capture_row(uint32_t layer, uint32_t position, struct ds41f_test_synthetic_row *row);
void ds41f_test_synthetic_embedding_row(uint32_t position, struct ds41f_test_synthetic_row *row);
void ds41f_test_synthetic_layer_compute(uint32_t layer, const struct ds41f_test_synthetic_row *input,
                                  struct ds41f_test_synthetic_layer *values);
float ds41f_test_synthetic_logit_of(int32_t token, uint32_t vocabulary_index, const uint16_t *head_input);
bool ds41f_test_synthetic_cards_set(struct device_card **cards,
			      struct ds41f_test_synthetic_kernels *kernels,
			      struct ds41f_test_synthetic_calls *calls);
void ds41f_test_synthetic_program_build(struct ds41f_test_synthetic_program *made,
                                        const struct ds41f_test_synthetic_kernels *kernels,
                                  bool is_fused);
bool ds41f_test_synthetic_mirror_holds(struct device_card *card);

#endif
