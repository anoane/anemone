/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * program_view.c: the reader of program_view.h. Test support: no part of the engine.
 *
 * Contract: program_view.h. The numbers of the program are checked before any launch is looked at; the kernels are
 * then asked of every open card, and each segment is read once, in the order of the program. The n-gram gate is
 * looked for in every segment: one outside a layer's would write streams that a test then puts its own over. A fused
 * launch is read as the launches it stands for: its record turned into theirs, field by field.
 */
#include "program_view.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/attention_output.h"
#include "anemone/expert_fused.h"
#include "anemone/hooks.h"
#include "anemone/official_reference.h"
#include "anemone/site_front.h"

#define NONE TEST_PROGRAM_VIEW_NONE
#define ATTENTION TEST_PROGRAM_ATTENTION_SITE
#define FFN TEST_PROGRAM_FFN_SITE
#define SITES TEST_PROGRAM_SITES
#define NO_FRONT SITE_FRONT_KINDS
/* The default for a caller that does not supply its logical cards' selected classes. */
#define REFERENCE_ORDER_CLASS ARITHMETIC_CLASS_SM_80

struct reading {
	const struct execution_program *program;
	struct test_program_view *view;
	struct hyper_kernels_record hyper[BACKEND_MAXIMUM_CARDS];
	struct router_kernels_record router[BACKEND_MAXIMUM_CARDS];
	struct engram_rows_kernels engram[BACKEND_MAXIMUM_CARDS];
	struct site_front_kernels fronts[BACKEND_MAXIMUM_CARDS];
	struct attention_output_kernels outputs[BACKEND_MAXIMUM_CARDS];
	struct expert_fused_kernels joins[BACKEND_MAXIMUM_CARDS];
	uint32_t layer_count;
	uint64_t layers;
};

/* Whether the program names only what exists, on cards that are open. */
static bool program_is_sound(const struct execution_program *program,
			     struct device_card *const *cards)
{
	bool sound =
		program->card_count - 1u < BACKEND_MAXIMUM_CARDS &&
		program->cards != NULL && program->segments != NULL &&
		program->segment_count != 0 &&
		(program->hand_offs != NULL || program->hand_off_count == 0) &&
		program->result_card < program->card_count;
	for (uint32_t index = 0; sound && index < program->card_count;
	     index++) {
		sound = (program->cards[index].launches != NULL ||
			 program->cards[index].launch_count == 0) &&
			(program->cards[index].step_offset == PROGRAM_NOWHERE ||
			 cards[index] != NULL);
	}
    for (uint32_t index = 0; sound && index < program->segment_count; index++) {
	    const struct segment_record *const segment =
		    &program->segments[index];
	    sound = segment->card < program->card_count &&
		    cards[segment->card] != NULL &&
		    segment->launch_count != 0 &&
		    (uint64_t)segment->first_launch + segment->launch_count <=
			    program->cards[segment->card].launch_count;
    }
    for (uint32_t index = 0; sound && index < program->hand_off_count; index++) {
	    const struct hand_off_record *const hand_off =
		    &program->hand_offs[index];
	    sound = hand_off->from_card < program->card_count &&
		    hand_off->to_card < program->card_count &&
		    cards[hand_off->from_card] != NULL &&
		    cards[hand_off->to_card] != NULL &&
		    hand_off->after_segment < program->segment_count;
    }
    return sound;
}

MUST_CHECK static result_status record_of(const struct kernel_launch *launch,
					  void *record, uint32_t byte_count)
{
    if (launch->argument_byte_count != byte_count) {
	    LOG_ERROR("program view: a launch's record holds %" PRIu32
		      " bytes, its kernel's %" PRIu32,
		      launch->argument_byte_count, byte_count);
	    return STATUS_INVALID_ARGUMENT;
    }
    memcpy(record, launch->argument, byte_count);
    return STATUS_OK;
}

/* The kind of site front a kernel is (enum site_front_kind), NO_FRONT for none. */
static uint32_t front_kind_of(const struct site_front_kernels *fronts,
			      uint32_t kernel)
{
	for (uint32_t instance = 0; instance < SITE_FRONT_INSTANCES;
	     instance++) {
		if (fronts->fronts[instance] == kernel) {
			return instance / SITE_FRONT_ROW_FORMS;
		}
	}
    return NO_FRONT;
}

static bool is_one_of(const uint32_t *kernels, uint32_t count, uint32_t kernel)
{
    for (uint32_t index = 0; index < count; index++) {
        if (kernels[index] == kernel) {
            return true;
        }
    }
    return false;
}

/* A site front's record as the mix finish's it stands for (its mixes in the front's scratch, site_front_records.h),
 * and of the FFN front the select's. */
MUST_CHECK static result_status
front_read(const struct kernel_launch *launch,
	   struct hyper_finish_arguments *finish_out,
	   struct router_select_arguments *router_out)
{
	struct site_front_arguments front;
	const result_status status = record_of(launch, &front, sizeof(front));
	if (status == STATUS_OK) {
		*finish_out = (struct hyper_finish_arguments){
			front.streams_offset,
			front.pre_offset,
			front.scratch_offset +
				offsetof(struct site_front_scratch, mixes),
			front.scale_offset,
			front.base_offset,
			front.post_offset,
			front.comb_offset,
			front.norm_weight_offset,
			front.hidden_offset
		};
		*router_out = front.select;
	}
    return status;
}

/* A fused post's record (the attention's projection, the FFN site's join or shared join) as the post's it stands for;
 * a join that stores no row sums is refused: a capture reads them. */
MUST_CHECK static result_status
fused_post_read(const struct reading *reading, uint32_t card,
		const struct kernel_launch *launch,
		struct hyper_post_arguments *post_out)
{
	const struct expert_fused_kernels *const joins = &reading->joins[card];
	result_status status = STATUS_OK;
	if (is_one_of(reading->outputs[card].projection, LINEAR_SLICE_INSTANCES,
		      launch->kernel)) {
		struct attention_projection_arguments projection;
		status = record_of(launch, &projection, sizeof(projection));
		if (status == STATUS_OK) {
			*post_out = (struct hyper_post_arguments){
				projection.streams_offset,
				projection.post_offset,
				projection.comb_offset,
				projection.output_offset,
				0,
				0,
				0
			};
		}
	} else if (launch->kernel == joins->join) {
		struct expert_join_arguments join;
		status = record_of(launch, &join, sizeof(join));
		if (status == STATUS_OK) {
			*post_out = (struct hyper_post_arguments){
				join.streams_offset,
				join.post_offset,
				join.comb_offset,
				join.shared_output_offset,
				join.row_sums_offset,
				1,
				0
			};
		}
	} else {
		struct expert_shared_join_arguments shared_join;
		status = record_of(launch, &shared_join, sizeof(shared_join));
		if (status == STATUS_OK) {
			*post_out = (struct hyper_post_arguments){
				shared_join.streams_offset,
				shared_join.post_offset,
				shared_join.comb_offset,
				shared_join.output_offset,
				shared_join.row_sums_offset,
				1,
				0
			};
		}
	}
	if (status == STATUS_OK && post_out->is_combined != 0 &&
	    post_out->sums_offset == EXPERT_JOIN_NO_ROW_SUMS) {
		LOG_ERROR(
			"program view: a join that stores no row sums, which a capture reads");
		status = STATUS_INVALID_ARGUMENT;
	}
    return status;
}

/* A layer's segment: its n-gram gate, then its two sites around its router, in the order of the contract, on the
 * card's arrays. */
MUST_CHECK static result_status layer_read(const struct reading *reading,
					   const struct segment_record *segment)
{
	const struct program_card_record *const card =
		&reading->program->cards[segment->card];
	const struct hyper_kernels_record *const hyper =
		&reading->hyper[segment->card];
	const struct engram_rows_kernels *const engram =
		&reading->engram[segment->card];
	const uint32_t select = reading->router[segment->card].select;
	struct test_program_layer *const layer =
		&reading->view->layers[segment->layer];
	uint32_t finishes = 0, posts = 0, routers = 0, router_launch = NONE,
		 post_launch[SITES] = { NONE, NONE };
	uint32_t gates = 0;
	struct engram_gate_arguments gate = { 0 };
	result_status status = STATUS_OK;
	const struct expert_fused_kernels *const joins =
		&reading->joins[segment->card];
	for (uint32_t index = segment->first_launch;
	     index < segment->first_launch + segment->launch_count; index++) {
		const struct kernel_launch *const launch =
			&card->launches[index];
		/* a site front stands for a mix finish (the input front, the first of a cut front, for none), the FFN
         * front for the select too; a fused projection or join for a post */
		const uint32_t front = front_kind_of(
			&reading->fronts[segment->card], launch->kernel);
		const bool is_finish =
			launch->kernel == hyper->mix_finish ||
			(front != NO_FRONT && front != SITE_FRONT_KIND_INPUT);
		const bool is_router = launch->kernel == select ||
				       front == SITE_FRONT_KIND_FFN;
		const bool is_fused_post =
			is_one_of(reading->outputs[segment->card].projection,
				  LINEAR_SLICE_INSTANCES, launch->kernel) ||
			launch->kernel == joins->join ||
			is_one_of(joins->shared_join, LINEAR_SLICE_INSTANCES,
				  launch->kernel);
		const bool is_post = launch->kernel == hyper->post ||
				     is_fused_post;
		struct router_select_arguments
			site_select; /* a site front's, which holds no select */
		if (launch->kernel == engram->gate && gates == 0) {
			layer->gate_launch = index;
			status = status != STATUS_OK ?
					 status :
					 record_of(launch, &gate, sizeof(gate));
		}
		gates += launch->kernel == engram->gate ? 1u : 0u;
		if (is_finish && finishes < SITES) {
			struct router_select_arguments *const front_select =
				front == SITE_FRONT_KIND_FFN ? &layer->router :
							       &site_select;
			layer->finish_launch[finishes] = index;
			status = status != STATUS_OK ?
					 status :
				 front != NO_FRONT ?
					 front_read(launch,
						    &layer->finish[finishes],
						    front_select) :
					 record_of(launch,
						   &layer->finish[finishes],
						   sizeof(layer->finish[0]));
		}
		if (is_post && posts < SITES) {
			post_launch[posts] = index;
			status = status != STATUS_OK ?
					 status :
				 is_fused_post ?
					 fused_post_read(reading, segment->card,
							 launch,
							 &layer->post[posts]) :
					 record_of(launch, &layer->post[posts],
						   sizeof(layer->post[0]));
		}
		if (is_router && routers == 0) {
			router_launch = index;
			status =
				status != STATUS_OK ||
						front ==
							SITE_FRONT_KIND_FFN /* read as a finish */
					?
					status :
					record_of(launch, &layer->router,
						  sizeof(layer->router));
		}
		finishes += is_finish ? 1u : 0u;
		posts += is_post ? 1u : 0u;
		routers += is_router ? 1u : 0u;
		status = launch->kernel == hyper->embed ||
					 launch->kernel ==
						 hyper->final_collapse ?
				 STATUS_INVALID_ARGUMENT :
				 status;
	}
	bool in_place =
		status == STATUS_OK && finishes == SITES && posts == SITES &&
		routers == 1 &&
		layer->finish_launch[ATTENTION] < post_launch[ATTENTION] &&
		post_launch[ATTENTION] < layer->finish_launch[FFN] &&
		layer->finish_launch[FFN] <= router_launch &&
		router_launch < post_launch[FFN] &&
		layer->post[ATTENTION].is_combined == 0 &&
		layer->post[FFN].is_combined == 1 &&
		layer->router.row_count == 1 && gates <= 1 &&
		(gates == 0 ||
		 (layer->gate_launch < layer->finish_launch[ATTENTION] &&
		  gate.streams_offset == card->streams_offset));
	for (uint32_t site = 0; in_place && site < SITES; site++) {
		in_place = layer->finish[site].streams_offset ==
				   card->streams_offset &&
			   layer->post[site].streams_offset ==
				   card->streams_offset &&
			   layer->finish[site].pre_offset ==
				   card->carried_pre_offset;
	}
    if (!in_place) {
	    LOG_ERROR(
		    "program view: the segment of layer %" PRIu32
		    " is not its n-gram gate, then two sites "
		    "around a router of one row, on its card's streams and carried pre (%" PRIu32
		    " n-gram "
		    "gates, %" PRIu32 " mix finishes, %" PRIu32
		    " posts, %" PRIu32 " routers)",
		    segment->layer, gates, finishes, posts, routers);
	    return STATUS_INVALID_ARGUMENT;
    }
    return status;
}

MUST_CHECK static result_status segment_read(const struct reading *reading,
					     uint32_t segment_index)
{
	const struct execution_program *const program = reading->program;
	const struct segment_record *const segment =
		&program->segments[segment_index];
	const struct program_card_record *const card =
		&program->cards[segment->card];
	const struct hyper_kernels_record *const hyper =
		&reading->hyper[segment->card];
	struct test_program_view *const view = reading->view;
	const struct engram_rows_kernels *const engram =
		&reading->engram[segment->card];
	uint32_t embeds = 0, finals = 0, gates = 0, embed_launch = 0,
		 final_launch = 0;
	for (uint32_t index = segment->first_launch;
	     index < segment->first_launch + segment->launch_count; index++) {
		const uint32_t kernel = card->launches[index].kernel;
		embed_launch = kernel == hyper->embed ? index : embed_launch;
		final_launch = kernel == hyper->final_collapse ? index :
								 final_launch;
		embeds += kernel == hyper->embed ? 1u : 0u;
		finals += kernel == hyper->final_collapse ? 1u : 0u;
		gates += kernel == engram->gate ? 1u : 0u;
	}
	if (segment->layer != SEGMENT_NO_LAYER) {
		/* the front's last segment, and the rest of its layer behind it on its card: one range of launches */
		const bool ends_front = segment_index + 1u ==
					program->front_segment_count;
		const struct segment_record *const front =
			segment_index != 0 ? segment - 1 : segment;
		const bool goes_on =
			segment_index != 0 &&
			segment_index == program->front_segment_count &&
			segment->layer < reading->layer_count &&
			view->layers[segment->layer].segment ==
				segment_index - 1u &&
			front->card == segment->card &&
			front->first_launch + front->launch_count ==
				segment->first_launch;
		if (segment->layer >= reading->layer_count ||
		    (view->layers[segment->layer].segment != NONE &&
		     !goes_on)) {
			LOG_ERROR("program view: segment %" PRIu32
				  " runs layer %" PRIu32
				  ", which the model has not or "
				  "another segment runs",
				  segment_index, segment->layer);
			return STATUS_INVALID_ARGUMENT;
		}
		view->segment_kinds[segment_index] = TEST_PROGRAM_SEGMENT_LAYER;
		view->layers[segment->layer].first_segment =
			goes_on ? view->layers[segment->layer].first_segment :
				  segment_index;
		view->layers[segment->layer].segment = segment_index;
		const struct segment_record whole = {
			segment->card,
			segment->stage,
			segment->layer,
			front->first_launch,
			front->launch_count + segment->launch_count,
			segment->pause
		};
		return (reading->layers >> segment->layer & 1u) == 0 ||
				       ends_front ?
			       STATUS_OK :
			       layer_read(reading, goes_on ? &whole : segment);
	}
    const bool is_embed = embeds == 1 && finals == 0 && view->embed_segment == NONE;
    const bool is_head = finals == 1 && embeds == 0 &&
			 view->head_segment == NONE &&
			 segment->card == program->result_card &&
			 card->logits_offset != PROGRAM_NOWHERE &&
			 card->result_offset != PROGRAM_NOWHERE;
    if ((embeds != 0 && !is_embed) || (finals != 0 && !is_head) || gates != 0) {
	    LOG_ERROR(
		    "program view: segment %" PRIu32
		    " is a second embedding or head, both, a head without the "
		    "result card's logits and result, or holds an n-gram gate outside a layer",
		    segment_index);
	    return STATUS_INVALID_ARGUMENT;
    }
    view->segment_kinds[segment_index] = is_embed ? TEST_PROGRAM_SEGMENT_EMBED
                                       : is_head ? TEST_PROGRAM_SEGMENT_HEAD : TEST_PROGRAM_SEGMENT_OTHER;
    view->embed_segment = is_embed ? segment_index : view->embed_segment;
    view->head_segment = is_head ? segment_index : view->head_segment;
    return is_embed ? record_of(&card->launches[embed_launch], &view->embed,
				sizeof(view->embed)) :
	   is_head  ? record_of(&card->launches[final_launch], &view->final,
				sizeof(view->final)) :
		      STATUS_OK;
}

result_status test_program_view_create(
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *card_descriptions,
	const struct execution_program *program, uint32_t layer_count,
	uint64_t layers, struct test_program_view **view_out)
{
    const bool given = backend != NULL && cards != NULL && program != NULL && view_out != NULL &&
                       layer_count - 1u < TEST_PROGRAM_VIEW_MOST_LAYERS &&
                       (layer_count == TEST_PROGRAM_VIEW_MOST_LAYERS || layers >> layer_count == 0);
    if (!given || !program_is_sound(program, cards)) {
	    LOG_ERROR(
		    "program view: a missing argument, layers beyond the model's, or a program that names a "
		    "card, launch, segment or hand-off that does not exist or a card that is not open");
	    return STATUS_INVALID_ARGUMENT;
    }
    struct reading reading = {.program = program, .layer_count = layer_count, .layers = layers};
    reading.view = calloc(1, sizeof(struct test_program_view));
    result_status status = reading.view == NULL ? STATUS_OUT_OF_MEMORY :
						  STATUS_OK;
    if (status == STATUS_OK) {
	    reading.view->segment_kinds =
		    calloc(program->segment_count, sizeof(uint8_t));
	    reading.view->layers =
		    calloc(layer_count, sizeof(struct test_program_layer));
	    reading.view->embed_segment = NONE;
	    reading.view->head_segment = NONE;
	    status = reading.view->segment_kinds == NULL ||
				     reading.view->layers == NULL ?
			     STATUS_OUT_OF_MEMORY :
			     status;
    }
    if (status == STATUS_OUT_OF_MEMORY) {
	    LOG_ERROR("program view: no memory for a view of %" PRIu32
		      " segments",
		      program->segment_count);
    }
    for (uint32_t layer = 0; status == STATUS_OK && layer < layer_count;
	 layer++) {
	    reading.view->layers[layer].first_segment = NONE;
	    reading.view->layers[layer].segment = NONE;
	    reading.view->layers[layer].gate_launch = NONE;
    }
    for (uint32_t card = 0; status == STATUS_OK && card < program->card_count;
	 card++) {
	    const uint32_t arithmetic_class =
		    card_descriptions == NULL ?
			    REFERENCE_ORDER_CLASS :
			    card_descriptions[card].arithmetic_class;
	    status = cards[card] == NULL ?
			     status :
			     hyper_kernels_prepare(backend, cards[card],
						   &reading.hyper[card]);
	    status = status != STATUS_OK || cards[card] == NULL ?
			     status :
			     router_kernels_prepare(backend, cards[card],
						    &reading.router[card]);
	    status = status != STATUS_OK || cards[card] == NULL ?
			     status :
			     engram_rows_kernels_prepare(backend, cards[card],
							 &reading.engram[card]);
	    status = status != STATUS_OK || cards[card] == NULL ?
			     status :
			     site_front_kernels_prepare(backend, cards[card],
							&reading.fronts[card]);
	    status = status != STATUS_OK || cards[card] == NULL ?
			     status :
			     attention_output_kernels_prepare(
				     backend, cards[card], arithmetic_class,
				     &reading.outputs[card]);
	    status = status != STATUS_OK || cards[card] == NULL ?
			     status :
			     expert_fused_kernels_prepare(backend, cards[card],
							  arithmetic_class,
							  &reading.joins[card]);
	    if (status == STATUS_OK && cards[card] != NULL &&
		program->limits.official_reference != 0) {
		    struct official_reference_kernels official;
		    status = official_reference_kernels_prepare(
			    backend, cards[card], arithmetic_class, &official);
		    if (status == STATUS_OK) {
			    reading.hyper[card].mix_finish = official.site;
			    reading.hyper[card].final_collapse =
				    official.final_kept;
			    reading.engram[card].gate = official.engram;
		    }
	    }
    }
    for (uint32_t segment = 0;
	 status == STATUS_OK && segment < program->segment_count; segment++) {
	    status = segment_read(&reading, segment);
    }
    const struct test_program_view *const view =
	    status == STATUS_OK ? reading.view : NULL;
    bool whole = view == NULL || view->embed_segment != NONE;
    for (uint32_t layer = 0; view != NULL && layer < layer_count; layer++) {
        whole = whole && ((layers >> layer & 1u) == 0 || view->layers[layer].segment != NONE);
    }
    if (view != NULL && !whole) {
	    LOG_ERROR(
		    "program view: the program has no embedding segment, or no segment of a layer asked for");
	    status = STATUS_INVALID_ARGUMENT;
    }
    if (status != STATUS_OK) {
	    test_program_view_destroy(reading.view);
	    return status;
    }
    *view_out = reading.view;
    return STATUS_OK;
}

void test_program_view_destroy(struct test_program_view *view)
{
    if (view != NULL) {
        free(view->segment_kinds);
        free(view->layers);
        free(view);
    }
}
