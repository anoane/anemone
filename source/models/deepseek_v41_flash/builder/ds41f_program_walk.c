/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_walk.c: stage hand-offs, visits, leases, program_prepare and its modes, the image program,
 * program_destroy (split from model_v41_program.c by whole items).
 */
#include "common/checked_size.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_layer.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_program.h"
#include "models/model_registry.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "models/deepseek_v41_flash/attention/ds41f_attention_core.h"
#include "models/deepseek_v41_flash/attention/ds41f_attention_output.h"
#include "models/deepseek_v41_flash/attention/ds41f_attention_state.h"
#include "engine/backend/fill_records.h"
#include "engine/library/bf16_linear/dense_linear.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "engine/library/bf16_linear/dense_stream_records.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_draft_share_records.h"
#include "models/deepseek_v41_flash/engram/ds41f_engram_rows.h"
#include "engine/backend/copy_records.h"
#include "engine/library/block_scaled_linear/expert_relaxed.h"
#include "engine/library/exl3/exl3.h"
#include "engine/library/exl3/exl3_tc.h"
#include "engine/library/exl3/exl3_tp.h"
#include "engine/library/moe/expert_producer.h"
#include "common/hooks.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_hyper_connection.h"
#include "engine/library/tensor_parallel/peer_row_copy.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"
#include "models/deepseek_v41_flash/vision/ds41f_image_routing.h"
#include "models/deepseek_v41_flash/moe/ds41f_router.h"
#include "engine/library/block_scaled_linear/relaxed_linear.h"
#include "engine/library/sampling/sampler.h"
#include "engine/library/sampling/sampler_shard_records.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_site_front.h"
#include "engine/library/tensor_parallel/tp_join.h"
#include "engine/library/moe/ep_filter.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/wire_codecs/quantized_wire_records.h"
#include "engine/library/tensor_parallel/tp_exchange.h"
#include "engine/library/collectives/ordered_push_parallel.h"
#include "engine/library/prepare_checks.h"

#include "ds41f_builder.h"
#include "ds41f_program_internal.h"

#define CHUNK_LAUNCHES \
	(1u + ROUTER_WIDE_LAUNCH_COUNT + EXPERT_RELAXED_WIDE_LAUNCH_COUNT)
#define VISION_FFN_EXTRA_LAUNCHES 2u
/* original_head_walk: the result card's final norm and sampler beside a broadcast a rank (a rank: dense,
 * gather) */
#define ORIGINAL_HEAD_FIXED_LAUNCHES 2u

/* Where the walk moves on to another card: the streams and the carried pre go with it (2.8), and the stage changes. */
static void stage_hand_off(struct ds41f_builder *b, uint32_t card)
{
	const uint64_t *const from = b->plan->cards[b->card].array_offset,
			      *const to = b->plan->cards[card].array_offset;
	ds41f_hand_off(b, card, from[DS41F_ARRAY_STREAMS],
		       to[DS41F_ARRAY_STREAMS], DS41F_STREAMS_ROW_BYTES, 0, 0);
	ds41f_hand_off(b, card, from[DS41F_ARRAY_CARRIED_PRE],
		       to[DS41F_ARRAY_CARRIED_PRE], DS41F_SITE_ROW_BYTES, 0, 0);
	b->stage++;
}

/* Whether the walk's unit on card begins a visit (model_seam.h) that holds a layer: the embedding where layer 0 goes on
 * on its card, or a layer on another card than the unit before. */
static bool visit_begins_layers(const struct ds41f_builder *b, uint32_t unit,
				uint32_t card)
{
	return unit == 0 ? b->plan->layer_card[0] == card :
			   unit <= DS41F_LAYERS && card != b->card;
}

/* The walk: the embedding (unit 0), layer l (unit l + 1), the head (the last unit). Where the card changes, the stage
 * does and the card is selected; a visit that holds a layer begins with its step segment. */
MUST_CHECK static result_status program_walk(struct ds41f_builder *b)
{
	const struct ds41f_plan *const plan = b->plan;
	uint32_t source = 0, group = 0;
	result_status status = DS41F_OK;
	for (uint32_t unit = 0; unit <= DS41F_LAYERS + 1u && status == DS41F_OK;
	     unit++) {
		const uint32_t layer = unit - 1u; /* the head's: DS41F_LAYERS */
		const uint32_t card = unit == 0 ? plan->embedding_card :
				      layer < DS41F_LAYERS ?
						  plan->layer_card[layer] :
						  plan->result_card;
		const bool begins_layers = visit_begins_layers(b, unit, card);
		if (unit != 0 && card != b->card) {
			stage_hand_off(b, card);
		}
		status = unit != 0 && card == b->card ?
				 DS41F_OK :
				 ds41f_card_select(b, card);
		uint32_t first_launch =
			status == DS41F_OK ?
				b->made->cards[b->card].launch_count :
				0;
		if (status == DS41F_OK &&
		    begins_layers) { /* the visit's step segment */
			status = ds41f_step_launches(b, false);
			ds41f_segment_add(b, SEGMENT_NO_LAYER, first_launch);
			first_launch = b->made->cards[b->card].launch_count;
		}
		if (status == DS41F_OK && layer < DS41F_LAYERS) {
			source = (DS41F_SOURCE_LAYERS >> layer & 1u) != 0 ?
					 layer :
					 source;
			group = (DS41F_INDEX_LAYERS >> layer & 1u) != 0 ?
					layer :
					group;
			const bool is_cut = layer == DS41F_DECODER_FIRST_LAYER;
			status =
				plan->limits.draft_positions != 0 &&
						layer >=
							DS41F_DRAFT_FIRST_TAP_LAYER ?
					ds41f_draft_tap_launch(b, layer) :
					status;
			status = DS41F_THEN(ds41f_layer_steps(
				b, layer, source, group,
				DS41F_ENGRAM_STEPS | DS41F_FRONT_STEPS |
					(is_cut ? 0 : DS41F_REST_STEPS)));
			if (status == DS41F_OK &&
			    is_cut) { /* the front ends: its copies follow */
				ds41f_front_end(b, layer, &first_launch);
				status = ds41f_layer_steps(b, layer, source,
							   group,
							   DS41F_REST_STEPS);
			}
			status = DS41F_THEN(
				ds41f_ffn_launches(b, layer, &first_launch));
			ds41f_segment_add(b, layer, first_launch);
			ds41f_copies_hand_off(
				b, layer,
				layer == DS41F_DECODER_FIRST_LAYER ?
					DS41F_LATER_COPIES :
					DS41F_EVERY_COPY);
			if (status == DS41F_OK &&
			    plan->limits.draft_positions != 0 &&
			    layer + 1u == DS41F_LAYERS) {
				first_launch =
					b->made->cards[b->card].launch_count;
				status = ds41f_draft_context_launches(b, 0);
				ds41f_segment_add(b, SEGMENT_NO_LAYER,
						  first_launch);
			}
		} else if (status ==
			   DS41F_OK) { /* the embedding (unit 0), the head */
			if (unit != 0 && plan->original_head.enabled) {
				status = ds41f_original_head_walk(b);
			} else {
				status = unit == 0 ?
						 ds41f_step_launches(b, true) :
						 ds41f_head_launches(b);
				ds41f_segment_add(b, SEGMENT_NO_LAYER,
						  first_launch);
			}
		}
	}
	return plan->limits.draft_positions != 0 ?
		       DS41F_THEN(ds41f_draft_walk(b)) :
		       status;
}

/* The steps of layer (enum layer_step) over each slab of the chunk in turn, through its view, the queries
 * (QUERIES)
 * over each span ahead of its first slab; with is_pieces over each piece of at most LINEAR_WIDE_MOST_ROWS rows
 * instead: the launches over the chunk. */
MUST_CHECK result_status ds41f_steps_over(struct ds41f_builder *b,
					  uint32_t layer, uint32_t source,
					  uint32_t group, uint32_t steps,
					  bool is_pieces)
{
	const uint32_t span_slabs = DS41F_SPAN_ROWS / LINEAR_MAXIMUM_ROWS;
	result_status status = DS41F_OK;
	for (uint32_t part = 0;
	     part < (is_pieces ? ds41f_piece_count_of(b) : b->slab_count) &&
	     status == DS41F_OK;
	     part++) {
		if (!is_pieces && (steps & DS41F_QUERIES) != 0 &&
		    part % span_slabs == 0) {
			ds41f_span_select(b, part / span_slabs);
			status = ds41f_layer_steps(b, layer, source, group,
						   DS41F_QUERIES);
		}
		if (is_pieces) {
			ds41f_piece_select(b, part);
		} else {
			ds41f_slab_select(b, part);
		}
		status = DS41F_THEN(ds41f_layer_steps(
			b, layer, source, group,
			is_pieces ? steps : steps & ~(uint32_t)DS41F_QUERIES));
	}
	return status;
}

/* A chunk's walk (model_seam.h): a wide program's (WIDE, agent H's B) the embedding, layers 0 to 19 and layer 20's
 * front; a full program's every layer, no head (its steps carry no row of logits: the engine leaves a prompt's last
 * rows to thin steps). A card's step segment holds each slab's rotation step; the embedding is one launch over the
 * chunk; a layer is one segment (its experts resident). Of a layer every launch that reads no state runs over the
 * chunk, a piece of up to 1,024 rows at a time (the wide launches of linear.h and dense_linear.h, the n-gram rows' and
 * the hyper sites' over their rows: the bytes of the slabs' thin launches), each behind every launch that writes what
 * it reads: the n-gram rows, engram.wkv and the gate; the mix and finish; the quantization, wq_a with wkv, a source's
 * compressor products, qr alone (ds41f_attention_state.h), wq_b, the indexer's and its weights; then each slab's state
 * writes, scan and core in slab order, so that a slab's attention finishes before the next slab's state writes, as
 * consecutive thin steps would, the queries of each span of 8 slabs (a span's arrays, model_v41.h) ahead of its first
 * slab; wo_a, wo_b and the post; the FFN (wide_ffn_launches). Layer 20's front (its
 * state writes) is a segment of its own behind which its copies go, as in a thin program; a wide walk ends there,
 * every segment a front step's; a full walk takes the rest of layer 20 (the front of all slabs has written every
 * window, pool entry and key the rest reads, and QUERY_NORMED is a chunk's), then its FFN and later copies. */
MUST_CHECK static result_status chunk_walk(struct ds41f_builder *b,
					   bool is_full)
{
	const struct ds41f_plan *const plan = b->plan;
	const uint32_t last_unit = is_full ? DS41F_LAYERS :
					     DS41F_DECODER_FIRST_LAYER + 1u;
	uint32_t source = 0, group = 0;
	result_status status = DS41F_OK;
	for (uint32_t unit = 0; unit <= last_unit && status == DS41F_OK;
	     unit++) {
		const uint32_t layer = unit - 1u;
		const uint32_t card = unit == 0 ? plan->embedding_card :
						  plan->layer_card[layer];
		const bool begins_layers = visit_begins_layers(b, unit, card);
		if (unit != 0 && card != b->card) {
			stage_hand_off(b, card);
		}
		status = unit != 0 && card == b->card ?
				 DS41F_OK :
				 ds41f_card_select(b, card);
		uint32_t first_launch =
			status == DS41F_OK ?
				b->made->cards[b->card].launch_count :
				0;
		if (status == DS41F_OK &&
		    begins_layers) { /* the visit's step segment */
			for (uint32_t slab = 0;
			     slab < b->slab_count && status == DS41F_OK;
			     slab++) {
				ds41f_slab_select(b, slab);
				status = ds41f_step_launches(b, false);
			}
			ds41f_segment_add(b, SEGMENT_NO_LAYER, first_launch);
			first_launch = b->made->cards[b->card].launch_count;
		}
		if (status == DS41F_OK && unit == 0) { /* the embedding */
			ds41f_chunk_select(b);
			status = ds41f_step_launches(b, true);
			ds41f_segment_add(b, SEGMENT_NO_LAYER, first_launch);
		}
		if (status != DS41F_OK || unit == 0) {
			continue;
		}
		source = (DS41F_SOURCE_LAYERS >> layer & 1u) != 0 ? layer :
								    source;
		group = (DS41F_INDEX_LAYERS >> layer & 1u) != 0 ? layer : group;
		const bool is_cut = layer == DS41F_DECODER_FIRST_LAYER;
		if (plan->limits.draft_positions != 0 &&
		    layer >= DS41F_DRAFT_FIRST_TAP_LAYER) {
			ds41f_chunk_select(b);
			status = ds41f_draft_tap_launch(b, layer);
		}
		/* over the chunk: ahead of the slabs' state writes, and the rest's products behind qr alone (layer 20's
         * behind its front, whose prepares wrote qr) */
		const uint32_t ahead = DS41F_ENGRAM_STEPS | DS41F_SITE_MIX |
				       DS41F_INPUT_PRODUCTS |
				       DS41F_POOL_PRODUCTS;
		const uint32_t queries = DS41F_QUERY_PRODUCTS |
					 DS41F_INDEX_WEIGHTS_PRODUCT;
		status = DS41F_THEN(ds41f_steps_over(
			b, layer, source, group,
			ahead | (is_cut ? 0 : DS41F_QUERY_NORM | queries),
			true));
		status = DS41F_THEN(ds41f_steps_over(
			b, layer, source, group,
			DS41F_STATE_WRITES |
				(is_cut ? 0 : DS41F_QUERIES | DS41F_CORE),
			false));
		if (status == DS41F_OK && is_cut) {
			ds41f_front_end(b, layer, &first_launch);
			status = is_full ? ds41f_steps_over(b, layer, source,
							    group, queries,
							    true) :
					   DS41F_OK;
			status = is_full ? DS41F_THEN(ds41f_steps_over(
						   b, layer, source, group,
						   DS41F_QUERIES | DS41F_CORE,
						   false)) :
					   status;
		}
		if (status == DS41F_OK && (!is_cut || is_full)) {
			status = ds41f_steps_over(
				b, layer, source, group,
				DS41F_OUTPUT_PRODUCTS | DS41F_SITE_POST, true);
			status = DS41F_THEN(ds41f_wide_ffn_launches(b, layer));
			ds41f_segment_add(b, layer, first_launch);
			ds41f_copies_hand_off(b, layer,
					      is_cut ? DS41F_LATER_COPIES :
						       DS41F_EVERY_COPY);
			if (status == DS41F_OK &&
			    plan->limits.draft_positions != 0 &&
			    layer + 1u == DS41F_LAYERS) {
				first_launch =
					b->made->cards[b->card].launch_count;
				status = ds41f_draft_context_launches(
					b, LINEAR_WIDE_MOST_ROWS);
				ds41f_segment_add(b, SEGMENT_NO_LAYER,
						  first_launch);
			}
		}
	}
	return status;
}

/* The program's leases (model_seam.h): on each card, of each index layer's selection and of the candidates, from the
 * visit that writes them (the index layer, layer 20) or takes them in (the card's first behind the writer, where
 * their hand-off goes up, whether or not it reads them) to the last visit of the card that reads them (the group's
 * layers, the candidate readers), where those two differ. */
static uint32_t leases_make(const struct ds41f_plan *plan,
			    struct ds41f_made_program *made)
{
	const struct execution_program *const program = &made->program;
	const struct segment_record *const segments = made->segments;
	uint32_t layer_segment[DS41F_LAYERS], visit_start[DS41F_MOST_SEGMENTS],
		count = 0;
	for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
		layer_segment[layer] = UINT32_MAX;
	}
	for (uint32_t segment = 0; segment < program->segment_count;
	     segment++) {
		visit_start[segment] =
			segment != 0 && segments[segment - 1u].card ==
						segments[segment].card ?
				visit_start[segment - 1u] :
				segment;
		if (segments[segment].layer < DS41F_LAYERS) {
			layer_segment[segments[segment].layer] =
				segment; /* its last: layer 20's rest */
		}
	}
	for (uint32_t card = 0; card < plan->card_count; card++) {
		for (uint32_t group = 0; group <= DS41F_LAYERS;
		     group++) { /* DS41F_LAYERS: candidates */
			const bool is_candidates = group == DS41F_LAYERS;
			const uint32_t writer =
				is_candidates ? DS41F_CANDIDATE_SOURCE : group;
			uint32_t last_read = UINT32_MAX,
				 taken = layer_segment[writer];
			for (uint32_t layer = writer;
			     layer < DS41F_LAYERS &&
			     (is_candidates ||
			      (DS41F_INDEX_LAYERS >> group & 1u) != 0);
			     layer++) {
				const bool reads =
					is_candidates ?
						(DS41F_CANDIDATE_READERS >>
							 layer &
						 1u) != 0 :
						layer == group ||
							(DS41F_INDEX_LAYERS >>
								 layer &
							 1u) == 0;
				if (!is_candidates && layer != group &&
				    !reads) {
					break; /* the next index layer: the group ends */
				}
				const bool is_read =
					reads &&
					plan->layer_card[layer] == card &&
					layer_segment[layer] != UINT32_MAX;
				last_read = is_read ? layer_segment[layer] :
						      last_read;
			}
			while (taken < program->segment_count &&
			       segments[taken].card != card) {
				taken++; /* the writer's, or the card's first segment behind */
			}
			if (last_read != UINT32_MAX && taken < last_read &&
			    visit_start[taken] != visit_start[last_read] &&
			    count < DS41F_MOST_LEASES) {
				made->leases[count++] =
					(struct visit_lease){ card, taken,
							      last_read };
			}
		}
	}
	return count;
}

MUST_CHECK static result_status
program_prepare_mode(const struct ds41f_plan *plan,
		     const struct backend_operations *backend,
		     struct device_card *const *cards,
		     const struct card_description_record *descriptions,
		     enum program_kind kind, uint32_t row_count,
		     bool vision_active, struct execution_program **program_out)
{
	if (plan == NULL || backend == NULL || cards == NULL ||
	    descriptions == NULL || program_out == NULL ||
	    plan->card_count - 1u >= BACKEND_MAXIMUM_CARDS ||
	    row_count - 1u >= plan->limits.row_capacity ||
	    (uint32_t)kind >= PROGRAM_KIND_COUNT ||
	    (vision_active && !plan->vision.enabled)) {
		LOG_ERROR(
			"program_prepare: a missing argument, a plan of no cards, %" PRIu32
			" rows (1 to the plan's) "
			"or kind %" PRIu32,
			row_count, (uint32_t)kind);
		return STATUS_INVALID_ARGUMENT;
	}
	if (kind != PROGRAM_THIN && kind != PROGRAM_WIDE &&
	    kind != PROGRAM_FULL) {
		LOG_ERROR(
			"program_prepare: kind %" PRIu32
			": this build makes thin, wide and full programs alone",
			(uint32_t)kind);
		return STATUS_UNSUPPORTED;
	}
	const bool is_wide = kind != PROGRAM_THIN,
		   is_full = kind == PROGRAM_FULL; /* chunked */
	if (is_wide ? row_count < STEP_WIDE_LEAST_ROWS :
		      row_count > ds41f_slab_rows(plan->limits.row_capacity)) {
		LOG_ERROR("program_prepare: a %s program of %" PRIu32
			  " rows: a thin one takes at most %" PRIu32
			  ", a chunked one at least %u",
			  is_wide ? "chunked" : "thin", row_count,
			  ds41f_slab_rows(plan->limits.row_capacity),
			  STEP_WIDE_LEAST_ROWS);
		return STATUS_INVALID_ARGUMENT;
	}
	if (!ds41f_plan_is_valid(plan)) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (plan->limits.official_reference != 0 &&
	    (kind != PROGRAM_THIN || row_count != 1 || vision_active))
		return STATUS_UNSUPPORTED;
	if (plan->limits.official_hc_rows != 0) {
		if (is_wide || vision_active ||
		    plan->limits.draft_positions != 0 ||
		    (row_count != 1 &&
		     row_count != DS41F_OFFICIAL_HC_ODD_PROMPT_ROWS &&
		     row_count != DS41F_STREAMS &&
		     row_count != LINEAR_MAXIMUM_ROWS))
			return STATUS_UNSUPPORTED;
		for (uint32_t card = 0; card < plan->card_count; card++) {
			if (ds41f_card_has_exl3(plan, card))
				return STATUS_UNSUPPORTED;
		}
	}
	/* Ordinary PP-wide EXL3 experts are not qualified by the exact TP slab composition. */
	for (uint32_t card = 0; card < plan->card_count && is_wide; card++) {
		if (ds41f_card_has_exl3(plan, card))
			return STATUS_UNSUPPORTED;
	}
	/* the layers a chunk's walk takes: a wide one layers 0 to 20 (20's front), a full one all */
	const uint64_t chunk_layers =
		is_full ? UINT64_MAX :
			  (1ull << (DS41F_DECODER_FIRST_LAYER + 1u)) - 1u;
	for (uint32_t layer = 0; layer < DS41F_LAYERS && is_wide; layer++) {
		if ((chunk_layers >> layer & 1u) != 0 &&
		    plan->host_expert_count[layer] != 0) {
			LOG_ERROR(
				"program_prepare: a chunked program needs the routed experts of its layers on their "
				"cards; layer %" PRIu32 " keeps %" PRIu32
				" in host memory",
				layer, plan->host_expert_count[layer]);
			return STATUS_UNSUPPORTED;
		}
	}
	const uint32_t slabs = is_wide ? (row_count + LINEAR_MAXIMUM_ROWS -
					  1u) / LINEAR_MAXIMUM_ROWS :
					 1u;
	uint32_t launch_capacity[BACKEND_MAXIMUM_CARDS] = { 0 },
		 launch_total = 0;
	for (uint32_t card = 0; card < plan->card_count;
	     card++) { /* each card's part of the launches */
		const uint32_t layers = (uint32_t)__builtin_popcountll(
			plan->cards[card].layers &
			(is_wide ? chunk_layers : UINT64_MAX));
		launch_capacity[card] =
			is_wide ? layers * (slabs * DS41F_LAYER_MOST_LAUNCHES +
					    CHUNK_LAUNCHES) +
					  slabs * DS41F_CARD_MOST_LAUNCHES :
				  layers * DS41F_LAYER_MOST_LAUNCHES +
					  DS41F_CARD_MOST_LAUNCHES;
		for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
			if ((plan->cards[card].layers >> layer & 1u) != 0 &&
			    plan->exl3_bits[layer] != 0)
				launch_capacity[card] +=
					DS41F_EXL3_ATTENTION_EXTRA_LAUNCHES *
					slabs;
		}
		if (vision_active) {
			launch_capacity[card] +=
				layers * VISION_FFN_EXTRA_LAUNCHES +
				(card == 0 ? slabs : 0u);
		}
		if (plan->limits.draft_positions != 0 &&
		    card == plan->result_card && (!is_wide || is_full)) {
			const uint32_t pieces =
				is_wide ? (row_count + LINEAR_WIDE_MOST_ROWS -
					   1u) / LINEAR_WIDE_MOST_ROWS :
					  1u;
			launch_capacity[card] +=
				DS41F_DRAFT_TAPS +
				pieces * DS41F_CONTEXT_PIECE_LAUNCHES +
				DS41F_DRAFT_STAGES +
				(is_wide ?
					 0 :
					 DS41F_DRAFT_MOST_LAUNCHES(
						 plan->limits.draft_positions));
		}
		if (plan->limits.official_reference != 0)
			launch_capacity[card] += layers * LINEAR_MAXIMUM_ROWS +
						 LINEAR_MAXIMUM_ROWS;
		if (!is_wide && plan->original_head.enabled)
			launch_capacity[card] +=
				DS41F_HEAD_RANKS + ORIGINAL_HEAD_FIXED_LAUNCHES;
		launch_total += launch_capacity[card];
	}
	struct ds41f_builder *const b = FAULT_POINT(model_v41_program_memory) ?
						NULL :
						calloc(1, sizeof(*b));
	struct ds41f_made_program *const made =
		b == NULL || FAULT_POINT(model_v41_program_memory) ?
			NULL :
			calloc(1, sizeof(*made) +
					  (size_t)launch_total *
						  sizeof(made->launches[0]));
	if (made == NULL) {
		LOG_ERROR("program_prepare: no memory for a program of %" PRIu32
			  " rows",
			  row_count);
		free(b);
		return STATUS_OUT_OF_MEMORY;
	}
	*b = (struct ds41f_builder){ .plan = plan,
				     .backend = backend,
				     .cards = cards,
				     .descriptions = descriptions,
				     .made = made,
				     .rows = row_count,
				     .card = plan->embedding_card,
				     .chunk_rows = row_count,
				     .program_rows = row_count,
				     .slab_count = slabs,
				     .vision_active = vision_active,
				     .head_stream_kernel = BACKEND_NO_KERNEL,
				     .head_mma_kernel = BACKEND_NO_KERNEL };
	for (uint32_t card = 0, base = 0; card < plan->card_count; card++) {
		const struct ds41f_card *const planned = &plan->cards[card];
		const uint64_t *const arrays = planned->array_offset,
				      nowhere = PROGRAM_NOWHERE;
		const bool has_views = is_wide &&
				       (card == plan->embedding_card ||
					(planned->layers & chunk_layers) != 0);
		b->launches[card] = made->launches + base;
		b->launch_capacity[card] = launch_capacity[card];
		base += launch_capacity[card];
		for (uint32_t slab = 0; slab < slabs && has_views; slab++) {
			made->views[card][slab] = (struct step_view){
				arrays[DS41F_ARRAY_VIEWS] +
					(uint64_t)slab * DS41F_VIEW_BYTES,
				slab * LINEAR_MAXIMUM_ROWS, LINEAR_MAXIMUM_ROWS
			};
		}
		made->cards[card] = (struct program_card_record){
			b->launches[card],
			card == 0 && plan->vision.enabled &&
					!plan->original_head.enabled ?
				plan->vision.text_arena_bytes :
				planned->arena_byte_count,
			arrays[DS41F_ARRAY_STEP],
			arrays[DS41F_ARRAY_SESSIONS],
			arrays[DS41F_ARRAY_RESULT],
			arrays[DS41F_ARRAY_LOGITS],
			planned->layers != 0 ? arrays[DS41F_ARRAY_STREAMS] :
					       nowhere,
			planned->layers != 0 ? arrays[DS41F_ARRAY_CARRIED_PRE] :
					       nowhere,
			arrays[DS41F_ARRAY_ENGRAM_ROWS],
			planned->engram_tables,
			0,
			has_views ? made->views[card] : NULL,
			has_views ? slabs : 0,
			!is_wide && plan->limits.draft_positions != 0 &&
					card == plan->result_card ?
				arrays[DS41F_ARRAY_ROUND] :
				nowhere
		};
	}
	const result_status status = is_wide ? chunk_walk(b, is_full) :
					       program_walk(b);
	if (status == DS41F_OK) {
		const uint32_t draft_first =
			!is_wide && plan->limits.draft_positions != 0 ?
				made->program.draft_segment_first :
				made->program.segment_count;
		made->program = (struct execution_program){
			.cards = made->cards,
			.segments = made->segments,
			.hand_offs = made->hand_offs,
			.limits = plan->limits,
			.card_count = plan->card_count,
			.segment_count = made->program.segment_count,
			.hand_off_count = made->program.hand_off_count,
			.row_count = row_count,
			.result_card = plan->result_card,
			.expert_pauses = made->expert_pauses,
			.host_expert_offsets = made->host_expert_offsets,
			.expert_pause_count = made->program.expert_pause_count,
			.host_expert_count = made->program.host_expert_count,
			.kind = kind,
			.lease_count = 0,
			.session_ranges = made->session_ranges,
			.session_range_count = ds41f_session_ranges_make(
				plan, made->session_ranges),
			.front_segment_count =
				made->program.front_segment_count,
			.cone_rows = DS41F_CONE_ROWS,
			.checkpoint_count = plan->limits.draft_positions != 0 ?
						    DS41F_WALK_LAYERS :
						    DS41F_LAYERS,
			.checkpoints = made->checkpoints,
			.leases = made->leases,
			.draft_segment_first = draft_first
		};
		ds41f_checkpoints_make(plan, made->checkpoints);
		made->program.lease_count = leases_make(plan, made);
		if (!is_wide && plan->original_head.enabled &&
		    plan->original_head.bank_count == 0) {
			uint32_t first = 0;
			while (first < made->program.segment_count &&
			       made->segments[first].card != plan->result_card)
				first++;
			ASSERT(first < made->program.segment_count &&
			       made->program.lease_count < DS41F_MOST_LEASES);
			made->leases[made->program.lease_count++] =
				(struct visit_lease){
					plan->result_card, first,
					made->program.segment_count - 1u
				};
		}
		*program_out = &made->program;
	} else {
		free(made);
	}
	free(b);
	return status;
}

MUST_CHECK result_status ds41f_program_prepare(
	const struct ds41f_plan *plan, const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *descriptions,
	enum program_kind kind, uint32_t row_count,
	struct execution_program **program_out)
{
	return program_prepare_mode(plan, backend, cards, descriptions, kind,
				    row_count, false, program_out);
}

result_status
ds41f_image_program_prepare(const struct ds41f_plan *plan,
			    const struct backend_operations *backend,
			    struct device_card *const *cards,
			    const struct card_description_record *descriptions,
			    enum program_kind kind, uint32_t row_count,
			    struct execution_program **program_out)
{
	return program_prepare_mode(plan, backend, cards, descriptions, kind,
				    row_count, true, program_out);
}

void ds41f_program_destroy(struct execution_program *program)
{
	free(program); /* the address of its struct made_program */
}
