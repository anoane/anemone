/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_program_walk.c: the TP walk, session checkpoints, graph ranges, the TP and TP2 program prepares (split from
 * model_v41_program.c by whole items).
 */
#include "common/checked_size.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "ds41f_tp_layer.h"
#include "ds41f_tp_program.h"
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
#include "ds41f_draft_share_records.h"
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

#include "models/deepseek_v41_flash/builder/ds41f_builder.h"
#include "ds41f_tp_program_internal.h"

#define DRAFT_SHARD_FIXED_LAUNCHES 2u /* wait, cut */
#define DRAFT_SHARD_LAUNCHES(positions) \
	(DRAFT_SHARD_FIXED_LAUNCHES +   \
	 DS41F_DRAFT_POSITION_LAUNCHES * (positions))
/* graph_parts' boundaries: the first segment, the cuts between its parts, the front and the end (parts + 2) */
#define GRAPH_OUTER_BOUNDARIES 2u
/* a BATCH verifier's draft walk: so many launches beyond DRAFT_MOST_LAUNCHES (its verify included) */
#define BATCH_WALK_MORE_LAUNCHES 2u

/* Full text-only TP2 composition is separate from every default PP factory above. */
#define TP2_PROGRAM_CHECKPOINTS \
	(DS41F_WALK_LAYERS *    \
	 DS41F_TP4_RANKS) /* draft_layers: every rank's draft stages too */

MUST_CHECK static result_status
tp2_program_walk(struct ds41f_builder *build,
		 const struct ds41f_tp2_program_description *description,
		 struct ds41f_tp2_layer_plan *tensor,
		 struct step_companion_pair *pairs, uint32_t *pair_count)
{
	build->stage = 0;
	result_status status =
		ds41f_tp2_program_select(build, description, tensor, 0, 0);
	uint32_t first_launch = build->made->cards[0].launch_count;
	const bool replicated = ds41f_tp2_program_replicated(description);
	if (!replicated) {
		build->tensor_layer =
			NULL; /* embedding is a single unsharded lookup/expansion */
		const enum program_kind kind = build->made->program.kind;
		if (status == DS41F_OK && build->head_stage != NULL &&
		    build->card == description->canonical->result_card &&
		    (kind == PROGRAM_THIN || kind == PROGRAM_BATCH))
			status = ds41f_settings_push_launch(
				build); /* head_shards: every rank's settings copy for its samplers */
		status = DS41F_THEN(ds41f_step_launches(build, true));
		status = DS41F_THEN(ds41f_tp2_program_segment(
			build, SEGMENT_NO_LAYER, first_launch));
		for (uint32_t rank = 1;
		     rank < description->stages[0]->description.rank_count &&
		     status == DS41F_OK;
		     rank++)
			status = ds41f_tp2_program_streams(build, rank);
	}
	status = DS41F_THEN(
		ds41f_tp2_program_rotations(build, description, tensor, pairs,
					    pair_count, 0, 0, replicated));
	uint32_t source = 0, group = 0;
	const bool chunked = build->slab_count > 1u;
	/* the draft context's pieces: batch_context_slabs, a BATCH program's slabs (the thin steps' product forms) */
	const uint32_t context_rows =
		!chunked ? 0u :
		build->made->program.kind == PROGRAM_BATCH &&
				description
					->stages[description->canonical
							 ->result_card]
					->description.batch_context_slabs ?
			   LINEAR_MAXIMUM_ROWS :
			   LINEAR_WIDE_MOST_ROWS;
	const bool front_only = build->made->program.kind == PROGRAM_WIDE;
	for (uint32_t layer = 0; layer < DS41F_LAYERS && status == DS41F_OK;
	     layer++) {
		source = (DS41F_SOURCE_LAYERS >> layer & 1u) != 0 ? layer :
								    source;
		group = (DS41F_INDEX_LAYERS >> layer & 1u) != 0 ? layer : group;
		const uint32_t owner =
			ds41f_tp2_program_layer_card(description, layer);
		if (layer != 0 && owner != ds41f_tp2_program_layer_card(
						   description, layer - 1u)) {
			build->stage++;
			status = ds41f_tp2_program_rotations(build, description,
							     tensor, pairs,
							     pair_count, owner,
							     layer, false);
		}
		if (description->stages[owner]->description.rank_count > 1u) {
			if (layer == DS41F_DECODER_FIRST_LAYER)
				status =
					DS41F_THEN(ds41f_tp2_program_layer_pair(
						build, description, tensor,
						pairs, pair_count, layer,
						source, group, true));
			if (layer == DS41F_DECODER_FIRST_LAYER && front_only)
				return status;
			status = DS41F_THEN(ds41f_tp2_program_layer_pair(
				build, description, tensor, pairs, pair_count,
				layer, source, group, false));
		} else {
			status = DS41F_THEN(ds41f_tp2_program_select(
				build, description, tensor, owner, layer));
			first_launch = build->made->cards[owner].launch_count;
			status =
				description->canonical->limits.draft_positions !=
							0 &&
						layer >=
							DS41F_DRAFT_FIRST_TAP_LAYER ?
					DS41F_THEN(ds41f_draft_tap_launch(
						build, layer)) :
					status;
			const bool divided = layer == DS41F_DECODER_FIRST_LAYER;
			status = DS41F_THEN(ds41f_tp2_program_layer_steps(
				build, layer, source, group, divided));
			if (divided) {
				status = DS41F_THEN(ds41f_tp2_program_segment(
					build, layer, first_launch));
				status = DS41F_THEN(ds41f_tp2_program_copies(
					build, layer, DS41F_FRONT_COPIES));
				build->made->program.front_segment_count =
					build->made->program.segment_count;
				if (front_only)
					return status;
				first_launch =
					build->made->cards[owner].launch_count;
				status = DS41F_THEN(
					ds41f_tp2_program_layer_steps(
						build, layer, source, group,
						false));
			}
			status = chunked ?
					 DS41F_THEN(ds41f_wide_ffn_launches(
						 build, layer)) :
					 DS41F_THEN(ds41f_ffn_launches(
						 build, layer, &first_launch));
			status = DS41F_THEN(ds41f_tp2_program_segment(
				build, layer, first_launch));
			status = DS41F_THEN(ds41f_tp2_program_copies(
				build, layer,
				divided ? DS41F_LATER_COPIES :
					  DS41F_EVERY_COPY));
			status = DS41F_THEN(ds41f_tp2_program_boundary(
				build, description, layer));
		}
	}
	if (status == DS41F_OK && ds41f_tp2_program_exl3_all(description)) {
		/* The last companion is rank3. Head and unsharded native draft always read rank0's completed rows. */
		status = ds41f_tp2_program_select(
			build, description, tensor,
			description->canonical->result_card,
			DS41F_DRAFT_FIRST_LAYER);
	}
	if (status == DS41F_OK &&
	    description->canonical->limits.draft_positions != 0) {
		first_launch = build->made->cards[build->card].launch_count;
		status = ds41f_draft_context_launches(build, context_rows);
		status = DS41F_THEN(ds41f_tp2_program_segment(
			build, SEGMENT_NO_LAYER, first_launch));
	}
	for (uint32_t rank = 0;
	     status == DS41F_OK &&
	     description->canonical->limits.draft_positions != 0 &&
	     rank < description->canonical->card_count;
	     rank++) {
		/* draft_layers: every other rank's own draft context (its rank of the draft reads its own window) */
		if (rank == description->canonical->result_card ||
		    !description->stages[rank]->description.draft_layers)
			continue;
		status = ds41f_tp2_program_select(build, description, tensor,
						  rank,
						  DS41F_DRAFT_FIRST_LAYER);
		first_launch = build->made->cards[rank].launch_count;
		status = DS41F_THEN(
			ds41f_draft_context_launches(build, context_rows));
		status = DS41F_THEN(ds41f_tp2_program_segment(
			build, SEGMENT_NO_LAYER, first_launch));
	}
	const uint32_t result = description->canonical->result_card;
	if (build->made->program.kind == PROGRAM_BATCH) {
		/* BATCH: each sampling card's heads slab by slab, the result card's last, each slab's own result
		 * area */
		for (uint32_t step = 0;
		     status == DS41F_OK &&
		     step < description->canonical->card_count;
		     step++) {
			const uint32_t card =
				step + 1u == description->canonical->card_count ?
					result :
				step < result ? step :
						step + 1u;
			if (card != result &&
			    !description->stages[card]->description.head_shards)
				continue;
			status = ds41f_tp2_program_select(
				build, description, tensor, card,
				DS41F_DRAFT_FIRST_LAYER);
			first_launch = build->made->cards[card].launch_count;
			if (description->stages[card]
				    ->description.batch_wide_head)
				status = ds41f_batch_wide_head_launches(
					build, description->stages[card]);
			for (uint32_t slab = 0;
			     slab < build->slab_count && status == DS41F_OK &&
			     !description->stages[card]
				      ->description.batch_wide_head;
			     slab++) {
				ds41f_batch_slab_select(build, slab);
				build->arrays[DS41F_ARRAY_RESULT] =
					(struct linear_range){
						description->stages[card]
								->batch_results
								.byte_offset +
							(uint64_t)slab *
								PROGRAM_BATCH_SLAB_RESULT_BYTES,
						PROGRAM_BATCH_SLAB_RESULT_BYTES
					};
				build->sampler_call = slab;
				status = ds41f_head_launches(build);
			}
			build->sampler_call = 0;
			ds41f_chunk_select(build);
			const struct program_limits *const own =
				&description->canonical->limits;
			for (uint32_t verifier = 0;
			     status == DS41F_OK && card != result &&
			     own->draft_positions != 0 &&
			     description->stages[card]
				     ->description.draft_shards &&
			     !description->stages[card]
				      ->description.draft_layers &&
			     verifier <
				     ds41f_batch_verifiers(description, own,
							   build->chunk_rows);
			     verifier++)
				status = ds41f_draft_shard_rank_launches(
					build, verifier,
					build->slab_count +
						verifier *
							own->draft_positions);
			status = DS41F_THEN(ds41f_tp2_program_segment(
				build, SEGMENT_NO_LAYER, first_launch));
		}
		/* then each verifier's draft walk on the result card (ds41f_dspark_records.h's round table, slab
		 * results),
	     * each a
         * thin program's walk: the card's kernels selected as a thin program's (the draft layers' site fronts) */
		const struct program_limits *const limits =
			&description->canonical->limits;
		build->batch_stage = description->stages[result];
		build->batch_slabs = build->slab_count;
		build->slab_count = 1;
		/* batch_draft_wide 2: the wide forms for a batched draft of more than 8 rows, where its layers can take
         * them (TP draft layers on the BF16 ring, no exact dense shards:
         * batch_draft_rows_walk's once) */
		const struct ds41f_tp2_stage_description *const draft_stage =
			&build->batch_stage->description;
		build->draft_wide_kernels =
			draft_stage->batch_draft_rows &&
			draft_stage->draft_layers &&
			draft_stage->wire_format == ORDERED_PUSH_BF16 &&
			!draft_stage->exact_dense &&
			draft_stage->batch_draft_wide >=
				DS41F_DRAFT_WIDE_PRODUCTS &&
			ds41f_batch_verifiers(description, limits,
					      build->chunk_rows) *
					limits->draft_positions >
				LINEAR_MAXIMUM_ROWS;
		if (status == DS41F_OK && limits->draft_positions != 0)
			status = ds41f_tp2_program_select(
				build, description, tensor, result,
				DS41F_DRAFT_FIRST_LAYER);
		if (status == DS41F_OK && limits->draft_positions != 0 &&
		    build->batch_stage->description.batch_draft_rows) {
			const uint32_t verifiers = ds41f_batch_verifiers(
				description, limits, build->chunk_rows);
			status = ds41f_batch_draft_rows_walk(build, verifiers);
			for (uint32_t rank = 0;
			     status == DS41F_OK &&
			     build->batch_stage->description.draft_layers &&
			     rank < description->canonical->card_count;
			     rank++) {
				/* draft_layers: every other rank's batched draft behind the result card's, its own
				 * areas */
				if (rank == result)
					continue;
				status = ds41f_tp2_program_select(
					build, description, tensor, rank,
					DS41F_DRAFT_FIRST_LAYER);
				build->batch_stage = description->stages[rank];
				status = DS41F_THEN(ds41f_batch_draft_rows_walk(
					build, verifiers));
			}
			build->slab_count = build->batch_slabs;
			build->draft_wide_kernels = false;
			return status;
		}
		for (uint32_t verifier = 0;
		     status == DS41F_OK && limits->draft_positions != 0 &&
		     verifier < ds41f_batch_verifiers(description, limits,
						      build->chunk_rows);
		     verifier++) {
			build->batch_draft = true;
			build->batch_verifier = verifier;
			ds41f_chunk_select(build);
			status = ds41f_draft_walk(build);
		}
		build->batch_draft = false;
		build->slab_count = build->batch_slabs;
		build->draft_wide_kernels = false;
		return status;
	}
	if (chunked)
		return status;
	for (uint32_t rank = 0;
	     status == DS41F_OK && rank < description->canonical->card_count;
	     rank++) {
		/* head_shards: the other ranks' heads and sampler shards, before the result card's (head segments
		 * only) */
		if (rank == result ||
		    !description->stages[rank]->description.head_shards)
			continue;
		status = ds41f_tp2_program_select(build, description, tensor,
						  rank,
						  DS41F_DRAFT_FIRST_LAYER);
		first_launch = build->made->cards[rank].launch_count;
		status = DS41F_THEN(ds41f_head_launches(build));
		status = DS41F_THEN(ds41f_tp2_program_segment(
			build, SEGMENT_NO_LAYER, first_launch));
	}
	if (status == DS41F_OK && build->card != result)
		status = ds41f_tp2_program_select(build, description, tensor,
						  result,
						  DS41F_DRAFT_FIRST_LAYER);
	first_launch = build->made->cards[build->card].launch_count;
	status = DS41F_THEN(ds41f_head_launches(build));
	status = DS41F_THEN(ds41f_tp2_program_segment(build, SEGMENT_NO_LAYER,
						      first_launch));
	if (description->canonical->limits.draft_positions == 0)
		return status;
	if (!description->stages[result]->description.draft_layers)
		return DS41F_THEN(ds41f_draft_walk(build));
	for (uint32_t rank = 0;
	     status == DS41F_OK && rank < description->canonical->card_count;
	     rank++) {
		/* draft_layers: the draft tail a segment a rank, the result card's first (its head's visit goes on) */
		const uint32_t card = rank == 0	     ? result :
				      rank <= result ? rank - 1u :
						       rank;
		status = ds41f_tp2_program_select(build, description, tensor,
						  card,
						  DS41F_DRAFT_FIRST_LAYER);
		status = DS41F_THEN(ds41f_draft_walk(build));
	}
	return status;
}

static uint32_t
tp2_program_checkpoints(const struct ds41f_tp2_program_description *description,
			struct window_checkpoint *checkpoints)
{
	const struct ds41f_plan *const plan = description->canonical;
	uint32_t count = 0;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		for (uint32_t card = 0; card < plan->card_count; card++) {
			if ((ds41f_walk_layers(&plan->cards[card]) >> layer &
			     1u) == 0)
				continue;
			if (description->stages[card]
				    ->description.attention_owner &&
			    ds41f_tp4_attention_owner(layer) != card) {
				if (plan->cards[card].ring_offset[layer] !=
					    PROGRAM_NOWHERE ||
				    plan->cards[card].checkpoint_offset[layer] !=
					    PROGRAM_NOWHERE)
					return UINT32_MAX;
				continue;
			}
			if (count >= TP2_PROGRAM_CHECKPOINTS ||
			    plan->cards[card].ring_offset[layer] ==
				    PROGRAM_NOWHERE ||
			    plan->cards[card].checkpoint_offset[layer] ==
				    PROGRAM_NOWHERE)
				return UINT32_MAX;
			checkpoints[count++] = (struct window_checkpoint){
				plan->cards[card].ring_offset[layer],
				plan->cards[card].checkpoint_offset[layer],
				DS41F_RING_SESSION_BYTES,
				DS41F_RING_SLOTS,
				DS41F_STATE_ROW_BYTES,
				DS41F_WINDOW - 1u,
				card
			};
		}
	}
	return count;
}

/* The four CMPs compute their own persistent source/index state. Their only outgoing hand-offs target the PRO,
 * whose work follows the complete prefix. Merge rank-local launch ranges up to the prefill-cone boundary,
 * then merge the remaining prefix. Embedding distribution and the native PRO/draft boundaries stay explicit. */
MUST_CHECK static result_status
tp4_graph_ranges_merge(struct ds41f_made_program *made,
		       struct step_companion_pair *pairs, uint32_t *pair_count,
		       uint32_t graph_parts)
{
	struct execution_program *const program = &made->program;
	if (*pair_count == 0 || program->segment_count >= DS41F_MOST_SEGMENTS)
		return STATUS_INVALID_ARGUMENT;
	struct segment_record merged[DS41F_MOST_SEGMENTS];
	uint32_t mapping[DS41F_MOST_SEGMENTS], count = 0, groups = 0;
	const uint32_t first = pairs[0].first_segment;
	const uint32_t end = pairs[*pair_count - 1u].second_segment + 1u;
	const uint32_t front = program->front_segment_count;
	if (first >= front || front > end)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t segment = 0; segment < first; segment++) {
		mapping[segment] = count;
		merged[count++] = made->segments[segment];
	}
	/* graph_parts (--tp_graph_parts): the cone's groups (a decode step's work; the same in every program, which the
     * driver holds equal) cut at graph_parts - 1 group starts spread evenly, so that a card's step begins after a
     * smaller graph's launch; the part behind the cone stays one */
	uint32_t boundaries[GRAPH_OUTER_BOUNDARIES + DS41F_TP_MOST_GRAPH_PARTS],
		boundary_count = 0;
	uint32_t starts[DS41F_MOST_SEGMENTS], start_count = 0;
	for (uint32_t index = 0; index < *pair_count; index++)
		if (pairs[index].first_segment > first &&
		    pairs[index].first_segment < front)
			starts[start_count++] = pairs[index].first_segment;
	const uint32_t parts = graph_parts < 1u ? 1u :
			       graph_parts > start_count + 1u ?
						  start_count + 1u :
						  graph_parts;
	boundaries[boundary_count++] = first;
	for (uint32_t cut = 1;
	     cut < parts && parts <= DS41F_TP_MOST_GRAPH_PARTS; cut++)
		boundaries[boundary_count++] =
			starts[cut * (start_count + 1u) / parts - 1u];
	boundaries[boundary_count++] = front;
	if (end > front)
		boundaries[boundary_count++] = end;
	for (uint32_t part = 0; part + 1u < boundary_count; part++) {
		if (boundaries[part] == boundaries[part + 1u])
			continue;
		const uint32_t group_first = count;
		for (uint32_t card = 0; card < DS41F_TP4_RANKS; card++) {
			struct segment_record combined = { 0 };
			bool found = false;
			for (uint32_t segment = boundaries[part];
			     segment < boundaries[part + 1u]; segment++) {
				const struct segment_record *const original =
					&made->segments[segment];
				if (original->card != card)
					continue;
				if (original->pause != 0 ||
				    (found &&
				     (original->stage != combined.stage ||
				      original->first_launch !=
					      combined.first_launch +
						      combined.launch_count)))
					return STATUS_INVALID_ARGUMENT;
				if (!found) {
					combined = *original;
					combined.layer = SEGMENT_NO_LAYER;
					found = true;
				} else {
					combined.launch_count +=
						original->launch_count;
				}
				mapping[segment] = count;
			}
			if (!found)
				return STATUS_INVALID_ARGUMENT;
			merged[count++] = combined;
		}
		pairs[groups++] =
			(struct step_companion_pair){ group_first, count - 1u };
		if (boundaries[part + 1u] == front)
			program->front_segment_count = count;
	}
	for (uint32_t segment = end; segment < program->segment_count;
	     segment++) {
		mapping[segment] = count;
		merged[count++] = made->segments[segment];
	}
	mapping[program->segment_count] = count;
	for (uint32_t index = 0; index < program->hand_off_count; index++) {
		struct hand_off_record *const transfer =
			&made->hand_offs[index];
		if (transfer->after_segment >= program->segment_count)
			return STATUS_INVALID_ARGUMENT;
		/* Moving a transfer across its consumer would change the program. Group-internal transfers are
		 * refused. */
		if (transfer->after_segment >= first &&
		    transfer->after_segment < end &&
		    transfer->to_card < DS41F_TP2_PRO_CARD)
			return STATUS_UNSUPPORTED;
		transfer->after_segment = mapping[transfer->after_segment];
	}
	program->draft_segment_first = mapping[program->draft_segment_first];
	memcpy(made->segments, merged, (size_t)count * sizeof(merged[0]));
	program->segment_count = count;
	*pair_count = groups;
	return DS41F_OK;
}

result_status ds41f_tp_program_prepare(
	const struct ds41f_tp2_program_description *description,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *card_descriptions,
	enum program_kind kind, uint32_t row_count,
	struct execution_program **program_out,
	struct step_companion_program *companions_out)
{
	if (program_out == NULL || companions_out == NULL ||
	    !ds41f_tp2_program_is_valid(description, backend, cards,
					card_descriptions, kind, row_count))
		return STATUS_INVALID_ARGUMENT;
	const struct ds41f_plan *const canonical = description->canonical;
	const bool batch = kind == PROGRAM_BATCH;
	/* a BATCH program of 8 rows or fewer walks as a thin one: no views (step_validation.c) */
	const bool chunked = kind != PROGRAM_THIN &&
			     !(batch && row_count <= LINEAR_MAXIMUM_ROWS);
	const bool exl3_all = ds41f_tp2_program_exl3_all(description);
	const uint64_t chunk_layers =
		kind == PROGRAM_FULL || batch ?
			UINT64_MAX :
			(1ull << (DS41F_DECODER_FIRST_LAYER + 1u)) - 1u;
	const uint32_t slabs = kind == PROGRAM_THIN ?
				       1u :
				       (row_count + LINEAR_MAXIMUM_ROWS - 1u) /
					       LINEAR_MAXIMUM_ROWS;
	const uint32_t context_pieces =
		!chunked ? 1u :
		batch && description->stages[canonical->result_card]
					->description.batch_context_slabs ?
			   slabs :
			   (row_count + LINEAR_WIDE_MOST_ROWS - 1u) /
				   LINEAR_WIDE_MOST_ROWS;
	uint32_t launch_capacity[DS41F_TP2_PROGRAM_CARDS], launch_total = 0;
	for (uint32_t card = 0; card < canonical->card_count; card++) {
		const uint32_t layers = (uint32_t)__builtin_popcountll(
			canonical->cards[card].layers);
		launch_capacity[card] =
			layers *
				(exl3_all || card < DS41F_TP2_PRO_CARD ?
					 DS41F_TP2_LAYER_LAUNCHES :
					 DS41F_LAYER_MOST_LAUNCHES) *
				slabs +
			DS41F_CARD_MOST_LAUNCHES;
		for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
			if ((canonical->cards[card].layers >> layer & 1u) !=
				    0 &&
			    canonical->exl3_bits[layer] != 0)
				launch_capacity[card] +=
					DS41F_EXL3_ATTENTION_EXTRA_LAUNCHES *
					slabs;
		}
		if (card == canonical->result_card &&
		    canonical->limits.draft_positions != 0)
			launch_capacity[card] +=
				DS41F_DRAFT_TAPS +
				context_pieces * DS41F_CONTEXT_PIECE_LAUNCHES +
				DS41F_DRAFT_STAGES +
				DS41F_DRAFT_MOST_LAUNCHES(
					canonical->limits.draft_positions);
		if (batch) /* a BATCH program: the head of each slab on a card that samples; the verifiers' draft walks
			    * (with
                    * batch_draft_rows: a slab of draft rows as many launches as a walk at most, plus a verify a
                    * walk) */
			launch_capacity[card] +=
				slabs * DS41F_CARD_MOST_LAUNCHES +
				(card == canonical->result_card &&
						 canonical->limits.draft_positions !=
							 0 ?
					 ds41f_batch_verifiers(
						 description,
						 &canonical->limits,
						 row_count) *
							 (DS41F_DRAFT_MOST_LAUNCHES(
								  canonical
									  ->limits
									  .draft_positions) +
							  BATCH_WALK_MORE_LAUNCHES) +
						 DS41F_BATCH_DRAFT_SLABS *
							 DS41F_DRAFT_MOST_LAUNCHES(
								 canonical
									 ->limits
									 .draft_positions) +
						 1u :
					 0u);
		if (description->stages[card]->description.head_shards &&
		    card == canonical->result_card)
			launch_capacity[card] +=
				1u; /* head_shards: the settings push */
		if (description->stages[card]->description.draft_layers &&
		    canonical->limits.draft_positions != 0) {
			/* draft_layers: every rank's taps, context and draft walk with its stages' tensor-parallel
			 * launches (a
             * BATCH program: the batched draft's slabs each, and its verifies) */
			const uint32_t positions =
				canonical->limits.draft_positions;
			const uint32_t walks = batch ? DS41F_BATCH_DRAFT_SLABS :
						       1u;
			launch_capacity[card] +=
				DS41F_DRAFT_TAPS +
				context_pieces * DS41F_CONTEXT_PIECE_LAUNCHES +
				DS41F_DRAFT_STAGES +
				walks * (DS41F_DRAFT_MOST_LAUNCHES(positions) +
					 DS41F_DRAFT_STAGES *
						 DS41F_TP2_LAYER_LAUNCHES) +
				(batch ? ds41f_batch_verifiers(
						 description,
						 &canonical->limits,
						 row_count) *
							 (DS41F_DRAFT_MOST_LAUNCHES(
								  positions) +
							  1u) +
						 1u :
					 0u);
		}
		/* draft_shards: the result card's share; another rank's wait, cut and each position's Markov and
		 * sampler */
		if (description->stages[card]
			    ->description
			    .draft_shards) /* a BATCH program: each verifier's */
			launch_capacity[card] +=
				(batch ? ds41f_batch_verifiers(
						 description,
						 &canonical->limits,
						 row_count) :
					 1u) *
				(card == canonical->result_card ?
					 1u :
					 DRAFT_SHARD_LAUNCHES(
						 canonical->limits
							 .draft_positions));
		launch_total += launch_capacity[card];
	}
	_Static_assert(sizeof(struct kernel_launch) %
				       _Alignof(struct window_checkpoint) ==
			       0,
		       "checkpoint tail follows aligned launch storage");
	const size_t launch_end = size_add(
		sizeof(struct ds41f_made_program),
		size_multiply(launch_total, sizeof(struct kernel_launch)));
	size_t made_bytes = size_add(
		launch_end, size_multiply((size_t)TP2_PROGRAM_CHECKPOINTS,
					  sizeof(struct window_checkpoint)));
	made_bytes = size_add(
		made_bytes, size_multiply(DS41F_TP2_PROGRAM_PAIRS,
					  sizeof(struct step_companion_pair)));
	if (made_bytes == SIZE_MAX) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct ds41f_builder *const build = calloc(1, sizeof(*build));
	struct ds41f_tp2_layer_plan *const tensor = calloc(1, sizeof(*tensor));
	struct ds41f_made_program *const made = calloc(1, made_bytes);
	if (build == NULL || tensor == NULL || made == NULL) {
		free(made);
		free(tensor);
		free(build);
		return STATUS_OUT_OF_MEMORY;
	}
	void *const checkpoint_storage = (unsigned char *)made + launch_end;
	struct window_checkpoint *const checkpoints = checkpoint_storage;
	struct step_companion_pair *const pairs =
		(struct step_companion_pair *)(checkpoints +
					       (size_t)TP2_PROGRAM_CHECKPOINTS);
	/* split_overlap: two half views behind the slabs' where a chunked program's rows split on slabs */
	const uint32_t half_rows = row_count /
				   (DS41F_CHUNK_HALVES * LINEAR_MAXIMUM_ROWS) *
				   LINEAR_MAXIMUM_ROWS;
	const bool halves = (kind == PROGRAM_WIDE || kind == PROGRAM_FULL) &&
			    description->stages[0]->description.split_overlap &&
			    half_rows >= DS41F_SPAN_ROWS &&
			    row_count - half_rows <= LINEAR_WIDE_MOST_ROWS;
	*build = (struct ds41f_builder){
		.plan = canonical,
		.backend = backend,
		.cards = cards,
		.descriptions = card_descriptions,
		.made = made,
		.rows = row_count,
		.chunk_rows = row_count,
		.program_rows = row_count,
		.slab_count = slabs,
		.half_view_first = halves ? slabs + 1u : 0u,
		.half_rows = halves ? half_rows : 0u,
		.tensor_slices = description->slice_count,
		.head_stream_kernel = BACKEND_NO_KERNEL
	};
	made->program.kind = kind;
	for (uint32_t card = 0, base = 0; card < canonical->card_count;
	     card++) {
		const struct ds41f_card *const planned =
			&canonical->cards[card];
		const uint64_t *const arrays = planned->array_offset;
		const bool has_views = chunked &&
				       (card == canonical->embedding_card ||
					(planned->layers & chunk_layers) != 0);
		build->launches[card] = made->launches + base;
		build->launch_capacity[card] = launch_capacity[card];
		base += launch_capacity[card];
		for (uint32_t slab = 0; slab < slabs && has_views; slab++) {
			made->views[card][slab] = (struct step_view){
				arrays[DS41F_ARRAY_VIEWS] +
					(uint64_t)slab * DS41F_VIEW_BYTES,
				slab * LINEAR_MAXIMUM_ROWS, LINEAR_MAXIMUM_ROWS
			};
		}
		/* split_overlap: the two halves' views behind the slabs' (their blocks the halves' rows of each
		 * step) */
		for (uint32_t half = 0;
		     half < DS41F_HALF_VIEWS && has_views && halves; half++)
			made->views[card][slabs + half] = (struct step_view){
				arrays[DS41F_ARRAY_VIEWS] +
					(uint64_t)slabs * DS41F_VIEW_BYTES +
					half * (STEP_BLOCK_BYTES_VALUE +
						(uint64_t)half_rows *
							STEP_ROW_BYTES_VALUE),
				half * half_rows,
				half == 0 ? half_rows : row_count - half_rows
			};
		made->cards[card] = (struct program_card_record){
			build->launches[card], planned->arena_byte_count,
			arrays[DS41F_ARRAY_STEP], arrays[DS41F_ARRAY_SESSIONS],
			/* a BATCH model: BATCH and chunked programs' result area the slab results (one driver's cards
			 * agree;
             * the chunked ones sample nothing) */
			description->stages[0]->description.batch &&
					kind != PROGRAM_THIN &&
					card == canonical->result_card ?
				description->stages[card]
					->batch_results.byte_offset :
				arrays[DS41F_ARRAY_RESULT],
			arrays[DS41F_ARRAY_LOGITS], arrays[DS41F_ARRAY_STREAMS],
			arrays[DS41F_ARRAY_CARRIED_PRE],
			arrays[DS41F_ARRAY_ENGRAM_ROWS], planned->engram_tables,
			0, has_views ? made->views[card] : NULL,
			has_views ? slabs + (halves ? DS41F_HALF_VIEWS : 0u) :
				    0,
			canonical->limits.draft_positions == 0 ||
					card != canonical->result_card ?
				PROGRAM_NOWHERE :
			batch	 ? description->stages[card]
					   ->batch_rounds.byte_offset :
			!chunked ? arrays[DS41F_ARRAY_ROUND] :
				   PROGRAM_NOWHERE
		};
	}
	uint32_t pair_count = 0;
	result_status status = tp2_program_walk(build, description, tensor,
						pairs, &pair_count);
	if (status == DS41F_OK &&
	    made->program.segment_count >= DS41F_MOST_SEGMENTS)
		status =
			STATUS_INVALID_ARGUMENT; /* segment_add may have dropped one: no program */
	if (status != DS41F_OK)
		LOG_ERROR(
			"tensor program: walk failed at card %u stage %u after %u segments: %u",
			build->card, build->stage, made->program.segment_count,
			(uint32_t)status);
	if (status == DS41F_OK) {
		const uint32_t draft_first =
			(kind == PROGRAM_THIN || batch) &&
					canonical->limits.draft_positions != 0 ?
				made->program.draft_segment_first :
				made->program.segment_count;
		const uint32_t checkpoints_count =
			tp2_program_checkpoints(description, checkpoints);
		uint32_t expected_checkpoints = 0, expected_groups = 0;
		for (uint32_t card = 0; card < canonical->card_count; card++) {
			const struct ds41f_tp2_stage_description *const stage =
				&description->stages[card]->description;
			for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS;
			     layer++) {
				if ((ds41f_walk_layers(
					     &canonical->cards[card]) >>
					     layer &
				     1u) != 0 &&
				    (!stage->attention_owner ||
				     ds41f_tp4_attention_owner(layer) == card))
					expected_checkpoints++;
			}
			if (stage->rank == 0 && stage->rank_count > 1u) {
				const uint32_t last =
					kind == PROGRAM_WIDE ?
						DS41F_DECODER_FIRST_LAYER +
							(stage->attention_owner ?
								 0u :
								 1u) :
						DS41F_LAYERS;
				if (stage->first_target_layer < last) {
					const uint32_t left =
						last -
						stage->first_target_layer;
					expected_groups +=
						(stage->target_layer_count <
								 left ?
							 stage->target_layer_count :
							 left) +
						1u;
					expected_groups +=
						!stage->attention_owner &&
						kind != PROGRAM_WIDE &&
						(canonical->cards[card].layers >>
							 DS41F_DECODER_FIRST_LAYER &
						 1u) != 0;
				}
			}
		}
		const uint32_t layout =
			exl3_all ? DS41F_TP_LAYOUT_QUAD_ALL :
			description->stages[0]->description.rank_count ==
					DS41F_TP4_RANKS ?
				   DS41F_TP_LAYOUT_QUAD :
			description->stages[DS41F_TP2_PAIR_B_CARD]
						->description.rank_count == 1u ?
				   DS41F_TP_LAYOUT_HYBRID :
				   DS41F_TP_LAYOUT_PAIRS;
		status = checkpoints_count == expected_checkpoints &&
					 pair_count == expected_groups ?
				 DS41F_OK :
				 STATUS_INVALID_ARGUMENT;
		if (status != DS41F_OK)
			LOG_ERROR(
				"tensor program: %u/%u checkpoints and %u/%u groups",
				checkpoints_count, expected_checkpoints,
				pair_count, expected_groups);
		if (status == DS41F_OK) {
			made->program = (struct execution_program){
				.cards = made->cards,
				.segments = made->segments,
				.hand_offs = made->hand_offs,
				.limits = canonical->limits,
				.card_count = canonical->card_count,
				.segment_count = made->program.segment_count,
				.hand_off_count = made->program.hand_off_count,
				.row_count = row_count,
				.result_card = canonical->result_card,
				.expert_pauses = made->expert_pauses,
				.host_expert_offsets =
					made->host_expert_offsets,
				.expert_pause_count = 0,
				.host_expert_count = 0,
				.kind = kind,
				.lease_count = 0,
				.session_ranges = made->session_ranges,
				.session_range_count = ds41f_session_ranges_make(
					canonical, made->session_ranges),
				.front_segment_count =
					made->program.front_segment_count,
				.cone_rows = DS41F_CONE_ROWS,
				.checkpoint_count = checkpoints_count,
				.checkpoints = checkpoints,
				.leases = made->leases,
				.draft_segment_first = draft_first
			};
			const bool whole_graph =
				(layout == DS41F_TP_LAYOUT_QUAD || exl3_all) &&
				!description->stages[0]
					 ->description.attention_owner &&
				(description->stages[0]
					 ->description.exact_output_shards ||
				 description->stages[0]
						 ->description.wire_format ==
					 ORDERED_PUSH_BF16);
			if (whole_graph)
				status = tp4_graph_ranges_merge(
					made, pairs, &pair_count,
					description->stages[0]
						->description.graph_parts);
			if (status != DS41F_OK) {
				free(made);
				free(tensor);
				free(build);
				return status;
			}
			*companions_out = (struct step_companion_program){
				&made->program, pairs, pair_count, 0,
				(description->stages[0]
						 ->description
						 .exact_output_shards ?
					 DS41F_TP4_EXACT_RECIPE :
					 DS41F_TP2_PROGRAM_RECIPE) ^
					description->slice_count ^
					(description->stages[0]->description
									 .wire_format !=
								 0 &&
							 (layout ==
								  DS41F_TP_LAYOUT_QUAD ||
							  exl3_all) ?
						 DS41F_TP4_RING_IDENTITY :
						 0) ^
					((uint64_t)description->stages[0]
						 ->description.wire_format
					 << DS41F_TP2_WIRE_IDENTITY_SHIFT) ^
					((uint64_t)layout
					 << DS41F_TP_LAYOUT_IDENTITY_SHIFT) ^
					(canonical->limits.row_capacity >
							 LINEAR_MAXIMUM_ROWS ?
						 DS41F_TP_WIDE_IDENTITY :
						 0) ^
					(layout == DS41F_TP_LAYOUT_QUAD ?
						 DS41F_TP4_ALIGNED_IDENTITY :
						 0) ^
					(description->stages[0]
							 ->description
							 .attention_owner ?
						 DS41F_TP4_ATTENTION_OWNER_IDENTITY :
						 0) ^
					(exl3_all ?
						 DS41F_TP4_EXL3_IDENTITY ^
							 ((uint64_t)description
								  ->stages[0]
								  ->description
								  .exl3_product_order
							  << DS41F_TP4_EXL3_ORDER_SHIFT) :
						 0)
			};
			*program_out = &made->program;
		}
	}
	if (status != DS41F_OK)
		free(made);
	free(tensor);
	free(build);
	return status;
}

result_status ds41f_tp2_program_prepare(
	const struct ds41f_tp2_program_description *description,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *card_descriptions,
	uint32_t row_count, struct execution_program **program_out,
	struct step_companion_program *companions_out)
{
	return ds41f_tp_program_prepare(description, backend, cards,
					card_descriptions, PROGRAM_THIN,
					row_count, program_out, companions_out);
}
