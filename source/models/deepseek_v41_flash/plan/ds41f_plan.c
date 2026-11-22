/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_plan.c: describe, plan create and destroy, the cards' layout (split from model_v41.c by whole items).
 */
#include "model_v41.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_layer.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_stage.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/exl3/exl3.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark_records.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_draft_share_records.h"
#include "engine/library/sampling/sampler_shard_records.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine/backend/fill_records.h"
#include "engine/library/block_scaled_linear/expert_relaxed.h"
#include "common/hooks.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/library/block_scaled_linear/relaxed_linear.h"
#include "engine/library/prepare_checks.h"

#include "ds41f_plan.h"
#include "models/deepseek_v41_flash/tensors/ds41f_tensors.h"

#define ARRAY(array, row_bytes, fixed_bytes, holder, rows) \
	[DS41F_ARRAY_##array] = { (uint64_t)(row_bytes),   \
				  (uint64_t)(fixed_bytes), holder, rows },
static const struct array_rule {
	uint64_t row_bytes;
	uint64_t fixed_bytes;
	uint32_t holder;
	uint32_t rows; /* enum ds41f_rows */
} array_rules[DS41F_ARRAY_COUNT] = { DS41F_ARRAYS(ARRAY) };
#undef ARRAY
_Static_assert(DS41F_WIDEST_ROW >= DS41F_ENGRAM_WIDTH &&
		       DS41F_WIDEST_ROW >= DS41F_HIDDEN &&
		       DS41F_ENGRAM_WIDTH == DS41F_ENGRAM_ROWS_PER_TOKEN *
						     ENGRAM_VALUE_BYTES_VALUE,
	       "every quantized row of a layer fits the row arrays");

uint64_t ds41f_sum(uint64_t first, uint64_t second)
{
	return first > UINT64_MAX - second ? UINT64_MAX : first + second;
}

uint64_t ds41f_product(uint64_t first, uint64_t second)
{
	return second != 0 && first > UINT64_MAX / second ? UINT64_MAX :
							    first * second;
}

/* The latest layer of mask at or before layer, or DS41F_LAYERS for none. */
uint32_t ds41f_latest_of(uint64_t mask, uint32_t layer)
{
	uint32_t latest = DS41F_LAYERS;
	for (uint32_t earlier = 0; earlier <= layer; earlier++) {
		latest = (mask & DS41F_BIT(earlier)) != 0 ? earlier : latest;
	}
	return latest;
}

MUST_CHECK result_status ds41f_model_check(const struct pack_record *pack)
{
	const char *const name = pack_header_of(pack)->model_name;
	if (strncmp(name, DS41F_MODEL_NAME, sizeof(DS41F_MODEL_NAME)) != 0) {
		LOG_ERROR(
			"the pack names another model than " DS41F_MODEL_NAME);
		return STATUS_UNSUPPORTED;
	}
	return STATUS_OK;
}

MUST_CHECK result_status ds41f_limits_check(const struct program_limits *limits)
{
	const uint32_t context = limits->context_token_limit;
	const uint32_t width = (uint32_t)limits->state_width;
	const uint32_t draft =
		limits->draft_positions; /* the draft block's rows take a slab's arrays */
	const bool valid =
		context != 0 && context <= DS41F_CONTEXT_MOST &&
		context % DS41F_CONTEXT_GRANULE == 0 &&
		limits->row_capacity != 0 &&
		limits->row_capacity <= STEP_WIDE_MOST_ROWS &&
		limits->session_capacity != 0 &&
		(width == STATE_WIDTH_8 || width == STATE_WIDTH_16) &&
		(uint32_t)limits->linear_profile <= LINEAR_PROFILE_SPEED &&
		(uint32_t)limits->linear_order <= LINEAR_ORDER_REFERENCE &&
		(uint32_t)limits->cmp_order <= CMP_ORDER_RELAXED &&
		(limits->official_hc_rows == 0 ||
		 limits->official_hc_rows == 1 ||
		 limits->official_hc_rows ==
			 DS41F_OFFICIAL_HC_ODD_PROMPT_ROWS ||
		 limits->official_hc_rows == DS41F_STREAMS ||
		 limits->official_hc_rows == LINEAR_MAXIMUM_ROWS) &&
		(draft == 0 ||
		 (draft >= DS41F_DRAFT_BLOCK && draft <= STEP_DRAFT_MOST &&
		  draft <= ds41f_slab_rows(limits->row_capacity)));
	if (!valid || width != STATE_WIDTH_16) {
		LOG_ERROR("limits %s: %" PRIu32 " tokens, %" PRIu32
			  " rows, %" PRIu32 " sessions, %" PRIu32
			  " bits, profile %" PRIu32 ", order %" PRIu32
			  ", CMP order %" PRIu32 ", %" PRIu32
			  " draft positions",
			  valid ? "beyond this build" : "the model has none of",
			  context, limits->row_capacity,
			  limits->session_capacity, width,
			  (uint32_t)limits->linear_profile,
			  (uint32_t)limits->linear_order,
			  (uint32_t)limits->cmp_order, draft);
		return valid ? STATUS_UNSUPPORTED : STATUS_INVALID_ARGUMENT;
	}
	if (limits->official_reference > 1u ||
	    (limits->official_reference != 0 &&
	     (limits->row_capacity != 1u || limits->session_capacity != 1u ||
	      limits->official_hc_rows != 1u || draft != 0 ||
	      limits->context_token_limit >
		      DS41F_OFFICIAL_REFERENCE_CONTEXT_LIMIT)))
		return STATUS_UNSUPPORTED;
	if (limits->official_hc_rows != 0 &&
	    (draft != 0 || limits->row_capacity > LINEAR_MAXIMUM_ROWS ||
	     limits->linear_profile != LINEAR_PROFILE_PRECISION ||
	     limits->linear_order != LINEAR_ORDER_REFERENCE ||
	     limits->cmp_order != CMP_ORDER_REFERENCE))
		return STATUS_UNSUPPORTED;
	return STATUS_OK;
}

/* The tensor of prefix and a rule's name, where the layout puts it. */
MUST_CHECK static result_status
tensor_place(const struct pack_record *pack, const struct load_layout *layout,
	     const char *prefix, const struct ds41f_tensor_rule *rule,
	     struct ds41f_tensor *place_out)
{
	char name[DS41F_NAME_CAPACITY];
	const int length =
		snprintf(name, sizeof(name), "%s%s", prefix, rule->name);
	const uint32_t found =
		length > 0 && length < DS41F_NAME_CAPACITY ?
			pack_find_tensor(pack, name, (uint32_t)length) :
			PACK_NOT_FOUND;
	const struct pack_tensor *const tensor =
		found == PACK_NOT_FOUND ? NULL : &pack_tensors_of(pack)[found];
	if (tensor == NULL || tensor->element_type != rule->element_type ||
	    tensor->dimension_count !=
		    (rule->dimensions[1] == 0 ? 1u : DS41F_MATRIX_DIMENSIONS) ||
	    tensor->dimensions[0] != rule->dimensions[0] ||
	    tensor->dimensions[1] != rule->dimensions[1] ||
	    tensor->weight_format != rule->weight_format ||
	    tensor->bits_per_weight != rule->bits_per_weight) {
		LOG_ERROR("the pack holds no tensor %s of type %" PRIu32
			  ", %" PRIu32 " by %" PRIu32 ", format %" PRIu32,
			  name, rule->element_type, rule->dimensions[0],
			  rule->dimensions[1], rule->weight_format);
		return STATUS_CORRUPT_PACK;
	}
	const uint32_t card = layout->card_of_extent[tensor->extent_index];
	const uint64_t offset =
		ds41f_sum(layout->byte_offset_of_extent[tensor->extent_index],
			  tensor->byte_offset_in_extent);
	if (card != PLACEMENT_IN_HOST_MEMORY &&
	    (card >= layout->card_count ||
	     offset > layout->card_byte_count[card] ||
	     tensor->byte_count > layout->card_byte_count[card] - offset)) {
		LOG_ERROR(
			"the layout puts %s beyond the weights of card %" PRIu32,
			name, card);
		return STATUS_INVALID_ARGUMENT;
	}
	*place_out = (struct ds41f_tensor){
		.stored = { offset, tensor->byte_count },
		.weight_format = (enum weight_format_code)rule->weight_format,
		.shape = { rule->dimensions[0],
			   rule->dimensions[1] == 0 ? 1u : rule->dimensions[1],
			   rule->bits_per_weight, 0 },
		.card = card
	};
	return STATUS_OK;
}

/* Where a card's image begins: behind its slots. */
uint64_t ds41f_image_offset_of(const struct ds41f_card *card)
{
	return card->state_offset + card->slots_byte_count;
}

/* Takes the bytes of an array at the next free offset: PROGRAM_NOWHERE for none. */
uint64_t ds41f_take(uint64_t *next, uint64_t byte_count)
{
	const uint64_t offset = *next;
	const uint64_t whole =
		ds41f_sum(byte_count, DS41F_ARRAY_ALIGNMENT_BYTES - 1u);
	const uint64_t aligned =
		whole == UINT64_MAX ?
			UINT64_MAX :
			whole - whole % DS41F_ARRAY_ALIGNMENT_BYTES;
	*next = byte_count == 0 ? offset : ds41f_sum(offset, aligned);
	return byte_count == 0 ? PROGRAM_NOWHERE : offset;
}

/* The bytes of an array (model_v41.h) on a card of those holders: its kind of rows, but those sized apart. A chunk of
 * one slab runs no wide select and needs no views: their arrays take none. */
static uint64_t array_bytes_of(const struct program_limits *limits,
			       bool chunk_attention_arrays, uint32_t array,
			       uint64_t index_entries, uint64_t engram_tables,
			       uint32_t holders, bool reads_candidates)
{
	const struct array_rule *const rule = &array_rules[array];
	const uint64_t rows = limits->row_capacity,
		       slab_rows = ds41f_slab_rows(limits->row_capacity);
	const uint64_t slabs = (rows + slab_rows - 1u) / slab_rows,
		       thin_jobs = slab_rows * DS41F_EXPERTS_PER_ROW;
	const uint64_t wide_jobs =
		(uint64_t)ROUTER_WIDE_JOBS(limits->row_capacity);
	const enum ds41f_rows kind = ds41f_array_rows(
		chunk_attention_arrays, (enum ds41f_array)array,
		(enum ds41f_rows)rule->rows);
	const uint64_t kind_rows =
		kind == DS41F_ROWS_DRAFT ?
			limits->draft_positions :
		kind == DS41F_ROWS_SLAB ?
			slab_rows :
		kind == DS41F_ROWS_SPAN ?
			ds41f_span_rows(limits->row_capacity) :
			rows;
	const uint64_t rowed =
		ds41f_sum(rule->fixed_bytes, kind_rows * rule->row_bytes);
	const bool is_wide = slabs > 1u;
	switch (array) {
	case DS41F_ARRAY_COMPRESSOR_KV: /* a 1-to-1 source's product alone: bfloat16 */
		return (holders & DS41F_BIT(DS41F_HOLDER_POOLED)) != 0 ?
			       rowed :
			       rows * DS41F_HEAD_DIMS * DS41F_BF16;
	case DS41F_ARRAY_SESSIONS:
		return ds41f_product(limits->session_capacity,
				     STEP_SESSION_BYTES);
	case DS41F_ARRAY_INDEX_SCORES:
		/* Candidate readers scan a fixed score extent even when their source keys have a shorter context. */
		if ((limits->official_reference != 0 || reads_candidates) &&
		    index_entries < DS41F_CONTEXT_WITHOUT_CANDIDATES)
			index_entries = DS41F_CONTEXT_WITHOUT_CANDIDATES;
		/* Each prepared index description names its score payload. Split selection uses the annex immediately
         * after that payload; smaller layers may reuse the unused tail of this largest score allocation. Slabs,
         * layers and ordinary/candidate stages run serially on a card, so one annex suffices. */
		return slab_rows *
		       (index_entries * DS41F_BF16 +
			(index_entries >= DS41F_INDEX_SPLIT_MIN_ENTRIES ?
				 DS41F_INDEX_SPLIT_ROW_BYTES :
				 0u));
	case DS41F_ARRAY_ENGRAM_ROWS:
		return engram_tables * rows * DS41F_ENGRAM_ROWS_PER_TOKEN *
		       ENGRAM_ROW_BYTES;
	case DS41F_ARRAY_JOBS:
		return (wide_jobs > thin_jobs ? wide_jobs : thin_jobs) *
		       EXPERT_JOB_BYTES;
	case DS41F_ARRAY_ROUTER_COUNTS:
		return is_wide ? (rows + ROUTER_SORT_ROWS - 1u) /
					 ROUTER_SORT_ROWS * DS41F_EXPERTS *
					 sizeof(uint32_t) :
				 0;
	case DS41F_ARRAY_ROUTER_TILES:
		return is_wide ? ROUTER_WIDE_TILES(rows) *
					 sizeof(struct wide_expert_tile) :
				 0;
	case DS41F_ARRAY_VIEWS: /* the slabs' views, then split_overlap's two halves' (ds41f_tp_stage.h) */
		return is_wide ? slabs * (STEP_BLOCK_BYTES_VALUE +
					  slab_rows * STEP_ROW_BYTES_VALUE) +
					 (uint64_t)(DS41F_HALF_VIEWS *
						    STEP_BLOCK_BYTES_VALUE) +
					 rows * STEP_ROW_BYTES_VALUE :
				 0;
	case DS41F_ARRAY_ROUTER_PICKS:
	case DS41F_ARRAY_ROUTER_PAIRS:
	case DS41F_ARRAY_ROUTER_RUNS:
		return is_wide ? rowed : 0;
	default:
		return rowed;
	}
}

/* Lays out a card's state from state_offset on (model_v41.h); false when it does not fit 64 bits. A draft stage reads
 * no pool, selection or keys and has no n-gram table: it takes its table, ring and prompt-checkpoint window. */
bool ds41f_card_lay_out_attention(struct ds41f_plan *plan,
				  struct ds41f_card *card,
				  uint64_t attention_layers)
{
	const uint64_t rows = plan->limits.row_capacity;
	const uint64_t sessions = plan->limits.session_capacity;
	const uint64_t walked = ds41f_walk_layers(card);
	card->slots_byte_count =
		ds41f_product(card->slot_count, plan->slot_byte_count);
	uint64_t next = ds41f_sum(card->state_offset, card->slots_byte_count);
	uint64_t sources_read = 0, sources_scored = 0, groups_read = 0,
		 index_entries = 0, engram_tables = 0;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		const uint64_t on_card = card->layers & DS41F_BIT(layer);
		const uint64_t attention_on_card = attention_layers & on_card;
		const uint32_t source =
			ds41f_latest_of(DS41F_SOURCE_LAYERS, layer);
		const uint32_t group =
			ds41f_latest_of(DS41F_INDEX_LAYERS, layer);
		sources_read |= attention_on_card != 0 &&
						source != DS41F_LAYERS ?
					DS41F_BIT(source) :
					0;
		groups_read |= attention_on_card != 0 && group != DS41F_LAYERS ?
				       DS41F_BIT(group) :
				       0;
		if ((on_card & DS41F_INDEX_LAYERS) != 0 &&
		    source < DS41F_LAYERS) {
			sources_scored |=
				attention_on_card != 0 ? DS41F_BIT(source) : 0;
			index_entries =
				plan->key_session_bytes[source] /
							DS41F_KEY_BYTES >
						index_entries ?
					plan->key_session_bytes[source] /
						DS41F_KEY_BYTES :
					index_entries;
		}
		engram_tables += (on_card & DS41F_ENGRAM_LAYERS) != 0 ? 1u : 0u;
		card->expert_table_offset[layer] =
			ds41f_take(&next, (walked & DS41F_BIT(layer)) != 0 ?
						  DS41F_EXPERT_OFFSETS_BYTES :
						  0);
	}
	for (uint32_t array = 0; array < DS41F_ARRAY_COUNT; array++) {
		const uint64_t bytes = array_bytes_of(
			&plan->limits, plan->chunk_attention_arrays, array,
			index_entries, engram_tables, card->holders,
			(card->layers & DS41F_CANDIDATE_READERS) != 0);
		card->array_offset[array] = ds41f_take(
			&next, (card->holders &
				DS41F_BIT(array_rules[array].holder)) != 0 ?
				       bytes :
				       0);
		card->image_byte_count =
			array == DS41F_ARRAY_FREQUENCIES && card->layers != 0 ?
				next - ds41f_image_offset_of(card) :
				card->image_byte_count;
	}
	card->official_scratch_offset =
		plan->limits.official_reference != 0 ?
			ds41f_take(&next, DS41F_OFFICIAL_SCRATCH_BYTES) :
			PROGRAM_NOWHERE;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		const uint64_t bit = DS41F_BIT(layer);
		const bool on_card = (attention_layers & bit) != 0;
		card->selection_offset[layer] = ds41f_take(
			&next, (groups_read & bit) != 0 ?
				       rows * DS41F_SELECTION_ROW_BYTES :
				       0);
		card->ring_offset[layer] = ds41f_take(
			&next, on_card ?
				       ds41f_product(sessions,
						     DS41F_RING_SESSION_BYTES) :
				       0);
		card->checkpoint_offset[layer] = ds41f_take(
			&next,
			on_card ?
				ds41f_product(sessions,
					      DS41F_CHECKPOINT_SESSION_BYTES) :
				0);
		card->compressor_offset[layer] = ds41f_take(
			&next,
			on_card && (DS41F_POOLED_LAYERS & bit) != 0 ?
				ds41f_product(sessions,
					      DS41F_COMPRESSOR_SESSION_BYTES) :
				0);
		card->pool_offset[layer] = ds41f_take(
			&next,
			(sources_read & bit) != 0 ?
				ds41f_product(sessions,
					      plan->pool_session_bytes[layer]) :
				0);
		card->key_tiles_offset[layer] = ds41f_take(
			&next,
			(sources_scored & bit) != 0 ?
				ds41f_product(sessions,
					      plan->key_session_bytes[layer]) :
				0);
	}
	card->front_scratch_offset = PROGRAM_NOWHERE;
	card->producer_arrivals_offset = PROGRAM_NOWHERE;
	if (card->layers != 0) {
		if (rows >= STEP_WIDE_LEAST_ROWS) {
			const uint64_t first =
				card->array_offset[DS41F_ARRAY_ROUTER_PICKS];
			const uint64_t end =
				card->array_offset[DS41F_ARRAY_JOBS];
			if (first == PROGRAM_NOWHERE || end < first ||
			    end - first < DS41F_FUSED_SCRATCH_BYTES) {
				return false;
			}
			card->front_scratch_offset = first;
		} else {
			card->front_scratch_offset =
				ds41f_take(&next, DS41F_FUSED_SCRATCH_BYTES);
		}
		card->producer_arrivals_offset =
			ds41f_sum(card->front_scratch_offset,
				  DS41F_SITE_FRONT_SCRATCH_BYTES);
	}
	bool has_exl3 = false;
	for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++)
		has_exl3 = has_exl3 ||
			   ((card->layers & DS41F_BIT(layer)) != 0 &&
			    plan->exl3_bits[layer] != 0);
	card->exl3_scratch_offset =
		has_exl3 ?
			ds41f_take(&next, exl3_scratch_byte_count(
						  plan->limits.row_capacity *
							  DS41F_EXPERTS_PER_ROW,
						  DS41F_HIDDEN)) :
			PROGRAM_NOWHERE;
	card->arena_byte_count = next;
	return next != UINT64_MAX;
}

bool ds41f_card_lay_out(struct ds41f_plan *plan, struct ds41f_card *card)
{
	return ds41f_card_lay_out_attention(plan, card,
					    ds41f_walk_layers(card));
}

/* The prefix of a walk layer's tensor names: "layers.<l>.", a draft stage's "mtp.<l - 40>.". */
void ds41f_layer_prefix_of(uint32_t layer,
			   char prefix[DS41F_LAYER_PREFIX_CAPACITY])
{
	const bool is_draft = layer >= DS41F_DRAFT_FIRST_LAYER;
	(void)snprintf(prefix, DS41F_LAYER_PREFIX_CAPACITY, "%s%" PRIu32 ".",
		       is_draft ? "mtp." : "layers.",
		       is_draft ? layer - DS41F_DRAFT_FIRST_LAYER : layer);
}

/* The routed experts of a walk layer, before the cards are laid out: the entries of its expert table, those on its
 * card spanning plan->experts, those in host memory numbered by tag behind the slots at the end of the card's weights.
 * A draft stage's lie on its card: the draft runs no expert pause. */
MUST_CHECK static result_status experts_place(const struct pack_record *pack,
					      const struct load_layout *layout,
					      struct ds41f_plan *plan,
					      uint32_t layer)
{
	const uint32_t card = plan->layer_card[layer];
	const bool is_draft = layer >= DS41F_DRAFT_FIRST_LAYER;
	uint64_t first = UINT64_MAX, end = 0;
	char layer_prefix[DS41F_LAYER_PREFIX_CAPACITY],
		prefix[DS41F_NAME_CAPACITY];
	ds41f_layer_prefix_of(layer, layer_prefix);
	for (uint32_t expert = 0;
	     expert < (is_draft ? DS41F_DRAFT_EXPERTS : DS41F_EXPERTS);
	     expert++) {
		struct ds41f_tensor places[DS41F_EXPERT_MATRICES];
		uint64_t *const entries = plan->expert_tables[layer][expert];
		uint64_t low = UINT64_MAX, high = 0, own = 0;
		uint32_t on_card = 0, in_host = 0;
		(void)snprintf(prefix, sizeof(prefix),
			       "%sffn.experts.%" PRIu32 ".", layer_prefix,
			       expert);
		for (uint32_t matrix = 0; matrix < DS41F_EXPERT_MATRICES;
		     matrix++) {
			const result_status status = tensor_place(
				pack, layout, prefix,
				&ds41f_expert_rules[matrix], &places[matrix]);
			if (status != STATUS_OK) {
				return status;
			}
			const struct linear_range *const stored =
				&places[matrix].stored;
			on_card += places[matrix].card == card ? 1u : 0u;
			in_host += places[matrix].card ==
						   PLACEMENT_IN_HOST_MEMORY ?
					   1u :
					   0u;
			low = stored->byte_offset < low ? stored->byte_offset :
							  low;
			high = ds41f_sum(stored->byte_offset,
					 stored->byte_count) > high ?
				       ds41f_sum(stored->byte_offset,
						 stored->byte_count) :
				       high;
			own = ds41f_sum(
				own,
				ds41f_sum(stored->byte_count,
					  WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES));
		}
		const uint32_t tag = plan->host_expert_count[layer];
		const bool is_on_card = on_card == DS41F_EXPERT_MATRICES;
		const bool is_first_in_host = !is_on_card &&
					      plan->expert_byte_count == 0;
		bool is_placed =
			is_on_card ||
			(!is_draft && in_host == DS41F_EXPERT_MATRICES &&
			 high - low < own && high <= layout->host_byte_count &&
			 (is_first_in_host ||
			  plan->expert_byte_count == high - low));
		for (uint32_t matrix = 0; matrix < DS41F_EXPERT_MATRICES;
		     matrix++) {
			is_placed = is_placed &&
				    (is_on_card || is_first_in_host ||
				     plan->expert_matrix_offsets[matrix] ==
					     places[matrix].stored.byte_offset -
						     low);
		}
		if (!is_placed) {
			LOG_ERROR(
				"expert %" PRIu32 " of %s %" PRIu32 " is %s",
				expert, is_draft ? "draft stage" : "layer",
				is_draft ? layer - DS41F_DRAFT_FIRST_LAYER :
					   layer,
				is_draft ?
					"not on the draft's card, and the draft runs no expert pause" :
					"neither on its layer's card nor whole in host memory with its matrices "
					"together as in the others");
			return STATUS_UNSUPPORTED;
		}
		for (uint32_t matrix = 0; matrix < DS41F_EXPERT_MATRICES;
		     matrix++) {
			entries[matrix] =
				is_on_card ?
					places[matrix].stored.byte_offset :
					ds41f_sum(
						layout->card_byte_count[card],
						(uint64_t)tag *
							LINEAR_OFFSET_ALIGNMENT_BYTES);
		}
		first = is_on_card && low < first ? low : first;
		end = is_on_card && high > end ? high : end;
		if (!is_on_card) {
			for (uint32_t matrix = 0;
			     matrix < DS41F_EXPERT_MATRICES; matrix++) {
				plan->expert_matrix_offsets[matrix] =
					places[matrix].stored.byte_offset - low;
			}
			plan->host_expert_offsets[layer][tag] = low;
			plan->host_expert_count[layer] = tag + 1u;
			plan->expert_byte_count = high - low;
		}
	}
	plan->experts[layer] =
		first == UINT64_MAX ?
			(struct linear_range){ 0, 0 } :
			(struct linear_range){ first, end - first };
	return STATUS_OK;
}

/* Where a layer keeps routed experts in host memory: its card's slots, and its experts' range to their end. */
static void slots_place(const struct load_layout *layout,
			struct ds41f_plan *plan)
{
	const uint64_t whole = ds41f_sum(plan->expert_byte_count,
					 DS41F_ARRAY_ALIGNMENT_BYTES - 1u);
	plan->slot_byte_count = whole - whole % DS41F_ARRAY_ALIGNMENT_BYTES;
	for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
		struct ds41f_card *const card =
			&plan->cards[plan->layer_card[layer]];
		const uint64_t slots =
			layout->card_byte_count[plan->layer_card[layer]];
		if (plan->host_expert_count[layer] != 0) {
			card->slot_count =
				DS41F_EXPERTS_PER_ROW *
				ds41f_slab_rows(plan->limits.row_capacity);
			const uint64_t first =
				plan->experts[layer].byte_count == 0 ?
					slots :
					plan->experts[layer].byte_offset;
			plan->experts[layer] = (struct linear_range){
				first,
				ds41f_sum(slots,
					  ds41f_product(card->slot_count,
							plan->slot_byte_count)) -
					first
			};
		}
	}
}

/* The first tensor of the draft or of layers 37 to 39 off the head's card, which the admission's line names. */
struct apart_tensor {
	char name[DS41F_NAME_CAPACITY];
	uint32_t card;
};
static void apart_note(struct apart_tensor *apart, const char *prefix,
		       const struct ds41f_tensor_rule *rule,
		       const struct ds41f_tensor *place, uint32_t head_card)
{
	if (apart->name[0] == '\0' && place->stored.byte_count != 0 &&
	    place->card != head_card) {
		(void)snprintf(apart->name, sizeof(apart->name), "%s%s", prefix,
			       rule->name);
		apart->card = place->card;
	}
}

/* The plan of a layout (model_v41.h): plan->limits is set, every other member zero. */
MUST_CHECK static result_status plan_build(const struct pack_record *pack,
					   const struct load_layout *layout,
					   struct ds41f_plan *plan)
{
	float words[DS41F_THETAS][DS41F_FREQUENCIES];
	const uint32_t card_count = layout->card_count;
	const bool has_draft = plan->limits.draft_positions != 0;
	const uint32_t walk_count = has_draft ? DS41F_WALK_LAYERS :
						DS41F_LAYERS;
	result_status status = STATUS_OK;
	uint32_t apart =
		DS41F_LAYERS; /* the first layer that is not on one card */
	struct apart_tensor draft_apart = { { 0 }, 0 };
	plan->card_count = card_count;
	char prefix[DS41F_LAYER_PREFIX_CAPACITY];
	for (uint32_t tensor = 0;
	     tensor < DS41F_TENSOR_GLOBALS && status == STATUS_OK; tensor++) {
		status = tensor_place(pack, layout, "",
				      &ds41f_global_rules[tensor],
				      &plan->globals[tensor]);
	}
	const uint32_t head_card = plan->globals[DS41F_TENSOR_HEAD].card;
	for (uint32_t tensor = 0;
	     has_draft && tensor < DS41F_TENSOR_DRAFT_GLOBALS &&
	     status == STATUS_OK;
	     tensor++) {
		status = tensor_place(pack, layout, "",
				      &ds41f_draft_rules[tensor],
				      &plan->draft_globals[tensor]);
		apart_note(&draft_apart, "", &ds41f_draft_rules[tensor],
			   &plan->draft_globals[tensor], head_card);
	}
	for (uint32_t layer = 0, table = 0;
	     layer < walk_count && status == STATUS_OK; layer++) {
		const bool is_draft = layer >= DS41F_DRAFT_FIRST_LAYER;
		const bool is_drafted = has_draft &&
					(is_draft || (DS41F_DRAFT_TAP_LAYERS &
						      DS41F_BIT(layer)) != 0);
		struct ds41f_tensor *const tensors = plan->layer_tensors[layer];
		ds41f_layer_prefix_of(layer, prefix);
		for (uint32_t tensor = 0;
		     tensor < DS41F_TENSOR_LAYER_COUNT && status == STATUS_OK;
		     tensor++) {
			const struct ds41f_tensor_rule *const rule =
				is_draft && tensor >= DS41F_TENSOR_ROUTER &&
						tensor <=
							DS41F_TENSOR_ROUTER_BIAS ?
					&ds41f_draft_router_rules
						[tensor - DS41F_TENSOR_ROUTER] :
					&ds41f_layer_rules[tensor];
			status = (rule->layers & DS41F_BIT(layer)) == 0 ?
					 STATUS_OK :
					 tensor_place(pack, layout, prefix,
						      rule, &tensors[tensor]);
			apart = !is_draft && layer < apart &&
						tensors[tensor].stored
								.byte_count !=
							0 &&
						tensors[tensor].card !=
							tensors[DS41F_TENSOR_ATTENTION_NORM]
								.card ?
					layer :
					apart;
			if (is_drafted) {
				apart_note(&draft_apart, prefix, rule,
					   &tensors[tensor], head_card);
			}
		}
		plan->layer_card[layer] =
			tensors[DS41F_TENSOR_ATTENTION_NORM].card;
		if (is_draft) {
			continue; /* its card is marked behind the checks */
		}
		apart = layer < apart && plan->layer_card[layer] >= card_count ?
				layer :
				apart;
		struct ds41f_card *const card =
			&plan->cards[apart < DS41F_LAYERS ?
					     0 :
					     plan->layer_card[layer]];
		card->layers |= DS41F_BIT(layer);
		card->holders |= (uint32_t)(DS41F_BIT(DS41F_HOLDER_TRUNK) |
					    DS41F_BIT(DS41F_HOLDER_LAYER));
		card->holders |=
			(uint32_t)((DS41F_INDEX_LAYERS & DS41F_BIT(layer)) !=
						   0 ?
					   DS41F_BIT(DS41F_HOLDER_INDEX) :
					   0);
		card->holders |=
			(uint32_t)((DS41F_SOURCE_LAYERS & DS41F_BIT(layer)) !=
						   0 ?
					   DS41F_BIT(DS41F_HOLDER_SOURCE) :
					   0);
		card->holders |=
			(uint32_t)((DS41F_POOLED_LAYERS & DS41F_BIT(layer)) !=
						   0 ?
					   DS41F_BIT(DS41F_HOLDER_POOLED) :
					   0);
		card->holders |=
			(uint32_t)(layer == DS41F_CANDIDATE_SOURCE ||
						   (DS41F_CANDIDATE_READERS &
						    DS41F_BIT(layer)) != 0 ?
					   DS41F_BIT(DS41F_HOLDER_CANDIDATES) :
					   0);
		if ((DS41F_ENGRAM_LAYERS & DS41F_BIT(layer)) != 0) {
			card->holders |=
				(uint32_t)DS41F_BIT(DS41F_HOLDER_ENGRAM);
			card->engram_tables |= (uint32_t)DS41F_BIT(table++);
		}
		if ((DS41F_SOURCE_LAYERS & DS41F_BIT(layer)) != 0) {
			const uint64_t entries =
				plan->limits.context_token_limit /
				((DS41F_POOLED_LAYERS & DS41F_BIT(layer)) != 0 ?
					 DS41F_POOLED_RATIO :
					 DS41F_DIRECT_RATIO);
			plan->pool_session_bytes[layer] =
				entries * DS41F_STATE_ROW_BYTES;
			plan->key_session_bytes[layer] =
				entries * DS41F_KEY_BYTES;
		}
	}
	plan->embedding_card = plan->globals[DS41F_TENSOR_EMBEDDING].card;
	plan->result_card = plan->globals[DS41F_TENSOR_HEAD].card;
	if (status == STATUS_OK && apart < DS41F_LAYERS) {
		LOG_ERROR("this build runs a layer on one card: layer %" PRIu32
			  " is not",
			  apart);
		status = STATUS_UNSUPPORTED;
	} else if (status == STATUS_OK &&
		   (plan->embedding_card >= card_count ||
		    plan->result_card >= card_count ||
		    plan->globals[DS41F_TENSOR_NORM].card !=
			    plan->result_card)) {
		LOG_ERROR(
			"this build runs the embedding on a card and the head with the norm on one card");
		status = STATUS_UNSUPPORTED;
	} else if (status == STATUS_OK && draft_apart.name[0] != '\0') {
		LOG_ERROR(
			"the draft needs layers 37 to 39, the head and the draft head on one card, but %s lies on "
			"card %" PRIu32 " and the head on card %" PRIu32
			": place them together (--layout, or "
			"--card_order for the fill)",
			draft_apart.name, draft_apart.card, plan->result_card);
		status = STATUS_UNSUPPORTED;
	}
	if (status != STATUS_OK) {
		return status;
	}
	status = ds41f_model_vision_weights(pack, layout, plan->layer_card,
					    &plan->vision);
	if (status != STATUS_OK ||
	    (plan->vision.enabled && plan->embedding_card != 0)) {
		return status != STATUS_OK ? status : STATUS_UNSUPPORTED;
	}
	plan->cards[plan->embedding_card].holders |=
		(uint32_t)DS41F_BIT(DS41F_HOLDER_TRUNK);
	plan->cards[plan->result_card].holders |=
		(uint32_t)(DS41F_BIT(DS41F_HOLDER_TRUNK) |
			   DS41F_BIT(DS41F_HOLDER_RESULT) |
			   (has_draft ?
				    DS41F_BIT(DS41F_HOLDER_DRAFT) |
					    DS41F_BIT(DS41F_HOLDER_DRAFT_COPY) :
				    0));
	for (uint32_t layer = 0; layer < walk_count && status == STATUS_OK;
	     layer++) {
		status = experts_place(pack, layout, plan, layer);
	}
	if (status == STATUS_OK) {
		slots_place(layout, plan);
	}
	ds41f_frequency_words(words);
	for (uint32_t card_index = 0;
	     card_index < card_count && status == STATUS_OK; card_index++) {
		struct ds41f_card *const card = &plan->cards[card_index];
		card->state_offset = layout->card_byte_count[card_index];
		const bool aligned =
			card->state_offset % DS41F_ARRAY_ALIGNMENT_BYTES == 0;
		if (!aligned || !ds41f_card_lay_out(plan, card)) {
			LOG_ERROR(
				"card %" PRIu32
				": its weights end at byte %" PRIu64 ", %s %u",
				card_index, card->state_offset,
				aligned ?
					"and no state fits 64 bits behind them on a multiple of" :
					"not on a multiple of",
				DS41F_ARRAY_ALIGNMENT_BYTES);
			status = STATUS_INVALID_ARGUMENT;
		}
		if (status == STATUS_OK && card_index == 0 &&
		    plan->vision.enabled) {
			status = ds41f_model_vision_arena(
				&plan->vision, plan->limits.session_capacity,
				card->arena_byte_count,
				&card->arena_byte_count);
		}
		if (status == STATUS_OK && card->image_byte_count != 0) {
			card->image = FAULT_POINT(model_v41_image_memory) ?
					      NULL :
					      calloc(1, card->image_byte_count);
			status = card->image == NULL ? STATUS_OUT_OF_MEMORY :
						       STATUS_OK;
		}
		if (card->image != NULL) {
			memcpy(card->image +
				       (card->array_offset
						[DS41F_ARRAY_FREQUENCIES] -
					ds41f_image_offset_of(card)),
			       words, sizeof(words));
		}
		for (uint32_t layer = 0;
		     layer < DS41F_WALK_LAYERS && card->image != NULL;
		     layer++) {
			if ((ds41f_walk_layers(card) & DS41F_BIT(layer)) != 0) {
				memcpy(card->image +
					       (card->expert_table_offset[layer] -
						ds41f_image_offset_of(card)),
				       plan->expert_tables[layer],
				       sizeof(plan->expert_tables[layer]));
			}
		}
	}
	return status;
}

result_status ds41f_describe(const struct pack_record *pack,
			     struct model_description_record *description_out)
{
	if (pack == NULL || description_out == NULL) {
		LOG_ERROR("describe: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	const result_status status = ds41f_model_check(pack);
	if (status != STATUS_OK) {
		return status;
	}
	*description_out = (struct model_description_record){
		DS41F_LAYERS, DS41F_HIDDEN, DS41F_VOCABULARY_VALUE,
		DS41F_CONTEXT_MOST, DS41F_END_TOKEN
	};
	return STATUS_OK;
}

result_status ds41f_plan_create(const struct pack_record *pack,
				const struct load_layout *layout,
				const struct program_limits *limits,
				struct ds41f_plan **plan_out)
{
	struct ds41f_plan *plan = NULL;
	if (pack == NULL || layout == NULL || limits == NULL ||
	    plan_out == NULL) {
		LOG_ERROR("plan_create: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = ds41f_model_check(pack);
	status = status == STATUS_OK ? ds41f_limits_check(limits) : status;
	if (status == STATUS_OK &&
	    (layout->extent_count != pack_header_of(pack)->extent_count ||
	     layout->card_count == 0 ||
	     layout->card_count > BACKEND_MAXIMUM_CARDS)) {
		LOG_ERROR("plan_create: a layout of %" PRIu32
			  " extents on %" PRIu32 " cards",
			  layout->extent_count, layout->card_count);
		status = STATUS_INVALID_ARGUMENT;
	}
	if (status == STATUS_OK) {
		plan = FAULT_POINT(model_v41_plan_memory) ?
			       NULL :
			       calloc(1, sizeof(*plan));
		status = plan == NULL ? STATUS_OUT_OF_MEMORY : STATUS_OK;
	}
	if (status == STATUS_OK) {
		plan->limits = *limits;
		status = plan_build(pack, layout, plan);
	}
	if (status == STATUS_OK) {
		*plan_out = plan;
	} else {
		ds41f_plan_destroy(plan);
	}
	return status;
}

void ds41f_plan_destroy(struct ds41f_plan *plan)
{
	for (uint32_t card = 0; plan != NULL && card < BACKEND_MAXIMUM_CARDS;
	     card++) {
		free(plan->cards[card].image);
	}
	free(plan);
}
