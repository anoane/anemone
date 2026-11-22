/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_attention.c: a layer's attention and Engram launches (split from model_v41_program.c by whole items).
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

/* The attention steps of layer with the pool of source, the latest source at or before the layer, and the selection
 * of group, the latest index layer (0 for none: layer 0 is neither). The selections are of the rows the launches
 * take. The site's and the state's launches are prepared again for each call that puts one of them. */
MUST_CHECK static result_status
attention_launches(struct ds41f_builder *b, uint32_t layer, uint32_t source,
		   uint32_t group, uint32_t steps)
{
	const uint64_t selection_into =
		(uint64_t)b->first_row * DS41F_SELECTION_ROW_BYTES;
	const struct ds41f_plan *const plan = b->plan;
	const struct ds41f_card *const card = &plan->cards[b->card];
	const struct ds41f_tensor *const weights = plan->layer_tensors[layer];
	const bool is_draft = layer >= DS41F_DRAFT_FIRST_LAYER;
	const bool is_thin = b->slab_count == 1u;
	const bool is_exl3 = b->plan->exl3_bits[layer] != 0;
	/* draft_bf16_rows: a draft layer's wq_a, wkv, wq_b and wo_b of its BF16 rows, as an EXL3 layer's
	 * (drafts
	 * only) */
	const bool bf16_rows = is_exl3 || (is_draft && b->draft_bf16_rows);
	const bool fused_projection =
		b->plan->limits.official_reference == 0 && is_thin &&
		!bf16_rows && b->tensor_layer == NULL &&
		!ds41f_trunk_is_relaxed(b, DS41F_WEIGHTS(OUTPUT));

	const uint32_t ratio =
		(DS41F_SOURCE_LAYERS >> source & 1u) == 0 ? 0 :
		(DS41F_POOLED_LAYERS >> source & 1u) != 0 ? DS41F_POOLED_RATIO :
							    DS41F_DIRECT_RATIO;
	const bool has_index = group == layer && layer != 0,
		   writes_pool = source == layer && ratio != 0;
	const float index_weights_scale = DS41F_INDEX_WEIGHTS_SCALE;
	const uint64_t sessions = plan->limits.session_capacity, rows = b->rows;
	const uint32_t context = plan->limits.context_token_limit,
		       entries = ratio == 0 ? 0 : context / ratio;
	const struct linear_range pool = {
		card->pool_offset[source],
		sessions * plan->pool_session_bytes[source]
	};
	const struct linear_range keys = {
		card->key_tiles_offset[source],
		sessions * plan->key_session_bytes[source]
	};
	const struct linear_range ring = {
		card->ring_offset[layer], sessions * DS41F_RING_SESSION_BYTES
	};
	/* wkv's product, packed at its row: float32 at ratio 2, bfloat16 at 1 (model_v41.h) */
	const uint64_t kv_row_bytes =
		(uint64_t)DS41F_HEAD_DIMS *
		(ratio == DS41F_POOLED_RATIO ? DS41F_F32 : DS41F_BF16);
	const struct linear_range pool_input =
		ds41f_packed_rows(b, DS41F_ARRAY_COMPRESSOR_KV, kv_row_bytes);
	struct kernel_launch site[DS41F_HYPER_SITE_LAUNCH_COUNT],
		state[DS41F_ATTENTION_STATE_LAUNCH_COUNT];
	struct kernel_launch index_launches[DS41F_INDEX_SCAN_LAUNCH_COUNT];
	struct kernel_launch core_launches[DS41F_ATTENTION_CORE_MOST_LAUNCHES];
	uint32_t index_launch_count = 0, core_launch_count = 0;
	const uint32_t head_count =
		b->tensor_layer == NULL ?
			DS41F_HEADS_VALUE :
			DS41F_HEADS_VALUE / b->tensor_layer->rank_count;
	const uint32_t first_head = b->tensor_layer == NULL ?
					    0 :
					    b->tensor_layer->rank * head_count;
	const struct ds41f_attention_state_description state_description = {
		DS41F_ARRAY(STEP),
		DS41F_ARRAY(ROTATION),
		DS41F_ARRAY(QUERY_RANK),
		DS41F_WEIGHTS(QUERY_NORM)->stored,
		DS41F_ARRAY(QUERY_NORMED),
		DS41F_ARRAY(WINDOW_INPUT),
		DS41F_WEIGHTS(WINDOW_NORM)->stored,
		ring,
		DS41F_ARRAY(HEADS),
		DS41F_ARRAY(QUERIES),
		DS41F_ARRAY(INDEX_HEADS),
		DS41F_ARRAY(INDEX_QUERIES),
		pool_input,
		DS41F_ARRAY(COMPRESSOR_SCORE),
		{ card->compressor_offset[layer],
		  sessions * DS41F_COMPRESSOR_SESSION_BYTES },
		DS41F_WEIGHTS(COMPRESSOR_NORM)->stored,
		DS41F_ARRAY(LATENT),
		pool,
		DS41F_ARRAY(KEY_INPUT),
		DS41F_WEIGHTS(KEY_NORM)->stored,
		keys,
		b->rows,
		(uint32_t)sessions,
		context,
		ratio != 0 ? 1u : 0u,
		first_head,
		head_count,
		has_index ? 1u : 0u,
		writes_pool ? ratio : 0,
		0
	};
	const struct ds41f_attention_query_norm_description query_norm = {
		DS41F_ARRAY(QUERY_RANK), DS41F_WEIGHTS(QUERY_NORM)->stored,
		DS41F_ARRAY(QUERY_NORMED), b->rows, 0
	};
	const struct ds41f_attention_queries_description queries = {
		DS41F_ARRAY(ROTATION),
		DS41F_ARRAY(HEADS),
		DS41F_ARRAY(QUERIES),
		DS41F_ARRAY(INDEX_HEADS),
		DS41F_ARRAY(INDEX_QUERIES),
		b->rows,
		ratio != 0 ? 1u : 0u,
		first_head,
		head_count,
		has_index ? 1u : 0u,
		0
	};
	const uint32_t candidate_role =
		layer == DS41F_CANDIDATE_SOURCE ?
			DS41F_INDEX_CANDIDATES_SOURCE :
		(DS41F_CANDIDATE_READERS >> layer & 1u) != 0 ?
			DS41F_INDEX_CANDIDATES_READER :
			DS41F_INDEX_CANDIDATES_NONE;
	const uint64_t row_scores =
		candidate_role == DS41F_INDEX_CANDIDATES_READER ?
			DS41F_CONTEXT_WITHOUT_CANDIDATES :
			entries;
	/* index_span: a span's rows of the view from index_row_first on, their scores in the stage's (index_scans) */
	const bool grouped = b->index_grouped && b->tensor_layer != NULL;
	const struct ds41f_index_scan_description scan = {
		DS41F_ARRAY(STEP),
		keys,
		DS41F_ARRAY(INDEX_QUERIES),
		DS41F_ARRAY(INDEX_WEIGHTS),
		{ grouped ? b->tensor_layer->index_scores.byte_offset :
			    DS41F_ARRAY(INDEX_SCORES).byte_offset,
		  rows * row_scores * DS41F_INDEX_SCORE_BYTES },
		{ card->selection_offset[layer] + selection_into,
		  rows * DS41F_SELECTION_ROW_BYTES },
		candidate_role != DS41F_INDEX_CANDIDATES_NONE ?
			DS41F_ARRAY(CANDIDATES) :
			ds41f_no_range,
		context,
		(uint32_t)sessions,
		ratio,
		b->rows,
		candidate_role,
		grouped ? b->index_row_first : 0u,
		grouped ? LINEAR_MAXIMUM_ROWS : 0u
	};
	const struct ds41f_attention_core_description core = {
		DS41F_ARRAY(STEP),
		DS41F_ARRAY(QUERIES),
		ring,
		pool,
		{ card->selection_offset[group] + selection_into,
		  rows * DS41F_SELECTION_ROW_BYTES },
		DS41F_WEIGHTS(SINK)->stored,
		DS41F_ARRAY(ROTATION),
		DS41F_ARRAY(ATTENTION_OUTPUT),
		DS41F_ARRAY(ATTENTION_SCORES),
		b->rows,
		(uint32_t)sessions,
		context,
		ratio,
		first_head,
		head_count,
		is_draft ? LINEAR_ORDER_FIXED : b->plan->limits.linear_order,
		b->arithmetic_class,
		is_draft ? DS41F_ATTENTION_CORE_WINDOW_BLOCK :
			   DS41F_ATTENTION_CORE_WINDOW_CAUSAL,
		0,
		0,
		b->core_heads
	};
	struct linear_trunk_bfloat16_description groups = {
		.values = DS41F_ARRAY(ATTENTION_OUTPUT),
		.output = DS41F_ARRAY(GROUPS),
		.row_count = b->rows,
		.segment_count = b->tensor_layer == NULL ?
					 DS41F_OUTPUT_GROUPS :
					 DS41F_OUTPUT_GROUPS /
						 b->tensor_layer->rank_count
	};
	/* front_branch, a thin TP step's whole attention site: its front as the input front, then the coefficient front
     * as a graph side branch (it reads the streams, which nothing writes before the site's post, and writes post,
     * comb and the next pre, which the input front read before it) beside the products, the core and the sum,
     * joined at the post: the same launches' bits (ds41f_site_front_records.h), the mix project off the chain */
	const bool front_branch =
		b->plan->limits.official_hc_rows == 0 && is_thin &&
		b->front_branch != 0 && b->tensor_layer != NULL &&
		!fused_projection &&
		(steps & (DS41F_SITE_MIX | DS41F_SITE_POST)) ==
			(DS41F_SITE_MIX | DS41F_SITE_POST);
	const uint32_t side_launch = b->made->cards[b->card].launch_count +
				     1u; /* the coefficient front's place */
	result_status status =
		front_branch ?
			ds41f_site_front_of(b, weights, false, NULL,
					    DS41F_SITE_FRONT_KIND_INPUT) :
			DS41F_OK;
	status = front_branch ? DS41F_THEN(ds41f_site_front_of(
					b, weights, false, NULL,
					DS41F_SITE_FRONT_KIND_COEFFICIENTS)) :
				status;
	status = front_branch ?
			 status :
		 (steps & DS41F_SITE_MIX) != 0 ?
			 (is_thin && b->plan->limits.official_hc_rows == 0 ?
				  ds41f_site_front(b, weights, false, NULL) :
				  ds41f_site_begin(b, weights, false, site))

		 :
		 (steps & (DS41F_SITE_POST | DS41F_SITE_POST_ONLY)) != 0 ?
			 ds41f_site_prepare(b, weights, false, site) :
			 DS41F_OK;
	/* draft_bf16_operands: a thin draft site front's rows carried again (the front's own operands are E4M3) */
	if (is_draft && is_thin && !front_branch &&
	    (steps & DS41F_SITE_MIX) != 0 &&
	    b->draft_carry_kernel != BACKEND_NO_KERNEL)
		status = DS41F_THEN(
			ds41f_quantize(b, DS41F_ARRAY(HIDDEN), DS41F_HIDDEN));
	if (is_thin && !fused_projection && (steps & DS41F_SITE_POST) != 0) {
		status =
			DS41F_THEN(ds41f_site_prepare(b, weights, false, site));
	}
	status = (steps & (DS41F_STATE_WRITES | DS41F_STATE_POOL)) == 0 ?
			 status :
			 DS41F_THEN(ds41f_attention_state_prepare(
				 &state_description, b->arena, &b->state,
				 state));
	if ((steps & DS41F_INPUT_PRODUCTS) != 0) {
		/* an EXL3 layer's wq_a and wkv (exl3_trunk_bfloat16) read the hidden BF16 rows: no
		 * quantization of them
         * in a chunked or BATCH step (an n-gram layer quantizes its own values; launches_untrimmed: as before) */
		const bool unread = is_exl3 && b->tensor_layer != NULL &&
				    !b->tensor_layer->launches_untrimmed;
		status = (is_thin && b->plan->limits.official_hc_rows == 0) ||
					 unread ?
				 status :
				 DS41F_THEN(ds41f_quantize(
					 b, DS41F_ARRAY(HIDDEN), DS41F_HIDDEN));
		if (bf16_rows) {
			status = DS41F_THEN(ds41f_exl3_trunk_bfloat16(
				b, DS41F_WEIGHTS(QUERY_DOWN),
				DS41F_ARRAY(HIDDEN), DS41F_ARRAY(QUERY_RANK)));
			status = DS41F_THEN(ds41f_exl3_trunk_bfloat16(
				b, DS41F_WEIGHTS(WINDOW), DS41F_ARRAY(HIDDEN),
				DS41F_ARRAY(WINDOW_INPUT)));
			if (status == DS41F_OK)
				ds41f_trunk_mma_pair_merge(
					b); /* wq_a and wkv in one launch where both took the pair form's kernel */
		} else {
			status = DS41F_THEN(ds41f_trunk(
				b, DS41F_WEIGHTS(QUERY_DOWN),
				DS41F_ARRAY(QUERY_RANK), DS41F_WEIGHTS(WINDOW),
				DS41F_ARRAY(WINDOW_INPUT)));
		}
	}
	if (status == DS41F_OK && b->plan->limits.official_reference != 0) {
		state[DS41F_ATTENTION_STATE_PREPARE].kernel =
			b->official.window;
		state[DS41F_ATTENTION_STATE_PREPARE].grid[0] =
			state[DS41F_ATTENTION_STATE_PREPARE].grid[1] = 1;
		state[DS41F_ATTENTION_STATE_PREPARE].block[0] =
			DS41F_OFFICIAL_STATE_THREADS;
		state[DS41F_ATTENTION_STATE_COMPRESSOR].kernel =
			b->official.compressor;
		state[DS41F_ATTENTION_STATE_COMPRESSOR].block[0] =
			DS41F_OFFICIAL_STATE_THREADS;
		state[DS41F_ATTENTION_STATE_KEY].kernel = b->official.key;
		state[DS41F_ATTENTION_STATE_KEY].block[0] =
			DS41F_OFFICIAL_STATE_THREADS;
	}
	if ((steps & DS41F_STATE_WRITES) != 0) {
		if (b->plan->limits.official_reference != 0)
			status = DS41F_THEN(ds41f_official_norm(
				b, DS41F_ARRAY(QUERY_RANK),
				DS41F_WEIGHTS(QUERY_NORM)->stored,
				DS41F_ARRAY(QUERY_NORMED), DS41F_QUERY_RANK));
		status = DS41F_THEN(
			ds41f_put(b, &state[DS41F_ATTENTION_STATE_PREPARE]));
	}
	if ((steps & DS41F_WINDOW_WRITES) !=
	    0) { /* the prepare's window part by its device code (its qr: QUERY_NORM's) */
		const struct ds41f_attention_window_description window = {
			DS41F_ARRAY(STEP),
			DS41F_ARRAY(ROTATION),
			DS41F_ARRAY(WINDOW_INPUT),
			DS41F_WEIGHTS(WINDOW_NORM)->stored,
			ring,
			b->rows,
			(uint32_t)sessions,
			context,
			ratio != 0 ? DS41F_ATTENTION_ROTATION_POOLED :
				     DS41F_ATTENTION_ROTATION_WINDOW_ONLY,
			0
		};
		status = DS41F_THEN(ds41f_attention_window_prepare(
			&window, b->arena, &b->state, ds41f_slots(b, 1)));
	}
	if ((steps & DS41F_POOL_PRODUCTS) != 0 &&
	    writes_pool) { /* the pool's entry's products (2.1 launch 10) */
		/* pool_gather: the rank's part of the half's rows (gather_share's), then those rows gathered */
		const bool gathered = ds41f_pool_gather_applies(b);
		const uint32_t view = b->view, first = b->first_row,
			       all_rows = b->rows;
		uint32_t own_first = 0, own_rows = all_rows;
		if (gathered) {
			ds41f_gather_share(b, all_rows, LINEAR_MAXIMUM_ROWS,
					   &own_first, &own_rows);
			ds41f_rows_select(b, view, first + own_first, own_rows);
		}
		const struct linear_range products =
			gathered ?
				ds41f_packed_rows(b, DS41F_ARRAY_COMPRESSOR_KV,
						  kv_row_bytes) :
				pool_input;
		status = own_rows == 0 ? status :
					 DS41F_THEN(ds41f_dense(
						 b, DS41F_WEIGHTS(COMPRESSOR),
						 DS41F_ARRAY(HIDDEN), products,
						 ratio == DS41F_POOLED_RATIO ?
							 DS41F_ELEMENT_F32 :
							 DS41F_ELEMENT_BF16,
						 0, 1.0f));
		status = ratio != DS41F_POOLED_RATIO || own_rows == 0 ?
				 status :
				 DS41F_THEN(ds41f_dense(
					 b, DS41F_WEIGHTS(COMPRESSOR_GATE),
					 DS41F_ARRAY(HIDDEN),
					 DS41F_ARRAY(COMPRESSOR_SCORE),
					 DS41F_ELEMENT_F32, 0, 1.0f));
		if (gathered) {
			ds41f_rows_select(b, view, first, all_rows);
			status = DS41F_THEN(ds41f_pool_rows_gather(
				b, kv_row_bytes, first + own_first, own_rows,
				ratio == DS41F_POOLED_RATIO));
		}
	}
	status = (steps & DS41F_POOL_CHUNK) == 0 || !writes_pool ?
			 status :
			 DS41F_THEN(ds41f_attention_pool_prepare(
				 &state_description, b->arena, &b->state,
				 state));
	/* the pool's entry, its key (2.1: 11 to 13) */
	if ((steps &
	     (DS41F_STATE_WRITES | DS41F_STATE_POOL | DS41F_POOL_CHUNK)) != 0 &&
	    writes_pool) {
		const bool deferred_slots = (steps & DS41F_POOL_CHUNK) != 0 &&
					    b->made->program.kind ==
						    PROGRAM_BATCH &&
					    ratio == DS41F_POOLED_RATIO;
		struct ds41f_attention_compressor_arguments compressor;
		if (deferred_slots) {
			memcpy(&compressor,
			       state[DS41F_ATTENTION_STATE_COMPRESSOR].argument,
			       sizeof(compressor));
			compressor.slot_mode =
				DS41F_ATTENTION_COMPRESSOR_SLOTS_DEFER;
			memcpy(state[DS41F_ATTENTION_STATE_COMPRESSOR].argument,
			       &compressor, sizeof(compressor));
		}
		status = DS41F_THEN(
			ds41f_put(b, &state[DS41F_ATTENTION_STATE_COMPRESSOR]));
		if (deferred_slots) {
			compressor.slot_mode =
				DS41F_ATTENTION_COMPRESSOR_SLOTS_KEEP;
			memcpy(state[DS41F_ATTENTION_STATE_COMPRESSOR].argument,
			       &compressor, sizeof(compressor));
			status = DS41F_THEN(ds41f_put(
				b, &state[DS41F_ATTENTION_STATE_COMPRESSOR]));
		}
		status = DS41F_THEN(ds41f_dense(
			b, DS41F_WEIGHTS(KEY), DS41F_ARRAY(LATENT),
			DS41F_ARRAY(KEY_INPUT), DS41F_ELEMENT_BF16, 0, 1.0f));
		status = DS41F_THEN(
			ds41f_put(b, &state[DS41F_ATTENTION_STATE_KEY]));
	}
	if ((steps & DS41F_QUERY_NORM) != 0) {
		status = b->plan->limits.official_reference != 0 ?
				 DS41F_THEN(ds41f_official_norm(
					 b, DS41F_ARRAY(QUERY_RANK),
					 DS41F_WEIGHTS(QUERY_NORM)->stored,
					 DS41F_ARRAY(QUERY_NORMED),
					 DS41F_QUERY_RANK)) :
				 DS41F_THEN(ds41f_attention_query_norm_prepare(
					 &query_norm, b->arena, &b->state,
					 ds41f_slots(b, 1)));
	}
	if ((steps & DS41F_QUERY_PRODUCTS) != 0) {
		/* a TP rank's wq_b writes its heads compact; its queries read them there when they take these rows: in
		 * this
         * call, or a chunk of one span and one piece (its queries over the span, the products over the piece); else
         * (and launches_untrimmed) the heads copied into the full rows */
		const bool compact_heads =
			b->tensor_layer != NULL &&
			!b->tensor_layer->launches_untrimmed &&
			((steps & DS41F_QUERIES) != 0 ||
			 (b->slab_count > 1u && b->half_view_first == 0 &&
			  b->chunk_rows <= DS41F_SPAN_ROWS &&
			  b->first_row == b->chunk_first &&
			  b->rows == b->chunk_rows));
		b->compact_layer = layer;
		b->compact_first_row = b->first_row;
		b->compact_rows = compact_heads ? b->rows : 0u;
		if (bf16_rows) {
			status = DS41F_THEN(ds41f_exl3_trunk_bfloat16(
				b, DS41F_WEIGHTS(QUERY_UP),
				DS41F_ARRAY(QUERY_NORMED),
				b->tensor_layer == NULL ?
					DS41F_ARRAY(HEADS) :
					b->tensor_layer->compact));
			if (b->tensor_layer != NULL && !compact_heads) {
				struct linear_range heads = DS41F_ARRAY(HEADS);
				const uint64_t into = (uint64_t)first_head *
						      DS41F_HEAD_DIMS *
						      DS41F_BF16;
				heads.byte_offset += into;
				heads.byte_count -= into;
				status = DS41F_THEN(ds41f_tensor_row_copy(
					b, b->tensor_layer->compact, heads,
					head_count * DS41F_HEAD_DIMS,
					head_count * DS41F_HEAD_DIMS,
					DS41F_HEADS_VALUE * DS41F_HEAD_DIMS));
			}
			status = !has_index ?
					 status :
					 DS41F_THEN(ds41f_exl3_trunk_bfloat16(
						 b,
						 DS41F_WEIGHTS(INDEX_QUERY_UP),
						 DS41F_ARRAY(QUERY_NORMED),
						 DS41F_ARRAY(INDEX_HEADS)));
		} else {
			status = DS41F_THEN(
				ds41f_quantize(b, DS41F_ARRAY(QUERY_NORMED),
					       DS41F_QUERY_RANK));
			status = DS41F_THEN(
				ds41f_trunk(b, DS41F_WEIGHTS(QUERY_UP),
					    b->tensor_layer == NULL ?
						    DS41F_ARRAY(HEADS) :
						    b->tensor_layer->compact,
					    NULL, ds41f_no_range));
			if (b->tensor_layer != NULL && !compact_heads) {
				struct linear_range heads = DS41F_ARRAY(HEADS);
				const uint64_t into = (uint64_t)first_head *
						      DS41F_HEAD_DIMS *
						      DS41F_BF16;
				heads.byte_offset += into;
				heads.byte_count -= into;
				status = DS41F_THEN(ds41f_tensor_row_copy(
					b, b->tensor_layer->compact, heads,
					head_count * DS41F_HEAD_DIMS,
					head_count * DS41F_HEAD_DIMS,
					DS41F_HEADS_VALUE * DS41F_HEAD_DIMS));
			}
			status = has_index ?
					 DS41F_THEN(ds41f_trunk(
						 b,
						 DS41F_WEIGHTS(INDEX_QUERY_UP),
						 DS41F_ARRAY(INDEX_HEADS), NULL,
						 ds41f_no_range)) :
					 status;
		}
	}
	if ((steps & DS41F_INDEX_WEIGHTS_PRODUCT) != 0 &&
	    has_index) { /* the indexer's weights (2.1 launch 8) */
		status = DS41F_THEN(ds41f_dense(
			b, DS41F_WEIGHTS(INDEX_WEIGHTS), DS41F_ARRAY(HIDDEN),
			DS41F_ARRAY(INDEX_WEIGHTS), DS41F_ELEMENT_BF16, 0,
			index_weights_scale));
	}
	if ((steps & DS41F_QUERIES) !=
	    0) { /* the queries (2.1 launch 9): the state's QUERIES launch alone */
		struct ds41f_attention_queries_description read = queries;
		if (b->compact_rows != 0 &&
		    b->compact_layer ==
			    layer) { /* wq_b's compact rows: these rows, or refused */
			read.heads =
				b->tensor_layer != NULL &&
						b->first_row ==
							b->compact_first_row &&
						b->rows == b->compact_rows ?
					b->tensor_layer->compact :
					ds41f_no_range;
			read.heads_compact = 1;
		}
		if (b->plan->limits.official_reference != 0) {
			const struct ds41f_official_rope_description rotated = {
				DS41F_ARRAY(HEADS),
				{ DS41F_ARRAY(ROTATION).byte_offset +
					  (uint64_t)(ratio != 0 ? 1u : 0u) *
						  (uint64_t)
							  DS41F_OFFICIAL_ROPE_WIDTH *
						  sizeof(float),
				  (size_t)DS41F_OFFICIAL_ROPE_WIDTH *
					  sizeof(float) },
				DS41F_ARRAY(QUERIES),
				DS41F_HEAD_DIMS,
				DS41F_HEADS_VALUE,
				1,
				0
			};
			status = DS41F_THEN(ds41f_official_rope_prepare(
				&rotated, b->arena, b->official.rope,
				ds41f_slots(b, 1)));
			if (has_index && status == DS41F_OK) {
				struct kernel_launch launch;
				status = ds41f_attention_queries_prepare(
					&read, b->arena, &b->state, &launch);
				launch.kernel = b->official.index_queries;
				launch.grid[0] = DS41F_INDEX_HEADS;
				launch.grid[1] =
					launch.grid[LAUNCH_DIMENSIONS - 1u] = 1;
				launch.block[0] = DS41F_OFFICIAL_STATE_THREADS;
				status = DS41F_THEN(ds41f_put(b, &launch));
			}
		} else {
			status = DS41F_THEN(ds41f_attention_queries_prepare(
				&read, b->arena, &b->state, ds41f_slots(b, 1)));
		}
	}
	if ((steps & (DS41F_CORE | DS41F_SCAN | DS41F_CORE_ALONE)) !=
	    0) { /* the scan, the core (chunk_core: either alone) */
		if (has_index && (steps & DS41F_CORE_ALONE) == 0 &&
		    b->plan->limits.official_reference != 0) {
			status = DS41F_THEN(ds41f_official_index_prepare(
				&scan, b->arena, b->official.index,
				&index_launches[0]));
			index_launch_count = 1;
		} else {
			status = has_index && (steps & DS41F_CORE_ALONE) == 0 ?
					 DS41F_THEN(ds41f_index_scan_prepare(
						 &scan, b->arena, &b->scan,
						 index_launches,
						 &index_launch_count)) :
					 status;
		}
		for (uint32_t launch = 0;
		     status == DS41F_OK && launch < index_launch_count;
		     launch++) {
			status = ds41f_put(b, &index_launches[launch]);
		}
		/* a TP rank's core writes its heads into the compact rows alone when wo_a takes them next from there:
		 * in this
         * call, or a chunk of one piece (its slabs' cores at their rows in, wo_a over the piece); else (and
         * launches_untrimmed) the full rows and the copy */
		struct ds41f_attention_core_description written =
			core; /* core_heads: more blocks, the same bits */
		/* a full program's decoder layer: the rows the engine's prefill cone marks past it run no core */
		written.cone_layer =
			b->made->program.kind == PROGRAM_FULL && !is_draft &&
					layer >= DS41F_DECODER_FIRST_LAYER &&
					layer < DS41F_LAYERS ?
				layer :
				0u;
		/* core_compact_halves (ds41f_tp_stage.h): a split half's core too (a half selected: its wo_a takes the
         * rows from the half's first; a chunk's core of a program with half views keeps the copy, its wo_a may run a
         * half at a time: a non-pipelined split layer) */
		const bool halves_compact =
			b->tp_description != NULL && b->tensor_layer != NULL &&
			b->chunk_view != 0 &&
			b->tp_description->stages[b->card]
				->description.core_compact_halves;
		const bool chunk_compact =
			b->slab_count > 1u && ds41f_piece_count_of(b) == 1u &&
			(b->half_view_first == 0 || halves_compact);
		b->core_compact =
			b->tensor_layer != NULL &&
			!b->tensor_layer->launches_untrimmed &&
			((steps & DS41F_OUTPUT_PRODUCTS) != 0 || chunk_compact);
		b->core_compact_layer = layer;
		b->core_compact_first = (steps & DS41F_OUTPUT_PRODUCTS) == 0 &&
							chunk_compact ?
						b->chunk_first :
						b->first_row;
		if (b->core_compact && (steps & DS41F_SCAN) == 0) {
			const uint64_t row_bytes = (uint64_t)head_count *
						   DS41F_HEAD_DIMS * DS41F_BF16;
			const uint64_t into =
				(uint64_t)(b->first_row -
					   b->core_compact_first) *
				row_bytes;
			written.output =
				b->tensor_layer->compact.byte_count >=
						into + b->rows * row_bytes ?
					(struct linear_range){
						b->tensor_layer->compact
								.byte_offset +
							into,
						b->rows * row_bytes
					} :
					ds41f_no_range;
			written.output_compact = 1;
		}
		if ((steps & DS41F_SCAN) == 0 &&
		    b->plan->limits.official_reference != 0) {
			status = DS41F_THEN(ds41f_official_attention_prepare(
				&written,
				DS41F_OFFICIAL_ATTENTION_QUERY_BFLOAT16 |
					DS41F_OFFICIAL_ATTENTION_RING_BFLOAT16,
				b->arena, b->official.attention,
				&core_launches[0]));
			core_launch_count = 1;
		} else {
			status =
				(steps & DS41F_SCAN) != 0 ?
					status :
					DS41F_THEN(ds41f_attention_core_prepare(
						&written, b->arena, &b->core,
						core_launches,
						&core_launch_count));
		}
		for (uint32_t launch = 0;
		     status == DS41F_OK && launch < core_launch_count;
		     launch++) {
			status = ds41f_put(b, &core_launches[launch]);
		}
	}
	if ((steps & DS41F_OUTPUT_PRODUCTS) != 0) {
		if (b->plan->limits.official_reference != 0)
			status = DS41F_THEN(ds41f_official_roundtrip(
				b, DS41F_ARRAY(ATTENTION_OUTPUT),
				DS41F_HEADS_VALUE * DS41F_HEAD_DIMS));
		const bool core_compact = b->tensor_layer != NULL &&
					  b->core_compact &&
					  b->core_compact_layer == layer;
		if (core_compact) /* the core's compact rows from this step's first row, or refused */
			groups.values = b->first_row == b->core_compact_first ?
						b->tensor_layer->compact :
						ds41f_no_range;
		if (b->tensor_layer != NULL && !core_compact) {
			struct linear_range core_values =
				DS41F_ARRAY(ATTENTION_OUTPUT);
			const uint64_t into = (uint64_t)first_head *
					      DS41F_HEAD_DIMS * DS41F_BF16;
			core_values.byte_offset += into;
			core_values.byte_count -= into;
			groups.values = b->tensor_layer->compact;
			status = DS41F_THEN(ds41f_tensor_row_copy(
				b, core_values, groups.values,
				head_count * DS41F_HEAD_DIMS,
				DS41F_HEADS_VALUE * DS41F_HEAD_DIMS,
				head_count * DS41F_HEAD_DIMS));
		}
		status = DS41F_THEN(ds41f_matrix_of(
			b, DS41F_WEIGHTS(OUTPUT_GROUPS),
			LINEAR_FAMILY_TRUNK_BFLOAT16, &groups.matrix));
		const bool exact_dense = b->tensor_layer != NULL &&
					 b->tensor_layer->exact_dense;
		if (exact_dense) {
			/* the rank's two groups (full-K wo_a rows) into its owner slot; all eight gathered row-major;
             * wo_b's 1280 owned outputs over the full K = 8192 into the slot again; the complete sublayer output
             * gathered */
			const uint32_t group_width =
				groups.matrix.shape.output_size;
			const uint32_t owned =
				DS41F_WEIGHTS(OUTPUT)->shape.output_size;
			/* a piece of a chunked step (more than 8 rows): the products over the piece, the gathers a slab
			 * at
		     * a time */
			const bool wide = b->rows > LINEAR_MAXIMUM_ROWS;
			groups.output = wide ? ds41f_dense_own(b, group_width) :
					       ds41f_dense_slot(b, group_width);
			status = DS41F_THEN(ds41f_groups_product(
				b, &groups,
				DS41F_WEIGHTS(OUTPUT_GROUPS)->fold_reference));
			status = DS41F_THEN(
				wide ? ds41f_dense_gather_slabs(
					       b, 0, group_width,
					       DS41F_ARRAY(GROUPS)) :
				       ds41f_dense_gather(b, 0, group_width,
							  DS41F_ARRAY(GROUPS)));
			const struct linear_range own_outputs =
				wide ? ds41f_dense_own(b, owned) :
				       ds41f_dense_slot(b, owned);
			if (is_exl3) {
				status = DS41F_THEN(ds41f_exl3_trunk_bfloat16(
					b, DS41F_WEIGHTS(OUTPUT),
					DS41F_ARRAY(GROUPS), own_outputs));
			} else {
				status = DS41F_THEN(ds41f_quantize(
					b, DS41F_ARRAY(GROUPS),
					DS41F_WEIGHTS(OUTPUT)
						->shape.input_size));
				status = DS41F_THEN(ds41f_trunk(
					b, DS41F_WEIGHTS(OUTPUT), own_outputs,
					NULL, ds41f_no_range));
			}
			status = DS41F_THEN(
				wide ? ds41f_dense_gather_slabs(
					       b, 1, owned,
					       DS41F_ARRAY(SUBLAYER_OUTPUT)) :
				       ds41f_dense_gather(
					       b, 1, owned,
					       DS41F_ARRAY(SUBLAYER_OUTPUT)));
		} else if (b->plan->limits.official_reference == 0 && is_thin &&
			   b->arithmetic_class ==
				   ARITHMETIC_CLASS_SM_120_NATIVE) {
			const struct ds41f_attention_groups_description
				fused_groups = { groups.matrix,
						 groups.values,
						 groups.output,
						 DS41F_ARRAY(ROW_CODES),
						 DS41F_ARRAY(ROW_EXPONENTS),
						 DS41F_ARRAY(ROW_OPERANDS),
						 b->rows,
						 groups.segment_count };
			status = DS41F_THEN(ds41f_attention_groups_prepare(
				&fused_groups, b->arena, &b->linear, &b->output,
				ds41f_slots(b, 1)));
		} else {
			status = DS41F_THEN(ds41f_groups_product(
				b, &groups,
				DS41F_WEIGHTS(OUTPUT_GROUPS)->fold_reference));
			/* an EXL3 layer's wo_b (exl3_trunk_bfloat16 below) reads the groups' BF16
			 * rows: no quantization of
             * them (launches_untrimmed: as before) */
			const bool unread =
				bf16_rows && !fused_projection &&
				b->tensor_layer != NULL &&
				!b->tensor_layer->launches_untrimmed;
			status = unread ? status :
					  DS41F_THEN(ds41f_quantize(
						  b, DS41F_ARRAY(GROUPS),
						  DS41F_WEIGHTS(OUTPUT)
							  ->shape.input_size));
		}
		if (exact_dense) {
			/* wo_b and its gather are above */
		} else if (fused_projection) {
			struct ds41f_attention_projection_description
				projection = {
					.codes = DS41F_ARRAY(ROW_CODES),
					.exponents = DS41F_ARRAY(ROW_EXPONENTS),
					.operands = DS41F_ARRAY(ROW_OPERANDS),
					.output = DS41F_ARRAY(SUBLAYER_OUTPUT),
					.streams = DS41F_ARRAY(STREAMS),
					.post = DS41F_ARRAY(POST),
					.comb = DS41F_ARRAY(COMB),
					.row_count = b->rows
				};
			status = DS41F_THEN(ds41f_matrix_of(
				b, DS41F_WEIGHTS(OUTPUT), DS41F_TRUNK,
				&projection.matrix));
			status = DS41F_THEN(ds41f_attention_projection_prepare(
				&projection, b->arena, &b->linear, &b->output,
				ds41f_slots(b, 1)));
		} else if (bf16_rows) {
			status = DS41F_THEN(ds41f_exl3_trunk_bfloat16(
				b, DS41F_WEIGHTS(OUTPUT), DS41F_ARRAY(GROUPS),
				DS41F_ARRAY(SUBLAYER_OUTPUT)));
		} else {
			status = DS41F_THEN(
				ds41f_trunk(b, DS41F_WEIGHTS(OUTPUT),
					    DS41F_ARRAY(SUBLAYER_OUTPUT), NULL,
					    ds41f_no_range));
		}
	}
	if ((steps & (DS41F_SITE_POST | DS41F_SITE_SUM)) != 0 &&
	    b->tensor_layer != NULL && !b->tensor_layer->exact_dense)
		status = DS41F_THEN(ds41f_tensor_complete_slabs(
			b, 0, DS41F_ARRAY(SUBLAYER_OUTPUT), ds41f_no_range));
	status =
		(steps & (DS41F_SITE_POST | DS41F_SITE_POST_ONLY)) != 0 &&
				!fused_projection ?
			DS41F_THEN(ds41f_put(b, &site[DS41F_HYPER_SITE_POST])) :
			status;
	if (front_branch &&
	    status ==
		    DS41F_OK) /* the side run (the coefficient front), its resume and its join (the post) */
		ds41f_front_branch_mark(b, side_launch);
	return status;
}

/* The steps of layer's n-gram rows (an n-gram layer's) and attention site, in that order. */
MUST_CHECK result_status ds41f_layer_steps(struct ds41f_builder *b,
					   uint32_t layer, uint32_t source,
					   uint32_t group, uint32_t steps)
{
	const bool has_engram = (DS41F_ENGRAM_LAYERS >> layer & 1u) != 0;
	result_status status = has_engram && (steps & DS41F_ENGRAM_STEPS) != 0 ?
				       ds41f_engram_launches(b, layer, steps) :
				       DS41F_OK;
	/* a full program's decoder layer after the first: its wide trunks leave the rows whose output at the layer
     * before nothing reads (the engine's prefill cone: none of the layer's own outputs or window rows of them is
     * read) */
	const uint32_t cone_before = b->trunk_cone_layer;
	b->trunk_cone_layer =
		b->made->program.kind == PROGRAM_FULL &&
				layer > DS41F_DECODER_FIRST_LAYER &&
				layer < DS41F_LAYERS ?
			layer - 1u :
			0u;
	status = (steps & ~(uint32_t)DS41F_ENGRAM_STEPS) != 0 ?
			 DS41F_THEN(attention_launches(b, layer, source, group,
						       steps)) :
			 status;
	b->trunk_cone_layer = cone_before;
	return status;
}
