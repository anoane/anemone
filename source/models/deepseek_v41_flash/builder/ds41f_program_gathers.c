/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_gathers.c: dense slots, exchange phases, gathers, cone bounds, pool gathers, TP completion
 * (split from
 * model_v41_program.c by whole items).
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

/* exact_dense: rank r's owner-major slot of the gather input, rows of `width` bfloat16 values. */
struct linear_range ds41f_dense_slot(const struct ds41f_builder *b,
				     uint32_t width)
{
	const uint64_t bytes =
		(uint64_t)b->rows * width * LINEAR_BFLOAT16_BYTES;
	return (struct linear_range){ b->tensor_layer->dense_input.byte_offset +
					      b->tensor_layer->rank * bytes,
				      bytes };
}

/* exact_dense: every rank's slot of `width` columns gathered into the row-major output of width x ranks columns. */
MUST_CHECK result_status ds41f_dense_gather(struct ds41f_builder *b,
					    uint32_t site, uint32_t width,
					    struct linear_range output)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	const uint32_t bytes =
		b->rows * width * tensor->rank_count * LINEAR_BFLOAT16_BYTES;
	const struct linear_range input = { tensor->dense_input.byte_offset,
					    bytes };
	return ds41f_exact_gather_bank(b, site,
				       tensor->dense_ring_offsets[site],
				       tensor->dense_ring_bank_bytes[site], 0,
				       input, output, bytes, b->rows);
}

/* exact_dense, more than 8 rows (a chunked step's piece): the rank's own outputs [rows][width] behind the slot region
 * of the thin gathers (that region the guard below the virtual bases). */
#define DENSE_SLOT_REGION_BYTES                             \
	((uint64_t)LINEAR_MAXIMUM_ROWS * DS41F_WIDEST_ROW * \
	 LINEAR_BFLOAT16_BYTES)
struct linear_range ds41f_dense_own(const struct ds41f_builder *b,
				    uint32_t width)
{
	return (struct linear_range){ b->tensor_layer->dense_input.byte_offset +
					      DENSE_SLOT_REGION_BYTES,
				      (uint64_t)b->rows * width *
					      LINEAR_BFLOAT16_BYTES };
}

/* exact_dense, a piece: each slab of 8 rows (the last its remaining rows) gathered from every rank's own outputs into
 * its rows of the row-major output. A rank reads only its own range of a gather's owner-major input, so its input
 * base is its slab's own rows less its owner prefix (inside the slot region's guard). */
MUST_CHECK result_status ds41f_dense_gather_slabs(struct ds41f_builder *b,
						  uint32_t site, uint32_t width,
						  struct linear_range output)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	const uint32_t rows = b->rows;
	const uint64_t own_row = (uint64_t)width * LINEAR_BFLOAT16_BYTES;
	const struct linear_range own = ds41f_dense_own(b, width);
	if ((uint64_t)tensor->rank * LINEAR_MAXIMUM_ROWS * own_row >
		    DENSE_SLOT_REGION_BYTES ||
	    output.byte_count < (uint64_t)rows * own_row * tensor->rank_count)
		return STATUS_INVALID_ARGUMENT;
	result_status status = DS41F_OK;
	for (uint32_t at = 0; at < rows && status == DS41F_OK;
	     at += LINEAR_MAXIMUM_ROWS) {
		const uint32_t count = rows - at < LINEAR_MAXIMUM_ROWS ?
					       rows - at :
					       LINEAR_MAXIMUM_ROWS;
		const uint32_t bytes =
			(uint32_t)(count * own_row * tensor->rank_count);
		const struct linear_range input = {
			own.byte_offset + at * own_row -
				(uint64_t)tensor->rank * count * own_row,
			bytes
		};
		const struct linear_range into = {
			output.byte_offset + at * own_row * tensor->rank_count,
			bytes
		};
		b->rows = count; /* a gather's bank is its slab's */
		status = ds41f_exact_gather_bank(
			b, site, tensor->dense_ring_offsets[site],
			tensor->dense_ring_bank_bytes[site], 0, input, into,
			bytes, count);
	}
	b->rows = rows;
	return status;
}

/* Local product rounding is part of the relaxed TP contract. The unchanged post reads a complete BF16
 * result. */
/* The sums' phase of the layer (tp_exchange_records.h): 8 l + 2, and 8 l + 4 in split_overlap's second half; its
 * pipeline's gathers below and between them, each on the channel no other exchange uses beside it: the index gathers
 * (half A: attention site, half B: FFN site), the router's of half A (FFN site, beside half B's attention sums) and of
 * half B (attention site, beside half A's FFN sums). */
#define SUMS_PHASE 2u
/* split_overlap's second half: its sums so many phases on (8 l + 2 + 2 h) */
#define SUMS_HALF_PHASES 2u
#define GATHER_ROUTER_FIRST_PHASE 1u
#define GATHER_ROUTER_SECOND_PHASE 5u
/* pool_gather's of a half, at the layer before's phase 6 (behind its channel's sums and router gather, ahead of this
 * layer's index gather): half A on the attention site's channel (beside B's FFN sums of the layer before), half B on
 * the FFN site's (beside A's attention sums) */
#define GATHER_POOL_PHASE 6u
static uint32_t exchange_phase(const struct ds41f_builder *b)
{
	return DS41F_TP_EXCHANGE_LAYER_PHASES * b->tensor_layer->layer_index +
	       SUMS_PHASE +
	       (b->half_view_first != 0 &&
				b->chunk_view == b->half_view_first + 1u ?
			SUMS_HALF_PHASES :
			0u);
}

/* A copy engine's copy of bytes from the card's arena to a place in it (word EXCHANGE_OWN_ARENA) or in the
 * arena a peer_connect address word names (backend_seam.h). */
MUST_CHECK static result_status copy_put(struct ds41f_builder *b,
					 uint64_t source, uint64_t word,
					 uint64_t offset, uint64_t bytes,
					 uint32_t branch)
{
	const struct exchange_copy_arguments copy = {
		{ EXCHANGE_OWN_ARENA, source }, { word, offset }, bytes
	};
	struct kernel_launch launch = { .kernel = BACKEND_COPY,
					.argument_byte_count = sizeof(copy),
					.branch = branch };
	memcpy(launch.argument, &copy, sizeof(copy));
	return ds41f_put(b, &launch);
}

/* The branch of the step-th of an exchange's copies (rank + 1, rank + 2, ..., the rank's own last): the peers' side
 * runs at once (SIDE, then SIDE_MORE), the last on the main run (RESUME); the barrier behind them JOINs. */
static uint32_t copy_branch(uint32_t step, uint32_t steps)
{
	return step + 1u == steps ? LAUNCH_RESUME :
	       step == 0	  ? LAUNCH_SIDE :
				    LAUNCH_SIDE_MORE;
}

/* split_overlap's pipeline's gathers apply to the selected half: copy-engine sums (direct routes, four ranks,
 * BF16) on
 * both sites' channels and whole quarters of slabs. */
bool ds41f_gathers_apply(const struct ds41f_builder *b)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	bool apply = tensor != NULL &&
		     tensor->split_overlap >= DS41F_SPLIT_PIPELINE &&
		     b->half_view_first != 0 &&
		     b->chunk_view >= b->half_view_first &&
		     tensor->shared_channels && !b->tensor_local_stub &&
		     !tensor->parallel_publish &&
		     tensor->wide_exchange >= DS41F_TP_EXCHANGE_CE_OWNER_SUMS &&
		     tensor->rank_count == DS41F_TP4_RANKS &&
		     b->chunk_rows % (DS41F_TP4_RANKS * LINEAR_MAXIMUM_ROWS) ==
			     0;
	for (uint32_t site = 0; site < DS41F_TP2_SITES && apply; site++)
		apply = tensor->collectives[site].arguments.incoming_host_mask ==
				0 &&
			tensor->collectives[site].arguments.payload_format ==
				ORDERED_PUSH_BF16;
	return apply;
}

/* tp_exchange.h's wide form: its launches in their order */
enum wide_exchange_launch {
	WIDE_EXCHANGE_PUBLISH,
	WIDE_EXCHANGE_READY,
	WIDE_EXCHANGE_SUM,
	WIDE_EXCHANGE_DONE
};

/* A gather of the selected half's rows: each range's copies to every peer (on the main run: no fork beside a split
 * part's), then READY and DONE on the site's channel at exchange phase `phase` (tp_exchange_records.h: 8 l + p): every
 * rank holds every rank's rows. */
MUST_CHECK result_status
ds41f_rows_gather(struct ds41f_builder *b, uint32_t site, uint32_t phase,
		  const struct ds41f_gather_range *ranges, uint32_t range_count)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	struct ordered_push_description collective = tensor->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * DS41F_HIDDEN;
	arguments->step_offset = DS41F_ARRAY(STEP).byte_offset;
	struct kernel_launch wide[TP_EXCHANGE_WIDE_LAUNCHES];
	result_status status = tp_exchange_wide_prepare(
		&collective, phase, TP_EXCHANGE_PHASES, b->tensor_ce_kernels,
		TP_EXCHANGE_FORM_CE_OWNER_SUMS, wide);
	const uint64_t word_of_rank =
		arguments->peer_table_offset +
		offsetof(struct ordered_collective_peer, arena_address);
	for (uint32_t step = 1; step < DS41F_TP4_RANKS && status == DS41F_OK;
	     step++) {
		const uint32_t rank = (tensor->rank + step) % DS41F_TP4_RANKS;
		for (uint32_t range = 0;
		     range < range_count && status == DS41F_OK; range++)
			status =
				ranges[range].bytes == 0 ?
					DS41F_OK :
					copy_put(
						b, ranges[range].own,
						word_of_rank +
							rank * sizeof(struct ordered_collective_peer),
						ranges[range].peers[rank],
						ranges[range].bytes,
						LAUNCH_CHAIN);
	}
	status = DS41F_THEN(ds41f_put(b, &wide[WIDE_EXCHANGE_READY]));
	return DS41F_THEN(ds41f_put(b, &wide[WIDE_EXCHANGE_DONE]));
}

/* prefill_dead_halves (ds41f_tp_stage.h full_cone_behind): half h (0 or 1) of a full program of
 * STEP_FULL_GRANULE_ROWS rows or more has no row whose output at decoder layer `layer` the cone reads: its
 * last row (A's: B's rows nearer the end) lies at least the bound before its prompt's cone end's last row, so its
 * rows' cone counts are at most that row's (ds41f_geometry.h DS41F_CONE_LAYERS). */
static uint32_t cone_bound_of(const struct ds41f_builder *b, uint32_t layer)
{
	const uint32_t behind = b->tp_description == NULL ||
						b->tensor_layer == NULL ?
					0u :
					b->tp_description->stages[b->card]
						->description.full_cone_behind;
	/* whole granules of 256 rows: the exact plan's chunks before a tail (a tail program's lie in the tail) */
	return b->made->program.kind != PROGRAM_FULL ||
			       layer < DS41F_DECODER_FIRST_LAYER ||
			       layer >= DS41F_LAYERS ||
			       b->program_rows < STEP_FULL_GRANULE_ROWS ||
			       b->program_rows % STEP_FULL_GRANULE_ROWS != 0 ||
			       b->half_view_first == 0 ?
		       0u :
		       behind;
}
bool ds41f_half_dead(const struct ds41f_builder *b, uint32_t layer,
		     uint32_t half)
{
	const uint32_t behind = cone_bound_of(b, layer);
	const uint32_t after = half == 0 ? b->program_rows - b->half_rows : 0u;
	return behind != 0 && half <= 1u &&
	       DS41F_CONE_LAYERS(behind + after) <=
		       layer - DS41F_DECODER_FIRST_LAYER;
}

/* prefill_dead_halves 2 (ds41f_tp_stage.h dead_rows_trimmed): of the selected rows of a split decoder layer's
 * half, the first whose output at the tensor layer the cone may read (half_dead's bound; 0: every row's). */
static uint32_t live_first_row(const struct ds41f_builder *b)
{
	const uint32_t layer =
		b->tensor_layer == NULL ? 0u : b->tensor_layer->layer_index;
	const uint32_t behind = cone_bound_of(b, layer);
	if (behind == 0 ||
	    !b->tp_description->stages[b->card]->description.dead_rows_trimmed ||
	    b->chunk_view < b->half_view_first)
		return 0;
	uint32_t row = 0;
	while (row < b->rows && DS41F_CONE_LAYERS(behind + b->program_rows -
						  1u - (b->first_row + row)) <=
					layer - DS41F_DECODER_FIRST_LAYER)
		row++;
	return row;
}

/* The selected half of split_overlap's chunk: 0 (A) or 1 (B); CHUNK_HALVES without one selected. */
uint32_t ds41f_selected_half(const struct ds41f_builder *b)
{
	return b->half_view_first != 0 && b->chunk_view >= b->half_view_first ?
		       b->chunk_view - b->half_view_first :
		       DS41F_CHUNK_HALVES;
}

/* pool_gather (ds41f_tp_stage.h) applies to the selected half: split_overlap's pipeline's gathers, its whole rows
 * selected (one piece). */
bool ds41f_pool_gather_applies(const struct ds41f_builder *b)
{
	return b->tensor_layer != NULL && b->tp_description != NULL &&
	       b->tensor_layer->layer_index != 0 &&
	       b->tp_description->stages[b->card]->description.pool_gather &&
	       ds41f_gathers_apply(b) && b->first_row == b->chunk_first &&
	       b->rows == b->chunk_rows;
}

/* pool_gather: rows first .. first + rows of the compressor's products (wkv packed at kv_row_bytes, wgate's scores)
 * gathered from the ranks that computed them (gather_share's parts). */
MUST_CHECK result_status ds41f_pool_rows_gather(struct ds41f_builder *b,
						uint64_t kv_row_bytes,
						uint32_t first, uint32_t rows,
						bool scores)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	static const enum ds41f_array arrays[] = {
		DS41F_ARRAY_COMPRESSOR_KV, DS41F_ARRAY_COMPRESSOR_SCORE
	};
	struct ds41f_gather_range ranges[sizeof(arrays) / sizeof(arrays[0])];
	for (uint32_t range = 0; range < sizeof(arrays) / sizeof(arrays[0]);
	     range++) {
		const uint64_t row_bytes =
			range == 0 ? kv_row_bytes :
				     ds41f_array_bytes[arrays[range]].row_bytes;
		const uint64_t into = (uint64_t)first * row_bytes;
		ranges[range] = (struct ds41f_gather_range){
			b->plan->cards[b->card].array_offset[arrays[range]] +
				into,
			range == 0 || scores ? (uint64_t)rows * row_bytes : 0u,
			{ 0 }
		};
		for (uint32_t rank = 0; rank < DS41F_TP4_RANKS; rank++) {
			const uint32_t card =
				tensor->card - tensor->rank + rank;
			ranges[range].peers[rank] =
				b->tp_description->views[card]
					->cards[card]
					.array_offset[arrays[range]] +
				into;
		}
	}
	const bool second = b->chunk_view == b->half_view_first + 1u;
	return ds41f_rows_gather(b, second ? 1u : 0u,
				 DS41F_TP_EXCHANGE_LAYER_PHASES *
						 (tensor->layer_index - 1u) +
					 GATHER_POOL_PHASE,
				 ranges, sizeof(arrays) / sizeof(arrays[0]));
}

/* wide_exchange CE_OWNER_SUMS, the attention site (no routed sums: the BF16 values are the push's payload
 * as they
 * are): rank d's quarter of the piece's prepared rows copied into every rank's inbox for it (slot 0, tp_exchange.h),
 * READY, the rank's quarter summed into its output array, that quarter copied into every other rank's output array,
 * DONE, and the BF16 rows copied back into values: the owner sums' bits, the bytes moved by copy
 * engines. */
MUST_CHECK static result_status tensor_complete_ce(struct ds41f_builder *b,
						   uint32_t site,
						   struct linear_range values)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	struct ordered_push_description collective = tensor->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * DS41F_HIDDEN;
	arguments->step_offset = b->view != 0 ? DS41F_ARRAY(STEP).byte_offset :
						arguments->step_offset;
	const uint64_t quarter = (uint64_t)b->rows / DS41F_TP4_RANKS *
				 DS41F_HIDDEN * LINEAR_BFLOAT16_BYTES;
	const uint64_t word_of_rank =
		arguments->peer_table_offset +
		offsetof(struct ordered_collective_peer, arena_address);
	struct kernel_launch wide[TP_EXCHANGE_WIDE_LAUNCHES];
	result_status status = tp_exchange_wide_prepare(
		&collective, exchange_phase(b), TP_EXCHANGE_PHASES,
		b->tensor_ce_kernels, TP_EXCHANGE_FORM_CE_OWNER_SUMS, wide);
	/* dead_rows_trimmed: each quarter's rows from the half's first live one (live_first_row: a quarter of none
     * copies nothing; the side runs over the copies that remain) */
	const uint32_t live = live_first_row(b),
		       quarter_rows = b->rows / DS41F_TP4_RANKS;
	uint64_t skipped[DS41F_TP4_RANKS];
	uint32_t moved = 0;
	for (uint32_t rank = 0; rank < DS41F_TP4_RANKS; rank++) {
		const uint32_t first = rank * quarter_rows;
		skipped[rank] = (uint64_t)(live <= first ? 0u :
					   live - first < quarter_rows ?
							   live - first :
							   quarter_rows) *
				DS41F_HIDDEN * LINEAR_BFLOAT16_BYTES;
		moved += skipped[rank] < quarter ? 1u : 0u;
	}
	/* the quarters into the inboxes */
	for (uint32_t step = 1, put_copies = 0;
	     step <= DS41F_TP4_RANKS && status == DS41F_OK; step++) {
		const uint32_t rank = (tensor->rank + step) % DS41F_TP4_RANKS;
		if (skipped[rank] == quarter)
			continue;
		status = copy_put(
			b, values.byte_offset + rank * quarter + skipped[rank],
			rank == tensor->rank ?
				EXCHANGE_OWN_ARENA :
				word_of_rank +
					rank * sizeof(struct ordered_collective_peer),
			(rank == tensor->rank ?
				 arguments->region_offset :
				 tensor->peer_regions[site][rank]) +
				arguments->inbox_offset +
				(uint64_t)tensor->rank *
					arguments->inbox_rank_stride +
				rank * quarter + skipped[rank],
			quarter - skipped[rank],
			moved > 1u ? copy_branch(put_copies, moved) :
				     LAUNCH_CHAIN);
		put_copies++;
	}
	wide[WIDE_EXCHANGE_READY].branch = moved > 1u ? LAUNCH_JOIN :
							LAUNCH_CHAIN;
	status = DS41F_THEN(ds41f_put(b, &wide[WIDE_EXCHANGE_READY]));
	status = DS41F_THEN(ds41f_put(
		b, &wide[WIDE_EXCHANGE_SUM])); /* the rank's quarter summed */
	const uint64_t own_skipped = skipped[tensor->rank];
	for (uint32_t step = 1; step < DS41F_TP4_RANKS && status == DS41F_OK &&
				own_skipped < quarter;
	     step++) {
		const uint32_t rank = (tensor->rank + step) % DS41F_TP4_RANKS;
		status = copy_put(
			b,
			arguments->region_offset + arguments->output_offset +
				tensor->rank * quarter + own_skipped,
			word_of_rank +
				rank * sizeof(struct ordered_collective_peer),
			tensor->peer_regions[site][rank] +
				arguments->output_offset +
				tensor->rank * quarter + own_skipped,
			quarter - own_skipped,
			copy_branch(step - 1u, DS41F_TP4_RANKS - 1u));
	}
	wide[WIDE_EXCHANGE_DONE].branch = own_skipped < quarter ? LAUNCH_JOIN :
								  LAUNCH_CHAIN;
	status = DS41F_THEN(ds41f_put(b, &wide[WIDE_EXCHANGE_DONE]));
	const struct tp2_join_description back = {
		.source = { arguments->region_offset + arguments->output_offset,
			    quarter * DS41F_TP4_RANKS },
		.output = values,
		.width = DS41F_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = DS41F_HIDDEN,
		.output_row_stride = DS41F_HIDDEN,
		.operation = TP2_JOIN_BFLOAT16_COPY
	};
	return DS41F_THEN(tp2_join_prepare(
		&back, b->arena, b->tensor_join_kernel, ds41f_slots(b, 1)));
}

/* wide_exchange CE_RINGS, the FFN site of a piece of more than 8 rows: values plus the routed sums rounded to
 * BF16
 * into the collective's input array in the ring's owner-major order (tp2_join's RING_PACK), each rank's segment d
 * copied into rank d's inbox (slot 0), READY, the rank's segment added in the ring's order into its output array,
 * copied into the other ranks' output arrays, DONE, and the rows unpacked back into values: each slab's
 * BF16 ring
 * sum, the same bits, the bytes moved by copy engines. */
MUST_CHECK result_status ds41f_tensor_complete_ce_rings(
	struct ds41f_builder *b, uint32_t site, struct linear_range values,
	struct linear_range routed)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	struct ordered_push_description collective = tensor->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * DS41F_HIDDEN;
	arguments->step_offset = b->view != 0 ? DS41F_ARRAY(STEP).byte_offset :
						arguments->step_offset;
	const uint64_t segment = (uint64_t)b->rows * DS41F_HIDDEN /
				 DS41F_TP4_RANKS * LINEAR_BFLOAT16_BYTES;
	const uint64_t staging =
		arguments->region_offset + arguments->input_offset;
	const uint64_t output =
		arguments->region_offset + arguments->output_offset;
	const uint64_t word_of_rank =
		arguments->peer_table_offset +
		offsetof(struct ordered_collective_peer, arena_address);
	struct kernel_launch wide[TP_EXCHANGE_WIDE_LAUNCHES];
	result_status status = tp_exchange_wide_prepare(
		&collective, exchange_phase(b), TP_EXCHANGE_PHASES,
		b->tensor_ce_ring_kernels, TP_EXCHANGE_FORM_CE_OWNER_RING_SUMS,
		wide);
	const struct tp2_join_description pack = {
		.source = values,
		.routed_sums = routed,
		.output = { staging, segment * DS41F_TP4_RANKS },
		.width = DS41F_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = DS41F_HIDDEN,
		.routed_row_stride = routed.byte_count == 0 ? 0 : DS41F_HIDDEN,
		.output_row_stride = DS41F_HIDDEN,
		.operation = TP2_JOIN_RING_PACK
	};
	status = DS41F_THEN(tp2_join_prepare(
		&pack, b->arena, b->tensor_join_kernel, ds41f_slots(b, 1)));
	/* dead_rows_trimmed: each segment's slabs from the one of the half's first live row on (live_first_row) */
	const uint64_t skipped =
		(uint64_t)(live_first_row(b) / TP2_JOIN_RING_SLAB_ROWS) *
		TP2_JOIN_RING_SLAB_ROWS * DS41F_HIDDEN / DS41F_TP4_RANKS *
		LINEAR_BFLOAT16_BYTES;
	if (skipped >= segment)
		return STATUS_INVALID_ARGUMENT; /* a half of no live row leaves its sums (half_dead) */
	for (uint32_t step = 1; step <= DS41F_TP4_RANKS && status == DS41F_OK;
	     step++) { /* the segments into the inboxes */
		const uint32_t rank = (tensor->rank + step) % DS41F_TP4_RANKS;
		status = copy_put(
			b, staging + rank * segment + skipped,
			rank == tensor->rank ?
				EXCHANGE_OWN_ARENA :
				word_of_rank +
					rank * sizeof(struct ordered_collective_peer),
			(rank == tensor->rank ?
				 arguments->region_offset :
				 tensor->peer_regions[site][rank]) +
				arguments->inbox_offset +
				(uint64_t)tensor->rank *
					arguments->inbox_rank_stride +
				rank * segment + skipped,
			segment - skipped,
			copy_branch(step - 1u, DS41F_TP4_RANKS));
	}
	wide[WIDE_EXCHANGE_READY].branch = LAUNCH_JOIN;
	status = DS41F_THEN(ds41f_put(b, &wide[WIDE_EXCHANGE_READY]));
	/* the rank's segment added */
	status = DS41F_THEN(ds41f_put(b, &wide[WIDE_EXCHANGE_SUM]));
	for (uint32_t step = 1; step < DS41F_TP4_RANKS && status == DS41F_OK;
	     step++) {
		const uint32_t rank = (tensor->rank + step) % DS41F_TP4_RANKS;
		status = copy_put(
			b, output + tensor->rank * segment + skipped,
			word_of_rank +
				rank * sizeof(struct ordered_collective_peer),
			tensor->peer_regions[site][rank] +
				arguments->output_offset +
				tensor->rank * segment + skipped,
			segment - skipped,
			copy_branch(step - 1u, DS41F_TP4_RANKS - 1u));
	}
	wide[WIDE_EXCHANGE_DONE].branch = LAUNCH_JOIN;
	status = DS41F_THEN(ds41f_put(b, &wide[WIDE_EXCHANGE_DONE]));
	const struct tp2_join_description unpack = {
		.source = { output, segment * DS41F_TP4_RANKS },
		.output = values,
		.width = DS41F_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = DS41F_HIDDEN,
		.output_row_stride = DS41F_HIDDEN,
		.operation = TP2_JOIN_RING_UNPACK
	};
	return DS41F_THEN(tp2_join_prepare(
		&unpack, b->arena, b->tensor_join_kernel, ds41f_slots(b, 1)));
}

MUST_CHECK result_status ds41f_tensor_complete(struct ds41f_builder *b,
					       uint32_t site,
					       struct linear_range values,
					       struct linear_range routed)
{
	struct ordered_push_description collective =
		b->tensor_layer->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * DS41F_HIDDEN;
	const uint64_t bytes =
		(uint64_t)arguments->element_count * sizeof(float);
	const struct linear_range step = { DS41F_ARRAY(STEP).byte_offset,
					   STEP_BLOCK_BYTES_VALUE };
	const struct linear_range input = { arguments->region_offset +
						    arguments->input_offset,
					    arguments->parity_stride + bytes };
	const struct linear_range output = { arguments->region_offset +
						     arguments->output_offset,
					     arguments->parity_stride + bytes };
	/* Reuse the prepared owner exchange and its separate READY/DONE protocol. Its BF16 publisher
	 * rounds each
     * rank's partial once; the owner adds in FP32 rank order and rounds the result once. Drafts and FULL/WIDE
     * chunks retain their collectives; prompt tails using THIN/BATCH share the new sums. This changes target bits. */
	const bool round_once = !b->reads_draft_step &&
				b->tp_description != NULL &&
				b->tp_description->stages[b->card]
					->description.decode_round_once &&
				(b->made->program.kind == PROGRAM_THIN ||
				 b->made->program.kind == PROGRAM_BATCH);
	if (round_once &&
	    (!b->tensor_layer->bf16_ring ||
	     b->tensor_layer->rank_count != DS41F_TP4_RANKS ||
	     !b->tensor_layer->shared_channels ||
	     b->tensor_layer->parallel_publish || b->tensor_local_stub ||
	     b->tensor_layer->wide_exchange < DS41F_TP_EXCHANGE_OWNER_SUMS ||
	     arguments->payload_format != ORDERED_PUSH_BF16 ||
	     arguments->incoming_host_mask != 0))
		return STATUS_UNSUPPORTED;
	if (!round_once && b->tensor_layer->bf16_ring &&
	    b->rows <= b->tensor_layer->ring_rows) {
		/* the BF16 ring: values plus the routed sums added and rounded in one pass into the
		 * packed rows (the
         * bits of the FP32 publish and the round it replaces), then the ring's BF16 sum back into values;
         * with no
         * routed sums (the attention site, or an FFN site whose combine added them: shared_sum) the values are those
         * rows already: the ring sums them in place (launches_untrimmed) */
		if (routed.byte_count == 0 &&
		    !b->tensor_layer->launches_untrimmed)
			return ds41f_exact_gather(b, site, values, values,
						  b->rows * DS41F_HIDDEN *
							  LINEAR_BFLOAT16_BYTES,
						  b->rows);
		const struct linear_range packed =
			b->tensor_layer->exact_hidden;
		const struct tp2_join_description sum = {
			.source = values,
			.routed_sums = routed,
			.output = packed,
			.width = DS41F_HIDDEN,
			.row_count = b->rows,
			.source_row_stride = DS41F_HIDDEN,
			.routed_row_stride =
				routed.byte_count == 0 ? 0 : DS41F_HIDDEN,
			.output_row_stride = DS41F_HIDDEN,
			.operation = TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16
		};
		const result_status summed =
			tp2_join_prepare(&sum, b->arena, b->tensor_join_kernel,
					 ds41f_slots(b, 1));
		return summed != DS41F_OK ?
			       summed :
			       ds41f_exact_gather(b, site, packed, values,
						  b->rows * DS41F_HIDDEN *
							  LINEAR_BFLOAT16_BYTES,
						  b->rows);
	}
	/* CE_OWNER_SUMS: the attention site's piece of whole quarters by copy engines (direct routes) */
	if (!round_once && b->tensor_layer->shared_channels &&
	    b->tensor_layer->wide_exchange >= DS41F_TP_EXCHANGE_CE_OWNER_SUMS &&
	    b->rows > LINEAR_MAXIMUM_ROWS && !b->tensor_local_stub &&
	    !b->tensor_layer->parallel_publish && routed.byte_count == 0 &&
	    arguments->incoming_host_mask == 0 &&
	    b->tensor_layer->rank_count == DS41F_TP4_RANKS &&
	    b->rows % DS41F_TP4_RANKS == 0 &&
	    arguments->payload_format == ORDERED_PUSH_BF16)
		return tensor_complete_ce(b, site, values);
	struct tp2_join_description description = {
		.source = values,
		.routed_sums = routed,
		.output = input,
		.step = step,
		.output_slot_stride = arguments->parity_stride,
		.width = DS41F_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = DS41F_HIDDEN,
		.routed_row_stride = routed.byte_count == 0 ? 0 : DS41F_HIDDEN,
		.output_row_stride = DS41F_HIDDEN,
		.operation = TP2_JOIN_BFLOAT16_TO_FLOAT32
	};
	result_status status = tp2_join_prepare(&description, b->arena,
						b->tensor_join_kernel,
						ds41f_slots(b, 1));
	const bool wide = b->tensor_layer->shared_channels &&
			  b->tensor_layer->wide_exchange != 0 &&
			  (round_once || b->rows > LINEAR_MAXIMUM_ROWS) &&
			  !b->tensor_local_stub &&
			  !b->tensor_layer->parallel_publish;
	/* owner sums need direct routes (tp_exchange.h): a host route keeps the push, the same bits */
	const bool owner_sums = wide &&
				b->tensor_layer->wide_exchange >=
					DS41F_TP_EXCHANGE_OWNER_SUMS &&
				arguments->incoming_host_mask == 0;
	if (b->tensor_local_stub) {
		description = (struct tp2_join_description){
			.source = b->tensor_layer->stub_values[site],
			.output = output,
			.step = step,
			.output_slot_stride = arguments->parity_stride,
			.width = DS41F_HIDDEN,
			.row_count = b->rows,
			.source_row_stride = DS41F_HIDDEN,
			.output_row_stride = DS41F_HIDDEN,
			.operation = TP2_JOIN_FLOAT32_COPY
		};
		status = DS41F_THEN(tp2_join_prepare(&description, b->arena,
						     b->tensor_join_kernel,
						     ds41f_slots(b, 1)));
	} else if (b->tensor_layer->parallel_publish) {
		status = DS41F_THEN(ordered_push_parallel_prepare(
			&collective, b->tensor_parallel_kernels,
			ds41f_slots(b, ORDERED_PUSH_PARALLEL_LAUNCHES)));
	} else if (wide) {
		/* wide_exchange: the piece's push and sums by many CTAs (the same payload, order and bits); owner sums
		 * leave
         * the BF16 rows in the output array. Through a view (split_overlap's half) its step block's
         * rows. */
		arguments->step_offset = b->view != 0 || round_once ?
						 DS41F_ARRAY(STEP).byte_offset :
						 arguments->step_offset;
		status = DS41F_THEN(tp_exchange_wide_prepare(
			&collective, exchange_phase(b), TP_EXCHANGE_PHASES,
			owner_sums ? b->tensor_owner_kernels :
				     b->tensor_exchange_kernels,
			owner_sums ? TP_EXCHANGE_FORM_OWNER_SUMS :
				     TP_EXCHANGE_FORM_PUSH,
			ds41f_slots(b, TP_EXCHANGE_WIDE_LAUNCHES)));
	} else {
		status = b->tensor_layer->shared_channels ?
				 DS41F_THEN(tp_exchange_prepare(
					 &collective, exchange_phase(b),
					 TP_EXCHANGE_PHASES,
					 b->tensor_push_kernel,
					 ds41f_slots(b, 1))) :
				 DS41F_THEN(ordered_push_prepare(
					 &collective, b->tensor_push_kernel,
					 ds41f_slots(b, 1)));
	}
	description = (struct tp2_join_description){
		.source = output,
		.output = values,
		.step = step,
		.source_slot_stride = arguments->parity_stride,
		.width = DS41F_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = DS41F_HIDDEN,
		.output_row_stride = DS41F_HIDDEN,
		.operation = owner_sums ? TP2_JOIN_BFLOAT16_COPY :
					  TP2_JOIN_FLOAT32_TO_BFLOAT16
	};
	return DS41F_THEN(tp2_join_prepare(&description, b->arena,
					   b->tensor_join_kernel,
					   ds41f_slots(b, 1)));
}

/* wide_exchange OWNER_RINGS, the FFN site of a piece of more than 8 rows: values plus the routed sums rounded to
 * BF16
 * (the ring's packed rows) into the collective's input array, the owner ring sums (tp_exchange.h: each slab's
 * BF16
 * ring sum, the same bits) and their BF16 rows copied back into values, in place of a ring a slab. */
MUST_CHECK result_status ds41f_tensor_complete_owner_rings(
	struct ds41f_builder *b, uint32_t site, struct linear_range values,
	struct linear_range routed)
{
	struct ordered_push_description collective =
		b->tensor_layer->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * DS41F_HIDDEN;
	const uint64_t bytes =
		(uint64_t)arguments->element_count * LINEAR_BFLOAT16_BYTES;
	const struct linear_range step = { DS41F_ARRAY(STEP).byte_offset,
					   STEP_BLOCK_BYTES_VALUE };
	const struct linear_range input = { arguments->region_offset +
						    arguments->input_offset,
					    arguments->parity_stride + bytes };
	const struct linear_range output = { arguments->region_offset +
						     arguments->output_offset,
					     arguments->parity_stride + bytes };
	struct tp2_join_description description = {
		.source = values,
		.routed_sums = routed,
		.output = input,
		.step = step,
		.output_slot_stride = arguments->parity_stride,
		.width = DS41F_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = DS41F_HIDDEN,
		.routed_row_stride = routed.byte_count == 0 ? 0 : DS41F_HIDDEN,
		.output_row_stride = DS41F_HIDDEN,
		.operation = TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16
	};
	result_status status = tp2_join_prepare(&description, b->arena,
						b->tensor_join_kernel,
						ds41f_slots(b, 1));
	status = DS41F_THEN(tp_exchange_wide_prepare(
		&collective, exchange_phase(b), TP_EXCHANGE_PHASES,
		b->tensor_ring_kernels, TP_EXCHANGE_FORM_OWNER_RING_SUMS,
		ds41f_slots(b, TP_EXCHANGE_WIDE_LAUNCHES)));
	description = (struct tp2_join_description){
		.source = output,
		.output = values,
		.step = step,
		.source_slot_stride = arguments->parity_stride,
		.width = DS41F_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = DS41F_HIDDEN,
		.output_row_stride = DS41F_HIDDEN,
		.operation = TP2_JOIN_BFLOAT16_COPY
	};
	return DS41F_THEN(tp2_join_prepare(&description, b->arena,
					   b->tensor_join_kernel,
					   ds41f_slots(b, 1)));
}

/* bf16_slab_rings (ds41f_tp_stage.h), and always in a BATCH or thin-tail program (each slab the thin steps'
 * BF16
 * ring sum, not one FP32 ordered push over the sessions' rows): a piece of more than 8 rows completes a slab at a time
 * by the BF16 ring, the last slab its remaining rows; otherwise tensor_complete over the rows. */
MUST_CHECK result_status ds41f_tensor_complete_slabs(struct ds41f_builder *b,
						     uint32_t site,
						     struct linear_range values,
						     struct linear_range routed)
{
	const uint32_t rows = b->rows, first = b->first_row,
		       piece = b->tensor_layer->ring_rows;
	if (!b->tensor_layer->bf16_ring || rows <= LINEAR_MAXIMUM_ROWS ||
	    (!b->tensor_layer->bf16_slab_rings && !ds41f_thin_forms_of(b)) ||
	    rows <= piece)
		return ds41f_tensor_complete(
			b, site, values,
			routed); /* at most ring_rows rows: one BF16 ring */
	if (b->view != 0)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t value_row =
		(uint64_t)DS41F_HIDDEN * LINEAR_BFLOAT16_BYTES;
	const uint64_t routed_row = (uint64_t)DS41F_HIDDEN * sizeof(float);
	result_status status = DS41F_OK;
	for (uint32_t at = 0; at < rows && status == DS41F_OK;
	     at += piece) { /* pieces of ring_rows rows (8: slabs) */
		const uint32_t count = rows - at < piece ? rows - at : piece;
		ds41f_rows_select(b, 0, first + at, count);
		status = ds41f_tensor_complete(
			b, site,
			(struct linear_range){ values.byte_offset +
						       at * value_row,
					       count * value_row },
			routed.byte_count == 0 ?
				routed :
				(struct linear_range){ routed.byte_offset +
							       at * routed_row,
						       count * routed_row });
	}
	ds41f_rows_select(b, 0, first, rows);
	return status;
}

/* The n-gram steps of layer, in the card's rows of its table (model_seam.h). */
MUST_CHECK result_status ds41f_engram_launches(struct ds41f_builder *b,
					       uint32_t layer, uint32_t steps)
{
	const struct ds41f_tensor *const weights =
		b->plan->layer_tensors[layer];
	const uint64_t token_bytes =
		(uint64_t)DS41F_ENGRAM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES;
	const uint32_t table = (uint32_t)__builtin_popcountll(
		DS41F_ENGRAM_LAYERS & ((1ull << layer) - 1u));
	const uint64_t below = (uint64_t)__builtin_popcount(
		b->plan->cards[b->card].engram_tables & ((1u << table) - 1u));
	struct kernel_launch launches[DS41F_ENGRAM_ROWS_LAUNCH_COUNT];
	const struct ds41f_engram_rows_description description = {
		{ DS41F_ARRAY(ENGRAM_ROWS).byte_offset +
			  (below * b->plan->limits.row_capacity + b->first_row) *
				  token_bytes,
		  b->rows * token_bytes },
		DS41F_ARRAY(ENGRAM_VALUES),
		DS41F_ARRAY(ENGRAM_PROJECTION),
		DS41F_WEIGHTS(ENGRAM_QUERY)->stored,
		DS41F_WEIGHTS(ENGRAM_KEY)->stored,
		DS41F_ARRAY(STREAMS),
		DS41F_ENGRAM_ROWS_PER_TOKEN,
		b->rows
	};
	result_status status =
		(steps & (DS41F_ENGRAM_VALUES | DS41F_ENGRAM_GATE)) == 0 ?
			DS41F_OK :
		b->vision_active ?
			ds41f_image_engram_rows_prepare(
				&description, DS41F_ARRAY(STEP), b->arena,
				&b->engram, &b->image_routing, launches) :
			ds41f_engram_rows_prepare(&description, b->arena,
						  &b->engram, launches);
	if (status == DS41F_OK && b->plan->limits.official_reference != 0) {
		launches[DS41F_ENGRAM_ROWS_GATE_LAUNCH].kernel =
			b->official.engram;
		launches[DS41F_ENGRAM_ROWS_GATE_LAUNCH].block[0] =
			DS41F_OFFICIAL_STATE_THREADS;
	}
	if ((steps & DS41F_ENGRAM_VALUES) != 0) {
		status = DS41F_THEN(ds41f_put(
			b, &launches[DS41F_ENGRAM_ROWS_VALUES_LAUNCH]));
	}
	if ((steps & DS41F_ENGRAM_PRODUCT) != 0) {
		status = DS41F_THEN(ds41f_quantize(
			b, DS41F_ARRAY(ENGRAM_VALUES),
			DS41F_WEIGHTS(ENGRAM)->shape.input_size));
		status = DS41F_THEN(ds41f_trunk(b, DS41F_WEIGHTS(ENGRAM),
						DS41F_ARRAY(ENGRAM_PROJECTION),
						NULL, ds41f_no_range));
	}
	return (steps & DS41F_ENGRAM_GATE) != 0 ?
		       DS41F_THEN(ds41f_put(
			       b, &launches[DS41F_ENGRAM_ROWS_GATE_LAUNCH])) :
		       status;
}

/* One lossless ring gather (or the BF16 recipe's ring sum) through the bank of b->rows rows of the site's
 * channel (its
 * peer table and region); a bank holds its mode's two epochs of slots or the ring is refused. */
MUST_CHECK result_status ds41f_exact_gather_bank(
	struct ds41f_builder *b, uint32_t site, uint64_t ring_offset,
	uint64_t bank_bytes, uint64_t row_bytes, struct linear_range input,
	struct linear_range output, uint32_t bytes, uint32_t vectors)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	if (tensor == NULL || site >= DS41F_TP2_SITES || bank_bytes == 0 ||
	    (!tensor->exact_output_shards && !tensor->bf16_ring) ||
	    b->rows == 0 ||
	    b->rows > (ring_offset == tensor->ring_offsets[site] ?
			       tensor->ring_rows :
			       LINEAR_MAXIMUM_ROWS) ||
	    bytes == 0 ||
	    bytes % (DS41F_TP4_RANKS * EXACT_RING_PACK_BYTES) != 0 ||
	    input.byte_count < bytes || output.byte_count < bytes ||
	    input.byte_offset > b->arena ||
	    bytes > b->arena - input.byte_offset ||
	    output.byte_offset > b->arena ||
	    bytes > b->arena - output.byte_offset)
		return STATUS_INVALID_ARGUMENT;
	const struct ordered_push_arguments *const channel =
		&tensor->collectives[site].arguments;
	const uint64_t bank =
		ring_offset +
		ds41f_ring_bank_start(bank_bytes, row_bytes, b->rows);
	const struct linear_range bank_range = {
		bank, bank_bytes + row_bytes * b->rows
	};
	const uint64_t packs =
		bytes / (DS41F_TP4_RANKS * EXACT_RING_PACK_BYTES);
	if (EXACT_RING_METADATA_BYTES +
		    packs * (tensor->bf16_ring ?
				     EXACT_RING_SUM_EPOCH_HOPS *
					     EXACT_RING_PACK_BYTES :
				     EXACT_RING_EPOCH_HOPS *
					     EXACT_RING_PACKET_BYTES) >
	    bank_range.byte_count)
		return STATUS_INVALID_ARGUMENT;
	/* the BF16 sum may run in place (input the output: kernel_cut_through.cuh reads every owner's
	 * pack before it
     * writes that pack); otherwise the input shares no byte with the output */
	const bool in_place = tensor->bf16_ring &&
			      input.byte_offset == output.byte_offset;
	const struct checked_range ranges[] = {
		{ "the ring output", &output, bytes, false },
		{ "the ring bank", &bank_range, bank_range.byte_count, false },
		{ "the ring input", &input, bytes, false }
	};
	if (bank < channel->region_offset ||
	    !prepare_ranges_are_valid("the TP ring", ranges,
				      in_place ? DS41F_TP2_SITES :
						 DS41F_TP2_SITES + 1u,
				      b->arena))
		return STATUS_INVALID_ARGUMENT;
	const struct exact_ring_arguments arguments = {
		.input_offset = input.byte_offset,
		.output_offset = output.byte_offset,
		.bank_offset = bank,
		.peer_table_offset = channel->peer_table_offset,
		.peer_bank_delta = bank - channel->region_offset,
		.deadline_cycles = DS41F_TP2_DEADLINE_CYCLES,
		.packs_per_owner =
			bytes / (DS41F_TP4_RANKS * EXACT_RING_PACK_BYTES),
		.vectors = vectors,
		.rank = tensor->rank,
		/* ring_column_owners: a BF16 sum's owners the quarters of every row, not of the rows (the
		 * input
		 * row-major) */
		.mode = !tensor->bf16_ring ? EXACT_RING_GATHER :
			b->ring_column_owners && vectors != 0 &&
					(uint64_t)packs % vectors == 0 ?
					     EXACT_RING_BF16_SUM_COLUMNS :
					     EXACT_RING_BF16_SUM
	};
	uint32_t blocks = 1;
	while (blocks < EXACT_RING_BLOCKS &&
	       blocks * ORDERED_COLLECTIVE_THREADS < arguments.packs_per_owner)
		blocks *= EXACT_RING_GRID_GROWTH;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->exact_ring_kernel, blocks, 1u,
			    ORDERED_COLLECTIVE_THREADS, 0, &arguments,
			    sizeof(arguments));
	return ds41f_put(b, &launch);
}

/* A gather's part of units (slabs, rows) the rank computes: its first and count. A quarter each; gather_balance: the
 * ranks of the narrower expert slice (their shared expert's inner rows, the routed experts' cut alike) in equal parts,
 * the last of them the rest, the wider ranks none (they set the pace at every exchange); _WEIGHTED: a wider rank
 * DS41F_GATHER_WIDE_SIXTEENTHS of the units, the narrower ones the rest alike, each part whole granules (rows: a
 * dense launch's 8-row groups), the ranks' parts in rank order. */
void ds41f_gather_share(const struct ds41f_builder *b, uint32_t units,
			uint32_t granule, uint32_t *first_out,
			uint32_t *count_out)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	*first_out = tensor->rank * (units / DS41F_TP4_RANKS);
	*count_out = units / DS41F_TP4_RANKS;
	const uint32_t balance =
		b->tp_description->stages[b->card]->description.gather_balance;
	if (balance == 0 || (balance == DS41F_GATHER_BALANCE_WIDE &&
			     b->made->program.kind != PROGRAM_WIDE))
		return;
	uint32_t widths[DS41F_TP4_RANKS], narrowest = UINT32_MAX, narrow = 0,
					  before = 0;
	for (uint32_t rank = 0; rank < DS41F_TP4_RANKS; rank++) {
		const uint32_t card = tensor->card - tensor->rank + rank;
		widths[rank] = b->tp_description->views[card]
				       ->layer_tensors[tensor->layer_index]
						      [DS41F_TENSOR_SHARED_GATE]
				       .shape.output_size;
		narrowest = widths[rank] < narrowest ? widths[rank] : narrowest;
	}
	for (uint32_t rank = 0; rank < DS41F_TP4_RANKS; rank++) {
		before += rank < tensor->rank && widths[rank] == narrowest ?
				  1u :
				  0u;
		narrow += widths[rank] == narrowest ? 1u : 0u;
	}
	if (balance == DS41F_GATHER_BALANCE_WEIGHTED) {
		const uint32_t wide_part = units / granule *
					   DS41F_GATHER_WIDE_SIXTEENTHS /
					   DS41F_GATHER_SIXTEENTHS * granule;
		const uint32_t narrow_units =
			units - (DS41F_TP4_RANKS - narrow) * wide_part;
		const uint32_t narrow_part =
			narrow_units / narrow / granule * granule;
		uint32_t first = 0, narrow_seen = 0;
		for (uint32_t rank = 0; rank <= tensor->rank; rank++) {
			narrow_seen += widths[rank] == narrowest ? 1u : 0u;
			const uint32_t count =
				widths[rank] != narrowest ?
					wide_part :
				narrow_seen == narrow ?
					narrow_units -
						(narrow - 1u) * narrow_part :
					narrow_part;
			*first_out = first;
			*count_out = count;
			first += count;
		}
		return;
	}
	const uint32_t part = units / narrow;
	*first_out = before * part;
	*count_out = widths[tensor->rank] != narrowest ? 0u :
		     before + 1u == narrow	       ? units - before * part :
							 part;
}

MUST_CHECK result_status ds41f_router_gather(struct ds41f_builder *b,
					     const struct ds41f_tensor *router)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	const uint32_t view = b->view, first = b->first_row, rows = b->rows;
	uint32_t own_first = 0,
		 quarter =
			 0; /* the rank's rows: a quarter, or gather_balance's part */
	ds41f_gather_share(b, rows, LINEAR_MAXIMUM_ROWS, &own_first, &quarter);
	ds41f_rows_select(b, view, first + own_first, quarter);
	result_status status = quarter != 0 ? ds41f_router_scores(b, router) :
					      DS41F_OK;
	ds41f_rows_select(b, view, first, rows);
	const uint64_t into =
		(uint64_t)(first + own_first) * DS41F_ROUTER_ROW_BYTES;
	struct ds41f_gather_range range = {
		b->plan->cards[b->card].array_offset[DS41F_ARRAY_ROUTER_SCORES] +
			into,
		(uint64_t)quarter * DS41F_ROUTER_ROW_BYTES,
		{ 0 }
	};
	for (uint32_t rank = 0; rank < DS41F_TP4_RANKS; rank++) {
		const uint32_t card = tensor->card - tensor->rank + rank;
		range.peers[rank] =
			b->tp_description->views[card]
				->cards[card]
				.array_offset[DS41F_ARRAY_ROUTER_SCORES] +
			into;
	}
	const bool second = b->chunk_view == b->half_view_first + 1u;
	return DS41F_THEN(ds41f_rows_gather(
		b, second ? 0u : 1u,
		DS41F_TP_EXCHANGE_LAYER_PHASES * tensor->layer_index +
			(second ? GATHER_ROUTER_SECOND_PHASE :
				  GATHER_ROUTER_FIRST_PHASE),
		&range, 1u));
}
