/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_plan.c: the TP plan's preparation and its cards' load (split from commands.c by whole items).
 */
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include "models/deepseek_v41_flash/prompt/ds41f_chat_template.h"
#include "common/hooks.h"
#include "engine/storage/load/loader.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/runtime/placement/placement.h"
#include "engine/storage/replicas/replicas.h"
#include "engine/engram/engram_disk.h"
#include "engine/engram/engram_fit.h"
#include "engine/storage/io/local_device.h"
#include "engine/storage/profile/storage_profile.h"
#include "engine/storage/autotune/storage_autotune.h"
#include "engine/server/server.h"
#include "engine/runtime/placement/session_fit.h"
#include "engine/cli/cli_stats_client.h"
#include "engine/cli/cli_policy.h"
#include "engine/storage/io/storage_exl3.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/cli/cli_model_folder.h"
#include "models/model_registry.h"
#include "engine/runtime/scheduler/serving_cursor.h"
#include "engine/cli/cli_axes.h"
#include "engine/runtime/step/step.h"
#include "engine/runtime/step/step_images.h"
#include "engine/input/tokenizer/tokenizer.h"
#include "engine/runtime/speculation/draft_lookup.h"
#include "engine/formats/pack/tokenizer_tables_pack.h"
#include "common/json.h"
#include "engine/engram/engram_step.h"
#include "engine/engram/engram_hot.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "ds41f_tp_model.h"
#include "ds41f_tp_peer.h"
#include "engine/cli/commands.h"
#include "engine/cli/cli_golden_capture.h"

#include "engine/cli/cli_internal.h"
#include "ds41f_tp_plan.h"

#define BYTES_PER_GB 1e9
#define ROUTER_MMA_WIDE 2u /* --tp_router_mma=2 */

/* Cold TP bootstrap. The ordinary unit map records canonical homes for identity/statistics only: selected
 * tensor jobs, rather than this map, load every replica/shard. All GPU edges are tested on the actual arenas. */
MUST_CHECK result_status
ds41f_tensor_plan_prepare(const struct program_limits *limits,
			  uint64_t kept_free_bytes, uint32_t *unit_homes)
{
	const bool exl3 = execution_placement == GPU_SPLIT_TP4_EXL3;
	if (card_count != (exl3 ? DS41F_TP4_RANKS : DS41F_TP2_PROGRAM_CARDS) ||
	    unit_homes == NULL)
		return STATUS_UNSUPPORTED;
	struct ds41f_tp2_model_description description = {
		.limits = *limits,
		.kept_free_bytes = kept_free_bytes,
		.wire_format = tp_wire_bf16,
		.exact_output_shards = exl3 ? tp_wire_bf16 == 0 :
					      numerics == NUMERICS_EXACT,
		.exl3_product_order =
			exl3_expert_parts == EXL3_EXPERT_OPTIONAL_PARTS ?
				BB_EXL3_EIGHT_PART_ORDER :
				BB_EXL3_REFERENCE_ORDER,
		.original_code_wire = tp_code_wire != 0,
		.attention_owner = tp_attention_owner != 0,
		.head_shards = tp_head_shards != 0,
		.exact_dense = exl3 && tp_wire_bf16 == 0 && tp_exact_dense != 0,
		.draft_shards = tp_draft_shards != 0,
		.draft_layers = exl3 && tp_wire_bf16 != 0 &&
				tp_draft_layers != 0,
		.head_stream = tp_head_stream != 0,
		.exl3_fast_products = exl3_fast_products != 0,
		.exl3_wide_entry = exl3_wide_entry != 0,
		.exl3_tc_moe = tc_moe_selected(),
		.exl3_tc_product_blocks = exl3_tc_product_blocks,
		.exl3_expert_rotation =
			tc_moe_selected() ? exl3_expert_rotation : 0u,
		.head_mma = exl3 && tp_wire_bf16 != 0 && tp_head_mma != 0,
		.trunk_mma = exl3 && tp_wire_bf16 != 0 && tp_trunk_mma != 0,
		.trunk_mma_fused = tp_trunk_mma_fused != 0,
		.trunk_mma_thin_splits = tp_trunk_mma_thin_splits != 0,
		.router_mma = exl3 && tp_wire_bf16 != 0 && tp_router_mma != 0,
		.router_mma_wide = exl3 && tp_wire_bf16 != 0 &&
				   tp_router_mma == ROUTER_MMA_WIDE,
		.gather_balance = exl3 ? tp_gather_balance : 0u,
		.pool_chunk = exl3 && tp_pool_chunk != 0,
		.pool_gather = exl3 && tp_pool_gather != 0,
		.router_first_whole = exl3 && tp_router_first_whole != 0,
		.tcp_two_blocks = exl3 && exl3_tcp_two_blocks != 0,
		.queries_piece = exl3 && tp_queries_piece != 0,
		.index_span = exl3 && tp_index_span != 0,
		.core_compact_halves = exl3 && tp_core_compact_halves != 0,
		.dense_skinny_bf16 = exl3 ? dense_skinny_bf16 : 0u,
		.dense_skinny_f32 = exl3 ? dense_skinny_f32 : 0u,
		.dense_staged = exl3 ? dense_staged : original_dense_staged,
		.original_wide_experts = original_wide_experts != 0,
		.graph_parts = tp_graph_parts,
		.ordered_wide = tp_ordered_wide != 0,
		.graph_branches = tp_graph_branches,
		.piece_rings = tp_piece_rings != 0,
		.trunk_shape_slices = tp_trunk_shape_slices != 0,
		.bf16_slab_rings = tp_bf16_slab_rings != 0,
		.wide_exchange = tp_wide_exchange,
		.split_overlap = tp_split_overlap,
		.split_least_rows = tp_split_least_rows,
		.launches_untrimmed = tp_launch_trims == 0,
		.full_split_overlap = tp_full_split_overlap,
		.tail_rows = prefill_tail_rows,
		.tail_overlay = prefill_tail_overlay != 0,
		.tail_sets = prefill_tail_sets != 0,
		.thin_tail = prefill_thin_tail != 0,
		.full_cone_behind =
			prefill_dead_halves == 0 || prefill_prefix_stable != 0 ?
				0u :
				engine_full_cone_behind(
					prefill_cone, chunks_in_flight,
					prefill_fast_plan, prefill_margin,
					prefill_cone_skip),
		.dead_rows_trimmed = prefill_dead_halves == DEAD_ROWS_TRIMMED,
		.chunk_core = (exl3 || numerics != NUMERICS_EXACT) ?
				      tp_chunk_core != 0 :
				      original_chunk_core != 0,
		.replicated_embedding = exl3 && tp_replicated_embedding != 0,
		.batch = batch_rows[0] != 0,
		.original_batch = original_batch != 0,
		.batch_draft_rows = batch_draft_rows != 0,
		.batch_wide_head = batch_wide_head != 0,
		.batch_context_slabs = batch_context_slabs != 0,
		.batch_thin_bits = batch_thin_bits != 0,
		.batch_grouped_rows = batch_grouped_rows,
		.batch_tc_paired = batch_tc_paired,
		.batch_draft_slots_head = batch_draft_slots_head != 0,
		.batch_row_sets = batch_row_sets,
		.batch_core_once = batch_core_once,
		.front_branch = tp_front_branch,
		.core_heads = tp_core_heads,
		.decode_round_once = tp_decode_round_once != 0,
		.draft_bf16_rows = draft_bf16_rows != 0,
		.batch_draft_wide = batch_draft_wide,
		.draft_bf16_operands = draft_bf16_operands != 0,
		.ring_column_owners = tp_ring_column_owners != 0,
		.tensor_layout =
			exl3 ? DS41F_TP_LAYOUT_QUAD_ALL :
			execution_placement == GPU_SPLIT_TP4_PRO ?
			       DS41F_TP_LAYOUT_QUAD :
			execution_placement == GPU_SPLIT_TP2_SINGLE_SINGLE_PRO ?
			       DS41F_TP_LAYOUT_HYBRID :
			       DS41F_TP_LAYOUT_PAIRS
	};
	for (uint32_t card = 0; card < card_count; card++)
		description.cards[card] = descriptions[card];
	result_status status =
		ds41f_tp2_model_create(pack, &description, &tensor_model);
	if (status != STATUS_OK)
		return status;
	plan = tensor_model->canonical;
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	for (uint32_t unit = 0; unit < pack_header_of(pack)->unit_count;
	     unit++) {
		unit_homes[unit] = units[unit].unit_kind ==
						   PLACEMENT_UNIT_HOST_ONLY ?
					   PLACEMENT_IN_HOST_MEMORY :
					   0u;
	}
	for (uint32_t index = 0;
	     index < tensor_model->gpu_load.description.job_count; index++) {
		const struct tp_load_job *const job =
			&tensor_model->jobs[index];
		const uint32_t extent =
			pack_tensors_of(pack)[job->tensor_index].extent_index;
		unit_homes[pack_extents_of(pack)[extent].unit_index] =
			job->card -
			tensor_model->stages[job->card]->description.rank;
	}
	return status;
}

MUST_CHECK result_status ds41f_tensor_cards_load(const void *tables,
						 uint64_t table_bytes)
{
	result_status status = STATUS_OK;
	load_section_apply();
	TIMER_BEGIN(tp_card_open);
	for (uint32_t card = 0; card < card_count && status == STATUS_OK;
	     card++) {
		const uint64_t bytes = plan->cards[card].arena_byte_count;
		(void)printf("TP card %" PRIu32 " backend=%" PRIu32
			     " arena=%" PRIu64 " bytes\n",
			     card, card_order[card], bytes);
		status = backend_cuda_value.card_open(card_order[card], bytes,
						      &cards[card]);
	}
	TIMER_END(tp_card_open);
	struct ds41f_tp2_model_load_request request = {
		.external_host_extents =
			engram_disk_external_extents(engram_disk),
		.backend = &backend_cuda_value,
		.cards = cards,
		.storage = storage_operations,
		.reader = reader,
		.wait_limit_nanoseconds = CARD_WAIT_LIMIT_NANOSECONDS,
		.load_threads = tp_load_threads,
		.load_reads_per_copy = tp_load_reads_per_copy,
		.load_host_reads_per_copy = tp_load_host_reads_per_copy,
		.load_host_threads = tp_load_host_threads,
		.load_copy_split = tp_load_copy_split,
		.load_piece_bytes = tp_load_piece_bytes
	};
	if (tuned_load_active && load_tuned_weights != 0)
		memcpy(request.load_copy_weights, tuned_load_weights,
		       sizeof(request.load_copy_weights));
	TIMER_BEGIN(tp_peer_qualify);
	status = status != STATUS_OK ?
			 status :
			 ds41f_tp2_peer_qualify(tensor_model,
						&backend_cuda_value, cards,
						CARD_WAIT_LIMIT_NANOSECONDS,
						request.qualified_writer_masks);
	TIMER_END(tp_peer_qualify);
	status = status != STATUS_OK ?
			 status :
			 ds41f_tp2_model_load(tensor_model, &request);
	status = status != STATUS_OK ?
			 status :
			 ds41f_tp2_model_front_end_match(tensor_model, tables,
							 table_bytes);
	if (status == STATUS_OK)
		(void)printf("TP load: source=%" PRIu64 " uploaded=%" PRIu64
			     " host=%" PRIu64 " bytes; tensors=%" PRIu32
			     " verified_extents=%" PRIu32 "+%" PRIu32 "\n",
			     tensor_model->gpu_result.source_bytes_read,
			     tensor_model->gpu_result.uploaded_bytes,
			     tensor_model->host_bytes,
			     tensor_model->gpu_result.tensors_uploaded,
			     tensor_model->gpu_result.extents_verified,
			     tensor_model->host_result.extents_verified);
	/* per card: the sum over its loaded ranges of their on-card digests at their offsets (tp_load.h) */
	for (uint32_t card = 0; card < card_count && status == STATUS_OK;
	     card++)
		(void)printf("%s %016" PRIx64 "%s",
			     card == 0 ? "TP load receipts:" : "",
			     tensor_model->gpu_result.card_receipts[card],
			     card + 1 == card_count ? "\n" : "");
	for (uint32_t copy = 0;
	     copy < copy_count && status == STATUS_OK && tp_load_threads != 0;
	     copy++) {
		const struct tp_load_copies *const copies =
			&tensor_model->copies;
		(void)printf(
			"TP load copy %" PRIu32
			": %.2f GB in %.2f s busy (%.2f GB/s), %" PRIu32
			" failed reads\n",
			copy,
			(double)copies->delivered_bytes[copy] / BYTES_PER_GB,
			(double)copies->busy_nanoseconds[copy] /
				NANOSECONDS_PER_SECOND,
			copies->busy_nanoseconds[copy] != 0 ?
				(double)copies->delivered_bytes[copy] /
					(double)copies->busy_nanoseconds[copy] :
				0.0,
			copies->failed_reads[copy]);
	}
	/* what each copy delivered to each role: under the default split a copy serves the other role once its own
	 * has claimed its last read (tp_load.h) */
	for (uint32_t copy = 0;
	     copy < copy_count && status == STATUS_OK && tp_load_threads != 0;
	     copy++) {
		const struct tp_load_copies *const copies =
			&tensor_model->copies;
		(void)printf(
			"%scopy %" PRIu32 " host %.2f cards %.2f GB%s",
			copy == 0 ? "TP load roles: " : "", copy,
			(double)copies->role_delivered_bytes[TP_LOAD_ROLE_HOST]
							    [copy] /
				BYTES_PER_GB,
			(double)copies->role_delivered_bytes[TP_LOAD_ROLE_CARDS]
							    [copy] /
				BYTES_PER_GB,
			copy + 1u == copy_count ? "" : "; ");
		if (copy + 1u == copy_count)
			(void)printf("; split %" PRIu32 "\n",
				     tp_load_copy_split);
	}
	return status;
}
