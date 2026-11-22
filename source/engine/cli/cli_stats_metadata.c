/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_stats_metadata.c: the stats metadata bound (split from commands.c by whole items).
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
#include "cli_stats_client.h"
#include "cli_policy.h"
#include "engine/storage/io/storage_exl3.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "cli_model_folder.h"
#include "models/model_registry.h"
#include "engine/runtime/scheduler/serving_cursor.h"
#include "cli_axes.h"
#include "engine/runtime/step/step.h"
#include "engine/runtime/step/step_images.h"
#include "engine/input/tokenizer/tokenizer.h"
#include "engine/runtime/speculation/draft_lookup.h"
#include "engine/formats/pack/tokenizer_tables_pack.h"
#include "common/json.h"
#include "engine/engram/engram_step.h"
#include "engine/engram/engram_hot.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_model.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_peer.h"
#include "commands.h"
#include "cli_golden_capture.h"

#include "cli_internal.h"

#define STATS_LAYER_MASK_BITS 64u
#define STATS_HEAD_LANGUAGE 1ull
#define STATS_HEAD_DRAFT 2ull
#define STATS_HEAD_CONFIDENCE 4ull

/* Placement is prepared once. It is immutable while requests run; no model or program changes for statistics. */
MUST_CHECK result_status stats_metadata_bound(const uint32_t *card_of_unit)
{
	struct engine_card_stats metadata[BACKEND_MAXIMUM_CARDS] = { 0 };
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	const uint32_t unit_count = pack_header_of(pack)->unit_count;
	const char *const head_names[] = {
		"head.weight", "mtp.2.markov_head.head.weight",
		"mtp.2.confidence_head.proj.weight"
	};
	const uint64_t head_bits[] = { STATS_HEAD_LANGUAGE, STATS_HEAD_DRAFT,
				       STATS_HEAD_CONFIDENCE };
	size_t order_bytes = 0, placement_bytes = 0;
	for (uint32_t card = 0; card < card_count; card++) {
		metadata[card].engine_index = card;
		metadata[card].backend_index = card_order[card];
		metadata[card].description = descriptions[card];
		result_status status = instance.operations->arena_byte_count(
			plan, card, &metadata[card].arena_bytes);
		if (status != STATUS_OK) {
			return status;
		}
		for (uint32_t unit = 0; unit < unit_count; unit++) {
			if (card_of_unit[unit] == card &&
			    units[unit].unit_kind == PLACEMENT_UNIT_STAGE &&
			    units[unit].layer_index < STATS_LAYER_MASK_BITS) {
				metadata[card].layer_mask |=
					1ull << units[unit].layer_index;
				metadata[card].holds_draft =
					metadata[card].holds_draft ||
					units[unit].layer_index >=
						DS41F_DRAFT_FIRST_LAYER;
			}
		}
		for (uint32_t head = 0;
		     head < sizeof(head_names) / sizeof(head_names[0]);
		     head++) {
			const uint32_t tensor_index = pack_find_tensor(
				pack, head_names[head],
				(uint32_t)strlen(head_names[head]));
			if (tensor_index != PACK_NOT_FOUND) {
				const struct pack_tensor *const tensor =
					pack_tensors_of(pack) + tensor_index;
				const uint32_t unit =
					pack_extents_of(
						pack)[tensor->extent_index]
						.unit_index;
				if (card_of_unit[unit] == card) {
					metadata[card].head_mask |=
						head_bits[head];
					metadata[card].holds_draft =
						metadata[card].holds_draft ||
						head != 0;
				}
			}
		}
		if (tensor_model != NULL) {
			metadata[card].layer_mask = plan->cards[card].layers;
		}
		const int order_written = snprintf(
			stats_card_order + order_bytes,
			sizeof(stats_card_order) - order_bytes, "%s%" PRIu32,
			card != 0 ? "," : "", card_order[card]);
		const int placement_written = snprintf(
			stats_placement + placement_bytes,
			sizeof(stats_placement) - placement_bytes,
			"%scard%" PRIu32 " layers=0x%" PRIx64
			" heads=0x%" PRIx64 " draft=%s",
			card != 0 ? "; " : "", card, metadata[card].layer_mask,
			metadata[card].head_mask,
			metadata[card].holds_draft ? "yes" : "no");
		if (order_written < 0 ||
		    (size_t)order_written >=
			    sizeof(stats_card_order) - order_bytes ||
		    placement_written < 0 ||
		    (size_t)placement_written >=
			    sizeof(stats_placement) - placement_bytes) {
			return STATUS_INVALID_ARGUMENT;
		}
		order_bytes += (size_t)order_written;
		placement_bytes += (size_t)placement_written;
	}
	return engine_stats_cards(engine, metadata, card_count);
}
