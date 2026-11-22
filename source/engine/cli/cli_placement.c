/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_placement.c: the weights placement, the tensor placement and its drain, the startup placement status (split from
 * commands.c by whole items).
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

/* --weights against --placement: an omitted placement is the family's default split (original: PP5; EXL3: TP4_EXL3),
 * an explicit one must run the family. load runs no TP placement: it keeps its own fill and only picks the family's
 * pack, so an explicit PP5 is load's own placement for either family. */
bool weights_placement_matches(bool placement_was_given)
{
	const bool exl3 = weights_family == WEIGHTS_EXL3;
	if (weights_family == WEIGHTS_UNSET ||
	    (!is_generating && execution_placement == GPU_SPLIT_PP5) ||
	    (placement_was_given &&
	     (execution_placement == GPU_SPLIT_TP4_EXL3) == exl3))
		return true;
	if (!placement_was_given) {
		if (exl3 && is_generating) {
			execution_placement = GPU_SPLIT_TP4_EXL3;
			(void)fputs(
				"weights exl3 -> GPU split TP4_EXL3 (default for the family)\n",
				stderr);
		}
		return true;
	}
	(void)fprintf(stderr,
		      "--weights=%s conflicts with --placement=%s: %s\n",
		      exl3 ? "exl3" : "original", placement_text,
		      exl3 ? "EXL3 weights run on GPU split TP4_EXL3" :
			     "GPU split TP4_EXL3 runs EXL3 weights");
	return false;
}

bool tensor_placement_selected(void)
{
	return execution_placement == GPU_SPLIT_TP2_PP2_PRO ||
	       execution_placement == GPU_SPLIT_TP4_PRO ||
	       execution_placement == GPU_SPLIT_TP2_SINGLE_SINGLE_PRO ||
	       execution_placement == GPU_SPLIT_TP4_EXL3;
}

/* Every peer reader must retire before either arena of its pair can close. Unknown completion retains the
 * entire pair through process teardown; neither a failed wait nor closing its partner is a retirement proof. */
uint32_t tensor_cards_drain(void)
{
	uint32_t retained = (1u << card_count) - 1u;
	const uint32_t members = (execution_placement == GPU_SPLIT_TP4_PRO ||
				  execution_placement == GPU_SPLIT_TP4_EXL3) ?
					 DS41F_TP4_RANKS :
					 DS41F_TP2_RANKS;
	const result_status status = ds41f_tp_group_drain(
		&backend_cuda_value, cards, CARD_WAIT_LIMIT_NANOSECONDS,
		members, &retained);
	(void)status; /* retained remains the fail-closed ownership decision even after a failed wait */
	return retained;
}

int startup_placement_status(void)
{
	const struct startup_policy execution_policy = {
		.numerics = numerics,
		.placement = execution_placement,
		.cmp_order = cmp_order
	};
	if (startup_policy_admit(&execution_policy) != STATUS_OK) {
		(void)fprintf(
			stderr,
			"the requested TP placement is unsupported: tensor weight, state and program"
			" integration is not admitted by this build\n");
		return PROGRAM_FAILED;
	}
	if (execution_placement == GPU_SPLIT_TP4_EXL3) {
		(void)printf(
			"TP4 EXL3: four CMP cards; expert parts %u; %s\n",
			exl3_expert_parts,
			tp_wire_bf16 != 0 ?
				"local-K partial products and BF16 ring; approximate" :
				"full-K products and lossless FP16/FP32 transport");
	} else if (numerics_was_given && numerics != NUMERICS_EXACT) {
		if (tensor_placement_selected()) {
			(void)printf(
				"TP numerics: original packed weights and state; reference local CMP products;"
				" BF16 partial outputs; %s\n",
				tp_wire_bf16 != 0 ?
					"thin BF16 ring hop sums, wide ordered BF16 wire" :
					"FP32 wire, fixed rank-order FP32 sums and BF16 completion");
		} else {
			(void)printf(
				"numerics %s: %s CMP trunk/expert order; original weight/cache/head formats;"
				" fast currently adds no further PP5 approximation\n",
				numerics == NUMERICS_FAST ? "fast" : "relaxed",
				cmp_order == CMP_ORDER_RELAXED ? "relaxed" :
								 "reference");
		}
	}
	if (tp_attention_owner != 0 &&
	    (execution_placement != GPU_SPLIT_TP4_PRO ||
	     numerics != NUMERICS_EXACT || tp_wire_bf16 != 0)) {
		(void)fprintf(
			stderr,
			"attention ownership requires exact TP4 placement\n");
		return PROGRAM_FAILED;
	}
	if (tp_wire_bf16 != 0 &&
	    (!tensor_placement_selected() || numerics == NUMERICS_EXACT)) {
		(void)fprintf(
			stderr,
			"BF16 wire requires an explicit relaxed TP placement\n");
		return PROGRAM_FAILED;
	}
	if (tensor_placement_selected() &&
	    (!is_generating || pro_emulation != 0 ||
	     cmp_order != CMP_ORDER_REFERENCE ||
	     context_tokens < DS41F_CONTEXT_WITHOUT_CANDIDATES ||
	     step_rows > LINEAR_MAXIMUM_ROWS ||
	     speculative < SPECULATIVE_DEVICE)) {
		(void)fprintf(
			stderr,
			execution_placement == GPU_SPLIT_TP4_EXL3 ?
				"TP4 EXL3 requires reference CMP products, context>=16384,"
				" decode rows<=8 and DSpark\n" :
				"TP requires native PRO, reference local CMP products, context>=16384,"
				" decode rows<=8 and DSpark\n");
		return PROGRAM_FAILED;
	}
	return PROGRAM_SUCCEEDED;
}
