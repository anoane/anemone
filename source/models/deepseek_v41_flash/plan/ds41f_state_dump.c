/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_state_dump.c: the draft ring and state digest dumps (split from commands.c by whole items).
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
#include "model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_model.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_peer.h"
#include "engine/cli/commands.h"
#include "engine/cli/cli_golden_capture.h"

#include "engine/cli/cli_internal.h"
#include "ds41f_state_dump.h"

#define DRAFT_RING_DUMP_ROWS \
	512u /* the dump's window rows of session 0, positions 0 .. 511, a draft stage */
#define DIGEST_MIX_PRIME \
	1099511628211ull /* FNV-1a 64: mixes a layer's tensor digests in order */
#define DRAFT_LAST_STAGE \
	2u /* the dump's third draft stage ring (DS41F_DRAFT_STAGES) */
/* the dump's four pieces of the draft logits' first 2 MiB */
enum draft_logits_piece {
	DRAFT_LOGITS_FIRST,
	DRAFT_LOGITS_SECOND,
	DRAFT_LOGITS_THIRD,
	DRAFT_LOGITS_FOURTH
};

/* generate diagnostics (--dump_draft_ring): on the result card, session 0's window rows 0 .. DRAFT_RING_DUMP_ROWS - 1
 * of each draft stage's ring, stage by stage, then the last draft's rotations (8 rows), step area, collapse rows,
 * result areas, normed rows (HIDDEN, 8 rows) and Markov rows, to path, behind the generation. True when it failed. */
MUST_CHECK bool ds41f_draft_ring_dump_failed(const char *path)
{
	const uint32_t card = plan->result_card;
	const uint64_t bytes =
		(uint64_t)DRAFT_RING_DUMP_ROWS * DS41F_STATE_ROW_BYTES;
	const uint64_t positions = plan->limits.draft_positions,
		       *const arrays = plan->cards[card].array_offset;
	const struct linear_range ranges[] = {
		{ plan->cards[card].ring_offset[DS41F_DRAFT_FIRST_LAYER],
		  bytes },
		{ plan->cards[card].ring_offset[DS41F_DRAFT_FIRST_LAYER + 1u],
		  bytes },
		{ plan->cards[card].ring_offset[DS41F_DRAFT_FIRST_LAYER +
						DRAFT_LAST_STAGE],
		  bytes },
		{ arrays[DS41F_ARRAY_ROTATION],
		  (uint64_t)(LINEAR_MAXIMUM_ROWS * DS41F_ROTATION_ROW_BYTES) },
		{ arrays[DS41F_ARRAY_DRAFT_STEP],
		  STEP_BLOCK_BYTES_VALUE + positions * STEP_ROW_BYTES_VALUE },
		{ arrays[DS41F_ARRAY_DRAFT_COLLAPSE],
		  positions * DS41F_HIDDEN_ROW_BYTES },
		{ arrays[DS41F_ARRAY_DRAFT_RESULTS],
		  positions * STEP_DRAFT_RESULT_BYTES },
		{ arrays[DS41F_ARRAY_HIDDEN],
		  (uint64_t)(LINEAR_MAXIMUM_ROWS * DS41F_HIDDEN_ROW_BYTES) },
		{ arrays[DS41F_ARRAY_DRAFT_MARKOV],
		  positions * DS41F_MARKOV_RANK * sizeof(uint16_t) },
		/* the draft logits' first 2 MiB (DIAG_DRAFT_PROBE's probes; else the logits) */
		{ arrays[DS41F_ARRAY_DRAFT_LOGITS], bytes },
		{ arrays[DS41F_ARRAY_DRAFT_LOGITS] + bytes, bytes },
		{ arrays[DS41F_ARRAY_DRAFT_LOGITS] + DRAFT_LOGITS_THIRD * bytes,
		  bytes },
		{ arrays[DS41F_ARRAY_DRAFT_LOGITS] +
			  DRAFT_LOGITS_FOURTH * bytes,
		  bytes },
	};
	void *staging = NULL;
	result_status status =
		backend_cuda_value.staging_allocate(bytes, &staging);
	FILE *const file = status == STATUS_OK ? fopen(path, "wb") : NULL;
	status = status == STATUS_OK && file == NULL ?
			 STATUS_INPUT_OUTPUT_ERROR :
			 status;
	for (uint32_t range = 0;
	     range < sizeof(ranges) / sizeof(ranges[0]) && status == STATUS_OK;
	     range++) {
		uint64_t ticket = 0;
		const uint64_t range_bytes = ranges[range].byte_count;
		status = range_bytes > bytes ?
				 STATUS_INVALID_ARGUMENT :
				 backend_cuda_value.download_begin(
					 cards[card], ranges[range].byte_offset,
					 staging, range_bytes, &ticket);
		status = status == STATUS_OK ?
				 backend_cuda_value.upload_wait(
					 cards[card], ticket,
					 CARD_WAIT_LIMIT_NANOSECONDS) :
				 status;
		status = status == STATUS_OK && fwrite(staging, 1, range_bytes,
						       file) != range_bytes ?
				 STATUS_INPUT_OUTPUT_ERROR :
				 status;
	}
	status = file != NULL && fclose(file) != 0 && status == STATUS_OK ?
			 STATUS_INPUT_OUTPUT_ERROR :
			 status;
	if (staging != NULL)
		backend_cuda_value.staging_free(staging);
	/* the weights the result card holds, digested (memory a step might overwrite): each walk layer's tensors mixed
	 * into one line (the draft stages' one a tensor too), the globals and the draft's globals one a line */
	for (uint32_t layer = 0;
	     layer < DS41F_WALK_LAYERS && status == STATUS_OK; layer++) {
		uint64_t mixed = 0;
		for (uint32_t index = 0;
		     index < DS41F_TENSOR_LAYER_COUNT && status == STATUS_OK;
		     index++) {
			const struct ds41f_tensor *const tensor =
				&plan->layer_tensors[layer][index];
			uint64_t digest = 0;
			if (tensor->stored.byte_count == 0 ||
			    tensor->card != card)
				continue;
			status = aligned_digest(card, tensor->stored, &digest);
			mixed = (mixed ^ digest) * DIGEST_MIX_PRIME;
			if (layer >= DS41F_DRAFT_FIRST_LAYER)
				(void)fprintf(stderr,
					      "weights draft stage %" PRIu32
					      " tensor %" PRIu32
					      " digest %016" PRIx64 "\n",
					      layer - DS41F_DRAFT_FIRST_LAYER,
					      index, digest);
		}
		(void)fprintf(stderr,
			      "weights layer %" PRIu32 " digest %016" PRIx64
			      "\n",
			      layer, mixed);
	}
	for (uint32_t index = 0;
	     index < DS41F_TENSOR_GLOBALS + DS41F_TENSOR_DRAFT_GLOBALS &&
	     status == STATUS_OK;
	     index++) {
		const struct ds41f_tensor *const tensor =
			index < DS41F_TENSOR_GLOBALS ?
				&plan->globals[index] :
				&plan->draft_globals[index -
						     DS41F_TENSOR_GLOBALS];
		uint64_t digest = 0;
		if (tensor->stored.byte_count == 0 || tensor->card != card)
			continue;
		status = aligned_digest(card, tensor->stored, &digest);
		(void)fprintf(stderr,
			      "weights global %" PRIu32 " digest %016" PRIx64
			      "\n",
			      index, digest);
	}
	return step_failed("dump the draft rings", status);
}

/* generate diagnostics (DIAG_STATE_DIGESTS in the environment): behind the generation, the digest of every
 * STATE_DIGEST_BLOCK_BYTES block of session 0's state on each card, a line each on stderr: per walk layer the card
 * holds, its ring, compressor, pool and key tiles (two runs' lines name the first layer, kind and block they differ
 * in: which rows' arithmetic depends on the chunking). True when it failed. */
#define STATE_DIGEST_BLOCK_BYTES 65536ull
#define STATE_DIGEST_KINDS 4u
MUST_CHECK bool ds41f_state_digests_failed(void)
{
	static const char *const kind_names[STATE_DIGEST_KINDS] = {
		"ring", "compressor", "pool", "keys"
	};
	result_status status = STATUS_OK;
	for (uint32_t card = 0; card < plan->card_count && status == STATUS_OK;
	     card++) {
		const struct ds41f_card *const holder = &plan->cards[card];
		const uint64_t layers = ds41f_walk_layers(holder);
		for (uint32_t layer = 0;
		     layer < DS41F_WALK_LAYERS && status == STATUS_OK;
		     layer++) {
			if ((layers >> layer & 1u) == 0)
				continue;
			const struct linear_range ranges[STATE_DIGEST_KINDS] = {
				{ holder->ring_offset[layer],
				  DS41F_RING_SESSION_BYTES },
				{ holder->compressor_offset[layer],
				  DS41F_COMPRESSOR_SESSION_BYTES },
				{ holder->pool_offset[layer],
				  plan->pool_session_bytes[layer] },
				{ holder->key_tiles_offset[layer],
				  plan->key_session_bytes[layer] }
			};
			for (uint32_t kind = 0;
			     kind < STATE_DIGEST_KINDS && status == STATUS_OK;
			     kind++) {
				for (uint64_t block = 0;
				     ranges[kind].byte_offset !=
					     PROGRAM_NOWHERE &&
				     block * STATE_DIGEST_BLOCK_BYTES <
					     ranges[kind].byte_count &&
				     status == STATUS_OK;
				     block++) {
					const uint64_t left =
						ranges[kind].byte_count -
						block * STATE_DIGEST_BLOCK_BYTES;
					const struct linear_range part = {
						ranges[kind].byte_offset +
							block * STATE_DIGEST_BLOCK_BYTES,
						left < STATE_DIGEST_BLOCK_BYTES ?
							left :
							STATE_DIGEST_BLOCK_BYTES
					};
					uint64_t digest = 0;
					status = aligned_digest(card, part,
								&digest);
					(void)fprintf(stderr,
						      "state card %" PRIu32
						      " layer %" PRIu32
						      " %s block %" PRIu64
						      " digest %016" PRIx64
						      "\n",
						      card, layer,
						      kind_names[kind], block,
						      digest);
				}
			}
		}
	}
	return step_failed("digest the state", status);
}
