/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_generate.c: a prompt's generation and the capture (split from commands.c by whole items).
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

/* generate: the prompt as one command, a line for each token its events bring, then the figures. True when it
 * failed. */
MUST_CHECK bool prompt_failed_to_generate(const uint32_t *token_ids,
					  uint32_t token_count)
{
	const struct generation_command_record command = {
		.new_token_ids = token_ids,
		.sampler_seed = seed,
		.new_token_count = token_count,
		.generated_token_limit = generated_token_limit,
		.sampler_temperature = temperature,
		.sampler_top_p = top_p
	};
	struct generation_event event = { 0 };
	int descriptor = -1;
	uint32_t generated = 0;
	bool is_available = false;
	(void)printf("prompt ids");
	for (uint32_t index = 0; index < token_count; index++) {
		(void)printf(" %" PRIu32, token_ids[index]);
	}
	(void)printf("\n");
	const double started_seconds = monotonic_seconds();
	double first_token_seconds = 0.0, token_seconds = started_seconds;
	result_status status = engine_event_descriptor(engine, &descriptor);
	status = status != STATUS_OK ? status : engine_submit(engine, &command);
	while (status == STATUS_OK &&
	       event.event_kind == GENERATION_EVENT_TOKEN) {
		struct pollfd poll_entry = {
			descriptor, POLLIN, 0
		}; /* the engine bounds every step */
		status = poll(&poll_entry, 1, -1) < 0 && errno != EINTR ?
				 STATUS_INPUT_OUTPUT_ERROR :
				 engine_next_event(engine, &event,
						   &is_available);
		if (status == STATUS_OK && is_available &&
		    event.event_kind == GENERATION_EVENT_TOKEN) {
			const double now_seconds = monotonic_seconds();
			(void)printf("token %" PRIu32 " %" PRIu32
				     " %.9g %.6f\n",
				     generated, event.token_id,
				     (double)event.logit,
				     now_seconds - token_seconds);
			token_seconds = now_seconds;
			first_token_seconds =
				generated == 0 ? now_seconds - started_seconds :
						 first_token_seconds;
			generated++;
		}
		status = status == STATUS_OK &&
					 event.event_kind ==
						 GENERATION_EVENT_FAILED ?
				 (result_status)event.status :
				 status;
	}
	const double decode_seconds =
		monotonic_seconds() - started_seconds - first_token_seconds;
	if (status ==
	    STATUS_OK) { /* a run that failed ends on its error line */
		(void)printf("prompt of %" PRIu32
			     " tokens and first token %.3f s, %" PRIu32
			     " more tokens %.3f s: %.2f "
			     "tokens/s\n",
			     token_count, first_token_seconds,
			     generated > 0 ? generated - 1u : 0, decode_seconds,
			     generated > 1 && decode_seconds > 0.0 ?
				     (double)(generated - 1u) / decode_seconds :
				     0.0);
	}
	return status != STATUS_OK && step_failed("run a step", status);
}

/* capture: the cases of golden_capture_options through the engine, on the driver's first thin program's parts. */
MUST_CHECK bool capture_failed(int argument_count, char **arguments)
{
	const struct execution_program *thin = NULL;
	for (uint32_t index = 0; index < program_count && thin == NULL; index++)
		thin = programs[index]->kind == PROGRAM_THIN ? programs[index] :
							       NULL;
	const char *const map = copy_count != 0	   ? copy_paths[0] :
				model_pack != NULL ? model_pack->path :
						     "";
	const char *const slash = strrchr(map, '/');
	const struct golden_capture_run run = {
		.engine = engine,
		.driver = driver,
		.program = thin,
		.argument_count = argument_count,
		.arguments = arguments,
		.version = PROGRAM_ATTRIBUTION_NOTICE,
		.placement = placement_text != NULL ? placement_text :
						      "default",
		.weights = storage_operations == &storage_exl3 ? "exl3" :
								 "original",
		.map = slash != NULL ? slash + 1 : map,
		.pack_header = pack_header_of(pack),
		.pack_header_bytes = sizeof(struct pack_header),
		/* the map map_descriptor_check read the descriptor of: the instance's */
		.map_path = storage_operations != &storage_exl3 ?
				    NULL :
				    instance.map_path
	};
	return step_failed("capture", golden_capture_run(&run));
}
