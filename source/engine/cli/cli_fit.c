/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_fit.c: the fits: prompt token ids, row capacity, limits, slots, sessions and context, the TP plan's fit (split
 * from commands.c by whole items).
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

/* Explicit token input is a complete JSON uint32 array, with one context slot
 * left for generation. Tokenization and chat rendering are not applied. */
MUST_CHECK result_status prompt_token_ids_parse(const char *text,
						size_t byte_count,
						uint32_t vocabulary_size,
						uint32_t *token_ids_out,
						uint32_t token_capacity,
						uint32_t *token_count_out)
{
	struct json_value document = { 0 }, value = { 0 };
	if (text == NULL || vocabulary_size == 0 || token_ids_out == NULL ||
	    token_count_out == NULL || token_capacity <= 1u ||
	    json_read_document(text, byte_count, &document) !=
		    JSON_DOCUMENT_VALID ||
	    document.text[document.begin] != '[')
		return STATUS_INVALID_ARGUMENT;
	size_t cursor = document.begin;
	uint32_t count = 0;
	while (json_next(document, &cursor, NULL, &value)) {
		uint32_t token_id = 0;
		for (size_t offset = value.begin; offset < value.end;
		     offset++) {
			const char digit = value.text[offset];
			if (digit < '0' || digit > '9')
				return STATUS_INVALID_ARGUMENT;
			const uint32_t part = (uint32_t)(digit - '0');
			if (token_id > (UINT32_MAX - part) / DECIMAL_BASE)
				return STATUS_INVALID_ARGUMENT;
			token_id = token_id * DECIMAL_BASE + part;
		}
		if (token_id >= vocabulary_size)
			return STATUS_INVALID_ARGUMENT;
		if (count >= token_capacity - 1u)
			return STATUS_UNSUPPORTED;
		token_ids_out[count++] = token_id;
	}
	if (count == 0)
		return STATUS_INVALID_ARGUMENT;
	*token_count_out = count;
	return STATUS_OK;
}

/* The device runs its fixed block even when the host verifies fewer rows. Admission and its program check share
 * this capacity; ordinary plans retain their original capacity. */
uint32_t effective_row_capacity(void)
{
	uint32_t rows = wide_rows > step_rows ? wide_rows : step_rows;
	for (uint32_t index = 0; index < MOST_BATCH_PROGRAMS; index++)
		rows = batch_rows[index] > rows ? batch_rows[index] : rows;
	return speculative >= SPECULATIVE_DEVICE && draft_block > rows ?
		       draft_block :
		       rows;
}

/* The model's limits of the parsed options (run_program's, and the host tests'). */
struct program_limits options_limits(void)
{
	return (struct program_limits){
		.context_token_limit = context_tokens,
		.row_capacity = effective_row_capacity(),
		.session_capacity = session_slots,
		.state_width = STATE_WIDTH_16,
		.linear_profile = (enum linear_profile_code)linear_profile,
		.linear_order = (enum linear_order_code)linear_order,
		.cmp_order = cmp_order,
		.official_hc_rows = official_hc_rows,
		.official_reference = official_reference,
		.draft_positions =
			speculative >= SPECULATIVE_DEVICE ? draft_block : 0
	};
}

/* What the open adds to the plan's arenas after plan_create (session_fit's plan_grown, so the fit sees the arenas
 * that are opened, and the program dumps'): the original head offload's shards on every card, and with banked heads a
 * bank for each pipeline step. Nothing else grows an arena between the plan and the open (run_program, which names
 * each of the two as a step). */
MUST_CHECK result_status plan_grown(struct ds41f_plan *grown)
{
	result_status status =
		original_head_offload != 0 ?
			ds41f_original_head_plan(grown, descriptions) :
			STATUS_OK;
	if (status == STATUS_OK &&
	    original_head_offload == DS41F_HEAD_MODE_BANKED)
		status = ds41f_original_head_banks_plan(grown, pipeline_steps);
	return status;
}

#define FIT_BISECTION_PARTS 2u /* each trial halves what is left */

/* Whether limits' sessions fit, as session_fit says (STATUS_OUT_OF_MEMORY: they do not; any other refusal is the
 * placement's, the limits' or the plan's own); the TP plan is made, then released (no plan is left). */
MUST_CHECK static result_status slots_fit(const struct slot_fit *fit,
					  const struct program_limits *limits)
{
	if (!tensor_placement_selected())
		return session_fit(fit->cards, limits);
	const result_status status = ds41f_tensor_plan_prepare(
		limits, fit->kept_free_bytes, fit->unit_homes);
	if (status == STATUS_OK) {
		ds41f_tp2_model_destroy(tensor_model);
		tensor_model = NULL;
		plan = NULL;
	}
	return status;
}

/* A BATCH serve runs exactly STEP_BATCH_MAXIMUM_SESSIONS sessions (parse_arguments). */
bool batch_serving(void)
{
	return is_serving && (original_batch != 0 || batch_rows[0] != 0);
}

/* The most sessions of limits' context that fit, 1 to SERVING_SLOTS_MOST by bisection (the arena grows with the count;
 * a trial count's refusal is taken as not fitting); a BATCH serve tries only its STEP_BATCH_MAXIMUM_SESSIONS. 0 when
 * none does: then *refusal_out is the refusal of the last count tried (one session; BATCH's eight) other than
 * STATUS_OUT_OF_MEMORY, which no count changes, else STATUS_OK. */
uint32_t slots_most(const struct slot_fit *fit,
		    const struct program_limits *limits,
		    result_status *refusal_out)
{
	struct program_limits trial = *limits;
	result_status status = STATUS_OK;
	if (batch_serving()) {
		trial.session_capacity = STEP_BATCH_MAXIMUM_SESSIONS;
		status = slots_fit(fit, &trial);
		*refusal_out = status == STATUS_OUT_OF_MEMORY ? STATUS_OK :
								status;
		return status == STATUS_OK ? STEP_BATCH_MAXIMUM_SESSIONS : 0u;
	}
	/* one session first: a refusal other than memory there is the placement's own, which no count changes */
	uint32_t fits = 0, refused = SERVING_SLOTS_MOST + 1u;
	trial.session_capacity = 1u;
	status = slots_fit(fit, &trial);
	if (status == STATUS_OK)
		fits = 1u;
	else
		refused = 1u;
	while (refused - fits > 1u) {
		trial.session_capacity =
			fits + (refused - fits) / FIT_BISECTION_PARTS;
		if (slots_fit(fit, &trial) == STATUS_OK)
			fits = trial.session_capacity;
		else
			refused = trial.session_capacity;
	}
	*refusal_out = fits == 0u && status != STATUS_OUT_OF_MEMORY ? status :
								      STATUS_OK;
	return fits;
}

/* The longest context, in DS41F_CONTEXT_GRANULE steps up to limits' (TP: from DS41F_CONTEXT_WITHOUT_CANDIDATES), at
 * which limits' sessions fit, by bisection (a trial context's refusal is taken as not fitting): 0 when none does, with
 * *refusal_out as slots_most's. */
static uint32_t context_most(const struct slot_fit *fit,
			     const struct program_limits *limits,
			     result_status *refusal_out)
{
	struct program_limits trial = *limits;
	const uint32_t least = tensor_placement_selected() ?
				       DS41F_CONTEXT_WITHOUT_CANDIDATES /
					       DS41F_CONTEXT_GRANULE :
				       1u;
	uint32_t fits = least - 1u,
		 refused = limits->context_token_limit / DS41F_CONTEXT_GRANULE +
			   1u; /* in granules */
	result_status status = STATUS_OK;
	while (refused - fits > 1u) {
		const uint32_t middle =
			fits + (refused - fits) / FIT_BISECTION_PARTS;
		trial.context_token_limit = middle * DS41F_CONTEXT_GRANULE;
		status = slots_fit(fit, &trial);
		if (status == STATUS_OK)
			fits = middle;
		else
			refused = middle;
	}
	*refusal_out = fits < least && status != STATUS_OUT_OF_MEMORY ?
			       status :
			       STATUS_OK;
	return fits >= least ? fits * DS41F_CONTEXT_GRANULE : 0u;
}

/* The TP plan of limits' sessions. For --session_slots=max the most sessions its cards hold (slots_most), then their
 * plan; limits' session_capacity is that count, or 1 with STATUS_OUT_OF_MEMORY when not even one fits (or one
 * session's own refusal). A refused count makes nothing (ds41f_tp2_model_create destroys its partial model). */
MUST_CHECK result_status tensor_plan_fit(struct program_limits *limits,
					 uint64_t kept_free_bytes,
					 uint32_t *unit_homes)
{
	if (!session_slots_max)
		return ds41f_tensor_plan_prepare(limits, kept_free_bytes,
						 unit_homes);
	const struct slot_fit fit = { NULL, kept_free_bytes, unit_homes };
	result_status refusal = STATUS_OK;
	const uint32_t most = slots_most(&fit, limits, &refusal);
	limits->session_capacity = most != 0 ? most : 1u;
	return refusal != STATUS_OK ? refusal :
	       most == 0 ? STATUS_OUT_OF_MEMORY :
			   ds41f_tensor_plan_prepare(limits, kept_free_bytes,
						     unit_homes);
}

/* A TP pipeline's queue depth, once the session count is known (single_session_defaults may set it). */
void tp_queue_depth_report(void)
{
	if (tensor_placement_selected() && pipeline_steps > 1u)
		(void)printf(
			tp_pipeline_overlap != 0 ?
				"TP queue depth %u%s: separate CMP-prefix and native-PRO-tail workers\n" :
				"TP queue depth %u%s: one coordinator executes complete flights serially;"
				" no GPU overlap\n",
			pipeline_steps,
			pipeline_steps_defaulted ?
				" (the default for several serving slots with the device draft)" :
				"");
}

/* --session_slots=max resolved to one session: the defaults of --session_slots=1
 * (session_defaults), so one session runs what an explicit count of one runs, where one session still fits under them
 * (the PP fit again; the TP plan reads neither the depth nor the draft policy) and they admit the given knobs. NULL
 * then; else the multi-session defaults stay, and the reason. */
const char *single_session_defaults(const struct slot_fit *fit,
				    const struct program_limits *limits)
{
	const uint32_t depth = pipeline_steps,
		       policy = draft_policy.depth_policy;
	const bool defaulted = pipeline_steps_defaulted;
	session_defaults(1u);
	const char *const kept =
		tp_pipeline_overlap != 0 && pipeline_steps == 1u ?
			"--tp_pipeline_overlap needs two flights" :
		fit != NULL && !tensor_placement_selected() &&
				slots_fit(fit, limits) != STATUS_OK ?
			"one session does not fit under them" :
			NULL;
	if (kept != NULL) {
		pipeline_steps = depth;
		draft_policy.depth_policy = policy;
		pipeline_steps_defaulted = defaulted;
	}
	return kept;
}

/* The count --session_slots=max resolved to, with each card's free memory and budget (MiB); at one session, why the
 * multi-session defaults stay (kept: single_session_defaults' reason), if they do. */
void sessions_max_report(const struct program_limits *limits,
			 const uint64_t *budget_bytes_of_card, const char *kept)
{
	(void)printf("session slots: max -> %" PRIu32 " of %" PRIu32
		     " tokens each%s%s%s; cards (free/budget MiB):",
		     limits->session_capacity, limits->context_token_limit,
		     kept != NULL ? " (multi-session defaults kept: " : "",
		     kept != NULL ? kept : "", kept != NULL ? ")" : "");
	for (uint32_t card = 0; card < card_count; card++)
		(void)printf(
			" %" PRIu64 "/%" PRIu64,
			(uint64_t)(descriptions[card].memory_free_bytes /
				   BYTES_PER_MIB),
			(uint64_t)(budget_bytes_of_card[card] / BYTES_PER_MIB));
	(void)putchar('\n');
}

/* Every serve start: the most sessions of the context the cards hold, so the fitted count is always
 * visible. Where the selected row's set needs one session count (TP4_EXL3 BATCH's 8) and the serve runs its
 * concurrency=balanced row instead, the line says so. */
void sessions_fit_report(uint32_t most, const struct program_limits *limits,
			 const struct axis_choice *slots_unmet)
{
	if (slots_unmet != NULL)
		(void)printf("concurrency=%s: ",
			     slots_unmet->value[AXIS_CONCURRENCY_VALUE]);
	(void)printf("%" PRIu32 " session(s) fit at %" PRIu32
		     " tokens of context on this split",
		     most, limits->context_token_limit);
	if (!session_slots_max)
		(void)printf("; serving %" PRIu32 " (--session_slots)",
			     limits->session_capacity);
	(void)fputs("; a shorter --context_tokens gives more", stdout);
	if (slots_unmet != NULL)
		(void)printf("; BATCH needs exactly %u",
			     slots_unmet->serve_slots_exactly);
	(void)putchar('\n');
}

/* Not even one session of the context fits: refused, the context never shrunk. True (failed). */
bool sessions_max_refused(const struct program_limits *limits)
{
	(void)fprintf(stderr,
		      "session slots: max: not even one session of %" PRIu32
		      " tokens fits the cards; the context is never shrunk "
		      "(give a smaller --context_tokens or --session_slots)\n",
		      limits->context_token_limit);
	return true;
}

/* An explicit --session_slots=N that the cards do not hold at the context: refused before any load, with the fit's
 * numbers (the most that fit at this context, the longest context of N, each card's free memory and budget); the
 * count and the context are never shrunk. True (failed). */
bool sessions_refused(const struct slot_fit *fit,
		      const struct program_limits *limits,
		      const uint64_t *budget_bytes_of_card)
{
	result_status most_refusal = STATUS_OK, context_refusal = STATUS_OK;
	const bool batch = batch_serving(); /* BATCH runs its count only */
	const uint32_t most = batch ? 0u :
				      slots_most(fit, limits, &most_refusal);
	const uint32_t context = context_most(fit, limits, &context_refusal);
	(void)fprintf(stderr,
		      "session slots: %" PRIu32 " sessions of %" PRIu32
		      " tokens do not fit the cards",
		      limits->session_capacity, limits->context_token_limit);
	if (!batch)
		(void)fprintf(
			stderr,
			"; at this context %" PRIu32
			" fit (--session_slots=max, the default, takes them)",
			most);
	if (context != 0)
		(void)fprintf(stderr,
			      "; %" PRIu32
			      " sessions fit up to --context_tokens=%" PRIu32,
			      limits->session_capacity, context);
	(void)fprintf(stderr, "%s; nothing is shrunk; cards (free/budget MiB):",
		      most_refusal != STATUS_OK ||
				      context_refusal != STATUS_OK ?
			      " (a trial plan was refused)" :
			      "");
	for (uint32_t card = 0; card < card_count; card++)
		(void)fprintf(
			stderr, " %" PRIu64 "/%" PRIu64,
			(uint64_t)(descriptions[card].memory_free_bytes /
				   BYTES_PER_MIB),
			(uint64_t)(budget_bytes_of_card[card] / BYTES_PER_MIB));
	(void)fputc('\n', stderr);
	return true;
}
