/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_run.c: run_program and its step helpers (split from commands.c by whole items).
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
#include "cli_autotune.h"
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
#include "models/deepseek_v41_flash/options/ds41f_option_texts.h"
#include "models/deepseek_v41_flash/plan/ds41f_state_dump.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_plan.h"

#define BYTES_PER_GIB 1073741824.0
#define SALT_OFFSET \
	0xCBF29CE484222325ull /* FNV-1a, 64 bits: a snapshot's arithmetic salt */
#define SALT_PRIME 0x100000001B3ull
#define CMP_SALT_TAG \
	0x434D505F4F524445ull /* separates relaxed CMP placement from the native PRO's salt */
#define HOST_EXPERT_SALT_TAG \
	0x484F53545F455850ull /* HOST_EXP: host-expert layers keep reference expert order */
/* PRFXSTBL: distinct numerical identity; the old plan 0 salt is unchanged. */
#define PREFIX_STABLE_SALT 0x505246585354424Cull
static double step_started_seconds; /* when the step that runs now began */

/* Seconds on the monotonic clock; zero when it cannot be read. */
double monotonic_seconds(void)
{
	struct timespec now = { 0 };
	const bool clock_was_read = clock_gettime(CLOCK_MONOTONIC, &now) == 0;
	return clock_was_read ?
		       (double)now.tv_sec +
			       (double)now.tv_nsec / NANOSECONDS_PER_SECOND :
		       0.0;
}

/* Ends a step: its line with its duration, or the error line that names it. True when it failed: the run ends. */
bool step_failed(const char *step_name, result_status status)
{
	const double now_seconds = monotonic_seconds();
	if (status == STATUS_OK) {
		(void)printf("%-20s %9.3f s\n", step_name,
			     now_seconds - step_started_seconds);
	} else { /* a product output, as the line of a step done: release compiles the log hooks out */
		(void)fprintf(stderr, "the step \"%s\" failed: %s\n", step_name,
			      status_text(status));
	}
	step_started_seconds = now_seconds;
	return status != STATUS_OK;
}

/* Select descriptions only: a PRO is skipped without opening its CUDA arena. */
MUST_CHECK static result_status
exl3_card_order_prepare(const struct backend_operations *backend,
			uint32_t found)
{
	uint32_t selected[DS41F_TP4_RANKS], count = 0;
	for (uint32_t ordinal = 0; ordinal < found; ordinal++) {
		struct card_description_record description = { 0 };
		const result_status status =
			backend->card_describe(ordinal, &description);
		if (status != STATUS_OK)
			return status;
		if (description.arithmetic_class != ARITHMETIC_CLASS_SM_80 ||
		    strstr(description.name, "CMP") == NULL)
			continue;
		if (count == DS41F_TP4_RANKS)
			return STATUS_UNSUPPORTED;
		selected[count++] = ordinal;
	}
	if (count != DS41F_TP4_RANKS)
		return STATUS_UNSUPPORTED;
	(void)memcpy(card_order, selected, sizeof(selected));
	card_count = count;
	return STATUS_OK;
}

/* The cards a serve takes, as backend numbers and describes them (run_program's backend_cuda_value; a test's
 * stand-in of the workstation): with no --card_order every card in the backend's order, TP4_EXL3 its four CMPs
 * (exl3_card_order_prepare), each described into descriptions; refused for no card, more than BACKEND_MAXIMUM_CARDS or
 * a --card_order card the backend does not have. */
MUST_CHECK result_status
serve_cards_describe(const struct backend_operations *backend)
{
	const bool every_card_is_taken = card_count == 0;
	uint32_t cards_found = 0;
	result_status status = backend->card_count(&cards_found);
	if (status == STATUS_OK &&
	    (cards_found == 0 || cards_found > BACKEND_MAXIMUM_CARDS)) {
		/* No card to load onto, or more cards than these arrays hold. */
		status = STATUS_UNSUPPORTED;
	}
	if (status == STATUS_OK && every_card_is_taken &&
	    execution_placement == GPU_SPLIT_TP4_EXL3)
		status = exl3_card_order_prepare(backend, cards_found);
	for (; status == STATUS_OK && every_card_is_taken &&
	       execution_placement != GPU_SPLIT_TP4_EXL3 &&
	       card_count < cards_found;
	     card_count++) {
		card_order[card_count] = card_count;
	}
	for (uint32_t card = 0; card < card_count && status == STATUS_OK;
	     card++) {
		status = card_order[card] >= cards_found ?
				 STATUS_INVALID_ARGUMENT /* no such card */
				 :
				 backend->card_describe(card_order[card],
							&descriptions[card]);
	}
	return status;
}

/* Each card's arithmetic as the serve takes it, a line where it is not the reference's: the PRO's own unless
 * --pro_emulation=1 (the plan, programs and kernels follow the class), a CMP's relaxed order */
void serve_cards_arithmetic(void)
{
	for (uint32_t card = 0; card < card_count; card++) {
		if (descriptions[card].arithmetic_class ==
			    ARITHMETIC_CLASS_SM_80 &&
		    cmp_order == CMP_ORDER_RELAXED) {
			(void)printf(
				"card %" PRIu32
				": relaxed CMP trunk/expert order:"
				" sums and state bits change;"
				" --cmp_order=reference restores the reference order\n",
				card);
		}
		if (descriptions[card].arithmetic_class ==
			    ARITHMETIC_CLASS_SM_120 &&
		    pro_emulation == 0) {
			descriptions[card].arithmetic_class =
				ARITHMETIC_CLASS_SM_120_NATIVE;
			(void)printf(
				"card %" PRIu32
				": native arithmetic (its own tensor cores): outputs depend"
				" on the layers it holds; --pro_emulation=1: the same bits on every card,"
				" 2-4x slower\n",
				card);
		}
	}
}

/* The autotune stages after the copy, in their order (the owner's split of 9 Nov 2026): the load stage, then the
 * Engram stage; storage/autotune runs them from this table, the composition root's, and names neither */
const struct storage_autotune_stage cli_autotune_stages[CLI_AUTOTUNE_STAGES] = {
	{ "load", AUTOTUNE_STAGE_OPTION "load", storage_autotune_load_current,
	  storage_autotune_load_cli, AUTOTUNE_SECONDS_MOST, false },
	{ "engram", AUTOTUNE_STAGE_OPTION "engram",
	  storage_autotune_engram_current, storage_autotune_engram_cli,
	  AUTOTUNE_ENGRAM_SECONDS_MOST, true }
};

int run_program(int argument_count, char **arguments,
		const struct axis_arguments_record *axis_arguments)
{
	uint64_t budget_bytes_of_card[BACKEND_MAXIMUM_CARDS] = { 0 };
	uint64_t weight_budget_bytes_of_card[BACKEND_MAXIMUM_CARDS] = { 0 };
	struct model_description_record model_description = { 0 };
	struct text_tokenizer *tokenizer = NULL;
	char *front_end_files[FRONT_END_FILES] = { NULL };
	uint64_t front_end_byte_counts[FRONT_END_FILES] = { 0 };
	char *prompt = NULL;
	size_t prompt_byte_count = 0;
	uint32_t *token_ids = NULL;
	uint32_t token_count = 0;
	uint32_t *card_of_unit = NULL;
	uint32_t on_demand_units_per_owner = 0;
	result_status status = STATUS_OK;
	double first_step_started_seconds = 0.0;
	bool failed = false;
	/* From the first log, a closed pipe is an error, not a signal.
	 * Descriptors 0 to 2 are open first: the lowest free descriptor proves those below it are open. */
	(void)signal(SIGPIPE, SIG_IGN);
	for (int descriptor = STDIN_FILENO; descriptor <= STDERR_FILENO;
	     descriptor++) {
		if (fcntl(descriptor, F_GETFD) < 0 &&
		    open("/dev/null", O_RDWR) != descriptor) {
			LOG_ERROR(
				"descriptor %d is closed and could not be opened",
				descriptor);
			return PROGRAM_FAILED;
		}
	}
	(void)setvbuf(stdout, NULL, _IOLBF,
		      0); /* a step's line appears when the step is done */
	if (argument_count == ARGUMENTS_BEFORE_THE_OPTIONS &&
	    strcmp(arguments[1], "--version") == 0)
		return fputs(PROGRAM_ATTRIBUTION_NOTICE, stdout) == EOF ?
			       PROGRAM_FAILED :
			       PROGRAM_SUCCEEDED;
	if (argument_count > 1 && strcmp(arguments[1], "stats") == 0) {
		return stats_client_cli(argument_count, arguments);
	}
	if (argument_count > 1 && strcmp(arguments[1], "keygen") == 0) {
		return server_keygen_cli(argument_count, arguments);
	}
	if (argument_count > 1 &&
	    (strcmp(arguments[1], "storage-profile") == 0 ||
	     strcmp(arguments[1], "storage-device") == 0 ||
	     strcmp(arguments[1], "storage-room") == 0 ||
	     strcmp(arguments[1], "storage-report") == 0 ||
	     strcmp(arguments[1], "storage-scan") == 0))
		return storage_profile_cli(argument_count, arguments);
	if (argument_count > 1 && strcmp(arguments[1], "autotune") == 0)
		return storage_autotune_cli(argument_count, arguments,
					    cli_autotune_stages,
					    CLI_AUTOTUNE_STAGES);
	if (argument_count > 1 && strcmp(arguments[1], "replicate") == 0) {
		return replicate_cli(argument_count, arguments);
	}
	if (argument_count == ARGUMENTS_BEFORE_THE_OPTIONS + 1 &&
	    strcmp(arguments[1], "serve") == 0 &&
	    strcmp(arguments[ARGUMENTS_BEFORE_THE_OPTIONS],
		   "--help-scheduling") == 0) {
		scheduling_usage(stdout);
		return PROGRAM_SUCCEEDED;
	}
	if (!parse_arguments(argument_count, arguments)) {
		/* The axis lines first, so a refused knob is traced to the axis value that inserted it. */
		axis_log(axis_arguments, stderr);
		if (is_capturing)
			(void)fputs(
				"usage: anemone capture [generate's options and axis flags] --capture_golden=ROOT"
				" --capture_cases=KEYS --capture_out=OUT [--capture_rows=1..8 (default"
				" speculative_depth + 1)] [--capture_positions=N (default all)] [--capture_free=0|1]"
				" TABLES PACK | --model_dir=FOLDER\n",
				stderr);
		/* Separate usage strings stay within ISO C99's 4095-byte literal limit. */
		(void)fputs(
			"usage: anemone load|generate|serve [--card_order=1,0,2]"
			" [--numerics=exact|relaxed|fast (default exact; relaxed/fast enable fixed alternate CMP sums)]"
			" [--placement=pp5|tp4_pro|tp4_exl3|tp2_pp2_pro|tp2_single_single_pro"
			" (default pp5; TP experimental)]",
			stderr);
		(void)fputs(ds41f_usage_lines[DS41F_USAGE_EXL3_EXPERT_PARTS],
			    stderr);
		(void)fputs(
			" [--weights=original|exl3 (default: what --placement and the pack imply; alone, exl3 selects"
			" tp4_exl3; a conflicting --placement or pack refuses)]"
			" [--model_dir=FOLDER (tokenizer.bin and original.anemone or NAME.exl3map (--exl3_map)"
			" from FOLDER, in place of the tables and pack arguments; links are followed to regular files)]"
			" [--exl3_map=NAME (with --model_dir and the EXL3 map: NAME.exl3map in FOLDER, default model;"
			" one folder holds several maps of one model)]"
			" [--exl3_unrated_map=0|1 (default 0; 1: an EXL3 map whose descriptor records no quality"
			" (quality none), or with no descriptor and a pack the model does not list as established, is"
			" served with no quality claim; refused otherwise)]",
			stderr);
		(void)fputs(ds41f_usage_lines[DS41F_USAGE_TP_QUERIES_PIECE],
			    stderr);
		(void)fputs(
			" [--tp_parallel_launch=0|1 (default 0; TP: the cards' graphs of"
			" a step's group launched from a thread a card, not in turn; the same launches a card, the same"
			" bits)]",
			stderr);
		(void)fputs(ds41f_usage_lines[DS41F_USAGE_TP_GRAPH_PARTS],
			    stderr);
		(void)fputs(
			" [--batch_rows=R[,R..] (1..64 each; tp4_exl3 with --pipeline_steps=1: BATCH programs of those"
			" rows in place of the thin ones, several sessions' rounds a step; 8 rows or fewer walk as thin"
			" programs)]",
			stderr);
		(void)fputs(ds41f_usage_lines[DS41F_USAGE_ORIGINAL_BATCH],
			    stderr);
		(void)fputs(
			" [--batch_verifiers=V[,V..] --batch_step_us=0|US[,US..] (0: off; qualified BF16/DSpark5 menu"
			" defaults for rows 8,8,12,12,16,24,32,48 / verifiers 1,2,2,4,4,4,8,8 when policy is omitted)]"
			" [--batch_plan_ties=0|1 (default 0; 1: the plan verifies drafts of equal reach all or none;"
			" drafts only)]",
			stderr);
		(void)fputs(
			ds41f_usage_lines[DS41F_USAGE_EXL3_TC_PRODUCT_BLOCKS],
			stderr);
		(void)fputs(
			" [--prefill_ladder_1536=0|1 (default 1; with --wide_rows above 1536:"
			" programs of 1536 rows in the ladder too, the exact plan's same bits in fewer steps)]"
			" [--prefill_ladder_fine=0|1 (default 1 with --wide_rows=2048 and no BATCH; 1792, 1280 and 768"
			" rows alike)]",
			stderr);
		/* the load's options: a literal of their own (each at most 4095 bytes, ISO C99) */
		(void)fprintf(
			stderr,
			" [--tp_load_threads=0..%u (default %u; TP placements: the cards' load on N workers beside the"
			" host arena's, the same bytes; 0: the serial loads)]"
			" [--tp_load_reads_per_copy=0..%u (default %u; reads in flight a copy at most, 0: no limit)]"
			" [--tp_load_host_threads=0..%u (default %u; the host arena's threads; 0: as many as"
			" --tp_load_threads)] [--tp_load_copy_split=0..2 (default 2: the host arena's chunks prefer"
			" copy 0, the cards' extents the last copy, and a role that has claimed its last read leaves"
			" its copy to the other; 1: each role only its preferred copy; 0: each read from the copy"
			" soonest to finish it)]"
			" [--storage_profile=PATH|off (default auto reload; never auto tune)]"
			" [--replica_external=0|1 (default 0; networks never qualify)]"
			" [--replica_floor_percent=0..99 (default 20)]"
			" [--engram_ram_percent=auto|0..100 (default auto: 100 when the tables fit MemAvailable,"
			" else the largest whole share that fits, else 0; immutable tables only)]"
			" [--engram_fit_only=0|1 (generate: print the Engram fit and stop before any card)]"
			" [--engram_sources=PACK[,PACK...] (optional independent table sources)]"
			" [--engram_io=0|1 (default 0; 1: prepared io_uring batches)]"
			" [--engram_cache_mib=0..32768 (default 0; bounded disk block cache)]"
			" [--turbo-nvme-engrams=0|1 (default 0; interleave admitted full replicas)]"
			" [--engram_step_io=0|1 (default 1 serving PP5 or TP4_PRO from disk tables, else 0; 1: a step's"
			" blocks read at once on I/O threads per copy)]"
			" [--engram_io_threads=1..4 (default 2)] [--engram_spin_us=0..100000 (default 200)]"
			" [--engram_io_pin=0|1 (default 1)]"
			" [--engram_scan_threads=0..4 (default 2: the startup digest scan reads every copy at once"
			" with two readers a copy; 0: one reader, copy by copy)]"
			" [--engram_hot_set=FILE --engram_hot_sha256=HEX (a static hot set of table blocks, from"
			" `engram_io_bench hot_build`, held in RAM and served like the RAM prefix; default none)]"
			" [--engram_hot_mib=1..65536 (default 512: the most RAM the hot set may hold)]"
			" [--turbo_load=0|1 (default 1: the source's receipted replicas, from `anemone replicate`,"
			" on drives that hold none of the given copies are read beside them; 0: exactly the given"
			" copies)]"
			" [--load_tuned_weights=0|1 (default 0: with a storage profile the readers steal pieces;"
			" 1: its weights pick each piece's copy)]"
			" [--map_reads=0..4 (default 4: 3 with replicas, else 1; 3: an EXL3 map's aligned pieces"
			" O_DIRECT, the rest buffered then dropped from the page cache; 0: all buffered and kept; 1:"
			" all buffered then dropped; 2: aligned O_DIRECT, rest kept)]",
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_THREADS,
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_READS_PER_COPY,
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_HOST_THREADS);
		(void)fprintf(
			stderr,
			" [--reader_threads_per_copy=1..%d (default %d: the PP loader's readers a copy)]"
			" [--card_memory_kept_free_mib=N]"
			" [--load_piece_mib=0..64 (default 0: 16 MiB; both loaders' read: the PP piece and slot, the TP"
			" host chunk; the knob wins over a profile, a profile over the default)]"
			" [--pro_emulation=0|1 (default 0: the PRO's own arithmetic, bits depend on the placement)]"
			" [generate and serve: --rows=1..%u"
			" --context_tokens=1..%u (default %u: 1M a session, a multiple of 16)"
			" --generated_tokens=N --seed=N --temperature=X --top_p=X"
			" --reasoning_effort=0..%d --linear_profile=0|1 --linear_order=0|1",
			LOADER_MAXIMUM_READER_THREADS_PER_COPY,
			LOADER_DEFAULT_READER_THREADS_PER_COPY,
			LINEAR_MAXIMUM_ROWS, DS41F_CONTEXT_MOST,
			DS41F_CONTEXT_MOST, DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT);
		(void)fputs(ds41f_usage_lines[DS41F_USAGE_OFFICIAL_HC_ROWS],
			    stderr);
		(void)fprintf(
			stderr,
			" [--cmp_order=reference|relaxed (default reference;"
			" relaxed changes CMP trunk/expert sums and state bits)]"
			" --step_graphs=0|1 (default 0/eager) --pipeline_steps=1..%u"
			" (default 5 for PP5 multi-session device THIN serving, 1 elsewhere)"
			" [--tp_pipeline_overlap=0|1 (default 0; 1: original TP4_PRO prefix and native tail on"
			" disjoint workers, depth>=2; per-flight downloads retire before prefix scratch reuse)]"
			" --prefill_cone=0|1"
			" --wide_rows=0|64..%u (default 0; with --prefill_cone=1) --full_steps=0|1 (default 1)"
			" [--prefill_margin=1..3967 (default 127; 3967 the reference's; with --chunks_in_flight=1,"
			" the exact plan"
			" (64-row chunks only where 1,024-row chunks, two in flight, take them): full rows beyond"
			" a prompt's cone; fewer: more front rows, the same bits, a kept prefix there runs a cone"
			" again)] [--prefill_fast_plan=0|1 (default 0; 1 with --chunks_in_flight=1: the largest"
			" chunk that fits, the reference tail's rows in larger chunks too: other bits)]"
			" [--prefill_order=0|1|2|3 (bounded BATCH default 2, else 0; 3: shortest with"
			" concurrent short tails;"
			" 2: prompts one at a time,"
			" the one with the fewest rows left first; 1: in their commands' order, each first token behind"
			" its own prompt; 0: the turns, a chunk each; the same chunks)] [--prefill_order_wait=0..1024"
			" (default 16; with 2: after N"
			" steps of shorter prompts in a row, a step of the oldest; 0: none)]"
			" [--prefill_cone_skip=0..3 (default 1: a full step's rows past the cone skip the decoder"
			" layers' cores, routed experts and wide trunks, the same bits; a later prompt keeping"
			" --prefill_margin tokens fewer or more runs no cone again; 2: the last row's cone, one fewer;"
			" 3: 1 with every trunk; 0: none skip)]"
			" --chunks_in_flight=1..%u (default 1, the exact plan; 2 the reference plan)"
			" [--prefill_first=0|1] [--draft_learned_acceptance=0|1 (default 1 for tp4_exl3, 0 elsewhere)]"
			" [--draft_session_acceptance=0..11 (default 0)]"
			" [--draft_start_economic=0|1 (default 1; 1: a"
			" command's first DSpark round at the policy's depth, not a probe of one draft)]"
			" --speculative=0|1|2|3 (default 0; with --rows=2 or more) --draft=lookup|dspark|hybrid"
			" --draft_block=5..7 (default 5)"
			" --draft_positions=1..%u (default %u, at most --rows less 1)"
			" --speculative_depth=N (alias; device draft at most --draft_block;"
			" multi-session needs pipeline)"
			" --draft_policy=0..3 (fixed|cost|rate|adaptive; tp4_exl3 default 0 single-slot THIN, 2"
			" BATCH/multiple; elsewhere 1) --draft_threshold=0..1000 (default 100)"
			" --draft_base_us=1..1000000000 (tp4_exl3 default 14300, elsewhere 12700)"
			" --draft_tail_us=0..1000000000 (tp4_exl3 default 0, elsewhere 2200)"
			" --draft_row_us=1..1000000000 (tp4_exl3 default 690, elsewhere 2700)"
			" --draft_probe_rounds=1..1048576 (default 8)"
			" --hybrid_lookup_prior=0..1000 (default 500, weight two reached outcomes)"
			" --draft_calibration=FILE (112 probabilities per thousand: mode/position/bin)"
			" --hybrid_lookup_min=1..16 (default 3; suffix match, at most --lookup_ngram_max)"
			" --lookup_ngram_max=1..%u (default %u) --lookup_ngram_min=1..max (default %u)]"
			" [--layout=<placement text (placement.h), in place of the fill>]",
			STEP_PIPELINE_MOST, STEP_WIDE_MOST_ROWS,
			STEP_PIPELINE_MOST, SERVING_DRAFT_MOST,
			SERVING_DRAFT_MOST, DRAFT_LOOKUP_MOST_NGRAM,
			DEFAULT_LOOKUP_NGRAM_MOST, DEFAULT_LOOKUP_NGRAM_LEAST);
		(void)fprintf(
			stderr,
			" [serve: --session_slots=1..%u|max (default max: as many sessions as fit at the"
			" context, no cap)"
			" --queue_depth=1..%u (default %u)"
			" --user_queue_depth=1..queue_depth (default half queue_depth, at least 1)"
			" --draft_usage=none|compact|full (default compact)"
			" --dsml_parsing=tolerant|reference (default tolerant)"
			" --bind_address=%s --port=1..%u --key_file=<file> --model_alias=<name>"
			" --session_directory=<snapshots of sessions>"
			" --prefix_reuse=0|1 (default 1; 0 disables GPU prefix reuse and refuses session_directory)"
			" --prefix_reuse_exact=0|1|2 (default 0; 1 final-row repeat; 2 cold boundaries;"
			" fixed exact plan; prefill_chunk_rows=prefill_decode_rows=0;"
			" multiple slots require prefill_first=1 or prefill_order; no snapshots)"
			" --metrics=0|1 --live_stats=0|100..60000 --live_stats_format=text|json|table"
			" --stats_users=<names>] [generate: <messages file>]"
			" [generate and serve: <front-end tables file>] <pack copy> [... up to %d copies]\n",
			SERVING_SLOTS_MOST, SERVER_MOST_WAITING,
			SERVER_DEFAULT_QUEUE_DEPTH, SERVER_DEFAULT_ADDRESS,
			SERVER_MOST_PORT, STORAGE_MAXIMUM_COPIES);
		(void)fputs(
			"       anemone stats [--host=127.0.0.1] --port=N [--key_file=F]"
			" [--interval_ms=1000] [--format=text|json|table] [--once]\n"
			"       anemone keygen --user=<username>\n",
			stderr);
		/* the tail programs' options: a literal of their own (each at most 4095 bytes, ISO C99) */
		(void)fputs(
			"       serve caches: [--cache_scope=0|1|2 (default 0: API user plus session/cache signals;"
			" 1: API user; 2: global)] [--session_grace_seconds=0..3600"
			" (default 30; 0 disables affinity grace)]"
			" [--anthropic_messages=0|1 (default 0; bounded Messages API)]"
			" [--session_fork=0|1 (default 0; exact same-owner independent slot clone, cache_scope=0)]"
			" [--session_save_tokens=N (default 1024)] [--session_load_tokens=N (default 256)]"
			" [--session_idle_save_seconds=1..86400 (default 120)]\n",
			stderr);
		(void)fputs(
			"       generate and serve, the exact plan's tail: [--prefill_tail_rows=0|128|320|448"
			" (default 128, 448 with --wide_rows=2048 and no BATCH, where it"
			" applies: tp4_exl3, --chunks_in_flight=1, --wide_rows of 1024 or more, --exl3_tc_moe=1,"
			" --tp_trunk_mma=1: the exact plan's 64-row tail chunks two (320: five, 448: seven or five,"
			" else two) a step, each in the 64-row chunk's forms; 0 one a step)]"
			" [--prefill_ladder_1024=0|1 (default 1; 1 with --wide_rows above 1024: programs of 1024 rows"
			" in the ladder too, the exact plan's same bits in fewer steps)]"
			" [--prefill_front_tail=0|1|2 (default 1 where the tail program applies, 2 with the 448-row"
			" tail: a wide one for the front region's tail too, its bits; 2: its 320- and 448-row ones and"
			" the thin-tail ones too)]",
			stderr);
		(void)fputs(ds41f_usage_lines[DS41F_USAGE_PREFILL_TAIL_OVERLAY],
			    stderr);
		(void)fputs(
			" [--prefill_thin_tail=0|1 (default 1 with the tail program and no BATCH: full programs of 16"
			" to 56 rows take the tail's"
			" rows short of a 64-row chunk in one step, each row in a thin step's forms)]",
			stderr);
		(void)fputs(ds41f_usage_lines[DS41F_USAGE_DENSE_STAGED],
			    stderr);
		(void)fputs(
			" [serve, with --prefill_order=1|2|3: --prefill_decode_every=0..64 (default 0; N: after N chunk"
			" steps of the head prompt in a row, a step of the decoding sessions first; 0: the turns"
			" alone)]\n",
			stderr);
		(void)fputs(
			"       TP n-gram gathers: [--engram_shared=0|1 (default 0;"
			" identical table masks share host rows)]\n"
			"       exact prefill plan: [--prefill_chunk_rows=0|256..2048 (whole 256-row granules;"
			" default 0; cap chunks before the original tail)] [serve: --prefill_decode_rows=0|256..2048"
			" (BATCH recipe default 1024, other recipes 0; cap while a decoder is ready;"
			" --prefill_decode_every=N overrides the interval)]"
			" [--prefill_decode_rounds=1..64 (ordinary decode rounds per interval;"
			" see --help-scheduling for defaults)]"
			" [--prefill_decode_waiters=0|1 (BATCH default 0, others 0; 1: original chunks and one decode"
			" round while"
			" multiple prompts remain)]\n",
			stderr);
		(void)fputs(
			"       scheduling examples and measured tradeoffs: anemone serve --help-scheduling\n",
			stderr);
		(void)fputs("       prefix-stable prefill: "
			    "[--prefill_prefix_stable=0|1 (default 0; "
			    "experimental TP4 BF16"
			    " fixed-capacity plan at wide_rows 1024/2048; new "
			    "cold references; one chunk, caps 0,"
			    " reserved prefill; no images or snapshots)]\n",
			    stderr);
		(void)fputs(
			"       generate explicit tokens: anemone generate [OPTIONS] --prompt_token_ids=IDS_JSON"
			" TABLES WEIGHTS (one pack; replaces MESSAGES; JSON uint32 array, no chat formatting).\n"
			"       generate/serve: --ignore_eos=0|1 (default 0; non-speculative THIN only;"
			" 1 continues after end-token draws to the token/context limit).\n",
			stderr);
		(void)fputs(
			"       anemone replicate [--rate_mib=N] [--floor_gib=N] [--prune=0|1] SOURCE [DIRECTORY ...]"
			" (turbo load's replicas: replicas.h)\n",
			stderr);
		(void)fputs("       anemone --version\n", stderr);
		return PROGRAM_WRONG_USAGE;
	}
	/* Product attribution stays on stderr; generated-token stdout retains its protocol. */
	if (fputs(PROGRAM_ATTRIBUTION_NOTICE, stderr) == EOF)
		return PROGRAM_FAILED;
	storage_profile_apply(argument_count, arguments);
	if (!storage_profile_is_valid && !storage_profile_is_off)
		storage_autotune_hint(stderr);
	storage_profile_roles(tuned_load_active, tuned_engram_active,
			      engram_ram_percent < ENGRAM_RAM_PERCENT);
	axis_log(axis_arguments, stderr);
	if (!engram_settings_resolve())
		return PROGRAM_WRONG_USAGE;
	const int placement_status = startup_placement_status();
	if (placement_status != PROGRAM_SUCCEEDED)
		return placement_status;
	first_step_started_seconds = step_started_seconds = monotonic_seconds();
	if (model_dir != NULL) {
		failed = step_failed("resolve the model folder",
				     model_folder_files(is_generating));
		status = failed ? STATUS_INPUT_OUTPUT_ERROR : status;
	}
	for (uint32_t file = 0;
	     is_generating && status == STATUS_OK && file < FRONT_END_FILES;
	     file++) {
		front_end_files[file] =
			front_end_paths[file] != NULL ?
				file_read(
					front_end_paths[file],
					file == TABLES_FILE &&
							model_dir != NULL ?
						&model_files
							[CLI_MODEL_FILE_TABLES] :
						NULL,
					&front_end_byte_counts[file]) :
				NULL;
		status = front_end_paths[file] != NULL &&
					 front_end_files[file] == NULL ?
				 STATUS_INPUT_OUTPUT_ERROR :
				 status;
	}
	status = !is_generating || status != STATUS_OK ?
			 status :
			 tokenizer_open(front_end_files[TABLES_FILE],
					front_end_byte_counts[TABLES_FILE],
					&tokenizer);
	if (is_serving) {
		const struct server_options options = {
			.storage_replica_info = storage_replica_info,
			.storage_replica_info_bytes =
				storage_replica_info_bytes,
			.bind_address = bind_address,
			.anthropic_messages = anthropic_messages != 0,
			.key_file = key_file,
			.stats_users = stats_users,
			.model_alias = model_alias,
			.disable_prefix_reuse = prefix_reuse == 0,
			.exact_prefix_reuse = prefix_reuse_exact,
			.cache_scope = cache_scope,
			.session_fork = session_fork,
			.session_grace_seconds = session_grace_seconds,
			.session_save_tokens = session_save_tokens,
			.session_load_tokens = session_load_tokens,
			.session_idle_save_seconds = session_idle_save_seconds,
			.session_directory = session_directory,
			.session_directory_bytes =
				SERVER_SESSION_DIRECTORY_BYTES,
			.metrics = metrics != 0,
			.live_stats_interval_ms = live_stats_interval_ms,
			.live_stats_json = strcmp(live_stats_format, "json") ==
					   0,
			.live_stats_table =
				strcmp(live_stats_format, "table") == 0,
			.card_order = stats_card_order,
			.placement = stats_placement,
			.port = port,
			.queue_depth = queue_depth,
			.user_queue_depth = user_queue_depth,
			.defaults = { .seed = seed,
				      .max_tokens = generated_token_limit,
				      .reasoning_effort = reasoning_effort,
				      .draft_usage =
					      strcmp(draft_usage, "full") == 0 ?
						      CHAT_DRAFT_USAGE_FULL :
					      strcmp(draft_usage, "none") == 0 ?
						      CHAT_DRAFT_USAGE_NONE :
						      CHAT_DRAFT_USAGE_COMPACT,
				      .temperature = temperature,
				      .top_p = top_p,
				      .is_seeded = is_seed_given,
				      .is_calls_reference =
					      strcmp(calls_parsing,
						     "reference") == 0 }
		};
		failed = step_failed("open the tables and the socket",
				     status != STATUS_OK ?
					     status :
					     server_open(&options, &server));
	} else if (is_generating && !is_capturing) {
		const struct model_prompt_options options = {
			.reasoning_effort =
				reasoning_effort != 0 ?
					reasoning_effort :
					DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT,
			.is_thinking = reasoning_effort != 0,
			.drops_thinking = true
		};
		const size_t prompt_capacity =
			(size_t)context_tokens * TOKENIZER_MAXIMUM_TOKEN_BYTES;
		prompt = prompt_token_ids_path == NULL ?
				 malloc(prompt_capacity) :
				 NULL;
		token_ids = calloc(context_tokens, sizeof(*token_ids));
		struct model_prompt_spans trusted = { 0 };
		status = status != STATUS_OK ?
				 status :
			 (prompt_token_ids_path == NULL && prompt == NULL) ||
					 token_ids == NULL ?
				 STATUS_OUT_OF_MEMORY :
			 prompt_token_ids_path != NULL ?
				 prompt_token_ids_parse(
					 front_end_files[MESSAGES_FILE],
					 (size_t)front_end_byte_counts
						 [MESSAGES_FILE],
					 tokenizer_vocabulary_size(tokenizer),
					 token_ids, context_tokens,
					 &token_count) :
				 ds41f_chat_template_render_spans(
					 front_end_files[MESSAGES_FILE],
					 front_end_byte_counts[MESSAGES_FILE],
					 NULL, 0, &options, prompt,
					 prompt_capacity, &prompt_byte_count,
					 &trusted);
		const bool is_rendered = status == STATUS_OK;
		status = is_rendered && prompt_token_ids_path == NULL ?
				 tokenizer_encode_spans(
					 tokenizer, prompt, prompt_byte_count,
					 trusted.spans, trusted.count,
					 token_ids, context_tokens,
					 &token_count) :
				 status;
		free(trusted.spans);
		/* Too long: render returns UNSUPPORTED, encode INVALID_ARGUMENT (its only refusal here).
		 * One more token must fit. */
		status = (is_rendered && status == STATUS_INVALID_ARGUMENT) ||
					 (status == STATUS_OK &&
					  (token_count == 0 ||
					   token_count == context_tokens)) ?
				 STATUS_UNSUPPORTED :
				 status;
		if (status == STATUS_UNSUPPORTED) {
			LOG_ERROR(
				"the prompt is empty, or it and one more token exceed --context_tokens");
		}
		failed = step_failed(prompt_token_ids_path != NULL ?
					     "read prompt token ids" :
					     "encode the prompt",
				     status);
	}
	if (!failed && copy_count != 0)
		storage_operations = pack_storage();
	/* auto opens the direct descriptors (DIRECT_DROPPED) and falls back to DROPPED when no replica is found
	 * below */
	storage_exl3_read_mode(map_reads == MAP_READS_AUTO ?
				       EXL3_READS_DIRECT_DROPPED :
				       (enum exl3_read_mode)map_reads);
	failed =
		failed ||
		step_failed("open the copies",
			    storage_operations->reader_open(
				    copy_paths, copy_count, &reader)) ||
		step_failed("open the pack",
			    pack_open_entry(storage_operations, reader, &pack));
	if (!failed && model_pack != NULL)
		failed = step_failed("check the folder's pack",
				     model_pack_opened(storage_operations,
						       reader,
						       pack_header_of(pack)));
	if (!failed && weights_family != WEIGHTS_UNSET)
		failed = step_failed("check the weight family",
				     pack_family_status(pack));
	if (!failed)
		model_instance_bind(pack,
				    model_pack != NULL ? model_pack->path :
							 pack_argument,
				    &instance);
	if (!failed && pack_holds_exl3(pack))
		failed = step_failed(
			"check the map's descriptor",
			map_descriptor_check(
				front_end_files[TABLES_FILE],
				front_end_byte_counts[TABLES_FILE]));
	if (!failed)
		failed = !turbo_load_copies(pack_header_of(pack));

	for (uint32_t copy = 0;
	     !failed && !tuned_load_active && copy < copy_count; copy++) {
		struct local_device_description device = { 0 };
		/* an EXL3 map: its largest source's drive */
		char bytes_path[EXL3_MAP_PATH_BYTES];
		const result_status described = local_device_describe(
			storage_device_path(copy_paths[copy], bytes_path),
			&device);
		const char *reason =
			described != STATUS_OK ?
				"device_disappeared_or_unknown" :
				local_device_media_refusal(
					&device, replica_external != 0);
		failed = step_failed(
			"record storage selection",
			storage_profile_note_loaded(stderr, &device, reason,
						    replica_floor_percent));
	}

	if (!failed && is_generating)
		failed = step_failed("fit the Engram tables in host memory",
				     engram_ram_fit());
	if (!failed && engram_ram_percent < ENGRAM_RAM_PERCENT) {
		status = engram != NULL ? STATUS_OK :
					  engram_open(storage_operations,
						      reader, pack, &engram);
		if (status == STATUS_OK)
			status = engram_sources_prepare();
		if (status == STATUS_OK && engram_hot_set_path != NULL)
			status = engram_hot_start();
		if (status == STATUS_OK)
			status = engram_attach_disk(engram, engram_disk);
		if (status == STATUS_OK && engram_step_io != 0)
			status = engram_steps_start_or_fall_back();
		else if (status == STATUS_OK)
			(void)fprintf(
				stderr,
				"engram step io: off, per-row gathers at each card's visit (%s)\n",
				engram_step_io_default ?
					"not given: the default only serving PP5 and TP4_PRO; "
					"--engram_step_io=1 reads a step's blocks at submit" :
					"--engram_step_io=0");
		failed = step_failed("prepare disk Engram tables", status);
		if (failed && engram_ram_source == ENGRAM_RAM_AUTO)
			engram_fit_disk_refusal_print(
				stderr, engram_ram_percent,
				storage_operations == &storage_direct);
	}
	/* --engram_fit_only: the fit, and disk tables it chose prepared (their sources admitted and scanned) */
	const bool stops_after_fit = !failed && engram_fit_only != 0;
	failed = failed || stops_after_fit;
	if (map_reads == MAP_READS_AUTO && copy_count == 1)
		storage_exl3_read_mode(EXL3_READS_DROPPED);
	if (!failed && is_generating) {
		instance.operations = instance.model != NULL ?
					      instance.model->operations :
					      NULL;
		failed = step_failed("describe the model",
				     instance.operations == NULL ?
					     STATUS_UNSUPPORTED :
					     instance.operations->describe(
						     pack,
						     &model_description)) ||
			 step_failed("match the tokenizer",
				     tokenizer_vocabulary_size(tokenizer) ==
						     model_description
							     .vocabulary_size ?
					     STATUS_OK :
					     STATUS_INVALID_ARGUMENT);
	}
	struct program_limits limits = options_limits();
	if (!failed) {
		const uint64_t bytes_kept_free =
			(uint64_t)card_memory_kept_free_mib * BYTES_PER_MIB;
		const uint32_t unit_count = pack_header_of(pack)->unit_count;
		card_of_unit = calloc(unit_count, sizeof(*card_of_unit));
		status = serve_cards_describe(&backend_cuda_value);
		failed = step_failed("describe the cards", status);
		if (!failed)
			serve_cards_arithmetic();
		/* a card's budget: its free memory less what the backend reserves and less what the user keeps free */
		for (uint32_t card = 0; card < card_count && !failed; card++) {
			const struct card_description_record *const description =
				&descriptions[card];
			const uint64_t unreserved_bytes =
				description->memory_free_bytes >
						description
							->memory_reserved_bytes ?
					description->memory_free_bytes -
						description
							->memory_reserved_bytes :
					0;
			budget_bytes_of_card[card] =
				unreserved_bytes > bytes_kept_free ?
					unreserved_bytes - bytes_kept_free :
					0;
			weight_budget_bytes_of_card[card] =
				budget_bytes_of_card[card];
		}
		/* --session_slots=max: the most sessions the cards hold, by the same passes, plan and budgets as the
		 * placement below; on the TP splits the TP plan refuses a count it does not hold */
		uint32_t fitting = session_slots_max ? 0u : session_slots;
		const struct session_fit_cards fit_cards = {
			.model = instance.operations,
			.pack = pack,
			.layout_text = layout_text,
			.budget_bytes_of_card = budget_bytes_of_card,
			.card_count = card_count,
			.unit_count = unit_count,
			.card_of_unit = card_of_unit,
			.plan_grown = plan_grown
		};
		const struct slot_fit fit = { &fit_cards, bytes_kept_free,
					      card_of_unit };
		/* An explicit count's fitted most, for the report: searched before its TP plan is made (each trial plan
		 * is released). A refusal other than memory is no fit: the placement
		 * below, or the TP plan, makes it again at one session and names its step. */
		result_status fit_refusal = STATUS_OK;
		const uint32_t fitted_most =
			!failed && is_serving && !session_slots_max ?
				slots_most(&fit, &limits, &fit_refusal) :
				0u;
		if (!failed && session_slots_max &&
		    !tensor_placement_selected())
			fitting = session_fit_most(&fit_cards, &limits,
						   SERVING_SLOTS_MOST,
						   &fit_refusal);
		/* an explicit count the cards do not hold: refused with the fit's numbers, before any load */
		if (!failed && is_serving && !session_slots_max &&
		    !tensor_placement_selected() &&
		    session_fit(&fit_cards, &limits) == STATUS_OUT_OF_MEMORY)
			failed = sessions_refused(&fit, &limits,
						  budget_bytes_of_card);
		if (tensor_placement_selected()) {
			const result_status planned =
				failed ? STATUS_OK :
					 tensor_plan_fit(&limits,
							 bytes_kept_free,
							 card_of_unit);
			fitting = limits.session_capacity;
			failed = failed ||
				 (is_serving && !session_slots_max &&
				  planned == STATUS_OUT_OF_MEMORY &&
				  sessions_refused(&fit, &limits,
						   budget_bytes_of_card)) ||
				 (session_slots_max &&
				  planned == STATUS_OUT_OF_MEMORY &&
				  sessions_max_refused(&limits)) ||
				 step_failed("plan TP2 shards and state",
					     planned);
		} else if (!failed && session_slots_max && fitting == 0u &&
			   fit_refusal == STATUS_OK) {
			failed = sessions_max_refused(&limits);
		}
		if (!failed && session_slots_max) {
			limits.session_capacity = fitting != 0u ? fitting : 1u;
			session_slots = limits.session_capacity;
			if (fit_refusal == STATUS_OK)
				sessions_max_report(
					&limits, budget_bytes_of_card,
					fitting == 1u ? single_session_defaults(
								&fit, &limits) :
							NULL);
		}
		if (!failed)
			tp_queue_depth_report();
		if (!failed && is_serving && fit_refusal == STATUS_OK)
			sessions_fit_report(
				session_slots_max ? fitting : fitted_most,
				&limits,
				axis_arguments != NULL ?
					axis_arguments->slots_unmet :
					NULL);
		if (!tensor_placement_selected()) {
			/* Generate: subtract the previous placement's state, then place again. These budgets
			 * are replayed by ds41f_placement_test.c; changes must be made there too. */
			const struct placement_record placement = {
				card_of_unit, unit_count, card_count
			};
			status = failed		      ? status :
				 card_of_unit == NULL ? STATUS_OUT_OF_MEMORY :
							STATUS_OK;
			const uint32_t pass_count =
				is_generating ? PLACEMENT_STATE_PASSES : 1;
			for (uint32_t pass = 0;
			     pass < pass_count && status == STATUS_OK; pass++) {
				status =
					layout_text != NULL ?
						placement_from_layout_text(
							pack, layout_text,
							weight_budget_bytes_of_card,
							card_count, unit_count,
							card_of_unit) :
						placement_fill_in_order(
							pack,
							weight_budget_bytes_of_card,
							card_count, unit_count,
							card_of_unit,
							&on_demand_units_per_owner);
				for (uint32_t card = 0;
				     is_generating && card < card_count &&
				     status == STATUS_OK;
				     card++) {
					uint64_t state_bytes = 0;
					status = instance.operations
							 ->state_byte_count(
								 pack,
								 &placement,
								 &limits, card,
								 &state_bytes);
					weight_budget_bytes_of_card[card] =
						budget_bytes_of_card[card] >
								state_bytes ?
							budget_bytes_of_card
									[card] -
								state_bytes :
							0;
				}
			}
			uint64_t pinned_bytes =
				0; /* the routed experts left in host memory: refused before the load */
			status = !is_generating || status != STATUS_OK ?
					 status :
					 instance.operations->state_byte_count(
						 pack, &placement, &limits,
						 PLACEMENT_IN_HOST_MEMORY,
						 &pinned_bytes);
			if (is_generating && status == STATUS_OK &&
			    pinned_bytes != 0) {
				LOG_ERROR("the placement leaves %" PRIu64
					  " routed-expert bytes in host memory",
					  pinned_bytes);
				status = STATUS_UNSUPPORTED;
			}
			failed =
				failed ||
				step_failed("budget the cards", STATUS_OK) ||
				step_failed(
					speculative >= SPECULATIVE_DEVICE ?
						"place DSpark (layers 37-39, head and draft need one card: "
						"--layout / --card_order)" :
						"place the units",
					status) ||
				step_failed("lay out the extents",
					    load_layout_create(
						    pack, card_of_unit,
						    unit_count, card_count,
						    &layout)) ||
				(is_generating &&
				 step_failed(
					 speculative >= SPECULATIVE_DEVICE ?
						 "plan DSpark (layers 37-39, head and draft need one card: "
						 "--layout / --card_order)" :
						 "plan the state",
					 instance.operations->plan_create(
						 pack, layout, &limits,
						 &plan)));
		}
	}
	/* the arenas grown as the fit grew them (plan_grown), each step named */
	if (!failed && original_head_offload != 0) {
		failed = step_failed("plan original target head offload",
				     ds41f_original_head_plan(plan,
							      descriptions));
	}
	if (!failed && original_head_offload == DS41F_HEAD_MODE_BANKED)
		failed = step_failed(
			"plan original head flight banks",
			ds41f_original_head_banks_plan(plan, pipeline_steps));
	if (!failed && tensor_model != NULL) {
		failed = step_failed(
			"load and qualify TP2 arenas",
			ds41f_tensor_cards_load(
				front_end_files[TABLES_FILE],
				front_end_byte_counts[TABLES_FILE]));
	}
	if (!failed && tensor_model == NULL) {
		struct load_request request = {
			.storage = storage_operations,
			.reader = reader,
			.backend = &backend_cuda_value,
			.pack = pack,
			.layout = layout,
			.external_host_extents =
				engram_disk_external_extents(engram_disk),
			.cards = cards,
			.card_descriptions = descriptions,
			.reader_threads_per_copy = reader_threads_per_copy,
			.piece_bytes = (uint64_t)load_piece_mib * BYTES_PER_MIB,
			.stall_limit_nanoseconds =
				LOADER_DEFAULT_STALL_LIMIT_NANOSECONDS
		};
		if (tuned_load_active && load_tuned_weights != 0)
			memcpy(request.copy_weights, tuned_load_weights,
			       sizeof(request.copy_weights));
		/* Open receiving cards with an arena for their placed weights and model state. */
		for (uint32_t card = 0;
		     card < card_count && status == STATUS_OK; card++) {
			uint64_t arena_bytes = layout->card_byte_count[card];
			status = is_generating ?
					 instance.operations->arena_byte_count(
						 plan, card, &arena_bytes) :
					 STATUS_OK;
			status = status == STATUS_OK &&
						 arena_bytes >
							 budget_bytes_of_card
								 [card] ?
					 STATUS_OUT_OF_MEMORY :
					 status;
			if (status == STATUS_OK && arena_bytes != 0) {
				status = backend_cuda_value.card_open(
					card_order[card], arena_bytes,
					&cards[card]);
			}
		}
		failed = step_failed("open the cards", status) ||
			 step_failed("transfer and check",
				     load_run_entry(&request, &result)) ||
			 (is_generating &&
			  step_failed(
				  "bind the front end",
				  tokenizer_tables_match(
					  pack, layout, &result,
					  front_end_files[TABLES_FILE],
					  front_end_byte_counts[TABLES_FILE])));
	}
	if (!failed) {
		if (tensor_model == NULL) {
			/* a destination's rate is its bytes over the time until its own last piece arrived */
			const double transfer_seconds =
				(double)result.transfer_nanoseconds /
				NANOSECONDS_PER_SECOND;
			const double host_seconds =
				(double)result.host_arrival_nanoseconds /
				NANOSECONDS_PER_SECOND;
			const double host_gib =
				(double)layout->host_byte_count / BYTES_PER_GIB;
			for (uint32_t card = 0; card < card_count; card++) {
				const double card_seconds =
					(double)result
						.card_arrival_nanoseconds[card] /
					NANOSECONDS_PER_SECOND;
				const double card_gib =
					(double)layout->card_byte_count[card] /
					BYTES_PER_GIB;
				(void)printf(
					"card %" PRIu32
					" (%.*s, %.*s): %.2f GiB, %.2f GiB/s\n",
					card_order[card],
					CARD_NAME_CAPACITY - 1,
					descriptions[card].name,
					CARD_BUS_ADDRESS_CAPACITY - 1,
					descriptions[card].bus_address,
					card_gib,
					card_seconds > 0.0 ?
						card_gib / card_seconds :
						0.0);
			}
			for (uint32_t copy_index = 0; copy_index < copy_count;
			     copy_index++) {
				(void)printf(
					"copy %s: %.2f GiB read\n",
					copy_paths[copy_index],
					(double)result.byte_count_read_of_copy
							[copy_index] /
						BYTES_PER_GIB);
			}
			(void)printf(
				"host memory %.2f GiB, %.2f GiB/s, %" PRIu32
				" on-demand units per stage unit on its card, "
				"transfer %.3f s, check %.3f s, %" PRIu32
				" extents read again, %" PRIu32
				" copies dropped, "
				"total %.3f s\n",
				host_gib,
				host_seconds > 0.0 ? host_gib / host_seconds :
						     0.0,
				on_demand_units_per_owner, transfer_seconds,
				(double)result.check_nanoseconds /
					NANOSECONDS_PER_SECOND,
				result.extents_read_again,
				result.copies_dropped,
				step_started_seconds -
					first_step_started_seconds);
		}
		/* a snapshot's bits follow the layers native cards compute: they salt its identity (0: no such card) */
		uint64_t arithmetic_salt =
			tensor_model != NULL ? tensor_model->recipe_identity :
					       0;
		for (uint32_t unit = 0; unit < pack_header_of(pack)->unit_count;
		     unit++) {
			const struct pack_unit_record_2 *const pack_unit =
				&pack_units_of(pack)[unit];
			if (pack_unit->unit_kind == PLACEMENT_UNIT_STAGE &&
			    pack_unit->layer_index != PACK_NO_LAYER &&
			    descriptions[card_of_unit[unit]].arithmetic_class ==
				    ARITHMETIC_CLASS_SM_120_NATIVE) {
				arithmetic_salt = ((arithmetic_salt != 0 ?
							    arithmetic_salt :
							    SALT_OFFSET) ^
						   pack_unit->layer_index) *
						  SALT_PRIME;
			}
		}
		if (cmp_order == CMP_ORDER_RELAXED) {
			arithmetic_salt =
				((arithmetic_salt != 0 ? arithmetic_salt :
							 SALT_OFFSET) ^
				 CMP_SALT_TAG) *
				SALT_PRIME;
			for (uint32_t unit = 0;
			     unit < pack_header_of(pack)->unit_count; unit++) {
				const struct pack_unit_record_2 *const
					pack_unit = &pack_units_of(pack)[unit];
				if (pack_unit->unit_kind ==
					    PLACEMENT_UNIT_STAGE &&
				    pack_unit->layer_index != PACK_NO_LAYER &&
				    descriptions[card_of_unit[unit]]
						    .arithmetic_class ==
					    ARITHMETIC_CLASS_SM_80) {
					arithmetic_salt =
						(arithmetic_salt ^
						 pack_unit->layer_index) *
						SALT_PRIME;
				}
				if (pack_unit->unit_kind ==
					    PLACEMENT_UNIT_ON_DEMAND &&
				    card_of_unit[unit] ==
					    PLACEMENT_IN_HOST_MEMORY) {
					arithmetic_salt =
						(arithmetic_salt ^
						 HOST_EXPERT_SALT_TAG ^ unit) *
						SALT_PRIME;
				}
			}
			/* Relaxed wide and decode forms have different bits; bind the prompt's chosen forms too. */
			const uint32_t chunk_settings[] = {
				wide_rows,	full_steps,
				prefill_cone,	chunks_in_flight,
				prefill_margin, prefill_fast_plan
			};
			for (uint32_t index = 0;
			     index <
			     sizeof(chunk_settings) / sizeof(chunk_settings[0]);
			     index++) {
				arithmetic_salt = (arithmetic_salt ^
						   chunk_settings[index]) *
						  SALT_PRIME;
			}
			/* --prefill_ladder_1024 (plans other than the exact one take other chunks); unset, the
			 * salt is as before */
			if (prefill_prefix_stable != 0)
				arithmetic_salt =
					(arithmetic_salt ^ PREFIX_STABLE_SALT) *
					SALT_PRIME;
			if (prefill_ladder_1024 != 0)
				arithmetic_salt = (arithmetic_salt ^
						   DS41F_CHUNK_LADDER_TOP) *
						  SALT_PRIME;
			if (prefill_ladder_1536 != 0)
				arithmetic_salt = (arithmetic_salt ^
						   DS41F_LADDER_1536_ROWS) *
						  SALT_PRIME;
			if (prefill_ladder_fine != 0)
				arithmetic_salt = (arithmetic_salt ^
						   DS41F_LADDER_FINE_COUNT) *
						  SALT_PRIME;
		}
		struct engine_description_record engine_description = {
			.model_name = pack_header_of(pack)->model_name,
			.front_end_tables = front_end_files[TABLES_FILE],
			.front_end_table_byte_count =
				front_end_byte_counts[TABLES_FILE],
			.session_slot_count = limits.session_capacity,
			.context_token_limit = context_tokens,
			.vocabulary_size = model_description.vocabulary_size,
			.end_token_id = model_description.end_token_id,
			.ignore_eos = ignore_eos,
			.front_steps = prefill_cone,
			.wide_rows = wide_rows,
			.chunks_in_flight = chunks_in_flight,
			.prefill_margin = prefill_margin,
			.prefill_fast_plan = prefill_fast_plan,
			.prefill_prefix_stable = prefill_prefix_stable,
			.prefill_order = prefill_order,
			.prefill_order_wait =
				prefill_order >= PREFILL_ORDER_SHORTEST ?
					prefill_order_wait :
					0,
			.prefill_decode_every = prefill_decode_every,
			.prefill_chunk_rows = prefill_chunk_rows,
			.prefill_decode_rows = prefill_decode_rows,
			.prefill_decode_rounds = prefill_decode_rounds,
			.prefill_decode_waiters = prefill_decode_waiters,
			.prefill_cone_skip = prefill_cone_skip,
			.prefill_tail_rows = prefill_tail_rows,
			.speculative_depth =
				speculative == SPECULATIVE_OFF ? 0 :
				draft_positions < step_rows ? draft_positions :
							      step_rows - 1u,
			.lookup_ngram_most = lookup_ngram_max,
			.lookup_ngram_least = lookup_ngram_min,
			.snapshot_identity = arithmetic_salt,
			.image_math_identity =
				model_images_are_enabled(plan) ?
					descriptions[0].arithmetic_class :
					0,
			.hybrid_lookup_min = speculative == SPECULATIVE_HYBRID ?
						     hybrid_lookup_min :
						     0,
			.draft_trace = draft_trace,
			.draft_policy = &draft_policy
		};
		memcpy(engine_description.batch_step_us, batch_step_us,
		       sizeof(batch_step_us));
		engine_description.prefill_first = prefill_first;
		engine_description.prefill_prompt_chunks =
			prefill_prompt_chunks;
		engine_description.draft_learned_acceptance =
			draft_learned_acceptance;
		engine_description.draft_session_acceptance =
			draft_session_acceptance;
		engine_description.draft_start_economic = draft_start_economic;
		engine_description.batch_plan_ties = batch_plan_ties;
		engine_description.prefix_cold_plan_qualified =
			prefix_cold_plan_is_qualified() ? 1u : 0u;
		failed = is_generating &&
			 (engine_failed_to_start(&engine_description) ||
			  stats_metadata_bound(card_of_unit) != STATUS_OK ||
			  (is_serving	? requests_failed(tokenizer) :
			   is_capturing ? capture_failed(argument_count,
							 arguments) :
					  prompt_failed_to_generate(
						  token_ids, token_count)));
		failed = failed ||
			 (is_generating && !is_serving &&
			  dump_draft_ring != NULL &&
			  ds41f_draft_ring_dump_failed(dump_draft_ring));
		failed = failed || (is_generating && !is_serving &&
				    getenv("DIAG_STATE_DIGESTS") != NULL &&
				    ds41f_state_digests_failed());
		failed = ((fflush(stdout) != 0 || ferror(stdout) != 0) &&
			  step_failed("write the report",
				      STATUS_INPUT_OUTPUT_ERROR)) ||
			 failed;
	}
	engine_destroy(
		engine); /* the engine, then what it stepped, then the cards */
	step_driver_destroy(driver);
	ds41f_original_head_banks_destroy(head_program_banks);
	for (uint32_t index = 0;
	     instance.operations != NULL && index < program_count; index++) {
		instance.operations->program_destroy(programs[index]);
		instance.operations->program_destroy(image_programs[index]);
	}
	engram_close(engram);
	if (engram_steps != NULL)
		engram_steps_report();
	if (engram_steps_live(engram_steps) != 0)
		(void)fprintf(
			stderr,
			"engram step io: %u step records kept at shutdown: a read did not return in its limit, "
			"so their staging stays allocated\n",
			engram_steps_live(engram_steps));
	engram_steps_close(
		engram_steps); /* every driver's step records are gone with the driver, but a kept one */
	engram_disk_close(engram_disk);
	engram_hot_close(engram_hot); /* after the store that served it */
	storage_direct.reader_close(engram_source_reader);
	server_close(server);
	uint32_t head_retained_cards = 0;
	if (original_head_offload != 0 && plan != NULL &&
	    plan->original_head.connected) {
		failed = step_failed("drain original head peer group",
				     ds41f_original_head_drain(
					     &backend_cuda_value, cards,
					     CARD_WAIT_LIMIT_NANOSECONDS,
					     &head_retained_cards)) ||
			 failed;
	}
	if (instance.operations != NULL && tensor_model == NULL) {
		instance.operations->plan_destroy(plan);
	}
	const uint32_t retained_cards = tensor_model != NULL ?
						tensor_cards_drain() :
						head_retained_cards;
	if (retained_cards != 0) {
		(void)fprintf(
			stderr,
			"%s peer arenas retained after an unproven group drain: mask=0x%" PRIx32
			"\n",
			tensor_model != NULL ? "TP2" : "original head",
			retained_cards);
		failed = true;
	}
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
		if ((retained_cards >> card & 1u) == 0)
			backend_cuda_value.card_close(
				cards[card], CARD_WAIT_LIMIT_NANOSECONDS);
	}
	if (retained_cards == 0)
		ds41f_tp2_model_destroy(tensor_model);
	if (layout !=
	    NULL) { /* behind the cards; no arena was mapped without a layout that sizes it */
		load_host_arena_release(result.host_arena,
					layout->host_byte_count);
		load_layout_destroy(layout);
	}
	free(card_of_unit);
	pack_close(pack);
	storage_operations->reader_close(reader);
	tokenizer_close(tokenizer);
	for (uint32_t file = 0; file < FRONT_END_FILES; file++) {
		free(front_end_files[file]);
	}
	free(prompt);
	free(token_ids);
#if TIMING_HOOKS_ENABLED
	hooks_report(stderr);
#endif
	return failed && !stops_after_fit ? PROGRAM_FAILED : PROGRAM_SUCCEEDED;
}
