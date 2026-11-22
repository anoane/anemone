/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_load.c: the load settings: the storage profile and TP load sections, replicas and turbo-load copies (split from
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

static bool explicit_tp_reads_per_copy, explicit_load_piece;
static const char *tuned_paths[STORAGE_MAXIMUM_COPIES];
static const char *tuned_load_paths[STORAGE_MAXIMUM_COPIES];
static char tuned_engram_sources[STORAGE_MAXIMUM_COPIES * REPLICA_PATH_CAPACITY];
/* turbo load: the copies once the receipted replicas beside the given copies are added (replicas.h) */
static char replica_paths[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
static const char *turbo_copy_paths[STORAGE_MAXIMUM_COPIES];

/* The TP host arena's read: readers a copy and the piece, each from its knob when given, else from the profile's load
 * section when it is enabled, names this pack and exactly these copies (by their device keys) and was measured with
 * this load's host threads and split (and, for an EXL3 map, an explicit --map_reads equal to the measured mode),
 * else the defaults; one line says which, with the section's reason when it is
 * not applied. The section's readers cap the host arena's reads alone (the stage measured them alone): the choice
 * gives them as its host cap, which goes to tp_load_host_reads_per_copy; --tp_load_reads_per_copy caps the whole
 * load's. */
void load_section_apply(void)
{
	struct storage_load_section section;
	char reason[STORAGE_PROFILE_REASON_BYTES] = "profile_off";
	bool usable = storage_profile_resolved[0] != '\0' &&
		      storage_load_section_read(storage_profile_resolved,
						&section, reason) &&
		      storage_load_section_matches(
			      &section, pack_header_of(pack), copy_paths,
			      copy_count, replica_external != 0,
			      tp_load_host_threads != 0 ? tp_load_host_threads :
							  tp_load_threads,
			      tp_load_copy_split, reason) &&
		      (storage_operations != &storage_exl3 ||
		       storage_load_section_reads_match(copy_count, map_reads,
							reason));
	if (usable && section.enabled == 0) {
		(void)snprintf(reason, sizeof(reason), "%s",
			       "load_section_candidate");
		usable = false;
	}
	struct storage_load_choice choice = {
		.tp_piece_bytes = (uint64_t)load_piece_mib * BYTES_PER_MIB,
		.tp_readers_per_copy = tp_load_reads_per_copy,
		.tp_host_readers_per_copy = tp_load_host_reads_per_copy
	};
	storage_load_choose(usable ? &section : NULL,
			    explicit_tp_reads_per_copy, explicit_load_piece,
			    &choice);
	tp_load_host_reads_per_copy = choice.tp_host_readers_per_copy;
	const uint32_t readers = choice.readers_from_profile ?
					 choice.tp_host_readers_per_copy :
					 choice.tp_readers_per_copy;
	tp_load_piece_bytes = choice.tp_piece_bytes;
	(void)printf("TP load read: %" PRIu32
		     " readers a copy (%s), piece %" PRIu64 " bytes (%s)%s%s\n",
		     readers,
		     !choice.readers_from_profile ?
			     (explicit_tp_reads_per_copy ? "knob" : "default") :
		     readers == DEFAULT_TP_LOAD_READS_PER_COPY ?
			     "profile = default" :
			     "profile",
		     tp_load_piece_bytes != 0 ?
			     tp_load_piece_bytes :
			     (uint64_t)TP_HOST_LOAD_CHUNK_BYTES,
		     !choice.piece_from_profile ?
			     (explicit_load_piece ? "knob" : "default") :
		     tp_load_piece_bytes == TP_HOST_LOAD_CHUNK_BYTES ?
			     "profile = default" :
			     "profile",
		     usable ? "" : "; load section not applied: ",
		     usable ? "" : reason);
}

/* Turbo load: the replicas found beside the given copies (found_paths, replicas_add's search) that open together
 * with the copies already kept follow the given copies in a new reader (replicas_reader_open: one line names each
 * replica left out and why); when none opens, the given copies alone stay (one line says so). */
MUST_CHECK result_status replicas_found_add(
	const char (*found_paths)[REPLICA_PATH_CAPACITY], uint32_t found)
{
	const uint32_t given = copy_count;
	struct storage_reader *wider = NULL;
	const uint32_t kept = replicas_reader_open(
		storage_operations, copy_paths, given, found_paths, found,
		turbo_copy_paths, &wider, stdout);
	if (found != 0 && kept == 0) {
		(void)printf("turbo load: %" PRIu32
			     " replicas do not open: %s\n",
			     found,
			     given == 1u ? "the source alone" :
					   "the given copies alone");
		return STATUS_OK;
	}
	struct storage_description opened = { 0 };
	if (kept != 0 && model_pack != NULL &&
	    (storage_operations->reader_describe(wider, &opened) != STATUS_OK ||
	     !model_file_is(model_pack, opened.first_device,
			    opened.first_inode))) {
		storage_operations->reader_close(wider);
		(void)fprintf(
			stderr,
			"model_dir: %s changed between its check and its open\n",
			model_pack->path);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	if (kept != 0) {
		storage_operations->reader_close(reader);
		reader = wider;
		copy_paths = turbo_copy_paths;
		copy_count = given + kept;
	}
	for (uint32_t index = given; index < given + kept; index++)
		(void)printf("turbo load: copy %" PRIu32 " %s\n", index,
			     turbo_copy_paths[index]);
	return STATUS_OK;
}

bool turbo_load_searches(void)
{
	return !tuned_load_active && turbo_load != 0 &&
	       copy_count < STORAGE_MAXIMUM_COPIES;
}

uint32_t (*turbo_load_search)(const char *const *copies, uint32_t copy_count,
			      const struct pack_header *header,
			      char (*paths)[REPLICA_PATH_CAPACITY],
			      uint32_t capacity,
			      bool external) = replicas_find_policy;

/* Turbo load's search: the receipted replicas of the given copies' source on drives that hold none of them
 * (replicas.h), then replicas_found_add. */
MUST_CHECK static result_status replicas_add(const struct pack_header *header)
{
	const uint32_t found = turbo_load_search(
		copy_paths, copy_count, header, replica_paths,
		STORAGE_MAXIMUM_COPIES - copy_count, replica_external != 0);
	return replicas_found_add(
		(const char (*)[REPLICA_PATH_CAPACITY])replica_paths, found);
}

/* A load given more than one copy names every copy it reads and where each came from: the command line, or turbo
 * load's search beside it (with --turbo_load=0 none comes from the search). */
static void copies_print(uint32_t given)
{
	(void)printf("turbo load: the load reads %" PRIu32 " copies:",
		     copy_count);
	for (uint32_t copy = 0; copy < copy_count; copy++)
		(void)printf(" %" PRIu32 " %s (%s)%s", copy, copy_paths[copy],
			     copy < given ? "given" : "search",
			     copy + 1u == copy_count ? "\n" : ",");
}

bool turbo_load_copies(const struct pack_header *header)
{
	const uint32_t given = copy_count;
	if (turbo_load_searches() &&
	    step_failed("find the replicas", replicas_add(header)))
		return false;
	/* a tuned load's copies are the profile's, named by its storage load: line (a profile applies to one given
	   copy alone) */
	if (given > 1u && !tuned_load_active)
		copies_print(given);
	return true;
}

/* Profile preparation runs once before model loading; it never tunes or changes the machine. */
void storage_profile_apply(int argument_count, char **arguments)
{
	storage_profile_is_valid = false;
	for (int index = ARGUMENTS_BEFORE_THE_OPTIONS; index < argument_count;
	     index++) {
		explicit_tp_reads_per_copy =
			explicit_tp_reads_per_copy ||
			strncmp(arguments[index], "--tp_load_reads_per_copy=",
				strlen("--tp_load_reads_per_copy=")) == 0;
		explicit_load_piece = explicit_load_piece ||
				      strncmp(arguments[index],
					      "--load_piece_mib=",
					      strlen("--load_piece_mib=")) == 0;
		explicit_engram_threads =
			explicit_engram_threads ||
			strncmp(arguments[index], "--engram_io_threads=",
				strlen("--engram_io_threads=")) == 0;
		explicit_engram_spin =
			explicit_engram_spin ||
			strncmp(arguments[index], "--engram_spin_us=",
				strlen("--engram_spin_us=")) == 0;
		explicit_engram_pin = explicit_engram_pin ||
				      strncmp(arguments[index],
					      "--engram_io_pin=",
					      strlen("--engram_io_pin=")) == 0;
	}
	char default_path[REPLICA_PATH_CAPACITY];
	const char *path = storage_profile_path != NULL ?
				   storage_profile_path :
				   getenv("ANEMONE_STORAGE_PROFILE");
	storage_profile_is_off = path != NULL && strcmp(path, "off") == 0;
	if (storage_profile_is_off)
		return;
	if (path == NULL) {
		const char *config = getenv("XDG_CONFIG_HOME");
		const char *home = getenv("HOME");
		const int length =
			config != NULL ?
				snprintf(default_path, sizeof(default_path),
					 "%s/anemone/storage-profile.json",
					 config) :
			home != NULL ?
				snprintf(
					default_path, sizeof(default_path),
					"%s/.config/anemone/storage-profile.json",
					home) :
				-1;
		if (length < 0 || (size_t)length >= sizeof(default_path)) {
			(void)fprintf(
				stderr,
				"storage profile: fallback no_config_directory\n");
			return;
		}
		path = default_path;
	}
	(void)snprintf(storage_profile_resolved,
		       sizeof(storage_profile_resolved), "%s", path);
	for (int index = ARGUMENTS_BEFORE_THE_OPTIONS; index < argument_count;
	     index++)
		if (strncmp(arguments[index], "--replica_external=",
			    strlen("--replica_external=")) == 0 ||
		    strncmp(arguments[index], "--replica_floor_percent=",
			    strlen("--replica_floor_percent=")) == 0) {
			(void)fputs(
				"storage profile: fallback explicit_replica_policy\n",
				stderr);
			return;
		}
	char reason[STORAGE_PROFILE_REASON_BYTES];
	if (!storage_profile_read(path, &tuned_storage, reason)) {
		(void)fprintf(stderr, "storage profile: fallback %s\n", reason);
		return;
	}
	char real_source[REPLICA_PATH_CAPACITY];
	const bool resolved = copy_count == 1 &&
			      realpath(copy_paths[0], real_source) != NULL;
	if (copy_count != 1 ||
	    (resolved ? strcmp(real_source, tuned_storage.source) != 0 :
			!tuned_storage.source_unavailable ||
				strcmp(copy_paths[0], tuned_storage.source) !=
					0)) {
		(void)fprintf(stderr,
			      "storage profile: fallback source_mismatch\n");
		return;
	}
	struct pack_header source_header = { 0 };
	uint64_t source_bytes = 0;
	char pack_identity[STORAGE_PROFILE_KEY_BYTES];
	/* an EXL3 map's logical header, through the map reader */
	const bool source_matches = storage_pack_header_read(
		tuned_storage.source_unavailable ? tuned_storage.paths[0] :
						   real_source,
		&source_header, &source_bytes);
	if (source_matches)
		(void)snprintf(pack_identity, sizeof(pack_identity),
			       "%016" PRIx64, source_header.header_digest);
	if (!source_matches ||
	    source_header.file_byte_count != tuned_storage.source_bytes ||
	    strcmp(pack_identity, tuned_storage.pack_identity) != 0) {
		(void)fprintf(
			stderr,
			"storage profile: fallback source_identity_mismatch\n");
		return;
	}
	replica_external = tuned_storage.replica_external;
	replica_floor_percent = tuned_storage.replica_floor_percent;
	bool explicit_load = false, explicit_engram = false;
	for (int index = ARGUMENTS_BEFORE_THE_OPTIONS; index < argument_count;
	     index++) {
		const char *text = arguments[index];
		explicit_load =
			explicit_load ||
			strncmp(text,
				"--turbo_load=", strlen("--turbo_load=")) ==
				0 ||
			strncmp(text, "--reader_threads_per_copy=",
				strlen("--reader_threads_per_copy=")) == 0 ||
			strncmp(text,
				"--"
				"tp_load_",
				strlen("--"
				       "tp_load_")) == 0 ||
			strncmp(text, "--map_reads=", strlen("--map_reads=")) ==
				0;
		explicit_engram = explicit_engram ||
				  storage_profile_engram_argument(text);
	}
	tuned_load_active = tuned_storage.load_enabled != 0 && !explicit_load;
	tuned_engram_active = tuned_storage.engram_enabled != 0 &&
			      !explicit_engram && is_generating;
	for (uint32_t index = 0; index < tuned_storage.count; index++)
		tuned_paths[index] = tuned_storage.paths[index];
	if (tuned_storage.source_unavailable && !tuned_load_active) {
		copy_paths = tuned_paths;
		copy_count = 1;
		(void)fputs(
			"storage: source unavailable; using one verified replica with default load scheduling\n",
			stderr);
	}
	if (tuned_load_active) {
		copy_count = storage_profile_load_list(
			&tuned_storage, tuned_load_paths, tuned_load_weights);
		copy_paths = tuned_load_paths;
		reader_threads_per_copy = tuned_storage.load_threads;
		tp_load_copy_split = TP_LOAD_SPLIT_SOONEST;
		(void)fprintf(
			stderr,
			"storage load: %u copies, %u readers a copy, %s\n",
			copy_count, reader_threads_per_copy,
			load_tuned_weights != 0 ?
				"each piece's copy by the profile's weights" :
				"the readers steal pieces (profile weights not applied)");
	}
	if (tuned_engram_active) {
		uint32_t sources = 0;
		if (!storage_profile_engram_sources(
			    &tuned_storage, tuned_engram_sources,
			    sizeof(tuned_engram_sources), &sources)) {
			tuned_engram_active = false;
			(void)fprintf(
				stderr,
				"storage profile: fallback source_list_overflow\n");
		} else if (sources == 0) {
			/* only an EXL3 map was verified: no pack holds the tables for the disk store */
			tuned_engram_active = false;
			(void)fprintf(
				stderr,
				"storage profile: fallback engram_sources_none (an EXL3 map serves no disk lookup)\n");
		}
		if (tuned_engram_active) {
			engram_sources_text = tuned_engram_sources;
			engram_ram_percent = tuned_storage.ram_percent;
			engram_ram_source = ENGRAM_RAM_PROFILE;
			engram_cache_mib = tuned_storage.cache_mib;
			engram_io = tuned_storage.engram_io;
			turbo_nvme_engrams = sources > 1 ? 1u : 0u;
		}
	}
	(void)fprintf(stderr,
		      "storage profile: %s age_seconds=%" PRIu64
		      " load=%u engram=%u replicas=%u excluded=%u %s\n",
		      tuned_storage.identity,
		      (uint64_t)time(NULL) - tuned_storage.created_unix,
		      tuned_load_active ? 1u : 0u,
		      tuned_engram_active ? 1u : 0u, tuned_storage.count,
		      tuned_storage.excluded, reason);
	storage_profile_is_valid = true;
}

MUST_CHECK result_status storage_metrics_prepare(void)
{
	FILE *const stream = fmemopen(storage_replica_info,
				      sizeof(storage_replica_info), "w");
	if (stream == NULL)
		return STATUS_OUT_OF_MEMORY;
	const result_status status = storage_profile_metrics(
		stream, storage_profile_decisions(), true);
	const int flushed = fflush(stream);
	const long bytes = ftell(stream);
	const int closed = fclose(stream);
	if (status == STATUS_OK && flushed == 0 && closed == 0 && bytes > 0 &&
	    (uint64_t)bytes < sizeof(storage_replica_info))
		storage_replica_info_bytes = (size_t)bytes;
	else
		return STATUS_INPUT_OUTPUT_ERROR;
	return STATUS_OK;
}
