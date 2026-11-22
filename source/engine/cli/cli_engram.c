/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_engram.c: Engram's settings, RAM fit, sources and step start (split from commands.c by whole items).
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

#define NANOSECONDS_PER_MICROSECOND 1000ull
#define NANOSECONDS_PER_MILLISECOND 1000000ull
#define AGE_QUANTILE_PARTS 1000u
#define AGE_QUANTILE_PART \
	999u /* step I/O's shutdown line: the 99.9 % read age */
static char engram_copy_paths[STORAGE_MAXIMUM_COPIES][REPLICA_PATH_CAPACITY];
static uint32_t engram_copy_count;

/* --engram_ram_percent=auto: the default, given once */
bool take_engram_ram_auto(const char *argument, bool *given)
{
	if (*given || strcmp(argument, ENGRAM_RAM_AUTO_OPTION) != 0)
		return false;
	*given = true;
	return true;
}

/* The static hot set: checked against its recorded SHA-256, the pack and the store, its blocks read through the
 * store's verified path, then served like the RAM prefix. */
MUST_CHECK result_status engram_hot_start(void)
{
	char reason[ENGRAM_HOT_REASON_BYTES] = "";
	struct timespec started, ended;
	(void)clock_gettime(CLOCK_MONOTONIC, &started);
	const result_status status =
		engram_hot_open(engram_hot_set_path, engram_hot_sha256,
				(uint64_t)engram_hot_mib * BYTES_PER_MIB, pack,
				engram_disk, &engram_hot, reason);
	(void)clock_gettime(CLOCK_MONOTONIC, &ended);
	if (status != STATUS_OK) {
		(void)fprintf(stderr, "engram hot set refused: %s\n", reason);
		return status;
	}
	engram_disk_attach_hot(engram_disk, engram_hot);
	(void)fprintf(stderr,
		      "engram hot set: %" PRIu64 " blocks (%" PRIu64
		      " MiB) from %s, read and verified in %.3f s\n",
		      engram_hot_block_count(engram_hot),
		      (uint64_t)(engram_hot_block_count(engram_hot) *
				 PACK_BLOCK_BYTES / BYTES_PER_MIB),
		      engram_hot_set_path,
		      (double)(ended.tv_sec - started.tv_sec) +
			      (double)(ended.tv_nsec - started.tv_nsec) /
				      NANOSECONDS_PER_SECOND);
	return STATUS_OK;
}

/* Step I/O's default and its guards, once the storage profile was applied: a profile's turbo_engram puts the tables
 * on disk (it sets engram_ram_percent), and --engram_step_io and the hot set are no profile argument, so they keep it.
 * Serving PP5 or TP4_PRO from the disk store reads a step's blocks at submit by default, every stream identical to
 * all-RAM. PP5 (8-9 Nov 2026): cold prefill 27.38 against 27.18 s all-RAM at 131k tokens and 1.75x today's per-row
 * path; decode C1 172.2 against 176.0 tok/s, C4 and C8 equal; 0.8 cores, hybrid and pinned. TP4_PRO (9 Nov 2026, two
 * runs): decode C1 4.6-5.9 % over the per-row path; cold prefill 9.87 / 35.18 s at 8180 / 32782 tokens against
 * 9.74 / 35.08 s all-RAM and 12.63 / 37.98 s per-row. --engram_step_io=0 keeps the per-row path; other GPU splits
 * keep it until measured there. Returns false, with its reason printed, for step I/O or a hot set with every table
 * in RAM. */
static void engram_step_io_default_apply(void)
{
	if (engram_step_io_default && is_serving &&
	    engram_ram_percent < ENGRAM_RAM_PERCENT &&
	    (execution_placement == GPU_SPLIT_PP5 ||
	     execution_placement == GPU_SPLIT_TP4_PRO))
		engram_step_io = 1;
}

/* At startup, before auto's fit (which applies the step I/O default again once it chose disk tables). */
MUST_CHECK bool engram_settings_resolve(void)
{
	engram_step_io_default_apply();
	if (engram_ram_percent == ENGRAM_RAM_PERCENT &&
	    (engram_step_io != 0 || engram_hot_set_path != NULL)) {
		(void)fputs(
			"engram: --engram_step_io=1 and --engram_hot_set need disk tables (--engram_ram_percent below "
			"100, or a storage profile that puts them on disk)\n",
			stderr);
		return false;
	}
	return true;
}

/* The Engram fit (engram_fit.h), once, before the load and before any card: auto's decision, and the fit printed for a
 * given percent too. The n-gram object opened here is the one the load attaches; a pack without n-gram tables has
 * nothing to fit. STATUS_OUT_OF_MEMORY when the model's own host memory does not fit (the lines name it). */
MUST_CHECK result_status engram_ram_fit(void)
{
	result_status status =
		engram != NULL ?
			STATUS_OK :
			engram_open(storage_operations, reader, pack, &engram);
	if (status == STATUS_UNSUPPORTED)
		return STATUS_OK;
	struct engram_fit_needs needs = { 0 };
	status = status == STATUS_OK ?
			 engram_fit_needs_of_pack(
				 pack, engram_description_of(engram),
				 tensor_placement_selected(), card_count,
				 (uint64_t)load_piece_mib * BYTES_PER_MIB,
				 tp_load_threads, &needs) :
			 status;
	if (status != STATUS_OK)
		return status;
	needs.cache_bytes = (uint64_t)engram_cache_mib * BYTES_PER_MIB;
	needs.hot_bytes = engram_hot_set_path != NULL ?
				  (uint64_t)engram_hot_mib * BYTES_PER_MIB :
				  0;
	engram_fit_memory_read(&needs);
	struct engram_fit_decision decision;
	engram_fit_decide(&needs, engram_ram_source, engram_ram_percent,
			  &decision);
	engram_fit_print(stderr, &needs, &decision, engram_ram_source);
	if (decision.fit_case == ENGRAM_FIT_MODEL_OVER)
		return STATUS_OUT_OF_MEMORY;
	engram_ram_percent = decision.percent;
	engram_step_io_default_apply();
	return STATUS_OK;
}

/* Where a step I/O parameter came from: its knob, the profile's engram section (equal to the default, or not), or the
 * default. */
static const char *engram_source_tag(bool from_profile, bool explicit_knob,
				     bool equals_default)
{
	return from_profile ?
		       (equals_default ? "profile = default" : "profile") :
	       explicit_knob ? "knob" :
			       "default";
}

/* Step I/O's threads a copy, spin and pin: each from its knob when given, else from the profile's engram section when
 * it is enabled and names this pack, exactly the disk store's copies (by their device keys, in order) and this
 * flight's shape, else the default (storage_engram_section_apply). The tables are on disk here: step I/O starts only
 * then. reason gets why the section is not applied, empty when it is. Never the RAM share: that is the RAM fit's. */
static void engram_section_apply(char reason[STORAGE_PROFILE_REASON_BYTES],
				 struct storage_engram_choice *choice)
{
	/* the admitted copies' device keys, read only with a profile (a copy without one matches none) */
	char keys[STORAGE_MAXIMUM_COPIES][STORAGE_PROFILE_KEY_BYTES] = { { 0 } };
	for (uint32_t copy = 0;
	     storage_profile_resolved[0] != '\0' && copy < engram_copy_count;
	     copy++) {
		struct local_device_description device;
		if (storage_device_key_for(engram_copy_paths[copy],
					   replica_external != 0, keys[copy],
					   &device) != STATUS_OK)
			keys[copy][0] = '\0';
	}
	const struct storage_engram_serving serving = {
		.profile = storage_profile_resolved,
		.header = pack_header_of(pack),
		.copy_keys = (const char (*)[STORAGE_PROFILE_KEY_BYTES])keys,
		.copy_count = engram_copy_count,
		.steps = 1u + STEP_PIPELINE_MOST,
		.slots = ENGRAM_STEP_SLOTS,
		.ram_percent = engram_ram_percent,
		.explicit_threads = explicit_engram_threads,
		.explicit_spin = explicit_engram_spin,
		.explicit_pin = explicit_engram_pin
	};
	*choice = (struct storage_engram_choice){ .threads_per_copy =
							  engram_io_threads,
						  .spin_us = engram_spin_us,
						  .pin = engram_io_pin };
	storage_engram_section_apply(&serving, choice, reason);
	engram_io_threads = choice->threads_per_copy;
	engram_spin_us = choice->spin_us;
	engram_io_pin = choice->pin;
}

/* Step I/O over the disk store: one record for a driver without a pipeline and one per pipeline slot, each sized for
 * the largest step any program takes. A read's deadline fits its re-reads on every other copy into half the cards'
 * wait limit (1.25 s for three copies): a pipeline that sees no progress for that limit fails the step, so a stalled
 * read fails over to another copy before it does. */
MUST_CHECK result_status engram_steps_start(void)
{
	if (engram_disk == NULL)
		return STATUS_INVALID_ARGUMENT;
	char reason[STORAGE_PROFILE_REASON_BYTES];
	struct storage_engram_choice choice;
	engram_section_apply(reason, &choice);
	struct engram_steps_options options = {
		.threads_per_copy = engram_io_threads,
		.steps = 1u + STEP_PIPELINE_MOST,
		.slots_per_step = ENGRAM_STEP_SLOTS,
		.rows_most = effective_row_capacity(),
		.spin_nanoseconds =
			(uint64_t)engram_spin_us * NANOSECONDS_PER_MICROSECOND,
		.read_limit_nanoseconds = engram_steps_read_limit_within(
			CARD_WAIT_LIMIT_NANOSECONDS,
			engram_disk_copy_count(engram_disk))
	};
	const struct storage_reader *const engram_reader =
		engram_disk_reader(engram_disk);
	for (uint32_t copy = 0; copy < STORAGE_MAXIMUM_COPIES; copy++)
		for (uint32_t thread = 0;
		     thread < STORAGE_FLIGHT_THREADS_PER_COPY_MOST; thread++)
			options.cpus[copy][thread] =
				engram_io_pin != 0 &&
						copy < engram_disk_copy_count(
							       engram_disk) ?
					storage_flight_queue_cpu(engram_reader,
								 copy, thread) :
					STORAGE_FLIGHT_CPU_FREE;
	result_status status =
		engram_steps_open(engram_disk, &options, &engram_steps);
	if (status == STATUS_OK)
		status = engram_attach_steps(engram, engram_steps);
	if (status == STATUS_OK) {
		(void)fprintf(
			stderr,
			"engram step io: %u copies, %u threads a copy (%s), spin %u us (%s), pin %u (%s), %u steps "
			"of %u slots, cpus",
			engram_disk_copy_count(engram_disk),
			options.threads_per_copy,
			engram_source_tag(
				choice.threads_from_profile,
				explicit_engram_threads,
				engram_io_threads ==
					ENGRAM_STEP_IO_THREADS_DEFAULT),
			engram_spin_us,
			engram_source_tag(
				choice.spin_from_profile, explicit_engram_spin,
				engram_spin_us == ENGRAM_STEP_SPIN_US_DEFAULT),
			engram_io_pin,
			engram_source_tag(choice.pin_from_profile,
					  explicit_engram_pin,
					  engram_io_pin == 1u),
			options.steps, options.slots_per_step);
		for (uint32_t copy = 0;
		     copy < engram_disk_copy_count(engram_disk); copy++)
			for (uint32_t thread = 0;
			     thread < options.threads_per_copy; thread++)
				(void)fprintf(stderr, " %d",
					      options.cpus[copy][thread]);
		const uint32_t plain = storage_flight_plain_rings(
			engram_steps_flight(engram_steps));
		(void)fprintf(stderr, ", read limit %" PRIu64 " ms",
			      (uint64_t)(options.read_limit_nanoseconds /
					 NANOSECONDS_PER_MILLISECOND));
		if (plain != 0)
			(void)fprintf(
				stderr,
				", %u rings read without registered buffers (the locked-memory limit)",
				plain);
		(void)fprintf(
			stderr, " (%s)%s%s\n",
			engram_step_io_default ?
				"the default for disk Engram on PP5 and TP4_PRO; "
				"--engram_step_io=0: per-row gathers" :
				"--engram_step_io=1",
			reason[0] != '\0' ? "; engram section not applied: " :
					    "; engram section applied",
			reason);
	}
	return status;
}

/* Step I/O's counts at shutdown, which a gate can hold to zero retirements: the reads the drives answered, their ages
 * from the hand-off to the ring (the most, and the bucket bound under which 99.9 % fell) against the read deadline,
 * the rings retired past that deadline or for any other reason, the rings set up again, and the probations. */
void engram_steps_report(void)
{
	struct storage_flight_counts counts;
	storage_flight_counts(engram_steps_flight(engram_steps), &counts);
	(void)fprintf(
		stderr,
		"engram step io: shutdown: %" PRIu64
		" reads answered, oldest %" PRIu64
		" us, 99.9 %% within %" PRIu64 " us, deadline %" PRIu64
		" us (%" PRIu64
		"x the oldest); retirements %u past the deadline, %u other; rings set up again %u; probations %u\n",
		counts.answered,
		(uint64_t)(counts.age_most_ns / NANOSECONDS_PER_MICROSECOND),
		(uint64_t)(storage_flight_age_quantile(&counts,
						       AGE_QUANTILE_PARTS,
						       AGE_QUANTILE_PART) /
			   NANOSECONDS_PER_MICROSECOND),
		(uint64_t)(counts.read_limit_ns / NANOSECONDS_PER_MICROSECOND),
		counts.age_most_ns != 0 ?
			counts.read_limit_ns / counts.age_most_ns :
			(uint64_t)0u,
		counts.deadline_retirements, counts.other_retirements,
		counts.reinstatements, engram_steps_probations(engram_steps));
}

/* Step I/O started; or, when it is the default rather than asked for, the host refused it (io_uring disabled or
 * filtered, a ring the kernel will not set up): the serve keeps the per-row gathers it had before, and says why. */
MUST_CHECK result_status engram_steps_start_or_fall_back(void)
{
	const result_status status = engram_steps_start();
	if (status == STATUS_OK || !engram_step_io_default)
		return status;
	engram_steps_close(
		engram_steps); /* open or attach failed: nothing is attached */
	engram_steps = NULL;
	engram_step_io = 0;
	(void)fprintf(
		stderr,
		"engram step io: off, per-row gathers at each card's visit (the default could not start: %s)\n",
		status_text(status));
	return STATUS_OK;
}

/* Runtime table sources are independent of the model loader's copies and may use different pack layouts. */
MUST_CHECK result_status engram_sources_prepare(void)
{
	const char *paths[STORAGE_MAXIMUM_COPIES] = { 0 };
	uint32_t count = copy_count;
	char *text = NULL;
	result_status status = STATUS_OK;
	if (engram_sources_text != NULL) {
		const size_t length = strnlen(engram_sources_text,
					      (size_t)STORAGE_MAXIMUM_COPIES *
						      REPLICA_PATH_CAPACITY);
		if (length == 0 || length == (size_t)STORAGE_MAXIMUM_COPIES *
						     REPLICA_PATH_CAPACITY)
			return STATUS_INVALID_ARGUMENT;
		text = strdup(engram_sources_text);
		if (text == NULL)
			return STATUS_OUT_OF_MEMORY;
		count = 0;
		char *next = text;
		while (next != NULL && status == STATUS_OK) {
			char *comma = strchr(next, ',');
			if (comma != NULL)
				*comma = '\0';
			if (*next == '\0' || count == STORAGE_MAXIMUM_COPIES ||
			    strlen(next) >= REPLICA_PATH_CAPACITY)
				status = STATUS_INVALID_ARGUMENT;
			else
				paths[count++] = next;
			next = comma != NULL ? comma + 1 : NULL;
		}
	} else {
		if (storage_operations != &storage_direct)
			return STATUS_UNSUPPORTED;
		for (uint32_t copy = 0; copy < count; copy++)
			paths[copy] = copy_paths[copy];
	}
	struct local_device_description devices[STORAGE_MAXIMUM_COPIES];
	uint32_t accepted = 0;
	for (uint32_t copy = 0; status == STATUS_OK && copy < count; copy++) {
		struct local_device_description current = { 0 };
		const result_status described =
			local_device_describe(paths[copy], &current);
		const char *reason =
			described != STATUS_OK ?
				"device_disappeared_or_unknown" :
				local_device_media_refusal(
					&current, replica_external != 0);
		for (uint32_t previous = 0;
		     reason == NULL && previous < accepted; previous++)
			if (local_device_overlaps(&current, &devices[previous]))
				reason = "duplicate_physical_device";
		local_device_print(stderr, &current,
				   reason != NULL ? reason : "eligible", 0,
				   replica_floor_percent,
				   tuned_engram_active ?
					   tuned_storage.load_weights[copy] :
					   0,
				   tuned_engram_active ?
					   tuned_storage.engram_weights[copy] :
					   0);
		if (!tuned_engram_active) {
			const result_status noted = storage_profile_note(
				&current, reason != NULL ? reason : "eligible",
				STORAGE_REPLICA_LOOKUP);
			if (noted != STATUS_OK)
				status = noted;
		}
		if (reason != NULL)
			continue;
		paths[accepted] = paths[copy];
		devices[accepted++] = current;
	}
	if (status == STATUS_OK &&
	    (accepted == 0 || (tuned_engram_active && accepted != count)))
		status = STATUS_UNSUPPORTED;
	count = accepted;
	for (uint32_t copy = 0; copy < count; copy++)
		(void)snprintf(engram_copy_paths[copy],
			       sizeof(engram_copy_paths[copy]), "%s",
			       paths[copy]);
	engram_copy_count = count;
	if (count == 1u)
		turbo_nvme_engrams = 0;
	if (status == STATUS_OK)
		status = engram_disk_sources_open(paths, count, pack, engram,
						  &engram_source_reader);
	engram_disk_scan_threads(engram_scan_threads);
	if (status == STATUS_OK)
		status = engram_disk_open(
			engram_description_of(engram), pack, &storage_direct,
			engram_source_reader != NULL ? engram_source_reader :
						       reader,
			engram_ram_percent,
			(uint64_t)engram_cache_mib * BYTES_PER_MIB,
			turbo_nvme_engrams, engram_io, &engram_disk);
	if (status == STATUS_OK && tuned_engram_active)
		status = engram_disk_weights(engram_disk,
					     tuned_storage.engram_weights,
					     tuned_storage.count);
	free(text);
	return status;
}
