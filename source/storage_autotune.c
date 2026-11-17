/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "storage_autotune.h"
#include "anemone/hooks.h"
#include "anemone/storage_exl3.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define NS_SECOND 1000000000ull
#define NS_MICROSECOND 1000ull
#define DECIMAL_BASE 10u
#define TEMPERATURE_TEXT_BYTES 32u
#define NVME_CLASS_DIRECTORY "/sys/class/nvme"
#define NVME_NAME_PREFIX "nvme"
#define HWMON_NAME_PREFIX "hwmon"
#define TEMPERATURE_FILE \
	"temp1_input" /* the composite temperature, in millidegrees Celsius */
#define AUTOTUNE_TOKENS_MOST_BYTES \
	(1u << 20u) /* every --tokens file together */
#define PROFILE_PATH_SUFFIX ".replicas.json"
#define COPY_WORKERS_OPTION "--copy_workers="
#define COPY_HASH_PAIRS_OPTION "--copy_hash_pairs="
#define COPY_HASH_PAIRS_ON 1u /* --copy_hash_pairs=0|1 */
/* The owner's split (9 Nov 2026): replica preparation, then the load stage, then the Engram stage, each with its own
 * parameters, mechanism, state and profile section. --stage=copy prepares and verifies the replicas alone: no tuning
 * and no profile; its state is the copy's own (the decisions beside the profile, the manifest, each replica's
 * journal and receipts). Without --stage the copy is followed by the load stage and the Engram stage
 * (stages_after_copy); the earlier combined tuning is retired (v0.0.4). */
enum autotune_stage { AUTOTUNE_STAGE_ALL, AUTOTUNE_STAGE_COPY };
static const char *const stage_names[] = { "all", "copy" };
#define LOAD_STAGE_OPTION AUTOTUNE_STAGE_OPTION "load"
#define ENGRAM_STAGE_OPTION AUTOTUNE_STAGE_OPTION "engram"
#define TOKENS_OPTION "--tokens="
#define RETUNE_OPTION "--retune"
#define RETUNE_SERIES_SUFFIX ".retune-series"
#define RETUNE_SERIES_BYTES 64u /* "load\nengram\n" and room */
#define STAGE_LINE_BYTES 16u /* a later stage's name and its newline */
/* anemone autotune PACK --stage=NAME --profile= --seconds= --replica_external= */
#define STAGE_FIXED_ARGUMENTS 7u
#define DECIMAL_DIGITS_MOST 10u /* a uint32_t */
static volatile sig_atomic_t autotune_stop;
static void autotune_signal(int signal_number)
{
	(void)signal_number;
	autotune_stop = 1;
}
void storage_autotune_hint(FILE *output)
{
	if (output == NULL)
		return;
	(void)fputs(
		"storage: no tuned storage profile: run `anemone autotune` before serving for best load/lookup "
		"performance\n",
		output);
}
uint64_t autotune_now_nanoseconds(void)
{
	struct timespec time;
	(void)clock_gettime(CLOCK_MONOTONIC, &time);
	return (uint64_t)time.tv_sec * NS_SECOND + (uint64_t)time.tv_nsec;
}

uint64_t autotune_cpu_nanoseconds(void)
{
	struct rusage usage;
	if (getrusage(RUSAGE_SELF, &usage) != 0)
		return 0;
	return ((uint64_t)usage.ru_utime.tv_sec +
		(uint64_t)usage.ru_stime.tv_sec) *
		       NS_SECOND +
	       ((uint64_t)usage.ru_utime.tv_usec +
		(uint64_t)usage.ru_stime.tv_usec) *
		       NS_MICROSECOND;
}

/* A controller's composite temperature file: its first hwmon's. */
static bool sensor_find(const char *controller, char *file, size_t capacity)
{
	DIR *const sensors = opendir(controller);
	if (sensors == NULL)
		return false;
	bool found = false;
	const struct dirent *sensor;
	while (!found && (sensor = readdir(sensors)) != NULL)
		if (strncmp(sensor->d_name, HWMON_NAME_PREFIX,
			    strlen(HWMON_NAME_PREFIX)) == 0) {
			const int length = snprintf(file, capacity,
						    "%s/%s/" TEMPERATURE_FILE,
						    controller, sensor->d_name);
			found = length > 0 && (size_t)length < capacity;
		}
	(void)closedir(sensors);
	return found;
}

void autotune_drives_read(struct autotune_drives *drives)
{
	memset(drives, 0, sizeof(*drives));
	DIR *const controllers = opendir(NVME_CLASS_DIRECTORY);
	if (controllers == NULL)
		return;
	const struct dirent *entry;
	while (drives->count < AUTOTUNE_DRIVES_MOST &&
	       (entry = readdir(controllers)) != NULL) {
		char controller[PATH_MAX], file[PATH_MAX],
			text[TEMPERATURE_TEXT_BYTES];
		const int length = snprintf(controller, sizeof(controller),
					    "%s/%s", NVME_CLASS_DIRECTORY,
					    entry->d_name);
		if (strncmp(entry->d_name, NVME_NAME_PREFIX,
			    strlen(NVME_NAME_PREFIX)) != 0 ||
		    strlen(entry->d_name) >= AUTOTUNE_DRIVE_NAME_BYTES ||
		    length <= 0 || (size_t)length >= sizeof(controller) ||
		    !sensor_find(controller, file, sizeof(file)))
			continue;
		const int descriptor = open(file, O_RDONLY | O_CLOEXEC);
		const ssize_t count =
			descriptor >= 0 ?
				read(descriptor, text, sizeof(text) - 1u) :
				-1;
		if (descriptor >= 0)
			(void)close(descriptor);
		if (count <= 0)
			continue;
		text[count] = '\0';
		char *end = NULL;
		const long long value = strtoll(text, &end, (int)DECIMAL_BASE);
		if (end == text)
			continue;
		(void)snprintf(drives->names[drives->count],
			       sizeof(drives->names[drives->count]), "%.*s",
			       (int)(AUTOTUNE_DRIVE_NAME_BYTES - 1u),
			       entry->d_name);
		drives->millicelsius[drives->count++] = value;
	}
	(void)closedir(controllers);
}

void autotune_drives_json(FILE *output, const char *name,
			  const struct autotune_drives *drives)
{
	(void)fprintf(output, ",\"%s\":{", name);
	for (uint32_t index = 0; index < drives->count; index++) {
		(void)fputs(index == 0 ? "" : ",", output);
		prepare_json_text(output, drives->names[index]);
		(void)fprintf(output, ":%" PRId64, drives->millicelsius[index]);
	}
	(void)fputc('}', output);
}

bool autotune_number_option(const char *text, uint32_t most, uint32_t *output)
{
	if (*text == '\0')
		return false;
	uint64_t value = 0;
	for (; *text != '\0'; text++) {
		if (*text < '0' || *text > '9' || value > most / DECIMAL_BASE)
			return false;
		value = value * DECIMAL_BASE + (unsigned)(*text - '0');
		if (value > most)
			return false;
	}
	*output = (uint32_t)value;
	return true;
}
static bool path_suffix(char *output, size_t capacity, const char *prefix,
			const char *suffix)
{
	const int length = snprintf(output, capacity, "%s%s", prefix, suffix);
	return length >= 0 && (size_t)length < capacity;
}
enum directory_state { DIRECTORY_OWNED, DIRECTORY_MISSING, DIRECTORY_REFUSED };
/* Only configuration/copy directories chosen by the explicit operator are created (create), every component walked
 * without following a link; --plan walks the same way and creates nothing (a missing component: missing). */
static enum directory_state directory_state(const char *path, bool create)
{
	char current[PATH_MAX];
	if (path[0] != '/' || strlen(path) >= sizeof(current))
		return DIRECTORY_REFUSED;
	(void)snprintf(current, sizeof(current), "%s", path);
	for (char *end = current + 1;; end++) {
		if (*end != '/' && *end != '\0')
			continue;
		const char saved = *end;
		*end = '\0';
		struct stat file;
		if (lstat(current, &file) != 0) {
			if (errno == ENOENT && !create)
				return DIRECTORY_MISSING;
			if (errno != ENOENT || mkdir(current, S_IRWXU) != 0)
				return DIRECTORY_REFUSED;
		} else if (!S_ISDIR(file.st_mode)) {
			return DIRECTORY_REFUSED;
		}
		*end = saved;
		if (saved == '\0')
			break;
	}
	struct stat file;
	if (stat(path, &file) != 0 || file.st_uid != geteuid())
		return DIRECTORY_REFUSED;
	const bool shared = (file.st_mode & (S_IWGRP | S_IWOTH)) != 0;
	if (shared && strstr(path, "/anemone_engram_rnd/") != NULL)
		return !create || chmod(path, S_IRWXU) == 0 ? DIRECTORY_OWNED :
							      DIRECTORY_REFUSED;
	return shared ? DIRECTORY_REFUSED : DIRECTORY_OWNED;
}
static bool owned_directory(const char *path)
{
	return directory_state(path, true) == DIRECTORY_OWNED;
}
bool autotune_refused(const char *reason, const char *path)
{
	(void)fprintf(stderr, "autotune: refused: %s: %s\n", reason, path);
	return false;
}
result_status
prepare_destination_device(const char *mount_table, const char *directory,
			   struct local_device_description *device)
{
	char ancestor[PATH_MAX];
	(void)snprintf(ancestor, sizeof(ancestor), "%s", directory);
	result_status described =
		local_device_describe_mounts(mount_table, ancestor, device);
	/* Climb only past a directory that does not exist yet: an automount that has not fired, or any other device
	   that cannot be described, refuses the destination before anything is created below it. */
	for (uint32_t depth = 0; described == STATUS_INPUT_OUTPUT_ERROR &&
				 depth < LOCAL_DEVICE_CANDIDATES_MOST;
	     depth++) {
		char *const end = strrchr(ancestor, '/');
		if (end == NULL || end == ancestor)
			break;
		*end = '\0';
		described = local_device_describe_mounts(mount_table, ancestor,
							 device);
	}
	return described;
}

void prepare_decision_json(FILE *output, const struct prepare_target *target)
{
	(void)fputs("{\"path\":", output);
	prepare_json_text(output, target->path);
	(void)fputs(",\"key\":", output);
	prepare_json_text(output, target->key);
	(void)fprintf(output, ",\"included\":%u,\"resumable\":%u,\"reason\":",
		      target->included ? 1u : 0u, target->resumable ? 1u : 0u);
	prepare_json_text(output, target->reason != NULL ? target->reason :
							   "invalid_decision");
	(void)fputs(",\"mount\":", output);
	prepare_json_text(output, target->device.mount);
	(void)fputs(",\"model\":", output);
	prepare_json_text(output, target->device.model);
	(void)fputs(",\"transport\":", output);
	prepare_json_text(output, target->device.transport);
	(void)fputs(",\"filesystem\":", output);
	prepare_json_text(output, target->device.filesystem);
	(void)fputs(",\"physical_devices\":[", output);
	for (uint32_t index = 0; index < target->device.count; index++)
		(void)fprintf(output, "%s\"%016" PRIx64 "\"",
			      index == 0 ? "" : ",",
			      target->device.physical[index]);
	/* receipt_trusted_bytes: the bytes a final's matching verification receipt vouched for, unread (only its spot
	 * chunks were read); verified and trusted sum to the pack's bytes for a final admitted on its receipt */
	(void)fprintf(output,
		      "],\"written_bytes\":%" PRIu64
		      ",\"verified_bytes\":%" PRIu64
		      ",\"receipt_trusted_bytes\":%" PRIu64
		      ",\"extent_count\":%" PRIu64
		      ",\"largest_extent\":%" PRIu64 ",\"copy_seconds\":%.6f}",
		      target->written, target->verified, target->trusted,
		      target->extents.count, target->extents.largest,
		      target->seconds);
}
result_status prepare_decisions_write(const char *path,
				      const struct prepare_target *targets,
				      uint32_t count, bool complete,
				      uint32_t copy_workers)
{
	struct prepare_file file;
	const result_status opened = prepare_file_open(path, &file);
	if (opened != STATUS_OK)
		return opened;
	(void)fprintf(
		file.stream,
		"{\"schema\":1,\"complete\":%u,\"copy_workers\":%u,\"decisions\":[",
		complete ? 1u : 0u, copy_workers);
	for (uint32_t index = 0; index < count; index++) {
		if (index != 0)
			(void)fputc(',', file.stream);
		prepare_decision_json(file.stream, &targets[index]);
	}
	(void)fputs("]}\n", file.stream);
	return prepare_file_commit(&file);
}
/* --plan: what a run would do with the checkpoint beside one selected copy, read-only (directory_state without
 * creating, then prepare_checkpoint_classify, both in the run's order): one line per target, and its kind counted. No
 * run removes a foreign partial, and none is removed by a plan. */
static void plan_checkpoint(const struct prepare_manifest *manifest,
			    const struct prepare_target *target,
			    uint32_t floor_percent,
			    uint32_t kinds[PREPARE_CHECKPOINT_KINDS])
{
	struct prepare_checkpoint checkpoint = {
		.kind = PREPARE_CHECKPOINT_NONE, .why = ""
	};
	char shown[PATH_MAX], directory[PATH_MAX];
	local_device_display(shown, sizeof(shown), target->path);
	(void)snprintf(directory, sizeof(directory), "%s", target->path);
	char *const slash = strrchr(directory, '/');
	if (slash != NULL)
		*slash = '\0';
	const enum directory_state state =
		slash == NULL ? DIRECTORY_REFUSED :
				directory_state(directory, false);
	if (state == DIRECTORY_REFUSED) {
		checkpoint.kind = PREPARE_CHECKPOINT_UNKNOWN;
		checkpoint.refusal = "destination_directory_policy";
	} else if (state == DIRECTORY_OWNED &&
		   prepare_checkpoint_classify(manifest, target->path,
					       target->key, floor_percent,
					       &checkpoint) != STATUS_OK) {
		checkpoint.kind = PREPARE_CHECKPOINT_UNKNOWN;
		checkpoint.error = ENAMETOOLONG;
	} else if (state == DIRECTORY_MISSING) {
		/* the run creates the directory, then admits the whole pack above the floor before any copy */
		checkpoint.refusal = local_device_space_refusal(
			target->device.total_bytes,
			target->device.available_bytes, manifest->bytes,
			floor_percent, 0, target->device.readonly);
	}
	kinds[checkpoint.kind]++;
	static const char *const kind_names[PREPARE_CHECKPOINT_KINDS] = {
		"none", "engine", "adoptable", "foreign", "held", "unknown"
	};
	const bool final = checkpoint.beside_final;
	char action[PATH_MAX];
	if (checkpoint.kind == PREPARE_CHECKPOINT_UNKNOWN)
		(void)snprintf(
			action, sizeof(action),
			"%s%s%s: a run refuses the target and examines nothing",
			checkpoint.refusal != NULL ? "refused as " :
						     "unknown (",
			checkpoint.refusal != NULL ? checkpoint.refusal :
						     strerror(checkpoint.error),
			checkpoint.refusal != NULL ? "" : ")");
	else if (checkpoint.kind == PREPARE_CHECKPOINT_HELD)
		(void)snprintf(
			action, sizeof(action),
			"refused as %s and left untouched, not examined: %s",
			checkpoint.refusal, checkpoint.why);
	else if (checkpoint.refusal != NULL)
		(void)snprintf(
			action, sizeof(action), "refused as %s: %s",
			checkpoint.refusal,
			checkpoint.kind == PREPARE_CHECKPOINT_NONE ?
				"no copy starts" :
			strcmp(checkpoint.refusal,
			       "finalized_replica_unverified") == 0 ?
				"the final is never trusted before it verifies; it and this partial are kept" :
			checkpoint.removed ?
				"the engine's checkpoint is removed with its journal" :
			checkpoint.adopted ?
				"the engine's adopted checkpoint is kept; never removed" :
			checkpoint.kind == PREPARE_CHECKPOINT_ENGINE ?
				"the engine's checkpoint is kept" :
				"kept, not examined for adoption; the engine never removes or changes it");
	else if (checkpoint.kind == PREPARE_CHECKPOINT_NONE)
		(void)snprintf(
			action, sizeof(action), "%s",
			final && checkpoint.receipt ?
				"no partial; the final matches its verification receipt: a run spot-checks it "
				"(--verify=full reads every byte) and uses it if the chunks match" :
			final ? "no partial; a run verifies the final file every byte and uses it if it matches" :
			state == DIRECTORY_MISSING ?
				"no partial; a run creates the directory and starts a fresh copy" :
				"no partial; a run starts a fresh copy");
	else if (checkpoint.kind == PREPARE_CHECKPOINT_ENGINE)
		(void)snprintf(
			action, sizeof(action), "%s",
			checkpoint.adopted && final ?
				"the engine's adopted checkpoint beside a final: the final is verified and used if it "
				"matches; the checkpoint is kept, never removed" :
			checkpoint.adopted ?
				"the engine's adopted checkpoint: resumed; never removed" :
			final ? "the engine's checkpoint beside a final: removed once the final verifies every byte, "
				"otherwise kept" :
				"the engine's checkpoint: resumed; removed only for a space, read-only or policy"
				" refusal, otherwise kept");
	else if (checkpoint.kind == PREPARE_CHECKPOINT_ADOPTABLE)
		(void)snprintf(
			action, sizeof(action),
			"adopted after its leading %" PRIu64
			" bytes verified byte-equal to the source, then resumed; never removed",
			checkpoint.proven_bytes);
	else if (final)
		(void)snprintf(
			action, sizeof(action),
			"beside a final (%s): the final is verified and used if it matches; this partial is kept, "
			"never removed or changed",
			checkpoint.why);
	else
		(void)snprintf(
			action, sizeof(action),
			"refused as foreign_partial_kept and kept (%s); the engine never removes or changes it",
			checkpoint.why);
	(void)fprintf(stderr,
		      "storage: plan checkpoint %s" PREPARE_PARTIAL_SUFFIX
		      " %s bytes=%" PRIu64 " allocated=%" PRIu64 ": %s\n",
		      shown, kind_names[checkpoint.kind], checkpoint.bytes,
		      checkpoint.allocated, action);
}

bool autotune_profile_default(char *output, size_t capacity)
{
	const char *const xdg = getenv("XDG_CONFIG_HOME"), *const home = getenv(
								   "HOME");
	const int length =
		xdg != NULL && *xdg != '\0' ?
			snprintf(output, capacity,
				 "%s/anemone/storage-profile.json", xdg) :
		home != NULL ?
			snprintf(output, capacity,
				 "%s/.config/anemone/storage-profile.json",
				 home) :
			-1;
	return (length > 0 && (size_t)length < capacity) ||
	       autotune_refused(
		       (xdg == NULL || *xdg == '\0') && home == NULL ?
			       "no default profile without XDG_CONFIG_HOME or HOME" :
			       "the default profile's path does not fit",
		       "give --profile=ABSOLUTE");
}

/* A run without --stage after its copy (stages_after_copy): the pack's real path, the profile, the --tokens arguments
 * as given, the lease's end, the replica policy, --retune and --plan. */
struct stage_run {
	const char *source, *profile, *series;
	char *const *tokens;
	uint32_t token_count;
	uint64_t deadline_ns;
	bool external, retune, plan;
};
struct later_stage {
	const char *name, *option;
	bool (*current)(const char *source, const char *profile, bool external,
			char reason[STORAGE_PROFILE_REASON_BYTES]);
	int (*run)(int count, char **arguments);
	uint32_t seconds_most;
	bool takes_tokens;
};
static const struct later_stage later_stages[] = {
	{ "load", LOAD_STAGE_OPTION, storage_autotune_load_current,
	  storage_autotune_load_cli, AUTOTUNE_SECONDS_MOST, false },
	{ "engram", ENGRAM_STAGE_OPTION, storage_autotune_engram_current,
	  storage_autotune_engram_cli, AUTOTUNE_ENGRAM_SECONDS_MOST, true }
};
#define LATER_STAGES (sizeof(later_stages) / sizeof(later_stages[0]))
_Static_assert(sizeof(ENGRAM_STAGE_OPTION) >= sizeof(LOAD_STAGE_OPTION),
	       "a later stage's option fits the Engram stage's room");

/* The --retune series (storage_autotune.h): whether <profile>.retune-series, a regular file of this user's alone,
 * names the stage on a line of its own; series_record appends the stage's line once it finished. */
static bool series_finished(const char *series, const char *name)
{
	char text[RETUNE_SERIES_BYTES + 1u] = "", line[STAGE_LINE_BYTES];
	const int descriptor =
		open(series, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
	struct stat file;
	const bool owned = descriptor >= 0 && fstat(descriptor, &file) == 0 &&
			   S_ISREG(file.st_mode) && file.st_uid == geteuid() &&
			   (file.st_mode & (S_IWGRP | S_IWOTH)) == 0;
	const ssize_t length =
		owned ? read(descriptor, text, RETUNE_SERIES_BYTES) : -1;
	if (descriptor >= 0)
		(void)close(descriptor);
	const int line_length = snprintf(line, sizeof(line), "%s\n", name);
	if (length <= 0 || line_length <= 0 ||
	    (size_t)line_length >= sizeof(line))
		return false;
	text[length] = '\0';
	for (const char *found = strstr(text, line); found != NULL;
	     found = strstr(found + 1, line))
		if (found == text || found[-1] == '\n')
			return true;
	return false;
}
static void series_record(const char *series, const char *name)
{
	const int descriptor = open(
		series, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_CLOEXEC,
		S_IRUSR | S_IWUSR);
	struct stat file;
	if (descriptor >= 0 && fstat(descriptor, &file) == 0 &&
	    S_ISREG(file.st_mode) && file.st_uid == geteuid())
		(void)dprintf(descriptor, "%s\n", name);
	if (descriptor >= 0)
		(void)close(descriptor);
}

/* The stages after the copy for a run without --stage: the load stage, then the Engram stage, each through its own
 * command line with the profile, the replica policy and what remains of the lease (at most the stage's own bound), so
 * each keeps its own options, state and section. A stage whose section is current (it names this pack and these
 * copies: storage_autotune_*_current) is skipped, one line naming it, unless --retune (then only one that finished
 * in the same --retune series); the Engram stage is skipped, one line naming it, without --tokens. With --plan, one
 * line a stage says what a run would do. The first stage that does not finish ends the run with its exit:
 * AUTOTUNE_RETRY_EXIT (the lease, a stop signal), and the same command resumes it, or a failure. */
static int stages_after_copy(const struct stage_run *run)
{
	for (uint32_t index = 0; index < LATER_STAGES; index++) {
		const struct later_stage *const stage = &later_stages[index];
		char reason[STORAGE_PROFILE_REASON_BYTES] = RETUNE_OPTION;
		if (stage->takes_tokens && run->token_count == 0) {
			(void)fprintf(
				stderr,
				"storage: autotune stage=%s skipped: no --tokens (the Engram stage replays real "
				"prompts: give --tokens=TOKEN_IDS_U32, or run --stage=engram later)\n",
				stage->name);
			continue;
		}
		const bool finished = run->retune &&
				      series_finished(run->series, stage->name);
		if ((!run->retune || finished) &&
		    stage->current(run->source, run->profile, run->external,
				   reason)) {
			(void)fprintf(
				stderr,
				"storage: autotune stage=%s skipped: %s\n",
				stage->name,
				finished ?
					"it finished in this --retune series and its section names this pack and these "
					"copies" :
					"its section names this pack and these copies (--retune measures it again)");
			continue;
		}
		if (run->plan) {
			(void)fprintf(
				stderr,
				"storage: plan: the %s stage would run (%s); --stage=%s --plan shows its points\n",
				stage->name, reason, stage->name);
			continue;
		}
		const uint64_t now = autotune_now_nanoseconds();
		const uint64_t remaining =
			run->deadline_ns > now ?
				(run->deadline_ns - now) / NS_SECOND :
				0;
		/* a checking build's fault point: the lease spent before this stage */
		if (autotune_stop != 0 || remaining == 0 ||
		    FAULT_POINT(autotune_stage_lease)) {
			(void)fprintf(
				stderr,
				"storage: autotune stage=%s not started: %s; repeat the same command to resume\n",
				stage->name,
				autotune_stop != 0 ? "a stop signal" :
						     "the lease is spent");
			return AUTOTUNE_RETRY_EXIT;
		}
		(void)fprintf(stderr, "storage: autotune stage=%s runs (%s)\n",
			      stage->name, reason);
		char program[] = "anemone", command[] = "autotune",
		     source[PATH_MAX], option[sizeof(ENGRAM_STAGE_OPTION)],
		     profile[PATH_MAX + sizeof("--profile=")],
		     seconds[sizeof("--seconds=") + DECIMAL_DIGITS_MOST],
		     external[sizeof("--replica_external=") +
			      DECIMAL_DIGITS_MOST];
		const int profile_length = snprintf(
			profile, sizeof(profile), "--profile=%s", run->profile);
		(void)snprintf(source, sizeof(source), "%s", run->source);
		(void)snprintf(option, sizeof(option), "%s", stage->option);
		(void)snprintf(seconds, sizeof(seconds), "--seconds=%" PRIu64,
			       remaining < stage->seconds_most ?
				       remaining :
				       (uint64_t)stage->seconds_most);
		(void)snprintf(external, sizeof(external),
			       "--replica_external=%u",
			       run->external ? 1u : 0u);
		char *arguments[STAGE_FIXED_ARGUMENTS +
				AUTOTUNE_ENGRAM_TRACES_MOST] = {
			program, command, source,  option,
			profile, seconds, external
		};
		uint32_t count = STAGE_FIXED_ARGUMENTS;
		for (uint32_t token = 0;
		     stage->takes_tokens && token < run->token_count; token++)
			arguments[count++] = run->tokens[token];
		if (profile_length <= 0 ||
		    (size_t)profile_length >= sizeof(profile))
			return EXIT_FAILURE;
		const int exited = stage->run((int)count, arguments);
		if (exited != EXIT_SUCCESS)
			return exited;
		if (run->retune)
			series_record(run->series, stage->name);
	}
	return EXIT_SUCCESS;
}

int storage_autotune_cli(int count, char **arguments)
{
	/* the load and Engram stages have their own options, mechanism, state and section: storage_autotune_load.c,
	 * storage_autotune_engram.c */
	for (int index = AUTOTUNE_FIRST_OPTION; index < count; index++) {
		if (strcmp(arguments[index], LOAD_STAGE_OPTION) == 0)
			return storage_autotune_load_cli(count, arguments);
		if (strcmp(arguments[index], ENGRAM_STAGE_OPTION) == 0)
			return storage_autotune_engram_cli(count, arguments);
	}
	struct prepare_target *const decisions =
		calloc(LOCAL_DEVICE_CANDIDATES_MOST, sizeof(*decisions));
	struct prepare_manifest *const manifest = calloc(1, sizeof(*manifest));
	if (decisions == NULL || manifest == NULL) {
		free(decisions);
		free(manifest);
		return EXIT_FAILURE;
	}
	uint32_t seconds = AUTOTUNE_SECONDS_DEFAULT, decision_count = 0,
		 copy_workers = PREPARE_COPY_WORKERS_DEFAULT,
		 floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT,
		 token_count = 0;
	/* the Engram stage's --tokens arguments, as given, for the run without --stage */
	char *tokens[AUTOTUNE_ENGRAM_TRACES_MOST];
	const char *source_path = NULL, *manifest_path = NULL,
		   *profile_path = NULL;
	const char *destinations[PREPARE_TARGETS_MOST];
	uint32_t destination_count = 0;
	bool valid = true, help = false, plan = false, verify_full = false,
	     pair_source_hashes = PREPARE_PAIR_SOURCE_HASHES_DEFAULT,
	     external = false, retune = false;
	enum autotune_stage stage = AUTOTUNE_STAGE_ALL;
	for (int index = AUTOTUNE_FIRST_OPTION; valid && index < count;
	     index++) {
		const char *const argument = arguments[index];
		if (strcmp(argument, "--plan") == 0) {
			plan = true;
			continue;
		}
		if (strcmp(argument, "--help") == 0) {
			help = true;
			break;
		}
		if (strcmp(argument, RETUNE_OPTION) == 0) {
			retune = true;
			continue;
		}
		/* --verify=full: read every final in full, whatever its receipt says; spot (the default): a final
		 * that matches its verification receipt is spot-checked */
		if (strcmp(argument, "--verify=full") == 0 ||
		    strcmp(argument, "--verify=spot") == 0) {
			verify_full = strcmp(argument, "--verify=full") == 0;
			continue;
		}
		if (strncmp(argument, AUTOTUNE_STAGE_OPTION,
			    strlen(AUTOTUNE_STAGE_OPTION)) == 0) {
			valid = strcmp(argument + strlen(AUTOTUNE_STAGE_OPTION),
				       stage_names[AUTOTUNE_STAGE_COPY]) == 0 ||
				autotune_refused(
					"no such stage (copy, load and engram)",
					argument);
			stage = AUTOTUNE_STAGE_COPY;
			continue;
		}
		if (strncmp(argument, "--manifest=", strlen("--manifest=")) ==
		    0)
			manifest_path = argument + strlen("--manifest=");
		else if (strncmp(argument,
				 "--profile=", strlen("--profile=")) == 0)
			profile_path = argument + strlen("--profile=");
		else if (strncmp(argument, "--target=", strlen("--target=")) ==
				 0 &&
			 destination_count < PREPARE_TARGETS_MOST)
			destinations[destination_count++] =
				argument + strlen("--target=");
		else if (strncmp(argument, TOKENS_OPTION,
				 strlen(TOKENS_OPTION)) == 0 &&
			 token_count < AUTOTUNE_ENGRAM_TRACES_MOST)
			tokens[token_count++] = arguments[index];
		else if (strncmp(argument, TOKENS_OPTION,
				 strlen(TOKENS_OPTION)) == 0)
			valid = autotune_refused("at most 16 --tokens files",
						 argument);
		else if (strncmp(argument, "--prompt=", strlen("--prompt=")) ==
			 0)
			/* the combined tuning's prompt length, retired with it */
			valid = autotune_refused(AUTOTUNE_PROMPT_RETIRED,
						 argument);
		else if (strncmp(argument,
				 "--seconds=", strlen("--seconds=")) == 0)
			valid = autotune_number_option(
					argument + strlen("--seconds="),
					AUTOTUNE_SECONDS_MOST, &seconds) &&
				seconds != 0;
		else if (strncmp(argument, "--replica_external=",
				 strlen("--replica_external=")) == 0) {
			uint32_t external_value = 0;
			valid = autotune_number_option(
				argument + strlen("--replica_external="), 1,
				&external_value);
			external = external_value != 0;
		} else if (strncmp(argument, COPY_WORKERS_OPTION,
				   strlen(COPY_WORKERS_OPTION)) == 0) {
			/* the copy stage's own parameter: worker threads, each one chunk through every stage */
			valid = (autotune_number_option(
					 argument + strlen(COPY_WORKERS_OPTION),
					 PREPARE_COPY_WORKERS_MOST,
					 &copy_workers) &&
				 copy_workers != 0) ||
				autotune_refused(
					"--copy_workers takes 1 to 32 workers",
					argument);
		} else if (strncmp(argument, COPY_HASH_PAIRS_OPTION,
				   strlen(COPY_HASH_PAIRS_OPTION)) == 0) {
			/* the copy stage's two-chunk source hashing (pair_source_hashes) */
			uint32_t pairs = 0;
			valid = autotune_number_option(
					argument +
						strlen(COPY_HASH_PAIRS_OPTION),
					COPY_HASH_PAIRS_ON, &pairs) ||
				autotune_refused(
					"--copy_hash_pairs takes 0 or 1",
					argument);
			pair_source_hashes = pairs != 0;
		} else if (strncmp(argument, "--replica_floor_percent=",
				   strlen("--replica_floor_percent=")) == 0) {
			valid = autotune_number_option(
				argument + strlen("--replica_floor_percent="),
				LOCAL_DEVICE_PERCENT - 1u, &floor_percent);
		} else if (argument[0] != '-' && source_path == NULL)
			source_path = argument;
		else
			valid = false;
	}
	char default_profile[PATH_MAX] = "", default_manifest[PATH_MAX] = "",
	     progress[PATH_MAX], series[PATH_MAX] = "";
	/* before the default profile, which needs XDG_CONFIG_HOME or HOME: the stage's own refusal is named first */
	if (valid && stage == AUTOTUNE_STAGE_COPY && token_count != 0)
		valid = autotune_refused(
			"--tokens is the Engram stage's option, not the copy stage's",
			"--tokens");
	if (valid && stage == AUTOTUNE_STAGE_COPY && retune)
		valid = autotune_refused(
			"--retune measures the load and Engram stages again: the run without --stage takes it, not the "
			"copy stage",
			RETUNE_OPTION);
	/* --help prints the usage alone, whatever HOME is */
	if (profile_path == NULL) {
		valid = valid &&
			(help ||
			 autotune_profile_default(default_profile,
						  sizeof(default_profile)));
		profile_path = default_profile;
	}
	if (manifest_path == NULL) {
		valid = valid &&
			path_suffix(default_manifest, sizeof(default_manifest),
				    profile_path, PROFILE_PATH_SUFFIX);
		manifest_path = default_manifest;
	}
	valid = valid && source_path != NULL && profile_path[0] == '/' &&
		manifest_path[0] == '/' &&
		path_suffix(progress, sizeof(progress), profile_path,
			    ".copy.json") &&
		path_suffix(series, sizeof(series), profile_path,
			    RETUNE_SERIES_SUFFIX);
	/* The profile and manifest record these paths, and JSON text is UTF-8: refused before any copy, so no copy is
	   held that could never resume. The source is recorded by its real path, checked once it is resolved. */
	const char *const unrecorded =
		!prepare_utf8_text(profile_path)  ? "--profile" :
		!prepare_utf8_text(manifest_path) ? "--manifest" :
						    NULL;
	if (valid && unrecorded != NULL) {
		(void)fprintf(
			stderr,
			"autotune: the %s path is no UTF-8, so the profile cannot record it\n",
			unrecorded);
		valid = false;
	}
	if (!valid || help) {
		(void)fputs(
			"usage: anemone autotune PACK [--tokens=TOKEN_IDS_U32 ...] [--retune] "
			"[--profile=ABSOLUTE] [--manifest=ABSOLUTE] [--target=ABSOLUTE_FILE ...] [--seconds=1650] "
			"[--plan] [--verify=spot|full] [--replica_external=0|1] [--replica_floor_percent=20] "
			"[--copy_workers=1..32 (24)] [--copy_hash_pairs=0|1 (1)]\n"
			"       anemone autotune PACK --stage=copy [the same options without --tokens and --retune]\n"
			"       anemone autotune PACK --stage=load [--help: the load stage's own options]\n"
			"       anemone autotune PACK --stage=engram --tokens=TOKEN_IDS_U32 "
			"[--help: the Engram stage's own options]\n"
			"Without --stage: the copy, then the load stage, then the Engram stage (skipped without "
			"--tokens); a stage whose section names this pack and these copies is skipped unless "
			"--retune. Copies are engine-native; repeat the same command after exit75 to resume. "
			"--stage=copy prepares and verifies the replicas alone: no tuning, no profile. "
			"--copy_workers is one pool shared by all targets: 1 read + 1 hash + k writes + k "
			"readbacks a chunk for k targets; --copy_hash_pairs=1 (the default) hashes two source chunks a "
			"worker in one pass, 0 one at a time (the same digests). "
			"A stage's section is an inactive candidate until that stage's --qualify.\n",
			stderr);
		free(decisions);
		free(manifest);
		return help ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	char parent[PATH_MAX];
	(void)snprintf(parent, sizeof(parent), "%s", profile_path);
	char *slash = strrchr(parent, '/');
	if (slash != NULL)
		*(slash == parent ? slash + 1 : slash) = '\0';
	struct local_device_description source_device = { 0 };
	/* an EXL3 map: its largest source's drive */
	char bytes_path[EXL3_MAP_PATH_BYTES];
	const result_status described_source = local_device_describe(
		storage_device_path(source_path, bytes_path), &source_device);
	const char *const source_refusal =
		described_source != STATUS_OK ?
			"device_disappeared_or_unknown" :
			local_device_media_refusal(&source_device, external);
	if (source_refusal != NULL)
		local_device_print(stderr, &source_device, source_refusal, 0,
				   floor_percent, 0, 0);
	/* A run creates or keeps the profile's folder as this user's alone; --plan checks the same without creating
	 * it. */
	const enum directory_state profile_folder =
		source_refusal == NULL ? directory_state(parent, !plan) :
					 DIRECTORY_REFUSED;
	valid = source_refusal == NULL &&
		(profile_folder == DIRECTORY_OWNED ||
		 (plan && profile_folder == DIRECTORY_MISSING) ||
		 autotune_refused(
			 "the profile's folder is not this user's alone (owned by this user, writable by no group or "
			 "other user)",
			 parent)) &&
		(strcmp(profile_path, manifest_path) != 0 ||
		 autotune_refused("--profile and --manifest name the same file",
				  profile_path));
	char real_source[PATH_MAX];
	valid = valid &&
		(realpath(source_path, real_source) != NULL ||
		 autotune_refused("the pack has no real path", source_path));
	if (valid && !prepare_utf8_text(real_source)) {
		(void)fprintf(
			stderr,
			"autotune: the source path is no UTF-8, so the profile cannot record it\n");
		valid = false;
	}
	char existing_profile[PATH_MAX];
	if (valid && realpath(profile_path, existing_profile) != NULL &&
	    strcmp(existing_profile, real_source) == 0)
		valid = autotune_refused("the profile path is the pack itself",
					 profile_path);
	/* The copy stage writes no profile: one in place stays as it is. A run without --stage refuses, before any
	 * copy, a profile its later stages would refuse (the combined tuning's schema 2, another source's). */
	const char *refusal = NULL;
	if (valid && stage == AUTOTUNE_STAGE_ALL &&
	    storage_stage_section_writable(profile_path, real_source,
					   &refusal) != STATUS_OK)
		valid = autotune_refused(
			refusal != NULL ? refusal :
					  "the profile cannot be checked",
			profile_path);
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	result_status status =
		valid ? storage_direct.reader_open(
				(const char *const[]){ real_source }, 1,
				&reader) :
			STATUS_INVALID_ARGUMENT;
	if (status == STATUS_OK)
		status = pack_open_entry(&storage_direct, reader, &pack);
	if (status != STATUS_OK && valid)
		(void)fprintf(
			stderr,
			"autotune: refused: the pack does not open (%s): %s\n",
			status_text(status), real_source);
	if (status != STATUS_OK) {
		storage_direct.reader_close(reader);
		free(decisions);
		free(manifest);
		return EXIT_FAILURE;
	}
	const struct pack_header header = *pack_header_of(pack);
	pack_close(pack);
	storage_direct.reader_close(reader);
	char pack_identity[STORAGE_PROFILE_KEY_BYTES];
	(void)snprintf(pack_identity, sizeof(pack_identity), "%016" PRIx64,
		       header.header_digest);
	struct stat manifest_file;
	if (lstat(manifest_path, &manifest_file) == 0) {
		status = prepare_manifest_read(manifest_path, manifest);
		valid = (status == STATUS_OK ||
			 autotune_refused("the manifest does not read",
					  manifest_path)) &&
			((strcmp(manifest->source, real_source) == 0 &&
			  strcmp(manifest->pack_identity, pack_identity) == 0 &&
			  manifest->bytes == header.file_byte_count) ||
			 autotune_refused(
				 "the manifest belongs to another pack (its source, identity or size differ)",
				 manifest_path));
	} else {
		valid = errno == ENOENT ||
			autotune_refused("the manifest cannot be examined",
					 manifest_path);
		(void)snprintf(manifest->source, sizeof(manifest->source), "%s",
			       real_source);
		(void)snprintf(manifest->pack_identity,
			       sizeof(manifest->pack_identity), "%s",
			       pack_identity);
		manifest->bytes = header.file_byte_count;
		manifest->chunk_bytes = PREPARE_CHUNK_BYTES;
		valid = valid &&
			((manifest->bytes != 0 &&
			  (manifest->bytes - 1u) / manifest->chunk_bytes <
				  PREPARE_CHUNKS_MOST) ||
			 autotune_refused(
				 "the pack's size is outside the copy's chunk range",
				 real_source));
		manifest->chunks =
			valid ? (uint32_t)((manifest->bytes - 1u) /
						   manifest->chunk_bytes +
					   1u) :
				0;
	}
	struct prepare_target *const original = &decisions[decision_count++];
	(void)snprintf(original->path, sizeof(original->path), "%s",
		       real_source);
	status = storage_device_key_for(
		storage_device_path(real_source, bytes_path), external,
		original->key, &original->device);
	original->included = original->complete = status == STATUS_OK;
	original->reason = original->included ? "verified_identity_and_bytes" :
						"hardware_identity_unavailable";
	valid = valid &&
		(original->included ||
		 autotune_refused("the pack's device has no hardware identity",
				  real_source));
	struct local_device_description *scan = NULL;
	uint32_t scan_count = 0;
	if (valid && destination_count == 0) {
		scan = calloc(LOCAL_DEVICE_CANDIDATES_MOST, sizeof(*scan));
		valid = (scan != NULL &&
			 local_device_candidates(external, scan,
						 LOCAL_DEVICE_CANDIDATES_MOST,
						 &scan_count) == STATUS_OK) ||
			autotune_refused("the local devices cannot be listed",
					 "/proc/self/mounts");
	}
	struct prepare_target copies[PREPARE_TARGETS_MOST];
	memset(copies, 0, sizeof(copies));
	uint32_t copy_count = 0;
	const uint32_t candidates = destination_count != 0 ? destination_count :
							     scan_count;
	for (uint32_t index = 0; valid && index < candidates &&
				 decision_count < LOCAL_DEVICE_CANDIDATES_MOST;
	     index++) {
		struct prepare_target *const target =
			&decisions[decision_count++];
		char directory[PATH_MAX];
		if (destination_count != 0) {
			(void)snprintf(target->path, sizeof(target->path), "%s",
				       destinations[index]);
			(void)snprintf(directory, sizeof(directory), "%s",
				       target->path);
			slash = strrchr(directory, '/');
			if (slash != NULL)
				*slash = '\0';
			if (prepare_destination_device(
				    "/proc/self/mounts", directory,
				    &target->device) != STATUS_OK)
				target->reason =
					strcmp(target->device.filesystem,
					       "autofs") == 0 ?
						"automount_not_mounted" :
						"device_disappeared_or_unknown";
		} else {
			target->device = scan[index];
			const char *base = strcmp(target->device.mount, "/") ==
							   0 ?
						   getenv("HOME") :
						   target->device.mount;
			const int length =
				base != NULL ?
					snprintf(
						directory, sizeof(directory),
						"%s/anemone_engram_rnd/replicas",
						base) :
					-1;
			const int target_length =
				length > 0 && (size_t)length <
							sizeof(directory) ?
					snprintf(target->path,
						 sizeof(target->path),
						 "%s/%s.anemone", directory,
						 pack_identity) :
					-1;
			if (target_length <= 0 ||
			    (size_t)target_length >= sizeof(target->path))
				target->reason = "destination_directory_policy";
		}
		/* A path or mount point that is no UTF-8 cannot be recorded exactly in the profile: that decision alone
		   is refused (its JSON shows U+FFFD), and the profile stays readable. */
		if (target->reason == NULL &&
		    (!prepare_utf8_text(target->path) ||
		     !prepare_utf8_text(target->device.mount)))
			target->reason = "destination_directory_policy";
		if (target->reason == NULL)
			target->reason = local_device_media_refusal(
				&target->device, external);
		for (uint32_t previous = 0;
		     target->reason == NULL && previous + 1u < decision_count;
		     previous++)
			if (decisions[previous].included &&
			    local_device_overlaps(&target->device,
						  &decisions[previous].device))
				target->reason = "duplicate_physical_device";
		if (target->reason == NULL &&
		    copy_count == PREPARE_TARGETS_MOST)
			target->reason = "device_limit";
		if (target->reason == NULL && !plan &&
		    !owned_directory(directory))
			target->reason = "destination_directory_policy";
		if (target->reason == NULL &&
		    storage_device_key_for(
			    plan ? target->device.mount : directory, external,
			    target->key, &target->device) != STATUS_OK)
			target->reason = "hardware_identity_unavailable";
		if (target->reason == NULL) {
			target->included = true;
			target->reason = "eligible";
			copies[copy_count++] = *target;
		}
		local_device_print(stderr, &target->device, target->reason,
				   manifest->bytes, floor_percent, 0, 0);
	}
	free(scan);
	uint64_t token_bytes = 0;
	/* the Engram stage's token files, checked before any copy (the stage checks them again, with its vocabulary) */
	for (uint32_t index = 0; valid && index < token_count; index++) {
		const char *const token_path =
			tokens[index] + strlen(TOKENS_OPTION);
		struct stat token_file;
		valid = (lstat(token_path, &token_file) == 0 &&
			 S_ISREG(token_file.st_mode) &&
			 token_file.st_size > 0 &&
			 (uint64_t)token_file.st_size % sizeof(uint32_t) == 0 &&
			 (uint64_t)token_file.st_size <=
				 AUTOTUNE_TOKENS_MOST_BYTES - token_bytes) ||
			autotune_refused(
				"a --tokens file is no non-empty regular file of whole uint32 ids within the total"
				" limit",
				token_path);
		if (valid)
			token_bytes += (uint64_t)token_file.st_size;
	}
	struct stage_run run = { .source = real_source,
				 .profile = profile_path,
				 .series = series,
				 .tokens = tokens,
				 .token_count = token_count,
				 .external = external,
				 .retune = retune,
				 .plan = plan };
	if (plan) {
		(void)fputs(
			"storage: plan only; no copy, tuning or profile write\n",
			stderr);
		uint32_t kinds[PREPARE_CHECKPOINT_KINDS] = { 0 };
		for (uint32_t index = 0; valid && index < copy_count; index++)
			plan_checkpoint(manifest, &copies[index], floor_percent,
					kinds);
		(void)printf(
			"{\"valid\":%s,\"stage\":\"%s\",\"selected_copies\":%u,\"copy_workers\":%u,"
			"\"token_bytes\":%" PRIu64
			",\"checkpoints\":{\"engine\":%u,\"adoptable\":%u,\"foreign\":%u,\"held\":%u,"
			"\"unknown\":%u}}\n",
			valid ? "true" : "false", stage_names[stage],
			copy_count, copy_workers, token_bytes,
			kinds[PREPARE_CHECKPOINT_ENGINE],
			kinds[PREPARE_CHECKPOINT_ADOPTABLE],
			kinds[PREPARE_CHECKPOINT_FOREIGN],
			kinds[PREPARE_CHECKPOINT_HELD],
			kinds[PREPARE_CHECKPOINT_UNKNOWN]);
		free(decisions);
		free(manifest);
		/* a run without --stage: what each later stage would do */
		return valid && stage == AUTOTUNE_STAGE_ALL ?
			       stages_after_copy(&run) :
		       valid ? EXIT_SUCCESS :
			       EXIT_FAILURE;
	}
	struct timespec time;
	valid = valid && (clock_gettime(CLOCK_MONOTONIC, &time) == 0 ||
			  autotune_refused("the monotonic clock does not read",
					   "CLOCK_MONOTONIC"));
	run.deadline_ns = valid ? (uint64_t)time.tv_sec * NS_SECOND +
					  (uint64_t)time.tv_nsec +
					  (uint64_t)seconds * NS_SECOND :
				  0;
	autotune_stop = 0;
	struct sigaction action = { .sa_handler = autotune_signal }, old_term,
			 old_int;
	(void)sigemptyset(&action.sa_mask);
	const bool term_set = sigaction(SIGTERM, &action, &old_term) == 0;
	const bool int_set = sigaction(SIGINT, &action, &old_int) == 0;
	struct prepare_copy_options copy = { .manifest_path = manifest_path,
					     .manifest = manifest,
					     .targets = copies,
					     .count = copy_count,
					     .floor_percent = floor_percent,
					     .workers = copy_workers,
					     .external = external,
					     .deadline_ns = run.deadline_ns,
					     .stop = &autotune_stop,
					     .verify_full = verify_full,
					     .pair_source_hashes =
						     pair_source_hashes };
	valid = valid && ((term_set && int_set) ||
			  autotune_refused("the stop signals cannot be handled",
					   "SIGTERM, SIGINT"));
	status = valid ? prepare_replicas(&copy) : STATUS_INVALID_ARGUMENT;
	for (uint32_t index = 0; index < copy_count; index++)
		for (uint32_t decision = 1; decision < decision_count;
		     decision++)
			if (strcmp(copies[index].path,
				   decisions[decision].path) == 0)
				decisions[decision] = copies[index];
	const result_status recorded =
		prepare_decisions_write(progress, decisions, decision_count,
					status == STATUS_OK, copy_workers);
	if (status == STATUS_OK && recorded != STATUS_OK)
		status = recorded;
	(void)fprintf(stderr,
		      "storage: autotune stage=%s status=%s resumable=%u\n",
		      stage_names[AUTOTUNE_STAGE_COPY], status_text(status),
		      status == STATUS_TIMED_OUT ? 1u : 0u);
	int exited = status == STATUS_TIMED_OUT ? AUTOTUNE_RETRY_EXIT :
		     status == STATUS_OK	? EXIT_SUCCESS :
						  EXIT_FAILURE;
	/* this process's stop handlers stay through the later stages: each sets its own and restores these */
	if (exited == EXIT_SUCCESS && stage == AUTOTUNE_STAGE_ALL)
		exited = stages_after_copy(&run);
	/* a run that does not end resumable ends its --retune series */
	if (valid && stage == AUTOTUNE_STAGE_ALL &&
	    exited != AUTOTUNE_RETRY_EXIT)
		(void)unlink(series);
	if (stage == AUTOTUNE_STAGE_ALL)
		(void)fprintf(
			stderr,
			"storage: autotune stage=%s exit=%d resumable=%u\n",
			stage_names[AUTOTUNE_STAGE_ALL], exited,
			exited == AUTOTUNE_RETRY_EXIT ? 1u : 0u);
	if (term_set)
		(void)sigaction(SIGTERM, &old_term, NULL);
	if (int_set)
		(void)sigaction(SIGINT, &old_int, NULL);
	free(decisions);
	free(manifest);
	return exited;
}
