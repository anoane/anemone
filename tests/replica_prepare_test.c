/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "replica_prepare.h"
#include "storage_autotune.h"
#include "anemone/hooks.h"
#include "anemone/pack.h"
#include "anemone/replicas.h"
#include "anemone/storage_profile.h"
#include "engram_pack.h"
#include "json.h"
#include "log_expectation.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <linux/io_uring.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#define FIXTURE_BYTES (8u * PACK_BLOCK_BYTES)
#define FIXTURE_CHUNK (2u * PACK_BLOCK_BYTES)
#define FIXTURE_TARGETS 2u
#define RECEIPT_TEMPORARY_SUFFIX \
	".verified.new" /* replica_prepare.c's receipt before its rename */
#define AUTOTUNE_CAPTURE_BYTES 8192u
#define RECEIPT_TEST_BYTES 512u
#define RECEIPT_TEST_FIELDS 9
#define FIXTURE_SECONDS 60u
#define NS_SECOND 1000000000ull
#define POISON_RUN_SECONDS_MOST \
	10u /* a poisoned run ends at once: far inside the fixture's deadline */
#define PROBE_RING_ENTRIES 1u
#define STOP_AFTER_CHUNKS \
	2u /* the stop test's run stops after its second written chunk */
#define RESUME_READBACKS \
	3u /* chunk 1 at once, chunks 2 and 3 after their writes */
#define RESUME_INSIDE_LAG 2u /* chunk 1 (unclaimed) and chunk 3 (the last) */
#define CHANGED_BYTE 100 /* in chunk 0 */
#define COPY_COUNT_FIELDS 2
#define ALLOCATION_BLOCK_BYTES 512u
#define PATTERN_MODULUS 251u
#define EXT4_EXTENT_BYTES \
	(128ull           \
	 << 20u) /* ext4's longest initialised extent with 4 KiB blocks */
#define LARGE_REPLICA_EXTENTS 3850u /* a 481 GiB replica */
#define PHYSICAL_START (1ull << 40u)
#define SCATTERED_EXTENTS 200u
#define SCATTERED_BYTES (4ull << 20u)
#define SCATTERED_STRIDE \
	2u /* a scattered extent takes every other physical slot */
#define FIXTURE_TABLE_BYTES (2u * PATH_MAX + 64u) /* two mount table lines */
/* A deadline already past: the run opens (and journals) its targets, then stops at once. */
#define PAST_DEADLINE_NS 1u
/* An hour back: a foreign partial written that long ago is quiet enough to adopt. */
#define AGED_SECONDS 3600
/* Credit fixture: a device of 100 bytes; available bytes below, near and above the total, and credits. */
#define CREDIT_TOTAL 100u
#define CREDIT_AVAILABLE_LOW 10u
#define CREDIT_AVAILABLE_HIGH 90u
#define CREDIT_AVAILABLE_OVER 120u
#define CREDIT_SMALL 5u
#define CREDIT_LARGE 20u
/* Corrupt-chunk masks of the foreign partials: chunk 1; chunk 0; chunks 1 and 3 (the proof stops at chunk 1 and never
 * reads chunk 3). */
#define SECOND_CHUNK (1u << 1u)
#define FIRST_CHUNK (1u << 0u)
#define UNPROVEN_CHUNKS ((1u << 1u) | (1u << 3u))
/* r17: chunk 0, which the adoption proved and marked, is corrupted after the adoption too: three chunks re-copied. */
#define ADOPTED_REPAIRED_CHUNKS 3u
#define CAPTURE_BYTES 16384u
/* a two-target run's whole stderr, its closing stage lines included */
#define RUN_CAPTURE_BYTES 65536u
/* a captured line's field prefix, or a decision record's field and its value */
#define FIELD_TEXT_BYTES 128u
#define STAGE_LEAF_BYTES 32u /* "/<16 hex digits>.anemone" and its terminator */
#define STAGE_WORKERS_ABOVE_MOST \
	33u /* the --copy_workers value the refusal case names: one above the most */
#define STAGE_COPY_WORKERS 2u /* the stage case's --copy_workers=2 */
#define SUFFIX_ROOM \
	32u /* a path buffer's room for the longest receipt or state suffix */
#define DECIMAL_BASE 10
#define FILE_TIMES \
	2u /* utimensat and futimens: the access time, then the modification time */
#define FIXTURE_HEADER_DIGEST \
	1u /* the manifest's pack identity, 0000000000000001 */
#define SEARCH_DIRECTORY "/anemone_engram_rnd/replicas"
#define SEARCH_RECEIPT_TEMPORARY_SUFFIX \
	".receipt.new" /* replica_prepare.c's search receipt before its rename */
#define SEARCH_RECEIPT_TEST_BYTES (PATH_MAX + 256u)
static unsigned failures;
static void expect(bool good, const char *message)
{
	if (!good) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}
static uint64_t deadline(void)
{
	struct timespec time;
	if (clock_gettime(CLOCK_MONOTONIC, &time) != 0)
		return 0;
	return (uint64_t)time.tv_sec * NS_SECOND + (uint64_t)time.tv_nsec +
	       FIXTURE_SECONDS * NS_SECOND;
}
#if CHECKING_HOOKS_ENABLED
/* Whether the kernel sets up an io_uring here (the copier's workers then use one). */
static bool io_uring_available(void)
{
	struct io_uring_params parameters;
	memset(&parameters, 0, sizeof(parameters));
	const long ring =
		syscall(__NR_io_uring_setup, PROBE_RING_ENTRIES, &parameters);
	if (ring < 0)
		return false;
	(void)close((int)ring);
	return true;
}
#endif
#if CHECKING_HOOKS_ENABLED
/* The copy's readbacks and those inside the lag, from its report line. */
static bool copy_counts(const char *text, uint32_t *readbacks, uint32_t *inside)
{
	const char *const line = strstr(text, "storage: copy readbacks=");
	return line != NULL &&
	       sscanf(line, "storage: copy readbacks=%u inside_lag=%u",
		      readbacks, inside) == COPY_COUNT_FIELDS;
}
#endif
static bool file_matches(const char *path, const unsigned char *data)
{
	unsigned char bytes[FIXTURE_BYTES];
	const int descriptor = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
	if (descriptor < 0)
		return false;
	const bool good = read(descriptor, bytes, sizeof(bytes)) ==
				  (ssize_t)sizeof(bytes) &&
			  memcmp(bytes, data, sizeof(bytes)) == 0;
	return close(descriptor) == 0 && good;
}
static bool append_path(char *output, size_t capacity, const char *parent,
			const char *suffix)
{
	const size_t a = strlen(parent), b = strlen(suffix);
	if (a >= capacity || b >= capacity - a)
		return false;
	memcpy(output, parent, a);
	memcpy(output + a, suffix, b);
	output[a + b] = '\0';
	return true;
}
/* The earlier copier's lock beside a target: <directory>/.<name>.copy.lock. */
static bool copier_lock_path(char *output, size_t capacity, const char *target)
{
	const char *const name = strrchr(target, '/');
	const int length =
		name == NULL ? -1 :
			       snprintf(output, capacity, "%.*s/.%s.copy.lock",
					(int)(name - target), target, name + 1);
	return length > 0 && (size_t)length < capacity;
}
static void remove_target(const struct prepare_target *target)
{
	char path[PATH_MAX];
	/* the receipt and its temporary too: a case folder with one left in it cannot be removed */
	static const char *const suffixes[] = {
		"",
		".partial",
		".copy-state",
		".copy-lock",
		PREPARE_RECEIPT_SUFFIX,
		RECEIPT_TEMPORARY_SUFFIX,
		REPLICA_RECEIPT_SUFFIX,
		SEARCH_RECEIPT_TEMPORARY_SUFFIX
	};
	for (uint32_t index = 0; index < sizeof(suffixes) / sizeof(suffixes[0]);
	     index++) {
		const int length = snprintf(path, sizeof(path), "%s%s",
					    target->path, suffixes[index]);
		if (length > 0 && (size_t)length < sizeof(path))
			(void)unlink(path);
	}
	if (copier_lock_path(path, sizeof(path), target->path))
		(void)unlink(path);
}
/* A run with its stderr captured into text (NUL-terminated), so the kept lines can be checked. */
static bool run_captured(const struct prepare_copy_options *options,
			 result_status *status, char *text, size_t capacity)
{
	char name[] = "/tmp/anemone-prepare-stderr-XXXXXX";
	const int capture = mkstemp(name);
	if (capture < 0)
		return false;
	(void)unlink(name);
	(void)fflush(stderr);
	const int saved = dup(STDERR_FILENO);
	const bool redirected = saved >= 0 &&
				dup2(capture, STDERR_FILENO) == STDERR_FILENO;
	if (redirected)
		*status = prepare_replicas(options);
	(void)fflush(stderr);
	if (saved >= 0) {
		(void)dup2(saved, STDERR_FILENO);
		(void)close(saved);
	}
	const ssize_t got =
		redirected ? pread(capture, text, capacity - 1u, 0) : -1;
	(void)close(capture);
	if (got < 0)
		return false;
	text[got] = '\0';
	return true;
}
/* Another process's copy in progress holds <target>.copy-lock: this run refuses the target and leaves its partial
 * and journal as they are (they were deleted when the lock race was lost). */
static void locked_copy_untouched(const struct prepare_copy_options *options,
				  const unsigned char *data)
{
	const struct prepare_target *const target = &options->targets[0];
	char partial[PATH_MAX], journal[PATH_MAX], lock[PATH_MAX];
	const bool named =
		append_path(partial, sizeof(partial), target->path,
			    PREPARE_PARTIAL_SUFFIX) &&
		append_path(journal, sizeof(journal), target->path,
			    ".copy-state") &&
		append_path(lock, sizeof(lock), target->path, ".copy-lock");
	expect(named, "lock-race fixture paths bounded");
	if (!named)
		return;
	static const char held_journal[] = "held journal";
	bool prepared = true;
	const char *const files[] = { partial, journal };
	for (uint32_t index = 0; index < sizeof(files) / sizeof(files[0]);
	     index++) {
		const int descriptor = open(
			files[index], O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			S_IRUSR | S_IWUSR);
		const void *const bytes = index == 0 ?
						  (const void *)data :
						  (const void *)held_journal;
		const size_t count = index == 0 ? (size_t)PACK_BLOCK_BYTES :
						  sizeof(held_journal);
		prepared = prepared && descriptor >= 0 &&
			   write(descriptor, bytes, count) == (ssize_t)count;
		if (descriptor >= 0)
			(void)close(descriptor);
	}
	const int holder =
		open(lock, O_WRONLY | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR);
	prepared = prepared && holder >= 0 &&
		   flock(holder, LOCK_EX | LOCK_NB) == 0;
	expect(prepared,
	       "another process's partial, journal and lock prepared");
	struct prepare_copy_options locked = *options;
	locked.count = 1;
	locked.deadline_ns = deadline();
	struct prepare_checkpoint checkpoint;
	expect(prepared &&
		       prepare_checkpoint_classify(options->manifest,
						   target->path, target->key,
						   options->floor_percent,
						   &checkpoint) == STATUS_OK &&
		       checkpoint.kind == PREPARE_CHECKPOINT_HELD &&
		       checkpoint.refusal != NULL &&
		       strcmp(checkpoint.refusal, "copy_in_progress") == 0,
	       "the plan names a target whose copy-lock another copy holds held, not examined");
	static char text[CAPTURE_BYTES];
	result_status refused = STATUS_OK;
	expect(run_captured(&locked, &refused, text, sizeof(text)),
	       "lock-race stderr captured");
	struct stat kept_partial, kept_journal;
	expect(prepared && !target->complete &&
		       stat(partial, &kept_partial) == 0 &&
		       kept_partial.st_size == PACK_BLOCK_BYTES &&
		       stat(journal, &kept_journal) == 0 &&
		       kept_journal.st_size == (off_t)sizeof(held_journal),
	       "losing the copy-lock race leaves the holder's partial and journal");
	expect(target->reason != NULL &&
		       strcmp(target->reason, "copy_in_progress") == 0 &&
		       strstr(text, "kept (") != NULL &&
		       strstr(text,
			      "copy_in_progress; not examined: another copy holds its .copy-lock") !=
			       NULL &&
		       strstr(text, "no matching engine journal") == NULL &&
		       strstr(text, "remove it") == NULL,
	       "the kept line names the copy in progress and advises no removal, since nothing was examined");
	if (holder >= 0)
		(void)close(holder);
	remove_target(target);
}
/* The bytes a captured run's copy stage line reports for stage (summed over its workers); UINT64_MAX without it. */
static uint64_t stage_bytes(const char *text, const char *stage)
{
	char prefix[FIELD_TEXT_BYTES];
	const int length =
		snprintf(prefix, sizeof(prefix),
			 "storage: copy stage %s busy_seconds=", stage);
	const char *const line = length > 0 && (size_t)length < sizeof(prefix) ?
					 strstr(text, prefix) :
					 NULL;
	const char *const field = line != NULL ? strstr(line, " bytes=") : NULL;
	return field != NULL ?
		       strtoull(field + strlen(" bytes="), NULL, DECIMAL_BASE) :
		       UINT64_MAX;
}
/* An explicit destination below an automount that has not fired is refused there, never described as the disk of an
 * ancestor (whose directories autotune would create on the share once it mounts); a missing one climbs to its disk. */
static void destination_cases(void)
{
	char root[] = "/tmp/anemone-destination-XXXXXX";
	if (mkdtemp(root) == NULL) {
		expect(false, "destination fixture opens");
		return;
	}
	char table[PATH_MAX], below[PATH_MAX], plain[PATH_MAX], nas[PATH_MAX],
		lines[FIXTURE_TABLE_BYTES];
	(void)snprintf(table, sizeof(table), "%s/mounts", root);
	(void)snprintf(below, sizeof(below), "%s/nas/anemone_engram_rnd", root);
	(void)snprintf(plain, sizeof(plain), "%s/plain/anemone_engram_rnd",
		       root);
	(void)snprintf(nas, sizeof(nas), "%s/nas", root);
	(void)snprintf(
		lines, sizeof(lines),
		"/dev/fixture %s ext4 rw 0 0\nsystemd-1 %s autofs rw 0 0\n",
		root, nas);
	bool fixture = false;
	FILE *const mounts = fopen(table, "we");
	if (mounts != NULL) {
		fixture = fputs(lines, mounts) >= 0;
		if (fclose(mounts) != 0)
			fixture = false;
	}
	if (fixture)
		fixture = mkdir(nas, S_IRWXU) == 0;
	struct local_device_description idle = { 0 }, reachable = { 0 };
	result_status idle_status = STATUS_INVALID_ARGUMENT,
		      reachable_status = STATUS_INVALID_ARGUMENT;
	if (fixture) {
		idle_status = prepare_destination_device(table, below, &idle);
		reachable_status =
			prepare_destination_device(table, plain, &reachable);
	}
	expect(fixture && idle_status != STATUS_OK &&
		       strcmp(idle.filesystem, "autofs") == 0,
	       "a destination below an idle automount is refused there, before any ancestor is described");
	expect(fixture && reachable_status == STATUS_OK,
	       "a destination that does not exist yet climbs to its disk");
	(void)rmdir(nas);
	(void)unlink(table);
	(void)rmdir(root);
}

/* The source bytes with the first byte of each chunk in the corrupt mask (bit k: chunk k) changed. */
static void corrupted(unsigned char bytes[FIXTURE_BYTES],
		      const unsigned char *data, uint32_t corrupt)
{
	memcpy(bytes, data, (size_t)FIXTURE_BYTES);
	for (uint32_t chunk = 0; chunk < FIXTURE_BYTES / FIXTURE_CHUNK; chunk++)
		if ((corrupt & (1u << chunk)) != 0)
			bytes[(size_t)(chunk * FIXTURE_CHUNK)] ^= 1u;
}
/* A foreign partial (no engine journal) at path: the source bytes with the corrupt chunks changed, written now or an
 * hour back. Its state after writing into written. */
static bool foreign_partial(const char *path, const unsigned char *data,
			    uint32_t corrupt, bool aged, struct stat *written)
{
	unsigned char bytes[FIXTURE_BYTES];
	corrupted(bytes, data, corrupt);
	const int descriptor = open(
		path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC,
		S_IRUSR | S_IWUSR);
	struct timespec now = { 0 };
	bool good = descriptor >= 0 &&
		    write(descriptor, bytes, sizeof(bytes)) ==
			    (ssize_t)sizeof(bytes) &&
		    fsync(descriptor) == 0 &&
		    clock_gettime(CLOCK_REALTIME, &now) == 0;
	const struct timespec times[FILE_TIMES] = {
		{ .tv_sec = now.tv_sec - AGED_SECONDS },
		{ .tv_sec = now.tv_sec - AGED_SECONDS }
	};
	if (good && aged)
		good = futimens(descriptor, times) == 0;
	good = good && fstat(descriptor, written) == 0;
	if (descriptor >= 0)
		good = close(descriptor) == 0 && good;
	return good;
}
/* Whether the foreign partial is byte for byte and time for time as foreign_partial wrote it, with no journal. */
static bool foreign_untouched(const char *path, const char *journal,
			      const unsigned char *data, uint32_t corrupt,
			      const struct stat *written)
{
	unsigned char bytes[FIXTURE_BYTES];
	corrupted(bytes, data, corrupt);
	struct stat now;
	return file_matches(path, bytes) && stat(path, &now) == 0 &&
	       now.st_ino == written->st_ino &&
	       now.st_mtim.tv_sec == written->st_mtim.tv_sec &&
	       now.st_mtim.tv_nsec == written->st_mtim.tv_nsec &&
	       access(journal, F_OK) != 0;
}
/* An engine checkpoint at the target: a run whose deadline is past opens it, journals it and stops. */
static bool engine_checkpoint(const struct prepare_copy_options *options,
			      const char *partial, const char *journal)
{
	struct prepare_copy_options stopped = *options;
	stopped.deadline_ns = PAST_DEADLINE_NS;
	return prepare_replicas(&stopped) == STATUS_TIMED_OUT &&
	       access(partial, F_OK) == 0 && access(journal, F_OK) == 0;
}
static void pure_cases(void)
{
	expect(prepare_available_with_credit(CREDIT_TOTAL, CREDIT_AVAILABLE_LOW,
					     CREDIT_SMALL) ==
			       CREDIT_AVAILABLE_LOW + CREDIT_SMALL &&
		       prepare_available_with_credit(
			       CREDIT_TOTAL, CREDIT_AVAILABLE_HIGH,
			       CREDIT_LARGE) == CREDIT_TOTAL &&
		       prepare_available_with_credit(CREDIT_TOTAL,
						     CREDIT_AVAILABLE_OVER,
						     0u) == CREDIT_TOTAL,
	       "a checkpoint's credit adds to the available bytes, at most the total");
	uint32_t alignment = 1;
	expect(prepare_direct_alignment(-1, &alignment) ==
			       STATUS_INVALID_ARGUMENT &&
		       alignment == 1,
	       "invalid alignment output remains untouched");
	char hash[STORAGE_PROFILE_KEY_BYTES];
	expect(prepare_hash("abc", strlen("abc"), hash) == STATUS_OK &&
		       strcmp(hash,
			      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") ==
			       0,
	       "kernel SHA256 known vector");
	char *text = NULL;
	size_t bytes = 0;
	FILE *const stream = open_memstream(&text, &bytes);
	expect(stream != NULL, "startup hint capture");
	if (stream != NULL) {
		storage_autotune_hint(stream);
		const bool closed = fclose(stream) == 0;
		expect(closed && text != NULL &&
			       strcmp(text,
				      "storage: no tuned storage profile: run `anemone autotune` before serving for "
				      "best load/lookup performance\n") == 0,
		       "startup hint exact product text in every build");
	}
	free(text);
	/* A contiguous 481 GiB replica on ext4 maps as 128 MiB extents: one fragment, not fragmented. */
	struct prepare_extent_map contiguous = { 0 }, scattered = { 0 };
	for (uint64_t extent = 0; extent < LARGE_REPLICA_EXTENTS; extent++)
		prepare_extent_record(&contiguous, extent * EXT4_EXTENT_BYTES,
				      PHYSICAL_START +
					      extent * EXT4_EXTENT_BYTES,
				      EXT4_EXTENT_BYTES, 0);
	expect(contiguous.count == LARGE_REPLICA_EXTENTS &&
		       !prepare_extents_fragmented(&contiguous,
						   LARGE_REPLICA_EXTENTS *
							   EXT4_EXTENT_BYTES),
	       "contiguous capped extents form one fragment and never count as fragmented");
	for (uint64_t extent = 0; extent < SCATTERED_EXTENTS; extent++)
		prepare_extent_record(&scattered, extent * SCATTERED_BYTES,
				      PHYSICAL_START +
					      extent * SCATTERED_STRIDE *
						      SCATTERED_BYTES,
				      SCATTERED_BYTES, 0);
	expect(prepare_extents_fragmented(&scattered,
					  SCATTERED_EXTENTS * SCATTERED_BYTES),
	       "physically scattered extents still count as fragmented");
	/* A raw mount point that is no UTF-8 (the kernel writes bytes >= 0x80 unescaped) keeps the decision JSON a
	   readable document: the profile is never unreadable for one mount. */
	struct prepare_target raw = { .reason =
					      "destination_directory_policy" };
	(void)snprintf(raw.path, sizeof(raw.path), "%s",
		       "/mnt/Donn\xe9"
		       "es/anemone_engram_rnd/replicas/x.anemone");
	(void)snprintf(raw.device.mount, sizeof(raw.device.mount), "%s",
		       "/mnt/Donn\xe9"
		       "es");
	text = NULL;
	bytes = 0;
	FILE *const decision = open_memstream(&text, &bytes);
	expect(decision != NULL, "decision capture");
	if (decision != NULL) {
		prepare_decision_json(decision, &raw);
		struct json_value value;
		expect(fclose(decision) == 0 &&
			       json_read_document(text, bytes, &value) ==
				       JSON_DOCUMENT_VALID &&
			       strstr(text, "Donn\\ufffdes") != NULL,
		       "a decision with a non-UTF-8 path and mount is still a valid JSON document");
	}
	free(text);
	static const char partial_tail[] = ".partial";
	const size_t replicate_bytes = strlen(REPLICA_PARTIAL_SUFFIX);
	expect(strcmp(REPLICA_PARTIAL_SUFFIX, PREPARE_PARTIAL_SUFFIX) != 0 &&
		       replicate_bytes > strlen(partial_tail) &&
		       strcmp(REPLICA_PARTIAL_SUFFIX + replicate_bytes -
				      strlen(partial_tail),
			      partial_tail) == 0,
	       "anemone replicate never names autotune's resumable checkpoint, and its copy stays a .partial");
}
/* path with suffix appended into output; false when it does not fit. */
static bool suffixed(char *output, size_t capacity, const char *path,
		     const char *suffix)
{
	const int length = snprintf(output, capacity, "%s%s", path, suffix);
	return length > 0 && (size_t)length < capacity;
}
/* The autotune command with its stderr captured: whether it exits with expected_status and stderr holds expected. */
static bool autotune_says(int count, char **arguments, int expected_status,
			  const char *expected)
{
	FILE *const captured = tmpfile();
	const int saved = dup(STDERR_FILENO);
	if (captured == NULL || saved < 0 ||
	    dup2(fileno(captured), STDERR_FILENO) != STDERR_FILENO) {
		if (captured != NULL)
			(void)fclose(captured);
		if (saved >= 0)
			(void)close(saved);
		return false;
	}
	const int status = storage_autotune_cli(count, arguments);
	(void)fflush(stderr);
	const bool restored = dup2(saved, STDERR_FILENO) == STDERR_FILENO;
	(void)close(saved);
	char text[AUTOTUNE_CAPTURE_BYTES];
	const bool rewound = fseek(captured, 0L, SEEK_SET) == 0;
	clearerr(captured);
	const size_t bytes = fread(text, 1, sizeof(text) - 1u, captured);
	text[bytes] = '\0';
	(void)fclose(captured);
	return restored && rewound && status == expected_status &&
	       strstr(text, expected) != NULL;
}
/* The autotune command with its stderr captured whole into text through the file at capture_path (made O_EXCL and
 * removed after): its exit status, or -1 when the capture fails. */
static int autotune_captured(int count, char **arguments,
			     const char *capture_path, char *text,
			     size_t capacity)
{
	text[0] = '\0';
	(void)fflush(stderr);
	const int capture = open(capture_path,
				 O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
				 S_IRUSR | S_IWUSR);
	const int saved = dup(STDERR_FILENO);
	int status = -1;
	if (capture >= 0 && saved >= 0 &&
	    dup2(capture, STDERR_FILENO) == STDERR_FILENO) {
		status = storage_autotune_cli(count, arguments);
		(void)fflush(stderr);
		if (dup2(saved, STDERR_FILENO) != STDERR_FILENO ||
		    test_log_capture_read(capture, text, capacity) < 0)
			status = -1;
	}
	if (saved >= 0)
		(void)close(saved);
	if (capture >= 0) {
		(void)close(capture);
		(void)unlink(capture_path);
	}
	return status;
}
/* r18 (request 33 L01): every autotune refusal names its reason, and --plan checks the profile's folder as the
 * run does, so a folder another user or group may write refuses in both, never silently. */
static void autotune_refusal_cases(const char *source, const char *directory)
{
	char shared[PATH_MAX], profile[PATH_MAX], tokens[PATH_MAX];
	char profile_option[PATH_MAX + sizeof("--profile=")],
		tokens_option[PATH_MAX + sizeof("--tokens=")];
	const bool named =
		snprintf(shared, sizeof(shared), "%s/shared-profile",
			 directory) < (int)sizeof(shared) &&
		snprintf(profile, sizeof(profile), "%s/storage-profile.json",
			 shared) < (int)sizeof(profile) &&
		snprintf(tokens, sizeof(tokens), "%s/tokens.u32", directory) <
			(int)sizeof(tokens) &&
		snprintf(profile_option, sizeof(profile_option), "--profile=%s",
			 profile) < (int)sizeof(profile_option) &&
		snprintf(tokens_option, sizeof(tokens_option), "--tokens=%s",
			 tokens) < (int)sizeof(tokens_option);
	const bool shared_made = named && mkdir(shared, S_IRWXU) == 0 &&
				 chmod(shared, S_IRWXU | S_IRWXG) == 0;
	char program[] = "anemone", command[] = "autotune", plan[] = "--plan";
	char source_word[PATH_MAX];
	const bool source_named =
		suffixed(source_word, sizeof(source_word), source, "");
	char *arguments[] = { program,	      command,	     source_word,
			      profile_option, tokens_option, plan };
	const int with_plan = (int)(sizeof(arguments) / sizeof(arguments[0]));
	expect(shared_made && source_named &&
		       autotune_says(
			       with_plan, arguments, EXIT_FAILURE,
			       "autotune: refused: the profile's folder is not this user's alone") &&
		       autotune_says(
			       with_plan - 1, arguments, EXIT_FAILURE,
			       "autotune: refused: the profile's folder is not this user's alone"),
	       "a profile folder a group may write refuses in the plan and the run, with its reason");
	expect(shared_made && chmod(shared, S_IRWXU) == 0 &&
		       autotune_says(
			       with_plan - 1, arguments, EXIT_FAILURE,
			       "autotune: refused: the pack does not open"),
	       "a later refusal (the fixture source is no pack) names its reason too");
	(void)rmdir(shared);
}
/* The owner's split of autotune into stages: --stage takes copy, the copy stage refuses the lookup tuning's token
 * files, and without a stage the tuning still needs them; the copy stage's --copy_workers takes 1 to
 * PREPARE_COPY_WORKERS_MOST. Each is refused before any file is opened. */
static void stage_option_cases(void)
{
	char program[] = "anemone", command[] = "autotune",
	     source[] = "/nonexistent/stage-case.anemone",
	     copy[] = "--stage=copy", unknown[] = "--stage=tune",
	     tokens[] = "--tokens=/nonexistent/stage-case.u32",
	     no_workers[] = "--copy_workers=0",
	     too_many[] = "--copy_workers=33";
	char *with_unknown[] = { program, command, source, unknown };
	char *without_workers[] = { program, command, source, copy,
				    no_workers };
	char *above_most[] = { program, command, source, copy, too_many };
	char *copy_with_tokens[] = { program, command, source, copy, tokens };
	char *no_stage_no_tokens[] = { program, command, source };
	expect(autotune_says(
		       (int)(sizeof(with_unknown) / sizeof(with_unknown[0])),
		       with_unknown, EXIT_FAILURE,
		       "autotune: refused: no such stage (copy and load so far): --stage=tune"),
	       "a stage this build has not is refused by name");
	expect(autotune_says(
		       (int)(sizeof(copy_with_tokens) /
			     sizeof(copy_with_tokens[0])),
		       copy_with_tokens, EXIT_FAILURE,
		       "autotune: refused: --tokens and --prompt are the lookup tuning's, not the copy stage's"),
	       "the copy stage refuses the lookup tuning's token files");
	expect(autotune_says((int)(sizeof(no_stage_no_tokens) /
				   sizeof(no_stage_no_tokens[0])),
			     no_stage_no_tokens, EXIT_FAILURE,
			     "usage: anemone autotune PACK --tokens="),
	       "without a stage the tuning still needs its token files");
	expect(PREPARE_COPY_WORKERS_MOST + 1u == STAGE_WORKERS_ABOVE_MOST &&
		       autotune_says(
			       (int)(sizeof(without_workers) /
				     sizeof(without_workers[0])),
			       without_workers, EXIT_FAILURE,
			       "autotune: refused: --copy_workers takes 1 to 32 workers: --copy_workers=0") &&
		       autotune_says(
			       (int)(sizeof(above_most) / sizeof(above_most[0])),
			       above_most, EXIT_FAILURE,
			       "autotune: refused: --copy_workers takes 1 to 32 workers: --copy_workers=33"),
	       "the copy stage refuses no workers and more than the most, by name");
}
/* Whether the file at path holds exactly the bytes bytes of expected. */
static bool file_holds(const char *path, const unsigned char *expected,
		       uint64_t bytes)
{
	struct stat file;
	if (stat(path, &file) != 0 || (uint64_t)file.st_size != bytes)
		return false;
	unsigned char *const found = malloc((size_t)bytes);
	FILE *const input = fopen(path, "re");
	const bool same = found != NULL && input != NULL &&
			  fread(found, 1, (size_t)bytes, input) ==
				  (size_t)bytes &&
			  memcmp(found, expected, (size_t)bytes) == 0;
	if (input != NULL)
		(void)fclose(input);
	free(found);
	return same;
}
/* The file at path read into text (at most capacity - 1 bytes, terminated): its bytes, 0 when it does not open. */
static size_t file_text(const char *path, char *text, size_t capacity)
{
	FILE *const input = fopen(path, "re");
	const size_t bytes =
		input != NULL ? fread(text, 1, capacity - 1u, input) : 0;
	text[bytes] = '\0';
	if (input != NULL)
		(void)fclose(input);
	return bytes;
}
/* The source buffers a captured copy run keeps (its readbacks line); UINT32_MAX without the line. */
static uint32_t retained_buffers(const char *text)
{
	const char *const field = strstr(text, " retained_buffers=");
	return field != NULL ?
		       (uint32_t)strtoul(field + strlen(" retained_buffers="),
					 NULL, DECIMAL_BASE) :
		       UINT32_MAX;
}
/* --stage=copy end to end: a real pack (the n-gram test pack) beside the source copied to the first device and
 * verified, with its verification and search receipts, and nothing of a tuning written: no tune state, and the load
 * stage's profile in place (the documented copy -> load -> copy flow on one --profile) byte-identical after the stage
 * and after its --plan. The copy's decisions record it complete. The default pairs the source hashes and
 * --copy_hash_pairs=0 does not: a fresh copy at 0 keeps one source buffer a worker fewer (the pending chunk) and
 * writes the same manifest. */
static void stage_copy_case(const char *source_directory,
			    const char *target_directory)
{
	unsigned char *pack = NULL;
	struct engram_record *engram = NULL;
	struct pack_header header = { 0 };
	struct prepare_target target = { 0 };
	/* the paths joined with append_path, and the suffixed ones with room for their suffix: gcc 14's
	   format-truncation check reads a formatted path copy as long as its source array */
	char source[PATH_MAX], folder[PATH_MAX], profile[PATH_MAX],
		leaf[STAGE_LEAF_BYTES], tune_state[PATH_MAX + SUFFIX_ROOM],
		decisions[PATH_MAX + SUFFIX_ROOM],
		manifest[PATH_MAX + SUFFIX_ROOM],
		verified[PATH_MAX + SUFFIX_ROOM],
		receipt[PATH_MAX + SUFFIX_ROOM], capture[PATH_MAX],
		stage_profile[PATH_MAX + SUFFIX_ROOM];
	char target_option[PATH_MAX + sizeof("--target=")],
		profile_option[PATH_MAX + sizeof("--profile=")];
	bool good = test_engram_pack_open(&pack, &engram) == STATUS_OK;
	if (good)
		memcpy(&header, pack, sizeof(header));
	const int leaf_length = snprintf(leaf, sizeof(leaf),
					 "/%016" PRIx64 ".anemone",
					 header.header_digest);
	good = good && leaf_length > 0 && (size_t)leaf_length < sizeof(leaf) &&
	       append_path(target.path, sizeof(target.path), target_directory,
			   leaf) &&
	       append_path(source, sizeof(source), source_directory,
			   "/stage-copy.anemone") &&
	       append_path(folder, sizeof(folder), source_directory,
			   "/stage-profile") &&
	       append_path(capture, sizeof(capture), source_directory,
			   "/stage-stderr") &&
	       append_path(profile, sizeof(profile), folder,
			   "/storage-profile.json") &&
	       suffixed(tune_state, sizeof(tune_state), profile,
			".tune-state") &&
	       suffixed(decisions, sizeof(decisions), profile, ".copy.json") &&
	       suffixed(manifest, sizeof(manifest), profile,
			".replicas.json") &&
	       suffixed(verified, sizeof(verified), target.path,
			PREPARE_RECEIPT_SUFFIX) &&
	       suffixed(receipt, sizeof(receipt), target.path,
			REPLICA_RECEIPT_SUFFIX) &&
	       suffixed(target_option, sizeof(target_option),
			"--target=", target.path) &&
	       suffixed(profile_option, sizeof(profile_option),
			"--profile=", profile);
	const int written = good ? open(source,
					O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
					S_IRUSR | S_IWUSR) :
				   -1;
	good = good && written >= 0 &&
	       write(written, pack, (size_t)header.file_byte_count) ==
		       (ssize_t)header.file_byte_count &&
	       fsync(written) == 0;
	if (written >= 0)
		(void)close(written);
	const int stage_profile_length =
		snprintf(stage_profile, sizeof(stage_profile),
			 "{\"schema\":%u,\"source\":\"%s\"}\n",
			 STORAGE_PROFILE_STAGES_SCHEMA, source);
	const int profile_written =
		good && stage_profile_length > 0 &&
				(size_t)stage_profile_length <
					sizeof(stage_profile) &&
				mkdir(folder, S_IRWXU) == 0 ?
			open(profile, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			     S_IRUSR | S_IWUSR) :
			-1;
	good = good && profile_written >= 0 &&
	       write(profile_written, stage_profile,
		     (size_t)stage_profile_length) ==
		       (ssize_t)stage_profile_length &&
	       fsync(profile_written) == 0;
	if (profile_written >= 0)
		(void)close(profile_written);
	/* no floor: the case is about the stage, and a scratch device may be fuller than the default floor */
	char program[] = "anemone", command[] = "autotune",
	     stage[] = "--stage=copy", seconds[] = "--seconds=60",
	     floor_option[] = "--replica_floor_percent=0",
	     workers_option[] = "--copy_workers=2";
	char *arguments[] = { program, command,	      source,
			      stage,   target_option, profile_option,
			      seconds, floor_option,  workers_option };
	char said[CAPTURE_BYTES], plan_option[] = "--plan";
	char *planned[] = { program,	command,       source,
			    stage,	target_option, profile_option,
			    seconds,	floor_option,  workers_option,
			    plan_option };
	const int plan_exited =
		good ? autotune_captured(
			       (int)(sizeof(planned) / sizeof(planned[0])),
			       planned, capture, said, sizeof(said)) :
		       -1;
	expect(plan_exited == EXIT_SUCCESS &&
		       file_holds(profile, (const unsigned char *)stage_profile,
				  (uint64_t)stage_profile_length),
	       "the copy stage's --plan accepts a load stage profile in place and leaves it byte-identical");
	const int exited = good ? autotune_captured((int)(sizeof(arguments) /
							  sizeof(arguments[0])),
						    arguments, capture, said,
						    sizeof(said)) :
				  -1;
	expect(exited == EXIT_SUCCESS &&
		       strstr(said, "storage: autotune stage=copy status=") !=
			       NULL,
	       "the copy stage copies a real pack and exits as done");
	expect(exited == EXIT_SUCCESS &&
		       strstr(said, "storage: copy worker 1 mechanism=") !=
			       NULL &&
		       strstr(said, "storage: copy worker 2 mechanism=") ==
			       NULL,
	       "the copy stage runs the --copy_workers it was given: two workers");
	char text[CAPTURE_BYTES];
	const size_t record_bytes = file_text(decisions, text, sizeof(text));
	expect(good && file_holds(target.path, pack, header.file_byte_count) &&
		       access(verified, F_OK) == 0 &&
		       access(receipt, F_OK) == 0,
	       "the copy stage's replica holds the pack's bytes, with its verification and search receipts");
	expect(record_bytes != 0 && strstr(text, "\"complete\":1") != NULL &&
		       strstr(text, "\"copy_workers\":2") != NULL &&
		       file_holds(profile, (const unsigned char *)stage_profile,
				  (uint64_t)stage_profile_length) &&
		       access(tune_state, F_OK) != 0,
	       "the copy stage records its decisions complete, its two workers, no tune state, and leaves the load "
	       "stage profile byte-identical");
	char paired_manifest[CAPTURE_BYTES], plain_manifest[CAPTURE_BYTES],
		plain_option[] = "--copy_hash_pairs=0";
	const size_t paired_bytes =
		file_text(manifest, paired_manifest, sizeof(paired_manifest));
	const uint32_t paired = retained_buffers(said);
	remove_target(&target);
	(void)unlink(decisions);
	(void)unlink(manifest);
	char *plain[] = { program,	  command,	  source,  stage,
			  target_option,  profile_option, seconds, floor_option,
			  workers_option, plain_option };
	const int plain_exited =
		good ? autotune_captured(
			       (int)(sizeof(plain) / sizeof(plain[0])), plain,
			       capture, said, sizeof(said)) :
		       -1;
	const uint32_t unpaired = retained_buffers(said);
	const size_t plain_bytes =
		file_text(manifest, plain_manifest, sizeof(plain_manifest));
	expect(exited == EXIT_SUCCESS && plain_exited == EXIT_SUCCESS &&
		       unpaired != UINT32_MAX &&
		       paired == unpaired + STAGE_COPY_WORKERS &&
		       file_holds(target.path, pack, header.file_byte_count) &&
		       paired_bytes != 0 && plain_bytes == paired_bytes &&
		       memcmp(plain_manifest, paired_manifest, paired_bytes) ==
			       0,
	       "the copy stage pairs its source hashes by default, one pending buffer a worker, and not at "
	       "--copy_hash_pairs=0, with the same manifest");
	remove_target(&target);
	(void)unlink(decisions);
	(void)unlink(manifest);
	(void)unlink(profile);
	(void)rmdir(folder);
	(void)unlink(source);
	engram_close(engram);
	free(pack);
}

/* The receipt as the engine writes it, refreshed to the file's current stat: what a device's silent corruption
 * looks like (bytes changed, the receipt's view of the file intact). */
static bool receipt_refresh(const char *path, const char *receipt)
{
	char text[RECEIPT_TEST_BYTES], key[RECEIPT_TEST_BYTES],
		manifest[RECEIPT_TEST_BYTES], pack[RECEIPT_TEST_BYTES];
	unsigned schema = 0, chunks = 0;
	unsigned long long inode = 0, size = 0;
	long long mtime = 0, ctime = 0;
	FILE *const input = fopen(receipt, "r");
	const bool read = input != NULL &&
			  fgets(text, sizeof(text), input) != NULL;
	if (input != NULL)
		(void)fclose(input);
	struct stat file;
	if (!read || stat(path, &file) != 0 ||
	    sscanf(text,
		   "anemone-verified %u key=%127s inode=%llu size=%llu mtime_ns=%lld ctime_ns=%lld manifest=%127s "
		   "pack=%127s chunks=%u",
		   &schema, key, &inode, &size, &mtime, &ctime, manifest, pack,
		   &chunks) != RECEIPT_TEST_FIELDS)
		return false;
	FILE *const output = fopen(receipt, "w");
	const bool written =
		output != NULL &&
		fprintf(output,
			"anemone-verified %u key=%s inode=%" PRIu64
			" size=%" PRIu64 " mtime_ns=%" PRId64
			" ctime_ns=%" PRId64 " manifest=%s pack=%s chunks=%u\n",
			schema, key, (uint64_t)file.st_ino,
			(uint64_t)file.st_size,
			(int64_t)file.st_mtim.tv_sec * (int64_t)NS_SECOND +
				(int64_t)file.st_mtim.tv_nsec,
			(int64_t)file.st_ctim.tv_sec * (int64_t)NS_SECOND +
				(int64_t)file.st_ctim.tv_nsec,
			manifest, pack, chunks) > 0;
	return output != NULL && fclose(output) == 0 && written;
}
/* Whether the receipt names the device by its hardware key and never by its device number (a reboot may renumber). */
static bool receipt_names_key(const char *receipt, const char *key)
{
	char text[RECEIPT_TEST_BYTES], expected[RECEIPT_TEST_BYTES];
	FILE *const input = fopen(receipt, "r");
	const bool read = input != NULL &&
			  fgets(text, sizeof(text), input) != NULL;
	if (input != NULL)
		(void)fclose(input);
	const int length =
		snprintf(expected, sizeof(expected), " key=%s ", key);
	return read && length > 0 && (size_t)length < sizeof(expected) &&
	       strstr(text, expected) != NULL &&
	       strstr(text, "device=") == NULL;
}
/* Whether target's decision record holds field (a quoted JSON name and its colon) with value. */
static bool decision_says(const struct prepare_target *target,
			  const char *field, uint64_t value)
{
	char *text = NULL;
	size_t size = 0;
	FILE *const stream = open_memstream(&text, &size);
	if (stream == NULL)
		return false;
	prepare_decision_json(stream, target);
	const bool closed = fclose(stream) == 0;
	char expected[FIELD_TEXT_BYTES];
	const int length = snprintf(expected, sizeof(expected),
				    "%s%" PRIu64 ",", field, value);
	const bool found = closed && text != NULL && length > 0 &&
			   (size_t)length < sizeof(expected) &&
			   strstr(text, expected) != NULL;
	free(text);
	return found;
}
/* A spot seed whose one chunk is chunk, among chunks. */
static uint64_t seed_spotting(uint32_t chunk, uint32_t chunks)
{
	for (uint64_t seed = 1;; seed++) {
		uint32_t spot[PREPARE_SPOT_CHUNKS];
		if (prepare_spot_chunks(seed, chunks, 1, spot) == 1 &&
		    spot[0] == chunk)
			return seed;
	}
}
static bool byte_flip(const char *path, uint64_t offset)
{
	const int descriptor = open(path, O_RDWR | O_CLOEXEC);
	unsigned char byte = 0;
	bool flipped = descriptor >= 0 &&
		       pread(descriptor, &byte, 1, (off_t)offset) == 1;
	byte = (unsigned char)(byte ^ 1u);
	flipped = flipped && pwrite(descriptor, &byte, 1, (off_t)offset) == 1 &&
		  fsync(descriptor) == 0;
	if (descriptor >= 0)
		(void)close(descriptor);
	return flipped;
}
/* One hex digit of text replaced by another (a manifest hash or a pack identity edited to another valid value). */
static void hex_edit(char *text)
{
	text[0] = text[0] == '0' ? '1' : '0';
}
/* r18 review: what binds a receipt (the hardware key, the pack, the manifest), and a refusal unrelated to the final's
 * bytes keeps it. The final is spot-checked on chunk 0 alone, so only a full read sees the last chunk. */
static void receipt_binding_cases(struct prepare_copy_options *options,
				  const char *receipt)
{
	struct prepare_target *const target = &options->targets[0];
	const uint64_t bytes = options->manifest->bytes,
		       chunk_bytes = options->manifest->chunk_bytes;
	struct prepare_manifest *const manifest = options->manifest;
	expect(receipt_names_key(receipt, target->key),
	       "the receipt names the device by its hardware key, not its device number");
	expect(decision_says(target, "\"verified_bytes\":", chunk_bytes) &&
		       decision_says(target, "\"receipt_trusted_bytes\":",
				     bytes - chunk_bytes),
	       "the decision record gives a spot-checked final's receipt-trusted bytes beside its read ones");
	char saved[STORAGE_PROFILE_KEY_BYTES];
	memcpy(saved, target->key, sizeof(saved));
	hex_edit(target->key);
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && !target->complete &&
		       target->reason != NULL &&
		       strcmp(target->reason,
			      "hardware_identity_mismatch_retune_required") ==
			       0 &&
		       access(receipt, F_OK) == 0,
	       "a refusal unrelated to the final's bytes (its identity) keeps its receipt");
	memcpy(target->key, saved, sizeof(saved));
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       target->trusted == bytes - chunk_bytes,
	       "with its identity back the final is spot-checked on the receipt it kept");
	memcpy(saved, manifest->pack_identity, sizeof(saved));
	hex_edit(manifest->pack_identity);
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       target->trusted == 0 && target->verified == bytes,
	       "another pack identity forces a full verification");
	memcpy(manifest->pack_identity, saved, sizeof(saved));
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       target->trusted == 0 && target->verified == bytes,
	       "the receipt written for that pack does not vouch for this one");
	const uint32_t edited = manifest->chunks - 1u;
	memcpy(saved, manifest->hashes[edited], sizeof(saved));
	hex_edit(manifest->hashes[edited]);
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && !target->complete &&
		       target->trusted == 0 && target->reason != NULL &&
		       strcmp(target->reason, "finalized_replica_unverified") ==
			       0 &&
		       access(receipt, F_OK) != 0,
	       "an edited manifest hash forces a full read, which finds the final unlike it (a spot check alone would "
	       "pass it)");
	memcpy(manifest->hashes[edited], saved, sizeof(saved));
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       target->verified == bytes && access(receipt, F_OK) == 0,
	       "with the manifest restored the final verifies in full and gets its receipt again");
}
#if CHECKING_HOOKS_ENABLED
/* r18 review (checking builds): a receipt that cannot be completed leaves no temporary and says so; it binds the
 * state its verification started from, so a final that changes during the run is refused and loses it; a failed spot
 * check removes it at once, durably or with a line that says it may not be. Starts with the final intact and no
 * receipt; the byte at corrupted is the one the cases flip. */
static void receipt_fault_cases(struct prepare_copy_options *options,
				const char *receipt, uint64_t corrupted,
				uint32_t corrupted_chunk)
{
	struct prepare_target *const target = &options->targets[0];
	const uint64_t bytes = options->manifest->bytes;
	char temporary[PATH_MAX];
	static char text[RUN_CAPTURE_BYTES];
	result_status status = STATUS_INVALID_ARGUMENT;
	expect(suffixed(temporary, sizeof(temporary), target->path,
			RECEIPT_TEMPORARY_SUFFIX),
	       "receipt temporary path bounded");
	options->deadline_ns = deadline();
	expect(fault_arm("prepare_receipt_write", 1) &&
		       run_captured(options, &status, text, sizeof(text)) &&
		       fault_disarm() == 1 && status == STATUS_OK &&
		       target->complete && access(receipt, F_OK) != 0 &&
		       access(temporary, F_OK) != 0 &&
		       strstr(text, "verification receipt not written") != NULL,
	       "a receipt that cannot be written leaves no temporary and is reported not written");
	options->deadline_ns = deadline();
	expect(fault_arm("prepare_receipt_directory_sync", 1) &&
		       run_captured(options, &status, text, sizeof(text)) &&
		       fault_disarm() == 1 && status == STATUS_OK &&
		       target->complete && access(receipt, F_OK) == 0 &&
		       strstr(text, "written, but its directory sync failed") !=
			       NULL,
	       "a receipt renamed into place whose directory sync fails is reported written but unsynced");
	options->deadline_ns = deadline();
	expect(fault_arm("prepare_copy_target_changed", 1) &&
		       prepare_replicas(options) == STATUS_OK &&
		       fault_disarm() == 1 && !target->complete &&
		       target->reason != NULL &&
		       strcmp(target->reason, "finalized_replica_unverified") ==
			       0 &&
		       access(receipt, F_OK) != 0,
	       "a receipt-trusted final that changes during the run is refused and loses its receipt");
	options->verify_full = true;
	options->deadline_ns = deadline();
	expect(fault_arm("prepare_copy_target_changed", 1) &&
		       prepare_replicas(options) == STATUS_OK &&
		       fault_disarm() == 1 && !target->complete &&
		       target->reason != NULL &&
		       strcmp(target->reason, "finalized_replica_unverified") ==
			       0 &&
		       access(receipt, F_OK) != 0,
	       "a final read in full that changes before the run ends gets no receipt of its later state");
	options->verify_full = false;
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       target->verified == bytes && access(receipt, F_OK) == 0,
	       "unchanged since, the final verifies in full and gets its receipt");
	/* a spot check that hits the corrupted chunk: the receipt goes before the run ends */
	expect(byte_flip(target->path, corrupted) &&
		       receipt_refresh(target->path, receipt),
	       "corruption under an intact receipt prepared for a killed run");
	options->spot_chunks = 1;
	options->spot_seed =
		seed_spotting(corrupted_chunk, options->manifest->chunks);
	options->deadline_ns = deadline();
	const pid_t killed = fork();
	if (killed == 0)
		_exit(fault_arm("prepare_copy_killed_verified", 1) &&
				      prepare_replicas(options) == STATUS_OK ?
			      EXIT_SUCCESS :
			      EXIT_FAILURE);
	int killed_status = 0;
	expect(killed > 0 && waitpid(killed, &killed_status, 0) == killed &&
		       WIFSIGNALED(killed_status) &&
		       WTERMSIG(killed_status) == SIGKILL &&
		       access(receipt, F_OK) != 0,
	       "a failed spot check removes the receipt at once: a run killed before its end leaves none");
	expect(byte_flip(target->path, corrupted),
	       "the killed run's byte restored");
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       access(receipt, F_OK) == 0 &&
		       byte_flip(target->path, corrupted) &&
		       receipt_refresh(target->path, receipt),
	       "corruption under an intact receipt prepared again");
	options->deadline_ns = deadline();
	expect(fault_arm("prepare_receipt_remove_sync", 1) &&
		       run_captured(options, &status, text, sizeof(text)) &&
		       fault_disarm() == 1 && !target->complete &&
		       access(receipt, F_OK) != 0 &&
		       strstr(text, "may outlive a crash") != NULL,
	       "a failed final whose receipt removal cannot be synced says so");
	expect(byte_flip(target->path, corrupted),
	       "the byte restored once more");
	options->spot_chunks = 0;
	options->spot_seed = 0;
}
#endif
/* r18 (request 33 L02): a final's verification receipt. A final that matches it is spot-checked; a changed
 * mtime, inode or receipt field forces a full read. The trade-off: a corrupted chunk under an intact receipt passes a
 * spot check that misses it, and fails one that hits it and --verify=full. */
static void receipt_cases(struct prepare_copy_options *options,
			  const unsigned char *data)
{
	struct prepare_target *const target = &options->targets[0];
	const uint64_t bytes = options->manifest->bytes,
		       chunk_bytes = options->manifest->chunk_bytes;
	const uint32_t chunks = options->manifest->chunks;
	char receipt[PATH_MAX];
	expect(suffixed(receipt, sizeof(receipt), target->path,
			PREPARE_RECEIPT_SUFFIX) &&
		       access(receipt, F_OK) == 0,
	       "a full verification leaves the final's receipt");
	options->spot_chunks = 1;
	options->spot_seed = seed_spotting(0, chunks);
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       target->verified == chunk_bytes &&
		       target->trusted == bytes - chunk_bytes,
	       "a final matching its receipt is spot-checked, the rest trusted");
	receipt_binding_cases(options, receipt);
	const struct timespec later[FILE_TIMES] = { { 0, UTIME_OMIT },
						    { time(NULL) + 1, 0 } };
	options->deadline_ns = deadline();
	expect(utimensat(AT_FDCWD, target->path, later, 0) == 0 &&
		       prepare_replicas(options) == STATUS_OK &&
		       target->complete && target->trusted == 0 &&
		       target->verified == bytes,
	       "a changed mtime forces a full verification");
	char moved[PATH_MAX];
	const int copy =
		suffixed(moved, sizeof(moved), target->path, ".moved") ?
			open(moved, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			     S_IRUSR | S_IWUSR) :
			-1;
	const bool replaced =
		copy >= 0 && write(copy, data, bytes) == (ssize_t)bytes &&
		fsync(copy) == 0 && rename(moved, target->path) == 0;
	if (copy >= 0)
		(void)close(copy);
	options->deadline_ns = deadline();
	expect(replaced && prepare_replicas(options) == STATUS_OK &&
		       target->complete && target->trusted == 0 &&
		       target->verified == bytes,
	       "a replaced file (another inode) forces a full verification");
	FILE *const edited = fopen(receipt, "a");
	const bool appended = edited != NULL && fputs("x", edited) >= 0;
	if (edited != NULL)
		(void)fclose(edited);
	options->deadline_ns = deadline();
	expect(appended && prepare_replicas(options) == STATUS_OK &&
		       target->complete && target->trusted == 0 &&
		       target->verified == bytes,
	       "a receipt that differs in any field forces a full verification");
	/* a chunk that changes under an intact receipt */
	const uint32_t corrupted = chunks - 1u;
	expect(byte_flip(target->path, (uint64_t)corrupted * chunk_bytes) &&
		       receipt_refresh(target->path, receipt),
	       "corruption under an intact receipt prepared");
	options->spot_seed = seed_spotting(0, chunks);
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete,
	       "the trade-off: a spot check that misses the corrupted chunk passes the final");
	options->verify_full = true;
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && !target->complete &&
		       target->reason != NULL &&
		       strcmp(target->reason, "finalized_replica_unverified") ==
			       0 &&
		       access(receipt, F_OK) != 0,
	       "--verify=full catches it, and the failed final loses its receipt");
	options->verify_full = false;
	expect(receipt_refresh(target->path, receipt) == false &&
		       access(receipt, F_OK) != 0,
	       "no receipt is left to refresh");
	options->spot_seed = seed_spotting(corrupted, chunks);
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && !target->complete &&
		       target->trusted == 0,
	       "without a receipt the final is read in full and refused");
	expect(byte_flip(target->path, (uint64_t)corrupted * chunk_bytes),
	       "the corrupted byte restored");
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       target->verified == bytes && access(receipt, F_OK) == 0,
	       "the restored final verifies in full and gets its receipt again");
	/* the spot check that hits the corrupted chunk */
	expect(byte_flip(target->path, (uint64_t)corrupted * chunk_bytes) &&
		       receipt_refresh(target->path, receipt),
	       "corruption under an intact receipt prepared again");
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && !target->complete &&
		       target->reason != NULL &&
		       strcmp(target->reason, "finalized_replica_unverified") ==
			       0 &&
		       access(receipt, F_OK) != 0,
	       "a spot check that hits the corrupted chunk refuses the final and removes its receipt");
	expect(byte_flip(target->path, (uint64_t)corrupted * chunk_bytes),
	       "the corrupted byte restored again");
	options->spot_chunks = 0;
	options->spot_seed = 0;
#if CHECKING_HOOKS_ENABLED
	receipt_fault_cases(options, receipt, (uint64_t)corrupted * chunk_bytes,
			    corrupted);
#endif
}

/* r18 (the r17 fix's verification): a fresh partial is the run's own only once linked; a failed directory sync
 * after the link keeps the engine's journaled checkpoint; the earlier copier's lock guards every target not yet
 * final; a directory the run may not write refuses before anything is examined, in the run and in the plan. */
static void r18_creation_cases(struct prepare_copy_options *options,
			       const char *partial, const char *journal,
			       const unsigned char *data, const char *directory)
{
	struct prepare_target *const target = &options->targets[0];
	struct prepare_checkpoint checkpoint;
#if CHECKING_HOOKS_ENABLED
	options->deadline_ns = deadline();
	expect(fault_arm("prepare_copy_name_taken", 1),
	       "partial name collision armed");
	expect(prepare_replicas(options) == STATUS_OK && !target->complete &&
		       access(partial, F_OK) == 0 && access(journal, F_OK) != 0,
	       "a partial name another writer takes before the link is never removed");
	expect(fault_disarm() == 1, "partial name collision reached");
	(void)unlink(partial);
	remove_target(target);
	options->deadline_ns = deadline();
	expect(fault_arm("prepare_copy_directory_sync", 1),
	       "directory sync failure armed");
	expect(prepare_replicas(options) == STATUS_OK && !target->complete &&
		       target->reason != NULL &&
		       strcmp(target->reason, "copy_device_io_failure") == 0 &&
		       access(partial, F_OK) == 0 && access(journal, F_OK) == 0,
	       "a failed directory sync after the link names the I/O failure and keeps the journaled checkpoint");
	expect(fault_disarm() == 1, "directory sync failure reached");
	expect(prepare_checkpoint_classify(options->manifest, target->path,
					   target->key, options->floor_percent,
					   &checkpoint) == STATUS_OK &&
		       checkpoint.kind == PREPARE_CHECKPOINT_ENGINE,
	       "the plan finds that checkpoint the engine's own");
#endif
	/* every build: the engine's journaled checkpoint for the held-lock case (checking builds keep the one above) */
	if (access(partial, F_OK) != 0)
		expect(engine_checkpoint(options, partial, journal),
		       "an engine checkpoint before the copier's lock is held");
	char copier[PATH_MAX];
	const int holder =
		copier_lock_path(copier, sizeof(copier), target->path) ?
			open(copier,
			     O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
			     S_IRUSR | S_IWUSR) :
			-1;
	const bool held = holder >= 0 && flock(holder, LOCK_EX | LOCK_NB) == 0;
	struct stat before = { 0 }, after = { 0 };
	const bool had_partial = stat(partial, &before) == 0;
	static char text[RUN_CAPTURE_BYTES];
	result_status status = STATUS_INVALID_ARGUMENT;
	options->deadline_ns = deadline();
	expect(held && had_partial &&
		       prepare_checkpoint_classify(options->manifest,
						   target->path, target->key,
						   options->floor_percent,
						   &checkpoint) == STATUS_OK &&
		       checkpoint.kind == PREPARE_CHECKPOINT_HELD &&
		       strcmp(checkpoint.refusal, "copy_in_progress") == 0 &&
		       run_captured(options, &status, text, sizeof(text)) &&
		       status == STATUS_OK && !target->complete &&
		       target->reason != NULL &&
		       strcmp(target->reason, "copy_in_progress") == 0 &&
		       stat(partial, &after) == 0 &&
		       after.st_ino == before.st_ino &&
		       after.st_size == before.st_size &&
		       access(journal, F_OK) == 0,
	       "the earlier copier's held lock refuses any target not yet final, in the plan and the run, touching "
	       "nothing");
	expect(strstr(text,
		      "copy_in_progress; not examined: a copier holds its lock .<target>.copy.lock; the engine never "
		      "removes or changes it") != NULL &&
		       strstr(text, "no matching engine journal") == NULL &&
		       strstr(text, "remove it") == NULL,
	       "the kept line says the engine checkpoint was not examined: no journal claim, no removal advice");
	if (holder >= 0)
		(void)close(holder);
	options->deadline_ns = deadline();
	expect(prepare_replicas(options) == STATUS_OK && target->complete &&
		       file_matches(target->path, data),
	       "with the copier's lock free the run resumes or copies the target");
	receipt_cases(options, data);
	char receipt[PATH_MAX];
	if (suffixed(receipt, sizeof(receipt), target->path,
		     PREPARE_RECEIPT_SUFFIX))
		(void)unlink(receipt);
	remove_target(target);
#if CHECKING_HOOKS_ENABLED
	/* a published partial's receipt binds its state once durable: one that changes as it is published gets none */
	options->deadline_ns = deadline();
	expect(fault_arm("prepare_copy_target_changed", 1) &&
		       run_captured(options, &status, text, sizeof(text)) &&
		       fault_disarm() == 1 && status == STATUS_OK &&
		       target->complete && access(receipt, F_OK) != 0 &&
		       strstr(text, "changed while it was published") != NULL,
	       "a partial that changes as it is published gets no receipt, and the line says why");
	(void)unlink(receipt);
	remove_target(target);
#endif
	/* A directory the run may not write: no lock can be created, so the run refuses before it examines anything. */
	if (geteuid() != 0 && chmod(directory, S_IRUSR | S_IXUSR) == 0) {
		options->deadline_ns = deadline();
		expect(prepare_checkpoint_classify(options->manifest,
						   target->path, target->key,
						   options->floor_percent,
						   &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_HELD &&
			       strcmp(checkpoint.refusal,
				      "destination_directory_policy") == 0 &&
			       !checkpoint.removed &&
			       prepare_replicas(options) == STATUS_OK &&
			       !target->complete && target->reason != NULL &&
			       strcmp(target->reason,
				      "destination_directory_policy") == 0,
		       "an unwritable directory refuses before any examination, in the plan and the run");
		(void)chmod(directory, S_IRWXU);
	}
}

/* An autotune copy is found by the startup replica search without a profile: its full verification writes the search's
 * receipt (replicas.h) beside its .verified one. The copy is a target named by the pack identity in the search's
 * directory under a fixture mount point on the first scratch device; the search admits it when that device's media is
 * admitted and is not the source's. */
static void search_receipt_case(const struct prepare_copy_options *options,
				const struct prepare_target *template,
				const char *parent)
{
	char point[PATH_MAX], directory[PATH_MAX], container[PATH_MAX],
		receipt[PATH_MAX],
		table[PATH_MAX] = { 0 }, line[PATH_MAX + FIELD_TEXT_BYTES],
		expected[SEARCH_RECEIPT_TEST_BYTES],
		found_text[SEARCH_RECEIPT_TEST_BYTES] = { 0 };
	struct prepare_target target = { 0 };
	int length = snprintf(point, sizeof(point), "%s/search-XXXXXX", parent);
	bool prepared = length > 0 && (size_t)length < sizeof(point) &&
			mkdtemp(point) != NULL &&
			append_path(container, sizeof(container), point,
				    "/anemone_engram_rnd") &&
			mkdir(container, S_IRWXU) == 0 &&
			append_path(directory, sizeof(directory), point,
				    SEARCH_DIRECTORY) &&
			mkdir(directory, S_IRWXU) == 0;
	char name[STORAGE_PROFILE_KEY_BYTES + FIELD_TEXT_BYTES];
	length = snprintf(name, sizeof(name), "/%s.anemone",
			  options->manifest->pack_identity);
	prepared = prepared && length > 0 && (size_t)length < sizeof(name) &&
		   append_path(target.path, sizeof(target.path), directory,
			       name) &&
		   suffixed(receipt, sizeof(receipt), target.path,
			    REPLICA_RECEIPT_SUFFIX) &&
		   replicas_receipt_text(expected, sizeof(expected),
					 options->manifest->source,
					 options->manifest->bytes,
					 FIXTURE_HEADER_DIGEST);
	(void)snprintf(target.key, sizeof(target.key), "%s", template->key);
	target.device = template->device;
	struct prepare_copy_options single = *options;
	single.targets = &target;
	single.count = 1;
	single.workers = 1;
	single.verify_full = true;
	single.deadline_ns = deadline();
	expect(prepared && prepare_replicas(&single) == STATUS_OK &&
		       target.complete,
	       "an autotune copy into the search's directory completes");
	FILE *const file = prepared ? fopen(receipt, "re") : NULL;
	const size_t read =
		file != NULL ?
			fread(found_text, 1, sizeof(found_text) - 1u, file) :
			0;
	if (file != NULL)
		(void)fclose(file);
	expect(read == strlen(expected) &&
		       memcmp(found_text, expected, read) == 0,
	       "its full verification writes the search's receipt: the source, its size and header digest");
	struct local_device_description source_device, point_device;
	const bool admitted =
		local_device_describe(point, &point_device) == STATUS_OK &&
		local_device_media_refusal(&point_device, false) == NULL &&
		!(local_device_describe(options->manifest->source,
					&source_device) == STATUS_OK &&
		  local_device_overlaps(&source_device, &point_device));
	length = snprintf(line, sizeof(line), "/dev/fixture0 %s ext4 rw 0 0\n",
			  point);
	FILE *const mounts = prepared && length > 0 &&
					     (size_t)length < sizeof(line) &&
					     append_path(table, sizeof(table),
							 point, "/mounts") ?
				     fopen(table, "w+e") :
				     NULL;
	const bool listed = mounts != NULL && fputs(line, mounts) >= 0 &&
			    fseek(mounts, 0, SEEK_SET) == 0;
	const struct pack_header header = {
		.file_byte_count = options->manifest->bytes,
		.header_digest = FIXTURE_HEADER_DIGEST
	};
	char found[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
	const uint32_t count =
		listed ? replicas_find_mounts(mounts, options->manifest->source,
					      &header, found, REPLICAS_MOST,
					      false) :
			 0;
	if (mounts != NULL)
		(void)fclose(mounts);
	expect(listed && count == (admitted ? 1u : 0u) &&
		       (count == 0 || strcmp(found[0], target.path) == 0),
	       "the startup replica search finds the autotune copy without a profile");
	remove_target(&target);
	if (table[0] != '\0')
		(void)unlink(table);
	expect(!prepared || (rmdir(directory) == 0 && rmdir(container) == 0 &&
			     rmdir(point) == 0),
	       "the search case leaves its folders empty");
}

int main(void)
{
	pure_cases();
	destination_cases();
	stage_option_cases();
	const char *const first_parent =
		getenv("ANEMONE_NATIVE_COPY_TEST_FIRST");
	const char *const second_parent =
		getenv("ANEMONE_NATIVE_COPY_TEST_SECOND");
	if (first_parent == NULL || second_parent == NULL) {
		(void)fputs(
			"replica_prepare: real-filesystem cases require the admitted two-device scratch environment\n",
			stderr);
		return failures != 0 ? EXIT_FAILURE : EXIT_SUCCESS;
	}
	char source_directory[PATH_MAX],
		target_directory[FIXTURE_TARGETS][PATH_MAX] = { { 0 } };
	const char *const temporary = getenv("TMPDIR");
	int length = snprintf(source_directory, sizeof(source_directory),
			      "%s/anemone-native-source-XXXXXX",
			      temporary != NULL ? temporary : "/tmp");
	if (length <= 0 || (size_t)length >= sizeof(source_directory) ||
	    mkdtemp(source_directory) == NULL)
		return EXIT_FAILURE;
	const char *const parents[FIXTURE_TARGETS] = { first_parent,
						       second_parent };
	struct prepare_target targets[FIXTURE_TARGETS];
	memset(targets, 0, sizeof(targets));
	bool prepared = true;
	for (uint32_t index = 0; index < FIXTURE_TARGETS; index++) {
		length = snprintf(target_directory[index],
				  sizeof(target_directory[index]),
				  "%s/case-XXXXXX", parents[index]);
		prepared = prepared && length > 0 &&
			   (size_t)length < sizeof(target_directory[index]) &&
			   mkdtemp(target_directory[index]) != NULL;
		const bool named = append_path(targets[index].path,
					       sizeof(targets[index].path),
					       target_directory[index],
					       "/fixture.anemone");
		prepared = prepared && named &&
			   storage_device_key_for(target_directory[index],
						  false, targets[index].key,
						  &targets[index].device) ==
				   STATUS_OK;
	}
	struct prepare_manifest *const manifest = calloc(1, sizeof(*manifest));
	if (manifest == NULL)
		return EXIT_FAILURE;
	length = snprintf(manifest->source, sizeof(manifest->source),
			  "%s/source", source_directory);
	prepared = prepared && length > 0 &&
		   (size_t)length < sizeof(manifest->source);
	unsigned char data[FIXTURE_BYTES];
	for (uint32_t index = 0; index < FIXTURE_BYTES; index++)
		data[index] = (unsigned char)(index % PATTERN_MODULUS);
	int descriptor = open(manifest->source,
			      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			      S_IRUSR | S_IWUSR);
	prepared = prepared && descriptor >= 0 &&
		   write(descriptor, data, sizeof(data)) ==
			   (ssize_t)sizeof(data) &&
		   fsync(descriptor) == 0;
	if (descriptor >= 0)
		(void)close(descriptor);
	manifest->bytes = (uint64_t)FIXTURE_BYTES;
	manifest->chunk_bytes = (uint64_t)FIXTURE_CHUNK;
	manifest->chunks = FIXTURE_BYTES / FIXTURE_CHUNK;
	manifest->known = manifest->chunks;
	(void)snprintf(manifest->pack_identity, sizeof(manifest->pack_identity),
		       "%s", "0000000000000001");
	for (uint32_t chunk = 0; prepared && chunk < manifest->chunks; chunk++)
		prepared = prepare_hash(data + (size_t)(chunk * FIXTURE_CHUNK),
					(size_t)FIXTURE_CHUNK,
					manifest->hashes[chunk]) == STATUS_OK;
	char manifest_path[PATH_MAX] = { 0 };
	prepared = prepared &&
		   append_path(manifest_path, sizeof(manifest_path),
			       source_directory, "/manifest.json") &&
		   prepare_manifest_write(manifest_path, manifest) == STATUS_OK;
	expect(prepared,
	       "owned root/XFS/ext4 fixtures prepared on three physical devices");
	if (prepared) {
		uint64_t source_read = UINT64_MAX;
		static char run_text[RUN_CAPTURE_BYTES];
		result_status copied = STATUS_INVALID_ARGUMENT;
		struct prepare_copy_options options = {
			.manifest_path = manifest_path,
			.manifest = manifest,
			.targets = targets,
			.count = FIXTURE_TARGETS,
			.floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT,
			.workers = FIXTURE_TARGETS,
			.deadline_ns = deadline(),
			.source_bytes_read = &source_read
		};
		expect(run_captured(&options, &copied, run_text,
				    sizeof(run_text)) &&
			       copied == STATUS_OK,
		       "direct engine fan-out copy finishes");
		expect(source_read == (uint64_t)FIXTURE_BYTES,
		       "source payload read exactly once for two targets");
		/* r18 review: a fresh target (unnamed or named) holds allocated zeros, so with a known manifest each
		 * chunk is written and read back, never read and hashed first */
		expect(stage_bytes(run_text, "target_read") == 0 &&
			       stage_bytes(run_text, "target_hash") == 0 &&
			       stage_bytes(run_text, "write") ==
				       (uint64_t)FIXTURE_TARGETS *
					       (uint64_t)FIXTURE_BYTES &&
			       stage_bytes(run_text, "readback_read") ==
				       (uint64_t)FIXTURE_TARGETS *
					       (uint64_t)FIXTURE_BYTES,
		       "a fresh copy with a known manifest reads none of its new targets before writing them");
		for (uint32_t index = 0; index < FIXTURE_TARGETS; index++) {
			struct stat file;
			expect(targets[index].complete &&
				       file_matches(targets[index].path, data),
			       "published replica has exact source bytes");
			expect(stat(targets[index].path, &file) == 0 &&
				       file.st_size == (off_t)FIXTURE_BYTES &&
				       file.st_blocks > 0,
			       "full-size allocated replica");
			expect(targets[index].extents.count != 0 &&
				       targets[index].extents.largest >=
					       (uint64_t)FIXTURE_CHUNK,
			       "FIEMAP count/largest on real XFS/ext4 scratch");
		}
		search_receipt_case(&options, &targets[0], first_parent);
		options.deadline_ns = deadline();
		source_read = UINT64_MAX;
		expect(prepare_replicas(&options) == STATUS_OK &&
			       source_read == 0,
		       "idempotent finals are verified without rereading the source");
		char partial[PATH_MAX], journal[PATH_MAX], lock[PATH_MAX];
		expect(append_path(partial, sizeof(partial), targets[0].path,
				   PREPARE_PARTIAL_SUFFIX) &&
			       append_path(journal, sizeof(journal),
					   targets[0].path, ".copy-state") &&
			       append_path(lock, sizeof(lock), targets[0].path,
					   ".copy-lock"),
		       "partial, journal and lock paths bounded");
		/* A foreign checkpoint (no engine journal) beside a final another command published is kept.
		 * */
		descriptor = open(partial,
				  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
				  S_IRUSR | S_IWUSR);
		expect(descriptor >= 0 &&
			       posix_fallocate(descriptor, 0,
					       (off_t)FIXTURE_BYTES) == 0,
		       "full-size foreign checkpoint beside the published final");
		if (descriptor >= 0)
			(void)close(descriptor);
		struct prepare_checkpoint checkpoint;
		expect(prepare_checkpoint_classify(manifest, targets[0].path,
						   targets[0].key,
						   options.floor_percent,
						   &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_FOREIGN &&
			       checkpoint.beside_final &&
			       checkpoint.refusal == NULL &&
			       strcmp(checkpoint.why,
				      "a final file is beside it") == 0,
		       "the plan names a foreign checkpoint beside a usable final: the final is used, the partial "
		       "kept");
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       targets[0].complete &&
			       access(partial, F_OK) == 0,
		       "a foreign checkpoint beside a final is kept and the final still verifies");
		/* The engine's own checkpoint beside it is removed once the final verifies every byte. */
		for (uint32_t index = 0; index < FIXTURE_TARGETS; index++)
			remove_target(&targets[index]);
		options.count = 1;
		expect(engine_checkpoint(&options, partial, journal),
		       "a stopped run leaves a journaled engine checkpoint");
		descriptor = open(targets[0].path,
				  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
				  S_IRUSR | S_IWUSR);
		expect(descriptor >= 0 &&
			       write(descriptor, data, sizeof(data)) ==
				       (ssize_t)sizeof(data) &&
			       fsync(descriptor) == 0,
		       "a final published beside the engine checkpoint");
		if (descriptor >= 0)
			(void)close(descriptor);
		expect(prepare_checkpoint_classify(manifest, targets[0].path,
						   targets[0].key,
						   options.floor_percent,
						   &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_ENGINE &&
			       checkpoint.beside_final && !checkpoint.adopted &&
			       checkpoint.refusal == NULL,
		       "the plan names the engine checkpoint beside a final");
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       targets[0].complete &&
			       access(partial, F_OK) != 0 &&
			       access(journal, F_OK) != 0,
		       "an engine checkpoint beside a final is removed once the final verifies");
		remove_target(&targets[0]);
		/* An engine checkpoint is removed for a space refusal... */
		expect(engine_checkpoint(&options, partial, journal),
		       "engine checkpoint before a space refusal");
		options.floor_percent = LOCAL_DEVICE_PERCENT - 1u;
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete &&
			       strcmp(targets[0].reason,
				      "insufficient_space") == 0 &&
			       access(partial, F_OK) != 0 &&
			       access(journal, F_OK) != 0,
		       "a space refusal removes the engine checkpoint and its journal");
		options.floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT;
		/* ...and kept for an identity refusal, which may be transient. */
		expect(engine_checkpoint(&options, partial, journal),
		       "engine checkpoint before an identity refusal");
		char key[STORAGE_PROFILE_KEY_BYTES];
		memcpy(key, targets[0].key, sizeof(key));
		targets[0].key[0] = targets[0].key[0] == '0' ? '1' : '0';
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete &&
			       strcmp(targets[0].reason,
				      "hardware_identity_mismatch_retune_required") ==
				       0 &&
			       access(partial, F_OK) == 0 &&
			       access(journal, F_OK) == 0,
		       "an identity refusal keeps the checkpoint and its journal");
		memcpy(targets[0].key, key, sizeof(key));
		remove_target(&targets[0]);
		/* A foreign partial: refused and untouched while recent or unproven, adopted on proof. The read-only
		 * classifier agrees with each run and creates nothing. */
		struct stat written;
		expect(foreign_partial(partial, data, SECOND_CHUNK, false,
				       &written) &&
			       prepare_checkpoint_classify(
				       manifest, targets[0].path,
				       targets[0].key, options.floor_percent,
				       &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_FOREIGN &&
			       access(lock, F_OK) != 0 &&
			       foreign_untouched(partial, journal, data,
						 SECOND_CHUNK, &written),
		       "the plan names a recent foreign partial foreign and creates nothing");
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete &&
			       strcmp(targets[0].reason,
				      "foreign_partial_kept") == 0 &&
			       foreign_untouched(partial, journal, data,
						 SECOND_CHUNK, &written),
		       "a recent foreign partial is refused and left byte for byte");
		expect(foreign_partial(partial, data, FIRST_CHUNK, true,
				       &written) &&
			       prepare_checkpoint_classify(
				       manifest, targets[0].path,
				       targets[0].key, options.floor_percent,
				       &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_FOREIGN &&
			       strstr(checkpoint.why, "first chunk") != NULL,
		       "the plan names a quiet partial with a foreign first chunk foreign");
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete &&
			       strcmp(targets[0].reason,
				      "foreign_partial_kept") == 0 &&
			       foreign_untouched(partial, journal, data,
						 FIRST_CHUNK, &written),
		       "an unproven foreign partial is refused and left byte for byte");
		expect(foreign_partial(partial, data, UNPROVEN_CHUNKS, true,
				       &written) &&
			       prepare_checkpoint_classify(
				       manifest, targets[0].path,
				       targets[0].key, options.floor_percent,
				       &checkpoint) == STATUS_OK &&
			       checkpoint.kind ==
				       PREPARE_CHECKPOINT_ADOPTABLE &&
			       checkpoint.proven_bytes ==
				       (uint64_t)FIXTURE_CHUNK &&
			       foreign_untouched(partial, journal, data,
						 UNPROVEN_CHUNKS, &written),
		       "the plan proves a quiet foreign partial's first chunk and changes nothing");
		/* r17: a space refusal precedes the proof, in the plan as in the run, which names it unexamined. */
		static char text[CAPTURE_BYTES];
		result_status status = STATUS_OK;
		options.floor_percent = LOCAL_DEVICE_PERCENT - 1u;
		options.deadline_ns = deadline();
		expect(prepare_checkpoint_classify(manifest, targets[0].path,
						   targets[0].key,
						   options.floor_percent,
						   &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_FOREIGN &&
			       checkpoint.refusal != NULL &&
			       strcmp(checkpoint.refusal,
				      "insufficient_space") == 0 &&
			       !checkpoint.removed &&
			       checkpoint.why[0] == '\0' &&
			       run_captured(&options, &status, text,
					    sizeof(text)) &&
			       status == STATUS_OK && !targets[0].complete &&
			       strcmp(targets[0].reason,
				      "insufficient_space") == 0 &&
			       strstr(text, "not examined for adoption") !=
				       NULL &&
			       strstr(text, "remove it") == NULL &&
			       foreign_untouched(partial, journal, data,
						 UNPROVEN_CHUNKS, &written),
		       "a space refusal before the proof: the plan and the kept line say unexamined, nothing removed");
		options.floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT;
		/* r17: while the earlier copier holds its lock, the engine never adopts the partial. */
		char copier[PATH_MAX];
		const int copier_holder =
			copier_lock_path(copier, sizeof(copier),
					 targets[0].path) ?
				open(copier,
				     O_WRONLY | O_CREAT | O_NOFOLLOW |
					     O_CLOEXEC,
				     S_IRUSR | S_IWUSR) :
				-1;
		expect(copier_holder >= 0 &&
			       flock(copier_holder, LOCK_EX | LOCK_NB) == 0 &&
			       prepare_checkpoint_classify(
				       manifest, targets[0].path,
				       targets[0].key, options.floor_percent,
				       &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_HELD &&
			       strcmp(checkpoint.refusal, "copy_in_progress") ==
				       0 &&
			       strstr(checkpoint.why, "a copier holds") != NULL,
		       "the plan names a partial whose copier holds its lock held, as the run refuses it");
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete &&
			       strcmp(targets[0].reason, "copy_in_progress") ==
				       0 &&
			       foreign_untouched(partial, journal, data,
						 UNPROVEN_CHUNKS, &written),
		       "a held copier lock refuses the adoption and leaves the partial byte for byte");
		if (copier_holder >= 0)
			(void)close(copier_holder);
		/* 8 Nov: the proof proves chunk 0 only; the copy then verifies every adopted byte before any
		 * final.
		 * An interrupted adopted run publishes nothing. */
		expect(engine_checkpoint(&options, partial, journal) &&
			       access(targets[0].path, F_OK) != 0 &&
			       prepare_checkpoint_classify(
				       manifest, targets[0].path,
				       targets[0].key, options.floor_percent,
				       &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_ENGINE &&
			       checkpoint.adopted,
		       "a run adopts a quiet foreign partial on proof and its journal records the adoption");
		/* The proven, marked chunk 0 is never trusted from the bitmap: corrupted now, it is re-copied too. */
		descriptor = open(partial, O_WRONLY | O_NOFOLLOW | O_CLOEXEC);
		const unsigned char flipped = (unsigned char)(data[0] ^ 1u);
		expect(descriptor >= 0 &&
			       pwrite(descriptor, &flipped, 1, 0) == 1 &&
			       fsync(descriptor) == 0,
		       "chunk 0 of the adopted checkpoint corrupted");
		if (descriptor >= 0)
			(void)close(descriptor);
		options.floor_percent = LOCAL_DEVICE_PERCENT - 1u;
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete &&
			       strcmp(targets[0].reason,
				      "insufficient_space") == 0 &&
			       access(partial, F_OK) == 0 &&
			       access(journal, F_OK) == 0,
		       "a space refusal keeps an adopted checkpoint: the engine never removes it");
		options.floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT;
		options.deadline_ns = deadline();
		source_read = UINT64_MAX;
		expect(prepare_replicas(&options) == STATUS_OK &&
			       targets[0].complete &&
			       file_matches(targets[0].path, data) &&
			       targets[0].verified == (uint64_t)FIXTURE_BYTES &&
			       targets[0].written ==
				       (uint64_t)(ADOPTED_REPAIRED_CHUNKS *
						  FIXTURE_CHUNK) &&
			       source_read ==
				       (uint64_t)(ADOPTED_REPAIRED_CHUNKS *
						  FIXTURE_CHUNK),
		       "an adopted partial publishes only once every byte is verified: corrupt chunk 1, where the "
		       "proof "
		       "stopped, chunk 3, which it never read, and chunk 0, proven but corrupted since, are found and "
		       "re-copied, nothing else");
		remove_target(&targets[0]);
		options.floor_percent = LOCAL_DEVICE_PERCENT - 1u;
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete &&
			       strcmp(targets[0].reason,
				      "insufficient_space") == 0 &&
			       access(partial, F_OK) != 0,
		       "a fresh target's space refusal creates no partial");
		options.floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT;
		descriptor = open(targets[0].path,
				  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
				  S_IRUSR | S_IWUSR);
		expect(descriptor >= 0 && write(descriptor, "untouched",
						strlen("untouched")) ==
						  (ssize_t)strlen("untouched"),
		       "foreign finalized bytes fixture");
		if (descriptor >= 0)
			(void)close(descriptor);
		/* A checkpoint beside a final that does not verify is kept: the final is never trusted
		 * before it verifies. */
		descriptor = open(partial,
				  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
				  S_IRUSR | S_IWUSR);
		expect(descriptor >= 0 &&
			       posix_fallocate(descriptor, 0,
					       (off_t)FIXTURE_BYTES) == 0,
		       "full-size checkpoint beside a foreign final");
		if (descriptor >= 0)
			(void)close(descriptor);
		/* The plan cannot know a final's bytes without verifying them: as the run, it verifies the final first
		 * and
		 * keeps the partial (an owned final of at most the pack's size; a larger or foreign-owned one
		 * refuses). */
		expect(prepare_checkpoint_classify(manifest, targets[0].path,
						   targets[0].key,
						   options.floor_percent,
						   &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_FOREIGN &&
			       checkpoint.beside_final &&
			       checkpoint.refusal == NULL &&
			       strcmp(checkpoint.why,
				      "a final file is beside it") == 0,
		       "the plan names a partial beside a final the run verifies first: kept, the final checked");
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete,
		       "unverified final is never overwritten");
		expect(access(partial, F_OK) == 0 &&
			       targets[0].reason != NULL &&
			       strcmp(targets[0].reason,
				      "finalized_replica_unverified") == 0,
		       "a checkpoint beside an unverified final is kept, and the final named as unverified");
		descriptor = open(targets[0].path, O_RDONLY | O_CLOEXEC);
		char unchanged[sizeof("untouched")] = { 0 };
		expect(descriptor >= 0 &&
			       read(descriptor, unchanged, sizeof(unchanged)) ==
				       (ssize_t)strlen("untouched") &&
			       strcmp(unchanged, "untouched") == 0,
		       "atomic publication preserves existing bytes");
		if (descriptor >= 0)
			(void)close(descriptor);
		remove_target(&targets[0]);
		/* r17: a directory the plan cannot open, or outside anemone_engram_rnd, is unknown, never "none". */
		char elsewhere[PATH_MAX];
		expect(append_path(elsewhere, sizeof(elsewhere),
				   target_directory[0],
				   "/missing/fixture.anemone") &&
			       prepare_checkpoint_classify(
				       manifest, elsewhere, targets[0].key,
				       options.floor_percent,
				       &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_UNKNOWN &&
			       checkpoint.error == ENOENT &&
			       append_path(elsewhere, sizeof(elsewhere),
					   source_directory,
					   "/fixture.anemone") &&
			       prepare_checkpoint_classify(
				       manifest, elsewhere, targets[0].key,
				       options.floor_percent,
				       &checkpoint) == STATUS_OK &&
			       checkpoint.kind == PREPARE_CHECKPOINT_UNKNOWN &&
			       checkpoint.refusal != NULL &&
			       strcmp(checkpoint.refusal,
				      "destination_directory_policy") == 0,
		       "the plan names an unopenable or unowned directory unknown, as the run refuses it");
		locked_copy_untouched(&options, data);
		/* r18's cases need no hook beyond their own: they run in every build */
		r18_creation_cases(&options, partial, journal, data,
				   target_directory[0]);
		autotune_refusal_cases(manifest->source, source_directory);
		stage_copy_case(source_directory, target_directory[0]);
#if CHECKING_HOOKS_ENABLED
		/* A partial this run created and could not journal is removed: no later run could resume or adopt it.
		 * */
		options.deadline_ns = deadline();
		expect(fault_arm("prepare_copy_journal", 1),
		       "journal checkpoint armed");
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete &&
			       access(partial, F_OK) != 0 &&
			       access(journal, F_OK) != 0,
		       "an unjournaled partial this run created is removed");
		expect(fault_disarm() == 1, "journal checkpoint reached");
		remove_target(&targets[0]);
		/* r17: a process killed after creating the partial and before its journal leaves nothing at the
		 * partial's name (it is created unnamed and linked once journaled), so the next run starts afresh. */
		options.deadline_ns = deadline();
		const pid_t killed = fork();
		if (killed == 0)
			_exit(fault_arm("prepare_copy_killed", 1) &&
					      prepare_replicas(&options) ==
						      STATUS_OK ?
				      EXIT_SUCCESS :
				      EXIT_FAILURE);
		int killed_status = 0;
		expect(killed > 0 &&
			       waitpid(killed, &killed_status, 0) == killed &&
			       WIFSIGNALED(killed_status) &&
			       WTERMSIG(killed_status) == SIGKILL &&
			       access(partial, F_OK) != 0 &&
			       access(journal, F_OK) != 0,
		       "a run killed before its journal leaves no partial and no journal");
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       targets[0].complete &&
			       file_matches(targets[0].path, data),
		       "the next run copies the target afresh, refusing nothing");
		remove_target(&targets[0]);
		/* review finding 14: a worker whose ring is poisoned at a readback (as by an io_uring_enter refused
		 * mid-batch) trusts nothing more of it. The run fails at once, claims nothing from that ring, and no
		 * worker waits for a buffer. In a child: a quarantined worker refuses every later run of the
		 * process. */
		options.deadline_ns = deadline();
		const bool rings = io_uring_available();
		const pid_t poisoned = rings ? fork() : 0;
		if (rings && poisoned == 0) {
			const uint64_t started = deadline();
			const bool armed =
				fault_arm("prepare_copy_readback_poison", 1);
			const result_status run = prepare_replicas(&options);
			_exit(armed && run == STATUS_INPUT_OUTPUT_ERROR &&
					      deadline() - started <
						      POISON_RUN_SECONDS_MOST *
							      NS_SECOND &&
					      fault_disarm() != 0 &&
					      !targets[0].complete &&
					      options.targets[0].verified <
						      (uint64_t)FIXTURE_BYTES ?
				      EXIT_SUCCESS :
				      EXIT_FAILURE);
		}
		int poisoned_status = 0;
		expect(!rings || (poisoned > 0 &&
				  waitpid(poisoned, &poisoned_status, 0) ==
					  poisoned &&
				  WIFEXITED(poisoned_status) &&
				  WEXITSTATUS(poisoned_status) == EXIT_SUCCESS),
		       "a poisoned ring fails its worker at once, as an I/O failure: nothing claimed, no stall");
		if (!rings)
			(void)fputs(
				"replica_prepare_test: io_uring refused here: the poisoned-ring case did not run\n",
				stderr);
		remove_target(&targets[0]);
		/* review finding 15: a first run short of memory for its buffers is refused before its open creates and
		 * preallocates any partial */
		options.deadline_ns = deadline();
		expect(fault_arm("prepare_copy_memory_short", 1),
		       "the memory fault arms");
		expect(prepare_replicas(&options) ==
				       STATUS_INPUT_OUTPUT_ERROR &&
			       !targets[0].complete &&
			       targets[0].reason != NULL &&
			       strcmp(targets[0].reason,
				      "insufficient_memory") == 0,
		       "a first run short of memory is refused");
		expect(fault_disarm() != 0,
		       "the memory fault point is reached");
		expect(access(partial, F_OK) != 0 && access(journal, F_OK) != 0,
		       "the refused run created no partial and no journal");
		remove_target(&targets[0]);
		/* review finding 16 and NF5: one worker and a lag of one chunk, stopped after the second written chunk.
		 * Chunk 0 passed its lag (one more chunk written after it) and is read back and claimed; chunk 1, still
		 * waiting, is read back at the stop inside the lag, counted and left unclaimed */
		options.workers = 1;
		options.readback_lag_bytes = (uint64_t)FIXTURE_CHUNK;
		options.deadline_ns = deadline();
		expect(fault_arm("prepare_copy_stop_written",
				 STOP_AFTER_CHUNKS),
		       "the stop fault arms");
		result_status stopped = STATUS_INVALID_ARGUMENT;
		expect(run_captured(&options, &stopped, run_text,
				    sizeof(run_text)),
		       "the stopped run is captured");
		expect(fault_disarm() != 0, "the stop fault point is reached");
		bool claimed = false;
		uint32_t claims = 0, readbacks = 0, inside = 0;
		expect(stopped == STATUS_TIMED_OUT && !targets[0].complete &&
			       targets[0].resumable &&
			       targets[0].verified >= (uint64_t)FIXTURE_CHUNK,
		       "a stop reads back the chunks written before it and is resumable");
		expect(prepare_journal_claims(targets[0].path, 0, &claimed,
					      &claims) == STATUS_OK &&
			       claimed && claims == 1u,
		       "the journal claims only the chunk read back after its lag");
		expect(prepare_journal_claims(targets[0].path, 1, &claimed,
					      &claims) == STATUS_OK &&
			       !claimed,
		       "the chunk the stop read back inside the lag is not claimed");
		expect(copy_counts(run_text, &readbacks, &inside) &&
			       readbacks == STOP_AFTER_CHUNKS && inside == 1u,
		       "the stop counts its readback inside the lag");
		/* the resume reads chunk 0 (claimed: not counted) and chunk 1 (unclaimed: counted inside the lag) at
		 * once, writes the other two, and reads back chunk 2 after its lag and chunk 3, the last, inside it */
		options.deadline_ns = deadline();
		result_status resumed = STATUS_INVALID_ARGUMENT;
		expect(run_captured(&options, &resumed, run_text,
				    sizeof(run_text)) &&
			       resumed == STATUS_OK && targets[0].complete &&
			       file_matches(targets[0].path, data),
		       "the resume publishes the target");
		expect(copy_counts(run_text, &readbacks, &inside) &&
			       readbacks == RESUME_READBACKS &&
			       inside == RESUME_INSIDE_LAG,
		       "the resume counts its unclaimed chunk inside the lag");
		remove_target(&targets[0]);
		/* NF4: a claimed chunk found changed is written again and loses its claim, saved before the write, so a
		 * stop inside its lag leaves it unclaimed */
		options.deadline_ns = deadline();
		expect(fault_arm("prepare_copy_stop_written",
				 STOP_AFTER_CHUNKS) &&
			       prepare_replicas(&options) == STATUS_TIMED_OUT &&
			       fault_disarm() != 0 &&
			       prepare_journal_claims(targets[0].path, 0,
						      &claimed,
						      &claims) == STATUS_OK &&
			       claimed,
		       "a stopped run claims chunk 0");
		const int changed = open(partial, O_RDWR | O_CLOEXEC);
		unsigned char byte = 0;
		bool rewritten = changed >= 0 &&
				 pread(changed, &byte, 1, CHANGED_BYTE) == 1;
		byte = (unsigned char)(byte ^ 1u);
		rewritten = rewritten &&
			    pwrite(changed, &byte, 1, CHANGED_BYTE) == 1 &&
			    fsync(changed) == 0;
		if (changed >= 0)
			(void)close(changed);
		expect(rewritten, "chunk 0 of the partial is changed");
		options.deadline_ns = deadline();
		expect(fault_arm("prepare_copy_stop_written", 1) &&
			       prepare_replicas(&options) == STATUS_TIMED_OUT &&
			       fault_disarm() != 0,
		       "the resume rewrites chunk 0 and is stopped inside its lag");
		expect(prepare_journal_claims(targets[0].path, 0, &claimed,
					      &claims) == STATUS_OK &&
			       !claimed,
		       "the rewritten chunk keeps no claim");
		remove_target(&targets[0]);
		options.workers = FIXTURE_TARGETS;
		options.readback_lag_bytes = 0;
		/* NF2: a source chunk that fails its digest while the run stops ends it as an I/O failure, not a
		 * timeout */
		options.deadline_ns = deadline();
		expect(fault_arm("prepare_copy_stop_mismatch", 1),
		       "the stop mismatch fault arms");
		expect(prepare_replicas(&options) ==
				       STATUS_INPUT_OUTPUT_ERROR &&
			       !targets[0].complete,
		       "a digest failure at a stop is an I/O failure");
		expect(fault_disarm() != 0,
		       "the stop mismatch fault point is reached");
		remove_target(&targets[0]);
		/* NF3: a run that verifies a final, beside a new target on the final's device (its open refuses it as a
		 * duplicate), creates nothing, so the early memory check does not refuse it */
		options.deadline_ns = deadline();
		expect(prepare_replicas(&options) == STATUS_OK &&
			       targets[0].complete,
		       "a final to verify is published");
		const struct prepare_target second = targets[1];
		const bool beside =
			append_path(targets[1].path, sizeof(targets[1].path),
				    target_directory[0], "/beside.anemone");
		targets[1].device = targets[0].device;
		options.count = FIXTURE_TARGETS;
		options.deadline_ns = deadline();
		expect(beside && fault_arm("prepare_copy_memory_short", 1),
		       "the memory fault arms");
		expect(prepare_replicas(&options) == STATUS_OK &&
			       targets[0].complete &&
			       targets[1].reason != NULL &&
			       strcmp(targets[1].reason,
				      "duplicate_physical_device") == 0,
		       "the final is verified and the refused target named");
		expect(fault_disarm() == 0,
		       "the early memory check counts no refused target");
		targets[1] = second;
		options.count = 1;
		remove_target(&targets[0]);
		options.deadline_ns = deadline();
		expect(fault_arm("prepare_copy_pause", 1),
		       "preallocation checkpoint armed");
		expect(prepare_replicas(&options) == STATUS_TIMED_OUT &&
			       targets[0].resumable,
		       "interrupted preallocation recorded resumable");
		expect(fault_disarm() == 1, "preallocation checkpoint reached");
		struct stat allocated;
		expect(stat(partial, &allocated) == 0 &&
			       allocated.st_size == (off_t)FIXTURE_BYTES &&
			       allocated.st_blocks >= 0 &&
			       (uint64_t)allocated.st_blocks *
					       ALLOCATION_BLOCK_BYTES >=
				       (uint64_t)FIXTURE_BYTES &&
			       targets[0].written == 0,
		       "allocation precedes all writes");
		options.deadline_ns = deadline();
		expect(fault_arm("prepare_copy_publish", 1),
		       "publish checkpoint armed");
		expect(prepare_replicas(&options) == STATUS_OK &&
			       !targets[0].complete && targets[0].resumable &&
			       access(targets[0].path, F_OK) != 0,
		       "publication failure exposes only a recorded partial");
		expect(fault_disarm() == 1, "publication checkpoint reached");
#endif
	}
	/* every case folder must be left empty: a file the cases leave behind (C02 found receipts) fails the run */
	for (uint32_t index = 0; index < FIXTURE_TARGETS; index++) {
		remove_target(&targets[index]);
		if (target_directory[index][0] != '\0')
			expect(rmdir(target_directory[index]) == 0,
			       "the cases leave their target folder empty");
	}
	(void)unlink(manifest_path);
	(void)unlink(manifest->source);
	expect(rmdir(source_directory) == 0,
	       "the cases leave their source folder empty");
	free(manifest);
	return failures != 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
