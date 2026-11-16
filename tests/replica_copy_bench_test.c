/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The engine copy on real disks, for its per-stage timings (r18: where a copy's time goes). Off by default: with
 * ANEMONE_COPY_BENCH_BYTES (a multiple of the 64 MiB chunk), ANEMONE_COPY_BENCH_SOURCE (a folder) and
 * ANEMONE_COPY_BENCH_TARGETS (folders separated by ':', at most PREPARE_TARGETS_MOST) it writes a source of that size,
 * copies it with the production fan-out (ANEMONE_COPY_BENCH_WORKERS workers, 1 to PREPARE_COPY_WORKERS_MOST, default
 * PREPARE_COPY_WORKERS_DEFAULT) twice, once with an unknown manifest (the source is hashed) and once more as a full
 * re-verification of the finals, prints the engine's stage lines and the totals, then removes everything it wrote and
 * fails if any of it is left. A run reuses a source already there when it is a regular file of that size, and with
 * ANEMONE_COPY_BENCH_KEEP_SOURCE=1 leaves it for the next run (a sweep of worker counts writes its source once); the
 * run without it removes it. ANEMONE_COPY_BENCH_HASH_PAIRS=1 (or 0) sets two-chunk source hashing; every run prints a
 * digest of its manifest's chunk digests, the same whichever hashing ran. Without them, in the two-device scratch
 * environment of the copy tests (ANEMONE_NATIVE_COPY_TEST_FIRST and _SECOND, the source in TMPDIR), it checks itself
 * (self_check) in folders of its own, which it removes. */
#include "replica_prepare.h"
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define BENCH_SECONDS 3600u
#define BENCH_NS_SECOND 1000000000ull
#define BENCH_PATTERN_MULTIPLIER 2654435761u
#define BENCH_DECIMAL_GB 1e9
#define BENCH_TARGET_NAME "copy-bench.anemone"
#define BENCH_PASSES 2u
/* the self-check's chunks: one worker hashes a pair, then the last chunk alone */
#define BENCH_SELF_CHUNKS 3u
#define BENCH_SELF_BYTES ((uint64_t)BENCH_SELF_CHUNKS * PREPARE_CHUNK_BYTES)
#define BENCH_SELF_TARGETS \
	2u /* the self-check copies to both scratch folders */
#define BENCH_DECIMAL_BASE 10

static uint64_t bench_now(void)
{
	struct timespec time;
	return clock_gettime(CLOCK_MONOTONIC, &time) == 0 ?
		       (uint64_t)time.tv_sec * BENCH_NS_SECOND +
			       (uint64_t)time.tv_nsec :
		       0;
}

static bool source_write(const char *path, uint64_t bytes)
{
	const int descriptor = open(path,
				    O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
				    S_IRUSR | S_IWUSR);
	uint32_t *const block = malloc(PREPARE_CHUNK_BYTES);
	bool written = descriptor >= 0 && block != NULL;
	for (uint64_t offset = 0; written && offset < bytes;
	     offset += PREPARE_CHUNK_BYTES) {
		for (uint32_t word = 0;
		     word < PREPARE_CHUNK_BYTES / sizeof(uint32_t); word++)
			block[word] =
				(uint32_t)(offset / sizeof(uint32_t) + word) *
				BENCH_PATTERN_MULTIPLIER;
		written = pwrite(descriptor, block, PREPARE_CHUNK_BYTES,
				 (off_t)offset) == (ssize_t)PREPARE_CHUNK_BYTES;
	}
	written = written && fsync(descriptor) == 0;
	free(block);
	if (descriptor >= 0)
		(void)close(descriptor);
	return written;
}

/* Everything a run writes beside a target in folder: the target, the engine's checkpoint, journal and lock, the
 * verification receipt and its temporary, the replica search's receipt and its temporary (written after a full
 * verification, replica_prepare.c), and the earlier copier's lock the engine takes for a target not yet final. */
static void target_remove(const char *folder)
{
	static const char *const names[] = {
		BENCH_TARGET_NAME,
		BENCH_TARGET_NAME PREPARE_PARTIAL_SUFFIX,
		BENCH_TARGET_NAME ".copy-state",
		BENCH_TARGET_NAME ".copy-lock",
		BENCH_TARGET_NAME PREPARE_RECEIPT_SUFFIX,
		BENCH_TARGET_NAME ".verified.new",
		BENCH_TARGET_NAME REPLICA_RECEIPT_SUFFIX,
		BENCH_TARGET_NAME REPLICA_RECEIPT_SUFFIX ".new",
		"." BENCH_TARGET_NAME ".copy.lock"
	};
	for (size_t index = 0; index < sizeof(names) / sizeof(names[0]);
	     index++) {
		char name[PATH_MAX];
		if (snprintf(name, sizeof(name), "%s/%s", folder,
			     names[index]) < (int)sizeof(name))
			(void)unlink(name);
	}
}

/* Whether folder holds nothing named after the bench's target (what target_remove missed). */
static bool target_gone(const char *folder)
{
	DIR *const directory = opendir(folder);
	if (directory == NULL)
		return false;
	bool gone = true;
	for (const struct dirent *entry = readdir(directory); entry != NULL;
	     entry = readdir(directory))
		if (strstr(entry->d_name, BENCH_TARGET_NAME) != NULL) {
			(void)printf(
				"replica_copy_bench_test: left behind %s/%s\n",
				folder, entry->d_name);
			gone = false;
		}
	(void)closedir(directory);
	return gone;
}

/* One bench: a source of bytes in source_folder, copied by workers to target_folders (':'-separated) twice. */
static bool bench(uint64_t bytes, uint32_t workers, const char *source_folder,
		  const char *target_folders, bool keep_source, bool pairs,
		  char manifest_digest[STORAGE_PROFILE_KEY_BYTES])
{
	struct prepare_manifest *const manifest = calloc(1, sizeof(*manifest));
	struct prepare_target targets[PREPARE_TARGETS_MOST];
	memset(targets, 0, sizeof(targets));
	char folders[PATH_MAX], manifest_path[PATH_MAX];
	const int folders_length =
		snprintf(folders, sizeof(folders), "%s", target_folders);
	uint32_t count = 0;
	bool valid = manifest != NULL && bytes != 0 && folders_length > 0 &&
		     (size_t)folders_length < sizeof(folders) &&
		     bytes % PREPARE_CHUNK_BYTES == 0 &&
		     bytes / PREPARE_CHUNK_BYTES <= PREPARE_CHUNKS_MOST &&
		     workers != 0 && workers <= PREPARE_COPY_WORKERS_MOST;
	for (char *folder = strtok(folders, ":"); valid && folder != NULL;
	     folder = strtok(NULL, ":")) {
		valid = count < PREPARE_TARGETS_MOST &&
			snprintf(targets[count].path,
				 sizeof(targets[count].path),
				 "%s/" BENCH_TARGET_NAME,
				 folder) < (int)sizeof(targets[count].path) &&
			storage_device_key_for(folder, false,
					       targets[count].key,
					       &targets[count].device) ==
				STATUS_OK;
		count++;
	}
	valid = valid && count != 0 && manifest != NULL &&
		snprintf(manifest->source, sizeof(manifest->source),
			 "%s/copy-bench.source",
			 source_folder) < (int)sizeof(manifest->source) &&
		snprintf(manifest_path, sizeof(manifest_path),
			 "%s/copy-bench.manifest.json",
			 source_folder) < (int)sizeof(manifest_path);
	if (!valid) {
		(void)fputs("replica_copy_bench_test: invalid settings\n",
			    stderr);
		free(manifest);
		return false;
	}
	manifest->bytes = bytes;
	manifest->chunk_bytes = PREPARE_CHUNK_BYTES;
	manifest->chunks = (uint32_t)(bytes / PREPARE_CHUNK_BYTES);
	(void)snprintf(manifest->pack_identity, sizeof(manifest->pack_identity),
		       "%s", "00000000c0b1e5c4");
	const uint64_t written_started = bench_now();
	struct stat kept;
	const bool reused = stat(manifest->source, &kept) == 0 &&
			    S_ISREG(kept.st_mode) &&
			    (uint64_t)kept.st_size == bytes;
	valid = (reused || source_write(manifest->source, bytes)) &&
		prepare_manifest_write(manifest_path, manifest) == STATUS_OK;
	(void)printf("replica_copy_bench_test: source %" PRIu64
		     " bytes %s in %.3f s\n",
		     bytes, reused ? "reused" : "written",
		     (double)(bench_now() - written_started) / BENCH_NS_SECOND);
	static const char *const passes[] = { "copy", "full re-verification" };
	for (uint32_t pass = 0; valid && pass < BENCH_PASSES; pass++) {
		struct prepare_copy_options options = {
			.manifest_path = manifest_path,
			.manifest = manifest,
			.targets = targets,
			.count = count,
			.floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT,
			.workers = workers,
			.deadline_ns = bench_now() + (uint64_t)BENCH_SECONDS *
							     BENCH_NS_SECOND,
			.verify_full = true,
			.pair_source_hashes = pairs
		};
		const uint64_t started = bench_now();
		valid = prepare_replicas(&options) == STATUS_OK;
		const double seconds =
			(double)(bench_now() - started) / BENCH_NS_SECOND;
		for (uint32_t index = 0; index < count; index++) {
			valid = valid && targets[index].complete;
			(void)printf(
				"replica_copy_bench_test: %s %s workers=%" PRIu32
				" hash_pairs=%u written=%" PRIu64
				" verified=%" PRIu64
				" seconds=%.3f GB_per_second=%.3f complete=%u\n",
				passes[pass], targets[index].path, workers,
				pairs ? 1u : 0u, targets[index].written,
				targets[index].verified, seconds,
				seconds > 0 ? (double)bytes / BENCH_DECIMAL_GB /
						      seconds :
					      0.0,
				targets[index].complete ? 1u : 0u);
		}
	}
	/* the copy's chunk digests in one (two-chunk hashing must leave them as they are) */
	valid = valid && prepare_hash(manifest->hashes,
				      (size_t)manifest->chunks *
					      sizeof(manifest->hashes[0]),
				      manifest_digest) == STATUS_OK;
	if (valid)
		(void)printf("replica_copy_bench_test: manifest digests %s\n",
			     manifest_digest);
	for (uint32_t index = 0; index < count; index++) {
		/* the target's folder: its path up to the last slash (bench names every target <folder>/<name>) */
		char folder[PATH_MAX];
		const char *const slash = strrchr(targets[index].path, '/');
		const size_t length =
			slash != NULL ? (size_t)(slash - targets[index].path) :
					0;
		memcpy(folder, targets[index].path, length);
		folder[length] = '\0';
		target_remove(folder);
		valid = target_gone(folder) && valid;
	}
	if (!keep_source)
		(void)unlink(manifest->source);
	(void)unlink(manifest_path);
	free(manifest);
	return valid;
}

/* The bench in the copy tests' scratch environment: BENCH_SELF_CHUNKS chunks, folders of its own, all removed; run
 * twice, the first keeping its source and the second reusing then removing it with one worker and two-chunk source
 * hashing (a pair, then the odd last chunk alone at the run's end): both give the same manifest digests. */
static bool self_check(const char *first_parent, const char *second_parent)
{
	const char *const temporary = getenv("TMPDIR");
	char source[PATH_MAX], first[PATH_MAX], second[PATH_MAX],
		folders[BENCH_SELF_TARGETS * PATH_MAX + 1u];
	const bool named =
		snprintf(source, sizeof(source), "%s/copy-bench-source-XXXXXX",
			 temporary != NULL ? temporary : "/tmp") <
			(int)sizeof(source) &&
		snprintf(first, sizeof(first), "%s/copy-bench-XXXXXX",
			 first_parent) < (int)sizeof(first) &&
		snprintf(second, sizeof(second), "%s/copy-bench-XXXXXX",
			 second_parent) < (int)sizeof(second);
	const bool made = named && mkdtemp(source) != NULL &&
			  mkdtemp(first) != NULL && mkdtemp(second) != NULL;
	char single[STORAGE_PROFILE_KEY_BYTES],
		paired[STORAGE_PROFILE_KEY_BYTES];
	const bool passed =
		made &&
		snprintf(folders, sizeof(folders), "%s:%s", first, second) <
			(int)sizeof(folders) &&
		bench(BENCH_SELF_BYTES, PREPARE_COPY_WORKERS_DEFAULT, source,
		      folders, true, false, single) &&
		bench(BENCH_SELF_BYTES, 1u, source, folders, false, true,
		      paired) &&
		strcmp(single, paired) == 0;
	/* a folder the bench left anything in does not go: the check fails */
	const bool removed = made && rmdir(first) == 0 && rmdir(second) == 0 &&
			     rmdir(source) == 0;
	return passed && removed;
}

int main(void)
{
	const char *const
		size = getenv("ANEMONE_COPY_BENCH_BYTES"),
		*const source_folder = getenv("ANEMONE_COPY_BENCH_SOURCE"),
		*const target_folders = getenv("ANEMONE_COPY_BENCH_TARGETS"),
		*const workers_text = getenv("ANEMONE_COPY_BENCH_WORKERS"),
		*const keep_text = getenv("ANEMONE_COPY_BENCH_KEEP_SOURCE"),
		*const pairs_text = getenv("ANEMONE_COPY_BENCH_HASH_PAIRS"),
		*const first_parent = getenv("ANEMONE_NATIVE_COPY_TEST_FIRST"),
		*const second_parent =
			getenv("ANEMONE_NATIVE_COPY_TEST_SECOND");
	bool passed = true;
	char digest[STORAGE_PROFILE_KEY_BYTES];
	if (size != NULL && source_folder != NULL && target_folders != NULL)
		passed = bench(strtoull(size, NULL, BENCH_DECIMAL_BASE),
			       workers_text != NULL ?
				       (uint32_t)strtoul(workers_text, NULL,
							 BENCH_DECIMAL_BASE) :
				       PREPARE_COPY_WORKERS_DEFAULT,
			       source_folder, target_folders,
			       keep_text != NULL && strcmp(keep_text, "1") == 0,
			       pairs_text != NULL ?
				       strcmp(pairs_text, "1") == 0 :
				       PREPARE_PAIR_SOURCE_HASHES_DEFAULT,
			       digest);
	else if (first_parent != NULL && second_parent != NULL)
		passed = self_check(first_parent, second_parent);
	else
		(void)puts(
			"replica_copy_bench_test: skipped (set ANEMONE_COPY_BENCH_BYTES, _SOURCE and _TARGETS)");
	(void)puts(passed ? "replica_copy_bench_test: passed" :
			    "replica_copy_bench_test: FAILED");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
