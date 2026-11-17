/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The startup replica search over a fixture mount table: which mounts it may open a receipt on. Every fixture mount
 * holds a valid receipt of the source; inotify records each receipt the search opens. An EXL3 source is described by
 * the drive of its largest source, not by its map file's (storage_device_path). A receipt stands for its replica only
 * beside a regular file (no link) of the source's size; a replica on the drive of a copy the load was given is refused
 * as one on the source's (serve's search beside explicit copies), and beside a given copy, the source too, that no
 * topology describes or whose medium names no drive none is admitted (unknown_physical_topology, the copy named), while
 * a given copy on tmpfs shares no drive; turbo load's reader keeps the given copies first and, greedily, the replicas
 * that open with the copies already kept (replicas_reader_open). */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <unistd.h>
#include "anemone/local_device.h"
#include "anemone/replicas.h"
#include "anemone/storage_exl3.h"
#include "log_expectation.h"

#define FIXTURE_MODE 0700
#define FIXTURE_BYTES 4096u
#define FIXTURE_DIGEST UINT64_C(0x0123456789abcdef)
#define EVENT_BUFFER_BYTES 4096u
#define FIXTURE_NAME_BYTES \
	64u /* a replica or receipt name below its directory */
#define TABLE_LINE_BYTES 64u /* a fixture line besides its mount point */
#define CAPTURE_BYTES 8192u /* the search's stderr */
#define SKIP_LINE_BYTES 128u /* the btrfs line besides its mount point */
#define ENDED_LINE_BYTES \
	256u /* the search's end line besides the copy's path */
#define REPLICA_LAYOUT "anemone_engram_rnd/replicas"
#define LEGACY_LAYOUT \
	"anemone/replicas" /* the search's last layout under a mount point */
#define MAP_MAGIC_BYTES 8u
#define MAP_RESERVED_BYTES 40u
#define MAP_SOURCES 2u /* the map itself and its one shard */
#define MAP_SHARD 1u
#define STUB_COPIES_MOST \
	2u /* the stub storage's reader opens the source and one replica at most */
#define OUTPUT_BYTES 4096u /* the greedy open's lines */
#define FULL_SET_OPENS \
	1u /* the reader over every copy is tried once before any greedy open */
#define GIVEN_COPIES 2u /* the given copies case's: the source and one more */
#define ANONYMOUS_DEVICE_FILE \
	"/sys/fs/cgroup/cgroup.controllers" /* cgroup2: an anonymous st_dev */
#define NO_DRIVE_FILE \
	"/proc/version" /* proc: described, its medium naming no drive (as overlay's, a loop device's or FUSE's) */
#define ADMISSION_FIRST_DEVICE \
	5u /* the admission case's /dev/fixture numbers follow fixture0..4 */
#define UNDESCRIBED_SOURCE_DEVICE                    \
	(ADMISSION_FIRST_DEVICE + ADMISSION_MOUNTS + \
	 2u) /* after the passed-over and given copies cases' */
/* One filesystem of the scratch disk listed as vfat, btrfs and ext4, and a tmpfs point listed as ext4. */
enum fixture_mount {
	VFAT_MOUNT,
	BTRFS_MOUNT,
	VIRTUAL_MOUNT,
	INTERNAL_MOUNT,
	FIXTURE_MOUNTS
};

static unsigned failures;
static void expect(bool passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static bool joined(char path[PATH_MAX], const char *left, const char *right)
{
	const int length = snprintf(path, PATH_MAX, "%s/%s", left, right);
	return length > 0 && length < PATH_MAX;
}

static bool directories(const char *root, const char *relative)
{
	char path[PATH_MAX];
	if (!joined(path, root, relative))
		return false;
	for (char *cursor = path + strlen(root) + 1u; *cursor != '\0'; cursor++)
		if (*cursor == '/') {
			*cursor = '\0';
			const bool made = mkdir(path, FIXTURE_MODE) == 0 ||
					  errno == EEXIST;
			*cursor = '/';
			if (!made)
				return false;
		}
	return mkdir(path, FIXTURE_MODE) == 0 || errno == EEXIST;
}

static bool written(const char *path, const char *text)
{
	FILE *const file = fopen(path, "we");
	if (file == NULL)
		return false;
	const bool good = fputs(text, file) >= 0;
	return fclose(file) == 0 && good;
}

static bool written_bytes(const char *path, const void *bytes, size_t count)
{
	FILE *const file = fopen(path, "we");
	if (file == NULL)
		return false;
	const bool good = fwrite(bytes, 1, count, file) == count;
	return fclose(file) == 0 && good;
}

/* An .exl3map of FIXTURE_BYTES logical bytes, all from the one shard (storage_exl3.h's layout). */
struct map_record {
	unsigned char magic[MAP_MAGIC_BYTES];
	uint64_t bytes;
	uint32_t sources, segments;
	unsigned char reserved[MAP_RESERVED_BYTES];
	char paths[MAP_SOURCES][EXL3_MAP_PATH_BYTES];
	uint64_t segment_offset, segment_bytes, source_offset;
	uint32_t source, segment_reserved;
};
_Static_assert(sizeof(struct map_record) ==
		       EXL3_MAP_HEADER_BYTES +
			       MAP_SOURCES * EXL3_MAP_PATH_BYTES +
			       EXL3_MAP_SEGMENT_BYTES,
	       "the map fixture's layout");

/* The bytes of every fixture replica: as many as the receipt's (the search admits only a file of the source's size). */
static const unsigned char replica_bytes[FIXTURE_BYTES];

/* mount/layout/<digest>.anemone(.receipt) naming the source; the replica directory's watch. */
static bool receipted_mount(const char *mount, const char *layout,
			    const char *real_source, int events, int *watch)
{
	char directory[PATH_MAX], path[PATH_MAX], name[PATH_MAX],
		text[PATH_MAX], receipt[PATH_MAX + FIXTURE_NAME_BYTES];
	(void)snprintf(name, sizeof(name), "%016" PRIx64 ".anemone",
		       FIXTURE_DIGEST);
	const int length = snprintf(text, sizeof(text),
				    "anemone replica 1\nsource %s\nbytes %u\n"
				    "header_digest %016" PRIx64 "\n",
				    real_source, FIXTURE_BYTES, FIXTURE_DIGEST);
	if (length <= 0 || (size_t)length >= sizeof(text) ||
	    !directories(mount, layout) || !joined(directory, mount, layout) ||
	    !joined(path, directory, name) ||
	    !written_bytes(path, replica_bytes, sizeof(replica_bytes)))
		return false;
	const int receipt_length =
		snprintf(receipt, sizeof(receipt), "%s.receipt", path);
	/* watched only once the fixture is written: an event is the search's own open */
	if (receipt_length <= 0 || (size_t)receipt_length >= sizeof(receipt) ||
	    !written(receipt, text))
		return false;
	*watch = inotify_add_watch(events, directory, IN_OPEN);
	return *watch >= 0;
}

static void remove_mount(const char *mount)
{
	static const struct {
		const char *layout,
			*parent; /* a replica directory and the one above it */
	} layouts[] = { { REPLICA_LAYOUT, "anemone_engram_rnd" },
			{ LEGACY_LAYOUT, "anemone" } };
	static const char *const suffixes[] = { "", ".receipt" };
	char directory[PATH_MAX], path[PATH_MAX + FIXTURE_NAME_BYTES];
	for (size_t layout = 0; layout < sizeof(layouts) / sizeof(layouts[0]);
	     layout++) {
		if (!joined(directory, mount, layouts[layout].layout))
			continue;
		for (size_t index = 0;
		     index < sizeof(suffixes) / sizeof(suffixes[0]); index++) {
			const int length = snprintf(path, sizeof(path),
						    "%s/%016" PRIx64
						    ".anemone%s",
						    directory, FIXTURE_DIGEST,
						    suffixes[index]);
			if (length > 0 && (size_t)length < sizeof(path))
				(void)unlink(path);
		}
		(void)rmdir(directory);
		if (joined(directory, mount, layouts[layout].parent))
			(void)rmdir(directory);
	}
	(void)rmdir(mount);
}

/* The map file on a disk (/var/tmp: TMPDIR, or /tmp, may be a tmpfs, whose media the policy refuses), beside a
 * receipted replica's mount; its shard, which holds every logical byte, in /dev/shm. The replica is on another drive
 * than the source's bytes, so it is found where its media is admitted (before: the map file's drive was the source's,
 * and the replica was refused as duplicate_physical_device). */
static void exl3_source_case(const char *shared)
{
	char root[] = "/var/tmp/anemone-replica-search-exl3-XXXXXX";
	if (mkdtemp(root) == NULL) {
		expect(false, "an EXL3 fixture directory on /var/tmp");
		return;
	}
	char map[PATH_MAX] = { 0 }, real_map[PATH_MAX], shard[PATH_MAX] = { 0 },
	     mount[PATH_MAX] = { 0 }, table[PATH_MAX] = { 0 },
	     line[PATH_MAX + TABLE_LINE_BYTES],
	     replica[PATH_MAX + FIXTURE_NAME_BYTES];
	static unsigned char shard_bytes[FIXTURE_BYTES];
	static struct map_record record = { .magic = { 'A', 'N', 'E', 'X', 'L',
						       '3', 0, 0 },
					    .bytes = FIXTURE_BYTES,
					    .sources = MAP_SOURCES,
					    .segments = 1,
					    .segment_bytes = FIXTURE_BYTES,
					    .source = MAP_SHARD };
	const int events = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
	int watch = -1;
	bool prepared = events >= 0 && joined(map, root, "model.exl3map") &&
			joined(shard, shared, "shard.bin") &&
			joined(mount, root, "exl3_internal") &&
			joined(table, root, "exl3_mounts") &&
			strlen(shard) < EXL3_MAP_PATH_BYTES &&
			written_bytes(shard, shard_bytes, sizeof(shard_bytes));
	if (prepared)
		memcpy(record.paths[MAP_SHARD], shard, strlen(shard) + 1u);
	prepared = prepared && written_bytes(map, &record, sizeof(record)) &&
		   realpath(map, real_map) != NULL &&
		   mkdir(mount, FIXTURE_MODE) == 0 &&
		   receipted_mount(mount, REPLICA_LAYOUT, real_map, events,
				   &watch);
	const int length = prepared ? snprintf(line, sizeof(line),
					       "/dev/fixture4 %s ext4 rw 0 0\n",
					       mount) :
				      -1;
	prepared = prepared && length > 0 && (size_t)length < sizeof(line) &&
		   written(table, line);
	expect(prepared, "an EXL3 map fixture, its shard in /dev/shm");
	char bytes_path[EXL3_MAP_PATH_BYTES];
	expect(!prepared || (strcmp(storage_device_path(real_map, bytes_path),
				    shard) == 0 &&
			     storage_device_path(shard, bytes_path) == shard &&
			     storage_device_path(table, bytes_path) == table),
	       "a map's device path is its largest source; any other file is its own");
	if (prepared) {
		const struct pack_header header = {
			.file_byte_count = FIXTURE_BYTES,
			.header_digest = FIXTURE_DIGEST
		};
		char found[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
		FILE *const stream = fopen(table, "re");
		const char *const copies[] = { real_map };
		const uint32_t count =
			replicas_find_mounts(stream, copies, 1u, &header, found,
					     REPLICAS_MOST, false);
		if (stream != NULL)
			(void)fclose(stream);
		struct local_device_description internal;
		const bool admitted =
			local_device_describe(mount, &internal) == STATUS_OK &&
			local_device_media_refusal(&internal, false) == NULL;
		(void)snprintf(replica, sizeof(replica),
			       "%s/" REPLICA_LAYOUT "/%016" PRIx64 ".anemone",
			       mount, FIXTURE_DIGEST);
		expect(count == (admitted ? 1u : 0u) &&
			       (count == 0 || strcmp(found[0], replica) == 0),
		       "a replica beside an EXL3 map file is a copy on another drive than the source's bytes");
	}
	if (mount[0] == '/')
		remove_mount(mount);
	(void)unlink(table); /* an empty path names no file */
	(void)unlink(map);
	(void)unlink(shard);
	if (events >= 0)
		(void)close(events);
	(void)rmdir(root);
}

/* The search over a table of mount points listed as ext4 beside the source and, unless NULL, one more given copy, its
 * stderr captured through the file at capture_path (as main's case captures it) into text. */
static uint32_t captured_search(const char *table, const char *real_source,
				const char *given,
				char (*found)[REPLICA_PATH_CAPACITY],
				const char *capture_path, char *text,
				size_t capacity)
{
	const char *const copies[] = { real_source, given };
	const struct pack_header header = { .file_byte_count = FIXTURE_BYTES,
					    .header_digest = FIXTURE_DIGEST };
	text[0] = '\0';
	(void)fflush(stderr);
	const int capture = open(capture_path,
				 O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
				 S_IRUSR | S_IWUSR);
	const int saved = dup(STDERR_FILENO);
	bool redirected = false;
	if (capture >= 0 && saved >= 0)
		redirected = dup2(capture, STDERR_FILENO) >= 0;
	FILE *const stream = fopen(table, "re");
	const uint32_t count = replicas_find_mounts(
		stream, copies, given != NULL ? GIVEN_COPIES : 1u, &header,
		found, REPLICAS_MOST, false);
	if (stream != NULL)
		(void)fclose(stream);
	(void)fflush(stderr);
	if (saved >= 0) {
		(void)dup2(saved, STDERR_FILENO);
		(void)close(saved);
	}
	if (capture >= 0) {
		if (redirected)
			(void)test_log_capture_read(capture, text, capacity);
		(void)close(capture);
		(void)unlink(capture_path);
	}
	return count;
}

static uint32_t occurrences(const char *text, const char *word)
{
	uint32_t count = 0;
	for (const char *at = strstr(text, word); at != NULL;
	     at = strstr(at + strlen(word), word))
		count++;
	return count;
}

enum admission_mount {
	MISSING_MOUNT,
	SHORT_MOUNT,
	LINKED_MOUNT,
	FULL_MOUNT,
	ADMISSION_MOUNTS
};

/* Beside no file, a short file or a link to a full copy a receipt stands for nothing: the search passes over each
 * with replica_missing_short_or_unverified, and finds the full copy. The search reads each device once (its layouts
 * on that one pass), and the four fixture mounts are folders on one /var/tmp device, so each mount gets a search of its
 * own, over a mount table of its one line. The source is in /dev/shm, another device than the mounts. */
static void admission_case(const char *shared)
{
	char root[] = "/var/tmp/anemone-replica-search-admission-XXXXXX";
	char source[PATH_MAX] = { 0 }, real_source[PATH_MAX], table[PATH_MAX],
	     capture[PATH_MAX], mounts[ADMISSION_MOUNTS][PATH_MAX] = { { 0 } },
	     replicas[ADMISSION_MOUNTS][PATH_MAX + FIXTURE_NAME_BYTES],
	     text[CAPTURE_BYTES];
	static const char *const names[ADMISSION_MOUNTS] = { "missing", "short",
							     "linked", "full" };
	const int events = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
	bool prepared =
		mkdtemp(root) != NULL && events >= 0 &&
		joined(source, shared, "admission-source.anemone") &&
		written_bytes(source, replica_bytes, sizeof(replica_bytes)) &&
		realpath(source, real_source) != NULL &&
		joined(table, root, "mounts") &&
		joined(capture, root, "stderr");
	for (uint32_t index = 0; prepared && index < ADMISSION_MOUNTS;
	     index++) {
		int watch = -1;
		const int length = snprintf(replicas[index],
					    sizeof(replicas[index]),
					    "%s/%s/" REPLICA_LAYOUT
					    "/%016" PRIx64 ".anemone",
					    root, names[index], FIXTURE_DIGEST);
		prepared = length > 0 &&
			   (size_t)length < sizeof(replicas[index]) &&
			   joined(mounts[index], root, names[index]) &&
			   mkdir(mounts[index], FIXTURE_MODE) == 0 &&
			   receipted_mount(mounts[index], REPLICA_LAYOUT,
					   real_source, events, &watch);
	}
	prepared = prepared && unlink(replicas[MISSING_MOUNT]) == 0 &&
		   truncate(replicas[SHORT_MOUNT], FIXTURE_BYTES - 1u) == 0 &&
		   unlink(replicas[LINKED_MOUNT]) == 0 &&
		   symlink(replicas[FULL_MOUNT], replicas[LINKED_MOUNT]) == 0;
	expect(prepared,
	       "fixture mounts: a receipt beside no file, a short file, a link and a full copy");
	struct local_device_description device;
	const bool admitted =
		prepared &&
		local_device_describe(mounts[FULL_MOUNT], &device) ==
			STATUS_OK &&
		local_device_media_refusal(&device, false) == NULL;
	bool only_full = prepared, reasoned = prepared;
	for (uint32_t index = 0; prepared && index < ADMISSION_MOUNTS;
	     index++) {
		/* the table written line by line: no fixed buffer for the formatted mount point */
		FILE *const listed = fopen(table, "we");
		const bool listed_one =
			listed != NULL &&
			fprintf(listed, "/dev/fixture%u %s ext4 rw 0 0\n",
				index + ADMISSION_FIRST_DEVICE,
				mounts[index]) > 0;
		if (listed != NULL && fclose(listed) != 0)
			only_full = false;
		char found[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
		const uint32_t count =
			listed_one ?
				captured_search(table, real_source, NULL, found,
						capture, text, sizeof(text)) :
				0;
		const bool full = index == FULL_MOUNT;
		only_full =
			only_full && listed_one &&
			count == (full && admitted ? 1u : 0u) &&
			(count == 0 || strcmp(found[0], replicas[index]) == 0);
		reasoned =
			reasoned &&
			(!admitted ||
			 occurrences(text,
				     "replica_missing_short_or_unverified") ==
				 (full ? 0u : 1u));
		(void)unlink(table);
	}
	expect(only_full,
	       "only the full copy is found: a receipt stands for a regular file of the source's size");
	expect(reasoned,
	       "a receipt beside no file, a short file or a link is passed over with its reason");
	if (prepared && !admitted)
		(void)fputs(
			"replica_search: /var/tmp is on no admitted device: the admission case checks none is found\n",
			stderr);
	for (uint32_t index = 0; index < ADMISSION_MOUNTS; index++)
		if (mounts[index][0] == '/')
			remove_mount(mounts[index]);
	(void)unlink(table);
	(void)unlink(source);
	if (events >= 0)
		(void)close(events);
	(void)rmdir(root);
}

/* A stale receipt in the earlier layout of a mount whose later layout (anemone/replicas) holds a valid copy: the
 * search admits the copy and names the stale receipt, passed over with its reason, once (before, it passed it over
 * silently); a mount with no layout admitted keeps its one excluded line (admission_case). */
static void passed_over_case(const char *shared)
{
	char root[] = "/var/tmp/anemone-replica-search-passed-XXXXXX";
	char source[PATH_MAX] = { 0 }, real_source[PATH_MAX], table[PATH_MAX],
	     capture[PATH_MAX], mount[PATH_MAX] = { 0 },
	     stale[PATH_MAX + FIXTURE_NAME_BYTES],
	     legacy[PATH_MAX + FIXTURE_NAME_BYTES],
	     line[PATH_MAX + TABLE_LINE_BYTES], text[CAPTURE_BYTES];
	const int events = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
	int stale_watch = -1, legacy_watch = -1;
	bool prepared =
		mkdtemp(root) != NULL && events >= 0 &&
		joined(source, shared, "passed-source.anemone") &&
		written_bytes(source, replica_bytes, sizeof(replica_bytes)) &&
		realpath(source, real_source) != NULL &&
		joined(table, root, "mounts") &&
		joined(capture, root, "stderr") &&
		joined(mount, root, "mount") &&
		mkdir(mount, FIXTURE_MODE) == 0 &&
		receipted_mount(mount, REPLICA_LAYOUT, real_source, events,
				&stale_watch) &&
		receipted_mount(mount, LEGACY_LAYOUT, real_source, events,
				&legacy_watch);
	const int stale_length = snprintf(stale, sizeof(stale),
					  "%s/" REPLICA_LAYOUT "/%016" PRIx64
					  ".anemone",
					  mount, FIXTURE_DIGEST);
	const int legacy_length = snprintf(legacy, sizeof(legacy),
					   "%s/" LEGACY_LAYOUT "/%016" PRIx64
					   ".anemone",
					   mount, FIXTURE_DIGEST);
	const int line_length =
		snprintf(line, sizeof(line), "/dev/fixture%u %s ext4 rw 0 0\n",
			 ADMISSION_FIRST_DEVICE + ADMISSION_MOUNTS, mount);
	prepared = prepared && stale_length > 0 &&
		   (size_t)stale_length < sizeof(stale) && legacy_length > 0 &&
		   (size_t)legacy_length < sizeof(legacy) && line_length > 0 &&
		   (size_t)line_length < sizeof(line) && unlink(stale) == 0 &&
		   written(table, line);
	expect(prepared,
	       "fixture mount: a receipt beside no file in the first layout, a full copy in the legacy one");
	struct local_device_description device;
	const bool admitted =
		prepared &&
		local_device_describe(mount, &device) == STATUS_OK &&
		local_device_media_refusal(&device, false) == NULL;
	char found[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
	const uint32_t count = prepared ? captured_search(table, real_source,
							  NULL, found, capture,
							  text, sizeof(text)) :
					  0;
	expect(!prepared || (count == (admitted ? 1u : 0u) &&
			     (count == 0 || strcmp(found[0], legacy) == 0)),
	       "the legacy layout's copy is found behind a stale receipt");
	expect(!prepared || !admitted ||
		       (occurrences(
				text,
				" passed over: replica_missing_short_or_unverified") ==
				1 &&
			strstr(text, stale) != NULL &&
			occurrences(
				text,
				"excluded: replica_missing_short_or_unverified") ==
				0),
	       "the stale receipt before the admitted layout is named once, with its reason");
	if (mount[0] == '/')
		remove_mount(mount);
	(void)unlink(table);
	(void)unlink(source);
	if (events >= 0)
		(void)close(events);
	(void)rmdir(root);
}

/* Serve given two copies (17 Nov 2026): the source in /dev/shm, a second copy and a receipted replica's mount on one
 * /var/tmp device. Beside the source alone the replica is found; beside both given copies it is refused by name
 * (duplicate_physical_device), as a replica on the source's own drive is; beside a given copy no topology describes or
 * whose medium names no drive the search ends with the copy named (unknown_physical_topology); beside a given copy on
 * tmpfs, which shares no drive, the replica is found. */
static void given_copies_case(const char *shared)
{
	char root[] = "/var/tmp/anemone-replica-search-given-XXXXXX";
	char source[PATH_MAX] = { 0 }, real_source[PATH_MAX], table[PATH_MAX],
	     capture[PATH_MAX], mount[PATH_MAX] = { 0 },
	     given[PATH_MAX] = { 0 }, line[PATH_MAX + TABLE_LINE_BYTES],
	     text[CAPTURE_BYTES];
	const int events = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
	int watch = -1;
	bool prepared =
		mkdtemp(root) != NULL && events >= 0 &&
		joined(source, shared, "given-source.anemone") &&
		written_bytes(source, replica_bytes, sizeof(replica_bytes)) &&
		realpath(source, real_source) != NULL &&
		joined(given, root, "given.anemone") &&
		written_bytes(given, replica_bytes, sizeof(replica_bytes)) &&
		joined(table, root, "mounts") &&
		joined(capture, root, "stderr") &&
		joined(mount, root, "mount") &&
		mkdir(mount, FIXTURE_MODE) == 0 &&
		receipted_mount(mount, REPLICA_LAYOUT, real_source, events,
				&watch);
	const int length =
		snprintf(line, sizeof(line), "/dev/fixture%u %s ext4 rw 0 0\n",
			 ADMISSION_FIRST_DEVICE + ADMISSION_MOUNTS + 1u, mount);
	prepared = prepared && length > 0 && (size_t)length < sizeof(line) &&
		   written(table, line);
	expect(prepared,
	       "fixture: a source in /dev/shm, a given copy and a receipted mount on /var/tmp");
	struct local_device_description device;
	const bool admitted =
		prepared &&
		local_device_describe(mount, &device) == STATUS_OK &&
		local_device_media_refusal(&device, false) == NULL;
	char found[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
	const uint32_t alone = prepared ? captured_search(table, real_source,
							  NULL, found, capture,
							  text, sizeof(text)) :
					  0;
	expect(!prepared || alone == (admitted ? 1u : 0u),
	       "beside the source alone the replica on another drive is found");
	char memory[PATH_MAX] = { 0 };
	const bool memory_written =
		prepared && joined(memory, shared, "given-memory.anemone") &&
		written_bytes(memory, replica_bytes, sizeof(replica_bytes));
	const uint32_t beside_memory =
		memory_written ?
			captured_search(table, real_source, memory, found,
					capture, text, sizeof(text)) :
			0;
	expect(memory_written && beside_memory == (admitted ? 1u : 0u) &&
		       (!admitted ||
			(strstr(text, "eligible") != NULL &&
			 strstr(text, "unknown_physical_topology") == NULL)),
	       "beside a given copy on tmpfs, which shares no drive, the replica is found");
	const uint32_t beside =
		prepared ? captured_search(table, real_source, given, found,
					   capture, text, sizeof(text)) :
			   0;
	expect(!prepared ||
		       (beside == 0 &&
			(!admitted ||
			 strstr(text, "duplicate_physical_device") != NULL)),
	       "beside two given copies a replica on a given copy's drive is refused by name");
	/* A given copy no topology describes could share the replica's drive: a file on an anonymous st_dev (cgroup2's,
	   as a btrfs subvolume's is) where the host has one, a file whose medium is described but names no drive (a
	   /proc file, as overlay, a loop device or FUSE describe), and a stand-in whose describe fails (no such
	   file). */
	char missing[PATH_MAX] = { 0 }, ended[PATH_MAX + SKIP_LINE_BYTES];
	const char *const undescribed[] = { ANONYMOUS_DEVICE_FILE,
					    NO_DRIVE_FILE, missing };
	struct stat anonymous;
	const bool have_anonymous =
		stat(ANONYMOUS_DEVICE_FILE, &anonymous) == 0 &&
		local_device_describe(ANONYMOUS_DEVICE_FILE, &device) !=
			STATUS_OK;
	expect(local_device_describe(NO_DRIVE_FILE, &device) == STATUS_OK &&
		       device.count == 0 &&
		       strcmp(device.filesystem, "proc") == 0,
	       "a /proc file is described on a medium that names no drive");
	prepared = prepared && joined(missing, root, "undescribed.anemone");
	for (uint32_t variant = have_anonymous ? 0u : 1u;
	     prepared && variant < sizeof(undescribed) / sizeof(undescribed[0]);
	     variant++) {
		const uint32_t ended_search = captured_search(
			table, real_source, undescribed[variant], found,
			capture, text, sizeof(text));
		const int ended_length = snprintf(
			ended, sizeof(ended),
			"storage: replica search ends: given copy 1 %s is on no described drive, so no replica is "
			"admitted beside it: unknown_physical_topology\n",
			undescribed[variant]);
		expect(ended_search == 0 &&
			       (!admitted ||
				(ended_length > 0 &&
				 (size_t)ended_length < sizeof(ended) &&
				 strstr(text, ended) != NULL &&
				 strstr(text, "eligible") == NULL)),
		       "beside a given copy no topology describes no replica is admitted, the copy named with "
		       "unknown_physical_topology");
	}
	if (mount[0] == '/')
		remove_mount(mount);
	(void)unlink(table);
	(void)unlink(given);
	if (memory[0] == '/')
		(void)unlink(memory);
	(void)unlink(source);
	if (events >= 0)
		(void)close(events);
	(void)rmdir(root);
}

/* A source no topology describes, or whose medium names no drive, ends the search as any given copy does, named copy
 * 0: a file on an anonymous st_dev (cgroup2's) where the host has one, and a /proc file, each beside a mount of its own
 * on /var/tmp with a receipt naming it. A missing source never reaches the line: the search refuses it first, silently
 * (its real path). */
static void undescribed_source_case(void)
{
	char root[] = "/var/tmp/anemone-replica-search-source-XXXXXX";
	char table[PATH_MAX], capture[PATH_MAX],
		mount[PATH_MAX] = { 0 }, real_source[PATH_MAX],
		line[PATH_MAX + TABLE_LINE_BYTES], text[CAPTURE_BYTES],
		ended[PATH_MAX + ENDED_LINE_BYTES];
	const char *const sources[] = { ANONYMOUS_DEVICE_FILE, NO_DRIVE_FILE };
	const int events = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
	struct local_device_description device;
	struct stat anonymous;
	const bool have_anonymous =
		stat(ANONYMOUS_DEVICE_FILE, &anonymous) == 0 &&
		local_device_describe(ANONYMOUS_DEVICE_FILE, &device) !=
			STATUS_OK;
	bool prepared = mkdtemp(root) != NULL && events >= 0 &&
			joined(table, root, "mounts") &&
			joined(capture, root, "stderr") &&
			joined(mount, root, "mount");
	expect(prepared, "fixture: a scratch folder on /var/tmp");
	for (uint32_t variant = have_anonymous ? 0u : 1u;
	     prepared && variant < sizeof(sources) / sizeof(sources[0]);
	     variant++) {
		int watch = -1;
		const int length = snprintf(line, sizeof(line),
					    "/dev/fixture%u %s ext4 rw 0 0\n",
					    UNDESCRIBED_SOURCE_DEVICE, mount);
		const bool listed =
			realpath(sources[variant], real_source) != NULL &&
			mkdir(mount, FIXTURE_MODE) == 0 &&
			receipted_mount(mount, REPLICA_LAYOUT, real_source,
					events, &watch) &&
			length > 0 && (size_t)length < sizeof(line) &&
			written(table, line);
		const bool admitted =
			listed &&
			local_device_describe(mount, &device) == STATUS_OK &&
			local_device_media_refusal(&device, false) == NULL;
		char found[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
		const uint32_t count =
			listed ?
				captured_search(table, real_source, NULL, found,
						capture, text, sizeof(text)) :
				0;
		const int ended_length = snprintf(
			ended, sizeof(ended),
			"storage: replica search ends: given copy 0 %s is on no described drive, so no replica is "
			"admitted beside it: unknown_physical_topology\n",
			real_source);
		expect(listed && count == 0 &&
			       (!admitted ||
				(ended_length > 0 &&
				 (size_t)ended_length < sizeof(ended) &&
				 strstr(text, ended) != NULL &&
				 strstr(text, "eligible") == NULL)),
		       "beside a source no topology describes no replica is admitted, the source named as copy 0 "
		       "with unknown_physical_topology");
		remove_mount(mount);
		(void)unlink(table);
	}
	if (events >= 0)
		(void)close(events);
	(void)rmdir(root);
}

/* A storage whose reader opens no copy that names a missing file, and at most STUB_COPIES_MOST copies together: a
 * per-copy resource that fails only in combination, though each replica opens alone beside the source. */
static uint32_t stub_opens, stub_closes;
static unsigned char
	stub_reader[1]; /* the stub's reader: an address, never read */
MUST_CHECK static result_status stub_open(const char *const *paths,
					  uint32_t count,
					  struct storage_reader **reader_out)
{
	stub_opens++;
	for (uint32_t index = 0; index < count; index++)
		if (access(paths[index], F_OK) != 0)
			return STATUS_INPUT_OUTPUT_ERROR;
	if (count > STUB_COPIES_MOST)
		return STATUS_OUT_OF_MEMORY;
	void *const opaque = stub_reader;
	*reader_out = opaque;
	return STATUS_OK;
}
static void stub_close(struct storage_reader *reader)
{
	(void)reader;
	stub_closes++;
}
/* The product's storage, its opens and closes counted. */
static uint32_t counted_opens, counted_closes;
MUST_CHECK static result_status counted_open(const char *const *paths,
					     uint32_t count,
					     struct storage_reader **reader_out)
{
	counted_opens++;
	return storage_direct.reader_open(paths, count, reader_out);
}
static void counted_close(struct storage_reader *reader)
{
	counted_closes++;
	storage_direct.reader_close(reader);
}
static const struct storage_operations_record counted_storage = {
	.name = "counted",
	.reader_open = counted_open,
	.reader_close = counted_close
};
static const struct storage_operations_record stub_storage = {
	.name = "stub",
	.reader_open = stub_open,
	.reader_close = stub_close
};

/* The reader case's replicas: the first call tries the missing and the good one, the stub's the good and the other. */
enum reader_replica {
	MISSING_REPLICA,
	GOOD_REPLICA,
	OTHER_REPLICA,
	READER_REPLICAS
};

/* Turbo load's reader (commands.c replicas_add): the source and, greedily, each replica that opens with those kept. */
static void reader_cases(void)
{
	char root[] = "/var/tmp/anemone-replica-search-reader-XXXXXX";
	char source[PATH_MAX] = { 0 }, output[OUTPUT_BYTES] = { 0 };
	char replicas[READER_REPLICAS][REPLICA_PATH_CAPACITY] = { { 0 } };
	const char *copies[STORAGE_MAXIMUM_COPIES] = { 0 };
	const char *const given[] = { source, replicas[GOOD_REPLICA] };
	bool prepared =
		mkdtemp(root) != NULL &&
		joined(source, root, "source.anemone") &&
		joined(replicas[MISSING_REPLICA], root, "missing.anemone") &&
		joined(replicas[GOOD_REPLICA], root, "good.anemone") &&
		joined(replicas[OTHER_REPLICA], root, "other.anemone");
	for (uint32_t index = GOOD_REPLICA; prepared && index < READER_REPLICAS;
	     index++)
		prepared = written_bytes(replicas[index], replica_bytes,
					 sizeof(replica_bytes));
	prepared = prepared &&
		   written_bytes(source, replica_bytes, sizeof(replica_bytes));
	expect(prepared,
	       "a source, a missing replica and two good ones on /var/tmp");
	if (prepared) {
		/* the product's storage, every replica good: one open over all, nothing closed, both kept */
		struct storage_reader *reader = NULL;
		counted_opens = counted_closes = 0;
		FILE *const silent = fmemopen(output, sizeof(output) - 1u, "w");
		const uint32_t all =
			silent != NULL ?
				replicas_reader_open(
					&counted_storage, given, 1u,
					(const char (*)[REPLICA_PATH_CAPACITY]) &
						replicas[GOOD_REPLICA],
					READER_REPLICAS - GOOD_REPLICA, copies,
					&reader, silent) :
				0;
		if (silent != NULL)
			(void)fclose(silent);
		expect(all == READER_REPLICAS - GOOD_REPLICA &&
			       reader != NULL &&
			       counted_opens == FULL_SET_OPENS &&
			       counted_closes == 0 && output[0] == '\0' &&
			       strcmp(copies[1], replicas[GOOD_REPLICA]) == 0 &&
			       strcmp(copies[1u + OTHER_REPLICA - GOOD_REPLICA],
				      replicas[OTHER_REPLICA]) == 0,
		       "every replica good: one open over the full set, both kept, no line");
		if (reader != NULL)
			storage_direct.reader_close(reader);
		/* the missing replica refuses the full set and is left out with the open's status; the good one kept */
		reader = NULL;
		counted_opens = counted_closes = 0;
		FILE *const lines = fmemopen(output, sizeof(output) - 1u, "w");
		const uint32_t kept =
			lines != NULL ?
				replicas_reader_open(
					&counted_storage, given, 1u,
					(const char (*)[REPLICA_PATH_CAPACITY])
						replicas,
					OTHER_REPLICA, copies, &reader, lines) :
				0;
		if (lines != NULL)
			(void)fclose(lines);
		const char *const named =
			strstr(output, replicas[MISSING_REPLICA]);
		expect(kept == 1 && reader != NULL && copies[0] == source &&
			       counted_opens ==
				       FULL_SET_OPENS + OTHER_REPLICA &&
			       counted_closes == 0 &&
			       strcmp(copies[1], replicas[GOOD_REPLICA]) == 0 &&
			       strstr(output, "turbo load: replica ") ==
				       output &&
			       named != NULL &&
			       strstr(named,
				      " does not open beside the 1 copies kept (") ==
				       named + strlen(replicas[MISSING_REPLICA]),
		       "a missing replica is left out with the open's status and the good one kept");
		if (reader != NULL)
			storage_direct.reader_close(reader);
		/* all copies together do not open, each replica alone beside the source does: greedy keeps the first */
		reader = NULL;
		stub_opens = stub_closes = 0;
		FILE *const quiet = fmemopen(output, sizeof(output) - 1u, "w");
		const uint32_t greedy =
			quiet != NULL ?
				replicas_reader_open(
					&stub_storage, given, 1u,
					(const char (*)[REPLICA_PATH_CAPACITY]) &
						replicas[GOOD_REPLICA],
					READER_REPLICAS - GOOD_REPLICA, copies,
					&reader, quiet) :
				0;
		if (quiet != NULL)
			(void)fclose(quiet);
		struct storage_reader *alone = NULL;
		const char *const second[] = { source,
					       replicas[OTHER_REPLICA] };
		/* the full set refused, the good replica kept, the other left out; it alone beside the source opens */
		expect(greedy == 1 && reader != NULL &&
			       strcmp(copies[1], replicas[GOOD_REPLICA]) == 0 &&
			       stub_opens == FULL_SET_OPENS + READER_REPLICAS -
						     GOOD_REPLICA &&
			       stub_closes == 0 &&
			       stub_open(second,
					 (uint32_t)(sizeof(second) /
						    sizeof(second[0])),
					 &alone) == STATUS_OK,
		       "copies that open alone but not together: the greedy reader keeps a set that opened");
		/* two given copies: they come first and the replica after them, in one open */
		reader = NULL;
		counted_opens = counted_closes = 0;
		const uint32_t beside = replicas_reader_open(
			&counted_storage, given, GIVEN_COPIES,
			(const char (*)[REPLICA_PATH_CAPACITY]) &
				replicas[OTHER_REPLICA],
			1u, copies, &reader, stderr);
		expect(beside == 1u && reader != NULL &&
			       counted_opens == FULL_SET_OPENS &&
			       copies[0] == source &&
			       copies[1] == replicas[GOOD_REPLICA] &&
			       strcmp(copies[GIVEN_COPIES],
				      replicas[OTHER_REPLICA]) == 0,
		       "two given copies are kept first and the replica follows them");
		if (reader != NULL)
			storage_direct.reader_close(reader);
	}
	for (uint32_t index = 0; index < READER_REPLICAS; index++)
		if (replicas[index][0] == '/')
			(void)unlink(replicas[index]);
	(void)unlink(source);
	(void)rmdir(root);
}

int main(void)
{
	const char *const temporary = getenv("TMPDIR");
	char root[PATH_MAX],
		shared[] = "/dev/shm/anemone-replica-search-XXXXXX";
	(void)snprintf(root, sizeof(root), "%s/anemone-replica-search-XXXXXX",
		       temporary != NULL ? temporary : "/tmp");
	const bool have_root = mkdtemp(root) != NULL;
	const bool have_shared = mkdtemp(shared) != NULL;
	expect(have_root && have_shared,
	       "fixture directories on the scratch disk and on /dev/shm");
	char source[PATH_MAX], real_source[PATH_MAX], table[PATH_MAX];
	char mounts[FIXTURE_MOUNTS][PATH_MAX] = { { 0 } };
	int watches[FIXTURE_MOUNTS] = { -1, -1, -1, -1 };
	const int events = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
	bool prepared = have_root && have_shared && events >= 0 &&
			joined(source, root, "source.anemone") &&
			written(source, "source") &&
			realpath(source, real_source) != NULL &&
			joined(table, root, "mounts") &&
			joined(mounts[VFAT_MOUNT], root, "vfat") &&
			joined(mounts[BTRFS_MOUNT], root, "btrfs") &&
			joined(mounts[VIRTUAL_MOUNT], shared, "virtual") &&
			joined(mounts[INTERNAL_MOUNT], root, "internal");
	for (uint32_t index = 0; prepared && index < FIXTURE_MOUNTS; index++)
		prepared = mkdir(mounts[index], FIXTURE_MODE) == 0 &&
			   receipted_mount(mounts[index], REPLICA_LAYOUT,
					   real_source, events,
					   &watches[index]);
	char lines[FIXTURE_MOUNTS * (PATH_MAX + TABLE_LINE_BYTES)];
	const int length = prepared ? snprintf(lines, sizeof(lines),
					       "/dev/fixture0 %s vfat rw 0 0\n"
					       "/dev/fixture1 %s btrfs rw 0 0\n"
					       "/dev/fixture2 %s ext4 rw 0 0\n"
					       "/dev/fixture3 %s ext4 rw 0 0\n",
					       mounts[VFAT_MOUNT],
					       mounts[BTRFS_MOUNT],
					       mounts[VIRTUAL_MOUNT],
					       mounts[INTERNAL_MOUNT]) :
				      -1;
	prepared = prepared && length > 0 && (size_t)length < sizeof(lines) &&
		   written(table, lines);
	expect(prepared, "fixture mount table with four receipted mounts");
	if (prepared) {
		const struct pack_header header = {
			.file_byte_count = FIXTURE_BYTES,
			.header_digest = FIXTURE_DIGEST
		};
		char found[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
		/* stderr of the search, in a file under the scratch root: its product lines */
		char captured_path[PATH_MAX + TABLE_LINE_BYTES],
			captured[CAPTURE_BYTES] = { 0 },
			skipped[PATH_MAX + SKIP_LINE_BYTES];
		(void)snprintf(captured_path, sizeof(captured_path),
			       "%s/stderr", root);
		(void)snprintf(
			skipped, sizeof(skipped),
			"storage: replica search passes over %s (btrfs): "
			"unknown_physical_topology\n",
			mounts[BTRFS_MOUNT]);
		(void)fflush(stderr);
		const int capture = open(captured_path,
					 O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
					 S_IRUSR | S_IWUSR);
		const int saved = dup(STDERR_FILENO);
		bool redirected = false;
		if (capture >= 0 && saved >= 0)
			redirected = dup2(capture, STDERR_FILENO) >= 0;
		FILE *const stream = fopen(table, "re");
		const char *const copies[] = { real_source };
		(void)replicas_find_mounts(stream, copies, 1u, &header, found,
					   REPLICAS_MOST, false);
		if (stream != NULL)
			(void)fclose(stream);
		(void)fflush(stderr);
		if (saved >= 0) {
			(void)dup2(saved, STDERR_FILENO);
			(void)close(saved);
		}
		ssize_t captured_bytes = -1;
		if (capture >= 0) {
			captured_bytes = test_log_capture_read(
				capture, captured, sizeof(captured));
			(void)close(capture);
			(void)unlink(captured_path);
		}
		bool opened[FIXTURE_MOUNTS] = { false };
		char buffer[EVENT_BUFFER_BYTES];
		ssize_t bytes;
		while ((bytes = read(events, buffer, sizeof(buffer))) > 0)
			for (ssize_t offset = 0; offset < bytes;) {
				struct inotify_event event;
				memcpy(&event, buffer + offset, sizeof(event));
				const char *const name =
					buffer + offset + sizeof(event);
				for (uint32_t index = 0; index < FIXTURE_MOUNTS;
				     index++)
					opened[index] =
						opened[index] ||
						(event.wd == watches[index] &&
						 event.len != 0 &&
						 strstr(name, ".receipt") !=
							 NULL);
				offset += (ssize_t)(sizeof(event) + event.len);
			}
		struct local_device_description internal;
		const bool admitted =
			local_device_describe(mounts[INTERNAL_MOUNT],
					      &internal) == STATUS_OK &&
			local_device_media_refusal(&internal, false) == NULL;
		expect(!opened[VFAT_MOUNT],
		       "a vfat mount is never probed and takes no filesystem slot");
		expect(!opened[BTRFS_MOUNT] && redirected &&
			       captured_bytes > 0 &&
			       strstr(captured, skipped) != NULL,
		       "a btrfs mount is passed over with its reason, never probed and takes no slot");
		expect(!opened[VIRTUAL_MOUNT],
		       "a mount whose media the policy refuses (tmpfs) is never probed for a receipt");
		expect(opened[INTERNAL_MOUNT] == admitted,
		       "an ext4 mount is probed exactly when its media is admitted");
	}
	if (have_shared) {
		exl3_source_case(shared);
		admission_case(shared);
		passed_over_case(shared);
		given_copies_case(shared);
	}
	undescribed_source_case();
	reader_cases();
	for (uint32_t index = 0; index < FIXTURE_MOUNTS; index++)
		if (prepared || mounts[index][0] == '/')
			remove_mount(mounts[index]);
	(void)unlink(table);
	(void)unlink(source);
	if (have_root)
		(void)rmdir(root);
	if (have_shared)
		(void)rmdir(shared);
	if (events >= 0)
		(void)close(events);
	return failures != 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
