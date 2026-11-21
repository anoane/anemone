/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* replicas.c: turbo load's replicas of a pack (replicas.h): the replicate command and the search of a load. */
#include "replicas.h"
#include "engine/storage/io/local_device.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fs.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#include "engine/formats/digest/pack_digest.h"
#include "common/status.h"
#include "engine/storage/io/storage_exl3.h"

#define REPLICA_DIRECTORY "anemone_engram_rnd/replicas"
#define LEGACY_REPLICA_DIRECTORY "anemone/replicas" /* under a mount point */
#define REPLICA_NAME_FORMAT \
	"%016" PRIx64 ".anemone" /* the pack's header digest */
#define RECEIPT_SUFFIX REPLICA_RECEIPT_SUFFIX
#define RECEIPT_FORMAT                                 \
	"anemone replica 1\nsource %s\nbytes %" PRIu64 \
	"\nheader_digest %016" PRIx64 "\n"
#define RECEIPT_CAPACITY (REPLICA_PATH_CAPACITY + 256)
#define MOUNT_LINE_CAPACITY (2 * REPLICA_PATH_CAPACITY)
#define MOUNT_DEVICE_PREFIX "/dev/"
#define BOOT_PREFIX "/boot"
#define MOUNT_ENTRIES_MOST 4096u
#define MOUNTS_MOST 64u /* distinct filesystems looked at */
#define PIECE_BYTES \
	(64ull      \
	 << 20u) /* one read and one write of the copy, one digest piece of the check */
#define VERIFY_THREADS 16u
#define BYTES_PER_MIB (1ull << 20u)
#define BYTES_PER_GIB (1ull << 30u)
#define NANOSECONDS_PER_SECOND 1000000000ull
#define PERCENT 100u
#define DIRECTORY_MODE 0775
#define FILE_MODE 0644
#define DECIMAL 10
#define REPLICATE_FAILED 1
#define REPLICATE_USAGE 2
#define REPLICA_LAYOUTS \
	3u /* the search's directories on one filesystem: HOME's, then the two under its mount point */
#define NO_EXTENT \
	UINT32_MAX /* a verification that failed before any extent differed */
#define ADOPT_OPTION "--adopt="
#define FIRST_OPTION 2 /* after "anemone replicate" */
#define MOUNT_FIELDS 3 /* a mounts line's device, mount point and type */
#define ADOPT_DIRECTORY_PASSES \
	2u /* an adopted copy's directory: first one that exists, then one made */

static const char *const replica_filesystems[] = { "ext4", "xfs", "f2fs" };
/* A btrfs subvolume's st_dev is anonymous (major 0), so no block-device topology describes it: never searched. */
#define UNDESCRIBED_FILESYSTEM "btrfs"

static uint64_t now_nanoseconds(void)
{
	struct timespec now = { 0 };
	(void)clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
	       (uint64_t)now.tv_nsec;
}

/* The next mount point a replica may live on, of a /dev-backed ext4, xfs or f2fs filesystem not seen before (no vfat,
 * exfat, optical, squashfs or cluster filesystem is looked at or takes a slot; a btrfs one is passed over with one
 * product line); false at the end of the table. Only the mount table and a stat of the point. */
static bool next_mount_point(FILE *mounts, dev_t *seen, uint32_t *seen_count,
			     char point[REPLICA_PATH_CAPACITY])
{
	char line[MOUNT_LINE_CAPACITY];
	uint32_t examined = 0;
	while (examined++ < MOUNT_ENTRIES_MOST &&
	       fgets(line, sizeof(line), mounts) != NULL) {
		char device[REPLICA_PATH_CAPACITY];
		char type[REPLICA_PATH_CAPACITY];
		/* an escaped mount point (\040 for a space) is passed over */
		if (sscanf(line, "%4095s %4095s %4095s", device, point, type) !=
			    MOUNT_FIELDS ||
		    strncmp(device, MOUNT_DEVICE_PREFIX,
			    sizeof(MOUNT_DEVICE_PREFIX) - 1) != 0 ||
		    strncmp(point, BOOT_PREFIX, sizeof(BOOT_PREFIX) - 1) == 0 ||
		    strchr(point, '\\') != NULL)
			continue;
		if (strcmp(type, UNDESCRIBED_FILESYSTEM) == 0) {
			char shown[REPLICA_PATH_CAPACITY];
			local_device_display(shown, sizeof(shown), point);
			(void)fprintf(
				stderr,
				"storage: replica search passes over %s (btrfs): unknown_physical_topology\n",
				shown);
			continue;
		}
		bool known_type = false;
		for (size_t index = 0;
		     index < sizeof(replica_filesystems) /
				     sizeof(replica_filesystems[0]);
		     index++)
			known_type = known_type ||
				     strcmp(type, replica_filesystems[index]) ==
					     0;
		struct stat status;
		if (!known_type || stat(point, &status) != 0)
			continue;
		bool seen_before = false;
		for (uint32_t index = 0; index < *seen_count; index++)
			seen_before = seen_before ||
				      seen[index] == status.st_dev;
		if (seen_before || *seen_count == MOUNTS_MOST)
			continue;
		seen[(*seen_count)++] = status.st_dev;
		return true;
	}
	return false;
}

/* path = directory/name + suffix; false when it does not fit. */
static bool path_join(char path[REPLICA_PATH_CAPACITY], const char *directory,
		      const char *name, const char *suffix)
{
	const int length = snprintf(path, REPLICA_PATH_CAPACITY, "%s/%s%s",
				    directory, name, suffix);
	return length > 0 && length < REPLICA_PATH_CAPACITY;
}

bool replicas_receipt_text(char *output, size_t capacity,
			   const char *real_source, uint64_t byte_count,
			   uint64_t header_digest)
{
	const int length = snprintf(output, capacity, RECEIPT_FORMAT,
				    real_source, byte_count, header_digest);
	return length > 0 && (size_t)length < capacity;
}

/* The receipt a verified replica of this source has, word for word. */
static bool receipt_text(char text[RECEIPT_CAPACITY], const char *real_source,
			 const struct pack_header *header)
{
	return replicas_receipt_text(text, RECEIPT_CAPACITY, real_source,
				     header->file_byte_count,
				     header->header_digest);
}

static bool receipt_holds(const char *receipt_path, const char *expected)
{
	char found[RECEIPT_CAPACITY] = { 0 };
	FILE *const receipt = fopen(receipt_path, "re");
	if (receipt == NULL)
		return false;
	const size_t length = fread(found, 1, sizeof(found) - 1, receipt);
	(void)fclose(receipt);
	return length == strlen(expected) &&
	       memcmp(found, expected, length) == 0;
}

/* The directories a replica search reads on the filesystem mounted at point, in order: autotune's under HOME when
 * HOME is on that filesystem (storage_autotune.c places the root filesystem's replicas there, as a mount point owned by
 * root takes none), then REPLICA_DIRECTORY and LEGACY_REPLICA_DIRECTORY under the point. Returns their number. */
static uint32_t
replica_directories(const char *point,
		    char directories[REPLICA_LAYOUTS][REPLICA_PATH_CAPACITY])
{
	uint32_t count = 0;
	const char *const home = getenv("HOME");
	struct stat home_status, point_status;
	if (home != NULL && home[0] == '/' && stat(home, &home_status) == 0 &&
	    stat(point, &point_status) == 0 &&
	    home_status.st_dev == point_status.st_dev &&
	    path_join(directories[count], home, REPLICA_DIRECTORY, ""))
		count++;
	if (path_join(directories[count], point, REPLICA_DIRECTORY, ""))
		count++;
	if (path_join(directories[count], point, LEGACY_REPLICA_DIRECTORY, ""))
		count++;
	return count;
}

/* A given copy's drive can be compared: its description names its physical devices, or its filesystem is RAM-backed
 * (tmpfs, ramfs, by name), which shares no drive. */
static bool copy_comparable(const struct local_device_description *device)
{
	return (device->count != 0 &&
		device->count <= LOCAL_DEVICE_PHYSICAL_MOST) ||
	       strcmp(device->filesystem, "tmpfs") == 0 ||
	       strcmp(device->filesystem, "ramfs") == 0;
}

uint32_t replicas_find_mounts(FILE *mounts, const char *const *copies,
			      uint32_t copy_count,
			      const struct pack_header *header,
			      char (*paths)[REPLICA_PATH_CAPACITY],
			      uint32_t capacity, bool external)
{
	char real_source[REPLICA_PATH_CAPACITY];
	char expected[RECEIPT_CAPACITY];
	char name[REPLICA_PATH_CAPACITY];
	if (mounts == NULL || copies == NULL || copy_count == 0 ||
	    copy_count > STORAGE_MAXIMUM_COPIES || copies[0] == NULL ||
	    header == NULL || paths == NULL ||
	    realpath(copies[0], real_source) == NULL ||
	    !receipt_text(expected, real_source, header))
		return 0;
	(void)snprintf(name, sizeof(name), REPLICA_NAME_FORMAT,
		       header->header_digest);
	dev_t seen[MOUNTS_MOST];
	uint32_t seen_count = 0, found = 0;
	struct local_device_description selected[STORAGE_MAXIMUM_COPIES];
	uint32_t selected_count = 0;
	bool copies_described = false;
	char point[REPLICA_PATH_CAPACITY];
	while (found < capacity &&
	       next_mount_point(mounts, seen, &seen_count, point)) {
		/* Receipts are read only on media the policy admits (internal and not rotational, or external with the
		   override): a USB, SD, loop or virtual mount is described from sysfs and never probed. */
		struct local_device_description candidate;
		if (local_device_describe(point, &candidate) != STATUS_OK ||
		    local_device_media_refusal(&candidate, external) != NULL)
			continue;
		char directories[REPLICA_LAYOUTS][REPLICA_PATH_CAPACITY];
		char receipt[REPLICA_PATH_CAPACITY];
		const uint32_t layouts =
			replica_directories(point, directories);
		/* a receipt stands for its file only when that is a regular file (no link) of the source's size; one
		   that does not is stale: a bit per layout */
		bool receipted = false;
		uint32_t stale_layouts = 0;
		for (uint32_t layout = 0; !receipted && layout < layouts;
		     layout++) {
			struct stat replica;
			const bool holds =
				path_join(receipt, directories[layout], name,
					  RECEIPT_SUFFIX) &&
				path_join(paths[found], directories[layout],
					  name, "") &&
				receipt_holds(receipt, expected) &&
				strcmp(paths[found], real_source) != 0;
			receipted = holds &&
				    lstat(paths[found], &replica) == 0 &&
				    S_ISREG(replica.st_mode) &&
				    (uint64_t)replica.st_size ==
					    header->file_byte_count;
			stale_layouts |= holds && !receipted ? 1u << layout :
							       0u;
		}
		if (!receipted) {
			if (stale_layouts != 0)
				local_device_print(
					stderr, &candidate,
					"replica_missing_short_or_unverified",
					0, LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT,
					0, 0);
			continue;
		}
		/* The given copies are described once, for the first receipted replica (duplicates), each by the device
		   that holds its bytes: an EXL3 map's largest source, not the map file; the source by its real path.
		   A copy no topology describes (an anonymous st_dev: btrfs, ZFS) or whose medium names no drive
		   (overlay, a loop or virtual block device, FUSE, a network filesystem) could share any drive, so the
		   search ends with one line naming it and admits no replica; a RAM-backed copy shares none. */
		for (uint32_t copy = 0; !copies_described && copy < copy_count;
		     copy++) {
			char bytes_path[EXL3_MAP_PATH_BYTES];
			const char *const given = copy == 0 ? real_source :
							      copies[copy];
			if (given != NULL &&
			    local_device_describe(
				    storage_device_path(given, bytes_path),
				    &selected[selected_count]) == STATUS_OK &&
			    copy_comparable(&selected[selected_count])) {
				selected_count++;
				continue;
			}
			char shown[REPLICA_PATH_CAPACITY];
			local_device_display(shown, sizeof(shown),
					     given != NULL ? given : "");
			(void)fprintf(
				stderr,
				"storage: replica search ends: given copy %" PRIu32
				" %s is on no described drive, so no replica is admitted beside it: "
				"unknown_physical_topology\n",
				copy, shown);
			return 0;
		}
		copies_described = true;
		const char *refused = NULL;
		for (uint32_t index = 0;
		     refused == NULL && index < selected_count; index++)
			if (local_device_overlaps(&candidate, &selected[index]))
				refused = "duplicate_physical_device";
		if (refused == NULL && selected_count == STORAGE_MAXIMUM_COPIES)
			refused = "device_limit";
		local_device_print(stderr, &candidate,
				   refused != NULL ? refused : "eligible", 0,
				   LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, 0, 0);
		if (refused != NULL)
			continue;
		/* a stale receipt before the layout admitted on this mount is named too, never passed over silently */
		for (uint32_t layout = 0; stale_layouts >> layout != 0;
		     layout++)
			if ((stale_layouts >> layout & 1u) != 0 &&
			    path_join(receipt, directories[layout], name,
				      RECEIPT_SUFFIX))
				(void)fprintf(
					stderr,
					"storage: replica receipt %s passed over: "
					"replica_missing_short_or_unverified (a later layout on its mount is "
					"admitted)\n",
					receipt);
		selected[selected_count++] = candidate;
		found++;
	}
	return found;
}

uint32_t
replicas_reader_open(const struct storage_operations_record *operations,
		     const char *const *given, uint32_t given_count,
		     const char (*replicas)[REPLICA_PATH_CAPACITY],
		     uint32_t count, const char **copies_out,
		     struct storage_reader **reader_out, FILE *output)
{
	*reader_out = NULL;
	if (given_count == 0 || given_count > STORAGE_MAXIMUM_COPIES ||
	    count > STORAGE_MAXIMUM_COPIES - given_count)
		return 0;
	for (uint32_t copy = 0; copy < given_count; copy++)
		copies_out[copy] = given[copy];
	/* the full set first: when every replica opens, one open, as before; the greedy opens only on its refusal */
	for (uint32_t index = 0; index < count; index++)
		copies_out[given_count + index] = replicas[index];
	if (count != 0 &&
	    operations->reader_open(copies_out, given_count + count,
				    reader_out) == STATUS_OK)
		return count;
	*reader_out = NULL;
	uint32_t kept = 0;
	for (uint32_t index = 0; index < count; index++) {
		/* the replica on trial follows the given copies and those kept */
		const uint32_t trial = given_count + kept;
		copies_out[trial] = replicas[index];
		struct storage_reader *opened_reader = NULL;
		const result_status opened = operations->reader_open(
			copies_out, trial + 1u, &opened_reader);
		if (opened != STATUS_OK) {
			(void)fprintf(
				output,
				"turbo load: replica %s does not open beside the %" PRIu32
				" copies kept (%s): left out\n",
				replicas[index], trial, status_text(opened));
			continue;
		}
		if (*reader_out != NULL)
			operations->reader_close(*reader_out);
		*reader_out = opened_reader;
		kept++;
	}
	return kept;
}

uint32_t replicas_find_policy(const char *const *copies, uint32_t copy_count,
			      const struct pack_header *header,
			      char (*paths)[REPLICA_PATH_CAPACITY],
			      uint32_t capacity, bool external)
{
	FILE *const mounts = fopen("/proc/self/mounts", "re");
	if (mounts == NULL)
		return 0;
	const uint32_t found = replicas_find_mounts(
		mounts, copies, copy_count, header, paths, capacity, external);
	(void)fclose(mounts);
	return found;
}

uint32_t replicas_find(const char *source_path,
		       const struct pack_header *header,
		       char (*paths)[REPLICA_PATH_CAPACITY], uint32_t capacity)
{
	return replicas_find_policy(&source_path, 1u, header, paths, capacity,
				    false);
}

/* ---- the replicate command ---- */

struct verify_shared {
	const struct storage_reader *reader;
	const struct pack_extent *extents;
	uint32_t extent_count;
	pthread_mutex_t mutex; /* guards the four below */
	uint32_t next_extent;
	uint64_t next_byte; /* in next_extent */
	bool failed;
	uint64_t *digests; /* per extent: the sum of its read pieces' */
};

static void *verify_pieces(void *context)
{
	struct verify_shared *const shared = context;
	void *piece = NULL;
	if (posix_memalign(&piece, PACK_BLOCK_BYTES, PIECE_BYTES) != 0) {
		(void)pthread_mutex_lock(&shared->mutex);
		shared->failed = true;
		(void)pthread_mutex_unlock(&shared->mutex);
		return NULL;
	}
	for (;;) {
		(void)pthread_mutex_lock(&shared->mutex);
		while (shared->next_extent < shared->extent_count &&
		       shared->next_byte ==
			       shared->extents[shared->next_extent].byte_count) {
			shared->next_extent++;
			shared->next_byte = 0;
		}
		const bool taken = !shared->failed &&
				   shared->next_extent < shared->extent_count;
		const uint32_t extent = shared->next_extent;
		const uint64_t done = shared->next_byte;
		const uint64_t left =
			taken ? shared->extents[extent].byte_count - done : 0;
		const uint64_t bytes = left < PIECE_BYTES ? left : PIECE_BYTES;
		shared->next_byte += bytes;
		(void)pthread_mutex_unlock(&shared->mutex);
		if (!taken)
			break;
		const bool read =
			storage_direct.reader_read(
				shared->reader, 0,
				shared->extents[extent].byte_offset + done,
				piece, bytes) == STATUS_OK;
		const uint64_t digest =
			read ? pack_digest_of_piece(
				       piece, bytes,
				       done / PACK_DIGEST_WORD_BYTES) :
			       0;
		(void)pthread_mutex_lock(&shared->mutex);
		shared->digests[extent] += digest;
		shared->failed = shared->failed || !read;
		(void)pthread_mutex_unlock(&shared->mutex);
	}
	free(piece);
	return NULL;
}

/* Direct reads of the written replica: pack_open_entry checks its header and tables, then its header must be
 * the source's and every extent must have its digest. differing_out: the first extent whose digest differs, or
 * NO_EXTENT when none does (or the check failed before the digests: a header, a table or a read). */
static bool replica_verified(const char *path,
			     const struct pack_header *source_header,
			     uint32_t *differing_out)
{
	*differing_out = NO_EXTENT;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	if (storage_direct.reader_open(&path, 1, &reader) != STATUS_OK)
		return false;
	bool verified = pack_open_entry(&storage_direct, reader, &pack) ==
				STATUS_OK &&
			memcmp(pack_header_of(pack), source_header,
			       sizeof(*source_header)) == 0;
	struct verify_shared shared = {
		.reader = reader,
		.extents = verified ? pack_extents_of(pack) : NULL,
		.extent_count = verified ? source_header->extent_count : 0
	};
	shared.digests = verified ? calloc(shared.extent_count,
					   sizeof(shared.digests[0])) :
				    NULL;
	verified = verified && shared.digests != NULL &&
		   pthread_mutex_init(&shared.mutex, NULL) == 0;
	if (verified) {
		pthread_t threads[VERIFY_THREADS];
		uint32_t started = 0;
		while (started < VERIFY_THREADS &&
		       pthread_create(&threads[started], NULL, verify_pieces,
				      &shared) == 0)
			started++;
		(void)verify_pieces(&shared);
		for (uint32_t index = 0; index < started; index++)
			(void)pthread_join(threads[index], NULL);
		(void)pthread_mutex_destroy(&shared.mutex);
		verified = !shared.failed;
		for (uint32_t index = 0;
		     verified && index < shared.extent_count; index++) {
			verified = shared.digests[index] ==
				   shared.extents[index].digest;
			*differing_out = verified ? NO_EXTENT : index;
		}
		if (*differing_out != NO_EXTENT)
			(void)fprintf(
				stderr,
				"replica check %s: extent %" PRIu32 " (%" PRIu64
				" bytes at byte %" PRIu64
				") differs from the pack's digest\n",
				path, *differing_out,
				shared.extents[*differing_out].byte_count,
				shared.extents[*differing_out].byte_offset);
	}
	free(shared.digests);
	pack_close(pack);
	storage_direct.reader_close(reader);
	return verified;
}

/* Copies the source's bytes into a new file with direct writes, at most rate_mib MiB a second (0: no limit). */
static bool replica_copy(const struct storage_operations_record *storage,
			 const struct storage_reader *reader,
			 uint64_t byte_count, const char *path,
			 uint64_t rate_mib)
{
	void *piece = NULL;
	const int file = open(path,
			      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW |
				      O_DIRECT | O_CLOEXEC,
			      FILE_MODE);
	bool copied = file >= 0 &&
		      posix_memalign(&piece, PACK_BLOCK_BYTES, PIECE_BYTES) ==
			      0 &&
		      posix_fallocate(file, 0, (off_t)byte_count) == 0;
	const uint64_t started = now_nanoseconds();
	for (uint64_t done = 0; copied && done < byte_count;) {
		const uint64_t bytes = byte_count - done < PIECE_BYTES ?
					       byte_count - done :
					       PIECE_BYTES;
		copied = storage->reader_read(reader, 0, done, piece, bytes) ==
			 STATUS_OK;
		for (uint64_t written = 0; copied && written < bytes;) {
			const ssize_t count =
				pwrite(file, (unsigned char *)piece + written,
				       (size_t)(bytes - written),
				       (off_t)(done + written));
			copied = count > 0 || (count < 0 && errno == EINTR);
			written += count > 0 ? (uint64_t)count : 0;
		}
		done += bytes;
		const uint64_t due =
			rate_mib == 0 ?
				0 :
				done / (rate_mib * BYTES_PER_MIB) *
						NANOSECONDS_PER_SECOND +
					done % (rate_mib * BYTES_PER_MIB) *
						NANOSECONDS_PER_SECOND /
						(rate_mib * BYTES_PER_MIB);
		const uint64_t spent = now_nanoseconds() - started;
		if (spent < due) {
			const struct timespec pause = {
				.tv_sec = (time_t)((due - spent) /
						   NANOSECONDS_PER_SECOND),
				.tv_nsec = (long)((due - spent) %
						  NANOSECONDS_PER_SECOND)
			};
			(void)nanosleep(&pause, NULL);
		}
	}
	copied = copied && fsync(file) == 0;
	free(piece);
	if (file >= 0)
		copied = close(file) == 0 && copied;
	return copied;
}

/* Why the directory takes no replica of byte_count bytes, or NULL when it takes one. */
static const char *room_refusal(const char *directory, uint64_t byte_count,
				uint64_t floor_bytes, uint32_t floor_percent,
				bool external)
{
	struct local_device_description device;
	if (local_device_describe(directory, &device) != STATUS_OK)
		return "device_disappeared_or_unknown";
	const char *const refused =
		local_device_media_refusal(&device, external);
	return refused != NULL ?
		       refused :
		       local_device_space_refusal(device.total_bytes,
						  device.available_bytes,
						  byte_count, floor_percent,
						  floor_bytes, device.readonly);
}

static void policy_print(const char *directory, const char *reason,
			 uint64_t bytes, uint32_t floor_percent)
{
	struct local_device_description device = { 0 };
	const result_status described =
		local_device_describe(directory, &device);
	if (described != STATUS_OK)
		(void)snprintf(device.mount, sizeof(device.mount), "%s",
			       "unavailable");
	local_device_print(stderr, &device, reason, bytes, floor_percent, 0, 0);
}

static void partial_remove(const char *path)
{
	struct stat file;
	if (lstat(path, &file) == 0 && S_ISREG(file.st_mode) &&
	    file.st_uid == geteuid() && file.st_nlink == 1)
		(void)unlink(path);
}

/* --prune=1: the receipted replicas in directory of the same source but another header digest (the pack has changed
 * since) are removed, the replica before its receipt. */
static void stale_replicas_prune(const char *directory, const char *real_source,
				 const char *name)
{
	char prefix[RECEIPT_CAPACITY];
	const int prefix_length = snprintf(prefix, sizeof(prefix),
					   "anemone replica 1\nsource %s\n",
					   real_source);
	DIR *const entries = opendir(directory);
	if (entries == NULL || prefix_length <= 0 ||
	    prefix_length >= (int)sizeof(prefix)) {
		if (entries != NULL)
			(void)closedir(entries);
		return;
	}
	const size_t suffix_length = sizeof(RECEIPT_SUFFIX) - 1u;
	for (const struct dirent *entry = readdir(entries); entry != NULL;
	     entry = readdir(entries)) {
		const size_t length = strlen(entry->d_name);
		char receipt[REPLICA_PATH_CAPACITY];
		char replica[REPLICA_PATH_CAPACITY];
		char found[RECEIPT_CAPACITY] = { 0 };
		if (length <= suffix_length ||
		    strcmp(entry->d_name + length - suffix_length,
			   RECEIPT_SUFFIX) != 0 ||
		    strncmp(entry->d_name, name, strlen(name)) == 0 ||
		    !path_join(receipt, directory, entry->d_name, ""))
			continue;
		FILE *const file = fopen(receipt, "re");
		const size_t read =
			file != NULL ?
				fread(found, 1, sizeof(found) - 1, file) :
				0;
		if (file != NULL)
			(void)fclose(file);
		(void)snprintf(replica, sizeof(replica), "%.*s",
			       (int)(strlen(receipt) - suffix_length), receipt);
		if (read >= (size_t)prefix_length &&
		    memcmp(found, prefix, (size_t)prefix_length) == 0 &&
		    unlink(replica) == 0 && unlink(receipt) == 0)
			(void)printf(
				"replica %s: pruned (a replica of an older %s)\n",
				replica, real_source);
	}
	(void)closedir(entries);
}

/* One replica into directory: copied, verified, renamed, receipted. */
static int replicate_into(const char *directory, const char *real_source,
			  const struct storage_operations_record *storage,
			  const struct storage_reader *reader,
			  const struct pack_header *header, uint64_t rate_mib,
			  uint64_t floor_bytes, uint32_t floor_percent,
			  bool external, bool prune)
{
	char real_directory[REPLICA_PATH_CAPACITY];
	if (realpath(directory, real_directory) == NULL ||
	    (strstr(real_directory, "/anemone_engram_rnd/") == NULL &&
	     (strlen(real_directory) < strlen("/anemone_engram_rnd") ||
	      strcmp(real_directory + strlen(real_directory) -
			     strlen("/anemone_engram_rnd"),
		     "/anemone_engram_rnd") != 0))) {
		policy_print(directory, "destination_directory_policy", 0,
			     floor_percent);
		return 0;
	}
	char name[REPLICA_PATH_CAPACITY];
	char replica[REPLICA_PATH_CAPACITY];
	char partial[REPLICA_PATH_CAPACITY];
	char receipt[REPLICA_PATH_CAPACITY];
	char receipt_partial[REPLICA_PATH_CAPACITY];
	char text[RECEIPT_CAPACITY];
	(void)snprintf(name, sizeof(name), REPLICA_NAME_FORMAT,
		       header->header_digest);
	if (!path_join(replica, directory, name, "") ||
	    !path_join(partial, directory, name, REPLICA_PARTIAL_SUFFIX) ||
	    !path_join(receipt, directory, name, RECEIPT_SUFFIX) ||
	    !path_join(receipt_partial, directory, name,
		       RECEIPT_SUFFIX REPLICA_PARTIAL_SUFFIX) ||
	    !receipt_text(text, real_source, header)) {
		policy_print(directory, "destination_directory_policy", 0,
			     floor_percent);
		return false;
	}
	if (prune)
		stale_replicas_prune(directory, real_source, name);
	struct stat completed;
	if (receipt_holds(receipt, text) && stat(replica, &completed) == 0 &&
	    S_ISREG(completed.st_mode) && completed.st_size >= 0 &&
	    (uint64_t)completed.st_size == header->file_byte_count) {
		policy_print(directory, "eligible", 0, floor_percent);
		return true;
	}
	const char *const refusal =
		room_refusal(directory, header->file_byte_count, floor_bytes,
			     floor_percent, external);
	if (refusal != NULL) {
		partial_remove(partial);
		partial_remove(receipt_partial);
		policy_print(directory, refusal, header->file_byte_count,
			     floor_percent);
		return 0;
	}
	struct stat existing;
	if (lstat(replica, &existing) == 0) {
		partial_remove(partial);
		policy_print(directory, "unverified_final_exists", 0,
			     floor_percent);
		return 0;
	}
	partial_remove(partial);
	partial_remove(receipt_partial);
	(void)unlink(receipt);
	const uint64_t started = now_nanoseconds();
	if (!replica_copy(storage, reader, header->file_byte_count, partial,
			  rate_mib)) {
		const char *const changed = room_refusal(
			directory, 0, floor_bytes, floor_percent, external);
		partial_remove(partial);
		partial_remove(receipt_partial);
		policy_print(directory,
			     changed != NULL ? changed : "copy_io_error", 0,
			     floor_percent);
		return changed != NULL ? 0 : -1;
	}
	const uint64_t copied = now_nanoseconds();
	uint32_t differing = NO_EXTENT;
	if (!replica_verified(partial, header, &differing)) {
		partial_remove(partial);
		policy_print(directory, "verification_failed", 0,
			     floor_percent);
		return 0;
	}
	const uint64_t verified = now_nanoseconds();
	const char *const final_refusal = room_refusal(
		directory, 0, floor_bytes, floor_percent, external);
	if (final_refusal != NULL) {
		partial_remove(partial);
		partial_remove(receipt_partial);
		policy_print(directory, final_refusal, 0, floor_percent);
		return 0;
	}
	FILE *const receipt_file = fopen(receipt_partial, "we");
	const bool written =
		receipt_file != NULL && fputs(text, receipt_file) >= 0 &&
		fflush(receipt_file) == 0 && fsync(fileno(receipt_file)) == 0;
	if (receipt_file != NULL && fclose(receipt_file) != 0) {
		partial_remove(partial);
		partial_remove(receipt_partial);
		policy_print(directory, "copy_io_error", 0, floor_percent);
		return -1;
	}
	const char *const publish_refusal = room_refusal(
		directory, 0, floor_bytes, floor_percent, external);
	if (publish_refusal != NULL) {
		partial_remove(partial);
		partial_remove(receipt_partial);
		policy_print(directory, publish_refusal, 0, floor_percent);
		return 0;
	}
	const bool published = written &&
			       renameat2(AT_FDCWD, partial, AT_FDCWD, replica,
					 RENAME_NOREPLACE) == 0;
	const bool receipted = published &&
			       rename(receipt_partial, receipt) == 0;
	const int directory_descriptor =
		receipted ?
			open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC) :
			-1;
	const bool durable = directory_descriptor >= 0 &&
			     fsync(directory_descriptor) == 0;
	if (directory_descriptor >= 0)
		(void)close(directory_descriptor);
	if (!durable) {
		policy_print(directory, "copy_io_error", 0, floor_percent);
		partial_remove(partial);
		partial_remove(receipt_partial);
		if (published)
			partial_remove(replica);
		if (receipted)
			partial_remove(receipt);
		return -1;
	}
	policy_print(directory, "eligible", 0, floor_percent);
	(void)printf(
		"replica %s: %" PRIu64
		" bytes copied in %.1f s (%.2f GB/s), checked in %.1f s (%.2f GB/s)\n",
		replica, header->file_byte_count,
		(double)(copied - started) / NANOSECONDS_PER_SECOND,
		(double)header->file_byte_count / (double)(copied - started),
		(double)(verified - copied) / NANOSECONDS_PER_SECOND,
		(double)header->file_byte_count / (double)(verified - copied));
	return true;
}

/* Whether a and b name the same file as stat saw it: device, inode, size and both change times. */
static bool same_file_state(const struct stat *a, const struct stat *b)
{
	return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
	       a->st_size == b->st_size &&
	       a->st_mtim.tv_sec == b->st_mtim.tv_sec &&
	       a->st_mtim.tv_nsec == b->st_mtim.tv_nsec &&
	       a->st_ctim.tv_sec == b->st_ctim.tv_sec &&
	       a->st_ctim.tv_nsec == b->st_ctim.tv_nsec;
}

/* The search directory an adopted copy is linked into: the first of replica_directories(point) that is a directory
 * on the copy's filesystem, else the first that can be made there (its parent made too). False when none. */
static bool adopt_directory(const char *point, const struct stat *copy,
			    char directory[REPLICA_PATH_CAPACITY])
{
	char directories[REPLICA_LAYOUTS][REPLICA_PATH_CAPACITY] = { { 0 } };
	const uint32_t count = replica_directories(point, directories);
	struct stat found;
	for (uint32_t pass = 0; pass < ADOPT_DIRECTORY_PASSES; pass++)
		for (uint32_t layout = 0; layout < count; layout++) {
			char container[REPLICA_PATH_CAPACITY];
			memcpy(container, directories[layout],
			       sizeof(container)); /* one whole row, its NUL within */
			char *const slash = strrchr(container, '/');
			if (pass + 1u == ADOPT_DIRECTORY_PASSES &&
			    slash != NULL) {
				*slash = '\0';
				(void)mkdir(container, DIRECTORY_MODE);
				(void)mkdir(directories[layout],
					    DIRECTORY_MODE);
			}
			if (stat(directories[layout], &found) == 0 &&
			    S_ISDIR(found.st_mode) &&
			    found.st_dev == copy->st_dev) {
				memcpy(directory, directories[layout],
				       REPLICA_PATH_CAPACITY);
				return true;
			}
		}
	return false;
}

/* The receipt of a replica in directory, written beside it under a partial name, synced, renamed and the directory
 * synced: a search finds a whole receipt or none. */
static bool receipt_publish(const char *directory, const char *name,
			    const char *text)
{
	char receipt[REPLICA_PATH_CAPACITY], partial[REPLICA_PATH_CAPACITY];
	if (!path_join(receipt, directory, name, RECEIPT_SUFFIX) ||
	    !path_join(partial, directory, name,
		       RECEIPT_SUFFIX REPLICA_PARTIAL_SUFFIX))
		return false;
	partial_remove(partial);
	FILE *const file = fopen(partial, "wxe");
	bool written = file != NULL && fputs(text, file) >= 0 &&
		       fflush(file) == 0 && fsync(fileno(file)) == 0;
	written = file != NULL && fclose(file) == 0 && written;
	const bool renamed = written && rename(partial, receipt) == 0;
	const int descriptor =
		renamed ? open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC) :
			  -1;
	const bool durable = descriptor >= 0 && fsync(descriptor) == 0;
	if (descriptor >= 0)
		(void)close(descriptor);
	if (!renamed)
		partial_remove(partial);
	return durable;
}

enum replica_adoption replicas_adopt(const char *path, const char *point,
				     const char *real_source,
				     const struct pack_header *header,
				     uint32_t *differing_out)
{
	*differing_out = NO_EXTENT;
	char name[REPLICA_PATH_CAPACITY], directory[REPLICA_PATH_CAPACITY],
		replica[REPLICA_PATH_CAPACITY], text[RECEIPT_CAPACITY],
		real_path[REPLICA_PATH_CAPACITY];
	struct stat before, after, linked;
	(void)snprintf(name, sizeof(name), REPLICA_NAME_FORMAT,
		       header->header_digest);
	const char *refused = NULL;
	if (lstat(path, &before) != 0 || !S_ISREG(before.st_mode) ||
	    before.st_uid != geteuid() || realpath(path, real_path) == NULL)
		refused = "adopt_not_an_owned_regular_file";
	else if (before.st_size < 0 ||
		 (uint64_t)before.st_size != header->file_byte_count)
		refused = "adopt_size_differs";
	else if (strcmp(real_path, real_source) == 0)
		refused = "adopt_is_the_source";
	else if (!adopt_directory(point, &before, directory) ||
		 !path_join(replica, directory, name, "") ||
		 !receipt_text(text, real_source, header))
		refused = "destination_directory_policy";
	if (refused != NULL) {
		policy_print(path, refused, 0,
			     LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT);
		return REPLICA_ADOPT_REFUSED;
	}
	char receipt[REPLICA_PATH_CAPACITY];
	/* already adopted: the search's name is this file and its receipt holds */
	if (lstat(replica, &linked) == 0 && linked.st_dev == before.st_dev &&
	    linked.st_ino == before.st_ino &&
	    path_join(receipt, directory, name, RECEIPT_SUFFIX) &&
	    receipt_holds(receipt, text)) {
		(void)printf("replica %s: %s already adopted\n", replica, path);
		return REPLICA_ADOPTED;
	}
	const uint64_t started = now_nanoseconds();
	/* every block read and checked before anything is written; the file must not change while it is read */
	const bool verified = replica_verified(path, header, differing_out) &&
			      lstat(path, &after) == 0 &&
			      same_file_state(&before, &after);
	if (!verified) {
		policy_print(path, "verification_failed", 0,
			     LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT);
		return REPLICA_ADOPT_FAILED;
	}
	const uint64_t checked = now_nanoseconds();
	const bool placed = link(path, replica) == 0 ||
			    (errno == EEXIST && lstat(replica, &linked) == 0 &&
			     linked.st_dev == before.st_dev &&
			     linked.st_ino == before.st_ino);
	const int placed_error = errno;
	/* the name now holds the inode that was read, unchanged (a link only moves its change time) */
	const bool same = placed && lstat(replica, &linked) == 0 &&
			  linked.st_dev == before.st_dev &&
			  linked.st_ino == before.st_ino &&
			  linked.st_size == before.st_size &&
			  linked.st_mtim.tv_sec == before.st_mtim.tv_sec &&
			  linked.st_mtim.tv_nsec == before.st_mtim.tv_nsec;
	if (!same) {
		if (placed)
			(void)unlink(replica);
		policy_print(replica,
			     !placed && placed_error == EEXIST ?
				     "unverified_final_exists" :
				     "copy_io_error",
			     0, LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT);
		return REPLICA_ADOPT_FAILED;
	}
	if (!receipt_publish(directory, name, text)) {
		policy_print(replica, "copy_io_error", 0,
			     LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT);
		return REPLICA_ADOPT_FAILED;
	}
	(void)printf("replica %s: adopted %s, %" PRIu64
		     " bytes checked in %.1f s (%.2f GB/s)\n",
		     replica, path, header->file_byte_count,
		     (double)(checked - started) / NANOSECONDS_PER_SECOND,
		     (double)header->file_byte_count /
			     (double)(checked - started));
	return REPLICA_ADOPTED;
}

static bool number_option(const char *argument, const char *name,
			  uint64_t *value)
{
	const size_t length = strlen(name);
	char *end = NULL;
	if (strncmp(argument, name, length) != 0 || argument[length] == '\0')
		return false;
	errno = 0;
	const unsigned long long number =
		strtoull(argument + length, &end, DECIMAL);
	if (errno != 0 || *end != '\0' || argument[length] == '-')
		return false;
	*value = number;
	return true;
}

/* name=TEXT: true, and value is TEXT (not empty) */
static bool text_option(const char *argument, const char *name,
			const char **value)
{
	const size_t length = strlen(name);
	if (strncmp(argument, name, length) != 0 || argument[length] == '\0')
		return false;
	*value = argument + length;
	return true;
}

int replicate_cli(int argument_count, char **arguments)
{
	uint64_t rate_mib = REPLICA_DEFAULT_RATE_MIB;
	uint64_t floor_gib = REPLICA_DEFAULT_FLOOR_GIB;
	uint64_t prune = 0, external = 0,
		 floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT;
	const char *adopt = NULL;
	int index = FIRST_OPTION;
	while (index < argument_count &&
	       (text_option(arguments[index], ADOPT_OPTION, &adopt) ||
		number_option(arguments[index], "--rate_mib=", &rate_mib) ||
		number_option(arguments[index], "--floor_gib=", &floor_gib) ||
		number_option(arguments[index], "--prune=", &prune) ||
		number_option(arguments[index],
			      "--replica_external=", &external) ||
		number_option(arguments[index],
			      "--replica_floor_percent=", &floor_percent)))
		index++;
	if (index >= argument_count || arguments[index][0] == '-' ||
	    (adopt != NULL && index + 1 != argument_count) ||
	    floor_gib > UINT64_MAX / BYTES_PER_GIB || prune > 1u ||
	    external > 1u || floor_percent >= PERCENT ||
	    rate_mib > UINT64_MAX / BYTES_PER_MIB / NANOSECONDS_PER_SECOND) {
		(void)fprintf(
			stderr,
			"usage: anemone replicate [--rate_mib=N (default %u; 0: no limit)]"
			" [--floor_gib=N (default %u)] [--replica_floor_percent=0..99 (default20)]"
			" [--replica_external=0|1 (default0)] [--prune=0|1 (1: remove the receipted"
			" replicas of an older SOURCE first)] SOURCE [DIRECTORY ...]\n"
			"       anemone replicate --adopt=COPY [--replica_external=0|1] SOURCE (an existing full"
			" copy of SOURCE on another drive: read and checked in full, then linked into its drive's"
			" replica directory and receipted)\n",
			REPLICA_DEFAULT_RATE_MIB, REPLICA_DEFAULT_FLOOR_GIB);
		return REPLICATE_USAGE;
	}
	const char *const source = arguments[index++];
	char real_source[REPLICA_PATH_CAPACITY];
	const struct storage_operations_record *const storage =
		storage_for_path(source);
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct stat source_status;
	if (realpath(source, real_source) == NULL ||
	    stat(real_source, &source_status) != 0 ||
	    storage->reader_open(&source, 1, &reader) != STATUS_OK ||
	    pack_open_entry(storage, reader, &pack) != STATUS_OK) {
		(void)fprintf(stderr, "replicate: %s does not open as a pack\n",
			      source);
		storage->reader_close(reader);
		return REPLICATE_FAILED;
	}
	const struct pack_header *const header = pack_header_of(pack);
	uint32_t made = 0, wanted = 0;
	uint64_t selected[MOUNTS_MOST * LOCAL_DEVICE_PHYSICAL_MOST];
	uint32_t selected_count = 0;
	struct local_device_description source_device;
	char bytes_path[EXL3_MAP_PATH_BYTES];
	if (local_device_describe(storage_device_path(real_source, bytes_path),
				  &source_device) == STATUS_OK)
		for (uint32_t leaf = 0; leaf < source_device.count; leaf++)
			selected[selected_count++] =
				source_device.physical[leaf];
	if (adopt != NULL) {
		struct local_device_description device;
		const char *refused =
			local_device_describe(adopt, &device) != STATUS_OK ?
				"device_disappeared_or_unknown" :
				local_device_media_refusal(&device,
							   external != 0);
		for (uint32_t previous = 0;
		     refused == NULL && previous < selected_count; previous++)
			for (uint32_t leaf = 0; leaf < device.count; leaf++)
				if (selected[previous] == device.physical[leaf])
					refused = "duplicate_physical_device";
		uint32_t differing = NO_EXTENT;
		const enum replica_adoption adopted =
			refused != NULL ?
				REPLICA_ADOPT_REFUSED :
				replicas_adopt(adopt, device.mount, real_source,
					       header, &differing);
		if (refused != NULL)
			policy_print(adopt, refused, 0,
				     (uint32_t)floor_percent);
		pack_close(pack);
		storage->reader_close(reader);
		return adopted == REPLICA_ADOPTED ? 0 : REPLICATE_FAILED;
	}
	FILE *mounts = index < argument_count ?
			       NULL :
			       fopen("/proc/self/mounts", "re");
	dev_t seen[MOUNTS_MOST];
	uint32_t seen_count = 0;
	char point[REPLICA_PATH_CAPACITY];
	while (index < argument_count ||
	       (mounts != NULL &&
		next_mount_point(mounts, seen, &seen_count, point))) {
		char directory[REPLICA_PATH_CAPACITY];
		const bool explicit_directory = index < argument_count;
		const char *parent = explicit_directory ? arguments[index++] :
							  point;
		struct local_device_description candidate;
		const char *refused =
			local_device_describe(parent, &candidate) != STATUS_OK ?
				"device_disappeared_or_unknown" :
				local_device_media_refusal(&candidate,
							   external != 0);
		for (uint32_t previous = 0;
		     refused == NULL && previous < selected_count; previous++)
			for (uint32_t leaf = 0; leaf < candidate.count; leaf++)
				if (selected[previous] ==
				    candidate.physical[leaf])
					refused = "duplicate_physical_device";
		if (refused == NULL &&
		    selected_count + candidate.count >
			    MOUNTS_MOST * LOCAL_DEVICE_PHYSICAL_MOST)
			refused = "device_limit";
		if (refused == NULL)
			refused = room_refusal(parent, 0,
					       floor_gib * BYTES_PER_GIB,
					       (uint32_t)floor_percent,
					       external != 0);
		if (refused != NULL) {
			char refused_directory[REPLICA_PATH_CAPACITY],
				name[REPLICA_PATH_CAPACITY],
				partial[REPLICA_PATH_CAPACITY];
			(void)snprintf(name, sizeof(name), REPLICA_NAME_FORMAT,
				       header->header_digest);
			if ((explicit_directory ?
				     snprintf(refused_directory,
					      sizeof(refused_directory), "%s",
					      parent) > 0 :
				     path_join(refused_directory, parent,
					       REPLICA_DIRECTORY, "")) &&
			    path_join(partial, refused_directory, name,
				      REPLICA_PARTIAL_SUFFIX))
				partial_remove(partial);
			policy_print(parent, refused, header->file_byte_count,
				     (uint32_t)floor_percent);
			continue;
		}
		if (explicit_directory) {
			(void)snprintf(directory, sizeof(directory), "%s",
				       parent);
		} else {
			char container[REPLICA_PATH_CAPACITY];
			if (!path_join(container, point, "anemone_engram_rnd",
				       "") ||
			    !path_join(directory, point, REPLICA_DIRECTORY, ""))
				continue;
			(void)mkdir(container, DIRECTORY_MODE);
			(void)mkdir(directory, DIRECTORY_MODE);
		}
		const int outcome = replicate_into(
			directory, real_source, storage, reader, header,
			rate_mib, floor_gib * BYTES_PER_GIB,
			(uint32_t)floor_percent, external != 0, prune != 0);
		if (outcome != 0)
			wanted++;
		if (outcome > 0) {
			made++;
			for (uint32_t leaf = 0; leaf < candidate.count; leaf++)
				selected[selected_count++] =
					candidate.physical[leaf];
		}
	}
	if (mounts != NULL)
		(void)fclose(mounts);
	(void)printf("replicate %s: %" PRIu32 " of %" PRIu32
		     " replicas present\n",
		     real_source, made, wanted);
	pack_close(pack);
	storage->reader_close(reader);
	return made == wanted ? 0 : REPLICATE_FAILED;
}
