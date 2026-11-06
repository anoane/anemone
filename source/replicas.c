/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* replicas.c: turbo load's replicas of a pack (replicas.h): the replicate command and the search of a load. */
#include "anemone/replicas.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#include "anemone/pack_digest.h"
#include "anemone/storage_exl3.h"

#define REPLICA_DIRECTORY "anemone/replicas" /* under a mount point */
#define REPLICA_NAME_FORMAT \
	"%016" PRIx64 ".anemone" /* the pack's header digest */
#define PARTIAL_SUFFIX ".partial"
#define RECEIPT_SUFFIX ".receipt"
#define RECEIPT_FORMAT                                 \
	"anemone replica 1\nsource %s\nbytes %" PRIu64 \
	"\nheader_digest %016" PRIx64 "\n"
#define RECEIPT_CAPACITY (REPLICA_PATH_CAPACITY + 256)
#define MOUNT_LINE_CAPACITY (2 * REPLICA_PATH_CAPACITY)
#define MOUNT_DEVICE_PREFIX "/dev/"
#define BOOT_PREFIX "/boot"
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

static const char *const replica_filesystems[] = { "ext4", "xfs", "btrfs",
						   "f2fs" };

static uint64_t now_nanoseconds(void)
{
	struct timespec now = { 0 };
	(void)clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
	       (uint64_t)now.tv_nsec;
}

/* The next mount point a replica may live on, of a filesystem not seen before; false at the end of the table. */
static bool next_mount_point(FILE *mounts, dev_t *seen, uint32_t *seen_count,
			     char point[REPLICA_PATH_CAPACITY])
{
	char line[MOUNT_LINE_CAPACITY];
	while (fgets(line, sizeof(line), mounts) != NULL) {
		char device[REPLICA_PATH_CAPACITY];
		char type[REPLICA_PATH_CAPACITY];
		/* an escaped mount point (\040 for a space) is passed over */
		if (sscanf(line, "%4095s %4095s %4095s", device, point, type) !=
			    3 ||
		    strncmp(device, MOUNT_DEVICE_PREFIX,
			    sizeof(MOUNT_DEVICE_PREFIX) - 1) != 0 ||
		    strncmp(point, BOOT_PREFIX, sizeof(BOOT_PREFIX) - 1) == 0 ||
		    strchr(point, '\\') != NULL)
			continue;
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

/* The receipt a verified replica of this source has, word for word. */
static bool receipt_text(char text[RECEIPT_CAPACITY], const char *real_source,
			 const struct pack_header *header)
{
	const int length = snprintf(text, RECEIPT_CAPACITY, RECEIPT_FORMAT,
				    real_source, header->file_byte_count,
				    header->header_digest);
	return length > 0 && length < RECEIPT_CAPACITY;
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

uint32_t replicas_find(const char *source_path,
		       const struct pack_header *header,
		       char (*paths)[REPLICA_PATH_CAPACITY], uint32_t capacity)
{
	char real_source[REPLICA_PATH_CAPACITY];
	char expected[RECEIPT_CAPACITY];
	char name[REPLICA_PATH_CAPACITY];
	if (source_path == NULL || header == NULL || paths == NULL ||
	    realpath(source_path, real_source) == NULL ||
	    !receipt_text(expected, real_source, header))
		return 0;
	(void)snprintf(name, sizeof(name), REPLICA_NAME_FORMAT,
		       header->header_digest);
	FILE *const mounts = fopen("/proc/self/mounts", "re");
	if (mounts == NULL)
		return 0;
	dev_t seen[MOUNTS_MOST];
	uint32_t seen_count = 0, found = 0;
	char point[REPLICA_PATH_CAPACITY];
	while (found < capacity &&
	       next_mount_point(mounts, seen, &seen_count, point)) {
		char directory[REPLICA_PATH_CAPACITY];
		char receipt[REPLICA_PATH_CAPACITY];
		if (path_join(directory, point, REPLICA_DIRECTORY, "") &&
		    path_join(receipt, directory, name, RECEIPT_SUFFIX) &&
		    path_join(paths[found], directory, name, "") &&
		    receipt_holds(receipt, expected) &&
		    strcmp(paths[found], real_source) != 0)
			found++;
	}
	(void)fclose(mounts);
	return found;
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
 * the source's and every extent must have its digest. */
static bool replica_verified(const char *path,
			     const struct pack_header *source_header)
{
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
		     verified && index < shared.extent_count; index++)
			verified = shared.digests[index] ==
				   shared.extents[index].digest;
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
	const int file =
		open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC,
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
				uint64_t floor_bytes)
{
	struct statvfs filesystem;
	if (statvfs(directory, &filesystem) != 0)
		return "its filesystem cannot be measured";
	const uint64_t block = filesystem.f_frsize;
	const uint64_t total = (uint64_t)filesystem.f_blocks * block;
	const uint64_t available = (uint64_t)filesystem.f_bavail * block;
	const uint64_t used = total - (uint64_t)filesystem.f_bfree * block;
	return available < byte_count + floor_bytes ?
		       "the free-space floor would be crossed" :
	       (used + byte_count) * PERCENT >
			       total * REPLICA_MOST_USED_PERCENT ?
		       "its filesystem would be more than 90% full" :
		       NULL;
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
static bool replicate_into(const char *directory, const char *real_source,
			   const struct storage_operations_record *storage,
			   const struct storage_reader *reader,
			   const struct pack_header *header, uint64_t rate_mib,
			   uint64_t floor_bytes, bool prune)
{
	char name[REPLICA_PATH_CAPACITY];
	char replica[REPLICA_PATH_CAPACITY];
	char partial[REPLICA_PATH_CAPACITY];
	char receipt[REPLICA_PATH_CAPACITY];
	char receipt_partial[REPLICA_PATH_CAPACITY];
	char text[RECEIPT_CAPACITY];
	(void)snprintf(name, sizeof(name), REPLICA_NAME_FORMAT,
		       header->header_digest);
	if (!path_join(replica, directory, name, "") ||
	    !path_join(partial, directory, name, PARTIAL_SUFFIX) ||
	    !path_join(receipt, directory, name, RECEIPT_SUFFIX) ||
	    !path_join(receipt_partial, directory, name,
		       RECEIPT_SUFFIX PARTIAL_SUFFIX) ||
	    !receipt_text(text, real_source, header)) {
		(void)printf("replica %s: skipped: the path is too long\n",
			     directory);
		return false;
	}
	if (prune)
		stale_replicas_prune(directory, real_source, name);
	if (receipt_holds(receipt, text)) {
		(void)printf("replica %s: present and receipted\n", replica);
		return true;
	}
	const char *const refusal =
		room_refusal(directory, header->file_byte_count, floor_bytes);
	if (refusal != NULL) {
		(void)printf("replica %s: skipped: %s\n", directory, refusal);
		return false;
	}
	(void)unlink(receipt);
	const uint64_t started = now_nanoseconds();
	if (!replica_copy(storage, reader, header->file_byte_count, partial,
			  rate_mib)) {
		(void)printf("replica %s: the copy failed (%s): it stays %s\n",
			     replica, strerror(errno), partial);
		return false;
	}
	const uint64_t copied = now_nanoseconds();
	if (!replica_verified(partial, header)) {
		(void)printf("replica %s: the check failed: it stays %s\n",
			     replica, partial);
		return false;
	}
	const uint64_t verified = now_nanoseconds();
	FILE *const receipt_file = fopen(receipt_partial, "we");
	const bool written =
		receipt_file != NULL && fputs(text, receipt_file) >= 0 &&
		fflush(receipt_file) == 0 && fsync(fileno(receipt_file)) == 0;
	if (receipt_file != NULL && fclose(receipt_file) != 0)
		return false;
	if (!written || rename(partial, replica) != 0 ||
	    rename(receipt_partial, receipt) != 0) {
		(void)printf("replica %s: no receipt (%s)\n", replica,
			     strerror(errno));
		return false;
	}
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

int replicate_cli(int argument_count, char **arguments)
{
	uint64_t rate_mib = REPLICA_DEFAULT_RATE_MIB;
	uint64_t floor_gib = REPLICA_DEFAULT_FLOOR_GIB;
	uint64_t prune = 0;
	int index = 2;
	while (index < argument_count &&
	       (number_option(arguments[index], "--rate_mib=", &rate_mib) ||
		number_option(arguments[index], "--floor_gib=", &floor_gib) ||
		number_option(arguments[index], "--prune=", &prune)))
		index++;
	if (index >= argument_count || arguments[index][0] == '-' ||
	    floor_gib > UINT64_MAX / BYTES_PER_GIB || prune > 1u ||
	    rate_mib > UINT64_MAX / BYTES_PER_MIB / NANOSECONDS_PER_SECOND) {
		(void)fprintf(
			stderr,
			"usage: anemone replicate [--rate_mib=N (default %u; 0: no limit)]"
			" [--floor_gib=N (default %u)] [--prune=0|1 (1: remove the receipted"
			" replicas of an older SOURCE first)] SOURCE [DIRECTORY ...]\n",
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
	if (index < argument_count) {
		for (; index < argument_count; index++, wanted++)
			made += replicate_into(arguments[index], real_source,
					       storage, reader, header,
					       rate_mib,
					       floor_gib * BYTES_PER_GIB,
					       prune != 0) ?
					1u :
					0u;
	} else {
		/* every other drive: <mount>/anemone/replicas, made only where a replica fits */
		FILE *const mounts = fopen("/proc/self/mounts", "re");
		dev_t seen[MOUNTS_MOST] = { source_status.st_dev };
		uint32_t seen_count = 1;
		char point[REPLICA_PATH_CAPACITY];
		while (mounts != NULL &&
		       next_mount_point(mounts, seen, &seen_count, point)) {
			char parent[REPLICA_PATH_CAPACITY];
			char directory[REPLICA_PATH_CAPACITY];
			if (!path_join(parent, point, "anemone", "") ||
			    !path_join(directory, point, REPLICA_DIRECTORY, ""))
				continue;
			const char *const refusal =
				access(directory, W_OK) == 0 ?
					NULL :
					room_refusal(point,
						     header->file_byte_count,
						     floor_gib * BYTES_PER_GIB);
			if (refusal != NULL) {
				(void)printf("replica %s: skipped: %s\n", point,
					     refusal);
				continue;
			}
			(void)mkdir(parent, DIRECTORY_MODE);
			(void)mkdir(directory, DIRECTORY_MODE);
			if (access(directory, W_OK) != 0)
				continue;
			wanted++;
			made += replicate_into(directory, real_source, storage,
					       reader, header, rate_mib,
					       floor_gib * BYTES_PER_GIB,
					       prune != 0) ?
					1u :
					0u;
		}
		if (mounts != NULL)
			(void)fclose(mounts);
	}
	(void)printf("replicate %s: %" PRIu32 " of %" PRIu32
		     " replicas present\n",
		     real_source, made, wanted);
	pack_close(pack);
	storage->reader_close(reader);
	return made == wanted ? 0 : REPLICATE_FAILED;
}
