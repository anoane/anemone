/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Adopting existing copies of a pack as turbo load replicas (replicas.h, replicas_adopt). The source pack lies on
 * /dev/shm and its copies on /var/tmp (direct reads need a real disk; the search admits a copy only on another
 * device than the source's), each copy under a fixture mount point: a copy is read in full and checked before it gets
 * its search name (a hard link) and its receipt, and the replica search then finds it. A copy with one rotten byte,
 * a truncated one, a symbolic link, the source itself and a name already taken by another file get neither. */
#include <ftw.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "engine/storage/io/local_device.h"
#include "engine/formats/pack/pack.h"
#include "engine/storage/replicas/replicas.h"
#include "support/memory_storage.h"
#include "support/pack_builder.h"

#define FIXTURE_MODE 0700
#define FIXTURE_SEED 0x5EED0ADDull
#define FIXTURE_TENSORS 2u
#define FIXTURE_NAME_BYTES \
	64u /* a replica or receipt name below its directory */
#define FIXTURE_PATH_BYTES (PATH_MAX + FIXTURE_NAME_BYTES)
#define TABLE_LINE_BYTES 64u /* a fixture mount line besides its mount point */
#define RECEIPT_BYTES (PATH_MAX + 256u)
#define ROTTEN_EXTENT 2u /* the extent of the rotten byte */
#define ROTTEN_BYTE 4097u /* in that extent: its second block */
#define NO_EXTENT UINT32_MAX
#define FIXTURE_DEPTH 32 /* nftw's open directories */
#define SEARCH_DIRECTORY "anemone_engram_rnd/replicas"
#define LEGACY_DIRECTORY "anemone/replicas"

static const struct test_pack_unit_plan unit_plans[] = {
	{ PLACEMENT_UNIT_STAGE, 0, 2, 2 },
	{ PLACEMENT_UNIT_STAGE, 1, 1, 3 },
	{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, 2 }
};

static unsigned failures;
static void expect(bool passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static bool joined(char path[FIXTURE_PATH_BYTES], const char *left,
		   const char *right)
{
	const int length =
		snprintf(path, FIXTURE_PATH_BYTES, "%s/%s", left, right);
	return length > 0 && (size_t)length < FIXTURE_PATH_BYTES;
}

static bool written_bytes(const char *path, const void *bytes, size_t count)
{
	FILE *const file = fopen(path, "we");
	if (file == NULL)
		return false;
	const bool good = fwrite(bytes, 1, count, file) == count;
	return fclose(file) == 0 && good;
}

static bool exists(const char *path)
{
	struct stat status;
	return lstat(path, &status) == 0;
}

static bool same_inode(const char *first, const char *second)
{
	struct stat a, b;
	return stat(first, &a) == 0 && stat(second, &b) == 0 &&
	       a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

static bool receipt_is(const char *path, const char *expected)
{
	char found[RECEIPT_BYTES] = { 0 };
	FILE *const file = fopen(path, "re");
	if (file == NULL)
		return false;
	const size_t length = fread(found, 1, sizeof(found) - 1u, file);
	(void)fclose(file);
	return length == strlen(expected) &&
	       memcmp(found, expected, length) == 0;
}

/* One entry of a fixture tree, removed after its children (nftw walks the tree; nothing here recurses). */
static int fixture_entry_remove(const char *path, const struct stat *file,
				int kind, struct FTW *walk)
{
	(void)file;
	(void)kind;
	(void)walk;
	return remove(path);
}

static bool fixture_remove(const char *path)
{
	return nftw(path, fixture_entry_remove, FIXTURE_DEPTH,
		    FTW_DEPTH | FTW_PHYS) == 0;
}

/* The replicas the search finds on one fixture mount (a /dev line over point) for the source. */
static uint32_t search(const char *root, const char *point,
		       const char *real_source,
		       const struct pack_header *header,
		       char (*found)[REPLICA_PATH_CAPACITY])
{
	char table[FIXTURE_PATH_BYTES], line[PATH_MAX + TABLE_LINE_BYTES];
	const int length = snprintf(line, sizeof(line),
				    "/dev/fixture0 %s ext4 rw 0 0\n", point);
	FILE *stream = NULL;
	if (joined(table, root, "mounts") && length > 0 &&
	    (size_t)length < sizeof(line)) {
		stream = fopen(table, "w+e");
		if (stream != NULL && (fputs(line, stream) < 0 ||
				       fseek(stream, 0, SEEK_SET) != 0)) {
			(void)fclose(stream);
			stream = NULL;
		}
	}
	const uint32_t count =
		stream != NULL ?
			replicas_find_mounts(stream, &real_source, 1u, header,
					     found, REPLICAS_MOST, false) :
			0;
	if (stream != NULL)
		(void)fclose(stream);
	return count;
}

int main(void)
{
	char root[] = "/var/tmp/anemone-replica-adopt-XXXXXX",
	     shared[] = "/dev/shm/anemone-replica-adopt-XXXXXX";
	const bool have_root = mkdtemp(root) != NULL;
	const bool have_shared = mkdtemp(shared) != NULL;
	const struct test_pack_plan plan = {
		.model_name = "replica-adopt-test",
		.units = unit_plans,
		.unit_count = sizeof(unit_plans) / sizeof(unit_plans[0]),
		.tensors_per_extent = FIXTURE_TENSORS,
		.content_seed = FIXTURE_SEED
	};
	void *bytes = NULL;
	uint64_t byte_count = 0;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	const char *const memory_path = "memory";
	bool prepared = have_root && have_shared &&
			test_pack_build(&plan, &bytes, &byte_count) ==
				STATUS_OK;
	if (prepared) {
		test_memory_storage_set_pack(bytes, byte_count);
		prepared = test_memory_storage.reader_open(
				   &memory_path, 1, &reader) == STATUS_OK &&
			   pack_open_entry(&test_memory_storage, reader,
					   &pack) == STATUS_OK &&
			   pack_header_of(pack)->extent_count > ROTTEN_EXTENT;
	}
	const struct pack_header header = prepared && pack != NULL ?
						  *pack_header_of(pack) :
						  (struct pack_header){ 0 };
	const uint64_t rotten =
		prepared && pack != NULL ?
			pack_extents_of(pack)[ROTTEN_EXTENT].byte_offset +
				ROTTEN_BYTE :
			0;
	char source[FIXTURE_PATH_BYTES], real_source[PATH_MAX],
		home[FIXTURE_PATH_BYTES], point[FIXTURE_PATH_BYTES],
		old[FIXTURE_PATH_BYTES], copy[FIXTURE_PATH_BYTES],
		legacy_point[FIXTURE_PATH_BYTES], legacy[FIXTURE_PATH_BYTES],
		legacy_copy[FIXTURE_PATH_BYTES],
		rotten_copy[FIXTURE_PATH_BYTES], short_copy[FIXTURE_PATH_BYTES],
		symbolic[FIXTURE_PATH_BYTES], name[FIXTURE_NAME_BYTES],
		expected[RECEIPT_BYTES];
	(void)snprintf(name, sizeof(name), "%016" PRIx64 ".anemone",
		       header.header_digest);
	prepared =
		prepared && joined(source, shared, "source.anemone") &&
		written_bytes(source, bytes, (size_t)byte_count) &&
		realpath(source, real_source) != NULL &&
		joined(home, root, "home") && mkdir(home, FIXTURE_MODE) == 0 &&
		joined(point, root, "mount") &&
		mkdir(point, FIXTURE_MODE) == 0 && joined(old, point, "old") &&
		mkdir(old, FIXTURE_MODE) == 0 && joined(copy, old, name) &&
		written_bytes(copy, bytes, (size_t)byte_count) &&
		joined(rotten_copy, old, "rotten.anemone") &&
		written_bytes(rotten_copy, bytes, (size_t)byte_count) &&
		joined(short_copy, old, "short.anemone") &&
		written_bytes(short_copy, bytes,
			      (size_t)byte_count - PACK_BLOCK_BYTES) &&
		joined(symbolic, old, "link.anemone") &&
		symlink(copy, symbolic) == 0 &&
		joined(legacy_point, root, "legacy_mount") &&
		mkdir(legacy_point, FIXTURE_MODE) == 0 &&
		joined(legacy, legacy_point, "anemone") &&
		mkdir(legacy, FIXTURE_MODE) == 0 &&
		joined(legacy, legacy_point, LEGACY_DIRECTORY) &&
		mkdir(legacy, FIXTURE_MODE) == 0 &&
		joined(legacy_copy, legacy_point, "copy.anemone") &&
		written_bytes(legacy_copy, bytes, (size_t)byte_count) &&
		setenv("HOME", home, 1) == 0;
	int length = snprintf(expected, sizeof(expected),
			      "anemone replica 1\nsource %s\nbytes %" PRIu64
			      "\nheader_digest %016" PRIx64 "\n",
			      real_source, header.file_byte_count,
			      header.header_digest);
	prepared = prepared && length > 0 && (size_t)length < sizeof(expected);
	if (prepared) {
		unsigned char *const flipped = malloc((size_t)byte_count);
		prepared = flipped != NULL;
		if (prepared) {
			memcpy(flipped, bytes, (size_t)byte_count);
			flipped[rotten] ^= 1u;
			prepared = written_bytes(rotten_copy, flipped,
						 (size_t)byte_count);
		}
		free(flipped);
	}
	expect(prepared,
	       "a pack on /dev/shm and its copies under fixture mounts on /var/tmp");
	struct local_device_description internal;
	const bool admitted =
		local_device_describe(point, &internal) == STATUS_OK &&
		local_device_media_refusal(&internal, false) == NULL;
	char found[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
	if (prepared) {
		char home_directory[FIXTURE_PATH_BYTES],
			home_name[FIXTURE_PATH_BYTES],
			receipt[FIXTURE_PATH_BYTES],
			point_directory[FIXTURE_PATH_BYTES];
		uint32_t differing = 0;
		/* HOME on the copy's filesystem: its search directory, made, takes the name */
		const bool paths =
			joined(home_directory, home, SEARCH_DIRECTORY) &&
			joined(home_name, home_directory, name) &&
			snprintf(receipt, sizeof(receipt), "%s.receipt",
				 home_name) > 0 &&
			joined(point_directory, point, SEARCH_DIRECTORY);
		const enum replica_adoption adopted = replicas_adopt(
			copy, point, real_source, &header, &differing);
		expect(paths && adopted == REPLICA_ADOPTED &&
			       differing == NO_EXTENT &&
			       same_inode(copy, home_name) && exists(copy) &&
			       receipt_is(receipt, expected) &&
			       !exists(point_directory),
		       "a sound copy is linked under HOME's search directory and receipted; it keeps its own name");
		const uint32_t count =
			search(root, point, real_source, &header, found);
		expect(count == (admitted ? 1u : 0u) &&
			       (count == 0 || strcmp(found[0], home_name) == 0),
		       "the search finds the adopted copy under HOME's directory");
		expect(replicas_adopt(copy, point, real_source, &header,
				      &differing) == REPLICA_ADOPTED &&
			       same_inode(copy, home_name) &&
			       receipt_is(receipt, expected),
		       "adopting it again changes nothing");
		/* the same name taken by another file: it stays, and no receipt names the rotten copy */
		expect(unlink(receipt) == 0 &&
			       replicas_adopt(rotten_copy, point, real_source,
					      &header, &differing) ==
				       REPLICA_ADOPT_FAILED &&
			       differing == ROTTEN_EXTENT &&
			       same_inode(copy, home_name) && !exists(receipt),
		       "a copy with a rotten byte is read, refused and its extent named; it gets no name or receipt");
		expect(replicas_adopt(short_copy, point, real_source, &header,
				      &differing) == REPLICA_ADOPT_REFUSED &&
			       replicas_adopt(symbolic, point, real_source,
					      &header, &differing) ==
				       REPLICA_ADOPT_REFUSED &&
			       replicas_adopt(source, point, real_source,
					      &header, &differing) ==
				       REPLICA_ADOPT_REFUSED &&
			       !exists(receipt),
		       "a short copy, a symbolic link and the source itself are refused unread");
		/* a sound copy whose name is another file's: the name keeps that file, no receipt */
		expect(unlink(home_name) == 0 &&
			       link(short_copy, home_name) == 0 &&
			       replicas_adopt(copy, point, real_source, &header,
					      &differing) ==
				       REPLICA_ADOPT_FAILED &&
			       same_inode(short_copy, home_name) &&
			       !exists(receipt),
		       "a name another file holds is never taken over");
		/* without HOME there: the existing legacy directory under the mount point takes it */
		char legacy_name[FIXTURE_PATH_BYTES];
		expect(unsetenv("HOME") == 0 &&
			       joined(legacy_name, legacy, name) &&
			       replicas_adopt(legacy_copy, legacy_point,
					      real_source, &header,
					      &differing) == REPLICA_ADOPTED &&
			       same_inode(legacy_copy, legacy_name),
		       "without HOME on its filesystem a copy takes the existing legacy directory");
		const uint32_t legacy_count =
			search(root, legacy_point, real_source, &header, found);
		expect(legacy_count == (admitted ? 1u : 0u) &&
			       (legacy_count == 0 ||
				strcmp(found[0], legacy_name) == 0),
		       "the search finds the copy adopted into the legacy directory");
	}
	if (pack != NULL)
		pack_close(pack);
	if (reader != NULL)
		test_memory_storage.reader_close(reader);
	free(bytes);
	expect((!have_root || fixture_remove(root)) &&
		       (!have_shared || fixture_remove(shared)),
	       "fixture directories removed");
	if (failures == 0)
		(void)printf(
			"replica_adopt_test: copies read in full before their name and receipt; rotten, short, "
			"linked and taken names refused%s\n",
			admitted ? "" :
				   " (the search half: media refused here)");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
