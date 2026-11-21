/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* engram_hot_test: the static hot set (engram_hot.h) over the synthetic pack on two copies (table 0 one block, table 1
 * two): a file the builder's encoder writes loads, its blocks are exact and served without a read, a corrupt copy falls
 * back, and each named refusal holds (digest, magic, pack, order, range, budget, trailing bytes, link, not a file). */
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "engine/engram/engram_disk.h"
#include "engine/engram/engram_hot.h"
#include "support/engram_pack.h"
#include "engine/engram/engram_step.h"
#include "support/memory_storage.h"
#include "common/sha256.h"

#define PATH_BYTES 512u
#define BUDGET_BYTES (1ull << 20u)
#define AT_HEADER_DIGEST 16u
#define AT_TABLE1_SECOND_ID \
	120u /* header 40, table 0's section 32 and id 8, table 1's section 32 and first id 8 */
#define HEX_PER_BYTE 2u
#define GOOD_BLOCKS 3u /* every block of both tables */
#define TEST_SECONDS 60u
#define COPIES 2u
#define TABLES 2u
#define NAME_BYTES 64u
#define HOT_BLOCK \
	1u /* table 1's second block, the one the partial hot set holds */
#define HOT_TABLE 1u
#define PAIR 2u

static unsigned failures;

static void expect(bool passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

struct fixture {
	unsigned char *bytes;
	struct engram_record *engram;
	struct storage_reader *reader;
	struct pack_record *pack;
	struct engram_disk *disk;
	char directory[PATH_BYTES];
};

static void hex_of(const unsigned char *bytes, size_t count,
		   char hex[ENGRAM_HOT_SHA256_HEX_BYTES + 1])
{
	unsigned char digest[SHA256_DIGEST_BYTES];
	if (sha256_of(bytes, count, digest) != STATUS_OK)
		memset(digest, 0, sizeof(digest));
	for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++)
		(void)snprintf(hex + (size_t)HEX_PER_BYTE * index,
			       HEX_PER_BYTE + 1u, "%02x", digest[index]);
}

static bool write_file(const char *path, const unsigned char *bytes,
		       size_t count)
{
	(void)unlink(path);
	const int descriptor =
		open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR);
	const bool written = descriptor >= 0 &&
			     write(descriptor, bytes, count) == (ssize_t)count;
	if (descriptor >= 0)
		(void)close(descriptor);
	return written;
}

/* Byte b of table t's extent is b times the pattern multiplier plus t (engram_pack.h). */
static bool block_is_right(uint32_t table, uint64_t block,
			   const unsigned char *bytes)
{
	for (uint64_t byte = 0; bytes != NULL && byte < PACK_BLOCK_BYTES;
	     byte++)
		if (bytes[byte] !=
		    (unsigned char)((block * PACK_BLOCK_BYTES + byte) *
					    TEST_ENGRAM_PACK_PATTERN_MULTIPLIER +
				    table))
			return false;
	return bytes != NULL;
}

/* Opens the hot set written from bytes (its own SHA-256 unless hex is given) and returns the status and reason. */
MUST_CHECK static result_status
try_open(struct fixture *fixture, const char *name, const unsigned char *bytes,
	 size_t count, const char *hex, uint64_t budget,
	 struct engram_hot **hot, char *reason)
{
	char path[PATH_BYTES + NAME_BYTES],
		own[ENGRAM_HOT_SHA256_HEX_BYTES + 1];
	(void)snprintf(path, sizeof(path), "%s/%s", fixture->directory, name);
	hex_of(bytes, count, own);
	if (!write_file(path, bytes, count))
		return STATUS_INPUT_OUTPUT_ERROR;
	const result_status status =
		engram_hot_open(path, hex != NULL ? hex : own, budget,
				fixture->pack, fixture->disk, hot, reason);
	(void)unlink(path);
	return status;
}

static void refused(struct fixture *fixture, const char *name,
		    const unsigned char *bytes, size_t count, const char *hex,
		    uint64_t budget, const char *reason_part)
{
	struct engram_hot *hot = NULL;
	char reason[ENGRAM_HOT_REASON_BYTES] = "";
	const result_status status = try_open(fixture, name, bytes, count, hex,
					      budget, &hot, reason);
	expect(status != STATUS_OK && hot == NULL &&
		       strstr(reason, reason_part) != NULL,
	       reason_part);
	engram_hot_close(hot);
}

int main(void)
{
	(void)alarm(TEST_SECONDS);
	struct fixture fixture = { 0 };
	const char *const copies[] = { "synthetic0", "synthetic1" };
	(void)snprintf(fixture.directory, sizeof(fixture.directory),
		       "build/engram_hot_test.%ld", (long)getpid());
	if (mkdir(fixture.directory, S_IRWXU) != 0 ||
	    test_engram_pack_open(&fixture.bytes, &fixture.engram) !=
		    STATUS_OK ||
	    test_memory_storage.reader_open(copies, COPIES, &fixture.reader) !=
		    STATUS_OK ||
	    pack_open_entry(&test_memory_storage, fixture.reader,
			    &fixture.pack) != STATUS_OK ||
	    engram_description_of(fixture.engram) == NULL ||
	    engram_disk_open(engram_description_of(fixture.engram),
			     fixture.pack, &test_memory_storage, fixture.reader,
			     0, 0, 1, 0, &fixture.disk) != STATUS_OK) {
		(void)fprintf(stderr,
			      "engram_hot_test: the fixture does not open\n");
		return EXIT_FAILURE;
	}
	const struct engram_description *const description =
		engram_description_of(fixture.engram);
	const uint64_t blocks1 = engram_disk_block_count(fixture.disk, 1);
	const uint64_t ids0[] = { 0 }, ids1[PAIR] = { 0, HOT_BLOCK };
	const struct engram_hot_table tables[] = {
		{ .table = 0,
		  .extent_digest = engram_disk_extent_digest(fixture.disk, 0),
		  .block_count = engram_disk_block_count(fixture.disk, 0),
		  .id_count = 1,
		  .ids = ids0 },
		{ .table = 1,
		  .extent_digest = engram_disk_extent_digest(fixture.disk, 1),
		  .block_count = blocks1,
		  .id_count = PAIR,
		  .ids = ids1 }
	};
	const struct engram_hot_table partial[] = {
		{ .table = HOT_TABLE,
		  .extent_digest = tables[1].extent_digest,
		  .block_count = blocks1,
		  .id_count = 1,
		  .ids = &ids1[1] }
	};
	unsigned char *good = NULL, *part = NULL;
	size_t good_count = 0, part_count = 0;
	expect(tables[0].block_count == 1 && blocks1 == PAIR &&
		       engram_hot_encode(pack_header_of(fixture.pack), tables,
					 TABLES, &good,
					 &good_count) == STATUS_OK &&
		       engram_hot_encode(pack_header_of(fixture.pack), partial,
					 1, &part, &part_count) == STATUS_OK,
	       "the builder's encoder lays out a hot set");
	const uint64_t unsorted[] = { HOT_BLOCK, 0 };
	const struct engram_hot_table bad_order[] = { { .table = HOT_TABLE,
							.block_count = blocks1,
							.id_count = PAIR,
							.ids = unsorted } };
	unsigned char *unused = NULL;
	size_t unused_count = 0;
	expect(engram_hot_encode(pack_header_of(fixture.pack), bad_order, 1,
				 &unused,
				 &unused_count) == STATUS_INVALID_ARGUMENT,
	       "the encoder refuses ids out of order");

	struct engram_hot *hot = NULL;
	char reason[ENGRAM_HOT_REASON_BYTES] = "";
	expect(good != NULL &&
		       try_open(&fixture, "good", good, good_count, NULL,
				BUDGET_BYTES, &hot, reason) == STATUS_OK &&
		       engram_hot_block_count(hot) == GOOD_BLOCKS &&
		       block_is_right(0, 0, engram_hot_block(hot, 0, 0)) &&
		       block_is_right(1, 0, engram_hot_block(hot, 1, 0)) &&
		       block_is_right(1, HOT_BLOCK,
				      engram_hot_block(hot, 1, HOT_BLOCK)),
	       "a hot set loads its three blocks, exact");
	engram_hot_close(hot);
	hot = NULL;
	expect(part != NULL &&
		       try_open(&fixture, "part", part, part_count, NULL,
				BUDGET_BYTES, &hot, reason) == STATUS_OK &&
		       engram_hot_block_count(hot) == 1 &&
		       block_is_right(HOT_TABLE, HOT_BLOCK,
				      engram_hot_block(hot, HOT_TABLE,
						       HOT_BLOCK)) &&
		       engram_hot_block(hot, HOT_TABLE, 0) == NULL &&
		       engram_hot_block(hot, 0, 0) == NULL,
	       "only the listed blocks are held");
	engram_disk_attach_hot(fixture.disk, hot);
	const uint64_t reads = test_memory_storage_read_count(0) +
			       test_memory_storage_read_count(1);
	const uint64_t row = HOT_BLOCK * ENGRAM_ROWS_PER_BLOCK + 1u;
	const unsigned char *const held =
		engram_hot_block(hot, HOT_TABLE, HOT_BLOCK);
	unsigned char output[ENGRAM_ROW_BYTES];
	expect(held != NULL && row < description->row_count[HOT_TABLE] &&
		       engram_disk_gather(fixture.disk, HOT_TABLE, &row, 1,
					  output) == STATUS_OK &&
		       memcmp(output, held + ENGRAM_ROW_BYTES,
			      ENGRAM_ROW_BYTES) == 0 &&
		       test_memory_storage_read_count(0) +
				       test_memory_storage_read_count(1) ==
			       reads &&
		       engram_disk_ram_block(fixture.disk, HOT_TABLE,
					     HOT_BLOCK) == held,
	       "a hot row is served from RAM without a read, the step plan's RAM lookup included");
	engram_disk_attach_hot(fixture.disk, NULL);
	engram_hot_close(hot);
	hot = NULL;

	/* a corrupt copy falls back to the other; both corrupt fail the open */
	const uint64_t at_hot_block =
		pack_extents_of(fixture.pack)
			[pack_units_of(fixture.pack)
				 [description->table_unit_index[HOT_TABLE]]
					 .first_extent_index]
				.byte_offset +
		(uint64_t)HOT_BLOCK * PACK_BLOCK_BYTES;
	test_memory_storage_flip_bit(0, at_hot_block);
	expect(try_open(&fixture, "fallback", part, part_count, NULL,
			BUDGET_BYTES, &hot, reason) == STATUS_OK &&
		       block_is_right(HOT_TABLE, HOT_BLOCK,
				      engram_hot_block(hot, HOT_TABLE,
						       HOT_BLOCK)),
	       "a block corrupt on one copy is read from the other");
	engram_hot_close(hot);
	hot = NULL;
	test_memory_storage_flip_bit(1, at_hot_block);
	refused(&fixture, "corrupt", part, part_count, NULL, BUDGET_BYTES,
		"no copy delivers verified");

	unsigned char *const copy = malloc(good_count + 1u);
	if (copy == NULL || good == NULL || part == NULL) {
		free(copy);
		free(good);
		free(part);
		return EXIT_FAILURE;
	}
	refused(&fixture, "digest", good, good_count,
		"0000000000000000000000000000000000000000000000000000000000000000",
		BUDGET_BYTES, "recorded SHA-256");
	memcpy(copy, good, good_count);
	copy[0] ^= 1u;
	refused(&fixture, "magic", copy, good_count, NULL, BUDGET_BYTES,
		"not a hot-set file");
	memcpy(copy, good, good_count);
	copy[AT_HEADER_DIGEST] ^= 1u;
	refused(&fixture, "pack", copy, good_count, NULL, BUDGET_BYTES,
		"another pack");
	memcpy(copy, good, good_count);
	memcpy(copy + AT_TABLE1_SECOND_ID, &ids1[0], sizeof(uint64_t));
	refused(&fixture, "order", copy, good_count, NULL, BUDGET_BYTES,
		"strictly increasing");
	memcpy(copy, good, good_count);
	memcpy(copy + AT_TABLE1_SECOND_ID, &blocks1, sizeof(uint64_t));
	refused(&fixture, "range", copy, good_count, NULL, BUDGET_BYTES,
		"out of range");
	refused(&fixture, "budget", good, good_count, NULL,
		(uint64_t)(GOOD_BLOCKS - 1u) * PACK_BLOCK_BYTES, "budget");
	memcpy(copy, good, good_count);
	copy[good_count] = 0;
	refused(&fixture, "trailing", copy, good_count + 1u, NULL, BUDGET_BYTES,
		"after the last section");
	char path[PATH_BYTES + NAME_BYTES], link_path[PATH_BYTES + NAME_BYTES];
	(void)snprintf(path, sizeof(path), "%s/target", fixture.directory);
	(void)snprintf(link_path, sizeof(link_path), "%s/link",
		       fixture.directory);
	char own[ENGRAM_HOT_SHA256_HEX_BYTES + 1];
	hex_of(good, good_count, own);
	expect(write_file(path, good, good_count) &&
		       symlink(path, link_path) == 0 &&
		       engram_hot_open(link_path, own, BUDGET_BYTES,
				       fixture.pack, fixture.disk, &hot,
				       reason) != STATUS_OK &&
		       strstr(reason, "symbolic link") != NULL,
	       "a symbolic link is refused");
	expect(engram_hot_open(fixture.directory, own, BUDGET_BYTES,
			       fixture.pack, fixture.disk, &hot,
			       reason) != STATUS_OK &&
		       strstr(reason, "regular file") != NULL,
	       "a directory is refused");
	/* a FIFO is refused at once, not waited on for a writer (without O_NONBLOCK the open blocks
	 * and the alarm ends the test) */
	char fifo_path[PATH_BYTES + NAME_BYTES];
	(void)snprintf(fifo_path, sizeof(fifo_path), "%s/fifo",
		       fixture.directory);
	expect(mkfifo(fifo_path, S_IRUSR | S_IWUSR) == 0 &&
		       engram_hot_open(fifo_path, own, BUDGET_BYTES,
				       fixture.pack, fixture.disk, &hot,
				       reason) != STATUS_OK &&
		       strstr(reason, "regular file") != NULL,
	       "a FIFO is refused without waiting for a writer");
	(void)unlink(fifo_path);
	(void)unlink(link_path);
	(void)unlink(path);
	(void)rmdir(fixture.directory);
	free(copy);
	free(part);
	free(good);
	engram_disk_close(fixture.disk);
	engram_close(fixture.engram);
	pack_close(fixture.pack);
	test_memory_storage.reader_close(fixture.reader);
	free(fixture.bytes);
	if (failures != 0) {
		(void)fprintf(stderr, "engram_hot_test: %u failures\n",
			      failures);
		return EXIT_FAILURE;
	}
	(void)puts("engram_hot_test: passed");
	return EXIT_SUCCESS;
}
