/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* engram_fit_test: the Engram RAM fit (engram_fit.h). The needs of the synthetic n-gram pack match its disk store's
 * own block counts, and the store's sizes come from the one RAM-blocks formula; the decision holds at each boundary
 * (every table, the largest share, the disk store, the disk store inside the headroom, the refusal, an unreadable
 * MemAvailable, a given percent that fits or not); /proc/meminfo texts parse or are refused; the lines name the
 * percent, its source, the budget and, for a refusal, the model's largest item; the checking seam replaces
 * MemAvailable. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/engram_disk.h"
#include "anemone/engram_fit.h"
#include "anemone/hooks.h"
#include "anemone/loader.h"
#include "anemone/model_seam.h"
#include "anemone/pack.h"
#include "anemone/tp_load.h"
#include "engram_pack.h"
#include "engram_step.h"
#include "memory_storage.h"

#define TEST_SECONDS 60u
#define COPIES 1u
#define STAGING_CARDS 5u
#define LOAD_THREADS 16u
#define LARGE_PIECE_BYTES (64ull << 20u) /* --load_piece_mib=64, the most */
#define TP_WORKSPACES 2u
#define KIB 1024ull
#define GIB (1ull << 30u)
#define TABLE0_BLOCKS 100000u
#define TABLE1_BLOCKS 50000u
#define SHARE_PERCENT 37u
#define GIVEN_PERCENT 50u
#define OTHER_HOST_BYTES GIB
#define STAGING_BYTES (640ull << 20u)
#define MEMORY_TOTAL (64ull * GIB)
#define LARGE_TOTAL (1024ull * GIB)
#define LARGE_HEADROOM (32ull * GIB)
#define CACHE_BYTES (1ull << 20u)
#define CACHE_SLOTS (CACHE_BYTES / PACK_BLOCK_BYTES)
#define SLOT_RECORD_MOST 64u /* a cache slot's record is a few words */
#define SEAM_BYTES 123456789ull
#define LINE_BYTES 4096u
#define SYNTHETIC_TABLES 2u
#define FIRST_BLOCKS 1000u
#define FIRST_PREFIX 370u /* FIRST_BLOCKS at SHARE_PERCENT */
#define SECOND_BLOCKS 2000u
#define SECOND_PREFIX 740u
#define ODD_BLOCKS 199u
#define ODD_HALF 99u /* ODD_BLOCKS at GIVEN_PERCENT */
#define FEW_BLOCKS 5u
#define FEW_PREFIX 4u /* FEW_BLOCKS at 99 */
#define MEMINFO_TOTAL_KIB 527362048
#define MEMINFO_AVAILABLE_KIB 311534592
#define TEXT_OF(number) #number
#define TEXT_OF_VALUE(number) TEXT_OF(number)
#define MEMINFO_TEXT                                                 \
	"MemTotal:       " TEXT_OF_VALUE(                            \
		MEMINFO_TOTAL_KIB) " kB\n"                           \
				   "MemFree: 1 kB\n"                 \
				   "MemAvailable:   " TEXT_OF_VALUE( \
					   MEMINFO_AVAILABLE_KIB) " kB\n"
#define SHORT_DIVISOR 2u /* half of need(0): the model's own memory short */
#define COUNT_OF(array) (sizeof(array) / sizeof((array)[0]))

static unsigned failures;

static void expect(bool passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct engram_fit_needs synthetic_needs(uint64_t available)
{
	struct engram_fit_needs needs = { .table_blocks = { TABLE0_BLOCKS,
							    TABLE1_BLOCKS },
					  .table_count = SYNTHETIC_TABLES,
					  .other_host_bytes = OTHER_HOST_BYTES,
					  .staging_bytes = STAGING_BYTES,
					  .memory_total = MEMORY_TOTAL,
					  .memory_available = available };
	return needs;
}

static struct engram_fit_decision
decide(uint64_t available, enum engram_ram_source source, uint32_t given)
{
	const struct engram_fit_needs needs = synthetic_needs(available);
	struct engram_fit_decision decision;
	engram_fit_decide(&needs, source, given, &decision);
	return decision;
}

static void expect_decision(uint64_t available, uint32_t percent,
			    enum engram_fit_case fit_case, const char *message)
{
	const struct engram_fit_decision decision =
		decide(available, ENGRAM_RAM_AUTO, 0);
	expect(decision.percent == percent && decision.fit_case == fit_case,
	       message);
}

static void decisions(void)
{
	const struct engram_fit_needs needs = synthetic_needs(0);
	const uint64_t headroom = engram_fit_headroom_bytes(MEMORY_TOTAL);
	expect(headroom == ENGRAM_RAM_FIT_HEADROOM_LEAST_BYTES &&
		       engram_fit_headroom_bytes(0) ==
			       ENGRAM_RAM_FIT_HEADROOM_LEAST_BYTES &&
		       engram_fit_headroom_bytes(LARGE_TOTAL) == LARGE_HEADROOM,
	       "the headroom is max(the least, MemTotal / the share)");
	const uint64_t every =
		engram_fit_need_bytes(&needs, ENGRAM_RAM_PERCENT);
	const uint64_t share = engram_fit_need_bytes(&needs, SHARE_PERCENT);
	const uint64_t disk = engram_fit_need_bytes(&needs, 0);
	expect(every == ((uint64_t)TABLE0_BLOCKS + TABLE1_BLOCKS) *
					PACK_BLOCK_BYTES +
				OTHER_HOST_BYTES + STAGING_BYTES,
	       "need(100) is the tables' extents and the model");
	expect(disk == ((uint64_t)TABLE0_BLOCKS + TABLE1_BLOCKS) *
				       sizeof(uint64_t) +
			       OTHER_HOST_BYTES + STAGING_BYTES,
	       "need(0) is the block digests and the model");
	expect(share == engram_disk_host_bytes(needs.table_blocks,
					       SYNTHETIC_TABLES, SHARE_PERCENT,
					       0) +
				OTHER_HOST_BYTES + STAGING_BYTES,
	       "need(37) is the disk store's own sum and the model");
	expect_decision(UINT64_MAX, ENGRAM_RAM_PERCENT, ENGRAM_FIT_EVERY_TABLE,
			"plenty of memory: every table");
	expect_decision(every + headroom, ENGRAM_RAM_PERCENT,
			ENGRAM_FIT_EVERY_TABLE,
			"need(100) exactly: every table");
	expect_decision(every + headroom - 1, ENGRAM_RAM_PERCENT - 1,
			ENGRAM_FIT_SHARE, "one byte short: 99");
	expect_decision(share + headroom, SHARE_PERCENT, ENGRAM_FIT_SHARE,
			"need(37) exactly: 37");
	expect_decision(share + headroom - 1, SHARE_PERCENT - 1,
			ENGRAM_FIT_SHARE, "one byte short of 37: 36");
	expect_decision(disk + headroom, 0, ENGRAM_FIT_DISK_ONLY,
			"need(0) exactly: the disk store");
	expect_decision(disk, 0, ENGRAM_FIT_DISK_ONLY_IN_HEADROOM,
			"need(0) without the headroom: the disk store, warned");
	expect_decision(disk - 1, 0, ENGRAM_FIT_MODEL_OVER,
			"the model's own memory short: refused");
	expect_decision(0, ENGRAM_RAM_PERCENT, ENGRAM_FIT_MEMORY_UNKNOWN,
			"MemAvailable unreadable: every table");
	const struct engram_fit_decision knob =
		decide(1, ENGRAM_RAM_KNOB, GIVEN_PERCENT);
	const struct engram_fit_decision profile =
		decide(UINT64_MAX, ENGRAM_RAM_PROFILE, ENGRAM_RAM_PERCENT);
	const struct engram_fit_decision unknown =
		decide(0, ENGRAM_RAM_KNOB, 0);
	expect(knob.percent == GIVEN_PERCENT &&
		       knob.fit_case == ENGRAM_FIT_GIVEN_OVER &&
		       profile.percent == ENGRAM_RAM_PERCENT &&
		       profile.fit_case == ENGRAM_FIT_GIVEN &&
		       unknown.percent == 0 &&
		       unknown.fit_case == ENGRAM_FIT_GIVEN,
	       "a given percent is kept, whether it fits or not");
	expect(knob.need_bytes == engram_fit_need_bytes(&needs, GIVEN_PERCENT) +
					  headroom &&
		       knob.headroom_bytes == headroom,
	       "the decision's need includes the headroom");
}

static void store_sizes(void)
{
	expect(engram_disk_ram_blocks(FIRST_BLOCKS, SHARE_PERCENT) ==
			       FIRST_PREFIX &&
		       engram_disk_ram_blocks(ODD_BLOCKS, GIVEN_PERCENT) ==
			       ODD_HALF &&
		       engram_disk_ram_blocks(FEW_BLOCKS,
					      ENGRAM_RAM_PERCENT - 1) ==
			       FEW_PREFIX &&
		       engram_disk_ram_blocks(1, ENGRAM_RAM_PERCENT - 1) == 0 &&
		       engram_disk_ram_blocks(UINT64_MAX, GIVEN_PERCENT) ==
			       UINT64_MAX /
				       (ENGRAM_RAM_PERCENT / GIVEN_PERCENT),
	       "the RAM prefix rounds down to a block without wrapping");
	const uint64_t blocks[SYNTHETIC_TABLES] = { FIRST_BLOCKS,
						    SECOND_BLOCKS };
	const uint64_t store = engram_disk_host_bytes(blocks, SYNTHETIC_TABLES,
						      SHARE_PERCENT, 0);
	const uint64_t cached = engram_disk_host_bytes(
		blocks, SYNTHETIC_TABLES, SHARE_PERCENT, CACHE_BYTES);
	expect(store == ((uint64_t)FIRST_PREFIX + SECOND_PREFIX) *
					PACK_BLOCK_BYTES +
				((uint64_t)FIRST_BLOCKS + SECOND_BLOCKS) *
					sizeof(uint64_t),
	       "the store is each table's prefix and block digests");
	expect(cached - store >= CACHE_SLOTS * PACK_BLOCK_BYTES &&
		       cached - store < CACHE_SLOTS * (PACK_BLOCK_BYTES +
						       SLOT_RECORD_MOST),
	       "the cache adds its blocks and slot records");
}

static void pack_needs(void)
{
	unsigned char *bytes = NULL;
	struct engram_record *engram = NULL;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct engram_disk *disk = NULL;
	const char *const copies[] = { "synthetic0" };
	if (test_engram_pack_open(&bytes, &engram) != STATUS_OK ||
	    test_memory_storage.reader_open(copies, COPIES, &reader) !=
		    STATUS_OK ||
	    pack_open_entry(&test_memory_storage, reader, &pack) != STATUS_OK ||
	    engram_disk_open(engram_description_of(engram), pack,
			     &test_memory_storage, reader, 0, 0, 0, 0,
			     &disk) != STATUS_OK) {
		expect(false, "the synthetic pack and its disk store open");
	} else {
		const struct engram_description *const tables =
			engram_description_of(engram);
		struct engram_fit_needs pipeline = { 0 }, tensor = { 0 };
		struct engram_fit_needs pieces = { 0 }, uncounted = { 0 };
		expect(engram_fit_needs_of_pack(pack, tables, false,
						STAGING_CARDS, 0, 0,
						&pipeline) == STATUS_OK &&
			       engram_fit_needs_of_pack(pack, tables, true, 0,
							0, LOAD_THREADS,
							&tensor) == STATUS_OK &&
			       engram_fit_needs_of_pack(pack, tables, false,
							STAGING_CARDS,
							LARGE_PIECE_BYTES, 0,
							&pieces) == STATUS_OK &&
			       engram_fit_needs_of_pack(pack, tables, false, 0,
							0, 0, &uncounted) ==
				       STATUS_OK,
		       "the synthetic pack's needs");
		expect(pieces.staging_bytes ==
				       (uint64_t)STAGING_CARDS *
					       LOADER_STAGING_SLOTS_PER_CARD *
					       LARGE_PIECE_BYTES &&
			       pieces.staging_cards_most == 0 &&
			       pipeline.staging_cards_most == 0,
		       "PP: the staging slots are the load's piece, the counted cards' alone");
		expect(uncounted.staging_bytes ==
				       (uint64_t)ENGRAM_RAM_FIT_STAGING_CARDS_MOST *
					       LOADER_STAGING_SLOTS_PER_CARD *
					       LOADER_CHUNK_BYTES &&
			       uncounted.staging_cards_most ==
				       ENGRAM_RAM_FIT_STAGING_CARDS_MOST &&
			       tensor.staging_cards_most == 0,
		       "PP: cards not yet counted are the declared upper bound, which TP has not");
		bool blocks_match = pipeline.table_count == tables->table_count;
		for (uint32_t table = 0;
		     blocks_match && table < tables->table_count; table++)
			blocks_match = pipeline.table_blocks[table] ==
				       engram_disk_block_count(disk, table);
		expect(blocks_match, "the tables' blocks are the disk store's");
		uint64_t host_only = 0, largest_card_extent = 0;
		const struct pack_header *const header = pack_header_of(pack);
		for (uint32_t extent = 0; extent < header->extent_count;
		     extent++) {
			const struct pack_extent *const record =
				&pack_extents_of(pack)[extent];
			const bool is_host =
				pack_units_of(pack)[record->unit_index]
					.unit_kind == PLACEMENT_UNIT_HOST_ONLY;
			host_only += is_host ? record->byte_count : 0;
			largest_card_extent =
				!is_host && record->byte_count >
							largest_card_extent ?
					record->byte_count :
					largest_card_extent;
		}
		const uint64_t table_bytes =
			engram_fit_need_bytes(&pipeline, ENGRAM_RAM_PERCENT) -
			pipeline.other_host_bytes - pipeline.staging_bytes;
		expect(pipeline.other_host_bytes + table_bytes == host_only &&
			       pipeline.other_host_bytes != 0,
		       "the other host-only units are the constants beside the tables");
		expect(pipeline.pinned_expert_bytes == 0 &&
			       pipeline.staging_bytes ==
				       (uint64_t)STAGING_CARDS *
					       LOADER_STAGING_SLOTS_PER_CARD *
					       LOADER_CHUNK_BYTES,
		       "PP: no pinned experts, the loader's staging slots");
		expect(tensor.pinned_expert_bytes == tensor.other_host_bytes &&
			       tensor.staging_bytes ==
				       TP_WORKSPACES * largest_card_extent +
					       (uint64_t)LOAD_THREADS *
						       TP_WORKSPACES *
						       TP_LOAD_WORKER_SLOT_BYTES,
		       "TP: the host experts bound, its workspaces and slots");
		struct engram_description none = *tables;
		none.table_count = 0;
		expect(engram_fit_needs_of_pack(pack, &none, false, 1, 0, 0,
						&pipeline) ==
			       STATUS_INVALID_ARGUMENT,
		       "no table is refused");
	}
	engram_disk_close(disk);
	engram_close(engram);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	free(bytes);
}

static bool parses(const char *text, uint64_t *total, uint64_t *available)
{
	FILE *const stream =
		fmemopen((void *)(uintptr_t)text, strlen(text), "r");
	const bool parsed = stream != NULL &&
			    engram_fit_memory_of(stream, total, available);
	if (stream != NULL)
		(void)fclose(stream);
	return parsed;
}

static void meminfo(void)
{
	uint64_t total = 1, available = 1;
	expect(parses(MEMINFO_TEXT, &total, &available) &&
		       total == (uint64_t)MEMINFO_TOTAL_KIB * KIB &&
		       available == (uint64_t)MEMINFO_AVAILABLE_KIB * KIB,
	       "a /proc/meminfo text parses to bytes");
	expect(!parses("MemTotal: 1024 kB\nMemFree: 1 kB\n", &total,
		       &available) &&
		       total == 0 && available == 0,
	       "a text without MemAvailable is unreadable");
	expect(!parses("MemTotal: 1024 kB\nMemAvailable: 18446744073709551615 kB\n",
		       &total, &available) &&
		       available == 0,
	       "a MemAvailable that wraps in bytes is unreadable");
	struct engram_fit_needs needs = { 0 };
#if CHECKING_HOOKS_ENABLED
	expect(setenv(ENGRAM_RAM_FIT_MEMORY_SEAM, "123456789", 1) == 0,
	       "the seam is set");
	engram_fit_memory_read(&needs);
	expect(needs.memory_available == SEAM_BYTES,
	       "the checking seam replaces MemAvailable");
	expect(unsetenv(ENGRAM_RAM_FIT_MEMORY_SEAM) == 0, "the seam is unset");
#endif
	engram_fit_memory_read(&needs);
	expect(access("/proc/meminfo", R_OK) != 0 ||
		       (needs.memory_total != 0 && needs.memory_available != 0),
	       "the host's /proc/meminfo reads");
}

static bool printed(const struct engram_fit_needs *needs,
		    enum engram_ram_source source, uint32_t given,
		    const char *const *parts, size_t part_count)
{
	char text[LINE_BYTES] = "";
	FILE *const stream = fmemopen(text, sizeof(text) - 1, "w");
	if (stream == NULL)
		return false;
	struct engram_fit_decision decision;
	engram_fit_decide(needs, source, given, &decision);
	engram_fit_print(stream, needs, &decision, source);
	(void)fclose(stream);
	bool found = true;
	for (size_t part = 0; found && part < part_count; part++)
		found = strstr(text, parts[part]) != NULL;
	return found;
}

static void lines(void)
{
	const struct engram_fit_needs base = synthetic_needs(0);
	const uint64_t headroom = engram_fit_headroom_bytes(MEMORY_TOTAL);
	const struct engram_fit_needs share = synthetic_needs(
		engram_fit_need_bytes(&base, SHARE_PERCENT) + headroom);
	const char *const share_parts[] = {
		"engram ram percent: 37 (auto: the largest whole share that fits)\n",
		"engram ram fit: percent=37 MemAvailable=",
		" + headroom 8.00 GiB (max(8.00 GiB, MemTotal 64.00 GiB / 32))\n"
	};
	expect(printed(&share, ENGRAM_RAM_AUTO, 0, share_parts,
		       COUNT_OF(share_parts)),
	       "the share's percent, source and budget lines");
	struct engram_fit_needs bounded = share;
	bounded.staging_cards_most = ENGRAM_RAM_FIT_STAGING_CARDS_MOST;
	const char *const bounded_parts[] = {
		"engram ram fit: percent=37 MemAvailable=",
		" + load staging 0.62 (16 cards at most: counted after the fit) + headroom 8.00 GiB"
	};
	const char *const counted_parts[] = {
		" + load staging 0.62 + headroom"
	};
	expect(printed(&bounded, ENGRAM_RAM_AUTO, 0, bounded_parts,
		       COUNT_OF(bounded_parts)) &&
		       printed(&share, ENGRAM_RAM_AUTO, 0, counted_parts,
			       COUNT_OF(counted_parts)),
	       "the budget line names the staging's card bound, and only when the cards are not counted");
	const struct engram_fit_needs inside =
		synthetic_needs(engram_fit_need_bytes(&base, 0));
	const char *const inside_parts[] = {
		"(auto: the disk store only, inside the headroom)",
		"engram ram fit: warning: need "
	};
	expect(printed(&inside, ENGRAM_RAM_AUTO, 0, inside_parts,
		       COUNT_OF(inside_parts)),
	       "the disk store inside the headroom is warned");
	const struct engram_fit_needs over = synthetic_needs(
		engram_fit_need_bytes(&base, 0) / SHORT_DIVISOR);
	const char *const over_parts[] = {
		"engram ram fit: refused: the model's own host memory",
		"its largest item is other host-only units (1.00 GiB)",
		"not Engram's"
	};
	expect(printed(&over, ENGRAM_RAM_AUTO, 0, over_parts,
		       COUNT_OF(over_parts)),
	       "the refusal names the shortfall and the model's largest item");
	const char *const forced_parts[] = {
		"engram ram percent: 50 (--engram_ram_percent; forced, it does not fit)",
		"the given percent is kept"
	};
	expect(printed(&over, ENGRAM_RAM_KNOB, GIVEN_PERCENT, forced_parts,
		       COUNT_OF(forced_parts)),
	       "a knob that does not fit is kept, warned");
	const char *const profile_parts[] = {
		"engram ram percent: 100 (storage profile turbo_engram)\n"
	};
	const struct engram_fit_needs plenty = synthetic_needs(UINT64_MAX);
	expect(printed(&plenty, ENGRAM_RAM_PROFILE, ENGRAM_RAM_PERCENT,
		       profile_parts, COUNT_OF(profile_parts)),
	       "a profile's percent names its source");
	/* auto's disk refusal names what the storage takes: under auto --engram_sources alone is refused */
	char direct[LINE_BYTES] = "", map[LINE_BYTES] = "";
	FILE *const direct_stream = fmemopen(direct, sizeof(direct) - 1, "w");
	FILE *const map_stream = fmemopen(map, sizeof(map) - 1, "w");
	if (direct_stream != NULL)
		engram_fit_disk_refusal_print(direct_stream, SHARE_PERCENT,
					      true);
	if (map_stream != NULL)
		engram_fit_disk_refusal_print(map_stream, SHARE_PERCENT, false);
	if (direct_stream != NULL)
		(void)fclose(direct_stream);
	if (map_stream != NULL)
		(void)fclose(map_stream);
	expect(strstr(direct, "engram ram fit: refused: auto chose 37 ") ==
			       direct &&
		       strstr(direct, "turbo load's replicas, or "
				      "--engram_sources=PATH with "
				      "--engram_ram_percent=0..99") != NULL &&
		       strstr(direct, "--engram_ram_percent=100\n") != NULL,
	       "a direct pack's disk refusal names the replicas, and the sources with a percent");
	expect(strstr(map, "engram ram fit: refused: auto chose 37 ") == map &&
		       strstr(map, "replicas,") == NULL &&
		       strstr(map, "a flat pack on local media as "
				   "--engram_sources=PATH with "
				   "--engram_ram_percent=0..99") != NULL &&
		       strstr(map, "--engram_ram_percent=100\n") != NULL,
	       "an EXL3 map's disk refusal names a flat pack with a percent, not its replicas");
}

int main(void)
{
	(void)alarm(TEST_SECONDS);
	decisions();
	store_sizes();
	pack_needs();
	meminfo();
	lines();
	if (failures != 0) {
		(void)fprintf(stderr, "engram_fit_test: %u failures\n",
			      failures);
		return EXIT_FAILURE;
	}
	(void)puts("engram_fit_test: passed");
	return EXIT_SUCCESS;
}
