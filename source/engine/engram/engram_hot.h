/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_hot.h: a static hot set of n-gram table blocks, held in RAM beside a disk Engram store (engram_disk.h) and
 * served like its RAM prefix.
 *
 * File (little-endian; written by `engram_io_bench hot_build` from a trace corpus that is not the served prompts):
 *   "ANMHOTS1" (8 bytes), schema u32 (ENGRAM_HOT_SCHEMA), section count u32,
 *   the pack's header digest u64, tables digest u64 and file bytes u64; then each section:
 *   table index u32, zero u32, the table extent's digest u64, the table's block count u64, id count u64, and that
 *   many block ids u64, strictly increasing and below the block count. Nothing follows the last section.
 * Contract:
 *   - engram_hot_open refuses, naming the reason in reason_out: a path that is not a regular file owned by the
 *     process's user (opened O_NOFOLLOW), a file longer than budget_bytes allows, a SHA-256 other than the expected
 *     one (64 hex digits: what the storage profile or the operator recorded), another magic or schema, another pack
 *     (header digest, tables digest or size), a table index out of range or repeated, a table of another extent
 *     digest or block count, ids out of order, repeated or out of range, blocks over budget_bytes, or bytes after
 *     the last section.
 *   - It then reads every hot block through the store's verified path (each block against its pack digest, with
 *     copy fallback), on a few threads, into RAM: nothing unverified is held. A block that no copy delivers fails
 *     the open with that status.
 *   - engram_hot_block gives a held block's bytes or NULL; it is read-only and safe from any thread once open.
 *   - engram_hot_encode lays out a file's bytes (the builder's half); the caller frees *bytes_out.
 */
#ifndef ENGRAM_HOT_H
#define ENGRAM_HOT_H

#include <stddef.h>
#include <stdint.h>

#include "engram_disk.h"
#include "engine/formats/pack/pack.h"

#define ENGRAM_HOT_SCHEMA 1u
#define ENGRAM_HOT_BUDGET_DEFAULT_MIB 512u
#define ENGRAM_HOT_BUDGET_MOST_MIB 65536u
#define ENGRAM_HOT_REASON_BYTES 192u
#define ENGRAM_HOT_SHA256_HEX_BYTES 64u

struct engram_hot;
struct engram_hot_table {
	uint32_t table;
	uint64_t extent_digest, block_count, id_count;
	const uint64_t *ids;
};

MUST_CHECK result_status
engram_hot_encode(const struct pack_header *header,
		  const struct engram_hot_table *tables, uint32_t table_count,
		  unsigned char **bytes_out, size_t *byte_count_out);
MUST_CHECK result_status engram_hot_open(
	const char *path, const char *sha256_hex, uint64_t budget_bytes,
	const struct pack_record *pack, struct engram_disk *disk,
	struct engram_hot **hot_out, char reason_out[ENGRAM_HOT_REASON_BYTES]);
/* The ids of a hot-set file with every check engram_hot_open makes but the reads (for the builder's evaluation):
 * expected[table] names each table's extent digest and block count. On STATUS_OK ids_out[table] is malloc'd (each
 * freed by the caller, NULL for a table the file does not list) and counts_out[table] its count. */
MUST_CHECK result_status
engram_hot_read(const char *path, const char *sha256_hex, uint64_t budget_bytes,
		const struct pack_header *header,
		const struct engram_hot_table *expected, uint32_t table_count,
		uint64_t *ids_out[ENGRAM_MAXIMUM_TABLES],
		uint64_t counts_out[ENGRAM_MAXIMUM_TABLES],
		char reason_out[ENGRAM_HOT_REASON_BYTES]);
void engram_hot_close(struct engram_hot *hot);
const unsigned char *engram_hot_block(const struct engram_hot *hot,
				      uint32_t table, uint64_t block);
uint64_t engram_hot_block_count(const struct engram_hot *hot);

#endif
