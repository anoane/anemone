/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_pack.h: a small pack of n-gram tables in memory, and its n-gram object opened with every table attached,
 * for the tests that need n-gram rows without the model's pack (the step driver's). Test support: no part of the
 * engine.
 *
 * Contract:
 *   - The pack obeys pack.h and engram.h: unit 0 a stage unit, then a host-only unit for each of the
 *     TEST_ENGRAM_PACK_TABLES tables and one of the constants. A look-back of 2 and one head: one row a token and
 *     table. Token t maps to t % 11 of a vocabulary of TEST_ENGRAM_PACK_VOCABULARY; table t has its prime (13, 17)
 *     of rows, multipliers 3, 5, 7, 9 and offsets 0; byte b of table t's extent is b times
 *     TEST_ENGRAM_PACK_PATTERN_MULTIPLIER plus t, truncated to a byte.
 *   - test_engram_pack_open writes it (one allocation aligned to PACK_BLOCK_BYTES, which the caller frees
 *     after closing the object), opens it through test_memory_storage (reset first; the storage keeps the bytes as
 *     its pack) and attaches each table where its extent lies. A failure is the status of the memory, the storage,
 *     the pack, the n-gram object or a digest; it leaves both outputs untouched and nothing allocated.
 */
#ifndef TEST_ENGRAM_PACK_H
#define TEST_ENGRAM_PACK_H

#include "anemone/engram.h"
#include "anemone/status.h"

#define TEST_ENGRAM_PACK_TABLES 2u
#define TEST_ENGRAM_PACK_VOCABULARY 16u
#define TEST_ENGRAM_PACK_PATTERN_MULTIPLIER 6364136223846793005ull

MUST_CHECK result_status test_engram_pack_open(
	unsigned char **pack_out, struct engram_record **engram_out);

#endif
