/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_pack.h: a small pack of n-gram tables in memory, and its n-gram object opened with every table attached,
 * for the tests that need n-gram rows without the model's pack (the step driver's). Test support: no part of the
 * engine.
 *
 * Contract:
 *   - The pack obeys pack.h and engram.h: unit 0 a stage unit, then a host-only unit for each of the
 *     TEST_ENGRAM_PACK_TABLES tables and one of the constants. A look-back of 2 and heads heads: heads rows a token
 *     and table (test_engram_pack_open: TEST_ENGRAM_PACK_HEADS, one). Token t maps to t % 11 of a vocabulary of
 *     TEST_ENGRAM_PACK_VOCABULARY; every head of table t has the table's prime (13, 17), and head h's rows are the
 *     h-th run of that many rows (its offset h times the prime), so table t holds heads times its prime of rows;
 *     multipliers 3, 5, 7, 9; byte b of table t's extent is b times TEST_ENGRAM_PACK_PATTERN_MULTIPLIER plus t,
 *     truncated to a byte.
 *   - test_engram_pack_open_heads writes it (one allocation aligned to PACK_BLOCK_BYTES, which the caller frees
 *     after closing the object), opens it through test_memory_storage (reset first; the storage keeps the bytes as
 *     its pack) and attaches each table where its extent lies. heads of 0 or above ENGRAM_MAXIMUM_ROWS_PER_TOKEN,
 *     or a NULL output, is STATUS_INVALID_ARGUMENT; another failure is the status of the memory, the storage, the
 *     pack, the n-gram object or a digest. A failure leaves both outputs untouched and nothing allocated.
 */
#ifndef TEST_ENGRAM_PACK_H
#define TEST_ENGRAM_PACK_H

#include <stdint.h>

#include "anemone/engram.h"
#include "anemone/status.h"

#define TEST_ENGRAM_PACK_TABLES 2u
#define TEST_ENGRAM_PACK_HEADS 1u
#define TEST_ENGRAM_PACK_VOCABULARY 16u
#define TEST_ENGRAM_PACK_PATTERN_MULTIPLIER 6364136223846793005ull

MUST_CHECK result_status test_engram_pack_open(
	unsigned char **pack_out, struct engram_record **engram_out);
MUST_CHECK result_status
test_engram_pack_open_heads(uint32_t heads, unsigned char **pack_out,
			    struct engram_record **engram_out);

#endif
