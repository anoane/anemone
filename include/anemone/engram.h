/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram.h: the rows of a model's n-gram tables: which rows a token needs, and their bytes.
 *
 * Contract:
 *   - A model with n-gram tables carries them in its pack. Each table is a host-only unit of ONE extent that holds
 *     its rows: a row is ENGRAM_ROW_BYTES, the values of the checkpoint's row and then its scales, and
 *     ENGRAM_ROWS_PER_BLOCK rows fill the start of every pack block, so that no row crosses a block and
 *     one block read from a drive gives a whole row. Row r lies in block r / ROWS_PER_BLOCK at byte
 *     (r % ROWS_PER_BLOCK) * ROW_BYTES; the rest of a block, and the rows behind the last one, are zero. The
 *     extent's one tensor is "<table name>.rows", UINT8, [blocks, PACK_BLOCK_BYTES]; the converter names a
 *     table as its checkpoint does, and the engine finds a table by its unit, not by that name.
 *   - The constants of the hashing are tensors the converter computed with the model's own reference code, so
 *     that no tokenizer normaliser, random generator or prime search is rebuilt here. They lie in one extent of a
 *     host-only unit (the converter writes a unit of that one extent, behind every other unit), which
 *     engram_open reads through the storage reader (the pack itself holds no extent bytes) and checks
 *     against the extent's digest:
 *       engram.token_map     INT32 [vocabulary]            a token id to its id in the smaller space n-grams use
 *       engram.pad_id        INT32 [1]                     the smaller-space id that stands for "no token"
 *       engram.multipliers   INT64 [tables, lookback]      odd, one per table and distance back
 *       engram.primes        INT64 [tables, lookback - 1, heads]
 *       engram.offsets       INT64 [tables, rows per token]   where each (n-gram size, head) range of rows begins
 *       engram.row_counts    INT64 [tables]
 *       engram.table_units   INT32 [tables]                the unit of the pack that holds each table
 *     rows per token is (lookback - 1) * heads. engram_open finds them and refuses, with
 *     STATUS_CORRUPT_PACK and one error line that names the rule, a pack in which one is missing, has
 *     another type or shape, or does not lie in the extent of engram.token_map, or that extent's unit is not
 *     host-only; an extent whose bytes do not give its digest; a count that is zero (no table, a look-back below
 *     2, no head, an empty token map) or above the limits below; a token_map entry or pad_id that is negative, or
 *     whose product with a multiplier does not fit signed 64 bits (so no XOR below is negative); a multiplier that
 *     is even or not positive; a prime below ENGRAM_SMALLEST_PRIME; an offset that is negative, or a range
 *     (offset plus prime) that ends behind its table's row count; a table unit that is not host-only, does not
 *     have exactly one extent, or whose extent is not the blocks its row count needs. A pack without
 *     "engram.token_map" has no n-gram tables: STATUS_UNSUPPORTED, no error line. A read that fails keeps
 *     the storage's status (the next copy is tried first, as the pack does).
 *   - The rows of the token at a position (engram_row_ids_entry), exactly as the reference computes them:
 *     with t[d] the smaller-space id of the token d places back (d = 0 is the token itself), replaced by pad_id
 *     from the first d that reaches before the start of the sequence or a dead token (and for every d after it),
 *     and p[d] = t[d] * multipliers[table][d] in signed 64-bit arithmetic: for n = 1 .. lookback - 1 and every
 *     head h, row (n - 1) * heads + h of the table is (p[0] XOR .. XOR p[n]) modulo primes[table][n - 1][h], plus
 *     offsets[table][(n - 1) * heads + h]. A dead token (inside an image) is one whose dead flag is set.
 *   - engram_attach_entry gives the address where a table's extent lies once it is loaded; engram_gather_entry
 *     then copies rows by plain copies (measured: 48 rows in 2.6 us from host memory on 2 MiB pages; prefetching
 *     the addresses first gains nothing). Gathering from a table that is not attached is
 *     STATUS_INVALID_ARGUMENT.
 *   - An engram object is immutable after open except for attach, which the loader's caller does once per table
 *     before any thread gathers; row_ids and gather may then run from any number of threads.
 */
#ifndef ENGRAM_H
#define ENGRAM_H

#include <stdint.h>

#include "anemone/pack.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"

#define ENGRAM_VALUE_BYTES_VALUE 256
#define ENGRAM_SCALE_BYTES 8
#define ENGRAM_ROW_BYTES (ENGRAM_VALUE_BYTES_VALUE + ENGRAM_SCALE_BYTES)
#define ENGRAM_ROWS_PER_BLOCK (PACK_BLOCK_BYTES / ENGRAM_ROW_BYTES)
#define ENGRAM_MAXIMUM_TABLES 4
#define ENGRAM_MAXIMUM_LOOKBACK 8
#define ENGRAM_MAXIMUM_ROWS_PER_TOKEN 64
#define ENGRAM_SMALLEST_PRIME 2

struct engram_description {
	uint32_t table_count;
	uint32_t lookback; /* tokens an n-gram may span, the token itself included */
	uint32_t head_count;
	uint32_t rows_per_token; /* per table: (lookback - 1) * head_count */
	uint32_t vocabulary_size; /* token ids are below it */
	uint32_t reserved_zero;
	uint64_t row_count[ENGRAM_MAXIMUM_TABLES];
	uint32_t table_unit_index
		[ENGRAM_MAXIMUM_TABLES]; /* where the loader's layout puts each table */
};

struct engram_record;

/* First user: the program's load, before anything is loaded. storage and reader are the open copies of the pack.
 * On failure engram_out is untouched. The pack and the reader may be closed afterwards. */
MUST_CHECK result_status
engram_open(const struct storage_operations_record *storage,
	    const struct storage_reader *reader, const struct pack_record *pack,
	    struct engram_record **engram_out);
void engram_close(struct engram_record *engram); /* accepts NULL */
const struct engram_description *
engram_description_of(const struct engram_record *engram);

/* rows is where the table's extent lies in host memory, valid for as long as the engram is used. A NULL argument
 * or a table index at or above the count is STATUS_INVALID_ARGUMENT. */
MUST_CHECK result_status engram_attach_entry(struct engram_record *engram,
					     uint32_t table_index,
					     const void *rows);

/* The rows of the token at token_ids[position], for every table: row_ids_out receives table_count * rows_per_token
 * values, table after table. token_is_dead is NULL (no dead token) or one byte per token of the sequence, not zero
 * for a dead one. A NULL argument or a token id at or above the vocabulary is STATUS_INVALID_ARGUMENT. */
MUST_CHECK result_status engram_row_ids_entry(
	const struct engram_record *engram, const uint32_t *token_ids,
	const uint8_t *token_is_dead, uint64_t position, uint64_t *row_ids_out);

/* Copies row_count rows of one table, in the order of row_ids, into rows_out (row_count * ENGRAM_ROW_BYTES).
 * A row id at or above the table's row count is STATUS_INVALID_ARGUMENT and rows_out is then unspecified. */
MUST_CHECK result_status engram_gather_entry(const struct engram_record *engram,
					     uint32_t table_index,
					     const uint64_t *row_ids,
					     uint32_t row_count,
					     void *rows_out);

/* Asks the processor for every cache line of row_count rows of one table ahead of their gather, so that their
 * misses overlap; a row id out of range, a NULL engram or rows not attached ask nothing. No effect on any value. */
void engram_prefetch_entry(const struct engram_record *engram,
			   uint32_t table_index, const uint64_t *row_ids,
			   uint32_t row_count);

#endif
