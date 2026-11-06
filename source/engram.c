/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram.c: which rows of the n-gram tables a token needs, and their bytes. Contract: engram.h.
 *
 * An engram is ONE allocation: the object in its first block and behind it the extent of the constants, as the first
 * copy gave it with its digest; row_ids walks them where they lie. Open bounds all they hold, so row_ids checks only
 * token ids and gather only row ids. In open a rule is looked at only while none is broken, and no value before its
 * shape; the token map stands for a constant that is not in its extent until then. Fault point: engram_memory.
 */
#include "anemone/engram.h"

#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/pack_digest.h"

#define ENGRAM_CACHE_LINE_BYTES 64u /* prefetch: one request a line of a row */

/* The constants: the 32-bit ones, then by their dimensions. Open tells both from this order; the names follow it. */
enum constant { TOKEN_MAP, PAD_ID, TABLE_UNITS, ROW_COUNTS, MULTIPLIERS, OFFSETS, PRIMES, CONSTANT_COUNT };
enum dimension { OF_TABLES, OF_DISTANCES, OF_HEADS };    /* of engram.primes; the others share them from the first */
static const char *const names[CONSTANT_COUNT] = {"engram.token_map", "engram.pad_id", "engram.table_units",
    "engram.row_counts", "engram.multipliers", "engram.offsets", "engram.primes"};

struct engram_record {
	struct engram_description description;
	const unsigned char *rows
		[ENGRAM_MAXIMUM_TABLES]; /* attach: where the extent of each table lies */
	const int32_t *
		token_map; /* these four lie in the extent of the constants, a block behind */
	const int64_t *multipliers; /* [table][distance back] */
	const int64_t *primes; /* [table][row of a token] */
	const uint64_t
		*offsets; /* [table][row of a token]: open saw none negative */
	int64_t pad_id;
};
_Static_assert(sizeof(struct engram_record) <= PACK_BLOCK_BYTES,
	       "the constants begin one block behind it");

result_status engram_open(const struct storage_operations_record *storage,
			  const struct storage_reader *reader,
			  const struct pack_record *pack,
			  struct engram_record **engram_out)
{
    const uint32_t *shapes[CONSTANT_COUNT] = {NULL};
    const void *values[CONSTANT_COUNT] = {NULL};
    struct storage_description copies = { 0 };
    uint64_t digest = 0;
    if (storage == NULL || reader == NULL || pack == NULL || engram_out == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct pack_tensor *const tensors = pack_tensors_of(pack);
    const struct pack_extent *const extents = pack_extents_of(pack);
    const struct pack_unit_record_2 *const units = pack_units_of(pack);
    const uint32_t map_index = pack_find_tensor(
	    pack, names[TOKEN_MAP], (uint32_t)strlen(names[TOKEN_MAP]));
    if (map_index == PACK_NOT_FOUND) {
	    return STATUS_UNSUPPORTED; /* a model without n-gram tables */
    }
    const struct pack_extent *const extent =
	    &extents[tensors[map_index].extent_index];
    result_status status = storage->reader_describe(reader, &copies);
    struct engram_record *const engram =
	    status != STATUS_OK || FAULT_POINT(engram_memory) ?
		    NULL :
		    aligned_alloc(PACK_BLOCK_BYTES,
				  PACK_BLOCK_BYTES + extent->byte_count);
    if (engram == NULL) {
	    return status != STATUS_OK ? status : STATUS_OUT_OF_MEMORY;
    }
    unsigned char *const constants = (unsigned char *)engram + PACK_BLOCK_BYTES;
    status =
	    STATUS_INVALID_ARGUMENT; /* what a reader without a copy, or with too many, is: as for a pack */
    for (uint32_t copy_index = 0;
	 status != STATUS_OK && copy_index < copies.copy_count &&
	 copies.copy_count <= STORAGE_MAXIMUM_COPIES;
	 copy_index++) {
	    status = storage->reader_read(reader, copy_index,
					  extent->byte_offset, constants,
					  extent->byte_count);
	    status = status == STATUS_OK ?
			     pack_digest_of_host_bytes(
				     constants, extent->byte_count, &digest) :
			     status;
	    status = status == STATUS_OK && digest != extent->digest ?
			     STATUS_CORRUPT_PACK :
			     status;
    }
    const char *rule =
	    status != STATUS_OK ?
		    "a copy gives the extent of the token map with its digest" :
		    NULL;
    for (uint32_t constant = 0; constant < CONSTANT_COUNT; constant++) {
	    const uint32_t found = pack_find_tensor(
		    pack, names[constant], (uint32_t)strlen(names[constant]));
	    const bool is_there = found != PACK_NOT_FOUND &&
				  extents + tensors[found].extent_index ==
					  extent;
	    const struct pack_tensor *const tensor =
		    &tensors[is_there ? found : map_index];
	    shapes[constant] = tensor->dimensions;
	    values[constant] = constants + tensor->byte_offset_in_extent;
	    rule = rule != NULL || (is_there &&
				    units[extent->unit_index].unit_kind ==
					    PLACEMENT_UNIT_HOST_ONLY &&
				    tensor->element_type ==
					    (constant < ROW_COUNTS ?
						     PACK_ELEMENT_INT32 :
						     PACK_ELEMENT_INT64) &&
				    tensor->dimension_count ==
					    1u +
						    (constant >= MULTIPLIERS ?
							     1u :
							     0u) +
						    (constant == PRIMES ? 1u :
									  0u)) ?
			   rule :
			   "each is there with its type and dimensions, in one extent of a host-only unit";
    }
    const uint32_t table_count = shapes[ROW_COUNTS][OF_TABLES];
    const uint32_t lookback = shapes[MULTIPLIERS][OF_DISTANCES];
    const uint64_t rows_per_token = ((uint64_t)lookback - 1) * shapes[PRIMES][OF_HEADS];    /* wraps at lookback 0 */
    rule = rule != NULL ? rule :
	   shapes[TOKEN_MAP][0] == 0 || table_count == 0 ||
			   rows_per_token == 0 || lookback <= 1 ||
			   lookback > ENGRAM_MAXIMUM_LOOKBACK ||
			   table_count > ENGRAM_MAXIMUM_TABLES ||
			   rows_per_token > ENGRAM_MAXIMUM_ROWS_PER_TOKEN ?
			  "no count is zero or above its limit" :
	   shapes[PAD_ID][0] != 1 ||
			   shapes[TABLE_UNITS][OF_TABLES] != table_count ||
			   shapes[MULTIPLIERS][OF_TABLES] != table_count ||
			   shapes[OFFSETS][OF_TABLES] != table_count ||
			   shapes[OFFSETS][1] != rows_per_token ||
			   shapes[PRIMES][OF_TABLES] != table_count ||
			   shapes[PRIMES][OF_DISTANCES] != lookback - 1 ?
			  "each has the shape the counts give it" :
			  NULL;
    *engram = (struct engram_record){
	    .description = { .table_count = table_count,
			     .lookback = lookback,
			     .head_count = shapes[PRIMES][OF_HEADS],
			     .rows_per_token = (uint32_t)rows_per_token,
			     .vocabulary_size = shapes[TOKEN_MAP][0] },
	    .token_map = values[TOKEN_MAP],
	    .multipliers = values[MULTIPLIERS],
	    .primes = values[PRIMES],
	    .offsets = values[OFFSETS],
	    .pad_id = rule == NULL ? *(const int32_t *)values[PAD_ID] : 0
    };
    for (uint32_t factor_index = 0; rule == NULL && factor_index < table_count * lookback; factor_index++) {
        const int64_t multiplier = engram->multipliers[factor_index];
        rule = multiplier <= 0 || ((uint64_t)multiplier & 1u) == 0 ? "a multiplier is odd and positive" : NULL;
        for (uint64_t token_id = 0; rule == NULL && token_id <= shapes[TOKEN_MAP][0]; token_id++) {   /* last: pad_id */
            const int64_t id = token_id < shapes[TOKEN_MAP][0] ? engram->token_map[token_id] : engram->pad_id;
            rule = id < 0 || id > INT64_MAX / multiplier ? "an id is not negative and overflows no product" : NULL;
        }
    }
    for (uint32_t row = 0; rule == NULL && row < table_count * rows_per_token; row++) {
        const uint32_t table_index = row / (uint32_t)rows_per_token;
        const uint32_t unit_index = ((const uint32_t *)values[TABLE_UNITS])[table_index];   /* negative: too large */
        const uint64_t row_count = ((const uint64_t *)values[ROW_COUNTS])[table_index];     /* below 1: no blocks */
	rule = unit_index >= pack_header_of(pack)->unit_count ||
			       units[unit_index].extent_count != 1 ||
			       units[unit_index].unit_kind !=
				       PLACEMENT_UNIT_HOST_ONLY ||
			       extents[units[unit_index].first_extent_index]
						       .byte_count /
					       PACK_BLOCK_BYTES !=
				       (row_count - 1) / ENGRAM_ROWS_PER_BLOCK +
					       1 ?
		       "the unit of a table is host-only and one extent of the blocks its row count needs" :
	       engram->primes[row] < ENGRAM_SMALLEST_PRIME ?
		       "a prime is at least the smallest prime" :
	       engram->offsets[row] > row_count ||
			       (uint64_t)engram->primes[row] >
				       row_count - engram->offsets[row] ?
		       "a range of rows, an offset and its prime, ends inside its table" :
		       NULL;
	engram->description.row_count[table_index] = row_count;
        engram->description.table_unit_index[table_index] = unit_index;
    }
    if (rule != NULL) {
	    LOG_ERROR("the n-gram tables do not open, broken rule: %s", rule);
	    free(engram);
	    return status != STATUS_OK ? status : STATUS_CORRUPT_PACK;
    }
    *engram_out = engram;
    return STATUS_OK;
}

void engram_close(struct engram_record *engram)
{
	free(engram);
} /* the one allocation; NULL as free */

const struct engram_description *
engram_description_of(const struct engram_record *engram)
{
    return engram != NULL ? &engram->description : NULL;
}

result_status engram_attach_entry(struct engram_record *engram,
				  uint32_t table_index, const void *rows)
{
    if (engram == NULL || rows == NULL || table_index >= engram->description.table_count) {
	    return STATUS_INVALID_ARGUMENT;
    }
    engram->rows[table_index] = rows;
    return STATUS_OK;
}

result_status engram_row_ids_entry(const struct engram_record *engram,
				   const uint32_t *token_ids,
				   const uint8_t *token_is_dead,
				   uint64_t position, uint64_t *row_ids_out)
{
	int64_t ids[ENGRAM_MAXIMUM_LOOKBACK] = { 0 }; /* t[d] of the contract */
	bool is_cut =
		false; /* the look-back, at the start or a dead token: for good */
	if (engram == NULL || token_ids == NULL || row_ids_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	TIMER_BEGIN(engram_row_ids);
	for (uint32_t distance = 0; distance < engram->description.lookback;
	     distance++) {
		is_cut = is_cut || distance > position ||
			 (token_is_dead != NULL &&
			  token_is_dead[position - distance] != 0);
		if (!is_cut && token_ids[position - distance] >=
				       engram->description.vocabulary_size) {
			return STATUS_INVALID_ARGUMENT;
		}
		ids[distance] =
			is_cut ?
				engram->pad_id :
				engram->token_map[token_ids[position - distance]];
	}
    for (uint32_t table_index = 0, row = 0; table_index < engram->description.table_count; table_index++) {
        const int64_t *const multipliers = engram->multipliers + (size_t)table_index * engram->description.lookback;
        uint64_t hash = (uint64_t)(ids[0] * multipliers[0]);     /* no product leaves signed 64 bits: open's bounds */
        for (uint32_t distance = 1; distance < engram->description.lookback; distance++) {
            hash ^= (uint64_t)(ids[distance] * multipliers[distance]);
            for (uint32_t head = 0; head < engram->description.head_count; head++, row++) {
                const int64_t prime = engram->primes[row];
                const int64_t residue = (int64_t)hash % prime;       /* then the sign of the divisor: torch's % */
                row_ids_out[row] = (uint64_t)(residue < 0 ? residue + prime : residue) + engram->offsets[row];
            }
        }
    }
    TIMER_END(engram_row_ids);
    return STATUS_OK;
}

void engram_prefetch_entry(const struct engram_record *engram,
			   uint32_t table_index, const uint64_t *row_ids,
			   uint32_t row_count)
{
    if (engram == NULL || row_ids == NULL || table_index >= engram->description.table_count ||
        engram->rows[table_index] == NULL) {
        return;
    }
    TIMER_BEGIN(engram_prefetch);
    for (uint32_t row_index = 0; row_index < row_count; row_index++) {
        const uint64_t row_id = row_ids[row_index];
        if (row_id >= engram->description.row_count[table_index]) {
            continue;
        }
	const unsigned char *const row =
		engram->rows[table_index] +
		row_id / ENGRAM_ROWS_PER_BLOCK * PACK_BLOCK_BYTES +
		row_id % ENGRAM_ROWS_PER_BLOCK * ENGRAM_ROW_BYTES;
	for (uint32_t line = 0; line < ENGRAM_ROW_BYTES;
	     line += ENGRAM_CACHE_LINE_BYTES) {
		__builtin_prefetch(row + line, 0, 0);
	}
	__builtin_prefetch(row + ENGRAM_ROW_BYTES - 1u, 0, 0);
    }
    TIMER_END(engram_prefetch);
}

result_status engram_gather_entry(const struct engram_record *engram,
				  uint32_t table_index, const uint64_t *row_ids,
				  uint32_t row_count, void *rows_out)
{
    if (engram == NULL || row_ids == NULL || rows_out == NULL || table_index >= engram->description.table_count ||
        engram->rows[table_index] == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    TIMER_BEGIN(engram_gather);
    for (uint32_t row_index = 0; row_index < row_count; row_index++) {
        const uint64_t row_id = row_ids[row_index];
        if (row_id >= engram->description.row_count[table_index]) {
		return STATUS_INVALID_ARGUMENT;
	}
	memcpy((unsigned char *)rows_out + (size_t)row_index * ENGRAM_ROW_BYTES,
	       engram->rows[table_index] +
		       row_id / ENGRAM_ROWS_PER_BLOCK * PACK_BLOCK_BYTES +
		       row_id % ENGRAM_ROWS_PER_BLOCK * ENGRAM_ROW_BYTES,
	       ENGRAM_ROW_BYTES);
    }
    COUNTER_ADD(engram_gather_bytes, (uint64_t)row_count * ENGRAM_ROW_BYTES);
    TIMER_END(engram_gather);
    return STATUS_OK;
}
