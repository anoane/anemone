/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_pack.c: the pack of engram_pack.h, written field by field as pack.h lays it out: the digests of its extents
 * here, those of its header and tables by test_pack_reseal (pack_builder.h).
 */
#include "engram_pack.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack.h"
#include "anemone/pack_digest.h"
#include "memory_storage.h"
#include "pack_builder.h"

#define UNITS 4u                                        /* a stage unit, the two tables, the constants */
#define BLOCKS 7u                                       /* block 0, the tables, the four extents */
#define PACK_BYTES ((uint64_t)BLOCKS * PACK_BLOCK_BYTES)
#define FIRST_EXTENT_BLOCK 2u
#define CONSTANTS_EXTENT 3u
#define LOOKBACK 2u
#define TOKEN_MAP_SPACE 11u

enum tensor { TENSOR_STAGE, TENSOR_MULTIPLIERS, TENSOR_OFFSETS, TENSOR_PAD_ID, TENSOR_PRIMES, TENSOR_ROW_COUNTS,
              TENSOR_TABLE_UNITS, TENSOR_TOKEN_MAP, TENSOR_TABLE_0, TENSOR_TABLE_1, TENSORS };

/* The pack into PACK_BYTES zeroed bytes, its extents' digests taken; the header's and the tables' are not. */
MUST_CHECK static result_status pack_write(unsigned char *pack_bytes)
{
    static const uint32_t block_counts[UNITS] = {1, 1, 2, 1};
    static const int64_t primes[TEST_ENGRAM_PACK_TABLES] = {13, 17};
    static const int64_t multipliers[TEST_ENGRAM_PACK_TABLES * LOOKBACK] = {3, 5, 7, 9};
    static const char *const names[TENSORS] = {
        "a.stage", "engram.multipliers", "engram.offsets", "engram.pad_id", "engram.primes", "engram.row_counts",
        "engram.table_units", "engram.token_map", "table0.rows", "table1.rows"};
    static const uint32_t types[TENSORS] = {
	    PACK_ELEMENT_INT8,	PACK_ELEMENT_INT64, PACK_ELEMENT_INT64,
	    PACK_ELEMENT_INT32, PACK_ELEMENT_INT64, PACK_ELEMENT_INT64,
	    PACK_ELEMENT_INT32, PACK_ELEMENT_INT32, PACK_ELEMENT_UINT8,
	    PACK_ELEMENT_UINT8
    };
    static const uint32_t extents_of[TENSORS] = {0, 3, 3, 3, 3, 3, 3, 3, 1, 2};
    static const uint32_t shapes[TENSORS][PACK_MAXIMUM_DIMENSIONS] = {
	    { 4096 },	 { 2, 2 },   { 2, 1 }, { 1 },
	    { 2, 1, 1 }, { 2 },	     { 2 },    { TEST_ENGRAM_PACK_VOCABULARY },
	    { 1, 4096 }, { 2, 4096 }
    };
    static const uint64_t tensor_bytes[TENSORS] = {4096, 32, 16, 4, 16, 16, 8, 64, 4096, 8192};
    static const uint64_t tensor_offsets[TENSORS] = {0, 0, 64, 128, 192, 256, 320, 384, 0, 0};
    struct pack_header *const header = (void *)pack_bytes;
    struct pack_extent *const extents = (void *)(pack_bytes + PACK_BLOCK_BYTES);
    struct pack_unit_record *const units = (void *)(extents + UNITS);
    struct pack_tensor *const tensors = (void *)(units + UNITS);
    char *const name_bytes = (char *)(tensors + TENSORS);
    result_status status = STATUS_OK;
    *header = (struct pack_header){ .magic = PACK_MAGIC,
				    .format_version = PACK_FORMAT_VERSION,
				    .block_byte_count = PACK_BLOCK_BYTES,
				    .model_name = "engram-test-model",
				    .file_byte_count = PACK_BYTES,
				    .tables_byte_count = PACK_BLOCK_BYTES,
				    .extent_count = UNITS,
				    .unit_count = UNITS,
				    .tensor_count = TENSORS };
    uint64_t byte_offset = (uint64_t)FIRST_EXTENT_BLOCK * PACK_BLOCK_BYTES;
    for (uint32_t unit = 0; unit < UNITS; unit++) {
	    extents[unit] =
		    (struct pack_extent){ .byte_offset = byte_offset,
					  .unit_index = unit,
					  .byte_count =
						  (uint64_t)block_counts[unit] *
						  PACK_BLOCK_BYTES };
	    units[unit] = (struct pack_unit_record){
		    .unit_kind = unit == 0 ? PLACEMENT_UNIT_STAGE :
					     PLACEMENT_UNIT_HOST_ONLY,
		    .layer_index = unit == 0 ? 0 : PACK_NO_LAYER
	    };
	    byte_offset += extents[unit].byte_count;
    }
    for (uint32_t tensor = 0; tensor < TENSORS; tensor++) {
        const uint32_t name_length = (uint32_t)strlen(names[tensor]);
	tensors[tensor] = (struct pack_tensor){
		.byte_offset_in_extent = tensor_offsets[tensor],
		.byte_count = tensor_bytes[tensor],
		.extent_index = extents_of[tensor],
		.name_byte_offset = header->name_byte_count,
		.name_byte_count = name_length,
		.element_type = types[tensor]
	};
	for (uint32_t dimension = 0; dimension < PACK_MAXIMUM_DIMENSIONS;
	     dimension++) {
		tensors[tensor].dimensions[dimension] =
			shapes[tensor][dimension];
		tensors[tensor].dimension_count +=
			shapes[tensor][dimension] != 0 ? 1u : 0u;
	}
	memcpy(name_bytes + header->name_byte_count, names[tensor], name_length);
        header->name_byte_count += name_length;
    }
    unsigned char *const constants = pack_bytes + extents[CONSTANTS_EXTENT].byte_offset;
    for (uint32_t table = 0; table < TEST_ENGRAM_PACK_TABLES; table++) {
        const int32_t table_unit = (int32_t)table + 1;      /* a table's row count is its one prime; offsets 0 */
        memcpy(constants + tensor_offsets[TENSOR_PRIMES] + table * sizeof(int64_t), &primes[table], sizeof(int64_t));
        memcpy(constants + tensor_offsets[TENSOR_ROW_COUNTS] + table * sizeof(int64_t), &primes[table],
               sizeof(int64_t));
        memcpy(constants + tensor_offsets[TENSOR_TABLE_UNITS] + table * sizeof(int32_t), &table_unit,
               sizeof(table_unit));
        for (uint64_t byte = 0; byte < extents[table + 1u].byte_count; byte++) {
            pack_bytes[extents[table + 1u].byte_offset + byte] =
                (unsigned char)(byte * TEST_ENGRAM_PACK_PATTERN_MULTIPLIER + table);
        }
    }
    memcpy(constants + tensor_offsets[TENSOR_MULTIPLIERS], multipliers, sizeof(multipliers));
    for (uint32_t token = 0; token < TEST_ENGRAM_PACK_VOCABULARY; token++) {
        const int32_t mapped = (int32_t)(token % TOKEN_MAP_SPACE);
        memcpy(constants + tensor_offsets[TENSOR_TOKEN_MAP] + token * sizeof(int32_t), &mapped, sizeof(mapped));
    }
    for (uint32_t extent = 0; extent < UNITS && status == STATUS_OK; extent++) {
	    status = pack_digest_of_host_bytes(
		    pack_bytes + extents[extent].byte_offset,
		    extents[extent].byte_count, &extents[extent].digest);
    }
    return status;
}

result_status test_engram_pack_open(unsigned char **pack_out,
				    struct engram_record **engram_out)
{
    const char *const copy_path = "copy";
    struct storage_reader *reader = NULL;
    struct pack_record *pack = NULL;
    struct engram_record *engram = NULL;
    if (pack_out == NULL || engram_out == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    unsigned char *const pack_bytes =
	    aligned_alloc(PACK_BLOCK_BYTES, (size_t)PACK_BYTES);
    if (pack_bytes == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    memset(pack_bytes, 0, (size_t)PACK_BYTES);
    const struct pack_extent *const extents =
	    (const void *)(pack_bytes + PACK_BLOCK_BYTES);
    result_status status = pack_write(pack_bytes);
    if (status == STATUS_OK) {
	    test_pack_reseal(pack_bytes);
    }
    test_memory_storage_reset();
    test_memory_storage_set_pack(pack_bytes, PACK_BYTES);
    status = status != STATUS_OK ?
		     status :
		     test_memory_storage.reader_open(&copy_path, 1, &reader);
    status = status != STATUS_OK ?
		     status :
		     pack_open_entry(&test_memory_storage, reader, &pack);
    status = status != STATUS_OK ?
		     status :
		     engram_open(&test_memory_storage, reader, pack, &engram);
    for (uint32_t table = 0;
	 table < TEST_ENGRAM_PACK_TABLES && status == STATUS_OK; table++) {
	    status = engram_attach_entry(
		    engram, table,
		    pack_bytes + extents[table + 1u].byte_offset);
    }
    pack_close(pack);
    test_memory_storage.reader_close(reader);
    if (status != STATUS_OK) {
	    engram_close(engram);
	    test_memory_storage_reset(); /* it forgets the bytes before they are freed */
	    free(pack_bytes);
	    return status;
    }
    *pack_out = pack_bytes;
    *engram_out = engram;
    return STATUS_OK;
}
