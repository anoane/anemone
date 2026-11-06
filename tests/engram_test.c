/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_test: source/engram.c against engram.h, without the real tables.
 *
 * The test builds small packs in memory: a stage unit, a host-only unit for each table, a spare host-only unit of
 * two extents, the host-only unit of the constants and a last stage unit. Row r of a table holds a pattern drawn
 * from the table and r. Every rule of engram.h is broken by one change to a valid pack, sealed again so that the
 * rule and not a digest refuses it; changes that break no rule are shown to open. Row ids are held against the
 * reference's formula written the long way and against a case computed by hand, rows against the pattern.
 * What engram.h rules out is a negative XOR: open refuses a negative id, a multiplier that is not positive and a
 * product beyond signed 64 bits, so no product has its sign bit set, nor has their XOR. The reference's rule for a
 * negative left operand of % is held in the second implementation, and the engine to it up to the largest products.
 */
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/engram.h"
#include "anemone/hooks.h"
#include "anemone/pack.h"
#include "anemone/pack_digest.h"
#include "anemone/status.h"
#include "memory_storage.h"

#define TEST_TIME_LIMIT_SECONDS 600             /* the alarm ends the test here, whatever a thread of it does */
#define BLOCK \
	PACK_BLOCK_BYTES /* short names for the many places of this test that need them */
#define ALIGNMENT WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES
#define ROW_BYTES ENGRAM_ROW_BYTES
#define ROWS_PER_BLOCK ENGRAM_ROWS_PER_BLOCK
#define ROW_WORDS (ROW_BYTES / PACK_DIGEST_WORD_BYTES)
#define BLOCK_WORDS (BLOCK / PACK_DIGEST_WORD_BYTES)
#define INT32 PACK_ELEMENT_INT32
#define INT64 PACK_ELEMENT_INT64
#define STAGE PLACEMENT_UNIT_STAGE
#define NO_ERROR STATUS_OK
#define INVALID STATUS_INVALID_ARGUMENT
#define OPENS STATUS_OK /* what an open is expected to answer */
#define REFUSED STATUS_CORRUPT_PACK
#define UNREAD STATUS_INPUT_OUTPUT_ERROR
#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
/* Where the extent of a table lies in the bytes of a pack: what a load would attach. */
#define TABLE_ROWS(pack, table_index) ((pack)->bytes + (pack)->extents[(table_index) + 1].byte_offset)
#define COPY_PAIR 2
#define THREAD_COUNT 8
#define DEAD (-1)                               /* the reference's mark of a dead token in its cache */
#define TENSOR_UNIT_KIND UINT32_MAX             /* in place of a tensor index: the kinds of the units */
#define RANDOM_SEED 0x243F6A8885A308D3ull
#define ROW_SEED 0x13198A2E03707344ull
#define WHAT_CAPACITY 128                       /* more than the longest name of a case */
#define COMPRESSED_VOCABULARY 97                /* the ids of the smaller space a pack is built with: shared */
#define MULTIPLIER_HALVES 2                     /* the reference draws half a multiplier, doubles it and adds one */
#define PAD_TOKEN_ID 2                          /* the token whose id of the smaller space is pad_id */
#define CONSTANTS_BLOCKS 4                      /* the extent of the constants: more than any shape here fills */
#define UNITS_BESIDE_TABLES 4                   /* two stage units, the spare unit, the unit of the constants */
#define EXTENTS_BESIDE_TABLES 5                 /* one for each of them, and the second of the spare unit */
#define TENSORS_BESIDE_TABLES 9                 /* one of each stage unit, the seven constants */
#define SPARE_EXTENT_BEHIND_TABLES 1            /* the first extent of the spare unit, counted from the last table's */
#define CONSTANTS_EXTENT_BEHIND_TABLES 3
#define BUILD_MAXIMUM_TABLES \
	(ENGRAM_MAXIMUM_TABLES + 1) /* one above each limit of the engine */
#define BUILD_MAXIMUM_LOOKBACK (ENGRAM_MAXIMUM_LOOKBACK + 1)
#define BUILD_MAXIMUM_ROWS (ENGRAM_MAXIMUM_ROWS_PER_TOKEN + 1)
#define ROW_IDS_CAPACITY (BUILD_MAXIMUM_TABLES * BUILD_MAXIMUM_ROWS)
/* The shape of most packs of this test: the model's, but for the size of the vocabulary. */
#define MODEL_TABLES 2
#define MODEL_LOOKBACK 4
#define MODEL_HEADS 8
#define MODEL_ROWS ((MODEL_LOOKBACK - 1) * MODEL_HEADS)
#define MODEL_VOCABULARY 300
#define MODEL_FACTORS (MODEL_TABLES * MODEL_LOOKBACK)
#define MODEL_ROW_IDS (MODEL_TABLES * MODEL_ROWS)
#define MODEL_FIRST_TABLE_ROWS 963              /* the sum of the first 24 primes */
#define MODEL_ROWS_LEFT_IN_LAST_BLOCK (ROWS_PER_BLOCK - MODEL_FIRST_TABLE_ROWS % ROWS_PER_BLOCK)
#define MODEL_UNIT_SPARE (MODEL_TABLES + 1)
#define SEQUENCE_TOKENS 4096
#define DEAD_ONE_IN 6                           /* one token in so many is dead in the sequence drawn at random */
#define LARGEST_ID 7                            /* 2^63 - 1 is seven times an odd number: the largest product */
#define LARGEST_MULTIPLIER (INT64_MAX / LARGEST_ID)
#define HAND_LOOKBACK 3                         /* the case computed by hand: one table, two heads */
#define HAND_HEADS 2
#define HAND_ROWS ((HAND_LOOKBACK - 1) * HAND_HEADS)
#define HAND_TOKENS 3
#define TORCH_DIVIDEND (-7)                     /* torch: -7 % 5 is 3 */
#define TORCH_DIVISOR 5
#define TORCH_REMAINDER 3

struct pack_shape {
    uint32_t table_count;
    uint32_t lookback;
    uint32_t head_count;
    uint32_t vocabulary_size;
    int64_t first_prime;                        /* the primes of the pack are the next ones from here on, none twice */
};

/* A pack in memory with its parts and its constants where they lie, for a test to change and seal again. */
struct test_pack {
    unsigned char *bytes;
    uint64_t byte_count;
    struct pack_shape shape;
    uint32_t rows_per_token;
    uint32_t tensors_added;
    struct pack_header *header;
    struct pack_extent *
	    extents; /* a stage unit's, the tables', two spare, the constants', a stage's */
    struct pack_unit_record *unit_records;
    struct pack_tensor *tensors;
    char *name_bytes;
    int32_t *token_map;
    int32_t *pad_id;
    int64_t *multipliers;
    int64_t *primes;
    int64_t *offsets;
    int64_t *row_counts;
    int32_t *table_units;
};

/* The tensors of a pack by the order of their names and records; behind them the tables' and a stage unit's. */
enum tensor_index { TENSOR_STAGE, TENSOR_MULTIPLIERS, TENSOR_OFFSETS, TENSOR_PAD_ID, TENSOR_PRIMES, TENSOR_ROW_COUNTS,
                    TENSOR_TABLE_UNITS, TENSOR_TOKEN_MAP };

static const struct pack_shape model_shape = {MODEL_TABLES, MODEL_LOOKBACK, MODEL_HEADS, MODEL_VOCABULARY, 2};
static const char *const copy_paths[COPY_PAIR] = {"copy 0", "copy 1"};
static struct test_pack model_pack;             /* of the model's shape, with its rows: for tests that change none */
static uint32_t sequence[SEQUENCE_TOKENS];      /* token ids drawn at random, and which of them are dead */
static uint8_t sequence_is_dead[SEQUENCE_TOKENS];
static uint64_t random_state = RANDOM_SEED;
static int failures = 0;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected_status,
			  const char *what)
{
    if (status != expected_status) {
	    (void)fprintf(stderr, "FAILED: %s: status \"%s\", not \"%s\"\n",
			  what, status_text(status),
			  status_text(expected_status));
	    failures++;
    }
}

static _Noreturn void give_up(const char *what)
{
    (void)fprintf(stderr, "FAILED: %s\n", what);
    exit(EXIT_FAILURE);
}

static uint64_t next_random(void)
{
	random_state += PACK_DIGEST_POSITION_MULTIPLIER;
	return pack_digest_mix(random_state);
}

/* The smallest prime above the number, by trial division: the primes of a test are small. */
static int64_t prime_above(int64_t number)
{
    for (bool is_prime = false; !is_prime;) {
        number++;
	is_prime = number >= ENGRAM_SMALLEST_PRIME;
	for (int64_t divisor = ENGRAM_SMALLEST_PRIME;
	     is_prime && divisor * divisor <= number; divisor++) {
		is_prime = number % divisor != 0;
	}
    }
    return number;
}

/* The pattern of a row: its words are a bijection of their place, so no two words of a table are the same. */
static void write_pattern(uint32_t table_index, uint64_t row_id, uint64_t *words_out)
{
	const uint64_t first_place =
		pack_digest_mix(ROW_SEED + table_index) + row_id * ROW_WORDS;
	for (uint64_t word_index = 0; word_index < ROW_WORDS; word_index++) {
		words_out[word_index] =
			pack_digest_mix(first_place + word_index);
	}
}

static bool row_is_the_pattern(uint32_t table_index, uint64_t row_id, const unsigned char *row)
{
    uint64_t words[ROW_WORDS];
    write_pattern(table_index, row_id, words);
    return memcmp(row, words, sizeof(words)) == 0;
}

/* Makes every digest of the pack hold again after a test changed it. */
static void seal_pack(struct test_pack *pack)
{
	struct pack_header *header = pack->header;
	uint64_t header_digest = 0;
	bool is_sealed = true;
	for (uint32_t extent_index = 0; extent_index < header->extent_count;
	     extent_index++) {
		struct pack_extent *extent = &pack->extents[extent_index];
		is_sealed = is_sealed &&
			    pack_digest_of_host_bytes(
				    pack->bytes + extent->byte_offset,
				    extent->byte_count,
				    &extent->digest) == NO_ERROR;
	}
    header->header_digest = 0;
    if (!is_sealed ||
	pack_digest_of_host_bytes(pack->bytes + BLOCK, BLOCK,
				  &header->tables_digest) != NO_ERROR ||
	pack_digest_of_host_bytes(pack->bytes, BLOCK, &header_digest) !=
		NO_ERROR) {
	    give_up("the digests of a pack of the test could not be taken");
    }
    header->header_digest = header_digest;
}

/* Appends the record of a tensor: it begins behind the tensor before it when that lies in the same extent, on the
 * next alignment. A dimension given as 0 is one the tensor has not. Returns where its bytes lie. */
static void *add_tensor(struct test_pack *pack, const char *name, uint32_t extent_index, uint32_t element_type,
                        uint32_t first, uint32_t second, uint32_t third)
{
	struct pack_tensor *tensor = &pack->tensors[pack->tensors_added];
	const struct pack_tensor *before =
		pack->tensors_added > 0 ? tensor - 1 : tensor;
	const uint64_t byte_offset =
		before != tensor && before->extent_index == extent_index ?
			(before->byte_offset_in_extent + before->byte_count +
			 ALIGNMENT - 1) /
				ALIGNMENT * ALIGNMENT :
			0;
	const uint64_t width = element_type == INT64 ? sizeof(int64_t) :
			       element_type == INT32 ? sizeof(int32_t) :
						       1;
	uint32_t name_byte_count = 0;
	while (name[name_byte_count] != '\0') {
		pack->name_bytes[pack->header->name_byte_count +
				 name_byte_count] = name[name_byte_count];
		name_byte_count++;
	}
	*tensor = (struct pack_tensor){
		.byte_offset_in_extent = byte_offset,
		.extent_index = extent_index,
		.element_type = element_type,
		.byte_count = width * first * (second != 0 ? second : 1) *
			      (third != 0 ? third : 1),
		.name_byte_offset = pack->header->name_byte_count,
		.name_byte_count = name_byte_count,
		.dimension_count =
			1u + (second != 0 ? 1u : 0u) + (third != 0 ? 1u : 0u),
		.dimensions = { first, second, third }
	};
	pack->header->name_byte_count += name_byte_count;
	pack->tensors_added++;
	return pack->bytes + pack->extents[extent_index].byte_offset +
	       byte_offset;
}

/* Builds a valid pack of that shape, sealed; the caller frees pack->bytes. The primes are the next ones from the
 * shape's first on; each range of rows begins where the one before it ends, and a table has no row behind its last
 * range. The multipliers are drawn as the reference draws them, the first of each table the largest it can draw, and
 * the token map at random. With rows, every row holds its pattern. What does not fit its room, pack_open refuses. */
static void build_pack(const struct pack_shape *shape, bool with_rows, struct test_pack *pack)
{
    const uint32_t table_count = shape->table_count;
    const uint32_t lookback = shape->lookback;
    const uint32_t head_count = shape->head_count;
    const uint32_t rows_per_token = (lookback - 1) * head_count;
    const uint32_t extent_count = table_count + EXTENTS_BESIDE_TABLES;
    const uint32_t constants_extent = table_count + CONSTANTS_EXTENT_BEHIND_TABLES;
    const int64_t half_bound = INT64_MAX / COMPRESSED_VOCABULARY / MULTIPLIER_HALVES;
    uint64_t row_counts[BUILD_MAXIMUM_TABLES] = {0};
    uint64_t block_counts[BUILD_MAXIMUM_TABLES + EXTENTS_BESIDE_TABLES];
    uint64_t byte_count = (uint64_t)BLOCK + BLOCK;      /* block 0 and the one block of the tables */
    int64_t prime = shape->first_prime - 1;
    char table_name[] = "table0.rows";
    if (table_count == 0 || table_count > BUILD_MAXIMUM_TABLES || lookback > BUILD_MAXIMUM_LOOKBACK ||
        rows_per_token == 0 || rows_per_token > BUILD_MAXIMUM_ROWS) {
        give_up("a shape of the test is one its builder builds");
    }
    for (uint32_t row = 0; row < table_count * rows_per_token; row++) {
        prime = prime_above(prime);
        row_counts[row / rows_per_token] += (uint64_t)prime;
    }
    for (uint32_t extent_index = 0; extent_index < extent_count; extent_index++) {
        const bool holds_rows = extent_index >= 1 && extent_index < constants_extent;   /* a table, or as the first */
        const uint64_t row_count = row_counts[holds_rows && extent_index <= table_count ? extent_index - 1 : 0];
        block_counts[extent_index] = extent_index == constants_extent ? CONSTANTS_BLOCKS
                                     : holds_rows ? (row_count + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK : 1;
        byte_count += block_counts[extent_index] * BLOCK;
    }
    *pack = (struct test_pack){.bytes = aligned_alloc(BLOCK, (size_t)byte_count), .byte_count = byte_count,
                               .shape = *shape, .rows_per_token = rows_per_token};
    if (pack->bytes == NULL) {
        give_up("memory for a pack of the test");
    }
    memset(pack->bytes, 0, (size_t)byte_count);
    pack->header = (void *)pack->bytes;
    pack->extents = (void *)(pack->bytes + BLOCK);
    pack->unit_records = (void *)(pack->extents + extent_count);
    pack->tensors = (void *)(pack->unit_records + table_count + UNITS_BESIDE_TABLES);
    pack->name_bytes = (char *)(pack->tensors + table_count + TENSORS_BESIDE_TABLES);
    *pack->header = (struct pack_header){
	    .magic = PACK_MAGIC,
	    .format_version = PACK_FORMAT_VERSION,
	    .block_byte_count = BLOCK,
	    .model_name = "engram-test-model",
	    .file_byte_count = byte_count,
	    .tables_byte_count = BLOCK,
	    .extent_count = extent_count,
	    .unit_count = table_count + UNITS_BESIDE_TABLES,
	    .tensor_count = table_count + TENSORS_BESIDE_TABLES
    };
    byte_count = (uint64_t)BLOCK + BLOCK;
    for (uint32_t extent_index = 0; extent_index < extent_count; extent_index++) {
        /* behind the first extent of the spare unit, an extent is one ahead of its unit */
        const uint32_t unit_index = extent_index - (extent_index > table_count + SPARE_EXTENT_BEHIND_TABLES ? 1u : 0u);
        const bool is_stage = extent_index == 0 || extent_index == extent_count - 1;
	pack->extents[extent_index] =
		(struct pack_extent){ .byte_offset = byte_count,
				      .byte_count = block_counts[extent_index] *
						    BLOCK,
				      .unit_index = unit_index };
	pack->unit_records[unit_index] = (struct pack_unit_record){
		.unit_kind = is_stage ? STAGE : PLACEMENT_UNIT_HOST_ONLY,
		.layer_index = is_stage ? 0 : PACK_NO_LAYER
	};
	byte_count += block_counts[extent_index] * BLOCK;
    }
    (void)add_tensor(pack, "a.stage", 0, PACK_ELEMENT_INT8, BLOCK, 0, 0);
    pack->multipliers = add_tensor(pack, "engram.multipliers", constants_extent, INT64, table_count, lookback, 0);
    pack->offsets = add_tensor(pack, "engram.offsets", constants_extent, INT64, table_count, rows_per_token, 0);
    pack->pad_id = add_tensor(pack, "engram.pad_id", constants_extent, INT32, 1, 0, 0);
    pack->primes = add_tensor(pack, "engram.primes", constants_extent, INT64, table_count, lookback - 1, head_count);
    pack->row_counts = add_tensor(pack, "engram.row_counts", constants_extent, INT64, table_count, 0, 0);
    pack->table_units = add_tensor(pack, "engram.table_units", constants_extent, INT32, table_count, 0, 0);
    pack->token_map = add_tensor(pack, "engram.token_map", constants_extent, INT32, shape->vocabulary_size, 0, 0);
    for (uint32_t table_index = 0; table_index < table_count; table_index++) {
	    uint64_t *words = add_tensor(
		    pack, table_name, table_index + 1, PACK_ELEMENT_UINT8,
		    (uint32_t)block_counts[table_index + 1], BLOCK, 0);
	    for (uint64_t row_id = 0;
		 with_rows && row_id < row_counts[table_index];
		 row_id++) { /* a row: whole words */
		    uint64_t *row = words +
				    row_id / ROWS_PER_BLOCK * BLOCK_WORDS +
				    row_id % ROWS_PER_BLOCK * ROW_WORDS;
		    write_pattern(table_index, row_id, row);
	    }
        pack->row_counts[table_index] = (int64_t)row_counts[table_index];
        pack->table_units[table_index] = (int32_t)table_index + 1;
        table_name[sizeof("table") - 1] = (char)(table_name[sizeof("table") - 1] + 1);
    }
    (void)add_tensor(pack, "z.stage", extent_count - 1, PACK_ELEMENT_INT8,
		     BLOCK, 0, 0);
    prime = shape->first_prime - 1;
    for (uint32_t row = 0; row < table_count * rows_per_token; row++) {
        prime = prime_above(prime);
        pack->offsets[row] = row % rows_per_token == 0 ? 0 : pack->offsets[row - 1] + pack->primes[row - 1];
        pack->primes[row] = prime;
    }
    for (uint32_t factor_index = 0; factor_index < table_count * lookback; factor_index++) {
        const int64_t half = (int64_t)(next_random() % (uint64_t)half_bound);
        const bool is_first = factor_index % lookback == 0;
        pack->multipliers[factor_index] = (is_first ? half_bound - 1 : half) * MULTIPLIER_HALVES + 1;
    }
    for (uint32_t token_id = 0; token_id < shape->vocabulary_size; token_id++) {
        pack->token_map[token_id] = (int32_t)(next_random() % COMPRESSED_VOCABULARY);
    }
    pack->pad_id[0] = pack->token_map[PAD_TOKEN_ID < shape->vocabulary_size ? PAD_TOKEN_ID : 0];
    seal_pack(pack);
}

/* Opens the pack and then its engram through a memory storage of one copy; a fault a test armed stays. The pack and
 * the reader are closed before this returns: an engram needs neither once it is open. */
MUST_CHECK static result_status open_engram(const struct test_pack *pack,
					    struct engram_record **engram_out)
{
	struct storage_reader *reader = NULL;
	struct pack_record *opened_pack = NULL;
	result_status status =
		STATUS_INJECTED_FAULT; /* what no open of an engram answers */
	test_memory_storage_reset();
	test_memory_storage_set_pack(pack->bytes, pack->byte_count);
	if (test_memory_storage.reader_open(copy_paths, 1, &reader) !=
		    NO_ERROR ||
	    pack_open_entry(&test_memory_storage, reader, &opened_pack) !=
		    NO_ERROR) {
		expect(false,
		       "the pack of a case opens: it is the engram that the case is about");
	} else {
		status = engram_open(&test_memory_storage, reader, opened_pack,
				     engram_out);
	}
	pack_close(opened_pack);
	test_memory_storage.reader_close(reader);
	return status;
}

/* The engram of the pack, or NULL after a failure that is counted. */
static struct engram_record *engram_of(const struct test_pack *pack)
{
	struct engram_record *engram = NULL;
	expect_status(open_engram(pack, &engram), NO_ERROR,
		      "a valid pack of the test gives an engram");
	return engram;
}

/* Opens the engram of the pack and expects this status. A refusal must leave the output alone. */
static void expect_open(const struct test_pack *pack,
			result_status expected_status, const char *what)
{
    static uint64_t untouched_marker = 0;       /* its address stands for an engram that no open gave */
    struct engram_record *const untouched_engram = (void *)&untouched_marker;
    struct engram_record *engram = untouched_engram;
    const result_status status = open_engram(pack, &engram);
    expect_status(status, expected_status, what);
    expect((status == NO_ERROR) != (engram == untouched_engram), "only an open that succeeds writes engram_out");
    if (status == NO_ERROR && engram != untouched_engram) {
	    engram_close(engram);
    }
}

/* torch's % of two integers: the remainder has the sign of the divisor. */
static int64_t torch_remainder(int64_t dividend, int64_t divisor)
{
    const int64_t residue = dividend % divisor;
    return residue != 0 && (residue < 0) != (divisor < 0) ? residue + divisor : residue;
}

/* The second implementation: NgramHashState.forward of the reference for one position, the long way. cache holds
 * the id of the smaller space of every token of the sequence, DEAD for a dead one, as the reference keeps it. */
static void reference_row_ids(const struct test_pack *pack, const int64_t *cache, uint64_t position,
                              uint64_t *row_ids_out)
{
    const size_t lookback = pack->shape.lookback;
    const size_t head_count = pack->shape.head_count;
    int64_t tokens[BUILD_MAXIMUM_LOOKBACK] = {0};
    bool blocked = false;
    for (size_t shift = 0; shift < lookback; shift++) {
        const int64_t source = cache[position >= shift ? position - shift : 0];        /* the clamped gather */
        blocked = blocked || position < shift || source == DEAD;
        tokens[shift] = blocked ? pack->pad_id[0] : source;
    }
    for (size_t layer = 0; layer < pack->shape.table_count; layer++) {
        const int64_t *multipliers = pack->multipliers + layer * lookback;
        uint64_t rolling = (uint64_t)(tokens[0] * multipliers[0]);
        for (size_t size_index = 1; size_index < lookback; size_index++) {
            rolling ^= (uint64_t)(tokens[size_index] * multipliers[size_index]);
            for (size_t head = 0; head < head_count; head++) {
                const size_t column = layer * pack->rows_per_token + (size_index - 1) * head_count + head;
                const int64_t prime = pack->primes[(layer * (lookback - 1) + (size_index - 1)) * head_count + head];
                row_ids_out[column] = (uint64_t)(torch_remainder((int64_t)rolling, prime) + pack->offsets[column]);
            }
        }
    }
}

/* True when row_ids gives, for every position of the sequence, the rows the second implementation gives. */
static bool rows_are_the_reference(const struct engram_record *engram,
				   const struct test_pack *pack,
				   const uint32_t *token_ids,
				   const uint8_t *token_is_dead,
				   uint64_t token_count)
{
    const size_t row_id_count = (size_t)pack->shape.table_count * pack->rows_per_token;
    uint64_t engine_rows[ROW_IDS_CAPACITY];
    uint64_t reference_rows[ROW_IDS_CAPACITY];
    int64_t *cache = malloc((size_t)token_count * sizeof(*cache));
    bool are_equal = cache != NULL;
    for (uint64_t position = 0; cache != NULL && position < token_count; position++) {
        const bool is_dead = token_is_dead != NULL && token_is_dead[position] != 0;
        cache[position] = is_dead ? DEAD : pack->token_map[token_ids[position]];
    }
    for (uint64_t position = 0; are_equal && position < token_count; position++) {
        reference_row_ids(pack, cache, position, reference_rows);
	are_equal = engram_row_ids_entry(engram, token_ids, token_is_dead,
					 position, engine_rows) == NO_ERROR &&
		    memcmp(engine_rows, reference_rows,
			   row_id_count * sizeof(uint64_t)) == 0;
    }
    free(cache);
    return are_equal;
}

/* True when gather gives, for the rows of every token of the sequence in every table, the pattern of each row. */
static bool gathered_rows_are_the_pattern(const struct engram_record *engram,
					  const struct test_pack *pack,
					  const uint32_t *token_ids,
					  uint64_t token_count)
{
    unsigned char rows[BUILD_MAXIMUM_ROWS * ROW_BYTES];
    uint64_t row_ids[ROW_IDS_CAPACITY];
    bool is_pattern = true;
    for (uint64_t position = 0; is_pattern && position < token_count; position++) {
	    is_pattern = engram_row_ids_entry(engram, token_ids, NULL, position,
					      row_ids) == NO_ERROR;
	    for (uint32_t table_index = 0;
		 is_pattern && table_index < pack->shape.table_count;
		 table_index++) {
		    const uint64_t *table_ids =
			    row_ids +
			    (size_t)table_index *
				    pack->rows_per_token; /* this table's rows */
		    is_pattern = engram_gather_entry(engram, table_index,
						     table_ids,
						     pack->rows_per_token,
						     rows) == NO_ERROR;
		    for (size_t row = 0;
			 is_pattern && row < pack->rows_per_token; row++) {
			    is_pattern = row_is_the_pattern(
				    table_index, table_ids[row],
				    rows + row * ROW_BYTES);
		    }
        }
    }
    return is_pattern;
}

/* The description, and the reference's formula on a few thousand tokens drawn at random, dead ones among them. */
static void test_description_and_row_ids(void)
{
	struct engram_record *engram = engram_of(&model_pack);
	const struct engram_description *description =
		engram_description_of(engram);
	expect(engram_description_of(NULL) == NULL,
	       "no engram has no description");
	if (description == NULL) {
		expect(false, "an engram has a description");
		return;
	}
    expect(description->table_count == MODEL_TABLES && description->lookback == MODEL_LOOKBACK &&
               description->head_count == MODEL_HEADS && description->rows_per_token == MODEL_ROWS &&
               description->vocabulary_size == MODEL_VOCABULARY && description->reserved_zero == 0 &&
               description->row_count[0] == MODEL_FIRST_TABLE_ROWS,
           "the description holds the counts of the pack, and the first table the rows its primes add up to");
    for (uint32_t table_index = 0; table_index < ENGRAM_MAXIMUM_TABLES;
	 table_index++) {
	    const bool is_table = table_index < MODEL_TABLES;
	    expect(description->row_count[table_index] ==
				   (is_table ?
					    (uint64_t)model_pack
						    .row_counts[table_index] :
					    0) &&
			   description->table_unit_index[table_index] ==
				   (is_table ? table_index + 1 : 0),
		   "the description holds the row count and the unit of each table, and zero where there is none");
    }
    expect(rows_are_the_reference(engram, &model_pack, sequence, sequence_is_dead, SEQUENCE_TOKENS),
           "row_ids gives the rows of the reference from position 0 on, with a dead token at every distance back");
    expect(rows_are_the_reference(engram, &model_pack, sequence, NULL, SEQUENCE_TOKENS),
           "row_ids gives the rows of the reference without dead tokens");
    expect(torch_remainder(TORCH_DIVIDEND, TORCH_DIVISOR) == TORCH_REMAINDER &&
               torch_remainder(-TORCH_DIVIDEND, TORCH_DIVISOR) == TORCH_DIVISOR - TORCH_REMAINDER,
           "the second implementation takes the remainder of a negative number as torch does");
    engram_close(engram);
    engram_close(NULL);
}

/* One changed value in a pack of the model's shape: of a constant, or the kind of a unit. */
struct value_change {
    const char *what;
    uint32_t tensor_index;                      /* the constant, or TENSOR_UNIT_KIND */
    uint32_t value_index;
    result_status expected_status;
    bool is_added;                              /* the operand is added; otherwise it is the new value */
    int64_t operand;
};

/* Each of the refused ones breaks the rule it is named for and no other: with that rule's check taken out of
 * engram.c its pack would open. Where two are named, the change breaks both and the test cannot tell them apart. */
static const struct value_change value_changes[] = {
	{ "a token map entry is negative", TENSOR_TOKEN_MAP, 0, REFUSED, false,
	  -1 },
	{ "the last token map entry is the lowest", TENSOR_TOKEN_MAP,
	  MODEL_VOCABULARY - 1, REFUSED, false, INT32_MIN },
	{ "pad_id is negative", TENSOR_PAD_ID, 0, REFUSED, false, -1 },
	{ "a token map entry times a multiplier leaves 64 bits",
	  TENSOR_TOKEN_MAP, 1, REFUSED, false, INT32_MAX },
	{ "pad_id times a multiplier leaves 64 bits", TENSOR_PAD_ID, 0, REFUSED,
	  false, INT32_MAX },
	{ "a multiplier is even", TENSOR_MULTIPLIERS, 1, REFUSED, true, 1 },
	{ "the last multiplier is negative and odd", TENSOR_MULTIPLIERS,
	  MODEL_FACTORS - 1, REFUSED, false, -1 },
	{ "a prime is one below the smallest prime", TENSOR_PRIMES, 1, REFUSED,
	  false, ENGRAM_SMALLEST_PRIME - 1 },
	{ "a prime is zero", TENSOR_PRIMES, MODEL_ROWS, REFUSED, false, 0 },
	{ "a prime is negative", TENSOR_PRIMES, MODEL_ROW_IDS - 1, REFUSED,
	  false, -ENGRAM_SMALLEST_PRIME },
	{ "the last range ends a row behind its table, by its offset",
	  TENSOR_OFFSETS, MODEL_ROW_IDS - 1, REFUSED, true, 1 },
	{ "the last range ends a row behind its table, by its prime",
	  TENSOR_PRIMES, MODEL_ROW_IDS - 1, REFUSED, true, 1 },
	{ "the last range of the first table ends a row behind it",
	  TENSOR_OFFSETS, MODEL_ROWS - 1, REFUSED, true, 1 },
	{ "a table is a row shorter than its ranges", TENSOR_ROW_COUNTS, 0,
	  REFUSED, true, -1 },
	{ "a range reaches far behind its table", TENSOR_PRIMES, 1, REFUSED,
	  false, INT64_MAX },
	{ "an offset is negative", TENSOR_OFFSETS, 0, REFUSED, false, -1 },
	{ "a table unit is beyond the units", TENSOR_TABLE_UNITS, 0, REFUSED,
	  false, MODEL_TABLES + UNITS_BESIDE_TABLES },
	{ "a table unit is negative", TENSOR_TABLE_UNITS, MODEL_TABLES - 1,
	  REFUSED, false, -1 },
	{ "a table lies in a unit of two extents", TENSOR_TABLE_UNITS, 0,
	  REFUSED, false, MODEL_UNIT_SPARE },
	{ "a table lies in the unit of another, of other blocks",
	  TENSOR_TABLE_UNITS, 0, REFUSED, false, MODEL_TABLES },
	{ "the unit of a table is a stage unit", TENSOR_UNIT_KIND, 1, REFUSED,
	  false, STAGE },
	{ "a row count needs one block more", TENSOR_ROW_COUNTS, 0, REFUSED,
	  true, MODEL_ROWS_LEFT_IN_LAST_BLOCK + 1 },
	{ "a row count is zero: the blocks, and the ranges", TENSOR_ROW_COUNTS,
	  0, REFUSED, false, 0 },
	{ "a row count is negative", TENSOR_ROW_COUNTS, MODEL_TABLES - 1,
	  REFUSED, false, -1 },
	{ "the unit of the constants is a stage unit", TENSOR_UNIT_KIND,
	  MODEL_UNIT_SPARE + 1, REFUSED, false, STAGE },
	{ "a prime may be the smallest prime", TENSOR_PRIMES, 1, OPENS, false,
	  ENGRAM_SMALLEST_PRIME },
	{ "a multiplier may be one", TENSOR_MULTIPLIERS, 1, OPENS, false, 1 },
	{ "two ranges may share rows", TENSOR_OFFSETS, MODEL_ROW_IDS - 1, OPENS,
	  false, 0 },
	{ "an id of the smaller space may be zero", TENSOR_TOKEN_MAP,
	  MODEL_VOCABULARY - 1, OPENS, false, 0 },
	{ "a row count may fill the last block", TENSOR_ROW_COUNTS, 0, OPENS,
	  true, MODEL_ROWS_LEFT_IN_LAST_BLOCK },
	{ "a unit that holds no table may be of any kind", TENSOR_UNIT_KIND,
	  MODEL_UNIT_SPARE, OPENS, false, STAGE },
};

static void test_values(void)
{
    for (size_t change_index = 0; change_index < COUNT_OF(value_changes); change_index++) {
        const struct value_change *change = &value_changes[change_index];
        const bool is_kind = change->tensor_index == TENSOR_UNIT_KIND;
        struct test_pack pack;
        uint64_t value = 0;
        build_pack(&model_shape, false, &pack);
	const struct pack_tensor *tensor =
		&pack.tensors[is_kind ? TENSOR_TOKEN_MAP : change->tensor_index];
	/* little-endian: any number is the low bytes of a 64-bit one; a unit's kind, first in its record: an id's */
        const size_t width = tensor->element_type == INT64 ? sizeof(int64_t) : sizeof(int32_t);
        unsigned char *const values = is_kind ? (unsigned char *)pack.unit_records
            : pack.bytes + pack.extents[tensor->extent_index].byte_offset + tensor->byte_offset_in_extent;
	unsigned char *const place =
		values +
		change->value_index *
			(is_kind ? sizeof(struct pack_unit_record) : width);
	memcpy(&value, place, width);
        value = change->is_added ? value + (uint64_t)change->operand : (uint64_t)change->operand;
        memcpy(place, &value, width);
        seal_pack(&pack);
        expect_open(&pack, change->expected_status, change->what);
        free(pack.bytes);
    }
}

/* One change to the record of one constant, which leaves the pack a valid pack. */
enum form_change { FORM_MISSING, FORM_OTHER_TYPE, FORM_DIMENSION_MORE, FORM_FLATTENED, FORM_OTHER_EXTENT,
                   FORM_LONGER_FIRST, FORM_LONGER_SECOND, FORM_LONGER_LAST, FORM_CHANGE_COUNT };

static void change_form(struct test_pack *pack, struct pack_tensor *tensor,
			uint32_t form_change)
{
    const uint32_t last = tensor->dimension_count - 1;
    const uint32_t second = last > 0 ? 1 : 0;   /* of the primes: only the look-back is held against it */
    const uint32_t longer = form_change == FORM_LONGER_FIRST ? 0 : form_change == FORM_LONGER_SECOND ? second : last;
    const bool is_wide = tensor->element_type == INT64;
    if (form_change == FORM_MISSING) {          /* the last letter is the next one: the names stay in their order */
        char *const last_letter = &pack->name_bytes[tensor->name_byte_offset + tensor->name_byte_count - 1];
        *last_letter = (char)(*last_letter + 1);
    } else if (form_change == FORM_OTHER_TYPE) {        /* of as many bytes: two narrow elements for each wide one */
	    tensor->element_type = is_wide ? INT32 : PACK_ELEMENT_FLOAT32;
	    tensor->dimensions[last] *=
		    is_wide ? (uint32_t)(sizeof(int64_t) / sizeof(int32_t)) : 1;
    } else if (form_change == FORM_DIMENSION_MORE) {    /* of one element: the bytes stay */
        tensor->dimensions[tensor->dimension_count] = 1;
        tensor->dimension_count++;
    } else if (form_change == FORM_FLATTENED) {
        for (uint32_t dimension_index = 1; dimension_index <= last; dimension_index++) {
            tensor->dimensions[0] *= tensor->dimensions[dimension_index];
            tensor->dimensions[dimension_index] = 0;
        }
        tensor->dimension_count = 1;
    } else if (form_change == FORM_OTHER_EXTENT) {      /* the first of the spare unit: host-only, and large enough */
        tensor->extent_index = pack->shape.table_count + SPARE_EXTENT_BEHIND_TABLES;
    } else {                                    /* the bytes it gains lie in the fill of the extent or in a neighbour */
        tensor->byte_count = tensor->byte_count / tensor->dimensions[longer] * (tensor->dimensions[longer] + 1);
        tensor->dimensions[longer]++;
    }
}

/* Every constant in every changed form: refused, but for the token map, whose absence says that the model has no
 * n-gram tables and whose length is any, and for a tensor of one dimension, which is flat as it is. */
static void test_forms(void)
{
    for (uint32_t tensor_index = TENSOR_MULTIPLIERS; tensor_index <= TENSOR_TOKEN_MAP; tensor_index++) {
        for (uint32_t form_change = 0; form_change < FORM_CHANGE_COUNT; form_change++) {
            const bool is_token_map = tensor_index == TENSOR_TOKEN_MAP;
            const bool is_longer = form_change >= FORM_LONGER_FIRST;    /* the changes of a length are the last ones */
            struct test_pack pack;
            char what[WHAT_CAPACITY];
            build_pack(&model_shape, false, &pack);
	    struct pack_tensor *tensor = &pack.tensors[tensor_index];
	    const bool is_unchanged = form_change == FORM_FLATTENED && tensor->dimension_count == 1;
            (void)snprintf(what, sizeof(what), "%.*s in the changed form %u (enum form_change)",
                           (int)tensor->name_byte_count, pack.name_bytes + tensor->name_byte_offset, form_change);
            change_form(&pack, tensor, form_change);
            seal_pack(&pack);
	    expect_open(&pack,
			is_token_map && form_change == FORM_MISSING ?
				STATUS_UNSUPPORTED :
			is_unchanged || (is_token_map && is_longer) ? OPENS :
								      REFUSED,
			what);
	    free(pack.bytes);
        }
    }
}

/* Counts of zero, which the constants of a valid pack can say, and shapes at and one above each limit. A count of
 * zero is refused as another shape (decided here): with a look-back of no token row_ids has no token to begin from. */
struct counts_case {
    const char *what;
    struct pack_shape shape;                    /* built with it; with no first prime: only said by the constants */
    uint32_t size_count;                        /* then the second dimension of the primes: the look-back less one */
    result_status expected_status;
};

static const struct counts_case counts_cases[] = {
	{ "no table",
	  { 0, MODEL_LOOKBACK, MODEL_HEADS, MODEL_VOCABULARY, 0 },
	  MODEL_LOOKBACK - 1,
	  REFUSED },
	{ "a look-back of one token",
	  { MODEL_TABLES, 1, MODEL_HEADS, MODEL_VOCABULARY, 0 },
	  0,
	  REFUSED },
	{ "a look-back of no token, one less wrapping",
	  { MODEL_TABLES, 0, 0, MODEL_VOCABULARY, 0 },
	  UINT32_MAX,
	  REFUSED },
	{ "no head",
	  { MODEL_TABLES, MODEL_LOOKBACK, 0, MODEL_VOCABULARY, 0 },
	  MODEL_LOOKBACK - 1,
	  REFUSED },
	{ "an empty token map",
	  { MODEL_TABLES, MODEL_LOOKBACK, MODEL_HEADS, 0, 0 },
	  MODEL_LOOKBACK - 1,
	  REFUSED },
	{ "the most tables",
	  { ENGRAM_MAXIMUM_TABLES, MODEL_LOOKBACK, MODEL_HEADS,
	    MODEL_VOCABULARY, 2 },
	  0,
	  OPENS },
	{ "a table above the limit",
	  { BUILD_MAXIMUM_TABLES, MODEL_LOOKBACK, MODEL_HEADS, MODEL_VOCABULARY,
	    2 },
	  0,
	  REFUSED },
	{ "the longest look-back",
	  { MODEL_TABLES, ENGRAM_MAXIMUM_LOOKBACK, 9, MODEL_VOCABULARY, 2 },
	  0,
	  OPENS },
	{ "a look-back above the limit",
	  { MODEL_TABLES, BUILD_MAXIMUM_LOOKBACK, 8, MODEL_VOCABULARY, 2 },
	  0,
	  REFUSED },
	{ "the most rows of a token, in one n-gram size",
	  { MODEL_TABLES, 2, 64, MODEL_VOCABULARY, 2 },
	  0,
	  OPENS },
	{ "a row of a token above the limit, in one n-gram size",
	  { MODEL_TABLES, 2, 65, MODEL_VOCABULARY, 2 },
	  0,
	  REFUSED },
	{ "the smallest engram: a table, a look-back of two, a head, a token",
	  { 1, 2, 1, 1, 2 },
	  0,
	  OPENS },
};

/* Gives a constant that shape and the bytes of it; its element type and its number of dimensions stay. */
static void set_shape(struct pack_tensor *tensor, uint32_t first,
		      uint32_t second, uint32_t third)
{
	const uint32_t shape[PACK_MAXIMUM_DIMENSIONS] = { first, second,
							  third };
	uint64_t byte_count = tensor->element_type == INT64 ? sizeof(int64_t) :
							      sizeof(int32_t);
	for (uint32_t dimension_index = 0;
	     dimension_index < tensor->dimension_count; dimension_index++) {
		byte_count *= shape[dimension_index];
	}
    memcpy(tensor->dimensions, shape, sizeof(shape));
    tensor->byte_count = byte_count;
}

static void test_counts(void)
{
    for (size_t case_index = 0; case_index < COUNT_OF(counts_cases); case_index++) {
        const struct pack_shape *shape = &counts_cases[case_index].shape;
        const uint32_t size_count = counts_cases[case_index].size_count;
        const bool is_built = shape->first_prime != 0;
        struct test_pack pack;
        build_pack(is_built ? shape : &model_shape, false, &pack);
        if (!is_built) {
            set_shape(&pack.tensors[TENSOR_TOKEN_MAP], shape->vocabulary_size, 0, 0);
            set_shape(&pack.tensors[TENSOR_ROW_COUNTS], shape->table_count, 0, 0);
            set_shape(&pack.tensors[TENSOR_TABLE_UNITS], shape->table_count, 0, 0);
            set_shape(&pack.tensors[TENSOR_MULTIPLIERS], shape->table_count, shape->lookback, 0);
            set_shape(&pack.tensors[TENSOR_PRIMES], shape->table_count, size_count, shape->head_count);
            set_shape(&pack.tensors[TENSOR_OFFSETS], shape->table_count, size_count * shape->head_count, 0);
            seal_pack(&pack);
        }
        expect_open(&pack, counts_cases[case_index].expected_status, counts_cases[case_index].what);
        free(pack.bytes);
    }
}

/*
 * One table, a look-back of 3, 2 heads; multipliers 3, 5, 9; primes 5 and 7 for the pairs, 11 and 13 for the
 * triples; so the offsets are 0, 5, 12, 23 and the table has 36 rows. Token i has the id i + 1 and pad_id is 2.
 * The sequence is the tokens 4, 2, 6: the ids 5, 3, 7.
 *   position 0: ids 5, pad, pad; products 15, 10, 18; 15^10 = 5, 5^18 = 23;
 *               rows 5%5 + 0 = 0, 5%7 + 5 = 10, 23%11 + 12 = 13, 23%13 + 23 = 33
 *   position 1: ids 3, 5, pad; products 9, 25, 18; 9^25 = 16, 16^18 = 2; rows 1 + 0, 2 + 5, 2 + 12, 2 + 23
 *   position 2: ids 7, 3, 5; products 21, 15, 45; 21^15 = 26, 26^45 = 55; rows 1 + 0, 5 + 5, 0 + 12, 3 + 23
 *   position 2, token 1 dead: ids 7, pad, pad; products 21, 10, 18; 21^10 = 31, 31^18 = 13; rows 1, 3 + 5, 2 + 12, 23
 *   position 2, token 0 dead: ids 7, 3, pad; products 21, 15, 18; 26, 26^18 = 8; rows 1, 5 + 5, 8 + 12, 8 + 23
 *   position 1, token 0 dead: ids 3, pad, pad; products 9, 10, 18; 9^10 = 3, 3^18 = 17; rows 3, 3 + 5, 6 + 12, 4 + 23
 *   a dead token itself: ids pad, pad, pad; products 6, 10, 18; 6^10 = 12, 12^18 = 30; rows 2, 5 + 5, 8 + 12, 4 + 23
 */
struct hand_case {
    const char *what;
    uint64_t position;
    uint8_t token_is_dead[HAND_TOKENS];
    uint64_t row_ids[HAND_ROWS];
};

static const struct pack_shape hand_shape = {1, HAND_LOOKBACK, HAND_HEADS, 8, 5};       /* 8 tokens; primes from 5 */
static const int64_t hand_multipliers[HAND_LOOKBACK] = {3, 5, 9};
static const uint32_t hand_token_ids[HAND_TOKENS] = {4, 2, 6};
static const struct hand_case hand_cases[] = {
    {"by hand: position 0", 0, {0, 0, 0}, {0, 10, 13, 33}},
    {"by hand: position 1", 1, {0, 0, 0}, {1, 7, 14, 25}},
    {"by hand: position 2", 2, {0, 0, 0}, {1, 10, 12, 26}},
    {"by hand: position 2, the token before it dead", 2, {0, 1, 0}, {1, 8, 14, 23}},
    {"by hand: position 2, the token two before it dead", 2, {1, 0, 0}, {1, 10, 20, 31}},
    {"by hand: position 2, both tokens before it dead: the nearer one decides", 2, {1, 1, 0}, {1, 8, 14, 23}},
    {"by hand: position 1, the token before it dead", 1, {1, 0, 0}, {3, 8, 18, 27}},
    {"by hand: position 2, itself dead", 2, {0, 0, 1}, {2, 10, 20, 27}},
};

static void test_a_case_computed_by_hand(void)
{
    static const uint8_t no_token_is_dead[HAND_TOKENS] = {0};
    struct engram_record *engram = NULL;
    struct test_pack pack;
    build_pack(&hand_shape, false, &pack);
    memcpy(pack.multipliers, hand_multipliers, sizeof(hand_multipliers));
    for (uint32_t token_id = 0; token_id < hand_shape.vocabulary_size; token_id++) {
        pack.token_map[token_id] = (int32_t)token_id + 1;
    }
    pack.pad_id[0] = pack.token_map[1];         /* 2, the id of token 1 */
    seal_pack(&pack);
    engram = engram_of(&pack);
    for (size_t case_index = 0; engram != NULL && case_index < COUNT_OF(hand_cases); case_index++) {
        const struct hand_case *hand = &hand_cases[case_index];
        const bool has_no_dead = memcmp(hand->token_is_dead, no_token_is_dead, sizeof(no_token_is_dead)) == 0;
        uint64_t row_ids[HAND_ROWS] = {0};
	expect(engram_row_ids_entry(engram, hand_token_ids, hand->token_is_dead,
				    hand->position, row_ids) == NO_ERROR &&
		       memcmp(row_ids, hand->row_ids, sizeof(row_ids)) == 0,
	       hand->what);
	memset(row_ids, 0, sizeof(row_ids));
	expect(!has_no_dead ||
		       (engram_row_ids_entry(engram, hand_token_ids, NULL,
					     hand->position,
					     row_ids) == NO_ERROR &&
			memcmp(row_ids, hand->row_ids, sizeof(row_ids)) == 0),
	       "a case without a dead token gives the same rows when no flags are given");
    }
    engram_close(engram);
    free(pack.bytes);
}

/* The largest products open accepts: an id of 7 times a seventh of 2^63 - 1 is 2^63 - 1, every bit below the sign
 * set. The rows are the reference's there too; an id or a multiplier one step further is refused. */
static void test_the_largest_products(void)
{
	struct engram_record *engram = NULL;
	struct test_pack pack;
	build_pack(&model_shape, false, &pack);
	for (uint32_t token_id = 0; token_id < MODEL_VOCABULARY; token_id++) {
		const bool is_largest = token_id % MODEL_HEADS == 0;
		pack.token_map[token_id] =
			is_largest ?
				LARGEST_ID :
				(int32_t)(next_random() % (LARGEST_ID + 1));
	}
    pack.pad_id[0] = LARGEST_ID;
    pack.multipliers[0] = LARGEST_MULTIPLIER;
    pack.multipliers[MODEL_FACTORS - 1] = LARGEST_MULTIPLIER;
    seal_pack(&pack);
    expect((uint64_t)pack.token_map[0] * (uint64_t)pack.multipliers[0] == (uint64_t)INT64_MAX,
           "the largest product of the test is the largest number of 64 signed bits");
    engram = engram_of(&pack);
    expect(engram != NULL && rows_are_the_reference(engram, &pack, sequence, sequence_is_dead, SEQUENCE_TOKENS) &&
               rows_are_the_reference(engram, &pack, sequence, NULL, SEQUENCE_TOKENS),
           "row_ids gives the rows of the reference with the largest products");
    engram_close(engram);
    pack.token_map[MODEL_VOCABULARY - 1] = LARGEST_ID + 1;
    seal_pack(&pack);
    expect_open(&pack, REFUSED, "a token map entry one above the largest whose products fit");
    pack.token_map[MODEL_VOCABULARY - 1] = LARGEST_ID;
    pack.multipliers[0] = LARGEST_MULTIPLIER + MULTIPLIER_HALVES;
    seal_pack(&pack);
    expect_open(&pack, REFUSED, "the next odd multiplier above the largest whose products fit");
    free(pack.bytes);
}

/* Token ids at and above the vocabulary, where row_ids reads them and where it does not, and missing arguments. */
static void test_row_ids_arguments(void)
{
    uint32_t token_ids[MODEL_LOOKBACK + 1] = {MODEL_VOCABULARY, MODEL_VOCABULARY - 1};
    uint8_t token_is_dead[MODEL_LOOKBACK + 1] = {0};
    uint64_t row_ids[MODEL_ROW_IDS];
    struct engram_record *engram = engram_of(&model_pack);
    if (engram == NULL) {
        return;
    }
    expect(engram_row_ids_entry(NULL, token_ids, token_is_dead, MODEL_LOOKBACK,
				row_ids) == INVALID &&
		   engram_row_ids_entry(engram, NULL, token_is_dead,
					MODEL_LOOKBACK, row_ids) == INVALID &&
		   engram_row_ids_entry(engram, token_ids, token_is_dead,
					MODEL_LOOKBACK, NULL) == INVALID,
	   "row_ids refuses a missing engram, sequence and output");
    expect(engram_row_ids_entry(engram, token_ids, NULL, MODEL_LOOKBACK,
				row_ids) == NO_ERROR,
	   "the last token id of the vocabulary is one, and a token behind the look-back is not row_ids' to refuse");
    for (uint32_t distance = 0; distance < MODEL_LOOKBACK; distance++) {
        const uint32_t place = MODEL_LOOKBACK - distance;
        const uint32_t token_id = token_ids[place];
        token_ids[place] = MODEL_VOCABULARY;
	expect(engram_row_ids_entry(engram, token_ids, token_is_dead,
				    MODEL_LOOKBACK, row_ids) == INVALID,
	       "a token id at the vocabulary is refused at every distance back");
	token_ids[place] = UINT32_MAX;          /* decided here: the id of a dead token, or behind one, is not read */
        token_is_dead[place] = 1;
	expect(engram_row_ids_entry(engram, token_ids, token_is_dead,
				    MODEL_LOOKBACK, row_ids) == NO_ERROR,
	       "the id of a dead token is not read, whatever it is");
	token_is_dead[place] = 0;
        token_is_dead[MODEL_LOOKBACK] = distance > 0 ? 1 : 0;
	expect(engram_row_ids_entry(engram, token_ids, token_is_dead,
				    MODEL_LOOKBACK, row_ids) ==
		       (distance > 0 ? NO_ERROR : INVALID),
	       "a token id behind a dead token is not read, and the highest id is refused where it is");
	token_is_dead[MODEL_LOOKBACK] = 0;
        token_ids[place] = token_id;
    }
    engram_close(engram);
}

/* What one thread does with an engram that every thread reads at once. */
struct reader_thread {
    pthread_t thread;
    const struct engram_record *engram;
    bool has_passed;
};

static void *reader_thread_run(void *argument)
{
    struct reader_thread *self = argument;
    self->has_passed = rows_are_the_reference(self->engram, &model_pack, sequence, sequence_is_dead, SEQUENCE_TOKENS) &&
                       gathered_rows_are_the_pattern(self->engram, &model_pack, sequence, SEQUENCE_TOKENS);
    return NULL;
}

/* The bytes gather gives, what it and attach refuse, and eight threads that ask for row ids and gather at once. */
static void test_gather_and_threads(void)
{
    static unsigned char rows[COPY_PAIR * ROW_BYTES + 1];
    const uint64_t row_count = (uint64_t)model_pack.row_counts[1];
    struct reader_thread threads[THREAD_COUNT];
    struct engram_record *engram = engram_of(&model_pack);
    uint64_t row_ids[COPY_PAIR] = {ROWS_PER_BLOCK - 1, ROWS_PER_BLOCK};
    uint32_t started_count = 0;
    if (engram == NULL) {
        return;
    }
    expect(engram_gather_entry(engram, 1, row_ids, 1, rows) == INVALID,
	   "a table that is not attached gives no row");
    expect(engram_attach_entry(NULL, 0, TABLE_ROWS(&model_pack, 0)) ==
			   INVALID &&
		   engram_attach_entry(engram, 0, NULL) == INVALID &&
		   engram_attach_entry(engram, MODEL_TABLES,
				       TABLE_ROWS(&model_pack, 0)) == INVALID,
	   "attach refuses a missing engram, missing rows and a table at the count");
    expect(engram_attach_entry(engram, 1, TABLE_ROWS(&model_pack, 1)) ==
			   NO_ERROR &&
		   engram_gather_entry(engram, 0, row_ids, 1, rows) ==
			   INVALID &&
		   engram_attach_entry(engram, 0, TABLE_ROWS(&model_pack, 0)) ==
			   NO_ERROR,
	   "a table that is attached does not attach another, which is attached after it");
    expect(engram_gather_entry(NULL, 1, row_ids, 1, rows) == INVALID &&
		   engram_gather_entry(engram, 1, NULL, 1, rows) == INVALID &&
		   engram_gather_entry(engram, 1, row_ids, 1, NULL) ==
			   INVALID &&
		   engram_gather_entry(engram, MODEL_TABLES, row_ids, 1,
				       rows) == INVALID,
	   "gather refuses a missing engram, missing row ids, a missing output and a table at the count");
    rows[sizeof(rows) - 1] = 1;
    expect(engram_gather_entry(engram, 1, row_ids, COPY_PAIR, rows) ==
			   NO_ERROR &&
		   row_is_the_pattern(1, row_ids[0], rows) &&
		   row_is_the_pattern(1, row_ids[1], rows + ROW_BYTES) &&
		   rows[sizeof(rows) - 1] == 1,
	   "rows on both sides of a block's end are gathered in the order asked, and nothing behind them is written");
    row_ids[0] = row_count - 1;
    row_ids[1] = 0;
    expect(engram_gather_entry(engram, 1, row_ids, COPY_PAIR, rows) ==
			   NO_ERROR &&
		   row_is_the_pattern(1, row_count - 1, rows) &&
		   row_is_the_pattern(1, 0, rows + ROW_BYTES),
	   "the last row of a table is gathered, and its first after it");
    row_ids[1] = row_count;
    expect(engram_gather_entry(engram, 1, row_ids, COPY_PAIR, rows) == INVALID,
	   "a row id at the count is refused");
    row_ids[0] = UINT64_MAX;
    expect(engram_gather_entry(engram, 1, row_ids, 1, rows) == INVALID,
	   "the highest row id is refused");
    rows[0] = 1;
    expect(engram_gather_entry(engram, 1, row_ids, 0, rows) == NO_ERROR &&
		   rows[0] == 1,
	   "gathering no row writes nothing, whatever the row ids");
    while (started_count < THREAD_COUNT) {
        threads[started_count] = (struct reader_thread){.engram = engram};
        if (pthread_create(&threads[started_count].thread, NULL, reader_thread_run, &threads[started_count]) != 0) {
            break;
        }
        started_count++;
    }
    expect(started_count == THREAD_COUNT, "eight threads are started");
    for (uint32_t thread_index = 0; thread_index < started_count; thread_index++) {
        expect(pthread_join(threads[thread_index].thread, NULL) == 0 && threads[thread_index].has_passed,
               "a thread among eight gets the rows of the reference, and the pattern of every row of every token");
    }
    engram_close(engram);
}

/* What a storage says about its reader, as the test wants it said: at first that it has no copy, and this status. */
static struct storage_description changed_description = {
	.block_alignment_bytes = BLOCK
};
static result_status changed_describe_status = STATUS_OK;

MUST_CHECK static result_status
describe_as_changed(const struct storage_reader *reader,
		    struct storage_description *description_out)
{
    (void)reader;
    *description_out = changed_description;
    return changed_describe_status;
}

/* What two copies deliver of the extent of the constants, what open then says, and how often it read each copy. */
struct copy_scenario {
    const char *what;
    bool read_fails[COPY_PAIR];
    bool bit_is_flipped[COPY_PAIR];
    result_status expected_status;
    uint64_t expected_read_count[COPY_PAIR];
};

static const struct copy_scenario copy_scenarios[] = {
    {"two good copies: the second is not read", {false, false}, {false, false}, OPENS, {1, 0}},
    {"a first copy that cannot be read: the next is", {true, false}, {false, false}, OPENS, {1, 1}},
    {"a first copy with a flipped bit: the next is read", {false, false}, {true, false}, OPENS, {1, 1}},
    {"no copy can be read: the storage's status", {true, true}, {false, false}, UNREAD, {1, 1}},
    {"a flipped bit on every copy", {false, false}, {true, true}, REFUSED, {1, 1}},
    {"an unreadable copy, then one with a flipped bit", {true, false}, {false, true}, REFUSED, {1, 1}},
    {"a copy with a flipped bit, then an unreadable one", {false, true}, {true, false}, UNREAD, {1, 1}},
};

/* Copies that rot or cannot be read and, before the first of them, the arguments and readers open refuses. */
static void test_copies_and_arguments(void)
{
	const struct pack_tensor *token_map =
		&model_pack.tensors[TENSOR_TOKEN_MAP];
	const uint64_t last_byte_of_the_constants =
		model_pack.extents[token_map->extent_index].byte_offset +
		token_map->byte_offset_in_extent + token_map->byte_count - 1;
	struct storage_operations_record storage = test_memory_storage;
	for (size_t scenario_index = 0;
	     scenario_index < COUNT_OF(copy_scenarios); scenario_index++) {
		const struct copy_scenario *scenario =
			&copy_scenarios[scenario_index];
		struct storage_reader *reader = NULL;
		struct pack_record *opened_pack = NULL;
		struct engram_record *engram = NULL;
		uint64_t read_counts[COPY_PAIR] = { 0 };
		test_memory_storage_reset();
		test_memory_storage_set_pack(model_pack.bytes,
					     model_pack.byte_count);
		if (test_memory_storage.reader_open(copy_paths, COPY_PAIR,
						    &reader) != NO_ERROR ||
		    pack_open_entry(&test_memory_storage, reader,
				    &opened_pack) != NO_ERROR) {
			give_up("the memory storage opens a reader of two copies, and the pack of the test opens");
		}
		if (scenario_index == 0) {
			expect(engram_open(NULL, reader, opened_pack,
					   &engram) == INVALID &&
				       engram_open(&storage, NULL, opened_pack,
						   &engram) == INVALID &&
				       engram_open(&storage, reader, NULL,
						   &engram) == INVALID &&
				       engram_open(&storage, reader,
						   opened_pack,
						   NULL) == INVALID,
			       "open refuses a missing storage, reader, pack and output");
			storage.reader_describe = describe_as_changed;
			const result_status without_copy = engram_open(
				&storage, reader, opened_pack, &engram);
			changed_description.copy_count =
				STORAGE_MAXIMUM_COPIES +
				1; /* the reader itself has two */
			expect(without_copy == INVALID &&
				       engram_open(&storage, reader,
						   opened_pack,
						   &engram) == INVALID,
			       "a reader without a copy, and one with more than a storage may have, is an invalid argument");
			changed_describe_status = STATUS_DEVICE_ERROR;
			expect(engram_open(&storage, reader, opened_pack,
					   &engram) == STATUS_DEVICE_ERROR,
			       "a reader that cannot describe itself keeps its own status");
			expect(engram == NULL,
			       "an open that refuses its arguments or its reader writes no engram");
		}
		for (uint32_t copy_index = 0; copy_index < COPY_PAIR;
		     copy_index++) {
			read_counts[copy_index] =
				test_memory_storage_read_count(copy_index);
			if (scenario->read_fails[copy_index]) {
				test_memory_storage_fail_reads(
					copy_index,
					read_counts[copy_index] + 1);
			}
			if (scenario->bit_is_flipped[copy_index]) {
				test_memory_storage_flip_bit(
					copy_index, last_byte_of_the_constants);
			}
		}
		expect_status(engram_open(&test_memory_storage, reader,
					  opened_pack, &engram),
			      scenario->expected_status, scenario->what);
		expect((scenario->expected_status == OPENS) == (engram != NULL),
		       "only a good copy gives an engram");
		expect(test_memory_storage_read_count(0) - read_counts[0] ==
				       scenario->expected_read_count[0] &&
			       test_memory_storage_read_count(1) -
					       read_counts[1] ==
				       scenario->expected_read_count[1],
		       "open reads the extent of the constants once from each copy it needs, and from no other");
		engram_close(engram);
		pack_close(opened_pack);
		test_memory_storage.reader_close(reader);
	}
}

int main(void)
{
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    build_pack(&model_shape, true, &model_pack);
    for (uint64_t position = 0; position < SEQUENCE_TOKENS; position++) {
        sequence[position] = (uint32_t)(next_random() % MODEL_VOCABULARY);
        sequence_is_dead[position] = next_random() % DEAD_ONE_IN == 0 ? 1 : 0;
    }
    sequence_is_dead[0] = 1;                    /* the first dead token of the next positions: at every distance back */
    memset(&sequence_is_dead[1], 0, MODEL_LOOKBACK);
    test_description_and_row_ids();
    test_values();
    test_forms();
    test_counts();
    test_a_case_computed_by_hand();
    test_the_largest_products();
    test_row_ids_arguments();
    test_gather_and_threads();
    test_copies_and_arguments();
#if CHECKING_HOOKS_ENABLED
    expect(fault_arm("engram_memory", 1),
	   "a lack of memory is armed at the one allocation of an open");
    expect_open(&model_pack, STATUS_OUT_OF_MEMORY,
		"an open that lacks memory says so");
    expect(fault_disarm() == 1, "an open ends at the allocation that failed");
    expect(fault_arm("engram_memory", 1 + 1),
	   "a lack of memory is armed for a second allocation");
    expect_open(&model_pack, OPENS, "an open obtains memory once and no more");
    expect(fault_disarm() == 1,
	   "an open that succeeds reached its one allocation");
#endif
    /* a byte of the constants that changed in the pack itself, without a new seal: no copy has the digest */
    model_pack.token_map[0]++;
    expect_open(&model_pack, REFUSED, "an extent of the constants that does not give its digest is refused");
    free(model_pack.bytes);
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("engram_test passed (%d changed values, checking hooks %s)\n",
		 (int)COUNT_OF(value_changes),
		 CHECKING_HOOKS_ENABLED ? "enabled" : "compiled out");
    return EXIT_SUCCESS;
}
