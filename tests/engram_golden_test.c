/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_golden_test: the row ids engram_row_ids_entry gives for a stream of tokens against the ones the model's
 * own reference code gave for it, on a real pack.
 *
 *     engram_golden_test <golden file> <pack copy> [<pack copy> ...]
 *
 * The golden file is written by tools/golden/make_engram_golden.py (its text states the format); the pack is one
 * that holds the model's n-gram tables. Every row id of every token must be the reference's. It then times the
 * call: nanoseconds for the row ids of one token, as the median and the 99th percentile over chunks of
 * TIMED_CHUNK_TOKENS tokens. Without arguments, as `make check` runs every test, there is no pack to read and it
 * says that nothing was compared. A batch runs it with arguments in every build, and once with a golden file in
 * which one row id was changed, which it must report.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/engram.h"
#include "anemone/pack.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"

#define GOLDEN_MAGIC "ENGRAMID"
#define GOLDEN_MAGIC_BYTES ((size_t)8)
#define GOLDEN_FORMAT_VERSION 1u
#define GOLDEN_HEADER_BYTES ((size_t)32)
#define GOLDEN_TOKEN_COUNT_LIMIT ((uint64_t)16777216)    /* what the writer allows: the whole file is held in memory */
#define TOKEN_ID_BYTES ((size_t)4)
#define ROW_ID_BYTES ((size_t)8)
#define ARGUMENTS_BEFORE_THE_COPIES 2                     /* the program's name and the golden file */
#define TIMED_CHUNK_TOKENS ((uint64_t)1024)
#define MOST_TIMED_CHUNKS ((size_t)16384)
#define TIMED_PASSES 3                                    /* the first warms the caches; recorded up to the limit */
#define PERCENT 100u
#define MEDIAN_PERCENTILE 50u
#define HIGH_PERCENTILE 99u
#define NANOSECONDS_PER_SECOND 1e9
#define MISMATCHES_PRINTED 4

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };

struct golden_header {                                    /* little-endian, 32 bytes, nothing between its members */
    char magic[GOLDEN_MAGIC_BYTES];
    uint32_t format_version;
    uint32_t table_count;
    uint32_t rows_per_token;
    uint32_t vocabulary_size;
    uint64_t token_count;
};
_Static_assert(sizeof(struct golden_header) == GOLDEN_HEADER_BYTES, "the header has no padding");

static double monotonic_seconds(void)
{
    struct timespec now = {0};
    const int clock_status = clock_gettime(CLOCK_MONOTONIC, &now);
    return clock_status == 0 ? (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND : 0.0;
}

static int compare_doubles(const void *first, const void *second)
{
    const double first_value = *(const double *)first;
    const double second_value = *(const double *)second;
    return (first_value > second_value) - (first_value < second_value);
}

/* The whole file in memory (aligned for any type, as allocated), or NULL with one line that says why. */
static void *read_file(const char *path, size_t *byte_count_out)
{
    unsigned char *bytes = NULL;
    long end_position = -1;
    FILE *const file = fopen(path, "rb");
    if (file == NULL) {
        (void)fprintf(stderr, "FAILED: the golden file cannot be opened\n");
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) == 0) {
        end_position = ftell(file);
    }
    if (end_position >= 0 && fseek(file, 0, SEEK_SET) == 0) {
        bytes = malloc((size_t)end_position + 1);
    }
    if (bytes != NULL && fread(bytes, 1, (size_t)end_position, file) != (size_t)end_position) {
        free(bytes);
        bytes = NULL;
    }
    (void)fclose(file);
    if (bytes == NULL) {
        (void)fprintf(stderr, "FAILED: the golden file cannot be read\n");
        return NULL;
    }
    *byte_count_out = (size_t)end_position;
    return bytes;
}

/* Every row id of every token against the golden ones. The count of tokens with another row id. */
static uint64_t tokens_that_differ(const struct engram_record *engram,
				   const struct golden_header *header,
				   const uint32_t *token_ids,
				   const uint8_t *dead_flags,
				   const unsigned char *golden_row_ids)
{
    const size_t ids_per_token = (size_t)header->table_count * header->rows_per_token;
    uint64_t row_ids[ENGRAM_MAXIMUM_TABLES * ENGRAM_MAXIMUM_ROWS_PER_TOKEN] = {
	    0
    };
    uint64_t differing = 0;
    for (uint64_t position = 0; position < header->token_count; position++) {
        const unsigned char *const golden = golden_row_ids + position * ids_per_token * ROW_ID_BYTES;
        memset(row_ids, UINT8_MAX, sizeof(row_ids));      /* an id the call leaves alone is not the token's before */
	const result_status status = engram_row_ids_entry(
		engram, token_ids, dead_flags, position, row_ids);
	if (status == STATUS_OK &&
	    memcmp(row_ids, golden, ids_per_token * ROW_ID_BYTES) == 0) {
		continue;
	}
	if (differing < MISMATCHES_PRINTED) {             /* the first row id of the token that is another one */
            uint64_t golden_id = 0;
            size_t id_index = 0;
            for (; id_index < ids_per_token; id_index++) {
                memcpy(&golden_id, golden + id_index * ROW_ID_BYTES, sizeof(golden_id));
                if (golden_id != row_ids[id_index]) {
                    break;
                }
            }
	    (void)fprintf(stderr,
			  "FAILED: token %" PRIu64 " (id %" PRIu32
			  ", %s): %s, row id %zu of its %zu is %" PRIu64
			  ", the reference's %" PRIu64 "\n",
			  position, token_ids[position],
			  dead_flags[position] != 0 ? "dead" : "live",
			  status_text(status), id_index, ids_per_token,
			  id_index < ids_per_token ? row_ids[id_index] : 0,
			  golden_id);
	}
        differing++;
    }
    return differing;
}

/* Times the call over the stream: nanoseconds per token of every chunk, sorted. The count of chunks timed. */
static size_t timed_chunks(const struct engram_record *engram,
			   const struct golden_header *header,
			   const uint32_t *token_ids, const uint8_t *dead_flags,
			   double *nanoseconds_per_token,
			   uint64_t *checksum_out)
{
	uint64_t row_ids[ENGRAM_MAXIMUM_TABLES *
			 ENGRAM_MAXIMUM_ROWS_PER_TOKEN] = { 0 };
	size_t chunk_count = 0;
	uint64_t checksum = 0;
	for (int pass = 0; pass < TIMED_PASSES; pass++) {
		for (uint64_t first = 0;
		     first + TIMED_CHUNK_TOKENS <= header->token_count &&
		     chunk_count < MOST_TIMED_CHUNKS;
		     first += TIMED_CHUNK_TOKENS) {
			const double started = monotonic_seconds();
			for (uint64_t position = first;
			     position < first + TIMED_CHUNK_TOKENS;
			     position++) {
				if (engram_row_ids_entry(
					    engram, token_ids, dead_flags,
					    position, row_ids) == STATUS_OK) {
					checksum += row_ids
						[0]; /* the result is used: the call is not removed */
				}
			}
			nanoseconds_per_token[chunk_count] =
				(monotonic_seconds() - started) *
				NANOSECONDS_PER_SECOND /
				(double)TIMED_CHUNK_TOKENS;
			chunk_count++;
		}
    }
    qsort(nanoseconds_per_token, chunk_count, sizeof(*nanoseconds_per_token), compare_doubles);
    *checksum_out = checksum;
    return chunk_count;
}

int main(int argument_count, char **arguments)
{
    static double nanoseconds_per_token[MOST_TIMED_CHUNKS];
    struct storage_reader *reader = NULL;
    struct pack_record *pack = NULL;
    struct engram_record *engram = NULL;
    struct golden_header header = {0};
    void *golden_memory = NULL;
    const unsigned char *golden = NULL;                   /* the golden file as bytes */
    const uint32_t *golden_words = NULL;                  /* and as the 32-bit words its header and token ids are */
    const uint32_t *token_ids = NULL;
    size_t golden_byte_count = 0;
    uint64_t differing = 1;                               /* until every token was compared */
    if (argument_count == 1) {
        (void)printf("engram_golden_test passed (nothing compared: it needs a golden file and a pack)\n");
        return TEST_PASSED;
    }
    if (argument_count <= ARGUMENTS_BEFORE_THE_COPIES ||
	argument_count - ARGUMENTS_BEFORE_THE_COPIES > STORAGE_MAXIMUM_COPIES) {
	    (void)fprintf(
		    stderr,
		    "usage: engram_golden_test <golden file> <pack copy> [<pack copy> ...]\n");
	    return TEST_WRONG_USAGE;
    }
    const char *const *const copy_paths = (const char *const *)&arguments[ARGUMENTS_BEFORE_THE_COPIES];
    const uint32_t copy_count = (uint32_t)(argument_count - ARGUMENTS_BEFORE_THE_COPIES);
    result_status status =
	    storage_direct.reader_open(copy_paths, copy_count, &reader);
    status = status == STATUS_OK ?
		     pack_open_entry(&storage_direct, reader, &pack) :
		     status;
    status = status == STATUS_OK ?
		     engram_open(&storage_direct, reader, pack, &engram) :
		     status;
    if (status != STATUS_OK) {
	    (void)fprintf(stderr,
			  "FAILED: the pack's n-gram tables do not open: %s\n",
			  status_text(status));
    } else {
	    golden_memory = read_file(arguments[1], &golden_byte_count);
	    golden = golden_memory;
	    golden_words = golden_memory;
    }
    if (golden != NULL) {
	    const struct engram_description *const description =
		    engram_description_of(engram);
	    const size_t ids_per_token = (size_t)description->table_count *
					 description->rows_per_token;
	    const size_t bytes_per_token = TOKEN_ID_BYTES + sizeof(uint8_t) +
					   ids_per_token * ROW_ID_BYTES;
	    if (golden_byte_count >= sizeof(header)) {
		    memcpy(&header, golden, sizeof(header));
	    }
        /* a count the file states is used only once it is within the writer's limit: 0 stands for one above it */
        const size_t token_count = header.token_count <= GOLDEN_TOKEN_COUNT_LIMIT ? (size_t)header.token_count : 0;
        if (memcmp(header.magic, GOLDEN_MAGIC, GOLDEN_MAGIC_BYTES) != 0 ||
            header.format_version != GOLDEN_FORMAT_VERSION || header.table_count != description->table_count ||
            header.rows_per_token != description->rows_per_token ||
            header.vocabulary_size != description->vocabulary_size || token_count == 0 ||
            golden_byte_count != sizeof(header) + token_count * bytes_per_token) {
            (void)fprintf(stderr, "FAILED: the golden file is not one of this pack: %" PRIu32 " tables, %" PRIu32
                                  " rows per token, a vocabulary of %" PRIu32 ", %" PRIu64 " tokens in %zu bytes; "
                                  "the pack has %" PRIu32 ", %" PRIu32 " and %" PRIu32 "\n",
                          header.table_count, header.rows_per_token, header.vocabulary_size, header.token_count,
                          golden_byte_count, description->table_count, description->rows_per_token,
                          description->vocabulary_size);
        } else {
            token_ids = golden_words + GOLDEN_HEADER_BYTES / TOKEN_ID_BYTES;
        }
    }
    if (token_ids != NULL) {
        const size_t token_count = (size_t)header.token_count;
        const uint8_t *const dead_flags = golden + sizeof(header) + token_count * TOKEN_ID_BYTES;
        uint64_t checksum = 0;
        differing = tokens_that_differ(engram, &header, token_ids, dead_flags, dead_flags + token_count);
        const size_t chunk_count = timed_chunks(engram, &header, token_ids, dead_flags, nanoseconds_per_token,
                                                &checksum);
        (void)printf("%" PRIu64 " tokens, %" PRIu32 " tables, %" PRIu32 " rows per token: %" PRIu64
                     " tokens have another row id than the reference's\n", header.token_count, header.table_count,
                     header.rows_per_token, differing);
        if (chunk_count != 0) {
            (void)printf("row ids of one token: median %.1f ns, 99th percentile %.1f ns, fastest %.1f ns (%zu chunks "
                         "of %" PRIu64 " tokens, checksum %" PRIu64 ")\n",
                         nanoseconds_per_token[chunk_count * MEDIAN_PERCENTILE / PERCENT],
                         nanoseconds_per_token[chunk_count * HIGH_PERCENTILE / PERCENT], nanoseconds_per_token[0],
                         chunk_count, TIMED_CHUNK_TOKENS, checksum);
        }
    }
    free(golden_memory);
    engram_close(engram);
    pack_close(pack);
    storage_direct.reader_close(reader);
    if (differing != 0) {
        (void)fprintf(stderr, "engram_golden_test FAILED\n");
        return TEST_FAILED;
    }
    (void)printf("engram_golden_test passed\n");
    return TEST_PASSED;
}
