/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * row_table_probe: what the rows of an n-gram table cost, from host memory and cold from a drive.
 *
 * DeepSeek-V4.1-Flash looks up 24 rows per token in each of two tables of 384,006,168 rows; a row is 256 value
 * bytes and 8 scale bytes. One token therefore needs 48 rows (12,672 B), and a 4,096-token prefill chunk 98,304
 * rows per table. Other engines differ on where the rows live and how they are fetched, and each claims a figure:
 * this probe measures the choices on this machine.
 *
 *   memory   A table of 264-byte rows, 15 to a 4 KiB block so that no row crosses a block, in host memory on
 *            4 KiB or on 2 MiB pages. Measured: 48 random rows copied into one slot (per token: plain, and with
 *            the 48 addresses prefetched first), and 98,304 random rows gathered into a staging area with 1 to
 *            32 threads (per chunk).
 *   drive    48 random 4 KiB blocks of an existing large file read with O_DIRECT (one block per row in the layout
 *            above), and 96 (two per row, as the checkpoint stores values and scales apart): one after another,
 *            by 48 threads at once, and as one io_uring submission. Median and 99th percentile per token.
 *
 * Every gathered row is checked against the row number written into it, and --corrupt changes one stored row
 * first, which the check must report.
 *
 *   row_table_probe memory [--gib N] [--small-pages] [--corrupt]
 *   row_table_probe drive <large file> [--tokens N]
 *
 * Measurement tool: not part of the engine.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/io_uring.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <xmmintrin.h>

#define ROW_BYTES 264
#define BLOCK_BYTES 4096
#define ROWS_PER_BLOCK (BLOCK_BYTES / ROW_BYTES)     /* 15: 136 bytes of every block stay unused */
#define ROWS_PER_TOKEN 48
#define READS_PER_TOKEN_SPLIT (2 * ROWS_PER_TOKEN)   /* values and scales read apart */
#define ROWS_PER_CHUNK 98304                         /* 4,096 tokens times 24 rows, one table */
#define DEFAULT_TABLE_GIB 96
#define TOKEN_ROUNDS 200000
#define CHUNK_ROUNDS 12
#define DEFAULT_DRIVE_TOKENS 2000
#define HUGE_PAGE_BYTES (2ull * 1024ull * 1024ull)
#define BYTES_PER_GIB (1024ull * 1024ull * 1024ull)
#define MICROSECONDS_PER_SECOND 1.0e6
#define PERCENTILE_HIGH 0.99
#define RING_ENTRIES 128
static const int chunk_thread_counts[] = {1, 2, 4, 8, 16, 32};
#define CHUNK_THREAD_CASES (sizeof(chunk_thread_counts) / sizeof(chunk_thread_counts[0]))

static double seconds_now(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1.0e9;
}

static uint64_t mix_bits(uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
}

static int compare_doubles(const void *left, const void *right)
{
    const double difference = *(const double *)left - *(const double *)right;
    return (difference > 0) - (difference < 0);
}

/* ---- rows in host memory ---- */

static inline uint8_t *row_address(uint8_t *table, uint64_t row)
{
    return table + (row / ROWS_PER_BLOCK) * BLOCK_BYTES + (row % ROWS_PER_BLOCK) * ROW_BYTES;
}

static int rows_are_right(const uint8_t *slot, const uint64_t *rows, int row_count)
{
    for (int index = 0; index < row_count; index++) {
        uint64_t stored;
        memcpy(&stored, slot + (size_t)index * ROW_BYTES, sizeof(stored));
        if (stored != rows[index]) {
            return 0;
        }
    }
    return 1;
}

static int measure_memory(uint64_t table_gib, int use_huge_pages, int corrupt)
{
    const uint64_t table_bytes = table_gib * BYTES_PER_GIB;
    const uint64_t row_count = table_bytes / BLOCK_BYTES * ROWS_PER_BLOCK;
    uint8_t *table = mmap(NULL, table_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int all_right = 1;
    if (table == MAP_FAILED || madvise(table, table_bytes, use_huge_pages ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) {
        fprintf(stderr, "cannot map %lu GiB\n", (unsigned long)table_gib);
        return 0;
    }
    const double fill_started = seconds_now();
    #pragma omp parallel for schedule(static)
    for (uint64_t block = 0; block < table_bytes / BLOCK_BYTES; block++) {
        for (uint64_t row_in_block = 0; row_in_block < ROWS_PER_BLOCK; row_in_block++) {
            const uint64_t row = block * ROWS_PER_BLOCK + row_in_block;
            memcpy(table + block * BLOCK_BYTES + row_in_block * ROW_BYTES, &row, sizeof(row));
        }
    }
    printf("a table of %lu GiB (%lu rows) on %s pages, written once in %.1f s\n", (unsigned long)table_gib,
           (unsigned long)row_count, use_huge_pages ? "2 MiB" : "4 KiB", seconds_now() - fill_started);
    if (corrupt) {
        const uint64_t wrong = ~0ull;
        memcpy(row_address(table, mix_bits(1) % row_count), &wrong, sizeof(wrong));
    }

    static uint8_t slot[ROWS_PER_TOKEN * ROW_BYTES];
    uint64_t rows[ROWS_PER_TOKEN];
    for (int prefetch = 0; prefetch <= 1; prefetch++) {
        double gather_seconds = 0.0;
        int tokens_right = 1;
        for (uint64_t token = 0; token < TOKEN_ROUNDS; token++) {
            for (int index = 0; index < ROWS_PER_TOKEN; index++) {
                rows[index] = mix_bits(token * ROWS_PER_TOKEN + (uint64_t)index + (corrupt ? 0 : 1)) % row_count;
            }
            const double started = seconds_now();
            if (prefetch) {
                for (int index = 0; index < ROWS_PER_TOKEN; index++) {
                    const uint8_t *address = row_address(table, rows[index]);
                    _mm_prefetch((const char *)address, _MM_HINT_T0);
                    _mm_prefetch((const char *)address + ROW_BYTES - 1, _MM_HINT_T0);
                }
            }
            for (int index = 0; index < ROWS_PER_TOKEN; index++) {
                memcpy(slot + (size_t)index * ROW_BYTES, row_address(table, rows[index]), ROW_BYTES);
            }
            gather_seconds += seconds_now() - started;
            tokens_right &= rows_are_right(slot, rows, ROWS_PER_TOKEN);
        }
        printf("  one token, 48 rows %s: %6.2f us of host time; rows %s\n",
               prefetch ? "with their addresses prefetched first" : "copied one after another         ",
               gather_seconds / TOKEN_ROUNDS * MICROSECONDS_PER_SECOND, tokens_right ? "as stored" : "WRONG");
        all_right &= tokens_right;
    }

    uint64_t *chunk_rows = malloc(ROWS_PER_CHUNK * sizeof(*chunk_rows));
    uint8_t *staging = aligned_alloc(BLOCK_BYTES, (size_t)ROWS_PER_CHUNK * ROW_BYTES / BLOCK_BYTES * BLOCK_BYTES + BLOCK_BYTES);
    if (chunk_rows == NULL || staging == NULL) {
        return 0;
    }
    for (size_t count_case = 0; count_case < CHUNK_THREAD_CASES; count_case++) {
        const int thread_count = chunk_thread_counts[count_case];
        double gather_seconds = 0.0;
        int chunks_right = 1;
        for (int round = 0; round < CHUNK_ROUNDS; round++) {
            for (uint64_t index = 0; index < ROWS_PER_CHUNK; index++) {
                chunk_rows[index] = mix_bits(((uint64_t)round << 40) + (count_case << 32) + index + 2) % row_count;
            }
            const double started = seconds_now();
            #pragma omp parallel for schedule(static) num_threads(thread_count)
            for (uint64_t index = 0; index < ROWS_PER_CHUNK; index++) {
                memcpy(staging + index * ROW_BYTES, row_address(table, chunk_rows[index]), ROW_BYTES);
            }
            gather_seconds += seconds_now() - started;
            chunks_right &= rows_are_right(staging, chunk_rows, ROWS_PER_CHUNK);
        }
        printf("  one chunk, 98,304 rows on %2d threads: %7.2f ms; rows %s\n", thread_count,
               gather_seconds / CHUNK_ROUNDS * 1000.0, chunks_right ? "as stored" : "WRONG");
        all_right &= chunks_right;
    }
    munmap(table, table_bytes);
    return all_right;
}

/* ---- blocks from a drive ---- */

struct ring {
    int descriptor;
    unsigned *submission_tail, *submission_mask, *submission_array, *completion_head, *completion_tail, *completion_mask;
    struct io_uring_sqe *entries;
    struct io_uring_cqe *completions;
};

static int ring_open(struct ring *ring)
{
    struct io_uring_params parameters;
    memset(&parameters, 0, sizeof(parameters));
    ring->descriptor = (int)syscall(SYS_io_uring_setup, RING_ENTRIES, &parameters);
    if (ring->descriptor < 0) {
        return 0;
    }
    const size_t submission_bytes = parameters.sq_off.array + parameters.sq_entries * sizeof(unsigned);
    const size_t completion_bytes = parameters.cq_off.cqes + parameters.cq_entries * sizeof(struct io_uring_cqe);
    uint8_t *submission = mmap(NULL, submission_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring->descriptor, IORING_OFF_SQ_RING);
    uint8_t *completion = mmap(NULL, completion_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring->descriptor, IORING_OFF_CQ_RING);
    ring->entries = mmap(NULL, parameters.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring->descriptor, IORING_OFF_SQES);
    if (submission == MAP_FAILED || completion == MAP_FAILED || ring->entries == MAP_FAILED) {
        return 0;
    }
    ring->submission_tail = (unsigned *)(submission + parameters.sq_off.tail);
    ring->submission_mask = (unsigned *)(submission + parameters.sq_off.ring_mask);
    ring->submission_array = (unsigned *)(submission + parameters.sq_off.array);
    ring->completion_head = (unsigned *)(completion + parameters.cq_off.head);
    ring->completion_tail = (unsigned *)(completion + parameters.cq_off.tail);
    ring->completion_mask = (unsigned *)(completion + parameters.cq_off.ring_mask);
    ring->completions = (struct io_uring_cqe *)(completion + parameters.cq_off.cqes);
    return 1;
}

/* Submits read_count block reads at once and waits for all of them. Returns 1 when every read was whole. */
static int ring_read_blocks(struct ring *ring, int file, const uint64_t *block_numbers, int read_count, uint8_t *destination)
{
    unsigned tail = *ring->submission_tail;
    for (int index = 0; index < read_count; index++) {
        const unsigned position = tail & *ring->submission_mask;
        struct io_uring_sqe *entry = &ring->entries[position];
        memset(entry, 0, sizeof(*entry));
        entry->opcode = IORING_OP_READ;
        entry->fd = file;
        entry->addr = (uint64_t)(uintptr_t)(destination + (size_t)index * BLOCK_BYTES);
        entry->len = BLOCK_BYTES;
        entry->off = block_numbers[index] * BLOCK_BYTES;
        ring->submission_array[position] = position;
        tail++;
    }
    __atomic_store_n(ring->submission_tail, tail, __ATOMIC_RELEASE);
    if (syscall(SYS_io_uring_enter, ring->descriptor, (unsigned)read_count, (unsigned)read_count, IORING_ENTER_GETEVENTS, NULL, 0) < 0) {
        return 0;
    }
    int whole = 1;
    unsigned head = *ring->completion_head;
    while (head != __atomic_load_n(ring->completion_tail, __ATOMIC_ACQUIRE)) {
        whole &= ring->completions[head & *ring->completion_mask].res == BLOCK_BYTES;
        head++;
    }
    __atomic_store_n(ring->completion_head, head, __ATOMIC_RELEASE);
    return whole;
}

static void report_token_times(const char *label, double *microseconds, int token_count, int all_whole)
{
    qsort(microseconds, (size_t)token_count, sizeof(*microseconds), compare_doubles);
    printf("  %-58s %8.1f us median, %8.1f us at 99%%%s\n", label, microseconds[token_count / 2],
           microseconds[(int)(token_count * PERCENTILE_HIGH)], all_whole ? "" : "  (A READ FAILED OR DIFFERED)");
}

static int measure_drive(const char *path, int token_count)
{
    const int file = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    struct stat status;
    struct ring ring;
    if (file < 0 || fstat(file, &status) != 0 || status.st_size < (off_t)(BYTES_PER_GIB)) {
        fprintf(stderr, "cannot open a file of at least 1 GiB with O_DIRECT\n");
        return 0;
    }
    const uint64_t block_count = (uint64_t)status.st_size / BLOCK_BYTES;
    uint8_t *blocks = aligned_alloc(BLOCK_BYTES, (size_t)READS_PER_TOKEN_SPLIT * BLOCK_BYTES);
    uint8_t *again = aligned_alloc(BLOCK_BYTES, BLOCK_BYTES);
    double *microseconds = malloc((size_t)token_count * sizeof(*microseconds));
    uint64_t block_numbers[READS_PER_TOKEN_SPLIT];
    int all_right = 1;
    if (blocks == NULL || again == NULL || microseconds == NULL || !ring_open(&ring)) {
        fprintf(stderr, "no memory, or io_uring is not available\n");
        return 0;
    }
    printf("random 4 KiB blocks of a file of %.1f GiB, read with O_DIRECT, %d tokens each:\n",
           (double)status.st_size / (double)BYTES_PER_GIB, token_count);
    for (int reads_per_token = ROWS_PER_TOKEN; reads_per_token <= READS_PER_TOKEN_SPLIT; reads_per_token += ROWS_PER_TOKEN) {
        for (int method = 0; method < 3; method++) {
            int all_whole = 1;
            char label[96];
            for (int token = 0; token < token_count; token++) {
                for (int index = 0; index < reads_per_token; index++) {
                    block_numbers[index] = mix_bits(((uint64_t)method << 48) + ((uint64_t)reads_per_token << 40) + (uint64_t)token * READS_PER_TOKEN_SPLIT + (uint64_t)index) % block_count;
                }
                const double started = seconds_now();
                if (method == 0) {
                    for (int index = 0; index < reads_per_token; index++) {
                        all_whole &= pread(file, blocks + (size_t)index * BLOCK_BYTES, BLOCK_BYTES, (off_t)(block_numbers[index] * BLOCK_BYTES)) == BLOCK_BYTES;
                    }
                } else if (method == 1) {
                    #pragma omp parallel for schedule(static, 1) num_threads(ROWS_PER_TOKEN) reduction(&:all_whole)
                    for (int index = 0; index < reads_per_token; index++) {
                        all_whole &= pread(file, blocks + (size_t)index * BLOCK_BYTES, BLOCK_BYTES, (off_t)(block_numbers[index] * BLOCK_BYTES)) == BLOCK_BYTES;
                    }
                } else {
                    all_whole &= ring_read_blocks(&ring, file, block_numbers, reads_per_token, blocks);
                }
                microseconds[token] = (seconds_now() - started) * MICROSECONDS_PER_SECOND;
                /* the last block of every token is read once more the plain way and must hold the same bytes */
                all_whole &= pread(file, again, BLOCK_BYTES, (off_t)(block_numbers[reads_per_token - 1] * BLOCK_BYTES)) == BLOCK_BYTES &&
                             memcmp(again, blocks + (size_t)(reads_per_token - 1) * BLOCK_BYTES, BLOCK_BYTES) == 0;
            }
            snprintf(label, sizeof(label), "%d reads per token, %s:", reads_per_token,
                     method == 0 ? "one after another" : (method == 1 ? "48 threads at once" : "one io_uring submission"));
            report_token_times(label, microseconds, token_count, all_whole);
            all_right &= all_whole;
        }
    }
    return all_right;
}

int main(int argument_count, char **arguments)
{
    uint64_t table_gib = DEFAULT_TABLE_GIB;
    int use_huge_pages = 1, corrupt = 0, token_count = DEFAULT_DRIVE_TOKENS;
    const char *path = NULL;
    if (argument_count < 2 || (strcmp(arguments[1], "memory") != 0 && strcmp(arguments[1], "drive") != 0)) {
        fprintf(stderr, "usage: %s memory [--gib N] [--small-pages] [--corrupt] | drive <large file> [--tokens N]\n", arguments[0]);
        return EXIT_FAILURE;
    }
    for (int argument_index = 2; argument_index < argument_count; argument_index++) {
        const int has_value = argument_index + 1 < argument_count;
        if (strcmp(arguments[argument_index], "--gib") == 0 && has_value) {
            table_gib = (uint64_t)atoi(arguments[++argument_index]);
        } else if (strcmp(arguments[argument_index], "--tokens") == 0 && has_value) {
            token_count = atoi(arguments[++argument_index]);
        } else if (strcmp(arguments[argument_index], "--small-pages") == 0) {
            use_huge_pages = 0;
        } else if (strcmp(arguments[argument_index], "--corrupt") == 0) {
            corrupt = 1;
        } else if (path == NULL && arguments[argument_index][0] != '-') {
            path = arguments[argument_index];
        } else {
            fprintf(stderr, "unknown argument %s\n", arguments[argument_index]);
            return EXIT_FAILURE;
        }
    }
    if (strcmp(arguments[1], "memory") == 0) {
        if (table_gib < 1) {
            return EXIT_FAILURE;
        }
        const int all_right = measure_memory(table_gib, use_huge_pages, corrupt);
        printf("%s\n", corrupt ? (all_right ? "FAILED: the changed row was not noticed" : "the changed row was noticed")
                               : (all_right ? "every row was as stored" : "FAILED: a row was wrong"));
        return (corrupt ? !all_right : all_right) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (path == NULL || token_count < 100) {
        fprintf(stderr, "drive needs a file and at least 100 tokens\n");
        return EXIT_FAILURE;
    }
    return measure_drive(path, token_count) ? EXIT_SUCCESS : EXIT_FAILURE;
}
