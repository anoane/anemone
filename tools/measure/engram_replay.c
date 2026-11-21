/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Packet CPU probe: production RAM and disk gathers on a sealed engine-derived trace. */
#include <inttypes.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "engine/engram/engram_disk.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/storage/io/storage_batch.h"

#define CASES_MOST 16u
#define INPUT_FIELDS 4
#define STREAM_FANOUT 2u
#define ARGUMENTS 4
#define MAX_TOKENS 1048576u
#define PATH_BYTES 4096u
#define TABLES 2u
#define ROWS 24u
#define TOTAL_ROWS (TABLES * ROWS)
#define CHUNK_TOKENS 1024u
#define DECODE_STEPS 64u
#define STREAMS_MOST 8u
#define VERIFY_CHUNK_BYTES (16ull << 20u)
#define NANOSECONDS_PER_SECOND 1000000000ull
#define TEST_LIMIT_SECONDS 1500u
#define CACHE_SMALL_BYTES (64ull << 20u)
#define CACHE_BYTES (256ull << 20u)
#define HALF_PERCENT 50u
#define HUGE_PAGE_BYTES (2ull << 20u)
#define REFERENCE_MAXIMUM_BYTES (512ull << 20u)
struct trace_case {
	uint32_t id, prompt, count;
	uint64_t *rows;
	unsigned char *reference;
};
static uint64_t now(void)
{
	struct timespec time;
	if (clock_gettime(CLOCK_MONOTONIC, &time) != 0)
		return 0;
	return (uint64_t)time.tv_sec * NANOSECONDS_PER_SECOND +
	       (uint64_t)time.tv_nsec;
}
static bool cases_read(const char *path, struct trace_case *cases,
		       uint32_t *count)
{
	FILE *input = fopen(path, "re");
	if (!input)
		return false;
	if (fscanf(input, "%u", count) != 1 || *count == 0 ||
	    *count > CASES_MOST) {
		(void)fclose(input);
		return false;
	}
	uint64_t reference_bytes = 0;
	for (uint32_t index = 0; index < *count; index++) {
		char rows_path[PATH_BYTES];
		struct trace_case *entry = &cases[index];
		if (fscanf(input, "%u %u %u %4095s", &entry->id, &entry->prompt,
			   &entry->count, rows_path) != INPUT_FIELDS ||
		    entry->prompt == 0 || entry->prompt > entry->count ||
		    entry->count > MAX_TOKENS) {
			(void)fclose(input);
			return false;
		}
		const size_t bytes =
			(size_t)entry->count * TOTAL_ROWS * sizeof(uint64_t);
		const uint64_t payload_bytes =
			(uint64_t)entry->count * TOTAL_ROWS * ENGRAM_ROW_BYTES;
		if (payload_bytes > REFERENCE_MAXIMUM_BYTES - reference_bytes) {
			(void)fclose(input);
			return false;
		}
		reference_bytes += payload_bytes;
		entry->rows = malloc(bytes);
		FILE *rows = fopen(rows_path, "re");
		if (!rows || !entry->rows ||
		    fread(entry->rows, 1, bytes, rows) != bytes ||
		    fgetc(rows) != EOF) {
			if (rows)
				(void)fclose(rows);
			(void)fclose(input);
			return false;
		}
		if (fclose(rows) != 0) {
			(void)fclose(input);
			return false;
		}
	}
	return fclose(input) == 0;
}
static bool reference_matches(const struct engram_description *description,
			      unsigned char *const *ram,
			      const struct trace_case *entry, uint32_t position,
			      uint32_t table, const unsigned char *output)
{
	const size_t first = (size_t)position * TOTAL_ROWS + table * ROWS;
	if (entry->reference != NULL)
		return memcmp(output,
			      entry->reference + first * ENGRAM_ROW_BYTES,
			      ROWS * ENGRAM_ROW_BYTES) == 0;
	const uint64_t *ids = entry->rows + first;
	for (uint32_t index = 0; index < ROWS; index++) {
		if (ids[index] >= description->row_count[table])
			return false;
		const uint64_t offset =
			ids[index] / ENGRAM_ROWS_PER_BLOCK * PACK_BLOCK_BYTES +
			ids[index] % ENGRAM_ROWS_PER_BLOCK * ENGRAM_ROW_BYTES;
		if (memcmp(output + (size_t)index * ENGRAM_ROW_BYTES,
			   ram[table] + offset, ENGRAM_ROW_BYTES) != 0)
			return false;
	}
	return true;
}

struct simultaneous_reader {
	const char *mode;
	struct engram_record *engram;
	const struct trace_case *trace;
	unsigned char *const *ram;
	FILE *output;
	atomic_bool *start;
	uint32_t streams, stream;
	result_status status;
};
static void *simultaneous_read(void *context)
{
	struct simultaneous_reader *self = context;
	const uint64_t wait_started = now();
	while (!atomic_load_explicit(self->start, memory_order_acquire)) {
		if (now() - wait_started > NANOSECONDS_PER_SECOND) {
			self->status = STATUS_TIMED_OUT;
			return NULL;
		}
		(void)sched_yield();
	}
	const struct engram_description *description =
		engram_description_of(self->engram);
	if (description == NULL) {
		self->status = STATUS_INVALID_ARGUMENT;
		return NULL;
	}
	unsigned char bytes[TOTAL_ROWS * ENGRAM_ROW_BYTES];
	for (uint32_t step = 0;
	     step < DECODE_STEPS && self->status == STATUS_OK; step++) {
		const uint64_t *ids =
			self->trace->rows +
			(size_t)(self->trace->prompt - 1u + step) * TOTAL_ROWS;
		const uint64_t started = now();
		for (uint32_t table = 0; table < TABLES; table++)
			engram_prefetch_entry(self->engram, table,
					      ids + table * ROWS, ROWS);
		for (uint32_t table = 0;
		     table < TABLES && self->status == STATUS_OK; table++)
			self->status = engram_gather_entry(
				self->engram, table, ids + table * ROWS, ROWS,
				bytes + (size_t)table * ROWS *
						ENGRAM_ROW_BYTES);
		const uint64_t elapsed = now() - started;
		for (uint32_t table = 0;
		     table < TABLES && self->status == STATUS_OK; table++)
			if (!reference_matches(
				    description, self->ram, self->trace,
				    self->trace->prompt - 1u + step, table,
				    bytes + (size_t)table * ROWS *
						    ENGRAM_ROW_BYTES))
				self->status = STATUS_CORRUPT_PACK;
		if (fprintf(self->output,
			    "{\"mode\":\"%s\",\"kind\":\"simultaneous\",\"streams\":%u,\"stream\":%u,\"step\":%u,\"rows\":%u,\"ns\":%" PRIu64
			    ",\"status\":%d}\n",
			    self->mode, self->streams, self->stream, step,
			    TOTAL_ROWS, elapsed, (int)self->status) < 0)
			self->status = STATUS_INPUT_OUTPUT_ERROR;
	}
	return NULL;
}
static result_status run_simultaneous(const char *mode,
				      struct engram_record *engram,
				      struct trace_case *cases,
				      uint32_t case_count,
				      unsigned char *const *ram, FILE *output)
{
	uint32_t usable[CASES_MOST], count = 0;
	for (uint32_t index = 0; index < case_count; index++)
		if (cases[index].count - cases[index].prompt >= DECODE_STEPS)
			usable[count++] = index;
	for (uint32_t streams = 1; streams <= STREAMS_MOST && streams <= count;
	     streams *= STREAM_FANOUT) {
		pthread_t threads[STREAMS_MOST];
		struct simultaneous_reader readers[STREAMS_MOST];
		atomic_bool start;
		atomic_init(&start, false);
		uint32_t made = 0;
		for (; made < streams; made++) {
			readers[made] = (struct simultaneous_reader){
				.mode = mode,
				.engram = engram,
				.trace = &cases[usable[made]],
				.ram = ram,
				.output = output,
				.start = &start,
				.streams = streams,
				.stream = made,
				.status = STATUS_OK
			};
			if (pthread_create(&threads[made], NULL,
					   simultaneous_read,
					   &readers[made]) != 0)
				break;
		}
		atomic_store_explicit(&start, true, memory_order_release);
		result_status status = made == streams ? STATUS_OK :
							 STATUS_OUT_OF_MEMORY;
		for (uint32_t index = 0; index < made; index++) {
			if (pthread_join(threads[index], NULL) != 0)
				return STATUS_DEVICE_ERROR;
			if (readers[index].status != STATUS_OK)
				status = readers[index].status;
		}
		if (status != STATUS_OK)
			return status;
	}
	return STATUS_OK;
}

static result_status run(const char *mode, struct engram_record *engram,
			 struct trace_case *cases, uint32_t case_count,
			 unsigned char *const *ram, FILE *output)
{
	const struct engram_description *description =
		engram_description_of(engram);
	if (!description)
		return STATUS_INVALID_ARGUMENT;
	unsigned char *bytes =
		malloc((size_t)CHUNK_TOKENS * TABLES * ROWS * ENGRAM_ROW_BYTES);
	if (!bytes)
		return STATUS_OUT_OF_MEMORY;
	result_status status = STATUS_OK;
	for (uint32_t pass = 0; pass <= 1 && status == STATUS_OK; pass++) {
		for (uint32_t index = 0;
		     index < case_count && status == STATUS_OK; index++) {
			const struct trace_case *entry = &cases[index];
			for (uint32_t begin = 0;
			     begin + 1u < entry->prompt && status == STATUS_OK;
			     begin += CHUNK_TOKENS) {
				const uint32_t count =
					entry->prompt - 1u - begin <
							CHUNK_TOKENS ?
						entry->prompt - 1u - begin :
						CHUNK_TOKENS;
				const uint64_t started = now();
				for (uint32_t table = 0; table < TABLES;
				     table++)
					for (uint32_t token = 0; token < count;
					     token++)
						engram_prefetch_entry(
							engram, table,
							entry->rows +
								(size_t)(begin +
									 token) *
									TOTAL_ROWS +
								table * ROWS,
							ROWS);
				for (uint32_t table = 0;
				     table < TABLES && status == STATUS_OK;
				     table++)
					for (uint32_t token = 0;
					     token < count &&
					     status == STATUS_OK;
					     token++)
						status = engram_gather_entry(
							engram, table,
							entry->rows +
								(size_t)(begin +
									 token) *
									TOTAL_ROWS +
								table * ROWS,
							ROWS,
							bytes + ((size_t)table *
									 count +
								 token) *
									ROWS *
									ENGRAM_ROW_BYTES);
				const uint64_t elapsed = now() - started;
				for (uint32_t table = 0;
				     table < TABLES && status == STATUS_OK;
				     table++)
					for (uint32_t token = 0;
					     token < count &&
					     status == STATUS_OK;
					     token++)
						if (!reference_matches(
							    description, ram,
							    entry,
							    begin + token,
							    table,
							    bytes + ((size_t)table *
									     count +
								     token) *
									    ROWS *
									    ENGRAM_ROW_BYTES))
							status =
								STATUS_CORRUPT_PACK;
				if (fprintf(output,
					    "{\"mode\":\"%s\",\"kind\":\"prefill\",\"case\":%u,\"pass\":%u,\"begin\":%u,\"tokens\":%u,\"rows\":%u,\"ns\":%" PRIu64
					    ",\"status\":%d}\n",
					    mode, entry->id, pass, begin, count,
					    count * TOTAL_ROWS, elapsed,
					    (int)status) < 0)
					status = STATUS_INPUT_OUTPUT_ERROR;
				(void)fflush(output);
			}
		}
		uint32_t usable[CASES_MOST], usable_count = 0;
		for (uint32_t index = 0; index < case_count; index++)
			if (cases[index].count - cases[index].prompt >=
			    DECODE_STEPS)
				usable[usable_count++] = index;
		for (uint32_t streams = 1;
		     streams <= STREAMS_MOST && streams <= usable_count &&
		     status == STATUS_OK;
		     streams *= STREAM_FANOUT) {
			for (uint32_t step = 0;
			     step < DECODE_STEPS && status == STATUS_OK;
			     step++) {
				const uint64_t started = now();
				for (uint32_t table = 0; table < TABLES;
				     table++)
					for (uint32_t stream = 0;
					     stream < streams; stream++) {
						const struct trace_case *entry =
							&cases[usable[stream]];
						engram_prefetch_entry(
							engram, table,
							entry->rows +
								(size_t)(entry->prompt -
									 1u +
									 step) *
									TOTAL_ROWS +
								table * ROWS,
							ROWS);
					}
				for (uint32_t table = 0;
				     table < TABLES && status == STATUS_OK;
				     table++)
					for (uint32_t stream = 0;
					     stream < streams &&
					     status == STATUS_OK;
					     stream++) {
						const struct trace_case *entry =
							&cases[usable[stream]];
						status = engram_gather_entry(
							engram, table,
							entry->rows +
								(size_t)(entry->prompt -
									 1u +
									 step) *
									TOTAL_ROWS +
								table * ROWS,
							ROWS,
							bytes + ((size_t)table *
									 streams +
								 stream) *
									ROWS *
									ENGRAM_ROW_BYTES);
					}
				const uint64_t elapsed = now() - started;
				for (uint32_t table = 0;
				     table < TABLES && status == STATUS_OK;
				     table++)
					for (uint32_t stream = 0;
					     stream < streams &&
					     status == STATUS_OK;
					     stream++) {
						const struct trace_case *entry =
							&cases[usable[stream]];
						if (!reference_matches(
							    description, ram,
							    entry,
							    entry->prompt - 1u +
								    step,
							    table,
							    bytes + ((size_t)table *
									     streams +
								     stream) *
									    ROWS *
									    ENGRAM_ROW_BYTES))
							status =
								STATUS_CORRUPT_PACK;
					}
				if (fprintf(output,
					    "{\"mode\":\"%s\",\"kind\":\"decode\",\"pass\":%u,\"streams\":%u,\"step\":%u,\"rows\":%u,\"ns\":%" PRIu64
					    ",\"status\":%d}\n",
					    mode, pass, streams, step,
					    streams * TOTAL_ROWS, elapsed,
					    (int)status) < 0)
					status = STATUS_INPUT_OUTPUT_ERROR;
			}
		}
	}
	if (status == STATUS_OK)
		status = run_simultaneous(mode, engram, cases, case_count, ram,
					  output);
	free(bytes);
	return status;
}
/* Validate against exact traced payload bytes without retaining a second full table copy during disk tests. */
static result_status
reference_prepare(const struct engram_description *description,
		  unsigned char *const *ram, struct trace_case *cases,
		  uint32_t count, uint64_t *bytes_out)
{
	uint64_t total = 0;
	for (uint32_t index = 0; index < count; index++) {
		struct trace_case *entry = &cases[index];
		const uint64_t bytes =
			(uint64_t)entry->count * TOTAL_ROWS * ENGRAM_ROW_BYTES;
		if (bytes > REFERENCE_MAXIMUM_BYTES - total)
			return STATUS_UNSUPPORTED;
		entry->reference = malloc((size_t)bytes);
		if (entry->reference == NULL)
			return STATUS_OUT_OF_MEMORY;
		total += bytes;
		for (uint32_t token = 0; token < entry->count; token++) {
			for (uint32_t table = 0; table < TABLES; table++) {
				for (uint32_t row = 0; row < ROWS; row++) {
					const size_t cell =
						(size_t)token * TOTAL_ROWS +
						table * ROWS + row;
					const uint64_t id = entry->rows[cell];
					if (id >= description->row_count[table])
						return STATUS_INVALID_ARGUMENT;
					const uint64_t offset =
						id / ENGRAM_ROWS_PER_BLOCK *
							PACK_BLOCK_BYTES +
						id % ENGRAM_ROWS_PER_BLOCK *
							ENGRAM_ROW_BYTES;
					memcpy(entry->reference +
						       cell * ENGRAM_ROW_BYTES,
					       ram[table] + offset,
					       ENGRAM_ROW_BYTES);
				}
			}
		}
	}
	*bytes_out = total;
	return STATUS_OK;
}

int main(int argument_count, char **arguments)
{
	(void)alarm(TEST_LIMIT_SECONDS);
	if (argument_count != ARGUMENTS && argument_count != ARGUMENTS + 1)
		return EXIT_FAILURE;
	struct trace_case cases[CASES_MOST] = { 0 };
	uint32_t case_count = 0;
	if (!cases_read(arguments[2], cases, &case_count))
		return EXIT_FAILURE;
	const char *pack_path = arguments[1];
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct engram_record *engram = NULL;
	result_status status =
		storage_direct.reader_open(&pack_path, 1, &reader);
	if (status == STATUS_OK)
		status = pack_open_entry(&storage_direct, reader, &pack);
	if (status == STATUS_OK)
		status = engram_open(&storage_direct, reader, pack, &engram);
	const struct engram_description *description =
		engram_description_of(engram);
	if (status == STATUS_OK &&
	    (!description || description->table_count != TABLES ||
	     description->rows_per_token != ROWS))
		status = STATUS_UNSUPPORTED;
	unsigned char *ram[TABLES] = { 0 };
	uint64_t mapped[TABLES] = { 0 };
	FILE *output = status == STATUS_OK ? fopen(arguments[3], "wex") : NULL;
	if (!output)
		status = STATUS_INPUT_OUTPUT_ERROR;
	for (uint32_t table = 0; table < TABLES && status == STATUS_OK;
	     table++) {
		const struct pack_extent *extent = &pack_extents_of(
			pack)[pack_units_of(
				      pack)[description->table_unit_index[table]]
				      .first_extent_index];
		mapped[table] = (extent->byte_count + HUGE_PAGE_BYTES - 1u) /
				HUGE_PAGE_BYTES * HUGE_PAGE_BYTES;
		void *memory = mmap(NULL, (size_t)mapped[table],
				    PROT_READ | PROT_WRITE,
				    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (memory == MAP_FAILED) {
			status = STATUS_OUT_OF_MEMORY;
			break;
		}
		ram[table] = memory;
		if (madvise(memory, (size_t)mapped[table], MADV_HUGEPAGE) !=
		    0) {
			status = STATUS_UNSUPPORTED;
			break;
		}
		const uint64_t started = now();
		uint64_t digest = 0;
		for (uint64_t done = 0;
		     done < extent->byte_count && status == STATUS_OK;) {
			const uint64_t bytes =
				extent->byte_count - done < VERIFY_CHUNK_BYTES ?
					extent->byte_count - done :
					VERIFY_CHUNK_BYTES;
			status = storage_direct.reader_read(
				reader, 0, extent->byte_offset + done,
				ram[table] + done, bytes);
			if (status == STATUS_OK)
				digest += pack_digest_of_piece(
					ram[table] + done, bytes,
					done / PACK_DIGEST_WORD_BYTES);
			done += bytes;
		}
		if (status == STATUS_OK && digest != extent->digest)
			status = STATUS_CORRUPT_PACK;
		if (status == STATUS_OK)
			status = engram_attach_entry(engram, table, ram[table]);
		(void)fprintf(
			output,
			"{\"kind\":\"RAM_materialization\",\"table\":%u,\"bytes\":%" PRIu64
			",\"ns\":%" PRIu64 ",\"status\":%d}\n",
			table, extent->byte_count, now() - started,
			(int)status);
		(void)fflush(output);
	}
	if (status == STATUS_OK)
		status = run("ram", engram, cases, case_count, ram, output);
	if (status == STATUS_OK) {
		uint64_t reference_bytes = 0;
		const uint64_t started = now();
		status = reference_prepare(description, ram, cases, case_count,
					   &reference_bytes);
		(void)fprintf(
			output,
			"{\"kind\":\"compact_reference\",\"bytes\":%" PRIu64
			",\"ns\":%" PRIu64 ",\"status\":%d}\n",
			reference_bytes, now() - started, (int)status);
		(void)fflush(output);
		if (status == STATUS_OK) {
			engram_close(engram);
			engram = NULL;
			description = NULL;
			for (uint32_t table = 0; table < TABLES; table++) {
				if (munmap(ram[table], (size_t)mapped[table]) !=
				    0)
					status = STATUS_INPUT_OUTPUT_ERROR;
				ram[table] = NULL;
				mapped[table] = 0;
			}
			if (status == STATUS_OK)
				status = engram_open(&storage_direct, reader,
						     pack, &engram);
			description = engram_description_of(engram);
		}
	}
	const uint32_t shares[] = { 0, 0, 0, 0, HALF_PERCENT, 0 };
	const uint32_t batch[] = { 0, 1, 1, 1, 1, 1 };
	const uint64_t caches[] = { 0,		 0,	      CACHE_SMALL_BYTES,
				    CACHE_BYTES, CACHE_BYTES, CACHE_BYTES };
	const char *names[] = { "disk_serial",
				"disk_uring",
				"disk_uring_cache64",
				"disk_uring_cache256",
				"mixed50_uring_cache256",
				"turbo_uring_cache256" };
	struct storage_reader *mapped_reader = NULL, *single_reader = NULL;
	if (status == STATUS_OK)
		status = engram_disk_sources_open(&pack_path, 1, pack, engram,
						  &single_reader);
	if (status == STATUS_OK && argument_count == ARGUMENTS + 1) {
		const char *paths[] = { pack_path, arguments[ARGUMENTS] };
		status = engram_disk_sources_open(
			paths, sizeof(paths) / sizeof(paths[0]), pack, engram,
			&mapped_reader);
	}
	for (uint32_t mode = 0;
	     mode < sizeof(shares) / sizeof(shares[0]) && status == STATUS_OK;
	     mode++) {
		const bool turbo = mode + 1u ==
				   sizeof(shares) / sizeof(shares[0]);
		if (turbo && mapped_reader == NULL)
			continue;
		struct engram_disk *disk = NULL;
		const uint64_t started = now();
		status = engram_disk_open(description, pack, &storage_direct,
					  turbo ? mapped_reader : reader,
					  shares[mode], caches[mode],
					  turbo ? 1u : 0u, batch[mode], &disk);
		(void)fprintf(
			output,
			"{\"kind\":\"disk_preparation\",\"mode\":\"%s\",\"ns\":%" PRIu64
			",\"status\":%d}\n",
			names[mode], now() - started, (int)status);
		(void)fflush(output);
		if (status == STATUS_OK)
			status = engram_attach_disk(engram, disk);
		if (status == STATUS_OK)
			status = run(names[mode], engram, cases, case_count,
				     ram, output);
		engram_disk_close(disk);
	}
	if (output && fclose(output) != 0)
		status = STATUS_INPUT_OUTPUT_ERROR;
	storage_direct.reader_close(mapped_reader);
	storage_direct.reader_close(single_reader);
	engram_close(engram);
	pack_close(pack);
	storage_direct.reader_close(reader);
	for (uint32_t table = 0; table < TABLES; table++)
		if (ram[table])
			(void)munmap(ram[table], (size_t)mapped[table]);
	for (uint32_t index = 0; index < case_count; index++) {
		free(cases[index].rows);
		free(cases[index].reference);
	}
	(void)printf("{\"status\":%d}\n", (int)status);
	return status == STATUS_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
