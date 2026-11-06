/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* tp_load_card_test: the pipelined TP load (tp_load.h worker_count, copies) on the CUDA backend's first card: many
 * extents of three tensors each, one extent above a worker slot (the large lane), FULL jobs, two copies of which one
 * delivers a damaged byte. The serial run and the runs on 1, 4 and 16 workers (16 with the choice of copy) must place
 * the same arena bytes (downloaded and compared) and give equal results and receipts. Device memory: an arena of the
 * jobs (~120 MB) beside the context.
 * tp_load_card_test EXTENTS TENSORS BLOCKS: the same checks on one unit of EXTENTS extents of BLOCKS blocks and
 * TENSORS tensors each, with each run's time (a measurement of the TP load's own cost; arena = the pack). */
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack_digest.h"
#include "anemone/tp_load.h"
#include "memory_storage.h"
#include "pack_builder.h"

#define SMALL_EXTENTS 96u
#define SMALL_EXTENT_BLOCKS 64u /* 256 KiB */
#define LARGE_EXTENT_BLOCKS (TP_LOAD_WORKER_SLOT_BYTES / PACK_BLOCK_BYTES + 16u)
#define TENSORS_PER_EXTENT 3u
#define UNITS 3u
#define COPIES 2u
#define DAMAGED_COPY 1u
#define DAMAGED_EXTENT 5u
#define DAMAGED_BYTE 777u
#define CLOSE_NANOSECONDS 10000000000ull
#define SEED 0x5EEDu
#define DECIMAL 10

static const struct test_pack_unit_plan units[UNITS] = {
	{ PLACEMENT_UNIT_STAGE, 0, SMALL_EXTENTS, SMALL_EXTENT_BLOCKS },
	{ PLACEMENT_UNIT_STAGE, 1, 1, LARGE_EXTENT_BLOCKS },
	{ PLACEMENT_UNIT_STAGE, 2, SMALL_EXTENTS, SMALL_EXTENT_BLOCKS }
};
static const char *const copy_paths[COPIES] = { "first", "second" };
static int failures;

static double seconds_now(void)
{
	struct timespec now = { 0 };
	(void)clock_gettime(CLOCK_MONOTONIC, &now);
	return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

/* Zeroes the arena, runs the load on workers threads (copies: the choice of copy or NULL), downloads the arena. */
static bool load_once(struct tp_load_request *request, uint32_t workers,
		      struct tp_load_copies *copies, void *zeros,
		      void *downloaded, uint64_t arena_bytes,
		      struct tp_load_result *result, double *seconds)
{
	const struct backend_operations *const backend = request->backend;
	struct device_card *const card = request->cards[0];
	uint64_t ticket = 0;
	request->worker_count = workers;
	request->copies = copies;
	const bool zeroed =
		backend->upload_begin(card, 0, zeros, arena_bytes, &ticket) ==
			STATUS_OK &&
		backend->upload_wait(card, ticket, CLOSE_NANOSECONDS) ==
			STATUS_OK;
	const double started = seconds_now();
	const bool loaded = zeroed && tp_load_run(request, result) == STATUS_OK;
	*seconds = seconds_now() - started;
	return loaded &&
	       backend->download_begin(card, 0, downloaded, arena_bytes,
				       &ticket) == STATUS_OK &&
	       backend->upload_wait(card, ticket, CLOSE_NANOSECONDS) ==
		       STATUS_OK;
}

int main(int argument_count, char **arguments)
{
	const struct backend_operations *const backend = &backend_cuda_value;
	const bool scaled = argument_count == 4;
	const struct test_pack_unit_plan scaled_unit = {
		PLACEMENT_UNIT_STAGE, 0,
		scaled ? (uint32_t)strtoul(arguments[1], NULL, DECIMAL) : 0,
		scaled ? (uint32_t)strtoul(arguments[3], NULL, DECIMAL) : 0
	};
	const struct test_pack_plan plan = {
		.model_name = "tp-load-card-test",
		.units = scaled ? &scaled_unit : units,
		.unit_count = scaled ? 1u : UNITS,
		.tensors_per_extent =
			scaled ?
				(uint32_t)strtoul(arguments[2], NULL, DECIMAL) :
				TENSORS_PER_EXTENT,
		.content_seed = SEED
	};
	void *bytes = NULL;
	uint64_t byte_count = 0;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct device_card *card = NULL;
	if (test_pack_build(&plan, &bytes, &byte_count) != STATUS_OK)
		return EXIT_FAILURE;
	test_memory_storage_set_pack(bytes, byte_count);
	if (test_memory_storage.reader_open(copy_paths, COPIES, &reader) !=
		    STATUS_OK ||
	    pack_open_entry(&test_memory_storage, reader, &pack) != STATUS_OK)
		return EXIT_FAILURE;
	const struct pack_header *const header = pack_header_of(pack);
	const struct pack_tensor *const tensors = pack_tensors_of(pack);
	uint64_t arena_bytes = 0;
	uint32_t job_count = 0;
	struct tp_load_job *const jobs =
		calloc(header->tensor_count, sizeof(*jobs));
	if (jobs == NULL)
		return EXIT_FAILURE;
	for (uint32_t tensor = 0; tensor < header->tensor_count; tensor++) {
		jobs[job_count++] =
			(struct tp_load_job){ .tensor_index = tensor,
					      .destination_offset = arena_bytes,
					      .mode = TP_LOAD_FULL };
		arena_bytes += (tensors[tensor].byte_count +
				WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES - 1u) /
			       WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES *
			       WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES;
	}
	struct tp_load_description description = {
		.pack = pack,
		.jobs = jobs,
		.job_count = job_count,
		.card_count = 1,
		.card_arena_byte_count = { arena_bytes },
		.maximum_extent_byte_count = TP_LOAD_MAXIMUM_EXTENT_BYTES
	};
	struct tp_load_plan load_plan;
	void *zeros = NULL, *downloaded = NULL, *expected = NULL;
	struct tp_load_request request = { .plan = &load_plan,
					   .storage = &test_memory_storage,
					   .reader = reader,
					   .backend = backend,
					   .cards = &card,
					   .wait_limit_nanoseconds =
						   CLOSE_NANOSECONDS };
	if (tp_load_prepare(&description, &load_plan) != STATUS_OK ||
	    backend->card_open(0, arena_bytes, &card) != STATUS_OK ||
	    backend->staging_allocate(load_plan.source_workspace_byte_count,
				      &request.source_workspace) != STATUS_OK ||
	    backend->staging_allocate(load_plan.output_workspace_byte_count,
				      &request.output_workspace) != STATUS_OK ||
	    backend->staging_allocate(arena_bytes, &zeros) != STATUS_OK ||
	    backend->staging_allocate(arena_bytes, &downloaded) != STATUS_OK ||
	    (expected = malloc(arena_bytes)) == NULL) {
		(void)fprintf(stderr, "FAILED: card, plan or staging\n");
		return EXIT_FAILURE;
	}
	request.source_workspace_byte_count =
		load_plan.source_workspace_byte_count;
	request.output_workspace_byte_count =
		load_plan.output_workspace_byte_count;
	(void)memset(zeros, 0, arena_bytes);
	/* copy 1 delivers one extent with a byte damaged: whoever reads it there falls back to copy 0 */
	test_memory_storage_flip_bit(
		DAMAGED_COPY,
		pack_extents_of(pack)[DAMAGED_EXTENT].byte_offset +
			DAMAGED_BYTE);
	struct tp_load_result serial = { 0 };
	double seconds = 0.0;
	expect(load_once(&request, 0, NULL, zeros, downloaded, arena_bytes,
			 &serial, &seconds),
	       "serial load on the card");
	(void)memcpy(expected, downloaded, arena_bytes);
	expect(serial.tensors_uploaded == job_count &&
		       serial.extents_verified == header->extent_count,
	       "serial load: every job and extent");
	(void)printf("tp_load_card_test: serial: %" PRIu32
		     " jobs, %.3f GB in %.3f s (%.2f GB/s)\n",
		     job_count, (double)serial.uploaded_bytes / 1e9, seconds,
		     (double)serial.uploaded_bytes / 1e9 / seconds);
	static const uint32_t worker_counts[] = { 1u, 4u, 16u, 16u };
	for (uint32_t run = 0;
	     run < sizeof(worker_counts) / sizeof(worker_counts[0]); run++) {
		struct tp_load_copies copies;
		const bool striped = run == 3u;
		expect(tp_load_copies_init(&copies, 2u) == STATUS_OK, "copies");
		struct tp_load_result result = { 0 };
		const bool loaded = load_once(&request, worker_counts[run],
					      striped ? &copies : NULL, zeros,
					      downloaded, arena_bytes, &result,
					      &seconds);
		tp_load_copies_destroy(&copies);
		expect(loaded, "pipelined load on the card");
		expect(loaded && memcmp(expected, downloaded, arena_bytes) == 0,
		       "pipelined load: the serial load's arena bytes");
		expect(result.tensors_uploaded == serial.tensors_uploaded &&
			       result.uploaded_bytes == serial.uploaded_bytes &&
			       result.extents_verified ==
				       serial.extents_verified &&
			       result.card_receipts[0] ==
				       serial.card_receipts[0] &&
			       (striped || result.source_bytes_read ==
						   serial.source_bytes_read),
		       "pipelined load: the serial load's result and receipt");
		(void)printf(
			"tp_load_card_test: %u workers%s: %.3f s (%.2f GB/s), receipt %016" PRIx64
			", copies delivered %" PRIu64 " + %" PRIu64
			" bytes, %" PRIu32 " + %" PRIu32 " failed reads\n",
			worker_counts[run], striped ? " striped" : "", seconds,
			(double)result.uploaded_bytes / 1e9 / seconds,
			result.card_receipts[0], copies.delivered_bytes[0],
			copies.delivered_bytes[1], copies.failed_reads[0],
			copies.failed_reads[1]);
	}
	backend->staging_free(downloaded);
	backend->staging_free(zeros);
	backend->staging_free(request.output_workspace);
	backend->staging_free(request.source_workspace);
	backend->card_close(card, CLOSE_NANOSECONDS);
	free(expected);
	free(jobs);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	free(bytes);
	if (failures == 0)
		(void)puts(
			"tp_load_card_test: serial and pipelined loads (1, 4, 16 workers, striped copies) place the "
			"same bytes");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
