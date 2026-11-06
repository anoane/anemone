/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Selected extents, paired original-format shards, byte padding, fallback and non-retired staging lifetimes. */
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack_digest.h"
#include "anemone/tp_load.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "pack_builder.h"

#define TEST_CARDS 2u
#define TEST_COPIES 2u
#define TEST_UNITS 3u
#define TEST_TENSORS_PER_EXTENT 3u
#define TEST_EXTENT_BLOCKS 2u
#define TEST_EXTENT_BYTES ((uint64_t)TEST_EXTENT_BLOCKS * PACK_BLOCK_BYTES)
#define TEST_HOST_TENSOR_INDEX ((TEST_UNITS - 1u) * TEST_TENSORS_PER_EXTENT)
#define TEST_ARENA_BYTES 16384u
#define TEST_SECOND_MATRIX_OFFSET 1024u
#define TEST_PLAIN_OFFSET 4096u
#define TEST_SECOND_EXTENT_OFFSET 8192u
#define TEST_PLAIN_BYTES 12u
#define TEST_PLAIN_PADDED_BYTES 16u
#define TEST_PLAIN_ROWS 2u   /* a plain bfloat16 matrix [2][3]: rank r loads row r (tp_weights.h's row shards) */
#define TEST_PLAIN_INPUTS 3u
#define TEST_MATRIX_DIMENSIONS 2u
#define TEST_MATRIX_OUTPUTS 32u
#define TEST_MATRIX_INPUTS 64u
#define TEST_SHARD_OUTPUTS 16u
#define TEST_SHARD_INPUTS 32u
#define TEST_MXFP4_BITS 4u
#define TEST_FP8_BITS 8u
#define TEST_MXFP4_BYTES 1088u
#define TEST_FP8_BYTES 2050u
#define TEST_CLOSE_NANOSECONDS 1000000000ull
#define TEST_SEED 0x217A43B69F05ull
#define TEST_CORRUPT_BYTE 5u
#define TEST_EXPECTED_POISON UINT64_MAX
#define TEST_HOST_EXTENT (TEST_UNITS - 1u)
#define TEST_HOST_STREAM_BLOCKS \
	((uint32_t)(TP_HOST_LOAD_CHUNK_BYTES / PACK_BLOCK_BYTES) + 1u)
#define TEST_HOST_STREAM_BYTES (TP_HOST_LOAD_CHUNK_BYTES + PACK_BLOCK_BYTES)
#define TEST_HOST_STREAM_CHUNKS 2u
#define TEST_HOST_STREAM_EXTENTS 2u
#define TEST_WORKERS 3u
#define TEST_RETIRED_MOST (2u * TEST_WORKERS)

enum host_fault {
	HOST_SOUND,
	HOST_READ_FALLBACK,
	HOST_DIGEST_FALLBACK,
	HOST_ALL_READS_FAIL,
	HOST_ALL_DIGESTS_FAIL
};

enum test_job {
	TEST_MXFP4_FIRST,
	TEST_MXFP4_SECOND,
	TEST_FP8_FIRST,
	TEST_FP8_SECOND,
	TEST_PLAIN_FIRST,
	TEST_PLAIN_SECOND,
	TEST_LATER_TENSOR,
	TEST_JOB_COUNT
};
enum test_tensor {
	TEST_MXFP4_TENSOR,
	TEST_FP8_TENSOR,
	TEST_PLAIN_TENSOR,
	TEST_LATER_TENSOR_INDEX
};

static const struct test_pack_unit_plan unit_plans[TEST_UNITS] = {
	{ PLACEMENT_UNIT_STAGE, 0, 1, TEST_EXTENT_BLOCKS },
	{ PLACEMENT_UNIT_STAGE, 1, 1, TEST_EXTENT_BLOCKS },
	{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, TEST_EXTENT_BLOCKS }
};
static const char *const copy_paths[TEST_COPIES] = { "first", "second" };
static int failures;
/* The suite runs twice: the serial run (0) and the pipelined one on TEST_WORKERS threads, whose slots go through
 * staging_retire: retire_later keeps them until the fixture's cards are closed. */
static uint32_t test_workers;
static struct backend_operations retiring_backend;
static pthread_mutex_t retired_mutex =
	PTHREAD_MUTEX_INITIALIZER; /* workers retire at once */
static void *retired_blocks[TEST_RETIRED_MOST];
static uint32_t retired_count;

static void retire_later(void *memory)
{
	(void)pthread_mutex_lock(&retired_mutex);
	if (memory != NULL && retired_count < TEST_RETIRED_MOST)
		retired_blocks[retired_count++] = memory;
	(void)pthread_mutex_unlock(&retired_mutex);
}

struct fixture {
	void *bytes;
	uint64_t byte_count;
	struct storage_reader *reader;
	struct pack_record *pack;
	struct device_card *cards[TEST_CARDS];
	struct tp_load_job jobs[TEST_JOB_COUNT];
	struct tp_load_plan plan;
	struct tp_load_request request;
};

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

static bool checked(result_status status, const char *message)
{
	expect(status == STATUS_OK, message);
	return status == STATUS_OK;
}

static void fixture_close(struct fixture *fixture)
{
	for (uint32_t card = 0; card < TEST_CARDS; card++)
		test_host_memory_backend.card_close(fixture->cards[card],
						    TEST_CLOSE_NANOSECONDS);
	for (; retired_count != 0; retired_count--)
		test_host_memory_backend.staging_free(
			retired_blocks[retired_count - 1u]);
	test_host_memory_backend.staging_free(
		fixture->request.output_workspace);
	test_host_memory_backend.staging_free(
		fixture->request.source_workspace);
	pack_close(fixture->pack);
	test_memory_storage.reader_close(fixture->reader);
	free(fixture->bytes);
}

static struct pack_tensor *mutable_tensors(void *bytes)
{
	const struct pack_header *const header = bytes;
	unsigned char *const tables = (unsigned char *)bytes + PACK_BLOCK_BYTES;
	return (void *)(tables +
			(uint64_t)header->extent_count *
				sizeof(struct pack_extent) +
			(uint64_t)header->unit_count *
				sizeof(struct pack_unit_record));
}

static bool jobs_prepare(struct fixture *fixture)
{
	const struct pack_tensor *const tensors =
		pack_tensors_of(fixture->pack);
	for (uint32_t rank = 0; rank < TEST_CARDS; rank++) {
		const struct tensor_shard_description mx = {
			WEIGHT_FORMAT_E2M1_SCALED,
			{ TEST_MATRIX_OUTPUTS, TEST_MATRIX_INPUTS,
			  TEST_MXFP4_BITS, 0 },
			rank * TEST_SHARD_OUTPUTS,
			TEST_SHARD_OUTPUTS,
			0,
			TEST_MATRIX_INPUTS
		};
		const struct tensor_shard_description fp = {
			WEIGHT_FORMAT_E4M3_SCALED,
			{ TEST_MATRIX_OUTPUTS, TEST_MATRIX_INPUTS,
			  TEST_FP8_BITS, 0 },
			0,
			TEST_MATRIX_OUTPUTS,
			rank * TEST_SHARD_INPUTS,
			TEST_SHARD_INPUTS
		};
		fixture->jobs[TEST_MXFP4_FIRST + rank] =
			(struct tp_load_job){ .tensor_index = TEST_MXFP4_TENSOR,
					      .card = rank,
					      .mode = TP_LOAD_SHARD };
		fixture->jobs[TEST_FP8_FIRST + rank] =
			(struct tp_load_job){ .tensor_index = TEST_FP8_TENSOR,
					      .card = rank,
					      .destination_offset =
						      TEST_SECOND_MATRIX_OFFSET,
					      .mode = TP_LOAD_SHARD };
		fixture->jobs[TEST_PLAIN_FIRST + rank] =
			(struct tp_load_job){ .tensor_index = TEST_PLAIN_TENSOR,
					      .card = rank,
					      .destination_offset =
						      TEST_PLAIN_OFFSET,
					      .mode = TP_LOAD_SHARD };
		const struct tensor_shard_description plain = {
			.format = WEIGHT_FORMAT_NONE,
			.source_shape = { TEST_PLAIN_ROWS, TEST_PLAIN_INPUTS,
					  TENSOR_SHARD_PLAIN_BITS, 0 },
			.first_output = rank,
			.output_count = 1,
			.first_input = 0,
			.input_count = TEST_PLAIN_INPUTS
		};
		if (!checked(tensor_shard_prepare(
				     &plain,
				     &fixture->jobs[TEST_PLAIN_FIRST + rank]
					      .shard),
			     "prepare plain rows"))
			return false;
		if (!checked(tensor_shard_prepare(
				     &mx,
				     &fixture->jobs[TEST_MXFP4_FIRST + rank]
					      .shard),
			     "prepare MXFP4 rectangle") ||
		    !checked(tensor_shard_prepare(
				     &fp, &fixture->jobs[TEST_FP8_FIRST + rank]
						   .shard),
			     "prepare FP8 rectangle"))
			return false;
	}
	fixture->jobs[TEST_LATER_TENSOR] =
		(struct tp_load_job){ .tensor_index = TEST_LATER_TENSOR_INDEX,
				      .destination_offset =
					      TEST_SECOND_EXTENT_OFFSET,
				      .mode = TP_LOAD_FULL };
	struct tp_load_description description = { .pack = fixture->pack,
						   .jobs = fixture->jobs,
						   .job_count = TEST_JOB_COUNT,
						   .card_count = TEST_CARDS,
						   .maximum_extent_byte_count =
							   TEST_EXTENT_BYTES };
	for (uint32_t card = 0; card < TEST_CARDS; card++)
		description.card_arena_byte_count[card] = TEST_ARENA_BYTES;
	return tensors[TEST_LATER_TENSOR_INDEX].byte_count != 0 &&
	       checked(tp_load_prepare(&description, &fixture->plan),
		       "prepare selected tensor load");
}

static bool fixture_open(struct fixture *fixture)
{
	*fixture = (struct fixture){ 0 };
	test_memory_storage_reset();
	test_host_memory_backend_reset(TEST_CARDS, TEST_ARENA_BYTES);
	const struct test_pack_plan pack_plan = {
		.model_name = "tp-load-test",
		.units = unit_plans,
		.unit_count = TEST_UNITS,
		.tensors_per_extent = TEST_TENSORS_PER_EXTENT,
		.content_seed = TEST_SEED
	};
	if (!checked(test_pack_build(&pack_plan, &fixture->bytes,
				     &fixture->byte_count),
		     "build source pack"))
		return false;
	struct pack_tensor *const tensors = mutable_tensors(fixture->bytes);
	tensors[TEST_MXFP4_TENSOR].dimension_count = TEST_MATRIX_DIMENSIONS;
	tensors[TEST_MXFP4_TENSOR].dimensions[0] = TEST_MATRIX_OUTPUTS;
	tensors[TEST_MXFP4_TENSOR].dimensions[1] = TEST_MATRIX_INPUTS;
	tensors[TEST_MXFP4_TENSOR].element_type = PACK_ELEMENT_UINT8;
	tensors[TEST_MXFP4_TENSOR].weight_format = WEIGHT_FORMAT_E2M1_SCALED;
	tensors[TEST_MXFP4_TENSOR].bits_per_weight = TEST_MXFP4_BITS;
	tensors[TEST_MXFP4_TENSOR].byte_count = TEST_MXFP4_BYTES;
	tensors[TEST_FP8_TENSOR] = (struct pack_tensor){
		.byte_offset_in_extent =
			tensors[TEST_FP8_TENSOR].byte_offset_in_extent,
		.byte_count = TEST_FP8_BYTES,
		.extent_index = tensors[TEST_FP8_TENSOR].extent_index,
		.name_byte_offset = tensors[TEST_FP8_TENSOR].name_byte_offset,
		.name_byte_count = tensors[TEST_FP8_TENSOR].name_byte_count,
		.element_type = PACK_ELEMENT_UINT8,
		.dimension_count = TEST_MATRIX_DIMENSIONS,
		.dimensions = { TEST_MATRIX_OUTPUTS, TEST_MATRIX_INPUTS },
		.weight_format = WEIGHT_FORMAT_E4M3_SCALED,
		.bits_per_weight = TEST_FP8_BITS
	};
	tensors[TEST_PLAIN_TENSOR].byte_count = TEST_PLAIN_BYTES;
	tensors[TEST_PLAIN_TENSOR].dimension_count = TEST_MATRIX_DIMENSIONS;
	tensors[TEST_PLAIN_TENSOR].dimensions[0] = TEST_PLAIN_ROWS;
	tensors[TEST_PLAIN_TENSOR].dimensions[1] = TEST_PLAIN_INPUTS;
	tensors[TEST_PLAIN_TENSOR].element_type = PACK_ELEMENT_BFLOAT16;
	tensors[TEST_PLAIN_TENSOR].weight_format = WEIGHT_FORMAT_NONE;
	tensors[TEST_PLAIN_TENSOR].bits_per_weight = 0;
	test_pack_reseal(fixture->bytes);
	test_memory_storage_set_pack(fixture->bytes, fixture->byte_count);
	if (!checked(test_memory_storage.reader_open(copy_paths, TEST_COPIES,
						     &fixture->reader),
		     "open source copies") ||
	    !checked(pack_open_entry(&test_memory_storage, fixture->reader,
				     &fixture->pack),
		     "open selected pack") ||
	    !jobs_prepare(fixture))
		return false;
	fixture->request = (struct tp_load_request){
		.plan = &fixture->plan,
		.storage = &test_memory_storage,
		.reader = fixture->reader,
		.backend = test_workers != 0 ? &retiring_backend :
					       &test_host_memory_backend,
		.cards = fixture->cards,
		.source_workspace_byte_count =
			fixture->plan.source_workspace_byte_count,
		.output_workspace_byte_count =
			fixture->plan.output_workspace_byte_count,
		.wait_limit_nanoseconds = TEST_CLOSE_NANOSECONDS,
		.worker_count = test_workers
	};
	for (uint32_t card = 0; card < TEST_CARDS; card++) {
		if (!checked(test_host_memory_backend.card_open(
				     card, TEST_ARENA_BYTES,
				     &fixture->cards[card]),
			     "open destination card"))
			return false;
	}
	return checked(test_host_memory_backend.staging_allocate(
			       fixture->request.source_workspace_byte_count,
			       &fixture->request.source_workspace),
		       "source workspace") &&
	       checked(test_host_memory_backend.staging_allocate(
			       fixture->request.output_workspace_byte_count,
			       &fixture->request.output_workspace),
		       "output workspace");
}

static void output_check(const struct fixture *fixture)
{
	const struct pack_tensor *const tensors =
		pack_tensors_of(fixture->pack);
	const struct pack_extent *const extents =
		pack_extents_of(fixture->pack);
	unsigned char expected[TEST_EXTENT_BYTES] = { 0 };
	for (uint32_t index = 0; index < TEST_JOB_COUNT; index++) {
		const struct tp_load_job *const job = &fixture->jobs[index];
		const struct pack_tensor *const tensor =
			&tensors[job->tensor_index];
		const unsigned char *const source =
			(const unsigned char *)fixture->bytes +
			extents[tensor->extent_index].byte_offset +
			tensor->byte_offset_in_extent;
		uint64_t bytes = tensor->byte_count;
		if (job->mode == TP_LOAD_SHARD) {
			bytes = job->shard.shard_byte_count;
			expect(checked(tensor_shard_extract(&job->shard, source,
							    tensor->byte_count,
							    expected,
							    sizeof(expected)),
				       "independent extractor call"),
			       "expected shard available");
		} else {
			(void)memcpy(expected, source, (size_t)bytes);
		}
		const unsigned char *const delivered =
			test_host_memory_backend_arena_of(
				fixture->cards[job->card]) +
			job->destination_offset;
		expect(memcmp(delivered, expected, (size_t)bytes) == 0,
		       "original tensor/shard bytes on the selected card");
		const uint64_t padded = (bytes + PACK_DIGEST_WORD_BYTES - 1u) /
					PACK_DIGEST_WORD_BYTES *
					PACK_DIGEST_WORD_BYTES;
		for (uint64_t offset = bytes; offset < padded; offset++)
			expect(delivered[offset] == 0,
			       "non-word tensor and FP8 scale suffix have zero digest padding");
	}
}

static void successful_load(bool corrupt_first_copy, bool fail_first_copy)
{
	struct fixture fixture;
	if (!fixture_open(&fixture)) {
		fixture_close(&fixture);
		return;
	}
	const uint64_t reads = test_memory_storage_read_count(0);
	const struct pack_extent *const extents = pack_extents_of(fixture.pack);
	if (corrupt_first_copy)
		test_memory_storage_flip_bit(0, extents[0].byte_offset +
							TEST_CORRUPT_BYTE);
	if (fail_first_copy)
		test_memory_storage_fail_reads(0, reads + 1u);
	struct tp_load_result result = { 0 };
	if (checked(tp_load_run(&fixture.request, &result),
		    "run selected original tensor load")) {
		expect(result.extents_verified == TEST_CARDS &&
			       result.tensors_uploaded == TEST_JOB_COUNT,
		       "only the two selected extents and seven jobs run");
		expect(test_memory_storage_read_count(0) - reads == TEST_CARDS,
		       "each selected extent read once per attempted copy; host extent never read");
		expect(result.source_bytes_read ==
			       (uint64_t)(TEST_CARDS +
					  (corrupt_first_copy ? 1u : 0u)) *
				       TEST_EXTENT_BYTES,
		       "retained source read accounting includes a rejected corrupt copy");
		expect(test_host_memory_backend_staging_block_count() ==
				       TEST_CARDS + retired_count &&
			       retired_count == 2u * (test_workers != 0 ?
							      TEST_WORKERS :
							      0u),
		       "run retains both caller-owned workspaces and retires each worker's two slots");
		output_check(&fixture);
	}
	fixture_close(&fixture);
}

static struct tp_load_result poison_result(void)
{
	return (struct tp_load_result){ .source_bytes_read =
						TEST_EXPECTED_POISON,
					.uploaded_bytes = TEST_EXPECTED_POISON,
					.extents_verified = UINT32_MAX,
					.tensors_uploaded = UINT32_MAX };
}

static bool result_is_poison(const struct tp_load_result *result)
{
	return result->source_bytes_read == TEST_EXPECTED_POISON &&
	       result->uploaded_bytes == TEST_EXPECTED_POISON &&
	       result->extents_verified == UINT32_MAX &&
	       result->tensors_uploaded == UINT32_MAX;
}

static void failure_load(bool storage_failure, bool wrong_source,
			 bool wrong_upload)
{
	struct fixture fixture;
	if (!fixture_open(&fixture)) {
		fixture_close(&fixture);
		return;
	}
	const struct pack_extent *const extents = pack_extents_of(fixture.pack);
	for (uint32_t copy = 0; copy < TEST_COPIES; copy++) {
		if (storage_failure)
			test_memory_storage_fail_reads(
				copy,
				test_memory_storage_read_count(copy) + 1u);
		if (wrong_source)
			test_memory_storage_flip_bit(copy,
						     extents[0].byte_offset +
							     TEST_CORRUPT_BYTE);
	}
	if (wrong_upload)
		test_host_memory_backend_flip_bit_in_next_upload(0, 0);
	struct tp_load_result result = poison_result();
	const result_status expected = storage_failure ?
					       STATUS_INPUT_OUTPUT_ERROR :
					       STATUS_CORRUPT_PACK;
	expect(tp_load_run(&fixture.request, &result) == expected,
	       "source/link damage is refused");
	expect(result_is_poison(&result),
	       "failed run leaves its result untouched");
	/* pipelined, the sound second extent (one FULL job to card 0) may load beside the damaged first */
	if (!wrong_upload)
		expect(test_host_memory_backend_upload_count(
			       0) <= (test_workers != 0 && wrong_source ? 1u :
									  0u) &&
			       test_host_memory_backend_upload_count(1) == 0,
		       "no destination touched before original source extent verifies");
	fixture_close(&fixture);
}

static void prepare_refusal(const struct tp_load_description *description,
			    result_status expected)
{
	struct tp_load_plan output = {
		.source_workspace_byte_count = TEST_EXPECTED_POISON,
		.output_workspace_byte_count = TEST_EXPECTED_POISON
	};
	expect(tp_load_prepare(description, &output) == expected,
	       "malformed load preparation refused");
	expect(output.source_workspace_byte_count == TEST_EXPECTED_POISON &&
		       output.output_workspace_byte_count ==
			       TEST_EXPECTED_POISON,
	       "refused preparation preserves scalar outputs");
}

static void refusals(void)
{
	struct fixture fixture;
	if (!fixture_open(&fixture)) {
		fixture_close(&fixture);
		return;
	}
	struct tp_load_description changed = fixture.plan.description;
	prepare_refusal(NULL, STATUS_INVALID_ARGUMENT);
	changed.job_count = 0;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	changed = fixture.plan.description;
	changed.maximum_extent_byte_count = TP_LOAD_MAXIMUM_EXTENT_BYTES + 1u;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	changed.maximum_extent_byte_count = PACK_BLOCK_BYTES;
	prepare_refusal(&changed, STATUS_UNSUPPORTED);
	changed = fixture.plan.description;
	fixture.jobs[TEST_MXFP4_FIRST].tensor_index =
		pack_header_of(fixture.pack)->tensor_count;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	fixture.jobs[TEST_MXFP4_FIRST].tensor_index = TEST_MXFP4_TENSOR;
	fixture.jobs[TEST_MXFP4_FIRST].destination_offset = 1;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	fixture.jobs[TEST_MXFP4_FIRST].destination_offset = 0;
	changed.card_arena_byte_count[0] = UINT64_MAX;
	fixture.jobs[TEST_MXFP4_FIRST].destination_offset =
		UINT64_MAX - WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES + 1u;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	fixture.jobs[TEST_MXFP4_FIRST].destination_offset = 0;
	changed = fixture.plan.description;
	fixture.jobs[TEST_MXFP4_SECOND].card = 0;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	fixture.jobs[TEST_MXFP4_SECOND].card = 1;
	fixture.jobs[TEST_MXFP4_SECOND].card = TEST_CARDS;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	fixture.jobs[TEST_MXFP4_SECOND].card = 1;
	fixture.jobs[TEST_MXFP4_FIRST].shard.shard_byte_count++;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	fixture.jobs[TEST_MXFP4_FIRST].shard.shard_byte_count--;
	fixture.jobs[TEST_MXFP4_FIRST].mode = TP_LOAD_MODE_COUNT;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	fixture.jobs[TEST_MXFP4_FIRST].mode = TP_LOAD_SHARD;
	const struct tp_load_job saved = fixture.jobs[TEST_MXFP4_SECOND];
	fixture.jobs[TEST_MXFP4_SECOND] = fixture.jobs[TEST_LATER_TENSOR];
	fixture.jobs[TEST_MXFP4_SECOND].card = 1;
	fixture.jobs[TEST_MXFP4_SECOND].destination_offset = 0;
	prepare_refusal(&changed, STATUS_INVALID_ARGUMENT);
	fixture.jobs[TEST_MXFP4_SECOND] = saved;
	fixture.jobs[TEST_LATER_TENSOR].tensor_index = TEST_HOST_TENSOR_INDEX;
	prepare_refusal(&changed, STATUS_UNSUPPORTED);
	fixture.jobs[TEST_LATER_TENSOR].tensor_index = TEST_LATER_TENSOR_INDEX;
	struct tp_load_request request = fixture.request;
	struct tp_load_result result = poison_result();
	expect(tp_load_run(&request,
			   (struct tp_load_result *)request.source_workspace) ==
		       STATUS_INVALID_ARGUMENT,
	       "workspace overlapping result metadata refused");
	request.output_workspace = request.source_workspace;
	expect(tp_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "workspace alias refused");
	request = fixture.request;
	request.source_workspace_byte_count--;
	expect(tp_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "short source workspace refused");
	request = fixture.request;
	request.output_workspace_byte_count--;
	expect(tp_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "short output workspace refused");
	request = fixture.request;
	request.wait_limit_nanoseconds = 0;
	expect(tp_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "unbounded wait refused");
	expect(result_is_poison(&result), "run refusals preserve result");
	fixture_close(&fixture);
}

MUST_CHECK static result_status timeout_wait(struct device_card *card,
					     uint64_t ticket, uint64_t limit)
{
	(void)card;
	(void)ticket;
	(void)limit;
	return STATUS_TIMED_OUT;
}

static void incomplete_upload(void)
{
	struct fixture fixture;
	if (!fixture_open(&fixture)) {
		fixture_close(&fixture);
		return;
	}
	struct backend_operations backend = *fixture.request.backend;
	backend.upload_wait = timeout_wait;
	fixture.request.backend = &backend;
	struct tp_load_result result = poison_result();
	expect(tp_load_run(&fixture.request, &result) == STATUS_TIMED_OUT,
	       "incomplete upload returns bounded wait status");
	expect(test_host_memory_backend_staging_block_count() ==
			       TEST_CARDS + retired_count &&
		       result_is_poison(&result),
	       "incomplete copy retains caller workspaces and result");
	fixture_close(
		&fixture); /* Card drain precedes releasing the upload's source. */
}

static bool host_result_is_poison(const struct tp_host_load_result *result)
{
	return result->requested_bytes == UINT64_MAX &&
	       result->read_calls == UINT64_MAX &&
	       result->copy_retries == UINT64_MAX &&
	       result->extents_verified == UINT32_MAX &&
	       result->reserved_zero == UINT32_MAX;
}

/* The existing fixture's tiny HOST_ONLY extent is selected independently of every GPU tensor job. */
static void host_load(enum host_fault fault)
{
	struct fixture fixture;
	if (!fixture_open(&fixture)) {
		fixture_close(&fixture);
		return;
	}
	uint64_t offsets[TEST_UNITS];
	struct tp_host_load_plan plan;
	unsigned char *const arena =
		aligned_alloc(PACK_BLOCK_BYTES, TEST_EXTENT_BYTES);
	if (!checked(tp_host_load_prepare(fixture.pack, offsets, TEST_UNITS,
					  &plan),
		     "prepare independent host map") ||
	    arena == NULL) {
		expect(arena != NULL, "host arena allocation");
		free(arena);
		fixture_close(&fixture);
		return;
	}
	expect(plan.host_byte_count == TEST_EXTENT_BYTES &&
		       plan.host_extent_count == 1 &&
		       offsets[0] == TP_HOST_LOAD_NOWHERE &&
		       offsets[1] == TP_HOST_LOAD_NOWHERE &&
		       offsets[TEST_HOST_EXTENT] == 0,
	       "host map contains only the HOST_ONLY extent");
	const struct pack_extent *const extents = pack_extents_of(fixture.pack);
	const uint64_t source_offset = extents[TEST_HOST_EXTENT].byte_offset;
	const uint64_t work_before = test_host_memory_backend_work_count();
	if (fault == HOST_READ_FALLBACK || fault == HOST_ALL_READS_FAIL)
		test_memory_storage_fail_reads(
			0, test_memory_storage_read_count(0) + 1u);
	if (fault == HOST_ALL_READS_FAIL)
		test_memory_storage_fail_reads(
			1, test_memory_storage_read_count(1) + 1u);
	if (fault == HOST_DIGEST_FALLBACK || fault == HOST_ALL_DIGESTS_FAIL)
		test_memory_storage_flip_bit(0,
					     source_offset + TEST_CORRUPT_BYTE);
	if (fault == HOST_ALL_DIGESTS_FAIL)
		test_memory_storage_flip_bit(1,
					     source_offset + TEST_CORRUPT_BYTE);
	const struct tp_host_load_request request = {
		.plan = &plan,
		.storage = &test_memory_storage,
		.reader = fixture.reader,
		.host_arena = arena,
		.host_arena_byte_count = TEST_EXTENT_BYTES,
		.worker_count = test_workers
	};
	struct tp_host_load_result result = { UINT64_MAX, UINT64_MAX,
					      UINT64_MAX, UINT32_MAX,
					      UINT32_MAX };
	const result_status status = tp_host_load_run(&request, &result);
	if (fault == HOST_ALL_READS_FAIL || fault == HOST_ALL_DIGESTS_FAIL) {
		expect(status == (fault == HOST_ALL_READS_FAIL ?
					  STATUS_INPUT_OUTPUT_ERROR :
					  STATUS_CORRUPT_PACK) &&
			       host_result_is_poison(&result),
		       "failed host load preserves result and cannot attach");
	} else {
		const uint64_t copies = fault == HOST_SOUND ? 1u : TEST_COPIES;
		expect(status == STATUS_OK &&
			       result.requested_bytes ==
				       TEST_EXTENT_BYTES * copies &&
			       result.read_calls == copies &&
			       result.copy_retries == copies - 1u &&
			       result.extents_verified == 1 &&
			       result.reserved_zero == 0 &&
			       memcmp(arena,
				      (const unsigned char *)fixture.bytes +
					      source_offset,
				      TEST_EXTENT_BYTES) == 0,
		       "host extent original bytes/digest and successful retry counters");
	}
	expect(test_host_memory_backend_work_count() == work_before,
	       "host load makes no backend call");
	free(arena);
	fixture_close(&fixture);
}

/* A modest two-chunk extent plus a second extent checks digest positions across chunks and reset per extent. */
static void host_chunk_boundary(void)
{
	const struct test_pack_unit_plan units[] = {
		{ PLACEMENT_UNIT_STAGE, 0, 1, 1 },
		{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1,
		  TEST_HOST_STREAM_BLOCKS },
		{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, 1 }
	};
	const struct test_pack_plan source = { .model_name =
						       "tp-host-stream-test",
					       .units = units,
					       .unit_count = TEST_UNITS,
					       .tensors_per_extent = 1,
					       .content_seed = TEST_SEED };
	void *bytes = NULL, *arena = NULL;
	uint64_t byte_count = 0, offsets[TEST_UNITS];
	struct pack_record *pack = NULL;
	struct storage_reader *reader = NULL;
	struct tp_host_load_plan plan;
	test_memory_storage_reset();
	bool ready = checked(test_pack_build(&source, &bytes, &byte_count),
			     "build chunk-boundary pack");
	if (ready) {
		test_memory_storage_set_pack(bytes, byte_count);
		ready = checked(test_memory_storage.reader_open(
					copy_paths, TEST_COPIES, &reader),
				"open chunk-boundary copies") &&
			checked(pack_open_entry(&test_memory_storage, reader,
						&pack),
				"open chunk-boundary pack") &&
			checked(tp_host_load_prepare(pack, offsets, TEST_UNITS,
						     &plan),
				"prepare chunk map");
	}
	if (ready) {
		arena = aligned_alloc(PACK_BLOCK_BYTES, plan.host_byte_count);
		ready = arena != NULL;
		expect(ready, "allocate modest chunk-boundary host arena");
	}
	if (ready) {
		const struct pack_extent *const extents = pack_extents_of(pack);
		test_memory_storage_flip_bit(
			0, extents[1].byte_offset + TP_HOST_LOAD_CHUNK_BYTES +
				   TEST_CORRUPT_BYTE);
		const struct tp_host_load_request request = {
			.plan = &plan,
			.storage = &test_memory_storage,
			.reader = reader,
			.host_arena = arena,
			.host_arena_byte_count = plan.host_byte_count,
			.worker_count = test_workers
		};
		struct tp_host_load_result result;
		expect(tp_host_load_run(&request, &result) == STATUS_OK &&
			       plan.host_byte_count ==
				       TEST_HOST_STREAM_BYTES +
					       PACK_BLOCK_BYTES &&
			       offsets[0] == TP_HOST_LOAD_NOWHERE &&
			       offsets[1] == 0 &&
			       offsets[TEST_HOST_EXTENT] ==
				       TEST_HOST_STREAM_BYTES &&
			       result.requested_bytes ==
				       TEST_HOST_STREAM_BYTES * TEST_COPIES +
					       PACK_BLOCK_BYTES &&
			       result.read_calls ==
				       (uint64_t)TEST_COPIES *
						       TEST_HOST_STREAM_CHUNKS +
					       1u &&
			       result.copy_retries == 1 &&
			       result.extents_verified ==
				       TEST_HOST_STREAM_EXTENTS &&
			       memcmp(arena,
				      (const unsigned char *)bytes +
					      extents[1].byte_offset,
				      TEST_HOST_STREAM_BYTES) == 0 &&
			       memcmp((const unsigned char *)arena +
					      offsets[TEST_HOST_EXTENT],
				      (const unsigned char *)bytes +
					      extents[TEST_HOST_EXTENT]
						      .byte_offset,
				      PACK_BLOCK_BYTES) == 0,
		       "streamed positional digest, full-copy retry and concatenated host extents");
	}
	free(arena);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	free(bytes);
}

static void host_refusals(void)
{
	struct fixture fixture;
	if (!fixture_open(&fixture)) {
		fixture_close(&fixture);
		return;
	}
	uint64_t offsets[TEST_UNITS];
	struct tp_host_load_plan plan;
	unsigned char *const arena =
		aligned_alloc(PACK_BLOCK_BYTES, TEST_EXTENT_BYTES);
	if (!checked(tp_host_load_prepare(fixture.pack, offsets, TEST_UNITS,
					  &plan),
		     "prepare host refusal fixture") ||
	    arena == NULL) {
		expect(arena != NULL, "host refusal arena allocation");
		free(arena);
		fixture_close(&fixture);
		return;
	}
	const uint64_t reads_before = test_memory_storage_read_count(0) +
				      test_memory_storage_read_count(1);
	expect(tp_host_load_prepare(fixture.pack, offsets, TEST_UNITS - 1u,
				    &plan) == STATUS_INVALID_ARGUMENT,
	       "wrong host map count refused");
	expect(tp_host_load_prepare(fixture.pack, offsets, TEST_UNITS,
				    (void *)offsets) == STATUS_INVALID_ARGUMENT,
	       "host plan and map alias refused");
	struct tp_host_load_request request = { .plan = &plan,
						.storage = &test_memory_storage,
						.reader = fixture.reader,
						.host_arena = arena,
						.host_arena_byte_count =
							TEST_EXTENT_BYTES,
						.worker_count = test_workers };
	struct tp_host_load_result result = { UINT64_MAX, UINT64_MAX,
					      UINT64_MAX, UINT32_MAX,
					      UINT32_MAX };
	offsets[0] = 0;
	expect(tp_host_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "unselected host map entry refused");
	offsets[0] = TP_HOST_LOAD_NOWHERE;
	offsets[TEST_HOST_EXTENT] = PACK_BLOCK_BYTES;
	expect(tp_host_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "noncanonical host range refused");
	offsets[TEST_HOST_EXTENT] = 0;
	request.host_arena_byte_count--;
	expect(tp_host_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "short host arena refused");
	request.host_arena_byte_count++;
	request.host_arena = arena + 1u;
	expect(tp_host_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "reader-alignment violation refused");
	request.host_arena = (void *)(UINTPTR_MAX - PACK_BLOCK_BYTES);
	expect(tp_host_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "host address wrap refused");
	request.host_arena = &request;
	expect(tp_host_load_run(&request, &result) == STATUS_INVALID_ARGUMENT,
	       "host destination/request alias refused");
	request.host_arena = arena;
	expect(tp_host_load_run(&request, (void *)arena) ==
		       STATUS_INVALID_ARGUMENT,
	       "host destination/result alias refused");
	expect(host_result_is_poison(&result) &&
		       reads_before ==
			       test_memory_storage_read_count(0) +
				       test_memory_storage_read_count(1),
	       "host admission failures preserve result and perform no reads");
	free(arena);
	fixture_close(&fixture);
}

/* Turbo load's choice of copy: copy 1 delivers a damaged byte in the first extent and in the host extent; whichever
 * copy each read is given, the pipelined load falls back and places the original bytes, and the striped host
 * extent is read again copy by copy. */
static void striped_load(uint32_t split)
{
	struct fixture fixture;
	struct tp_load_copies copies;
	if (!fixture_open(&fixture) ||
	    !checked(tp_load_copies_init(&copies, 1u), "copies")) {
		fixture_close(&fixture);
		return;
	}
	const struct pack_extent *const extents = pack_extents_of(fixture.pack);
	test_memory_storage_flip_bit(1, extents[0].byte_offset +
						TEST_CORRUPT_BYTE);
	copies.split =
		split; /* 1: the cards' extents go to the damaged copy 1 first */
	fixture.request.copies = &copies;
	struct tp_load_result result = { 0 };
	if (checked(tp_load_run(&fixture.request, &result),
		    "striped load with a damaged copy")) {
		expect(result.extents_verified == TEST_CARDS &&
			       result.tensors_uploaded == TEST_JOB_COUNT,
		       "striped load verifies every extent and uploads every job");
		expect(copies.delivered_bytes[0] + copies.delivered_bytes[1] ==
				       result.source_bytes_read &&
			       copies.reads_in_flight[0] == 0 &&
			       copies.reads_in_flight[1] == 0,
		       "the copies' counters add up to what was read");
		output_check(&fixture);
	}
	uint64_t offsets[TEST_UNITS];
	struct tp_host_load_plan plan;
	unsigned char *const arena =
		aligned_alloc(PACK_BLOCK_BYTES, TEST_EXTENT_BYTES);
	test_memory_storage_flip_bit(1, extents[TEST_HOST_EXTENT].byte_offset +
						TEST_CORRUPT_BYTE);
	if (arena != NULL && checked(tp_host_load_prepare(fixture.pack, offsets,
							  TEST_UNITS, &plan),
				     "prepare striped host map")) {
		const struct tp_host_load_request request = {
			.plan = &plan,
			.storage = &test_memory_storage,
			.reader = fixture.reader,
			.host_arena = arena,
			.host_arena_byte_count = TEST_EXTENT_BYTES,
			.worker_count = TEST_WORKERS,
			.copies = &copies
		};
		struct tp_host_load_result host = { 0 };
		expect(tp_host_load_run(&request, &host) == STATUS_OK &&
			       host.extents_verified == 1 &&
			       memcmp(arena,
				      (const unsigned char *)fixture.bytes +
					      extents[TEST_HOST_EXTENT]
						      .byte_offset,
				      TEST_EXTENT_BYTES) == 0,
		       "striped host extent with a damaged copy holds the original bytes");
	}
	free(arena);
	tp_load_copies_destroy(&copies);
	fixture_close(&fixture);
}

static void suite(void)
{
	successful_load(false, false);
	successful_load(true, false);
	successful_load(false, true);
	failure_load(true, false, false);
	failure_load(false, true, false);
	failure_load(false, false, true);
	refusals();
	incomplete_upload();
	host_load(HOST_SOUND);
	host_load(HOST_READ_FALLBACK);
	host_load(HOST_DIGEST_FALLBACK);
	host_load(HOST_ALL_READS_FAIL);
	host_load(HOST_ALL_DIGESTS_FAIL);
	host_chunk_boundary();
	host_refusals();
}

int main(void)
{
	retiring_backend = test_host_memory_backend;
	retiring_backend.staging_retire = retire_later;
	suite();
	test_workers = TEST_WORKERS;
	suite();
	striped_load(0);
	striped_load(1);
	if (failures == 0)
		(void)puts(
			"tp_load_test: selected extents, original shards, padding, "
			"copy fallback, corruption and lifetime passed");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
