/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "index_select_split_fixture.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/model_layout.h"
#include "anemone/step_records.h"
#include "prepare_common.h"

static const struct backend_operations *const backend = &backend_cuda_value;
enum split_pass {
	HISTOGRAM_HIGH_PASS,
	CUT_HIGH_PASS,
	HISTOGRAM_LOW_PASS,
	CUT_LOW_PASS,
	OUTPUT_PASS
};

_Noreturn void test_split_fail(int line, const char *message)
{
	(void)fprintf(stderr, "FAILED split fixture at line %d: %s\n", line,
		      message);
	exit(EXIT_FAILURE);
}

void test_split_open(struct test_split_fixture *fixture, uint64_t arena_bytes,
		     uint32_t arithmetic_class)
{
	static const char *const names[TEST_SPLIT_KERNELS] = {
		KERNEL_INDEX_SELECT,
		KERNEL_INDEX_CANDIDATES,
		KERNEL_INDEX_SPLIT_HISTOGRAM_HIGH,
		KERNEL_INDEX_SPLIT_CUT_HIGH,
		KERNEL_INDEX_SPLIT_HISTOGRAM_LOW,
		KERNEL_INDEX_SPLIT_CUT_LOW,
		KERNEL_INDEX_SPLIT_OUTPUT
	};
	struct card_description_record description;
	uint32_t card_count = 0;
	*fixture = (struct test_split_fixture){ 0 };
	TEST_SPLIT_REQUIRE(backend->card_count(&card_count) == STATUS_OK &&
			   card_count == 1);
	TEST_SPLIT_REQUIRE(backend->card_describe(0, &description) ==
			   STATUS_OK);
	const bool is_cmp = arithmetic_class == ARITHMETIC_CLASS_SM_80;
	TEST_SPLIT_REQUIRE(
		is_cmp ? strstr(description.name, "CMP 170HX") != NULL :
			 strstr(description.name, "RTX PRO 6000") != NULL);
	TEST_SPLIT_REQUIRE(description.memory_free_bytes >=
				   description.memory_reserved_bytes &&
			   arena_bytes <=
				   description.memory_free_bytes -
					   description.memory_reserved_bytes);
	TEST_SPLIT_REQUIRE(backend->card_open(0, arena_bytes, &fixture->card) ==
			   STATUS_OK);
	fixture->arena_bytes = arena_bytes;
	void *staging = NULL;
	TEST_SPLIT_REQUIRE(backend->staging_allocate(TEST_SPLIT_STAGING_BYTES,
						     &staging) == STATUS_OK);
	fixture->staging = staging;
	const char *const file =
		arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			KERNEL_FILE_INDEX_SCAN_NATIVE :
			KERNEL_FILE_INDEX_SCAN;
	for (uint32_t kernel = 0; kernel < TEST_SPLIT_KERNELS; kernel++) {
		const uint32_t bytes = kernel == TEST_SPLIT_LEGACY_SELECT ?
					       INDEX_SELECT_ARGUMENT_BYTES :
				       kernel == TEST_SPLIT_LEGACY_CANDIDATES ?
					       INDEX_CANDIDATES_ARGUMENT_BYTES :
					       INDEX_SPLIT_ARGUMENT_BYTES;
		TEST_SPLIT_REQUIRE(backend->kernel_prepare(
					   fixture->card, file, names[kernel],
					   bytes, &fixture->handles[kernel]) ==
				   STATUS_OK);
	}
	(void)printf("DEVICE class=%" PRIu32 " name=%s bus=%s arena=%" PRIu64
		     "\n",
		     arithmetic_class, description.name,
		     description.bus_address, arena_bytes);
}

void test_split_close(struct test_split_fixture *fixture)
{
	backend->card_close(fixture->card, TEST_SPLIT_WAIT_NS);
	backend->staging_free(fixture->staging);
	fixture->card = NULL;
	fixture->staging = NULL;
}

static struct linear_range reserve(struct test_split_fixture *fixture,
				   uint64_t bytes)
{
	const struct linear_range range = { fixture->used_bytes, bytes };
	const uint64_t guard =
		(fixture->used_bytes + bytes + TEST_SPLIT_GUARD_BYTES - 1u) /
		TEST_SPLIT_GUARD_BYTES * TEST_SPLIT_GUARD_BYTES;
	TEST_SPLIT_REQUIRE(fixture->guard_count < TEST_SPLIT_RANGES &&
			   guard + TEST_SPLIT_GUARD_BYTES <=
				   fixture->arena_bytes);
	fixture->guard_offsets[fixture->guard_count++] = guard;
	fixture->used_bytes = guard + TEST_SPLIT_GUARD_BYTES;
	return range;
}

void test_split_layout(struct test_split_fixture *fixture, uint32_t rows,
		       uint32_t entries)
{
	fixture->used_bytes = 0;
	fixture->guard_count = 0;
	fixture->step =
		reserve(fixture, STEP_BLOCK_BYTES_VALUE +
					 (uint64_t)rows * STEP_ROW_BYTES_VALUE);
	fixture->scores =
		reserve(fixture, (uint64_t)rows * entries * INDEX_SCORE_BYTES);
	fixture->candidates =
		reserve(fixture, (uint64_t)rows * MODEL_CANDIDATES_ROW_BYTES);
	fixture->legacy =
		reserve(fixture, (uint64_t)rows * MODEL_CANDIDATES_ROW_BYTES);
	fixture->split =
		reserve(fixture, (uint64_t)rows * MODEL_CANDIDATES_ROW_BYTES);
	fixture->scratch =
		reserve(fixture, (uint64_t)rows * INDEX_SPLIT_ROW_BYTES);
	for (uint32_t guard = 0; guard < fixture->guard_count; guard++) {
		const struct linear_range range = {
			fixture->guard_offsets[guard], TEST_SPLIT_GUARD_BYTES
		};
		test_split_fill(fixture, range, TEST_SPLIT_GUARD_BYTE);
	}
}

void test_split_upload(struct test_split_fixture *fixture, uint64_t offset,
		       const void *source, uint64_t bytes)
{
	const unsigned char *next = source;
	while (bytes != 0) {
		const uint64_t count = bytes < TEST_SPLIT_STAGING_BYTES ?
					       bytes :
					       TEST_SPLIT_STAGING_BYTES;
		uint64_t ticket = 0;
		if (next != fixture->staging) {
			memcpy(fixture->staging, next, (size_t)count);
		}
		TEST_SPLIT_REQUIRE(backend->upload_begin(fixture->card, offset,
							 fixture->staging,
							 count,
							 &ticket) == STATUS_OK);
		TEST_SPLIT_REQUIRE(backend->upload_wait(fixture->card, ticket,
							TEST_SPLIT_WAIT_NS) ==
				   STATUS_OK);
		offset += count;
		bytes -= count;
		next += count;
	}
}

void test_split_download(struct test_split_fixture *fixture, uint64_t offset,
			 void *destination, uint64_t bytes)
{
	unsigned char *next = destination;
	while (bytes != 0) {
		const uint64_t count = bytes < TEST_SPLIT_STAGING_BYTES ?
					       bytes :
					       TEST_SPLIT_STAGING_BYTES;
		uint64_t ticket = 0;
		TEST_SPLIT_REQUIRE(
			backend->download_begin(fixture->card, offset,
						fixture->staging, count,
						&ticket) == STATUS_OK);
		TEST_SPLIT_REQUIRE(backend->upload_wait(fixture->card, ticket,
							TEST_SPLIT_WAIT_NS) ==
				   STATUS_OK);
		memcpy(next, fixture->staging, (size_t)count);
		offset += count;
		bytes -= count;
		next += count;
	}
}

void test_split_fill(struct test_split_fixture *fixture,
		     struct linear_range range, unsigned char value)
{
	memset(fixture->staging, value, TEST_SPLIT_STAGING_BYTES);
	for (uint64_t done = 0; done < range.byte_count;
	     done += TEST_SPLIT_STAGING_BYTES) {
		const uint64_t left = range.byte_count - done;
		test_split_upload(fixture, range.byte_offset + done,
				  fixture->staging,
				  left < TEST_SPLIT_STAGING_BYTES ?
					  left :
					  TEST_SPLIT_STAGING_BYTES);
	}
}

void test_split_check_guards(struct test_split_fixture *fixture)
{
	unsigned char actual[TEST_SPLIT_GUARD_BYTES];
	for (uint32_t guard = 0; guard < fixture->guard_count; guard++) {
		test_split_download(fixture, fixture->guard_offsets[guard],
				    actual, sizeof(actual));
		for (uint32_t byte = 0; byte < TEST_SPLIT_GUARD_BYTES; byte++) {
			TEST_SPLIT_REQUIRE(actual[byte] ==
					   TEST_SPLIT_GUARD_BYTE);
		}
	}
}

void test_split_launches(const struct test_split_fixture *fixture,
			 uint32_t rows, uint32_t context, uint32_t ratio,
			 bool is_block, bool is_reader, uint32_t segments,
			 struct kernel_launch *legacy,
			 struct kernel_launch *split)
{
	const struct index_select_arguments_record select = {
		fixture->step.byte_offset,
		fixture->scores.byte_offset,
		fixture->legacy.byte_offset,
		is_reader ? fixture->candidates.byte_offset :
			    INDEX_NO_CANDIDATES,
		context,
		ratio,
		0,
		0
	};
	const struct index_candidates_arguments candidates = {
		fixture->step.byte_offset,
		fixture->scores.byte_offset,
		fixture->legacy.byte_offset,
		context,
		context,
		0,
		0
	};
	prepare_launch_fill(
		legacy,
		fixture->handles[is_block ? TEST_SPLIT_LEGACY_CANDIDATES :
					    TEST_SPLIT_LEGACY_SELECT],
		rows, 1, INDEX_SELECT_THREADS_VALUE, 0,
		is_block ? (const void *)&candidates : (const void *)&select,
		is_block ? INDEX_CANDIDATES_ARGUMENT_BYTES :
			   INDEX_SELECT_ARGUMENT_BYTES);
	const struct index_split_arguments record = {
		fixture->step.byte_offset,
		fixture->scores.byte_offset,
		fixture->split.byte_offset,
		is_reader ? fixture->candidates.byte_offset :
			    INDEX_NO_CANDIDATES,
		fixture->scratch.byte_offset,
		context,
		ratio,
		segments,
		is_block ? 1u : 0u,
		0,
		0
	};
	for (uint32_t pass = 0; pass < INDEX_SPLIT_LAUNCH_COUNT; pass++) {
		const bool is_cut = pass == CUT_HIGH_PASS ||
				    pass == CUT_LOW_PASS;
		prepare_launch_fill(
			&split[pass],
			fixture->handles[TEST_SPLIT_FIRST_PASS + pass], rows,
			is_cut ? 1u : segments, INDEX_SPLIT_THREADS, 0, &record,
			INDEX_SPLIT_ARGUMENT_BYTES);
	}
}

void test_split_run(struct test_split_fixture *fixture,
		    const struct kernel_launch *launches, uint32_t count)
{
	TEST_SPLIT_REQUIRE(backend->launch_run(fixture->card, launches,
					       count) == STATUS_OK);
	TEST_SPLIT_REQUIRE(
		backend->launch_wait(fixture->card, TEST_SPLIT_WAIT_NS) ==
		STATUS_OK);
}
