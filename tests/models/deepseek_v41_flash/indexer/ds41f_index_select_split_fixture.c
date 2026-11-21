/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "ds41f_index_select_split_fixture.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "engine/program/step_records.h"
#include "engine/library/prepare_checks.h"

static const struct backend_operations *const backend = &backend_cuda_value;
enum split_pass {
	HISTOGRAM_HIGH_PASS,
	CUT_HIGH_PASS,
	HISTOGRAM_LOW_PASS,
	CUT_LOW_PASS,
	OUTPUT_PASS
};

_Noreturn void ds41f_test_split_fail(int line, const char *message)
{
	(void)fprintf(stderr, "FAILED split fixture at line %d: %s\n", line,
		      message);
	exit(EXIT_FAILURE);
}

void ds41f_test_split_open(struct ds41f_test_split_fixture *fixture,
			   uint64_t arena_bytes, uint32_t arithmetic_class)
{
	static const char *const names[DS41F_TEST_SPLIT_KERNELS] = {
		DS41F_KERNEL_INDEX_SELECT,
		DS41F_KERNEL_INDEX_CANDIDATES,
		DS41F_KERNEL_INDEX_SPLIT_HISTOGRAM_HIGH,
		DS41F_KERNEL_INDEX_SPLIT_CUT_HIGH,
		DS41F_KERNEL_INDEX_SPLIT_HISTOGRAM_LOW,
		DS41F_KERNEL_INDEX_SPLIT_CUT_LOW,
		DS41F_KERNEL_INDEX_SPLIT_OUTPUT
	};
	struct card_description_record description;
	uint32_t card_count = 0;
	*fixture = (struct ds41f_test_split_fixture){ 0 };
	DS41F_TEST_SPLIT_REQUIRE(backend->card_count(&card_count) ==
					 STATUS_OK &&
				 card_count == 1);
	DS41F_TEST_SPLIT_REQUIRE(backend->card_describe(0, &description) ==
				 STATUS_OK);
	const bool is_cmp = arithmetic_class == ARITHMETIC_CLASS_SM_80;
	DS41F_TEST_SPLIT_REQUIRE(
		is_cmp ? strstr(description.name, "CMP 170HX") != NULL :
			 strstr(description.name, "RTX PRO 6000") != NULL);
	DS41F_TEST_SPLIT_REQUIRE(
		description.memory_free_bytes >=
			description.memory_reserved_bytes &&
		arena_bytes <= description.memory_free_bytes -
				       description.memory_reserved_bytes);
	DS41F_TEST_SPLIT_REQUIRE(
		backend->card_open(0, arena_bytes, &fixture->card) ==
		STATUS_OK);
	fixture->arena_bytes = arena_bytes;
	void *staging = NULL;
	DS41F_TEST_SPLIT_REQUIRE(
		backend->staging_allocate(DS41F_TEST_SPLIT_STAGING_BYTES,
					  &staging) == STATUS_OK);
	fixture->staging = staging;
	const char *const file =
		arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			DS41F_MODULE_INDEX_SCAN_NATIVE :
			DS41F_MODULE_INDEX_SCAN;
	for (uint32_t kernel = 0; kernel < DS41F_TEST_SPLIT_KERNELS; kernel++) {
		const uint32_t bytes =
			kernel == DS41F_TEST_SPLIT_LEGACY_SELECT ?
				DS41F_INDEX_SELECT_ARGUMENT_BYTES :
			kernel == DS41F_TEST_SPLIT_LEGACY_CANDIDATES ?
				DS41F_INDEX_CANDIDATES_ARGUMENT_BYTES :
				DS41F_INDEX_SPLIT_ARGUMENT_BYTES;
		DS41F_TEST_SPLIT_REQUIRE(
			backend->kernel_prepare(
				fixture->card, file, names[kernel], bytes,
				&fixture->handles[kernel]) == STATUS_OK);
	}
	(void)printf("DEVICE class=%" PRIu32 " name=%s bus=%s arena=%" PRIu64
		     "\n",
		     arithmetic_class, description.name,
		     description.bus_address, arena_bytes);
}

void ds41f_test_split_close(struct ds41f_test_split_fixture *fixture)
{
	backend->card_close(fixture->card, DS41F_TEST_SPLIT_WAIT_NS);
	backend->staging_free(fixture->staging);
	fixture->card = NULL;
	fixture->staging = NULL;
}

static struct linear_range reserve(struct ds41f_test_split_fixture *fixture,
				   uint64_t bytes)
{
	const struct linear_range range = { fixture->used_bytes, bytes };
	const uint64_t guard = (fixture->used_bytes + bytes +
				DS41F_TEST_SPLIT_GUARD_BYTES - 1u) /
			       DS41F_TEST_SPLIT_GUARD_BYTES *
			       DS41F_TEST_SPLIT_GUARD_BYTES;
	DS41F_TEST_SPLIT_REQUIRE(
		fixture->guard_count < DS41F_TEST_SPLIT_RANGES &&
		guard + DS41F_TEST_SPLIT_GUARD_BYTES <= fixture->arena_bytes);
	fixture->guard_offsets[fixture->guard_count++] = guard;
	fixture->used_bytes = guard + DS41F_TEST_SPLIT_GUARD_BYTES;
	return range;
}

void ds41f_test_split_layout(struct ds41f_test_split_fixture *fixture,
			     uint32_t rows, uint32_t entries)
{
	fixture->used_bytes = 0;
	fixture->guard_count = 0;
	fixture->step =
		reserve(fixture, STEP_BLOCK_BYTES_VALUE +
					 (uint64_t)rows * STEP_ROW_BYTES_VALUE);
	fixture->scores = reserve(fixture, (uint64_t)rows * entries *
						   DS41F_INDEX_SCORE_BYTES);
	fixture->candidates =
		reserve(fixture, (uint64_t)rows * DS41F_CANDIDATES_ROW_BYTES);
	fixture->legacy =
		reserve(fixture, (uint64_t)rows * DS41F_CANDIDATES_ROW_BYTES);
	fixture->split =
		reserve(fixture, (uint64_t)rows * DS41F_CANDIDATES_ROW_BYTES);
	fixture->scratch =
		reserve(fixture, (uint64_t)rows * DS41F_INDEX_SPLIT_ROW_BYTES);
	for (uint32_t guard = 0; guard < fixture->guard_count; guard++) {
		const struct linear_range range = {
			fixture->guard_offsets[guard],
			DS41F_TEST_SPLIT_GUARD_BYTES
		};
		ds41f_test_split_fill(fixture, range,
				      DS41F_TEST_SPLIT_GUARD_BYTE);
	}
}

void ds41f_test_split_upload(struct ds41f_test_split_fixture *fixture,
			     uint64_t offset, const void *source,
			     uint64_t bytes)
{
	const unsigned char *next = source;
	while (bytes != 0) {
		const uint64_t count = bytes < DS41F_TEST_SPLIT_STAGING_BYTES ?
					       bytes :
					       DS41F_TEST_SPLIT_STAGING_BYTES;
		uint64_t ticket = 0;
		if (next != fixture->staging) {
			memcpy(fixture->staging, next, (size_t)count);
		}
		DS41F_TEST_SPLIT_REQUIRE(
			backend->upload_begin(fixture->card, offset,
					      fixture->staging, count,
					      &ticket) == STATUS_OK);
		DS41F_TEST_SPLIT_REQUIRE(
			backend->upload_wait(fixture->card, ticket,
					     DS41F_TEST_SPLIT_WAIT_NS) ==
			STATUS_OK);
		offset += count;
		bytes -= count;
		next += count;
	}
}

void ds41f_test_split_download(struct ds41f_test_split_fixture *fixture,
			       uint64_t offset, void *destination,
			       uint64_t bytes)
{
	unsigned char *next = destination;
	while (bytes != 0) {
		const uint64_t count = bytes < DS41F_TEST_SPLIT_STAGING_BYTES ?
					       bytes :
					       DS41F_TEST_SPLIT_STAGING_BYTES;
		uint64_t ticket = 0;
		DS41F_TEST_SPLIT_REQUIRE(
			backend->download_begin(fixture->card, offset,
						fixture->staging, count,
						&ticket) == STATUS_OK);
		DS41F_TEST_SPLIT_REQUIRE(
			backend->upload_wait(fixture->card, ticket,
					     DS41F_TEST_SPLIT_WAIT_NS) ==
			STATUS_OK);
		memcpy(next, fixture->staging, (size_t)count);
		offset += count;
		bytes -= count;
		next += count;
	}
}

void ds41f_test_split_fill(struct ds41f_test_split_fixture *fixture,
			   struct linear_range range, unsigned char value)
{
	memset(fixture->staging, value, DS41F_TEST_SPLIT_STAGING_BYTES);
	for (uint64_t done = 0; done < range.byte_count;
	     done += DS41F_TEST_SPLIT_STAGING_BYTES) {
		const uint64_t left = range.byte_count - done;
		ds41f_test_split_upload(fixture, range.byte_offset + done,
					fixture->staging,
					left < DS41F_TEST_SPLIT_STAGING_BYTES ?
						left :
						DS41F_TEST_SPLIT_STAGING_BYTES);
	}
}

void ds41f_test_split_check_guards(struct ds41f_test_split_fixture *fixture)
{
	unsigned char actual[DS41F_TEST_SPLIT_GUARD_BYTES];
	for (uint32_t guard = 0; guard < fixture->guard_count; guard++) {
		ds41f_test_split_download(fixture,
					  fixture->guard_offsets[guard], actual,
					  sizeof(actual));
		for (uint32_t byte = 0; byte < DS41F_TEST_SPLIT_GUARD_BYTES;
		     byte++) {
			DS41F_TEST_SPLIT_REQUIRE(actual[byte] ==
						 DS41F_TEST_SPLIT_GUARD_BYTE);
		}
	}
}

void ds41f_test_split_launches(const struct ds41f_test_split_fixture *fixture,
			       uint32_t rows, uint32_t context, uint32_t ratio,
			       bool is_block, bool is_reader, uint32_t segments,
			       struct kernel_launch *legacy,
			       struct kernel_launch *split)
{
	const struct ds41f_index_select_arguments_record select = {
		fixture->step.byte_offset,
		fixture->scores.byte_offset,
		fixture->legacy.byte_offset,
		is_reader ? fixture->candidates.byte_offset :
			    DS41F_INDEX_NO_CANDIDATES,
		context,
		ratio,
		0,
		0
	};
	const struct ds41f_index_candidates_arguments candidates = {
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
		fixture->handles[is_block ? DS41F_TEST_SPLIT_LEGACY_CANDIDATES :
					    DS41F_TEST_SPLIT_LEGACY_SELECT],
		rows, 1, DS41F_INDEX_SELECT_THREADS_VALUE, 0,
		is_block ? (const void *)&candidates : (const void *)&select,
		is_block ? DS41F_INDEX_CANDIDATES_ARGUMENT_BYTES :
			   DS41F_INDEX_SELECT_ARGUMENT_BYTES);
	const struct ds41f_index_split_arguments record = {
		fixture->step.byte_offset,
		fixture->scores.byte_offset,
		fixture->split.byte_offset,
		is_reader ? fixture->candidates.byte_offset :
			    DS41F_INDEX_NO_CANDIDATES,
		fixture->scratch.byte_offset,
		context,
		ratio,
		segments,
		is_block ? 1u : 0u,
		0,
		0
	};
	for (uint32_t pass = 0; pass < DS41F_INDEX_SPLIT_LAUNCH_COUNT; pass++) {
		const bool is_cut = pass == CUT_HIGH_PASS ||
				    pass == CUT_LOW_PASS;
		prepare_launch_fill(
			&split[pass],
			fixture->handles[DS41F_TEST_SPLIT_FIRST_PASS + pass],
			rows, is_cut ? 1u : segments, DS41F_INDEX_SPLIT_THREADS,
			0, &record, DS41F_INDEX_SPLIT_ARGUMENT_BYTES);
	}
}

void ds41f_test_split_run(struct ds41f_test_split_fixture *fixture,
			  const struct kernel_launch *launches, uint32_t count)
{
	DS41F_TEST_SPLIT_REQUIRE(backend->launch_run(fixture->card, launches,
						     count) == STATUS_OK);
	DS41F_TEST_SPLIT_REQUIRE(
		backend->launch_wait(fixture->card, DS41F_TEST_SPLIT_WAIT_NS) ==
		STATUS_OK);
}
