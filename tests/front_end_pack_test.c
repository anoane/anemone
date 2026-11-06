/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Byte-binding/lifecycle tests only. The small pack is deliberately not a model or an inference stand-in. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "front_end_pack.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "pack_builder.h"

#define TABLE_NAME "front_end.tables"
#define OTHER_NAME "other_end.tables"
#define TEST_MODEL_NAME "byte-binding-only"
#define TEST_CARD_BYTES 65536ull
#define TEST_WAIT_NANOSECONDS 1000000000ull
#define TABLE_SIDE 64u
#define TWO_DIMENSIONS 2u

static unsigned failures;

static void expect(result_status actual, result_status expected,
		   const char *label)
{
	if (actual != expected) {
		(void)fprintf(stderr, "%s: %s\n", label, status_text(actual));
		failures++;
	}
}

static void test_binding(bool present, uint32_t element_type, bool matrix_shape)
{
	const struct test_pack_unit_plan unit = { PLACEMENT_UNIT_HOST_ONLY,
						  PACK_NO_LAYER, 1, 1 };
	const struct test_pack_plan plan = { TEST_MODEL_NAME, &unit, 1, 1, 1 };
	const char *const paths[] = { "in-memory byte-contract fixture" };
	const uint32_t placement[] = { PLACEMENT_IN_HOST_MEMORY };
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct load_layout *layout = NULL;
	struct load_result loaded = { 0 };
	struct device_card *cards[] = { NULL };
	const struct card_description_record descriptions[] = {
		{ .arithmetic_class = 0 }
	};
	void *bytes = NULL;
	uint64_t byte_count = 0;
	unsigned char external[PACK_BLOCK_BYTES];
	result_status status = test_pack_build(&plan, &bytes, &byte_count);
	if (status == STATUS_OK) {
		struct pack_header *const header = bytes;
		struct pack_extent *const extents =
			(void *)((unsigned char *)bytes + PACK_BLOCK_BYTES);
		struct pack_unit_record *const units =
			(void *)(extents + header->extent_count);
		struct pack_tensor *const tensor =
			(void *)(units + header->unit_count);
		char *const names = (void *)(tensor + header->tensor_count);
		const char *const name = present ? TABLE_NAME : OTHER_NAME;
		memcpy(names + tensor->name_byte_offset, name,
		       sizeof(TABLE_NAME) - 1);
		tensor->name_byte_count = sizeof(TABLE_NAME) - 1;
		tensor->element_type = element_type;
		if (matrix_shape) {
			tensor->dimension_count = TWO_DIMENSIONS;
			tensor->dimensions[0] = TABLE_SIDE;
			tensor->dimensions[1] = TABLE_SIDE;
		}
		test_pack_reseal(bytes);
		test_memory_storage_reset();
		test_memory_storage_set_pack(bytes, byte_count);
		test_host_memory_backend_reset(1, TEST_CARD_BYTES);
		status = test_memory_storage.reader_open(paths, 1, &reader);
	}
	status = status != STATUS_OK ?
			 status :
			 pack_open_entry(&test_memory_storage, reader, &pack);
	status = status != STATUS_OK ?
			 status :
			 load_layout_create(pack, placement, 1, 1, &layout);
	if (status == STATUS_OK) {
		const struct load_request request = {
			.storage = &test_memory_storage,
			.reader = reader,
			.backend = &test_host_memory_backend,
			.pack = pack,
			.layout = layout,
			.cards = cards,
			.card_descriptions = descriptions,
			.reader_threads_per_copy = 1,
			.stall_limit_nanoseconds = TEST_WAIT_NANOSECONDS
		};
		status = load_run_entry(&request, &loaded);
	}
	expect(status, STATUS_OK,
	       "fixture uses the actual pack reader and digest-checking loader");
	if (status == STATUS_OK) {
		memcpy(external, loaded.host_arena, sizeof(external));
		const result_status expected =
			!present ? STATUS_UNSUPPORTED :
			element_type != PACK_ELEMENT_UINT8 || matrix_shape ?
				   STATUS_CORRUPT_PACK :
				   STATUS_OK;
		expect(front_end_pack_match(pack, layout, &loaded, external,
					    sizeof(external)),
		       expected,
		       "absence and valid-but-incompatible metadata are refused");
		if (expected == STATUS_OK) {
			for (size_t position = 0; position < sizeof(external);
			     position++) {
				external[position] ^= 1;
				expect(front_end_pack_match(pack, layout,
							    &loaded, external,
							    sizeof(external)),
				       STATUS_INVALID_ARGUMENT,
				       "every external byte is bound");
				external[position] ^= 1;
			}
			struct load_layout wrong_layout = *layout;
			struct load_result wrong_loaded = loaded;
			uint64_t bad_offset = UINT64_MAX;
			uint32_t bad_card = 0;
			expect(front_end_pack_match(pack, layout, &loaded,
						    external,
						    sizeof(external) - 1),
			       STATUS_INVALID_ARGUMENT, "length mismatch");
			wrong_layout.byte_offset_of_extent = &bad_offset;
			expect(front_end_pack_match(pack, &wrong_layout,
						    &loaded, external,
						    sizeof(external)),
			       STATUS_CORRUPT_PACK,
			       "overflowing host offset is never dereferenced");
			wrong_layout = *layout;
			wrong_layout.card_of_extent = &bad_card;
			expect(front_end_pack_match(pack, &wrong_layout,
						    &loaded, external,
						    sizeof(external)),
			       STATUS_CORRUPT_PACK,
			       "device placement is never dereferenced as host");
			wrong_layout = *layout;
			wrong_layout.host_byte_count--;
			expect(front_end_pack_match(pack, &wrong_layout,
						    &loaded, external,
						    sizeof(external)),
			       STATUS_CORRUPT_PACK, "short loaded extent");
			wrong_loaded.host_arena = NULL;
			expect(front_end_pack_match(pack, layout, &wrong_loaded,
						    external, sizeof(external)),
			       STATUS_CORRUPT_PACK, "missing loaded owner");
			expect(front_end_pack_match(NULL, layout, &loaded,
						    external, sizeof(external)),
			       STATUS_INVALID_ARGUMENT, "missing pack");
			expect(front_end_pack_match(pack, layout, &loaded,
						    external, sizeof(external)),
			       STATUS_OK, "restored external bytes bind again");
		}
	}
	if (layout != NULL) {
		load_host_arena_release(loaded.host_arena,
					layout->host_byte_count);
	}
	load_layout_destroy(layout);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	test_memory_storage_reset();
	free(bytes);
}

int main(void)
{
	test_binding(true, PACK_ELEMENT_UINT8, false);
	test_binding(false, PACK_ELEMENT_UINT8, false);
	test_binding(true, PACK_ELEMENT_INT8, false);
	test_binding(true, PACK_ELEMENT_UINT8, true);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
