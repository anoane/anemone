/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Overflow boundaries and malformed load metadata, without cards or model data. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "checked_size.h"
#include "json.h"
#include "step_driver.h"

#define TEST_MULTIPLIER 3u
#define TEST_BYTES 8u
#define TEST_PIPELINE_DEPTH 2u

static unsigned failures;
static unsigned staging_calls;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: quality safety: %s\n", message);
		failures++;
	}
}

MUST_CHECK static result_status test_staging_allocate(uint64_t bytes,
						      void **output)
{
	(void)bytes;
	(void)output;
	staging_calls++;
	return STATUS_OUT_OF_MEMORY;
}

static void test_sizes(void)
{
	expect(size_add(SIZE_MAX - TEST_BYTES, TEST_BYTES - 1u) ==
		       SIZE_MAX - 1u,
	       "largest representable valid sum");
	expect(size_add(SIZE_MAX - TEST_BYTES, TEST_BYTES + 1u) == SIZE_MAX,
	       "addition overflow is invalid");
	expect(size_multiply((SIZE_MAX - 1u) / TEST_MULTIPLIER,
			     TEST_MULTIPLIER) == SIZE_MAX - TEST_MULTIPLIER,
	       "bounded product accepted");
	expect(size_multiply(SIZE_MAX / TEST_MULTIPLIER + 1u,
			     TEST_MULTIPLIER) == SIZE_MAX,
	       "product overflow is invalid");
	expect(size_multiply(SIZE_MAX, 0) == SIZE_MAX,
	       "invalid metadata stays invalid across multiplication");
	expect(size_round_up(TEST_BYTES + 1u, TEST_BYTES) ==
		       TEST_BYTES + TEST_BYTES,
	       "alignment rounds to the next block");
	expect(size_round_up(SIZE_MAX, TEST_BYTES) == SIZE_MAX &&
		       size_round_up(1, 0) == SIZE_MAX,
	       "alignment overflow and zero alignment are invalid");
}

static void test_writer_overflow(void)
{
	const char bytes[TEST_BYTES] = { 0 };
	struct json_writer writer = { .length = SIZE_MAX - 1u };
	json_write_bytes(&writer, bytes, sizeof(bytes));
	expect(writer.length == SIZE_MAX,
	       "writer overflow saturates before an invalid length can wrap");
	json_write_bytes(&writer, bytes, 1);
	expect(writer.length == SIZE_MAX,
	       "overflow remains invalid after another write");
}

static void test_empty_containers(void)
{
	static const char *const accepted[] = { "{}", "[]", "{\"a\":{}}",
						"[[],{}]", "{\"a\":[{},[]]}" };
	static const char *const refused[] = { "{}}", "[]]", "[{}",
					       "{\"a\":[]" };
	for (size_t index = 0; index < sizeof(accepted) / sizeof(accepted[0]);
	     index++) {
		struct json_value value = { 0 };
		expect(json_read_document(accepted[index],
					  strlen(accepted[index]),
					  &value) == JSON_DOCUMENT_VALID,
		       "empty containers consume exactly their own closing byte");
	}
	for (size_t index = 0; index < sizeof(refused) / sizeof(refused[0]);
	     index++) {
		struct json_value value = { 0 };
		expect(json_read_document(refused[index],
					  strlen(refused[index]),
					  &value) == JSON_DOCUMENT_NOT_JSON,
		       "extra or missing closing byte is refused");
	}
}

static void test_pipeline_size(void)
{
	const struct backend_operations backend = {
		.staging_allocate = test_staging_allocate, .staging_free = free
	};
	const struct execution_program program = {
		.kind = PROGRAM_THIN, .limits = { .session_capacity = 1 }
	};
	const struct execution_program *programs[] = { &program };
	struct step_driver driver = { .backend = &backend,
				      .programs = programs,
				      .largest_row_count = 1,
				      .staging_bytes = SIZE_MAX };
	const result_status status =
		step_pipeline_start(&driver, TEST_PIPELINE_DEPTH);
	expect(status == STATUS_INVALID_ARGUMENT && staging_calls == 0 &&
		       driver.pipeline == NULL,
	       "pipeline alignment wrap is refused before allocation or publication");
}

int main(void)
{
	test_sizes();
	test_writer_overflow();
	test_empty_containers();
	test_pipeline_size();
	return failures != 0 ? 1 : 0;
}
