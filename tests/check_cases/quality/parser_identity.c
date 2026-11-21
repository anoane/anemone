/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Emit deterministic parser and formatter results for length-framed synthetic inputs. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine/server/api/chat_completions.h"
#include "engine/server/live_stats.h"
#include "common/json.h"
#include "engine/server/api/server_responses.h"

#define TEST_INPUT_BYTES 65536u
#define TEST_OUTPUT_BYTES 524288u
#define TEST_SCRATCH_BYTES (8u * TEST_INPUT_BYTES + 4096u)
#define TEST_CASES 65536u
#define TEST_COLUMNS 80u
#define TEST_LENGTH_BYTES 4u
#define TEST_BYTE_BITS 8u

static bool test_write(const void *bytes, size_t count)
{
	return count == 0 || fwrite(bytes, 1, count, stdout) == count;
}

int main(void)
{
	static char input[TEST_INPUT_BYTES], scratch_bytes[TEST_SCRATCH_BYTES],
		output[TEST_OUTPUT_BYTES];
	for (unsigned index = 0; index < TEST_CASES; index++) {
		unsigned char length[TEST_LENGTH_BYTES];
		const size_t read = fread(length, 1, sizeof(length), stdin);
		if (read == 0 && feof(stdin) != 0) {
			return 0;
		}
		if (read != sizeof(length)) {
			return 1;
		}
		size_t count = 0;
		for (size_t byte = 0; byte < sizeof(length); byte++) {
			count |= (size_t)length[byte]
				 << (byte * TEST_BYTE_BITS);
		}
		if (count > sizeof(input) ||
		    fread(input, 1, count, stdin) != count) {
			return 1;
		}
		struct json_value value = { 0 };
		const enum json_document document =
			json_read_document(input, count, &value);
		struct json_scratch scratch = { .bytes = scratch_bytes,
						.capacity =
							sizeof(scratch_bytes) };
		struct json_writer writer = { .bytes = output,
					      .capacity = sizeof(output) };
		bool repeated = false;
		if (document == JSON_DOCUMENT_VALID) {
			repeated = json_has_repeated_key(&scratch, value);
			if (!repeated && !scratch.is_short) {
				json_write_canonical(&writer, &scratch, value);
			}
		}
		const uint64_t header[] = {
			(uint64_t)document, value.begin,
			value.end,	    writer.length,
			repeated ? 1u : 0u, scratch.is_short ? 1u : 0u
		};
		if (!test_write(header, sizeof(header)) ||
		    !test_write(output, writer.length <= sizeof(output) ?
						writer.length :
						0)) {
			return 1;
		}
		struct chat_buffer text = { 0 };
		const result_status text_status = live_stats_text(
			input, count, TEST_COLUMNS, false, &text);
		const uint64_t text_head[] = { (uint64_t)text_status,
					       text_status == STATUS_OK ?
						       text.count :
						       0 };
		const bool text_written =
			test_write(text_head, sizeof(text_head)) &&
			test_write(text.bytes,
				   text_status == STATUS_OK ? text.count : 0);
		chat_buffer_free(&text);
		if (!text_written) {
			return 1;
		}
		const struct chat_request defaults = { .is_seeded = true,
						       .max_tokens = 1,
						       .top_p = 1.0f };
		struct chat_request request = { 0 };
		struct chat_refusal refusal = { 0 };
		struct chat_buffer normalized = { 0 }, fields = { 0 };
		const uint32_t status = server_responses_read_fields(
			input, count, "anemone", NULL, &defaults, &request,
			&normalized, &fields, &refusal);
		const uint64_t response_head[] = {
			status, status == 0 ? normalized.count : 0,
			status == 0 ? fields.count : 0
		};
		const bool response_written =
			test_write(response_head, sizeof(response_head)) &&
			test_write(normalized.bytes,
				   status == 0 ? normalized.count : 0) &&
			test_write(fields.bytes,
				   status == 0 ? fields.count : 0);
		chat_buffer_free(&normalized);
		chat_buffer_free(&fields);
		if (!response_written) {
			return 1;
		}
	}
	return 1;
}
