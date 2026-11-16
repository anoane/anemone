/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Isolated private-helper regression, linked with section garbage collection. */
#include "../../../source/tp_load.c"

#define TEST_TENSOR_BYTES 8u

int main(void)
{
	const struct tp_load_job job = { .mode = TP_LOAD_FULL };
	const struct pack_tensor tensor = { .byte_count = TEST_TENSOR_BYTES };
	unsigned char source[TEST_TENSOR_BYTES] = { 0 };
	unsigned char output[TEST_TENSOR_BYTES] = { 0 };
	uint64_t padded = 1, expected = 1;
	const result_status no_output =
		job_stage(&job, &tensor, source, NULL, &padded, &expected);
	const result_status no_source =
		job_stage(&job, &tensor, NULL, output, &padded, &expected);
	return no_output == STATUS_INVALID_ARGUMENT &&
			       no_source == STATUS_INVALID_ARGUMENT &&
			       padded == 1 && expected == 1 ?
		       0 :
		       1;
}
