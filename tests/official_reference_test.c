/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Every handle is prepared before publication, including success-without-handle backend failures. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "anemone/official_reference.h"

#define SENTINEL_BYTE 0xA5
#define INVALID_BACKEND_MODE 2u
static uint32_t calls, failure_at, failure_mode;
static int failures;

MUST_CHECK static result_status prepare(struct device_card *card,
					const char *file, const char *name,
					uint32_t bytes, uint32_t *kernel_out)
{
	(void)card;
	(void)file;
	(void)name;
	(void)bytes;
	const uint32_t call = calls++;
	if (call == failure_at) {
		if (failure_mode == 1)
			return STATUS_DEVICE_ERROR;
		return STATUS_OK;
	}
	*kernel_out = call;
	return STATUS_OK;
}

static void expect(bool value, const char *message)
{
	if (!value) {
		(void)printf("official_reference_test: %s\n", message);
		failures++;
	}
}

int main(void)
{
	const struct backend_operations backend = { .kernel_prepare = prepare };
	struct device_card *const card = (struct device_card *)(uintptr_t)1u;
	struct official_reference_kernels result, before;
	memset(&before, SENTINEL_BYTE, sizeof(before));
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120_NATIVE };
	for (uint32_t architecture = 0;
	     architecture < sizeof(classes) / sizeof(classes[0]);
	     architecture++) {
		calls = 0;
		failure_at = UINT32_MAX;
		result = before;
		expect(official_reference_kernels_prepare(&backend, card,
							  classes[architecture],
							  &result) == STATUS_OK,
		       "valid handles are published");
		const uint32_t count = calls;
		expect(result.linear.quantized == 0 &&
			       result.final_kept != BACKEND_NO_KERNEL,
		       "handle zero is valid and final collapse is explicitly prepared");
		for (uint32_t mode = 0; mode < INVALID_BACKEND_MODE; mode++) {
			for (uint32_t index = 0; index < count; index++) {
				calls = 0;
				failure_at = index;
				failure_mode = mode;
				result = before;
				expect(official_reference_kernels_prepare(
					       &backend, card,
					       classes[architecture],
					       &result) != STATUS_OK,
				       "an error or missing handle is refused");
				expect(memcmp(&result, &before,
					      sizeof(result)) == 0,
				       "refusal preserves the entire output bundle");
			}
		}
	}
	(void)printf("official_reference_test: %d failures\n", failures);
	return failures != 0;
}
