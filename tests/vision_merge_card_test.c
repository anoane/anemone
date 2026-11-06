/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Deferred card test: learned spans and session/position lookup into all four Hyper-Connection streams. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "vision_merge_fixture.h"

#define MERGE_CARD_WAIT_NANOSECONDS 30000000000ull
#define MERGE_CARD_DEADLINE_SECONDS 120u

static const struct backend_operations *const backend = &backend_cuda_value;

int main(void)
{
	(void)alarm(MERGE_CARD_DEADLINE_SECONDS);
	uint32_t count = 0;
	struct card_description_record description;
	if (backend->card_count(&count) != STATUS_OK || count != 1 ||
	    backend->card_describe(0, &description) != STATUS_OK ||
	    (strstr(description.name, "170HX") == NULL &&
	     strstr(description.name, "RTX PRO 6000") == NULL)) {
		(void)puts(
			"vision merge test requires exactly one explicitly exposed CMP 170HX or RTX PRO 6000");
		return EXIT_FAILURE;
	}
	struct device_card *card = NULL;
	void *staging = NULL;
	unsigned char *const initial =
		malloc((size_t)TEST_VISION_MERGE_CARD_ARENA_BYTES);
	struct vision_kernels kernels;
	const uint32_t arithmetic_class =
		description.arithmetic_class == ARITHMETIC_CLASS_SM_120 ?
			ARITHMETIC_CLASS_SM_120_NATIVE :
			description.arithmetic_class;
	struct test_vision_merge_fixture fixture;
	uint64_t ticket = 0;
	bool passed =
		initial != NULL &&
		backend->card_open(0, TEST_VISION_MERGE_CARD_ARENA_BYTES,
				   &card) == STATUS_OK &&
		backend->staging_allocate(TEST_VISION_MERGE_CARD_ARENA_BYTES,
					  &staging) == STATUS_OK &&
		vision_kernels_prepare(backend, card, arithmetic_class,
				       &kernels) == STATUS_OK &&
		test_vision_merge_fixture_prepare(true, &kernels, &fixture);
	if (passed) {
		test_vision_merge_fixture_fill(&fixture, initial);
		memcpy(staging, initial,
		       (size_t)TEST_VISION_MERGE_CARD_ARENA_BYTES);
		passed = backend->upload_begin(
				 card, 0, staging,
				 TEST_VISION_MERGE_CARD_ARENA_BYTES,
				 &ticket) == STATUS_OK &&
			 backend->upload_wait(card, ticket,
					      MERGE_CARD_WAIT_NANOSECONDS) ==
				 STATUS_OK &&
			 backend->launch_run(card, fixture.launches,
					     fixture.launch_count) ==
				 STATUS_OK &&
			 backend->launch_wait(card,
					      MERGE_CARD_WAIT_NANOSECONDS) ==
				 STATUS_OK &&
			 backend->download_begin(
				 card, 0, staging,
				 TEST_VISION_MERGE_CARD_ARENA_BYTES,
				 &ticket) == STATUS_OK &&
			 backend->upload_wait(card, ticket,
					      MERGE_CARD_WAIT_NANOSECONDS) ==
				 STATUS_OK &&
			 test_vision_merge_fixture_matches(&fixture, staging,
							   initial);
	}
	if (card != NULL) {
		backend->card_close(card, MERGE_CARD_WAIT_NANOSECONDS);
	}
	if (staging != NULL) {
		backend->staging_free(staging);
	}
	free(initial);
	(void)printf(
		"vision learned delimiters, two images, text/inactive guards, four streams on %s: %s\n",
		description.name, passed ? "passed" : "failed");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
