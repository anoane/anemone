/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Score-only fixture shared by the split selection card test and timing probe. No scan or key allocation.
 * Ranges have 256-byte canaries. The two outputs each reserve 2,048 words per row; ordinary selection uses
 * its legacy packed 512-word row stride. All copies use bounded backend waits and page-locked staging.
 * The caller supplies a permitted device through CUDA_VISIBLE_DEVICES before opening this fixture. */
#ifndef DS41F_INDEX_SELECT_SPLIT_FIXTURE_H
#define DS41F_INDEX_SELECT_SPLIT_FIXTURE_H

#include <stdbool.h>
#include <stdint.h>

#include "engine/backend/backend_seam.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"

#define DS41F_TEST_SPLIT_WAIT_NS 120000000000ull
#define DS41F_TEST_SPLIT_STAGING_BYTES (4u << 20u)
#define DS41F_TEST_SPLIT_GUARD_BYTES 256u
#define DS41F_TEST_SPLIT_GUARD_BYTE 0xA5u
#define DS41F_TEST_SPLIT_GUARD_WORD 0xA5A5A5A5u
#define DS41F_TEST_SPLIT_RANGES 6u
#define DS41F_TEST_SPLIT_KERNELS 7u
#define DS41F_TEST_SPLIT_LEGACY_SELECT 0u
#define DS41F_TEST_SPLIT_LEGACY_CANDIDATES 1u
#define DS41F_TEST_SPLIT_FIRST_PASS 2u
#define DS41F_TEST_SPLIT_REQUIRE(condition)                          \
	do {                                                         \
		if (!(condition)) {                                  \
			ds41f_test_split_fail(__LINE__, #condition); \
		}                                                    \
	} while (0)

struct ds41f_test_split_fixture {
	struct device_card *card;
	unsigned char *staging;
	uint32_t handles[DS41F_TEST_SPLIT_KERNELS];
	uint64_t arena_bytes;
	uint64_t used_bytes;
	uint64_t guard_offsets[DS41F_TEST_SPLIT_RANGES];
	uint32_t guard_count;
	struct linear_range step, scores, candidates, legacy, split, scratch;
};

_Noreturn void ds41f_test_split_fail(int line, const char *message);
void ds41f_test_split_open(struct ds41f_test_split_fixture *fixture,
			   uint64_t arena_bytes, uint32_t arithmetic_class);
void ds41f_test_split_close(struct ds41f_test_split_fixture *fixture);
void ds41f_test_split_layout(struct ds41f_test_split_fixture *fixture,
			     uint32_t rows, uint32_t entries);
void ds41f_test_split_upload(struct ds41f_test_split_fixture *fixture,
			     uint64_t offset, const void *source,
			     uint64_t bytes);
void ds41f_test_split_download(struct ds41f_test_split_fixture *fixture,
			       uint64_t offset, void *destination,
			       uint64_t bytes);
void ds41f_test_split_fill(struct ds41f_test_split_fixture *fixture,
			   struct linear_range range, unsigned char value);
void ds41f_test_split_check_guards(struct ds41f_test_split_fixture *fixture);
void ds41f_test_split_launches(const struct ds41f_test_split_fixture *fixture,
			       uint32_t rows, uint32_t context, uint32_t ratio,
			       bool is_block, bool is_reader, uint32_t segments,
			       struct kernel_launch *legacy,
			       struct kernel_launch *split);
void ds41f_test_split_run(struct ds41f_test_split_fixture *fixture,
			  const struct kernel_launch *launches, uint32_t count);

#endif
