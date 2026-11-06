/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent release sparse core. The caller owns disjoint query, persistent
 * ring/pool, selection, sinks, rotation and output ranges from attention_core.h.
 * One live row/session and all 64 heads are admitted; no TP, cone or block
 * window. The first token reads one window key; cached tokens read the source
 * 128-slot window and then selected compressed keys. The physical ring has
 * MODEL_RING_SLOTS slots. Output is BF16, after inverse RoPE. There is
 * no global scratch, optional observation, allocation or token-path preparation.
 * Finite source operands and exact FP16 carrier conversion are producer duties.
 * A refusal preserves launch_out. Actual source qualification remains scoped
 * to the pinned serial fixture; the independent API does not admit serving.
 */
#ifndef OFFICIAL_ATTENTION_H
#define OFFICIAL_ATTENTION_H
#include "anemone/attention_core.h"
#include "anemone/official_attention_records.h"

MUST_CHECK result_status official_attention_kernel_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t *kernel_out);
MUST_CHECK result_status
official_attention_prepare(const struct attention_core_description *description,
			   uint32_t operand_types, uint64_t arena_byte_count,
			   uint32_t kernel, struct kernel_launch *launch_out);
#endif
