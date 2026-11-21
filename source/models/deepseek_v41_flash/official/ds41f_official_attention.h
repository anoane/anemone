/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent release sparse core. The caller owns disjoint query, persistent
 * ring/pool, selection, sinks, rotation and output ranges from ds41f_attention_core.h.
 * One live row/session and all 64 heads are admitted; no TP, cone or block
 * window. The first token reads one window key; cached tokens read the source
 * 128-slot window and then selected compressed keys. The physical ring has
 * DS41F_RING_SLOTS slots. Output is BF16, after inverse RoPE. There is
 * no global scratch, optional observation, allocation or token-path preparation.
 * Finite source operands and exact FP16 carrier conversion are producer duties.
 * A refusal preserves launch_out. Actual source qualification remains scoped
 * to the pinned serial fixture; the independent API does not admit serving.
 */
#ifndef DS41F_OFFICIAL_ATTENTION_H
#define DS41F_OFFICIAL_ATTENTION_H
#include "models/deepseek_v41_flash/attention/ds41f_attention_core.h"
#include "ds41f_official_attention_records.h"

MUST_CHECK result_status ds41f_official_attention_kernel_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t *kernel_out);
MUST_CHECK result_status ds41f_official_attention_prepare(
	const struct ds41f_attention_core_description *description,
	uint32_t operand_types, uint64_t arena_byte_count, uint32_t kernel,
	struct kernel_launch *launch_out);
#endif
