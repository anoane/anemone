/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent original M1 selection for at most 512 context tokens. The source
 * topk(min(512,e)) followed by index sorting keeps every visible entry. With
 * at most 64 blocks, the candidate source likewise keeps every visible block.
 * Read-only: the step block and its one checked row. Writes: selection[512],
 * plus candidates[2048] for SOURCE alone. Neither keys, queries, weights nor
 * scores are read. BF16 index operands therefore need no FP16 carrier.
 * Admission reuses the stock index description's range/count checks, then
 * restricts row/session counts to one and context to MODEL_SELECTED.
 * No allocation, preparation or dispatch runs per token. A refusal preserves
 * launch_out. Empty steps leave every output byte untouched.
 */
#ifndef OFFICIAL_INDEX_H
#define OFFICIAL_INDEX_H
#include "anemone/index_scan.h"
#include "anemone/official_index_records.h"

MUST_CHECK result_status
official_index_kernel_prepare(const struct backend_operations *backend,
			      struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
official_index_prepare(const struct index_scan_description *description,
		       uint64_t arena_byte_count, uint32_t kernel,
		       struct kernel_launch *launch_out);
#endif
