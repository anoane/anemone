/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* A bounded source selection: every visible entry/block is kept, so no score
 * or query/key input is read. The source owner alone writes candidate blocks. */
#ifndef OFFICIAL_INDEX_RECORDS_H
#define OFFICIAL_INDEX_RECORDS_H
#include "anemone/index_scan_records.h"
#include "anemone/model_layout.h"
#include "anemone/step_records.h"

#define KERNEL_FILE_OFFICIAL_INDEX "kernel_official_index"
#define KERNEL_OFFICIAL_INDEX "official_index"
#define OFFICIAL_INDEX_ARGUMENT_BYTES 32u
#define OFFICIAL_INDEX_THREADS 256u

struct official_index_arguments {
	uint64_t step_offset;
	uint64_t selection_offset;
	uint64_t candidates_offset;
	uint32_t context_token_limit;
	uint32_t ratio;
};
MODEL_LAYOUT_HOLDS(sizeof(struct official_index_arguments) ==
		   OFFICIAL_INDEX_ARGUMENT_BYTES);
#endif
