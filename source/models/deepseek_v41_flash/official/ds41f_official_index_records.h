/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* A bounded source selection: every visible entry/block is kept, so no score
 * or query/key input is read. The source owner alone writes candidate blocks. */
#ifndef DS41F_OFFICIAL_INDEX_RECORDS_H
#define DS41F_OFFICIAL_INDEX_RECORDS_H
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan_records.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "engine/program/step_records.h"

#define DS41F_MODULE_OFFICIAL_INDEX "ds41f_official_index"
#define DS41F_KERNEL_OFFICIAL_INDEX "official_index"
#define DS41F_OFFICIAL_INDEX_ARGUMENT_BYTES 32u
#define DS41F_OFFICIAL_INDEX_THREADS 256u

struct ds41f_official_index_arguments {
	uint64_t step_offset;
	uint64_t selection_offset;
	uint64_t candidates_offset;
	uint32_t context_token_limit;
	uint32_t ratio;
};
RECORD_HOLDS(sizeof(struct ds41f_official_index_arguments) ==
	     DS41F_OFFICIAL_INDEX_ARGUMENT_BYTES);
#endif
