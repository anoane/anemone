/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Pinned serial M1 sparse core: K512 continuous BF16 scores, K64 online
 * probability updates, BF16 probabilities and source-order running sums.
 * Query/ring/pool words use the independently declared storage types. */
#ifndef OFFICIAL_ATTENTION_RECORDS_H
#define OFFICIAL_ATTENTION_RECORDS_H

#include "anemone/attention_core_records.h"

#define KERNEL_FILE_OFFICIAL_ATTENTION "kernel_official_attention"
#define KERNEL_OFFICIAL_ATTENTION "official_attention"
#define OFFICIAL_ATTENTION_ARGUMENT_BYTES 128u
#define OFFICIAL_ATTENTION_THREADS 256u
#define OFFICIAL_ATTENTION_SHARED_BYTES 98304u
#define OFFICIAL_ATTENTION_QUERY_BFLOAT16 1u
#define OFFICIAL_ATTENTION_RING_BFLOAT16 2u
#define OFFICIAL_ATTENTION_POOL_BFLOAT16 4u
#define OFFICIAL_ATTENTION_OPERAND_TYPES 7u

struct official_attention_arguments {
	struct attention_core_arguments_record source;
	uint32_t operand_types;
	uint32_t reserved_zero;
};
MODEL_LAYOUT_HOLDS(sizeof(struct official_attention_arguments) ==
		   OFFICIAL_ATTENTION_ARGUMENT_BYTES);
#endif
