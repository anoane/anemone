/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Pinned serial M1 sparse core: K512 continuous BF16 scores, K64 online
 * probability updates, BF16 probabilities and source-order running sums.
 * Query/ring/pool words use the independently declared storage types. */
#ifndef DS41F_OFFICIAL_ATTENTION_RECORDS_H
#define DS41F_OFFICIAL_ATTENTION_RECORDS_H

#include "models/deepseek_v41_flash/attention/ds41f_attention_core_records.h"

#define DS41F_MODULE_OFFICIAL_ATTENTION "ds41f_official_attention"
#define DS41F_KERNEL_OFFICIAL_ATTENTION "official_attention"
#define DS41F_OFFICIAL_ATTENTION_ARGUMENT_BYTES 128u
#define DS41F_OFFICIAL_ATTENTION_THREADS 256u
#define DS41F_OFFICIAL_ATTENTION_SHARED_BYTES 98304u
#define DS41F_OFFICIAL_ATTENTION_QUERY_BFLOAT16 1u
#define DS41F_OFFICIAL_ATTENTION_RING_BFLOAT16 2u
#define DS41F_OFFICIAL_ATTENTION_POOL_BFLOAT16 4u
#define DS41F_OFFICIAL_ATTENTION_OPERAND_TYPES 7u

struct ds41f_official_attention_arguments {
	struct ds41f_attention_core_arguments_record source;
	uint32_t operand_types;
	uint32_t reserved_zero;
};
RECORD_HOLDS(sizeof(struct ds41f_official_attention_arguments) ==
	     DS41F_OFFICIAL_ATTENTION_ARGUMENT_BYTES);
#endif
