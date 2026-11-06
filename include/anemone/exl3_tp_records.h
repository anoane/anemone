/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent activation adapter; original EXL3 arithmetic and job ABI. */
#ifndef BB_EXL3_SHARD_RECORDS_H
#define BB_EXL3_SHARD_RECORDS_H

#include "anemone/exl3_records.h"

#define BB_EXL3_SHARD_ACTIVATION_BYTES 96u
#define BB_EXL3_SHARD_KERNEL_FILE EXL3_KERNEL_FILE
#define BB_EXL3_SHARD_ACTIVATION_KERNEL "bb_exl3_activate_output_shard"

/* base describes the local gate/up output shard. Each job's down_offset names
 * its resident full-K down output shard, with the following actual dimensions.
 * The activation reads suh[down_input_begin : +base.inner_size] from that matrix.
 * Other base fields and all floating operations keep the original activation
 * contract. No pointer/address, allocator or dispatch is embedded in the record. */
struct bb_exl3_shard_activation_record {
	struct exl3_expert_record base;
	uint32_t down_input_size;
	uint32_t down_output_size;
	uint32_t down_input_begin;
	uint32_t reserved_zero;
};

#ifdef __cplusplus
static_assert(sizeof(struct bb_exl3_shard_activation_record) ==
		      BB_EXL3_SHARD_ACTIVATION_BYTES,
	      "EXL3 output shard activation ABI");
#else
_Static_assert(sizeof(struct bb_exl3_shard_activation_record) ==
		       BB_EXL3_SHARD_ACTIVATION_BYTES,
	       "EXL3 output shard activation ABI");
#endif

#endif
