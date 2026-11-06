/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Local TP completion records. C11/CUDA; row strides count elements, slot strides count bytes. */
#ifndef TP2_JOIN_RECORDS_H
#define TP2_JOIN_RECORDS_H

#include <stdint.h>

#include "anemone/model_layout.h"
#include "anemone/step_records.h"

#define KERNEL_FILE_TP2_JOIN "kernel_tp2_join"
#define KERNEL_TP2_JOIN "tp2_join"
#define TP2_JOIN_ARGUMENT_BYTES 80u
#define TP2_JOIN_THREADS 256u
#define TP2_JOIN_BFLOAT16_TO_FLOAT32 0u
#define TP2_JOIN_FLOAT32_TO_BFLOAT16 1u
#define TP2_JOIN_BFLOAT16_COPY 2u
#define TP2_JOIN_FLOAT32_COPY 3u
/* BFLOAT16_TO_FLOAT32 then FLOAT32_TO_BFLOAT16 in one pass (tp4_v3_claude): the same add and round, no FP32 store */
#define TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16 4u
/* the copy-engine owner ring sums (tp_exchange.h): BFLOAT16_SUM_TO_BFLOAT16 into the ring's owner-major order (owner o's
 * quarter of each slab of 8 rows, slab after slab; owner after owner), and its owner-major BF16 rows back in place */
#define TP2_JOIN_RING_PACK 5u
#define TP2_JOIN_RING_UNPACK 6u
#define TP2_JOIN_RING_OWNERS 4u
#define TP2_JOIN_RING_SLAB_ROWS 8u
#define TP2_JOIN_OPERATION_COUNT 7u

struct tp2_join_arguments {
	uint64_t source_offset;
	uint64_t routed_sums_offset;
	uint64_t output_offset;
	uint64_t step_offset;
	uint64_t source_slot_stride;
	uint64_t output_slot_stride;
	uint32_t width;
	uint32_t row_count;
	uint32_t source_row_stride;
	uint32_t routed_row_stride;
	uint32_t output_row_stride;
	uint32_t operation;
	uint32_t has_routed_sums;
	uint32_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(sizeof(struct tp2_join_arguments) ==
		   TP2_JOIN_ARGUMENT_BYTES);

#endif
