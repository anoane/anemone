/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* exl3_tc.h: the five launches of one EXL3 layer's routed experts on the tensor cores (exl3_tc_records.h,
 * --exl3_tc_moe). Load-time preparation: no allocation or dispatch per token. The caller follows them with its
 * collective of the record's FP32 output (the BF16 recipe's ring with the shared expert). A refusal leaves the launch
 * table unchanged. */
#ifndef EXL3_TC_H
#define EXL3_TC_H

#include "anemone/backend_seam.h"
#include "anemone/exl3_tc_records.h"

MUST_CHECK result_status exl3_tc_prepare(
	const struct exl3_tc_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TC_LAUNCHES]);

/* exl3_tc_prepare with the short forms (16 gate/up slices, 2 down slices) at every row count: a row's sums
 * those of a call of fewer than EXL3_TC_LONG_PAIRS pairs (a thin step's), bit for bit (batch_thin_bits). */
MUST_CHECK result_status exl3_tc_prepare_short(
	const struct exl3_tc_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TC_LAUNCHES]);
/* exl3_tc_prepare_short with the gate/up and down products two adjacent slices a warp item (no rotation): the
 * short forms' partials bit for bit, half the items. */
MUST_CHECK result_status exl3_tc_prepare_paired(
	const struct exl3_tc_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TC_LAUNCHES]);

/* The prefill form's launches of one call (row_count 65 to 2,048): fills launches_out and *launch_count_out (6 with
 * the joint gate/up operands and GEMM, 8 with gate then up). */
MUST_CHECK result_status exl3_tcp_prepare(
	const struct exl3_tcp_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TCP_MOST_LAUNCHES],
	uint32_t *launch_count_out);
/* exl3_tcp_prepare (which takes h640 for 640-column calls up to EXL3_TCP_JOINT_ROWS rows) with a
 * 640-column record's gate/up GEMMs always by the i640 form (_base) or always by the hybrid h640 form (_hybrid): the
 * same bits (exl3_tcp_forms_card_test); other records as exl3_tcp_prepare. */
MUST_CHECK result_status exl3_tcp_prepare_base(
	const struct exl3_tcp_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TCP_MOST_LAUNCHES],
	uint32_t *launch_count_out);
MUST_CHECK result_status exl3_tcp_prepare_hybrid(
	const struct exl3_tcp_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TCP_MOST_LAUNCHES],
	uint32_t *launch_count_out);

#endif
