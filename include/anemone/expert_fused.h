/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_fused.h: the routed experts' phase in fewer launches. The producer is one launch in place of the first two of
 * the routed experts (gate with up, then the quantization of the hidden vectors); the join one launch in place of
 * their row sum and the FFN site's post; the shared join one launch in place of the shared expert's down, the row sum
 * and the post. The kernels and their arithmetic: expert_fused_records.h.
 *
 * Contract: linear.h's for kernels, prepare functions, ranges and refusals (one error line,
 * STATUS_INVALID_ARGUMENT, the launch untouched); hyper_connection.h's for the streams of a site.
 *   - expert_fused_kernels_prepare asks a card for every kernel here, once: the producer's instances from the
 *     module of the routed experts and the shared join's from that of the trunk (on a card of
 *     ARITHMETIC_CLASS_SM_120_NATIVE their native builds), the join from the module of the hyper-connection
 *     sites.
 *   - A description that names launches of linear.h is first handed to their prepare functions (linear_kernels: the
 *     card's linear kernels, whose launches are discarded): what they refuse is refused here, behind their one error
 *     line. What the fusion adds is checked beyond them, and every range against every other.
 *   - The producer takes the description of the routed experts (linear.h, linear_routed_experts_prepare's
 *     rules) and the range of its arrival counters (uint32 [job_count][gate_up_shape.output_size / 32]), which must
 *     hold zeros before its first launch: each launch leaves them so. It stands for launches
 *     LINEAR_ROUTED_EXPERTS_GATE_UP and _QUANTIZE: it reads the jobs, the experts' matrices, the rows' arrays
 *     and the hidden values of every pair, and writes hidden (of the pairs that stand in a job, as gate with up does),
 *     hidden_codes, hidden_exponents and hidden_operands (of every pair, as the quantizer does); it neither reads nor
 *     writes pair_output and row_output. It refuses also a row_count above LINEAR_MAXIMUM_ROWS, more pairs
 *     than EXPERT_PRODUCER_MOST_PAIRS and an output_size that is no multiple of 32.
 *   - The join stands for launch LINEAR_ROUTED_EXPERTS_ROW_SUM and the FFN site's post (HYPER_SITE_POST
 *     with routed sums); the shared join also for the shared expert's down (LINEAR_SHARED_EXPERT_DOWN,
 *     linear_trunk_quantized_prepare's rules for its matrix and rows), whose matrix has MODEL_HIDDEN
 *     outputs. Both take 1 to LINEAR_MAXIMUM_ROWS rows, as the post does, and a pair or more a row. They
 *     write the row sums into row_sums where its byte_count is not 0 (as the row sum writes them: a capture of the
 *     layer reads them, tests/program_view.h); a row_sums of no bytes is not written, and its offset is not looked at.
 */
#ifndef EXPERT_FUSED_H
#define EXPERT_FUSED_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/expert_fused_records.h"
#include "anemone/linear.h"
#include "anemone/status.h"

/* The handles of the kernels here on one card: of the producer and the shared join one per count of
 * LINEAR_EVERY_SLICE_COUNT, in its order; then the join. */
struct expert_fused_kernels {
	uint32_t gate_up_quantize[LINEAR_SLICE_INSTANCES];
	uint32_t shared_join[LINEAR_SLICE_INSTANCES];
	uint32_t join;
};

/* First user: the model's load, once for each open card (and the card test). On failure kernels_out is untouched. */
MUST_CHECK result_status expert_fused_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct expert_fused_kernels *kernels_out);

/* One launch of expert_gate_up_quantize_<gate_up_slice_count>. */
MUST_CHECK result_status expert_producer_prepare(
	const struct linear_routed_experts_description *description,
	struct linear_range arrivals, uint64_t arena_byte_count,
	const struct linear_kernels_record *linear_kernels,
	const struct expert_fused_kernels *kernels,
	struct kernel_launch *launch_out);

/* A layer's FFN join for the rows of a thin step, or of a slab of a wide one (its rows' pairs from its first). */
struct expert_join_description {
	struct linear_range
		pair_output; /* read: values [row_count * pairs_per_row][MODEL_HIDDEN] */
	struct linear_range
		shared_output; /* read: values [row_count][MODEL_HIDDEN] */
	struct linear_range
		streams; /* read and written: bfloat16 [row_count][4][MODEL_HIDDEN] */
	struct linear_range post; /* read: the site's float32 [row_count][4] */
	struct linear_range comb; /* read: the site's float32 [row_count][4][4] */
	struct linear_range
		row_sums; /* written: float32 [row_count][MODEL_HIDDEN], or no bytes */
	uint32_t pairs_per_row;
	uint32_t row_count;
};
MUST_CHECK result_status expert_join_prepare(
	const struct expert_join_description *description,
	uint64_t arena_byte_count, const struct expert_fused_kernels *kernels,
	struct kernel_launch *launch_out);

/* The same join behind the shared expert's down: down's matrix and quantized rows as the shared expert's
 * description gives them (linear.h); its output is written to join.shared_output, which is not read before. */
struct expert_shared_join_description {
	struct linear_matrix
		down; /* WEIGHT_FORMAT_E4M3_SCALED, MODEL_HIDDEN outputs */
	struct linear_range
		codes; /* read: codes [row_count][down.shape.input_size] */
	struct linear_range
		exponents; /* read: exponents [row_count][down.shape.input_size / 32] */
	struct linear_range
		operands; /* read: operands [row_count][down.shape.input_size] */
	struct expert_join_description join;
};
MUST_CHECK result_status expert_shared_join_prepare(
	const struct expert_shared_join_description *description,
	uint64_t arena_byte_count,
	const struct linear_kernels_record *linear_kernels,
	const struct expert_fused_kernels *kernels,
	struct kernel_launch *launch_out);

#endif
