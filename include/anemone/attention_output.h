/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_output.h: the attention site's output span in two launches on the cards (attention_output_records.h): the
 * output groups with the quantization of what they give, and the output projection with the site's post, each
 * prepared once at load from a description that names every arena offset. They stand for four launches of the
 * separate kernels and leave every byte those leave: linear_trunk_bfloat16_prepare and
 * linear_quantize_prepare (the groups), linear_trunk_quantized_prepare of one matrix and the post of
 * hyper_site_prepare (the projection).
 *
 * Contract: linear.h's for the kernels and the prepare functions (once at load, arithmetic on a description, one error
 * line for a refusal, the launch untouched). A description is checked by the prepare functions of the separate
 * launches it stands for, with their rules and their refusals (linear_kernels: the card's linear kernels, whose
 * launches are discarded), and beyond them, with STATUS_INVALID_ARGUMENT: a NULL argument; a kernel it
 * launches not prepared; a reserved field that is not zero; a projection whose output_size is not
 * MODEL_HIDDEN, the width of a stream; a range of the post that breaks a rule of prepare_common.h for
 * row_count rows of model_layout.h's arrays; two ranges of one description that share a byte, the ranges of its
 * separate launches together. A slice count is the matrix's, as for its separate product: the bits depend on it on
 * class 120 native, on no other class. On class 120's emulation a launch takes at most
 * LINEAR_CLASS_120_EXACT_MOST_SLICES of the matrix's slices (attention_output_records.h).
 */
#ifndef ATTENTION_OUTPUT_H
#define ATTENTION_OUTPUT_H

#include <stdint.h>

#include "anemone/attention_output_records.h"
#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/status.h"

/* The handles of the two kernels on one card, one per instance in the order of LINEAR_EVERY_SLICE_COUNT, and
 * the class they were prepared for. */
struct attention_output_kernels {
	uint32_t groups[LINEAR_SLICE_INSTANCES];
	uint32_t projection[LINEAR_SLICE_INSTANCES];
	uint32_t arithmetic_class;
	uint32_t reserved_zero;
};
/* Prepares every instance on one card of arithmetic_class, as linear_kernels_prepare does (on class 120
 * native the *_NATIVE build). On failure kernels_out is untouched. */
MUST_CHECK result_status attention_output_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class,
	struct attention_output_kernels *kernels_out);

/* The output groups (attn.wo_a) of rows that are not quantized, and their quantization: one launch. */
struct attention_groups_description {
	struct linear_matrix
		matrix; /* WEIGHT_FORMAT_E4M3_SCALED, block-diagonal over segment_count */
	struct linear_range
		values; /* read: values [row_count][segment_count * input_size] */
	struct linear_range groups; /* written: values [row_count][output_size] */
	struct linear_range codes; /* written: codes [row_count][output_size] */
	struct linear_range
		exponents; /* written: exponents [row_count][output_size / 32] */
	struct linear_range
		operands; /* written: operands [row_count][output_size] */
	uint32_t row_count;
	uint32_t segment_count;
};
MUST_CHECK result_status
attention_groups_prepare(const struct attention_groups_description *description,
			 uint64_t arena_byte_count,
			 const struct linear_kernels_record *linear_kernels,
			 const struct attention_output_kernels *kernels,
			 struct kernel_launch *launch_out);

/* The output projection (attn.wo_b) of the quantized groups, and the attention site's post: one launch. */
struct attention_projection_description {
	struct linear_matrix
		matrix; /* WEIGHT_FORMAT_E4M3_SCALED, MODEL_HIDDEN outputs */
	struct linear_range codes; /* read: codes [row_count][input_size] */
	struct linear_range
		exponents; /* read: exponents [row_count][input_size / 32] */
	struct linear_range operands; /* read: operands [row_count][input_size] */
	struct linear_range
		output; /* written: values [row_count][output_size], the sublayer's output */
	struct linear_range streams; /* read and written: streams */
	struct linear_range post; /* read: post */
	struct linear_range comb; /* read: comb */
	uint32_t row_count;
	uint32_t reserved_zero;
};
MUST_CHECK result_status attention_projection_prepare(
	const struct attention_projection_description *description,
	uint64_t arena_byte_count,
	const struct linear_kernels_record *linear_kernels,
	const struct attention_output_kernels *kernels,
	struct kernel_launch *launch_out);

#endif
