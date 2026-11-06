/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_linear.h: the dense products on the cards (first_token_plan.md 2.3), each one launch of
 * dense_linear_records.h prepared at load: the head (REF:1005-1012: bfloat16 weights, float32 output,
 * DENSE_STEP_ROWS); the router's scores (REF:811) and a compressor's wkv and wgate at ratio 2 (REF:446-465):
 * float32; wkv at ratio 1 and indexer.wk (REF:518, 544): bfloat16; the index weights (REF:515, 555): bfloat16 with
 * output_scale DENSE_INDEX_WEIGHTS_SCALE; the hyper mixes (REF:952-954): float32 weights and output,
 * DENSE_ROW_SCALE, epsilon norm_eps. Contract: linear.h's for kernels_prepare and a prepare function, with
 * prepare_common.h's checks: a description that breaks a rule of dense_linear_records.h, or whose output_scale is
 * not finite and above zero or epsilon not finite and not below zero, is refused. The step area is a range only
 * with DENSE_STEP_ROWS. A description of more than LINEAR_MAXIMUM_ROWS rows, up to a chunk's
 * (MODEL_WIDE_MOST_ROWS) and without DENSE_STEP_ROWS, is a wide launch (dense_linear_records.h): the
 * bytes of the launches of its groups of 8 rows, in the wide form dense_linear_wide_kernels_prepare prepared
 * for its weight type, whose outputs a block must divide output_size (else it is refused, as it is where that form
 * is not prepared).
 */
#ifndef DENSE_LINEAR_H
#define DENSE_LINEAR_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/dense_linear_records.h"
#include "anemone/linear.h"
#include "anemone/status.h"

/* The handles of the functions of kernel_dense_linear on one card: the products, in the order of
 * DENSE_EVERY_INSTANCE, and of each weight type (enum dense_type_code) its wide form and that form's outputs
 * a block. kernels_prepare prepares the products and leaves the wide forms BACKEND_NO_KERNEL and 0 outputs.
 * wide_kernels_prepare, on a card that runs chunks, then prepares behind every other kernel, so that a thin program's
 * handles stay as they are (linear.h's wide kernels do the same), the wide form of each weight type that the card
 * tests measured fastest at 1,024 rows on the card's class (wide_small_card_test, 26 October 2026): on class 80 one
 * output a block of bfloat16 weights and 8 of float32 weights; on class 120 and 120 native (one sm_120a build), 8
 * and 4. Another class is STATUS_INVALID_ARGUMENT; a failure leaves the handles untouched. */
#define DENSE_SKINNY_MOST_OUTPUTS_VALUE \
	1024u /* a wide product of at most these outputs takes the skinny form */
#define DENSE_SKINNY_REGIMES \
	2u /* its forms: to DENSE_SKINNY_MANY_ROWS rows, then above */
#define DENSE_SKINNY_MANY_ROWS 48u

struct dense_linear_kernels {
	uint32_t products[DENSE_INSTANCES];
	uint32_t wide_products[DENSE_TYPE_COUNT];
	uint32_t wide_block_outputs[DENSE_TYPE_COUNT];
	/* a skinny matrix's wide form (outputs at most DENSE_SKINNY_MOST_OUTPUTS_VALUE): another outputs a block, the
     * same bits (dense_linear_records.h: an output's bytes are its row group's whatever the block holds) */
	uint32_t skinny_products[DENSE_TYPE_COUNT][DENSE_SKINNY_REGIMES];
	uint32_t skinny_block_outputs[DENSE_TYPE_COUNT][DENSE_SKINNY_REGIMES];
	/* --dense_staged: each type's staged wide form (dense_linear_records.h), or BACKEND_NO_KERNEL: a wide
     * product of bfloat16 weights (the float32 form is stamped for the card test), of DENSE_STAGED_LEAST_ROWS
     * rows or more without DENSE_ROW_SCALE, of at most
     * DENSE_STAGED_MOST_INPUTS inputs and outputs in whole blocks of staged_block_outputs takes it
     * before the skinny and wide forms (the same bits) */
	uint32_t staged_products[DENSE_TYPE_COUNT];
	uint32_t staged_block_outputs;
};
MUST_CHECK result_status dense_linear_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct dense_linear_kernels *kernels_out);
MUST_CHECK result_status dense_linear_wide_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct dense_linear_kernels *kernels);
/* Behind dense_linear_wide_kernels_prepare on class 80: a weight type's skinny form above
 * DENSE_SKINNY_MANY_ROWS rows at another stamped wide form's outputs a block (the same bits); 0 changes
 * nothing. Fails as kernels_prepare does, and refuses outputs no wide form of the type has. */
/* Behind dense_linear_wide_kernels_prepare: with staged 8, 16, 32 or 64 each type's staged
 * wide form; 0 changes nothing. Fails as kernels_prepare does. */
MUST_CHECK result_status dense_linear_staged_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t staged, struct dense_linear_kernels *kernels);
MUST_CHECK result_status dense_linear_skinny_kernel_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t type, uint32_t block_outputs,
	struct dense_linear_kernels *kernels);

struct dense_linear_description {
	struct linear_range
		weights; /* read: [output_size][input_size] of weight_type, exactly */
	struct linear_range rows; /* read: bfloat16 [row_count][input_size] */
	struct linear_range
		output; /* written: [row_count][output_size] of output_type */
	struct linear_range
		step; /* read with DENSE_STEP_ROWS: the step area of row_count rows;
                                                   with _CONE_ROWS their step rows (from the launch's first row) */
	uint32_t output_size;
	uint32_t input_size;
	uint32_t row_count; /* above LINEAR_MAXIMUM_ROWS: a wide launch */
	uint32_t weight_type; /* enum dense_type_code */
	uint32_t output_type;
	uint32_t flags; /* DENSE_* */
	float output_scale; /* 1 for none */
	float epsilon; /* of DENSE_ROW_SCALE */
	uint32_t cone_layer; /* DENSE_CONE_ROWS: a decoder layer; else 0 */
};
MUST_CHECK result_status dense_linear_prepare(
	const struct dense_linear_description *description,
	uint64_t arena_byte_count, const struct dense_linear_kernels *kernels,
	struct kernel_launch *launch_out);

#endif
