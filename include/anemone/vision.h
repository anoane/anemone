/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* vision.h: bounded ViT/projector preparation. No allocation or execution. All matrices/biases are BF16,
 * RMS weights F32 or BF16 widened on read, distinguished by their exact range size. Configurations up to the
 * official 32x1024/16/2816 ViT and 5120 projector support small
 * host/card fixtures through the same passes. Every shape, arena range and writable overlap is validated.
 * Scratch is private to one encoder execution; its results and the input patches lie outside scratch.
 * Preparation leaves scalar outputs untouched on failure. A caller may discard a partially filled launch
 * buffer. Text-only callers need neither this plan nor any extra kernel handles.
 * The prepared kernels retain the caller's explicit math class: 80/120 emulation select the pinned CMP
 * serial BF16 reduction policy, native 1200 selects the pinned PRO policy. The official MLP down and first
 * projector products use measured per-row K partitions; generic small fixtures retain whole-K products.
 */
#ifndef VISION_H
#define VISION_H

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/status.h"
#include "anemone/vision_records.h"

#define VISION_PASSES_PER_LAYER 11u
#define VISION_FIXED_PASSES 6u
#define VISION_MAXIMUM_LAUNCHES \
	(VISION_MAXIMUM_LAYERS * VISION_PASSES_PER_LAYER + VISION_FIXED_PASSES)

struct vision_config {
	uint32_t layer_count;
	uint32_t width;
	uint32_t heads;
	uint32_t intermediate_width;
	uint32_t language_width;
	uint32_t patch_size;
	uint32_t downsample_ratio;
	float rope_theta;
	float epsilon;
};

struct vision_layer_weights {
	struct linear_range norm1;
	struct linear_range qkv;
	struct linear_range qkv_bias;
	struct linear_range attention_output;
	struct linear_range attention_output_bias;
	struct linear_range norm2;
	struct linear_range gate_up;
	struct linear_range down;
};

struct vision_weights {
	struct linear_range patch;
	struct linear_range patch_bias;
	struct vision_layer_weights layers[VISION_MAXIMUM_LAYERS];
	struct linear_range final_norm;
	struct linear_range projector1;
	struct linear_range projector1_bias;
	struct linear_range projector2;
	struct linear_range projector2_bias;
	struct linear_range image_start;
	struct linear_range image_newline;
	struct linear_range image_end;
};

struct vision_kernels {
	uint32_t operations[VISION_OPERATIONS];
	uint32_t arithmetic_class;
};

struct vision_scratch {
	struct linear_range hidden;
	struct linear_range normalized;
	struct linear_range qkv;
	struct linear_range attention;
	struct linear_range gates;
	struct linear_range activated;
	struct linear_range unfolded;
	struct linear_range projected;
	uint64_t byte_count;
};

struct vision_description {
	struct vision_config config;
	const struct vision_weights *weights;
	struct linear_range patches;
	struct linear_range output;
	struct linear_range scratch;
	uint32_t grid_height;
	uint32_t grid_width;
};

/* Official configuration; a value, with no side effects. */
struct vision_config vision_official_config(void);
/* Explicit class 80, 120 emulation or native 1200. Unknown classes/missing arguments are INVALID_ARGUMENT.
 * Failure leaves kernels_out untouched; success retains the class and every prepared handle together. */
MUST_CHECK result_status vision_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct vision_kernels *kernels_out);
MUST_CHECK result_status
vision_scratch_plan(const struct vision_config *config, uint32_t grid_height,
		    uint32_t grid_width, uint64_t byte_offset,
		    struct vision_scratch *scratch_out);
MUST_CHECK result_status
vision_prepare(const struct vision_description *description,
	       uint64_t arena_byte_count, const struct vision_kernels *kernels,
	       struct kernel_launch *launches_out, uint32_t launch_capacity,
	       uint32_t *launch_count_out);

/* One image's whole span. output includes only this image's rows, each replicated copies times. */
MUST_CHECK result_status vision_merge_prepare(
	const struct vision_weights *weights,
	const struct linear_range *encoded, const struct linear_range *output,
	uint32_t language_width, uint32_t grid_height, uint32_t grid_width,
	uint32_t copies, uint64_t arena_byte_count,
	const struct vision_kernels *kernels, struct kernel_launch *launch_out);

/* Per-step overwrite after ordinary token embedding. Metadata/embeddings cover all session slots. */
MUST_CHECK result_status vision_step_merge_prepare(
	struct linear_range embeddings, struct linear_range metadata,
	struct linear_range step, struct linear_range output, uint32_t rows,
	uint32_t sessions, uint64_t arena_byte_count,
	const struct vision_kernels *kernels, struct kernel_launch *launch_out);

#endif
