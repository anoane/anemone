/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * Lossless rectangular shards of lane-ordered FP8/MXFP4 and EXL3 mul1 K2/K3 matrices. Preparation and extraction
 * happen only at load, with caller-owned buffers; this API neither changes the pack nor places weights on cards.
 * Codes and scale bytes are copied without decoding or requantization. A shard retains its original FP8 scale
 * blocks: its first output must be a multiple of 32. MXFP4 output bounds are multiples of 16 and input bounds of
 * 64; FP8 output counts are multiples of 16 and input bounds of 32. These are storage constraints; a caller must
 * separately meet its kernel's input-width, group, head, reduction and numerical admission constraints.
 * EXL3 origins and extents are whole 128-channel transforms; the shared EXL3 layout helper copies trellis codes
 * and original FP16 input/output scales without decode. This does not select an EXL3 product, arithmetic order,
 * placement, backend architecture or model recipe. A plain matrix of 16-bit elements (WEIGHT_FORMAT_NONE,
 * bits_per_weight TENSOR_SHARD_PLAIN_BITS in the description) takes shards of whole rows (outputs), any
 * first output and count: one contiguous byte range.
 *
 * prepare refuses nulls, invalid source shapes, empty/out-of-bounds rectangles and broken storage alignment with
 * INVALID_ARGUMENT. An unimplemented format or an FP8 output origin inside a shared scale block is UNSUPPORTED.
 * A failed prepare leaves plan_out untouched. extract revalidates the plan and refuses tampering, nulls, a source
 * size other than the original matrix, too little destination space, pointer-range overflow and overlapping buffers
 * before copying. Destination bytes behind shard_byte_count remain untouched. Neither function allocates or calls
 * CUDA; successful extraction is not proof that an inference placement is executable or numerically qualified.
 */
#ifndef TP_WEIGHTS_H
#define TP_WEIGHTS_H

#include <stdint.h>

#include "anemone/status.h"
#include "anemone/weight_format_seam.h"

#define TENSOR_SHARD_PLAIN_BITS \
	16u /* a plain (format NONE) matrix's element bits that row shards take */

struct tensor_shard_description {
	enum weight_format_code format;
	struct matrix_shape_record source_shape;
	uint32_t first_output;
	uint32_t output_count;
	uint32_t first_input;
	uint32_t input_count;
};

struct tensor_shard_plan {
	struct tensor_shard_description description;
	struct matrix_shape_record shard_shape;
	uint64_t source_byte_count;
	uint64_t shard_byte_count;
};

MUST_CHECK result_status
tensor_shard_prepare(const struct tensor_shard_description *description,
		     struct tensor_shard_plan *plan_out);
MUST_CHECK result_status
tensor_shard_extract(const struct tensor_shard_plan *plan, const void *source,
		     uint64_t source_byte_count, void *destination,
		     uint64_t destination_byte_count);

#endif
