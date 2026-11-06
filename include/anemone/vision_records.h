/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* vision_records.h: the ViT passes, C11 and CUDA. Every value array is row-major bfloat16, every matrix
 * [output][input]; RMS weights are float32 or packed bfloat16 widened to float32. Outputs round to nearest even
 * after every reference operation. QKV is [patch][Q,K,V][head][channel]. RoPE uses height then width frequencies
 * and split halves.
 * Attention visits 128-key tiles from last to first, with online float32 maxima, four denominator partials,
 * base-two exponentials and BF16 unnormalized probabilities for each value product. Accumulators rescale when
 * the maximum changes; four partials reduce only at final normalization and the output rounds to BF16.
 * UNFOLD pads bottom/right with zero and writes [cell][channel][dy][dx], as torch.nn.functional.unfold.
 * ADD, ROPE and GELU allow their documented in-place destination; other write ranges must be disjoint.
 * LINEAR uses native BF16 tensor products and FP32 accumulators. The K-partition field in flags is zero for
 * one complete product, or a multiple of 16 inputs: each partition starts fresh FP32 accumulators, adds the
 * preceding BF16 result, and rounds to BF16. A bias is added only in the first partition unless BIAS_LAST is
 * set: then the final partition adds it to its FP32 accumulator before the carried BF16 value and final rounding.
 * BIAS_LAST has no effect without BIAS. MMA_K8 selects two ascending K8 BF16 products per loaded K16 tile;
 * the default is one native K16 product. Both use FP32 accumulators and the same BF16 partition boundary.
 * A launch uses 256 threads. LINEAR owns 16 by 16 row/output tiles in grid.y/grid.x; element passes flatten grid.x.
 * RMS uses one block per row; ATTENTION owns 16 queries of one head per block. RoPE uses bounded FP64 polynomials
 * and packed FP32 trig corrections. Exact card/library agreement is qualified separately from this layout.
 * MERGE uses grid_height/grid_width of the language grid and writes START, cells/newlines, END, repeating
 * each row copies times (Hyper-Connections). source is the cell grid; weights/bias/auxiliary the delimiters.
 */
#ifndef VISION_RECORDS_H
#define VISION_RECORDS_H

#include <stdint.h>

#define VISION_THREADS 256u
#define VISION_LINEAR_TILE 16u
#define VISION_ATTENTION_KEY_TILE 128u
#define VISION_ATTENTION_QUERY_TILE 16u
#define VISION_ATTENTION_SUM_LANES 4u
#define VISION_ATTENTION_COLUMNS_PER_LANE 2u
#define VISION_ARGUMENT_BYTES 80u
#define VISION_MAXIMUM_LAYERS 32u
#define VISION_MAXIMUM_WIDTH 1024u
#define VISION_MAXIMUM_INTERMEDIATE 2816u
#define VISION_MAXIMUM_LANGUAGE_WIDTH 5120u
#define VISION_MAXIMUM_PATCHES 9216u
#define VISION_MAXIMUM_TOKENS 1024u
#define VISION_PATCH_SIZE 14u
#define VISION_DOWNSAMPLE_RATIO 3u
#define VISION_HEADS 16u
#define VISION_ROPE_THETA 10000.0f
#define VISION_NORM_EPSILON 1e-6f
#define VISION_CHANNELS 3u
#define VISION_QKV_COMPONENTS 3u
#define VISION_GATE_COMPONENTS 2u
#define VISION_HALF_DIVISOR 2u
#define VISION_ROPE_AXES 2u
#define VISION_ROPE_PARTS 4u
#define VISION_ALIGNMENT 256u
#define VISION_BFLOAT16_BYTES 2u
#define VISION_FLOAT_BYTES 4u
#define VISION_LINEAR_BIAS 1u
#define VISION_LINEAR_BIAS_LAST 2u
#define VISION_LINEAR_MMA_K8 4u
#define VISION_LINEAR_K_PARTITION_SHIFT 8u
#define VISION_LINEAR_K_PARTITION_MASK 0x3FFF00u
#define VISION_NORM_BFLOAT16_WEIGHT 2u
#define VISION_MAXIMUM_COPIES 4u
#define VISION_MAXIMUM_IMAGES 8u
#define VISION_SESSION_METADATA_BYTES 256u
#define VISION_SPAN_RESERVED_WORDS 3u
#define VISION_KERNEL_FILE "kernel_vision"

enum vision_operation {
	VISION_LINEAR,
	VISION_RMS,
	VISION_ROPE,
	VISION_ATTENTION,
	VISION_ADD,
	VISION_SILU,
	VISION_UNFOLD,
	VISION_GELU,
	VISION_MERGE,
	VISION_STEP_MERGE,
	VISION_OPERATIONS
};

struct vision_span {
	uint32_t token_begin;
	uint32_t token_count;
	uint32_t embedding_row;
	uint32_t reserved_zero;
};

/* One entry per engine slot, at SESSION_METADATA_BYTES stride; unused bytes stay zero. */
struct vision_session {
	uint32_t image_count;
	uint32_t reserved_zero[VISION_SPAN_RESERVED_WORDS];
	struct vision_span spans[VISION_MAXIMUM_IMAGES];
};

struct vision_arguments {
	uint64_t source_offset;
	uint64_t weights_offset;
	uint64_t bias_offset;
	uint64_t destination_offset;
	uint64_t auxiliary_offset;
	uint32_t rows;
	uint32_t input_width;
	uint32_t output_width;
	uint32_t grid_height;
	uint32_t grid_width;
	uint32_t heads;
	uint32_t flags;
	uint32_t copies;
	float theta;
	float epsilon;
};

#ifdef __cplusplus
static_assert(sizeof(struct vision_arguments) == VISION_ARGUMENT_BYTES,
	      "vision record");
#else
_Static_assert(sizeof(struct vision_arguments) == VISION_ARGUMENT_BYTES,
	       "vision record");
#endif

#endif
