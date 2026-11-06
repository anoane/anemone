/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Numerical policy prepared from pinned original-reference cuBLAS launch maps.
 * The official image grid admits MLP rows >=1022 and projector rows >=164. This lookup covers those ranges
 * through the product's fixed patch/cell capacities. Smaller synthetic fixtures use their whole-K policy.
 * SM80 and SM120 emulation use the CMP map; SM120_NATIVE uses the PRO map. Unknown classes/products/rows or a
 * NULL output are INVALID_ARGUMENT without changing output. Success gives the measured serial split count
 * and a tile-rounded partition width; width zero means one whole-K product. No allocation or card query occurs.
 * tile_k is 32 unless the measured K64 tile changes the partition width. bias_last selects the reference's
 * final-partition bias epilogue; whole-K policies keep it false. mma_k8 selects the measured BF16 K8 instruction
 * family independently of partitioning. Build-generated declarations contain only measured numeric boundaries.
 * Product descriptors, class selection and lookup control remain hand-written.
 * The tables pin the measured PyTorch 2.14.1+cu130/default BF16-reduction dispatch; numerical qualification of
 * each partition policy remains separate. A different reference/library epoch requires a new measured map.
 */
#ifndef VISION_GEMM_POLICY_H
#define VISION_GEMM_POLICY_H

#include <stdbool.h>
#include <stdint.h>
#include "anemone/status.h"

#define VISION_GEMM_MINIMUM_MLP_ROWS 1022u
#define VISION_GEMM_MAXIMUM_MLP_ROWS 9216u
#define VISION_GEMM_MINIMUM_PROJECTOR_ROWS 164u
#define VISION_GEMM_MAXIMUM_PROJECTOR_ROWS 1024u

enum vision_gemm_product {
	VISION_GEMM_MLP_DOWN,
	VISION_GEMM_PROJECTOR_FIRST,
	VISION_GEMM_MLP_UP,
	VISION_GEMM_QUERY_KEY_VALUE,
	VISION_GEMM_ATTENTION_OUTPUT,
	VISION_GEMM_PROJECTOR_SECOND,
	VISION_GEMM_PRODUCT_COUNT
};

struct vision_gemm_policy_record {
	uint32_t parts;
	uint32_t partition_width;
	uint32_t tile_k;
	bool bias_last;
	bool mma_k8;
};

MUST_CHECK result_status vision_gemm_policy_prepare(
	uint32_t arithmetic_class, enum vision_gemm_product product,
	uint32_t rows, struct vision_gemm_policy_record *policy_out);

#endif
