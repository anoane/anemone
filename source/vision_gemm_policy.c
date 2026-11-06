/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Measured GEMM policy data is rendered into build/; this consumer and the product descriptors are hand-written. */
#include "vision_gemm_policy.h"

#include <stddef.h>

#include "anemone/backend_seam.h"

#define VISION_POLICY_LARGE_TILE_OPTION 1u
#define VISION_POLICY_BIAS_LAST_OPTION 2u
#define VISION_POLICY_MMA_K8_OPTION 4u
#define VISION_POLICY_K_TILE 32u
#define VISION_POLICY_LARGE_K_TILE 64u
#define VISION_POLICY_ROW_RESIDUES 8u
#define VISION_POLICY_HIDDEN_WIDTH 1024u
#define VISION_POLICY_MLP_INNER_WIDTH 2816u
#define VISION_POLICY_PROJECTOR_INNER_WIDTH 9216u
#define VISION_POLICY_LANGUAGE_WIDTH 5120u

enum policy_card_class { POLICY_CMP, POLICY_PRO_NATIVE, POLICY_CARD_CLASSES };

/* The generator emits only these numeric arrays and their residue offsets. */
struct split_boundary {
	uint16_t through;
	uint8_t parts;
	uint8_t options;
};

#include "../build/vision_gemm_policy.inc"

struct policy_table {
	const struct split_boundary *boundaries;
	size_t count;
	const uint16_t *residue_offsets;
	uint32_t minimum_rows, maximum_rows, inner_width;
};

#define PATCH_POLICY(data, inner)           \
	{ data,                             \
	  sizeof(data) / sizeof((data)[0]), \
	  NULL,                             \
	  VISION_GEMM_MINIMUM_MLP_ROWS,     \
	  VISION_GEMM_MAXIMUM_MLP_ROWS,     \
	  inner }
#define PROJECTOR_POLICY(data, offsets, inner) \
	{ data,                                \
	  sizeof(data) / sizeof((data)[0]),    \
	  offsets,                             \
	  VISION_GEMM_MINIMUM_PROJECTOR_ROWS,  \
	  VISION_GEMM_MAXIMUM_PROJECTOR_ROWS,  \
	  inner }

static const struct policy_table
	policies[POLICY_CARD_CLASSES][VISION_GEMM_PRODUCT_COUNT] = {
		[POLICY_CMP] = {
			[VISION_GEMM_MLP_DOWN] = PATCH_POLICY(cmp_mlp_down, VISION_POLICY_MLP_INNER_WIDTH),
			[VISION_GEMM_PROJECTOR_FIRST] =
				PROJECTOR_POLICY(cmp_projector_first, cmp_projector_first_offsets,
					 VISION_POLICY_PROJECTOR_INNER_WIDTH),
			[VISION_GEMM_MLP_UP] = PATCH_POLICY(cmp_mlp_up, VISION_POLICY_HIDDEN_WIDTH),
			[VISION_GEMM_QUERY_KEY_VALUE] = PATCH_POLICY(cmp_mlp_up, VISION_POLICY_HIDDEN_WIDTH),
			[VISION_GEMM_ATTENTION_OUTPUT] = PATCH_POLICY(cmp_mlp_up, VISION_POLICY_HIDDEN_WIDTH),
			[VISION_GEMM_PROJECTOR_SECOND] =
				PROJECTOR_POLICY(cmp_projector_second, NULL, VISION_POLICY_LANGUAGE_WIDTH),
		},
		[POLICY_PRO_NATIVE] = {
			[VISION_GEMM_MLP_DOWN] = PATCH_POLICY(pro_mlp_down, VISION_POLICY_MLP_INNER_WIDTH),
			[VISION_GEMM_PROJECTOR_FIRST] =
				PROJECTOR_POLICY(pro_projector_first, pro_projector_first_offsets,
					 VISION_POLICY_PROJECTOR_INNER_WIDTH),
			[VISION_GEMM_MLP_UP] = PATCH_POLICY(pro_mlp_up, VISION_POLICY_HIDDEN_WIDTH),
			[VISION_GEMM_QUERY_KEY_VALUE] =
				PATCH_POLICY(pro_query_key_value, VISION_POLICY_HIDDEN_WIDTH),
			[VISION_GEMM_ATTENTION_OUTPUT] =
				PATCH_POLICY(pro_attention_output, VISION_POLICY_HIDDEN_WIDTH),
			[VISION_GEMM_PROJECTOR_SECOND] =
				PROJECTOR_POLICY(pro_projector_second, NULL, VISION_POLICY_LANGUAGE_WIDTH),
		},
	};

#undef PATCH_POLICY
#undef PROJECTOR_POLICY

result_status
vision_gemm_policy_prepare(uint32_t arithmetic_class,
			   enum vision_gemm_product product, uint32_t rows,
			   struct vision_gemm_policy_record *policy_out)
{
	if (policy_out == NULL ||
	    (uint32_t)product >= VISION_GEMM_PRODUCT_COUNT ||
	    (arithmetic_class != ARITHMETIC_CLASS_SM_80 &&
	     arithmetic_class != ARITHMETIC_CLASS_SM_120 &&
	     arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const enum policy_card_class card_class =
		arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			POLICY_PRO_NATIVE :
			POLICY_CMP;
	const struct policy_table *const table = &policies[card_class][product];
	if (rows < table->minimum_rows || rows > table->maximum_rows) {
		return STATUS_INVALID_ARGUMENT;
	}
	size_t begin = 0, end = table->count;
	if (table->residue_offsets != NULL) {
		const uint32_t residue = rows % VISION_POLICY_ROW_RESIDUES;
		begin = table->residue_offsets[residue];
		end = table->residue_offsets[residue + 1u];
	}
	if (begin > end || end > table->count) {
		return STATUS_INVALID_ARGUMENT;
	}
	for (size_t index = begin; index < end; index++) {
		const struct split_boundary boundary = table->boundaries[index];
		if (rows > boundary.through) {
			continue;
		}
		const uint32_t parts = boundary.parts;
		const uint32_t options = boundary.options;
		if (parts == 0 ||
		    (options & ~(VISION_POLICY_LARGE_TILE_OPTION |
				 VISION_POLICY_BIAS_LAST_OPTION |
				 VISION_POLICY_MMA_K8_OPTION)) != 0) {
			return STATUS_INVALID_ARGUMENT;
		}
		const uint32_t tile =
			(options & VISION_POLICY_LARGE_TILE_OPTION) != 0 ?
				VISION_POLICY_LARGE_K_TILE :
				VISION_POLICY_K_TILE;
		const uint32_t division = parts * tile;
		const struct vision_gemm_policy_record selected = {
			.parts = parts,
			.partition_width = parts == 1 ? 0 :
							(table->inner_width +
							 division - 1u) /
								division * tile,
			.tile_k = tile,
			.bias_last =
				(options & VISION_POLICY_BIAS_LAST_OPTION) != 0,
			.mma_k8 = (options & VISION_POLICY_MMA_K8_OPTION) != 0
		};
		*policy_out = selected;
		return STATUS_OK;
	}
	return STATUS_INVALID_ARGUMENT;
}
