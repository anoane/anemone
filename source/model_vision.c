/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* model_vision.c: weights already packed verbatim by the vision stage unit; shapes pinned to config.json. */
#include "model_vision.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "anemone/hooks.h"

#define VISION_NAME_CAPACITY 128
#define VISION_MATRIX_DIMENSIONS 2u
#define VISION_LAYER_WEIGHT_COUNT 8u
#define VISION_GLOBAL_WEIGHT_COUNT 10u
#define VISION_NORM_WEIGHT_ELEMENT PACK_ELEMENT_TYPE_COUNT

struct vision_tensor_rule {
	const char *name;
	uint32_t output_width;
	uint32_t input_width;
	uint32_t element_type;
};

MUST_CHECK static result_status
place(const struct pack_record *pack, const struct load_layout *layout,
      const char *prefix, struct vision_tensor_rule rule, uint32_t card,
      struct linear_range *range_out)
{
	char name[VISION_NAME_CAPACITY];
	const int count =
		snprintf(name, sizeof(name), "%s%s", prefix, rule.name);
	const uint32_t found =
		count > 0 && count < VISION_NAME_CAPACITY ?
			pack_find_tensor(pack, name, (uint32_t)count) :
			PACK_NOT_FOUND;
	if (found == PACK_NOT_FOUND) {
		LOG_ERROR("vision: missing tensor %s", name);
		return STATUS_CORRUPT_PACK;
	}
	const struct pack_tensor *const tensor = &pack_tensors_of(pack)[found];
	const bool element_matches =
		rule.element_type == VISION_NORM_WEIGHT_ELEMENT ?
			tensor->element_type == PACK_ELEMENT_FLOAT32 ||
				tensor->element_type == PACK_ELEMENT_BFLOAT16 :
			tensor->element_type == rule.element_type;
	if (!element_matches || tensor->weight_format != WEIGHT_FORMAT_NONE ||
	    tensor->bits_per_weight != 0 ||
	    tensor->dimension_count !=
		    (rule.input_width == 0 ? 1u : VISION_MATRIX_DIMENSIONS) ||
	    tensor->dimensions[0] != rule.output_width ||
	    tensor->dimensions[1] != rule.input_width ||
	    tensor->extent_index >= layout->extent_count) {
		LOG_ERROR("vision: tensor %s has an incompatible type or shape",
			  name);
		return STATUS_CORRUPT_PACK;
	}
	const uint32_t extent = tensor->extent_index;
	const uint64_t start = layout->byte_offset_of_extent[extent];
	if (card >= layout->card_count ||
	    layout->card_of_extent[extent] != card) {
		LOG_ERROR("vision: tensor %s is not on its required card %u",
			  name, card);
		return STATUS_UNSUPPORTED;
	}
	if (tensor->byte_offset_in_extent > UINT64_MAX - start) {
		return STATUS_CORRUPT_PACK;
	}
	const uint64_t offset = start + tensor->byte_offset_in_extent;
	if (offset % VISION_ALIGNMENT != 0 ||
	    offset > layout->card_byte_count[card] ||
	    tensor->byte_count > layout->card_byte_count[card] - offset) {
		return STATUS_CORRUPT_PACK;
	}
	*range_out = (struct linear_range){ .byte_offset = offset,
					    .byte_count = tensor->byte_count };
	return STATUS_OK;
}

result_status model_vision_weights(const struct pack_record *pack,
				   const struct load_layout *layout,
				   const uint32_t *layer_cards,
				   struct model_vision *vision_out)
{
	if (pack == NULL || layout == NULL || layer_cards == NULL ||
	    vision_out == NULL || layout->card_count == 0 ||
	    layout->card_count > BACKEND_MAXIMUM_CARDS ||
	    layout->byte_offset_of_extent == NULL ||
	    layout->card_of_extent == NULL ||
	    layout->extent_count != pack_header_of(pack)->extent_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct model_vision vision = { 0 };
	static const char patch_name[] = "vision.patch_embed.proj.weight";
	const uint32_t first = pack_find_tensor(
		pack, patch_name, (uint32_t)(sizeof(patch_name) - 1u));
	if (first == PACK_NOT_FOUND) {
		static const char vision_prefix[] = "vision.";
		static const char aligner_prefix[] = "aligner.";
		static const char image_prefix[] = "image_";
		const struct pack_tensor *const tensors = pack_tensors_of(pack);
		const char *const names = pack_name_bytes_of(pack);
		for (uint32_t index = 0;
		     index < pack_header_of(pack)->tensor_count; index++) {
			const char *const name =
				names + tensors[index].name_byte_offset;
			const uint32_t size = tensors[index].name_byte_count;
			if ((size >= sizeof(vision_prefix) - 1u &&
			     memcmp(name, vision_prefix,
				    sizeof(vision_prefix) - 1u) == 0) ||
			    (size >= sizeof(aligner_prefix) - 1u &&
			     memcmp(name, aligner_prefix,
				    sizeof(aligner_prefix) - 1u) == 0) ||
			    (size >= sizeof(image_prefix) - 1u &&
			     memcmp(name, image_prefix,
				    sizeof(image_prefix) - 1u) == 0)) {
				LOG_ERROR(
					"vision: a partial tower has no patch projection");
				return STATUS_CORRUPT_PACK;
			}
		}
		*vision_out = vision;
		return STATUS_OK;
	}
	const uint32_t width = VISION_MAXIMUM_WIDTH;
	const uint32_t language = VISION_MAXIMUM_LANGUAGE_WIDTH;
	const uint32_t patch_width =
		VISION_CHANNELS * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
	const uint32_t unfolded =
		width * VISION_DOWNSAMPLE_RATIO * VISION_DOWNSAMPLE_RATIO;
	const uint32_t bf16 = PACK_ELEMENT_BFLOAT16;
	const uint32_t f32 = PACK_ELEMENT_FLOAT32;
	const struct vision_tensor_rule globals[VISION_GLOBAL_WEIGHT_COUNT] = {
		{ "vision.patch_embed.proj.weight", width, patch_width, bf16 },
		{ "vision.patch_embed.proj.bias", width, 0, bf16 },
		{ "vision.norm.weight", width, 0, VISION_NORM_WEIGHT_ELEMENT },
		{ "aligner.w1.weight", language, unfolded, bf16 },
		{ "aligner.w1.bias", language, 0, bf16 },
		{ "aligner.w2.weight", language, language, bf16 },
		{ "aligner.w2.bias", language, 0, bf16 },
		{ "image_start", language, 0, bf16 },
		{ "image_newline", language, 0, bf16 },
		{ "image_end", language, 0, bf16 }
	};
	struct linear_range *const destinations[VISION_GLOBAL_WEIGHT_COUNT] = {
		&vision.weights.patch,		 &vision.weights.patch_bias,
		&vision.weights.final_norm,	 &vision.weights.projector1,
		&vision.weights.projector1_bias, &vision.weights.projector2,
		&vision.weights.projector2_bias, &vision.weights.image_start,
		&vision.weights.image_newline,	 &vision.weights.image_end
	};
	result_status status = STATUS_OK;
	for (uint32_t index = 0;
	     index < VISION_GLOBAL_WEIGHT_COUNT && status == STATUS_OK;
	     index++) {
		status = place(pack, layout, "", globals[index], 0,
			       destinations[index]);
	}
	const struct vision_tensor_rule layers[VISION_LAYER_WEIGHT_COUNT] = {
		{ "norm1.weight", width, 0, VISION_NORM_WEIGHT_ELEMENT },
		{ "attn.wqkv.weight", VISION_QKV_COMPONENTS * width, width,
		  bf16 },
		{ "attn.wqkv.bias", VISION_QKV_COMPONENTS * width, 0, bf16 },
		{ "attn.wo.weight", width, width, bf16 },
		{ "attn.wo.bias", width, 0, bf16 },
		{ "norm2.weight", width, 0, VISION_NORM_WEIGHT_ELEMENT },
		{ "mlp.w1.weight",
		  VISION_GATE_COMPONENTS * VISION_MAXIMUM_INTERMEDIATE, width,
		  bf16 },
		{ "mlp.w2.weight", width, VISION_MAXIMUM_INTERMEDIATE, bf16 }
	};
	char prefix[VISION_NAME_CAPACITY];
	for (uint32_t layer = 0;
	     layer < VISION_MAXIMUM_LAYERS && status == STATUS_OK; layer++) {
		const int count = snprintf(prefix, sizeof(prefix),
					   "vision.blocks.%u.", layer);
		if (count <= 0 || count >= VISION_NAME_CAPACITY) {
			return STATUS_INVALID_ARGUMENT;
		}
		struct vision_layer_weights *const current =
			&vision.weights.layers[layer];
		struct linear_range *const ranges[VISION_LAYER_WEIGHT_COUNT] = {
			&current->norm1,
			&current->qkv,
			&current->qkv_bias,
			&current->attention_output,
			&current->attention_output_bias,
			&current->norm2,
			&current->gate_up,
			&current->down
		};
		for (uint32_t index = 0;
		     index < VISION_LAYER_WEIGHT_COUNT && status == STATUS_OK;
		     index++) {
			status = place(pack, layout, prefix, layers[index], 0,
				       ranges[index]);
		}
	}
	for (uint32_t layer = 0; layer < MODEL_LAYERS && status == STATUS_OK;
	     layer++) {
		const int count =
			snprintf(prefix, sizeof(prefix), "layers.%u.", layer);
		if (count <= 0 || count >= VISION_NAME_CAPACITY) {
			return STATUS_INVALID_ARGUMENT;
		}
		const struct vision_tensor_rule bias = { "ffn.gate.bias_vl",
							 MODEL_EXPERTS, 0,
							 f32 };
		status = place(pack, layout, prefix, bias, layer_cards[layer],
			       &vision.router_bias[layer]);
	}
	if (status == STATUS_OK) {
		vision.enabled = true;
		*vision_out = vision;
	}
	return status;
}

/* All sizes are bounded by configuration; only the caller's starting offset can approach UINT64_MAX. */
static bool take(struct linear_range *range, uint64_t *next, uint64_t bytes)
{
	const uint64_t aligned = (bytes + VISION_ALIGNMENT - 1u) /
				 VISION_ALIGNMENT * VISION_ALIGNMENT;
	if (aligned > UINT64_MAX - *next) {
		return false;
	}
	*range = (struct linear_range){ .byte_offset = *next,
					.byte_count = bytes };
	*next += aligned;
	return true;
}

result_status model_vision_arena(struct model_vision *vision, uint32_t sessions,
				 uint64_t arena_offset, uint64_t *arena_end_out)
{
	if (vision == NULL || arena_end_out == NULL || sessions == 0 ||
	    arena_offset % VISION_ALIGNMENT != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (!vision->enabled) {
		*arena_end_out = arena_offset;
		return STATUS_OK;
	}
	struct model_vision prepared = *vision;
	prepared.text_arena_bytes = arena_offset;
	const struct vision_config config = vision_official_config();
	/* A rectangular maximum supplies the longest legal sequence and hence bounds every scratch array.
	 * 3 x 3063 patches gives 1 x 1021 cells, exactly 1024 language tokens. */
#define VISION_MAXIMUM_GRID_HEIGHT 3u
#define VISION_MAXIMUM_GRID_WIDTH 3063u
	struct vision_scratch scratch;
	result_status status = vision_scratch_plan(&config,
						   VISION_MAXIMUM_GRID_HEIGHT,
						   VISION_MAXIMUM_GRID_WIDTH,
						   arena_offset, &scratch);
	if (status != STATUS_OK) {
		return status;
	}
	uint64_t next = arena_offset;
	const uint64_t patch_bytes = (uint64_t)VISION_MAXIMUM_PATCHES *
				     VISION_CHANNELS * VISION_PATCH_SIZE *
				     VISION_PATCH_SIZE * VISION_BFLOAT16_BYTES;
	const uint64_t output_bytes = (uint64_t)VISION_MAXIMUM_TOKENS *
				      VISION_MAXIMUM_LANGUAGE_WIDTH *
				      VISION_BFLOAT16_BYTES;
	prepared.session_embedding_stride =
		output_bytes * VISION_MAXIMUM_IMAGES;
	if (!take(&prepared.scratch, &next, scratch.byte_count) ||
	    !take(&prepared.patches, &next, patch_bytes) ||
	    !take(&prepared.encoded, &next, output_bytes) ||
	    !take(&prepared.session_embeddings, &next,
		  prepared.session_embedding_stride * sessions) ||
	    !take(&prepared.session_metadata, &next,
		  (uint64_t)sessions * VISION_SESSION_METADATA_BYTES)) {
		return STATUS_INVALID_ARGUMENT;
	}
	*vision = prepared;
	*arena_end_out = next;
	return STATUS_OK;
}
