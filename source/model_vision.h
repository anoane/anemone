/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* model_vision.h: optional, complete official tower on card 0. Absence preserves legacy text-only arenas.
 * Partial towers, wrong tensor types/shapes and a split placement are refused before opening a card.
 * Scratch is one serialized encoder; output spans/metadata are private to each session and snapshot identity.
 */
#ifndef MODEL_VISION_H
#define MODEL_VISION_H

#include <stdbool.h>

#include "anemone/loader.h"
#include "anemone/model_layout.h"
#include "anemone/vision.h"

struct model_vision {
	struct vision_weights weights;
	struct linear_range router_bias[MODEL_LAYERS];
	struct linear_range patches;
	struct linear_range scratch;
	struct linear_range encoded;
	struct linear_range session_embeddings;
	struct linear_range session_metadata;
	uint64_t session_embedding_stride;
	uint64_t text_arena_bytes;
	bool enabled;
};

MUST_CHECK result_status model_vision_weights(const struct pack_record *pack,
					      const struct load_layout *layout,
					      const uint32_t *layer_cards,
					      struct model_vision *vision_out);
MUST_CHECK result_status model_vision_arena(struct model_vision *vision,
					    uint32_t sessions,
					    uint64_t arena_offset,
					    uint64_t *arena_end_out);

#endif
