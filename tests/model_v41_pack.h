/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_pack.h: packs of DeepSeek-V4.1-Flash's names and shapes, written as tables only, for the tests of the
 * model (model_v41.h) and of its programs. Test support: no part of the engine.
 *
 * Contract:
 *   - The tensors are the reference's (REF and CFG as model_layout.h), stated here and not taken from model_v41.c:
 *     the tables below, each tensor with the enumerator of model_v41.h it stands for (MODEL_LAYERS for one no
 *     program reads) and the kind of layers that have it.
 *   - Units: unit 0 the embedding; layer l a stage unit at TEST_MODEL_V41_LAYER_UNIT(l), then its 384 experts, each an
 *     on-demand unit; the head last (norm.weight, head.weight). One extent a unit: its tensors in the order of their
 *     names, each on a multiple of 64 bytes, at least one byte behind the last, whole blocks.
 *   - test_model_v41_pack_open writes block 0 and the tables of that pack with one change and opens them through
 *     test_memory_storage, which then holds them (memory_storage.h, reset first): the pack has its whole size, about
 *     278 GiB, and no extent byte exists. TEST_MODEL_V41_CHANGE_NONE writes the model's pack. A failure is the
 *     status of the memory, the storage or pack_open_entry, and leaves pack_out untouched.
 *   - test_model_v41_five_cards writes the placement of first_token_plan.md 1.3: layer l and its experts on card l /
 *     8, the embedding on card 0, the head on card 4; then unit moved_unit (below TEST_MODEL_V41_UNIT_COUNT) on
 *     moved_to, a card or PLACEMENT_IN_HOST_MEMORY.
 *   - test_model_v41_draft_pack_open writes the same pack with the draft head behind the head, as the converter
 *     orders it (dspark_v1/MAP_A_draft_model.md 2.1): draft stage s a stage unit outside the layers at
 *     TEST_MODEL_V41_DRAFT_STAGE_UNIT(s), of its stage tensors and of the draft tensors its stage holds, then its 128
 *     experts, each an on-demand unit. test_model_v41_draft_place puts every draft unit on one card.
 */
#ifndef TEST_MODEL_V41_PACK_H
#define TEST_MODEL_V41_PACK_H

#include <stdint.h>

#include "anemone/pack.h"
#include "anemone/status.h"

#define TEST_MODEL_V41_UNITS_PER_LAYER (1u + MODEL_EXPERTS)
#define TEST_MODEL_V41_UNIT_COUNT \
	(MODEL_LAYERS * TEST_MODEL_V41_UNITS_PER_LAYER + 2u)
#define TEST_MODEL_V41_HEAD_UNIT (TEST_MODEL_V41_UNIT_COUNT - 1u)
#define TEST_MODEL_V41_LAYER_UNIT(layer) (1u + (layer) * TEST_MODEL_V41_UNITS_PER_LAYER)
#define TEST_MODEL_V41_EXPERT_UNIT(layer, expert) (TEST_MODEL_V41_LAYER_UNIT(layer) + 1u + (expert))
#define TEST_MODEL_V41_CARDS 5u
#define TEST_MODEL_V41_LAYERS_PER_CARD 8u
#define TEST_MODEL_V41_DIMENSIONS 2u
#define TEST_MODEL_V41_LAYER_TENSORS 32u
#define TEST_MODEL_V41_EXPERT_TENSORS 3u
#define TEST_MODEL_V41_GLOBAL_TENSORS 3u
#define TEST_MODEL_V41_DRAFT_UNITS_PER_STAGE (1u + MODEL_DRAFT_EXPERTS)
#define TEST_MODEL_V41_DRAFT_UNIT_COUNT \
	(TEST_MODEL_V41_UNIT_COUNT +    \
	 MODEL_DRAFT_STAGES * TEST_MODEL_V41_DRAFT_UNITS_PER_STAGE)
#define TEST_MODEL_V41_DRAFT_STAGE_UNIT(stage) \
    (TEST_MODEL_V41_UNIT_COUNT + (stage) * TEST_MODEL_V41_DRAFT_UNITS_PER_STAGE)
#define TEST_MODEL_V41_DRAFT_EXPERT_UNIT(stage, expert) (TEST_MODEL_V41_DRAFT_STAGE_UNIT(stage) + 1u + (expert))
#define TEST_MODEL_V41_DRAFT_STAGE_TENSORS 22u      /* of each stage: a layer's 21 that a stage has, the bias_vl */
#define TEST_MODEL_V41_DRAFT_GLOBAL_TENSORS 6u

/* The layers that have a tensor: every one, the sources, the 2-to-1 sources, the index layers, the n-gram layers. */
enum test_model_v41_layers { TEST_MODEL_V41_EVERY, TEST_MODEL_V41_SOURCE, TEST_MODEL_V41_POOLED,
                             TEST_MODEL_V41_INDEX, TEST_MODEL_V41_ENGRAM, TEST_MODEL_V41_LAYER_KINDS };

struct test_model_v41_tensor {
    uint32_t place;                             /* its enumerator of model_v41.h */
    const char *name;                           /* behind "layers.<l>." or "layers.<l>.ffn.experts.<e>." */
    uint32_t element_type;
    uint32_t dimensions[TEST_MODEL_V41_DIMENSIONS];   /* outputs, inputs; 0 inputs for a vector */
    uint32_t weight_format;
    uint32_t bits_per_weight;
    uint32_t layers;                            /* enum test_model_v41_layers */
};
/* TEST_MODEL_V41_LAYER_TENSORS, _EXPERT_TENSORS (in the order of their names), _GLOBAL_TENSORS of them; the layers
 * of each kind, bit l for layer l. */
extern const struct test_model_v41_tensor *const test_model_v41_layer_tensors;
extern const struct test_model_v41_tensor *const test_model_v41_expert_tensors;
extern const struct test_model_v41_tensor *const test_model_v41_global_tensors;
extern const uint64_t *const test_model_v41_layers_of_kind;
/* The draft's: TEST_MODEL_V41_DRAFT_STAGE_TENSORS of each stage (names behind "mtp.<s>."; place MODEL_LAYERS
 * for the bias_vl, which no program reads), and TEST_MODEL_V41_DRAFT_GLOBAL_TENSORS by whole name (place: their
 * enumerator of enum v41_draft_global), each in the unit of the stage its name gives. Its experts are
 * test_model_v41_expert_tensors behind "mtp.<s>.ffn.experts.<e>.". */
extern const struct test_model_v41_tensor *const test_model_v41_draft_stage_tensors;
extern const struct test_model_v41_tensor *const test_model_v41_draft_global_tensors;

/* What a test changes in the pack before it is written: a tensor taken out, given another element type, number of
 * outputs or weight format (bits 0 for none), moved into another unit; or the pack names another model. */
enum test_model_v41_change_kind { TEST_MODEL_V41_CHANGE_NONE, TEST_MODEL_V41_CHANGE_REMOVE,
                                  TEST_MODEL_V41_CHANGE_ELEMENT, TEST_MODEL_V41_CHANGE_OUTPUTS,
                                  TEST_MODEL_V41_CHANGE_FORMAT, TEST_MODEL_V41_CHANGE_UNIT,
                                  TEST_MODEL_V41_CHANGE_MODEL, TEST_MODEL_V41_CHANGE_EXL3 };
struct test_model_v41_change {
    enum test_model_v41_change_kind kind;
    const char *name;                           /* the whole name of the tensor it changes */
    uint32_t value;
};

MUST_CHECK result_status test_model_v41_pack_open(
	struct test_model_v41_change change, struct pack_record **pack_out);
MUST_CHECK result_status test_model_v41_draft_pack_open(
	struct test_model_v41_change change, struct pack_record **pack_out);
void test_model_v41_five_cards(uint32_t *card_of_unit, uint32_t moved_unit, uint32_t moved_to);
void test_model_v41_draft_place(uint32_t *card_of_unit, uint32_t card);

#endif
