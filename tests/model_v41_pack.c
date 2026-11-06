/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_pack.c: the packs of model_v41_pack.h. Test support: no part of the engine.
 *
 * Contract: model_v41_pack.h. The tensors are listed, changed, sorted by name and laid out unit by unit; the
 * tables are written as pack.h lays them out, with both digests, and served at the pack's whole size by a storage of
 * this file to test_memory_storage_hold_tables_of, which keeps its own copy.
 */
#include "model_v41_pack.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack_digest.h"
#include "anemone/storage_seam.h"
#include "anemone/weight_format_seam.h"
#include "memory_storage.h"
#include "model_v41.h"

#define BLOCK PACK_BLOCK_BYTES
#define NAME_CAPACITY 64
#define OTHER_MODEL_NAME "DeepSeek-V4.1-Lite"
#define DRAFT_NAME_PREFIX "mtp."                /* a draft tensor's name: then its stage */
#define DECIMAL_BASE 10
#define TEST_EXL3_K2_LAYERS UINT64_C(0x7C0000)
#define TEST_EXL3_K2_BITS 2u
#define TEST_EXL3_K3_BITS 3u

enum {
	BF16 = PACK_ELEMENT_BFLOAT16,
	F32 = PACK_ELEMENT_FLOAT32,
	U8 = PACK_ELEMENT_UINT8
};
enum {
	PLAIN = WEIGHT_FORMAT_NONE,
	E4M3 = WEIGHT_FORMAT_E4M3_SCALED,
	E2M1 = WEIGHT_FORMAT_E2M1_SCALED
};
enum { EVERY = TEST_MODEL_V41_EVERY, SOURCE = TEST_MODEL_V41_SOURCE, POOLED = TEST_MODEL_V41_POOLED,
       INDEX = TEST_MODEL_V41_INDEX, ENGRAM = TEST_MODEL_V41_ENGRAM };

/* REF:640-650, 657-661, 437-450, 505-521, 344-347, 791-806, 868-870, 936-944; CFG:3, 4, 13, 14, 20-25, 34, 35, 45 */
static const struct test_model_v41_tensor
	layer_tensors[TEST_MODEL_V41_LAYER_TENSORS] = {
		{ V41_DATA_ATTENTION_NORM,
		  "attn_norm.weight",
		  BF16,
		  { 5120, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_FFN_NORM,
		  "ffn_norm.weight",
		  BF16,
		  { 5120, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_ATTENTION_MIX,
		  "hc_attn_fn",
		  F32,
		  { 24, 20480 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_ATTENTION_MIX_BASE,
		  "hc_attn_base",
		  F32,
		  { 24, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_ATTENTION_MIX_SCALE,
		  "hc_attn_scale",
		  F32,
		  { 3, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_FFN_MIX,
		  "hc_ffn_fn",
		  F32,
		  { 24, 20480 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_FFN_MIX_BASE,
		  "hc_ffn_base",
		  F32,
		  { 24, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_FFN_MIX_SCALE,
		  "hc_ffn_scale",
		  F32,
		  { 3, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_SINK,
		  "attn.attn_sink",
		  F32,
		  { 64, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_QUERY_DOWN,
		  "attn.wq_a.weight",
		  U8,
		  { 1280, 5120 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_QUERY_NORM,
		  "attn.q_norm.weight",
		  BF16,
		  { 1280, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_QUERY_UP,
		  "attn.wq_b.weight",
		  U8,
		  { 32768, 1280 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_WINDOW,
		  "attn.wkv.weight",
		  U8,
		  { 512, 5120 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_WINDOW_NORM,
		  "attn.kv_norm.weight",
		  BF16,
		  { 512, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_OUTPUT_GROUPS,
		  "attn.wo_a.weight",
		  U8,
		  { 8192, 4096 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_OUTPUT,
		  "attn.wo_b.weight",
		  U8,
		  { 5120, 8192 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_ROUTER,
		  "ffn.gate.weight",
		  BF16,
		  { 384, 5120 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_ROUTER_BIAS,
		  "ffn.gate.bias",
		  F32,
		  { 384, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_SHARED_GATE,
		  "ffn.shared_experts.w1.weight",
		  U8,
		  { 2304, 5120 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_SHARED_UP,
		  "ffn.shared_experts.w3.weight",
		  U8,
		  { 2304, 5120 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_SHARED_DOWN,
		  "ffn.shared_experts.w2.weight",
		  U8,
		  { 5120, 2304 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_COMPRESSOR,
		  "attn.compressor.wkv.weight",
		  BF16,
		  { 512, 5120 },
		  PLAIN,
		  0,
		  SOURCE },
		{ V41_DATA_COMPRESSOR_GATE,
		  "attn.compressor.wgate.weight",
		  BF16,
		  { 512, 5120 },
		  PLAIN,
		  0,
		  POOLED },
		{ V41_DATA_COMPRESSOR_NORM,
		  "attn.compressor.norm.weight",
		  BF16,
		  { 512, 0 },
		  PLAIN,
		  0,
		  SOURCE },
		{ V41_DATA_KEY,
		  "attn.indexer.wk.weight",
		  BF16,
		  { 128, 512 },
		  PLAIN,
		  0,
		  SOURCE },
		{ V41_DATA_KEY_NORM,
		  "attn.indexer.k_norm.weight",
		  BF16,
		  { 128, 0 },
		  PLAIN,
		  0,
		  SOURCE },
		{ V41_DATA_INDEX_QUERY_UP,
		  "attn.indexer.wq_b.weight",
		  U8,
		  { 4096, 1280 },
		  E4M3,
		  8,
		  INDEX },
		{ V41_DATA_INDEX_WEIGHTS,
		  "attn.indexer.weights_proj.weight",
		  BF16,
		  { 32, 5120 },
		  PLAIN,
		  0,
		  INDEX },
		{ V41_DATA_ENGRAM,
		  "engram.wkv.weight",
		  U8,
		  { 25600, 6144 },
		  E4M3,
		  8,
		  ENGRAM },
		{ V41_DATA_ENGRAM_QUERY,
		  "engram.q_weight",
		  BF16,
		  { 4, 5120 },
		  PLAIN,
		  0,
		  ENGRAM },
		{ V41_DATA_ENGRAM_KEY,
		  "engram.k_weight",
		  BF16,
		  { 4, 5120 },
		  PLAIN,
		  0,
		  ENGRAM },
		{ MODEL_LAYERS,
		  "ffn.gate.bias_vl",
		  F32,
		  { 384, 0 },
		  PLAIN,
		  0,
		  EVERY }, /* the image spans' bias */
	};
/* REF:836-838: gate w1, down w2, up w3, in the order of their names */
static const struct test_model_v41_tensor
	expert_tensors[TEST_MODEL_V41_EXPERT_TENSORS] = {
		{ V41_DATA_GATE,
		  "w1.weight",
		  U8,
		  { 2304, 5120 },
		  E2M1,
		  4,
		  EVERY },
		{ V41_DATA_DOWN,
		  "w2.weight",
		  U8,
		  { 5120, 2304 },
		  E2M1,
		  4,
		  EVERY },
		{ V41_DATA_UP, "w3.weight", U8, { 2304, 5120 }, E2M1, 4, EVERY },
	};
/* REF:1171-1176, 1001-1012; CFG:2 */
static const struct test_model_v41_tensor
	global_tensors[TEST_MODEL_V41_GLOBAL_TENSORS] = {
		{ V41_DATA_EMBEDDING,
		  "embed.weight",
		  BF16,
		  { 129280, 5120 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_NORM,
		  "norm.weight",
		  BF16,
		  { 5120, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_HEAD,
		  "head.weight",
		  BF16,
		  { 129280, 5120 },
		  PLAIN,
		  0,
		  EVERY },
	};
/* CFG:27, 28, 52 (ratio 2: layers 2 .. 19), 43 */
static const uint64_t layers_of_kind[TEST_MODEL_V41_LAYER_KINDS] = {0xFFFFFFFFFFull, 0x104104ull, 0x4104ull,
                                                                     0x1111104104ull, 0x4002ull};
/* The checkpoint index's mtp.* entries (dspark_v1/MAP_A_draft_model.md 2.1): a stage's layer tensors, its router of
 * 128 experts (CFG:11), and the tensors of stages 0 and 2 alone (REF:1116-1121, 1147-1157). */
static const struct test_model_v41_tensor
	draft_stage_tensors[TEST_MODEL_V41_DRAFT_STAGE_TENSORS] = {
		{ V41_DATA_ATTENTION_NORM,
		  "attn_norm.weight",
		  BF16,
		  { 5120, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_FFN_NORM,
		  "ffn_norm.weight",
		  BF16,
		  { 5120, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_ATTENTION_MIX,
		  "hc_attn_fn",
		  F32,
		  { 24, 20480 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_ATTENTION_MIX_BASE,
		  "hc_attn_base",
		  F32,
		  { 24, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_ATTENTION_MIX_SCALE,
		  "hc_attn_scale",
		  F32,
		  { 3, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_FFN_MIX,
		  "hc_ffn_fn",
		  F32,
		  { 24, 20480 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_FFN_MIX_BASE,
		  "hc_ffn_base",
		  F32,
		  { 24, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_FFN_MIX_SCALE,
		  "hc_ffn_scale",
		  F32,
		  { 3, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_SINK,
		  "attn.attn_sink",
		  F32,
		  { 64, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_QUERY_DOWN,
		  "attn.wq_a.weight",
		  U8,
		  { 1280, 5120 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_QUERY_NORM,
		  "attn.q_norm.weight",
		  BF16,
		  { 1280, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_QUERY_UP,
		  "attn.wq_b.weight",
		  U8,
		  { 32768, 1280 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_WINDOW,
		  "attn.wkv.weight",
		  U8,
		  { 512, 5120 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_WINDOW_NORM,
		  "attn.kv_norm.weight",
		  BF16,
		  { 512, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_OUTPUT_GROUPS,
		  "attn.wo_a.weight",
		  U8,
		  { 8192, 4096 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_OUTPUT,
		  "attn.wo_b.weight",
		  U8,
		  { 5120, 8192 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_ROUTER,
		  "ffn.gate.weight",
		  BF16,
		  { 128, 5120 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_ROUTER_BIAS,
		  "ffn.gate.bias",
		  F32,
		  { 128, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_SHARED_GATE,
		  "ffn.shared_experts.w1.weight",
		  U8,
		  { 2304, 5120 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_SHARED_UP,
		  "ffn.shared_experts.w3.weight",
		  U8,
		  { 2304, 5120 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_SHARED_DOWN,
		  "ffn.shared_experts.w2.weight",
		  U8,
		  { 5120, 2304 },
		  E4M3,
		  8,
		  EVERY },
		{ MODEL_LAYERS,
		  "ffn.gate.bias_vl",
		  F32,
		  { 128, 0 },
		  PLAIN,
		  0,
		  EVERY },
	};
static const struct test_model_v41_tensor
	draft_global_tensors[TEST_MODEL_V41_DRAFT_GLOBAL_TENSORS] = {
		{ V41_DATA_DRAFT_MAIN_PROJECTION,
		  "mtp.0.main_proj.weight",
		  U8,
		  { 5120, 15360 },
		  E4M3,
		  8,
		  EVERY },
		{ V41_DATA_DRAFT_MAIN_NORM,
		  "mtp.0.main_norm.weight",
		  BF16,
		  { 5120, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_DRAFT_NORM,
		  "mtp.2.norm.weight",
		  BF16,
		  { 5120, 0 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_DRAFT_MARKOV_EMBEDDING,
		  "mtp.2.markov_head.embed.weight",
		  BF16,
		  { 129280, 256 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_DRAFT_MARKOV_HEAD,
		  "mtp.2.markov_head.head.weight",
		  BF16,
		  { 129280, 256 },
		  PLAIN,
		  0,
		  EVERY },
		{ V41_DATA_DRAFT_CONFIDENCE,
		  "mtp.2.confidence_head.proj.weight",
		  BF16,
		  { 1, 5376 },
		  PLAIN,
		  0,
		  EVERY },
	};
const struct test_model_v41_tensor *const test_model_v41_layer_tensors = layer_tensors;
const struct test_model_v41_tensor *const test_model_v41_expert_tensors = expert_tensors;
const struct test_model_v41_tensor *const test_model_v41_global_tensors = global_tensors;
const uint64_t *const test_model_v41_layers_of_kind = layers_of_kind;
const struct test_model_v41_tensor *const test_model_v41_draft_stage_tensors = draft_stage_tensors;
const struct test_model_v41_tensor *const test_model_v41_draft_global_tensors = draft_global_tensors;
static const uint32_t element_bytes[] = {[BF16] = 2, [F32] = 4, [U8] = 1};

struct built_tensor {
    char name[NAME_CAPACITY];
    struct test_model_v41_tensor spec;
    uint32_t unit;
    uint64_t byte_offset_in_extent;
};

static unsigned char *tables_bytes = NULL;      /* block 0 and the tables, while they are held */
static uint64_t tables_byte_count = 0;
static uint64_t whole_pack_byte_count = 0;

static int name_order(const void *first, const void *second)
{
    return strcmp(((const struct built_tensor *)first)->name, ((const struct built_tensor *)second)->name);
}

/* The bytes of a tensor: of a matrix as its format stores it (0 for a format this build does not carry). */
static uint64_t tensor_bytes(const struct test_model_v41_tensor *spec)
{
	const struct matrix_shape_record shape = { spec->dimensions[0],
						   spec->dimensions[1],
						   spec->bits_per_weight, 0 };
	const struct weight_format_operations *const format =
		weight_format_operations_of(
			(enum weight_format_code)spec->weight_format);
	uint64_t byte_count = 0;
	if (spec->weight_format == PLAIN) {
		return (uint64_t)element_bytes[spec->element_type] *
		       spec->dimensions[0] *
		       (spec->dimensions[1] == 0 ? 1u : spec->dimensions[1]);
	}
	return format != NULL &&
			       format->stored_byte_count(&shape, &byte_count) ==
				       STATUS_OK ?
		       byte_count :
		       0;
}

static uint64_t round_up(uint64_t value, uint64_t unit)
{
    return (value + unit - 1u) / unit * unit;
}

/* Every tensor of the model (with has_draft its draft head's too) with the change, sorted by name; their count, or 0
 * for no memory. */
static uint32_t tensors_list(struct test_model_v41_change change, bool has_draft, struct built_tensor **tensors_out)
{
	const uint32_t most =
		TEST_MODEL_V41_GLOBAL_TENSORS +
		MODEL_LAYERS * (TEST_MODEL_V41_LAYER_TENSORS +
				MODEL_EXPERTS * TEST_MODEL_V41_EXPERT_TENSORS) +
		MODEL_DRAFT_STAGES *
			(TEST_MODEL_V41_DRAFT_STAGE_TENSORS +
			 MODEL_DRAFT_EXPERTS * TEST_MODEL_V41_EXPERT_TENSORS) +
		TEST_MODEL_V41_DRAFT_GLOBAL_TENSORS;
	struct built_tensor *const tensors = calloc(most, sizeof(*tensors));
	uint32_t count = 0;
	if (tensors == NULL) {
		return 0;
	}
    for (uint32_t global = 0; global < TEST_MODEL_V41_GLOBAL_TENSORS; global++) {
        tensors[count] = (struct built_tensor){.spec = test_model_v41_global_tensors[global],
                                               .unit = global == 0 ? 0 : TEST_MODEL_V41_HEAD_UNIT};
        (void)snprintf(tensors[count].name, NAME_CAPACITY, "%s", tensors[count].spec.name);
        count++;
    }
    for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
	    for (uint32_t index = 0; index < TEST_MODEL_V41_LAYER_TENSORS;
		 index++) {
		    const struct test_model_v41_tensor *const spec =
			    &test_model_v41_layer_tensors[index];
		    if ((test_model_v41_layers_of_kind[spec->layers] >> layer &
			 1u) != 0) {
			    tensors[count] = (struct built_tensor){
				    .spec = *spec,
				    .unit = TEST_MODEL_V41_LAYER_UNIT(layer)
			    };
			    (void)snprintf(tensors[count].name, NAME_CAPACITY,
					   "layers.%" PRIu32 ".%s", layer,
					   spec->name);
			    count++;
		    }
	    }
	    for (uint32_t expert = 0; expert < MODEL_EXPERTS; expert++) {
		    for (uint32_t index = 0;
			 index < TEST_MODEL_V41_EXPERT_TENSORS; index++) {
			    const struct test_model_v41_tensor *const spec =
				    &test_model_v41_expert_tensors[index];
			    tensors[count] = (struct built_tensor){
				    .spec = *spec,
				    .unit = TEST_MODEL_V41_EXPERT_UNIT(layer,
								       expert)
			    };
			    if (change.kind == TEST_MODEL_V41_CHANGE_EXL3) {
				    tensors[count].spec.weight_format =
					    WEIGHT_FORMAT_EXL3;
				    tensors[count].spec.bits_per_weight =
					    (TEST_EXL3_K2_LAYERS >> layer &
					     1u) != 0 ?
						    TEST_EXL3_K2_BITS :
						    TEST_EXL3_K3_BITS;
			    }
			    (void)snprintf(tensors[count].name, NAME_CAPACITY,
					   "layers.%" PRIu32
					   ".ffn.experts.%" PRIu32 ".%s",
					   layer, expert, spec->name);
			    count++;
		    }
	    }
    }
    for (uint32_t stage = 0; has_draft && stage < MODEL_DRAFT_STAGES; stage++) {
	    for (uint32_t index = 0; index < TEST_MODEL_V41_DRAFT_STAGE_TENSORS;
		 index++) {
		    tensors[count] = (struct built_tensor){
			    .spec = test_model_v41_draft_stage_tensors[index],
			    .unit = TEST_MODEL_V41_DRAFT_STAGE_UNIT(stage)
		    };
		    (void)snprintf(tensors[count].name, NAME_CAPACITY,
				   DRAFT_NAME_PREFIX "%" PRIu32 ".%s", stage,
				   tensors[count].spec.name);
		    count++;
	    }
	    for (uint32_t expert = 0; expert < MODEL_DRAFT_EXPERTS; expert++) {
		    for (uint32_t index = 0;
			 index < TEST_MODEL_V41_EXPERT_TENSORS; index++) {
			    tensors[count] = (struct built_tensor){
				    .spec = test_model_v41_expert_tensors[index],
				    .unit = TEST_MODEL_V41_DRAFT_EXPERT_UNIT(
					    stage, expert)
			    };
			    (void)snprintf(tensors[count].name, NAME_CAPACITY,
					   DRAFT_NAME_PREFIX
					   "%" PRIu32 ".ffn.experts.%" PRIu32
					   ".%s",
					   stage, expert,
					   tensors[count].spec.name);
			    count++;
		    }
	    }
    }
    for (uint32_t index = 0; has_draft && index < TEST_MODEL_V41_DRAFT_GLOBAL_TENSORS; index++) {
        const char *const name = test_model_v41_draft_global_tensors[index].name;
        const unsigned long stage = strtoul(name + strlen(DRAFT_NAME_PREFIX), NULL, DECIMAL_BASE);
        tensors[count] = (struct built_tensor){.spec = test_model_v41_draft_global_tensors[index],
                                               .unit = TEST_MODEL_V41_DRAFT_STAGE_UNIT((uint32_t)stage)};
        (void)snprintf(tensors[count].name, NAME_CAPACITY, "%s", name);
        count++;
    }
    for (uint32_t index = 0; index < count; index++) {
        struct test_model_v41_tensor *const spec = &tensors[index].spec;
        if (change.name == NULL || strcmp(tensors[index].name, change.name) != 0) {
            continue;
        }
        if (change.kind == TEST_MODEL_V41_CHANGE_EXL3)
            spec->bits_per_weight = change.value;
        spec->element_type = change.kind == TEST_MODEL_V41_CHANGE_ELEMENT ? change.value : spec->element_type;
        spec->dimensions[0] = change.kind == TEST_MODEL_V41_CHANGE_OUTPUTS ? change.value : spec->dimensions[0];
        spec->weight_format = change.kind == TEST_MODEL_V41_CHANGE_FORMAT ? change.value : spec->weight_format;
        spec->bits_per_weight = spec->weight_format == PLAIN ? 0 : spec->bits_per_weight;
        tensors[index].unit = change.kind == TEST_MODEL_V41_CHANGE_UNIT ? change.value : tensors[index].unit;
        if (change.kind == TEST_MODEL_V41_CHANGE_REMOVE) {
            tensors[index] = tensors[--count];
        }
        break;                                  /* names are unique */
    }
    qsort(tensors, count, sizeof(*tensors), name_order);
    *tensors_out = tensors;
    return count;
}

/* Block 0 and the tables of the pack of these tensors in unit_count units, into tables_bytes: false for no memory. */
static bool tables_write(struct built_tensor *tensors, uint32_t count, bool names_another_model, uint32_t unit_count)
{
    uint64_t *const extent_bytes = calloc(unit_count, sizeof(uint64_t));
    uint64_t name_bytes = 0;
    for (uint32_t index = 0; extent_bytes != NULL && index < count; index++) {
	    tensors[index].byte_offset_in_extent =
		    round_up(extent_bytes[tensors[index].unit],
			     WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES);
	    extent_bytes[tensors[index].unit] =
		    tensors[index].byte_offset_in_extent +
		    tensor_bytes(&tensors[index].spec);
	    name_bytes += strlen(tensors[index].name);
    }
    const uint64_t records = unit_count * (uint64_t)(PACK_EXTENT_RECORD_BYTES +
						     PACK_UNIT_RECORD_BYTES) +
			     count * (uint64_t)PACK_TENSOR_RECORD_BYTES;
    tables_byte_count = BLOCK + round_up(records + name_bytes, BLOCK);
    tables_bytes = extent_bytes == NULL ? NULL : aligned_alloc(BLOCK, (size_t)tables_byte_count);
    if (tables_bytes == NULL) {
        free(extent_bytes);
        return false;
    }
    memset(tables_bytes, 0, (size_t)tables_byte_count);
    unsigned char *cursor = tables_bytes + BLOCK;
    uint64_t file_offset = tables_byte_count;
    for (uint32_t unit = 0; unit < unit_count; unit++) {
	    const struct pack_extent extent = {
		    file_offset, round_up(extent_bytes[unit] + 1u, BLOCK), 0,
		    unit, 0
	    };
	    memcpy(cursor, &extent, sizeof(extent));
	    cursor += sizeof(extent);
	    file_offset += extent.byte_count;
    }
    for (uint32_t unit = 0; unit < unit_count; unit++) {
        const bool is_draft = unit >= TEST_MODEL_V41_UNIT_COUNT;   /* the draft head's units: outside the layers */
        const bool outside = unit == 0 || unit == TEST_MODEL_V41_HEAD_UNIT || is_draft;
        const bool is_stage = is_draft ? (unit - TEST_MODEL_V41_UNIT_COUNT) % TEST_MODEL_V41_DRAFT_UNITS_PER_STAGE == 0
                                       : outside || (unit - 1u) % TEST_MODEL_V41_UNITS_PER_LAYER == 0;
	const struct pack_unit_record record = {
		is_stage ? PLACEMENT_UNIT_STAGE : PLACEMENT_UNIT_ON_DEMAND,
		outside ? PACK_NO_LAYER :
			  (unit - 1u) / TEST_MODEL_V41_UNITS_PER_LAYER
	};
	memcpy(cursor, &record, sizeof(record));
        cursor += sizeof(record);
    }
    unsigned char *const names =
	    cursor + count * (uint64_t)PACK_TENSOR_RECORD_BYTES;
    for (uint32_t index = 0, name_offset = 0; index < count; index++) {
        const struct test_model_v41_tensor *const spec = &tensors[index].spec;
        const uint32_t length = (uint32_t)strlen(tensors[index].name);
	const struct pack_tensor record = {
		tensors[index].byte_offset_in_extent,
		tensor_bytes(spec),
		tensors[index].unit,
		name_offset,
		length,
		spec->element_type,
		spec->dimensions[1] == 0 ? 1u : TEST_MODEL_V41_DIMENSIONS,
		{ spec->dimensions[0], spec->dimensions[1], 0, 0 },
		spec->weight_format,
		spec->bits_per_weight,
		{ 0 }
	};
	memcpy(cursor, &record, sizeof(record));
        cursor += sizeof(record);
        memcpy(names + name_offset, tensors[index].name, length);
        name_offset += length;
    }
    struct pack_header header = { .format_version = PACK_FORMAT_VERSION,
				  .block_byte_count = BLOCK,
				  .file_byte_count = file_offset,
				  .tables_byte_count =
					  tables_byte_count - BLOCK,
				  .extent_count = unit_count,
				  .unit_count = unit_count,
				  .tensor_count = count,
				  .name_byte_count = (uint32_t)name_bytes };
    memcpy(header.magic, PACK_MAGIC, sizeof(header.magic));
    (void)snprintf(header.model_name, sizeof(header.model_name), "%s",
		   names_another_model ? OTHER_MODEL_NAME :
					 V41_DATA_MODEL_NAME);
    const bool digested = pack_digest_of_host_bytes(tables_bytes + BLOCK,
						    header.tables_byte_count,
						    &header.tables_digest) ==
			  STATUS_OK;
    memcpy(tables_bytes, &header, sizeof(header));
    const bool sealed = pack_digest_of_host_bytes(tables_bytes, BLOCK,
						  &header.header_digest) ==
			STATUS_OK;
    memcpy(tables_bytes, &header, sizeof(header));
    whole_pack_byte_count = file_offset;
    free(extent_bytes);
    return digested && sealed;
}

MUST_CHECK static result_status
source_describe(const struct storage_reader *reader,
		struct storage_description *description_out)
{
    (void)reader;
    *description_out =
	    (struct storage_description){ .byte_count = whole_pack_byte_count,
					  .copy_count = 1,
					  .block_alignment_bytes = BLOCK };
    return STATUS_OK;
}

MUST_CHECK static result_status
source_read(const struct storage_reader *reader, uint32_t copy_index,
	    uint64_t byte_offset, void *destination, uint64_t byte_count)
{
    (void)reader;
    if (copy_index != 0 || byte_offset > tables_byte_count || byte_count > tables_byte_count - byte_offset) {
	    return STATUS_INVALID_ARGUMENT;
    }
    memcpy(destination, tables_bytes + byte_offset, (size_t)byte_count);
    return STATUS_OK;
}

static const struct storage_operations_record source_storage = {
	.name = "test model tables",
	.reader_describe = source_describe,
	.reader_read = source_read
};

MUST_CHECK static result_status pack_open(struct test_model_v41_change change,
					  bool has_draft,
					  struct pack_record **pack_out)
{
    static const char *const one_path[] = {"model"};
    struct built_tensor *tensors = NULL;
    struct storage_reader *reader = NULL;
    const uint32_t count = tensors_list(change, has_draft, &tensors);
    result_status status =
	    count != 0 && tables_write(
				  tensors, count,
				  change.kind == TEST_MODEL_V41_CHANGE_MODEL,
				  has_draft ? TEST_MODEL_V41_DRAFT_UNIT_COUNT :
					      TEST_MODEL_V41_UNIT_COUNT) ?
		    STATUS_OK :
		    STATUS_OUT_OF_MEMORY;
    test_memory_storage_reset();
    status = status == STATUS_OK ?
		     test_memory_storage_hold_tables_of(
			     &source_storage, (const void *)one_path) :
		     status;
    status = status == STATUS_OK ?
		     test_memory_storage.reader_open(one_path, 1, &reader) :
		     status;
    status = status == STATUS_OK ?
		     pack_open_entry(&test_memory_storage, reader, pack_out) :
		     status;
    test_memory_storage.reader_close(reader);
    free(tables_bytes);
    tables_bytes = NULL;
    free(tensors);
    return status;
}

result_status test_model_v41_pack_open(struct test_model_v41_change change,
				       struct pack_record **pack_out)
{
    return pack_open(change, false, pack_out);
}

result_status
test_model_v41_draft_pack_open(struct test_model_v41_change change,
			       struct pack_record **pack_out)
{
    return pack_open(change, true, pack_out);
}

void test_model_v41_draft_place(uint32_t *card_of_unit, uint32_t card)
{
    for (uint32_t unit = TEST_MODEL_V41_UNIT_COUNT; unit < TEST_MODEL_V41_DRAFT_UNIT_COUNT; unit++) {
        card_of_unit[unit] = card;
    }
}

void test_model_v41_five_cards(uint32_t *card_of_unit, uint32_t moved_unit, uint32_t moved_to)
{
    card_of_unit[0] = 0;
    card_of_unit[TEST_MODEL_V41_HEAD_UNIT] = TEST_MODEL_V41_CARDS - 1u;
    for (uint32_t unit = 1; unit < TEST_MODEL_V41_HEAD_UNIT; unit++) {
        card_of_unit[unit] = (unit - 1u) / TEST_MODEL_V41_UNITS_PER_LAYER / TEST_MODEL_V41_LAYERS_PER_CARD;
    }
    if (moved_unit < TEST_MODEL_V41_UNIT_COUNT) {
        card_of_unit[moved_unit] = moved_to;
    }
}
