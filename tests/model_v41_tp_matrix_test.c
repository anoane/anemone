/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Real original-geometry metadata, cold quad/hybrid programs. No model payload or CUDA callback executes. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "host_memory_backend.h"
#include "model_v41_pack.h"
#include "model_v41_tp2_model.h"

#define MATRIX_CARDS 5u
#define MATRIX_PRO 4u
#define MATRIX_ROWS 8u
#define MATRIX_OMITTED_TAIL_SEGMENTS 2u
#define MATRIX_DRAFT 7u
#define MATRIX_CARD_BYTES 4096u
#define MATRIX_MEMORY_BYTES (1ull << 40u)
#define MATRIX_WAIT_NS 1000000000ull
#define MATRIX_OWNER_FIRST_SEGMENT (1u + MATRIX_PRO)
#define MATRIX_OWNER_LAYER_SEGMENTS (1u + MATRIX_PRO)
#define MATRIX_OWNER_FRONT_LAYER 20u
#define MATRIX_RESIDUAL_ARRAYS 2u
#define MATRIX_FINAL_CMP 3u
#define MATRIX_LAST_CMP_LAYER 28u
#define MATRIX_STATE_KINDS 4u
#define MATRIX_POOL_KIND 2u
#define MATRIX_TAIL_STATE_COPIES 4u
#define MATRIX_FRONT_STATE_COPIES 2u
#define MATRIX_OWNER_COMPLETE_HAND_OFFS                               \
	((V41_DATA_TP2_PREFIX_LAYERS + 1u) * MATRIX_RESIDUAL_ARRAYS * \
		 (MATRIX_PRO - 1u) +                                  \
	 MATRIX_TAIL_STATE_COPIES + MATRIX_RESIDUAL_ARRAYS)
#define MATRIX_OWNER_FRONT_HAND_OFFS                                \
	((MATRIX_OWNER_FRONT_LAYER + 1u) * MATRIX_RESIDUAL_ARRAYS * \
		 (MATRIX_PRO - 1u) +                                \
	 MATRIX_FRONT_STATE_COPIES)

static const uint64_t attention_layers[MATRIX_PRO] = { 0xFFull, 0x3F00ull,
						       0xFC000ull,
						       0x1FF00000ull };

struct expected_layout {
	uint32_t layout, segments, groups, checkpoints, front, wide_groups;
	bool exact_output_shards, attention_owner;
	uint64_t layers[MATRIX_CARDS];
};

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "model_v41_tp_matrix_test: %s\n",
			      message);
		failures++;
	}
}

static void check_owner_program(const struct model_plan_record *plan,
				const struct execution_program *program)
{
	expect(program->hand_off_count ==
		       (program->kind == PROGRAM_WIDE ?
				MATRIX_OWNER_FRONT_HAND_OFFS :
				MATRIX_OWNER_COMPLETE_HAND_OFFS),
	       "owner programs retain every residual and native-tail mirror handoff exactly once");
	const struct v41_card *const producer = &plan->cards[MATRIX_FINAL_CMP];
	const struct v41_card *const tail = &plan->cards[MATRIX_PRO];
	uint32_t pool_copies = 0, key_copies = 0, selection_copies = 0,
		 candidate_copies = 0;
	for (uint32_t index = 0; index < program->hand_off_count; index++) {
		const struct hand_off_record *const transfer =
			&program->hand_offs[index];
		if (transfer->to_card != MATRIX_PRO)
			continue;
		if (transfer->source_offset ==
		    producer->pool_offset[MATRIX_OWNER_FRONT_LAYER]) {
			expect(transfer->from_card == MATRIX_FINAL_CMP &&
				       transfer->after_segment + 1u ==
					       program->front_segment_count &&
				       transfer->destination_offset ==
					       tail->pool_offset
						       [MATRIX_OWNER_FRONT_LAYER] &&
				       transfer->byte_count ==
					       MODEL_STATE_ROW_BYTES &&
				       transfer->entry_ratio ==
					       MODEL_DIRECT_RATIO &&
				       transfer->session_stride ==
					       plan->pool_session_bytes
						       [MATRIX_OWNER_FRONT_LAYER],
			       "source-20 pool mirror is copied from its real producer at the front boundary");
			pool_copies++;
		}
		if (transfer->source_offset ==
		    producer->key_tiles_offset[MATRIX_OWNER_FRONT_LAYER]) {
			expect(transfer->from_card == MATRIX_FINAL_CMP &&
				       transfer->after_segment + 1u ==
					       program->front_segment_count &&
				       transfer->destination_offset ==
					       tail->key_tiles_offset
						       [MATRIX_OWNER_FRONT_LAYER] &&
				       transfer->byte_count ==
					       MODEL_KEY_BYTES &&
				       transfer->entry_ratio ==
					       MODEL_DIRECT_RATIO &&
				       transfer->session_stride ==
					       plan->key_session_bytes
						       [MATRIX_OWNER_FRONT_LAYER],
			       "source-20 key mirror uses its original session stride and producer boundary");
			key_copies++;
		}
		if (transfer->source_offset ==
		    producer->selection_offset[MATRIX_LAST_CMP_LAYER]) {
			expect(transfer->from_card == MATRIX_FINAL_CMP &&
				       transfer->destination_offset ==
					       tail->selection_offset
						       [MATRIX_LAST_CMP_LAYER] &&
				       transfer->byte_count ==
					       MODEL_SELECTION_ROW_BYTES &&
				       transfer->entry_ratio == 0 &&
				       transfer->session_stride == 0,
			       "the tail receives its layer-28 selection from the actual index owner");
			selection_copies++;
		}
		if (transfer->source_offset ==
		    producer->array_offset[V41_DATA_ARRAY_CANDIDATES]) {
			expect(transfer->from_card == MATRIX_FINAL_CMP &&
				       transfer->destination_offset ==
					       tail->array_offset
						       [V41_DATA_ARRAY_CANDIDATES] &&
				       transfer->byte_count ==
					       MODEL_CANDIDATES_ROW_BYTES &&
				       transfer->after_segment ==
					       program->front_segment_count,
			       "candidate mirror follows the source-20 continuation before the native tail");
			candidate_copies++;
		}
	}
	const uint32_t later_copies = program->kind == PROGRAM_WIDE ? 0u : 1u;
	expect(pool_copies == 1u && key_copies == 1u &&
		       selection_copies == later_copies &&
		       candidate_copies == later_copies,
	       "the tail has exactly the source, index and candidate mirrors it reads");
	for (uint32_t layer = 0; layer < V41_DATA_TP2_PREFIX_LAYERS; layer++) {
		const uint32_t segment =
			MATRIX_OWNER_FIRST_SEGMENT +
			layer * MATRIX_OWNER_LAYER_SEGMENTS +
			(layer >= MATRIX_OWNER_FRONT_LAYER ? 1u : 0u);
		if (segment >= program->segment_count)
			break;
		uint32_t owner = 0;
		while (owner < MATRIX_PRO &&
		       (attention_layers[owner] >> layer & 1u) == 0)
			owner++;
		if (owner == MATRIX_PRO) {
			expect(false, "independent closed-group owner exists");
			return;
		}
		uint32_t state_kinds[MATRIX_STATE_KINDS] = { 0 };
		for (uint32_t index = 0; index < program->session_range_count;
		     index++) {
			const struct session_range *const range =
				&program->session_ranges[index];
			if (range->identity / MATRIX_STATE_KINDS != layer)
				continue;
			const uint32_t kind =
				range->identity % MATRIX_STATE_KINDS;
			expect(range->card == owner ||
				       (layer == MATRIX_OWNER_FRONT_LAYER &&
					kind >= MATRIX_POOL_KIND &&
					range->card == MATRIX_PRO),
			       "snapshot identities enumerate only their attention owner and live native-tail mirrors");
			state_kinds[kind]++;
		}
		expect(state_kinds[0] == 1u,
		       "snapshot serialization retains exactly one canonical window per layer");
		if (layer == MATRIX_OWNER_FRONT_LAYER)
			expect(state_kinds[MATRIX_POOL_KIND] ==
					       MATRIX_RESIDUAL_ARRAYS &&
				       state_kinds[MATRIX_POOL_KIND + 1u] ==
					       MATRIX_RESIDUAL_ARRAYS,
			       "source-20 snapshot loads restore both the producer and native-tail mirrors");
		expect(program->segments[segment].card == owner &&
			       program->segments[segment].layer == layer,
		       "the attention owner runs before the FFN companion group");
		uint32_t copies[MATRIX_PRO] = { 0 };
		for (uint32_t index = 0; index < program->hand_off_count;
		     index++) {
			const struct hand_off_record *const transfer =
				&program->hand_offs[index];
			if (transfer->after_segment != segment ||
			    transfer->to_card >= MATRIX_PRO)
				continue;
			const uint64_t *const from =
				plan->cards[owner].array_offset;
			const uint64_t *const to =
				plan->cards[transfer->to_card].array_offset;
			expect(transfer->from_card == owner &&
				       transfer->to_card != owner &&
				       transfer->entry_ratio == 0 &&
				       transfer->session_stride == 0 &&
				       transfer->view == 0 &&
				       ((transfer->source_offset ==
						 from[V41_DATA_ARRAY_STREAMS] &&
					 transfer->destination_offset ==
						 to[V41_DATA_ARRAY_STREAMS] &&
					 transfer->byte_count ==
						 MODEL_STREAMS_ROW_BYTES) ||
					(transfer->source_offset ==
						 from[V41_DATA_ARRAY_CARRIED_PRE] &&
					 transfer->destination_offset ==
						 to[V41_DATA_ARRAY_CARRIED_PRE] &&
					 transfer->byte_count ==
						 MODEL_SITE_ROW_BYTES)),
			       "lossless residual handoffs name the exact owner and peer arrays");
			copies[transfer->to_card]++;
		}
		for (uint32_t card = 0; card < MATRIX_PRO; card++)
			expect(copies[card] == (card == owner ?
							0u :
							MATRIX_RESIDUAL_ARRAYS),
			       "every other FFN rank receives both residual arrays exactly once");
	}
}

static void check_layout(const struct pack_record *pack,
			 const struct expected_layout *expected,
			 uint32_t row_capacity)
{
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES, row_capacity, 1,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE,
			    MATRIX_DRAFT, 0, 0 },
		.tensor_layout = expected->layout,
		.exact_output_shards = expected->exact_output_shards,
		.attention_owner = expected->attention_owner
	};
	struct device_card *cards[MATRIX_CARDS] = { 0 };
	struct v41_tp2_model *model = NULL;
	struct execution_program *program = NULL;
	struct step_companion_program groups = { 0 };
	test_host_memory_backend_reset(MATRIX_CARDS, MATRIX_CARD_BYTES);
	bool opened = true;
	for (uint32_t card = 0; card < MATRIX_CARDS; card++) {
		description.cards[card].arithmetic_class =
			card == MATRIX_PRO ? ARITHMETIC_CLASS_SM_120_NATIVE :
					     ARITHMETIC_CLASS_SM_80;
		description.cards[card].memory_total_bytes =
			MATRIX_MEMORY_BYTES;
		description.cards[card].memory_free_bytes = MATRIX_MEMORY_BYTES;
		test_host_memory_backend_set_arithmetic_class(
			card, card == MATRIX_PRO ? ARITHMETIC_CLASS_SM_120 :
						   ARITHMETIC_CLASS_SM_80);
		opened = test_host_memory_backend.card_open(
				 card, MATRIX_CARD_BYTES, &cards[card]) ==
				 STATUS_OK &&
			 opened;
	}
	result_status status =
		opened ? v41_tp2_model_create(pack, &description, &model) :
			 STATUS_OUT_OF_MEMORY;
	expect(status == STATUS_OK && model != NULL,
	       "complete cold layout prepares");
	if (model != NULL) {
		if (expected->attention_owner) {
			struct v41_tp2_model_description refused = description;
			struct v41_tp2_model *untouched = model;
			refused.exact_output_shards = false;
			expect(v41_tp2_model_create(pack, &refused,
						    &untouched) ==
					       STATUS_INVALID_ARGUMENT &&
				       untouched == model,
			       "attention ownership refuses a relaxed recipe without changing the output");
			struct v41_tp2_stage_description bad_stage =
				model->stages[0]->description;
			struct v41_tp2_stage_plan *stage_untouched =
				model->stages[0];
			bad_stage.target_layer_count--;
			expect(v41_tp2_stage_plan_create(pack, &bad_stage,
							 &stage_untouched) ==
					       STATUS_INVALID_ARGUMENT &&
				       stage_untouched == model->stages[0],
			       "attention ownership refuses an incomplete fixed source-group geometry");
			for (uint32_t card = 0; card < MATRIX_PRO; card++) {
				const struct v41_card *const planned =
					&model->canonical->cards[card];
				for (uint32_t layer = 0;
				     layer < V41_DATA_TP2_PREFIX_LAYERS;
				     layer++) {
					const bool owned =
						(attention_layers[card] >>
							 layer &
						 1u) != 0;
					expect((planned->ring_offset[layer] !=
						PROGRAM_NOWHERE) == owned &&
						       (planned->checkpoint_offset
								[layer] !=
							PROGRAM_NOWHERE) ==
							       owned,
					       "only the attention owner keeps its window and rollback checkpoint");
				}
			}
			expect(model->canonical->cards[MATRIX_PRO].pool_offset
						       [MODEL_DECODER_FIRST_LAYER] !=
					       PROGRAM_NOWHERE &&
				       model->canonical->cards[MATRIX_PRO].key_tiles_offset
						       [MODEL_DECODER_FIRST_LAYER] !=
					       PROGRAM_NOWHERE,
			       "native tail retains its source-20 pool and key mirrors");
		}
		for (uint32_t card = 0; card < MATRIX_CARDS; card++)
			expect(model->canonical->cards[card].layers ==
				       expected->layers[card],
			       "independent physical ownership masks match");
		struct v41_tp2_program_description build;
		status = v41_tp2_model_program_description(model, &build);
		if (status == STATUS_OK)
			status = v41_tp2_program_prepare(
				&build, &test_host_memory_backend, cards,
				description.cards, MATRIX_ROWS, &program,
				&groups);
		expect(status == STATUS_OK && program != NULL,
		       "full original-shard program prepares");
		if (program != NULL) {
			if (expected->attention_owner)
				check_owner_program(model->canonical, program);
			expect(program->segment_count == expected->segments &&
				       groups.pair_count == expected->groups,
			       "independent segment and group counts match");
			expect(program->checkpoint_count ==
					       expected->checkpoints &&
				       program->front_segment_count ==
					       expected->front,
			       "all replica checkpoints and source-front boundaries are retained");
			expect(groups.arithmetic_identity ==
				       model->recipe_identity,
			       "loader and program use the same placement arithmetic identity");
			const uint32_t ranks =
				model->stages[0]->description.rank_count;
			for (uint32_t index = 0; index < groups.pair_count;
			     index++)
				expect(groups.pairs[index].second_segment -
						       groups.pairs[index]
							       .first_segment +
						       1u ==
					       ranks,
				       "every group names all participants");
		}
		free(program);
		program = NULL;
		if (row_capacity > MATRIX_ROWS) {
			const enum program_kind kinds[] = { PROGRAM_WIDE,
							    PROGRAM_FULL };
			for (uint32_t index = 0;
			     index < sizeof(kinds) / sizeof(kinds[0]);
			     index++) {
				status = v41_tp_program_prepare(
					&build, &test_host_memory_backend,
					cards, description.cards, kinds[index],
					row_capacity, &program, &groups);
				expect(status == STATUS_OK && program != NULL,
				       "ordered state slabs and wide shard products prepare");
				if (program != NULL) {
					if (expected->attention_owner)
						check_owner_program(
							model->canonical,
							program);
					expect(program->kind == kinds[index] &&
						       program->row_count ==
							       row_capacity &&
						       program->front_segment_count ==
							       expected->front &&
						       groups.arithmetic_identity ==
							       model->recipe_identity,
					       "chunk kind, cone boundary and numerical identity match");
					expect(program->segment_count ==
						       (kinds[index] == PROGRAM_WIDE ?
								expected->front :
								expected->segments -
									MATRIX_OMITTED_TAIL_SEGMENTS),
					       "wide stops at the source front; full omits head and draft tail");
					expect(groups.pair_count ==
						       (kinds[index] == PROGRAM_WIDE ?
								expected->wide_groups :
								expected->groups),
					       "chunk groups retain the thin prefix");
				}
				free(program);
				program = NULL;
			}
			for (uint32_t card = 0; card < MATRIX_PRO; card++) {
				const struct v41_tp2_stage_plan *const stage =
					model->stages[card];
				if (stage->channel_count > TP2_SITES)
					expect(stage->channels[0].collective
								       .arguments
								       .region_offset ==
							       stage->channels[TP2_SITES]
								       .collective
								       .arguments
								       .region_offset &&
						       stage->channels[0]
								       .collective
								       .arguments
								       .region_offset !=
							       stage->channels[1]
								       .collective
								       .arguments
								       .region_offset,
					       "layers reuse one region per distinct reduction site");
			}
		}
	}
	expect(test_host_memory_backend_work_count() == 0,
	       "cold geometry and programs queue no GPU work");
	free(program);
	v41_tp2_model_destroy(model);
	for (uint32_t card = 0; card < MATRIX_CARDS; card++) {
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card],
							    MATRIX_WAIT_NS);
	}
}

#define MATRIX_EXL3_K2_MASK UINT64_C(0x7C0000)
#define MATRIX_EXL3_K2_BITS 2u
#define MATRIX_EXL3_K3_BITS 3u
#define MATRIX_EXL3_SMALL 512u
#define MATRIX_EXL3_LARGE 640u
#define MATRIX_EXL3_DOWN 1280u

static struct v41_tp2_model_description exl3_cold_description(void)
{
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES, MATRIX_ROWS, 1,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE,
			    MATRIX_DRAFT, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.exact_output_shards = true
	};
	for (uint32_t card = 0; card < MATRIX_PRO; card++) {
		description.cards[card].arithmetic_class =
			ARITHMETIC_CLASS_SM_80;
		description.cards[card].memory_total_bytes =
			MATRIX_MEMORY_BYTES;
		description.cards[card].memory_free_bytes = MATRIX_MEMORY_BYTES;
	}
	return description;
}

static void check_exl3_cold_model(const struct pack_record *pack)
{
	struct v41_tp2_model_description description = exl3_cold_description();
	struct v41_tp2_model *model = NULL;
	result_status status = v41_tp2_model_create(pack, &description, &model);
	expect(status == STATUS_OK && model != NULL,
	       "EXL3 four-CMP cold model prepares");
	if (model == NULL)
		return;
	expect(model->canonical->card_count == MATRIX_PRO &&
		       model->canonical->result_card == 0 &&
		       model->stages[MATRIX_PRO] == NULL &&
		       model->views[MATRIX_PRO] == NULL,
	       "four-CMP metadata owns head/draft on CMP0 and creates no PRO stage");
	for (uint32_t card = 0; card < MATRIX_PRO; card++) {
		const struct v41_tp2_stage_plan *const stage =
			model->stages[card];
		const uint32_t expected_inner = card % TP2_RANKS == 0 ?
							MATRIX_EXL3_SMALL :
							MATRIX_EXL3_LARGE;
		expect(stage->description.include_head_and_draft ==
				       (card == 0) &&
			       stage->model->cards[card].exl3_scratch_offset !=
				       PROGRAM_NOWHERE,
		       "only CMP0 owns native draft while every target rank reserves EXL3 scratch");
		for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
			const uint32_t bits =
				(MATRIX_EXL3_K2_MASK >> layer & 1u) != 0 ?
					MATRIX_EXL3_K2_BITS :
					MATRIX_EXL3_K3_BITS;
			expect(model->canonical->exl3_bits[layer] == bits &&
				       stage->model->exl3_bits[layer] == bits &&
				       stage->exl3_shards[layer]
						       .gate_up.output_count ==
					       expected_inner &&
				       stage->exl3_shards[layer]
						       .gate_up.input_count ==
					       MODEL_HIDDEN &&
				       stage->exl3_shards[layer]
						       .down.output_count ==
					       MATRIX_EXL3_DOWN &&
				       stage->exl3_shards[layer]
						       .down.input_count ==
					       V41_DATA_EXPERT_INNER,
			       "every full-K EXL3 layer has its pinned bits and unequal output rectangles");
		}
	}
	uint32_t target_jobs = 0, draft_jobs = 0;
	const struct pack_tensor *const tensors = pack_tensors_of(pack);
	for (uint32_t index = 0; index < model->gpu_load.description.job_count;
	     index++) {
		const struct tp_load_job *const job = &model->jobs[index];
		const struct pack_tensor *const tensor =
			&tensors[job->tensor_index];
		if (tensor->weight_format == WEIGHT_FORMAT_EXL3) {
			const bool down = tensor->dimensions[0] == MODEL_HIDDEN;
			const uint32_t width = down ? MATRIX_EXL3_DOWN :
					       job->card % TP2_RANKS == 0 ?
						      MATRIX_EXL3_SMALL :
						      MATRIX_EXL3_LARGE;
			expect(job->mode == TP_LOAD_SHARD &&
				       job->shard.description.first_input ==
					       0 &&
				       job->shard.shard_shape.input_size ==
					       tensor->dimensions[1] &&
				       job->shard.shard_shape.output_size ==
					       width,
			       "each selected target job retains full K and the rank's exact output shape");
			target_jobs++;
		} else if (tensor->weight_format == WEIGHT_FORMAT_E2M1_SCALED) {
			expect(job->card == 0 && job->mode == TP_LOAD_FULL,
			       "every native draft expert matrix is whole on CMP0");
			draft_jobs++;
		}
	}
	expect(target_jobs == MODEL_LAYERS * MODEL_EXPERTS *
				       MODEL_EXPERT_MATRICES * MATRIX_PRO &&
		       draft_jobs == MODEL_DRAFT_STAGES * MODEL_DRAFT_EXPERTS *
					     MODEL_EXPERT_MATRICES,
	       "target and full native draft expert job counts are complete");
	struct v41_tp2_program_description program_description = {
		.slice_count = MATRIX_DRAFT
	};
	expect(v41_tp2_model_program_description(model, &program_description) ==
			       STATUS_OK &&
		       program_description.slice_count == TP2_DEFAULT_SLICES &&
		       program_description.canonical == model->canonical &&
		       program_description.stages[MATRIX_PRO] == NULL &&
		       program_description.views[MATRIX_PRO] == NULL,
	       "four-card public description publishes metadata without a fifth stage");
	const struct v41_tp2_model_load_request empty_request = { 0 };
	expect(v41_tp2_model_load(model, &empty_request) ==
			       STATUS_INVALID_ARGUMENT &&
		       !model->load_attempted,
	       "invalid four-card load refuses before callbacks or mutation");
	struct v41_tp2_model *untouched = model;
	description.original_code_wire = true;
	expect(v41_tp2_model_create(pack, &description, &untouched) ==
			       STATUS_INVALID_ARGUMENT &&
		       untouched == model,
	       "EXL3 refuses BF16-derived original-code wire without changing outputs");
	description.original_code_wire = false;
	description.attention_owner = true;
	expect(v41_tp2_model_create(pack, &description, &untouched) ==
			       STATUS_INVALID_ARGUMENT &&
		       untouched == model,
	       "EXL3 full40 does not combine with original-hybrid attention ownership");
	description.attention_owner = false;
	description.exl3_product_order = BB_EXL3_EIGHT_PART_ORDER;
	struct v41_tp2_model *optional = NULL;
	status = v41_tp2_model_create(pack, &description, &optional);
	expect(status == STATUS_OK && optional != NULL,
	       "explicit eight-part cold metadata prepares");
	if (optional != NULL) {
		expect(optional->recipe_identity != model->recipe_identity &&
			       optional->stages[0]->aligned_weight_byte_count ==
				       model->stages[0]
					       ->aligned_weight_byte_count,
		       "arithmetic order changes identity but never packed weights");
		v41_tp2_model_destroy(optional);
	}
	v41_tp2_model_destroy(model);
}

static void test_exl3_cold(void)
{
	struct pack_record *pack = NULL;
	struct test_model_v41_change change = {
		.kind = TEST_MODEL_V41_CHANGE_EXL3
	};
	result_status status = test_model_v41_draft_pack_open(change, &pack);
	expect(status == STATUS_OK,
	       "independent EXL3 target/native draft fixture opens");
	if (status == STATUS_OK)
		check_exl3_cold_model(pack);
	pack_close(pack);
	pack = NULL;
	change.name = "layers.18.ffn.experts.1.w3.weight";
	change.value = MATRIX_EXL3_K3_BITS;
	status = test_model_v41_draft_pack_open(change, &pack);
	expect(status == STATUS_OK,
	       "wrong per-expert K2/K3 fixture is valid pack metadata");
	if (status == STATUS_OK) {
		struct v41_tp2_model_description description =
			exl3_cold_description();
		struct v41_tp2_model *model = NULL;
		expect(v41_tp2_model_create(pack, &description, &model) ==
				       STATUS_CORRUPT_PACK &&
			       model == NULL,
		       "per-expert bitrate mismatch refuses before any model publishes");
	}
	pack_close(pack);
}

int main(void)
{
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = {
		.kind = TEST_MODEL_V41_CHANGE_NONE
	};
	const result_status status =
		test_model_v41_draft_pack_open(change, &pack);
	expect(status == STATUS_OK, "independent original-geometry pack opens");
	static const struct expected_layout layouts[] = {
		{ V41_DATA_TP_LAYOUT_QUAD,
		  140u,
		  31u,
		  130u,
		  89u,
		  22u,
		  false,
		  false,
		  { 0x1FFFFFFFull, 0x1FFFFFFFull, 0x1FFFFFFFull, 0x1FFFFFFFull,
		    0xFFE0000000ull } },
		{ V41_DATA_TP_LAYOUT_HYBRID,
		  64u,
		  15u,
		  57u,
		  39u,
		  15u,
		  false,
		  false,
		  { 0x3FFFull, 0x3FFFull, 0x1FC000ull, 0x1FE00000ull,
		    0xFFE0000000ull } },
		{ V41_DATA_TP_LAYOUT_QUAD,
		  166u,
		  30u,
		  43u,
		  106u,
		  21u,
		  true,
		  true,
		  { 0x1FFFFFFFull, 0x1FFFFFFFull, 0x1FFFFFFFull, 0x1FFFFFFFull,
		    0xFFE0000000ull } }
	};
	if (status == STATUS_OK) {
		const uint32_t capacities[] = { MATRIX_ROWS,
						MODEL_WIDE_LEAST_ROWS,
						LINEAR_WIDE_MOST_ROWS };
		for (uint32_t index = 0;
		     index < sizeof(layouts) / sizeof(layouts[0]); index++) {
			for (uint32_t shape = 0;
			     shape < sizeof(capacities) / sizeof(capacities[0]);
			     shape++)
				check_layout(pack, &layouts[index],
					     capacities[shape]);
		}
	}
	pack_close(pack);
	test_exl3_cold();
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
