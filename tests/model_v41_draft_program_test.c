/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_draft_program_test: the target's taps and context and the fused draft's data flow, from table-only packs.
 * No card computes: offsets, weights, row counts, module choices and ordering are checked against independent
 * expectations on class 80, class 120 native and emulated class 120, in both target orders and thin/full programs.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "anemone/attention_core_records.h"
#include "anemone/attention_output_records.h"
#include "anemone/expert_fused_records.h"
#include "anemone/attention_state_records.h"
#include "anemone/draft_records.h"
#include "anemone/hyper_connection_records.h"
#include "anemone/linear_records.h"
#include "anemone/loader.h"
#include "anemone/relaxed_linear_records.h"
#include "anemone/router_records.h"
#include "anemone/sampler_records.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"

#define CARDS TEST_MODEL_V41_CARDS
#define RESULT_CARD (CARDS - 1u)
#define HOST_BYTES 4096u
#define CONTEXT 16384u
#define SESSIONS 4u
#define CAPACITY 2048u
#define FULL_ROWS 1100u
#define WIDE_ROWS 64u
#define PIECE_LAUNCHES 6u
#define DRAFT_SITES_PER_STAGE 2u
#define DRAFT_PREFIX_LAUNCHES 3u /* verify, rotation, embedding */
#define DRAFT_HEAD_LAUNCHES 2u
#define CONTEXT_TO_DRAFT_SEGMENTS 2u /* context, target head, draft */
#define DRAFT_SUFFIX_LAUNCHES \
	(DRAFT_HEAD_LAUNCHES + MODEL_DRAFT_BLOCK * 2u + 1u)
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

static int failures;
static uint32_t maximum_prepared_kernels;

static void expect(bool holds, const char *what)
{
	if (!holds) {
		(void)printf("FAILED: draft program: %s\n", what);
		failures++;
	}
}

static const struct kernel_launch *
launch_at(const struct execution_program *program, uint32_t segment,
	  uint32_t index)
{
	const struct segment_record *const part = &program->segments[segment];
	return &program->cards[part->card].launches[part->first_launch + index];
}

static bool named(const struct kernel_launch *launch, const char *name)
{
	struct test_host_memory_backend_kernel kernel;
	return test_host_memory_backend_kernel_of(RESULT_CARD, launch->kernel,
						  &kernel) &&
	       strcmp(kernel.function_name, name) == 0;
}

static bool module_is(const struct kernel_launch *launch, const char *module)
{
	struct test_host_memory_backend_kernel kernel;
	return test_host_memory_backend_kernel_of(RESULT_CARD, launch->kernel,
						  &kernel) &&
	       strcmp(kernel.kernel_file_name, module) == 0;
}

static void check_taps(const struct model_plan_record *plan,
		       const struct execution_program *program)
{
	const struct v41_card *const card = &plan->cards[RESULT_CARD];
	uint32_t found = 0;
	for (uint32_t segment = 0; segment < program->draft_segment_first;
	     segment++) {
		const struct segment_record *const part =
			&program->segments[segment];
		if (part->layer >= MODEL_DRAFT_FIRST_TAP_LAYER &&
		    part->layer < MODEL_LAYERS) {
			struct hyper_draft_inputs_arguments tap;
			const struct kernel_launch *const launch =
				launch_at(program, segment, 0);
			memcpy(&tap, launch->argument, sizeof(tap));
			expect(part->card == RESULT_CARD &&
				       named(launch, KERNEL_HYPER_DRAFT_INPUTS),
			       "each tap precedes its whole target layer");
			expect(tap.input_offset ==
					       card->array_offset
						       [V41_DATA_ARRAY_STREAMS] &&
				       tap.output_offset ==
					       card->array_offset
						       [V41_DATA_ARRAY_DRAFT_TAPS] &&
				       tap.output_row_values ==
					       MODEL_DRAFT_FEATURES &&
				       tap.column_offset ==
					       (part->layer -
						MODEL_DRAFT_FIRST_TAP_LAYER) *
						       MODEL_HIDDEN &&
				       tap.mode == HYPER_STREAM_MEAN &&
				       launch->grid[1] == program->row_count,
			       "tap uses the layer input, interleaved tap columns and all chunk rows");
			found++;
		}
	}
	expect(found == MODEL_DRAFT_TAPS, "three target layer inputs captured");
}

static void check_context_product(const struct kernel_launch *launch,
				  uint64_t codes, uint64_t exponents,
				  uint64_t operands, uint64_t first_weight,
				  uint64_t first_output, uint64_t second_weight,
				  uint64_t second_output, uint32_t width,
				  uint32_t rows)
{
	struct trunk_linear_quantized_arguments product;
	memcpy(&product, launch->argument, sizeof(product));
	/* CMP full programs take the ordered record, whose input is the explicit operand range. */
	if (module_is(launch, KERNEL_FILE_TRUNK_RELAXED)) {
		struct trunk_relaxed_quantized_arguments ordered;
		memcpy(&ordered, launch->argument, sizeof(ordered));
		expect(ordered.operands_offset == operands &&
			       ordered.weights_offset[0] == first_weight &&
			       ordered.weights_offset[1] == second_weight &&
			       ordered.output_offset[0] == first_output &&
			       ordered.output_offset[1] == second_output &&
			       ordered.input_size == width &&
			       ordered.row_count == rows,
		       "context ordered product reads its own operands and writes its own rows");
	} else {
		expect(product.codes_offset == codes &&
			       product.exponents_offset == exponents &&
			       product.operands_offset == operands &&
			       product.weights_offset[0] == first_weight &&
			       product.weights_offset[1] == second_weight &&
			       product.output_offset[0] == first_output &&
			       product.output_offset[1] == second_output &&
			       product.input_size == width &&
			       product.row_count == rows,
		       "context product reads its own codes/exponents/operands and weights");
	}
}

static void check_context(const struct model_plan_record *plan,
			  const struct execution_program *program,
			  uint32_t context_segment)
{
	const struct v41_card *const card = &plan->cards[RESULT_CARD];
	const uint64_t *const arrays = card->array_offset;
	const struct segment_record *const segment =
		&program->segments[context_segment];
	const bool is_full = program->kind == PROGRAM_FULL;
	const uint32_t pieces =
		is_full ? (program->row_count + LINEAR_WIDE_MOST_ROWS - 1u) /
				  LINEAR_WIDE_MOST_ROWS :
			  1u;
	expect(segment->card == RESULT_CARD &&
		       segment->layer == SEGMENT_NO_LAYER &&
		       segment->launch_count ==
			       pieces * PIECE_LAUNCHES + MODEL_DRAFT_STAGES,
	       "context lies behind layer 39 and has six products a piece and three window launches");
	uint32_t at = 0;
	for (uint32_t piece = 0; piece < pieces; piece++) {
		const uint32_t first_row = piece * LINEAR_WIDE_MOST_ROWS;
		const uint32_t left = program->row_count - first_row;
		const uint32_t rows = left < LINEAR_WIDE_MOST_ROWS ?
					      left :
					      LINEAR_WIDE_MOST_ROWS;
		const uint64_t codes =
			arrays[V41_DATA_ARRAY_DRAFT_TAP_CODES] +
			(uint64_t)first_row * MODEL_DRAFT_FEATURES;
		const uint64_t exponents =
			arrays[V41_DATA_ARRAY_DRAFT_TAP_EXPONENTS] +
			(uint64_t)first_row * MODEL_DRAFT_FEATURES /
				LINEAR_VALUES_PER_BLOCK;
		const uint64_t operands =
			arrays[V41_DATA_ARRAY_DRAFT_TAP_OPERANDS] +
			(uint64_t)first_row * MODEL_DRAFT_TAP_ROW_BYTES;
		const uint64_t main =
			arrays[V41_DATA_ARRAY_DRAFT_MAIN] +
			(uint64_t)first_row * MODEL_HIDDEN_ROW_BYTES;
		const uint64_t main_x =
			arrays[V41_DATA_ARRAY_DRAFT_MAIN_X] +
			(uint64_t)first_row * MODEL_HIDDEN_ROW_BYTES;
		struct activation_quantize_arguments quantize;
		struct hyper_draft_inputs_arguments norm;
		const struct kernel_launch *launch =
			launch_at(program, context_segment, at++);
		memcpy(&quantize, launch->argument, sizeof(quantize));
		expect(named(launch, KERNEL_ACTIVATION_QUANTIZE) &&
			       quantize.width == MODEL_DRAFT_FEATURES &&
			       quantize.values_offset ==
				       arrays[V41_DATA_ARRAY_DRAFT_TAPS] +
					       (uint64_t)first_row *
						       MODEL_DRAFT_TAP_ROW_BYTES &&
			       quantize.codes_offset == codes &&
			       quantize.exponents_offset == exponents &&
			       quantize.operands_offset == operands &&
			       launch->grid[1] == rows,
		       "tap quantization uses piece offsets and full 15,360 width");
		check_context_product(
			launch_at(program, context_segment, at++), codes,
			exponents, operands,
			plan->draft_globals[V41_DATA_DRAFT_MAIN_PROJECTION]
				.stored.byte_offset,
			main, 0, 0, MODEL_DRAFT_FEATURES, rows);
		launch = launch_at(program, context_segment, at++);
		memcpy(&norm, launch->argument, sizeof(norm));
		expect(named(launch, KERNEL_HYPER_DRAFT_INPUTS) &&
			       norm.mode == HYPER_ROW_NORM &&
			       norm.input_offset == main &&
			       norm.output_offset == main_x &&
			       norm.weight_offset ==
				       plan->draft_globals
					       [V41_DATA_DRAFT_MAIN_NORM]
						       .stored.byte_offset,
		       "main norm consumes the projection and its own norm weight");
		launch = launch_at(program, context_segment, at++);
		memcpy(&quantize, launch->argument, sizeof(quantize));
		expect(quantize.values_offset == main_x &&
			       quantize.width == MODEL_HIDDEN &&
			       quantize.codes_offset == codes &&
			       quantize.exponents_offset == exponents &&
			       quantize.operands_offset == operands,
		       "context norm quantizes into the draft arrays at hidden width");
		const uint64_t window_into =
			(uint64_t)first_row * MODEL_STATE_ROW_BYTES;
		check_context_product(
			launch_at(program, context_segment, at++), codes,
			exponents, operands,
			plan->layer_tensors[MODEL_DRAFT_FIRST_LAYER]
					   [V41_DATA_WINDOW]
						   .stored.byte_offset,
			arrays[V41_DATA_ARRAY_DRAFT_WINDOW_0] + window_into,
			plan->layer_tensors[MODEL_DRAFT_FIRST_LAYER + 1u]
					   [V41_DATA_WINDOW]
						   .stored.byte_offset,
			arrays[V41_DATA_ARRAY_DRAFT_WINDOW_1] + window_into,
			MODEL_HIDDEN, rows);
		check_context_product(
			launch_at(program, context_segment, at++), codes,
			exponents, operands,
			plan->layer_tensors[V41_DATA_WALK_LAYERS - 1u]
					   [V41_DATA_WINDOW]
						   .stored.byte_offset,
			arrays[V41_DATA_ARRAY_DRAFT_WINDOW_2] + window_into, 0,
			0, MODEL_HIDDEN, rows);
	}
	for (uint32_t stage = 0; stage < MODEL_DRAFT_STAGES; stage++) {
		struct attention_prepare_arguments window;
		const struct kernel_launch *const launch =
			launch_at(program, context_segment, at++);
		memcpy(&window, launch->argument, sizeof(window));
		expect(named(launch, KERNEL_ATTENTION_WINDOW) &&
			       window.step_offset ==
				       arrays[V41_DATA_ARRAY_STEP] &&
			       window.window_input_offset ==
				       arrays[V41_DATA_ARRAY_DRAFT_WINDOW_0 +
					      stage] &&
			       window.ring_offset ==
				       card->ring_offset[MODEL_DRAFT_FIRST_LAYER +
							 stage] &&
			       launch->grid[0] == program->row_count,
		       "context windows use all step positions and the corresponding draft rings");
	}
}

static void check_draft(const struct model_plan_record *plan,
			const struct execution_program *program,
			uint32_t arithmetic_class)
{
	const struct v41_card *const card = &plan->cards[RESULT_CARD];
	const uint64_t *const arrays = card->array_offset;
	const uint32_t first = program->draft_segment_first;
	expect(first + 1u == program->segment_count &&
		       program->segments[first].layer == SEGMENT_NO_LAYER &&
		       program->segments[first].card == RESULT_CARD &&
		       program->segments[first].pause == 0,
	       "draft is one tail segment on the result card without pause");
	for (uint32_t handoff = 0; handoff < program->hand_off_count;
	     handoff++) {
		expect(program->hand_offs[handoff].after_segment < first,
		       "no handoff follows any draft launch");
	}
	struct verify_matches_arguments verify;
	struct hyper_embed_arguments embed;
	memcpy(&verify, launch_at(program, first, 0)->argument, sizeof(verify));
	memcpy(&embed,
	       launch_at(program, first, DRAFT_PREFIX_LAUNCHES - 1u)->argument,
	       sizeof(embed));
	expect(named(launch_at(program, first, 0), KERNEL_VERIFY_MATCHES) &&
		       verify.step_offset == arrays[V41_DATA_ARRAY_STEP] &&
		       verify.result_offset == arrays[V41_DATA_ARRAY_RESULT] &&
		       verify.round_offset == arrays[V41_DATA_ARRAY_ROUND] &&
		       verify.draft_step_offset ==
			       arrays[V41_DATA_ARRAY_DRAFT_STEP] &&
		       verify.draft_views_offset ==
			       arrays[V41_DATA_ARRAY_DRAFT_VIEWS] &&
		       verify.row_count == program->row_count &&
		       verify.draft_positions == MODEL_DRAFT_BLOCK &&
		       verify.end_token_id == V41_DATA_END_TOKEN,
	       "verify connects target results to round and all draft step/view rows");
	expect(embed.step_offset == arrays[V41_DATA_ARRAY_DRAFT_STEP] &&
		       embed.embedding_offset ==
			       arrays[V41_DATA_ARRAY_DRAFT_EMBEDDING],
	       "draft embedding reads the card's local embedding copy and the verified anchor");
	uint32_t core_count = 0, router_count = 0, kept_count = 0;
	uint32_t site_count = 0, groups_count = 0, projection_count = 0,
		 join_count = 0;
	const struct segment_record *const segment = &program->segments[first];
	for (uint32_t index = DRAFT_PREFIX_LAUNCHES;
	     index < segment->launch_count - DRAFT_SUFFIX_LAUNCHES; index++) {
		const struct kernel_launch *const launch =
			launch_at(program, first, index);
		struct test_host_memory_backend_kernel kernel;
		const bool known = test_host_memory_backend_kernel_of(
			RESULT_CARD, launch->kernel, &kernel);
		expect(known, "draft launch has a prepared handle");
		if (!known) {
			continue;
		}
		const char *const name = kernel.function_name;
		if (strcmp(name, "site_front_8") == 0) {
			struct site_front_arguments front;
			memcpy(&front, launch->argument, sizeof(front));
			const uint32_t stage =
				site_count / DRAFT_SITES_PER_STAGE;
			const bool is_ffn =
				site_count % DRAFT_SITES_PER_STAGE != 0;
			expect(stage < MODEL_DRAFT_STAGES,
			       "two fronts per draft stage");
			if (stage < MODEL_DRAFT_STAGES) {
				const struct v41_tensor *const weights =
					plan->layer_tensors
						[MODEL_DRAFT_FIRST_LAYER +
						 stage];
				expect(front.select.row_count ==
						       MODEL_DRAFT_BLOCK &&
					       front.scratch_offset ==
						       card->front_scratch_offset &&
					       front.streams_offset ==
						       arrays[V41_DATA_ARRAY_STREAMS] &&
					       front.mix_weights_offset ==
						       weights[is_ffn ?
								       V41_DATA_FFN_MIX :
								       V41_DATA_ATTENTION_MIX]
							       .stored
							       .byte_offset &&
					       front.codes_offset ==
						       arrays[V41_DATA_ARRAY_ROW_CODES] &&
					       front.select.jobs_offset == 0,
				       "draft site front uses stage weights and scratch with separate routing");
			}
			site_count++;
		}
		if (strncmp(name, KERNEL_FFN_FRONT_STEM,
			    sizeof(KERNEL_FFN_FRONT_STEM) - 1u) == 0 ||
		    strncmp(name, KERNEL_EXPERT_GATE_UP_QUANTIZE_STEM,
			    sizeof(KERNEL_EXPERT_GATE_UP_QUANTIZE_STEM) - 1u) ==
			    0) {
			expect(false,
			       "draft native arithmetic never takes target FFN front or emulated producer");
		}
		if (strncmp(name, KERNEL_ATTENTION_GROUPS_STEM,
			    sizeof(KERNEL_ATTENTION_GROUPS_STEM) - 1u) == 0) {
			expect(arithmetic_class != ARITHMETIC_CLASS_SM_80 &&
				       module_is(
					       launch,
					       KERNEL_FILE_ATTENTION_OUTPUT_NATIVE),
			       "draft groups fusion uses native PRO arithmetic alone");
			groups_count++;
		}
		if (strncmp(name, KERNEL_ATTENTION_PROJECTION_STEM,
			    sizeof(KERNEL_ATTENTION_PROJECTION_STEM) - 1u) ==
		    0) {
			struct attention_projection_arguments projection;
			memcpy(&projection, launch->argument,
			       sizeof(projection));
			expect(projection.row_count == MODEL_DRAFT_BLOCK &&
				       projection.streams_offset ==
					       arrays[V41_DATA_ARRAY_STREAMS] &&
				       module_is(
					       launch,
					       arithmetic_class ==
							       ARITHMETIC_CLASS_SM_80 ?
						       KERNEL_FILE_ATTENTION_OUTPUT :
						       KERNEL_FILE_ATTENTION_OUTPUT_NATIVE),
			       "draft projection fusion posts its rows in native arithmetic");
			projection_count++;
		}
		if (strncmp(name, KERNEL_EXPERT_SHARED_JOIN_STEM,
			    sizeof(KERNEL_EXPERT_SHARED_JOIN_STEM) - 1u) == 0) {
			struct expert_shared_join_arguments join;
			memcpy(&join, launch->argument, sizeof(join));
			expect(join.row_count == MODEL_DRAFT_BLOCK &&
				       join.pairs_per_row ==
					       MODEL_DRAFT_EXPERTS_PER_ROW &&
				       join.row_sums_offset ==
					       arrays[V41_DATA_ARRAY_ROW_SUMS] &&
				       module_is(
					       launch,
					       arithmetic_class ==
							       ARITHMETIC_CLASS_SM_80 ?
						       KERNEL_FILE_TRUNK_LINEAR :
						       KERNEL_FILE_TRUNK_LINEAR_NATIVE),
			       "draft shared join keeps three-slot sums available to capture and uses native down");
			join_count++;
		}
		if (named(launch, KERNEL_ATTENTION_CORE_BLOCK)) {
			struct attention_core_arguments_record core;
			memcpy(&core, launch->argument, sizeof(core));
			expect(core_count < MODEL_DRAFT_STAGES &&
				       core.step_offset ==
					       arrays[V41_DATA_ARRAY_DRAFT_STEP] &&
				       core.pool_ratio == 0 &&
				       core.ring_offset ==
					       card->ring_offset
						       [MODEL_DRAFT_FIRST_LAYER +
							core_count] &&
				       module_is(
					       launch,
					       arithmetic_class ==
							       ARITHMETIC_CLASS_SM_80 ?
						       KERNEL_FILE_ATTENTION_CORE :
						       KERNEL_FILE_ATTENTION_CORE_NATIVE),
			       "draft core uses whole block windows, its own ring and native module");
			core_count++;
		}
		if (named(launch, KERNEL_ROUTER_SELECT_DRAFT)) {
			struct router_select_arguments router;
			memcpy(&router, launch->argument, sizeof(router));
			expect(router_count < MODEL_DRAFT_STAGES &&
				       router.step_offset ==
					       arrays[V41_DATA_ARRAY_DRAFT_STEP] &&
				       router.expert_offsets_offset ==
					       card->expert_table_offset
						       [MODEL_DRAFT_FIRST_LAYER +
							router_count] &&
				       router.reference_order == 0 &&
				       router.row_count == MODEL_DRAFT_BLOCK,
			       "draft routes 128 experts and three picks in fixed order");
			router_count++;
		}
		if (module_is(launch, KERNEL_FILE_TRUNK_LINEAR) &&
		    arithmetic_class != ARITHMETIC_CLASS_SM_80) {
			expect(false,
			       "the draft must not use emulated trunk products");
		}
	}
	expect(site_count == DRAFT_SITES_PER_STAGE * MODEL_DRAFT_STAGES &&
		       groups_count ==
			       (arithmetic_class == ARITHMETIC_CLASS_SM_80 ?
					0u :
					MODEL_DRAFT_STAGES) &&
		       projection_count == MODEL_DRAFT_STAGES &&
		       join_count == MODEL_DRAFT_STAGES,
	       "every draft stage takes every applicable fusion");
	const uint32_t head = segment->launch_count - DRAFT_SUFFIX_LAUNCHES;
	struct hyper_finish_arguments final;
	const struct kernel_launch *launch = launch_at(program, first, head);
	memcpy(&final, launch->argument, sizeof(final));
	kept_count += named(launch, KERNEL_HYPER_FINAL_KEPT) ? 1u : 0;
	expect(final.comb_offset == arrays[V41_DATA_ARRAY_DRAFT_COLLAPSE] &&
		       final.weight_offset ==
			       plan->draft_globals[V41_DATA_DRAFT_NORM]
				       .stored.byte_offset,
	       "draft final keeps the pre-norm collapse and uses mtp.2.norm");
	expect(core_count == MODEL_DRAFT_STAGES &&
		       router_count == MODEL_DRAFT_STAGES && kept_count == 1u,
	       "each draft stage has block attention and 128/3 routing, with one kept final");
	uint32_t at = head + DRAFT_HEAD_LAUNCHES;
	for (uint32_t position = 0; position < MODEL_DRAFT_BLOCK; position++) {
		struct draft_markov_bias_arguments markov;
		struct sampler_arguments sampler;
		launch = launch_at(program, first, at++);
		memcpy(&markov, launch->argument, sizeof(markov));
		expect(named(launch, KERNEL_DRAFT_MARKOV_BIAS) &&
			       markov.position == position &&
			       markov.draft_step_offset ==
				       arrays[V41_DATA_ARRAY_DRAFT_STEP] &&
			       markov.results_offset ==
				       arrays[V41_DATA_ARRAY_DRAFT_RESULTS] &&
			       markov.logits_offset ==
				       arrays[V41_DATA_ARRAY_DRAFT_LOGITS] +
					       (uint64_t)position *
						       MODEL_LOGITS_ROW_BYTES &&
			       markov.markov_offset ==
				       arrays[V41_DATA_ARRAY_DRAFT_MARKOV] +
					       (uint64_t)position *
						       DRAFT_MARKOV_ROW_BYTES,
		       "Markov bias reads preceding result before each coupled sample");
		launch = launch_at(program, first, at++);
		memcpy(&sampler, launch->argument, sizeof(sampler));
		expect(named(launch, KERNEL_SAMPLER) &&
			       sampler.session_offset ==
				       arrays[V41_DATA_ARRAY_SESSIONS] &&
			       launch->grid[1] == 1u &&
			       sampler.step_offset ==
				       arrays[V41_DATA_ARRAY_DRAFT_VIEWS] +
					       (uint64_t)position *
						       STEP_DRAFT_VIEW_BYTES &&
			       sampler.result_offset ==
				       arrays[V41_DATA_ARRAY_DRAFT_RESULTS] +
					       (uint64_t)position *
						       STEP_DRAFT_RESULT_BYTES &&
			       sampler.logits_offset == markov.logits_offset,
		       "sampler reads its one-row view and biased logits and writes its own result");
	}
	/* static: GCC stores this 64-byte record with an aligned AVX-512 move, and ASan's fake stack
	 * (detect_stack_use_after_return) gives a local only 32-byte alignment */
	static struct draft_confidence_arguments confidence;
	launch = launch_at(program, first, at);
	memcpy(&confidence, launch->argument, sizeof(confidence));
	expect(named(launch, KERNEL_DRAFT_CONFIDENCE) &&
		       at + 1u == segment->launch_count &&
		       confidence.collapse_offset ==
			       arrays[V41_DATA_ARRAY_DRAFT_COLLAPSE] &&
		       confidence.markov_offset ==
			       arrays[V41_DATA_ARRAY_DRAFT_MARKOV] &&
		       confidence.results_offset ==
			       arrays[V41_DATA_ARRAY_DRAFT_RESULTS] &&
		       confidence.round_offset == arrays[V41_DATA_ARRAY_ROUND],
	       "confidence completes proposals and scores in the same downloaded round");
}

static void run_case(const struct model_operations *model,
		     const struct pack_record *pack, uint32_t arithmetic_class,
		     enum linear_order_code order)
{
	static uint32_t placement[TEST_MODEL_V41_DRAFT_UNIT_COUNT];
	struct load_layout *layout = NULL;
	struct model_plan_record *plan = NULL;
	struct device_card *cards[CARDS] = { NULL };
	struct card_description_record descriptions[CARDS];
	const struct model_limits limits = { CONTEXT,
					     CAPACITY,
					     SESSIONS,
					     STATE_WIDTH_16,
					     LINEAR_PROFILE_PRECISION,
					     order,
					     CMP_ORDER_REFERENCE,
					     MODEL_DRAFT_BLOCK,
					     0,
					     0 };
	test_model_v41_five_cards(placement, 0, 0);
	test_model_v41_draft_place(placement, RESULT_CARD);
	result_status status = load_layout_create(
		pack, placement, COUNT_OF(placement), CARDS, &layout);
	status = status == STATUS_OK ?
			 model->plan_create(pack, layout, &limits, &plan) :
			 status;
	expect(status == STATUS_OK, "draft plan admitted");
	test_host_memory_backend_reset(CARDS, HOST_BYTES);
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card,
			card == RESULT_CARD ?
				arithmetic_class == ARITHMETIC_CLASS_SM_80 ?
				ARITHMETIC_CLASS_SM_80 :
				ARITHMETIC_CLASS_SM_120 :
				ARITHMETIC_CLASS_SM_80);
		expect(test_host_memory_backend.card_describe(
			       card, &descriptions[card]) == STATUS_OK &&
			       test_host_memory_backend.card_open(
				       card, HOST_BYTES, &cards[card]) ==
				       STATUS_OK,
		       "table-only card opened");
	}
	/* The backend's class names the physical architecture; the model description selects native arithmetic. */
	descriptions[RESULT_CARD].arithmetic_class = arithmetic_class;
	const struct {
		uint32_t kind, rows;
	} cases[] = { { PROGRAM_THIN, 1u },
		      { PROGRAM_THIN, LINEAR_MAXIMUM_ROWS },
		      { PROGRAM_FULL, WIDE_ROWS },
		      { PROGRAM_FULL, FULL_ROWS },
		      { PROGRAM_WIDE, WIDE_ROWS } };
	for (uint32_t index = 0; plan != NULL && index < COUNT_OF(cases);
	     index++) {
		struct execution_program *program = NULL;
		status = model->program_prepare(
			plan, &test_host_memory_backend, cards, descriptions,
			(enum program_kind)cases[index].kind, cases[index].rows,
			&program);
		expect(status == STATUS_OK && program != NULL,
		       "draft-enabled program prepared");
		if (program == NULL) {
			continue;
		}
		if (program->kind != PROGRAM_WIDE) {
			check_taps(plan, program);
			const uint32_t context_segment =
				program->kind == PROGRAM_FULL ?
					program->segment_count - 1u :
					program->draft_segment_first -
						CONTEXT_TO_DRAFT_SEGMENTS;
			check_context(plan, program, context_segment);
		}
		if (program->kind == PROGRAM_THIN) {
			check_draft(plan, program, arithmetic_class);
			expect(program->cards[RESULT_CARD].round_offset ==
				       plan->cards[RESULT_CARD].array_offset
					       [V41_DATA_ARRAY_ROUND],
			       "thin round is on the result card");
		} else {
			expect(program->draft_segment_first ==
					       program->segment_count &&
				       program->cards[RESULT_CARD].round_offset ==
					       PROGRAM_NOWHERE,
			       "chunk programs have no conditional draft tail or round");
		}
		for (uint32_t card = 0; card < CARDS; card++) {
			struct test_host_memory_backend_kernel kernel;
			uint32_t count = 0;
			while (count < BACKEND_MAXIMUM_KERNELS &&
			       test_host_memory_backend_kernel_of(card, count,
								  &kernel)) {
				count++;
			}
			maximum_prepared_kernels =
				count > maximum_prepared_kernels ?
					count :
					maximum_prepared_kernels;
		}
		model->program_destroy(program);
	}
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend.card_close(cards[card], 0);
	}
	model->plan_destroy(plan);
	load_layout_destroy(layout);
}

int main(void)
{
	struct pack_record *pack = NULL;
	test_memory_storage_reset();
	expect(test_model_v41_draft_pack_open(
		       (struct test_model_v41_change){
			       TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
		       &pack) == STATUS_OK,
	       "draft table-only pack opens");
	const struct model_operations *const model =
		model_operations_of(V41_DATA_MODEL_NAME);
	expect(model != NULL, "model operations available");
	const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
				     ARITHMETIC_CLASS_SM_120_NATIVE,
				     ARITHMETIC_CLASS_SM_120 };
	for (uint32_t index = 0;
	     pack != NULL && model != NULL && index < COUNT_OF(classes);
	     index++) {
		run_case(model, pack, classes[index], LINEAR_ORDER_FIXED);
		run_case(model, pack, classes[index], LINEAR_ORDER_REFERENCE);
	}
	pack_close(pack);
	test_memory_storage_reset();
	(void)printf("draft fusion programs: maximum prepared kernels %" PRIu32
		     " of %d\n",
		     maximum_prepared_kernels, BACKEND_MAXIMUM_KERNELS);
	return failures != 0 ? 1 : 0;
}
