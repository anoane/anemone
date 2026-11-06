/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Four actual host-backed cards prepare immutable EXL3 TP programs. No launch or model payload executes. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/dense_linear_records.h"
#include "anemone/dense_stream_records.h"
#include "anemone/draft_records.h"
#include "anemone/draft_share_records.h"
#include "anemone/exact_ring_records.h"
#include "anemone/exl3_records.h"
#include "anemone/hyper_connection_records.h"
#include "anemone/linear_records.h"
#include "anemone/sampler_records.h"
#include "anemone/sampler_shard_records.h"
#include "anemone/tp_exchange_records.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41_pack.h"
#include "model_v41_tp2_model.h"

#define EXL3_CARDS TP4_RANKS
#define EXL3_DRAFT 7u
#define EXL3_CARD_BYTES 4096u
#define EXL3_REPORTED_BYTES (1ull << 40u)
#define EXL3_WAIT_NS 1000000000ull
#define EXL3_CHECKPOINTS (EXL3_CARDS * MODEL_LAYERS + MODEL_DRAFT_STAGES)
#define EXL3_EMBED_COPIES (2u * (EXL3_CARDS - 1u))
#define EXL3_CONE_SEGMENTS (1u + EXL3_CARDS)
#define EXL3_FULL_SEGMENTS (1u + 2u * EXL3_CARDS + 1u)
#define EXL3_THIN_SEGMENTS (EXL3_FULL_SEGMENTS + 2u)
#define EXL3_UNTOUCHED_PAIRS 123u
#define EXL3_SECOND_CARD 1u
#define EXL3_BAD_DOWN_DELTA 128u
#define EXL3_THIN_SIX 6u
#define EXL3_SESSIONS 4u
#define EXL3_C8_SESSIONS 8u
#define EXL3_FULL_TARGET_GROUPS 2u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "model_v41_exl3_program_test: %s\n",
			      message);
		failures++;
	}
}

/* Every launch of every card names a kernel prepared on that card, with that kernel's record size. */
/* PROGRAM_HISTOGRAM (debug): each card's kernel launches of a program by function name, on the standard error. */
static void program_histogram(const struct execution_program *program,
			      const char *what)
{
	enum { KINDS = 128, NAME = 96 };
	if (getenv("PROGRAM_HISTOGRAM") == NULL || program == NULL)
		return;
	for (uint32_t card = 0; card < program->card_count; card++) {
		static char names[KINDS][NAME];
		uint32_t counts[KINDS] = { 0 }, kinds = 0;
		for (uint32_t index = 0; index < program->cards[card].launch_count; index++) {
			struct test_host_memory_backend_kernel kernel;
			if (!test_host_memory_backend_kernel_of(card, program->cards[card].launches[index].kernel, &kernel))
				continue;
			uint32_t at = 0;
			while (at < kinds && strncmp(names[at], kernel.function_name, NAME - 1u) != 0)
				at++;
			if (at == kinds && kinds < KINDS)
				(void)snprintf(names[kinds++], NAME, "%s", kernel.function_name);
			if (at < kinds)
				counts[at]++;
		}
		(void)fprintf(stderr, "%s card %u launches %u segments %u\n", what, card,
			      program->cards[card].launch_count, program->segment_count);
		for (uint32_t at = 0; at < kinds; at++)
			(void)fprintf(stderr, "  %6u %s\n", counts[at], names[at]);
	}
}

static void check_handles(const struct execution_program *program,
			  const char *what)
{
	for (uint32_t card = 0; card < program->card_count; card++) {
		for (uint32_t index = 0; index < program->cards[card].launch_count; index++) {
			const struct kernel_launch *const launch =
				&program->cards[card].launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (launch->kernel == BACKEND_COPY)
				continue;
			const bool known = test_host_memory_backend_kernel_of(card, launch->kernel, &kernel);
			if (!known || kernel.argument_byte_count != launch->argument_byte_count) {
				(void)fprintf(stderr, "card %u launch %u kernel %u (%s %u bytes, launch %u bytes): %s\n", card,
					      index, launch->kernel, known ? kernel.function_name : "unknown",
					      known ? kernel.argument_byte_count : 0u, launch->argument_byte_count, what);
				expect(false, "a launch names a kernel of its card with its record's size");
				return;
			}
		}
	}
}

static void refused(const struct v41_tp2_program_description *description,
		    struct device_card *const *cards,
		    const struct card_description_record *reported)
{
	struct execution_program *program = NULL;
	struct step_companion_program companions = {
		.pair_count = EXL3_UNTOUCHED_PAIRS
	};
	const result_status status = v41_tp2_program_prepare(
		description, &test_host_memory_backend, cards, reported,
		EXL3_THIN_SIX, &program, &companions);
	expect(status == STATUS_INVALID_ARGUMENT && program == NULL &&
		       companions.pair_count == EXL3_UNTOUCHED_PAIRS &&
		       companions.program == NULL,
	       "malformed topology refuses without publishing either output");
}

static void check_refusals(struct v41_tp2_model *model,
			   struct v41_tp2_program_description description,
			   struct device_card *const *cards,
			   const struct card_description_record *reported)
{
	description.stages[EXL3_CARDS] = description.stages[0];
	refused(&description, cards, reported);
	description.stages[EXL3_CARDS] = NULL;
	description.views[EXL3_CARDS] = description.views[0];
	refused(&description, cards, reported);
	description.views[EXL3_CARDS] = NULL;
	description.stages[EXL3_CARDS - 1u] = NULL;
	refused(&description, cards, reported);
	description.stages[EXL3_CARDS - 1u] = model->stages[EXL3_CARDS - 1u];
	model->canonical->result_card = EXL3_CARDS;
	refused(&description, cards, reported);
	model->canonical->result_card = 0;
	model->stages[EXL3_SECOND_CARD]->description.rank = 0;
	refused(&description, cards, reported);
	model->stages[EXL3_SECOND_CARD]->description.rank = EXL3_SECOND_CARD;
	model->stages[EXL3_SECOND_CARD]->description.original_code_wire = true;
	refused(&description, cards, reported);
	model->stages[EXL3_SECOND_CARD]->description.original_code_wire = false;
	model->stages[EXL3_SECOND_CARD]->description.exl3_all_targets = false;
	refused(&description, cards, reported);
	model->stages[EXL3_SECOND_CARD]->description.exl3_all_targets = true;
	model->stages[0]->exl3_shards[0].down.output_count +=
		EXL3_BAD_DOWN_DELTA;
	refused(&description, cards, reported);
	model->stages[0]->exl3_shards[0].down.output_count -=
		EXL3_BAD_DOWN_DELTA;
	model->views[0]->card_count++;
	refused(&description, cards, reported);
	model->views[0]->card_count--;
}

/* Match the five corrected products by immutable weight ownership, independently of launch ordering. */
static void check_bfloat16_attention(const struct v41_tp2_model *model,
				     const struct execution_program *program)
{
	static const uint32_t matrices[] = { V41_DATA_QUERY_DOWN,
					     V41_DATA_WINDOW, V41_DATA_QUERY_UP,
					     V41_DATA_INDEX_QUERY_UP,
					     V41_DATA_OUTPUT };
	static const uint32_t inputs[] = { V41_DATA_ARRAY_HIDDEN,
					   V41_DATA_ARRAY_HIDDEN,
					   V41_DATA_ARRAY_QUERY_NORMED,
					   V41_DATA_ARRAY_QUERY_NORMED,
					   V41_DATA_ARRAY_GROUPS };
	static const uint32_t outputs[] = { V41_DATA_ARRAY_QUERY_RANK,
					    V41_DATA_ARRAY_WINDOW_INPUT,
					    V41_DATA_ARRAY_HEADS,
					    V41_DATA_ARRAY_INDEX_HEADS,
					    V41_DATA_ARRAY_SUBLAYER_OUTPUT };
	enum {
		PROJECTIONS = sizeof(matrices) / sizeof(matrices[0]),
		INDEX_PROJECTION = 3u,
		INPUT_PROJECTIONS = 2u
	};
	const bool wide = program->kind == PROGRAM_WIDE;
	const uint32_t complete = wide ? MODEL_DECODER_FIRST_LAYER :
					 MODEL_LAYERS;
	const uint64_t complete_mask = (1ull << complete) - 1u;
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		uint32_t counts[PROJECTIONS] = { 0 };
		const struct model_plan_record *const view = model->views[card];
		const struct program_card_record *const local =
			&program->cards[card];
		for (uint32_t index = 0; index < local->launch_count; index++) {
			const struct kernel_launch *const launch =
				&local->launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (!test_host_memory_backend_kernel_of(
				    card, launch->kernel, &kernel))
				continue;
			const char *const thin_stem =
				KERNEL_TRUNK_LINEAR_BFLOAT16_STEM;
			if (strncmp(kernel.function_name, thin_stem,
				    strlen(thin_stem)) != 0)
				continue;
			struct trunk_linear_bfloat16_arguments record;
			memcpy(&record, launch->argument, sizeof(record));
			for (uint32_t layer = 0; layer < MODEL_LAYERS;
			     layer++) {
				for (uint32_t projection = 0;
				     projection < PROJECTIONS; projection++) {
					const struct v41_tensor *const matrix =
						&view->layer_tensors
							 [layer]
							 [matrices[projection]];
					if (matrix->stored.byte_count == 0 ||
					    record.weights_offset !=
						    matrix->stored.byte_offset)
						continue;
					counts[projection]++;
					const char *const expected =
						program->kind == PROGRAM_THIN ?
							KERNEL_TRUNK_LINEAR_BFLOAT16_STEM
							"4" :
							KERNEL_TRUNK_LINEAR_BFLOAT16_WIDE_STEM
							"4";
					expect(strcmp(kernel.function_name,
						      expected) == 0 &&
						       record.values_offset ==
							       view->cards[card].array_offset
								       [inputs[projection]] &&
						       record.output_offset ==
							       (!model->description.exact_output_shards &&
										matrices[projection] ==
											V41_DATA_QUERY_UP ?
									model->stages[card]
										->compact
										.byte_offset :
									view->cards[card]
										.array_offset
											[outputs[projection]]) &&
						       record.input_size ==
							       matrix->shape
								       .input_size &&
						       record.output_size ==
							       matrix->shape
								       .output_size &&
						       record.segment_count ==
							       1u &&
						       record.row_count ==
							       program->row_count,
					       "every replicated rank preserves BF16 attention inputs "
					       "and four ordered slices");
				}
			}
		}
		for (uint32_t projection = 0; projection < PROJECTIONS;
		     projection++) {
			const uint32_t expected =
				projection == INDEX_PROJECTION ?
					(uint32_t)__builtin_popcountll(
						V41_DATA_INDEX_LAYERS &
						complete_mask) :
					complete +
						(wide && projection < INPUT_PROJECTIONS ?
							 1u :
							 0u);
			expect(counts[projection] == expected,
			       "all five corrected projection families cover each actual rank");
		}
	}
}

static void check_program(const struct v41_tp2_model *model,
			  const struct execution_program *program,
			  const struct step_companion_program *companions)
{
	const bool wide = program->kind == PROGRAM_WIDE;
	const bool thin = program->kind == PROGRAM_THIN;
	check_bfloat16_attention(model, program);
	expect(program->card_count == EXL3_CARDS && program->result_card == 0 &&
		       program->checkpoint_count == EXL3_CHECKPOINTS,
	       "every kind retains four-card metadata and all163 replica/draft checkpoints");
	expect(program->front_segment_count == EXL3_CONE_SEGMENTS &&
		       program->segment_count == (wide ? EXL3_CONE_SEGMENTS :
						  thin ? EXL3_THIN_SEGMENTS :
							 EXL3_FULL_SEGMENTS) &&
		       companions->pair_count ==
			       (wide ? 1u : EXL3_FULL_TARGET_GROUPS),
	       "target graphs merge at the source front and leave native result/draft segments separate");
	expect(program->hand_off_count == EXL3_EMBED_COPIES &&
		       companions->arithmetic_identity ==
			       model->recipe_identity,
	       "only initial embedding residual copies remain and order identities match");
	for (uint32_t index = 0; index < program->hand_off_count; index++) {
		const struct hand_off_record *const transfer =
			&program->hand_offs[index];
		expect(transfer->from_card == 0 && transfer->to_card > 0 &&
			       transfer->to_card < EXL3_CARDS &&
			       transfer->after_segment == 0,
		       "all transfers precede the target groups and name a real CMP");
	}
	for (uint32_t group = 0; group < companions->pair_count; group++) {
		const struct step_companion_pair *const pair =
			&companions->pairs[group];
		expect(pair->second_segment - pair->first_segment + 1u ==
			       EXL3_CARDS,
		       "each companion group has exactly four participants");
		for (uint32_t rank = 0; rank < EXL3_CARDS; rank++)
			expect(program->segments[pair->first_segment + rank]
					       .card == rank,
			       "group ordering uses physical CMP ranks0..3");
	}
	const uint32_t target_end =
		companions->pairs[companions->pair_count - 1u].second_segment +
		1u;
	for (uint32_t segment = target_end; segment < program->segment_count;
	     segment++) {
		const struct segment_record *const tail =
			&program->segments[segment];
		expect(tail->card == 0,
		       "context, target head and full native draft execute on CMP0");
		if (tail->card != 0)
			continue;
		for (uint32_t index = tail->first_launch;
		     index < tail->first_launch + tail->launch_count; index++) {
			struct test_host_memory_backend_kernel kernel;
			const struct kernel_launch *const launch =
				&program->cards[0].launches[index];
			expect(test_host_memory_backend_kernel_of(
				       0, launch->kernel, &kernel) &&
				       strcmp(kernel.kernel_file_name,
					      EXL3_KERNEL_FILE) != 0 &&
				       strcmp(kernel.function_name,
					      KERNEL_EXACT_RING) != 0 &&
				       strcmp(kernel.function_name,
					      KERNEL_EXACT_RING_UNEQUAL) != 0,
			       "full native draft and head never borrow EXL3 target shards or gather handles");
		}
	}
	expect(program->draft_segment_first ==
		       (thin ? program->segment_count - 1u :
			       program->segment_count),
	       "only thin programs publish a separate draft tail");
	uint32_t taps = 0;
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		const struct program_card_record *const local =
			&program->cards[card];
		expect((local->round_offset != PROGRAM_NOWHERE) ==
			       (thin && card == 0),
		       "only CMP0 thin programs publish a draft round record");
		for (uint32_t index = 0; index < local->launch_count; index++) {
			const struct kernel_launch *const launch =
				&local->launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (!test_host_memory_backend_kernel_of(
				    card, launch->kernel, &kernel)) {
				expect(false,
				       "each launch references an immutable prepared kernel");
				continue;
			}
			if (!model->description.exact_output_shards) {
				expect(strcmp(kernel.function_name,
					      KERNEL_EXACT_RING_UNEQUAL) != 0,
				       "approximate inner shards never gather full hidden operands");
				if (strcmp(kernel.function_name,
					   KERNEL_EXACT_RING) == 0) {
					struct exact_ring_arguments record;
					memcpy(&record, launch->argument,
					       sizeof(record));
					expect(record.mode ==
						       EXACT_RING_BF16_SUM,
					       "the explicit approximate mode selects BF16 ring sums only");
				}
			}

			if (strcmp(kernel.function_name,
				   KERNEL_HYPER_DRAFT_INPUTS) == 0) {
				struct hyper_draft_inputs_arguments record;
				memcpy(&record, launch->argument,
				       sizeof(record));
				if (record.mode == HYPER_STREAM_MEAN) {
					expect(card == 0 && !wide &&
						       record.column_offset ==
							       taps * MODEL_HIDDEN,
					       "three ordered stream taps are local to CMP0");
					taps++;
				}
			}
		}
	}
	expect(taps == (wide ? 0u : MODEL_DRAFT_TAPS),
	       "the target captures every required draft tap once");
	for (uint32_t index = 0; index < program->checkpoint_count; index++)
		expect(program->checkpoints[index].card < EXL3_CARDS,
		       "every checkpoint names a real CMP");
	for (uint32_t index = 0; index < program->session_range_count;
	     index++) {
		const struct session_range *const range =
			&program->session_ranges[index];
		expect(range->card < EXL3_CARDS,
		       "every snapshot range names a real CMP");
		if (range->card < EXL3_CARDS) {
			const uint64_t arena =
				program->cards[range->card].arena_byte_count;
			const uint64_t bytes =
				range->unit_count * range->unit_bytes +
				(uint64_t)(program->limits.session_capacity -
					   1u) *
					range->session_stride;
			expect(range->offset <= arena &&
				       bytes <= arena - range->offset,
			       "replicated session ranges remain within each actual arena");
		}
	}
}

struct load_failure_state {
	struct device_card *cards[EXL3_CARDS];
	uint32_t allocations, retired, waited, wait_mask;
};
static struct load_failure_state load_state;
static _Alignas(BACKEND_STAGING_ALIGNMENT_BYTES) unsigned char load_staging;

MUST_CHECK static result_status load_allocate_fail(uint64_t bytes,
						   void **memory_out)
{
	(void)bytes;
	load_state.allocations++;
	if (load_state.allocations == 1u) {
		*memory_out = &load_staging;
		return STATUS_OK;
	}
	return STATUS_INJECTED_FAULT;
}

static void load_retire(void *memory)
{
	expect(memory == NULL || memory == &load_staging,
	       "only owned staging is retired after failed load");
	load_state.retired++;
}

MUST_CHECK static result_status load_wait(struct device_card *card,
					  uint64_t wait)
{
	expect(wait == EXL3_WAIT_NS,
	       "every load drain retains the bounded wait");
	load_state.waited++;
	bool found = false;
	for (uint32_t rank = 0; rank < EXL3_CARDS; rank++) {
		if (load_state.cards[rank] == card) {
			expect((load_state.wait_mask >> rank & 1u) == 0,
			       "each actual card drains once");
			load_state.wait_mask |= 1u << rank;
			found = true;
		}
	}
	expect(found, "failed load never waits on an inactive fifth card");
	return STATUS_TIMED_OUT;
}

MUST_CHECK static result_status load_peer_never(struct device_card *card,
						struct device_card *owner,
						uint64_t owner_offset,
						uint64_t address_word_offset)
{
	(void)card;
	(void)owner;
	(void)owner_offset;
	(void)address_word_offset;
	expect(false,
	       "no peer connection precedes the injected workspace failure");
	return STATUS_DEVICE_ERROR;
}

static void check_load_admission(struct v41_tp2_model *model,
				 struct device_card *const *cards)
{
	const char *const paths[] = { "tables-only" };
	struct storage_reader *reader = NULL;
	const result_status opened =
		test_memory_storage.reader_open(paths, 1, &reader);
	expect(opened == STATUS_OK,
	       "metadata reader opens for load admission controls");
	if (opened != STATUS_OK)
		return;
	load_state = (struct load_failure_state){ 0 };
	memcpy(load_state.cards, cards, sizeof(load_state.cards));
	struct backend_operations backend = test_host_memory_backend;
	backend.staging_allocate = load_allocate_fail;
	backend.staging_retire = load_retire;
	backend.launch_wait = load_wait;
	backend.peer_connect = load_peer_never;
	struct v41_tp2_model_load_request request = {
		.backend = &backend,
		.cards = cards,
		.storage = &test_memory_storage,
		.reader = reader,
		.wait_limit_nanoseconds = EXL3_WAIT_NS
	};
	expect(v41_tp2_model_load(model, &request) == STATUS_INVALID_ARGUMENT &&
		       load_state.allocations == 0 && !model->load_attempted,
	       "unqualified quad refuses before any callback");
	for (uint32_t card = 0; card < EXL3_CARDS; card++)
		request.qualified_writer_masks[card] = (1u << EXL3_CARDS) - 1u;
	struct device_card *missing[EXL3_CARDS];
	memcpy(missing, cards, sizeof(missing));
	missing[EXL3_CARDS - 1u] = NULL;
	request.cards = missing;
	expect(v41_tp2_model_load(model, &request) == STATUS_INVALID_ARGUMENT &&
		       load_state.allocations == 0 && !model->load_attempted,
	       "missing fourth card refuses before any callback");
	request.cards = cards;
	backend.staging_retire = NULL;
	expect(v41_tp2_model_load(model, &request) == STATUS_INVALID_ARGUMENT &&
		       load_state.allocations == 0 && !model->load_attempted,
	       "missing safe retirement refuses before any callback");
	backend.staging_retire = load_retire;
	request.qualified_writer_masks[0] &= ~(1u << EXL3_SECOND_CARD);
	expect(v41_tp2_model_load(model, &request) == STATUS_INVALID_ARGUMENT &&
		       load_state.allocations == 0 && !model->load_attempted,
	       "one missing remote-writer edge refuses before work");
	request.qualified_writer_masks[0] = (1u << EXL3_CARDS) - 1u;
	expect(v41_tp2_model_load(model, &request) == STATUS_INJECTED_FAULT &&
		       load_state.allocations == TP2_RANKS &&
		       load_state.retired == TP2_RANKS &&
		       load_state.waited == EXL3_CARDS &&
		       load_state.wait_mask == (1u << EXL3_CARDS) - 1u &&
		       model->load_attempted && !model->ready &&
		       model->host_arena == NULL && model->host_offsets == NULL,
	       "failed quad load drains four actual cards, retires staging and publishes no ready state");
	expect(v41_tp2_model_load(model, &request) == STATUS_INVALID_ARGUMENT &&
		       load_state.allocations == TP2_RANKS &&
		       load_state.waited == EXL3_CARDS,
	       "a failed load object remains single attempt");
	test_memory_storage.reader_close(reader);
}

static void check_case(const struct pack_record *pack, uint32_t capacity,
		       enum bb_exl3_product_order order, bool exact)
{
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES, capacity,
			    EXL3_SESSIONS, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    CMP_ORDER_REFERENCE,
			    exact ? EXL3_DRAFT : MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.exact_output_shards = exact,
		.wire_format = exact ? 0u : ORDERED_PUSH_BF16,
		.exl3_product_order = order
	};
	/* These arrays have four elements so ASan detects any accidental fifth-card read. */
	struct device_card *cards[EXL3_CARDS] = { 0 };
	struct card_description_record reported[EXL3_CARDS] = { 0 };
	test_host_memory_backend_reset(EXL3_CARDS, EXL3_CARD_BYTES);
	bool opened = true;
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes = EXL3_REPORTED_BYTES;
		reported[card].memory_free_bytes = EXL3_REPORTED_BYTES;
		description.cards[card] = reported[card];
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		opened = test_host_memory_backend.card_open(
				 card, EXL3_CARD_BYTES, &cards[card]) ==
				 STATUS_OK &&
			 opened;
	}
	struct v41_tp2_model *model = NULL;
	result_status status =
		opened ? v41_tp2_model_create(pack, &description, &model) :
			 STATUS_OUT_OF_MEMORY;
	expect(status == STATUS_OK && model != NULL,
	       "four-card EXL3 model metadata prepares");
	if (model != NULL) {
		struct v41_tp2_program_description build = { 0 };
		status = v41_tp2_model_program_description(model, &build);
		expect(status == STATUS_OK &&
			       build.canonical == model->canonical,
		       "program matrix uses the public four-card description");
		check_refusals(model, build, cards, reported);
		struct v41_tp2_model_description wrong_mode = description;
		struct v41_tp2_model *untouched = model;
		wrong_mode.wire_format = exact ? ORDERED_PUSH_BF16 : 0u;
		expect(v41_tp2_model_create(pack, &wrong_mode, &untouched) ==
				       STATUS_INVALID_ARGUMENT &&
			       untouched == model,
		       "exact/gather and approximate/BF16 mode pairs cannot be mixed");
		if (!exact) {
			wrong_mode = description;
			wrong_mode.exl3_product_order =
				BB_EXL3_EIGHT_PART_ORDER;
			expect(v41_tp2_model_create(pack, &wrong_mode,
						    &untouched) ==
					       STATUS_INVALID_ARGUMENT &&
				       untouched == model,
			       "Stage1 approximate products retain the original four-part order");
		}
		const struct pack_tensor *const tensors = pack_tensors_of(pack);
		for (uint32_t index = 0;
		     index < model->gpu_load.description.job_count; index++) {
			const struct tp_load_job *const job =
				&model->jobs[index];
			const struct pack_tensor *const source =
				&tensors[job->tensor_index];
			if (source->weight_format != WEIGHT_FORMAT_EXL3)
				continue;
			const uint32_t inner = job->card % TP2_RANKS == 0 ?
						       TP4_SMALL_INNER :
						       TP4_LARGE_INNER;
			const bool down = source->dimensions[0] == MODEL_HIDDEN;
			expect(job->mode == TP_LOAD_SHARD &&
				       job->shard.shard_shape.output_size ==
					       (down ? (exact ? MODEL_HIDDEN /
									TP4_RANKS :
								MODEL_HIDDEN) :
						       inner) &&
				       job->shard.shard_shape.input_size ==
					       (down ? (exact ? V41_DATA_EXPERT_INNER :
								inner) :
						       MODEL_HIDDEN),
			       "source jobs preserve exact full-K output shards or explicit local-K partials");
		}

		const enum program_kind kinds[] = { PROGRAM_THIN, PROGRAM_WIDE,
						    PROGRAM_FULL };
		for (uint32_t kind = 0; kind < sizeof(kinds) / sizeof(kinds[0]);
		     kind++) {
			if (kinds[kind] != PROGRAM_THIN &&
			    capacity < MODEL_WIDE_LEAST_ROWS)
				continue;
			const uint32_t rows = kinds[kind] == PROGRAM_THIN ?
						      EXL3_THIN_SIX :
						      capacity;
			struct execution_program *program = NULL;
			struct step_companion_program companions = { 0 };
			status = v41_tp_program_prepare(
				&build, &test_host_memory_backend, cards,
				reported, kinds[kind], rows, &program,
				&companions);
			expect(status == STATUS_OK && program != NULL,
			       "internal four-CMP program prepares");
			if (program != NULL) {
				check_program(model, program, &companions);
				check_handles(program, "four-CMP program");
			}
			free(program);
		}
		struct v41_tp2_program_description published = { 0 };
		const struct v41_tp2_model_load_request invalid_load = { 0 };
		expect(v41_tp2_model_program_description(model, &published) ==
				       STATUS_OK &&
			       published.canonical == model->canonical &&
			       published.slice_count == TP2_DEFAULT_SLICES &&
			       published.views[EXL3_CARDS] == NULL &&
			       published.stages[EXL3_CARDS] == NULL,
		       "public four-card descriptions retain a NULL inactive fifth slot");
		for (uint32_t card = 0; card < EXL3_CARDS; card++)
			expect(published.views[card] == model->views[card] &&
				       published.stages[card] ==
					       model->stages[card],
			       "published views and stage metadata name every actual CMP");
		expect(v41_tp2_model_load(model, &invalid_load) ==
				       STATUS_INVALID_ARGUMENT &&
			       !model->load_attempted,
		       "invalid load refuses before consuming the one permitted attempt");
		check_load_admission(model, cards);
	}
	expect(test_host_memory_backend_work_count() == 0,
	       "host planning queues no GPU or payload work");
	v41_tp2_model_destroy(model);
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card],
							    EXL3_WAIT_NS);
	}
}

/* head_shards (model_v41_tp2_stage.h): ranks 1-3 load the norm and a head copy; a thin program ends with each
 * rank's final, head cut (dense_stream's kernel with head_stream) and sampler shard; wide/full programs take none. */
static void check_head_shards(const struct pack_record *pack, uint32_t capacity,
			      bool stream, bool exact, bool drafts)
{
	const uint32_t shard = MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS;
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES, capacity,
			    EXL3_SESSIONS, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    CMP_ORDER_REFERENCE,
			    exact ? EXL3_DRAFT : MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.exact_output_shards = exact,
		.wire_format = exact ? 0u : ORDERED_PUSH_BF16,
		.head_shards = true,
		.head_stream = stream,
		.draft_shards = drafts,
		.exl3_fast_products = true,
		.exl3_wide_entry = true,
		.bf16_slab_rings = !exact && drafts,
		.wide_exchange = true,
		.replicated_embedding = stream,
		.exact_dense = exact && drafts
	};
	struct device_card *cards[EXL3_CARDS] = { 0 };
	struct card_description_record reported[EXL3_CARDS] = { 0 };
	test_host_memory_backend_reset(EXL3_CARDS, EXL3_CARD_BYTES);
	bool opened = true;
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes = reported[card].memory_free_bytes = EXL3_REPORTED_BYTES;
		description.cards[card] = reported[card];
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		opened = test_host_memory_backend.card_open(
				 card, EXL3_CARD_BYTES, &cards[card]) ==
				 STATUS_OK &&
			 opened;
	}
	struct v41_tp2_model *model = NULL;
	struct v41_tp2_model_description refused_description = description;
	refused_description.tensor_layout = V41_DATA_TP_LAYOUT_QUAD;
	expect(opened &&
		       v41_tp2_model_create(pack, &refused_description,
					    &model) ==
			       STATUS_INVALID_ARGUMENT &&
		       model == NULL,
	       "head shards refuse layouts other than QUAD_ALL");
	expect(v41_tp2_model_create(pack, &description, &model) == STATUS_OK &&
		       model != NULL,
	       "head shards: four-card metadata prepares");
	if (model != NULL) {
		uint32_t head_jobs = 0;
		const uint32_t head_tensor = pack_find_tensor(
			pack, "head.weight", (uint32_t)strlen("head.weight"));
		for (uint32_t index = 0; index < model->gpu_load.description.job_count; index++) {
			const struct tp_load_job *const job =
				&model->jobs[index];
			if (job->tensor_index != head_tensor)
				continue;
			head_jobs++;
			const bool cut = job->card != 0 || drafts;
			expect(cut ? job->mode == TP_LOAD_SHARD &&
					       job->shard.shard_shape
							       .output_size ==
						       shard &&
					       job->shard.description
							       .first_output ==
						       job->card * shard :
				     job->mode == TP_LOAD_FULL,
			       "ranks 1-3 load their quarter of the head's rows, rank 0 too where the draft is cut");
		}
		for (uint32_t card = 1; card < EXL3_CARDS; card++)
			expect(model->views[card]->globals[V41_DATA_HEAD].card ==
					       card &&
				       model->views[card]
						       ->globals[V41_DATA_HEAD]
						       .shape.output_size ==
					       shard &&
				       model->views[card]
						       ->globals[V41_DATA_NORM]
						       .card == card &&
				       model->stages[card]
						       ->sampler_exchange
						       .byte_count ==
					       SAMPLER_SHARD_EXCHANGE_BYTES,
			       "ranks 1-3 hold their own norm, head copy and sampler areas");
		expect(head_jobs == EXL3_CARDS, "one head job a rank");
		uint32_t embed_jobs = 0;
		const uint32_t embed_tensor = pack_find_tensor(
			pack, "embed.weight", (uint32_t)strlen("embed.weight"));
		for (uint32_t index = 0; index < model->gpu_load.description.job_count; index++)
			embed_jobs += model->jobs[index].tensor_index == embed_tensor;
		expect(embed_jobs == (description.replicated_embedding ? EXL3_CARDS : 1u),
		       "replicated_embedding: every rank loads the embedding, else card 0 alone");
		struct v41_tp2_program_description build = { 0 };
		expect(v41_tp2_model_program_description(model, &build) ==
			       STATUS_OK,
		       "head shards: program description");
		const enum program_kind kinds[] = { PROGRAM_THIN, PROGRAM_WIDE,
						    PROGRAM_FULL };
		struct execution_program
			*programs[sizeof(kinds) / sizeof(kinds[0])] = { 0 };
		uint32_t program_count = 0;
		for (uint32_t kind = 0; kind < sizeof(kinds) / sizeof(kinds[0]); kind++) {
			if (kinds[kind] != PROGRAM_THIN &&
			    capacity < MODEL_WIDE_LEAST_ROWS)
				continue;
			struct execution_program *program = NULL;
			struct step_companion_program companions = { 0 };
			expect(v41_tp_program_prepare(
				       &build, &test_host_memory_backend, cards,
				       reported, kinds[kind],
				       kinds[kind] == PROGRAM_THIN ?
					       EXL3_THIN_SIX :
					       capacity,
				       &program, &companions) == STATUS_OK &&
				       program != NULL,
			       "head shards: every kind prepares");
			for (uint32_t card = 0; program != NULL && card < EXL3_CARDS; card++) {
				uint32_t samplers = 0, streams = 0, heads = 0, shares = 0, calls = 0, pushes = 0, fast = 0, grouped = 0;
				uint32_t
					exchanges = 0,
					wide_exchange
						[TP_EXCHANGE_WIDE_LAUNCHES] = { 0 },
					embeds = 0;
				const char *const
					wide_names[TP_EXCHANGE_WIDE_LAUNCHES] = {
						KERNEL_TP_EXCHANGE_PUBLISH,
						KERNEL_TP_EXCHANGE_READY,
						KERNEL_TP_EXCHANGE_SUM,
						KERNEL_TP_EXCHANGE_DONE
					};
				for (uint32_t index = 0; index < program->cards[card].launch_count; index++) {
					const struct kernel_launch *const launch =
						&program->cards[card]
							 .launches[index];
					struct test_host_memory_backend_kernel kernel;
					if (!test_host_memory_backend_kernel_of(card, launch->kernel, &kernel))
						continue;
					struct dense_linear_arguments dense;
					memcpy(&dense, launch->argument, sizeof(dense));
					const bool head =
						strcmp(kernel.function_name,
						       KERNEL_DENSE_STREAM) ==
							0 ||
						(strncmp(kernel.function_name,
							 KERNEL_DENSE_LINEAR_STEM,
							 strlen(KERNEL_DENSE_LINEAR_STEM)) ==
							 0 &&
						 (dense.flags &
						  DENSE_STEP_ROWS) != 0);
					streams +=
						strcmp(kernel.function_name,
						       KERNEL_DENSE_STREAM) ==
						0;
					shares +=
						strcmp(kernel.function_name,
						       card == 0 ?
							       KERNEL_DRAFT_SHARE :
							       KERNEL_DRAFT_SHARE_WAIT) ==
						0;
					heads += head && dense.output_size == shard;
					grouped += strstr(kernel.function_name, "_t4_g4p") != NULL;
					exchanges +=
						strcmp(kernel.function_name,
						       KERNEL_TP_EXCHANGE) == 0;
					embeds += strcmp(kernel.function_name,
							 KERNEL_HYPER_EMBED) ==
						  0;
					for (uint32_t part = 0;
					     part < TP_EXCHANGE_WIDE_LAUNCHES;
					     part++) {
						if (strcmp(kernel.function_name, wide_names[part]) != 0)
							continue;
						struct test_host_memory_backend_kernel previous;
						wide_exchange[part]++;
						expect(launch->grid[0] ==
								       (part % 2u == 0 ?
										TP_EXCHANGE_WIDE_BLOCKS :
										1u) &&
							       launch->block[0] ==
								       ORDERED_COLLECTIVE_THREADS &&
							       (part == 0 ||
								(index > 0 &&
								 test_host_memory_backend_kernel_of(
									 card,
									 launch[-1]
										 .kernel,
									 &previous) &&
								 strcmp(previous.function_name,
									wide_names
										[part -
										 1u]) ==
									 0)),
						       "wide_exchange: publish (64 CTAs), ready, sums (64 CTAs), done, in order");
						break;
					}
					fast += strncmp(kernel.function_name,
							"exl3_c_",
							strlen("exl3_c_")) ==
							0 ||
						strncmp(kernel.function_name,
							"exl3_w_",
							strlen("exl3_w_")) == 0;
					if (strcmp(kernel.function_name,
						   KERNEL_SAMPLER_SETTINGS_PUSH) ==
					    0) {
						struct sampler_settings_push_arguments
							push;
						memcpy(&push, launch->argument, sizeof(push));
						pushes += index == 0 &&
							  push.peer_table_offset ==
								  model->stages[card]->sampler_settings_peers.byte_offset &&
							  push.session_capacity == EXL3_SESSIONS;
					}
					const struct v41_tensor *const own_head =
						&model->views[card]
							 ->globals[V41_DATA_HEAD];
					expect(!head ||
						       dense.output_size !=
							       shard ||
						       dense.weights_offset ==
							       own_head->stored.byte_offset +
								       (own_head->shape.output_size ==
											shard ?
										0u :
										(uint64_t)card *
											shard *
											MODEL_HIDDEN *
											sizeof(uint16_t)),
					       "a rank's head cut reads its quarter of the head's rows");
					if (strcmp(kernel.function_name,
						   KERNEL_SAMPLER_SHARD) != 0)
						continue;
					struct sampler_shard_arguments arguments;
					memcpy(&arguments, launch->argument, sizeof(arguments));
					samplers++;
					calls |= 1u << arguments.call;
					expect(arguments.rank == card &&
						       arguments.first_token ==
							       card * shard &&
						       arguments.shard_size ==
							       shard &&
						       arguments.vocabulary_size ==
							       MODEL_VOCABULARY_VALUE &&
						       launch->grid[1] ==
							       (arguments.call ==
										0 ?
									EXL3_THIN_SIX :
									1u) &&
						       arguments.exchange_offset ==
							       model->stages[card]
								       ->sampler_exchange
								       .byte_offset &&
						       arguments.peer_table_offset ==
							       model->stages[card]
								       ->sampler_peers
								       .byte_offset &&
						       arguments.session_offset ==
							       model->stages[card]
								       ->sampler_settings
								       .byte_offset,
					       "a rank's sampler shard names its tokens, its exchange areas and its settings copy");
				}
				const bool thin = kinds[kind] == PROGRAM_THIN;
				const uint32_t positions = description.limits.draft_positions;
				const uint32_t cuts = thin ? (drafts ? 2u : 1u) : 0u;
				expect(fast != 0, "exl3_fast_products: the EXL3 layers take kernel_exl3_wide's transforms and products");
				expect(card == 0 || (embeds != 0) == description.replicated_embedding,
				       "replicated_embedding: ranks 1-3 embed their own rows");
				expect(program->hand_off_count == (description.replicated_embedding ? 0u : EXL3_EMBED_COPIES),
				       "replicated_embedding: no streams leave card 0 through the host");
				/* the BF16 recipe's chunked attention: slab rings (with draft shards here) or the wide exchange */
				const bool wide_pushes = !thin && !exact && !description.bf16_slab_rings;
				expect(exchanges == 0 && (wide_exchange[0] != 0) == wide_pushes &&
					       wide_exchange[1] == wide_exchange[0] && wide_exchange[2] == wide_exchange[0] &&
					       wide_exchange[3] == wide_exchange[0],
				       "wide_exchange: every piece's FP32 ordered push in its four launches, none in one CTA");
				expect((grouped != 0) == !thin,
				       "exl3_wide_entry: a chunk's EXL3 experts in one call of the grouped products");
				expect(pushes == (thin && card == 0 ? 1u : 0u),
				       "thin programs: the result card's first launch pushes the settings to every rank's copy");
				expect(samplers == (!thin ? 0u : drafts ? 1u + positions : 1u) && heads == cuts &&
					       shares == (thin && drafts ? 1u : 0u) &&
					       calls == (!thin ? 0u : drafts ? (2u << positions) - 1u : 1u),
				       "thin programs: the target's cut and sampler shard a rank, with draft shards the draft's cut, "
				       "share and one sampler shard a position; wide/full none");
				/* with head_stream: every cut, and rank 0's draft's whole head where the draft is not cut */
				expect(streams == (!stream || !thin ? 0u : cuts + (card == 0 && !drafts ? 1u : 0u)),
				       "head_stream takes dense_stream's kernel for every head product");
			}
			if (program != NULL) {
				program_histogram(
					program, kinds[kind] == PROGRAM_THIN ?
							 (exact ? "THIN exact" :
								  "THIN bf16") :
							 "chunked");
				check_handles(program, "head shards");
				programs[program_count++] = program;
			}
		}
		/* The fixture has no Engram tables: every other rule of the driver's (step_validation.c) holds. */
		for (uint32_t index = 0; index < program_count; index++) {
			for (uint32_t card = 0; card < programs[index]->card_count; card++) {
				const uint32_t none = 0;
				memcpy((void *)(uintptr_t)&programs[index]->cards[card].engram_tables, &none, sizeof(none));
			}
		}
		struct step_driver *driver = NULL;
		expect(step_driver_create(
			       &test_host_memory_backend, cards,
			       (const struct execution_program *const *)programs,
			       program_count, NULL, EXL3_WAIT_NS,
			       &driver) == STATUS_OK &&
			       driver != NULL,
		       "head shards: the step driver takes the thin, wide and full programs");
		step_driver_destroy(driver);
		for (uint32_t index = 0; index < program_count; index++)
			free(programs[index]);
	}
	v41_tp2_model_destroy(model);
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card], EXL3_WAIT_NS);
	}
}

/* batch: BATCH programs of 12 to 64 rows (model_seam.h): FULL's layers and draft context, then each sampling card's
 * heads slab by slab (sampler uses 0 .. slabs - 1 with head shards), each slab's own result area. */
static void check_batch(const struct pack_record *pack, bool shards,
			bool drafts, bool wide, bool rows_draft,
			uint32_t sessions, uint32_t positions, bool exact)
{
	const uint32_t shard = MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS;
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES,
			    MODEL_WIDE_LEAST_ROWS, sessions, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    CMP_ORDER_REFERENCE, positions, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.exact_output_shards = exact,
		.wire_format = exact ? 0u : ORDERED_PUSH_BF16,
		.exl3_fast_products = !exact,
		.exl3_wide_entry = !exact,
		.wide_exchange = true,
		.exact_dense = exact,
		.replicated_embedding = true,
		.head_shards = shards,
		.draft_shards = drafts,
		.batch = true,
		.batch_wide_head = wide,
		.batch_draft_rows = rows_draft
	};
	struct device_card *cards[EXL3_CARDS] = { 0 };
	struct card_description_record reported[EXL3_CARDS] = { 0 };
	test_host_memory_backend_reset(EXL3_CARDS, EXL3_CARD_BYTES);
	bool opened = true;
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes = reported[card].memory_free_bytes = EXL3_REPORTED_BYTES;
		description.cards[card] = reported[card];
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		opened = test_host_memory_backend.card_open(
				 card, EXL3_CARD_BYTES, &cards[card]) ==
				 STATUS_OK &&
			 opened;
	}
	struct v41_tp2_model *model = NULL;
	expect(opened &&
		       v41_tp2_model_create(pack, &description, &model) ==
			       STATUS_OK &&
		       model != NULL,
	       "batch: four-card metadata prepares");
	struct v41_tp2_program_description build = { 0 };
	expect(model != NULL && v41_tp2_model_program_description(
					model, &build) == STATUS_OK,
	       "batch: program description");
	const uint32_t rows[] = { 12u, 24u, 48u, PROGRAM_BATCH_MOST_ROWS,
				  LINEAR_MAXIMUM_ROWS };
	for (uint32_t index = 0; model != NULL && index < sizeof(rows) / sizeof(rows[0]); index++) {
		struct execution_program *program = NULL;
		struct step_companion_program companions = { 0 };
		expect(v41_tp_program_prepare(&build, &test_host_memory_backend,
					      cards, reported, PROGRAM_BATCH,
					      rows[index], &program,
					      &companions) == STATUS_OK &&
			       program != NULL &&
			       program->kind == PROGRAM_BATCH,
		       "a BATCH program prepares");
		const uint32_t slabs =
			(rows[index] + LINEAR_MAXIMUM_ROWS - 1u) /
			LINEAR_MAXIMUM_ROWS;
		for (uint32_t card = 0; program != NULL && card < EXL3_CARDS; card++) {
			uint32_t samplers = 0, results = 0, verifies = 0, confidences = 0, verifiers = 0, shares = 0, wides = 0;
			uint32_t draft_wides = 0, draft_rows_seen = 0, batched_markovs = 0, pushes = 0, rings = 0;
			uint64_t calls = 0;
			uint64_t draft_steps[LINEAR_MAXIMUM_ROWS] = { 0 };
			const uint64_t base = model->stages[card]->batch_results.byte_offset;
			for (uint32_t launch = 0; launch < program->cards[card].launch_count; launch++) {
				const struct kernel_launch *const own =
					&program->cards[card].launches[launch];
				struct test_host_memory_backend_kernel kernel;
				if (!test_host_memory_backend_kernel_of(card, own->kernel, &kernel))
					continue;
				shares +=
					strcmp(kernel.function_name,
					       card == 0 ?
						       KERNEL_DRAFT_SHARE :
						       KERNEL_DRAFT_SHARE_WAIT) ==
					0;
				pushes += strcmp(kernel.function_name,
						 KERNEL_TP_EXCHANGE) == 0 ||
					  strcmp(kernel.function_name,
						 KERNEL_TP_EXCHANGE_PUBLISH) ==
						  0;
				rings += strcmp(kernel.function_name,
						KERNEL_EXACT_RING) == 0;
				if (strcmp(kernel.function_name,
					   KERNEL_DENSE_STREAM_WIDE) == 0) {
					struct dense_linear_arguments arguments;
					memcpy(&arguments, own->argument, sizeof(arguments));
					wides +=
						arguments.row_count ==
							rows[index] &&
						arguments.output_offset ==
							model->stages[card]
								->batch_logits
								.byte_offset &&
						arguments.output_size ==
							(shards ?
								 shard :
								 MODEL_VOCABULARY_VALUE) &&
						(arguments.flags &
						 DENSE_STEP_ROWS) != 0;
					/* the batched draft's one wide head: every draft row, the whole head, no gate
					 * (into the logits' slot rows, or behind them: fewer verifiers than slots) */
					const uint32_t part =
						V41_DATA_BATCH_DRAFT_LOGITS;
					const uint64_t logits_at =
						model->stages[card]
							->batch_rows
							.byte_offset +
						v41_batch_rows_offset(part);
					if (rows_draft &&
					    arguments.output_offset >=
						    logits_at &&
					    arguments.output_offset <
						    logits_at +
							    v41_batch_rows_bytes(
								    part)) {
						draft_wides +=
							arguments.flags == 0 &&
							arguments.output_size ==
								MODEL_VOCABULARY_VALUE;
						draft_rows_seen = arguments.row_count;
					}
				}
				if (strcmp(kernel.function_name,
					   KERNEL_SAMPLER_SHARD) == 0) {
					struct sampler_shard_arguments arguments;
					memcpy(&arguments, own->argument, sizeof(arguments));
					calls |= 1ull << arguments.call;
					if (arguments.call >= slabs)
						continue; /* a draft position's */
					samplers++;
					results +=
						arguments.result_offset ==
						base + (uint64_t)arguments.call *
								PROGRAM_BATCH_SLAB_RESULT_BYTES;
					expect(arguments.shard_size == shard && arguments.first_token == card * shard,
					       "a slab's sampler shard names the rank's tokens");
				} else if (strcmp(kernel.function_name,
						  KERNEL_SAMPLER) == 0) {
					struct sampler_arguments arguments;
					memcpy(&arguments, own->argument, sizeof(arguments));
					if (arguments.result_offset >= base &&
					    arguments.result_offset < base + model->stages[card]->batch_results.byte_count) {
						results +=
							arguments.result_offset ==
							base + (uint64_t)samplers *
									PROGRAM_BATCH_SLAB_RESULT_BYTES;
						samplers++;
					}
				} else if (strcmp(kernel.function_name,
						  KERNEL_VERIFY_MATCHES) == 0) {
					struct verify_matches_arguments
						arguments;
					memcpy(&arguments, own->argument, sizeof(arguments));
					expect(arguments.flags ==
							       (DRAFT_ROUND_TABLE |
								DRAFT_SLAB_RESULTS |
								(rows_draft ?
									 DRAFT_BATCHED :
									 0u)) &&
						       (!rows_draft ||
							(arguments.draft_step_offset ==
								 model->stages[card]
									 ->batch_rows
									 .byte_offset &&
							 arguments.draft_slabs !=
								 0)) &&
						       arguments.verifier ==
							       verifies &&
						       arguments.result_offset ==
							       base &&
						       arguments.round_offset ==
							       model->stages[card]
								       ->batch_rounds
								       .byte_offset &&
						       arguments.round_capacity ==
							       sessions &&
						       arguments.row_count ==
							       rows[index],
					       "a verifier's walk: its index, the slab results and the round table");
					if (verifies < LINEAR_MAXIMUM_ROWS)
						draft_steps[verifies] = arguments.draft_step_offset;
					verifies++;
				} else if (strcmp(kernel.function_name,
						  KERNEL_DRAFT_MARKOV_BIAS) ==
					   0) {
					struct draft_markov_bias_arguments
						arguments;
					memcpy(&arguments, own->argument, sizeof(arguments));
					batched_markovs += arguments.flags ==
							   DRAFT_BATCHED;
				} else if (strcmp(kernel.function_name,
						  KERNEL_DRAFT_CONFIDENCE) ==
					   0) {
					struct draft_confidence_arguments
						arguments;
					memcpy(&arguments, own->argument, sizeof(arguments));
					confidences +=
						arguments.flags ==
							(DRAFT_ROUND_TABLE |
							 (rows_draft ?
								  DRAFT_BATCHED :
								  0u)) &&
						(rows_draft ?
							 arguments.verifier ==
								 confidences :
							 arguments.draft_step_offset ==
								 draft_steps
									 [confidences]);
				}
			}
			const bool samples = shards || card == 0;
			uint32_t batch_walks = rows[index] / (positions + 1u) < sessions ?
						       rows[index] / (positions + 1u) : sessions;
			batch_walks = batch_walks == 0 ? 1u : batch_walks;
			const uint32_t uses = slabs + (drafts ? batch_walks * positions : 0u);
			expect(samplers == (samples ? slabs : 0u) && results == samplers &&
				       (!shards || calls == (uses >= 64u ? ~0ull : (1ull << uses) - 1u)) &&
				       shares == (drafts ? batch_walks : 0u) && wides == (wide && samples ? 1u : 0u),
			       "every sampling card samples each slab into the slab's own result area; with draft shards "
			       "each verifier's share or wait and sampler uses behind the slabs'");
			/* the BF16 recipe: the attention and FFN sums of all the step's rows by one BF16 ring a sublayer
			 * (ef38611), no FP32 push */
			expect(pushes == 0 &&
				       rings >=
					       (exact ? 0u : 2u * MODEL_LAYERS),
			       "a BATCH program sums its rows by one ring a sublayer, the attention as the FFN");
			expect(batched_markovs == (rows_draft && card == 0 ? positions : 0u),
			       "the batched draft: one Markov bias a position over every slot");
			/* fewer verifiers than slots: the head of their own rows, then each position's logits to its
			 * slot (one verifier in place; more where the rows fit behind the slots' in the area's 56) */
			const uint32_t own_rows = batch_walks * positions;
			const uint32_t slot_rows =
				positions * DRAFT_BATCH_SLOTS;
			const bool own = batch_walks == 1u ||
					 (batch_walks < DRAFT_BATCH_SLOTS &&
					  slot_rows + own_rows <=
						  V41_DATA_BATCH_DRAFT_ROWS);
			expect(draft_wides == (wide && rows_draft && card == 0 ? 1u : 0u) &&
				       (draft_wides == 0 ||
					draft_rows_seen == (own ? own_rows : slot_rows)),
			       "batch_wide_head with the batched draft: one wide draft head over every position's 8 slots, "
			       "or one verifier's own rows");
			verifiers = card != 0 ? 0u : batch_walks;
			bool distinct = true;
			for (uint32_t first = 0;
			     first < verifies && first < LINEAR_MAXIMUM_ROWS;
			     first++)
				for (uint32_t second = first + 1u;
				     second < verifies &&
				     second < LINEAR_MAXIMUM_ROWS;
				     second++)
					distinct = distinct && draft_steps[first] != draft_steps[second];
			expect(verifies == verifiers && confidences == verifiers && (distinct || rows_draft),
			       "one draft walk a verifier, each with its own draft arrays, its confidence the round table's");
		}
		if (program != NULL)
			check_handles(program, "batch");
		if (index == 0 && sessions == EXL3_C8_SESSIONS)
			program_histogram(program, exact ? "BATCH-12 exact" : "BATCH-12 bf16");
		expect(program == NULL || (program->cards[0].result_offset == model->stages[0]->batch_results.byte_offset &&
					   program->cards[0].round_offset == model->stages[0]->batch_rounds.byte_offset),
		       "the result card's result area is its first slab's, its round area the round table");
		free(program);
	}
	struct execution_program *program = NULL;
	struct step_companion_program companions = { 0 };
	expect(model == NULL || v41_tp_program_prepare(
					&build, &test_host_memory_backend,
					cards, reported, PROGRAM_BATCH,
					PROGRAM_BATCH_MOST_ROWS + 1u, &program,
					&companions) == STATUS_INVALID_ARGUMENT,
	       "BATCH programs take at most 64 rows");
	/* the step driver takes a BATCH cohort and the chunked programs (main.c's --batch_rows set): BATCH programs of
	 * 8 (a thin walk), 12, 24, 48 and 64 rows with the batched draft rows; with draft shards or without the
	 * batched draft rows one BATCH size (a verifier's walk its own segments: main.c refuses more) */
	if (model != NULL) {
		enum { SET = 7 };
		struct execution_program *set[SET] = { 0 };
		struct step_companion_program set_companions[SET] = { 0 };
		const enum program_kind kinds[SET] = {
			PROGRAM_BATCH, PROGRAM_BATCH, PROGRAM_BATCH,
			PROGRAM_BATCH, PROGRAM_BATCH, PROGRAM_WIDE,
			PROGRAM_FULL
		};
		const uint32_t rows_of[SET] = { 12u,
						LINEAR_MAXIMUM_ROWS,
						24u,
						48u,
						PROGRAM_BATCH_MOST_ROWS,
						MODEL_WIDE_LEAST_ROWS,
						MODEL_WIDE_LEAST_ROWS };
		const uint32_t first_chunked = SET - 2u;
		uint32_t count = 0;
		bool prepared = true;
		for (uint32_t index = 0; index < SET; index++) {
			if ((drafts || !rows_draft) && index != 0 && index < first_chunked)
				continue;
			prepared = v41_tp_program_prepare(
					   &build, &test_host_memory_backend,
					   cards, reported, kinds[index],
					   rows_of[index], &set[count],
					   &set_companions[count]) ==
					   STATUS_OK &&
				   prepared;
			count++;
		}
		for (uint32_t index = 0; index < count && prepared; index++)
			for (uint32_t card = 0; card < set[index]->card_count; card++) {
				const uint32_t none = 0;
				memcpy((void *)(uintptr_t)&set[index]->cards[card].engram_tables, &none, sizeof(none));
			}
		struct step_driver *driver = NULL;
		const result_status created =
			prepared ?
				step_driver_create(
					&test_host_memory_backend, cards,
					(const struct execution_program *const *)
						set,
					count, NULL, EXL3_WAIT_NS, &driver) :
				STATUS_INVALID_ARGUMENT;
		if (getenv("BATCH_DRIVER_DEBUG") != NULL)
			(void)fprintf(stderr, "batch driver: prepared %d created %u (shards %d drafts %d wide %d rows_draft %d "
				      "sessions %u positions %u exact %d)\n", prepared, (uint32_t)created, shards, drafts, wide,
				      rows_draft, sessions, positions, exact);
		expect(prepared && created == STATUS_OK && driver != NULL,
		       "the step driver takes the BATCH programs (8 rows: a thin walk) with the wide and full programs");
		step_driver_destroy(driver);
		for (uint32_t index = 0; index < count; index++)
			free(set[index]);
	}
	v41_tp2_model_destroy(model);
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card], EXL3_WAIT_NS);
	}
}

/* exact_dense (model_v41_tp2_stage.h): wq_b, wo_a and wo_b load as output shards on every rank; every program kind
 * prepares with every launch's handle on its card; a thin program's layers each run two dense gathers. */
static void check_exact_dense(const struct pack_record *pack, uint32_t capacity)
{
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES, capacity,
			    EXL3_SESSIONS, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    CMP_ORDER_REFERENCE, EXL3_DRAFT, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.exact_output_shards = true,
		.exact_dense = true,
		.head_shards = true,
		.draft_shards = true
	};
	struct device_card *cards[EXL3_CARDS] = { 0 };
	struct card_description_record reported[EXL3_CARDS] = { 0 };
	test_host_memory_backend_reset(EXL3_CARDS, EXL3_CARD_BYTES);
	bool opened = true;
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes = reported[card].memory_free_bytes = EXL3_REPORTED_BYTES;
		description.cards[card] = reported[card];
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		opened = test_host_memory_backend.card_open(
				 card, EXL3_CARD_BYTES, &cards[card]) ==
				 STATUS_OK &&
			 opened;
	}
	struct v41_tp2_model *model = NULL;
	expect(opened &&
		       v41_tp2_model_create(pack, &description, &model) ==
			       STATUS_OK &&
		       model != NULL,
	       "exact dense: four-card metadata prepares");
	if (model != NULL) {
		for (uint32_t card = 0; card < EXL3_CARDS; card++) {
			const struct v41_tensor *const layer0 =
				model->stages[card]->model->layer_tensors[0];
			expect(layer0[V41_DATA_QUERY_UP].shape.output_size ==
					       MODEL_HEADS_VALUE *
						       MODEL_HEAD_DIMS /
						       EXL3_CARDS &&
				       layer0[V41_DATA_OUTPUT]
						       .shape.output_size ==
					       MODEL_HIDDEN / EXL3_CARDS &&
				       model->stages[card]
						       ->dense_input
						       .byte_count != 0,
			       "exact dense: a rank holds its quarter of wq_b and wo_b and the gathers' slots");
		}
		struct v41_tp2_program_description build = { 0 };
		expect(v41_tp2_model_program_description(model, &build) ==
			       STATUS_OK,
		       "exact dense: program description");
		const enum program_kind kinds[] = { PROGRAM_THIN, PROGRAM_WIDE,
						    PROGRAM_FULL };
		for (uint32_t kind = 0; kind < sizeof(kinds) / sizeof(kinds[0]); kind++) {
			if (kinds[kind] != PROGRAM_THIN &&
			    capacity < MODEL_WIDE_LEAST_ROWS)
				continue;
			struct execution_program *program = NULL;
			struct step_companion_program companions = { 0 };
			expect(v41_tp_program_prepare(
				       &build, &test_host_memory_backend, cards,
				       reported, kinds[kind],
				       kinds[kind] == PROGRAM_THIN ?
					       EXL3_THIN_SIX :
					       capacity,
				       &program, &companions) == STATUS_OK &&
				       program != NULL,
			       "exact dense: every kind prepares");
			if (program == NULL)
				continue;
			check_handles(program, "exact dense");
			uint32_t gathers = 0;
			for (uint32_t index = 0; index < program->cards[1].launch_count; index++) {
				struct test_host_memory_backend_kernel kernel;
				if (test_host_memory_backend_kernel_of(1, program->cards[1].launches[index].kernel, &kernel))
					gathers += strcmp(kernel.function_name,
							  KERNEL_EXACT_RING) ==
						   0;
			}
			if (kinds[kind] == PROGRAM_THIN)
				expect(gathers >= 2u * MODEL_LAYERS,
				       "exact dense: two dense gathers a layer (and the layers' own)");
			free(program);
		}
	}
	v41_tp2_model_destroy(model);
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card], EXL3_WAIT_NS);
	}
}

/* draft_layers (model_v41_tp2_stage.h): the BF16 recipe's draft tensor-parallel. Every rank loads its cut of the draft
 * stages and its draft globals; a thin program's draft tail holds a segment on every card (the result card's first);
 * a BATCH program's (batch_draft_rows, batch_wide_head) every card's batched draft. */
static void check_draft_layers(const struct pack_record *pack,
			       uint32_t capacity, bool batch, bool branches)
{
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES,
			    batch ? MODEL_WIDE_LEAST_ROWS : capacity,
			    batch ? EXL3_C8_SESSIONS : EXL3_SESSIONS,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE,
			    MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.wire_format = ORDERED_PUSH_BF16,
		.head_shards = true,
		.draft_shards = true,
		.draft_layers = true,
		.exl3_fast_products = true,
		.exl3_wide_entry = true,
		.wide_exchange = true,
		.replicated_embedding = true,
		.batch = batch,
		.batch_wide_head = batch,
		.batch_draft_rows = batch,
		.graph_branches = branches ? 2u : 0u,
		.exl3_tc_moe = branches,
		.trunk_mma = branches
	};
	struct device_card *cards[EXL3_CARDS] = { 0 };
	struct card_description_record reported[EXL3_CARDS] = { 0 };
	test_host_memory_backend_reset(EXL3_CARDS, EXL3_CARD_BYTES);
	bool opened = true;
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes = reported[card].memory_free_bytes = EXL3_REPORTED_BYTES;
		description.cards[card] = reported[card];
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		opened = test_host_memory_backend.card_open(
				 card, EXL3_CARD_BYTES, &cards[card]) ==
				 STATUS_OK &&
			 opened;
	}
	struct v41_tp2_model *model = NULL;
	expect(opened &&
		       v41_tp2_model_create(pack, &description, &model) ==
			       STATUS_OK &&
		       model != NULL,
	       "draft layers: four-card metadata prepares");
	for (uint32_t card = 1; model != NULL && card < EXL3_CARDS; card++)
		expect(model->views[card]->draft_globals[V41_DATA_DRAFT_NORM]
					       .card == card &&
			       model->views[card]
					       ->draft_globals
						       [V41_DATA_DRAFT_MAIN_PROJECTION]
					       .card == card &&
			       model->views[card]
					       ->layer_tensors
						       [MODEL_DRAFT_FIRST_LAYER]
						       [V41_DATA_OUTPUT]
					       .card == card &&
			       model->stages[card]->channel_count ==
				       (MODEL_LAYERS + MODEL_DRAFT_STAGES) *
					       TP2_SITES,
		       "draft layers: ranks 1-3 hold their draft stages, globals and channels");
	struct v41_tp2_program_description build = { 0 };
	expect(model != NULL && v41_tp2_model_program_description(
					model, &build) == STATUS_OK,
	       "draft layers: program description");
	const enum program_kind kinds[] = { PROGRAM_THIN, PROGRAM_WIDE,
					    PROGRAM_FULL, PROGRAM_BATCH };
	struct execution_program *programs[sizeof(kinds) / sizeof(kinds[0])] = {
		0
	};
	uint32_t program_count = 0;
	for (uint32_t kind = 0; model != NULL && kind < sizeof(kinds) / sizeof(kinds[0]); kind++) {
		if ((kinds[kind] == PROGRAM_BATCH) != batch &&
		    kinds[kind] != PROGRAM_THIN)
			continue;
		if (kinds[kind] != PROGRAM_THIN &&
		    kinds[kind] != PROGRAM_BATCH &&
		    description.limits.row_capacity < MODEL_WIDE_LEAST_ROWS)
			continue;
		const uint32_t rows = kinds[kind] == PROGRAM_THIN ?
					      EXL3_THIN_SIX :
				      kinds[kind] == PROGRAM_BATCH ?
					      48u :
					      description.limits.row_capacity;
		struct execution_program *program = NULL;
		struct step_companion_program companions = { 0 };
		const result_status prepared = v41_tp_program_prepare(
			&build, &test_host_memory_backend, cards, reported,
			kinds[kind], rows, &program, &companions);
		if (prepared != STATUS_OK)
			fprintf(stderr, "draft layers: kind %u rows %u: %s\n",
				(unsigned)kinds[kind], (unsigned)rows,
				status_text(prepared));
		expect(prepared == STATUS_OK && program != NULL,
		       "draft layers: every kind prepares");
		if (program == NULL)
			continue;
		uint32_t tail_cards = 0, branches_seen = 0;
		for (uint32_t segment = program->draft_segment_first; segment < program->segment_count; segment++)
			tail_cards |= 1u << program->segments[segment].card;
		for (uint32_t card = 0; card < EXL3_CARDS; card++)
			for (uint32_t index = 0; index < program->cards[card].launch_count; index++)
				branches_seen += program->cards[card]
							 .launches[index]
							 .branch !=
						 LAUNCH_CHAIN;
		expect(kinds[kind] == PROGRAM_WIDE ||
			       kinds[kind] == PROGRAM_FULL ||
			       (tail_cards == (1u << EXL3_CARDS) - 1u &&
				program->segments[program->draft_segment_first]
						.card == program->result_card),
		       "draft layers: the draft tail a segment on every card, the result card's first");
		expect(kinds[kind] == PROGRAM_WIDE ||
			       kinds[kind] == PROGRAM_FULL ||
			       (branches ? branches_seen != 0 :
					   branches_seen == 0),
		       "graph branches: the TC-MoE layers' shared experts branch, none without the knob");
		for (uint32_t card = 0; card < program->card_count; card++) {
			const uint32_t none = 0;
			memcpy((void *)(uintptr_t)&program->cards[card].engram_tables, &none, sizeof(none));
		}
		programs[program_count++] = program;
	}
	if (program_count != 0 && !batch) {
		struct step_driver *driver = NULL;
		const result_status created = step_driver_create(
			&test_host_memory_backend, cards,
			(const struct execution_program *const *)programs,
			program_count, NULL, EXL3_WAIT_NS, &driver);
		if (created != STATUS_OK)
			fprintf(stderr, "draft layers: the step driver: %s\n",
				status_text(created));
		expect(created == STATUS_OK && driver != NULL,
		       "draft layers: the step driver takes the thin, wide and full programs");
		step_driver_destroy(driver);
	}
	for (uint32_t index = 0; index < program_count; index++)
		free(programs[index]);
	v41_tp2_model_destroy(model);
	for (uint32_t card = 0; card < EXL3_CARDS; card++) {
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card], EXL3_WAIT_NS);
	}
}

int main(void)
{
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = {
		.kind = TEST_MODEL_V41_CHANGE_EXL3
	};
	const result_status status =
		test_model_v41_draft_pack_open(change, &pack);
	expect(status == STATUS_OK,
	       "independent EXL3 target/native draft metadata fixture opens");
	if (status == STATUS_OK) {
		const uint32_t capacities[] = { LINEAR_MAXIMUM_ROWS,
						MODEL_WIDE_LEAST_ROWS,
						LINEAR_WIDE_MOST_ROWS };
		const enum bb_exl3_product_order orders[] = {
			BB_EXL3_REFERENCE_ORDER, BB_EXL3_EIGHT_PART_ORDER
		};
		for (uint32_t shape = 0;
		     shape < sizeof(capacities) / sizeof(capacities[0]);
		     shape++) {
			for (uint32_t order = 0;
			     order < sizeof(orders) / sizeof(orders[0]);
			     order++)
				check_case(pack, capacities[shape],
					   orders[order], true);
			check_case(pack, capacities[shape],
				   BB_EXL3_REFERENCE_ORDER, false);
			check_head_shards(pack, capacities[shape], false, true, false);
			check_head_shards(pack, capacities[shape], true, true, false);
			check_head_shards(pack, capacities[shape], true, false, false);
			check_head_shards(pack, capacities[shape], false, true, true);
			check_head_shards(pack, capacities[shape], true, false, true);
		}
	}
	if (status == STATUS_OK) {
		check_exact_dense(pack, LINEAR_MAXIMUM_ROWS);
		check_exact_dense(pack, MODEL_WIDE_LEAST_ROWS);
		check_batch(pack, false, false, false, false, EXL3_SESSIONS, EXL3_DRAFT, true);
		check_batch(pack, true, false, false, false, EXL3_SESSIONS, EXL3_DRAFT, true);
		check_batch(pack, true, true, false, false, EXL3_SESSIONS, EXL3_DRAFT, true);
		check_batch(pack, false, false, true, false, EXL3_SESSIONS, EXL3_DRAFT, true);
		check_batch(pack, true, true, true, false, EXL3_SESSIONS, EXL3_DRAFT, true);
		check_batch(pack, false, false, false, true, EXL3_SESSIONS, EXL3_DRAFT, true);
		check_batch(pack, true, false, true, true, EXL3_SESSIONS, EXL3_DRAFT, true);
		/* the C8 cell: eight sessions, eight verifier walks at 64 rows (the segment table's most) */
		check_batch(pack, true, true, false, false, EXL3_C8_SESSIONS, EXL3_DRAFT, true);
		check_batch(pack, false, false, false, false, EXL3_C8_SESSIONS, EXL3_DRAFT, true);
		/* K5 (the serving width): 8 walks from 48 rows; the batched draft holds 8 x 5 = 40 draft rows */
		check_batch(pack, true, true, true, false, EXL3_C8_SESSIONS,
			    MODEL_DRAFT_BLOCK, true);
		check_batch(pack, false, false, true, true, EXL3_C8_SESSIONS,
			    MODEL_DRAFT_BLOCK, true);
		check_batch(pack, true, false, false, true, EXL3_C8_SESSIONS, 6u, true); /* K6: 8 x 6 = 48 batched draft rows */
		check_batch(pack, false, false, true, true, EXL3_C8_SESSIONS, EXL3_DRAFT, true); /* K7: 8 x 7 = 56, the most */
		check_batch(pack, true, true, false, false, EXL3_C8_SESSIONS,
			    MODEL_DRAFT_BLOCK, false);
		check_batch(pack, true, false, true, true, EXL3_C8_SESSIONS, 6u, false);
		check_draft_layers(pack, LINEAR_MAXIMUM_ROWS, false, false);
		check_draft_layers(pack, MODEL_WIDE_LEAST_ROWS, false, true);
		check_draft_layers(pack, MODEL_WIDE_LEAST_ROWS, true, false);
		check_draft_layers(pack, MODEL_WIDE_LEAST_ROWS, true, true);
	}
	pack_close(pack);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
