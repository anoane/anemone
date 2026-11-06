/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Cold real-model metadata, budgets, borrowed image ownership and failure before any payload/card work.
 * MAIN owns execution. The storage holds only model tables; no real tensor payload or GPU is touched here.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41_pack.h"
#include "model_v41_tp2_model.h"

#define TEST_CARDS V41_DATA_TP2_PROGRAM_CARDS
#define TEST_PRO V41_DATA_TP2_PRO_CARD
#define TEST_MEMORY_BYTES (128ull << 30u)
#define TEST_KEPT_BYTES 4096ull
#define TEST_RESERVED_BYTES 8192ull
#define TEST_WAIT_NANOSECONDS 1000000000ull

static int failures;
static uint32_t allocations, retired, waited;
static unsigned char card_handles[TEST_CARDS];
static _Alignas(BACKEND_STAGING_ALIGNMENT_BYTES) unsigned char staged;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "TP2 model: %s\n", message);
		failures++;
	}
}

static struct v41_tp2_model_description description_make(void)
{
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES,
			    LINEAR_MAXIMUM_ROWS, 1, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    CMP_ORDER_REFERENCE, STEP_DRAFT_MOST, 0, 0 },
		.kept_free_bytes = TEST_KEPT_BYTES
	};
	for (uint32_t card = 0; card < TEST_CARDS; card++) {
		description.cards[card].arithmetic_class =
			card == TEST_PRO ? ARITHMETIC_CLASS_SM_120_NATIVE :
					   ARITHMETIC_CLASS_SM_80;
		description.cards[card].memory_total_bytes = TEST_MEMORY_BYTES;
		description.cards[card].memory_free_bytes = TEST_MEMORY_BYTES;
		description.cards[card].memory_reserved_bytes =
			TEST_RESERVED_BYTES;
	}
	return description;
}

static void metadata_check(const struct v41_tp2_model *model)
{
	expect(!model->ready && model->host_arena == NULL &&
		       model->host_offsets == NULL && !model->load_attempted,
	       "cold object publishes no attachable RAM or readiness");
	expect(model->canonical->embedding_card == 0 &&
		       model->canonical->result_card == TEST_PRO &&
		       model->canonical->globals[V41_DATA_EMBEDDING].card ==
			       0 &&
		       model->canonical->globals[V41_DATA_HEAD].card ==
			       TEST_PRO,
	       "canonical global embedding/head ownership resolved");
	uint64_t jobs = 0, destination_end[TEST_CARDS] = { 0 };
	const struct pack_tensor *const tensors =
		pack_tensors_of(model->host_load.pack);
	uint32_t previous_extent = 0, previous_tensor = 0, previous_card = 0;
	for (uint32_t index = 0; index < model->gpu_load.description.job_count;
	     index++) {
		const struct tp_load_job *const job = &model->jobs[index];
		const uint32_t extent = tensors[job->tensor_index].extent_index;
		expect(index == 0 || extent > previous_extent ||
			       (extent == previous_extent &&
				(job->tensor_index > previous_tensor ||
				 (job->tensor_index == previous_tensor &&
				  job->card > previous_card))),
		       "merged jobs ordered by extent/tensor/card");
		expect(job->destination_offset >= destination_end[job->card],
		       "per-card destinations remain monotone");
		destination_end[job->card] =
			job->destination_offset +
			(job->mode == TP_LOAD_FULL ?
				 tensors[job->tensor_index].byte_count :
				 job->shard.shard_byte_count);
		previous_extent = extent;
		previous_tensor = job->tensor_index;
		previous_card = job->card;
	}
	for (uint32_t card = 0; card < TEST_CARDS; card++) {
		jobs += model->stages[card]->job_count;
		expect(memcmp(model->views[card]->cards,
			      model->canonical->cards,
			      sizeof(model->canonical->cards)) == 0 &&
			       memcmp(model->views[card]->layer_card,
				      model->canonical->layer_card,
				      sizeof(model->canonical->layer_card)) ==
				       0 &&
			       model->canonical->cards[card].image ==
				       model->stages[card]
					       ->model->cards[card]
					       .image,
		       "all views borrow canonical card metadata; stage remains the only image owner");
		for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
			if ((model->canonical->cards[card].layers >> layer &
			     1u) != 0)
				expect(model->views[card]
						       ->layer_tensors
							       [layer]
							       [V41_DATA_QUERY_UP]
						       .card == card,
				       "owned layer tensors retain local rank/card");
		}
	}
	expect(jobs == model->gpu_load.description.job_count,
	       "merged loader includes every stage tensor job");
	expect(model->output_workspace_bytes >=
			       V41_DATA_EMBEDDING_COPY_PIECE_BYTES &&
		       model->output_workspace_bytes >=
			       model->gpu_load.output_workspace_byte_count,
	       "shared initializer workspace includes the draft embedding copy chunk");
	struct v41_tp2_program_description program = { 0 };
	expect(v41_tp2_model_program_description(model, &program) ==
			       STATUS_OK &&
		       program.canonical == model->canonical &&
		       program.slice_count == TP2_DEFAULT_SLICES &&
		       model->recipe_identity == (V41_DATA_TP2_PROGRAM_RECIPE ^
						  program.slice_count),
	       "program factory descriptor and arithmetic identity agree");
}

static bool card_metadata_equal(const struct v41_card *first,
				const struct v41_card *second)
{
	struct v41_card left = *first, right = *second;
	left.image = NULL;
	right.image = NULL;
	return memcmp(&left, &right, sizeof(left)) == 0 &&
	       (first->image_byte_count == 0 ||
		memcmp(first->image, second->image, first->image_byte_count) ==
			0);
}

static void original_wide_scratch_check(const struct pack_record *pack,
					uint32_t capacity)
{
	struct v41_tp2_model_description description = description_make();
	description.tensor_layout = V41_DATA_TP_LAYOUT_QUAD;
	description.exact_output_shards = true;
	description.limits.linear_order = LINEAR_ORDER_REFERENCE;
	description.limits.row_capacity = capacity;
	struct v41_tp2_model *zero = NULL, *repeated = NULL, *wide = NULL;
	expect(v41_tp2_model_create(pack, &description, &zero) == STATUS_OK &&
		       zero != NULL,
	       "original wide scratch: default-off exact QUAD prepares");
	description.original_wide_experts = false;
	expect(v41_tp2_model_create(pack, &description, &repeated) ==
			       STATUS_OK &&
		       repeated != NULL,
	       "original wide scratch: explicit zero prepares");
	description.original_wide_experts = true;
	expect(v41_tp2_model_create(pack, &description, &wide) == STATUS_OK &&
		       wide != NULL,
	       "original wide scratch: enabled exact QUAD prepares");
	if (zero == NULL || repeated == NULL || wide == NULL)
		goto done;
	expect(zero->gpu_load.description.job_count ==
			       repeated->gpu_load.description.job_count &&
		       zero->gpu_load.description.job_count ==
			       wide->gpu_load.description.job_count &&
		       memcmp(zero->jobs, repeated->jobs,
			      (size_t)zero->gpu_load.description.job_count *
				      sizeof(*zero->jobs)) == 0 &&
		       memcmp(zero->jobs, wide->jobs,
			      (size_t)zero->gpu_load.description.job_count *
				      sizeof(*zero->jobs)) == 0,
	       "original wide scratch: both controls and enabled path retain literal loader jobs");
	const uint64_t alignment = V41_DATA_ARRAY_ALIGNMENT_BYTES;
	const uint64_t owned = (V41_DATA_FUSED_SCRATCH_BYTES + alignment - 1u) /
			       alignment * alignment;
	for (uint32_t card = 0; card < TEST_CARDS; card++) {
		const struct v41_tp2_stage_plan *const old = zero->stages[card];
		const struct v41_tp2_stage_plan *const stage =
			wide->stages[card];
		const struct v41_card *const before =
			&zero->canonical->cards[card];
		const struct v41_card *const after =
			&wide->canonical->cards[card];
		expect(card_metadata_equal(before,
					   &repeated->canonical->cards[card]) &&
			       old->communication_byte_count ==
				       repeated->stages[card]
					       ->communication_byte_count,
		       "original wide scratch: zero retains images, arena and channel bytes");
		if (card == TEST_PRO) {
			expect(!stage->description.original_wide_experts &&
				       card_metadata_equal(before, after) &&
				       old->communication_byte_count ==
					       stage->communication_byte_count,
			       "original wide scratch: native PRO allocation remains literal");
			continue;
		}
		const uint64_t base_end =
			after->state_offset + stage->base_state_byte_count;
		const uint64_t gathered_growth =
			(uint64_t)(capacity - LINEAR_MAXIMUM_ROWS) *
			(MODEL_EXPERTS_PER_ROW * V41_DATA_EXPERT_INNER *
				 sizeof(uint16_t) +
			 MODEL_HIDDEN * sizeof(float));
		expect(before->front_scratch_offset ==
				       before->array_offset
					       [V41_DATA_ARRAY_ROUTER_PICKS] &&
			       old->compact.byte_offset == base_end &&
			       after->front_scratch_offset == base_end &&
			       after->producer_arrivals_offset ==
				       base_end + SITE_FRONT_SCRATCH_BYTES &&
			       stage->compact.byte_offset == base_end + owned &&
			       base_end >=
				       after->array_offset[V41_DATA_ARRAY_JOBS] &&
			       base_end >=
				       after->array_offset
						       [V41_DATA_ARRAY_ROUTER_PICKS] +
					       (uint64_t)capacity *
						       MODEL_EXPERTS_PER_ROW *
						       sizeof(struct router_pick) &&
			       base_end >= after->state_offset +
						   after->slots_byte_count +
						   after->image_byte_count &&
			       base_end <= after->arena_byte_count &&
			       owned <= after->arena_byte_count - base_end,
		       "original wide scratch: counters own an aligned, cleared range beyond router arrays");
		expect(after->arena_byte_count - before->arena_byte_count ==
				       owned + gathered_growth &&
			       stage->communication_byte_count -
					       old->communication_byte_count ==
				       owned + gathered_growth &&
			       memcmp(after, &stage->model->cards[card],
				      sizeof(*after)) == 0,
		       "original wide scratch: arena and channels count scratch and gathers once");
		struct v41_card normalized = *after;
		normalized.front_scratch_offset = before->front_scratch_offset;
		normalized.producer_arrivals_offset =
			before->producer_arrivals_offset;
		normalized.arena_byte_count = before->arena_byte_count;
		expect(card_metadata_equal(before, &normalized),
		       "original wide scratch: other card fields, arrays and weights remain literal");
		for (uint32_t view = 0; view < TEST_CARDS; view++)
			expect(memcmp(after, &wide->views[view]->cards[card],
				      sizeof(*after)) == 0,
			       "original wide scratch: every program view inherits the dedicated canonical offsets");
	}
done:
	v41_tp2_model_destroy(wide);
	v41_tp2_model_destroy(repeated);
	v41_tp2_model_destroy(zero);
}

static void exl3_wide_scratch_check(const struct pack_record *pack,
				    uint32_t capacity)
{
	struct v41_tp2_model_description description = description_make();
	description.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL;
	description.exact_output_shards = true;
	description.exl3_wide_entry = true;
	description.limits.linear_order = LINEAR_ORDER_REFERENCE;
	description.limits.row_capacity = capacity;
	struct v41_tp2_model *zero = NULL, *repeated = NULL;
	expect(v41_tp2_model_create(pack, &description, &zero) == STATUS_OK &&
		       zero != NULL,
	       "original wide scratch: independent EXL3 wide control prepares");
	description.original_wide_experts = false;
	expect(v41_tp2_model_create(pack, &description, &repeated) ==
			       STATUS_OK &&
		       repeated != NULL,
	       "original wide scratch: independent EXL3 explicit-zero control prepares");
	if (zero != NULL && repeated != NULL) {
		for (uint32_t card = 0; card < TP4_RANKS; card++) {
			const struct v41_card *const planned =
				&zero->canonical->cards[card];
			expect(planned->front_scratch_offset ==
					       planned->array_offset
						       [V41_DATA_ARRAY_ROUTER_PICKS] &&
				       card_metadata_equal(
					       planned, &repeated->canonical
								 ->cards[card]),
			       "original wide scratch: EXL3 retains its alias and literal arena layout");
		}
	}
	v41_tp2_model_destroy(repeated);
	v41_tp2_model_destroy(zero);
}

MUST_CHECK static result_status allocation_fail(uint64_t bytes, void **out)
{
	(void)bytes;
	allocations++;
	if (allocations == 1u) {
		/* No callback may read this stand-in: the second allocation fails before any I/O or upload. */
		*out = &staged;
		return STATUS_OK;
	}
	return STATUS_INJECTED_FAULT;
}

static void staging_retire(void *memory)
{
	expect(memory == NULL || memory == &staged,
	       "retirement receives only the owned staging stand-in");
	retired++;
}

MUST_CHECK static result_status card_wait(struct device_card *card,
					  uint64_t limit)
{
	(void)card;
	(void)limit;
	waited++;
	return STATUS_TIMED_OUT;
}

MUST_CHECK static result_status peer_connect(struct device_card *accessor,
					     struct device_card *owner,
					     uint64_t owner_offset,
					     uint64_t address_word_offset)
{
	(void)accessor;
	(void)owner;
	(void)owner_offset;
	(void)address_word_offset;
	expect(false, "no peer setup before failed workspace allocation");
	return STATUS_UNSUPPORTED;
}

static void load_failure(struct v41_tp2_model *model,
			 const struct storage_reader *reader)
{
	struct backend_operations backend = test_host_memory_backend;
	backend.staging_allocate = allocation_fail;
	backend.staging_retire = staging_retire;
	backend.launch_wait = card_wait;
	backend.peer_connect = peer_connect;
	struct device_card *cards[TEST_CARDS];
	for (uint32_t card = 0; card < TEST_CARDS; card++)
		cards[card] = (void *)&card_handles[card];
	struct v41_tp2_model_load_request request = {
		.backend = &backend,
		.cards = cards,
		.storage = &test_memory_storage,
		.reader = reader,
		.wait_limit_nanoseconds = TEST_WAIT_NANOSECONDS
	};
	expect(v41_tp2_model_load(model, &request) == STATUS_INVALID_ARGUMENT &&
		       allocations == 0,
	       "unqualified pairs refuse before allocations or work");
	for (uint32_t card = 0; card < TEST_PRO; card++)
		request.qualified_writer_masks[card] = 1u << (card ^ 1u);
	backend.staging_retire = NULL;
	expect(v41_tp2_model_load(model, &request) == STATUS_INVALID_ARGUMENT &&
		       allocations == 0,
	       "missing safe staging retirement refuses before work");
	backend.staging_retire = staging_retire;
	expect(v41_tp2_model_load(model, &request) == STATUS_INJECTED_FAULT &&
		       allocations == TP2_RANKS && retired == TP2_RANKS &&
		       waited == TEST_CARDS && !model->ready &&
		       model->host_arena == NULL && model->host_offsets == NULL,
	       "allocation failure drains every card, retires owned staging and never publishes RAM/ready");
	expect(v41_tp2_model_load(model, &request) == STATUS_INVALID_ARGUMENT &&
		       allocations == TP2_RANKS,
	       "failed load object cannot be reused");
}

int main(void)
{
	struct pack_record *pack = NULL;
	struct v41_tp2_model *model = NULL;
	struct storage_reader *reader = NULL;
	const char *const paths[] = { "tables-only" };
	test_memory_storage_reset();
	const struct test_model_v41_change change = {
		.kind = TEST_MODEL_V41_CHANGE_NONE
	};
	bool ready =
		test_model_v41_draft_pack_open(change, &pack) == STATUS_OK &&
		test_memory_storage.reader_open(paths, 1, &reader) == STATUS_OK;
	struct v41_tp2_model_description description = description_make();
	if (ready)
		ready = v41_tp2_model_create(pack, &description, &model) ==
			STATUS_OK;
	expect(ready,
	       "cold full TP2 load object prepares from independent real-model metadata");
	if (ready) {
		metadata_check(model);
		struct v41_tp2_model *refused = NULL;
		description.limits.context_token_limit = MODEL_CONTEXT_GRANULE;
		expect(v41_tp2_model_create(pack, &description, &refused) ==
				       STATUS_INVALID_ARGUMENT &&
			       refused == NULL,
		       "small-context index scratch refused for TP only");
		description = description_make();
		description.cards[0].memory_free_bytes =
			model->canonical->cards[0].arena_byte_count +
			TEST_RESERVED_BYTES + TEST_KEPT_BYTES - 1u;
		expect(v41_tp2_model_create(pack, &description, &refused) ==
				       STATUS_OUT_OF_MEMORY &&
			       refused == NULL,
		       "arena plus reserved and kept bytes bounded before GPU open");
		load_failure(model, reader);
	}
	v41_tp2_model_destroy(model);
	v41_tp2_model_destroy(NULL);
	if (pack != NULL) {
		original_wide_scratch_check(pack, MODEL_WIDE_LEAST_ROWS);
		original_wide_scratch_check(pack,
					    MODEL_WIDE_LEAST_ROWS * TP2_RANKS);
		original_wide_scratch_check(pack, LINEAR_WIDE_MOST_ROWS);
	}
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	test_memory_storage_reset();
	pack = NULL;
	const struct test_model_v41_change exl3 = {
		.kind = TEST_MODEL_V41_CHANGE_EXL3
	};
	expect(test_model_v41_draft_pack_open(exl3, &pack) == STATUS_OK,
	       "original wide scratch: independent EXL3 metadata opens");
	if (pack != NULL) {
		exl3_wide_scratch_check(pack, MODEL_WIDE_LEAST_ROWS);
		exl3_wide_scratch_check(pack,
					MODEL_WIDE_LEAST_ROWS * TP2_RANKS);
		exl3_wide_scratch_check(pack, LINEAR_WIDE_MOST_ROWS);
	}
	pack_close(pack);
	test_memory_storage_reset();
	if (failures == 0)
		(void)puts(
			"TP2 model: cold metadata/ownership/budget and failed-load retirement fixture passed");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
