/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* MAIN executes. Host stand-ins qualify startup scheduling/oracle/refusals, never actual peer communication. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host_memory_backend.h"
#include "model_v41_tp2_peer.h"

#define PEER_TEST_CARDS 5u
#define PEER_TEST_CMP_CARDS 4u
#define PEER_TEST_ARENA_BYTES (2ull << 20u)
#define PEER_TEST_ROWS 8u
#define PEER_TEST_STEP 128u
#define PEER_TEST_REGION 256u
#define PEER_TEST_WAIT 1000000u
#define PEER_TEST_UNCHANGED 0xDEADBEEFu
#define PEER_TEST_LAUNCHES 32u
#define PEER_TEST_PAIRS (PEER_TEST_CMP_CARDS / TP2_RANKS)
#define PEER_TEST_PAIR_MASK ((1u << TP2_RANKS) - 1u)
#define PEER_TEST_ALL_MASK ((1u << PEER_TEST_CARDS) - 1u)
#define PEER_TEST_CMP_MASK ((1u << PEER_TEST_CMP_CARDS) - 1u)
#define PEER_TEST_PRO_MASK (1u << V41_DATA_TP2_PRO_CARD)
#define PEER_TEST_PAIR_B_LAST 3u
#define PEER_TEST_PAIR_B_FIRST 2u
#define PEER_TEST_VALUE_BUFFERS 2u
#define PEER_TEST_DEADLINE_CYCLES 1000000ull

struct peer_test_context {
	struct device_card *cards[PEER_TEST_CARDS];
	struct ordered_push_arguments arguments[PEER_TEST_CMP_CARDS];
	unsigned char *arenas[PEER_TEST_CMP_CARDS];
	uint32_t pending[PEER_TEST_PAIRS];
	uint32_t ranks;
	uint32_t active_cards;
	uint32_t inactive_callbacks;
	uint32_t launches;
	uint32_t premature_waits;
	uint32_t retired;
	bool corrupt;
	bool deny_peer;
	bool fail_allocation;
	bool lose_completion;
};

static struct peer_test_context context;
static int failures;
static result_status drain_status[PEER_TEST_CARDS];
static uint32_t drain_attempted;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: TP2 peer startup: %s\n", message);
		failures++;
	}
}
static bool active_card(struct device_card *card)
{
	for (uint32_t index = 0; index < context.active_cards; index++)
		if (card != NULL && context.cards[index] == card)
			return true;
	context.inactive_callbacks++;
	return false;
}

MUST_CHECK static result_status guarded_upload(struct device_card *card,
					       uint64_t offset,
					       const void *source,
					       uint64_t bytes, uint64_t *ticket)
{
	return active_card(card) ?
		       test_host_memory_backend.upload_begin(
			       card, offset, source, bytes, ticket) :
		       STATUS_INVALID_ARGUMENT;
}
MUST_CHECK static result_status guarded_download(struct device_card *card,
						 uint64_t offset, void *output,
						 uint64_t bytes,
						 uint64_t *ticket)
{
	return active_card(card) ?
		       test_host_memory_backend.download_begin(
			       card, offset, output, bytes, ticket) :
		       STATUS_INVALID_ARGUMENT;
}
MUST_CHECK static result_status
guarded_upload_wait(struct device_card *card, uint64_t ticket, uint64_t wait)
{
	return active_card(card) ?
		       test_host_memory_backend.upload_wait(card, ticket,
							    wait) :
		       STATUS_INVALID_ARGUMENT;
}
MUST_CHECK static result_status guarded_kernel(struct device_card *card,
					       const char *file,
					       const char *function,
					       uint32_t bytes, uint32_t *kernel)
{
	return active_card(card) ?
		       test_host_memory_backend.kernel_prepare(
			       card, file, function, bytes, kernel) :
		       STATUS_INVALID_ARGUMENT;
}
MUST_CHECK static result_status
guarded_launch(struct device_card *card, const struct kernel_launch *launches,
	       uint32_t count)
{
	return active_card(card) ?
		       test_host_memory_backend.launch_run(card, launches,
							   count) :
		       STATUS_INVALID_ARGUMENT;
}

static bool stand_in(const struct kernel_launch *launch, uint32_t card,
		     unsigned char *arena, uint64_t arena_bytes, void *unused)
{
	(void)unused;
	if (card >= PEER_TEST_CMP_CARDS ||
	    arena_bytes != PEER_TEST_ARENA_BYTES ||
	    launch->argument_byte_count !=
		    sizeof(struct ordered_push_arguments))
		return false;
	(void)memcpy(&context.arguments[card], launch->argument,
		     sizeof(context.arguments[card]));
	context.arenas[card] = arena;
	context.launches++;
	const uint32_t pair = card / context.ranks,
		       first = pair * context.ranks;
	context.pending[pair] |= 1u << (card % context.ranks);
	if (context.pending[pair] != (1u << context.ranks) - 1u)
		return true;
	const struct ordered_push_arguments *const left =
		&context.arguments[first];
	struct step_block steps[PEER_TEST_CMP_CARDS] = { 0 };
	const unsigned char *inputs[PEER_TEST_CMP_CARDS];
	for (uint32_t rank = 0; rank < context.ranks; rank++) {
		const struct ordered_push_arguments *const args =
			&context.arguments[first + rank];
		(void)memcpy(&steps[rank],
			     context.arenas[first + rank] + args->step_offset,
			     sizeof(steps[rank]));
		if (steps[rank].sequence != steps[0].sequence ||
		    args->element_count != left->element_count)
			return false;
		inputs[rank] = context.arenas[first + rank] +
			       args->region_offset + args->input_offset +
			       (steps[0].sequence & 1u) * args->parity_stride;
	}
	const uint64_t parity = steps[0].sequence & 1u;
	for (uint32_t column = 0; column < left->element_count; column++) {
		volatile float sum = 0.0f;
		for (uint32_t rank = 0; rank < context.ranks; rank++) {
			float input_value;
			(void)memcpy(&input_value,
				     inputs[rank] +
					     (uint64_t)column * sizeof(float),
				     sizeof(input_value));
			sum = sum + input_value;
		}
		const float value = sum;
		for (uint32_t rank = 0; rank < context.ranks; rank++) {
			const struct ordered_push_arguments *const args =
				&context.arguments[first + rank];
			(void)memcpy(context.arenas[first + rank] +
					     args->region_offset +
					     args->output_offset +
					     parity * args->parity_stride +
					     column * sizeof(float),
				     &value, sizeof(value));
		}
	}
	if (context.corrupt)
		context.arenas[first][left->region_offset + left->output_offset +
				      parity * left->parity_stride] ^= 1u;
	context.pending[pair] = 0;
	return true;
}

MUST_CHECK static result_status peer_connect(struct device_card *accessor,
					     struct device_card *owner,
					     uint64_t owner_offset,
					     uint64_t word_offset)
{
	if (!active_card(accessor) || !active_card(owner) ||
	    owner_offset >= PEER_TEST_ARENA_BYTES ||
	    word_offset > PEER_TEST_ARENA_BYTES - sizeof(uint64_t))
		return STATUS_INVALID_ARGUMENT;
	return context.deny_peer && accessor != owner ? STATUS_UNSUPPORTED :
							STATUS_OK;
}

MUST_CHECK static result_status allocate(uint64_t bytes, void **output)
{
	return context.fail_allocation ?
		       STATUS_OUT_OF_MEMORY :
		       test_host_memory_backend.staging_allocate(bytes, output);
}

static void retire(void *memory)
{
	context.retired++;
	test_host_memory_backend.staging_free(memory);
}

MUST_CHECK static result_status wait_card(struct device_card *card,
					  uint64_t wait)
{
	if (!active_card(card))
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t index = 0; index < PEER_TEST_CMP_CARDS; index++) {
		if (context.cards[index] == card &&
		    context.pending[index / context.ranks] != 0) {
			context.premature_waits++;
			return STATUS_TIMED_OUT;
		}
	}
	if (context.lose_completion && context.launches != 0) {
		context.lose_completion = false;
		return STATUS_TIMED_OUT;
	}
	return test_host_memory_backend.launch_wait(card, wait);
}

static bool unchanged(const uint32_t masks[PEER_TEST_CARDS])
{
	for (uint32_t card = 0; card < PEER_TEST_CARDS; card++) {
		if (masks[card] != PEER_TEST_UNCHANGED)
			return false;
	}
	return true;
}

static void mask_reset(uint32_t masks[PEER_TEST_CARDS])
{
	for (uint32_t card = 0; card < PEER_TEST_CARDS; card++)
		masks[card] = PEER_TEST_UNCHANGED;
	context.launches = context.retired = context.premature_waits = 0;
	context.pending[0] = context.pending[1] = 0;
}

MUST_CHECK static result_status drain_wait(struct device_card *card,
					   uint64_t wait)
{
	if (!active_card(card))
		return STATUS_INVALID_ARGUMENT;
	if (wait != PEER_TEST_WAIT)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t index = 0; index < PEER_TEST_CARDS; index++) {
		if (context.cards[index] == card) {
			drain_attempted |= 1u << index;
			return drain_status[index];
		}
	}
	return STATUS_INVALID_ARGUMENT;
}

static void drain_cases(struct backend_operations *backend)
{
	backend->launch_wait = drain_wait;
	uint32_t retained = PEER_TEST_UNCHANGED;
	result_status status = v41_tp2_peer_drain(backend, context.cards,
						  PEER_TEST_WAIT, &retained);
	expect(status == STATUS_OK && retained == 0 &&
		       drain_attempted == PEER_TEST_ALL_MASK,
	       "successful teardown attempts every participant before permitting any close");
	drain_attempted = 0;
	drain_status[0] = STATUS_TIMED_OUT;
	drain_status[PEER_TEST_PAIR_B_LAST] = STATUS_DEVICE_ERROR;
	drain_status[V41_DATA_TP2_PRO_CARD] = STATUS_UNSUPPORTED;
	status = v41_tp2_peer_drain(backend, context.cards, PEER_TEST_WAIT,
				    &retained);
	expect(status == STATUS_TIMED_OUT && retained == PEER_TEST_ALL_MASK &&
		       drain_attempted == PEER_TEST_ALL_MASK,
	       "first failure survives later errors and uncertain completion retains both CMP pairs and PRO");
	(void)memset(drain_status, 0, sizeof(drain_status));
	drain_attempted = 0;
	drain_status[1] = STATUS_DEVICE_ERROR;
	drain_status[PEER_TEST_PAIR_B_FIRST] = STATUS_TIMED_OUT;
	status = v41_tp2_peer_drain(backend, context.cards, PEER_TEST_WAIT,
				    &retained);
	expect(status == STATUS_DEVICE_ERROR &&
		       retained == PEER_TEST_CMP_MASK &&
		       drain_attempted == PEER_TEST_ALL_MASK,
	       "either member's failed wait retains its complete pair while a drained PRO may close");
	(void)memset(drain_status, 0, sizeof(drain_status));
	drain_attempted = 0;
	drain_status[V41_DATA_TP2_PRO_CARD] = STATUS_TIMED_OUT;
	status = v41_tp2_peer_drain(backend, context.cards, PEER_TEST_WAIT,
				    &retained);
	expect(status == STATUS_TIMED_OUT && retained == PEER_TEST_PRO_MASK &&
		       drain_attempted == PEER_TEST_ALL_MASK,
	       "PRO uncertainty retains only PRO after attempting all cards");
	struct device_card *partial[PEER_TEST_CARDS] = { NULL, context.cards[1],
							 NULL, NULL, NULL };
	drain_status[1] = STATUS_TIMED_OUT;
	drain_attempted = 0;
	status =
		v41_tp2_peer_drain(backend, partial, PEER_TEST_WAIT, &retained);
	expect(status == STATUS_TIMED_OUT && retained == PEER_TEST_PAIR_MASK &&
		       drain_attempted == (1u << 1u),
	       "partial-open cleanup skips absent cards and expands an uncertain CMP to its pair");
	backend->launch_wait = NULL;
	status = v41_tp2_peer_drain(backend, context.cards, PEER_TEST_WAIT,
				    &retained);
	expect(status == STATUS_UNSUPPORTED && retained == PEER_TEST_ALL_MASK,
	       "missing drain support retains every open group");
	retained = PEER_TEST_UNCHANGED;
	expect(v41_tp2_peer_drain(NULL, context.cards, PEER_TEST_WAIT,
				  &retained) == STATUS_INVALID_ARGUMENT &&
		       retained == PEER_TEST_UNCHANGED,
	       "NULL backend leaves output untouched");
	expect(v41_tp2_peer_drain(backend, NULL, PEER_TEST_WAIT, &retained) ==
			       STATUS_INVALID_ARGUMENT &&
		       retained == PEER_TEST_UNCHANGED,
	       "NULL cards leave output untouched");
	expect(v41_tp2_peer_drain(backend, context.cards, 0, &retained) ==
			       STATUS_INVALID_ARGUMENT &&
		       retained == PEER_TEST_UNCHANGED,
	       "zero drain limit leaves output untouched");
	expect(v41_tp2_peer_drain(backend, context.cards, PEER_TEST_WAIT,
				  NULL) == STATUS_INVALID_ARGUMENT,
	       "NULL drain output refused");
	backend->launch_wait = wait_card;
}

static void quad_drain_cases(struct backend_operations *backend)
{
	backend->launch_wait = drain_wait;
	uint32_t retained = PEER_TEST_UNCHANGED;
	result_status status = v41_tp_group_drain(
		backend, context.cards, PEER_TEST_WAIT, TP4_RANKS, &retained);
	expect(status == STATUS_OK && retained == 0 &&
		       drain_attempted == PEER_TEST_CMP_MASK,
	       "quad teardown attempts only four opened participants, ignoring the NULL fifth slot");
	drain_attempted = 0;
	drain_status[PEER_TEST_PAIR_B_FIRST] = STATUS_TIMED_OUT;
	status = v41_tp_group_drain(backend, context.cards, PEER_TEST_WAIT,
				    TP4_RANKS, &retained);
	expect(status == STATUS_TIMED_OUT && retained == PEER_TEST_CMP_MASK &&
		       drain_attempted == PEER_TEST_CMP_MASK,
	       "one uncertain quad participant retains all four actual peers");
	(void)memset(drain_status, 0, sizeof(drain_status));
	struct device_card *partial[PEER_TEST_CARDS] = { NULL, context.cards[1],
							 NULL, NULL, NULL };
	drain_status[1] = STATUS_TIMED_OUT;
	drain_attempted = 0;
	status = v41_tp_group_drain(backend, partial, PEER_TEST_WAIT, TP4_RANKS,
				    &retained);
	expect(status == STATUS_TIMED_OUT && retained == PEER_TEST_CMP_MASK &&
		       drain_attempted == (1u << 1u),
	       "partial quad drain retains the complete group while touching only its opened member");
	backend->launch_wait = NULL;
	status = v41_tp_group_drain(backend, context.cards, PEER_TEST_WAIT,
				    TP4_RANKS, &retained);
	expect(status == STATUS_UNSUPPORTED && retained == PEER_TEST_CMP_MASK,
	       "missing quad drain callback retains four active cards and no inactive fifth bit");
	(void)memset(drain_status, 0, sizeof(drain_status));
	drain_attempted = 0;
	backend->launch_wait = wait_card;
}

static void admission_cases(struct v41_tp2_model *model,
			    struct backend_operations *backend,
			    uint32_t masks[PEER_TEST_CARDS])
{
	mask_reset(masks);
	const uint32_t last_active = context.active_cards - 1u;
	struct device_card *const saved_card = context.cards[last_active];
	context.cards[last_active] = NULL;
	expect(v41_tp2_peer_qualify(model, backend, context.cards,
				    PEER_TEST_WAIT,
				    masks) == STATUS_INVALID_ARGUMENT &&
		       unchanged(masks),
	       "every active card is required before qualification");
	context.cards[last_active] = saved_card;
	if (context.active_cards == PEER_TEST_CMP_CARDS) {
		model->canonical->card_count = PEER_TEST_CARDS;
		expect(v41_tp2_peer_qualify(model, backend, context.cards,
					    PEER_TEST_WAIT,
					    masks) == STATUS_INVALID_ARGUMENT &&
			       unchanged(masks),
		       "QUAD_ALL refuses a five-card canonical count");
		model->canonical->card_count = PEER_TEST_CMP_CARDS;
		model->description.tensor_layout = V41_DATA_TP_LAYOUT_PAIRS;
		expect(v41_tp2_peer_qualify(model, backend, context.cards,
					    PEER_TEST_WAIT,
					    masks) == STATUS_INVALID_ARGUMENT &&
			       unchanged(masks),
		       "ordinary layouts still require all five cards");
		model->description.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL;
		struct v41_tp2_stage_plan *const saved_stage =
			model->stages[last_active];
		model->stages[last_active] = NULL;
		expect(v41_tp2_peer_qualify(model, backend, context.cards,
					    PEER_TEST_WAIT,
					    masks) == STATUS_INVALID_ARGUMENT &&
			       unchanged(masks),
		       "QUAD_ALL requires every active stage");
		model->stages[last_active] = saved_stage;
		saved_stage->description.rank_count = TP2_RANKS;
		expect(v41_tp2_peer_qualify(model, backend, context.cards,
					    PEER_TEST_WAIT,
					    masks) == STATUS_INVALID_ARGUMENT &&
			       unchanged(masks),
		       "QUAD_ALL refuses pair-width stage metadata");
		saved_stage->description.rank_count = TP4_RANKS;
		const uint32_t callbacks = context.inactive_callbacks;
		expect(backend->launch_wait(
			       context.cards[V41_DATA_TP2_PRO_CARD],
			       PEER_TEST_WAIT) == STATUS_INVALID_ARGUMENT &&
			       context.inactive_callbacks == callbacks + 1u,
		       "the callback guard detects any attempt to use the NULL inactive fifth card");
		context.inactive_callbacks = callbacks;
	}
}

static void run_fixture(uint32_t ranks, uint32_t active_cards)
{
	(void)memset(&context, 0, sizeof(context));
	(void)memset(drain_status, 0, sizeof(drain_status));
	drain_attempted = 0;
	context.ranks = ranks;
	context.active_cards = active_cards;
	struct v41_tp2_model *const model = calloc(1, sizeof(*model));
	if (model == NULL) {
		failures++;
		return;
	}
	model->canonical = calloc(1, sizeof(*model->canonical));
	if (model->canonical == NULL) {
		free(model);
		failures++;
		return;
	}
	model->canonical->card_count = active_cards;
	model->description.tensor_layout = active_cards == PEER_TEST_CMP_CARDS ?
						   V41_DATA_TP_LAYOUT_QUAD_ALL :
						   V41_DATA_TP_LAYOUT_PAIRS;
	model->canonical->limits.row_capacity = PEER_TEST_ROWS;
	const uint64_t bytes =
		(uint64_t)PEER_TEST_ROWS * MODEL_HIDDEN * sizeof(float);
	const uint64_t signal = (uint64_t)ORDERED_PUSH_PHASES * context.ranks *
				ORDERED_COLLECTIVE_LINE_BYTES;
	const uint64_t stride =
		(signal + (context.ranks + PEER_TEST_VALUE_BUFFERS) * bytes +
		 V41_DATA_ARRAY_ALIGNMENT_BYTES - 1u) /
		V41_DATA_ARRAY_ALIGNMENT_BYTES * V41_DATA_ARRAY_ALIGNMENT_BYTES;
	test_host_memory_backend_reset(active_cards, PEER_TEST_ARENA_BYTES);
	for (uint32_t card = 0; card < active_cards; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		if (test_host_memory_backend.card_open(
			    card, PEER_TEST_ARENA_BYTES,
			    &context.cards[card]) != STATUS_OK)
			goto failed_setup;
		model->description.cards[card].arithmetic_class =
			ARITHMETIC_CLASS_SM_80;
		model->canonical->cards[card].arena_byte_count =
			PEER_TEST_ARENA_BYTES;
		model->canonical->cards[card].array_offset[V41_DATA_ARRAY_STEP] =
			PEER_TEST_STEP;
		if (card == V41_DATA_TP2_PRO_CARD)
			continue;
		model->stages[card] = calloc(1, sizeof(*model->stages[card]));
		if (model->stages[card] == NULL)
			goto failed_setup;
		struct v41_tp2_stage_plan *const stage = model->stages[card];
		stage->channels = calloc(1, sizeof(*stage->channels));
		if (stage->channels == NULL)
			goto failed_setup;
		stage->channel_rows = PEER_TEST_ROWS;
		stage->channel_count = 1;
		stage->description.card = card;
		stage->description.rank = card % context.ranks;
		stage->description.rank_count = context.ranks;
		stage->channels[0].layer =
			context.ranks == TP4_RANKS ||
					card < PEER_TEST_PAIR_B_FIRST ?
				0u :
				V41_DATA_TP2_PAIR_A_LAYERS;
		stage->channels[0]
			.collective = (struct ordered_push_description){
			.arena_byte_count = PEER_TEST_ARENA_BYTES,
			.arguments = { .region_offset = PEER_TEST_REGION,
				       .input_offset = signal,
				       .output_offset = signal + bytes,
				       .inbox_offset =
					       signal +
					       PEER_TEST_VALUE_BUFFERS * bytes,
				       .step_offset = PEER_TEST_STEP,
				       .parity_stride = stride,
				       .deadline_cycles =
					       PEER_TEST_DEADLINE_CYCLES,
				       .inbox_rank_stride = bytes,
				       .element_count =
					       PEER_TEST_ROWS * MODEL_HIDDEN,
				       .rank_count = context.ranks,
				       .rank = card % context.ranks,
				       .payload_format = ORDERED_PUSH_FP32 }
		};
	}
	expect(test_host_memory_backend_set_stand_in(
		       KERNEL_FILE_ORDERED_PUSH, KERNEL_ORDERED_PUSH, stand_in,
		       NULL) == STATUS_OK,
	       "stand-in registered");
	struct backend_operations backend = test_host_memory_backend;
	backend.peer_connect = peer_connect;
	backend.staging_allocate = allocate;
	backend.staging_retire = retire;
	backend.launch_wait = wait_card;
	backend.upload_begin = guarded_upload;
	backend.download_begin = guarded_download;
	backend.upload_wait = guarded_upload_wait;
	backend.kernel_prepare = guarded_kernel;
	backend.launch_run = guarded_launch;
	uint32_t masks[PEER_TEST_CARDS];
	mask_reset(masks);
	result_status status = v41_tp2_peer_qualify(
		model, &backend, context.cards, PEER_TEST_WAIT, masks);
	uint32_t expected[PEER_TEST_CARDS] = { 0 };
	for (uint32_t card = 0; card < active_cards; card++)
		expected[card] = card < PEER_TEST_CMP_CARDS ?
					 ((1u << ranks) - 1u)
						 << (card / ranks * ranks) :
					 1u << card;
	expect(status == STATUS_OK &&
		       memcmp(masks, expected, sizeof(masks)) == 0,
	       "active allocation-oracle groups publish tested writer masks and leave the inactive slot zero");
	expect(context.launches == PEER_TEST_LAUNCHES &&
		       context.premature_waits == 0 && context.retired == 1,
	       "both sizes/parities/reuses enqueue buddies before waits and retire one scratch block");
	mask_reset(masks);
	context.corrupt = true;
	status = v41_tp2_peer_qualify(model, &backend, context.cards,
				      PEER_TEST_WAIT, masks);
	expect(status == STATUS_DEVICE_ERROR && unchanged(masks) &&
		       context.retired == 1,
	       "one corrupt output bit refuses qualification and retires scratch without publishing masks");
	context.corrupt = false;
	mask_reset(masks);
	context.deny_peer = true;
	status = v41_tp2_peer_qualify(model, &backend, context.cards,
				      PEER_TEST_WAIT, masks);
	expect(status == STATUS_UNSUPPORTED && unchanged(masks) &&
		       context.launches == 0 && context.retired == 1,
	       "denied actual peer grant refuses before protocol launch");
	context.deny_peer = false;
	mask_reset(masks);
	context.lose_completion = true;
	status = v41_tp2_peer_qualify(model, &backend, context.cards,
				      PEER_TEST_WAIT, masks);
	expect(status == STATUS_TIMED_OUT && unchanged(masks) &&
		       context.launches == context.ranks &&
		       context.retired == 1,
	       "lost completion keeps masks unpublished and relinquishes staging");
	mask_reset(masks);
	context.fail_allocation = true;
	status = v41_tp2_peer_qualify(model, &backend, context.cards,
				      PEER_TEST_WAIT, masks);
	expect(status == STATUS_OUT_OF_MEMORY && unchanged(masks) &&
		       context.retired == 0,
	       "allocation failure publishes nothing");
	context.fail_allocation = false;
	backend.peer_connect = NULL;
	expect(v41_tp2_peer_qualify(model, &backend, context.cards,
				    PEER_TEST_WAIT,
				    masks) == STATUS_UNSUPPORTED &&
		       unchanged(masks),
	       "missing peer callback refused");
	backend.peer_connect = peer_connect;
	model->load_attempted = true;
	expect(v41_tp2_peer_qualify(model, &backend, context.cards,
				    PEER_TEST_WAIT,
				    masks) == STATUS_INVALID_ARGUMENT &&
		       unchanged(masks),
	       "late qualification refused");
	model->load_attempted = false;
	expect(v41_tp2_peer_qualify(model, &backend, context.cards, 0, masks) ==
			       STATUS_INVALID_ARGUMENT &&
		       unchanged(masks),
	       "unbounded/nonwaiting startup refused");
	admission_cases(model, &backend, masks);
	if (active_cards == PEER_TEST_CARDS)
		drain_cases(&backend);
	else
		quad_drain_cases(&backend);
	uint32_t retained = PEER_TEST_UNCHANGED;
	status = v41_tp_group_drain(&backend, context.cards, PEER_TEST_WAIT,
				    ranks, &retained);
	expect(status == STATUS_OK && retained == 0,
	       "fixture drains all cards before any actual close");
	expect(context.inactive_callbacks == 0,
	       "startup, fault paths and drains never invoke inactive-card callbacks");
cleanup:
	for (uint32_t card = 0; card < PEER_TEST_CARDS; card++) {
		if (context.cards[card] != NULL)
			test_host_memory_backend.card_close(context.cards[card],
							    PEER_TEST_WAIT);
		if (model->stages[card] != NULL) {
			free(model->stages[card]->channels);
			free(model->stages[card]);
		}
	}
	free(model->canonical);
	free(model);
	return;
failed_setup:
	failures++;
	goto cleanup;
}

int main(void)
{
	run_fixture(TP2_RANKS, PEER_TEST_CARDS);
	run_fixture(TP4_RANKS, PEER_TEST_CMP_CARDS);
	(void)printf(
		"TP pair/quad peer startup host fixture: %s (no GPU/protocol proof)\n",
		failures == 0 ? "PASS" : "FAIL");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
