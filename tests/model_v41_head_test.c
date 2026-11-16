/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Append-only planner/refusal and whole-group failure retirement. No card is opened or model executed. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_v41.h"
#include "byte_identity.h"

#define TEST_ARENA_BYTES (16ull << 20u)
#define TEST_WAIT 1000u
#define CONNECTS_A_RANK \
	2u /* each rank's outgoing connect, then its incoming one */
#define TEST_BANKS 2
#define TEST_ROW_CAPACITY 1024
#define TEST_CONTEXT 16384
#define TEST_SESSIONS 8
/* layer 29 on card 3: a placement the original head's recipe does not take */
#define TEST_PLACED_LAYER 29
#define TEST_OTHER_CARD 3
#define TEST_RETAINED_MARK 123

struct device_card {
	uint32_t index;
};
static uint32_t failures, waited_mask, failed_mask, connect_calls,
	failed_connect_call;

static void expect(bool value, const char *what)
{
	if (!value) {
		(void)fprintf(stderr, "FAILED: %s\n", what);
		failures++;
	}
}

MUST_CHECK static result_status waited(struct device_card *card, uint64_t limit)
{
	expect(limit == TEST_WAIT,
	       "each group member receives the bounded wait");
	waited_mask |= 1u << card->index;
	return (failed_mask >> card->index & 1u) != 0 ? STATUS_DEVICE_ERROR :
							STATUS_OK;
}

MUST_CHECK static result_status connected(struct device_card *card,
					  struct device_card *owner,
					  uint64_t owner_offset,
					  uint64_t word_offset)
{
	const uint32_t rank = connect_calls / CONNECTS_A_RANK;
	const bool outgoing = connect_calls % CONNECTS_A_RANK == 0;
	expect(card->index == (outgoing ? V41_HEAD_RANKS : rank) &&
		       owner->index == (outgoing ? rank : V41_HEAD_RANKS) &&
		       owner_offset == 0 &&
		       word_offset % V41_DATA_ARRAY_ALIGNMENT_BYTES == 0,
	       "attachments name the original allocation and correct directed peer word");
	connect_calls++;
	return connect_calls == failed_connect_call ? STATUS_DEVICE_ERROR :
						      STATUS_OK;
}

/* the bank plan's refusals; the last one is the switch's default */
enum bank_refusal {
	REFUSAL_DISABLED,
	REFUSAL_CONNECTED,
	REFUSAL_INITIALISED,
	REFUSAL_ONE_BANK,
	REFUSAL_NO_BANKS,
	REFUSAL_BANKS_PAST,
	REFUSAL_ARENA_WRAP,
	REFUSALS
};

static void bank_plans(const struct model_plan_record *base)
{
	struct model_plan_record *const copy = malloc(sizeof(*copy));
	struct model_plan_record *const saved = malloc(sizeof(*saved));
	if (copy == NULL || saved == NULL) {
		expect(false,
		       "bank planner controls allocate their private plan snapshots");
		free(copy);
		free(saved);
		return;
	}
	for (uint32_t banks = 1; banks <= V41_HEAD_BANKS; banks++) {
		*copy = *base;
		expect(v41_original_head_banks_plan(copy, banks) == STATUS_OK &&
			       copy->original_head.bank_count == banks,
		       "all bounded bank counts admit append-only planes");
		for (uint32_t card = 0; card < V41_HEAD_CARDS; card++) {
			const uint64_t per_bank =
				(uint64_t)LINEAR_MAXIMUM_ROWS *
				(card == V41_HEAD_RANKS ?
					 MODEL_LOGITS_ROW_BYTES :
					 MODEL_HIDDEN_ROW_BYTES +
						 V41_HEAD_SHARD_ROWS *
							 sizeof(float));
			expect(copy->cards[card].arena_byte_count ==
					       base->cards[card].arena_byte_count +
						       (banks - 1u) *
							       per_bank &&
				       memcmp(&copy->original_head.cards[card],
					      &base->original_head.cards[card],
					      offsetof(
						      struct v41_original_head_card,
						      bank_hidden)) == 0,
			       "only extra planes grow arenas; original head weights/control/planes stay unchanged");
			copy->cards[card].arena_byte_count =
				base->cards[card].arena_byte_count;
		}
		expect(memcmp(copy, base,
			      offsetof(struct model_plan_record,
				       original_head)) == 0,
		       "bank planning preserves every original model/state/vision field");
	}
	for (uint32_t refusal = 0; refusal < REFUSALS; refusal++) {
		*copy = *base;
		uint32_t banks = TEST_BANKS;
		switch (refusal) {
		case REFUSAL_DISABLED:
			copy->original_head.enabled = false;
			break;
		case REFUSAL_CONNECTED:
			copy->original_head.connected = true;
			break;
		case REFUSAL_INITIALISED:
			copy->original_head.initialised = true;
			break;
		case REFUSAL_ONE_BANK:
			copy->original_head.bank_count = 1;
			break;
		case REFUSAL_NO_BANKS:
			banks = 0;
			break;
		case REFUSAL_BANKS_PAST:
			banks = V41_HEAD_BANKS + 1u;
			break;
		default:
			copy->cards[V41_HEAD_RANKS].arena_byte_count =
				UINT64_MAX - V41_DATA_ARRAY_ALIGNMENT_BYTES;
			break;
		}
		*saved = *copy;
		expect(v41_original_head_banks_plan(copy, banks) ==
				       STATUS_INVALID_ARGUMENT &&
			       bytes_identical(copy, saved, sizeof(*copy)),
		       "disabled/late/repeated/out-of-range/overflow bank plans refuse without any mutation");
	}
	expect(v41_original_head_banks_plan(NULL, 1) == STATUS_INVALID_ARGUMENT,
	       "missing bank plan refuses before access");
	free(saved);
	free(copy);
}

int main(void)
{
	struct model_plan_record *const plan = calloc(1, sizeof(*plan));
	struct model_plan_record *const before = calloc(1, sizeof(*before));
	if (plan == NULL || before == NULL) {
		free(plan);
		free(before);
		return EXIT_FAILURE;
	}
	plan->card_count = V41_HEAD_CARDS;
	plan->result_card = V41_HEAD_RANKS;
	plan->limits.row_capacity = TEST_ROW_CAPACITY;
	plan->limits.context_token_limit = TEST_CONTEXT;
	plan->limits.session_capacity = TEST_SESSIONS;
	static const uint32_t last[] = { 8, 15, 22, 29, 40 };
	for (uint32_t layer = 0, card = 0; layer < MODEL_LAYERS; layer++) {
		card += layer == last[card] ? 1u : 0u;
		plan->layer_card[layer] = card;
		plan->cards[card].layers |= 1ull << layer;
	}
	struct card_description_record descriptions[V41_HEAD_CARDS] = { 0 };
	for (uint32_t card = 0; card < V41_HEAD_CARDS; card++) {
		const bool pro = card == V41_HEAD_RANKS;
		(void)snprintf(
			descriptions[card].name, CARD_NAME_CAPACITY, "%s",
			pro ? "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" :
			      "NVIDIA CMP 170HX");
		descriptions[card].arithmetic_class =
			pro ? ARITHMETIC_CLASS_SM_120_NATIVE :
			      ARITHMETIC_CLASS_SM_80;
		plan->cards[card].arena_byte_count = TEST_ARENA_BYTES;
	}
	*before = *plan;
	expect(v41_original_head_plan(plan, descriptions) == STATUS_OK,
	       "canonical native PP5 admits appended target head storage");
	bank_plans(plan);
	for (uint32_t card = 0; card < V41_HEAD_CARDS; card++) {
		expect((card < V41_HEAD_RANKS ?
				plan->original_head.cards[card]
					.weights.byte_offset :
				plan->original_head.cards[card]
					.peer_words[0]
					.byte_offset) >= TEST_ARENA_BYTES &&
			       plan->cards[card].arena_byte_count >
				       TEST_ARENA_BYTES,
		       "append includes the original whole arena");
		plan->cards[card].arena_byte_count = TEST_ARENA_BYTES;
	}
	expect(memcmp(plan, before,
		      offsetof(struct model_plan_record, original_head)) == 0,
	       "every original tensor, array, layer and state field stays unchanged");
	*plan = *before;
	plan->limits.official_reference = 1;
	*before = *plan;
	expect(v41_original_head_plan(plan, descriptions) ==
			       STATUS_UNSUPPORTED &&
		       bytes_identical(plan, before, sizeof(*plan)),
	       "official recipe is refused without changing any plan byte");
	plan->limits.official_reference = 0;
	plan->layer_card[TEST_PLACED_LAYER] = TEST_OTHER_CARD;
	*before = *plan;
	expect(v41_original_head_plan(plan, descriptions) ==
			       STATUS_UNSUPPORTED &&
		       bytes_identical(plan, before, sizeof(*plan)),
	       "alternative placement is refused without mutation");
	plan->layer_card[TEST_PLACED_LAYER] = V41_HEAD_RANKS;
	descriptions[V41_HEAD_RANKS].arithmetic_class = ARITHMETIC_CLASS_SM_120;
	*before = *plan;
	expect(v41_original_head_plan(plan, descriptions) ==
			       STATUS_UNSUPPORTED &&
		       bytes_identical(plan, before, sizeof(*plan)),
	       "emulated PRO arithmetic is refused without mutation");
	descriptions[V41_HEAD_RANKS].arithmetic_class =
		ARITHMETIC_CLASS_SM_120_NATIVE;
	plan->cards[0].arena_byte_count =
		UINT64_MAX - V41_DATA_ARRAY_ALIGNMENT_BYTES;
	*before = *plan;
	expect(v41_original_head_plan(plan, descriptions) ==
			       STATUS_INVALID_ARGUMENT &&
		       bytes_identical(plan, before, sizeof(*plan)),
	       "an overflowing append is refused without publishing a partial plan");
	plan->cards[0].arena_byte_count = TEST_ARENA_BYTES;
	struct device_card storage[V41_HEAD_CARDS];
	struct device_card *cards[V41_HEAD_CARDS];
	for (uint32_t card = 0; card < V41_HEAD_CARDS; card++) {
		storage[card].index = card;
		cards[card] = &storage[card];
	}
	struct backend_operations backend = { .launch_wait = waited,
					      .peer_connect = connected };
	descriptions[V41_HEAD_RANKS].arithmetic_class =
		ARITHMETIC_CLASS_SM_120_NATIVE;
	expect(v41_original_head_plan(plan, descriptions) == STATUS_OK,
	       "connection controls use a freshly admitted append-only plan");
	for (uint32_t call = 1; call <= CONNECTS_A_RANK * V41_HEAD_RANKS;
	     call++) {
		plan->original_head.connected = false;
		connect_calls = 0;
		failed_connect_call = call;
		expect(v41_original_head_connect(plan, &backend, cards) ==
				       STATUS_DEVICE_ERROR &&
			       plan->original_head.connected &&
			       connect_calls == call,
		       "first and later failed attachments retain ownership and propagate failure immediately");
	}
	plan->original_head.connected = false;
	connect_calls = failed_connect_call = 0;
	expect(v41_original_head_connect(plan, &backend, cards) == STATUS_OK &&
		       plan->original_head.connected &&
		       connect_calls == CONNECTS_A_RANK * V41_HEAD_RANKS,
	       "all eight directed current-allocation attachments are required");
	expect(v41_original_head_connect(plan, &backend, cards) ==
			       STATUS_INVALID_ARGUMENT &&
		       connect_calls == CONNECTS_A_RANK * V41_HEAD_RANKS &&
		       plan->original_head.connected,
	       "a connected plan refuses reattachment before any new work");
	uint32_t retained = UINT32_MAX;
	expect(v41_original_head_drain(&backend, cards, TEST_WAIT, &retained) ==
			       STATUS_OK &&
		       retained == 0 &&
		       waited_mask == (1u << V41_HEAD_CARDS) - 1u,
	       "success proves completion of every open peer arena");
	for (uint32_t failed = 0; failed < V41_HEAD_CARDS; failed++) {
		waited_mask = 0;
		failed_mask = 1u << failed;
		expect(v41_original_head_drain(&backend, cards, TEST_WAIT,
					       &retained) ==
				       STATUS_DEVICE_ERROR &&
			       retained == (1u << V41_HEAD_CARDS) - 1u &&
			       waited_mask == retained,
		       "one failed wait retains the complete group and still waits every other owner");
	}
	backend.launch_wait = NULL;
	expect(v41_original_head_drain(&backend, cards, TEST_WAIT, &retained) ==
			       STATUS_UNSUPPORTED &&
		       retained == (1u << V41_HEAD_CARDS) - 1u,
	       "missing completion primitive retains every open member");
	retained = TEST_RETAINED_MARK;
	expect(v41_original_head_drain(NULL, cards, TEST_WAIT, &retained) ==
			       STATUS_INVALID_ARGUMENT &&
		       retained == TEST_RETAINED_MARK,
	       "invalid drain leaves the ownership result unchanged");
	free(before);
	free(plan);
	(void)printf(
		"model_v41_head_test: %u failures; append-only/refusal and five-card retention controls\n",
		failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
