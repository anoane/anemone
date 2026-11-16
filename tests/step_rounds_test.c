/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host-only fixture. MAIN executes it; the build script only compiles and links. */
#include "step_rounds.h"
#include "step_driver.h"
#include "byte_identity.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CONTEXT 256u
#define TEST_VOCABULARY 4096u
#define TEST_END_TOKEN (TEST_VOCABULARY - 1u)
#define TEST_DRAFTS 5u
#define TEST_OWNER_ROWS (TEST_DRAFTS + 1u)
#define TEST_ROWS (STEP_BATCH_MAXIMUM_SESSIONS * TEST_OWNER_ROWS)
#define TEST_SEQUENCE ((UINT64_C(1) << 32) + UINT64_C(17))
#define TEST_POISON 0xa5
/* Fixture values: the positions and tokens of slot s start at a stride of their own, so no two slots share one. */
#define TEST_FIRST_POSITION 10u
#define TEST_POSITION_STRIDE 16u
#define TEST_FIRST_TOKEN 10u
#define TEST_TOKEN_STRIDE 20u
#define TEST_REJECTED_TOKEN 700u
#define TEST_FIRST_PROPOSAL 1000u
#define TEST_PROPOSAL_STRIDE 16u
#define TEST_CONFIDENCE_STEP 0.125f
#define TEST_LOGIT_STEP 0.25f
#define TEST_DRAFT_CONFIDENCE 0.25f
/* Masks of sessions: every one of the 8, and slots 0, TEST_MIDDLE_SLOT and TEST_LAST_SLOT. */
#define TEST_EVERY_SESSION 0xffu
#define TEST_KEPT_SLOTS 0x89u
#define TEST_MIDDLE_SLOT 3u
#define TEST_LAST_SLOT 7u
#define TEST_SESSIONS 8u
#define TEST_SECOND_PROPOSED_ROW 2u
/* A stride coprime to TEST_ROWS: a permutation of the rows that moves rows between owners. */
#define TEST_PERMUTATION_STRIDE 17u
#define TEST_ORDINARY_POSITIONS 128u
#define TEST_END_TOKEN_CASES 2u
#define TEST_SEQUENCE_HIGH_BIT (UINT64_C(1) << 32)
#define TEST_UNKNOWN_ROW_FLAG UINT32_C(0x80000000)
#define TEST_SHORT_ROUND_CAPACITY 2
#define TEST_HALF_ROUND_CAPACITY 4
#define TEST_ROUND_OFFSET 1024
#define TEST_RESULT_AREA_BYTES 2048
#define REQUIRE(condition)                                             \
	do {                                                           \
		if (!(condition)) {                                    \
			(void)fprintf(stderr, "%s:%d: %s\n", __FILE__, \
				      __LINE__, #condition);           \
			exit(1);                                       \
		}                                                      \
	} while (0)

struct fixture {
	struct step_row_record rows[STEP_BATCH_MAXIMUM_ROWS];
	struct step_row_result results[STEP_BATCH_MAXIMUM_ROWS];
	struct step_round rounds[STEP_BATCH_MAXIMUM_SESSIONS];
	struct step_result header;
	struct step_round_description description;
};

static uint32_t checks, refusals;

static void bind(struct fixture *fixture)
{
	fixture->description.rows = fixture->rows;
	fixture->description.results = fixture->results;
	fixture->description.rounds = fixture->rounds;
	fixture->description.result_header = &fixture->header;
}

static void copy(struct fixture *destination, const struct fixture *source)
{
	*destination = *source;
	bind(destination);
}

static uint32_t owner_row(uint32_t slot, uint32_t position)
{
	return slot * TEST_OWNER_ROWS + position;
}

static void round_set(struct fixture *fixture, uint32_t slot, uint32_t accepted,
		      uint32_t proposed, bool end_token)
{
	REQUIRE(accepted <= proposed && proposed <= TEST_DRAFTS);
	const uint32_t first = owner_row(slot, 0);
	for (uint32_t position = 0; position <= proposed; position++) {
		const uint32_t row = first + position;
		fixture->results[row].token_id =
			position < accepted ? fixture->rows[row + 1u].token_id :
					      (int32_t)(TEST_REJECTED_TOKEN +
							slot + position);
	}
	if (end_token)
		fixture->results[first + accepted].token_id = TEST_END_TOKEN;
	struct step_round *const round = &fixture->rounds[slot];
	memset(round, 0, sizeof(*round));
	round->sequence = TEST_SEQUENCE;
	round->session_slot = slot;
	round->emitted_count = accepted + 1u;
	round->anchor_position =
		fixture->rows[first].position + (int32_t)round->emitted_count;
	round->anchor_token = fixture->results[first + accepted].token_id;
	round->proposal_count = TEST_DRAFTS;
	for (uint32_t position = 0; position < TEST_DRAFTS; position++) {
		round->proposals[position] =
			(int32_t)(TEST_FIRST_PROPOSAL +
				  slot * TEST_PROPOSAL_STRIDE + position);
		round->confidence_logits[position] =
			(float)(position + 1u) * TEST_CONFIDENCE_STEP;
	}
}

static void initialize(struct fixture *fixture)
{
	memset(fixture, 0, sizeof(*fixture));
	fixture->description = (struct step_round_description){
		.sequence = TEST_SEQUENCE,
		.row_count = TEST_ROWS,
		.context_token_limit = TEST_CONTEXT,
		.session_capacity = STEP_BATCH_MAXIMUM_SESSIONS,
		.vocabulary_size = TEST_VOCABULARY,
		.draft_positions = TEST_DRAFTS,
		.end_token_id = TEST_END_TOKEN,
		.round_capacity = STEP_BATCH_MAXIMUM_SESSIONS
	};
	fixture->header = (struct step_result){ TEST_SEQUENCE, TEST_ROWS, 0 };
	bind(fixture);
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		for (uint32_t position = 0; position < TEST_OWNER_ROWS;
		     position++) {
			const uint32_t row = owner_row(slot, position);
			fixture->rows[row] = (struct step_row_record){
				.position =
					(int32_t)(TEST_FIRST_POSITION +
						  slot * TEST_POSITION_STRIDE +
						  position),
				.token_id = (int32_t)(TEST_FIRST_TOKEN +
						      slot * TEST_TOKEN_STRIDE +
						      position),
				.session_slot = slot,
				.flags = STEP_ROW_LOGITS |
					 (position == 0 ? STEP_ROW_DRAFTS :
							  STEP_ROW_PROPOSED)
			};
			fixture->results[row].logit =
				(float)row * -TEST_LOGIT_STEP;
		}
		round_set(fixture, slot, slot % TEST_OWNER_ROWS, TEST_DRAFTS,
			  false);
	}
}

static struct step_round_table
accept(const struct step_round_description *description, uint32_t mask)
{
	struct step_round_table output;
	memset(&output, TEST_POISON, sizeof(output));
	REQUIRE(step_rounds_validate(description, &output) == STATUS_OK);
	REQUIRE(output.valid_mask == mask);
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		if ((mask & (1u << slot)) == 0) {
			struct step_round inactive = { 0 };
			inactive.session_slot = STEP_NO_SESSION;
			REQUIRE(bytes_identical(&inactive, &output.rounds[slot],
						sizeof(inactive)));
		} else {
			const uint32_t index =
				description->round_capacity <= 1u ? 0 : slot;
			REQUIRE(bytes_identical(&description->rounds[index],
						&output.rounds[slot],
						sizeof(output.rounds[slot])));
		}
	}
	checks++;
	return output;
}

static void refuse(const struct step_round_description *description)
{
	struct step_round_table output, before;
	memset(&output, TEST_POISON, sizeof(output));
	memcpy(&before, &output, sizeof(before));
	REQUIRE(step_rounds_validate(description, &output) != STATUS_OK);
	REQUIRE(bytes_identical(&before, &output, sizeof(output)));
	refusals++;
}

static void owner_only(struct fixture *fixture, uint32_t slot,
		       uint32_t proposed)
{
	const uint32_t first = owner_row(slot, 0);
	memmove(fixture->rows, &fixture->rows[first],
		(proposed + 1u) * sizeof(fixture->rows[0]));
	memmove(fixture->results, &fixture->results[first],
		(proposed + 1u) * sizeof(fixture->results[0]));
	fixture->description.row_count = proposed + 1u;
	fixture->header.row_count = proposed + 1u;
}

static void ordinary(struct fixture *fixture)
{
	for (uint32_t row = 0; row < fixture->description.row_count; row++) {
		fixture->rows[row].flags = 0;
		fixture->results[row].token_id = STEP_NO_TOKEN;
	}
	fixture->description.draft_positions = 0;
	fixture->description.rounds = NULL;
}

static void valid_cases(const struct fixture *base)
{
	struct fixture fixture;
	const struct step_round_table reference =
		accept(&base->description, TEST_EVERY_SESSION);
	for (uint32_t shift = 0; shift < STEP_BATCH_MAXIMUM_SESSIONS; shift++) {
		copy(&fixture, base);
		for (uint32_t row = 0; row < TEST_ROWS; row++) {
			const uint32_t source =
				((row / TEST_OWNER_ROWS + shift) %
				 STEP_BATCH_MAXIMUM_SESSIONS) *
					TEST_OWNER_ROWS +
				row % TEST_OWNER_ROWS;
			fixture.rows[row] = base->rows[source];
			fixture.results[row] = base->results[source];
		}
		const struct step_round_table output =
			accept(&fixture.description, TEST_EVERY_SESSION);
		REQUIRE(bytes_identical(&reference, &output, sizeof(output)));
	}
	/* The old permutation oracle is deliberately not a cache-write admission oracle. */
	for (uint32_t shift = 0; shift < TEST_ROWS; shift++) {
		copy(&fixture, base);
		for (uint32_t row = 0; row < TEST_ROWS; row++) {
			const uint32_t source =
				(row * TEST_PERMUTATION_STRIDE + shift) %
				TEST_ROWS;
			fixture.rows[row] = base->rows[source];
			fixture.results[row] = base->results[source];
		}
		refuse(&fixture.description);
	}
	copy(&fixture, base);
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		if (slot == 0 || slot == TEST_MIDDLE_SLOT ||
		    slot == TEST_LAST_SLOT)
			continue;
		fixture.rows[owner_row(slot, 0)].flags &=
			~(uint32_t)STEP_ROW_DRAFTS;
		memset(&fixture.rounds[slot], TEST_POISON,
		       sizeof(fixture.rounds[slot]));
	}
	(void)accept(&fixture.description, TEST_KEPT_SLOTS);
	for (uint32_t capacity = 0; capacity <= 1u; capacity++) {
		for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS;
		     slot++) {
			copy(&fixture, base);
			owner_only(&fixture, slot, TEST_DRAFTS);
			fixture.rounds[0] = base->rounds[slot];
			fixture.description.round_capacity = capacity;
			const struct step_round_table output =
				accept(&fixture.description, 1u << slot);
			REQUIRE(bytes_identical(
				&output.rounds[slot], &reference.rounds[slot],
				sizeof(reference.rounds[slot])));
		}
	}
	copy(&fixture, base);
	ordinary(&fixture);
	(void)accept(&fixture.description, 0);
	for (uint32_t row = 0; row < TEST_ROWS; row++)
		fixture.rows[row].flags = STEP_ROW_FRONT;
	(void)accept(&fixture.description, 0);
	for (uint32_t row = 0; row < STEP_BATCH_MAXIMUM_ROWS; row++) {
		fixture.rows[row] = (struct step_row_record){
			.position = (int32_t)(row % TEST_ORDINARY_POSITIONS),
			.token_id = 1,
			.session_slot = row / TEST_ORDINARY_POSITIONS,
			.flags = 0
		};
		fixture.results[row].token_id = STEP_NO_TOKEN;
	}
	fixture.description.row_count = STEP_BATCH_MAXIMUM_ROWS;
	fixture.header.row_count = STEP_BATCH_MAXIMUM_ROWS;
	(void)accept(&fixture.description, 0);
	copy(&fixture, base);
	fixture.results[0].logit = NAN;
	fixture.results[1].logit = -INFINITY;
	fixture.rounds[TEST_LAST_SLOT].confidence_logits[TEST_DRAFTS] = -0.0f;
	(void)accept(&fixture.description, TEST_EVERY_SESSION);
	/* A change to one valid session cannot change any other session's round. */
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		copy(&fixture, base);
		round_set(&fixture, slot, (slot + 1u) % TEST_OWNER_ROWS,
			  TEST_DRAFTS, false);
		const struct step_round_table output =
			accept(&fixture.description, TEST_EVERY_SESSION);
		for (uint32_t other = 0; other < STEP_BATCH_MAXIMUM_SESSIONS;
		     other++)
			if (other != slot)
				REQUIRE(bytes_identical(
					&output.rounds[other],
					&reference.rounds[other],
					sizeof(output.rounds[other])));
	}
}

static void acceptance_cases(const struct fixture *base)
{
	struct fixture fixture;
	for (uint32_t proposed = 0; proposed <= TEST_DRAFTS; proposed++) {
		for (uint32_t accepted = 0; accepted <= proposed; accepted++) {
			for (uint32_t eos = 0; eos < TEST_END_TOKEN_CASES;
			     eos++) {
				copy(&fixture, base);
				round_set(&fixture, 0, accepted, proposed,
					  eos != 0);
				if (eos != 0 && accepted < proposed)
					fixture.rows[accepted + 1u].token_id =
						TEST_END_TOKEN;
				owner_only(&fixture, 0, proposed);
				(void)accept(&fixture.description, 1u);
				if (accepted < proposed) {
					fixture.rounds[0].emitted_count++;
					fixture.rounds[0].anchor_position++;
					fixture.rounds[0].anchor_token =
						fixture.results[accepted + 1u]
							.token_id;
					refuse(&fixture.description);
				}
			}
		}
	}
	/* Earlier ordinary prompt rows are permitted before a DRAFTS anchor. */
	copy(&fixture, base);
	owner_only(&fixture, 0, TEST_DRAFTS);
	memmove(&fixture.rows[1], fixture.rows,
		TEST_OWNER_ROWS * sizeof(fixture.rows[0]));
	memmove(&fixture.results[1], fixture.results,
		TEST_OWNER_ROWS * sizeof(fixture.results[0]));
	fixture.rows[0] = fixture.rows[1];
	fixture.rows[0].position--;
	fixture.rows[0].flags = 0;
	fixture.results[0].token_id = STEP_NO_TOKEN;
	fixture.header.row_count++;
	fixture.description.row_count++;
	(void)accept(&fixture.description, 1u);
	for (uint32_t drafts = 1; drafts <= STEP_DRAFT_MOST; drafts++) {
		copy(&fixture, base);
		const uint32_t proposed = drafts < TEST_DRAFTS ? drafts :
								 TEST_DRAFTS;
		round_set(&fixture, 0, proposed, proposed, false);
		owner_only(&fixture, 0, proposed);
		fixture.description.draft_positions = drafts;
		fixture.description.session_capacity = 1;
		fixture.description.round_capacity = 0;
		fixture.rounds[0].proposal_count = drafts;
		for (uint32_t position = 0; position < STEP_DRAFT_MOST;
		     position++) {
			fixture.rounds[0].proposals[position] =
				position < drafts ?
					(int32_t)(TEST_FIRST_PROPOSAL +
						  position) :
					0;
			fixture.rounds[0].confidence_logits[position] =
				position < drafts ? -TEST_DRAFT_CONFIDENCE :
						    0.0f;
		}
		(void)accept(&fixture.description, 1u);
	}
}

enum round_corruption {
	ROUND_SEQUENCE,
	ROUND_SESSION_SLOT,
	ROUND_NO_EMITTED,
	ROUND_EMITTED_PAST_OWNER,
	ROUND_ANCHOR_POSITION,
	ROUND_ANCHOR_TOKEN,
	ROUND_PROPOSAL_COUNT,
	ROUND_RESERVED,
	ROUND_NEGATIVE_PROPOSAL,
	ROUND_PROPOSAL_PAST_VOCABULARY,
	ROUND_NAN_CONFIDENCE,
	ROUND_INFINITE_CONFIDENCE,
	ROUND_NEGATIVE_INFINITE_CONFIDENCE,
	ROUND_PROPOSAL_PAST_DRAFTS,
	ROUND_CONFIDENCE_PAST_DRAFTS,
	ROUND_NAN_PAST_DRAFTS,
	ROUND_CORRUPTIONS
};

static void corrupt_rounds(const struct fixture *base)
{
	struct fixture fixture;
	/* Every owner is independently corrupted, including the last owner after seven valid ones. */
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		for (uint32_t field = 0; field < ROUND_CORRUPTIONS; field++) {
			copy(&fixture, base);
			struct step_round *const round = &fixture.rounds[slot];
			switch (field) {
			case ROUND_SEQUENCE:
				round->sequence ^= TEST_SEQUENCE_HIGH_BIT;
				break;
			case ROUND_SESSION_SLOT:
				round->session_slot =
					(slot + 1u) % TEST_SESSIONS;
				break;
			case ROUND_NO_EMITTED:
				round->emitted_count = 0;
				break;
			case ROUND_EMITTED_PAST_OWNER:
				round->emitted_count = TEST_OWNER_ROWS + 1u;
				break;
			case ROUND_ANCHOR_POSITION:
				round->anchor_position++;
				break;
			case ROUND_ANCHOR_TOKEN:
				round->anchor_token++;
				break;
			case ROUND_PROPOSAL_COUNT:
				round->proposal_count--;
				break;
			case ROUND_RESERVED:
				round->reserved_zero = 1;
				break;
			case ROUND_NEGATIVE_PROPOSAL:
				round->proposals[0] = -1;
				break;
			case ROUND_PROPOSAL_PAST_VOCABULARY:
				round->proposals[0] = TEST_VOCABULARY;
				break;
			case ROUND_NAN_CONFIDENCE:
				round->confidence_logits[0] = NAN;
				break;
			case ROUND_INFINITE_CONFIDENCE:
				round->confidence_logits[0] = INFINITY;
				break;
			case ROUND_NEGATIVE_INFINITE_CONFIDENCE:
				round->confidence_logits[0] = -INFINITY;
				break;
			case ROUND_PROPOSAL_PAST_DRAFTS:
				round->proposals[TEST_DRAFTS] = 1;
				break;
			case ROUND_CONFIDENCE_PAST_DRAFTS:
				round->confidence_logits[TEST_DRAFTS] = 1.0f;
				break;
			case ROUND_NAN_PAST_DRAFTS:
				round->confidence_logits[TEST_DRAFTS] = NAN;
				break;
			default:
				REQUIRE(false);
				break;
			}
			refuse(&fixture.description);
		}
	}
}

enum row_corruption {
	ROW_SESSION_PAST_LAST,
	ROW_NEGATIVE_POSITION,
	ROW_POSITION_PAST_CONTEXT,
	ROW_NEGATIVE_TOKEN,
	ROW_TOKEN_PAST_VOCABULARY,
	ROW_UNKNOWN_FLAG,
	ROW_FRONT_FLAG,
	ROW_PROPOSED_ANCHOR,
	ROW_NO_LOGITS,
	ROW_SECOND_ANCHOR,
	ROW_REPEATED_POSITION,
	ROW_POSITION_GAP,
	ROW_UNPROPOSED_DRAFT,
	ROW_NO_RESULT_TOKEN,
	ROW_RESULT_PAST_VOCABULARY,
	ROW_CONTEXT_SHORT,
	ROW_HEADER_SEQUENCE,
	ROW_HEADER_ROW_COUNT,
	ROW_HEADER_RESERVED,
	ROW_ORDINARY_TOKEN_ZERO,
	ROW_PROPOSED_WITHOUT_DRAFTS,
	ROW_LOGITS_ONLY,
	ROW_CORRUPTIONS
};

static void corrupt_rows(const struct fixture *base)
{
	struct fixture fixture;
	const uint32_t anchor = owner_row(TEST_LAST_SLOT, 0);
	for (uint32_t field = 0; field < ROW_CORRUPTIONS; field++) {
		copy(&fixture, base);
		struct step_row_record *const row = &fixture.rows[anchor];
		switch (field) {
		case ROW_SESSION_PAST_LAST:
			row->session_slot = STEP_BATCH_MAXIMUM_SESSIONS;
			break;
		case ROW_NEGATIVE_POSITION:
			row->position = -1;
			break;
		case ROW_POSITION_PAST_CONTEXT:
			row->position = TEST_CONTEXT;
			break;
		case ROW_NEGATIVE_TOKEN:
			row->token_id = -1;
			break;
		case ROW_TOKEN_PAST_VOCABULARY:
			row->token_id = TEST_VOCABULARY;
			break;
		case ROW_UNKNOWN_FLAG:
			row->flags |= TEST_UNKNOWN_ROW_FLAG;
			break;
		case ROW_FRONT_FLAG:
			row->flags |= STEP_ROW_FRONT;
			break;
		case ROW_PROPOSED_ANCHOR:
			row->flags |= STEP_ROW_PROPOSED;
			break;
		case ROW_NO_LOGITS:
			row->flags &= ~(uint32_t)STEP_ROW_LOGITS;
			break;
		case ROW_SECOND_ANCHOR:
			fixture.rows[anchor + 1u].flags = row->flags;
			break;
		case ROW_REPEATED_POSITION:
			fixture.rows[anchor + 1u].position = row->position;
			break;
		case ROW_POSITION_GAP:
			fixture.rows[anchor + TEST_DRAFTS].position++;
			break;
		case ROW_UNPROPOSED_DRAFT:
			fixture.rows[anchor + 1u].flags = STEP_ROW_LOGITS;
			break;
		case ROW_NO_RESULT_TOKEN:
			fixture.results[anchor].token_id = STEP_NO_TOKEN;
			break;
		case ROW_RESULT_PAST_VOCABULARY:
			fixture.results[anchor].token_id = TEST_VOCABULARY;
			break;
		case ROW_CONTEXT_SHORT:
			fixture.description.context_token_limit =
				(uint32_t)row->position + TEST_OWNER_ROWS;
			break;
		case ROW_HEADER_SEQUENCE:
			fixture.header.sequence ^= TEST_SEQUENCE_HIGH_BIT;
			break;
		case ROW_HEADER_ROW_COUNT:
			fixture.header.row_count--;
			break;
		case ROW_HEADER_RESERVED:
			fixture.header.reserved_zero = 1;
			break;
		case ROW_ORDINARY_TOKEN_ZERO:
			ordinary(&fixture);
			fixture.results[anchor].token_id = 0;
			break;
		case ROW_PROPOSED_WITHOUT_DRAFTS:
			row->flags = STEP_ROW_LOGITS | STEP_ROW_PROPOSED;
			break;
		case ROW_LOGITS_ONLY:
			row->flags = STEP_ROW_LOGITS;
			fixture.rows[anchor + TEST_SECOND_PROPOSED_ROW].flags =
				STEP_ROW_LOGITS;
			break;
		default:
			REQUIRE(false);
			break;
		}
		refuse(&fixture.description);
	}
	copy(&fixture, base);
	ordinary(&fixture);
	fixture.rows[0].flags = STEP_ROW_FRONT;
	refuse(&fixture.description);
}

enum description_corruption {
	DESCRIPTION_NO_ROWS,
	DESCRIPTION_NO_RESULTS,
	DESCRIPTION_NO_HEADER,
	DESCRIPTION_NO_ROUNDS,
	DESCRIPTION_ZERO_SEQUENCE,
	DESCRIPTION_RESERVED,
	DESCRIPTION_NO_ROW,
	DESCRIPTION_ROWS_PAST_BATCH,
	DESCRIPTION_NO_CONTEXT,
	DESCRIPTION_CONTEXT_MAXIMUM,
	DESCRIPTION_NO_SESSION,
	DESCRIPTION_SESSIONS_PAST_BATCH,
	DESCRIPTION_NO_VOCABULARY,
	DESCRIPTION_VOCABULARY_MAXIMUM,
	DESCRIPTION_NO_DRAFT,
	DESCRIPTION_DRAFTS_PAST_MOST,
	DESCRIPTION_END_PAST_VOCABULARY,
	DESCRIPTION_SHORT_ROUND_CAPACITY,
	DESCRIPTION_ROUND_CAPACITY_MAXIMUM,
	DESCRIPTION_NO_ROUND_CAPACITY,
	DESCRIPTION_ONE_ROUND,
	DESCRIPTION_HALF_ROUND_CAPACITY,
	DESCRIPTION_SEQUENCE,
	DESCRIPTION_CORRUPTIONS
};

static void corrupt_description(const struct fixture *base)
{
	struct fixture fixture;
	for (uint32_t field = 0; field < DESCRIPTION_CORRUPTIONS; field++) {
		copy(&fixture, base);
		struct step_round_description *const description =
			&fixture.description;
		switch (field) {
		case DESCRIPTION_NO_ROWS:
			description->rows = NULL;
			break;
		case DESCRIPTION_NO_RESULTS:
			description->results = NULL;
			break;
		case DESCRIPTION_NO_HEADER:
			description->result_header = NULL;
			break;
		case DESCRIPTION_NO_ROUNDS:
			description->rounds = NULL;
			break;
		case DESCRIPTION_ZERO_SEQUENCE:
			description->sequence = 0;
			break;
		case DESCRIPTION_RESERVED:
			description->reserved_zero = 1;
			break;
		case DESCRIPTION_NO_ROW:
			description->row_count = 0;
			break;
		case DESCRIPTION_ROWS_PAST_BATCH:
			description->row_count = STEP_BATCH_MAXIMUM_ROWS + 1u;
			break;
		case DESCRIPTION_NO_CONTEXT:
			description->context_token_limit = 0;
			break;
		case DESCRIPTION_CONTEXT_MAXIMUM:
			description->context_token_limit = UINT32_MAX;
			break;
		case DESCRIPTION_NO_SESSION:
			description->session_capacity = 0;
			break;
		case DESCRIPTION_SESSIONS_PAST_BATCH:
			description->session_capacity =
				STEP_BATCH_MAXIMUM_SESSIONS + 1u;
			break;
		case DESCRIPTION_NO_VOCABULARY:
			description->vocabulary_size = 0;
			break;
		case DESCRIPTION_VOCABULARY_MAXIMUM:
			description->vocabulary_size = UINT32_MAX;
			break;
		case DESCRIPTION_NO_DRAFT:
			description->draft_positions = 0;
			break;
		case DESCRIPTION_DRAFTS_PAST_MOST:
			description->draft_positions = STEP_DRAFT_MOST + 1u;
			break;
		case DESCRIPTION_END_PAST_VOCABULARY:
			description->end_token_id = TEST_VOCABULARY;
			break;
		case DESCRIPTION_SHORT_ROUND_CAPACITY:
			description->round_capacity = TEST_SHORT_ROUND_CAPACITY;
			break;
		case DESCRIPTION_ROUND_CAPACITY_MAXIMUM:
			description->round_capacity = UINT32_MAX;
			break;
		case DESCRIPTION_NO_ROUND_CAPACITY:
			description->round_capacity = 0;
			break;
		case DESCRIPTION_ONE_ROUND:
			description->round_capacity = 1;
			break;
		case DESCRIPTION_HALF_ROUND_CAPACITY:
			description->round_capacity = TEST_HALF_ROUND_CAPACITY;
			break;
		case DESCRIPTION_SEQUENCE:
			description->sequence ^= TEST_SEQUENCE_HIGH_BIT;
			break;
		default:
			REQUIRE(false);
			break;
		}
		refuse(description);
	}
	refuse(NULL);
	REQUIRE(step_rounds_validate(&base->description, NULL) ==
		STATUS_INVALID_ARGUMENT);
	refusals++;
}

enum overlap {
	OVERLAP_ROWS,
	OVERLAP_RESULTS,
	OVERLAP_ROUNDS,
	OVERLAP_HEADER,
	OVERLAP_DESCRIPTION,
	OVERLAPS
};

static void overlap_cases(const struct fixture *base)
{
	/* This valid storage is never dereferenced through the wrong view: overlap is rejected first. */
	union storage {
		struct step_round_table table;
		struct step_row_record rows[TEST_ROWS];
		struct step_row_result results[TEST_ROWS];
		struct step_round rounds[STEP_BATCH_MAXIMUM_SESSIONS];
		struct step_result header;
		struct step_round_description description;
	} storage, before;
	for (uint32_t field = 0; field < OVERLAPS; field++) {
		struct step_round_description description = base->description;
		memset(&storage, TEST_POISON, sizeof(storage));
		const struct step_round_description *input = &description;
		switch (field) {
		case OVERLAP_ROWS:
			description.rows = storage.rows;
			break;
		case OVERLAP_RESULTS:
			description.results = storage.results;
			break;
		case OVERLAP_ROUNDS:
			description.rounds = storage.rounds;
			break;
		case OVERLAP_HEADER:
			description.result_header = &storage.header;
			break;
		case OVERLAP_DESCRIPTION:
			storage.description = description;
			input = &storage.description;
			break;
		default:
			REQUIRE(false);
			break;
		}
		memcpy(&before, &storage, sizeof(storage));
		REQUIRE(step_rounds_validate(input, &storage.table) ==
			STATUS_INVALID_ARGUMENT);
		REQUIRE(bytes_identical(&before, &storage, sizeof(storage)));
		refusals++;
	}
}

static void slab_bytes(unsigned char *bytes, const struct fixture *fixture)
{
	for (uint32_t slab = 0;
	     slab < STEP_BATCH_MAXIMUM_ROWS / STEP_BATCH_SLAB_ROWS; slab++) {
		const uint32_t first = slab * STEP_BATCH_SLAB_ROWS;
		const uint32_t remaining =
			fixture->description.row_count > first ?
				fixture->description.row_count - first :
				0u;
		const uint32_t count = remaining < STEP_BATCH_SLAB_ROWS ?
					       remaining :
					       STEP_BATCH_SLAB_ROWS;
		const struct step_result header = { fixture->header.sequence,
						    count, 0 };
		unsigned char *const area =
			bytes + (size_t)slab * STEP_BATCH_SLAB_RESULT_BYTES;
		memcpy(area, &header, sizeof(header));
		if (count != 0)
			memcpy(area + STEP_RESULT_BYTES,
			       &fixture->results[first],
			       (size_t)count * sizeof(fixture->results[0]));
	}
}

enum header_corruption {
	HEADER_SEQUENCE,
	HEADER_ROW_COUNT,
	HEADER_RESERVED,
	HEADER_CORRUPTIONS
};

static void driver_boundary(const struct fixture *base)
{
	struct step_driver empty = { 0 };
	const struct step_driver retained_empty = empty;
	REQUIRE(!step_rounds_is_batch(NULL) && !step_rounds_is_batch(&empty));
	REQUIRE(step_rounds_domain_attach(&empty, TEST_VOCABULARY,
					  TEST_END_TOKEN) ==
		STATUS_INVALID_ARGUMENT);
	REQUIRE(bytes_identical(&empty, &retained_empty, sizeof(empty)));

	struct program_card_record card = { .result_offset = 0,
					    .round_offset = TEST_ROUND_OFFSET };
	struct execution_program program = {
		.cards = &card,
		.kind = PROGRAM_BATCH,
		.row_count = STEP_BATCH_MAXIMUM_ROWS,
		.limits = { .context_token_limit = TEST_CONTEXT,
			    .session_capacity = STEP_BATCH_MAXIMUM_SESSIONS,
			    .draft_positions = TEST_DRAFTS }
	};
	const struct execution_program *programs[] = { &program };
	struct step_driver driver = { .programs = programs,
				      .program_count = 1,
				      .largest_row_count =
					      STEP_BATCH_MAXIMUM_ROWS };
	REQUIRE(step_rounds_domain_attach(&driver, TEST_VOCABULARY,
					  TEST_END_TOKEN) == STATUS_OK);
	REQUIRE(step_rounds_domain_attach(&driver, TEST_VOCABULARY,
					  TEST_END_TOKEN) == STATUS_OK);
	REQUIRE(step_rounds_domain_attach(&driver, TEST_VOCABULARY,
					  TEST_END_TOKEN - 1u) != STATUS_OK);
	unsigned char bytes[TEST_RESULT_AREA_BYTES] = { 0 };
	slab_bytes(bytes, base);
	memcpy(bytes + card.round_offset, base->rounds, sizeof(base->rounds));
	struct step_row_result output[TEST_ROWS], before[TEST_ROWS];
	memset(output, TEST_POISON, sizeof(output));
	memcpy(before, output, sizeof(before));
	struct step_round corrupted = base->rounds[TEST_LAST_SLOT];
	corrupted.anchor_token ^= 1;
	memcpy(bytes + card.round_offset + TEST_LAST_SLOT * sizeof(corrupted),
	       &corrupted, sizeof(corrupted));
	REQUIRE(step_batch_result_take(&driver, bytes, TEST_SEQUENCE,
				       base->rows, TEST_ROWS,
				       output) == STATUS_DEVICE_ERROR);
	REQUIRE(bytes_identical(before, output, sizeof(before)) &&
		driver.last_rounds.valid_mask == 0);
	/* Isolate every header fault: all rounds are valid before the header is changed. */
	memcpy(bytes + card.round_offset, base->rounds, sizeof(base->rounds));
	for (uint32_t slab = 0;
	     slab < STEP_BATCH_MAXIMUM_ROWS / STEP_BATCH_SLAB_ROWS; slab++) {
		for (uint32_t field = 0; field < HEADER_CORRUPTIONS; field++) {
			slab_bytes(bytes, base);
			struct step_result header;
			unsigned char *const area =
				bytes +
				(size_t)slab * STEP_BATCH_SLAB_RESULT_BYTES;
			memcpy(&header, area, sizeof(header));
			if (field == HEADER_SEQUENCE)
				header.sequence ^= TEST_SEQUENCE_HIGH_BIT;
			if (field == HEADER_ROW_COUNT)
				header.row_count++;
			if (field == HEADER_RESERVED)
				header.reserved_zero = 1;
			memcpy(area, &header, sizeof(header));
			REQUIRE(step_batch_result_take(
					&driver, bytes, TEST_SEQUENCE,
					base->rows, TEST_ROWS,
					output) == STATUS_DEVICE_ERROR);
			REQUIRE(bytes_identical(output, before,
						sizeof(before)) &&
				driver.last_rounds.valid_mask == 0);
		}
	}
	slab_bytes(bytes, base);
	memcpy(bytes + card.round_offset, base->rounds, sizeof(base->rounds));
	REQUIRE(step_batch_result_take(&driver, bytes, TEST_SEQUENCE,
				       base->rows, TEST_ROWS,
				       output) == STATUS_OK);
	REQUIRE(bytes_identical(output, base->results, sizeof(output)) &&
		driver.last_rounds.valid_mask == TEST_EVERY_SESSION);
	driver.has_finished = true;
	struct step_round next, retained;
	memset(&next, TEST_POISON, sizeof(next));
	memcpy(&retained, &next, sizeof(next));
	REQUIRE(step_round_take(&driver, &next) == STATUS_INVALID_ARGUMENT);
	REQUIRE(bytes_identical(&next, &retained, sizeof(next)));
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		REQUIRE(step_round_take_for_session(&driver, slot, &next) ==
			STATUS_OK);
		REQUIRE(bytes_identical(&next, &base->rounds[slot],
					sizeof(next)));
	}
	REQUIRE(step_rounds_domain_attach(&driver, TEST_VOCABULARY,
					  TEST_END_TOKEN) != STATUS_OK);
	struct fixture ordinary_fixture;
	copy(&ordinary_fixture, base);
	ordinary(&ordinary_fixture);
	slab_bytes(bytes, &ordinary_fixture);
	REQUIRE(step_batch_result_take(&driver, bytes, TEST_SEQUENCE,
				       ordinary_fixture.rows, TEST_ROWS,
				       output) == STATUS_OK);
	REQUIRE(driver.last_rounds.valid_mask == 0);
	REQUIRE(step_round_take_for_session(&driver, TEST_LAST_SLOT, &next) ==
		STATUS_INVALID_ARGUMENT);
	REQUIRE(step_round_take(&driver, &next) == STATUS_OK &&
		next.session_slot == STEP_NO_SESSION);
}

int main(void)
{
	struct fixture base;
	initialize(&base);
	driver_boundary(&base);
	valid_cases(&base);
	acceptance_cases(&base);
	corrupt_rounds(&base);
	corrupt_rows(&base);
	corrupt_description(&base);
	overlap_cases(&base);
	printf("tp_batch round table PASS: %u valid, %u refused; record=%u bytes, owners=%u; no CUDA/NVML\n",
	       checks, refusals, STEP_ROUND_BYTES, STEP_BATCH_MAXIMUM_SESSIONS);
	return 0;
}
