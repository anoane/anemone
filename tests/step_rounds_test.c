/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host-only fixture. MAIN executes it; the build script only compiles and links. */
#include "step_rounds.h"
#include "step_driver.h"

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
#define REQUIRE(condition)                                                 \
	do {                                                               \
		if (!(condition)) {                                        \
			fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
				#condition);                               \
			exit(1);                                           \
		}                                                          \
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
					      (int32_t)(700u + slot + position);
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
			(int32_t)(1000u + slot * 16u + position);
		round->confidence_logits[position] =
			(float)(position + 1u) * 0.125f;
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
					(int32_t)(10u + slot * 16u + position),
				.token_id =
					(int32_t)(10u + slot * 20u + position),
				.session_slot = slot,
				.flags = STEP_ROW_LOGITS |
					 (position == 0 ? STEP_ROW_DRAFTS :
							  STEP_ROW_PROPOSED)
			};
			fixture->results[row].logit = (float)row * -0.25f;
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
			REQUIRE(memcmp(&inactive, &output.rounds[slot],
				       sizeof(inactive)) == 0);
		} else {
			const uint32_t index =
				description->round_capacity <= 1u ? 0 : slot;
			REQUIRE(memcmp(&description->rounds[index],
				       &output.rounds[slot],
				       sizeof(output.rounds[slot])) == 0);
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
	REQUIRE(memcmp(&before, &output, sizeof(output)) == 0);
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
		accept(&base->description, 0xffu);
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
			accept(&fixture.description, 0xffu);
		REQUIRE(memcmp(&reference, &output, sizeof(output)) == 0);
	}
	/* The old permutation oracle is deliberately not a cache-write admission oracle. */
	for (uint32_t shift = 0; shift < TEST_ROWS; shift++) {
		copy(&fixture, base);
		for (uint32_t row = 0; row < TEST_ROWS; row++) {
			const uint32_t source = (row * 17u + shift) % TEST_ROWS;
			fixture.rows[row] = base->rows[source];
			fixture.results[row] = base->results[source];
		}
		refuse(&fixture.description);
	}
	copy(&fixture, base);
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		if (slot == 0 || slot == 3 || slot == 7)
			continue;
		fixture.rows[owner_row(slot, 0)].flags &=
			~(uint32_t)STEP_ROW_DRAFTS;
		memset(&fixture.rounds[slot], TEST_POISON,
		       sizeof(fixture.rounds[slot]));
	}
	(void)accept(&fixture.description, 0x89u);
	for (uint32_t capacity = 0; capacity <= 1u; capacity++) {
		for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS;
		     slot++) {
			copy(&fixture, base);
			owner_only(&fixture, slot, TEST_DRAFTS);
			fixture.rounds[0] = base->rounds[slot];
			fixture.description.round_capacity = capacity;
			const struct step_round_table output =
				accept(&fixture.description, 1u << slot);
			REQUIRE(memcmp(&output.rounds[slot],
				       &reference.rounds[slot],
				       sizeof(reference.rounds[slot])) == 0);
		}
	}
	copy(&fixture, base);
	ordinary(&fixture);
	(void)accept(&fixture.description, 0);
	for (uint32_t row = 0; row < TEST_ROWS; row++)
		fixture.rows[row].flags = STEP_ROW_FRONT;
	(void)accept(&fixture.description, 0);
	for (uint32_t row = 0; row < STEP_BATCH_MAXIMUM_ROWS; row++) {
		fixture.rows[row] =
			(struct step_row_record){ .position =
							  (int32_t)(row % 128u),
						  .token_id = 1,
						  .session_slot = row / 128u,
						  .flags = 0 };
		fixture.results[row].token_id = STEP_NO_TOKEN;
	}
	fixture.description.row_count = STEP_BATCH_MAXIMUM_ROWS;
	fixture.header.row_count = STEP_BATCH_MAXIMUM_ROWS;
	(void)accept(&fixture.description, 0);
	copy(&fixture, base);
	fixture.results[0].logit = NAN;
	fixture.results[1].logit = -INFINITY;
	fixture.rounds[7].confidence_logits[TEST_DRAFTS] = -0.0f;
	(void)accept(&fixture.description, 0xffu);
	/* A change to one valid session cannot change any other session's round. */
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		copy(&fixture, base);
		round_set(&fixture, slot, (slot + 1u) % TEST_OWNER_ROWS,
			  TEST_DRAFTS, false);
		const struct step_round_table output =
			accept(&fixture.description, 0xffu);
		for (uint32_t other = 0; other < STEP_BATCH_MAXIMUM_SESSIONS;
		     other++)
			if (other != slot)
				REQUIRE(memcmp(&output.rounds[other],
					       &reference.rounds[other],
					       sizeof(output.rounds[other])) ==
					0);
	}
}

static void acceptance_cases(const struct fixture *base)
{
	struct fixture fixture;
	for (uint32_t proposed = 0; proposed <= TEST_DRAFTS; proposed++) {
		for (uint32_t accepted = 0; accepted <= proposed; accepted++) {
			for (uint32_t eos = 0; eos < 2u; eos++) {
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
					(int32_t)(1000u + position) :
					0;
			fixture.rounds[0].confidence_logits[position] =
				position < drafts ? -0.25f : 0.0f;
		}
		(void)accept(&fixture.description, 1u);
	}
}

static void corrupt_rounds(const struct fixture *base)
{
	struct fixture fixture;
	/* Every owner is independently corrupted, including the last owner after seven valid ones. */
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		for (uint32_t field = 0; field < 16u; field++) {
			copy(&fixture, base);
			struct step_round *const round = &fixture.rounds[slot];
			switch (field) {
			case 0:
				round->sequence ^= UINT64_C(1) << 32;
				break;
			case 1:
				round->session_slot = (slot + 1u) % 8u;
				break;
			case 2:
				round->emitted_count = 0;
				break;
			case 3:
				round->emitted_count = TEST_OWNER_ROWS + 1u;
				break;
			case 4:
				round->anchor_position++;
				break;
			case 5:
				round->anchor_token++;
				break;
			case 6:
				round->proposal_count--;
				break;
			case 7:
				round->reserved_zero = 1;
				break;
			case 8:
				round->proposals[0] = -1;
				break;
			case 9:
				round->proposals[0] = TEST_VOCABULARY;
				break;
			case 10:
				round->confidence_logits[0] = NAN;
				break;
			case 11:
				round->confidence_logits[0] = INFINITY;
				break;
			case 12:
				round->confidence_logits[0] = -INFINITY;
				break;
			case 13:
				round->proposals[TEST_DRAFTS] = 1;
				break;
			case 14:
				round->confidence_logits[TEST_DRAFTS] = 1.0f;
				break;
			case 15:
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

static void corrupt_rows(const struct fixture *base)
{
	struct fixture fixture;
	const uint32_t anchor = owner_row(7u, 0);
	for (uint32_t field = 0; field < 22u; field++) {
		copy(&fixture, base);
		struct step_row_record *const row = &fixture.rows[anchor];
		switch (field) {
		case 0:
			row->session_slot = STEP_BATCH_MAXIMUM_SESSIONS;
			break;
		case 1:
			row->position = -1;
			break;
		case 2:
			row->position = TEST_CONTEXT;
			break;
		case 3:
			row->token_id = -1;
			break;
		case 4:
			row->token_id = TEST_VOCABULARY;
			break;
		case 5:
			row->flags |= UINT32_C(0x80000000);
			break;
		case 6:
			row->flags |= STEP_ROW_FRONT;
			break;
		case 7:
			row->flags |= STEP_ROW_PROPOSED;
			break;
		case 8:
			row->flags &= ~(uint32_t)STEP_ROW_LOGITS;
			break;
		case 9:
			fixture.rows[anchor + 1u].flags = row->flags;
			break;
		case 10:
			fixture.rows[anchor + 1u].position = row->position;
			break;
		case 11:
			fixture.rows[anchor + TEST_DRAFTS].position++;
			break;
		case 12:
			fixture.rows[anchor + 1u].flags = STEP_ROW_LOGITS;
			break;
		case 13:
			fixture.results[anchor].token_id = STEP_NO_TOKEN;
			break;
		case 14:
			fixture.results[anchor].token_id = TEST_VOCABULARY;
			break;
		case 15:
			fixture.description.context_token_limit =
				(uint32_t)row->position + TEST_OWNER_ROWS;
			break;
		case 16:
			fixture.header.sequence ^= UINT64_C(1) << 32;
			break;
		case 17:
			fixture.header.row_count--;
			break;
		case 18:
			fixture.header.reserved_zero = 1;
			break;
		case 19:
			ordinary(&fixture);
			fixture.results[anchor].token_id = 0;
			break;
		case 20:
			row->flags = STEP_ROW_LOGITS | STEP_ROW_PROPOSED;
			break;
		case 21:
			row->flags = STEP_ROW_LOGITS;
			fixture.rows[anchor + 2u].flags = STEP_ROW_LOGITS;
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

static void corrupt_description(const struct fixture *base)
{
	struct fixture fixture;
	for (uint32_t field = 0; field < 23u; field++) {
		copy(&fixture, base);
		struct step_round_description *const description =
			&fixture.description;
		switch (field) {
		case 0:
			description->rows = NULL;
			break;
		case 1:
			description->results = NULL;
			break;
		case 2:
			description->result_header = NULL;
			break;
		case 3:
			description->rounds = NULL;
			break;
		case 4:
			description->sequence = 0;
			break;
		case 5:
			description->reserved_zero = 1;
			break;
		case 6:
			description->row_count = 0;
			break;
		case 7:
			description->row_count = STEP_BATCH_MAXIMUM_ROWS + 1u;
			break;
		case 8:
			description->context_token_limit = 0;
			break;
		case 9:
			description->context_token_limit = UINT32_MAX;
			break;
		case 10:
			description->session_capacity = 0;
			break;
		case 11:
			description->session_capacity =
				STEP_BATCH_MAXIMUM_SESSIONS + 1u;
			break;
		case 12:
			description->vocabulary_size = 0;
			break;
		case 13:
			description->vocabulary_size = UINT32_MAX;
			break;
		case 14:
			description->draft_positions = 0;
			break;
		case 15:
			description->draft_positions = STEP_DRAFT_MOST + 1u;
			break;
		case 16:
			description->end_token_id = TEST_VOCABULARY;
			break;
		case 17:
			description->round_capacity = 2;
			break;
		case 18:
			description->round_capacity = UINT32_MAX;
			break;
		case 19:
			description->round_capacity = 0;
			break;
		case 20:
			description->round_capacity = 1;
			break;
		case 21:
			description->round_capacity = 4;
			break;
		case 22:
			description->sequence ^= UINT64_C(1) << 32;
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
	for (uint32_t field = 0; field < 5u; field++) {
		struct step_round_description description = base->description;
		memset(&storage, TEST_POISON, sizeof(storage));
		const struct step_round_description *input = &description;
		switch (field) {
		case 0:
			description.rows = storage.rows;
			break;
		case 1:
			description.results = storage.results;
			break;
		case 2:
			description.rounds = storage.rounds;
			break;
		case 3:
			description.result_header = &storage.header;
			break;
		case 4:
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
		REQUIRE(memcmp(&before, &storage, sizeof(storage)) == 0);
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

static void driver_boundary(const struct fixture *base)
{
	struct step_driver empty = { 0 };
	const struct step_driver retained_empty = empty;
	REQUIRE(!step_rounds_is_batch(NULL) && !step_rounds_is_batch(&empty));
	REQUIRE(step_rounds_domain_attach(&empty, TEST_VOCABULARY,
					  TEST_END_TOKEN) ==
		STATUS_INVALID_ARGUMENT);
	REQUIRE(memcmp(&empty, &retained_empty, sizeof(empty)) == 0);

	struct program_card_record card = { .result_offset = 0,
					    .round_offset = 1024 };
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
	unsigned char bytes[2048] = { 0 };
	slab_bytes(bytes, base);
	memcpy(bytes + card.round_offset, base->rounds, sizeof(base->rounds));
	struct step_row_result output[TEST_ROWS], before[TEST_ROWS];
	memset(output, TEST_POISON, sizeof(output));
	memcpy(before, output, sizeof(before));
	struct step_round corrupted = base->rounds[7];
	corrupted.anchor_token ^= 1;
	memcpy(bytes + card.round_offset + 7u * sizeof(corrupted), &corrupted,
	       sizeof(corrupted));
	REQUIRE(step_batch_result_take(&driver, bytes, TEST_SEQUENCE,
				       base->rows, TEST_ROWS,
				       output) == STATUS_DEVICE_ERROR);
	REQUIRE(memcmp(before, output, sizeof(before)) == 0 &&
		driver.last_rounds.valid_mask == 0);
	/* Isolate every header fault: all rounds are valid before the header is changed. */
	memcpy(bytes + card.round_offset, base->rounds, sizeof(base->rounds));
	for (uint32_t slab = 0;
	     slab < STEP_BATCH_MAXIMUM_ROWS / STEP_BATCH_SLAB_ROWS; slab++) {
		for (uint32_t field = 0; field < 3; field++) {
			slab_bytes(bytes, base);
			struct step_result header;
			unsigned char *const area =
				bytes +
				(size_t)slab * STEP_BATCH_SLAB_RESULT_BYTES;
			memcpy(&header, area, sizeof(header));
			if (field == 0)
				header.sequence ^= UINT64_C(1) << 32;
			if (field == 1)
				header.row_count++;
			if (field == 2)
				header.reserved_zero = 1;
			memcpy(area, &header, sizeof(header));
			REQUIRE(step_batch_result_take(
					&driver, bytes, TEST_SEQUENCE,
					base->rows, TEST_ROWS,
					output) == STATUS_DEVICE_ERROR);
			REQUIRE(memcmp(output, before, sizeof(before)) == 0 &&
				driver.last_rounds.valid_mask == 0);
		}
	}
	slab_bytes(bytes, base);
	memcpy(bytes + card.round_offset, base->rounds, sizeof(base->rounds));
	REQUIRE(step_batch_result_take(&driver, bytes, TEST_SEQUENCE,
				       base->rows, TEST_ROWS,
				       output) == STATUS_OK);
	REQUIRE(memcmp(output, base->results, sizeof(output)) == 0 &&
		driver.last_rounds.valid_mask == 0xffu);
	driver.has_finished = true;
	struct step_round next, retained;
	memset(&next, TEST_POISON, sizeof(next));
	memcpy(&retained, &next, sizeof(next));
	REQUIRE(step_round_take(&driver, &next) == STATUS_INVALID_ARGUMENT);
	REQUIRE(memcmp(&next, &retained, sizeof(next)) == 0);
	for (uint32_t slot = 0; slot < STEP_BATCH_MAXIMUM_SESSIONS; slot++) {
		REQUIRE(step_round_take_for_session(&driver, slot, &next) ==
			STATUS_OK);
		REQUIRE(memcmp(&next, &base->rounds[slot], sizeof(next)) == 0);
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
	REQUIRE(step_round_take_for_session(&driver, 7, &next) ==
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
