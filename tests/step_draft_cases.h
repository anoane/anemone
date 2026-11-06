/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_STEP_DRAFT_CASES_H
#define TEST_STEP_DRAFT_CASES_H
#include <stddef.h>
#ifdef TEST_STEP_FIXTURE_IMPLEMENTATION

#include "anemone/model_layout.h"

#define DEVICE_DRAFT_OPERATION 5u
#define DEVICE_DRAFT_ROUND_OFFSET (RESULT_OFFSET + 256u)
#define DEVICE_DRAFT_EMITTED 2u
#define DEVICE_DRAFT_MODES 2u /* eager and pipeline */
enum device_draft_corruption {
	DEVICE_DRAFT_SOUND,
	DEVICE_DRAFT_SEQUENCE,
	DEVICE_DRAFT_SESSION,
	DEVICE_DRAFT_COUNT,
	DEVICE_DRAFT_POSITION,
	DEVICE_DRAFT_TOKEN,
	DEVICE_DRAFT_PROPOSAL_COUNT,
	DEVICE_DRAFT_PROPOSAL_TOKEN,
	DEVICE_DRAFT_CONFIDENCE,
	DEVICE_DRAFT_RESERVED,
	DEVICE_DRAFT_CORRUPTIONS
};

/* The device drafter's tail stands in for a verify plus confidence launch. Target results stay those of the
 * ordinary function model. Each corruption changes one field independently, after otherwise sound results. */
static enum device_draft_corruption device_draft_break;
static bool device_draft_stand_in(const struct kernel_launch *launch,
				  uint32_t card, unsigned char *arena,
				  uint64_t bytes, void *context)
{
	struct fill_record record;
	memcpy(&record, launch->argument, sizeof(record));
	if (record.value != DEVICE_DRAFT_OPERATION) {
		return model_stand_in(launch, card, arena, bytes, context);
	}
	struct step_block block;
	memcpy(&block, arena + STEP_OFFSET, sizeof(block));
	struct step_round round = { .sequence = block.sequence,
				    .session_slot = STEP_NO_SESSION,
				    .emitted_count = DEVICE_DRAFT_EMITTED,
				    .proposal_count = MODEL_DRAFT_BLOCK };
	for (uint32_t row = 0; row < block.row_count; row++) {
		struct step_row_record input;
		memcpy(&input,
		       arena + STEP_OFFSET + STEP_BLOCK_BYTES_VALUE +
			       (size_t)row * STEP_ROW_BYTES_VALUE,
		       sizeof(input));
		if ((input.flags & STEP_ROW_DRAFTS) != 0) {
			round.session_slot = input.session_slot;
			round.anchor_position =
				input.position + (int32_t)DEVICE_DRAFT_EMITTED;
		}
		if (input.session_slot == round.session_slot &&
		    input.position + 1 == round.anchor_position) {
			struct step_row_result result;
			memcpy(&result,
			       arena + RESULT_OFFSET + STEP_RESULT_BYTES +
				       (size_t)row * STEP_ROW_RESULT_BYTES,
			       sizeof(result));
			round.anchor_token = result.token_id;
		}
	}
	for (uint32_t position = 0; position < MODEL_DRAFT_BLOCK; position++) {
		round.proposals[position] = (int32_t)position + 1;
		round.confidence_logits[position] = (float)position;
	}
	switch (device_draft_break) {
	case DEVICE_DRAFT_SOUND:
		break;
	case DEVICE_DRAFT_SEQUENCE:
		round.sequence++;
		break;
	case DEVICE_DRAFT_SESSION:
		round.session_slot = SESSIONS;
		break;
	case DEVICE_DRAFT_COUNT:
		round.emitted_count = LIMIT_ROWS;
		break;
	case DEVICE_DRAFT_POSITION:
		round.anchor_position++;
		break;
	case DEVICE_DRAFT_TOKEN:
		round.anchor_token =
			(round.anchor_token + 1) % (int32_t)VOCABULARY;
		break;
	case DEVICE_DRAFT_PROPOSAL_COUNT:
		round.proposal_count--;
		break;
	case DEVICE_DRAFT_PROPOSAL_TOKEN:
		round.proposals[0] = (int32_t)VOCABULARY;
		break;
	case DEVICE_DRAFT_CONFIDENCE:
		round.confidence_logits[0] = INFINITY;
		break;
	case DEVICE_DRAFT_RESERVED:
		round.reserved_zero = 1;
		break;
	case DEVICE_DRAFT_CORRUPTIONS:
		return false;
	}
	memcpy(arena + DEVICE_DRAFT_ROUND_OFFSET, &round, sizeof(round));
	return true;
}
/* Append one result-card tail to both thin row shapes. The underlying target model, state and hand-offs stay the
 * fixture's; only this segment writes the round, so a normal step or graph must stop before it. */
static void
device_fixture_build(struct fixture *fixture,
		     struct segment_record segments[PROGRAMS][SEGMENTS + 1u],
		     uint32_t kernel)
{
	fixture_build(fixture, kernel);
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		memcpy(segments[program], fixture->segments[program],
		       sizeof(fixture->segments[program]));
		struct program_card_record *const result =
			&fixture->program_cards[program][LAST_CARD];
		const struct fill_record record = {
			.value = DEVICE_DRAFT_OPERATION
		};
		prepare_launch_fill(&fixture->launches[program][LAST_CARD]
						      [result->launch_count],
				    kernel,
				    fixture->programs[program].row_count, 1, 1,
				    0, &record, sizeof(record));
		segments[program][SEGMENTS] =
			(struct segment_record){ .card = LAST_CARD,
						 .layer = SEGMENT_NO_LAYER,
						 .first_launch =
							 result->launch_count,
						 .launch_count = 1 };
		result->launch_count++;
		result->round_offset = DEVICE_DRAFT_ROUND_OFFSET;
		fixture->programs[program].segments = segments[program];
		fixture->programs[program].segment_count = SEGMENTS + 1u;
		fixture->programs[program].limits.draft_positions =
			MODEL_DRAFT_BLOCK;
	}
}

#endif
#endif
