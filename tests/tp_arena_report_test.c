/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* tp_arena_report_test: the arena of each card of an EXL3 TP4 plan of the BF16 recipe (the serving flags: head and
 * draft shards, TC MoE, trunk and head MMA, ordered wide, owner sums, split_overlap mode 6, chunk_core, pool chunk)
 * for THIN and BATCH serving at 1,024 and 2,048 wide rows and the draft's placements (context 65,536, 8 sessions),
 * on the model's tables
 * (model_v41_pack.h): its weights, state, every array of a chunk's rows and the stage's areas, largest first, and
 * the difference between the configurations, so a plan that does not fit shows what its rows hold. It passes when
 * every plan is made; the report is its output. */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_v41_pack.h"
#include "model_v41_tp2_model.h"

#define REPORT_CONTEXT 65536u
#define REPORT_SESSIONS 8u
#define REPORT_REPORTED_BYTES (1ull << 40u)
#define REPORT_CONFIGURATIONS 7u
#define REPORT_PARTS 96u
#define REPORT_MIB (1024.0 * 1024.0)

struct part {
	const char *name;
	uint64_t bytes;
};

/* A serving configuration: its programs (BATCH or THIN), wide rows and draft placement (main.c's flags). */
struct configuration {
	const char *title;
	bool batch;
	uint32_t wide_rows;
	bool draft_layers, draft_shards, batch_draft_rows;
};
static const struct configuration configurations[] = {
	{ "THIN 1,024 (TP draft layers)", false, 1024u, true, true, false },
	{ "THIN 2,048 (TP draft layers)", false, 2048u, true, true, false },
	{ "BATCH 1,024 (gsd03: TP draft layers, batch draft rows)", true, 1024u, true, true, true },
	{ "BATCH 2,048 (TP draft layers, batch draft rows)", true, 2048u, true, true, true },
	{ "THIN 2,048 (gsc02: draft shards)", false, 2048u, false, true, false },
	{ "BATCH 2,048 (draft shards, no batch draft rows)", true, 2048u, false, true, false },
	{ "BATCH 2,048 (no draft shards, batch draft rows)", true, 2048u, false, false, true },
};
_Static_assert(sizeof(configurations) / sizeof(configurations[0]) == REPORT_CONFIGURATIONS, "one row a configuration");

static uint32_t failures;
static struct part parts[REPORT_CONFIGURATIONS][TP4_RANKS][REPORT_PARTS];
static uint32_t part_counts[REPORT_CONFIGURATIONS][TP4_RANKS];
static uint64_t arenas[REPORT_CONFIGURATIONS][TP4_RANKS];

#define ARRAY_NAME(array, ...) #array,
static const char *const array_names[V41_DATA_ARRAY_COUNT] = { V41_DATA_ARRAYS(
	ARRAY_NAME) };
#undef ARRAY_NAME

static void part_add(uint32_t configuration, uint32_t card, const char *name, uint64_t bytes)
{
	if (part_counts[configuration][card] < REPORT_PARTS && bytes != 0)
		parts[configuration][card][part_counts[configuration][card]++] = (struct part){ name, bytes };
}

static int by_bytes(const void *left, const void *right)
{
	const struct part *const a = left, *const b = right;
	return a->bytes < b->bytes ? 1 : a->bytes > b->bytes ? -1 : 0;
}

/* The arrays' sizes from their offsets (laid out one after another in enum order, model_v41.c), the state areas by
 * kind, and the stage's named areas. */
static void card_report(uint32_t configuration,
			const struct v41_tp2_model *model, uint32_t card)
{
	const struct v41_card *const planned = &model->views[card]->cards[card];
	const struct v41_tp2_stage_plan *const stage = model->stages[card];
	uint64_t held[V41_DATA_ARRAY_COUNT + 1u];
	uint32_t order[V41_DATA_ARRAY_COUNT], count = 0;
	for (uint32_t array = 0; array < V41_DATA_ARRAY_COUNT; array++)
		if (planned->array_offset[array] != PROGRAM_NOWHERE)
			order[count++] = array;
	uint64_t state_first = UINT64_MAX;
	for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
		const uint64_t offsets[] = { planned->selection_offset[layer], planned->ring_offset[layer],
					     planned->checkpoint_offset[layer], planned->compressor_offset[layer],
					     planned->pool_offset[layer], planned->key_tiles_offset[layer] };
		for (uint32_t kind = 0; kind < sizeof(offsets) / sizeof(offsets[0]); kind++)
			state_first = offsets[kind] != PROGRAM_NOWHERE &&
						      offsets[kind] <
							      state_first ?
					      offsets[kind] :
					      state_first;
	}
	for (uint32_t index = 0; index < count; index++)
		held[index] = planned->array_offset[order[index]];
	held[count] = state_first;
	for (uint32_t index = 0; index < count; index++)
		part_add(configuration, card, array_names[order[index]], held[index + 1u] - held[index]);
	uint64_t selections = 0, rings = 0, checkpoints = 0, compressors = 0, pools = 0, keys = 0;
	const uint64_t sessions = REPORT_SESSIONS, rows = model->description.limits.row_capacity;
	for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
		selections += planned->selection_offset[layer] !=
					      PROGRAM_NOWHERE ?
				      rows * MODEL_SELECTION_ROW_BYTES :
				      0;
		rings += planned->ring_offset[layer] != PROGRAM_NOWHERE ?
				 sessions * V41_DATA_RING_SESSION_BYTES :
				 0;
		checkpoints +=
			planned->checkpoint_offset[layer] != PROGRAM_NOWHERE ?
				sessions * V41_DATA_CHECKPOINT_SESSION_BYTES :
				0;
		compressors +=
			planned->compressor_offset[layer] != PROGRAM_NOWHERE ?
				sessions * V41_DATA_COMPRESSOR_SESSION_BYTES :
				0;
		pools +=
			planned->pool_offset[layer] != PROGRAM_NOWHERE ?
				sessions * model->views[card]
						   ->pool_session_bytes[layer] :
				0;
		keys += planned->key_tiles_offset[layer] != PROGRAM_NOWHERE ?
				sessions * model->views[card]
						   ->key_session_bytes[layer] :
				0;
	}
	part_add(configuration, card, "state: selections (rows)", selections);
	part_add(configuration, card, "state: rings", rings);
	part_add(configuration, card, "state: checkpoints", checkpoints);
	part_add(configuration, card, "state: compressors", compressors);
	part_add(configuration, card, "state: pools", pools);
	part_add(configuration, card, "state: keys", keys);
	part_add(configuration, card, "weights and slots (to the image)", held[0]);
	const struct {
		const char *name;
		struct linear_range range;
	} named[] = {
		{ "stage: compact", stage->compact },
		{ "stage: exact_hidden", stage->exact_hidden },
		{ "stage: exact_sums", stage->exact_sums },
		{ "stage: dense_input", stage->dense_input },
		{ "stage: exl3_tc_workspace", stage->exl3_tc_workspace },
		{ "stage: exl3_tcp_workspace", stage->exl3_tcp_workspace },
		{ "stage: trunk_mma_partials", stage->trunk_mma_partials },
		{ "stage: router_partials", stage->router_partials },
		{ "stage: batch_results", stage->batch_results },
		{ "stage: batch_logits", stage->batch_logits },
		{ "stage: batch_rounds", stage->batch_rounds },
		{ "stage: batch_drafts", stage->batch_drafts },
		{ "stage: batch_rows", stage->batch_rows },
		{ "stage: draft_area", stage->draft_area },
		{ "stage: sampler_exchange", stage->sampler_exchange },
	};
	for (uint32_t index = 0; index < sizeof(named) / sizeof(named[0]); index++)
		part_add(configuration, card, named[index].name, named[index].range.byte_count);
	part_add(configuration, card, "stage: communication (channels, regions, banks)",
		 stage->communication_byte_count);
	arenas[configuration][card] = planned->arena_byte_count;
}

static bool model_report(uint32_t configuration)
{
	const struct configuration *const chosen = &configurations[configuration];
	const bool batch = chosen->batch;
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = { .kind = TEST_MODEL_V41_CHANGE_EXL3 };
	if (test_model_v41_draft_pack_open(change, &pack) != STATUS_OK) {
		printf("tp_arena_report_test: the fixture did not open\n");
		return false;
	}
	struct v41_tp2_model_description description = {
		.limits = { REPORT_CONTEXT, chosen->wide_rows, REPORT_SESSIONS,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE,
			    MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.wire_format = ORDERED_PUSH_BF16,
		.head_shards = true,
		.draft_shards = chosen->draft_shards,
		.draft_layers = chosen->draft_layers,
		.head_stream = true,
		.exl3_fast_products = true,
		.exl3_wide_entry = true,
		.replicated_embedding = true,
		.exl3_tc_moe = true,
		.trunk_mma = true,
		.head_mma = true,
		.ordered_wide = true,
		.wide_exchange = V41_DATA_TP_EXCHANGE_CE_RINGS,
		.split_overlap = V41_DATA_SPLIT_JOINT,
		.chunk_core = true,
		.pool_chunk = true,
		.batch = batch,
		.batch_wide_head = batch,
		.batch_draft_rows = chosen->batch_draft_rows,
	};
	for (uint32_t card = 0; card < TP4_RANKS; card++) {
		description.cards[card].arithmetic_class =
			ARITHMETIC_CLASS_SM_80;
		description.cards[card].memory_total_bytes = description.cards[card].memory_free_bytes =
			REPORT_REPORTED_BYTES;
	}
	struct v41_tp2_model *model = NULL;
	const result_status created =
		v41_tp2_model_create(pack, &description, &model);
	if (created != STATUS_OK) {
		printf("tp_arena_report_test: the model (%s): %s\n",
		       chosen->title, status_text(created));
		pack_close(pack);
		return false;
	}
	for (uint32_t card = 0; card < TP4_RANKS; card++)
		card_report(configuration, model, card);
	v41_tp2_model_destroy(model);
	pack_close(pack);
	return true;
}

int main(void)
{
	for (uint32_t configuration = 0; configuration < REPORT_CONFIGURATIONS; configuration++)
		if (!model_report(configuration))
			failures++;
	for (uint32_t configuration = 0; configuration < REPORT_CONFIGURATIONS && failures == 0; configuration++) {
		for (uint32_t card = 0; card < TP4_RANKS; card++) {
			printf("== %s, card %" PRIu32 ": arena %.1f MiB\n", configurations[configuration].title, card,
			       (double)arenas[configuration][card] / REPORT_MIB);
			struct part sorted[REPORT_PARTS];
			memcpy(sorted, parts[configuration][card], sizeof(sorted));
			qsort(sorted, part_counts[configuration][card], sizeof(sorted[0]), by_bytes);
			for (uint32_t index = 0; index < part_counts[configuration][card]; index++) {
				if (sorted[index].bytes < (1u << 20u))
					continue;
				uint64_t thin_1024 = 0; /* the same part in THIN 1,024 */
				for (uint32_t other = 0; other < part_counts[0][card]; other++)
					thin_1024 = strcmp(parts[0][card][other].name, sorted[index].name) == 0 ?
							    parts[0][card][other].bytes : thin_1024;
				printf("  %-48s %10.1f MiB  (THIN 1,024 %+9.1f)\n", sorted[index].name,
				       (double)sorted[index].bytes / REPORT_MIB,
				       ((double)sorted[index].bytes - (double)thin_1024) / REPORT_MIB);
			}
		}
	}
	printf(failures == 0 ? "tp_arena_report_test: passed\n" : "tp_arena_report_test: %u failed\n", failures);
	return failures == 0 ? 0 : 1;
}
