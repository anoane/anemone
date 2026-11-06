/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Cold full-program composition: real model metadata, original-format shards, host-only cards, no computation.
 * MAIN owns execution. This fixture is also useful with CUDA_VISIBLE_DEVICES=; it never calls the CUDA backend.
 * Expectations follow the chosen 14/15/11 placement and independently count the replicated state and boundaries.
 */
#include <inttypes.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41_pack.h"
#include "model_v41_tp2.h"
#include "model_v41_tp2_program.h"
#include "model_v41_tp2_model.h"
#include "pack_builder.h"
#include "anemone/pack_digest.h"
#include "anemone/ordered_push_parallel.h"
#include "anemone/ep_filter.h"
#include "anemone/exact_ring_records.h"
#include "anemone/site_front_records.h"
#include "anemone/draft_records.h"
#include "anemone/sampler_records.h"
#include "anemone/attention_core_records.h"
#include "step_driver.h"

#define CARDS 5u
#define PRO_CARD 4u
#define HOST_BYTES 4096u
#define EXPECTED_PAIRS 32u
#define EXPECTED_SEGMENTS 80u
#define EXPECTED_FRONT_SEGMENTS 47u
#define EXPECTED_DRAFT_FIRST 79u
#define EXPECTED_CHECKPOINTS 72u
#define EXPECTED_SESSION_RANGES 96u
#define EXPECTED_HAND_OFFS 12u
#define CONTEXT_TOKENS 16384u
#define ENGRAM_TABLES TEST_RANKS
#define ENGRAM_LOOKBACK 7u
#define ENGRAM_HEADS 4u
#define ENGRAM_ROWS 24u
#define ENGRAM_PRIME 13u
#define ENGRAM_UNITS 4u
#define ENGRAM_EXTENTS 4u
#define ENGRAM_TENSORS 9u
#define ENGRAM_CONSTANTS_EXTENT 3u
#define DRIVER_WAIT_NANOSECONDS 1000000000ull
#define TEST_RANKS TP2_RANKS
#define TEST_PAIR_B_CARD 2u
#define TEST_PAIR_B_FOLLOWER 3u
#define TEST_PAIR_A_LAYERS 14u
#define TEST_PREFIX_LAYERS 29u
#define TEST_PAIR_A_LAST 13u
#define TEST_PAIR_B_LAST 28u
#define TEST_FRONT_LAYER 20u
#define TEST_MAX_ROWS LINEAR_MAXIMUM_ROWS
#define TEST_SLICES 4u
#define TEST_BAD_SLICES 3u
#define TEST_DIMENSIONS 3u
#define TEST_PACK_FIXED_BLOCKS 5u
#define TEST_PACK_METADATA_BLOCKS 2u
#define TEST_TABLE_1_EXTENT 2u
#define TEST_ODD_MULTIPLIER_STRIDE 2
#define TEST_ROTATIONS 2u
#define TEST_STREAM_COPIES 8u
#define TEST_STATE_COPIES 4u
#define TEST_SESSION_KINDS 4u
#define TEST_COMPRESSOR_KIND 1u
#define TEST_POOL_KIND 2u
#define TEST_KEY_KIND 3u
#define TEST_COMPRESSOR_RANGES 6u
#define TEST_HISTORY_RANGES 9u
#define TEST_UNCHANGED_PAIRS 99u
#define TEST_PROTOTYPE_PLANS 3u
#define TEST_MIXED_MEMORY_BYTES (1ull << 40u)
#define TEST_MIXED_PAIR_ROWS 2u
#define TEST_MIXED_MEDIUM_ROWS 256u
#define TEST_ORIGINAL_WIDE_ROWS 128u
#define TEST_ORIGINAL_WIDE_VARIANTS 2u
#define TEST_ORIGINAL_CORE_ROWS 128u
#define TEST_ORIGINAL_CORE_VARIANTS 2u
#define TEST_ORIGINAL_CORE_KEYS 640u
#define TEST_ORIGINAL_CORE_PICKS 512u
#define TEST_ORIGINAL_CORE_RING_SLOTS 4096u
/* AVX-512 fixed-size copies can use aligned 64-byte stores on the ASAN fake stack. */
#define TEST_ORIGINAL_BATCH_RECORD_ALIGNMENT 64u
#define TEST_MIXED_OMITTED_TAIL_SEGMENTS 2u
#define EXPECTED_PAIR_WIDE_GROUPS 23u
#define EXPECTED_QUAD_SEGMENTS 140u
#define EXPECTED_QUAD_FRONT_SEGMENTS 89u
#define EXPECTED_QUAD_GROUPS 31u
#define EXPECTED_QUAD_WIDE_GROUPS 22u
#define EXPECTED_HYBRID_SEGMENTS 64u
#define EXPECTED_HYBRID_FRONT_SEGMENTS 39u
#define EXPECTED_HYBRID_GROUPS 15u
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

struct fixture {
	struct v41_tp2_stage_plan *stages[CARDS];
	struct model_plan_record *canonical;
	struct model_plan_record *views[CARDS];
	struct device_card *cards[CARDS];
	struct card_description_record card_descriptions[CARDS];
	struct v41_tp2_program_description description;
	struct engram_record *engram;
	unsigned char *engram_bytes;
};

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: full TP2 program: %s\n", message);
		failures++;
	}
}

static uint32_t owner_of(uint32_t layer)
{
	return layer < TEST_PAIR_A_LAYERS ? 0u :
	       layer < TEST_PREFIX_LAYERS ? TEST_PAIR_B_CARD :
					    PRO_CARD;
}

/* Real parser and model-sized Engram geometry, with tiny synthetic row contents. Only driver admission uses it;
 * no step, model kernel or gather runs in this fixture. Public APIs own the object, with no opaque-layout mock. */
static bool driver_engram_create(struct fixture *fixture)
{
	enum {
		MULTIPLIERS,
		OFFSETS,
		PAD_ID,
		PRIMES,
		ROW_COUNTS,
		TABLE_UNITS,
		TOKEN_MAP,
		TABLE_0,
		TABLE_1
	};
	const struct {
		const char *name;
		uint32_t type, dimensions[TEST_DIMENSIONS];
	} definitions[ENGRAM_TENSORS] = {
		{ "engram.multipliers",
		  PACK_ELEMENT_INT64,
		  { ENGRAM_TABLES, ENGRAM_LOOKBACK, 0 } },
		{ "engram.offsets",
		  PACK_ELEMENT_INT64,
		  { ENGRAM_TABLES, ENGRAM_ROWS, 0 } },
		{ "engram.pad_id", PACK_ELEMENT_INT32, { 1u, 0, 0 } },
		{ "engram.primes",
		  PACK_ELEMENT_INT64,
		  { ENGRAM_TABLES, ENGRAM_LOOKBACK - 1u, ENGRAM_HEADS } },
		{ "engram.row_counts",
		  PACK_ELEMENT_INT64,
		  { ENGRAM_TABLES, 0, 0 } },
		{ "engram.table_units",
		  PACK_ELEMENT_INT32,
		  { ENGRAM_TABLES, 0, 0 } },
		{ "engram.token_map",
		  PACK_ELEMENT_INT32,
		  { MODEL_VOCABULARY_VALUE, 0, 0 } },
		{ "table0.rows",
		  PACK_ELEMENT_UINT8,
		  { 1u, PACK_BLOCK_BYTES, 0 } },
		{ "table1.rows",
		  PACK_ELEMENT_UINT8,
		  { 1u, PACK_BLOCK_BYTES, 0 } }
	};
	uint64_t offsets[ENGRAM_TENSORS] = { 0 }, sizes[ENGRAM_TENSORS] = { 0 },
		 constants_bytes = 0;
	for (uint32_t tensor = 0; tensor < ENGRAM_TENSORS; tensor++) {
		if (tensor != 0 && strcmp(definitions[tensor - 1u].name,
					  definitions[tensor].name) >= 0)
			return false;
		sizes[tensor] = definitions[tensor].type == PACK_ELEMENT_INT32 ?
					sizeof(int32_t) :
				definitions[tensor].type == PACK_ELEMENT_INT64 ?
					sizeof(int64_t) :
					sizeof(uint8_t);
		for (uint32_t dimension = 0;
		     dimension < COUNT_OF(definitions[tensor].dimensions);
		     dimension++)
			sizes[tensor] *=
				definitions[tensor].dimensions[dimension] != 0 ?
					definitions[tensor]
						.dimensions[dimension] :
					1u;
		if (tensor < TABLE_0) {
			offsets[tensor] = constants_bytes;
			constants_bytes =
				(constants_bytes + sizes[tensor] +
				 WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES - 1u) /
				WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES *
				WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES;
		}
		if (offsets[tensor] % WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES != 0)
			return false;
	}
	constants_bytes = (constants_bytes + PACK_BLOCK_BYTES - 1u) /
			  PACK_BLOCK_BYTES * PACK_BLOCK_BYTES;
	const uint64_t byte_count =
		TEST_PACK_FIXED_BLOCKS * (uint64_t)PACK_BLOCK_BYTES +
		constants_bytes;
	unsigned char *const bytes =
		aligned_alloc(PACK_BLOCK_BYTES, (size_t)byte_count);
	if (bytes == NULL)
		return false;
	memset(bytes, 0, (size_t)byte_count);
	struct pack_header *const header = (void *)bytes;
	struct pack_extent *const extents = (void *)(bytes + PACK_BLOCK_BYTES);
	struct pack_unit_record *const units =
		(void *)(extents + ENGRAM_EXTENTS);
	struct pack_tensor *const tensors = (void *)(units + ENGRAM_UNITS);
	char *const name_bytes = (void *)(tensors + ENGRAM_TENSORS);
	*header =
		(struct pack_header){ .magic = PACK_MAGIC,
				      .format_version = PACK_FORMAT_VERSION,
				      .block_byte_count = PACK_BLOCK_BYTES,
				      .model_name =
					      "full-tp2-driver-engram-fixture",
				      .file_byte_count = byte_count,
				      .tables_byte_count = PACK_BLOCK_BYTES,
				      .extent_count = ENGRAM_EXTENTS,
				      .unit_count = ENGRAM_UNITS,
				      .tensor_count = ENGRAM_TENSORS };
	uint64_t next = TEST_PACK_METADATA_BLOCKS * (uint64_t)PACK_BLOCK_BYTES;
	for (uint32_t extent = 0; extent < ENGRAM_EXTENTS; extent++) {
		extents[extent] = (struct pack_extent){
			.byte_offset = next,
			.unit_index = extent,
			.byte_count = extent == ENGRAM_CONSTANTS_EXTENT ?
					      constants_bytes :
					      PACK_BLOCK_BYTES
		};
		units[extent] = (struct pack_unit_record){
			.unit_kind = extent == 0 ? PLACEMENT_UNIT_STAGE :
						   PLACEMENT_UNIT_HOST_ONLY,
			.layer_index = extent == 0 ? 0u : PACK_NO_LAYER
		};
		next += extents[extent].byte_count;
	}
	for (uint32_t tensor = 0; tensor < ENGRAM_TENSORS; tensor++) {
		const uint32_t name_length =
			(uint32_t)strlen(definitions[tensor].name);
		tensors[tensor] = (struct pack_tensor){
			.byte_offset_in_extent = offsets[tensor],
			.byte_count = sizes[tensor],
			.extent_index = tensor < TABLE_0 ?
						ENGRAM_CONSTANTS_EXTENT :
					tensor == TABLE_0 ? 1u :
							    TEST_TABLE_1_EXTENT,
			.name_byte_offset = header->name_byte_count,
			.name_byte_count = name_length,
			.element_type = definitions[tensor].type
		};
		for (uint32_t dimension = 0;
		     dimension < COUNT_OF(definitions[tensor].dimensions);
		     dimension++) {
			tensors[tensor].dimensions[dimension] =
				definitions[tensor].dimensions[dimension];
			tensors[tensor].dimension_count +=
				definitions[tensor].dimensions[dimension] != 0 ?
					1u :
					0u;
		}
		memcpy(name_bytes + header->name_byte_count,
		       definitions[tensor].name, name_length);
		header->name_byte_count += name_length;
	}
	unsigned char *const constants =
		bytes + extents[ENGRAM_CONSTANTS_EXTENT].byte_offset;
	for (uint32_t token = 0; token < MODEL_VOCABULARY_VALUE; token++) {
		const int32_t mapped = (int32_t)(token % ENGRAM_PRIME);
		memcpy(constants + offsets[TOKEN_MAP] + token * sizeof(mapped),
		       &mapped, sizeof(mapped));
	}
	for (uint32_t table = 0; table < ENGRAM_TABLES; table++) {
		const int32_t unit = (int32_t)table + 1;
		const int64_t prime = ENGRAM_PRIME;
		memcpy(constants + offsets[TABLE_UNITS] + table * sizeof(unit),
		       &unit, sizeof(unit));
		memcpy(constants + offsets[ROW_COUNTS] + table * sizeof(prime),
		       &prime, sizeof(prime));
		for (uint32_t distance = 0; distance < ENGRAM_LOOKBACK;
		     distance++) {
			const int64_t multiplier =
				(int64_t)distance * TEST_ODD_MULTIPLIER_STRIDE +
				1;
			memcpy(constants + offsets[MULTIPLIERS] +
				       (table * ENGRAM_LOOKBACK + distance) *
					       sizeof(multiplier),
			       &multiplier, sizeof(multiplier));
		}
		for (uint32_t row = 0; row < ENGRAM_ROWS; row++)
			memcpy(constants + offsets[PRIMES] +
				       (table * ENGRAM_ROWS + row) *
					       sizeof(prime),
			       &prime, sizeof(prime));
	}
	result_status status = STATUS_OK;
	for (uint32_t extent = 0;
	     extent < ENGRAM_EXTENTS && status == STATUS_OK; extent++)
		status = pack_digest_of_host_bytes(
			bytes + extents[extent].byte_offset,
			extents[extent].byte_count, &extents[extent].digest);
	if (status == STATUS_OK)
		test_pack_reseal(bytes);
	test_memory_storage_reset();
	test_memory_storage_set_pack(bytes, byte_count);
	const char *const path = "driver-engram-fixture";
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	status = status == STATUS_OK ?
			 test_memory_storage.reader_open(&path, 1, &reader) :
			 status;
	status = status == STATUS_OK ?
			 pack_open_entry(&test_memory_storage, reader, &pack) :
			 status;
	status = status == STATUS_OK ? engram_open(&test_memory_storage, reader,
						   pack, &fixture->engram) :
				       status;
	for (uint32_t table = 0; table < ENGRAM_TABLES && status == STATUS_OK;
	     table++)
		status = engram_attach_entry(
			fixture->engram, table,
			bytes + extents[table + 1u].byte_offset);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	if (status != STATUS_OK) {
		engram_close(fixture->engram);
		fixture->engram = NULL;
		test_memory_storage_reset();
		free(bytes);
		return false;
	}
	fixture->engram_bytes = bytes;
	const struct engram_description *const description =
		engram_description_of(fixture->engram);
	return description->table_count == ENGRAM_TABLES &&
	       description->rows_per_token == ENGRAM_ROWS &&
	       description->vocabulary_size == MODEL_VOCABULARY_VALUE;
}

static bool fixture_create(const struct pack_record *pack,
			   struct fixture *fixture)
{
	const struct model_limits limits = { CONTEXT_TOKENS,
					     TEST_MAX_ROWS,
					     1u,
					     STATE_WIDTH_16,
					     LINEAR_PROFILE_PRECISION,
					     LINEAR_ORDER_FIXED,
					     CMP_ORDER_REFERENCE,
					     STEP_DRAFT_MOST,
					     0,
					     0 };
	static const uint32_t first[CARDS] = { 0u, 0u, 14u, 14u, 29u };
	static const uint32_t count[CARDS] = { 14u, 14u, 15u, 15u, 11u };
	for (uint32_t card = 0; card < CARDS; card++) {
		struct v41_tp2_stage_description stage = {
			.limits = limits,
			.card = card,
			.rank = card == PRO_CARD ? 0u : card % TEST_RANKS,
			.rank_count = card == PRO_CARD ? 1u : TEST_RANKS,
			.first_target_layer = first[card],
			.target_layer_count = count[card],
			.include_embedding = card == 0,
			.include_head_and_draft = card == PRO_CARD
		};
		if (card != PRO_CARD)
			stage.limits.draft_positions = 0;
		const result_status status = v41_tp2_stage_plan_create(
			pack, &stage, &fixture->stages[card]);
		expect(status == STATUS_OK,
		       "five cold stage inventories prepare");
		if (status != STATUS_OK)
			return false;
	}
	fixture->canonical = calloc(1, sizeof(*fixture->canonical));
	if (fixture->canonical == NULL)
		return false;
	struct model_plan_record *const canonical = fixture->canonical;
	*canonical = *fixture->stages[0]->model;
	canonical->limits = limits;
	canonical->card_count = CARDS;
	canonical->embedding_card = 0;
	canonical->result_card = PRO_CARD;
	for (uint32_t card = 0; card < CARDS; card++)
		canonical->cards[card] =
			fixture->stages[card]->model->cards[card];
	canonical->globals[V41_DATA_NORM] =
		fixture->stages[PRO_CARD]->model->globals[V41_DATA_NORM];
	canonical->globals[V41_DATA_HEAD] =
		fixture->stages[PRO_CARD]->model->globals[V41_DATA_HEAD];
	memcpy(canonical->draft_globals,
	       fixture->stages[PRO_CARD]->model->draft_globals,
	       sizeof(canonical->draft_globals));
	for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
		const uint32_t owner = owner_of(layer);
		const struct model_plan_record *const source =
			fixture->stages[owner]->model;
		canonical->layer_card[layer] = owner;
		memcpy(canonical->layer_tensors[layer],
		       source->layer_tensors[layer],
		       sizeof(canonical->layer_tensors[layer]));
		canonical->experts[layer] = source->experts[layer];
		memcpy(canonical->expert_tables[layer],
		       source->expert_tables[layer],
		       sizeof(canonical->expert_tables[layer]));
	}
	for (uint32_t card = 0; card < CARDS; card++) {
		fixture->views[card] = malloc(sizeof(*fixture->views[card]));
		if (fixture->views[card] == NULL)
			return false;
		struct model_plan_record *const view = fixture->views[card];
		*view = *fixture->stages[card]->model;
		view->limits = limits;
		view->card_count = CARDS;
		view->embedding_card = canonical->embedding_card;
		view->result_card = canonical->result_card;
		memcpy(view->cards, canonical->cards, sizeof(view->cards));
		memcpy(view->globals, canonical->globals,
		       sizeof(view->globals));
		memcpy(view->draft_globals, canonical->draft_globals,
		       sizeof(view->draft_globals));
		memcpy(view->layer_card, canonical->layer_card,
		       sizeof(view->layer_card));
		memcpy(view->pool_session_bytes, canonical->pool_session_bytes,
		       sizeof(view->pool_session_bytes));
		memcpy(view->key_session_bytes, canonical->key_session_bytes,
		       sizeof(view->key_session_bytes));
		fixture->description.views[card] = view;
		fixture->description.stages[card] = fixture->stages[card];
	}
	fixture->description.canonical = canonical;
	fixture->description.slice_count = TEST_SLICES;
	test_host_memory_backend_reset(CARDS, HOST_BYTES);
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, card == PRO_CARD ? ARITHMETIC_CLASS_SM_120 :
						 ARITHMETIC_CLASS_SM_80);
		result_status status = test_host_memory_backend.card_describe(
			card, &fixture->card_descriptions[card]);
		status = status == STATUS_OK ?
				 test_host_memory_backend.card_open(
					 card, HOST_BYTES,
					 &fixture->cards[card]) :
				 status;
		expect(status == STATUS_OK, "five host-only cards open");
		if (status != STATUS_OK)
			return false;
	}
	fixture->card_descriptions[PRO_CARD].arithmetic_class =
		ARITHMETIC_CLASS_SM_120_NATIVE;
	return true;
}

static void fixture_destroy(struct fixture *fixture)
{
	engram_close(fixture->engram);
	test_memory_storage_reset();
	free(fixture->engram_bytes);
	for (uint32_t card = 0; card < CARDS; card++) {
		if (fixture->cards[card] != NULL)
			test_host_memory_backend.card_close(
				fixture->cards[card], 0);
		/* These shallow merged copies borrow the stage images; only the stages destroy those images. */
		free(fixture->views[card]);
		v41_tp2_stage_plan_destroy(fixture->stages[card]);
	}
	free(fixture->canonical);
}

static void check_program(const struct fixture *fixture,
			  const struct execution_program *program,
			  const struct step_companion_program *companions,
			  uint32_t rows)
{
	expect(program->kind == PROGRAM_THIN && program->card_count == CARDS &&
		       program->row_count == rows &&
		       program->result_card == PRO_CARD &&
		       program->expert_pause_count == 0 &&
		       program->host_expert_count == 0 &&
		       program->lease_count == 0,
	       "single-flight THIN metadata is complete");
	expect(program->segment_count == EXPECTED_SEGMENTS &&
		       program->front_segment_count ==
			       EXPECTED_FRONT_SEGMENTS &&
		       program->draft_segment_first == EXPECTED_DRAFT_FIRST,
	       "embedding/front/target/head/draft boundaries agree");
	expect(companions->program == program &&
		       companions->pair_count == EXPECTED_PAIRS &&
		       companions->reserved_zero == 0 &&
		       companions->arithmetic_identity != 0,
	       "owned companion map has a recipe");
	uint32_t layer_pairs[TEST_PREFIX_LAYERS] = { 0 }, rotations = 0;
	for (uint32_t index = 0; index < companions->pair_count; index++) {
		const struct step_companion_pair *const pair =
			&companions->pairs[index];
		expect(pair->second_segment == pair->first_segment + 1u &&
			       pair->second_segment < program->segment_count,
		       "companions are adjacent valid segments");
		if (pair->second_segment >= program->segment_count)
			continue;
		const struct segment_record *const first =
			&program->segments[pair->first_segment];
		const struct segment_record *const second =
			&program->segments[pair->second_segment];
		expect(first->card < PRO_CARD &&
			       first->card % TEST_RANKS == 0 &&
			       second->card == first->card + 1u &&
			       first->stage == second->stage &&
			       first->layer == second->layer &&
			       first->pause == 0 && second->pause == 0,
		       "each pair is the same layer/stage on its two CMP ranks");
		expect(program->front_segment_count != pair->second_segment &&
			       program->draft_segment_first !=
				       pair->second_segment,
		       "front and draft boundaries never bisect a pair");
		if (first->layer == SEGMENT_NO_LAYER)
			rotations++;
		else if (first->layer < COUNT_OF(layer_pairs)) {
			layer_pairs[first->layer]++;
			expect(first->card == owner_of(first->layer),
			       "TP target layer belongs to its expected pair");
		} else
			expect(false, "no PRO or draft layer is paired");
	}
	expect(rotations == TEST_ROTATIONS, "two paired rotation segments");
	for (uint32_t layer = 0; layer < COUNT_OF(layer_pairs); layer++)
		expect(layer_pairs[layer] ==
			       (layer == TEST_FRONT_LAYER ? TEST_RANKS : 1u),
		       "every TP target has one pair, layer20 has FRONT/REST");
	expect(program->segments[program->front_segment_count - 1u].layer ==
			       TEST_FRONT_LAYER &&
		       program->segments[program->front_segment_count - 1u]
				       .card == TEST_PAIR_B_FOLLOWER &&
		       program->segments[program->front_segment_count].layer ==
			       TEST_FRONT_LAYER &&
		       program->segments[program->front_segment_count].card ==
			       TEST_PAIR_B_CARD,
	       "layer20 FRONT pair ends before REST pair");
	uint32_t collective_count[CARDS] = { 0 };
	for (uint32_t segment = 0; segment < program->segment_count;
	     segment++) {
		const struct segment_record *const part =
			&program->segments[segment];
		uint32_t sites = 0;
		for (uint32_t index = 0; index < part->launch_count; index++) {
			const struct kernel_launch *const launch =
				&program->cards[part->card]
					 .launches[part->first_launch + index];
			struct test_host_memory_backend_kernel kernel;
			if (!test_host_memory_backend_kernel_of(
				    part->card, launch->kernel, &kernel)) {
				expect(false,
				       "all launch handles name prepared kernels");
				continue;
			}
			if (strcmp(kernel.function_name, KERNEL_ORDERED_PUSH) ==
			    0) {
				struct ordered_push_arguments arguments;
				memcpy(&arguments, launch->argument,
				       sizeof(arguments));
				expect(part->card < PRO_CARD &&
					       part->layer <
						       TEST_PREFIX_LAYERS &&
					       sites < TP2_SITES,
				       "only TP target REST segments contain ordered all-reduces");
				if (part->card < PRO_CARD &&
				    part->layer < TEST_PREFIX_LAYERS &&
				    sites < TP2_SITES) {
					const struct v41_tp2_stage_plan *const
						stage = fixture->stages
								[part->card];
					const uint32_t channel =
						(part->layer -
						 stage->description
							 .first_target_layer) *
							TP2_SITES +
						sites;
					const struct ordered_push_arguments *const
						prepared =
							&stage->channels[channel]
								 .collective
								 .arguments;
					expect(arguments.rank ==
							       part->card %
								       TEST_RANKS &&
						       arguments.rank_count ==
							       TEST_RANKS &&
						       arguments.element_count ==
							       rows * MODEL_HIDDEN &&
						       arguments.region_offset ==
							       prepared->region_offset &&
						       arguments.peer_table_offset ==
							       prepared->peer_table_offset &&
						       arguments.step_offset ==
							       prepared->step_offset,
					       "each TP site uses its own rank/layer/channel and actual program rows");
				}
				collective_count[part->card]++;
				sites++;
			}
		}
		const bool target_rest =
			part->card < PRO_CARD &&
			part->layer < TEST_PREFIX_LAYERS &&
			!(part->layer == TEST_FRONT_LAYER &&
			  segment < program->front_segment_count);
		expect(sites == (target_rest ? TP2_SITES : 0u),
		       "two collectives per TP layer, none in FRONT/rotation/PRO");
	}
	static const uint32_t expected_collectives[CARDS] = { 28u, 28u, 30u,
							      30u, 0u };
	expect(memcmp(collective_count, expected_collectives,
		      sizeof(expected_collectives)) == 0,
	       "116 ordered collective launches cover both sites of all 29 paired target layers");
	expect(program->hand_off_count == EXPECTED_HAND_OFFS,
	       "twelve required stream/state hand-offs, no replica copies");
	uint32_t streams = 0, state = 0;
	for (uint32_t index = 0; index < program->hand_off_count; index++) {
		const struct hand_off_record *const hand_off =
			&program->hand_offs[index];
		expect(hand_off->after_segment < program->segment_count,
		       "hand-off names a valid producer segment");
		if (hand_off->after_segment >= program->segment_count)
			continue;
		const struct segment_record *const producer =
			&program->segments[hand_off->after_segment];
		expect(producer->card == hand_off->from_card &&
			       hand_off->from_card < hand_off->to_card &&
			       hand_off->from_card % TEST_RANKS == 0,
		       "canonical first rank emits the hand-off at its actual segment");
		const uint64_t *const arrays =
			fixture->canonical->cards[hand_off->from_card]
				.array_offset;
		const bool is_stream =
			hand_off->source_offset ==
				arrays[V41_DATA_ARRAY_STREAMS] ||
			hand_off->source_offset ==
				arrays[V41_DATA_ARRAY_CARRIED_PRE];
		if (is_stream) {
			streams++;
			expect((producer->layer == SEGMENT_NO_LAYER &&
				hand_off->from_card == 0u &&
				hand_off->to_card == 1u) ||
				       (producer->layer == TEST_PAIR_A_LAST &&
					(hand_off->to_card == TEST_PAIR_B_CARD ||
					 hand_off->to_card ==
						 TEST_PAIR_B_FOLLOWER)) ||
				       (producer->layer == TEST_PAIR_B_LAST &&
					hand_off->to_card == PRO_CARD),
			       "stream broadcast follows embedding or stage cut");
		} else {
			state++;
			expect(hand_off->from_card == TEST_PAIR_B_CARD &&
				       hand_off->to_card == PRO_CARD &&
				       (producer->layer == TEST_FRONT_LAYER ||
					producer->layer == TEST_PAIR_B_LAST),
			       "external state copies only from pairB to PRO");
			expect((fixture->canonical->cards[hand_off->to_card]
						.layers >>
					producer->layer &
				1u) == 0,
			       "locally computed source/index replicas receive no hand-off");
		}
		for (uint32_t pair = 0; pair < companions->pair_count; pair++) {
			const struct step_companion_pair *const linked =
				&companions->pairs[pair];
			if (linked->first_segment == hand_off->after_segment)
				expect(hand_off->to_card !=
					       program->segments
						       [linked->second_segment]
							       .card,
				       "first rank never hands off inside its own companion pair");
		}
	}
	expect(streams == TEST_STREAM_COPIES && state == TEST_STATE_COPIES,
	       "eight stream and four external source/index copies");
	expect(program->checkpoint_count == EXPECTED_CHECKPOINTS,
	       "all 58 TP and 14 PRO/draft checkpoints are present");
	static const uint32_t checkpoint_count[CARDS] = { 14u, 14u, 15u, 15u,
							  14u };
	uint32_t found_checkpoints[CARDS] = { 0 };
	for (uint32_t index = 0; index < program->checkpoint_count; index++) {
		const struct window_checkpoint *const checkpoint =
			&program->checkpoints[index];
		expect(checkpoint->card < CARDS, "checkpoint card is valid");
		if (checkpoint->card < CARDS)
			found_checkpoints[checkpoint->card]++;
		expect(checkpoint->ring_session_stride ==
				       V41_DATA_RING_SESSION_BYTES &&
			       checkpoint->row_bytes == MODEL_STATE_ROW_BYTES &&
			       checkpoint->window_rows == MODEL_WINDOW - 1u,
		       "checkpoint geometry preserves PP state layout");
	}
	expect(memcmp(found_checkpoints, checkpoint_count,
		      sizeof(checkpoint_count)) == 0,
	       "both ranks of both pairs own checkpoints");
	expect(program->session_range_count == EXPECTED_SESSION_RANGES,
	       "all replicated snapshot ranges are present");
	uint32_t kinds[TEST_SESSION_KINDS] = { 0 };
	for (uint32_t index = 0; index < program->session_range_count;
	     index++) {
		const struct session_range *const range =
			&program->session_ranges[index];
		kinds[range->identity % COUNT_OF(kinds)]++;
		if (index != 0)
			expect(range->identity >=
				       program->session_ranges[index - 1u]
					       .identity,
			       "snapshot identities are sorted");
	}
	expect(kinds[0] == EXPECTED_CHECKPOINTS &&
		       kinds[TEST_COMPRESSOR_KIND] == TEST_COMPRESSOR_RANGES &&
		       kinds[TEST_POOL_KIND] == TEST_HISTORY_RANGES &&
		       kinds[TEST_KEY_KIND] == TEST_HISTORY_RANGES,
	       "rings, compressors, pools and keys include every needed rank and external replica");
	expect(test_host_memory_backend_work_count() == 0,
	       "factory queues no launch/copy/digest work");
	(void)printf("rows=%" PRIu32 " segments=%" PRIu32 " pairs=%" PRIu32
		     " front=%" PRIu32 " draft=%" PRIu32 " checkpoints=%" PRIu32
		     " ranges=%" PRIu32 " handoffs=%" PRIu32 "\n",
		     rows, program->segment_count, companions->pair_count,
		     program->front_segment_count, program->draft_segment_first,
		     program->checkpoint_count, program->session_range_count,
		     program->hand_off_count);
}

static void expect_refusal(struct fixture *fixture, uint32_t rows,
			   const char *message)
{
	struct execution_program sentinel = { 0 }, *program = &sentinel;
	const struct step_companion_program untouched = {
		.program = &sentinel, .pair_count = TEST_UNCHANGED_PAIRS
	};
	struct step_companion_program companions = untouched;
	const result_status status = v41_tp2_program_prepare(
		&fixture->description, &test_host_memory_backend,
		fixture->cards, fixture->card_descriptions, rows, &program,
		&companions);
	expect(status == STATUS_INVALID_ARGUMENT, message);
	expect(program == &sentinel &&
		       companions.program == untouched.program &&
		       companions.pair_count == untouched.pair_count &&
		       companions.pairs == untouched.pairs &&
		       companions.arithmetic_identity ==
			       untouched.arithmetic_identity,
	       "refusal leaves both outputs untouched");
	if (status == STATUS_OK)
		free(program);
}

static void check_refusals(struct fixture *fixture)
{
	fixture->stages[0]->channels[0].collective.arguments.payload_format =
		ORDERED_PUSH_BF16;
	expect_refusal(
		fixture, 1u,
		"a codec changed after immutable stage planning is refused");
	fixture->stages[0]->channels[0].collective.arguments.payload_format =
		ORDERED_PUSH_FP32;
	expect_refusal(fixture, 0, "zero rows refused");
	expect_refusal(fixture, TEST_MAX_ROWS + 1u,
		       "rows above thin bound refused");
	fixture->description.slice_count = TEST_BAD_SLICES;
	expect_refusal(fixture, 1u, "non-power-of-two slice recipe refused");
	fixture->description.slice_count = TEST_SLICES;
	fixture->views[1]->globals[V41_DATA_EMBEDDING].card = 1u;
	expect_refusal(fixture, 1u, "a stale global tensor reference refused");
	fixture->views[1]->globals[V41_DATA_EMBEDDING] =
		fixture->canonical->globals[V41_DATA_EMBEDDING];
	fixture->canonical->host_expert_count[0] = 1u;
	expect_refusal(fixture, 1u, "host expert pause is unsupported");
	fixture->canonical->host_expert_count[0] = 0;
	fixture->stages[1]->description.rank = 0;
	expect_refusal(fixture, 1u, "wrong physical rank refused");
	fixture->stages[1]->description.rank = 1u;
	fixture->stages[0]->channels[0].collective.arguments.rank_count = 1u;
	expect_refusal(fixture, 1u, "wrong channel rank count refused");
	fixture->stages[0]->channels[0].collective.arguments.rank_count =
		TEST_RANKS;
	fixture->card_descriptions[0].arithmetic_class =
		ARITHMETIC_CLASS_SM_120;
	expect_refusal(fixture, 1u, "non-CMP pair member refused");
	fixture->card_descriptions[0].arithmetic_class = ARITHMETIC_CLASS_SM_80;
	const uint32_t saved_context =
		fixture->canonical->limits.context_token_limit;
	fixture->canonical->limits.context_token_limit =
		CONTEXT_TOKENS - MODEL_CONTEXT_GRANULE;
	expect_refusal(
		fixture, 1u,
		"full TP factory refuses the inherited small-context score limitation");
	fixture->canonical->limits.context_token_limit = saved_context;
}

static void check_layer_prototype(const struct pack_record *pack,
				  const struct fixture *fixture)
{
	struct v41_tp2_layer_plan *plans[TEST_PROTOTYPE_PLANS] = { 0 };
	struct model_limits limits = fixture->canonical->limits;
	limits.draft_positions = 0;
	struct v41_tp2_layer_program *const program =
		calloc(1, sizeof(*program));
	if (program == NULL) {
		expect(false, "layer prototype program allocation succeeds");
		return;
	}
	const uint32_t cards[TEST_PROTOTYPE_PLANS] = { 0, 0, 1 };
	const uint32_t ranks[TEST_PROTOTYPE_PLANS] = { 0, 0, 1 };
	const uint32_t counts[TEST_PROTOTYPE_PLANS] = { 1, TEST_RANKS,
							TEST_RANKS };
	for (uint32_t index = 0; index < TEST_PROTOTYPE_PLANS; index++) {
		const uint64_t prefix =
			index == 1u && plans[0] != NULL ?
				plans[0]->model->cards[0].arena_byte_count :
				0;
		result_status status = v41_tp2_layer_plan_create(
			pack, &limits, cards[index], ranks[index],
			counts[index], prefix, &plans[index]);
		expect(status == STATUS_OK,
		       "isolated PP/TP layer plans prepare with the real probe prefixes");
		static const uint32_t rows[] = { 1u, 6u };
		for (uint32_t shape = 0;
		     status == STATUS_OK && shape < COUNT_OF(rows); shape++) {
			status = v41_tp2_layer_program_prepare(
				plans[index], &test_host_memory_backend,
				fixture->cards, fixture->card_descriptions,
				rows[shape], TP2_DEFAULT_SLICES, false,
				program);
			expect(status == STATUS_OK &&
				       program->launch_count != 0,
			       "complete isolated PP/TP layer preparation accepts real router/expert ranges");
		}
	}
	for (uint32_t rank = 0; plans[0] != NULL && rank < TEST_RANKS; rank++) {
		struct v41_tp2_layer_plan *replicated = NULL;
		result_status status = v41_tp2_layer_variant_create(
			pack, &limits, rank, rank, TEST_RANKS, 0, true,
			&replicated);
		expect(status == STATUS_OK && replicated != NULL,
		       "full attention plus TP2 experts has an explicit cold plan");
		if (replicated != NULL) {
			const struct v41_tensor *const selected =
				replicated->model->layer_tensors[0];
			const struct v41_tensor *const original =
				plans[0]->model->layer_tensors[0];
			expect(selected[V41_DATA_QUERY_UP].shape.output_size ==
					       original[V41_DATA_QUERY_UP]
						       .shape.output_size &&
				       selected[V41_DATA_OUTPUT]
						       .shape.input_size ==
					       original[V41_DATA_OUTPUT]
						       .shape.input_size &&
				       selected[V41_DATA_SHARED_GATE]
							       .shape
							       .output_size *
						       TEST_RANKS ==
					       original[V41_DATA_SHARED_GATE]
						       .shape.output_size,
			       "attention stays full while shared experts retain their independent TP2 shard");
			for (uint32_t parallel = 0;
			     parallel <= 1u && status == STATUS_OK;
			     parallel++) {
				replicated->parallel_publish = parallel != 0;
				status = v41_tp2_layer_program_prepare(
					replicated, &test_host_memory_backend,
					fixture->cards,
					fixture->card_descriptions,
					TEST_MAX_ROWS, TP2_DEFAULT_SLICES,
					false, program);
				expect(status == STATUS_OK,
				       "one-reduction and parallel-publication variants prepare");
				uint32_t reductions = 0, publications = 0;
				for (uint32_t launch = 0;
				     status == STATUS_OK &&
				     launch < program->launch_count;
				     launch++) {
					struct test_host_memory_backend_kernel
						kernel;
					if (program->launches[launch].kernel ==
					    BACKEND_COPY)
						continue;
					const bool found =
						test_host_memory_backend_kernel_of(
							rank,
							program->launches[launch]
								.kernel,
							&kernel);
					expect(found,
					       "every component launch resolves to its prepared function");
					if (found) {
						reductions +=
							strcmp(kernel.function_name,
							       parallel != 0 ?
								       KERNEL_ORDERED_PUSH_PARALLEL_REDUCE :
								       KERNEL_ORDERED_PUSH) ==
									0 ?
								1u :
								0u;
						publications +=
							strcmp(kernel.function_name,
							       KERNEL_ORDERED_PUSH_PARALLEL_PUBLISH) ==
									0 ?
								1u :
								0u;
					}
				}
				expect(reductions == 1u &&
					       publications == parallel,
				       "full attention legally removes exactly one rank reduction");
			}
		}
		v41_tp2_layer_plan_destroy(replicated);
	}
    for (uint32_t rank = 0; rank < TEST_RANKS; rank++) {
        for (uint32_t attention = 0; attention <= 1u; attention++) {
		struct v41_tp2_layer_plan *partitioned = NULL;
		result_status status = v41_tp2_layer_choices_create(
			pack, &limits, rank, rank, TEST_RANKS, 0,
			attention != 0, true, &partitioned);
		expect(status == STATUS_OK && partitioned != NULL,
		       "explicit EP2 plan prepares");
		if (partitioned == NULL)
			continue;
		const uint64_t *const dummy =
			partitioned->model->expert_tables[0][rank ^ 1u];
		uint32_t owned = 0;
		for (uint32_t expert = 0; expert < MODEL_EXPERTS; expert++) {
			const uint64_t *const offsets =
				partitioned->model->expert_tables[0][expert];
			if (expert % TEST_RANKS == rank) {
				owned++;
				expect(offsets[V41_DATA_GATE] !=
					       partitioned->unowned_gate_offset,
				       "owned experts cannot alias the unowned gate");
			} else {
				expect(memcmp(offsets, dummy,
					      sizeof(partitioned->model->expert_tables
							     [0][expert])) == 0,
				       "every unowned expert aliases the same resident dummy");
			}
		}
		expect(owned * TEST_RANKS == MODEL_EXPERTS &&
			       partitioned->experts_partitioned &&
			       partitioned->unowned_gate_offset ==
				       dummy[V41_DATA_GATE],
		       "ownership covers every expert once");
		status = v41_tp2_layer_program_prepare(
			partitioned, &test_host_memory_backend, fixture->cards,
			fixture->card_descriptions, TEST_MAX_ROWS,
			TP2_DEFAULT_SLICES, false, program);
		expect(status == STATUS_OK,
		       "EP2 full routed shapes coexist with the TP2 shared expert");
		uint32_t filters = 0, routed_products = 0;
		for (uint32_t launch = 0;
		     status == STATUS_OK && launch < program->launch_count;
		     launch++) {
			if (program->launches[launch].kernel == BACKEND_COPY)
				continue;
			struct test_host_memory_backend_kernel kernel;
			const bool found = test_host_memory_backend_kernel_of(
				rank, program->launches[launch].kernel,
				&kernel);
			expect(found, "EP2 launch resolves");
			if (!found)
				continue;
			if (strcmp(kernel.function_name, KERNEL_EP_FILTER) ==
			    0) {
				struct ep_filter_arguments record;
				memcpy(&record,
				       program->launches[launch].argument,
				       sizeof(record));
				expect(record.hidden_width ==
						       V41_DATA_EXPERT_INNER &&
					       record.output_width ==
						       MODEL_HIDDEN &&
					       record.pair_count ==
						       TEST_MAX_ROWS *
							       MODEL_EXPERTS_PER_ROW &&
					       record.unowned_gate_offset ==
						       partitioned
							       ->unowned_gate_offset,
				       "EP2 clears BF16 full-width pairs and filters exactly the dummy jobs");
				filters++;
			}
			if (strncmp(kernel.function_name, "expert_gate_up_",
				    sizeof("expert_gate_up_") - 1u) == 0) {
				uint32_t output_size = 0;
				memcpy(&output_size,
				       program->launches[launch].argument +
					       offsetof(
						       struct expert_gate_up_arguments,
						       output_size),
				       sizeof(output_size));
				expect(output_size == V41_DATA_EXPERT_INNER,
				       "EP2 routed products retain full2304 width");
				routed_products++;
			}
		}
	    expect(filters == 1u && routed_products == 1u,
		   "one filter precedes one full-width routed product");
	    v41_tp2_layer_plan_destroy(partitioned);
	}
    }
    for (uint32_t index = 0; index < TEST_PROTOTYPE_PLANS; index++)
	    v41_tp2_layer_plan_destroy(plans[index]);
    free(program);
}

/* Admit the exact mixed program family used by the wide model probe, including real parsed Engram metadata.
 * The host cards hold only tiny registries; no arena content, model load, launch or copy is used here. */
static void check_mixed_driver(const struct pack_record *pack, uint32_t layout,
			       uint32_t wire)
{
	const bool exact_shards = getenv("TP_TEST_EXACT") != NULL;
	if (exact_shards &&
	    (layout != V41_DATA_TP_LAYOUT_QUAD || wire != ORDERED_PUSH_FP32))
		return;
	static const struct {
		enum program_kind kind;
		uint32_t rows;
	} cases[] = { { PROGRAM_THIN, 1u },
		      { PROGRAM_THIN, TEST_MIXED_PAIR_ROWS },
		      { PROGRAM_THIN, LINEAR_MAXIMUM_ROWS },
		      { PROGRAM_WIDE, MODEL_WIDE_LEAST_ROWS },
		      { PROGRAM_WIDE, TEST_MIXED_MEDIUM_ROWS },
		      { PROGRAM_WIDE, LINEAR_WIDE_MOST_ROWS },
		      { PROGRAM_FULL, MODEL_WIDE_LEAST_ROWS },
		      { PROGRAM_FULL, TEST_MIXED_MEDIUM_ROWS },
		      { PROGRAM_FULL, LINEAR_WIDE_MOST_ROWS } };
	static const struct {
		uint32_t segments, front, groups, wide_groups;
	} expected[] = { { EXPECTED_SEGMENTS, EXPECTED_FRONT_SEGMENTS,
			   EXPECTED_PAIRS, EXPECTED_PAIR_WIDE_GROUPS },
			 { EXPECTED_QUAD_SEGMENTS, EXPECTED_QUAD_FRONT_SEGMENTS,
			   EXPECTED_QUAD_GROUPS, EXPECTED_QUAD_WIDE_GROUPS },
			 { EXPECTED_HYBRID_SEGMENTS,
			   EXPECTED_HYBRID_FRONT_SEGMENTS,
			   EXPECTED_HYBRID_GROUPS, EXPECTED_HYBRID_GROUPS } };
	struct fixture fixture = { 0 };
	struct v41_tp2_model *model = NULL;
	struct execution_program *owned[COUNT_OF(cases)] = { 0 };
	const struct execution_program *programs[COUNT_OF(cases)] = { 0 };
	struct step_companion_program groups[COUNT_OF(cases)] = { 0 };
	struct v41_tp2_model_description description = {
		.limits = { CONTEXT_TOKENS, LINEAR_WIDE_MOST_ROWS, 1u,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE,
			    STEP_DRAFT_MOST, 0, 0 },
		.tensor_layout = layout,
		.exact_output_shards = exact_shards,
		.original_code_wire = exact_shards &&
				      getenv("TP_TEST_RAW") == NULL,
		.wire_format = wire
	};
	test_host_memory_backend_reset(CARDS, HOST_BYTES);
	bool complete = true;
	for (uint32_t card = 0; card < CARDS; card++) {
		description.cards[card].arithmetic_class =
			card == PRO_CARD ? ARITHMETIC_CLASS_SM_120_NATIVE :
					   ARITHMETIC_CLASS_SM_80;
		description.cards[card].memory_total_bytes =
			TEST_MIXED_MEMORY_BYTES;
		description.cards[card].memory_free_bytes =
			TEST_MIXED_MEMORY_BYTES;
		test_host_memory_backend_set_arithmetic_class(
			card, card == PRO_CARD ? ARITHMETIC_CLASS_SM_120 :
						 ARITHMETIC_CLASS_SM_80);
		complete = test_host_memory_backend.card_open(
				   card, HOST_BYTES, &fixture.cards[card]) ==
				   STATUS_OK &&
			   complete;
	}
	result_status status =
		complete ? v41_tp2_model_create(pack, &description, &model) :
			   STATUS_OUT_OF_MEMORY;
	expect(status == STATUS_OK && model != NULL,
	       "mixed original-geometry layout prepares");
	complete = status == STATUS_OK && model != NULL &&
		   driver_engram_create(&fixture);
	struct v41_tp2_program_description build;
	if (complete)
		complete = v41_tp2_model_program_description(model, &build) ==
			   STATUS_OK;
	(void)printf(
		"mixed_driver layout=%u wire=%u rows=%u parsed_engram=%u\n",
		layout, wire, LINEAR_WIDE_MOST_ROWS,
		fixture.engram != NULL ? 1u : 0u);
	for (uint32_t index = 0; complete && index < COUNT_OF(cases); index++) {
		status = v41_tp_program_prepare(
			&build, &test_host_memory_backend, fixture.cards,
			description.cards, cases[index].kind, cases[index].rows,
			&owned[index], &groups[index]);
		expect(status == STATUS_OK && owned[index] != NULL,
		       "each mixed THIN/WIDE/FULL program prepares");
		complete = status == STATUS_OK && owned[index] != NULL;
		if (!complete)
			break;
		const struct execution_program *const program = owned[index];
		programs[index] = program;
		const bool thin = program->kind == PROGRAM_THIN;
		const bool wide = program->kind == PROGRAM_WIDE;
		const bool whole_graph = layout == V41_DATA_TP_LAYOUT_QUAD &&
					 (exact_shards || wire != 0);
		const uint32_t merged_prefix_savings =
			whole_graph ? (wide ? EXPECTED_QUAD_WIDE_GROUPS - 1u :
					      EXPECTED_QUAD_GROUPS -
						       TEST_MIXED_PAIR_ROWS) *
					      TP4_RANKS :
				      0;
		const uint32_t wanted_segments =
			thin ? expected[layout].segments :
			wide ? expected[layout].front :
			       expected[layout].segments -
					TEST_MIXED_OMITTED_TAIL_SEGMENTS;
		const uint32_t wanted_groups =
			wide ? expected[layout].wide_groups :
			       expected[layout].groups;
		expect(program->segment_count == wanted_segments - merged_prefix_savings &&
			       groups[index].pair_count ==
                           (whole_graph ? (wide ? 1u : TEST_MIXED_PAIR_ROWS) : wanted_groups),
		       "mixed kind retains its independently counted segment/group boundary");
		uint32_t launches[CARDS] = { 0 }, hand_offs = 0;
		for (uint32_t segment = 0; segment < program->segment_count;
		     segment++) {
			const struct segment_record *const own =
				&program->segments[segment];
			expect(own->card < CARDS, "mixed segment names a card");
			if (own->card < CARDS) {
				expect(own->first_launch == launches[own->card],
				       "mixed segments cover launch arrays in order without gaps");
				launches[own->card] += own->launch_count;
			}
		}
		for (uint32_t copy = 0; copy < programs[0]->hand_off_count;
		     copy++)
			hand_offs +=
				programs[0]->hand_offs[copy].after_segment <
				program->segment_count;
		expect(program->hand_off_count == hand_offs,
		       "mixed hand-offs retain the complete thin schedule prefix");
		for (uint32_t card = 0; card < CARDS; card++) {
			const struct program_card_record *const area =
				&program->cards[card];
			const uint32_t views =
				thin || launches[card] == 0 ?
					0 :
					cases[index].rows / LINEAR_MAXIMUM_ROWS;
			expect(area->launch_count == launches[card],
			       "every mixed card launch belongs to exactly one segment");
			expect(area->round_offset ==
				       (thin && card == PRO_CARD ?
						model->canonical->cards[card].array_offset
							[V41_DATA_ARRAY_ROUND] :
						PROGRAM_NOWHERE),
			       "only THIN result-card programs publish a draft round record");
			expect(area->view_count == views &&
				       ((area->views != NULL) == (views != 0)),
			       "working chunk cards publish views; THIN and inactive cards do not");
			for (uint32_t view = 0;
			     area->views != NULL && view < area->view_count;
			     view++) {
				expect(area->views[view].block_offset ==
						       model->canonical->cards[card]
								       .array_offset
									       [V41_DATA_ARRAY_VIEWS] +
							       (uint64_t)view *
								       (STEP_BLOCK_BYTES_VALUE +
									LINEAR_MAXIMUM_ROWS *
										STEP_ROW_BYTES_VALUE) &&
					       area->views[view].first_row ==
						       view * LINEAR_MAXIMUM_ROWS &&
					       area->views[view].row_limit ==
						       LINEAR_MAXIMUM_ROWS,
				       "chunk views name consecutive canonical 8-row STEP slabs");
			}
			(void)printf(
				"mixed_area kind=%u rows=%u card=%u launches=%u/%u views=%u/%u\n",
				(uint32_t)program->kind, program->row_count,
				card, area->launch_count, launches[card],
				area->view_count, views);
		}
		(void)printf(
			"mixed_schedule kind=%u rows=%u segments=%u/%u groups=%u/%u handoffs=%u/%u\n",
			(uint32_t)program->kind, program->row_count,
			program->segment_count, wanted_segments,
			groups[index].pair_count, wanted_groups,
			program->hand_off_count, hand_offs);
	}
	if (complete) {
		struct step_driver *driver = NULL;
		status = step_driver_create(&test_host_memory_backend,
					    fixture.cards, programs,
					    COUNT_OF(programs), fixture.engram,
					    DRIVER_WAIT_NANOSECONDS, &driver);
		expect(status == STATUS_OK && driver != NULL,
		       "actual driver admits nine mixed programs with parsed full Engram geometry");
		if (status == STATUS_OK) {
			status = step_groups_attach(
				driver, groups, COUNT_OF(groups),
				model->stages[0]->description.rank_count);
			expect(status == STATUS_OK,
			       "actual group attachment admits every mixed program prefix");
		}
		step_driver_destroy(driver);
		driver = NULL;
		status = step_driver_create(&test_host_memory_backend,
					    fixture.cards, programs,
					    COUNT_OF(programs), NULL,
					    DRIVER_WAIT_NANOSECONDS, &driver);
		expect(status == STATUS_INVALID_ARGUMENT && driver == NULL,
		       "missing required Engram metadata still refuses without publishing a driver");
		step_driver_destroy(driver);
	} else
		expect(false, "mixed driver fixture is complete");
	expect(test_host_memory_backend_work_count() == 0,
	       "mixed creation and admission queue no launch, copy or digest work");
	expect(test_host_memory_backend_staging_block_count() == 0,
	       "mixed driver destruction retires all staging");
	for (uint32_t index = 0; index < COUNT_OF(owned); index++)
		free(owned[index]);
	fixture_destroy(&fixture);
	v41_tp2_model_destroy(model);
}

/* Check records at the first expert, independently of the program's segment coalescing. */
static void check_original_wide_records(const struct execution_program *program,
					const struct v41_tp2_model *model,
					uint32_t rows, bool wide)
{
	const uint32_t slab_rows = LINEAR_MAXIMUM_ROWS;
	const uint32_t slabs = rows / slab_rows;
	const uint32_t pairs = rows * MODEL_EXPERTS_PER_ROW;
	const uint64_t hidden_row = (uint64_t)MODEL_EXPERTS_PER_ROW *
				    V41_DATA_EXPERT_INNER *
				    LINEAR_BFLOAT16_BYTES;
	const uint64_t sum_row = (uint64_t)MODEL_HIDDEN * LINEAR_SUM_BYTES;
	for (uint32_t card = 0; card < TP4_RANKS; card++) {
		const struct program_card_record *const area =
			&program->cards[card];
		const struct v41_card *const memory =
			&model->canonical->cards[card];
		const struct v41_tp2_stage_plan *const stage =
			model->stages[card];
		uint32_t gate_count = 0, front_count = 0;
		for (uint32_t index = 0; index < area->launch_count; index++) {
			const struct kernel_launch *const launch =
				&area->launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (!test_host_memory_backend_kernel_of(
				    card, launch->kernel, &kernel)) {
				expect(false,
				       "original wide records have prepared kernels");
				continue;
			}
			if (strncmp(kernel.function_name, KERNEL_FFN_FRONT_STEM,
				    strlen(KERNEL_FFN_FRONT_STEM)) == 0) {
				struct site_front_arguments front;
				memcpy(&front, launch->argument, sizeof(front));
				expect(front.select.row_count == slab_rows &&
					       front.scratch_offset ==
						       memory->front_scratch_offset,
				       "original FFN fronts retain eight-row arithmetic and use the planned scratch");
				front_count++;
			}
			if (strncmp(kernel.function_name, "expert_gate_up_",
				    strlen("expert_gate_up_")) != 0)
				continue;
			struct expert_gate_up_arguments gate;
			memcpy(&gate, launch->argument, sizeof(gate));
			expect(gate.output_size == V41_DATA_EXPERT_INNER /
							   TP4_RANKS &&
				       gate.input_size == MODEL_HIDDEN &&
				       gate.pair_count ==
					       (wide ? pairs :
						       slab_rows *
								MODEL_EXPERTS_PER_ROW),
			       "original gate retains its full K and balanced output shard at the selected row count");
			gate_count++;
			if (!wide || gate_count != 1u)
				continue;
			const uint32_t following = index + 1u;
			const uint32_t quantize = following + slabs;
			const uint32_t down = quantize + 1u, sum = down + 1u;
			if (sum + slabs >= area->launch_count) {
				expect(false,
				       "original wide chain includes every gather, quantizer, down and sum");
				continue;
			}
			expect(gate.hidden_offset ==
				       memory->array_offset
						       [V41_DATA_ARRAY_PAIR_HIDDEN] +
					       (uint64_t)card * rows *
						       hidden_row / TP4_RANKS,
			       "original gate writes the chunk's owner-major hidden rectangle");
			struct activation_quantize_arguments activation;
			struct expert_down_arguments down_record;
			struct expert_row_sum_arguments row_sum;
			memcpy(&activation, area->launches[quantize].argument,
			       sizeof(activation));
			memcpy(&down_record, area->launches[down].argument,
			       sizeof(down_record));
			memcpy(&row_sum, area->launches[sum].argument,
			       sizeof(row_sum));
			expect(activation.values_offset ==
					       stage->exact_hidden.byte_offset &&
				       activation.width ==
					       V41_DATA_EXPERT_INNER &&
				       area->launches[quantize].grid[1] ==
					       pairs,
			       "one quantizer consumes every gathered full-width original pair");
			expect(down_record.input_size ==
					       V41_DATA_EXPERT_INNER &&
				       down_record.output_size ==
					       MODEL_HIDDEN / TP4_RANKS &&
				       down_record.pair_count == pairs &&
				       row_sum.width ==
					       MODEL_HIDDEN / TP4_RANKS &&
				       row_sum.pairs_per_row ==
					       MODEL_EXPERTS_PER_ROW &&
				       area->launches[sum].grid[1] == rows,
			       "original down and row sum preserve full K and the six-pair sum order");
			for (uint32_t slab = 0; slab < slabs; slab++) {
				const uint32_t at = slab * slab_rows;
				for (uint32_t site = 0; site < TP2_SITES;
				     site++) {
					const uint32_t gather =
						site == 0 ? following + slab :
							    sum + 1u + slab;
					const struct kernel_launch *const copy =
						&area->launches[gather];
					struct exact_ring_arguments record;
					memcpy(&record, copy->argument,
					       sizeof(record));
					const uint64_t row_bytes =
						site == 0 ? hidden_row :
							    sum_row;
					const uint64_t input =
						memory->array_offset
							[site == 0 ?
								 V41_DATA_ARRAY_PAIR_HIDDEN :
								 V41_DATA_ARRAY_ROW_SUMS];
					const uint64_t output =
						site == 0 ?
							stage->exact_hidden
								.byte_offset :
							stage->exact_sums
								.byte_offset;
					const uint64_t correction =
						((uint64_t)card *
							 (rows - slab_rows) +
						 at) *
						row_bytes / TP4_RANKS;
					expect(test_host_memory_backend_kernel_of(
						       card, copy->kernel,
						       &kernel) &&
						       strcmp(kernel.function_name,
							      KERNEL_EXACT_RING) ==
							       0 &&
						       record.input_offset ==
							       input + correction &&
						       record.output_offset ==
							       output +
								       (uint64_t)at *
									       row_bytes &&
						       record.vectors ==
							       slab_rows *
								       (site == 0 ?
										MODEL_EXPERTS_PER_ROW :
										1u) &&
						       record.packs_per_owner ==
							       slab_rows *
								       row_bytes /
								       TP4_RANKS /
								       EXACT_RING_PACK_BYTES &&
						       record.bank_offset ==
							       stage->channels[site]
									       .ring_offset +
								       v41_ring_bank_start(
									       stage->channels[site]
										       .ring_bank_bytes,
									       stage->channels[site]
										       .ring_row_bytes,
									       slab_rows) &&
						       record.rank == card &&
						       record.mode ==
							       EXACT_RING_GATHER,
					       "every original slab gathers its owner rectangle into canonical rows with unchanged banks");
				}
			}
		}
		expect(gate_count != 0,
		       "the original wide fixture inspects target experts on every CMP");
		expect(front_count == gate_count * (wide ? slabs : 1u),
		       "only expert calls coalesce; each original eight-row front remains present");
	}
}

static void check_original_wide_experts(const struct pack_record *pack)
{
	struct v41_tp2_model_description description = {
		.limits = { CONTEXT_TOKENS, TEST_ORIGINAL_WIDE_ROWS, 1u,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_REFERENCE, CMP_ORDER_REFERENCE,
			    STEP_DRAFT_MOST, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD,
		.exact_output_shards = true,
		.original_wide_experts = true
	};
	struct v41_tp2_model *models[TEST_ORIGINAL_WIDE_VARIANTS] = { NULL,
								      NULL };
	struct fixture fixture = { 0 };
	test_host_memory_backend_reset(CARDS, HOST_BYTES);
	bool complete = true;
	for (uint32_t card = 0; card < CARDS; card++) {
		description.cards[card].arithmetic_class =
			card == PRO_CARD ? ARITHMETIC_CLASS_SM_120_NATIVE :
					   ARITHMETIC_CLASS_SM_80;
		description.cards[card].memory_total_bytes =
			TEST_MIXED_MEMORY_BYTES;
		description.cards[card].memory_free_bytes =
			TEST_MIXED_MEMORY_BYTES;
		test_host_memory_backend_set_arithmetic_class(
			card, card == PRO_CARD ? ARITHMETIC_CLASS_SM_120 :
						 ARITHMETIC_CLASS_SM_80);
		complete = test_host_memory_backend.card_open(
				   card, HOST_BYTES, &fixture.cards[card]) ==
				   STATUS_OK &&
			   complete;
	}
	enum {
		BAD_CODE_WIRE,
		BAD_LAYOUT,
		BAD_BATCH,
		BAD_SPLIT,
		BAD_ORDER,
		BAD_WIRE,
		BAD_PROFILE,
		BAD_ORIGINAL_WIDE_CASES
	};
	for (uint32_t rejected = 0; rejected < BAD_ORIGINAL_WIDE_CASES;
	     rejected++) {
		struct v41_tp2_model_description invalid = description;
		switch (rejected) {
		case BAD_CODE_WIRE:
			invalid.original_code_wire = true;
			break;
		case BAD_LAYOUT:
			invalid.tensor_layout = V41_DATA_TP_LAYOUT_PAIRS;
			break;
		case BAD_BATCH:
			invalid.batch = true;
			break;
		case BAD_SPLIT:
			invalid.split_overlap = 1u;
			break;
		case BAD_ORDER:
			invalid.limits.linear_order = LINEAR_ORDER_FIXED;
			break;
		case BAD_WIRE:
			invalid.wire_format = ORDERED_PUSH_BF16;
			break;
		default:
			invalid.limits.linear_profile = LINEAR_PROFILE_SPEED;
			break;
		}
		struct v41_tp2_model *untouched = NULL;
		expect(v41_tp2_model_create(pack, &invalid, &untouched) ==
				       STATUS_INVALID_ARGUMENT &&
			       untouched == NULL,
		       "original wide admission refuses unqualified wire, layout, BATCH, split, order and profile");
	}
	for (uint32_t variant = 0; variant < COUNT_OF(models) && complete;
	     variant++) {
		description.original_wide_experts = variant != 0;
		const result_status status = v41_tp2_model_create(
			pack, &description, &models[variant]);
		expect(status == STATUS_OK && models[variant] != NULL,
		       "legacy and experimental original wide cold plans prepare");
		complete = status == STATUS_OK && models[variant] != NULL;
	}
	if (complete) {
		const uint64_t alignment = V41_DATA_ARRAY_ALIGNMENT_BYTES;
		const uint64_t scratch =
			(V41_DATA_FUSED_SCRATCH_BYTES + alignment - 1u) /
			alignment * alignment;
		const uint64_t extra =
			scratch +
			(uint64_t)(TEST_ORIGINAL_WIDE_ROWS -
				   LINEAR_MAXIMUM_ROWS) *
				(MODEL_EXPERTS_PER_ROW * V41_DATA_EXPERT_INNER *
					 LINEAR_BFLOAT16_BYTES +
				 MODEL_HIDDEN * LINEAR_SUM_BYTES);
		for (uint32_t card = 0; card < CARDS; card++) {
			expect(models[1]->canonical->cards[card]
						       .arena_byte_count -
					       models[0]
						       ->canonical->cards[card]
						       .arena_byte_count ==
				       (card == PRO_CARD ? 0u : extra),
			       "CMP arenas own chunk gathers and dedicated scratch; PRO storage is unchanged");
			if (card < TP4_RANKS) {
				expect(models[0]->stages[card]->exact_hidden
							       .byte_count ==
						       (uint64_t)LINEAR_MAXIMUM_ROWS *
							       MODEL_EXPERTS_PER_ROW *
							       V41_DATA_EXPERT_INNER *
							       LINEAR_BFLOAT16_BYTES &&
					       models[1]->stages[card]
							       ->exact_hidden
							       .byte_count ==
						       (uint64_t)TEST_ORIGINAL_WIDE_ROWS *
							       MODEL_EXPERTS_PER_ROW *
							       V41_DATA_EXPERT_INNER *
							       LINEAR_BFLOAT16_BYTES &&
					       models[0]->stages[card]
							       ->ring_rows ==
						       models[1]
							       ->stages[card]
							       ->ring_rows,
				       "legacy hidden storage stays at eight rows and wide storage does not widen ring banks");
			}
		}
		expect(models[0]->recipe_identity == models[1]->recipe_identity,
		       "row batching selects no different arithmetic identity before numerical qualification");
		for (uint32_t variant = 0; variant < COUNT_OF(models);
		     variant++) {
			struct v41_tp2_program_description build;
			complete = v41_tp2_model_program_description(
					   models[variant], &build) ==
				   STATUS_OK;
			struct execution_program *program = NULL;
			struct step_companion_program groups = { 0 };
			const result_status status =
				complete ?
					v41_tp_program_prepare(
						&build,
						&test_host_memory_backend,
						fixture.cards,
						description.cards, PROGRAM_WIDE,
						TEST_ORIGINAL_WIDE_ROWS,
						&program, &groups) :
					STATUS_INVALID_ARGUMENT;
			expect(status == STATUS_OK && program != NULL,
			       "legacy and original wide programs prepare");
			if (status == STATUS_OK && program != NULL)
				check_original_wide_records(
					program, models[variant],
					TEST_ORIGINAL_WIDE_ROWS, variant != 0);
			free(program);
			if (variant != 0 && complete) {
				static const struct {
					enum program_kind kind;
					uint32_t rows;
				} extra_cases[] = {
					{ PROGRAM_THIN, LINEAR_MAXIMUM_ROWS },
					{ PROGRAM_FULL,
					  TEST_ORIGINAL_WIDE_ROWS }
				};
				for (uint32_t extra_case = 0;
				     extra_case < COUNT_OF(extra_cases);
				     extra_case++) {
					program = NULL;
					const result_status extra_status =
						v41_tp_program_prepare(
							&build,
							&test_host_memory_backend,
							fixture.cards,
							description.cards,
							extra_cases[extra_case]
								.kind,
							extra_cases[extra_case]
								.rows,
							&program, &groups);
					expect(extra_status == STATUS_OK &&
						       program != NULL,
					       "original wide option retains THIN admission and prepares the complete FULL native PRO path");
					if (extra_status == STATUS_OK &&
					    program != NULL)
						check_original_wide_records(
							program,
							models[variant],
							extra_cases[extra_case]
								.rows,
							extra_cases[extra_case]
									.kind !=
								PROGRAM_THIN);
					free(program);
				}
				models[variant]
					->stages[1]
					->exact_hidden.byte_count = 0;
				program = NULL;
				expect(v41_tp_program_prepare(
					       &build,
					       &test_host_memory_backend,
					       fixture.cards, description.cards,
					       PROGRAM_WIDE,
					       TEST_ORIGINAL_WIDE_ROWS,
					       &program, &groups) ==
						       STATUS_INVALID_ARGUMENT &&
					       program == NULL,
				       "program admission rejects a missing gathered hidden range before preparing work");
			}
		}
	}
	expect(test_host_memory_backend_work_count() == 0,
	       "original wide cold checks queue no device work");
	fixture_destroy(&fixture);
	for (uint32_t variant = 0; variant < COUNT_OF(models); variant++)
		v41_tp2_model_destroy(models[variant]);
}

/* Identify the two reference core launches independently of segment/graph coalescing. All other records,
 * including state/window writes, queries, compressor/key/index operations and the native PRO, must stay literal. */
static bool original_core_launch(uint32_t card,
				 const struct kernel_launch *launch)
{
	struct test_host_memory_backend_kernel kernel;
	return launch->kernel != BACKEND_COPY &&
	       test_host_memory_backend_kernel_of(card, launch->kernel,
						  &kernel) &&
	       (strcmp(kernel.function_name, KERNEL_ATTENTION_CORE_SCORES) ==
			0 ||
		strcmp(kernel.function_name, KERNEL_ATTENTION_CORE_VALUES) ==
			0);
}

static void
check_original_core_normalized(const struct execution_program *baseline,
			       const struct execution_program *candidate,
			       uint32_t card)
{
	const struct program_card_record *const old = &baseline->cards[card];
	const struct program_card_record *const next = &candidate->cards[card];
	const uint32_t slabs = candidate->row_count / LINEAR_MAXIMUM_ROWS;
	expect(ATTENTION_CORE_KEYS == TEST_ORIGINAL_CORE_KEYS &&
		       MODEL_SELECTED == TEST_ORIGINAL_CORE_PICKS &&
		       MODEL_RING_SLOTS == TEST_ORIGINAL_CORE_RING_SLOTS,
	       "original core qualification retains 640 keys, 512 picks and 4096 physical ring slots");
	expect(old->view_count == slabs && next->view_count == slabs &&
		       old->views != NULL && next->views != NULL &&
		       memcmp(old->views, next->views,
			      (size_t)slabs * sizeof(*old->views)) == 0,
	       "core coalescing retains canonical STEP view descriptors for every eight-row slab");
	if (old->views == NULL || next->views == NULL)
		return;
	for (uint32_t index = 0; index < next->launch_count; index++) {
		const struct kernel_launch *const launch =
			&next->launches[index];
		if (!original_core_launch(card, launch))
			continue;
		struct attention_core_arguments_record core;
		struct test_host_memory_backend_kernel kernel;
		memcpy(&core, launch->argument, sizeof(core));
		const bool known = test_host_memory_backend_kernel_of(
			card, launch->kernel, &kernel);
		const bool scores = known &&
				    strcmp(kernel.function_name,
					   KERNEL_ATTENTION_CORE_SCORES) == 0;
		const uint32_t heads =
			(scores ? ATTENTION_CORE_SCORE_TILES_MOST :
				  ATTENTION_CORE_VALUE_TILES_MOST) *
			ATTENTION_CORE_HEADS_PER_BLOCK;
		expect(known && core.step_offset == next->step_offset &&
			       core.ring_session_stride ==
				       V41_DATA_RING_SESSION_BYTES &&
			       core.context_token_limit == CONTEXT_TOKENS &&
			       core.session_capacity == 1u &&
			       (core.pool_ratio == 0 ?
					core.pool_offset == 0 &&
						core.pool_session_stride == 0 &&
						core.selection_offset == 0 :
					core.pool_session_stride ==
						(uint64_t)CONTEXT_TOKENS /
							core.pool_ratio *
							MODEL_STATE_ROW_BYTES) &&
			       core.first_head == 0 &&
			       core.output_row_heads == 0 &&
			       core.heads_per_block == heads &&
			       launch->grid[0] == candidate->row_count &&
			       launch->grid[1] * heads == MODEL_HEADS_VALUE &&
			       launch->grid[2] ==
				       (scores ?
						ATTENTION_CORE_KEYS /
							ATTENTION_CORE_TILE_KEYS :
						MODEL_HEAD_DIMS /
							ATTENTION_CORE_TILE_DIMS),
		       "chunk reference cores keep fixed 640-key score extent, 4096-slot ring and full-head CTA geometry");
		uint32_t matched = 0;
		bool covered[LINEAR_WIDE_MOST_ROWS / LINEAR_MAXIMUM_ROWS] = {
			false
		};
		for (uint32_t before = 0; before < old->launch_count;
		     before++) {
			const struct kernel_launch *const slab_launch =
				&old->launches[before];
			if (slab_launch->kernel != launch->kernel ||
			    !original_core_launch(card, slab_launch))
				continue;
			struct attention_core_arguments_record slab;
			memcpy(&slab, slab_launch->argument, sizeof(slab));
			if (slab.sinks_offset != core.sinks_offset)
				continue;
			const struct step_view *view = NULL;
			for (uint32_t at = 0; at < old->view_count; at++) {
				if (old->views[at].block_offset ==
				    slab.step_offset)
					view = &old->views[at];
			}
			const bool valid_view =
				view != NULL &&
				view->row_limit == LINEAR_MAXIMUM_ROWS &&
				view->first_row % LINEAR_MAXIMUM_ROWS == 0 &&
				view->first_row + LINEAR_MAXIMUM_ROWS <=
					candidate->row_count;
			expect(valid_view,
			       "each old core reads a canonical eight-row STEP view of this chunk");
			if (!valid_view)
				continue;
			expect(!covered[view->first_row / LINEAR_MAXIMUM_ROWS],
			       "each canonical slab contributes exactly one core of this kind to the chunk");
			covered[view->first_row / LINEAR_MAXIMUM_ROWS] = true;
			const uint64_t row = view->first_row;
			struct attention_core_arguments_record normalized =
				core;
			normalized.step_offset = view->block_offset;
			normalized.queries_offset +=
				row * MODEL_HEADS_ROW_BYTES;
			normalized.output_offset += row * MODEL_HEADS_ROW_BYTES;
			normalized.scores_offset +=
				row * ATTENTION_CORE_SCORES_ROW_BYTES;
			normalized.rotation_offset +=
				row * MODEL_ROTATION_ROW_BYTES;
			if (core.pool_ratio != 0)
				normalized.selection_offset +=
					row * MODEL_SELECTION_ROW_BYTES;
			expect(slab_launch->grid[0] == LINEAR_MAXIMUM_ROWS &&
				       slab_launch->grid[1] ==
					       launch->grid[1] &&
				       slab_launch->grid[2] ==
					       launch->grid[2] &&
				       memcmp(slab_launch->block, launch->block,
					      sizeof(launch->block)) == 0 &&
				       slab_launch->dynamic_shared_byte_count ==
					       launch->dynamic_shared_byte_count &&
				       slab_launch->argument_byte_count ==
					       ATTENTION_CORE_ARGUMENT_BYTES &&
				       launch->argument_byte_count ==
					       ATTENTION_CORE_ARGUMENT_BYTES &&
				       slab_launch->branch == launch->branch &&
				       memcmp(&slab, &normalized,
					      sizeof(slab)) == 0,
			       "normalized core records differ only in STEP view and row rebases; pool/512-pick/rotation/sink/cone/context fields and geometry stay literal");
			matched++;
		}
		expect(matched == slabs,
		       "one chunk core covers exactly the retained core records of every slab of its layer");
	}
}

static void
check_original_core_records(const struct execution_program *baseline,
			    const struct execution_program *candidate,
			    const struct step_companion_program *old_groups,
			    const struct step_companion_program *new_groups,
			    bool changes_core)
{
	expect(baseline->segment_count == candidate->segment_count &&
		       baseline->front_segment_count ==
			       candidate->front_segment_count &&
		       baseline->draft_segment_first ==
			       candidate->draft_segment_first &&
		       old_groups->pair_count == new_groups->pair_count &&
		       old_groups->arithmetic_identity ==
			       new_groups->arithmetic_identity &&
		       memcmp(old_groups->pairs, new_groups->pairs,
			      (size_t)old_groups->pair_count *
				      sizeof(old_groups->pairs[0])) == 0,
	       "original core scheduling retains every companion, front/draft boundary and recipe identity");
	for (uint32_t segment = 0; segment < baseline->segment_count &&
				   segment < candidate->segment_count;
	     segment++) {
		const struct segment_record *const old =
			&baseline->segments[segment];
		const struct segment_record *const next =
			&candidate->segments[segment];
		expect(old->card == next->card && old->stage == next->stage &&
			       old->layer == next->layer &&
			       old->pause == next->pause,
		       "original core scheduling preserves the segment topology");
	}
	const uint32_t core_layers = candidate->kind == PROGRAM_WIDE ?
					     MODEL_DECODER_FIRST_LAYER :
					     TEST_PREFIX_LAYERS;
	for (uint32_t card = 0; card < CARDS; card++) {
		const struct program_card_record *const old =
			&baseline->cards[card];
		const struct program_card_record *const next =
			&candidate->cards[card];
		expect(old->arena_byte_count == next->arena_byte_count &&
			       old->step_offset == next->step_offset &&
			       old->result_offset == next->result_offset,
		       "the original core option grows no arena or program storage range");
		const bool changed = changes_core && card != PRO_CARD;
		uint32_t before = 0, after = 0, old_cores = 0, new_cores = 0;
		while (before < old->launch_count ||
		       after < next->launch_count) {
			while (changed && before < old->launch_count &&
			       original_core_launch(card,
						    &old->launches[before])) {
				old_cores++;
				before++;
			}
			while (changed && after < next->launch_count &&
			       original_core_launch(card,
						    &next->launches[after])) {
				const struct kernel_launch *const launch =
					&next->launches[after];
				struct attention_core_arguments_record core;
				memcpy(&core, launch->argument, sizeof(core));
				expect(launch->grid[0] ==
						       candidate->row_count &&
					       core.first_head == 0 &&
					       launch->grid[1] *
							       core.heads_per_block ==
						       MODEL_HEADS_VALUE &&
					       core.output_row_heads == 0,
				       "the original chunk core retains all 64 heads and complete output rows");
				new_cores++;
				after++;
			}
			if (before == old->launch_count ||
			    after == next->launch_count)
				break;
			expect(memcmp(&old->launches[before],
				      &next->launches[after],
				      sizeof(old->launches[before])) == 0,
			       "all non-core records retain their arguments, geometry and relative order");
			before++;
			after++;
		}
		expect(before == old->launch_count &&
			       after == next->launch_count,
		       "no non-core launch is added or removed, including the native PRO path");
		if (changed)
			check_original_core_normalized(baseline, candidate,
						       card);
		if (changed)
			expect(new_cores == core_layers *
						       ATTENTION_CORE_MOST_LAUNCHES &&
				       old_cores ==
					       new_cores *
						       (candidate->row_count /
							LINEAR_MAXIMUM_ROWS),
			       "one complete core replaces each layer's slab cores and layer 20 front stays coreless");
	}
}

static bool
original_core_cards_open(struct fixture *fixture,
			 struct v41_tp2_model_description *description)
{
	test_host_memory_backend_reset(CARDS, HOST_BYTES);
	bool complete = true;
	for (uint32_t card = 0; card < CARDS; card++) {
		description->cards[card].arithmetic_class =
			card == PRO_CARD ? ARITHMETIC_CLASS_SM_120_NATIVE :
					   ARITHMETIC_CLASS_SM_80;
		description->cards[card].memory_total_bytes =
			TEST_MIXED_MEMORY_BYTES;
		description->cards[card].memory_free_bytes =
			TEST_MIXED_MEMORY_BYTES;
		test_host_memory_backend_set_arithmetic_class(
			card, card == PRO_CARD ? ARITHMETIC_CLASS_SM_120 :
						 ARITHMETIC_CLASS_SM_80);
		complete = test_host_memory_backend.card_open(
				   card, HOST_BYTES, &fixture->cards[card]) ==
				   STATUS_OK &&
			   complete;
	}
	return complete;
}

static void original_core_flag_set(struct v41_tp2_model *model, bool enabled)
{
	for (uint32_t card = 0; card < CARDS; card++)
		model->stages[card]->description.chunk_core = enabled;
}

static void check_original_chunk_core(const struct pack_record *pack)
{
	struct v41_tp2_model_description description = {
		.limits = { CONTEXT_TOKENS, LINEAR_WIDE_MOST_ROWS, 1u,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_REFERENCE, CMP_ORDER_REFERENCE,
			    MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD,
		.exact_output_shards = true
	};
	struct fixture fixture = { 0 };
	struct v41_tp2_model *models[TEST_ORIGINAL_CORE_VARIANTS] = { NULL,
								      NULL };
	bool complete = original_core_cards_open(&fixture, &description);
	for (uint32_t variant = 0; variant < COUNT_OF(models) && complete;
	     variant++) {
		description.chunk_core = variant != 0;
		const result_status status = v41_tp2_model_create(
			pack, &description, &models[variant]);
		expect(status == STATUS_OK && models[variant] != NULL,
		       "both original core options create the complete cold five-card model");
		complete = status == STATUS_OK && models[variant] != NULL;
	}
	if (complete) {
		for (uint32_t card = 0; card < CARDS; card++)
			expect(models[0]->canonical->cards[card]
						       .arena_byte_count ==
					       models[1]
						       ->canonical->cards[card]
						       .arena_byte_count &&
				       memcmp(models[0]
						      ->canonical->cards[card]
						      .array_offset,
					      models[1]
						      ->canonical->cards[card]
						      .array_offset,
					      sizeof(models[0]
							     ->canonical
							     ->cards[card]
							     .array_offset)) ==
					       0,
			       "core-only scheduling changes no cold arena size or array offset");
		static const struct {
			enum program_kind kind;
			uint32_t rows;
		} cases[] = { { PROGRAM_THIN, 1u },
			      { PROGRAM_THIN, LINEAR_MAXIMUM_ROWS },
			      { PROGRAM_WIDE, MODEL_WIDE_LEAST_ROWS },
			      { PROGRAM_FULL, MODEL_WIDE_LEAST_ROWS },
			      { PROGRAM_WIDE, TEST_ORIGINAL_CORE_ROWS },
			      { PROGRAM_FULL, TEST_ORIGINAL_CORE_ROWS },
			      { PROGRAM_WIDE, LINEAR_WIDE_MOST_ROWS },
			      { PROGRAM_FULL, LINEAR_WIDE_MOST_ROWS },
			      { PROGRAM_WIDE, MODEL_WIDE_LEAST_ROWS + 1u },
			      { PROGRAM_FULL, MODEL_WIDE_LEAST_ROWS + 1u },
			      { PROGRAM_WIDE, TEST_ORIGINAL_CORE_ROWS - 1u },
			      { PROGRAM_FULL, TEST_ORIGINAL_CORE_ROWS - 1u },
			      { PROGRAM_WIDE, LINEAR_WIDE_MOST_ROWS - 1u },
			      { PROGRAM_FULL, LINEAR_WIDE_MOST_ROWS - 1u } };
		for (uint32_t shape = 0; shape < COUNT_OF(cases); shape++) {
			struct execution_program
				*programs[TEST_ORIGINAL_CORE_VARIANTS] = {
					NULL, NULL
				};
			struct step_companion_program
				groups[TEST_ORIGINAL_CORE_VARIANTS] = { 0 };
			bool prepared = true;
			for (uint32_t variant = 0; variant < COUNT_OF(models);
			     variant++) {
				struct v41_tp2_program_description build;
				result_status status =
					v41_tp2_model_program_description(
						models[variant], &build);
				if (status == STATUS_OK)
					status = v41_tp_program_prepare(
						&build,
						&test_host_memory_backend,
						fixture.cards,
						description.cards,
						cases[shape].kind,
						cases[shape].rows,
						&programs[variant],
						&groups[variant]);
				expect(status == STATUS_OK &&
					       programs[variant] != NULL,
				       "original core and partial-group fallback programs prepare with both flag settings");
				prepared = prepared && status == STATUS_OK &&
					   programs[variant] != NULL;
			}
			if (prepared)
				check_original_core_records(
					programs[0], programs[1], &groups[0],
					&groups[1],
					cases[shape].kind != PROGRAM_THIN &&
						cases[shape].rows %
								LINEAR_MAXIMUM_ROWS ==
							0);
			if (prepared && cases[shape].kind == PROGRAM_FULL &&
			    cases[shape].rows == TEST_ORIGINAL_CORE_ROWS) {
				original_core_flag_set(models[1], false);
				struct v41_tp2_program_description build;
				struct execution_program *disabled = NULL;
				struct step_companion_program restored = { 0 };
				result_status status =
					v41_tp2_model_program_description(
						models[1], &build);
				if (status == STATUS_OK)
					status = v41_tp_program_prepare(
						&build,
						&test_host_memory_backend,
						fixture.cards,
						description.cards,
						cases[shape].kind,
						cases[shape].rows, &disabled,
						&restored);
				expect(status == STATUS_OK && disabled != NULL,
				       "the old flag 0 core program remains directly selectable");
				if (status == STATUS_OK && disabled != NULL)
					check_original_core_records(
						programs[0], disabled,
						&groups[0], &restored, false);
				free(disabled);
				original_core_flag_set(models[1], true);
			}
			for (uint32_t variant = 0; variant < COUNT_OF(models);
			     variant++)
				free(programs[variant]);
		}
	}
	expect(test_host_memory_backend_work_count() == 0,
	       "original core metadata checks enqueue no GPU work");
	fixture_destroy(&fixture);
	for (uint32_t variant = 0; variant < COUNT_OF(models); variant++)
		v41_tp2_model_destroy(models[variant]);
}

/* A stage's opt-in cannot widen the numerical scope of an original BATCH model's FULL prefill or the independent
 * original expert experiment. Compare complete launch records with its inactive flag 0 path, not just core counts. */
static void check_original_core_exclusions(const struct pack_record *pack,
					   bool batch)
{
	struct v41_tp2_model_description description = {
		.limits = { CONTEXT_TOKENS, MODEL_WIDE_LEAST_ROWS,
			    batch ? V41_DATA_ORIGINAL_BATCH_SESSIONS : 1u,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_REFERENCE, CMP_ORDER_REFERENCE,
			    MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD,
		.exact_output_shards = true,
		.original_batch = batch,
		.batch = batch,
		.batch_context_slabs = batch,
		.original_wide_experts = !batch
	};
	struct fixture fixture = { 0 };
	struct v41_tp2_model *model = NULL;
	bool complete = original_core_cards_open(&fixture, &description);
	result_status status =
		complete ? v41_tp2_model_create(pack, &description, &model) :
			   STATUS_OUT_OF_MEMORY;
	expect(status == STATUS_OK && model != NULL,
	       "an excluded original model creates for fallback checks");
	complete = status == STATUS_OK && model != NULL;
	const enum program_kind kinds[] = { PROGRAM_FULL,
					    batch ? PROGRAM_BATCH :
						    PROGRAM_WIDE };
	for (uint32_t kind = 0; complete && kind < COUNT_OF(kinds); kind++) {
		struct execution_program
			*programs[TEST_ORIGINAL_CORE_VARIANTS] = { NULL, NULL };
		struct step_companion_program
			groups[TEST_ORIGINAL_CORE_VARIANTS] = { 0 };
		bool prepared = true;
		for (uint32_t variant = 0; variant < COUNT_OF(programs);
		     variant++) {
			original_core_flag_set(model, variant != 0);
			struct v41_tp2_program_description build;
			status = v41_tp2_model_program_description(model,
								   &build);
			if (status == STATUS_OK)
				status = v41_tp_program_prepare(
					&build, &test_host_memory_backend,
					fixture.cards, description.cards,
					kinds[kind],
					kinds[kind] == PROGRAM_BATCH ?
						V41_DATA_ORIGINAL_BATCH_ROWS :
						MODEL_WIDE_LEAST_ROWS,
					&programs[variant], &groups[variant]);
			expect(status == STATUS_OK && programs[variant] != NULL,
			       "excluded original BATCH/FULL/wide-expert programs retain their old admission");
			prepared = prepared && status == STATUS_OK &&
				   programs[variant] != NULL;
		}
		if (prepared)
			check_original_core_records(programs[0], programs[1],
						    &groups[0], &groups[1],
						    false);
		for (uint32_t variant = 0; variant < COUNT_OF(programs);
		     variant++)
			free(programs[variant]);
	}
	expect(test_host_memory_backend_work_count() == 0,
	       "excluded core options enqueue no device work");
	fixture_destroy(&fixture);
	v41_tp2_model_destroy(model);
}

/* Cold composition only: native-PRO slab heads and independent five-row draft walks.
 * No model launch, transfer, host stand-in arithmetic or real tensor payload executes. */
static void check_original_batch_refusals(
	const struct pack_record *pack,
	const struct v41_tp2_model_description *description)
{
	enum {
		ORIGINAL_BATCH_NO_FLAG,
		ORIGINAL_BATCH_NO_BATCH,
		ORIGINAL_BATCH_LAYOUT,
		ORIGINAL_BATCH_RELAXED,
		ORIGINAL_BATCH_WIRE,
		ORIGINAL_BATCH_ORDER,
		ORIGINAL_BATCH_PROFILE,
		ORIGINAL_BATCH_CMP,
		ORIGINAL_BATCH_EMULATED_PRO,
		ORIGINAL_BATCH_HEAD_SHARDS,
		ORIGINAL_BATCH_DRAFT_SHARDS,
		ORIGINAL_BATCH_WIDE_HEAD,
		ORIGINAL_BATCH_DRAFT_ROWS,
		ORIGINAL_BATCH_CONTEXT,
		ORIGINAL_BATCH_CORE,
		ORIGINAL_BATCH_WIDE_EXPERTS,
		ORIGINAL_BATCH_SPLIT,
		ORIGINAL_BATCH_SESSIONS,
		ORIGINAL_BATCH_DRAFT_WIDTH,
		ORIGINAL_BATCH_OWNER,
		ORIGINAL_BATCH_BAD_CASES
	};
	for (uint32_t rejected = 0; rejected < ORIGINAL_BATCH_BAD_CASES;
	     rejected++) {
		struct v41_tp2_model_description invalid = *description;
		switch (rejected) {
		case ORIGINAL_BATCH_NO_FLAG:
			invalid.original_batch = false;
			break;
		case ORIGINAL_BATCH_NO_BATCH:
			invalid.batch = false;
			break;
		case ORIGINAL_BATCH_LAYOUT:
			invalid.tensor_layout = V41_DATA_TP_LAYOUT_PAIRS;
			break;
		case ORIGINAL_BATCH_RELAXED:
			invalid.exact_output_shards = false;
			break;
		case ORIGINAL_BATCH_WIRE:
			invalid.wire_format = ORDERED_PUSH_BF16;
			break;
		case ORIGINAL_BATCH_ORDER:
			invalid.limits.linear_order = LINEAR_ORDER_FIXED;
			break;
		case ORIGINAL_BATCH_PROFILE:
			invalid.limits.linear_profile = LINEAR_PROFILE_SPEED;
			break;
		case ORIGINAL_BATCH_CMP:
			invalid.limits.cmp_order = CMP_ORDER_RELAXED;
			break;
		case ORIGINAL_BATCH_EMULATED_PRO:
			invalid.cards[PRO_CARD].arithmetic_class =
				ARITHMETIC_CLASS_SM_120;
			break;
		case ORIGINAL_BATCH_HEAD_SHARDS:
			invalid.head_shards = true;
			break;
		case ORIGINAL_BATCH_DRAFT_SHARDS:
			invalid.draft_shards = true;
			break;
		case ORIGINAL_BATCH_WIDE_HEAD:
			invalid.batch_wide_head = true;
			break;
		case ORIGINAL_BATCH_DRAFT_ROWS:
			invalid.batch_draft_rows = true;
			break;
		case ORIGINAL_BATCH_CONTEXT:
			invalid.batch_context_slabs = false;
			break;
		case ORIGINAL_BATCH_CORE:
			invalid.batch_core_once = 1u;
			break;
		case ORIGINAL_BATCH_WIDE_EXPERTS:
			invalid.original_wide_experts = true;
			break;
		case ORIGINAL_BATCH_SPLIT:
			invalid.split_overlap = 1u;
			break;
		case ORIGINAL_BATCH_SESSIONS:
			invalid.limits.session_capacity--;
			break;
		case ORIGINAL_BATCH_DRAFT_WIDTH:
			invalid.limits.draft_positions++;
			break;
		default:
			invalid.attention_owner = true;
			break;
		}
		struct v41_tp2_model *untouched = NULL;
		expect(v41_tp2_model_create(pack, &invalid, &untouched) ==
				       STATUS_INVALID_ARGUMENT &&
			       untouched == NULL,
		       "original BATCH refuses a missing opt-in or an unqualified role, arithmetic or joined optimization");
	}
}

static void
check_original_batch_records(const struct execution_program *program,
			     const struct v41_tp2_model *model)
{
	const struct v41_tp2_stage_plan *const tail = model->stages[PRO_CARD];
	const uint32_t slabs =
		V41_DATA_ORIGINAL_BATCH_ROWS / LINEAR_MAXIMUM_ROWS;
	const uint32_t positions = MODEL_DRAFT_BLOCK;
	const uint64_t stride =
		v41_batch_draft_offset(V41_DATA_BATCH_DRAFT_PARTS, positions);
	uint32_t samplers = 0, target_samplers = 0, verifies = 0,
		 confidences = 0;
	uint64_t draft_steps[V41_DATA_ORIGINAL_BATCH_SESSIONS] = { 0 };
	expect(program->card_count == CARDS &&
		       program->result_card == PRO_CARD &&
		       program->cards[PRO_CARD].result_offset ==
			       tail->batch_results.byte_offset &&
		       program->cards[PRO_CARD].round_offset ==
			       tail->batch_rounds.byte_offset,
	       "five-card original BATCH publishes native-PRO slab results and the session round table");
	expect(tail->batch_rounds.byte_offset >=
			       tail->batch_results.byte_offset &&
		       tail->batch_rounds.byte_offset -
				       tail->batch_results.byte_offset +
				       tail->batch_rounds.byte_count <=
			       STEP_BATCH_RESULT_SPAN_MOST &&
		       tail->batch_rounds.byte_count ==
			       (uint64_t)V41_DATA_ORIGINAL_BATCH_SESSIONS *
				       STEP_ROUND_BYTES &&
		       tail->batch_drafts.byte_count ==
			       (V41_DATA_ORIGINAL_BATCH_SESSIONS - 1u) * stride,
	       "one bounded result download holds every slab and round; seven extra verifier areas fit");
	for (uint32_t card = 0; card < CARDS; card++) {
		const struct program_card_record *const area =
			&program->cards[card];
		for (uint32_t index = 0; index < area->launch_count; index++) {
			const struct kernel_launch *const launch =
				&area->launches[index];
			struct test_host_memory_backend_kernel kernel;
			const bool found = test_host_memory_backend_kernel_of(
				card, launch->kernel, &kernel);
			expect(found,
			       "every BATCH launch resolves on its own card");
			if (!found)
				continue;
			if (strcmp(kernel.function_name, KERNEL_SAMPLER) == 0) {
				_Alignas(
					TEST_ORIGINAL_BATCH_RECORD_ALIGNMENT) struct sampler_arguments
					arguments;
				memcpy(&arguments, launch->argument,
				       sizeof(arguments));
				samplers++;
				expect(card == PRO_CARD &&
					       arguments.session_capacity ==
						       V41_DATA_ORIGINAL_BATCH_SESSIONS,
				       "the unsharded native PRO samples every target and draft for all owner slots");
				if (arguments.result_offset >=
					    tail->batch_results.byte_offset &&
				    arguments.result_offset -
						    tail->batch_results
							    .byte_offset <
					    tail->batch_results.byte_count) {
					expect(target_samplers <
							       area->view_count &&
						       arguments.result_offset ==
							       tail->batch_results
									       .byte_offset +
								       (uint64_t)target_samplers *
									       PROGRAM_BATCH_SLAB_RESULT_BYTES &&
						       arguments.step_offset ==
							       area->views[target_samplers]
								       .block_offset,
					       "each target slab reads its own dynamic step view and writes one distinct result header");
					target_samplers++;
				}
			} else if (strcmp(kernel.function_name,
					  KERNEL_VERIFY_MATCHES) == 0) {
				_Alignas(
					TEST_ORIGINAL_BATCH_RECORD_ALIGNMENT) struct verify_matches_arguments
					arguments;
				memcpy(&arguments, launch->argument,
				       sizeof(arguments));
				const uint64_t expected_step =
					verifies == 0 ?
						model->canonical
							->cards[PRO_CARD]
							.array_offset
								[V41_DATA_ARRAY_DRAFT_STEP] :
						tail->batch_drafts.byte_offset +
							(uint64_t)(verifies -
								   1u) *
								stride;
				const uint64_t expected_views =
					verifies == 0 ?
						model->canonical
							->cards[PRO_CARD]
							.array_offset
								[V41_DATA_ARRAY_DRAFT_VIEWS] :
						expected_step +
							v41_batch_draft_offset(
								V41_DATA_BATCH_DRAFT_VIEWS,
								positions);
				expect(card == PRO_CARD &&
					       verifies <
						       V41_DATA_ORIGINAL_BATCH_SESSIONS &&
					       arguments.verifier == verifies &&
					       arguments.flags ==
						       (DRAFT_ROUND_TABLE |
							DRAFT_SLAB_RESULTS) &&
					       arguments.row_count ==
						       V41_DATA_ORIGINAL_BATCH_ROWS &&
					       arguments.draft_positions ==
						       positions &&
					       arguments.round_capacity ==
						       V41_DATA_ORIGINAL_BATCH_SESSIONS &&
					       arguments.result_offset ==
						       tail->batch_results
							       .byte_offset &&
					       arguments.round_offset ==
						       tail->batch_rounds
							       .byte_offset &&
					       arguments.draft_step_offset ==
						       expected_step &&
					       arguments.draft_views_offset ==
						       expected_views,
				       "each verifier reads the slab results and prepares its separate legacy five-row draft block");
				if (verifies < V41_DATA_ORIGINAL_BATCH_SESSIONS)
					draft_steps[verifies] =
						arguments.draft_step_offset;
				verifies++;
			} else if (strcmp(kernel.function_name,
					  KERNEL_DRAFT_CONFIDENCE) == 0) {
				_Alignas(
					TEST_ORIGINAL_BATCH_RECORD_ALIGNMENT) struct draft_confidence_arguments
					arguments;
				memcpy(&arguments, launch->argument,
				       sizeof(arguments));
				expect(card == PRO_CARD &&
					       confidences <
						       V41_DATA_ORIGINAL_BATCH_SESSIONS &&
					       arguments.flags ==
						       DRAFT_ROUND_TABLE &&
					       arguments.draft_positions ==
						       positions &&
					       arguments.round_offset ==
						       tail->batch_rounds
							       .byte_offset &&
					       (confidences >=
							V41_DATA_ORIGINAL_BATCH_SESSIONS ||
						arguments.draft_step_offset ==
							draft_steps[confidences]),
				       "each confidence retains its verifier block and publishes its owner's table entry");
				confidences++;
			}
		}
	}
	expect(target_samplers == slabs &&
		       verifies == V41_DATA_ORIGINAL_BATCH_SESSIONS &&
		       confidences == verifies &&
		       samplers == slabs + verifies * positions,
	       "all six slab result headers and eight legacy verifier walks exist, including inactive-owner guards");
	for (uint32_t first = 0; first < V41_DATA_ORIGINAL_BATCH_SESSIONS;
	     first++)
		for (uint32_t second = first + 1u;
		     second < V41_DATA_ORIGINAL_BATCH_SESSIONS; second++)
			expect(draft_steps[first] != draft_steps[second],
			       "legacy verifier step blocks never alias");
}

static void check_original_batch(const struct pack_record *pack)
{
	struct v41_tp2_model_description description = {
		.limits = { CONTEXT_TOKENS, MODEL_WIDE_LEAST_ROWS,
			    V41_DATA_ORIGINAL_BATCH_SESSIONS, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_REFERENCE,
			    CMP_ORDER_REFERENCE, MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD,
		.exact_output_shards = true,
		.original_batch = true,
		.batch = true,
		.batch_context_slabs = true
	};
	struct fixture fixture = { 0 };
	struct v41_tp2_model *models[TEST_ORIGINAL_WIDE_VARIANTS] = { NULL,
								      NULL };
	struct execution_program *thin[TEST_ORIGINAL_WIDE_VARIANTS] = { NULL,
									NULL };
	struct execution_program *owned[3] = { NULL, NULL, NULL };
	const struct execution_program *programs[3] = { NULL, NULL, NULL };
	struct step_companion_program groups[3] = { 0 };
	test_host_memory_backend_reset(CARDS, HOST_BYTES);
	bool complete = true;
	for (uint32_t card = 0; card < CARDS; card++) {
		description.cards[card].arithmetic_class =
			card == PRO_CARD ? ARITHMETIC_CLASS_SM_120_NATIVE :
					   ARITHMETIC_CLASS_SM_80;
		description.cards[card].memory_total_bytes =
			TEST_MIXED_MEMORY_BYTES;
		description.cards[card].memory_free_bytes =
			TEST_MIXED_MEMORY_BYTES;
		test_host_memory_backend_set_arithmetic_class(
			card, card == PRO_CARD ? ARITHMETIC_CLASS_SM_120 :
						 ARITHMETIC_CLASS_SM_80);
		complete = test_host_memory_backend.card_open(
				   card, HOST_BYTES, &fixture.cards[card]) ==
				   STATUS_OK &&
			   complete;
	}
	check_original_batch_refusals(pack, &description);
	for (uint32_t variant = 0; variant < COUNT_OF(models) && complete;
	     variant++) {
		struct v41_tp2_model_description chosen = description;
		chosen.original_batch = variant != 0;
		chosen.batch = variant != 0;
		chosen.batch_context_slabs = variant != 0;
		const result_status created =
			v41_tp2_model_create(pack, &chosen, &models[variant]);
		expect(created == STATUS_OK && models[variant] != NULL,
		       "legacy and explicit original BATCH cold plans prepare");
		complete = created == STATUS_OK && models[variant] != NULL;
		struct v41_tp2_program_description build = { 0 };
		struct step_companion_program companion = { 0 };
		if (complete)
			complete = v41_tp2_model_program_description(
					   models[variant], &build) ==
				   STATUS_OK;
		if (complete) {
			const result_status prepared = v41_tp_program_prepare(
				&build, &test_host_memory_backend,
				fixture.cards, chosen.cards, PROGRAM_THIN,
				LINEAR_MAXIMUM_ROWS, &thin[variant],
				&companion);
			expect(prepared == STATUS_OK && thin[variant] != NULL,
			       "the original BATCH cold option also preserves the old THIN builder");
			complete = prepared == STATUS_OK &&
				   thin[variant] != NULL;
		}
	}
	if (complete) {
		expect(thin[0]->segment_count == thin[1]->segment_count &&
			       memcmp(thin[0]->segments, thin[1]->segments,
				      (size_t)thin[0]->segment_count *
					      sizeof(thin[0]->segments[0])) ==
				       0 &&
			       thin[0]->hand_off_count ==
				       thin[1]->hand_off_count &&
			       memcmp(thin[0]->hand_offs, thin[1]->hand_offs,
				      (size_t)thin[0]->hand_off_count *
					      sizeof(thin[0]->hand_offs[0])) ==
				       0,
		       "the experimental cold option leaves the old THIN segment and hand-off records byte-identical");
		for (uint32_t card = 0; card < CARDS; card++) {
			const struct program_card_record
				*const legacy = &thin[0]->cards[card],
				       *const candidate = &thin[1]->cards[card];
			expect(legacy->launch_count ==
					       candidate->launch_count &&
				       memcmp(legacy->launches,
					      candidate->launches,
					      (size_t)legacy->launch_count *
						      sizeof(legacy->launches
								     [0])) == 0,
			       "every old THIN launch record remains byte-identical, including native PRO products");
			if (card != PRO_CARD)
				expect(models[0]->canonical->cards[card]
						       .arena_byte_count ==
					       models[1]
						       ->canonical->cards[card]
						       .arena_byte_count,
				       "original BATCH adds no CMP arena bytes and retains the old eight-row gather banks");
		}
		struct v41_tp2_program_description build = { 0 };
		complete = v41_tp2_model_program_description(
				   models[1], &build) == STATUS_OK &&
			   driver_engram_create(&fixture);
		static const enum program_kind kinds[] = { PROGRAM_BATCH,
							   PROGRAM_WIDE,
							   PROGRAM_FULL };
		for (uint32_t index = 0; index < COUNT_OF(owned) && complete;
		     index++) {
			build.batch_verifiers =
				index == 0 ? V41_DATA_ORIGINAL_BATCH_SESSIONS :
					     0u;
			const uint32_t rows =
				index == 0 ? V41_DATA_ORIGINAL_BATCH_ROWS :
					     MODEL_WIDE_LEAST_ROWS;
			const result_status prepared = v41_tp_program_prepare(
				&build, &test_host_memory_backend,
				fixture.cards, description.cards, kinds[index],
				rows, &owned[index], &groups[index]);
			expect(prepared == STATUS_OK && owned[index] != NULL,
			       "the real five-card BATCH48 plus WIDE/FULL program set prepares");
			complete = prepared == STATUS_OK &&
				   owned[index] != NULL;
			programs[index] = owned[index];
		}
		if (complete) {
			check_original_batch_records(owned[0], models[1]);
			struct step_driver *driver = NULL;
			result_status status = step_driver_create(
				&test_host_memory_backend, fixture.cards,
				programs, COUNT_OF(programs), fixture.engram,
				DRIVER_WAIT_NANOSECONDS, &driver);
			expect(status == STATUS_OK && driver != NULL,
			       "the actual driver admits original BATCH48 with WIDE/FULL and parsed Engram ranges");
			if (status == STATUS_OK) {
				status = step_groups_attach(driver, groups,
							    COUNT_OF(groups),
							    TP4_RANKS);
				expect(status == STATUS_OK,
				       "original BATCH attaches all four CMP companion members");
				const uint32_t verifiers[] = {
					V41_DATA_ORIGINAL_BATCH_SESSIONS, 0u, 0u
				};
				status = step_batch_verifiers_attach(
					driver, verifiers, COUNT_OF(verifiers));
				expect(status == STATUS_OK,
				       "original BATCH attaches exactly eight legacy verifiers");
				status = step_rounds_domain_attach(
					driver, MODEL_VOCABULARY_VALUE,
					V41_DATA_END_TOKEN);
				expect(status == STATUS_OK,
				       "original BATCH binds its result token domain");
				struct step_row_record
					rows[V41_DATA_ORIGINAL_BATCH_SESSIONS];
				for (uint32_t slot = 0; slot < COUNT_OF(rows);
				     slot++)
					rows[slot] = (struct step_row_record){
						0, 1, slot,
						STEP_ROW_LOGITS |
							STEP_ROW_DRAFTS
					};
				expect(step_rows_check(
					       driver,
					       &rows[COUNT_OF(rows) - 1u], 1u,
					       true) == STATUS_OK &&
					       step_rows_check(driver, rows,
							       COUNT_OF(rows),
							       true) ==
						       STATUS_OK,
				       "cold row admission accepts one active last-slot owner with seven inactive slots and all eight owners");
				status = step_pipeline_start(driver, 1u);
				expect(status == STATUS_OK,
				       "original BATCH retains the eager single-flight declaration");
			}
			step_driver_destroy(driver);
			struct v41_tp2_stage_plan *const tail =
				models[1]->stages[PRO_CARD];
			tail->batch_drafts.byte_count--;
			struct execution_program *untouched = NULL;
			struct step_companion_program companion = { 0 };
			build.batch_verifiers =
				V41_DATA_ORIGINAL_BATCH_SESSIONS;
			expect(v41_tp_program_prepare(
				       &build, &test_host_memory_backend,
				       fixture.cards, description.cards,
				       PROGRAM_BATCH,
				       V41_DATA_ORIGINAL_BATCH_ROWS, &untouched,
				       &companion) == STATUS_INVALID_ARGUMENT &&
				       untouched == NULL,
			       "program admission refuses a shortened verifier scratch area before building");
			tail->batch_drafts.byte_count++;
			const uint32_t old_rank_count =
				tail->description.rank_count;
			tail->description.rank_count = TP2_RANKS;
			expect(v41_tp_program_prepare(
				       &build, &test_host_memory_backend,
				       fixture.cards, description.cards,
				       PROGRAM_BATCH,
				       V41_DATA_ORIGINAL_BATCH_ROWS, &untouched,
				       &companion) == STATUS_INVALID_ARGUMENT &&
				       untouched == NULL,
			       "program admission refuses a mutated native-PRO role");
			tail->description.rank_count = old_rank_count;
			build.batch_verifiers--;
			expect(v41_tp_program_prepare(
				       &build, &test_host_memory_backend,
				       fixture.cards, description.cards,
				       PROGRAM_BATCH,
				       V41_DATA_ORIGINAL_BATCH_ROWS, &untouched,
				       &companion) == STATUS_INVALID_ARGUMENT &&
				       untouched == NULL,
			       "original BATCH admission refuses an unqualified verifier topology");
		}
	}
	expect(test_host_memory_backend_work_count() == 0,
	       "original BATCH cold checks queue no model work");
	for (uint32_t index = 0; index < COUNT_OF(owned); index++)
		free(owned[index]);
	for (uint32_t variant = 0; variant < COUNT_OF(models); variant++) {
		free(thin[variant]);
		v41_tp2_model_destroy(models[variant]);
	}
	fixture_destroy(&fixture);
}

/* A real independently constructed EXL3 metadata pack must fail before any payload/card work.
 * Exercise both roles: topology flags alone do not establish original expert format. */
static void check_original_batch_nonoriginal_pack(void)
{
	test_memory_storage_reset();
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = {
		.kind = TEST_MODEL_V41_CHANGE_EXL3
	};
	const result_status opened =
		test_model_v41_draft_pack_open(change, &pack);
	expect(opened == STATUS_OK,
	       "the independent non-original expert metadata fixture opens");
	for (uint32_t role = 0;
	     opened == STATUS_OK && role < TEST_ORIGINAL_WIDE_VARIANTS;
	     role++) {
		const bool tail = role != 0;
		struct v41_tp2_stage_description description = {
			.limits = { CONTEXT_TOKENS, MODEL_WIDE_LEAST_ROWS,
				    V41_DATA_ORIGINAL_BATCH_SESSIONS,
				    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
				    LINEAR_ORDER_REFERENCE, CMP_ORDER_REFERENCE,
				    tail ? MODEL_DRAFT_BLOCK : 0u, 0, 0 },
			.card = tail ? PRO_CARD : 0u,
			.rank = 0u,
			.rank_count = tail ? 1u : TP4_RANKS,
			.first_target_layer = tail ? TP4_ATTENTION_PREFIX_END :
						     0u,
			.target_layer_count =
				tail ? MODEL_LAYERS - TP4_ATTENTION_PREFIX_END :
				       TP4_ATTENTION_PREFIX_END,
			.include_embedding = !tail,
			.include_head_and_draft = tail,
			.exact_output_shards = true,
			.original_batch = true,
			.batch = true,
			.batch_context_slabs = true
		};
		struct v41_tp2_stage_plan *untouched = NULL;
		expect(v41_tp2_stage_plan_create(pack, &description,
						 &untouched) ==
				       STATUS_UNSUPPORTED &&
			       untouched == NULL,
		       "original BATCH rejects EXL3 expert payload metadata in both the CMP prefix and native PRO tail");
	}
	pack_close(pack);
}

int main(void)
{
	struct fixture fixture = { 0 };
	struct pack_record *pack = NULL;
	test_memory_storage_reset();
	const result_status opened = test_model_v41_draft_pack_open(
		(struct test_model_v41_change){ TEST_MODEL_V41_CHANGE_NONE,
						NULL, 0 },
		&pack);
	expect(opened == STATUS_OK,
	       "full original-geometry draft metadata pack opens");
	if (opened == STATUS_OK && fixture_create(pack, &fixture) &&
	    driver_engram_create(&fixture)) {
		static const uint32_t rows[] = { 1u, 6u, 8u };
		struct execution_program *owned[COUNT_OF(rows)] = { 0 };
		const struct execution_program *programs[COUNT_OF(rows)] = { 0 };
		struct step_companion_program
			companion_programs[COUNT_OF(rows)] = { 0 };
		bool complete = true;
		for (uint32_t index = 0; index < COUNT_OF(rows); index++) {
			struct execution_program *program = NULL;
			struct step_companion_program companions = { 0 };
			const result_status status = v41_tp2_program_prepare(
				&fixture.description, &test_host_memory_backend,
				fixture.cards, fixture.card_descriptions,
				rows[index], &program, &companions);
			expect(status == STATUS_OK && program != NULL,
			       "full TP2 thin program prepares");
			if (program != NULL) {
				check_program(&fixture, program, &companions,
					      rows[index]);
				owned[index] = program;
				programs[index] = program;
				companion_programs[index] = companions;
			} else
				complete = false;
		}
		if (complete) {
			struct step_driver *driver = NULL;
			result_status status = step_driver_create(
				&test_host_memory_backend, fixture.cards,
				programs, COUNT_OF(programs), fixture.engram,
				DRIVER_WAIT_NANOSECONDS, &driver);
			expect(status == STATUS_OK && driver != NULL,
			       "actual step driver admits all real program offsets/ranges/checkpoints"
			       " and row variants");
			if (status == STATUS_OK) {
				status = step_companions_attach(
					driver, companion_programs,
					COUNT_OF(companion_programs));
				expect(status == STATUS_OK,
				       "actual companion attachment admits the full TP schedule");
				expect(test_host_memory_backend_work_count() ==
					       0,
				       "driver creation and attachment queue no launch/copy/digest work");
				(void)printf(
					"driver_create=%s companion_attach=%s\n",
					"OK", status_text(status));
			}
			step_driver_destroy(driver);
			expect(test_host_memory_backend_staging_block_count() ==
				       0,
			       "driver destruction retires its cold staging allocations");
		}
		for (uint32_t index = 0; index < COUNT_OF(owned); index++)
			free(owned[index]);
		check_refusals(&fixture);
		check_layer_prototype(pack, &fixture);
		for (uint32_t card = 0; card < CARDS; card++) {
			fixture.stages[card]->description.wire_format =
				ORDERED_PUSH_BF16;
			for (uint32_t channel = 0;
			     channel < fixture.stages[card]->channel_count;
			     channel++)
				fixture.stages[card]
					->channels[channel]
					.collective.arguments.payload_format =
					ORDERED_PUSH_BF16;
		}
		struct execution_program *bf16_program = NULL;
		struct step_companion_program bf16_companions = { 0 };
		const result_status bf16_status = v41_tp2_program_prepare(
			&fixture.description, &test_host_memory_backend,
			fixture.cards, fixture.card_descriptions, rows[1],
			&bf16_program, &bf16_companions);
		expect(bf16_status == STATUS_OK && bf16_program != NULL,
		       "explicit BF16 wire prepares all original TP layers");
		if (bf16_program != NULL) {
			check_program(&fixture, bf16_program, &bf16_companions,
				      rows[1]);
			expect(bf16_companions.arithmetic_identity ==
				       (V41_DATA_TP2_PROGRAM_RECIPE ^
					TEST_SLICES ^
					((uint64_t)ORDERED_PUSH_BF16
					 << V41_DATA_TP2_WIRE_IDENTITY_SHIFT)),
			       "wire rounding has a distinct immutable arithmetic identity");
		}
		free(bf16_program);
	} else
		expect(false, "cold fixture prepares");
	fixture_destroy(&fixture);
	if (opened == STATUS_OK) {
		check_original_wide_experts(pack);
		check_original_batch(pack);
		check_original_chunk_core(pack);
		check_original_core_exclusions(pack, false);
		check_original_core_exclusions(pack, true);
		for (uint32_t layout = 0; layout < V41_DATA_TP_LAYOUT_COUNT;
		     layout++) {
			for (uint32_t wire = ORDERED_PUSH_FP32;
			     wire <= ORDERED_PUSH_BF16; wire++)
				check_mixed_driver(pack, layout, wire);
		}
	}
	pack_close(pack);
	check_original_batch_nonoriginal_pack();
	test_memory_storage_reset();
	expect(test_host_memory_backend_open_card_count() == 0 &&
		       test_host_memory_backend_staging_block_count() == 0,
	       "host-only fixture releases cards and staging");
	return failures != 0 ? 1 : 0;
}
