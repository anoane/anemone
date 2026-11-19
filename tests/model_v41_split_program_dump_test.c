/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_split_program_dump_test: the programs each GPU split prepares, through commands.c's own parse, card choice,
 * plans and program list, as canonical digests, so that two source epochs can be shown to prepare the same programs.
 * It extends model_v41_program_dump_test (the PP thin, wide and full programs of a fixed placement) to every split, TP
 * layout, BATCH program, image program and head bank.
 *
 *     model_v41_split_program_dump_test [--dump] [--raw] [--case=NAME]
 *
 * - The cases: one for each distinct setting of the axis rows (cli_axes_data.h: a row's GPU split, inserted settings
 *   and slot rule; a held row inserts none and is counted), named by its first row and run at --session_slots=8 where
 *   the row applies only there; then the knob cases below, which no axis row reaches. Every row that is not held must
 *   parse, or the run fails. A case whose programs equal an earlier case's byte for byte is dumped as that case's.
 * - A case runs in a child of its own (pristine commands.c state): its serve's options axis-expanded and parsed as
 *   commands.c does, on the model's tables (model_v41_pack.h, with the draft; an EXL3 row's in their EXL3 form). Its
 *   cards are the workstation's as serve_cards_describe takes them from a stand-in backend (four CMP 170HX at
 *   ordinals 0, 1, 3 and 4, the PRO 6000 at 2) with serve_cards_arithmetic's classes, described with roomy memory so
 *   that no plan is refused for memory; host-memory cards hold the programs.
 * - Its plan: on PP5 the layout text (or the fill), plan_create, the tower behind card 0's text arena (as
 *   model_vision_program_test registers it; the pack holds one, so every PP5 case but pp5_text) and the head offload's
 *   plan and banks; with relaxed CMP order the folds as load refuses them all, or in an *_admitted case as load admits
 *   them (synthetic: model_v41_program_dump_test's trunk folds on the E4M3-scaled tensors, and every layer's routed
 *   experts the eligibility of expert_folds_prepare). On the TP splits tensor_plan_prepare and the TP program
 *   description. Of the serve's --session_slots, or DUMP_SESSIONS for max (which the cards' fit resolves).
 * - Its programs: serve_programs_choose's list, in its order, each of which must prepare; on a grid case also THIN of
 *   1 to 8 rows and WIDE and FULL of each of chunk_rows, up to the plan's row capacity, that the list does not hold (a
 *   refused one is a line with its status, so a refusal is compared too; THIN within the capacity and the chunks of 64
 *   and 1,024 rows must prepare); the image programs over them where model_images_are_enabled; with
 *   --original_head_offload=2 the head's flight banks over them.
 * - --dump prints, per case: its expanded options, the cards' order and classes, each plan field by field (every
 *   scalar, offset, tensor, expert table, vision and head range, each card's image bytes) as a section's byte count and
 *   SHA-256 (with --raw its bytes): PP5's plan, or a TP split's canonical, views and stage plans (with their jobs and
 *   channels) and its load jobs; the serve's program list; per program its scalars, limits, leases, checkpoints,
 *   segments, hand-offs, expert pauses, host experts, session ranges, each card's offsets, views and launches and the
 *   TP companions; and each card's prepared kernels in handle order (module, function, record bytes): a launch names
 *   its kernel by handle, so a renamed module is compared through that table. --case runs one case. Without --dump
 *   each case prepares its serve's list, images and banks and prints nothing but the verdict. No CUDA context, no model
 *   extent and no kernel run.
 * - Not dumped: the tower as plan_create places it from a pack (registered here), the image encoder's per-image
 *   launches (vision_prepare), the step graphs' grouping, the driver's attachments (BATCH verifiers, TP groups and
 *   launchers, flight programs and image programs) and a TP stage's description (its input; its plan is dumped).
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "anemone/loader.h"
#include "anemone/placement.h"
#include "anemone/relaxed_linear_records.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"
#include "model_v41_tp2_model.h"
#include "model_v41_tp2_stage.h"
#include "sha256.h"
#include "commands.h"

#define DUMP_SESSIONS \
	2u /* more than one, so that every per-session stride is in the records */
#define DUMP_HOST_CARD_BYTES 4096u
#define DUMP_ROOMY_BYTES (1ull << 42u) /* a card that holds any plan */
#define DUMP_WAIT_NANOSECONDS 1000000000ull
#define DUMP_SMOKE_CHUNK_ROWS 1024u
#define DUMP_CASE_MOST_WORDS 12u /* a case's options */
#define DUMP_WORD_BYTES 64u
#define DUMP_MOST_PROGRAMS 96u
#define DUMP_MOST_CASES 64u
#define DUMP_NO_CARD UINT32_MAX
#define DUMP_CASE_OPTION "--case="
#define DUMP_BATCH_SLOTS \
	"--session_slots=8" /* the slot count a BATCH axis row applies at */
#define DUMP_READ_BYTES 65536u
#define DUMP_GROWTH 2u /* a buffer grows to twice what it must hold */
#define DUMP_NAME_BYTES 128u
#define DUMP_PIPE_ENDS 2 /* read, write */
#define DUMP_RANGE_WORDS 2u /* struct linear_range's 64-bit words */
#define DUMP_SHAPE_FIELDS 4u /* struct matrix_shape_record's 32-bit fields */
#define DUMP_MACHINE_CARDS 5u
#define DUMP_MACHINE_PRO 2u /* the workstation's CUDA ordinal of the PRO */
#define DUMP_HEAD_CARD_RANGES 29u /* struct v41_original_head_card's ranges */
#define DUMP_VISION_WEIGHT_RANGES \
	(10u + 8u * VISION_MAXIMUM_LAYERS) /* struct vision_weights' ranges */
#define DUMP_PUSH_WORDS 9u /* struct ordered_push_arguments' 64-bit words */
#define DUMP_PUSH_FIELDS 6u /* and its 32-bit ones */
#define DUMP_LIMIT_FIELDS 10u /* struct model_limits' 32-bit fields */
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

/* The records hashed whole have no padding (each of their fields 64 or 32 bits wide, as counted here). */
_Static_assert(sizeof(struct linear_range) ==
		       DUMP_RANGE_WORDS * sizeof(uint64_t),
	       "a range has no padding");
_Static_assert(sizeof(struct matrix_shape_record) ==
		       DUMP_SHAPE_FIELDS * sizeof(uint32_t),
	       "a shape has no padding");
_Static_assert(sizeof(struct model_limits) ==
		       DUMP_LIMIT_FIELDS * sizeof(uint32_t),
	       "the limits have no padding");
_Static_assert(sizeof(struct v41_original_head_card) ==
		       DUMP_HEAD_CARD_RANGES * sizeof(struct linear_range),
	       "a head card is ranges");
_Static_assert(sizeof(struct vision_weights) ==
		       DUMP_VISION_WEIGHT_RANGES * sizeof(struct linear_range),
	       "the tower's weights are ranges");
_Static_assert(sizeof(struct ordered_push_arguments) ==
		       DUMP_PUSH_WORDS * sizeof(uint64_t) +
			       DUMP_PUSH_FIELDS * sizeof(uint32_t),
	       "a push's arguments have no padding");

static const char cmp_name[] = "NVIDIA CMP 170HX";
static const char pro_name[] =
	"NVIDIA RTX PRO 6000 Blackwell Workstation Edition";
/* An axis row's words, in startup_axes.h's order (cli_axes_data.h's domains) */
static const char *const axis_prefixes[AXIS_COUNT] = {
	"--category=", "--exactness=", "--concurrency=", "--focus="
};
/* The PP5 base's layout text places the tower first; the model's tables hold none, so the statement is left out. */
static const char vision_statement[] = "vision: card 0; ";

/* The two TP2 splits have no axis base: tp4_pro's machine and draft lines, in its words */
#define TP2_OPTIONS                                               \
	"--card_order=0,1,3,4,2", "--rows=8", "--wide_rows=1024", \
		"--prefill_cone=1", "--speculative=2", "--draft_positions=5"

enum dump_folds {
	DUMP_FOLDS_REFUSED,
	DUMP_FOLDS_ADMITTED
}; /* relaxed CMP order: load's fold checks */

/* A serve's words around a case's options: the program and command before them, the tables and pack after */
enum dump_fixed_word {
	DUMP_PROGRAM_WORD,
	DUMP_COMMAND_WORD,
	DUMP_TABLES_WORD,
	DUMP_PACK_WORD,
	DUMP_FIXED_WORDS
};

struct dump_case {
	char name[DUMP_NAME_BYTES];
	const char *options[DUMP_CASE_MOST_WORDS];
	enum test_model_v41_change_kind pack_kind;
	bool text_only; /* PP5: a pack without the tower */
	bool grid; /* the grid's programs beside the serve's */
	enum dump_folds folds;
};

/* The cases no axis row reaches: each split's grid, the head offloads, the relaxed CMP order's two fold outcomes, a
 * smaller context, the emulated PRO, a pack without the tower, the TP knobs and the TP2 splits */
static const struct dump_case knob_cases[] = {
	{ "pp5",
	  { "--placement=pp5" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  true,
	  DUMP_FOLDS_REFUSED },
	{ "pp5_text",
	  { "--placement=pp5" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  true,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "pp5_head_leased",
	  { "--placement=pp5", "--original_head_offload=1" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "pp5_head_banks",
	  { "--placement=pp5", "--original_head_offload=2" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "pp5_head_banks_multiple",
	  { "--placement=pp5", "--concurrency=multiple",
	    "--original_head_offload=2" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "pp5_relaxed",
	  { "--placement=pp5", "--numerics=relaxed", "--cmp_order=relaxed" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "pp5_relaxed_admitted",
	  { "--placement=pp5", "--numerics=relaxed", "--cmp_order=relaxed" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_ADMITTED },
	{ "pp5_context_131072",
	  { "--placement=pp5", "--context_tokens=131072" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "pp5_pro_emulation",
	  { "--placement=pp5", "--pro_emulation=1" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "tp4_pro",
	  { "--placement=tp4_pro" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  true,
	  DUMP_FOLDS_REFUSED },
	{ "tp4_pro_attention_owner",
	  { "--placement=tp4_pro", "--tp_attention_owner=1" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "tp4_pro_bf16",
	  { "--placement=tp4_pro", "--numerics=relaxed", "--tp_wire_bf16=1" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "tp4_pro_original_batch",
	  { "--placement=tp4_pro", "--original_batch=1", "--pipeline_steps=1",
	    "--batch_rows=48", "--batch_verifiers=8", "--session_slots=8",
	    "--batch_context_slabs=1", "--batch_core_once=0",
	    "--exl3_wide_entry=0", "--tp_split_overlap=0",
	    "--batch_draft_wide=0" },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  false,
	  DUMP_FOLDS_REFUSED },
	{ "tp2_pp2_pro",
	  { "--placement=tp2_pp2_pro", "--numerics=relaxed", TP2_OPTIONS },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  true,
	  DUMP_FOLDS_REFUSED },
	{ "tp2_single_single_pro",
	  { "--placement=tp2_single_single_pro", "--numerics=relaxed",
	    TP2_OPTIONS },
	  TEST_MODEL_V41_CHANGE_NONE,
	  false,
	  true,
	  DUMP_FOLDS_REFUSED },
	{ "tp4_exl3",
	  { "--placement=tp4_exl3" },
	  TEST_MODEL_V41_CHANGE_EXL3,
	  false,
	  true,
	  DUMP_FOLDS_REFUSED },
};

/* The grid's WIDE and FULL rows: the engine start's ladder, its tails and thin tails, and the wide rows it takes */
static const uint32_t chunk_rows[] = { 16,   24,   32,	 40,   48,  56,	 64,
				       128,  256,  320,	 448,  512, 768, 1024,
				       1280, 1536, 1792, 2048, 4096 };

static int failures;
static bool is_dumped, is_raw;
static struct backend_operations
	machine; /* the workstation's cards as the CUDA backend numbers them */

static void failed(const char *what, result_status status)
{
	(void)fprintf(stderr, "FAILED: %s: %s\n", what, status_text(status));
	failures++;
}

/* One canonical section: its name, card (or DUMP_NO_CARD), byte count and SHA-256, or with --raw its bytes. */
static void section(const char *name, uint32_t card, const void *bytes,
		    size_t count)
{
	if (!is_dumped)
		return;
	if (card != DUMP_NO_CARD)
		(void)printf("  %s %" PRIu32 " %zu ", name, card, count);
	else
		(void)printf("  %s %zu ", name, count);
	unsigned char digest[SHA256_DIGEST_BYTES];
	const result_status status =
		sha256_of(count == 0 ? "" : bytes, count, digest);
	if (is_raw && count != 0 && fwrite(bytes, 1, count, stdout) != count)
		failed("a section could not be written",
		       STATUS_INVALID_ARGUMENT);
	for (uint32_t index = 0;
	     !is_raw && status == STATUS_OK && index < SHA256_DIGEST_BYTES;
	     index++)
		(void)printf("%02x", (unsigned int)digest[index]);
	if (status != STATUS_OK)
		failed("a section's digest", status);
	(void)putchar('\n');
}

/* A record's fields one after another, each its own bytes: never a padded struct whole */
struct canonical {
	unsigned char *bytes;
	size_t count, capacity;
};

static void put(struct canonical *own, const void *bytes, size_t count)
{
	if (own->count + count > own->capacity) {
		const size_t capacity = (own->count + count) * DUMP_GROWTH;
		unsigned char *const grown = realloc(own->bytes, capacity);
		if (grown == NULL) {
			failed("a section's bytes", STATUS_OUT_OF_MEMORY);
			return;
		}
		own->bytes = grown;
		own->capacity = capacity;
	}
	(void)memcpy(own->bytes + own->count, bytes, count);
	own->count += count;
}
#define PUT(own, field) put((own), &(field), sizeof(field))

/* The section of what own holds, which then starts empty */
static void canonical_section(struct canonical *own, const char *name,
			      uint32_t card)
{
	section(name, card, own->bytes, own->count);
	own->count = 0;
}

static void tensor_put(struct canonical *own, const struct v41_tensor *tensor)
{
	PUT(own, tensor->stored);
	PUT(own, tensor->shape);
	const uint32_t format = (uint32_t)tensor->weight_format;
	PUT(own, format);
	PUT(own, tensor->card);
	PUT(own, tensor->fold_reference);
}

static void card_put(struct canonical *own, const struct v41_card *card)
{
	PUT(own, card->array_offset);
	PUT(own, card->expert_table_offset);
	PUT(own, card->selection_offset);
	PUT(own, card->ring_offset);
	PUT(own, card->checkpoint_offset);
	PUT(own, card->compressor_offset);
	PUT(own, card->pool_offset);
	PUT(own, card->key_tiles_offset);
	PUT(own, card->state_offset);
	PUT(own, card->arena_byte_count);
	PUT(own, card->official_scratch_offset);
	PUT(own, card->front_scratch_offset);
	PUT(own, card->exl3_scratch_offset);
	PUT(own, card->producer_arrivals_offset);
	PUT(own, card->slots_byte_count);
	PUT(own, card->image_byte_count);
	const uint32_t has_image = card->image != NULL ? 1u : 0u;
	PUT(own, has_image);
	PUT(own, card->layers);
	PUT(own, card->holders);
	PUT(own, card->engram_tables);
	PUT(own, card->slot_count);
	PUT(own, card->reserved_zero);
}

/* A plan field by field: its scalars, tensors, per-layer tables, tower, head and cards, and each card's image bytes */
static void plan_dump(const char *label, uint32_t index,
		      const struct model_plan_record *own_plan,
		      struct canonical *own)
{
	if (is_dumped)
		(void)printf("plan %s %" PRIu32 " cards %" PRIu32 "\n", label,
			     index, own_plan->card_count);
	PUT(own, own_plan->limits);
	PUT(own, own_plan->expert_matrix_offsets);
	PUT(own, own_plan->expert_byte_count);
	PUT(own, own_plan->slot_byte_count);
	PUT(own, own_plan->embedding_card);
	PUT(own, own_plan->result_card);
	PUT(own, own_plan->card_count);
	PUT(own, own_plan->reserved_zero);
	PUT(own, own_plan->chunk_attention_arrays);
	canonical_section(own, "plan_scalars", DUMP_NO_CARD);
	for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++)
		for (uint32_t tensor = 0; tensor < V41_DATA_LAYER_TENSOR_COUNT;
		     tensor++)
			tensor_put(own,
				   &own_plan->layer_tensors[layer][tensor]);
	for (uint32_t tensor = 0; tensor < V41_DATA_GLOBALS; tensor++)
		tensor_put(own, &own_plan->globals[tensor]);
	for (uint32_t tensor = 0; tensor < V41_DATA_DRAFT_GLOBALS; tensor++)
		tensor_put(own, &own_plan->draft_globals[tensor]);
	canonical_section(own, "tensors", DUMP_NO_CARD);
	PUT(own, own_plan->experts);
	PUT(own, own_plan->expert_tables);
	PUT(own, own_plan->host_expert_offsets);
	PUT(own, own_plan->host_expert_count);
	PUT(own, own_plan->exl3_bits);
	PUT(own, own_plan->expert_folds_admitted);
	PUT(own, own_plan->pool_session_bytes);
	PUT(own, own_plan->key_session_bytes);
	PUT(own, own_plan->layer_card);
	canonical_section(own, "layers", DUMP_NO_CARD);
	const struct model_vision *const vision = &own_plan->vision;
	PUT(own, vision->weights);
	PUT(own, vision->router_bias);
	PUT(own, vision->patches);
	PUT(own, vision->scratch);
	PUT(own, vision->encoded);
	PUT(own, vision->session_embeddings);
	PUT(own, vision->session_metadata);
	PUT(own, vision->session_embedding_stride);
	PUT(own, vision->text_arena_bytes);
	PUT(own, vision->enabled);
	canonical_section(own, "vision", DUMP_NO_CARD);
	const struct v41_original_head *const head = &own_plan->original_head;
	PUT(own, head->cards);
	PUT(own, head->enabled);
	PUT(own, head->connected);
	PUT(own, head->initialised);
	PUT(own, head->bank_count);
	canonical_section(own, "head", DUMP_NO_CARD);
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++)
		card_put(own, &own_plan->cards[card]);
	canonical_section(own, "cards", DUMP_NO_CARD);
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++)
		if (own_plan->cards[card].image != NULL)
			section("image", card, own_plan->cards[card].image,
				(size_t)own_plan->cards[card].image_byte_count);
}

static void jobs_put(struct canonical *own, const struct tp_load_job *jobs,
		     uint32_t count)
{
	for (uint32_t index = 0; jobs != NULL && index < count; index++) {
		const struct tp_load_job *const job = &jobs[index];
		PUT(own, job->tensor_index);
		PUT(own, job->card);
		PUT(own, job->destination_offset);
		const uint32_t mode = (uint32_t)job->mode,
			       format = (uint32_t)job->shard.description.format;
		PUT(own, mode);
		PUT(own, job->reserved_zero);
		PUT(own, format);
		PUT(own, job->shard.description.source_shape);
		PUT(own, job->shard.description.first_output);
		PUT(own, job->shard.description.output_count);
		PUT(own, job->shard.description.first_input);
		PUT(own, job->shard.description.input_count);
		PUT(own, job->shard.shard_shape);
		PUT(own, job->shard.source_byte_count);
		PUT(own, job->shard.shard_byte_count);
	}
}

/* A TP stage: its plan, then its own ranges, counts, load plan, jobs and channels field by field */
static void stage_dump(uint32_t card, const struct v41_tp2_stage_plan *stage,
		       struct canonical *own)
{
	plan_dump("stage", card, stage->model, own);
	PUT(own, stage->compact);
	PUT(own, stage->exact_hidden);
	PUT(own, stage->exact_sums);
	PUT(own, stage->dense_input);
	PUT(own, stage->index_scores);
	PUT(own, stage->index_rows);
	PUT(own, stage->sampler_exchange);
	PUT(own, stage->sampler_peers);
	PUT(own, stage->sampler_settings);
	PUT(own, stage->sampler_settings_peers);
	PUT(own, stage->draft_area);
	PUT(own, stage->draft_peers);
	PUT(own, stage->draft_counters);
	PUT(own, stage->batch_results);
	PUT(own, stage->batch_logits);
	PUT(own, stage->batch_rounds);
	PUT(own, stage->batch_drafts);
	PUT(own, stage->batch_rows);
	PUT(own, stage->exl3_tc_workspace);
	PUT(own, stage->exl3_tcp_workspace);
	PUT(own, stage->exl3_tail_workspace);
	PUT(own, stage->trunk_mma_partials);
	PUT(own, stage->trunk_mma_set_partials);
	PUT(own, stage->router_partials);
	PUT(own, stage->exl3_shards);
	PUT(own, stage->tensor_payload_byte_count);
	PUT(own, stage->aligned_weight_byte_count);
	PUT(own, stage->base_state_byte_count);
	PUT(own, stage->communication_byte_count);
	PUT(own, stage->draft_embedding_byte_count);
	PUT(own, stage->external_pool_sources);
	PUT(own, stage->external_key_sources);
	PUT(own, stage->external_selection_groups);
	PUT(own, stage->job_count);
	PUT(own, stage->channel_count);
	PUT(own, stage->channel_rows);
	PUT(own, stage->ring_rows);
	PUT(own, stage->external_draft_embedding);
	PUT(own, stage->vision_excluded);
	PUT(own, stage->sampler_entry);
	PUT(own, stage->load.description.job_count);
	PUT(own, stage->load.description.card_count);
	PUT(own, stage->load.description.card_arena_byte_count);
	PUT(own, stage->load.description.maximum_extent_byte_count);
	PUT(own, stage->load.source_workspace_byte_count);
	PUT(own, stage->load.output_workspace_byte_count);
	canonical_section(own, "stage", card);
	jobs_put(own, stage->jobs, stage->job_count);
	canonical_section(own, "stage_jobs", card);
	for (uint32_t index = 0;
	     stage->channels != NULL && index < stage->channel_count; index++) {
		const struct v41_tp2_stage_channel *const channel =
			&stage->channels[index];
		PUT(own, channel->collective.arguments);
		PUT(own, channel->collective.arena_byte_count);
		PUT(own, channel->layer);
		PUT(own, channel->site);
		PUT(own, channel->ring_offset);
		PUT(own, channel->ring_bank_bytes);
		PUT(own, channel->ring_row_bytes);
		PUT(own, channel->dense_ring_offset);
		PUT(own, channel->dense_ring_bank_bytes);
	}
	canonical_section(own, "stage_channels", card);
}

/* detail: a text program's BATCH verifiers, a bank program's bank */
static void program_dump(const char *label, enum program_kind kind,
			 uint32_t rows, uint32_t detail, result_status status,
			 const struct execution_program *program,
			 const struct step_companion_program *companions)
{
	if (is_dumped)
		(void)printf("program %s kind %u rows %" PRIu32
			     " detail %" PRIu32 ": %s\n",
			     label, (unsigned int)kind, rows, detail,
			     status_text(status));
	if (status != STATUS_OK || program == NULL)
		return;
	const uint32_t scalars[] = {
		program->card_count,	      program->segment_count,
		program->hand_off_count,      program->row_count,
		program->result_card,	      program->expert_pause_count,
		program->host_expert_count,   (uint32_t)program->kind,
		program->lease_count,	      program->session_range_count,
		program->front_segment_count, program->cone_rows,
		program->checkpoint_count,    program->draft_segment_first
	};
	section("scalars", DUMP_NO_CARD, scalars, sizeof(scalars));
	section("limits", DUMP_NO_CARD, &program->limits,
		sizeof(program->limits));
	section("leases", DUMP_NO_CARD, program->leases,
		(size_t)program->lease_count * sizeof(*program->leases));
	section("checkpoints", DUMP_NO_CARD, program->checkpoints,
		(size_t)program->checkpoint_count *
			sizeof(*program->checkpoints));
	section("segments", DUMP_NO_CARD, program->segments,
		(size_t)program->segment_count * sizeof(*program->segments));
	section("hand_offs", DUMP_NO_CARD, program->hand_offs,
		(size_t)program->hand_off_count * sizeof(*program->hand_offs));
	section("expert_pauses", DUMP_NO_CARD, program->expert_pauses,
		(size_t)program->expert_pause_count *
			sizeof(*program->expert_pauses));
	section("host_experts", DUMP_NO_CARD, program->host_expert_offsets,
		(size_t)program->host_expert_count *
			sizeof(*program->host_expert_offsets));
	section("session_ranges", DUMP_NO_CARD, program->session_ranges,
		(size_t)program->session_range_count *
			sizeof(*program->session_ranges));
	for (uint32_t card = 0; card < program->card_count; card++) {
		const struct program_card_record *const own =
			&program->cards[card];
		const uint64_t offsets[] = {
			own->arena_byte_count,	 own->step_offset,
			own->session_offset,	 own->result_offset,
			own->logits_offset,	 own->streams_offset,
			own->carried_pre_offset, own->engram_rows_offset,
			own->engram_tables,	 own->launch_count,
			own->view_count,	 own->round_offset
		};
		section("card", card, offsets, sizeof(offsets));
		section("views", card, own->views,
			(size_t)own->view_count * sizeof(*own->views));
		section("launches", card, own->launches,
			(size_t)own->launch_count * sizeof(*own->launches));
	}
	if (companions == NULL)
		return;
	const uint64_t companion_scalars[] = {
		companions->pair_count, companions->reserved_zero,
		companions->arithmetic_identity,
		companions->program == program ? 1u : 0u
	};
	section("companions", DUMP_NO_CARD, companion_scalars,
		sizeof(companion_scalars));
	section("pairs", DUMP_NO_CARD, companions->pairs,
		(size_t)companions->pair_count * sizeof(*companions->pairs));
}

static void kernels_dump(struct device_card *const *cards)
{
	for (uint32_t card = 0; is_dumped && card < card_count; card++) {
		struct test_host_memory_backend_kernel kernel;
		for (uint32_t handle = 0;
		     cards[card] != NULL &&
		     test_host_memory_backend_kernel_of(card, handle, &kernel);
		     handle++)
			(void)printf("kernel %" PRIu32 " %" PRIu32
				     " %s %s %" PRIu32 "\n",
				     card, handle, kernel.kernel_file_name,
				     kernel.function_name,
				     kernel.argument_byte_count);
	}
}

/* The workstation as the CUDA backend sees it: five cards, the PRO at DUMP_MACHINE_PRO, roomy memory */
MUST_CHECK static result_status machine_card_count(uint32_t *card_count_out)
{
	*card_count_out = DUMP_MACHINE_CARDS;
	return STATUS_OK;
}

MUST_CHECK static result_status
machine_card_describe(uint32_t ordinal,
		      struct card_description_record *description_out)
{
	if (ordinal >= DUMP_MACHINE_CARDS)
		return STATUS_INVALID_ARGUMENT;
	const bool pro = ordinal == DUMP_MACHINE_PRO;
	*description_out = (struct card_description_record){
		.arithmetic_class = pro ? ARITHMETIC_CLASS_SM_120 :
					  ARITHMETIC_CLASS_SM_80,
		.memory_total_bytes = DUMP_ROOMY_BYTES,
		.memory_free_bytes = DUMP_ROOMY_BYTES,
		.memory_reserved_bytes = BACKEND_CUDA_MEMORY_RESERVED_BYTES
	};
	(void)memcpy(description_out->name, pro ? pro_name : cmp_name,
		     pro ? sizeof(pro_name) : sizeof(cmp_name));
	return STATUS_OK;
}

/* The tower registered on the plan as model_vision_program_test registers it, behind the text arena of card 0; its
 * router biases are dummy ranges of another F32 tensor (no model extent is read). */
MUST_CHECK static result_status vision_register(void)
{
	const uint64_t text_arena = plan->cards[0].arena_byte_count;
	uint64_t arena_end = 0;
	plan->vision.enabled = true;
	const result_status status =
		model_vision_arena(&plan->vision, plan->limits.session_capacity,
				   text_arena, &arena_end);
	plan->cards[0].arena_byte_count = arena_end;
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		const uint64_t offset =
			plan->layer_tensors[layer][V41_DATA_ATTENTION_MIX]
				.stored.byte_offset;
		plan->vision.router_bias[layer] = (struct linear_range){
			(offset + VISION_ALIGNMENT - 1u) / VISION_ALIGNMENT *
				VISION_ALIGNMENT,
			MODEL_ROUTER_ROW_BYTES
		};
	}
	return status;
}

/* Relaxed CMP order's folds as load admits them (no scale is read): every E4M3-scaled trunk tensor at the reference
 * fold (model_v41_program_dump_test's), every layer's routed experts where expert_folds_prepare checks them */
static void folds_admit(void)
{
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++)
		for (uint32_t index = 0; index < V41_DATA_LAYER_TENSOR_COUNT;
		     index++)
			if (plan->layer_tensors[layer][index].weight_format ==
			    WEIGHT_FORMAT_E4M3_SCALED)
				plan->layer_tensors[layer][index]
					.fold_reference =
					RELAXED_FOLD_REFERENCE;
	for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++)
		plan->expert_folds_admitted[layer] =
			plan->exl3_bits[layer] == 0 &&
			plan->experts[layer].byte_count != 0 &&
			plan->host_expert_count[layer] == 0;
}

/* PP5's plan as run_program and plan_grown make it: the layout text (or the fill), the plan (with the tower behind
 * card 0's text arena where the pack has one, where plan_create puts it), the head offload. */
MUST_CHECK static result_status
pipeline_plan(const struct model_operations *model,
	      const struct model_limits *limits, bool tower)
{
	const uint32_t unit_count = pack_header_of(pack)->unit_count;
	uint64_t budgets[BACKEND_MAXIMUM_CARDS] = { 0 };
	for (uint32_t card = 0; card < card_count; card++)
		budgets[card] = DUMP_ROOMY_BYTES;
	uint32_t *const card_of_unit =
		calloc(unit_count, sizeof(*card_of_unit));
	uint32_t on_demand_units = 0;
	struct load_layout *layout = NULL;
	const char *const text =
		layout_text != NULL &&
				strncmp(layout_text, vision_statement,
					sizeof(vision_statement) - 1u) == 0 ?
			layout_text + sizeof(vision_statement) - 1u :
			layout_text;
	result_status status =
		card_of_unit == NULL ?
			STATUS_OUT_OF_MEMORY :
		text != NULL ?
			placement_from_layout_text(pack, text, budgets,
						   card_count, unit_count,
						   card_of_unit) :
			placement_fill_in_order(pack, budgets, card_count,
						unit_count, card_of_unit,
						&on_demand_units);
	if (status == STATUS_OK)
		status = load_layout_create(pack, card_of_unit, unit_count,
					    card_count, &layout);
	if (status == STATUS_OK)
		status = model->plan_create(pack, layout, limits, &plan);
	if (status == STATUS_OK && tower)
		status = vision_register();
	if (status == STATUS_OK && original_head_offload != 0)
		status = v41_original_head_plan(plan, descriptions);
	if (status == STATUS_OK &&
	    original_head_offload == V41_HEAD_MODE_BANKED)
		status = v41_original_head_banks_plan(plan, pipeline_steps);
	load_layout_destroy(layout);
	free(card_of_unit);
	return status;
}

struct prepared {
	struct execution_program *programs[DUMP_MOST_PROGRAMS];
	struct execution_program *images[DUMP_MOST_PROGRAMS];
	struct step_companion_program companions[DUMP_MOST_PROGRAMS];
	uint32_t count;
};

/* One text program as the serve's start prepares it (TP: THIN by the TP2 form, the others by the TP form); a must
 * program that is refused fails the case */
static void program_add(struct prepared *made,
			const struct model_operations *model,
			const struct v41_tp2_program_description *build,
			struct device_card *const *cards,
			const struct serve_program *own, bool must)
{
	if (made->count == DUMP_MOST_PROGRAMS) {
		failed("more programs than the dump holds",
		       STATUS_OUT_OF_MEMORY);
		return;
	}
	struct execution_program **const program = &made->programs[made->count];
	struct step_companion_program *const companions =
		&made->companions[made->count];
	struct v41_tp2_program_description described =
		build != NULL ? *build :
				(struct v41_tp2_program_description){ 0 };
	described.batch_verifiers = own->verifiers;
	const result_status status =
		build == NULL ?
			model->program_prepare(plan, &test_host_memory_backend,
					       cards, descriptions, own->kind,
					       own->rows, program) :
		own->kind == PROGRAM_THIN ?
			v41_tp2_program_prepare(&described,
						&test_host_memory_backend,
						cards, descriptions, own->rows,
						program, companions) :
			v41_tp_program_prepare(&described,
					       &test_host_memory_backend, cards,
					       descriptions, own->kind,
					       own->rows, program, companions);
	program_dump(own->rule == SERVE_PROGRAM_RULES ? "grid" : "text",
		     own->kind, own->rows, own->verifiers, status, *program,
		     build != NULL ? companions : NULL);
	made->count += status == STATUS_OK ? 1u : 0u;
	if (status != STATUS_OK && must)
		failed("a program every serve of its split prepares", status);
}

/* Whether the serve's list holds a program of kind and rows */
static bool listed(const struct serve_programs *chosen, enum program_kind kind,
		   uint32_t rows)
{
	for (uint32_t index = 0; index < chosen->count; index++)
		if (chosen->programs[index].kind == kind &&
		    chosen->programs[index].rows == rows)
			return true;
	return false;
}

/* The grid's programs the list does not hold: THIN of 1 to 8 rows, WIDE and FULL of chunk_rows, to the capacity */
static void grid_add(struct prepared *made,
		     const struct model_operations *model,
		     const struct v41_tp2_program_description *build,
		     struct device_card *const *cards,
		     const struct serve_programs *chosen, uint32_t capacity)
{
	for (uint32_t rows = 1; rows <= LINEAR_MAXIMUM_ROWS && rows <= capacity;
	     rows++)
		if (!listed(chosen, PROGRAM_THIN, rows))
			program_add(made, model, build, cards,
				    &(const struct serve_program){
					    PROGRAM_THIN, rows, 0,
					    SERVE_PROGRAM_RULES },
				    true);
	for (uint32_t kind = PROGRAM_WIDE; kind <= PROGRAM_FULL; kind++)
		for (uint32_t index = 0;
		     kind != PROGRAM_CONE && index < COUNT_OF(chunk_rows) &&
		     chunk_rows[index] <= capacity;
		     index++)
			if (!listed(chosen, (enum program_kind)kind,
				    chunk_rows[index]))
				program_add(
					made, model, build, cards,
					&(const struct serve_program){
						(enum program_kind)kind,
						chunk_rows[index], 0,
						SERVE_PROGRAM_RULES },
					chunk_rows[index] ==
							MODEL_WIDE_LEAST_ROWS ||
						chunk_rows[index] ==
							DUMP_SMOKE_CHUNK_ROWS);
}

static void banks_dump(const struct prepared *made,
		       struct device_card *const *cards, bool images)
{
	if (plan == NULL)
		return;
	struct v41_head_program_banks *banks = NULL;
	const result_status status = v41_original_head_banks_create(
		plan, &test_host_memory_backend, cards,
		(const struct execution_program *const *)made->programs,
		images ? (const struct execution_program *const *)made->images :
			 NULL,
		made->count, &banks);
	if (is_dumped)
		(void)printf("banks %" PRIu32 " programs %" PRIu32 ": %s\n",
			     plan->original_head.bank_count, made->count,
			     status_text(status));
	if (status != STATUS_OK) {
		failed("the head's flight banks", status);
		return;
	}
	const struct step_flight_programs *const tables =
		v41_original_head_bank_tables(banks);
	const struct visit_lease *const guard =
		v41_original_head_owner_guard(banks);
	section("owner_guard", DUMP_NO_CARD, guard,
		guard != NULL ? sizeof(*guard) : 0u);
	for (uint32_t bank = 0; bank < plan->original_head.bank_count; bank++) {
		for (uint32_t index = 0; index < made->count; index++) {
			const struct execution_program *const own =
				tables[bank].programs[index];
			const uint32_t borrowed =
				own == made->programs[index] ? 1u : 0u;
			section("borrowed", bank, &borrowed, sizeof(borrowed));
			program_dump("bank", own->kind, own->row_count, bank,
				     STATUS_OK, own, NULL);
			const struct execution_program *const image =
				tables[bank].image_programs != NULL ?
					tables[bank].image_programs[index] :
					NULL;
			if (image != NULL)
				program_dump("bank_image", image->kind,
					     image->row_count, bank, STATUS_OK,
					     image, NULL);
		}
	}
	v41_original_head_banks_destroy(banks);
}

/* The cards the serve takes (serve_cards_describe on the stand-in machine, serve_cards_arithmetic's classes), opened
 * in host memory of their classes */
MUST_CHECK static result_status cards_open(struct device_card **cards)
{
	result_status status = serve_cards_describe(&machine);
	if (status != STATUS_OK)
		return status;
	serve_cards_arithmetic();
	uint32_t classes[BACKEND_MAXIMUM_CARDS] = { 0 };
	for (uint32_t card = 0; card < card_count; card++)
		classes[card] = descriptions[card].arithmetic_class;
	section("card_order", DUMP_NO_CARD, card_order,
		(size_t)card_count * sizeof(card_order[0]));
	section("card_classes", DUMP_NO_CARD, classes,
		(size_t)card_count * sizeof(classes[0]));
	test_host_memory_backend_reset(card_count, DUMP_HOST_CARD_BYTES);
	for (uint32_t card = 0; status == STATUS_OK && card < card_count;
	     card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, classes[card] == ARITHMETIC_CLASS_SM_80 ?
				      ARITHMETIC_CLASS_SM_80 :
				      ARITHMETIC_CLASS_SM_120);
		status = test_host_memory_backend.card_open(
			card, DUMP_HOST_CARD_BYTES, &cards[card]);
	}
	return status;
}

/* The TP plans: the canonical, each card's view and stage (the stage's own plan, ranges, jobs and channels) and the
 * load jobs, in the canonical's card order */
static void tensor_dump(struct canonical *own)
{
	plan_dump("canonical", 0, tensor_model->canonical, own);
	const uint32_t cards = tensor_model->canonical->card_count;
	for (uint32_t card = 0; card < cards; card++)
		plan_dump("view", card, tensor_model->views[card], own);
	for (uint32_t card = 0; card < cards; card++)
		stage_dump(card, tensor_model->stages[card], own);
	uint32_t jobs = 0;
	for (uint32_t card = 0; card < cards; card++)
		jobs += tensor_model->stages[card]->job_count;
	jobs_put(own, tensor_model->jobs, jobs);
	canonical_section(own, "load_jobs", DUMP_NO_CARD);
}

/* The serve's program list, a line each */
static void serve_dump(const struct serve_programs *chosen, bool images)
{
	if (!is_dumped)
		return;
	(void)printf(
		"serve %" PRIu32 " programs, %" PRIu32
		" first, most rows %" PRIu32
		", batch %d, repeated rows %d, verifiers given %d, images %d, banks %d\n",
		chosen->count, chosen->first_count, chosen->most_rows,
		chosen->batch ? 1 : 0, chosen->rows_repeat ? 1 : 0,
		chosen->verifiers_given ? 1 : 0, images ? 1 : 0,
		original_head_offload == V41_HEAD_MODE_BANKED ? 1 : 0);
	for (uint32_t index = 0; index < chosen->count; index++)
		(void)printf("  serve kind %u rows %" PRIu32
			     " verifiers %" PRIu32 " rule %u\n",
			     (unsigned int)chosen->programs[index].kind,
			     chosen->programs[index].rows,
			     chosen->programs[index].verifiers,
			     (unsigned int)chosen->programs[index].rule);
}

/* One case, in the child: parse, cards, plan, the serve's programs (and the grid's), images and banks; true when every
 * step it takes succeeds. */
static bool case_run(const struct dump_case *own_case)
{
	static char words[DUMP_FIXED_WORDS + DUMP_CASE_MOST_WORDS]
			 [DUMP_WORD_BYTES];
	char *arguments[DUMP_FIXED_WORDS + DUMP_CASE_MOST_WORDS] = { NULL };
	const char *given[DUMP_FIXED_WORDS + DUMP_CASE_MOST_WORDS] = {
		[DUMP_PROGRAM_WORD] = "anemone", [DUMP_COMMAND_WORD] = "serve"
	};
	int count = DUMP_TABLES_WORD;
	for (uint32_t word = 0;
	     word < DUMP_CASE_MOST_WORDS && own_case->options[word] != NULL;
	     word++)
		given[count++] = own_case->options[word];
	given[count++] = "tables";
	given[count++] = "weights";
	for (int word = 0; word < count; word++) {
		(void)snprintf(words[word], sizeof(words[word]), "%s",
			       given[word]);
		arguments[word] = words[word];
	}
	struct axis_arguments_record expanded = { 0 };
	const struct test_model_v41_change change = {
		.kind = own_case->pack_kind
	};
	if (!axis_expand(count, arguments, &expanded) ||
	    !parse_arguments(expanded.count, expanded.values)) {
		failed("parse the case's serve", STATUS_INVALID_ARGUMENT);
		return false;
	}
	if (is_dumped) {
		(void)printf("case %s:", own_case->name);
		for (int word = 0; word < expanded.count; word++)
			(void)printf(" %s", expanded.values[word]);
		(void)putchar('\n');
	}
	if (session_slots_max)
		session_slots = DUMP_SESSIONS;
	result_status status = test_model_v41_draft_pack_open(change, &pack);
	if (status != STATUS_OK)
		failed("open the model's tables", status);
	struct device_card *cards[BACKEND_MAXIMUM_CARDS] = { NULL };
	if (status == STATUS_OK) {
		status = cards_open(cards);
		if (status != STATUS_OK)
			failed("describe and open the cards", status);
	}
	const bool tensor = execution_placement != EXECUTION_PP5;
	const struct model_operations *const model =
		model_operations_of(V41_DATA_MODEL_NAME);
	struct model_limits limits = options_limits();
	limits.session_capacity = session_slots;
	uint32_t *const unit_homes =
		status == STATUS_OK ? calloc(pack_header_of(pack)->unit_count,
					     sizeof(*unit_homes)) :
				      NULL;
	struct v41_tp2_program_description build = { 0 };
	if (status == STATUS_OK) {
		status = model == NULL || unit_homes == NULL ?
				 STATUS_OUT_OF_MEMORY :
			 tensor ? tensor_plan_prepare(
					  &limits,
					  (uint64_t)card_memory_kept_free_mib *
						  BYTES_PER_MIB,
					  unit_homes) :
				  pipeline_plan(model, &limits,
						!own_case->text_only);
		if (status != STATUS_OK)
			failed("plan the split", status);
	}
	if (status == STATUS_OK && !tensor &&
	    own_case->folds == DUMP_FOLDS_ADMITTED &&
	    plan->limits.cmp_order == CMP_ORDER_RELAXED)
		folds_admit();
	if (status == STATUS_OK && tensor) {
		status =
			v41_tp2_model_program_description(tensor_model, &build);
		if (status != STATUS_OK)
			failed("describe the TP programs", status);
	}
	struct canonical canon = { 0 };
	if (status == STATUS_OK && tensor)
		tensor_dump(&canon);
	else if (status == STATUS_OK)
		plan_dump("pipeline", 0, plan, &canon);
	free(canon.bytes);
	static struct prepared made;
	struct serve_programs chosen = { 0 };
	const bool images = status == STATUS_OK && !tensor &&
			    model_images_are_enabled(plan);
	if (status == STATUS_OK) {
		serve_programs_choose(tensor, &chosen);
		serve_dump(&chosen, images);
	}
	for (uint32_t index = 0; status == STATUS_OK && index < chosen.count;
	     index++)
		program_add(&made, model, tensor ? &build : NULL, cards,
			    &chosen.programs[index], true);
	if (status == STATUS_OK && is_dumped && own_case->grid)
		grid_add(&made, model, tensor ? &build : NULL, cards, &chosen,
			 limits.row_capacity);
	for (uint32_t index = 0; images && index < made.count; index++) {
		const struct execution_program *const text =
			made.programs[index];
		const result_status registered = v41_image_program_prepare(
			plan, &test_host_memory_backend, cards, descriptions,
			text->kind, text->row_count, &made.images[index]);
		program_dump("image", text->kind, text->row_count, 0,
			     registered, made.images[index], NULL);
		if (registered != STATUS_OK)
			failed("an image program", registered);
	}
	if (status == STATUS_OK && !tensor &&
	    original_head_offload == V41_HEAD_MODE_BANKED)
		banks_dump(&made, cards, images);
	kernels_dump(cards);
	for (uint32_t index = 0; index < made.count; index++) {
		if (model != NULL) {
			model->program_destroy(made.images[index]);
			model->program_destroy(made.programs[index]);
		}
	}
	if (tensor) {
		v41_tp2_model_destroy(tensor_model);
		tensor_model = NULL;
		plan = NULL;
	} else if (model != NULL) {
		model->plan_destroy(plan);
		plan = NULL;
	}
	for (uint32_t card = 0; card < card_count; card++)
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(
				cards[card], DUMP_WAIT_NANOSECONDS);
	free(unit_homes);
	pack_close(pack);
	pack = NULL;
	test_memory_storage_reset();
	axis_arguments_free(&expanded);
	return status == STATUS_OK && failures == 0;
}

/* The axis rows' cases: one a distinct (split, settings, serve settings, slot rule), named by its first row, in row
 * order; held: the rows that insert no settings */
static uint32_t axis_cases(struct dump_case *cases, uint32_t capacity,
			   uint32_t *rows_out, uint32_t *held_out)
{
	size_t rows = 0;
	const struct axis_choice *const choices = axis_choices_entry(&rows);
	uint32_t count = 0, held = 0;
	static char words[DUMP_MOST_CASES][AXIS_COUNT + 1u][DUMP_WORD_BYTES];
	for (size_t row = 0; row < rows; row++) {
		const struct axis_choice *const own = &choices[row];
		if (own->settings[0] == '\0') {
			held++;
			continue;
		}
		bool seen = false;
		for (size_t before = 0; before < row && !seen; before++) {
			const struct axis_choice *const other =
				&choices[before];
			seen = strcmp(other->gpu_split, own->gpu_split) == 0 &&
			       strcmp(other->settings, own->settings) == 0 &&
			       strcmp(other->serve_settings,
				      own->serve_settings) == 0 &&
			       other->serve_slots_exactly ==
				       own->serve_slots_exactly;
		}
		if (seen)
			continue;
		if (count == capacity) {
			failed("more axis cases than the dump holds",
			       STATUS_OUT_OF_MEMORY);
			break;
		}
		struct dump_case *const made = &cases[count];
		*made = (struct dump_case){
			.pack_kind = strcmp(own->weight_family, "exl3") == 0 ?
					     TEST_MODEL_V41_CHANGE_EXL3 :
					     TEST_MODEL_V41_CHANGE_NONE,
			.folds = DUMP_FOLDS_REFUSED
		};
		(void)snprintf(made->name, sizeof(made->name),
			       "axis_%s_%s_%s_%s_%s", own->gpu_split,
			       own->value[AXIS_CATEGORY_VALUE],
			       own->value[AXIS_EXACTNESS_VALUE],
			       own->value[AXIS_CONCURRENCY_VALUE],
			       own->value[AXIS_FOCUS_VALUE]);
		(void)snprintf(words[count][0], DUMP_WORD_BYTES,
			       "--placement=%s", own->gpu_split);
		made->options[0] = words[count][0];
		for (uint32_t axis = 0; axis < AXIS_COUNT; axis++) {
			(void)snprintf(words[count][axis + 1u], DUMP_WORD_BYTES,
				       "%s%s", axis_prefixes[axis],
				       own->value[axis]);
			made->options[axis + 1u] = words[count][axis + 1u];
		}
		if (own->serve_slots_exactly != 0)
			made->options[AXIS_COUNT + 1u] = DUMP_BATCH_SLOTS;
		count++;
	}
	*rows_out = (uint32_t)rows;
	*held_out = held;
	return count;
}

/* The child's whole output, read from its pipe */
struct output {
	unsigned char *bytes;
	size_t count, capacity;
};

static bool output_read(int descriptor, struct output *own)
{
	for (;;) {
		if (own->capacity - own->count < DUMP_READ_BYTES) {
			const size_t capacity =
				own->capacity * DUMP_GROWTH + DUMP_READ_BYTES;
			unsigned char *const grown =
				realloc(own->bytes, capacity);
			if (grown == NULL)
				return false;
			own->bytes = grown;
			own->capacity = capacity;
		}
		const ssize_t got = read(descriptor, own->bytes + own->count,
					 own->capacity - own->count);
		if (got == 0)
			return true;
		if (got < 0)
			return false;
		own->count += (size_t)got;
	}
}

/* A case in a child whose stdout is a pipe; the parent prints its output, or for programs an earlier case printed
 * the case's line and that case's name. */
static bool case_spawn(const struct dump_case *own_case,
		       unsigned char (*bodies)[SHA256_DIGEST_BYTES],
		       const char **names, uint32_t index)
{
	int ends[DUMP_PIPE_ENDS];
	(void)fflush(stdout);
	if (pipe(ends) != 0)
		return false;
	const pid_t child = fork();
	if (child == 0) {
		(void)close(ends[0]);
		if (dup2(ends[1], STDOUT_FILENO) < 0)
			_exit(EXIT_FAILURE);
		failures = 0;
		const bool passed = case_run(own_case);
		(void)fflush(stdout);
		_exit(passed ? EXIT_SUCCESS : EXIT_FAILURE);
	}
	(void)close(ends[1]);
	struct output output = { 0 };
	const bool read_all = child > 0 && output_read(ends[0], &output);
	(void)close(ends[0]);
	int status = 0;
	const bool exited = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) &&
			    WEXITSTATUS(status) == EXIT_SUCCESS && read_all;
	const unsigned char *const line_end =
		output.count != 0 ? memchr(output.bytes, '\n', output.count) :
				    NULL;
	const size_t line = line_end != NULL ?
				    (size_t)(line_end - output.bytes) + 1u :
				    output.count;
	const void *const body = output.count == line ?
					 (const void *)"" :
					 (const void *)(output.bytes + line);
	const bool digested = sha256_of(body, output.count - line,
					bodies[index]) == STATUS_OK;
	const bool passed = exited && digested;
	names[index] = own_case->name;
	uint32_t same = index;
	for (uint32_t before = 0;
	     passed && is_dumped && before < index && same == index; before++)
		if (memcmp(bodies[before], bodies[index],
			   SHA256_DIGEST_BYTES) == 0)
			same = before;
	const size_t printed = !is_dumped    ? 0u :
			       same != index ? line :
					       output.count;
	if (printed != 0 && fwrite(output.bytes, 1, printed, stdout) != printed)
		failures++;
	if (same != index)
		(void)printf("  programs as case %s\n", names[same]);
	free(output.bytes);
	return passed;
}

int main(int argument_count, char **arguments)
{
	const char *selected = NULL;
	bool arguments_valid = true;
	for (int argument = 1; argument < argument_count; argument++) {
		if (strcmp(arguments[argument], "--dump") == 0)
			is_dumped = true;
		else if (strcmp(arguments[argument], "--raw") == 0)
			is_raw = true;
		else if (strncmp(arguments[argument], DUMP_CASE_OPTION,
				 strlen(DUMP_CASE_OPTION)) == 0)
			selected =
				arguments[argument] + strlen(DUMP_CASE_OPTION);
		else
			arguments_valid = false;
	}
	machine = test_host_memory_backend;
	machine.card_count = machine_card_count;
	machine.card_describe = machine_card_describe;
	static struct dump_case cases[DUMP_MOST_CASES];
	uint32_t rows = 0, held = 0;
	const uint32_t axis_count = axis_cases(
		cases, DUMP_MOST_CASES - COUNT_OF(knob_cases), &rows, &held);
	uint32_t case_count = axis_count;
	for (uint32_t index = 0; index < COUNT_OF(knob_cases); index++)
		cases[case_count++] = knob_cases[index];
	static unsigned char bodies[DUMP_MOST_CASES][SHA256_DIGEST_BYTES];
	static const char *names[DUMP_MOST_CASES];
	uint32_t run = 0, distinct = 0;
	for (uint32_t index = 0; arguments_valid && index < case_count;
	     index++) {
		if (selected != NULL &&
		    strcmp(selected, cases[index].name) != 0)
			continue;
		if (!case_spawn(&cases[index], bodies, names, run)) {
			(void)fprintf(stderr, "FAILED: case %s\n",
				      cases[index].name);
			failures++;
		}
		bool repeated = false;
		for (uint32_t before = 0; before < run && !repeated; before++)
			repeated = memcmp(bodies[before], bodies[run],
					  SHA256_DIGEST_BYTES) == 0;
		distinct += repeated ? 0u : 1u;
		run++;
	}
	if (!arguments_valid || run == 0) {
		(void)fprintf(
			stderr,
			"usage: model_v41_split_program_dump_test [--dump] [--raw] [--case=NAME]\n");
		return EXIT_FAILURE;
	}
	if (is_dumped)
		(void)printf("axis rows %" PRIu32 ": %" PRIu32
			     " held, the others in %" PRIu32
			     " axis cases; %u knob cases; %" PRIu32
			     " cases run, %" PRIu32 " distinct\n",
			     rows, held, axis_count, COUNT_OF(knob_cases), run,
			     distinct);
	else
		(void)printf("model_v41_split_program_dump_test %s: %" PRIu32
			     " cases (%" PRIu32 " of the axis rows)\n",
			     failures == 0 ? "passed" : "FAILED", run,
			     axis_count);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
