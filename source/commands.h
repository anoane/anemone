/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* What commands.c shares with main.c (run_program) and with the tests that drive its parse, its start-up and its
 * plan's fit without a model or a card: the options as parsed (each one's meaning and default stay at its definition
 * in commands.c) and the steps that read them. An includer includes it after its other headers: some of their inline
 * functions name a parameter as one of these globals, which would shadow it (-Wshadow). */
#ifndef COMMANDS_H
#define COMMANDS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/draft_policy.h"
#include "anemone/front_end_seam.h"
#include "anemone/linear.h"
#include "anemone/model_seam.h"
#include "anemone/replicas.h"
#include "anemone/startup_policy.h"
#include "anemone/status.h"
#include "model_folder.h"
#include "startup_axes.h"

#define BYTES_PER_MIB 1048576ull
#define ARGUMENTS_BEFORE_THE_OPTIONS 2 /* the program's name and the command */
#define CARD_WAIT_LIMIT_NANOSECONDS \
	10000000000ull /* ten seconds: for what is in flight, and for a step */
#define FRONT_END_FILES \
	2 /* of generate: the messages, the tokenizer's tables */
#define MOST_BATCH_PROGRAMS \
	8u /* --batch_rows: BATCH programs in place of the thin ones (model_seam.h) */
#define ORIGINAL_PP5_MULTI_PIPELINE_STEPS 5u
/* A TP serve's depth with several slots and the device draft (the TP bases' 2): one flight is refused there without
 * BATCH, so a serve that leaves its depth (and BATCH) unset takes this. */
#define TP_MULTI_PIPELINE_STEPS 2u
/* serve's --session_slots=max (owner, 8 Nov), serve's default since 9 Nov ("as many as 1M context can fit"): the most
 * sessions of the context that fit the cards' memory, no cap, resolved at the placement step; parsing, its defaults and
 * admission meanwhile see SESSION_SLOTS_MAX_PARSED sessions (several), and a max that resolves to one session takes
 * the defaults of one (session_defaults) */
#define SESSION_SLOTS_MAX_OPTION "--session_slots=max"
#define SESSION_SLOTS_MAX_PARSED 8u
#define PREFILL_DECODE_DEFAULT_ROWS 1024u

enum program_status {
	PROGRAM_SUCCEEDED = 0,
	PROGRAM_FAILED = 1,
	PROGRAM_WRONG_USAGE = 2
};
enum speculative_mode {
	SPECULATIVE_OFF,
	SPECULATIVE_LOOKUP,
	SPECULATIVE_DEVICE,
	SPECULATIVE_HYBRID
};
enum front_end_file { MESSAGES_FILE, TABLES_FILE };
enum weights_family { WEIGHTS_UNSET, WEIGHTS_ORIGINAL, WEIGHTS_EXL3 };
/* What a session count's fit is tried on: the pipeline's cards (session_fit), or the TP plan's budgets. */
struct slot_fit {
	const struct session_fit_cards *cards;
	uint64_t kept_free_bytes;
	uint32_t *unit_homes;
};

/* Measured TP4 BF16 programs, sorted by rows then verifiers. Other layouts need explicit prices. */
static const uint32_t default_batch_rows[MOST_BATCH_PROGRAMS] = { 8,  8,  12,
								  12, 16, 24,
								  32, 48 };
static const uint32_t default_batch_verifiers[MOST_BATCH_PROGRAMS] = { 1, 2, 2,
								       4, 4, 4,
								       8, 8 };
static const uint32_t default_batch_step_us[MOST_BATCH_PROGRAMS] = {
	19500, 21500, 28600, 30000, 32000, 39000, 46000, 58000
};
/* the bounded defaults, names rather than macros: each equals its unbounded fallback, which the conditional keeps */
static const unsigned int prefill_decode_default_rounds = 1u;
static const unsigned int prefill_decode_default_waiters = 0u;

/* The options and the state as parsed and resolved, in commands.c's order. */
extern const char *const *copy_paths;
extern const char *front_end_paths[FRONT_END_FILES];
extern uint32_t copy_count;
extern enum weights_family weights_family;
extern const char *model_dir;
extern struct model_file_record model_files[MODEL_FILES];
extern const struct model_file_record *model_pack;
extern bool is_generating;
extern uint32_t card_order[BACKEND_MAXIMUM_CARDS];
extern uint32_t card_count;
extern uint32_t reader_threads_per_copy;
extern uint32_t engram_ram_percent;
extern uint32_t engram_cache_mib;
extern uint32_t engram_io;
extern uint32_t turbo_nvme_engrams;
extern uint32_t engram_step_io;
extern bool engram_step_io_default;
extern uint32_t engram_io_threads;
extern uint32_t engram_io_pin;
extern struct engram_steps *engram_steps;
extern struct engram_disk *engram_disk;
extern const char *engram_sources_text;
extern uint32_t card_memory_kept_free_mib;
extern bool pipeline_steps_defaulted;
extern uint32_t session_slots, step_graphs, pipeline_steps, prefill_cone;
extern bool session_slots_max;
extern uint32_t tp_pipeline_overlap;
extern uint32_t pro_emulation;
extern enum cmp_order_code cmp_order;
extern enum numerics_code numerics;
extern uint32_t tp_code_wire;
extern uint32_t tp_attention_owner;
extern uint32_t tp_head_shards;
extern uint32_t tp_draft_shards;
extern uint32_t tp_draft_layers;
extern uint32_t tp_head_stream;
extern uint32_t original_dense_staged;
extern uint32_t original_head_offload;
extern uint32_t original_wide_experts;
extern uint32_t original_chunk_core;
extern uint32_t batch_rows[MOST_BATCH_PROGRAMS];
extern uint32_t original_batch;
extern uint32_t batch_core_once;
extern uint32_t batch_step_us[SERVING_BATCH_COSTS];
extern uint32_t batch_verifiers[MOST_BATCH_PROGRAMS];
extern enum execution_placement_code execution_placement;
extern uint32_t prefix_reuse;
extern uint32_t full_steps;
extern uint32_t chunks_in_flight;
extern uint32_t prefill_prefix_stable;
extern uint32_t prefill_order;
extern uint32_t prefill_decode_rows;
extern uint32_t prefill_decode_rounds;
extern uint32_t prefill_decode_waiters;
extern uint32_t prefill_first;
extern uint32_t prefill_prompt_chunks;
extern uint32_t draft_learned_acceptance;
extern uint32_t draft_session_acceptance;
extern uint32_t batch_plan_ties;
extern uint32_t wide_rows;
extern uint32_t step_rows;
extern uint32_t draft_block;
extern uint32_t speculative, draft_positions;
extern bool is_draft_given;
extern struct draft_policy_record draft_policy;
extern const char *prompt_token_ids_path;
extern uint32_t ignore_eos;
extern uint32_t context_tokens;
extern uint32_t seed, linear_profile, linear_order;
extern uint32_t official_hc_rows;
extern uint32_t official_reference;
extern bool is_seed_given;
extern float temperature, top_p;
extern const char *layout_text;
extern uint32_t metrics;
extern struct storage_reader *reader;
extern const struct storage_operations_record *storage_operations;
extern struct pack_record *pack;
extern struct card_description_record descriptions[BACKEND_MAXIMUM_CARDS];
extern struct model_plan_record *plan;
extern struct v41_tp2_model *tensor_model;
extern uint32_t tp_wire_bf16;
extern struct engram_record *engram;

/* The steps the tests drive, then the command main.c runs. */
bool take_option(const char *argument, const char *name, uint32_t least,
		 uint32_t most, uint32_t most_numbers, bool repeats,
		 uint32_t *numbers_out, uint32_t *number_count_out);
bool take_fraction(const char *argument, const char *name, float least,
		   float most, float *value_out, uint32_t *given_count);
MUST_CHECK result_status model_folder_files(bool reads_tables);
bool prefix_cold_plan_is_qualified(void);
bool parse_arguments(int argument_count, char **arguments);
MUST_CHECK result_status
model_pack_opened(const struct storage_operations_record *storage,
		  const struct storage_reader *opened_reader,
		  const struct pack_header *header);
const struct storage_operations_record *pack_storage(void);
char *file_read(const char *path, const struct model_file_record *expected,
		uint64_t *byte_count_out);
MUST_CHECK result_status prompt_token_ids_parse(const char *text,
						size_t byte_count,
						uint32_t vocabulary_size,
						uint32_t *token_ids_out,
						uint32_t token_capacity,
						uint32_t *token_count_out);
uint32_t effective_row_capacity(void);
MUST_CHECK result_status tensor_plan_prepare(const struct model_limits *limits,
					     uint64_t kept_free_bytes,
					     uint32_t *unit_homes);
struct model_limits options_limits(void);
bool batch_serving(void);
uint32_t slots_most(const struct slot_fit *fit,
		    const struct model_limits *limits,
		    result_status *refusal_out);
MUST_CHECK result_status tensor_plan_fit(struct model_limits *limits,
					 uint64_t kept_free_bytes,
					 uint32_t *unit_homes);
MUST_CHECK result_status replicas_found_add(
	const char (*found_paths)[REPLICA_PATH_CAPACITY], uint32_t found);
MUST_CHECK bool engram_settings_resolve(void);
MUST_CHECK result_status engram_steps_start(void);
MUST_CHECK result_status engram_steps_start_or_fall_back(void);
int startup_placement_status(void);
const char *single_session_defaults(const struct slot_fit *fit,
				    const struct model_limits *limits);
void sessions_fit_report(uint32_t most, const struct model_limits *limits,
			 const struct axis_choice *slots_unmet);
bool sessions_refused(const struct slot_fit *fit,
		      const struct model_limits *limits,
		      const uint64_t *budget_bytes_of_card);
int run_program(int argument_count, char **arguments,
		const struct axis_arguments_record *axis_arguments);

#endif
