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

#include "engine/backend/backend_seam.h"
#include "engine/runtime/speculation/draft_policy.h"
#include "engine/runtime/scheduler/front_end_seam.h"
#include "engine/library/block_scaled_linear/linear.h"
#include "engine/program/model_seam.h"
#include "engine/storage/replicas/replicas.h"
#include "cli_policy.h"
#include "common/status.h"
#include "cli_model_folder.h"
#include "cli_axes.h"

#define BYTES_PER_MIB 1048576ull
#define ARGUMENTS_BEFORE_THE_OPTIONS 2 /* the program's name and the command */
#define CARD_WAIT_LIMIT_NANOSECONDS \
	10000000000ull /* ten seconds: for what is in flight, and for a step */
#define FRONT_END_FILES \
	2 /* of generate: the messages, the tokenizer's tables */
#define MOST_BATCH_PROGRAMS \
	8u /* --batch_rows: BATCH programs in place of the thin ones (model_seam.h) */
#define DS41F_ORIGINAL_PP5_MULTI_PIPELINE_STEPS 5u
/* A TP serve's depth with several slots and the device draft (the TP bases' 2): one flight is refused there without
 * BATCH, so a serve that leaves its depth (and BATCH) unset takes this. */
#define DS41F_TP_MULTI_PIPELINE_STEPS 2u
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
extern const uint32_t ds41f_default_batch_rows[MOST_BATCH_PROGRAMS];
extern const uint32_t ds41f_default_batch_verifiers[MOST_BATCH_PROGRAMS];
extern const uint32_t ds41f_default_batch_step_us[MOST_BATCH_PROGRAMS];
/* the bounded defaults, names rather than macros: each equals its unbounded fallback, which the conditional keeps */
static const unsigned int prefill_decode_default_rounds = 1u;
static const unsigned int prefill_decode_default_waiters = 0u;

/* The options and the state as parsed and resolved, in commands.c's order. */
extern const char *const *copy_paths;
extern const char *front_end_paths[FRONT_END_FILES];
extern uint32_t copy_count;
extern enum weights_family weights_family;
extern const char *model_dir;
extern struct model_file_record model_files[CLI_MODEL_FILES];
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
extern enum gpu_split_code execution_placement;
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
extern struct ds41f_plan *plan;
extern struct ds41f_tp2_model *tensor_model;
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
MUST_CHECK result_status
ds41f_tensor_plan_prepare(const struct program_limits *limits,
			  uint64_t kept_free_bytes, uint32_t *unit_homes);
struct program_limits options_limits(void);
bool batch_serving(void);
uint32_t slots_most(const struct slot_fit *fit,
		    const struct program_limits *limits,
		    result_status *refusal_out);
MUST_CHECK result_status tensor_plan_fit(struct program_limits *limits,
					 uint64_t kept_free_bytes,
					 uint32_t *unit_homes);
MUST_CHECK result_status replicas_found_add(
	const char (*found_paths)[REPLICA_PATH_CAPACITY], uint32_t found);
/* Whether turbo load's search runs beside the copies serve was given (one or more since 17 Nov 2026): unless
 * --turbo_load=0, a storage profile's tuned load chose the copies, or they are already the most a load reads. */
bool turbo_load_searches(void);
/* A storage profile's tuned load chose the copies (storage_profile_apply). */
extern bool tuned_load_active;
/* Turbo load's search: replicas_find_policy (replicas.h), or a test's stand-in. */
extern uint32_t (*turbo_load_search)(const char *const *copies,
				     uint32_t copy_count,
				     const struct pack_header *header,
				     char (*paths)[REPLICA_PATH_CAPACITY],
				     uint32_t capacity, bool external);
/* Serve's copies before its load: turbo load's search beside the given copies when turbo_load_searches() (with its
 * "find the replicas" step line), then, for a load given more than one copy, the one `turbo load: the load reads N
 * copies:` line naming each copy and whether it was given or found by the search (none for a tuned load: the profile's
 * copies, which its storage load: line names). False when the search's step failed (its line names the failure). */
MUST_CHECK bool turbo_load_copies(const struct pack_header *header);
MUST_CHECK bool engram_settings_resolve(void);
MUST_CHECK result_status engram_steps_start(void);
MUST_CHECK result_status engram_steps_start_or_fall_back(void);
int startup_placement_status(void);
const char *single_session_defaults(const struct slot_fit *fit,
				    const struct program_limits *limits);
void sessions_fit_report(uint32_t most, const struct program_limits *limits,
			 const struct axis_choice *slots_unmet);
bool sessions_refused(const struct slot_fit *fit,
		      const struct program_limits *limits,
		      const uint64_t *budget_bytes_of_card);
int run_program(int argument_count, char **arguments,
		const struct axis_arguments_record *axis_arguments);

/* The programs a serve's start prepares, in that order (the step driver's), each with the rule that chose it;
 * serve_programs_choose reads the parsed options alone (tensor: a TP split), so a test lists what a serve prepares. The
 * image programs mirror the list where model_images_are_enabled, and --original_head_offload=2's flight banks hold
 * it. */
#define MOST_THIN_PROGRAMS \
	3 /* four-owner capacity4: programs1/2/4; single-session retains1/max */
#define MOST_CHUNK_PROGRAMS \
	4 /* of each kind: 1,024, 256 and 64 rows; 2,048, 512, 128 and 64 */
#define EXTRA_CHUNK_PROGRAMS \
	26u /* --prefill_ladder_1024's two, --prefill_ladder_1536's two, --prefill_ladder_fine's
	       six, --prefill_front_tail's three, the two long tails' and --prefill_thin_tail's six
	       and twelve past the slack */
#define MOST_PROGRAMS                                                         \
	(MOST_THIN_PROGRAMS + MOST_BATCH_PROGRAMS + 2 * MOST_CHUNK_PROGRAMS + \
	 EXTRA_CHUNK_PROGRAMS) /* and wide and full ones, the knobs' */
enum serve_program_rule {
	SERVE_BATCH_ROWS, /* --batch_rows on a TP split, in place of the thin ones */
	SERVE_THIN_ROWS, /* 1, 2 and the most rows a step */
	SERVE_CHUNK_LADDER, /* wide_rows down to the least wide rows and the ladder knobs' sizes: WIDE, FULL too
			       with --full_steps */
	SERVE_TAIL, /* a TP split's --prefill_tail_rows and --prefill_front_tail */
	SERVE_THIN_TAIL, /* a TP split's --prefill_thin_tail */
	SERVE_PROGRAM_RULES
};
struct serve_program {
	enum program_kind kind;
	uint32_t rows;
	uint32_t verifiers; /* a BATCH program's, else 0 */
	enum serve_program_rule rule;
};
struct serve_programs {
	struct serve_program programs[MOST_PROGRAMS];
	uint32_t count;
	uint32_t first_count; /* the BATCH or THIN ones, first */
	uint32_t most_rows; /* the engine's most rows a step: the THIN ones' or the BATCH ones' */
	bool batch;
	bool rows_repeat; /* a BATCH row count given twice: it needs --batch_verifiers */
	bool verifiers_given; /* --batch_verifiers, each BATCH program's not 0 */
};
void serve_programs_choose(bool tensor, struct serve_programs *chosen);
bool model_images_are_enabled(const struct ds41f_plan *model_plan);
/* The cards a serve takes, numbered and described by backend, and their arithmetic (run_program's card step) */
MUST_CHECK result_status
serve_cards_describe(const struct backend_operations *backend);
void serve_cards_arithmetic(void);
/* What the serve start prepares programs on: the model and its plan (a TP split's canonical), a backend's cards and
 * their descriptions, and a TP split's program description (NULL for PP5). engine_failed_to_start's is the run's;
 * ds41f_split_program_dump_test's its stand-in cards, so the dump takes the serve's own dispatch. */
struct step_companion_program;
struct ds41f_tp2_program_description;
struct ds41f_head_program_banks;
struct serve_program_target {
	const struct model_program_operations *model;
	const struct ds41f_plan *plan;
	const struct backend_operations *backend;
	struct device_card *const *cards;
	const struct card_description_record *descriptions;
	struct ds41f_tp2_program_description *tensor_description;
};
/* A program of the serve's list (or of the kind and rows own names), as the serve's start prepares it: on a TP split
 * THIN by the TP2 composition and the others by the TP form, with own's verifiers and the companions; else the
 * model's */
MUST_CHECK result_status serve_program_prepare(
	const struct serve_program_target *target,
	const struct serve_program *own, struct execution_program **program_out,
	struct step_companion_program *companions_out);
MUST_CHECK result_status
serve_image_program_prepare(const struct serve_program_target *target,
			    const struct execution_program *text,
			    struct execution_program **image_out);
MUST_CHECK result_status serve_head_banks_create(
	const struct serve_program_target *target,
	const struct execution_program *const *texts,
	const struct execution_program *const *images, uint32_t count,
	struct ds41f_head_program_banks **banks_out);
MUST_CHECK result_status plan_grown(struct ds41f_plan *grown);

#endif
