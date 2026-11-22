/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_internal.h: what the cli units share (split from commands.c by whole items).
 */
#ifndef CLI_INTERNAL_H
#define CLI_INTERNAL_H

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include "models/deepseek_v41_flash/prompt/ds41f_chat_template.h"
#include "common/hooks.h"
#include "engine/storage/load/loader.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "models/deepseek_v41_flash/options/ds41f_splits.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/runtime/placement/placement.h"
#include "engine/storage/replicas/replicas.h"
#include "engine/engram/engram_disk.h"
#include "engine/engram/engram_fit.h"
#include "engine/storage/io/local_device.h"
#include "engine/storage/profile/storage_profile.h"
#include "engine/storage/autotune/storage_autotune.h"
#include "engine/server/server.h"
#include "engine/runtime/placement/session_fit.h"
#include "cli_stats_client.h"
#include "cli_policy.h"
#include "engine/storage/io/storage_exl3.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "cli_model_folder.h"
#include "models/model_registry.h"
#include "engine/runtime/scheduler/serving_cursor.h"
#include "cli_axes.h"
#include "engine/runtime/step/step.h"
#include "engine/runtime/step/step_images.h"
#include "engine/input/tokenizer/tokenizer.h"
#include "engine/runtime/speculation/draft_lookup.h"
#include "engine/formats/pack/tokenizer_tables_pack.h"
#include "common/json.h"
#include "engine/engram/engram_step.h"
#include "engine/engram/engram_hot.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_model.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_peer.h"
#include "commands.h"
#include "cli_golden_capture.h"

/* REUSE-IgnoreStart: the tag below is the printed notice's, not this file's */
#define PROGRAM_ATTRIBUTION_NOTICE                                                               \
	"Anemone v0.0.7-alpha\nCopyright (C) 2026 Fabiano Anemone <fabiano.anemone@gmail.com>\n" \
	"SPDX-License-Identifier: AGPL-3.0-or-later\n" /* REUSE-IgnoreEnd */
#define NANOSECONDS_PER_SECOND 1e9
#define DECIMAL_BASE 10
#define DEFAULT_TP_LOAD_THREADS \
	32u /* the TP load's workers for the cards (rh04, 4 Nov: 31.1 s against 33.9 at 16) */
#define DEFAULT_TP_LOAD_HOST_THREADS \
	TP_LOAD_HOST_THREADS_DEFAULT /* and the host arena's threads beside them */
#define DEFAULT_TP_LOAD_READS_PER_COPY TP_LOAD_READS_PER_COPY_DEFAULT
#define MAP_READS_AUTO \
	EXL3_READ_MODES /* --map_reads beyond the storage's modes: chosen by the copies found */
#define FRONT_TAIL_ALL \
	2u /* --prefill_front_tail=2: the front region's tail takes the full region's tail programs */
#define DEAD_ROWS_TRIMMED \
	2u /* --prefill_dead_halves: the sums' copies of the rows the cone reads alone too */
#define DEFAULT_LOOKUP_NGRAM_MOST \
	4u /* the lookup drafter's longest suffix, in tokens */
#define DEFAULT_LOOKUP_NGRAM_LEAST 2u
#define EXL3_EXPERT_OPTIONAL_PARTS 8u
#define ENGRAM_RAM_AUTO_OPTION "--engram_ram_percent=auto"
extern const char *pack_argument;
extern const char *exl3_map_name;
extern const char *placement_text;
extern bool is_serving;
extern bool is_capturing;
extern uint32_t tp_load_threads;
extern uint32_t tp_load_reads_per_copy;
extern uint32_t tp_load_host_reads_per_copy;
extern uint32_t tp_load_host_threads;
extern uint32_t tp_load_copy_split;
extern uint32_t load_piece_mib;
extern uint64_t tp_load_piece_bytes;
extern char storage_profile_resolved[REPLICA_PATH_CAPACITY];
extern uint32_t turbo_load;
extern uint32_t load_tuned_weights;
extern enum engram_ram_source engram_ram_source;
extern uint32_t engram_fit_only;
extern uint32_t replica_external;
extern uint32_t replica_floor_percent;
extern uint32_t engram_spin_us;
extern bool explicit_engram_threads;
extern bool explicit_engram_spin;
extern bool explicit_engram_pin;
extern uint32_t engram_scan_threads;
extern const char *storage_profile_path;
extern const char *engram_hot_set_path;
extern const char *engram_hot_sha256;
extern uint32_t engram_hot_mib;
extern struct engram_hot *engram_hot;
extern struct storage_profile tuned_storage;
extern char storage_replica_info[STORAGE_PROFILE_METRICS_BYTES];
extern size_t storage_replica_info_bytes;
extern bool tuned_engram_active;
extern bool storage_profile_is_valid;
extern bool storage_profile_is_off;
extern uint32_t tuned_load_weights[STORAGE_MAXIMUM_COPIES];
extern struct storage_reader *engram_source_reader;
extern uint32_t map_reads;
extern uint32_t tp_exact_dense;
extern uint32_t exl3_fast_products;
extern uint32_t exl3_wide_entry;
extern uint32_t exl3_tc_moe;
extern uint32_t exl3_tc_product_blocks;
extern uint32_t exl3_expert_rotation;
extern uint32_t exl3_unrated_map;
extern uint32_t tp_head_mma;
extern uint32_t tp_trunk_mma;
extern uint32_t tp_trunk_mma_fused;
extern uint32_t tp_trunk_mma_thin_splits;
extern uint32_t tp_router_mma;
extern uint32_t tp_gather_balance;
extern uint32_t tp_pool_chunk;
extern uint32_t tp_pool_gather;
extern uint32_t tp_router_first_whole;
extern uint32_t exl3_tcp_two_blocks;
extern uint32_t tp_queries_piece;
extern uint32_t tp_index_span;
extern uint32_t tp_core_compact_halves;
extern uint32_t dense_skinny_bf16;
extern uint32_t dense_skinny_f32;
extern uint32_t dense_staged;
extern uint32_t tp_parallel_launch;
extern uint32_t tp_graph_parts;
extern uint32_t tp_ordered_wide;
extern uint32_t tp_graph_branches;
extern uint32_t tp_piece_rings;
extern uint32_t tp_trunk_shape_slices;
extern uint32_t tp_bf16_slab_rings;
extern uint32_t tp_wide_exchange;
extern uint32_t tp_split_overlap;
extern uint32_t tp_split_least_rows;
extern uint32_t tp_launch_trims;
extern uint32_t tp_full_split_overlap;
extern uint32_t tp_chunk_core;
extern uint32_t tp_replicated_embedding;
extern uint32_t batch_draft_rows;
extern uint32_t batch_wide_head;
extern uint32_t batch_context_slabs;
extern uint32_t batch_thin_bits;
extern uint32_t batch_grouped_rows;
extern uint32_t batch_tc_paired;
extern uint32_t batch_draft_slots_head;
extern uint32_t batch_row_sets;
extern uint32_t tp_front_branch;
extern uint32_t tp_core_heads;
extern uint32_t tp_decode_round_once;
extern uint32_t draft_bf16_rows;
extern uint32_t batch_draft_wide;
extern uint32_t draft_bf16_operands;
extern uint32_t tp_ring_column_owners;
extern uint32_t exl3_expert_parts;
extern bool numerics_was_given;
extern uint32_t prefix_reuse_exact;
extern uint32_t cache_scope;
extern uint32_t anthropic_messages;
extern uint32_t session_fork;
extern uint32_t session_grace_seconds;
extern uint32_t session_save_tokens;
extern uint32_t session_load_tokens;
extern uint32_t session_idle_save_seconds;
extern uint32_t prefill_margin;
extern uint32_t prefill_fast_plan;
extern uint32_t prefill_order_wait;
extern uint32_t prefill_decode_every;
extern uint32_t prefill_chunk_rows;
extern uint32_t engram_shared;
extern uint32_t prefill_cone_skip;
extern uint32_t prefill_tail_rows;
extern uint32_t prefill_dead_halves;
extern uint32_t prefill_ladder_1024;
extern uint32_t prefill_ladder_fine;
extern uint32_t prefill_ladder_1536;
extern uint32_t prefill_front_tail;
extern uint32_t prefill_tail_overlay;
extern uint32_t prefill_tail_sets;
extern uint32_t prefill_thin_tail;
extern uint32_t draft_start_economic;
extern uint32_t generated_token_limit;
extern uint32_t lookup_ngram_max;
extern uint32_t lookup_ngram_min;
extern uint32_t hybrid_lookup_min;
extern uint32_t draft_trace;
extern uint32_t reasoning_effort;
extern const char *bind_address;
extern const char *key_file;
extern const char *model_alias;
extern const char *session_directory;
extern uint32_t port;
extern uint32_t queue_depth;
extern uint32_t user_queue_depth;
extern const char *draft_usage;
extern const char *calls_parsing;
extern const char *dump_draft_ring;
extern uint32_t live_stats_interval_ms;
extern const char *live_stats_format;
extern const char *stats_users;
extern char stats_card_order[LIVE_STATS_CARD_ORDER_BYTES];
extern char stats_placement[LIVE_STATS_PLACEMENT_BYTES];
extern struct load_layout *layout;
extern struct device_card *cards[BACKEND_MAXIMUM_CARDS];
extern struct load_result result;
extern struct model_instance instance;
extern struct execution_program *programs[MOST_PROGRAMS];
extern struct execution_program *image_programs[MOST_PROGRAMS];
extern struct ds41f_head_program_banks *head_program_banks;
extern uint32_t program_count;
extern struct step_driver *driver;
extern struct inference_engine *engine;
extern struct http_server *server;
MUST_CHECK result_status stats_metadata_bound(const uint32_t *card_of_unit);
double monotonic_seconds(void);
bool step_failed(const char *step_name, result_status status);
bool take_engram_ram_auto(const char *argument, bool *given);
bool take_model_source(const char *argument);
enum model_file folder_pack_file(void);
bool weights_placement_matches(bool placement_was_given);
void session_defaults(uint32_t sessions);
bool pack_holds_exl3(const struct pack_record *opened_pack);
MUST_CHECK result_status
pack_family_status(const struct pack_record *opened_pack);
bool tc_moe_selected(void);
MUST_CHECK result_status map_descriptor_check(const char *tables,
					      uint64_t table_byte_count);
bool tensor_placement_selected(void);
uint32_t tensor_cards_drain(void);
void load_section_apply(void);
void storage_profile_apply(int argument_count, char **arguments);
MUST_CHECK result_status storage_metrics_prepare(void);
MUST_CHECK result_status engram_hot_start(void);
MUST_CHECK result_status engram_ram_fit(void);
void engram_steps_report(void);
MUST_CHECK result_status engram_sources_prepare(void);
MUST_CHECK bool
engine_failed_to_start(const struct engine_description_record *description);
MUST_CHECK result_status aligned_digest(uint32_t card,
					struct linear_range range,
					uint64_t *digest_out);
MUST_CHECK bool prompt_failed_to_generate(const uint32_t *token_ids,
					  uint32_t token_count);
MUST_CHECK bool capture_failed(int argument_count, char **arguments);
MUST_CHECK bool requests_failed(const struct text_tokenizer *tokenizer);
void scheduling_usage(FILE *output);
void tp_queue_depth_report(void);
void sessions_max_report(const struct program_limits *limits,
			 const uint64_t *budget_bytes_of_card,
			 const char *kept);
bool sessions_max_refused(const struct program_limits *limits);

#endif
