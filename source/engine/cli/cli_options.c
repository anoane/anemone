/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_options.c: the generic option state, the option takers and parse_arguments with its tables (split from
 * commands.c by whole items).
 */
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

#include "cli_internal.h"

#define MOST_CARD_MEMORY_KEPT_FREE_MIB \
	1048576 /* a TiB: above any card, and its bytes fit in 64 bits */
#define FRACTION_OPTIONS 2 /* --temperature and --top_p */
#define TEXT_OPTIONS \
	13 /* serving text, the hot set, storage profile/sources, generate's ring dump and layout */
#define LAYOUT_TEXT (TEXT_OPTIONS - 1u) /* the one text every command takes */
#define ENGRAM_HOT_SET_TEXT \
	(TEXT_OPTIONS - 6u) /* then its SHA-256, then the sources */
#define ENGRAM_SOURCES_TEXT (TEXT_OPTIONS - 4u)
#define STORAGE_PROFILE_TEXT (TEXT_OPTIONS - 3u)
#define DUMP_TEXT \
	(TEXT_OPTIONS - 2u) /* generate's diagnostic dump of the draft rings */
#define DEFAULT_BATCH_GROUPED_ROWS 16u /* --batch_grouped_rows */
#define DEFAULT_BATCH_CORE_ONCE \
	3u /* step-wide pools, then deferred compressor slot retention */
#define DEFAULT_OTHER_BATCH_CORE_ONCE 2u
#define LEAST_MEASURED_BATCH_CORE_ONCE \
	2u /* measured menu: step-wide window and core, with optional pools */
#define DEFAULT_BATCH_DRAFT_WIDE \
	3u /* target-independent draft walk over complete verifier blocks */
#define MOST_BATCH_GROUPED_ROWS 48u
#define DEFAULT_GENERATED 64u /* tokens */
#define TAIL_LEAST_WIDE_ROWS 1024u
#define THIN_TAIL_AUTO \
	UINT32_MAX /* --prefill_thin_tail not given: 1 with the tail program and no BATCH (a BATCH step
				    takes the rows short of a 64-row chunk with the row of logits in one; EVIDENCE) */
#define OVERLAP_LEAST_PIPELINE_STEPS \
	2u /* --tp_pipeline_overlap: a step beside the next */
#define LADDER_FINE_AUTO \
	UINT32_MAX /* --prefill_ladder_fine not given: 1 with TAIL_LONG_WIDE_ROWS or more, no BATCH */
#define TAIL_ROWS_AUTO \
	UINT32_MAX /* --prefill_tail_rows not given: DS41F_TAIL_PROGRAM_ROWS where the tail program applies,
				    DS41F_TAIL_WIDEST_ROWS (and so DS41F_TAIL_LONG_ROWS too) where also --wide_rows is
				    TAIL_LONG_WIDE_ROWS or more and no BATCH, DS41F_TAIL_LONG_ROWS with BATCH (gsk03:
				    with the tail overlay the 640-column cards keep 341 MiB free, cold prefill +2.3%) */
#define TAIL_LONG_WIDE_ROWS \
	2048u /* --prefill_tail_rows=320's default (ri03), 448's (gsk01: identical, the 2K cell
				     1,696 -> 1,722 tok/s, cold prefill 3,798 -> 3,818) with 2,048-row chunks */
#define FRONT_TAIL_AUTO \
	UINT32_MAX /* --prefill_front_tail not given: 1 where the tail program applies, FRONT_TAIL_ALL
				      with the 448-row tail (THIN; rk04k: identical, the 30K prompt 8.214 -> 8.190 s) */
#define DEFAULT_CHUNKS_IN_FLIGHT \
	1u /* 1: scheduler.c's exact plan (the 1,024-row two-in-flight reference's bits in the largest chunks;
	      rc02ab 30K prompt 11.859 -> 9.430 s, Fraser 2.431 -> 1.857 s, identical); else so many chunks at least */
#define DEFAULT_PREFILL_MARGIN \
	127u /* the exact plan's full rows beyond a prompt's cone (scheduler.c; rc02ab: the same bits, -1.0 s at 30K) */
#define DRAFT_LOOKUP_OPTION "--draft=lookup"
#define DRAFT_DEVICE_OPTION "--draft=dspark"
#define DRAFT_HYBRID_OPTION "--draft=hybrid"
#define DEFAULT_CODE_DRAFT_BASE_US 14300u
#define DEFAULT_CODE_DRAFT_ROW_US 690u
#define DRAFT_CALIBRATION_WORD_BYTES 8u
#define DRAFT_CALIBRATION_FILE_BYTES 4096u
#define DRAFT_CALIBRATION_SPACES " \t\r\n\v\f"
#define EXL3_EXPERT_REFERENCE_PARTS 4u

enum command_kind {
	TAKEN_BY_LOAD = 1,
	TAKEN_BY_GENERATE = 2,
	TAKEN_BY_SERVE = 4,
	TAKEN_BY_ALL = 7
};

const char *const *copy_paths; /* the copies of the pack, as typed */
const char *
	pack_argument; /* the first pack argument as typed (its descriptor's place), or NULL */
const char *front_end_paths
	[FRONT_END_FILES]; /* by enum front_end_file, as typed; NULL: none */
uint32_t copy_count;
/* --weights: unset keeps the family that --placement and the pack imply; --model_dir: the folder whose fixed names
 * (cli_model_folder.h) give the tables and the pack, each with its real path and its device and inode */
enum weights_family weights_family;
const char *model_dir;
const char *
	exl3_map_name; /* --exl3_map=NAME: the folder's EXL3 map is NAME.exl3map; NULL: model.exl3map */
struct model_file_record model_files[CLI_MODEL_FILES];
const struct model_file_record *model_pack; /* the folder's pack, or NULL */
const char *placement_text; /* as typed */
bool is_generating; /* generate or serve */
bool is_serving; /* serve */
bool is_capturing; /* capture: generate's path, cli_golden_capture.h's front end */
uint32_t card_order
	[BACKEND_MAXIMUM_CARDS]; /* backend cards, in the order the stages fill them */
uint32_t card_count; /* 0 until the cards are counted: all, as enumerated */
uint32_t reader_threads_per_copy = LOADER_DEFAULT_READER_THREADS_PER_COPY;
/* the TP placements' load: 0 the serial loads (the reference), N the host arena beside the cards, each on N threads */
uint32_t tp_load_threads = DEFAULT_TP_LOAD_THREADS;
/* with tp_load_threads: reads in flight a copy at most (0: no limit) */
uint32_t tp_load_reads_per_copy = DEFAULT_TP_LOAD_READS_PER_COPY;
/* with tp_load_threads: the host arena's own reads a copy at most, from an applied load section (0: no limit) */
uint32_t tp_load_host_reads_per_copy = DEFAULT_TP_LOAD_READS_PER_COPY;
/* with tp_load_threads: the host arena's threads (0: as many as the cards' load) */
uint32_t tp_load_host_threads = DEFAULT_TP_LOAD_HOST_THREADS;
/* with tp_load_threads and copies (tp_load.h, enum tp_load_split): 2 (default) host-arena chunks prefer copy 0 and
 * card extents the last copy, and a role that has claimed its last read leaves its copy to the other; 1: each role
 * only its preferred copy (tcb96's default); 0: each read from the copy soonest to finish it */
uint32_t tp_load_copy_split = TP_LOAD_SPLIT_STEAL;
/* the load's read size in MiB, both loaders' (the PP loader's piece and staging slot, the TP host arena's chunk); 0
 * (default): LOADER_CHUNK_BYTES and TP_HOST_LOAD_CHUNK_BYTES, 16 MiB. The knob wins over a profile, a profile over
 * the default. */
uint32_t load_piece_mib;
/* The TP host arena's read (load_section_apply): a knob given wins, then the storage profile's enabled load section
 * when it names this pack and exactly these copies, then the default; tp_load_reads_per_copy holds the readers. */
uint64_t tp_load_piece_bytes;
char storage_profile_resolved[REPLICA_PATH_CAPACITY]; /* empty: none, or off */
/* 1 (default): a single source gains its receipted replicas on other drives as copies (replicas.h); 0: it alone */
uint32_t turbo_load = 1;
/* 1: a storage profile's per-copy load weights pick each piece's copy (a static split, for comparison); 0: the
 * readers of every copy steal the next piece, so a faster drive reads more */
uint32_t load_tuned_weights;
/* the tables' RAM share: given (the knob, or a storage profile's), else auto, which the fit (engram_fit.h) decides
 * before the load; it reads ENGRAM_RAM_PERCENT until then */
uint32_t engram_ram_percent = ENGRAM_RAM_PERCENT;
enum engram_ram_source engram_ram_source = ENGRAM_RAM_AUTO;
/* 1: generate prints the Engram fit's lines and stops before the first card is touched (a diagnostic) */
uint32_t engram_fit_only;
uint32_t replica_external,
	replica_floor_percent = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT;
uint32_t engram_cache_mib;
uint32_t engram_io;
uint32_t turbo_nvme_engrams;
/* step I/O (engram_step.h), the default serving PP5 or TP4_PRO from disk tables: one deduplicated read of a step's
 * blocks at submit, on I/O threads per copy that verify and scatter rows; threads per copy, their busy-poll window,
 * and pinning to the drive's completion CPU (its first I/O queue's vector and that CPU's SMT sibling) */
uint32_t engram_step_io;
/* engram_step_io not given: it is its default (1 serving PP5 or TP4_PRO from the disk store, else 0) */
bool engram_step_io_default;
uint32_t engram_io_threads = ENGRAM_STEP_IO_THREADS_DEFAULT;
uint32_t engram_spin_us = ENGRAM_STEP_SPIN_US_DEFAULT;
uint32_t engram_io_pin = 1;
/* Step I/O's parameters given as knobs (each wins over the profile's engram section), and the disk store's admitted
 * copies in their order (engram_sources_prepare), which the section must name by their device keys. */
bool explicit_engram_threads, explicit_engram_spin, explicit_engram_pin;
uint32_t engram_scan_threads = ENGRAM_DISK_SCAN_THREADS_DEFAULT;
struct engram_steps *engram_steps;
struct engram_disk *engram_disk;
const char *engram_sources_text;
const char *storage_profile_path;
/* a static hot set of table blocks (engram_hot.h): its file, its recorded SHA-256 and its budget */
const char *engram_hot_set_path, *engram_hot_sha256;
uint32_t engram_hot_mib = ENGRAM_HOT_BUDGET_DEFAULT_MIB;
struct engram_hot *engram_hot;
struct storage_profile tuned_storage;
char storage_replica_info[STORAGE_PROFILE_METRICS_BYTES];
size_t storage_replica_info_bytes;
bool tuned_load_active;
bool tuned_engram_active, storage_profile_is_valid;
/* Profile loading turned off explicitly (--storage_profile=off or the environment): no autotune hint. */
bool storage_profile_is_off;
uint32_t tuned_load_weights[STORAGE_MAXIMUM_COPIES];
struct storage_reader *engram_source_reader;
/* an EXL3 map's reads (enum exl3_read_mode, storage_exl3.h): 0 buffered, 1 buffered then dropped from the
 * page cache, 2 direct where aligned (rh01, 3 Nov: TP4 loads 36.7-38.4 / 29.8-31.5 / 32.3 s), 3 direct where
 * aligned and dropped for the rest; MAP_READS_AUTO (default): 3 when replicas were found (rh06 / rh08, 4 Nov, with
 * the copy split: 29.7 s against 31.0), else 1 (one copy: rh08t 62.5 s with 3 against 45.9 with 1) */
uint32_t map_reads = MAP_READS_AUTO;
uint32_t card_memory_kept_free_mib;
/* pipeline_steps took its TP default, DS41F_TP_MULTI_PIPELINE_STEPS (commands.h) */
bool pipeline_steps_defaulted;
/* --pipeline_steps and --draft_policy were given: session_defaults keeps them */
static bool pipeline_steps_was_given, draft_policy_was_given;
uint32_t session_slots = 1, step_graphs, pipeline_steps = 1, prefill_cone;
/* --session_slots=max was given: resolved at the placement step, as commands.h says at SESSION_SLOTS_MAX_* */
bool session_slots_max;
/* 0: serial companion flights; 1: disjoint CMP-prefix and native-PRO-tail workers. */
uint32_t tp_pipeline_overlap;
/* 0 (the default): a PRO computes with its own tensor cores (class 120 native, backend_seam.h), its outputs then
 * depend on the layers it holds; 1: it emulates class 80's sums, the same bits on every card, at 2-4 times its time */
uint32_t pro_emulation;
enum cmp_order_code cmp_order = CMP_ORDER_REFERENCE;
enum numerics_code numerics = NUMERICS_EXACT;
uint32_t tp_code_wire;
uint32_t tp_attention_owner;
uint32_t tp_head_shards;
uint32_t tp_exact_dense =
	1; /* tp4_exl3 exact recipe: attention output shards (xdense01: same bits) */
uint32_t tp_draft_shards;
uint32_t tp_draft_layers; /* tp4_exl3 BF16 recipe, thin: the draft's layers tensor-parallel (GNmac's DSpark) */
uint32_t tp_head_stream;
uint32_t exl3_fast_products =
	1; /* kernel_exl3_wide's products (identical words); 0 the retained chain */
uint32_t exl3_wide_entry =
	1; /* chunked steps run an EXL3 layer's experts once over the chunk (fast01/02) */
uint32_t exl3_tc_moe; /* tp4_exl3 BF16 recipe: GNmac's tensor-core EXL3 MoE (exl3_tc.h; GNmac numerics) */
uint32_t exl3_tc_product_blocks; /* decode product grid override; 0 keeps the measured 148-block grid */
#define TC_PRODUCT_BLOCKS_MOST \
	592u /* eight product blocks per CMP multiprocessor */
uint32_t exl3_expert_rotation; /* with exl3_tc_moe: per-expert chunk rotation 1 or 2 (exl3_tc_records.h) */
uint32_t exl3_unrated_map; /* 1: an EXL3 pack with no quality record is served: its descriptor's none, or no descriptor
			   * and a pack the model does not list (exl3_descriptor.h) */
uint32_t tp_head_mma; /* tp4_exl3 BF16 recipe: the head products on the tensor cores (kernel_head_mma) */
uint32_t tp_trunk_mma; /* tp4_exl3 BF16 recipe: thin BF16-row trunk products on the tensor cores (GNmac) */
uint32_t tp_trunk_mma_fused; /* trunk_mma's _64 form adds its own splits (same bits; relbench03: 1-5 us slower) */
uint32_t tp_trunk_mma_thin_splits; /* trunk_mma's _64 form splits as the thin form: the thin steps' bits */
uint32_t tp_router_mma; /* tp4_exl3 BF16 recipe: a BATCH step's router by split-K tensor-core MMAs (GNmac's) */
uint32_t tp_gather_balance; /* tp4_exl3: a chunk's index scans and router rows on the narrower expert ranks */
uint32_t tp_pool_chunk =
	1; /* tp4_exl3: a chunk's pool entries and keys ahead of its slabs' scans (same bits;
				       rc02ab 30K prompt 9.828 -> 9.430 s); 0: in the slabs' order */
uint32_t tp_pool_gather =
	1; /* tp4_exl3: a pooled source layer's compressor products a quarter of a half's
				       rows a rank, gathered (the same bits; ri01); 0: every row on every rank */
uint32_t tp_router_first_whole; /* tp4_exl3: the first half's router over its rows on every rank, no gather */
uint32_t exl3_tcp_two_blocks =
	1; /* the prefill TC MoE's GEMMs at two blocks a multiprocessor (the same bits;
					    ri03: a 2,048-row call -15%, Fraser first token -17 ms); 0: one block */
uint32_t tp_queries_piece =
	1; /* tp4_exl3: a wide program's queries over each piece with wq_b (the same bits;
					 ri05 30K -77 ms); 0: a span at a time from the heads copied into the full
					 rows */
uint32_t tp_index_span =
	1; /* tp4_exl3: a chunk's index scans and selections a span of rows a launch (same
				      bits; ri05 30K -51 ms); 0: a slab of 8 rows a launch */
uint32_t tp_core_compact_halves =
	1; /* tp4_exl3: a split half's core into wo_a's compact rows (same bits;
					       ri06: -9 / -11 ms); 0: the full rows and a copy */
uint32_t dense_skinny_bf16; /* tp4_exl3 class 80: BF16 skinny dense products' outputs a block above 48 rows */
uint32_t dense_skinny_f32; /* tp4_exl3 class 80: float32-weight skinny products' outputs a block above 48 rows */
#define DENSE_SKINNY_MOST_OUTPUTS \
	8u /* --dense_skinny_bf16: the widest stamped wide form (dense_linear.h) */
/* tp4_exl3 class 80: the wide dense products of 64 rows or more staged, 16 outputs a block (the same bits; rk05:
 * IDENTICAL, the 2K cell prompt 1.175 -> 1.150 s, Fraser 1.393 -> 1.385, 30K equal) */
uint32_t dense_staged = DENSE_STAGED_OUTPUTS;
/* Original TP4 CMP prefill: opt in to the existing guarded BF16 staged dense form. */
uint32_t original_dense_staged;
uint32_t original_head_offload; /* experimental target-only four-CMP head; default keeps native PRO */
uint32_t original_wide_experts; /* compatible TP4_PRO recipes default to one wide expert pass */
uint32_t tp_parallel_launch; /* TP: each card's graphs of a step's group launched from a thread of its own */
uint32_t tp_graph_parts =
	1; /* TP: a card's step graph behind the prefill cone in this many parts */
uint32_t tp_ordered_wide; /* TP EXL3: wide attention trunk products by the ordered wide form (same bits) */
uint32_t tp_graph_branches =
	1; /* TP EXL3 TC MoE: the shared expert a graph branch beside the routed (same bits,
					  engab18 -0.34 ms a C1 round); 2: BATCH steps too; 0: one chain */
uint32_t tp_piece_rings; /* TP BF16 recipe: wide steps' BF16 rings over 64-row pieces, not 8-row slabs */
uint32_t tp_trunk_shape_slices =
	1; /* tp4_exl3: thin trunk products at the slice count measured for their
					       shard shape (trunk01; ordered chains: the same bits at any count); 0:
					       4 */
uint32_t tp_bf16_slab_rings; /* 1: BF16 recipe chunked attention summed a slab at a time by the ring */
/* BF16 recipe chunked steps' sums: copy-engine owner sums and owner ring sums with the layer's halves overlapped (the
 * push's and the slab rings' bits; pf19: 30K-token prefill 19.08 -> 14.80 s) */
uint32_t tp_wide_exchange = DS41F_TP_EXCHANGE_CE_RINGS;
uint32_t tp_split_overlap =
	DS41F_SPLIT_JOINT_DOWN; /* ri01: identical (6: rb02 12.370 -> 12.057 s) */
uint32_t tp_split_least_rows; /* split_overlap: chunks of fewer rows run whole (0: halves of more than 64 rows) */
uint32_t tp_launch_trims =
	1; /* 0: the launches as before 3 Nov (ring input copies, wo_b's quantization) */
uint32_t tp_full_split_overlap; /* the full programs' split_overlap mode (0: --tp_split_overlap's) */
uint32_t tp_chunk_core =
	1; /* chunked steps' attention core over the chunk (same bits; pf16: 18.93 -> 17.80 s) */
uint32_t original_chunk_core; /* default 0: exact original attention retains its eight-row cores */
uint32_t tp_replicated_embedding =
	1; /* tp4_exl3: every rank embeds its rows (same bits); 0: card 0 hands off */
/* BATCH (tp4_v3's host half, tp4_v3_b's programs): --batch_rows=R[,R...] (9 to 64 each) prepares BATCH programs of
 * those rows in place of the thin ones; with the queue's depth 1 (step_pipeline_start(driver, 1)); a row count
 * may repeat with --batch_verifiers (another count of draft walks: e.g. BATCH-8 of one for C1 and of two for C2) */
uint32_t batch_rows[MOST_BATCH_PROGRAMS];
uint32_t batch_draft_rows, batch_wide_head;
uint32_t original_batch; /* default 0: original TP4_PRO keeps THIN; 1: fixed native-tail BATCH48 */
uint32_t batch_context_slabs; /* BATCH: a step's draft context in slabs of 8 rows (the thin steps' forms) */
/* BATCH: a BATCH program's rows the thin steps' bits (trunk_mma's _64 form at the thin splits, the target head at
 * the thin split, the TC MoE's short forms at every row count): a session's tokens independent of the batch */
uint32_t batch_thin_bits = 1;
/* BATCH, with batch_thin_bits: a step of 9 to this many rows (at most 48) runs its trunk products by the grouped forms
 * (the 1-8-row forms over its row groups of 8: the thin bits in one launch, no partials); 0: trunk_mma's _64 form */
uint32_t batch_grouped_rows = DEFAULT_BATCH_GROUPED_ROWS;
uint32_t batch_tc_paired; /* with batch_thin_bits: the TC MoE's short forms paired from so many rows */
uint32_t batch_draft_slots_head; /* with --batch_wide_head: one verifier's draft head over 8 slots a position */
uint32_t batch_row_sets; /* with batch_thin_bits: trunk products in row sets above --batch_grouped_rows */
/* A BATCH program's core, window and pools once over its slabs; retain compressor slots after the pools read them.
 * Level 3 preserves both reference widths and mixed-user outputs; level 2 remains selectable. */
uint32_t batch_core_once = DEFAULT_BATCH_CORE_ONCE;
uint32_t tp_front_branch; /* thin coefficient fronts as side branches: 1 attention, 2 both (same bits) */
uint32_t tp_core_heads; /* the attention core's most heads a block, 0 the measured rule (same bits) */
uint32_t tp_decode_round_once; /* BF16 decode partials summed in FP32 at their owners, rounded once */
uint32_t draft_bf16_rows; /* the draft layers' attention products of BF16 rows (drafts only) */
uint32_t batch_draft_wide =
	DEFAULT_BATCH_DRAFT_WIDE; /* 1 sums; 2 products; 3 cores over whole blocks */
uint32_t draft_bf16_operands; /* class 80: the draft's quantized products of its rows' BF16 values (drafts) */
#define MOST_CORE_HEADS 64u
#define CORE_HEADS_STEP \
	8u /* the attention core's tile of heads (ds41f_attention_core_records.h) */
#define MOST_BATCH_ROW_SETS 24u
/* the BF16 ring sums with column owners: a row's order of additions independent of its place in the step, so a
 * T=0 stream's bits do not depend on the batch or the schedule (default since 3 Nov, owner: new references 30K
 * 41eeb9c16e10, Fraser f49e8badff47); 0: row-quarter owners, the old reference order */
uint32_t tp_ring_column_owners = 1;
/* BATCH: a step's microseconds on each BATCH program, fewest rows first: each step verifies the drafts that buy the
 * most expected tokens a microsecond (scheduler.c's step plan); none: the draft policy's depths */
uint32_t batch_step_us[SERVING_BATCH_COSTS];
/* BATCH: each --batch_rows program's drafting sessions (its draft walks), apart from its rows: a step then takes the
 * fewest-row program holding its rows and its drafting sessions (--batch_step_us can then trim rows); none: rows / 6 */
uint32_t batch_verifiers[MOST_BATCH_PROGRAMS];
uint32_t exl3_expert_parts = EXL3_EXPERT_REFERENCE_PARTS;
enum gpu_split_code execution_placement = GPU_SPLIT_PP5;
bool numerics_was_given;
uint32_t prefix_reuse_exact;
uint32_t prefix_reuse =
	1u; /* serve: ordinary prefix reuse unless explicitly disabled */
uint32_t cache_scope = SERVER_CACHE_SESSION;
uint32_t anthropic_messages;
uint32_t session_fork; /* experimental independent same-owner clones; qualification required */
#define DEFAULT_SESSION_GRACE_SECONDS 30u
uint32_t session_grace_seconds = DEFAULT_SESSION_GRACE_SECONDS;
uint32_t session_save_tokens = SERVER_DEFAULT_SAVE_TOKENS;
uint32_t session_load_tokens = SERVER_DEFAULT_LOAD_TOKENS;
uint32_t session_idle_save_seconds = SERVER_DEFAULT_IDLE_SAVE_SECONDS;
uint32_t full_steps =
	1; /* with wide rows: the rows behind the front in full chunks (model_seam.h's FULL) */
uint32_t chunks_in_flight = DEFAULT_CHUNKS_IN_FLIGHT;
uint32_t prefill_margin =
	DEFAULT_PREFILL_MARGIN; /* full rows beyond a prompt's cone (scheduler.c) */
uint32_t prefill_fast_plan; /* chunks_in_flight 1: the largest program that fits (other bits than exact) */
uint32_t prefill_prefix_stable; /* fixed-capacity prefill; default off */
#define PREFIX_STABLE_SMALL_ROWS 1024u
#define PREFIX_STABLE_LARGE_ROWS 2048u
/* The measured bounded BATCH recipe selects shortest first; other recipes retain turns. */
#define PREFILL_ORDER_AUTO UINT32_MAX
uint32_t prefill_order =
	PREFILL_ORDER_AUTO; /* serve: prompts one at a time (scheduler.c prompt_head) */
#define PREFILL_ORDER_DEFAULT_WAIT \
	16u /* --prefill_order_wait: with --prefill_order=2, the steps before the oldest's */
#define PREFILL_ORDER_MOST_WAIT 1024u
uint32_t prefill_order_wait = PREFILL_ORDER_DEFAULT_WAIT;
#define PREFILL_DECODE_MOST_EVERY \
	64u /* --prefill_decode_every: the most chunk steps between decoding steps */
uint32_t prefill_decode_every; /* with prefill_order: a decoding step after so many head chunks (scheduler.c) */
#define PREFILL_SCHEDULING_AUTO UINT32_MAX
#define PREFILL_SCHEDULING_DEFAULT_WIDE_ROWS 1024u
/* An exact whole-granule cap even without another stream decoding. */
uint32_t prefill_chunk_rows;
uint32_t prefill_decode_rows = PREFILL_SCHEDULING_AUTO;
uint32_t prefill_decode_rounds = PREFILL_SCHEDULING_AUTO;
uint32_t prefill_decode_waiters = PREFILL_SCHEDULING_AUTO;
uint32_t engram_shared; /* TP host gathers: identical table masks share one immutable staging block */
uint32_t prefill_cone_skip =
	1; /* full steps' rows past their cone: no core or routed experts (scheduler.c) */
uint32_t prefill_tail_rows =
	TAIL_ROWS_AUTO; /* the 64-row tail chunks two a step, their forms (scheduler.c) */
uint32_t prefill_dead_halves =
	DEAD_ROWS_TRIMMED; /* full programs leave the sums of a decoder layer's half
							     the cone never reads; 2: and every sum's copies the rows
							     before a half's first one it reads (ri01); 0: none */
uint32_t prefill_ladder_1024 =
	1; /* a ladder above DS41F_CHUNK_LADDER_TOP takes a program of that many rows too
					    (ri01: with front_tail identical, 30K -82 ms) */
uint32_t prefill_ladder_fine =
	LADDER_FINE_AUTO; /* the ladder's every whole-granule count to wide_rows (with
							   1,024 and 1,536; ri07 / gsi10: identical, cold prefill
							   +0.8%, ~200 MiB a card of programs) */
uint32_t prefill_ladder_1536 =
	1; /* a ladder above DS41F_LADDER_1536_ROWS takes a program of that many rows too
					    (ri06: identical, a 1,905-token prompt -84 ms) */
uint32_t prefill_front_tail =
	FRONT_TAIL_AUTO; /* with the tail program: a wide one for the front region's tail
							   too (by default where the tail program applies) */
uint32_t prefill_tail_overlay =
	1; /* the tail programs' TC MoE workspace over the prefill form's (the same
					     launches and bits; 320-row tail: 244 MiB a card less) */
/* a tail program's trunk products one launch over its 64-row groups (the same bits; rk03c: the 2K prompt -12 ms with
 * the 320-row tail, -15 ms with the 448-row one, identical) */
uint32_t prefill_tail_sets = 1;
uint32_t prefill_thin_tail =
	THIN_TAIL_AUTO; /* the tail's rows short of a 64-row chunk in one full step,
							thin steps' forms (by default with the tail program, THIN) */
uint32_t prefill_first; /* serve: any session's wide or full chunk before other steps (scheduler.c) */
uint32_t prefill_prompt_chunks; /* opt-in after the measured C8 cost */
uint32_t draft_learned_acceptance =
	1; /* the policy's acceptance table follows outcomes (scheduler.c) */
uint32_t draft_session_acceptance; /* a session's acceptances scaled by its own outcomes (scheduler.c) */
/* a command's first DSpark round at the policy's depth (scheduler.c), not a probe of one draft: other steps, so other
 * bits while the BF16 ring's owners follow a row's place in the step (with --tp_ring_column_owners=1, the default:
 * the same); 0: the probe round */
uint32_t draft_start_economic = 1;
uint32_t batch_plan_ties; /* the BATCH step plan cuts only between drafts of unequal reach (drafts only) */
uint32_t wide_rows; /* 0: no wide steps; else a wide program of so many rows (model_seam.h) */
uint32_t step_rows = 1;
uint32_t generated_token_limit; /* 0 until given: the command's default */
/* speculative decoding (front_end_seam.h): off by default; a round verifies at most draft_positions proposals, and at
 * most --rows less 1, of the lookup drafter's n-grams of lookup_ngram_max down to lookup_ngram_min tokens */
uint32_t draft_block = DS41F_DRAFT_BLOCK;
uint32_t speculative, draft_positions = SERVING_DRAFT_MOST;
uint32_t lookup_ngram_max = DEFAULT_LOOKUP_NGRAM_MOST,
	 lookup_ngram_min = DEFAULT_LOOKUP_NGRAM_LEAST;
bool is_draft_given;
uint32_t hybrid_lookup_min = DRAFT_DEFAULT_LOOKUP_MIN;
uint32_t draft_trace; /* diagnostics: each device round's anchor and proposals to stderr */
struct draft_policy_record draft_policy;
static const char *draft_calibration_path;
const char *prompt_token_ids_path;
uint32_t ignore_eos;
uint32_t context_tokens =
	DS41F_CONTEXT_MOST; /* the default (owner, 9 Nov): 1M tokens a session, generate's and every served one's */
uint32_t seed,
	linear_profile = LINEAR_PROFILE_PRECISION,
	linear_order =
		LINEAR_ORDER_REFERENCE; /* the exact profile: the reference's orders (core, router) */
uint32_t reasoning_effort;
uint32_t official_hc_rows; /* 0 retains existing arithmetic; else pinned HC reference call shape. */
uint32_t official_reference; /* 0: existing forms; 1: original PP THIN serial M1 reference recipe. */
bool is_seed_given; /* serve: else a request without a seed draws one */
float temperature = 0.0f, top_p = 1.0f;
const char *bind_address = SERVER_DEFAULT_ADDRESS, *key_file, *model_alias,
	   *session_directory;
const char *
	layout_text; /* placement.h's layout text in place of the fill; NULL: the fill */
uint32_t port = SERVER_DEFAULT_PORT;
uint32_t queue_depth = SERVER_DEFAULT_QUEUE_DEPTH;
uint32_t user_queue_depth; /* zero: half the global queue, at least one */
const char *draft_usage = "compact";
const char *calls_parsing =
	"tolerant"; /* tool calls: tolerant, or v0.0.5-alpha's reference rule */
const char *
	dump_draft_ring; /* generate diagnostics: the draft rings' first rows to this file at the end */
uint32_t live_stats_interval_ms;
uint32_t metrics;
const char *live_stats_format = "text";
const char *stats_users;
char stats_card_order[LIVE_STATS_CARD_ORDER_BYTES];
char stats_placement[LIVE_STATS_PLACEMENT_BYTES];
/* What a run opens, closed by run_program on every path. */
struct storage_reader *reader;
const struct storage_operations_record *storage_operations = &storage_direct;
struct pack_record *pack;
struct load_layout *layout;
struct device_card *cards[BACKEND_MAXIMUM_CARDS];
struct card_description_record descriptions[BACKEND_MAXIMUM_CARDS];
struct load_result result;
struct model_instance
	instance; /* the run's model (model_registry.h): bound when its pack opens */
struct ds41f_plan *plan;
struct ds41f_tp2_model *tensor_model;
uint32_t tp_wire_bf16; /* TP-only explicit wire cast; PP retains its original numerical contract. */
struct execution_program *programs[MOST_PROGRAMS];
struct execution_program *image_programs[MOST_PROGRAMS];
struct ds41f_head_program_banks *head_program_banks;
uint32_t program_count;
struct engram_record *engram;
struct step_driver *driver;
struct inference_engine *engine;
struct http_server *server;

/* True when the argument is the option of this name, was not given before (*number_count_out is 0) and its value
 * is a number or a list like "1,0,2": digits only, each from least to most and none twice (unless repeats: a list of
 * counts, one a program), at most most_numbers. */
bool take_option(const char *argument, const char *name, uint32_t least,
		 uint32_t most, uint32_t most_numbers, bool repeats,
		 uint32_t *numbers_out, uint32_t *number_count_out)
{
	const size_t name_length = strlen(name);
	const char *text = argument;
	uint32_t number_count = 0;
	char *end = NULL;
	if (strncmp(argument, name, name_length) != 0 ||
	    *number_count_out != 0) {
		return false;
	}
	errno = 0;
	/* a digit must come first: strtoull alone would take blanks and a sign */
	for (text += name_length;
	     number_count < most_numbers && *text >= '0' && *text <= '9';
	     text = end + 1) {
		unsigned long long number = strtoull(text, &end, DECIMAL_BASE);
		for (uint32_t earlier_index = 0;
		     earlier_index < number_count && !repeats;
		     earlier_index++) {
			number = numbers_out[earlier_index] == number ?
					 UINT64_MAX :
					 number; /* twice: refused below */
		}
		if (errno != 0 || number < least || number > most) {
			return false;
		}
		numbers_out[number_count] = (uint32_t)number;
		number_count++;
		if (*end != ',') {
			*number_count_out = *end == '\0' ? number_count :
							   0; /* 0: as it was */
			return *end == '\0';
		}
	}
	return false;
}

/* The same for a number with a fraction, from least to most: digits first, then what strtof reads, all of it. */
bool take_fraction(const char *argument, const char *name, float least,
		   float most, float *value_out, uint32_t *given_count)
{
	const size_t name_length = strlen(name);
	char *end = NULL;
	if (strncmp(argument, name, name_length) != 0 || *given_count != 0 ||
	    argument[name_length] < '0' || argument[name_length] > '9') {
		return false;
	}
	errno = 0;
	const float value = strtof(argument + name_length, &end);
	*given_count = errno == 0 && *end == '\0' && value >= least &&
				       value <= most ?
			       1u :
			       0u;
	*value_out = *given_count != 0 ? value : *value_out;
	return *given_count != 0;
}

/* The same for a text: what follows the name, not empty. */
static bool take_text(const char *argument, const char *name,
		      const char **value_out, uint32_t *given_count)
{
	const size_t name_length = strlen(name);
	if (strncmp(argument, name, name_length) != 0 || *given_count != 0 ||
	    argument[name_length] == '\0') {
		return false;
	}
	*value_out = argument + name_length;
	*given_count = 1;
	return true;
}

/* A refused command line: one line naming the refused option and why (run_program prints the usage after it). */
static bool parse_refused(const char *option, const char *why)
{
	(void)fprintf(stderr, "refused: %s: %s\n", option, why);
	return false;
}

/* One refusal of a command line: whether it holds, the option it names and why. */
struct parse_refusal {
	bool refused;
	const char *option, *why;
};

/* True when no refusal holds; else false, the first that holds named (each condition is evaluated: none has a side
 * effect). */
static bool none_refused(const struct parse_refusal *refusals, size_t count)
{
	for (size_t index = 0; index < count; index++)
		if (refusals[index].refused)
			return parse_refused(refusals[index].option,
					     refusals[index].why);
	return true;
}

/* A numeric option: its name with the "=", where its values go, their range, how many it takes and the commands that
 * take it (TAKEN_BY_*). */
struct number_option {
	const char *name;
	uint32_t *numbers;
	uint32_t least, most, most_numbers, commands;
};

/* Whether the option's list may repeat a value: BATCH's two lists of counts, one a program. */
static bool numbers_repeat(const struct number_option *option)
{
	return option->numbers == batch_verifiers ||
	       option->numbers == batch_rows;
}

/* Why the option scan took no argument: a numeric option (option, given times before) the command does not take,
 * given twice or not one of its values (a list: how many, each in its range, distinct unless it repeats); else
 * (NULL) whether the catalogue knows the name (a value, a command or a repeat its parser refuses) or not. */
static bool option_refused(const char *argument,
			   const struct number_option *option, uint32_t command,
			   uint32_t given)
{
	if (option == NULL)
		return parse_refused(
			argument,
			knob_catalogued(argument) ?
				"a value, a command or a repeat this option refuses (anemone help knobs NAME)" :
				"no such option (anemone help knobs lists them)");
	if ((option->commands & command) == 0)
		return parse_refused(argument, "not an option of this command");
	if (given != 0)
		return parse_refused(argument, "given twice");
	if (option->most_numbers > 1u)
		(void)fprintf(
			stderr,
			"refused: %s: takes a list of %snumbers, at most %" PRIu32
			" values, each %" PRIu32 " to %" PRIu32 "\n",
			argument, numbers_repeat(option) ? "" : "distinct ",
			option->most_numbers, option->least, option->most);
	else
		(void)fprintf(stderr,
			      "refused: %s: takes a number %" PRIu32
			      " to %" PRIu32 "\n",
			      argument, option->least, option->most);
	return false;
}

static bool take_session_slots_max(const char *argument)
{
	if (session_slots_max ||
	    strcmp(argument, SESSION_SLOTS_MAX_OPTION) != 0)
		return false;
	session_slots_max = true;
	session_slots = SESSION_SLOTS_MAX_PARSED;
	return true;
}

/* One pinned order for the CMP trunk; the PRO, router and attention orders keep their own options. */
static bool take_cmp_order(const char *argument, bool *was_given)
{
	const bool reference = strcmp(argument, "--cmp_order=reference") == 0;
	const bool relaxed = strcmp(argument, "--cmp_order=relaxed") == 0;
	if (*was_given || (!reference && !relaxed)) {
		return false;
	}
	cmp_order = relaxed ? CMP_ORDER_RELAXED : CMP_ORDER_REFERENCE;
	*was_given = true;
	return true;
}

/* As every startup option: one --name=value token, exact case, no empty value and no repeated option. */
static bool take_execution_policy(const char *argument,
				  bool *placement_was_given)
{
	const char numerics_option[] = "--numerics=",
		   placement_option[] = "--placement=";
	if (strncmp(argument, numerics_option, sizeof(numerics_option) - 1u) ==
		    0 &&
	    !numerics_was_given) {
		numerics_was_given =
			numerics_from_text(argument + sizeof(numerics_option) -
						   1u,
					   &numerics) == STATUS_OK;
		return numerics_was_given;
	}
	if (strncmp(argument, placement_option,
		    sizeof(placement_option) - 1u) == 0 &&
	    !*placement_was_given) {
		placement_text = argument + sizeof(placement_option) - 1u;
		*placement_was_given =
			execution_placement_from_text(placement_text,
						      &execution_placement) ==
			STATUS_OK;
		return *placement_was_given;
	}
	return false;
}

/* Calibration files contain exactly 2 * 5 * 8 decimal probabilities per thousand, mode/position/bin order.
 * This is load-time input, never parsed on the token path. */
static bool draft_calibration_read(void)
{
	if (draft_calibration_path == NULL) {
		return true;
	}
	FILE *file = fopen(draft_calibration_path, "r");
	if (file == NULL) {
		return false;
	}
	char text[DRAFT_CALIBRATION_FILE_BYTES];
	const size_t byte_count = fread(text, 1, sizeof(text), file);
	const bool was_read = byte_count < sizeof(text) && feof(file) != 0 &&
			      ferror(file) == 0;
	const bool was_closed = fclose(file) == 0;
	if (!was_read || !was_closed) {
		return false;
	}
	text[byte_count] = '\0';
	const char *cursor = text;
	bool valid = true;
	for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES && valid; mode++) {
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS && valid;
		     position++) {
			for (uint32_t bin = 0;
			     bin < DRAFT_CONFIDENCE_BINS && valid; bin++) {
				cursor += strspn(cursor,
						 DRAFT_CALIBRATION_SPACES);
				const size_t length = strcspn(
					cursor, DRAFT_CALIBRATION_SPACES);
				char number[DRAFT_CALIBRATION_WORD_BYTES] = {
					0
				};
				uint32_t value = 0, count = 0;
				valid = length != 0 && length < sizeof(number);
				if (valid) {
					memcpy(number, cursor, length);
					valid = take_option(
						number, "", 0,
						DRAFT_PROBABILITY_SCALE, 1,
						false, &value, &count);
				}
				if (valid) {
					draft_policy.acceptance[mode][position]
							       [bin] = value;
					cursor += length;
				}
			}
		}
	}
	cursor += strspn(cursor, DRAFT_CALIBRATION_SPACES);
	return valid && (size_t)(cursor - text) == byte_count;
}

/* Exact cold-boundary eligibility is an actual recipe capability, independent of the selected same-session policy:
 * several sessions need their prompts reserved (--prefill_first or --prefill_order). */
bool prefix_cold_plan_is_qualified(void)
{
	return execution_placement == GPU_SPLIT_TP4_EXL3 &&
	       tp_ring_column_owners != 0 && batch_thin_bits != 0 &&
	       chunks_in_flight == 1u && prefill_fast_plan == 0 &&
	       prefill_chunk_rows == 0 && prefill_decode_rows == 0 &&
	       (session_slots <= 1u || prefill_first != 0 ||
		prefill_order != 0);
}

/* The knob given that serves one session (--official_reference always; the stable prefix,
 * exact prefix reuse and session forks unless the prompts are reserved by --prefill_first or --prefill_order), or
 * NULL. A serve without --session_slots then takes one session, named; a given count above one, or max, is refused,
 * named. */
static const char *one_session_knob(void)
{
	const bool reserved = prefill_first != 0 || prefill_order != 0;
	return official_reference != 0	  ? "--official_reference" :
	       reserved			  ? NULL :
	       prefill_prefix_stable != 0 ? "--prefill_prefix_stable" :
	       session_fork != 0	  ? "--session_fork" :
	       prefix_reuse_exact != 0	  ? "--prefix_reuse_exact" :
					    NULL;
}

/* Every parse-time default that reads the session count, for sessions of them (the same count
 * runs the same configuration, however it was reached); an explicit --pipeline_steps or --draft_policy is kept:
 * - a serve's pipeline depth with a device draft (DSpark or hybrid) and without BATCH: PP5
 *   DS41F_ORIGINAL_PP5_MULTI_PIPELINE_STEPS and the TP splits DS41F_TP_MULTI_PIPELINE_STEPS (one flight is refused
 *   there) for several sessions, else 1;
 * - TP4_EXL3's draft depth policy: the rate objective for BATCH or several sessions, else fixed.
 * parse_arguments takes them for the parsed count (SESSION_SLOTS_MAX_PARSED for max), and the fit again for one
 * session when max resolves to one (single_session_defaults). */
void session_defaults(uint32_t sessions)
{
	const bool several = is_serving && sessions > 1u && batch_rows[0] == 0;
	const bool pp5_flights = several &&
				 execution_placement == GPU_SPLIT_PP5 &&
				 speculative >= SPECULATIVE_DEVICE;
	pipeline_steps_defaulted = several && tensor_placement_selected() &&
				   speculative >= SPECULATIVE_DEVICE &&
				   !pipeline_steps_was_given;
	if (!pipeline_steps_was_given)
		pipeline_steps =
			pipeline_steps_defaulted ?
				DS41F_TP_MULTI_PIPELINE_STEPS :
			pp5_flights ? DS41F_ORIGINAL_PP5_MULTI_PIPELINE_STEPS :
				      1u;
	if (execution_placement == GPU_SPLIT_TP4_EXL3 &&
	    !draft_policy_was_given)
		draft_policy.depth_policy =
			batch_rows[0] != 0 || sessions > 1u ? DRAFT_DEPTH_RATE :
							      DRAFT_DEPTH_FIXED;
}

/* False for a command line the usage does not describe. */
bool parse_arguments(int argument_count, char **arguments)
{
	draft_policy_default(&draft_policy);
	/* The local fixed objective is resolved after parsing the configured concurrency. */
	draft_policy.depth_policy = DRAFT_DEPTH_FIXED;
	draft_policy.base_us = DEFAULT_CODE_DRAFT_BASE_US;
	draft_policy.tail_us = 0;
	draft_policy.row_us = DEFAULT_CODE_DRAFT_ROW_US;
	static const struct number_option options[] = {
		{ "--card_order=", card_order, 0, BACKEND_MAXIMUM_CARDS - 1,
		  BACKEND_MAXIMUM_CARDS, TAKEN_BY_ALL },
		{ "--metrics=", &metrics, 0, 1, 1, TAKEN_BY_SERVE },
		{ "--live_stats=", &live_stats_interval_ms, 0,
		  LIVE_STATS_MOST_INTERVAL_MS, 1, TAKEN_BY_SERVE },
		{ "--reader_threads_per_copy=", &reader_threads_per_copy, 1,
		  LOADER_MAXIMUM_READER_THREADS_PER_COPY, 1, TAKEN_BY_ALL },
		{ "--tp_load_threads=", &tp_load_threads, 0,
		  TP_LOAD_MAXIMUM_WORKERS, 1, TAKEN_BY_ALL },
		{ "--load_piece_mib=", &load_piece_mib, 0,
		  LOADER_PIECE_BYTES_MOST / BYTES_PER_MIB, 1, TAKEN_BY_ALL },
		{ "--turbo_load=", &turbo_load, 0, 1, 1, TAKEN_BY_ALL },
		{ "--load_tuned_weights=", &load_tuned_weights, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--engram_io=", &engram_io, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--replica_external=", &replica_external, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--replica_floor_percent=", &replica_floor_percent, 0,
		  LOCAL_DEVICE_PERCENT - 1u, 1, TAKEN_BY_ALL },
		{ "--engram_ram_percent=", &engram_ram_percent, 0,
		  ENGRAM_RAM_PERCENT, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--engram_fit_only=", &engram_fit_only, 0, 1, 1,
		  TAKEN_BY_GENERATE },
		{ "--engram_cache_mib=", &engram_cache_mib, 0,
		  ENGRAM_CACHE_MAXIMUM_BYTES / BYTES_PER_MIB, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--turbo-nvme-engrams=", &turbo_nvme_engrams, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--engram_step_io=", &engram_step_io, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--engram_io_threads=", &engram_io_threads, 1,
		  STORAGE_FLIGHT_THREADS_PER_COPY_MOST, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--engram_spin_us=", &engram_spin_us, 0,
		  ENGRAM_STEP_SPIN_US_MOST, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--engram_io_pin=", &engram_io_pin, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--engram_scan_threads=", &engram_scan_threads, 0,
		  ENGRAM_DISK_SCAN_THREADS_MOST, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--engram_hot_mib=", &engram_hot_mib, 1,
		  ENGRAM_HOT_BUDGET_MOST_MIB, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--map_reads=", &map_reads, 0, MAP_READS_AUTO, 1,
		  TAKEN_BY_ALL },
		{ "--tp_load_host_threads=", &tp_load_host_threads, 0,
		  TP_LOAD_MAXIMUM_WORKERS, 1, TAKEN_BY_ALL },
		{ "--tp_load_copy_split=", &tp_load_copy_split, 0,
		  TP_LOAD_SPLITS - 1u, 1, TAKEN_BY_ALL },
		{ "--tp_load_reads_per_copy=", &tp_load_reads_per_copy, 0,
		  TP_LOAD_MAXIMUM_WORKERS, 1, TAKEN_BY_ALL },
		{ "--card_memory_kept_free_mib=", &card_memory_kept_free_mib, 0,
		  MOST_CARD_MEMORY_KEPT_FREE_MIB, 1, TAKEN_BY_ALL },
		{ "--pro_emulation=", &pro_emulation, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_code_wire=", &tp_code_wire, 0, 1, 1, TAKEN_BY_ALL },
		{ "--exl3_expert_parts=", &exl3_expert_parts,
		  EXL3_EXPERT_REFERENCE_PARTS, EXL3_EXPERT_OPTIONAL_PARTS, 1,
		  TAKEN_BY_ALL },
		{ "--tp_attention_owner=", &tp_attention_owner, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_head_shards=", &tp_head_shards, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_exact_dense=", &tp_exact_dense, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_draft_shards=", &tp_draft_shards, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_draft_layers=", &tp_draft_layers, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_head_stream=", &tp_head_stream, 0, 1, 1, TAKEN_BY_ALL },
		{ "--exl3_fast_products=", &exl3_fast_products, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--exl3_wide_entry=", &exl3_wide_entry, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--exl3_tc_moe=", &exl3_tc_moe, 0, 1, 1, TAKEN_BY_ALL },
		{ "--exl3_tc_product_blocks=", &exl3_tc_product_blocks, 0,
		  TC_PRODUCT_BLOCKS_MOST, 1, TAKEN_BY_ALL },
		{ "--exl3_expert_rotation=", &exl3_expert_rotation, 0,
		  DS41F_EXL3_TC_ROTATION_THREE_WAY, 1, TAKEN_BY_ALL },
		{ "--exl3_unrated_map=", &exl3_unrated_map, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_head_mma=", &tp_head_mma, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_trunk_mma=", &tp_trunk_mma, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_trunk_mma_fused=", &tp_trunk_mma_fused, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_trunk_mma_thin_splits=", &tp_trunk_mma_thin_splits, 0,
		  1, 1, TAKEN_BY_ALL },
		{ "--tp_router_mma=", &tp_router_mma, 0, 2, 1, TAKEN_BY_ALL },
		{ "--tp_gather_balance=", &tp_gather_balance, 0,
		  DS41F_GATHER_BALANCE_WEIGHTED, 1, TAKEN_BY_ALL },
		{ "--tp_pool_chunk=", &tp_pool_chunk, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_pool_gather=", &tp_pool_gather, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_router_first_whole=", &tp_router_first_whole, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--exl3_tcp_two_blocks=", &exl3_tcp_two_blocks, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_queries_piece=", &tp_queries_piece, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_index_span=", &tp_index_span, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_core_compact_halves=", &tp_core_compact_halves, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--dense_skinny_bf16=", &dense_skinny_bf16, 0,
		  DENSE_SKINNY_MOST_OUTPUTS, 1, TAKEN_BY_ALL },
		{ "--dense_skinny_f32=", &dense_skinny_f32, 0,
		  DENSE_SKINNY_MOST_OUTPUTS, 1, TAKEN_BY_ALL },
		{ "--dense_staged=", &dense_staged, 0,
		  DENSE_STAGED_MOST_OUTPUTS, 1, TAKEN_BY_ALL },
		{ "--original_dense_staged=", &original_dense_staged, 0,
		  DENSE_STAGED_OUTPUTS, 1, TAKEN_BY_ALL },
		{ "--original_head_offload=", &original_head_offload, 0,
		  DS41F_HEAD_MODE_BANKED, 1, TAKEN_BY_ALL },
		{ "--original_wide_experts=", &original_wide_experts, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_parallel_launch=", &tp_parallel_launch, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_graph_parts=", &tp_graph_parts, 1,
		  DS41F_TP_MOST_GRAPH_PARTS, 1, TAKEN_BY_ALL },
		{ "--tp_ordered_wide=", &tp_ordered_wide, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_graph_branches=", &tp_graph_branches, 0, 2, 1,
		  TAKEN_BY_ALL },
		{ "--tp_piece_rings=", &tp_piece_rings, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_trunk_shape_slices=", &tp_trunk_shape_slices, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_bf16_slab_rings=", &tp_bf16_slab_rings, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_wide_exchange=", &tp_wide_exchange, 0,
		  DS41F_TP_EXCHANGE_CE_RINGS, 1, TAKEN_BY_ALL },
		{ "--tp_split_overlap=", &tp_split_overlap, 0,
		  DS41F_SPLIT_JOINT_DOWN, 1, TAKEN_BY_ALL },
		{ "--tp_split_least_rows=", &tp_split_least_rows, 0,
		  DS41F_TP_SPLIT_MOST_ROWS, 1, TAKEN_BY_ALL },
		{ "--tp_launch_trims=", &tp_launch_trims, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_full_split_overlap=", &tp_full_split_overlap, 0,
		  DS41F_SPLIT_JOINT_DOWN, 1, TAKEN_BY_ALL },
		{ "--tp_chunk_core=", &tp_chunk_core, 0, 1, 1, TAKEN_BY_ALL },
		{ "--original_chunk_core=", &original_chunk_core, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--original_batch=", &original_batch, 0, 1, 1,
		  TAKEN_BY_SERVE },
		{ "--tp_replicated_embedding=", &tp_replicated_embedding, 0, 1,
		  1, TAKEN_BY_ALL },
		{ "--batch_rows=", batch_rows, 1, PROGRAM_BATCH_MOST_ROWS,
		  MOST_BATCH_PROGRAMS, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_draft_rows=", &batch_draft_rows, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_wide_head=", &batch_wide_head, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_context_slabs=", &batch_context_slabs, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_thin_bits=", &batch_thin_bits, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_tc_paired=", &batch_tc_paired, 0,
		  PROGRAM_BATCH_MOST_ROWS, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_draft_slots_head=", &batch_draft_slots_head, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_row_sets=", &batch_row_sets, 0, MOST_BATCH_ROW_SETS,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_core_once=", &batch_core_once, 0, 3, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--tp_front_branch=", &tp_front_branch, 0, 2, 1,
		  TAKEN_BY_ALL },
		{ "--tp_core_heads=", &tp_core_heads, 0, MOST_CORE_HEADS, 1,
		  TAKEN_BY_ALL },
		{ "--tp_decode_round_once=", &tp_decode_round_once, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--draft_bf16_rows=", &draft_bf16_rows, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--batch_draft_wide=", &batch_draft_wide, 0, 3, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_bf16_operands=", &draft_bf16_operands, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--batch_grouped_rows=", &batch_grouped_rows, 0,
		  MOST_BATCH_GROUPED_ROWS, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--tp_ring_column_owners=", &tp_ring_column_owners, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--batch_step_us=", batch_step_us, 0, DRAFT_MOST_COST_US,
		  SERVING_BATCH_COSTS, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_verifiers=", batch_verifiers, 1,
		  STEP_BATCH_MAXIMUM_SESSIONS, MOST_BATCH_PROGRAMS,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--tp_wire_bf16=", &tp_wire_bf16, ORDERED_PUSH_FP32,
		  ORDERED_PUSH_BF16, 1, TAKEN_BY_ALL },
		{ "--full_steps=", &full_steps, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_first=", &prefill_first, 0, 1, 1, TAKEN_BY_SERVE },
		{ "--prefill_prompt_chunks=", &prefill_prompt_chunks, 0, 1, 1,
		  TAKEN_BY_SERVE },
		{ "--draft_learned_acceptance=", &draft_learned_acceptance, 0,
		  1, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_session_acceptance=", &draft_session_acceptance, 0,
		  11, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--batch_plan_ties=", &batch_plan_ties, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_start_economic=", &draft_start_economic, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--chunks_in_flight=", &chunks_in_flight, 1,
		  STEP_PIPELINE_MOST, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_margin=", &prefill_margin, 1, SERVING_CONE_MARGIN,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_fast_plan=", &prefill_fast_plan, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_prefix_stable=", &prefill_prefix_stable, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_order=", &prefill_order, 0,
		  PREFILL_ORDER_SHORT_ROWS, 1, TAKEN_BY_SERVE },
		{ "--prefill_order_wait=", &prefill_order_wait, 0,
		  PREFILL_ORDER_MOST_WAIT, 1, TAKEN_BY_SERVE },
		{ "--prefill_decode_every=", &prefill_decode_every, 0,
		  PREFILL_DECODE_MOST_EVERY, 1, TAKEN_BY_SERVE },
		{ "--prefill_decode_waiters=", &prefill_decode_waiters, 0, 1, 1,
		  TAKEN_BY_SERVE },
		{ "--prefill_decode_rounds=", &prefill_decode_rounds, 1,
		  PREFILL_DECODE_MOST_ROUNDS, 1, TAKEN_BY_SERVE },
		{ "--prefill_chunk_rows=", &prefill_chunk_rows, 0,
		  STEP_WIDE_MOST_ROWS, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_decode_rows=", &prefill_decode_rows, 0,
		  STEP_WIDE_MOST_ROWS, 1, TAKEN_BY_SERVE },
		{ "--prefill_cone_skip=", &prefill_cone_skip, 0, 3, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_tail_rows=", &prefill_tail_rows, 0,
		  DS41F_TAIL_WIDEST_ROWS, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_ladder_1024=", &prefill_ladder_1024, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_ladder_1536=", &prefill_ladder_1536, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_ladder_fine=", &prefill_ladder_fine, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_front_tail=", &prefill_front_tail, 0,
		  FRONT_TAIL_ALL, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_tail_overlay=", &prefill_tail_overlay, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_tail_sets=", &prefill_tail_sets, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_thin_tail=", &prefill_thin_tail, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_dead_halves=", &prefill_dead_halves, 0,
		  DEAD_ROWS_TRIMMED, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefix_reuse=", &prefix_reuse, 0, 1, 1, TAKEN_BY_SERVE },
		{ "--prefix_reuse_exact=", &prefix_reuse_exact, 0,
		  SERVER_PREFIX_COLD_PLAN, 1, TAKEN_BY_SERVE },
		{ "--cache_scope=", &cache_scope, SERVER_CACHE_SESSION,
		  SERVER_CACHE_GLOBAL, 1, TAKEN_BY_SERVE },
		{ "--anthropic_messages=", &anthropic_messages, 0, 1, 1,
		  TAKEN_BY_SERVE },
		{ "--session_fork=", &session_fork, 0, 1, 1, TAKEN_BY_SERVE },
		{ "--session_grace_seconds=", &session_grace_seconds, 0,
		  SERVER_MOST_GRACE_SECONDS, 1, TAKEN_BY_SERVE },
		{ "--session_save_tokens=", &session_save_tokens, 1,
		  DS41F_CONTEXT_MOST, 1, TAKEN_BY_SERVE },
		{ "--session_load_tokens=", &session_load_tokens, 1,
		  DS41F_CONTEXT_MOST, 1, TAKEN_BY_SERVE },
		{ "--session_idle_save_seconds=", &session_idle_save_seconds, 1,
		  SERVER_MOST_IDLE_SAVE_SECONDS, 1, TAKEN_BY_SERVE },
		{ "--session_slots=", &session_slots, 1, SERVING_SLOTS_MOST, 1,
		  TAKEN_BY_SERVE },
		{ "--queue_depth=", &queue_depth, 1, SERVER_MOST_WAITING, 1,
		  TAKEN_BY_SERVE },
		{ "--user_queue_depth=", &user_queue_depth, 1,
		  SERVER_MOST_WAITING, 1, TAKEN_BY_SERVE },
		{ "--step_graphs=", &step_graphs, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--engram_shared=", &engram_shared, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--pipeline_steps=", &pipeline_steps, 1, STEP_PIPELINE_MOST,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--tp_pipeline_overlap=", &tp_pipeline_overlap, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_cone=", &prefill_cone, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--wide_rows=", &wide_rows, 0, STEP_WIDE_MOST_ROWS, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--rows=", &step_rows, 1, LINEAR_MAXIMUM_ROWS, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--context_tokens=", &context_tokens, 1, DS41F_CONTEXT_MOST,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--generated_tokens=", &generated_token_limit, 1, UINT32_MAX,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--seed=", &seed, 0, UINT32_MAX, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--reasoning_effort=", &reasoning_effort, 0,
		  DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--linear_profile=", &linear_profile, 0, LINEAR_PROFILE_SPEED,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--linear_order=", &linear_order, 0, LINEAR_ORDER_REFERENCE,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--official_hc_rows=", &official_hc_rows, 0,
		  LINEAR_MAXIMUM_ROWS, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--official_reference=", &official_reference, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--ignore_eos=", &ignore_eos, 0, 1, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--port=", &port, 1, SERVER_MOST_PORT, 1, TAKEN_BY_SERVE },
		{ "--speculative=", &speculative, SPECULATIVE_OFF,
		  SPECULATIVE_HYBRID, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_policy=", &draft_policy.depth_policy,
		  DRAFT_DEPTH_FIXED, DRAFT_DEPTH_ADAPTIVE, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_threshold=", &draft_policy.threshold, 0,
		  DRAFT_PROBABILITY_SCALE, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_base_us=", &draft_policy.base_us, 1,
		  DRAFT_MOST_COST_US, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_tail_us=", &draft_policy.tail_us, 0,
		  DRAFT_MOST_COST_US, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_row_us=", &draft_policy.row_us, 1,
		  DRAFT_MOST_COST_US, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_probe_rounds=", &draft_policy.probe_rounds, 1,
		  DRAFT_MOST_PROBE_ROUNDS, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--hybrid_lookup_prior=", &draft_policy.lookup_prior, 0,
		  DRAFT_PROBABILITY_SCALE, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--hybrid_lookup_min=", &hybrid_lookup_min, 1,
		  DRAFT_LOOKUP_MOST_NGRAM, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_trace=", &draft_trace, 0, TIMING_HOOKS_ENABLED, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_block=", &draft_block, DS41F_DRAFT_BLOCK,
		  STEP_DRAFT_MOST, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--draft_positions=", &draft_positions, 1, SERVING_DRAFT_MOST,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--speculative_depth=", &draft_positions, 1,
		  SERVING_DRAFT_MOST, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--lookup_ngram_max=", &lookup_ngram_max, 1,
		  DRAFT_LOOKUP_MOST_NGRAM, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--lookup_ngram_min=", &lookup_ngram_min, 1,
		  DRAFT_LOOKUP_MOST_NGRAM, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE }
	};
	static const struct text_option {
		const char *name;
		const char **text;
	} texts[TEXT_OPTIONS] = {
		{ "--bind_address=", &bind_address },
		{ "--key_file=", &key_file },
		{ "--stats_users=", &stats_users },
		{ "--model_alias=", &model_alias },
		{ "--session_directory=", &session_directory },
		{ "--draft_usage=", &draft_usage },
		{ "--dsml_parsing=", &calls_parsing },
		{ "--engram_hot_set=", &engram_hot_set_path },
		{ "--engram_hot_sha256=", &engram_hot_sha256 },
		{ "--engram_sources=", &engram_sources_text },
		{ "--storage_profile=", &storage_profile_path },
		{ "--dump_draft_ring=", &dump_draft_ring },
		{ "--layout=", &layout_text }
	};
	const uint32_t option_count = sizeof(options) / sizeof(options[0]);
	uint32_t given[sizeof(options) / sizeof(options[0]) + FRACTION_OPTIONS +
		       TEXT_OPTIONS] = { 0 }; /* then the rest */
	uint32_t *const given_fractions = &given[option_count];
	uint32_t *const given_texts = &given[option_count + FRACTION_OPTIONS];
	int argument_index = ARGUMENTS_BEFORE_THE_OPTIONS;
	bool cmp_order_was_given = false;
	bool placement_was_given = false;
	bool speculative_was_given = false;
	bool depth_was_given = false;
	pipeline_steps_was_given = false;
	bool original_wide_experts_was_given = false;
	bool engram_step_io_was_given = false;
	bool engram_ram_auto_was_given = false;
	draft_policy_was_given = false;
	bool batch_costs_were_given = false;
	bool draft_learning_was_given = false, draft_base_was_given = false;
	bool draft_tail_was_given = false, draft_row_was_given = false;
	bool batch_core_was_given = false;
	uint32_t draft_kind = SPECULATIVE_OFF;
	uint32_t calibration_given = 0, stats_format_given = 0;
	uint32_t prompt_token_ids_given = 0;
	is_serving = argument_count > 1 && strcmp(arguments[1], "serve") == 0;
	is_capturing = argument_count > 1 &&
		       strcmp(arguments[1], "capture") == 0;
	is_generating =
		is_serving || is_capturing ||
		(argument_count > 1 && strcmp(arguments[1], "generate") == 0);
	if (argument_count < ARGUMENTS_BEFORE_THE_OPTIONS)
		return parse_refused("the command", "none given");
	if (!is_generating && strcmp(arguments[1], "load") != 0)
		return parse_refused(arguments[1], "not a command");
	const uint32_t command = is_serving    ? TAKEN_BY_SERVE :
				 is_generating ? TAKEN_BY_GENERATE :
						 TAKEN_BY_LOAD;
	for (; argument_index < argument_count &&
	       strncmp(arguments[argument_index], "--", strlen("--")) == 0;
	     argument_index++) {
		const char *argument = arguments[argument_index];
		bool capture_valid = true;
		if (is_capturing &&
		    golden_capture_option_taken(argument, &capture_valid)) {
			if (!capture_valid)
				return parse_refused(
					argument,
					"a value out of its range, or given twice");
			continue;
		}
		bool is_taken =
			is_serving &&
			take_text(argument,
				  "--live_stats_format=", &live_stats_format,
				  &stats_format_given);
		is_taken =
			is_taken ||
			(!is_serving && is_generating &&
			 take_text(argument, "--prompt_token_ids=",
				   &prompt_token_ids_path,
				   &prompt_token_ids_given)) ||
			take_execution_policy(argument, &placement_was_given) ||
			take_model_source(argument) ||
			(is_serving && take_session_slots_max(argument)) ||
			(is_generating &&
			 take_engram_ram_auto(argument,
					      &engram_ram_auto_was_given)) ||
			(is_generating &&
			 (take_cmp_order(argument, &cmp_order_was_given) ||
			  take_text(argument, "--draft_calibration=",
				    &draft_calibration_path,
				    &calibration_given) ||
			  take_fraction(argument, "--temperature=", 0.0f,
					FLT_MAX, &temperature,
					&given_fractions[0]) ||
			  take_fraction(argument, "--top_p=", FLT_TRUE_MIN,
					1.0f, &top_p, &given_fractions[1]) ||
			  (!is_draft_given &&
			   (strcmp(argument, DRAFT_LOOKUP_OPTION) == 0 ||
			    strcmp(argument, DRAFT_DEVICE_OPTION) == 0 ||
			    strcmp(argument, DRAFT_HYBRID_OPTION) == 0))));
		if (is_taken && (strcmp(argument, DRAFT_LOOKUP_OPTION) == 0 ||
				 strcmp(argument, DRAFT_DEVICE_OPTION) == 0 ||
				 strcmp(argument, DRAFT_HYBRID_OPTION) == 0)) {
			draft_kind =
				strcmp(argument, DRAFT_HYBRID_OPTION) == 0 ?
					SPECULATIVE_HYBRID :
				strcmp(argument, DRAFT_DEVICE_OPTION) == 0 ?
					SPECULATIVE_DEVICE :
					SPECULATIVE_LOOKUP;
		}
		is_draft_given = is_draft_given ||
				 (is_taken && draft_kind != SPECULATIVE_OFF);
		for (uint32_t index = 0; index < TEXT_OPTIONS && !is_taken;
		     index++) {
			if (!is_serving && index != LAYOUT_TEXT &&
			    index != DUMP_TEXT &&
			    index != STORAGE_PROFILE_TEXT &&
			    !(is_generating && index >= ENGRAM_HOT_SET_TEXT &&
			      index <= ENGRAM_SOURCES_TEXT)) {
				continue;
			}
			is_taken = take_text(argument, texts[index].name,
					     texts[index].text,
					     &given_texts[index]);
		}
		for (uint32_t index = 0; index < option_count && !is_taken;
		     index++) {
			is_taken = (options[index].commands & command) != 0 &&
				   take_option(argument, options[index].name,
					       options[index].least,
					       options[index].most,
					       options[index].most_numbers,
					       numbers_repeat(&options[index]),
					       options[index].numbers,
					       &given[index]);
			is_seed_given =
				is_seed_given ||
				(is_taken && options[index].numbers == &seed);
			speculative_was_given =
				speculative_was_given ||
				(is_taken &&
				 options[index].numbers == &speculative);
			depth_was_given =
				depth_was_given ||
				(is_taken &&
				 options[index].numbers == &draft_positions);
			original_wide_experts_was_given =
				original_wide_experts_was_given ||
				(is_taken && options[index].numbers ==
						     &original_wide_experts);
			engram_step_io_was_given =
				engram_step_io_was_given ||
				(is_taken &&
				 options[index].numbers == &engram_step_io);
			pipeline_steps_was_given =
				pipeline_steps_was_given ||
				(is_taken &&
				 options[index].numbers == &pipeline_steps);
			draft_policy_was_given =
				draft_policy_was_given ||
				(is_taken &&
				 options[index].numbers ==
					 &draft_policy.depth_policy);
			draft_learning_was_given =
				draft_learning_was_given ||
				(is_taken && options[index].numbers ==
						     &draft_learned_acceptance);
			draft_base_was_given =
				draft_base_was_given ||
				(is_taken && options[index].numbers ==
						     &draft_policy.base_us);
			draft_tail_was_given =
				draft_tail_was_given ||
				(is_taken && options[index].numbers ==
						     &draft_policy.tail_us);
			draft_row_was_given =
				draft_row_was_given ||
				(is_taken && options[index].numbers ==
						     &draft_policy.row_us);
			batch_core_was_given =
				batch_core_was_given ||
				(is_taken &&
				 options[index].numbers == &batch_core_once);
			if (is_taken &&
			    options[index].numbers == batch_step_us) {
				batch_costs_were_given = true;
				for (uint32_t price = 0; price < given[index];
				     price++)
					if (batch_step_us[price] == 0 &&
					    given[index] != 1u)
						return parse_refused(
							argument,
							"a single 0 turns the planning off; a list holds prices only");
			}
		}
		if (!is_taken) {
			uint32_t index = 0;
			while (index < option_count &&
			       strncmp(argument, options[index].name,
				       strlen(options[index].name)) != 0)
				index++;
			/* a second max or auto is given twice, as a second number is */
			const bool word_again =
				(session_slots_max &&
				 strcmp(argument, SESSION_SLOTS_MAX_OPTION) ==
					 0) ||
				(engram_ram_auto_was_given &&
				 strcmp(argument, ENGRAM_RAM_AUTO_OPTION) == 0);
			return option_refused(
				argument,
				index < option_count ? &options[index] : NULL,
				command,
				index < option_count ?
					given[index] + (word_again ? 1u : 0u) :
					0);
		}
	}
	if (!weights_placement_matches(placement_was_given))
		return false;
	if (is_draft_given) {
		if (speculative_was_given && speculative != draft_kind)
			return parse_refused(
				"--speculative",
				"conflicts with the --draft given");
		speculative = draft_kind;
	}
	if (speculative >= SPECULATIVE_DEVICE &&
	    draft_positions > draft_block) {
		if (depth_was_given)
			return parse_refused("--draft_positions",
					     "above --draft_block");
		draft_positions = draft_block;
	}
	card_count = given[0];
	bool engram_ram_percent_was_given = false;
	for (uint32_t index = 0; index < option_count; index++)
		engram_ram_percent_was_given =
			engram_ram_percent_was_given ||
			(options[index].numbers == &engram_ram_percent &&
			 given[index] != 0);
	if (engram_ram_percent_was_given && engram_ram_auto_was_given)
		return parse_refused("--engram_ram_percent",
				     "auto or a percent, not both");
	engram_ram_source = engram_ram_percent_was_given ? ENGRAM_RAM_KNOB :
							   ENGRAM_RAM_AUTO;
	bool slots_given = false;
	for (uint32_t index = 0; index < option_count; index++)
		slots_given = slots_given ||
			      (options[index].numbers == &session_slots &&
			       given[index] != 0);
	if (session_slots_max && slots_given)
		return parse_refused("--session_slots",
				     "max or a count, not both");
	const bool slots_defaulted = is_serving && !slots_given &&
				     !session_slots_max;
	if (slots_defaulted) {
		session_slots_max = true; /* serve's default: as many as fit */
		session_slots = SESSION_SLOTS_MAX_PARSED;
	}
	/* A BATCH serve runs exactly STEP_BATCH_MAXIMUM_SESSIONS sessions (the BATCH programs'), named */
	if (is_serving && (original_batch != 0 || batch_rows[0] != 0) &&
	    (session_slots_max ||
	     session_slots != STEP_BATCH_MAXIMUM_SESSIONS)) {
		if (session_slots_max)
			(void)fprintf(
				stderr,
				"serve: BATCH (--original_batch, --batch_rows) runs exactly %u sessions, and "
				"--session_slots=max (the default) is the fit: give --session_slots=%u, or serve "
				"without the BATCH knobs (thin steps) for as many sessions as fit\n",
				STEP_BATCH_MAXIMUM_SESSIONS,
				STEP_BATCH_MAXIMUM_SESSIONS);
		else
			(void)fprintf(
				stderr,
				"serve: BATCH (--original_batch, --batch_rows) runs exactly %u sessions, not "
				"--session_slots=%" PRIu32
				": give --session_slots=%u, or serve without the "
				"BATCH knobs (thin steps) for %" PRIu32
				" sessions\n",
				STEP_BATCH_MAXIMUM_SESSIONS, session_slots,
				STEP_BATCH_MAXIMUM_SESSIONS, session_slots);
		return false;
	}
	/* A context the model refuses at any count is refused here, by name, not as a fit */
	if (context_tokens % DS41F_CONTEXT_GRANULE != 0u) {
		const uint32_t below =
			context_tokens - context_tokens % DS41F_CONTEXT_GRANULE;
		(void)fprintf(
			stderr,
			"--context_tokens=%" PRIu32
			" is not a multiple of the context's granule, %u tokens: give %" PRIu32
			"\n",
			context_tokens, DS41F_CONTEXT_GRANULE,
			below != 0u ? below : DS41F_CONTEXT_GRANULE);
		return false;
	}
	if (official_reference != 0 &&
	    execution_placement == GPU_SPLIT_TP4_EXL3) {
		(void)fputs("reference arithmetic (EXL3: not official-exact)\n",
			    stderr);
		return false;
	}
	/* capture: the cases decide what generate's command line gives */
	const struct parse_refusal capture_refusals[] = {
		{ (golden_capture_options.given & CAPTURE_REQUIRED) !=
			  CAPTURE_REQUIRED,
		  "capture",
		  "needs --capture_golden, --capture_cases and --capture_out" },
		{ given_fractions[0] && temperature != 0.0f, "--temperature",
		  "a capture draws greedily: the witnesses are T=0 draws" },
		{ given_fractions[1] && top_p != 1.0f, "--top_p",
		  "a capture draws greedily: the witnesses are T=0 draws" },
		{ generated_token_limit != 0, "--generated_tokens",
		  "each case's continuation decides" },
		{ ignore_eos != 0, "--ignore_eos",
		  "a forced continuation ends at its limit alone" },
		{ prompt_token_ids_path != NULL, "--prompt_token_ids",
		  "each case's prompt_ids.json decides" },
		{ reasoning_effort != 0, "--reasoning_effort",
		  "each case's prompt decides" },
		{ batch_rows[0] != 0, "--batch_rows",
		  "BATCH heads are out of capture's scope" },
		{ original_head_offload != 0, "--original_head_offload",
		  "the offloaded head is out of capture's scope" },
		{ dump_draft_ring != NULL, "--dump_draft_ring",
		  "a diagnostic, out of capture's scope" }
	};
	if (is_capturing && !none_refused(capture_refusals,
					  sizeof(capture_refusals) /
						  sizeof(capture_refusals[0])))
		return false;
	generated_token_limit = generated_token_limit != 0 ?
					generated_token_limit :
				is_serving ? UINT32_MAX :
					     DEFAULT_GENERATED;
	const uint32_t file_count =
		is_serving || is_capturing || prompt_token_ids_path != NULL ?
			1 :
		is_generating ? FRONT_END_FILES :
				0; /* serve's: the tables */
	/* with --model_dir the folder gives the tables and the pack; only generate's messages stay positional */
	const uint32_t typed_count = model_dir == NULL ? file_count :
				     file_count == FRONT_END_FILES ? 1u :
								     0u;
	for (uint32_t file = 0;
	     file < typed_count && argument_index < argument_count; file++) {
		front_end_paths[FRONT_END_FILES - file_count + file] =
			arguments[argument_index++];
	}
	copy_paths = (const char *const *)&arguments[argument_index];
	copy_count = argument_index < argument_count ?
			     (uint32_t)(argument_count - argument_index) :
			     0;
	pack_argument = copy_count != 0 ? copy_paths[0] : NULL;
	if (model_dir != NULL && copy_count != 0) {
		(void)fprintf(
			stderr,
			"--model_dir replaces the tables and pack arguments: %s is one too many\n",
			copy_paths[0]);
		return false;
	}
	if (exl3_map_name != NULL &&
	    (model_dir == NULL || folder_pack_file() != CLI_MODEL_FILE_EXL3)) {
		(void)fprintf(
			stderr,
			"--exl3_map=%s names the EXL3 map in --model_dir's folder: give it with --model_dir and "
			"--weights=exl3 or --placement=tp4_exl3\n",
			exl3_map_name);
		return false;
	}
	if (weights_family != WEIGHTS_UNSET && copy_count != 0 &&
	    (storage_for_path(copy_paths[0]) == &storage_exl3) !=
		    (weights_family == WEIGHTS_EXL3)) {
		(void)fprintf(
			stderr, "--weights=%s needs %s; %s is %s\n",
			weights_family == WEIGHTS_EXL3 ? "exl3" : "original",
			weights_family == WEIGHTS_EXL3 ?
				"an .exl3map pack" :
				"an original pack, not an .exl3map",
			copy_paths[0],
			weights_family == WEIGHTS_EXL3 ? "an original pack" :
							 "an EXL3 map");
		return false;
	}
	if (prompt_token_ids_path != NULL)
		front_end_paths[MESSAGES_FILE] = prompt_token_ids_path;
	if (ignore_eos != 0 &&
	    (speculative != SPECULATIVE_OFF || batch_rows[0] != 0))
		return parse_refused(
			"--ignore_eos",
			"non-speculative THIN only (no --speculative, no --batch_rows)");
	/* --prefill_tail_rows: by default where the tail program applies; 0 the old path */
	const bool tail_applies = execution_placement == GPU_SPLIT_TP4_EXL3 &&
				  chunks_in_flight == 1u &&
				  prefill_fast_plan == 0 && full_steps != 0 &&
				  wide_rows >= TAIL_LEAST_WIDE_ROWS &&
				  exl3_tc_moe != 0 && tp_trunk_mma != 0;
	if (prefill_tail_rows == TAIL_ROWS_AUTO)
		prefill_tail_rows = !tail_applies ? 0u :
				    wide_rows >= TAIL_LONG_WIDE_ROWS &&
						    batch_rows[0] == 0 ?
						    DS41F_TAIL_WIDEST_ROWS :
						    DS41F_TAIL_PROGRAM_ROWS;
	/* Only the measured BATCH recipe gains defaults. THIN, generate, explicit zero and incompatible plans
	 * retain their old behavior; a user can still select every scheduling component independently. */
	const bool bounded_batch =
		is_serving && batch_rows[0] != 0 && tail_applies &&
		prefill_first != 0 &&
		wide_rows == PREFILL_SCHEDULING_DEFAULT_WIDE_ROWS &&
		tp_wire_bf16 != 0 && tp_ring_column_owners != 0 &&
		batch_thin_bits != 0;
	if (prefill_decode_rows == PREFILL_SCHEDULING_AUTO)
		prefill_decode_rows =
			bounded_batch ? PREFILL_DECODE_DEFAULT_ROWS : 0u;
	const bool bounded_default = bounded_batch && prefill_decode_rows != 0;
	if (prefill_decode_rounds == PREFILL_SCHEDULING_AUTO)
		prefill_decode_rounds =
			bounded_default ? prefill_decode_default_rounds : 1u;
	if (prefill_decode_waiters == PREFILL_SCHEDULING_AUTO)
		prefill_decode_waiters =
			bounded_default ? prefill_decode_default_waiters : 0u;
	const bool selected_combination =
		bounded_default &&
		prefill_decode_rows == PREFILL_DECODE_DEFAULT_ROWS &&
		prefill_decode_rounds == prefill_decode_default_rounds &&
		prefill_decode_waiters == prefill_decode_default_waiters;
	/* Concurrent prompt rows require the measured row-independent EXL3 BATCH forms. */
	if (prefill_order == PREFILL_ORDER_SHORT_ROWS &&
	    (!bounded_batch || prefix_reuse_exact != 0 || session_fork != 0 ||
	     prefill_prefix_stable != 0))
		return parse_refused(
			"--prefill_order",
			"concurrent prompt rows need the measured TP4_EXL3 BATCH serve recipe, without "
			"--prefix_reuse_exact, --session_fork or --prefill_prefix_stable");
	if (prefill_order == PREFILL_ORDER_AUTO)
		prefill_order = selected_combination ? PREFILL_ORDER_SHORTEST :
						       0u;
	if (prefill_thin_tail == THIN_TAIL_AUTO)
		prefill_thin_tail =
			prefill_tail_rows != 0 && batch_rows[0] == 0 ? 1u : 0u;
	if (prefill_front_tail == FRONT_TAIL_AUTO)
		prefill_front_tail = prefill_tail_rows ==
						     DS41F_TAIL_WIDEST_ROWS ?
					     FRONT_TAIL_ALL :
				     prefill_tail_rows != 0 ? 1u :
							      0u;
	/* --prefill_ladder_fine: by default where the 448-row tail is (THIN, --wide_rows of 2,048 or more); BATCH keeps
	 * 0 for a 1M slot's memory (rk10: BATCH at 1M boots, 53 MiB free on the 640-column cards, only without it) */
	if (prefill_ladder_fine == LADDER_FINE_AUTO)
		prefill_ladder_fine = wide_rows >= TAIL_LONG_WIDE_ROWS &&
						      batch_rows[0] == 0 ?
					      1u :
					      0u;
	const char *const single_knob = is_serving ? one_session_knob() : NULL;
	const char *const unreserved =
		official_reference != 0 ?
			"" :
			" without --prefill_first or --prefill_order";
	if (single_knob != NULL && slots_defaulted) {
		(void)fprintf(
			stderr,
			"serve: %s serves one session%s: --session_slots defaults to 1\n",
			single_knob, unreserved);
		session_slots_max = false;
		session_slots = 1u;
	} else if (single_knob != NULL && session_slots_max) {
		(void)fprintf(
			stderr,
			"serve: %s serves one session%s: give --session_slots=1, not max\n",
			single_knob, unreserved);
		return false;
	} else if (single_knob != NULL && session_slots > 1u) {
		(void)fprintf(
			stderr,
			"serve: %s serves one session%s: give --session_slots=1, not %" PRIu32
			"\n",
			single_knob, unreserved, session_slots);
		return false;
	}
	/* The original multi-session serving gate qualifies five PP flights. A default never turns a working serve
	 * into a refusal: serve's slots default to max (several) since U5a, and a TP serve of several slots with the
	 * device draft needs two flights unless it runs BATCH (refused below). Explicit depths (an explicit
	 * --pipeline_steps=1 keeps that refusal) and BATCH retain their own admission and selected values. */
	session_defaults(session_slots);
	/* Use the measured wide expert pass only where its existing admission allows it.
	 * Explicit zero retains the slab path, including in generated axis recipes. */
	if (!original_wide_experts_was_given &&
	    execution_placement == GPU_SPLIT_TP4_PRO &&
	    numerics == NUMERICS_EXACT && pro_emulation == 0 &&
	    cmp_order == CMP_ORDER_REFERENCE &&
	    linear_order == LINEAR_ORDER_REFERENCE &&
	    linear_profile == LINEAR_PROFILE_PRECISION &&
	    wide_rows >= STEP_WIDE_LEAST_ROWS &&
	    wide_rows <= LINEAR_WIDE_MOST_ROWS && exl3_tc_moe == 0 &&
	    tp_wire_bf16 == 0 && tp_code_wire == 0 && tp_attention_owner == 0 &&
	    batch_rows[0] == 0 && exl3_wide_entry == 0 &&
	    tp_split_overlap == 0 && tp_full_split_overlap == 0 &&
	    original_chunk_core == 0)
		original_wide_experts = 1;
	/* step I/O's default and the hot set's guard wait for the storage profile (engram_settings_resolve) */
	engram_step_io_default = !engram_step_io_was_given;
	if (engram_ram_percent == ENGRAM_RAM_PERCENT &&
	    (engram_cache_mib != 0 || turbo_nvme_engrams != 0 ||
	     engram_io != 0 || engram_sources_text != NULL)) {
		(void)fputs(
			"engram: --engram_cache_mib, --turbo-nvme-engrams, --engram_io and --engram_sources need"
			" disk tables: give --engram_ram_percent=0..99 with them (auto keeps every table in RAM"
			" when they fit)\n",
			stderr);
		return false;
	}
	if ((engram_hot_set_path == NULL) != (engram_hot_sha256 == NULL))
		return parse_refused(
			"--engram_hot_set",
			"taken only with its recorded --engram_hot_sha256, and that only with it");
	const struct parse_refusal refusals[] = {
		{ user_queue_depth > queue_depth, "--user_queue_depth",
		  "above --queue_depth" },
		{ strcmp(draft_usage, "compact") != 0 &&
			  strcmp(draft_usage, "none") != 0 &&
			  strcmp(draft_usage, "full") != 0,
		  "--draft_usage", "takes none, compact or full" },
		{ strcmp(calls_parsing, "tolerant") != 0 &&
			  strcmp(calls_parsing, "reference") != 0,
		  "--dsml_parsing", "takes tolerant or reference" },
		{ live_stats_interval_ms != 0 &&
			  live_stats_interval_ms < LIVE_STATS_LEAST_INTERVAL_MS,
		  "--live_stats", "0, or at least the least interval" },
		{ strcmp(live_stats_format, "text") != 0 &&
			  strcmp(live_stats_format, "json") != 0 &&
			  strcmp(live_stats_format, "table") != 0,
		  "--live_stats_format", "takes text, json or table" },
		{ session_slots != 1 && !is_serving, "--session_slots",
		  "serve's alone" },
		{ wide_rows != 0 && (wide_rows < STEP_WIDE_LEAST_ROWS ||
				     prefill_cone == 0),
		  "--wide_rows",
		  "0, or at least the least wide rows with --prefill_cone=1" },
		{ (prefill_chunk_rows != 0 || prefill_decode_rows != 0) &&
			  (chunks_in_flight != 1u || prefill_fast_plan != 0 ||
			   prefill_chunk_rows % STEP_FULL_GRANULE_ROWS != 0 ||
			   prefill_decode_rows % STEP_FULL_GRANULE_ROWS != 0 ||
			   prefill_chunk_rows > wide_rows ||
			   prefill_decode_rows > wide_rows),
		  "--prefill_chunk_rows and --prefill_decode_rows",
		  "whole full-cone granules up to --wide_rows, with --chunks_in_flight=1 and --prefill_fast_plan=0" },
		{ prefill_tail_rows != 0 &&
			  ((prefill_tail_rows != DS41F_TAIL_PROGRAM_ROWS &&
			    prefill_tail_rows != DS41F_TAIL_LONG_ROWS &&
			    prefill_tail_rows != DS41F_TAIL_WIDEST_ROWS) ||
			   !tail_applies),
		  "--prefill_tail_rows",
		  "0, 128, 320 or 448, on the TP4_EXL3 tail recipe alone (one chunk in flight, the exact plan, full "
		  "steps, --wide_rows of 1024 or more, --exl3_tc_moe=1 and --tp_trunk_mma=1)" },
		{ prefill_front_tail != 0 && prefill_tail_rows == 0,
		  "--prefill_front_tail", "needs --prefill_tail_rows" },
		{ prefill_thin_tail != 0 && prefill_tail_rows == 0,
		  "--prefill_thin_tail", "needs --prefill_tail_rows" },
		{ dense_staged != 0 &&
			  (dense_staged < DENSE_STAGED_LEAST_OUTPUTS ||
			   (dense_staged & (dense_staged - 1u)) != 0),
		  "--dense_staged",
		  "0, or a power of two of at least the least staged outputs" },
		{ original_dense_staged != 0 &&
			  (original_dense_staged != DENSE_STAGED_OUTPUTS ||
			   execution_placement != GPU_SPLIT_TP4_PRO),
		  "--original_dense_staged",
		  "the staged outputs' count, on TP4_PRO alone" },
		{ tp_pipeline_overlap != 0 &&
			  (execution_placement != GPU_SPLIT_TP4_PRO ||
			   pipeline_steps < OVERLAP_LEAST_PIPELINE_STEPS ||
			   batch_rows[0] != 0 || numerics != NUMERICS_EXACT ||
			   pro_emulation != 0 ||
			   cmp_order != CMP_ORDER_REFERENCE ||
			   linear_order != LINEAR_ORDER_REFERENCE ||
			   linear_profile != LINEAR_PROFILE_PRECISION ||
			   tp_wire_bf16 != 0 || tp_code_wire != 0 ||
			   tp_attention_owner != 0),
		  "--tp_pipeline_overlap",
		  "TP4_PRO's exact reference recipe at two pipeline steps or more, without BATCH or another wire" },
		{ original_wide_experts != 0 &&
			  (execution_placement != GPU_SPLIT_TP4_PRO ||
			   numerics != NUMERICS_EXACT || pro_emulation != 0 ||
			   cmp_order != CMP_ORDER_REFERENCE ||
			   linear_order != LINEAR_ORDER_REFERENCE ||
			   linear_profile != LINEAR_PROFILE_PRECISION ||
			   wide_rows < STEP_WIDE_LEAST_ROWS ||
			   tp_wire_bf16 != 0 || tp_code_wire != 0 ||
			   tp_attention_owner != 0 || batch_rows[0] != 0 ||
			   exl3_wide_entry != 0 || tp_split_overlap != 0 ||
			   tp_full_split_overlap != 0),
		  "--original_wide_experts",
		  "TP4_PRO's exact reference recipe with wide rows, without BATCH, another wire or the split "
		  "overlaps" },
		{ original_batch != 0 &&
			  (execution_placement != GPU_SPLIT_TP4_PRO ||
			   pipeline_steps != 1u ||
			   batch_rows[0] != DS41F_ORIGINAL_BATCH_ROWS ||
			   batch_rows[1] != 0 ||
			   batch_verifiers[0] != DS41F_ORIGINAL_BATCH_SESSIONS ||
			   batch_verifiers[1] != 0 ||
			   session_slots != DS41F_ORIGINAL_BATCH_SESSIONS ||
			   wide_rows < STEP_WIDE_LEAST_ROWS ||
			   speculative < SPECULATIVE_DEVICE ||
			   draft_block != DS41F_DRAFT_BLOCK ||
			   draft_positions != DS41F_DRAFT_BLOCK ||
			   numerics != NUMERICS_EXACT || pro_emulation != 0 ||
			   linear_profile != LINEAR_PROFILE_PRECISION ||
			   linear_order != LINEAR_ORDER_REFERENCE ||
			   cmp_order != CMP_ORDER_REFERENCE ||
			   tp_wire_bf16 != 0 || tp_code_wire != 0 ||
			   tp_attention_owner != 0 || tp_head_shards != 0 ||
			   tp_draft_shards != 0 || tp_draft_layers != 0 ||
			   tp_head_stream != 0 || tp_head_mma != 0 ||
			   tp_trunk_mma != 0 || tp_router_mma != 0 ||
			   batch_wide_head != 0 || batch_draft_rows != 0 ||
			   batch_context_slabs != 1u || batch_core_once != 0 ||
			   batch_draft_wide != 0 || draft_bf16_rows != 0 ||
			   draft_bf16_operands != 0 ||
			   original_wide_experts != 0 ||
			   original_dense_staged != 0 || exl3_wide_entry != 0 ||
			   tp_split_overlap != 0 ||
			   tp_full_split_overlap != 0 ||
			   tp_pipeline_overlap != 0),
		  "--original_batch",
		  "TP4_PRO's exact BATCH recipe alone: --pipeline_steps=1, one BATCH program of its rows and sessions, "
		  "the device draft at the draft block, and none of the TP, draft and BATCH form knobs" },
		{ original_chunk_core != 0 &&
			  wide_rows % LINEAR_MAXIMUM_ROWS != 0,
		  "--original_chunk_core",
		  "needs --wide_rows a multiple of the linear's rows" },
		{ original_chunk_core != 0 &&
			  (execution_placement != GPU_SPLIT_TP4_PRO ||
			   numerics != NUMERICS_EXACT || pro_emulation != 0 ||
			   linear_profile != LINEAR_PROFILE_PRECISION ||
			   linear_order != LINEAR_ORDER_REFERENCE ||
			   cmp_order != CMP_ORDER_REFERENCE ||
			   wide_rows < STEP_WIDE_LEAST_ROWS ||
			   wide_rows > LINEAR_WIDE_MOST_ROWS ||
			   tp_wire_bf16 != 0 || tp_code_wire != 0 ||
			   tp_attention_owner != 0 || batch_rows[0] != 0 ||
			   original_wide_experts != 0 || original_batch != 0 ||
			   tp_split_overlap != 0 || tp_full_split_overlap != 0),
		  "--original_chunk_core",
		  "TP4_PRO's exact reference recipe with wide rows in the linear's range, without BATCH, another wire, "
		  "--original_wide_experts or the split overlaps" },
		{ lookup_ngram_min > lookup_ngram_max, "--lookup_ngram_min",
		  "above --lookup_ngram_max" },
		{ speculative != SPECULATIVE_OFF && step_rows == 1, "--rows",
		  "speculation needs more than one row" },
		{ speculative >= SPECULATIVE_DEVICE && session_slots > 1u &&
			  pipeline_steps == 1u && batch_rows[0] == 0,
		  "--pipeline_steps",
		  "several sessions with the device draft need two pipeline steps or more, or BATCH" },
		{ batch_rows[0] != 0 &&
			  ((!original_batch &&
			    execution_placement != GPU_SPLIT_TP4_EXL3) ||
			   pipeline_steps != 1u),
		  "--batch_rows",
		  "BATCH runs on TP4_EXL3 (or with --original_batch=1) at --pipeline_steps=1" },
		/* without the batched draft rows each verifier's walk is its own segments: the step driver runs one
		 * BATCH program beside the chunked ones, not several (ds41f_exl3_program_test) */
		{ batch_rows[1] != 0 && speculative >= SPECULATIVE_DEVICE &&
			  batch_draft_rows == 0,
		  "--batch_rows",
		  "several BATCH programs with the device draft need --batch_draft_rows=1" },
		{ batch_row_sets % LINEAR_MAXIMUM_ROWS != 0, "--batch_row_sets",
		  "a multiple of the linear's rows" },
		{ tp_core_heads % CORE_HEADS_STEP != 0, "--tp_core_heads",
		  "a multiple of the attention core's tile of heads" },
		{ (batch_draft_rows != 0 || batch_wide_head != 0 ||
		   batch_context_slabs != 0 || batch_step_us[0] != 0 ||
		   batch_verifiers[0] != 0) &&
			  batch_rows[0] == 0,
		  "--batch_draft_rows, --batch_wide_head, --batch_context_slabs, --batch_step_us and --batch_verifiers",
		  "BATCH's own: give --batch_rows" }
	};
	if (!none_refused(refusals, sizeof(refusals) / sizeof(refusals[0])))
		return false;
	const struct startup_policy_request policy_request = {
		.numerics = numerics,
		.placement = execution_placement,
		.cmp_order = cmp_order,
		.numerics_was_given = numerics_was_given,
		.cmp_order_was_given = cmp_order_was_given
	};
	struct startup_policy policy;
	if (startup_policy_prepare(&policy_request, &policy) != STATUS_OK)
		return parse_refused(
			"--numerics",
			"exact needs the reference CMP order and runs on PP5, TP4_PRO and TP4_EXL3 alone (the TP2 and "
			"TP5 splits take relaxed or fast)");
	if (execution_placement != GPU_SPLIT_PP5 && layout_text != NULL)
		return parse_refused("--layout", "PP5's alone");
	numerics = policy.numerics;
	cmp_order = policy.cmp_order;
	if (original_head_offload != 0 &&
	    (!is_generating || execution_placement != GPU_SPLIT_PP5 ||
	     pro_emulation != 0 || official_reference != 0 ||
	     official_hc_rows != 0)) {
		(void)fputs(
			"original_head_offload requires native original PP5 and the ordinary head recipe\n",
			stderr);
		return false;
	}
	if (official_reference != 0 &&
	    (execution_placement != GPU_SPLIT_PP5 || step_rows != 1u ||
	     wide_rows != 0 || full_steps != 0 ||
	     context_tokens > DS41F_OFFICIAL_REFERENCE_CONTEXT_LIMIT ||
	     session_slots != 1u || speculative != SPECULATIVE_OFF ||
	     batch_rows[0] != 0 || original_batch != 0 ||
	     pipeline_steps != 1u || chunks_in_flight != 1u ||
	     official_hc_rows != 1u || pro_emulation != 0 ||
	     numerics != NUMERICS_EXACT || cmp_order != CMP_ORDER_REFERENCE ||
	     linear_profile != LINEAR_PROFILE_PRECISION ||
	     linear_order != LINEAR_ORDER_REFERENCE ||
	     tp_pipeline_overlap != 0 || tp_wire_bf16 != 0 ||
	     tp_code_wire != 0 || tp_attention_owner != 0 ||
	     original_chunk_core != 0 || original_dense_staged != 0 ||
	     original_wide_experts != 0 ||
	     (is_serving && (prefix_reuse != 0 || prefix_reuse_exact != 0 ||
			     session_fork != 0)))) {
		(void)fputs(
			"official_reference requires original PP THIN serial M1, "
			"reference numerics and no speculation; serve requires "
			"prefix_reuse=0, prefix_reuse_exact=0 and session_fork=0\n",
			stderr);
		return false;
	}
	const struct parse_refusal tensor_refusals[] = {
		{ exl3_expert_parts != EXL3_EXPERT_REFERENCE_PARTS &&
			  exl3_expert_parts != EXL3_EXPERT_OPTIONAL_PARTS,
		  "--exl3_expert_parts", "4 (the reference partition) or 8" },
		{ exl3_expert_parts == EXL3_EXPERT_OPTIONAL_PARTS &&
			  (execution_placement != GPU_SPLIT_TP4_EXL3 ||
			   numerics == NUMERICS_EXACT ||
			   cmp_order != CMP_ORDER_REFERENCE ||
			   tp_wire_bf16 != 0),
		  "--exl3_expert_parts",
		  "8 parts need TP4_EXL3 with relaxed or fast numerics, the reference CMP order and the FP32 wire" },
		{ execution_placement == GPU_SPLIT_TP4_EXL3 &&
			  (tp_code_wire != 0 || tp_attention_owner != 0),
		  "--tp_code_wire and --tp_attention_owner",
		  "not on TP4_EXL3" },
		{ (tp_head_shards != 0 || tp_head_stream != 0) &&
			  execution_placement != GPU_SPLIT_TP4_EXL3,
		  "--tp_head_shards and --tp_head_stream", "TP4_EXL3's alone" },
		{ tp_draft_shards != 0 && tp_head_shards == 0,
		  "--tp_draft_shards", "needs --tp_head_shards=1" },
		{ tp_decode_round_once != 0 &&
			  (execution_placement != GPU_SPLIT_TP4_EXL3 ||
			   tp_wire_bf16 == 0 ||
			   tp_wide_exchange < DS41F_TP_EXCHANGE_OWNER_SUMS ||
			   wide_rows <= LINEAR_MAXIMUM_ROWS),
		  "--tp_decode_round_once",
		  "needs TP4_EXL3 with the BF16 wire, an owner-sum --tp_wide_exchange and wide rows above the "
		  "linear's" },
		{ exl3_expert_rotation != 0 &&
			  (execution_placement != GPU_SPLIT_TP4_EXL3 ||
			   tp_wire_bf16 == 0 || exl3_tc_moe == 0 ||
			   exl3_expert_parts != EXL3_EXPERT_REFERENCE_PARTS),
		  "--exl3_expert_rotation",
		  "needs TP4_EXL3 with the BF16 wire, --exl3_tc_moe=1 and the reference expert parts" },
		{ tp_draft_layers != 0 &&
			  (tp_draft_shards == 0 || tp_wire_bf16 == 0 ||
			   execution_placement != GPU_SPLIT_TP4_EXL3 ||
			   (batch_rows[0] != 0 &&
			    (batch_draft_rows == 0 || batch_wide_head == 0))),
		  "--tp_draft_layers",
		  "needs TP4_EXL3 with the BF16 wire and --tp_draft_shards=1 (BATCH: also --batch_draft_rows=1 and "
		  "--batch_wide_head=1)" }
	};
	if (!none_refused(tensor_refusals,
			  sizeof(tensor_refusals) / sizeof(tensor_refusals[0])))
		return false;
	if (session_fork != 0 && cache_scope != SERVER_CACHE_SESSION)
		return parse_refused("--session_fork", "needs --cache_scope=0");
	if (prefill_prefix_stable != 0 &&
	    (execution_placement != GPU_SPLIT_TP4_EXL3 || tp_wire_bf16 == 0 ||
	     tp_ring_column_owners == 0 ||
	     tp_wide_exchange != DS41F_TP_EXCHANGE_CE_RINGS ||
	     tp_gather_balance != 0 ||
	     (tp_full_split_overlap != 0 &&
	      tp_full_split_overlap != tp_split_overlap) ||
	     batch_thin_bits == 0 || full_steps == 0 || prefill_cone == 0 ||
	     prefill_cone_skip != 1u || chunks_in_flight != 1u ||
	     prefill_fast_plan != 0 || prefill_chunk_rows != 0 ||
	     prefill_decode_rows != 0 || session_directory != NULL ||
	     prefill_margin < SERVING_STABLE_GRANULE - 1u ||
	     (wide_rows != PREFIX_STABLE_SMALL_ROWS &&
	      wide_rows != PREFIX_STABLE_LARGE_ROWS) ||
	     (session_slots > 1u && prefill_first == 0 && prefill_order == 0)))
		return parse_refused(
			"--prefill_prefix_stable",
			"the TP4_EXL3 BF16 column-owner THIN recipe alone: the CE-ring exchange, unweighted "
			"gathers, thin bits, full steps, the cone at skip 1, one chunk in flight, the exact "
			"plan, no prefill caps or session directory, a margin of a stable granule, --wide_rows "
			"of 1024 or 2048, and reserved prefill for several slots");
	if ((prefix_reuse_exact != 0 || session_fork != 0) &&
	    !prefix_cold_plan_is_qualified())
		return parse_refused(
			session_fork != 0 ? "--session_fork" :
					    "--prefix_reuse_exact",
			"needs TP4_EXL3 column owners, the invariant THIN forms, --chunks_in_flight=1, the exact plan, "
			"prefill caps 0 and, for several slots, reserved prefill (--prefill_first or --prefill_order)");
	/* The measured CLI defaults belong to the EXL3 family. Other placements retain each omitted
	 * field's previous default; explicit choices remain independent. */
	if (execution_placement != GPU_SPLIT_TP4_EXL3) {
		struct draft_policy_record previous;
		draft_policy_default(&previous);
		if (!draft_policy_was_given)
			draft_policy.depth_policy = previous.depth_policy;
		if (!draft_learning_was_given)
			draft_learned_acceptance = 0;
		if (!draft_base_was_given)
			draft_policy.base_us = previous.base_us;
		if (!draft_tail_was_given)
			draft_policy.tail_us = previous.tail_us;
		if (!draft_row_was_given)
			draft_policy.row_us = previous.row_us;
		if (!batch_core_was_given)
			batch_core_once = DEFAULT_OTHER_BATCH_CORE_ONCE;
	} else if (!draft_policy_was_given) {
		/* Explicit policies keep their no-menu behavior. Unpriced EXL3 shapes keep the rate
		 * objective (session_defaults) without inventing timings; supplied prices are always retained. */
		const bool measured_batch =
			execution_placement == GPU_SPLIT_TP4_EXL3 &&
			tp_wire_bf16 == ORDERED_PUSH_BF16 &&
			speculative == SPECULATIVE_DEVICE &&
			draft_block == DS41F_DRAFT_BLOCK &&
			draft_positions == DS41F_DRAFT_BLOCK &&
			pipeline_steps == 1u && exl3_tc_moe == 1u &&
			tp_trunk_mma == 1u && tp_head_mma == 1u &&
			tp_head_shards == 1u && tp_head_stream == 1u &&
			tp_draft_shards == 1u && tp_draft_layers == 1u &&
			batch_wide_head == 1u && batch_draft_rows == 1u &&
			batch_thin_bits == 1u &&
			batch_draft_wide == DEFAULT_BATCH_DRAFT_WIDE &&
			linear_order == LINEAR_ORDER_REFERENCE &&
			cmp_order == CMP_ORDER_REFERENCE &&
			tp_decode_round_once == 0 &&
			exl3_expert_rotation == 0 &&
			exl3_tc_product_blocks == 0 &&
			tp_trunk_mma_fused == 0 && batch_tc_paired == 0 &&
			batch_row_sets == 0 &&
			batch_grouped_rows == DEFAULT_BATCH_GROUPED_ROWS &&
			tp_graph_branches == 1u && tp_launch_trims == 1u &&
			batch_core_once >= LEAST_MEASURED_BATCH_CORE_ONCE;
		if (measured_batch && !batch_costs_were_given &&
		    memcmp(batch_rows, ds41f_default_batch_rows,
			   sizeof(batch_rows)) == 0 &&
		    memcmp(batch_verifiers, ds41f_default_batch_verifiers,
			   sizeof(batch_verifiers)) == 0)
			memcpy(batch_step_us, ds41f_default_batch_step_us,
			       sizeof(ds41f_default_batch_step_us));
	}
	/* generate's messages stay positional under --model_dir (or --prompt_token_ids replaces them) */
	if (model_dir != NULL && is_generating && !is_serving &&
	    front_end_paths[MESSAGES_FILE] == NULL &&
	    prompt_token_ids_path == NULL)
		return parse_refused(
			"--model_dir",
			"generate still takes MESSAGES, or --prompt_token_ids");
	/* --model_dir gives one copy once the folder is resolved; the calibration file is read last, as before */
	const struct parse_refusal file_refusals[] = {
		{ copy_count == 0 && model_dir == NULL, "the pack",
		  "none given (or --model_dir)" },
		{ prompt_token_ids_path != NULL && copy_count != 1u &&
			  model_dir == NULL,
		  "--prompt_token_ids", "takes one pack" },
		{ copy_count > STORAGE_MAXIMUM_COPIES, "the packs",
		  "more copies than a load reads" },
		{ speculative == SPECULATIVE_HYBRID &&
			  hybrid_lookup_min > lookup_ngram_max,
		  "--hybrid_lookup_min", "above --lookup_ngram_max" }
	};
	if (!none_refused(file_refusals,
			  sizeof(file_refusals) / sizeof(file_refusals[0])))
		return false;
	if (!draft_calibration_read())
		return parse_refused(
			"--draft_calibration",
			"the file does not read as the calibration's probabilities");
	return draft_policy_valid(&draft_policy) ||
	       parse_refused("--draft_policy",
			     "outside the draft policy's ranges");
}
