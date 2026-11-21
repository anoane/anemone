/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* commands.c: load/generate/serve, which main.c's main runs; run_program prints exact CLI usage.
 * Unique decimal --name=value options precede files; invalid usage refuses before load.
 * One plan owns cards/weights/state. Generate prints tokens; serve follows server.h.
 * Every exit closes cards before host arena release. Numerical/ownership gates remain required.
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
/* last, as commands.h says */
#include "commands.h"
#include "cli_golden_capture.h"
/* REUSE-IgnoreStart: the tag below is the printed notice's, not this file's */
#define PROGRAM_ATTRIBUTION_NOTICE                                                               \
	"Anemone v0.0.6-alpha\nCopyright (C) 2026 Fabiano Anemone <fabiano.anemone@gmail.com>\n" \
	"SPDX-License-Identifier: AGPL-3.0-or-later\n" /* REUSE-IgnoreEnd */
#define NANOSECONDS_PER_SECOND 1e9
#define BYTES_PER_GIB 1073741824.0
#define NANOSECONDS_PER_MICROSECOND 1000ull
#define NANOSECONDS_PER_MILLISECOND 1000000ull
#define AGE_QUANTILE_PARTS 1000u
#define AGE_QUANTILE_PART \
	999u /* step I/O's shutdown line: the 99.9 % read age */
#define DECIMAL_BASE 10
#define MOST_CARD_MEMORY_KEPT_FREE_MIB \
	1048576 /* a TiB: above any card, and its bytes fit in 64 bits */
#define BYTES_PER_GB 1e9
#define DEFAULT_TP_LOAD_THREADS \
	32u /* the TP load's workers for the cards (rh04, 4 Nov: 31.1 s against 33.9 at 16) */
#define DEFAULT_TP_LOAD_HOST_THREADS \
	TP_LOAD_HOST_THREADS_DEFAULT /* and the host arena's threads beside them */
#define DEFAULT_TP_LOAD_READS_PER_COPY TP_LOAD_READS_PER_COPY_DEFAULT
#define MAP_READS_AUTO \
	EXL3_READ_MODES /* --map_reads beyond the storage's modes: chosen by the copies found */
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
#define DRAFT_RING_DUMP_ROWS \
	512u /* the dump's window rows of session 0, positions 0 .. 511, a draft stage */
#define DIGEST_MIX_PRIME \
	1099511628211ull /* FNV-1a 64: mixes a layer's tensor digests in order */
#define PIPE_ENDS 2 /* of a pipe: read, write */
#define STOP_SIGNALS 3 /* of serve: SIGINT, SIGTERM, SIGHUP */
#define CHUNK_LADDER_TOP \
	1024u /* above it, wide_rows then 512, 256 and 64 rows (scheduler.c's exact plan: whole multiples of 256) */
#define CHUNK_ROWS_DIVISOR \
	4u /* wide and full programs of wide_rows, a quarter of it, ..., down to the least wide rows */
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
/* --prefill_tail_rows: a full program of two of the exact plan's 64-row tail chunks, beside the ladder's (BATCH
 * programs take the thin ones' place, so MOST_PROGRAMS holds it), with a ladder of no 128-row program of its own */
#define TAIL_PROGRAM_ROWS 128u
#define TAIL_LONG_ROWS \
	320u /* --prefill_tail_rows=320: a full program of five tail chunks beside the 128-row one */
#define TAIL_WIDEST_ROWS \
	448u /* --prefill_tail_rows=448: and one of seven beside the 320-row one */
#define TAIL_LEAST_WIDE_ROWS 1024u
#define FRONT_TAIL_ALL \
	2u /* --prefill_front_tail=2: the front region's tail takes the full region's tail programs */
#define THIN_TAIL_KINDS \
	2u /* --prefill_thin_tail with FRONT_TAIL_ALL: full and wide thin-tail programs */
#define THIN_TAIL_LEAST_ROWS \
	16u /* --prefill_thin_tail: full programs of 16 to 56 rows, 8 apart (thin steps' forms) */
#define THIN_TAIL_AUTO \
	UINT32_MAX /* --prefill_thin_tail not given: 1 with the tail program and no BATCH (a BATCH step
				    takes the rows short of a 64-row chunk with the row of logits in one; EVIDENCE) */
#define LADDER_1536_ROWS \
	1536u /* --prefill_ladder_1536: six whole granules of the exact plan (scheduler.c) */
#define LADDER_FINE_COUNT \
	3u /* --prefill_ladder_fine: 1,792, 1,280 and 768 rows (7, 5 and 3 whole granules) */
#define PREFILL_CHUNK_CAPS \
	2u /* global and decoder-only caps may each need another prepared chunk size */
#define LADDER_INSERTED_SIZES \
	2u /* --prefill_ladder_1024's and --prefill_ladder_1536's sizes, inserted behind wide_rows */
#define LADDER_SECOND 1u /* chunk_rows' place behind wide_rows */
#define LADDER_HALVING \
	2u /* above CHUNK_LADDER_TOP: wide_rows, then 512 and 256 by halving */
#define OVERLAP_LEAST_PIPELINE_STEPS \
	2u /* --tp_pipeline_overlap: a step beside the next */
#define ROUTER_MMA_WIDE 2u /* --tp_router_mma=2 */
#define DRAFT_LAST_STAGE \
	2u /* the dump's third draft stage ring (DS41F_DRAFT_STAGES) */
/* the dump's four pieces of the draft logits' first 2 MiB */
enum draft_logits_piece {
	DRAFT_LOGITS_FIRST,
	DRAFT_LOGITS_SECOND,
	DRAFT_LOGITS_THIRD,
	DRAFT_LOGITS_FOURTH
};
#define LADDER_FINE_AUTO \
	UINT32_MAX /* --prefill_ladder_fine not given: 1 with TAIL_LONG_WIDE_ROWS or more, no BATCH */
#define TAIL_ROWS_AUTO \
	UINT32_MAX /* --prefill_tail_rows not given: TAIL_PROGRAM_ROWS where the tail program applies,
				    TAIL_WIDEST_ROWS (and so TAIL_LONG_ROWS too) where also --wide_rows is
				    TAIL_LONG_WIDE_ROWS or more and no BATCH, TAIL_LONG_ROWS with BATCH (gsk03: with the
				    tail overlay the 640-column cards keep 341 MiB free, cold prefill +2.3%) */
#define TAIL_LONG_WIDE_ROWS \
	2048u /* --prefill_tail_rows=320's default (ri03), 448's (gsk01: identical, the 2K cell
				     1,696 -> 1,722 tok/s, cold prefill 3,798 -> 3,818) with 2,048-row chunks */
#define FRONT_TAIL_AUTO \
	UINT32_MAX /* --prefill_front_tail not given: 1 where the tail program applies, FRONT_TAIL_ALL
				      with the 448-row tail (THIN; rk04k: identical, the 30K prompt 8.214 -> 8.190 s) */
#define DEAD_ROWS_TRIMMED \
	2u /* --prefill_dead_halves: the sums' copies of the rows the cone reads alone too */
#define DEFAULT_CHUNKS_IN_FLIGHT \
	1u /* 1: scheduler.c's exact plan (the 1,024-row two-in-flight reference's bits in the largest chunks;
	      rc02ab 30K prompt 11.859 -> 9.430 s, Fraser 2.431 -> 1.857 s, identical); else so many chunks at least */
#define DEFAULT_PREFILL_MARGIN \
	127u /* the exact plan's full rows beyond a prompt's cone (scheduler.c; rc02ab: the same bits, -1.0 s at 30K) */
#define DEFAULT_LOOKUP_NGRAM_MOST \
	4u /* the lookup drafter's longest suffix, in tokens */
#define DEFAULT_LOOKUP_NGRAM_LEAST 2u
#define DRAFT_LOOKUP_OPTION "--draft=lookup"
#define DRAFT_DEVICE_OPTION "--draft=dspark"
#define DRAFT_HYBRID_OPTION "--draft=hybrid"
#define DEFAULT_CODE_DRAFT_BASE_US 14300u
#define DEFAULT_CODE_DRAFT_ROW_US 690u
#define DRAFT_CALIBRATION_WORD_BYTES 8u
#define DRAFT_CALIBRATION_FILE_BYTES 4096u
#define DRAFT_CALIBRATION_SPACES " \t\r\n\v\f"
#define SALT_OFFSET \
	0xCBF29CE484222325ull /* FNV-1a, 64 bits: a snapshot's arithmetic salt */
#define SALT_PRIME 0x100000001B3ull
#define CMP_SALT_TAG \
	0x434D505F4F524445ull /* separates relaxed CMP placement from the native PRO's salt */
#define HOST_EXPERT_SALT_TAG \
	0x484F53545F455850ull /* HOST_EXP: host-expert layers keep reference expert order */
#define STATS_LAYER_MASK_BITS 64u
#define STATS_HEAD_LANGUAGE 1ull
#define STATS_HEAD_DRAFT 2ull
#define STATS_HEAD_CONFIDENCE 4ull
#define EXL3_EXPERT_REFERENCE_PARTS 4u
#define EXL3_EXPERT_OPTIONAL_PARTS 8u

enum command_kind {
	TAKEN_BY_LOAD = 1,
	TAKEN_BY_GENERATE = 2,
	TAKEN_BY_SERVE = 4,
	TAKEN_BY_ALL = 7
};

const char *const *copy_paths; /* the copies of the pack, as typed */
static const char *
	pack_argument; /* the first pack argument as typed (its descriptor's place), or NULL */
const char *front_end_paths
	[FRONT_END_FILES]; /* by enum front_end_file, as typed; NULL: none */
uint32_t copy_count;
/* --weights: unset keeps the family that --placement and the pack imply; --model_dir: the folder whose fixed names
 * (cli_model_folder.h) give the tables and the pack, each with its real path and its device and inode */
enum weights_family weights_family;
const char *model_dir;
static const char *
	exl3_map_name; /* --exl3_map=NAME: the folder's EXL3 map is NAME.exl3map; NULL: model.exl3map */
static char exl3_map_file[CLI_MODEL_FOLDER_NAME_BYTES] =
	CLI_MODEL_FOLDER_DEFAULT_MAP
		CLI_MODEL_FOLDER_MAP_SUFFIX; /* its file name in the folder */
struct model_file_record model_files[CLI_MODEL_FILES];
const struct model_file_record *model_pack; /* the folder's pack, or NULL */
static const char *placement_text; /* as typed */
bool is_generating; /* generate or serve */
static bool is_serving; /* serve */
static bool
	is_capturing; /* capture: generate's path, cli_golden_capture.h's front end */
uint32_t card_order
	[BACKEND_MAXIMUM_CARDS]; /* backend cards, in the order the stages fill them */
uint32_t card_count; /* 0 until the cards are counted: all, as enumerated */
uint32_t reader_threads_per_copy = LOADER_DEFAULT_READER_THREADS_PER_COPY;
/* the TP placements' load: 0 the serial loads (the reference), N the host arena beside the cards, each on N threads */
static uint32_t tp_load_threads = DEFAULT_TP_LOAD_THREADS;
/* with tp_load_threads: reads in flight a copy at most (0: no limit) */
static uint32_t tp_load_reads_per_copy = DEFAULT_TP_LOAD_READS_PER_COPY;
/* with tp_load_threads: the host arena's own reads a copy at most, from an applied load section (0: no limit) */
static uint32_t tp_load_host_reads_per_copy = DEFAULT_TP_LOAD_READS_PER_COPY;
/* with tp_load_threads: the host arena's threads (0: as many as the cards' load) */
static uint32_t tp_load_host_threads = DEFAULT_TP_LOAD_HOST_THREADS;
/* with tp_load_threads and copies (tp_load.h, enum tp_load_split): 2 (default) host-arena chunks prefer copy 0 and
 * card extents the last copy, and a role that has claimed its last read leaves its copy to the other; 1: each role
 * only its preferred copy (tcb96's default); 0: each read from the copy soonest to finish it */
static uint32_t tp_load_copy_split = TP_LOAD_SPLIT_STEAL;
/* the load's read size in MiB, both loaders' (the PP loader's piece and staging slot, the TP host arena's chunk); 0
 * (default): LOADER_CHUNK_BYTES and TP_HOST_LOAD_CHUNK_BYTES, 16 MiB. The knob wins over a profile, a profile over
 * the default. */
static uint32_t load_piece_mib;
/* The TP host arena's read (load_section_apply): a knob given wins, then the storage profile's enabled load section
 * when it names this pack and exactly these copies, then the default; tp_load_reads_per_copy holds the readers. */
static uint64_t tp_load_piece_bytes;
static bool explicit_tp_reads_per_copy, explicit_load_piece;
static char
	storage_profile_resolved[REPLICA_PATH_CAPACITY]; /* empty: none, or off */
/* 1 (default): a single source gains its receipted replicas on other drives as copies (replicas.h); 0: it alone */
static uint32_t turbo_load = 1;
/* 1: a storage profile's per-copy load weights pick each piece's copy (a static split, for comparison); 0: the
 * readers of every copy steal the next piece, so a faster drive reads more */
static uint32_t load_tuned_weights;
/* the tables' RAM share: given (the knob, or a storage profile's), else auto, which the fit (engram_fit.h) decides
 * before the load; it reads ENGRAM_RAM_PERCENT until then */
uint32_t engram_ram_percent = ENGRAM_RAM_PERCENT;
static enum engram_ram_source engram_ram_source = ENGRAM_RAM_AUTO;
#define ENGRAM_RAM_AUTO_OPTION "--engram_ram_percent=auto"
/* 1: generate prints the Engram fit's lines and stops before the first card is touched (a diagnostic) */
static uint32_t engram_fit_only;
static uint32_t replica_external,
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
static uint32_t engram_spin_us = ENGRAM_STEP_SPIN_US_DEFAULT;
uint32_t engram_io_pin = 1;
/* Step I/O's parameters given as knobs (each wins over the profile's engram section), and the disk store's admitted
 * copies in their order (engram_sources_prepare), which the section must name by their device keys. */
static bool explicit_engram_threads, explicit_engram_spin, explicit_engram_pin;
static char engram_copy_paths[STORAGE_MAXIMUM_COPIES][REPLICA_PATH_CAPACITY];
static uint32_t engram_copy_count;
static uint32_t engram_scan_threads = ENGRAM_DISK_SCAN_THREADS_DEFAULT;
struct engram_steps *engram_steps;
struct engram_disk *engram_disk;
const char *engram_sources_text;
static const char *storage_profile_path;
/* a static hot set of table blocks (engram_hot.h): its file, its recorded SHA-256 and its budget */
static const char *engram_hot_set_path, *engram_hot_sha256;
static uint32_t engram_hot_mib = ENGRAM_HOT_BUDGET_DEFAULT_MIB;
static struct engram_hot *engram_hot;
static struct storage_profile tuned_storage;
static char storage_replica_info[STORAGE_PROFILE_METRICS_BYTES];
static size_t storage_replica_info_bytes;
bool tuned_load_active;
static bool tuned_engram_active, storage_profile_is_valid;
/* Profile loading turned off explicitly (--storage_profile=off or the environment): no autotune hint. */
static bool storage_profile_is_off;
static const char *tuned_paths[STORAGE_MAXIMUM_COPIES];
static const char *tuned_load_paths[STORAGE_MAXIMUM_COPIES];
static uint32_t tuned_load_weights[STORAGE_MAXIMUM_COPIES];
static char tuned_engram_sources[STORAGE_MAXIMUM_COPIES * REPLICA_PATH_CAPACITY];
static struct storage_reader *engram_source_reader;
/* an EXL3 map's reads (enum exl3_read_mode, storage_exl3.h): 0 buffered, 1 buffered then dropped from the
 * page cache, 2 direct where aligned (rh01, 3 Nov: TP4 loads 36.7-38.4 / 29.8-31.5 / 32.3 s), 3 direct where
 * aligned and dropped for the rest; MAP_READS_AUTO (default): 3 when replicas were found (rh06 / rh08, 4 Nov, with
 * the copy split: 29.7 s against 31.0), else 1 (one copy: rh08t 62.5 s with 3 against 45.9 with 1) */
static uint32_t map_reads = MAP_READS_AUTO;
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
static uint32_t tp_exact_dense =
	1; /* tp4_exl3 exact recipe: attention output shards (xdense01: same bits) */
uint32_t tp_draft_shards;
uint32_t tp_draft_layers; /* tp4_exl3 BF16 recipe, thin: the draft's layers tensor-parallel (GNmac's DSpark) */
uint32_t tp_head_stream;
static uint32_t exl3_fast_products =
	1; /* kernel_exl3_wide's products (identical words); 0 the retained chain */
static uint32_t exl3_wide_entry =
	1; /* chunked steps run an EXL3 layer's experts once over the chunk (fast01/02) */
static uint32_t
	exl3_tc_moe; /* tp4_exl3 BF16 recipe: GNmac's tensor-core EXL3 MoE (exl3_tc.h; GNmac numerics) */
static uint32_t
	exl3_tc_product_blocks; /* decode product grid override; 0 keeps the measured 148-block grid */
#define TC_PRODUCT_BLOCKS_MOST \
	592u /* eight product blocks per CMP multiprocessor */
static uint32_t
	exl3_expert_rotation; /* with exl3_tc_moe: per-expert chunk rotation 1 or 2 (exl3_tc_records.h) */
static uint32_t
	exl3_unrated_map; /* 1: an EXL3 pack with no quality record is served: its descriptor's none, or no descriptor
			   * and a pack the model does not list (exl3_descriptor.h) */
static uint32_t
	tp_head_mma; /* tp4_exl3 BF16 recipe: the head products on the tensor cores (kernel_head_mma) */
static uint32_t
	tp_trunk_mma; /* tp4_exl3 BF16 recipe: thin BF16-row trunk products on the tensor cores (GNmac) */
static uint32_t
	tp_trunk_mma_fused; /* trunk_mma's _64 form adds its own splits (same bits; relbench03: 1-5 us slower) */
static uint32_t
	tp_trunk_mma_thin_splits; /* trunk_mma's _64 form splits as the thin form: the thin steps' bits */
static uint32_t
	tp_router_mma; /* tp4_exl3 BF16 recipe: a BATCH step's router by split-K tensor-core MMAs (GNmac's) */
static uint32_t
	tp_gather_balance; /* tp4_exl3: a chunk's index scans and router rows on the narrower expert ranks */
static uint32_t tp_pool_chunk =
	1; /* tp4_exl3: a chunk's pool entries and keys ahead of its slabs' scans (same bits;
				       rc02ab 30K prompt 9.828 -> 9.430 s); 0: in the slabs' order */
static uint32_t tp_pool_gather =
	1; /* tp4_exl3: a pooled source layer's compressor products a quarter of a half's
				       rows a rank, gathered (the same bits; ri01); 0: every row on every rank */
static uint32_t
	tp_router_first_whole; /* tp4_exl3: the first half's router over its rows on every rank, no gather */
static uint32_t exl3_tcp_two_blocks =
	1; /* the prefill TC MoE's GEMMs at two blocks a multiprocessor (the same bits;
					    ri03: a 2,048-row call -15%, Fraser first token -17 ms); 0: one block */
static uint32_t tp_queries_piece =
	1; /* tp4_exl3: a wide program's queries over each piece with wq_b (the same bits;
					 ri05 30K -77 ms); 0: a span at a time from the heads copied into the full
					 rows */
static uint32_t tp_index_span =
	1; /* tp4_exl3: a chunk's index scans and selections a span of rows a launch (same
				      bits; ri05 30K -51 ms); 0: a slab of 8 rows a launch */
static uint32_t tp_core_compact_halves =
	1; /* tp4_exl3: a split half's core into wo_a's compact rows (same bits;
					       ri06: -9 / -11 ms); 0: the full rows and a copy */
static uint32_t
	dense_skinny_bf16; /* tp4_exl3 class 80: BF16 skinny dense products' outputs a block above 48 rows */
static uint32_t
	dense_skinny_f32; /* tp4_exl3 class 80: float32-weight skinny products' outputs a block above 48 rows */
#define DENSE_SKINNY_MOST_OUTPUTS \
	8u /* --dense_skinny_bf16: the widest stamped wide form (dense_linear.h) */
/* tp4_exl3 class 80: the wide dense products of 64 rows or more staged, 16 outputs a block (the same bits; rk05:
 * IDENTICAL, the 2K cell prompt 1.175 -> 1.150 s, Fraser 1.393 -> 1.385, 30K equal) */
static uint32_t dense_staged = DENSE_STAGED_OUTPUTS;
/* Original TP4 CMP prefill: opt in to the existing guarded BF16 staged dense form. */
uint32_t original_dense_staged;
uint32_t original_head_offload; /* experimental target-only four-CMP head; default keeps native PRO */
uint32_t original_wide_experts; /* compatible TP4_PRO recipes default to one wide expert pass */
static uint32_t
	tp_parallel_launch; /* TP: each card's graphs of a step's group launched from a thread of its own */
static uint32_t tp_graph_parts =
	1; /* TP: a card's step graph behind the prefill cone in this many parts */
static uint32_t
	tp_ordered_wide; /* TP EXL3: wide attention trunk products by the ordered wide form (same bits) */
static uint32_t tp_graph_branches =
	1; /* TP EXL3 TC MoE: the shared expert a graph branch beside the routed (same bits,
					  engab18 -0.34 ms a C1 round); 2: BATCH steps too; 0: one chain */
static uint32_t
	tp_piece_rings; /* TP BF16 recipe: wide steps' BF16 rings over 64-row pieces, not 8-row slabs */
static uint32_t tp_trunk_shape_slices =
	1; /* tp4_exl3: thin trunk products at the slice count measured for their
					       shard shape (trunk01; ordered chains: the same bits at any count); 0:
					       4 */
static uint32_t
	tp_bf16_slab_rings; /* 1: BF16 recipe chunked attention summed a slab at a time by the ring */
/* BF16 recipe chunked steps' sums: copy-engine owner sums and owner ring sums with the layer's halves overlapped (the
 * push's and the slab rings' bits; pf19: 30K-token prefill 19.08 -> 14.80 s) */
static uint32_t tp_wide_exchange = DS41F_TP_EXCHANGE_CE_RINGS;
static uint32_t tp_split_overlap =
	DS41F_SPLIT_JOINT_DOWN; /* ri01: identical (6: rb02 12.370 -> 12.057 s) */
static uint32_t
	tp_split_least_rows; /* split_overlap: chunks of fewer rows run whole (0: halves of more than 64 rows) */
static uint32_t tp_launch_trims =
	1; /* 0: the launches as before 3 Nov (ring input copies, wo_b's quantization) */
static uint32_t
	tp_full_split_overlap; /* the full programs' split_overlap mode (0: --tp_split_overlap's) */
static uint32_t tp_chunk_core =
	1; /* chunked steps' attention core over the chunk (same bits; pf16: 18.93 -> 17.80 s) */
uint32_t original_chunk_core; /* default 0: exact original attention retains its eight-row cores */
static uint32_t tp_replicated_embedding =
	1; /* tp4_exl3: every rank embeds its rows (same bits); 0: card 0 hands off */
/* BATCH (tp4_v3's host half, tp4_v3_b's programs): --batch_rows=R[,R...] (9 to 64 each) prepares BATCH programs of
 * those rows in place of the thin ones; with the queue's depth 1 (step_pipeline_start(driver, 1)); a row count
 * may repeat with --batch_verifiers (another count of draft walks: e.g. BATCH-8 of one for C1 and of two for C2) */
uint32_t batch_rows[MOST_BATCH_PROGRAMS];
static uint32_t batch_draft_rows, batch_wide_head;
uint32_t original_batch; /* default 0: original TP4_PRO keeps THIN; 1: fixed native-tail BATCH48 */
static uint32_t
	batch_context_slabs; /* BATCH: a step's draft context in slabs of 8 rows (the thin steps' forms) */
/* BATCH: a BATCH program's rows the thin steps' bits (trunk_mma's _64 form at the thin splits, the target head at
 * the thin split, the TC MoE's short forms at every row count): a session's tokens independent of the batch */
static uint32_t batch_thin_bits = 1;
/* BATCH, with batch_thin_bits: a step of 9 to this many rows (at most 48) runs its trunk products by the grouped forms
 * (the 1-8-row forms over its row groups of 8: the thin bits in one launch, no partials); 0: trunk_mma's _64 form */
static uint32_t batch_grouped_rows = DEFAULT_BATCH_GROUPED_ROWS;
static uint32_t
	batch_tc_paired; /* with batch_thin_bits: the TC MoE's short forms paired from so many rows */
static uint32_t
	batch_draft_slots_head; /* with --batch_wide_head: one verifier's draft head over 8 slots a position */
static uint32_t
	batch_row_sets; /* with batch_thin_bits: trunk products in row sets above --batch_grouped_rows */
/* A BATCH program's core, window and pools once over its slabs; retain compressor slots after the pools read them.
 * Level 3 preserves both reference widths and mixed-user outputs; level 2 remains selectable. */
uint32_t batch_core_once = DEFAULT_BATCH_CORE_ONCE;
static uint32_t
	tp_front_branch; /* thin coefficient fronts as side branches: 1 attention, 2 both (same bits) */
static uint32_t
	tp_core_heads; /* the attention core's most heads a block, 0 the measured rule (same bits) */
static uint32_t
	tp_decode_round_once; /* BF16 decode partials summed in FP32 at their owners, rounded once */
static uint32_t
	draft_bf16_rows; /* the draft layers' attention products of BF16 rows (drafts only) */
static uint32_t batch_draft_wide =
	DEFAULT_BATCH_DRAFT_WIDE; /* 1 sums; 2 products; 3 cores over whole blocks */
static uint32_t
	draft_bf16_operands; /* class 80: the draft's quantized products of its rows' BF16 values (drafts) */
#define MOST_CORE_HEADS 64u
#define CORE_HEADS_STEP \
	8u /* the attention core's tile of heads (ds41f_attention_core_records.h) */
#define MOST_BATCH_ROW_SETS 24u
/* the BF16 ring sums with column owners: a row's order of additions independent of its place in the step, so a
 * T=0 stream's bits do not depend on the batch or the schedule (default since 3 Nov, owner: new references 30K
 * 41eeb9c16e10, Fraser f49e8badff47); 0: row-quarter owners, the old reference order */
static uint32_t tp_ring_column_owners = 1;
/* BATCH: a step's microseconds on each BATCH program, fewest rows first: each step verifies the drafts that buy the
 * most expected tokens a microsecond (scheduler.c's step plan); none: the draft policy's depths */
uint32_t batch_step_us[SERVING_BATCH_COSTS];
/* BATCH: each --batch_rows program's drafting sessions (its draft walks), apart from its rows: a step then takes the
 * fewest-row program holding its rows and its drafting sessions (--batch_step_us can then trim rows); none: rows / 6 */
uint32_t batch_verifiers[MOST_BATCH_PROGRAMS];
static uint32_t exl3_expert_parts = EXL3_EXPERT_REFERENCE_PARTS;
enum gpu_split_code execution_placement = GPU_SPLIT_PP5;
static bool numerics_was_given;
static uint32_t prefix_reuse_exact;
uint32_t prefix_reuse =
	1u; /* serve: ordinary prefix reuse unless explicitly disabled */
static uint32_t cache_scope = SERVER_CACHE_SESSION;
static uint32_t anthropic_messages;
static uint32_t
	session_fork; /* experimental independent same-owner clones; qualification required */
#define DEFAULT_SESSION_GRACE_SECONDS 30u
static uint32_t session_grace_seconds = DEFAULT_SESSION_GRACE_SECONDS;
static uint32_t session_save_tokens = SERVER_DEFAULT_SAVE_TOKENS;
static uint32_t session_load_tokens = SERVER_DEFAULT_LOAD_TOKENS;
static uint32_t session_idle_save_seconds = SERVER_DEFAULT_IDLE_SAVE_SECONDS;
uint32_t full_steps =
	1; /* with wide rows: the rows behind the front in full chunks (model_seam.h's FULL) */
uint32_t chunks_in_flight = DEFAULT_CHUNKS_IN_FLIGHT;
static uint32_t prefill_margin =
	DEFAULT_PREFILL_MARGIN; /* full rows beyond a prompt's cone (scheduler.c) */
static uint32_t
	prefill_fast_plan; /* chunks_in_flight 1: the largest program that fits (other bits than exact) */
uint32_t prefill_prefix_stable; /* fixed-capacity prefill; default off */
#define PREFIX_STABLE_SMALL_ROWS 1024u
#define PREFIX_STABLE_LARGE_ROWS 2048u
/* PRFXSTBL: distinct numerical identity; the old plan 0 salt is unchanged. */
#define PREFIX_STABLE_SALT 0x505246585354424Cull
/* The measured bounded BATCH recipe selects shortest first; other recipes retain turns. */
#define PREFILL_ORDER_AUTO UINT32_MAX
uint32_t prefill_order =
	PREFILL_ORDER_AUTO; /* serve: prompts one at a time (scheduler.c prompt_head) */
#define PREFILL_ORDER_DEFAULT_WAIT \
	16u /* --prefill_order_wait: with --prefill_order=2, the steps before the oldest's */
#define PREFILL_ORDER_MOST_WAIT 1024u
static uint32_t prefill_order_wait = PREFILL_ORDER_DEFAULT_WAIT;
#define PREFILL_DECODE_MOST_EVERY \
	64u /* --prefill_decode_every: the most chunk steps between decoding steps */
static uint32_t
	prefill_decode_every; /* with prefill_order: a decoding step after so many head chunks (scheduler.c) */
#define PREFILL_SCHEDULING_AUTO UINT32_MAX
#define PREFILL_SCHEDULING_DEFAULT_WIDE_ROWS 1024u
/* An exact whole-granule cap even without another stream decoding. */
static uint32_t prefill_chunk_rows;
uint32_t prefill_decode_rows = PREFILL_SCHEDULING_AUTO;
uint32_t prefill_decode_rounds = PREFILL_SCHEDULING_AUTO;
uint32_t prefill_decode_waiters = PREFILL_SCHEDULING_AUTO;
static uint32_t
	engram_shared; /* TP host gathers: identical table masks share one immutable staging block */
static uint32_t prefill_cone_skip =
	1; /* full steps' rows past their cone: no core or routed experts (scheduler.c) */
static uint32_t prefill_tail_rows =
	TAIL_ROWS_AUTO; /* the 64-row tail chunks two a step, their forms (scheduler.c) */
static uint32_t prefill_dead_halves =
	DEAD_ROWS_TRIMMED; /* full programs leave the sums of a decoder layer's half
							     the cone never reads; 2: and every sum's copies the rows
							     before a half's first one it reads (ri01); 0: none */
static uint32_t prefill_ladder_1024 =
	1; /* a ladder above CHUNK_LADDER_TOP takes a program of that many rows too
					    (ri01: with front_tail identical, 30K -82 ms) */
static uint32_t prefill_ladder_fine =
	LADDER_FINE_AUTO; /* the ladder's every whole-granule count to wide_rows (with
							   1,024 and 1,536; ri07 / gsi10: identical, cold prefill
							   +0.8%, ~200 MiB a card of programs) */
static uint32_t prefill_ladder_1536 =
	1; /* a ladder above LADDER_1536_ROWS takes a program of that many rows too
					    (ri06: identical, a 1,905-token prompt -84 ms) */
static uint32_t prefill_front_tail =
	FRONT_TAIL_AUTO; /* with the tail program: a wide one for the front region's tail
							   too (by default where the tail program applies) */
static uint32_t prefill_tail_overlay =
	1; /* the tail programs' TC MoE workspace over the prefill form's (the same
					     launches and bits; 320-row tail: 244 MiB a card less) */
/* a tail program's trunk products one launch over its 64-row groups (the same bits; rk03c: the 2K prompt -12 ms with
 * the 320-row tail, -15 ms with the 448-row one, identical) */
static uint32_t prefill_tail_sets = 1;
static uint32_t prefill_thin_tail =
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
static uint32_t draft_start_economic = 1;
uint32_t batch_plan_ties; /* the BATCH step plan cuts only between drafts of unequal reach (drafts only) */
uint32_t wide_rows; /* 0: no wide steps; else a wide program of so many rows (model_seam.h) */
uint32_t step_rows = 1;
static uint32_t generated_token_limit; /* 0 until given: the command's default */
/* speculative decoding (front_end_seam.h): off by default; a round verifies at most draft_positions proposals, and at
 * most --rows less 1, of the lookup drafter's n-grams of lookup_ngram_max down to lookup_ngram_min tokens */
uint32_t draft_block = DS41F_DRAFT_BLOCK;
uint32_t speculative, draft_positions = SERVING_DRAFT_MOST;
static uint32_t lookup_ngram_max = DEFAULT_LOOKUP_NGRAM_MOST,
		lookup_ngram_min = DEFAULT_LOOKUP_NGRAM_LEAST;
bool is_draft_given;
static uint32_t hybrid_lookup_min = DRAFT_DEFAULT_LOOKUP_MIN;
static uint32_t
	draft_trace; /* diagnostics: each device round's anchor and proposals to stderr */
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
static uint32_t reasoning_effort;
uint32_t official_hc_rows; /* 0 retains existing arithmetic; else pinned HC reference call shape. */
uint32_t official_reference; /* 0: existing forms; 1: original PP THIN serial M1 reference recipe. */
bool is_seed_given; /* serve: else a request without a seed draws one */
float temperature = 0.0f, top_p = 1.0f;
static double step_started_seconds; /* when the step that runs now began */
static const char *bind_address = SERVER_DEFAULT_ADDRESS, *key_file,
		  *model_alias, *session_directory;
const char *
	layout_text; /* placement.h's layout text in place of the fill; NULL: the fill */
static uint32_t port = SERVER_DEFAULT_PORT;
static uint32_t queue_depth = SERVER_DEFAULT_QUEUE_DEPTH;
static uint32_t user_queue_depth; /* zero: half the global queue, at least one */
static const char *draft_usage = "compact";
static const char *calls_parsing =
	"tolerant"; /* tool calls: tolerant, or v0.0.5-alpha's reference rule */
static const char *
	dump_draft_ring; /* generate diagnostics: the draft rings' first rows to this file at the end */
static uint32_t live_stats_interval_ms;
uint32_t metrics;
static const char *live_stats_format = "text";
static const char *stats_users;
static char stats_card_order[LIVE_STATS_CARD_ORDER_BYTES];
static char stats_placement[LIVE_STATS_PLACEMENT_BYTES];
static int stop_pipe[PIPE_ENDS] = {
	-1, -1
}; /* serve's: a signal writes, the server polls, serve closes */
/* What a run opens, closed by run_program on every path. */
struct storage_reader *reader;
const struct storage_operations_record *storage_operations = &storage_direct;
struct pack_record *pack;
/* turbo load: the copies once the receipted replicas beside the given copies are added (replicas.h) */
static char replica_paths[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
static const char *turbo_copy_paths[STORAGE_MAXIMUM_COPIES];
static struct load_layout *layout;
static struct device_card *cards[BACKEND_MAXIMUM_CARDS];
struct card_description_record descriptions[BACKEND_MAXIMUM_CARDS];
static struct load_result result;
static struct model_instance
	instance; /* the run's model (model_registry.h): bound when its pack opens */
struct ds41f_plan *plan;
struct ds41f_tp2_model *tensor_model;
uint32_t tp_wire_bf16; /* TP-only explicit wire cast; PP retains its original numerical contract. */
static struct step_companion_program tensor_companions[MOST_PROGRAMS];
static struct execution_program *programs[MOST_PROGRAMS];
static struct execution_program *image_programs[MOST_PROGRAMS];
static struct ds41f_head_program_banks *head_program_banks;
static uint32_t program_count;
struct engram_record *engram;
static struct step_driver *driver;
static struct inference_engine *engine;
static struct http_server *server;

/* Placement is prepared once. It is immutable while requests run; no model or program changes for statistics. */
MUST_CHECK static result_status
stats_metadata_bound(const uint32_t *card_of_unit)
{
	struct engine_card_stats metadata[BACKEND_MAXIMUM_CARDS] = { 0 };
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	const uint32_t unit_count = pack_header_of(pack)->unit_count;
	const char *const head_names[] = {
		"head.weight", "mtp.2.markov_head.head.weight",
		"mtp.2.confidence_head.proj.weight"
	};
	const uint64_t head_bits[] = { STATS_HEAD_LANGUAGE, STATS_HEAD_DRAFT,
				       STATS_HEAD_CONFIDENCE };
	size_t order_bytes = 0, placement_bytes = 0;
	for (uint32_t card = 0; card < card_count; card++) {
		metadata[card].engine_index = card;
		metadata[card].backend_index = card_order[card];
		metadata[card].description = descriptions[card];
		result_status status = instance.operations->arena_byte_count(
			plan, card, &metadata[card].arena_bytes);
		if (status != STATUS_OK) {
			return status;
		}
		for (uint32_t unit = 0; unit < unit_count; unit++) {
			if (card_of_unit[unit] == card &&
			    units[unit].unit_kind == PLACEMENT_UNIT_STAGE &&
			    units[unit].layer_index < STATS_LAYER_MASK_BITS) {
				metadata[card].layer_mask |=
					1ull << units[unit].layer_index;
				metadata[card].holds_draft =
					metadata[card].holds_draft ||
					units[unit].layer_index >=
						DS41F_DRAFT_FIRST_LAYER;
			}
		}
		for (uint32_t head = 0;
		     head < sizeof(head_names) / sizeof(head_names[0]);
		     head++) {
			const uint32_t tensor_index = pack_find_tensor(
				pack, head_names[head],
				(uint32_t)strlen(head_names[head]));
			if (tensor_index != PACK_NOT_FOUND) {
				const struct pack_tensor *const tensor =
					pack_tensors_of(pack) + tensor_index;
				const uint32_t unit =
					pack_extents_of(
						pack)[tensor->extent_index]
						.unit_index;
				if (card_of_unit[unit] == card) {
					metadata[card].head_mask |=
						head_bits[head];
					metadata[card].holds_draft =
						metadata[card].holds_draft ||
						head != 0;
				}
			}
		}
		if (tensor_model != NULL) {
			metadata[card].layer_mask = plan->cards[card].layers;
		}
		const int order_written = snprintf(
			stats_card_order + order_bytes,
			sizeof(stats_card_order) - order_bytes, "%s%" PRIu32,
			card != 0 ? "," : "", card_order[card]);
		const int placement_written = snprintf(
			stats_placement + placement_bytes,
			sizeof(stats_placement) - placement_bytes,
			"%scard%" PRIu32 " layers=0x%" PRIx64
			" heads=0x%" PRIx64 " draft=%s",
			card != 0 ? "; " : "", card, metadata[card].layer_mask,
			metadata[card].head_mask,
			metadata[card].holds_draft ? "yes" : "no");
		if (order_written < 0 ||
		    (size_t)order_written >=
			    sizeof(stats_card_order) - order_bytes ||
		    placement_written < 0 ||
		    (size_t)placement_written >=
			    sizeof(stats_placement) - placement_bytes) {
			return STATUS_INVALID_ARGUMENT;
		}
		order_bytes += (size_t)order_written;
		placement_bytes += (size_t)placement_written;
	}
	return engine_stats_cards(engine, metadata, card_count);
}

/* Seconds on the monotonic clock; zero when it cannot be read. */
static double monotonic_seconds(void)
{
	struct timespec now = { 0 };
	const bool clock_was_read = clock_gettime(CLOCK_MONOTONIC, &now) == 0;
	return clock_was_read ?
		       (double)now.tv_sec +
			       (double)now.tv_nsec / NANOSECONDS_PER_SECOND :
		       0.0;
}

/* Ends a step: its line with its duration, or the error line that names it. True when it failed: the run ends. */
static bool step_failed(const char *step_name, result_status status)
{
	const double now_seconds = monotonic_seconds();
	if (status == STATUS_OK) {
		(void)printf("%-20s %9.3f s\n", step_name,
			     now_seconds - step_started_seconds);
	} else { /* a product output, as the line of a step done: release compiles the log hooks out */
		(void)fprintf(stderr, "the step \"%s\" failed: %s\n", step_name,
			      status_text(status));
	}
	step_started_seconds = now_seconds;
	return status != STATUS_OK;
}

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

/* --engram_ram_percent=auto: the default, given once */
static bool take_engram_ram_auto(const char *argument, bool *given)
{
	if (*given || strcmp(argument, ENGRAM_RAM_AUTO_OPTION) != 0)
		return false;
	*given = true;
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

/* --weights=original|exl3, --model_dir=FOLDER and --exl3_map=NAME, as every startup option: exact case, no empty value,
 * no repeat. */
static bool take_model_source(const char *argument)
{
	static const char weights_option[] = "--weights=",
			  folder_option[] = "--model_dir=",
			  map_option[] = "--exl3_map=";
	if (strncmp(argument, weights_option, strlen(weights_option)) == 0 &&
	    weights_family == WEIGHTS_UNSET) {
		const char *const weights = argument + strlen(weights_option);
		weights_family = strcmp(weights, "original") == 0 ?
					 WEIGHTS_ORIGINAL :
				 strcmp(weights, "exl3") == 0 ? WEIGHTS_EXL3 :
								WEIGHTS_UNSET;
		return weights_family != WEIGHTS_UNSET;
	}
	if (strncmp(argument, folder_option, strlen(folder_option)) == 0 &&
	    model_dir == NULL && argument[strlen(folder_option)] != '\0') {
		model_dir = argument + strlen(folder_option);
		return true;
	}
	/* an empty name is model_folder_map_file's refusal */
	if (strncmp(argument, map_option, strlen(map_option)) == 0 &&
	    exl3_map_name == NULL) {
		exl3_map_name = argument + strlen(map_option);
		return model_folder_map_file(exl3_map_name, exl3_map_file) ==
		       STATUS_OK;
	}
	return false;
}

/* The folder's pack: the EXL3 map for --weights=exl3 or GPU split TP4_EXL3, otherwise the original pack. */
static enum model_file folder_pack_file(void)
{
	return weights_family == WEIGHTS_EXL3 ||
			       (weights_family == WEIGHTS_UNSET &&
				execution_placement == GPU_SPLIT_TP4_EXL3) ?
		       CLI_MODEL_FILE_EXL3 :
		       CLI_MODEL_FILE_ORIGINAL;
}

/* A model file's name in the folder: the EXL3 map's is --exl3_map's */
static const char *folder_file_name(enum model_file file)
{
	return file == CLI_MODEL_FILE_EXL3 ? exl3_map_file :
					     model_file_names[file];
}

/* --model_dir, once the arguments are parsed (the parser opens no file): the tables, when the command reads them, and
 * the pack from the folder (folder_pack_file, by its name there). */
MUST_CHECK result_status model_folder_files(bool reads_tables)
{
	static const char *pack_paths[1];
	const enum model_file pack_file = folder_pack_file();
	result_status status =
		reads_tables ? model_folder_resolve(
				       model_dir,
				       model_file_names[CLI_MODEL_FILE_TABLES],
				       &model_files[CLI_MODEL_FILE_TABLES]) :
			       STATUS_OK;
	status = status == STATUS_OK ?
			 model_folder_resolve(model_dir,
					      folder_file_name(pack_file),
					      &model_files[pack_file]) :
			 status;
	if (status != STATUS_OK)
		return status;
	if (reads_tables)
		front_end_paths[TABLES_FILE] =
			model_files[CLI_MODEL_FILE_TABLES].path;
	model_pack = &model_files[pack_file];
	pack_paths[0] = model_pack->path;
	copy_paths = pack_paths;
	copy_count = 1;
	return STATUS_OK;
}

/* --weights against --placement: an omitted placement is the family's default split (original: PP5; EXL3: TP4_EXL3),
 * an explicit one must run the family. load runs no TP placement: it keeps its own fill and only picks the family's
 * pack, so an explicit PP5 is load's own placement for either family. */
static bool weights_placement_matches(bool placement_was_given)
{
	const bool exl3 = weights_family == WEIGHTS_EXL3;
	if (weights_family == WEIGHTS_UNSET ||
	    (!is_generating && execution_placement == GPU_SPLIT_PP5) ||
	    (placement_was_given &&
	     (execution_placement == GPU_SPLIT_TP4_EXL3) == exl3))
		return true;
	if (!placement_was_given) {
		if (exl3 && is_generating) {
			execution_placement = GPU_SPLIT_TP4_EXL3;
			(void)fputs(
				"weights exl3 -> GPU split TP4_EXL3 (default for the family)\n",
				stderr);
		}
		return true;
	}
	(void)fprintf(stderr,
		      "--weights=%s conflicts with --placement=%s: %s\n",
		      exl3 ? "exl3" : "original", placement_text,
		      exl3 ? "EXL3 weights run on GPU split TP4_EXL3" :
			     "GPU split TP4_EXL3 runs EXL3 weights");
	return false;
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

static bool tensor_placement_selected(void);

/* Every parse-time default that reads the session count, for sessions of them (the same count
 * runs the same configuration, however it was reached); an explicit --pipeline_steps or --draft_policy is kept:
 * - a serve's pipeline depth with a device draft (DSpark or hybrid) and without BATCH: PP5
 *   DS41F_ORIGINAL_PP5_MULTI_PIPELINE_STEPS and the TP splits DS41F_TP_MULTI_PIPELINE_STEPS (one flight is refused
 *   there) for several sessions, else 1;
 * - TP4_EXL3's draft depth policy: the rate objective for BATCH or several sessions, else fixed.
 * parse_arguments takes them for the parsed count (SESSION_SLOTS_MAX_PARSED for max), and the fit again for one
 * session when max resolves to one (single_session_defaults). */
static void session_defaults(uint32_t sessions)
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
		  TAIL_WIDEST_ROWS, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
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
						    TAIL_WIDEST_ROWS :
						    TAIL_PROGRAM_ROWS;
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
		prefill_front_tail = prefill_tail_rows == TAIL_WIDEST_ROWS ?
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
			  ((prefill_tail_rows != TAIL_PROGRAM_ROWS &&
			    prefill_tail_rows != TAIL_LONG_ROWS &&
			    prefill_tail_rows != TAIL_WIDEST_ROWS) ||
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

/* The whole of a file, not empty, in memory of malloc (aligned for any type) that the caller frees; else NULL. */
/* The folder's pack as the reader opened it must be the file resolved: its device and inode, then its identity. */
MUST_CHECK result_status
model_pack_opened(const struct storage_operations_record *storage,
		  const struct storage_reader *opened_reader,
		  const struct pack_header *header)
{
	struct storage_description opened = { 0 };
	const char *const name =
		folder_file_name((enum model_file)(model_pack - model_files));
	if (storage->reader_describe(opened_reader, &opened) != STATUS_OK ||
	    !model_file_is(model_pack, opened.first_device,
			   opened.first_inode)) {
		(void)fprintf(
			stderr,
			"model_dir: %s changed between its check and its open\n",
			model_pack->path);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	(void)fprintf(
		stderr,
		"model_dir: %s -> %s (device %u:%u); pack %s, header digest %016" PRIx64
		", tables digest %016" PRIx64 "\n",
		name, model_pack->path, major(model_pack->device),
		minor(model_pack->device), header->model_name,
		header->header_digest, header->tables_digest);
	return STATUS_OK;
}

/* The pack's reader: in the folder form its fixed name sets it, never a link target's suffix. */
const struct storage_operations_record *pack_storage(void)
{
	return model_pack == NULL ? storage_for_path(copy_paths[0]) :
	       model_pack == &model_files[CLI_MODEL_FILE_EXL3] ?
				    &storage_exl3 :
				    &storage_direct;
}

static bool pack_holds_exl3(const struct pack_record *opened_pack)
{
	const struct pack_header *const header = pack_header_of(opened_pack);
	const struct pack_tensor *const tensors = pack_tensors_of(opened_pack);
	bool holds_exl3 = false;
	for (uint32_t tensor = 0; tensor < header->tensor_count && !holds_exl3;
	     tensor++)
		holds_exl3 = tensors[tensor].weight_format ==
			     WEIGHT_FORMAT_EXL3;
	return holds_exl3;
}

/* --weights against the pack: an EXL3 map's pack holds EXL3 matrices, an original pack none. */
MUST_CHECK static result_status
pack_family_status(const struct pack_record *opened_pack)
{
	const struct pack_header *const header = pack_header_of(opened_pack);
	const bool holds_exl3 = pack_holds_exl3(opened_pack);
	if (holds_exl3 == (weights_family == WEIGHTS_EXL3))
		return STATUS_OK;
	(void)fprintf(stderr, "--weights=%s, but the pack %s holds %s\n",
		      weights_family == WEIGHTS_EXL3 ? "exl3" : "original",
		      header->model_name,
		      holds_exl3 ? "EXL3 matrices" : "no EXL3 matrix");
	return STATUS_UNSUPPORTED;
}

/* --exl3_tc_moe=1 takes effect: TP4_EXL3 with the BF16 wire (the exact recipe ignores it), in the plan and in the
 * descriptor's check alike */
static bool tc_moe_selected(void)
{
	return execution_placement == GPU_SPLIT_TP4_EXL3 && tp_wire_bf16 != 0 &&
	       exl3_tc_moe != 0;
}

/* A pack of EXL3 matrices against its map's descriptor and its model (model_registry.h), beside the folder's map or
 * the first pack argument, or without one against the model's established packs: the tables when the command read
 * them, the quality opt-in, and with the tensor-core MoE the model against its kernels. */
MUST_CHECK static result_status map_descriptor_check(const char *tables,
						     uint64_t table_byte_count)
{
	return model_instance_map_check(
		&instance, pack, tables, table_byte_count,
		exl3_unrated_map != 0, tc_moe_selected() ? DS41F_TP4_RANKS : 0u,
		exl3_expert_rotation);
}

/* A whole file; with expected (a --model_dir file), only the resolved file itself. */
char *file_read(const char *path, const struct model_file_record *expected,
		uint64_t *byte_count_out)
{
	/* a --model_dir file opens without blocking (a FIFO swapped in refuses below), then reads blocking */
	const int descriptor =
		expected == NULL ? -1 :
				   open(path, O_RDONLY | O_CLOEXEC |
						      O_NONBLOCK | O_NOCTTY);
	if (descriptor >= 0 &&
	    fcntl(descriptor, F_SETFL,
		  fcntl(descriptor, F_GETFL) & ~O_NONBLOCK) != 0) {
		(void)close(descriptor);
		*byte_count_out = 0;
		return NULL;
	}
	FILE *const file = expected == NULL ? fopen(path, "rb") :
			   descriptor >= 0  ? fdopen(descriptor, "rb") :
					      NULL;
	if (expected != NULL && descriptor >= 0 && file == NULL)
		(void)close(descriptor);
	long byte_count = -1;
	char *bytes = NULL;
	struct stat metadata;
	if (file != NULL && expected != NULL &&
	    (fstat(fileno(file), &metadata) != 0 ||
	     !model_file_is(expected, metadata.st_dev, metadata.st_ino))) {
		(void)fprintf(
			stderr,
			"model_dir: %s changed between its check and its open\n",
			path);
		(void)fclose(file);
		*byte_count_out = 0;
		return NULL;
	}
	if (file != NULL && expected != NULL)
		(void)fprintf(stderr, "model_dir: %s -> %s (device %u:%u)\n",
			      model_file_names[CLI_MODEL_FILE_TABLES], path,
			      major(metadata.st_dev), minor(metadata.st_dev));
	if (file != NULL) {
		byte_count = fseek(file, 0, SEEK_END) == 0 ? ftell(file) : -1;
		bytes = byte_count > 0 && fseek(file, 0, SEEK_SET) == 0 ?
				malloc((size_t)byte_count) :
				NULL;
		if (bytes != NULL && fread(bytes, 1, (size_t)byte_count,
					   file) != (size_t)byte_count) {
			free(bytes);
			bytes = NULL;
		}
		(void)fclose(file);
	}
	*byte_count_out = bytes != NULL ? (uint64_t)byte_count : 0;
	return bytes;
}

/* Explicit token input is a complete JSON uint32 array, with one context slot
 * left for generation. Tokenization and chat rendering are not applied. */
MUST_CHECK result_status prompt_token_ids_parse(const char *text,
						size_t byte_count,
						uint32_t vocabulary_size,
						uint32_t *token_ids_out,
						uint32_t token_capacity,
						uint32_t *token_count_out)
{
	struct json_value document = { 0 }, value = { 0 };
	if (text == NULL || vocabulary_size == 0 || token_ids_out == NULL ||
	    token_count_out == NULL || token_capacity <= 1u ||
	    json_read_document(text, byte_count, &document) !=
		    JSON_DOCUMENT_VALID ||
	    document.text[document.begin] != '[')
		return STATUS_INVALID_ARGUMENT;
	size_t cursor = document.begin;
	uint32_t count = 0;
	while (json_next(document, &cursor, NULL, &value)) {
		uint32_t token_id = 0;
		for (size_t offset = value.begin; offset < value.end;
		     offset++) {
			const char digit = value.text[offset];
			if (digit < '0' || digit > '9')
				return STATUS_INVALID_ARGUMENT;
			const uint32_t part = (uint32_t)(digit - '0');
			if (token_id > (UINT32_MAX - part) / DECIMAL_BASE)
				return STATUS_INVALID_ARGUMENT;
			token_id = token_id * DECIMAL_BASE + part;
		}
		if (token_id >= vocabulary_size)
			return STATUS_INVALID_ARGUMENT;
		if (count >= token_capacity - 1u)
			return STATUS_UNSUPPORTED;
		token_ids_out[count++] = token_id;
	}
	if (count == 0)
		return STATUS_INVALID_ARGUMENT;
	*token_count_out = count;
	return STATUS_OK;
}

/* The device runs its fixed block even when the host verifies fewer rows. Admission and its program check share
 * this capacity; ordinary plans retain their original capacity. */
uint32_t effective_row_capacity(void)
{
	uint32_t rows = wide_rows > step_rows ? wide_rows : step_rows;
	for (uint32_t index = 0; index < MOST_BATCH_PROGRAMS; index++)
		rows = batch_rows[index] > rows ? batch_rows[index] : rows;
	return speculative >= SPECULATIVE_DEVICE && draft_block > rows ?
		       draft_block :
		       rows;
}

static bool tensor_placement_selected(void)
{
	return execution_placement == GPU_SPLIT_TP2_PP2_PRO ||
	       execution_placement == GPU_SPLIT_TP4_PRO ||
	       execution_placement == GPU_SPLIT_TP2_SINGLE_SINGLE_PRO ||
	       execution_placement == GPU_SPLIT_TP4_EXL3;
}

/* Select descriptions only: a PRO is skipped without opening its CUDA arena. */
MUST_CHECK static result_status
exl3_card_order_prepare(const struct backend_operations *backend,
			uint32_t found)
{
	uint32_t selected[DS41F_TP4_RANKS], count = 0;
	for (uint32_t ordinal = 0; ordinal < found; ordinal++) {
		struct card_description_record description = { 0 };
		const result_status status =
			backend->card_describe(ordinal, &description);
		if (status != STATUS_OK)
			return status;
		if (description.arithmetic_class != ARITHMETIC_CLASS_SM_80 ||
		    strstr(description.name, "CMP") == NULL)
			continue;
		if (count == DS41F_TP4_RANKS)
			return STATUS_UNSUPPORTED;
		selected[count++] = ordinal;
	}
	if (count != DS41F_TP4_RANKS)
		return STATUS_UNSUPPORTED;
	(void)memcpy(card_order, selected, sizeof(selected));
	card_count = count;
	return STATUS_OK;
}

/* The cards a serve takes, as backend numbers and describes them (run_program's backend_cuda_value; a test's
 * stand-in of the workstation): with no --card_order every card in the backend's order, TP4_EXL3 its four CMPs
 * (exl3_card_order_prepare), each described into descriptions; refused for no card, more than BACKEND_MAXIMUM_CARDS or
 * a --card_order card the backend does not have. */
MUST_CHECK result_status
serve_cards_describe(const struct backend_operations *backend)
{
	const bool every_card_is_taken = card_count == 0;
	uint32_t cards_found = 0;
	result_status status = backend->card_count(&cards_found);
	if (status == STATUS_OK &&
	    (cards_found == 0 || cards_found > BACKEND_MAXIMUM_CARDS)) {
		/* No card to load onto, or more cards than these arrays hold. */
		status = STATUS_UNSUPPORTED;
	}
	if (status == STATUS_OK && every_card_is_taken &&
	    execution_placement == GPU_SPLIT_TP4_EXL3)
		status = exl3_card_order_prepare(backend, cards_found);
	for (; status == STATUS_OK && every_card_is_taken &&
	       execution_placement != GPU_SPLIT_TP4_EXL3 &&
	       card_count < cards_found;
	     card_count++) {
		card_order[card_count] = card_count;
	}
	for (uint32_t card = 0; card < card_count && status == STATUS_OK;
	     card++) {
		status = card_order[card] >= cards_found ?
				 STATUS_INVALID_ARGUMENT /* no such card */
				 :
				 backend->card_describe(card_order[card],
							&descriptions[card]);
	}
	return status;
}

/* Each card's arithmetic as the serve takes it, a line where it is not the reference's: the PRO's own unless
 * --pro_emulation=1 (the plan, programs and kernels follow the class), a CMP's relaxed order */
void serve_cards_arithmetic(void)
{
	for (uint32_t card = 0; card < card_count; card++) {
		if (descriptions[card].arithmetic_class ==
			    ARITHMETIC_CLASS_SM_80 &&
		    cmp_order == CMP_ORDER_RELAXED) {
			(void)printf(
				"card %" PRIu32
				": relaxed CMP trunk/expert order:"
				" sums and state bits change;"
				" --cmp_order=reference restores the reference order\n",
				card);
		}
		if (descriptions[card].arithmetic_class ==
			    ARITHMETIC_CLASS_SM_120 &&
		    pro_emulation == 0) {
			descriptions[card].arithmetic_class =
				ARITHMETIC_CLASS_SM_120_NATIVE;
			(void)printf(
				"card %" PRIu32
				": native arithmetic (its own tensor cores): outputs depend"
				" on the layers it holds; --pro_emulation=1: the same bits on every card,"
				" 2-4x slower\n",
				card);
		}
	}
}

/* Cold TP bootstrap. The ordinary unit map records canonical homes for identity/statistics only: selected
 * tensor jobs, rather than this map, load every replica/shard. All GPU edges are tested on the actual arenas. */
MUST_CHECK result_status
ds41f_tensor_plan_prepare(const struct program_limits *limits,
			  uint64_t kept_free_bytes, uint32_t *unit_homes)
{
	const bool exl3 = execution_placement == GPU_SPLIT_TP4_EXL3;
	if (card_count != (exl3 ? DS41F_TP4_RANKS : DS41F_TP2_PROGRAM_CARDS) ||
	    unit_homes == NULL)
		return STATUS_UNSUPPORTED;
	struct ds41f_tp2_model_description description = {
		.limits = *limits,
		.kept_free_bytes = kept_free_bytes,
		.wire_format = tp_wire_bf16,
		.exact_output_shards = exl3 ? tp_wire_bf16 == 0 :
					      numerics == NUMERICS_EXACT,
		.exl3_product_order =
			exl3_expert_parts == EXL3_EXPERT_OPTIONAL_PARTS ?
				BB_EXL3_EIGHT_PART_ORDER :
				BB_EXL3_REFERENCE_ORDER,
		.original_code_wire = tp_code_wire != 0,
		.attention_owner = tp_attention_owner != 0,
		.head_shards = tp_head_shards != 0,
		.exact_dense = exl3 && tp_wire_bf16 == 0 && tp_exact_dense != 0,
		.draft_shards = tp_draft_shards != 0,
		.draft_layers = exl3 && tp_wire_bf16 != 0 &&
				tp_draft_layers != 0,
		.head_stream = tp_head_stream != 0,
		.exl3_fast_products = exl3_fast_products != 0,
		.exl3_wide_entry = exl3_wide_entry != 0,
		.exl3_tc_moe = tc_moe_selected(),
		.exl3_tc_product_blocks = exl3_tc_product_blocks,
		.exl3_expert_rotation =
			tc_moe_selected() ? exl3_expert_rotation : 0u,
		.head_mma = exl3 && tp_wire_bf16 != 0 && tp_head_mma != 0,
		.trunk_mma = exl3 && tp_wire_bf16 != 0 && tp_trunk_mma != 0,
		.trunk_mma_fused = tp_trunk_mma_fused != 0,
		.trunk_mma_thin_splits = tp_trunk_mma_thin_splits != 0,
		.router_mma = exl3 && tp_wire_bf16 != 0 && tp_router_mma != 0,
		.router_mma_wide = exl3 && tp_wire_bf16 != 0 &&
				   tp_router_mma == ROUTER_MMA_WIDE,
		.gather_balance = exl3 ? tp_gather_balance : 0u,
		.pool_chunk = exl3 && tp_pool_chunk != 0,
		.pool_gather = exl3 && tp_pool_gather != 0,
		.router_first_whole = exl3 && tp_router_first_whole != 0,
		.tcp_two_blocks = exl3 && exl3_tcp_two_blocks != 0,
		.queries_piece = exl3 && tp_queries_piece != 0,
		.index_span = exl3 && tp_index_span != 0,
		.core_compact_halves = exl3 && tp_core_compact_halves != 0,
		.dense_skinny_bf16 = exl3 ? dense_skinny_bf16 : 0u,
		.dense_skinny_f32 = exl3 ? dense_skinny_f32 : 0u,
		.dense_staged = exl3 ? dense_staged : original_dense_staged,
		.original_wide_experts = original_wide_experts != 0,
		.graph_parts = tp_graph_parts,
		.ordered_wide = tp_ordered_wide != 0,
		.graph_branches = tp_graph_branches,
		.piece_rings = tp_piece_rings != 0,
		.trunk_shape_slices = tp_trunk_shape_slices != 0,
		.bf16_slab_rings = tp_bf16_slab_rings != 0,
		.wide_exchange = tp_wide_exchange,
		.split_overlap = tp_split_overlap,
		.split_least_rows = tp_split_least_rows,
		.launches_untrimmed = tp_launch_trims == 0,
		.full_split_overlap = tp_full_split_overlap,
		.tail_rows = prefill_tail_rows,
		.tail_overlay = prefill_tail_overlay != 0,
		.tail_sets = prefill_tail_sets != 0,
		.thin_tail = prefill_thin_tail != 0,
		.full_cone_behind =
			prefill_dead_halves == 0 || prefill_prefix_stable != 0 ?
				0u :
				engine_full_cone_behind(
					prefill_cone, chunks_in_flight,
					prefill_fast_plan, prefill_margin,
					prefill_cone_skip),
		.dead_rows_trimmed = prefill_dead_halves == DEAD_ROWS_TRIMMED,
		.chunk_core = (exl3 || numerics != NUMERICS_EXACT) ?
				      tp_chunk_core != 0 :
				      original_chunk_core != 0,
		.replicated_embedding = exl3 && tp_replicated_embedding != 0,
		.batch = batch_rows[0] != 0,
		.original_batch = original_batch != 0,
		.batch_draft_rows = batch_draft_rows != 0,
		.batch_wide_head = batch_wide_head != 0,
		.batch_context_slabs = batch_context_slabs != 0,
		.batch_thin_bits = batch_thin_bits != 0,
		.batch_grouped_rows = batch_grouped_rows,
		.batch_tc_paired = batch_tc_paired,
		.batch_draft_slots_head = batch_draft_slots_head != 0,
		.batch_row_sets = batch_row_sets,
		.batch_core_once = batch_core_once,
		.front_branch = tp_front_branch,
		.core_heads = tp_core_heads,
		.decode_round_once = tp_decode_round_once != 0,
		.draft_bf16_rows = draft_bf16_rows != 0,
		.batch_draft_wide = batch_draft_wide,
		.draft_bf16_operands = draft_bf16_operands != 0,
		.ring_column_owners = tp_ring_column_owners != 0,
		.tensor_layout =
			exl3 ? DS41F_TP_LAYOUT_QUAD_ALL :
			execution_placement == GPU_SPLIT_TP4_PRO ?
			       DS41F_TP_LAYOUT_QUAD :
			execution_placement == GPU_SPLIT_TP2_SINGLE_SINGLE_PRO ?
			       DS41F_TP_LAYOUT_HYBRID :
			       DS41F_TP_LAYOUT_PAIRS
	};
	for (uint32_t card = 0; card < card_count; card++)
		description.cards[card] = descriptions[card];
	result_status status =
		ds41f_tp2_model_create(pack, &description, &tensor_model);
	if (status != STATUS_OK)
		return status;
	plan = tensor_model->canonical;
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	for (uint32_t unit = 0; unit < pack_header_of(pack)->unit_count;
	     unit++) {
		unit_homes[unit] = units[unit].unit_kind ==
						   PLACEMENT_UNIT_HOST_ONLY ?
					   PLACEMENT_IN_HOST_MEMORY :
					   0u;
	}
	for (uint32_t index = 0;
	     index < tensor_model->gpu_load.description.job_count; index++) {
		const struct tp_load_job *const job =
			&tensor_model->jobs[index];
		const uint32_t extent =
			pack_tensors_of(pack)[job->tensor_index].extent_index;
		unit_homes[pack_extents_of(pack)[extent].unit_index] =
			job->card -
			tensor_model->stages[job->card]->description.rank;
	}
	return status;
}

/* The model's limits of the parsed options (run_program's, and the host tests'). */
struct program_limits options_limits(void)
{
	return (struct program_limits){
		.context_token_limit = context_tokens,
		.row_capacity = effective_row_capacity(),
		.session_capacity = session_slots,
		.state_width = STATE_WIDTH_16,
		.linear_profile = (enum linear_profile_code)linear_profile,
		.linear_order = (enum linear_order_code)linear_order,
		.cmp_order = cmp_order,
		.official_hc_rows = official_hc_rows,
		.official_reference = official_reference,
		.draft_positions =
			speculative >= SPECULATIVE_DEVICE ? draft_block : 0
	};
}

/* What the open adds to the plan's arenas after plan_create (session_fit's plan_grown, so the fit sees the arenas
 * that are opened, and the program dumps'): the original head offload's shards on every card, and with banked heads a
 * bank for each pipeline step. Nothing else grows an arena between the plan and the open (run_program, which names
 * each of the two as a step). */
MUST_CHECK result_status plan_grown(struct ds41f_plan *grown)
{
	result_status status =
		original_head_offload != 0 ?
			ds41f_original_head_plan(grown, descriptions) :
			STATUS_OK;
	if (status == STATUS_OK &&
	    original_head_offload == DS41F_HEAD_MODE_BANKED)
		status = ds41f_original_head_banks_plan(grown, pipeline_steps);
	return status;
}

#define FIT_BISECTION_PARTS 2u /* each trial halves what is left */

/* Whether limits' sessions fit, as session_fit says (STATUS_OUT_OF_MEMORY: they do not; any other refusal is the
 * placement's, the limits' or the plan's own); the TP plan is made, then released (no plan is left). */
MUST_CHECK static result_status slots_fit(const struct slot_fit *fit,
					  const struct program_limits *limits)
{
	if (!tensor_placement_selected())
		return session_fit(fit->cards, limits);
	const result_status status = ds41f_tensor_plan_prepare(
		limits, fit->kept_free_bytes, fit->unit_homes);
	if (status == STATUS_OK) {
		ds41f_tp2_model_destroy(tensor_model);
		tensor_model = NULL;
		plan = NULL;
	}
	return status;
}

/* A BATCH serve runs exactly STEP_BATCH_MAXIMUM_SESSIONS sessions (parse_arguments). */
bool batch_serving(void)
{
	return is_serving && (original_batch != 0 || batch_rows[0] != 0);
}

/* The most sessions of limits' context that fit, 1 to SERVING_SLOTS_MOST by bisection (the arena grows with the count;
 * a trial count's refusal is taken as not fitting); a BATCH serve tries only its STEP_BATCH_MAXIMUM_SESSIONS. 0 when
 * none does: then *refusal_out is the refusal of the last count tried (one session; BATCH's eight) other than
 * STATUS_OUT_OF_MEMORY, which no count changes, else STATUS_OK. */
uint32_t slots_most(const struct slot_fit *fit,
		    const struct program_limits *limits,
		    result_status *refusal_out)
{
	struct program_limits trial = *limits;
	result_status status = STATUS_OK;
	if (batch_serving()) {
		trial.session_capacity = STEP_BATCH_MAXIMUM_SESSIONS;
		status = slots_fit(fit, &trial);
		*refusal_out = status == STATUS_OUT_OF_MEMORY ? STATUS_OK :
								status;
		return status == STATUS_OK ? STEP_BATCH_MAXIMUM_SESSIONS : 0u;
	}
	/* one session first: a refusal other than memory there is the placement's own, which no count changes */
	uint32_t fits = 0, refused = SERVING_SLOTS_MOST + 1u;
	trial.session_capacity = 1u;
	status = slots_fit(fit, &trial);
	if (status == STATUS_OK)
		fits = 1u;
	else
		refused = 1u;
	while (refused - fits > 1u) {
		trial.session_capacity =
			fits + (refused - fits) / FIT_BISECTION_PARTS;
		if (slots_fit(fit, &trial) == STATUS_OK)
			fits = trial.session_capacity;
		else
			refused = trial.session_capacity;
	}
	*refusal_out = fits == 0u && status != STATUS_OUT_OF_MEMORY ? status :
								      STATUS_OK;
	return fits;
}

/* The longest context, in DS41F_CONTEXT_GRANULE steps up to limits' (TP: from DS41F_CONTEXT_WITHOUT_CANDIDATES), at
 * which limits' sessions fit, by bisection (a trial context's refusal is taken as not fitting): 0 when none does, with
 * *refusal_out as slots_most's. */
static uint32_t context_most(const struct slot_fit *fit,
			     const struct program_limits *limits,
			     result_status *refusal_out)
{
	struct program_limits trial = *limits;
	const uint32_t least = tensor_placement_selected() ?
				       DS41F_CONTEXT_WITHOUT_CANDIDATES /
					       DS41F_CONTEXT_GRANULE :
				       1u;
	uint32_t fits = least - 1u,
		 refused = limits->context_token_limit / DS41F_CONTEXT_GRANULE +
			   1u; /* in granules */
	result_status status = STATUS_OK;
	while (refused - fits > 1u) {
		const uint32_t middle =
			fits + (refused - fits) / FIT_BISECTION_PARTS;
		trial.context_token_limit = middle * DS41F_CONTEXT_GRANULE;
		status = slots_fit(fit, &trial);
		if (status == STATUS_OK)
			fits = middle;
		else
			refused = middle;
	}
	*refusal_out = fits < least && status != STATUS_OUT_OF_MEMORY ?
			       status :
			       STATUS_OK;
	return fits >= least ? fits * DS41F_CONTEXT_GRANULE : 0u;
}

/* The TP plan of limits' sessions. For --session_slots=max the most sessions its cards hold (slots_most), then their
 * plan; limits' session_capacity is that count, or 1 with STATUS_OUT_OF_MEMORY when not even one fits (or one
 * session's own refusal). A refused count makes nothing (ds41f_tp2_model_create destroys its partial model). */
MUST_CHECK result_status tensor_plan_fit(struct program_limits *limits,
					 uint64_t kept_free_bytes,
					 uint32_t *unit_homes)
{
	if (!session_slots_max)
		return ds41f_tensor_plan_prepare(limits, kept_free_bytes,
						 unit_homes);
	const struct slot_fit fit = { NULL, kept_free_bytes, unit_homes };
	result_status refusal = STATUS_OK;
	const uint32_t most = slots_most(&fit, limits, &refusal);
	limits->session_capacity = most != 0 ? most : 1u;
	return refusal != STATUS_OK ? refusal :
	       most == 0 ? STATUS_OUT_OF_MEMORY :
			   ds41f_tensor_plan_prepare(limits, kept_free_bytes,
						     unit_homes);
}

/* Every peer reader must retire before either arena of its pair can close. Unknown completion retains the
 * entire pair through process teardown; neither a failed wait nor closing its partner is a retirement proof. */
static uint32_t tensor_cards_drain(void)
{
	uint32_t retained = (1u << card_count) - 1u;
	const uint32_t members = (execution_placement == GPU_SPLIT_TP4_PRO ||
				  execution_placement == GPU_SPLIT_TP4_EXL3) ?
					 DS41F_TP4_RANKS :
					 DS41F_TP2_RANKS;
	const result_status status = ds41f_tp_group_drain(
		&backend_cuda_value, cards, CARD_WAIT_LIMIT_NANOSECONDS,
		members, &retained);
	(void)status; /* retained remains the fail-closed ownership decision even after a failed wait */
	return retained;
}

/* The TP host arena's read: readers a copy and the piece, each from its knob when given, else from the profile's load
 * section when it is enabled, names this pack and exactly these copies (by their device keys) and was measured with
 * this load's host threads and split (and, for an EXL3 map, an explicit --map_reads equal to the measured mode),
 * else the defaults; one line says which, with the section's reason when it is
 * not applied. The section's readers cap the host arena's reads alone (the stage measured them alone): the choice
 * gives them as its host cap, which goes to tp_load_host_reads_per_copy; --tp_load_reads_per_copy caps the whole
 * load's. */
static void load_section_apply(void)
{
	struct storage_load_section section;
	char reason[STORAGE_PROFILE_REASON_BYTES] = "profile_off";
	bool usable = storage_profile_resolved[0] != '\0' &&
		      storage_load_section_read(storage_profile_resolved,
						&section, reason) &&
		      storage_load_section_matches(
			      &section, pack_header_of(pack), copy_paths,
			      copy_count, replica_external != 0,
			      tp_load_host_threads != 0 ? tp_load_host_threads :
							  tp_load_threads,
			      tp_load_copy_split, reason) &&
		      (storage_operations != &storage_exl3 ||
		       storage_load_section_reads_match(copy_count, map_reads,
							reason));
	if (usable && section.enabled == 0) {
		(void)snprintf(reason, sizeof(reason), "%s",
			       "load_section_candidate");
		usable = false;
	}
	struct storage_load_choice choice = {
		.tp_piece_bytes = (uint64_t)load_piece_mib * BYTES_PER_MIB,
		.tp_readers_per_copy = tp_load_reads_per_copy,
		.tp_host_readers_per_copy = tp_load_host_reads_per_copy
	};
	storage_load_choose(usable ? &section : NULL,
			    explicit_tp_reads_per_copy, explicit_load_piece,
			    &choice);
	tp_load_host_reads_per_copy = choice.tp_host_readers_per_copy;
	const uint32_t readers = choice.readers_from_profile ?
					 choice.tp_host_readers_per_copy :
					 choice.tp_readers_per_copy;
	tp_load_piece_bytes = choice.tp_piece_bytes;
	(void)printf("TP load read: %" PRIu32
		     " readers a copy (%s), piece %" PRIu64 " bytes (%s)%s%s\n",
		     readers,
		     !choice.readers_from_profile ?
			     (explicit_tp_reads_per_copy ? "knob" : "default") :
		     readers == DEFAULT_TP_LOAD_READS_PER_COPY ?
			     "profile = default" :
			     "profile",
		     tp_load_piece_bytes != 0 ?
			     tp_load_piece_bytes :
			     (uint64_t)TP_HOST_LOAD_CHUNK_BYTES,
		     !choice.piece_from_profile ?
			     (explicit_load_piece ? "knob" : "default") :
		     tp_load_piece_bytes == TP_HOST_LOAD_CHUNK_BYTES ?
			     "profile = default" :
			     "profile",
		     usable ? "" : "; load section not applied: ",
		     usable ? "" : reason);
}

MUST_CHECK static result_status tensor_cards_load(const void *tables,
						  uint64_t table_bytes)
{
	result_status status = STATUS_OK;
	load_section_apply();
	TIMER_BEGIN(tp_card_open);
	for (uint32_t card = 0; card < card_count && status == STATUS_OK;
	     card++) {
		const uint64_t bytes = plan->cards[card].arena_byte_count;
		(void)printf("TP card %" PRIu32 " backend=%" PRIu32
			     " arena=%" PRIu64 " bytes\n",
			     card, card_order[card], bytes);
		status = backend_cuda_value.card_open(card_order[card], bytes,
						      &cards[card]);
	}
	TIMER_END(tp_card_open);
	struct ds41f_tp2_model_load_request request = {
		.external_host_extents =
			engram_disk_external_extents(engram_disk),
		.backend = &backend_cuda_value,
		.cards = cards,
		.storage = storage_operations,
		.reader = reader,
		.wait_limit_nanoseconds = CARD_WAIT_LIMIT_NANOSECONDS,
		.load_threads = tp_load_threads,
		.load_reads_per_copy = tp_load_reads_per_copy,
		.load_host_reads_per_copy = tp_load_host_reads_per_copy,
		.load_host_threads = tp_load_host_threads,
		.load_copy_split = tp_load_copy_split,
		.load_piece_bytes = tp_load_piece_bytes
	};
	if (tuned_load_active && load_tuned_weights != 0)
		memcpy(request.load_copy_weights, tuned_load_weights,
		       sizeof(request.load_copy_weights));
	TIMER_BEGIN(tp_peer_qualify);
	status = status != STATUS_OK ?
			 status :
			 ds41f_tp2_peer_qualify(tensor_model,
						&backend_cuda_value, cards,
						CARD_WAIT_LIMIT_NANOSECONDS,
						request.qualified_writer_masks);
	TIMER_END(tp_peer_qualify);
	status = status != STATUS_OK ?
			 status :
			 ds41f_tp2_model_load(tensor_model, &request);
	status = status != STATUS_OK ?
			 status :
			 ds41f_tp2_model_front_end_match(tensor_model, tables,
							 table_bytes);
	if (status == STATUS_OK)
		(void)printf("TP load: source=%" PRIu64 " uploaded=%" PRIu64
			     " host=%" PRIu64 " bytes; tensors=%" PRIu32
			     " verified_extents=%" PRIu32 "+%" PRIu32 "\n",
			     tensor_model->gpu_result.source_bytes_read,
			     tensor_model->gpu_result.uploaded_bytes,
			     tensor_model->host_bytes,
			     tensor_model->gpu_result.tensors_uploaded,
			     tensor_model->gpu_result.extents_verified,
			     tensor_model->host_result.extents_verified);
	/* per card: the sum over its loaded ranges of their on-card digests at their offsets (tp_load.h) */
	for (uint32_t card = 0; card < card_count && status == STATUS_OK;
	     card++)
		(void)printf("%s %016" PRIx64 "%s",
			     card == 0 ? "TP load receipts:" : "",
			     tensor_model->gpu_result.card_receipts[card],
			     card + 1 == card_count ? "\n" : "");
	for (uint32_t copy = 0;
	     copy < copy_count && status == STATUS_OK && tp_load_threads != 0;
	     copy++) {
		const struct tp_load_copies *const copies =
			&tensor_model->copies;
		(void)printf(
			"TP load copy %" PRIu32
			": %.2f GB in %.2f s busy (%.2f GB/s), %" PRIu32
			" failed reads\n",
			copy,
			(double)copies->delivered_bytes[copy] / BYTES_PER_GB,
			(double)copies->busy_nanoseconds[copy] /
				NANOSECONDS_PER_SECOND,
			copies->busy_nanoseconds[copy] != 0 ?
				(double)copies->delivered_bytes[copy] /
					(double)copies->busy_nanoseconds[copy] :
				0.0,
			copies->failed_reads[copy]);
	}
	/* what each copy delivered to each role: under the default split a copy serves the other role once its own
	 * has claimed its last read (tp_load.h) */
	for (uint32_t copy = 0;
	     copy < copy_count && status == STATUS_OK && tp_load_threads != 0;
	     copy++) {
		const struct tp_load_copies *const copies =
			&tensor_model->copies;
		(void)printf(
			"%scopy %" PRIu32 " host %.2f cards %.2f GB%s",
			copy == 0 ? "TP load roles: " : "", copy,
			(double)copies->role_delivered_bytes[TP_LOAD_ROLE_HOST]
							    [copy] /
				BYTES_PER_GB,
			(double)copies->role_delivered_bytes[TP_LOAD_ROLE_CARDS]
							    [copy] /
				BYTES_PER_GB,
			copy + 1u == copy_count ? "" : "; ");
		if (copy + 1u == copy_count)
			(void)printf("; split %" PRIu32 "\n",
				     tp_load_copy_split);
	}
	return status;
}

/* Turbo load: the replicas found beside the given copies (found_paths, replicas_add's search) that open together
 * with the copies already kept follow the given copies in a new reader (replicas_reader_open: one line names each
 * replica left out and why); when none opens, the given copies alone stay (one line says so). */
MUST_CHECK result_status replicas_found_add(
	const char (*found_paths)[REPLICA_PATH_CAPACITY], uint32_t found)
{
	const uint32_t given = copy_count;
	struct storage_reader *wider = NULL;
	const uint32_t kept = replicas_reader_open(
		storage_operations, copy_paths, given, found_paths, found,
		turbo_copy_paths, &wider, stdout);
	if (found != 0 && kept == 0) {
		(void)printf("turbo load: %" PRIu32
			     " replicas do not open: %s\n",
			     found,
			     given == 1u ? "the source alone" :
					   "the given copies alone");
		return STATUS_OK;
	}
	struct storage_description opened = { 0 };
	if (kept != 0 && model_pack != NULL &&
	    (storage_operations->reader_describe(wider, &opened) != STATUS_OK ||
	     !model_file_is(model_pack, opened.first_device,
			    opened.first_inode))) {
		storage_operations->reader_close(wider);
		(void)fprintf(
			stderr,
			"model_dir: %s changed between its check and its open\n",
			model_pack->path);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	if (kept != 0) {
		storage_operations->reader_close(reader);
		reader = wider;
		copy_paths = turbo_copy_paths;
		copy_count = given + kept;
	}
	for (uint32_t index = given; index < given + kept; index++)
		(void)printf("turbo load: copy %" PRIu32 " %s\n", index,
			     turbo_copy_paths[index]);
	return STATUS_OK;
}

bool turbo_load_searches(void)
{
	return !tuned_load_active && turbo_load != 0 &&
	       copy_count < STORAGE_MAXIMUM_COPIES;
}

uint32_t (*turbo_load_search)(const char *const *copies, uint32_t copy_count,
			      const struct pack_header *header,
			      char (*paths)[REPLICA_PATH_CAPACITY],
			      uint32_t capacity,
			      bool external) = replicas_find_policy;

/* Turbo load's search: the receipted replicas of the given copies' source on drives that hold none of them
 * (replicas.h), then replicas_found_add. */
MUST_CHECK static result_status replicas_add(const struct pack_header *header)
{
	const uint32_t found = turbo_load_search(
		copy_paths, copy_count, header, replica_paths,
		STORAGE_MAXIMUM_COPIES - copy_count, replica_external != 0);
	return replicas_found_add(
		(const char (*)[REPLICA_PATH_CAPACITY])replica_paths, found);
}

/* A load given more than one copy names every copy it reads and where each came from: the command line, or turbo
 * load's search beside it (with --turbo_load=0 none comes from the search). */
static void copies_print(uint32_t given)
{
	(void)printf("turbo load: the load reads %" PRIu32 " copies:",
		     copy_count);
	for (uint32_t copy = 0; copy < copy_count; copy++)
		(void)printf(" %" PRIu32 " %s (%s)%s", copy, copy_paths[copy],
			     copy < given ? "given" : "search",
			     copy + 1u == copy_count ? "\n" : ",");
}

bool turbo_load_copies(const struct pack_header *header)
{
	const uint32_t given = copy_count;
	if (turbo_load_searches() &&
	    step_failed("find the replicas", replicas_add(header)))
		return false;
	/* a tuned load's copies are the profile's, named by its storage load: line (a profile applies to one given
	   copy alone) */
	if (given > 1u && !tuned_load_active)
		copies_print(given);
	return true;
}

/* Profile preparation runs once before model loading; it never tunes or changes the machine. */
static void storage_profile_apply(int argument_count, char **arguments)
{
	storage_profile_is_valid = false;
	for (int index = ARGUMENTS_BEFORE_THE_OPTIONS; index < argument_count;
	     index++) {
		explicit_tp_reads_per_copy =
			explicit_tp_reads_per_copy ||
			strncmp(arguments[index], "--tp_load_reads_per_copy=",
				strlen("--tp_load_reads_per_copy=")) == 0;
		explicit_load_piece = explicit_load_piece ||
				      strncmp(arguments[index],
					      "--load_piece_mib=",
					      strlen("--load_piece_mib=")) == 0;
		explicit_engram_threads =
			explicit_engram_threads ||
			strncmp(arguments[index], "--engram_io_threads=",
				strlen("--engram_io_threads=")) == 0;
		explicit_engram_spin =
			explicit_engram_spin ||
			strncmp(arguments[index], "--engram_spin_us=",
				strlen("--engram_spin_us=")) == 0;
		explicit_engram_pin = explicit_engram_pin ||
				      strncmp(arguments[index],
					      "--engram_io_pin=",
					      strlen("--engram_io_pin=")) == 0;
	}
	char default_path[REPLICA_PATH_CAPACITY];
	const char *path = storage_profile_path != NULL ?
				   storage_profile_path :
				   getenv("ANEMONE_STORAGE_PROFILE");
	storage_profile_is_off = path != NULL && strcmp(path, "off") == 0;
	if (storage_profile_is_off)
		return;
	if (path == NULL) {
		const char *config = getenv("XDG_CONFIG_HOME");
		const char *home = getenv("HOME");
		const int length =
			config != NULL ?
				snprintf(default_path, sizeof(default_path),
					 "%s/anemone/storage-profile.json",
					 config) :
			home != NULL ?
				snprintf(
					default_path, sizeof(default_path),
					"%s/.config/anemone/storage-profile.json",
					home) :
				-1;
		if (length < 0 || (size_t)length >= sizeof(default_path)) {
			(void)fprintf(
				stderr,
				"storage profile: fallback no_config_directory\n");
			return;
		}
		path = default_path;
	}
	(void)snprintf(storage_profile_resolved,
		       sizeof(storage_profile_resolved), "%s", path);
	for (int index = ARGUMENTS_BEFORE_THE_OPTIONS; index < argument_count;
	     index++)
		if (strncmp(arguments[index], "--replica_external=",
			    strlen("--replica_external=")) == 0 ||
		    strncmp(arguments[index], "--replica_floor_percent=",
			    strlen("--replica_floor_percent=")) == 0) {
			(void)fputs(
				"storage profile: fallback explicit_replica_policy\n",
				stderr);
			return;
		}
	char reason[STORAGE_PROFILE_REASON_BYTES];
	if (!storage_profile_read(path, &tuned_storage, reason)) {
		(void)fprintf(stderr, "storage profile: fallback %s\n", reason);
		return;
	}
	char real_source[REPLICA_PATH_CAPACITY];
	const bool resolved = copy_count == 1 &&
			      realpath(copy_paths[0], real_source) != NULL;
	if (copy_count != 1 ||
	    (resolved ? strcmp(real_source, tuned_storage.source) != 0 :
			!tuned_storage.source_unavailable ||
				strcmp(copy_paths[0], tuned_storage.source) !=
					0)) {
		(void)fprintf(stderr,
			      "storage profile: fallback source_mismatch\n");
		return;
	}
	struct pack_header source_header = { 0 };
	uint64_t source_bytes = 0;
	char pack_identity[STORAGE_PROFILE_KEY_BYTES];
	/* an EXL3 map's logical header, through the map reader */
	const bool source_matches = storage_pack_header_read(
		tuned_storage.source_unavailable ? tuned_storage.paths[0] :
						   real_source,
		&source_header, &source_bytes);
	if (source_matches)
		(void)snprintf(pack_identity, sizeof(pack_identity),
			       "%016" PRIx64, source_header.header_digest);
	if (!source_matches ||
	    source_header.file_byte_count != tuned_storage.source_bytes ||
	    strcmp(pack_identity, tuned_storage.pack_identity) != 0) {
		(void)fprintf(
			stderr,
			"storage profile: fallback source_identity_mismatch\n");
		return;
	}
	replica_external = tuned_storage.replica_external;
	replica_floor_percent = tuned_storage.replica_floor_percent;
	bool explicit_load = false, explicit_engram = false;
	for (int index = ARGUMENTS_BEFORE_THE_OPTIONS; index < argument_count;
	     index++) {
		const char *text = arguments[index];
		explicit_load =
			explicit_load ||
			strncmp(text,
				"--turbo_load=", strlen("--turbo_load=")) ==
				0 ||
			strncmp(text, "--reader_threads_per_copy=",
				strlen("--reader_threads_per_copy=")) == 0 ||
			strncmp(text,
				"--"
				"tp_load_",
				strlen("--"
				       "tp_load_")) == 0 ||
			strncmp(text, "--map_reads=", strlen("--map_reads=")) ==
				0;
		explicit_engram = explicit_engram ||
				  storage_profile_engram_argument(text);
	}
	tuned_load_active = tuned_storage.load_enabled != 0 && !explicit_load;
	tuned_engram_active = tuned_storage.engram_enabled != 0 &&
			      !explicit_engram && is_generating;
	for (uint32_t index = 0; index < tuned_storage.count; index++)
		tuned_paths[index] = tuned_storage.paths[index];
	if (tuned_storage.source_unavailable && !tuned_load_active) {
		copy_paths = tuned_paths;
		copy_count = 1;
		(void)fputs(
			"storage: source unavailable; using one verified replica with default load scheduling\n",
			stderr);
	}
	if (tuned_load_active) {
		copy_count = storage_profile_load_list(
			&tuned_storage, tuned_load_paths, tuned_load_weights);
		copy_paths = tuned_load_paths;
		reader_threads_per_copy = tuned_storage.load_threads;
		tp_load_copy_split = TP_LOAD_SPLIT_SOONEST;
		(void)fprintf(
			stderr,
			"storage load: %u copies, %u readers a copy, %s\n",
			copy_count, reader_threads_per_copy,
			load_tuned_weights != 0 ?
				"each piece's copy by the profile's weights" :
				"the readers steal pieces (profile weights not applied)");
	}
	if (tuned_engram_active) {
		uint32_t sources = 0;
		if (!storage_profile_engram_sources(
			    &tuned_storage, tuned_engram_sources,
			    sizeof(tuned_engram_sources), &sources)) {
			tuned_engram_active = false;
			(void)fprintf(
				stderr,
				"storage profile: fallback source_list_overflow\n");
		} else if (sources == 0) {
			/* only an EXL3 map was verified: no pack holds the tables for the disk store */
			tuned_engram_active = false;
			(void)fprintf(
				stderr,
				"storage profile: fallback engram_sources_none (an EXL3 map serves no disk lookup)\n");
		}
		if (tuned_engram_active) {
			engram_sources_text = tuned_engram_sources;
			engram_ram_percent = tuned_storage.ram_percent;
			engram_ram_source = ENGRAM_RAM_PROFILE;
			engram_cache_mib = tuned_storage.cache_mib;
			engram_io = tuned_storage.engram_io;
			turbo_nvme_engrams = sources > 1 ? 1u : 0u;
		}
	}
	(void)fprintf(stderr,
		      "storage profile: %s age_seconds=%" PRIu64
		      " load=%u engram=%u replicas=%u excluded=%u %s\n",
		      tuned_storage.identity,
		      (uint64_t)time(NULL) - tuned_storage.created_unix,
		      tuned_load_active ? 1u : 0u,
		      tuned_engram_active ? 1u : 0u, tuned_storage.count,
		      tuned_storage.excluded, reason);
	storage_profile_is_valid = true;
}

MUST_CHECK static result_status storage_metrics_prepare(void)
{
	FILE *const stream = fmemopen(storage_replica_info,
				      sizeof(storage_replica_info), "w");
	if (stream == NULL)
		return STATUS_OUT_OF_MEMORY;
	const result_status status = storage_profile_metrics(
		stream, storage_profile_decisions(), true);
	const int flushed = fflush(stream);
	const long bytes = ftell(stream);
	const int closed = fclose(stream);
	if (status == STATUS_OK && flushed == 0 && closed == 0 && bytes > 0 &&
	    (uint64_t)bytes < sizeof(storage_replica_info))
		storage_replica_info_bytes = (size_t)bytes;
	else
		return STATUS_INPUT_OUTPUT_ERROR;
	return STATUS_OK;
}

/* The static hot set: checked against its recorded SHA-256, the pack and the store, its blocks read through the
 * store's verified path, then served like the RAM prefix. */
MUST_CHECK static result_status engram_hot_start(void)
{
	char reason[ENGRAM_HOT_REASON_BYTES] = "";
	struct timespec started, ended;
	(void)clock_gettime(CLOCK_MONOTONIC, &started);
	const result_status status =
		engram_hot_open(engram_hot_set_path, engram_hot_sha256,
				(uint64_t)engram_hot_mib * BYTES_PER_MIB, pack,
				engram_disk, &engram_hot, reason);
	(void)clock_gettime(CLOCK_MONOTONIC, &ended);
	if (status != STATUS_OK) {
		(void)fprintf(stderr, "engram hot set refused: %s\n", reason);
		return status;
	}
	engram_disk_attach_hot(engram_disk, engram_hot);
	(void)fprintf(stderr,
		      "engram hot set: %" PRIu64 " blocks (%" PRIu64
		      " MiB) from %s, read and verified in %.3f s\n",
		      engram_hot_block_count(engram_hot),
		      (uint64_t)(engram_hot_block_count(engram_hot) *
				 PACK_BLOCK_BYTES / BYTES_PER_MIB),
		      engram_hot_set_path,
		      (double)(ended.tv_sec - started.tv_sec) +
			      (double)(ended.tv_nsec - started.tv_nsec) /
				      NANOSECONDS_PER_SECOND);
	return STATUS_OK;
}

/* Step I/O's default and its guards, once the storage profile was applied: a profile's turbo_engram puts the tables
 * on disk (it sets engram_ram_percent), and --engram_step_io and the hot set are no profile argument, so they keep it.
 * Serving PP5 or TP4_PRO from the disk store reads a step's blocks at submit by default, every stream identical to
 * all-RAM. PP5 (8-9 Nov 2026): cold prefill 27.38 against 27.18 s all-RAM at 131k tokens and 1.75x today's per-row
 * path; decode C1 172.2 against 176.0 tok/s, C4 and C8 equal; 0.8 cores, hybrid and pinned. TP4_PRO (9 Nov 2026, two
 * runs): decode C1 4.6-5.9 % over the per-row path; cold prefill 9.87 / 35.18 s at 8180 / 32782 tokens against
 * 9.74 / 35.08 s all-RAM and 12.63 / 37.98 s per-row. --engram_step_io=0 keeps the per-row path; other GPU splits
 * keep it until measured there. Returns false, with its reason printed, for step I/O or a hot set with every table
 * in RAM. */
static void engram_step_io_default_apply(void)
{
	if (engram_step_io_default && is_serving &&
	    engram_ram_percent < ENGRAM_RAM_PERCENT &&
	    (execution_placement == GPU_SPLIT_PP5 ||
	     execution_placement == GPU_SPLIT_TP4_PRO))
		engram_step_io = 1;
}

/* At startup, before auto's fit (which applies the step I/O default again once it chose disk tables). */
MUST_CHECK bool engram_settings_resolve(void)
{
	engram_step_io_default_apply();
	if (engram_ram_percent == ENGRAM_RAM_PERCENT &&
	    (engram_step_io != 0 || engram_hot_set_path != NULL)) {
		(void)fputs(
			"engram: --engram_step_io=1 and --engram_hot_set need disk tables (--engram_ram_percent below "
			"100, or a storage profile that puts them on disk)\n",
			stderr);
		return false;
	}
	return true;
}

/* The Engram fit (engram_fit.h), once, before the load and before any card: auto's decision, and the fit printed for a
 * given percent too. The n-gram object opened here is the one the load attaches; a pack without n-gram tables has
 * nothing to fit. STATUS_OUT_OF_MEMORY when the model's own host memory does not fit (the lines name it). */
MUST_CHECK static result_status engram_ram_fit(void)
{
	result_status status =
		engram != NULL ?
			STATUS_OK :
			engram_open(storage_operations, reader, pack, &engram);
	if (status == STATUS_UNSUPPORTED)
		return STATUS_OK;
	struct engram_fit_needs needs = { 0 };
	status = status == STATUS_OK ?
			 engram_fit_needs_of_pack(
				 pack, engram_description_of(engram),
				 tensor_placement_selected(), card_count,
				 (uint64_t)load_piece_mib * BYTES_PER_MIB,
				 tp_load_threads, &needs) :
			 status;
	if (status != STATUS_OK)
		return status;
	needs.cache_bytes = (uint64_t)engram_cache_mib * BYTES_PER_MIB;
	needs.hot_bytes = engram_hot_set_path != NULL ?
				  (uint64_t)engram_hot_mib * BYTES_PER_MIB :
				  0;
	engram_fit_memory_read(&needs);
	struct engram_fit_decision decision;
	engram_fit_decide(&needs, engram_ram_source, engram_ram_percent,
			  &decision);
	engram_fit_print(stderr, &needs, &decision, engram_ram_source);
	if (decision.fit_case == ENGRAM_FIT_MODEL_OVER)
		return STATUS_OUT_OF_MEMORY;
	engram_ram_percent = decision.percent;
	engram_step_io_default_apply();
	return STATUS_OK;
}

/* Where a step I/O parameter came from: its knob, the profile's engram section (equal to the default, or not), or the
 * default. */
static const char *engram_source_tag(bool from_profile, bool explicit_knob,
				     bool equals_default)
{
	return from_profile ?
		       (equals_default ? "profile = default" : "profile") :
	       explicit_knob ? "knob" :
			       "default";
}

/* Step I/O's threads a copy, spin and pin: each from its knob when given, else from the profile's engram section when
 * it is enabled and names this pack, exactly the disk store's copies (by their device keys, in order) and this
 * flight's shape, else the default (storage_engram_section_apply). The tables are on disk here: step I/O starts only
 * then. reason gets why the section is not applied, empty when it is. Never the RAM share: that is the RAM fit's. */
static void engram_section_apply(char reason[STORAGE_PROFILE_REASON_BYTES],
				 struct storage_engram_choice *choice)
{
	/* the admitted copies' device keys, read only with a profile (a copy without one matches none) */
	char keys[STORAGE_MAXIMUM_COPIES][STORAGE_PROFILE_KEY_BYTES] = { { 0 } };
	for (uint32_t copy = 0;
	     storage_profile_resolved[0] != '\0' && copy < engram_copy_count;
	     copy++) {
		struct local_device_description device;
		if (storage_device_key_for(engram_copy_paths[copy],
					   replica_external != 0, keys[copy],
					   &device) != STATUS_OK)
			keys[copy][0] = '\0';
	}
	const struct storage_engram_serving serving = {
		.profile = storage_profile_resolved,
		.header = pack_header_of(pack),
		.copy_keys = (const char (*)[STORAGE_PROFILE_KEY_BYTES])keys,
		.copy_count = engram_copy_count,
		.steps = 1u + STEP_PIPELINE_MOST,
		.slots = ENGRAM_STEP_SLOTS,
		.ram_percent = engram_ram_percent,
		.explicit_threads = explicit_engram_threads,
		.explicit_spin = explicit_engram_spin,
		.explicit_pin = explicit_engram_pin
	};
	*choice = (struct storage_engram_choice){ .threads_per_copy =
							  engram_io_threads,
						  .spin_us = engram_spin_us,
						  .pin = engram_io_pin };
	storage_engram_section_apply(&serving, choice, reason);
	engram_io_threads = choice->threads_per_copy;
	engram_spin_us = choice->spin_us;
	engram_io_pin = choice->pin;
}

/* Step I/O over the disk store: one record for a driver without a pipeline and one per pipeline slot, each sized for
 * the largest step any program takes. A read's deadline fits its re-reads on every other copy into half the cards'
 * wait limit (1.25 s for three copies): a pipeline that sees no progress for that limit fails the step, so a stalled
 * read fails over to another copy before it does. */
MUST_CHECK result_status engram_steps_start(void)
{
	if (engram_disk == NULL)
		return STATUS_INVALID_ARGUMENT;
	char reason[STORAGE_PROFILE_REASON_BYTES];
	struct storage_engram_choice choice;
	engram_section_apply(reason, &choice);
	struct engram_steps_options options = {
		.threads_per_copy = engram_io_threads,
		.steps = 1u + STEP_PIPELINE_MOST,
		.slots_per_step = ENGRAM_STEP_SLOTS,
		.rows_most = effective_row_capacity(),
		.spin_nanoseconds =
			(uint64_t)engram_spin_us * NANOSECONDS_PER_MICROSECOND,
		.read_limit_nanoseconds = engram_steps_read_limit_within(
			CARD_WAIT_LIMIT_NANOSECONDS,
			engram_disk_copy_count(engram_disk))
	};
	const struct storage_reader *const engram_reader =
		engram_disk_reader(engram_disk);
	for (uint32_t copy = 0; copy < STORAGE_MAXIMUM_COPIES; copy++)
		for (uint32_t thread = 0;
		     thread < STORAGE_FLIGHT_THREADS_PER_COPY_MOST; thread++)
			options.cpus[copy][thread] =
				engram_io_pin != 0 &&
						copy < engram_disk_copy_count(
							       engram_disk) ?
					storage_flight_queue_cpu(engram_reader,
								 copy, thread) :
					STORAGE_FLIGHT_CPU_FREE;
	result_status status =
		engram_steps_open(engram_disk, &options, &engram_steps);
	if (status == STATUS_OK)
		status = engram_attach_steps(engram, engram_steps);
	if (status == STATUS_OK) {
		(void)fprintf(
			stderr,
			"engram step io: %u copies, %u threads a copy (%s), spin %u us (%s), pin %u (%s), %u steps "
			"of %u slots, cpus",
			engram_disk_copy_count(engram_disk),
			options.threads_per_copy,
			engram_source_tag(
				choice.threads_from_profile,
				explicit_engram_threads,
				engram_io_threads ==
					ENGRAM_STEP_IO_THREADS_DEFAULT),
			engram_spin_us,
			engram_source_tag(
				choice.spin_from_profile, explicit_engram_spin,
				engram_spin_us == ENGRAM_STEP_SPIN_US_DEFAULT),
			engram_io_pin,
			engram_source_tag(choice.pin_from_profile,
					  explicit_engram_pin,
					  engram_io_pin == 1u),
			options.steps, options.slots_per_step);
		for (uint32_t copy = 0;
		     copy < engram_disk_copy_count(engram_disk); copy++)
			for (uint32_t thread = 0;
			     thread < options.threads_per_copy; thread++)
				(void)fprintf(stderr, " %d",
					      options.cpus[copy][thread]);
		const uint32_t plain = storage_flight_plain_rings(
			engram_steps_flight(engram_steps));
		(void)fprintf(stderr, ", read limit %" PRIu64 " ms",
			      (uint64_t)(options.read_limit_nanoseconds /
					 NANOSECONDS_PER_MILLISECOND));
		if (plain != 0)
			(void)fprintf(
				stderr,
				", %u rings read without registered buffers (the locked-memory limit)",
				plain);
		(void)fprintf(
			stderr, " (%s)%s%s\n",
			engram_step_io_default ?
				"the default for disk Engram on PP5 and TP4_PRO; "
				"--engram_step_io=0: per-row gathers" :
				"--engram_step_io=1",
			reason[0] != '\0' ? "; engram section not applied: " :
					    "; engram section applied",
			reason);
	}
	return status;
}

/* Step I/O's counts at shutdown, which a gate can hold to zero retirements: the reads the drives answered, their ages
 * from the hand-off to the ring (the most, and the bucket bound under which 99.9 % fell) against the read deadline,
 * the rings retired past that deadline or for any other reason, the rings set up again, and the probations. */
static void engram_steps_report(void)
{
	struct storage_flight_counts counts;
	storage_flight_counts(engram_steps_flight(engram_steps), &counts);
	(void)fprintf(
		stderr,
		"engram step io: shutdown: %" PRIu64
		" reads answered, oldest %" PRIu64
		" us, 99.9 %% within %" PRIu64 " us, deadline %" PRIu64
		" us (%" PRIu64
		"x the oldest); retirements %u past the deadline, %u other; rings set up again %u; probations %u\n",
		counts.answered,
		(uint64_t)(counts.age_most_ns / NANOSECONDS_PER_MICROSECOND),
		(uint64_t)(storage_flight_age_quantile(&counts,
						       AGE_QUANTILE_PARTS,
						       AGE_QUANTILE_PART) /
			   NANOSECONDS_PER_MICROSECOND),
		(uint64_t)(counts.read_limit_ns / NANOSECONDS_PER_MICROSECOND),
		counts.age_most_ns != 0 ?
			counts.read_limit_ns / counts.age_most_ns :
			(uint64_t)0u,
		counts.deadline_retirements, counts.other_retirements,
		counts.reinstatements, engram_steps_probations(engram_steps));
}

/* Step I/O started; or, when it is the default rather than asked for, the host refused it (io_uring disabled or
 * filtered, a ring the kernel will not set up): the serve keeps the per-row gathers it had before, and says why. */
MUST_CHECK result_status engram_steps_start_or_fall_back(void)
{
	const result_status status = engram_steps_start();
	if (status == STATUS_OK || !engram_step_io_default)
		return status;
	engram_steps_close(
		engram_steps); /* open or attach failed: nothing is attached */
	engram_steps = NULL;
	engram_step_io = 0;
	(void)fprintf(
		stderr,
		"engram step io: off, per-row gathers at each card's visit (the default could not start: %s)\n",
		status_text(status));
	return STATUS_OK;
}

/* Runtime table sources are independent of the model loader's copies and may use different pack layouts. */
MUST_CHECK static result_status engram_sources_prepare(void)
{
	const char *paths[STORAGE_MAXIMUM_COPIES] = { 0 };
	uint32_t count = copy_count;
	char *text = NULL;
	result_status status = STATUS_OK;
	if (engram_sources_text != NULL) {
		const size_t length = strnlen(engram_sources_text,
					      (size_t)STORAGE_MAXIMUM_COPIES *
						      REPLICA_PATH_CAPACITY);
		if (length == 0 || length == (size_t)STORAGE_MAXIMUM_COPIES *
						     REPLICA_PATH_CAPACITY)
			return STATUS_INVALID_ARGUMENT;
		text = strdup(engram_sources_text);
		if (text == NULL)
			return STATUS_OUT_OF_MEMORY;
		count = 0;
		char *next = text;
		while (next != NULL && status == STATUS_OK) {
			char *comma = strchr(next, ',');
			if (comma != NULL)
				*comma = '\0';
			if (*next == '\0' || count == STORAGE_MAXIMUM_COPIES ||
			    strlen(next) >= REPLICA_PATH_CAPACITY)
				status = STATUS_INVALID_ARGUMENT;
			else
				paths[count++] = next;
			next = comma != NULL ? comma + 1 : NULL;
		}
	} else {
		if (storage_operations != &storage_direct)
			return STATUS_UNSUPPORTED;
		for (uint32_t copy = 0; copy < count; copy++)
			paths[copy] = copy_paths[copy];
	}
	struct local_device_description devices[STORAGE_MAXIMUM_COPIES];
	uint32_t accepted = 0;
	for (uint32_t copy = 0; status == STATUS_OK && copy < count; copy++) {
		struct local_device_description current = { 0 };
		const result_status described =
			local_device_describe(paths[copy], &current);
		const char *reason =
			described != STATUS_OK ?
				"device_disappeared_or_unknown" :
				local_device_media_refusal(
					&current, replica_external != 0);
		for (uint32_t previous = 0;
		     reason == NULL && previous < accepted; previous++)
			if (local_device_overlaps(&current, &devices[previous]))
				reason = "duplicate_physical_device";
		local_device_print(stderr, &current,
				   reason != NULL ? reason : "eligible", 0,
				   replica_floor_percent,
				   tuned_engram_active ?
					   tuned_storage.load_weights[copy] :
					   0,
				   tuned_engram_active ?
					   tuned_storage.engram_weights[copy] :
					   0);
		if (!tuned_engram_active) {
			const result_status noted = storage_profile_note(
				&current, reason != NULL ? reason : "eligible",
				STORAGE_REPLICA_LOOKUP);
			if (noted != STATUS_OK)
				status = noted;
		}
		if (reason != NULL)
			continue;
		paths[accepted] = paths[copy];
		devices[accepted++] = current;
	}
	if (status == STATUS_OK &&
	    (accepted == 0 || (tuned_engram_active && accepted != count)))
		status = STATUS_UNSUPPORTED;
	count = accepted;
	for (uint32_t copy = 0; copy < count; copy++)
		(void)snprintf(engram_copy_paths[copy],
			       sizeof(engram_copy_paths[copy]), "%s",
			       paths[copy]);
	engram_copy_count = count;
	if (count == 1u)
		turbo_nvme_engrams = 0;
	if (status == STATUS_OK)
		status = engram_disk_sources_open(paths, count, pack, engram,
						  &engram_source_reader);
	engram_disk_scan_threads(engram_scan_threads);
	if (status == STATUS_OK)
		status = engram_disk_open(
			engram_description_of(engram), pack, &storage_direct,
			engram_source_reader != NULL ? engram_source_reader :
						       reader,
			engram_ram_percent,
			(uint64_t)engram_cache_mib * BYTES_PER_MIB,
			turbo_nvme_engrams, engram_io, &engram_disk);
	if (status == STATUS_OK && tuned_engram_active)
		status = engram_disk_weights(engram_disk,
					     tuned_storage.engram_weights,
					     tuned_storage.count);
	free(text);
	return status;
}

/* The exact reference recipe admits text even when the pack also contains a vision tower. */
bool model_images_are_enabled(const struct ds41f_plan *model_plan)
{
	return model_plan != NULL && model_plan->vision.enabled &&
	       model_plan->limits.official_reference == 0;
}

/* The programs a serve prepares, in engine_failed_to_start's order (the step driver's), of the parsed options alone: on
 * a TP split with --batch_rows its BATCH programs, else THIN ones of 1, 2 and the most rows; then the chunk ladder's
 * WIDE ones and with --full_steps its FULL ones; on a TP split the tails and thin tails. */
void serve_programs_choose(bool tensor, struct serve_programs *chosen)
{
	*chosen = (struct serve_programs){ 0 };
	uint32_t most_rows = step_rows < LINEAR_MAXIMUM_ROWS ?
				     step_rows :
				     LINEAR_MAXIMUM_ROWS;
	const bool prepares_pair = session_slots > 1 &&
				   most_rows > SERVING_PAIR_ROWS;
	const uint32_t program_rows[MOST_THIN_PROGRAMS] = {
		1, prepares_pair ? SERVING_PAIR_ROWS : most_rows, most_rows
	};
	const uint32_t programs_to_prepare = prepares_pair ?
						     MOST_THIN_PROGRAMS :
					     most_rows > 1 ? SERVING_PAIR_ROWS :
							     1;
	/* --batch_rows: BATCH programs of those rows in place of the thin ones (the engine's decode steps, any
	 * sessions) */
	chosen->batch = batch_rows[0] != 0 && tensor;
	chosen->verifiers_given = batch_verifiers[0] != 0;
	for (uint32_t index = 0;
	     index < MOST_BATCH_PROGRAMS && batch_rows[index] != 0; index++)
		for (uint32_t other = 0; other < index; other++)
			chosen->rows_repeat = chosen->rows_repeat ||
					      batch_rows[other] ==
						      batch_rows[index];
	for (uint32_t index = 0; chosen->batch && index < MOST_BATCH_PROGRAMS &&
				 batch_rows[index] != 0;
	     index++) {
		chosen->programs[chosen->count++] = (struct serve_program){
			PROGRAM_BATCH, batch_rows[index],
			batch_verifiers[index], SERVE_BATCH_ROWS
		};
		chosen->verifiers_given = chosen->verifiers_given &&
					  batch_verifiers[index] != 0;
		most_rows = batch_rows[index] > most_rows ? batch_rows[index] :
							    most_rows;
	}
	for (uint32_t thin = 0; !chosen->batch && thin < programs_to_prepare;
	     thin++)
		chosen->programs[chosen->count++] = (struct serve_program){
			PROGRAM_THIN, program_rows[thin], 0, SERVE_THIN_ROWS
		};
	chosen->first_count = chosen->count;
	chosen->most_rows = most_rows;
	/* behind the thin programs, chunks of wide_rows, a quarter of it, ..., at least the least wide rows, which ends
	 * the ladder (a prompt's rows below it go in thin steps): wide ones for the front rows, full ones for the rows
	 * behind the front (the engine picks a size: scheduler.c). Above CHUNK_LADDER_TOP: wide_rows, 512, 256, 64 (the
	 * exact plan's whole multiples of 256 and its tail's 64; the fast plan's 2,048 + 512 + 64 of a 2.7K prompt). */
	uint32_t chunk_rows[MOST_CHUNK_PROGRAMS + LADDER_INSERTED_SIZES +
			    LADDER_FINE_COUNT + PREFILL_CHUNK_CAPS],
		chunk_count = 0;
	for (uint32_t rows = wide_rows;
	     rows >= STEP_WIDE_LEAST_ROWS &&
			     chunk_count<MOST_CHUNK_PROGRAMS; rows = rows>
				     CHUNK_LADDER_TOP ?
		     CHUNK_LADDER_TOP / LADDER_HALVING :
	     rows == CHUNK_LADDER_TOP / LADDER_HALVING &&
			     wide_rows > CHUNK_LADDER_TOP ?
		     rows / LADDER_HALVING :
		     rows / CHUNK_ROWS_DIVISOR)
		chunk_rows[chunk_count++] = rows;
	if (chunk_count != 0 && chunk_count < MOST_CHUNK_PROGRAMS &&
	    chunk_rows[chunk_count - 1u] != STEP_WIDE_LEAST_ROWS)
		chunk_rows[chunk_count++] = STEP_WIDE_LEAST_ROWS;
	/* --prefill_ladder_1024: CHUNK_LADDER_TOP's own programs behind wide_rows' (the exact plan's whole multiples of
	 * 256: a region of 1,024 to 2,047 whole rows in one step fewer; BATCH programs take the thin ones' place, so
	 * MOST_PROGRAMS holds them) */
	if (prefill_ladder_1024 != 0 && chunk_count > 1u &&
	    chunk_rows[0] > CHUNK_LADDER_TOP) {
		memmove(&chunk_rows[LADDER_SECOND + 1u],
			&chunk_rows[LADDER_SECOND],
			(chunk_count - 1u) * sizeof(chunk_rows[0]));
		chunk_rows[LADDER_SECOND] = CHUNK_LADDER_TOP;
		chunk_count++;
	}
	/* --prefill_ladder_1536: LADDER_1536_ROWS' programs too (a region of 1,536 to 2,047 whole rows, e.g. Curtis's
	 * 2K cell's 1,024 + 512, in one step; each split half 768 rows) */
	if (prefill_ladder_1536 != 0 && chunk_count > 1u &&
	    chunk_rows[0] > LADDER_1536_ROWS) {
		memmove(&chunk_rows[LADDER_SECOND + 1u],
			&chunk_rows[LADDER_SECOND],
			(chunk_count - 1u) * sizeof(chunk_rows[0]));
		chunk_rows[LADDER_SECOND] = LADDER_1536_ROWS;
		chunk_count++;
	}
	/* Fine ladder and explicit caps, in the ladder's order. A single slot cannot decode beside another prompt, so
	 * its decoder-only cap prepares no extra programs. The global cap is also used to qualify its numerics. */
	static const uint32_t fine_rows[LADDER_FINE_COUNT] = { 1792u, 1280u,
							       768u };
	uint32_t extra_rows[LADDER_FINE_COUNT + PREFILL_CHUNK_CAPS] = { 0 };
	for (uint32_t fine = 0;
	     prefill_ladder_fine != 0 && fine < LADDER_FINE_COUNT; fine++)
		extra_rows[fine] = fine_rows[fine];
	extra_rows[LADDER_FINE_COUNT] = prefill_chunk_rows;
	extra_rows[LADDER_FINE_COUNT + 1u] =
		session_slots > 1u ? prefill_decode_rows : 0u;
	for (uint32_t extra = 0;
	     chunk_count > 1u && extra < LADDER_FINE_COUNT + PREFILL_CHUNK_CAPS;
	     extra++) {
		if (extra_rows[extra] == 0)
			continue;
		uint32_t at = 0;
		while (at < chunk_count && chunk_rows[at] > extra_rows[extra])
			at++;
		if (at == 0 || at == chunk_count ||
		    chunk_rows[at] == extra_rows[extra])
			continue;
		memmove(&chunk_rows[at + 1u], &chunk_rows[at],
			(chunk_count - at) * sizeof(chunk_rows[0]));
		chunk_rows[at] = extra_rows[extra];
		chunk_count++;
	}
	const enum program_kind chunk_kinds[] = {
		PROGRAM_WIDE, full_steps != 0 ? PROGRAM_FULL : PROGRAM_WIDE
	};
	for (uint32_t kind = 0;
	     kind < sizeof(chunk_kinds) / sizeof(chunk_kinds[0]) &&
	     (kind == 0 || chunk_kinds[kind] != chunk_kinds[0]);
	     kind++)
		for (uint32_t step = 0;
		     step < chunk_count && chosen->count < MOST_PROGRAMS;
		     step++)
			chosen->programs[chosen->count++] =
				(struct serve_program){ chunk_kinds[kind],
							chunk_rows[step], 0,
							SERVE_CHUNK_LADDER };
	/* --prefill_tail_rows: a full program of the exact plan's tail, two 64-row chunks a step in their forms (and
	 * with 320 one of five, with 448 one of five and one of seven); --prefill_front_tail: a wide one of two for the
	 * front region's tail too */
	const struct {
		enum program_kind kind;
		uint32_t rows;
		bool wanted;
	} tails[] = { { PROGRAM_FULL, TAIL_PROGRAM_ROWS,
			prefill_tail_rows != 0 },
		      { PROGRAM_FULL, TAIL_LONG_ROWS,
			prefill_tail_rows >= TAIL_LONG_ROWS },
		      { PROGRAM_FULL, TAIL_WIDEST_ROWS,
			prefill_tail_rows == TAIL_WIDEST_ROWS },
		      { PROGRAM_WIDE, TAIL_PROGRAM_ROWS,
			prefill_tail_rows != 0 && prefill_front_tail != 0 },
		      { PROGRAM_WIDE, TAIL_LONG_ROWS,
			prefill_tail_rows >= TAIL_LONG_ROWS &&
				prefill_front_tail == FRONT_TAIL_ALL },
		      { PROGRAM_WIDE, TAIL_WIDEST_ROWS,
			prefill_tail_rows == TAIL_WIDEST_ROWS &&
				prefill_front_tail == FRONT_TAIL_ALL } };
	for (uint32_t tail = 0; tail < sizeof(tails) / sizeof(tails[0]) &&
				tensor && chosen->count < MOST_PROGRAMS;
	     tail++)
		if (tails[tail].wanted)
			chosen->programs[chosen->count++] =
				(struct serve_program){ tails[tail].kind,
							tails[tail].rows, 0,
							SERVE_TAIL };
	/* --prefill_thin_tail: full programs of 16 to 56 rows, 8 apart: the tail's rows short of a 64-row chunk but its
	 * row of logits in one step, each row in a thin step's forms; with --prefill_front_tail=2 wide ones too */
	for (uint32_t kind = 0;
	     kind <
	     (prefill_front_tail == FRONT_TAIL_ALL ? THIN_TAIL_KINDS : 1u);
	     kind++)
		for (uint32_t rows = THIN_TAIL_LEAST_ROWS;
		     prefill_thin_tail != 0 && rows < STEP_WIDE_LEAST_ROWS &&
		     tensor && chosen->count < MOST_PROGRAMS;
		     rows += LINEAR_MAXIMUM_ROWS)
			chosen->programs[chosen->count++] =
				(struct serve_program){
					kind == 0 ? PROGRAM_FULL : PROGRAM_WIDE,
					rows, 0, SERVE_THIN_TAIL
				};
}

/* The step line of a program of the serve's list */
static const char *serve_program_step(const struct serve_program *own)
{
	static const char *const steps[SERVE_PROGRAM_RULES] = {
		"prepare a BATCH program", "prepare a program",
		"prepare a wide program", "prepare a tail program",
		"prepare a thin-tail program"
	};
	return own->rule == SERVE_CHUNK_LADDER && own->kind == PROGRAM_FULL ?
		       "prepare a full program" :
		       steps[own->rule];
}

/* A program of the serve's list on target, a TP split's with its companions */
MUST_CHECK result_status serve_program_prepare(
	const struct serve_program_target *target,
	const struct serve_program *own, struct execution_program **program_out,
	struct step_companion_program *companions_out)
{
	if (target->tensor_description == NULL)
		return target->model->program_prepare(
			target->plan, target->backend, target->cards,
			target->descriptions, own->kind, own->rows,
			program_out);
	target->tensor_description->batch_verifiers = own->verifiers;
	return own->kind == PROGRAM_THIN ?
		       ds41f_tp2_program_prepare(target->tensor_description,
						 target->backend, target->cards,
						 target->descriptions,
						 own->rows, program_out,
						 companions_out) :
		       ds41f_tp_program_prepare(target->tensor_description,
						target->backend, target->cards,
						target->descriptions, own->kind,
						own->rows, program_out,
						companions_out);
}

/* The image program of a text program on target: of its kind and rows */
MUST_CHECK result_status
serve_image_program_prepare(const struct serve_program_target *target,
			    const struct execution_program *text,
			    struct execution_program **image_out)
{
	return ds41f_image_program_prepare(target->plan, target->backend,
					   target->cards, target->descriptions,
					   text->kind, text->row_count,
					   image_out);
}

/* --original_head_offload=2's flight banks on target, over count text programs and their image programs (or NULL) */
MUST_CHECK result_status serve_head_banks_create(
	const struct serve_program_target *target,
	const struct execution_program *const *texts,
	const struct execution_program *const *images, uint32_t count,
	struct ds41f_head_program_banks **banks_out)
{
	return ds41f_original_head_banks_create(target->plan, target->backend,
						target->cards, texts, images,
						count, banks_out);
}

/* Once the pack is loaded: the n-gram tables, the state, the programs, the driver and the engine over them. Every
 * step ends as step_failed says; true when one failed. */
MUST_CHECK static bool
engine_failed_to_start(const struct engine_description_record *description)
{
	const bool image_programs_enabled = model_images_are_enabled(plan);
	struct serve_programs chosen;
	serve_programs_choose(tensor_model != NULL, &chosen);
	result_status status =
		engram != NULL ?
			STATUS_OK :
			engram_open(storage_operations, reader, pack, &engram);
	status = status == STATUS_UNSUPPORTED ?
			 STATUS_OK :
			 status; /* a model without n-grams */
	const struct engram_description *const tables =
		engram != NULL ? engram_description_of(engram) : NULL;
	const void *const host_arena = tensor_model != NULL ?
					       tensor_model->host_arena :
					       result.host_arena;
	const uint64_t *const host_offsets =
		tensor_model != NULL ? tensor_model->host_offsets :
				       layout->byte_offset_of_extent;
	for (uint32_t table = 0;
	     engram_disk == NULL && tables != NULL &&
	     table < tables->table_count && status == STATUS_OK;
	     table++) {
		const uint32_t extent =
			pack_units_of(pack)[tables->table_unit_index[table]]
				.first_extent_index;
		status = host_arena == NULL ?
				 STATUS_INVALID_ARGUMENT :
				 engram_attach_entry(
					 engram, table,
					 (const unsigned char *)host_arena +
						 host_offsets[extent]);
	}
	bool failed = step_failed("open the n-gram tables", status) ||
		      (tensor_model == NULL &&
		       step_failed("initialise the state",
				   instance.operations->arena_initialise(
					   plan, &backend_cuda_value, cards,
					   CARD_WAIT_LIMIT_NANOSECONDS)));
	failed = failed ||
		 (original_head_offload != 0 &&
		  step_failed("load and qualify original target head offload",
			      ds41f_original_head_initialise(
				      plan, pack, storage_operations, reader,
				      &backend_cuda_value, cards,
				      CARD_WAIT_LIMIT_NANOSECONDS)));
	if (!failed && original_head_offload == DS41F_HEAD_MODE_LEASED)
		(void)fprintf(
			stderr,
			"original target head offload: 4 x %u BF16 rows; CMP dense1/stream148; "
			"native PRO normalization/sampler/draft; shared planes under PRO visit lease\n",
			DS41F_HEAD_SHARD_ROWS);
	if (!failed && original_head_offload == DS41F_HEAD_MODE_BANKED)
		(void)fprintf(
			stderr,
			"original target head offload: %u immutable flight banks; CMP dense1/stream148; "
			"native PRO normalization/sampler/draft; owner-scoped PRO visit ordering\n",
			plan->original_head.bank_count);
	struct ds41f_tp2_program_description tensor_description = { 0 };
	failed = failed ||
		 (tensor_model != NULL &&
		  step_failed("describe TP programs",
			      ds41f_tp2_model_program_description(
				      tensor_model, &tensor_description)));
	const struct serve_program_target target = {
		.model = instance.operations,
		.plan = plan,
		.backend = &backend_cuda_value,
		.cards = cards,
		.descriptions = descriptions,
		.tensor_description =
			tensor_model != NULL ? &tensor_description : NULL
	};
	const uint32_t row_capacity = effective_row_capacity();
	program_count = 0;
	uint32_t verifiers[MOST_PROGRAMS] = {
		0
	}; /* --batch_verifiers: each BATCH program's, the others 0 */
	failed = failed ||
		 (chosen.batch && chosen.rows_repeat &&
		  batch_verifiers[0] == 0 &&
		  step_failed(
			  "a repeated BATCH row count needs --batch_verifiers",
			  STATUS_INVALID_ARGUMENT));
	/* the BATCH or THIN programs first, each THIN one admitted for the serve's rows and sessions with its experts
	 * resident; a failed THIN one counts, so the stop destroys what it made */
	uint32_t entry = 0;
	for (; entry < chosen.first_count && !failed; entry++) {
		const struct serve_program *const own = &chosen.programs[entry];
		verifiers[program_count] = own->verifiers;
		failed = step_failed(
			serve_program_step(own),
			serve_program_prepare(
				&target, own, &programs[program_count],
				&tensor_companions[program_count]));
		if (own->rule == SERVE_BATCH_ROWS) {
			program_count += failed ? 0u : 1u;
			continue;
		}
		if (!failed &&
		    !serving_program_matches(
			    programs[program_count], own->rows, row_capacity,
			    session_slots, description->context_token_limit)) {
			failed = step_failed(
				"require admitted row/session programs",
				STATUS_UNSUPPORTED);
		}
		if (!failed &&
		    (programs[program_count]->host_expert_count != 0 ||
		     programs[program_count]->expert_pause_count != 0)) {
			failed = step_failed("require resident experts",
					     STATUS_UNSUPPORTED);
		}
		program_count++;
	}
	failed = failed ||
		 (batch_verifiers[0] != 0 && !chosen.verifiers_given &&
		  step_failed("one verifier count a BATCH program",
			      STATUS_INVALID_ARGUMENT));
	for (; entry < chosen.count && !failed; entry++) {
		failed =
			step_failed(serve_program_step(&chosen.programs[entry]),
				    serve_program_prepare(
					    &target, &chosen.programs[entry],
					    &programs[program_count],
					    &tensor_companions[program_count]));
		program_count += failed ? 0u : 1u;
	}
	for (uint32_t index = 0;
	     !failed && image_programs_enabled && index < program_count;
	     index++) {
		failed = step_failed(
			"prepare an image program",
			serve_image_program_prepare(&target, programs[index],
						    &image_programs[index]));
	}
	if (!failed && original_head_offload == DS41F_HEAD_MODE_BANKED)
		failed = step_failed(
			"prepare immutable original head program banks",
			serve_head_banks_create(
				&target,
				(const struct execution_program *const *)
					programs,
				image_programs_enabled ?
					(const struct execution_program *const *)
						image_programs :
					NULL,
				program_count, &head_program_banks));
	if (head_program_banks != NULL) {
		const struct ds41f_head_bank_memory memory =
			ds41f_original_head_banks_memory(head_program_banks);
		(void)fprintf(
			stderr,
			"original head program banks: host_heap_bytes=%zu source_launch_bytes=%zu "
			"source_leases=%zu thin_programs=%u programs=%u banks=%u program_metadata_bytes=%zu\n",
			memory.heap_bytes, memory.source_launch_bytes,
			memory.source_lease_count, memory.thin_programs,
			memory.program_count, memory.bank_count,
			memory.program_metadata_bytes);
	}
	const struct step_flight_programs *const head_banks =
		ds41f_original_head_bank_tables(head_program_banks);
	return failed ||
	       step_failed(
		       "create the driver",
		       step_driver_create(
			       &backend_cuda_value, cards,
			       head_banks != NULL ?
				       head_banks[0].programs :
				       (const struct execution_program *const *)
					       programs,
			       program_count, engram,
			       CARD_WAIT_LIMIT_NANOSECONDS, &driver)) ||
	       (chosen.verifiers_given &&
		step_failed("attach BATCH verifiers",
			    step_batch_verifiers_attach(driver, verifiers,
							program_count))) ||
	       (tensor_model != NULL &&
		step_failed("attach TP groups",
			    step_groups_attach(
				    driver, tensor_companions, program_count,
				    tensor_model->stages[0]
					    ->description.rank_count))) ||
	       (engram_shared != 0 && step_failed("share TP n-gram gathers",
						  step_engram_share(driver))) ||
	       (tensor_model != NULL && tp_parallel_launch != 0 &&
		step_failed("start the TP launchers",
			    step_launchers_start(driver))) ||
	       (plan->limits.official_reference == 0 &&
		step_failed("attach image encoder",
			    step_images_attach(
				    driver, plan,
				    descriptions[0].arithmetic_class))) ||
	       (image_programs_enabled &&
		step_failed(
			"attach image programs",
			step_image_programs_attach(
				driver,
				head_banks != NULL ?
					head_banks[0].image_programs :
					(const struct execution_program *const *)
						image_programs,
				program_count))) ||
	       (head_banks != NULL &&
		step_failed("attach immutable original head flight banks",
			    step_flight_programs_attach(
				    driver, head_banks,
				    plan->original_head.bank_count,
				    program_count,
				    ds41f_original_head_owner_guard(
					    head_program_banks)))) ||
	       step_failed("page-lock host experts",
			   step_host_experts_attach(
				   driver, host_arena,
				   tensor_model != NULL ?
					   tensor_model->host_bytes :
					   layout->host_byte_count)) ||
	       (step_graphs != 0 && step_failed("prepare step graphs",
						step_graphs_prepare(driver))) ||
	       ((pipeline_steps > 1 || batch_rows[0] != 0) &&
		step_failed(
			"start the step pipeline",
			tp_pipeline_overlap != 0 ?
				step_pipeline_overlap_start(driver,
							    pipeline_steps) :
				step_pipeline_start(driver, pipeline_steps))) ||
	       step_failed("start the engine",
			   engine_create(driver, description, chosen.most_rows,
					 &engine));
}

/* The digest of the words that hold range on card (the backend digests whole words). */
MUST_CHECK static result_status
aligned_digest(uint32_t card, struct linear_range range, uint64_t *digest_out)
{
	const uint64_t word = PACK_DIGEST_WORD_BYTES,
		       first = range.byte_offset / word * word;
	const uint64_t end =
		(range.byte_offset + range.byte_count + word - 1u) / word *
		word;
	return backend_cuda_value.digest(cards[card], first, end - first,
					 CARD_WAIT_LIMIT_NANOSECONDS,
					 digest_out);
}

/* generate diagnostics (--dump_draft_ring): on the result card, session 0's window rows 0 .. DRAFT_RING_DUMP_ROWS - 1
 * of each draft stage's ring, stage by stage, then the last draft's rotations (8 rows), step area, collapse rows,
 * result areas, normed rows (HIDDEN, 8 rows) and Markov rows, to path, behind the generation. True when it failed. */
MUST_CHECK static bool draft_ring_dump_failed(const char *path)
{
	const uint32_t card = plan->result_card;
	const uint64_t bytes =
		(uint64_t)DRAFT_RING_DUMP_ROWS * DS41F_STATE_ROW_BYTES;
	const uint64_t positions = plan->limits.draft_positions,
		       *const arrays = plan->cards[card].array_offset;
	const struct linear_range ranges[] = {
		{ plan->cards[card].ring_offset[DS41F_DRAFT_FIRST_LAYER],
		  bytes },
		{ plan->cards[card].ring_offset[DS41F_DRAFT_FIRST_LAYER + 1u],
		  bytes },
		{ plan->cards[card].ring_offset[DS41F_DRAFT_FIRST_LAYER +
						DRAFT_LAST_STAGE],
		  bytes },
		{ arrays[DS41F_ARRAY_ROTATION],
		  (uint64_t)(LINEAR_MAXIMUM_ROWS * DS41F_ROTATION_ROW_BYTES) },
		{ arrays[DS41F_ARRAY_DRAFT_STEP],
		  STEP_BLOCK_BYTES_VALUE + positions * STEP_ROW_BYTES_VALUE },
		{ arrays[DS41F_ARRAY_DRAFT_COLLAPSE],
		  positions * DS41F_HIDDEN_ROW_BYTES },
		{ arrays[DS41F_ARRAY_DRAFT_RESULTS],
		  positions * STEP_DRAFT_RESULT_BYTES },
		{ arrays[DS41F_ARRAY_HIDDEN],
		  (uint64_t)(LINEAR_MAXIMUM_ROWS * DS41F_HIDDEN_ROW_BYTES) },
		{ arrays[DS41F_ARRAY_DRAFT_MARKOV],
		  positions * DS41F_MARKOV_RANK * sizeof(uint16_t) },
		/* the draft logits' first 2 MiB (DIAG_DRAFT_PROBE's probes; else the logits) */
		{ arrays[DS41F_ARRAY_DRAFT_LOGITS], bytes },
		{ arrays[DS41F_ARRAY_DRAFT_LOGITS] + bytes, bytes },
		{ arrays[DS41F_ARRAY_DRAFT_LOGITS] + DRAFT_LOGITS_THIRD * bytes,
		  bytes },
		{ arrays[DS41F_ARRAY_DRAFT_LOGITS] +
			  DRAFT_LOGITS_FOURTH * bytes,
		  bytes },
	};
	void *staging = NULL;
	result_status status =
		backend_cuda_value.staging_allocate(bytes, &staging);
	FILE *const file = status == STATUS_OK ? fopen(path, "wb") : NULL;
	status = status == STATUS_OK && file == NULL ?
			 STATUS_INPUT_OUTPUT_ERROR :
			 status;
	for (uint32_t range = 0;
	     range < sizeof(ranges) / sizeof(ranges[0]) && status == STATUS_OK;
	     range++) {
		uint64_t ticket = 0;
		const uint64_t range_bytes = ranges[range].byte_count;
		status = range_bytes > bytes ?
				 STATUS_INVALID_ARGUMENT :
				 backend_cuda_value.download_begin(
					 cards[card], ranges[range].byte_offset,
					 staging, range_bytes, &ticket);
		status = status == STATUS_OK ?
				 backend_cuda_value.upload_wait(
					 cards[card], ticket,
					 CARD_WAIT_LIMIT_NANOSECONDS) :
				 status;
		status = status == STATUS_OK && fwrite(staging, 1, range_bytes,
						       file) != range_bytes ?
				 STATUS_INPUT_OUTPUT_ERROR :
				 status;
	}
	status = file != NULL && fclose(file) != 0 && status == STATUS_OK ?
			 STATUS_INPUT_OUTPUT_ERROR :
			 status;
	if (staging != NULL)
		backend_cuda_value.staging_free(staging);
	/* the weights the result card holds, digested (memory a step might overwrite): each walk layer's tensors mixed
	 * into one line (the draft stages' one a tensor too), the globals and the draft's globals one a line */
	for (uint32_t layer = 0;
	     layer < DS41F_WALK_LAYERS && status == STATUS_OK; layer++) {
		uint64_t mixed = 0;
		for (uint32_t index = 0;
		     index < DS41F_TENSOR_LAYER_COUNT && status == STATUS_OK;
		     index++) {
			const struct ds41f_tensor *const tensor =
				&plan->layer_tensors[layer][index];
			uint64_t digest = 0;
			if (tensor->stored.byte_count == 0 ||
			    tensor->card != card)
				continue;
			status = aligned_digest(card, tensor->stored, &digest);
			mixed = (mixed ^ digest) * DIGEST_MIX_PRIME;
			if (layer >= DS41F_DRAFT_FIRST_LAYER)
				(void)fprintf(stderr,
					      "weights draft stage %" PRIu32
					      " tensor %" PRIu32
					      " digest %016" PRIx64 "\n",
					      layer - DS41F_DRAFT_FIRST_LAYER,
					      index, digest);
		}
		(void)fprintf(stderr,
			      "weights layer %" PRIu32 " digest %016" PRIx64
			      "\n",
			      layer, mixed);
	}
	for (uint32_t index = 0;
	     index < DS41F_TENSOR_GLOBALS + DS41F_TENSOR_DRAFT_GLOBALS &&
	     status == STATUS_OK;
	     index++) {
		const struct ds41f_tensor *const tensor =
			index < DS41F_TENSOR_GLOBALS ?
				&plan->globals[index] :
				&plan->draft_globals[index -
						     DS41F_TENSOR_GLOBALS];
		uint64_t digest = 0;
		if (tensor->stored.byte_count == 0 || tensor->card != card)
			continue;
		status = aligned_digest(card, tensor->stored, &digest);
		(void)fprintf(stderr,
			      "weights global %" PRIu32 " digest %016" PRIx64
			      "\n",
			      index, digest);
	}
	return step_failed("dump the draft rings", status);
}

/* generate diagnostics (DIAG_STATE_DIGESTS in the environment): behind the generation, the digest of every
 * STATE_DIGEST_BLOCK_BYTES block of session 0's state on each card, a line each on stderr: per walk layer the card
 * holds, its ring, compressor, pool and key tiles (two runs' lines name the first layer, kind and block they differ
 * in: which rows' arithmetic depends on the chunking). True when it failed. */
#define STATE_DIGEST_BLOCK_BYTES 65536ull
#define STATE_DIGEST_KINDS 4u
MUST_CHECK static bool state_digests_failed(void)
{
	static const char *const kind_names[STATE_DIGEST_KINDS] = {
		"ring", "compressor", "pool", "keys"
	};
	result_status status = STATUS_OK;
	for (uint32_t card = 0; card < plan->card_count && status == STATUS_OK;
	     card++) {
		const struct ds41f_card *const holder = &plan->cards[card];
		const uint64_t layers = ds41f_walk_layers(holder);
		for (uint32_t layer = 0;
		     layer < DS41F_WALK_LAYERS && status == STATUS_OK;
		     layer++) {
			if ((layers >> layer & 1u) == 0)
				continue;
			const struct linear_range ranges[STATE_DIGEST_KINDS] = {
				{ holder->ring_offset[layer],
				  DS41F_RING_SESSION_BYTES },
				{ holder->compressor_offset[layer],
				  DS41F_COMPRESSOR_SESSION_BYTES },
				{ holder->pool_offset[layer],
				  plan->pool_session_bytes[layer] },
				{ holder->key_tiles_offset[layer],
				  plan->key_session_bytes[layer] }
			};
			for (uint32_t kind = 0;
			     kind < STATE_DIGEST_KINDS && status == STATUS_OK;
			     kind++) {
				for (uint64_t block = 0;
				     ranges[kind].byte_offset !=
					     PROGRAM_NOWHERE &&
				     block * STATE_DIGEST_BLOCK_BYTES <
					     ranges[kind].byte_count &&
				     status == STATUS_OK;
				     block++) {
					const uint64_t left =
						ranges[kind].byte_count -
						block * STATE_DIGEST_BLOCK_BYTES;
					const struct linear_range part = {
						ranges[kind].byte_offset +
							block * STATE_DIGEST_BLOCK_BYTES,
						left < STATE_DIGEST_BLOCK_BYTES ?
							left :
							STATE_DIGEST_BLOCK_BYTES
					};
					uint64_t digest = 0;
					status = aligned_digest(card, part,
								&digest);
					(void)fprintf(stderr,
						      "state card %" PRIu32
						      " layer %" PRIu32
						      " %s block %" PRIu64
						      " digest %016" PRIx64
						      "\n",
						      card, layer,
						      kind_names[kind], block,
						      digest);
				}
			}
		}
	}
	return step_failed("digest the state", status);
}

/* generate: the prompt as one command, a line for each token its events bring, then the figures. True when it
 * failed. */
MUST_CHECK static bool prompt_failed_to_generate(const uint32_t *token_ids,
						 uint32_t token_count)
{
	const struct generation_command_record command = {
		.new_token_ids = token_ids,
		.sampler_seed = seed,
		.new_token_count = token_count,
		.generated_token_limit = generated_token_limit,
		.sampler_temperature = temperature,
		.sampler_top_p = top_p
	};
	struct generation_event event = { 0 };
	int descriptor = -1;
	uint32_t generated = 0;
	bool is_available = false;
	(void)printf("prompt ids");
	for (uint32_t index = 0; index < token_count; index++) {
		(void)printf(" %" PRIu32, token_ids[index]);
	}
	(void)printf("\n");
	const double started_seconds = monotonic_seconds();
	double first_token_seconds = 0.0, token_seconds = started_seconds;
	result_status status = engine_event_descriptor(engine, &descriptor);
	status = status != STATUS_OK ? status : engine_submit(engine, &command);
	while (status == STATUS_OK &&
	       event.event_kind == GENERATION_EVENT_TOKEN) {
		struct pollfd poll_entry = {
			descriptor, POLLIN, 0
		}; /* the engine bounds every step */
		status = poll(&poll_entry, 1, -1) < 0 && errno != EINTR ?
				 STATUS_INPUT_OUTPUT_ERROR :
				 engine_next_event(engine, &event,
						   &is_available);
		if (status == STATUS_OK && is_available &&
		    event.event_kind == GENERATION_EVENT_TOKEN) {
			const double now_seconds = monotonic_seconds();
			(void)printf("token %" PRIu32 " %" PRIu32
				     " %.9g %.6f\n",
				     generated, event.token_id,
				     (double)event.logit,
				     now_seconds - token_seconds);
			token_seconds = now_seconds;
			first_token_seconds =
				generated == 0 ? now_seconds - started_seconds :
						 first_token_seconds;
			generated++;
		}
		status = status == STATUS_OK &&
					 event.event_kind ==
						 GENERATION_EVENT_FAILED ?
				 (result_status)event.status :
				 status;
	}
	const double decode_seconds =
		monotonic_seconds() - started_seconds - first_token_seconds;
	if (status ==
	    STATUS_OK) { /* a run that failed ends on its error line */
		(void)printf("prompt of %" PRIu32
			     " tokens and first token %.3f s, %" PRIu32
			     " more tokens %.3f s: %.2f "
			     "tokens/s\n",
			     token_count, first_token_seconds,
			     generated > 0 ? generated - 1u : 0, decode_seconds,
			     generated > 1 && decode_seconds > 0.0 ?
				     (double)(generated - 1u) / decode_seconds :
				     0.0);
	}
	return status != STATUS_OK && step_failed("run a step", status);
}

/* capture: the cases of golden_capture_options through the engine, on the driver's first thin program's parts. */
MUST_CHECK static bool capture_failed(int argument_count, char **arguments)
{
	const struct execution_program *thin = NULL;
	for (uint32_t index = 0; index < program_count && thin == NULL; index++)
		thin = programs[index]->kind == PROGRAM_THIN ? programs[index] :
							       NULL;
	const char *const map = copy_count != 0	   ? copy_paths[0] :
				model_pack != NULL ? model_pack->path :
						     "";
	const char *const slash = strrchr(map, '/');
	const struct golden_capture_run run = {
		.engine = engine,
		.driver = driver,
		.program = thin,
		.argument_count = argument_count,
		.arguments = arguments,
		.version = PROGRAM_ATTRIBUTION_NOTICE,
		.placement = placement_text != NULL ? placement_text :
						      "default",
		.weights = storage_operations == &storage_exl3 ? "exl3" :
								 "original",
		.map = slash != NULL ? slash + 1 : map,
		.pack_header = pack_header_of(pack),
		.pack_header_bytes = sizeof(struct pack_header),
		/* the map map_descriptor_check read the descriptor of: the instance's */
		.map_path = storage_operations != &storage_exl3 ?
				    NULL :
				    instance.map_path
	};
	return step_failed("capture", golden_capture_run(&run));
}

/* serve's signals: one byte into the pipe the server polls. */
static void stop_requested(int signal_number)
{
	const int saved_errno = errno;
	const unsigned char byte = (unsigned char)signal_number;
	const ssize_t written =
		write(stop_pipe[1], &byte,
		      sizeof(byte)); /* a full pipe already says it */
	(void)written;
	errno = saved_errno;
}

/* serve: the stop pipe and its signals, the line that says the server is up, the requests until a signal; then the
 * signals' handlers as they were, and the pipe closed (left open if a handler could not be given back: it writes). */
MUST_CHECK static bool requests_failed(const struct text_tokenizer *tokenizer)
{
	static const int stop_signals[STOP_SIGNALS] = { SIGINT, SIGTERM,
							SIGHUP };
	struct sigaction action = { .sa_handler = stop_requested },
			 before[STOP_SIGNALS];
	uint32_t taken = 0;
	bool is_ready = sigemptyset(&action.sa_mask) == 0 &&
			pipe2(stop_pipe, O_CLOEXEC | O_NONBLOCK) == 0;
	for (; is_ready && taken < STOP_SIGNALS; taken += is_ready ? 1u : 0u) {
		is_ready = sigaction(stop_signals[taken], &action,
				     &before[taken]) == 0;
	}
	if (is_ready) {
		(void)printf("serving %s on %s port %" PRIu32 "\n",
			     pack_header_of(pack)->model_name, bind_address,
			     port);
	}
	result_status info_status = is_ready ? storage_metrics_prepare() :
					       STATUS_INPUT_OUTPUT_ERROR;
	if (info_status == STATUS_OK)
		info_status = server_storage_info(server, storage_replica_info,
						  storage_replica_info_bytes);
	const result_status status =
		info_status == STATUS_OK ?
			server_run(server, engine, tokenizer, stop_pipe[0]) :
			info_status;
	bool is_restored = true;
	for (uint32_t index = 0; index < taken; index++) {
		is_restored = sigaction(stop_signals[index], &before[index],
					NULL) == 0 &&
			      is_restored;
	}
	for (uint32_t end = 0; is_restored && end < PIPE_ENDS; end++) {
		if (stop_pipe[end] >= 0) {
			(void)close(stop_pipe[end]);
		}
	}
	if (!is_restored) {
		LOG_ERROR(
			"SIGINT or SIGTERM could not be given its handler back: the stop pipe stays open");
	}
	return step_failed("serve", status == STATUS_OK && !is_restored ?
					    STATUS_INPUT_OUTPUT_ERROR :
					    status);
}

/* A separate literal keeps the complete scheduling examples below C11's 4,095-byte translation limit. */
static void scheduling_usage(FILE *output)
{
	(void)fputs(
		"Scheduling: anemone serve --help-scheduling\n"
		"A ready decoder runs its ordinary decode program between exact whole-granule prompt chunks.\n"
		"--prefill_decode_rows=0|256..2048 caps chunks only while another stream decodes; 0 keeps old"
		" behavior.\n"
		"--prefill_decode_rounds=1..64 sets ordinary decode rounds at each interval (old value 1).\n"
		"--prefill_decode_waiters=1 uses original chunks and one round while multiple prompts wait;\n"
		"  the configured cap and burst apply to the final pending prompt. 0 keeps a fixed policy.\n"
		"--prefill_decode_every=0 uses one chunk per interval with the cap; N>0 permits N chunks.\n"
		"--prefill_chunk_rows=0|256..2048 caps chunks even without a decoder (default 0).\n"
		"Caps are multiples of 256, require --chunks_in_flight=1 and the exact prefill plan, and retain\n"
		"the original final tail. --prefill_order=0 uses turns; 1 oldest; 2 shortest with bounded oldest"
		" wait.\n"
		"Order 3 shares short tails in the measured row-independent EXL3 BATCH recipe only.\n"
		"--prefill_prompt_chunks=1 keeps runnable prompt chunks out of decode BATCH steps; 0 restores mixing.\n"
		"It defaults to 0; the measured code C8 cost is 1.12% stream decode / 3.00% aggregate.\n"
		"BATCH default: 1024 rows / 1 round / waiters0 / order2 with --prefill_first=1 and wide_rows=1024,\n"
		"BF16 column-owner sums, thin decode bits and the exact chunk plan. THIN retains old defaults.\n"
		"Explicit options override defaults; --prefill_decode_rows=0 also keeps rounds1/waiters0/order0.\n"
		"Examples: $BATCH stands for your normal BATCH recipe options; TABLES and MAP are model paths.\n"
		"Place these scheduling options before the model paths; do not repeat options already in $BATCH.\n"
		"  anemone serve $BATCH TABLES MAP                         # default: 1024/1, waiters0, order2\n"
		"  anemone serve $BATCH --prefill_decode_rows=1024 --prefill_decode_rounds=1"
		" --prefill_decode_waiters=0 --prefill_order=2 TABLES MAP    # full chunks, "
		"one decode round (see table)\n"
		"  anemone serve $BATCH --prefill_decode_rows=1024 --prefill_decode_rounds=4"
		" --prefill_decode_waiters=0 --prefill_order=2 TABLES MAP    # faster reply completion\n"
		"  anemone serve $BATCH --prefill_decode_rows=256 --prefill_decode_rounds=8"
		" --prefill_decode_waiters=0 --prefill_order=2 TABLES MAP    # shortest measured pauses\n"
		"  anemone serve $BATCH --prefill_chunk_rows=0 --prefill_decode_rows=0 --prefill_decode_rounds=1"
		" --prefill_decode_waiters=0 --prefill_decode_every=0 --prefill_order=0 TABLES MAP # old defaults\n"
		"\nMeasured 4-CMP BATCH order2, prefix reuse off: 30K prompt + four short streams, T=0, max64 tokens\n"
		"(one short ends at EOS50). Seconds; short TTFT/completion are medians, pause is largest SSE content"
		" gap.\n"
		"Completion is time after the first token. Fraser replies are 2,048 tokens; aggregate includes"
		" first-token wait.\n"
		"Policy       ShortTTFT Complete Pause LongTTFT Fraser C4/C8 median TTFT  C4/C8 aggregate tok/s\n"
		"old/order2      3.75    11.53  5.45   14.29       3.94/7.13                 321/384\n"
		"512/1           4.10     5.70  0.29   16.16       4.40/8.38                 312/358\n"
		"1024/4          4.36     2.82  0.46   15.99       4.61/9.25                 316/356\n"
		"256/8           5.08     1.27  0.21   17.34       7.93/18.77                266/274\n"
		"waiter512/4     3.92     3.84  0.47   15.32       4.08/7.61                 316/368\n"
		"Waiter row: one pilot Fraser batch; mixed table uses the together arrival.\n"
		"1024/1          3.92     7.77  0.48   15.38       4.10/7.64                 322/373\n"
		"1024/1 row: one final pilot batch per Fraser concurrency.\n"
		"Old and 512/1 have three Fraser batches; 1024/4 and 256/8 are one-batch pilots. Mixed texts equal"
		" solo.\n"
		"Shorter caps and larger bursts reduce pauses/completion time but add first-token work and can lower\n"
		"aggregate throughput. These are workload measurements, not latency bounds.\n"
		"See the RND_M evidence log.\n",
		output);
}

int startup_placement_status(void)
{
	const struct startup_policy execution_policy = {
		.numerics = numerics,
		.placement = execution_placement,
		.cmp_order = cmp_order
	};
	if (startup_policy_admit(&execution_policy) != STATUS_OK) {
		(void)fprintf(
			stderr,
			"the requested TP placement is unsupported: tensor weight, state and program"
			" integration is not admitted by this build\n");
		return PROGRAM_FAILED;
	}
	if (execution_placement == GPU_SPLIT_TP4_EXL3) {
		(void)printf(
			"TP4 EXL3: four CMP cards; expert parts %u; %s\n",
			exl3_expert_parts,
			tp_wire_bf16 != 0 ?
				"local-K partial products and BF16 ring; approximate" :
				"full-K products and lossless FP16/FP32 transport");
	} else if (numerics_was_given && numerics != NUMERICS_EXACT) {
		if (tensor_placement_selected()) {
			(void)printf(
				"TP numerics: original packed weights and state; reference local CMP products;"
				" BF16 partial outputs; %s\n",
				tp_wire_bf16 != 0 ?
					"thin BF16 ring hop sums, wide ordered BF16 wire" :
					"FP32 wire, fixed rank-order FP32 sums and BF16 completion");
		} else {
			(void)printf(
				"numerics %s: %s CMP trunk/expert order; original weight/cache/head formats;"
				" fast currently adds no further PP5 approximation\n",
				numerics == NUMERICS_FAST ? "fast" : "relaxed",
				cmp_order == CMP_ORDER_RELAXED ? "relaxed" :
								 "reference");
		}
	}
	if (tp_attention_owner != 0 &&
	    (execution_placement != GPU_SPLIT_TP4_PRO ||
	     numerics != NUMERICS_EXACT || tp_wire_bf16 != 0)) {
		(void)fprintf(
			stderr,
			"attention ownership requires exact TP4 placement\n");
		return PROGRAM_FAILED;
	}
	if (tp_wire_bf16 != 0 &&
	    (!tensor_placement_selected() || numerics == NUMERICS_EXACT)) {
		(void)fprintf(
			stderr,
			"BF16 wire requires an explicit relaxed TP placement\n");
		return PROGRAM_FAILED;
	}
	if (tensor_placement_selected() &&
	    (!is_generating || pro_emulation != 0 ||
	     cmp_order != CMP_ORDER_REFERENCE ||
	     context_tokens < DS41F_CONTEXT_WITHOUT_CANDIDATES ||
	     step_rows > LINEAR_MAXIMUM_ROWS ||
	     speculative < SPECULATIVE_DEVICE)) {
		(void)fprintf(
			stderr,
			execution_placement == GPU_SPLIT_TP4_EXL3 ?
				"TP4 EXL3 requires reference CMP products, context>=16384,"
				" decode rows<=8 and DSpark\n" :
				"TP requires native PRO, reference local CMP products, context>=16384,"
				" decode rows<=8 and DSpark\n");
		return PROGRAM_FAILED;
	}
	return PROGRAM_SUCCEEDED;
}

/* A TP pipeline's queue depth, once the session count is known (single_session_defaults may set it). */
static void tp_queue_depth_report(void)
{
	if (tensor_placement_selected() && pipeline_steps > 1u)
		(void)printf(
			tp_pipeline_overlap != 0 ?
				"TP queue depth %u%s: separate CMP-prefix and native-PRO-tail workers\n" :
				"TP queue depth %u%s: one coordinator executes complete flights serially;"
				" no GPU overlap\n",
			pipeline_steps,
			pipeline_steps_defaulted ?
				" (the default for several serving slots with the device draft)" :
				"");
}

/* --session_slots=max resolved to one session: the defaults of --session_slots=1
 * (session_defaults), so one session runs what an explicit count of one runs, where one session still fits under them
 * (the PP fit again; the TP plan reads neither the depth nor the draft policy) and they admit the given knobs. NULL
 * then; else the multi-session defaults stay, and the reason. */
const char *single_session_defaults(const struct slot_fit *fit,
				    const struct program_limits *limits)
{
	const uint32_t depth = pipeline_steps,
		       policy = draft_policy.depth_policy;
	const bool defaulted = pipeline_steps_defaulted;
	session_defaults(1u);
	const char *const kept =
		tp_pipeline_overlap != 0 && pipeline_steps == 1u ?
			"--tp_pipeline_overlap needs two flights" :
		fit != NULL && !tensor_placement_selected() &&
				slots_fit(fit, limits) != STATUS_OK ?
			"one session does not fit under them" :
			NULL;
	if (kept != NULL) {
		pipeline_steps = depth;
		draft_policy.depth_policy = policy;
		pipeline_steps_defaulted = defaulted;
	}
	return kept;
}

/* The count --session_slots=max resolved to, with each card's free memory and budget (MiB); at one session, why the
 * multi-session defaults stay (kept: single_session_defaults' reason), if they do. */
static void sessions_max_report(const struct program_limits *limits,
				const uint64_t *budget_bytes_of_card,
				const char *kept)
{
	(void)printf("session slots: max -> %" PRIu32 " of %" PRIu32
		     " tokens each%s%s%s; cards (free/budget MiB):",
		     limits->session_capacity, limits->context_token_limit,
		     kept != NULL ? " (multi-session defaults kept: " : "",
		     kept != NULL ? kept : "", kept != NULL ? ")" : "");
	for (uint32_t card = 0; card < card_count; card++)
		(void)printf(
			" %" PRIu64 "/%" PRIu64,
			(uint64_t)(descriptions[card].memory_free_bytes /
				   BYTES_PER_MIB),
			(uint64_t)(budget_bytes_of_card[card] / BYTES_PER_MIB));
	(void)putchar('\n');
}

/* Every serve start: the most sessions of the context the cards hold, so the fitted count is always
 * visible. Where the selected row's set needs one session count (TP4_EXL3 BATCH's 8) and the serve runs its
 * concurrency=balanced row instead, the line says so. */
void sessions_fit_report(uint32_t most, const struct program_limits *limits,
			 const struct axis_choice *slots_unmet)
{
	if (slots_unmet != NULL)
		(void)printf("concurrency=%s: ",
			     slots_unmet->value[AXIS_CONCURRENCY_VALUE]);
	(void)printf("%" PRIu32 " session(s) fit at %" PRIu32
		     " tokens of context on this split",
		     most, limits->context_token_limit);
	if (!session_slots_max)
		(void)printf("; serving %" PRIu32 " (--session_slots)",
			     limits->session_capacity);
	(void)fputs("; a shorter --context_tokens gives more", stdout);
	if (slots_unmet != NULL)
		(void)printf("; BATCH needs exactly %u",
			     slots_unmet->serve_slots_exactly);
	(void)putchar('\n');
}

/* Not even one session of the context fits: refused, the context never shrunk. True (failed). */
static bool sessions_max_refused(const struct program_limits *limits)
{
	(void)fprintf(stderr,
		      "session slots: max: not even one session of %" PRIu32
		      " tokens fits the cards; the context is never shrunk "
		      "(give a smaller --context_tokens or --session_slots)\n",
		      limits->context_token_limit);
	return true;
}

/* An explicit --session_slots=N that the cards do not hold at the context: refused before any load, with the fit's
 * numbers (the most that fit at this context, the longest context of N, each card's free memory and budget); the
 * count and the context are never shrunk. True (failed). */
bool sessions_refused(const struct slot_fit *fit,
		      const struct program_limits *limits,
		      const uint64_t *budget_bytes_of_card)
{
	result_status most_refusal = STATUS_OK, context_refusal = STATUS_OK;
	const bool batch = batch_serving(); /* BATCH runs its count only */
	const uint32_t most = batch ? 0u :
				      slots_most(fit, limits, &most_refusal);
	const uint32_t context = context_most(fit, limits, &context_refusal);
	(void)fprintf(stderr,
		      "session slots: %" PRIu32 " sessions of %" PRIu32
		      " tokens do not fit the cards",
		      limits->session_capacity, limits->context_token_limit);
	if (!batch)
		(void)fprintf(
			stderr,
			"; at this context %" PRIu32
			" fit (--session_slots=max, the default, takes them)",
			most);
	if (context != 0)
		(void)fprintf(stderr,
			      "; %" PRIu32
			      " sessions fit up to --context_tokens=%" PRIu32,
			      limits->session_capacity, context);
	(void)fprintf(stderr, "%s; nothing is shrunk; cards (free/budget MiB):",
		      most_refusal != STATUS_OK ||
				      context_refusal != STATUS_OK ?
			      " (a trial plan was refused)" :
			      "");
	for (uint32_t card = 0; card < card_count; card++)
		(void)fprintf(
			stderr, " %" PRIu64 "/%" PRIu64,
			(uint64_t)(descriptions[card].memory_free_bytes /
				   BYTES_PER_MIB),
			(uint64_t)(budget_bytes_of_card[card] / BYTES_PER_MIB));
	(void)fputc('\n', stderr);
	return true;
}

int run_program(int argument_count, char **arguments,
		const struct axis_arguments_record *axis_arguments)
{
	uint64_t budget_bytes_of_card[BACKEND_MAXIMUM_CARDS] = { 0 };
	uint64_t weight_budget_bytes_of_card[BACKEND_MAXIMUM_CARDS] = { 0 };
	struct model_description_record model_description = { 0 };
	struct text_tokenizer *tokenizer = NULL;
	char *front_end_files[FRONT_END_FILES] = { NULL };
	uint64_t front_end_byte_counts[FRONT_END_FILES] = { 0 };
	char *prompt = NULL;
	size_t prompt_byte_count = 0;
	uint32_t *token_ids = NULL;
	uint32_t token_count = 0;
	uint32_t *card_of_unit = NULL;
	uint32_t on_demand_units_per_owner = 0;
	result_status status = STATUS_OK;
	double first_step_started_seconds = 0.0;
	bool failed = false;
	/* From the first log, a closed pipe is an error, not a signal.
	 * Descriptors 0 to 2 are open first: the lowest free descriptor proves those below it are open. */
	(void)signal(SIGPIPE, SIG_IGN);
	for (int descriptor = STDIN_FILENO; descriptor <= STDERR_FILENO;
	     descriptor++) {
		if (fcntl(descriptor, F_GETFD) < 0 &&
		    open("/dev/null", O_RDWR) != descriptor) {
			LOG_ERROR(
				"descriptor %d is closed and could not be opened",
				descriptor);
			return PROGRAM_FAILED;
		}
	}
	(void)setvbuf(stdout, NULL, _IOLBF,
		      0); /* a step's line appears when the step is done */
	if (argument_count == ARGUMENTS_BEFORE_THE_OPTIONS &&
	    strcmp(arguments[1], "--version") == 0)
		return fputs(PROGRAM_ATTRIBUTION_NOTICE, stdout) == EOF ?
			       PROGRAM_FAILED :
			       PROGRAM_SUCCEEDED;
	if (argument_count > 1 && strcmp(arguments[1], "stats") == 0) {
		return stats_client_cli(argument_count, arguments);
	}
	if (argument_count > 1 && strcmp(arguments[1], "keygen") == 0) {
		return server_keygen_cli(argument_count, arguments);
	}
	if (argument_count > 1 &&
	    (strcmp(arguments[1], "storage-profile") == 0 ||
	     strcmp(arguments[1], "storage-device") == 0 ||
	     strcmp(arguments[1], "storage-room") == 0 ||
	     strcmp(arguments[1], "storage-report") == 0 ||
	     strcmp(arguments[1], "storage-scan") == 0))
		return storage_profile_cli(argument_count, arguments);
	if (argument_count > 1 && strcmp(arguments[1], "autotune") == 0)
		return storage_autotune_cli(argument_count, arguments);
	if (argument_count > 1 && strcmp(arguments[1], "replicate") == 0) {
		return replicate_cli(argument_count, arguments);
	}
	if (argument_count == ARGUMENTS_BEFORE_THE_OPTIONS + 1 &&
	    strcmp(arguments[1], "serve") == 0 &&
	    strcmp(arguments[ARGUMENTS_BEFORE_THE_OPTIONS],
		   "--help-scheduling") == 0) {
		scheduling_usage(stdout);
		return PROGRAM_SUCCEEDED;
	}
	if (!parse_arguments(argument_count, arguments)) {
		/* The axis lines first, so a refused knob is traced to the axis value that inserted it. */
		axis_log(axis_arguments, stderr);
		if (is_capturing)
			(void)fputs(
				"usage: anemone capture [generate's options and axis flags] --capture_golden=ROOT"
				" --capture_cases=KEYS --capture_out=OUT [--capture_rows=1..8 (default"
				" speculative_depth + 1)] [--capture_positions=N (default all)] [--capture_free=0|1]"
				" TABLES PACK | --model_dir=FOLDER\n",
				stderr);
		/* Separate usage strings stay within ISO C99's 4095-byte literal limit. */
		(void)fputs(
			"usage: anemone load|generate|serve [--card_order=1,0,2]"
			" [--numerics=exact|relaxed|fast (default exact; relaxed/fast enable fixed alternate CMP sums)]"
			" [--placement=pp5|tp4_pro|tp4_exl3|tp2_pp2_pro|tp2_single_single_pro"
			" (default pp5; TP experimental)]"
			" [--exl3_expert_parts=4|8 (default 4; tp4_exl3 uses four CMPs, exact wire by default;"
			" 8 requires relaxed/fast numerics)]"
			" [--tp_code_wire=0|1 (default 0; exact original code/scale transport)]"
			" [--tp_attention_owner=0|1 (default 0; experimental TP4 attention state ownership)]"
			" [--tp_head_shards=0|1 (default 0; tp4_exl3: target head and sampler cut over the ranks, same"
			" bits)] [--tp_exact_dense=0|1 (default 1; tp4_exl3 exact recipe: attention wq_b/wo_a/wo_b"
			" output shards, same bits; other placements and the BF16 recipe ignore it)]"
			" [--tp_head_stream=0|1 (default 0; tp4_exl3: head products by the dense_stream kernel, same"
			" bits)] [--exl3_fast_products=0|1 (default 1; tp4_exl3: the wide kernel's EXL3 products,"
			" identical words; 0: the retained chain)]"
			" [--exl3_wide_entry=0|1 (default 1; tp4_exl3: a chunk's EXL3 experts in one call over its"
			" rows, the same words; 0: a call a slab)]"
			" [--exl3_tc_moe=0|1 (default 0; tp4_exl3 BF16 recipe: GNmac's tensor-core EXL3 MoE, its"
			" arithmetic (FP16 tensor-core products on a hi/lo split of the activations, FP64 transforms);"
			" the exact recipe ignores it)] [--exl3_expert_rotation=0|1|2 (default 0; with --exl3_tc_moe=1:"
			" 1 rank r of layer L holds chunk (r + L + e) % 4 of expert e (Morrow's); 2 rank 0 chunk 0,"
			" ranks 1-3 three ways: other TP sum order)] [--tp_head_mma=0|1 (default 0; tp4_exl3 BF16"
			" recipe: the head products (target, draft, a BATCH step's wide head) by BF16 tensor-core MMAs,"
			" FP32 sums in the MMA order, GNmac's class of head arithmetic)]"
			" [--tp_trunk_mma_thin_splits=0|1 (default 0; with --tp_trunk_mma=1: the _64 form splits as the"
			" 1-8-row form: a 9-64-row step's trunk rows get the thin steps' bits)]"
			" [--tp_trunk_mma_fused=0|1 (default 0; with --tp_trunk_mma=1: the _64 form adds its own"
			" splits, no reduce launch, the same bits; relbench03: 1-5 us slower than the form and its"
			" reduce)] [--tp_gather_balance=0..3 (default 0; tp4_exl3: a chunk's index scans and router"
			" rows by the 512-column expert ranks alone, the same bits; 2: wide steps' alone; 3: the"
			" 640-column ranks 3/16 each, the 512-column ones the rest)] [--tp_full_split_overlap=0..7"
			" (default 0: --tp_split_overlap's; the full steps' mode, the same bits)]"
			" [--tp_pool_chunk=0|1 (default 1;"
			" tp4_exl3: a chunk's pool entries and keys in a launch a piece ahead of its slabs' scans,"
			" the same bits)]"
			" [--tp_pool_gather=0|1 (default 1; tp4_exl3 split pipeline: a pooled source layer's compressor"
			" products a quarter of a half's rows a rank, gathered by copy engines, not on every rank;"
			" the same bits)]"
			" [--exl3_tcp_two_blocks=0|1 (default 1; with --exl3_tc_moe=1: the prefill TC MoE's GEMMs at"
			" two blocks a multiprocessor, 128 registers, Morrow's occupancy; the same bits)]"
			" [--tp_router_first_whole=0|1 (default 0; tp4_exl3 split pipeline: the first half's router"
			" over its rows on every rank, no gather beside the second half's attention sums; the same"
			" bits)]"
			" [--prefill_dead_halves=0|1|2 (default 2; 1 with the exact plan and the cone: a full program"
			" of 256 rows or more leaves the TP sums and gathers of a decoder layer's half whose outputs"
			" there the cone never reads (its last row at least 256 less the margin rows before the cone"
			" end); 2: and each sum's copies the rows before a half's first such row; the same bits)]"
			" [--tp_router_mma=0|1|2 (default 0; 2: prefill steps' routers too; tp4_exl3 BF16 recipe: a"
			" BATCH step's router (9-64 rows) by split-K"
			" BF16 tensor-core MMAs and a fixed-order reduce, GNmac's router class; another order than the"
			" dense one)]",
			stderr);
		(void)fputs(
			" [--weights=original|exl3 (default: what --placement and the pack imply; alone, exl3 selects"
			" tp4_exl3; a conflicting --placement or pack refuses)]"
			" [--model_dir=FOLDER (tokenizer.bin and original.anemone or NAME.exl3map (--exl3_map)"
			" from FOLDER, in place of the tables and pack arguments; links are followed to regular files)]"
			" [--exl3_map=NAME (with --model_dir and the EXL3 map: NAME.exl3map in FOLDER, default model;"
			" one folder holds several maps of one model)]"
			" [--exl3_unrated_map=0|1 (default 0; 1: an EXL3 map whose descriptor records no quality"
			" (quality none), or with no descriptor and a pack the model does not list as established, is"
			" served with no quality claim; refused otherwise)]",
			stderr);
		(void)fputs(
			" [--tp_queries_piece=0|1 (default 1; tp4_exl3: a wide program's queries over each piece with"
			" wq_b, from its compact heads, not a span at a time; the same bits)]"
			" [--tp_index_span=0|1 (default 1; tp4_exl3: a chunk's index scans and selections up to 64 rows"
			" a launch, not 8, within 32 MiB of scores a card; the same bits)]"
			" [--dense_skinny_bf16=0|4|8 (default 0; tp4_exl3 class 80: the BF16 skinny dense products"
			" above 48 rows at so many outputs a block, not 1; the same bits)]"
			" [--dense_skinny_f32=0|1|8 (default 0: 4; the float32-weight ones (routers) alike)]"
			" [--tp_core_compact_halves=0|1 (default 1; tp4_exl3: a split half's core writes wo_a's compact"
			" rows, no copy; the same bits)]"
			" [--tp_trunk_mma=0|1 (default 0; tp4_exl3 BF16 recipe: the trunk products of BF16 rows in"
			" steps of 1-8 rows by GNmac's FP8 gemv arithmetic (E4M3 weights made exact BF16, BF16"
			" tensor-core MMAs, FP32 sums))] [--tp_parallel_launch=0|1 (default 0; TP: the cards' graphs of"
			" a step's group launched from a thread a card, not in turn; the same launches a card, the same"
			" bits)] [--tp_graph_parts=1..8 (default 1; TP: a card's step graph behind the prefill cone in"
			" this many parts, so that a step begins after a smaller launch; the same launches in the same"
			" order, the same bits)] [--tp_graph_branches=0|1|2 (default 1; tp4_exl3 with --exl3_tc_moe=1:"
			" a thin layer's shared expert a step-graph branch beside its routed experts, joined by the"
			" ring, as GNmac's auxiliary stream, the same bits; 2: BATCH steps' too; 0: one chain)]"
			" [--tp_ordered_wide=0|1 (default 0; tp4_exl3: attention trunk products of more than 8 rows by"
			" the ordered wide form on the tensor cores, the reference's bits)]"
			" [--tp_piece_rings=0|1 (default 0; tp4_exl3 BF16 recipe: steps of more than 8 rows complete"
			" the FFN (and, with --tp_bf16_slab_rings=1, the attention) by BF16 rings over 64-row pieces,"
			" not 8-row slabs; same bits)] [--tp_trunk_shape_slices=0|1 (default 1; tp4_exl3: thin trunk"
			" products at the slice count measured for their shard shape, the same bits (ordered chains);"
			" 0: four slices)] [--tp_bf16_slab_rings=0|1 (default 0; tp4_exl3 BF16 recipe: chunked steps"
			" sum the attention a slab at a time by the ring, as thin steps, not one FP32 push a piece:"
			" other bits than 0)] [--tp_chunk_core=0|1 (default 1; tp4_exl3: a chunked step's attention"
			" core in one launch over the chunk's rows behind every slab's state writes and scan, not a"
			" slab at a time; the same bits; pf16: 30K-token prefill 18.93 -> 17.80 s; 0: a core a slab)]"
			" [--tp_split_least_rows=N (default 0; fewer rows unsplit)]"
			" [--tp_launch_trims=0|1 (default 1)]"
			" [--tp_split_overlap=0..7 (default 7: 6 with that call's gate/up tiles over both halves and"
			" its down GEMM and combine a half each, the first half's FFN sums beside the second's down,"
			" same bits; 6: 5 with one expert call a chunk, same bits;"
			" 5: pf25 14.54 -> 12.92 s, identical; tp4_exl3"
			" BF16 recipe: a chunked step's layer from wo_a on in"
			" two halves of its rows, each half's sums as a graph side run beside the other half's work"
			" (Morrow's PREFILL_REDUCE_OVERLAP on step graphs), the same bits; 2: the halves one behind the"
			" other; 3: only the attention's sums beside work; 4: only the FFN's; 5: 1 with the attention"
			" in halves too, each layer's last sums beside the next layer's first half, the same bits)]"
			" [--tp_wide_exchange=0|1|2|3|4|5 (default 5 with --tp_split_overlap=1: pf19 30K-token prefill"
			" 14.80 s against 19.08, identical; 4: the owner sums' bytes moved by copy engines; 5: 4 and"
			" the FFN's slab rings as copy-engine owner ring sums; both the same bits; was 2; tp4_exl3 BF16"
			" recipe: a chunked piece's sums: 2 each rank sums its shard of rows, half the push's bytes"
			" (pf07: 30K-token prefill 22.05 -> 18.85 s, the same bits); 1 the ordered push by many CTAs;"
			" 3: 2 and the FFN's slab rings as one exchange a piece (the same bits, slower: 23.25 s); 0:"
			" one CTA)] [--tp_replicated_embedding=0|1 (default 1; tp4_exl3: every rank embeds its own"
			" rows, the same bits; 0: card 0 embeds and hands the streams off through the host)]",
			stderr);
		(void)fputs(
			" [--batch_rows=R[,R..] (1..64 each; tp4_exl3 with --pipeline_steps=1: BATCH programs of those"
			" rows in place of the thin ones, several sessions' rounds a step; 8 rows or fewer walk as thin"
			" programs)] [serve: --original_batch=0|1 (default 0; original exact TP4_PRO BATCH48 pilot:"
			" eight slots/verifiers, depth1, DSpark5, context_slabs1, core_once0, draft_wide0,"
			" exl3_wide_entry0, split_overlap0; native per-verifier draft walks, wide head/draft rows off)]"
			" [--batch_draft_rows=0|1] [--batch_wide_head=0|1] [--batch_context_slabs=0|1 (default 0; a"
			" BATCH step's draft context products in slabs of 8 rows, the thin steps' forms)]"
			" [--batch_thin_bits=0|1 (default 1; a BATCH program's rows the thin steps' bits: trunk_mma"
			" thin splits, head split 4, TC MoE short forms)]"
			" [--batch_grouped_rows=0..48 (default 16; with it, a BATCH step of 9 to so many rows: trunk"
			" products by the 1-8-row forms over its row groups of 8, one launch, the same bits)]"
			" [--batch_tc_paired=0..64 (default 0; with it, BATCH steps of so many rows or more: the TC"
			" MoE's short forms two K slices a warp item, the same bits)] [--batch_row_sets=0|8|16|24"
			" (default 0; a BATCH step above --batch_grouped_rows: trunk products by the 1-8-row forms in"
			" sets of so many rows, the same bits)] [--batch_draft_slots_head=0|1 (default 0; 1: a BATCH"
			" program of one verifier"
			" runs its draft head over 8 slots a position, as before; drafts only)]"
			" [--batch_core_once=0..3 (default 3 for tp4_exl3, 2 elsewhere; 3: pools and deferred slots; a"
			" BATCH step's core and window once"
			" over its slabs; 1: the core alone; 0: a slab at a time; the same bits)]"
			" [--tp_front_branch=0|1|2 (default 0; a thin step's attention mix coefficients a graph branch"
			" beside the site; 2: the FFN site's too; the same bits)] [--tp_core_heads=0|8|16 (default 0;"
			" the attention core's blocks of at most so many heads, the same bits)]"
			" [--draft_bf16_rows=0|1 (default 0; the draft layers' wq_a / wkv / wq_b / wo_b of BF16 rows;"
			" drafts only)] [--batch_draft_wide=0..3 (default 3; a batched draft's sums and posts, 2: and"
			" products, over all its rows at once, same bits; 3: and cores, whole blocks, drafts only)]"
			" [--draft_bf16_operands=0|1 (default 0;"
			" class 80: the draft's quantized products of its rows' BF16 values; drafts only)]"
			" [--tp_ring_column_owners=0|1 (default 1; the BF16 ring sums' owners the quarters of each row,"
			" so a row's sum does not depend on its place in the step or the step's rows: other bits"
			" than 0)]"
			" [--batch_verifiers=V[,V..] --batch_step_us=0|US[,US..] (0: off; qualified BF16/DSpark5 menu"
			" defaults for rows 8,8,12,12,16,24,32,48 / verifiers 1,2,2,4,4,4,8,8 when policy is omitted)]"
			" [--batch_plan_ties=0|1 (default 0; 1: the plan verifies drafts of equal reach all or none;"
			" drafts only)]"
			" [--exl3_tc_product_blocks=0..592 (default 0: decode TC products use 148 blocks; same bits)]"
			" [--tp_decode_round_once=0|1 (default 0; BF16 decode partials summed in FP32 and rounded once;"
			" higher precision, new references)]"
			" [--tp_draft_shards=0|1 (default 0; with --tp_head_shards=1: the draft's head, Markov and"
			" sampler cut too)] [--tp_draft_layers=0|1 (default 0; BF16 recipe with --tp_draft_shards=1,"
			" BATCH with --batch_draft_rows=1 --batch_wide_head=1: the draft's three layers"
			" tensor-parallel, GNmac's DSpark, each rank its vocabulary cut of the draft; drafts only, the"
			" verified tokens unchanged)] [--tp_wire_bf16=0|1 (default 0; experimental relaxed TP wire"
			" cast, FP32 sums)] [--prefill_ladder_1536=0|1 (default 1; with --wide_rows above 1536:"
			" programs of 1536 rows in the ladder too, the exact plan's same bits in fewer steps)]"
			" [--prefill_ladder_fine=0|1 (default 1 with --wide_rows=2048 and no BATCH; 1792, 1280 and 768"
			" rows alike)]",
			stderr);
		/* the load's options: a literal of their own (each at most 4095 bytes, ISO C99) */
		(void)fprintf(
			stderr,
			" [--tp_load_threads=0..%u (default %u; TP placements: the cards' load on N workers beside the"
			" host arena's, the same bytes; 0: the serial loads)]"
			" [--tp_load_reads_per_copy=0..%u (default %u; reads in flight a copy at most, 0: no limit)]"
			" [--tp_load_host_threads=0..%u (default %u; the host arena's threads; 0: as many as"
			" --tp_load_threads)] [--tp_load_copy_split=0..2 (default 2: the host arena's chunks prefer"
			" copy 0, the cards' extents the last copy, and a role that has claimed its last read leaves"
			" its copy to the other; 1: each role only its preferred copy; 0: each read from the copy"
			" soonest to finish it)]"
			" [--storage_profile=PATH|off (default auto reload; never auto tune)]"
			" [--replica_external=0|1 (default 0; networks never qualify)]"
			" [--replica_floor_percent=0..99 (default 20)]"
			" [--engram_ram_percent=auto|0..100 (default auto: 100 when the tables fit MemAvailable,"
			" else the largest whole share that fits, else 0; immutable tables only)]"
			" [--engram_fit_only=0|1 (generate: print the Engram fit and stop before any card)]"
			" [--engram_sources=PACK[,PACK...] (optional independent table sources)]"
			" [--engram_io=0|1 (default 0; 1: prepared io_uring batches)]"
			" [--engram_cache_mib=0..32768 (default 0; bounded disk block cache)]"
			" [--turbo-nvme-engrams=0|1 (default 0; interleave admitted full replicas)]"
			" [--engram_step_io=0|1 (default 1 serving PP5 or TP4_PRO from disk tables, else 0; 1: a step's"
			" blocks read at once on I/O threads per copy)]"
			" [--engram_io_threads=1..4 (default 2)] [--engram_spin_us=0..100000 (default 200)]"
			" [--engram_io_pin=0|1 (default 1)]"
			" [--engram_scan_threads=0..4 (default 2: the startup digest scan reads every copy at once"
			" with two readers a copy; 0: one reader, copy by copy)]"
			" [--engram_hot_set=FILE --engram_hot_sha256=HEX (a static hot set of table blocks, from"
			" `engram_io_bench hot_build`, held in RAM and served like the RAM prefix; default none)]"
			" [--engram_hot_mib=1..65536 (default 512: the most RAM the hot set may hold)]"
			" [--turbo_load=0|1 (default 1: the source's receipted replicas, from `anemone replicate`,"
			" on drives that hold none of the given copies are read beside them; 0: exactly the given"
			" copies)]"
			" [--load_tuned_weights=0|1 (default 0: with a storage profile the readers steal pieces;"
			" 1: its weights pick each piece's copy)]"
			" [--map_reads=0..4 (default 4: 3 with replicas, else 1; 3: an EXL3 map's aligned pieces"
			" O_DIRECT, the rest buffered then dropped from the page cache; 0: all buffered and kept; 1:"
			" all buffered then dropped; 2: aligned O_DIRECT, rest kept)]",
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_THREADS,
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_READS_PER_COPY,
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_HOST_THREADS);
		(void)fprintf(
			stderr,
			" [--reader_threads_per_copy=1..%d (default %d: the PP loader's readers a copy)]"
			" [--card_memory_kept_free_mib=N]"
			" [--load_piece_mib=0..64 (default 0: 16 MiB; both loaders' read: the PP piece and slot, the TP"
			" host chunk; the knob wins over a profile, a profile over the default)]"
			" [--pro_emulation=0|1 (default 0: the PRO's own arithmetic, bits depend on the placement)]"
			" [generate and serve: --rows=1..%u"
			" --context_tokens=1..%u (default %u: 1M a session, a multiple of 16)"
			" --generated_tokens=N --seed=N --temperature=X --top_p=X"
			" --reasoning_effort=0..%d --linear_profile=0|1 --linear_order=0|1"
			" [--official_hc_rows=0|1|3|4|8 (default 0; experimental HC only, original thin PP)]"
			" [--official_reference=0|1 (default 0; original PP THIN serial M1 reference arithmetic)]"
			" [--cmp_order=reference|relaxed (default reference;"
			" relaxed changes CMP trunk/expert sums and state bits)]"
			" --step_graphs=0|1 (default 0/eager) --pipeline_steps=1..%u"
			" (default 5 for PP5 multi-session device THIN serving, 1 elsewhere)"
			" [--tp_pipeline_overlap=0|1 (default 0; 1: original TP4_PRO prefix and native tail on"
			" disjoint workers, depth>=2; per-flight downloads retire before prefix scratch reuse)]"
			" --prefill_cone=0|1"
			" --wide_rows=0|64..%u (default 0; with --prefill_cone=1) --full_steps=0|1 (default 1)"
			" [--prefill_margin=1..3967 (default 127; 3967 the reference's; with --chunks_in_flight=1,"
			" the exact plan"
			" (64-row chunks only where 1,024-row chunks, two in flight, take them): full rows beyond"
			" a prompt's cone; fewer: more front rows, the same bits, a kept prefix there runs a cone"
			" again)] [--prefill_fast_plan=0|1 (default 0; 1 with --chunks_in_flight=1: the largest"
			" chunk that fits, the reference tail's rows in larger chunks too: other bits)]"
			" [--prefill_order=0|1|2|3 (bounded BATCH default 2, else 0; 3: shortest with"
			" concurrent short tails;"
			" 2: prompts one at a time,"
			" the one with the fewest rows left first; 1: in their commands' order, each first token behind"
			" its own prompt; 0: the turns, a chunk each; the same chunks)] [--prefill_order_wait=0..1024"
			" (default 16; with 2: after N"
			" steps of shorter prompts in a row, a step of the oldest; 0: none)]"
			" [--prefill_cone_skip=0..3 (default 1: a full step's rows past the cone skip the decoder"
			" layers' cores, routed experts and wide trunks, the same bits; a later prompt keeping"
			" --prefill_margin tokens fewer or more runs no cone again; 2: the last row's cone, one fewer;"
			" 3: 1 with every trunk; 0: none skip)]"
			" --chunks_in_flight=1..%u (default 1, the exact plan; 2 the reference plan)"
			" [--prefill_first=0|1] [--draft_learned_acceptance=0|1 (default 1 for tp4_exl3, 0 elsewhere)]"
			" [--draft_session_acceptance=0..11 (default 0)]"
			" [--draft_start_economic=0|1 (default 1; 1: a"
			" command's first DSpark round at the policy's depth, not a probe of one draft)]"
			" --speculative=0|1|2|3 (default 0; with --rows=2 or more) --draft=lookup|dspark|hybrid"
			" --draft_block=5..7 (default 5)"
			" --draft_positions=1..%u (default %u, at most --rows less 1)"
			" --speculative_depth=N (alias; device draft at most --draft_block;"
			" multi-session needs pipeline)"
			" --draft_policy=0..3 (fixed|cost|rate|adaptive; tp4_exl3 default 0 single-slot THIN, 2"
			" BATCH/multiple; elsewhere 1) --draft_threshold=0..1000 (default 100)"
			" --draft_base_us=1..1000000000 (tp4_exl3 default 14300, elsewhere 12700)"
			" --draft_tail_us=0..1000000000 (tp4_exl3 default 0, elsewhere 2200)"
			" --draft_row_us=1..1000000000 (tp4_exl3 default 690, elsewhere 2700)"
			" --draft_probe_rounds=1..1048576 (default 8)"
			" --hybrid_lookup_prior=0..1000 (default 500, weight two reached outcomes)"
			" --draft_calibration=FILE (112 probabilities per thousand: mode/position/bin)"
			" --hybrid_lookup_min=1..16 (default 3; suffix match, at most --lookup_ngram_max)"
			" --lookup_ngram_max=1..%u (default %u) --lookup_ngram_min=1..max (default %u)]"
			" [--layout=<placement text (placement.h), in place of the fill>]",
			LOADER_MAXIMUM_READER_THREADS_PER_COPY,
			LOADER_DEFAULT_READER_THREADS_PER_COPY,
			LINEAR_MAXIMUM_ROWS, DS41F_CONTEXT_MOST,
			DS41F_CONTEXT_MOST, DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT,
			STEP_PIPELINE_MOST, STEP_WIDE_MOST_ROWS,
			STEP_PIPELINE_MOST, SERVING_DRAFT_MOST,
			SERVING_DRAFT_MOST, DRAFT_LOOKUP_MOST_NGRAM,
			DEFAULT_LOOKUP_NGRAM_MOST, DEFAULT_LOOKUP_NGRAM_LEAST);
		(void)fprintf(
			stderr,
			" [serve: --session_slots=1..%u|max (default max: as many sessions as fit at the"
			" context, no cap)"
			" --queue_depth=1..%u (default %u)"
			" --user_queue_depth=1..queue_depth (default half queue_depth, at least 1)"
			" --draft_usage=none|compact|full (default compact)"
			" --dsml_parsing=tolerant|reference (default tolerant)"
			" --bind_address=%s --port=1..%u --key_file=<file> --model_alias=<name>"
			" --session_directory=<snapshots of sessions>"
			" --prefix_reuse=0|1 (default 1; 0 disables GPU prefix reuse and refuses session_directory)"
			" --prefix_reuse_exact=0|1|2 (default 0; 1 final-row repeat; 2 cold boundaries;"
			" fixed exact plan; prefill_chunk_rows=prefill_decode_rows=0;"
			" multiple slots require prefill_first=1 or prefill_order; no snapshots)"
			" --metrics=0|1 --live_stats=0|100..60000 --live_stats_format=text|json|table"
			" --stats_users=<names>] [generate: <messages file>]"
			" [generate and serve: <front-end tables file>] <pack copy> [... up to %d copies]\n",
			SERVING_SLOTS_MOST, SERVER_MOST_WAITING,
			SERVER_DEFAULT_QUEUE_DEPTH, SERVER_DEFAULT_ADDRESS,
			SERVER_MOST_PORT, STORAGE_MAXIMUM_COPIES);
		(void)fputs(
			"       anemone stats [--host=127.0.0.1] --port=N [--key_file=F]"
			" [--interval_ms=1000] [--format=text|json|table] [--once]\n"
			"       anemone keygen --user=<username>\n",
			stderr);
		/* the tail programs' options: a literal of their own (each at most 4095 bytes, ISO C99) */
		(void)fputs(
			"       serve caches: [--cache_scope=0|1|2 (default 0: API user plus session/cache signals;"
			" 1: API user; 2: global)] [--session_grace_seconds=0..3600"
			" (default 30; 0 disables affinity grace)]"
			" [--anthropic_messages=0|1 (default 0; bounded Messages API)]"
			" [--session_fork=0|1 (default 0; exact same-owner independent slot clone, cache_scope=0)]"
			" [--session_save_tokens=N (default 1024)] [--session_load_tokens=N (default 256)]"
			" [--session_idle_save_seconds=1..86400 (default 120)]\n",
			stderr);
		(void)fputs(
			"       generate and serve, the exact plan's tail: [--prefill_tail_rows=0|128|320|448"
			" (default 128, 448 with --wide_rows=2048 and no BATCH, where it"
			" applies: tp4_exl3, --chunks_in_flight=1, --wide_rows of 1024 or more, --exl3_tc_moe=1,"
			" --tp_trunk_mma=1: the exact plan's 64-row tail chunks two (320: five, 448: seven or five,"
			" else two) a step, each in the 64-row chunk's forms; 0 one a step)]"
			" [--prefill_ladder_1024=0|1 (default 1; 1 with --wide_rows above 1024: programs of 1024 rows"
			" in the ladder too, the exact plan's same bits in fewer steps)]"
			" [--prefill_front_tail=0|1|2 (default 1 where the tail program applies, 2 with the 448-row"
			" tail: a wide one for the front region's tail too, its bits; 2: its 320- and 448-row ones and"
			" the thin-tail ones too)]"
			" [--prefill_tail_overlay=0|1 (default 1: the tail programs'"
			" TC MoE workspace over the prefill MoE's, the same bits; 0 its own)] [--prefill_tail_sets=0|1"
			" (default 1: a tail program's trunk products one launch over its 64-row groups, each group's"
			" bits; 0 a launch a group)]"
			" [--prefill_thin_tail=0|1 (default 1 with the tail program and no BATCH: full programs of 16"
			" to 56 rows take the tail's"
			" rows short of a 64-row chunk in one step, each row in a thin step's forms)]"
			" [--dense_staged=0|8|16|32|64 (default 16: a wide dense product's 8-row group"
			" staged once in shared memory for the selected number of outputs, each output's sums as"
			" before: the same bits; 0 the wide forms)]"
			" [--original_dense_staged=0|16 (default 0; original TP4 CMP BF16 wide dense candidate;"
			" 0 retains the old forms)]"
			" [--original_head_offload=0|1|2 (default 0; experimental native PP5 target head: 1 leased, 2"
			" flight banks on four CMP cuts; PRO normalization/sampler/draft unchanged; 331 MB weights per"
			" CMP plus scratch)] [--original_wide_experts=0|1 (default 1 for compatible TP4_PRO; 0: slab"
			" path; wide jobs across the chunk, eight-row lossless gathers; requires --exl3_wide_entry=0"
			" --tp_split_overlap=0)] [--original_chunk_core=0|1 (default 0; 1: exact original full-head"
			" core over one chunk of whole eight-row slabs;"
			" state writes/scans stay in eight-row order; no BATCH, wide experts or split)]"
			" [serve, with --prefill_order=1|2|3: --prefill_decode_every=0..64 (default 0; N: after N chunk"
			" steps of the head prompt in a row, a step of the decoding sessions first; 0: the turns"
			" alone)]\n",
			stderr);
		(void)fputs(
			"       TP n-gram gathers: [--engram_shared=0|1 (default 0;"
			" identical table masks share host rows)]\n"
			"       exact prefill plan: [--prefill_chunk_rows=0|256..2048 (whole 256-row granules;"
			" default 0; cap chunks before the original tail)] [serve: --prefill_decode_rows=0|256..2048"
			" (BATCH recipe default 1024, other recipes 0; cap while a decoder is ready;"
			" --prefill_decode_every=N overrides the interval)]"
			" [--prefill_decode_rounds=1..64 (ordinary decode rounds per interval;"
			" see --help-scheduling for defaults)]"
			" [--prefill_decode_waiters=0|1 (BATCH default 0, others 0; 1: original chunks and one decode"
			" round while"
			" multiple prompts remain)]\n",
			stderr);
		(void)fputs(
			"       scheduling examples and measured tradeoffs: anemone serve --help-scheduling\n",
			stderr);
		(void)fputs("       prefix-stable prefill: "
			    "[--prefill_prefix_stable=0|1 (default 0; "
			    "experimental TP4 BF16"
			    " fixed-capacity plan at wide_rows 1024/2048; new "
			    "cold references; one chunk, caps 0,"
			    " reserved prefill; no images or snapshots)]\n",
			    stderr);
		(void)fputs(
			"       generate explicit tokens: anemone generate [OPTIONS] --prompt_token_ids=IDS_JSON"
			" TABLES WEIGHTS (one pack; replaces MESSAGES; JSON uint32 array, no chat formatting).\n"
			"       generate/serve: --ignore_eos=0|1 (default 0; non-speculative THIN only;"
			" 1 continues after end-token draws to the token/context limit).\n",
			stderr);
		(void)fputs(
			"       anemone replicate [--rate_mib=N] [--floor_gib=N] [--prune=0|1] SOURCE [DIRECTORY ...]"
			" (turbo load's replicas: replicas.h)\n",
			stderr);
		(void)fputs("       anemone --version\n", stderr);
		return PROGRAM_WRONG_USAGE;
	}
	/* Product attribution stays on stderr; generated-token stdout retains its protocol. */
	if (fputs(PROGRAM_ATTRIBUTION_NOTICE, stderr) == EOF)
		return PROGRAM_FAILED;
	storage_profile_apply(argument_count, arguments);
	if (!storage_profile_is_valid && !storage_profile_is_off)
		storage_autotune_hint(stderr);
	storage_profile_roles(tuned_load_active, tuned_engram_active,
			      engram_ram_percent < ENGRAM_RAM_PERCENT);
	axis_log(axis_arguments, stderr);
	if (!engram_settings_resolve())
		return PROGRAM_WRONG_USAGE;
	const int placement_status = startup_placement_status();
	if (placement_status != PROGRAM_SUCCEEDED)
		return placement_status;
	first_step_started_seconds = step_started_seconds = monotonic_seconds();
	if (model_dir != NULL) {
		failed = step_failed("resolve the model folder",
				     model_folder_files(is_generating));
		status = failed ? STATUS_INPUT_OUTPUT_ERROR : status;
	}
	for (uint32_t file = 0;
	     is_generating && status == STATUS_OK && file < FRONT_END_FILES;
	     file++) {
		front_end_files[file] =
			front_end_paths[file] != NULL ?
				file_read(
					front_end_paths[file],
					file == TABLES_FILE &&
							model_dir != NULL ?
						&model_files
							[CLI_MODEL_FILE_TABLES] :
						NULL,
					&front_end_byte_counts[file]) :
				NULL;
		status = front_end_paths[file] != NULL &&
					 front_end_files[file] == NULL ?
				 STATUS_INPUT_OUTPUT_ERROR :
				 status;
	}
	status = !is_generating || status != STATUS_OK ?
			 status :
			 tokenizer_open(front_end_files[TABLES_FILE],
					front_end_byte_counts[TABLES_FILE],
					&tokenizer);
	if (is_serving) {
		const struct server_options options = {
			.storage_replica_info = storage_replica_info,
			.storage_replica_info_bytes =
				storage_replica_info_bytes,
			.bind_address = bind_address,
			.anthropic_messages = anthropic_messages != 0,
			.key_file = key_file,
			.stats_users = stats_users,
			.model_alias = model_alias,
			.disable_prefix_reuse = prefix_reuse == 0,
			.exact_prefix_reuse = prefix_reuse_exact,
			.cache_scope = cache_scope,
			.session_fork = session_fork,
			.session_grace_seconds = session_grace_seconds,
			.session_save_tokens = session_save_tokens,
			.session_load_tokens = session_load_tokens,
			.session_idle_save_seconds = session_idle_save_seconds,
			.session_directory = session_directory,
			.session_directory_bytes =
				SERVER_SESSION_DIRECTORY_BYTES,
			.metrics = metrics != 0,
			.live_stats_interval_ms = live_stats_interval_ms,
			.live_stats_json = strcmp(live_stats_format, "json") ==
					   0,
			.live_stats_table =
				strcmp(live_stats_format, "table") == 0,
			.card_order = stats_card_order,
			.placement = stats_placement,
			.port = port,
			.queue_depth = queue_depth,
			.user_queue_depth = user_queue_depth,
			.defaults = { .seed = seed,
				      .max_tokens = generated_token_limit,
				      .reasoning_effort = reasoning_effort,
				      .draft_usage =
					      strcmp(draft_usage, "full") == 0 ?
						      CHAT_DRAFT_USAGE_FULL :
					      strcmp(draft_usage, "none") == 0 ?
						      CHAT_DRAFT_USAGE_NONE :
						      CHAT_DRAFT_USAGE_COMPACT,
				      .temperature = temperature,
				      .top_p = top_p,
				      .is_seeded = is_seed_given,
				      .is_calls_reference =
					      strcmp(calls_parsing,
						     "reference") == 0 }
		};
		failed = step_failed("open the tables and the socket",
				     status != STATUS_OK ?
					     status :
					     server_open(&options, &server));
	} else if (is_generating && !is_capturing) {
		const struct model_prompt_options options = {
			.reasoning_effort =
				reasoning_effort != 0 ?
					reasoning_effort :
					DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT,
			.is_thinking = reasoning_effort != 0,
			.drops_thinking = true
		};
		const size_t prompt_capacity =
			(size_t)context_tokens * TOKENIZER_MAXIMUM_TOKEN_BYTES;
		prompt = prompt_token_ids_path == NULL ?
				 malloc(prompt_capacity) :
				 NULL;
		token_ids = calloc(context_tokens, sizeof(*token_ids));
		struct model_prompt_spans trusted = { 0 };
		status = status != STATUS_OK ?
				 status :
			 (prompt_token_ids_path == NULL && prompt == NULL) ||
					 token_ids == NULL ?
				 STATUS_OUT_OF_MEMORY :
			 prompt_token_ids_path != NULL ?
				 prompt_token_ids_parse(
					 front_end_files[MESSAGES_FILE],
					 (size_t)front_end_byte_counts
						 [MESSAGES_FILE],
					 tokenizer_vocabulary_size(tokenizer),
					 token_ids, context_tokens,
					 &token_count) :
				 ds41f_chat_template_render_spans(
					 front_end_files[MESSAGES_FILE],
					 front_end_byte_counts[MESSAGES_FILE],
					 NULL, 0, &options, prompt,
					 prompt_capacity, &prompt_byte_count,
					 &trusted);
		const bool is_rendered = status == STATUS_OK;
		status = is_rendered && prompt_token_ids_path == NULL ?
				 tokenizer_encode_spans(
					 tokenizer, prompt, prompt_byte_count,
					 trusted.spans, trusted.count,
					 token_ids, context_tokens,
					 &token_count) :
				 status;
		free(trusted.spans);
		/* Too long: render returns UNSUPPORTED, encode INVALID_ARGUMENT (its only refusal here).
		 * One more token must fit. */
		status = (is_rendered && status == STATUS_INVALID_ARGUMENT) ||
					 (status == STATUS_OK &&
					  (token_count == 0 ||
					   token_count == context_tokens)) ?
				 STATUS_UNSUPPORTED :
				 status;
		if (status == STATUS_UNSUPPORTED) {
			LOG_ERROR(
				"the prompt is empty, or it and one more token exceed --context_tokens");
		}
		failed = step_failed(prompt_token_ids_path != NULL ?
					     "read prompt token ids" :
					     "encode the prompt",
				     status);
	}
	if (!failed && copy_count != 0)
		storage_operations = pack_storage();
	/* auto opens the direct descriptors (DIRECT_DROPPED) and falls back to DROPPED when no replica is found
	 * below */
	storage_exl3_read_mode(map_reads == MAP_READS_AUTO ?
				       EXL3_READS_DIRECT_DROPPED :
				       (enum exl3_read_mode)map_reads);
	failed =
		failed ||
		step_failed("open the copies",
			    storage_operations->reader_open(
				    copy_paths, copy_count, &reader)) ||
		step_failed("open the pack",
			    pack_open_entry(storage_operations, reader, &pack));
	if (!failed && model_pack != NULL)
		failed = step_failed("check the folder's pack",
				     model_pack_opened(storage_operations,
						       reader,
						       pack_header_of(pack)));
	if (!failed && weights_family != WEIGHTS_UNSET)
		failed = step_failed("check the weight family",
				     pack_family_status(pack));
	if (!failed)
		model_instance_bind(pack,
				    model_pack != NULL ? model_pack->path :
							 pack_argument,
				    &instance);
	if (!failed && pack_holds_exl3(pack))
		failed = step_failed(
			"check the map's descriptor",
			map_descriptor_check(
				front_end_files[TABLES_FILE],
				front_end_byte_counts[TABLES_FILE]));
	if (!failed)
		failed = !turbo_load_copies(pack_header_of(pack));

	for (uint32_t copy = 0;
	     !failed && !tuned_load_active && copy < copy_count; copy++) {
		struct local_device_description device = { 0 };
		/* an EXL3 map: its largest source's drive */
		char bytes_path[EXL3_MAP_PATH_BYTES];
		const result_status described = local_device_describe(
			storage_device_path(copy_paths[copy], bytes_path),
			&device);
		const char *reason =
			described != STATUS_OK ?
				"device_disappeared_or_unknown" :
				local_device_media_refusal(
					&device, replica_external != 0);
		failed = step_failed(
			"record storage selection",
			storage_profile_note_loaded(stderr, &device, reason,
						    replica_floor_percent));
	}

	if (!failed && is_generating)
		failed = step_failed("fit the Engram tables in host memory",
				     engram_ram_fit());
	if (!failed && engram_ram_percent < ENGRAM_RAM_PERCENT) {
		status = engram != NULL ? STATUS_OK :
					  engram_open(storage_operations,
						      reader, pack, &engram);
		if (status == STATUS_OK)
			status = engram_sources_prepare();
		if (status == STATUS_OK && engram_hot_set_path != NULL)
			status = engram_hot_start();
		if (status == STATUS_OK)
			status = engram_attach_disk(engram, engram_disk);
		if (status == STATUS_OK && engram_step_io != 0)
			status = engram_steps_start_or_fall_back();
		else if (status == STATUS_OK)
			(void)fprintf(
				stderr,
				"engram step io: off, per-row gathers at each card's visit (%s)\n",
				engram_step_io_default ?
					"not given: the default only serving PP5 and TP4_PRO; "
					"--engram_step_io=1 reads a step's blocks at submit" :
					"--engram_step_io=0");
		failed = step_failed("prepare disk Engram tables", status);
		if (failed && engram_ram_source == ENGRAM_RAM_AUTO)
			engram_fit_disk_refusal_print(
				stderr, engram_ram_percent,
				storage_operations == &storage_direct);
	}
	/* --engram_fit_only: the fit, and disk tables it chose prepared (their sources admitted and scanned) */
	const bool stops_after_fit = !failed && engram_fit_only != 0;
	failed = failed || stops_after_fit;
	if (map_reads == MAP_READS_AUTO && copy_count == 1)
		storage_exl3_read_mode(EXL3_READS_DROPPED);
	if (!failed && is_generating) {
		instance.operations = instance.model != NULL ?
					      instance.model->operations :
					      NULL;
		failed = step_failed("describe the model",
				     instance.operations == NULL ?
					     STATUS_UNSUPPORTED :
					     instance.operations->describe(
						     pack,
						     &model_description)) ||
			 step_failed("match the tokenizer",
				     tokenizer_vocabulary_size(tokenizer) ==
						     model_description
							     .vocabulary_size ?
					     STATUS_OK :
					     STATUS_INVALID_ARGUMENT);
	}
	struct program_limits limits = options_limits();
	if (!failed) {
		const uint64_t bytes_kept_free =
			(uint64_t)card_memory_kept_free_mib * BYTES_PER_MIB;
		const uint32_t unit_count = pack_header_of(pack)->unit_count;
		card_of_unit = calloc(unit_count, sizeof(*card_of_unit));
		status = serve_cards_describe(&backend_cuda_value);
		failed = step_failed("describe the cards", status);
		if (!failed)
			serve_cards_arithmetic();
		/* a card's budget: its free memory less what the backend reserves and less what the user keeps free */
		for (uint32_t card = 0; card < card_count && !failed; card++) {
			const struct card_description_record *const description =
				&descriptions[card];
			const uint64_t unreserved_bytes =
				description->memory_free_bytes >
						description
							->memory_reserved_bytes ?
					description->memory_free_bytes -
						description
							->memory_reserved_bytes :
					0;
			budget_bytes_of_card[card] =
				unreserved_bytes > bytes_kept_free ?
					unreserved_bytes - bytes_kept_free :
					0;
			weight_budget_bytes_of_card[card] =
				budget_bytes_of_card[card];
		}
		/* --session_slots=max: the most sessions the cards hold, by the same passes, plan and budgets as the
		 * placement below; on the TP splits the TP plan refuses a count it does not hold */
		uint32_t fitting = session_slots_max ? 0u : session_slots;
		const struct session_fit_cards fit_cards = {
			.model = instance.operations,
			.pack = pack,
			.layout_text = layout_text,
			.budget_bytes_of_card = budget_bytes_of_card,
			.card_count = card_count,
			.unit_count = unit_count,
			.card_of_unit = card_of_unit,
			.plan_grown = plan_grown
		};
		const struct slot_fit fit = { &fit_cards, bytes_kept_free,
					      card_of_unit };
		/* An explicit count's fitted most, for the report: searched before its TP plan is made (each trial plan
		 * is released). A refusal other than memory is no fit: the placement
		 * below, or the TP plan, makes it again at one session and names its step. */
		result_status fit_refusal = STATUS_OK;
		const uint32_t fitted_most =
			!failed && is_serving && !session_slots_max ?
				slots_most(&fit, &limits, &fit_refusal) :
				0u;
		if (!failed && session_slots_max &&
		    !tensor_placement_selected())
			fitting = session_fit_most(&fit_cards, &limits,
						   SERVING_SLOTS_MOST,
						   &fit_refusal);
		/* an explicit count the cards do not hold: refused with the fit's numbers, before any load */
		if (!failed && is_serving && !session_slots_max &&
		    !tensor_placement_selected() &&
		    session_fit(&fit_cards, &limits) == STATUS_OUT_OF_MEMORY)
			failed = sessions_refused(&fit, &limits,
						  budget_bytes_of_card);
		if (tensor_placement_selected()) {
			const result_status planned =
				failed ? STATUS_OK :
					 tensor_plan_fit(&limits,
							 bytes_kept_free,
							 card_of_unit);
			fitting = limits.session_capacity;
			failed = failed ||
				 (is_serving && !session_slots_max &&
				  planned == STATUS_OUT_OF_MEMORY &&
				  sessions_refused(&fit, &limits,
						   budget_bytes_of_card)) ||
				 (session_slots_max &&
				  planned == STATUS_OUT_OF_MEMORY &&
				  sessions_max_refused(&limits)) ||
				 step_failed("plan TP2 shards and state",
					     planned);
		} else if (!failed && session_slots_max && fitting == 0u &&
			   fit_refusal == STATUS_OK) {
			failed = sessions_max_refused(&limits);
		}
		if (!failed && session_slots_max) {
			limits.session_capacity = fitting != 0u ? fitting : 1u;
			session_slots = limits.session_capacity;
			if (fit_refusal == STATUS_OK)
				sessions_max_report(
					&limits, budget_bytes_of_card,
					fitting == 1u ? single_session_defaults(
								&fit, &limits) :
							NULL);
		}
		if (!failed)
			tp_queue_depth_report();
		if (!failed && is_serving && fit_refusal == STATUS_OK)
			sessions_fit_report(
				session_slots_max ? fitting : fitted_most,
				&limits,
				axis_arguments != NULL ?
					axis_arguments->slots_unmet :
					NULL);
		if (!tensor_placement_selected()) {
			/* Generate: subtract the previous placement's state, then place again. These budgets
			 * are replayed by ds41f_placement_test.c; changes must be made there too. */
			const struct placement_record placement = {
				card_of_unit, unit_count, card_count
			};
			status = failed		      ? status :
				 card_of_unit == NULL ? STATUS_OUT_OF_MEMORY :
							STATUS_OK;
			const uint32_t pass_count =
				is_generating ? PLACEMENT_STATE_PASSES : 1;
			for (uint32_t pass = 0;
			     pass < pass_count && status == STATUS_OK; pass++) {
				status =
					layout_text != NULL ?
						placement_from_layout_text(
							pack, layout_text,
							weight_budget_bytes_of_card,
							card_count, unit_count,
							card_of_unit) :
						placement_fill_in_order(
							pack,
							weight_budget_bytes_of_card,
							card_count, unit_count,
							card_of_unit,
							&on_demand_units_per_owner);
				for (uint32_t card = 0;
				     is_generating && card < card_count &&
				     status == STATUS_OK;
				     card++) {
					uint64_t state_bytes = 0;
					status = instance.operations
							 ->state_byte_count(
								 pack,
								 &placement,
								 &limits, card,
								 &state_bytes);
					weight_budget_bytes_of_card[card] =
						budget_bytes_of_card[card] >
								state_bytes ?
							budget_bytes_of_card
									[card] -
								state_bytes :
							0;
				}
			}
			uint64_t pinned_bytes =
				0; /* the routed experts left in host memory: refused before the load */
			status = !is_generating || status != STATUS_OK ?
					 status :
					 instance.operations->state_byte_count(
						 pack, &placement, &limits,
						 PLACEMENT_IN_HOST_MEMORY,
						 &pinned_bytes);
			if (is_generating && status == STATUS_OK &&
			    pinned_bytes != 0) {
				LOG_ERROR("the placement leaves %" PRIu64
					  " routed-expert bytes in host memory",
					  pinned_bytes);
				status = STATUS_UNSUPPORTED;
			}
			failed =
				failed ||
				step_failed("budget the cards", STATUS_OK) ||
				step_failed(
					speculative >= SPECULATIVE_DEVICE ?
						"place DSpark (layers 37-39, head and draft need one card: "
						"--layout / --card_order)" :
						"place the units",
					status) ||
				step_failed("lay out the extents",
					    load_layout_create(
						    pack, card_of_unit,
						    unit_count, card_count,
						    &layout)) ||
				(is_generating &&
				 step_failed(
					 speculative >= SPECULATIVE_DEVICE ?
						 "plan DSpark (layers 37-39, head and draft need one card: "
						 "--layout / --card_order)" :
						 "plan the state",
					 instance.operations->plan_create(
						 pack, layout, &limits,
						 &plan)));
		}
	}
	/* the arenas grown as the fit grew them (plan_grown), each step named */
	if (!failed && original_head_offload != 0) {
		failed = step_failed("plan original target head offload",
				     ds41f_original_head_plan(plan,
							      descriptions));
	}
	if (!failed && original_head_offload == DS41F_HEAD_MODE_BANKED)
		failed = step_failed(
			"plan original head flight banks",
			ds41f_original_head_banks_plan(plan, pipeline_steps));
	if (!failed && tensor_model != NULL) {
		failed = step_failed(
			"load and qualify TP2 arenas",
			tensor_cards_load(front_end_files[TABLES_FILE],
					  front_end_byte_counts[TABLES_FILE]));
	}
	if (!failed && tensor_model == NULL) {
		struct load_request request = {
			.storage = storage_operations,
			.reader = reader,
			.backend = &backend_cuda_value,
			.pack = pack,
			.layout = layout,
			.external_host_extents =
				engram_disk_external_extents(engram_disk),
			.cards = cards,
			.card_descriptions = descriptions,
			.reader_threads_per_copy = reader_threads_per_copy,
			.piece_bytes = (uint64_t)load_piece_mib * BYTES_PER_MIB,
			.stall_limit_nanoseconds =
				LOADER_DEFAULT_STALL_LIMIT_NANOSECONDS
		};
		if (tuned_load_active && load_tuned_weights != 0)
			memcpy(request.copy_weights, tuned_load_weights,
			       sizeof(request.copy_weights));
		/* Open receiving cards with an arena for their placed weights and model state. */
		for (uint32_t card = 0;
		     card < card_count && status == STATUS_OK; card++) {
			uint64_t arena_bytes = layout->card_byte_count[card];
			status = is_generating ?
					 instance.operations->arena_byte_count(
						 plan, card, &arena_bytes) :
					 STATUS_OK;
			status = status == STATUS_OK &&
						 arena_bytes >
							 budget_bytes_of_card
								 [card] ?
					 STATUS_OUT_OF_MEMORY :
					 status;
			if (status == STATUS_OK && arena_bytes != 0) {
				status = backend_cuda_value.card_open(
					card_order[card], arena_bytes,
					&cards[card]);
			}
		}
		failed = step_failed("open the cards", status) ||
			 step_failed("transfer and check",
				     load_run_entry(&request, &result)) ||
			 (is_generating &&
			  step_failed(
				  "bind the front end",
				  tokenizer_tables_match(
					  pack, layout, &result,
					  front_end_files[TABLES_FILE],
					  front_end_byte_counts[TABLES_FILE])));
	}
	if (!failed) {
		if (tensor_model == NULL) {
			/* a destination's rate is its bytes over the time until its own last piece arrived */
			const double transfer_seconds =
				(double)result.transfer_nanoseconds /
				NANOSECONDS_PER_SECOND;
			const double host_seconds =
				(double)result.host_arrival_nanoseconds /
				NANOSECONDS_PER_SECOND;
			const double host_gib =
				(double)layout->host_byte_count / BYTES_PER_GIB;
			for (uint32_t card = 0; card < card_count; card++) {
				const double card_seconds =
					(double)result
						.card_arrival_nanoseconds[card] /
					NANOSECONDS_PER_SECOND;
				const double card_gib =
					(double)layout->card_byte_count[card] /
					BYTES_PER_GIB;
				(void)printf(
					"card %" PRIu32
					" (%.*s, %.*s): %.2f GiB, %.2f GiB/s\n",
					card_order[card],
					CARD_NAME_CAPACITY - 1,
					descriptions[card].name,
					CARD_BUS_ADDRESS_CAPACITY - 1,
					descriptions[card].bus_address,
					card_gib,
					card_seconds > 0.0 ?
						card_gib / card_seconds :
						0.0);
			}
			for (uint32_t copy_index = 0; copy_index < copy_count;
			     copy_index++) {
				(void)printf(
					"copy %s: %.2f GiB read\n",
					copy_paths[copy_index],
					(double)result.byte_count_read_of_copy
							[copy_index] /
						BYTES_PER_GIB);
			}
			(void)printf(
				"host memory %.2f GiB, %.2f GiB/s, %" PRIu32
				" on-demand units per stage unit on its card, "
				"transfer %.3f s, check %.3f s, %" PRIu32
				" extents read again, %" PRIu32
				" copies dropped, "
				"total %.3f s\n",
				host_gib,
				host_seconds > 0.0 ? host_gib / host_seconds :
						     0.0,
				on_demand_units_per_owner, transfer_seconds,
				(double)result.check_nanoseconds /
					NANOSECONDS_PER_SECOND,
				result.extents_read_again,
				result.copies_dropped,
				step_started_seconds -
					first_step_started_seconds);
		}
		/* a snapshot's bits follow the layers native cards compute: they salt its identity (0: no such card) */
		uint64_t arithmetic_salt =
			tensor_model != NULL ? tensor_model->recipe_identity :
					       0;
		for (uint32_t unit = 0; unit < pack_header_of(pack)->unit_count;
		     unit++) {
			const struct pack_unit_record_2 *const pack_unit =
				&pack_units_of(pack)[unit];
			if (pack_unit->unit_kind == PLACEMENT_UNIT_STAGE &&
			    pack_unit->layer_index != PACK_NO_LAYER &&
			    descriptions[card_of_unit[unit]].arithmetic_class ==
				    ARITHMETIC_CLASS_SM_120_NATIVE) {
				arithmetic_salt = ((arithmetic_salt != 0 ?
							    arithmetic_salt :
							    SALT_OFFSET) ^
						   pack_unit->layer_index) *
						  SALT_PRIME;
			}
		}
		if (cmp_order == CMP_ORDER_RELAXED) {
			arithmetic_salt =
				((arithmetic_salt != 0 ? arithmetic_salt :
							 SALT_OFFSET) ^
				 CMP_SALT_TAG) *
				SALT_PRIME;
			for (uint32_t unit = 0;
			     unit < pack_header_of(pack)->unit_count; unit++) {
				const struct pack_unit_record_2 *const
					pack_unit = &pack_units_of(pack)[unit];
				if (pack_unit->unit_kind ==
					    PLACEMENT_UNIT_STAGE &&
				    pack_unit->layer_index != PACK_NO_LAYER &&
				    descriptions[card_of_unit[unit]]
						    .arithmetic_class ==
					    ARITHMETIC_CLASS_SM_80) {
					arithmetic_salt =
						(arithmetic_salt ^
						 pack_unit->layer_index) *
						SALT_PRIME;
				}
				if (pack_unit->unit_kind ==
					    PLACEMENT_UNIT_ON_DEMAND &&
				    card_of_unit[unit] ==
					    PLACEMENT_IN_HOST_MEMORY) {
					arithmetic_salt =
						(arithmetic_salt ^
						 HOST_EXPERT_SALT_TAG ^ unit) *
						SALT_PRIME;
				}
			}
			/* Relaxed wide and decode forms have different bits; bind the prompt's chosen forms too. */
			const uint32_t chunk_settings[] = {
				wide_rows,	full_steps,
				prefill_cone,	chunks_in_flight,
				prefill_margin, prefill_fast_plan
			};
			for (uint32_t index = 0;
			     index <
			     sizeof(chunk_settings) / sizeof(chunk_settings[0]);
			     index++) {
				arithmetic_salt = (arithmetic_salt ^
						   chunk_settings[index]) *
						  SALT_PRIME;
			}
			/* --prefill_ladder_1024 (plans other than the exact one take other chunks); unset, the
			 * salt is as before */
			if (prefill_prefix_stable != 0)
				arithmetic_salt =
					(arithmetic_salt ^ PREFIX_STABLE_SALT) *
					SALT_PRIME;
			if (prefill_ladder_1024 != 0)
				arithmetic_salt =
					(arithmetic_salt ^ CHUNK_LADDER_TOP) *
					SALT_PRIME;
			if (prefill_ladder_1536 != 0)
				arithmetic_salt =
					(arithmetic_salt ^ LADDER_1536_ROWS) *
					SALT_PRIME;
			if (prefill_ladder_fine != 0)
				arithmetic_salt =
					(arithmetic_salt ^ LADDER_FINE_COUNT) *
					SALT_PRIME;
		}
		struct engine_description_record engine_description = {
			.model_name = pack_header_of(pack)->model_name,
			.front_end_tables = front_end_files[TABLES_FILE],
			.front_end_table_byte_count =
				front_end_byte_counts[TABLES_FILE],
			.session_slot_count = limits.session_capacity,
			.context_token_limit = context_tokens,
			.vocabulary_size = model_description.vocabulary_size,
			.end_token_id = model_description.end_token_id,
			.ignore_eos = ignore_eos,
			.front_steps = prefill_cone,
			.wide_rows = wide_rows,
			.chunks_in_flight = chunks_in_flight,
			.prefill_margin = prefill_margin,
			.prefill_fast_plan = prefill_fast_plan,
			.prefill_prefix_stable = prefill_prefix_stable,
			.prefill_order = prefill_order,
			.prefill_order_wait =
				prefill_order >= PREFILL_ORDER_SHORTEST ?
					prefill_order_wait :
					0,
			.prefill_decode_every = prefill_decode_every,
			.prefill_chunk_rows = prefill_chunk_rows,
			.prefill_decode_rows = prefill_decode_rows,
			.prefill_decode_rounds = prefill_decode_rounds,
			.prefill_decode_waiters = prefill_decode_waiters,
			.prefill_cone_skip = prefill_cone_skip,
			.prefill_tail_rows = prefill_tail_rows,
			.speculative_depth =
				speculative == SPECULATIVE_OFF ? 0 :
				draft_positions < step_rows ? draft_positions :
							      step_rows - 1u,
			.lookup_ngram_most = lookup_ngram_max,
			.lookup_ngram_least = lookup_ngram_min,
			.snapshot_identity = arithmetic_salt,
			.image_math_identity =
				model_images_are_enabled(plan) ?
					descriptions[0].arithmetic_class :
					0,
			.hybrid_lookup_min = speculative == SPECULATIVE_HYBRID ?
						     hybrid_lookup_min :
						     0,
			.draft_trace = draft_trace,
			.draft_policy = &draft_policy
		};
		memcpy(engine_description.batch_step_us, batch_step_us,
		       sizeof(batch_step_us));
		engine_description.prefill_first = prefill_first;
		engine_description.prefill_prompt_chunks =
			prefill_prompt_chunks;
		engine_description.draft_learned_acceptance =
			draft_learned_acceptance;
		engine_description.draft_session_acceptance =
			draft_session_acceptance;
		engine_description.draft_start_economic = draft_start_economic;
		engine_description.batch_plan_ties = batch_plan_ties;
		engine_description.prefix_cold_plan_qualified =
			prefix_cold_plan_is_qualified() ? 1u : 0u;
		failed = is_generating &&
			 (engine_failed_to_start(&engine_description) ||
			  stats_metadata_bound(card_of_unit) != STATUS_OK ||
			  (is_serving	? requests_failed(tokenizer) :
			   is_capturing ? capture_failed(argument_count,
							 arguments) :
					  prompt_failed_to_generate(
						  token_ids, token_count)));
		failed = failed || (is_generating && !is_serving &&
				    dump_draft_ring != NULL &&
				    draft_ring_dump_failed(dump_draft_ring));
		failed = failed || (is_generating && !is_serving &&
				    getenv("DIAG_STATE_DIGESTS") != NULL &&
				    state_digests_failed());
		failed = ((fflush(stdout) != 0 || ferror(stdout) != 0) &&
			  step_failed("write the report",
				      STATUS_INPUT_OUTPUT_ERROR)) ||
			 failed;
	}
	engine_destroy(
		engine); /* the engine, then what it stepped, then the cards */
	step_driver_destroy(driver);
	ds41f_original_head_banks_destroy(head_program_banks);
	for (uint32_t index = 0;
	     instance.operations != NULL && index < program_count; index++) {
		instance.operations->program_destroy(programs[index]);
		instance.operations->program_destroy(image_programs[index]);
	}
	engram_close(engram);
	if (engram_steps != NULL)
		engram_steps_report();
	if (engram_steps_live(engram_steps) != 0)
		(void)fprintf(
			stderr,
			"engram step io: %u step records kept at shutdown: a read did not return in its limit, "
			"so their staging stays allocated\n",
			engram_steps_live(engram_steps));
	engram_steps_close(
		engram_steps); /* every driver's step records are gone with the driver, but a kept one */
	engram_disk_close(engram_disk);
	engram_hot_close(engram_hot); /* after the store that served it */
	storage_direct.reader_close(engram_source_reader);
	server_close(server);
	uint32_t head_retained_cards = 0;
	if (original_head_offload != 0 && plan != NULL &&
	    plan->original_head.connected) {
		failed = step_failed("drain original head peer group",
				     ds41f_original_head_drain(
					     &backend_cuda_value, cards,
					     CARD_WAIT_LIMIT_NANOSECONDS,
					     &head_retained_cards)) ||
			 failed;
	}
	if (instance.operations != NULL && tensor_model == NULL) {
		instance.operations->plan_destroy(plan);
	}
	const uint32_t retained_cards = tensor_model != NULL ?
						tensor_cards_drain() :
						head_retained_cards;
	if (retained_cards != 0) {
		(void)fprintf(
			stderr,
			"%s peer arenas retained after an unproven group drain: mask=0x%" PRIx32
			"\n",
			tensor_model != NULL ? "TP2" : "original head",
			retained_cards);
		failed = true;
	}
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
		if ((retained_cards >> card & 1u) == 0)
			backend_cuda_value.card_close(
				cards[card], CARD_WAIT_LIMIT_NANOSECONDS);
	}
	if (retained_cards == 0)
		ds41f_tp2_model_destroy(tensor_model);
	if (layout !=
	    NULL) { /* behind the cards; no arena was mapped without a layout that sizes it */
		load_host_arena_release(result.host_arena,
					layout->host_byte_count);
		load_layout_destroy(layout);
	}
	free(card_of_unit);
	pack_close(pack);
	storage_operations->reader_close(reader);
	tokenizer_close(tokenizer);
	for (uint32_t file = 0; file < FRONT_END_FILES; file++) {
		free(front_end_files[file]);
	}
	free(prompt);
	free(token_ids);
#if TIMING_HOOKS_ENABLED
	hooks_report(stderr);
#endif
	return failed && !stops_after_fit ? PROGRAM_FAILED : PROGRAM_SUCCEEDED;
}
