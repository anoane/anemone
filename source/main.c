/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* main.c: load/generate/serve; main prints exact CLI usage.
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
#include <time.h>
#include <unistd.h>

#include "anemone/chat_template.h"
#include "anemone/hooks.h"
#include "anemone/loader.h"
#include "anemone/model_layout.h"
#include "anemone/official_reference.h"
#include "anemone/pack_digest.h"
#include "anemone/placement.h"
#include "anemone/replicas.h"
#include "anemone/server.h"
#include "anemone/stats_client.h"
#include "anemone/startup_policy.h"
#include "anemone/storage_exl3.h"
#include "anemone/exl3_tc_records.h"
#include "serving_cursor.h"
#include "startup_axes.h"
#include "anemone/step.h"
#include "anemone/step_images.h"
#include "anemone/tokenizer.h"
#include "draft_lookup.h"
#include "front_end_pack.h"
#include "json.h"
#include "model_v41.h"
#include "model_v41_tp2_model.h"
#include "model_v41_tp2_peer.h"

#define PROGRAM_ATTRIBUTION_NOTICE                                                  \
	"Anemone v0.0.2-alpha\nCopyright (C) 2026 Fabiano Anemone <fabiano.anemone@gmail.com>\n" \
	"SPDX-License-Identifier: AGPL-3.0-or-later\n"
#define NANOSECONDS_PER_SECOND 1e9
#define BYTES_PER_GIB 1073741824.0
#define BYTES_PER_MIB 1048576ull
#define DECIMAL_BASE 10
#define ARGUMENTS_BEFORE_THE_OPTIONS 2 /* the program's name and the command */
#define MOST_CARD_MEMORY_KEPT_FREE_MIB \
	1048576 /* a TiB: above any card, and its bytes fit in 64 bits */
#define BYTES_PER_GB 1e9
#define DEFAULT_TP_LOAD_THREADS 32u /* the TP load's workers for the cards (rh04, 4 Nov: 31.1 s against 33.9 at 16) */
#define DEFAULT_TP_LOAD_HOST_THREADS 16u /* and the host arena's threads beside them */
#define DEFAULT_TP_LOAD_READS_PER_COPY 0u
#define MAP_READS_AUTO \
	EXL3_READ_MODES /* --map_reads beyond the storage's modes: chosen by the copies found */
#define CARD_WAIT_LIMIT_NANOSECONDS \
	10000000000ull /* ten seconds: for what is in flight, and for a step */
#define FRONT_END_FILES \
	2 /* of generate: the messages, the tokenizer's tables */
#define FRACTION_OPTIONS 2 /* --temperature and --top_p */
#define TEXT_OPTIONS \
	8 /* serve: address, keys, model, sessions, stats users and draft usage; generate: the ring dump; all: layout */
#define LAYOUT_TEXT (TEXT_OPTIONS - 1u) /* the one text every command takes */
#define DUMP_TEXT (TEXT_OPTIONS - 2u) /* generate's diagnostic dump of the draft rings */
#define DRAFT_RING_DUMP_ROWS 512u
#define DIGEST_MIX_PRIME 1099511628211ull /* FNV-1a 64: mixes a layer's tensor digests in order */ /* the dump's window rows of session 0, positions 0 .. 511, a draft stage */
#define PIPE_ENDS 2 /* of a pipe: read, write */
#define STOP_SIGNALS 3 /* of serve: SIGINT, SIGTERM, SIGHUP */
#define MOST_THIN_PROGRAMS \
	3 /* four-owner capacity4: programs1/2/4; single-session retains1/max */
#define CHUNK_LADDER_TOP \
	1024u /* above it, wide_rows then 512, 256 and 64 rows (engine.c's exact plan: whole multiples of 256) */
#define CHUNK_ROWS_DIVISOR \
	4u /* wide and full programs of wide_rows, a quarter of it, ..., down to the least wide rows */
#define MOST_CHUNK_PROGRAMS 4 /* of each kind: 1,024, 256 and 64 rows; 2,048, 512, 128 and 64 */
#define MOST_BATCH_PROGRAMS 8u /* --batch_rows: BATCH programs in place of the thin ones (model_seam.h) */
#define DEFAULT_BATCH_GROUPED_ROWS 16u /* --batch_grouped_rows */
#define DEFAULT_BATCH_CORE_ONCE \
	3u /* step-wide pools, then deferred compressor slot retention */
#define DEFAULT_OTHER_BATCH_CORE_ONCE 2u
#define LEAST_MEASURED_BATCH_CORE_ONCE \
	2u /* measured menu: step-wide window and core, with optional pools */
#define DEFAULT_BATCH_DRAFT_WIDE \
	3u /* target-independent draft walk over complete verifier blocks */
#define MOST_BATCH_GROUPED_ROWS 48u
#define MOST_PROGRAMS         \
	(MOST_THIN_PROGRAMS + MOST_BATCH_PROGRAMS + \
	 2 * MOST_CHUNK_PROGRAMS + EXTRA_CHUNK_PROGRAMS) /* and wide and full ones, the knobs' */
#define DEFAULT_GENERATED 64u /* tokens */
/* --prefill_tail_rows: a full program of two of the exact plan's 64-row tail chunks, beside the ladder's (BATCH
 * programs take the thin ones' place, so MOST_PROGRAMS holds it), with a ladder of no 128-row program of its own */
#define TAIL_PROGRAM_ROWS 128u
#define TAIL_LONG_ROWS 320u /* --prefill_tail_rows=320: a full program of five tail chunks beside the 128-row one */
#define TAIL_WIDEST_ROWS 448u /* --prefill_tail_rows=448: and one of seven beside the 320-row one */
#define TAIL_LEAST_WIDE_ROWS 1024u
#define EXTRA_CHUNK_PROGRAMS 26u /* --prefill_ladder_1024's two, --prefill_ladder_1536's two, --prefill_ladder_fine's
				   six, --prefill_front_tail's three, the two long tails' and --prefill_thin_tail's six
				   and twelve past the slack */
#define FRONT_TAIL_ALL 2u /* --prefill_front_tail=2: the front region's tail takes the full region's tail programs */
#define THIN_TAIL_KINDS 2u /* --prefill_thin_tail with FRONT_TAIL_ALL: full and wide thin-tail programs */
#define THIN_TAIL_LEAST_ROWS 16u /* --prefill_thin_tail: full programs of 16 to 56 rows, 8 apart (thin steps' forms) */
#define THIN_TAIL_AUTO UINT32_MAX /* --prefill_thin_tail not given: 1 with the tail program and no BATCH (a BATCH step
				    takes the rows short of a 64-row chunk with the row of logits in one; EVIDENCE) */
#define LADDER_1536_ROWS 1536u /* --prefill_ladder_1536: six whole granules of the exact plan (engine.c) */
#define LADDER_FINE_COUNT 3u /* --prefill_ladder_fine: 1,792, 1,280 and 768 rows (7, 5 and 3 whole granules) */
#define PREFILL_CHUNK_CAPS \
	2u /* global and decoder-only caps may each need another prepared chunk size */
#define LADDER_FINE_AUTO UINT32_MAX /* --prefill_ladder_fine not given: 1 with TAIL_LONG_WIDE_ROWS or more, no BATCH */
#define TAIL_ROWS_AUTO UINT32_MAX /* --prefill_tail_rows not given: TAIL_PROGRAM_ROWS where the tail program applies,
				    TAIL_WIDEST_ROWS (and so TAIL_LONG_ROWS too) where also --wide_rows is
				    TAIL_LONG_WIDE_ROWS or more and no BATCH, TAIL_LONG_ROWS with BATCH (gsk03: with the
				    tail overlay the 640-column cards keep 341 MiB free, cold prefill +2.3%) */
#define TAIL_LONG_WIDE_ROWS 2048u /* --prefill_tail_rows=320's default (ri03), 448's (gsk01: identical, the 2K cell
				     1,696 -> 1,722 tok/s, cold prefill 3,798 -> 3,818) with 2,048-row chunks */
#define FRONT_TAIL_AUTO UINT32_MAX /* --prefill_front_tail not given: 1 where the tail program applies, FRONT_TAIL_ALL
				      with the 448-row tail (THIN; rk04k: identical, the 30K prompt 8.214 -> 8.190 s) */
#define DEAD_ROWS_TRIMMED 2u /* --prefill_dead_halves: the sums' copies of the rows the cone reads alone too */
#define DEFAULT_CHUNKS_IN_FLIGHT \
	1u /* 1: engine.c's exact plan (the 1,024-row two-in-flight reference's bits in the largest chunks;
	      rc02ab 30K prompt 11.859 -> 9.430 s, Fraser 2.431 -> 1.857 s, identical); else so many chunks at least */
#define DEFAULT_PREFILL_MARGIN \
	127u /* the exact plan's full rows beyond a prompt's cone (engine.c; rc02ab: the same bits, -1.0 s at 30K) */
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
enum command_kind {
	TAKEN_BY_LOAD = 1,
	TAKEN_BY_GENERATE = 2,
	TAKEN_BY_SERVE = 4,
	TAKEN_BY_ALL = 7
};

static const char *const *copy_paths; /* the copies of the pack, as typed */
static const char *front_end_paths
	[FRONT_END_FILES]; /* by enum front_end_file, as typed; NULL: none */
static uint32_t copy_count;
static bool is_generating, is_serving; /* generate or serve; serve */
static uint32_t card_order
	[BACKEND_MAXIMUM_CARDS]; /* backend cards, in the order the stages fill them */
static uint32_t
	card_count; /* 0 until the cards are counted: all, as enumerated */
static uint32_t reader_threads_per_copy =
	LOADER_DEFAULT_READER_THREADS_PER_COPY;
/* the TP placements' load: 0 the serial loads (the reference), N the host arena beside the cards, each on N threads */
static uint32_t tp_load_threads = DEFAULT_TP_LOAD_THREADS;
/* with tp_load_threads: reads in flight a copy at most (0: no limit) */
static uint32_t tp_load_reads_per_copy = DEFAULT_TP_LOAD_READS_PER_COPY;
/* with tp_load_threads: the host arena's threads (0: as many as the cards' load) */
static uint32_t tp_load_host_threads = DEFAULT_TP_LOAD_HOST_THREADS;
/* with tp_load_threads and copies: 1 (default) host-arena chunks from copy 0, card extents from the last copy
 * (tp_load.h); 0: each read from the copy soonest to finish it */
static uint32_t tp_load_copy_split = 1;
/* 1 (default): a single source gains its receipted replicas on other drives as copies (replicas.h); 0: it alone */
static uint32_t turbo_load = 1;
/* an EXL3 map's reads (enum exl3_read_mode, storage_exl3.h): 0 buffered, 1 buffered then dropped from the
 * page cache, 2 direct where aligned (rh01, 3 Nov: TP4 loads 36.7-38.4 / 29.8-31.5 / 32.3 s), 3 direct where
 * aligned and dropped for the rest; MAP_READS_AUTO (default): 3 when replicas were found (rh06 / rh08, 4 Nov, with
 * the copy split: 29.7 s against 31.0), else 1 (one copy: rh08t 62.5 s with 3 against 45.9 with 1) */
static uint32_t map_reads = MAP_READS_AUTO;
static uint32_t card_memory_kept_free_mib;
#define ORIGINAL_PP5_MULTI_PIPELINE_STEPS 5u
static uint32_t session_slots = 1, step_graphs, pipeline_steps = 1,
		prefill_cone;
/* 0: serial companion flights; 1: disjoint CMP-prefix and native-PRO-tail workers. */
static uint32_t tp_pipeline_overlap;
/* 0 (the default): a PRO computes with its own tensor cores (class 120 native, backend_seam.h), its outputs then
 * depend on the layers it holds; 1: it emulates class 80's sums, the same bits on every card, at 2-4 times its time */
static uint32_t pro_emulation;
static enum cmp_order_code cmp_order = CMP_ORDER_REFERENCE;
static enum numerics_code numerics = NUMERICS_EXACT;
static uint32_t tp_code_wire;
static uint32_t tp_attention_owner;
static uint32_t tp_head_shards;
static uint32_t tp_exact_dense = 1; /* tp4_exl3 exact recipe: attention output shards (xdense01: same bits) */
static uint32_t tp_draft_shards;
static uint32_t tp_draft_layers; /* tp4_exl3 BF16 recipe, thin: the draft's layers tensor-parallel (GNmac's DSpark) */
static uint32_t tp_head_stream;
static uint32_t exl3_fast_products = 1; /* kernel_exl3_wide's products (identical words); 0 the retained chain */
static uint32_t exl3_wide_entry = 1; /* chunked steps run an EXL3 layer's experts once over the chunk (fast01/02) */
static uint32_t exl3_tc_moe; /* tp4_exl3 BF16 recipe: GNmac's tensor-core EXL3 MoE (exl3_tc.h; GNmac numerics) */
static uint32_t
	exl3_tc_product_blocks; /* decode product grid override; 0 keeps the measured 148-block grid */
#define TC_PRODUCT_BLOCKS_MOST \
	592u /* eight product blocks per CMP multiprocessor */
static uint32_t exl3_expert_rotation; /* with exl3_tc_moe: per-expert chunk rotation 1 or 2 (exl3_tc_records.h) */
static uint32_t tp_head_mma; /* tp4_exl3 BF16 recipe: the head products on the tensor cores (kernel_head_mma) */
static uint32_t tp_trunk_mma; /* tp4_exl3 BF16 recipe: thin BF16-row trunk products on the tensor cores (GNmac) */
static uint32_t tp_trunk_mma_fused; /* trunk_mma's _64 form adds its own splits (same bits; relbench03: 1-5 us slower) */
static uint32_t tp_trunk_mma_thin_splits; /* trunk_mma's _64 form splits as the thin form: the thin steps' bits */
static uint32_t tp_router_mma; /* tp4_exl3 BF16 recipe: a BATCH step's router by split-K tensor-core MMAs (GNmac's) */
static uint32_t tp_gather_balance; /* tp4_exl3: a chunk's index scans and router rows on the narrower expert ranks */
static uint32_t tp_pool_chunk = 1; /* tp4_exl3: a chunk's pool entries and keys ahead of its slabs' scans (same bits;
				       rc02ab 30K prompt 9.828 -> 9.430 s); 0: in the slabs' order */
static uint32_t tp_pool_gather = 1; /* tp4_exl3: a pooled source layer's compressor products a quarter of a half's
				       rows a rank, gathered (the same bits; ri01); 0: every row on every rank */
static uint32_t tp_router_first_whole; /* tp4_exl3: the first half's router over its rows on every rank, no gather */
static uint32_t exl3_tcp_two_blocks = 1; /* the prefill TC MoE's GEMMs at two blocks a multiprocessor (the same bits;
					    ri03: a 2,048-row call -15%, Fraser first token -17 ms); 0: one block */
static uint32_t tp_queries_piece = 1; /* tp4_exl3: a wide program's queries over each piece with wq_b (the same bits;
					 ri05 30K -77 ms); 0: a span at a time from the heads copied into the full rows */
static uint32_t tp_index_span = 1; /* tp4_exl3: a chunk's index scans and selections a span of rows a launch (same
				      bits; ri05 30K -51 ms); 0: a slab of 8 rows a launch */
static uint32_t tp_core_compact_halves = 1; /* tp4_exl3: a split half's core into wo_a's compact rows (same bits;
					       ri06: -9 / -11 ms); 0: the full rows and a copy */
static uint32_t dense_skinny_bf16; /* tp4_exl3 class 80: BF16 skinny dense products' outputs a block above 48 rows */
static uint32_t dense_skinny_f32; /* tp4_exl3 class 80: float32-weight skinny products' outputs a block above 48 rows */
#define DENSE_SKINNY_MOST_OUTPUTS 8u /* --dense_skinny_bf16: the widest stamped wide form (dense_linear.h) */
/* tp4_exl3 class 80: the wide dense products of 64 rows or more staged, 16 outputs a block (the same bits; rk05:
 * IDENTICAL, the 2K cell prompt 1.175 -> 1.150 s, Fraser 1.393 -> 1.385, 30K equal) */
static uint32_t dense_staged = DENSE_STAGED_OUTPUTS;
/* Original TP4 CMP prefill: opt in to the existing guarded BF16 staged dense form. */
static uint32_t original_dense_staged;
static uint32_t
	original_wide_experts; /* compatible TP4_PRO recipes default to one wide expert pass */
static uint32_t tp_parallel_launch; /* TP: each card's graphs of a step's group launched from a thread of its own */
static uint32_t tp_graph_parts = 1; /* TP: a card's step graph behind the prefill cone in this many parts */
static uint32_t tp_ordered_wide; /* TP EXL3: wide attention trunk products by the ordered wide form (same bits) */
static uint32_t tp_graph_branches = 1; /* TP EXL3 TC MoE: the shared expert a graph branch beside the routed (same bits,
					  engab18 -0.34 ms a C1 round); 2: BATCH steps too; 0: one chain */
static uint32_t tp_piece_rings; /* TP BF16 recipe: wide steps' BF16 rings over 64-row pieces, not 8-row slabs */
static uint32_t tp_trunk_shape_slices = 1; /* tp4_exl3: thin trunk products at the slice count measured for their
					       shard shape (trunk01; ordered chains: the same bits at any count); 0: 4 */
static uint32_t tp_bf16_slab_rings; /* 1: BF16 recipe chunked attention summed a slab at a time by the ring */
/* BF16 recipe chunked steps' sums: copy-engine owner sums and owner ring sums with the layer's halves overlapped (the
 * push's and the slab rings' bits; pf19: 30K-token prefill 19.08 -> 14.80 s) */
static uint32_t tp_wide_exchange = V41_DATA_TP_EXCHANGE_CE_RINGS;
static uint32_t tp_split_overlap =
	V41_DATA_SPLIT_JOINT_DOWN; /* ri01: identical (6: rb02 12.370 -> 12.057 s) */
static uint32_t tp_split_least_rows; /* split_overlap: chunks of fewer rows run whole (0: halves of more than 64 rows) */
static uint32_t tp_launch_trims = 1; /* 0: the launches as before 3 Nov (ring input copies, wo_b's quantization) */
static uint32_t tp_full_split_overlap; /* the full programs' split_overlap mode (0: --tp_split_overlap's) */
static uint32_t tp_chunk_core = 1; /* chunked steps' attention core over the chunk (same bits; pf16: 18.93 -> 17.80 s) */
static uint32_t
	original_chunk_core; /* default 0: exact original attention retains its eight-row cores */
static uint32_t tp_replicated_embedding = 1; /* tp4_exl3: every rank embeds its rows (same bits); 0: card 0 hands off */
/* BATCH (tp4_v3's host half, tp4_v3_claude's programs): --batch_rows=R[,R...] (9 to 64 each) prepares BATCH programs of
 * those rows in place of the thin ones; with the queue's depth 1 (step_pipeline_start(driver, 1)); a row count
 * may repeat with --batch_verifiers (another count of draft walks: e.g. BATCH-8 of one for C1 and of two for C2) */
static uint32_t batch_rows[MOST_BATCH_PROGRAMS], batch_draft_rows, batch_wide_head;
static uint32_t
	original_batch; /* default 0: original TP4_PRO keeps THIN; 1: fixed native-tail BATCH48 */
static uint32_t batch_context_slabs; /* BATCH: a step's draft context in slabs of 8 rows (the thin steps' forms) */
/* BATCH: a BATCH program's rows the thin steps' bits (trunk_mma's _64 form at the thin splits, the target head at
 * the thin split, the TC MoE's short forms at every row count): a session's tokens independent of the batch */
static uint32_t batch_thin_bits = 1;
/* BATCH, with batch_thin_bits: a step of 9 to this many rows (at most 48) runs its trunk products by the grouped forms
 * (the 1-8-row forms over its row groups of 8: the thin bits in one launch, no partials); 0: trunk_mma's _64 form */
static uint32_t batch_grouped_rows = DEFAULT_BATCH_GROUPED_ROWS;
static uint32_t batch_tc_paired; /* with batch_thin_bits: the TC MoE's short forms paired from so many rows */
static uint32_t batch_draft_slots_head; /* with --batch_wide_head: one verifier's draft head over 8 slots a position */
static uint32_t batch_row_sets; /* with batch_thin_bits: trunk products in row sets above --batch_grouped_rows */
/* A BATCH program's core, window and pools once over its slabs; retain compressor slots after the pools read them.
 * Level 3 preserves both reference widths and mixed-user outputs; level 2 remains selectable. */
static uint32_t batch_core_once = DEFAULT_BATCH_CORE_ONCE;
static uint32_t tp_front_branch; /* thin coefficient fronts as side branches: 1 attention, 2 both (same bits) */
static uint32_t tp_core_heads; /* the attention core's most heads a block, 0 the measured rule (same bits) */
static uint32_t
	tp_decode_round_once; /* BF16 decode partials summed in FP32 at their owners, rounded once */
static uint32_t draft_bf16_rows; /* the draft layers' attention products of BF16 rows (drafts only) */
static uint32_t batch_draft_wide =
	DEFAULT_BATCH_DRAFT_WIDE; /* 1 sums; 2 products; 3 cores over whole blocks */
static uint32_t draft_bf16_operands; /* class 80: the draft's quantized products of its rows' BF16 values (drafts) */
#define MOST_CORE_HEADS 64u
#define CORE_HEADS_STEP 8u /* the attention core's tile of heads (attention_core_records.h) */
#define MOST_BATCH_ROW_SETS 24u
/* the BF16 ring sums with column owners: a row's order of additions independent of its place in the step, so a
 * T=0 stream's bits do not depend on the batch or the schedule (default since 3 Nov, owner: new references 30K
 * 41eeb9c16e10, Fraser f49e8badff47); 0: row-quarter owners, the old reference order */
static uint32_t tp_ring_column_owners = 1;
/* BATCH: a step's microseconds on each BATCH program, fewest rows first: each step verifies the drafts that buy the
 * most expected tokens a microsecond (engine.c's step plan); none: the draft policy's depths */
static uint32_t batch_step_us[SERVING_BATCH_COSTS];
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
/* BATCH: each --batch_rows program's drafting sessions (its draft walks), apart from its rows: a step then takes the
 * fewest-row program holding its rows and its drafting sessions (--batch_step_us can then trim rows); none: rows / 6 */
static uint32_t batch_verifiers[MOST_BATCH_PROGRAMS];
static uint32_t exl3_expert_parts = EXL3_EXPERT_REFERENCE_PARTS;
static enum execution_placement_code execution_placement = EXECUTION_PP5;
static bool numerics_was_given;
static uint32_t prefix_reuse_exact;
static uint32_t prefix_reuse = 1u; /* serve: ordinary prefix reuse unless explicitly disabled */
static uint32_t cache_scope = SERVER_CACHE_SESSION;
static uint32_t
	session_fork; /* experimental independent same-owner clones; qualification required */
#define DEFAULT_SESSION_GRACE_SECONDS 30u
static uint32_t session_grace_seconds = DEFAULT_SESSION_GRACE_SECONDS;
static uint32_t session_save_tokens = SERVER_DEFAULT_SAVE_TOKENS;
static uint32_t session_load_tokens = SERVER_DEFAULT_LOAD_TOKENS;
static uint32_t session_idle_save_seconds = SERVER_DEFAULT_IDLE_SAVE_SECONDS;
static uint32_t full_steps =
	1; /* with wide rows: the rows behind the front in full chunks (model_seam.h's FULL) */
static uint32_t chunks_in_flight = DEFAULT_CHUNKS_IN_FLIGHT;
static uint32_t prefill_margin = DEFAULT_PREFILL_MARGIN; /* full rows beyond a prompt's cone (engine.c) */
static uint32_t prefill_fast_plan; /* chunks_in_flight 1: the largest program that fits (other bits than exact) */
static uint32_t prefill_prefix_stable; /* fixed-capacity prefill; default off */
#define PREFIX_STABLE_SMALL_ROWS 1024u
#define PREFIX_STABLE_LARGE_ROWS 2048u
/* PRFXSTBL: distinct numerical identity; the old plan 0 salt is unchanged. */
#define PREFIX_STABLE_SALT 0x505246585354424Cull
/* The measured bounded BATCH recipe selects shortest first; other recipes retain turns. */
#define PREFILL_ORDER_AUTO UINT32_MAX
static uint32_t prefill_order =
	PREFILL_ORDER_AUTO; /* serve: prompts one at a time (engine.c prompt_head) */
#define PREFILL_ORDER_DEFAULT_WAIT 16u /* --prefill_order_wait: with --prefill_order=2, the steps before the oldest's */
#define PREFILL_ORDER_MOST_WAIT 1024u
static uint32_t prefill_order_wait = PREFILL_ORDER_DEFAULT_WAIT;
#define PREFILL_DECODE_MOST_EVERY 64u /* --prefill_decode_every: the most chunk steps between decoding steps */
static uint32_t prefill_decode_every; /* with prefill_order: a decoding step after so many head chunks (engine.c) */
#define PREFILL_SCHEDULING_AUTO UINT32_MAX
#define PREFILL_SCHEDULING_DEFAULT_WIDE_ROWS 1024u
#define PREFILL_DECODE_DEFAULT_ROWS 1024u
#define PREFILL_DECODE_DEFAULT_ROUNDS 1u
#define PREFILL_DECODE_DEFAULT_WAITERS 0u
/* An exact whole-granule cap even without another stream decoding. */
static uint32_t prefill_chunk_rows;
static uint32_t prefill_decode_rows = PREFILL_SCHEDULING_AUTO;
static uint32_t prefill_decode_rounds = PREFILL_SCHEDULING_AUTO;
static uint32_t prefill_decode_waiters = PREFILL_SCHEDULING_AUTO;
static uint32_t engram_shared; /* TP host gathers: identical table masks share one immutable staging block */
static uint32_t prefill_cone_skip = 1; /* full steps' rows past their cone: no core or routed experts (engine.c) */
static uint32_t prefill_tail_rows = TAIL_ROWS_AUTO; /* the 64-row tail chunks two a step, their forms (engine.c) */
static uint32_t prefill_dead_halves = DEAD_ROWS_TRIMMED; /* full programs leave the sums of a decoder layer's half
							     the cone never reads; 2: and every sum's copies the rows
							     before a half's first one it reads (ri01); 0: none */
static uint32_t prefill_ladder_1024 = 1; /* a ladder above CHUNK_LADDER_TOP takes a program of that many rows too
					    (ri01: with front_tail identical, 30K -82 ms) */
static uint32_t prefill_ladder_fine = LADDER_FINE_AUTO; /* the ladder's every whole-granule count to wide_rows (with
							   1,024 and 1,536; ri07 / gsi10: identical, cold prefill
							   +0.8%, ~200 MiB a card of programs) */
static uint32_t prefill_ladder_1536 = 1; /* a ladder above LADDER_1536_ROWS takes a program of that many rows too
					    (ri06: identical, a 1,905-token prompt -84 ms) */
static uint32_t prefill_front_tail = FRONT_TAIL_AUTO; /* with the tail program: a wide one for the front region's tail
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
static uint32_t
	prefill_first; /* serve: any session's wide or full chunk before other steps (engine.c) */
static uint32_t prefill_prompt_chunks; /* opt-in after the measured C8 cost */
static uint32_t draft_learned_acceptance =
	1; /* the policy's acceptance table follows outcomes (engine.c) */
static uint32_t
	draft_session_acceptance; /* a session's acceptances scaled by its own outcomes (engine.c) */
/* a command's first DSpark round at the policy's depth (engine.c), not a probe of one draft: other steps, so other
 * bits while the BF16 ring's owners follow a row's place in the step (with --tp_ring_column_owners=1, the default:
 * the same); 0: the probe round */
static uint32_t draft_start_economic = 1;
static uint32_t batch_plan_ties; /* the BATCH step plan cuts only between drafts of unequal reach (drafts only) */
static uint32_t
	wide_rows; /* 0: no wide steps; else a wide program of so many rows (model_seam.h) */
static uint32_t step_rows = 1,
		generated_token_limit; /* 0 until given: the command's default */
/* speculative decoding (front_end_seam.h): off by default; a round verifies at most draft_positions proposals, and at
 * most --rows less 1, of the lookup drafter's n-grams of lookup_ngram_max down to lookup_ngram_min tokens */
static uint32_t draft_block = MODEL_DRAFT_BLOCK;
static uint32_t speculative, draft_positions = SERVING_DRAFT_MOST,
			     lookup_ngram_max = DEFAULT_LOOKUP_NGRAM_MOST,
			     lookup_ngram_min = DEFAULT_LOOKUP_NGRAM_LEAST;
static bool is_draft_given;
static uint32_t hybrid_lookup_min = DRAFT_DEFAULT_LOOKUP_MIN;
static uint32_t draft_trace; /* diagnostics: each device round's anchor and proposals to stderr */
static struct draft_policy_record draft_policy;
static const char *draft_calibration_path;
static const char *prompt_token_ids_path;
static uint32_t ignore_eos;
static uint32_t context_tokens =
	MODEL_CONTEXT_WITHOUT_CANDIDATES; /* the default: a modest state */
static uint32_t seed, reasoning_effort,
	linear_profile = LINEAR_PROFILE_PRECISION,
	linear_order =
		LINEAR_ORDER_REFERENCE; /* the exact profile: the reference's orders (core, router) */
static uint32_t
	official_hc_rows; /* 0 retains existing arithmetic; else pinned HC reference call shape. */
static uint32_t
	official_reference; /* 0: existing forms; 1: original PP THIN serial M1 reference recipe. */
static bool is_seed_given; /* serve: else a request without a seed draws one */
static float temperature = 0.0f, top_p = 1.0f;
static double step_started_seconds; /* when the step that runs now began */
static const char *
	bind_address = SERVER_DEFAULT_ADDRESS,
       *key_file, *model_alias, *session_directory,
       *layout_text; /* placement.h's layout text in place of the fill; NULL: the fill */
static uint32_t port = SERVER_DEFAULT_PORT;
static uint32_t queue_depth = SERVER_DEFAULT_QUEUE_DEPTH;
static uint32_t user_queue_depth; /* zero: half the global queue, at least one */
static const char *draft_usage = "compact";
static const char *dump_draft_ring; /* generate diagnostics: the draft rings' first rows to this file at the end */
static uint32_t live_stats_interval_ms;
static const char *live_stats_format = "text";
static const char *stats_users;
static char stats_card_order[LIVE_STATS_CARD_ORDER_BYTES];
static char stats_placement[LIVE_STATS_PLACEMENT_BYTES];
static int stop_pipe[PIPE_ENDS] = {
	-1, -1
}; /* serve's: a signal writes, the server polls, serve closes */
/* What a run opens, closed by main on every path. */
static struct storage_reader *reader;
static const struct storage_operations_record *storage_operations =
	&storage_direct;
static struct pack_record *pack;
/* turbo load: the copies once the source's receipted replicas are added (replicas.h) */
static char replica_paths[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
static const char *turbo_copy_paths[STORAGE_MAXIMUM_COPIES];
static struct load_layout *layout;
static struct device_card *cards[BACKEND_MAXIMUM_CARDS];
static struct card_description_record descriptions[BACKEND_MAXIMUM_CARDS];
static struct load_result result;
static const struct model_operations *model;
static struct model_plan_record *plan;
static struct v41_tp2_model *tensor_model;
static uint32_t
	tp_wire_bf16; /* TP-only explicit wire cast; PP retains its original numerical contract. */
static struct step_companion_program tensor_companions[MOST_PROGRAMS];
static struct execution_program *programs[MOST_PROGRAMS];
static struct execution_program *image_programs[MOST_PROGRAMS];
static uint32_t program_count;
static struct engram_record *engram;
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
		result_status status = model->arena_byte_count(
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
						MODEL_DRAFT_FIRST_LAYER;
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
static bool take_option(const char *argument, const char *name, uint32_t least,
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
		for (uint32_t earlier_index = 0; earlier_index < number_count && !repeats;
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
static bool take_fraction(const char *argument, const char *name, float least,
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
		*placement_was_given =
			execution_placement_from_text(
				argument + sizeof(placement_option) - 1u,
				&execution_placement) == STATUS_OK;
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
	FILE *const file = fopen(draft_calibration_path, "r");
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

/* Exact cold-boundary eligibility is an actual recipe capability, independent of the selected same-session policy. */
static bool prefix_cold_plan_is_qualified(void)
{
	return execution_placement == EXECUTION_TP4_EXL3 &&
	       tp_ring_column_owners != 0 && batch_thin_bits != 0 &&
	       chunks_in_flight == 1u && prefill_fast_plan == 0 &&
	       prefill_chunk_rows == 0 && prefill_decode_rows == 0 &&
	       (session_slots <= 1u || prefill_first != 0 ||
		prefill_order != 0);
}

/* False for a command line the usage does not describe. */
static bool parse_arguments(int argument_count, char **arguments)
{
	draft_policy_default(&draft_policy);
	/* The local fixed objective is resolved after parsing the configured concurrency. */
	draft_policy.depth_policy = DRAFT_DEPTH_FIXED;
	draft_policy.base_us = DEFAULT_CODE_DRAFT_BASE_US;
	draft_policy.tail_us = 0;
	draft_policy.row_us = DEFAULT_CODE_DRAFT_ROW_US;
	struct number_option {
		const char *name;
		uint32_t *numbers;
		uint32_t least, most, most_numbers, commands;
	};
	static const struct number_option options[] = {
		{ "--card_order=", card_order, 0, BACKEND_MAXIMUM_CARDS - 1,
		  BACKEND_MAXIMUM_CARDS, TAKEN_BY_ALL },
		{ "--live_stats=", &live_stats_interval_ms, 0,
		  LIVE_STATS_MOST_INTERVAL_MS, 1, TAKEN_BY_SERVE },
		{ "--reader_threads_per_copy=", &reader_threads_per_copy, 1,
		  LOADER_MAXIMUM_READER_THREADS_PER_COPY, 1, TAKEN_BY_ALL },
		{ "--tp_load_threads=", &tp_load_threads, 0,
		  TP_LOAD_MAXIMUM_WORKERS, 1, TAKEN_BY_ALL },
		{ "--turbo_load=", &turbo_load, 0, 1, 1, TAKEN_BY_ALL },
		{ "--map_reads=", &map_reads, 0, MAP_READS_AUTO, 1,
		  TAKEN_BY_ALL },
		{ "--tp_load_host_threads=", &tp_load_host_threads, 0,
		  TP_LOAD_MAXIMUM_WORKERS, 1, TAKEN_BY_ALL },
		{ "--tp_load_copy_split=", &tp_load_copy_split, 0, 1, 1,
		  TAKEN_BY_ALL },
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
		  EXL3_TC_ROTATION_THREE_WAY, 1, TAKEN_BY_ALL },
		{ "--tp_head_mma=", &tp_head_mma, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_trunk_mma=", &tp_trunk_mma, 0, 1, 1, TAKEN_BY_ALL },
		{ "--tp_trunk_mma_fused=", &tp_trunk_mma_fused, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_trunk_mma_thin_splits=", &tp_trunk_mma_thin_splits, 0,
		  1, 1, TAKEN_BY_ALL },
		{ "--tp_router_mma=", &tp_router_mma, 0, 2, 1, TAKEN_BY_ALL },
		{ "--tp_gather_balance=", &tp_gather_balance, 0,
		  V41_DATA_GATHER_BALANCE_WEIGHTED, 1, TAKEN_BY_ALL },
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
		{ "--original_wide_experts=", &original_wide_experts, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_parallel_launch=", &tp_parallel_launch, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_graph_parts=", &tp_graph_parts, 1, TP_MOST_GRAPH_PARTS,
		  1, TAKEN_BY_ALL },
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
		  V41_DATA_TP_EXCHANGE_CE_RINGS, 1, TAKEN_BY_ALL },
		{ "--tp_split_overlap=", &tp_split_overlap, 0,
		  V41_DATA_SPLIT_JOINT_DOWN, 1, TAKEN_BY_ALL },
		{ "--tp_split_least_rows=", &tp_split_least_rows, 0,
		  V41_DATA_TP_SPLIT_MOST_ROWS, 1, TAKEN_BY_ALL },
		{ "--tp_launch_trims=", &tp_launch_trims, 0, 1, 1,
		  TAKEN_BY_ALL },
		{ "--tp_full_split_overlap=", &tp_full_split_overlap, 0,
		  V41_DATA_SPLIT_JOINT_DOWN, 1, TAKEN_BY_ALL },
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
		  SERVING_SESSION_CAPACITY, MOST_BATCH_PROGRAMS,
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
		  MODEL_WIDE_MOST_ROWS, 1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--prefill_decode_rows=", &prefill_decode_rows, 0,
		  MODEL_WIDE_MOST_ROWS, 1, TAKEN_BY_SERVE },
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
		{ "--session_fork=", &session_fork, 0, 1, 1, TAKEN_BY_SERVE },
		{ "--session_grace_seconds=", &session_grace_seconds, 0,
		  SERVER_MOST_GRACE_SECONDS, 1, TAKEN_BY_SERVE },
		{ "--session_save_tokens=", &session_save_tokens, 1,
		  MODEL_CONTEXT_MOST, 1, TAKEN_BY_SERVE },
		{ "--session_load_tokens=", &session_load_tokens, 1,
		  MODEL_CONTEXT_MOST, 1, TAKEN_BY_SERVE },
		{ "--session_idle_save_seconds=", &session_idle_save_seconds, 1,
		  SERVER_MOST_IDLE_SAVE_SECONDS, 1, TAKEN_BY_SERVE },
		{ "--session_slots=", &session_slots, 1,
		  SERVING_SESSION_CAPACITY, 1, TAKEN_BY_SERVE },
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
		{ "--wide_rows=", &wide_rows, 0, MODEL_WIDE_MOST_ROWS, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--rows=", &step_rows, 1, LINEAR_MAXIMUM_ROWS, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--context_tokens=", &context_tokens, 1, MODEL_CONTEXT_MOST,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--generated_tokens=", &generated_token_limit, 1, UINT32_MAX,
		  1, TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--seed=", &seed, 0, UINT32_MAX, 1,
		  TAKEN_BY_GENERATE | TAKEN_BY_SERVE },
		{ "--reasoning_effort=", &reasoning_effort, 0,
		  CHAT_TEMPLATE_HIGHEST_EFFORT, 1,
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
		{ "--draft_block=", &draft_block, MODEL_DRAFT_BLOCK,
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
	} texts[TEXT_OPTIONS] = { { "--bind_address=", &bind_address },
				  { "--key_file=", &key_file },
				  { "--stats_users=", &stats_users },
				  { "--model_alias=", &model_alias },
				  { "--session_directory=",
				    &session_directory },
				  { "--draft_usage=", &draft_usage },
				  { "--dump_draft_ring=", &dump_draft_ring },
				  { "--layout=", &layout_text } };
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
	bool pipeline_steps_was_given = false;
	bool original_wide_experts_was_given = false;
	bool draft_policy_was_given = false, batch_costs_were_given = false;
	bool draft_learning_was_given = false, draft_base_was_given = false;
	bool draft_tail_was_given = false, draft_row_was_given = false;
	bool batch_core_was_given = false;
	uint32_t draft_kind = SPECULATIVE_OFF;
	uint32_t calibration_given = 0, stats_format_given = 0;
	uint32_t prompt_token_ids_given = 0;
	is_serving = argument_count > 1 && strcmp(arguments[1], "serve") == 0;
	is_generating = is_serving || (argument_count > 1 &&
				       strcmp(arguments[1], "generate") == 0);
	if (argument_count < ARGUMENTS_BEFORE_THE_OPTIONS ||
	    (!is_generating && strcmp(arguments[1], "load") != 0)) {
		return false;
	}
	const uint32_t command = is_serving    ? TAKEN_BY_SERVE :
				 is_generating ? TAKEN_BY_GENERATE :
						 TAKEN_BY_LOAD;
	for (; argument_index < argument_count &&
	       strncmp(arguments[argument_index], "--", strlen("--")) == 0;
	     argument_index++) {
		const char *argument = arguments[argument_index];
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
			if (!is_serving && index != LAYOUT_TEXT && index != DUMP_TEXT) {
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
					       options[index].numbers == batch_verifiers ||
						       options[index].numbers == batch_rows,
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
						return false; /* A single zero disables planning; lists contain only prices. */
			}
		}
		if (!is_taken) {
			return false;
		}
	}
	if (is_draft_given) {
		if (speculative_was_given && speculative != draft_kind) {
			return false;
		}
		speculative = draft_kind;
	}
	if (speculative >= SPECULATIVE_DEVICE &&
	    draft_positions > draft_block) {
		if (depth_was_given) {
			return false;
		}
		draft_positions = draft_block;
	}
	card_count = given[0];
	if (official_reference != 0 &&
	    execution_placement == EXECUTION_TP4_EXL3) {
		(void)fputs("reference arithmetic (EXL3: not official-exact)\n",
			    stderr);
		return false;
	}
	generated_token_limit = generated_token_limit != 0 ?
					generated_token_limit :
				is_serving ? UINT32_MAX :
					     DEFAULT_GENERATED;
	const uint32_t file_count =
		is_serving || prompt_token_ids_path != NULL ? 1 :
		is_generating				    ? FRONT_END_FILES :
				0; /* serve's: the tables */
	for (uint32_t file = 0;
	     file < file_count && argument_index < argument_count; file++) {
		front_end_paths[FRONT_END_FILES - file_count + file] =
			arguments[argument_index++];
	}
	copy_paths = (const char *const *)&arguments[argument_index];
	copy_count = argument_index < argument_count ?
			     (uint32_t)(argument_count - argument_index) :
			     0;
	if (prompt_token_ids_path != NULL)
		front_end_paths[MESSAGES_FILE] = prompt_token_ids_path;
	if (ignore_eos != 0 &&
	    (speculative != SPECULATIVE_OFF || batch_rows[0] != 0))
		return false;
	/* --prefill_tail_rows: by default where the tail program applies; 0 the old path */
	const bool tail_applies = execution_placement == EXECUTION_TP4_EXL3 &&
				  chunks_in_flight == 1u &&
				  prefill_fast_plan == 0 && full_steps != 0 &&
				  wide_rows >= TAIL_LEAST_WIDE_ROWS &&
				  exl3_tc_moe != 0 && tp_trunk_mma != 0;
	if (prefill_tail_rows == TAIL_ROWS_AUTO)
		prefill_tail_rows = !tail_applies ? 0u :
				    wide_rows >= TAIL_LONG_WIDE_ROWS && batch_rows[0] == 0 ? TAIL_WIDEST_ROWS :
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
			bounded_default ? PREFILL_DECODE_DEFAULT_ROUNDS : 1u;
	if (prefill_decode_waiters == PREFILL_SCHEDULING_AUTO)
		prefill_decode_waiters =
			bounded_default ? PREFILL_DECODE_DEFAULT_WAITERS : 0u;
	const bool selected_combination =
		bounded_default &&
		prefill_decode_rows == PREFILL_DECODE_DEFAULT_ROWS &&
		prefill_decode_rounds == PREFILL_DECODE_DEFAULT_ROUNDS &&
		prefill_decode_waiters == PREFILL_DECODE_DEFAULT_WAITERS;
	/* Concurrent prompt rows require the measured row-independent EXL3 BATCH forms. */
	if (prefill_order == PREFILL_ORDER_SHORT_ROWS &&
	    (!bounded_batch || prefix_reuse_exact != 0 || session_fork != 0 ||
	     prefill_prefix_stable != 0))
		return false;
	if (prefill_order == PREFILL_ORDER_AUTO)
		prefill_order = selected_combination ? PREFILL_ORDER_SHORTEST :
						       0u;
	if (prefill_thin_tail == THIN_TAIL_AUTO)
		prefill_thin_tail =
			prefill_tail_rows != 0 && batch_rows[0] == 0 ? 1u : 0u;
	if (prefill_front_tail == FRONT_TAIL_AUTO)
		prefill_front_tail = prefill_tail_rows == TAIL_WIDEST_ROWS ? FRONT_TAIL_ALL :
				     prefill_tail_rows != 0 ? 1u : 0u;
	/* --prefill_ladder_fine: by default where the 448-row tail is (THIN, --wide_rows of 2,048 or more); BATCH keeps
	 * 0 for a 1M slot's memory (rk10: BATCH at 1M boots, 53 MiB free on the 640-column cards, only without it) */
	if (prefill_ladder_fine == LADDER_FINE_AUTO)
		prefill_ladder_fine = wide_rows >= TAIL_LONG_WIDE_ROWS && batch_rows[0] == 0 ? 1u : 0u;
	/* The original multi-session serving gate qualifies five PP flights. Explicit
	 * depths and BATCH retain their own admission and selected values. */
	if (is_serving && execution_placement == EXECUTION_PP5 &&
	    session_slots > 1u && batch_rows[0] == 0 &&
	    speculative == SPECULATIVE_DEVICE && !pipeline_steps_was_given)
		pipeline_steps = ORIGINAL_PP5_MULTI_PIPELINE_STEPS;
	/* Use the measured wide expert pass only where its existing admission allows it.
	 * Explicit zero retains the slab path, including in generated axis recipes. */
	if (!original_wide_experts_was_given &&
	    execution_placement == EXECUTION_TP4_PRO &&
	    numerics == NUMERICS_EXACT && pro_emulation == 0 &&
	    cmp_order == CMP_ORDER_REFERENCE &&
	    linear_order == LINEAR_ORDER_REFERENCE &&
	    linear_profile == LINEAR_PROFILE_PRECISION &&
	    wide_rows >= MODEL_WIDE_LEAST_ROWS &&
	    wide_rows <= LINEAR_WIDE_MOST_ROWS && exl3_tc_moe == 0 &&
	    tp_wire_bf16 == 0 && tp_code_wire == 0 && tp_attention_owner == 0 &&
	    batch_rows[0] == 0 && exl3_wide_entry == 0 &&
	    tp_split_overlap == 0 && tp_full_split_overlap == 0 &&
	    original_chunk_core == 0)
		original_wide_experts = 1;
	if (user_queue_depth > queue_depth ||
	    (strcmp(draft_usage, "compact") != 0 &&
	     strcmp(draft_usage, "none") != 0 &&
	     strcmp(draft_usage, "full") != 0) ||
	    (live_stats_interval_ms != 0 &&
	     live_stats_interval_ms < LIVE_STATS_LEAST_INTERVAL_MS) ||
	    (strcmp(live_stats_format, "text") != 0 &&
	     strcmp(live_stats_format, "json") != 0 &&
	     strcmp(live_stats_format, "table") != 0) ||
	    (session_slots != 1 && !is_serving) ||
	    (wide_rows != 0 &&
	     (wide_rows < MODEL_WIDE_LEAST_ROWS || prefill_cone == 0)) ||
	    ((prefill_chunk_rows != 0 || prefill_decode_rows != 0) &&
	     (chunks_in_flight != 1u || prefill_fast_plan != 0 ||
	      prefill_chunk_rows % MODEL_FULL_CONE_LEAST_ROWS != 0 ||
	      prefill_decode_rows % MODEL_FULL_CONE_LEAST_ROWS != 0 ||
	      prefill_chunk_rows > wide_rows ||
	      prefill_decode_rows > wide_rows)) ||
	    (prefill_tail_rows != 0 &&
	     ((prefill_tail_rows != TAIL_PROGRAM_ROWS &&
	       prefill_tail_rows != TAIL_LONG_ROWS &&
	       prefill_tail_rows != TAIL_WIDEST_ROWS) ||
	      !tail_applies)) ||
	    (prefill_front_tail != 0 && prefill_tail_rows == 0) ||
	    (prefill_thin_tail != 0 && prefill_tail_rows == 0) ||
	    (dense_staged != 0 &&
	     (dense_staged < DENSE_STAGED_LEAST_OUTPUTS ||
	      (dense_staged & (dense_staged - 1u)) != 0)) ||
	    (original_dense_staged != 0 &&
	     (original_dense_staged != DENSE_STAGED_OUTPUTS ||
	      execution_placement != EXECUTION_TP4_PRO)) ||
	    (tp_pipeline_overlap != 0 &&
	     (execution_placement != EXECUTION_TP4_PRO || pipeline_steps < 2u ||
	      batch_rows[0] != 0 || numerics != NUMERICS_EXACT ||
	      pro_emulation != 0 || cmp_order != CMP_ORDER_REFERENCE ||
	      linear_order != LINEAR_ORDER_REFERENCE ||
	      linear_profile != LINEAR_PROFILE_PRECISION || tp_wire_bf16 != 0 ||
	      tp_code_wire != 0 || tp_attention_owner != 0)) ||
	    (original_wide_experts != 0 &&
	     (execution_placement != EXECUTION_TP4_PRO ||
	      numerics != NUMERICS_EXACT || pro_emulation != 0 ||
	      cmp_order != CMP_ORDER_REFERENCE ||
	      linear_order != LINEAR_ORDER_REFERENCE ||
	      linear_profile != LINEAR_PROFILE_PRECISION ||
	      wide_rows < MODEL_WIDE_LEAST_ROWS || tp_wire_bf16 != 0 ||
	      tp_code_wire != 0 || tp_attention_owner != 0 ||
	      batch_rows[0] != 0 || exl3_wide_entry != 0 ||
	      tp_split_overlap != 0 || tp_full_split_overlap != 0)) ||
	    (original_batch != 0 &&
	     (execution_placement != EXECUTION_TP4_PRO ||
	      pipeline_steps != 1u ||
	      batch_rows[0] != V41_DATA_ORIGINAL_BATCH_ROWS ||
	      batch_rows[1] != 0 ||
	      batch_verifiers[0] != V41_DATA_ORIGINAL_BATCH_SESSIONS ||
	      batch_verifiers[1] != 0 ||
	      session_slots != V41_DATA_ORIGINAL_BATCH_SESSIONS ||
	      wide_rows < MODEL_WIDE_LEAST_ROWS ||
	      speculative < SPECULATIVE_DEVICE ||
	      draft_block != MODEL_DRAFT_BLOCK ||
	      draft_positions != MODEL_DRAFT_BLOCK ||
	      numerics != NUMERICS_EXACT || pro_emulation != 0 ||
	      linear_profile != LINEAR_PROFILE_PRECISION ||
	      linear_order != LINEAR_ORDER_REFERENCE ||
	      cmp_order != CMP_ORDER_REFERENCE || tp_wire_bf16 != 0 ||
	      tp_code_wire != 0 || tp_attention_owner != 0 ||
	      tp_head_shards != 0 || tp_draft_shards != 0 ||
	      tp_draft_layers != 0 || tp_head_stream != 0 || tp_head_mma != 0 ||
	      tp_trunk_mma != 0 || tp_router_mma != 0 || batch_wide_head != 0 ||
	      batch_draft_rows != 0 || batch_context_slabs != 1u ||
	      batch_core_once != 0 || batch_draft_wide != 0 ||
	      draft_bf16_rows != 0 || draft_bf16_operands != 0 ||
	      original_wide_experts != 0 || original_dense_staged != 0 ||
	      exl3_wide_entry != 0 || tp_split_overlap != 0 ||
	      tp_full_split_overlap != 0 || tp_pipeline_overlap != 0)) ||
	    (original_chunk_core != 0 &&
	     wide_rows % LINEAR_MAXIMUM_ROWS != 0) ||
	    (original_chunk_core != 0 &&
	     (execution_placement != EXECUTION_TP4_PRO ||
	      numerics != NUMERICS_EXACT || pro_emulation != 0 ||
	      linear_profile != LINEAR_PROFILE_PRECISION ||
	      linear_order != LINEAR_ORDER_REFERENCE ||
	      cmp_order != CMP_ORDER_REFERENCE ||
	      wide_rows < MODEL_WIDE_LEAST_ROWS ||
	      wide_rows > LINEAR_WIDE_MOST_ROWS || tp_wire_bf16 != 0 ||
	      tp_code_wire != 0 || tp_attention_owner != 0 ||
	      batch_rows[0] != 0 || original_wide_experts != 0 ||
	      original_batch != 0 || tp_split_overlap != 0 ||
	      tp_full_split_overlap != 0)) ||
	    lookup_ngram_min > lookup_ngram_max ||
	    (speculative != SPECULATIVE_OFF && step_rows == 1) ||
	    (speculative >= SPECULATIVE_DEVICE && session_slots > 1u &&
	     pipeline_steps == 1u && batch_rows[0] == 0) ||
	    (batch_rows[0] != 0 &&
	     ((!original_batch && execution_placement != EXECUTION_TP4_EXL3) ||
	      pipeline_steps != 1u)) ||
	    /* without the batched draft rows each verifier's walk is its own segments: the step driver runs one BATCH
	     * program beside the chunked ones, not several (model_v41_exl3_program_test) */
	    (batch_rows[1] != 0 && speculative >= SPECULATIVE_DEVICE &&
	     batch_draft_rows == 0) ||
	    batch_row_sets % LINEAR_MAXIMUM_ROWS != 0 ||
	    tp_core_heads % CORE_HEADS_STEP != 0 ||
	    ((batch_draft_rows != 0 || batch_wide_head != 0 ||
	      batch_context_slabs != 0 || batch_step_us[0] != 0 ||
	      batch_verifiers[0] != 0) &&
	     batch_rows[0] == 0)) {
		return false;
	}
	const struct startup_policy_request policy_request = {
		.numerics = numerics,
		.placement = execution_placement,
		.cmp_order = cmp_order,
		.numerics_was_given = numerics_was_given,
		.cmp_order_was_given = cmp_order_was_given
	};
	struct startup_policy policy;
	if (startup_policy_prepare(&policy_request, &policy) != STATUS_OK ||
	    (execution_placement != EXECUTION_PP5 && layout_text != NULL)) {
		return false;
	}
	numerics = policy.numerics;
	cmp_order = policy.cmp_order;
	if (official_reference != 0 &&
	    (execution_placement != EXECUTION_PP5 || step_rows != 1u ||
	     wide_rows != 0 || full_steps != 0 ||
	     context_tokens > OFFICIAL_REFERENCE_CONTEXT_LIMIT ||
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
	if ((exl3_expert_parts != EXL3_EXPERT_REFERENCE_PARTS &&
	     exl3_expert_parts != EXL3_EXPERT_OPTIONAL_PARTS) ||
	    (exl3_expert_parts == EXL3_EXPERT_OPTIONAL_PARTS &&
	     (execution_placement != EXECUTION_TP4_EXL3 ||
	      numerics == NUMERICS_EXACT || cmp_order != CMP_ORDER_REFERENCE ||
	      tp_wire_bf16 != 0)) ||
	    (execution_placement == EXECUTION_TP4_EXL3 &&
	     (tp_code_wire != 0 || tp_attention_owner != 0)) ||
	    ((tp_head_shards != 0 || tp_head_stream != 0) &&
	     execution_placement != EXECUTION_TP4_EXL3) ||
	    (tp_draft_shards != 0 && tp_head_shards == 0) ||
	    (tp_decode_round_once != 0 &&
	     (execution_placement != EXECUTION_TP4_EXL3 || tp_wire_bf16 == 0 ||
	      tp_wide_exchange < V41_DATA_TP_EXCHANGE_OWNER_SUMS ||
	      wide_rows <= LINEAR_MAXIMUM_ROWS)) ||
	    (exl3_expert_rotation != 0 &&
	     (execution_placement != EXECUTION_TP4_EXL3 || tp_wire_bf16 == 0 ||
	      exl3_tc_moe == 0 ||
	      exl3_expert_parts != EXL3_EXPERT_REFERENCE_PARTS)) ||
	    (tp_draft_layers != 0 &&
	     (tp_draft_shards == 0 || tp_wire_bf16 == 0 ||
	      execution_placement != EXECUTION_TP4_EXL3 ||
	      (batch_rows[0] != 0 &&
	       (batch_draft_rows == 0 || batch_wide_head == 0)))))
		return false;
	if (session_fork != 0 && cache_scope != SERVER_CACHE_SESSION) {
		LOG_ERROR("serve: session forks require cache_scope=0");
		return false;
	}
	if (prefill_prefix_stable != 0 &&
	    (execution_placement != EXECUTION_TP4_EXL3 || tp_wire_bf16 == 0 ||
	     tp_ring_column_owners == 0 ||
	     tp_wide_exchange != V41_DATA_TP_EXCHANGE_CE_RINGS ||
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
		return false;
	if ((prefix_reuse_exact != 0 || session_fork != 0) &&
	    !prefix_cold_plan_is_qualified()) {
		if (session_fork != 0)
			LOG_ERROR(
				"serve: session forks require TP4 EXL3 column owners, invariant THIN forms, "
				"chunks_in_flight=1, the exact plan, caps=0 and reserved multi-slot prefill");
		return false;
	}
	/* The measured CLI defaults belong to the EXL3 family. Other placements retain each omitted
	 * field's previous default; explicit choices remain independent. */
	if (execution_placement != EXECUTION_TP4_EXL3) {
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
		 * objective without inventing timings; supplied prices are always retained. */
		draft_policy.depth_policy = batch_rows[0] != 0 ||
							    session_slots > 1u ?
						    DRAFT_DEPTH_RATE :
						    DRAFT_DEPTH_FIXED;
		const bool measured_batch =
			execution_placement == EXECUTION_TP4_EXL3 &&
			tp_wire_bf16 == ORDERED_PUSH_BF16 &&
			speculative == SPECULATIVE_DEVICE &&
			draft_block == MODEL_DRAFT_BLOCK &&
			draft_positions == MODEL_DRAFT_BLOCK &&
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
		    memcmp(batch_rows, default_batch_rows,
			   sizeof(batch_rows)) == 0 &&
		    memcmp(batch_verifiers, default_batch_verifiers,
			   sizeof(batch_verifiers)) == 0)
			memcpy(batch_step_us, default_batch_step_us,
			       sizeof(default_batch_step_us));
	}
	return copy_count >= 1 &&
	       (prompt_token_ids_path == NULL || copy_count == 1u) &&
	       copy_count <= STORAGE_MAXIMUM_COPIES &&
	       (speculative != SPECULATIVE_HYBRID ||
		hybrid_lookup_min <= lookup_ngram_max) &&
	       draft_calibration_read() && draft_policy_valid(&draft_policy);
}

/* The whole of a file, not empty, in memory of malloc (aligned for any type) that the caller frees; else NULL. */
static char *file_read(const char *path, uint64_t *byte_count_out)
{
	FILE *const file = fopen(path, "rb");
	long byte_count = -1;
	char *bytes = NULL;
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
MUST_CHECK static result_status
prompt_token_ids_parse(const char *text, size_t byte_count,
		       uint32_t vocabulary_size, uint32_t *token_ids_out,
		       uint32_t token_capacity, uint32_t *token_count_out)
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
static uint32_t effective_row_capacity(void)
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
	return execution_placement == EXECUTION_TP2_PP2_PRO ||
	       execution_placement == EXECUTION_TP4_PRO ||
	       execution_placement == EXECUTION_TP2_SINGLE_SINGLE_PRO ||
	       execution_placement == EXECUTION_TP4_EXL3;
}

/* Select descriptions only: a PRO is skipped without opening its CUDA arena. */
MUST_CHECK static result_status exl3_card_order_prepare(uint32_t found)
{
	uint32_t selected[TP4_RANKS], count = 0;
	for (uint32_t ordinal = 0; ordinal < found; ordinal++) {
		struct card_description_record description = { 0 };
		const result_status status =
			backend_cuda_value.card_describe(ordinal, &description);
		if (status != STATUS_OK)
			return status;
		if (description.arithmetic_class != ARITHMETIC_CLASS_SM_80 ||
		    strstr(description.name, "CMP") == NULL)
			continue;
		if (count == TP4_RANKS)
			return STATUS_UNSUPPORTED;
		selected[count++] = ordinal;
	}
	if (count != TP4_RANKS)
		return STATUS_UNSUPPORTED;
	(void)memcpy(card_order, selected, sizeof(selected));
	card_count = count;
	return STATUS_OK;
}

/* Cold TP bootstrap. The ordinary unit map records canonical homes for identity/statistics only: selected
 * tensor jobs, rather than this map, load every replica/shard. All GPU edges are tested on the actual arenas. */
MUST_CHECK static result_status
tensor_plan_prepare(const struct model_limits *limits, uint64_t kept_free_bytes,
		    uint32_t *unit_homes)
{
	const bool exl3 = execution_placement == EXECUTION_TP4_EXL3;
	if (card_count != (exl3 ? TP4_RANKS : V41_DATA_TP2_PROGRAM_CARDS) ||
	    unit_homes == NULL)
		return STATUS_UNSUPPORTED;
	struct v41_tp2_model_description description = {
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
		.exl3_tc_moe = exl3 && tp_wire_bf16 != 0 && exl3_tc_moe != 0,
		.exl3_tc_product_blocks = exl3_tc_product_blocks,
		.exl3_expert_rotation = exl3 && tp_wire_bf16 != 0 &&
							exl3_tc_moe != 0 ?
						exl3_expert_rotation :
						0u,
		.head_mma = exl3 && tp_wire_bf16 != 0 && tp_head_mma != 0,
		.trunk_mma = exl3 && tp_wire_bf16 != 0 && tp_trunk_mma != 0,
		.trunk_mma_fused = tp_trunk_mma_fused != 0,
		.trunk_mma_thin_splits = tp_trunk_mma_thin_splits != 0,
		.router_mma = exl3 && tp_wire_bf16 != 0 && tp_router_mma != 0,
		.router_mma_wide = exl3 && tp_wire_bf16 != 0 &&
				   tp_router_mma == 2u,
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
			exl3 ? V41_DATA_TP_LAYOUT_QUAD_ALL :
			execution_placement == EXECUTION_TP4_PRO ?
			       V41_DATA_TP_LAYOUT_QUAD :
			execution_placement == EXECUTION_TP2_SINGLE_SINGLE_PRO ?
			       V41_DATA_TP_LAYOUT_HYBRID :
			       V41_DATA_TP_LAYOUT_PAIRS
	};
	for (uint32_t card = 0; card < card_count; card++)
		description.cards[card] = descriptions[card];
	result_status status =
		v41_tp2_model_create(pack, &description, &tensor_model);
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

/* Every peer reader must retire before either arena of its pair can close. Unknown completion retains the
 * entire pair through process teardown; neither a failed wait nor closing its partner is a retirement proof. */
static uint32_t tensor_cards_drain(void)
{
	uint32_t retained = (1u << card_count) - 1u;
	const uint32_t members = (execution_placement == EXECUTION_TP4_PRO ||
				  execution_placement == EXECUTION_TP4_EXL3) ?
					 TP4_RANKS :
					 TP2_RANKS;
	const result_status status = v41_tp_group_drain(
		&backend_cuda_value, cards, CARD_WAIT_LIMIT_NANOSECONDS,
		members, &retained);
	(void)status; /* retained remains the fail-closed ownership decision even after a failed wait */
	return retained;
}

MUST_CHECK static result_status tensor_cards_load(const void *tables,
						  uint64_t table_bytes)
{
	result_status status = STATUS_OK;
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
	struct v41_tp2_model_load_request request = {
		.backend = &backend_cuda_value,
		.cards = cards,
		.storage = storage_operations,
		.reader = reader,
		.wait_limit_nanoseconds = CARD_WAIT_LIMIT_NANOSECONDS,
		.load_threads = tp_load_threads,
		.load_reads_per_copy = tp_load_reads_per_copy,
		.load_host_threads = tp_load_host_threads,
		.load_copy_split = tp_load_copy_split
	};
	TIMER_BEGIN(tp_peer_qualify);
	status = status != STATUS_OK ?
			 status :
			 v41_tp2_peer_qualify(tensor_model, &backend_cuda_value,
					      cards,
					      CARD_WAIT_LIMIT_NANOSECONDS,
					      request.qualified_writer_masks);
	TIMER_END(tp_peer_qualify);
	status = status != STATUS_OK ?
			 status :
			 v41_tp2_model_load(tensor_model, &request);
	status = status != STATUS_OK ?
			 status :
			 v41_tp2_model_front_end_match(tensor_model, tables,
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
		(void)printf("%s %016" PRIx64 "%s", card == 0 ? "TP load receipts:" : "",
			     tensor_model->gpu_result.card_receipts[card], card + 1 == card_count ? "\n" : "");
	for (uint32_t copy = 0;
	     copy < copy_count && status == STATUS_OK && tp_load_threads != 0;
	     copy++) {
		const struct tp_load_copies *const copies =
			&tensor_model->copies;
		(void)printf("TP load copy %" PRIu32 ": %.2f GB in %.2f s busy (%.2f GB/s), %" PRIu32 " failed reads\n",
			     copy, (double)copies->delivered_bytes[copy] / BYTES_PER_GB,
			     (double)copies->busy_nanoseconds[copy] / NANOSECONDS_PER_SECOND,
			     copies->busy_nanoseconds[copy] != 0 ?
				     (double)copies->delivered_bytes[copy] / (double)copies->busy_nanoseconds[copy] :
				     0.0,
			     copies->failed_reads[copy]);
	}
	return status;
}

/* Turbo load: the single source's receipted replicas (replicas.h) become copies 1.. of a new reader; when that reader
 * does not open, the source alone stays (one line says so). */
MUST_CHECK static result_status replicas_add(void)
{
	const uint32_t found = replicas_find(copy_paths[0],
					     pack_header_of(pack),
					     replica_paths, REPLICAS_MOST);
	struct storage_reader *wider = NULL;
	turbo_copy_paths[0] = copy_paths[0];
	for (uint32_t index = 0; index < found; index++)
		turbo_copy_paths[index + 1u] = replica_paths[index];
	if (found != 0 &&
	    storage_operations->reader_open(turbo_copy_paths, found + 1u,
					    &wider) != STATUS_OK) {
		(void)printf("turbo load: %" PRIu32
			     " replicas do not open: the source alone\n",
			     found);
		return STATUS_OK;
	}
	if (found != 0) {
		storage_operations->reader_close(reader);
		reader = wider;
		copy_paths = turbo_copy_paths;
		copy_count = found + 1u;
	}
	for (uint32_t index = 0; index < found; index++)
		(void)printf("turbo load: copy %" PRIu32 " %s\n", index + 1u,
			     replica_paths[index]);
	return STATUS_OK;
}

/* The exact reference recipe admits text even when the pack also contains a vision tower. */
static bool model_images_are_enabled(const struct model_plan_record *model_plan)
{
	return model_plan != NULL && model_plan->vision.enabled &&
	       model_plan->limits.official_reference == 0;
}

/* Once the pack is loaded: the n-gram tables, the state, the programs, the driver and the engine over them. Every
 * step ends as step_failed says; true when one failed. */
MUST_CHECK static bool
engine_failed_to_start(const struct engine_description_record *description)
{
	const bool image_programs_enabled = model_images_are_enabled(plan);
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
	result_status status =
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
	     tables != NULL && table < tables->table_count &&
	     status == STATUS_OK;
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
				   model->arena_initialise(
					   plan, &backend_cuda_value, cards,
					   CARD_WAIT_LIMIT_NANOSECONDS)));
	struct v41_tp2_program_description tensor_description = { 0 };
	failed = failed ||
		 (tensor_model != NULL &&
		  step_failed("describe TP programs",
			      v41_tp2_model_program_description(
				      tensor_model, &tensor_description)));
	const uint32_t row_capacity = effective_row_capacity();
	/* --batch_rows: BATCH programs of those rows in place of the thin ones (the engine's decode steps, any sessions) */
	const bool batch = batch_rows[0] != 0 && tensor_model != NULL;
	program_count = 0;
	uint32_t verifiers[MOST_PROGRAMS] = { 0 }; /* --batch_verifiers: each BATCH program's, the others 0 */
	bool verifiers_given = batch_verifiers[0] != 0, rows_repeat = false;
	for (uint32_t index = 0; index < MOST_BATCH_PROGRAMS && batch_rows[index] != 0; index++)
		for (uint32_t other = 0; other < index; other++)
			rows_repeat = rows_repeat || batch_rows[other] == batch_rows[index];
	failed = failed ||
		 (batch && rows_repeat && !verifiers_given &&
		  step_failed(
			  "a repeated BATCH row count needs --batch_verifiers",
			  STATUS_INVALID_ARGUMENT));
	for (uint32_t index = 0; batch && index < MOST_BATCH_PROGRAMS && batch_rows[index] != 0 && !failed; index++) {
		tensor_description.batch_verifiers = batch_verifiers[index];
		verifiers[program_count] = batch_verifiers[index];
		verifiers_given = verifiers_given && batch_verifiers[index] != 0;
		failed = step_failed(
			"prepare a BATCH program",
			v41_tp_program_prepare(
				&tensor_description, &backend_cuda_value, cards,
				descriptions, PROGRAM_BATCH, batch_rows[index],
				&programs[program_count],
				&tensor_companions[program_count]));
		most_rows = batch_rows[index] > most_rows ? batch_rows[index] : most_rows;
		program_count += failed ? 0u : 1u;
	}
	tensor_description.batch_verifiers = 0;
	failed = failed || (batch_verifiers[0] != 0 && !verifiers_given &&
			    step_failed("one verifier count a BATCH program",
					STATUS_INVALID_ARGUMENT));
	for (program_count = batch ? program_count : 0; !batch && program_count < programs_to_prepare && !failed;
	     program_count++) {
		failed = step_failed(
			"prepare a program",
			tensor_model != NULL ?
				v41_tp2_program_prepare(
					&tensor_description,
					&backend_cuda_value, cards,
					descriptions,
					program_rows[program_count],
					&programs[program_count],
					&tensor_companions[program_count]) :
				model->program_prepare(
					plan, &backend_cuda_value, cards,
					descriptions, PROGRAM_THIN,
					program_rows[program_count],
					&programs[program_count]));
		if (!failed &&
		    !serving_program_matches(
			    programs[program_count],
			    program_rows[program_count], row_capacity,
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
	}
	/* behind the thin programs, chunks of wide_rows, a quarter of it, ..., at least the least wide rows, which ends
	 * the ladder (a prompt's rows below it go in thin steps): wide ones for the front rows, full ones for the rows
	 * behind the front (the engine picks a size: engine.c). Above CHUNK_LADDER_TOP: wide_rows, 512, 256, 64 (the
	 * exact plan's whole multiples of 256 and its tail's 64; the fast plan's 2,048 + 512 + 64 of a 2.7K prompt). */
	uint32_t chunk_rows[MOST_CHUNK_PROGRAMS + 2u + LADDER_FINE_COUNT +
			    PREFILL_CHUNK_CAPS],
		chunk_count = 0;
	for (uint32_t rows = wide_rows;
	     rows >= MODEL_WIDE_LEAST_ROWS &&
			     chunk_count<MOST_CHUNK_PROGRAMS; rows = rows>
				     CHUNK_LADDER_TOP ?
		     CHUNK_LADDER_TOP / 2u :
	     rows == CHUNK_LADDER_TOP / 2u && wide_rows > CHUNK_LADDER_TOP ?
		     rows / 2u :
		     rows / CHUNK_ROWS_DIVISOR)
		chunk_rows[chunk_count++] = rows;
	if (chunk_count != 0 && chunk_count < MOST_CHUNK_PROGRAMS &&
	    chunk_rows[chunk_count - 1u] != MODEL_WIDE_LEAST_ROWS)
		chunk_rows[chunk_count++] = MODEL_WIDE_LEAST_ROWS;
	/* --prefill_ladder_1024: CHUNK_LADDER_TOP's own programs behind wide_rows' (the exact plan's whole multiples of
	 * 256: a region of 1,024 to 2,047 whole rows in one step fewer; BATCH programs take the thin ones' place, so
	 * MOST_PROGRAMS holds them) */
	if (prefill_ladder_1024 != 0 && chunk_count > 1u && chunk_rows[0] > CHUNK_LADDER_TOP) {
		memmove(&chunk_rows[2], &chunk_rows[1], (chunk_count - 1u) * sizeof(chunk_rows[0]));
		chunk_rows[1] = CHUNK_LADDER_TOP;
		chunk_count++;
	}
	/* --prefill_ladder_1536: LADDER_1536_ROWS' programs too (a region of 1,536 to 2,047 whole rows, e.g. Curtis's 2K
	 * cell's 1,024 + 512, in one step; each split half 768 rows) */
	if (prefill_ladder_1536 != 0 && chunk_count > 1u && chunk_rows[0] > LADDER_1536_ROWS) {
		memmove(&chunk_rows[2], &chunk_rows[1], (chunk_count - 1u) * sizeof(chunk_rows[0]));
		chunk_rows[1] = LADDER_1536_ROWS;
		chunk_count++;
	}
	/* Fine ladder and explicit caps, in the ladder's order. A single slot cannot decode beside another prompt, so
	 * its decoder-only cap prepares no extra programs. The global cap is also used to qualify its numerics. */
	static const uint32_t fine_rows[LADDER_FINE_COUNT] = { 1792u, 1280u, 768u };
	uint32_t extra_rows[LADDER_FINE_COUNT + PREFILL_CHUNK_CAPS] = { 0 };
	for (uint32_t fine = 0; prefill_ladder_fine != 0 && fine < LADDER_FINE_COUNT; fine++)
		extra_rows[fine] = fine_rows[fine];
	extra_rows[LADDER_FINE_COUNT] = prefill_chunk_rows;
	extra_rows[LADDER_FINE_COUNT + 1u] = session_slots > 1u ? prefill_decode_rows : 0u;
	for (uint32_t extra = 0; chunk_count > 1u && extra < LADDER_FINE_COUNT + PREFILL_CHUNK_CAPS; extra++) {
		if (extra_rows[extra] == 0)
			continue;
		uint32_t at = 0;
		while (at < chunk_count && chunk_rows[at] > extra_rows[extra])
			at++;
		if (at == 0 || at == chunk_count || chunk_rows[at] == extra_rows[extra])
			continue;
		memmove(&chunk_rows[at + 1u], &chunk_rows[at], (chunk_count - at) * sizeof(chunk_rows[0]));
		chunk_rows[at] = extra_rows[extra];
		chunk_count++;
	}
	const enum program_kind chunk_kinds[] = {
		PROGRAM_WIDE, full_steps != 0 ? PROGRAM_FULL : PROGRAM_WIDE
	};
	for (uint32_t kind = 0;
	     kind < sizeof(chunk_kinds) / sizeof(chunk_kinds[0]) &&
	     (kind == 0 || chunk_kinds[kind] != chunk_kinds[0]);
	     kind++) {
		for (uint32_t step = 0; !failed && step < chunk_count && program_count < MOST_PROGRAMS; step++) {
			const uint32_t rows = chunk_rows[step];
			failed = step_failed(
				chunk_kinds[kind] == PROGRAM_WIDE ?
					"prepare a wide program" :
					"prepare a full program",
				tensor_model != NULL ?
					v41_tp_program_prepare(
						&tensor_description,
						&backend_cuda_value, cards,
						descriptions, chunk_kinds[kind],
						rows, &programs[program_count],
						&tensor_companions
							[program_count]) :
					model->program_prepare(
						plan, &backend_cuda_value,
						cards, descriptions,
						chunk_kinds[kind], rows,
						&programs[program_count]));
			program_count += failed ? 0u : 1u;
		}
	}
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
	for (uint32_t tail = 0; tail < sizeof(tails) / sizeof(tails[0]) && !failed && tensor_model != NULL &&
				program_count < MOST_PROGRAMS;
	     tail++) {
		if (!tails[tail].wanted)
			continue;
		failed = step_failed(
			"prepare a tail program",
			v41_tp_program_prepare(
				&tensor_description, &backend_cuda_value, cards,
				descriptions, tails[tail].kind,
				tails[tail].rows, &programs[program_count],
				&tensor_companions[program_count]));
		program_count += failed ? 0u : 1u;
	}
	/* --prefill_thin_tail: full programs of 16 to 56 rows, 8 apart: the tail's rows short of a 64-row chunk but its
	 * row of logits in one step, each row in a thin step's forms; with --prefill_front_tail=2 wide ones too */
	for (uint32_t kind = 0; kind < (prefill_front_tail == FRONT_TAIL_ALL ? THIN_TAIL_KINDS : 1u); kind++)
		for (uint32_t rows = THIN_TAIL_LEAST_ROWS;
		     prefill_thin_tail != 0 && rows < MODEL_WIDE_LEAST_ROWS &&
		     !failed && tensor_model != NULL &&
		     program_count < MOST_PROGRAMS;
		     rows += LINEAR_MAXIMUM_ROWS) {
			failed = step_failed(
				"prepare a thin-tail program",
				v41_tp_program_prepare(
					&tensor_description,
					&backend_cuda_value, cards,
					descriptions,
					kind == 0 ? PROGRAM_FULL : PROGRAM_WIDE,
					rows, &programs[program_count],
					&tensor_companions[program_count]));
			program_count += failed ? 0u : 1u;
		}
	for (uint32_t index = 0;
	     !failed && image_programs_enabled && index < program_count;
	     index++) {
		failed = step_failed(
			"prepare an image program",
			v41_image_program_prepare(plan, &backend_cuda_value,
						  cards, descriptions,
						  programs[index]->kind,
						  programs[index]->row_count,
						  &image_programs[index]));
	}
	return failed ||
	       step_failed("create the driver",
			   step_driver_create(
				   &backend_cuda_value, cards,
				   (const struct execution_program *const *)
					   programs,
				   program_count, engram,
				   CARD_WAIT_LIMIT_NANOSECONDS, &driver)) ||
	       (verifiers_given &&
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
		step_failed("attach image programs",
			    step_image_programs_attach(
				    driver,
				    (const struct execution_program *const *)
					    image_programs,
				    program_count))) ||
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
			   engine_create(driver, description, most_rows,
					 &engine));
}

/* The digest of the words that hold range on card (the backend digests whole words). */
MUST_CHECK static result_status
aligned_digest(uint32_t card, struct linear_range range, uint64_t *digest_out)
{
	const uint64_t word = PACK_DIGEST_WORD_BYTES,
		       first = range.byte_offset / word * word;
	const uint64_t end = (range.byte_offset + range.byte_count + word - 1u) / word * word;
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
		(uint64_t)DRAFT_RING_DUMP_ROWS * MODEL_STATE_ROW_BYTES;
	const uint64_t positions = plan->limits.draft_positions, *const arrays = plan->cards[card].array_offset;
	const struct linear_range ranges[] = {
		{ plan->cards[card].ring_offset[MODEL_DRAFT_FIRST_LAYER],
		  bytes },
		{ plan->cards[card].ring_offset[MODEL_DRAFT_FIRST_LAYER + 1u],
		  bytes },
		{ plan->cards[card].ring_offset[MODEL_DRAFT_FIRST_LAYER + 2u],
		  bytes },
		{ arrays[V41_DATA_ARRAY_ROTATION],
		  LINEAR_MAXIMUM_ROWS * MODEL_ROTATION_ROW_BYTES },
		{ arrays[V41_DATA_ARRAY_DRAFT_STEP],
		  STEP_BLOCK_BYTES_VALUE + positions * STEP_ROW_BYTES_VALUE },
		{ arrays[V41_DATA_ARRAY_DRAFT_COLLAPSE],
		  positions * MODEL_HIDDEN_ROW_BYTES },
		{ arrays[V41_DATA_ARRAY_DRAFT_RESULTS],
		  positions * STEP_DRAFT_RESULT_BYTES },
		{ arrays[V41_DATA_ARRAY_HIDDEN],
		  LINEAR_MAXIMUM_ROWS * MODEL_HIDDEN_ROW_BYTES },
		{ arrays[V41_DATA_ARRAY_DRAFT_MARKOV],
		  positions * MODEL_MARKOV_RANK * sizeof(uint16_t) },
		/* the draft logits' first 2 MiB (DIAG_DRAFT_PROBE's probes; else the logits) */
		{ arrays[V41_DATA_ARRAY_DRAFT_LOGITS], bytes },
		{ arrays[V41_DATA_ARRAY_DRAFT_LOGITS] + bytes, bytes },
		{ arrays[V41_DATA_ARRAY_DRAFT_LOGITS] + 2u * bytes, bytes },
		{ arrays[V41_DATA_ARRAY_DRAFT_LOGITS] + 3u * bytes, bytes },
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
	     layer < V41_DATA_WALK_LAYERS && status == STATUS_OK; layer++) {
		uint64_t mixed = 0;
		for (uint32_t index = 0;
		     index < V41_DATA_LAYER_TENSOR_COUNT && status == STATUS_OK;
		     index++) {
			const struct v41_tensor *const tensor =
				&plan->layer_tensors[layer][index];
			uint64_t digest = 0;
			if (tensor->stored.byte_count == 0 || tensor->card != card)
				continue;
			status = aligned_digest(card, tensor->stored, &digest);
			mixed = (mixed ^ digest) * DIGEST_MIX_PRIME;
			if (layer >= MODEL_DRAFT_FIRST_LAYER)
				(void)fprintf(stderr,
					      "weights draft stage %" PRIu32
					      " tensor %" PRIu32
					      " digest %016" PRIx64 "\n",
					      layer - MODEL_DRAFT_FIRST_LAYER,
					      index, digest);
		}
		(void)fprintf(stderr, "weights layer %" PRIu32 " digest %016" PRIx64 "\n", layer, mixed);
	}
	for (uint32_t index = 0;
	     index < V41_DATA_GLOBALS + V41_DATA_DRAFT_GLOBALS &&
	     status == STATUS_OK;
	     index++) {
		const struct v41_tensor *const tensor =
			index < V41_DATA_GLOBALS ?
				&plan->globals[index] :
				&plan->draft_globals[index - V41_DATA_GLOBALS];
		uint64_t digest = 0;
		if (tensor->stored.byte_count == 0 || tensor->card != card)
			continue;
		status = aligned_digest(card, tensor->stored, &digest);
		(void)fprintf(stderr, "weights global %" PRIu32 " digest %016" PRIx64 "\n", index, digest);
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
	static const char *const kind_names[STATE_DIGEST_KINDS] = { "ring", "compressor", "pool", "keys" };
	result_status status = STATUS_OK;
	for (uint32_t card = 0; card < plan->card_count && status == STATUS_OK;
	     card++) {
		const struct v41_card *const holder = &plan->cards[card];
		const uint64_t layers = v41_walk_layers(holder);
		for (uint32_t layer = 0;
		     layer < V41_DATA_WALK_LAYERS && status == STATUS_OK;
		     layer++) {
			if ((layers >> layer & 1u) == 0)
				continue;
			const struct linear_range ranges[STATE_DIGEST_KINDS] = {
				{ holder->ring_offset[layer],
				  V41_DATA_RING_SESSION_BYTES },
				{ holder->compressor_offset[layer],
				  V41_DATA_COMPRESSOR_SESSION_BYTES },
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
						ranges[kind].byte_count - block * STATE_DIGEST_BLOCK_BYTES;
					const struct linear_range part = {
						ranges[kind].byte_offset +
							block * STATE_DIGEST_BLOCK_BYTES,
						left < STATE_DIGEST_BLOCK_BYTES ?
							left :
							STATE_DIGEST_BLOCK_BYTES
					};
					uint64_t digest = 0;
					status = aligned_digest(card, part, &digest);
					(void)fprintf(stderr,
						      "state card %" PRIu32 " layer %" PRIu32 " %s block %" PRIu64
						      " digest %016" PRIx64 "\n",
						      card, layer, kind_names[kind], block, digest);
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
	const result_status status =
		is_ready ? server_run(server, engine, tokenizer, stop_pipe[0]) :
			   STATUS_INPUT_OUTPUT_ERROR;
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

static int startup_placement_status(void)
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
	if (execution_placement == EXECUTION_TP4_EXL3) {
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
	    (execution_placement != EXECUTION_TP4_PRO ||
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
	     context_tokens < MODEL_CONTEXT_WITHOUT_CANDIDATES ||
	     step_rows > LINEAR_MAXIMUM_ROWS ||
	     speculative < SPECULATIVE_DEVICE)) {
		(void)fprintf(
			stderr,
			execution_placement == EXECUTION_TP4_EXL3 ?
				"TP4 EXL3 requires reference CMP products, context>=16384,"
				" decode rows<=8 and DSpark\n" :
				"TP requires native PRO, reference local CMP products, context>=16384,"
				" decode rows<=8 and DSpark\n");
		return PROGRAM_FAILED;
	}
	if (tensor_placement_selected() && pipeline_steps > 1u)
		(void)printf(
			tp_pipeline_overlap != 0 ?
				"TP queue depth %u: separate CMP-prefix and native-PRO-tail workers\n" :
				"TP queue depth %u: one coordinator executes complete flights serially; no GPU overlap\n",
			pipeline_steps);
	return PROGRAM_SUCCEEDED;
}

static int run_program(int argument_count, char **arguments,
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
	uint32_t cards_found = 0;
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
			" [--tp_head_shards=0|1 (default 0; tp4_exl3: target head and sampler cut over the ranks, same bits)]"
			" [--tp_exact_dense=0|1 (default 1; tp4_exl3 exact recipe: attention wq_b/wo_a/wo_b output shards, same"
			" bits; other placements and the BF16 recipe ignore it)]"
			" [--tp_head_stream=0|1 (default 0; tp4_exl3: head products by the dense_stream kernel, same bits)]"
			" [--exl3_fast_products=0|1 (default 1; tp4_exl3: exl3_v3_claude's EXL3 products, identical words; 0: the"
			" retained chain)]"
			" [--exl3_wide_entry=0|1 (default 1; tp4_exl3: a chunk's EXL3 experts in one call over its rows, the same"
			" words; 0: a call a slab)]"
			" [--exl3_tc_moe=0|1 (default 0; tp4_exl3 BF16 recipe: GNmac's tensor-core EXL3 MoE, its arithmetic (FP16"
			" tensor-core products on a hi/lo split of the activations, FP64 transforms); the exact recipe ignores it)]"
			" [--exl3_expert_rotation=0|1|2 (default 0; with --exl3_tc_moe=1: 1 rank r of layer L holds chunk"
			" (r + L + e) % 4 of expert e (Morrow's); 2 rank 0 chunk 0, ranks 1-3 three ways: other TP sum order)]"
			" [--tp_head_mma=0|1 (default 0; tp4_exl3 BF16 recipe: the head products (target, draft, a BATCH step's"
			" wide head) by BF16 tensor-core MMAs, FP32 sums in the MMA order, GNmac's class of head arithmetic)]"
			" [--tp_trunk_mma_thin_splits=0|1 (default 0; with --tp_trunk_mma=1: the _64 form splits as the 1-8-row"
			" form: a 9-64-row step's trunk rows get the thin steps' bits)]"
			" [--tp_trunk_mma_fused=0|1 (default 0; with --tp_trunk_mma=1: the _64 form adds its own splits, no reduce"
			" launch, the same bits; relbench03: 1-5 us slower than the form and its reduce)]"
			" [--tp_gather_balance=0..3 (default 0; tp4_exl3: a chunk's index scans and router rows by"
			" the 512-column expert ranks alone, the same bits; 2: wide steps' alone; 3: the 640-column"
			" ranks 3/16 each, the 512-column ones the rest)]"
			" [--tp_full_split_overlap=0..7 (default 0: --tp_split_overlap's; the full steps' mode,"
			" the same bits)]"
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
			" [--tp_router_mma=0|1|2 (default 0; 2: prefill steps' routers too; tp4_exl3 BF16 recipe: a BATCH step's"
			" router (9-64 rows) by split-K"
			" BF16 tensor-core MMAs and a fixed-order reduce, GNmac's router class; another order than the dense one)]",
			stderr);
		(void)fputs(
			" [--tp_queries_piece=0|1 (default 1; tp4_exl3: a wide program's queries over each piece with"
			" wq_b, from its compact heads, not a span at a time; the same bits)]"
			" [--tp_index_span=0|1 (default 1; tp4_exl3: a chunk's index scans and selections up to 64 rows a"
			" launch, not 8, within 32 MiB of scores a card; the same bits)]"
			" [--dense_skinny_bf16=0|4|8 (default 0; tp4_exl3 class 80: the BF16 skinny dense products above 48"
			" rows at so many outputs a block, not 1; the same bits)]"
			" [--dense_skinny_f32=0|1|8 (default 0: 4; the float32-weight ones (routers) alike)]"
			" [--tp_core_compact_halves=0|1 (default 1; tp4_exl3: a split half's core writes wo_a's compact"
			" rows, no copy; the same bits)]"
			" [--tp_trunk_mma=0|1 (default 0; tp4_exl3 BF16 recipe: the trunk products of BF16 rows in steps of 1-8"
			" rows by GNmac's FP8 gemv arithmetic (E4M3 weights made exact BF16, BF16 tensor-core MMAs, FP32 sums))]"
			" [--tp_parallel_launch=0|1 (default 0; TP: the cards' graphs of a step's group launched from a thread a"
			" card, not in turn; the same launches a card, the same bits)]"
			" [--tp_graph_parts=1..8 (default 1; TP: a card's step graph behind the prefill cone in this many parts, so"
			" that a step begins after a smaller launch; the same launches in the same order, the same bits)]"
			" [--tp_graph_branches=0|1|2 (default 1; tp4_exl3 with --exl3_tc_moe=1: a thin layer's shared expert a"
			" step-graph branch beside its routed experts, joined by the ring, as GNmac's auxiliary stream, the same"
			" bits; 2: BATCH steps' too; 0: one chain)]"
			" [--tp_ordered_wide=0|1 (default 0; tp4_exl3: attention trunk products of more than 8 rows by the"
			" ordered wide form on the tensor cores, the reference's bits)]"
			" [--tp_piece_rings=0|1 (default 0; tp4_exl3 BF16 recipe: steps of more than 8 rows complete the FFN (and,"
			" with --tp_bf16_slab_rings=1, the attention) by BF16 rings over 64-row pieces, not 8-row slabs; same bits)]"
			" [--tp_trunk_shape_slices=0|1 (default 1; tp4_exl3: thin trunk products at the slice count measured for"
			" their shard shape, the same bits (ordered chains); 0: four slices)]"
			" [--tp_bf16_slab_rings=0|1 (default 0; tp4_exl3 BF16 recipe: chunked steps sum the attention a slab at a"
			" time by the ring, as thin steps, not one FP32 push a piece: other bits than 0)]"
			" [--tp_chunk_core=0|1 (default 1; tp4_exl3: a chunked step's attention core in one launch over the"
			" chunk's rows behind every slab's state writes and scan, not a slab at a time; the same bits; pf16:"
			" 30K-token prefill 18.93 -> 17.80 s; 0: a core a slab)]"
			" [--tp_split_least_rows=N (default 0; fewer rows unsplit)]"
			" [--tp_launch_trims=0|1 (default 1)]"
			" [--tp_split_overlap=0..7 (default 7: 6 with that call's gate/up tiles over both halves and"
			" its down GEMM and combine a half each, the first half's FFN sums beside the second's down,"
			" same bits; 6: 5 with one expert call a chunk, same bits;"
			" 5: pf25 14.54 -> 12.92 s, identical; tp4_exl3"
			" BF16 recipe: a chunked step's layer from wo_a on in"
			" two halves of its rows, each half's sums as a graph side run beside the other half's work (Morrow's"
			" PREFILL_REDUCE_OVERLAP on step graphs), the same bits; 2: the halves one behind the other; 3: only the"
			" attention's sums beside work; 4: only the FFN's; 5: 1 with the attention in halves too, each layer's"
			" last sums beside the next layer's first half, the same bits)]"
			" [--tp_wide_exchange=0|1|2|3|4|5 (default 5 with --tp_split_overlap=1: pf19 30K-token prefill 14.80 s"
			" against 19.08, identical; 4: the owner sums' bytes moved by copy engines; 5: 4 and the FFN's slab"
			" rings as copy-engine owner ring sums; both the same bits; was 2; tp4_exl3 BF16 recipe: a chunked piece's sums: 2 each rank sums"
			" its shard of rows, half the push's bytes (pf07: 30K-token prefill 22.05 -> 18.85 s, the same bits); 1"
			" the ordered push by many CTAs; 3: 2 and the FFN's slab rings as one exchange a piece (the same bits,"
			" slower: 23.25 s); 0: one CTA)]"
			" [--tp_replicated_embedding=0|1 (default 1; tp4_exl3: every rank embeds its own rows, the same bits; 0:"
			" card 0 embeds and hands the streams off through the host)]",
			stderr);
		(void)fputs(
			" [--batch_rows=R[,R..] (1..64 each; tp4_exl3 with --pipeline_steps=1: BATCH programs of those rows in place"
			" of the thin ones, several sessions' rounds a step; 8 rows or fewer walk as thin programs)]"
			" [serve: --original_batch=0|1 (default 0; original exact TP4_PRO BATCH48 pilot: eight slots/verifiers,"
			" depth1, DSpark5, context_slabs1, core_once0, draft_wide0, exl3_wide_entry0, split_overlap0;"
			" native per-verifier draft walks, wide head/draft rows off)]"
			" [--batch_draft_rows=0|1] [--batch_wide_head=0|1] [--batch_context_slabs=0|1 (default 0; a BATCH step's"
			" draft context products in slabs of 8 rows, the thin steps' forms)]"
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
			" [--batch_core_once=0..3 (default 3 for tp4_exl3, 2 elsewhere; 3: pools and deferred slots; a BATCH step's core and"
			" window once"
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
			" [--tp_draft_shards=0|1 (default 0; with --tp_head_shards=1: the draft's head, Markov and sampler"
			" cut too)] [--tp_draft_layers=0|1 (default 0; BF16 recipe with --tp_draft_shards=1, BATCH with"
			" --batch_draft_rows=1 --batch_wide_head=1: the draft's three layers tensor-parallel, GNmac's DSpark,"
			" each rank its vocabulary cut of the draft; drafts only, the verified tokens unchanged)]"
			" [--tp_wire_bf16=0|1 (default 0; experimental relaxed TP wire cast, FP32 sums)]"
			" [--prefill_ladder_1536=0|1 (default 1; with --wide_rows above 1536: programs of 1536 rows in the"
			" ladder too, the exact plan's same bits in fewer steps)]"
			" [--prefill_ladder_fine=0|1 (default 1 with --wide_rows=2048 and no BATCH; 1792, 1280 and 768"
			" rows alike)]",
			stderr);
		/* the load's options: a literal of their own (each at most 4095 bytes, ISO C99) */
		(void)fprintf(
			stderr,
			" [--tp_load_threads=0..%u (default %u; TP placements: the cards' load on N workers beside the host"
			" arena's, the same bytes; 0: the serial loads)]"
			" [--tp_load_reads_per_copy=0..%u (default %u; reads in flight a copy at most, 0: no limit)]"
			" [--tp_load_host_threads=0..%u (default %u; the host arena's threads; 0: as many as --tp_load_threads)]"
			" [--tp_load_copy_split=0|1 (default 1: the host arena's chunks from copy 0, the cards' extents from"
			" the last copy; 0: each read from the copy soonest to finish it)]"
			" [--turbo_load=0|1 (default 1: the source's receipted replicas on other drives, from"
			" `anemone replicate`, are read beside it; 0: the source alone)]"
			" [--map_reads=0..4 (default 4: 3 with replicas, else 1; 3: an EXL3 map's aligned pieces O_DIRECT, the"
			" rest buffered then dropped from the page cache; 0: all buffered and kept; 1: all buffered then dropped;"
			" 2: aligned O_DIRECT, rest kept)]",
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_THREADS,
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_READS_PER_COPY,
			TP_LOAD_MAXIMUM_WORKERS, DEFAULT_TP_LOAD_HOST_THREADS);
		(void)fprintf(
			stderr,
			" [--reader_threads_per_copy=1..%d] [--card_memory_kept_free_mib=N]"
			" [--pro_emulation=0|1 (default 0: the PRO's own arithmetic, bits depend on the placement)]"
			" [generate and serve: --rows=1..%u"
			" --context_tokens=1..%u --generated_tokens=N --seed=N --temperature=X --top_p=X"
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
			" --draft_policy=0..3 (fixed|cost|rate|adaptive; tp4_exl3 default 0 single-slot THIN, 2 BATCH/multiple; elsewhere 1)"
			" --draft_threshold=0..1000 (default 100)"
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
			LINEAR_MAXIMUM_ROWS, MODEL_CONTEXT_MOST,
			CHAT_TEMPLATE_HIGHEST_EFFORT, STEP_PIPELINE_MOST,
			MODEL_WIDE_MOST_ROWS, STEP_PIPELINE_MOST,
			SERVING_DRAFT_MOST, SERVING_DRAFT_MOST,
			DRAFT_LOOKUP_MOST_NGRAM, DEFAULT_LOOKUP_NGRAM_MOST,
			DEFAULT_LOOKUP_NGRAM_LEAST);
		(void)fprintf(
			stderr,
			" [serve: --session_slots=1..%u"
			" --queue_depth=1..%u (default %u)"
			" --user_queue_depth=1..queue_depth (default half queue_depth, at least 1)"
			" --draft_usage=none|compact|full (default compact)"
			" --bind_address=%s --port=1..%u --key_file=<file> --model_alias=<name>"
			" --session_directory=<snapshots of sessions>"
			" --prefix_reuse=0|1 (default 1; 0 disables GPU prefix reuse and refuses session_directory)"
			" --prefix_reuse_exact=0|1|2 (default 0; 1 final-row repeat; 2 cold boundaries;"
			" fixed exact plan; prefill_chunk_rows=prefill_decode_rows=0;"
			" multiple slots require prefill_first=1 or prefill_order; no snapshots)"
			" --live_stats=0|100..60000 --live_stats_format=text|json|table --stats_users=<names>]"
			" [generate: <messages file>]"
			" [generate and serve: <front-end tables file>] <pack copy> [... up to %d copies]\n",
			SERVING_SESSION_CAPACITY, SERVER_MOST_WAITING,
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
			" [--session_fork=0|1 (default 0; exact same-owner independent slot clone, cache_scope=0)]"
			" [--session_save_tokens=N (default 1024)] [--session_load_tokens=N (default 256)]"
			" [--session_idle_save_seconds=1..86400 (default 120)]\n",
			stderr);
		(void)fputs(
			"       generate and serve, the exact plan's tail: [--prefill_tail_rows=0|128|320|448"
			" (default 128, 448 with --wide_rows=2048 and no BATCH, where it"
			" applies: tp4_exl3, --chunks_in_flight=1, --wide_rows of 1024 or more, --exl3_tc_moe=1,"
			" --tp_trunk_mma=1: the exact plan's 64-row tail chunks two (320: five, 448: seven or five, else"
			" two) a step, each in the 64-row chunk's forms; 0 one a step)]"
			" [--prefill_ladder_1024=0|1 (default 1; 1 with --wide_rows above 1024: programs of 1024 rows"
			" in the ladder too, the exact plan's same bits in fewer steps)]"
			" [--prefill_front_tail=0|1|2 (default 1 where the tail program applies, 2 with the 448-row"
			" tail: a wide one for the front region's tail too, its bits; 2: its 320- and 448-row ones and"
			" the thin-tail ones too)]"
			" [--prefill_tail_overlay=0|1 (default 1: the tail programs'"
			" TC MoE workspace over the prefill MoE's, the same bits; 0 its own)] [--prefill_tail_sets=0|1"
			" (default 1: a tail program's trunk products one launch over its 64-row groups, each group's"
			" bits; 0 a launch a group)]"
			" [--prefill_thin_tail=0|1 (default 1 with the tail program and no BATCH: full programs of 16 to"
			" 56 rows take the tail's"
			" rows short of a 64-row chunk in one step, each row in a thin step's forms)]"
			" [--dense_staged=0|8|16|32|64 (default 16: a wide dense product's 8-row group"
			" staged once in shared memory for the selected number of outputs, each output's sums as before:"
			" the same bits; 0 the wide forms)]"
			" [--original_dense_staged=0|16 (default 0; original TP4 CMP BF16 wide dense candidate;"
			" 0 retains the old forms)]"
			" [--original_wide_experts=0|1 (default 1 for compatible TP4_PRO; 0: slab path; wide jobs across"
			" the chunk, eight-row lossless gathers; requires --exl3_wide_entry=0 --tp_split_overlap=0)]"
			" [--original_chunk_core=0|1 (default 0; 1: exact original full-head core over one chunk of whole"
			" eight-row slabs;"
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
	axis_log(axis_arguments, stderr);
	const int placement_status = startup_placement_status();
	if (placement_status != PROGRAM_SUCCEEDED)
		return placement_status;
	first_step_started_seconds = step_started_seconds = monotonic_seconds();
	for (uint32_t file = 0; is_generating && file < FRONT_END_FILES;
	     file++) {
		front_end_files[file] =
			front_end_paths[file] != NULL ?
				file_read(front_end_paths[file],
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
			.bind_address = bind_address,
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
				      .is_seeded = is_seed_given }
		};
		failed = step_failed("open the tables and the socket",
				     status != STATUS_OK ?
					     status :
					     server_open(&options, &server));
	} else if (is_generating) {
		const struct chat_template_options options = {
			.reasoning_effort =
				reasoning_effort != 0 ?
					reasoning_effort :
					CHAT_TEMPLATE_HIGHEST_EFFORT,
			.is_thinking = reasoning_effort != 0,
			.drops_thinking = true
		};
		const size_t prompt_capacity =
			(size_t)context_tokens * TOKENIZER_MAXIMUM_TOKEN_BYTES;
		prompt = prompt_token_ids_path == NULL ?
				 malloc(prompt_capacity) :
				 NULL;
		token_ids = calloc(context_tokens, sizeof(*token_ids));
		struct chat_template_spans trusted = { 0 };
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
				 chat_template_render_spans(
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
		storage_operations = storage_for_path(copy_paths[0]);
	/* auto opens the direct descriptors (DIRECT_DROPPED) and falls back to DROPPED when no replica is found below */
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
	if (!failed && turbo_load != 0 && copy_count == 1)
		failed = step_failed("find the replicas", replicas_add());
	if (map_reads == MAP_READS_AUTO && copy_count == 1)
		storage_exl3_read_mode(EXL3_READS_DROPPED);
	if (!failed && is_generating) {
		model = model_operations_of(pack_header_of(pack)->model_name);
		failed = step_failed(
				 "describe the model",
				 model == NULL ?
					 STATUS_UNSUPPORTED :
					 model->describe(pack,
							 &model_description)) ||
			 step_failed("match the tokenizer",
				     tokenizer_vocabulary_size(tokenizer) ==
						     model_description
							     .vocabulary_size ?
					     STATUS_OK :
					     STATUS_INVALID_ARGUMENT);
	}
	const struct model_limits limits = {
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
	if (!failed) {
		const uint64_t bytes_kept_free =
			(uint64_t)card_memory_kept_free_mib * BYTES_PER_MIB;
		const uint32_t unit_count = pack_header_of(pack)->unit_count;
		const bool every_card_is_taken = card_count == 0;
		card_of_unit = calloc(unit_count, sizeof(*card_of_unit));
		status = backend_cuda_value.card_count(&cards_found);
		if (status == STATUS_OK &&
		    (cards_found == 0 || cards_found > BACKEND_MAXIMUM_CARDS)) {
			/* No card to load onto, or more cards than these arrays hold. */
			status = STATUS_UNSUPPORTED;
		}
		if (status == STATUS_OK && every_card_is_taken &&
		    execution_placement == EXECUTION_TP4_EXL3)
			status = exl3_card_order_prepare(cards_found);
		for (; status == STATUS_OK && every_card_is_taken &&
		       execution_placement != EXECUTION_TP4_EXL3 &&
		       card_count < cards_found;
		     card_count++) {
			card_order[card_count] = card_count;
		}
		for (uint32_t card = 0;
		     card < card_count && status == STATUS_OK; card++) {
			status =
				card_order[card] >= cards_found ?
					STATUS_INVALID_ARGUMENT /* no such card */
					:
					backend_cuda_value.card_describe(
						card_order[card],
						&descriptions[card]);
		}
		failed = step_failed("describe the cards", status);
		/* the PRO's own arithmetic unless --pro_emulation=1: the plan, programs and kernels follow the class */
		for (uint32_t card = 0; card < card_count && !failed; card++) {
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
		if (tensor_placement_selected()) {
			failed = failed ||
				 step_failed("plan TP2 shards and state",
					     tensor_plan_prepare(
						     &limits, bytes_kept_free,
						     card_of_unit));
		} else {
			/* Generate: subtract the previous placement's state, then place again. These budgets
			 * are replayed by tests/model_v41_placement_test.c; changes must be made there too. */
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
					status = model->state_byte_count(
						pack, &placement, &limits, card,
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
					 model->state_byte_count(
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
					 model->plan_create(pack, layout,
							    &limits, &plan)));
		}
	}
	if (!failed && tensor_model != NULL) {
		failed = step_failed(
			"load and qualify TP2 arenas",
			tensor_cards_load(front_end_files[TABLES_FILE],
					  front_end_byte_counts[TABLES_FILE]));
	}
	if (!failed && tensor_model == NULL) {
		const struct load_request request = {
			.storage = storage_operations,
			.reader = reader,
			.backend = &backend_cuda_value,
			.pack = pack,
			.layout = layout,
			.cards = cards,
			.card_descriptions = descriptions,
			.reader_threads_per_copy = reader_threads_per_copy,
			.stall_limit_nanoseconds =
				LOADER_DEFAULT_STALL_LIMIT_NANOSECONDS
		};
		/* Open receiving cards with an arena for their placed weights and model state. */
		for (uint32_t card = 0;
		     card < card_count && status == STATUS_OK; card++) {
			uint64_t arena_bytes = layout->card_byte_count[card];
			status = is_generating ?
					 model->arena_byte_count(plan, card,
								 &arena_bytes) :
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
				  front_end_pack_match(
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
			const uint32_t chunk_settings[] = { wide_rows,
							    full_steps,
							    prefill_cone,
							    chunks_in_flight,
							    prefill_margin,
							    prefill_fast_plan };
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
				arithmetic_salt = (arithmetic_salt ^ CHUNK_LADDER_TOP) * SALT_PRIME;
			if (prefill_ladder_1536 != 0)
				arithmetic_salt = (arithmetic_salt ^ LADDER_1536_ROWS) * SALT_PRIME;
			if (prefill_ladder_fine != 0)
				arithmetic_salt = (arithmetic_salt ^ LADDER_FINE_COUNT) * SALT_PRIME;
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
		memcpy(engine_description.batch_step_us, batch_step_us, sizeof(batch_step_us));
		engine_description.prefill_first = prefill_first;
		engine_description.prefill_prompt_chunks =
			prefill_prompt_chunks;
		engine_description.draft_learned_acceptance = draft_learned_acceptance;
		engine_description.draft_session_acceptance = draft_session_acceptance;
		engine_description.draft_start_economic = draft_start_economic;
		engine_description.batch_plan_ties = batch_plan_ties;
		engine_description.prefix_cold_plan_qualified =
			prefix_cold_plan_is_qualified() ? 1u : 0u;
		failed = is_generating &&
			 (engine_failed_to_start(&engine_description) ||
			  stats_metadata_bound(card_of_unit) != STATUS_OK ||
			  (is_serving ? requests_failed(tokenizer) :
					prompt_failed_to_generate(
						token_ids, token_count)));
		failed = failed || (is_generating && !is_serving && dump_draft_ring != NULL &&
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
	for (uint32_t index = 0; model != NULL && index < program_count;
	     index++) {
		model->program_destroy(programs[index]);
		model->program_destroy(image_programs[index]);
	}
	engram_close(engram);
	server_close(server);
	if (model != NULL && tensor_model == NULL) {
		model->plan_destroy(plan);
	}
	const uint32_t retained_cards =
		tensor_model != NULL ? tensor_cards_drain() : 0;
	if (retained_cards != 0) {
		(void)fprintf(
			stderr,
			"TP2 peer arenas retained after an unproven group drain: mask=0x%" PRIx32
			"\n",
			retained_cards);
		failed = true;
	}
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
		if ((retained_cards >> card & 1u) == 0)
			backend_cuda_value.card_close(
				cards[card], CARD_WAIT_LIMIT_NANOSECONDS);
	}
	if (retained_cards == 0)
		v41_tp2_model_destroy(tensor_model);
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
	return failed ? PROGRAM_FAILED : PROGRAM_SUCCEEDED;
}

/* Keep expanded storage alive through shutdown: text options may refer to its bytes. */
int main(int argument_count, char **arguments)
{
	if (argument_count > 1 && strcmp(arguments[1], "help") == 0) {
		if (argument_count == 3 && strcmp(arguments[2], "axes") == 0)
			return help_axes(stdout);
		if ((argument_count == 3 || argument_count == 4) &&
		    strcmp(arguments[2], "knobs") == 0)
			return help_knobs(argument_count == 4 ? arguments[3] :
								"",
					  stdout);
		(void)fputs(
			"usage: anemone help axes | anemone help knobs [prefix]\n",
			stderr);
		return PROGRAM_WRONG_USAGE;
	}
	struct axis_arguments_record expanded = { 0 };
	if (!axis_expand(argument_count, arguments, &expanded)) {
		axis_arguments_free(&expanded);
		(void)fputs("axes: invalid selection; see anemone help axes\n",
			    stderr);
		return PROGRAM_WRONG_USAGE;
	}
	const int status =
		run_program(expanded.count, expanded.values, &expanded);
	axis_arguments_free(&expanded);
	return status;
}
