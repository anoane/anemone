<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Anemone knob catalog — Track S, 5 November 2026 (parser3d883e5 / main39980c7)

Parser/source base: engine_rnd_s merge `3d883e5` of verified main `39980c7` (tcb78); generated catalog and measured-table changes have separate commits. Numeric defaults come from the fresh Linux probe plus the documented placement-dependent resolution. The six P/Q cache controls are primary; defaults remain0/0/30/1024/256/120. Earlier O/HC/N/M integrations remain. S09/f139 frozen inputs retain their own epoch; source availability, numerical qualification and axis admission remain separate. Values/defaults for the numeric table were evaluated by a Linux
the recorded native host probe, rather than inferred usage prose.
The table has one row per option and command context, including auxiliary subcommands and separately
labelled track-only options. `anemone help knobs [prefix]` is generated from this file. The packet and
`engine_rnd_s/tools/cli/KNOBS.md` are identical. Regenerate help with `python3 tools/cli/generate_help.py`.

`same bits` means a candidate preserves the stated recipe's arithmetic, not that every value has passed both
references. A numerical selector is classified by its changed path, not its default. `precision up`, `sum order`,
`lossy`, and `draft-only` follow `gnmac_port/RND_PIPELINE.md`. EXL3 checkpoint fidelity and original DeepSeek
bitwise equivalence are different gates. Original exact is limited to the admitted serial M1 text/C1 recipe;
that scope does not qualify other original or EXL3 tuples.
Draft-policy, calibration, proposal and cost controls are draft-only only when the selected target
verification widths/forms preserve target bits. They have conditional sum-order consequences otherwise;
matching token IDs or text alone does not prove selected-logit preservation. See gate D and the controlled
S06 policy0/policy1 result below. A historical draft-only label carries this qualification.
For sampler/output/template controls, "sum order" conservatively marks changed output rather than an arithmetic
identity claim; they do not imply approximate storage. Non-numerical resource/I/O controls are same bits subject
to the usual workload/schedule boundary. Mixed/prefix scheduling may change a BF16 cold plan; identity needs proof.

Gate definitions (each row adds its specific conditions):

- G0: exact-case `--name=value` before positional files; strict numeric range/arity, no empty text, duplicates
  refused by main (replicate is the documented last-wins exception). Generate needs messages+tables+pack;
  serve needs tables+pack; load needs pack. 1..maximum readable copies and validated pack extents required.
  Numeric lists require distinct elements except batch_rows/batch_verifiers. Runtime allocation, card architecture,
  residency, prepared program shape and drain checks remain mandatory.
- T: TP switches affect the named tensor path only. TP4_EXL3 uses exactly four class-80 CMPs and the EXL3 map;
  original PP5/TP4_PRO/TP2 variants require the original pack and admitted card geometry. Inert switches do not
  force a recipe. TP admission requires generate/serve (load refuses), DSpark/hybrid (speculative>=2),
  pro_emulation=0, context>=16384, state16, reference CMP order, allocated draft_block5..7 and ordinary rows<=8.
  TP2 placements require relaxed/fast numerics. Context is a multiple16; overall TP row capacity is<=8 or>=64,
  and at most2048 with split overlap or1024 without. See `source/startup_policy.c`, commands.c startup_placement_status,
  tensor description and `model_v41_tp2_model.c` validation. Proposal depth is a separate1..7 cap.
- B: EXL3 BATCH requires TP4_EXL3, pipeline depth1 and a valid program/verifier/cost list. Multiple EXL3
  device-draft forms require batched draft rows. The distinct original_batch=1 pilot requires original TP4_PRO,
  exactly BATCH48/eight verifiers/eight slots/depth1 and every exclusion in its row; ordinary EXL3 BATCH knobs
  do not admit arbitrary original BATCH forms. Prepared geometry, verifier, workspace and engine gates remain
  authoritative. The original pilot's nonzero flag is not selected by current S axes.
- F: nonzero tails require TP4_EXL3, one chunk in flight, exact planner, full steps, wide>=1024, TC MoE and trunk
  MMA. Front/thin tails require a tail. Wide rows require a cone and supported geometry. Additional row-specific
  gates below supplement these shared gates; unrelated prefill switches need not enable tails.
- D: Target-bit evidence qualifier, not an additional raw CLI refusal. Proposal/policy/calibration/cost changes
  can alter actual target verification widths or dispatched forms.
  Classify them as draft-only only with target-bit preservation for those widths/forms and the stated recipe;
  otherwise retain conditional sum order and hold any same-bits balanced admission. Full selected-record gates
  are distinct from token-ID/text checks. S06's controlled policy0-to1 comparison changes six selected F32
  logits despite eight equal IDs and unchanged head flags; its exact target/provenance cause remains open.

Measurements retain their original binary, power, shape, prompt and recipe scope. A joint recipe gain is never
an isolated knob gain. Unmeasured means no isolated result was found in the reviewed RND logs/matrix/gap sources.
Benchmark methods and dated results are described in docs/results/COMPARISON_MATRIX.md. No speed is extrapolated to 1M.
All seven PLAN_0.1_ALPHA section5 buckets retain their accuracy/capacity gates; axis help maps each tuple.
Unsupported exact tuples (more than one session, TP4_PRO) remain held; original exact selects its admitted
serial M1 text/C1 recipe, and EXL3 exact the one-session exact-shaping set on quantized weights (not official-exact).
No fast axis tuple selects an unqualified lossy setting.

| Name | Values | Default | Meaning | Commands | Placement / recipe | Exactness | Gates | Interactions | Measured effect / source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| --version | standalone flag | absent | Print v0.0.5-alpha, author attribution and the existing license identifier, then exit without loading a model. | standalone | no model, weights or cards | product output only | Standard descriptors and SIGPIPE handling precede output. | Startup attribution uses stderr; generated-token stdout is unchanged. | GPU-free CLI output check; no inference path or numeric default changes. |
| --card_order | 0..15; comma list, at most16 | auto: all backend cards; TP4_EXL3 chooses CMPs only | Choose backend card ordinals in stage-fill order. | load, generate, serve | all placements/recipes | same bits | G0 | TP4_EXL3 auto-selects exactly four CMPs when omitted; explicit order must fit selected placement. | engine_rnd_s/source/main.c:757; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --metrics | 0 / 1 | 0 | Enable Prometheus GET /metrics with stats auth/ACL; anonymous without key_file. | serve | all admitted recipes | same arithmetic; native equivalence pending | host contract | Scrapes use fixed engine counter copies; histogram updates occur at request boundaries. | source/server_metrics.c; native overhead unqualified. |
| --live_stats | 0..60000 | 0 | Print server counters at this interval in milliseconds; zero disables them. | serve | all placements/recipes | same bits | G0 | Nonzero must be >=100ms. | engine_rnd_s/source/main.c:760; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --reader_threads_per_copy | 1..16 | 4 | Allocate this many asynchronous readers per pack copy. | load, generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | Real PP5 serve loads (15 Nov 2026, from the source and three replicas, A D D A): 4 readers loaded in 16.442 and 16.399 s against 16.854 and 17.207 s with 2 (+3.58 %, spread 2.07 %), every extent digest-checked; 2 is the earlier default. |
| --load_piece_mib | 0..64 | 0 (16 MiB) | Read the pack in pieces of this many MiB: the PP loader's piece and staging slot, the TP host arena's chunk; 0 is today's 16 MiB. The knob wins over a storage profile's load section, a profile over the default. | load, generate, serve | all placements/recipes (PP pieces and slots; TP host chunks; TP card extents stay whole) | same bits | G0 | Every extent's digest is checked as before; 8 staging slots a card of the piece are page-locked. | source/commands.c (load_piece_mib), source/loader.c, source/tp_load.c; real PP5 serve loads (15 Nov 2026): 64 MiB was no faster than 16 MiB at 2 readers a copy (16.901 against 16.916 and 16.967 s) and slower at 4 and 8 (17.840 against 16.448 s, 18.604 against 16.474 s), so the 16 MiB default stands; on TP4_PRO, the load stage's host candidate (3 readers a copy, 64 MiB) loaded 1.54 % slower than the defaults in real serve loads (A D D A), so TP keeps 16 MiB too. |
| --tp_load_threads | 0..64 | 32 | Load TP card weights concurrently; zero keeps the serial loader. | load, generate, serve | all TP placements; original or EXL3 | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:765; rh04: 32 workers 31.1 s versus 16 workers 33.9 s; same loaded bytes (RND_H.md). |
| --turbo_load | 0..1 | 1 | Discover and read receipted replicas beside the given copies, on drives holding none of them; 0: exactly the given copies. | load, generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | rh03 serve ready32.48s with replica versus47.93s single-copy and192.18s serial; receipts equal. Joint loader epoch, not an isolated storage benchmark (RND_H.md). Measured on 15 Nov 2026 with one binary: PP5 from PACK and one more copy, which turned the search off, loaded 2 copies in 29.63 s against 16.94 s from PACK alone with the search's 4; since 17 Nov the search runs beside given copies too. |
| --load_tuned_weights | 0..1 | 0 | With an enabled storage profile, let its per-copy load weights pick each piece's copy (a static split kept for comparison); 0 keeps the profile's reader count and copies and lets every copy's readers steal the next piece, so a faster drive reads more. | load, generate, serve | an enabled storage profile with load tuning | same bits | held; no axis selection | Ignored without an enabled profile, whose default load already steals. | C01 characterise (9 Nov): an equal split is paced by the slowest drive, 22.1 GB/s from three drives of 13.1/10.9/7.43 GB/s; the full-pack steal against split comparison is request 5. |
| --replica_external | 0,1 | 0 | Include external/removable block devices in replica eligibility; network/virtual filesystems never qualify. | load/generate/serve, replicate, autotune (every stage), storage-device/room/scan | Both startup and Engram replicas; identity and measured weights still required. | Same bytes | No implicit benchmark or copy; explicit override bypasses the saved policy. | replica_floor_percent, storage_profile | Policy only; no performance or axis promotion. |
| --replica_floor_percent | 0..99 | 20 | Minimum available filesystem fraction after copying, checked before allocation and publication. | load/generate/serve, replicate, autotune (no stage, --stage=copy), storage-room | Shared physical-device replica policy. | Same bytes | Insufficient space excludes this device; owned refused partial removed. | replica_external, storage_profile | Configurable policy; native placement gates remain held. |
| --storage_profile | absolute JSON path or off | automatic configured profile when present; new behavior disabled without enabled profile | Reload the profile once at startup: a schema 3 profile's qualified load and engram sections (anemone autotune --stage=load, --stage=engram), or an earlier schema 2 profile; never benchmark at startup. | load, generate, serve | source-matched physical local devices; held until native qualification | exact bytes intended | G0; owned bounded profile, hardware identity and SHA256 replica validation | Explicit action-family CLI settings override the corresponding profile section. Missing or invalid profile retains ordinary turbo-load and all-RAM defaults. | Host reload/crash/fallback tests prepared; native and aggregate performance gates pending. |
| --engram_sources | comma-separated flat pack paths | unset | Use independent immutable Engram table sources, with each source's own extent offsets. | generate, serve | PP/TP; original or EXL3 maps with explicit flat table sources | same bytes intended | held; no axis selection | Every hashing constant, table size and digest must match. Distinct admitted physical media required; source model weights are ignored. No copies created. | CPU source mapping and native qualification pending. |
| --engram_hot_set | file path | unset | Hold a static hot set of n-gram table blocks in RAM and serve it like the RAM prefix (the step plan and every gather path). | generate, serve | disk Engram only, with --engram_hot_sha256 | same bytes intended | held; no axis selection | The file (engram_io_bench hot_build, from a corpus that is not the served prompts) must be a regular file of this user, opened without following links, of this pack, its tables and their sizes, with sorted unique ids within --engram_hot_mib; its blocks are read and verified against their pack digests with copy fallback at startup. Refused with all RAM. | Bench A2 (9 Nov): 512 MiB of blocks profiled on other cases cut C1 p50 26 % (mean 40 %), C2 36 %, C4 31 %, C8 and prefill flat; native gates pending. |
| --engram_hot_sha256 | 64 hex digits | unset | The SHA-256 the operator or the storage profile recorded for --engram_hot_set; another file refuses. | generate, serve | with --engram_hot_set | same bytes intended | held; no axis selection | Required with --engram_hot_set and refused without it. | Binds the hot set file to its recorded digest. |
| --engram_hot_mib | 1..65536 | 512 | The most RAM the static hot set may hold, in MiB; a larger file refuses. | generate, serve | with --engram_hot_set | same bytes intended | held; no axis selection | Bounds the file length and the blocks held. | Bench A2: the gain saturates at 512 MiB. |
| --engram_ram_percent | auto or 0..100 | auto: 100 when the tables fit MemAvailable, else the largest whole share that fits, else 0 | Keep this percent of immutable Engram blocks resident; 50 is mixed, 0 is disk. auto compares the plan's host memory (tables, disk store, other host-only units, pinned host experts, load staging, headroom max(8 GiB, MemTotal / 32)) with MemAvailable before the load and prints it on two lines; a given percent or a storage profile's is kept, its fit printed. | generate, serve | PP/TP loaders; explicit flat table sources for EXL3 maps | same bytes intended | held; no axis selection (a capacity decision: outputs are the same at every percent) | Disk placement requires admitted local media; auto below 100 without an admitted disk copy, or a model whose own host memory exceeds MemAvailable, is refused before the load. Draft history and constants stay in RAM. | CPU byte/failure qualification; native numerical and performance gates pending. On the reference workstation the tables fit: auto is 100. |
| --engram_fit_only | 0..1 | 0 | Print the Engram fit (the percent with its source, and the budget) and stop before any card is touched; disk tables it chose are prepared first (their sources admitted and scanned). | generate | all placements/recipes | product output only | diagnostic; no axis selection | Opens the pack, the n-gram constants and any chosen disk tables; exits 0 after the lines, 1 for a refused fit or disk tables that cannot be prepared. | Host diagnostic of the RAM-fit decision; no inference path. |
| --engram_cache_mib | 0..32768 | 0 | Bound the disk store hot cache in MiB, rounded down to 16 blocks. | generate, serve | disk Engram only | same bytes intended | held; no axis selection | Cache buffers and metadata are fixed before serving. Nonzero with all RAM refuses. | Component only; native gates pending. |
| --engram_io | 0..1 | 0 | Select serial direct reads or prepared io_uring batches of 32 blocks. | generate, serve | flat table sources on admitted media | same bytes intended | held; no axis selection | io_uring must be available with verified direct alignment; native cost unqualified. | CPU random-gather microbenchmark supports batched direct reads; paired native gate pending. |
| --turbo-nvme-engrams | 0..1 | 0 | Interleave runtime block reads over distinct physical table sources. | generate, serve | disk Engram with at least two admitted table sources | same bytes intended | held; no axis selection | Uses startup replicas or explicit independent table sources; no copies are created during serve. Nonzero with all RAM refuses. | Component byte/failure evidence only; multi-device native benefit unqualified. |
| --engram_step_io | 0..1 | 1 serving PP5 or TP4_PRO from disk tables; else 0 | Read a step's n-gram blocks at once when its row ids are known (one deduplicated submission, on I/O threads per copy that verify each block's digest, fall back to another copy and scatter rows); 0 keeps the per-row gathers. Serving PP5 or TP4_PRO from the disk store it is the default (hybrid: spin 200 us, then sleep; pinned; two threads a copy); every other GPU split and generate keep 0 until measured there. | generate, serve | disk Engram only | same bytes intended | held; no axis selection | Requires disk tables (refused with all RAM). Rows from the RAM prefix and cache are served at issue. A card's visit waits only for its own tables. | Bench (engram_io_bench A1, 9 Nov): decode C1 377->142 us, C8 1635->366 us, prefill chunk 324->45 ms against today's path; native gates pending. Serving (request 7 r2, PP5, ABBA, every stream identical to all-RAM): cold TTFT 27.38 s at 131k tokens against 27.18 s all-RAM and 47.85 s per-row; decode C1 172.2 against 176.0 tok/s (per-row 170.9), C8 equal; 0.78 cores. TP4_PRO (requests 9 and 9b r2, ABBA, every stream identical): decode C1 4.6-5.9 % over per-row (on/off -2.3..-3.6 % against -6.5..-8.9 %); cold TTFT 9.87 / 35.18 s at 8180 / 32782 tokens against 9.74 / 35.08 s all-RAM and 12.63 / 37.98 s per-row. |
| --engram_io_threads | 1..4 | 2 | I/O threads per table copy for step I/O. A knob wins over a storage profile's enabled engram section (anemone autotune --stage=engram), the section over the default, only while tables are read from disk. | generate, serve | disk Engram with --engram_step_io=1 | same bytes intended | G0; focus axis on every split, every value empty (the --load_piece_mib precedent): its measured value comes from the storage profile's engram section | Each thread owns one ring of its copy; more threads spread the per-read kernel cost. | Bench A1: 2 a drive best of 1/2 measured (A1b scales 1-4); native gates pending. The Engram stage measures it on the host's drives (anemone autotune --stage=engram); serving A/B found the step-I/O variants within noise in tok/s at C1-C8 and in TTFT. |
| --engram_spin_us | 0..100000 | 200 | Busy-poll window of a step-I/O thread after its last work, in microseconds, before it sleeps in the ring's wait. A knob wins over a storage profile's enabled engram section (anemone autotune --stage=engram), the section over the default, only while tables are read from disk. | generate, serve | disk Engram with --engram_step_io=1 | same bytes intended | G0; focus axis on every split, every value empty (the --load_piece_mib precedent): its measured value comes from the storage profile's engram section | Bounds the threads' CPU between steps; 0 sleeps at once. | Bench A1: spin and enter completion both measured; native cores per C level pending. The Engram stage measures it on the host's drives (anemone autotune --stage=engram); serving A/B found the step-I/O variants within noise in tok/s at C1-C8 and in TTFT. |
| --engram_io_pin | 0..1 | 1 | Pin each copy's first two step-I/O threads to the CPU that completes the drive's first I/O queue and its SMT sibling. A knob wins over a storage profile's enabled engram section (anemone autotune --stage=engram), the section over the default, only while tables are read from disk. | generate, serve | disk Engram with --engram_step_io=1 | same bytes intended | G0; focus axis on every split, every value empty (the --load_piece_mib precedent): its measured value comes from the storage profile's engram section | Read from /proc/interrupts and sysfs at start; a copy on no NVMe drive runs its threads unpinned. | Bench A1: pinned per-drive threads 152 us C1 against 212 us for one thread; native gates pending. The Engram stage measures it on the host's drives (anemone autotune --stage=engram); serving A/B found the step-I/O variants within noise in tok/s at C1-C8 and in TTFT. |
| --engram_scan_threads | 0..4 | 2 | Readers per copy of the startup digest scan of the disk Engram tables: every copy at once, each reader taking the next 16 MiB piece; when that fails, one copy at a time; 0 reads one copy after another with one reader. | generate, serve | disk Engram only | same bytes intended | held; no axis selection | Every block digest and the extent digest are computed as before (the extent digest is a sum over pieces); no unverified RAM is published. | Bench option A (9 Nov): the parallel scan of 209.7 GB in 8.5 s (24.7 GB/s) against about 30 s for one reader on copy 0; turbo launch-to-ready 60.5 s against 48.5 s all-RAM in serving (request 4). |
| --map_reads | 0 buffered;1 drop;2 direct;3 direct+drop;4 auto | 4 (auto:3 with replicas, else1) | Choose buffered/direct EXL3-map reads and page-cache eviction policy. | load, generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:768; rh01: modes 0/1/2 TP4 load 36.7..38.4/29.8..31.5/32.3 s; rh08 single-copy mode3 62.5 s versus mode1 45.9 s (RND_H.md). |
| --tp_load_host_threads | 0..64 | 16;0 inherits TP loaders | Load the host arena with this many workers; zero follows TP load threads. | load, generate, serve | all TP placements; original or EXL3 | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:770; rh03:32 host workers34.616s,8 workers34.780s; rh04 chooses32 card/16 host workers after measured sweep; receipts equal (RND_H.md). |
| --tp_load_copy_split | 0..2 | 2 | Choose each TP load read's copy. 2: host chunks prefer copy zero and device extents the last copy, and a role that has claimed its last read leaves its copy to the other (a middle copy serves both); 1: each role only its preferred copy (tcb96's default, for comparison); 0: every read from the copy soonest to finish it. | load, generate, serve | all TP placements; original or EXL3 | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Every extent is still digest-checked with copy fallback; a storage profile's load list uses 0. | rh06/rh08 with replica reads: fixed roles 29.7 versus 31.0 s soonest; joint policy, no isolated speed claim (RND_H.md). Request 16 (TP4_EXL3, fixed roles): copy 1 idle 9.3 s while copy 0 read on; the stealing default is measured in request 17. |
| --tp_load_reads_per_copy | 0..64 | 0 (unlimited) | Limit TP in-flight reads per copy; zero is unlimited. | load, generate, serve | all TP placements; original or EXL3 | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:774; rh03:limit4 36.245s,limit8 35.854s; rh01 limit2 79.957s in older epoch. No selected limit improves retained unlimited default (RND_H.md). |
| --card_memory_kept_free_mib | 0..1048576 | 0 | Subtract this many MiB from each card's placement budget. | load, generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:776; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --pro_emulation | 0..1 | 0 | Emulate class-80 sums on the PRO; zero uses class-120 native arithmetic. | load, generate, serve | original PP5 PRO layers; nonzero refused on TP | sum order | G0 | Native PRO outputs depend on layers placed there; emulation costs2..4x on affected PP5 PRO products. Nonzero is refused by every current TP startup path (MEMORY.md; commands.c startup_placement_status). | engine_rnd_s/source/main.c:778; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_code_wire | 0..1 | 0 | Transport original quantized codes/scales losslessly across TP ranks. | load, generate, serve | original TP4_PRO/TP2 variants | same bits | G0; T | Refused for TP4_EXL3. Original-weight TP paths only. | engine_rnd_s/source/main.c:779; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --exl3_expert_parts | 4 or8 | 4 | Choose four reference or eight alternate expert product parts. | load, generate, serve | TP4_EXL3 THIN/BATCH | sum order | G0; T | Only4 or8;8 requires TP4_EXL3, numerics relaxed/fast, reference CMP order and FP32 wire. | engine_rnd_s/source/main.c:780; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_attention_owner | 0..1 | 0 | Keep experimental TP4 attention state on its owning rank. | load, generate, serve | exact original TP4_PRO, FP32 wire | same bits | G0; T | Nonzero requires original TP4_PRO, numerics=exact and FP32 wire; other placements refuse it at startup. | engine_rnd_s/source/main.c:783; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_head_shards | 0..1 | 0 | Shard target head products and sampling over TP4_EXL3 ranks. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits for an unchanged product form; sum order when a different head form is activated | G0; T | Nonzero requires TP4_EXL3; enables tp_draft_shards. Head-MMA activation depends on the combined head strategy. Canonical THIN uses head_shards1/head_stream1/draft_shards1 and draft_layers0. S06 holds these head flags equal and changes only draft_policy0 to1: all8 IDs agree but6 selected F32 logits differ (max about7.63e-5). Policy1 MD5ff1489cffe4194bb52fe36eef08056c9 also equals S04gxf15's older unsharded policy1 file. Neither comparison isolates the head or sharding. N confirms the stored field is the target sampler global_logit; THIN1/THIN8 and small dense-stream/eligible multirow head-MMA dispatch are concrete possible boundaries, not attribution without matching committed-token operands/kernel/arguments. Earlier target widths remain possible causes. This is not a full-vocabulary comparison or an established invariant balanced-runtime form. Current policy1 speed is unmeasured (RND_S04:38; RND_R05:12; RND_N05:19,5Nov). | engine_rnd_s/source/main.c:785; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_exact_dense | 0..1 | 1 | Use exact output-sharded attention dense products on the FP32 EXL3 recipe. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits (FP32 exact recipe) | G0; T | Active only TP4_EXL3 with FP32 wire; ignored on BF16/other placements. | engine_rnd_s/source/main.c:786; xdense01: same-bits exact dense shards; isolated whole-workload speed unmeasured (RND_D.md). |
| --tp_draft_shards | 0..1 | 0 | Shard draft head, Markov projection and sampler with the target head. | load, generate, serve | TP4_EXL3 THIN/BATCH | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Requires tp_head_shards=1. | engine_rnd_s/source/main.c:787; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_draft_layers | 0..1 | 0 | Tensor-parallelize the draft model's three layers. | load, generate, serve | TP4_EXL3 THIN/BATCH | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Requires TP4_EXL3, BF16 wire, draft shards and replicated_embedding; BATCH additionally draft_rows and wide_head. | engine_rnd_s/source/main.c:789; Measured S06/d71cd05 trueTHIN at65536, changing only layers0->1 with wide2048/pipeline2: code230.959->258.683(+12.0%), prose158.954->163.601(+2.9%), structured330.108->335.422(+1.6%)tok/s; selected8/256 and15MiaAI signatures pass. Cold3909->3904tok/s; Fraser median1379.40->1385.25ms(+5.85), worst1395.10->1390.00ms(-5.10). Strict zero-margin median gate fails; no balanced promotion or1M extrapolation. FreshTHIN mixed112self/96cross signatures pass; native decode during incomplete prefill is not observed. S10 same-source pair scheduled. Earlier probe07 5-10%C4/C8 andengab21 20.54->20.46ms remain different joint epochs (RND_D.md/GAP_ANALYSIS.md; S evidence/gxf_s06_single_tp_draft_decode_summary_v1.json). S xf18 matched frozenS10/e4c188e trueTHIN2048/pipeline2 at65K: layers0->1 code230.347->258.344tok/s(+12.15%),prose158.591->163.744,structured332.793->338.442; Fraser median/worst1381.35/1390.70->1384.35/1393.90ms, strict no-worse FAIL. Full selected8/256 and15MiaAI signatures per arm pass; minfree3181->823MiB. Only code/balanced/single/decode selects1; balanced focus keeps0. No layer1 1M fit or full-logit/all-width identity inference (evidence/gxf_s10_single_layers0_strict_xf18_v1.json; layers1 counterpart). |
| --tp_head_stream | 0..1 | 0 | Use the dense-stream target head products. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits for an unchanged product form; sum order when a different head form is activated | G0; T | Nonzero requires TP4_EXL3. Head-MMA activation depends on the combined head strategy. Canonical THIN uses head_shards1/head_stream1/draft_shards1 and draft_layers0. S06 holds these head flags equal and changes only draft_policy0 to1: all8 IDs agree but6 selected F32 logits differ (max about7.63e-5). Policy1 MD5ff1489cffe4194bb52fe36eef08056c9 also equals S04gxf15's older unsharded policy1 file. Neither comparison isolates the head or sharding. N confirms the stored field is the target sampler global_logit; THIN1/THIN8 and small dense-stream/eligible multirow head-MMA dispatch are concrete possible boundaries, not attribution without matching committed-token operands/kernel/arguments. Earlier target widths remain possible causes. This is not a full-vocabulary comparison or an established invariant balanced-runtime form. Current policy1 speed is unmeasured (RND_S04:38; RND_R05:12; RND_N05:19,5Nov). | engine_rnd_s/source/main.c:791; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --exl3_fast_products | 0..1 | 1 | Use wide-kernel EXL3 products instead of the retained chain. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:792; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --exl3_wide_entry | 0..1 | 1 | Run each chunk's EXL3 experts once over the full chunk. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:794; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --exl3_tc_moe | 0..1 | 0 | Use GNmac-style tensor-core EXL3 routed MoE products. | load, generate, serve | TP4_EXL3 THIN/BATCH | precision up (hi/lo operands); changed sums | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:796; tcmoe02 matches GNmac/Curtis at6/12/36/43/48/64 rows;48-row I6401490 versus1544us. engab03 joint TC-MoE+head:163.4 versus138.6 tok/s; only11/256 target tokens agree with old recipe (GAP_ANALYSIS.md). |
| --exl3_tc_product_blocks | 0..592 | 0 (retained148-block grid) | Override the two EXL3 TC-MoE product grids in THIN/BATCH programs. | load, generate, serve | effective EXL3 TC-MoE THIN/BATCH product path; FULL/WIDE unchanged; no extra placement gate | same bits | G0; T | Parser accepts0..592 for load/generate/serve, with no extra placement gate. Zero retains the existing148-block grid; positive values replace only gate/up and down grid[0] in eligible EXL3 TC-MoE THIN/BATCH programs. FULL/WIDE grids and each item's arithmetic/owners/folds remain unchanged; inert outside that prepared path. Same-bits geometry control. Current axes leave0; nonzero is a raw tuning override, not an automatically selected faster form (S main.c:751; model_v41_program.c:2935). | engine_rnd_s/source/main.c:797; N xd06 fixed5 BATCH8/one-verifier pilot at 150 W: grids148/74/222/296 give structured318.94/279.34/322.29/323.94, code245.93/212.46/245.48/247.53 and prose156.35/134.60/156.36/157.53 tok/s. All400/95/400 selected records agree. This is an inherited pilot, not a native180W/full-serving or fresh S measurement. main 242e6ef merges the selector with default0 (retained148-block grid); no axis selects a nonzero override (packet RND_N.md 4 Nov19:56; source model_v41_program.c:2935). |
| --exl3_expert_rotation | 0 fixed chunks;1 four-way rotation;2 three-way rotation | 0 | Rotate per-expert TC chunks across four ranks or three non-head ranks. | load, generate, serve | TP4_EXL3 THIN/BATCH | sum order | G0; T | Nonzero requires TP4_EXL3, BF16 wire, TC MoE and four expert parts. | engine_rnd_s/source/main.c:799; rb02 rotation2 about1%30K gain with changed numerics; rotation1 OOM on old57.9GB/card recipe. Not a current fit guarantee (RND_B.md/RND_E.md). |
| --exl3_unrated_map | 0..1 | 0 (refused: a map whose descriptor records no quality, or with no descriptor and an unlisted pack) | Serve an EXL3 map with no quality record (its descriptor's quality none, or no descriptor and a pack the model does not list as established), with no quality claim. | load, generate, serve | any pack of EXL3 matrices | same bits | G0 | Selection, not a measured option: no axis maps it. The map's descriptor (<map>.descriptor.json beside its real file, tools/convert/make_virtual_pack.py) is read and checked against the open pack, the model and the tables before anything loads; quality none is refused, naming the map, unless 1; established and recorded serve by default (a recorded capture's sha256 is checked). With no descriptor, what the open pack carries decides, whatever path or reader opened it: a pack whose header digest and size the model lists as established (V41_DATA_ESTABLISHED_EXL3_PACKS: TP4_EXL3's 2.9 bpw map) serves by default, any other is refused, naming its digest, unless 1. Inert for an original pack. | source/exl3_descriptor.c, source/commands.c map_descriptor_check, source/model_v41.h; no effect on the arithmetic. |
| --tp_head_mma | 0..1 | 0 | Use tensor-core target-head products on the BF16 recipe. | load, generate, serve | TP4_EXL3 THIN/BATCH | sum order | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:801; head01/02:48-row443 versus dense_stream4642us, relative difference<=1e-5; split-K head8 284..291us at1/6/8 rows. Changes old target numerics; not a same-bits toggle (GAP_ANALYSIS.md). |
| --tp_trunk_mma | 0..1 | 0 | Use tensor-core BF16-row trunk products on the BF16 recipe. | load, generate, serve | TP4_EXL3 THIN/BATCH | sum order | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:802; relbench02 six-row wq_a16.38 versus reference21.50us; BATCH12 wq_a95.2 ->23.6us. Fixed alternate sums; old-target equality not established (GAP_ANALYSIS.md). |
| --tp_trunk_mma_fused | 0..1 | 0 | Sum trunk MMA split partials inside the 64-row kernel. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:803; rj03p BATCH48: 38.44 -> 42.98 ms (+11.8%, slower), same bits (RND_J.md). |
| --tp_trunk_mma_thin_splits | 0..1 | 0 | Give 64-row trunk MMA products the THIN split geometry. | load, generate, serve | TP4_EXL3 THIN/BATCH | sum order | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:805; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_router_mma | 0 dense;1 BATCH router MMA;2 also prefill routers | 0 | Use split-K tensor-core BATCH router products (two alternate modes). | load, generate, serve | TP4_EXL3 THIN/BATCH | sum order | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:807; dense02 48-row router60.4 ->25.6us, maxscore-relative error6e-7 (other sum order); rb02 mode2 about1% 30K gain in old joint recipe (GAP_ANALYSIS.md). |
| --tp_gather_balance | 0 off;1 all chunk gathers;2 wide only;3 weighted by rank width | 0 | Assign chunk index/router work to narrower expert ranks (four policies). | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:808; rc04 mode1 Fraser1.857 ->1.914s (+3.1%, slower), IDENTICAL; ri05 mode3 30K8.488 ->8.598s (+110ms), identical (RND_C.md/RND_I.md). |
| --tp_pool_chunk | 0..1 | 1 | Prepare pooled entries/keys before chunk-slab scans. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:810; rc02ab 30K prefill 9.828 -> 9.430 s; same bits (RND_C.md). |
| --tp_pool_gather | 0..1 | 1 | Partition pooled-source compressor rows over ranks and gather results. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:811; ri01 same bits; standalone speed unmeasured (RND_I.md). |
| --tp_router_first_whole | 0..1 | 0 | Compute first-half router rows on every rank without gathering. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:812; ri01 with joint/dead/pool:30K +54ms, Fraser neutral; identical, kept off (RND_I.md). |
| --exl3_tcp_two_blocks | 0..1 | 1 | Launch prefill TC MoE products at two blocks per multiprocessor. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:814; ri03 2048-row product call -15%, Fraser first token -17 ms; same bits (RND_I.md). |
| --tp_queries_piece | 0..1 | 1 | Project wide-program queries by piece alongside wq_b. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:816; ri05 30K prompt -77 ms; same bits (RND_I.md). |
| --tp_index_span | 0..1 | 1 | Scan/select a span of chunk index rows per launch. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:818; ri05 30K prompt -51 ms; same bits (RND_I.md). |
| --tp_core_compact_halves | 0..1 | 1 | Write split-half core results into compact wo_a rows. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:819; ri06 -9/-11 ms on prompt controls; same bits (RND_I.md). |
| --dense_skinny_bf16 | 0 retain default;1/4/8 outputs per block (other CLI-range values fail preparation) | 0 (retain measured wide1 output/block) | Tile BF16 skinny dense outputs per block above 48 rows; zero retains wide form. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:821; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --dense_skinny_f32 | 0 retain default;1/4/8 outputs per block (other CLI-range values fail preparation) | 0 (retain measured skinny4 outputs/block above48 rows) | Tile FP32-weight skinny dense outputs per block above 48 rows. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:823; ri05 mode1:30K8.488 ->8.694s (+206ms, slower), Fraser1.459s; identical (RND_I.md). |
| --dense_staged | 0,8,16,32,64 | 16 | Stage wide dense-product inputs with a selectable output-block width. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Admits0/8/16/32/64; nonzero must be a supported power of two. M leaves default16; wider standalone wins do not qualify a whole-model promotion. | engine_rnd_s/source/main.c:825; rk05 2K prompt1.175 ->1.150 s; Fraser1.393 ->1.385 s, 30K equal; same bits (RND_K.md). |
| --original_dense_staged | parser0..16; admitted0 or16 | 0 | Stage original TP4 BF16/FP32 dense products. | load, generate, serve | original TP4_PRO CMP wide dense products | same bits | G0; T | Parser accepts0..16, but nonzero preparation admits only16 and placement=tp4_pro. There is no additional exact-only numerics gate; eligible BF16 input, BF16/F32 output, size/row-scale and >=64-row dense restrictions remain. Ineligible shapes retain or refuse through the named preparation path (S main.c:1224; dense_linear.c). | engine_rnd_s/source/main.c:827; Merged default0. Inherited XO04 eight dense shapes are byte-identical. XO05 selected/digest/canonical-state continuity passes Fraser/2K/30K; strict nine-request cold controls all cached0 give staged967 versus TP965 tok/s. Older962/965 values were cold-HOLD. No material whole-model gain, fresh S speed result or automatic prefill/ttft axis promotion (original_axis_handoff_v2.md:77,101; RND_O.md). |
| --original_head_offload | 0..2 | 0 | Target-only exact BF16 vocabulary cuts on four CMPs, F32 gather and unchanged native PRO sampler. | generate, serve | Original native PP5 8/7/7/7/11 on four CMP170HX and RTX PRO6000 Blackwell | Mode2 scoped bitwise equality to native-PRO balanced; strict original exact unqualified | Mode2 is a qualified raw alternative, not selected by an axis after TP4_PRO at pipeline depth 2 with 1 serving slot wins the C1 decode comparison; rebased binary merge gate pending | 0 retains the native PRO head. With 1 or 2 the serve's session fit counts the offloaded head's arenas (its CMP shards and, with 2, a flight bank per pipeline step), so max and an explicit count are what opens. 1 uses shared planes and a PRO visit lease. 2 uses immutable flight banks and owner-scoped PRO ordering, including headless prompts; pipeline depth bounds bank storage. Refuses TP/EXL3, emulated PRO, official recipes and other placement. Each CMP adds330956800 HEAD bytes plus bounded planes/control; FULL/WIDE retain their prefix. | Banked mode2 has conditional full-vector/state and paired native C1 evidence across all three text genres; C2 and higher concurrency regress and are excluded. Prefill is unmeasured. Mode1 is retained without a new run and has no automatic selection. Defaults remain0; explicit0 restores the PRO head. The current head path includes shared immutable programs/graphs; its native output/resource scope is recorded separately. The rebased binary requires default/canonical and C1 integration confirmation. source/model_v41_head.c, source/model_v41_head_banks.c and source/step_flight_programs.c. |
| --original_wide_experts | 0..1 | 1 for compatible TP4_PRO recipes; 0 elsewhere | Run original TP4 expert jobs over complete chunks with lossless slab gathers. | load, generate, serve | original exact TP4_PRO CMP wide experts; native PRO retained | same bits in recorded current-recipe fixtures; joins unqualified | G0; T | Nonzero requires TP4_PRO exact, native PRO, precision/reference linear and CMP-reference, FP32 wire, wide64..1024, code_wire0, attention_owner0, no BATCH, exl3_wide_entry0 and split/full-split0. Model additionally requires quad exact-output shards, no EXL3 TC-MoE and compatible state/row geometry. Separate core/overlap combinations need their own gates (S main.c:1236; model_v41_tp2_model.c:98). | engine_rnd_s/source/main.c:829; Measured O source75 raw/alias both-parent gate now passes: cold-prefill median967->1363tok/s(+40.95%); Fraser C1/C4/C8 medianTTFT4417.75/17609.85/35087.25->3197.60/12706.85/25290.30ms(-27.62/-27.84/-27.92%); aggregate end-to-end throughput improves9.63/9.72/9.69%, while C8per-stream rate falls0.48%. All46fullFraser responses plus9fixedAPI outputs/counts/finishes agree; fixed-input GEN00/10 preserve selected records,904960activeF32 logits and canonical state. Cold nonce calls have small prompt-count differences, not identical input IDs. NativeV1windows remain exit1; independent repair-aware whole-proof equality/alias lineage passes, first50sCPUreader attempt failed then unchanged120sallowance passed. Not currentS10/official/overlap+wide/mixed/1M proof. Existing raw default0/no S axis; conditional original/balanced/single-or-multiple/prefill-or-ttft mapping awaits final-main/default-resolution gates (O REPORT_v3.md; evidence/o3_tp_wide_both_parent_root02.json SHA0da4dae92d6e9a72ad6b1cadc26f5d34a2eec4ea2ea6b337bdc70dcff2a6df38). Historical GEN06 pilot remains in RND_O.md. Integration batch selects this default only within existing admission, with chunk_core0; explicit0 retains the slab path. |
| --tp_parallel_launch | 0..1 | 0 | Launch each TP card's graph groups from a separate host thread. | load, generate, serve | all TP placements; PP ignores this tensor control | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:831; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_graph_parts | 1..8 | 1 | Partition each card's graph behind the prefill cone. | load, generate, serve | all TP placements; PP ignores this tensor control | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:833; ri05 parts8:30K8.488 ->8.871s (+383ms, slower), identical (RND_I.md). |
| --tp_ordered_wide | 0..1 | 0 | Use ordered-wide attention trunk products for EXL3 prefill. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:835; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_graph_branches | 0..2 | 1 | Branch shared-expert work beside routed experts; mode two includes BATCH. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:837; engab18 C1 round -0.34 ms; same bits (RND_E.md). |
| --tp_piece_rings | 0..1 | 0 | Exchange BF16 wide results over 64-row pieces instead of eight-row slabs. | load, generate, serve | TP4_EXL3 THIN/BATCH | sum order; same bits scoped to column-owner recipe | G0; T | Changes ring ownership/order when tp_ring_column_owners=0; retain column_owners=1 for the same-bits recipe. BF16 TP exchange only; no isolated performance claim. | engine_rnd_s/source/main.c:839; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_trunk_shape_slices | 0..1 | 1 | Use the measured slice count per THIN trunk shard shape; zero forces four. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:840; trunk01 ordered chains maxdiff0 at1..16 slices; engab02 same tokens,255token2.032 ->1.975s (+2.9% rate). Six-row wkv25.6 ->15.4us (GAP_ANALYSIS.md). |
| --tp_bf16_slab_rings | 0..1 | 0 | Use BF16 slab-ring attention sums instead of whole-row FP32 accumulation. | load, generate, serve | TP4_EXL3 THIN/BATCH | lossy (BF16 intermediate sums); sum order | G0; T | BF16 rings only; inert with FP32 wire. BATCH/thin-tail forms already retain slab-ring completion. For wider attention, enabling this changes intermediate rounding versus FP32 whole-row accumulation (model_v41_program.c:1960). | engine_rnd_s/source/main.c:842; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_wide_exchange | 0 legacy;1 push;2 owner sums;3 owner rings;4 CE owner sums;5 CE rings | 5 | Choose wide transport: push, slab rings, owner sums or copy-engine rings. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:844; pf19 CE owner sums/overlap: 30K prefill19.08 ->14.80 s; joint recipe (GAP_ANALYSIS.md). |
| --tp_split_overlap | 0 off;1 overlap;2 serial;3 attention;4 FFN;5 pipeline;6 joint;7 joint down | 7 | Overlap layer halves and wide exchange with a chosen split schedule. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:846; ri01 same bits; earlier mode6 rb02 12.370 ->12.057 s (RND_I.md). |
| --tp_split_least_rows | 0..2048 | 0 | Run chunks below this row count whole instead of split. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:848; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_launch_trims | 0..1 | 1 | Remove redundant ring-input copies and wo_b quantization launches. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:850; rd04-series BATCH12 launches/card2112 ->1906 across retained trim package; reference identity qualified before windows; no isolated percentage inferred (RND_D.md). |
| --tp_full_split_overlap | 0 inherit;1..7 as tp_split_overlap | 0 | Override full-program split overlap; zero inherits tp_split_overlap. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:852; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_chunk_core | 0..1 | 1 | Run a chunk's attention core over the whole chunk. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:854; pf16 30K prefill18.93 ->17.80 s; same bits (GAP_ANALYSIS.md). |
| --original_chunk_core | 0..1 | 0 | Use original TP4 full-head attention core on whole eight-row slabs. | load, generate, serve | original exact TP4_PRO CMP full-head chunk core; native PRO retained | same bits in recorded current-recipe fixtures; joins unqualified | G0; T | Nonzero requires TP4_PRO exact/native PRO/precision/reference/CMP-reference/FP32, wide64..1024 divisible by8, code_wire0, attention_owner0, no BATCH/original_batch/wide experts and split/full-split0. Effective only on eligible CMP original full-head chunks with multiple slabs, one piece, no vision, state16 and the retained prefix topology; other core schedules remain separate (S main.c:1272; model_v41_program.c:6056). | engine_rnd_s/source/main.c:855; Merged default0. Inherited source8a434dc/GEN03 completes3 Fraser/2K/30K knob0/1 pairs and3 XO05 continuity controls with power restoration. Selected records, prompt IDs, digests, canonical SAVE and active last-target full-vocabulary F32 bits agree. Single-arm CLI prompt+first diagnostics improve about2%, not HTTP TTFT, fresh S serving or a core/query/wide join. No active S axis selects nonzero; no official-reference claim (RND_O.md 5 Nov02:32,02:42). |
| --original_batch | 0..1 | 0 | Admit the fixed original TP4 BATCH48/eight-verifier/eight-slot pilot. | serve | original exact TP4_PRO BATCH48, eight slots/verifiers, pipeline1 | same bits in recorded current-recipe scope; all-step scope unqualified | G0; T; B | Serve-only. Nonzero requires TP4_PRO exact/native PRO/precision/reference/CMP-reference/FP32, exactly one batch_rows48/verifiers8 entry, session_slots8, pipeline1, wide64..1024, DSpark/hybrid block5/positions5. Requires context_slabs1, core_once0, draft_wide0. Refuses code/owner/head/draft shards, draft layers, head streaming/MMA, trunk/router MMA, wide_head, draft_rows, BF16 draft variants, original wide/staged, EXL3 entry, split/full-split and pipeline overlap. Defaults are not rewritten to satisfy this pilot (S main.c:1246; model_v41_tp2_model.c:95). | engine_rnd_s/source/main.c:857; Merged default0. Inherited XO08 target/GEN preservation and XO09 source434a895/rndo_05 serving pass. Six eligible architecture windows each have12 cells and141 request/text/native-count comparisons; BATCH code C1/C2/C4/C8=76.97/119.64/214.25/347.99 tok/s. BATCH C8 beats TP overlap304.48 but loses to PP5 at pipeline depth 8 with 8 serving slots at 539.36. Eight concurrent requests are not an all-eight-owner STEP logits/state or mixed-rollback proof. PP5 depth1 remains a preserved rejection; campaign complete=false with independently valid subsets. No active S axis selects the pilot; original balanced-exactness multiple/decode remains the inherited candidate scope (RND_O.md 5 Nov01:25,01:42). |
| --tp_replicated_embedding | 0..1 | 1 | Embed input rows on every rank instead of handing off rank zero. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:859; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --batch_rows | 1..64; comma list, at most8 | none (THIN) | Prepare BATCH programs at these row capacities instead of THIN programs. | generate, serve | EXL3 BATCH or the separately constrained original_batch pilot | same bits | G0; T; B | Requires pipeline_steps=1 and either TP4_EXL3 or the separately constrained original_batch1 TP4_PRO pilot. Multiple EXL3 programs with device draft require batch_draft_rows=1; repeated row counts require explicit distinct verifier counts. Overall prepared capacity max(rows,wide_rows,batch_rows,draft_block) must be at least64; EXL3 BATCH programs may use1..64 rows. Original BATCH requires exactly48 rows/eight verifiers and all original_batch row gates; it is not arbitrary original BATCH admission. | engine_rnd_s/source/main.c:861; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --batch_draft_rows | 0..1 | 0 | Batch verifier draft rows into the BATCH program. | generate, serve | TP4_EXL3 BATCH | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; B; D | Nonzero requires batch_rows; with draft_shards it requires draft_layers; device draft layers also need wide_head. | engine_rnd_s/source/main.c:864; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --batch_wide_head | 0..1 | 0 | Use the wide target head for BATCH verification. | generate, serve | TP4_EXL3 BATCH | sum order; same bits with batch_thin_bits=1 in the retained recipe | G0; T; B | With tp_head_mma=1 and batch_thin_bits=0, wide and per-slab heads use different split geometry. Nonzero requires BATCH; no isolated speed or quality claim (model_v41_program.c). | engine_rnd_s/source/main.c:866; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --batch_context_slabs | 0..1 | 0 | Process BATCH draft context in THIN-sized eight-row slabs. | generate, serve | EXL3 BATCH or the separately constrained original_batch pilot | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; T; B; D | Nonzero requires batch_rows. | engine_rnd_s/source/main.c:868; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --batch_thin_bits | 0..1 | 1 | Use THIN numerical splits in BATCH trunk/head/TC-MoE products. | generate, serve | TP4_EXL3 BATCH | sum order | G0; T; B | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:870; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --batch_tc_paired | 0..64 | 0 | Pair TC MoE short forms from this many BATCH rows; zero disables pairing. | generate, serve | TP4_EXL3 BATCH | same bits | G0; T; B | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:872; rj01_card all compared forms identical; paired48 I640 +8.2us/call,I512 +18.4us; all tested sizes slower, kept off (RND_J.md). |
| --batch_draft_slots_head | 0..1 | 0 | Run one verifier's draft head across eight slots at each position. | generate, serve | TP4_EXL3 BATCH | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; B; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:875; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --batch_row_sets | 0,8,16,24 | 0 | Use trunk-product row sets above the grouped-row threshold. | generate, serve | TP4_EXL3 BATCH | same bits | G0; T; B | Must be a multiple of8; used above batch_grouped_rows. | engine_rnd_s/source/main.c:877; rj02s same bits; six-shape48-row total r16/r24/r8 284.7/298.0/330.8us versus192.5us _64+reduce; all slower, kept off (RND_J.md). |
| --batch_core_once | 0..3 | EXL3 CLI3 (active only for eligible BATCH); other placements2; explicit values win | Compute BATCH attention core once; mode2 adds window writes and mode3 pools/deferred slots. | generate, serve | TP4_EXL3 BATCH | same bits | G0; T; B | Accepts0..3. main 39980 omitted EXL3 CLI default3 is active on eligible BATCH; other placements retain2, API description0 is separate. Explicit S code/balanced/multiple/decode selects3; other recorded S BATCH recipes retain2. S10/e4c188e fixed/core3 versus RATE2/core2 full matrix and selected8/256 pass: C8code469.537/432.990, prose382.997/450.231, structured787.722/787.726tok/s. Multiple knobs differ, so these are recipe observations, not an isolated core gain; own CPU compilation overlaps part of decode. Same24 isolated mixed core2/3 text gates pass112cross/96self with code/prose C8task-geometric+0.529%/+1.115%. N actual26 code267.308/298.437/379.376/472.154 is cross-binary. Dedicated reference/safety proofs retain their epochs. See axes.json and RND_N/S.md. | engine_rnd_s/source/main.c:879; rj03p BATCH48 grouped48: 43.63 -> 43.02/42.41 ms at1/2 (-1.4/-2.8%), same bits; gsj06a 84/84 texts, 24..48-row programs -2.1..-4.3% (RND_J.md). |
| --tp_front_branch | 0..2 | 0 | Branch THIN attention mix coefficients; mode two also branches FFN coefficients. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:881; rj03p THIN C1 round 20.23 -> 19.94 ms (-1.4%), same bits; level2 20.12 ms (RND_J.md). S default remains0. |
| --tp_core_heads | 0,8,..64 | 0 | Limit attention-core heads per block; zero uses the measured rule. | load, generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Must be a multiple of8; zero chooses per-shape policy. | engine_rnd_s/source/main.c:883; rj03p THIN C1 20.23 -> 20.22 ms at8 (-0.05%, noise), same bits (RND_J.md). |
| --tp_decode_round_once | 0..1 | 0 | Sum published BF16 target partials in ascending-rank FP32 and round the result once. | load, generate, serve | TP4_EXL3 THIN/BATCH | precision up; changed target bits | G0; T | Nonzero startup requires TP4_EXL3, BF16 wire, tp_wide_exchange>=2 and wide_rows>8. Effective only on non-draft THIN/BATCH target completion, including prompt tails using those forms; draft and FULL/WIDE collectives retain old arithmetic. Preparation requires a BF16 four-rank ring, direct shared channels, BF16 payloads, no parallel publisher/local stub/host incoming routes; otherwise refuses. Enabled completion bypasses the old BF16 ring/CE completion paths, preserves READY/DONE sequencing, adds BF16 rank partials in ascending-rank FP32 and rounds once. Changes target bits; independent references required. Default0/current axes retain old sums; nonzero is a raw research option with no active exact/balanced/fast mapping (S main.c:1236; model_v41_program.c:1858). | engine_rnd_s/source/main.c:885; N xd09_v7 same24/core2, RATE2/session0/learn1/costs14300,0,690 three-pass C1/C8 control0/candidate1/repeat0 tests pass each arm's 72 concurrent plus15 C1 text/native-count/finish comparisons per genre and exact control repeat. Candidate prose differs from legacy; code/structured match in these fixtures. Structured rates334.877/782.980 ->269.850/584.648, code241.247/448.561 ->198.730/345.576, prose170.027/469.349 ->139.366/351.201 tok/s (C1/C8). Candidate is about18-19% slower at C1 and23-25% slower at C8. Full-vocabulary deviation/quality capture was deferred. main 242e6ef contains this research implementation, but default0 remains; nonzero has no active axis mapping or default/merge recommendation. Inherited N evidence is not a new S qualification (packet RND_N.md 5 Nov03:28; xd09_v7_budget/precision_serving_fair.json and control_repeat_fair.json). |
| --draft_bf16_rows | 0..1 | 0 | Use BF16 rows for draft attention products. | load, generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:887; gsj10/12 versus07/08: prose C1 +2.6%, C8 -1.7%; code C1 -0.65%, C2 -2.5%; draft-only, 126/126 texts (RND_J.md). |
| --batch_draft_wide | 0..3 | 3 | Widen draft work: one sums/posts, two products, three cores. | generate, serve | TP4_EXL3 BATCH | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; B; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:889; gsl05l3 level3 codeC8 641.6tok/s, proseC2/C4/C8 280.9/437.6/496.4; fixed-target texts retained; level3 became default after rl08 refs/window (RND_L.md). Joint epoch; isolated percentage not inferred. |
| --draft_bf16_operands | 0..1 | 0 | Use BF16 operands for class-80 draft quantized products. | load, generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:891; gsl03e serving proseC1 +0.1%, no BATCH win; THIN generate +5.9% in scoped probe, draft-only (RND_L.md). |
| --batch_grouped_rows | 0..48 | 16 | Use grouped THIN trunk forms through this BATCH row count. | generate, serve | TP4_EXL3 BATCH | same bits | G0; T; B | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:893; rj03p BATCH48 grouped16 38.44 versus grouped48 43.63 ms (-11.9%); same bits (RND_J.md). |
| --tp_ring_column_owners | 0..1 | 1 | Assign BF16 ring ownership by column quarter for batch-independent addition order. | load, generate, serve | TP4_EXL3 THIN/BATCH | sum order | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:896; mc_check2 refs since7f9c389: 30K41eeb9c16e10, Fraser f49e8badff47; different from old row-owner sums (RND_PIPELINE.md). |
| --batch_step_us | 0 off, or distinct positive costs1..4294967295; never mix0 into a list; comma list, at most8 | auto canonical menu only for the measured EXL3 family with policy/costs omitted; otherwise none; explicit0 disables | Supply per-program measured costs for the BATCH expected-token planner. | generate, serve | TP4_EXL3 BATCH | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; T; B; D | A sole0 explicitly disables the menu, including on unbatched shapes. Any multi-entry list containing0 refuses. Positive entries require batch_rows and exactly one distinct cost per BATCH program, ordered by row count then verifier count; duplicate prices refuse. With omitted policy/costs only, the measured EXL3 BF16/DSpark5/reference/core2-or3 family and exact canonical rows8,8,12,12,16,24,32,48/verifiers1,2,2,4,4,4,8,8 receives19500,21500,28600,30000,32000,39000,46000,58000. Explicit policy, custom/incomplete/reordered geometry or nonqualifying math gets no automatic menu. Explicit positive prices remain selectable; source/engine.c validates count. | engine_rnd_s/source/main.c:898; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --batch_verifiers | 1..8; comma list, at most8 | auto: rows/(draft_block+1), clamped to at least1 and available sessions/max8 | Supply each BATCH program's drafting-session capacity. | generate, serve | EXL3 BATCH or the separately constrained original_batch pilot | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; T; B; D | Nonempty requires batch_rows and an entry for every BATCH program; extra trailing entries are ignored. Each used count must fit session_slots and each BATCH row/verifier pair must be unique. Attachment requires an idle, unstarted BATCH driver (source/step_image_programs.c:177). | engine_rnd_s/source/main.c:901; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --tp_wire_bf16 | 0..1 | 0 | Round TP transport partials/sums to BF16; zero retains FP32 wire. | load, generate, serve | all admitted TP placements; nonzero refused on PP5 | lossy | G0; T | Nonzero requires an admitted TP placement and relaxed/fast numerics. PP5 and numerics=exact refuse it at startup, including EXL3. Original exact output shards require FP32 wire. | engine_rnd_s/source/main.c:904; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --full_steps | 0..1 | 1 | Use full programs behind the prefill front. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:906; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --prefill_first | 0..1 | 0 | Prefer another session's wide/full prefill chunk before ordinary steps. | serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:908; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --prefill_prompt_chunks | 0..1 | 0 | Keep runnable FULL/WIDE prompt chunks out of ordinary BATCH packing and retain decoder turns on forced landings. | serve | TP4 EXL3 BF16 BATCH measured; explicit selection on other admitted placements remains unqualified | same solo arithmetic; concurrent schedule selects solo prompt programs | G0; mixed96/96, donor423/423 and canonical2048/1024 | Explicit1 selects the measured first0/order0 guard. Category code/structured/prose/balanced, concurrency multiple, focus decode/balanced, accuracy-oriented profile candidate; strict original-reference exact remains held. No active axis is changed here. | N round7 final guard: code C1 264.738 versus264.902, C8 stream84.009 versus84.965, aggregate633.350 versus652.906tok/s; C8 costs1.12%/3.00%, so default0 under owner delegation. Eight host progress regressions and20-request live identity/progress pass; initial stats schema mismatch is corrected in a saved-data audit and fresh progress gate. Final references/cold details are in N REPORT_v7.md. |
| --draft_learned_acceptance | 0..1 | EXL3 CLI1; other placements/API0 unless explicitly set | Update the draft policy's shared confidence table from outcomes. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:909; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --draft_session_acceptance | 0 shared table;1 session ratio;2 log-odds shift;3 Platt scale+shift;4 shift+hindsight;5 full depth while measured match>=0.82;6 position-local EMA;7 middle scout;8 second-position scout;9 natural selector;10 conditional residual calibration;11 actual successful BATCH shape costs | 0 | Scale proposal acceptance by the current session's measured outcomes. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Current S/main242e6ef accepts 0..11, default0. Modes6..11 are implemented:6 position-local EMA,7 middle scout,8 second-position scout,9 natural selector,10 conditional residual calibration,11 actual successful BATCH shape costs. Modes0..10 adjust draft-depth acceptance/calibration; mode11 additionally uses worker-private measured prices. They do not directly replace a target operator, but changed proposal depth can select different target widths/forms: draft-only classification requires target-bit preservation (gate D). Session acceptance is a measured-route modifier. No Original tuple selects6 after the decode win of TP4_PRO at pipeline depth 2 with 1 serving slot; banked PP5/session6 stays a raw alternative. Original TP4_PRO (pipeline depth 2, 1 serving slot) and PP5 (pipeline depth 8, 8 serving slots) decode routes retain0; structured stays0. PP5 depth5 session6 and TTFT10 observations remain unselected component metadata; no held tuple is admitted. The session6 proseC4 p90 TTFT cost is about26ms; session10 preserves that measured PP5 TTFT tail. EXL3 retains its recorded0/5 assignments; modes7..9 and11 remain explicit alternatives. Exact and prefill mappings are unchanged. Inherited N text/ID checks alone do not establish selected-logit invariance or a universal tuple. Mode11 has no new integration recommendation under the later N too-slow/no-merge review; parser availability is distinct from promotion (S main.c:857; engine.c:60-80; RND_N.md 03:34 and RND_S/N/R.md 5 Nov04:38). | engine_rnd_s/source/main.c:911; gxd01b session5+ties: code C8 724.8 versus661.5 (+9.6%); prose572.8 versus492.9 (+16.2%); code C1 240.9 versus244.8 (-1.6%); joint recipe (RND_N.md). |
| --batch_plan_ties | 0..1 | 0 | Cut planned drafts only between unequal predicted reach. | generate, serve | TP4_EXL3 BATCH | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; B; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:913; gxd01b joint session5+ties results: code/prose C8 +9.6/+16.2%; isolated gain unmeasured (RND_N.md). |
| --draft_start_economic | 0..1 | 1 | Start DSpark at policy depth instead of a single-proposal probe. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:915; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --chunks_in_flight | 1..8 | 1 | Bound chunk steps in flight; one uses the exact cone planner. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:917; rc02ab exact1 versus2: 30K11.859 ->9.430 s, Fraser2.431 ->1.857 s; same bits in that recipe (RND_C.md). |
| --prefill_margin | 1..3967 | 127 | Keep this many full rows beyond the prefill cone. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:920; rc02ab margin127 exact plan: 30K about -1.0 s; joint recipe, same bits (RND_C.md). |
| --prefill_fast_plan | 0..1 | 0 | Choose the largest fitting prefill program instead of the exact plan. | generate, serve | all placements/recipes | sum order | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:923; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --prefill_order | 0 round-robin chunks;1 command arrival order;2 shortest remaining prompt;3 shortest with concurrent short residuals | auto:2 for qualified BATCH1024 tuple;0 otherwise | Choose round-robin chunks, command arrival order, shortest remaining prompt or opt-in shared short residuals. | serve | all placements/recipes | same bits | G0 | Order3 requires the measured row-independent EXL3 BF16 BATCH family; rejects prefix_reuse_exact1/2, session_fork1 and prefill_prefix_stable1. Automatic order2 and explicit0/1/2 retain their defaults and gates. | engine_rnd_s/source/main.c:925; N xd13/frozen24 prefix_reuse1 same-binary order2->0 warm-code probe: C1/C2/C4/C8 TTFT medians15.348/26.221/52.603/131.152->15.295/26.310/42.449/55.001ms; following-token rates268.148/347.388/448.756/541.992->268.891/347.679/530.509/722.848tok/s. Each arm has47 strict records with paired cache classes and equal T0 text/native-count/finish; candidate follows control by44.0-44.5s. BATCH8 one-owner C1 is not THIN. Actual fixed0/session0/learn1/costs14300,0,690/core2 has no batch_step_us menu. Code-only warm TTFT/decode evidence, not all-category, mixed, long-prompt continuity or a changed S default/axis (RND_N/M.md5Nov05:17,05:32; evidence/xd13_v1_measure/warm_order_ttft.json). Branch-only N isolated8dd9dfc/main39980 adds order3: short no-chunk residuals fitting the prepared THIN budget share a head thin turn, retaining order2 fairness, long chunks and decode-due exclusions. CLI requires qualified bounded_batch1024/BF16 row-independent family and prefix_reuse_exact0; generic API fixtures are not broader CLI admission. Future P/Q join must also exclude stable1 canonical form even global exact0/fork0, and fork1/global0, until their cold-plan proof. Preserve prior automatic0/2 outcomes or refusals. Three builds,123parser cases, focused ASan/TSan and87CUDA-image identity pass; paired xd18 on frozen35 passes both-width reference stages; complete cross-order/Fraser/warm review and a new-default three-pass gate remain separate. S accepts0..2; no axis enables3 (RND_N.md5Nov13:02,13:29; RND_Q.md12:41). N xd18/frozen35 completes255responses(40coldmixed,74Fraser,141warm) plusboth-width references with observed outputs/counts/request identities equal. Order2->3 warmC4/C8codeTTFT52.731/127.670->42.439/54.534ms, aggregate427.096/516.019->471.466/610.583tok/s. Matchedorder0C4/C8TTFT41.916/54.317ms; code3stays+0.523/+0.217ms aboveit. Strict coldgap468.254->470.319ms andFraserC4/C8median4104.95/7628.50->4110.40/7632.40ms FAIL; no tolerance/default pass. This is35RATE2/canonical-menu data, not pure37/defaultthree-pass qualification. S stilldoesnotparse3 orselectit (RND_N.md5Nov14:55; xd18saved-dataanalysis).  N four-commit integration487000f/6c2aa48/3d1c9ec/a110d06 now accepts3 within that gate; no axis selects3. Pure37 xd20 warm C4/C8 code throughput improves7.856%/16.844% versus2; C2/cold gates fail, so automatic2 remains. |
| --prefill_order_wait | 0..1024 | 16 | Bound shortest-order waiting in serving scheduler ticks. | serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:927; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --prefill_decode_every | 0..64 | 0 | Insert one decode round after this many head chunks; zero disables it. | serve | all placements/recipes | same bits | G0 | Nonzero requires prefill_order=1 or2; order0 refuses at engine creation. Old implementation has whole-chunk gaps; no bounded-pause guarantee. | engine_rnd_s/source/main.c:929; gsk13 order2/every1:short post-first11.45 ->9.25s,longTTFT13.93 ->14.83s; FraserC4/C8 TTFT3745/6763 ->3814/6994ms;20/20 text agreement; remains off (RND_K.md). |
| --prefill_decode_waiters | 0 or1 | 0 | Keep whole chunks/one round while multiple prompts wait. | serve | all placements with exact one-flight prefill; automatic dynamic policy only qualified BATCH1024 | same bits | G0 | M7d86c45 is integrated; S uses the final M default where its qualified BATCH1024 admission matches; the all-balanced development tuple explicitly requests1024/1/waiters0/order2. Other explicit policies remain selectable. | Gxp31 waiter512/4:0.470s max content gap, worst TTFT ratio1.07907 including all three arrivals; C8 E2E368.24tok/s. Fixed1024/1/waiters0 selected with ratio1.07868 and pilot C8 E2E373.41. These are joint policies, not an isolated waiter gain (packet RND_M.md19:14). |
| --prefill_decode_rounds | 1..64 | 1 | Run this many ordinary decode rounds between capped chunks. | serve | all placements with exact one-flight prefill; automatic dynamic policy only qualified BATCH1024 | same bits | G0 | Requires M burst/fairness fix; bound not a TTFT guarantee. Explicit cap0 restores rounds1/waiters0/order0 unless overridden. | Gxp33 fixed1024/1:0.477757s maximum gap, together short/long TTFT3.92157/15.38475s, short post-first completion7.76530s. Gxp26 actual default:0.476522s gap; refs/mixed/decode identity pass. Joint-policy scope (packet RND_M.md19:14). |
| --prefill_chunk_rows | parser0..4096; nonzero multiple256 and <=wide_rows; measured scope<=2048 | 0 | Cap all exact-plan prefill chunks. | generate, serve | all placements with exact one-flight prefill; automatic dynamic policy only qualified BATCH1024 | same bits | G0 | Requires chunks_in_flight1, prefill_fast_plan0; reject with P exact reuse until combined proof. | Fixed-cap solo references pass. M7d86c45 default passes three reference fixtures and15/15mixed texts; this is not blanket proof of every cap/placement (packet RND_M.md19:14). |
| --prefill_decode_rows | parser0..4096; nonzero multiple256 and <=wide_rows; measured scope<=2048 | auto:1024 only for qualified BATCH serve;0 otherwise | Cap prefill chunks while decoders are active. | serve | all placements with exact one-flight prefill; automatic dynamic policy only qualified BATCH1024 | same bits | G0 | Automatic selection requires EXL3 BF16 BATCH, wide_rows1024, TC MoE/trunk MMA, prefill_first1, column owners, batch_thin_bits and exact one-flight plan. Positive cap enables fair decode cursor even when nonbinding. Nonzero cap refused with exact prefix reuse. | M7d86c45/gxp26 actual default1024/1/waiters0/order2:0.476522s gap; Fraser C1/C4/C8 TTFT1569.6/4106.4/7631.6ms; aggregate E2E205.09/317.50/373.37tok/s versus old order2 C4/C8 320.79/384.30. Both reference widths,15/15mixed and141/141decode pairs pass; THIN/generate unchanged. Joint policy, not an isolated cap gain (packet RND_M.md19:14). |
| --prefill_cone_skip | 0 none;1 dead decoder core/MoE/wide trunk;2 last-row cone;3 mode1 with every trunk | 1 | Skip dead cone work; modes two/three extend the retained optimization. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:940; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --prefill_tail_rows | 0,128,320,448 | auto:0 if inapplicable;448 THIN wide>=2048;128 otherwise | Prepare 128/320/448-row full tail programs; zero disables them. | generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T; F | Nonzero only128/320/448; requires TP4_EXL3, chunks_in_flight1, fast_plan0, full_steps1, wide_rows>=1024, TC MoE and trunk MMA. | engine_rnd_s/source/main.c:942; gsk01 448 versus320 THIN: 2K1696 ->1722 tok/s, cold3798 ->3818; same bits. BATCH defaults128 for memory (RND_K.md). |
| --prefill_ladder_1024 | 0..1 | 1 | Add 1024-row programs below larger wide programs. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:944; ri01 with front_tail: 30K -82 ms; same bits, joint recipe (RND_I.md). |
| --prefill_ladder_1536 | 0..1 | 1 | Add 1536-row programs below larger wide programs. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:946; ri06 1905-token prompt -84 ms; same bits (RND_I.md). |
| --prefill_ladder_fine | 0..1 | auto:1 if wide>=2048 and THIN;0 otherwise | Add 1792/1280/768-row exact-plan programs. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:948; ri07/gsi10 THIN cold prefill +0.8%, about200 MiB/card extra; same bits (RND_I.md). |
| --prefill_front_tail | 0..2 | auto:2 with448 tail;1 with other tail;0 otherwise | Use tail programs for the front region; two also uses the longer full tails. | generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T; F | Nonzero requires a tail program. | engine_rnd_s/source/main.c:950; rk04k THIN with448 tail: 30K8.214 ->8.190 s; same bits (RND_K.md). |
| --prefill_tail_overlay | 0..1 | 1 | Overlay tail TC MoE workspace on prefill workspace. | generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:952; 320-tail workspace -244 MiB/card; same launches/bits (RND_K.md). |
| --prefill_tail_sets | 0..1 | 1 | Run tail trunk products over their 64-row groups in one launch. | generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:954; rk03c 2K prompt -12 ms at320 tail, -15 ms at448; same bits (RND_K.md). |
| --prefill_thin_tail | 0..1 | auto:1 with tail and THIN;0 otherwise | Use 16..56-row programs for rows below a 64-row tail. | generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T; F | Nonzero requires a tail program; automatic default only without BATCH. | engine_rnd_s/source/main.c:956; rk09y/z 2K1.144 ->1.117s (-27ms),1905token1.141 ->1.075s (-66ms),same bits after final fixes; earlier rejected versions must not be reused (RND_K.md). |
| --prefill_dead_halves | 0..2 | 2 | Trim cone-dead sums; mode two also trims dead copies. | generate, serve | TP4_EXL3 THIN/BATCH | same bits | G0; T | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:958; ri01 mode1 Fraser -76ms,30K -68ms; joint mode7+pool+dead2 Fraser -117ms/30K -276ms; same bits (RND_I.md). |
| --prefix_reuse | 0..1 | 1 | Reuse eligible cached prompt prefixes instead of prefilling cold. | serve | all placements/recipes | sum order (cross-plan risk; see P) | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:960; xc01 baseline: cold-solo TTFT median0.712954s versus legacy repeat0.014448s; repeat text matches cold14/16. 25005-token cold7.697177s versus repeat0.015356s agrees; 6004-token repeat differs. Observed speed/identity tradeoff, not exact-cache qualification (RND_P.md 16:07; baseline_repeat_cost.json). |
| --prefix_reuse_exact | 0 legacy;1 final-row repeat;2 cold-plan boundaries | 0 | Admit same-length cold-provenance prompt prefixes through repeat or cold-plan boundaries. | serve | serve: mode0 legacy on admitted placements; nonzero TP4_EXL3 only | same bits within the qualified cold recipe; target-form invariance required | G0; nonzero startup and cold-provenance gates in Interactions | Nonzero requires TP4_EXL3, ring_column_owners1, batch_thin_bits1, chunks_in_flight1, prefill_fast_plan0, prefill_chunk_rows0 and prefill_decode_rows0; multiple slots require prefill_first1 or prefill_order1/2. Namespace equality precedes numeric admission. Modes1/2 require known cold_prompt_count equal to this request length, no image and no imported snapshot. Kept tokens round down to an eight-token boundary. Mode1 additionally keeps exactly prompt_count-1 and not before decoder_from; mode2 calls the child-length cold-plan boundary helper and validates any fallback checkpoint. Ordinary reach lag is<=3968; otherwise only a valid checkpoint can reuse. Prefix_reuse0 disables reuse. Server open refuses any session_directory with exact mode1/2. No nonshrinking/growing stable1 or fork admission exists in this merged source (S main.c:935,1368; server.c:66-135,1232). | engine_rnd_s/source/main.c:961; Merged raw/default0 retains legacy policy. P c52ce71 modes1/2 retain16cold+160warm+14stress and separate114coldABA/xc08 checkpoint proofs; Q final06 gpq05 closes its57request BATCH coldABA, not a fresh114matrix. main 5029986 mpqa2048/mpqb1024 passes full8/256 selected records06:15:58-06:19:13; fresh S8d6bb95 qualification is separate. No automatic axis selection or original-weight-equivalence claim. Unmerged P stable1 extension evidence is confined to its branch row (RND_P/Q.md; RND_Q5Nov06:23). |
| --cache_scope | 0 session;1 user;2 global | 0 | Choose the namespace used for tokenizer, slot and snapshot reuse. | serve | serve: all admitted placements; explicit namespace policy | same bits; service and reuse eligibility policy | G0; namespace/snapshot/idle eligibility in Interactions | 0 SESSION combines authenticated user with every recognized body/metadata/session-header identity; absent/null/empty values add no signal. UTF-8 identity strings<=512bytes each and total<=8192bytes; invalid types/limits or repeated identity headers return400. 1 USER scopes by authenticated user only; 2 GLOBAL explicitly permits cross-user/session content reuse. Anonymous requests share user "-". Namespaces gate tokenizer/KV/snapshot matching and convey neither authorization nor stored-response semantics. Snapshotv4 binds the complete namespace on the opened descriptor; old unscopedv3 files remain untouched. No session_fork or automatic same-owner cloning is parsed by S (S main.c:937; server_cache.c:180; server_sessions.c:39,150,255; server.h:42). | engine_rnd_s/source/main.c:963; Merged default0. Frozen Q gpq01 namespace112/112, final06 native/Responses and gpq05 cold controls pass within their epochs; the real-client gpq06 passes18Responses/17laterhits, not paired latency evidence. main 5029986 both-width references and one-slot two-user Chat/Responses smoke pass: each HTTP200/1905input/8output, equal content/reasoning, cache0 for both users. First pre-binary permission failure remains retained. Service policy orthogonal to all axes; no S axes/default change (RND_Q.md5Nov03:20,04:20,06:19,06:23). |
| --session_grace_seconds | 0..3600 | 30 | Prefer retaining a recently finished namespace when choosing an idle slot. | serve | serve: all admitted placements; idle-slot policy | same bits | G0; namespace/snapshot/idle eligibility in Interactions | CLI0..3600 seconds; default30, zero disables grace. Protection is a preference, never a reservation that stalls dispatch: if all free slots are protected one is still chosen. Only idle slots qualify; explicit-session affinity breaks equal-prefix ties without bypassing namespace/content/image/reach checks (S main.c:939; server.c:137-197; server.h:50,114). | engine_rnd_s/source/main.c:965; Merged default30. Q namespace/affinity and merged smoke proofs are aggregate service-policy evidence; isolated latency/capacity effect unmeasured. Orthogonal to the four axes; no automatic mapping. |
| --session_save_tokens | 1..1048576 | 1024 | Require this many processed tokens absent from a snapshot before saving. | serve | serve: configured eligible snapshot store on admitted placements | same bits | G0; namespace/snapshot/idle eligibility in Interactions | CLI1..1048576; default1024. API zero selects1024 but is not a CLI spelling. With a session directory, eviction/idle/shutdown saves require at least this many processed-minus-saved tokens; a continuing same-namespace prefix also checks the discarded unsaved suffix. Snapshotv4, namespace, image and engine state eligibility remain; this knob cannot enable a directory with prefix_reuse0 or exact reuse1/2 (S main.c:941; server.c:1232; server_sessions.c:407-423,616-654; server.h:111). | engine_rnd_s/source/main.c:967; Merged default1024. Snapshot and namespace correctness proofs do not supply an isolated performance result; no automatic axes mapping. |
| --session_load_tokens | 1..1048576 | 256 | Require this much reusable-token gain before loading a snapshot. | serve | serve: configured eligible snapshot store on admitted placements | same bits | G0; namespace/snapshot/idle eligibility in Interactions | CLI1..1048576; default256. API zero selects256 but is not a CLI spelling. Select only a namespace/image-compatible valid snapshot that keeps more than the current slot and gains>=this threshold. Complete opened-file v4 header/token/namespace and engine load checks remain. Session directory requires prefix_reuse1 and exact reuse0 (S main.c:943; server.c:1232; server_sessions.c:255-286,423-445; server.h:112). | engine_rnd_s/source/main.c:969; Merged default256. Version/namespace/replacement correctness gates do not measure an isolated latency or throughput gain; no automatic axes mapping. |
| --session_idle_save_seconds | 1..86400 | 120 | Wait this long without work before starting an eligible idle snapshot save. | serve | serve: configured eligible snapshot store on admitted placements | same bits | G0; namespace/snapshot/idle eligibility in Interactions | CLI1..86400 seconds; default120. API zero selects120 but is not a CLI spelling. Idle saving requires store present, healthy engine, no queued or active commands and a slot meeting session_save_tokens; then saves one eligible slot at a time. Shutdown saving uses its separate drain path and does not wait this idle timer. Directory requires prefix_reuse1 and exact reuse0 (S main.c:945; server_sessions.c:616-654; server.h:113-115). | engine_rnd_s/source/main.c:971; Merged default120. No isolated snapshot-I/O/performance measurement or automatic axes mapping. |
| --session_slots | 1..1024 (SERVING_SLOTS_MOST), or max | max on serve: as many as fit, no cap | Allocate this many independent active session slots; max (serve's default since 9 Nov): the most of the context (each its own --context_tokens) the cards hold, resolved at startup by the placement fit by bisection (printed with each card's free and budget MiB; refused, never shrunk, when not even one fits; parsing and its defaults see 8, and a max that resolves to one session takes --session_slots=1's defaults: pipeline depth, TP4_EXL3's draft policy). A knob that serves one session (--official_reference; --prefill_prefix_stable, --prefix_reuse_exact or --session_fork without --prefill_first or --prefill_order) given without --session_slots takes one session, named on stderr; with a count above 1, or max, it is refused, named. An explicit count the cards do not hold is refused before any load with the most that fit and the longest context of that count. A serve with BATCH knobs needs exactly --session_slots=8. | serve | all placements/recipes | same bits | G0 | Values above1 require serve; device draft plus THIN multiple slots requires pipeline_steps>1. Measured S10/e4c188e one1048576-token trueTHIN slot allocates with15MiB minimum CMP free: selected8/256, both2048/1024 reference pairs,15MiaAI signatures and cold9 pass; cold3795tok/s, Fraser1435.85/1443.60ms. Capacity plus short requests only, not full1Mwork/pi/peak envelope. Earlier S04 BATCH8/one-verifier slot125MiB/cold3207 retains its different recipe/source. | engine_rnd_s/source/main.c:973; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --queue_depth | 1..64 | 16 | Bound all waiting server requests. | serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:975; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --user_queue_depth | 1..64 | auto: max(1, queue_depth/2) | Bound waiting requests per API user. | serve | all placements/recipes | same bits | G0 | Must not exceed queue_depth; omitted becomes half global queue (at least1). | engine_rnd_s/source/main.c:977; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --step_graphs | 0..1 | 0 | Capture/replay prepared CUDA step graphs instead of eager launches. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:979; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --engram_shared | 0 or1 | 0 | Gather identical TP Engram inputs once into a shared pinned block. | generate, serve | all TP placements where shared gathers are admitted | same bits | G0; T | TP-only, immutable per-step input; no CPU expert offload. | xp04k2024-token fixture: baseline gather0/dense16 TTFT1.121s; gather1/dense16 1.111s; gather1/dense8/32/64 1.109/1.106/1.108s. Selected records identical, natural EOS after2tokens; single pass, not cold full benchmark or256-token decode proof. Prior gather A/B1.115/1.109/1.106/1.116s. Default0 retained (RND_M.md19:36). |
| --pipeline_steps | 1..8 | 5 for PP5 multi-session device THIN serving; 2 for a TP serve of several slots with the device draft and no BATCH; 1 elsewhere | Bound ordinary step pipeline depth. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:983; Original PP5 source75 repaired pair: depth2->5 cold prefill3502->4450tok/s; Fraser C1/C4/C8 medianTTFT1475.55/6739.35/12005.55->1278.75/4308.65/7165.80ms, aggregateE2E159.306/272.776/281.481->161.620/446.244/465.998tok/s. Independent literal source/config/power/cold/lifetime and46Fraser+9fixed API identities pass. NativeV1 check_failed/exit1 are preserved; repair-aware controller/root derived proof passes. Not the separate three-genre decode sweep,1M,official equality or S10 serving result. New f52c9c5 omitted-depth5 original serve/multiple default proposal is branch-only pending candidate refs; S explicitly selects measured PP5 decode at pipeline depth 8 with 8 serving slots and raw defaults remain main39980. No original TTFT/prefill tuple is admitted by this note (RND_O.md5Nov11:21). Integration batch applies f52c9c5; explicit depths retain their values. |
| --tp_pipeline_overlap | 0..1 | 0 | Run the CMP prefix and native PRO tail on disjoint workers. | generate, serve | original TP4_PRO, native PRO tail, pipeline depth>=2 | same bits in recorded current-recipe scope; joins unqualified | G0; T | Nonzero requires TP4_PRO, pipeline_steps>=2, no BATCH, exact numerics, native PRO, precision/reference linear and CMP-reference, FP32 wire, code_wire0 and attention_owner0. Prepared overlap topology refuses images, mirrors, backward transfers and expert pauses. Source availability does not enable an axis or qualify an untested join (S main.c:1227; step_pipeline.c:682). | engine_rnd_s/source/main.c:986; Merged default0. Inherited XO06 preservation and XO09 source434a895/rndo_05 gxo10/11 serving pass:12 cells and141 request/text/native-count pairs per window. TP C2-C8 improves41-47%; code C1/C2/C4/C8 serial220.75/201.47/206.24/207.30 versus overlap220.50/284.58/295.79/304.48 tok/s. Best measured PP5 remains faster at C2-C8. Prefix_reuse1 warm decode scope, not fresh S prefix_reuse0, all-step logits/state, mixed or TTFT qualification. No active S axis selects nonzero (RND_O.md 5 Nov01:19). |
| --prefill_cone | 0..1 | 0 | Use a bounded front region with a prepared prefill cone. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:988; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --wide_rows | 0..4096 | 0 | Prepare a wide prefill program of this many rows; zero disables it. | generate, serve | all placements/recipes | same bits | G0 | Nonzero requires prefill_cone and >=64 rows. TP preparation permits at most2048 rows with split_overlap enabled, otherwise1024. Overall row capacity must fit the model geometry. | engine_rnd_s/source/main.c:990; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --rows | 1..8 | 1 | Set ordinary THIN step row capacity (and proposal capacity minus one). | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:992; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --context_tokens | 1..1048576 | 1048576 (owner, 9 Nov: 1M for every session, generate's too) | Allocate the maximum prompt-plus-generation context per slot. | generate, serve | all placements/recipes | same bits | G0 | Model capacity must be positive and a multiple of16 (MODEL_CONTEXT_GRANULE; another value is refused at parse, naming the granule); TP also requires at least16384. Measured S10 trueTHIN one1048576-token slot allocates with15MiB minimum CMP free and short C1/cold/Fraser work passes (gxf_s10_single_1m_strict_s10_v1.json); not full1M work, pi, peak memory or eight1Mslots. Earlier S04 BATCH8/one-verifier125MiB is historical and separate. Capacity is an explicit raw override, not an inserted axis default. | engine_rnd_s/source/main.c:994; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --generated_tokens | 1..4294967295 | generate64; serve4294967295 | Set the generation token ceiling; serving also accepts per-request limits. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:997; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --seed | 0..4294967295 | generate0; serve random when omitted | Set sampler seed; omitted serving requests draw a seed. | generate, serve | all placements/recipes | sum order | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:999; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --reasoning_effort | 0..100 | 0 | Set template reasoning effort from zero (off) through 100. | generate, serve | all placements/recipes | sum order | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1001; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --linear_profile | 0 precision;1 speed | 0 | Choose precision or speed linear-product preparation. | generate, serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1004; Selects measured slice counts; reference-order chains remain identical across slices. Isolated profile0/1 whole-workload effect unmeasured (source/linear.c, trunk01). |
| --linear_order | 0 fixed;1 reference | 1 | Choose fixed (zero) or source reference (one) product accumulation order. | generate, serve | all placements/recipes | sum order | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1007; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --official_hc_rows | parser0..8; admitted0/1/3/4/8 | 0 | Select pinned HC arithmetic with an explicit reference call shape. | generate, serve | original PP THIN only for nonzero; TP and EXL3 unsupported | sum order toward pinned-source HC; full-model identity unqualified | G0 | Parser accepts0..8; model preparation admits only0/1/3/4/8. Nonzero requires original PP THIN, actual rows1/3/4/8, no allocated draft positions, row_capacity<=8, state16, linear_profile0, linear_order1 and CMP reference. WIDE, vision, TP and EXL3 are unsupported. Static reference shape does not switch prompt8 to cached1 or establish verification invariance; conditional snapshot identity retains legacy domain0. Exact-axis composition remains held (S main.c:964; model_v41.c:186,204; model_v41_program.c:5044). | engine_rnd_s/source/main.c:1010; Merged default0 through the R HC subset in main75d3d17/S829ac34. Inherited XR04 ordinary preservation343/343 and first-attention HC1/HC8 checks retain their limited runtime scope. XR06 passes64 crossed direct/staged complete HC sites,48 fixed-F HC prefixes and corrected norm/rotation/observer gates. Not complete runtime composition, persisted logical-state, generation or speculation proof. Component CMP R1/F1 projection11.296us versus legacy5.040; stagedR8/F8 22.488us versus direct92.864/legacy12.064. No full S C1/C8 step-cost inference. Only a held exactness=exact dependency; no nonzero balanced/fast/EXL3 HC axis mapping (RND_R.md 5 Nov02:46; FIXES.md:23-26). |
| --official_reference | 0 / 1 | 0 | Select composed official-reference arithmetic through prepared original-weight programs. | generate, serve | original PP THIN serial M1 only | official-reference arithmetic on the recorded original PP THIN serial M1 fixtures; final-binary release receipts required | G0; nonzero requires original PP THIN, rows1, one session, context capacity at most512, no wide/full or speculative program, native PRO and HC reference rows1. Serving also requires prefix_reuse0/prefix_reuse_exact0/session_fork0. TP and EXL3 refuse. | Exact axis inserts the canonical placement and supporting flags; conflicting exact recipe overrides refuse. Balanced and omitted-axis defaults are unchanged. | XR57 production prelude and two autonomous 256-draw CLI source comparisons pass; XR59 canonical 2048/1024 hashes pass. Earlier XR50b/XR53 release components and XR48/XR49 packet source/state proofs retain their stated scope. Final-binary CLI/server/default and quiet-cost receipts, including final C1 cost, are maintained externally in COSTS.md/REPORT_v4.md and must pass before shipping. |
| --ignore_eos | 0 / 1 | 0 | Continue generation through end-token draws to the requested token/context limit. | generate, serve | non-speculative THIN only | same mathematical draws; explicit stopping-policy change | G0; rejects speculation and BATCH | Immutable per engine; end-token draws remain ordinary token events. Normal default0 still stops at EOS. | Track R serial fixed-horizon source gate uses1; normal stopping has136 draws including EOS on that fixture. |
| --prompt_token_ids | JSON file path | none | Read explicit prompt IDs instead of rendering chat messages or tokenizing text. | generate only | generate [OPTIONS] --prompt_token_ids=IDS_JSON TABLES WEIGHTS; one explicit pack | same input IDs; no mathematical change | JSON nonempty uint32 array; IDs below tokenizer vocabulary; at most context capacity minus1 | Replaces the MESSAGES positional argument; empty/duplicate paths, extra messages or explicit pack copies refuse. Automatic turbo replicas remain available. | Track R original serial M1 fixture; sampled recurrence uses the normal engine command. |
| --port | 1..65535 | 8321 | Listen on this TCP port. | serve | all placements/recipes | same bits | G0 | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1013; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --speculative | 0 off;1 lookup;2 DSpark;3 hybrid | 0 | Choose off, n-gram lookup, checkpoint DSpark or hybrid proposals. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Nonzero requires rows>1. Every TP startup requires DSpark or hybrid (2/3); off/lookup refuse there. Device/hybrid depth cannot exceed draft_block; multi-slot THIN requires pipeline_steps>1. | engine_rnd_s/source/main.c:1015; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --draft_policy | 0 fixed;1 cost;2 rate;3 adaptive | CLI EXL3 auto:0 for one session THIN (also a serve whose --session_slots=max resolves to one),2 for BATCH or several sessions; other placements1; explicit values win | Choose fixed, cost, rate or adaptive proposal-depth selection. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1017; Controlled S06/d71cd05 binary76c98fc7 evidence: canonical head_shards1/head_stream1/draft_shards1/draft_layers0 with policy0 passes full8/256 selected-record references and15 C1 MiaAI native/text/finish checks in gxf_s06_code_single_short. Fixed0 C1 code230.959/prose158.954/structured330.108 tok/s; cold prefill3909 tok/s with cached0; Fraser C1 median/worst1379.4/1395.1 ms. Changing only policy0 to1, with the other concrete flags equal, fails the30K selected-record gate: all8 IDs agree but6 selected F32 logits differ, max about7.63e-5. Policy1 MD5ff1489cffe4194bb52fe36eef08056c9 equals the older S04gxf15 policy1 file despite that older recipe being unsharded. This controlled result does not establish a head-sharding cause; target width/form or selected-logit provenance remains under investigation. No current policy1 speed is measured, and fixed0 results do not transfer to it. Those S06 raw defaults were unchanged during that experiment. Current main39980c7 resolves omitted EXL3 policies to fixed0 or RATE2; other placements retain1. Cost1 still lacks same-bits-to-fixed-canonical balanced qualification. Source: evidence/xf_axes_numerics_v3/thin_current_30k.*; RND_S/N/R.md 5 Nov04:38. Historical N gxd01c/01a code264.3/244.8 and prose162.7/168.7 tok/s retain their different joint-recipe scope, not this controlled S comparison. |
| --draft_threshold | 0..1000 | 100 | Set the minimum draft confidence in thousandths. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1020; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --draft_base_us | 1..1000000000 | EXL3 CLI14300; other placements/API12700 unless explicitly set | Estimate the base verification-step cost in microseconds. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1023; Measured S06 xf11/gxf_s06_rate_control_pair versus gxf_s06_rate_priors_pair: RATE2, learn1/session0/ties0/core2 and identical MiaAI→M-arrivals→Fraser. Joint base/row prices 13000/1200 to 14300/690 microseconds, tail fixed0. Code C1/C2/C4/C8 changes +6.001/+3.526/+7.396/+0.446%; prose C2/C8 changes -6.168/-2.463%. Full selected8/256 references and141 cross-arm request/text/native-count/finish signatures agree; sampled native decode advances during long prefill in both warm repetitions. Fraser median deltas C1/C4/C8 +0.65/+4.50/+8.55 ms; worst deltas -5.70/+28.30/+7.00 ms. Warm-short maximum SSE gap 0.473578 to0.469387 s; no native-token timing claim. The historical zero-margin Fraser screen fails. The maintainer later accepted the N31 raw-default bucket under its noise interpretation; that selection does not rewrite this failure or by itself qualify a bounded S tuple. CPU builds/tests overlapped parts of these windows; no isolated causal or universal optimum claim. No cold/mixed/1M/full-logit/official qualification in this pair. Source: evidence/s06_rate_pair_review_v2.json. |
| --draft_tail_us | 0..1000000000 | EXL3 CLI0; other placements/API2200 unless explicitly set | Estimate fixed tail cost in microseconds. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1026; Tail0 is held fixed in the scoped S06 xf11 joint base/row-price pair; no isolated tail effect was measured. |
| --draft_row_us | 1..1000000000 | EXL3 CLI690; other placements/API2700 unless explicitly set | Estimate each additional verification row's cost in microseconds. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1029; Measured S06 xf11/gxf_s06_rate_control_pair versus gxf_s06_rate_priors_pair: RATE2, learn1/session0/ties0/core2 and identical MiaAI→M-arrivals→Fraser. Joint base/row prices 13000/1200 to 14300/690 microseconds, tail fixed0. Code C1/C2/C4/C8 changes +6.001/+3.526/+7.396/+0.446%; prose C2/C8 changes -6.168/-2.463%. Full selected8/256 references and141 cross-arm request/text/native-count/finish signatures agree; sampled native decode advances during long prefill in both warm repetitions. Fraser median deltas C1/C4/C8 +0.65/+4.50/+8.55 ms; worst deltas -5.70/+28.30/+7.00 ms. Warm-short maximum SSE gap 0.473578 to0.469387 s; no native-token timing claim. The historical zero-margin Fraser screen fails. The maintainer later accepted the N31 raw-default bucket under its noise interpretation; that selection does not rewrite this failure or by itself qualify a bounded S tuple. CPU builds/tests overlapped parts of these windows; no isolated causal or universal optimum claim. No cold/mixed/1M/full-logit/official qualification in this pair. Source: evidence/s06_rate_pair_review_v2.json. |
| --draft_probe_rounds | 1..1048576 | 8 | Probe full draft depth after this many policy rounds. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1032; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --hybrid_lookup_prior | 0..1000 | 500 | Set lookup acceptance prior in thousandths. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1035; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --hybrid_lookup_min | 1..16 | 3 | Use lookup in hybrid mode only from this n-gram length. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Hybrid requires <=lookup_ngram_max. | engine_rnd_s/source/main.c:1038; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --draft_trace | 0 in release; 0 or 1 in debug/thread/profile | 0 | Print per-round draft anchors and proposals for diagnostics. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | same bits | G0; T | Release accepts only 0; debug/thread/profile accept 0 or 1, with optional diagnostics compiled only when timing hooks are enabled (N c565010, S095a3eb). No axis enables it. | engine_rnd_s/source/main.c:1041; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --draft_block | 5..7 | 5 | Set the device draft block capacity. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Device/hybrid; TP allocates a block of5..7. Explicit draft_positions above the block refuses; omitted depth clamps to the block. | engine_rnd_s/source/main.c:1044; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --draft_positions | 1..7 | 7; device/hybrid clamps to draft_block (default5) | Bound proposals verified per round. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Also set by speculative_depth. Device/hybrid explicit depth must not exceed draft_block; actual depth is min(value,rows-1). TP allocates draft_block5..7 independently, so proposal caps1..4 remain admitted. | engine_rnd_s/source/main.c:1047; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --speculative_depth | 1..7 | same as draft_positions | Alias of draft_positions. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Same destination as draft_positions; axis expansion treats aliases as one override key. | engine_rnd_s/source/main.c:1050; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --lookup_ngram_max | 1..16 | 4 | Set the longest lookup suffix searched. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Prepared model/driver validation remains authoritative; ignored outside the named implementation path. | engine_rnd_s/source/main.c:1053; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --lookup_ngram_min | 1..16 | 2 | Set the shortest lookup suffix searched. | generate, serve | generate/serve with corresponding drafter; all placements where admitted | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Must not exceed lookup_ngram_max. | engine_rnd_s/source/main.c:1056; Unmeasured isolated effect. Source implementation/CLI contract only. |
| --numerics | exact / relaxed / fast | exact | Choose admitted startup numerical policy. | load, generate, serve | all placements/recipes | sum order | G0 | Explicit exact refuses relaxed CMP order; relaxed/fast default to relaxed CMP order only for PP5. TP placements retain reference CMP order and refuse explicit relaxed. Exact is admitted only for PP5, TP4_PRO and TP4_EXL3. | source/startup_policy.c:101..135; RND_PIPELINE.md. Unmeasured isolated effect. |
| --placement | pp5 / tp4_pro / tp4_exl3 / tp2_pp2_pro / tp2_single_single_pro | pp5 | Choose prepared weight/state placement. | load, generate, serve | all placements/recipes | sum order | G0 | tp5 parses in policy helper but is refused by admission. PP5 permits layout; TP layouts refuse it. Native PRO output depends on its layers. | source/startup_policy.c; COMPARISON_MATRIX.md recipe epochs; no isolated effect. |
| --weights | original / exl3 | unset (as --placement and the pack imply) | Name the weight family. | load, generate, serve | all placements/recipes | same bits | G0 | Omitted: unchanged. Given without --placement: the family's default split (original PP5, EXL3 TP4_EXL3; printed without an axis flag, named by the axes gpu_split line with one), also for the axis tables (keyed on split and family); a value other than original or exl3 refuses as such; load keeps its own placement (an explicit PP5 included) and only takes the family's pack. An explicit --placement of the other family, a pack argument of the other kind (.exl3map or not) or a pack whose matrices are of the other family refuses, naming both. | source/commands.c take_model_source, weights_placement_matches, pack_family_status; no isolated effect. |
| --model_dir | a folder path, at most 1023 bytes | none (positional tables and pack) | Read the model from one folder. | load, generate, serve | all placements/recipes | same bits | G0 | Replaces the tables and pack arguments (generate keeps its messages file); names tokenizer.bin, original.anemone and the EXL3 map NAME.exl3map (--exl3_map, default model; with --weights=exl3 or tp4_exl3). Each name may be a link: resolved once to a regular file whose device and inode the open must match (a swap refuses). Startup prints each file's real path and device, and the pack's name and digests; identity stays the pack's. | source/model_folder.c, source/commands.c model_folder_files and model_pack_opened; no isolated effect. |
| --exl3_map | a map's name in the folder: not empty, no '/', at most 247 bytes | model (model.exl3map) | Pick the EXL3 map NAME.exl3map in --model_dir's folder. | load, generate, serve | where --model_dir's pack is its EXL3 map (--weights=exl3 or --placement=tp4_exl3) | same bits | G0 | Selection, not a measured option: no axis maps it. One folder holds several maps of one model with one tokenizer.bin and one original.anemone, each NAME.exl3map with its descriptor beside its real file. Refused, naming it, without --model_dir or where the folder's pack is original.anemone; a name with '/' or too long refuses as such; the name is resolved as the folder's other files are (a link followed, device and inode kept). | source/model_folder.c model_folder_map_file, source/commands.c take_model_source and folder_file_name; no effect on the arithmetic. |
| --cmp_order | reference / relaxed | reference | Choose CMP trunk/expert sum order. | generate, serve | all placements/recipes | sum order | G0 | Explicit numerics=exact requires reference; omitted numerics resolves legacy relaxed CMP to relaxed policy. | source/startup_policy.c. Unmeasured isolated effect. |
| --temperature | finite float >=0 | 0 | Set sampling temperature; zero is greedy. | generate, serve | all placements/recipes | sum order | G0 | Ordinary command/parser and runtime value gates apply, including under exactness=exact; temperature, top_p and seed are unrestricted within their ordinary ranges. | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --top_p | float >0 and <=1 | 1 | Set nucleus probability cutoff. | generate, serve | all placements/recipes | sum order | G0 | Ordinary command/parser and runtime value gates apply. | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --bind_address | numeric IPv4/IPv6 | 127.0.0.1 | Bind the server listener. | serve | all placements/recipes | same bits | G0 | Ordinary command/parser and runtime value gates apply. | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --key_file | path | none | Read API-user authentication records at startup. | serve | all placements/recipes | same bits | G0 | Regular, nonempty file <=1MiB with no group/other permission bits; at most64 unique users/keys, exact record format validated (server_keys.c:43..143). | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --stats_users | comma list of <=64 usernames; each1..32 ASCII letters/digits/._- | all authenticated users; open stats when auth disabled | Authorize named users to access the statistics endpoint. | serve | all placements/recipes | same bits | G0 | Requires key_file; no empty names; endpoint authorization allowlist, independent of live-statistics presentation (server.c:595..600,1056..1072). | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --model_alias | 1..63 bytes | none | Add a request/listing model alias. | serve | all placements/recipes | same bits | G0 | Supplied alias must fit63 bytes; adds a name without replacing the model identity (server.c:1011..1013). | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --session_directory | path | none | Persist eligible session snapshots in this directory (requires prefix_reuse=1). | serve | all placements/recipes | same bits | G0 | Ordinary command/parser and runtime value gates apply. | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --draft_usage | none / compact / full | compact | Choose draft statistics detail in responses. | serve | all placements/recipes | same bits | G0 | Ordinary command/parser and runtime value gates apply. | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --dump_draft_ring | path | none | Write diagnostic draft-ring rows at shutdown. | generate | all placements/recipes | same bits | G0 | Ordinary command/parser and runtime value gates apply. | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --capture_golden | dataset root path | none | The golden dataset root (ROOT/cases/<set>/<id>/: prompt_ids.json, token_ids.json, metadata.json), read only; ROOT/MANIFEST.json's SHA-256 (null when absent) joins the plan digest. | capture only | capture [generate's options and axis flags] --capture_golden=ROOT --capture_cases=KEYS --capture_out=OUT TABLES PACK, or --model_dir=FOLDER; every split and map generate takes, one session | same bits (output only; generate's and serve's paths unchanged) | required with capture; a nonempty path; once | Each case is read and checked before the first card: schema_version 1, temperature 0, autoregressive_greedy, generated_positions, hashes of the canonical IDs, the model's vocabulary, the case within --context_tokens. | B3 (16 Nov 2026): source/golden_capture.c, tests/golden_capture_test.c and parse_refusal_test.c; no isolated cost measured (capture only). |
| --capture_cases | text file path | none | The keys to capture, one set/id a line, in run order: tools/golden/selections/release54.txt (the release's 54), or tools/golden/golden_cases.py ROOT (all). | capture only | capture [generate's options and axis flags] --capture_golden=ROOT --capture_cases=KEYS --capture_out=OUT TABLES PACK, or --model_dir=FOLDER; every split and map generate takes, one session | same bits (output only) | required with capture; a nonempty path; once; each key set/id of [A-Za-z0-9_.-] | The keys file's SHA-256 joins the plan digest. | B3 (16 Nov 2026): source/golden_capture.c, tests/golden_capture_test.c and parse_refusal_test.c; no isolated cost measured (capture only). |
| --capture_out | folder path | none | The capture root: OUT/cases/<set>/<id>/ in the scorer's candidate layout (tools/golden_score.md) with logits.f32 the head's F32 rows the sampler read, witness.json, CAPTURE_MANIFEST.json and cases.jsonl. | capture only | capture [generate's options and axis flags] --capture_golden=ROOT --capture_cases=KEYS --capture_out=OUT TABLES PACK, or --model_dir=FOLDER; every split and map generate takes, one session | same bits (output only) | required with capture; a nonempty path; once; new, or a capture of the same plan; free space at least the remaining rows' bytes and 1 GiB | Resume: a case with its folder and its cases.jsonl line is skipped, a .partial folder is run again; a folder holding anything else, or another plan's capture, is refused. | B3 (16 Nov 2026): source/golden_capture.c, tests/golden_capture_test.c and parse_refusal_test.c; no isolated cost measured (capture only). |
| --capture_rows | 1..8 | speculative_depth + 1 (1 without speculation) | The drawing rows of a forced step: 1, one row a step; 2 to speculative_depth + 1, the verify form whose rows carry the next golden tokens as accepted proposals (TP4_EXL3's head arithmetic depends on the step's rows). | capture only | capture [generate's options and axis flags] --capture_golden=ROOT --capture_cases=KEYS --capture_out=OUT TABLES PACK, or --model_dir=FOLDER; every split and map generate takes, one session | the rows' own head arithmetic: recorded in metadata.json (rows) | 1 to 8; at most speculative_depth + 1; once | B3 ruling 2: TP4_EXL3's quality number is R = 6, with R = 1 recorded beside it. | B3 (16 Nov 2026): source/golden_capture.c, tests/golden_capture_test.c and parse_refusal_test.c; no isolated cost measured (capture only). |
| --capture_positions | 1..4294967294 | all | The first N positions of each case (pilots); metadata.json says truncated, and golden_capture_score.py refuses a truncated capture. | capture only | capture [generate's options and axis flags] --capture_golden=ROOT --capture_cases=KEYS --capture_out=OUT TABLES PACK, or --model_dir=FOLDER; every split and map generate takes, one session | same bits (fewer rows) | 1 or more; once | The continuation hash and token_ids.json cover the positions kept. | B3 (16 Nov 2026): source/golden_capture.c, tests/golden_capture_test.c and parse_refusal_test.c; no isolated cost measured (capture only). |
| --capture_free | 0 or 1 | 0 | After each forced case, one ordinary T=0 command of its prompt on generate's path, its limit the positions: free_token_ids.json and free_logits.json (its events' tokens and logit bits). | capture only | capture [generate's options and axis flags] --capture_golden=ROOT --capture_cases=KEYS --capture_out=OUT TABLES PACK, or --model_dir=FOLDER; every split and map generate takes, one session | same bits (one more command a case) | 0 or 1; once | B3 ruling 5: the pilot, and TP4_EXL3 2.9 at R = 6 on the 54; the scorer reports its first divergence. | B3 (16 Nov 2026): source/golden_capture.c, tests/golden_capture_test.c and parse_refusal_test.c; no isolated cost measured (capture only). |
| --layout | <=4096-byte statements joined by semicolon+space: <units>: card <c> or experts kept: <units>: <n> | automatic fill | Override PP5 unit placement. | load, generate, serve | all placements/recipes | sum order | G0 | PP5 only. Units are embedding/vision/head/draft head or layers a..b, c (up to16 ranges/statement); exact spaces required. Each stage must be assigned once, ranges must exist in order, card indices and kept counts must fit, and card budgets may not be exceeded (placement.h:16..35). | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --live_stats_format | text / json / table | text | Choose live server statistics format. | serve | all placements/recipes | same bits | G0 | Ordinary command/parser and runtime value gates apply. | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. |
| --draft_calibration | path to <4096-byte file; exactly112 integers0..1000 | none | Override draft acceptance priors in mode/position/bin order. | generate, serve | all placements/recipes | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Exactly112 whitespace-separated unsigned integers0..1000, optional trailing whitespace, no extra tokens, at most4095 bytes. Overrides acceptance priors only; cost knobs are independent (commands.c draft_calibration_read). | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. Draft policy/calibration target-bit qualification follows gate D and the controlled S06 policy finding. |
| --draft | lookup / dspark / hybrid | off | Select speculative proposal source by name. | generate, serve | all placements/recipes | draft-only with target-bit invariance across verification widths/forms; conditional sum order otherwise | G0; D | Draft aliases must agree with explicit speculative; layout only PP5; no secrets printed. | source/commands.c custom parser and named implementation gates. Unmeasured isolated effect. Draft policy/calibration target-bit qualification follows gate D and the controlled S06 policy finding. |
| --axes | off | unset: the split's all-balanced row (its base) | Stage 3: off keeps the raw engine defaults with nothing inserted, the no-axis behaviour before the base recipes; gates and references pin it. | load, generate, serve | every placement | raw defaults | G0 | Refuses with any axis flag or any value but off; the resolver removes it before the parser. | Stage-3 decision 1, 8 Nov (source/startup_axes.c axes_off_expand); no output change against the no-axis command before stage 3. |
| --category | code / prose / structured / vision / balanced | unset: balanced (an empty set on every split) | Choose the workload objective; EXL3 balanced requires its scoped mixed-user benchmark; on TP4_EXL3, code adds fixed draft prices. The measured TP4_PRO single-user decode recipe is GPU split TP4_PRO's concurrency=single set. | generate, serve | GPU split tables pp5 (default), tp4_pro and tp4_exl3: --placement picks the split and no tuple changes it; balanced/fast overrides win and exact recipe conflicts refuse | axis-dependent; see help axes | No axes: CLI unchanged. A value the split's table holds refuses with its reason, and a split without a table refuses any axis flag; ordinary startup/model gates remain authoritative. | Explicit low-level knobs and existing aliases override inserted balanced/fast settings in either order; conflicting exact recipe overrides refuse. Duplicate axis selection refuses. No named presets or axis aliases exist. An unset axis takes balanced, so a partial tuple resolves to that row of the split's table; under exact an unset concurrency takes the reference recipe's single session. Omitted temperature is 0 and top_p is 1; axes leave both defaults unchanged. Within the admitted original exact scope, temperature/top_p affect sampling without changing the reference-arithmetic logits; token identity requires T=0 or matching RNG and sampler order. Our RNG/order do not match the reference; stochastic token identity is unqualified. | tools/cli/axes.json -> source/cli_axes_data.h / tools/cli/AXES.md; startup_axes.c; tests/startup_axes_test.c. S04 gxf11/gxf12 and completed S05 gxf_s05_default1024 are scoped measurements. Corrected S06 THIN/numerical and S07 original windows have separate pending gates; each strategy identifies its source epoch. |
| --exactness | exact / balanced / fast | unset: balanced (an empty set on every split) | Choose reference arithmetic, current reproducible arithmetic, or qualified bounded-loss arithmetic. | generate, serve | GPU split tables pp5 (default), tp4_pro and tp4_exl3: --placement picks the split and no tuple changes it; balanced/fast overrides win and exact recipe conflicts refuse | axis-dependent; see help axes | No axes: CLI unchanged. A value the split's table holds refuses with its reason, and a split without a table refuses any axis flag; ordinary startup/model gates remain authoritative. | Explicit low-level knobs and existing aliases override inserted balanced/fast settings in either order; conflicting exact recipe overrides refuse. Duplicate axis selection refuses. No named presets or axis aliases exist. An unset axis takes balanced, so a partial tuple resolves to that row of the split's table; under exact an unset concurrency takes the reference recipe's single session. Omitted temperature is 0 and top_p is 1; axes leave both defaults unchanged. Within the admitted original exact scope, temperature/top_p affect sampling without changing the reference-arithmetic logits; token identity requires T=0 or matching RNG and sampler order. Our RNG/order do not match the reference; stochastic token identity is unqualified. | tools/cli/axes.json -> source/cli_axes_data.h / tools/cli/AXES.md; startup_axes.c; tests/startup_axes_test.c. S04 gxf11/gxf12 and completed S05 gxf_s05_default1024 are scoped measurements. Corrected S06 THIN/numerical and S07 original windows have separate pending gates; each strategy identifies its source epoch. |
| --concurrency | single / multiple / balanced | unset: balanced (an empty set on every split) | Choose a C1 recipe, a C2-C8 recipe, or a measured compromise. | generate, serve | GPU split tables pp5 (default), tp4_pro and tp4_exl3: --placement picks the split and no tuple changes it; balanced/fast overrides win and exact recipe conflicts refuse | axis-dependent; see help axes | No axes: CLI unchanged. A value the split's table holds refuses with its reason, and a split without a table refuses any axis flag; ordinary startup/model gates remain authoritative. | Explicit low-level knobs and existing aliases override inserted balanced/fast settings in either order; conflicting exact recipe overrides refuse. Duplicate axis selection refuses. No named presets or axis aliases exist. An unset axis takes balanced, so a partial tuple resolves to that row of the split's table; under exact an unset concurrency takes the reference recipe's single session. Omitted temperature is 0 and top_p is 1; axes leave both defaults unchanged. Within the admitted original exact scope, temperature/top_p affect sampling without changing the reference-arithmetic logits; token identity requires T=0 or matching RNG and sampler order. Our RNG/order do not match the reference; stochastic token identity is unqualified. | tools/cli/axes.json -> source/cli_axes_data.h / tools/cli/AXES.md; startup_axes.c; tests/startup_axes_test.c. S04 gxf11/gxf12 and completed S05 gxf_s05_default1024 are scoped measurements. Corrected S06 THIN/numerical and S07 original windows have separate pending gates; each strategy identifies its source epoch. |
| --focus | balanced / ttft / decode / prefill | unset: balanced (an empty set on every split) | Choose the optimization objective within the selected numerical tier. | generate, serve | GPU split tables pp5 (default), tp4_pro and tp4_exl3: --placement picks the split and no tuple changes it; balanced/fast overrides win and exact recipe conflicts refuse | axis-dependent; see help axes | No axes: CLI unchanged. A value the split's table holds refuses with its reason, and a split without a table refuses any axis flag; ordinary startup/model gates remain authoritative. | Explicit low-level knobs and existing aliases override inserted balanced/fast settings in either order; conflicting exact recipe overrides refuse. Duplicate axis selection refuses. No named presets or axis aliases exist. An unset axis takes balanced, so a partial tuple resolves to that row of the split's table; under exact an unset concurrency takes the reference recipe's single session. Omitted temperature is 0 and top_p is 1; axes leave both defaults unchanged. Within the admitted original exact scope, temperature/top_p affect sampling without changing the reference-arithmetic logits; token identity requires T=0 or matching RNG and sampler order. Our RNG/order do not match the reference; stochastic token identity is unqualified. | tools/cli/axes.json -> source/cli_axes_data.h / tools/cli/AXES.md; startup_axes.c; tests/startup_axes_test.c. S04 gxf11/gxf12 and completed S05 gxf_s05_default1024 are scoped measurements. Corrected S06 THIN/numerical and S07 original windows have separate pending gates; each strategy identifies its source epoch. |
| --host | numeric IPv4/IPv6 | 127.0.0.1 | Connect to a statistics server. | stats | all placements/recipes | same bits | G0; stats numeric host/port required; duplicates refuse | Stats command contract only. | source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. |
| --port | 1..65535 | required | Choose statistics server port. | stats | all placements/recipes | same bits | G0; stats numeric host/port required; duplicates refuse | Stats command contract only. | source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. |
| --key_file | path | none | Read a statistics API key without exposing it. | stats | all placements/recipes | same bits | G0; stats numeric host/port required; duplicates refuse | Single-key file:16..4096 ASCII letters/digits/-._~+/=, optional final CR/LF; regular file with no group/other permissions. Multi-record/commented serve key files refuse (server.c:961..995). | source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. |
| --interval_ms | 100..60000 | 1000 | Poll counters at this interval. | stats | all placements/recipes | same bits | G0; stats numeric host/port required; duplicates refuse | Stats command contract only. | source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. |
| --format | text / json / table | text | Choose statistics presentation. | stats | all placements/recipes | same bits | G0; stats numeric host/port required; duplicates refuse | Stats command contract only. | source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. |
| --once | flag without value | false | Print one statistics response and exit. | stats | all placements/recipes | same bits | G0; stats numeric host/port required; duplicates refuse | Stats command contract only. | source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. |
| --user | 1..32 ASCII letters/digits/dot/underscore/hyphen | required | Choose the user for key generation. | keygen | all placements/recipes | same bits | exactly one --user argument | None beyond command contract. | source/server_keys.c:230. Unmeasured; no model computation. |
| --rate_mib | 0..overflow-safe UINT64 bound | 2048 | Throttle replica copy rate in MiB/s; zero is unlimited. | replicate | all placements/recipes | same bits | source pack must open; target capacity/receipt/device gates | Parser accepts repeated replica options with last value; source/target are positional. | source/replicas.c:488, include/anemone/replicas.h. Unmeasured isolated effect. |
| --floor_gib | 0..UINT64_MAX/1073741824 | 100 | Keep this much free space on replica volumes. | replicate | all placements/recipes | same bits | source pack must open; target capacity/receipt/device gates | Parser accepts repeated replica options with last value; source/target are positional. | source/replicas.c:488, include/anemone/replicas.h. Unmeasured isolated effect. |
| --prune | 0 or1 | 0 | Remove receipted older-source replicas before copying. | replicate | all placements/recipes | same bits | source pack must open; target capacity/receipt/device gates | Parser accepts repeated replica options with last value; source/target are positional. | source/replicas.c:488, include/anemone/replicas.h. Unmeasured isolated effect. |
| --adopt | COPY (a path) | unset | Adopt an existing full copy of SOURCE as a replica without copying a byte: read and checked in full against the pack's digests (the file unchanged while read), then hard-linked into its drive's replica directory and receipted; a differing extent is named and gets no receipt. | replicate | all placements/recipes | same bits | source pack must open; the copy an owned regular file of the source's size, not the source, on another admitted drive | Takes one SOURCE and no DIRECTORY; --replica_external applies. Its directory is the first of the search's on that drive that exists, else one made: $HOME/anemone_engram_rnd/replicas on HOME's filesystem, then <mount>/anemone_engram_rnd/replicas and <mount>/anemone/replicas. | source/replicas.c (replicas_adopt), include/anemone/replicas.h, tests/replica_adopt_test.c. The check reads the whole copy: disk-timed. |
| --stage | copy / load / engram | unset (the copy, then the load stage, then the Engram stage) | Run one autotune stage alone: copy prepares and verifies the replicas with their receipts and writes no profile; load tunes the TP host arena's read and writes the profile's load section as a candidate; engram tunes serving's step I/O and writes the profile's engram section as a candidate. | autotune | all placements/recipes | same bits | another stage is refused by name (copy, load and engram) | Each stage has its own options, state and profile section: the copy stage refuses --tokens and --retune, the load stage the copy stage's options and the Engram stage's, the Engram stage the copy and load stages'. Without --stage a stage whose section names this pack and these copies is skipped unless --retune, and the Engram stage is skipped without --tokens, each with one line; the combined tuning is retired (v0.0.4). | source/storage_autotune.c (storage_autotune_cli, stages_after_copy), source/storage_autotune_load.c (storage_autotune_load_cli), source/storage_autotune_engram.c (storage_autotune_engram_cli); docs/user/STORAGE_PROFILE.md. No model computation. |
| --tokens | FILE of little-endian uint32 token ids | required with --stage=engram; without --stage the Engram stage is skipped, one line naming it | One real prompt as little-endian uint32 token ids (a case a file) that the Engram stage replays at its seven levels through serving's step I/O; repeat for up to 16 files, at most 1 MiB together. | autotune --stage=engram, autotune (no stage) | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone; ids of this pack's vocabulary | tools/measure/engram_trace tokenize makes them from public prompts (the golden dataset's, Hugging Face anoane/anemone-ds41f1017-golden); the profile records their digest, cases, tokens and rows, never a token. --stage=copy and --stage=load refuse it. | source/storage_autotune.c (storage_autotune_cli), source/storage_autotune_engram.c (autotune_engram_trace_open); docs/user/STORAGE_PROFILE.md. Tunes the step I/O only; no output changes. |
| --prompt | refused | none (retired with the combined tuning in v0.0.4) | The combined tuning's prompt length; refused by name, as the Engram stage replays every --tokens case whole. | autotune (no stage, --stage=copy, --stage=load, --stage=engram) | all placements/recipes | same bits | refused by name by every stage, with the same reason | None. | source/storage_autotune.c (storage_autotune_cli), source/storage_autotune_load.c, source/storage_autotune_engram.c; docs/user/STORAGE_PROFILE.md. |
| --retune | flag without value | false | Run the load and Engram stages even when their sections name this pack and these copies; each writes its section again as a candidate. | autotune (no stage) | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone; --stage=copy refuses it by name | Without it a stage whose section is current (a candidate or qualified) is skipped, one line naming it; the Engram stage still needs --tokens. A --retune run is a series until a run of it ends with an exit other than 75: <profile>.retune-series names each stage that finished in it, so the same command repeated after exit 75 skips a finished stage whose section is still current and resumes the rest. | source/storage_autotune.c (stages_after_copy); docs/user/STORAGE_PROFILE.md. |
| --profile | ABSOLUTE path | $XDG_CONFIG_HOME/anemone/storage-profile.json, else $HOME/.config/anemone/storage-profile.json | The storage profile the run writes and serving reads: the copy stage's decisions beside it (.copy.json), the load stage's load section and its progress (.load-state), the Engram stage's engram section and its progress (.engram-state). | autotune (every stage) | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone; an absolute UTF-8 path whose folder is this user's alone; without it, and without XDG_CONFIG_HOME and HOME, refused by name; a schema 2 profile (the retired combined tuning's) or another source's is refused before any copy or point | serve and generate read the same default when XDG_CONFIG_HOME is unset or non-empty (or --storage_profile, or ANEMONE_STORAGE_PROFILE). | source/storage_autotune.c (storage_autotune_cli), source/storage_autotune_load.c (storage_autotune_load_cli), source/storage_autotune_engram.c (storage_autotune_engram_cli); docs/user/STORAGE_PROFILE.md. |
| --manifest | ABSOLUTE path | the profile path with .replicas.json appended | The replica manifest the copy writes and later runs verify against. | autotune (no stage, --stage=copy) | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone; an absolute UTF-8 path | Recorded in the profile. | source/storage_autotune.c (storage_autotune_cli); docs/user/STORAGE_PROFILE.md. |
| --target | ABSOLUTE_FILE | the replica search's own targets | A replica file to prepare and verify; repeat for each target (at most the copies a load opens, less the source). | autotune (no stage, --stage=copy) | all placements/recipes | same bytes | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone; the replica device policy (--replica_external, --replica_floor_percent) and the folder rule apply | Copies are engine-native; a run that ends unfinished resumes when repeated. | source/storage_autotune.c (storage_autotune_cli); docs/user/STORAGE_PROFILE.md. |
| --seconds | 1..1700 (--stage=engram 1..1650) | 1650 | The run's time budget: a run that ends before its work is done exits 75 and the same command resumes; without --stage the copy, the load stage and the Engram stage share it, each later stage given what remains (at most its own bound). | autotune (every stage) | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone | Progress is kept beside the profile. | source/storage_autotune.c (storage_autotune_cli), source/storage_autotune_load.c (storage_autotune_load_cli), source/storage_autotune_engram.c (AUTOTUNE_ENGRAM_SECONDS_MOST); docs/user/STORAGE_PROFILE.md. |
| --plan | flag without value | false | Plan only: check the devices, folders and partials in the run's order and print what a run would do, without copying, tuning, creating a folder or writing a profile; without --stage, one line a later stage says whether it would run or be skipped; the Engram stage's reads no table byte. | autotune (every stage) | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone | Read only. | source/storage_autotune.c (storage_autotune_cli), source/storage_autotune_load.c (storage_autotune_load_cli), source/storage_autotune_engram.c (storage_autotune_engram_cli); docs/user/STORAGE_PROFILE.md. |
| --verify | spot / full | spot | full reads every final copy in full whatever its receipt says; spot spot-checks a final that matches its verification receipt. | autotune (no stage, --stage=copy) | all placements/recipes | same bytes | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone | Applies to finals already in place. | source/storage_autotune.c (storage_autotune_cli); docs/user/STORAGE_PROFILE.md. |
| --copy_workers | 1..32 | 24 | The copy stage's worker threads, one pool shared by all targets: each takes one 64 MiB chunk through the read, its SHA-256, the writes, the flush, the readback and the compare. | autotune (no stage, --stage=copy) | all placements/recipes | same bytes | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone; 0 and above 32 refused by name | The decisions record it; --copy_workers=4 is the earlier pace. | source/replica_prepare.h (PREPARE_COPY_WORKERS_DEFAULT); measured on 15 Nov 2026 (96 GiB from the source's NVMe drive, A D D A against 4): one target 4.993 GB/s against 3.617, two targets 4.545 against 2.773, the same bytes and digests. |
| --copy_hash_pairs | 0 / 1 | 1 | Hash the source two chunks at a time in one interleaved SHA-256 pass (1) or one chunk at a time (0); the digests and the manifest are the same. | autotune (no stage, --stage=copy) | all placements/recipes | same bytes | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone; the load stage refuses it by name | Keeps one more 64 MiB buffer a worker. | source/replica_prepare.h (PREPARE_PAIR_SOURCE_HASHES_DEFAULT); measured on 15 Nov 2026 (96 GiB to one target at 24 workers, A D D A): 4.927 and 5.024 GB/s paired against 4.843 and 4.996 (+1.14 %), the same manifest digests. |
| --window_gib | 1..1024 | 64 | The GiB of whole host-only extents a load point reads; the A D D A reads twice the window. | autotune --stage=load | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone | Successive points read different regions of the arena. | source/storage_autotune_load.c (storage_autotune_load_cli); docs/user/STORAGE_PROFILE.md. |
| --points_per_run | 0..78 (--stage=engram 0..72) | 0 (all) | Measure at most this many points before exiting 75 to resume: the load stage's 35 arms in a balanced order and then reversed (70 points), then its eight-point A D D A D A A D confirmation; the Engram stage's grid of 32, each disturbed grid point once more, then its A D D A's 4, each disturbed A D D A point once more in its slot. | autotune --stage=load, --stage=engram | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone | Progress is kept in <profile>.load-state or <profile>.engram-state. | source/storage_autotune_load.c (storage_autotune_load_cli), source/storage_autotune_engram.c (storage_autotune_engram_cli); docs/user/STORAGE_PROFILE.md. |
| --qualify | flag without value | false | Enable the stage's candidate section (load or engram) once it names this pack and these copies (the Engram stage's: and serving's flight shape). | autotune --stage=load, --stage=engram | all placements/recipes | same bits | one --name=value token before or after PACK; a stage refuses another stage's options by name (the load stage the copy stage's and the Engram stage's, the Engram stage the copy and load stages'); the copy stage and the run without --stage refuse a load-stage option with the usage alone; a section of another pack or other copies is refused by name | Serving applies an enabled section only to the pack, copies and shape it names; the engram section only while tables are read from disk. | source/storage_autotune_load.c (storage_autotune_load_cli), source/storage_autotune_engram.c (storage_autotune_engram_cli); docs/user/STORAGE_PROFILE.md. |
| --help | flag without value | false | Print the stage's usage and exit 0. | autotune (every stage) | all placements/recipes | same bits | none | None. | source/storage_autotune.c (storage_autotune_cli), source/storage_autotune_load.c (storage_autotune_load_cli), source/storage_autotune_engram.c (storage_autotune_engram_cli). |
| --tp5_expert_csf | 0 or1 | 0 | Load losslessly encoded full448 target expert scales from an A20 sidecar. | load, generate, serve (pending O; unavailable in S) | engine_rnd_o_tp5_csf78 / guarded engine_rnd_o_tp5_join only; experimental original TP5 | same bits intended for storage-only codec; runtime unqualified | Not parsed by this S base; requires owning track merge | Original standalone CSF branch; latest guarded engine_rnd_o_tp5_join/11777a6 also applies: TP5 original raw448/512 and all native-PRO/one-session/flight/THIN/reference-linear gates. Nonzero iff tp5_expert_sidecar is supplied; absolute path required, target layers must be complete40 (or default0 resolving40). Full coverage, metadata, code/scale/source-pack and bounds identities are checked before loading. No partial sidecar or fallback. Guarded join11777a6 supports the individual paths but refuses CSF+third together before sidecar I/O or stage planning. Source availability/host factories do not qualify device fit. | Historical source25/8ba73bde full448 program factory FAILED before device load. Corrected27 repairs W1/W3/W2 engine ordering from W1/W2/W3 storage without changing sidecar ordinals. Guarded join794acd4+style11777a6 passes16release/debug host/factory invocations,151actual-parser controls,all90CUDA images literal27 and zero hooks in8host executables/3newCSF modules. Full448 sidecar274651646720bytes/230400entries has source byte/word proof. Actual five-rank device load/numerics/ring/physical fit/C1 speed remain UNMEASURED. Default0; CSF+third composition explicitly refuses before I/O/planning. No active axis, only held original/balanced/single/decode dependency (RND_O.md5Nov14:03,14:20). |
| --tp5_expert_sidecar | absolute path | none | Select the full448 A20 expert sidecar for the explicit CSF target codec. | load, generate, serve (pending O; unavailable in S) | engine_rnd_o_tp5_csf78 / guarded engine_rnd_o_tp5_join only; experimental original TP5 | same bits intended for storage-only codec; runtime unqualified | Not parsed by this S base; requires owning track merge | engine_rnd_o_tp5_csf78 only. Required iff tp5_expert_csf1; refuses relative paths, non-TP5, partial target-layer coverage or inconsistent source/layout/metadata. Does not replace the positional original pack or authorize other experimental joins. Guarded join11777a6 supports the individual paths but refuses CSF+third together before sidecar I/O or stage planning. Source availability/host factories do not qualify device fit. | Absolute full448 A20 path remains paired iff tp5_expert_csf1; guarded join794acd4/11777a6 explicitly refuses CSF plus raw384/768 third-share before sidecar I/O/stage planning. Full448 storage/source proof and corrected host factory pass; historical source25factory failure retained. No device/load/fault/lifetime/physical-fit or isolated path-selection performance claim. Defaults absent/codec0, no active axis (RND_O.md5Nov14:20; conditional MERGE_REQUEST_v3.md). |
| --tp5_pro_third | 0 or1 | 0 | Use raw384/768 TP5 geometry with target head and enabled drafts on a CMP. | load, generate, serve (pending O; unavailable in S) | engine_rnd_o_tp5_third78 / guarded engine_rnd_o_tp5_join only; experimental original TP5 | sum order; new placement arithmetic unqualified | Not parsed by this S base; requires owning track merge | engine_rnd_o_tp5_third78/f388b980 only; nonzero TP5 and exact physical card order0,1,3,4,2. All TP5 original/raw/native-PRO/reference-linear/one-session/flight/THIN/capacity gates remain. Target head and any enabled drafts move to logicalrank3/physicalGPU4/class80; target PRO remains native. Mutually separate from the CSF448 branch, with no inferred composability. Guarded join11777a6 supports the individual paths but refuses CSF+third together before sidecar I/O or stage planning. Source availability/host factories do not qualify device fit. | Guarded join794acd4+style11777a6 passes16release/debug host/factory invocations,151actual-parser cases including127old vectors+24join controls,90literalCUDA images and zero hooks in8release host executables/3newCSF modules. Actual raw384/768 allocation plans pass; no CUDA physical-fit/device/head/draft/ring/generation/rollback/speed proof. CSF+third composition explicitly refuses before I/O/planning. Default0 retains raw448/512/PRO head; no active axis, only held original/balanced/single/decode candidate. Earlier26/f388b980 host proof remains historical (RND_O.md5Nov14:03,14:20). |
| --session_snapshot_format | 4 or5 | 4 | Choose the legacy snapshot wire format or the checkpoint-bearing v5 format. | serve (pending P; unavailable in S) | engine_rnd_p only; see source | same bits for exact-copy service policy; target-form qualification required | Not parsed by this S base; requires owning track merge | Branch-only P full prototype f95615f/18. Format5 requires prefill_prefix_stable1 and prefix_reuse_exact2. Stable snapshots require format5; stable/namespace/hard-owner/opened-FD gates remain. Version5 uses a separate header/subdirectory, strict payload and checkpoint layout identity, drains actual LOAD copies and publishes only after ordinary plus optional checkpoint completion. Imported stable provenance is recognized only after verified LOAD. Legacy4 remains the default. | Branch-only P23/b12744b plus test-only80f23b9, default4. Actual format5 eviction/restart LOADs retain25000/25016tokens and260771840/260817920bytes; native/visible output equals cold, HTTP TTFT311.510/282.318ms versus7246.065cold. Actual8192-generated source serializesG8272/255604736payloadbytes. Overall xc38v4 exit1: anonymous child correctly cold-misses under Q hard-owner rule; harness wrongly expected a fork hit. Full checkpoint restart/fork and authenticated import proof remain UNQUALIFIED; prepared40v2 not queued under pause. No namespace or product gate weakened, no automatic axis (RND_P.md5Nov13:16,13:28; results/gxc38v4snapshot/partial_read.json qualified=false). |
| --tp5_host_channels | 0 or1 | 0 | Exchange TP5 messages through explicitly mapped pinned-host inboxes. | load, generate, serve (pending O; unavailable in S) | engine_rnd_o_tp5_main78 only; experimental original TP5 | same bits intended; composed device proof pending | Not parsed by this S base; requires owning track merge | Branch-only TP5. Nonzero refuses every other placement; TP5 requires original raw448/512, native PRO, relaxed numerics, reference precision/order, one session/flight, THIN rows, wide_rows0, context<=65536 and no composed O paths. Explicit mapped-host allocation/backend qualification is required; no automatic fallback. This branch uses all five devices, including the PRO, without changing its device settings. | Current main78 candidate aab1daf (engine_rnd_o_tp5_main78/source/main.c:906,1413,2946), following 5166a00. Source/CPU-only, no GPU transport identity, physical fit, serving speed or official-exact proof. Candidate original text/balanced/single/decode diagnostic only; no active axis. Direct0 retained. Source21/23 has12 focused checks,54 current-main parser controls and unchanged87 CUDA images (RND_O.md5Nov09:34,10:17). |
| --tp5_original_draft | 0 or1 | 0 | Own the unsharded original DSpark draft tail on native PRO in TP5. | load, generate, serve (pending O; unavailable in S) | engine_rnd_o_tp5_main78 only; experimental original TP5 | draft-only intended; target-bit and rollback proof pending | Not parsed by this S base; requires owning track merge | Branch-only TP5; all TP5 geometry/allocation gates apply. Nonzero requires speculative2, block5, positions5, rows8, fixed policy0, learned/session acceptance0. TP5 admits original text, one session/flight, wide_rows0, context<=65536, native PRO, precision/reference linear, relaxed numerics, no official HC or composed O paths. Target-only0 requires speculation off. No fallback to a different placement. | Source d9388b7/21 and main78 aab1daf/23 (main.c:908,1522,2946) have CPU construction/parser proof only; current54-case parser suite fixes guard ordering after placement defaults. All40 64K DSpark arena plan is77.335GB PRO and60.581/59.257/59.257/58.669GB CMP plus optional918528000 host bytes: an allocation plan, not physical-fit proof. No generation/rollback/speed/official equality, no automatic axis; held original text/balanced/single/decode candidate (RND_O.md5Nov09:34,10:17). |
| --tp_core_queries | 0 or1 | 0 | Fuse eligible true-THIN target query projection into the SCORE load. | load, generate, serve (pending N; unavailable in S) | archived M25 frozen32 only; removed from live pure37 and absent from S | same bits intended; safety/reference campaign incomplete | Not parsed by this S base; requires owning track merge | Archived M25 dfd803f/frozen32. CLI nonzero requires TP4_EXL3, ordered BF16 wire, code_wire0, attention_owner0, reference linear order, precision linear profile and pro_emulation0. Effective only on SM80 true THIN target non-index SCORE, <=8rows and same-call/alias/overflow eligibility. BATCH, prefill, draft, index, WIDE/FULL and native120 retain baseline paths; these are fallbacks rather than additional CLI refusals. | Historical M25 prototype dfd803f/frozen32 only; removed from isolated main78 N8dd9dfc/frozen36. Default0/NO MERGE. xd17b passes memcheck/racecheck/synccheck (eight cases each, zero reported errors), then first required fault control fails: generic CUDA launch failure does not prove the intended overflow guard. Later controls, OFF/ON references, native ABA, HTTP and speed remain unrun. Prior loader/LSan failures remain failures. No active axis; proposed code/balanced/single/decode stays held (RND_N.md5Nov12:56,13:02,13:29; M25_VERDICT_v4.md). |
| --help-scheduling | flag without value | not requested | Print scheduling examples and measured pause/TTFT tradeoffs without opening a model. | serve | host help; no model or card opened | same bits | Exactly anemone serve --help-scheduling | Exactly anemone serve --help-scheduling; do not append model paths or axis options. Preserve the dedicated help when merging topic routing. | M7d86c45/xp04: all required reference/boot/mixed/decode gates pass; table retains scoped pilot tradeoffs and final default. M12qualified BATCH1024 only; S qualification independent (packet RND_M.md19:14). |
| --original_queries_piece | 0 or1 | 0 | Prepare original query projection pieces for TP4 exact attention. | load, generate, serve (pending O; unavailable in S) | engine_rnd_o only; see source | same bits in recorded current-recipe fixtures; joins unqualified | Not parsed by this S base; requires owning track merge | Original exact TP4_PRO, precision/reference/CMP-reference, native PRO, FP32 wire, wide64..1024; no BATCH, wide-expert, code/owner/split-overlap experiments. | Source74b24a5 completes three Fraser/2K/30K knob0/1 pairs and three XO05 baseline controls, including selected records, prompt IDs, digests, canonical SAVE and active last-target full-vocabulary F32 bits; power restoration passes. CLI prompt+first changes -0.09%/-0.06%/+0.16%; no standalone speed win, HTTP TTFT or serving/join qualification. Default0 retained for later composition. Held original balanced-exactness prefill/ttft candidate; no S axis enables it (RND_O.md 5 Nov 02:42). |
| --original_dense_shards | 0 or1 | 0 | Shard original full-K attention dense output products and gather losslessly. | load, generate, serve (pending O; unavailable in S) | engine_rnd_o only; see source | same bits intended; device-unqualified | Not parsed by this S base; requires owning track merge | Original exact TP4_PRO, precision/reference/CMP-reference, native PRO, FP32 wire; excludes other O pilots, split/owner/code paths, dead-half/tail variants and alternate products. EXL3 admission unchanged. | O1cb6928/ff91cfe host preparation remains separate from device proof. XO18 dense starts0 of9 planned arms: 197 work seconds remained against a 900-second floor. This is a budget deferral, not a device or numerical failure. Default0; GPU identity/speed and any original balanced-exactness prefill/ttft mapping remain held; no S axis enables it (RND_O.md 5 Nov 02:42). |
| --prefill_prefix_stable | 0 or1 | 0 | Use fixed-capacity prefill forms with closed eight-row prefixes. | generate, serve | TP4_EXL3 stable-prefix recipe | sum order (new cold form); same bits relative to its qualified own-cold recipe | Default off; real parser and engine admission apply | Requires TP4_EXL3 BF16 wire, column owners, CE5, gather0, matched FULL/WIDE split forms, batch_thin_bits1, full_steps1/cone1/skip1, chunks1, fast_plan0, both M caps0, reserved prefill, margin>=7, wide1024 or2048; no images; isolated prefix19 refuses session storage (full P23 format5 is a separate unqualified prototype); multi-slot requires prefill-first or order1/2. Growing reuse needs known cold provenance, nonshrinking child, generated/sampling-row clipping, floor8, child cold-plan/cone and post-rounding reach checks; hard-owner authorization is separate. Parent/child must retain immutable stable1 identity. Integrated in merge_batch. | Prefix-only19/8c18f8a on main39980 is P final knob-default0/exact-cache-tier request; do not merge full P80 copy/v5 prototype. Fresh39 six complete cold arms14/19/14,114requests pass unchanged no-loss reader: THIN prefill ratio0.99884138(95%0.99679661-1.00268666),TTFT0.99777899(0.99398414-1.00548634); BATCH prefill0.99985033(0.99910103-1.00116771),TTFT1.00159642(0.99389811-1.00713069). Historical source15THIN strict failure retained. Whole39 exit1 belongs to separate copy32OOM diagnostic, not cold-arm failure. Core174native/checkpoint/1M and shared accepted14/Q11 references pass; standalone19 CLI refs DID NOT RUN. A resulting-main old/new2048/1024 gate still required; no automatic S exact tuple or full official-reference claim (RND_P.md5Nov14:32,14:42; evidence/xc39_v3/cold_performance.json; MERGE_v3.md). Integrated prefix-only candidate8c18f8a; no copy32 or snapshotv5. |
| --prefill_small_copy_rows | parser0..32; admitted0 or32 | 0 | Prepare an extra FULL program with bounded padded prefill copies. | generate, serve (pending P; unavailable in S) | engine_rnd_p only; see source | same bits within qualified64K stable1; strict cold/1M admission held | Not parsed by this S base; requires owning track merge | Nonzero requires prefill_prefix_stable1 and every stable1 gate. Additional immutable FULL retains largest physical wide capacity; only padded headless steps with actual rows<=32 select it. Attention-quarter and FFN owner-segment CE copies stop after32program rows; physical offsets, math/owners/reduction and READY/DONE/empty handshakes remain. Ordinary steps exclude this alternative; preparation rejects non-FULL/non-EXL3/non-BF16-CE5 or invalid eight-row bounds (P main.c:908,1380,1980; step_image_programs.c:172; model_v41_program.c:5701). | Frozen18 xc38v4 full306 native/observable PASS:102cold+204warm, all accepted IDs/selected-logit bits equal own cold;102/102 cold controls equal accepted form14. Short append warm ABA362.311/214.198/362.160ms, ratio0.59301(95%0.58865-0.59810); phase345.339/199.033/344.657ms, ratio0.57608(0.56739-0.58491); tool-cycle warm ratio0.99616(0.91396-1.00434). Strict cold gate FAILS: short ratio1.00116886(1.00069573-1.00243898); tool1.00172992,phase0.92088957. Sampling intervals do not bound hardware drift. 1M readiness fails before token0. Manual/default0/HOLD; no active axes, default or legacy-speed parity. Full window exit1 remains separate from completed306 leaves (RND_P.md5Nov13:10,13:28; results/gxc38v4/small_copy_report.json). |
| --anthropic_messages | 0 or 1 | 0 | Enable the bounded Anthropic Messages text and client-tool endpoint. | serve | All otherwise admitted serving recipes | Protocol adapter; no arithmetic selection or exactness qualification | Default off; supported version header and strict request validation required | Reuses native engine, user queues, ownership and cache isolation. Enabled/adaptive thinking uses native effort and an enforced budget; top_k is a shared request control. Unsupported blocks are refused. See docs/ANTHROPIC_MESSAGES.md. | CPU contract and client tests are separate from native-model qualification; no speed claim. |
| --session_fork | 0 or1 | 0 | Copy an eligible same-owner prefix into an independent child session slot. | serve | Qualified stable-prefix TP4_EXL3 recipe | same bits for exact-copy service policy; target-form qualification required | Default off; cache_scope0 and qualified producer capability required | Startup requires cache_scope0. Runtime needs prefix reuse, >1slots, max_tokens>0, healthy engine and no prior attempt. Idle/read-unleased donor: different full namespace, same authenticated hard owner; hints confer no authority. Cold-plan kept>0, reach>=kept, reach-kept<=3968; ordinary state only, never checkpoint-only. Reserve/check real generations/read lease, drain copies before publication; failed/cancelled accepted clone clears destination history and may retry cold only if engine healthy. Backend failure poisons the engine; no aliases. Growing children additionally need P stable1 provenance; no invented snapshot/checkpoint proof (Qfork main.c:947,1377; server_forks.c:19,34,130,205). Latest native92af3e2 requires stable1 for every new cross-session copy; stable0 falls back cold. Existing legacy same-session behavior is retained, not newly qualified bit-exact. | Q92af3e2 is integrated default off. Complete gpq12 fresh24-request legacy fallback supplement passed; gpq11 retains its earlier whole-job failure. New cross-session copies require stable1; donor ownership, namespace, leases and generations remain checked. An integration smoke on main is independently required (MERGE_A21_v4.md). |

Diagnostic environment switches (not axis defaults):

| Environment | Values / default | Scope / exactness | Gate / interaction / evidence |
| --- | --- | --- | --- |
| ANEMONE_DIAG_STATE_DIGESTS | presence; absent | generate, same bits observation | Emits state digests; commands.c near shutdown; unmeasured overhead. |
| ANEMONE_DIAG_DRAFT_STAGES | strtoul stage count; absent=3 | draft-only diagnostic candidate; gate D | Caps draft stages at3; no strict CLI validation; source/model_v41_program.c:4450; unmeasured, target-width/bit invariance not established by this metadata. |
| ANEMONE_DIAG_DRAFT_SKIP_FFN | exactly1; absent=off | draft-only diagnostic candidate; gate D | Omits draft FFN work; stage probes only; source/model_v41_program.c:4451; unmeasured, no qualified serving/verified-bit claim. |
| ANEMONE_DIAG_DRAFT_PROBE | presence; absent | draft-only diagnostic candidate; gate D | Requires a batch join kernel; writes intermediate probes instead of normal draft logits; diagnostic path, unmeasured, no qualified normal serving or automatic axis. |

Current S source is8d6bb95, the clean main5029986 merge after doc errata3762ba0; the S09/f139 frozen inputs
remain historical and unchanged. The 06:23 tcb77 merge note is the source boundary for the six merged cache
controls. P/Q/R scope updates below retain their actual frozen epochs through06:23; P small-copy is pinned
separately to6413c72. S06 target-bit qualification remains corrected at04:38, with N/R attribution limits
at05:12/05:19. Prior main75d3d17 O/HC and main242e6ef N controls remain inherited unchanged.
The product CLI matrix above and its fresh S inventory remain the completeness boundary: component APIs,
packet-only environment controls and CPU arithmetic hypotheses below are not extra accepted Anemone
options and do not increase product CLI row counts. Numerical defaults come from the corresponding fresh
Linux probe, not a newer N packet's initializers or historical usage text.

Current S8d6bb95/main5029986 parses official_hc_rows with range0..8 and preparation-admitted values
0/1/3/4/8, default0. The nonzero runtime form remains original PP THIN only; TP, WIDE, draft, vision
and EXL3 HC composition are unsupported. Source availability is not exact-axis admission.
XR06 closes the corrected carried-input HC replay and observer gates, not complete runtime composition,
logical-state equality, generation or verifier-width invariance. The component results must not be promoted
to an exact-axis recipe. Separately, XR09 qualifies original PP THIN HC0->HC1 short-driver means
C1 12.651->13.951ms and C8 15.419->16.593ms; these are explicit HC runtime observations, not full exact-form
composition or S EXL3 costs. XR12 merged HC captures preserve XR09, while its complete campaign remains
failed on a missing WoA profiler report/export. O core/query and corrected wide numerical
fixtures now pass within their independent epochs. S also parses default0 original_batch,
original_chunk_core, original_dense_staged, original_wide_experts and tp_pipeline_overlap.
The original_queries_piece and original_dense_shards selectors are still absent from S and retain their
separate branch-only rows; neither is silently promoted by this merge. Historical dense-shard device
work remains budget-deferred in the reviewed epoch.

Legacy original TP4_PRO serial depth 2 and PP5 depths 2/5/8 already run on main 7d86c45/S with the new O
switches omitted; they do not require an unmerged O builder/driver port. Existing S original balanced/decode
tuples retain their generated mappings. O09 prefix_reuse=1 warm rates remain historical evidence and do not
qualify S prefix_reuse=0 recipes. New overlap/BATCH/wide/core/query/dense joins still need their own gates.

M scheduling, shared Engram and dense_staged0/8/16/32/64 remain integrated. Earlier S/main242e6ef introduced exl3_tc_product_blocks0..592 and tp_decode_round_once0/1, both default0, and admits
batch_core_once0..3 (then default2) and draft_session_acceptance0..11/default0. Current placement-dependent defaults are below. These are now primary
S options/ranges, not unavailable N rows. Modes10/11 are present in the actual S parser and engine;
their presence does not imply a new active axis recommendation. The product-grid default0 keeps the
148-block grid; nonzero is raw tuning only. Round-once1 changes target arithmetic and is raw research
only: N's own control/repeat C1/C8 serving result is18-25% slower, full-vocabulary quality was deferred,
and no axis selects it. Fresh merged S qualification is separate from those N receipts.

Current S merges verified main39980c7/tcb78 as3d883e5. Omitted EXL3 CLI fields now select
fixed5/policy0 for trueTHIN1slot, or RATE2 for BATCH/configured slots>1, learning1, costs14300/0/690
and core3. Other placements preserve each omitted field's legacy policy1/learning0/costs12700/2200/2700/
core2; explicit fields and API defaults stay unchanged. A sole batch_step_us0 explicitly disables planning;
zero cannot occur in a multi-entry cost list. Automatic prices require omitted policy/costs plus the exact
measured reference/BF16/DSpark5/core2-or3 geometry and canonical rows/verifiers; explicit policies retain
no-automatic-menu behavior. Existing axis tables continue to set their documented explicit controls.

N31/dbd1219, byte-identical ordinary binary to tcb78 (de617b4f), passes both-width references, all code
cells and fullmixed112cross/96self per arm. The maintainer accepted that default bucket under the owner's
faster-identical rule, treating the observed small Fraser shifts as noise. S's older strict zero-margin
S06 prior-pair screen remains failed; neither those observations nor its limits are rewritten. New current-main
controls use tcb78. Existing frozen S10 measurements keep their original e4c188e/5029986 provenance;
profile/axis admission remains separate from raw-default selection. No implicit axis default is chosen.

Existing FIXED0/session0/learn1 plus a BATCH cost menu already permits full available local depth for
unplanned/single-owner flights and joint economic caps for planned multi-owner flights. It is an existing
proposal-policy setting, not a new concurrency selector; draft-only exactness requires gate D. Same23 code C1/C2/C4/C8 rates are
265.99/283.91/364.49/443.63 versus RATE2+menu 240.42/289.76/361.20/438.26 tok/s. Its full mixed control
passes 112 cross/96 self identities. Pure fixed/no-menu wins the identical-prompt code matrix but loses the
recorded diverse mixed subset; RATE2's direct full mixed gate remains unmeasured. No automatic combined
score or universal axis choice follows (packet RND_N.md 5 Nov 02:37).

Controlled S06 THIN result, 5 November, 04:38 CET: canonical fixed policy0 in
gxf_s06_code_single_short passes both full 8/256 selected-record references and 15 C1 MiaAI
native/text/finish checks. Code/prose/structured C1 rates are 230.959/158.954/330.108 tok/s;
cold prefill is 3909 tok/s with cached0; Fraser C1 median/worst is 1379.4/1395.1 ms. These numbers
belong to that fixed0 tuple, not a mixed-traffic/pause or full-capacity qualification.

With canonical tp_head_shards=1/tp_head_stream=1/tp_draft_shards=1/tp_draft_layers=0 unchanged, changing only
draft_policy=0 to1 fails the 30K selected-record gate: all 8 IDs agree, but 6 selected F32 logits differ,
maximum about 7.63e-5. The policy1 selected file MD5 ff1489cffe4194bb52fe36eef08056c9 exactly matches
S04gxf15's older unsharded policy1 output. The earlier joint mismatch must not be attributed to
head sharding. Width/form-dependent target arithmetic versus selected-logit provenance remains
under investigation; this selected-record result is not a full-vocabulary deviation metric.
N05:19 identifies the stored field as target sampler global_logit, not draft confidence. THIN1 versusTHIN8
selection and dense-stream versus eligible multirow head-MMA are possible concrete dispatch boundaries,
but neither head nor sharding is isolated until committed-token normalized BF16 head operands and
kernel/arguments agree. Earlier target forms remain possible causes. R05:12 also retains XR09's original
PP C1/C8 full-F32 invariance; concurrency alone is not a general explanation. The original fixed-M1 head
component is not proved to preserve S's current fixed-EXL3 baseline and supplies no balanced-policy repair.
No current THIN policy1 speed is measured. Policy1 remains an available raw recipe/default, while this
THIN tuple's same-bits-to-fixed-canonical balanced admission is unqualified. No CLI default or axis setting is
changed by this metadata correction. Source: evidence/xf_axes_numerics_v3/thin_current_30k.* and
RND_S/N/R.md 5 November, 04:38 CET. Independent teacher-capture repeats do not turn the later failed
ordinary selected-reference gate or unattempted subsequent arms into a complete numerical-job pass.

P frozen14/4942b8a repairs the private PADDED empty handshake and passes its new/old both-width references
and 68 own-cold native growing-prefix/repeat comparisons. Private allow_empty_payload is not a product knob.
The original xc28 exit1 and corrected high-water audit remain distinct. Later xc36 passes174/174 native
ID/selected-bit mixed/stress comparisons and actual checkpoint restore, but outerexit1 is retained:
its old0 1M-slot control fails model planning before weight load, so no stable1 1M boot/copy was measured.
Own-form114coldABA/actual1M-follow-up/newest-main gates still bound conditional knob/off admission. Owner05:20 accepts
stable1 as an exact-tier cache form for qualified fixed TP4_EXL3/BF16 text, with new references alongside old0;
it is neither original-weight equivalence nor a raw/default/automatic-axis change. P6413c72 now implements
prefill_small_copy_rows, separately branch-only/manual/default0 and absent from frozen14/15 and Q's join.
Its reported host copy-byte savings are not GPU, TTFT or graph-memory evidence. Snapshot-v5 remains a proposal.

Q final06's gpq05 cold-control bracket and the real-client gpq06 are closed within their frozen epochs.
The newer main5029986 merge passes both-width8/256 identity and the two-user stateless Chat/Responses
smoke06:15:58-06:19:13, plus its reported109ASan/12TSan, headers/CLI/default/image/release qualification.
The earlier helper-permission failure happened before the binary ran and remains retained. Current S8d6bb95
now parses prefix_reuse_exact, cache_scope and the four session thresholds with actual defaults0/0/30/1024/256/120;
these are primary S rows, not branch-only P/Q controls. Nonzero exact reuse in this merged source still
requires same-length cold provenance, refuses snapshots/images and refuses a session directory; it does not
include P stable1 growing-prefix or Q session-fork changes. Scope0 retains full namespace equality.
Qfork's latest reported frozen08/8c494b8 joins P7128f76/core7849c23/server239, with full joined host gates
passing06:14; that does not close queued weightless gpq07, child T0/reference/performance or growing-prefix
gates. S still lacks session_fork and P stable/small-copy. Store:false and snapshotv4 are now integrated;
stateful response chains have no selector. Service/cache policy remains orthogonal to four axes
(RND_P/Q.md5Nov06:14,06:21,06:23; merged S main.c:935-946/server.c:66-135,1232).

N's same-binary prefix1 order2->0 code-only warm probe measures the existing prefill_order selector;
its47records/arm retain paired cache classes and T0 text/count/finish. C4/C8 gains do not close M/O's
long-prompt continuity or all-category/mixed gates. Its C1 is one-owner BATCH8, not THIN; frozen24 has no
batch_step_us cost menu. Existing original O gates use prefix0/order0 and do not qualify that EXL3 warm tuple.
O's new TP5 lossless CPU storage census and memory arithmetic plan do not provide CUDA decode, residency,
two-layer/full-model or C1 evidence; no TP5 selector/axis is promoted. The isolated original BATCH8/1 research
uses existing batch_rows/verifiers and must retain common-prefix/maximum-segment PRO-tail guards; current
original_batch defaults0 and legacy48/8 admission stay unchanged (RND_N/M05:17,05:32; RND_O05:27,05:41).

The temporary N prefix-alignment option was retired in 2af7162 and has no catalog row or axis setting.
M's unqualified prefill product-grid proposal is absent from the clean source. N's device-confidence mask
and G07 shared-slice redistribution have no implemented selectors. M25's branch-only selector is documented
above; no active tuple selects it. Neither numerics=official
nor a complete automatic prompt-to-decode reference policy exists. None is silently added by this catalog.
The G0/T/B/F implementation gates and D target-bit evidence qualifier are part of each row's stated contract.

Track R components and packet diagnostics (not product CLI options):

| Form / control | Values / default | Actual availability and admission | Evidence / scope |
| --- | --- | --- | --- |
| HC source-shape / staged APIs | Actual R and reference F each 1/3/4/8; no separate staged runtime default | anemone_official_hc_shape_prepare retains dot47 at F1 and source owners/folds at F3/4/8. Staged replay is a component alternative, not a serving flag. The distinct official_hc_rows selector is the restricted S runtime row above; complete composition remains held. | XR06 passes 64 crossed direct/staged complete HC sites and 48 fixed-F HC prefix comparisons; complete runtime/speculation remains open. FIXES.md:23-26; engine_rnd_r/source/official_hc.c:56. |
| Two-pass F1 HC API | Actual R1/3/4/8, fixed F1; 4512 scratch bytes per actual row; no runtime selector/default | anemone_official_hc_partitioned_prepare, commit b2c9a22. Caller owns disjoint F32 scratch and both ordered launches; old direct form retained. No model/runtime caller or automatic axes. | XR07 complete original-input attention/FFN, direct preservation, all12 fixed-F1 prefixes and four sanitizer gates pass on CMP0/PRO2. Direct->two-pass attention us CMP R1/R8 11.344/18.464->8.728/18.216; PRO8.190/8.204->6.140/10.234. PRO R8 regresses; no universal faster/free or C1/C8 integrated-step claim. evidence/xr07/round2/xr07_components_review_v2.json retains failed v1; official_hc_partitioned.c:55; FIXES.md:24. |
| Source-shape BF16 norm API | Width 512/1280/5120; actual R and reference F each 1/3/4/8; no serving selector/default | anemone_official_norm_shape_prepare. Reference shape and actual extent are independent; F32 pooled compressor/indexer norms and arbitrary widths are outside admission. | XR06 crossed norm/prefix/sanitizer controls pass. CMP5120 R1/F1 warm 5.464 us versus legacy 3.312; PRO 4.109 versus 2.058. Separate-process component costs, not C1/C8 step costs or a no-loss claim. official_norm.c:32; FIXES.md:27. |
| Independent BF16 RoPE API | Width 64 or 512, heads 1..64, rows 1..8, inverse 0/1; no serving selector/default | anemone_official_rope_prepare, commit 498b2ec; identical or disjoint input/output only. Retains explicit BF16 assignment and the recorded FMA form. | Source admission exists; XR06 qualifying rotation cells remain fixture-scoped, not universal hidden-F32/runtime order proof. Separate component and joined-model costs/gates remain distinct. official_rope.c:30; FIXES.md:29-30. |
| Fused window composition API | Width 512; actual R and reference F each 1/3/4/8; no serving selector/default | anemone_official_window_prepare, commit 239d604: source norm owners/tree, BF16 weighted norm, RoPE/BF16 assignment, group32 FP8 roundtrip. Logical BF16 output; no runtime FP16-cache scatter or model caller. Independent passes retained. | XR08 all32 crossed cells,24 fixed-F prefixes, three-pass/fused output/mean/inverse preservation and eight sanitizer captures pass. Separate->fused us for R1/F1,R8/F1,R1/F8,R8/F8: CMP6.336/6.800/7.264/7.744->4.264/4.288/6.152/6.136; PRO4.108/4.157/6.141/6.141->4.094/4.092/4.109/4.109. Excludes legacy producer/cache scatter; no full-step/free claim. evidence/xr07/round2/xr07_components_review_v2.json; official_window.c:28; FIXES.md:30. |
| Independent FP8 roundtrip API | BF16 group32 E4M3/E8M0 roundtrip; recorded R1/3/4/8 and separate/in-place forms; no model selector/default | anemone_official_roundtrip_prepare. Window KV and inverse-RoPE pre-WoA barriers are different composition sites; EXL3 weight decoding is a separate operation. | Original-input component/codec and sanitizer scopes pass. Runtime insertion and cache/verifier composition remain unqualified; inserting a locally correct barrier need not improve full-model distance. FIXES.md:32-33. |
| TRACK_R_LAYER0_QR_REFERENCE_ROWS | 0/1/3/4/8; absent or 0 retains old diagnostic path | Packet capture_layer0_window_v1.c only; no anemone product flag. Original PP THIN serial L0 observer; independent norm shape, not automatic whole-network repair. | XR06 QR0/1/8 stored L0 values agree; the actual saved PP query path already matches the original prompt fixture. No measured repair benefit. capture_layer0_window_v1.c:689; RND_R.md 5 Nov 02:46. |
| TRACK_R_LAYER0_OUTPUT_ROUNDTRIP | 0/1; absent or 0 off | Packet observer only; optional attention-output roundtrip before grouped WoA. No serving selector. | XR06 codec/upstream-preservation gates pass, but original-prompt logit RMS changes 0.539166->0.882317 while upstream core/window input is wrong. Not a complete model improvement or generation proof. capture_layer0_window_v1.c:684; RND_R.md 5 Nov 02:46. |
| TRACK_R_LAYER0_WINDOW_REFERENCE_ROWS | 0/1/8; absent or 0 off | Packet observer only: original PP THIN serial L0, one row/session0, positions0..7; retained norm/RoPE with diagnostic host transfers. No runtime selector. | XR11 source/control/codec gates pass. XR12 HC1+windowF1+windowRT1+outputRT1 matches actual cached1 L0 residual163840/163840 and carried-pre/FFNnorm endpoints. Not every L0 operator or all40/logits: those still differ. The separate window BF16-to-FP16 mapper still admits only8704 exactly representable words; all-finite private-core identity below does not widen this cache domain. capture_layer0_window_v1.c:703; window_join_impl.h; exact_half_carrier.h; RND_R.md5Nov05:30. |
| TRACK_R_LAYER0_WINDOW_ROUNDTRIP | 0/1; absent or0 off;1 requires nonzero WINDOW_REFERENCE_ROWS | Packet observer only; adds group32 roundtrip to the preceding window diagnostic. No serving selector, production default or automatic axes. | XR11/12 joined fixture observations pass in the explicit preceding scope; outputRT erases the small private-core difference on this fixture. Complete XR12 exits1 because WoA profiler report/export is absent despite15 data/sanitizer processes exiting0. Cache rollover, arbitrary positions, prefill/decode, verifier and speed admission remain unqualified. capture_layer0_window_v1.c:712; evidence/xr12/round2/review; RND_R.md5Nov05:30. |
| Actual M1 head arithmetic component | Actual R1/3/4/8; output cuts positive multiples4 up to129280;16 strided chains and8/4/2/1 halving; no runtime selector/default | anemone_official_head_prepare and anemone_official_head_reused_prepare are present in main75. Independent direct/reused GPU components, no model/runtime head caller. Product domain: finite BF16 input/weights, normal F32 or signed-zero products and finite partials. Actual M1 and auxiliary M8 remain different targets; no R>8. | XR11 all20 records/card, original-input/direct-reused full-vocab/cuts,12 fixed-M1 prefixes,4 cut joins and4 sanitizer captures pass CMP0/PRO2; CPU9-head/1163520-value proof retained. Direct R1/R8 ms CMP1.497/11.212,PRO0.839/6.665; reused2.653/3.198,1.386/1.534. R8 improves/R1 regresses; generic dense control is not selected serving baseline. CPU host tests overlap lease04:31:45-04:36:43. No integrated step/share, S fixed-EXL3 preservation or verifier/rollback proof. official_head.c:44,99; evidence/xr11/round2/xr11_head_review_v1.json; COSTS.md:93. |
| Existing BF16 core assignment before inverse RoPE | Existing arithmetic; no new flag/default/operator | core_output_bits already rounds the normalized core value to BF16, widens that rounded value for inverse rotation, then rounds the rotated result again. This is not a missing barrier to add. | XR08 source/kernel audit and XR12 private-output replay retain the assignment while pure/inverse outputs still differ in4/5 BF16 values on the fixture. No incremental work or separately isolated epilogue cost; full core/cache/verifier invariance remains unqualified. S source/kernel_attention_core.cu:547-575; R FIXES.md:32/RND_R.md5Nov06:09. |
| Private core-output / signed-zero identity observer | Packet TRACK_R_LAYER0_CORE_SHADOW0/1; absent or0 off; no serving selector | Replays unchanged values into private output; normal epilogue plus positive/negative-zero inverse rotations reconstructs source BF16 signed zeros. Diagnostic original PP L0/R1/session0 scope, not a new cache carrier or target operator. | XR12 native/memcheck/racecheck CMP0/PRO2 recover all65280 finite BF16 patterns,65536checks/run. Private core differs4/262144 BF16 beforeinverse,5 afterward; outputRT erases this fixture's difference. WoA12/65536 differs,WoB0/40960; raw router1537/3072 F32 differs while experts/routed/shared/residual BF16 match. Endpoint equality is not full-L0/model proof; campaign exit1 and unmeasured WoA geometry retained. capture_layer0_core_v1.c:736; core_identity_card_test.c; recover_core_identity.py; evidence/xr12/round2/review; RND_R.md5Nov05:30. |
| Cached-M1 router projection component | Fixed384x5120; actual R1/3/4/8; BF16 input/weights,F32 output;32 chains,16/8/4/2/1 halving; no runtime selector/default | anemone_official_router_projection_prepare, final8a6315f after3dbda8d/a6b8686. Source API/kernel only, no gate/expert/model/runtime caller; separate from head16-chain arithmetic. Normal F32 or signed-zero products and finite sums required. Original BF16 weight values are checked before diagnostic conversion; EXL3 operand/domain/runtime proof absent. | CPU6264-form search matches3072 original cached-L0 router F32 values; four builds,8 focused host checks,184headersx3, module/release audits reported. XR13 frozen8a6315f/85inputs awaits GPU component/prefix/sanitizer/cost and guarded L0 replacement; no serving speed or automatic exact-axis promotion. official_router_projection.c:28; kernel_official_router_projection.cu:1; RND_R.md5Nov05:30. |
| TRACK_R_LAYER0_ROUTER_REFERENCE | Packet0/1; absent or0 off;1 requires TRACK_R_ROUTER_WEIGHT_REFERENCE=path to exact384x5120 BF16 matrix | Packet capture_layer0_router_v1.c only; no product CLI option or runtime selector. Original PP THIN L0/R1/session0/positions0..7 only; mutually exclusive with CORE_SHADOW. Retains original input/output, then replaces raw logits with the independent32-owner product only after all1966080 GPU F32 weights equal the frozen BF16 widenings. File length must be exactly3932160bytes; nonexact/nonfinite/different weights refuse with no rounding fallback. | Frozen XR13 control/repair/repeat, device prefix/sanitizer/cost and guarded join remain pending at06:12. Host transfers/weight scans are diagnostics, not C1/C8 serving costs or full gate/later-layer/width/cache/rollback proof. Held exact-tier dependency only; EXL3 implementation/domain/runtime proofs absent. capture_layer0_router_v1.c:793-808; router_join_impl.h:11,36; FIXES.md:41; RND_R.md5Nov06:09-06:12. |
| FP8 K32 RN/RZ and alignment hypotheses | CPU arithmetic forms only; no device API or runtime selector/default | reconstruct_fp8_rz.py and search_fp8_alignment.py investigate original FP8 accumulation. EXL3 decoded/codebook products need their own implementation and domain proof. | 317440 saved BF16 outputs match selected hypotheses, while guarded F32 matrices still have counterexamples. Do not label a BF16-only match universal MMA equivalence or a runtime repair. FIXES.md:35-37. |

XR17b update, 5November10:48: packet CORE_FORM7, LAYER0_MISSING_RECORDS1 and LAYER0_WOA261
are diagnostics/guarded joins, not product CLI/defaults. The early independent reader10:42:37 passes
all six sparse-core F32 grids, every observed L0 boundary/residual163840/carried-pre32 and routed/shared
expert/quantization records on eight serial cached-one positions. The BF16 WoA join closes12/65536
differences; its privately staged64MiB exact BF16 weights and transfers are diagnostic costs. Complete
40-layer/logit/generation/verification/cache/serving proof, composed cost and EXL3 runtime port remain open.
Model26 differs from native PRO in952/16777216 intermediate and3/65536 final F32 values despite BF16
equality; a new CPU operand-anchor hypothesis still needs device proof. These later scoped positives
supersede earlier pending L0 observations above, not their historical source or overall-failure status
(R FIXES/COSTS.md and RND_R.md5Nov10:48). No exact-axis admission or free claim follows.

Original balanced-distance diagnostic: O's actual-driver xo03 PP5 and TP4_PRO cached-one captures
use the same eight teacher IDs as xr06_serial_fixture_v2, no drafts, native PRO, original pack517258440704B,
reference CMP/linear order and precision profile. Both placement captures are literally equal. Aligned
xo16_serial_reference_distance_v1.json compares1,034,240F32 values against R's original hybrid parentPRO2 /
unchanged sparse coreCMP0 source:1,034,238different, maximum4.2760492563/RMS0.5683911806,
KL(reference||candidate) mean0.0407171825/max0.1449727082, argmax8/8. Source manifesteef6679cfa643ad6cf8170467e28467981d5785ecb4d8b3405da1e019ebf29cd;
O source75 replays equal those baseline captures. These are measured eight-teacher-position diagnostics,
not allPRO reference, corpusPPL, generated-sequence agreement, currentS10 full-model distance or
official-exact admission (RND_O/S.md5Nov09:26, R10:25).

Latest branch qualifications (5 November15:24 checkpoint): P current-main39 complete114 cold arms
pass the unchanged no-loss reader; historical source15 strict THIN failure remains. Prefix-only19 is a
knob-default0 integration request with standalone CLI/resulting main references still required; full P80
copy32/v5 must not be merged. Copy32 full306native passes and short/phase warm TTFT falls40.7%/42.4%,
but strict cold and actual1M CUDA graph OOM gates fail. V5 has actual LOAD, not a qualified authenticated
checkpoint/fork composition. Q11 passes both reference families,64canonical requests and actualpi;
fallback times out before its last readiness, so whole11 staysfailed and all24fresh fallback controls
await12. Earlier10 legacy selected-logit discrepancy remains. New copies require stable1 plus hard owner.
N M25 remains NO MERGE and absent from isolated pure37. Order3 completes255paired responses plus both-width references on frozen35,
but strict cold continuity/Fraser/warm gates FAIL despite warm code gains; new-default source/three-pass proof remains open;
protected canonical/exact-prefix/fork joins remain excluded. O corrected CSF27/guardedjoin11777a6 host
factories now pass; earlier25factory failure remains historical. CSF+third refuses before I/O/planning;
physical fit, actual device ring/head/draft/numerics and serving speed remain unmeasured.
R XR20 has nineteen passing component/sanitizer phases, but overall exit1: H64 PRO replay refuses141312B
shared memory before launch. Engram128 matches327680BF16 values/card, FP4 division/midpoint all101calls/
262656values/form/card, compressor32-owner24576F32/card; component scope only, all new costs unmeasured.
Complete L1/later layers/head/generation/reference-axis composition remains held. R later resident XR22
is queued after its owner-only resume; CPU audit finds15/10485760BF16 query values change under FP16
conversion. This source-domain observation does not qualify a live operator, serving knob or fast form.
S xf19 both raw-main fresh M->mixed windows now complete/independent-readerPASS, cards freed15:20:43.
BATCH/THIN C1 whole-window128.661/122.756tok/s and C8burst181.848/122.622;112cross-family
request/text/reason/native-count/finish signatures agree. The two-recipe90%mixed screen passes, but
SSE repeat+4.747ms has different preceding traffic and is not a waived zero-latency gate or axis admission.
Actual numbers/scopes are generated in the axis appendix; numerical settings/holds remain unchanged.
Owner13:15 pause applies: finish already queued qualification and current code/test commit, then pause;
no new S experiment is admitted. No branch outcome automatically selects an S tuple.

These R forms map only to held exactness=exact dependencies for compatible non-vision categories,
concurrency and focus values. Potential EXL3 reuse requires its own operand/domain, implementation and
runtime proofs and the label **reference arithmetic (EXL3: not official-exact)**. Balanced keeps today's
arithmetic; R supplies no qualified fast form. C8 concurrent requests do not imply actual R8 or reference
F8. The canonical cached-one-token target and prompt-eight target remain distinct. No full C1/C8 step
cost, verification invariance, generation identity, corpus PPL or automatic axis admission is inferred
from these components, packet diagnostics or CPU models.

<!-- generated-axis-table -->
# Axis table

Generated from `tools/cli/axes.json` by `tools/cli/generate_axes.py`.
Registry SHA256: `868a1b97251f6577058071a0a978e504fc93e5839f102c28ab54577b026d4fa8`.

The four selectors apply to generate and serve. There are no selectable names or aliases.

| Axis | Values | Meaning |
| --- | --- | --- |
| `--category` | code / prose / structured / vision / balanced | Workload kind; balanced is beneficial across the applicable measured workloads. Scope and missing mixed-user gates remain explicit per strategy. |
| `--exactness` | exact / balanced / fast | Exact requires the pinned original arithmetic, state, sampling and execution contracts. Balanced admits scoped zero-added-error or empirically established noise; fast also permits a separately qualified accuracy/speed tradeoff. Identical admitted non-lossy balanced plans are available at fast without adding arithmetic changes. No noise tolerance is inferred. |
| `--concurrency` | single / multiple / balanced | C1 workload, separately qualified C2-C8 workloads, or a measured compromise across both; this selector is not session_slots or a numeric batch size. |
| `--focus` | balanced / ttft / decode / prefill | Measured compromise, first-token latency, decode rate, or prefill rate. |

| Exactness tier | Error class | Current admission |
| --- | --- | --- |
| exact | Original weights: official-reference arithmetic on the admitted PP THIN serial M1 scope. EXL3: reference arithmetic (EXL3: not official-exact). | GPU split PP5: the original_reference_serial_m1 recipe; GPU split TP4_PRO: holds exact; GPU split TP4_EXL3: the exl3_exact_shaping recipe. |
| balanced | Each GPU split composes its base (its measured route's knobs over the raw engine defaults: draft-only, and on TP4_EXL3 relaxed numerics with the BF16 wire) with the measured knob set of each selected axis value; every set is same-bits or draft-only, so balanced adds no lossy arithmetic. Exact/EXL3 contracts are unchanged. | Current-arithmetic knob sets per GPU split; scope below. |
| fast | A fast tuple equals its balanced tuple while the fast set is empty. Additional lossy arithmetic is opt-in and needs paired numerical and whole-workload receipts. | Additional lossy routes are held: no admitted empirical error budget or complete paired numerical/whole-workload receipts. This hold does not prohibit identical non-lossy balanced plans. |

Each GPU split has its own table below. Original official-reference arithmetic is scoped to its
admitted serial M1 recipe. EXL3 uses the label **reference arithmetic (EXL3: not official-exact)** when
compatible forms are qualified.

- --placement picks the GPU split; without it the split is PP5, the engine default. An unset axis takes its balanced value, so a generate or serve without axis flags runs the split's all-balanced row: its base. --axes=off runs the raw engine defaults with nothing inserted.
- Original versus EXL3 is a weight-family admission condition, not a fifth preset axis. The runtime uses explicit placement or the existing PP5 default.
- Explicit low-level knobs override inserted settings; ordinary validation still refuses invalid combinations. Overrides change the measured recipe.
- All balanced choices retain current HC/current forms and add no new lossy arithmetic. A separate R component result is not a complete S official-error measurement.
- Context and sessions are the user's: the raw defaults are 1M context per session (--context_tokens=1048576) and, on serve, --session_slots=max (the most sessions of that context that fit the cards, no cap; refused when none fits, the context never shrunk), and no axis set pins either (9 Nov); only the exact reference recipes keep their own geometry (one session; PP5's at 512 tokens). Startup prints how many sessions fit at the context on every serve. Concurrency=balanced adds nothing. On TP4_EXL3, concurrency=multiple's BATCH set needs exactly --session_slots=8 on serve and otherwise runs the concurrency=balanced row. Original single-user decode is fastest on GPU split TP4_PRO at pipeline depth 2: pass --placement=tp4_pro. On PP5, concurrency=single is --rows=6 at the default depth 5 (r7c g3).
- Balanced-code selection must maximize code among feasible choices: non-code margins and no-worse default Fraser median/worst/pause comparisons require same-epoch measurements.
- The table holds one row per GPU split and full tuple. A partial selection fills each unset axis with balanced (concurrency with single under exact).
- Seven-bucket entries are targets for the recorded scope; every unlisted bucket and every missing accuracy/capacity gate remains unqualified.
- PP5 and TP4_PRO recipes yield the draft block, depth, policy and session acceptance (and TP4_PRO's rows) to category=code and focus=decode: the recipe's measured values (fixed policy 0, which is no Original CLI default) run wherever those sets give none. TP4_EXL3 sets take the CLI draft policy (fixed for one session, as a serve at the default 1M runs, where one fits: commands.c single_session_defaults; rate on BATCH and for several sessions) and the BATCH core default (commands.c DEFAULT_BATCH_CORE_ONCE, 3), so a change to those defaults changes those rows; explicit user controls still win.
- Legacy original TP4_PRO and PP5 routes existed in main7d86c45. The current table also pins newer O/HC controls to zero and therefore requires the merged main75/S parser. PP5 layouts remain single argv items with embedded spaces; generated settings use newline separators, never shell parsing. S07 cold-prefix measurements are source-scoped separately from inherited O warm results. Generate keeps its one-slot default and pipeline2; serve selects one serial TP slot or eight PP5 slots at measured depth8.
- Exactness=exact selects the split's reference recipe as a second baseline: the other three axes add nothing and concurrency must be single or unset. PP5's is the PP THIN serial M1 recipe and TP4_EXL3's the exact-shaping set (exl3_exact_shaping: exact numerics, the reference order and the FP32 wire; reference arithmetic, EXL3: not official-exact); TP4_PRO holds exact. Explicit vision remains held. Recipe arithmetic/placement overrides must retain the admitted exact values.
- Exactness=fast has an empty set on every split, so a fast tuple equals its balanced tuple, with the same argv and exclusions. A lossy knob could enter only the fast set, with admitted receipts.
- Retained route proposals below are metadata only: no argv is inserted from them. Each proposal names all four axes, constraints, old fallback and missing gates. An explicit low-level knob from an unintegrated branch is not thereby accepted by this binary.
- The banked PP5/session6 route remains raw-only, with its lower TTFT and numerical scope retained. No axis value enables banking.
- Each knob belongs to one axis per split (generate_axes.py refuses a second owner), except a recipe's yielded fallback: a yielded knob is set by its recipe and exactly one other axis, whose value wins where it gives one. Session acceptance and the other measured modifiers enter only after their per-split confirmation: focus=decode carries arm 1 (session acceptance 6) on PP5 (r1m) and TP4_PRO (r5s); every other Original non-exact row takes the CLI value 0. No-axis defaults and exact/EXL3 contracts are unchanged.
- --load_piece_mib (the load's read size: the PP piece and staging slot, the TP host arena's chunk) changes load time only, with the same bits. It belongs to the focus axis on every split with every value empty (balanced, ttft, decode and prefill set nothing): its measured value comes from the storage profile's load section (anemone autotune --stage=load), and the knob wins over the profile, the profile over the default (0: 16 MiB).
- --engram_io_threads, --engram_spin_us and --engram_io_pin (step I/O's threads a copy, completion spin window and pinning, active only while Engram tables are read from disk) change I/O time and CPU only, with the same bits. They belong to the focus axis on every split with every value empty (balanced, ttft, decode and prefill set nothing): serving A/B found the step-I/O variants within noise in tok/s at C1-C8 and in TTFT, so no value has a measured gain to map. Their measured values come from the storage profile's engram section (anemone autotune --stage=engram), and the knob wins over the profile, the profile over the default (2 threads a copy, 200 us, pinned). A per-value split (single from D1, multiple from D2-D8, ttft from P1 and P2) enters only after a serving gain at that value.

## GPU split tables

`--placement` picks the GPU split; without it the split is PP5, the engine default. Each selected axis
value adds its knob set to the split's base. An unset axis takes balanced, so a run without axis flags
runs the split's all-balanced row, its base. No set pins the context or the slots: the raw defaults
serve as many sessions of 1M context as fit. `--axes=off` inserts nothing. Each knob belongs to one axis
per split. Under exact, the split's reference recipe is the whole configuration. Explicit knobs win over
every set. The last three columns derive from each set's own settings (`default`: the set leaves that
knob alone).

### GPU split PP5

Weight family: original. Exact: the reference recipe `original_reference_serial_m1`, with concurrency single or unset. Base (from `original_pp_depth8`, for each knob no selected set gives): `--card_order=0,1,3,4,2` `--rows=8` `--wide_rows=1024` `--prefill_cone=1` `--step_graphs=1` `--speculative=2` `--draft_positions=5` `--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4`. The knobs PP5's measured routes share (the sweep's no-axis route gq_rows6_on_s6_t85_v1 off_a, original_pp_depth8 and r7c g3), read from original_pp_depth8: pipeline depth, the draft policy, its costs and acceptance belong to the axes; the slots, prefix reuse, prefill order and batch are the measurement protocol's or the concurrency sets' (8 Nov).

| Axis | Value | Knob set | Evidence | GPU split (`--placement`) | Pipeline depth (`--pipeline_steps`) | Serving slots (`--session_slots`) |
| --- | --- | --- | --- | --- | --- | --- |
| category | code | `--draft_block=7` `--draft_positions=7` | Depth 7 on PP5 at pipeline depth 8 with 8 serving slots, tcb88 same-binary ABBA, code, identical outputs in 47 streams: with focus=decode's arm 1 (r4m) C1/C2/C4/C8 171.1/248.0/345.8/391.0 -> 190.9/251.5/366.5/417.9 tok/s (+11.55/+1.43/+5.97/+6.88 %), TTFT p50 flat; alone (r3m) +2.21/-1.74/+0.86/-1.62 %, the C2 and C8 losses inside the arms' spread (10.20/5.14 and 0.02/6.23 %). Code only: no other genre was measured. | PP5 | default | default |
| category | prose | (empty) | - | PP5 | default | default |
| category | structured | (empty) | - | PP5 | default | default |
| category | vision | HELD | Vision-axis selection awaits a vision benchmark and applicable runtime/identity evidence. | PP5 | - | - |
| category | balanced | (empty) | - | PP5 | default | default |
| exactness | exact | `original_reference_serial_m1` (reference recipe) | The reference recipe's evidence: see Knob values. | PP5 | 1 | 1 |
| exactness | balanced | (empty) | - | PP5 | default | default |
| exactness | fast | (empty) | - | PP5 | default | default |
| concurrency | single | `--rows=6` | r7c g3: PP5 at the default pipeline depth 5 with 8 serving slots and 6 rows, the fastest C1 decode arm in every genre (+1.5-1.6 % over g1), outputs identical (decision G). The measurement's --session_slots=8 is not inserted (9 Nov: context is the user's choice and sessions are as many as fit): the raw default --session_slots=max keeps the multi-session pipeline depth 5. | PP5 | default | default |
| concurrency | multiple | `original_pp_depth8` without `--prefix_reuse`, `--session_slots`; yields `--draft_block`, `--draft_positions`, `--draft_policy`, `--draft_session_acceptance` where another set gives them | PP5 at pipeline depth 8 with 8 serving slots: S10 gxf_s10_original_multiple, C1-C8 in three genres, against depth 5 in S07 (recipe original_pp_depth8). --prefix_reuse=0 pinned the measurement protocol and is dropped (r6a: no cell loses beyond spread). The recipe yields its draft block, depth, policy and session acceptance (block 5, depth 5, fixed policy 0, acceptance 0) to category=code and focus=decode: its measured values run wherever those sets give none. The fixed policy is no CLI default (the Original placements default to the cost-priced depth), so a dropped policy would change every other row. The recipe's --session_slots=8 is dropped too (9 Nov): the row serves as many sessions as fit at the context (the raw default max: 6 at 1M on v0.0.4-alpha, measured 17 Nov 2026 by a plain default serve), at pipeline depth 8; unmeasured above and below 8 until U6. | PP5 | 8 (generate 2) | default |
| concurrency | balanced | (empty) | - | PP5 | default | default |
| focus | balanced | (empty) | - | PP5 | default | default |
| focus | ttft | (empty) | - | PP5 | default | default |
| focus | decode | `--draft_policy=1` `--draft_session_acceptance=6` | Arm 1 (cost-priced draft depth, session acceptance 6) on PP5 at pipeline depth 8 with 8 serving slots, tcb88 same-binary ABBA r1m, identical outputs in 141 streams, TTFT unchanged: code C1/C2/C4/C8 +4.69/+8.87/+1.74/+2.44 %, prose +3.83/+3.32/+1.72/+1.74 %, structured +0.13/-0.58/-0.05/-0.10 % (within spread). Decode only: the same-binary rule takes the faster identical path. | PP5 | default | default |
| focus | prefill | (empty) | - | PP5 | default | default |

### GPU split TP4_PRO

Weight family: original. Exact: held. Base (from `original_tp_serial2`, for each knob no selected set gives): `--card_order=0,1,3,4,2` `--rows=8` `--wide_rows=1024` `--prefill_cone=1` `--step_graphs=1` `--speculative=2` `--draft_positions=5` `--draft_policy=0` `--pipeline_steps=2` `--original_wide_experts=0`. TP4_PRO's measured single-user recipe (pipeline depth 2, 1 serving slot) without its measurement-protocol pins: concurrency=single is then empty (8 Nov).

| Axis | Value | Knob set | Evidence | GPU split (`--placement`) | Pipeline depth (`--pipeline_steps`) | Serving slots (`--session_slots`) |
| --- | --- | --- | --- | --- | --- | --- |
| category | code | `--draft_block=7` `--draft_positions=7` | Depth 7 on TP4_PRO at pipeline depth 2 with 1 serving slot, tcb88 same-binary ABBA, code C1, identical outputs in 5 streams: alone (r3s) 201.3 -> 210.9 tok/s (+4.80 %, TTFT p50 104.1 -> 104.4 ms); with arm 1 (r4s) 224.9 tok/s (+11.02 %, TTFT flat). With focus=decode's --rows=6 both are clamped to 5 (rows - 1), which is r5s; unclamped (r5c) the stack lost 10.2 % to r5s. Code only: no other genre was measured. | TP4_PRO | default | default |
| category | prose | (empty) | - | TP4_PRO | default | default |
| category | structured | (empty) | - | TP4_PRO | default | default |
| category | vision | HELD | Vision-axis selection awaits a vision benchmark and applicable runtime/identity evidence. | TP4_PRO | - | - |
| category | balanced | (empty) | - | TP4_PRO | default | default |
| exactness | exact | HELD | GPU split TP4_PRO has no reference recipe; the exact serial M1 reference runs on GPU split PP5. | TP4_PRO | - | - |
| exactness | balanced | (empty) | - | TP4_PRO | default | default |
| exactness | fast | (empty) | - | TP4_PRO | default | default |
| concurrency | single | (empty) | - | TP4_PRO | default | default |
| concurrency | multiple | HELD | GPU split TP4_PRO has no multi-session recipe: its measured recipe serves one slot. GPU split PP5 serves C2-C8. | TP4_PRO | - | - |
| concurrency | balanced | (empty) | - | TP4_PRO | default | default |
| focus | balanced | (empty) | - | TP4_PRO | default | default |
| focus | ttft | (empty) | - | TP4_PRO | default | default |
| focus | decode | `--draft_policy=1` `--draft_session_acceptance=6` `--rows=6` | Arm 1 (cost-priced draft depth, session acceptance 6) with 6 rows on TP4_PRO at pipeline depth 2 with 1 serving slot, tcb88 same-binary ABBA r5s, C1, identical outputs in 15 streams: structured 285.1 -> 296.0 (+3.82 %), code 203.2 -> 232.3 (+14.31 %), prose 148.1 -> 153.4 (+3.54 %) tok/s; TTFT p50 +12 to +27 ms, the cost of 6 rows (r2s), so decode only. | TP4_PRO | default | default |
| focus | prefill | (empty) | - | TP4_PRO | default | default |

### GPU split TP4_EXL3

Weight family: exl3. Exact: the reference recipe `exl3_exact_shaping`, with concurrency single or unset. Base (from `thin_fixed`, for each knob no selected set gives): `--numerics=relaxed` `--tp_wire_bf16=1` `--rows=8` `--wide_rows=2048` `--prefill_cone=1` `--pipeline_steps=2` `--step_graphs=1` `--speculative=2` `--draft_positions=5` `--exl3_tc_moe=1` `--tp_head_mma=1` `--tp_trunk_mma=1` `--tp_ordered_wide=1` `--tp_head_shards=1` `--tp_head_stream=1` `--tp_draft_shards=1` `--draft_learned_acceptance=0` `--draft_session_acceptance=0` `--draft_base_us=12700` `--draft_tail_us=2200` `--draft_row_us=2700` `--exl3_tc_product_blocks=0` `--tp_decode_round_once=0`. TP4_EXL3's measured THIN recipe (one serving slot, pipeline depth 2, 2048-row prefill) without the knobs category and focus own and the protocol pins: concurrency=single is then empty (8 Nov).

| Axis | Value | Knob set | Evidence | GPU split (`--placement`) | Pipeline depth (`--pipeline_steps`) | Serving slots (`--session_slots`) |
| --- | --- | --- | --- | --- | --- | --- |
| category | code | `--draft_policy=0` | Fixed draft prices on BATCH: S10 batch_fixed against batch_adaptive gives code C1/C2/C4/C8 265.9/296.8/377.4/469.5 against 224.2/278.3/359.3/433.0 tok/s, while prose loses, so code only. On one-slot THIN policy 0 is also the CLI value. The BATCH core is left at its CLI value 3 on every BATCH tuple (isolated core 2/3 text gates: code/prose C8 +0.53/+1.12 %). | TP4_EXL3 | default | default |
| category | prose | `--batch_step_us=19500,21500,28600,30000,32000,39000,46000,58000`; only with concurrency=multiple | BATCH step prices (batch_adaptive's menu; generate's one price) with the CLI's rate policy on BATCH, as every prose and structured BATCH tuple ran before the per-split tables. Only with concurrency=multiple: THIN and the raw split have no BATCH programs to price. Per-split x3 (tcb88, code/balanced/multiple/decode, the menu with fixed prices against none): C4 -6.57 %, C8 -6.40 % aggregate decode beyond spread, TTFT flat, outputs identical, so code keeps no menu and category=balanced holds none (where no policy is given, the CLI adds this same menu to a measured BATCH plan by itself, commands.c). | TP4_EXL3 | default | default |
| category | structured | `--batch_step_us=19500,21500,28600,30000,32000,39000,46000,58000`; only with concurrency=multiple | BATCH step prices (batch_adaptive's menu; generate's one price) with the CLI's rate policy on BATCH, as every prose and structured BATCH tuple ran before the per-split tables. Only with concurrency=multiple: THIN and the raw split have no BATCH programs to price. Per-split x3 (tcb88, code/balanced/multiple/decode, the menu with fixed prices against none): C4 -6.57 %, C8 -6.40 % aggregate decode beyond spread, TTFT flat, outputs identical, so code keeps no menu and category=balanced holds none (where no policy is given, the CLI adds this same menu to a measured BATCH plan by itself, commands.c). | TP4_EXL3 | default | default |
| category | vision | HELD | Vision-axis selection awaits a vision benchmark and applicable runtime/identity evidence. | TP4_EXL3 | - | - |
| category | balanced | (empty) | - | TP4_EXL3 | default | default |
| exactness | exact | `exl3_exact_shaping` (reference recipe) | The reference recipe's evidence: see Knob values. | TP4_EXL3 | default | 1 |
| exactness | balanced | (empty) | - | TP4_EXL3 | default | default |
| exactness | fast | (empty) | - | TP4_EXL3 | default | default |
| concurrency | single | (empty) | - | TP4_EXL3 | default | default |
| concurrency | multiple | `batch_adaptive` without `--draft_policy`, `--batch_core_once`, `--tp_draft_layers`, `--prefix_reuse`, `--batch_step_us`, `--session_slots`; on serve only at exactly `--session_slots=8` (otherwise the concurrency=balanced row) | BATCH, eight serving slots, measured row/verifier menu and step prices: S10 gxf_s10_adaptive_multiple (recipe batch_adaptive). Draft policy (CLI 2 on BATCH) and the step prices belong to category (the CLI adds the step menu to a measured serve BATCH plan by itself; generate's BATCH argv takes no price on category=balanced, a known difference from tcb88's 19500 accepted at the stage 3 review (D4), with no generate speed claim), layer drafting to focus; the BATCH core keeps its CLI value 3. --prefix_reuse=0 pinned the measurement protocol and is dropped. BATCH runs exactly 8 sessions, so on serve this set applies only where the final --session_slots is exactly 8 (given by the user, for example --context_tokens=65536 --session_slots=8, as measured); elsewhere the row runs its concurrency=balanced row, thin steps for as many sessions as fit, and startup prints how many fit (9 Nov). | TP4_EXL3 | 1 | default |
| concurrency | balanced | (empty) | - | TP4_EXL3 | default | default |
| focus | balanced | (empty) | - | TP4_EXL3 | default | default |
| focus | ttft | (empty) | - | TP4_EXL3 | default | default |
| focus | decode | `--tp_draft_layers=1`; needs concurrency=single/multiple/balanced | Layer drafting: S06/d71cd05 true THIN at 65536, only layers 0->1: code +12.0 %, prose +2.9 %, structured +1.6 % C1 decode, Fraser median +5.85 ms, so decode only; S10 thin_full_layers +12.15 % code. It needs the BF16 wire and draft shards of the single or multiple set. | TP4_EXL3 | default | default |
| focus | prefill | (empty) | - | TP4_EXL3 | default | default |


## Retained route proposals

These records are held metadata, never executable resolver settings. Tuple order is
category / exactness / concurrency / focus; listed values are proposals requiring the stated gates.

### pp5_head_lease

Family: original. Proposed: `code,prose,structured / balanced,fast / single / decode`.

Branch control: `original_head_offload=1`. Old fallback: `original_head_offload=0; native PRO head`.

Native PP5, original 8/7/7/7/11 layer map, native PRO norm/sampler/draft; text; measured short-context DSpark recipe.

HOLD: Retained without a new native run: prior C1 benefit is no better than banked, with larger C4/C8 losses. No active mapping.

### pp5_eight_layers

Family: original. Proposed: `code,prose,structured / fast / multiple / decode`.

Branch control: `layout=8/8/8/8/8`. Old fallback: `layout=8/7/7/7/11`.

Native PP5 PRO layers change; C1 regressions exclude single; measured additional certain-token error excludes balanced/exact.

HOLD: Parked after paired C1/C2 regressions and changed outputs. Earlier one-pass C4/C8 gains do not justify the fast-only precision tradeoff; no selectable tuple.

### pp5_head_mask

Family: original. Proposed: `code,prose,structured / balanced,fast / single,multiple / decode`.

Branch control: `original_head_offload=2 original_head_stream_mask=1`. Old fallback: `original_head_stream_mask=0`.

Requires unintegrated banked head; stored-row masks; synthetic/component timing and native short smoke do not establish model-vector or native speed qualification.

HOLD: Forced actual THIN2/THIN8 full vectors, NLL/PPL/top agreement, canonical state and outputs; paired native C1 versus C2-C8 including real overlap.

### pp5_queue_ahead

Family: original. Proposed: `code,prose,structured / balanced,fast / single,multiple / decode`.

Branch control: `pipeline_queue_ahead=1`. Old fallback: `pipeline_queue_ahead=0`.

Native PP5 coordinator; bounded numerical/output evidence; measured speed is flat/noise.

HOLD: Convincing paired gain in a specialized tuple, objective cross-effects and relevant speculative/concurrent state gates.

### pp5_prefill_cadence

Family: original. Proposed: `code,prose,structured / balanced,fast / multiple / ttft,prefill`.

Branch control: `prefill_decode_every>0 with prefill_order=0`. Old fallback: `prefill_decode_every=0`.

Round-robin prompt chunks, independent decoder owners; source/host fixture only.

HOLD: Native overlapping prefills and decoder fairness; full-vector/state/output gates; paired TTFT/decode/prefill and C2-C8 performance.

### tp_round_once

Family: exl3. Proposed: `code,prose,structured / balanced,fast / single,multiple / decode`.

Branch control: `tp_decode_round_once=1`. Old fallback: `tp_decode_round_once=0`.

BF16 partials summed in rank-order FP32 then one rounding; different canonical values; measured serving regressions.

HOLD: No selectable tuple: full-vocabulary quality and a useful specialized speed tradeoff remain unqualified.

### tp_prompt_chunks

Family: exl3. Proposed: `code,prose,structured / balanced,fast / multiple / decode,ttft`.

Branch control: `prefill_prompt_chunks=1 prefill_first=0 prefill_order=0`. Old fallback: `prefill_prompt_chunks=0`.

Explicit raw launch guard; scoped mixed-output evidence only; cold BATCH composition and solo golden are distinct.

HOLD: Independent full-logit/NLL/PPL/state qualification, current-source paired objective effects; no strict-exact claim for EXL3.

### prefix_fork

Family: exl3. Proposed: `code,prose,structured / balanced,fast / multiple / ttft`.

Branch control: `session_fork=1 prefill_prefix_stable=1`. Old fallback: `session_fork=0; cold prefill`.

Same authenticated owner, generation/lease checks and native-qualified stable prefix form; no cross-user reuse.

HOLD: Current-source complete recipe and mixed traffic qualification; no arbitrary extension, legacy-form or original exact admission.


## Route parameters

Each route is described by three separate parameters, derived from its settings: the GPU split
(`--placement`: TP4_PRO is tensor parallel across the four CMP 170HX with the RTX PRO 6000 as a pipeline
stage, PP5 the five GPUs in pipeline, TP4_EXL3 tensor parallel across the four CMP 170HX), the pipeline
depth (`--pipeline_steps`: the steps in flight) and the serving slots (`--session_slots`). A generate value
that differs from the serve value follows in parentheses.

| Route | GPU split (`--placement`) | Pipeline depth (`--pipeline_steps`) | Serving slots (`--session_slots`) |
| --- | --- | --- | --- |
| `batch_fixed` | TP4_EXL3 | 1 | 8 |
| `batch_adaptive` | TP4_EXL3 | 1 | 8 |
| `batch_session` | TP4_EXL3 | 1 | 8 |
| `thin_fixed` | TP4_EXL3 | 2 | 1 |
| `batch_compromise` | TP4_EXL3 | 1 | 8 |
| `original_tp_serial2` | TP4_PRO | 2 | 1 |
| `original_pp_depth8` | PP5 | 8 (generate 2) | 8 |
| `original_pp_depth5` | PP5 | 5 (generate 2) | 8 |
| `thin_full_layers` | TP4_EXL3 | 2 | 1 |
| `original_reference_serial_m1` | PP5 | 1 | 1 |
| `original_pp_banked_single` | PP5 | 5 (generate 1) | 8 |
| `original_pp_session6` | PP5 | 5 (generate 1) | 8 |
| `original_pp_ttft10` | PP5 | 5 (generate 1) | 8 |
| `original_pp_structured0` | PP5 | 5 (generate 1) | 8 |
| `engram_disk_held` | default | default | default |
| `exl3_exact_shaping` | TP4_EXL3 | default | 1 |

## Knob values

### batch_fixed

Full-depth code-first BATCH with the merged pool/deferred-slot core3; cold identity and current-source serving qualification are mandatory.

GPU split: TP4_EXL3. Pipeline depth: 1. Serving slots: 8.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_learned_acceptance=1
--draft_base_us=13000
--draft_tail_us=0
--draft_row_us=1200
--tp_split_least_rows=512
--card_memory_kept_free_mib=0
--batch_rows=8,8,12,12,16,24,32,48
--batch_verifiers=1,2,2,4,4,4,8,8
--batch_wide_head=1
--batch_draft_rows=1
--draft_policy=0
--batch_core_once=3
--draft_session_acceptance=0
--batch_plan_ties=0
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
```

Serve additions:

```text
--session_slots=8
--prefill_first=1
--prefix_reuse=0
--prefill_decode_rows=1024
--prefill_decode_rounds=1
--prefill_decode_waiters=0
--prefill_order=2
```

Generate base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_learned_acceptance=1
--draft_base_us=13000
--draft_tail_us=0
--draft_row_us=1200
--tp_split_least_rows=512
--card_memory_kept_free_mib=0
--batch_rows=8
--batch_verifiers=1
--batch_wide_head=1
--batch_draft_rows=1
--draft_policy=0
--batch_core_once=3
--draft_session_acceptance=0
--batch_plan_ties=0
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
```

Measured frozen S10/e4c188e (base5029986), ordinary binary8d9b8b05, context65536/eight slots/four CMPs180W. C1 per stream; C2/C4/C8 aggregate. code C1/C2/C4/C8 265.854/296.834/377.388/469.537 tok/s. prose C1/C2/C4/C8 163.711/217.742/291.146/382.997 tok/s. structured C1/C2/C4/C8 339.241/452.462/613.633/787.722 tok/s. Fraser median/worst ms: C1 1579.20/1585.00; C4 4140.55/6846.50; C8 7763.70/14601.50; ten/three/three batches. Warm-short/all-short max SSE content gaps 0.480538/0.488661s; sampled native decode/prefill progress in both warm repetitions. These are not native token clocks or an isolated GPU pause. Minimum CMP free at readiness 425MiB. Full selected8/256 references and141 MiaAI request/text/native-count/finish signatures pass;35 M identities pass. Observed post-window device/cap restoration and full artifact/source closure pass. Own one-CPU low-priority build10:07:10-10:20:44 overlaps MiaAI and most M; Fraser begins after it. Sequential observations; no isolated causal or global-optimum claim. Cold prefill, full mixed traffic and1M work were not requested. No bounded-balanced admission or current-main default comparison follows from this window. Current S10 full-vocabulary/reference distance and PPL are unmeasured; the S06 cached-eight diagnostic remains separately sourced. Current HC/forms, no added lossy arithmetic; not official-exact. Source: evidence/gxf_s10_fixed_multiple_strict_s10_v1.json. Current raw tcb78/39980 control (not a selected axis): code C1/C2/C4/C8 239.200/284.234/388.513/438.798tok/s; prose C1/C2/C4/C8 169.926/258.172/373.774/446.555tok/s; structured C1/C2/C4/C8 340.663/453.337/613.833/791.158tok/s; Fraser C1/C4/C8 median/worst1578.10/1583.40,4110.05/6752.30,7620.10/14295.10ms; warm-short SSE max0.469012s. Code dominates the old adaptive/session control in all four cells; prose/structured stay within90% of observed best. This is not an independently proven zero-regression comparison or all-balanced admission. N build36 overlaps part of this control Fraser; no isolated causal inference. Source:evidence/gxf_tcb78_control_matrix_strict_xf18_v1.json. Fresh raw-main mixed comparison xf19: tcb78/39980, explicit65K; scenario order C1/C2burst/C2stagger/C4burst/C4stagger/C8burst/C8stagger. BATCH8 whole-window rates 128.661/145.732/133.334/161.279/159.154/181.848/179.383; trueTHIN1 122.756/122.618/119.106/122.719/122.579/122.622/122.647tok/s. These14.3k/0/690 controls use BATCHcore3 or THINwide1024/pipeline1/layers1, not the S10 axis settings. 112cross-family request/text/reason/native-count/finish signatures and96non-solo comparisons per arm pass; each has selected8/256+35M closure. M warm-short SSE gaps BATCH/THIN 0.473759/0.020419s; native overlap both warm True/False. Minimum CMP free 425/3203MiB. SSE gaps are not native token clocks; one-slot serialization is not pause-free overlap. BATCH90%mixed-floor screen=True; same-recipe SSE repeat delta +4.747ms with different preceding traffic. No latency gate waived or tuple admitted. S10freshmixed17 was not run; other concurrencies/full1M remain unmeasured. Source:evidence/xf19_raw_mixed_comparison_v1.json (both independently pinned strict summaries).

### batch_adaptive

BATCH with the measured row/verifier menu and adaptive draft prices.

GPU split: TP4_EXL3. Pipeline depth: 1. Serving slots: 8.

Numerical class: `same_bits`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_learned_acceptance=1
--draft_base_us=13000
--draft_tail_us=0
--draft_row_us=1200
--tp_split_least_rows=512
--card_memory_kept_free_mib=0
--batch_rows=8,8,12,12,16,24,32,48
--batch_verifiers=1,2,2,4,4,4,8,8
--batch_wide_head=1
--batch_draft_rows=1
--batch_step_us=19500,21500,28600,30000,32000,39000,46000,58000
--draft_policy=2
--batch_core_once=2
--draft_session_acceptance=0
--batch_plan_ties=0
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
```

Serve additions:

```text
--session_slots=8
--prefill_first=1
--prefix_reuse=0
--prefill_decode_rows=1024
--prefill_decode_rounds=1
--prefill_decode_waiters=0
--prefill_order=2
```

Generate base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_learned_acceptance=1
--draft_base_us=13000
--draft_tail_us=0
--draft_row_us=1200
--tp_split_least_rows=512
--card_memory_kept_free_mib=0
--batch_rows=8
--batch_verifiers=1
--batch_wide_head=1
--batch_draft_rows=1
--batch_step_us=19500
--draft_policy=2
--batch_core_once=2
--draft_session_acceptance=0
--batch_plan_ties=0
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
```

Measured frozen S10/e4c188e (base5029986), ordinary binary8d9b8b05, context65536/eight slots/four CMPs180W. C1 per stream; C2/C4/C8 aggregate. code C1/C2/C4/C8 224.170/278.327/359.324/432.990 tok/s. prose C1/C2/C4/C8 167.648/274.233/361.298/450.231 tok/s. structured C1/C2/C4/C8 338.899/451.601/611.101/787.726 tok/s. Fraser median/worst ms: C1 1580.00/1586.60; C4 4105.00/6731.10; C8 7622.45/14307.60; ten/three/three batches. Warm-short/all-short max SSE content gaps 0.473398/0.483963s; sampled native decode/prefill progress in both warm repetitions. These are not native token clocks or an isolated GPU pause. Minimum CMP free at readiness 423MiB. Full selected8/256 references and141 MiaAI request/text/native-count/finish signatures pass;35 M identities pass. Observed post-window device/cap restoration and full artifact/source closure pass. Own one-CPU startup checks10:28:35-10:28:45 overlap MiaAI; M/Fraser follow later. Sequential observations; no isolated causal or global-optimum claim. Cold prefill, full mixed traffic and1M work were not requested. No bounded-balanced admission or current-main default comparison follows from this window. Current S10 full-vocabulary/reference distance and PPL are unmeasured; the S06 cached-eight diagnostic remains separately sourced. Current HC/forms, no added lossy arithmetic; not official-exact. Source: evidence/gxf_s10_adaptive_multiple_strict_s10_v1.json. Current raw tcb78/39980 control (not a selected axis): code C1/C2/C4/C8 239.200/284.234/388.513/438.798tok/s; prose C1/C2/C4/C8 169.926/258.172/373.774/446.555tok/s; structured C1/C2/C4/C8 340.663/453.337/613.833/791.158tok/s; Fraser C1/C4/C8 median/worst1578.10/1583.40,4110.05/6752.30,7620.10/14295.10ms; warm-short SSE max0.469012s. Code dominates the old adaptive/session control in all four cells; prose/structured stay within90% of observed best. This is not an independently proven zero-regression comparison or all-balanced admission. N build36 overlaps part of this control Fraser; no isolated causal inference. Source:evidence/gxf_tcb78_control_matrix_strict_xf18_v1.json. Fresh raw-main mixed comparison xf19: tcb78/39980, explicit65K; scenario order C1/C2burst/C2stagger/C4burst/C4stagger/C8burst/C8stagger. BATCH8 whole-window rates 128.661/145.732/133.334/161.279/159.154/181.848/179.383; trueTHIN1 122.756/122.618/119.106/122.719/122.579/122.622/122.647tok/s. These14.3k/0/690 controls use BATCHcore3 or THINwide1024/pipeline1/layers1, not the S10 axis settings. 112cross-family request/text/reason/native-count/finish signatures and96non-solo comparisons per arm pass; each has selected8/256+35M closure. M warm-short SSE gaps BATCH/THIN 0.473759/0.020419s; native overlap both warm True/False. Minimum CMP free 425/3203MiB. SSE gaps are not native token clocks; one-slot serialization is not pause-free overlap. BATCH90%mixed-floor screen=True; same-recipe SSE repeat delta +4.747ms with different preceding traffic. No latency gate waived or tuple admitted. S10freshmixed17 was not run; other concurrencies/full1M remain unmeasured. Source:evidence/xf19_raw_mixed_comparison_v1.json (both independently pinned strict summaries).

### batch_session

BATCH with session acceptance mode 5 and tied plan cuts.

GPU split: TP4_EXL3. Pipeline depth: 1. Serving slots: 8.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_learned_acceptance=1
--draft_base_us=13000
--draft_tail_us=0
--draft_row_us=1200
--tp_split_least_rows=512
--card_memory_kept_free_mib=0
--batch_rows=8,8,12,12,16,24,32,48
--batch_verifiers=1,2,2,4,4,4,8,8
--batch_wide_head=1
--batch_draft_rows=1
--batch_step_us=19500,21500,28600,30000,32000,39000,46000,58000
--draft_policy=2
--draft_session_acceptance=5
--batch_plan_ties=1
--batch_core_once=2
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
```

Serve additions:

```text
--session_slots=8
--prefill_first=1
--prefix_reuse=0
--prefill_decode_rows=1024
--prefill_decode_rounds=1
--prefill_decode_waiters=0
--prefill_order=2
```

Generate base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_learned_acceptance=1
--draft_base_us=13000
--draft_tail_us=0
--draft_row_us=1200
--tp_split_least_rows=512
--card_memory_kept_free_mib=0
--batch_rows=8
--batch_verifiers=1
--batch_wide_head=1
--batch_draft_rows=1
--batch_step_us=19500
--draft_policy=2
--draft_session_acceptance=5
--batch_plan_ties=1
--batch_core_once=2
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
```

Measured frozen S10/e4c188e (base5029986), ordinary binary8d9b8b05, context65536/eight slots/four CMPs180W. C1 per stream; C2/C4/C8 aggregate. code C1/C2/C4/C8 227.868/283.425/369.675/433.150 tok/s. prose C1/C2/C4/C8 168.169/273.004/377.822/489.290 tok/s. structured C1/C2/C4/C8 339.781/451.463/612.148/787.766 tok/s. Fraser median/worst ms: C1 1578.35/1584.90; C4 4116.50/6760.10; C8 7628.85/14294.90; ten/three/three batches. Warm-short/all-short max SSE content gaps 0.473932/0.483005s; sampled native decode/prefill progress in both warm repetitions. These are not native token clocks or an isolated GPU pause. Minimum CMP free at readiness 423MiB. Full selected8/256 references and141 MiaAI request/text/native-count/finish signatures pass;35 M identities pass. Observed post-window device/cap restoration and full artifact/source closure pass. No own C build or host-test overlap. Sequential observations; no isolated causal or global-optimum claim. Cold prefill, full mixed traffic and1M work were not requested. No bounded-balanced admission or current-main default comparison follows from this window. Current S10 full-vocabulary/reference distance and PPL are unmeasured; the S06 cached-eight diagnostic remains separately sourced. Current HC/forms, no added lossy arithmetic; not official-exact. Source: evidence/gxf_s10_session_multiple_strict_s10_v1.json. Current raw tcb78/39980 control (not a selected axis): code C1/C2/C4/C8 239.200/284.234/388.513/438.798tok/s; prose C1/C2/C4/C8 169.926/258.172/373.774/446.555tok/s; structured C1/C2/C4/C8 340.663/453.337/613.833/791.158tok/s; Fraser C1/C4/C8 median/worst1578.10/1583.40,4110.05/6752.30,7620.10/14295.10ms; warm-short SSE max0.469012s. Code dominates the old adaptive/session control in all four cells; prose/structured stay within90% of observed best. This is not an independently proven zero-regression comparison or all-balanced admission. N build36 overlaps part of this control Fraser; no isolated causal inference. Source:evidence/gxf_tcb78_control_matrix_strict_xf18_v1.json. Fresh raw-main mixed comparison xf19: tcb78/39980, explicit65K; scenario order C1/C2burst/C2stagger/C4burst/C4stagger/C8burst/C8stagger. BATCH8 whole-window rates 128.661/145.732/133.334/161.279/159.154/181.848/179.383; trueTHIN1 122.756/122.618/119.106/122.719/122.579/122.622/122.647tok/s. These14.3k/0/690 controls use BATCHcore3 or THINwide1024/pipeline1/layers1, not the S10 axis settings. 112cross-family request/text/reason/native-count/finish signatures and96non-solo comparisons per arm pass; each has selected8/256+35M closure. M warm-short SSE gaps BATCH/THIN 0.473759/0.020419s; native overlap both warm True/False. Minimum CMP free 425/3203MiB. SSE gaps are not native token clocks; one-slot serialization is not pause-free overlap. BATCH90%mixed-floor screen=True; same-recipe SSE repeat delta +4.747ms with different preceding traffic. No latency gate waived or tuple admitted. S10freshmixed17 was not run; other concurrencies/full1M remain unmeasured. Source:evidence/xf19_raw_mixed_comparison_v1.json (both independently pinned strict summaries).

### thin_fixed

THIN with fixed full draft depth five, one serving slot and 2048-row prefill. Canonical sharded THIN head and draft-head strategy.

GPU split: TP4_EXL3. Pipeline depth: 2. Serving slots: 1.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=2048
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=2
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--card_memory_kept_free_mib=0
--draft_policy=0
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=0
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
--batch_core_once=2
```

Serve additions:

```text
--session_slots=1
--prefix_reuse=0
```

Generate base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=2048
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=2
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--card_memory_kept_free_mib=0
--draft_policy=0
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=0
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
--batch_core_once=2
```

Measured frozen S10/e4c188e, ordinary binary8d9b8b05, context65536/one true-THIN slot, wide2048/pipeline2/tp_draft_layers0, four CMPs180W. C1 code 230.347tok/s. C1 prose 158.591tok/s. C1 structured 332.793tok/s. Fraser10cold median/worst 1381.35/1390.70ms; cold prefill9 median 3904tok/s, allcached0; minimum CMP free 3181MiB. Full selected8/256 references,15 MiaAI signatures,source/artifact closure and observed cap restoration pass. No own C build/test overlap. C2-C8, mixed, pause and full1M work unmeasured by this window. Separate S10 one1M-slot allocation/short window: C1 code 229.339, prose 157.465, structured 327.399tok/s; 15MiB minimum free, both2048/1024 mc_check2 pairs pass. Allocation/short work only, no full1M/pi proof; context1048576 is a raw override, never inserted by this axis. Source:evidence/gxf_s10_single_1m_strict_s10_v1.json. Current HC/forms, no added lossy arithmetic; not official-exact. Current full-vocabulary reference distance, corpusPPL and all-width bit equality unmeasured. Historical S06 eight-position numerics remain in the source registry. Source:evidence/gxf_s10_single_layers0_strict_xf18_v1.json. Fresh raw-main mixed comparison xf19: tcb78/39980, explicit65K; scenario order C1/C2burst/C2stagger/C4burst/C4stagger/C8burst/C8stagger. BATCH8 whole-window rates 128.661/145.732/133.334/161.279/159.154/181.848/179.383; trueTHIN1 122.756/122.618/119.106/122.719/122.579/122.622/122.647tok/s. These14.3k/0/690 controls use BATCHcore3 or THINwide1024/pipeline1/layers1, not the S10 axis settings. 112cross-family request/text/reason/native-count/finish signatures and96non-solo comparisons per arm pass; each has selected8/256+35M closure. M warm-short SSE gaps BATCH/THIN 0.473759/0.020419s; native overlap both warm True/False. Minimum CMP free 425/3203MiB. SSE gaps are not native token clocks; one-slot serialization is not pause-free overlap. BATCH90%mixed-floor screen=True; same-recipe SSE repeat delta +4.747ms with different preceding traffic. No latency gate waived or tuple admitted. S10freshmixed17 was not run; other concurrencies/full1M remain unmeasured. Source:evidence/xf19_raw_mixed_comparison_v1.json (both independently pinned strict summaries).

### batch_compromise

Qualification candidate: BATCH with 2048-row cold prefill and the current 1024-row active-prefill cap.

GPU split: TP4_EXL3. Pipeline depth: 1. Serving slots: 8.

Numerical class: `same_bits`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=2048
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_learned_acceptance=1
--draft_base_us=13000
--draft_tail_us=0
--draft_row_us=1200
--tp_split_least_rows=512
--card_memory_kept_free_mib=0
--batch_rows=8,8,12,12,16,24,32,48
--batch_verifiers=1,2,2,4,4,4,8,8
--batch_wide_head=1
--batch_draft_rows=1
--batch_step_us=19500,21500,28600,30000,32000,39000,46000,58000
--draft_policy=2
--batch_core_once=2
--draft_session_acceptance=0
--batch_plan_ties=0
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
```

Serve additions:

```text
--session_slots=8
--prefill_first=1
--prefix_reuse=0
--prefill_decode_rows=1024
--prefill_decode_rounds=1
--prefill_decode_waiters=0
--prefill_order=2
```

Generate base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=2048
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_learned_acceptance=1
--draft_base_us=13000
--draft_tail_us=0
--draft_row_us=1200
--tp_split_least_rows=512
--card_memory_kept_free_mib=0
--batch_rows=8
--batch_verifiers=1
--batch_wide_head=1
--batch_draft_rows=1
--batch_step_us=19500
--draft_policy=2
--batch_core_once=2
--draft_session_acceptance=0
--batch_plan_ties=0
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
```

Measured S05 gxf_s05_mix2048: both full8/256 generate files equal the frozen BATCH family controls (30K24b0b22314d2, Fraserf49e8badff47). Eight-slot/65536-token serve fails the model memory planner before readiness; no serving rates, TTFT, pauses or memory-ready values are measured. The source candidate is held. This is a resource refusal, not a numerical mismatch. Source: gnmac_port/evidence/gxf_s05_mix2048.

### original_tp_serial2

Original-weight native PRO tail with the existing serial four-CMP prefix, two pipeline steps and one serving slot.

GPU split: TP4_PRO. Pipeline depth: 2. Serving slots: 1.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=tp4_pro
--numerics=exact
--cmp_order=reference
--linear_profile=0
--linear_order=1
--pro_emulation=0
--card_order=0,1,3,4,2
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_policy=0
--card_memory_kept_free_mib=0
--pipeline_steps=2
--official_hc_rows=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
```

Serve additions:

```text
--session_slots=1
--prefix_reuse=0
--prefill_order=0
--original_batch=0
```

Generate base:

```text
--placement=tp4_pro
--numerics=exact
--cmp_order=reference
--linear_profile=0
--linear_order=1
--pro_emulation=0
--card_order=0,1,3,4,2
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_policy=0
--card_memory_kept_free_mib=0
--pipeline_steps=2
--official_hc_rows=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
```

Measured frozen S10/e4c188e (base5029986), ordinary binary 8d9b8b05. Original pack, native PRO and four CMPs at 180 W, context 65536, 1 slot(s), prefix reuse off. C1 rates are per stream; C2/C4/C8 rates are aggregates. code C1/C2/C4/C8: 207.123/unmeasured/unmeasured/unmeasured tok/s. MiaAI median TTFT: 100.60/unmeasured/unmeasured/unmeasured ms (small decode prompts, not Fraser). prose C1/C2/C4/C8: 150.795/unmeasured/unmeasured/unmeasured tok/s. MiaAI median TTFT: 78.95/unmeasured/unmeasured/unmeasured ms (small decode prompts, not Fraser). structured C1/C2/C4/C8: 288.582/unmeasured/unmeasured/unmeasured tok/s. MiaAI median TTFT: 73.30/unmeasured/unmeasured/unmeasured ms (small decode prompts, not Fraser). Full selected 8/256 records and 15 MiaAI request/text/native-count/finish signatures pass. Minimum CMP free at readiness: 5503 MiB. Startup expansion, artifact/source pins, original-pack stat and device/cap restoration pass. Five C1 batches and three batches at each requested C2/C4/C8 cell. No own C-build/host-test overlap. Sequential observations, no causal or global-optimum claim. Cold prefill, Fraser, mixed traffic, long-prefill pauses, 1M fit and full-context work were not requested. Current HC/current forms; no added lossy arithmetic, not official-exact. Source: evidence/gxf_s10_original_single_strict_s10_v1.json. Historical original numerical diagnostic, O xo03/xo16: eight aligned cached-one teacher inputs, 1,034,240 F32 logits. PP5 and TP4_PRO captures are equal; against R original parent PRO2 with unchanged sparse core CMP0, mean/max KL 0.0407172/0.144973 nats, raw RMS/max 0.568391/4.276049, teacher top1 8/8. Source75 baseline replay preserves those captures. Not untouched all-PRO, corpus PPL, generated agreement or current-S10 full-vocabulary distance. See evidence/xo16_serial_reference_distance_v1.json and RND_O/S 5Nov09:26. Same-binary C1 route comparison also qualifies balanced-category single/decode: TP4_PRO at pipeline depth 2 with 1 serving slot (session0) beats banked PP5/session6 in all three genres with identical full outputs. TP versus banked structured284.74/250.51, code203.40/193.00, prose148.02/131.28tok/s. Banked has lower TTFT, so this decision is decode-only. Session6 on that TP4_PRO route is flat within repeat spread and is not selected.

### original_pp_depth8

Original-weight PP5 placement at measured depth8 for all supported multiple-user decode categories; eight serving slots.

GPU split: PP5. Pipeline depth: 8 (generate 2). Serving slots: 8.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_profile=0
--linear_order=1
--pro_emulation=0
--card_order=0,1,3,4,2
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_policy=0
--card_memory_kept_free_mib=0
--pipeline_steps=8
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--official_hc_rows=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
```

Serve additions:

```text
--session_slots=8
--prefix_reuse=0
--prefill_order=0
--original_batch=0
```

Generate base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_profile=0
--linear_order=1
--pro_emulation=0
--card_order=0,1,3,4,2
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_policy=0
--card_memory_kept_free_mib=0
--pipeline_steps=2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--official_hc_rows=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
```

Measured frozen S10/e4c188e (base5029986), ordinary binary 8d9b8b05. Original pack, native PRO and four CMPs at 180 W, context 65536, 8 slot(s), prefix reuse off. C1 rates are per stream; C2/C4/C8 rates are aggregates. code C1/C2/C4/C8: 171.738/250.459/347.808/406.960 tok/s. MiaAI median TTFT: 52.09/80.32/141.81/270.50 ms (small decode prompts, not Fraser). prose C1/C2/C4/C8: 125.831/245.219/404.221/408.524 tok/s. MiaAI median TTFT: 43.18/72.21/122.19/223.41 ms (small decode prompts, not Fraser). structured C1/C2/C4/C8: 248.989/472.013/763.691/783.115 tok/s. MiaAI median TTFT: 41.95/64.10/109.81/190.95 ms (small decode prompts, not Fraser). Full selected 8/256 records and 141 MiaAI request/text/native-count/finish signatures pass. Non-speculative 8/256 controls also pass. Minimum CMP free at readiness: 3253 MiB. Startup expansion, artifact/source pins, original-pack stat and device/cap restoration pass. Five C1 batches and three batches at each requested C2/C4/C8 cell. No own C-build/host-test overlap. Sequential observations, no causal or global-optimum claim. Cold prefill, Fraser, mixed traffic, long-prefill pauses, 1M fit and full-context work were not requested. Current HC/current forms; no added lossy arithmetic, not official-exact. Source: evidence/gxf_s10_original_multiple_strict_s10_v1.json. Historical original numerical diagnostic, O xo03/xo16: eight aligned cached-one teacher inputs, 1,034,240 F32 logits. PP5 and TP4_PRO captures are equal; against R original parent PRO2 with unchanged sparse core CMP0, mean/max KL 0.0407172/0.144973 nats, raw RMS/max 0.568391/4.276049, teacher top1 8/8. Source75 baseline replay preserves those captures. Not untouched all-PRO, corpus PPL, generated agreement or current-S10 full-vocabulary distance. See evidence/xo16_serial_reference_distance_v1.json and RND_O/S 5Nov09:26.

### original_pp_depth5

Original-weight PP5 placement, five pipeline steps and eight serving slots for multiple-user prose/structured decode.

GPU split: PP5. Pipeline depth: 5 (generate 2). Serving slots: 8.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_profile=0
--linear_order=1
--pro_emulation=0
--card_order=0,1,3,4,2
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_policy=0
--card_memory_kept_free_mib=0
--pipeline_steps=5
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--official_hc_rows=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
```

Serve additions:

```text
--session_slots=8
--prefix_reuse=0
--prefill_order=0
--original_batch=0
```

Generate base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_profile=0
--linear_order=1
--pro_emulation=0
--card_order=0,1,3,4,2
--rows=8
--wide_rows=1024
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--draft_policy=0
--card_memory_kept_free_mib=0
--pipeline_steps=2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--official_hc_rows=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
```

Measured gxf_s07_original_other_multiple on S07/source7a066f2, original packed weights, native PRO and four CMPs at180W; prefix_reuse0, context65536, 8 slots. C1 is per stream; C2/C4/C8 are aggregate. code C1/C2/C4/C8 171.69/225.38/348.57/388.56tok/s. prose C1/C2/C4/C8 125.79/244.86/403.40/406.76tok/s. structured C1/C2/C4/C8 249.23/471.77/762.82/780.20tok/s. Full selected8/256 control records agree; MiaAI request/text/native-count/finish 141/141 agree. Minimum CMP free memory at readiness 3202MiB. Fraser TTFT, cold prefill, pauses, mixed traffic, 1M work and the balanced envelope were not measured in this window. This is current arithmetic, not official-exact. First window's initial30s overlapped S's CPU build, which finished05:21:59; later windows did not overlap that build. Source: evidence/s07_original_axes_summary_v1.json and evidence/gxf_s07_original_other_multiple.

### thin_full_layers

Single-user code decode: full fixed draft depth five with layer drafting enabled.

GPU split: TP4_EXL3. Pipeline depth: 2. Serving slots: 1.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=2048
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=2
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--card_memory_kept_free_mib=0
--draft_policy=0
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
--batch_core_once=2
```

Serve additions:

```text
--session_slots=1
--prefix_reuse=0
```

Generate base:

```text
--placement=tp4_exl3
--numerics=relaxed
--cmp_order=reference
--tp_wire_bf16=1
--rows=8
--wide_rows=2048
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--pipeline_steps=2
--step_graphs=1
--speculative=2
--draft_block=5
--draft_positions=5
--exl3_tc_moe=1
--tp_head_mma=1
--tp_trunk_mma=1
--tp_ordered_wide=1
--card_memory_kept_free_mib=0
--draft_policy=0
--tp_head_shards=1
--tp_head_stream=1
--tp_draft_shards=1
--tp_draft_layers=1
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--exl3_tc_product_blocks=0
--tp_decode_round_once=0
--official_hc_rows=0
--batch_core_once=2
```

Measured frozen S10/e4c188e, ordinary binary8d9b8b05, context65536/one true-THIN slot, wide2048/pipeline2/tp_draft_layers1, four CMPs180W. C1 code 258.344tok/s. C1 prose 163.744tok/s. C1 structured 338.442tok/s. Fraser10cold median/worst 1384.35/1393.90ms; cold prefill9 median 3901tok/s, allcached0; minimum CMP free 823MiB. Full selected8/256 references,15 MiaAI signatures,source/artifact closure and observed cap restoration pass. No own C build/test overlap. C2-C8, mixed, pause and full1M work unmeasured by this window. Compared with matched layer0: code +12.15%; prose/structured also increase; Fraser median/worst +3.00/+3.20ms. FAIL strict no-worse TTFT for a balanced-focus choice; selected only for explicit code/single/decode. No1M capacity inference: this arm has2358MiB less free memory at65K. Current HC/forms, no added lossy arithmetic; not official-exact. Current full-vocabulary reference distance, corpusPPL and all-width bit equality unmeasured. Historical S06 eight-position numerics remain in the source registry. Source:evidence/gxf_s10_single_layers1_strict_xf18_v1.json. Fresh raw-main mixed comparison xf19: tcb78/39980, explicit65K; scenario order C1/C2burst/C2stagger/C4burst/C4stagger/C8burst/C8stagger. BATCH8 whole-window rates 128.661/145.732/133.334/161.279/159.154/181.848/179.383; trueTHIN1 122.756/122.618/119.106/122.719/122.579/122.622/122.647tok/s. These14.3k/0/690 controls use BATCHcore3 or THINwide1024/pipeline1/layers1, not the S10 axis settings. 112cross-family request/text/reason/native-count/finish signatures and96non-solo comparisons per arm pass; each has selected8/256+35M closure. M warm-short SSE gaps BATCH/THIN 0.473759/0.020419s; native overlap both warm True/False. Minimum CMP free 425/3203MiB. SSE gaps are not native token clocks; one-slot serialization is not pause-free overlap. BATCH90%mixed-floor screen=True; same-recipe SSE repeat delta +4.747ms with different preceding traffic. No latency gate waived or tuple admitted. S10freshmixed17 was not run; other concurrencies/full1M remain unmeasured. Source:evidence/xf19_raw_mixed_comparison_v1.json (both independently pinned strict summaries).

### original_reference_serial_m1

Original-weight official-reference arithmetic on the PP THIN serial M1 schedule with capacity512; default off. Standard batched prompts, physical ring wrap, rollback and TP remain open scope.

GPU split: PP5. Pipeline depth: 1. Serving slots: 1.

Numerical class: `reference_arithmetic`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_profile=0
--linear_order=1
--pro_emulation=0
--official_hc_rows=1
--official_reference=1
--card_order=0,1,3,4,2
--rows=1
--context_tokens=512
--wide_rows=0
--full_steps=0
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--speculative=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
--card_memory_kept_free_mib=0
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
```

Serve additions:

```text
--session_slots=1
--prefix_reuse=0
--prefix_reuse_exact=0
--session_fork=0
--prefill_order=0
--original_batch=0
```

Generate base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_profile=0
--linear_order=1
--pro_emulation=0
--official_hc_rows=1
--official_reference=1
--card_order=0,1,3,4,2
--rows=1
--context_tokens=512
--wide_rows=0
--full_steps=0
--chunks_in_flight=1
--pipeline_steps=1
--step_graphs=1
--speculative=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
--card_memory_kept_free_mib=0
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
```

The original-weight reference recipe uses PP THIN serial M1, one session, context capacity 512. XR57 qualifies the production prepared-program prelude over 263 positions: all 34,000,640 full-vocabulary F32 logits and the first-eight-position block/carry endpoints across all 40 layers match the pinned original serial source. Two autonomous 256-draw real CLI arms match every source token ID and selected F32 logit bit; exact and control repeats pass. The fixed horizon includes post-EOS draws with --ignore_eos=1; ordinary EOS stopping remains the default. XR59 preserves canonical mc_check2 hashes 41eeb9c16e10/f49e8badff47 at both 2048/1024 widths. XR50b and XR53 provide the stated release linear/core/router and state/site/Engram/projection component proofs, including BF16 carrier and all-visible-index controls. Earlier XR48/XR49 retain their scoped packet source/state/cache proofs; those do not become a new production general-cache claim. Final-binary CLI/server/default and quiet-cost receipts, including final C1 cost, are maintained externally in COSTS.md and REPORT_v4.md and must pass before shipping. Prefix reuse/forks, speculative decoding, standard batched-prompt prefill, longer contexts, physical ring wrap, rollback, TP and image execution are outside this initial recipe. No balanced/default arithmetic or profile promotion is selected.

### original_pp_banked_single

Retained raw Original PP5 banked head/session6 route; no default tuple selects it after the same-binary decode win of TP4_PRO at pipeline depth 2 with 1 serving slot.

GPU split: PP5. Pipeline depth: 5 (generate 1). Serving slots: 8.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_order=1
--linear_profile=0
--pro_emulation=0
--card_order=0,1,3,4,2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--wide_rows=1024
--rows=8
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--draft=dspark
--draft_block=5
--draft_positions=5
--card_memory_kept_free_mib=0
--original_head_offload=2
--pipeline_steps=5
--official_hc_rows=0
--official_reference=0
--context_tokens=65536
--draft_policy=1
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--draft_learned_acceptance=0
--draft_session_acceptance=6
--batch_plan_ties=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
```

Serve additions:

```text
--session_slots=8
--prefix_reuse=1
--prefill_order=0
--original_batch=0
```

Generate base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_order=1
--linear_profile=0
--pro_emulation=0
--card_order=0,1,3,4,2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--wide_rows=1024
--rows=8
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--draft=dspark
--draft_block=5
--draft_positions=5
--card_memory_kept_free_mib=0
--original_head_offload=2
--official_hc_rows=0
--official_reference=0
--context_tokens=65536
--pipeline_steps=1
--draft_policy=1
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--draft_learned_acceptance=0
--draft_session_acceptance=6
--batch_plan_ties=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
```

Retained raw PP5 banked head/session6. Historical 575 independent full-F32 head vectors and short logical-state scopes match native-PRO balanced, not strict original exactness. Banked+6 paired C1 versus PP5 session0 preserves all outputs and gains0.10/9.65/7.99percent structured/code/prose; five-bank captures78/42/42/42/42 reuse144 graphs, preparation0.380s versus1.554s in separate leases and factory heap5450664 versus6813888bytes by the old same-shape formula. Later same-binary head-to-head preserves all outputs but TP4_PRO at pipeline depth 2 with 1 serving slot (session0) is faster in every decode genre:284.74/203.40/148.02 versus250.51/193.00/131.28tok/s. Banked retains much lower TTFT but no axis now selects it. Raw recipe uses native PRO PP5 layout,8allocated slots/pipeline5 at65536context, DSpark5/COST1 and one active workload. Generate uses1slot/pipeline1 with no speed claim. C2 regresses; prefill, vision, full-context and strict exactness remain unqualified. Rebased integration requires its native merge gate; defaults stay0.

### original_pp_session6

Original PP5 balanced/code/prose balanced/decode plan with session acceptance6; eight allocated serving slots. Measured PP5 component candidate only; no active tuple is added and no TP4_PRO route or PP5 route at pipeline depth 8 is replaced.

GPU split: PP5. Pipeline depth: 5 (generate 1). Serving slots: 8.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_order=1
--linear_profile=0
--pro_emulation=0
--card_order=0,1,3,4,2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--wide_rows=1024
--rows=8
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--draft=dspark
--draft_block=5
--draft_positions=5
--card_memory_kept_free_mib=0
--original_head_offload=0
--pipeline_steps=5
--official_hc_rows=0
--official_reference=0
--context_tokens=65536
--draft_policy=1
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--draft_learned_acceptance=0
--draft_session_acceptance=6
--batch_plan_ties=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
```

Serve additions:

```text
--session_slots=8
--prefix_reuse=1
--prefill_order=0
--original_batch=0
```

Generate base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_order=1
--linear_profile=0
--pro_emulation=0
--card_order=0,1,3,4,2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--wide_rows=1024
--rows=8
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--draft=dspark
--draft_block=5
--draft_positions=5
--card_memory_kept_free_mib=0
--original_head_offload=0
--official_hc_rows=0
--official_reference=0
--context_tokens=65536
--pipeline_steps=1
--draft_policy=1
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--draft_learned_acceptance=0
--draft_session_acceptance=6
--batch_plan_ties=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
```

Paired Original PP5 confirmation: six arms off/6/10/10/6/off, C1/C4/C8, identical T0 outputs; seeded cold T1 off/6/6/off also identical. Acceptance6 code gains8.79/13.36/6.88%, prose6.94/7.83/7.73%; structured within spread. Acceptance10 code3.48/6.98/3.55%, prose3.73/4.40/4.93%. CodeC4 TTFTp90 off/6/10=139.8/166.8/143.4ms. Select6 for balanced/decode focus and10 for TTFT; this is a scoped tradeoff, not an absolute latency optimum. Measured at65536 context, eight allocated slots, native PRO, canonical PP5 layout, COST1/12700/2200/2700, DSpark5 and head0. Generate uses one slot/pipeline1 without a generate speed claim. T1 identity covers the observed seeds, not distributional or strict original equivalence. Vision and full-context trajectories remain held; no per-value575 recapture is claimed. Later same-binary resolver confirmation at C1/C4 preserves all outputs but excludes session6 from structured balanced/decode after a0.231 percent C1 loss beyond0.0071 percent repeat spread. Code/prose keep6 for decode gains; the balanced category retains6 by the qualified workload decision. ProseC4 TTFTp90 off/6/10=112.5/138.3/112.6ms: about26ms cost for6, so TTFT focus keeps10. Structured TTFT keeps10; its decode changes are within measured spread. These are per-category decisions, not a no-loss claim for every genre on session6. Session policies are route-local modifiers. These PP5 depth5 observations do not qualify them on TP4_PRO (pipeline depth 2, 1 serving slot) or PP5 (pipeline depth 8, 8 serving slots); the older faster measured routes and all prior holds remain. No Original TTFT tuple was selectable in the pre-modifier registry, so session10 remains candidate metadata until a scoped tuple is jointly admitted.

### original_pp_ttft10

Original PP5 TTFT resolver plan with session acceptance10; eight allocated serving slots, one or multiple active workloads. Measured PP5 component candidate only; no active tuple is added and no TP4_PRO route or PP5 route at pipeline depth 8 is replaced.

GPU split: PP5. Pipeline depth: 5 (generate 1). Serving slots: 8.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_order=1
--linear_profile=0
--pro_emulation=0
--card_order=0,1,3,4,2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--wide_rows=1024
--rows=8
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--draft=dspark
--draft_block=5
--draft_positions=5
--card_memory_kept_free_mib=0
--original_head_offload=0
--pipeline_steps=5
--official_hc_rows=0
--official_reference=0
--context_tokens=65536
--draft_policy=1
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--draft_learned_acceptance=0
--draft_session_acceptance=10
--batch_plan_ties=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
```

Serve additions:

```text
--session_slots=8
--prefix_reuse=1
--prefill_order=0
--original_batch=0
```

Generate base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_order=1
--linear_profile=0
--pro_emulation=0
--card_order=0,1,3,4,2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--wide_rows=1024
--rows=8
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--draft=dspark
--draft_block=5
--draft_positions=5
--card_memory_kept_free_mib=0
--original_head_offload=0
--official_hc_rows=0
--official_reference=0
--context_tokens=65536
--pipeline_steps=1
--draft_policy=1
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--draft_learned_acceptance=0
--draft_session_acceptance=10
--batch_plan_ties=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
```

Paired Original PP5 confirmation: six arms off/6/10/10/6/off, C1/C4/C8, identical T0 outputs; seeded cold T1 off/6/6/off also identical. Acceptance6 code gains8.79/13.36/6.88%, prose6.94/7.83/7.73%; structured within spread. Acceptance10 code3.48/6.98/3.55%, prose3.73/4.40/4.93%. CodeC4 TTFTp90 off/6/10=139.8/166.8/143.4ms. Select6 for balanced/decode focus and10 for TTFT; this is a scoped tradeoff, not an absolute latency optimum. Measured at65536 context, eight allocated slots, native PRO, canonical PP5 layout, COST1/12700/2200/2700, DSpark5 and head0. Generate uses one slot/pipeline1 without a generate speed claim. T1 identity covers the observed seeds, not distributional or strict original equivalence. Vision and full-context trajectories remain held; no per-value575 recapture is claimed. Later same-binary resolver confirmation at C1/C4 preserves all outputs but excludes session6 from structured balanced/decode after a0.231 percent C1 loss beyond0.0071 percent repeat spread. Code/prose keep6 for decode gains; the balanced category retains6 by the qualified workload decision. ProseC4 TTFTp90 off/6/10=112.5/138.3/112.6ms: about26ms cost for6, so TTFT focus keeps10. Structured TTFT keeps10; its decode changes are within measured spread. These are per-category decisions, not a no-loss claim for every genre on session6. Session policies are route-local modifiers. These PP5 depth5 observations do not qualify them on TP4_PRO (pipeline depth 2, 1 serving slot) or PP5 (pipeline depth 8, 8 serving slots); the older faster measured routes and all prior holds remain. No Original TTFT tuple was selectable in the pre-modifier registry, so session10 remains candidate metadata until a scoped tuple is jointly admitted.

### original_pp_structured0

Original PP5 structured balanced/decode plan with session acceptance0, the measured head0 control; eight allocated serving slots. Measured PP5 component candidate only; no active tuple is added and no TP4_PRO route or PP5 route at pipeline depth 8 is replaced.

GPU split: PP5. Pipeline depth: 5 (generate 1). Serving slots: 8.

Numerical class: `draft_only`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_order=1
--linear_profile=0
--pro_emulation=0
--card_order=0,1,3,4,2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--wide_rows=1024
--rows=8
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--draft=dspark
--draft_block=5
--draft_positions=5
--card_memory_kept_free_mib=0
--original_head_offload=0
--pipeline_steps=5
--official_hc_rows=0
--official_reference=0
--context_tokens=65536
--draft_policy=1
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
```

Serve additions:

```text
--session_slots=8
--prefix_reuse=1
--prefill_order=0
--original_batch=0
```

Generate base:

```text
--placement=pp5
--numerics=exact
--cmp_order=reference
--linear_order=1
--linear_profile=0
--pro_emulation=0
--card_order=0,1,3,4,2
--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4
--wide_rows=1024
--rows=8
--full_steps=1
--prefill_cone=1
--chunks_in_flight=1
--step_graphs=1
--draft=dspark
--draft_block=5
--draft_positions=5
--card_memory_kept_free_mib=0
--original_head_offload=0
--official_hc_rows=0
--official_reference=0
--context_tokens=65536
--pipeline_steps=1
--draft_policy=1
--draft_base_us=12700
--draft_tail_us=2200
--draft_row_us=2700
--draft_learned_acceptance=0
--draft_session_acceptance=0
--batch_plan_ties=0
--tp_pipeline_overlap=0
--original_chunk_core=0
--original_dense_staged=0
--original_wide_experts=0
```

Same-binary six-arm session0/6/10 confirmation at C1/C4 preserved complete outputs in all arms. Session6 structured C1 loses0.231 percent beyond0.0071 percent repeat spread; structured balanced/decode therefore selects the measured session0 control. Session10 remains the TTFT plan. All other PP5 COST1/DSpark5 settings are unchanged. Generate retains one slot/pipeline1 without a speed claim. No strict exactness, new575-vector, vision or full-context qualification. Session policies are route-local modifiers. These PP5 depth5 observations do not qualify them on TP4_PRO (pipeline depth 2, 1 serving slot) or PP5 (pipeline depth 8, 8 serving slots); the older faster measured routes and all prior holds remain. No Original TTFT tuple was selectable in the pre-modifier registry, so session10 remains candidate metadata until a scoped tuple is jointly admitted.

### engram_disk_held

Experimental immutable Engram placement with resident prefixes, bounded block cache and serial/direct or prepared io_uring reads. No axis enables it before native qualification.

GPU split: default. Pipeline depth: default. Serving slots: default.

Numerical class: `same_bits`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
(none)
```

Serve additions:

```text
(none)
```

Generate base:

```text
(none)
```

CPU byte, corruption, replica-loss and concurrent-reader tests; random I/O microbenchmark only. No native serving gain claimed.

### exl3_exact_shaping

EXL3 exact mode (owner, 6 Nov 2026: "On EXL3, exact takes the same shaping and strictness but is not official-exact"): GPU split TP4_EXL3 on the four CMPs with exact numerics, the reference CMP order and the FP32 wire (fixed rank-order sums, no BF16 ring, no relaxed order), DSpark at eight rows, pipeline depth 1 and one session. The weights are quantized, so it is labelled not official-exact.

GPU split: TP4_EXL3. Pipeline depth: default. Serving slots: 1.

Numerical class: `reference_arithmetic`. Declared lossy knobs: none.

Each line below is one argv token; embedded spaces and semicolons stay in that token.

Serve base:

```text
--placement=tp4_exl3
--numerics=exact
--tp_wire_bf16=0
--speculative=2
--rows=8
```

Serve additions:

```text
--session_slots=1
```

Generate base:

```text
--placement=tp4_exl3
--numerics=exact
--tp_wire_bf16=0
--speculative=2
--rows=8
```

Per-split request 4 q4 on arms on_a/on_d (r7f4_gq_split_q4_tcb88_v1, per_split_axes_v1/ADMITTED_4.md): `--placement=tp4_exl3 --speculative=2 --rows=8` at pipeline depth 1 and 1 serving slot, exact numerics and FP32 wire, 158.13/117.18/85.76 tok/s structured/code/prose C1, TTFT p50 26.0/35.9/35.4 ms, the pair self-consistent and outputs recorded. --numerics=exact, --tp_wire_bf16=0 and --session_slots=1 are the split's raw defaults, stated so exact never inherits relaxed arithmetic.

## Cross-axis rules

Five kinds of rule join the sets of one row; each is declared in axes.json and applied by the
generator or, for the slot count, by startup. A cap clamps a knob that one source sets above another
source's bound; a floor raises a knob below what another source's setting needs; a set may apply only
with values of another axis, and is left out elsewhere, never refused; a concurrency set may apply on
serve only at one final `--session_slots` count, and elsewhere the row runs its concurrency=balanced row;
a recipe may yield one of its knobs to the axis that owns it, so the recipe's measured value runs
wherever that axis's set of the row gives none. Startup prints each applied cap or floor after its key
on the `axes from` line. A bound reads its bounding knob after that knob's own bounds, one level (the
generator refuses a longer chain), and a knob several bounds name takes the first, in declared order,
that changes it.

Caps and floors:

- cap: `--rows` bounds `--draft_block`, `--draft_positions`: --rows caps the proposals verified per round at rows - 1 (commands.c). A draft block or depth that another source of the row sets above that cap is clamped to it: the extra block positions cost time and verify nothing (r5c on TP4_PRO: depth 7 on rows 6 with arm 1 lost 10.2 % to r5s, rows 6 with arm 1 at depth 5, 232.3 -> 208.5 tok/s code C1). Bounds act on the final values, explicit ones included, and change only inserted knobs: an explicit --rows=6 keeps the clamp, --rows=8 lifts it; an explicit bounded knob is never altered and is validated as usual. A cap never lowers a knob below the least its parser takes (--draft_block 5, --draft_positions 1), also where the final speculation is off and the clamped draft knobs are inert, noted inactive (a raw --rows=1 --speculative=0 parses).
- floor: `--tp_draft_shards`=`1` (under `--batch_draft_rows`) raises `--tp_draft_layers`: Sharded BATCH draft rows require draft layers (model_v41.c: batch_draft_rows with draft_shards needs draft_layers). Where the final --tp_draft_shards=1 and --batch_draft_rows are composed or given (TP4_EXL3 concurrency=multiple), an inserted --tp_draft_layers is raised to 1, the value every BATCH tuple ran with before the per-split tables. Bounds act on the final values: an explicit --tp_draft_shards=1 keeps the raise, --tp_draft_shards=0 or --batch_draft_rows=0 removes it; an explicit --tp_draft_layers is never altered and is validated as usual.
- cap: `--tp_head_shards` bounds `--tp_draft_shards`: The draft's shards need the target head's (commands.c: --tp_draft_shards needs --tp_head_shards=1). TP4_EXL3's base inserts --tp_head_shards=1 and --tp_draft_shards=1, and an explicit knob replaces only its own key, so before this cap an explicit --tp_head_shards=0 left the inserted draft shards at 1 and the parser refused the command: where the final --tp_head_shards is 0, an inserted --tp_draft_shards is lowered to 0, and the explicit knob wins alone. An explicit --tp_draft_shards is never altered and is validated as usual.
- cap: `--tp_draft_shards` bounds `--tp_draft_layers`: This cap mirrors the draft-shards part of the parser's layers rule (commands.c: --tp_draft_layers needs TP4_EXL3 with the BF16 wire and --tp_draft_shards=1, on BATCH also --batch_draft_rows=1 and --batch_wide_head=1): where the final --tp_draft_shards is 0, given or lowered by the head-shards cap, an inserted --tp_draft_layers (focus=decode's, or the BATCH rows' raised one) is lowered to 0. No bound follows the rule's other parts, so an explicit --tp_wire_bf16=0 or, on BATCH, --batch_draft_rows=0 under focus=decode's layers, or --batch_wide_head=0 under any BATCH row's, is still refused. A bound reads its bounding knob after that knob's own bounds, one level (the generator refuses a longer chain), and a knob several bounds name takes the first, in declared order, that changes it, so the floor above no longer raises the layers once the head-shards cap has lowered the draft shards. An explicit --tp_draft_layers is never altered and is validated as usual.

Sets that apply only with values of another axis:

- TP4_EXL3 category=prose: only with concurrency=multiple
- TP4_EXL3 category=structured: only with concurrency=multiple

Sets that apply on serve only at one serving slot count (elsewhere the row runs its
concurrency=balanced row, and startup says how many sessions fit):

- TP4_EXL3 concurrency=multiple: only at exactly `--session_slots=8`

Recipe knobs yielded to the axis that owns them:

- PP5 concurrency=multiple (`original_pp_depth8`): `--draft_block`, `--draft_positions`, `--draft_policy`, `--draft_session_acceptance`

Rows with an applied cap or floor: 36.
- PP5 code/balanced/single/balanced: --draft_block 7 -> 5 (bounded by --rows=6 from concurrency=single); --draft_positions 7 -> 5 (bounded by --rows=6 from concurrency=single)
- PP5 code/balanced/single/ttft: --draft_block 7 -> 5 (bounded by --rows=6 from concurrency=single); --draft_positions 7 -> 5 (bounded by --rows=6 from concurrency=single)
- PP5 code/balanced/single/decode: --draft_block 7 -> 5 (bounded by --rows=6 from concurrency=single); --draft_positions 7 -> 5 (bounded by --rows=6 from concurrency=single)
- PP5 code/balanced/single/prefill: --draft_block 7 -> 5 (bounded by --rows=6 from concurrency=single); --draft_positions 7 -> 5 (bounded by --rows=6 from concurrency=single)
- PP5 code/fast/single/balanced: --draft_block 7 -> 5 (bounded by --rows=6 from concurrency=single); --draft_positions 7 -> 5 (bounded by --rows=6 from concurrency=single)
- PP5 code/fast/single/ttft: --draft_block 7 -> 5 (bounded by --rows=6 from concurrency=single); --draft_positions 7 -> 5 (bounded by --rows=6 from concurrency=single)
- PP5 code/fast/single/decode: --draft_block 7 -> 5 (bounded by --rows=6 from concurrency=single); --draft_positions 7 -> 5 (bounded by --rows=6 from concurrency=single)
- PP5 code/fast/single/prefill: --draft_block 7 -> 5 (bounded by --rows=6 from concurrency=single); --draft_positions 7 -> 5 (bounded by --rows=6 from concurrency=single)
- TP4_PRO code/balanced/single/decode: --draft_block 7 -> 5 (bounded by --rows=6 from focus=decode); --draft_positions 7 -> 5 (bounded by --rows=6 from focus=decode)
- TP4_PRO code/balanced/balanced/decode: --draft_block 7 -> 5 (bounded by --rows=6 from focus=decode); --draft_positions 7 -> 5 (bounded by --rows=6 from focus=decode)
- TP4_PRO code/fast/single/decode: --draft_block 7 -> 5 (bounded by --rows=6 from focus=decode); --draft_positions 7 -> 5 (bounded by --rows=6 from focus=decode)
- TP4_PRO code/fast/balanced/decode: --draft_block 7 -> 5 (bounded by --rows=6 from focus=decode); --draft_positions 7 -> 5 (bounded by --rows=6 from focus=decode)
- TP4_EXL3 code/balanced/multiple/balanced: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 code/balanced/multiple/ttft: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 code/balanced/multiple/prefill: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 code/fast/multiple/balanced: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 code/fast/multiple/ttft: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 code/fast/multiple/prefill: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 prose/balanced/multiple/balanced: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 prose/balanced/multiple/ttft: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 prose/balanced/multiple/prefill: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 prose/fast/multiple/balanced: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 prose/fast/multiple/ttft: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 prose/fast/multiple/prefill: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 structured/balanced/multiple/balanced: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 structured/balanced/multiple/ttft: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 structured/balanced/multiple/prefill: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 structured/fast/multiple/balanced: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 structured/fast/multiple/ttft: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 structured/fast/multiple/prefill: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 balanced/balanced/multiple/balanced: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 balanced/balanced/multiple/ttft: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 balanced/balanced/multiple/prefill: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 balanced/fast/multiple/balanced: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 balanced/fast/multiple/ttft: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)
- TP4_EXL3 balanced/fast/multiple/prefill: --tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)

## Inactive draft knobs

Each axis value's set is inserted on every row of its split. A set of draft knobs acts only with speculation: its `axes from` line carries
`(inactive: no DSpark on this route)`, which startup prints only where the final speculation is off (a route without DSpark, or an explicit `--speculative=0`). Every base runs DSpark, so the rows below are those whose composition has none.

Rows with inactive draft knobs: none.

## Quality budget

Proposed conservative incremental screen, not an established quality guarantee or an owner-selected default. No lossy candidate has passed; additional lossy fast routes remain held.

Budget admitted: `false`.
Required strata: code/short, code/long, prose/short, prose/long, structured/short, structured/long.

A lossy strategy needs a paired, held-out receipt per knob and stratum. Every receipt records
source hashes, metric and agreement denominators, raw and centered errors, uncertainty and
a whole-workload rate gain. Synthetic receipts cannot enter generated runtime data.

- Paired token-weighted KL and worst-row KL against current balanced arithmetic.
- Full-logit deviation with input, call shape and rounding scope recorded.
- Held-out teacher-forced PPL/NLL fixture and its token denominator.
- T=0 selected-token and full-output agreement rate, including native counts and finish.
- Reference-tier comparison where an applicable R target exists.
- Whole-workload speed benefit under the same request/capacity/runtime conditions.
- Apply every limit separately to code/prose/structured and context strata; predeclare held-out inputs and unique-request/token denominators.
- Report raw and row-offset-removed logit errors, tail/worst strata and uncertainty; no corpus-PPL claim from an eight-token smoke probe.
- Reject a candidate exceeding any limit from every axis tuple; raw low-level use remains outside the qualified table.

Proposed limits:

```json
{
  "mean_KL_nats_max": 0.001,
  "p99_KL_nats_max": 0.01,
  "PPL_ratio_vs_balanced_max": 1.01,
  "row_offset_removed_logit_RMS_max": 0.02,
  "row_offset_removed_logit_absolute_max": 0.1,
  "teacher_forced_top1_agreement_min": 0.999,
  "complete_T0_response_agreement_min": 0.99,
  "pinned_reference_tool_schema_cases_agreement_min": 1.0
}
```

## Performance envelope

Proposed 10% performance envelope for paired qualification. The all-balanced2048 EXL3 candidate failed server memory admission and is held. A replacement must pass its own paired gates before becoming a recommendation; no CLI axis default is selected.

```json
{
  "mixed_C1_rate_vs_best_THIN_min": 0.9,
  "mixed_C2_C4_C8_rate_vs_best_BATCH_min": 0.9,
  "code_balanced_non_code_rate_vs_own_best_min": 0.9,
  "Fraser_median_TTFT_delta_vs_default_ms_max": 0,
  "Fraser_worst_TTFT_delta_vs_default_ms_max": 0,
  "long_prefill_decode_pause_delta_vs_default_s_max": 0
}
```

Apply the mixed floor separately at C1/C2/C4/C8 and the non-code floor separately to prose and structured at each code-balanced cell. Use paired default/candidate controls with the same binary, client, prompts and capacity; record cold/warm scope, input identities, repetitions and uncertainty. Fraser median/worst TTFT and the long-prefill decode pause must be no worse than the default. A failed envelope restores a nonempty candidate hold without changing the limits.

S04 gxf11 fixed-depth versus gxf12 session-adaptive measurements give prose C4 290.6 versus 377.5 tok/s (23.0% lower) and C8 381.8 versus 488.8 (21.9% lower), rejecting that fixed-depth multiple-user compromise at 10%. C1 prose 163.6 versus 167.8 is 2.5% lower and fits the proposed margin. Those C1 points are BATCH measurements, not proof against best THIN. The new current-math candidate must prove the actual mixed C1/C2-C8 envelope, memory, identity and latency objectives.

A qualification-candidate rule names every axis value explicitly and can bypass only the
concurrency-compromise hold. It cannot bypass tier, original-weight, vision or numerical
gates. A failed envelope returns the rule to a nonempty hold; the margin is not relaxed.

## Complete tuple map

The map contains 540 rows: 3 GPU splits by 180 full tuples. Each cell names the axis values whose sets it inserts (raw: none). A held row inserts nothing.

| Category | Exactness | Concurrency | Focus | PP5 | TP4_PRO | TP4_EXL3 | Seven-bucket targets |
| --- | --- | --- | --- | --- | --- | --- | --- |
| code | exact | single | balanced | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, decode_accuracy |
| code | exact | single | ttft | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, prefill_accuracy |
| code | exact | single | decode | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, decode_accuracy |
| code | exact | single | prefill | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, prefill_accuracy |
| code | exact | multiple | balanced | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| code | exact | multiple | ttft | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| code | exact | multiple | decode | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| code | exact | multiple | prefill | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| code | exact | balanced | balanced | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| code | exact | balanced | ttft | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| code | exact | balanced | decode | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| code | exact | balanced | prefill | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| code | balanced | single | balanced | category=code+concurrency=single; draft_only | category=code; draft_only | category=code; draft_only | decode_accuracy, decode_fast |
| code | balanced | single | ttft | category=code+concurrency=single; draft_only | category=code; draft_only | category=code; draft_only | prefill_accuracy, prefill_fast |
| code | balanced | single | decode | category=code+concurrency=single+focus=decode; draft_only | category=code+focus=decode; draft_only | category=code+focus=decode; draft_only | decode_accuracy, decode_fast |
| code | balanced | single | prefill | category=code+concurrency=single; draft_only | category=code; draft_only | category=code; draft_only | prefill_accuracy, prefill_fast |
| code | balanced | multiple | balanced | category=code+concurrency=multiple; draft_only | held concurrency=multiple | category=code+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast |
| code | balanced | multiple | ttft | category=code+concurrency=multiple; draft_only | held concurrency=multiple | category=code+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| code | balanced | multiple | decode | category=code+concurrency=multiple+focus=decode; draft_only | held concurrency=multiple | category=code+concurrency=multiple+focus=decode; draft_only | throughput_accuracy, throughput_fast |
| code | balanced | multiple | prefill | category=code+concurrency=multiple; draft_only | held concurrency=multiple | category=code+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| code | balanced | balanced | balanced | category=code; draft_only | category=code; draft_only | category=code; draft_only | throughput_accuracy, throughput_fast |
| code | balanced | balanced | ttft | category=code; draft_only | category=code; draft_only | category=code; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| code | balanced | balanced | decode | category=code+focus=decode; draft_only | category=code+focus=decode; draft_only | category=code+focus=decode; draft_only | throughput_accuracy, throughput_fast |
| code | balanced | balanced | prefill | category=code; draft_only | category=code; draft_only | category=code; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| code | fast | single | balanced | category=code+concurrency=single; draft_only | category=code; draft_only | category=code; draft_only | decode_fast |
| code | fast | single | ttft | category=code+concurrency=single; draft_only | category=code; draft_only | category=code; draft_only | prefill_fast |
| code | fast | single | decode | category=code+concurrency=single+focus=decode; draft_only | category=code+focus=decode; draft_only | category=code+focus=decode; draft_only | decode_fast |
| code | fast | single | prefill | category=code+concurrency=single; draft_only | category=code; draft_only | category=code; draft_only | prefill_fast |
| code | fast | multiple | balanced | category=code+concurrency=multiple; draft_only | held concurrency=multiple | category=code+concurrency=multiple; draft_only | throughput_fast |
| code | fast | multiple | ttft | category=code+concurrency=multiple; draft_only | held concurrency=multiple | category=code+concurrency=multiple; draft_only | throughput_fast, prefill_fast |
| code | fast | multiple | decode | category=code+concurrency=multiple+focus=decode; draft_only | held concurrency=multiple | category=code+concurrency=multiple+focus=decode; draft_only | throughput_fast |
| code | fast | multiple | prefill | category=code+concurrency=multiple; draft_only | held concurrency=multiple | category=code+concurrency=multiple; draft_only | throughput_fast, prefill_fast |
| code | fast | balanced | balanced | category=code; draft_only | category=code; draft_only | category=code; draft_only | throughput_fast |
| code | fast | balanced | ttft | category=code; draft_only | category=code; draft_only | category=code; draft_only | throughput_fast, prefill_fast |
| code | fast | balanced | decode | category=code+focus=decode; draft_only | category=code+focus=decode; draft_only | category=code+focus=decode; draft_only | throughput_fast |
| code | fast | balanced | prefill | category=code; draft_only | category=code; draft_only | category=code; draft_only | throughput_fast, prefill_fast |
| prose | exact | single | balanced | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, decode_accuracy |
| prose | exact | single | ttft | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, prefill_accuracy |
| prose | exact | single | decode | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, decode_accuracy |
| prose | exact | single | prefill | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, prefill_accuracy |
| prose | exact | multiple | balanced | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| prose | exact | multiple | ttft | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| prose | exact | multiple | decode | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| prose | exact | multiple | prefill | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| prose | exact | balanced | balanced | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| prose | exact | balanced | ttft | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| prose | exact | balanced | decode | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| prose | exact | balanced | prefill | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| prose | balanced | single | balanced | concurrency=single; draft_only | base; draft_only | base; draft_only | decode_accuracy, decode_fast |
| prose | balanced | single | ttft | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_accuracy, prefill_fast |
| prose | balanced | single | decode | concurrency=single+focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | decode_accuracy, decode_fast |
| prose | balanced | single | prefill | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_accuracy, prefill_fast |
| prose | balanced | multiple | balanced | concurrency=multiple; draft_only | held concurrency=multiple | category=prose+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast |
| prose | balanced | multiple | ttft | concurrency=multiple; draft_only | held concurrency=multiple | category=prose+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| prose | balanced | multiple | decode | concurrency=multiple+focus=decode; draft_only | held concurrency=multiple | category=prose+concurrency=multiple+focus=decode; draft_only | throughput_accuracy, throughput_fast |
| prose | balanced | multiple | prefill | concurrency=multiple; draft_only | held concurrency=multiple | category=prose+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| prose | balanced | balanced | balanced | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast |
| prose | balanced | balanced | ttft | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| prose | balanced | balanced | decode | focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | throughput_accuracy, throughput_fast |
| prose | balanced | balanced | prefill | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| prose | fast | single | balanced | concurrency=single; draft_only | base; draft_only | base; draft_only | decode_fast |
| prose | fast | single | ttft | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_fast |
| prose | fast | single | decode | concurrency=single+focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | decode_fast |
| prose | fast | single | prefill | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_fast |
| prose | fast | multiple | balanced | concurrency=multiple; draft_only | held concurrency=multiple | category=prose+concurrency=multiple; draft_only | throughput_fast |
| prose | fast | multiple | ttft | concurrency=multiple; draft_only | held concurrency=multiple | category=prose+concurrency=multiple; draft_only | throughput_fast, prefill_fast |
| prose | fast | multiple | decode | concurrency=multiple+focus=decode; draft_only | held concurrency=multiple | category=prose+concurrency=multiple+focus=decode; draft_only | throughput_fast |
| prose | fast | multiple | prefill | concurrency=multiple; draft_only | held concurrency=multiple | category=prose+concurrency=multiple; draft_only | throughput_fast, prefill_fast |
| prose | fast | balanced | balanced | base; draft_only | base; draft_only | base; draft_only | throughput_fast |
| prose | fast | balanced | ttft | base; draft_only | base; draft_only | base; draft_only | throughput_fast, prefill_fast |
| prose | fast | balanced | decode | focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | throughput_fast |
| prose | fast | balanced | prefill | base; draft_only | base; draft_only | base; draft_only | throughput_fast, prefill_fast |
| structured | exact | single | balanced | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, decode_accuracy |
| structured | exact | single | ttft | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, prefill_accuracy |
| structured | exact | single | decode | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, decode_accuracy |
| structured | exact | single | prefill | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, prefill_accuracy |
| structured | exact | multiple | balanced | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| structured | exact | multiple | ttft | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| structured | exact | multiple | decode | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| structured | exact | multiple | prefill | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| structured | exact | balanced | balanced | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| structured | exact | balanced | ttft | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| structured | exact | balanced | decode | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| structured | exact | balanced | prefill | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| structured | balanced | single | balanced | concurrency=single; draft_only | base; draft_only | base; draft_only | decode_accuracy, decode_fast |
| structured | balanced | single | ttft | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_accuracy, prefill_fast |
| structured | balanced | single | decode | concurrency=single+focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | decode_accuracy, decode_fast |
| structured | balanced | single | prefill | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_accuracy, prefill_fast |
| structured | balanced | multiple | balanced | concurrency=multiple; draft_only | held concurrency=multiple | category=structured+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast |
| structured | balanced | multiple | ttft | concurrency=multiple; draft_only | held concurrency=multiple | category=structured+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| structured | balanced | multiple | decode | concurrency=multiple+focus=decode; draft_only | held concurrency=multiple | category=structured+concurrency=multiple+focus=decode; draft_only | throughput_accuracy, throughput_fast |
| structured | balanced | multiple | prefill | concurrency=multiple; draft_only | held concurrency=multiple | category=structured+concurrency=multiple; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| structured | balanced | balanced | balanced | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast |
| structured | balanced | balanced | ttft | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| structured | balanced | balanced | decode | focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | throughput_accuracy, throughput_fast |
| structured | balanced | balanced | prefill | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| structured | fast | single | balanced | concurrency=single; draft_only | base; draft_only | base; draft_only | decode_fast |
| structured | fast | single | ttft | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_fast |
| structured | fast | single | decode | concurrency=single+focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | decode_fast |
| structured | fast | single | prefill | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_fast |
| structured | fast | multiple | balanced | concurrency=multiple; draft_only | held concurrency=multiple | category=structured+concurrency=multiple; draft_only | throughput_fast |
| structured | fast | multiple | ttft | concurrency=multiple; draft_only | held concurrency=multiple | category=structured+concurrency=multiple; draft_only | throughput_fast, prefill_fast |
| structured | fast | multiple | decode | concurrency=multiple+focus=decode; draft_only | held concurrency=multiple | category=structured+concurrency=multiple+focus=decode; draft_only | throughput_fast |
| structured | fast | multiple | prefill | concurrency=multiple; draft_only | held concurrency=multiple | category=structured+concurrency=multiple; draft_only | throughput_fast, prefill_fast |
| structured | fast | balanced | balanced | base; draft_only | base; draft_only | base; draft_only | throughput_fast |
| structured | fast | balanced | ttft | base; draft_only | base; draft_only | base; draft_only | throughput_fast, prefill_fast |
| structured | fast | balanced | decode | focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | throughput_fast |
| structured | fast | balanced | prefill | base; draft_only | base; draft_only | base; draft_only | throughput_fast, prefill_fast |
| vision | exact | single | balanced | held vision | held vision | held vision | exact, decode_accuracy |
| vision | exact | single | ttft | held vision | held vision | held vision | exact, prefill_accuracy |
| vision | exact | single | decode | held vision | held vision | held vision | exact, decode_accuracy |
| vision | exact | single | prefill | held vision | held vision | held vision | exact, prefill_accuracy |
| vision | exact | multiple | balanced | held vision | held vision | held vision | exact, throughput_accuracy |
| vision | exact | multiple | ttft | held vision | held vision | held vision | exact, throughput_accuracy, prefill_accuracy |
| vision | exact | multiple | decode | held vision | held vision | held vision | exact, throughput_accuracy |
| vision | exact | multiple | prefill | held vision | held vision | held vision | exact, throughput_accuracy, prefill_accuracy |
| vision | exact | balanced | balanced | held vision | held vision | held vision | exact, throughput_accuracy |
| vision | exact | balanced | ttft | held vision | held vision | held vision | exact, throughput_accuracy, prefill_accuracy |
| vision | exact | balanced | decode | held vision | held vision | held vision | exact, throughput_accuracy |
| vision | exact | balanced | prefill | held vision | held vision | held vision | exact, throughput_accuracy, prefill_accuracy |
| vision | balanced | single | balanced | held vision | held vision | held vision | decode_accuracy, decode_fast |
| vision | balanced | single | ttft | held vision | held vision | held vision | prefill_accuracy, prefill_fast |
| vision | balanced | single | decode | held vision | held vision | held vision | decode_accuracy, decode_fast |
| vision | balanced | single | prefill | held vision | held vision | held vision | prefill_accuracy, prefill_fast |
| vision | balanced | multiple | balanced | held vision | held vision | held vision | throughput_accuracy, throughput_fast |
| vision | balanced | multiple | ttft | held vision | held vision | held vision | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| vision | balanced | multiple | decode | held vision | held vision | held vision | throughput_accuracy, throughput_fast |
| vision | balanced | multiple | prefill | held vision | held vision | held vision | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| vision | balanced | balanced | balanced | held vision | held vision | held vision | throughput_accuracy, throughput_fast |
| vision | balanced | balanced | ttft | held vision | held vision | held vision | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| vision | balanced | balanced | decode | held vision | held vision | held vision | throughput_accuracy, throughput_fast |
| vision | balanced | balanced | prefill | held vision | held vision | held vision | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| vision | fast | single | balanced | held vision | held vision | held vision | decode_fast |
| vision | fast | single | ttft | held vision | held vision | held vision | prefill_fast |
| vision | fast | single | decode | held vision | held vision | held vision | decode_fast |
| vision | fast | single | prefill | held vision | held vision | held vision | prefill_fast |
| vision | fast | multiple | balanced | held vision | held vision | held vision | throughput_fast |
| vision | fast | multiple | ttft | held vision | held vision | held vision | throughput_fast, prefill_fast |
| vision | fast | multiple | decode | held vision | held vision | held vision | throughput_fast |
| vision | fast | multiple | prefill | held vision | held vision | held vision | throughput_fast, prefill_fast |
| vision | fast | balanced | balanced | held vision | held vision | held vision | throughput_fast |
| vision | fast | balanced | ttft | held vision | held vision | held vision | throughput_fast, prefill_fast |
| vision | fast | balanced | decode | held vision | held vision | held vision | throughput_fast |
| vision | fast | balanced | prefill | held vision | held vision | held vision | throughput_fast, prefill_fast |
| balanced | exact | single | balanced | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, decode_accuracy |
| balanced | exact | single | ttft | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, prefill_accuracy |
| balanced | exact | single | decode | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, decode_accuracy |
| balanced | exact | single | prefill | original_reference_serial_m1; reference_arithmetic | held exactness=exact | exl3_exact_shaping; reference_arithmetic | exact, prefill_accuracy |
| balanced | exact | multiple | balanced | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| balanced | exact | multiple | ttft | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| balanced | exact | multiple | decode | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| balanced | exact | multiple | prefill | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| balanced | exact | balanced | balanced | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| balanced | exact | balanced | ttft | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| balanced | exact | balanced | decode | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy |
| balanced | exact | balanced | prefill | held exact scope | held exactness=exact | held exact scope | exact, throughput_accuracy, prefill_accuracy |
| balanced | balanced | single | balanced | concurrency=single; draft_only | base; draft_only | base; draft_only | decode_accuracy, decode_fast |
| balanced | balanced | single | ttft | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_accuracy, prefill_fast |
| balanced | balanced | single | decode | concurrency=single+focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | decode_accuracy, decode_fast |
| balanced | balanced | single | prefill | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_accuracy, prefill_fast |
| balanced | balanced | multiple | balanced | concurrency=multiple; draft_only | held concurrency=multiple | concurrency=multiple; same_bits | throughput_accuracy, throughput_fast |
| balanced | balanced | multiple | ttft | concurrency=multiple; draft_only | held concurrency=multiple | concurrency=multiple; same_bits | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| balanced | balanced | multiple | decode | concurrency=multiple+focus=decode; draft_only | held concurrency=multiple | concurrency=multiple+focus=decode; draft_only | throughput_accuracy, throughput_fast |
| balanced | balanced | multiple | prefill | concurrency=multiple; draft_only | held concurrency=multiple | concurrency=multiple; same_bits | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| balanced | balanced | balanced | balanced | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast |
| balanced | balanced | balanced | ttft | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| balanced | balanced | balanced | decode | focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | throughput_accuracy, throughput_fast |
| balanced | balanced | balanced | prefill | base; draft_only | base; draft_only | base; draft_only | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| balanced | fast | single | balanced | concurrency=single; draft_only | base; draft_only | base; draft_only | decode_fast |
| balanced | fast | single | ttft | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_fast |
| balanced | fast | single | decode | concurrency=single+focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | decode_fast |
| balanced | fast | single | prefill | concurrency=single; draft_only | base; draft_only | base; draft_only | prefill_fast |
| balanced | fast | multiple | balanced | concurrency=multiple; draft_only | held concurrency=multiple | concurrency=multiple; same_bits | throughput_fast |
| balanced | fast | multiple | ttft | concurrency=multiple; draft_only | held concurrency=multiple | concurrency=multiple; same_bits | throughput_fast, prefill_fast |
| balanced | fast | multiple | decode | concurrency=multiple+focus=decode; draft_only | held concurrency=multiple | concurrency=multiple+focus=decode; draft_only | throughput_fast |
| balanced | fast | multiple | prefill | concurrency=multiple; draft_only | held concurrency=multiple | concurrency=multiple; same_bits | throughput_fast, prefill_fast |
| balanced | fast | balanced | balanced | base; draft_only | base; draft_only | base; draft_only | throughput_fast |
| balanced | fast | balanced | ttft | base; draft_only | base; draft_only | base; draft_only | throughput_fast, prefill_fast |
| balanced | fast | balanced | decode | focus=decode; draft_only | focus=decode; draft_only | focus=decode; draft_only | throughput_fast |
| balanced | fast | balanced | prefill | base; draft_only | base; draft_only | base; draft_only | throughput_fast, prefill_fast |

Hold conditions:

- fast: Additional lossy routes are held: no admitted empirical error budget or complete paired numerical/whole-workload receipts. This hold does not prohibit identical non-lossy balanced plans.
- vision: Vision-axis selection awaits a vision benchmark and applicable runtime/identity evidence.
