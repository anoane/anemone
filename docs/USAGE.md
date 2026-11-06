<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Anemone user manual

See [the v0.0.2-alpha release notes](../RELEASE_NOTES.md) for verified current benchmark scope and explicit deferred work; [the matrix](results/COMPARISON_MATRIX.md) preserves dated results and individual dispositions.

For **v0.0.2-alpha** setup and the three release serve recipes, use
[SERVER_PROFILES.md](SERVER_PROFILES.md). It maps every current serve option to
the real-parser defaults and resolved profile values at version source
`6b9e276fe4da6750be46c874e5fb5c0ee2a46871`. The detailed definitions and dated
example receipts below retain their original numerical-engine epoch; current
GPU execution results are in [the comparison matrix](results/COMPARISON_MATRIX.md).

**6 November 2026 post-merge refresh.** This manual describes `engine_batch` main **69d4d967880fb39c9ed3f82bf646f95149374d04**, distributed as **bin/tcb82/anemone**. Stable-prefix prefill, session forks, all four axes and order3 are now main options. Copy32, snapshot v5, TP5 and M25 remain branch work, as do explicitly pending O selectors. [Section 15](#15-compact-serve-flag-reference) is the compact reference for every accepted serve flag.

The implementation is authoritative. Validation receipts remain in the documentation packet; source filenames below are plain references where a location-bound hyperlink would fail after moving this file from the source root to docs/. In particular, an option being recognized is different from its selected placement, model, CUDA program and memory plan being admitted. Every executable Anemone example below has a receipt in evidence/examples.json. The clean release rebuild is byte-identical to bin/tcb82. Positive checks use its unchanged axis resolver, parser and early startup policy with one early return before model files, sockets or the CUDA backend; negative checks also run the actual shipped tcb82 binary. The additional parsed data files are the public numeric calibration fixture and the public eight-ID prompt fixture; credential files were not read. They establish syntax, option interactions checked there and resolved defaults; they do not establish GPU residency, numerical equivalence, HTTP client compatibility or throughput. Refused examples retain the binary's own refusal. No inference or GPU job was run for this manual.

`anemone` in the examples means this snapshot's binary. Run from a directory containing `messages.json`, `tokenizer.bin`, and either `model.exl3map` or `original.anemone`, or substitute your absolute paths. These are explicit path placeholders, not downloads or files created by the example. `tokenizer.bin` is Anemone's prepared tokenizer tables, not an arbitrary Hugging Face tokenizer JSON. `model.exl3map` is a valid prepared EXL3 map with its referenced files present; `original.anemone` is the original-weight pack. The local parser checks use these literal path arguments without loading them.

## Table of contents

- [1. What Anemone is](#1-what-anemone-is)
- [2. Hardware and placement](#2-hardware-and-placement)
  - [2.1 The five-card machine](#21-the-five-card-machine)
  - [2.2 Placement choices](#22-placement-choices)
  - [2.3 Native PRO arithmetic policy](#23-native-pro-arithmetic-policy)
- [3. Weight formats and prepared inputs](#3-weight-formats-and-prepared-inputs)
  - [3.1 EXL3 weights](#31-exl3-weights)
  - [3.2 Original FP8 / FP4 weights](#32-original-fp8--fp4-weights)
  - [3.3 Messages and model identity](#33-messages-and-model-identity)
- [4. Building from source](#4-building-from-source)
- [5. Quick start for every placement](#5-quick-start-for-every-placement)
  - [5.1 TP4_EXL3: one request](#51-tp4_exl3-one-request)
  - [5.2 TP4_PRO: original weights, one request](#52-tp4_pro-original-weights-one-request)
  - [5.3 PP5: original weights, multiple requests](#53-pp5-original-weights-multiple-requests)
  - [5.4 TP5: deliberate main refusals](#54-tp5-deliberate-main-refusals)
- [6. The HTTP API](#6-the-http-api)
  - [6.1 Transport, endpoints and authentication](#61-transport-endpoints-and-authentication)
  - [6.2 Chat Completions and streaming](#62-chat-completions-and-streaming)
  - [6.3 Stateless Responses](#63-stateless-responses)
  - [6.4 Tools and their results](#64-tools-and-their-results)
  - [6.5 Session identity and cache isolation](#65-session-identity-and-cache-isolation)
  - [6.6 Stable prefixes, session forks and parent hints](#66-stable-prefixes-session-forks-and-parent-hints)
  - [6.7 pi configuration](#67-pi-configuration)
  - [6.8 Codex CLI configuration](#68-codex-cli-configuration)
- [7. Four-axis configuration](#7-four-axis-configuration)
  - [7.1 Availability and meanings](#71-availability-and-meanings)
  - [7.2 Partial selections and explicit overrides](#72-partial-selections-and-explicit-overrides)
  - [7.3 Pinned tuple map and limits](#73-pinned-tuple-map-and-limits)
  - [7.4 Mapping individual knobs](#74-mapping-individual-knobs)
  - [7.5 Native help and held tuples](#75-native-help-and-held-tuples)
- [8. Every knob, grouped by subsystem](#8-every-knob-grouped-by-subsystem)
  - [8.1. Loader and turbo replicas](#81-loader-and-turbo-replicas)
  - [8.2. Memory, slots, context and 1M](#82-memory-slots-context-and-1m)
  - [8.3. Prefill and chunk planning](#83-prefill-and-chunk-planning)
  - [8.4. Decode and scheduling](#84-decode-and-scheduling)
  - [8.5. Draft and speculative policy](#85-draft-and-speculative-policy)
  - [8.6. Attention, index and Engram](#86-attention-index-and-engram)
  - [8.7. MoE and expert products](#87-moe-and-expert-products)
  - [8.8. Tensor transport, graphs and overlap](#88-tensor-transport-graphs-and-overlap)
  - [8.9. Precision, exactness, head and dense forms](#89-precision-exactness-head-and-dense-forms)
  - [8.10. Cache, prefix reuse and snapshots](#810-cache-prefix-reuse-and-snapshots)
  - [8.11. HTTP server, authentication and queues](#811-http-server-authentication-and-queues)
  - [8.12. Sampling, output and template controls](#812-sampling-output-and-template-controls)
  - [8.13. Logging, profiling and diagnostic dumps](#813-logging-profiling-and-diagnostic-dumps)
  - [8.14. Original weights and PRO arithmetic](#814-original-weights-and-pro-arithmetic)
  - [8.15. Axes and branch feature boundaries](#815-axes-and-branch-feature-boundaries)
  - [8.16. Environment variables, debugging and sanitizers](#816-environment-variables-debugging-and-sanitizers)
- [9. Complete recipes and measured tradeoffs](#9-complete-recipes-and-measured-tradeoffs)
  - [9.1 C1 code: fixed-depth EXL3 THIN](#91-c1-code-fixed-depth-exl3-thin)
  - [9.2 C8 mixed serving: BATCH with bounded decode opportunities](#92-c8-mixed-serving-batch-with-bounded-decode-opportunities)
  - [9.3 Long context and 1M](#93-long-context-and-1m)
  - [9.4 Fraser-style latency and long-prefill pauses](#94-fraser-style-latency-and-long-prefill-pauses)
  - [9.5 Opt-in order3 for concurrent short residuals](#95-opt-in-order3-for-concurrent-short-residuals)
  - [9.6 Original weights: TP4_PRO single and PP5 multiple](#96-original-weights-tp4_pro-single-and-pp5-multiple)
  - [9.7 The exactness=exact tier: original-weight reference profile](#97-the-exactnessexact-tier-original-weight-reference-profile)
- [10. Precision words and how to compare results](#10-precision-words-and-how-to-compare-results)
- [11. Monitoring, logs and receipts](#11-monitoring-logs-and-receipts)
- [12. Troubleshooting](#12-troubleshooting)
  - [12.1 Out of memory at 1M](#121-out-of-memory-at-1m)
  - [12.2 Boot failures and unsupported combinations](#122-boot-failures-and-unsupported-combinations)
  - [12.3 CLI refusal examples](#123-cli-refusal-examples)
  - [12.4 Safe recovery and reproduction](#124-safe-recovery-and-reproduction)
- [13. Source map, version boundaries and example evidence](#13-source-map-version-boundaries-and-example-evidence)
- [14. Alphabetical knob index](#14-alphabetical-knob-index)
- [15. Compact serve-flag reference](#15-compact-serve-flag-reference)
  - [Loader and turbo replicas](#loader-and-turbo-replicas)
  - [Memory, slots, context and 1M](#memory-slots-context-and-1m)
  - [Prefill and chunk planning](#prefill-and-chunk-planning)
  - [Decode and scheduling](#decode-and-scheduling)
  - [Draft and speculative policy](#draft-and-speculative-policy)
  - [Attention, index and Engram](#attention-index-and-engram)
  - [MoE and expert products](#moe-and-expert-products)
  - [Tensor transport, graphs and overlap](#tensor-transport-graphs-and-overlap)
  - [Precision, exactness, head and dense forms](#precision-exactness-head-and-dense-forms)
  - [Cache, prefix reuse and snapshots](#cache-prefix-reuse-and-snapshots)
  - [HTTP server, authentication and queues](#http-server-authentication-and-queues)
  - [Sampling, output and template controls](#sampling-output-and-template-controls)
  - [Logging, profiling and diagnostic dumps](#logging-profiling-and-diagnostic-dumps)
  - [Original weights and PRO arithmetic](#original-weights-and-pro-arithmetic)
  - [Axes and branch feature boundaries](#axes-and-branch-feature-boundaries)
  - [Additional main options in tcb82](#additional-main-options-in-tcb82)

## 1. What Anemone is

Anemone is a native C11/CUDA inference and serving engine for DeepSeek-V4.1-Flash. Its prepared pack/map, fixed model geometry, tokenizer, chat template, draft model, placement planner and CUDA programs work together. It is not a general loader for arbitrary model architectures. The same engine can load weights for inspection, generate from a messages file, or serve an OpenAI-compatible HTTP interface. The serving process schedules independent session slots and can reuse eligible prompt state or persist snapshots.

The three primary commands are `load`, `generate` and `serve`. All options precede positional files. Generate normally takes a messages file, tokenizer tables, and one or more copies of the weight source. The generate-only --prompt_token_ids route replaces the messages positional input and requires exactly one weight source; section 9.7 gives its public reference fixture. Serve takes tokenizer tables and weight copies. Load takes weight copies. Do not supply generate's messages file to serve: it would become the tokenizer argument. The parser does not resolve filenames by extension.

A THIN program verifies one drafting session at a time; it can still have several target rows for speculative verification. A BATCH program combines rows from several verifiers in one step and needs compatible row capacities, verifier counts, draft forms and memory. A wide or full program handles prompt chunks. These are prepared programs with specific shapes, rather than arbitrary row counts created on demand.

## 2. Hardware and placement

### 2.1 The five-card machine

The reference machine has **four CMP 170HX cards** and an **RTX PRO 6000**. The CMPs are GA100, arithmetic class 80, compiled for `sm_80`; the measured machine has 74 SMs and 64 GiB per CMP, on PCIe Gen2 x16. The PRO is Blackwell, **class 120 native**, compiled for **`sm_120a`**, with 96 GiB. The comparison work uses an AMD EPYC 9J14 host. Large host memory matters: the Engram tables alone occupy approximately 188.8 GiB before other host allocations. Storage bandwidth, pinned memory and card-to-host topology all affect startup and runtime. See COMPARISON_MATRIX.md, hardware and Z1 and the `Makefile`.

On the measured machine the CMP backend ordinals are `0,1,3,4` and the PRO is `2`. Thus `--card_order=0,1,3,4,2` puts the PRO last in the five-card order. Those are recorded machine-specific backend ordinals. Recheck the engine's card descriptions on another host; do not assume an NVML index or an environment visibility list has the same numbering. TP4_EXL3 automatically selects four class-80 CMPs when card order is omitted. Card ordinals are not PP5 layout indices: layout `card 4` means the fifth selected card.

### 2.2 Placement choices

| Placement | Cards and division | Weight family | Main status and intended use |
| --- | --- | --- | --- |
| `tp4_exl3` / TP4_EXL3 | All 40 target layers across four CMPs; head/draft ownership follows the selected sharding knobs. PRO excluded. | EXL3 experts plus the prepared non-expert weights | Main, experimental TP admission. The principal EXL3 THIN and BATCH serving path. FP32 wire and BF16 wire are different arithmetic recipes. |
| `tp4_pro` / TP4_PRO | Four-CMP TP prefix, layers 0–28; native PRO tail, layers 29–39, head and DSpark | Original packed FP8/FP4 family | Main. A useful original-weight single-session route. A queue depth above one is serial unless the separately gated overlap flag is enabled. |
| `pp5` / PP5 | Layer pipeline over the selected five cards; automatic fill or an explicit layout | Original packed FP8/FP4 family | Main and the no-placement default. Pipeline depth and session slots determine how multiple requests occupy the stages. Qualifying multi-session DSpark serve selects depth5 when omitted; the single-slot/generate raw default stays1 and explicit depths are retained. |
| `tp5` / TP5 | Experimental heterogeneous tensor geometry including the PRO | Original branch experiments | **Branch, not in main yet.** Main knows the enum/name internally but refuses this geometry. Host-channel/sidecar experiments have separate gates; no production quick start is claimed. |
| `tp2_pp2_pro` | Two-rank expert pairs and a PRO tail | Original | Additional main experimental selector. Requires relaxed/fast numerical admission and compatible cards/programs. |
| `tp2_single_single_pro` | Two-rank prefix plus single-card companion stages and PRO tail | Original | Additional main experimental selector, with its own geometry gates; not interchangeable with TP4_PRO. |

Do not combine an EXL3 map with an original-weight placement. Selecting a placement does not convert weights. The TP paths require generation or serving, DSpark or hybrid speculation, reference local CMP order, native PRO mode, context at least 16,384 and ordinary rows no greater than eight. `load` is not an admitted TP inference command. Context allocation must meet the model's alignment and state gates even where the lexical numeric range is broader.

### 2.3 Native PRO arithmetic policy

**The RTX PRO 6000 uses its native math by default (`--pro_emulation=0`). Outputs therefore depend on which layers it holds.** Moving the PP5 boundary can change arithmetic even with unchanged checkpoint bytes. The opt-in class-80 emulation is **`--pro_emulation=1`**. It retains class-80 summation behavior on applicable PRO paths at a reported 2–4× cost on the affected products. Current TP startup refuses nonzero emulation; use the PP5 path where that experiment is admitted. This is a numerical choice, not a device rename. Sources: `source/main.c`, `include/anemone/startup_policy.h`, `Makefile`.

## 3. Weight formats and prepared inputs

### 3.1 EXL3 weights

The EXL3 route uses routed experts from the pinned **LibertAI 2.9-bpw checkpoint**, together with the original pack's non-expert tensors. Layers 18–22 use K2 and the other main layers use K3. [MODEL_DATA.md](user/MODEL_DATA.md) gives the exact repositories, revisions, source hashes and CPU preparation commands. The `.exl3map` reader validates its logical pack extents and digests. Keep the map and its referenced source files together or preserve their expected paths. Turbo replicas can hold receipted flat copies of those logical bytes on other drives.

Four expert parts are the reference partition. Eight parts are a separate order-changing option. The FP32/full-K route and the BF16/TC route use different product, transport and reduction forms. An EXL3 reference result is a reference for that weight/arithmetic family, not official-original-model equivalence. Changing weights is distinct from changing activation or cache arithmetic.

### 3.2 Original FP8 / FP4 weights

The original family retains the prepared original FP8 dense and FP4 routed-expert weights and their scales. Use an Anemone original pack, not raw checkpoint shards as positional arguments. The recorded original frontend pack is 517,258,440,704 bytes; the raw prepared file is much larger than the EXL3 expert checkpoint. The loader's digests protect the stored contents; they do not prove arithmetic equivalence between PP5, TP, native PRO and an external inference implementation. RND_S.md, original actual-driver capture identity.

Anemone has no main command-line switch that automatically converts original FP8/FP4 weights to EXL3, and no general runtime `--kv_fp8` selector in this snapshot. The existing state includes FP16 window/pool/index representations; proposed packed reference-rounded caches are separate R&D, not an omitted user setting.

### 3.3 Messages and model identity

A minimal generate messages file is a JSON array:

```json
[{"role":"user","content":"Write a Python function that adds two integers."}]
```

The CLI renders this through the model's chat template. The server receives the equivalent array in the HTTP `messages` field. The prepared pack contains the canonical model name. Omit the HTTP `model` field to use that canonical name, or set `--model_alias=anemone` on serve and use `"model":"anemone"` as in this manual's HTTP examples. A different non-null model name receives 404.

## 4. Building from source

Build on the Linux inference host. The repository Makefile explicitly says to build on the server, not the development laptop. This snapshot targets Zen 4 (`-march=znver4`), links the CUDA driver library, and builds CUDA images for `sm_80` and `sm_120a`, including native-PRO variants. The recorded snapshot environment uses GCC 15.2 and CUDA 13.4; use a compatible toolkit/driver and toolchain rather than treating a successful laptop compile as qualification. `Makefile`, COMPARISON_MATRIX.md, A7.

Export or copy the pinned source into an isolated Linux directory, excluding any existing build tree and `.git`. From that engine directory:


```sh
make -j32 build/release/anemone
```

The recorded release binary and clean-build verification are documented in [BUILD.md](../BUILD.md)
and [VALIDATION.md](VALIDATION.md). Compare your binary hash against the recorded toolchain and flags.

The snapshot prints its full usage for top-level `--help` and `serve --help`, with exit2; those are usage refusals rather than a dedicated successful help command. `serve --help-scheduling` is dedicated help and exits0. Stats/replicate help prints their usage with exit2, while keygen reports its required user syntax with exit1. The captured help receipts preserve these differences.

The executable is `build/release/anemone`. Build outputs stay under your checkout's `build/`
directory; `make clean` removes them. A source archive is sufficient to build: the Makefile
discovers source files rather than requiring a Git worktree.

| Target | Purpose | Operational notes |
| --- | --- | --- |
| `release` | Optimized, hardened shipping binary | Checking and timing hooks compiled out. A release range can therefore refuse diagnostic settings accepted by profile/debug. |
| `profile` | Release optimization with timing hooks | Use for measured profiling; do not assume its overhead is the release overhead. |
| `debug` | Address/undefined-behavior sanitizers and hooks | Host debugging; card tests remain separately authorized work. |
| `thread` | ThreadSanitizer and hooks | Host concurrency checking. Do not infer GPU kernel coverage. |
| `check` | Lint, header checks, analysis, tests and self-test | A complete development check suite, not a required first server launch. |
| `card_tests` | Tests needing cards | Run only in an allocated, isolated GPU window. This manual did not run it. |
| `clean` | Remove generated build outputs | Scope cleanup to your own checkout/build. |

Compile parallelism consumes CPU and temporary disk. Keep sufficient free space for CUDA images, linked tests and final binaries. A full build can produce many test executables; preserve the shipping binary and receipts before removing your own build directory. Never delete another session's builds to make room.

## 5. Quick start for every placement

These are complete argument vectors with explicit source-family paths. They include workload settings, not axis selectors. The original recipes pin an explicit layer placement where needed and retain native PRO arithmetic. Their parser checks passed; model/card/program admission is still performed when you actually launch them. The examples use 65,536-token capacity to match the cited short-window evidence rather than imply a 1M serving fit.

### 5.1 TP4_EXL3: one request

The THIN recipe uses fixed five-proposal DSpark, a single serving slot, 2,048-row prefill, BF16 wire and the canonical sharded head. The fixed policy is explicit: the final-main omitted-policy default can differ with configured concurrency. The example disables prefix reuse for a reproducible cold comparison. Turn reuse on deliberately after reading the cache section.


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=2048 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=1 --prefix_reuse=0 --context_tokens=65536 tokenizer.bin model.exl3map
```

Check: `quick_tp4_exl3_serve` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.



```sh
anemone generate --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=2048 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --context_tokens=65536 messages.json tokenizer.bin model.exl3map
```

Check: `quick_tp4_exl3_generate` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


### 5.2 TP4_PRO: original weights, one request

This is the original-weight serial two-flight recipe with one serving slot. Two queued flights are not evidence of concurrent CMP/PRO execution; `--tp_pipeline_overlap` is separately gated and defaults off. This final-main example selects the compatible wide-expert geometry and omits `--original_wide_experts`, which resolves to1. Explicit0 retains the slab path. Historical serial/slab measurements are not transferred to this changed wide recipe.


```sh
anemone serve --placement=tp4_pro --numerics=exact --cmp_order=reference --linear_profile=0 --linear_order=1 --pro_emulation=0 --card_order=0,1,3,4,2 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_policy=0 --card_memory_kept_free_mib=0 --pipeline_steps=2 --official_hc_rows=0 --tp_pipeline_overlap=0 --original_chunk_core=0 --original_dense_staged=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --session_slots=1 --prefix_reuse=0 --prefill_order=0 --original_batch=0 --exl3_wide_entry=0 --tp_split_overlap=0 --tp_full_split_overlap=0 --context_tokens=65536 tokenizer.bin original.anemone
```

Check: `quick_tp4_pro_serve` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.



```sh
anemone generate --placement=tp4_pro --numerics=exact --cmp_order=reference --linear_profile=0 --linear_order=1 --pro_emulation=0 --card_order=0,1,3,4,2 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_policy=0 --card_memory_kept_free_mib=0 --pipeline_steps=2 --official_hc_rows=0 --tp_pipeline_overlap=0 --original_chunk_core=0 --original_dense_staged=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_wide_entry=0 --tp_split_overlap=0 --tp_full_split_overlap=0 --context_tokens=65536 messages.json tokenizer.bin original.anemone
```

Check: `quick_tp4_pro_generate` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


### 5.3 PP5: original weights, multiple requests

The serve example uses eight slots and omits pipeline depth, so the qualifying final-main PP5/DSpark default resolves to **five**. Explicit depth8 remains selectable and is used by the shipped original multiple-user decode axis recipe; the raw default and that explicit axis setting are distinct. Its quoted layout is one argv item; do not split it on spaces or semicolons. The generate example uses the corresponding one-slot, depth-two configuration because generate is a single request.


```sh
anemone serve --placement=pp5 --numerics=exact --cmp_order=reference --linear_profile=0 --linear_order=1 --pro_emulation=0 --card_order=0,1,3,4,2 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_policy=0 --card_memory_kept_free_mib=0 '--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4' --official_hc_rows=0 --tp_pipeline_overlap=0 --original_chunk_core=0 --original_dense_staged=0 --original_wide_experts=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --session_slots=8 --prefix_reuse=0 --prefill_order=0 --original_batch=0 --context_tokens=65536 tokenizer.bin original.anemone
```

Check: `quick_pp5_serve` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.



```sh
anemone generate --placement=pp5 --numerics=exact --cmp_order=reference --linear_profile=0 --linear_order=1 --pro_emulation=0 --card_order=0,1,3,4,2 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_policy=0 --card_memory_kept_free_mib=0 --pipeline_steps=2 '--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4' --official_hc_rows=0 --tp_pipeline_overlap=0 --original_chunk_core=0 --original_dense_staged=0 --original_wide_experts=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --context_tokens=65536 messages.json tokenizer.bin original.anemone
```

Check: `quick_pp5_generate` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


### 5.4 TP5: deliberate main refusals

**Branch, not in main yet.** The following two forms are refusal demonstrations, not working TP5 launches. Both are checked against main and rejected before any model/GPU work. The internal text parser recognizing `tp5` does not admit the geometry. The main help intentionally omits it from the supported placement menu.


```sh
anemone serve --placement=tp5 --speculative=2 --rows=8 tokenizer.bin original.anemone
```

Check: `quick_tp5_serve` — expected tcb82 refusal: usage: anemone load|generate|serve [--card_order=1,0,2] [--numerics=exact|relaxed|fast (default exact; relaxed/fast enable fixed alternate CMP sums)] [--placement=pp5|tp4_pro|tp4_exl3|tp2_pp2_pro|tp2_ Receipt: examples.json.



```sh
anemone generate --placement=tp5 --speculative=2 --rows=8 messages.json tokenizer.bin original.anemone
```

Check: `quick_tp5_generate` — expected tcb82 refusal: usage: anemone load|generate|serve [--card_order=1,0,2] [--numerics=exact|relaxed|fast (default exact; relaxed/fast enable fixed alternate CMP sums)] [--placement=pp5|tp4_pro|tp4_exl3|tp2_pp2_pro|tp2_ Receipt: examples.json.


Do not “fix” a TP5 refusal by silently substituting TP4_PRO: that changes placement and arithmetic. Use a specifically qualified Track O branch binary and its own receipt if running that research route.

## 6. The HTTP API

### 6.1 Transport, endpoints and authentication

The default listener is numeric address `127.0.0.1`, port **8321**. Serve is HTTP/1.1, with HTTP/1.0 also accepted, one request per connection and `Connection: close`. It is not a TLS listener. For another machine's access, select an appropriate bind address and supply transport/authentication through your deployment. With a key file, all routes except health require `Authorization: Bearer …`; requests without valid credentials receive 401 before their bodies are read. No key material belongs in command-line literals or this manual.

| Method and path | Behavior |
| --- | --- |
| `GET /health` | Health endpoint; engine failure changes health to 503. |
| `GET /v1/models` | Canonical model and configured alias information. |
| `POST /v1/chat/completions` | Full Chat request, JSON or SSE response. |
| `POST /v1/responses` | Stateless plaintext Responses adapter over the same rendering/decoder. |
| `GET /v1/anemone/stats` | Live counters, subject to authentication and stats-user policy. |

A POST needs one `Content-Length`; curl supplies it for the fixed inline JSON examples. Chunked `Transfer-Encoding` is refused (501). The maximum request body is 32 MiB, header bytes 16,384, header fields 64, and message count 4,096. Output/request rendering buffers are bounded at 64 MiB. A prompt must leave room for at least one generated token within the slot context. Queue exhaustion returns 503 with `Retry-After`, not a silent fallback to a smaller model. See `include/anemone/server.h`, `include/anemone/chat_completions.h`.

### 6.2 Chat Completions and streaming

Before these HTTP examples, serve with `--model_alias=anemone`, or replace/omit the model field. `ANEMONE_API_KEY` is a placeholder environment variable set privately by the operator. The example commands are not sent to a running server during documentation checks; their bodies are checked with the compiled native request readers.


```sh
curl -N -sS -H "Authorization: Bearer $ANEMONE_API_KEY" -H 'Content-Type: application/json' -H 'x-session-affinity: manual-session-001' -H 'x-client-request-id: request-001' http://127.0.0.1:8321/v1/chat/completions --data-binary '{"model":"anemone","messages":[{"role":"user","content":"Write a Python function that adds two integers."}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":256,"temperature":0,"store":false,"session_id":"manual-session-001"}'
```

Check: `http_chat_stream` — native tcb82 HTTP body reader accepted the body; curl was not sent. Receipt: examples.json.


Chat accepts system, developer, user, assistant and tool roles. Developer renders as system. Content can be text, null or supported text/image parts. PNG/JPEG base64 image data URLs have additional validation; successful text-placement admission does not establish vision qualification for an axis tuple. Prior assistant `reasoning_content`, tool calls and tool result IDs must be retained when the application needs the same rendered history.

The key request controls are:

| Field | Accepted values and meaning |
| --- | --- |
| `model` | Absent/null, canonical name or configured alias; otherwise 404. |
| `max_tokens`, `max_completion_tokens` | Integer output limits; when both are present, the smaller applies. The context still bounds total prompt plus output. |
| `temperature` | HTTP finite value 0–2; 0 is greedy. This HTTP bound is narrower than the CLI float parser. |
| `top_p` | Greater than 0 and no greater than 1. |
| `seed` | HTTP unsigned 64-bit; CLI seed is unsigned 32-bit. Omitted server seed uses the request's random seed policy. |
| `stream` | Boolean; true selects server-sent events. |
| `stream_options.include_usage` | Boolean; optional terminal usage chunk. No other stream-options control is promised. |
| `stop` | One nonempty string or up to four strings, each at most 256 bytes. Matches content, not reasoning. |
| `n` | Only 1. Multiple choices per request are not implemented. |
| `reasoning_effort` | Numeric 0–100 or aliases: none/off→0; minimal/low→50; medium/high/xhigh→75; max→100. |
| `thinking` | Object holding only `type: enabled` or `disabled`. Disabled overrides effort; enabled with zero effort refuses. Null is malformed. |
| `store` | False only; no durable response-object service. |
| `service_tier` | `auto` or `default` only. |
| `anemone_draft_usage` | `none`, `compact`, `full`; overrides the response draft-statistics verbosity. |

Streaming Chat emits a role chunk, text/reasoning/tool deltas, a finish chunk, optional usage, then `[DONE]`. A failure after the response head emits an error and no `[DONE]`; clients must not count an interrupted stream as completion. Completion tokens and reasoning tokens come from native token-phase accounting, not characters or an external tokenizer. Stops or EOS normally finish `stop`; the configured output limit finishes `length`; properly emitted closed calls finish `tool_calls` unless the limit was reached.

Unsupported generation controls receive a 400 naming the field. Examples include forced tool choice, strict function schema enforcement, non-text response formats, legacy `functions` / `function_call`, prompt-cache retention, thinking-token budgets and priority guarantees. Unknown nonempty fields are not assumed harmless. A null, false, zero or empty compatibility member is only tolerated according to the actual reader's rules.

### 6.3 Stateless Responses

Send complete plaintext history on every request and **`store:false`**. `input` can be a string or supported message/function-call/output items; `instructions` supplies the instruction text. Responses does not restore `previous_response_id`, conversations or encrypted reasoning state. This route is an adapter over the same model/decoder, not a hosted response database.


```sh
curl -N -sS -H "Authorization: Bearer $ANEMONE_API_KEY" -H 'Content-Type: application/json' -H 'x-session-affinity: manual-session-001' -H 'x-client-request-id: request-001' http://127.0.0.1:8321/v1/responses --data-binary '{"model":"anemone","input":[{"role":"user","content":[{"type":"input_text","text":"Write a Python function that adds two integers."}]}],"stream":true,"store":false,"max_output_tokens":256,"metadata":{"session_id":"manual-session-001"}}'
```

Check: `http_responses_stream` — native tcb82 HTTP body reader accepted the body; curl was not sent. Receipt: examples.json.


Responses streaming uses item, text and function events, with terminal completed, incomplete or failed events. It does not use Chat's `[DONE]` protocol as its completion contract. Request identity is computed from the original Responses body and recognized headers before internal Chat conversion. Preserve full prior function calls and outputs when resubmitting history. `source/server_responses_request.c`.

### 6.4 Tools and their results

Function tools are rendered into the prompt. The engine emits a function call; the application executes the tool and sends the result in later history. Anemone does not run arbitrary tools on the HTTP server's behalf. Use `tool_choice:auto` or `none`; `parallel_tool_calls:true` is accepted, while a false single-call guarantee is refused. Function `strict:false` or null is accepted; true is refused.


```sh
curl -N -sS -H "Authorization: Bearer $ANEMONE_API_KEY" -H 'Content-Type: application/json' -H 'x-session-affinity: manual-session-001' -H 'x-client-request-id: request-001' http://127.0.0.1:8321/v1/chat/completions --data-binary '{"model":"anemone","messages":[{"role":"user","content":"Get the weather in Rome."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Look up weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},"strict":false}}],"tool_choice":"auto","parallel_tool_calls":true,"max_tokens":256,"temperature":0,"store":false}'
```

Check: `http_tools` — native tcb82 HTTP body reader accepted the body; curl was not sent. Receipt: examples.json.


A Responses follow-up can carry the full user turn, function call and result:


```sh
curl -N -sS -H "Authorization: Bearer $ANEMONE_API_KEY" -H 'Content-Type: application/json' -H 'x-session-affinity: manual-session-001' -H 'x-client-request-id: request-001' http://127.0.0.1:8321/v1/responses --data-binary '{"model":"anemone","input":[{"role":"user","content":"Get the weather in Rome."},{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Rome\"}"},{"type":"function_call_output","call_id":"call_1","output":"{\"temperature_c\":18}"}],"store":false,"max_output_tokens":256}'
```

Check: `http_tool_results` — native tcb82 HTTP body reader accepted the body; curl was not sent. Receipt: examples.json.


An application must match the model-generated `call_id` rather than always using the illustrative `call_1`. Validate tool arguments before executing them. Partial or malformed DSML blocks can remain text; only properly recognized calls receive tool-call semantics. Streaming calls are delivered as complete call deltas once the native reader has an entire invoke, not as a guarantee of one character per event.

### 6.5 Session identity and cache isolation

Cache identity is not authentication. An API key establishes an authenticated user; client identity signals divide that user's cache namespace. The no-key mode shares the anonymous username `-`, so supply stable explicit identities if isolation is required.

| `--cache_scope` | Namespace | Consequence |
| --- | --- | --- |
| `0` (default) | Authenticated user plus every supplied recognized identity | Keeps sessions/projects distinct. Keep the complete set of identity values stable across turns. |
| `1` | Authenticated user only | Explicitly permits that user's sessions to share eligible prompt content. |
| `2` | Global | Explicitly permits cross-user/session content reuse; not the default isolation policy. |

Recognized body strings are `user`, `user_id`, `safety_identifier`, `prompt_cache_key`, `cache_salt`, `session_id`, `conversation_id`. Recognized metadata strings are `session_id`, `sessionId`, `conversation_id`, `conversationId`, `prompt_cache_key`, `cache_key`, `user_id`. Missing, null and empty strings add no identity. Values are exact decoded UTF-8, with 512 bytes per string and 8,192 bytes in the tagged namespace; bad types/limits and repeated recognized headers return 400.

Recognized headers are `session_id`, `session-id`, `x-session-id`, **`x-session-affinity`**, `x-opencode-session`, `x-opencode-session-id`, `x-conversation-id`, `agent-session-id`, `x-litellm-session-id`, `thread-id`, `x-openwebui-chat-id`, `x-openwebui-user-id`, `x-task-id`, `x-opencode-project`, `openai-project`, `openai-organization`. Their tagged presence matters: sending a session ID in a different recognized header is not necessarily the same namespace.

**`x-client-request-id` is ignored for cache identity.** It can vary per request. It is not a replacement for a session identifier. A stable `session_id` body value and/or `x-session-affinity` header can preserve a namespace; changing either recognized value changes the namespace in scope 0. Do not confuse `x-client-request-id` with the recognized `x-session-id`. See `source/server_cache.c`.

Namespace equality is only the first reuse gate. The engine also checks the rendered token prefix, reachable state, cold provenance, compressor boundaries, snapshot arithmetic salt, images and the selected exact-reuse policy. `cached_tokens=0` can therefore be correct even with a stable session ID. A grace interval reduces needless eviction when another unprotected idle slot is available; it does not reserve a slot or refuse new work when every slot is protected.

`--prefix_reuse=0` forces cold prefill and conflicts with a session directory. Mode 0 of `--prefix_reuse_exact` is the legacy policy, not a blanket same-bits assertion. Modes 1 and 2 restrict reuse to qualified same-length cold-provenance prefixes and compatible row-invariant forms; growing stable-prefix reuse is available through the main, default-off `--prefill_prefix_stable=1` form and its own admission gates. A snapshot persists model state, not a stored Responses object. Snapshot v4 preserves the complete namespace; unscoped old v3 files are not silently imported into scoped reuse.

### 6.6 Stable prefixes, session forks and parent hints

`--prefill_prefix_stable=0` retains the legacy cold form. Opting into1 creates a **new numerical identity**: fixed-capacity prefill forms and closed eight-row prefixes. Its own-cold references are separate from stable0. It requires TP4_EXL3/BF16, CE5, column owners and invariant THIN/BATCH bits, matched FULL/WIDE split modes, gather0, full/cone/skip1, one chunk, both chunk caps0, sufficient margin, wide1024 or2048, and reserved multi-slot prefill. Images and a session directory are refused on this main stable path. Copy32 and checkpoint-bearing v5 snapshots are **branch, not in main yet**.


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=2048 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --batch_core_once=2 --session_slots=1 --prefill_prefix_stable=1 --prefix_reuse=1 --prefix_reuse_exact=2 tokenizer.bin model.exl3map
```

Check: `stable_prefix` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


`--session_fork=0` remains the default. Opting into1 permits an independent child-slot copy only after matching the authenticated hard owner and a qualified stable1 donor. Startup also requires scope0 and immutable producer cold-plan capability; that capability is derived by the engine and is not a user flag. With stable0, a parser-accepted fork flag does not authorize cross-session copies: the runtime falls back cold. No-axis defaults and same-session prefix policy are not promoted to a new exactness claim by enabling forks.


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=1 --step_graphs=1 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_learned_acceptance=1 --draft_base_us=13000 --draft_tail_us=0 --draft_row_us=1200 --tp_split_least_rows=512 --card_memory_kept_free_mib=0 --batch_rows=8,8,12,12,16,24,32,48 --batch_verifiers=1,2,2,4,4,4,8,8 --batch_wide_head=1 --batch_draft_rows=1 --draft_policy=0 --batch_core_once=3 --draft_session_acceptance=0 --batch_plan_ties=0 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=8 --prefill_first=1 --prefill_decode_rounds=1 --prefill_decode_waiters=0 --prefill_order=2 --key_file=/secure/anemone.keys --session_fork=1 --prefill_prefix_stable=1 --prefill_decode_rows=0 --prefill_chunk_rows=0 --cache_scope=0 --prefix_reuse=1 --prefix_reuse_exact=0 tokenizer.bin model.exl3map
```

Check: `fork` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


A donor must be idle, not read-leased, from a different complete namespace, and carry valid ordinary state, live generations, prefix reach and stable provenance. Anonymous namespaces cannot establish authenticated hard ownership. Failed/cancelled copies must drain and clear unpublished destination history; backend failure poisons the engine. Checkpoint-only donor state and branch v5 imports are not admitted by this policy.

With forks enabled, `x-anemone-fork-of` is a session hint and `x-codex-parent-thread-id` is a thread hint; the latter can also occur in Responses `client_metadata`. Responses `client_metadata.session_id` and `thread_id` are additional tagged namespace signals in this mode. Hints are not ownership credentials and do not join the child's namespace. Values must be valid nonempty strings of1–512bytes without control characters, duplicate hint headers refuse, and supplied aliases must agree. A thread hint never matches a root-session identity. With forks disabled these fork-specific identity extensions are not enabled. `source/server_cache.c`, `source/server_forks.c`.

### 6.7 Pi configuration

Use the installed Pi agent's `~/.pi/agent/models.json`. Merge this provider/model
entry into the existing file; preserve unrelated providers. The display name is
exactly **Anemone DeepSeek-V4.1-Flash-1017**. The wire model ID stays `anemone`,
matching `--model_alias=anemone` in the
[complete server commands](SERVER_PROFILES.md#choose-a-recipe). The name does not
select an exactness tier, allocate contexts or promise four 1M sessions.

This is a template: replace the entire `baseUrl` and `apiKey` placeholder strings
before connecting. `baseUrl` is the endpoint reachable from the Pi client,
ending in `/v1`. For Pi on the same host as the example server, that is
`http://127.0.0.1:8321/v1`. For `apiKey`, use the literal JSON expression
`"$ANEMONE_API_KEY"` after setting that environment variable privately as below.
Pi's [provider documentation](https://raw.githubusercontent.com/badlogic/pi-mono/main/packages/coding-agent/docs/models.md)
describes its environment interpolation; no credential command is needed.

```json
{
  "$schema": "https://pi.dev/schemas/models.schema.json",
  "providers": {
    "anemone": {
      "baseUrl": "<populate>",
      "api": "openai-completions",
      "apiKey": "<populate: see the private key instructions below>",
      "models": [
        {
          "id": "anemone",
          "name": "Anemone DeepSeek-V4.1-Flash-1017",
          "reasoning": false,
          "input": [
            "text"
          ],
          "contextWindow": 65536,
          "maxTokens": 8192,
          "samplingParams": {
            "temperature": 1,
            "top_p": 1,
            "reasoning_effort": 0,
            "store": false
          }
        }
      ]
    }
  }
}
```

`input` enables text only. `reasoning:false` and the explicit request effort 0
keep this entry on the documented no-reasoning route. The sampling defaults are
T=1/top_p=1 and `store:false`; they are accepted Anemone request fields. This
configuration does not claim source-identical stochastic tokens. See
[request controls](SERVER_PROFILES.md#request-sampling-and-output-controls) for
T0 fixtures, reasoning and the difference between startup defaults and HTTP limits.

Set the two client limits to match the server you actually started:

| Running server recipe | `contextWindow` | Example `maxTokens` | Scope |
| --- | ---: | ---: | --- |
| Original balanced | 65536 | 8192 | Matches the published per-slot allocation; no full-context or Pi-session qualification is inferred. |
| EXL3 balanced or separate guard | 65536 | 8192 | Same client limits; the guard's identity/quality limits remain separate. |
| Original strict exact | 512 | 128 | One session only; the complete Pi prompt, tool definitions, history and output must fit 512. This is not an agent-session qualification. |

These are per-request/per-slot values. Increasing `contextWindow` in Pi does not
allocate memory or raise the engine limit. The complete prompt plus output must
fit the slot, so `maxTokens` alone is not a total-context allowance.

#### Create the server's user/key list

Run the following from the built repository on the server, choosing a new private
file. The first command refuses to overwrite an existing list. Generated keys go
directly into the file; they are not printed to the terminal:

```sh
(
  set -e -C
  umask 077
  ./build/release/anemone keygen --user=alice > anemone-users.keys
  ./build/release/anemone keygen --user=bob >> anemone-users.keys
  chmod 600 anemone-users.keys
)
```

Use one unique username/key per line, up to 64 users. Names contain 1–32 ASCII
letters, digits, dot, underscore or hyphen. Keygen emits the username, underscore
and 64 lowercase hex characters; the **complete line** is that user's bearer
key. Duplicate users/keys and malformed lists refuse. The file must be a nonempty
regular file with no group/other permissions. The server's named-user list and
the detached `stats` client's single-key file are different formats.

Keygen obtains 32 random bytes (256 bits) with Linux `getrandom(..., 0)`, using
the kernel's cryptographically secure random source and waiting for its
initialization. The implementation retries interrupted/partial reads, fails on
other read errors and clears its temporary random buffer. See
[Linux getrandom(2)](https://man7.org/linux/man-pages/man2/getrandom.2.html) and
[the implementation](../source/server_keys.c).

Add `--key_file=/absolute/private/anemone-users.keys` to the chosen full serve
command, **before its two data filenames**. It is read at startup. All routes
except health then require a valid key; the file is a user list, not an environment
variable or a JSON object. See [server authentication](#811-http-server-authentication-and-queues).

Give the Pi client only its own complete user-key line through a private channel.
In **Bash** on that client, enter it without putting the value into a command or
shell history, then start Pi from that environment:

```sh
IFS= read -r -s -p 'Anemone user key: ' ANEMONE_API_KEY
printf '\n'
export ANEMONE_API_KEY
```

Set the template's `apiKey` to `"$ANEMONE_API_KEY"`; Pi resolves it in its own
process. In Pi, open `/model` and select `Anemone DeepSeek-V4.1-Flash-1017` from the
`anemone` provider. A static shared session header would collapse independent
sessions; use the client's supported per-session identity mechanism when needed.

The template and balanced/exact capacity variants passed the installed Pi 1.3.0
public JSON schema on Linux. Actual CLI key generation, private file creation,
native user-list parsing and duplicate/permission refusals were checked with
disposable keys that were deleted without printing them. The public Pi loader
was inspected without loading user configuration, auth storage, extensions or
credential commands. This checks configuration structure and Anemone key handling;
it is not an end-to-end Pi tool/reasoning/streaming run. The separate four
[model/HTTP smokes](SERVER_PROFILES.md#evidence-and-limits-of-this-setup-check)
passed on their stated 6b9/spdx_02 epoch.

### 6.8 Codex CLI configuration

Merge this provider into the Codex user configuration. `env_key` names the variable without a dollar sign; it does not contain the secret. The provider uses HTTP Responses with SSE and disables the WebSocket transport because Anemone exposes HTTP here. [Official OpenAI configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

```toml
model = "anemone"
model_provider = "anemone"

[model_providers.anemone]
name = "Anemone"
base_url = "http://127.0.0.1:8321/v1"
env_key = "ANEMONE_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
```

Select the `anemone` provider/model in Codex and keep the server alias consistent. The client must send full plaintext history with `store:false`; a client version requiring stored response chains, encrypted reasoning restoration, hosted tools or unsupported strict schemas needs adaptation. This connection template and syntax checks do not establish that every feature of every Codex release is supported. Anemone's actual refusal identifies an unsupported request field.

## 7. Four-axis configuration

### 7.1 Availability and meanings

**All four selectors are main options at 69d4d96.** The shipped resolver, registry and generated help are authoritative; the Phase1 e4c188e catalog is retained only in the archived Phase1 packet. They apply to generate/serve, not load. There are no named presets, aliases or `--profile=NAME` in the current four-axis design. Do not copy historical profile examples from an old R&D note. Pinned axes, registry.

| Axis | Every value and its meaning |
| --- | --- |
| `--category` | `code`: code workload priority; `prose`: prose workload; `structured`: structured output; `vision`: vision benchmark category, held; `balanced`: mixed code/prose/structured measurement with code winning ties among feasible choices. |
| `--exactness` | `exact`: reference arithmetic after all required equivalence/state/form gates; `balanced`: current HC/current forms, no newly selected unqualified loss; `fast`: opt-in bounded-loss choices with paired numerical and workload receipts. Original weights admit the bounded PP THIN serial M1 exact recipe described in section 9.7. EXL3 exact and the fast tier remain refused/held. |
| `--concurrency` | `single`: C1/THIN route; `multiple`: C2–C8 route, typically BATCH for EXL3 and PP5 for original; `balanced`: one measured compromise across C1 and multiple users, held where its joint gates are incomplete. This is not a numeric slot count. |
| `--focus` | `balanced`: measured tradeoff among objectives; `ttft`: time to first token; `decode`: output rate; `prefill`: prompt-processing rate. A focus name does not promise a latency bound. |

The weight family is an admission condition, not a fifth axis. Explicit `--placement=tp4_exl3` selects the EXL3 family; absent placement uses the existing original/PP5 route. An axis cannot silently replace your checkpoint. Each family has its own tuple qualification.

### 7.2 Partial selections and explicit overrides

With **no axes selected, final-main raw CLI defaults are unchanged by the resolver**. This means the current PP5-depth and compatible TP4_PRO-wide defaults described here, not the older tcb78 defaults. There are no implicit default axis values. When some axes are omitted, S considers compatible full tuples. It applies a partial selection only when those completions resolve to a common plan. If compatible tuples require different settings or a held dependency, it refuses; it does not invent the missing category, numerical tier or focus.

Explicit low-level knobs win over axis-inserted settings in either argument order. This rule does not waive ordinary numeric, duplicate-option, startup, program or memory gates. Repeating an explicit low-level option is still an error; explicit precedence is not permission to supply two raw values. An explicit override also changes the measured recipe, so the source tuple's performance or identity evidence no longer automatically applies.

This complete EXL3 code/balanced/single/decode tuple is accepted on main. It selects the shipped `thin_full_layers` strategy; this differs from the explicitly pinned `thin_fixed` quick start:


```sh
anemone serve --placement=tp4_exl3 --speculative=2 --rows=8 --context_tokens=65536 --category=code --exactness=balanced --concurrency=single --focus=decode tokenizer.bin model.exl3map
```

Check: `axes_c1` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


A category-only partial is refused because compatible completions need different plans. A sufficiently constrained partial can succeed without filling in unset axes:


```sh
anemone serve --placement=tp4_exl3 --speculative=2 --rows=8 --context_tokens=65536 --category=code tokenizer.bin model.exl3map
```

Check: `axis_partial` — expected tcb82 refusal: axes category=code exactness=unset concurrency=unset focus=unset weight_family=exl3 axes refused: selected values need different settings for the unset axes; supply those axes. No axis defaults have b Receipt: examples.json.



```sh
anemone serve --placement=tp4_exl3 --speculative=2 --rows=8 --context_tokens=65536 --category=prose --exactness=balanced --concurrency=single tokenizer.bin model.exl3map
```

Check: `axis_partial_same_plan` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


These two examples put an explicit depth override on opposite sides of the axes. Both resolve to depth3; explicit aliases also retain the documented override key.


```sh
anemone serve --draft_positions=3 --placement=tp4_exl3 --category=code --exactness=balanced --concurrency=single --focus=decode tokenizer.bin model.exl3map
```

Check: `axis_override_before` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.



```sh
anemone serve --placement=tp4_exl3 --category=code --exactness=balanced --concurrency=single --focus=decode --draft_positions=3 tokenizer.bin model.exl3map
```

Check: `axis_override_after` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


### 7.3 Pinned tuple map and limits

| Weight family / exactness | Category | Concurrency | Focus | S strategy and scope |
| --- | --- | --- | --- | --- |
| EXL3 balanced | code | single | balanced | `thin_fixed` — source-scoped qualification; explicit overrides change the recipe |
| EXL3 balanced | code | single | decode | `thin_full_layers` — source-scoped qualification; explicit overrides change the recipe |
| EXL3 balanced | code | single | ttft, prefill | `thin_fixed` — source-scoped qualification; explicit overrides change the recipe |
| EXL3 balanced | code | multiple | decode | `batch_fixed` — source-scoped qualification; explicit overrides change the recipe |
| EXL3 balanced | code | multiple | balanced | `batch_session` — **held allocation/qualification candidate** |
| EXL3 balanced | code | multiple | ttft, prefill | `batch_adaptive` — source-scoped qualification; explicit overrides change the recipe |
| EXL3 balanced | prose, structured, balanced | single | balanced, ttft, decode, prefill | `thin_fixed` — source-scoped qualification; explicit overrides change the recipe |
| EXL3 balanced | prose | multiple | balanced | `batch_adaptive` — **held allocation/qualification candidate** |
| EXL3 balanced | structured | multiple | balanced | `batch_adaptive` — **held allocation/qualification candidate** |
| EXL3 balanced | balanced | multiple | balanced | `batch_adaptive` — **held allocation/qualification candidate** |
| EXL3 balanced | prose, structured, balanced | multiple | ttft, decode, prefill | `batch_adaptive` — source-scoped qualification; explicit overrides change the recipe |
| EXL3 balanced | balanced | balanced | balanced | `batch_compromise` — **held allocation/qualification candidate** |
| Original | code, prose, structured | single | decode | `original_tp_serial2` — source-scoped qualification; explicit overrides change the recipe |
| Original | code | multiple | decode | `original_pp_depth8` — source-scoped qualification; explicit overrides change the recipe |
| Original | prose, structured | multiple | decode | `original_pp_depth8` — source-scoped qualification; explicit overrides change the recipe |
| Original exact | code, prose, structured, balanced | single | balanced, ttft, decode, prefill | `original_reference_serial_m1` — one session, capacity 512, serial M1; no speculation or batching |
| EXL3 exact / both fast | any | any | any | EXL3 exact refuses; fast, vision and unsupported partial selections remain held. |

The all-balanced EXL3 development candidate used 2,048-row prefill and failed eight-slot allocation even at 65,536 context. It is held, not a general-purpose recommendation. Original exact is admitted only in the bounded serial M1 scope in section 9.7; broader generation/state/cache forms remain open. Fast requires measured error budgets. Vision, original TTFT/prefill, and unlisted concurrency compromises retain their own holds. An unrelated component proof does not close a full tuple's gates.

### 7.4 Mapping individual knobs

Every knob entry below states its numerical class, relevant axis domains, and the actual values written by the pinned S strategies where present. A knob omitted by the shipped resolver remains an explicit user control or inherited raw default; that is not a missing “balanced” value. Resource, authentication and logging knobs are usually orthogonal to the axes. A mapping describes intended optimization scope, not automatic admission of arbitrary joins. The canonical strategies are expanded into real commands in the recipes and examples; their names are documentation identifiers, not selectable CLI profiles.

### 7.5 Native help and held tuples

The shipped binary now exposes generated help and prefix-filtered knob lookup:


```sh
anemone help axes
```

Check: `help_axes` — tcb82 help printed; no model/backend opened. Receipt: examples.json.



```sh
anemone help knobs prefill_
```

Check: `help_knobs_prefill` — tcb82 help printed; no model/backend opened. Receipt: examples.json.


Original-weight exactness=exact now admits the one-session PP THIN serial M1 recipe at capacity 512. EXL3 exact and the fast tier remain held/refused. A feature being merged does not establish its complete model, state or quality qualification. The code/multiple/balanced and other full mixed-envelope holds remain explicit in the current registry. The tuple table below marks every held rule, including mixed rules whose component recipes separately passed. No automatic axis recipe selects order3, stable1 or session_fork1.

## 8. Every knob, grouped by subsystem

Admission shorthand follows the pinned catalog: **G0** is the general exact-case, range/arity, uniqueness, positional-file and loaded-pack contract; **T** adds TP placement/program/state gates; **B** adds compatible BATCH geometry/verifier/cost/slot gates; **F** adds the EXL3 exact-cone/tail prerequisites; **D** is the target-bit evidence condition across speculative verification widths/forms. These labels do not replace the specific interaction text below or authorize an untested join.

Use the complete recipes in [SERVER_PROFILES.md](SERVER_PROFILES.md) as a starting point; add overrides before the data filenames. The reference below covers every numeric record in main's option table and every custom or auxiliary CLI name. **All main numeric values are unsigned32 decimal scalars or bounded comma lists.** A range shown as parser bounds is lexical acceptance; the following semantic values/gates can be narrower. For lists, the range applies to each element and the listed arity is a maximum, not a requirement to fill zero placeholders. Options precede files and repeated identical option spellings refuse. The distinct draft_positions/speculative_depth aliases share a value but have separate occurrence counters: supplying both is accepted and the later value wins on this main parser. Prefer one spelling. Replicate has its separately documented last-wins repeat rule. No spaces, signs, hex, empty fields or trailing commas are accepted in a numeric list.

Defaults below come from the compiled native parser under raw PP5 and minimally selected EXL3 THIN, then explicit conditional rules in main.c. They are not silently copied from the older S catalog. A sample may intentionally set a control to0 to retain/disable a path; the receipt records the actual value. Runtime/weight/card validation remains required for all samples.

### 8.1. Loader and turbo replicas

The loader validates the pack/map and every loaded extent. TP card loading and host-arena loading have separate worker pools. Copies mean identical readable logical weight sources, not different checkpoints. Receipted replicas on other physical drives can reduce startup time without changing weight bytes. Direct I/O and page-cache eviction interact with replicas and storage topology; the auto policy is conditional. Replica creation is storage-intensive and should run outside a competing inference/load window. Replicate options are unsigned64 and are the documented last-wins exception to the main parser's duplicate rule.

<a id="knob-card_order"></a>

#### `--card_order`

**Type/range:** unsigned32 comma list, at most 16 elements; parser 0..15; semantic values: 0..15; comma list, at most16. **Default:** Automatic backend enumeration/fill; TP4_EXL3 auto-selects four CMPs. The table's zero is array storage, not a request for card0 only. **Commands:** load, generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Choose backend card ordinals in stage-fill order.

Native help describes the selected path as: `--card_order=1,0,2`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** TP4_EXL3 auto-selects exactly four CMPs when omitted; explicit order must fit selected placement. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. Shipped resolver writes: `original_tp_serial2`: `0,1,3,4,2` (all/generate); `original_pp_depth8`: `0,1,3,4,2` (all/generate); `original_pp_depth5`: `0,1,3,4,2` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0,1,3,4,2` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-reader_threads_per_copy"></a>

#### `--reader_threads_per_copy`

**Type/range:** unsigned32 decimal scalar; parser 1..16; semantic values: 1..16. **Default:** 2. **Commands:** load, generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Allocate this many asynchronous readers per pack copy.

Native help describes the selected path as: `--reader_threads_per_copy=1..16`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to tune the named read/worker path while checking startup time and identical load receipts; it does not tune token generation.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_load_threads"></a>

#### `--tp_load_threads`

**Type/range:** unsigned32 decimal scalar; parser 0..64; semantic values: 0..64. **Default:** 32. **Commands:** load, generate, serve. **Placement:** all TP placements; original or EXL3.

**Mechanics.** Load TP card weights concurrently; zero keeps the serial loader.

Native help describes the selected path as: `--tp_load_threads=0..64 (default 32; TP placements: the cards' load on N workers beside the host arena's, the same bytes; 0: the serial loads)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to tune the named read/worker path while checking startup time and identical load receipts; it does not tune token generation.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rh04: 32 workers 31.1 s versus 16 workers 33.9 s; same loaded bytes (RND_H.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-turbo_load"></a>

#### `--turbo_load`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Discover and read receipted replicas on other drives.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to tune the named read/worker path while checking startup time and identical load receipts; it does not tune token generation.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rh03 serve ready32.48s with replica versus47.93s single-copy and192.18s serial; receipts equal. Joint loader epoch, not an isolated storage benchmark (RND_H.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-map_reads"></a>

#### `--map_reads`

**Type/range:** unsigned32 decimal scalar; parser 0..4; semantic values: 0 buffered;1 drop;2 direct;3 direct+drop;4 auto. **Default:** 4 auto:3 when receipted replicas are found, otherwise1. **Commands:** load, generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Choose buffered/direct EXL3-map reads and page-cache eviction policy.

Native help describes the selected path as: `--map_reads=0..4 (default 4: 3 with replicas, else 1; 3: an EXL3 map's aligned pieces O_DIRECT, the rest buffered then dropped from the page cache; 0: all buffered and kept; 1: all buffered then dropped; 2: aligned O_DIRECT, rest kept)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to tune the named read/worker path while checking startup time and identical load receipts; it does not tune token generation.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rh01: modes 0/1/2 TP4 load 36.7..38.4/29.8..31.5/32.3 s; rh08 single-copy mode3 62.5 s versus mode1 45.9 s (RND_H.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_load_host_threads"></a>

#### `--tp_load_host_threads`

**Type/range:** unsigned32 decimal scalar; parser 0..64; semantic values: 0..64. **Default:** 16; explicit0 inherits tp_load_threads. **Commands:** load, generate, serve. **Placement:** all TP placements; original or EXL3.

**Mechanics.** Load the host arena with this many workers; zero follows TP load threads.

Native help describes the selected path as: `--tp_load_host_threads=0..64 (default 16; the host arena's threads; 0: as many as --tp_load_threads)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to tune the named read/worker path while checking startup time and identical load receipts; it does not tune token generation.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rh03:32 host workers34.616s,8 workers34.780s; rh04 chooses32 card/16 host workers after measured sweep; receipts equal (RND_H.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_load_copy_split"></a>

#### `--tp_load_copy_split`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** all TP placements; original or EXL3.

**Mechanics.** Read host chunks from copy zero and device extents from the last copy.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to tune the named read/worker path while checking startup time and identical load receipts; it does not tune token generation.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rh06/rh08 with replica reads: 29.7 versus 31.0 s; joint policy, no isolated speed claim (RND_H.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_load_reads_per_copy"></a>

#### `--tp_load_reads_per_copy`

**Type/range:** unsigned32 decimal scalar; parser 0..64; semantic values: 0..64. **Default:** 0. **Commands:** load, generate, serve. **Placement:** all TP placements; original or EXL3.

**Mechanics.** Limit TP in-flight reads per copy; zero is unlimited.

Native help describes the selected path as: `--tp_load_reads_per_copy=0..64 (default 0; reads in flight a copy at most, 0: no limit)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to tune the named read/worker path while checking startup time and identical load receipts; it does not tune token generation.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rh03:limit4 36.245s,limit8 35.854s; rh01 limit2 79.957s in older epoch. No selected limit improves retained unlimited default (RND_H.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-rate_mib-replicate"></a>

#### `--rate_mib` — replicate

**Type/range:** unsigned64 scalar; 0..17592 MiB/s (UINT64_MAX /1048576 /1000000000). **Default:** 2048 **Commands:** replicate. **Placement:** all placements/recipes.

**Mechanics.** Throttle replica copy rate in MiB/s; zero is unlimited.

**Use and interactions.** Parser accepts repeated replica options with last value; source/target are positional. Admission: source pack must open; target capacity/receipt/device gates.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/replicas.c:488, include/anemone/replicas.h. Unmeasured isolated effect. Source: `source/replicas.c` and shipped knob catalog.


<a id="knob-floor_gib-replicate"></a>

#### `--floor_gib` — replicate

**Type/range:** unsigned64 scalar; 0..17179869183 GiB (UINT64_MAX /1073741824). **Default:** 100 **Commands:** replicate. **Placement:** all placements/recipes.

**Mechanics.** Keep this much free space on replica volumes.

**Use and interactions.** Parser accepts repeated replica options with last value; source/target are positional. Admission: source pack must open; target capacity/receipt/device gates.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/replicas.c:488, include/anemone/replicas.h. Unmeasured isolated effect. Source: `source/replicas.c` and shipped knob catalog.


<a id="knob-prune-replicate"></a>

#### `--prune` — replicate

**Type/range:** unsigned64 scalar; 0 or1. **Default:** 0 **Commands:** replicate. **Placement:** all placements/recipes.

**Mechanics.** Remove receipted older-source replicas before copying.

**Use and interactions.** Parser accepts repeated replica options with last value; source/target are positional. Admission: source pack must open; target capacity/receipt/device gates.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/replicas.c:488, include/anemone/replicas.h. Unmeasured isolated effect. Source: `source/replicas.c` and shipped knob catalog.


### 8.2. Memory, slots, context and 1M

Context capacity is prompt plus generated tokens per independent slot. Ordinary rows size a THIN verification program; wide rows size prefill programs. Neither is a client batch size or a guarantee of available card memory. The numeric parser deliberately has broad bounds, followed by alignment, placement and prepared-program gates. The planner subtracts kept-free bytes before placing weights/state/scratch. An explicit PP5 layout names selected-card indices, includes every required stage exactly once and is one quoted argv item.

<a id="knob-card_memory_kept_free_mib"></a>

#### `--card_memory_kept_free_mib`

**Type/range:** unsigned32 decimal scalar; parser 0..1048576; semantic values: 0..1048576. **Default:** 0. **Commands:** load, generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Subtract this many MiB from each card's placement budget.

Native help describes the selected path as: `--card_memory_kept_free_mib=N`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. Shipped resolver writes: `batch_fixed`: `0` (all/generate); `batch_adaptive`: `0` (all/generate); `batch_session`: `0` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `0` (all/generate); `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate); `thin_full_layers`: `0` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-session_slots"></a>

#### `--session_slots`

**Type/range:** unsigned32 decimal scalar; parser 1..8; semantic values: 1..8. **Default:** 1. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Allocate this many independent active session slots.

**Use and interactions.** Values above1 require serve; device draft plus THIN multiple slots requires pipeline_steps>1. Measured S10/e4c188e one1048576-token trueTHIN slot allocates with15MiB minimum CMP free: selected8/256, both2048/1024 reference pairs,15MiaAI signatures and cold9 pass; cold3795tok/s, Fraser1435.85/1443.60ms. Capacity plus short requests only, not full1Mwork/pi/peak envelope. Earlier S04 BATCH8/one-verifier slot125MiB/cold3207 retains its different recipe/source. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. Shipped resolver writes: `batch_fixed`: `8` (serve); `batch_adaptive`: `8` (serve); `batch_session`: `8` (serve); `thin_fixed`: `1` (serve); `batch_compromise`: `8` (serve); `original_tp_serial2`: `1` (serve); `original_pp_depth8`: `8` (serve); `original_pp_depth5`: `8` (serve); `thin_full_layers`: `1` (serve). The `original_reference_serial_m1` exact strategy additionally inserts `1` for serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-wide_rows"></a>

#### `--wide_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..4096; semantic values: 0..4096. **Default:** 0. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Prepare a wide prefill program of this many rows; zero disables it.

**Use and interactions.** Nonzero requires prefill_cone and >=64 rows. TP preparation permits at most2048 rows with split_overlap enabled, otherwise1024. Overall row capacity must fit the model geometry. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. Shipped resolver writes: `batch_fixed`: `1024` (all/generate); `batch_adaptive`: `1024` (all/generate); `batch_session`: `1024` (all/generate); `thin_fixed`: `2048` (all/generate); `batch_compromise`: `2048` (all/generate); `original_tp_serial2`: `1024` (all/generate); `original_pp_depth8`: `1024` (all/generate); `original_pp_depth5`: `1024` (all/generate); `thin_full_layers`: `2048` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-rows"></a>

#### `--rows`

**Type/range:** unsigned32 decimal scalar; parser 1..8; semantic values: 1..8. **Default:** 1 on every raw placement; the usable speculative TP recipes explicitly select8. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Set ordinary THIN step row capacity (and proposal capacity minus one).

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. Shipped resolver writes: `batch_fixed`: `8` (all/generate); `batch_adaptive`: `8` (all/generate); `batch_session`: `8` (all/generate); `thin_fixed`: `8` (all/generate); `batch_compromise`: `8` (all/generate); `original_tp_serial2`: `8` (all/generate); `original_pp_depth8`: `8` (all/generate); `original_pp_depth5`: `8` (all/generate); `thin_full_layers`: `8` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `1` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-context_tokens"></a>

#### `--context_tokens`

**Type/range:** unsigned32 decimal scalar; parser 1..1048576; semantic values: 1..1048576. **Default:** 16384 per slot for every raw placement; larger capacities are explicit. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Allocate the maximum prompt-plus-generation context per slot.

**Use and interactions.** Model capacity must be positive and a multiple of16; TP also requires at least16384. Measured S10 trueTHIN one1048576-token slot allocates with15MiB minimum CMP free and short C1/cold/Fraser work passes (gxf_s10_single_1m_strict_s10_v1.json); not full1M work, pi, peak memory or eight1Mslots. Earlier S04 BATCH8/one-verifier125MiB is historical and separate. Capacity is an explicit raw override, not an inserted axis default. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The `original_reference_serial_m1` exact strategy additionally inserts `512` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-layout"></a>

#### `--layout`

**Type/range:** text/enum; <=4096-byte statements joined by semicolon+space: <units>: card <c> or experts kept: <units>: <n>. **Default:** automatic fill **Commands:** load, generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Override PP5 unit placement.

**Use and interactions.** PP5 only. Units are embedding/vision/head/draft head or layers a..b, c (up to16 ranges/statement); exact spaces required. Each stage must be assigned once, ranges must exist in order, card indices and kept counts must fit, and card budgets may not be exceeded (placement.h:16..35). Admission: G0.

**Precision class:** order-changing. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. Shipped resolver writes: `original_pp_depth8`: `vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4` (all/generate); `original_pp_depth5`: `vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts the pinned card layout in section 9.7 for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


### 8.3. Prefill and chunk planning

One chunk in flight uses the exact cone planner. The front region, full programs behind it, whole256-row granules, tail programs and THIN tail remainders form one planned sequence. Adding ladder entries can reduce wasted work but increases the prepared set and memory. The automatic tail choices are conditional: eligible THIN with wide2048 gets tail448/front2/thin-tail1/fine1; eligible BATCH retains tail128/front1/thin-tail0/fine0 for the measured 1M memory plan. Ineligible routes use tail0. Explicit values override these choices but must pass the shared tail gates.

<a id="knob-full_steps"></a>

#### `--full_steps`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Use full programs behind the prefill front.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `original_tp_serial2`: `1` (all/generate); `original_pp_depth8`: `1` (all/generate); `original_pp_depth5`: `1` (all/generate); `thin_full_layers`: `1` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-chunks_in_flight"></a>

#### `--chunks_in_flight`

**Type/range:** unsigned32 decimal scalar; parser 1..8; semantic values: 1..8. **Default:** 1. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Bound chunk steps in flight; one uses the exact cone planner.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `original_tp_serial2`: `1` (all/generate); `original_pp_depth8`: `1` (all/generate); `original_pp_depth5`: `1` (all/generate); `thin_full_layers`: `1` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `1` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** rc02ab exact1 versus2: 30K11.859 ->9.430 s, Fraser2.431 ->1.857 s; same bits in that recipe (RND_C.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_margin"></a>

#### `--prefill_margin`

**Type/range:** unsigned32 decimal scalar; parser 1..3967; semantic values: 1..3967. **Default:** 127. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Keep this many full rows beyond the prefill cone.

Native help describes the selected path as: `--prefill_margin=1..3967 (default 127; 3967 the reference's; with --chunks_in_flight=1, the exact plan (64-row chunks only where 1,024-row chunks, two in flight, take them): full rows beyond a prompt's cone; fewer: more front rows, the same bits, a kept prefix there runs a cone again)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rc02ab margin127 exact plan: 30K about -1.0 s; joint recipe, same bits (RND_C.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_fast_plan"></a>

#### `--prefill_fast_plan`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Choose the largest fitting prefill program instead of the exact plan.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** order-changing. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_chunk_rows"></a>

#### `--prefill_chunk_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..4096; semantic values: parser0..4096; nonzero multiple256 and <=wide_rows; measured scope<=2048. **Default:** 0. **Commands:** generate, serve. **Placement:** all placements with exact one-flight prefill; automatic dynamic policy only qualified BATCH1024.

**Mechanics.** Cap all exact-plan prefill chunks.

Native help describes the selected path as: `--prefill_chunk_rows=0|256..2048 (whole 256-row granules; default 0; cap chunks before the original tail)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Requires chunks_in_flight1, prefill_fast_plan0; reject with P exact reuse until combined proof. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Fixed-cap solo references pass. M7d86c45 default passes three reference fixtures and15/15mixed texts; this is not blanket proof of every cap/placement (packet RND_M.md19:14). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_cone_skip"></a>

#### `--prefill_cone_skip`

**Type/range:** unsigned32 decimal scalar; parser 0..3; semantic values: 0 none;1 dead decoder core/MoE/wide trunk;2 last-row cone;3 mode1 with every trunk. **Default:** 1. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Skip dead cone work; modes two/three extend the retained optimization.

Native help describes the selected path as: `--prefill_cone_skip=0..3 (default 1: a full step's rows past the cone skip the decoder layers' cores, routed experts and wide trunks, the same bits; a later prompt keeping --prefill_margin tokens fewer or more runs no cone again; 2: the last row's cone, one fewer; 3: 1 with every trunk; 0: none skip)] --chunks_in_flight=1..8 (default 1, the exact plan; 2 the reference plan)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_tail_rows"></a>

#### `--prefill_tail_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..448; semantic values: 0,128,320,448. **Default:** 0 when ineligible;128 with eligible wide1024 or BATCH;448 with eligible wide2048+ THIN. BATCH retains128. **Commands:** generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Prepare 128/320/448-row full tail programs; zero disables them.

Native help describes the selected path as: `--prefill_tail_rows=0|128|320|448 (default 128, 448 with --wide_rows=2048 and no BATCH, where it applies: tp4_exl3, --chunks_in_flight=1, --wide_rows of 1024 or more, --exl3_tc_moe=1, --tp_trunk_mma=1: the exact plan's 64-row tail chunks two (320: five, 448: seven or five, else two) a step, each in the 64-row chunk's forms; 0 one a step)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Nonzero only128/320/448; requires TP4_EXL3, chunks_in_flight1, fast_plan0, full_steps1, wide_rows>=1024, TC MoE and trunk MMA. Admission: G0; T; F.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** gsk01 448 versus320 THIN: 2K1696 ->1722 tok/s, cold3798 ->3818; same bits. BATCH defaults128 for memory (RND_K.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_ladder_1024"></a>

#### `--prefill_ladder_1024`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Add 1024-row programs below larger wide programs.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri01 with front_tail: 30K -82 ms; same bits, joint recipe (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_ladder_1536"></a>

#### `--prefill_ladder_1536`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Add 1536-row programs below larger wide programs.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri06 1905-token prompt -84 ms; same bits (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_ladder_fine"></a>

#### `--prefill_ladder_fine`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1 for wide2048+ THIN;0 for BATCH or smaller wide programs. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Add 1792/1280/768-row exact-plan programs.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri07/gsi10 THIN cold prefill +0.8%, about200 MiB/card extra; same bits (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_front_tail"></a>

#### `--prefill_front_tail`

**Type/range:** unsigned32 decimal scalar; parser 0..2; semantic values: 0..2. **Default:** 0 without tail;1 with other tails;2 with tail448. **Commands:** generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use tail programs for the front region; two also uses the longer full tails.

Native help describes the selected path as: `--prefill_front_tail=0|1|2 (default 1 where the tail program applies, 2 with the 448-row tail: a wide one for the front region's tail too, its bits; 2: its 320- and 448-row ones and the thin-tail ones too)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Nonzero requires a tail program. Admission: G0; T; F.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rk04k THIN with448 tail: 30K8.214 ->8.190 s; same bits (RND_K.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_tail_overlay"></a>

#### `--prefill_tail_overlay`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Overlay tail TC MoE workspace on prefill workspace.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** 320-tail workspace -244 MiB/card; same launches/bits (RND_K.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_tail_sets"></a>

#### `--prefill_tail_sets`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Run tail trunk products over their 64-row groups in one launch.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rk03c 2K prompt -12 ms at320 tail, -15 ms at448; same bits (RND_K.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_thin_tail"></a>

#### `--prefill_thin_tail`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1 with a tail and no BATCH;0 otherwise. **Commands:** generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use 16..56-row programs for rows below a 64-row tail.

**Use and interactions.** Nonzero requires a tail program; automatic default only without BATCH. Admission: G0; T; F.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rk09y/z 2K1.144 ->1.117s (-27ms),1905token1.141 ->1.075s (-66ms),same bits after final fixes; earlier rejected versions must not be reused (RND_K.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_dead_halves"></a>

#### `--prefill_dead_halves`

**Type/range:** unsigned32 decimal scalar; parser 0..2; semantic values: 0..2. **Default:** 2. **Commands:** generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Trim cone-dead sums; mode two also trims dead copies.

Native help describes the selected path as: `--prefill_dead_halves=0|1|2 (default 2; 1 with the exact plan and the cone: a full program of 256 rows or more leaves the TP sums and gathers of a decoder layer's half whose outputs there the cone never reads (its last row at least 256 less the margin rows before the cone end); 2: and each sum's copies the rows before a half's first such row; the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri01 mode1 Fraser -76ms,30K -68ms; joint mode7+pool+dead2 Fraser -117ms/30K -276ms; same bits (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_cone"></a>

#### `--prefill_cone`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Use a bounded front region with a prepared prefill cone.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this when changing the planned chunk/tail path; compare cold-prefill and TTFT receipts and check the additional prepared-memory cost.

**Precision class:** same bits. **Four-axis mapping:** Category: content independent within supported text/vision gates. Exactness: use the entry's same-bits/order qualification. Concurrency: prepared THIN or BATCH geometry. Focus: prefill and TTFT; memory/decoder pauses interact. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `original_tp_serial2`: `1` (all/generate); `original_pp_depth8`: `1` (all/generate); `original_pp_depth5`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_prefix_stable"></a>

#### `--prefill_prefix_stable`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0 or1. **Default:** 0. **Commands:** generate, serve. **Placement:** TP4_EXL3 stable-prefix recipe.

**Mechanics.** Use fixed-capacity prefill forms with closed eight-row prefixes.

**Use and interactions.** Requires TP4_EXL3 BF16 wire, column owners, CE5, gather0, matched FULL/WIDE split forms, batch_thin_bits1, full_steps1/cone1/skip1, chunks1, fast_plan0, both M caps0, reserved prefill, margin>=7, wide1024 or2048; no images; isolated prefix19 refuses session storage (full P23 format5 is a separate unqualified prototype); multi-slot requires prefill-first or order1/2. Growing reuse needs known cold provenance, nonshrinking child, generated/sampling-row clipping, floor8, child cold-plan/cone and post-rounding reach checks; hard-owner authorization is separate. Parent/child must retain immutable stable1 identity. Integrated in merge_batch. Admission: Default off; real parser and engine admission apply.

**Precision class:** order-changing (new cold form); same bits relative to its qualified own-cold recipe. **Four-axis mapping:** Candidate TP4_EXL3 text cache form in exactness=exact, conditional on its cold/state/capacity gates; no automatic category/concurrency/focus selection. Copy32 targets prefill/TTFT copying only after device proof.

**Measured effect.** Prefix-only19/8c18f8a on main39980 is P final knob-default0/exact-cache-tier request; do not merge full P80 copy/v5 prototype. Fresh39 six complete cold arms14/19/14,114requests pass unchanged no-loss reader: THIN prefill ratio0.99884138(95%0.99679661-1.00268666),TTFT0.99777899(0.99398414-1.00548634); BATCH prefill0.99985033(0.99910103-1.00116771),TTFT1.00159642(0.99389811-1.00713069). Historical source15THIN strict failure retained. Whole39 exit1 belongs to separate copy32OOM diagnostic, not cold-arm failure. Core174native/checkpoint/1M and shared accepted14/Q11 references pass; standalone19 CLI refs DID NOT RUN. Main resulting old/new2048/1024 gate still required; no automatic S exact tuple or full official-reference claim (RND_P.md5Nov14:32,14:42; evidence/xc39_v3/cold_performance.json; MERGE_v3.md). Integrated prefix-only candidate8c18f8a; no copy32 or snapshotv5. Source: `source/main.c` and shipped knob catalog.


### 8.4. Decode and scheduling

Prefill ordering chooses which prompt advances; decode cadence chooses when already-started requests get their ordinary program. The two decisions solve different waits. M12's bounded BATCH defaults apply only to serve, wide1024, prefill-first, BF16 column-owner sums, THIN bits and the exact one-chunk plan. They resolve to cap1024/round1/waiters0/order2. THIN, generate and other routes keep cap0/round1/waiters0/order0. Explicit cap0 opts out. BATCH row lists and verifier lists are parallel prepared-program descriptions; duplicates in these two lists are allowed, while most numeric lists require distinct values.

<a id="knob-batch_rows"></a>

#### `--batch_rows`

**Type/range:** unsigned32 comma list, at most 8 elements; parser 1..64; semantic values: 1..64; comma list, at most8. **Default:** No BATCH list (zero-filled internal array); THIN programs are prepared. **Commands:** generate, serve. **Placement:** EXL3 BATCH or the separately constrained original_batch pilot.

**Mechanics.** Prepare BATCH programs at these row capacities instead of THIN programs.

Native help describes the selected path as: `--batch_rows=R[,R..] (1..64 each; tp4_exl3 with --pipeline_steps=1: BATCH programs of those rows in place of the thin ones, several sessions' rounds a step; 8 rows or fewer walk as thin programs)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Requires pipeline_steps=1 and either TP4_EXL3 or the separately constrained original_batch1 TP4_PRO pilot. Multiple EXL3 programs with device draft require batch_draft_rows=1; repeated row counts require explicit distinct verifier counts. Overall prepared capacity max(rows,wide_rows,batch_rows,draft_block) must be at least64; EXL3 BATCH programs may use1..64 rows. Original BATCH requires exactly48 rows/eight verifiers and all original_batch row gates; it is not arbitrary original BATCH admission. Admission: G0; T; B.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_fixed`: `8,8,12,12,16,24,32,48`, `8` (generate); `batch_adaptive`: `8,8,12,12,16,24,32,48`, `8` (generate); `batch_session`: `8,8,12,12,16,24,32,48`, `8` (generate); `batch_compromise`: `8,8,12,12,16,24,32,48`, `8` (generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_context_slabs"></a>

#### `--batch_context_slabs`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** EXL3 BATCH or the separately constrained original_batch pilot.

**Mechanics.** Process BATCH draft context in THIN-sized eight-row slabs.

**Use and interactions.** Nonzero requires batch_rows. Admission: G0; T; B; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_thin_bits"></a>

#### `--batch_thin_bits`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Use THIN numerical splits in BATCH trunk/head/TC-MoE products.

Native help describes the selected path as: `--batch_thin_bits=0|1 (default 1; a BATCH program's rows the thin steps' bits: trunk_mma thin splits, head split 4, TC MoE short forms)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T; B.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** order-changing. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_tc_paired"></a>

#### `--batch_tc_paired`

**Type/range:** unsigned32 decimal scalar; parser 0..64; semantic values: 0..64. **Default:** 0. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Pair TC MoE short forms from this many BATCH rows; zero disables pairing.

Native help describes the selected path as: `--batch_tc_paired=0..64 (default 0; with it, BATCH steps of so many rows or more: the TC MoE's short forms two K slices a warp item, the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T; B.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rj01_card all compared forms identical; paired48 I640 +8.2us/call,I512 +18.4us; all tested sizes slower, kept off (RND_J.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_row_sets"></a>

#### `--batch_row_sets`

**Type/range:** unsigned32 decimal scalar; parser 0..24; semantic values: 0,8,16,24. **Default:** 0. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Use trunk-product row sets above the grouped-row threshold.

Native help describes the selected path as: `--batch_row_sets=0|8|16|24 (default 0; a BATCH step above --batch_grouped_rows: trunk products by the 1-8-row forms in sets of so many rows, the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Must be a multiple of8; used above batch_grouped_rows. Admission: G0; T; B.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rj02s same bits; six-shape48-row total r16/r24/r8 284.7/298.0/330.8us versus192.5us _64+reduce; all slower, kept off (RND_J.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_core_once"></a>

#### `--batch_core_once`

**Type/range:** unsigned32 decimal scalar; parser 0..3; semantic values: 0..3. **Default:** PP5/TP4_PRO and other original placements: 2; TP4_EXL3 THIN: 3. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Compute BATCH attention core once; mode2 adds window writes and mode3 pools/deferred slots.

Native help describes the selected path as: `--batch_core_once=0..3 (default 3 for tp4_exl3, 2 elsewhere; 3: pools and deferred slots; a BATCH step's core and window once over its slabs; 1: the core alone; 0: a slab at a time; the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Accepts0..3. Main39980 omitted EXL3 CLI default3 is active on eligible BATCH; other placements retain2, API description0 is separate. Explicit S code/balanced/multiple/decode selects3; other recorded S BATCH recipes retain2. S10/e4c188e fixed/core3 versus RATE2/core2 full matrix and selected8/256 pass: C8code469.537/432.990, prose382.997/450.231, structured787.722/787.726tok/s. Multiple knobs differ, so these are recipe observations, not an isolated core gain; own CPU compilation overlaps part of decode. Same24 isolated mixed core2/3 text gates pass112cross/96self with code/prose C8task-geometric+0.529%/+1.115%. N actual26 code267.308/298.437/379.376/472.154 is cross-binary. Dedicated reference/safety proofs retain their epochs. See axes.json and RND_N/S.md. Admission: G0; T; B.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_fixed`: `3` (all/generate); `batch_adaptive`: `2` (all/generate); `batch_session`: `2` (all/generate); `thin_fixed`: `2` (all/generate); `batch_compromise`: `2` (all/generate); `thin_full_layers`: `2` (all/generate).

**Measured effect.** rj03p BATCH48 grouped48: 43.63 -> 43.02/42.41 ms at1/2 (-1.4/-2.8%), same bits; gsj06a 84/84 texts, 24..48-row programs -2.1..-4.3% (RND_J.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_grouped_rows"></a>

#### `--batch_grouped_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..48; semantic values: 0..48. **Default:** 16. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Use grouped THIN trunk forms through this BATCH row count.

Native help describes the selected path as: `--batch_grouped_rows=0..48 (default 16; with it, a BATCH step of 9 to so many rows: trunk products by the 1-8-row forms over its row groups of 8, one launch, the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T; B.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rj03p BATCH48 grouped16 38.44 versus grouped48 43.63 ms (-11.9%); same bits (RND_J.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_step_us"></a>

#### `--batch_step_us`

**Type/range:** unsigned32 comma list, at most 8 elements; parser 0..1000000000; semantic values: 0 off, or distinct positive costs1..4294967295; never mix0 into a list; comma list, at most8. **Default:** No raw prices. Eligible omitted-policy measured EXL3 BATCH receives19500,21500,28600,30000,32000,39000,46000,58000us. Explicit policy or costs preserve their former behavior. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Supply per-program measured costs for the BATCH expected-token planner.

**Use and interactions.** A sole0 explicitly disables the menu, including on unbatched shapes. Any multi-entry list containing0 refuses. Positive entries require batch_rows and exactly one distinct cost per BATCH program, ordered by row count then verifier count; duplicate prices refuse. With omitted policy/costs only, the measured EXL3 BF16/DSpark5/reference/core2-or3 family and exact canonical rows8,8,12,12,16,24,32,48/verifiers1,2,2,4,4,4,8,8 receives19500,21500,28600,30000,32000,39000,46000,58000. Explicit policy, custom/incomplete/reordered geometry or nonqualifying math gets no automatic menu. Explicit positive prices remain selectable; source/engine.c validates count. Admission: G0; T; B; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_adaptive`: `19500,21500,28600,30000,32000,39000,46000,58000`, `19500` (generate); `batch_session`: `19500,21500,28600,30000,32000,39000,46000,58000`, `19500` (generate); `batch_compromise`: `19500,21500,28600,30000,32000,39000,46000,58000`, `19500` (generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_verifiers"></a>

#### `--batch_verifiers`

**Type/range:** unsigned32 comma list, at most 8 elements; parser 1..8; semantic values: 1..8; comma list, at most8. **Default:** Unset/derived with the BATCH geometry; not a literal single verifier0. **Commands:** generate, serve. **Placement:** EXL3 BATCH or the separately constrained original_batch pilot.

**Mechanics.** Supply each BATCH program's drafting-session capacity.

Native help describes the selected path as: `--batch_verifiers=V[,V..] --batch_step_us=0|US[,US..] (0: off; qualified BF16/DSpark5 menu defaults for rows 8,8,12,12,16,24,32,48 / verifiers 1,2,2,4,4,4,8,8 when policy is omitted)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Nonempty requires batch_rows and an entry for every BATCH program; extra trailing entries are ignored. Each used count must fit session_slots and each BATCH row/verifier pair must be unique. Attachment requires an idle, unstarted BATCH driver (source/step_image_programs.c:177). Admission: G0; T; B; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_fixed`: `1,2,2,4,4,4,8,8`, `1` (generate); `batch_adaptive`: `1,2,2,4,4,4,8,8`, `1` (generate); `batch_session`: `1,2,2,4,4,4,8,8`, `1` (generate); `batch_compromise`: `1,2,2,4,4,4,8,8`, `1` (generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_first"></a>

#### `--prefill_first`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Prefer another session's wide/full prefill chunk before ordinary steps.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_fixed`: `1` (serve); `batch_adaptive`: `1` (serve); `batch_session`: `1` (serve); `batch_compromise`: `1` (serve).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_order"></a>

#### `--prefill_order`

**Type/range:** unsigned32 decimal scalar; parser 0..3; semantic values: 0 round-robin chunks;1 command arrival order;2 shortest remaining prompt;3 shortest with concurrent short residuals. **Default:** 2 only for the qualified default BATCH1024 cap/round/waiter combination;0 otherwise. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Choose round-robin chunks, command arrival order, shortest remaining prompt or opt-in shared short residuals.

Native help describes the selected path as: `--prefill_order=0|1|2|3 (bounded BATCH default 2, else 0; 3: shortest with concurrent short tails; 2: prompts one at a time, the one with the fewest rows left first; 1: in their commands' order, each first token behind its own prompt; 0: the turns, a chunk each; the same chunks)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Order3 requires the measured row-independent EXL3 BF16 BATCH family; rejects prefix_reuse_exact1/2, session_fork1 and prefill_prefix_stable1. Automatic order2 and explicit0/1/2 retain their defaults and gates. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_fixed`: `2` (serve); `batch_adaptive`: `2` (serve); `batch_session`: `2` (serve); `batch_compromise`: `2` (serve); `original_tp_serial2`: `0` (serve); `original_pp_depth8`: `0` (serve); `original_pp_depth5`: `0` (serve). The `original_reference_serial_m1` exact strategy additionally inserts `0` for serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** N xd13/frozen24 prefix_reuse1 same-binary order2->0 warm-code probe: C1/C2/C4/C8 TTFT medians15.348/26.221/52.603/131.152->15.295/26.310/42.449/55.001ms; following-token rates268.148/347.388/448.756/541.992->268.891/347.679/530.509/722.848tok/s. Each arm has47 strict records with paired cache classes and equal T0 text/native-count/finish; candidate follows control by44.0-44.5s. BATCH8 one-owner C1 is not THIN. Actual fixed0/session0/learn1/costs14300,0,690/core2 has no batch_step_us menu. Code-only warm TTFT/decode evidence, not all-category, mixed, long-prompt continuity or a changed S default/axis (RND_N/M.md5Nov05:17,05:32; evidence/xd13_v1_measure/warm_order_ttft.json). Historical isolated N source8dd9dfc/main39980 adds order3: short no-chunk residuals fitting the prepared THIN budget share a head thin turn, retaining order2 fairness, long chunks and decode-due exclusions. CLI requires qualified bounded_batch1024/BF16 row-independent family and prefix_reuse_exact0; generic API fixtures are not broader CLI admission. Future P/Q join must also exclude stable1 canonical form even global exact0/fork0, and fork1/global0, until their cold-plan proof. Preserve prior automatic0/2 outcomes or refusals. Three builds,123parser cases, focused ASan/TSan and87CUDA-image identity pass; paired xd18 on frozen35 passes both-width reference stages; complete cross-order/Fraser/warm review and a new-default three-pass gate remain separate. That S epoch accepted0..2; shipped main accepts3 under its strict gate; no axis enables3 (RND_N.md5Nov13:02,13:29; RND_Q.md12:41). N xd18/frozen35 completes255responses(40coldmixed,74Fraser,141warm) plusboth-width references with observed outputs/counts/request identities equal. Order2->3 warmC4/C8codeTTFT52.731/127.670->42.439/54.534ms, aggregate427.096/516.019->471.466/610.583tok/s. Matchedorder0C4/C8TTFT41.916/54.317ms; code3stays+0.523/+0.217ms aboveit. Strict coldgap468.254->470.319ms andFraserC4/C8median4104.95/7628.50->4110.40/7632.40ms FAIL; no tolerance/default pass. This is35RATE2/canonical-menu data, not pure37/defaultthree-pass qualification. That historical S source did not parse3 or selectit (RND_N.md5Nov14:55; xd18saved-dataanalysis).  N four-commit integration487000f/6c2aa48/3d1c9ec/a110d06 now accepts3 within that gate; no axis selects3. Pure37 xd20 warm C4/C8 code throughput improves7.856%/16.844% versus2; C2/cold gates fail, so automatic2 remains. Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_order_wait"></a>

#### `--prefill_order_wait`

**Type/range:** unsigned32 decimal scalar; parser 0..1024; semantic values: 0..1024. **Default:** 16. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Bound shortest-order waiting in serving scheduler ticks.

Native help describes the selected path as: `--prefill_order_wait=0..1024 (default 16; with 2: after N steps of shorter prompts in a row, a step of the oldest; 0: none)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_decode_every"></a>

#### `--prefill_decode_every`

**Type/range:** unsigned32 decimal scalar; parser 0..64; semantic values: 0..64. **Default:** 0. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Insert one decode round after this many head chunks; zero disables it.

**Use and interactions.** Nonzero requires prefill_order=1 or2; order0 refuses at engine creation. Old implementation has whole-chunk gaps; no bounded-pause guarantee. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** gsk13 order2/every1:short post-first11.45 ->9.25s,longTTFT13.93 ->14.83s; FraserC4/C8 TTFT3745/6763 ->3814/6994ms;20/20 text agreement; remains off (RND_K.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_decode_waiters"></a>

#### `--prefill_decode_waiters`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0 or1. **Default:** 0 on every no-axis route. **Commands:** serve. **Placement:** all placements with exact one-flight prefill; automatic dynamic policy only qualified BATCH1024.

**Mechanics.** Keep whole chunks/one round while multiple prompts wait.

**Use and interactions.** M7d86c45 is integrated; S uses the final M default where its qualified BATCH1024 admission matches; the all-balanced development tuple explicitly requests1024/1/waiters0/order2. Other explicit policies remain selectable. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_fixed`: `0` (serve); `batch_adaptive`: `0` (serve); `batch_session`: `0` (serve); `batch_compromise`: `0` (serve).

**Measured effect.** Gxp31 waiter512/4:0.470s max content gap, worst TTFT ratio1.07907 including all three arrivals; C8 E2E368.24tok/s. Fixed1024/1/waiters0 selected with ratio1.07868 and pilot C8 E2E373.41. These are joint policies, not an isolated waiter gain (packet RND_M.md19:14). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_decode_rounds"></a>

#### `--prefill_decode_rounds`

**Type/range:** unsigned32 decimal scalar; parser 1..64; semantic values: 1..64. **Default:** 1 (including qualified BATCH), unless explicitly overridden. **Commands:** serve. **Placement:** all placements with exact one-flight prefill; automatic dynamic policy only qualified BATCH1024.

**Mechanics.** Run this many ordinary decode rounds between capped chunks.

Native help describes the selected path as: `--prefill_decode_rounds=1..64 (ordinary decode rounds per interval; see --help-scheduling for defaults)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Requires M burst/fairness fix; bound not a TTFT guarantee. Explicit cap0 restores rounds1/waiters0/order0 unless overridden. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_fixed`: `1` (serve); `batch_adaptive`: `1` (serve); `batch_session`: `1` (serve); `batch_compromise`: `1` (serve).

**Measured effect.** Gxp33 fixed1024/1:0.477757s maximum gap, together short/long TTFT3.92157/15.38475s, short post-first completion7.76530s. Gxp26 actual default:0.476522s gap; refs/mixed/decode identity pass. Joint-policy scope (packet RND_M.md19:14). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_decode_rows"></a>

#### `--prefill_decode_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..4096; semantic values: parser0..4096; nonzero multiple256 and <=wide_rows; measured scope<=2048. **Default:** Serve qualified BATCH1024:1024; THIN/generate/other:0. Explicit0 opts out. **Commands:** serve. **Placement:** all placements with exact one-flight prefill; automatic dynamic policy only qualified BATCH1024.

**Mechanics.** Cap prefill chunks while decoders are active.

**Use and interactions.** Automatic selection requires EXL3 BF16 BATCH, wide_rows1024, TC MoE/trunk MMA, prefill_first1, column owners, batch_thin_bits and exact one-flight plan. Positive cap enables fair decode cursor even when nonbinding. Nonzero cap refused with exact prefix reuse. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: workload mix dependent. Exactness: cold-plan/row-form invariance required for a same-bits claim. Concurrency: single or multiple as admitted. Focus: decode, TTFT and prefill tradeoff. Shipped resolver writes: `batch_fixed`: `1024` (serve); `batch_adaptive`: `1024` (serve); `batch_session`: `1024` (serve); `batch_compromise`: `1024` (serve).

**Measured effect.** M7d86c45/gxp26 actual default1024/1/waiters0/order2:0.476522s gap; Fraser C1/C4/C8 TTFT1569.6/4106.4/7631.6ms; aggregate E2E205.09/317.50/373.37tok/s versus old order2 C4/C8 320.79/384.30. Both reference widths,15/15mixed and141/141decode pairs pass; THIN/generate unchanged. Joint policy, not an isolated cap gain (packet RND_M.md19:14). Source: `source/main.c` and shipped knob catalog.



<a id="knob-prefill_prompt_chunks"></a>

#### `--prefill_prompt_chunks`

**Type/range:** unsigned32 scalar, 0 or 1. **Default:** 0 on PP5, TP4_PRO and TP4_EXL3. **Command:** serve only. The real parser refuses generate and values above 1.

**Mechanics.** With value 1, a chunk-ready prompt owner keeps its prepared FULL/WIDE program and cannot donate a foreign row to ordinary decode BATCH. The scheduler retains the selected prompt or decoder turn when a full pipeline or checkpoint barrier forces a landing; successful submissions advance ownership. Decoders continue to progress during prefill, and short tails with no prepared chunk can still pack. Value 0 preserves the previous mixed-row behavior and its identity. A1 has a conditional version-2 snapshot salt; the P/Q namespace grammar is unchanged.

**Use and interactions.** The measured output-stability recipe is `--prefill_prompt_chunks=1 --prefill_first=0 --prefill_order=0` on TP4_EXL3 BF16 BATCH. It removes the observed transition where prompt position 128 enters BATCH48 and the next FULL128 chunk starts at 129, selecting different reduction forms. The guard is a scheduling change with no model/kernel/arithmetic change; it is not the rejected `--tp_batch_invariant` arithmetic prototype. Other placements/shapes receive no new identity or cost qualification from these measurements.

**Measured identity and cost.** Final A passes **96/96** mixed concurrent-vs-solo comparisons of full text, reasoning, native completion count and finish reason, with prefix reuse disabled and cached_tokens=0; all **16/16** solo signatures match tcb80, donor identities pass **423/423**, and both canonical 2,048/1,024-row selected-record hashes are retained. In the matched same-binary off→on C8 comparison, code aggregate throughput is **650.632→639.194 tok/s (−1.76%)**, while code stream decode is **84.730→85.488 (+0.90%)**. Prose aggregate is **494.805→509.798 (+3.03%, approximately +3.0%)**. The code aggregate cost keeps the global default at 0; no C1 speedup is claimed from the variable off-arm C1 result.

Frozen cold HTTP bodies, three repetitions and zero cached tokens give actual lengths 2,054/32,770: wall time **1.174852/8.982810→1.180643/9.019600 s**, approximately **+0.49%/+0.41%** (about **+0.5%**). These are prompt-plus-first-token costs, not isolated kernel times. No full-vocabulary zero-delta measurement or general quality gain is inferred from the selected-record/output identities. Source: Track N `FINAL_7.md` and `MERGE_REQUEST_A_v7.md`, final release 60 / head 8aebac2.

**When to choose it versus prefill_first=1.** Choose this explicit guard when retaining the measured first0/order0 decode behavior while needing the stronger mixed-request output stability. First1 is the separate broader prompt-priority choice and also passes 96/96 in that campaign; it has a different measured throughput/latency tradeoff. Keep first1/order2 as the separate long-prefill latency option. A's progress test passes 20/20 identities with an already-emitting stream advancing during the same unfinished 29,775-token prompt; its largest content gap is 1.577896 s. This proves progress, not parity with the unmatched historical order2 0.466 s result. A avoids the measured foreign-row odd start; it does not globally repair the separately open cached odd-start FULL compressor carry-slot hazard.

**Precision class and axes.** Scheduling/output-stability policy within the recorded arithmetic family; A0's identity is unchanged. The proposed measured mapping is category code/prose/structured, exactness=balanced, concurrency=multiple, focus=decode/balanced; code's fastest aggregate setting remains 0, while 1 is the prose/output-stability candidate. For **exactness above balanced**, prefer this scheduling rule as a dependency only after qualification with the original arithmetic; reject the B arithmetic rewrite. The EXL3 96-request gate does not qualify the original exact profile. Category-balanced/compromise, single, vision, fast and original PP5 obtain no new admission, and **no active axis registry automatically enables this flag**.

**Example.** The measured flags and power settings remain those of the Track N receipts; this parser-checked deployment command is not a fresh reproduction of the rate table.

```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=1 --step_graphs=1 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_learned_acceptance=1 --draft_base_us=13000 --draft_tail_us=0 --draft_row_us=1200 --tp_split_least_rows=512 --card_memory_kept_free_mib=0 --batch_rows=8,8,12,12,16,24,32,48 --batch_verifiers=1,2,2,4,4,4,8,8 --batch_wide_head=1 --batch_draft_rows=1 --draft_policy=0 --batch_core_once=3 --draft_session_acceptance=0 --batch_plan_ties=0 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=8 --prefix_reuse=0 --prefill_decode_rows=1024 --prefill_decode_rounds=1 --prefill_decode_waiters=0 --prefill_first=0 --prefill_order=0 --prefill_prompt_chunks=1 tokenizer.bin model.exl3map
```

Check: `recipe_prompt_chunks` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

### 8.5. Draft and speculative policy

Speculation proposes tokens and verifies them with the target. Allocated DSpark block depth, proposal cap, confidence policy and economic prices are separate controls. Reducing verified proposals does not necessarily reduce device draft work. In tcb78, omitted policy resolves to fixed0 for true EXL3 single-slot THIN and rate2 for EXL3 BATCH or configured multiple slots; EXL3 CLI priors are14300/0/690us, learn1, session0, core3. Other placements retain cost1, learn0,12700/2200/2700us,core2. Only a complete measured EXL3 BATCH geometry with omitted policy and omitted costs receives the automatic eight-price menu. Explicit policy2 retains the old no-automatic-menu behavior; explicit batch_step_us0 disables planning prices. A draft change is same-target-bits only with verification-width/form invariance.

<a id="knob-tp_draft_shards"></a>

#### `--tp_draft_shards`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Shard draft head, Markov projection and sampler with the target head.

**Use and interactions.** Requires tp_head_shards=1. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_draft_layers"></a>

#### `--tp_draft_layers`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Tensor-parallelize the draft model's three layers.

**Use and interactions.** Requires TP4_EXL3, BF16 wire, draft shards and replicated_embedding; BATCH additionally draft_rows and wide_head. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** Measured S06/d71cd05 trueTHIN at65536, changing only layers0->1 with wide2048/pipeline2: code230.959->258.683(+12.0%), prose158.954->163.601(+2.9%), structured330.108->335.422(+1.6%)tok/s; selected8/256 and15MiaAI signatures pass. Cold3909->3904tok/s; Fraser median1379.40->1385.25ms(+5.85), worst1395.10->1390.00ms(-5.10). Strict zero-margin median gate fails; no balanced promotion or1M extrapolation. FreshTHIN mixed112self/96cross signatures pass; native decode during incomplete prefill is not observed. S10 same-source pair scheduled. Earlier probe07 5-10%C4/C8 andengab21 20.54->20.46ms remain different joint epochs (RND_D.md/GAP_ANALYSIS.md; S evidence/gxf_s06_single_tp_draft_decode_summary_v1.json). S xf18 matched frozenS10/e4c188e trueTHIN2048/pipeline2 at65K: layers0->1 code230.347->258.344tok/s(+12.15%),prose158.591->163.744,structured332.793->338.442; Fraser median/worst1381.35/1390.70->1384.35/1393.90ms, strict no-worse FAIL. Full selected8/256 and15MiaAI signatures per arm pass; minfree3181->823MiB. Only code/balanced/single/decode selects1; balanced focus keeps0. No layer1 1M fit or full-logit/all-width identity inference (evidence/gxf_s10_single_layers0_strict_xf18_v1.json; layers1 counterpart). Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_draft_rows"></a>

#### `--batch_draft_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Batch verifier draft rows into the BATCH program.

**Use and interactions.** Nonzero requires batch_rows; with draft_shards it requires draft_layers; device draft layers also need wide_head. Admission: G0; B; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `batch_compromise`: `1` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_wide_head"></a>

#### `--batch_wide_head`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Use the wide target head for BATCH verification.

**Use and interactions.** With tp_head_mma=1 and batch_thin_bits=0, wide and per-slab heads use different split geometry. Nonzero requires BATCH; no isolated speed or quality claim (model_v41_program.c). Admission: G0; T; B.

**Precision class:** order-changing; same bits with batch_thin_bits=1 in the retained recipe. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `batch_compromise`: `1` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_draft_slots_head"></a>

#### `--batch_draft_slots_head`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Run one verifier's draft head across eight slots at each position.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; B; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_bf16_rows"></a>

#### `--draft_bf16_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Use BF16 rows for draft attention products.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** gsj10/12 versus07/08: prose C1 +2.6%, C8 -1.7%; code C1 -0.65%, C2 -2.5%; draft-only, 126/126 texts (RND_J.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_draft_wide"></a>

#### `--batch_draft_wide`

**Type/range:** unsigned32 decimal scalar; parser 0..3; semantic values: 0..3. **Default:** 3. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Widen draft work: one sums/posts, two products, three cores.

Native help describes the selected path as: `--batch_draft_wide=0..3 (default 3; a batched draft's sums and posts, 2: and products, over all its rows at once, same bits; 3: and cores, whole blocks, drafts only)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; B; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** gsl05l3 level3 codeC8 641.6tok/s, proseC2/C4/C8 280.9/437.6/496.4; fixed-target texts retained; level3 became default after rl08 refs/window (RND_L.md). Joint epoch; isolated percentage not inferred. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_bf16_operands"></a>

#### `--draft_bf16_operands`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Use BF16 operands for class-80 draft quantized products.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** gsl03e serving proseC1 +0.1%, no BATCH win; THIN generate +5.9% in scoped probe, draft-only (RND_L.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_learned_acceptance"></a>

#### `--draft_learned_acceptance`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** PP5/TP4_PRO and other original placements: 0; TP4_EXL3 THIN: 1. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Update the draft policy's shared confidence table from outcomes.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `1` (all/generate); `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate); `thin_full_layers`: `0` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_session_acceptance"></a>

#### `--draft_session_acceptance`

**Type/range:** unsigned32 decimal scalar; parser 0..11; semantic values: 0 shared table;1 session ratio;2 log-odds shift;3 Platt scale+shift;4 shift+hindsight;5 full depth while measured match>=0.82;6 position-local EMA;7 middle scout;8 second-position scout;9 natural selector;10 conditional residual calibration;11 actual successful BATCH shape costs. **Default:** 0. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Scale proposal acceptance by the current session's measured outcomes.

Native help describes the selected path as: `--draft_session_acceptance=0..11 (default 0)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Current S/main242e6ef accepts 0..11, default0. Modes6..11 are implemented:6 position-local EMA,7 middle scout,8 second-position scout,9 natural selector,10 conditional residual calibration,11 actual successful BATCH shape costs. Modes0..10 adjust draft-depth acceptance/calibration; mode11 additionally uses worker-private measured prices. They do not directly replace a target operator, but changed proposal depth can select different target widths/forms: draft-only classification requires target-bit preservation (gate D). Modes6..11 have no active S axis mapping; existing tuples retain0 or5. Inherited N text/ID checks alone do not establish selected-logit invariance or a universal tuple. Mode11 has no new integration recommendation under the later N too-slow/no-merge review; parser availability is distinct from promotion (S main.c:857; engine.c:60-80; RND_N.md 03:34 and RND_S/N/R.md 5 Nov04:38). Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `0` (all/generate); `batch_adaptive`: `0` (all/generate); `batch_session`: `5` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `0` (all/generate); `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate); `thin_full_layers`: `0` (all/generate).

**Measured effect.** gxd01b session5+ties: code C8 724.8 versus661.5 (+9.6%); prose572.8 versus492.9 (+16.2%); code C1 240.9 versus244.8 (-1.6%); joint recipe (RND_N.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-batch_plan_ties"></a>

#### `--batch_plan_ties`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** TP4_EXL3 BATCH.

**Mechanics.** Cut planned drafts only between unequal predicted reach.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; B; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `0` (all/generate); `batch_adaptive`: `0` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `0` (all/generate); `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate); `thin_full_layers`: `0` (all/generate).

**Measured effect.** gxd01b joint session5+ties results: code/prose C8 +9.6/+16.2%; isolated gain unmeasured (RND_N.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_start_economic"></a>

#### `--draft_start_economic`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Start DSpark at policy depth instead of a single-proposal probe.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-speculative"></a>

#### `--speculative`

**Type/range:** unsigned32 decimal scalar; parser 0..3; semantic values: 0 off;1 lookup;2 DSpark;3 hybrid. **Default:** PP5/TP4_PRO and other original placements: 0; TP4_EXL3 THIN: 2. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Choose off, n-gram lookup, checkpoint DSpark or hybrid proposals.

**Use and interactions.** Nonzero requires rows>1. Every TP startup requires DSpark or hybrid (2/3); off/lookup refuse there. Device/hybrid depth cannot exceed draft_block; multi-slot THIN requires pipeline_steps>1. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `2` (all/generate); `batch_adaptive`: `2` (all/generate); `batch_session`: `2` (all/generate); `thin_fixed`: `2` (all/generate); `batch_compromise`: `2` (all/generate); `original_tp_serial2`: `2` (all/generate); `original_pp_depth8`: `2` (all/generate); `original_pp_depth5`: `2` (all/generate); `thin_full_layers`: `2` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_policy"></a>

#### `--draft_policy`

**Type/range:** unsigned32 decimal scalar; parser 0..3; semantic values: 0 fixed;1 cost;2 rate;3 adaptive. **Default:** Original placements1 (cost); EXL3 true single-slot THIN0 (fixed), EXL3 BATCH or configured multiple slots2 (rate). Explicit values bypass the omitted-policy rule. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Choose fixed, cost, rate or adaptive proposal-depth selection.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `0` (all/generate); `batch_adaptive`: `2` (all/generate); `batch_session`: `2` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `2` (all/generate); `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate); `thin_full_layers`: `0` (all/generate).

**Measured effect.** Controlled S06/d71cd05 binary76c98fc7 evidence: canonical head_shards1/head_stream1/draft_shards1/draft_layers0 with policy0 passes full8/256 selected-record references and15 C1 MiaAI native/text/finish checks in gxf_s06_code_single_short. Fixed0 C1 code230.959/prose158.954/structured330.108 tok/s; cold prefill3909 tok/s with cached0; Fraser C1 median/worst1379.4/1395.1 ms. Changing only policy0 to1, with the other concrete flags equal, fails the30K selected-record gate: all8 IDs agree but6 selected F32 logits differ, max about7.63e-5. Policy1 MD5ff1489cffe4194bb52fe36eef08056c9 equals the older S04gxf15 policy1 file despite that older recipe being unsharded. This controlled result does not establish a head-sharding cause; target width/form or selected-logit provenance remains under investigation. No current policy1 speed is measured, and fixed0 results do not transfer to it. Those S06 raw defaults were unchanged during that experiment. Current main39980c7 resolves omitted EXL3 policies to fixed0 or RATE2; other placements retain1. Cost1 still lacks same-bits-to-fixed-canonical balanced qualification. Source: evidence/xf_axes_numerics_v3/thin_current_30k.*; RND_S/N/R.md 5 Nov04:38. Historical N gxd01c/01a code264.3/244.8 and prose162.7/168.7 tok/s retain their different joint-recipe scope, not this controlled S comparison. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_threshold"></a>

#### `--draft_threshold`

**Type/range:** unsigned32 decimal scalar; parser 0..1000; semantic values: 0..1000. **Default:** 100. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Set the minimum draft confidence in thousandths.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_base_us"></a>

#### `--draft_base_us`

**Type/range:** unsigned32 decimal scalar; parser 1..1000000000; semantic values: 1..1000000000. **Default:** PP5/TP4_PRO and other original placements: 12700; TP4_EXL3 THIN: 14300. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Estimate the base verification-step cost in microseconds.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `13000` (all/generate); `batch_adaptive`: `13000` (all/generate); `batch_session`: `13000` (all/generate); `thin_fixed`: `12700` (all/generate); `batch_compromise`: `13000` (all/generate); `original_tp_serial2`: `12700` (all/generate); `original_pp_depth8`: `12700` (all/generate); `original_pp_depth5`: `12700` (all/generate); `thin_full_layers`: `12700` (all/generate).

**Measured effect.** Measured S06 xf11/gxf_s06_rate_control_pair versus gxf_s06_rate_priors_pair: RATE2, learn1/session0/ties0/core2 and identical MiaAI→M-arrivals→Fraser. Joint base/row prices 13000/1200 to 14300/690 microseconds, tail fixed0. Code C1/C2/C4/C8 changes +6.001/+3.526/+7.396/+0.446%; prose C2/C8 changes -6.168/-2.463%. Full selected8/256 references and141 cross-arm request/text/native-count/finish signatures agree; sampled native decode advances during long prefill in both warm repetitions. Fraser median deltas C1/C4/C8 +0.65/+4.50/+8.55 ms; worst deltas -5.70/+28.30/+7.00 ms. Warm-short maximum SSE gap 0.473578 to0.469387 s; no native-token timing claim. The historical zero-margin Fraser screen fails. Main later accepted the N31 raw-default bucket under its noise interpretation; that selection does not rewrite this failure or by itself qualify a bounded S tuple. CPU builds/tests overlapped parts of these windows; no isolated causal or universal optimum claim. No cold/mixed/1M/full-logit/official qualification in this pair. Source: evidence/s06_rate_pair_review_v2.json. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_tail_us"></a>

#### `--draft_tail_us`

**Type/range:** unsigned32 decimal scalar; parser 0..1000000000; semantic values: 0..1000000000. **Default:** PP5/TP4_PRO and other original placements: 2200; TP4_EXL3 THIN: 0. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Estimate fixed tail cost in microseconds.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `0` (all/generate); `batch_adaptive`: `0` (all/generate); `batch_session`: `0` (all/generate); `thin_fixed`: `2200` (all/generate); `batch_compromise`: `0` (all/generate); `original_tp_serial2`: `2200` (all/generate); `original_pp_depth8`: `2200` (all/generate); `original_pp_depth5`: `2200` (all/generate); `thin_full_layers`: `2200` (all/generate).

**Measured effect.** Tail0 is held fixed in the scoped S06 xf11 joint base/row-price pair; no isolated tail effect was measured. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_row_us"></a>

#### `--draft_row_us`

**Type/range:** unsigned32 decimal scalar; parser 1..1000000000; semantic values: 1..1000000000. **Default:** PP5/TP4_PRO and other original placements: 2700; TP4_EXL3 THIN: 690. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Estimate each additional verification row's cost in microseconds.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `1200` (all/generate); `batch_adaptive`: `1200` (all/generate); `batch_session`: `1200` (all/generate); `thin_fixed`: `2700` (all/generate); `batch_compromise`: `1200` (all/generate); `original_tp_serial2`: `2700` (all/generate); `original_pp_depth8`: `2700` (all/generate); `original_pp_depth5`: `2700` (all/generate); `thin_full_layers`: `2700` (all/generate).

**Measured effect.** Measured S06 xf11/gxf_s06_rate_control_pair versus gxf_s06_rate_priors_pair: RATE2, learn1/session0/ties0/core2 and identical MiaAI→M-arrivals→Fraser. Joint base/row prices 13000/1200 to 14300/690 microseconds, tail fixed0. Code C1/C2/C4/C8 changes +6.001/+3.526/+7.396/+0.446%; prose C2/C8 changes -6.168/-2.463%. Full selected8/256 references and141 cross-arm request/text/native-count/finish signatures agree; sampled native decode advances during long prefill in both warm repetitions. Fraser median deltas C1/C4/C8 +0.65/+4.50/+8.55 ms; worst deltas -5.70/+28.30/+7.00 ms. Warm-short maximum SSE gap 0.473578 to0.469387 s; no native-token timing claim. The historical zero-margin Fraser screen fails. Main later accepted the N31 raw-default bucket under its noise interpretation; that selection does not rewrite this failure or by itself qualify a bounded S tuple. CPU builds/tests overlapped parts of these windows; no isolated causal or universal optimum claim. No cold/mixed/1M/full-logit/official qualification in this pair. Source: evidence/s06_rate_pair_review_v2.json. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_probe_rounds"></a>

#### `--draft_probe_rounds`

**Type/range:** unsigned32 decimal scalar; parser 1..1048576; semantic values: 1..1048576. **Default:** 8. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Probe full draft depth after this many policy rounds.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-hybrid_lookup_prior"></a>

#### `--hybrid_lookup_prior`

**Type/range:** unsigned32 decimal scalar; parser 0..1000; semantic values: 0..1000. **Default:** 500. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Set lookup acceptance prior in thousandths.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-hybrid_lookup_min"></a>

#### `--hybrid_lookup_min`

**Type/range:** unsigned32 decimal scalar; parser 1..16; semantic values: 1..16. **Default:** 3. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Use lookup in hybrid mode only from this n-gram length.

**Use and interactions.** Hybrid requires <=lookup_ngram_max. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_block"></a>

#### `--draft_block`

**Type/range:** unsigned32 decimal scalar; parser 5..7; semantic values: 5..7. **Default:** 5. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Set the device draft block capacity.

**Use and interactions.** Device/hybrid; TP allocates a block of5..7. Explicit draft_positions above the block refuses; omitted depth clamps to the block. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `5` (all/generate); `batch_adaptive`: `5` (all/generate); `batch_session`: `5` (all/generate); `thin_fixed`: `5` (all/generate); `batch_compromise`: `5` (all/generate); `original_tp_serial2`: `5` (all/generate); `original_pp_depth8`: `5` (all/generate); `original_pp_depth5`: `5` (all/generate); `thin_full_layers`: `5` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_positions"></a>

#### `--draft_positions`

**Type/range:** unsigned32 decimal scalar; parser 1..7; semantic values: 1..7. **Default:** Raw cap7; DSpark/hybrid with default block5 clips an omitted cap to5. An explicit cap above the allocated block refuses. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Bound proposals verified per round.

**Use and interactions.** Also set by speculative_depth. Device/hybrid explicit depth must not exceed draft_block; actual depth is min(value,rows-1). TP allocates draft_block5..7 independently, so proposal caps1..4 remain admitted. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. Shipped resolver writes: `batch_fixed`: `5` (all/generate); `batch_adaptive`: `5` (all/generate); `batch_session`: `5` (all/generate); `thin_fixed`: `5` (all/generate); `batch_compromise`: `5` (all/generate); `original_tp_serial2`: `5` (all/generate); `original_pp_depth8`: `5` (all/generate); `original_pp_depth5`: `5` (all/generate); `thin_full_layers`: `5` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-speculative_depth"></a>

#### `--speculative_depth`

**Type/range:** unsigned32 decimal scalar; parser 1..7; semantic values: 1..7. **Default:** Same destination as draft_positions: raw7, default DSpark clip5. Each spelling is unique, but supplying both distinct aliases is accepted on main and the later value wins. Prefer one spelling. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Alias of draft_positions.

**Use and interactions.** Same destination as draft_positions; axis expansion treats aliases as one override key. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


Main has a checked alias edge case: both spellings together are accepted, and this example resolves depth5. Reversing their order resolves4; repeating the same spelling is refused. S alias expansion treats them as one explicit override key, which is a separate behavior.


```sh
anemone serve --placement=tp4_exl3 --speculative=2 --rows=8 --draft_positions=4 --speculative_depth=5 tokenizer.bin model.exl3map
```

Check: `alias_depth_last5` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

<a id="knob-lookup_ngram_max"></a>

#### `--lookup_ngram_max`

**Type/range:** unsigned32 decimal scalar; parser 1..16; semantic values: 1..16. **Default:** 4. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Set the longest lookup suffix searched.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; D.

Use this in a controlled proposal/verification experiment; retain a fixed-policy baseline and compare selected target logits as well as acceptance and rate.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-lookup_ngram_min"></a>

#### `--lookup_ngram_min`

**Type/range:** unsigned32 decimal scalar; parser 1..16; semantic values: 1..16. **Default:** 2. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Set the shortest lookup suffix searched.

**Use and interactions.** Must not exceed lookup_ngram_max. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_calibration"></a>

#### `--draft_calibration`

**Type/range:** text/enum; path to <4096-byte file; exactly112 integers0..1000. **Default:** none **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Override draft acceptance priors in mode/position/bin order.

**Use and interactions.** Exactly112 whitespace-separated unsigned integers0..1000, optional trailing whitespace, no extra tokens, at most4095 bytes. Overrides acceptance priors only; cost knobs are independent (main.c draft_calibration_read). Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Draft policy/calibration target-bit qualification follows gate D and the controlled S06 policy finding. Source: `source/main.c` and shipped knob catalog.


The example's `usage_evidence/calibration.txt` contains exactly112 whitespace-separated500 values. This is a public synthetic prior fixture, not a credential. The file must be shorter than4096bytes, values0..1000, with no extra token. The stale80-value source comment does not override the 2×7×8 loop.

<a id="knob-draft"></a>

#### `--draft`

**Type/range:** text/enum; lookup / dspark / hybrid. **Default:** off **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Select speculative proposal source by name.

**Use and interactions.** Draft aliases must agree with explicit speculative; layout only PP5; no secrets printed. Admission: G0; D.

**Precision class:** draft-only with target-bit invariance across verification widths/forms; conditional order-changing otherwise. **Four-axis mapping:** Category: workload/acceptance dependent. Exactness: draft-only only with target-width/form invariance; otherwise order-changing. Concurrency: THIN/single and/or BATCH/multiple as admitted. Focus: decode economics; scheduling can affect TTFT/prefill. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Draft policy/calibration target-bit qualification follows gate D and the controlled S06 policy finding. Source: `source/main.c` and shipped knob catalog.


Lookup-only uses the host n-gram proposer; DSpark uses the device draft; hybrid can use both. Consistent draft/speculative spellings can coexist only if they select the same mode.


```sh
anemone serve --placement=pp5 --rows=8 --draft=lookup tokenizer.bin original.anemone
```

Check: `draft_lookup` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


```sh
anemone serve --placement=tp4_exl3 --rows=8 --draft=hybrid --context_tokens=65536 tokenizer.bin model.exl3map
```

Check: `draft_hybrid` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

### 8.6. Attention, index and Engram

Attention state, compressor pools and sparse index selection have ownership and live-range constraints across slabs and chunks. A same-bits optimization can gather only the necessary rows or write a compact half without changing the attention reduction. Experimental owner/state choices have stricter placement gates. Engram host staging remains a host-memory path; sharing identical masks reduces duplicate staging, not the logical contents of the tables.

<a id="knob-tp_attention_owner"></a>

#### `--tp_attention_owner`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** exact original TP4_PRO, FP32 wire.

**Mechanics.** Keep experimental TP4 attention state on its owning rank.

**Use and interactions.** Nonzero requires original TP4_PRO, numerics=exact and FP32 wire; other placements refuse it at startup. Admission: G0; T.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_exact_dense"></a>

#### `--tp_exact_dense`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use exact output-sharded attention dense products on the FP32 EXL3 recipe.

**Use and interactions.** Active only TP4_EXL3 with FP32 wire; ignored on BF16/other placements. Admission: G0; T.

**Precision class:** same bits (FP32 exact recipe). **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** xdense01: same-bits exact dense shards; isolated whole-workload speed unmeasured (RND_D.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_gather_balance"></a>

#### `--tp_gather_balance`

**Type/range:** unsigned32 decimal scalar; parser 0..3; semantic values: 0 off;1 all chunk gathers;2 wide only;3 weighted by rank width. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Assign chunk index/router work to narrower expert ranks (four policies).

Native help describes the selected path as: `--tp_gather_balance=0..3 (default 0; tp4_exl3: a chunk's index scans and router rows by the 512-column expert ranks alone, the same bits; 2: wide steps' alone; 3: the 640-column ranks 3/16 each, the 512-column ones the rest)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to test the named state/index work reduction; check live positions, row geometry and canonical state equality.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rc04 mode1 Fraser1.857 ->1.914s (+3.1%, slower), IDENTICAL; ri05 mode3 30K8.488 ->8.598s (+110ms), identical (RND_C.md/RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_pool_chunk"></a>

#### `--tp_pool_chunk`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Prepare pooled entries/keys before chunk-slab scans.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to test the named state/index work reduction; check live positions, row geometry and canonical state equality.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rc02ab 30K prefill 9.828 -> 9.430 s; same bits (RND_C.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_pool_gather"></a>

#### `--tp_pool_gather`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Partition pooled-source compressor rows over ranks and gather results.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to test the named state/index work reduction; check live positions, row geometry and canonical state equality.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri01 same bits; standalone speed unmeasured (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_router_first_whole"></a>

#### `--tp_router_first_whole`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Compute first-half router rows on every rank without gathering.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to test the named state/index work reduction; check live positions, row geometry and canonical state equality.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri01 with joint/dead/pool:30K +54ms, Fraser neutral; identical, kept off (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_queries_piece"></a>

#### `--tp_queries_piece`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Project wide-program queries by piece alongside wq_b.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to test the named state/index work reduction; check live positions, row geometry and canonical state equality.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri05 30K prompt -77 ms; same bits (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_index_span"></a>

#### `--tp_index_span`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Scan/select a span of chunk index rows per launch.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to test the named state/index work reduction; check live positions, row geometry and canonical state equality.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri05 30K prompt -51 ms; same bits (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_core_compact_halves"></a>

#### `--tp_core_compact_halves`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Write split-half core results into compact wo_a rows.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to test the named state/index work reduction; check live positions, row geometry and canonical state equality.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri06 -9/-11 ms on prompt controls; same bits (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_chunk_core"></a>

#### `--tp_chunk_core`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Run a chunk's attention core over the whole chunk.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to test the named state/index work reduction; check live positions, row geometry and canonical state equality.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** pf16 30K prefill18.93 ->17.80 s; same bits (GAP_ANALYSIS.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_core_heads"></a>

#### `--tp_core_heads`

**Type/range:** unsigned32 decimal scalar; parser 0..64; semantic values: 0,8,..64. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Limit attention-core heads per block; zero uses the measured rule.

Native help describes the selected path as: `--tp_core_heads=0|8|16 (default 0; the attention core's blocks of at most so many heads, the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Must be a multiple of8; zero chooses per-shape policy. Admission: G0; T.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rj03p THIN C1 20.23 -> 20.22 ms at8 (-0.05%, noise), same bits (RND_J.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-engram_shared"></a>

#### `--engram_shared`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0 or1. **Default:** 0. **Commands:** generate, serve. **Placement:** all TP placements where shared gathers are admitted.

**Mechanics.** Gather identical TP Engram inputs once into a shared pinned block.

**Use and interactions.** TP-only, immutable per-step input; no CPU expert offload. Admission: G0; T.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** xp04k2024-token fixture: baseline gather0/dense16 TTFT1.121s; gather1/dense16 1.111s; gather1/dense8/32/64 1.109/1.106/1.108s. Selected records identical, natural EOS after2tokens; single pass, not cold full benchmark or256-token decode proof. Prior gather A/B1.115/1.109/1.106/1.116s. Default0 retained (RND_M.md19:36). Source: `source/main.c` and shipped knob catalog.


### 8.7. MoE and expert products

EXL3 weight decode, TC products, expert partition and transport are distinct choices. Wide-kernel product and entry optimizations can preserve existing words, while TC arithmetic, rotations and alternative parts can change rounding/order. The 148-block grid and occupancy choices were measured for specific shapes; a larger block count is not inherently faster. MoE choices only affect the named EXL3 BF16/TC path when it is enabled; an inert flag does not convert a weight family.

<a id="knob-exl3_expert_parts"></a>

#### `--exl3_expert_parts`

**Type/range:** unsigned32 decimal scalar; parser 4..8; semantic values: 4 or8. **Default:** 4. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Choose four reference or eight alternate expert product parts.

Native help describes the selected path as: `--exl3_expert_parts=4|8 (default 4; tp4_exl3 uses four CMPs, exact wire by default; 8 requires relaxed/fast numerics)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Only4 or8;8 requires TP4_EXL3, numerics relaxed/fast, reference CMP order and FP32 wire. Admission: G0; T.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-exl3_fast_products"></a>

#### `--exl3_fast_products`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use wide-kernel EXL3 products instead of the retained chain.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this only in the corresponding product family; compare numerical records and whole-request performance, not only a single kernel timer.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-exl3_wide_entry"></a>

#### `--exl3_wide_entry`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Run each chunk's EXL3 experts once over the full chunk.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this only in the corresponding product family; compare numerical records and whole-request performance, not only a single kernel timer.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-exl3_tc_moe"></a>

#### `--exl3_tc_moe`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use GNmac-style tensor-core EXL3 routed MoE products.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this only in the corresponding product family; compare numerical records and whole-request performance, not only a single kernel timer.

**Precision class:** precision up (hi/lo operands); changed sums. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** tcmoe02 matches GNmac/Curtis at6/12/36/43/48/64 rows;48-row I6401490 versus1544us. engab03 joint TC-MoE+head:163.4 versus138.6 tok/s; only11/256 target tokens agree with old recipe (GAP_ANALYSIS.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-exl3_tc_product_blocks"></a>

#### `--exl3_tc_product_blocks`

**Type/range:** unsigned32 decimal scalar; parser 0..592; semantic values: 0..592. **Default:** 0. **Commands:** load, generate, serve. **Placement:** effective EXL3 TC-MoE THIN/BATCH product path; FULL/WIDE unchanged; no extra placement gate.

**Mechanics.** Override the two EXL3 TC-MoE product grids in THIN/BATCH programs.

Native help describes the selected path as: `--exl3_tc_product_blocks=0..592 (default 0: decode TC products use 148 blocks; same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Parser accepts0..592 for load/generate/serve, with no extra placement gate. Zero retains the existing148-block grid; positive values replace only gate/up and down grid[0] in eligible EXL3 TC-MoE THIN/BATCH programs. FULL/WIDE grids and each item's arithmetic/owners/folds remain unchanged; inert outside that prepared path. Same-bits geometry control. Current axes leave0; nonzero is a raw tuning override, not an automatically selected faster form (S main.c:751; model_v41_program.c:2935). Admission: G0; T.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `0` (all/generate); `batch_adaptive`: `0` (all/generate); `batch_session`: `0` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `0` (all/generate); `thin_full_layers`: `0` (all/generate).

**Measured effect.** N xd06 fixed5 BATCH8/one-verifier pilot at 150 W: grids148/74/222/296 give structured318.94/279.34/322.29/323.94, code245.93/212.46/245.48/247.53 and prose156.35/134.60/156.36/157.53 tok/s. All400/95/400 selected records agree. This is an inherited pilot, not a native180W/full-serving or fresh S measurement. Main242e6ef merges the selector with default0 (retained148-block grid); no axis selects a nonzero override (packet RND_N.md 4 Nov19:56; source model_v41_program.c:2935). Source: `source/main.c` and shipped knob catalog.


<a id="knob-exl3_expert_rotation"></a>

#### `--exl3_expert_rotation`

**Type/range:** unsigned32 decimal scalar; parser 0..2; semantic values: 0 fixed chunks;1 four-way rotation;2 three-way rotation. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Rotate per-expert TC chunks across four ranks or three non-head ranks.

Native help describes the selected path as: `--exl3_expert_rotation=0|1|2 (default 0; with --exl3_tc_moe=1: 1 rank r of layer L holds chunk (r + L + e) % 4 of expert e (Morrow's); 2 rank 0 chunk 0, ranks 1-3 three ways: other TP order-changing)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Nonzero requires TP4_EXL3, BF16 wire, TC MoE and four expert parts. Admission: G0; T.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rb02 rotation2 about1%30K gain with changed numerics; rotation1 OOM on old57.9GB/card recipe. Not a current fit guarantee (RND_B.md/RND_E.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_router_mma"></a>

#### `--tp_router_mma`

**Type/range:** unsigned32 decimal scalar; parser 0..2; semantic values: 0 dense;1 BATCH router MMA;2 also prefill routers. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use split-K tensor-core BATCH router products (two alternate modes).

Native help describes the selected path as: `--tp_router_mma=0|1|2 (default 0; 2: prefill steps' routers too; tp4_exl3 BF16 recipe: a BATCH step's router (9-64 rows) by split-K BF16 tensor-core MMAs and a fixed-order reduce, GNmac's router class; another order than the dense one)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this only in the corresponding product family; compare numerical records and whole-request performance, not only a single kernel timer.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** dense02 48-row router60.4 ->25.6us, maxscore-relative error6e-7 (other order-changing); rb02 mode2 about1% 30K gain in old joint recipe (GAP_ANALYSIS.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-exl3_tcp_two_blocks"></a>

#### `--exl3_tcp_two_blocks`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Launch prefill TC MoE products at two blocks per multiprocessor.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this only in the corresponding product family; compare numerical records and whole-request performance, not only a single kernel timer.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri03 2048-row product call -15%, Fraser first token -17 ms; same bits (RND_I.md). Source: `source/main.c` and shipped knob catalog.


### 8.8. Tensor transport, graphs and overlap

Transport chooses where partials are copied and summed; overlap chooses when independent work can run. Copy-engine exchange can preserve a named sum order, whereas changing FP32 wire to BF16 adds rounding. Graph partitions, side branches, slab/piece rings and two-half schedules must share valid program dependencies and memory ownership. An extra pipeline step can merely queue another complete flight. Only the separately gated original TP4_PRO overlap path creates distinct CMP-prefix/PRO-tail workers.

<a id="knob-tp_parallel_launch"></a>

#### `--tp_parallel_launch`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** all TP placements; PP ignores this tensor control.

**Mechanics.** Launch each TP card's graph groups from a separate host thread.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_graph_parts"></a>

#### `--tp_graph_parts`

**Type/range:** unsigned32 decimal scalar; parser 1..8; semantic values: 1..8. **Default:** 1. **Commands:** load, generate, serve. **Placement:** all TP placements; PP ignores this tensor control.

**Mechanics.** Partition each card's graph behind the prefill cone.

Native help describes the selected path as: `--tp_graph_parts=1..8 (default 1; TP: a card's step graph behind the prefill cone in this many parts, so that a step begins after a smaller launch; the same launches in the same order, the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri05 parts8:30K8.488 ->8.871s (+383ms, slower), identical (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_ordered_wide"></a>

#### `--tp_ordered_wide`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use ordered-wide attention trunk products for EXL3 prefill.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_graph_branches"></a>

#### `--tp_graph_branches`

**Type/range:** unsigned32 decimal scalar; parser 0..2; semantic values: 0..2. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Branch shared-expert work beside routed experts; mode two includes BATCH.

Native help describes the selected path as: `--tp_graph_branches=0|1|2 (default 1; tp4_exl3 with --exl3_tc_moe=1: a thin layer's shared expert a step-graph branch beside its routed experts, joined by the ring, as GNmac's auxiliary stream, the same bits; 2: BATCH steps' too; 0: one chain)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** engab18 C1 round -0.34 ms; same bits (RND_E.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_piece_rings"></a>

#### `--tp_piece_rings`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Exchange BF16 wide results over 64-row pieces instead of eight-row slabs.

**Use and interactions.** Changes ring ownership/order when tp_ring_column_owners=0; retain column_owners=1 for the same-bits recipe. BF16 TP exchange only; no isolated performance claim. Admission: G0; T.

**Precision class:** order-changing; same bits scoped to column-owner recipe. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_trunk_shape_slices"></a>

#### `--tp_trunk_shape_slices`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use the measured slice count per THIN trunk shard shape; zero forces four.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** trunk01 ordered chains maxdiff0 at1..16 slices; engab02 same tokens,255token2.032 ->1.975s (+2.9% rate). Six-row wkv25.6 ->15.4us (GAP_ANALYSIS.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_bf16_slab_rings"></a>

#### `--tp_bf16_slab_rings`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use BF16 slab-ring attention sums instead of whole-row FP32 accumulation.

**Use and interactions.** BF16 rings only; inert with FP32 wire. BATCH/thin-tail forms already retain slab-ring completion. For wider attention, enabling this changes intermediate rounding versus FP32 whole-row accumulation (model_v41_program.c:1960). Admission: G0; T.

**Precision class:** lossy (BF16 intermediate sums); order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_wide_exchange"></a>

#### `--tp_wide_exchange`

**Type/range:** unsigned32 decimal scalar; parser 0..5; semantic values: 0 legacy;1 push;2 owner sums;3 owner rings;4 CE owner sums;5 CE rings. **Default:** 5. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Choose wide transport: push, slab rings, owner sums or copy-engine rings.

Native help describes the selected path as: `--tp_wide_exchange=0|1|2|3|4|5 (default 5 with --tp_split_overlap=1: pf19 30K-token prefill 14.80 s against 19.08, identical; 4: the owner sums' bytes moved by copy engines; 5: 4 and the FFN's slab rings as copy-engine owner ring sums; both the same bits; was 2; tp4_exl3 BF16 recipe: a chunked piece's sums: 2 each rank sums its shard of rows, half the push's bytes (pf07: 30K-token prefill 22.05 -> 18.85 s, the same bits); 1 the ordered push by many CTAs; 3: 2 and the FFN's slab rings as one exchange a piece (the same bits, slower: 23.25 s); 0: one CTA)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** pf19 CE owner sums/overlap: 30K prefill19.08 ->14.80 s; joint recipe (GAP_ANALYSIS.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_split_overlap"></a>

#### `--tp_split_overlap`

**Type/range:** unsigned32 decimal scalar; parser 0..7; semantic values: 0 off;1 overlap;2 serial;3 attention;4 FFN;5 pipeline;6 joint;7 joint down. **Default:** 7. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Overlap layer halves and wide exchange with a chosen split schedule.

Native help describes the selected path as: `--tp_split_overlap=0..7 (default 7: 6 with that call's gate/up tiles over both halves and its down GEMM and combine a half each, the first half's FFN sums beside the second's down, same bits; 6: 5 with one expert call a chunk, same bits; 5: pf25 14.54 -> 12.92 s, identical; tp4_exl3 BF16 recipe: a chunked step's layer from wo_a on in two halves of its rows, each half's sums as a graph side run beside the other half's work (Morrow's PREFILL_REDUCE_OVERLAP on step graphs), the same bits; 2: the halves one behind the other; 3: only the attention's sums beside work; 4: only the FFN's; 5: 1 with the attention in halves too, each layer's last sums beside the next layer's first half, the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri01 same bits; earlier mode6 rb02 12.370 ->12.057 s (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_split_least_rows"></a>

#### `--tp_split_least_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..2048; semantic values: 0..2048. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Run chunks below this row count whole instead of split.

Native help describes the selected path as: `--tp_split_least_rows=N (default 0; fewer rows unsplit)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `512` (all/generate); `batch_adaptive`: `512` (all/generate); `batch_session`: `512` (all/generate); `batch_compromise`: `512` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_launch_trims"></a>

#### `--tp_launch_trims`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Remove redundant ring-input copies and wo_b quantization launches.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rd04-series BATCH12 launches/card2112 ->1906 across retained trim package; reference identity qualified before windows; no isolated percentage inferred (RND_D.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_full_split_overlap"></a>

#### `--tp_full_split_overlap`

**Type/range:** unsigned32 decimal scalar; parser 0..7; semantic values: 0 inherit;1..7 as tp_split_overlap. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Override full-program split overlap; zero inherits tp_split_overlap.

Native help describes the selected path as: `--tp_full_split_overlap=0..7 (default 0: --tp_split_overlap's; the full steps' mode, the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_replicated_embedding"></a>

#### `--tp_replicated_embedding`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Embed input rows on every rank instead of handing off rank zero.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_ring_column_owners"></a>

#### `--tp_ring_column_owners`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Assign BF16 ring ownership by column quarter for batch-independent addition order.

Native help describes the selected path as: `--tp_ring_column_owners=0|1 (default 1; the BF16 ring sums' owners the quarters of each row, so a row's sum does not depend on its place in the step or the step's rows: other bits than 0)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** mc_check2 refs since7f9c389: 30K41eeb9c16e10, Fraser f49e8badff47; different from old row-owner sums (RND_PIPELINE.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-step_graphs"></a>

#### `--step_graphs`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Capture/replay prepared CUDA step graphs instead of eager launches.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `original_tp_serial2`: `1` (all/generate); `original_pp_depth8`: `1` (all/generate); `original_pp_depth5`: `1` (all/generate); `thin_full_layers`: `1` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `1` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-pipeline_steps"></a>

#### `--pipeline_steps`

**Type/range:** unsigned32 decimal scalar; parser 1..8; semantic values: 1..8. **Default:** Raw default1 on all placements. Qualifying original PP5 serve with session_slots>1, no BATCH and DSpark/speculative2 resolves omitted depth to5. Explicit depth is retained; the shipped original multiple-user decode axis explicitly selects8. Generate and single-slot raw defaults stay1. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Bound ordinary step pipeline depth.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to isolate the named launch/transport dependency; compare complete target/state receipts before calling the overlap same bits.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `2` (all/generate); `batch_compromise`: `1` (all/generate); `original_tp_serial2`: `2` (all/generate); `original_pp_depth8`: `8`, `2` (generate); `original_pp_depth5`: `5`, `2` (generate); `thin_full_layers`: `2` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `1` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Original PP5 source75 repaired pair: depth2->5 cold prefill3502->4450tok/s; Fraser C1/C4/C8 medianTTFT1475.55/6739.35/12005.55->1278.75/4308.65/7165.80ms, aggregateE2E159.306/272.776/281.481->161.620/446.244/465.998tok/s. Independent literal source/config/power/cold/lifetime and46Fraser+9fixed API identities pass. NativeV1 check_failed/exit1 are preserved; repair-aware controller/root derived proof passes. Not the separate three-genre decode sweep,1M,official equality or S10 serving result. The f52c9c5 depth5 default is now merged for the qualifying raw serve route; current parser receipts verify it. The shipped original decode axes still explicitly choose PP8. No original TTFT/prefill tuple is admitted by this note (RND_O.md5Nov11:21). Integration batch applies f52c9c5; explicit depths retain their values. Source: `source/main.c` and shipped knob catalog.


### 8.9. Precision, exactness, head and dense forms

The exact/raw numerical selector is separate from S's exactness axis. Changing product forms, head dispatch or row splits can change target bits even when the storage type is unchanged. The reference local CMP order is required by TP admission. Some selectors restore THIN-style split order across larger BATCH forms, but those claims retain their stated shape and source boundary. Profile/checking controls are constrained by compile-time hooks; the evaluated release range below takes precedence over an old help catalog.

<a id="knob-tp_head_shards"></a>

#### `--tp_head_shards`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Shard target head products and sampling over TP4_EXL3 ranks.

**Use and interactions.** Nonzero requires TP4_EXL3; enables tp_draft_shards. Head-MMA activation depends on the combined head strategy. Canonical THIN uses head_shards1/head_stream1/draft_shards1 and draft_layers0. S06 holds these head flags equal and changes only draft_policy0 to1: all8 IDs agree but6 selected F32 logits differ (max about7.63e-5). Policy1 MD5ff1489cffe4194bb52fe36eef08056c9 also equals S04gxf15's older unsharded policy1 file. Neither comparison isolates the head or sharding. N confirms the stored field is the target sampler global_logit; THIN1/THIN8 and small dense-stream/eligible multirow head-MMA dispatch are concrete possible boundaries, not attribution without matching committed-token operands/kernel/arguments. Earlier target widths remain possible causes. This is not a full-vocabulary comparison or an established invariant balanced-runtime form. Current policy1 speed is unmeasured (RND_S04:38; RND_R05:12; RND_N05:19,5Nov). Admission: G0; T.

**Precision class:** same bits for an unchanged product form; order-changing when a different head form is activated. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_head_stream"></a>

#### `--tp_head_stream`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use the dense-stream target head products.

**Use and interactions.** Nonzero requires TP4_EXL3. Head-MMA activation depends on the combined head strategy. Canonical THIN uses head_shards1/head_stream1/draft_shards1 and draft_layers0. S06 holds these head flags equal and changes only draft_policy0 to1: all8 IDs agree but6 selected F32 logits differ (max about7.63e-5). Policy1 MD5ff1489cffe4194bb52fe36eef08056c9 also equals S04gxf15's older unsharded policy1 file. Neither comparison isolates the head or sharding. N confirms the stored field is the target sampler global_logit; THIN1/THIN8 and small dense-stream/eligible multirow head-MMA dispatch are concrete possible boundaries, not attribution without matching committed-token operands/kernel/arguments. Earlier target widths remain possible causes. This is not a full-vocabulary comparison or an established invariant balanced-runtime form. Current policy1 speed is unmeasured (RND_S04:38; RND_R05:12; RND_N05:19,5Nov). Admission: G0; T.

**Precision class:** same bits for an unchanged product form; order-changing when a different head form is activated. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_head_mma"></a>

#### `--tp_head_mma`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use tensor-core target-head products on the BF16 recipe.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** head01/02:48-row443 versus dense_stream4642us, relative difference<=1e-5; split-K head8 284..291us at1/6/8 rows. Changes old target numerics; not a same-bits toggle (GAP_ANALYSIS.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_trunk_mma"></a>

#### `--tp_trunk_mma`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Use tensor-core BF16-row trunk products on the BF16 recipe.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** relbench02 six-row wq_a16.38 versus reference21.50us; BATCH12 wq_a95.2 ->23.6us. Fixed alternate sums; old-target equality not established (GAP_ANALYSIS.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_trunk_mma_fused"></a>

#### `--tp_trunk_mma_fused`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Sum trunk MMA split partials inside the 64-row kernel.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rj03p BATCH48: 38.44 -> 42.98 ms (+11.8%, slower), same bits (RND_J.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_trunk_mma_thin_splits"></a>

#### `--tp_trunk_mma_thin_splits`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Give 64-row trunk MMA products the THIN split geometry.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-dense_skinny_bf16"></a>

#### `--dense_skinny_bf16`

**Type/range:** unsigned32 decimal scalar; parser 0..8; semantic values: 0 retain default;1/4/8 outputs per block (other CLI-range values fail preparation). **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Tile BF16 skinny dense outputs per block above 48 rows; zero retains wide form.

Native help describes the selected path as: `--dense_skinny_bf16=0|4|8 (default 0; tp4_exl3 class 80: the BF16 skinny dense products above 48 rows at so many outputs a block, not 1; the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-dense_skinny_f32"></a>

#### `--dense_skinny_f32`

**Type/range:** unsigned32 decimal scalar; parser 0..8; semantic values: 0 retain default;1/4/8 outputs per block (other CLI-range values fail preparation). **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Tile FP32-weight skinny dense outputs per block above 48 rows.

Native help describes the selected path as: `--dense_skinny_f32=0|1|8 (default 0: 4; the float32-weight ones (routers) alike)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** ri05 mode1:30K8.488 ->8.694s (+206ms, slower), Fraser1.459s; identical (RND_I.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-dense_staged"></a>

#### `--dense_staged`

**Type/range:** unsigned32 decimal scalar; parser 0..64; semantic values: 0,8,16,32,64. **Default:** 16. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Stage wide dense-product inputs with a selectable output-block width.

Native help describes the selected path as: `--dense_staged=0|8|16|32|64 (default 16: a wide dense product's 8-row group staged once in shared memory for the selected number of outputs, each output's sums as before: the same bits; 0 the wide forms)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Admits0/8/16/32/64; nonzero must be a supported power of two. M leaves default16; wider standalone wins do not qualify a whole-model promotion. Admission: G0; T.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rk05 2K prompt1.175 ->1.150 s; Fraser1.393 ->1.385 s, 30K equal; same bits (RND_K.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_front_branch"></a>

#### `--tp_front_branch`

**Type/range:** unsigned32 decimal scalar; parser 0..2; semantic values: 0..2. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Branch THIN attention mix coefficients; mode two also branches FFN coefficients.

Native help describes the selected path as: `--tp_front_branch=0|1|2 (default 0; a thin step's attention mix coefficients a graph branch beside the site; 2: the FFN site's too; the same bits)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0; T.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** rj03p THIN C1 round 20.23 -> 19.94 ms (-1.4%), same bits; level2 20.12 ms (RND_J.md). S default remains0. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_decode_round_once"></a>

#### `--tp_decode_round_once`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** TP4_EXL3 THIN/BATCH.

**Mechanics.** Sum published BF16 target partials in ascending-rank FP32 and round the result once.

**Use and interactions.** Nonzero startup requires TP4_EXL3, BF16 wire, tp_wide_exchange>=2 and wide_rows>8. Effective only on non-draft THIN/BATCH target completion, including prompt tails using those forms; draft and FULL/WIDE collectives retain old arithmetic. Preparation requires a BF16 four-rank ring, direct shared channels, BF16 payloads, no parallel publisher/local stub/host incoming routes; otherwise refuses. Enabled completion bypasses the old BF16 ring/CE completion paths, preserves READY/DONE sequencing, adds BF16 rank partials in ascending-rank FP32 and rounds once. Changes target bits; independent references required. Default0/current axes retain old sums; nonzero is a raw research option with no active exact/balanced/fast mapping (S main.c:1236; model_v41_program.c:1858). Admission: G0; T.

**Precision class:** precision up; changed target bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `0` (all/generate); `batch_adaptive`: `0` (all/generate); `batch_session`: `0` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `0` (all/generate); `thin_full_layers`: `0` (all/generate).

**Measured effect.** N xd09_v7 same24/core2, RATE2/session0/learn1/costs14300,0,690 three-pass C1/C8 control0/candidate1/repeat0 tests pass each arm's 72 concurrent plus15 C1 text/native-count/finish comparisons per genre and exact control repeat. Candidate prose differs from legacy; code/structured match in these fixtures. Structured rates334.877/782.980 ->269.850/584.648, code241.247/448.561 ->198.730/345.576, prose170.027/469.349 ->139.366/351.201 tok/s (C1/C8). Candidate is about18-19% slower at C1 and23-25% slower at C8. Full-vocabulary deviation/quality capture was deferred. Main242e6ef contains this research implementation, but default0 remains; nonzero has no active axis mapping or default/merge recommendation. Inherited N evidence is not a new S qualification (packet RND_N.md 5 Nov03:28; xd09_v7_budget/precision_serving_fair.json and control_repeat_fair.json). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_wire_bf16"></a>

#### `--tp_wire_bf16`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** all admitted TP placements; nonzero refused on PP5.

**Mechanics.** Round TP transport partials/sums to BF16; zero retains FP32 wire.

**Use and interactions.** Nonzero requires an admitted TP placement and relaxed/fast numerics. PP5 and numerics=exact refuse it at startup, including EXL3. Original exact output shards require FP32 wire. Admission: G0; T.

**Precision class:** lossy. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `1` (all/generate); `batch_adaptive`: `1` (all/generate); `batch_session`: `1` (all/generate); `thin_fixed`: `1` (all/generate); `batch_compromise`: `1` (all/generate); `thin_full_layers`: `1` (all/generate).

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-linear_profile"></a>

#### `--linear_profile`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0 precision;1 speed. **Default:** 0. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Choose precision or speed linear-product preparation.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Selects measured slice counts; reference-order chains remain identical across slices. Isolated profile0/1 whole-workload effect unmeasured (source/linear.c, trunk01). Source: `source/main.c` and shipped knob catalog.


<a id="knob-linear_order"></a>

#### `--linear_order`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0 fixed;1 reference. **Default:** 1. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Choose fixed (zero) or source reference (one) product accumulation order.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to compare a specific numerical form against a pinned control. A matching output string is insufficient for an arithmetic-identity claim.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `original_tp_serial2`: `1` (all/generate); `original_pp_depth8`: `1` (all/generate); `original_pp_depth5`: `1` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `1` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-official_hc_rows"></a>

#### `--official_hc_rows`

**Type/range:** unsigned32 decimal scalar; parser 0..8; semantic values: parser0..8; admitted0/1/3/4/8. **Default:** 0. **Commands:** generate, serve. **Placement:** original PP THIN only for nonzero; TP and EXL3 unsupported.

**Mechanics.** Select pinned HC arithmetic with an explicit reference call shape.

Native help describes the selected path as: `--official_hc_rows=0|1|3|4|8 (default 0; experimental HC only, original thin PP)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Parser accepts0..8; model preparation admits only0/1/3/4/8. Nonzero requires original PP THIN, actual rows1/3/4/8, no allocated draft positions, row_capacity<=8, state16, linear_profile0, linear_order1 and CMP reference. WIDE, vision, TP and EXL3 are unsupported. Static reference shape does not switch prompt8 to cached1 or establish verification invariance; conditional snapshot identity retains legacy domain0. The original exact axis now uses official_reference1 for the bounded serial M1 composite; broader HC shape/state/verification joins remain open (S main.c:964; model_v41.c:186,204; model_v41_program.c:5044). Admission: G0.

**Precision class:** order-changing toward pinned-source HC; the standalone component does not establish full-model identity. The composite original reference profile has the bounded section 9.7 proof. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `0` (all/generate); `batch_adaptive`: `0` (all/generate); `batch_session`: `0` (all/generate); `thin_fixed`: `0` (all/generate); `batch_compromise`: `0` (all/generate); `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate); `thin_full_layers`: `0` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `1` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Merged default0 through the R HC subset in main75d3d17/S829ac34. Inherited XR04 ordinary preservation343/343 and first-attention HC1/HC8 checks retain their limited runtime scope. XR06 passes64 crossed direct/staged complete HC sites,48 fixed-F HC prefixes and corrected norm/rotation/observer gates. Not complete runtime composition, persisted logical-state, generation or speculation proof. Component CMP R1/F1 projection11.296us versus legacy5.040; stagedR8/F8 22.488us versus direct92.864/legacy12.064. No full S C1/C8 step-cost inference. Those XR06 results were a component dependency; the current original exact profile now selects HC1 under official_reference1, with its bounded composite proof in section 9.7. No nonzero balanced/fast/EXL3 HC axis mapping (RND_R.md 5 Nov02:46; FIXES.md:23-26). Source: `source/main.c` and shipped knob catalog.


<a id="knob-numerics"></a>

#### `--numerics`

**Type/range:** text/enum; exact / relaxed / fast. **Default:** exact **Commands:** load, generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Choose admitted startup numerical policy.

**Use and interactions.** Explicit exact refuses relaxed CMP order; relaxed/fast default to relaxed CMP order only for PP5. TP placements retain reference CMP order and refuse explicit relaxed. Exact is admitted only for PP5, TP4_PRO and TP4_EXL3. Admission: G0.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `relaxed` (all/generate); `batch_adaptive`: `relaxed` (all/generate); `batch_session`: `relaxed` (all/generate); `thin_fixed`: `relaxed` (all/generate); `batch_compromise`: `relaxed` (all/generate); `original_tp_serial2`: `exact` (all/generate); `original_pp_depth8`: `exact` (all/generate); `original_pp_depth5`: `exact` (all/generate); `thin_full_layers`: `relaxed` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `exact` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** source/startup_policy.c:101..135; RND_PIPELINE.md. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-placement"></a>

#### `--placement`

**Type/range:** text/enum; pp5 / tp4_pro / tp4_exl3 / tp2_pp2_pro / tp2_single_single_pro. **Default:** pp5 **Commands:** load, generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Choose prepared weight/state placement.

**Use and interactions.** tp5 parses in policy helper but is refused by admission. PP5 permits layout; TP layouts refuse it. Native PRO output depends on its layers. Admission: G0.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `tp4_exl3` (all/generate); `batch_adaptive`: `tp4_exl3` (all/generate); `batch_session`: `tp4_exl3` (all/generate); `thin_fixed`: `tp4_exl3` (all/generate); `batch_compromise`: `tp4_exl3` (all/generate); `original_tp_serial2`: `tp4_pro` (all/generate); `original_pp_depth8`: `pp5` (all/generate); `original_pp_depth5`: `pp5` (all/generate); `thin_full_layers`: `tp4_exl3` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `pp5` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** source/startup_policy.c; COMPARISON_MATRIX.md recipe epochs; no isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-cmp_order"></a>

#### `--cmp_order`

**Type/range:** text/enum; reference / relaxed. **Default:** reference **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Choose CMP trunk/expert order-changing.

**Use and interactions.** Explicit numerics=exact requires reference; omitted numerics resolves legacy relaxed CMP to relaxed policy. Admission: G0.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `batch_fixed`: `reference` (all/generate); `batch_adaptive`: `reference` (all/generate); `batch_session`: `reference` (all/generate); `thin_fixed`: `reference` (all/generate); `batch_compromise`: `reference` (all/generate); `original_tp_serial2`: `reference` (all/generate); `original_pp_depth8`: `reference` (all/generate); `original_pp_depth5`: `reference` (all/generate); `thin_full_layers`: `reference` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `reference` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** source/startup_policy.c. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.



<a id="knob-official_reference"></a>

#### `--official_reference`

**Type/range:** unsigned32, 0 or 1. **Default:** 0 for every raw placement. **Commands:** generate, serve. The original-weight exact axis inserts 1 with the bounded PP THIN serial M1 settings; balanced/no-axis paths retain 0.

**Mechanics and interactions.** Select the prepared original-weight reference compositions, including the qualified HC, norm/rotation, round-trip, window, router/core and head forms. Requires PP5, one ordinary row, no wide/full program, capacity at most 512, one session, one pipeline step and one chunk in flight, no speculation/BATCH, official_hc_rows1, native PRO, exact/reference CMP and linear arithmetic, and no incompatible TP/overlap/original experiment path. Serving additionally requires prefix_reuse0, prefix_reuse_exact0 and session_fork0. The exact axis supplies its pinned layout; conflicting overrides refuse.

**Precision, effects and axes.** Original reference arithmetic within the admitted serial M1 scope; no new balanced arithmetic. Exact F32 logits do not guarantee stochastic ID identity, since the RNG/sampler order differs at T>0. The qualified T=0 fixtures and C1 cost of 9.7375× PP5 are detailed in section 9.7. Axis mapping: original weights, exactness=exact, concurrency=single, category code/prose/structured/balanced and all four focus values, always constrained to this 512-token recipe. EXL3 exact refuses; wider/batched/TP/speculative/vision scopes remain open. Source: source/main.c, tools/cli/axes.json and Track R USAGE_REFERENCE_v4.md.


### 8.10. Cache, prefix reuse and snapshots

The default namespace policy is scope0 and prefix reuse is enabled, with exact-reuse mode0. A cache hit needs both namespace equality and eligible numerical/state provenance. Exact modes1/2 restrict same-length cold-provenance reuse and require row-invariant forms; Main stable1 adds qualified growing-prefix reuse with a separate cold identity; it does not enable the copy32/v5 branch. Snapshot policy uses actual saved token progress and arithmetic/namespace identity. Omitted save/load/idle thresholds use1024/256/120; the main CLI requires positive values. The server API zero-initialized default convention is not accepted explicit CLI syntax.

<a id="knob-prefix_reuse"></a>

#### `--prefix_reuse`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Reuse eligible cached prompt prefixes instead of prefilling cold.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** order-changing (cross-plan risk; see P). **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. Shipped resolver writes: `batch_fixed`: `0` (serve); `batch_adaptive`: `0` (serve); `batch_session`: `0` (serve); `thin_fixed`: `0` (serve); `batch_compromise`: `0` (serve); `original_tp_serial2`: `0` (serve); `original_pp_depth8`: `0` (serve); `original_pp_depth5`: `0` (serve); `thin_full_layers`: `0` (serve). The `original_reference_serial_m1` exact strategy additionally inserts `0` for serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** xc01 baseline: cold-solo TTFT median0.712954s versus legacy repeat0.014448s; repeat text matches cold14/16. 25005-token cold7.697177s versus repeat0.015356s agrees; 6004-token repeat differs. Observed speed/identity tradeoff, not exact-cache qualification (RND_P.md 16:07; baseline_repeat_cost.json). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefix_reuse_exact"></a>

#### `--prefix_reuse_exact`

**Type/range:** unsigned32 decimal scalar; parser 0..2; semantic values: 0 legacy;1 final-row repeat;2 cold-plan boundaries. **Default:** 0. **Commands:** serve. **Placement:** serve: mode0 legacy on admitted placements; nonzero TP4_EXL3 only.

**Mechanics.** Admit same-length cold-provenance prompt prefixes through repeat or cold-plan boundaries.

**Use and interactions.** Nonzero requires TP4_EXL3, ring_column_owners1, batch_thin_bits1, chunks_in_flight1, prefill_fast_plan0, prefill_chunk_rows0 and prefill_decode_rows0; multiple slots require prefill_first1 or prefill_order1/2. Namespace equality precedes numeric admission. Modes1/2 require known cold_prompt_count equal to this request length, no image and no imported snapshot. Kept tokens round down to an eight-token boundary. Mode1 additionally keeps exactly prompt_count-1 and not before decoder_from; mode2 calls the child-length cold-plan boundary helper and validates any fallback checkpoint. Ordinary reach lag is<=3968; otherwise only a valid checkpoint can reuse. Prefix_reuse0 disables reuse. Server open refuses any session_directory with exact mode1/2. No nonshrinking/growing stable1 or fork admission exists in this merged source (S main.c:935,1368; server.c:66-135,1232). Admission: G0; nonzero startup and cold-provenance gates in Interactions.

**Precision class:** same bits within the qualified cold recipe; target-form invariance required. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The `original_reference_serial_m1` exact strategy additionally inserts `0` for serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Merged raw/default0 retains legacy policy. P c52ce71 modes1/2 retain16cold+160warm+14stress and separate114coldABA/xc08 checkpoint proofs; Q final06 gpq05 closes its57request BATCH coldABA, not a fresh114matrix. Main5029986 mpqa2048/mpqb1024 passes full8/256 selected records06:15:58-06:19:13; fresh S8d6bb95 qualification is separate. No automatic axis selection or original-weight-equivalence claim. Unmerged P stable1 extension evidence is confined to its branch row (RND_P/Q.md; RND_Q5Nov06:23). Source: `source/main.c` and shipped knob catalog.


<a id="knob-cache_scope"></a>

#### `--cache_scope`

**Type/range:** unsigned32 decimal scalar; parser 0..2; semantic values: 0 session;1 user;2 global. **Default:** 0. **Commands:** serve. **Placement:** serve: all admitted placements; explicit namespace policy.

**Mechanics.** Choose the namespace used for tokenizer, slot and snapshot reuse.

Native help describes the selected path as: `--cache_scope=0|1|2 (default 0: API user plus session/cache signals; 1: API user; 2: global)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** 0 SESSION combines authenticated user with every recognized body/metadata/session-header identity; absent/null/empty values add no signal. UTF-8 identity strings<=512bytes each and total<=8192bytes; invalid types/limits or repeated identity headers return400. 1 USER scopes by authenticated user only; 2 GLOBAL explicitly permits cross-user/session content reuse. Anonymous requests share user "-". Namespaces gate tokenizer/KV/snapshot matching and convey neither authorization nor stored-response semantics. Snapshotv4 binds the complete namespace on the opened descriptor; old unscopedv3 files remain untouched. No session_fork or automatic same-owner cloning is parsed by S (S main.c:937; server_cache.c:180; server_sessions.c:39,150,255; server.h:42). Admission: G0; namespace/snapshot/idle eligibility in Interactions.

**Precision class:** same bits; service and reuse eligibility policy. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Merged default0. Frozen Q gpq01 namespace112/112, final06 native/Responses and gpq05 cold controls pass within their epochs; actualCodex gpq06 passes18Responses/17laterhits, not paired latency evidence. Main5029986 both-width references and one-slot two-user Chat/Responses smoke pass: each HTTP200/1905input/8output, equal content/reasoning, cache0 for both users. First pre-binary permission failure remains retained. Service policy orthogonal to all axes; no S axes/default change (RND_Q.md5Nov03:20,04:20,06:19,06:23). Source: `source/main.c` and shipped knob catalog.


<a id="knob-session_grace_seconds"></a>

#### `--session_grace_seconds`

**Type/range:** unsigned32 decimal scalar; parser 0..3600; semantic values: 0..3600. **Default:** 30. **Commands:** serve. **Placement:** serve: all admitted placements; idle-slot policy.

**Mechanics.** Prefer retaining a recently finished namespace when choosing an idle slot.

Native help describes the selected path as: `--session_grace_seconds=0..3600 (default 30; 0 disables affinity grace)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** CLI0..3600 seconds; default30, zero disables grace. Protection is a preference, never a reservation that stalls dispatch: if all free slots are protected one is still chosen. Only idle slots qualify; explicit-session affinity breaks equal-prefix ties without bypassing namespace/content/image/reach checks (S main.c:939; server.c:137-197; server.h:50,114). Admission: G0; namespace/snapshot/idle eligibility in Interactions.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Merged default30. Q namespace/affinity and merged smoke proofs are aggregate service-policy evidence; isolated latency/capacity effect unmeasured. Orthogonal to the four axes; no automatic mapping. Source: `source/main.c` and shipped knob catalog.


<a id="knob-session_save_tokens"></a>

#### `--session_save_tokens`

**Type/range:** unsigned32 decimal scalar; parser 1..1048576; semantic values: 1..1048576. **Default:** 1024 when omitted; explicit CLI0 is refused (server API zero-initialization is a separate convention). **Commands:** serve. **Placement:** serve: configured eligible snapshot store on admitted placements.

**Mechanics.** Require this many processed tokens absent from a snapshot before saving.

Native help describes the selected path as: `--session_save_tokens=N (default 1024)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** CLI1..1048576; default1024. API zero selects1024 but is not a CLI spelling. With a session directory, eviction/idle/shutdown saves require at least this many processed-minus-saved tokens; a continuing same-namespace prefix also checks the discarded unsaved suffix. Snapshotv4, namespace, image and engine state eligibility remain; this knob cannot enable a directory with prefix_reuse0 or exact reuse1/2 (S main.c:941; server.c:1232; server_sessions.c:407-423,616-654; server.h:111). Admission: G0; namespace/snapshot/idle eligibility in Interactions.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Merged default1024. Snapshot and namespace correctness proofs do not supply an isolated performance result; no automatic axes mapping. Source: `source/main.c` and shipped knob catalog.


<a id="knob-session_load_tokens"></a>

#### `--session_load_tokens`

**Type/range:** unsigned32 decimal scalar; parser 1..1048576; semantic values: 1..1048576. **Default:** 256 when omitted; explicit CLI0 is refused. **Commands:** serve. **Placement:** serve: configured eligible snapshot store on admitted placements.

**Mechanics.** Require this much reusable-token gain before loading a snapshot.

Native help describes the selected path as: `--session_load_tokens=N (default 256)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** CLI1..1048576; default256. API zero selects256 but is not a CLI spelling. Select only a namespace/image-compatible valid snapshot that keeps more than the current slot and gains>=this threshold. Complete opened-file v4 header/token/namespace and engine load checks remain. Session directory requires prefix_reuse1 and exact reuse0 (S main.c:943; server.c:1232; server_sessions.c:255-286,423-445; server.h:112). Admission: G0; namespace/snapshot/idle eligibility in Interactions.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Merged default256. Version/namespace/replacement correctness gates do not measure an isolated latency or throughput gain; no automatic axes mapping. Source: `source/main.c` and shipped knob catalog.


<a id="knob-session_idle_save_seconds"></a>

#### `--session_idle_save_seconds`

**Type/range:** unsigned32 decimal scalar; parser 1..86400; semantic values: 1..86400. **Default:** 120 when omitted; explicit CLI0 is refused. **Commands:** serve. **Placement:** serve: configured eligible snapshot store on admitted placements.

**Mechanics.** Wait this long without work before starting an eligible idle snapshot save.

Native help describes the selected path as: `--session_idle_save_seconds=1..86400 (default 120)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** CLI1..86400 seconds; default120. API zero selects120 but is not a CLI spelling. Idle saving requires store present, healthy engine, no queued or active commands and a slot meeting session_save_tokens; then saves one eligible slot at a time. Shutdown saving uses its separate drain path and does not wait this idle timer. Directory requires prefix_reuse1 and exact reuse0 (S main.c:945; server_sessions.c:616-654; server.h:113-115). Admission: G0; namespace/snapshot/idle eligibility in Interactions.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Merged default120. No isolated snapshot-I/O/performance measurement or automatic axes mapping. Source: `source/main.c` and shipped knob catalog.


<a id="knob-session_directory"></a>

#### `--session_directory`

**Type/range:** text/enum; path. **Default:** none **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Persist eligible session snapshots in this directory (requires prefix_reuse=1).

**Use and interactions.** Ordinary command/parser and runtime value gates apply. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-session_fork"></a>

#### `--session_fork`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0 or1. **Default:** 0. **Commands:** serve. **Placement:** Qualified stable-prefix TP4_EXL3 recipe.

**Mechanics.** Copy an eligible same-owner prefix into an independent child session slot.

**Use and interactions.** Startup requires cache_scope0. Runtime needs prefix reuse, >1slots, max_tokens>0, healthy engine and no prior attempt. Idle/read-unleased donor: different full namespace, same authenticated hard owner; hints confer no authority. Cold-plan kept>0, reach>=kept, reach-kept<=3968; ordinary state only, never checkpoint-only. Reserve/check real generations/read lease, drain copies before publication; failed/cancelled accepted clone clears destination history and may retry cold only if engine healthy. Backend failure poisons the engine; no aliases. Growing children additionally need P stable1 provenance; no invented snapshot/checkpoint proof (Qfork main.c:947,1377; server_forks.c:19,34,130,205). Latest native92af3e2 requires stable1 for every new cross-session copy; stable0 falls back cold. Existing legacy same-session behavior is retained, not newly qualified bit-exact. Admission: Default off; cache_scope0 and qualified producer capability required.

**Precision class:** same bits for exact-copy service policy; target-form qualification required. **Four-axis mapping:** Orthogonal service policy; all categories only where qualified, concurrency=multiple independent slots, cache reuse/TTFT scope. No automatic tuple or official-exact claim. The `original_reference_serial_m1` exact strategy additionally inserts `0` for serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Q92af3e2 is integrated default off. Complete gpq12 fresh24-request legacy fallback supplement passed; gpq11 retains its earlier whole-job failure. New cross-session copies require stable1; donor ownership, namespace, leases and generations remain checked. Main integration smoke is independently required (MERGE_A21_v4.md). Source: `source/main.c` and shipped knob catalog.


### 8.11. HTTP server, authentication and queues

The listening socket, bearer-user authentication and fair waiting queues are independent of model arithmetic. A named server key file can contain several users; a detached stats client takes a single-key file with a different format. Never read or paste a key file into a troubleshooting log. Waiting queue depth excludes active slots. Per-user FIFO and inter-user round robin prevent one waiting user from consuming every reserved queue place unless the operator explicitly chooses that policy.

Generate a named key privately, then protect the resulting key file with owner-only permissions before serving. The documentation check discards the generated stdout and never reads a key file. Keygen itself does not set the file mode; use an owner-only umask in the deployment shell.

<a id="knob-queue_depth"></a>

#### `--queue_depth`

**Type/range:** unsigned32 decimal scalar; parser 1..64; semantic values: 1..64. **Default:** 16. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Bound all waiting server requests.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-user_queue_depth"></a>

#### `--user_queue_depth`

**Type/range:** unsigned32 decimal scalar; parser 1..64; semantic values: 1..64. **Default:** Omitted: automatic max(1,queue_depth/2), represented internally by0. Explicit CLI0 is refused; accepted explicit values are1..queue_depth. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Bound waiting requests per API user.

**Use and interactions.** Must not exceed queue_depth; omitted becomes half global queue (at least1). Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-port"></a>

#### `--port`

**Type/range:** unsigned32 decimal scalar; parser 1..65535; semantic values: 1..65535. **Default:** 8321. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Listen on this TCP port.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this within the named subsystem and retain a startup/result receipt when changing it.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-bind_address"></a>

#### `--bind_address`

**Type/range:** text/enum; numeric IPv4/IPv6. **Default:** 127.0.0.1 **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Bind the server listener.

**Use and interactions.** Ordinary command/parser and runtime value gates apply. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-key_file"></a>

#### `--key_file`

**Type/range:** text/enum; path. **Default:** none **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Read API-user authentication records at startup.

**Use and interactions.** Regular, nonempty file <=1MiB with no group/other permission bits; at most64 unique users/keys, exact record format validated (server_keys.c:43..143). Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-stats_users"></a>

#### `--stats_users`

**Type/range:** text/enum; comma list of <=64 usernames; each1..32 ASCII letters/digits/._-. **Default:** all authenticated users; open stats when auth disabled **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Authorize named users to access the statistics endpoint.

**Use and interactions.** Requires key_file; no empty names; endpoint authorization allowlist, independent of live-statistics presentation (server.c:595..600,1056..1072). Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-model_alias"></a>

#### `--model_alias`

**Type/range:** text/enum; 1..63 bytes. **Default:** none **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Add a request/listing model alias.

**Use and interactions.** Supplied alias must fit63 bytes; adds a name without replacing the model identity (server.c:1011..1013). Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-port-stats"></a>

#### `--port` — stats

**Type/range:** text/enum; 1..65535. **Default:** required **Commands:** stats. **Placement:** all placements/recipes.

**Mechanics.** Choose statistics server port.

**Use and interactions.** Stats command contract only. Admission: G0; stats numeric host/port required; duplicates refuse.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. Source: `source/stats_client.c` and shipped knob catalog.


<a id="knob-key_file-stats"></a>

#### `--key_file` — stats

**Type/range:** text/enum; path. **Default:** none **Commands:** stats. **Placement:** all placements/recipes.

**Mechanics.** Read a statistics API key without exposing it.

**Use and interactions.** Single-key file:16..4096 ASCII letters/digits/-._~+/=, optional final CR/LF; regular file with no group/other permissions. Multi-record/commented serve key files refuse (server.c:961..995). Admission: G0; stats numeric host/port required; duplicates refuse.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. Source: `source/stats_client.c` and shipped knob catalog.


<a id="knob-user-keygen"></a>

#### `--user` — keygen

**Type/range:** text/enum; 1..32 ASCII letters/digits/dot/underscore/hyphen. **Default:** required **Commands:** keygen. **Placement:** all placements/recipes.

**Mechanics.** Choose the user for key generation.

**Use and interactions.** None beyond command contract. Admission: exactly one --user argument.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/server_keys.c:230. Unmeasured; no model computation. Source: `source/server_keys.c` and shipped knob catalog.


### 8.12. Sampling, output and template controls

The CLI establishes defaults for serving requests; the HTTP body can override supported controls. Greedy temperature0 is deterministic only with the same numerical path, prompt, state, seed-sensitive controls and schedule invariants. CLI seed32 and HTTP seed64 have different ranges. The output limit is generated tokens, not total context. Reasoning effort changes template/thinking behavior and is not a hard budget on hidden reasoning tokens.

<a id="knob-generated_tokens"></a>

#### `--generated_tokens`

**Type/range:** unsigned32 decimal scalar; parser 1..4294967295; semantic values: 1..4294967295. **Default:** Generate64; serve4294967295 (effective context still bounds it). **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Set the output-token cap for generate. In serve, set the default output limit for requests that omit one; an explicit request max_tokens/max_completion_tokens replaces this default. Context capacity still bounds execution.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to set the request default deliberately; hold the rendered prompt and numerical recipe fixed when comparing output.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-seed"></a>

#### `--seed`

**Type/range:** unsigned32 decimal scalar; parser 0..4294967295; semantic values: 0..4294967295. **Default:** Generate0; serve draws a request seed when neither request nor server explicitly seeds. CLI value0 explicitly seeds, not random. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Set sampler seed; omitted serving requests draw a seed.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to set the request default deliberately; hold the rendered prompt and numerical recipe fixed when comparing output.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-reasoning_effort"></a>

#### `--reasoning_effort`

**Type/range:** unsigned32 decimal scalar; parser 0..100; semantic values: 0..100. **Default:** 0. **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Set template reasoning effort from zero (off) through 100.

**Use and interactions.** Prepared model/driver validation remains authoritative; ignored outside the named implementation path. Admission: G0.

Use this to set the request default deliberately; hold the rendered prompt and numerical recipe fixed when comparing output.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-temperature"></a>

#### `--temperature`

**Type/range:** finite floating-point scalar; finite float >=0. **Default:** 0 **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Set sampling temperature; zero is greedy.

**Use and interactions.** Ordinary command/parser and runtime value gates apply. Admission: G0.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-top_p"></a>

#### `--top_p`

**Type/range:** finite floating-point scalar; float >0 and <=1. **Default:** 1 **Commands:** generate, serve. **Placement:** all placements/recipes.

**Mechanics.** Set nucleus probability cutoff.

**Use and interactions.** Ordinary command/parser and runtime value gates apply. Admission: G0.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_usage"></a>

#### `--draft_usage`

**Type/range:** text/enum; none / compact / full. **Default:** compact **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Choose draft statistics detail in responses.

**Use and interactions.** Ordinary command/parser and runtime value gates apply. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.



<a id="knob-ignore_eos"></a>

#### `--ignore_eos`

**Type/range:** unsigned32, 0 or 1. **Default:** 0 on every placement. **Commands:** generate, serve; non-speculative THIN only. Value 1 continues after EOS until the ordinary output/context limit, useful for the reference fixture's fixed 256-draw horizon. Normal serving should retain EOS stopping. Nonzero refuses speculation or BATCH; existing placement/engine gates still apply.

**Precision and axes.** Output stopping policy, not a model-arithmetic change. It does not synchronize stochastic RNG or enable a new numerical tier. The exact reference fixture uses T=0 and explicit1 to continue after draw135; no axis turns it on automatically. Cost is the additional requested post-EOS generation, with no isolated speed measurement. Source: source/main.c and Track R USAGE_REFERENCE_v4.md.


<a id="knob-prompt_token_ids"></a>

#### `--prompt_token_ids`

**Type/range:** nonempty file path; default absent on every placement. **Command:** generate only. The file is a nonempty JSON array of unsigned decimal IDs below the pack vocabulary size, with room left for generation. It replaces the messages-file input, bypasses tokenization/chat rendering and accepts exactly one weight source. It supplies prompt IDs, never generated tokens or activations. Standard positional syntax is TOKENIZER_TABLES WEIGHT_SOURCE after this option, with no messages positional argument. Serve refuses the option.

**Precision, effects and axes.** Input/template control, with no change to model arithmetic. Use it to reproduce the published eight-ID serial source without an appended assistant prefix. The original exact fixture's full-head/ID/logit proofs are scoped in section 9.7; no isolated speed gain or general exactness admission is implied by bypassing tokenization. No axis inserts a token-ID filename. Source: source/main.c prompt_token_ids_parse and Track R USAGE_REFERENCE_v4.md.


### 8.13. Logging, profiling and diagnostic dumps

Logging and counter presentation normally observe the selected computation, but observation overhead is not measured as zero. Release compiles timing/checking hooks out. The main parser's evaluated bounds can therefore reduce a diagnostic option to zero only. Dumps and environment probes are research interfaces with file/format and semantic constraints; they are not normal API output or stable model-result formats.

<a id="knob-live_stats"></a>

#### `--live_stats`

**Type/range:** unsigned32 decimal scalar; parser 0..60000; semantic values: 0..60000. **Default:** 0. **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Print server counters at this interval in milliseconds; zero disables them.

**Use and interactions.** Nonzero must be >=100ms. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-draft_trace"></a>

#### `--draft_trace`

**Type/range:** unsigned32 decimal scalar; parser 0..0; semantic values: 0 in release; 0 or 1 in debug/thread/profile. **Default:** 0. **Commands:** generate, serve. **Placement:** generate/serve with corresponding drafter; all placements where admitted.

**Mechanics.** Print per-round draft anchors and proposals for diagnostics.

**Use and interactions.** Release accepts only 0; debug/thread/profile accept 0 or 1, with optional diagnostics compiled only when timing hooks are enabled (N c565010, S095a3eb). No axis enables it. Admission: G0; T.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


In this release build the evaluated upper bound is0, so the checked example disables tracing. A profile/debug build with timing hooks has a different upper bound; repeat its parser check before enabling1.

<a id="knob-dump_draft_ring"></a>

#### `--dump_draft_ring`

**Type/range:** text/enum; path. **Default:** none **Commands:** generate; serve parser also accepts it but serving performs no dump. **Placement:** all placements/recipes.

**Mechanics.** Write diagnostic draft-ring rows at shutdown.

**Use and interactions.** Ordinary command/parser and runtime value gates apply. The common text parser also recognizes this spelling on serve; the dump itself is guarded by !is_serving and remains inert there. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-live_stats_format"></a>

#### `--live_stats_format`

**Type/range:** text/enum; text / json / table. **Default:** text **Commands:** serve. **Placement:** all placements/recipes.

**Mechanics.** Choose live server statistics format.

**Use and interactions.** Ordinary command/parser and runtime value gates apply. Admission: G0.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/main.c custom parser and named implementation gates. Unmeasured isolated effect. Source: `source/main.c` and shipped knob catalog.


<a id="knob-host-stats"></a>

#### `--host` — stats

**Type/range:** text/enum; numeric IPv4/IPv6. **Default:** 127.0.0.1 **Commands:** stats. **Placement:** all placements/recipes.

**Mechanics.** Connect to a statistics server.

**Use and interactions.** Stats command contract only. Admission: G0; stats numeric host/port required; duplicates refuse.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. Source: `source/stats_client.c` and shipped knob catalog.


<a id="knob-interval_ms-stats"></a>

#### `--interval_ms` — stats

**Type/range:** text/enum; 100..60000. **Default:** 1000 **Commands:** stats. **Placement:** all placements/recipes.

**Mechanics.** Poll counters at this interval.

**Use and interactions.** Stats command contract only. Admission: G0; stats numeric host/port required; duplicates refuse.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. Source: `source/stats_client.c` and shipped knob catalog.


<a id="knob-format-stats"></a>

#### `--format` — stats

**Type/range:** text/enum; text / json / table. **Default:** text **Commands:** stats. **Placement:** all placements/recipes.

**Mechanics.** Choose statistics presentation.

**Use and interactions.** Stats command contract only. Admission: G0; stats numeric host/port required; duplicates refuse.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. Source: `source/stats_client.c` and shipped knob catalog.


<a id="knob-once-stats"></a>

#### `--once` — stats

**Type/range:** flag; flag without value. **Default:** false **Commands:** stats. **Placement:** all placements/recipes.

**Mechanics.** Print one statistics response and exit.

**Use and interactions.** Stats command contract only. Admission: G0; stats numeric host/port required; duplicates refuse.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** source/stats_client.c:76 and server.c stats key reader. Unmeasured performance effect. Source: `source/stats_client.c` and shipped knob catalog.


<a id="knob-help-scheduling"></a>

#### `--help-scheduling`

**Type/range:** flag; flag without value. **Default:** not requested **Commands:** serve. **Placement:** host help; no model or card opened.

**Mechanics.** Print scheduling examples and measured pause/TTFT tradeoffs without opening a model.

**Use and interactions.** Exactly anemone serve --help-scheduling; do not append model paths or axis options. Preserve the dedicated help when merging topic routing. Admission: Exactly anemone serve --help-scheduling.

**Precision class:** same bits. **Four-axis mapping:** Category: orthogonal to content. Exactness: preserve the selected recipe subject to the stated cache/layout/diagnostic gates. Concurrency: resource/namespace policy, not an automatic single/multiple choice. Focus: operational startup, capacity, cache or observation as described. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** M7d86c45/xp04: all required reference/boot/mixed/decode gates pass; table retains scoped pilot tradeoffs and final default. M12qualified BATCH1024 only; S qualification independent (packet RND_M.md19:14). Source: `source/main.c` and shipped knob catalog.


### 8.14. Original weights and PRO arithmetic

These controls belong to the original FP8/FP4 placement family. Native PRO math is the default and layer boundaries are part of the numerical recipe. The original experiment controls have strict composition gates: original BATCH is exactly48rows/eight verifiers/eight slots/depth1 with its named exclusions, not arbitrary EXL3 BATCH porting. Original wide experts, exact chunk core and staging each need their own allowed shapes. Compatible TP4_PRO wide experts now default1; other named pilots retain0. PP5 multi-session DSpark serve defaults depth5 when omitted, while explicit axis settings retain their chosen depths.

<a id="knob-pro_emulation"></a>

#### `--pro_emulation`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** original PP5 PRO layers; nonzero refused on TP.

**Mechanics.** Emulate class-80 sums on the PRO; zero uses class-120 native arithmetic.

Native help describes the selected path as: `--pro_emulation=0|1 (default 0: the PRO's own arithmetic, bits depend on the placement)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Native PRO outputs depend on layers placed there; emulation costs2..4x on affected PP5 PRO products. Nonzero is refused by every current TP startup path (MEMORY.md; main.c startup_placement_status). Admission: G0.

**Precision class:** order-changing. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_code_wire"></a>

#### `--tp_code_wire`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** original TP4_PRO/TP2 variants.

**Mechanics.** Transport original quantized codes/scales losslessly across TP ranks.

**Use and interactions.** Refused for TP4_EXL3. Original-weight TP paths only. Admission: G0; T.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Unmeasured isolated effect. Source implementation/CLI contract only. Source: `source/main.c` and shipped knob catalog.


<a id="knob-original_dense_staged"></a>

#### `--original_dense_staged`

**Type/range:** unsigned32 decimal scalar; parser 0..16; semantic values: parser0..16; admitted0 or16. **Default:** 0. **Commands:** load, generate, serve. **Placement:** original TP4_PRO CMP wide dense products.

**Mechanics.** Stage original TP4 BF16/FP32 dense products.

Native help describes the selected path as: `--original_dense_staged=0|16 (default 0; original TP4 CMP BF16 wide dense candidate; 0 retains the old forms)`. Evaluated defaults/ranges above take precedence where this prose is build-dependent.

**Use and interactions.** Parser accepts0..16, but nonzero preparation admits only16 and placement=tp4_pro. There is no additional exact-only numerics gate; eligible BF16 input, BF16/F32 output, size/row-scale and >=64-row dense restrictions remain. Ineligible shapes retain or refuse through the named preparation path (S main.c:1224; dense_linear.c). Admission: G0; T.

**Precision class:** same bits. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Merged default0. Inherited XO04 eight dense shapes are byte-identical. XO05 selected/digest/canonical-state continuity passes Fraser/2K/30K; strict nine-request cold controls all cached0 give staged967 versus TP965 tok/s. Older962/965 values were cold-HOLD. No material whole-model gain, fresh S speed result or automatic prefill/ttft axis promotion (original_axis_handoff_v2.md:77,101; RND_O.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-original_wide_experts"></a>

#### `--original_wide_experts`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 1 only for compatible exact TP4_PRO wide64..1024/reference/native-PRO/FP32-wire recipes with no TC-MoE, BATCH, code/owner path, EXL3 wide entry, split/full-split or original_chunk_core.0 elsewhere; explicit0 retains slabs. **Commands:** load, generate, serve. **Placement:** original exact TP4_PRO CMP wide experts; native PRO retained.

**Mechanics.** Run original TP4 expert jobs over complete chunks with lossless slab gathers.

**Use and interactions.** Nonzero requires TP4_PRO exact, native PRO, precision/reference linear and CMP-reference, FP32 wire, wide64..1024, code_wire0, attention_owner0, no BATCH, exl3_wide_entry0 and split/full-split0. Model additionally requires quad exact-output shards, no EXL3 TC-MoE and compatible state/row geometry. Separate core/overlap combinations need their own gates (S main.c:1236; model_v41_tp2_model.c:98). Admission: G0; T.

**Precision class:** same bits in recorded current-recipe fixtures; joins unqualified. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Measured O source75 raw/alias both-parent gate now passes: cold-prefill median967->1363tok/s(+40.95%); Fraser C1/C4/C8 medianTTFT4417.75/17609.85/35087.25->3197.60/12706.85/25290.30ms(-27.62/-27.84/-27.92%); aggregate end-to-end throughput improves9.63/9.72/9.69%, while C8per-stream rate falls0.48%. All46fullFraser responses plus9fixedAPI outputs/counts/finishes agree; fixed-input GEN00/10 preserve selected records,904960activeF32 logits and canonical state. Cold nonce calls have small prompt-count differences, not identical input IDs. NativeV1windows remain exit1; independent repair-aware whole-proof equality/alias lineage passes, first50sCPUreader attempt failed then unchanged120sallowance passed. Not currentS10/official/overlap+wide/mixed/1M proof. Historical raw default0/no S axis; final main now selects1 only within compatible admission; conditional original/balanced/single-or-multiple/prefill-or-ttft mapping was held at the measurement epoch; current parser default resolution is independently checked here (O REPORT_v3.md; evidence/o3_tp_wide_both_parent_root02.json SHA0da4dae92d6e9a72ad6b1cadc26f5d34a2eec4ea2ea6b337bdc70dcff2a6df38). Historical GEN06 pilot remains in RND_O.md. Integration batch selects this default only within existing admission, with chunk_core0; explicit0 retains the slab path. Source: `source/main.c` and shipped knob catalog.


<a id="knob-original_chunk_core"></a>

#### `--original_chunk_core`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** load, generate, serve. **Placement:** original exact TP4_PRO CMP full-head chunk core; native PRO retained.

**Mechanics.** Use original TP4 full-head attention core on whole eight-row slabs.

**Use and interactions.** Nonzero requires TP4_PRO exact/native PRO/precision/reference/CMP-reference/FP32, wide64..1024 divisible by8, code_wire0, attention_owner0, no BATCH/original_batch/wide experts and split/full-split0. Effective only on eligible CMP original full-head chunks with multiple slabs, one piece, no vision, state16 and the retained prefix topology; other core schedules remain separate (S main.c:1272; model_v41_program.c:6056). Admission: G0; T.

**Precision class:** same bits in recorded current-recipe fixtures; joins unqualified. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Merged default0. Inherited source8a434dc/GEN03 completes3 Fraser/2K/30K knob0/1 pairs and3 XO05 continuity controls with power restoration. Selected records, prompt IDs, digests, canonical SAVE and active last-target full-vocabulary F32 bits agree. Single-arm CLI prompt+first diagnostics improve about2%, not HTTP TTFT, fresh S serving or a core/query/wide join. No active S axis selects nonzero; no official-reference claim (RND_O.md 5 Nov02:32,02:42). Source: `source/main.c` and shipped knob catalog.


<a id="knob-original_batch"></a>

#### `--original_batch`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** serve. **Placement:** original exact TP4_PRO BATCH48, eight slots/verifiers, pipeline1.

**Mechanics.** Admit the fixed original TP4 BATCH48/eight-verifier/eight-slot pilot.

**Use and interactions.** Serve-only. Nonzero requires TP4_PRO exact/native PRO/precision/reference/CMP-reference/FP32, exactly one batch_rows48/verifiers8 entry, session_slots8, pipeline1, wide64..1024, DSpark/hybrid block5/positions5. Requires context_slabs1, core_once0, draft_wide0. Refuses code/owner/head/draft shards, draft layers, head streaming/MMA, trunk/router MMA, wide_head, draft_rows, BF16 draft variants, original wide/staged, EXL3 entry, split/full-split and pipeline overlap. Defaults are not rewritten to satisfy this pilot (S main.c:1246; model_v41_tp2_model.c:95). Admission: G0; T; B.

**Precision class:** same bits in recorded current-recipe scope; all-step scope unqualified. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `original_tp_serial2`: `0` (serve); `original_pp_depth8`: `0` (serve); `original_pp_depth5`: `0` (serve). The `original_reference_serial_m1` exact strategy additionally inserts `0` for serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Merged default0. Inherited XO08 target/GEN preservation and XO09 source434a895/rndo_05 serving pass. Six eligible architecture windows each have12 cells and141 request/text/native-count comparisons; BATCH code C1/C2/C4/C8=76.97/119.64/214.25/347.99 tok/s. BATCH C8 beats TP overlap304.48 but loses PP8 539.36. Eight concurrent requests are not an all-eight-owner STEP logits/state or mixed-rollback proof. PP5 depth1 remains a preserved rejection; campaign complete=false with independently valid subsets. No active S axis selects the pilot; original balanced-exactness multiple/decode remains the inherited candidate scope (RND_O.md 5 Nov01:25,01:42). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_pipeline_overlap"></a>

#### `--tp_pipeline_overlap`

**Type/range:** unsigned32 decimal scalar; parser 0..1; semantic values: 0..1. **Default:** 0. **Commands:** generate, serve. **Placement:** original TP4_PRO, native PRO tail, pipeline depth>=2.

**Mechanics.** Run the CMP prefix and native PRO tail on disjoint workers.

**Use and interactions.** Nonzero requires TP4_PRO, pipeline_steps>=2, no BATCH, exact numerics, native PRO, precision/reference linear and CMP-reference, FP32 wire, code_wire0 and attention_owner0. Prepared overlap topology refuses images, mirrors, backward transfers and expert pauses. Source availability does not enable an axis or qualify an untested join (S main.c:1227; step_pipeline.c:682). Admission: G0; T.

**Precision class:** same bits in recorded current-recipe scope; joins unqualified. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. Shipped resolver writes: `original_tp_serial2`: `0` (all/generate); `original_pp_depth8`: `0` (all/generate); `original_pp_depth5`: `0` (all/generate). The `original_reference_serial_m1` exact strategy additionally inserts `0` for generate and serve, within the one-session capacity-512 profile in section 9.7.

**Measured effect.** Merged default0. Inherited XO06 preservation and XO09 source434a895/rndo_05 gxo10/11 serving pass:12 cells and141 request/text/native-count pairs per window. TP C2-C8 improves41-47%; code C1/C2/C4/C8 serial220.75/201.47/206.24/207.30 versus overlap220.50/284.58/295.79/304.48 tok/s. Best measured PP5 remains faster at C2-C8. Prefix_reuse1 warm decode scope, not fresh S prefix_reuse0, all-step logits/state, mixed or TTFT qualification. No active S axis selects nonzero (RND_O.md 5 Nov01:19). Source: `source/main.c` and shipped knob catalog.


### 8.15. Axes and branch feature boundaries

The four axes are main selectors with explicit holds; they are not branch-only. Other entries in this subsection are still branch, not in main yet: copy32, snapshotv5, TP5/M25 and the pending O query/dense-shard experiments. A branch refusal is not a branch boot proof. Source availability, numerical qualification, capacity and automatic tuple admission remain separate. Stable1 and session_fork are now documented in their main cache/prefill subsystems, default0, without automatic axes or an official-reference claim.

<a id="knob-category"></a>

#### `--category`

**Type/range:** text/enum; code / prose / structured / vision / balanced. **Default:** unset; no axis default selected **Commands:** generate, serve. **Placement:** tuple table: original / EXL3; explicit placement selects family, omission selects original; tuple chooses placement and explicit placement wins.

**Mechanics.** Choose the workload objective; balanced requires the full mixed-user benchmark.

**Use and interactions.** Explicit low-level knobs and existing aliases override inserted settings in either order. Duplicate axis selection refuses. No named presets or aliases exist. Partial tuples apply only a common plan without filling unset axes. Admission: No axes: CLI unchanged. Held dependencies and unsupported partial tuples refuse; ordinary startup/model gates remain authoritative..

**Precision class:** axis-dependent; see help axes. **Four-axis mapping:** Selector itself; all values and partial-resolution rules are in section7. No axis is selected by default. This selector is parsed on main; held tuples and incompatible partial selections refuse.

**Measured effect.** tools/cli/axes.json -> source/cli_axes_data.h / tools/cli/AXES.md; startup_axes.c; tests/startup_axes_test.c. S04 gxf11/gxf12 and completed S05 gxf_s05_default1024 are scoped measurements. Corrected S06 THIN/numerical and S07 original windows have separate pending gates; each strategy identifies its source epoch. Source: `source/main.c` and shipped knob catalog.


<a id="knob-exactness"></a>

#### `--exactness`

**Type/range:** text/enum; exact / balanced / fast. **Default:** unset; no axis default selected **Commands:** generate, serve. **Placement:** tuple table: original / EXL3; explicit placement selects family, omission selects original; tuple chooses placement and explicit placement wins.

**Mechanics.** Choose original-weight serial M1 reference arithmetic, current reproducible arithmetic, or qualified bounded-loss arithmetic. Original exact is now admitted only within section 9.7; EXL3 exact refuses.

**Use and interactions.** Explicit low-level knobs and existing aliases override inserted settings in either order. Duplicate axis selection refuses. No named presets or aliases exist. Partial tuples apply only a common plan without filling unset axes. Admission: No axes: CLI unchanged. Held dependencies and unsupported partial tuples refuse; ordinary startup/model gates remain authoritative..

**Precision class:** axis-dependent; see help axes. **Four-axis mapping:** Selector itself; all values and partial-resolution rules are in section7. No axis is selected by default. This selector is parsed on main; held tuples and incompatible partial selections refuse.

**Measured effect.** tools/cli/axes.json -> source/cli_axes_data.h / tools/cli/AXES.md; startup_axes.c; tests/startup_axes_test.c. S04 gxf11/gxf12 and completed S05 gxf_s05_default1024 are scoped measurements. Corrected S06 THIN/numerical and S07 original windows have separate pending gates; each strategy identifies its source epoch. Source: `source/main.c` and shipped knob catalog.



<a id="knob-concurrency"></a>

#### `--concurrency`

**Type/range:** text/enum; single / multiple / balanced. **Default:** unset; no axis default selected **Commands:** generate, serve. **Placement:** tuple table: original / EXL3; explicit placement selects family, omission selects original; tuple chooses placement and explicit placement wins.

**Mechanics.** Choose a C1 recipe, a C2-C8 recipe, or a measured compromise.

**Use and interactions.** Explicit low-level knobs and existing aliases override inserted settings in either order. Duplicate axis selection refuses. No named presets or aliases exist. Partial tuples apply only a common plan without filling unset axes. Admission: No axes: CLI unchanged. Held dependencies and unsupported partial tuples refuse; ordinary startup/model gates remain authoritative..

**Precision class:** axis-dependent; see help axes. **Four-axis mapping:** Selector itself; all values and partial-resolution rules are in section7. No axis is selected by default. This selector is parsed on main; held tuples and incompatible partial selections refuse.

**Measured effect.** tools/cli/axes.json -> source/cli_axes_data.h / tools/cli/AXES.md; startup_axes.c; tests/startup_axes_test.c. S04 gxf11/gxf12 and completed S05 gxf_s05_default1024 are scoped measurements. Corrected S06 THIN/numerical and S07 original windows have separate pending gates; each strategy identifies its source epoch. Source: `source/main.c` and shipped knob catalog.


<a id="knob-focus"></a>

#### `--focus`

**Type/range:** text/enum; balanced / ttft / decode / prefill. **Default:** unset; no axis default selected **Commands:** generate, serve. **Placement:** tuple table: original / EXL3; explicit placement selects family, omission selects original; tuple chooses placement and explicit placement wins.

**Mechanics.** Choose the optimization objective within the selected numerical tier.

**Use and interactions.** Explicit low-level knobs and existing aliases override inserted settings in either order. Duplicate axis selection refuses. No named presets or aliases exist. Partial tuples apply only a common plan without filling unset axes. Admission: No axes: CLI unchanged. Held dependencies and unsupported partial tuples refuse; ordinary startup/model gates remain authoritative..

**Precision class:** axis-dependent; see help axes. **Four-axis mapping:** Selector itself; all values and partial-resolution rules are in section7. No axis is selected by default. This selector is parsed on main; held tuples and incompatible partial selections refuse.

**Measured effect.** tools/cli/axes.json -> source/cli_axes_data.h / tools/cli/AXES.md; startup_axes.c; tests/startup_axes_test.c. S04 gxf11/gxf12 and completed S05 gxf_s05_default1024 are scoped measurements. Corrected S06 THIN/numerical and S07 original windows have separate pending gates; each strategy identifies its source epoch. Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp5_expert_csf"></a>

#### `--tp5_expert_csf` — branch, not in main yet

**Type/range:** text/enum; 0 or1. **Default:** 0 **Commands:** load, generate, serve (pending O; unavailable in S). **Placement:** engine_rnd_o_tp5_csf78 / guarded engine_rnd_o_tp5_join only; experimental original TP5.

**Mechanics.** Load losslessly encoded full448 target expert scales from an A20 sidecar.

**Use and interactions.** Original standalone CSF branch; latest guarded engine_rnd_o_tp5_join/11777a6 also applies: TP5 original raw448/512 and all native-PRO/one-session/flight/THIN/reference-linear gates. Nonzero iff tp5_expert_sidecar is supplied; absolute path required, target layers must be complete40 (or default0 resolving40). Full coverage, metadata, code/scale/source-pack and bounds identities are checked before loading. No partial sidecar or fallback. Guarded join11777a6 supports the individual paths but refuses CSF+third together before sidecar I/O or stage planning. Source availability/host factories do not qualify device fit. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** same bits intended for storage-only codec; runtime unqualified. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Historical source25/8ba73bde full448 program factory FAILED before device load. Corrected27 repairs W1/W3/W2 engine ordering from W1/W2/W3 storage without changing sidecar ordinals. Guarded join794acd4+style11777a6 passes16release/debug host/factory invocations,151actual-parser controls,all90CUDA images literal27 and zero hooks in8host executables/3newCSF modules. Full448 sidecar274651646720bytes/230400entries has source byte/word proof. Actual five-rank device load/numerics/ring/physical fit/C1 speed remain UNMEASURED. Default0; CSF+third composition explicitly refuses before I/O/planning. No active axis, only held original/balanced/single/decode dependency (RND_O.md5Nov14:03,14:20). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp5_expert_sidecar"></a>

#### `--tp5_expert_sidecar` — branch, not in main yet

**Type/range:** text/enum; absolute path. **Default:** none **Commands:** load, generate, serve (pending O; unavailable in S). **Placement:** engine_rnd_o_tp5_csf78 / guarded engine_rnd_o_tp5_join only; experimental original TP5.

**Mechanics.** Select the full448 A20 expert sidecar for the explicit CSF target codec.

**Use and interactions.** engine_rnd_o_tp5_csf78 only. Required iff tp5_expert_csf1; refuses relative paths, non-TP5, partial target-layer coverage or inconsistent source/layout/metadata. Does not replace the positional original pack or authorize other experimental joins. Guarded join11777a6 supports the individual paths but refuses CSF+third together before sidecar I/O or stage planning. Source availability/host factories do not qualify device fit. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** same bits intended for storage-only codec; runtime unqualified. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Absolute full448 A20 path remains paired iff tp5_expert_csf1; guarded join794acd4/11777a6 explicitly refuses CSF plus raw384/768 third-share before sidecar I/O/stage planning. Full448 storage/source proof and corrected host factory pass; historical source25factory failure retained. No device/load/fault/lifetime/physical-fit or isolated path-selection performance claim. Defaults absent/codec0, no active axis (RND_O.md5Nov14:20; conditional MERGE_REQUEST_v3.md). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp5_pro_third"></a>

#### `--tp5_pro_third` — branch, not in main yet

**Type/range:** text/enum; 0 or1. **Default:** 0 **Commands:** load, generate, serve (pending O; unavailable in S). **Placement:** engine_rnd_o_tp5_third78 / guarded engine_rnd_o_tp5_join only; experimental original TP5.

**Mechanics.** Use raw384/768 TP5 geometry with target head and enabled drafts on a CMP.

**Use and interactions.** engine_rnd_o_tp5_third78/f388b980 only; nonzero TP5 and exact physical card order0,1,3,4,2. All TP5 original/raw/native-PRO/reference-linear/one-session/flight/THIN/capacity gates remain. Target head and any enabled drafts move to logicalrank3/physicalGPU4/class80; target PRO remains native. Mutually separate from the CSF448 branch, with no inferred composability. Guarded join11777a6 supports the individual paths but refuses CSF+third together before sidecar I/O or stage planning. Source availability/host factories do not qualify device fit. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** order-changing; new placement arithmetic unqualified. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Guarded join794acd4+style11777a6 passes16release/debug host/factory invocations,151actual-parser cases including127old vectors+24join controls,90literalCUDA images and zero hooks in8release host executables/3newCSF modules. Actual raw384/768 allocation plans pass; no CUDA physical-fit/device/head/draft/ring/generation/rollback/speed proof. CSF+third composition explicitly refuses before I/O/planning. Default0 retains raw448/512/PRO head; no active axis, only held original/balanced/single/decode candidate. Earlier26/f388b980 host proof remains historical (RND_O.md5Nov14:03,14:20). Source: `source/main.c` and shipped knob catalog.


<a id="knob-session_snapshot_format"></a>

#### `--session_snapshot_format` — branch, not in main yet

**Type/range:** text/enum; 4 or5. **Default:** 4 **Commands:** serve (pending P; unavailable in S). **Placement:** engine_rnd_p only; see source.

**Mechanics.** Choose the legacy snapshot wire format or the checkpoint-bearing v5 format.

**Use and interactions.** Branch-only P full prototype f95615f/18. Format5 requires prefill_prefix_stable1 and prefix_reuse_exact2. Stable snapshots require format5; stable/namespace/hard-owner/opened-FD gates remain. Version5 uses a separate header/subdirectory, strict payload and checkpoint layout identity, drains actual LOAD copies and publishes only after ordinary plus optional checkpoint completion. Imported stable provenance is recognized only after verified LOAD. Legacy4 remains the default. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** same bits for exact-copy service policy; target-form qualification required. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Branch-only P23/b12744b plus test-only80f23b9, default4. Actual format5 eviction/restart LOADs retain25000/25016tokens and260771840/260817920bytes; native/visible output equals cold, HTTP TTFT311.510/282.318ms versus7246.065cold. Actual8192-generated source serializesG8272/255604736payloadbytes. Overall xc38v4 exit1: anonymous child correctly cold-misses under Q hard-owner rule; harness wrongly expected a fork hit. Full checkpoint restart/fork and authenticated import proof remain UNQUALIFIED; prepared40v2 not queued under pause. No namespace or product gate weakened, no automatic axis (RND_P.md5Nov13:16,13:28; results/gxc38v4snapshot/partial_read.json qualified=false). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp5_host_channels"></a>

#### `--tp5_host_channels` — branch, not in main yet

**Type/range:** text/enum; 0 or1. **Default:** 0 **Commands:** load, generate, serve (pending O; unavailable in S). **Placement:** engine_rnd_o_tp5_main78 only; experimental original TP5.

**Mechanics.** Exchange TP5 messages through explicitly mapped pinned-host inboxes.

**Use and interactions.** Branch-only TP5. Nonzero refuses every other placement; TP5 requires original raw448/512, native PRO, relaxed numerics, reference precision/order, one session/flight, THIN rows, wide_rows0, context<=65536 and no composed O paths. Explicit mapped-host allocation/backend qualification is required; no automatic fallback. This branch uses all five devices, including the PRO, without changing its device settings. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** same bits intended; composed device proof pending. **Four-axis mapping:** Candidate original text/category; exactness=balanced; concurrency=single; focus=decode diagnostic. No active tuple; exact/fast/vision/multiple/TTFT held.

**Measured effect.** Current main78 candidate aab1daf (engine_rnd_o_tp5_main78/source/main.c:906,1413,2946), following 5166a00. Source/CPU-only, no GPU transport identity, physical fit, serving speed or official-exact proof. Candidate original text/balanced/single/decode diagnostic only; no active axis. Direct0 retained. Source21/23 has12 focused checks,54 current-main parser controls and unchanged87 CUDA images (RND_O.md5Nov09:34,10:17). Source: `source/main.c` and shipped knob catalog. Later branch details retain their named RND epoch.


<a id="knob-tp5_original_draft"></a>

#### `--tp5_original_draft` — branch, not in main yet

**Type/range:** text/enum; 0 or1. **Default:** 0 **Commands:** load, generate, serve (pending O; unavailable in S). **Placement:** engine_rnd_o_tp5_main78 only; experimental original TP5.

**Mechanics.** Own the unsharded original DSpark draft tail on native PRO in TP5.

**Use and interactions.** Branch-only TP5; all TP5 geometry/allocation gates apply. Nonzero requires speculative2, block5, positions5, rows8, fixed policy0, learned/session acceptance0. TP5 admits original text, one session/flight, wide_rows0, context<=65536, native PRO, precision/reference linear, relaxed numerics, no official HC or composed O paths. Target-only0 requires speculation off. No fallback to a different placement. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** draft-only intended; target-bit and rollback proof pending. **Four-axis mapping:** Category: all supported workloads unless the named path restricts it. Exactness: the entry's precision class and source gates apply. Concurrency: only the prepared THIN/BATCH/placement forms named here. Focus: target product/transport/runtime cost; no isolated objective gain implied. The shipped resolver does not insert this control; it remains explicit/inherited, with no automatic axis mapping.

**Measured effect.** Source d9388b7/21 and main78 aab1daf/23 (main.c:908,1522,2946) have CPU construction/parser proof only; current54-case parser suite fixes guard ordering after placement defaults. All40 64K DSpark arena plan is77.335GB PRO and60.581/59.257/59.257/58.669GB CMP plus optional918528000 host bytes: an allocation plan, not physical-fit proof. No generation/rollback/speed/official equality, no automatic axis; held original text/balanced/single/decode candidate (RND_O.md5Nov09:34,10:17). Source: `source/main.c` and shipped knob catalog.


<a id="knob-tp_core_queries"></a>

#### `--tp_core_queries` — branch, not in main yet

**Type/range:** text/enum; 0 or1. **Default:** 0 **Commands:** load, generate, serve (pending N; unavailable in S). **Placement:** archived M25 frozen32 only; removed from live pure37 and absent from S.

**Mechanics.** Fuse eligible true-THIN target query projection into the SCORE load.

**Use and interactions.** Archived M25 dfd803f/frozen32. CLI nonzero requires TP4_EXL3, ordered BF16 wire, code_wire0, attention_owner0, reference linear order, precision linear profile and pro_emulation0. Effective only on SM80 true THIN target non-index SCORE, <=8rows and same-call/alias/overflow eligibility. BATCH, prefill, draft, index, WIDE/FULL and native120 retain baseline paths; these are fallbacks rather than additional CLI refusals. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** same bits intended; safety/reference campaign incomplete. **Four-axis mapping:** Candidate category=code; exactness=balanced relative to the pinned Anemone family; concurrency=single true-THIN; focus=decode. All other mappings held; no active tuple.

**Measured effect.** Historical M25 prototype dfd803f/frozen32 only; removed from isolated main78 N8dd9dfc/frozen36. Default0/NO MERGE. xd17b passes memcheck/racecheck/synccheck (eight cases each, zero reported errors), then first required fault control fails: generic CUDA launch failure does not prove the intended overflow guard. Later controls, OFF/ON references, native ABA, HTTP and speed remain unrun. Prior loader/LSan failures remain failures. No active axis; proposed code/balanced/single/decode stays held (RND_N.md5Nov12:56,13:02,13:29; M25_VERDICT_v4.md). Source: `source/main.c` and shipped knob catalog. Later branch details retain their named RND epoch.


<a id="knob-original_queries_piece"></a>

#### `--original_queries_piece` — branch, not in main yet

**Type/range:** text/enum; 0 or1. **Default:** 0 **Commands:** load, generate, serve (pending O; unavailable in S). **Placement:** engine_rnd_o only; see source.

**Mechanics.** Prepare original query projection pieces for TP4 exact attention.

**Use and interactions.** Original exact TP4_PRO, precision/reference/CMP-reference, native PRO, FP32 wire, wide64..1024; no BATCH, wide-expert, code/owner/split-overlap experiments. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** same bits in recorded current-recipe fixtures; joins unqualified. **Four-axis mapping:** Held original-weight balanced-exactness prefill/TTFT experiment; no S tuple sets a nonzero value. Reference/exact claims require separate proof.

**Measured effect.** Source74b24a5 completes three Fraser/2K/30K knob0/1 pairs and three XO05 baseline controls, including selected records, prompt IDs, digests, canonical SAVE and active last-target full-vocabulary F32 bits; power restoration passes. CLI prompt+first changes -0.09%/-0.06%/+0.16%; no standalone speed win, HTTP TTFT or serving/join qualification. Default0 retained for later composition. Held original balanced-exactness prefill/ttft candidate; no S axis enables it (RND_O.md 5 Nov 02:42). Source: `source/main.c` and shipped knob catalog.


<a id="knob-original_dense_shards"></a>

#### `--original_dense_shards` — branch, not in main yet

**Type/range:** text/enum; 0 or1. **Default:** 0 **Commands:** load, generate, serve (pending O; unavailable in S). **Placement:** engine_rnd_o only; see source.

**Mechanics.** Shard original full-K attention dense output products and gather losslessly.

**Use and interactions.** Original exact TP4_PRO, precision/reference/CMP-reference, native PRO, FP32 wire; excludes other O pilots, split/owner/code paths, dead-half/tail variants and alternate products. EXL3 admission unchanged. Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** same bits intended; device-unqualified. **Four-axis mapping:** Held original-weight balanced-exactness prefill/TTFT experiment; no S tuple sets a nonzero value. Reference/exact claims require separate proof.

**Measured effect.** O1cb6928/ff91cfe host preparation remains separate from device proof. XO18 dense starts0 of9 planned arms: 197 work seconds remained against a 900-second floor. This is a budget deferral, not a device or numerical failure. Default0; GPU identity/speed and any original balanced-exactness prefill/ttft mapping remain held; no S axis enables it (RND_O.md 5 Nov 02:42). Source: `source/main.c` and shipped knob catalog.


<a id="knob-prefill_small_copy_rows"></a>

#### `--prefill_small_copy_rows` — branch, not in main yet

**Type/range:** text/enum; parser0..32; admitted0 or32. **Default:** 0 **Commands:** generate, serve (pending P; unavailable in S). **Placement:** engine_rnd_p only; see source.

**Mechanics.** Prepare an extra FULL program with bounded padded prefill copies.

**Use and interactions.** Nonzero requires prefill_prefix_stable1 and every stable1 gate. Additional immutable FULL retains largest physical wide capacity; only padded headless steps with actual rows<=32 select it. Attention-quarter and FFN owner-segment CE copies stop after32program rows; physical offsets, math/owners/reduction and READY/DONE/empty handshakes remain. Ordinary steps exclude this alternative; preparation rejects non-FULL/non-EXL3/non-BF16-CE5 or invalid eight-row bounds (P main.c:908,1380,1980; step_image_programs.c:172; model_v41_program.c:5701). Admission: Not parsed by this S base; requires owning track merge.

**Precision class:** same bits within qualified64K stable1; strict cold/1M admission held. **Four-axis mapping:** Candidate TP4_EXL3 text cache form in exactness=exact, conditional on its cold/state/capacity gates; no automatic category/concurrency/focus selection. Copy32 targets prefill/TTFT copying only after device proof.

**Measured effect.** Frozen18 xc38v4 full306 native/observable PASS:102cold+204warm, all accepted IDs/selected-logit bits equal own cold;102/102 cold controls equal accepted form14. Short append warm ABA362.311/214.198/362.160ms, ratio0.59301(95%0.58865-0.59810); phase345.339/199.033/344.657ms, ratio0.57608(0.56739-0.58491); tool-cycle warm ratio0.99616(0.91396-1.00434). Strict cold gate FAILS: short ratio1.00116886(1.00069573-1.00243898); tool1.00172992,phase0.92088957. Sampling intervals do not bound hardware drift. 1M readiness fails before token0. Manual/default0/HOLD; no active axes, default or legacy-speed parity. Full window exit1 remains separate from completed306 leaves (RND_P.md5Nov13:10,13:28; results/gxc38v4/small_copy_report.json). Source: `source/main.c` and shipped knob catalog.


### 8.16. Environment variables, debugging and sanitizers

#### Environment names renamed on 6 Nov

| Old name | New name |
| --- | --- |
| `ANEMONE_DIAG_DRAFT_PROBE` | `DIAG_DRAFT_PROBE` |
| `ANEMONE_DIAG_DRAFT_SKIP_FFN` | `DIAG_DRAFT_SKIP_FFN` |
| `ANEMONE_DIAG_DRAFT_STAGES` | `DIAG_DRAFT_STAGES` |
| `ANEMONE_DIAG_STATE_DIGESTS` | `DIAG_STATE_DIGESTS` |
| `ANEMONE_SOCKET_DESCRIPTOR` | `INFERENCE_SOCKET_FD` |
| `ANEMONE_TRANSPORT_CHECK_ONLY` | `DEVICE_TRANSPORT_CHECK` |

Only the old-name column is retained for migration lookup. Headings, assignments, receipt labels, anchors and index targets use the new names. `ANEMONE_API_KEY` remains a client-only placeholder; public API names, binary/file names, CLI flags and format magic were not renamed. The transport/socket names belong to the standalone measurement worker rather than normal serve configuration.


The production source has exactly four literal `getenv` names. They are diagnostic interfaces; there is no environment-based alternative spelling for every CLI knob. The scan receipt is getenv_locations.txt. Presence tests count even an empty value or string0 as enabled. No engine default reads a user key from an environment variable; `ANEMONE_API_KEY` in this manual belongs to HTTP clients.

<a id="knob-diag_state_digests"></a>

#### `DIAG_STATE_DIGESTS`

**Type/range/default:** presence; absent by default. **Mechanics/use:** At generate shutdown, compute and emit digests of state tensors after generated work. Generate only; serving does not enable this shutdown path. **Precision and axes:** Same-bits observation; diagnostic overhead unmeasured. Category/exactness/concurrency/focus orthogonal, no automatic axis setting. **Measured effect:** no isolated timing or whole-workload gain is established. Source: `source/main.c`, `source/model_v41_program.c`.


```sh
DIAG_STATE_DIGESTS=1 anemone generate --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=2048 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 messages.json tokenizer.bin model.exl3map
```

Check: `env_DIAG_STATE_DIGESTS` — environment supplied and argv parsed; diagnostic computation was not executed. Receipt: examples.json.

<a id="knob-diag_draft_stages"></a>

#### `DIAG_DRAFT_STAGES`

**Type/range/default:** strtoul decimal then uint32 cast; absent means3 stages. **Mechanics/use:** Limit the draft stage loop to the smaller of the parsed count and the three model draft stages.0 executes no draft stage. This is not the strict CLI number parser; malformed strings and overflow do not gain normal CLI validation. **Precision and axes:** Changes draft computation; target-bit invariance across resulting verification forms is required. Raw diagnostic only, no S tuple. **Measured effect:** no isolated timing or whole-workload gain is established. Source: `source/main.c`, `source/model_v41_program.c`.


```sh
DIAG_DRAFT_STAGES=1 anemone generate --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=2048 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 messages.json tokenizer.bin model.exl3map
```

Check: `env_DIAG_DRAFT_STAGES` — environment supplied and argv parsed; diagnostic computation was not executed. Receipt: examples.json.

<a id="knob-diag_draft_skip_ffn"></a>

#### `DIAG_DRAFT_SKIP_FFN`

**Type/range/default:** exact string1 enables; absent/other strings disable. **Mechanics/use:** Skip FFN sites in the draft stage probe. This deliberately changes draft computation and is not a production speed recipe. **Precision and axes:** Draft-only conditional/order-changing if target verification forms change. No axis admission. **Measured effect:** no isolated timing or whole-workload gain is established. Source: `source/main.c`, `source/model_v41_program.c`.


```sh
DIAG_DRAFT_SKIP_FFN=1 anemone generate --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=2048 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 messages.json tokenizer.bin model.exl3map
```

Check: `env_DIAG_DRAFT_SKIP_FFN` — environment supplied and argv parsed; diagnostic computation was not executed. Receipt: examples.json.

<a id="knob-diag_draft_probe"></a>

#### `DIAG_DRAFT_PROBE`

**Type/range/default:** presence; absent by default. **Mechanics/use:** With the required batch-join kernel, expose stage0 intermediate query/attention/rotation streams in the logits probe layout and omit normal head/following work. It can affect preparation of the joined draft path. Output is diagnostic, not valid ordinary draft logits. **Precision and axes:** Draft diagnostic with altered semantics; no normal exactness or performance claim and no automatic axis mapping. **Measured effect:** no isolated timing or whole-workload gain is established. Source: `source/main.c`, `source/model_v41_program.c`.


```sh
DIAG_DRAFT_PROBE=1 anemone generate --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=2048 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 messages.json tokenizer.bin model.exl3map
```

Check: `env_DIAG_DRAFT_PROBE` — environment supplied and argv parsed; diagnostic computation was not executed. Receipt: examples.json.

#### Test and measurement programs are separate executables

The repository also has `getenv` calls in host/card tests and standalone measurement tools. They are not `anemone serve` knobs, do not become production settings, and must not be used to bypass GPU allocation. The full C-source scan is retained in all_environment_uses.txt. Test/probe variables include the following; their accepted values/defaults are scoped to the named executable, not the main parser.

| Variable or family | Executable scope / behavior |
| --- | --- |
| `TP_TEST_EXACT`, `TP_TEST_RAW` | TP program tests; presence selects comparison modes. |
| `RING_DUMP_DIRECTORY`, `RING_BANK_AUDIT` | Ring dump/audit tests; path or presence gate. |
| `PROGRAM_HISTOGRAM`, `BATCH_DRIVER_DEBUG` | EXL3 program tests; histogram/debug observation. |
| `VISION_TEST_ALIGNER_BIAS_LAST` | Vision reference helper; explicitly selected aligner ordering. |
| `VISION_TRACE_DIRECTORY`, `VISION_TRACE_REFERENCE_DIRECTORY` | Vision test trace destinations/reference input. |
| `UNDER_GPU_RUN` | Card-test authorization guard, not a main placement selector. |
| `TMPDIR` | Teacher-forced test temporary directory; fallback `/tmp`. |
| `DEVICE_TRANSPORT_CHECK`, `INFERENCE_SOCKET_FD` | Standalone transport measurement worker: check-only presence and supplied socket descriptor. |
| `CROSS_CLASS_MMA_ONLY`, `CROSS_CLASS_KERNEL` | Standalone cross-class arithmetic measurement: restrict mode or select kernel name. |
| `MINIMUM_EXPONENT` | Cross-class model-fit tool; absent sentinel -100000. |

Dynamic environment names in those test helpers are developer fixtures, not a supported server environment API. Standard loader/driver/sanitizer variables are interpreted by those external runtimes. This manual does not promise that `CUDA_VISIBLE_DEVICES` replaces the engine's backend card-order contract. Sanitizer build configuration lives in the Makefile; sanitizer option variables do not alter numerical admission. Only the separately recorded host tests were executed; no card-test/tool invocation or GPU job was run.

## 9. Complete recipes and measured tradeoffs

### 9.1 C1 code: fixed-depth EXL3 THIN

Use the TP4_EXL3 quick start for the canonical one-slot fixed-five recipe. Its concrete target-head strategy matters: `tp_head_shards=1`, `tp_head_stream=1`, `tp_draft_shards=1`, `tp_draft_layers=0`. The S06 short window at four CMPs/180 W and 65,536 context measured code **231.0 tok/s**, prose **159.0**, structured **330.1** per stream; cold-prefill median **3,909 tok/s**, Fraser C1 median/worst TTFT **1,379.4 / 1,395.1 ms**, and 15/15 native/text/count/finish matches. These are the pinned S06 joint recipe's measurements, not a tcb80 rerun or a per-knob gain. Pinned Track S thin_fixed evidence.


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=2048 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=1 --prefix_reuse=0 tokenizer.bin model.exl3map
```

Check: `recipe_c1_code` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


Changing only fixed policy 0 to cost policy 1 in the controlled S06 comparison retained eight token IDs but changed six selected F32 logits, maximum about `7.63e-5`. Matching decoded text alone therefore does not establish target-bit invariance. The current head flags were held equal in that pair; the observation does not isolate head sharding as the cause. The omitted-policy tcb78 default is described separately in the draft entries.

### 9.2 C8 mixed serving: BATCH with bounded decode opportunities

This reproducible baseline prepares eight BATCH forms and eight slots, uses one pipeline step, disables legacy prefix reuse, and gives ready decoders one round between 1,024-row prompt chunks. It is a starting recipe for mixed traffic, not a qualification of the held all-balanced axis tuple. Fixed policy 0 is explicit. To evaluate the **configured-concurrency default inherited by tcb80**, remove the policy and price overrides as a separate experiment; explicitly writing policy 2 does not trigger the same automatic menu insertion.


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=1 --step_graphs=1 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_learned_acceptance=1 --draft_base_us=13000 --draft_tail_us=0 --draft_row_us=1200 --tp_split_least_rows=512 --card_memory_kept_free_mib=0 --batch_rows=8,8,12,12,16,24,32,48 --batch_verifiers=1,2,2,4,4,4,8,8 --batch_wide_head=1 --batch_draft_rows=1 --draft_policy=0 --batch_core_once=3 --draft_session_acceptance=0 --batch_plan_ties=0 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=8 --prefill_first=1 --prefix_reuse=0 --prefill_decode_rows=1024 --prefill_decode_rounds=1 --prefill_decode_waiters=0 --prefill_order=2 tokenizer.bin model.exl3map
```

Check: `recipe_c8_mixed` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


The following command exercises the actual tcb80 omitted-policy default on the canonical eight-slot BATCH geometry. It omits policy, learning, prior, core and menu overrides. The parser resolves rate2, learn1,14300/0/690us, core3 and the eight measured prices. Its receipt stores all resolved arrays; this is the command to compare when checking the newly shipped default.


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=1 --step_graphs=1 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=1 --speculative=2 --draft_block=5 --draft_positions=5 --tp_split_least_rows=512 --card_memory_kept_free_mib=0 --batch_rows=8,8,12,12,16,24,32,48 --batch_verifiers=1,2,2,4,4,4,8,8 --batch_wide_head=1 --batch_draft_rows=1 --draft_session_acceptance=0 --batch_plan_ties=0 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=8 --prefill_first=1 --prefix_reuse=0 --prefill_decode_rows=1024 --prefill_decode_rounds=1 --prefill_decode_waiters=0 --prefill_order=2 tokenizer.bin model.exl3map
```

Check: `recipe_tcb80_default` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


S04's older core2 BATCH recipe measured structured aggregate C1/C2/C4/C8 **335.3 / 451.5 / 612.8 / 784.6 tok/s**, code **265.3 / 295.8 / 376.4 / 468.1**, prose **163.6 / 217.5 / 290.6 / 381.8**; 141 comparisons matched. The core3 command above is a later strategy, so those numbers remain historical context. N's scoped code fixed0/menu0/core3 source29 medians were **265.799 / 296.778 / 377.579 / 470.281 tok/s**; the default-policy main integration has its own evidence. A joint recipe change is never an isolated speed claim for core3. Track S batch_fixed, RND_N.md.

### 9.3 Long context and 1M

`--context_tokens=1048576` allocates capacity for **prompt plus generation per slot**. It does not mean every step processes one million live tokens, and it does not make eight slots fit. Use the following one-slot THIN form as a syntax-checked capacity recipe:


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=2 --step_graphs=1 --speculative=2 --draft_block=5 --draft_positions=5 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --card_memory_kept_free_mib=0 --draft_policy=0 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=0 --draft_learned_acceptance=0 --draft_session_acceptance=0 --batch_plan_ties=0 --draft_base_us=12700 --draft_tail_us=2200 --draft_row_us=2700 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=1 --prefix_reuse=0 --context_tokens=1048576 --wide_rows=2048 tokenizer.bin model.exl3map
```

Check: `recipe_1m_thin` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


For BATCH at 1M, retain 1,024-row wide programs, tail128 and no fine ladder; the exact parser defaults already choose that family of tail settings. The recipe deliberately uses **one active slot** because the historical one-slot fit is not an eight-slot fit:


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=1 --step_graphs=1 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_learned_acceptance=1 --draft_base_us=13000 --draft_tail_us=0 --draft_row_us=1200 --tp_split_least_rows=512 --card_memory_kept_free_mib=0 --batch_rows=8,8,12,12,16,24,32,48 --batch_verifiers=1,2,2,4,4,4,8,8 --batch_wide_head=1 --batch_draft_rows=1 --draft_policy=0 --batch_core_once=3 --draft_session_acceptance=0 --batch_plan_ties=0 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --prefill_first=1 --prefix_reuse=0 --prefill_decode_rows=1024 --prefill_decode_rounds=1 --prefill_decode_waiters=0 --prefill_order=2 --context_tokens=1048576 --session_slots=1 tokenizer.bin model.exl3map
```

Check: `recipe_1m_batch` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


Earlier tcb74-equivalent one-slot boots left only **15 MiB THIN / 53 MiB BATCH** on the tight cards. Those are fragile, source-scoped allocation observations, not available-memory promises for this build, a different graph set, stable-prefix branch or a client workload. Do not extrapolate the 65,536-context rates to 1M. Two 1M slots, eight 1M slots and a larger program ladder require their own allocation and runtime evidence. COMPARISON_MATRIX.md P2, RND_PIPELINE.md.

### 9.4 Fraser-style latency and long-prefill pauses

This BATCH variant keeps 1,024-row chunks but permits four decoder rounds per interval. It trades prompt-first-token work and aggregate rate for faster completion of already-started short replies:


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=1 --step_graphs=1 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_learned_acceptance=1 --draft_base_us=13000 --draft_tail_us=0 --draft_row_us=1200 --tp_split_least_rows=512 --card_memory_kept_free_mib=0 --batch_rows=8,8,12,12,16,24,32,48 --batch_verifiers=1,2,2,4,4,4,8,8 --batch_wide_head=1 --batch_draft_rows=1 --draft_policy=0 --batch_core_once=3 --draft_session_acceptance=0 --batch_plan_ties=0 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=8 --prefill_first=1 --prefix_reuse=0 --prefill_decode_rows=1024 --prefill_decode_waiters=0 --prefill_order=2 --prefill_decode_rounds=4 tokenizer.bin model.exl3map
```

Check: `recipe_fraser` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


The engine's own scheduling help reports the following four-CMP BATCH/order2/prefix-off workload: one 30K prompt plus four short streams, temperature zero, maximum 64 generated tokens; one short stream ends at EOS50. Completion is measured after first token; pause is the largest SSE content gap. Fraser replies are 2,048 tokens. These are measured pilots, not service-level bounds.

| Policy | Short median TTFT s | Short completion s | Largest pause s | Long TTFT s | Fraser C4/C8 median TTFT s | C4/C8 aggregate tok/s |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Old order2 | 3.75 | 11.53 | 5.45 | 14.29 | 3.94 / 7.13 | 321 / 384 |
| 512/1 | 4.10 | 5.70 | 0.29 | 16.16 | 4.40 / 8.38 | 312 / 358 |
| 1024/4 | 4.36 | 2.82 | 0.46 | 15.99 | 4.61 / 9.25 | 316 / 356 |
| 256/8 | 5.08 | 1.27 | 0.21 | 17.34 | 7.93 / 18.77 | 266 / 274 |
| Waiter512/4 | 3.92 | 3.84 | 0.47 | 15.32 | 4.08 / 7.61 | 316 / 368 |
| Default1024/1 | 3.92 | 7.77 | 0.48 | 15.38 | 4.10 / 7.64 | 322 / 373 |

Old and 512/1 have three Fraser batches; 1024/4 and 256/8 are one-batch pilots. Waiter512/4 and final1024/1 each have one Fraser batch per concurrency in the help's reported scope. Mixed texts equal solo controls. A smaller chunk cap and larger decode burst can reduce pauses while making new requests wait longer. Always retain the original final-tail plan. Captured native scheduling help.

### 9.5 Opt-in order3 for concurrent short residuals

Order3 extends shortest-first scheduling to let eligible short no-chunk residuals share a THIN turn in the measured row-independent EXL3 BATCH family. It preserves long-chunk/decode-due exclusions and order2 fairness. It is an explicit opt-in: qualified BATCH still defaults to2 and other raw routes retain0.


```sh
anemone serve --placement=tp4_exl3 --numerics=relaxed --cmp_order=reference --tp_wire_bf16=1 --rows=8 --wide_rows=1024 --full_steps=1 --prefill_cone=1 --chunks_in_flight=1 --pipeline_steps=1 --step_graphs=1 --exl3_tc_moe=1 --tp_head_mma=1 --tp_trunk_mma=1 --tp_ordered_wide=1 --tp_head_shards=1 --tp_head_stream=1 --tp_draft_shards=1 --tp_draft_layers=1 --speculative=2 --draft_block=5 --draft_positions=5 --draft_learned_acceptance=1 --draft_base_us=13000 --draft_tail_us=0 --draft_row_us=1200 --tp_split_least_rows=512 --card_memory_kept_free_mib=0 --batch_rows=8,8,12,12,16,24,32,48 --batch_verifiers=1,2,2,4,4,4,8,8 --batch_wide_head=1 --batch_draft_rows=1 --draft_policy=0 --batch_core_once=3 --draft_session_acceptance=0 --batch_plan_ties=0 --exl3_tc_product_blocks=0 --tp_decode_round_once=0 --official_hc_rows=0 --session_slots=8 --prefill_first=1 --prefix_reuse=0 --prefill_decode_rows=1024 --prefill_decode_rounds=1 --prefill_decode_waiters=0 --prefill_order=3 tokenizer.bin model.exl3map
```

Check: `order3` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.


Startup requires the bounded BATCH1024/BF16/column-owner/THIN-bit/exact-plan geometry. Combining3 with `prefix_reuse_exact=1|2`, `prefill_prefix_stable=1` or `session_fork=1` is refused even if another cache control is0. These joins lack the required cold-plan qualification. The refusal checks are in examples.json; defaults were not weakened to admit them. The shipped knob entry retains Track N's paired workload results with their own source epoch; this documentation refresh ran no GPU timing.

### 9.6 Original weights: TP4_PRO single and PP5 multiple

The original-weight quick starts are complete commands for both final-main routes: compatible TP4_PRO now resolves wide experts1, and the multi-session PP5 raw depth resolves5. The following S07 numbers are historical explicit slab/depth8 measurements, not measurements of those refreshed commands. An explicit depth8 remains available and is pinned by the original multiple-user decode axis tuple. Their fixed draft policy and numerical choices are pinned rather than inherited from EXL3. S07's TP4_PRO one-slot window measured code **207.49**, prose **150.90**, structured **288.00 tok/s** at 65,536 capacity, with selected8/256 and 15/15 request/text/count/finish agreement. PP5/depth8/eight slots measured aggregate C8 code **406.60**, prose **408.89**, structured **782.59 tok/s**, with 141/141 request comparisons and both selected-record references. Cold prefill, Fraser TTFT, long-prefill pauses and 1M work were not measured in those S07 windows. Pinned original strategy evidence.

Pipeline depth eight beat depth five in all twelve same-epoch homogeneous cells, but this does not imply every mixed workload or latency metric improved. Native PRO remains enabled. Depth5 is now the raw default for qualifying multi-session serve; the separately pinned axis still chooses8. The explicit PP5 layer boundaries are part of the numerical recipe; moving them changes which arithmetic executes those layers.

### 9.7 The exactness=exact tier: original-weight reference profile

`--exactness=exact` selects the ordinary engine's **original-weight PP THIN serial M1 reference arithmetic**. It uses one session and a **512-token capacity including prompt and completion**. The pinned placement maps logical cards 0–4 to backend ordinals `0,1,3,4,2`; the native PRO holds target layers 29–39 and the head. Prepared release kernels, scratch, arenas and graphs are created before serving. This is an opt-in profile; `--official_reference` defaults to 0 outside it, and balanced arithmetic is unchanged.

A normal messages prompt is rendered and processed one token at a time:

```sh
anemone generate --exactness=exact --generated_tokens=256 messages.json tokenizer.bin original.anemone
```

Check: `reference_generate` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

Every ordinary valid temperature, top_p and seed is accepted without changing model arithmetic. For production sampling, set **T=1 / top_p=1** explicitly:

```sh
anemone serve --exactness=exact --temperature=1 --top_p=1 tokenizer.bin original.anemone
```

Check: `reference_serve` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

The tcb82 parser retains the ordinary omitted-temperature default of 0; selecting the exact axis alone does not insert temperature=1. The explicit production command above avoids relying on a different sampler default. CLI temperature is finite and nonnegative; HTTP retains its ordinary 0–2 bound.

**Exact model logits are distinct from stochastic token identity.** The ordinary sampler consumes the official-exact F32 logit words; the profile adds no alternate rounding or sampler reduction. Drawn IDs match the pinned source on the admitted **T=0** fixtures. At T>0, Anemone's token/position hash, exponential race and existing top_p cutoff differ from the source's Torch exponential draws after softmax. The RNG and sampler order do not match, so stochastic token identity is not claimed. A valid T=1 run is not evidence of source-identical stochastic output.

The complete raw generate setting is available for inspecting the composite profile:

```sh
anemone generate --placement=pp5 --numerics=exact --cmp_order=reference --linear_profile=0 --linear_order=1 --pro_emulation=0 --official_hc_rows=1 --official_reference=1 --card_order=0,1,3,4,2 --rows=1 --context_tokens=512 --wide_rows=0 --full_steps=0 --chunks_in_flight=1 --pipeline_steps=1 --step_graphs=1 --speculative=0 --tp_pipeline_overlap=0 --original_chunk_core=0 --original_dense_staged=0 --original_wide_experts=0 --card_memory_kept_free_mib=0 '--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4' --temperature=1 --top_p=1 --generated_tokens=256 messages.json tokenizer.bin original.anemone
```

Check: `reference_generate_explicit` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

For the exact eight-ID source fixture, use the public prompt-token ingestion route instead of a messages filename. Tokenization and assistant-prefix insertion are bypassed; the file supplies prompt IDs only, never generated tokens or activations:

```sh
anemone generate --exactness=exact --temperature=0 --top_p=1 --seed=0 --generated_tokens=256 --ignore_eos=1 --prompt_token_ids=serial_m1_ids.json tokenizer.bin original.anemone
```

Check: `reference_source_ids` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

The public `serial_m1_ids.json` file contains:

```json
[0,128803,19417,566,345,11592,17437,16]
```

`--ignore_eos=0` is the ordinary default. The explicit value 1 above permits the fixed 256-draw comparison after the source EOS at draw 135. Normal serving retains EOS stopping. The ID array must be nonempty, contain only unsigned decimal vocabulary IDs, and leave at least one context token for generation; this route accepts exactly one weight source. Serve does not accept `--prompt_token_ids`.

The server profile also selects `--session_slots=1 --prefix_reuse=0 --prefix_reuse_exact=0 --session_fork=0 --prefill_order=0 --original_batch=0`. Incompatible arithmetic/placement overrides, speculation and batching refuse admission. Images are unsupported. **EXL3 refuses exact** with the scope label **reference arithmetic (EXL3: not official-exact)**:

```sh
anemone serve --placement=tp4_exl3 --exactness=exact tokenizer.bin model.exl3map
```

Check: `reference_exl3_refusal` — expected tcb82 refusal: axes category=unset exactness=exact concurrency=unset focus=unset weight_family=exl3 axes refused: reference arithmetic (EXL3: not official-exact) axes: invalid selection; see anemone help axes Receipt: examples.json.

Qualified scope and cost are inherited from Track R's `USAGE_REFERENCE_v4.md`, `COSTS.md`, `REPORT_v4.md` and final merge receipts. XR62 passes both autonomous 256-draw public-axis CLI/source-ID and selected-F32 comparisons, the 263-full-head prelude and all 40 block/carry endpoints at positions 0–7 (40,555,520 represented F32 values, zero differing bits). XR63 preserves the canonical default hashes at both widths. XR66 supersedes the earlier temperature refusal: all 16 T=0 HTTP IDs/text/native counts/finish reasons match the CLI; T=1 returns HTTP 200, completes 16 tokens and finishes with length. These are the recorded source/operator/state/fixture scopes; this docs refresh reran parser checks only.

The full-step C1 source fixture costs **124.392 ms exact versus 12.7745 ms PP5 control: 9.7375×, approximately 9.7×**. Four CMPs were at 180 W and the PRO was unchanged, with R builds/tests idle and empty compiler snapshots. The separate ordinary Fraser default-off/pristine-a4 medians were 13.000 ms each. C8 and new per-operation CMP shares were not measured. Native PRO packed FP8 is selected only for the measured C1 WQB/WoB shapes; none of these observations is a general profile speed promise.

**Open scope:** standard batched-prompt prefill, prefill/decode shape switching, the physical 4,096-position ring wrap, rollback/speculation, TP placements, image execution, contexts above 512 and stochastic RNG identity. The bounded all-visible index form selects chronologically; unused raw index scores are omitted. Matching the reference stochastic sampler is a possible follow-up, not a claim of this release. The raw EXL3 FP32-wire `--numerics=exact` experiment elsewhere in this manual remains separate from original-weight `--exactness=exact`.

## 10. Precision words and how to compare results

| Term | Meaning in this manual | What it does not establish |
| --- | --- | --- |
| **Exact** | The explicitly named arithmetic/reference contract, with its scope stated. Raw `numerics=exact` chooses the engine's exact route. | Equality to the official original model unless all required official gates are proved. |
| **Same bits** | Identical compared bits for the named tensors/records/state under the stated source, recipe, input and schedule. Often an optimization with unchanged arithmetic. | All-vocabulary, all-context, all-row-width or all-placement equivalence from a limited comparison. |
| **Reference arithmetic** | A pinned set of weight formats, product forms, norm/HC operations, reduction order, cache rounding, head and sampling behavior. | A universal numeric mode transferable across weight families. EXL3 reference arithmetic is not official-exact. |
| **Order-changing** | Different products, reductions, dispatched numerical forms, or sampler/output behavior; rounding can differ even with unchanged stored weights. Catalog `sum order` is rendered this way. | Necessarily fewer bits of storage, or necessarily a measurable quality loss. |
| **Lossy** | An explicit rounding/storage/operand approximation relative to the comparison family, such as BF16 transport instead of lossless FP32. | An admitted fast tier without paired error and whole-workload performance evidence. |
| **Precision up** | A format/arithmetic change intended to retain more precision, still changing results relative to the old reference. | Automatic official equivalence or a free same-bits optimization. |
| **Draft-only, conditional** | A proposal-policy/draft computation change whose target verifier retains the required bits for every selected width/form. | Target invariance merely because accepted text or token IDs match. If width/form invariance is unproved, retain order-changing qualification. |

Always name what was compared: token IDs, selected F32 logits, full-vocabulary logits, prompt IDs, canonical saved state, tools/reasoning text, token counts or finish reasons. Selected-record equality is more informative than decoded-text equality but narrower than full logits/state. A speedup at C8 is aggregate unless explicitly per stream; C1 is ordinarily per stream. Prefill throughput, decode throughput, TTFT, end-to-end aggregate rate and maximum pause are different metrics.

The pinned S numerical smoke compared eight teacher-forced EXL3 rows with the original cached-one-row fixture: mean/max KL **0.22399302 / 1.10730062 nats**, raw logit RMS/max **0.965879499 / 6.40786219**, top1 **7/8**. This combines weight/arithmetic/placement/head effects and is not held-out corpus PPL, T=0 generation agreement or an isolated head/draft measurement. Do not recast it as a universal quality bound. S thin_fixed numerical receipt, COMPARISON_MATRIX.md precision definitions.

## 11. Monitoring, logs and receipts

Enable `--live_stats` for periodic counters and choose text, JSON or table formatting. The server stats endpoint and detached `stats` client report a running process without loading another model. The client accepts a numeric IPv4/IPv6 address, requires a port, and can print once. A client key file is a single-key file; it is not the named multi-user server key database. No credentials were read to validate this example:


```sh
anemone stats --host=127.0.0.1 --port=8321 --key_file=/secure/anemone-stats.key --interval_ms=1000 --format=table --once
```

Check: `stats_poll` — native tcb82 stats parser accepted; no connection or credential read. Receipt: examples.json.


Inspect readiness and placement logs before accepting traffic: selected cards, weight family, arithmetic/wire class, context/slot capacity, prepared rows/programs, loader receipts and free memory. Keep the command line, binary hash, commit, model/map/pack identity and any explicit layout together. A command that parses but fails planning is a refused run, not a performance sample. A server “ready” line is an allocation/startup event, not an official-equivalence proof.

For performance receipts include power settings, concurrency, prompt/output shape, temperature/seed policy, warm/cold status, cached tokens, prefix policy, native output counts, finish reasons, TTFT distributions and inter-token pauses. An allocation result at 1M and a benchmark at 65,536 are separate receipts. Keep control/candidate source epochs visible; inherited numbers are labeled throughout this manual.

Use a profile build for enabled timing hooks. Release compiles those hooks out: `--draft_trace` accepts only 0 in release and 0 or 1 in debug/thread/profile. `--linear_profile` accepts 0 or 1 in every build mode, subject to the selected recipe's precision requirements. Sanitizer results should state the build and test scope rather than imply GPU execution from a host-only pass.

`SIGINT` and `SIGTERM` stop the server. **SIGHUP reloads configured server keys transactionally**: a failed reload retains the old set, lifetime user totals retain their ordinals, and in-flight requests keep their authenticated username. Do not use SIGHUP as a shutdown command. This is the server's implemented signal contract; the main signal-array name alone is not sufficient to infer it. Closing/resetting a client connection cancels its active generation at the next step or removes its queued request. Streaming errors and cancellations must remain distinct from normal finish events. The engine drains in-flight work before releasing state, with bounded shutdown waits. `source/main.c`, `include/anemone/server.h`.

## 12. Troubleshooting

### 12.1 Out of memory at 1M

First distinguish lexical acceptance, model/state planning, arena allocation, graph preparation and a runtime allocation failure. The `context_tokens` numeric maximum is not a guarantee that the selected slots, wide rows, BATCH forms, tails, graphs and scratch banks fit. Every slot reserves its own state; more slots multiply important allocations. A larger wide ladder adds prepared programs and workspace. Other processes and kept-free reservations reduce the available budget.

Start from the one-slot 1M recipe. For BATCH, keep wide1024/tail128/fine0; do not add the 2K compromise ladder. Check the actual card free-memory receipt. Reduce slot count or context when capacity is insufficient; do not disable validation or reinterpret an OOM as a parser bug. `card_memory_kept_free_mib` reserves more space, so increasing it reduces placement capacity. Main stable-prefix and fork opt-ins, and the separate branch-only small-copy/v5 features, have their own capacity and prepared-set evidence. Do not transfer a stable0 fit to stable1 or a checkpoint-bearing branch.

The historical 15/53 MiB headroom is small enough that a source change or an additional graph can invalidate it. No eight-slot 1M guarantee follows from either historical boot. The exact selected recipe and requested card set must be requalified in an authorized GPU window.

### 12.2 Boot failures and unsupported combinations

| Symptom | Likely cause and corrective action |
| --- | --- |
| Usage text, exit 2 | Check exact case, `--name=value`, decimal range, list arity, repeated option spellings and conflicting draft/speculative selectors, command scope and that every option is before the first file. |
| TP requires DSpark/reference CMP/context | Use DSpark/hybrid, reference CMP products, native PRO, rows≤8 and context≥16384. A scalar-row non-speculative raw command is not an admitted TP boot. |
| BF16 wire requires relaxed TP | Select relaxed/fast numerics explicitly and a TP placement; exact plus BF16 wire is refused. |
| TP5 refuses | Expected: TP5 remains branch-only. A recognized internal enum does not admit the geometry. |
| Axis command refuses | Axes are main; inspect invalid values, duplicate selectors, incompatible partial plans or a scope/refusal for EXL3 exact, fast, vision or mixed qualification. |
| A TP `load` command refuses | TP is an inference startup route here, not the load-only command's admitted path. |
| Original experiment refuses | Exact TP4_PRO pilots have strict exclusion lists. Do not join BATCH, wide experts, chunk core, overlap, code/owner or staging controls unless that composition is admitted. |
| Card/peer/program preparation fails | Check actual card classes/order, CUDA driver, topology, memory budgets and prepared row/state shape. Syntax success proves none of those resources. |
| Pack/map fails or digest mismatch | Use the correct prepared family and readable source extents. Restore matching original bytes or a valid receipted replica; do not bypass digests. |
| Snapshot is ignored | Check namespace, arithmetic salt, format version, token prefix and valid reach. A snapshot is not a durable Responses object. |
| Cache unexpectedly cold | Check complete identity set, scope, rendered tokens, prefix mode and cold provenance; missing exact-reuse boundary or a changed header can be sufficient. |
| Cache unexpectedly shared | Check scope1/2 and anonymous `-`; request IDs alone do not isolate a session. |
| HTTP400 | Read `error.param` and the message. Common causes are unsupported fields, strict tools, stored-response chains, malformed identity and an overfull context. |
| HTTP401/403 | Confirm server key permissions, bearer authentication and stats-user authorization. Do not print key files for diagnosis. |
| HTTP503 / Retry-After | Queue or per-user waiting limit, or failed engine. Distinguish capacity pressure from terminal health failure. |
| Long pauses behind a prompt | Use an admitted exact chunk/decode cadence. More decode rounds can improve completion/pause while worsening new-request TTFT. |
| Same text but changed numerical receipt | Compare native selected/full logits and state, not just tokens. Policy-driven verifier widths and PRO layer boundaries can change arithmetic. |

### 12.3 CLI refusal examples

Stable-prefix and session-fork examples above are now accepted main commands. Small-copy, snapshot v5 and M25 remain checked main refusals:


```sh
anemone serve --placement=tp4_exl3 --speculative=2 --rows=8 --context_tokens=65536 --prefill_prefix_stable=1 --prefill_small_copy_rows=32 tokenizer.bin model.exl3map
```

Check: `copy32` — expected tcb82 refusal: usage: anemone load|generate|serve [--card_order=1,0,2] [--numerics=exact|relaxed|fast (default exact; relaxed/fast enable fixed alternate CMP sums)] [--placement=pp5|tp4_pro|tp4_exl3|tp2_pp2_pro|tp2_ Receipt: examples.json.



```sh
anemone serve --placement=tp4_exl3 --speculative=2 --rows=8 --context_tokens=65536 --session_snapshot_format=5 tokenizer.bin model.exl3map
```

Check: `branch_session_snapshot_format` — expected tcb82 refusal: usage: anemone load|generate|serve [--card_order=1,0,2] [--numerics=exact|relaxed|fast (default exact; relaxed/fast enable fixed alternate CMP sums)] [--placement=pp5|tp4_pro|tp4_exl3|tp2_pp2_pro|tp2_ Receipt: examples.json.



```sh
anemone serve --placement=tp4_exl3 --speculative=2 --rows=8 --context_tokens=65536 --tp_core_queries=1 tokenizer.bin model.exl3map
```

Check: `branch_m25` — expected tcb82 refusal: usage: anemone load|generate|serve [--card_order=1,0,2] [--numerics=exact|relaxed|fast (default exact; relaxed/fast enable fixed alternate CMP sums)] [--placement=pp5|tp4_pro|tp4_exl3|tp2_pp2_pro|tp2_ Receipt: examples.json.


A branch host-channel TP5 launch is also refused by this main parser:


```sh
anemone serve --placement=tp5 --tp5_host_channels=1 --speculative=2 --rows=8 tokenizer.bin original.anemone
```

Check: `branch_tp5_host` — expected tcb82 refusal: usage: anemone load|generate|serve [--card_order=1,0,2] [--numerics=exact|relaxed|fast (default exact; relaxed/fast enable fixed alternate CMP sums)] [--placement=pp5|tp4_pro|tp4_exl3|tp2_pp2_pro|tp2_ Receipt: examples.json.


### 12.4 Safe recovery and reproduction

Preserve the failing argv, stderr, binary/source hashes, source-family identity and any planning receipt before changing settings. Change one subsystem at a time where possible. A failed clone/backend operation can poison the engine; retrying a request on that process is not a substitute for a healthy restart. Use a fresh cold baseline when investigating reuse. Reproduce a reported performance problem with the same power, context allocation, source epoch and prefix policy before attributing it to one knob.

No GPU diagnostic command is hidden in this manual's validation scripts. The evidence binaries stop before file/socket/backend initialization, and the HTTP probe calls native request readers only. For an actual smoke or sanitizer run, coordinate an isolated GPU window with the owning session first.

## 13. Source map, version boundaries and example evidence

Current implementation and the [server profile guide](SERVER_PROFILES.md) define
the supported contract. Dated parser checks and benchmark results retain their
original scope; a parser check does not load weights or qualify a full workload.

| Source | Purpose |
| --- | --- |
| `source/main.c` | Numeric options, parser, resolved defaults and startup admission. |
| `include/anemone/startup_policy.h` | Weight-family, placement and numerical contracts. |
| `include/anemone/server.h`, `include/anemone/chat_completions.h` | HTTP, authentication and request contracts. |
| `source/server_cache.c`, `source/server_forks.c` | Cache and session identity. |
| `source/server_responses_request.c` | Stateless Responses fields and explicit refusals. |
| `tools/cli/KNOBS.md`, `tools/cli/axes.json` | Reviewed knobs and axis registry. |

See [benchmark methods](results/BENCHMARKS.md) for formulas and qualification limits,
and [development](DEVELOPMENT.md) for generators and host checks. Exact remains
the bounded one-session, capacity-512 original-weight recipe. Client templates
do not establish full agent, tool, reasoning or stored-response compatibility.

## 14. Alphabetical knob index

| Knob | Context | Availability |
| --- | --- | --- |
| [`--batch_context_slabs`](#knob-batch_context_slabs) | main CLI | main |
| [`--batch_core_once`](#knob-batch_core_once) | main CLI | main |
| [`--batch_draft_rows`](#knob-batch_draft_rows) | main CLI | main |
| [`--batch_draft_slots_head`](#knob-batch_draft_slots_head) | main CLI | main |
| [`--batch_draft_wide`](#knob-batch_draft_wide) | main CLI | main |
| [`--batch_grouped_rows`](#knob-batch_grouped_rows) | main CLI | main |
| [`--batch_plan_ties`](#knob-batch_plan_ties) | main CLI | main |
| [`--batch_row_sets`](#knob-batch_row_sets) | main CLI | main |
| [`--batch_rows`](#knob-batch_rows) | main CLI | main |
| [`--batch_step_us`](#knob-batch_step_us) | main CLI | main |
| [`--batch_tc_paired`](#knob-batch_tc_paired) | main CLI | main |
| [`--batch_thin_bits`](#knob-batch_thin_bits) | main CLI | main |
| [`--batch_verifiers`](#knob-batch_verifiers) | main CLI | main |
| [`--batch_wide_head`](#knob-batch_wide_head) | main CLI | main |
| [`--bind_address`](#knob-bind_address) | main CLI | main |
| [`--cache_scope`](#knob-cache_scope) | main CLI | main |
| [`--card_memory_kept_free_mib`](#knob-card_memory_kept_free_mib) | main CLI | main |
| [`--card_order`](#knob-card_order) | main CLI | main |
| [`--category`](#knob-category) | main CLI | main |
| [`--chunks_in_flight`](#knob-chunks_in_flight) | main CLI | main |
| [`--cmp_order`](#knob-cmp_order) | main CLI | main |
| [`--concurrency`](#knob-concurrency) | main CLI | main |
| [`--context_tokens`](#knob-context_tokens) | main CLI | main |
| [`--dense_skinny_bf16`](#knob-dense_skinny_bf16) | main CLI | main |
| [`--dense_skinny_f32`](#knob-dense_skinny_f32) | main CLI | main |
| [`--dense_staged`](#knob-dense_staged) | main CLI | main |
| [`--draft_base_us`](#knob-draft_base_us) | main CLI | main |
| [`--draft_bf16_operands`](#knob-draft_bf16_operands) | main CLI | main |
| [`--draft_bf16_rows`](#knob-draft_bf16_rows) | main CLI | main |
| [`--draft_block`](#knob-draft_block) | main CLI | main |
| [`--draft_calibration`](#knob-draft_calibration) | main CLI | main |
| [`--draft_learned_acceptance`](#knob-draft_learned_acceptance) | main CLI | main |
| [`--draft_policy`](#knob-draft_policy) | main CLI | main |
| [`--draft_positions`](#knob-draft_positions) | main CLI | main |
| [`--draft_probe_rounds`](#knob-draft_probe_rounds) | main CLI | main |
| [`--draft_row_us`](#knob-draft_row_us) | main CLI | main |
| [`--draft_session_acceptance`](#knob-draft_session_acceptance) | main CLI | main |
| [`--draft_start_economic`](#knob-draft_start_economic) | main CLI | main |
| [`--draft_tail_us`](#knob-draft_tail_us) | main CLI | main |
| [`--draft_threshold`](#knob-draft_threshold) | main CLI | main |
| [`--draft_trace`](#knob-draft_trace) | main CLI | main |
| [`--draft_usage`](#knob-draft_usage) | main CLI | main |
| [`--draft`](#knob-draft) | main CLI | main |
| [`--dump_draft_ring`](#knob-dump_draft_ring) | main CLI | main |
| [`--engram_shared`](#knob-engram_shared) | main CLI | main |
| [`--exactness`](#knob-exactness) | main CLI | main |
| [`--exl3_expert_parts`](#knob-exl3_expert_parts) | main CLI | main |
| [`--exl3_expert_rotation`](#knob-exl3_expert_rotation) | main CLI | main |
| [`--exl3_fast_products`](#knob-exl3_fast_products) | main CLI | main |
| [`--exl3_tc_moe`](#knob-exl3_tc_moe) | main CLI | main |
| [`--exl3_tc_product_blocks`](#knob-exl3_tc_product_blocks) | main CLI | main |
| [`--exl3_tcp_two_blocks`](#knob-exl3_tcp_two_blocks) | main CLI | main |
| [`--exl3_wide_entry`](#knob-exl3_wide_entry) | main CLI | main |
| [`--floor_gib`](#knob-floor_gib-replicate) | replicate | main |
| [`--focus`](#knob-focus) | main CLI | main |
| [`--format`](#knob-format-stats) | stats | main |
| [`--full_steps`](#knob-full_steps) | main CLI | main |
| [`--generated_tokens`](#knob-generated_tokens) | main CLI | main |
| [`--help-scheduling`](#knob-help-scheduling) | main CLI | main |
| [`--host`](#knob-host-stats) | stats | main |
| [`--hybrid_lookup_min`](#knob-hybrid_lookup_min) | main CLI | main |
| [`--hybrid_lookup_prior`](#knob-hybrid_lookup_prior) | main CLI | main |
| [`--ignore_eos`](#knob-ignore_eos) | main CLI | main |
| [`--interval_ms`](#knob-interval_ms-stats) | stats | main |
| [`--key_file`](#knob-key_file) | main CLI | main |
| [`--key_file`](#knob-key_file-stats) | stats | main |
| [`--layout`](#knob-layout) | main CLI | main |
| [`--linear_order`](#knob-linear_order) | main CLI | main |
| [`--linear_profile`](#knob-linear_profile) | main CLI | main |
| [`--live_stats_format`](#knob-live_stats_format) | main CLI | main |
| [`--live_stats`](#knob-live_stats) | main CLI | main |
| [`--lookup_ngram_max`](#knob-lookup_ngram_max) | main CLI | main |
| [`--lookup_ngram_min`](#knob-lookup_ngram_min) | main CLI | main |
| [`--map_reads`](#knob-map_reads) | main CLI | main |
| [`--model_alias`](#knob-model_alias) | main CLI | main |
| [`--numerics`](#knob-numerics) | main CLI | main |
| [`--official_hc_rows`](#knob-official_hc_rows) | main CLI | main |
| [`--official_reference`](#knob-official_reference) | main CLI | main |
| [`--once`](#knob-once-stats) | stats | main |
| [`--original_batch`](#knob-original_batch) | main CLI | main |
| [`--original_chunk_core`](#knob-original_chunk_core) | main CLI | main |
| [`--original_dense_shards`](#knob-original_dense_shards) | main CLI | branch, not in main yet |
| [`--original_dense_staged`](#knob-original_dense_staged) | main CLI | main |
| [`--original_queries_piece`](#knob-original_queries_piece) | main CLI | branch, not in main yet |
| [`--original_wide_experts`](#knob-original_wide_experts) | main CLI | main |
| [`--pipeline_steps`](#knob-pipeline_steps) | main CLI | main |
| [`--placement`](#knob-placement) | main CLI | main |
| [`--port`](#knob-port) | main CLI | main |
| [`--port`](#knob-port-stats) | stats | main |
| [`--prefill_chunk_rows`](#knob-prefill_chunk_rows) | main CLI | main |
| [`--prefill_cone_skip`](#knob-prefill_cone_skip) | main CLI | main |
| [`--prefill_cone`](#knob-prefill_cone) | main CLI | main |
| [`--prefill_dead_halves`](#knob-prefill_dead_halves) | main CLI | main |
| [`--prefill_decode_every`](#knob-prefill_decode_every) | main CLI | main |
| [`--prefill_decode_rounds`](#knob-prefill_decode_rounds) | main CLI | main |
| [`--prefill_decode_rows`](#knob-prefill_decode_rows) | main CLI | main |
| [`--prefill_decode_waiters`](#knob-prefill_decode_waiters) | main CLI | main |
| [`--prefill_fast_plan`](#knob-prefill_fast_plan) | main CLI | main |
| [`--prefill_first`](#knob-prefill_first) | main CLI | main |
| [`--prefill_front_tail`](#knob-prefill_front_tail) | main CLI | main |
| [`--prefill_ladder_1024`](#knob-prefill_ladder_1024) | main CLI | main |
| [`--prefill_ladder_1536`](#knob-prefill_ladder_1536) | main CLI | main |
| [`--prefill_ladder_fine`](#knob-prefill_ladder_fine) | main CLI | main |
| [`--prefill_margin`](#knob-prefill_margin) | main CLI | main |
| [`--prefill_order_wait`](#knob-prefill_order_wait) | main CLI | main |
| [`--prefill_order`](#knob-prefill_order) | main CLI | main |
| [`--prefill_prefix_stable`](#knob-prefill_prefix_stable) | main CLI | main |
| [`--prefill_prompt_chunks`](#knob-prefill_prompt_chunks) | main CLI | main |
| [`--prefill_small_copy_rows`](#knob-prefill_small_copy_rows) | main CLI | branch, not in main yet |
| [`--prefill_tail_overlay`](#knob-prefill_tail_overlay) | main CLI | main |
| [`--prefill_tail_rows`](#knob-prefill_tail_rows) | main CLI | main |
| [`--prefill_tail_sets`](#knob-prefill_tail_sets) | main CLI | main |
| [`--prefill_thin_tail`](#knob-prefill_thin_tail) | main CLI | main |
| [`--prefix_reuse_exact`](#knob-prefix_reuse_exact) | main CLI | main |
| [`--prefix_reuse`](#knob-prefix_reuse) | main CLI | main |
| [`--pro_emulation`](#knob-pro_emulation) | main CLI | main |
| [`--prompt_token_ids`](#knob-prompt_token_ids) | main CLI | main |
| [`--prune`](#knob-prune-replicate) | replicate | main |
| [`--queue_depth`](#knob-queue_depth) | main CLI | main |
| [`--rate_mib`](#knob-rate_mib-replicate) | replicate | main |
| [`--reader_threads_per_copy`](#knob-reader_threads_per_copy) | main CLI | main |
| [`--reasoning_effort`](#knob-reasoning_effort) | main CLI | main |
| [`--rows`](#knob-rows) | main CLI | main |
| [`--seed`](#knob-seed) | main CLI | main |
| [`--session_directory`](#knob-session_directory) | main CLI | main |
| [`--session_fork`](#knob-session_fork) | main CLI | main |
| [`--session_grace_seconds`](#knob-session_grace_seconds) | main CLI | main |
| [`--session_idle_save_seconds`](#knob-session_idle_save_seconds) | main CLI | main |
| [`--session_load_tokens`](#knob-session_load_tokens) | main CLI | main |
| [`--session_save_tokens`](#knob-session_save_tokens) | main CLI | main |
| [`--session_slots`](#knob-session_slots) | main CLI | main |
| [`--session_snapshot_format`](#knob-session_snapshot_format) | main CLI | branch, not in main yet |
| [`--speculative_depth`](#knob-speculative_depth) | main CLI | main |
| [`--speculative`](#knob-speculative) | main CLI | main |
| [`--stats_users`](#knob-stats_users) | main CLI | main |
| [`--step_graphs`](#knob-step_graphs) | main CLI | main |
| [`--temperature`](#knob-temperature) | main CLI | main |
| [`--top_p`](#knob-top_p) | main CLI | main |
| [`--tp5_expert_csf`](#knob-tp5_expert_csf) | main CLI | branch, not in main yet |
| [`--tp5_expert_sidecar`](#knob-tp5_expert_sidecar) | main CLI | branch, not in main yet |
| [`--tp5_host_channels`](#knob-tp5_host_channels) | main CLI | branch, not in main yet |
| [`--tp5_original_draft`](#knob-tp5_original_draft) | main CLI | branch, not in main yet |
| [`--tp5_pro_third`](#knob-tp5_pro_third) | main CLI | branch, not in main yet |
| [`--tp_attention_owner`](#knob-tp_attention_owner) | main CLI | main |
| [`--tp_bf16_slab_rings`](#knob-tp_bf16_slab_rings) | main CLI | main |
| [`--tp_chunk_core`](#knob-tp_chunk_core) | main CLI | main |
| [`--tp_code_wire`](#knob-tp_code_wire) | main CLI | main |
| [`--tp_core_compact_halves`](#knob-tp_core_compact_halves) | main CLI | main |
| [`--tp_core_heads`](#knob-tp_core_heads) | main CLI | main |
| [`--tp_core_queries`](#knob-tp_core_queries) | main CLI | branch, not in main yet |
| [`--tp_decode_round_once`](#knob-tp_decode_round_once) | main CLI | main |
| [`--tp_draft_layers`](#knob-tp_draft_layers) | main CLI | main |
| [`--tp_draft_shards`](#knob-tp_draft_shards) | main CLI | main |
| [`--tp_exact_dense`](#knob-tp_exact_dense) | main CLI | main |
| [`--tp_front_branch`](#knob-tp_front_branch) | main CLI | main |
| [`--tp_full_split_overlap`](#knob-tp_full_split_overlap) | main CLI | main |
| [`--tp_gather_balance`](#knob-tp_gather_balance) | main CLI | main |
| [`--tp_graph_branches`](#knob-tp_graph_branches) | main CLI | main |
| [`--tp_graph_parts`](#knob-tp_graph_parts) | main CLI | main |
| [`--tp_head_mma`](#knob-tp_head_mma) | main CLI | main |
| [`--tp_head_shards`](#knob-tp_head_shards) | main CLI | main |
| [`--tp_head_stream`](#knob-tp_head_stream) | main CLI | main |
| [`--tp_index_span`](#knob-tp_index_span) | main CLI | main |
| [`--tp_launch_trims`](#knob-tp_launch_trims) | main CLI | main |
| [`--tp_load_copy_split`](#knob-tp_load_copy_split) | main CLI | main |
| [`--tp_load_host_threads`](#knob-tp_load_host_threads) | main CLI | main |
| [`--tp_load_reads_per_copy`](#knob-tp_load_reads_per_copy) | main CLI | main |
| [`--tp_load_threads`](#knob-tp_load_threads) | main CLI | main |
| [`--tp_ordered_wide`](#knob-tp_ordered_wide) | main CLI | main |
| [`--tp_parallel_launch`](#knob-tp_parallel_launch) | main CLI | main |
| [`--tp_piece_rings`](#knob-tp_piece_rings) | main CLI | main |
| [`--tp_pipeline_overlap`](#knob-tp_pipeline_overlap) | main CLI | main |
| [`--tp_pool_chunk`](#knob-tp_pool_chunk) | main CLI | main |
| [`--tp_pool_gather`](#knob-tp_pool_gather) | main CLI | main |
| [`--tp_queries_piece`](#knob-tp_queries_piece) | main CLI | main |
| [`--tp_replicated_embedding`](#knob-tp_replicated_embedding) | main CLI | main |
| [`--tp_ring_column_owners`](#knob-tp_ring_column_owners) | main CLI | main |
| [`--tp_router_first_whole`](#knob-tp_router_first_whole) | main CLI | main |
| [`--tp_router_mma`](#knob-tp_router_mma) | main CLI | main |
| [`--tp_split_least_rows`](#knob-tp_split_least_rows) | main CLI | main |
| [`--tp_split_overlap`](#knob-tp_split_overlap) | main CLI | main |
| [`--tp_trunk_mma_fused`](#knob-tp_trunk_mma_fused) | main CLI | main |
| [`--tp_trunk_mma_thin_splits`](#knob-tp_trunk_mma_thin_splits) | main CLI | main |
| [`--tp_trunk_mma`](#knob-tp_trunk_mma) | main CLI | main |
| [`--tp_trunk_shape_slices`](#knob-tp_trunk_shape_slices) | main CLI | main |
| [`--tp_wide_exchange`](#knob-tp_wide_exchange) | main CLI | main |
| [`--tp_wire_bf16`](#knob-tp_wire_bf16) | main CLI | main |
| [`--turbo_load`](#knob-turbo_load) | main CLI | main |
| [`--user_queue_depth`](#knob-user_queue_depth) | main CLI | main |
| [`--user`](#knob-user-keygen) | keygen | main |
| [`--wide_rows`](#knob-wide_rows) | main CLI | main |
| [`DIAG_DRAFT_PROBE`](#knob-diag_draft_probe) | environment | main diagnostic |
| [`DIAG_DRAFT_SKIP_FFN`](#knob-diag_draft_skip_ffn) | environment | main diagnostic |
| [`DIAG_DRAFT_STAGES`](#knob-diag_draft_stages) | environment | main diagnostic |
| [`DIAG_STATE_DIGESTS`](#knob-diag_state_digests) | environment | main diagnostic |

## 15. Compact serve-flag reference

**Main `69d4d967880fb39c9ed3f82bf646f95149374d04`, `bin/tcb82/anemone`.** This is the compact companion to the sections above. It lists every flag the final-main serve parser accepts; it does not import the older production tree's transport/scheduler switches.

Defaults are stated for PP5, TP4_PRO and TP4_EXL3 wherever they differ. Unqualified numeric defaults apply to all three. A conditional default is part of a recipe: omitted pipeline depth becomes5 only for qualifying PP5 multi-session DSpark serve; compatible TP4_PRO wide experts become1; explicit values retain precedence. Four unset axes insert nothing. PRO uses class120 native/sm_120a math by default; `pro_emulation=1` is the opt-in class80 emulation and is refused by current TP startup.

All 170 flag lines have a successful native-parser receipt in serve_flags.json and examples.json, including their full argv. Each line names its check ID; it is a reference entry, not a standalone shell command. Parsing stops before model/socket/backend use. Resource, model, arithmetic, producer capability and runtime gates still apply.

Every option precedes tokenizer tables and the weight source, uses exact `--name=value` spelling except the dedicated help flag, and is unique by spelling. The two depth aliases share a value but separate occurrence counters: if both are given, the later wins. `help axes` and `help knobs [prefix]` are additional main help commands, documented in the manual.

Copy32, snapshot v5, TP5/M25 and pending O query/dense-shard controls are branch-only and are excluded from this main serve list. Reference component APIs being shipped does not enable them in serving; official_hc_rows defaults0.

### Loader and turbo replicas

- [`--card_order`](#knob-card_order) — **Default:** Automatic backend enumeration/fill; TP4_EXL3 auto-selects four CMPs. The table's zero is array storage, not a request for card0 only. **Meaning:** Choose backend card ordinals in stage-fill order. Check: `knob_card_order` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--reader_threads_per_copy`](#knob-reader_threads_per_copy) — **Default:** 2. **Meaning:** Allocate this many asynchronous readers per pack copy. Check: `knob_reader_threads_per_copy` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_load_threads`](#knob-tp_load_threads) — **Default:** 32. **Meaning:** Load TP card weights concurrently; zero keeps the serial loader. Check: `knob_tp_load_threads` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--turbo_load`](#knob-turbo_load) — **Default:** 1. **Meaning:** Discover and read receipted replicas on other drives. Check: `knob_turbo_load` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--map_reads`](#knob-map_reads) — **Default:** 4 auto:3 when receipted replicas are found, otherwise1. **Meaning:** Choose buffered/direct EXL3-map reads and page-cache eviction policy. Check: `knob_map_reads` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_load_host_threads`](#knob-tp_load_host_threads) — **Default:** 16; explicit0 inherits tp_load_threads. **Meaning:** Load the host arena with this many workers; zero follows TP load threads. Check: `knob_tp_load_host_threads` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_load_copy_split`](#knob-tp_load_copy_split) — **Default:** 1. **Meaning:** Read host chunks from copy zero and device extents from the last copy. Check: `knob_tp_load_copy_split` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_load_reads_per_copy`](#knob-tp_load_reads_per_copy) — **Default:** 0. **Meaning:** Limit TP in-flight reads per copy; zero is unlimited. Check: `knob_tp_load_reads_per_copy` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Memory, slots, context and 1M

- [`--card_memory_kept_free_mib`](#knob-card_memory_kept_free_mib) — **Default:** 0. **Meaning:** Subtract this many MiB from each card's placement budget. Check: `knob_card_memory_kept_free_mib` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--session_slots`](#knob-session_slots) — **Default:** 1. **Meaning:** Allocate this many independent active session slots. Check: `knob_session_slots` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--wide_rows`](#knob-wide_rows) — **Default:** 0. **Meaning:** Prepare a wide prefill program of this many rows; zero disables it. Check: `knob_wide_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--rows`](#knob-rows) — **Default:** 1 on every raw placement; the usable speculative TP recipes explicitly select8. **Meaning:** Set ordinary THIN step row capacity (and proposal capacity minus one). Check: `knob_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--context_tokens`](#knob-context_tokens) — **Default:** 16384 per slot for every raw placement; larger capacities are explicit. **Meaning:** Allocate the maximum prompt-plus-generation context per slot. Check: `knob_context_tokens` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--layout`](#knob-layout) — **Default:** automatic fill **Meaning:** Override PP5 unit placement. Check: `knob_layout` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Prefill and chunk planning

- [`--full_steps`](#knob-full_steps) — **Default:** 1. **Meaning:** Use full programs behind the prefill front. Check: `knob_full_steps` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--chunks_in_flight`](#knob-chunks_in_flight) — **Default:** 1. **Meaning:** Bound chunk steps in flight; one uses the exact cone planner. Check: `knob_chunks_in_flight` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_margin`](#knob-prefill_margin) — **Default:** 127. **Meaning:** Keep this many full rows beyond the prefill cone. Check: `knob_prefill_margin` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_fast_plan`](#knob-prefill_fast_plan) — **Default:** 0. **Meaning:** Choose the largest fitting prefill program instead of the exact plan. Check: `knob_prefill_fast_plan` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_chunk_rows`](#knob-prefill_chunk_rows) — **Default:** 0. **Meaning:** Cap all exact-plan prefill chunks. Check: `knob_prefill_chunk_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_cone_skip`](#knob-prefill_cone_skip) — **Default:** 1. **Meaning:** Skip dead cone work; modes two/three extend the retained optimization. Check: `knob_prefill_cone_skip` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_tail_rows`](#knob-prefill_tail_rows) — **Default:** 0 when ineligible;128 with eligible wide1024 or BATCH;448 with eligible wide2048+ THIN. BATCH retains128. **Meaning:** Prepare 128/320/448-row full tail programs; zero disables them. Check: `knob_prefill_tail_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_ladder_1024`](#knob-prefill_ladder_1024) — **Default:** 1. **Meaning:** Add 1024-row programs below larger wide programs. Check: `knob_prefill_ladder_1024` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_ladder_1536`](#knob-prefill_ladder_1536) — **Default:** 1. **Meaning:** Add 1536-row programs below larger wide programs. Check: `knob_prefill_ladder_1536` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_ladder_fine`](#knob-prefill_ladder_fine) — **Default:** 1 for wide2048+ THIN;0 for BATCH or smaller wide programs. **Meaning:** Add 1792/1280/768-row exact-plan programs. Check: `knob_prefill_ladder_fine` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_front_tail`](#knob-prefill_front_tail) — **Default:** 0 without tail;1 with other tails;2 with tail448. **Meaning:** Use tail programs for the front region; two also uses the longer full tails. Check: `knob_prefill_front_tail` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_tail_overlay`](#knob-prefill_tail_overlay) — **Default:** 1. **Meaning:** Overlay tail TC MoE workspace on prefill workspace. Check: `knob_prefill_tail_overlay` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_tail_sets`](#knob-prefill_tail_sets) — **Default:** 1. **Meaning:** Run tail trunk products over their 64-row groups in one launch. Check: `knob_prefill_tail_sets` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_thin_tail`](#knob-prefill_thin_tail) — **Default:** 1 with a tail and no BATCH;0 otherwise. **Meaning:** Use 16..56-row programs for rows below a 64-row tail. Check: `knob_prefill_thin_tail` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_dead_halves`](#knob-prefill_dead_halves) — **Default:** 2. **Meaning:** Trim cone-dead sums; mode two also trims dead copies. Check: `knob_prefill_dead_halves` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_cone`](#knob-prefill_cone) — **Default:** 0. **Meaning:** Use a bounded front region with a prepared prefill cone. Check: `knob_prefill_cone` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_prefix_stable`](#knob-prefill_prefix_stable) — **Default:** 0. **Meaning:** Use fixed-capacity prefill forms with closed eight-row prefixes. Check: `stable_prefix` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Decode and scheduling

- [`--batch_rows`](#knob-batch_rows) — **Default:** No BATCH list (zero-filled internal array); THIN programs are prepared. **Meaning:** Prepare BATCH programs at these row capacities instead of THIN programs. Check: `knob_batch_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_context_slabs`](#knob-batch_context_slabs) — **Default:** 0. **Meaning:** Process BATCH draft context in THIN-sized eight-row slabs. Check: `knob_batch_context_slabs` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_thin_bits`](#knob-batch_thin_bits) — **Default:** 1. **Meaning:** Use THIN numerical splits in BATCH trunk/head/TC-MoE products. Check: `knob_batch_thin_bits` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_tc_paired`](#knob-batch_tc_paired) — **Default:** 0. **Meaning:** Pair TC MoE short forms from this many BATCH rows; zero disables pairing. Check: `knob_batch_tc_paired` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_row_sets`](#knob-batch_row_sets) — **Default:** 0. **Meaning:** Use trunk-product row sets above the grouped-row threshold. Check: `knob_batch_row_sets` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_core_once`](#knob-batch_core_once) — **Default:** PP5/TP4_PRO and other original placements: 2; TP4_EXL3 THIN: 3. **Meaning:** Compute BATCH attention core once; mode2 adds window writes and mode3 pools/deferred slots. Check: `knob_batch_core_once` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_grouped_rows`](#knob-batch_grouped_rows) — **Default:** 16. **Meaning:** Use grouped THIN trunk forms through this BATCH row count. Check: `knob_batch_grouped_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_step_us`](#knob-batch_step_us) — **Default:** No raw prices. Eligible omitted-policy measured EXL3 BATCH receives19500,21500,28600,30000,32000,39000,46000,58000us. Explicit policy or costs preserve their former behavior. **Meaning:** Supply per-program measured costs for the BATCH expected-token planner. Check: `knob_batch_step_us` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_verifiers`](#knob-batch_verifiers) — **Default:** Unset/derived with the BATCH geometry; not a literal single verifier0. **Meaning:** Supply each BATCH program's drafting-session capacity. Check: `knob_batch_verifiers` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_first`](#knob-prefill_first) — **Default:** 0. **Meaning:** Prefer another session's wide/full prefill chunk before ordinary steps. Check: `knob_prefill_first` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_order`](#knob-prefill_order) — **Default:** 2 only for the qualified default BATCH1024 cap/round/waiter combination;0 otherwise. **Meaning:** Choose round-robin chunks, command arrival order, shortest remaining prompt or opt-in shared short residuals. Check: `knob_prefill_order` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_order_wait`](#knob-prefill_order_wait) — **Default:** 16. **Meaning:** Bound shortest-order waiting in serving scheduler ticks. Check: `knob_prefill_order_wait` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_decode_every`](#knob-prefill_decode_every) — **Default:** 0. **Meaning:** Insert one decode round after this many head chunks; zero disables it. Check: `knob_prefill_decode_every` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_decode_waiters`](#knob-prefill_decode_waiters) — **Default:** 0 on every no-axis route. **Meaning:** Keep whole chunks/one round while multiple prompts wait. Check: `knob_prefill_decode_waiters` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_decode_rounds`](#knob-prefill_decode_rounds) — **Default:** 1 (including qualified BATCH), unless explicitly overridden. **Meaning:** Run this many ordinary decode rounds between capped chunks. Check: `knob_prefill_decode_rounds` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefill_decode_rows`](#knob-prefill_decode_rows) — **Default:** Serve qualified BATCH1024:1024; THIN/generate/other:0. Explicit0 opts out. **Meaning:** Cap prefill chunks while decoders are active. Check: `knob_prefill_decode_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Draft and speculative policy

- [`--tp_draft_shards`](#knob-tp_draft_shards) — **Default:** 0. **Meaning:** Shard draft head, Markov projection and sampler with the target head. Check: `knob_tp_draft_shards` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_draft_layers`](#knob-tp_draft_layers) — **Default:** 0. **Meaning:** Tensor-parallelize the draft model's three layers. Check: `knob_tp_draft_layers` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_draft_rows`](#knob-batch_draft_rows) — **Default:** 0. **Meaning:** Batch verifier draft rows into the BATCH program. Check: `knob_batch_draft_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_wide_head`](#knob-batch_wide_head) — **Default:** 0. **Meaning:** Use the wide target head for BATCH verification. Check: `knob_batch_wide_head` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_draft_slots_head`](#knob-batch_draft_slots_head) — **Default:** 0. **Meaning:** Run one verifier's draft head across eight slots at each position. Check: `knob_batch_draft_slots_head` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_bf16_rows`](#knob-draft_bf16_rows) — **Default:** 0. **Meaning:** Use BF16 rows for draft attention products. Check: `knob_draft_bf16_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_draft_wide`](#knob-batch_draft_wide) — **Default:** 3. **Meaning:** Widen draft work: one sums/posts, two products, three cores. Check: `knob_batch_draft_wide` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_bf16_operands`](#knob-draft_bf16_operands) — **Default:** 0. **Meaning:** Use BF16 operands for class-80 draft quantized products. Check: `knob_draft_bf16_operands` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_learned_acceptance`](#knob-draft_learned_acceptance) — **Default:** PP5/TP4_PRO and other original placements: 0; TP4_EXL3 THIN: 1. **Meaning:** Update the draft policy's shared confidence table from outcomes. Check: `knob_draft_learned_acceptance` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_session_acceptance`](#knob-draft_session_acceptance) — **Default:** 0. **Meaning:** Scale proposal acceptance by the current session's measured outcomes. Check: `knob_draft_session_acceptance` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--batch_plan_ties`](#knob-batch_plan_ties) — **Default:** 0. **Meaning:** Cut planned drafts only between unequal predicted reach. Check: `knob_batch_plan_ties` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_start_economic`](#knob-draft_start_economic) — **Default:** 1. **Meaning:** Start DSpark at policy depth instead of a single-proposal probe. Check: `knob_draft_start_economic` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--speculative`](#knob-speculative) — **Default:** PP5/TP4_PRO and other original placements: 0; TP4_EXL3 THIN: 2. **Meaning:** Choose off, n-gram lookup, checkpoint DSpark or hybrid proposals. Check: `knob_speculative` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_policy`](#knob-draft_policy) — **Default:** PP5/TP4_PRO: cost1. EXL3 single-slot THIN: fixed0; BATCH or multiple slots: rate2. **Meaning:** Choose fixed, cost, rate or adaptive proposal-depth selection. Check: `knob_draft_policy` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_threshold`](#knob-draft_threshold) — **Default:** 100. **Meaning:** Set the minimum draft confidence in thousandths. Check: `knob_draft_threshold` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_base_us`](#knob-draft_base_us) — **Default:** PP5/TP4_PRO and other original placements: 12700; TP4_EXL3 THIN: 14300. **Meaning:** Estimate the base verification-step cost in microseconds. Check: `knob_draft_base_us` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_tail_us`](#knob-draft_tail_us) — **Default:** PP5/TP4_PRO and other original placements: 2200; TP4_EXL3 THIN: 0. **Meaning:** Estimate fixed tail cost in microseconds. Check: `knob_draft_tail_us` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_row_us`](#knob-draft_row_us) — **Default:** PP5/TP4_PRO and other original placements: 2700; TP4_EXL3 THIN: 690. **Meaning:** Estimate each additional verification row's cost in microseconds. Check: `knob_draft_row_us` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_probe_rounds`](#knob-draft_probe_rounds) — **Default:** 8. **Meaning:** Probe full draft depth after this many policy rounds. Check: `knob_draft_probe_rounds` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--hybrid_lookup_prior`](#knob-hybrid_lookup_prior) — **Default:** 500. **Meaning:** Set lookup acceptance prior in thousandths. Check: `knob_hybrid_lookup_prior` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--hybrid_lookup_min`](#knob-hybrid_lookup_min) — **Default:** 3. **Meaning:** Use lookup in hybrid mode only from this n-gram length. Check: `knob_hybrid_lookup_min` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_block`](#knob-draft_block) — **Default:** 5. **Meaning:** Set the device draft block capacity. Check: `knob_draft_block` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_positions`](#knob-draft_positions) — **Default:** Raw cap7; DSpark/hybrid with default block5 clips an omitted cap to5. An explicit cap above the allocated block refuses. **Meaning:** Bound proposals verified per round. Check: `knob_draft_positions` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--speculative_depth`](#knob-speculative_depth) — **Default:** Same destination as draft_positions: raw7, default DSpark clip5. Each spelling is unique, but supplying both distinct aliases is accepted on main and the later value wins. Prefer one spelling. **Meaning:** Alias of draft_positions. Check: `knob_speculative_depth` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--lookup_ngram_max`](#knob-lookup_ngram_max) — **Default:** 4. **Meaning:** Set the longest lookup suffix searched. Check: `knob_lookup_ngram_max` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--lookup_ngram_min`](#knob-lookup_ngram_min) — **Default:** 2. **Meaning:** Set the shortest lookup suffix searched. Check: `knob_lookup_ngram_min` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_calibration`](#knob-draft_calibration) — **Default:** none **Meaning:** Override draft acceptance priors in mode/position/bin order. Check: `knob_draft_calibration` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft`](#knob-draft) — **Default:** off **Meaning:** Select speculative proposal source by name. Check: `knob_draft` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Attention, index and Engram

- [`--tp_attention_owner`](#knob-tp_attention_owner) — **Default:** 0. **Meaning:** Keep experimental TP4 attention state on its owning rank. Check: `knob_tp_attention_owner` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_exact_dense`](#knob-tp_exact_dense) — **Default:** 1. **Meaning:** Use exact output-sharded attention dense products on the FP32 EXL3 recipe. Check: `knob_tp_exact_dense` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_gather_balance`](#knob-tp_gather_balance) — **Default:** 0. **Meaning:** Assign chunk index/router work to narrower expert ranks (four policies). Check: `knob_tp_gather_balance` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_pool_chunk`](#knob-tp_pool_chunk) — **Default:** 1. **Meaning:** Prepare pooled entries/keys before chunk-slab scans. Check: `knob_tp_pool_chunk` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_pool_gather`](#knob-tp_pool_gather) — **Default:** 1. **Meaning:** Partition pooled-source compressor rows over ranks and gather results. Check: `knob_tp_pool_gather` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_router_first_whole`](#knob-tp_router_first_whole) — **Default:** 0. **Meaning:** Compute first-half router rows on every rank without gathering. Check: `knob_tp_router_first_whole` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_queries_piece`](#knob-tp_queries_piece) — **Default:** 1. **Meaning:** Project wide-program queries by piece alongside wq_b. Check: `knob_tp_queries_piece` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_index_span`](#knob-tp_index_span) — **Default:** 1. **Meaning:** Scan/select a span of chunk index rows per launch. Check: `knob_tp_index_span` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_core_compact_halves`](#knob-tp_core_compact_halves) — **Default:** 1. **Meaning:** Write split-half core results into compact wo_a rows. Check: `knob_tp_core_compact_halves` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_chunk_core`](#knob-tp_chunk_core) — **Default:** 1. **Meaning:** Run a chunk's attention core over the whole chunk. Check: `knob_tp_chunk_core` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_core_heads`](#knob-tp_core_heads) — **Default:** 0. **Meaning:** Limit attention-core heads per block; zero uses the measured rule. Check: `knob_tp_core_heads` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--engram_shared`](#knob-engram_shared) — **Default:** 0. **Meaning:** Gather identical TP Engram inputs once into a shared pinned block. Check: `knob_engram_shared` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### MoE and expert products

- [`--exl3_expert_parts`](#knob-exl3_expert_parts) — **Default:** 4. **Meaning:** Choose four reference or eight alternate expert product parts. Check: `knob_exl3_expert_parts` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--exl3_fast_products`](#knob-exl3_fast_products) — **Default:** 1. **Meaning:** Use wide-kernel EXL3 products instead of the retained chain. Check: `knob_exl3_fast_products` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--exl3_wide_entry`](#knob-exl3_wide_entry) — **Default:** 1. **Meaning:** Run each chunk's EXL3 experts once over the full chunk. Check: `knob_exl3_wide_entry` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--exl3_tc_moe`](#knob-exl3_tc_moe) — **Default:** 0. **Meaning:** Use GNmac-style tensor-core EXL3 routed MoE products. Check: `knob_exl3_tc_moe` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--exl3_tc_product_blocks`](#knob-exl3_tc_product_blocks) — **Default:** 0. **Meaning:** Override the two EXL3 TC-MoE product grids in THIN/BATCH programs. Check: `knob_exl3_tc_product_blocks` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--exl3_expert_rotation`](#knob-exl3_expert_rotation) — **Default:** 0. **Meaning:** Rotate per-expert TC chunks across four ranks or three non-head ranks. Check: `knob_exl3_expert_rotation` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_router_mma`](#knob-tp_router_mma) — **Default:** 0. **Meaning:** Use split-K tensor-core BATCH router products (two alternate modes). Check: `knob_tp_router_mma` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--exl3_tcp_two_blocks`](#knob-exl3_tcp_two_blocks) — **Default:** 1. **Meaning:** Launch prefill TC MoE products at two blocks per multiprocessor. Check: `knob_exl3_tcp_two_blocks` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Tensor transport, graphs and overlap

- [`--tp_parallel_launch`](#knob-tp_parallel_launch) — **Default:** 0. **Meaning:** Launch each TP card's graph groups from a separate host thread. Check: `knob_tp_parallel_launch` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_graph_parts`](#knob-tp_graph_parts) — **Default:** 1. **Meaning:** Partition each card's graph behind the prefill cone. Check: `knob_tp_graph_parts` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_ordered_wide`](#knob-tp_ordered_wide) — **Default:** 0. **Meaning:** Use ordered-wide attention trunk products for EXL3 prefill. Check: `knob_tp_ordered_wide` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_graph_branches`](#knob-tp_graph_branches) — **Default:** 1. **Meaning:** Branch shared-expert work beside routed experts; mode two includes BATCH. Check: `knob_tp_graph_branches` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_piece_rings`](#knob-tp_piece_rings) — **Default:** 0. **Meaning:** Exchange BF16 wide results over 64-row pieces instead of eight-row slabs. Check: `knob_tp_piece_rings` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_trunk_shape_slices`](#knob-tp_trunk_shape_slices) — **Default:** 1. **Meaning:** Use the measured slice count per THIN trunk shard shape; zero forces four. Check: `knob_tp_trunk_shape_slices` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_bf16_slab_rings`](#knob-tp_bf16_slab_rings) — **Default:** 0. **Meaning:** Use BF16 slab-ring attention sums instead of whole-row FP32 accumulation. Check: `knob_tp_bf16_slab_rings` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_wide_exchange`](#knob-tp_wide_exchange) — **Default:** 5. **Meaning:** Choose wide transport: push, slab rings, owner sums or copy-engine rings. Check: `knob_tp_wide_exchange` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_split_overlap`](#knob-tp_split_overlap) — **Default:** 7. **Meaning:** Overlap layer halves and wide exchange with a chosen split schedule. Check: `knob_tp_split_overlap` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_split_least_rows`](#knob-tp_split_least_rows) — **Default:** 0. **Meaning:** Run chunks below this row count whole instead of split. Check: `knob_tp_split_least_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_launch_trims`](#knob-tp_launch_trims) — **Default:** 1. **Meaning:** Remove redundant ring-input copies and wo_b quantization launches. Check: `knob_tp_launch_trims` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_full_split_overlap`](#knob-tp_full_split_overlap) — **Default:** 0. **Meaning:** Override full-program split overlap; zero inherits tp_split_overlap. Check: `knob_tp_full_split_overlap` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_replicated_embedding`](#knob-tp_replicated_embedding) — **Default:** 1. **Meaning:** Embed input rows on every rank instead of handing off rank zero. Check: `knob_tp_replicated_embedding` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_ring_column_owners`](#knob-tp_ring_column_owners) — **Default:** 1. **Meaning:** Assign BF16 ring ownership by column quarter for batch-independent addition order. Check: `knob_tp_ring_column_owners` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--step_graphs`](#knob-step_graphs) — **Default:** 0. **Meaning:** Capture/replay prepared CUDA step graphs instead of eager launches. Check: `knob_step_graphs` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--pipeline_steps`](#knob-pipeline_steps) — **Default:** Raw default1 on all placements. Qualifying original PP5 serve with session_slots>1, no BATCH and DSpark/speculative2 resolves omitted depth to5. Explicit depth is retained; the shipped original multiple-user decode axis explicitly selects8. Generate and single-slot raw defaults stay1. **Meaning:** Bound ordinary step pipeline depth. Check: `knob_pipeline_steps` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Precision, exactness, head and dense forms

- [`--tp_head_shards`](#knob-tp_head_shards) — **Default:** 0. **Meaning:** Shard target head products and sampling over TP4_EXL3 ranks. Check: `knob_tp_head_shards` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_head_stream`](#knob-tp_head_stream) — **Default:** 0. **Meaning:** Use the dense-stream target head products. Check: `knob_tp_head_stream` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_head_mma`](#knob-tp_head_mma) — **Default:** 0. **Meaning:** Use tensor-core target-head products on the BF16 recipe. Check: `knob_tp_head_mma` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_trunk_mma`](#knob-tp_trunk_mma) — **Default:** 0. **Meaning:** Use tensor-core BF16-row trunk products on the BF16 recipe. Check: `knob_tp_trunk_mma` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_trunk_mma_fused`](#knob-tp_trunk_mma_fused) — **Default:** 0. **Meaning:** Sum trunk MMA split partials inside the 64-row kernel. Check: `knob_tp_trunk_mma_fused` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_trunk_mma_thin_splits`](#knob-tp_trunk_mma_thin_splits) — **Default:** 0. **Meaning:** Give 64-row trunk MMA products the THIN split geometry. Check: `knob_tp_trunk_mma_thin_splits` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--dense_skinny_bf16`](#knob-dense_skinny_bf16) — **Default:** 0. **Meaning:** Tile BF16 skinny dense outputs per block above 48 rows; zero retains wide form. Check: `knob_dense_skinny_bf16` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--dense_skinny_f32`](#knob-dense_skinny_f32) — **Default:** 0. **Meaning:** Tile FP32-weight skinny dense outputs per block above 48 rows. Check: `knob_dense_skinny_f32` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--dense_staged`](#knob-dense_staged) — **Default:** 16. **Meaning:** Stage wide dense-product inputs with a selectable output-block width. Check: `knob_dense_staged` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_front_branch`](#knob-tp_front_branch) — **Default:** 0. **Meaning:** Branch THIN attention mix coefficients; mode two also branches FFN coefficients. Check: `knob_tp_front_branch` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_decode_round_once`](#knob-tp_decode_round_once) — **Default:** 0. **Meaning:** Sum published BF16 target partials in ascending-rank FP32 and round the result once. Check: `knob_tp_decode_round_once` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_wire_bf16`](#knob-tp_wire_bf16) — **Default:** 0. **Meaning:** Round TP transport partials/sums to BF16; zero retains FP32 wire. Check: `knob_tp_wire_bf16` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--linear_profile`](#knob-linear_profile) — **Default:** 0. **Meaning:** Choose precision or speed linear-product preparation. Check: `knob_linear_profile` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--linear_order`](#knob-linear_order) — **Default:** 1. **Meaning:** Choose fixed (zero) or source reference (one) product accumulation order. Check: `knob_linear_order` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--official_hc_rows`](#knob-official_hc_rows) — **Default:** 0. **Meaning:** Select pinned HC arithmetic with an explicit reference call shape. Check: `knob_official_hc_rows` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--numerics`](#knob-numerics) — **Default:** exact **Meaning:** Choose admitted startup numerical policy. Check: `knob_numerics` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--placement`](#knob-placement) — **Default:** pp5 **Meaning:** Choose prepared weight/state placement. Check: `knob_placement` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--cmp_order`](#knob-cmp_order) — **Default:** reference **Meaning:** Choose CMP trunk/expert sum order. Check: `knob_cmp_order` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Cache, prefix reuse and snapshots

- [`--prefix_reuse`](#knob-prefix_reuse) — **Default:** 1. **Meaning:** Reuse eligible cached prompt prefixes instead of prefilling cold. Check: `knob_prefix_reuse` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--prefix_reuse_exact`](#knob-prefix_reuse_exact) — **Default:** 0. **Meaning:** Admit same-length cold-provenance prompt prefixes through repeat or cold-plan boundaries. Check: `knob_prefix_reuse_exact` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--cache_scope`](#knob-cache_scope) — **Default:** 0. **Meaning:** Choose the namespace used for tokenizer, slot and snapshot reuse. Check: `knob_cache_scope` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--session_grace_seconds`](#knob-session_grace_seconds) — **Default:** 30. **Meaning:** Prefer retaining a recently finished namespace when choosing an idle slot. Check: `knob_session_grace_seconds` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--session_save_tokens`](#knob-session_save_tokens) — **Default:** 1024 when omitted; explicit CLI0 is refused (server API zero-initialization is a separate convention). **Meaning:** Require this many processed tokens absent from a snapshot before saving. Check: `knob_session_save_tokens` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--session_load_tokens`](#knob-session_load_tokens) — **Default:** 256 when omitted; explicit CLI0 is refused. **Meaning:** Require this much reusable-token gain before loading a snapshot. Check: `knob_session_load_tokens` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--session_idle_save_seconds`](#knob-session_idle_save_seconds) — **Default:** 120 when omitted; explicit CLI0 is refused. **Meaning:** Wait this long without work before starting an eligible idle snapshot save. Check: `knob_session_idle_save_seconds` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--session_directory`](#knob-session_directory) — **Default:** none **Meaning:** Persist eligible session snapshots in this directory (requires prefix_reuse=1). Check: `knob_session_directory` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--session_fork`](#knob-session_fork) — **Default:** 0. **Meaning:** Copy an eligible same-owner prefix into an independent child session slot. Check: `fork` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### HTTP server, authentication and queues

- [`--queue_depth`](#knob-queue_depth) — **Default:** 16. **Meaning:** Bound all waiting server requests. Check: `knob_queue_depth` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--user_queue_depth`](#knob-user_queue_depth) — **Default:** Omitted: automatic max(1,queue_depth/2), represented internally by0. Explicit CLI0 is refused; accepted explicit values are1..queue_depth. **Meaning:** Bound waiting requests per API user. Check: `knob_user_queue_depth` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--port`](#knob-port) — **Default:** 8321. **Meaning:** Listen on this TCP port. Check: `knob_port` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--bind_address`](#knob-bind_address) — **Default:** 127.0.0.1 **Meaning:** Bind the server listener. Check: `knob_bind_address` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--key_file`](#knob-key_file) — **Default:** none **Meaning:** Read API-user authentication records at startup. Check: `knob_key_file` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--stats_users`](#knob-stats_users) — **Default:** all authenticated users; open stats when auth disabled **Meaning:** Authorize named users to access the statistics endpoint. Check: `knob_stats_users` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--model_alias`](#knob-model_alias) — **Default:** none **Meaning:** Add a request/listing model alias. Check: `knob_model_alias` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Sampling, output and template controls

- [`--generated_tokens`](#knob-generated_tokens) — **Default:** Generate64; serve4294967295 (effective context still bounds it). **Meaning:** Cap generate output; supply the omitted-request output default in serve. Explicit request output limits replace this default within context capacity. Check: `knob_generated_tokens` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--seed`](#knob-seed) — **Default:** Generate0; serve draws a request seed when neither request nor server explicitly seeds. CLI value0 explicitly seeds, not random. **Meaning:** Set sampler seed; omitted serving requests draw a seed. Check: `knob_seed` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--reasoning_effort`](#knob-reasoning_effort) — **Default:** 0. **Meaning:** Set template reasoning effort from zero (off) through 100. Check: `knob_reasoning_effort` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--temperature`](#knob-temperature) — **Default:** 0 **Meaning:** Set sampling temperature; zero is greedy. Check: `knob_temperature` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--top_p`](#knob-top_p) — **Default:** 1 **Meaning:** Set nucleus probability cutoff. Check: `knob_top_p` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_usage`](#knob-draft_usage) — **Default:** compact **Meaning:** Choose draft statistics detail in responses. Check: `knob_draft_usage` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Logging, profiling and diagnostic dumps

- [`--live_stats`](#knob-live_stats) — **Default:** 0. **Meaning:** Print server counters at this interval in milliseconds; zero disables them. Check: `knob_live_stats` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--draft_trace`](#knob-draft_trace) — **Default:** 0. **Meaning:** Print per-round draft anchors and proposals for diagnostics. Check: `knob_draft_trace` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--dump_draft_ring`](#knob-dump_draft_ring) — **Default:** none **Meaning:** Shared parser accepts the path; serve performs no dump (generate-only effect). Check: `serve_flag_dump_draft_ring` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--live_stats_format`](#knob-live_stats_format) — **Default:** text **Meaning:** Choose live server statistics format. Check: `knob_live_stats_format` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--help-scheduling`](#knob-help-scheduling) — **Default:** not requested **Meaning:** Print scheduling examples and measured pause/TTFT tradeoffs without opening a model. Check: `scheduling` — tcb82 help printed; no model/backend opened. Receipt: examples.json.
### Original weights and PRO arithmetic

- [`--pro_emulation`](#knob-pro_emulation) — **Default:** 0. **Meaning:** Emulate class-80 sums on the PRO; zero uses class-120 native arithmetic. Check: `knob_pro_emulation` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_code_wire`](#knob-tp_code_wire) — **Default:** 0. **Meaning:** Transport original quantized codes/scales losslessly across TP ranks. Check: `knob_tp_code_wire` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--original_dense_staged`](#knob-original_dense_staged) — **Default:** 0. **Meaning:** Stage original TP4 BF16/FP32 dense products. Check: `knob_original_dense_staged` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--original_wide_experts`](#knob-original_wide_experts) — **Default:** 1 only for compatible exact TP4_PRO wide64..1024/reference/native-PRO/FP32-wire recipes with no TC-MoE, BATCH, code/owner path, EXL3 wide entry, split/full-split or original_chunk_core.0 elsewhere; explicit0 retains slabs. **Meaning:** Run original TP4 expert jobs over complete chunks with lossless slab gathers. Check: `knob_original_wide_experts` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--original_chunk_core`](#knob-original_chunk_core) — **Default:** 0. **Meaning:** Use original TP4 full-head attention core on whole eight-row slabs. Check: `knob_original_chunk_core` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--original_batch`](#knob-original_batch) — **Default:** 0. **Meaning:** Admit the fixed original TP4 BATCH48/eight-verifier/eight-slot pilot. Check: `knob_original_batch` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--tp_pipeline_overlap`](#knob-tp_pipeline_overlap) — **Default:** 0. **Meaning:** Run the CMP prefix and native PRO tail on disjoint workers. Check: `knob_tp_pipeline_overlap` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
### Axes and branch feature boundaries

- [`--category`](#knob-category) — **Default:** unset; no axis default selected **Meaning:** Choose the workload objective; balanced requires the full mixed-user benchmark. Check: `axes_c1` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--exactness`](#knob-exactness) — **Default:** unset; no axis default selected **Meaning:** Choose reference arithmetic, current reproducible arithmetic, or qualified bounded-loss arithmetic. Check: `axes_c1` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--concurrency`](#knob-concurrency) — **Default:** unset; no axis default selected **Meaning:** Choose a C1 recipe, a C2-C8 recipe, or a measured compromise. Check: `axes_c1` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--focus`](#knob-focus) — **Default:** unset; no axis default selected **Meaning:** Choose the optimization objective within the selected numerical tier. Check: `axes_c1` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

Original exact is limited to the capacity-512 serial M1 recipe. EXL3 exact, fast, vision and full mixed-envelope holds still apply. Order3 is opt-in and refuses exact-prefix modes1/2, stable1 and fork1. Forking additionally needs authenticated hard-owner equality, stable1 donor provenance, independent slots and immutable cold-plan capability; stable0 follows the legacy cold fallback. Snapshot format remains v4; main has no v5 format-selection flag.

### Additional main options in tcb82

- [`--prefill_prompt_chunks`](#knob-prefill_prompt_chunks) — **Default:** 0 on all placements. **Meaning:** Keep chunk-ready prompt owners out of decode BATCH; the measured first0/order0 guard trades code C8 aggregate −1.76% for the recorded mixed-output identity. Check: `knob_prefill_prompt_chunks` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--official_reference`](#knob-official_reference) — **Default:** 0 on raw placements; original exact inserts1. **Meaning:** Enable the bounded PP THIN serial M1 reference runtime, one session and capacity512; production T=1/top_p=1 is explicit, stochastic source ID identity is not claimed. Check: `knob_official_reference` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.
- [`--ignore_eos`](#knob-ignore_eos) — **Default:** 0 on all placements. **Meaning:** Continue after EOS in non-speculative THIN fixed-horizon checks; ordinary serving retains stopping. Check: `knob_ignore_eos` — accepted by the tcb82 resolver/parser; stopped before model/socket/backend I/O. Receipt: examples.json.

`--prompt_token_ids` is generate-only and therefore does not appear as a serve flag.
