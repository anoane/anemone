<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Axis table

Generated from `tools/cli/axes.json` by `tools/cli/generate_axes.py`.
Registry SHA256: `3e7af62a7895a006bd23e5aff1fb96b67391dda01fe5024b1fc5d03ddf2196d5`.

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
- Each knob belongs to one axis per split (generate_axes.py refuses a second owner), except a recipe's yielded fallback: a yielded knob is set by its recipe and exactly one other axis, whose value wins where it gives one (stage 3 review D3). Session acceptance and the other measured modifiers enter only after their per-split confirmation: focus=decode carries arm 1 (session acceptance 6) on PP5 (r1m) and TP4_PRO (r5s); every other Original non-exact row takes the CLI value 0. No-axis defaults and exact/EXL3 contracts are unchanged.
- --load_piece_mib (the load's read size: the PP piece and staging slot, the TP host arena's chunk) changes load time only, with the same bits. It belongs to the focus axis on every split with every value empty (balanced, ttft, decode and prefill set nothing): its measured value comes from the storage profile's load section (anemone autotune --stage=load), and the knob wins over the profile, the profile over the default (0: 16 MiB).

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
| concurrency | multiple | `original_pp_depth8` without `--prefix_reuse`, `--session_slots`; yields `--draft_block`, `--draft_positions`, `--draft_policy`, `--draft_session_acceptance` where another set gives them | PP5 at pipeline depth 8 with 8 serving slots: S10 gxf_s10_original_multiple, C1-C8 in three genres, against depth 5 in S07 (recipe original_pp_depth8). --prefix_reuse=0 pinned the measurement protocol and is dropped (r6a: no cell loses beyond spread). The recipe yields its draft block, depth, policy and session acceptance (block 5, depth 5, fixed policy 0, acceptance 0) to category=code and focus=decode: its measured values run wherever those sets give none. The fixed policy is no CLI default (the Original placements default to the cost-priced depth), so a dropped policy would change every other row. The recipe's --session_slots=8 is dropped too (9 Nov): the row serves as many sessions as fit at the context (the raw default max; 5 at 1M), at pipeline depth 8; unmeasured above and below 8 until U6. | PP5 | 8 (generate 2) | default |
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
on the `axes from` line.

Caps and floors:

- cap: `--rows` bounds `--draft_block`, `--draft_positions`: --rows caps the proposals verified per round at rows - 1 (commands.c). A draft block or depth that another source of the row sets above that cap is clamped to it: the extra block positions cost time and verify nothing (r5c on TP4_PRO: depth 7 on rows 6 with arm 1 lost 10.2 % to r5s, rows 6 with arm 1 at depth 5, 232.3 -> 208.5 tok/s code C1). Bounds act on the final values, explicit ones included, and change only inserted knobs (stage 3 review D1): an explicit --rows=6 keeps the clamp, --rows=8 lifts it; an explicit bounded knob is never altered and is validated as usual. A cap never lowers a knob below the least its parser takes (--draft_block 5, --draft_positions 1), also where the final speculation is off and the clamped draft knobs are inert, noted inactive (stage 3 review D5 with D2: a raw --rows=1 --speculative=0 parses).
- floor: `--tp_draft_shards`=`1` (under `--batch_draft_rows`) raises `--tp_draft_layers`: Sharded BATCH draft rows require draft layers (model_v41.c: batch_draft_rows with draft_shards needs draft_layers). Where the final --tp_draft_shards=1 and --batch_draft_rows are composed or given (TP4_EXL3 concurrency=multiple), an inserted --tp_draft_layers is raised to 1, the value every BATCH tuple ran with before the per-split tables. Bounds act on the final values (stage 3 review D1): an explicit --tp_draft_shards=1 keeps the raise, --tp_draft_shards=0 or --batch_draft_rows=0 removes it; an explicit --tp_draft_layers is never altered and is validated as usual.

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
