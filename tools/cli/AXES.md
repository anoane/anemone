<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Axis table

Generated from `tools/cli/axes.json` by `tools/cli/generate_axes.py`.
Registry SHA256: `257eaa9096da08d7acd604cfcfb263da12d346fc5d0edc2792dae572fe685f76`.

The four selectors apply to generate and serve. There are no selectable names or aliases.

| Axis | Values | Meaning |
| --- | --- | --- |
| `--category` | code / prose / structured / vision / balanced | Workload kind; balanced is a mixed code/prose/structured measurement, with code winning ties. |
| `--exactness` | exact / balanced / fast | Reference arithmetic, current arithmetic, or a measured bounded-loss alternative. Exact uses official-exact logits unchanged at every temperature/top_p; defaults are temperature=0, top_p=1. Draw identity requires T=0 or matching RNG/sampler order; ours do not match the reference stochastic RNG/order. |
| `--concurrency` | single / multiple / balanced | C1/THIN, C2-C8/BATCH, or one measured compromise across both. |
| `--focus` | balanced / ttft / decode / prefill | Measured compromise, first-token latency, decode rate, or prefill rate. |

| Exactness tier | Error class | Current admission |
| --- | --- | --- |
| exact | Original weights: official-reference arithmetic on the admitted PP THIN serial M1 scope. EXL3: reference arithmetic (EXL3: not official-exact). | Original serial M1 rule below; EXL3 remains held. |
| balanced | Current HC/current forms; no added lossy setting. Admitted single-user tuples use canonical fixed0: cost1 failed selected-F32 identity despite equal IDs. S06 EXL3 cached-eight target smoke has KLmean0.224/RMS0.966/top1=7/8 versus the pinned hybrid source; not corpus quality or official equality. | Current-arithmetic candidates; qualification scope below. |
| fast | Measured bounded-loss arithmetic only; opt-in. Each selected lossy knob must carry paired numerical and whole-workload receipts. | No candidate has complete paired KL, logit deviation, held-out PPL, T=0 agreement and whole-workload speed evidence within an admitted error budget. Unqualified or excessive-loss settings remain raw CLI knobs only. |

Original weights and EXL3 have separate admission records for every full tuple.
Original official-reference arithmetic is scoped to its admitted serial M1 recipe. EXL3 uses the label
**reference arithmetic (EXL3: not official-exact)** when compatible forms are qualified.

- With no axes selected, existing CLI behavior is unchanged. Unset axes have no implicit owner-selected defaults.
- Original versus EXL3 is a weight-family admission condition, not a fifth preset axis. The runtime uses explicit placement or the existing PP5 default.
- Explicit low-level knobs override inserted settings; ordinary validation still refuses invalid combinations. Overrides change the measured recipe.
- All balanced choices retain current HC/current forms and add no new lossy arithmetic. A separate R component result is not a complete S official-error measurement.
- Category balanced needs the full mixed-user benchmark. The complete EXL3 all-balanced development candidate is held after its eight-slot memory refusal; other balanced-concurrency rows remain held. No axes are selected by default.
- Balanced-code selection must maximize code among feasible choices: non-code margins and no-worse default Fraser median/worst/pause comparisons require same-epoch measurements.
- The table contains full tuples only. The parser handles partial selections without inventing an axis default.
- Seven-bucket entries are targets for the recorded scope; every unlisted bucket and every missing accuracy/capacity gate remains unqualified.
- Measured drafting priors and BATCH core mode are explicit in the table. A later raw CLI default change does not silently change these recorded recipes; explicit user controls still win.
- Legacy original TP4_PRO and PP5 routes existed in main7d86c45. The current table also pins newer O/HC controls to zero and therefore requires the merged main75/S parser. PP5 layouts remain single argv items with embedded spaces; generated settings use newline separators, never shell parsing. S07 cold-prefix measurements are source-scoped separately from inherited O warm results. Generate keeps its one-slot default and pipeline2; serve selects one serial TP slot or eight PP5 slots at measured depth8.
- Selecting only --exactness=exact on original weights resolves to the supported text, single-session PP THIN serial M1 recipe. Explicit vision or multiple/balanced concurrency remains held. Recipe arithmetic/placement overrides must retain the admitted exact values; balanced-axis override behavior is unchanged.

## Knob values

### batch_fixed

Full-depth code-first BATCH with the merged pool/deferred-slot core3; cold identity and current-source serving qualification are mandatory.

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

Measured frozen S10/e4c188e (base5029986), ordinary binary 8d9b8b05. Original pack, native PRO and four CMPs at 180 W, context 65536, 1 slot(s), prefix reuse off. C1 rates are per stream; C2/C4/C8 rates are aggregates. code C1/C2/C4/C8: 207.123/unmeasured/unmeasured/unmeasured tok/s. MiaAI median TTFT: 100.60/unmeasured/unmeasured/unmeasured ms (small decode prompts, not Fraser). prose C1/C2/C4/C8: 150.795/unmeasured/unmeasured/unmeasured tok/s. MiaAI median TTFT: 78.95/unmeasured/unmeasured/unmeasured ms (small decode prompts, not Fraser). structured C1/C2/C4/C8: 288.582/unmeasured/unmeasured/unmeasured tok/s. MiaAI median TTFT: 73.30/unmeasured/unmeasured/unmeasured ms (small decode prompts, not Fraser). Full selected 8/256 records and 15 MiaAI request/text/native-count/finish signatures pass. Minimum CMP free at readiness: 5503 MiB. Startup expansion, artifact/source pins, original-pack stat and device/cap restoration pass. Five C1 batches and three batches at each requested C2/C4/C8 cell. No own C-build/host-test overlap. Sequential observations, no causal or global-optimum claim. Cold prefill, Fraser, mixed traffic, long-prefill pauses, 1M fit and full-context work were not requested. Current HC/current forms; no added lossy arithmetic, not official-exact. Source: evidence/gxf_s10_original_single_strict_s10_v1.json. Historical original numerical diagnostic, O xo03/xo16: eight aligned cached-one teacher inputs, 1,034,240 F32 logits. PP5 and TP4_PRO captures are equal; against R original parent PRO2 with unchanged sparse core CMP0, mean/max KL 0.0407172/0.144973 nats, raw RMS/max 0.568391/4.276049, teacher top1 8/8. Source75 baseline replay preserves those captures. Not untouched all-PRO, corpus PPL, generated agreement or current-S10 full-vocabulary distance. See evidence/xo16_serial_reference_distance_v1.json and RND_O/S 5Nov09:26.

### original_pp_depth8

Original-weight PP5 placement at measured depth8 for all supported multiple-user decode categories; eight serving slots.

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

## Quality budget

Proposed conservative incremental screen, not an established quality guarantee or an owner-selected default. No lossy candidate has passed; exactness=fast remains held.

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

The map contains 360 weight-family records across 180 full tuples.
A held row never inserts its displayed candidate knobs.

| Category | Exactness | Concurrency | Focus | EXL3 knob set / admission | Original knob set / admission | Seven-bucket targets |
| --- | --- | --- | --- | --- | --- | --- |
| code | exact | single | balanced | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, decode_accuracy |
| code | exact | single | ttft | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, prefill_accuracy |
| code | exact | single | decode | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, decode_accuracy |
| code | exact | single | prefill | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, prefill_accuracy |
| code | exact | multiple | balanced | held qualification candidate | held original | exact, throughput_accuracy |
| code | exact | multiple | ttft | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| code | exact | multiple | decode | held qualification candidate | held original | exact, throughput_accuracy |
| code | exact | multiple | prefill | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| code | exact | balanced | balanced | held qualification candidate | held original | exact, throughput_accuracy |
| code | exact | balanced | ttft | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| code | exact | balanced | decode | held qualification candidate | held original | exact, throughput_accuracy |
| code | exact | balanced | prefill | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| code | balanced | single | balanced | `thin_fixed`; draft_only candidate | held original | decode_accuracy, decode_fast |
| code | balanced | single | ttft | `thin_fixed`; draft_only candidate | held original | prefill_accuracy, prefill_fast |
| code | balanced | single | decode | `thin_full_layers`; draft_only candidate | `original_tp_serial2`; draft_only candidate; see source-scoped GPU observations | decode_accuracy, decode_fast |
| code | balanced | single | prefill | `thin_fixed`; draft_only candidate | held original | prefill_accuracy, prefill_fast |
| code | balanced | multiple | balanced | held qualification candidate; candidate `batch_session` | held original | throughput_accuracy, throughput_fast |
| code | balanced | multiple | ttft | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| code | balanced | multiple | decode | `batch_fixed`; draft_only candidate | `original_pp_depth8`; draft_only candidate; see source-scoped GPU observations | throughput_accuracy, throughput_fast |
| code | balanced | multiple | prefill | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| code | balanced | balanced | balanced | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast |
| code | balanced | balanced | ttft | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| code | balanced | balanced | decode | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast |
| code | balanced | balanced | prefill | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| code | fast | single | balanced | held tier | held tier | decode_fast |
| code | fast | single | ttft | held tier | held tier | prefill_fast |
| code | fast | single | decode | held tier | held tier | decode_fast |
| code | fast | single | prefill | held tier | held tier | prefill_fast |
| code | fast | multiple | balanced | held tier | held tier | throughput_fast |
| code | fast | multiple | ttft | held tier | held tier | throughput_fast, prefill_fast |
| code | fast | multiple | decode | held tier | held tier | throughput_fast |
| code | fast | multiple | prefill | held tier | held tier | throughput_fast, prefill_fast |
| code | fast | balanced | balanced | held tier | held tier | throughput_fast |
| code | fast | balanced | ttft | held tier | held tier | throughput_fast, prefill_fast |
| code | fast | balanced | decode | held tier | held tier | throughput_fast |
| code | fast | balanced | prefill | held tier | held tier | throughput_fast, prefill_fast |
| prose | exact | single | balanced | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, decode_accuracy |
| prose | exact | single | ttft | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, prefill_accuracy |
| prose | exact | single | decode | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, decode_accuracy |
| prose | exact | single | prefill | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, prefill_accuracy |
| prose | exact | multiple | balanced | held qualification candidate | held original | exact, throughput_accuracy |
| prose | exact | multiple | ttft | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| prose | exact | multiple | decode | held qualification candidate | held original | exact, throughput_accuracy |
| prose | exact | multiple | prefill | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| prose | exact | balanced | balanced | held qualification candidate | held original | exact, throughput_accuracy |
| prose | exact | balanced | ttft | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| prose | exact | balanced | decode | held qualification candidate | held original | exact, throughput_accuracy |
| prose | exact | balanced | prefill | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| prose | balanced | single | balanced | `thin_fixed`; draft_only candidate | held original | decode_accuracy, decode_fast |
| prose | balanced | single | ttft | `thin_fixed`; draft_only candidate | held original | prefill_accuracy, prefill_fast |
| prose | balanced | single | decode | `thin_fixed`; draft_only candidate | `original_tp_serial2`; draft_only candidate; see source-scoped GPU observations | decode_accuracy, decode_fast |
| prose | balanced | single | prefill | `thin_fixed`; draft_only candidate | held original | prefill_accuracy, prefill_fast |
| prose | balanced | multiple | balanced | held qualification candidate; candidate `batch_adaptive` | held original | throughput_accuracy, throughput_fast |
| prose | balanced | multiple | ttft | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| prose | balanced | multiple | decode | `batch_adaptive`; same_bits candidate | `original_pp_depth8`; draft_only candidate; see source-scoped GPU observations | throughput_accuracy, throughput_fast |
| prose | balanced | multiple | prefill | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| prose | balanced | balanced | balanced | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast |
| prose | balanced | balanced | ttft | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| prose | balanced | balanced | decode | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast |
| prose | balanced | balanced | prefill | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| prose | fast | single | balanced | held tier | held tier | decode_fast |
| prose | fast | single | ttft | held tier | held tier | prefill_fast |
| prose | fast | single | decode | held tier | held tier | decode_fast |
| prose | fast | single | prefill | held tier | held tier | prefill_fast |
| prose | fast | multiple | balanced | held tier | held tier | throughput_fast |
| prose | fast | multiple | ttft | held tier | held tier | throughput_fast, prefill_fast |
| prose | fast | multiple | decode | held tier | held tier | throughput_fast |
| prose | fast | multiple | prefill | held tier | held tier | throughput_fast, prefill_fast |
| prose | fast | balanced | balanced | held tier | held tier | throughput_fast |
| prose | fast | balanced | ttft | held tier | held tier | throughput_fast, prefill_fast |
| prose | fast | balanced | decode | held tier | held tier | throughput_fast |
| prose | fast | balanced | prefill | held tier | held tier | throughput_fast, prefill_fast |
| structured | exact | single | balanced | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, decode_accuracy |
| structured | exact | single | ttft | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, prefill_accuracy |
| structured | exact | single | decode | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, decode_accuracy |
| structured | exact | single | prefill | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, prefill_accuracy |
| structured | exact | multiple | balanced | held qualification candidate | held original | exact, throughput_accuracy |
| structured | exact | multiple | ttft | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| structured | exact | multiple | decode | held qualification candidate | held original | exact, throughput_accuracy |
| structured | exact | multiple | prefill | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| structured | exact | balanced | balanced | held qualification candidate | held original | exact, throughput_accuracy |
| structured | exact | balanced | ttft | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| structured | exact | balanced | decode | held qualification candidate | held original | exact, throughput_accuracy |
| structured | exact | balanced | prefill | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| structured | balanced | single | balanced | `thin_fixed`; draft_only candidate | held original | decode_accuracy, decode_fast |
| structured | balanced | single | ttft | `thin_fixed`; draft_only candidate | held original | prefill_accuracy, prefill_fast |
| structured | balanced | single | decode | `thin_fixed`; draft_only candidate | `original_tp_serial2`; draft_only candidate; see source-scoped GPU observations | decode_accuracy, decode_fast |
| structured | balanced | single | prefill | `thin_fixed`; draft_only candidate | held original | prefill_accuracy, prefill_fast |
| structured | balanced | multiple | balanced | held qualification candidate; candidate `batch_adaptive` | held original | throughput_accuracy, throughput_fast |
| structured | balanced | multiple | ttft | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| structured | balanced | multiple | decode | `batch_adaptive`; same_bits candidate | `original_pp_depth8`; draft_only candidate; see source-scoped GPU observations | throughput_accuracy, throughput_fast |
| structured | balanced | multiple | prefill | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| structured | balanced | balanced | balanced | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast |
| structured | balanced | balanced | ttft | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| structured | balanced | balanced | decode | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast |
| structured | balanced | balanced | prefill | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| structured | fast | single | balanced | held tier | held tier | decode_fast |
| structured | fast | single | ttft | held tier | held tier | prefill_fast |
| structured | fast | single | decode | held tier | held tier | decode_fast |
| structured | fast | single | prefill | held tier | held tier | prefill_fast |
| structured | fast | multiple | balanced | held tier | held tier | throughput_fast |
| structured | fast | multiple | ttft | held tier | held tier | throughput_fast, prefill_fast |
| structured | fast | multiple | decode | held tier | held tier | throughput_fast |
| structured | fast | multiple | prefill | held tier | held tier | throughput_fast, prefill_fast |
| structured | fast | balanced | balanced | held tier | held tier | throughput_fast |
| structured | fast | balanced | ttft | held tier | held tier | throughput_fast, prefill_fast |
| structured | fast | balanced | decode | held tier | held tier | throughput_fast |
| structured | fast | balanced | prefill | held tier | held tier | throughput_fast, prefill_fast |
| vision | exact | single | balanced | held qualification candidate | held vision | exact, decode_accuracy |
| vision | exact | single | ttft | held qualification candidate | held vision | exact, prefill_accuracy |
| vision | exact | single | decode | held qualification candidate | held vision | exact, decode_accuracy |
| vision | exact | single | prefill | held qualification candidate | held vision | exact, prefill_accuracy |
| vision | exact | multiple | balanced | held qualification candidate | held vision | exact, throughput_accuracy |
| vision | exact | multiple | ttft | held qualification candidate | held vision | exact, throughput_accuracy, prefill_accuracy |
| vision | exact | multiple | decode | held qualification candidate | held vision | exact, throughput_accuracy |
| vision | exact | multiple | prefill | held qualification candidate | held vision | exact, throughput_accuracy, prefill_accuracy |
| vision | exact | balanced | balanced | held qualification candidate | held vision | exact, throughput_accuracy |
| vision | exact | balanced | ttft | held qualification candidate | held vision | exact, throughput_accuracy, prefill_accuracy |
| vision | exact | balanced | decode | held qualification candidate | held vision | exact, throughput_accuracy |
| vision | exact | balanced | prefill | held qualification candidate | held vision | exact, throughput_accuracy, prefill_accuracy |
| vision | balanced | single | balanced | held vision | held vision | decode_accuracy, decode_fast |
| vision | balanced | single | ttft | held vision | held vision | prefill_accuracy, prefill_fast |
| vision | balanced | single | decode | held vision | held vision | decode_accuracy, decode_fast |
| vision | balanced | single | prefill | held vision | held vision | prefill_accuracy, prefill_fast |
| vision | balanced | multiple | balanced | held vision | held vision | throughput_accuracy, throughput_fast |
| vision | balanced | multiple | ttft | held vision | held vision | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| vision | balanced | multiple | decode | held vision | held vision | throughput_accuracy, throughput_fast |
| vision | balanced | multiple | prefill | held vision | held vision | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| vision | balanced | balanced | balanced | held vision | held vision | throughput_accuracy, throughput_fast |
| vision | balanced | balanced | ttft | held vision | held vision | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| vision | balanced | balanced | decode | held vision | held vision | throughput_accuracy, throughput_fast |
| vision | balanced | balanced | prefill | held vision | held vision | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| vision | fast | single | balanced | held tier | held tier | decode_fast |
| vision | fast | single | ttft | held tier | held tier | prefill_fast |
| vision | fast | single | decode | held tier | held tier | decode_fast |
| vision | fast | single | prefill | held tier | held tier | prefill_fast |
| vision | fast | multiple | balanced | held tier | held tier | throughput_fast |
| vision | fast | multiple | ttft | held tier | held tier | throughput_fast, prefill_fast |
| vision | fast | multiple | decode | held tier | held tier | throughput_fast |
| vision | fast | multiple | prefill | held tier | held tier | throughput_fast, prefill_fast |
| vision | fast | balanced | balanced | held tier | held tier | throughput_fast |
| vision | fast | balanced | ttft | held tier | held tier | throughput_fast, prefill_fast |
| vision | fast | balanced | decode | held tier | held tier | throughput_fast |
| vision | fast | balanced | prefill | held tier | held tier | throughput_fast, prefill_fast |
| balanced | exact | single | balanced | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, decode_accuracy |
| balanced | exact | single | ttft | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, prefill_accuracy |
| balanced | exact | single | decode | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, decode_accuracy |
| balanced | exact | single | prefill | held qualification candidate | `original_reference_serial_m1`; reference_arithmetic candidate; see source-scoped GPU observations | exact, prefill_accuracy |
| balanced | exact | multiple | balanced | held qualification candidate | held original | exact, throughput_accuracy |
| balanced | exact | multiple | ttft | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| balanced | exact | multiple | decode | held qualification candidate | held original | exact, throughput_accuracy |
| balanced | exact | multiple | prefill | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| balanced | exact | balanced | balanced | held qualification candidate | held original | exact, throughput_accuracy |
| balanced | exact | balanced | ttft | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| balanced | exact | balanced | decode | held qualification candidate | held original | exact, throughput_accuracy |
| balanced | exact | balanced | prefill | held qualification candidate | held original | exact, throughput_accuracy, prefill_accuracy |
| balanced | balanced | single | balanced | `thin_fixed`; draft_only candidate | held original | decode_accuracy, decode_fast |
| balanced | balanced | single | ttft | `thin_fixed`; draft_only candidate | held original | prefill_accuracy, prefill_fast |
| balanced | balanced | single | decode | `thin_fixed`; draft_only candidate | held original | decode_accuracy, decode_fast |
| balanced | balanced | single | prefill | `thin_fixed`; draft_only candidate | held original | prefill_accuracy, prefill_fast |
| balanced | balanced | multiple | balanced | held qualification candidate; candidate `batch_adaptive` | held original | throughput_accuracy, throughput_fast |
| balanced | balanced | multiple | ttft | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| balanced | balanced | multiple | decode | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast |
| balanced | balanced | multiple | prefill | `batch_adaptive`; same_bits candidate | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| balanced | balanced | balanced | balanced | held qualification candidate; candidate `batch_compromise` | held original | throughput_accuracy, throughput_fast |
| balanced | balanced | balanced | ttft | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| balanced | balanced | balanced | decode | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast |
| balanced | balanced | balanced | prefill | held compromise; candidate `batch_session` | held original | throughput_accuracy, throughput_fast, prefill_accuracy, prefill_fast |
| balanced | fast | single | balanced | held tier | held tier | decode_fast |
| balanced | fast | single | ttft | held tier | held tier | prefill_fast |
| balanced | fast | single | decode | held tier | held tier | decode_fast |
| balanced | fast | single | prefill | held tier | held tier | prefill_fast |
| balanced | fast | multiple | balanced | held tier | held tier | throughput_fast |
| balanced | fast | multiple | ttft | held tier | held tier | throughput_fast, prefill_fast |
| balanced | fast | multiple | decode | held tier | held tier | throughput_fast |
| balanced | fast | multiple | prefill | held tier | held tier | throughput_fast, prefill_fast |
| balanced | fast | balanced | balanced | held tier | held tier | throughput_fast |
| balanced | fast | balanced | ttft | held tier | held tier | throughput_fast, prefill_fast |
| balanced | fast | balanced | decode | held tier | held tier | throughput_fast |
| balanced | fast | balanced | prefill | held tier | held tier | throughput_fast, prefill_fast |

Hold conditions:

- fast: No candidate has complete paired KL, logit deviation, held-out PPL, T=0 agreement and whole-workload speed evidence within an admitted error budget. Unqualified or excessive-loss settings remain raw CLI knobs only.
- original: This tuple has no qualified original rule. Legacy serial TP4_PRO and PP5depth2/5/8 exist, and main75 now parses overlap/original BATCH/wide/chunk-core/staged/HC selectors. Their independent evidence does not qualify arbitrary joins, mixed/balanced concurrency or prefill/TTFT envelopes. Query-piece/dense-shard selectors and complete reference forms remain separate dependencies. No placement or weights are silently replaced.
- vision: Vision-axis selection awaits a vision benchmark and applicable runtime/identity evidence.
- compromise: The one-recipe C1/C2-C8 compromise is not qualified. N policy/choice, non-code margins, Fraser median/worst and long-prefill pause comparisons remain required. The all-balanced tuple is the intended general-purpose recommendation after these gates.
