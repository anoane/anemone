<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Benchmark methods and source evidence

The full [comparison matrix](COMPARISON_MATRIX.md) retains all dated measurements, profiles and limits.
The [machine-readable result summary](RESULTS_6NOV.json) contains curated public data. Raw workstation logs, process identities and duplicate historical documents
are maintained separately. Original measurements are not relabelled as new tests.

<a id="runtime-1b073-current"></a>

## Runtime measured on 6 November 2026

The current runtime measurements use engine/help source
`1b0731a36b7482477b96d7a28b27c93e4dd49e3f` and release binary SHA-256
`acdf429da22ae7a39504b897106e8cd56a22c3661bf086f60b5458ba5313f2af`.
The non-golden catalogue contains 86 jobs: 42 attempted, of which 40 pass,
one fails TP4 mixed-output identity and one lacks a qualified capacity cleanup.
The remaining 44 jobs were not attempted. Current golden GPU replay and Pi
coverage remain unmeasured. The matrix distinguishes these results from older
measurements, source reviews and proposed capabilities.

## Curtis decode and SPARK

All six jobs passed native protocol checks, actual guard and parent waits,
owned-process retirement and all-five-card cap restoration. Jobs 401/402 are
balanced TP4 EXL3 / original PP5; 403 is original exact C1. Jobs 404/405/406 are
the corresponding sampled SPARK runs. Original PP5 uses native PRO arithmetic;
its outputs depend on the layers placed there.

The T=0 decode client uses its frozen public count/code/prose prompts and
400-output-token cap, thinking disabled and cold unique cache salts. Each
balanced job has 525 native observations: 414 measured, 108 first-use and three
warmups. Exact has 21: 15 measured, three first-use and three warmups.
The headline below is the median of the first three batch mean per-user
streaming rates, not a pooled token/time rate. Morrow-style whole-run rates use
five batches and include TTFT. Same-job C1 full text/count/finish identity passes
at the admitted T=0 widths. These are not original-reference or full-logit scores.

| Job | Users | Count / code / prose per-user streaming tok/s |
| --- | --- | --- |
| 401 | C1 | 343.06 / 298.92 / 173.60 |
| 401 | C2 | 237.88 / 235.55 / 130.63 |
| 401 | C3 | 196.19 / 158.99 / 99.52 |
| 401 | C4 | 180.85 / 179.62 / 97.30 |
| 401 | C5 | 151.38 / 145.25 / 72.80 |
| 401 | C6 | 130.58 / 109.00 / 68.82 |
| 401 | C7 | 118.65 / 119.83 / 64.80 |
| 401 | C8 | 115.45 / 118.51 / 62.54 |
| 402 | C1 | 248.76 / 211.75 / 121.72 |
| 402 | C2 | 245.17 / 208.18 / 120.60 |
| 402 | C3 | 239.87 / 197.52 / 116.10 |
| 402 | C4 | 203.37 / 167.17 / 97.68 |
| 402 | C5 | 162.36 / 133.00 / 78.01 |
| 402 | C6 | 135.86 / 112.24 / 65.08 |
| 402 | C7 | 116.45 / 95.96 / 55.78 |
| 402 | C8 | 102.63 / 84.83 / 49.13 |
| 403 | C1 | 7.88 / 7.71 / 7.82 |

SPARK runs two reversed passes over eight categories plus a ceiling fixture.
Its headline is the mean of the two pass means over the eight categories;
the ceiling is excluded. There are 201 native observations per balanced job (198 measured and three
warmups) and 21 for exact C1 (18 measured and three warmups). Successful source/input/terminal/cache checks do not establish
sampled cross-request identity, source-RNG equivalence or model quality.

| Job | Users | Aggregate tok/s | Mean decode tok/s per user | Mean TTFT (s) |
| --- | --- | --- | --- | --- |
| 404 | u1 | 165.90 | 192.92 | 0.12 |
| 404 | u2 | 233.58 | 144.38 | 0.23 |
| 404 | u8 | 354.74 | 62.27 | 0.80 |
| 405 | u1 | 122.13 | 133.60 | 0.10 |
| 405 | u2 | 210.91 | 126.86 | 0.17 |
| 405 | u8 | 350.23 | 55.20 | 0.47 |
| 406 | u1 | 5.51 | 8.06 | 10.54 |

Raw per-batch values, full-precision headlines, native usage and sample counts
are retained in each job's `decode-summary.json` or `spark-summary.json`, adapter
verdict and response observations. Jobs 403/406 admit only one session and a
512-token total capacity. No concurrent, long-context, Pi or Q exact result is inferred.

## Q snapshot transfer: functional evidence only

All four Q jobs passed: TP4 and PP5, each with tmpfs and NVMe storage. Native
requests use primary / other / primary prompts of 1,048,514 / 7 / 1,048,514
prompt tokens and one output token. The restored request reports 1,048,512
cached tokens and the visible one-token response agrees. The recorded synthetic
snapshot transfer and counters, actual mount metadata, nested model/client waits,
owned retirement and all-five-card restoration pass their bounded protocol.

These jobs explicitly allowed CPU overlap. Transfer timing payloads and response
wall-clock fields are omitted from the public copy; their original raw files
remain immutable in the measurement epoch. The separate functional projection
retains native counts and primary-versus-restored visible-token equality, with
original response hashes. Transfer rates, latency rankings and performance
claims are withheld. This is not a full logical-state, full-context-quality or
original-reference equivalence proof.

## Actual 100K-class generate pilot

Both balanced placements passed two fresh generate repetitions, each reporting
99,972 native rendered prompt tokens and one selected output token/logit record. The
selected token/logit records agree between repetitions of the same placement, with no qualified reference match;
fresh process/KV does not clear OS or storage caches. There is no full-vocabulary or long-context task-quality claim. The native first
token times are TP4 23.368 / 23.478 s and PP5 45.997 / 46.038 s. The native clock starts after model boot and includes prompt work plus the first draw.
The corresponding prompt-count / first-token-time medians are 4268.14 and 2172.48 tok/s.
These are two-run point estimates, including the work before the first output;
they are not an isolated prefill-kernel timer. Whole-job elapsed times, which also
include setup and cleanup, are 130.889 s and 199.870 s and are not substituted
for first-token timing. No 128K measurement is relabelled as 100K.

## Fraser reasoning aliases

Two current balanced jobs pass nine native requests per placement: three each
at low=50, high=75 and max=100, T=0/top_p=1, with a 2,048-token combined
reasoning/content cap. Native alias resolution, request accounting, source
protocol, terminal and owned-envelope checks pass. The output fingerprint is a
12-hex diagnostic only; full output identity and golden quality were not tested.
The standard Fraser prompt/cap is outside exact C1/capacity512 admission.

## Unqualified capacity/loader attempt

Job 57 observes native 1M capacity and one-slot readiness, but fails the required
server actual-shell-wait/owned-retirement gate. Both its guard and controller
parent waits return 1. It is **not a qualified capacity pass**; all rates/startup
timing are withheld. Jobs 58 and 60 did not start. Main subsequently observed
all captured roles absent and all five standing caps restored/free cards, but
that cleanup observation does not repair the missing original wait. The raw
failed completion, native client verdict and retirement error remain separate.

## Clean candidate build and quality producer

A fresh Git ZIP of documentation source `e8dd6da` (engine/help source `1b0731a`)
was built inside Linux isolation. All 768 exported files were checked unchanged;
117 CUDA modules were freshly compiled, with 117 zero-warning reports. All 125
release host programs and 594 header compilations passed. The resulting binary
matches the release SHA above. The observed worker time was 381.056 s; this is
CPU build evidence during the Q functional window, not a performance result.
The exact final-tagged archive has its own proof in the distributor's companion
release manifest; this earlier candidate proof does not substitute for it.

The current golden observer/capture build passes 13 CPU parser/loop controls and
117 embedded-payload comparisons. It has captured **0/435 cases and 0/66,409
logical positions per balanced format on the GPU**. Current quality metrics are
unmeasured. The 45-job full campaign (19 EXL3, 26 PP5), needles, seed/repeat/stream
isolation, soak, remaining instrumentation, individual Pi cells and any other
unexecuted matrix protocols are deferred by the published snapshot boundary.
Existing historical scores remain dated; the guard has no independent golden score.

## Implementation review notes

The following 297 notes explain bounded source findings behind matrix citations. A source review is not
a GPU, numerical or quality pass. Statements refer to their recorded implementation review; current
runtime qualification and remaining gates are stated in the matrix. Historical claims marked corrected
were corrected in the published matrix. No archived operator instructions are required to read these notes.

<a id="source-audit-a-a1"></a>

### A1: Version / tag / commit hash

**Review disposition:** Historical claim corrected.

The runtime notice and actual clean-build/opening identities contradict a4d19/tcb80 as the current identity.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-a-a2"></a>

### A2: Release date and cadence

**Review disposition:** Bounded source finding.

v0.0.2-alpha is the current program version. Historical tcb build timestamps do not define a public release cadence.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-a-a3"></a>

### A3: Base framework

**Review disposition:** Bounded source finding.

The build and embedded-module loader establish dependencies, not a performance advantage or absence of all allocation.

Implementation: [Makefile](../../Makefile), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-a-a4"></a>

### A4: Model runner

**Review disposition:** Bounded source finding.

Prepared THIN/WIDE/CONE/FULL/BATCH programs use the shared step driver; step_graphs chooses graph replay and raw default is 0. Published recipes explicitly choose 1.

Implementation: [include/anemone/model_seam.h](../../include/anemone/model_seam.h), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c), [source/main.c](../../source/main.c).

<a id="source-audit-a-a5"></a>

### A5: Languages and build

**Review disposition:** Historical claim corrected.

C11/CUDA C++17 and checked-in optimized flags remain; current tree has 182 test executables (125 host/57 card), not 215. Compile inventory is not an executed-test count.

Implementation: [Makefile](../../Makefile).

<a id="source-audit-a-a6"></a>

### A6: License and attribution

**Review disposition:** Historical claim corrected.

Existing project license is AGPL-3.0-or-later, with SPDX metadata and version/startup attribution. The current project is not accurately described as private-license code.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-a-a7"></a>

### A7: Distribution form

**Review disposition:** Historical claim corrected.

The current candidate is one native executable with embedded CUDA modules; tokenizer, original pack and EXL3 map/data remain separate. An earlier clean candidate build reproduced da399; delivered archives keep their own identities. The current release identity is recorded above.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c), [Makefile](../../Makefile), [source/storage_exl3_linux.c](../../source/storage_exl3_linux.c).

<a id="source-audit-a-a8"></a>

### A8: Reproducibility

**Review disposition:** Bounded source finding.

Current opening canonical checks cover both widths2048/1024 and complete 8/256 controls; the current TP4 donor suite matched 423 same-job C1 text/count/finish signatures. No official/full-state or stochastic equivalence follows.

Implementation: [source/main.c](../../source/main.c), [include/anemone/pack_digest.h](../../include/anemone/pack_digest.h).

<a id="source-audit-a-a9"></a>

### A9: Launch workflow

**Review disposition:** Historical claim corrected.

Current workflow uses immutable absolute guarded scripts with recorded argv. Public setup has explicit recipes/four axes; no named --profile switch is admitted.

Implementation: [source/startup_axes.c](../../source/startup_axes.c), [source/main.c](../../source/main.c).

<a id="source-audit-a-a10"></a>

### A10: Startup qualification gates

**Review disposition:** Bounded source finding.

Actual peer qualification and drain contracts are narrower than a blanket count of 115 weight digests or all kernel forms.

Implementation: [source/model_v41_tp2_peer.h](../../source/model_v41_tp2_peer.h), [source/startup_policy.c](../../source/startup_policy.c), [source/model_v41.c](../../source/model_v41.c).

<a id="source-audit-a-a11"></a>

### A11: Cold start to ready (s)

**Review disposition:** Historical claim corrected.

Current admitted TP4 default boot-to-ready is 30.130703687667847s. Fresh process; OS/storage caches were not cleared; loader-only time is not isolated.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-a-a12"></a>

### A12: Warm restart

**Review disposition:** Not implemented.

Loader workers, replica discovery and automatic direct/dropped reads exist; no process-surviving persistent host-arena cache is exposed. Warm-restart gain is not isolated.

Implementation: [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c), [source/main.c](../../source/main.c), [source/storage_exl3_linux.c](../../source/storage_exl3_linux.c).

<a id="source-audit-a-a13"></a>

### A13: Served model names

**Review disposition:** Bounded source finding.

--model_alias supplies an optional request/listing alias. Published profile commands use anemone; benchmark windows may explicitly use ds41. There is no universal ds41 default.

Implementation: [source/chat_completions.c](../../source/chat_completions.c), [source/server_head.c](../../source/server_head.c).

<a id="source-audit-a-a14"></a>

### A14: Default lane / profiles

**Review disposition:** Historical claim corrected.

Omitted axes select no implicit tuple. The literal all-balanced tuple refuses; explicit EXL3 balanced recipe remains separate from axis admission and default mixed identity failure.

Implementation: [source/startup_axes.c](../../source/startup_axes.c), [source/main.c](../../source/main.c).

<a id="source-audit-a-a15"></a>

### A15: Configuration surface

**Review disposition:** Historical claim corrected.

Current serve guide enumerates 169 accepted spellings:149 numeric/array and 20 special. Catalog has 196 records/180 CLI names; branch-only catalog entries are not parser options.

Implementation: [source/main.c](../../source/main.c), [source/startup_axes.c](../../source/startup_axes.c).

<a id="source-audit-a-a16"></a>

### A16: Telemetry / metrics

**Review disposition:** Bounded source finding.

Stats route, live_stats and stats client exist. Stats distinguish server/session/program counters; live_stats defaults 0 and responses are not a Prometheus endpoint.

Implementation: [source/server_head.c](../../source/server_head.c), [source/server_stats.c](../../source/server_stats.c), [source/server.c](../../source/server.c).

<a id="source-audit-a-a17"></a>

### A17: Profiling hooks

**Review disposition:** Historical claim corrected.

External Nsight tracing can observe release execution, but optional timing/checking hooks compile out of release. draft_trace=1 refuses in release; no current ANEMONE_DIAG API is established by the old claim.

Implementation: [include/anemone/hooks.h](../../include/anemone/hooks.h), [source/main.c](../../source/main.c), [source/engine.c](../../source/engine.c).

<a id="source-audit-a-a18"></a>

### A18: Generation defaults on the server

**Review disposition:** Bounded source finding.

Raw CLI temperature0/top_p=1 defaults remain and HTTP can override them. Public operating recipes explicitly set startup T1/top_p=1; deterministic fixture requests explicitly use T0.

Implementation: [source/main.c](../../source/main.c), [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-a-a19"></a>

### A19: Test suites

**Review disposition:** Historical claim corrected.

Current tree contains 182 test program sources:125 host and 57 card. Fresh current release evidence passed125 host programs and 594 header compilations; this is separate from source 964436b remainder checks and card qualification.

Implementation: [Makefile](../../Makefile).

<a id="source-audit-a-a20"></a>

### A20: Hot kernel A/B without reload

**Review disposition:** Not implemented.

Backend loads embedded modules and prepares immutable graph executables; no exposed hot kernel-node replacement/rollback API is implemented.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-a-b1"></a>

### B1: GPUs used

**Review disposition:** Bounded source finding.

TP4 EXL3 chooses four CMP cards; current opening receipt leaves physical PRO 2 idle.

Implementation: [source/main.c](../../source/main.c), [source/model_v41_tp2_peer.h](../../source/model_v41_tp2_peer.h).

<a id="source-audit-a-b2"></a>

### B2: PCIe / topology

**Review disposition:** needs-runtime-test.

The backend queries directed peer access; source does not establish negotiated PCIe generation/width, bridges or current topology.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c), [source/model_v41_tp2_peer.h](../../source/model_v41_tp2_peer.h).

<a id="source-audit-a-b3"></a>

### B3: Host CPU / NUMA

**Review disposition:** needs-runtime-test.

Host code is compiled for znver4; that flag does not prove current CPU SKU/core count/NUMA topology.

Implementation: [Makefile](../../Makefile).

<a id="source-audit-a-b4"></a>

### B4: Thread pinning / IRQ

**Review disposition:** Bounded source finding.

Reviewed host dispatch creates workers without sched_setaffinity/pthread_setaffinity calls. CPU/IRQ pinning is not implemented by the engine path.

Implementation: [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c), [source/loader.c](../../source/loader.c).

<a id="source-audit-a-b5"></a>

### B5: Host RAM used

**Review disposition:** needs-runtime-test.

Engram tables are host-only extents and runtime gathers rows from their attached host allocation. About 188.8 GiB is a model-data budget, not measured total process RAM.

Implementation: [include/anemone/engram.h](../../include/anemone/engram.h), [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c).

<a id="source-audit-a-b6"></a>

### B6: Hugepages / THP

**Review disposition:** Historical claim corrected.

Host arena allocation requests THP using MADV_HUGEPAGE; TP arena is 2 MiB-aligned. Failure continues with small pages, so code does not guarantee physical huge pages.

Implementation: [source/loader.c](../../source/loader.c), [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c).

<a id="source-audit-a-b7"></a>

### B7: P2P enablement

**Review disposition:** Bounded source finding.

CUDA peer access is enabled only when available and mappings are qualified; transport prepares explicit directed mappings/exchanges. It is not unconditional BAR/P2P success on every device pair.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c), [source/model_v41_tp2_peer.h](../../source/model_v41_tp2_peer.h), [include/anemone/ordered_push.h](../../include/anemone/ordered_push.h).

<a id="source-audit-a-b8"></a>

### B8: Driver / CUDA

**Review disposition:** needs-runtime-test.

Build uses CUDA 13.4/GCC 15.2 in the recorded clean-build epoch; source includes the CUDA driver API but does not establish the current installed driver version.

Implementation: [Makefile](../../Makefile), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-a-b9"></a>

### B9: Arch targets

**Review disposition:** Bounded source finding.

Checked-in module targets are sm_80 and sm_120a, with selected native-PRO variants and sm_80-only tensor-core modules.

Implementation: [Makefile](../../Makefile).

<a id="source-audit-a-b10"></a>

### B10: Power cap for headlines

**Review disposition:** Bounded source finding.

Current opening window records CMP 180 W and restores saved caps; PRO is idle400 W under TP4, not retuned.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-a-b11"></a>

### B11: Clock policy

**Review disposition:** Bounded source finding.

Existing guard locks the CMP clock policy; current opening readiness records CMP 1455–1485 MHz, rather than assuming the requested 1695 MHz is achieved.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-a-b12"></a>

### B12: GPU memory per card

**Review disposition:** Bounded source finding.

Current TP4 opening readiness uses55197/64508/53684/64508 MiB on physical 0/1/3/4 for 65536 context×8 slots; this is total device use, not isolated weights.

Implementation: [source/model_v41.c](../../source/model_v41.c).

<a id="source-audit-a-b13"></a>

### B13: Docker / runtime settings

**Review disposition:** Bounded source finding.

The admitted command is a native executable launched by the existing guard; no container is required by the Makefile/runtime launch.

Implementation: [Makefile](../../Makefile), [source/server.c](../../source/server.c).

<a id="source-audit-a-b14"></a>

### B14: IOMMU

**Review disposition:** needs-runtime-test.

Peer-route source contains no evidence of the live IOMMU kernel policy; a successful peer mapping does not uniquely determine it.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c).

<a id="source-audit-a-c1"></a>

### C1: Checkpoint

**Review disposition:** Bounded source finding.

EXL3 map supplies quantized routed experts while retaining original non-routed, vision, draft and Engram tensors; public producer manifests pin both source families.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/storage_exl3_linux.c](../../source/storage_exl3_linux.c).

<a id="source-audit-a-c2"></a>

### C2: EXL3 pack

**Review disposition:** Bounded source finding.

EXL3 loader admits mul1 K2/K3 trellis weights;2.9 bpw is the pinned public checkpoint family label, not one uniform integer width.

Implementation: [include/anemone/exl3_records.h](../../include/anemone/exl3_records.h), [include/anemone/exl3_layout.h](../../include/anemone/exl3_layout.h), [source/model_v41.c](../../source/model_v41.c).

<a id="source-audit-a-c3"></a>

### C3: Expert bit widths

**Review disposition:** Bounded source finding.

The EXL3_K2_LAYERS mask0x7c0000 selects target layers 18–22 as K2; the other target layers are K3. Loader validates per-tensor bit widths.

Implementation: [source/model_v41.c](../../source/model_v41.c), [include/anemone/exl3_records.h](../../include/anemone/exl3_records.h).

<a id="source-audit-a-c4"></a>

### C4: Shared expert precision

**Review disposition:** Bounded source finding.

Shared-expert tensors remain original E4M3_SCALED with block scales; the tensor rules validate their format/shapes before loading. Decode/math order is selected separately.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu).

<a id="source-audit-a-c5"></a>

### C5: Attention projections

**Review disposition:** Bounded source finding.

wq_a/wkv/wq_b/wo_a/wo_b retain original E4M3_SCALED tensors with their declared shapes. Tensor format does not alone qualify accumulation order.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu).

<a id="source-audit-a-c6"></a>

### C6: Dense non-expert linears

**Review disposition:** Bounded source finding.

Original quantized dense weights retain their codes/scales; prepared BF16-row MMA paths and alternate class/order forms are explicit. This is not every dense tensor: some model tensors are BF16/FP32.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [include/anemone/dense_linear_records.h](../../include/anemone/dense_linear_records.h).

<a id="source-audit-a-c7"></a>

### C7: Vocabulary head

**Review disposition:** Bounded source finding.

Target head has 129280×5120 BF16 weights; tp_head_shards cuts 32320 vocabulary rows per rank when selected.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/kernel_head_mma.cu](../../source/kernel_head_mma.cu).

<a id="source-audit-a-c8"></a>

### C8: Embedding

**Review disposition:** Bounded source finding.

Embedding weights are BF16. EXL3 replicated_embedding may copy them per TP rank; that is an explicit placement optimization, not a new weight family.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-c9"></a>

### C9: Draft (DSpark) weights

**Review disposition:** Bounded source finding.

Checkpoint DSpark has three stages 40–42, original FP8 dense/FP4 experts and a BF16 draft head. The published EXL3 BATCH recipe enables sharded draft head/layers.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/model_v41.c](../../source/model_v41.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-c10"></a>

### C10: Indexer weights / keys

**Review disposition:** Historical claim corrected.

Indexer wq_b is E4M3 but wk and weights_proj rules are BF16. Balanced index keys/queries use FP16 carriers; calling all indexer weights FP8 is inaccurate.

Implementation: [source/model_v41.c](../../source/model_v41.c), [include/anemone/attention_state_records.h](../../include/anemone/attention_state_records.h), [include/anemone/index_scan_records.h](../../include/anemone/index_scan_records.h).

<a id="source-audit-a-c11"></a>

### C11: Compressor weights

**Review disposition:** Bounded source finding.

Compressor wkv/wgate/norm rules are BF16 at the declared source/pooled layers; loader rejects another element/shape/format.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/model_v41.h](../../source/model_v41.h).

<a id="source-audit-a-c12"></a>

### C12: Router / gate

**Review disposition:** Bounded source finding.

Target gate weights are BF16[384][5120], bias FP32; scoring/normalization is FP32. tp_router_mma is an opt-in alternate product order, raw 0.

Implementation: [source/model_v41.c](../../source/model_v41.c), [include/anemone/router_records.h](../../include/anemone/router_records.h), [source/router.c](../../source/router.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-c13"></a>

### C13: mHC weights

**Review disposition:** Bounded source finding.

Attention/FFN mHC mix weights are FP32[24][20480], with FP32 bases/scales. These are distinct from BF16 residual streams.

Implementation: [source/model_v41.c](../../source/model_v41.c), [include/anemone/hyper_connection_records.h](../../include/anemone/hyper_connection_records.h).

<a id="source-audit-a-c14"></a>

### C14: Norm weights

**Review disposition:** Bounded source finding.

RMS norm weights are BF16 and model norm epsilon1e-20; mHC Sinkhorn epsilon is separately1e-6.

Implementation: [source/model_v41.c](../../source/model_v41.c), [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/attention_state_records.h](../../include/anemone/attention_state_records.h).

<a id="source-audit-a-c15"></a>

### C15: Engram backing

**Review disposition:** Bounded source finding.

Engram layers 1/14 use host-only tables;24 rows/token×264 bytes=6336 bytes/table/token are gathered and uploaded, separate from the full resident tables.

Implementation: [source/model_v41.h](../../source/model_v41.h), [include/anemone/engram.h](../../include/anemone/engram.h).

<a id="source-audit-a-c16"></a>

### C16: Load-time repacking

**Review disposition:** Bounded source finding.

Load-time EXL3 slice copy preserves codes/scales without FP arithmetic; native tensors are validated/repacked separately. Extent digests detect accidental corruption and are not cryptographic SHA proof.

Implementation: [include/anemone/exl3_layout.h](../../include/anemone/exl3_layout.h), [source/model_v41.c](../../source/model_v41.c), [include/anemone/pack_digest.h](../../include/anemone/pack_digest.h).

<a id="source-audit-a-c17"></a>

### C17: Load time / verification

**Review disposition:** Historical claim corrected.

Current TP4 process-start to readiness30.130703687667847s is a complete boot measurement, not isolated conversion/read/upload time. Pack/digest checks are separate source gates.

Implementation: [source/main.c](../../source/main.c), [include/anemone/pack_digest.h](../../include/anemone/pack_digest.h).

<a id="source-audit-a-c18"></a>

### C18: Weight memory per card

**Review disposition:** needs-runtime-test.

Arena planning includes weights, mirrored state, scratch and programs; current readiness memory cannot be labeled weight-only bytes.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c).

<a id="source-audit-a-c19"></a>

### C19: Vision tower

**Review disposition:** needs-runtime-test.

Vision/image routing and bounded encoder/projector preparation are implemented. Text-only opening controls do not prove current image execution, accuracy or throughput.

Implementation: [include/anemone/vision.h](../../include/anemone/vision.h), [source/vision.c](../../source/vision.c).

<a id="source-audit-a-d1"></a>

### D1: TP degree / process mapping

**Review disposition:** Bounded source finding.

One engine process drives four qualified CMP ranks under tp4_exl3; optional parallel host launches do not create four model-server processes.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/main.c](../../source/main.c).

<a id="source-audit-a-d2"></a>

### D2: Attention heads per rank

**Review disposition:** Bounded source finding.

TP4 splits 64 heads into16 heads/rank; dense projections select replicated and head/group/input shards. Compact query/core buffers avoid specific copies where the chosen flags permit.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-d3"></a>

### D3: Expert TP split

**Review disposition:** Bounded source finding.

Balanced EXL3 inner expert width 2304 is split 512/640/512/640 on 128-column transforms. Exact output-shard helpers are a different explicitly prepared arithmetic path.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/startup_policy.c](../../source/startup_policy.c), [include/anemone/exl3_layout.h](../../include/anemone/exl3_layout.h).

<a id="source-audit-a-d4"></a>

### D4: Expert rotation

**Review disposition:** needs-runtime-test.

Expert rotation modes 0/1/2 exist; raw 0. Modes1/2 redistribute complete 128-column chunks and can change TP grouping/order.

Implementation: [include/anemone/exl3_tc_records.h](../../include/anemone/exl3_tc_records.h).

<a id="source-audit-a-d5"></a>

### D5: EP vs TP for experts

**Review disposition:** Bounded source finding.

Published EXL3 execution tensor-shards routed experts; no independent expert-parallel rank ownership is selected by that recipe.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [include/anemone/exl3_tc_records.h](../../include/anemone/exl3_tc_records.h).

<a id="source-audit-a-d6"></a>

### D6: Shared expert split

**Review disposition:** needs-runtime-test.

Shared/routed expert branches and shared-sum combination are prepared conditionally; graph_branches selects the branch form.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-d7"></a>

### D7: Dense layer split

**Review disposition:** Bounded source finding.

Dense attention splits use head/group outputs and input-parallel wo_b; gather_balance is a separate optional distribution policy, raw 0.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-d8"></a>

### D8: Head split

**Review disposition:** Bounded source finding.

Published EXL3 recipe enables head_shards/head_stream/head_mma; head rows32320/rank. Raw defaults for these switches are 0.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/kernel_head_mma.cu](../../source/kernel_head_mma.cu), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-d9"></a>

### D9: Draft placement

**Review disposition:** Historical claim corrected.

Published multi-session EXL3 BATCH recipe selects tp_draft_layers=1 with sharded head/draft rows. Unsharded THIN is a different one-slot recipe; historical C8 gains are not defaults.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-d10"></a>

### D10: Pipeline parallel

**Review disposition:** Bounded source finding.

Published EXL3 uses TP4 and ordinary pipeline depth1; it is distinct from original layer PP5.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/startup_policy.c](../../source/startup_policy.c).

<a id="source-audit-a-d11"></a>

### D11: Rank imbalance (640 vs 512)

**Review disposition:** needs-runtime-test.

512/640 expert shards and rank 0 global/draft obligations create different allocation/work ownership; that does not establish a current timing imbalance.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [include/anemone/exl3_tc_records.h](../../include/anemone/exl3_tc_records.h).

<a id="source-audit-a-d12"></a>

### D12: Data parallel / replicas

**Review disposition:** Not implemented.

The documented server prepares one model across its admitted TP/PP placement; no data-parallel serving-replica controller is implemented by this launch.

Implementation: [source/startup_policy.c](../../source/startup_policy.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [include/anemone/model_seam.h](../../include/anemone/model_seam.h).

<a id="source-audit-a-e1"></a>

### E1: Decode all-reduce

**Review disposition:** Bounded source finding.

Balanced TP4 decode uses explicitly selected BF16 ring sums; FP32/lossless-gather paths are separate. Target attention/FFN sites communicate under prepared banks.

Implementation: [include/anemone/exact_ring_records.h](../../include/anemone/exact_ring_records.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-e2"></a>

### E2: Prefill all-reduce

**Review disposition:** Historical claim corrected.

tp_wide_exchange=5 uses CE owner sums at attention and CE owner-ring sums at FFN where dispatch conditions fit. It is not FP32 round-once arithmetic at both sites; mode 7 is the current raw split policy.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/model_v41_program.c](../../source/model_v41_program.c), [source/main.c](../../source/main.c).

<a id="source-audit-a-e3"></a>

### E3: All-gather / gather uses

**Review disposition:** Bounded source finding.

Prepared split-pipeline gathers distribute selected pool/index/router ranges using disjoint exchange phases; draft sharing has its own prepared ranges.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [include/anemone/tp_exchange_records.h](../../include/anemone/tp_exchange_records.h), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-e4"></a>

### E4: Copy-engine usage

**Review disposition:** needs-runtime-test.

The CE forms insert peer copies and ordered READY/DONE phases for row quarters, with explicit side-run dependencies.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-a-e5"></a>

### E5: Wire dtype

**Review disposition:** Bounded source finding.

Balanced TP4 wire uses BF16 where selected. Owner sums use FP32 ordered accumulation/rounding; FFN ring modes preserve BF16 per-add rounding. FP32 wire/gather are distinct paths.

Implementation: [include/anemone/ordered_push.h](../../include/anemone/ordered_push.h), [include/anemone/tp_exchange.h](../../include/anemone/tp_exchange.h), [include/anemone/exact_ring_records.h](../../include/anemone/exact_ring_records.h).

<a id="source-audit-a-e6"></a>

### E6: Deterministic order

**Review disposition:** Historical claim corrected.

Order is fixed per selected collective, not one universal ascending rank chain: owner sums use ascending-rank FP32; BF16 ring order depends on owner/column policy.

Implementation: [include/anemone/ordered_push.h](../../include/anemone/ordered_push.h), [include/anemone/tp_exchange.h](../../include/anemone/tp_exchange.h), [include/anemone/exact_ring_records.h](../../include/anemone/exact_ring_records.h).

<a id="source-audit-a-e7"></a>

### E7: Collectives per layer / token

**Review disposition:** needs-runtime-test.

Prepared target graph has attention/FFN exchange sites plus conditional draft collectives; actual launch counts depend on THIN/BATCH/wide/branch forms.

Implementation: [include/anemone/tp_exchange_records.h](../../include/anemone/tp_exchange_records.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-e8"></a>

### E8: Latency per all-reduce

**Review disposition:** needs-runtime-test.

Collective preparation and bounded-wait code do not measure all-reduce latency.

Implementation: [include/anemone/tp_exchange.h](../../include/anemone/tp_exchange.h), [source/tp_exchange.c](../../source/tp_exchange.c).

<a id="source-audit-a-e9"></a>

### E9: Measured P2P bandwidth

**Review disposition:** needs-runtime-test.

Directed peer access and CE/SM paths are implemented; no current bandwidth is established by the source or opening token controls.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c), [source/model_v41_tp2_peer.h](../../source/model_v41_tp2_peer.h).

<a id="source-audit-a-e10"></a>

### E10: Comm/compute overlap

**Review disposition:** needs-runtime-test.

Split-pipeline and shared-expert side branches encode specific overlap opportunities. Raw split_overlap=7 and optional overrides must be distinguished from prior mode 5/6 runs.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-a-e11"></a>

### E11: Exchange protocol

**Review disposition:** Bounded source finding.

READY/DONE phases, strictly increasing sequences, disjoint site/bank ownership and bounded waits guard collective reuse; a failed bank must drain before reuse.

Implementation: [include/anemone/tp_exchange_records.h](../../include/anemone/tp_exchange_records.h), [include/anemone/ordered_push.h](../../include/anemone/ordered_push.h), [include/anemone/exact_ring_records.h](../../include/anemone/exact_ring_records.h).

<a id="source-audit-a-e12"></a>

### E12: Collective fusion with norm/residual

**Review disposition:** needs-runtime-test.

Combination can add routed sums before an in-place BF16 ring; site fronts fuse mix/norm work. Dispatch depends on selected form and routed buffer.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [include/anemone/site_front.h](../../include/anemone/site_front.h).

<a id="source-audit-a-e13"></a>

### E13: Cooperative kernels

**Review disposition:** Not implemented.

Current backend prepares ordinary CUDA kernel/graph launches; no cooperative-grid launch protocol is selected. Intra-block cooperation in TC kernels is a different concept.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-a-e14"></a>

### E14: Host / NCCL tuning

**Review disposition:** Bounded source finding.

The runtime uses its own CUDA driver exchanges; no NCCL link/API tuning is part of this executable.

Implementation: [Makefile](../../Makefile), [source/backend_cuda.c](../../source/backend_cuda.c), [source/tp_exchange.c](../../source/tp_exchange.c).

<a id="source-audit-a-e15"></a>

### E15: Failure handling

**Review disposition:** Bounded source finding.

Collective deadlines fail closed; backend launch_wait returns failure and whole-group drain retains uncertain groups before teardown. Release need not emit the old diagnostic text because optional logging compiles out.

Implementation: [source/model_v41_tp2_peer.h](../../source/model_v41_tp2_peer.h), [include/anemone/exact_ring_records.h](../../include/anemone/exact_ring_records.h), [include/anemone/hooks.h](../../include/anemone/hooks.h).

<a id="source-audit-a-e16"></a>

### E16: NCCL use

**Review disposition:** Bounded source finding.

No NCCL inference dependency is linked or called; collectives use prepared own CUDA kernels/copies.

Implementation: [Makefile](../../Makefile), [source/tp_exchange.c](../../source/tp_exchange.c).

<a id="source-audit-a-e17"></a>

### E17: Host-memory all-reduce

**Review disposition:** Bounded source finding.

Generic ordered-push supports qualified direct/host-mirror routes, but current admitted TP peer qualifier requires direct group edges and does not admit a host fallback.

Implementation: [include/anemone/ordered_push.h](../../include/anemone/ordered_push.h), [source/model_v41_tp2_peer.h](../../source/model_v41_tp2_peer.h).

<a id="source-audit-a-e18"></a>

### E18: Lossless compressed wire

**Review disposition:** Bounded source finding.

No general compressed-wire optimization is admitted by the balanced recipe. tp_code_wire is a separate original code/scale path; exact_ring lossless sentinel encoding/gather is not a compression-performance claim.

Implementation: [include/anemone/exact_ring_records.h](../../include/anemone/exact_ring_records.h), [source/main.c](../../source/main.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-e19"></a>

### E19: Round-once sums

**Review disposition:** Historical claim corrected.

Round-once behavior is form-specific: attention owner sums differ from the default FFN BF16 ring; decode_round_once is an explicit target-bit-changing option, raw 0.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [include/anemone/tp_exchange.h](../../include/anemone/tp_exchange.h).

<a id="source-audit-a-f1"></a>

### F1: mHC implementation, streams

**Review disposition:** Bounded source finding.

Model has 4 residual streams,24mixes and 20 Sinkhorn iterations with epsilon1e-6. Balanced thin site/FFN fronts fuse specific mix/norm/router work; wide/BATCH forms differ.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/hyper_connection_records.h](../../include/anemone/hyper_connection_records.h), [source/kernel_hyper_connection.cuh](../../source/kernel_hyper_connection.cuh), [include/anemone/site_front.h](../../include/anemone/site_front.h).

<a id="source-audit-a-f2"></a>

### F2: post / finish fusion

**Review disposition:** Bounded source finding.

Balanced hyper_mix_finish/hyper_post and site fronts explicitly combine residual/post work; BATCH and THIN have different preparation. Fusion is conditional, not one universal launch sequence.

Implementation: [include/anemone/hyper_connection_records.h](../../include/anemone/hyper_connection_records.h), [include/anemone/site_front.h](../../include/anemone/site_front.h), [include/anemone/model_seam.h](../../include/anemone/model_seam.h).

<a id="source-audit-a-f3"></a>

### F3: Residual precision

**Review disposition:** Bounded source finding.

Balanced residual streams are BF16[row][4][5120]; pre/post/comb coefficients are FP32. Storage type does not establish every arithmetic intermediate type.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/hyper_connection_records.h](../../include/anemone/hyper_connection_records.h).

<a id="source-audit-a-f4"></a>

### F4: Sinkhorn precision

**Review disposition:** Bounded source finding.

Sinkhorn implementation uses FP32,20 iterations and explicit row/column sum order; the stated epsilon is 1e-6.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/kernel_hyper_connection.cuh](../../source/kernel_hyper_connection.cuh).

<a id="source-audit-a-f5"></a>

### F5: mHC time per step

**Review disposition:** needs-runtime-test.

Current code fuses selected site fronts and prepares separate wide/BATCH mixes. It does not establish current 2.25ms/3.7ms totals.

Implementation: [include/anemone/site_front.h](../../include/anemone/site_front.h), [include/anemone/hyper_connection_records.h](../../include/anemone/hyper_connection_records.h), [include/anemone/model_seam.h](../../include/anemone/model_seam.h).

<a id="source-audit-a-f6"></a>

### F6: Exactness

**Review disposition:** needs-runtime-test.

Explicit deterministic mHC arithmetic exists; a blanket bit-identical-front claim needs its original fixture/shape/math-class scope.

Implementation: [include/anemone/hyper_connection_records.h](../../include/anemone/hyper_connection_records.h), [source/kernel_hyper_connection.cuh](../../source/kernel_hyper_connection.cuh), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-g1"></a>

### G1: Attention variant per layer

**Review disposition:** Bounded source finding.

Model masks/constants encode window-only layers 0–1, ratio 2 layers 2–19, ratio 1 layers 20–39 and pool sources 2/8/14/20.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/model_v41.h](../../source/model_v41.h), [source/model_v41.c](../../source/model_v41.c).

<a id="source-audit-a-g2"></a>

### G2: Heads and dims

**Review disposition:** Bounded source finding.

Model constants are 64 heads×512 dims (448 non-RoPE+64 RoPE), qrank 1280,8 output groups, window 128,512 selected entries, index 32×128.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h).

<a id="source-audit-a-g3"></a>

### G3: Q / KV projection kernels

**Review disposition:** Bounded source finding.

EXL3 balanced trunk-MMA/ordered-wide forms are selected by explicit knobs; grouped/thin-split variants have separate shapes/orders.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-g4"></a>

### G4: Q norm, RoPE

**Review disposition:** Historical claim corrected.

Balanced query/index carriers are FP16 with model-specific BF16 rounding/RoPE boundaries and FP32 rotation words. Do not apply that carrier claim to original exact.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/attention_state_records.h](../../include/anemone/attention_state_records.h), [include/anemone/official_state_records.h](../../include/anemone/official_state_records.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-g5"></a>

### G5: KV compressor

**Review disposition:** Bounded source finding.

Compressor writes ratio-q pool entry at completed group positions; source rules are BF16 and state producers choose the declared carriers. pool_chunk/pool_gather are conditional preparation policies.

Implementation: [include/anemone/attention_state_records.h](../../include/anemone/attention_state_records.h), [source/model_v41.c](../../source/model_v41.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-g6"></a>

### G6: Lightning indexer scoring

**Review disposition:** Historical claim corrected.

Balanced index scoring uses FP16 tiled queries/keys and specified FP32 reductions; index source mask includes 2/8/14/20/24/28/32/36.

Implementation: [include/anemone/index_scan_records.h](../../include/anemone/index_scan_records.h), [source/model_v41.h](../../source/model_v41.h), [source/index_scan.c](../../source/index_scan.c), [source/official_index.c](../../source/official_index.c).

<a id="source-audit-a-g7"></a>

### G7: Top-k selection

**Review disposition:** Bounded source finding.

Balanced index selection has deterministic ascending indices/tie handling; split histogram stages and the2048×8 candidate bound are explicit prepared forms.

Implementation: [include/anemone/index_scan_records.h](../../include/anemone/index_scan_records.h), [source/index_scan.c](../../source/index_scan.c).

<a id="source-audit-a-g8"></a>

### G8: Index scan partitioning

**Review disposition:** Bounded source finding.

EXL3 split-pipeline can partition index/router/pool work and gather selected row ranges under fixed phases; distribution depends on gather and split controls.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [include/anemone/tp_exchange_records.h](../../include/anemone/tp_exchange_records.h).

<a id="source-audit-a-g9"></a>

### G9: Sparse core passes

**Review disposition:** Bounded source finding.

Balanced core supports fixed one-launch and reference scores/values forms; up to 128window+512 selected keys are described. Actual dispatch follows prepared order/math class.

Implementation: [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h), [source/attention_core.c](../../source/attention_core.c).

<a id="source-audit-a-g10"></a>

### G10: Core tiles

**Review disposition:** Bounded source finding.

Ordinary core geometry includes 8 heads/block,128threads and shape-specific reference variants; prefill uses1024 attention slabs/256 scan slabs. These constants are not universal exact-core launch sizes.

Implementation: [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h), [include/anemone/model_layout.h](../../include/anemone/model_layout.h).

<a id="source-audit-a-g11"></a>

### G11: Sliding window

**Review disposition:** Bounded source finding.

Balanced window is 128 tokens in a4096-slot physical ring,1024 bytes/row, with the kept-prefix reach policy separate from logical window size.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/attention_state_records.h](../../include/anemone/attention_state_records.h).

<a id="source-audit-a-g12"></a>

### G12: Attention sink / softmax

**Review disposition:** Bounded source finding.

Balanced attention adds the FP32 sink term to its softmax denominator and rounds output to BF16; selected value/probability carriers and reduction order matter.

Implementation: [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h).

<a id="source-audit-a-g13"></a>

### G13: Output projection wo_a / wo_b

**Review disposition:** Bounded source finding.

EXL3 dense sharding and compact-head paths can feed wo_a/wo_b directly with BF16 rows; choices are prepared, not a new token-path tuner.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu).

<a id="source-audit-a-g14"></a>

### G14: Decode vs prefill kernel

**Review disposition:** Bounded source finding.

Ordinary THIN core and chunk-wide prefill core are separate prepared forms; fixed/reference/core-chunk controls determine the actual path.

Implementation: [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-g15"></a>

### G15: Chunk-wide attention

**Review disposition:** Bounded source finding.

Chunk core is a conditional prepared prefill path whose state-write/scan dependencies precede the core; it is distinct from ordinary per-slab execution.

Implementation: [include/anemone/model_seam.h](../../include/anemone/model_seam.h), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h).

<a id="source-audit-a-g16"></a>

### G16: Attention time

**Review disposition:** needs-runtime-test.

Current attention code/dispatch identifies core, prepare and index work; it does not establish new per-round/per-30K microseconds or seconds.

Implementation: [source/attention_core.c](../../source/attention_core.c), [source/index_scan.c](../../source/index_scan.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-g17"></a>

### G17: Long-context behaviour

**Review disposition:** needs-runtime-test.

The candidate stage and source limits support planning up to 1048576 tokens; bounded candidate storage does not prove512K/1M generation correctness or speed.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/index_scan.c](../../source/index_scan.c).

<a id="source-audit-a-g18"></a>

### G18: Metadata launches

**Review disposition:** needs-runtime-test.

Step metadata is uploaded for prepared graph/program runs; source does not establish12 copies/~40us as a current invariant.

Implementation: [include/anemone/model_seam.h](../../include/anemone/model_seam.h), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-a-h1"></a>

### H1: KV dtype

**Review disposition:** Historical claim corrected.

Balanced window/pool/index-key storage uses FP16 carriers. State 16 is a width selector, not proof that every profile stores IEEE FP16 values.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/attention_state_records.h](../../include/anemone/attention_state_records.h), [source/model_v41.c](../../source/model_v41.c), [include/anemone/official_state_records.h](../../include/anemone/official_state_records.h).

<a id="source-audit-a-h2"></a>

### H2: Indexer cache dtype

**Review disposition:** Historical claim corrected.

Balanced index-key tiles contain FP16 values,256 bytes/key; that does not make the original-exact path an FP16 scored indexer.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/attention_state_records.h](../../include/anemone/attention_state_records.h), [include/anemone/index_scan_records.h](../../include/anemone/index_scan_records.h).

<a id="source-audit-a-h3"></a>

### H3: Cache layout

**Review disposition:** Bounded source finding.

State arrays are arena-relative per-session rings/pools/key tiles with explicit strides and source-layer ownership/mirroring.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/model_v41.c](../../source/model_v41.c), [source/model_v41.h](../../source/model_v41.h).

<a id="source-audit-a-h4"></a>

### H4: Paging / block size

**Review disposition:** Bounded source finding.

Server allocates fixed session slots and static arena state; no vLLM-style block-table paging allocator is selected.

Implementation: [include/anemone/model_seam.h](../../include/anemone/model_seam.h), [source/model_v41.c](../../source/model_v41.c), [source/server.c](../../source/server.c).

<a id="source-audit-a-h5"></a>

### H5: Prefix caching

**Review disposition:** needs-runtime-test.

Prefix reuse and on-disk snapshots are implemented and ordinary serve prefix_reuse defaults 1. Current TP4 warm-cache probe failed identity; do not refresh blanket same-bits or99.1%hit claims.

Implementation: [source/main.c](../../source/main.c), [source/engine.c](../../source/engine.c), [source/server_sessions.c](../../source/server_sessions.c).

<a id="source-audit-a-h6"></a>

### H6: Max context and 1M slots

**Review disposition:** Historical claim corrected.

Raw model context limit 1048576 and source admission are distinct from actual fit. TP4 current opening used 65536×8; its prior one 1M boot remains dated until the prepared current job executes.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/model_v41.c](../../source/model_v41.c), [source/main.c](../../source/main.c).

<a id="source-audit-a-h7"></a>

### H7: Bytes per token (all caches, per card)

**Review disposition:** Bounded source finding.

For replicated TP4 source caches, pool/key rules yield 3200 bytes/context-token plus target rings:40×4096×1024 bytes=160 MiB/session before drafts/scratch/other arrays. This is a source derivation, not measured total GPU use.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/model_v41.h](../../source/model_v41.h), [include/anemone/model_layout.h](../../include/anemone/model_layout.h).

<a id="source-audit-a-h8"></a>

### H8: KV capacity at 1M

**Review disposition:** Historical claim corrected.

TP4 current opening is 65536×8; the current one 1M readiness job is separate and pending. Earlier one-slot/2 slotOOM evidence retains its epoch.

Implementation: [source/model_v41.c](../../source/model_v41.c), [include/anemone/model_layout.h](../../include/anemone/model_layout.h).

<a id="source-audit-a-h9"></a>

### H9: Eviction / preemption

**Review disposition:** Bounded source finding.

Fixed session slots use bounded global/per-user waiting queues: raw queue_depth=16 and omitted user queue sentinel resolves to 8. This is queuing/slot selection, not recompute preemption.

Implementation: [source/server.c](../../source/server.c).

<a id="source-audit-a-h10"></a>

### H10: Host offload of KV

**Review disposition:** Not implemented.

There is no active host-KV-offload execution mode in these recipes. Session snapshots explicitly serialize state to disk; that is persistence, not token-path KV offload.

Implementation: [source/server_sessions.c](../../source/server_sessions.c), [source/model_v41.c](../../source/model_v41.c).

<a id="source-audit-a-h11"></a>

### H11: Session persistence

**Review disposition:** Bounded source finding.

session_directory implements named snapshot persistence with identity/version checks, temporary file+rename and cache namespaces. Current format is v4.

Implementation: [source/server_sessions.c](../../source/server_sessions.c).

<a id="source-audit-a-h12"></a>

### H12: Draft state

**Review disposition:** Bounded source finding.

Checkpoint DSpark has 3 stage rings and taps at target layers 37–39; TP-layer drafting changes placement but not these model constants.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-h13"></a>

### H13: Memory utilization

**Review disposition:** Bounded source finding.

Current TP4 readiness memory is total device use at 65536×8, including model/state/workspaces, not a fractional allocator target.

Implementation: [source/model_v41.c](../../source/model_v41.c).

<a id="source-audit-a-h14"></a>

### H14: KV pool at 262K (tokens)

**Review disposition:** Bounded source finding.

Anemone has explicit static context/session allocations, not one shared paged262K token pool. The recorded default opening allocates 8×65536 per-slot capacities.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/server.c](../../source/server.c).

<a id="source-audit-a-i1"></a>

### I1: Experts, top-k, shared

**Review disposition:** Bounded source finding.

Target model constants are 384 routed experts, top-6 and 2304 intermediate width; shared expert tensors are separate from routed expert jobs.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/model_v41.c](../../source/model_v41.c), [source/router.c](../../source/router.c).

<a id="source-audit-a-i2"></a>

### I2: Hash-routed layers

**Review disposition:** Bounded source finding.

Current target router reads dense scores, bias and sqrt-softplus top-6 for every target layer; no target hash-router branch is selected. Engram hashing is a separate mechanism.

Implementation: [include/anemone/router_records.h](../../include/anemone/router_records.h), [source/kernel_router.cu](../../source/kernel_router.cu), [source/model_v41.h](../../source/model_v41.h).

<a id="source-audit-a-i3"></a>

### I3: Scoring

**Review disposition:** Bounded source finding.

Target router uses FP32 sqrt(softplus), bias-ranked top-6, epsilon1e-20 andscale1.5. Normalization order is explicit: ascending expert or reference library tree; ties use engine lower-index order.

Implementation: [include/anemone/router_records.h](../../include/anemone/router_records.h), [source/kernel_router.cu](../../source/kernel_router.cu).

<a id="source-audit-a-i4"></a>

### I4: Router kernel

**Review disposition:** needs-runtime-test.

Thin FFN front can fuse router work; wide router prepares select/count/place/jobs and small fused forms; optional router_mma changes product order.

Implementation: [include/anemone/site_front.h](../../include/anemone/site_front.h), [include/anemone/router_records.h](../../include/anemone/router_records.h), [source/router.c](../../source/router.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-i5"></a>

### I5: Tie-break

**Review disposition:** Bounded source finding.

Router deterministically prefers the lower expert among equal affinity+bias values. The reference top-k tie order is explicitly described as open/unstable.

Implementation: [include/anemone/router_records.h](../../include/anemone/router_records.h), [source/kernel_router.cu](../../source/kernel_router.cu).

<a id="source-audit-a-i6"></a>

### I6: Grouping / permute

**Review disposition:** Bounded source finding.

Router produces ascending-expert jobs and per-expert row pairs/runs; wide execution explicitly counts/places rows. No extra generic tensor permute is implied, but wide packing work still exists.

Implementation: [include/anemone/router_records.h](../../include/anemone/router_records.h), [source/router.c](../../source/router.c).

<a id="source-audit-a-i7"></a>

### I7: Decode expert kernels

**Review disposition:** needs-runtime-test.

Published EXL3 recipe explicitly enablesTC-MoE; raw exl3_tc_moe=0 retains the separate chain. TC input/activation/combination numerical boundaries differ from simple FP32-only claims.

Implementation: [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/exl3_tc.c](../../source/exl3_tc.c), [source/kernel_exl3_tc.cuh](../../source/kernel_exl3_tc.cuh).

<a id="source-audit-a-i8"></a>

### I8: Mixed widths

**Review disposition:** Bounded source finding.

K2/K3 EXL3 formats and mixed-width rotation geometry are explicitly supported. Layer mask18–22 selects K2 in the current target EXL3 map family.

Implementation: [include/anemone/exl3_records.h](../../include/anemone/exl3_records.h), [source/model_v41.c](../../source/model_v41.c), [include/anemone/exl3_tc_records.h](../../include/anemone/exl3_tc_records.h).

<a id="source-audit-a-i9"></a>

### I9: Prefill expert kernels

**Review disposition:** needs-runtime-test.

EXL3 prefill uses dedicated TC operand/product/activation/combine forms over chunked routed jobs, with shape/width validation.

Implementation: [source/kernel_exl3_tcp.cuh](../../source/kernel_exl3_tcp.cuh), [source/exl3_tc.c](../../source/exl3_tc.c).

<a id="source-audit-a-i10"></a>

### I10: Expert tiles

**Review disposition:** Bounded source finding.

TC prefill chooses h640 only for a nonrotated640-column record in its bounded row range; other shapes retain their i/mixed variants. The dispatch is explicit rather than one universal640 kernel.

Implementation: [source/exl3_tc.c](../../source/exl3_tc.c).

<a id="source-audit-a-i11"></a>

### I11: Activation quantization

**Review disposition:** Bounded source finding.

TC-MoE uses FP16 hi/lo operands with explicit FP64 input transforms and THIN activation/combine work; prefill has its own FP32 operand/activation and FP64 combination boundaries. The chain is a separate path.

Implementation: [source/kernel_exl3_tc.cuh](../../source/kernel_exl3_tc.cuh), [source/kernel_exl3_tcp.cuh](../../source/kernel_exl3_tcp.cuh).

<a id="source-audit-a-i12"></a>

### I12: SwiGLU

**Review disposition:** Bounded source finding.

Prepared expert activation limit is 10: gate is upper-clamped, up is clamped to[-10,10], then SiLU(gate)×up. TC/prefill use their documented precision boundaries.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [source/kernel_exl3_tc.cuh](../../source/kernel_exl3_tc.cuh), [source/kernel_exl3_tcp.cuh](../../source/kernel_exl3_tcp.cuh).

<a id="source-audit-a-i13"></a>

### I13: Combine

**Review disposition:** needs-runtime-test.

EXL3 routed partials combine locally before selected shared-sum/ring dispatch; THIN TC and prefill combination are not both plain FP32.

Implementation: [source/kernel_exl3_tcp.cuh](../../source/kernel_exl3_tcp.cuh), [source/kernel_exl3_tc.cuh](../../source/kernel_exl3_tc.cuh), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-i14"></a>

### I14: Prefill rows per call

**Review disposition:** Bounded source finding.

EXL3 prefill work can cover the configured 1024/2048chunk under validated TC geometry; row capacity and actual emitted pieces are explicit preparation choices.

Implementation: [include/anemone/model_seam.h](../../include/anemone/model_seam.h), [source/exl3_tc.c](../../source/exl3_tc.c).

<a id="source-audit-a-i15"></a>

### I15: Per-rank expert balance

**Review disposition:** needs-runtime-test.

Expert-width/rotation ownership is the same source mechanism asD3/D4/D11; there is no independent fresh rank-balance measurement in this row.

Implementation: [include/anemone/exl3_tc_records.h](../../include/anemone/exl3_tc_records.h), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-i16"></a>

### I16: EXL3 dequant

**Review disposition:** needs-runtime-test.

EXL3 trellis decode and 128-wide transforms/scales exist in separate chain and TC paths; products can decode weights on demand.

Implementation: [include/anemone/exl3_records.h](../../include/anemone/exl3_records.h), [include/anemone/exl3_tp.h](../../include/anemone/exl3_tp.h), [source/kernel_exl3_tc.cuh](../../source/kernel_exl3_tc.cuh).

<a id="source-audit-a-i17"></a>

### I17: Confidence-gated expert skip

**Review disposition:** Not implemented.

Current router writes selected expert jobs and draft policy chooses proposal depth; the documented target confidence-sentinel MoE skip/forced rejection is not exposed as a current serving control.

Implementation: [include/anemone/router_records.h](../../include/anemone/router_records.h), [source/router.c](../../source/router.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-i18"></a>

### I18: MoE time at C1

**Review disposition:** needs-runtime-test.

Source identifies expert operand/product/activation/combine work, but no current per-roundC1MoE time is established here.

Implementation: [source/exl3_tc.c](../../source/exl3_tc.c), [source/kernel_exl3_tcp.cuh](../../source/kernel_exl3_tcp.cuh).

<a id="source-audit-a-i19"></a>

### I19: Draft experts

**Review disposition:** Bounded source finding.

DSpark model constants are 128draft experts/top-3; original FP4 draft products and optional draft TP placement are separate from target EXL3 weights.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/router_records.h](../../include/anemone/router_records.h), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-j1"></a>

### J1: Dense kernels decode / prefill

**Review disposition:** needs-runtime-test.

EXL3 trunk-MMA has 1–8 row and 9–64 row forms plus ordered-wide/linear alternatives; dense BF16/FP32 tensors use their separate validated preparations.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [include/anemone/dense_linear_records.h](../../include/anemone/dense_linear_records.h), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-j2"></a>

### J2: FP8 on sm80

**Review disposition:** Bounded source finding.

sm_80 E4M3 decode explicitly places bits and applies power-of-two scales before BF16 MMA/FP32 sums. Exact representable weight recovery is distinct from accumulation-order identity.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [include/anemone/linear_records.h](../../include/anemone/linear_records.h).

<a id="source-audit-a-j3"></a>

### J3: Output projection precision

**Review disposition:** Bounded source finding.

Original output-projection tensors remain checkpoint FP8 codes/scales, decoded for the chosen product. Using a BF16 copy of those weights need not change stored-weight values, but products/orders remain independently qualified.

Implementation: [source/model_v41.c](../../source/model_v41.c), [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [include/anemone/linear_records.h](../../include/anemone/linear_records.h).

<a id="source-audit-a-j4"></a>

### J4: Vocabulary head

**Review disposition:** needs-runtime-test.

EXL3 head stores BF16 weights, shards32320 rows/rank when selected, and dispatches explicitTC/dense-stream forms. Weight precision and deterministic output do not themselves make the target head official-exact.

Implementation: [source/kernel_head_mma.cu](../../source/kernel_head_mma.cu), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-a-j5"></a>

### J5: Trunk MMA / ordered wide

**Review disposition:** needs-runtime-test.

TrunkMMA, ordered-wide, shape-slice, fused andthin-split forms are explicit controls; raw fused/thin-split remain0. Grouped forms preserve their documented per-row instruction order under stated shapes.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h).

<a id="source-audit-a-j6"></a>

### J6: RMSNorm fusion

**Review disposition:** Bounded source finding.

Balanced site/FFN fronts fuse norm with selected mix/router/quantization work, and other prepare kernels retain their stated norm duties.

Implementation: [include/anemone/site_front.h](../../include/anemone/site_front.h), [include/anemone/hyper_connection_records.h](../../include/anemone/hyper_connection_records.h), [include/anemone/attention_state_records.h](../../include/anemone/attention_state_records.h).

<a id="source-audit-a-j7"></a>

### J7: Accumulation

**Review disposition:** Historical claim corrected.

FP32 is common but not universal: EXL3 TC input/activation/combine and prefill combination explicitly use FP64, alongside BF16/FP16 carriers and FP32 products.

Implementation: [source/kernel_exl3_tc.cuh](../../source/kernel_exl3_tc.cuh), [source/kernel_exl3_tcp.cuh](../../source/kernel_exl3_tcp.cuh).

<a id="source-audit-a-j8"></a>

### J8: Tensor-core instructions

**Review disposition:** Bounded source finding.

TC kernels explicitly emitmma.syncm16n8k16 BF16/FP16-class products and staged load/copy forms. Presence of an instruction does not qualify every GPU architecture or model output.

Implementation: [source/kernel_head_mma.cu](../../source/kernel_head_mma.cu), [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [source/kernel_exl3_tcp.cuh](../../source/kernel_exl3_tcp.cuh).

<a id="source-audit-a-j9"></a>

### J9: Libraries inventory

**Review disposition:** Bounded source finding.

Runtime links CUDA driver/libm and loads its own embedded modules; no cuBLAS/CUTLASS/Marlin/NCCL inference-library API is selected in the reviewed source boundary.

Implementation: [Makefile](../../Makefile), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c), [source/tp_exchange.c](../../source/tp_exchange.c).

<a id="source-audit-b-j10"></a>

### J10: Grouped wo_a at 17-64 rows

**Review disposition:** Bounded source finding.

EXL3 grouped trunk forms reuse each decoded E4M3 weight across row groups; grouped 16/24/32/40/48 forms retain the 8-row form arithmetic. The separate relaxed _64 form accepts 9–64 rows and has its own split/reduction order. The old 17.4–38.9 us figures remain a dated trace, not a current measurement.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [source/linear.c](../../source/linear.c).

<a id="source-audit-b-j11"></a>

### J11: Single weight copy

**Review disposition:** Bounded source finding.

The trunk dispatcher consumes the matrix stored as WEIGHT_FORMAT_E4M3_SCALED; the tensor-core kernel decodes it during the product instead of requiring a second persistent expanded BF16 trunk matrix. This is a trunk-format statement, not a claim that every tensor or temporary across the model has one allocation.

Implementation: [source/linear.c](../../source/linear.c), [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu).

<a id="source-audit-b-k1"></a>

### K1: Chunk size

**Review disposition:** Historical claim corrected.

Chunk programs are selected from prepared row inventories. Eligible THIN wide>=2048 may add the fine ladder; published EXL3 BATCH uses wide 1024, tail 128 and fine 0. A 1024 decoder cap is conditional on prefill_first and the compatible bounded recipe; published prefill_first 0 resolves prefill_decode_rows 0, so the old blanket BATCH default cap is stale. Old planning OOM/headroom observations stay historical.

Implementation: [source/main.c](../../source/main.c), [source/engine.c](../../source/engine.c).

<a id="source-audit-b-k2"></a>

### K2: Chunks in flight / tails

**Review disposition:** Bounded source finding.

chunks_in_flight 1 retains the exact chunk planner and its reference-form tail. Tail auto resolution is eligibility-dependent: THIN wide>=2048 gets448; eligible BATCH gets128; incompatible plans get0. Tail sets/overlay/staged dense controls remain selectable. Source arithmetic/planner contracts do not refresh the old first-token timings or establish arbitrary cached seams.

Implementation: [source/main.c](../../source/main.c), [source/engine.c](../../source/engine.c).

<a id="source-audit-b-k3"></a>

### K3: Prefill cone

**Review disposition:** Bounded source finding.

Cone/dead-half skipping is implemented, with exact-plan/margin prerequisites. The builder computes whether a decoder half is wholly dead and trims only live rows for dead-halves2; stable-prefix mode disables that bound. The cone is a dependency optimization, not a license to discard needed target rows.

Implementation: [source/engine.c](../../source/engine.c), [source/main.c](../../source/main.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-k4"></a>

### K4: Integer prep / packing fusion

**Review disposition:** Bounded source finding.

The current EXL3 builder can prepare pool entries/window writes per chunk, compute queries per piece, scan index spans, and write compact half outputs. These controls are wired from the CLI and guarded by applicable chunk/tensor layouts. They are not a single all-40-layer integer-preparation kernel. Historical launch-count/speed savings require their old or fresh traces.

Implementation: [source/main.c](../../source/main.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-k5"></a>

### K5: Split overlap

**Review disposition:** Bounded source finding.

EXL3 split_overlap defaults to V41_DATA_SPLIT_JOINT_DOWN (mode 7). The builder selects a joint-down phase schedule with SIDE/RESUME/JOIN dependencies and skips dead sums where permitted. Explicit other modes remain parser controls. The timing delta quoted in the old row is historical.

Implementation: [source/main.c](../../source/main.c), [source/model_v41_tp2_stage.h](../../source/model_v41_tp2_stage.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-k6"></a>

### K6: Prefill / decode interleave

**Review disposition:** Historical claim corrected.

Chunk/decode scheduling exists, but the old qualified-recipe default description is not the published current EXL3 default: its prefill_first 0 resolves caps/order 0. The default mixed workload has recorded identity failures; rates must stay withheld. The separate prompt_chunks 1 guard prevents runnable prompt chunks donating rows into ordinary BATCH; its functional/identity scope must not be transferred to the default. Original PP5 and exact C1 have separate qualification.

Implementation: [source/main.c](../../source/main.c), [source/engine.c](../../source/engine.c).

<a id="source-audit-b-k7"></a>

### K7: Graphs in prefill

**Review disposition:** Bounded source finding.

step_graphs controls startup graph preparation over the prepared programs, including wide/full programs. Raw CLI omission is0; all three published measured launch commands explicitly use1. Graph capture can fall back eagerly at a BUSY graph-capacity boundary, so a parsed1 alone does not prove every program was captured.

Implementation: [source/step_graphs.c](../../source/step_graphs.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-k8"></a>

### K8: Launches per 30K prompt

**Review disposition:** needs-runtime-test.

The program builder emits per-slab/chunk launches; source can identify fusion and counts in a prepared inventory, but cannot refresh ~330K executed kernels or 4987 launches for a hardware run.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-k9"></a>

### K9: Cold first-run penalty

**Review disposition:** needs-runtime-test.

Graph/module preparation is startup work; no source constant establishes a zero cold-first-run penalty for the default or the old 7–11% observation.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-b-k10"></a>

### K10: Prefill rates

**Review disposition:** needs-runtime-test.

Prefill throughput depends on actual profile, prompt, cache state, hardware and timing window. Current source retains both prepared chunk and exact target paths; historical cold-grid rates cannot be promoted by inspection.

Implementation: [source/engine.c](../../source/engine.c).

<a id="source-audit-b-k11"></a>

### K11: TTFT C1 / C4 / C8

**Review disposition:** needs-runtime-test.

TTFT C1/C4/C8 needs the owning current timing protocol; scheduler ordering and chunk sharing differ by profile, so old THIN/BATCH epochs remain explicitly dated.

Implementation: [source/engine.c](../../source/engine.c).

<a id="source-audit-b-k12"></a>

### K12: Per-layer prefill breakdown

**Review disposition:** needs-runtime-test.

Source distinguishes routed experts, dense/trunk work, attention/index and joins; old per-layer wall-time percentages need a new profile trace.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-k13"></a>

### K13: Engram in prefill

**Review disposition:** Bounded source finding.

Engram rows are gathered on the CPU into each flight staging block before queueing. The TP coordinator later uploads these already gathered rows; submit can overlap with the preceding in-flight step, while TP whole-flight GPU work is serialized. The old 17.4 ms cost remains a trace observation.

Implementation: [source/step_pipeline.c](../../source/step_pipeline.c), [source/step.c](../../source/step.c), [source/engram.c](../../source/engram.c).

<a id="source-audit-b-k14"></a>

### K14: Between-chunk host gap

**Review disposition:** needs-runtime-test.

Engram gathers are performed at submit and graph/upload work remains; neither the old 32 ms gap nor 9.5/4/3.6 ms residuals can be refreshed from source.

Implementation: [source/step_pipeline.c](../../source/step_pipeline.c).

<a id="source-audit-b-k15"></a>

### K15: Prefill margin

**Review disposition:** Bounded source finding.

prefill_margin participates in the exact cone bound; default127 is below the reference cone margin and keeps dependency closure. engine_full_cone_behind returns0 for incompatible/disabled plans. Kept-prefix behavior depends on the restart/cache provenance; old timing and exact-hash evidence stay scoped.

Implementation: [source/engine.c](../../source/engine.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-l1"></a>

### L1: Scheduler

**Review disposition:** Historical claim corrected.

The engine supports THIN and BATCH owners with a prepared row/verifier menu. Current defaults are profile-dependent: published EXL3 uses BATCH, core_once 3, draft_wide 3, pipeline 1 and order 0; original balanced is PP5 THIN with pipeline 5; exact is serial C1 without speculation. The old universal core_once 2/order 2 scheduler text is stale.

Implementation: [source/main.c](../../source/main.c), [source/engine.c](../../source/engine.c).

<a id="source-audit-b-l2"></a>

### L2: Batch sizes / capture sizes

**Review disposition:** Bounded source finding.

Current BATCH menu capacity is eight, and the published EXL3 parser snapshot resolves rows8,8,12,12,16,24,32,48 with verifiers1,2,2,4,4,4,8,8. The engine chooses the first menu entry that can hold both live rows and drafting owners. Exact/original published profiles do not enable this EXL3 menu.

Implementation: [source/engine.c](../../source/engine.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-l3"></a>

### L3: CUDA graphs

**Review disposition:** Bounded source finding.

Graphs are created from explicit prepared launches and branch dependencies, with SIDE/RESUME/JOIN converted into CUDA graph edges. Native counts depend on the prepared program/shape; source confirms launch trims and graph mechanisms, not the historical 1302→1093 or 2238→1860 runtime counts.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-l4"></a>

### L4: Host overhead per step

**Review disposition:** needs-runtime-test.

Host round-boundary/upload/launch overhead is a runtime timing quantity; the source supplies async-copy, graph launch and wait mechanisms, not a current 0.15/0.5 ms bound.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-b-l5"></a>

### L5: Steps in flight

**Review disposition:** Historical claim corrected.

Steps in flight are profile-dependent: original balanced serving automatically selects five PP flights when multi-slot THIN DSpark is used and no explicit depth is given; published EXL3 BATCH retains 1; strict exact requires 1. The blanket THIN2/BATCH1 statement omits the current PP5 gate. TP whole flights remain serialized by the group coordinator.

Implementation: [source/main.c](../../source/main.c), [source/step_pipeline.c](../../source/step_pipeline.c).

<a id="source-audit-b-l6"></a>

### L6: Step time C1 / C2 / C4 / C8 (ms)

**Review disposition:** needs-runtime-test.

Step milliseconds vary with actual row/verifier menu, accepted tokens, pipeline and program costs. The engine records landed step time and adapts menu costs, but historical C1/C2/C4/C8 milliseconds are not refreshed by source inspection.

Implementation: [source/engine.c](../../source/engine.c).

<a id="source-audit-b-l7"></a>

### L7: Lone + multi-session

**Review disposition:** Bounded source finding.

A BATCH program may hold one verifier as well as several; selection is by rows and verifier capacity, so an 8-row program can serve a lone session. The old 319.5 tok/s comparison remains historical, not a current parity claim.

Implementation: [source/engine.c](../../source/engine.c), [source/step.c](../../source/step.c).

<a id="source-audit-b-l8"></a>

### L8: Admission / fairness

**Review disposition:** Historical claim corrected.

Global waiting queue defaults 16 and omitted user queue derives half that (8). Ordering modes/cadence are selectable; current published EXL3 resolves order 0 rather than the conditional bounded order 2. session_grace_seconds and namespace-aware cache/session paths are merged, not branch-only. Scheduler mechanics do not erase current default mixed identity failures or establish universal fairness.

Implementation: [source/server.c](../../source/server.c), [source/engine.c](../../source/engine.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-l9"></a>

### L9: Max concurrent sessions

**Review disposition:** Bounded source finding.

session_slots is a startup allocation, not a universal hardcoded8. Published balanced recipes allocate8 × 65536; strict exact allocates C1/512. Separate PP5 two/four × 1M readiness receipts prove allocation only and do not expand full-context inference qualification.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-b-l10"></a>

### L10: Sampler placement

**Review disposition:** Bounded source finding.

Sampler draws execute on GPU, with optional vocabulary shards, integer-mass top-p and deterministic race keys formed from seed, position and token ID. This makes the sampler row/order-independent for identical logits; it does not make relaxed target logits invariant across batch/cache paths.

Implementation: [source/kernel_sampler.cu](../../source/kernel_sampler.cu), [source/kernel_sampler_shard.cu](../../source/kernel_sampler_shard.cu), [include/anemone/sampler_records.h](../../include/anemone/sampler_records.h).

<a id="source-audit-b-l11"></a>

### L11: Streaming / detokenization

**Review disposition:** Bounded source finding.

The C tokenizer consumes generated binary tables. C HTTP/SSE formatting decodes content, reasoning and tool deltas, buffers stop prefixes and emits completion/usage termination. Four current published-command functional smokes passed; tool/stream edge cases have separate host witnesses.

Implementation: [source/main.c](../../source/main.c), [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-b-l12"></a>

### L12: Tools / reasoning

**Review disposition:** Bounded source finding.

Chat templating, function tools, reasoning-effort mapping and thinking controls are implemented. Stateless Responses conversion is merged as a separate route. Historical Pi zero-error runs are not a new task-quality or Pi end-to-end qualification; current Pi documentation explicitly distinguishes display name and wire alias.

Implementation: [source/chat_completions.c](../../source/chat_completions.c), [source/server_responses_request.c](../../source/server_responses_request.c).

<a id="source-audit-b-l13"></a>

### L13: Uniform vs per-request width

**Review disposition:** Historical claim corrected.

BATCH per-owner depth can be selected from confidence/costs and menu constraints; depth is not unconditionally five. Current EXL3 core_once default is3, with draft_wide 3; original profile restores core_once 2. Session modes6–11 are current code and optional, not unmerged experiments. The published command resolves session mode 0 and ties 0.

Implementation: [source/engine.c](../../source/engine.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-l14"></a>

### L14: Step plan from costs

**Review disposition:** Historical claim corrected.

BATCH costs start from supplied/qualified menu prices and move one eighth toward each landed step. Mode11 additionally records executed (rows,verifiers) shape costs for decode-only observations. Modes10/11 are implemented and parser-selectable on current source, not branch-only; neither is the published default or a general speed winner.

Implementation: [source/engine.c](../../source/engine.c), [source/main.c](../../source/main.c), [tests/draft_session_test.c](../../tests/draft_session_test.c), [tests/engine_batch_test.c](../../tests/engine_batch_test.c).

<a id="source-audit-b-m1"></a>

### M1: Draft method

**Review disposition:** Bounded source finding.

DSpark retains three MTP stages40–42, taps37–39, block5, rank256 and confidence output. CLI supports lookup/DSpark/hybrid source selection; published balanced commands choose DSpark, while strict exact disables speculation. Historical hybrid performance/lookup acceptance is retained as historical.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/draft_policy.c](../../source/draft_policy.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-m2"></a>

### M2: Depth / width

**Review disposition:** Historical claim corrected.

There is no one global fixed-K5 default: non-EXL3 restores draft_policy_default cost 1; EXL3 omitted policy is rate 2 when BATCH or multi-session and fixed 0 otherwise. The three tested published profiles resolve original policy 1, EXL3 policy 2, exact speculation-off; draft_wide 3 and session_acceptance 0 are independent fields.

Implementation: [source/draft_policy.c](../../source/draft_policy.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-m3"></a>

### M3: Positions / tree

**Review disposition:** Bounded source finding.

DSpark uses a five-position block and coupled Markov/sampler passes; source records are linear verifier blocks, not an implemented general branching token tree. Draft depth can be less than five under policy decisions. Strict exact has no speculative chain.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-m4"></a>

### M4: Verification policy

**Review disposition:** Historical claim corrected.

Fixed/cost/rate/adaptive policies are implemented, with target-keyed verification and optional per-session modes through 11. Actual omitted policy depends on placement/workload; fixed-five is selectable, not a universal serving default. All modes remain subject to startup and target numerical qualification.

Implementation: [source/draft_policy.c](../../source/draft_policy.c), [source/engine.c](../../source/engine.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-m5"></a>

### M5: Proposals at T=1

**Review disposition:** Bounded source finding.

Draft acceptance tables have separate greedy/sampled indices, and draft Markov bias feeds the same keyed sampler machinery at positive temperature. This confirms mechanism, not a measured T1 proposal-quality or Pi speed comparison. The strict exact source RNG proof remains T0-only even though native T1 can run.

Implementation: [source/draft_policy.c](../../source/draft_policy.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-m6"></a>

### M6: Rejection / distribution

**Review disposition:** Bounded source finding.

Speculative verification accepts target-keyed matches; saved work depends on drafts, but target results are independent of proposal choice only where the target logits are invariant. Relaxed BATCH/cache paths can change those logits; this conditional statement does not qualify current default mixed execution.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [source/kernel_sampler.cu](../../source/kernel_sampler.cu), [include/anemone/sampler_records.h](../../include/anemone/sampler_records.h), [source/kernel_draft.cu](../../source/kernel_draft.cu).

<a id="source-audit-b-m7"></a>

### M7: Acceptance by content (accepted drafts a step, C1 donor)

**Review disposition:** needs-runtime-test.

Accepted drafts per round are native counters conditioned on actual trajectory, content and policy. Current engine implements them; the old structured/code/prose medians must be replaced or explicitly retained as dated runtime evidence, not inferred from table priors.

Implementation: [source/engine.c](../../source/engine.c), [source/draft_policy.c](../../source/draft_policy.c).

<a id="source-audit-b-m8"></a>

### M8: Tokens per step (pi)

**Review disposition:** needs-runtime-test.

Pi tokens per step is a workload/runtime statistic. Current per-profile sampling, policy, draft depth and cache settings must be held before comparing it; historical Pi runs do not establish new agentic outcome quality.

Implementation: [source/engine.c](../../source/engine.c).

<a id="source-audit-b-m9"></a>

### M9: Draft sampling integration

**Review disposition:** Bounded source finding.

The draft program prepares vocabulary-sharded or ordinary head products, Markov bias then coupled sampler passes; the bias and sampler record share per-position result/state slices. Historical serial milliseconds are not refreshed.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-m10"></a>

### M10: Draft TP / shards / KV

**Review disposition:** Bounded source finding.

EXL3 draft shards/layers are conditional startup controls: draft_shards requires head_shards, and draft_layers requires EXL3 BF16 wire, shards and compatible BATCH draft/head rows. Published EXL3 enables them; original PP5 uses its native draft placement; strict exact speculation is off.

Implementation: [source/main.c](../../source/main.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-m11"></a>

### M11: Verifiers under batching

**Review disposition:** Bounded source finding.

BATCH lays out an anchor and consecutive proposed rows per session, validates flags/positions/depth, and selects programs holding rows and verifier count. Whole-block draft preparation exists. Proposal counts and acceptance remain workload measurements rather than source constants.

Implementation: [source/step.c](../../source/step.c), [source/engine.c](../../source/engine.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-m12"></a>

### M12: Per-session adaptive

**Review disposition:** Historical claim corrected.

Per-session modes6–11 are merged: position-conditional lifting, scout/second/natural probing, residual estimates and executed-shape cost estimates are dispatched in current engine. Published profiles keep session_acceptance 0 and ties 0; availability does not promote optional modes or prove their speed/quality.

Implementation: [source/engine.c](../../source/engine.c), [tests/draft_session_test.c](../../tests/draft_session_test.c), [tests/engine_batch_test.c](../../tests/engine_batch_test.c).

<a id="source-audit-b-m13"></a>

### M13: Confidence head use

**Review disposition:** Bounded source finding.

The draft confidence kernel output is binned by fixed edges, with separate greedy/sampled acceptance tables; cost/rate/adaptive policies consume confidence, and fixed policy returns full permitted depth. There is no blanket confidence gate on all published requests.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [source/draft_policy.c](../../source/draft_policy.c).

<a id="source-audit-b-m14"></a>

### M14: Draft cost a step

**Review disposition:** needs-runtime-test.

Draft-layer, Markov, sampler and wait costs are separate runtime categories. Their implementation is present, but the old 1.07/1.62/1.2 ms timings cannot be refreshed from source.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-m15"></a>

### M15: Draft head precision

**Review disposition:** Bounded source finding.

Draft head uses the same stored BF16 head weights as target head, optionally sharded; head_product writes F32 logits. Current code contains no supplied draft_head_fp8 startup option. The future FP8-head suggestion is a proposal, not a missing release deliverable.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-n1"></a>

### N1: Sampling parameters

**Review disposition:** Historical claim corrected.

HTTP requests accept temperature0..2, top_p>0..1 and uint64 seed; GPU positive-temperature scaling floors at 1e-5. CLI temperature has its separately documented wider parser range. Top-k/penalty execution is absent. Unknown truthy members and named unsupported controls refuse, but generic null/false/zero compatibility members may be accepted; the blanket nothing-silently-ignored claim is too broad.

Implementation: [source/chat_completions.c](../../source/chat_completions.c), [source/main.c](../../source/main.c), [include/anemone/sampler_records.h](../../include/anemone/sampler_records.h), [tests/chat_completions_test.c](../../tests/chat_completions_test.c).

<a id="source-audit-b-n2"></a>

### N2: Seeds

**Review disposition:** Bounded source finding.

Explicit request uint64 seed, absolute position and token ID determine sampler keys for identical logits. An omitted request seed is obtained from Linux getrandom with GRND_NONBLOCK and failure is reported; the stored CLI default seed0 alone is not a fixed per-request seed. Source keyed determinism does not prove target logit invariance.

Implementation: [source/chat_completions.c](../../source/chat_completions.c), [source/kernel_sampler.cu](../../source/kernel_sampler.cu).

<a id="source-audit-b-n3"></a>

### N3: T=0 determinism across runs

**Review disposition:** Bounded source finding.

Fresh 6 November canonical receipts retain two independent boots per own-family prompt/control (8/256 selected records), while 423 donor text/count/finish comparisons per format equal same-job C1. This is scoped T0 evidence; source keys/ordering do not extend it to arbitrary warm-cache, T1 or full-state equality.

Implementation: [source/kernel_sampler.cu](../../source/kernel_sampler.cu), [source/engine.c](../../source/engine.c).

<a id="source-audit-b-n4"></a>

### N4: Batch invariance (C1 vs C8)

**Review disposition:** Bounded source finding.

Sampler output depends only on a row and its seed/position, but target programs/cache choices can change logits. Current donor receipts supply423 full text/count/finish identities per format; the separate default mixed failures remain failures with rates withheld. No source or donor-only evidence establishes universal C1/C8 or warm-cache invariance.

Implementation: [include/anemone/sampler_records.h](../../include/anemone/sampler_records.h), [source/engine.c](../../source/engine.c).

<a id="source-audit-b-n5"></a>

### N5: Chunk invariance

**Review disposition:** Bounded source finding.

The exact chunk planner retains reference-form tails and chooses whole256-row granules from prepared programs. Fresh canonical 2048/1024 checks support the retained selected-record scope; arbitrary cached seams, owner order, forms or full-state equality are separate gates. Strict exact profile itself disables wide chunks.

Implementation: [source/engine.c](../../source/engine.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-n6"></a>

### N6: Logits precision

**Review disposition:** Bounded source finding.

Model logits are F32 with517120 bytes per full129280-token row; ordinary and sharded samplers read F32 logits and form exponentials/race scores in F32, with integer top-p masses and a double threshold product. FP32 output storage does not make upstream original/EXL3 arithmetic identical.

Implementation: [include/anemone/model_layout.h](../../include/anemone/model_layout.h), [include/anemone/sampler_records.h](../../include/anemone/sampler_records.h), [source/kernel_sampler.cu](../../source/kernel_sampler.cu).

<a id="source-audit-b-n7"></a>

### N7: Logprobs

**Review disposition:** Historical claim corrected.

Logprob output is unimplemented. Enabling an unknown truthy logprobs/top_logprobs member is refused400, but generic false/null/zero compatibility values may be accepted and produce no logprobs. The old unqualified every-logprobs-field-refused statement should be narrowed.

Implementation: [source/chat_completions.c](../../source/chat_completions.c), [tests/chat_completions_test.c](../../tests/chat_completions_test.c).

<a id="source-audit-b-o1"></a>

### O1: Reference definition

**Review disposition:** Bounded source finding.

Reference definitions are profile-specific: strict original exact is serial PP5 C1, context<=512, no speculation, official_reference1 and strict controls; original balanced may use numerics exact without exactness exact; EXL3 balanced is relaxed/native reference arithmetic. Native PRO placement changes arithmetic by assigned layer. Existing selected/control evidence is not full official-model equality.

Implementation: [source/main.c](../../source/main.c), [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h).

<a id="source-audit-b-o2"></a>

### O2: Bit-identity tests

**Review disposition:** Historical claim corrected.

Bit-identity proof must name its actual scope: selected records, full conditional heads, persisted-state bytes, or text/count/finish are different witnesses. State diagnostic code uses DIAG_STATE_DIGESTS (without ANEMONE_ prefix), emits session 0 ring/compressor/pool/key blocks of 64 KiB, and does not alone prove every state or official equality. Fresh opening own-family identities remain separately scoped.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-b-o3"></a>

### O3: Lossy / order knobs and defaults

**Review disposition:** Historical claim corrected.

Arithmetic/order defaults depend on placement and explicit profile controls; EXL3 BF16 wire, relaxed products and TC paths are not universal bit-exact defaults. tp_decode_round_once exists on current main with restrictive admission, so branch-only language is stale. Published recipes/real-parser snapshots define active values; optional availability does not promote experiments.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-b-o4"></a>

### O4: Max logit deviation (30K, 8 tokens)

**Review disposition:** needs-runtime-test.

Zero difference of selected logits does not bound max deviation over all 129280 logits. The shipped scorer validates complete F32 conditional captures and reports full-vocabulary delta metrics; numerical values require executed aligned captures, not source inspection or selected records.

Implementation: [tools/golden_score](../../tools/golden_score.py), [source/main.c](../../source/main.c).

<a id="source-audit-b-o5"></a>

### O5: KL vs reference

**Review disposition:** needs-runtime-test.

Full-vocabulary KL scoring is implemented for aligned teacher-forced captures using unit-temperature distributions even though generation is T0. No KL value can be inferred from selected token/logit equality or source; the current full-golden publication owns the measured result.

Implementation: [tools/golden_score](../../tools/golden_score.py).

<a id="source-audit-b-o6"></a>

### O6: Top-1 agreement

**Review disposition:** needs-runtime-test.

Top1 agreement with an own-family retained continuation differs from agreement with official/source golden conditional heads. The scorer defines lowest-ID ties and certain-margin buckets. Current per-position/top1 values require the full-golden captures; selected 256-record identity alone is a narrower executed witness.

Implementation: [tools/golden_score](../../tools/golden_score.py).

<a id="source-audit-b-o7"></a>

### O7: Perplexity / NLL fixture

**Review disposition:** needs-runtime-test.

The shipped golden scorer computes NLL in nats and perplexity as exp(mean NLL), with observed teacher-forced conditioning required. A tool and synthetic host harness do not establish a comparable real-model fixture result; the current full-golden campaign owns that evidence.

Implementation: [tools/golden_score](../../tools/golden_score.py), [tests/teacher_forced_test.c](../../tests/teacher_forced_test.c).

<a id="source-audit-b-o8"></a>

### O8: Long-context quality

**Review disposition:** needs-runtime-test.

Long-context quality is distinct from parser admission, allocation/readiness, short donor identity and strict exact C1/512. The current source allocates context capacities but cannot establish needle/task quality at 128K/512K/1M. These long-term alpha goals remain honestly deferred unless separately run.

Implementation: [source/main.c](../../source/main.c), [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h).

<a id="source-audit-b-o9"></a>

### O9: Agentic outcome

**Review disposition:** needs-runtime-test.

Historical Pi exit/request/error counts establish those executions, not task correctness or a new comparison between policies. New Pi setup schema/keygen checks and four tiny server smokes likewise do not execute Pi tasks or measure agentic outcome quality.

Implementation: [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-b-o10"></a>

### O10: Numerical safety

**Review disposition:** Historical claim corrected.

Sampler NaN keys sort below every numeric logit and positive-temperature weights use fmax(exp(...),0); if every logit is NaN, the defined fallback is token 0, so never-picks-NaN is too broad. Router NaN order is explicitly defined (maximal biased-key order). Checked modules enforce invariants; release hooks compile out checks and logs.

Implementation: [source/kernel_sampler.cuh](../../source/kernel_sampler.cuh), [include/anemone/sampler_records.h](../../include/anemone/sampler_records.h), [source/kernel_router.cuh](../../source/kernel_router.cuh), [include/anemone/hooks.h](../../include/anemone/hooks.h), [tests/sampler_test.c](../../tests/sampler_test.c).

<a id="source-audit-b-o11"></a>

### O11: HumanEval / GSM8K

**Review disposition:** needs-runtime-test.

No current HumanEval/GSM8K outcome follows from source, server startup or donor identity. The available conditional golden scorer measures likelihood/logit metrics, not these downstream task scores. This remains a separate benchmark scope, not an automatically required release change.

Implementation: [tools/golden_score](../../tools/golden_score.py).

<a id="source-audit-b-o12"></a>

### O12: Sensitivity statement

**Review disposition:** needs-runtime-test.

No universal sensitivity conclusion follows from exactness labels or selected-token identity. Sensitivity must be stated for an explicit precision/order change and aligned fixture with numerical/task outcomes; existing native PRO and EXL3 distinctions remain material.

Implementation: [source/main.c](../../source/main.c), [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h).

<a id="source-audit-b-p1"></a>

### P1: Allocator

**Review disposition:** Historical claim corrected.

Model memory is planned into per-card static arenas. Host arenas/staging are aligned/rounded to 2 MiB and request MADV_HUGEPAGE, but failure continues on small pages; source therefore does not guarantee 2 MiB huge pages. CUDA page registration is a separate successful load requirement.

Implementation: [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c), [source/loader.c](../../source/loader.c), [source/backend_cuda.c](../../source/backend_cuda.c).

<a id="source-audit-b-p2"></a>

### P2: Workspace / graph memory

**Review disposition:** Historical claim corrected.

Model weights/state/scratch and eligible tail overlays use prepared arena plans. CUDA graph definitions/executables are separately created/instantiated/uploaded driver objects, not wholly inside the arena; graph table metadata also uses host allocation. Historical BATCH bank/workspace/headroom byte counts need a current plan/runtime receipt for the exact profile.

Implementation: [source/step_graphs.c](../../source/step_graphs.c), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c), [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c).

<a id="source-audit-b-p3"></a>

### P3: Fragmentation over long runs

**Review disposition:** Historical claim corrected.

Static GPU model/state/scratch plans avoid a per-token device allocator, but no-fragmentation for the entire process or driver is too broad. Server requests, graph definitions, snapshot metadata and other host/driver resources have their own allocations/lifetimes. Long-run stability still needs an executed soak.

Implementation: [source/step_graphs.c](../../source/step_graphs.c), [source/server.c](../../source/server.c), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-b-p4"></a>

### P4: Headroom at 1M

**Review disposition:** needs-runtime-test.

Free headroom is an actual hardware allocation/readiness measurement for an exact profile. Fresh PP5 two/four × 1M readiness is allocation-only; it neither refreshes all old per-card free-MiB cells nor proves full-context numerical/task quality. Strict exact remains C1/512.

Implementation: [source/main.c](../../source/main.c), [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c).

<a id="source-audit-b-p5"></a>

### P5: Host pinned buffers

**Review disposition:** Bounded source finding.

Backend staging uses mmap plus portable cuMemHostRegister; row gathers populate step staging and the coordinator uploads them. Host-only Engram storage and transient pinned buffers have different ownership/lifetimes; source confirms data path, not current pinning throughput.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c), [source/step.c](../../source/step.c), [source/step_pipeline.c](../../source/step_pipeline.c).

<a id="source-audit-b-q1"></a>

### Q1: Kernels a decode step / a prefill token

**Review disposition:** needs-runtime-test.

Prepared launches and graph trims exist, but executed kernels per decode step/prefill token depend on chosen shape, branches and profile. Current source cannot refresh historical1225–1345/1093/1860/~330K counts.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-q2"></a>

### Q2: Launch overhead / persistent

**Review disposition:** needs-runtime-test.

The backend launches prepared CUDA graphs or explicit kernels; no native persistent inference-kernel controller is exposed by these paths. Historical graph-node/removed-launch microseconds require runtime measurement and cannot be extrapolated to new shapes or hardware.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-b-q3"></a>

### Q3: CTA / occupancy

**Review disposition:** Bounded source finding.

Kernel launch geometry is explicit: sampler uses1024 threads, EXL3 prefill operands map tasks to blocks and GEMMs have one/two-block variants. Geometry is source-audited; achieved occupancy, register spills and performance are compiler/device observations, not guaranteed by a comment. Original/exact paths differ from EXL3 TC dispatch.

Implementation: [include/anemone/sampler_records.h](../../include/anemone/sampler_records.h), [source/kernel_exl3_tcp.cu](../../source/kernel_exl3_tcp.cu), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-q4"></a>

### Q4: Fusion inventory

**Review disposition:** Bounded source finding.

Current builder fuses eligible site fronts, pool/window work, query/index spans and joint-down schedules; trims and switches are explicit. Fusion applies by format/shape/profile and cannot be stated as one global program shared by exact, original and EXL3. Historical savings remain separately dated.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-q5"></a>

### Q5: FP8 handling on sm80

**Review disposition:** Bounded source finding.

On sm80 the stored E4M3 trunk bytes/scales are decoded to BF16 operands and BF16 tensor-core MMA produces F32 accumulations; this is software FP8 weight handling, not native sm80 FP8 instructions. Native sm120a module selection and PRO placement retain their own arithmetic classes.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [Makefile](../../Makefile).

<a id="source-audit-b-q6"></a>

### Q6: Digest / verification kernels

**Review disposition:** Historical claim corrected.

Pack digest kernels validate load-time data. State diagnostics are enabled by DIAG_STATE_DIGESTS, not ANEMONE_DIAG_STATE_DIGESTS, and emit 64 KiB blocks of session 0 ring/compressor/pool/key ranges after generation. Historical counts 115 and diagnostic overhead require the actual program/run, not source inference.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-q7"></a>

### Q7: Compile flags

**Review disposition:** Bounded source finding.

Actual flags include host C11 -O3 -fno-tree-vectorize -flto=auto -ffp-contract=off, hardening and release hooks0/0; CUDA uses C++17 -O3 --fmad=false and warnings-as-errors for sm80/sm120a. Clean source-export build actually passed with recorded CUDA13.4/GCC15.2 toolchain. No source-RNG or numerical equivalence follows from flags alone.

Implementation: [Makefile](../../Makefile).

<a id="source-audit-b-q8"></a>

### Q8: Kernel languages

**Review disposition:** Bounded source finding.

Device kernels are checked-in CUDA C++ sources compiled to embedded modules. Host engine/server are C11; optional model producers/scorers/generators use Python. Kernel languages CUDA C++ only is valid when scoped to device kernels, not the entire repository.

Implementation: [Makefile](../../Makefile), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-b-r1"></a>

### R1: Process model

**Review disposition:** Bounded source finding.

Serving is one native process with an engine pthread. TP flights use one group coordinator that owns all cards and promises no GPU overlap; PP pipeline instead uses per-card workers where applicable. Four cards is specific to EXL3 TP4, while original published PP5 uses five.

Implementation: [source/engine.c](../../source/engine.c), [source/step_pipeline.c](../../source/step_pipeline.c).

<a id="source-audit-b-r2"></a>

### R2: CPU threads

**Review disposition:** Bounded source finding.

CPU threads depend on mode: loader reader threads transfer/check pieces; engine owns its stepping thread; pipeline may create one TP group worker or one worker per PP card. tp_parallel_launch is an optional control. There is no single fixed thread count for all profiles.

Implementation: [source/loader.c](../../source/loader.c), [source/engine.c](../../source/engine.c), [source/step_pipeline.c](../../source/step_pipeline.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-r3"></a>

### R3: Tokenizer

**Review disposition:** Historical claim corrected.

Runtime uses the project C tokenizer over converted tokenizer.bin/front_end.tables with validated binary header/magic/version/counts. Official tokenizer.json is the producer input, not the file consumed directly by the inference tokenizer. The bounded producer regenerated the actual binary byte-identically.

Implementation: [source/main.c](../../source/main.c), [source/tokenizer.c](../../source/tokenizer.c).

<a id="source-audit-b-r4"></a>

### R4: HTTP / OpenAI API

**Review disposition:** Bounded source finding.

The native C server exposes health/models, Chat Completions, SSE, named-user auth/stats and merged stateless Responses. Four literal published-command smokes passed health/models/tiny nonstream T0; host tests cover many protocol/refusal cases. This is not a fresh SDK/tool-cycle or Pi task qualification.

Implementation: [source/server.c](../../source/server.c), [source/chat_completions.c](../../source/chat_completions.c), [source/server_responses_request.c](../../source/server_responses_request.c).

<a id="source-audit-b-r5"></a>

### R5: Logging / observability

**Review disposition:** Historical claim corrected.

Release hooks0/0 compile out LOG_ERROR and warning macros as well as diagnostic logs. Explicit product fprintf/fputs startup/version/errors and live stats remain separate code paths; errors never dropped applies only when logging is compiled. Per-program BATCH cost/stat reporting does not change that distinction.

Implementation: [include/anemone/hooks.h](../../include/anemone/hooks.h), [Makefile](../../Makefile), [source/main.c](../../source/main.c), [tests/hooks_test.c](../../tests/hooks_test.c).

<a id="source-audit-b-r6"></a>

### R6: Health / restart

**Review disposition:** needs-runtime-test.

Health readiness exists and four current profiles returned HTTP200. Full cold model restart time is hardware/storage/profile-dependent; the old ~11.6min value is not a current guarantee. Existing installed workers are not modified by this audit.

Implementation: [source/chat_completions.c](../../source/chat_completions.c), [source/loader.c](../../source/loader.c).

<a id="source-audit-b-r7"></a>

### R7: Defaults selection

**Review disposition:** Historical claim corrected.

Defaults are selected by placement, axes, compatible recipe and explicit controls; current EXL3 balanced uses relaxed arithmetic. Fastest bit-identical path default is not a universal promise, especially with default mixed identity failure and placement-dependent native PRO arithmetic. Real parser snapshots and strict exact refusals define the contract.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-b-s1"></a>

### S1: Max model length / 1M slots

**Review disposition:** Bounded source finding.

Context/session capacities are startup allocations. Published balanced commands use8 × 65536, strict exact C1/512; current original PP5 two/four × 1048576 readiness receipts expand allocation evidence only. Full-context inference, all-slot quality and arbitrary EXL3 multi-1M capacity remain separate gates.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-b-s2"></a>

### S2: Prefix cache

**Review disposition:** Historical claim corrected.

Exact-prefix modes, stable-prefix controls, namespace scope and session forks are merged current code, not branch-only. Admission restricts stable/exact/fork reuse to compatible EXL3 plans; published balanced uses legacy reuse 1/exact 0/stable 0/fork 0, strict exact disables reuse. Default mixed and arbitrary warm/cold invariance remain unqualified/failing as separately recorded.

Implementation: [source/main.c](../../source/main.c), [source/server_cache.c](../../source/server_cache.c), [source/server_forks.c](../../source/server_forks.c).

<a id="source-audit-b-s3"></a>

### S3: Multi-turn reuse

**Review disposition:** Historical claim corrected.

Namespace-aware session reuse/snapshot v4, grace controls and save/load thresholds are merged. Snapshots bind model/tokenizer identity, namespace, lengths and payload; old-version files are isolated. Defaults include save 1024/load 256/idle 120. Exact profile remains restricted and measured old transfer latencies remain historical page-cache observations.

Implementation: [source/main.c](../../source/main.c), [source/server_sessions.c](../../source/server_sessions.c), [source/server_cache.c](../../source/server_cache.c).

<a id="source-audit-b-s4"></a>

### S4: Tool calling

**Review disposition:** Historical claim corrected.

Stateless full-history /v1/responses is merged alongside Chat; store:true and nonnull previous_response_id/conversation are refused. Native tools/SSE formatting exists. The old main-remains-Chat-only/branch-not-merged text is false. Host conversion/refusal tests are executed; current tiny profile smokes exercised Chat, not a new full Responses/SDK tool cycle.

Implementation: [source/server.c](../../source/server.c), [source/server_responses_request.c](../../source/server_responses_request.c), [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-b-s5"></a>

### S5: Reasoning effort

**Review disposition:** Bounded source finding.

Reasoning aliases map none/off0, minimal/low50, medium/high/xhigh75, max100, or integer0..100. thinking accepts only enabled/disabled; disabled forces0, enabled with effective0 refuses. Published recipes explicitly choose0. These controls render prompt/reasoning output rather than implementing arbitrary thinking-token budgets.

Implementation: [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-b-s6"></a>

### S6: Streaming / stop

**Review disposition:** Bounded source finding.

SSE streaming/usage/DONE and stop matching are implemented. Request stops accept one string or up to 4 strings of 1..256 bytes; formatter holds partial stop/reasoning/tool prefixes before emission and cancellation. Tiny published smokes are nonstream, so host stream/stop witnesses remain explicitly separate.

Implementation: [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-b-s7"></a>

### S7: JSON / structured output

**Review disposition:** Historical claim corrected.

Only response_format type text is supported; JSON/schema-constrained response modes are refused. Function tools allow strict:false or strict:null; strict:true is refused. Thus false-only is slightly too narrow. General JSON constrained decoding remains an intentionally unimplemented optional feature, not a missing release prerequisite.

Implementation: [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-b-s8"></a>

### S8: n > 1, logprobs, cancellation

**Review disposition:** Bounded source finding.

n must be1; enabled logprob output is unsupported, with disabled compatibility values governed by generic member handling. Closing/resetting the connection requests engine cancellation, which is observed at engine scheduling/retirement boundaries rather than instant GPU preemption.

Implementation: [source/chat_completions.c](../../source/chat_completions.c), [source/server.c](../../source/server.c), [source/engine.c](../../source/engine.c), [tests/chat_completions_test.c](../../tests/chat_completions_test.c).

<a id="source-audit-b-s9"></a>

### S9: Aliases

**Review disposition:** Bounded source finding.

Startup model_alias is accepted after nonempty/bounded validation, listed alongside model name and recognized in requests. Published recipes use wire ID anemone; Pi display name Anemone DeepSeek-V4.1-Flash-1017 is client metadata and is distinct from the wire ID. Four published smokes verified listing/request alias behavior.

Implementation: [source/server.c](../../source/server.c), [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-b-s10"></a>

### S10: Images

**Review disposition:** Bounded source finding.

Image routing, PNG/JPEG decode/preprocess and vision units are implemented with bounded data URLs. Remote HTTP(S) image URLs are explicitly refused instead of fetched. Text-only profile documentation does not qualify image task quality or convert tiny text smokes into vision coverage.

Implementation: [source/chat_completions.c](../../source/chat_completions.c), [source/chat_images.c](../../source/chat_images.c).

<a id="source-audit-b-s11"></a>

### S11: Authentication

**Review disposition:** Bounded source finding.

key_file loads a regular nonempty file with no group/other permission bits; the documented setup creates0600, but source does not enforce an owner UID or exact 0600 bits. Named-user lists are validated (one legacy key line remains compatible). Keygen obtains32 random bytes through Linux getrandom(...,0), retries partial/EINTR, returns errors and zeros its temporary buffer. Disposable CLI/native-parser checks passed; no real key/config was read and crypto strength was not benchmarked.

Implementation: [source/server_keys.c](../../source/server_keys.c), [source/server.c](../../source/server.c).

<a id="source-audit-b-s12"></a>

### S12: Engram placement fallback

**Review disposition:** Bounded source finding.

Engram table units are host-only pack extents whose rows are loaded into host RAM; CPU gathers copy them into staging for upload. No current CLI tier silently falls back to an NVMe row store during inference. Source loading/storage access is distinct from runtime gather placement.

Implementation: [source/engram.c](../../source/engram.c), [source/loader.c](../../source/loader.c).

<a id="source-audit-b-s13"></a>

### S13: Without P2P

**Review disposition:** Historical claim corrected.

No-P2P behavior is placement/component-specific. Reusable ordered collectives can prepare explicit host routes/staged fallback, but the current EXL3 TP model requires qualified peer writer masks and peer_connect; CUDA returns STATUS_UNSUPPORTED when peer access is absent. Therefore blanket host-route support for every published placement is false. PP host handoffs are a separate data path.

Implementation: [source/ordered_collective.c](../../source/ordered_collective.c), [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c), [source/backend_cuda.c](../../source/backend_cuda.c).

<a id="source-audit-b-t1"></a>

### T1: Watts a card, decode C1 / C8

**Review disposition:** needs-runtime-test.

Decode watts per card require actual telemetry over the admitted performance window; neither configured limits nor clocks are measurements of power.draw. Current runtime receipts must bind physical device IDs and concurrent window before deriving watts.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c).

<a id="source-audit-b-t2"></a>

### T2: Watts a card, prefill

**Review disposition:** needs-runtime-test.

Prefill watts per card likewise require recorded physical-device power.draw over that prefill window; source/backend readiness cannot supply it.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c).

<a id="source-audit-b-t3"></a>

### T3: tok/s per W

**Review disposition:** needs-runtime-test.

Tokens per joule/watt requires a measured generation numerator and power/time denominator from the same window. Configured180 W multiplied by wall time is not automatically measured energy.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c).

<a id="source-audit-b-t4"></a>

### T4: Power cap policy

**Review disposition:** needs-runtime-test.

The operating power-cap policy is owner/harness authorization, not an inference-source constant. Current receipts must list all physical caps, including the native PRO standing cap, and restoration; do not silently generalize an old180 W comparison to every device.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c).

<a id="source-audit-b-t5"></a>

### T5: Idle power / clocks

**Review disposition:** needs-runtime-test.

Clock-lock and reset policy belongs to the admitted runner. The historical1695 MHz line is not a current engine setting or evidence of measured clocks; record requested and observed clocks and restoration separately.

Implementation: [source/backend_cuda.c](../../source/backend_cuda.c).

<a id="source-audit-b-u1"></a>

### U1: Known failure modes

**Review disposition:** Historical claim corrected.

Known limitations must be current and profile-scoped: default EXL3 mixed identity failures remain rates-withheld, while guard and original PP5 results are separate. Cache/namespace/Responses/profile code is now merged, and original PP5 two/four × 1M allocation has new readiness receipts. Historical OOM/headroom and branch-only cells are not a current universal failure list.

Implementation: [source/main.c](../../source/main.c), [source/engine.c](../../source/engine.c), [source/server_responses_request.c](../../source/server_responses_request.c).

<a id="source-audit-b-u2"></a>

### U2: Long-run stability

**Review disposition:** needs-runtime-test.

Long-run model/server stability requires an actual timed workload with memory/errors/cleanup observed. Historical Pi runs and the four tiny current functional smokes do not constitute a20minute C8 soak. No new soak is claimed or launched by this source audit.

Implementation: [source/server.c](../../source/server.c), [source/engine.c](../../source/engine.c).

<a id="source-audit-b-u3"></a>

### U3: Error rates

**Review disposition:** needs-runtime-test.

Error rates need an observed request denominator, request types and actual response outcomes. Four current tiny smokes yielded12 HTTP 200s; historical Pi0failed applies only to those runs and says nothing about task outcome or all API modes.

Implementation: [source/server.c](../../source/server.c), [source/chat_completions.c](../../source/chat_completions.c).

<a id="source-audit-b-u4"></a>

### U4: Determinism across restarts

**Review disposition:** Bounded source finding.

Fresh6 November own-family canonical two-boot selected records and846 donor comparisons retain scoped T0 restart/concurrency evidence. Exact-prefix/stable/namespace controls are merged, so old branch-only phrasing is stale, but arbitrary warm/cold restart invariance and official full-state equality remain unqualified.

Implementation: [source/main.c](../../source/main.c), [source/server_sessions.c](../../source/server_sessions.c), [source/engine.c](../../source/engine.c).

<a id="source-audit-b-u5"></a>

### U5: Qualification coverage

**Review disposition:** Bounded source finding.

Qualification consists of separately scoped host, card, selected-control, conditional-head, donor and API tests. Fresh exported6b9 release build ran125 host programs plus594 header compilations; raw parser snapshots and201override probes are real executed host evidence. CUDA form tests and full-golden metrics require their own retained runtime receipts.

Implementation: [tests/model_v41_draft_program_test.c](../../tests/model_v41_draft_program_test.c), [tests/teacher_forced_test.c](../../tests/teacher_forced_test.c).

<a id="source-audit-b-u6"></a>

### U6: Process hygiene

**Review disposition:** Bounded source finding.

Owned PID/executable/start-tick retirement and all-five-standing-cap restoration are independently recorded for current opening jobs and four functional profile smokes. These are runner receipts, not behavior of model arithmetic, and this audit starts no worker or changes no cap.

Implementation: [source/engine.c](../../source/engine.c).

<a id="source-audit-b-z1"></a>

### Z1: Tables

**Review disposition:** Bounded source finding.

The retained pack has two host-only Engram tables (layers 1 and 14),188.8 GiB aggregate; packed rows are264 B and actual configured gather is24 rows/token/table. Runtime derives table/row counts from validated constants instead of hardcoding every model. Model-data hashes/manifest and bounded producer proof retain actual artifacts.

Implementation: [source/engram.c](../../source/engram.c), [include/anemone/engram.h](../../include/anemone/engram.h), [tests/teacher_forced_test.c](../../tests/teacher_forced_test.c).

<a id="source-audit-b-z2"></a>

### Z2: Placement

**Review disposition:** Bounded source finding.

Engram table units must be host-only extents; loader reads them into retained host arena RAM before runtime gather. Host placement is not on-demand NVMe paging, even when the source pack lives on NVMe. Hugepage advice can fall back to small pages.

Implementation: [source/engram.c](../../source/engram.c), [source/loader.c](../../source/loader.c).

<a id="source-audit-b-z3"></a>

### Z3: Gather path

**Review disposition:** Bounded source finding.

CPU prefetch/gather copies each validated row from the retained host table into step staging, then asynchronous uploads bring the selected bytes to each card. The proposed device-side/UVA gather is not the current published implementation.

Implementation: [source/engram.c](../../source/engram.c), [source/step.c](../../source/step.c).

<a id="source-audit-b-z4"></a>

### Z4: Overlap with compute

**Review disposition:** Bounded source finding.

TP flight submit computes row IDs and gathers Engram rows before queueing; this CPU work may overlap the preceding in-flight step. The coordinator uploads already-gathered rows and serializes whole TP GPU flights. Thus gather overlap exists without claiming concurrent TP GPU flight execution.

Implementation: [source/step_pipeline.c](../../source/step_pipeline.c).

<a id="source-audit-b-z5"></a>

### Z5: Prefill handling

**Review disposition:** Bounded source finding.

Staging bytes scale as row_count×rows_per_token×264 B per used table/card; published EXL3 wide 1024 differs from historical THIN2048. Gathering at submit removes that gather from the coordinator inter-step path, but old2×13MB/~17 ms timing observations are historical rather than current measured costs.

Implementation: [source/step.c](../../source/step.c), [source/step_pipeline.c](../../source/step_pipeline.c), [source/engram.c](../../source/engram.c).

<a id="source-audit-b-z6"></a>

### Z6: Worker restart

**Review disposition:** Bounded source finding.

Model worker startup loads host-only table extents into its own arena; there is no resident cross-worker Engram service in this source path. OS page-cache warmness may affect reload time but is not an inference placement change or measured cold-device latency.

Implementation: [source/loader.c](../../source/loader.c).

<a id="source-audit-b-aa1"></a>

### AA1: C1 round / step (traced; untraced)

**Review disposition:** needs-runtime-test.

C1 total step time is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/engine.c](../../source/engine.c).

<a id="source-audit-b-aa2"></a>

### AA2: C1 routed MoE

**Review disposition:** needs-runtime-test.

C1 routed MoE cost is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-aa3"></a>

### AA3: C1 all-reduce

**Review disposition:** needs-runtime-test.

C1 all-reduce/wait cost is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-aa4"></a>

### AA4: C1 FP8 trunk / dense

**Review disposition:** needs-runtime-test.

C1 trunk/dense cost is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu).

<a id="source-audit-b-aa5"></a>

### AA5: C1 mHC + router

**Review disposition:** needs-runtime-test.

C1 mHC/router cost is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/kernel_router.cu](../../source/kernel_router.cu).

<a id="source-audit-b-aa6"></a>

### AA6: C1 attention

**Review disposition:** needs-runtime-test.

C1 attention cost is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h).

<a id="source-audit-b-aa7"></a>

### AA7: C1 head + draft + sampling

**Review disposition:** needs-runtime-test.

C1 head/draft/sampling cost is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-aa8"></a>

### AA8: C8 step (structured, step = ms per draft step x users)

**Review disposition:** needs-runtime-test.

C8 BATCH step milliseconds is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/engine.c](../../source/engine.c).

<a id="source-audit-b-aa9"></a>

### AA9: C8 step composition (traced)

**Review disposition:** needs-runtime-test.

C8 traced category composition is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-aa10"></a>

### AA10: 30K prefill total (one card)

**Review disposition:** needs-runtime-test.

30K prefill total wall time is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/engine.c](../../source/engine.c).

<a id="source-audit-b-aa11"></a>

### AA11: 30K prefill experts

**Review disposition:** needs-runtime-test.

30K prefill expert time is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/kernel_exl3_tcp.cu](../../source/kernel_exl3_tcp.cu).

<a id="source-audit-b-aa12"></a>

### AA12: 30K prefill non-expert dense

**Review disposition:** needs-runtime-test.

30K non-expert dense time is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu).

<a id="source-audit-b-aa13"></a>

### AA13: 30K prefill attention + indexer

**Review disposition:** needs-runtime-test.

30K attention/index time is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-aa14"></a>

### AA14: 30K prefill communication exposed

**Review disposition:** needs-runtime-test.

30K exposed communication/waits is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-aa15"></a>

### AA15: Kernels a prefill token

**Review disposition:** needs-runtime-test.

Kernels per prefill token is a hardware/profile trace metric. Current source has the cited work category, but the old traced/untraced times, call counts, percentages and comparative speed gaps remain dated evidence. This source audit does not rerun, rescale or promote them; competitor historical cells remain intact.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-ac1"></a>

### AC1: Packaging

**Review disposition:** Bounded source finding.

Current Anemone release packaging is one native executable plus source/docs and separately prepared model/tokenizer/map inputs. A Python-free/Git-free isolated source-export build actually passed; the competitor22GBimage/private-stage cell remains historical external evidence, not re-audited here.

Implementation: [Makefile](../../Makefile), [source/main.c](../../source/main.c).

<a id="source-audit-b-ac2"></a>

### AC2: Lanes

**Review disposition:** Historical claim corrected.

Anemone now implements four orthogonal axis controls and explicit original-exact/original-balanced/EXL3-balanced measured recipes; presets planned is stale. Original balanced numerics exact is not strict exactness exact. Strict exact is serial C1/512; native PRO layer placement remains arithmetic-dependent. Historical Curtis lanes are unchanged.

Implementation: [source/main.c](../../source/main.c).

<a id="source-audit-b-ac3"></a>

### AC3: KV

**Review disposition:** Bounded source finding.

Anemone attention ring/pool/query storage is FP16, with BF16 output and profile/order-dependent probability rounding. FP16 has a different mantissa/exponent tradeoff from BF16; more precise must not be read as universal model-quality/equivalence. Cache precision and placement are separate from allocated context capacity. Curtis historical KV cell is retained.

Implementation: [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h).

<a id="source-audit-b-ac4"></a>

### AC4: Dense / projections

**Review disposition:** Bounded source finding.

Anemone has its own stored-E4M3 trunk products with explicit ordered and tensor-core grouped forms; arithmetic order varies by profile/knob. Source confirms dispatch/mechanism, while parity-at-C1 is a separately scoped benchmark/numerical claim, not an all-format theorem. Historical Curtis/GNmac text is unchanged.

Implementation: [source/kernel_trunk_mma.cu](../../source/kernel_trunk_mma.cu), [source/linear.c](../../source/linear.c).

<a id="source-audit-b-ac5"></a>

### AC5: Experts by batch size

**Review disposition:** Historical claim corrected.

EXL3 BF16/TC dispatch uses small-row routed TC kernels and separate prefill-sized TC programs when admitted. Anemone does not use TC MoE for every <=64 row call across original/exact profiles; main masks EXL3-only controls and the strict original path uses its own arithmetic. Historical Curtis path table remains unchanged.

Implementation: [source/main.c](../../source/main.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-ac6"></a>

### AC6: Attention

**Review disposition:** Bounded source finding.

Anemone attention has fixed-order and reference-order score/value forms, with native/emulated PRO class-specific modules and row-local contracts. Which path is active is profile-dependent. Parity at C1 remains scoped runtime evidence, not a replacement for the strict exact or mixed/cache gate. Historical Curtis attention description is preserved.

Implementation: [include/anemone/attention_core_records.h](../../include/anemone/attention_core_records.h).

<a id="source-audit-b-ac7"></a>

### AC7: Collectives

**Review disposition:** Bounded source finding.

Anemone TP uses its own peer exchanges/owner sums/ring variants; reusable ordered collectives separately support explicit direct/host routes. Current EXL3 load still requires qualified peer masks, so a generic no-P2P host fallback cannot be inferred. Historical Curtis transport/NUMA/NCCL description is not requalified.

Implementation: [source/main.c](../../source/main.c), [source/model_v41_tp2_model.c](../../source/model_v41_tp2_model.c), [source/ordered_collective.c](../../source/ordered_collective.c).

<a id="source-audit-b-ac8"></a>

### AC8: Prefill

**Review disposition:** Historical claim corrected.

Current EXL3 exact-plan mechanics include cone/margin127, chunks_in_flight 1, pool/query/index chunk work and split mode 7, not the stale mode 6 current-default text. Published measured BATCH uses wide 1024/tail 128/fine 0; THIN2048 is a separate recipe. Strict original exact disables wide/full prefill. Historical Curtis MOE_RAGGED cell is retained.

Implementation: [source/main.c](../../source/main.c), [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-ac9"></a>

### AC9: Draft / sampler

**Review disposition:** Bounded source finding.

Anemone uses stored BF16 target/draft head weights with optional EXL3 vocabulary shards and Markov-bias/sampler passes. Strict exact has speculation off; original and EXL3 balanced use their own placement. This is not the competitor FP8 draft-head recipe. Historical Curtis/GNmac cell remains unchanged.

Implementation: [source/model_v41_program.c](../../source/model_v41_program.c).

<a id="source-audit-b-ac10"></a>

### AC10: Verification policy

**Review disposition:** Historical claim corrected.

Anemone fixed policy 0 is selectable but omitted policy depends on placement/workload: original restores cost 1, EXL3 BATCH/multi-session rate 2, EXL3 single-session/noBATCH fixed 0. Published commands and parser snapshots define actual controls. Historical Curtis fixed 5/controller conclusions remain external historical evidence.

Implementation: [source/draft_policy.c](../../source/draft_policy.c), [source/main.c](../../source/main.c).

<a id="source-audit-b-ac11"></a>

### AC11: T=1 proposals

**Review disposition:** Bounded source finding.

Anemone target draws are keyed by seed/position/token, and draft tables split greedy/sampled modes. A target sampler guarantee holds for identical logits; current T1 native execution is not source RNG equality or a measured proposal-quality win. Historical Curtis unseeded sampled/greedy comparisons remain unchanged.

Implementation: [source/kernel_sampler.cu](../../source/kernel_sampler.cu), [source/draft_policy.c](../../source/draft_policy.c).

<a id="source-audit-b-ac12"></a>

### AC12: Graphs / memory

**Review disposition:** Historical claim corrected.

Anemone prepares model/state/scratch arenas and startup graph tables, but CUDA graph definitions/executables are driver-created objects, so graph memory entirely inside a static arena is too broad. Graph counts, driver overhead and remaining headroom require current runtime receipts. Historical Curtis graph-bank sizes remain unchanged.

Implementation: [source/step_graphs.c](../../source/step_graphs.c), [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-b-ac13"></a>

### AC13: Hot swap

**Review disposition:** Not implemented.

Anemone does not expose a runtime drained lane hot-swap controller equivalent to the historical competitor node-swap mechanism. It prepares and uploads graphs at startup and destroys them at card close. This is an intentionally absent feature; old11.6minute reload is a historical runtime observation, not a current guaranteed delay.

Implementation: [source/backend_cuda_launch.c](../../source/backend_cuda_launch.c).

<a id="source-audit-b-ac14"></a>

### AC14: Qualification

**Review disposition:** Bounded source finding.

Anemone qualification now has an actual exported-source build,125 release host programs,594 header checks, parser/alias/refusal snapshots and four published-profile functional smokes, plus separately pinned card/donor/golden evidence. Source audit is not an extra GPU pass and historical competitor gate counts are neither copied nor rerun.

Implementation: [tests/model_v41_draft_program_test.c](../../tests/model_v41_draft_program_test.c), [tests/teacher_forced_test.c](../../tests/teacher_forced_test.c).

<a id="source-audit-b-ac15"></a>

### AC15: Precision evaluation

**Review disposition:** Historical claim corrected.

Bitwise identity to the original without scope is too strong. Strict original exact is admitted C1/512/T0 source-fixture scope; original balanced and EXL3 balanced have distinct arithmetic/placement. Selected own-family records, donor text identity and full conditional-head metrics are different witnesses and none creates universal official equality. Historical Curtis precision evaluation is retained unchanged.

Implementation: [source/main.c](../../source/main.c), [tools/golden_score](../../tools/golden_score.py).

<a id="source-audit-b-ac16"></a>

### AC16: Engram

**Review disposition:** Bounded source finding.

Anemone retains host-RAM Engram tables and gathers rows on CPU at flight submit for upload; it does not run the historical competitor resident tmpfs/UVA gather/replacement service. Source pack storage location and runtime host table placement are distinct. Historical competitor placement text is preserved.

Implementation: [source/engram.c](../../source/engram.c), [source/step_pipeline.c](../../source/step_pipeline.c).

<a id="source-audit-b-ac17"></a>

### AC17: Weak spots

**Review disposition:** Bounded source finding.

This historical competitor weak-spots row has no Anemone comparison text to validate. Current Anemone caveats remain explicit elsewhere: default mixed identity rates withheld, strict exact C1/512 only, and1M allocation/readiness not full-context quality. No claim is made about current competitor rates, topology, private controls or regressions.

Implementation: [source/main.c](../../source/main.c), [source/engine.c](../../source/engine.c).

