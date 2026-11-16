<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Release notes

## v0.0.3-alpha

Anemone v0.0.3-alpha is a C11/CUDA engine for DeepSeek-V4.1-Flash-1017, on four CMP 170HX cards for EXL3, and on four
CMP 170HX plus an RTX PRO 6000 Blackwell for the original weights. The [README](README.md) has the overview, the
1M-session table and the first-time setup.

### Serving
- **1M tokens for every session.** `--context_tokens` defaults to 1,048,576 and `--session_slots` to `max`: the
  startup fit serves as many full-context sessions as the cards hold, and is never shrunk automatically.
- **GPU split plus four axes.** `--placement` (PP5, TP4_PRO, TP4_EXL3) picks the split. `--category`, `--exactness`,
  `--concurrency` and `--focus` each carry a measured knob set per split. `--weights` and `--model_dir` name the weight
  family and the model folder.
- **Opt-in Anthropic `POST /v1/messages`**, with thinking controls and the same `/metrics` accounting as chat
  completions.
- **Opt-in Prometheus `/metrics`**, matching vLLM V1's metric names.

### Loading and storage
- **Turbo load.** Verified replicas on every local NVMe drive are found by their receipts and read in parallel with
  O_DIRECT and io_uring. The TP host arena steals work between copies. `anemone replicate --adopt` admits an existing
  copy after a full digest check.
- **Turbo Engram.** The Engram tables stay in RAM when they fit. Otherwise `auto` keeps the largest share that does and
  reads the rest from NVMe a step at a time on io_uring threads per copy, every block verified and a stalled read
  retried on another copy. The decision and its budget are printed at startup.
- **`anemone autotune` in stages.** `--stage=copy` prepares and verifies the replicas (24 workers by default);
  `--stage=load` tunes the TP host load into a profile section, enabled by `--qualify`.
- **New load defaults** ([STORAGE_PROFILE.md](docs/user/STORAGE_PROFILE.md)):
  - `anemone autotune --copy_hash_pairs=1` hashes the copy's source two chunks at a time, with the same digests:
    96 GiB to one target copied at 4.93 and 5.02 GB/s against 4.84 and 5.00 with one hash a chunk. `=0` is the
    earlier default.
  - The PP loader reads each copy with 4 readers (`--reader_threads_per_copy=4`, from 2): real PP5 serve loads took
    16.442 and 16.399 s against 16.854 and 17.207 s with 2, +3.58 % beyond a 2.07 % spread
    ([USAGE.md](docs/USAGE.md#knob-reader_threads_per_copy)).
  - `--load_piece_mib` stays 16 MiB on PP and TP: 64 MiB was no faster on PP5 loads, and TP4_PRO's tuned candidate
    loaded 1.54 % slower than the defaults ([KNOBS.md](tools/cli/KNOBS.md)).

### EXL3
- EXL3 widths K2, K2.5, K3 and K4, read per matrix from the pack. The 2.68 bpw map fits 2 sessions of 1M on TP4_EXL3;
  it is opt-in, with no quality claim yet.
- The EXL3 expert kernels are also built for the RTX PRO 6000.
- The converter admits a checkpoint by validation and writes a descriptor beside each map, with its provenance and a
  quality status.

### Quality and code
- **All linters clean, with no exceptions:** on a clean checkout `make analysis` (cppcheck at the exhaustive level,
  clang-tidy with every warning an error) exits 0, and the layout, literal and MUST_CHECK lints report nothing. There
  is no suppression and no list of accepted findings: the last named ones were fixed in code (the libFuzzer entry
  points are asm labels, a test counts allocations with its own allocator).
- **Lint cleanup, behaviour-neutral:** the source is rewrapped to 120 columns, the cppcheck and clang-tidy findings
  are fixed in code and padded structs are reordered. Every cleanup commit kept its objects identical, or its outputs
  identical on the three GPU splits' gated routes (PP5, TP4_PRO and TP4_EXL3; one user, 15 streams each).
- **`source/commands.c`:** the program's code moved out of `source/main.c` into `source/commands.c` and
  `source/commands.h`, a unit linked into the program and every test like any other source file. `main.c` keeps only
  `main()`, and no test includes it. This is the first slice of the source reorganisation.
- **Golden reference:**
  [anoane/anemone-ds41f1017-golden](https://huggingface.co/datasets/anoane/anemone-ds41f1017-golden)
  holds 435 cases and 66,409 positions, with the full F32 heads and a compact top-64 reference.
  `tools/golden_score.py --compact` scores NLL, perplexity, top-1 and top-k agreement and certain-token agreement
  against it.
- **Proven changes:** the lint and cleanup merges kept objects or outputs identical; every merge ran under
  sanitizers with an adversarial review; the default and feature changes are in their own sections above. The
  [release gate](#release-gate)'s identity check compares the release with the last GPU-gated build (source
  `288fa1e`) on the three GPU splits' gated routes (PP5, TP4_PRO and TP4_EXL3; one user, 15 streams each).
- [CONTRIBUTING.md](CONTRIBUTING.md) sets the bar for pull requests and issues.

### Limits
- Alpha software. The exact profile stays bounded to one session of 512 tokens. Vision is experimental. The comparison
  tables in the README keep their 6 November measurements, within the v0.0.2-alpha [measured scope](#measured-scope).
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` is required: without it the RTX PRO 6000 can be enumerated first and the PP5 startup
  fit fails with a message that does not name the cause ([INSTALL.md](INSTALL.md#card-placement)). The engine does
  not yet order the cards by itself.
- `make host_tests` needs `HOME` set: under `env -i` without it, a case of `replica_prepare_test` fails, so
  [BUILD.md](BUILD.md)'s command gives it an empty folder.

### Release gate
The release executable ([VALIDATION.md](docs/VALIDATION.md)) was gated on the reference machine before the tag.
- **Identity:** on PP5, TP4_PRO and TP4_EXL3, one user and 15 streams a route, its outputs matched the last gated build
  before the release commit (source `288fa1e`): PP5 15/15, TP4_PRO 15/15, TP4_EXL3 15/15; and the earlier production
  build (source `84858e7`): PP5 15/15, TP4_PRO 15/15, TP4_EXL3 15/15.
- **1M sessions:** on a PP5 serve with `--session_slots=max`, `--context_tokens=1048576`, `--generated_tokens=1048576`,
  `--temperature=1` and `--top_p=1` (the requests at T=0), the startup fit admitted 5 session slots of 1,048,576 tokens
  each. Served: one fresh 1M-token session, HTTP 200, the needle found, prefill 3087 tok/s (1,022,969 prompt tokens in
  331.3 s), decode 12.9 ms/token; four at once, HTTP 200 for all 8 requests, all 4 needles found, prefill about 766
  tok/s each (1333.5 to 1336.9 s), decode 238.1 tok/s in all (4.20 ms/token).
- **Fraser:** the same serve with `--context_tokens=524288` and `--generated_tokens=524288`, T=0, thinking off, 2,048
  tokens, 1 user, 10 repetitions: median 173.2 tok/s (5.78 ms/token), best 179.4 tok/s (5.57 ms/token).

## v0.0.2-alpha

Anemone provides a C11/CUDA DeepSeek-V4.1-Flash-1017 engine targeting four CMP
170HX cards for EXL3, and four CMP 170HX cards plus an RTX PRO 6000 Blackwell
for original weights. See [README](README.md), [BUILD](BUILD.md),
[INSTALL](INSTALL.md), [model preparation](docs/user/MODEL_DATA.md) and
[the complete server recipes](docs/SERVER_PROFILES.md).

This public layout keeps the full comparison matrix, concise benchmark methods,
source-review notes and curated machine-readable results. Internal handoffs,
development diaries, raw workstation receipts and duplicate historical documents
are archived separately. Complete build/generator source is included; executable
Python tools use explicit `.py` filenames, and build outputs are not shipped.
This snapshot contains no Git metadata or history.

Licensing metadata, version/startup attribution and five corrected help entries
preserve the existing inference defaults and CUDA payloads. All serve options
are documented. The Pi template uses the requested model display name and endpoint/
key placeholders, with private user-list creation and Linux getrandom guidance.
Required author attribution and the original license terms remain intact.

### Measured scope

The [full matrix](docs/results/COMPARISON_MATRIX.md) retains all dated comparisons
and source/binary identities. Current non-golden coverage is 42 attempts from an
86-job catalogue: 40 passes, one TP4 mixed identity failure and one unqualified
capacity cleanup attempt, with 44 jobs unattempted. Both failed/unqualified rates
are withheld. The separate prompt-chunk guard and PP5 mixed tests pass 96/96.

Six Curtis jobs, four functional snapshot jobs, two actual 100K-class pilots and
two Fraser reasoning jobs pass their bounded protocols. The 100K prompts contain
99,972 native input tokens plus one output, with two fresh repetitions per
balanced format. Q timings with CPU overlap are not performance results.
Fraser's short diagnostics and one-token restoration do not establish full
quality or logical-state equivalence. Exact is one session and 512 total tokens;
original balanced output depends on native PRO layer placement.

Current full golden replay remains 0/435 cases per balanced format, and actual
Pi remains unrun. Historical 54-case conditional-head scores keep their date
and denominator. Needles, soak, seeded isolation, other unexecuted protocols,
fixture gates and unsupported capabilities remain explicit. Source reviews are
not GPU tests. No extra HumanEval/GSM8K acquisition or evaluation is included.

The public-package companion manifest records actual source-only build checks,
archive identities and executable hashes; [VALIDATION.md](docs/VALIDATION.md)
explains their scope. No inference optimization or new GPU benchmark is implied
by this repository cleanup.
