# Anemone v0.0.4-alpha

SPDX-License-Identifier: AGPL-3.0-or-later

Anemone is a native C11/CUDA inference engine for DeepSeek-V4.1-Flash-1017.
Copyright (C) 2026 Fabiano Anemone <fabiano.anemone@gmail.com>.

**Alpha software:** This project is experimental and probably contains many bugs.
It is developed with careful coding practices, sanitizers in debug/test builds,
and many test cases.

Features include **experimental vision support** for PNG/JPEG image inputs and an
included [pi.dev configuration template](docs/USAGE.md#67-pi-configuration).
Vision execution and quality qualification remain incomplete; the original exact
recipe is text-only. See the [user manual](docs/USAGE.md) for input and profile limits.

## What makes Anemone different

- **Turbo load.** Verified copies of the model on every local NVMe drive are read in parallel as one weighted array,
  with O_DIRECT and io_uring, checking each chunk's digest as it is read. The engine finds the copies by their
  receipts, so there is no RAID to set up. A prepared model is serving in **30 s (EXL3 TP4) and 48 s (original PP5)**,
  against about 205 s to 10 minutes for the vLLM-based engines on the same cards
  ([A11](docs/results/COMPARISON_MATRIX.md#a-identity-provenance-packaging)).
- **Turbo Engram.** The 189 GiB of Engram n-gram tables stay in host RAM by default. When they don't fit, the engine
  keeps the largest share that does, reads the rest from NVMe and prints why. Disk tables are read a whole step at a
  time on io_uring threads for each copy, with every block verified and a stalled read retried on another copy.
  Serving PP5 from disk, a 131K-token cold prefill took **27.4 s against 27.2 s from RAM**. Decode was 2.2 % slower
  at one session and equal at four and eight ([ENGRAM_PLACEMENT.md](docs/user/ENGRAM_PLACEMENT.md)).
- **Mining cards plus one workstation card.** The original FP8/FP4 weights run as a five-stage pipeline over four CMP
  170HX and one RTX PRO 6000 Blackwell, with native Blackwell arithmetic on the PRO. In the comparison below it has
  the fastest strict cold prefill (**4,436 tok/s**) and the highest Fraser aggregates at four and eight users. EXL3
  runs on the four CMP 170HX alone.
- **1M tokens for every session.** Each session gets the full 1,048,576-token context by default, and the startup
  fit serves as many sessions as the cards hold ([below](#sessions-and-context)).
- **Exactness by choice.** The exact profile reproduces the pinned original serial reference: over 263 positions all
  34,000,640 full-vocabulary F32 logits match (XR57, the exact recipe's qualification,
  [original_reference_serial_m1](tools/cli/AXES.md#original_reference_serial_m1)). It is the reference the golden
  scores compare against: on the 6 November subset of 7,878 positions the balanced profiles reach 98.21 % (PP5) and
  96.81 % (EXL3) top-1 agreement with it ([GOLDEN_METRICS.md](docs/results/GOLDEN_METRICS.md)). Four profile axes
  (category, exactness, concurrency, focus) pick the rest.
- **Lossless, compact state.** FP16 state carriers at about 3,200 bytes per token per card: the smallest lossless
  footprint in the comparison ([H7](docs/results/COMPARISON_MATRIX.md#h-kv-cache-and-state-memory)).
- **Storage autotune.** `anemone autotune` prepares and verifies the copies and tunes the load and the Engram step
  I/O for each machine, in separate stages with their own parameters: the copy stage records its decisions beside the
  profile, the load stage writes the profile's load section, and the Engram stage, replaying real prompts, its engram
  section ([STORAGE_PROFILE.md](docs/user/STORAGE_PROFILE.md)).
- **Proven changes.** Every merge has CPU proof on the reference machine:
  - address, thread and undefined-behaviour sanitizers;
  - an adversarial review of the code;
  - object or dry-run identity outside its intended changes.
  GPU output identity is checked at the [release gate](RELEASE_NOTES.md#release-gate), on PP5, TP4_PRO and TP4_EXL3.
- **Small and self-contained.** C11 host code and CUDA kernels, linking only libc, libm and the CUDA driver. Formats,
  hashing, asynchronous I/O and the HTTP server are all implemented in the tree.

## Sessions and context

Every session gets the full 1M context (`--context_tokens=1048576`, the default). The number of sessions defaults
to `--session_slots=max`: at startup the engine fits as many full-context sessions as the cards hold, and prints
`session slots: max -> N of 1048576 tokens each`. A context is never shrunk automatically. If not even one session
fits, the start is refused.

| Profile and placement | Sessions at 1M (default) |
| --- | ---: |
| Original weights, PP5 (4 × CMP 170HX + RTX PRO 6000), the default | 6 |
| Original weights, `--placement=tp4_pro` | 1 |
| EXL3 balanced, TP4, 2.9 bpw map (default) | 1 |
| EXL3, TP4, 2.68 bpw map (opt-in, no quality claim yet) | 2 |
| Original exact | 1 session of 512 tokens (bounded recipe) |

Measured with the startup fit of a plain default serve of each placement on the reference machine, with no other
engine on the cards: the placement, the tables, the pack or map and its verified copies, and no other option. PP5 was
measured on 17 Nov 2026 (v0.0.4-alpha), the others on 16 Nov 2026 (v0.0.3-alpha), whose plans v0.0.4-alpha
leaves as they were. TP4_PRO's start prints `session slots: max -> 1 of 1048576 tokens each`. For PP5 that is
`serve --placement=pp5` with the tables and the pack only: the default resolver applies the PP5 base recipe itself, and
the start prints `session slots: max -> 6 of 1048576 tokens each` (5 on v0.0.3-alpha). On 16 Nov 2026 it was
ready in about 24 s, loading 195 GiB of host
memory at 16.6 GiB/s from three verified copies. That start read each copy with 2 PP loader readers; the default is now
4, which loaded real PP5 serves in 16.42 s against 17.03 s with 2
([`--reader_threads_per_copy`](docs/USAGE.md#knob-reader_threads_per_copy)). The benchmark recipes in
[SERVER_PROFILES.md](docs/SERVER_PROFILES.md) pin 65,536 tokens × 8 sessions for the comparison runs: drop those two
options to serve 1M sessions. More memory per card, or a smaller context, fits more sessions.

To trade context for sessions by hand:
- `--context_tokens=N`: the context of every session, a multiple of 16, at least 16,384 on TP placements. For
  example, the original balanced PP5 recipe fits 30 sessions at `--context_tokens=65536` (28 on v0.0.3-alpha).
- `--session_slots=N`: a fixed count. A count the cards cannot hold is refused before the load, with the most that
  fit.

See [SERVER_PROFILES.md](docs/SERVER_PROFILES.md#memory-slots-context-and-1m) for the memory options.

## First-time setup: replicas and autotune

Run this once per machine, and again after adding or replacing a drive.
[STORAGE_PROFILE.md](docs/user/STORAGE_PROFILE.md) has every option.

1. **Prepare the model packs** ([MODEL_DATA.md](docs/user/MODEL_DATA.md)) and build ([BUILD.md](BUILD.md)).
2. **Copy the pack to your other NVMe drives.** The copy stage finds the local drives, copies the pack and verifies
   every chunk. It writes the receipts that serving uses to find the copies:
   ```sh
   anemone autotune /srv/model.anemone --stage=copy --profile=/srv/runtime/storage-profile.json
   ```
   Copies live in `<mount>/anemone_engram_rnd/replicas/`, owned by the user that runs the engine. On a mount whose
   root that user cannot write, create `<mount>/anemone_engram_rnd` once and give it to that user (`sudo mkdir`,
   `sudo chown`). Give `--target=<mount>/anemone_engram_rnd/replicas/<name>.anemone` to choose drives yourself. A run
   that stops resumes when you run the same command again.
3. **Tune the load (optional).** The load stage measures the TP host load on these copies and writes a candidate
   section into the profile. `--qualify` enables it:
   ```sh
   anemone autotune /srv/model.anemone --stage=load --profile=/srv/runtime/storage-profile.json
   anemone autotune /srv/model.anemone --stage=load --profile=/srv/runtime/storage-profile.json --qualify
   ```
4. **Tune the Engram step I/O (optional).** It matters only when the Engram fit puts tables on disk. The Engram
   stage replays real prompts through serving's step I/O on these copies and writes a candidate section into the
   profile; `--qualify` enables it. Make its token files from the prompts of the public golden dataset
   ([anoane/anemone-ds41f1017-golden](https://huggingface.co/datasets/anoane/anemone-ds41f1017-golden)), one
   messages file a prompt, with `tools/measure/engram_trace` (`make -f tools/measure/engram_trace.mk`):
   ```sh
   build/release/engram_trace tokenize tokenizer.bin prompt1.json /srv/prompt1_ids.u32
   anemone autotune /srv/model.anemone --stage=engram --tokens=/srv/prompt1_ids.u32 --tokens=/srv/prompt2_ids.u32 --profile=/srv/runtime/storage-profile.json
   anemone autotune /srv/model.anemone --stage=engram --profile=/srv/runtime/storage-profile.json --qualify
   ```
   `anemone autotune /srv/model.anemone --tokens=... --profile=...` without `--stage` runs steps 2, 3 and 4 in order
   and skips a stage whose section already names this pack and these copies (`--retune` measures it again).
5. **Serve** with a command from [SERVER_PROFILES.md](docs/SERVER_PROFILES.md). Turbo load opens the verified copies
   by itself: the startup lines `turbo load: copy N <path>` name them. The Engram fit prints where the tables go
   (`engram ram percent: …`). The engine numbers the visible cards by PCI address itself, whatever the order of a
   `CUDA_VISIBLE_DEVICES` list (`--card_order` orders them); every command still sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
   so that a `CUDA_VISIBLE_DEVICES` list of ordinals selects the same cards ([INSTALL.md](INSTALL.md#card-placement)).

## Target hardware

| Weight format | Target GPU configuration |
| --- | --- |
| EXL3 | **4 × NVIDIA CMP 170HX** |
| Original FP8/FP4 weights | **4 × NVIDIA CMP 170HX + 1 × NVIDIA RTX PRO 6000 Blackwell** |

The reference build targets Linux x86-64 and an AMD Zen 4 CPU. Both weight
families also use the original pack for shared tensors and Engram; runtime
Engram alone needs about 188.8 GiB of host RAM before other allocations.
Original balanced inference uses native PRO arithmetic, so results depend on
which layers the PRO owns. Hardware targets do not imply that every context,
concurrency or exactness combination is qualified.

## Build

From the repository root, with the CUDA toolkit and driver development library
installed:

```sh
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin make -j32 release
env -i HOME="$(mktemp -d)" PATH=/usr/local/cuda/bin:/usr/bin:/bin make host_tests
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin make -j32 header_check
```

The recorded toolchain is CUDA 13.5 V13.5.37, GCC 14.2.0 and GNU Make 4.4.1.
Release compilation and inference use C/CUDA; Python is an optional dependency
for model preparation, code generation and analysis tools. No model weights
are required to compile or run native host tests. See [BUILD.md](BUILD.md).

## Prepare models and serve

The current recipes use these model sources:

| Weight source | Hugging Face repository | Pinned revision |
| --- | --- | --- |
| EXL3 routed experts, 2.9 bpw | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | [0be63c3337](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/tree/0be63c3337efbb2a0061d5b8bb4006fa186d433f) |
| Original weights and shared tensors | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | [1bae0e1088](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) |

EXL3 still requires the original pack for shared tensors, vision, DSpark and Engram.

[MODEL_DATA.md](docs/user/MODEL_DATA.md) gives the public Hugging Face repositories,
exact revisions, source manifests and tokenizer/pack/EXL3-map commands.
[INSTALL.md](INSTALL.md) explains installation, authentication and HTTP use.

Choose a complete launch command from [SERVER_PROFILES.md](docs/SERVER_PROFILES.md):

| Profile | Scope |
| --- | --- |
| EXL3 balanced | Four-CMP TP4; separate quantized weight family. The default mixed-output identity gate has a recorded failure. |
| Original balanced | Original weights over PP5 with native PRO arithmetic. |
| Original exact | Bounded original PP THIN serial M1: one session, 512 total tokens. |

The profile guide explains the 169 serve option spellings of the 6 November parser
and `--metrics`, marking the rows with current values, with resolved defaults,
request controls and unsupported combinations; later options are in the user manual
and [the knob catalogue](tools/cli/KNOBS.md). The [user manual](docs/USAGE.md)
covers the HTTP API, other commands and a [Pi configuration template](docs/USAGE.md#67-pi-configuration)
with private user/key-list generation. Its placeholders contain no real endpoint
or API key.

## Measured performance and limits

Retained **6 November 2026** measurements. Donor decode permits prefix reuse;
conditional-head quality covers 54 cases and 7,878 positions, with exact quality
reported as reference self-comparison. Current reruns,
protocols and qualification limits are in the [full matrix](docs/results/COMPARISON_MATRIX.md),
[benchmark methods](docs/results/BENCHMARKS.md) and [golden metrics](docs/results/GOLDEN_METRICS.md).

| 6 November 2026 measurement | TP4 EXL3 balanced | Original PP5 balanced | Original exact |
| --- | ---: | ---: | ---: |
| C1 structured donor decode (tok/s) | 337.2 | 248.7 | 7.8 |
| C8 structured aggregate donor decode (tok/s) | 920.8 | 810.2 | outside one-session, capacity 512 scope |
| Strict cold prefill, nine cached=0 requests (tok/s) | 3360.0 | 4420.0 | outside one-session, capacity 512 scope |
| Paired conditional-head top1 agreement | 7627/7878 (96.81%) | 7737/7878 (98.21%) | 7878/7878 (100.00%) |
| Paired conditional-head perplexity | 1.121 | 1.096 | 1.083 |

## Engine comparison

A compact view of the configurations recorded in the [full matrix](docs/results/COMPARISON_MATRIX.md).
Original PP5 uses native PRO arithmetic; outputs depend on assigned layers.

| Property | Anemone EXL3 balanced | Anemone original PP5 weights balanced | Curtis | Morrow | GNmac |
| --- | --- | --- | --- | --- | --- |
| Runtime | C11/CUDA | C11/CUDA | vLLM | vLLM | vLLM |
| GPUs | 4 × CMP 170HX | 4 × CMP 170HX + RTX PRO 6000 Blackwell | 4 × CMP 170HX | 4 × CMP 170HX | 4 × CMP 170HX |
| Placement | TP4 | Layer PP5 | TP4 | TP4 | TP4; PP4 option |
| Routed weights | EXL3 hybrid | Original FP8/FP4 | EXL3 hybrid | EXL3 hybrid | EXL3 hybrid |
| State/KV storage | FP16 carriers | FP16 carriers | BF16 | BF16 | FP8 + BF16 RoPE |
| Drafting | DSpark | DSpark | DSpark | DSpark | DSpark |

## Benchmark comparison

[C1/C8 donor decode](docs/results/COMPARISON_MATRIX.md#v1-11-donor-decode-gnmacs-pinned-client-180-w-aggregate-toks-c1-per-stream)
uses Anemone's **6 November, `6b9e276/spdx_02`** epoch, with prefix reuse allowed.
[Cold prefill](docs/results/COMPARISON_MATRIX.md#v2-prefill) uses current Anemone
**`1b073/spdx_04`** results.
Peer values retain their recorded dates and recipes; Curtis uses the speed lane.
Power settings and sample counts are recorded in the full matrix. These are point estimates.

Anemone's [startup times (A11)](docs/results/COMPARISON_MATRIX.md#a-identity-provenance-packaging)
run from a fresh process to serving readiness, with model files and optional NVMe
replicas already prepared. Anemone excludes the one-time NVMe replica creation/copy;
OS/storage caches were not cleared. Peer measurements retain their process/container
boundaries and startup gates. [Fraser C1 TTFT](docs/results/COMPARISON_MATRIX.md#v3-fraser-27k-prompt-thinking-off-2048-token-cap-t0)
measures time to first token for the 2.7K-token prompt.

| Metric | Anemone EXL3 balanced | Anemone original PP5 weights balanced | Curtis | Morrow | GNmac |
| --- | ---: | ---: | ---: | ---: | ---: |
| Donor C1 structured (tok/s) | 335.8 | 248.4 | 291.4 | 312.5 | 355.8 |
| Donor C1 code (tok/s) | 265.5 | 175.7 | 212.3 | 245.5 | 259.8 |
| Donor C1 prose (tok/s) | 170.7 | 121.4 | 150.9 | 164.2 | 193.6 |
| Donor C8 structured, aggregate (tok/s) | 915.8 | 807.5 | 806.8 | 779.8 | 886.3 |
| Donor C8 code, aggregate (tok/s) | 625.0 | 504.2 | 475.0 | 436.0 | 609.3 |
| Donor C8 prose, aggregate (tok/s) | 496.3 | 394.0 | 348.6 | 433.7 | 448.4 |
| Strict cold prefill, nine requests (tok/s) | 3383.5 | 4436.4 | 1746.0 | 3807.0 | 2343.0 |
| Fraser C1 TTFT (ms) | 1533.6 | 1281.8 | 1518.0 | 857.0 | 1118.0 |
| Prepared-model start to serving (s; Anemone replica copy excluded) | 30.1 | 48.4 | 499.0 | ~210–420 | ~205 |

## Fraser comparison

[Fraser protocol](docs/results/COMPARISON_MATRIX.md#v3-fraser-27k-prompt-thinking-off-2048-token-cap-t0):
2.7K-token prompt, thinking off, 2,048-token output cap and T=0. Anemone uses
**6 November, `1b073/spdx_04`** results; peer measurements retain their recorded dates
and recipes. Curtis uses the speed lane. C1, C4 and C8 mean one, four and eight concurrent users.
Values are medians; end-to-end aggregate rates include time to first token (TTFT).
Original PP5 uses native PRO arithmetic, so outputs depend on assigned layers.

| Metric | Anemone EXL3 balanced | Anemone original PP5 weights balanced | Curtis | Morrow | GNmac |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 per-stream decode (tok/s) | 244.1 | 176.4 | 214.1 | 231.8 | 253.8 |
| C1 end-to-end aggregate (tok/s) | 205.2 | 158.2 | 183.9 | 210.2 | 221.6 |
| C1 TTFT (ms) | 1533.6 | 1281.8 | 1518.0 | 857.0 | 1118.0 |
| C4 per-stream decode (tok/s) | 107.2 | 138.8 | 74.2 | 94.6 | 103.0 |
| C4 end-to-end aggregate (tok/s) | 320.8 | 412.2 | 258.6 | 316.2 | 339.8 |
| C4 TTFT (ms) | 6077.2 | 4321.1 | 4610.0 | 3547.0 | 3450.0 |
| C8 per-stream decode (tok/s) | 68.5 | 72.8 | 55.5 | 55.2 | 63.1 |
| C8 end-to-end aggregate (tok/s) | 385.3 | 440.1 | 361.9 | 361.9 | 421.1 |
| C8 TTFT (ms) | 12031.6 | 7352.9 | 6993.0 | 6825.0 | 5081.0 |

## R&D and contributions

A large R&D pipeline remains to improve kernels, loading, memory use, parallelism,
scheduling, long-context handling, vision and numerical qualification. Help, reviews,
feedback, bug and issue reports, and pull requests are all appreciated.
See the [full matrix](docs/results/COMPARISON_MATRIX.md) for remaining work and
[development guide](docs/DEVELOPMENT.md) for contributing and validation commands.
[CONTRIBUTING.md](CONTRIBUTING.md) sets the bar for pull requests and issues: one change at a time, only the code
that is needed, every linter clean, long testing on all four builds, and evidence for every claim.

## Repository guide

- [Documentation index](docs/README.md): public guides and results.
- [Development](docs/DEVELOPMENT.md): checks, generators and source layout.
- [Tools](tools/README.md): source programs, dependencies and examples.
- [Release notes](RELEASE_NOTES.md): version scope and known limits.
- [Contributing](CONTRIBUTING.md): the standards for pull requests, issues and requests.
- [Validation](docs/VALIDATION.md): build verification and evidence boundaries.

The tree contains source, required build inputs and curated public results.
Development notebooks, operator handoffs and raw workstation logs are retained
outside the public repository. Generated build outputs belong in `build/`.

## Credits

Special thanks to:

- **Curtis** ([@BBacktesting](https://x.com/BBacktesting)) — [CurtisAccelerate](https://github.com/CurtisAccelerate).
- **Morrow** ([@Morrowmake](https://x.com/Morrowmake)) — [CMP 170HX inference work](https://github.com/Morrowmake/vllm-cmp170hx).
- **GNmac** ([@GNmac](https://x.com/GNmac)) — [CMP 170HX project](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx).
- **Sean Phan** ([@seanphan](https://x.com/seanphan)) — [PixelML/club-170hx](https://github.com/PixelML/club-170hx).
- **Turboderp** ([@turboderp_](https://x.com/turboderp_)) — [ExLlamaV3](https://github.com/turboderp-org/exllamav3).
- **Petri Krohn** ([@PetriKrohn](https://github.com/PetriKrohn)) — [CMP 170HX unlock research](https://github.com/dartraiden/NVIDIA-patcher/issues/73) and the [FACEB13D talk](https://talks.mrmcd.net/2026/talk/ZDRJLV/), with Jon Pry, Cab, Humvee55, BlackSun, duggasco, Amogh Munikote and the others named in the [full credits](docs/CREDITS.md).
- Members of the private **CMP 170HX Owners** group chat on X.
- The [**cmpunlocker Discord channel**](https://discord.gg/CdHSakKSFv).

Thanks also to the authors, maintainers, contributors and PR reviewers behind the
engines, kernels, quantization tools, recipes and GPU tooling reviewed during Anemone's
development, including DeepSeek, Turboderp/ExLlamaV3, vLLM, SGLang, llama.cpp,
Morrowmake, CurtisAccelerate, GNmac-lab, PixelML, FlashInfer, DeepGEMM, FlashMLA,
TensorFold, Strata, FreeToken, EXLR8 and the CMP 170HX community.
The [full credits](docs/CREDITS.md) link the reference projects, pinned source revisions,
consulted PRs and their contributors. Research references and inspiration are credited
alongside benchmark baselines; their individual licenses and notices remain with their projects.

## License and attribution

Licensed under [AGPL-3.0-or-later](LICENSE.md), including its existing section 7(b)
author-attribution term. Preserve Fabiano Anemone <fabiano.anemone@gmail.com> in
LICENSE, README and the program's version/startup notice. Source headers and
`REUSE.toml` describe file licensing. Modified network-serving versions must
provide their corresponding source as required by the existing license.
