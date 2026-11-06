# Anemone v0.0.2-alpha

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
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin make host_tests
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin make -j32 header_check
```

The recorded toolchain is CUDA 13.4 V13.4.92, GCC 15.2.0 and GNU Make 4.4.1.
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

The profile guide explains all 169 serve option spellings, resolved defaults,
request controls and unsupported combinations. The [user manual](docs/USAGE.md)
covers the HTTP API, other commands and a [Pi configuration template](docs/USAGE.md#67-pi-configuration)
with private user/key-list generation. Its placeholders contain no real endpoint
or API key.

## Measured performance and limits

Retained **6 November 2026, tcb82** measurements. Donor decode permits prefix reuse;
conditional-head quality covers 54 cases and 7,878 positions, with exact quality
reported as reference self-comparison. Current reruns,
protocols and qualification limits are in the [full matrix](docs/results/COMPARISON_MATRIX.md),
[benchmark methods](docs/results/BENCHMARKS.md) and [golden metrics](docs/results/GOLDEN_METRICS.md).

| 6 November 2026, tcb82 measurement | TP4 EXL3 balanced | Original PP5 balanced | Original exact |
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

## Repository guide

- [Documentation index](docs/README.md): public guides and results.
- [Development](docs/DEVELOPMENT.md): checks, generators and source layout.
- [Tools](tools/README.md): source programs, dependencies and examples.
- [Release notes](RELEASE_NOTES.md): version scope and known limits.
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
