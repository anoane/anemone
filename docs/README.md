<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Documentation

| Guide | Contents |
| --- | --- |
| [Build](../BUILD.md) | Linux toolchain, source-only build and native checks. |
| [Install](../INSTALL.md) | Prepared models, deployment, authentication and HTTP requests. |
| [Server profiles](SERVER_PROFILES.md) | Original exact, original balanced and EXL3 balanced; all 169 serve option spellings. |
| [User manual](USAGE.md) | Commands, HTTP API, sampling, sessions, advanced knobs and client templates. |
| [Model data](user/MODEL_DATA.md) | Hugging Face revisions, input manifests, tokenizer, pack and EXL3-map preparation. |
| [Full comparison matrix](results/COMPARISON_MATRIX.md) | All component/benchmark/profile rows and dated comparisons. |
| [Benchmark methods](results/BENCHMARKS.md) | Rates, sample counts, qualification limits and implementation citations. |
| [Golden metrics](results/GOLDEN_METRICS.md) | Current unmeasured scope and clearly dated measured quality. |
| [Development](DEVELOPMENT.md) | Source layout, generators and optional checks. |
| [Credits](CREDITS.md) | Community acknowledgements, reference projects and PR contributors. |
| [Style](STYLE.md) | C/CUDA source conventions. |
| [Validation](VALIDATION.md) | Build proof and the limits of recorded tests. |

The public tree contains current guides and curated numerical results. Private
operator handoffs, development diaries, process logs and duplicate historical
Markdown are archived separately. Measured comparisons keep their dates,
source/binary identities and limitations; removing notebooks does not turn
missing or failed tests into passes.
