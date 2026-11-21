<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Validation and evidence boundaries

The reference platform is Linux x86-64, CUDA 13.5 V13.5.37, GCC 14.2.0 and GNU Make 4.4.1. The engine targets the GPU
configurations in [README](../README.md). The v0.0.6-alpha release executable, built from
source `5047a717778210bd09277902b8d2bff895572d9b`, has SHA256:

```text
ec209bdc264b611fafa3e9c7622260e7a1ceee7ab21dc12a83646aed809d3d6a
```

A clean build of a `git archive` export of that source runs the [BUILD.md](../BUILD.md)
commands with an empty environment (`HOME` an empty folder) and no visible GPU. It
compiles all 119 CUDA modules of the release kernel set, whose 119 assembler reports
hold no warning, runs 165 native host programs and 681 header compilations, and its
executable has the SHA256 above. Source hashes are checked before and after. Host
tests use loopback fixtures; no model is loaded.

The v0.0.5-alpha record is kept as history. Its executable, built from source
`d208e0150c639f4412e0b508618a073ccef29383` with CUDA 13.5 V13.5.37, GCC 14.2.0 and
GNU Make 4.4.1, has SHA256:

```text
ff47580be4694b7050964c4c57e2df1d0882be345b6b93ca3b700671784793b0
```

Its clean build compiled all 119 CUDA modules, whose 119 assembler reports held no warning,
ran 164 native host programs and 684 header compilations.

The v0.0.4-alpha record is kept as history. Its executable, built from source
`6ecf1bb3e8b5e2abc57d8de3a90d9d6b064f8584` with CUDA 13.5 V13.5.37, GCC 14.2.0 and
GNU Make 4.4.1, has SHA256:

```text
15abb30da13582cd5a9b004014654b02b2da4d508001d049a932667f55458e49
```

Its clean build compiled all 119 CUDA modules, whose 119 assembler reports held no warning,
ran 163 native host programs and 684 header compilations.

The v0.0.3-alpha record is kept as history. Its executable, built from source
`f57d81375a8830db16925e42740aae1fe0773e90` with CUDA 13.5 V13.5.37, GCC 14.2.0 and
GNU Make 4.4.1, has SHA256:

```text
2afdfc7138b74baa94f9bfe9f426d188e6af1be8e6729230e6c4f21f070d317b
```

Its clean build compiled all 119 CUDA modules, whose 119 assembler reports held no warning,
ran 157 native host programs and 675 header compilations.

The v0.0.2-alpha record is kept as history. Its executable, built with CUDA 13.4
V13.4.92, GCC 15.2.0 and GNU Make 4.4.1, has SHA256:

```text
acdf429da22ae7a39504b897106e8cd56a22c3661bf086f60b5458ba5313f2af
```

Its clean source verification compiled all 117 CUDA modules, checked all 117 reports
for warnings, ran 125 native host programs and 594 header compilations, audited
release instrumentation, and compared module and executable hashes. Source
hashes were checked before and after. These checks used an isolated Linux tree,
empty HOME, private networking and no GPU device nodes. Host tests used private
loopback fixtures; no model was loaded. The separate optional instrumentation
audits use the source programs in `tools/check/`.

v0.0.6-alpha is published as one source snapshot commit, tagged `v0.0.6-alpha`,
on top of the v0.0.5-alpha snapshot. Its tree is release source
`5047a717778210bd09277902b8d2bff895572d9b` with this record and the other documentation (the
release notes, BUILD, INSTALL, USAGE and the server-profile page) and the parser-check data
`docs/models/deepseek_v41_flash/SERVER_PROFILE_COMMANDS.json` changed. The BUILD.md
commands read none of these files, so a build of the snapshot has the same inputs, and its
executable is the release executable above.

v0.0.5-alpha was published as one source snapshot commit, tagged `v0.0.5-alpha`,
on top of the v0.0.4-alpha snapshot. Its tree is release source
`d208e0150c639f4412e0b508618a073ccef29383` with its record and the other documentation (the
release notes, the golden metrics page, README, BUILD, INSTALL, USAGE and the server-profile
page) and the parser-check data `docs/models/deepseek_v41_flash/SERVER_PROFILE_COMMANDS.json` changed. The BUILD.md
commands read none of these files, so a build of the snapshot had the same inputs, and its
executable was v0.0.5-alpha's above.

v0.0.4-alpha was published as one source snapshot commit, tagged `v0.0.4-alpha`,
on top of the v0.0.3-alpha snapshot, as v0.0.3-alpha was on v0.0.2-alpha's. Its tree is
release source `6ecf1bb3e8b5e2abc57d8de3a90d9d6b064f8584` with its record, the other
documentation, the parser-check data `docs/models/deepseek_v41_flash/SERVER_PROFILE_COMMANDS.json`, `REUSE.toml`,
one licence text's file name (`LICENSES/GPL-2.0.txt` becomes `LICENSES/GPL-2.0-only.txt`,
the same text) and `tools/models/deepseek_v41_flash/model_data.sh` (the descriptor it writes beside the EXL3 map
records the quality `established`) changed. The BUILD.md commands read none of these
files: `tools/models/deepseek_v41_flash/model_data.sh` prepares model data and is no build input. A build of the
snapshot had the same inputs, and its executable was v0.0.4-alpha's above. The
v0.0.3-alpha snapshot was release source `f57d813` with its record, the other
documentation, the parser-check data `docs/models/deepseek_v41_flash/SERVER_PROFILE_COMMANDS.json`, two licence
texts (`LICENSES/Apache-2.0.txt` and `LICENSES/GPL-2.0.txt`, added), `REUSE.toml`, one
metrics data file (`tools/metrics/vllm_dashboard_queries.json`), six receipt paths of
the axis route baseline (`tools/models/deepseek_v41_flash/cli/axes_route_baseline.json`) and one entry of the
layout lint (`tools/check/layout_lint.py`) changed.
For v0.0.2-alpha, the accompanying public-package manifest recorded the
exact publication commit, archive/file hashes and actual verification result, with
curated proof counts rather than workstation paths, process identities or private
logs.

[Benchmark methods](models/deepseek_v41_flash/results/BENCHMARKS.md), the [full matrix](models/deepseek_v41_flash/results/COMPARISON_MATRIX.md)
and [result data](models/deepseek_v41_flash/results/RESULTS_6NOV.json) distinguish source findings, parser
checks, allocation readiness, functional tests, numerical comparison and
performance. Failed identity and unqualified cleanup rates are withheld. Q
snapshot timings with CPU overlap are not published as performance results.
Pi runs remain unrun. The golden scores of all 435 cases (18 Nov 2026) and the historical
measured quality subset, with its own scope, are in [GOLDEN_METRICS.md](models/deepseek_v41_flash/results/GOLDEN_METRICS.md).

Every shipped file states its copyright and licence, in its own header or through
`REUSE.toml` (strict-format data and deliberately malformed test fixtures through
`REUSE.toml` alone), and every licence text is in `LICENSES/`. `reuse lint` 5.0.2
reports the tree compliant with version 3.3 of the REUSE Specification: 1065 of
1065 files with copyright and licence information; the licences AGPL-3.0-or-later,
Apache-2.0 and GPL-2.0-only used; none missing, unused or deprecated. The two
third-party files keep their own: the Linux kernel's `.clang-format` (GPL-2.0-only,
its header verbatim) and the vLLM dashboard's queries (Apache-2.0). The original
license and required author attribution are preserved. The published repository
holds the source snapshots only; development history, notebooks and operator logs
stay outside it. Source revision identifiers (such as `6ecf1bb`, `f57d813`,
`288fa1e`, `84858e7`, `ab815b6` and `6b9e276`) are reproducibility references to that
development history and are not in the published repository.
