<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Validation and evidence boundaries

The reference platform is Linux x86-64, CUDA 13.5 V13.5.37, GCC 14.2.0 and
GNU Make 4.4.1. The engine targets the GPU configurations in [README](../README.md).
The v0.0.3-alpha release executable, built from source
`f57d81375a8830db16925e42740aae1fe0773e90`, has SHA256:

```text
2afdfc7138b74baa94f9bfe9f426d188e6af1be8e6729230e6c4f21f070d317b
```

A clean build of a `git archive` export of that source runs the [BUILD.md](../BUILD.md)
commands with an empty environment (`HOME` an empty folder) and no visible GPU. It
compiles all 119 CUDA modules of the release kernel set, whose 119 assembler reports
hold no warning, runs 157 native host programs and 675 header compilations, and its
executable has the SHA256 above. Source hashes are checked before and after. Host
tests use loopback fixtures; no model is loaded.

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

v0.0.3-alpha is published as one source snapshot commit, tagged `v0.0.3-alpha`,
on top of the v0.0.2-alpha snapshot. Its tree is release source
`f57d81375a8830db16925e42740aae1fe0773e90` with this record and the other
documentation, the licence texts, `REUSE.toml`, one metrics data file, six receipt
paths of the axis route baseline and one entry of the layout lint changed; the
BUILD.md commands read none of these files, so a build of the snapshot has the same
inputs. For v0.0.2-alpha, the accompanying public-package manifest recorded the
exact publication commit, archive/file hashes and actual verification result, with
curated proof counts rather than workstation paths, process identities or private
logs.

[Benchmark methods](results/BENCHMARKS.md), the [full matrix](results/COMPARISON_MATRIX.md)
and [result data](results/RESULTS_6NOV.json) distinguish source findings, parser
checks, allocation readiness, functional tests, numerical comparison and
performance. Failed identity and unqualified cleanup rates are withheld. Q
snapshot timings with CPU overlap are not published as performance results.
Current full golden replay and Pi runs remain unrun. The historical measured
quality subset retains its own scope in [GOLDEN_METRICS.md](results/GOLDEN_METRICS.md).

All shipped files have inline SPDX identifiers or `REUSE.toml` coverage, including
strict-format data and deliberately malformed test fixtures. The original
license and required author attribution are preserved. The published repository
holds the source snapshots only; development history, notebooks and operator logs
stay outside it. Source revision identifiers (such as `f57d813`, `288fa1e`,
`84858e7`, `ab815b6` and `6b9e276`) are reproducibility references to that
development history and are not in the published repository.
