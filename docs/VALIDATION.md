<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Validation and evidence boundaries

The reference platform is Linux x86-64, CUDA 13.4 V13.4.92, GCC 15.2.0 and
GNU Make 4.4.1. The engine targets the GPU configurations in [README](../README.md).
The recorded v0.0.2-alpha executable has SHA256:

```text
acdf429da22ae7a39504b897106e8cd56a22c3661bf086f60b5458ba5313f2af
```

A clean source verification compiles all 117 CUDA modules, checks all 117 reports
for warnings, runs 125 native host programs and 594 header compilations, audits
release instrumentation, and compares module and executable hashes. Source
hashes are checked before and after. These checks use an isolated Linux tree,
empty HOME, private networking and no GPU device nodes. Host tests use private
loopback fixtures; no model is loaded. The separate optional instrumentation
audits use the source programs in `tools/check/`.

The accompanying public-package manifest records the exact publication commit,
archive/file hashes and actual verification result. It contains curated proof
counts rather than workstation paths, process identities or private logs. The
source documentation describes the method and does not substitute for the
manifest of a particular delivered archive.

[Benchmark methods](results/BENCHMARKS.md), the [full matrix](results/COMPARISON_MATRIX.md)
and [result data](results/RESULTS_6NOV.json) distinguish source findings, parser
checks, allocation readiness, functional tests, numerical comparison and
performance. Failed identity and unqualified cleanup rates are withheld. Q
snapshot timings with CPU overlap are not published as performance results.
Current full golden replay and Pi runs remain unrun. The historical measured
quality subset retains its own scope in [GOLDEN_METRICS.md](results/GOLDEN_METRICS.md).

All shipped files have inline SPDX identifiers or `REUSE.toml` coverage, including
strict-format data and deliberately malformed test fixtures. The original
license and required author attribution are preserved. This source snapshot contains
no Git metadata or history. Source revision identifiers remain as reproducibility
references; development notebooks, operator logs and private history stay outside
this tree.
