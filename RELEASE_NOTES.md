<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# v0.0.2-alpha release notes

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

## Measured scope

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
