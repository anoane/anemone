<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Contributing to Anemone

Anemone has a high bar: every change must be correct, measured, small and proven. This page says what a contribution
must bring. [docs/STYLE.md](docs/STYLE.md) is the rulebook for the code; where the two differ, STYLE.md wins.

## Before you write code
- **Open an issue first** for anything beyond a small, obvious fix: a feature, a kernel, a placement, a new option, or
  a default change. Agree on the scope and the measurement before you spend time on it.
- **One change per pull request.** A behaviour change, a refactor and a formatting change are three pull requests.
  Small, reviewable diffs merge; large mixed ones are sent back.
- **No option without a measurement.** A new knob needs a measured reason, an entry in the knob catalogue and its
  profile-axis mapping ([SERVER_PROFILES.md](docs/SERVER_PROFILES.md)).

## The code
- **Less code.** Write only the code that is needed, in the fewest moving parts. No speculative generality, no
  duplication, no abstraction before it has two users. A pull request that deletes code is welcome. There is no line
  budget: a function is as long as its logic needs, and no longer.
- **Meaningful names.** `lower_case_underscore`, in full words (`expert_index`, not `idx`). A name says what the thing
  is.
- **No bare numbers.** Only 0 and 1 appear in code; every other number is a named `#define`, an enumerator or a
  `static const` table.
- **C11 host, CUDA kernels in a C subset.** Formatted with the repository's `.clang-format` (the Linux style), at
  most 120 columns.
- **Every input checked, every failure returned.** A function that can fail returns `MUST_CHECK result_status`
  (`include/anemone/status.h`). Bad input is a status, never a crash. External input (sockets, requests, files,
  packs, environment) is hostile: validated against named limits, with every size computation checked for overflow.
- **No allocation on the token path.** Every wait has a deadline, and every failure path is reachable by a test
  through a `FAULT_POINT(name)` (`include/anemone/hooks.h`).
- **No escape hatches.** `NOLINT`, `cppcheck-suppress`, diagnostic pragmas, `fast-math` and the rest of STYLE.md's
  list are forbidden text, comments included. Fix the finding instead.
- **No dependencies.** No new linked library and no vendored source. The engine links only libc, libm and the CUDA
  driver. Implement in-tree, and cite the published design you followed. Ideas only: no copied code.
- **Results never depend on a flag:** the host is built with `-ffp-contract=off` and kernels with `--fmad=false`.

## Quality passes before you open the pull request
1. **Correctness:** it works, with tests for the new behaviour and for each of its refusal paths.
2. **Simplification:** remove everything the change does not need; reuse what the tree already has.
3. **Naming:** every new name reads as what it is.
4. **Lint and analysis:** `make check` passes clean. The layout, literal and MUST_CHECK lints, cppcheck and
   clang-tidy report nothing at all: there is no suppression and no list of accepted findings.
5. **Adversarial self-review:** try to refute your own change: its contract, its checks, its numbers, its failure
   paths. A machine review pass is preferred, by **Fable with its defensive security review** or by **Daybreak
   Blue**. Attach its findings to the pull request, each one marked fixed or refuted with the evidence.

## Testing: long, and on the right builds
- **`make check` on Linux x86-64:** lint, header_check, analysis, the host suites in all four builds, python_tests,
  release_audit and self_test. The four builds are debug (ASan and UBSan), thread (TSan and UBSan), profile and
  release.
  [DEVELOPMENT.md](docs/DEVELOPMENT.md) has the commands.
- **Backend, kernel and loader changes** also pass `make card_tests`. A kernel has a double-precision host reference
  and a test that fails when one stored word changes. It runs clean under `compute-sanitizer` memcheck, racecheck,
  initcheck and synccheck.
- **Identity:** a change that claims to keep outputs shows them token for token against the base, on the same
  prompts, for every placement it can reach.
- **Quality against the golden reference:** a change that can move any number is scored with
  [golden_score.py](tools/golden_score.md) against the golden reference, the Hugging Face dataset
  [anoane/anemone-ds41f1017-golden](https://huggingface.co/datasets/anoane/anemone-ds41f1017-golden). The reference
  is 435 cases and 66,409 positions, generated at T=0 from the original full weights with the exact profile, which
  follows DeepSeek's reference inference arithmetic. Scored against its full F32 heads, the scorer gives the NLL,
  perplexity, top-1 agreement, the full KL, certain-token agreement and the logit deltas. Top-k agreement and the
  bucketed top-k KL (a lower bound of the full KL) need its compact top-64 reference (`--compact`), which gives every
  score below with the bucketed KL in place of the full KL, and no logit deltas. Report the scores beside the base's:
  - NLL and perplexity of the golden continuation;
  - top-1 agreement, and top-k agreement (`--compact`);
  - KL to the reference: the full KL, or the bucketed top-k KL with `--compact`;
  - agreement on the tokens the reference is certain of.
  Exact must stay bit-identical, and balanced must stay within the base's noise. A measurable loss can only ever be
  an opt-in fast option, never a default.
- **Speed:** a speed claim gives an A B B A (or longer) measurement on named hardware, with the command, every run's
  number and the spread. A faster path that keeps identical outputs becomes the default.
- **Hardware gates:** every merge needs CPU proof on the reference machine (4 × CMP 170HX + RTX PRO 6000 Blackwell):
  sanitizers, an independent audit and object or dry-run identity outside its intended changes. GPU output identity is
  checked at the release gate, on PP5, TP4_PRO and TP4_EXL3. You don't need that hardware to contribute; say what you
  could and could not run.

## Commits and pull requests
- The subject says what changed. The body says why, and what evidence shows it works.
- No tool or assistant names in code or commit messages, no private machine details, no generated code and no
  binaries.
- Every contribution must be licensed under the project's licence, AGPL-3.0-or-later. A new file starts with its
  `SPDX-License-Identifier` line, or `REUSE.toml` covers it where a header would change its format, and `reuse lint`
  must pass.
- Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) completely. A box left unticked needs a
  reason.

## Issues and requests
- **Bugs:** use the bug report template: version or commit, hardware, driver and CUDA, the exact command, what you
  expected and what happened, and the log with keys and tokens removed.
- **Features and performance:** use the feature request template. Say what problem it solves and how we would measure
  it.
- **Security issues** are reported like any other bug, in a public issue with the bug report template and a minimal
  reproduction.
