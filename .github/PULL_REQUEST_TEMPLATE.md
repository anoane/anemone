<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Read CONTRIBUTING.md and docs/STYLE.md first. A box left unticked needs a reason on its line. -->

## What changes and why
<!-- One change. Link the issue that agreed its scope. -->

Closes #

## Evidence
- **Outputs:** identical to the base (prompts, placements, how compared) / intentionally changed (why, and the quality
  measurement):
- **Speed (if claimed):** hardware, command, A B B A runs, spread:
- **What I could not run, and why** (for example card tests or the hardware gates):

## Checklist
- [ ] One change; no unrelated refactor or formatting.
- [ ] Only the code that is needed; nothing duplicated; names in full words; no bare numbers.
- [ ] Every input checked; every fallible function returns `MUST_CHECK result_status`; failure paths reachable
      through `FAULT_POINT(name)` and tested.
- [ ] No escape hatches (`NOLINT`, `cppcheck-suppress`, pragmas, fast-math) and no new dependency.
- [ ] Formatted with the repository's `.clang-format`.
- [ ] `make check` passes on Linux x86-64: lint, header_check, analysis (no findings), the host suites in all
      four builds, python_tests, release_audit and self_test.
- [ ] `make card_tests` passes (backend, kernel or loader changes); kernels clean under compute-sanitizer.
- [ ] New behaviour and its refusals have tests.
- [ ] Numbers that can move are scored against the golden reference (NLL, perplexity, top-1/top-k agreement, KL,
      certain-token agreement) beside the base's scores.
- [ ] A new option has a measurement, a knob catalogue entry and its profile-axis mapping.
- [ ] Docs updated where behaviour or options changed.
- [ ] Adversarial review done. Preferred: Fable with its defensive security review, or Daybreak Blue. Findings
      attached, each fixed or refuted with evidence.
- [ ] No tool or assistant names, private machine details, generated code or binaries.
