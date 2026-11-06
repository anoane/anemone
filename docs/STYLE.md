<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Anemone: how the code is written

`make check` enforces every rule here except those marked "(by review)", which the second reader holds. `tests/`
follows every rule and is linted and analysed like the engine. `tools/` (measurement probes, check scripts) is
exempt from the lint and the analysers; the rules apply to it by review.

## What the code looks like
- Host code is always C11: the CUDA backend drives the cards through the CUDA driver interface (`cuda.h`, a C
  header). Kernels are the only code in another language.
- `lower_case_underscore` for functions, variables, types and files; `UPPER_CASE` for macros and enumerators. A
  name says what the thing is, in full words: `expert_index`, not `idx`. (full words: by review)
- No bare numbers: only 0 and 1 may appear in code. Every other number gets its name in an object-like `#define`,
  an enumerator or a `static const` table. (a printable character used as a number: by review)
- At most 120 columns to a line; the layout is building_blocks/.clang-format's (the Linux format) wherever a file
  is formatted. No count of lines (the owner, 2026-10-25): write only the code that is needed, in the fewest moving
  parts the architecture allows: no speculative generality, no duplication, no thousand lines where better logic
  needs a hundred. Less code is easier to keep correct, to maintain and to make fast. Engine files are
  `include/anemone/<name>` and
  `source/<name>`, tests are `tests/<name>`, all `.c`, `.h`, `.cu` or `.cuh`; nothing else lies beside them.
- `source/main.c` is the program; every other source file is linked into it and into every test. A test is
  `tests/<name>_test.c`; one that needs the cards is `tests/<name>_card_test.c`, built with the others and run by
  `make card_tests`. What tests share (`tests/<name>.c`, `.h`) is prefixed `test_` in its names.
- No escape hatch: `NOLINT`, `cppcheck-suppress`, `#pragma GCC diagnostic`, `system_header` or `optimize`, the
  `optimize` and `no_sanitize` attributes, `<assert.h>`, `fast-math`, `use_fast_math` and `FP_CONTRACT` are
  forbidden text, in comments too. cppcheck takes no suppression from a source file.
- One way to do each thing. No option without a measurement behind it. No abstraction beyond the five seams
  (`include/anemone/*_seam.h`) until it has two users. (by review)

## What every function does
- It checks every input, size, bound and return value: bad input is an `anemone_status`, never a crash. (by
  review; a status that nobody looks at does not compile)
- If it can fail it returns `ANEMONE_MUST_CHECK anemone_status`. On failure it leaves every scalar output
  untouched; the contents of a caller's buffer it was filling are unspecified and must not be used. (by review)
- It allocates nothing on the token path: what a card holds is placed at load, at fixed offsets. Every wait has a
  deadline. Its failure paths can be reached by a test: an `ANEMONE_FAULT_POINT` stands where the outside world
  can fail. (by review)

## Builds
| Build | Checking hooks | Timing hooks | Compiled with |
|---|---|---|---|
| `debug` | 1 | 1 | `-O1`, the address and undefined-behaviour sanitizers |
| `thread` | 1 | 1 | `-O1`, the thread and undefined-behaviour sanitizers |
| `profile` | 0 | 1 | the flags of release, and `-g -fno-omit-frame-pointer` |
| `release` | 0 | 0 | `-O3 -flto=auto -fno-plt -DNDEBUG`, hardened compile and link |

- Checking hooks (`ANEMONE_CHECKING_HOOKS_ENABLED`): assertions, fault points, and the information, debug and
  trace logs. Timing hooks (`ANEMONE_TIMING_HOOKS_ENABLED`): timers, counters and their report. Logging is
  compiled only when either switch is enabled, including error/warning logs. Release retains status/error
  handling and product outputs. The build sets both switches; `hooks.h` refuses either being undefined.
- A hook's argument has no side effect: where the hook is compiled out it is not evaluated. (`++`, `--` and
  assignments are checked; a call that changes something: by review)
- A result never depends on a flag: the host is built with `-ffp-contract=off`, kernels with `--fmad=false`, and a
  fused multiply-add is written `fma()`. `-ffast-math`, `-Ofast` and `--use_fast_math` are candidates for fast
  mode only, with a measured gain and a pass of the quality gate. (the host flag is checked; the rest: by review)
- Warnings are errors. The undefined-behaviour sanitizer includes `float-cast-overflow` and `bounds-strict`, and
  the options of the sanitizers are set by the Makefile, never inherited.

## Kernels
- A `.cu` file holds device code only, in a C subset: plain functions and structs; a template only to stamp one
  kernel per shape, architecture or weight format; no classes, exceptions, lambdas or standard containers. It
  includes no engine header except record layouts written to compile as both C11 and CUDA. (by review)
- A kernel file is `source/kernel_<what it computes>.cu`; what kernel files share is a `.cuh` beside it. Both are
  pass the lint.
- Each is compiled once per architecture (`sm_80`, `sm_120a`) into one module under `build/kernels/<architecture>/`
  that the backend loads when it opens a card. The modules are the same in all four builds and are linked into
  the program as one table of bytes: a program reads no kernel from a path.
- nvcc: `-O3 -arch=<architecture> -std=c++17 --fmad=false -Werror all-warnings -Xcompiler
  -Wall,-Wextra,-Wconversion,-Wshadow,-Werror`; never `--use_fast_math`.
- Release modules additionally set `NDEBUG` and zero device checking. Host/module audits reject optional
  logging, assertion/fault/timing and sanitizer remnants; diagnostic builds and negative cases prove the
  audits detect them. Algorithmic validation, status propagation and bounded-progress deadlines stay enabled.
- A kernel has a host reference in double precision and `tests/<same name>_test.c`, which fails when one stored
  word is changed; kernel tests run under `compute-sanitizer --error-exitcode 1`; the digest of the modules is the
  kernel-build hash. (by review until the first kernel brings these to `make check`)

## Before a commit
1. `make check` passes on the server: lint, header_check, analysis, tests in all four builds, and self_test, which
   shows each of these checks failing once on a case under `tests/check_cases/`. A change to the backend, a kernel
   or the loader also passes `make card_tests`.
2. A second reader has tried to refute the file: its contract, its checks, its numbers.
3. The commit is authored by the owner's identity; its subject says what changed and nothing else. (by review)

## What never enters the repository (by review)
Code from another project (ideas only), generated code, anything built on a development laptop, private details of
any machine, and the names of tools or assistants used to write the code.
