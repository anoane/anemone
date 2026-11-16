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
- At most 120 columns to a line; the layout is the repository's `.clang-format` (the Linux format) wherever a
  file is formatted. No count of lines (the owner, 2026-10-25): write only the code that is needed, in the fewest moving
  parts the architecture allows: no speculative generality, no duplication, no thousand lines where better logic
  needs a hundred. Less code is easier to keep correct, to maintain and to make fast. Engine files are
  `include/anemone/<name>` and
  `source/<name>`, tests are `tests/<name>`, all `.c`, `.h`, `.cu` or `.cuh`; nothing else lies beside them.
- `source/main.c` holds only the program's `main()`; every other source file, `source/commands.c` (the commands)
  among them, is linked into the program and into every test. A test is
  `tests/<name>_test.c`; one that needs the cards is `tests/<name>_card_test.c`, built with the others and run by
  `make card_tests`. What tests share (`tests/<name>.c`, `.h`) is prefixed `test_` in its names.
- No escape hatch: `NOLINT`, `cppcheck-suppress`, `#pragma GCC diagnostic`, `system_header` or `optimize`, the
  `optimize` and `no_sanitize` attributes, `<assert.h>`, `fast-math`, `use_fast_math` and `FP_CONTRACT` are
  forbidden text, in comments too. cppcheck takes no suppression from a source file.
- One way to do each thing. No option without a measurement behind it. No abstraction beyond the five seams
  (`include/anemone/*_seam.h`) until it has two users. (by review)

## What every function does
- It checks every input, size, bound and return value: bad input is a `result_status`, never a crash. (by
  review; a status that nobody looks at does not compile)
- If it can fail it returns `MUST_CHECK result_status` (`include/anemone/status.h`). On failure it leaves every
  scalar output untouched; the contents of a caller's buffer it was filling are unspecified and must not be used.
  (by review)
- It allocates nothing on the token path: what a card holds is placed at load, at fixed offsets. Every wait has a
  deadline. Its failure paths can be reached by a test: a `FAULT_POINT(name)` (`include/anemone/hooks.h`) stands
  where the outside world can fail. (by review)

## Quality, safety and security bar
The owner's bar (2026-11-07): code of the highest quality, written to MISRA C practice, secure, audited, reviewed,
free of known defects, memory-efficient and small. The engine follows MISRA C:2012 (with its amendments) as a
practice, not a certification claim: where a rule is enforced by `make check` it is unmarked, otherwise it is
marked (by review) until `make analysis` runs a MISRA checker. Kernels follow the kernel rules below instead.
- Behaviour is defined: nothing relies on undefined, unspecified or implementation-defined behaviour beyond what
  Linux on x86-64 and the CUDA driver interface document. Every conversion that can change a value is explicit
  and its range proven where it is written (essential types; `-Wconversion` holds the rest).
- Control flow is plain: no recursion, so every stack is bounded; every loop and every wait has a bound or a
  deadline; every `switch` handles every enumerator; `goto` only jumps forward to a cleanup or refusal label of
  the same function (the one deviation from advisory rule 15.1). No side effect in the right operand of `&&`/`||`,
  in `sizeof`, or in a hook's argument; `++` and `--` stand alone. (by review)
- Types and pointers: no unions; type punning only through `memcpy`; pointer arithmetic stays inside one object;
  a cast between object pointer types goes only to or from `unsigned char *` or `void *` with alignment proven;
  every parameter that is not written is `const`; everything not exported is `static`. (by review)
- External input is hostile: every byte from a socket, request, file, pack or environment is validated against
  named limits (body size, nesting depth, string length, counts, enum ranges) before use. Every size computation
  that can overflow is checked before it is used to allocate, index or copy. Parsers of external input have fuzz
  cases under `tests/` run with the sanitizers. (by review)
- Library use is bounded: no `strcpy`, `strcat`, `sprintf`, `strtok`, `atoi`/`atof`/`atol`, `gets`,
  `setjmp`/`longjmp`; text goes through `snprintf` with its result checked, or `memcpy` with checked sizes.
  (by review until the lint refuses the names)
- Secrets: key comparison is constant-time; a key is never logged, echoed in an error, placed in a label or kept
  in a buffer longer than its check; failed authentication reveals nothing about which part failed. (by review)
- Memory: what the engine holds is sized from named constants and placed at load; nothing is allocated per token
  or per request on the serving path; every load-time allocation has its failure path and one owner that frees
  it. A new table, cache or buffer states its peak bytes in its comment and in the change's report. Prefer
  streaming over buffering a whole body. (by review)
- Footprint: one implementation per concept, table-driven where data says it better than code, no dead code,
  no copy of a helper that exists. A change reports its added and removed lines and the release binary's size
  difference. (by review)
- Bug free means no known defect is merged: every finding of the analysers, the sanitizers, the fuzz cases and
  the reviews is fixed before the merge, or recorded as a deviation (rule, scope, reason, the test that covers
  it) in this section. (by review)

### Retained frontend storage deviation
The shared HTTP/Chat frontend predates this bar and retains bounded dynamic `chat_buffer` storage for input,
normalized JSON, DSML arguments and pending output, plus the existing Chat/template reader workspaces. It may grow
those buffers during admission or formatting; this is a scoped deviation from the load-only frontend allocation
rule, not permission for token-worker or device allocation. Input and normalized bodies are limited to32MiB and
pending output to64MiB. The Messages adapter stores formatter state inline, reuses existing answer buffers and the
poller's already allocated prompt scratch, and adds no direct allocator or per-token table. Its temporary views
have one synchronous poll-thread owner. Allocation failures remain server errors; size/complexity limits are
client errors. Coverage: server_messages_test (including chat_memory injection), the hostile/oversized/socket
cases in messages_contract_test.py, and the release/sanitizer host suites. The engine, sampler and reasoning policy
allocate nothing per request/token. A complete pool conversion of the inherited frontend is outside this change;
its bounded ownership and default wire behavior are retained.

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
1. `make check` passes on the server: lint, header_check, analysis, tests in all four builds, python_tests,
   release_audit and self_test, which shows each of these checks failing once on a case under `tests/check_cases/`.
   A change to the backend, a kernel or the loader also passes `make card_tests`.
2. A second reader has tried to refute the file: its contract, its checks, its numbers. Before a merge the
   change is audited independently along correctness, default-path identity, security, memory and contract;
   each finding is confirmed or refuted on the code, and every confirmed one is fixed or recorded as a deviation.
   Kernels additionally pass `compute-sanitizer` memcheck, racecheck, initcheck and synccheck.
3. The commit is authored by the owner's identity; its subject says what changed and nothing else. (by review)

## What never enters the repository (by review)
Code from another project (ideas only), generated code, anything built on a development laptop, private details of
any machine, and the names of tools or assistants used to write the code.
