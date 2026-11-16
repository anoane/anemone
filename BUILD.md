<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Build Anemone from source

Anemone builds on Linux x86-64. The recorded toolchain is CUDA 13.5 V13.5.37,
GCC 14.2.0 and GNU Make 4.4.1. Install the CUDA toolkit, CUDA driver development
library (`libcuda`), a C/C++ compiler, binutils and Make through the target
system's normal setup. The reference host is AMD Zen 4; `HOST_ARCHITECTURE` in
[Makefile](Makefile) selects that host target. See [README](README.md#target-hardware)
for the EXL3 and original-weight GPU configurations.

## Clean build

Enter a clean checkout or extracted source tree and run:

```sh
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin make -j32 release
env -i HOME="$(mktemp -d)" PATH=/usr/local/cuda/bin:/usr/bin:/bin make host_tests
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin make -j32 header_check
sha256sum build/release/anemone
```

The release executable is `build/release/anemone`. The v0.0.3-alpha reference
executable SHA256 (source `f57d81375a8830db16925e42740aae1fe0773e90`) with the
recorded toolchain and flags is:

```text
2afdfc7138b74baa94f9bfe9f426d188e6af1be8e6729230e6c4f21f070d317b
```

Compilation and native host/header checks need no weights or GPU context.
Release compilation uses native C/CUDA helpers and pinned inputs, with no Python,
Git checkout, package download or code-generation overlay prerequisite. It
compiles 119 CUDA modules for the selected class-80/class-120 targets; host tests
run 157 programs excluding `_card_test`, and header checks cover 675 compilations.
`make host_tests` needs a `HOME`: under `env -i` without one, a case of
`replica_prepare_test` fails, so the command above gives it an empty folder.
The `v0.0.3-alpha` tag marks the published source snapshot. Its build inputs are
those of the development source `f57d813` named above, which is not in the
published repository; [VALIDATION.md](docs/VALIDATION.md) gives the scope.

`CUDA_DIRECTORY` defaults to `/usr/local/cuda`; override it for another installed
toolkit location. The engine links the CUDA driver library and libm. Do not
remove warning/error flags or numerical-order controls to make a build pass.
The current optimized host flags retain `-fno-tree-vectorize` because an earlier
GCC vectorizer changed program-builder behavior; this is a correctness setting.

## Build variants and checks

| Command | Purpose |
| --- | --- |
| `make release` | Optimized executable with optional hooks compiled out. |
| `make debug` | Address/undefined-behavior checking. |
| `make thread` | Thread sanitizer checking. |
| `make profile` | Optimized executable with timing hooks. |
| `make host_tests TEST_BUILD=release` | Native CPU tests for a selected build. |
| `make header_check` | Header-alone checks under the three hook combinations. |
| `make release_audit` | Host/module instrumentation audits; optional Python/binutils checks. |
| `make check_generated` | Explicit CLI/vision regeneration consistency. |
| `make clean` | Remove generated build output. |

The full development sources, generators, numeric fixtures and Python checks are
included in both the public checkout and source archive. They are described in
[DEVELOPMENT.md](docs/DEVELOPMENT.md) and [the tools index](tools/README.md).
Optional checks have their own dependencies. `make check` runs lint, header_check,
analysis, the host suites in all four builds, python_tests, release_audit and
self_test, and must pass clean. For v0.0.3-alpha every part but self_test was run
on the reference machine and passed: the lints and `make analysis` report nothing,
and every host suite passes in release, debug, thread and profile.

Models and tokenizer preparation are separate CPU operations documented in
[MODEL_DATA.md](docs/user/MODEL_DATA.md). Starting inference, deploying services
and HTTP requests are described in [INSTALL.md](INSTALL.md) and require prepared
model data. They are not part of a source-only build verification.
