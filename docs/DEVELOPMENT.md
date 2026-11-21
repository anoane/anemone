<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Development

The engine is C11 host code and C-like CUDA. `source/` holds it in module
folders, each with its headers beside its sources and a `module.mk`: `common/`
(what everything shares), `engine/` (every model-independent component: formats,
backend, program ABI, kernel library, storage, Engram, input, runtime, server,
command line) and `models/` (the model contract, the registry and one folder per
model architecture). `tests/` mirrors it with the native tests (`support/` for
what every test links), and `generated/` holds pinned generated inputs. [STYLE.md](STYLE.md) and the root `.clang-format`
define formatting. Build artifacts are written under `build/`.

## Build and host checks

Use the Linux toolchain in [BUILD.md](../BUILD.md):

```sh
make -j32 release
make host_tests
make -j32 header_check
```

Optional debug, thread-sanitizer and profiling variants use the same sources:

```sh
make -j32 debug thread profile
make host_tests TEST_BUILD=debug
make host_tests TEST_BUILD=thread
make host_tests TEST_BUILD=profile
```

These optional commands are separate from a release-only verification. Native
host tests exclude GPU card tests. GPU tests and measurement probes require
explicitly reserved devices and the operator's own execution guard; they do not
run as part of the source build.

## Regenerate pinned inputs

Python 3 is needed only for explicit development/data operations:

```sh
make regenerate_axes
make regenerate_help
make regenerate_vision
make check_generated
```

The axis and help generators live in `tools/cli/`. Vision generators, fixture
inputs and their source are included. Release compilation consumes pinned
headers and uses the native C kernel embedder, with no Python prerequisite.
See [the tools index](../tools/README.md) for source and optional dependencies.

Additional checks are available through `make python_tests`, `make release_audit`,
`make lint`, `make analysis` and `make self_test`. Some require NumPy, PyTorch,
clang-tidy or cppcheck. `make check` aggregates these broader checks and must
pass clean: the lints and both analysers report nothing, with no suppression and
no list of accepted findings ([CONTRIBUTING.md](../CONTRIBUTING.md)). Release
builds and native test results have their own measured scope.

`make -j3 analysis` runs its three hook combinations side by side. In each,
cppcheck runs `CPPCHECK_JOBS` processes (by default a third of the processors,
plus one) with its own work folder under `build/`, so that its checks across
files still run, and clang-tidy runs `CLANG_TIDY_JOBS` processes (as many by
default), a file each.

Keep generated binaries, object files, models, credentials and local environments
out of Git. Preserve numerical contracts and qualify changes against the relevant
native, generated-code and workload evidence. A parsed option, slot allocation
or source review does not establish model quality or full-context correctness.
