#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fails if an executable of the release build still holds anything a hook should have compiled out.

    release_has_no_hooks.py <release build directory>

Looks inside every executable under the directory (object files hold no final code and are skipped) for:

  - a symbol, function or static data, whose name starts with a prefix only the hook layer uses. A prefix, not a
    list of names: a hook added later is covered as long as it keeps the prefix;
  - text only a hook could have left: the hook layer's own messages, and the names and the log text that
    tests/hooks_test.c passes through every hook macro, so a macro that failed to compile out is seen at its use;
  - the absence of the symbol every program has. Without it the executable was stripped and proves nothing.

What a binary cannot show, that a compiled-out hook does not evaluate its argument, is the test's own job.

Check tool: not part of the engine.
"""
import os
import subprocess
import sys

EXPECTED_ARGUMENT_COUNT = 2
EXIT_STATUS_USAGE = 2

# Functions first, then the static data of source/hooks.c and the totals the timer and counter macros declare.
# Generic fault_* state also belongs to required runtime failures and synthetic
# test backends; only the hook layer's fault entry points identify instrumentation.
FORBIDDEN_SYMBOL_PREFIXES = ("hook_", "hooks_", "assertion_", "fault_arm", "fault_disarm", "fault_point_", "hook_total_",
                             "registered_totals", "armed_fault_", "timer_", "counter_",
                             "log_", "log_level_names", "log_threshold", "__asan_", "__ubsan_",
                             "__tsan_", "__sanitizer_", "__gcov_", "__llvm_profile_", "__cyg_profile_",
                             "mcount", "__fentry__", "__gnu_mcount", "nvtx", "cupti")
FORBIDDEN_TEXTS = (b"assertion failed", b"timer %s", b"counter %s", b"neither must this", b"hooks_test_timer",
                   b"hooks_test_counter", b"hooks_test_fault", b"hooks_test_release_error",
                   b"hooks_test_release_warning", b" anemone %s %s:%d: %s")
REQUIRED_SYMBOL = "main"
SKIPPED_SUFFIXES = (".o", ".d")


def leftovers(symbol_names, content):
    """What the symbol names and the bytes of one executable hold that a release build must not."""
    found = [f"symbol {name}" for name in sorted(symbol_names) if name.startswith(FORBIDDEN_SYMBOL_PREFIXES)]
    found += [f"text '{text.decode()}'" for text in FORBIDDEN_TEXTS if text in content]
    if REQUIRED_SYMBOL not in symbol_names:
        found.append(f"no symbol {REQUIRED_SYMBOL}: stripped, so nothing can be proved about it")
    return found


def symbol_names_of(path):
    """Every name in the symbol table of an executable, defined or not."""
    listing = subprocess.run(["nm", path], capture_output = True, text = True, check = True).stdout
    return {line.split()[-1] for line in listing.splitlines() if line.split()}


def main():
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT:
        print(__doc__, file = sys.stderr)
        return EXIT_STATUS_USAGE
    problems, checked_count = [], 0
    for directory, directory_names, names in os.walk(sys.argv[1]):
        directory_names.sort()
        for name in sorted(names):
            path = os.path.join(directory, name)
            if name.endswith(SKIPPED_SUFFIXES) or not os.path.isfile(path) or not os.access(path, os.X_OK):
                continue
            checked_count += 1
            try:
                with open(path, "rb") as handle:
                    problems += [f"{path}: {leftover}" for leftover in leftovers(symbol_names_of(path), handle.read())]
            except (OSError, subprocess.CalledProcessError) as error:
                problems.append(f"{path}: cannot be examined: {error}")
    if not checked_count:
        problems.append(f"{sys.argv[1]}: no executable to check")
    for problem in problems:
        print(problem, file = sys.stderr)
    print(f"release has no hooks: {checked_count} executables checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
