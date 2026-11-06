#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fails when the layout of the tree breaks a rule of STYLE.md.

    layout_lint.py <the tree's root>

Reported:
  - a line of an engine or test file wider than 120 columns (a tab reaches the next multiple of 8), a file that is
    not UTF-8 text;
  - in include/, source/ and tests/: a file that is not include/anemone/<name>, source/<name> or tests/<name> with
    a lower_case_underscore name ending .c, .h, .cu or .cuh, and any directory but include/anemone/. The build
    compiles neither, so nothing would check them. tests/check_cases/ is not looked into;
  - no engine file at all, and a top-level directory this tool does not know, where code would be checked by nothing.
There is no limit on lines (the owner, 2026-10-25): the code is as long as what it needs.

Check tool: not part of the engine.
"""
import os
import re
import sys

MOST_COLUMNS_PER_LINE = 120
COLUMNS_PER_TAB_STOP = 8
EXPECTED_ARGUMENT_COUNT = 2
EXIT_STATUS_USAGE = 2

ENGINE_FILE = re.compile(r"(?:include/anemone|source)/[a-z][a-z0-9_]*\.(?:c|h|cu|cuh)")
TEST_FILE = re.compile(r"tests/[a-z][a-z0-9_]*\.(?:c|h|cu|cuh)")
# (the directory walked, the paths accepted in it, where its files must be)
CHECKED_DIRECTORIES = (("include", ENGINE_FILE, "include/anemone/"), ("source", ENGINE_FILE, "source/"),
                       ("tests", TEST_FILE, "tests/"))
DIRECTORIES_THE_BUILD_COMPILES = ("include/anemone", )
CASES_DIRECTORY = "tests/check_cases"
KNOWN_TOP_LEVEL_DIRECTORIES = ("include", "source", "tests", "tools", "build", "docs", "generated", ".git")


def files_under(root, top, problems):
    """The files under root/top, as sorted paths relative to root. A directory the build does not compile is
    reported and not entered."""
    found = []
    for directory, directory_names, names in os.walk(os.path.join(root, top), followlinks = True):
        relative_directory = os.path.relpath(directory, root).replace(os.sep, "/")
        entered = []
        for name in sorted(directory_names):
            path = f"{relative_directory}/{name}"
            if path in DIRECTORIES_THE_BUILD_COMPILES:
                entered.append(name)
            elif path != CASES_DIRECTORY:
                problems.append(f"{path}/: a directory the build does not compile")
        directory_names[:] = entered
        found += [f"{relative_directory}/{name}" for name in sorted(names)]
    return found


def lines_checked(root, path, problems):
    """Each line of a file that is too wide, and a file that is not UTF-8, reported."""
    try:
        with open(os.path.join(root, path), encoding = "utf8") as handle:
            for number, line in enumerate(handle, start = 1):
                columns = len(line.rstrip("\n").expandtabs(COLUMNS_PER_TAB_STOP))
                if columns > MOST_COLUMNS_PER_LINE:
                    problems.append(f"{path}:{number}: {columns} columns, a line has at most {MOST_COLUMNS_PER_LINE}")
    except UnicodeDecodeError:
        problems.append(f"{path}: not UTF-8 text")


def main():
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT or not os.path.isdir(sys.argv[1]):
        print(__doc__, file = sys.stderr)
        return EXIT_STATUS_USAGE
    root = sys.argv[1]
    problems, engine_count, test_count = [], 0, 0
    for top, accepted_file, place in CHECKED_DIRECTORIES:
        for path in files_under(root, top, problems):
            if accepted_file.fullmatch(path) is None:
                problems.append(f"{path}: not a file the build compiles: "
                                f"only {place}<lower_case_underscore name>.c, .h, .cu or .cuh")
                continue
            lines_checked(root, path, problems)
            engine_count += 1 if accepted_file is ENGINE_FILE else 0
            test_count += 0 if accepted_file is ENGINE_FILE else 1
    for name in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, name)) and name not in KNOWN_TOP_LEVEL_DIRECTORIES:
            problems.append(f"{name}/: a top-level directory this tool does not know: nothing checks what it holds")
    if engine_count == 0:
        problems.append("no engine file found in include/anemone/ or source/")
    for problem in problems:
        print(problem, file = sys.stderr)
    print(f"layout: {engine_count} engine files, {test_count} test files; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
