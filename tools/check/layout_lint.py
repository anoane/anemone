#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fails when the layout of the tree breaks a rule of STYLE.md.

    layout_lint.py <the tree's root>

Reported:
  - a line of an engine or test file wider than 120 columns (a tab reaches the next multiple of 8), a file that is
    not UTF-8 text;
  - in source/ and tests/: a file that is not a lower_case_underscore name ending .c, .h, .cu or .cuh (in source/ a
    module.mk too), and in generated/ one that is not a lower_case_underscore name ending .inc: the build compiles
    nothing else, so nothing would check it. tests/check_cases/ is not looked into;
  - a folder more than three deep below source/, tests/ or generated/ (engine/library/block_scaled_linear/,
    models/<model>/<part>/);
  - an engine file in no module: a module is a folder below source/ that holds a module.mk, with the folders below
    it that hold none (a model's part folders); source/main.c alone lies at the root;
  - a name used twice in source/, tests/ and generated/: a kernel module and a test program are named by it, and an
    include spelled from its root finds one file;
  - no engine file at all, and a top-level directory this tool does not know, where code would be checked by nothing.
There is no limit on lines (the owner, 2026-10-25): the code is as long as what it needs.

Check tool: not part of the engine.
"""
import os
import re
import sys

MOST_COLUMNS_PER_LINE = 120
COLUMNS_PER_TAB_STOP = 8
MOST_FOLDER_DEPTH = 3
EXPECTED_ARGUMENT_COUNT = 2
EXIT_STATUS_USAGE = 2

CODE_NAME = re.compile(r"[a-z][a-z0-9_]*\.(?:c|h|cu|cuh)")
TABLE_NAME = re.compile(r"[a-z][a-z0-9_]*\.inc")
MODULE_FILE = "module.mk"
PROGRAM_FILE = "source/main.c"
# (the directory walked, the names accepted in it, what its files may be, whether lines are checked)
CHECKED_DIRECTORIES = (("source", CODE_NAME, ".c, .h, .cu or .cuh, or module.mk", True),
                       ("tests", CODE_NAME, ".c, .h, .cu or .cuh", True),
                       ("generated", TABLE_NAME, ".inc", False))
CASES_DIRECTORY = "tests/check_cases"
KNOWN_TOP_LEVEL_DIRECTORIES = ("source", "tests", "tools", "build", "docs", "generated", "LICENSES", ".git",
                               ".github")


def files_under(root, top):
    """The files under root/top, as sorted paths relative to root; tests/check_cases/ is not entered."""
    found = []
    for directory, directory_names, names in os.walk(os.path.join(root, top), followlinks = True):
        relative_directory = os.path.relpath(directory, root).replace(os.sep, "/")
        directory_names[:] = sorted(name for name in directory_names
                                    if f"{relative_directory}/{name}" != CASES_DIRECTORY)
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


def module_folder(root, path):
    """The folder of the module.mk nearest above path below source/, or None."""
    folder = os.path.dirname(path)
    while folder.startswith("source/"):
        if os.path.isfile(os.path.join(root, folder, MODULE_FILE)):
            return folder
        folder = os.path.dirname(folder)
    return None


def main():
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT or not os.path.isdir(sys.argv[1]):
        print(__doc__, file = sys.stderr)
        return EXIT_STATUS_USAGE
    root = sys.argv[1]
    problems, engine_count, test_count, names = [], 0, 0, {}
    for top, accepted_name, kinds, lines in CHECKED_DIRECTORIES:
        for path in files_under(root, top):
            name = os.path.basename(path)
            depth = path.count("/") - 1
            if depth > MOST_FOLDER_DEPTH:
                problems.append(f"{path}: more than {MOST_FOLDER_DEPTH} folders deep below {top}/")
                continue
            if top == "source" and name == MODULE_FILE:
                continue
            if accepted_name.fullmatch(name) is None:
                problems.append(f"{path}: not a file the build compiles: only {top}/[<folders>/]"
                                f"<lower_case_underscore name>{kinds}")
                continue
            if top == "source" and path != PROGRAM_FILE and module_folder(root, path) is None:
                problems.append(f"{path}: in no module: no {MODULE_FILE} in its folder or above it below source/")
            if name in names:
                problems.append(f"{path}: the name of {names[name]}: a name is the tree's once")
            names.setdefault(name, path)
            if lines:
                lines_checked(root, path, problems)
            engine_count += 1 if top == "source" else 0
            test_count += 1 if top == "tests" else 0
    for name in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, name)) and name not in KNOWN_TOP_LEVEL_DIRECTORIES:
            problems.append(f"{name}/: a top-level directory this tool does not know: nothing checks what it holds")
    if engine_count == 0:
        problems.append("no engine file found in source/")
    for problem in problems:
        print(problem, file = sys.stderr)
    print(f"layout: {engine_count} engine files, {test_count} test files; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
