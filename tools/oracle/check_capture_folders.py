#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checks every capture folder under a directory: the manifest says it is complete, and tensors.bin holds exactly
the bytes the manifest's records describe, one after another without gaps.

    check_capture_folders.py <directory of capture folders>

Measurement tool: not part of the engine.
"""
import json
import os
import sys

TENSORS_FILE_NAME = "tensors.bin"
MANIFEST_FILE_NAME = "manifest.json"
EXIT_CODE_PROBLEMS = 1
EXIT_CODE_USAGE = 2


def check_folder(folder):
    with open(os.path.join(folder, MANIFEST_FILE_NAME), encoding = "utf8") as handle:
        manifest = json.load(handle)
    file_bytes = os.path.getsize(os.path.join(folder, TENSORS_FILE_NAME))
    expected_offset, problems = 0, []
    for record in manifest["tensors"]:
        if record["byte_offset"] != expected_offset:
            problems.append(f"{record['name']} starts at {record['byte_offset']}, expected {expected_offset}")
        expected_offset = record["byte_offset"] + record["byte_count"]
    if expected_offset != file_bytes:
        problems.append(f"records end at {expected_offset}, the file has {file_bytes} bytes")
    if manifest.get("complete") is not True:
        problems.append("the manifest does not say complete")
    return manifest, file_bytes, problems


def main():
    if len(sys.argv) != 2:
        print(__doc__, file = sys.stderr)
        return EXIT_CODE_USAGE
    problem_count = 0
    for name in sorted(os.listdir(sys.argv[1])):
        folder = os.path.join(sys.argv[1], name)
        if not os.path.isfile(os.path.join(folder, MANIFEST_FILE_NAME)):
            continue
        manifest, file_bytes, problems = check_folder(folder)
        problem_count += len(problems)
        print(f"{name:16s} {len(manifest['tensors']):5d} tensors, {file_bytes:12d} bytes: {'consistent' if not problems else '; '.join(problems)}")
    return EXIT_CODE_PROBLEMS if problem_count else 0


if __name__ == "__main__":
    sys.exit(main())
