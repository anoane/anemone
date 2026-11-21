#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The case keys of a golden dataset, one `set/id` a line, for `anemone capture --capture_cases=KEYS`.

    golden_cases.py ROOT [--output KEYS]

The keys are the folders ROOT/cases/<set>/<id>/ that hold a metadata.json, sorted by set, then id (the capture runs
them in the file's order). A key is refused unless set and id are of [A-Za-z0-9_.-] (the capture's own rule). When
ROOT/index.json lists its cases (a list of "set/id" strings, or of objects with "key" or "set" and "id", alone or as
its "cases" member), the keys must be exactly those; any other index.json is reported, not read. The release's 54
cases are tools/models/deepseek_v41_flash/golden/selections/release54.txt. The output is create-only.

Measurement tool: not part of the engine.
"""
import argparse
import json
import re
import sys
from pathlib import Path

KEY_PART = re.compile(r"[A-Za-z0-9_.-]+")


def index_keys(index):
    """The keys an index.json lists, or None for a shape this tool does not read."""
    cases = index.get("cases") if isinstance(index, dict) else index
    if not isinstance(cases, list):
        return None
    keys = []
    for entry in cases:
        if isinstance(entry, str):
            keys.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("key"), str):
            keys.append(entry["key"])
        elif isinstance(entry, dict) and isinstance(entry.get("set"), str) and isinstance(entry.get("id"), str):
            keys.append("%s/%s" % (entry["set"], entry["id"]))
        else:
            return None
    return keys


def golden_keys(root):
    root = Path(root)
    folders = [path.parent for path in (root / "cases").glob("*/*/metadata.json")]
    keys = sorted("%s/%s" % (folder.parent.name, folder.name) for folder in folders)  # as strings, as the index's
    bad = [key for key in keys if not all(KEY_PART.fullmatch(part) and ".." not in part for part in key.split("/"))]
    if not keys or bad:
        raise ValueError("%s: %s" % (root, "no case folder" if not keys else "keys outside [A-Za-z0-9_.-]: %s" % bad[:4]))
    note = "no index.json"
    if (root / "index.json").is_file():
        listed = index_keys(json.loads((root / "index.json").read_text()))
        if listed is None:
            note = "index.json not read (a shape this tool does not know)"
        elif sorted(listed) != keys:
            raise ValueError("%s: index.json lists %d cases, the folders hold %d (first differing %s)" % (
                root, len(listed), len(keys), sorted(set(listed) ^ set(keys))[:4]))
        else:
            note = "index.json lists the same %d cases" % len(listed)
    return keys, note


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="write the keys once; an existing file is refused")
    options = parser.parse_args(arguments)
    keys, note = golden_keys(options.root)
    text = "".join(key + "\n" for key in keys)
    if options.output is None:
        sys.stdout.write(text)
    else:
        with options.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print("golden_cases: %d keys; %s" % (len(keys), note), file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("golden_cases refused: %s" % error, file=sys.stderr)
        raise SystemExit(1)
