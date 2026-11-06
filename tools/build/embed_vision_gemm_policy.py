#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build-only numeric data bridge: <gemm_policy.json> <build/vision_gemm_policy.inc>."""
import json
import os
from pathlib import Path
import sys

TABLES = {"cmp_mlp_down": (7, 9216), "cmp_projector_first": (82, 1024),
          "cmp_mlp_up": (1, 9216), "cmp_projector_second": (6, 1024),
          "pro_mlp_down": (12, 9216), "pro_projector_first": (77, 1024),
          "pro_mlp_up": (3, 9216), "pro_query_key_value": (7, 9216),
          "pro_attention_output": (5, 9216), "pro_projector_second": (10, 1024)}
OFFSET_TABLES = ("cmp_projector_first", "pro_projector_first")
MMA_K8_OPTION = 4
OPTION_MASK = 7


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    source, output = (Path(value) for value in sys.argv[1:])
    data = json.loads(source.read_text(encoding="ascii"))
    if (set(data) != {"boundaries", "offsets"} or set(data["boundaries"]) != set(TABLES) or
            set(data["offsets"]) != {name + "_offsets" for name in OFFSET_TABLES}):
        raise ValueError("unexpected GEMM policy table names")
    lines = ["/* SPDX-License-Identifier: AGPL-3.0-or-later */"]
    for name, (count, maximum) in TABLES.items():
        records = data["boundaries"][name]
        if (len(records) != count or any(not isinstance(row, list) or len(row) != 3 or
                any(type(value) is not int for value in row) or not 1 <= row[0] <= maximum or
                not 1 <= row[1] <= 11 or not 0 <= row[2] <= OPTION_MASK or
                (row[1] == 1 and row[2] not in (0, MMA_K8_OPTION)) for row in records)):
            raise ValueError(f"invalid boundary count or numeric range: {name}")
        if name in OFFSET_TABLES:
            offsets = data["offsets"][name + "_offsets"]
            if (len(offsets) != 9 or any(type(value) is not int for value in offsets) or
                    offsets[0] != 0 or offsets[-1] != count or
                    any(first >= last for first, last in zip(offsets, offsets[1:]))):
                raise ValueError(f"invalid residue offsets: {name}")
            ranges = [records[first:last] for first, last in zip(offsets, offsets[1:])]
            if any(any(row[0] % 8 != residue for row in rows) or
                   rows[-1][0] != maximum - (maximum - residue) % 8 for residue, rows in enumerate(ranges)):
                raise ValueError(f"invalid residue boundaries: {name}")
        else:
            ranges = [records]
            if records[-1][0] != maximum:
                raise ValueError(f"incomplete boundary range: {name}")
        if any(any(first[0] >= last[0] for first, last in zip(rows, rows[1:])) for rows in ranges):
            raise ValueError(f"nonmonotonic boundaries: {name}")
        lines += [f"static const struct split_boundary {name}[] = {{"]
        lines += ["\t{ " + ", ".join(str(value) for value in row) + " }," for row in records]
        lines += ["};"]
    for name in OFFSET_TABLES:
        symbol = name + "_offsets"
        lines += [f"static const uint16_t {symbol}[] = {{ " +
                  ", ".join(str(value) for value in data["offsets"][symbol]) + " };"]
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
