#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sizes of a model pack by kind of tensor, read from the safetensors headers only (no weights are loaded).

Prints the largest name patterns (every number in a tensor name replaced by '*'), the bytes of each transformer
layer, what lies outside the layers, and whether the layers fit the cards of a machine once the fixed costs are
taken off: the answer to "does this pack fit in VRAM, and if not, how much has to live in RAM".

    pack_size_report.py <model directory> [--cards 91:1,61.5:4] [--layer-prefix layers.] [--patterns 0]

--cards lists usable GiB per card and how many such cards (default: one RTX PRO 6000 and four CMP 170HX).
Measurement tool: not part of the engine.
"""
import argparse
import glob
import json
import os
import re
import struct
import sys

BYTES_PER_GIB = 2 ** 30
SAFETENSORS_HEADER_LENGTH_BYTES = 8
SAFETENSORS_HEADER_LENGTH_LIMIT = 512 * 2 ** 20       # a header larger than this is not a header
PATTERNS_PRINTED = 40
DEFAULT_CARDS = "91:1,61.5:4"
# Fixed costs per card when a 1,048,576-token session is open (proposal 4.2 and the attention piece note)
TABLE_NAME_MARKERS = (".engram.embed.", ".ngram_embedding.")   # the n-gram tables: host memory or disk, never VRAM
POOL_GIB_PER_CARD_WORST_CASE = 3.125
WORKSPACE_GIB_PER_CARD = 4.0


def read_safetensors_header(path):
    with open(path, "rb") as handle:
        length_bytes = handle.read(SAFETENSORS_HEADER_LENGTH_BYTES)
        if len(length_bytes) != SAFETENSORS_HEADER_LENGTH_BYTES:
            raise ValueError(f"{path}: too short for a safetensors file")
        (header_length,) = struct.unpack("<Q", length_bytes)
        if header_length == 0 or header_length > SAFETENSORS_HEADER_LENGTH_LIMIT:
            raise ValueError(f"{path}: implausible header length {header_length}")
        header = json.loads(handle.read(header_length))
    file_size = os.path.getsize(path)
    data_start = SAFETENSORS_HEADER_LENGTH_BYTES + header_length
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        first_byte, end_byte = entry["data_offsets"]
        if not 0 <= first_byte <= end_byte <= file_size - data_start:
            raise ValueError(f"{path}: tensor {name} lies outside the file")
        tensors[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "bytes": end_byte - first_byte}
    return tensors


def parse_cards(text):
    cards = []
    for item in text.split(","):
        usable_gib, count = item.split(":")
        cards += [float(usable_gib)] * int(count)
    return cards


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_directory")
    parser.add_argument("--cards", default = DEFAULT_CARDS)
    parser.add_argument("--layer-prefix", default = None, help = "name prefix of a transformer layer; found automatically if omitted")
    parser.add_argument("--patterns", type = int, default = PATTERNS_PRINTED, help = "how many name patterns to print; 0 prints all")
    arguments = parser.parse_args()

    files = sorted(glob.glob(os.path.join(arguments.model_directory, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors files in {arguments.model_directory}")
    tensors = {}
    repeated = []                                    # a name stored in more than one file: the first one is counted
    for path in files:
        for name, entry in read_safetensors_header(path).items():
            if name in tensors:
                repeated.append((name, tensors[name]["file"], os.path.basename(path), entry["bytes"]))
                continue
            tensors[name] = dict(entry, file = os.path.basename(path))
    total_bytes = sum(entry["bytes"] for entry in tensors.values())
    print(f"{arguments.model_directory}: {len(files)} files, {len(tensors)} tensors, {total_bytes / BYTES_PER_GIB:.2f} GiB of tensor data")

    for name, first_file, second_file, size in repeated:
        print(f"  repeated name, second copy not counted: {name} in {first_file} and {second_file} ({size / BYTES_PER_GIB:.2f} GiB)")

    by_dtype = {}
    for entry in tensors.values():
        by_dtype[entry["dtype"]] = by_dtype.get(entry["dtype"], 0) + entry["bytes"]
    print("by element type: " + ", ".join(f"{dtype} {size / BYTES_PER_GIB:.2f} GiB" for dtype, size in sorted(by_dtype.items(), key = lambda item: -item[1])))

    patterns = {}
    for name, entry in tensors.items():
        pattern = re.sub(r"\d+", "*", name)
        record = patterns.setdefault(pattern, {"bytes": 0, "count": 0, "dtypes": set(), "shape": entry["shape"]})
        record["bytes"] += entry["bytes"]
        record["count"] += 1
        record["dtypes"].add(entry["dtype"])
    patterns_printed = arguments.patterns if arguments.patterns > 0 else len(patterns)
    print(f"\nlargest name patterns ({min(patterns_printed, len(patterns))} of {len(patterns)}):")
    for pattern, record in sorted(patterns.items(), key = lambda item: -item[1]["bytes"])[:patterns_printed]:
        print(f"  {record['bytes'] / BYTES_PER_GIB:9.3f} GiB  {record['count']:6d} x  {'/'.join(sorted(record['dtypes'])):10s} "
              f"{str(record['shape']):26s} {pattern[:110]}")

    layer_prefix = arguments.layer_prefix
    if layer_prefix is None:
        candidates = {}
        for name in tensors:
            match = re.match(r"^((?:[A-Za-z_]+\.)*layers\.)\d+\.", name)
            if match:
                candidates[match.group(1)] = candidates.get(match.group(1), 0) + tensors[name]["bytes"]
        if not candidates:
            raise SystemExit("no '<prefix>layers.<n>.' names found; pass --layer-prefix")
        layer_prefix = max(candidates, key = candidates.get)
    layer_bytes = {}
    outside = {}
    table_bytes = 0
    for name, entry in tensors.items():
        match = re.match(re.escape(layer_prefix) + r"(\d+)\.", name)
        if any(marker in name for marker in TABLE_NAME_MARKERS):
            table_bytes += entry["bytes"]
        elif match:
            layer_bytes[int(match.group(1))] = layer_bytes.get(int(match.group(1)), 0) + entry["bytes"]
        else:
            group = re.sub(r"\d+", "*", name).split(".")[0]
            outside[group] = outside.get(group, 0) + entry["bytes"]
    layers_total = sum(layer_bytes.values())
    print(f"\nn-gram tables (not counted in the layers): {table_bytes / BYTES_PER_GIB:.2f} GiB")
    print(f"layers under '{layer_prefix}': {len(layer_bytes)}, {layers_total / BYTES_PER_GIB:.2f} GiB in all; "
          f"smallest {min(layer_bytes.values()) / BYTES_PER_GIB:.3f}, largest {max(layer_bytes.values()) / BYTES_PER_GIB:.3f} GiB")
    print("  per layer GiB: " + " ".join(f"{layer_bytes[index] / BYTES_PER_GIB:.2f}" for index in sorted(layer_bytes)))
    print("outside the layers: " + ", ".join(f"{group} {size / BYTES_PER_GIB:.2f} GiB" for group, size in sorted(outside.items(), key = lambda item: -item[1])))

    cards = parse_cards(arguments.cards)
    fixed_gib = POOL_GIB_PER_CARD_WORST_CASE + WORKSPACE_GIB_PER_CARD
    room_gib = sum(max(usable - fixed_gib, 0.0) for usable in cards)
    print(f"\nfit on {len(cards)} cards ({arguments.cards} usable GiB): {room_gib:.1f} GiB left for weights after "
          f"{fixed_gib:.3f} GiB per card of pools and workspace")
    print(f"  layers need {layers_total / BYTES_PER_GIB:.1f} GiB: " + (
        f"they fit with {room_gib - layers_total / BYTES_PER_GIB:.1f} GiB to spare (before the head, draft head and vision)"
        if layers_total / BYTES_PER_GIB <= room_gib else
        f"{layers_total / BYTES_PER_GIB - room_gib:.1f} GiB of them must live in RAM"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
