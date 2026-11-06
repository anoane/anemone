#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How many routed experts consecutive rows share, from a forward capture of reference_oracle.py.

A step that runs several rows of one sequence (a verify step of drafted tokens, the tail of a prompt) reads each
routed expert once, however many of its rows go to it (shared_rows_expert_probe: the cost of a layer's routed
experts follows the experts read). So the cost of such a step is decided by how many DISTINCT experts its rows
route to. This tool counts them in a real pass: for every layer and every window of 1, 2, 4, 6 and 8 consecutive
positions of the captured sequence, the distinct experts among the routed ones, and what that costs on each card
by the probe's figures (microseconds = fixed + per expert * distinct).

    expert_sharing.py <forward capture folder> [--cost NAME:FIXED:PER_EXPERT ...]

The default costs are those of batch b08 on the owner's cards (CMP 170HX: 15 + 25.9 per expert; RTX PRO 6000:
19 + 11.7 per expert, in microseconds, from its 27-expert and 48-expert cases).
Measurement tool: not part of the engine.
"""
import argparse
import json
import os
import re
import sys

import numpy

ROUTED_IDS_NAME = re.compile(r"layer_(?P<layer>\d+)\.mlp\.routed_expert_ids")
WINDOWS = (1, 2, 4, 6, 8)
LARGEST_WINDOW = WINDOWS[-1]
DEFAULT_COSTS = ("170HX:15.0:25.9", "PRO:19.0:11.7")
LOW_PERCENTILE, MEDIAN_PERCENTILE, HIGH_PERCENTILE, PERCENT = 10, 50, 90, 100
MANIFEST_FILE_NAME, TENSORS_FILE_NAME = "manifest.json", "tensors.bin"


def percentile(sorted_values, percent):
    return sorted_values[min(len(sorted_values) * percent // PERCENT, len(sorted_values) - 1)]


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_folder")
    parser.add_argument("--cost", action = "append", default = None, metavar = "NAME:FIXED:PER_EXPERT")
    arguments = parser.parse_args()
    costs = []
    for text in arguments.cost or DEFAULT_COSTS:
        name, fixed, per_expert = text.split(":")
        costs.append((name, float(fixed), float(per_expert)))
    with open(os.path.join(arguments.capture_folder, MANIFEST_FILE_NAME), encoding = "utf8") as handle:
        records = {record["name"]: record for record in json.load(handle)["tensors"]}
    layers = sorted(int(match["layer"]) for match in map(ROUTED_IDS_NAME.fullmatch, records) if match)
    if not layers:
        print("the capture holds no routed expert ids: a forward capture is expected", file = sys.stderr)
        return 1
    routed = []
    for layer in layers:
        record = records[f"layer_{layer}.mlp.routed_expert_ids"]
        routed.append(numpy.fromfile(os.path.join(arguments.capture_folder, TENSORS_FILE_NAME),
                                     dtype = record["dtype"], count = int(numpy.prod(record["shape"])),
                                     offset = record["byte_offset"]).reshape(record["shape"]))
    routed = numpy.stack(routed)                                           # [layer, position, routed expert]
    positions, routed_per_row = routed.shape[1], routed.shape[2]
    print(f"{len(layers)} layers, {positions} positions, {routed_per_row} routed experts per position, expert ids "
          f"below {int(routed.max()) + 1}")
    distinct_of_largest = None
    for window in WINDOWS:
        if window > positions:
            break
        distinct = numpy.array([[len(numpy.unique(routed[layer_index, first:first + window]))
                                 for first in range(positions - window + 1)] for layer_index in range(len(layers))])
        ordered = numpy.sort(distinct.reshape(-1))
        low, median, high = (int(percentile(ordered, percent))
                             for percent in (LOW_PERCENTILE, MEDIAN_PERCENTILE, HIGH_PERCENTILE))
        shared = PERCENT * (1 - ordered.mean() / (routed_per_row * window))
        priced = "; ".join(f"{name}: {fixed + per_expert * median:.0f} us at the median, "
                           f"{fixed + per_expert * high:.0f} us at the {HIGH_PERCENTILE}th percentile"
                           for name, fixed, per_expert in costs)
        print(f"{window} consecutive rows: {median} distinct experts per layer at the median, {low} to {high} from "
              f"the {LOW_PERCENTILE}th to the {HIGH_PERCENTILE}th percentile, {ordered.mean():.1f} on average of "
              f"{routed_per_row * window} routed ({shared:.0f}% shared); {priced}")
        distinct_of_largest = distinct if window == LARGEST_WINDOW else distinct_of_largest
    if distinct_of_largest is not None:
        print(f"{LARGEST_WINDOW} rows, average distinct experts by layer: "
              + " ".join(f"{value:.0f}" for value in distinct_of_largest.mean(axis = 1)))
    used = sorted(len(numpy.unique(routed[layer_index])) for layer_index in range(len(layers)))
    print(f"distinct experts one layer uses over the whole sequence: {used[0]} at least, {used[len(used) // 2]} at "
          f"the median, {used[-1]} at most")
    return 0


if __name__ == "__main__":
    sys.exit(main())
