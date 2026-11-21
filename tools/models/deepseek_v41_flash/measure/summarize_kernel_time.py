#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One table from the JSON files kernel_time_trace.py writes: per card and per row count, the GPU kernel time and
the launches of one layer, beside the wall time of the step.

    summarize_kernel_time.py <forward_*.json> [...]

Measurement tool: not part of the engine.
"""
import json
import re
import sys

WHOLE_BLOCK_PREFIX = "whole block"
# Kernel names of the port grouped by what they compute; the first pattern that matches wins.
KERNEL_FAMILIES = (
    ("routed experts", r"exl3_moe_coop_[ab]_kernel<3|exl3_moe_kernel<3|moe_unswizzle|f16acc"),
    ("shared expert", r"exl3_moe_coop_[ab]_kernel<5"),
    ("trunk linears", r"exl3_gemv|exl3_mgemm|exl3_gemm|reconstruct_had|gemm|cutlass|xmma|nvjet"),
    ("attention core", r"dsa_|topk|index"),
    ("hyper-connections", r"hc_"),
    ("norms", r"rms_norm|norm"),
    ("router", r"routing|route"),
)
OTHER_FAMILY = "other"


def kernel_family(kernel_name):
    for family, pattern in KERNEL_FAMILIES:
        if re.search(pattern, kernel_name):
            return family
    return OTHER_FAMILY


def print_family_table(results, row_key):
    entry = results[row_key]
    print(f"  {row_key.split('_')[1]} row(s): kernel ms per layer by family (launches per layer)")
    for card_index in sorted(entry["kernels"], key = int):
        layer_count = layers_on_card(entry["layers"], f"cuda:{card_index}")
        if not layer_count:
            continue
        milliseconds, launches = {}, {}
        for kernel_name, (kernel_ms, count) in entry["kernels"][card_index]["by_name"].items():
            family = kernel_family(kernel_name)
            milliseconds[family] = milliseconds.get(family, 0.0) + kernel_ms
            launches[family] = launches.get(family, 0.0) + count
        cells = [f"{family} {value / layer_count:.3f} ({launches[family] / layer_count:.1f})"
                 for family, value in sorted(milliseconds.items(), key = lambda item: -item[1])]
        print(f"    card {card_index}: " + "; ".join(cells))


def layers_on_card(layer_table, device_name):
    return sum(count for kind, (_, count) in layer_table.get(device_name, {}).items() if kind.startswith(WHOLE_BLOCK_PREFIX))


def block_milliseconds(layer_table, device_name):
    return sum(total for kind, (total, _) in layer_table.get(device_name, {}).items() if kind.startswith(WHOLE_BLOCK_PREFIX))


def main():
    for path in sys.argv[1:]:
        with open(path) as handle:
            results = json.load(handle)
        print(f"\n{path.split('/')[-1]} ({results['part']}); cards: {', '.join(results['cards'])}")
        print("  rows  wall ms/step | per card: layers, kernel ms per layer, launches per layer, synchronised wall ms per layer")
        row_keys = sorted((key for key in results if key.startswith("rows_")), key = lambda key: int(key.split("_")[1]))
        for key in row_keys:
            entry = results[key]
            wall = entry.get("wall_ms_plain", entry.get("wall_ms_synchronised"))
            cells = []
            for card_index in sorted(entry["kernels"], key = int):
                device_name = f"cuda:{card_index}"
                layer_count = layers_on_card(entry["layers"], device_name)
                kernels = entry["kernels"][card_index]
                if layer_count:
                    cells.append(f"card {card_index}: {layer_count:2d} L, {kernels['kernel_ms'] / layer_count:7.3f} ms, "
                                 f"{kernels['launches'] / layer_count:5.1f} launches, wall {block_milliseconds(entry['layers'], device_name) / layer_count:7.3f}")
            print(f"  {key.split('_')[1]:>4s}  {wall:10.2f}   | " + " | ".join(cells))
            print(f"        all cards: kernels {entry['kernel_ms_per_step']:.2f} ms in {entry['launches_per_step']:.0f} launches per step")
        for key in row_keys:
            print_family_table(results, key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
