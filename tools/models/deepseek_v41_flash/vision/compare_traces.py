#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-only BF16 pass comparison; run under CUDA_VISIBLE_DEVICES=.

The reference directory is emitted by original_fixtures.py --trace-directory. Candidate files have the same
pass_000.bf16 naming and layout. Report the first bit difference separately from the first dtype-tolerance failure.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def values(path):
    words = np.fromfile(path, dtype="<u2")
    return words, (words.astype("<u4") << 16).view("<f4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--first-only", action="store_true")
    arguments = parser.parse_args()
    manifest = json.loads((arguments.reference / "manifest.json").read_text())
    first_changed, first_failed = None, None
    compared = 0
    for item in manifest["passes"]:
        if (arguments.candidate / item["file"]).stat().st_size != item["bytes"]:
            raise ValueError(f"incompatible candidate byte count: {item['file']}")
        reference_words, reference = values(arguments.reference / item["file"])
        candidate_words, candidate = values(arguments.candidate / item["file"])
        if reference.shape != candidate.shape or reference_words.nbytes != item["bytes"]:
            raise ValueError(f"incompatible pass shape: {item['file']}")
        changed = int(np.count_nonzero(candidate_words != reference_words))
        difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
        failed = int(np.count_nonzero((~np.isfinite(difference)) |
                                     (difference > np.abs(reference.astype(np.float64)) * 0.016 + 1e-5)))
        if changed and first_changed is None:
            first_changed = item["pass"]
        if failed and first_failed is None:
            first_failed = item["pass"]
        denominator = float(np.mean(reference.astype(np.float64) ** 2))
        record = {"pass": item["pass"], "operation": item.get("operation"),
                  "shape": item["shape"], "elements": int(reference.size),
                  "bitwise_differences": changed, "outside_bf16_tolerance": failed,
                  "maximum_absolute": float(difference.max()),
                  "relative_rms": float(np.sqrt(np.mean(difference ** 2) / denominator))
                                  if denominator else (0.0 if not changed else None)}
        print(json.dumps(record), flush=True)
        compared += 1
        if arguments.first_only and changed:
            break
    print(json.dumps({"first_bitwise_difference_pass": first_changed,
                      "first_dtype_tolerance_failure_pass": first_failed,
                      "passes_compared": compared, "complete": compared == len(manifest["passes"]),
                      "rtol": 0.016, "atol": 1e-5}), flush=True)
    if first_failed is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
