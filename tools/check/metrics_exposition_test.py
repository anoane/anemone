#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strict finite Prometheus 0.0.4 subset emitted by the metrics endpoint."""
import math
import re
import sys
from pathlib import Path

NAME = r"[a-zA-Z_:][a-zA-Z_0-9:]*"
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z_0-9]*)="((?:[^"\\\n]|\\[\\"n])*)"(?:,|$)')
SAMPLE = re.compile(rf"({NAME})(?:\{{(.*)\}})? ([^ ]+)$")


def validate(text):
    if not text or not text.endswith("\n") or "\r" in text:
        raise ValueError("LF-terminated text required")
    types, helps, seen, histograms = {}, set(), set(), {}
    active = None
    closed = set()
    for line in text.splitlines():
        if line.startswith("# HELP "):
            _, _, name, help_text = line.split(" ", 3)
            if name in helps or not help_text or not re.fullmatch(NAME, name):
                raise ValueError("invalid/duplicate HELP")
            helps.add(name)
            continue
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ")
            if name in types or kind not in ("counter", "gauge", "histogram", "summary"):
                raise ValueError("invalid/duplicate TYPE")
            types[name] = kind
            continue
        match = SAMPLE.fullmatch(line)
        if not match:
            raise ValueError("invalid sample syntax")
        name, labels_text, number = match.groups()
        labels_text = labels_text or ""
        value = float(number)
        if not math.isfinite(value) or value < 0:
            raise ValueError("nonfinite/negative measurement")
        labels, offset = {}, 0
        while offset < len(labels_text):
            label = LABEL.match(labels_text, offset)
            if not label or label[1] in labels:
                raise ValueError("invalid/duplicate label")
            labels[label[1]] = label[2]
            offset = label.end()
        identity = (name, tuple(sorted(labels.items())))
        if identity in seen:
            raise ValueError("duplicate series")
        seen.add(identity)
        family = name
        suffix = ""
        if family not in types:
            for candidate in ("_bucket", "_sum", "_count"):
                if name.endswith(candidate) and name[:-len(candidate)] in types:
                    family, suffix = name[:-len(candidate)], candidate
                    break
        if family not in types or family not in helps:
            raise ValueError("missing preceding metadata")
        if active != family:
            if family in closed:
                raise ValueError("family samples must be contiguous")
            if active is not None:
                closed.add(active)
            active = family
        if types[family] == "summary":
            if suffix not in ("_sum", "_count") or "quantile" in labels:
                raise ValueError("summary requires finite sum/count only")
            continue
        if types[family] != "histogram":
            if suffix:
                raise ValueError("nonhistogram suffix")
            continue
        if suffix not in ("_bucket", "_sum", "_count"):
            raise ValueError("histogram sample suffix")
        bound = labels.pop("le", None)
        key = (family, tuple(sorted(labels.items())))
        histogram = histograms.setdefault(key, {"buckets": []})
        if suffix == "_bucket":
            if bound is None or (bound != "+Inf" and not math.isfinite(float(bound))):
                raise ValueError("invalid bucket boundary")
            if not value.is_integer():
                raise ValueError("nonintegral bucket count")
            histogram["buckets"].append((float(bound), value))
        else:
            if bound is not None:
                raise ValueError("le only belongs on buckets")
            histogram[suffix] = value
    for histogram in histograms.values():
        buckets = histogram["buckets"]
        if not buckets or buckets[-1][0] != math.inf or histogram.get("_count") != buckets[-1][1]:
            raise ValueError("+Inf must equal count")
        if "_sum" not in histogram or not histogram["_count"].is_integer():
            raise ValueError("missing sum or nonintegral count")
        if any(left[0] >= right[0] or left[1] > right[1] for left, right in zip(buckets, buckets[1:])):
            raise ValueError("unordered/noncumulative histogram")
    if not seen:
        raise ValueError("no samples")
    return len(types), len(seen)


def main():
    fixture = ('# HELP h Seconds.\n# TYPE h histogram\n'
               'h_bucket{model_name="x",le="1"} 1\n'
               'h_bucket{model_name="x",le="+Inf"} 2\n'
               'h_sum{model_name="x"} 3\nh_count{model_name="x"} 2\n')
    validate(fixture)
    for broken in (fixture[:-1], fixture.replace('} 3', '} NaN'),
                   fixture.replace('} 3', '} +Inf'), fixture.replace('h_count{model_name="x"} 2', 'h_count{model_name="x"} 1'),
                   fixture.replace('le="1"} 1', 'le="1"} 3'),
                   fixture + 'h_sum{model_name="x"} 3\n', fixture.replace('le="1"', 'le="1",le="2"')):
        try:
            validate(broken)
        except ValueError:
            continue
        raise ValueError("negative parser control unexpectedly passed")
    counts = validate(Path(sys.argv[1]).read_text())
    print(f"metrics exposition: {counts[0]} families, {counts[1]} unique finite series; seven negative controls rejected")


if __name__ == "__main__":
    main()
