#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How many of a position's routed experts were routed at the positions just before it, from forward captures.

    route_recall.py <capture folder> [<capture folder> ...] [--experts N] [--break-count]

A guess of the next step's experts made before the router has run (a weight read ahead, an expert warmed in a
cache) can only use what is known by then. This counts the cheapest guess, the experts recently routed: for every
layer, every position p that has 8 before it and each window of 1, 2, 4 and 8 positions, how many of the experts
routed at p were routed at p - 1 .. p - window (the recall), and how many different experts those positions routed
to (what a guess of that recall has to hold or read). Per capture and over all of them one line for each window,
with the recall a uniform router would show for a set of that size, and the recall layer by layer. A median of
the layers is the one expert_sharing.py prints: of an even count the upper of the middle two.
Checked in every run: both counts (the hits, and the experts a window held) are made twice, from a table of who
was routed where by cumulative sums and from sets, position by position; the two must agree for every layer and
window. --break-count shows the check failing: one hit is added to the first count of every capture, exactly that
one count of each must be reported, and only then is the exit status 0; such a run is no result, and its VERDICT
lines say so.

Refused: a folder without a manifest that says complete, or without routed expert ids (reference_oracle.py records
them in every forward and linears capture); ids that are no whole numbers of two dimensions inside tensors.bin, of
another shape in another layer, outside 0 .. N - 1 (--experts, 384), or twice at one position; a layer recorded
twice, or layers that are not 0 .. n - 1 (a line "by layer" would name other layers than it counts); a sequence of
8 positions or fewer; captures of different layer counts; two captures with the same routes (a folder given twice,
or a copy of it: its positions would count twice); --experts outside 1 .. 4096 (exit status 2).
What it does not show: the recall of a guess from the router's SCORES, which no capture stores, or of a router run
before attention; whether a guess pays (a miss costs what the guess read); routes of drafted or sampled tokens:
every capture is a teacher-forced pass over given tokens. Needs numpy. Exit status 0, 1 after a failure, 2 for a
command line it cannot read. Measurement tool: not part of the engine.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy

MANIFEST_FILE_NAME, TENSORS_FILE_NAME = "manifest.json", "tensors.bin"
ROUTED_IDS_NAME = re.compile(r"layer_(?P<layer>0|[1-9][0-9]*)\.mlp\.routed_expert_ids")
ID_DTYPES = {"int64": "<i8", "int32": "<i4"}               # tensors.bin holds little-endian bytes
ROUTE_DIMENSIONS = 2                                       # [position, routed expert]
DEFAULT_EXPERTS_PER_LAYER = 384
MAXIMUM_EXPERTS_PER_LAYER = 4096                           # --experts sizes a table of every position by every expert
WINDOWS = (1, 2, 4, 8)
FIRST_SCORED_POSITION = WINDOWS[-1]                        # every window is scored over the same positions
PERCENT = 100.0
EXIT_CODE_FAILURE = 1


class CaptureError(Exception):
    """A capture folder this tool refuses; the message names the folder and the rule."""


def read_routes(capture_folder, experts_per_layer):
    """The routed expert ids of a capture, int64 [layer, position, routed], after every refusal of the text at the
    top that one folder can show."""
    manifest_path = os.path.join(capture_folder, MANIFEST_FILE_NAME)
    tensors_path = os.path.join(capture_folder, TENSORS_FILE_NAME)
    try:
        with open(manifest_path, encoding = "utf8") as handle:
            manifest = json.load(handle)
        tensors_byte_count = os.path.getsize(tensors_path)
    except (OSError, ValueError) as failure:
        raise CaptureError(f"{capture_folder}: {failure}") from failure
    if not isinstance(manifest, dict) or manifest.get("complete") is not True \
            or not isinstance(manifest.get("tensors"), list):
        raise CaptureError(f"{manifest_path} does not say \"complete\": true with a list of tensors")
    routes_of_layers = {}
    for record in manifest["tensors"]:
        name = ROUTED_IDS_NAME.fullmatch(str(record.get("name"))) if isinstance(record, dict) else None
        if name is None:
            continue
        shape, byte_offset = record.get("shape"), record.get("byte_offset")
        dtype = ID_DTYPES.get(str(record.get("dtype")))
        if dtype is None or not isinstance(shape, list) or len(shape) != ROUTE_DIMENSIONS \
                or not all(type(size) is int and size > 0 for size in shape) or type(byte_offset) is not int \
                or not 0 <= byte_offset <= tensors_byte_count - math.prod(shape) * numpy.dtype(dtype).itemsize:
            raise CaptureError(f"{manifest_path}: {record['name']} is no whole numbers of {ROUTE_DIMENSIONS} "
                               f"dimensions inside {TENSORS_FILE_NAME}")
        if int(name["layer"]) in routes_of_layers:
            raise CaptureError(f"{manifest_path}: {record['name']} is recorded twice")
        routes_of_layers[int(name["layer"])] = numpy.fromfile(tensors_path, dtype = dtype, count = math.prod(shape),
                                                              offset = byte_offset).reshape(shape)
    if not routes_of_layers or len({layer_routes.shape for layer_routes in routes_of_layers.values()}) != 1:
        raise CaptureError(f"{manifest_path} holds no routed expert ids, or of another shape in another layer")
    if sorted(routes_of_layers) != list(range(len(routes_of_layers))):
        raise CaptureError(f"{manifest_path}: its routed expert ids are not those of the layers 0 .. "
                           f"{len(routes_of_layers) - 1}")
    routes = numpy.stack([routes_of_layers[layer] for layer in sorted(routes_of_layers)]).astype(numpy.int64)
    if routes.min() < 0 or routes.max() >= experts_per_layer \
            or not (numpy.diff(numpy.sort(routes, axis = -1), axis = -1) > 0).all():
        raise CaptureError(f"{capture_folder}: an expert id is outside 0 .. {experts_per_layer - 1}, or routed twice "
                           "at one position")
    return routes


def routed_table(routes, experts_per_layer):
    """Who was routed where: bool [layer, position, expert]."""
    table = numpy.zeros(routes.shape[:-1] + (experts_per_layer,), dtype = bool)
    numpy.put_along_axis(table, routes, True, axis = -1)
    return table


def recall_counts(routes, experts_per_layer):
    """For each window: the hits of every layer, int64 [windows, layer] (routed experts of the scored positions that
    the window before the position routed too), and the experts the windows held, summed, [windows, layer]."""
    table = routed_table(routes, experts_per_layer)
    position_count = table.shape[1]
    routed_before = numpy.zeros((table.shape[0], position_count + 1, experts_per_layer), dtype = numpy.int32)
    numpy.cumsum(table, axis = 1, dtype = routed_before.dtype, out = routed_before[:, 1:])     # entry p: before p
    hits, held = [], []
    for window in WINDOWS:
        in_window = (routed_before[:, FIRST_SCORED_POSITION:position_count]
                     - routed_before[:, FIRST_SCORED_POSITION - window:position_count - window]) > 0
        hits.append((in_window & table[:, FIRST_SCORED_POSITION:]).reshape(len(table), -1).sum(axis = 1))
        held.append(in_window.reshape(len(table), -1).sum(axis = 1))
    return numpy.array(hits, dtype = numpy.int64), numpy.array(held, dtype = numpy.int64)


def counts_by_sets(routes):
    """Both counts of recall_counts a second time, position by position with sets: int64 [windows, layer] each."""
    shape = (len(WINDOWS), len(routes))
    hits, held = numpy.zeros(shape, dtype = numpy.int64), numpy.zeros(shape, dtype = numpy.int64)
    for layer, positions in enumerate(routes.tolist()):
        for window_index, window in enumerate(WINDOWS):
            layer_hits = layer_held = 0
            for position in range(FIRST_SCORED_POSITION, len(positions)):
                in_window = set().union(*positions[position - window:position])
                layer_hits, layer_held = (layer_hits + len(in_window.intersection(positions[position])),
                                          layer_held + len(in_window))
            hits[window_index, layer], held[window_index, layer] = layer_hits, layer_held
    return hits, held


def median_of(values):
    """The median as expert_sharing.py takes it: of an even count the upper of the middle two."""
    return sorted(values)[len(values) // 2]


def same_routes_as(routes, captures_before):
    """The folder of the first of (folder, routes) that holds these very routes, or None."""
    return next((folder for folder, earlier in captures_before if numpy.array_equal(earlier, routes)), None)


def print_recall(label, hits, held, scored_positions, routed_per_position, experts_per_layer, verdict_fields):
    """One line per window from the sums of recall_counts over scored_positions positions of every layer; with
    verdict_fields (what follows the value of a VERDICT line) also the recall layer by layer and that line."""
    for window, window_hits, window_held in zip(WINDOWS, hits, held):
        recalls = PERCENT * window_hits / (scored_positions * routed_per_position)
        experts_held = window_held.sum() / (scored_positions * len(window_hits))
        before = f"{label}: the {window} positions before" if window > 1 else f"{label}: the position before"
        print(f"{before}: {recalls.mean():.1f}% of a position's routed experts were routed there, among "
              f"{experts_held:.1f} different experts (a uniform router: "
              f"{PERCENT * experts_held / experts_per_layer:.1f}%); by layer {recalls.min():.1f}% to "
              f"{recalls.max():.1f}%, {median_of(recalls.tolist()):.1f}% at the median")
        if verdict_fields:
            print(f"{before}, percent by layer: " + " ".join(f"{recall:.0f}" for recall in recalls))
            print(f"VERDICT figure=route_recall_previous_{window} value={recalls.mean() / PERCENT:.4f} "
                  f"experts_held={experts_held:.1f} {verdict_fields}")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_folders", nargs = "+")
    parser.add_argument("--experts", type = int, default = DEFAULT_EXPERTS_PER_LAYER, help = "of one layer")
    parser.add_argument("--break-count", action = "store_true", help = "the check must report")
    arguments = parser.parse_args()
    if not 1 <= arguments.experts <= MAXIMUM_EXPERTS_PER_LAYER:
        parser.error(f"--experts counts the experts of one layer, 1 to {MAXIMUM_EXPERTS_PER_LAYER}")
    total_hits = total_held = total_positions = 0
    captures, differing = [], []                               # (folder, routes); of each: the counts that differ
    shapes = set()                                             # (layers, routed experts of a position) of the captures
    try:
        for folder in arguments.capture_folders:
            routes = read_routes(folder, arguments.experts)
            layer_count, position_count, routed_per_position = routes.shape
            shapes.add((layer_count, routed_per_position))
            if position_count <= FIRST_SCORED_POSITION or len(shapes) != 1:
                raise CaptureError(f"{folder}: {position_count} positions are not more than {FIRST_SCORED_POSITION}, "
                                   "or it has not the layers and routed experts of the captures before it")
            if same_routes_as(routes, captures) is not None:
                raise CaptureError(f"{folder} holds the same routes as {same_routes_as(routes, captures)}: a capture "
                                   "is counted once")
            captures.append((folder, routes))
            hits, held = recall_counts(routes, arguments.experts)
            hits[0, 0] += arguments.break_count
            hits_again, held_again = counts_by_sets(routes)
            differing.append(int((hits != hits_again).sum() + (held != held_again).sum()))
            scored_positions = position_count - FIRST_SCORED_POSITION
            print(f"{folder}: {layer_count} layers, {position_count} positions of which the last {scored_positions} "
                  f"are scored, {routed_per_position} routed of {arguments.experts} experts")
            print_recall(folder, hits, held, scored_positions, routed_per_position, arguments.experts, None)
            total_hits, total_held = total_hits + hits, total_held + held
            total_positions += scored_positions
    except (CaptureError, OSError) as failure:
        print(f"route_recall FAILED: {failure}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    result_verdict = "not_a_result" if arguments.break_count else "measured"
    print_recall(f"all {len(captures)} captures", total_hits, total_held, total_positions, routed_per_position,
                 arguments.experts, f"captures={len(captures)} verdict={result_verdict}")
    as_required = all(count == int(arguments.break_count) for count in differing)
    verdict = "FAILED" if not as_required else "reported" if arguments.break_count else "ok"
    print(f"CHECK name=recall_counted_twice differing={sum(differing)} "
          f"of={(total_hits.size + total_held.size) * len(captures)} verdict={verdict}")
    return 0 if as_required else EXIT_CODE_FAILURE


if __name__ == "__main__":
    sys.exit(main())
