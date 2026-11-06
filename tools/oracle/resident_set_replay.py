#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How many routed experts a token finds off the cards under several choices of the experts a card holds.

    resident_set_replay.py <capture folder> [...] [--count-on <capture folder> ...] [--resident N] [--experts N]
                           [--token-ms T] [--break-count]

The cards hold N (250) of a layer's 384 experts and the processor computes a routed expert they do not hold. The
recorded routes of each forward capture are replayed; use is counted on its first half and every policy is scored
on its second half, as the processor experts of a token (all layers together), by layer, and in milliseconds at
0.21 and 0.40 ms an expert:
  - by index: experts 0 .. N - 1, what the loader holds today (placement.h);
  - by use on the earlier positions: the N most routed on the first half of the same sequence;
  - by use elsewhere: the N most routed in the --count-on captures, or without that option in the other captures
    given (all their positions): the placement a converter could make from other text;
  - slot cache: it starts as by index; a miss is computed by the processor and its expert uploaded for the positions
    after it, in place of the one of its layer routed longest ago. A link moves one expert in 11.1 ms, so the three
    cards take int(3 T / 11.1) uploads a token of T ms (--token-ms, 30); a token's misses are served from another
    layer each time. Also with every miss uploaded: the floor of such a cache;
  - the best fixed set for the scored positions themselves, counted where it is scored: a bound, not a policy.
Then the same over all captures, and what a uniform router would give. A median of the layers is the one
expert_sharing.py prints: of an even count the upper of the middle two.
Checked in every run: the misses of the placement by index are counted three times (through the code of the fixed
sets, through the slot cache with no upload allowed, and as the routed ids of N or more) and must agree position by
position. --break-count shows the check failing: one miss is added to the first count of every capture, exactly
the two other counts of that position must differ from it, and only then is the exit status 0; such a run is no
result, and its VERDICT lines say so.

Refused: what route_recall.py refuses of a capture folder; captures of different layer counts; a sequence of one
position; two captures with the same routes, scored or counted (a folder given twice or a copy of it: use counted
elsewhere would be counted on the positions it is scored on); --resident outside 1 .. --experts, --experts above
4096 or --token-ms that is no positive number (exit status 2).
What it does not show: whether a placement by use holds for a workload unlike the counted ones (the captures are
few: a prompt, held-out text, C sources); whether two captures share text (only the same routes are refused: use
counted on a capture that holds part of the scored one flatters "by use elsewhere"); the time of a token, which
changes T and with it the uploads; which card a layer is on (the three links are one budget here); several missed
experts computed at once; what an upload costs the step that runs beside it; drafted or sampled tokens: every
capture is a pass over given tokens.
Needs numpy. Exit status 0, 1 after a failure, 2 for a command line it cannot read.
Measurement tool: not part of the engine.
"""
import argparse
import math
import sys

import numpy

import route_recall                                        # the reader of a capture's routes, and its refusals

DEFAULT_RESIDENT_EXPERTS = 250                             # of a layer, on its card [RESEARCH.md 9.1]
COUNTED_SHARE = 0.5                                        # of a sequence's positions: use is counted on them
UPLOAD_MILLISECONDS = 11.1                                 # one expert to a card over four lanes [EVIDENCE.md]
CARD_LINKS = 3
DEFAULT_TOKEN_MILLISECONDS = 30.0                          # the checkpoint's format on three cards: 30 to 38 ms
PROCESSOR_EXPERT_MILLISECONDS = (0.21, 0.40)               # the fastest case and the 16-thread case of b04
NEVER_ROUTED = -1
RECOUNTS = 2                                               # of the check: the cache without uploads, ids of N or more
EXIT_CODE_FAILURE = 1


def use_counts(routes, experts_per_layer):
    """How often each expert of each layer is routed in routes [layer, position, routed]: int64 [layer, expert]."""
    return route_recall.routed_table(routes, experts_per_layer).sum(axis = 1)


def most_used(use, resident_count):
    """The resident_count most used experts of every layer, of equals the lower numbers: bool [layer, expert]."""
    resident = numpy.zeros(use.shape, dtype = bool)
    numpy.put_along_axis(resident, numpy.argsort(-use, axis = 1, kind = "stable")[:, :resident_count], True, axis = 1)
    return resident


def misses_of(resident, routes):
    """The routed experts of every position that a fixed set does not hold: int64 [layer, position]."""
    return (~numpy.take_along_axis(resident[:, None, :], routes, axis = -1)).sum(axis = -1)


def slot_cache(routes, resident, uploads_per_token):
    """The routes replayed in order against a set that changes: a missed expert takes the place of its layer's
    resident expert routed longest ago, for at most uploads_per_token misses of a position (None: all), taken
    from the layer position % layers on. Returns the misses, int64 [layer, position], and the uploads [position]."""
    layer_count, position_count, _ = routes.shape
    resident, layers = resident.copy(), numpy.arange(layer_count)[:, None]
    last_routed = numpy.full(resident.shape, NEVER_ROUTED, dtype = numpy.int64)
    misses, uploads = numpy.zeros((layer_count, position_count), dtype = numpy.int64), numpy.zeros(position_count)
    for position in range(position_count):
        routed = routes[:, position]
        missed = ~resident[layers, routed]
        misses[:, position] = missed.sum(axis = 1)
        last_routed[layers, routed] = position
        missed_layers, missed_places = numpy.nonzero(missed)
        served = numpy.argsort((missed_layers - position) % layer_count, kind = "stable")[:uploads_per_token]
        for layer, expert in zip(missed_layers[served].tolist(), routed[missed_layers[served], missed_places[served]]):
            longest_ago = numpy.where(resident[layer], last_routed[layer], position + 1).argmin()
            resident[layer, longest_ago], resident[layer, expert] = False, True
        uploads[position] = len(served)
    return misses, uploads


def replay(routes, elsewhere, resident_count, experts_per_layer, uploads_per_token, count_broken):
    """One capture: [(key, label, misses of the scored positions by layer, uploads of a scored position or None)]
    for every policy of the text at the top, the scored positions, and how many counts of the check differ.
    elsewhere: None, or where use was counted outside this capture and that use, int64 [layer, expert]."""
    first_scored = int(routes.shape[1] * COUNTED_SHARE)
    scored = routes[:, first_scored:]
    by_index = numpy.zeros((len(routes), experts_per_layer), dtype = bool)
    by_index[:, :resident_count] = True
    fixed = [("by_index", "by index, as the loader places them today", by_index),
             ("by_use_earlier", "by use on the earlier positions",
              most_used(use_counts(routes[:, :first_scored], experts_per_layer), resident_count))]
    if elsewhere is not None:
        fixed.append(("by_use_elsewhere", f"by use on {elsewhere[0]}", most_used(elsewhere[1], resident_count)))
    fixed.append(("best_fixed_set", "the best fixed set for the scored positions themselves (a bound)",
                  most_used(use_counts(scored, experts_per_layer), resident_count)))
    policies = [(key, label, misses_of(resident, scored), None) for key, label, resident in fixed]
    for key, label, allowed in (("slot_cache", f"slot cache, {uploads_per_token} uploads a token", uploads_per_token),
                                ("slot_cache_every_miss", "slot cache, every miss uploaded", None)):
        misses, uploads = slot_cache(routes, by_index, allowed)
        policies.insert(-1, (key, label, misses[:, first_scored:], uploads[first_scored:].mean()))
    _, _, misses_by_index, _ = policies[0]
    counted_first = misses_by_index.copy()
    counted_first[0, 0] += count_broken
    differing = int((counted_first != slot_cache(routes, by_index, 0)[0][:, first_scored:]).sum()
                    + (counted_first != (scored >= resident_count).sum(axis = -1)).sum())
    return [(key, label, misses.sum(axis = 1), uploads) for key, label, misses, uploads in policies], \
        scored.shape[1], differing


def policy_line(label, misses_by_layer, scored_positions, uploads):
    """What one policy costs: misses_by_layer [layer] are sums over scored_positions positions."""
    by_layer = misses_by_layer / scored_positions
    low, high = (by_layer.sum() * milliseconds for milliseconds in PROCESSOR_EXPERT_MILLISECONDS)
    links = "" if uploads is None else (f"; {uploads:.1f} uploads a token, {uploads * UPLOAD_MILLISECONDS:.0f} ms of "
                                        "the links")
    return (f"{label}: {by_layer.sum():.1f} processor experts a token ({low:.1f} to {high:.1f} ms); by layer "
            f"{by_layer.min():.2f} to {by_layer.max():.2f}, {route_recall.median_of(by_layer.tolist()):.2f} at the "
            f"median{links}")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_folders", nargs = "+")
    parser.add_argument("--count-on", nargs = "+", default = [], metavar = "FOLDER", help = "use elsewhere: there")
    parser.add_argument("--resident", type = int, default = DEFAULT_RESIDENT_EXPERTS, help = "experts a layer holds")
    parser.add_argument("--experts", type = int, default = route_recall.DEFAULT_EXPERTS_PER_LAYER, help = "of a layer")
    parser.add_argument("--token-ms", type = float, default = DEFAULT_TOKEN_MILLISECONDS, help = "the time of a token")
    parser.add_argument("--break-count", action = "store_true", help = "the check must report")
    arguments = parser.parse_args()
    if not 1 <= arguments.resident <= arguments.experts <= route_recall.MAXIMUM_EXPERTS_PER_LAYER \
            or not 0 < arguments.token_ms < math.inf:
        parser.error(f"--resident is from 1 to --experts, --experts at most {route_recall.MAXIMUM_EXPERTS_PER_LAYER}, "
                     "and --token-ms is a positive number")
    uploads_per_token = int(CARD_LINKS * arguments.token_ms / UPLOAD_MILLISECONDS)
    captures = []                                              # (folder, routes): the scored ones first
    try:
        for folder in arguments.capture_folders + arguments.count_on:
            routes = route_recall.read_routes(folder, arguments.experts)
            if route_recall.same_routes_as(routes, captures) is not None:
                raise route_recall.CaptureError(f"{folder} holds the same routes as "
                                                f"{route_recall.same_routes_as(routes, captures)}: use is counted "
                                                "and scored on different captures, each once")
            captures.append((folder, routes))
    except (route_recall.CaptureError, OSError) as failure:
        print(f"resident_set_replay FAILED: {failure}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    if len({(len(routes), routes.shape[-1]) for _, routes in captures}) != 1 \
            or min(routes.shape[1] for _, routes in captures) <= 1:
        print("resident_set_replay FAILED: the captures have not the same layers and routed experts, or one has a "
              "single position", file = sys.stderr)
        return EXIT_CODE_FAILURE
    uses = [use_counts(routes, arguments.experts) for _, routes in captures]
    scored_captures = captures[:len(arguments.capture_folders)]
    totals, differing, checked = {}, [], 0                     # differing: of each capture, the counts that differ
    for index, (folder, routes) in enumerate(scored_captures):
        elsewhere = None
        if arguments.count_on:
            elsewhere = (f"the {len(arguments.count_on)} captures of --count-on", sum(uses[len(scored_captures):]))
        elif len(captures) > 1:
            elsewhere = (f"the other {len(captures) - 1} captures", sum(uses) - uses[index])
        policies, scored_positions, differing_here = replay(routes, elsewhere, arguments.resident, arguments.experts,
                                                            uploads_per_token, arguments.break_count)
        routed_to = (uses[index] > 0).sum(axis = 1).tolist()
        print(f"{folder}: {len(routes)} layers, {routes.shape[1]} positions, the last {scored_positions} scored; "
              f"{arguments.resident} of a layer's {arguments.experts} experts are resident; over the sequence a layer "
              f"routes to {min(routed_to)} to {max(routed_to)} of them, {route_recall.median_of(routed_to)} at the "
              "median")
        for key, label, misses_by_layer, uploads in policies:
            print(f"{folder}: {policy_line(label, misses_by_layer, scored_positions, uploads)}")
            _, misses_so_far, positions_so_far, uploads_so_far = totals.get(key, (label, 0, 0, 0.0))
            totals[key] = (label, misses_so_far + misses_by_layer, positions_so_far + scored_positions,
                           uploads_so_far + (uploads or 0.0) * scored_positions)
        differing.append(differing_here)
        checked += RECOUNTS * len(routes) * scored_positions
    scope = f"captures={len(scored_captures)} count_on=" + (str(len(arguments.count_on)) if arguments.count_on
                                                             else "the_others" if len(captures) > 1 else "none")
    result_verdict = "not_a_result" if arguments.break_count else "measured"
    for key, (label, misses_by_layer, scored_positions, uploads) in totals.items():
        uploads = uploads / scored_positions if key.startswith("slot_cache") else None
        print(f"all {len(scored_captures)} captures: {policy_line(label, misses_by_layer, scored_positions, uploads)}")
        print(f"VERDICT figure=processor_experts_{key} {scope} value={misses_by_layer.sum() / scored_positions:.2f} "
              f"verdict={result_verdict}")
    layer_count, routed_per_position = captures[0][1].shape[0], captures[0][1].shape[-1]
    uniform = layer_count * routed_per_position * (arguments.experts - arguments.resident) / arguments.experts
    print(f"a uniform router: {uniform:.1f} processor experts a token")
    as_required = all(count == RECOUNTS * arguments.break_count for count in differing)
    verdict = "FAILED" if not as_required else "reported" if arguments.break_count else "ok"
    print(f"CHECK name=misses_by_index_counted_three_times differing={sum(differing)} of={checked} verdict={verdict}")
    return 0 if as_required else EXIT_CODE_FAILURE


if __name__ == "__main__":
    sys.exit(main())
