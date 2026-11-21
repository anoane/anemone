#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""r2_slice_table.py: the default slice counts of round 2, from the logs of a run of batch_i01_engine_linear_timing.sh,
by the rule of round 2 (RESEARCH.md 11.2 C item 6) as its reviews amended it, and what the reference order costs.

    r2_slice_table.py <log folder of the run> [<log folder of an earlier run> ...]

The rule, for each kernel family, shape and class of card (both CMP 170HX are one class: their medians are added):
  - a count's figure is the sum of its 1-row and 8-row medians (step mode, the gap of the run) over the class's cards;
  - the noise floor is the largest of: what baseline and baseline_again differ by in those medians, one step of the
    class's event clock per median, and FLOOR_SHARE of the least figure (the drift of an unchanged kernel between
    two runs: 1 to 2%);
  - the speed profile takes the SMALLEST count whose figure lies within the floor of the least (a tie goes to the
    smaller count); the precision profile takes 1 where its figure lies within the floor of the least, else the
    speed profile's count;
  - attn.wq_a and attn.wkv are one launch: both take the counts of attn.wq_a+attn.wkv.
It prints a TABLE line per row (every figure, the floor, the two picks), the rows of slice_defaults in linear.c, a
KNOB line per card (a layer's linear launches at the picked counts of the speed profile against 1 slice each, at 1
and 8 rows; engram.wkv apart; a launch whose median alone is above its whole step's counts as missing), an OUTLIER
line for each trunk case whose launch alone is slower than its whole step, AB lines (the steps that ran round 1's
engine and this one alternately), and SAME_PART lines: the picked counts' medians beside those of each earlier run
(same part, rows and variant).
Measurement tool: not part of the engine.
"""
import collections
import glob
import os
import re
import sys

FLOOR_SHARE = 0.02
CLOCK_STEP_US = {"sm_80": 1.024, "sm_120a": 0.1}   # the event clock's step in the medians each class prints
CLASS_OF_ARCHITECTURE = {"sm_80": 80, "sm_120a": 120}
STAMPED = (1, 2, 4, 8, 16)
BASELINE_SLICES = 4
TABLE_ROWS = (1, 8)
LOADS = 64                                          # inputs of one load: a matrix has at most inputs / 64 slices

# family, outputs, inputs, the log step, the part, --only, which count of slices:<A>x<B> varies (0: A, 1: B)
ROWS = (
    ("EXPERT_GATE_UP", 2304, 5120, "gate_up", "experts", "gate_up", 0),
    ("EXPERT_DOWN", 5120, 2304, "down", "experts", "down", 1),
    ("TRUNK", 1280, 5120, "trunk", "attn.wq_a+attn.wkv", "linear", 0),
    ("TRUNK", 512, 5120, "trunk", "attn.wq_a+attn.wkv", "linear", 0),
    ("TRUNK", 32768, 1280, "trunk", "attn.wq_b", "linear", 0),
    ("TRUNK", 4096, 1280, "trunk", "attn.indexer.wq_b", "linear", 0),
    ("TRUNK", 5120, 8192, "trunk", "attn.wo_b", "linear", 0),
    ("TRUNK", 25600, 6144, "trunk", "engram.wkv", "linear", 0),
    ("TRUNK", 2304, 5120, "shared_gate_up", "shared_expert", "gate_up", 0),
    ("TRUNK", 5120, 2304, "shared_down", "shared_expert", "down", 1),
    ("TRUNK_BFLOAT16", 8192, 4096, "bfloat16", "attn.wo_a_groups", "all", 0),
)
COMMENTS = {(1280, 5120): "attn.wq_a", (512, 5120): "attn.wkv", (32768, 1280): "attn.wq_b",
            (4096, 1280): "attn.indexer.wq_b", (5120, 8192): "attn.wo_b", (25600, 6144): "engram.wkv",
            (2304, 5120): "shared gate, up", (5120, 2304): "shared down", (8192, 4096): "attn.wo_a"}
KNOB_ROWS = [row for row in ROWS if not (row[0] == "TRUNK" and row[1:3] in ((512, 5120), (25600, 6144)))]
KNOB_ENGRAM = [row for row in ROWS if row[0] == "TRUNK" and row[1:3] == (25600, 6144)]
FIELD = re.compile(r"(\w+)=(\S+)")


def results(path):
    """The step-mode RESULT lines of a log as dictionaries, with part, only and rows of the case."""
    found = []
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8", errors="replace") as log:
        for line in log:
            if not line.startswith("RESULT "):
                continue
            fields = dict(FIELD.findall(line))
            if fields.get("mode") != "step":
                continue
            case = dict(item.split(":", 1) for item in fields["part"].split(",") if ":" in item)
            fields["case_part"], fields["case_only"] = case.get("part"), case.get("only")
            found.append(fields)
    return found


def variant_of(slices, axis):
    """The variant name of a count on the axis that varies; the baseline is 4 by 4."""
    if slices == BASELINE_SLICES:
        return "baseline"
    pair = [BASELINE_SLICES, BASELINE_SLICES]
    pair[axis] = slices
    return "slices:%dx%d" % tuple(pair)


def median(lines, part, only, rows, variant):
    for fields in lines:
        if (fields["case_part"], fields["case_only"], fields["rows"], fields["variant"]) == (part, only, str(rows),
                                                                                             variant):
            return float(fields["median_us"])
    return None


def cards_of(folder):
    """{card: architecture} of the run, from its pairs logs."""
    cards = {}
    for path in glob.glob(os.path.join(folder, "pairs_card_*.log")):
        for fields in results(path):
            cards[fields["card"]] = fields["arch"]
            break
    return cards


def picks(folder):
    cards = cards_of(folder)
    table = {}
    for architecture in sorted(set(cards.values())):
        class_cards = sorted(card for card, of in cards.items() if of == architecture)
        for family, outputs, inputs, step, part, only, axis in ROWS:
            logs = {card: results(os.path.join(folder, "%s_card_%s.log" % (step, card))) for card in class_cards}
            figures, noise, count = {}, 0.0, 0
            for slices in (s for s in STAMPED if s <= inputs // LOADS):
                total = 0.0
                for card in class_cards:
                    for rows in TABLE_ROWS:
                        value = median(logs[card], part, only, rows, variant_of(slices, axis))
                        total = None if value is None or total is None else total + value
                figures[slices] = total
            if any(value is None for value in figures.values()):
                print("MISSING %s %dx%d class %d: %s" % (family, outputs, inputs, CLASS_OF_ARCHITECTURE[architecture],
                                                         figures))
                continue
            for card in class_cards:
                for rows in TABLE_ROWS:
                    first = median(logs[card], part, only, rows, "baseline")
                    again = median(logs[card], part, only, rows, "baseline_again")
                    noise += abs(first - again)
                    count += 1
            least = min(figures.values())
            floor = max(noise, CLOCK_STEP_US[architecture] * count, FLOOR_SHARE * least)
            speed = min(s for s, value in figures.items() if value <= least + floor)
            precision = 1 if figures[1] <= least + floor else speed
            table[(family, outputs, inputs, architecture)] = (precision, speed)
            print("TABLE %s %dx%d class=%d figures_us=%s least=%.1f floor=%.1f (baseline noise %.1f) precision=%d "
                  "speed=%d" % (family, outputs, inputs, CLASS_OF_ARCHITECTURE[architecture],
                                ",".join("%d:%.1f" % item for item in sorted(figures.items())), least, floor, noise,
                                precision, speed))
    return cards, table


def print_rows(table):
    print("ROWS of slice_defaults (linear.c):")
    for family, outputs, inputs, *_ in ROWS:
        cells = []
        for architecture, name in (("sm_80", "SM_80"), ("sm_120a", "SM_120")):
            precision, speed = table.get((family, outputs, inputs, architecture), (0, 0))
            cells.append("{%s, %d, %d, %s, {%d, %d}}," % (family, outputs, inputs, name, precision, speed))
        comment = COMMENTS.get((outputs, inputs), "") if family != "EXPERT_GATE_UP" and family != "EXPERT_DOWN" else ""
        print("    %s %s%s" % (cells[0], cells[1], ("     /* %s */" % comment) if comment else ""))


def knob(folder, cards, table):
    for card, architecture in sorted(cards.items()):
        for name, rows_used in (("KNOB", KNOB_ROWS), ("KNOB_ENGRAM", KNOB_ENGRAM)):
            for rows in TABLE_ROWS:
                fixed = reference = 0.0
                missing = 0
                for family, outputs, inputs, step, part, only, axis in rows_used:
                    lines = results(os.path.join(folder, "%s_card_%s.log" % (step, card)))
                    speed = table.get((family, outputs, inputs, architecture), (0, 0))[1]
                    at_speed = median(lines, part, only, rows, variant_of(speed, axis))
                    at_one = median(lines, part, only, rows, variant_of(1, axis))
                    whole = [median(lines, part, "all", rows, variant_of(slices, axis)) for slices in (speed, 1)]
                    is_outlier = only == "linear" and any(
                        alone is not None and step_us is not None and alone > step_us
                        for alone, step_us in zip((at_speed, at_one), whole))
                    if at_speed is None or at_one is None or is_outlier:   # an outlier counts as missing
                        missing += 1
                        continue
                    fixed += at_speed
                    reference += at_one
                print("%s card=%s rows=%d launches=%d missing=%d table_us=%.1f reference_us=%.1f "
                      "extra_us_per_token=%.2f ratio=%.3f" % (name, card, rows, len(rows_used), missing, fixed,
                                                              reference, (reference - fixed) / rows,
                                                              reference / fixed if fixed else 0.0))


def outliers(folder, cards):
    for card in sorted(cards):
        lines = results(os.path.join(folder, "trunk_card_%s.log" % card))
        for fields in lines:
            if fields["case_only"] != "linear":
                continue
            whole = median(lines, fields["case_part"], "all", fields["rows"], fields["variant"])
            if whole is not None and float(fields["median_us"]) > whole:
                print("OUTLIER card=%s part=%s rows=%s variant=%s linear_us=%s all_us=%.1f" % (
                    card, fields["case_part"], fields["rows"], fields["variant"], fields["median_us"], whole))


def ab(folder, cards):
    turns = collections.defaultdict(list)
    for path in glob.glob(os.path.join(folder, "ab_round_*_card_*.log")):
        match = re.match(r"ab_round_(\d)_(\w+?)_(\d)_card_(\d+)\.log$", os.path.basename(path))
        if not match:
            continue
        engine, step, _, card = match.groups()
        for fields in results(path):
            key = (card, fields["case_part"], fields["case_only"], fields["rows"], fields["variant"])
            turns[key + (engine,)].append(float(fields["median_us"]))
    for key in sorted({key[:-1] for key in turns}):
        first, second = turns.get(key + ("1",), []), turns.get(key + ("2",), [])
        if first and second:
            one, two = sum(first) / len(first), sum(second) / len(second)
            print("AB card=%s part=%s only=%s rows=%s variant=%s round_1_us=%s round_2_us=%s ratio=%.3f" % (
                key[0], key[1], key[2], key[3], key[4], "/".join("%.1f" % value for value in first),
                "/".join("%.1f" % value for value in second), two / one))


def same_parts(folder, earlier, cards, table):
    seen = set()
    for family, outputs, inputs, step, part, only, axis in ROWS:
        if (step, part, only) in seen:
            continue
        seen.add((step, part, only))
        for card, architecture in sorted(cards.items()):
            speed = table.get((family, outputs, inputs, architecture), (0, 0))[1]
            for slices in sorted({speed, 1, BASELINE_SLICES}):
                variant = variant_of(slices, axis)
                figures = []
                for run in [folder] + earlier:
                    lines = results(os.path.join(run, "%s_card_%s.log" % (step, card)))
                    figures.append("/".join("%s" % median(lines, part, only, rows, variant) for rows in TABLE_ROWS))
                print("SAME_PART card=%s part=%s only=%s variant=%s rows_1/8_us this=%s earlier=%s" % (
                    card, part, only, variant, figures[0], " ".join(figures[1:])))


def main():
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print("usage: r2_slice_table.py <log folder of the run> [<log folder of an earlier run> ...]")
        return 2
    folder, earlier = sys.argv[1], sys.argv[2:]
    cards, table = picks(folder)
    print_rows(table)
    knob(folder, cards, table)
    outliers(folder, cards)
    ab(folder, cards)
    same_parts(folder, earlier, cards, table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
