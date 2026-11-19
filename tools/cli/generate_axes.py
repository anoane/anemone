#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the axis registry and its documentation from one reviewed data file."""
import argparse
import ast
import hashlib
import itertools
import json
import math
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools/cli/axes.json"
HEADER = ROOT / "source/cli_axes_data.h"
DOCUMENT = ROOT / "tools/cli/AXES.md"
CATALOG = ROOT / "tools/cli/KNOBS.md"
INVENTORY = ROOT / "tools/cli/inventory.json"  # commands.c's option table: each knob's parser range
AXIS_MARKER = "<!-- generated-axis-table -->"
AXIS_NAMES = ("category", "exactness", "concurrency", "focus")
STRATEGY_FIELDS = ("settings", "serve_settings", "generate_settings")
SPLITS = ("pp5", "tp4_pro", "tp4_exl3")
SET_CLASSES = ("same_bits", "draft_only", "lossy")
EXACT_SESSION_HOLD = ("The exact reference serves one session: use --concurrency=single or leave concurrency "
                      "unset.")
NO_SET_CLASS = "No knob set: the GPU split's base over its raw engine defaults."
BASE_FIELDS = {"route", "exclude", "raw_defaults", "evidence"}
# The user's choices (9 Nov): the context (default 1M) and the sessions (default as many as fit). No row pins
# either, with one exception: the exact rows, whose reference recipe keeps its own geometry (one session; PP5's at the
# official reference's context limit), which is part of what exact means.
SESSION_KNOBS = ("--context_tokens", "--session_slots")
SESSION_PINS_EXCEPTION = "exact"  # the exactness value whose rows may pin them
LITERAL_LIMIT = 4000
C_LINE_WIDTH = 112
NUMERICAL_CLASSES = {"same_bits", "draft_only", "lossy", "reference_arithmetic"}
METRIC_DENOMINATORS = {
    "mean_KL_nats": "kl_tokens", "p99_KL_nats": "kl_tokens",
    "PPL_ratio_vs_balanced": "ppl_tokens", "row_offset_removed_logit_RMS": "logit_values",
    "row_offset_removed_logit_absolute": "logit_values", "teacher_forced_top1_agreement": "top1_tokens",
    "complete_T0_response_agreement": "responses",
    "pinned_reference_tool_schema_cases_agreement": "pinned_cases",
}
AGREEMENT_COUNTS = {"teacher_forced_top1_agreement": "top1_matches",
                    "complete_T0_response_agreement": "response_matches",
                    "pinned_reference_tool_schema_cases_agreement": "pinned_matches"}
IDENTITY_FIELDS = ("binary_sha256", "settings_sha256", "input_sha256", "client_sha256",
                   "weights_sha256", "nonvarying_options_sha256")


def argument_tokens(value):
    """Accept legacy space-separated flags or an argv list with embedded spaces."""
    if isinstance(value, str):
        if any(character in value for character in "\0\r\n"):
            raise ValueError("legacy knob strings cannot contain NUL or newlines")
        arguments = value.split()
    elif isinstance(value, list):
        arguments = value
    else:
        raise ValueError("knob settings must be a string or argument list")
    for argument in arguments:
        if (not isinstance(argument, str) or
                any(character in argument for character in "\0\r\n") or
                not re.fullmatch(r"--[a-z][a-z0-9_]*=.+", argument)):
            raise ValueError("invalid knob argument; use one complete --key=value per argv item")
    return arguments


def strategy_assignments(strategy):
    return {argument for field in STRATEGY_FIELDS for argument in argument_tokens(strategy[field])}


def split_base(registry, split):
    """A GPU split's base (stage 3): its measured route's knobs without the excluded ones (the axes' own knobs
    and the measurement-protocol pins) and without each one equal to the split's raw engine default (a null default
    is automatic, never equal), so it holds only what the raw defaults lack. Every other route knob must declare its
    raw default. The settings and the serve additions, and the omitted (key, value) pins."""
    spec = registry["splits"][split]["base"]
    route = registry["strategies"][spec["route"]]
    defaults = spec["raw_defaults"]
    kept, omitted = {}, []
    for field in STRATEGY_FIELDS:
        kept[field] = []
        for argument in argument_tokens(route[field]):
            key = option_key(argument)
            if key == "--placement" or key in spec["exclude"]:
                continue
            if key not in defaults:
                raise ValueError(f"a base route knob needs its raw default or an exclusion: {split} {key}")
            if defaults[key] is not None and argument == f"{key}={defaults[key]}":
                omitted.append((key, defaults[key]))
            else:
                kept[field].append(argument)
    if kept["generate_settings"] and kept["generate_settings"] != kept["settings"]:
        raise ValueError(f"a base is one argv for both commands: {split}")
    return dict(settings=kept["settings"], serve_settings=kept["serve_settings"], generate_settings=None,
                omitted=list(dict.fromkeys(omitted)))


def base_classes(registry, split):
    """A base's numerical and error class: its route's, since the base runs the route's own
    knobs over the raw engine defaults, naming the arithmetic it changes from them (TP4_EXL3: relaxed numerics and the
    BF16 wire)."""
    route_name = registry["splits"][split]["base"]["route"]
    route = registry["strategies"][route_name]
    relaxed = [argument for argument in split_base(registry, split)["settings"] if argument in RELAXED_ARITHMETIC]
    arithmetic = (f" Its arithmetic is not the raw exact FP32 default: {' '.join(relaxed)} (relaxed numerics and the "
                  "BF16 wire)." if relaxed else "")
    return route["numerical_class"], (f"The GPU split's base, route {route_name} over the raw engine defaults: "
                                      f"{route['error_class']}{arithmetic}")


def split_set(registry, split, axis, value):
    """The knob set of one axis value on one GPU split, or None: its argv for both commands, its serve-only argv and
    its generate argv (None: the same as settings). A strategy-backed set is that recipe without its dropped knobs; the
    knobs it yields stay until composition leaves out each one another set of the row gives. A same_as set repeats
    another value's set of its axis, with its own evidence first."""
    entry = registry["splits"][split]["sets"].get(axis, {}).get(value)
    if entry is None:
        return None
    if "same_as" in entry:
        repeated = split_set(registry, split, axis, entry["same_as"])
        return dict(repeated, same_as=entry["same_as"],
                    evidence=f"{entry['evidence']} {axis}={entry['same_as']}'s set: {repeated['evidence']}")
    if "strategy" in entry:
        strategy = registry["strategies"][entry["strategy"]]
        keep = lambda field: [argument for argument in argument_tokens(strategy[field])
                              if option_key(argument) not in entry["drop"]]
        return dict(entry, settings=keep("settings"), serve_settings=keep("serve_settings"),
                    generate_settings=keep("generate_settings") or None, yields=entry.get("yields", []),
                    numerical_class=strategy["numerical_class"], error_class=strategy["error_class"],
                    raw_only=strategy.get("raw_only", False), lossy=strategy)
    return dict(entry, settings=argument_tokens(entry["settings"]),
                serve_settings=argument_tokens(entry.get("serve_settings", [])),
                generate_settings=argument_tokens(entry["generate_settings"]) if "generate_settings" in entry else None,
                raw_only=False, lossy=None)


def finite_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"nonfinite or missing quality value: {label}")
    return value


def positive_count(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"missing positive denominator: {label}")
    return value


def require_digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"missing SHA256 source identity: {label}")


def validate_metrics(record, limits=None, reference=False):
    metrics, denominators = record["metrics"], record["denominators"]
    for name in {*METRIC_DENOMINATORS.values(), "unique_requests"}:
        positive_count(denominators.get(name), name)
    if denominators["unique_requests"] > denominators["responses"]:
        raise ValueError("unique-request denominator exceeds responses")
    for name, denominator in METRIC_DENOMINATORS.items():
        actual_name = name.replace("vs_balanced", "vs_reference") if reference else name
        value = finite_number(metrics.get(actual_name), actual_name)
        if value < 0 or (name.startswith("PPL_") and value <= 0):
            raise ValueError(f"negative or zero quality metric: {actual_name}")
        if name in AGREEMENT_COUNTS:
            matches = record.get("agreement_counts", {}).get(AGREEMENT_COUNTS[name])
            count = denominators[denominator]
            if type(matches) is not int or not 0 <= matches <= count or not 0 <= value <= 1:
                raise ValueError(f"invalid agreement count: {actual_name}")
            if not math.isclose(value, matches / count, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"agreement does not match its denominator: {actual_name}")
    for name in ("raw_logit_RMS", "raw_logit_absolute", "worst_KL_nats"):
        if finite_number(metrics.get(name), name) < 0:
            raise ValueError(f"negative quality metric: {name}")
    if (metrics["raw_logit_RMS"] > metrics["raw_logit_absolute"] or
            metrics["row_offset_removed_logit_RMS"] > metrics["row_offset_removed_logit_absolute"] or
            metrics["worst_KL_nats"] < max(metrics["mean_KL_nats"], metrics["p99_KL_nats"])):
        raise ValueError("inconsistent quality extrema")
    if limits is not None:
        for limit_name, bound in limits.items():
            name, direction = limit_name.rsplit("_", 1)
            value = metrics[name]
            if (direction == "max" and value > bound) or (direction == "min" and value < bound):
                raise ValueError(f"quality limit exceeded: {limit_name}")


def validate_receipt(receipt, knob, budget, allow_synthetic=False):
    if type(receipt.get("synthetic")) is not bool or (receipt["synthetic"] and not allow_synthetic):
        raise ValueError("synthetic or unclassified receipt cannot qualify an axis strategy")
    if receipt.get("knob") != knob or not receipt.get("id") or not receipt.get("source"):
        raise ValueError("quality receipt has no knob/source identity")
    require_digest(receipt.get("source_sha256"), "quality source")
    fixture = receipt.get("fixture", {})
    if not fixture.get("id") or fixture.get("held_out") is not True or fixture.get("smoke_only") is not False:
        raise ValueError("quality fixture must be held out and not a smoke probe")
    require_digest(fixture.get("sha256"), "fixture")
    if receipt.get("stratum") not in budget["required_strata"] or not receipt.get("scope") or not receipt.get("uncertainty"):
        raise ValueError("quality receipt lacks a predeclared stratum/scope/uncertainty")
    baseline, candidate = receipt.get("baseline", {}), receipt.get("candidate", {})
    for identity in (baseline, candidate):
        for field in IDENTITY_FIELDS:
            require_digest(identity.get(field), field)
    for field in IDENTITY_FIELDS:
        if field != "settings_sha256" and baseline[field] != candidate[field]:
            raise ValueError(f"unpaired numerical source: {field}")
    if baseline["settings_sha256"] == candidate["settings_sha256"]:
        raise ValueError("lossy comparison has identical setting identities")
    validate_metrics(receipt, budget["limits"])
    workload = receipt.get("whole_workload", {})
    if not workload.get("source"):
        raise ValueError("missing whole-workload source")
    require_digest(workload.get("source_sha256"), "whole-workload source")
    for side, identity in (("baseline", baseline), ("candidate", candidate)):
        for field in ("binary_sha256", "settings_sha256"):
            if workload.get(side + "_" + field) != identity[field]:
                raise ValueError("whole-workload result uses a different numerical configuration")
    for field in ("input_sha256", "client_sha256", "weights_sha256"):
        require_digest(workload.get(field), "whole-workload " + field)
    if workload["weights_sha256"] != baseline["weights_sha256"]:
        raise ValueError("whole-workload result uses different weights")
    for field in ("requests", "unique_requests", "baseline_completion_tokens", "candidate_completion_tokens"):
        positive_count(workload.get(field), field)
    if workload["unique_requests"] > workload["requests"]:
        raise ValueError("whole-workload unique requests exceed requests")
    for field in ("baseline_seconds", "candidate_seconds"):
        if finite_number(workload.get(field), field) <= 0:
            raise ValueError("whole-workload time must be positive")
    baseline_rate = workload["baseline_completion_tokens"] / workload["baseline_seconds"]
    candidate_rate = workload["candidate_completion_tokens"] / workload["candidate_seconds"]
    if candidate_rate <= baseline_rate:
        raise ValueError("lossy knob has no measured whole-workload gain")
    reference = receipt.get("reference")
    if reference is None:
        if budget["reference_available"] or not receipt.get("reference_unavailable_reason"):
            raise ValueError("missing applicable reference comparison or reason")
    else:
        if not reference.get("source") or not reference.get("scope"):
            raise ValueError("missing reference scope/source")
        require_digest(reference.get("source_sha256"), "reference source")
        for field in IDENTITY_FIELDS:
            require_digest(reference.get("identity", {}).get(field), "reference " + field)
        for field in ("input_sha256", "client_sha256"):
            if reference["identity"][field] != baseline[field]:
                raise ValueError("unpaired reference inputs/client")
        validate_metrics(reference, reference=True)


def validate_lossy_strategy(name, strategy, budget, allow_synthetic=False):
    if not budget["admitted"]:
        raise ValueError(f"lossy strategy requires an admitted budget: {name}")
    knobs, receipts = strategy["lossy_knobs"], strategy["quality_receipts"]
    if not knobs or set(receipts) != set(knobs):
        raise ValueError(f"missing per-lossy-knob receipts: {name}")
    for knob in knobs:
        rows = receipts[knob]
        if not isinstance(rows, list) or len(rows) != len(budget["required_strata"]):
            raise ValueError(f"missing paired fixture/strata receipts: {name}/{knob}")
        if {row.get("stratum") for row in rows} != set(budget["required_strata"]):
            raise ValueError(f"incomplete quality strata: {name}/{knob}")
        if len({row.get("id") for row in rows}) != len(rows):
            raise ValueError(f"quality receipt identities are repeated: {name}/{knob}")
        for receipt in rows:
            validate_receipt(receipt, knob, budget, allow_synthetic)


def validate_numerical_contracts(registry, allow_synthetic=False):
    budget = registry["quality_budget"]
    if type(budget.get("admitted")) is not bool or type(budget.get("reference_available")) is not bool:
        raise ValueError("quality budget needs explicit admitted/reference_available booleans")
    strata = budget.get("required_strata")
    if (not isinstance(strata, list) or not strata or
            any(not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_/-]+", value) for value in strata) or
            len(set(strata)) != len(strata)):
        raise ValueError("quality strata must be predeclared and distinct")
    if any(len({value for value in strata if value.startswith(category + "/")}) < 2
           for category in ("code", "prose", "structured")):
        raise ValueError("quality strata need separate contexts for code, prose and structured")
    limits = budget.get("limits", {})
    if not isinstance(limits, dict) or {key.rsplit("_", 1)[0] for key in limits} != set(METRIC_DENOMINATORS):
        raise ValueError("quality limits must cover every current metric")
    for name, value in limits.items():
        metric, direction = name.rsplit("_", 1)
        expected = "min" if metric in AGREEMENT_COUNTS else "max"
        if direction != expected or finite_number(value, name) < 0 or (expected == "min" and value > 1):
            raise ValueError(f"invalid quality bound: {name}")
    fast = next(tier for tier in registry["tiers"] if tier["name"] == "fast")
    if not fast["hold"] and not budget["admitted"]:
        raise ValueError("cannot clear fast hold without an admitted budget")
    declared_lossy = set()
    for name, strategy in registry["strategies"].items():
        numerical_class = strategy.get("numerical_class")
        knobs, receipts = strategy.get("lossy_knobs"), strategy.get("quality_receipts")
        if numerical_class not in NUMERICAL_CLASSES or not isinstance(knobs, list) or not isinstance(receipts, dict):
            raise ValueError(f"missing numerical/receipt contract: {name}")
        if type(strategy.get("raw_only", False)) is not bool:
            raise ValueError(f"raw_only must be a boolean: {name}")
        if (any(not isinstance(knob, str) or not knob.startswith("--") or "=" not in knob for knob in knobs) or
                len(set(knobs)) != len(knobs)):
            raise ValueError(f"lossy knobs need exact distinct assignments: {name}")
        assignments = strategy_assignments(strategy)
        if not set(knobs) <= assignments:
            raise ValueError(f"declared lossy knob is not selected by its strategy: {name}")
        if numerical_class == "lossy":
            validate_lossy_strategy(name, strategy, budget, allow_synthetic)
        elif knobs or receipts:
            raise ValueError(f"non-lossy class declares lossy controls/receipts: {name}")
        declared_lossy.update(knobs)
    for name, strategy in registry["strategies"].items():
        assignments = strategy_assignments(strategy)
        if assignments & declared_lossy != set(strategy["lossy_knobs"]):
            raise ValueError(f"undeclared lossy assignment: {name}")
    for split, spec in registry["splits"].items():
        for axis, values in spec["sets"].items():
            for value in values:
                entry = split_set(registry, split, axis, value)
                if entry["numerical_class"] not in SET_CLASSES:
                    raise ValueError(f"a knob set needs a same-bits, draft-only or lossy class: {split} {axis}={value}")
                if entry["numerical_class"] == "lossy" and (axis, value) != ("exactness", "fast"):
                    raise ValueError(f"lossy strategy in {axis}={value} mapping")
                if entry["raw_only"]:
                    raise ValueError("raw-only candidate cannot enter an active tuple")


def validate_choice_contract(strategy, exactness, hold):
    numerical_class = strategy["numerical_class"]
    if numerical_class == "lossy" and exactness != "fast":
        raise ValueError(f"lossy strategy in {exactness} mapping")
    if exactness == "balanced" and numerical_class == "reference_arithmetic":
        raise ValueError("balanced mapping selects reference arithmetic")
    if not hold and strategy.get("raw_only", False):
        raise ValueError("raw-only candidate cannot enter an active tuple")
    if not hold and exactness == "exact" and numerical_class != "reference_arithmetic":
        raise ValueError("exact mapping needs reference arithmetic")


def lossy_evidence(strategy, budget):
    if strategy["numerical_class"] != "lossy":
        return ""
    pieces = []
    for knob in strategy["lossy_knobs"]:
        rows = strategy["quality_receipts"][knob]
        synthetic = "SYNTHETIC FIXTURE; parser semantics only. " if any(row["synthetic"] for row in rows) else ""
        pieces.append(synthetic + f"Lossy {knob}: worst measured figures over {len(rows)} declared strata;")
        for limit_name in budget["limits"]:
            metric, direction = limit_name.rsplit("_", 1)
            row = (min if direction == "min" else max)(rows, key=lambda item: item["metrics"][metric])
            count = row["denominators"][METRIC_DENOMINATORS[metric]]
            pieces.append(f"{metric}={row['metrics'][metric]:.8g} ({row['stratum']}, n={count});")
        for metric in ("raw_logit_RMS", "raw_logit_absolute", "worst_KL_nats"):
            row = max(rows, key=lambda item: item["metrics"][metric])
            pieces.append(f"{metric}={row['metrics'][metric]:.8g} ({row['stratum']});")
        unique = [row["denominators"]["unique_requests"] for row in rows]
        pieces.append(f"unique-request denominators {min(unique)}..{max(unique)} per stratum;")
        gains = [(row["whole_workload"]["candidate_completion_tokens"] / row["whole_workload"]["candidate_seconds"]) /
                 (row["whole_workload"]["baseline_completion_tokens"] / row["whole_workload"]["baseline_seconds"]) - 1
                 for row in rows]
        pieces.append(f"minimum observed whole-workload tok/s gain {100 * min(gains):.6g}%;")
        references = [(row["stratum"], row["reference"]) for row in rows if row["reference"] is not None]
        if references:
            pieces.append(f"reference comparisons available for {len(references)}/{len(rows)} strata;")
            for metric in (*METRIC_DENOMINATORS, "raw_logit_RMS", "raw_logit_absolute", "worst_KL_nats"):
                actual = metric.replace("vs_balanced", "vs_reference")
                choose = min if metric in AGREEMENT_COUNTS else max
                stratum, reference = choose(references, key=lambda item: item[1]["metrics"][actual])
                denominator = METRIC_DENOMINATORS.get(metric, "kl_tokens" if metric == "worst_KL_nats" else "logit_values")
                count = reference["denominators"][denominator]
                pieces.append(f"reference {actual}={reference['metrics'][actual]:.8g} ({stratum}, n={count});")
        else:
            pieces.append("reference comparison unavailable: " + rows[0]["reference_unavailable_reason"] + ";")
        sources = sorted({row["source"] + "@" + row["source_sha256"][:16] for row in rows})
        pieces.append("receipt sources " + ", ".join(sources) + ".")
    return " ".join(pieces)


def option_key(argument):
    name = argument.split("=", 1)[0]
    return {"--draft": "--speculative", "--speculative_depth": "--draft_positions"}.get(name, name)


ROUTE_PARAMETERS = (("GPU split (`--placement`)", "--placement"), ("Pipeline depth (`--pipeline_steps`)", "--pipeline_steps"),
                    ("Serving slots (`--session_slots`)", "--session_slots"))


def route_parameters(strategy):
    """A route's three parameters, derived from its settings and never stored: the GPU split (--placement, how the GPUs
    are divided), the pipeline depth (--pipeline_steps, the steps in flight) and the serving slots (--session_slots).
    Each is the serve value (settings, then serve_settings); a generate value that differs follows in parentheses."""
    serve = argument_tokens(strategy["settings"]) + argument_tokens(strategy["serve_settings"])
    generate = argument_tokens(strategy["generate_settings"] or strategy["settings"])

    def value(arguments, key):
        values = [argument.split("=", 1)[1] for argument in arguments if option_key(argument) == key]
        return (values[-1].upper() if key == "--placement" else values[-1]) if values else None

    columns = []
    for _, key in ROUTE_PARAMETERS:
        served, generated = value(serve, key), value(generate, key)
        shown = served if served is not None else "default"
        columns.append(shown + (f" (generate {generated})" if generated is not None and generated != served else ""))
    return columns


def placements_of(recipe):
    """Every --placement a recipe gives, in either command's argv."""
    return {argument for field in STRATEGY_FIELDS for argument in argument_tokens(recipe[field])
            if option_key(argument) == "--placement"}


def validate_splits(registry, domains):
    """One table per GPU split: a knob set per axis value, each knob owned by one axis, holds with reasons."""
    if tuple(registry.get("splits", {})) != SPLITS:
        raise ValueError("GPU split tables must be pp5/tp4_pro/tp4_exl3")
    strategies = registry["strategies"]
    for split, spec in registry["splits"].items():
        base = spec.get("base", {})
        if (set(spec) != {"weight_family", "base", "reference", "sets", "holds"} or
                spec["weight_family"] not in registry["weight_families"] or set(base) != BASE_FIELDS or
                base["route"] not in strategies or placements_of(strategies[base["route"]]) != {f"--placement={split}"}
                or not isinstance(base["evidence"], str) or not base["evidence"].strip() or
                not isinstance(base["exclude"], list) or not isinstance(base["raw_defaults"], dict) or
                any(value is not None and not isinstance(value, str) for value in base["raw_defaults"].values())):
            raise ValueError(f"GPU split table needs a weight family, a base (a route of its own split, exclusions, "
                             f"raw defaults, evidence), reference, sets and holds: {split}")
        split_base(registry, split)
        holds = spec["holds"]
        if any(axis not in domains or any(value not in domains[axis] or value == "balanced" or
                                          not isinstance(text, str) or not text.strip()
                                          for value, text in values.items())
               for axis, values in holds.items()):
            raise ValueError(f"a hold names a non-balanced axis value and its reason: {split}")
        reference = spec["reference"]
        if reference is None:
            if not holds.get("exactness", {}).get("exact"):
                raise ValueError(f"a GPU split without a reference recipe must hold exact: {split}")
        elif (reference not in strategies or strategies[reference]["numerical_class"] != "reference_arithmetic" or
              f"--placement={split}" not in argument_tokens(strategies[reference]["settings"]) or
              placements_of(strategies[reference]) - {f"--placement={split}"}):
            raise ValueError(f"original reference arithmetic must be a recipe of its own GPU split: {split}")
        elif not EXACT_ARITHMETIC <= set(argument_tokens(strategies[reference]["settings"])) or \
                set(argument_tokens(strategies[reference]["settings"])) & RELAXED_ARITHMETIC:
            # exact never inherits a base's relaxed numerics or BF16 wire (TP4_EXL3's no-axis default runs them)
            raise ValueError(f"an exact recipe gives --numerics=exact and the FP32 wire: {split}")
        owners, yielded = {}, []
        for axis, values in spec["sets"].items():
            if axis not in domains or axis == "exactness" and set(values) - {"fast"}:
                raise ValueError(f"knob sets name an axis value; exactness carries only a fast set: {split}")
            for value, entry in values.items():
                label = f"{split} {axis}={value}"
                if value not in domains[axis] or "same_as" in entry and (
                        value != "balanced" or set(entry) != {"same_as", "evidence"} or
                        entry["same_as"] not in values or "same_as" in values[entry["same_as"]]):
                    raise ValueError(f"only a balanced set may repeat another measured value's set: {label}")
                # Unbounded sessions: the owner's balanced rule (1M context, as many sessions as fit) is the raw
                # defaults, so a balanced set inserts nothing of its own.
                if value == "balanced" and "same_as" not in entry:
                    raise ValueError(f"a balanced set stays empty or repeats a measured value's set: {label}")
                slots = entry.get("applies_with_session_slots")
                if slots is not None and (axis != "concurrency" or isinstance(slots, bool) or
                                          not isinstance(slots, int) or slots < 1):
                    raise ValueError(f"only a concurrency set applies at a serving slot count, a positive count: "
                                     f"{label}")
                if holds.get(axis, {}).get(value):
                    raise ValueError(f"a held axis value cannot carry a knob set: {label}")
                if not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip():
                    raise ValueError(f"a knob set needs its evidence pointer: {label}")
                if "strategy" in entry:
                    if (set(entry) - {"strategy", "drop", "yields", "evidence", "requires", "applies_with",
                                      "applies_with_session_slots"} or
                            entry["strategy"] not in strategies or not isinstance(entry.get("drop"), list) or
                            not isinstance(entry.get("yields", []), list)):
                        raise ValueError(f"a recipe set names a strategy and the knobs it drops: {label}")
                    recipe = strategies[entry["strategy"]]
                    keys = {option_key(argument) for field in STRATEGY_FIELDS
                            for argument in argument_tokens(recipe[field])}
                    if not set(entry["drop"]) <= keys or placements_of(recipe) - {f"--placement={split}"}:
                        raise ValueError(f"a recipe set must belong to its GPU split and drop only its own knobs: {label}")
                    if not set(entry.get("yields", [])) <= keys - set(entry["drop"]):
                        raise ValueError(f"a recipe set yields only knobs it keeps: {label}")
                    yielded += [(key, axis, label) for key in entry.get("yields", [])]
                elif "same_as" not in entry and (  # a repeated set is checked as its own value
                        set(entry) - {"settings", "serve_settings", "generate_settings", "numerical_class",
                                      "error_class", "evidence", "requires", "applies_with",
                                      "applies_with_session_slots"} or
                        not isinstance(entry.get("error_class"), str) or
                        any(option_key(argument) == "--placement" for argument in argument_tokens(entry["settings"]))):
                    raise ValueError(f"an inline set needs settings, a class and no placement: {label}")
                for other, allowed in [*entry.get("requires", {}).items(), *entry.get("applies_with", {}).items()]:
                    if (other not in domains or other == axis or not allowed or
                            any(item not in domains[other] for item in allowed)):
                        raise ValueError(f"a set may require, or apply only with, values of another axis: {label}")
                resolved = split_set(registry, split, axis, value)
                pinned = [key for field in STRATEGY_FIELDS for key in map(option_key, resolved[field] or [])
                          if key in SESSION_KNOBS]
                if pinned:
                    raise ValueError(f"an axis set pins no context or session count (the user's): {label} {pinned[0]}")
                # Each command's argv: serve's settings and serve additions, generate's own settings.
                for arguments in (resolved["settings"] + resolved["serve_settings"],
                                  resolved["generate_settings"] or resolved["settings"]):
                    keys = [option_key(argument) for argument in arguments]
                    if len(set(keys)) != len(keys):
                        raise ValueError(f"duplicate knob destination: {label}")
                    for key in keys:
                        if key not in resolved.get("yields", []) and owners.setdefault(key, axis) != axis:
                            raise ValueError(f"{split}: {key} is set by both {owners[key]} and {axis}")
        # A yielded knob is set by its recipe and exactly one other axis (two other axes already
        # refuse).
        for key, axis, label in yielded:
            if owners.get(key, axis) == axis:
                raise ValueError(f"a recipe set yields a knob only to the one other axis that sets it: {label} {key}")


BOUND_FIELDS = {"cap": {"kind", "bound", "bounded", "offset", "rule"},
                "floor": {"kind", "bound", "value", "under", "bounded", "least", "default", "rule"}}


def validate_bounds(registry):
    knob = lambda key: isinstance(key, str) and re.fullmatch(r"--[a-z_]+", key) is not None
    for bound in registry.get("cross_axis_bounds", []):
        kind = bound.get("kind")
        if (set(bound) != BOUND_FIELDS.get(kind) or not knob(bound["bound"]) or not bound["bounded"] or
                not all(knob(key) for key in bound["bounded"]) or not isinstance(bound["rule"], str) or
                not bound["rule"].strip() or (kind == "cap" and type(bound["offset"]) is not int) or
                (kind == "floor" and (not knob(bound["under"]) or not isinstance(bound["value"], str) or
                                      type(bound["least"]) is not int or type(bound["default"]) is not int or
                                      bound["default"] >= bound["least"]))):
            raise ValueError("a cross-axis bound is a cap (bounding knob, bounded knobs, offset, rule) or a floor "
                             "(bounding knob and value, the knob it applies under, bounded knobs, least, default, "
                             "rule)")
    # startup reads a bounding (or under) knob after its own bounds, one level and without recursion
    # (startup_axes.c bounding_final_value): a bound on such a knob must read knobs no bound names
    bounds = registry.get("cross_axis_bounds", [])
    bounded = lambda key: [bound for bound in bounds if key in bound["bounded"]]
    reads = lambda bound: [bound["bound"], *([bound["under"]] if bound["kind"] == "floor" else [])]
    for bound in bounds:
        for knob_name in reads(bound):
            if any(bounded(read) for inner in bounded(knob_name) for read in reads(inner)):
                raise ValueError(f"a chain of cross-axis bounds longer than one through {knob_name}")
    capped = [key for bound in bounds if bound["kind"] == "cap" for key in bound["bounded"]]
    if len(capped) != len(set(capped)):
        raise ValueError("a knob two caps bound")


def parser_least(key):
    """The least value commands.c's parser takes for a knob (inventory.json), as a number: a named constant is read from
    its #define in include/anemone (the inventory spells it with the ANEMONE_ prefix)."""
    entry = next((entry for entry in json.loads(INVENTORY.read_text()) if entry["name"] == key[len("--"):]), None)
    if entry is None:
        raise ValueError(f"a bounded knob the parser does not take: {key}")
    least = entry["least"]
    if least.isdigit():
        return int(least)
    name = least.removeprefix("ANEMONE_")
    for header in sorted((ROOT / "include/anemone").glob("*.h")):
        found = re.search(rf"^#define {re.escape(name)} (\d+)u?\b", header.read_text(), re.M)
        if found:
            return int(found.group(1))
    raise ValueError(f"{key}'s least {least} has no numeric #define")


def command_arguments(entry, serving):
    """One source's argv for a command: serve's settings and serve additions, or generate's own settings."""
    return entry["settings"] + entry["serve_settings"] if serving else entry["generate_settings"] or entry["settings"]


def composed_values(used, serving):
    return {option_key(argument): (argument, f"{axis}={value}") for axis, value, entry in used
            for argument in command_arguments(entry, serving)}


def clamp_bounds(registry, used):
    """Each declared cross-axis bound is applied at composition, per command. A cap lowers a bounded knob that one
    source sets above the bounding knob's value - offset, where another source sets the bounding knob. A floor raises
    a bounded knob below its least (absent: its CLI default) where the composed argv holds the bounding knob at its
    value and the knob it applies under; an absent knob joins the bounding knob's source. Returns the sources and, per
    command, one record per bound applied: (source, new argument, old argument, bounding argument, bounding source,
    "bounded" or "required")."""
    bounds = registry.get("cross_axis_bounds", [])
    caps = {}
    for serving in (True, False):
        values = composed_values(used, serving)
        caps[serving] = {key: (int(values[bound["bound"]][0].split("=", 1)[1]) - bound["offset"],
                               *values[bound["bound"]])
                         for bound in bounds if bound["kind"] == "cap" and bound["bound"] in values
                         for key in bound["bounded"]}
    records = {True: [], False: []}

    def clamp(arguments, serving, label):
        clamped = []
        for argument in arguments:
            name, number = argument.split("=", 1)
            cap = caps[serving].get(option_key(argument))
            if cap is not None and number.isdigit() and int(number) > cap[0]:
                if cap[2] == label:
                    raise ValueError(f"{label} sets {argument} above its own {cap[1]}")
                records[serving].append((label, f"{name}={cap[0]}", argument, cap[1], cap[2], "bounded"))
                argument = f"{name}={cap[0]}"
            clamped.append(argument)
        return clamped

    result = []
    for axis, value, entry in used:
        label = f"{axis}={value}"
        settings = clamp(entry["settings"], True, label)
        generate = clamp(entry["generate_settings"] or entry["settings"], False, label)
        result.append((axis, value, dict(entry, settings=settings,
                                         serve_settings=clamp(entry["serve_settings"], True, label),
                                         generate_settings=generate if entry["generate_settings"] or
                                         generate != settings else None)))
    for bound in (bound for bound in bounds if bound["kind"] == "floor"):
        bounding = f"{bound['bound']}={bound['value']}"
        applies = {}
        for serving in (True, False):
            values = composed_values(result, serving)
            under = values.get(bound["under"], ("=0",))[0].split("=", 1)[1]
            applies[serving] = values.get(bound["bound"], ("",))[0] == bounding and under not in ("", "0")
        if not any(applies.values()):
            continue
        if not all(applies.values()):
            raise ValueError(f"the floor {bounding} holds for one command only")
        source = composed_values(result, True)[bound["bound"]][1]
        for key in bound["bounded"]:
            raised = f"{key}={bound['least']}"
            present = [(index, field) for index, (_, _, entry) in enumerate(result)
                       for field in ("settings", "serve_settings", "generate_settings")
                       if any(option_key(argument) == key for argument in entry[field] or [])]
            if not present:
                index = next(index for index, (axis, value, _) in enumerate(result) if f"{axis}={value}" == source)
                axis, value, entry = result[index]
                result[index] = (axis, value, dict(entry, settings=entry["settings"] + [raised],
                                                   generate_settings=entry["generate_settings"] and
                                                   entry["generate_settings"] + [raised]))
                for serving in (True, False):
                    records[serving].append((source, raised, f"{key}={bound['default']}", bounding, source,
                                             "required"))
                continue
            for index, field in present:
                axis, value, entry = result[index]
                for argument in entry[field]:
                    number = argument.split("=", 1)[1]
                    if option_key(argument) == key and number.isdigit() and int(number) < bound["least"]:
                        for serving in ((True,) if field == "serve_settings" else (False,) if field ==
                                        "generate_settings" else (True, False)):
                            records[serving].append((f"{axis}={value}", raised, argument, bounding, source,
                                                     "required"))
                result[index] = (axis, value, dict(entry, **{field: [
                    raised if option_key(argument) == key and argument.split("=", 1)[1].isdigit() and
                    int(argument.split("=", 1)[1]) < bound["least"] else argument for argument in entry[field]]}))
    # startup reads a bounding knob's composed value under its own bounds alone (bounding_final_value): no
    # composition clamp may change a knob another bound reads
    read = {bound["bound"] for bound in bounds} | {bound["under"] for bound in bounds if bound["kind"] == "floor"}
    for record in records[True] + records[False]:
        if option_key(record[1]) in read:
            raise ValueError(f"a bound clamps {option_key(record[1])}, which another bound reads, at composition")
    return result, records


def clamp_note(record):
    _, new, old, bounding, source, verb = record
    return f"{option_key(old)} {old.split('=', 1)[1]} -> {new.split('=', 1)[1]} ({verb} by {bounding} from {source})"


def validate_registry(registry, allow_synthetic=False):
    if registry["schema_version"] != 2:
        raise ValueError("unsupported axis registry schema")
    axes = registry["axes"]
    if tuple(axis["name"] for axis in axes) != AXIS_NAMES:
        raise ValueError("axis order must be category/exactness/concurrency/focus")
    for axis in axes:
        values = axis["values"]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"invalid values for {axis['name']}")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]*", value) for value in values):
            raise ValueError(f"invalid axis value in {axis['name']}")
    if registry["weight_families"] != ["original", "exl3"]:
        raise ValueError("weight-family order must remain original/exl3")
    if [tier["name"] for tier in registry["tiers"]] != axes[1]["values"]:
        raise ValueError("numerical tiers must follow exactness values")
    for name, strategy in registry["strategies"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"invalid shared strategy identifier: {name}")
        for field in STRATEGY_FIELDS:
            arguments = argument_tokens(strategy[field])
            keys = [option_key(value) for value in arguments]
            if len(set(keys)) != len(keys):
                raise ValueError(f"duplicate knob destination: {name}/{field}")
        serve_keys = [option_key(value) for field in ("settings", "serve_settings")
                      for value in argument_tokens(strategy[field])]
        if len(set(serve_keys)) != len(serve_keys):
            raise ValueError(f"duplicate serve destination: {name}")
    domains = {axis["name"]: axis["values"] for axis in axes}
    retained_ids = set()
    for route in registry.get("retained_routes", []):
        match = route.get("proposed_axes", {})
        name = route.get("id", "")
        if (not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in retained_ids or
                route.get("selectable") is not False or route.get("weight_family") not in registry["weight_families"] or
                set(match) != set(AXIS_NAMES) or
                any(not isinstance(match[axis], list) or not match[axis] or
                    len(set(match[axis])) != len(match[axis]) or
                    any(value not in domains[axis] for value in match[axis]) for axis in AXIS_NAMES) or
                any(not isinstance(route.get(field), str) or not route[field].strip()
                    for field in ("knob", "fallback", "constraints", "hold"))):
            raise ValueError("retained route needs four axes, a hold and fallback; it cannot be selectable")
        retained_ids.add(name)
    validate_splits(registry, domains)
    validate_bounds(registry)
    validate_numerical_contracts(registry, allow_synthetic)
    return registry


def read_registry():
    return validate_registry(json.loads(REGISTRY.read_text()))


def bucket_targets(exactness, concurrency, focus):
    if concurrency == "single":
        phase = "prefill" if focus in ("ttft", "prefill") else "decode"
        buckets = [phase + ("_fast" if exactness == "fast" else "_accuracy")]
        if exactness == "balanced":
            buckets.append(phase + "_fast")
    else:
        buckets = ["throughput_fast" if exactness == "fast" else "throughput_accuracy"]
        if exactness == "balanced":
            buckets.append("throughput_fast")
        if focus in ("ttft", "prefill"):
            buckets.append("prefill_fast" if exactness == "fast" else "prefill_accuracy")
            if exactness == "balanced":
                buckets.append("prefill_fast")
    if exactness == "exact":
        buckets.insert(0, "exact")
    return ", ".join(buckets)


def yield_knobs(entry, others):
    """A recipe set without each knob it yields that another set of the row gives, per command: serve's settings and
    serve additions against the others' serve argv, generate's argv against theirs."""
    if not entry.get("yields"):
        return entry
    serve_given = {option_key(argument) for other in others for argument in command_arguments(other, True)}
    generate_given = {option_key(argument) for other in others for argument in command_arguments(other, False)}
    kept = lambda arguments, given: [argument for argument in arguments
                                     if option_key(argument) not in given or option_key(argument) not in entry["yields"]]
    settings = kept(entry["settings"], serve_given)
    generate = kept(entry["generate_settings"] or entry["settings"], generate_given)
    return dict(entry, settings=settings, serve_settings=kept(entry["serve_settings"], serve_given),
                generate_settings=generate if entry["generate_settings"] or generate != settings else None)


def compose(registry, split, values):
    """One row of a GPU split's table: its hold, or the argv its axis values insert (the reference recipe under
    exact). Each knob comes from one axis value's set; the split's raw engine defaults are the baseline."""
    spec = registry["splits"][split]
    selected = dict(zip(AXIS_NAMES, values))
    if selected["category"] == "vision":
        return dict(hold=registry["holds"]["vision"], status="held vision")
    for axis in AXIS_NAMES:
        hold = spec["holds"].get(axis, {}).get(selected[axis], "")
        if hold:
            return dict(hold=hold, status=f"held {axis}={selected[axis]}")
    if selected["exactness"] == "exact":
        if selected["concurrency"] != "single":
            return dict(hold=EXACT_SESSION_HOLD, status="held exact scope")
        strategy = registry["strategies"][spec["reference"]]
        return dict(hold="", status="candidate", strategy=spec["reference"],
                    used=[("exactness", "exact", dict(settings=argument_tokens(strategy["settings"]),
                                                      serve_settings=argument_tokens(strategy["serve_settings"]),
                                                      generate_settings=argument_tokens(strategy["generate_settings"])
                                                      or None, numerical_class=strategy["numerical_class"],
                                                      error_class=strategy["error_class"],
                                                      evidence=strategy["evidence"], lossy=None))])
    used = []
    for axis in AXIS_NAMES:
        entry = split_set(registry, split, axis, selected[axis])
        if entry is None:
            continue
        for other, allowed in entry.get("requires", {}).items():
            if selected[other] not in allowed:
                return dict(hold=f"{axis}={selected[axis]} on GPU split {split.upper()} needs " +
                            " or ".join(f"--{other}={item}" for item in allowed) + ".",
                            status=f"held {axis}={selected[axis]} scope")
        # a set that applies only with other values of another axis is left out elsewhere (no hold)
        if any(selected[other] not in allowed for other, allowed in entry.get("applies_with", {}).items()):
            continue
        used.append((axis, selected[axis], entry))
    used = [(axis, value, yield_knobs(entry, [other for other_axis, _, other in used if other_axis != axis]))
            for axis, value, entry in used]
    # The split's base: its measured route over the raw defaults, for each knob no selected set gives, per command.
    base = split_base(registry, split)
    serve_given = {option_key(argument) for _, _, entry in used for argument in command_arguments(entry, True)}
    generate_given = {option_key(argument) for _, _, entry in used for argument in command_arguments(entry, False)}
    settings = [argument for argument in base["settings"] if option_key(argument) not in serve_given]
    generate = [argument for argument in base["settings"] if option_key(argument) not in generate_given]
    label = "+".join(f"{axis}={value}" for axis, value, _ in used) or "base"
    if settings and not generate:
        raise ValueError(f"the generate sets give every base knob, serve's do not: {split} {values}")
    if settings or generate or base["serve_settings"]:
        numerical_class, error_class = base_classes(registry, split)
        used.append(("base", split, dict(
            settings=settings, generate_settings=generate if generate != settings else None,
            serve_settings=[argument for argument in base["serve_settings"] if option_key(argument) not in serve_given],
            numerical_class=numerical_class, error_class=error_class, evidence=spec["base"]["evidence"], lossy=None)))
    used, clamps = clamp_bounds(registry, used)
    return dict(hold="", status="candidate", strategy=label, used=used, clamps=clamps)


INACTIVE_DRAFT_NOTE = "(inactive: no DSpark on this route)"
# An exact recipe's arithmetic: exact numerics, never a relaxed order or the BF16 wire (9 Nov).
EXACT_ARITHMETIC = {"--numerics=exact"}
RELAXED_ARITHMETIC = {"--numerics=relaxed", "--tp_wire_bf16=1"}


def inactive_sources(used, serving):
    """The sources whose knobs are all draft knobs on a command without speculation: inserted for a uniform
    composition, without effect until the route gains DSpark."""
    speculation = [argument.split("=", 1)[1] for _, _, entry in used for argument in command_arguments(entry, serving)
                   if option_key(argument) == "--speculative"]
    if speculation and speculation[-1] != "0":
        return []
    return [f"{axis}={value}" for axis, value, entry in used
            if all(option_key(argument).startswith("--draft_") for argument in command_arguments(entry, serving))]


def draft_sources(used, serving):
    """The sources whose knobs are all draft knobs. Each carries the inactive-draft note, which startup prints only
    where the final speculation is off: the route's, or an explicit --speculative=0."""
    return [f"{axis}={value}" for axis, value, entry in used
            if command_arguments(entry, serving) and
            all(option_key(argument).startswith("--draft_") for argument in command_arguments(entry, serving))]


def sources(used, serving):
    """One line per inserting axis value (or the split's baseline): the knobs it inserts for that command, and on a
    line of draft knobs only, the inactive-draft note (draft_sources)."""
    marked = draft_sources(used, serving)
    return "\n".join(f"{axis}={value}: " +
                     " ".join(option_key(argument) for argument in command_arguments(entry, serving)) +
                     (" " + INACTIVE_DRAFT_NOTE if f"{axis}={value}" in marked else "")
                     for axis, value, entry in used)


def make_choices(registry):
    """Every (GPU split, full tuple) row in split order: the composed argv, class, evidence and hold."""
    tiers = {tier["name"]: tier for tier in registry["tiers"]}
    choices = []
    for split in SPLITS:
        family = registry["splits"][split]["weight_family"]
        for values in itertools.product(*(axis["values"] for axis in registry["axes"])):
            category, exactness, concurrency, focus = values
            row = compose(registry, split, values)
            used = row.get("used", [])
            settings = [argument for _, _, entry in used for argument in entry["settings"]]
            generate = [argument for _, _, entry in used
                        for argument in (entry["generate_settings"] or entry["settings"])]
            serve = [argument for _, _, entry in used for argument in entry["serve_settings"]]
            pinned = [option_key(argument) for argument in settings + generate + serve
                      if option_key(argument) in SESSION_KNOBS]
            if pinned and exactness != SESSION_PINS_EXCEPTION:
                raise ValueError(f"a row pins no context or session count (the user's; only exact rows keep their "
                                 f"reference recipe's): {split} {'/'.join(values)} {pinned[0]}")
            classes = [entry["numerical_class"] for _, _, entry in used]
            for _, _, entry in used:
                validate_choice_contract(entry, exactness, row["hold"])
            numerical_class = next((name for name in ("reference_arithmetic", "lossy", "draft_only")
                                    if name in classes), "same_bits")
            if row["hold"]:
                error_class = tiers[exactness]["error_class"]
                evidence = f"GPU split {split.upper()} holds this tuple: {row['hold']} No settings are applied."
            elif all(axis == "base" for axis, _, _ in used):
                error_class = " ".join(entry["error_class"] for _, _, entry in used) or NO_SET_CLASS
                evidence = (f"No axis value of this tuple carries a knob set on GPU split {split.upper()}; its base "
                            "over the raw engine defaults applies." +
                            "".join(" " + entry["evidence"] for _, _, entry in used))
            else:
                error_class = " ".join(dict.fromkeys(entry["error_class"] for axis, _, entry in used
                                                     if axis != "base"))
                if numerical_class == "lossy":
                    error_class = "lossy; admitted incremental budget and paired fixture/strata limits apply"
                evidence = " ".join(f"{axis}={value}: {entry['evidence']}" for axis, value, entry in used
                                    if axis != "base")
                for _, _, entry in used:
                    if entry["numerical_class"] == "lossy":
                        evidence += " " + lossy_evidence(entry["lossy"], registry["quality_budget"])
            # A set that applies on serve only at one --session_slots count; elsewhere the row runs its
            # concurrency=balanced row (startup_axes.c). 0: none.
            slots_exactly = next((entry["applies_with_session_slots"] for _, _, entry in used
                                  if entry.get("applies_with_session_slots")), 0)
            clamps = row.get("clamps", {True: [], False: []})
            notes = list(dict.fromkeys(clamp_note(record) for record in clamps[True] + clamps[False]))
            if notes:
                evidence += " Cross-axis bound applied: " + "; ".join(notes) + "."
            lines = {serving: "\n".join(" ".join(record[1:]) for record in clamps[serving]) for serving in clamps}
            if any(" " in field or "\n" in field for records in clamps.values() for record in records
                   for field in record):
                raise ValueError("a clamped knob, its bound and its source are single argv words")
            choices.append(dict(values=values, family=family, split=split, strategy=row.get("strategy", "empty"),
                                settings="\n".join(settings), serve_settings="\n".join(serve),
                                generate_settings="\n".join(generate) if generate != settings else "",
                                buckets=bucket_targets(exactness, concurrency, focus),
                                error_class=error_class, evidence=evidence.rstrip(), hold=row["hold"],
                                status=row["status"], numerical_class=numerical_class,
                                sources=sources(used, True), generate_sources=sources(used, False),
                                clamps=lines[True], generate_clamps=lines[False], bounds=notes,
                                serve_slots_exactly=slots_exactly,
                                inactive=list(dict.fromkeys(inactive_sources(used, True) +
                                                            inactive_sources(used, False)))))
    return choices


def c_literal_lines(text):
    if len(text.encode("utf-8")) > LITERAL_LIMIT:
        raise ValueError(f"axis string exceeds {LITERAL_LIMIT} bytes")
    chunks, chunk = [], ""
    for character in text:
        if len(json.dumps(chunk + character)) > C_LINE_WIDTH - 4 and chunk:
            chunks.append(chunk)
            chunk = ""
        chunk += character
    chunks.append(chunk)
    return ["\t" + json.dumps(chunk) for chunk in chunks]


def header_text(registry, choices, registry_sha256):
    strings = {}

    def intern(text):
        if text not in strings:
            strings[text] = f"axis_text_{len(strings)}"
        return strings[text]

    tier_rows = [(tier["name"], intern(tier["error_class"]), intern(tier["hold"]))
                 for tier in registry["tiers"]]
    budget = registry["quality_budget"]
    budget_text = (budget["status"] + f"\nAdmitted: {str(budget['admitted']).lower()}\n" +
                   json.dumps(budget["limits"], sort_keys=True) + "\nRequired strata: " +
                   ", ".join(budget["required_strata"]) + "\n" + "\n".join(budget["required_metrics"]) + "\n")
    envelope = registry.get("performance_envelope")
    if envelope is not None:
        budget_text += ("\nPerformance envelope proposal:\n" + envelope["status"] + "\n" +
                        json.dumps(envelope["limits"], sort_keys=True) + "\n" +
                        envelope["protocol"] + "\n" + envelope["motivation"] + "\n")
    budget_pointer = intern(budget_text)
    choice_rows = []
    for choice in choices:
        pointers = [intern(choice[field]) for field in
                    ("settings", "serve_settings", "generate_settings", "buckets", "error_class", "evidence", "hold",
                     "sources", "generate_sources", "clamps", "generate_clamps")]
        choice_rows.append((choice, pointers))
    # REUSE-IgnoreStart: the tag below is the generated file's, not this program's
    lines = ["/* SPDX-License-Identifier: AGPL-3.0-or-later */",
             # REUSE-IgnoreEnd
             "/* Generated by tools/cli/generate_axes.py; do not edit.",
             f" * Registry SHA256: {registry_sha256} */",
             "#ifndef CLI_AXES_DATA_H", "#define CLI_AXES_DATA_H", "",
             '#include "startup_axes.h"', "#include <stddef.h>", ""]
    for axis in registry["axes"]:
        lines.append(f"static const char *const axis_{axis['name']}_values[] = {{")
        lines.append("\t" + ", ".join(json.dumps(value) for value in axis["values"]) + "")
        lines += ["};", ""]
        lines.append(f"static const char axis_{axis['name']}_meaning[] =")
        literals = c_literal_lines(axis["meaning"])
        literals[-1] += ";"
        lines += literals + [""]
    lines.append("static const struct axis_domain axis_domains[] = {")
    for axis in registry["axes"]:
        name = axis["name"]
        lines += [f'\t{{ "{name}", "--{name}=", axis_{name}_meaning, axis_{name}_values,',
                  f"\t  sizeof(axis_{name}_values) / sizeof(axis_{name}_values[0]) }},"]
    lines += ["};", ""]
    for text, identifier in strings.items():
        lines.append(f"static const char {identifier}[] =")
        literals = c_literal_lines(text)
        literals[-1] += ";"
        lines += literals + [""]
    lines += [f"static const char *const axis_quality_budget_text = {budget_pointer};", "",
              "/* --placement's engine default (startup_policy.c) and every GPU split with a table. */",
              f'static const char axis_default_split[] = "{SPLITS[0]}";',
              f'static const char axis_split_names[] = "{", ".join(SPLITS)}";', ""]
    # A cap is one entry per bounded knob, with that knob's parser least: it never lowers a knob below
    # it. A floor keeps its declared least.
    lines.append("static const struct axis_bound axis_bounds[] = {")
    for bound in registry.get("cross_axis_bounds", []):
        floor = bound["kind"] == "floor"
        for bounded in [bound["bounded"]] if floor else [[key] for key in bound["bounded"]]:
            least = bound["least"] if floor else parser_least(bounded[0])
            lines += [f'\t{{ {str(floor).lower()}, "{bound["bound"]}", "{bound.get("value", "")}", '
                      f'"{bound.get("under", "")}",',
                      f'\t  "{" ".join(bounded)}", {bound.get("offset", 0)}u, {least}u }},']
    lines += ["};", ""]
    lines.append("static const struct axis_tier axis_tiers[] = {")
    for name, error_class, hold in tier_rows:
        lines.append(f'\t{{ "{name}", {error_class}, {hold} }},')
    lines += ["};", "", "static const struct axis_choice axis_choices[] = {"]
    for choice, pointers in choice_rows:
        values = ", ".join(json.dumps(value) for value in choice["values"])
        lines += [f'\t{{ {{ {values} }}, "{choice["family"]}", "{choice["split"]}",',
                  "\t  " + ", ".join(pointers[:3]) + ",",
                  "\t  " + ", ".join(pointers[3:6]) + ",",
                  "\t  " + ", ".join(pointers[6:9]) + ",",
                  "\t  " + ", ".join(pointers[9:]) + f", {choice['serve_slots_exactly']}u }},"]
    lines += ["};", "", "static const struct axis_base axis_bases[] = {"]
    for split in SPLITS:
        base = split_base(registry, split)
        fields = [split, registry["splits"][split]["base"]["route"],
                  "\n".join(base["settings"] + base["serve_settings"]),
                  "\n".join(f"{key}={value}" for key, value in base["omitted"])]
        lines.append("\t{")
        for value in fields:
            literals = c_literal_lines(value)
            literals[-1] += ","
            lines += literals
        lines.append("\t},")
    lines += ["};", "", "static const struct axis_retained_route axis_retained_routes[] = {"]
    for route in registry["retained_routes"]:
        fields = [route["id"], route["weight_family"],
                  " / ".join(",".join(route["proposed_axes"][axis]) for axis in AXIS_NAMES),
                  route["knob"], route["fallback"], route["constraints"], route["hold"]]
        lines.append("\t{")
        for value in fields:
            literals = c_literal_lines(value)
            literals[-1] += ","
            lines += literals
        lines.append("\t},")
    lines += ["};", "", "#endif", ""]
    if any(len(line.expandtabs(8)) > 120 for line in lines):
        raise ValueError("generated axis header exceeds the source line width")
    return "\n".join(lines)


def split_tables(registry):
    """Per GPU split: each axis value's knob set, its evidence and the route columns derived from that set."""
    domains = {axis["name"]: axis["values"] for axis in registry["axes"]}
    lines = ["", "## GPU split tables", "",
             "`--placement` picks the GPU split; without it the split is PP5, the engine default. Each selected axis",
             "value adds its knob set to the split's base. An unset axis takes balanced, so a run without axis flags",
             "runs the split's all-balanced row, its base. No set pins the context or the slots: the raw defaults",
             "serve as many sessions of 1M context as fit. `--axes=off` inserts nothing. Each knob belongs to one axis",
             "per split. Under exact, the split's reference recipe is the whole configuration. Explicit knobs win over",
             "every set. The last three columns derive from each set's own settings (`default`: the set leaves that",
             "knob alone).", ""]
    for split, spec in registry["splits"].items():
        reference = spec["reference"]
        lines += [f"### GPU split {split.upper()}", "",
                  f"Weight family: {spec['weight_family']}. Exact: " +
                  (f"the reference recipe `{reference}`, with concurrency single or unset." if reference else
                   "held.") + " Base (from `" + spec["base"]["route"] + "`, for each knob no selected set gives): " +
                  (" ".join(f"`{argument}`" for argument in split_base(registry, split)["settings"] +
                            split_base(registry, split)["serve_settings"]) or "none") + ". " +
                  spec["base"]["evidence"], "",
                  "| Axis | Value | Knob set | Evidence | " + " | ".join(label for label, _ in ROUTE_PARAMETERS) + " |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for axis in AXIS_NAMES:
            for value in domains[axis]:
                hold = (registry["holds"]["vision"] if (axis, value) == ("category", "vision") else
                        spec["holds"].get(axis, {}).get(value, ""))
                entry = split_set(registry, split, axis, value)
                if hold:
                    cells = ["HELD", hold, split.upper(), "-", "-"]
                elif (axis, value) == ("exactness", "exact"):
                    cells = [f"`{reference}` (reference recipe)", "The reference recipe's evidence: see Knob values.",
                             *route_parameters(registry["strategies"][reference])]
                elif entry is None:
                    cells = ["(empty)", "-", split.upper(), "default", "default"]
                else:
                    knobs = (f"`{entry['strategy']}`" + (" without " + ", ".join(f"`{key}`" for key in entry["drop"])
                                                         if entry["drop"] else "") +
                             ("; yields " + ", ".join(f"`{key}`" for key in entry["yields"]) +
                              " where another set gives them" if entry["yields"] else "")
                             if "strategy" in entry else
                             " ".join(f"`{argument}`" for argument in entry["settings"] + entry["serve_settings"]))
                    if entry.get("requires"):
                        knobs += "; needs " + ", ".join(f"{other}={'/'.join(allowed)}"
                                                        for other, allowed in entry["requires"].items())
                    if entry.get("applies_with"):
                        knobs += "; only with " + ", ".join(f"{other}={'/'.join(allowed)}"
                                                            for other, allowed in entry["applies_with"].items())
                    if entry.get("applies_with_session_slots"):
                        knobs += (f"; on serve only at exactly `--session_slots={entry['applies_with_session_slots']}`"
                                  " (otherwise the concurrency=balanced row)")
                    columns = route_parameters(dict(
                        settings=[f"--placement={split}", *entry["settings"]], serve_settings=entry["serve_settings"],
                        generate_settings=[f"--placement={split}", *entry["generate_settings"]]
                        if entry["generate_settings"] else []))
                    if entry.get("same_as"):
                        knobs = f"as {axis}={entry['same_as']}: " + knobs
                    cells = [knobs, entry["evidence"], *columns]
                lines.append(f"| {axis} | {value} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def markdown_text(registry, choices, registry_sha256):
    lines = ["# Axis table", "", "Generated from `tools/cli/axes.json` by `tools/cli/generate_axes.py`.",
             f"Registry SHA256: `{registry_sha256}`.", "",
             "The four selectors apply to generate and serve. There are no selectable names or aliases.", "",
             "| Axis | Values | Meaning |", "| --- | --- | --- |"]
    for axis in registry["axes"]:
        lines.append(f"| `--{axis['name']}` | " + " / ".join(axis["values"]) + f" | {axis['meaning']} |")
    lines += ["", "| Exactness tier | Error class | Current admission |", "| --- | --- | --- |"]
    for tier in registry["tiers"]:
        admission = tier['hold'] or ("; ".join(
            f"GPU split {split.upper()}: " + (f"the {registry['splits'][split]['reference']} recipe"
                                              if registry["splits"][split]["reference"] else "holds exact")
            for split in SPLITS) + "." if tier['name'] == 'exact' else
                                     'Current-arithmetic knob sets per GPU split; scope below.')
        lines.append(f"| {tier['name']} | {tier['error_class']} | {admission} |")
    lines += ["", "Each GPU split has its own table below. Original official-reference arithmetic is scoped to its",
              "admitted serial M1 recipe. EXL3 uses the label **reference arithmetic (EXL3: not official-exact)** when",
              "compatible forms are qualified.", ""]
    for note in registry["notes"]:
        lines.append("- " + note)
    lines += split_tables(registry)
    lines += ["", "## Retained route proposals", "",
              "These records are held metadata, never executable resolver settings. Tuple order is",
              "category / exactness / concurrency / focus; listed values are proposals requiring the stated gates.", ""]
    for route in registry["retained_routes"]:
        scope = " / ".join(",".join(route["proposed_axes"][axis]) for axis in AXIS_NAMES)
        lines += [f"### {route['id']}", "", f"Family: {route['weight_family']}. Proposed: `{scope}`.", "",
                  f"Branch control: `{route['knob']}`. Old fallback: `{route['fallback']}`.", "",
                  route["constraints"], "", "HOLD: " + route["hold"], ""]
    lines += ["", "## Route parameters", "",
              "Each route is described by three separate parameters, derived from its settings: the GPU split",
              "(`--placement`: TP4_PRO is tensor parallel across the four CMP 170HX with the RTX PRO 6000 as a pipeline",
              "stage, PP5 the five GPUs in pipeline, TP4_EXL3 tensor parallel across the four CMP 170HX), the pipeline",
              "depth (`--pipeline_steps`: the steps in flight) and the serving slots (`--session_slots`). A generate value",
              "that differs from the serve value follows in parentheses.", "",
              "| Route | " + " | ".join(label for label, _ in ROUTE_PARAMETERS) + " |", "| --- | --- | --- | --- |"]
    for name, strategy in registry["strategies"].items():
        if name != "empty":
            lines.append(f"| `{name}` | " + " | ".join(route_parameters(strategy)) + " |")
    lines += ["", "## Knob values", ""]
    for name, strategy in registry["strategies"].items():
        if name == "empty":
            continue
        split, depth, slots = route_parameters(strategy)
        lines += [f"### {name}", "", strategy["description"], "",
                  f"GPU split: {split}. Pipeline depth: {depth}. Serving slots: {slots}.", "",
                  f"Numerical class: `{strategy['numerical_class']}`. Declared lossy knobs: " +
                  (", ".join(f"`{knob}`" for knob in strategy["lossy_knobs"]) or "none") + ".", "",
                  "Each line below is one argv token; embedded spaces and semicolons stay in that token.", ""]
        for label, field in (("Serve base", "settings"), ("Serve additions", "serve_settings"),
                             ("Generate base", "generate_settings")):
            value = strategy[field] or (strategy["settings"] if field == "generate_settings" else "")
            lines += [label + ":", "", "```text", "\n".join(argument_tokens(value)) or "(none)", "```", ""]
        lines += [strategy["evidence"], ""]
        if strategy["lossy_knobs"]:
            lines += [lossy_evidence(strategy, registry["quality_budget"]), ""]
    lines += ["## Cross-axis rules", "",
              "Five kinds of rule join the sets of one row; each is declared in axes.json and applied by the",
              "generator or, for the slot count, by startup. A cap clamps a knob that one source sets above another",
              "source's bound; a floor raises a knob below what another source's setting needs; a set may apply only",
              "with values of another axis, and is left out elsewhere, never refused; a concurrency set may apply on",
              "serve only at one final `--session_slots` count, and elsewhere the row runs its concurrency=balanced row;",
              "a recipe may yield one of its knobs to the axis that owns it, so the recipe's measured value runs",
              "wherever that axis's set of the row gives none. Startup prints each applied cap or floor after its key",
              "on the `axes from` line. A bound reads its bounding knob after that knob's own bounds, one level (the",
              "generator refuses a longer chain), and a knob several bounds name takes the first, in declared order,",
              "that changes it.", "", "Caps and floors:", ""]
    for bound in registry.get("cross_axis_bounds", []):
        lines.append(f"- {bound['kind']}: `{bound['bound']}`" +
                     (f"=`{bound['value']}` (under `{bound['under']}`) raises " if bound["kind"] == "floor" else
                      " bounds ") + ", ".join(f"`{key}`" for key in bound["bounded"]) + ": " + bound["rule"])
    lines += ["", "Sets that apply only with values of another axis:", ""]
    lines += [f"- {split.upper()} {axis}={value}: only with " +
              ", ".join(f"{other}={'/'.join(allowed)}" for other, allowed in entry["applies_with"].items())
              for split, spec in registry["splits"].items() for axis, values in spec["sets"].items()
              for value, entry in values.items() if entry.get("applies_with")] or ["- none"]
    lines += ["", "Sets that apply on serve only at one serving slot count (elsewhere the row runs its",
              "concurrency=balanced row, and startup says how many sessions fit):", ""]
    lines += [f"- {split.upper()} {axis}={value}: only at exactly `--session_slots="
              f"{entry['applies_with_session_slots']}`"
              for split, spec in registry["splits"].items() for axis, values in spec["sets"].items()
              for value, entry in values.items() if entry.get("applies_with_session_slots")] or ["- none"]
    lines += ["", "Recipe knobs yielded to the axis that owns them:", ""]
    lines += [f"- {split.upper()} {axis}={value} (`{entry['strategy']}`): " +
              ", ".join(f"`{key}`" for key in entry["yields"])
              for split, spec in registry["splits"].items() for axis, values in spec["sets"].items()
              for value, entry in values.items() if entry.get("yields")] or ["- none"]
    flagged = [choice for choice in choices if choice["bounds"]]
    lines += ["", f"Rows with an applied cap or floor: {len(flagged) or 'none'}."]
    for choice in flagged:
        lines.append(f"- {choice['split'].upper()} " + "/".join(choice["values"]) + ": " + "; ".join(choice["bounds"]))
    inactive = [choice for choice in choices if choice["inactive"]]
    lines += ["", "## Inactive draft knobs", "",
              "Each axis value's set is inserted on every row of its split. A set of draft knobs acts only with "
              "speculation: its `axes from` line carries",
              f"`{INACTIVE_DRAFT_NOTE}`, which startup prints only where the final speculation is off (a route "
              "without DSpark, or an explicit `--speculative=0`). Every base runs DSpark, so the rows below are those "
              "whose composition has none.", "",
              f"Rows with inactive draft knobs: {len(inactive) or 'none'}."]
    for choice in inactive:
        lines.append(f"- {choice['split'].upper()} " + "/".join(choice["values"]) + ": " +
                     ", ".join(choice["inactive"]))
    lines += [""]
    lines += ["## Quality budget", "", registry["quality_budget"]["status"], ""]
    lines += [f"Budget admitted: `{str(registry['quality_budget']['admitted']).lower()}`.",
              "Required strata: " + ", ".join(registry["quality_budget"]["required_strata"]) + ".", "",
              "A lossy strategy needs a paired, held-out receipt per knob and stratum. Every receipt records",
              "source hashes, metric and agreement denominators, raw and centered errors, uncertainty and",
              "a whole-workload rate gain. Synthetic receipts cannot enter generated runtime data.", ""]
    for metric in registry["quality_budget"]["required_metrics"]:
        lines.append("- " + metric)
    limits = registry["quality_budget"]["limits"]
    if limits is not None:
        lines += ["", "Proposed limits:", "", "```json", json.dumps(limits, indent=2), "```"]
    envelope = registry.get("performance_envelope")
    if envelope is not None:
        lines += ["", "## Performance envelope", "", envelope["status"], "", "```json",
                  json.dumps(envelope["limits"], indent=2), "```", "", envelope["protocol"], "",
                  envelope["motivation"], "",
                  "A qualification-candidate rule names every axis value explicitly and can bypass only the",
                  "concurrency-compromise hold. It cannot bypass tier, original-weight, vision or numerical",
                  "gates. A failed envelope returns the rule to a nonempty hold; the margin is not relaxed."]
    lines += ["", "## Complete tuple map", "",
              f"The map contains {len(choices)} rows: {len(SPLITS)} GPU splits by {len(choices) // len(SPLITS)} full "
              "tuples. Each cell names the axis values whose sets it inserts (raw: none). A held row inserts nothing.",
              "",
              "| Category | Exactness | Concurrency | Focus | " + " | ".join(split.upper() for split in SPLITS) +
              " | Seven-bucket targets |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    grouped = {}
    for choice in choices:
        grouped.setdefault(choice["values"], {})[choice["split"]] = choice
    for values, variants in grouped.items():
        cells = [variants[split]["status"] if variants[split]["hold"] else
                 variants[split]["strategy"] + f"; {variants[split]['numerical_class']}" for split in SPLITS]
        lines.append("| " + " | ".join([*values, *cells, variants[SPLITS[0]]["buckets"]]) + " |")
    lines += ["", "Hold conditions:", ""]
    for tier in registry["tiers"]:
        if tier["hold"]:
            lines.append(f"- {tier['name']}: {tier['hold']}")
    for name, hold in registry["holds"].items():
        lines.append(f"- {name}: {hold}")
    lines.append("")
    return "\n".join(lines)


def c_content(source):
    """Compare generated C content while accepting clang-format layout and literal splitting."""
    tokens = re.findall(r'"(?:\\.|[^"\\])*"(?:\s*"(?:\\.|[^"\\])*")*|[^\s]', source)
    return [("literal", ast.literal_eval("(" + token + ")")) if token.startswith('"')
            else ("token", token) for token in tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    registry = read_registry()
    choices = make_choices(registry)
    registry_sha256 = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    document = markdown_text(registry, choices, registry_sha256)
    catalog_prefix = CATALOG.read_text().split(AXIS_MARKER, 1)[0].rstrip()
    outputs = {HEADER: header_text(registry, choices, registry_sha256),
               # REUSE-IgnoreStart: the tag below is the generated file's, not this program's
               DOCUMENT: "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n" + document,
               # REUSE-IgnoreEnd
               CATALOG: catalog_prefix + "\n\n" + AXIS_MARKER + "\n" + document}
    for path, generated in outputs.items():
        if args.check:
            if not path.exists() or (c_content(path.read_text()) != c_content(generated) if path.suffix == ".h"
                                     else path.read_text() != generated):
                raise SystemExit(f"generated axis output is stale: {path.relative_to(ROOT)}")
        else:
            write_generated(path, generated)
    print(f"axis registry: {len(choices)} rows; {len(SPLITS)} GPU splits by {len(choices) // len(SPLITS)} full tuples; "
          "outputs current")


def write_generated(path, generated):
    if path.suffix == ".h" and path.exists() and c_content(path.read_text()) == c_content(generated):
        path.touch()
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as output:
        temporary = Path(output.name)
        output.write(generated)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
