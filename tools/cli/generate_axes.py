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
AXIS_MARKER = "<!-- generated-axis-table -->"
AXIS_NAMES = ("category", "exactness", "concurrency", "focus")
STRATEGY_FIELDS = ("settings", "serve_settings", "generate_settings")
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


def family_rules(registry, family, tier):
    if family == "original":
        return [rule for rule in registry.get("original_rules", [])
                if tier in rule["match"]["exactness"]]
    return registry.get(tier + "_exl3_rules", [])


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
    for family, tier_name in itertools.product(registry["weight_families"], ("balanced", "exact", "fast")):
        for rule in family_rules(registry, family, tier_name):
            strategy = registry["strategies"][rule["strategy"]]
            if strategy["numerical_class"] == "lossy" and tier_name != "fast":
                raise ValueError(f"lossy strategy in {tier_name} mapping")
            candidate = rule.get("qualification_candidate", False)
            if type(candidate) is not bool:
                raise ValueError("qualification_candidate must be a boolean")
            if tier_name == "exact" and "hold" in rule:
                if (not isinstance(rule["hold"], str) or
                        (family == "original" and strategy["numerical_class"] != "reference_arithmetic") or
                        (family == "exl3" and not rule["hold"])):
                    raise ValueError("exact scope requires original reference arithmetic or a held EXL3 rule")
                continue
            if "hold" in rule and not candidate:
                raise ValueError("a per-rule hold requires an explicit qualification candidate")
            if candidate:
                match = rule["match"]
                domains = {axis["name"]: axis["values"] for axis in registry["axes"]}
                if (family != "exl3" or tier_name != "balanced" or set(match) != set(AXIS_NAMES) or
                        any(not isinstance(match[name], list) or len(match[name]) != 1 or
                            match[name][0] not in domains[name] for name in AXIS_NAMES) or
                        match["exactness"] != ["balanced"] or
                        (match["concurrency"] != ["balanced"] and not rule.get("hold")) or
                        strategy["numerical_class"] not in ("same_bits", "draft_only") or
                        strategy.get("raw_only", False) or not isinstance(rule.get("hold", ""), str)):
                    raise ValueError("qualification exception needs a complete current-math balanced tuple")


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


def validate_registry(registry, allow_synthetic=False):
    if registry["schema_version"] != 1:
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
    rule_sets = [registry.get(tier + "_exl3_rules", []) for tier in domains["exactness"]]
    rule_sets.append(registry.get("original_rules", []))
    for rules in rule_sets:
        for rule in rules:
            match = rule["match"]
            if (not match or not set(match) <= set(AXIS_NAMES) or
                    rule["strategy"] not in registry["strategies"] or
                    any(not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed) or
                        any(value not in domains[name] for value in allowed)
                        for name, allowed in match.items())):
                raise ValueError("invalid axis selection rule")
    for rule in registry.get("original_rules", []):
        if set(rule["match"]) != set(AXIS_NAMES):
            raise ValueError("original rules must explicitly name all four axes")
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


def make_choices(registry):
    tiers = {tier["name"]: tier for tier in registry["tiers"]}
    choices = []
    rules = {(family, tier): family_rules(registry, family, tier)
             for family, tier in itertools.product(registry["weight_families"], tiers)}
    rule_uses = {key: [0 for unused in rows] for key, rows in rules.items()}
    for values in itertools.product(*(axis["values"] for axis in registry["axes"])):
        category, exactness, concurrency, focus = values
        selected = dict(zip(AXIS_NAMES, values))
        for family in registry["weight_families"]:
            strategy_name, hold, status = "empty", "", "candidate"
            family_tier_rules = rules[family, exactness]
            if tiers[exactness]["hold"]:
                hold, status = tiers[exactness]["hold"], "held tier"
            elif category == "vision" and not (family == "exl3" and exactness == "exact"):
                hold, status = registry["holds"]["vision"], "held vision"
            else:
                matches = [index for index, rule in enumerate(family_tier_rules)
                           if all(selected[name] in allowed for name, allowed in rule["match"].items())]
                exceptions = [index for index in matches
                              if family_tier_rules[index].get("qualification_candidate")]
                if family == "original" and not matches:
                    hold, status = registry["holds"]["original"], "held original"
                elif family == "exl3" and exactness != "exact" and concurrency == "balanced" and not exceptions:
                    strategy_name = "batch_session"
                    hold, status = registry["holds"]["compromise"], "held compromise"
                else:
                    if len(matches) != 1:
                        raise ValueError(f"{exactness} {family} tuple has {len(matches)} rules: {values}")
                    index = matches[0]
                    rule_uses[family, exactness][index] += 1
                    rule = family_tier_rules[index]
                    strategy_name = rule["strategy"]
                    if rule.get("qualification_candidate") or "hold" in rule:
                        hold = rule.get("hold", "")
                        status = "held qualification candidate" if hold else "qualification candidate"
            strategy = registry["strategies"][strategy_name]
            validate_choice_contract(strategy, exactness, hold)
            error_class = tiers[exactness]["error_class"] if hold else strategy["error_class"]
            if not hold and strategy["numerical_class"] == "lossy":
                error_class = "lossy; admitted incremental budget and paired fixture/strata limits apply"
            evidence = strategy["evidence"]
            if exactness == "exact" and hold:
                evidence = ("RND_R.md and track_r_v1/FIXES.md distinguish the qualified serial M1 "
                            "packet proof from production admission. This selected runtime scope is held: " + hold)
            elif exactness == "fast" and hold:
                evidence = ("This tuple has no selected qualified lossy strategy; numerical and serving "
                            "dependency gates remain open. No lossy knob is applied.")
            elif family == "original" and hold:
                evidence = ("Legacy serial TP4_PRO and PP5 depth2/5/8 already exist in main7d86c45/S; "
                            "there is no unmerged O dependency for those paths. This tuple has no admitted "
                            "original rule: balanced/mixed/vision or prefill/TTFT/envelope evidence is pending. "
                            "Overlap, original BATCH, wide and chunk-core selectors exist in main75; their joins remain unqualified. Query/dense-shard ports and complete R forms remain separate source dependencies.")
            if category == "balanced" and exactness != "exact" and not hold:
                evidence += (" Category balanced is assessed with the scoped full mixed-user benchmark; homogeneous "
                             "cells do not establish its aggregate or latency objective.")
            if category == "code" and focus == "balanced" and exactness != "exact" and not hold:
                floor = registry["performance_envelope"]["limits"]["code_balanced_non_code_rate_vs_own_best_min"]
                evidence += (f" The proposed non-code rate floor is {100 * floor:.6g}% of each kind's measured best. "
                             "Default Fraser median/worst TTFT and long-prefill pause deltas must be <=0. "
                             "These paired comparisons remain pending; no bounded-compromise claim is made yet.")
            if focus in ("ttft", "prefill") and not hold:
                evidence += " A phase-specific optimum requires paired phase measurements; a shared setting alone does not establish it."
            if not hold:
                evidence += " " + lossy_evidence(strategy, registry["quality_budget"])
            choices.append(dict(values=values, family=family, strategy=strategy_name,
                                **{field: "\n".join(argument_tokens(strategy[field]))
                                   for field in STRATEGY_FIELDS},
                                buckets=bucket_targets(exactness, concurrency, focus),
                                error_class=error_class, evidence=evidence.rstrip(), hold=hold, status=status,
                                numerical_class=strategy["numerical_class"]))
    for (family, tier), counts in rule_uses.items():
        if not tiers[tier]["hold"] and any(count == 0 for count in counts):
            raise ValueError(f"axis registry contains an unused {tier} {family} selection rule")
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
                    ("settings", "serve_settings", "generate_settings", "buckets", "error_class", "evidence", "hold")]
        choice_rows.append((choice, pointers))
    lines = ["/* SPDX-License-Identifier: AGPL-3.0-or-later */",
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
    lines += [f"static const char *const axis_quality_budget_text = {budget_pointer};", ""]
    lines.append("static const struct axis_tier axis_tiers[] = {")
    for name, error_class, hold in tier_rows:
        lines.append(f'\t{{ "{name}", {error_class}, {hold} }},')
    lines += ["};", "", "static const struct axis_choice axis_choices[] = {"]
    for choice, pointers in choice_rows:
        values = ", ".join(json.dumps(value) for value in choice["values"])
        lines += [f'\t{{ {{ {values} }}, "{choice["family"]}",',
                  "\t  " + ", ".join(pointers[:3]) + ",",
                  "\t  " + ", ".join(pointers[3:]) + " },"]
    lines += ["};", "", "#endif", ""]
    if any(len(line.expandtabs(8)) > 120 for line in lines):
        raise ValueError("generated axis header exceeds the source line width")
    return "\n".join(lines)


def markdown_text(registry, choices, registry_sha256):
    lines = ["# Axis table", "", "Generated from `tools/cli/axes.json` by `tools/cli/generate_axes.py`.",
             f"Registry SHA256: `{registry_sha256}`.", "",
             "The four selectors apply to generate and serve. There are no selectable names or aliases.", "",
             "| Axis | Values | Meaning |", "| --- | --- | --- |"]
    for axis in registry["axes"]:
        lines.append(f"| `--{axis['name']}` | " + " / ".join(axis["values"]) + f" | {axis['meaning']} |")
    lines += ["", "| Exactness tier | Error class | Current admission |", "| --- | --- | --- |"]
    for tier in registry["tiers"]:
        admission = tier['hold'] or ('Original serial M1 rule below; EXL3 remains held.'
                                     if tier['name'] == 'exact' else
                                     'Current-arithmetic candidates; qualification scope below.')
        lines.append(f"| {tier['name']} | {tier['error_class']} | {admission} |")
    lines += ["", "Original weights and EXL3 have separate admission records for every full tuple.",
              "Original official-reference arithmetic is scoped to its admitted serial M1 recipe. EXL3 uses the label",
              "**reference arithmetic (EXL3: not official-exact)** when compatible forms are qualified.", ""]
    for note in registry["notes"]:
        lines.append("- " + note)
    lines += ["", "## Knob values", ""]
    for name, strategy in registry["strategies"].items():
        if name == "empty":
            continue
        lines += [f"### {name}", "", strategy["description"], "",
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
              f"The map contains {len(choices)} weight-family records across {len(choices) // 2} full tuples.",
              "A held row never inserts its displayed candidate knobs.", "",
              "| Category | Exactness | Concurrency | Focus | EXL3 knob set / admission | Original knob set / admission | Seven-bucket targets |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    grouped = {}
    for choice in choices:
        grouped.setdefault(choice["values"], {})[choice["family"]] = choice
    for values, variants in grouped.items():
        exl3, original = variants["exl3"], variants["original"]
        exl3_status = exl3["status"] + (f"; candidate `{exl3['strategy']}`" if exl3["strategy"] != "empty" else "")
        if not exl3["hold"]:
            label = "qualification candidate" if exl3["status"] == "qualification candidate" else "candidate"
            exl3_status = f"`{exl3['strategy']}`; {exl3['numerical_class']} {label}"
        original_status = original["status"] if original["hold"] else (
            f"`{original['strategy']}`; {original['numerical_class']} candidate; see source-scoped GPU observations")
        lines.append("| " + " | ".join([*values, exl3_status, original_status, exl3["buckets"]]) + " |")
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
               DOCUMENT: "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n" + document,
               CATALOG: catalog_prefix + "\n\n" + AXIS_MARKER + "\n" + document}
    for path, generated in outputs.items():
        if args.check:
            if not path.exists() or (c_content(path.read_text()) != c_content(generated) if path.suffix == ".h"
                                     else path.read_text() != generated):
                raise SystemExit(f"generated axis output is stale: {path.relative_to(ROOT)}")
        else:
            write_generated(path, generated)
    print(f"axis registry: {len(choices)} family records; {len(choices) // 2} full tuples; outputs current")


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
