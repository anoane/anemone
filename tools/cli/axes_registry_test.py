#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic receipt/parser controls only; no model-quality or GPU measurement."""
import copy
import hashlib
import importlib.util
import json
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("axis_generator", HERE / "generate_axes.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)
KNOB = "--synthetic_loss=1"


def digest(label):
    return hashlib.sha256(("SYNTHETIC ONLY: " + label).encode()).hexdigest()


def registry():
    return json.loads((HERE / "axes.json").read_text())


def agreement(record, metric, value):
    fraction = Fraction(str(value))
    denominator = GENERATOR.METRIC_DENOMINATORS[metric]
    record["metrics"][metric] = value
    record["denominators"][denominator] = fraction.denominator * 1000
    record["agreement_counts"][GENERATOR.AGREEMENT_COUNTS[metric]] = fraction.numerator * 1000


def synthetic_fixture():
    """Exact-boundary invented numbers exercise parsing; they qualify no real knob."""
    budget = copy.deepcopy(registry()["quality_budget"])
    budget["admitted"] = True
    metrics = {name.rsplit("_", 1)[0]: value for name, value in budget["limits"].items()}
    metrics.update(raw_logit_RMS=0.03, raw_logit_absolute=0.2, worst_KL_nats=0.02)
    rows = []
    for stratum in budget["required_strata"]:
        baseline = {field: digest(field + stratum) for field in GENERATOR.IDENTITY_FIELDS}
        candidate = dict(baseline, settings_sha256=digest("candidate settings " + stratum))
        row = {
            "id": "synthetic-" + stratum, "knob": KNOB, "synthetic": True,
            "source": "synthetic://parser-fixture/" + stratum, "source_sha256": digest("receipt " + stratum),
            "fixture": {"id": "synthetic-held-out", "sha256": digest("fixture"),
                        "held_out": True, "smoke_only": False},
            "stratum": stratum, "scope": "Synthetic context and row-shape description; no execution.",
            "uncertainty": "Synthetic parser input; no statistical or quality inference.",
            "baseline": baseline, "candidate": candidate,
            "denominators": {"kl_tokens": 10000, "ppl_tokens": 10000, "top1_tokens": 1000,
                             "logit_values": 129280000, "responses": 1000,
                             "pinned_cases": 1000, "unique_requests": 1000},
            "metrics": copy.deepcopy(metrics), "agreement_counts": {},
            "whole_workload": {"source": "synthetic://whole-workload",
                               "source_sha256": digest("workload"), "requests": 1000,
                               "unique_requests": 1000, "baseline_completion_tokens": 10000,
                               "candidate_completion_tokens": 10000, "baseline_seconds": 10,
                               "candidate_seconds": 9},
            "reference": None, "reference_unavailable_reason": "Synthetic test has no reference execution.",
        }
        for side, identity in (("baseline", baseline), ("candidate", candidate)):
            for field in ("binary_sha256", "settings_sha256"):
                row["whole_workload"][side + "_" + field] = identity[field]
        row["whole_workload"].update(input_sha256=digest("workload inputs " + stratum),
                                     client_sha256=digest("workload client"),
                                     weights_sha256=baseline["weights_sha256"])
        for metric in GENERATOR.AGREEMENT_COUNTS:
            agreement(row, metric, row["metrics"][metric])
        rows.append(row)
    strategy = {"description": "SYNTHETIC TEST ONLY", "settings": KNOB, "serve_settings": "",
                "generate_settings": "", "numerical_class": "lossy", "lossy_knobs": [KNOB],
                "quality_receipts": {KNOB: rows}, "raw_only": False,
                "error_class": "Synthetic test only", "evidence": "No actual measurements."}
    return budget, strategy


class AxisNumericalContractTest(unittest.TestCase):
    def test_original_reference_scope_is_prepared_and_exl3_remains_held(self):
        data = registry()
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(data))
        original = next(row for row in rows if row['family'] == 'original' and
                        row['values'] == ('code', 'exact', 'single', 'decode'))
        self.assertEqual(original['strategy'], 'original_reference_serial_m1')
        self.assertEqual(original['numerical_class'], 'reference_arithmetic')
        settings = original['settings'].split('\n')
        for required in ('--official_reference=1', '--official_hc_rows=1',
                         '--rows=1', '--wide_rows=0', '--full_steps=0',
                         '--speculative=0', '--pro_emulation=0'):
            self.assertIn(required, settings)
        for required in ('--prefix_reuse=0', '--prefix_reuse_exact=0', '--session_fork=0'):
            self.assertIn(required, original['serve_settings'].split('\n'))
        for row in rows:
            if row['family'] == 'exl3' and row['values'][1] == 'exact':
                self.assertTrue(row['hold'])
                self.assertIn('reference arithmetic (EXL3: not official-exact)', row['hold'])

    def test_exl3_reference_hold_cannot_be_cleared(self):
        data = registry()
        data['exact_exl3_rules'][0]['hold'] = ''
        with self.assertRaisesRegex(ValueError, 'held EXL3'):
            GENERATOR.validate_registry(data)

    def test_reference_rule_cannot_select_current_arithmetic(self):
        data = registry()
        rule = next(rule for rule in data['original_rules'] if rule['match']['exactness'] == ['exact'])
        rule['strategy'] = 'empty'
        with self.assertRaisesRegex(ValueError, 'original reference arithmetic'):
            GENERATOR.validate_registry(data)

    def test_failed_multiple_balanced_choice_stays_held(self):
        data = registry()
        rule = next(rule for rule in data['balanced_exl3_rules']
                    if rule['match'] == dict(category=['code'], exactness=['balanced'],
                                             concurrency=['multiple'], focus=['balanced']))
        self.assertTrue(rule['qualification_candidate'])
        self.assertIn('fails the strict', rule['hold'])
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(data))
        row = next(row for row in rows if row['family'] == 'exl3' and
                   row['values'] == ('code', 'balanced', 'multiple', 'balanced'))
        self.assertEqual(row['hold'], rule['hold'])
        rule['hold'] = ''
        with self.assertRaisesRegex(ValueError, 'complete current-math balanced tuple'):
            GENERATOR.validate_registry(data)

    def test_rejected_thin_policy_cannot_reenter_an_axis(self):
        data = registry()
        rejected = data["rejected_strategies"]["thin_cost_policy1"]
        self.assertEqual(rejected["numerical_class"], "sum_order")
        self.assertTrue(rejected["raw_only"])
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(data))
        for row in rows:
            if row["family"] == "exl3" and row["values"][2] == "single" and not row["hold"]:
                self.assertIn("--draft_policy=0", row["settings"].split("\n"))
                self.assertNotIn("--draft_policy=1", row["settings"].split("\n"))
        data["balanced_exl3_rules"][0]["strategy"] = "thin_cost_policy1"
        with self.assertRaisesRegex(ValueError, "invalid axis selection rule"):
            GENERATOR.validate_registry(data)

    def test_layout_is_one_runtime_argument(self):
        arguments = ["--placement=pp5", "--layout=layers 0..7: card 0; head: card 4"]
        self.assertEqual(GENERATOR.argument_tokens(arguments), arguments)
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        original = next(row for row in rows if row["family"] == "original" and
                        row["values"] == ("code", "balanced", "multiple", "decode"))
        tokens = original["settings"].split("\n")
        layout = [token for token in tokens if token.startswith("--layout=")]
        self.assertEqual(len(layout), 1)
        self.assertIn("; layers 8..14: card 1;", layout[0])

    def test_runtime_argument_delimiter_cannot_be_injected(self):
        for invalid in (["--layout=first\n--placement=tp4_pro"],
                        ["--layout=first\rsecond"], ["--layout=first\0second"],
                        ["--layout="], ["layout=card 0"], [7],
                        "--layout=first\n--placement=tp4_pro"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                GENERATOR.argument_tokens(invalid)

    def test_legacy_flags_keep_their_argument_values(self):
        self.assertEqual(GENERATOR.argument_tokens("--rows=8 --draft_policy=0"),
                         ["--rows=8", "--draft_policy=0"])

    def test_boundary_receipts_prove_only_parser_semantics(self):
        budget, strategy = synthetic_fixture()
        GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)
        text = GENERATOR.lossy_evidence(strategy, budget)
        self.assertIn("SYNTHETIC FIXTURE", text)
        for metric in GENERATOR.METRIC_DENOMINATORS:
            self.assertIn(metric + "=", text)
        self.assertIn("whole-workload", text)

    def test_synthetic_receipt_cannot_qualify_normal_generation(self):
        budget, strategy = synthetic_fixture()
        with self.assertRaisesRegex(ValueError, "synthetic"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget)

    def test_missing_knob_receipts_are_rejected(self):
        budget, strategy = synthetic_fixture()
        strategy["quality_receipts"] = {}
        with self.assertRaisesRegex(ValueError, "per-lossy-knob"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_missing_stratum_is_rejected(self):
        budget, strategy = synthetic_fixture()
        strategy["quality_receipts"][KNOB].pop()
        with self.assertRaisesRegex(ValueError, "strata"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_each_above_bound_metric_is_rejected(self):
        for limit_name, bound in registry()["quality_budget"]["limits"].items():
            with self.subTest(limit=limit_name):
                budget, strategy = synthetic_fixture()
                row = strategy["quality_receipts"][KNOB][0]
                metric, direction = limit_name.rsplit("_", 1)
                value = bound - 0.001 if direction == "min" else bound + 0.001
                if metric in GENERATOR.AGREEMENT_COUNTS:
                    agreement(row, metric, value)
                else:
                    row["metrics"][metric] = value
                row["metrics"]["worst_KL_nats"] = max(0.02, row["metrics"]["p99_KL_nats"])
                with self.assertRaisesRegex(ValueError, "quality limit exceeded"):
                    GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_nonfinite_metric_is_rejected(self):
        budget, strategy = synthetic_fixture()
        strategy["quality_receipts"][KNOB][0]["metrics"]["mean_KL_nats"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_missing_required_metric_is_rejected(self):
        budget, strategy = synthetic_fixture()
        del strategy["quality_receipts"][KNOB][0]["metrics"]["PPL_ratio_vs_balanced"]
        with self.assertRaisesRegex(ValueError, "missing quality value"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_zero_denominator_is_rejected(self):
        budget, strategy = synthetic_fixture()
        strategy["quality_receipts"][KNOB][0]["denominators"]["ppl_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "denominator"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_mismatched_source_identity_is_rejected(self):
        budget, strategy = synthetic_fixture()
        strategy["quality_receipts"][KNOB][0]["candidate"]["input_sha256"] = digest("different input")
        with self.assertRaisesRegex(ValueError, "unpaired numerical source"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_agreement_must_match_its_recorded_denominator(self):
        budget, strategy = synthetic_fixture()
        strategy["quality_receipts"][KNOB][0]["agreement_counts"]["top1_matches"] -= 1
        with self.assertRaisesRegex(ValueError, "does not match its denominator"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_lossy_mapping_cannot_enter_balanced_or_exact(self):
        for tier in ("balanced", "exact"):
            with self.subTest(tier=tier):
                data = registry()
                data["quality_budget"], data["strategies"]["synthetic_lossy"] = synthetic_fixture()
                rule = {"match": {"category": ["code"], "concurrency": ["single"],
                                  "focus": ["balanced"]}, "strategy": "synthetic_lossy"}
                data[tier + "_exl3_rules"] = [rule]
                with self.assertRaisesRegex(ValueError, "lossy strategy in " + tier):
                    GENERATOR.validate_registry(data, allow_synthetic=True)

    def test_selected_lossy_tuple_cannot_omit_its_error_figures(self):
        data = registry()
        data["quality_budget"], data["strategies"]["synthetic_lossy"] = synthetic_fixture()
        next(tier for tier in data["tiers"] if tier["name"] == "fast")["hold"] = ""
        data["fast_exl3_rules"] = [{"match": {"category": ["code", "prose", "structured", "balanced"],
                                                   "concurrency": ["single", "multiple"],
                                                   "focus": ["balanced", "ttft", "decode", "prefill"]},
                                           "strategy": "synthetic_lossy"}]
        GENERATOR.validate_registry(data, allow_synthetic=True)
        choices = [row for row in GENERATOR.make_choices(data) if row["values"][1] == "fast" and not row["hold"]]
        self.assertEqual(len(choices), 32)
        for choice in choices:
            self.assertIn("lossy;", choice["error_class"])
            self.assertIn("SYNTHETIC FIXTURE", choice["evidence"])
            for metric in GENERATOR.METRIC_DENOMINATORS:
                self.assertIn(metric + "=", choice["evidence"])

    def test_numeric_settings_respect_command_admission(self):
        inventory = {entry["name"]: entry["commands"]
                     for entry in json.loads((HERE / "inventory.json").read_text())}
        for name, strategy in registry()["strategies"].items():
            for command in ("generate", "serve"):
                base = strategy.get("generate_settings") if command == "generate" else None
                arguments = GENERATOR.argument_tokens(base or strategy["settings"])
                if command == "serve":
                    arguments += GENERATOR.argument_tokens(strategy["serve_settings"])
                for argument in arguments:
                    knob = argument[2:].split("=", 1)[0]
                    if knob in inventory:
                        with self.subTest(strategy=name, command=command, knob=knob):
                            self.assertTrue("TAKEN_BY_ALL" in inventory[knob] or
                                            "TAKEN_BY_" + command.upper() in inventory[knob])

    def test_raw_only_candidate_cannot_enter_active_tuple(self):
        strategy = copy.deepcopy(registry()["rejected_strategies"]["thin_cost_policy1"])
        strategy["raw_only"] = True
        with self.assertRaisesRegex(ValueError, "raw-only"):
            GENERATOR.validate_choice_contract(strategy, "balanced", "")

    def test_clearing_fast_hold_requires_admitted_budget(self):
        data = registry()
        next(tier for tier in data["tiers"] if tier["name"] == "fast")["hold"] = ""
        with self.assertRaisesRegex(ValueError, "cannot clear fast hold"):
            GENERATOR.validate_registry(data)

    def test_no_whole_workload_gain_is_rejected(self):
        budget, strategy = synthetic_fixture()
        strategy["quality_receipts"][KNOB][0]["whole_workload"]["candidate_seconds"] = 10
        with self.assertRaisesRegex(ValueError, "whole-workload gain"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)

    def test_available_reference_comparison_is_required(self):
        budget, strategy = synthetic_fixture()
        budget["reference_available"] = True
        with self.assertRaisesRegex(ValueError, "reference comparison"):
            GENERATOR.validate_lossy_strategy("synthetic", strategy, budget, allow_synthetic=True)


if __name__ == "__main__":
    unittest.main()
