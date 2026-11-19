#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic receipt/parser controls only; no model-quality or GPU measurement."""
import copy
import hashlib
import importlib.util
import json
import re
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


def startup_bounds(data):
    """startup_axes.c's axis_bounds as generate_axes.py writes them, in declared order: a cap one entry a bounded
    knob, at that knob's parser least; a floor one entry, at its own least."""
    return [dict(bound, bounded=keys, least=bound["least"] if bound["kind"] == "floor" else
                 GENERATOR.parser_least(keys[0])) for bound in data["cross_axis_bounds"]
            for keys in ([bound["bounded"]] if bound["kind"] == "floor" else [[key] for key in bound["bounded"]])]


def leading_number(text):
    """strtoul's decimal value of a knob's text (0 where it starts with no digit)."""
    digits = re.match(r"\d*", text).group()
    return int(digits) if digits else 0


def bound_value(bound, value, bounding, under):
    """startup_axes.c bound_value: a cap lowers a value to the bounding knob's less the offset, never below its least;
    a floor raises it to its least where the bounding knob holds the floor's value and the under knob is not 0."""
    if bounding is None or not bounding[:1].isdigit():
        return value
    limit = leading_number(bounding)
    if bound["kind"] == "cap":
        return min(value, max(limit - bound["offset"] if limit >= bound["offset"] else 0, bound["least"]))
    raises = limit == int(bound["value"]) and under is not None and leading_number(under) != 0
    return bound["least"] if raises and value < bound["least"] else value


def startup_value(row, serving, key, bounds):
    """The value startup gives an inserted knob of a row without explicit knobs (startup_axes.c bound_setting): the
    first bound, in declared order, that changes the composition's unclamped value, each bounding knob read after its
    own first changing bound (bounding_final_value); else that unclamped value."""
    settings = (row["settings"] + "\n" + row["serve_settings"] if serving else
                row["generate_settings"] or row["settings"])
    composed = {GENERATOR.option_key(word): word.split("=", 1)[1] for word in settings.splitlines() if "=" in word}
    clamps = {GENERATOR.option_key(line.split()[0]): line.split()[1]
              for line in row["clamps" if serving else "generate_clamps"].splitlines()}

    def first_change(name, value, read):
        for bound in (bound for bound in bounds if name in bound["bounded"]):
            changed = bound_value(bound, value, read(bound["bound"]),
                                  read(bound["under"]) if bound["kind"] == "floor" else None)
            if changed != value:
                return changed
        return value

    def bounding_final(name):
        text = composed.get(name)
        return text if text is None or not text[:1].isdigit() else str(first_change(name, int(text), composed.get))
    return first_change(key, int(clamps.get(key, f"{key}={composed[key]}").split("=", 1)[1]), bounding_final)


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
    def test_banked_recipe_is_retained_without_default_selection(self):
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        self.assertFalse(any('--original_head_offload=2' in row[field].splitlines()
                             for row in rows for field in GENERATOR.STRATEGY_FIELDS))
        strategy = registry()['strategies']['original_pp_banked_single']
        self.assertTrue(strategy['raw_only'])
        for field in ('settings', 'generate_settings'):
            tokens = GENERATOR.argument_tokens(strategy[field])
            for token in ('--draft=dspark', '--original_head_offload=2', '--draft_policy=1', '--draft_session_acceptance=6'):
                self.assertIn(token, tokens)
            self.assertNotIn('--draft_policy=0', tokens)

    @staticmethod
    def recipe(name, field, drop=('--prefix_reuse',)):
        strategy = registry()['strategies'][name]
        return [word for word in GENERATOR.argument_tokens(strategy[field])
                if GENERATOR.option_key(word) not in drop]

    @staticmethod
    def stage2_draft(split, category, concurrency, focus):
        """The draft sets on the Original splits: category=code's depth 7 (clamped to rows - 1 where 6 rows apply: PP5's
        concurrency=single and TP4_PRO's focus=decode), focus=decode's arm 1 and, on TP4_PRO, its 6 rows."""
        tp = split == 'tp4_pro'
        rows = 6 if (tp and focus == 'decode') or (split == 'pp5' and concurrency == 'single') else 8
        depth = min(7, rows - 1)
        code = [f'--draft_block={depth}', f'--draft_positions={depth}'] if category == 'code' else []
        decode = (['--draft_policy=1', '--draft_session_acceptance=6'] + (['--rows=6'] if tp else [])
                  if focus == 'decode' else [])
        return code, decode

    @staticmethod
    def base(split):
        return GENERATOR.split_base(registry(), split)

    def test_original_rows_compose_the_base_and_the_sets(self):
        """Stage 3: each Original row is its sets in axis order (category, concurrency, focus), then the split's base
        for each knob no set gives; PP5 multiple keeps its recipe (with its yields), which gives every base knob."""
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        original = [row for row in rows if row['family'] == 'original' and row['values'][1] != 'exact'
                    and not row['hold']]
        self.assertEqual(len(original), 160)  # PP5: 32 single + 32 multiple + 32 balanced; TP4_PRO: 32 + 32
        for row in original:
            category, tier, concurrency, focus = row['values']
            code, decode = self.stage2_draft(row['split'], category, concurrency, focus)
            settings = row['settings'].splitlines()
            if (row['split'], concurrency) == ('pp5', 'multiple'):
                yielded = ('--prefix_reuse', '--session_slots', *{GENERATOR.option_key(word) for word in code + decode})
                self.assertEqual(settings, code + self.recipe('original_pp_depth8', 'settings', yielded) + decode)
                self.assertEqual(row['serve_settings'].splitlines(),
                                 self.recipe('original_pp_depth8', 'serve_settings', yielded))
                continue
            middle = ['--rows=6'] if (row['split'], concurrency) == ('pp5', 'single') else []
            given = {GENERATOR.option_key(word) for word in code + middle + decode}
            base = [word for word in self.base(row['split'])['settings'] if GENERATOR.option_key(word) not in given]
            self.assertEqual(settings, code + middle + decode + base, row['values'])
            # no set pins the context or the slots (9 Nov): the raw defaults serve as many 1M sessions as fit
            self.assertEqual(row['serve_settings'], '', row['values'])
            self.assertEqual('--draft_policy=0' in settings, row['split'] == 'tp4_pro' and focus != 'decode')
            self.assertEqual('--draft_session_acceptance=6' in settings, focus == 'decode')

    def test_original_rows_pin_the_card_order(self):
        """The TP model needs the SM_120 PRO at stage 4, and the engine's default order 0,1,2,3,4 puts it at ordinal 2
        on the reference workstation, so every composed Original row pins the card order its routes measured: the
        base, the recipes and the reference all carry it (stage 3 (b) will choose it by capability where it is
        omitted)."""
        pinned = '--card_order=0,1,3,4,2'
        for row in GENERATOR.make_choices(GENERATOR.validate_registry(registry())):
            if row['hold'] or row['split'] == 'tp4_exl3':
                continue
            for field in ('settings', 'generate_settings'):
                self.assertIn(pinned, (row[field] or row['settings']).splitlines(), (row['split'], row['values']))
        self.assertFalse(any(word.startswith('--card_order=') for word in self.base('tp4_exl3')['settings']))

    def test_single_user_decode_rows_are_the_measured_arms(self):
        """PP5 single/decode is r7c g3 (6 rows at the default pipeline depth; the serving slots are the raw default,
        as many as fit, since r7c g3's 8 is no longer inserted) with arm 1; TP4_PRO
        single/decode is r5s (the single recipe as the base, arm 1 with 6 rows); multiple is held on TP4_PRO."""
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        for category in ('code', 'prose', 'structured', 'balanced'):
            for tier in ('balanced', 'fast'):
                values = (category, tier, 'single', 'decode')
                pp5 = rows['pp5', values]['settings'].splitlines()
                self.assertTrue({'--rows=6', '--draft_policy=1', '--draft_session_acceptance=6', '--speculative=2',
                                 '--draft_positions=5'} <= set(pp5))
                self.assertFalse(any(word.startswith('--pipeline_steps=') for word in pp5))
                self.assertEqual(rows['pp5', values]['serve_settings'], '')
                tp = rows['tp4_pro', values]['settings'].splitlines()
                self.assertTrue({'--rows=6', '--draft_policy=1', '--pipeline_steps=2', '--draft_positions=5'} <= set(tp))
                self.assertEqual(rows['tp4_pro', values]['serve_settings'], '')
                self.assertIn('held', rows['tp4_pro', (category, tier, 'multiple', 'decode')]['status'])

    def test_a_recipe_keeps_a_yielded_knob_where_no_other_set_gives_it(self):
        """A recipe yields a knob to the axis that owns it: its measured value runs wherever that axis's set of the row
        gives none. PP5's depth-8 recipe keeps its fixed policy 0 (no CLI default: the Original placements default to
        the cost-priced depth), and so does TP4_PRO's base. A yield names a knob the recipe keeps and another axis's
        set gives."""
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        for split, concurrency in (('pp5', 'multiple'), ('tp4_pro', 'single')):
            for category in ('balanced', 'prose', 'structured'):
                for focus in ('balanced', 'ttft', 'prefill'):
                    row = rows[split, (category, 'balanced', concurrency, focus)]
                    self.assertIn('--draft_policy=0', row['settings'].splitlines())
            decode = rows[split, ('prose', 'balanced', concurrency, 'decode')]['settings'].splitlines()
            self.assertNotIn('--draft_policy=0', decode)
            self.assertIn('--draft_policy=1', decode)
        for word in ('--port', '--prefix_reuse'):
            data = registry()
            data['splits']['pp5']['sets']['concurrency']['multiple']['yields'].append(word)
            with self.assertRaisesRegex(ValueError, 'yields only knobs it keeps'):
                GENERATOR.validate_registry(data)
        # A yielded knob has its recipe and exactly one other axis: none refuses, and two refuse as
        # ever.
        data = registry()
        del data['splits']['pp5']['sets']['category']['code']
        with self.assertRaisesRegex(ValueError, 'only to the one other axis'):
            GENERATOR.validate_registry(data)
        data = registry()
        data['splits']['pp5']['sets']['focus']['ttft'] = {'settings': ['--draft_block=6'],
                                                          'numerical_class': 'draft_only', 'error_class': 'x',
                                                          'evidence': 'x'}
        with self.assertRaisesRegex(ValueError, 'is set by both category and focus'):
            GENERATOR.validate_registry(data)

    def test_all_balanced_rows_insert_the_base(self):
        """Stage 3: an all-balanced tuple, and so a run without axis flags, is the split's base (its measured route over
        the raw defaults). The owner's concurrency=balanced (every session at 1M context, as many as fit) is the raw
        defaults since U5a, so a balanced set is empty or a repeat of a measured value's set."""
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        for split in GENERATOR.SPLITS:
            row = rows[split, ('balanced',) * 4]
            self.assertFalse(row['hold'])
            self.assertEqual(row['settings'].splitlines(), self.base(split)['settings'])
            self.assertEqual((row['serve_settings'], row['generate_settings']), ('', ''))
        self.assertEqual([GENERATOR.option_key(word) for word in self.base('pp5')['settings']],
                         ['--card_order', '--rows', '--wide_rows', '--prefill_cone', '--step_graphs', '--speculative',
                          '--draft_positions', '--layout'])
        for axis, entry in (('concurrency', {'serve_settings': ['--session_slots=8'], 'settings': [],
                                             'numerical_class': 'same_bits', 'error_class': 'x', 'evidence': 'x'}),
                            ('focus', {'settings': [], 'serve_settings': ['--context_tokens=1048576'],
                                       'numerical_class': 'same_bits', 'error_class': 'x', 'evidence': 'x'})):
            data = registry()
            data['splits']['pp5']['sets'][axis] = dict(data['splits']['pp5']['sets'].get(axis, {}), balanced=entry)
            with self.assertRaisesRegex(ValueError, 'a balanced set stays empty'):
                GENERATOR.validate_registry(data)
        for split, axis, value, target in (('tp4_pro', 'concurrency', 'multiple', 'balanced'),
                                           ('pp5', 'focus', 'balanced', 'missing')):
            data = registry()
            data['splits'][split]['holds'].get(axis, {}).pop(value, None)
            data['splits'][split]['sets'].setdefault(axis, {})[value] = {'same_as': target, 'evidence': 'x'}
            with self.assertRaisesRegex(ValueError, 'only a balanced set may repeat'):
                GENERATOR.validate_registry(data)

    def test_no_all_balanced_row_selects_batch(self):
        """Unbounded sessions (9 Nov): serve's default is --session_slots=max, and a BATCH serve needs exactly 8
        sessions, so no split's all-balanced row (the run without axis flags) may carry a BATCH knob; it serves thin
        steps for as many sessions as fit."""
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        for split in GENERATOR.SPLITS:
            row = rows[split, ('balanced',) * 4]
            for words in (row['settings'], row['serve_settings'], row['generate_settings']):
                for word in words.splitlines():
                    key, _, value = word.partition('=')
                    self.assertFalse(key == '--batch_rows' or (key == '--original_batch' and value != '0'),
                                     f'{split}: the all-balanced row selects BATCH ({word})')

    def test_a_base_declares_each_route_knob(self):
        """Every knob of a base route is excluded or declares its raw engine default; a pin equal to it is omitted."""
        for split in GENERATOR.SPLITS:
            omitted = dict(self.base(split)['omitted'])
            self.assertEqual(omitted['--draft_block'], '5')
            self.assertNotIn('--card_order', omitted)
        data = registry()
        del data['splits']['pp5']['base']['raw_defaults']['--rows']
        with self.assertRaisesRegex(ValueError, 'needs its raw default or an exclusion'):
            GENERATOR.validate_registry(data)
        data = registry()
        data['splits']['pp5']['base']['route'] = 'original_tp_serial2'
        with self.assertRaisesRegex(ValueError, 'a route of its own split'):
            GENERATOR.validate_registry(data)

    def test_a_cross_axis_bound_clamps_every_row_it_meets(self):
        """A declared bound is applied by clamping, per command: 6 rows (TP4_PRO's focus=decode, PP5's
        concurrency=single) clamp category=code's depth 7 to 5 (r5c lost 10.2 % unclamped); the clamp is in the row's
        argv, its clamp lines and its evidence."""
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        clamped = {key for key, row in rows.items() if any('bounded by' in note for note in row['bounds'])}
        sources = {('tp4_pro', ('code', tier, concurrency, 'decode')): 'focus=decode'
                   for tier in ('balanced', 'fast') for concurrency in ('single', 'balanced')}
        sources.update({('pp5', ('code', tier, 'single', focus)): 'concurrency=single'
                        for tier in ('balanced', 'fast') for focus in ('balanced', 'ttft', 'decode', 'prefill')})
        self.assertEqual(clamped, set(sources))
        knobs = ('--draft_block', '--draft_positions')
        for key in clamped:
            row = rows[key]
            notes = [f'{knob} 7 -> 5 (bounded by --rows=6 from {sources[key]})' for knob in knobs]
            self.assertEqual(row['bounds'], notes)
            self.assertIn('Cross-axis bound applied: ' + '; '.join(notes) + '.', row['evidence'])
            self.assertEqual(row['settings'].splitlines()[:2], ['--draft_block=5', '--draft_positions=5'])
            for field in ('clamps', 'generate_clamps'):
                self.assertEqual(row[field].splitlines(),
                                 [f'{knob}=5 {knob}=7 --rows=6 {sources[key]} bounded' for knob in knobs])
        self.assertEqual(rows['tp4_pro', ('code', 'balanced', 'single', 'ttft')]['settings'].splitlines()[:2],
                         ['--draft_block=7', '--draft_positions=7'])
        self.assertEqual(rows['pp5', ('code', 'balanced', 'multiple', 'decode')]['settings'].splitlines()[:2],
                         ['--draft_block=7', '--draft_positions=7'])
        self.assertFalse(rows['tp4_pro', ('prose', 'balanced', 'single', 'decode')]['clamps'])

    def test_a_clamp_is_per_command_and_never_inside_one_source(self):
        data = registry()
        del data['splits']['pp5']['sets']['concurrency']['single']  # its rows would meet the synthetic focus rows
        multiple = data['splits']['pp5']['sets']['concurrency']['multiple']
        multiple['drop'].append('--rows')
        # Serve-only rows: serve clamps the shared depth, generate keeps it in its own argv.
        data['splits']['pp5']['sets']['focus']['ttft'] = {'settings': [], 'serve_settings': ['--rows=4'],
                                                          'numerical_class': 'draft_only', 'error_class': 'x',
                                                          'evidence': 'x'}
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(data))}
        row = rows['pp5', ('code', 'balanced', 'multiple', 'ttft')]
        self.assertEqual(row['settings'].splitlines()[:2], ['--draft_block=3', '--draft_positions=3'])
        self.assertEqual(row['generate_settings'].splitlines()[:2], ['--draft_block=7', '--draft_positions=7'])
        self.assertEqual(len(row['clamps'].splitlines()), 2)
        self.assertEqual(row['generate_clamps'], '')
        # One source that exceeds its own bound is a registry error, not a clamp.
        data['splits']['pp5']['sets']['category'] = {}
        multiple['yields'].remove('--draft_block')  # no other axis sets it now
        data['splits']['pp5']['sets']['focus']['ttft'] = {'settings': ['--rows=4', '--draft_positions=7'],
                                                          'numerical_class': 'draft_only', 'error_class': 'x',
                                                          'evidence': 'x'}
        with self.assertRaisesRegex(ValueError, 'above its own'):
            GENERATOR.make_choices(GENERATOR.validate_registry(data))

    def test_a_floor_raises_draft_layers_under_sharded_batch_draft_rows(self):
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        raised = {key for key, row in rows.items() if any('required by' in note for note in row['bounds'])}
        self.assertEqual(raised, {('tp4_exl3', (category, tier, 'multiple', focus))
                                  for category in ('code', 'prose', 'structured', 'balanced')
                                  for tier in ('balanced', 'fast') for focus in ('balanced', 'ttft', 'prefill')})
        for key in raised:
            for field in ('clamps', 'generate_clamps'):
                self.assertEqual(rows[key][field],
                                 '--tp_draft_layers=1 --tp_draft_layers=0 --tp_draft_shards=1 concurrency=multiple '
                                 'required')
        decode = rows['tp4_exl3', ('prose', 'balanced', 'multiple', 'decode')]
        self.assertFalse(decode['bounds'])
        self.assertEqual(decode['settings'].splitlines().count('--tp_draft_layers=1'), 1)
        self.assertNotIn('--tp_draft_layers=1', rows['tp4_exl3', ('prose', 'balanced', 'single', 'ttft')]['settings'])
        data = registry()
        data['cross_axis_bounds'][1]['default'] = 1
        with self.assertRaisesRegex(ValueError, 'a cross-axis bound is a cap'):
            GENERATOR.validate_registry(data)

    def test_the_shard_caps_clamp_nothing_at_composition_and_refuse_a_chain(self):
        """The two shard caps (--tp_head_shards bounds --tp_draft_shards, --tp_draft_shards bounds --tp_draft_layers)
        act at startup on an explicit 0 alone: every TP4_EXL3 row inserts head and draft shards 1, so no row is clamped
        by them and the floor's rows stay as they are; startup reads a bounding knob after its own bounds one level
        deep, so a longer chain, two caps of one knob, and a composition clamp of a knob another bound reads refuse."""
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        self.assertFalse([row for row in rows if any('--tp_head_shards=' in note or 'bounded by --tp_draft_shards' in
                                                     note for note in row['bounds'])])
        for row in rows:
            if '--tp_head_shards=1' in row['settings'].splitlines():
                self.assertIn('--tp_draft_shards=1', row['settings'].splitlines())
        data = registry()
        data['cross_axis_bounds'].append({'kind': 'cap', 'bound': '--rows', 'bounded': ['--tp_head_shards'],
                                          'offset': 7, 'rule': 'x'})
        with self.assertRaisesRegex(ValueError, 'longer than one through --tp_draft_shards'):
            GENERATOR.validate_registry(data)
        data = registry()
        data['cross_axis_bounds'].append({'kind': 'cap', 'bound': '--rows', 'bounded': ['--tp_draft_shards'],
                                          'offset': 7, 'rule': 'x'})
        with self.assertRaisesRegex(ValueError, 'a knob two caps bound'):
            GENERATOR.validate_registry(data)
        data = registry()
        data['cross_axis_bounds'].append({'kind': 'floor', 'bound': '--tp_head_shards', 'value': '1',
                                          'under': '--tp_head_stream', 'bounded': ['--batch_draft_rows'],
                                          'least': 2, 'default': 0, 'rule': 'x'})
        with self.assertRaisesRegex(ValueError, 'clamps --batch_draft_rows, which another bound reads'):
            GENERATOR.make_choices(GENERATOR.validate_registry(data))

    def test_startup_takes_the_first_bound_that_changes_a_knob_in_any_declared_order(self):
        """Startup runs every row's composition (startup_value: the first declared bound that changes a knob, else its
        unclamped value) with the bounds as declared; with the layers' draft-shards cap moved ahead of the BATCH floor,
        whose rows and clamps are the same and whose raised layers startup keeps; and with draft shards 0 in TP4_EXL3's
        base, where composition clamps focus=decode's layers to 0 and startup keeps the 0."""
        declared, reordered, unsharded = registry(), registry(), registry()
        bounds = reordered["cross_axis_bounds"]
        floor = next(index for index, bound in enumerate(bounds) if bound["kind"] == "floor")
        cap = next(index for index, bound in enumerate(bounds) if bound["kind"] == "cap" and
                   bound["bounded"] == ["--tp_draft_layers"])
        self.assertLess(floor, cap)
        bounds.insert(floor, bounds.pop(cap))
        for field in ("settings", "generate_settings"):
            strategy = unsharded["strategies"]["thin_fixed"]
            strategy[field] = strategy[field].replace("--tp_draft_shards=1", "--tp_draft_shards=0")
        rows = {}
        for name, data in (("declared", declared), ("reordered", reordered), ("unsharded", unsharded)):
            rows[name] = GENERATOR.make_choices(GENERATOR.validate_registry(data))
            startup = startup_bounds(data)
            for row in rows[name]:
                for serving in (True, False):
                    settings = (row["settings"] + "\n" + row["serve_settings"] if serving else
                                row["generate_settings"] or row["settings"])
                    for word in settings.splitlines():
                        key = GENERATOR.option_key(word)
                        if any(key in bound["bounded"] for bound in startup):
                            self.assertEqual(startup_value(row, serving, key, startup), int(word.split("=", 1)[1]),
                                             (name, row["split"], row["values"], serving, word))
        self.assertEqual(rows["reordered"], rows["declared"])
        batch = {(row["split"], row["values"]): row for row in rows["reordered"]}[
            "tp4_exl3", ("prose", "balanced", "multiple", "ttft")]
        self.assertIn("--tp_draft_layers=1", batch["settings"].splitlines())
        decode = {(row["split"], row["values"]): row for row in rows["unsharded"]}[
            "tp4_exl3", ("code", "balanced", "single", "decode")]
        self.assertIn("--tp_draft_layers=0", decode["settings"].splitlines())
        for field in ("clamps", "generate_clamps"):
            self.assertIn("--tp_draft_layers=0 --tp_draft_layers=1 --tp_draft_shards=0 base=tp4_exl3 bounded",
                          decode[field].splitlines())

    def test_a_set_applies_only_with_existing_axis_values_and_is_left_out_elsewhere(self):
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        for category in ('prose', 'structured'):
            for concurrency in ('single', 'balanced'):
                row = rows['tp4_exl3', (category, 'balanced', concurrency, 'ttft')]
                self.assertFalse(row['hold'])
                self.assertNotIn('--batch_step_us', row['settings'] + row['generate_settings'])
                self.assertNotIn(f'category={category}', row['sources'])
            batch = rows['tp4_exl3', (category, 'balanced', 'multiple', 'ttft')]
            self.assertIn('--batch_step_us=19500,', batch['settings'])
        for applies_with in ({'concurrency': ['many']}, {'width': ['multiple']}, {'category': ['code']}, {'focus': []}):
            data = registry()
            data['splits']['tp4_exl3']['sets']['category']['prose']['applies_with'] = applies_with
            with self.assertRaisesRegex(ValueError, 'apply only with'):
                GENERATOR.validate_registry(data)

    def test_draft_sets_without_speculation_are_marked_inactive(self):
        """Stage 3: every base carries DSpark, so no table row is inactive by composition; every line of draft knobs
        still carries the note, which startup prints only where the final speculation is off. A
        base without DSpark (synthetic) makes the draft sets of the rows it serves inactive."""
        table = {(row['split'], row['values']): row for row in
                 GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        self.assertFalse([key for key, row in table.items() if row['inactive']])
        for row in table.values():
            for line in row['sources'].splitlines() + row['generate_sources'].splitlines():
                knobs = line.split(': ', 1)[1].replace(GENERATOR.INACTIVE_DRAFT_NOTE, '').split()
                self.assertEqual(line.endswith(GENERATOR.INACTIVE_DRAFT_NOTE),
                                 bool(knobs) and all(knob.startswith('--draft_') for knob in knobs), line)
        self.assertEqual(table['pp5', ('code', 'balanced', 'single', 'decode')]['sources'].splitlines()[0],
                         'category=code: --draft_block --draft_positions ' + GENERATOR.INACTIVE_DRAFT_NOTE)
        self.assertNotIn('inactive', table['tp4_pro', ('code', 'balanced', 'single', 'decode')]['sources'].splitlines()[1])
        data = registry()
        data['splits']['pp5']['base']['exclude'].append('--speculative')
        del data['splits']['pp5']['sets']['concurrency']['single']
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(data))}
        inactive = {key for key, row in rows.items() if row['inactive']}
        self.assertTrue(inactive)
        self.assertTrue(all(split == 'pp5' and values[2] in ('single', 'balanced') for split, values in inactive))
        row = rows['pp5', ('code', 'balanced', 'single', 'decode')]
        self.assertEqual(row['sources'].splitlines()[:2],
                         ['category=code: --draft_block --draft_positions ' + GENERATOR.INACTIVE_DRAFT_NOTE,
                          'focus=decode: --draft_policy --draft_session_acceptance ' + GENERATOR.INACTIVE_DRAFT_NOTE])
        self.assertNotIn(('pp5', ('code', 'balanced', 'multiple', 'decode')), inactive)
        self.assertNotIn(('tp4_pro', ('code', 'balanced', 'balanced', 'decode')), inactive)

    def test_exact_never_runs_the_relaxed_base(self):
        """TP4_EXL3's base runs relaxed numerics and the BF16 wire (9 Nov: kept in its no-axis default). Its exact
        rows run the exact-shaping set instead (owner, 6 Nov: same shaping and strictness, not official-exact; q4's
        measured exact-FP32 arm), labelled so, and an exact recipe on any split gives --numerics=exact and the FP32
        wire."""
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        exl3_exact = [row for row in rows if row['split'] == 'tp4_exl3' and row['values'][1] == 'exact'
                      and row['values'][0] != 'vision']
        self.assertEqual(len(exl3_exact), 48)
        for row in exl3_exact:
            if row['values'][2] != 'single':
                self.assertEqual(row['hold'], GENERATOR.EXACT_SESSION_HOLD)
                continue
            self.assertFalse(row['hold'])
            self.assertEqual(row['strategy'], 'exl3_exact_shaping')
            self.assertEqual(row['settings'].splitlines(), ['--placement=tp4_exl3', '--numerics=exact',
                                                            '--tp_wire_bf16=0', '--speculative=2', '--rows=8'])
            self.assertEqual(row['serve_settings'].splitlines(), ['--session_slots=1'])
            self.assertEqual(row['error_class'], 'exact shaping on quantized weights (not official-exact)')
        for row in rows:
            if row['values'][1] == 'exact' and not row['hold']:
                words = set(row['settings'].splitlines())
                self.assertIn('--numerics=exact', words)
                self.assertFalse(words & {'--numerics=relaxed', '--tp_wire_bf16=1'})
        for split, change in (('pp5', '--numerics=relaxed'), ('pp5', '--tp_wire_bf16=1'),
                              ('tp4_exl3', '--numerics=relaxed'), ('tp4_exl3', '--tp_wire_bf16=1')):
            data = registry()
            strategy = data['strategies'][data['splits'][split]['reference']]
            strategy['settings'] = [word for word in GENERATOR.argument_tokens(strategy['settings'])
                                    if not word.startswith(change.split('=')[0] + '=')] + [change]
            with self.assertRaisesRegex(ValueError, 'exact recipe gives --numerics=exact'):
                GENERATOR.validate_registry(data)

    def test_each_knob_belongs_to_one_axis_per_split(self):
        data = registry()
        data['splits']['tp4_exl3']['sets']['focus']['ttft'] = {'settings': ['--draft_policy=2'],
                                                               'numerical_class': 'same_bits',
                                                               'error_class': 'x', 'evidence': 'x'}
        with self.assertRaisesRegex(ValueError, '--draft_policy is set by both category and focus'):
            GENERATOR.validate_registry(data)
        data = registry()
        data['splits']['pp5']['sets']['focus'] = {'decode': {'settings': ['--pipeline_steps=5'],
                                                             'numerical_class': 'same_bits',
                                                             'error_class': 'x', 'evidence': 'x'}}
        with self.assertRaisesRegex(ValueError, '--pipeline_steps is set by both concurrency and focus'):
            GENERATOR.validate_registry(data)

    def test_generate_argv_keeps_the_split_and_one_owner_per_knob(self):
        data = registry()
        data['strategies']['generate_elsewhere'] = dict(data['strategies']['batch_adaptive'])
        data['strategies']['generate_elsewhere']['generate_settings'] = [
            '--placement=pp5' if word.startswith('--placement=') else word
            for word in GENERATOR.argument_tokens(data['strategies']['batch_adaptive']['settings'])]
        data['splits']['tp4_exl3']['sets']['concurrency']['multiple']['strategy'] = 'generate_elsewhere'
        with self.assertRaisesRegex(ValueError, 'must belong to its GPU split'):
            GENERATOR.validate_registry(data)
        data = registry()
        recipe = data['strategies']['original_pp_depth8']
        data['strategies']['generate_rows'] = dict(recipe, generate_settings=GENERATOR.argument_tokens(
            recipe['generate_settings']) + ['--wide_rows_floor=1'])
        data['splits']['pp5']['sets']['concurrency']['multiple']['strategy'] = 'generate_rows'
        data['splits']['pp5']['sets']['focus'] = {'decode': {'settings': ['--wide_rows_floor=1'],
                                                             'numerical_class': 'same_bits',
                                                             'error_class': 'x', 'evidence': 'x'}}
        with self.assertRaisesRegex(ValueError, '--wide_rows_floor is set by both concurrency and focus'):
            GENERATOR.validate_registry(data)

    def test_exl3_sets_compose_by_axis(self):
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        owned = ('--draft_policy', '--batch_core_once', '--tp_draft_layers', '--prefix_reuse', '--batch_step_us',
                 '--session_slots')
        thin, batch = self.base('tp4_exl3')['settings'], self.recipe('batch_adaptive', 'settings', owned)
        menu = [word for word in self.recipe('batch_adaptive', 'settings', ()) if word.startswith('--batch_step_us=')]
        code_batch_decode = rows['tp4_exl3', ('code', 'balanced', 'multiple', 'decode')]
        self.assertEqual(code_batch_decode['settings'].splitlines(), ['--draft_policy=0', *batch, '--tp_draft_layers=1'])
        self.assertEqual(code_batch_decode['sources'].splitlines(),
                         ['category=code: --draft_policy ' + GENERATOR.INACTIVE_DRAFT_NOTE,
                          'concurrency=multiple: ' + ' '.join(
                             GENERATOR.option_key(word) for word in
                             batch + self.recipe('batch_adaptive', 'serve_settings', owned)),
                          'focus=decode: --tp_draft_layers'])
        self.assertEqual(rows['tp4_exl3', ('prose', 'balanced', 'single', 'ttft')]['settings'].splitlines(), thin)
        # concurrency=single and balanced run THIN, the base (balanced adds the owner's serving rule)
        self.assertEqual(rows['tp4_exl3', ('code', 'balanced', 'balanced', 'ttft')]['settings'].splitlines(),
                         ['--draft_policy=0', *thin])
        # BATCH's sharded draft rows need draft layers: a focus without layer drafting is raised to 1.
        prefill = rows['tp4_exl3', ('prose', 'balanced', 'multiple', 'prefill')]
        # prose and structured keep the BATCH step menu; code (fixed prices) and the balanced category have none
        self.assertEqual(prefill['settings'].splitlines(), menu + batch + ['--tp_draft_layers=1'])
        self.assertEqual(prefill['generate_settings'].splitlines()[0], '--batch_step_us=19500')
        for category in ('code', 'balanced'):
            self.assertFalse(any(word.startswith('--batch_step_us=') for row in rows.values()
                                 if row['split'] == 'tp4_exl3' and row['values'][0] == category
                                 for field in ('settings', 'generate_settings')
                                 for word in row[field].splitlines()))
        self.assertFalse(any(word.startswith('--batch_step_us=') for row in rows.values()
                             if row['split'] == 'tp4_exl3' and row['values'][2] != 'multiple'
                             for word in row['settings'].splitlines()))
        self.assertEqual(prefill['bounds'],
                         ['--tp_draft_layers 0 -> 1 (required by --tp_draft_shards=1 from concurrency=multiple)'])
        multiple_line = next(line for line in prefill['sources'].splitlines()
                             if line.startswith('concurrency=multiple: '))
        self.assertIn('--tp_draft_layers', multiple_line.split())
        self.assertEqual(prefill['sources'].splitlines()[0], 'category=prose: --batch_step_us')
        self.assertNotIn('--prefix_reuse=0', code_batch_decode['serve_settings'].splitlines())
        self.assertEqual(code_batch_decode['generate_sources'].splitlines()[1].split(': ')[1].split(),
                         [GENERATOR.option_key(word) for word in
                          self.recipe('batch_adaptive', 'generate_settings', owned)])
        # concurrency=balanced repeats THIN, so layer drafting applies there too
        decode = rows['tp4_exl3', ('prose', 'balanced', 'balanced', 'decode')]
        self.assertEqual(decode['settings'].splitlines(), ['--tp_draft_layers=1', *thin])  # the sets, then the base
        data = registry()
        data['splits']['tp4_exl3']['sets']['focus']['decode']['requires'] = {'concurrency': ['single', 'multiple']}
        held = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(data))}['tp4_exl3', decode['values']]
        self.assertIn('needs --concurrency=single or --concurrency=multiple', held['hold'])

    def test_structured_control_changes_only_session_acceptance(self):
        strategies = registry()['strategies']
        baseline = strategies['original_pp_session6']
        structured = strategies['original_pp_structured0']
        self.assertEqual(structured['serve_settings'], baseline['serve_settings'])
        for field in ('settings', 'generate_settings'):
            expected = [word.replace('--draft_session_acceptance=6', '--draft_session_acceptance=0')
                        for word in baseline[field]]
            self.assertEqual(structured[field], expected)

    @staticmethod
    def route_signature(row):
        def without_modifier(words):
            return [word for word in words.splitlines()
                    if not word.startswith('--draft_session_acceptance=')]
        recipe = {field: without_modifier(row[field])
                  for field in ('settings', 'generate_settings', 'serve_settings')}
        recipe['strategy'] = row['strategy']
        return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @staticmethod
    def session_values(row):
        return [next((word.split('=', 1)[1] for word in row[field].splitlines()
                      if word.startswith('--draft_session_acceptance=')), None)
                for field in ('settings', 'generate_settings')]

    def assert_route_envelope(self, rows):
        baseline = json.loads((HERE / 'axes_route_baseline.json').read_text())
        previous = {(row['split'], tuple(row['values'])): row for row in baseline['rows']}
        self.assertEqual(len(rows), len(previous))
        for row in rows:
            key = (row['split'], tuple(row['values']))
            before = previous[key]
            current = {'held': bool(row['hold']),
                       'route_sha256': None if row['hold'] else self.route_signature(row)}
            changed = any(before[field] != current[field] for field in current)
            if changed:
                receipts = [receipt for receipt in baseline['route_measurements']
                            if receipt['split'] == key[0] and tuple(receipt['values']) == key[1]
                            and receipt['before'] == {field: before[field] for field in current}
                            and receipt['after'] == current]
                self.assertTrue(receipts, 'unmeasured route or hold change: ' + str(key))
                for receipt in receipts:
                    self.assertTrue(receipt['outputs_identical'])
                    self.assertTrue(receipt['scope'])
                    for field in ('evidence_sha256', 'control_binary_sha256', 'candidate_binary_sha256'):
                        self.assertRegex(receipt[field], r'^[0-9a-f]{64}$')
            if not current['held'] and before['session_values'] != self.session_values(row):
                receipts = [receipt for receipt in baseline['modifier_measurements']
                            if receipt['split'] == key[0] and tuple(receipt['values']) == key[1]
                            and receipt['before'] == before['session_values']
                            and receipt['after'] == self.session_values(row)]
                self.assertTrue(receipts, 'unmeasured session modifier: ' + str(key))
                for receipt in receipts:
                    self.assertTrue(receipt['outputs_identical'])
                    self.assertTrue(receipt['scope'])
                    self.assertRegex(receipt['evidence_sha256'], r'^[0-9a-f]{64}$')

    def test_measured_routes_and_holds_need_recorded_measurements(self):
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        self.assert_route_envelope(rows)

    def test_every_stage2_change_names_its_basis(self):
        """Stage 2 re-recorded the envelope: each row it changed keeps its stage-1 values and names a route arm (same
        binary, identical outputs) or unmeasured (covered by the GPU confirmation set). The recipes yield their draft
        knobs, so no recipe row changes without a stage-2 set."""
        baseline = json.loads((HERE / 'axes_route_baseline.json').read_text())
        arms = baseline['route_arms']
        for arm in arms.values():
            self.assertTrue(arm['outputs_identical'] and arm['scope'] and arm['knobs'])
            for field in ('evidence_sha256', 'control_binary_sha256', 'candidate_binary_sha256'):
                self.assertRegex(arm[field], r'^[0-9a-f]{64}$')
        changed = [row for row in baseline['rows'] if 'stage1' in row]
        self.assertTrue(all('basis' not in row for row in baseline['rows'] if 'stage1' not in row))
        self.assertEqual(len(changed), 70)
        for row in changed:
            self.assertIn(row['basis'], [*arms, 'unmeasured'])
            self.assertNotEqual(row['stage1'],
                                {field: row[field] for field in ('held', 'route_sha256', 'session_values')})
            if row['basis'] in arms:
                self.assertEqual(arms[row['basis']]['route'].split(' (')[0],
                                 {'pp5': 'GPU split PP5, pipeline depth 8, serving slots 8',
                                  'tp4_pro': 'GPU split TP4_PRO, pipeline depth 2, serving slots 1'}[row['split']])
                if 'code at' in arms[row['basis']]['scope'] and 'structured' not in arms[row['basis']]['scope']:
                    self.assertEqual(row['values'][0], 'code')
            if row['basis'] == 'unmeasured':  # raw PP5 (TP4_PRO balanced repeats the measured single rows)
                self.assertTrue((row['split'], row['values'][2]) in {('pp5', 'single'), ('pp5', 'balanced')})
        self.assertEqual(sum(row['basis'] == 'unmeasured' for row in changed), 28)

    def test_every_stage3_change_names_its_basis(self):
        """Stage 3 re-recorded the envelope: each row it changed keeps its stage-2 values and names its basis: q4 on
        (TP4_EXL3's exact-shaping set, the measured exact-FP32 arm), r7c g3
        (PP5 single with focus=decode), the same parsed configuration (TP4_EXL3's single rows, whose base parses as
        the measured recipe did, startup_axes_test's raw_default_cases, and whose serve at 1M fits one session and
        takes --session_slots=1's defaults, single_session_cases), or unmeasured (the owner's balanced rule, PP5's
        other single rows and TP4_PRO's single rows, which serve the most sessions that fit; U6)."""
        baseline = json.loads((HERE / 'axes_route_baseline.json').read_text())
        changed = [row for row in baseline['rows'] if 'stage2' in row]
        self.assertTrue(all('stage3_basis' not in row for row in baseline['rows'] if 'stage2' not in row))
        counts = {}
        for row in changed:
            split, (_, _, concurrency, focus) = row['split'], row['values']
            basis = row['stage3_basis']
            counts[basis] = counts.get(basis, 0) + 1
            self.assertNotEqual(row['stage2'], {field: row[field] for field in ('held', 'route_sha256',
                                                                                'session_values')})
            exactness = row['values'][1]
            allowed = {'q4_on': split == 'tp4_exl3' and exactness == 'exact' and concurrency == 'single',
                       'r7c_g3': split == 'pp5' and concurrency == 'single' and focus == 'decode',
                       'same_parsed_configuration': split == 'tp4_exl3' and concurrency == 'single',
                       'unmeasured': concurrency == 'balanced' or (split == 'pp5' and concurrency == 'single' and
                                                                   focus != 'decode') or
                       (split == 'tp4_pro' and concurrency == 'single')}
            self.assertTrue(allowed.get(basis), str((split, row['values'], basis)))
        self.assertEqual(counts, {'q4_on': 16, 'r7c_g3': 8, 'same_parsed_configuration': 32, 'unmeasured': 152})

    def test_every_unbounded_change_names_its_basis(self):
        """Unbounded sessions (9 Nov) re-recorded the envelope: no axis set pins the context or the slots, so the
        concurrency=balanced rows (the owner's rule is the raw defaults), PP5 single and multiple (no inserted 8 slots)
        and TP4_EXL3 multiple (8 slots dropped; BATCH only at exactly 8 on serve) changed. Each keeps its stage-3 values
        and names unbounded_defaults, unmeasured until U6; exact and held rows did not change."""
        baseline = json.loads((HERE / 'axes_route_baseline.json').read_text())
        changed = [row for row in baseline['rows'] if 'stage3' in row]
        self.assertTrue(all('unbounded_basis' not in row for row in baseline['rows'] if 'stage3' not in row))
        counts = {}
        for row in changed:
            split, (_, exactness, concurrency, _) = row['split'], row['values']
            self.assertEqual(row['unbounded_basis'], 'unbounded_defaults')
            self.assertNotEqual(row['stage3'], {field: row[field] for field in ('held', 'route_sha256',
                                                                                'session_values')})
            self.assertFalse(row['held'] or row['stage3']['held'] or exactness == 'exact')
            counts[split, concurrency] = counts.get((split, concurrency), 0) + 1
        self.assertEqual(counts, {('pp5', 'balanced'): 32, ('pp5', 'single'): 32, ('pp5', 'multiple'): 32,
                                  ('tp4_pro', 'balanced'): 32, ('tp4_exl3', 'balanced'): 32,
                                  ('tp4_exl3', 'multiple'): 32})

    def test_no_axis_set_pins_the_context_or_the_slots(self):
        """9 Nov: context is the user's choice (default 1M) and serve takes as many sessions as fit, so no row but
        an exact reference recipe inserts --context_tokens or --session_slots, and a set that tries is refused."""
        for row in GENERATOR.make_choices(GENERATOR.validate_registry(registry())):
            if row['hold'] or row['values'][1] == 'exact':
                continue
            for field in ('settings', 'serve_settings', 'generate_settings'):
                self.assertFalse(any(GENERATOR.option_key(word) in GENERATOR.SESSION_KNOBS
                                     for word in row[field].splitlines()), (row['split'], row['values'], field))
        for split, axis, value, entry in (
                ('pp5', 'concurrency', 'single', dict(settings=['--rows=6'], serve_settings=['--session_slots=8'],
                                                      numerical_class='draft_only', error_class='x', evidence='x')),
                ('tp4_pro', 'focus', 'decode', dict(settings=['--context_tokens=65536'], numerical_class='draft_only',
                                                    error_class='x', evidence='x'))):
            data = registry()
            data['splits'][split]['sets'][axis][value] = entry
            with self.assertRaisesRegex(ValueError, 'pins no context or session count'):
                GENERATOR.validate_registry(data)
        data = registry()
        data['splits']['pp5']['sets']['concurrency']['multiple']['drop'].remove('--session_slots')
        with self.assertRaisesRegex(ValueError, 'pins no context or session count'):
            GENERATOR.validate_registry(data)

    def test_only_exact_rows_keep_their_reference_geometry(self):
        """9 Nov: the one exception to "no row pins the context or the slots" is exact, whose reference recipe
        keeps its own geometry: one session, and PP5's at the official reference's context limit (512). A base that
        would pin either on every other row is refused at composition."""
        self.assertEqual(GENERATOR.SESSION_PINS_EXCEPTION, 'exact')
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        pp5 = rows['pp5', ('code', 'exact', 'single', 'decode')]
        self.assertTrue({'--context_tokens=512'} <= set(pp5['settings'].splitlines()))
        self.assertIn('--session_slots=1', pp5['serve_settings'].splitlines())
        self.assertIn('--session_slots=1', rows['tp4_exl3', ('code', 'exact', 'single', 'decode')]
                      ['serve_settings'].splitlines())
        pinned = [key for key, row in rows.items() if not row['hold'] and any(
            GENERATOR.option_key(word) in GENERATOR.SESSION_KNOBS
            for field in ('settings', 'serve_settings', 'generate_settings') for word in row[field].splitlines())]
        self.assertTrue(pinned and all(values[1] == 'exact' for _, values in pinned))
        data = registry()
        base = data['splits']['tp4_pro']['base']  # original_tp_serial2: no set's recipe
        data['strategies'][base['route']]['settings'].append('--context_tokens=65536')
        data['strategies'][base['route']]['generate_settings'].append('--context_tokens=65536')
        base['raw_defaults']['--context_tokens'] = '1048576'
        with self.assertRaisesRegex(ValueError, 'a row pins no context or session count'):
            GENERATOR.make_choices(GENERATOR.validate_registry(data))

    def test_batch_applies_only_at_its_eight_slots(self):
        """9 Nov: TP4_EXL3 concurrency=multiple's BATCH set needs exactly 8 sessions, so on serve it applies only
        where the final --session_slots is exactly 8; elsewhere startup runs the row's concurrency=balanced row (thin
        steps for as many as fit). The table marks those rows with the count and keeps their balanced rows unheld;
        only a concurrency set may carry the rule."""
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        for (split, values), row in rows.items():
            batch = split == 'tp4_exl3' and values[2] == 'multiple' and not row['hold']
            self.assertEqual(row['serve_slots_exactly'], 8 if batch else 0, (split, values))
            if batch:
                self.assertIn('--batch_rows=8,8,12,12,16,24,32,48', row['settings'].splitlines())
                fallback = rows[split, (values[0], values[1], 'balanced', values[3])]
                self.assertFalse(fallback['hold'] or fallback['serve_slots_exactly'])
                self.assertFalse(any(word.startswith('--batch_rows=') for word in fallback['settings'].splitlines()))
        for axis, value, slots in (('focus', 'decode', 8), ('concurrency', 'multiple', 0),
                                   ('concurrency', 'multiple', True)):
            data = registry()
            data['splits']['tp4_exl3']['sets']['concurrency']['multiple'].pop('applies_with_session_slots')
            data['splits']['tp4_exl3']['sets'][axis][value]['applies_with_session_slots'] = slots
            with self.assertRaisesRegex(ValueError, 'only a concurrency set applies at a serving slot count'):
                GENERATOR.validate_registry(data)

    def test_route_guard_refuses_unmeasured_family_and_hold_changes(self):
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        changed = copy.deepcopy(rows)
        single = next(row for row in changed if row['split'] == 'tp4_pro' and
                      row['values'] == ('code', 'balanced', 'single', 'decode'))
        single['settings'] = single['settings'].replace('--pipeline_steps=2', '--pipeline_steps=5')
        with self.assertRaisesRegex(AssertionError, 'unmeasured route'):
            self.assert_route_envelope(changed)
        changed = copy.deepcopy(rows)
        held = next(row for row in changed if row['split'] == 'tp4_pro' and
                    row['values'] == ('code', 'balanced', 'multiple', 'decode'))
        held['hold'] = ''
        with self.assertRaisesRegex(AssertionError, 'unmeasured route or hold'):
            self.assert_route_envelope(changed)
        changed = copy.deepcopy(rows)
        multiple = next(row for row in changed if row['split'] == 'pp5' and
                        row['values'] == ('prose', 'balanced', 'multiple', 'decode'))
        multiple['settings'] = multiple['settings'].replace('--draft_session_acceptance=6',
                                                            '--draft_session_acceptance=10')
        with self.assertRaisesRegex(AssertionError, 'unmeasured session modifier'):
            self.assert_route_envelope(changed)

    def test_fast_reuses_only_selectable_nonlossy_balanced_plans(self):
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        indexed = {(row['split'], row['values']): row for row in rows}
        reused = 0
        for row in rows:
            category, tier, concurrency, focus = row['values']
            if tier != 'fast':
                continue
            source = indexed[row['split'], (category, 'balanced', concurrency, focus)]
            if source['hold']:
                self.assertTrue(row['hold'])
            else:
                reused += 1
                self.assertFalse(row['hold'])
                self.assertIn(source['numerical_class'], ('same_bits', 'draft_only'))
                for field in (*GENERATOR.STRATEGY_FIELDS, 'error_class', 'strategy', 'sources'):
                    self.assertEqual(row[field], source[field])
        self.assertEqual(reused, 128)

    def test_retained_records_cannot_be_admitted_or_omit_scope(self):
        for field, value in [('selectable', True), ('hold', ''), ('fallback', ''),
                             ('proposed_axes', {'concurrency': ['single']})]:
            data = registry()
            data['retained_routes'][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'retained route'):
                GENERATOR.validate_registry(data)

    def test_exactness_carries_no_set_beside_fast(self):
        data = registry()
        data['splits']['pp5']['sets']['exactness'] = {'balanced': {'settings': ['--rows=8'],
                                                                   'numerical_class': 'same_bits',
                                                                   'error_class': 'x', 'evidence': 'x'}}
        with self.assertRaisesRegex(ValueError, 'exactness carries only a fast set'):
            GENERATOR.validate_registry(data)

    def test_original_reference_scope_is_prepared_and_exl3_runs_exact_shaping(self):
        data = registry()
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(data))
        original = next(row for row in rows if row['split'] == 'pp5' and
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
            if row['family'] == 'exl3' and row['values'][1] == 'exact' and not row['hold']:
                self.assertEqual(row['strategy'], 'exl3_exact_shaping')
            if row['values'][1] == 'exact':
                self.assertEqual(not row['hold'], row['split'] in ('pp5', 'tp4_exl3') and
                                 row['values'][2] == 'single' and row['values'][0] != 'vision')

    def test_a_split_without_a_reference_must_hold_exact(self):
        data = registry()
        data['splits']['tp4_exl3']['reference'] = None
        with self.assertRaisesRegex(ValueError, 'must hold exact'):
            GENERATOR.validate_registry(data)

    def test_reference_rule_cannot_select_current_arithmetic(self):
        data = registry()
        data['splits']['pp5']['reference'] = 'original_pp_depth8'
        with self.assertRaisesRegex(ValueError, 'original reference arithmetic'):
            GENERATOR.validate_registry(data)

    def test_held_value_carries_no_set(self):
        data = registry()
        data['splits']['tp4_pro']['sets'].setdefault('concurrency', {})['multiple'] = {
            'strategy': 'original_pp_depth8', 'drop': [], 'evidence': 'x'}
        with self.assertRaisesRegex(ValueError, 'held axis value cannot carry a knob set'):
            GENERATOR.validate_registry(data)

    def test_rejected_thin_policy_cannot_reenter_an_axis(self):
        data = registry()
        rejected = data["rejected_strategies"]["thin_cost_policy1"]
        self.assertEqual(rejected["numerical_class"], "sum_order")
        self.assertTrue(rejected["raw_only"])
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(data))
        for row in rows:
            if row["family"] == "exl3" and row["values"][2] == "single" and not row["hold"]:
                self.assertNotIn("--draft_policy=1", row["settings"].split("\n"))
        data["splits"]["tp4_exl3"]["sets"]["concurrency"]["single"] = {"strategy": "thin_cost_policy1", "drop": [],
                                                                        "evidence": "x"}
        with self.assertRaisesRegex(ValueError, "a recipe set names a strategy"):
            GENERATOR.validate_registry(data)
        data["splits"]["tp4_exl3"]["sets"]["concurrency"].pop("single")
        data["splits"]["tp4_exl3"]["base"]["route"] = "thin_cost_policy1"
        with self.assertRaises((ValueError, KeyError)):
            GENERATOR.validate_registry(data)

    def test_layout_is_one_runtime_argument(self):
        arguments = ["--placement=pp5", "--layout=layers 0..7: card 0; head: card 4"]
        self.assertEqual(GENERATOR.argument_tokens(arguments), arguments)
        rows = GENERATOR.make_choices(GENERATOR.validate_registry(registry()))
        original = next(row for row in rows if row["split"] == "pp5" and
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
        for axis, value in (("category", "code"), ("focus", "decode"), ("concurrency", "single")):
            with self.subTest(axis=axis, value=value):
                data = registry()
                data["quality_budget"], data["strategies"]["synthetic_lossy"] = synthetic_fixture()
                data["splits"]["pp5"]["sets"][axis] = {value: {"strategy": "synthetic_lossy", "drop": [],
                                                               "evidence": "Synthetic test only."}}
                keep_owned_yields(data, "pp5")
                with self.assertRaisesRegex(ValueError, f"lossy strategy in {axis}={value}"):
                    GENERATOR.validate_registry(data, allow_synthetic=True)

    def test_selected_lossy_tuple_cannot_omit_its_error_figures(self):
        data = registry()
        data["quality_budget"], data["strategies"]["synthetic_lossy"] = synthetic_fixture()
        next(tier for tier in data["tiers"] if tier["name"] == "fast")["hold"] = ""
        data["splits"]["tp4_exl3"]["sets"]["exactness"] = {"fast": {"strategy": "synthetic_lossy", "drop": [],
                                                                    "evidence": "Synthetic test only."}}
        GENERATOR.validate_registry(data, allow_synthetic=True)
        choices = [row for row in GENERATOR.make_choices(data)
                   if row["values"][1] == "fast" and not row["hold"] and row["split"] == "tp4_exl3"]
        self.assertEqual(len(choices), 48)
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


ROOT = HERE.parents[1]
def keep_owned_yields(data, split):
    """A synthetic registry that replaces an axis's sets keeps only the yields another axis still
    sets."""
    sets = data["splits"][split]["sets"]
    owned = {GENERATOR.option_key(word) for axis, values in sets.items() for entry in values.values()
             if "strategy" not in entry for field in ("settings", "serve_settings", "generate_settings")
             for word in entry.get(field, [])}
    for values in sets.values():
        for entry in values.values():
            if "yields" in entry:
                entry["yields"] = [key for key in entry["yields"] if key in owned]


FOLDED_LABEL = re.compile(r"(?<![A-Za-z0-9_])(PP8|serial2)(?![A-Za-z0-9_])")
ROUTE_TABLES = ("tools/cli/AXES.md", "tools/cli/KNOBS.md", "docs/USAGE.md", "docs/SERVER_PROFILES.md")


def last_value(arguments, key):
    values = [argument.split("=", 1)[1] for argument in arguments if argument.startswith(key + "=")]
    return values[-1] if values else None


class RouteParameterTest(unittest.TestCase):
    def test_columns_are_each_routes_own_settings(self):
        for name, strategy in registry()["strategies"].items():
            if name == "empty":
                continue
            serve = (GENERATOR.argument_tokens(strategy["settings"]) +
                     GENERATOR.argument_tokens(strategy["serve_settings"]))
            generate = GENERATOR.argument_tokens(strategy["generate_settings"] or strategy["settings"])
            split, depth, slots = GENERATOR.route_parameters(strategy)
            # a strategy with no --placement (the raw-only Engram placement records knobs, not a route) shows default
            placement = last_value(serve, "--placement")
            self.assertEqual(split.split(" (")[0], placement.upper() if placement is not None else "default", name)
            self.assertEqual(depth.split(" (")[0], last_value(serve, "--pipeline_steps") or "default", name)
            self.assertEqual(slots.split(" (")[0], last_value(serve, "--session_slots") or "default", name)
            generated = last_value(generate, "--pipeline_steps")
            self.assertEqual(depth.endswith(f"(generate {generated})"),
                             generated is not None and generated != last_value(serve, "--pipeline_steps"), name)

    def test_generate_value_is_shown_beside_a_different_serve_value(self):
        strategy = {"settings": ["--placement=pp5", "--pipeline_steps=8"], "serve_settings": ["--session_slots=8"],
                    "generate_settings": ["--placement=pp5", "--pipeline_steps=2"]}
        self.assertEqual(GENERATOR.route_parameters(strategy), ["PP5", "8 (generate 2)", "8"])

    def test_every_route_table_row_carries_its_derived_columns(self):
        strategies = registry()["strategies"]
        for document in ROUTE_TABLES:
            rows = 0
            for line in (ROOT / document).read_text().splitlines():
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                routes = {name for name in re.findall(r"`([a-z0-9_]+)`", line) if name in strategies}
                if line.startswith("|") and len(routes) == 1 and len(cells) >= 4 and cells[-3] in (
                        "TP4_PRO", "PP5", "TP4_EXL3"):
                    rows += 1
                    # a set's columns are its recipe's without the knobs it names after "without" (AXES.md)
                    dropped = {key for clause in re.findall(r"` without ([^;|]*)", line)
                               for key in re.findall(r"`(--[a-z0-9_]+)`", clause)}
                    strategy = {field: [word for word in GENERATOR.argument_tokens(value)
                                        if GENERATOR.option_key(word) not in dropped]
                                for field, value in strategies[routes.pop()].items() if field in GENERATOR.STRATEGY_FIELDS}
                    expected = GENERATOR.route_parameters(strategy)
                    # a serve's slots left at the raw default are max since U5a; a guide may say so
                    slots = {expected[2], 'max'} if expected[2] == 'default' else {expected[2]}
                    self.assertEqual(cells[-3:-1], expected[:2], document + ": " + line)
                    self.assertIn(cells[-1], slots, document + ": " + line)
            self.assertGreater(rows, 0, document)

    def test_recorded_serve_recipes_give_every_knob_of_their_no_axis_row(self):
        """A recorded launch recipe without axis flags states what it runs: it gives every knob its split's
        all-balanced row would insert (9 Nov: pinned, not --axes=off)."""
        rows = {(row['split'], row['values']): row for row in
                GENERATOR.make_choices(GENERATOR.validate_registry(registry()))}
        text = (ROOT / 'docs/SERVER_PROFILES.md').read_text()
        recipes = [block for block in re.findall(r'```sh\n(.*?)```', text, re.S) if 'anemone serve' in block]
        checked = 0
        for block in recipes:
            words = re.findall(r"'?(--[a-z0-9_-]+(?:=[^' \\\n]+(?: [^' \\\n-][^'\\\n]*)?)?)'?", block)
            if '--axes=off' in words or any(GENERATOR.option_key(word) in
                                            ('--category', '--exactness', '--concurrency', '--focus') for word in words):
                continue
            given = {GENERATOR.option_key(word) for word in words}
            placement = next((word.split('=', 1)[1] for word in words if word.startswith('--placement=')), 'pp5')
            row = rows[placement, ('balanced',) * 4]
            missing = [word for word in row['settings'].splitlines() + row['serve_settings'].splitlines()
                       if GENERATOR.option_key(word) not in given]
            self.assertEqual(missing, [], placement)
            checked += 1
        self.assertGreaterEqual(checked, 2)

    def test_no_route_is_described_by_a_folded_depth_or_slot_label(self):
        texts = {"tools/cli/axes.json": json.dumps(registry())}
        texts.update({document: (ROOT / document).read_text() for document in ROUTE_TABLES})
        for document, text in texts.items():
            self.assertEqual(FOLDED_LABEL.findall(text), [], document)


if __name__ == "__main__":
    unittest.main()
