#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-only controls of golden_capture_score.py and golden_cases.py. Run on Linux:
python3 tools/golden/golden_capture_score_test.py. Synthetic captures in the layout `anemone capture` writes; no GPU.
"""
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


def loaded(name):
    loader = importlib.machinery.SourceFileLoader(name, str(Path(__file__).with_name(name + ".py")))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


RECORD = loaded("golden_capture_score")
CASES = loaded("golden_cases")
VOCABULARY = 6


def canonical(ids):
    return json.dumps(ids, separators=(",", ":"))


def digest(ids):
    return hashlib.sha256(canonical(ids).encode()).hexdigest()


class CaptureScoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="golden_capture_score_")
        self.root = Path(self.temporary.name)
        self.golden, self.capture = self.root / "golden", self.root / "capture"
        self.rng = np.random.default_rng(7)

    def tearDown(self):
        self.temporary.cleanup()

    def case(self, key, positions, shift=0.0, truncated=False, rows=6):
        """A golden case (greedy) and its capture (teacher-forced, logits shifted on token 0 by shift)."""
        logits = self.rng.normal(size=(positions, VOCABULARY)).astype("<f4")
        tokens, prompt = logits.argmax(axis=1).tolist(), [1, 2]
        conditioning = prompt + tokens[:-1]
        for folder, mode, values in ((self.golden, "autoregressive_greedy", logits),
                                     (self.capture, "teacher_forced", logits + np.float32(shift) * np.eye(
                                         1, VOCABULARY, 0, dtype="<f4"))):
            directory = folder / "cases" / key
            directory.mkdir(parents=True)
            metadata = dict(schema_version=1, vocabulary_size=VOCABULARY, generated_positions=positions, temperature=0,
                            alignment=dict(mode=mode, prompt_ids_sha256=digest(prompt),
                                           continuation_ids_sha256=digest(tokens),
                                           conditioning_ids_sha256=digest(conditioning)))
            if folder == self.capture:
                metadata["capture"] = dict(executable_sha256="e" * 64, argv_sha256="a" * 64, rows=rows,
                                           placement="tp4_exl3", weights="exl3", map="model.exl3map",
                                           truncated=truncated, logits_sha256="f" * 64, context_tokens=512)
            (directory / "metadata.json").write_text(json.dumps(metadata))
            for name, ids in (("prompt_ids.json", prompt), ("token_ids.json", tokens),
                              ("conditioning_ids.json", conditioning)):
                (directory / name).write_text(canonical(ids))
            np.ascontiguousarray(values, dtype="<f4").tofile(directory / "logits.f32")
        with (self.capture / "cases.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(case=key, positions=positions)) + "\n")
        (self.golden / "MANIFEST.json").write_text("{}")
        self.planned = getattr(self, "planned", 0) + 1
        self.manifest(self.planned, hashlib.sha256(b"{}").hexdigest())
        return logits, tokens

    def manifest(self, cases, golden_manifest_sha256):
        (self.capture / "CAPTURE_MANIFEST.json").write_text(json.dumps(dict(
            plan_sha256="p" * 64, golden_manifest_sha256=golden_manifest_sha256, cases=cases)))

    def recorded(self, *options):
        output = self.root / ("record%d.json" % len(list(self.root.glob("record*.json"))))
        with contextlib.redirect_stdout(io.StringIO()):
            RECORD.main([str(self.golden), str(self.capture), "--output", str(output), *options])
        return json.loads(output.read_text())

    def test_identity_and_token_weight(self):
        self.case("set/a", 3)
        self.case("set/b", 5)
        record = self.recorded()
        self.assertEqual((record["cases"], record["positions"]), (2, 8))
        self.assertEqual(record["candidate_nll"], record["golden_nll"])
        self.assertEqual((record["delta_nll"], record["max_absolute_logit_delta"], record["rms_logit_delta"]), (0, 0, 0))
        self.assertEqual(record["top1"], dict(positions=8, agreements=8, agreement_percent=100.0))
        self.assertEqual(record["capture"]["rows"], 6)

    def test_weighted_mean(self):
        self.case("set/a", 2, shift=3.0)
        self.case("set/b", 6)
        scores = [RECORD.SCORE.score(self.golden / "cases" / key, self.capture / "cases" / key) for key in ("set/a", "set/b")]
        record = self.recorded()
        expected = math.fsum(score["candidate_nll"] * score["positions"] for score in scores) / 8
        self.assertAlmostEqual(record["candidate_nll"], expected, places=12)
        self.assertNotEqual(record["candidate_nll"], math.fsum(score["candidate_nll"] for score in scores) / 2)
        self.assertGreater(record["max_absolute_logit_delta"], 2.9)

    def test_compact_equals_full_likelihoods(self):
        self.case("set/a", 4, shift=0.5)
        with contextlib.redirect_stdout(io.StringIO()):
            RECORD.SCORE.main([str(self.golden / "cases/set/a"), "--write-compact",
                               str(self.golden / "cases/set/a/topk64.bin")])
        full, compact = self.recorded(), self.recorded("--compact")
        self.assertEqual(compact["reference"], "compact")
        self.assertEqual((compact["candidate_nll"], compact["top1"]), (full["candidate_nll"], full["top1"]))
        self.assertIn("kl_top_k_tail_mean", compact)

    def test_truncated_refused(self):
        self.case("set/a", 2, truncated=True)
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.recorded()

    def test_mixed_runs_and_output_refused(self):
        self.case("set/a", 2, rows=6)
        self.case("set/b", 2, rows=1)
        with self.assertRaisesRegex(ValueError, "different runs"):
            self.recorded()
        output = self.root / "exists.json"
        output.write_text("{}")
        with self.assertRaisesRegex(ValueError, "exists"):
            RECORD.main([str(self.golden), str(self.capture), "--output", str(output)])

    def test_repeated_unfinished_and_other_golden_refused(self):
        self.case("set/a", 2)
        self.case("set/b", 2)
        lines = (self.capture / "cases.jsonl").read_text()
        (self.capture / "cases.jsonl").write_text(lines + lines.splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "a case twice"):
            self.recorded()
        (self.capture / "cases.jsonl").write_text(lines)
        self.manifest(3, hashlib.sha256(b"{}").hexdigest())
        with self.assertRaisesRegex(ValueError, "unfinished"):
            self.recorded()
        self.manifest(2, hashlib.sha256(b"{}").hexdigest())
        (self.golden / "MANIFEST.json").write_text('{"other": 1}')
        with self.assertRaisesRegex(ValueError, "not the golden captured"):
            self.recorded()
        (self.golden / "MANIFEST.json").unlink()
        self.manifest(2, None)
        self.assertEqual(self.recorded()["cases"], 2)

    def test_case_keys(self):
        self.case("set/b", 1)
        self.case("other/a", 1)
        self.assertEqual(CASES.golden_keys(self.golden), (["other/a", "set/b"], "no index.json"))
        (self.golden / "index.json").write_text(json.dumps(dict(cases=[dict(set="set", id="b"), "other/a"])))
        self.assertEqual(CASES.golden_keys(self.golden)[1], "index.json lists the same 2 cases")
        (self.golden / "index.json").write_text(json.dumps(["set/b"]))
        with self.assertRaisesRegex(ValueError, "index.json lists 1 cases"):
            CASES.golden_keys(self.golden)
        (self.golden / "index.json").unlink()
        self.case("set-b/c", 1)  # '-' sorts before '/': the index and the folders compared as strings
        (self.golden / "index.json").write_text(json.dumps(dict(cases=["other/a", "set-b/c", "set/b"])))
        self.assertEqual(CASES.golden_keys(self.golden), (["other/a", "set-b/c", "set/b"],
                                                          "index.json lists the same 3 cases"))
        (self.golden / "index.json").unlink()
        (self.golden / "cases/bad set/x").mkdir(parents=True)
        (self.golden / "cases/bad set/x/metadata.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "outside"):
            CASES.golden_keys(self.golden)


if __name__ == "__main__":
    unittest.main()
