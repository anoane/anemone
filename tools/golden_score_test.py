#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-only scorer controls. Run on Linux: python3 tools/golden_score_test.py.

Synthetic fixtures contain no benchmark prompts or private data. No GPU is used.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

LOADER = importlib.machinery.SourceFileLoader("golden_score", str(Path(__file__).with_name("golden_score.py")))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
SCORE = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(SCORE)


class GoldenScoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="golden_score_synthetic_")
        self.root = Path(self.temporary.name)
        self.golden = self.root / "golden"
        self.candidate = self.root / "candidate"

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, directory, logits, tokens=None, prompt=None, mode="autoregressive_greedy"):
        directory.mkdir()
        values = np.asarray(logits, dtype="<f4")
        if tokens is None:
            tokens = values.argmax(axis=1).tolist()
        prompt = [0] if prompt is None else prompt
        conditioning = prompt + tokens[:-1]
        metadata = dict(schema_version=1, vocabulary_size=values.shape[1], generated_positions=values.shape[0],
                        temperature=0, alignment=dict(mode=mode,
                            prompt_ids_sha256=SCORE.canonical_ids_digest(prompt),
                            continuation_ids_sha256=SCORE.canonical_ids_digest(tokens),
                            conditioning_ids_sha256=SCORE.canonical_ids_digest(conditioning)))
        for name, document in (("metadata", metadata), ("prompt_ids", prompt), ("token_ids", tokens),
                               ("conditioning_ids", conditioning)):
            (directory / (name + ".json")).write_text(json.dumps(document) + "\n")
        values.tofile(directory / "logits.f32")
        return metadata

    def pair(self, golden=None, candidate=None):
        golden = [[math.log(3), 0], [math.log(3), 0]] if golden is None else golden
        candidate = golden if candidate is None else candidate
        tokens = np.asarray(golden, dtype=np.float32).argmax(axis=1).tolist()
        self.fixture(self.golden, golden)
        self.fixture(self.candidate, candidate, tokens=tokens, mode="teacher_forced")

    def update_metadata(self, directory, change):
        path = directory / "metadata.json"
        metadata = json.loads(path.read_text())
        change(metadata)
        path.write_text(json.dumps(metadata) + "\n")

    def test_self_exact_zero_and_agreement(self):
        self.fixture(self.golden, [[math.log(2), 0, -2, -3], [0, 0, -1, -2], [1, 3, 0, 0]], prompt=[0, 1])
        result = SCORE.score(self.golden, self.golden, certain_margin=0.5)
        self.assertEqual(result["max_absolute_logit_delta"], 0)
        self.assertEqual(result["rms_logit_delta"], 0)
        self.assertEqual(result["kl_golden_candidate"]["maximum"], 0)
        self.assertEqual(result["top1"]["agreement_percent"], 100)
        self.assertEqual(result["certain"]["positions"], 2)
        self.assertEqual(result["uncertain"]["positions"], 1)
        self.assertEqual(result["certain"]["agreement_percent"], 100)
        self.assertEqual(result["uncertain"]["agreement_percent"], 100)
        self.assertGreater(result["candidate_nll"], 0)
        self.assertGreater(result["candidate_perplexity"], 1)
        self.assertIsNone(result["free_running"])
        self.assertTrue(result["alignment"]["witness_required"])
        self.assertEqual(len(result["input_sha256"]["golden"]["logits.f32"]), 64)

    def test_analytic_nll_kl_and_strict_argmax(self):
        self.pair(candidate=[[0, math.log(3)], [math.log(3), 0]])
        result = SCORE.score(self.golden, self.candidate)
        log_ratio = float(np.float32(math.log(3)))
        golden_probability = 1 / (1 + math.exp(-log_ratio))
        expected_first_kl = (2 * golden_probability - 1) * log_ratio
        expected_nll = (math.log1p(math.exp(log_ratio)) + math.log1p(math.exp(-log_ratio))) / 2
        self.assertAlmostEqual(result["candidate_nll"], expected_nll, places=14)
        self.assertAlmostEqual(result["candidate_perplexity"], math.exp(expected_nll), places=14)
        self.assertAlmostEqual(result["per_position"][0]["kl_golden_candidate"], expected_first_kl, places=14)
        self.assertEqual(result["per_position"][1]["kl_golden_candidate"], 0)
        self.assertAlmostEqual(result["kl_golden_candidate"]["mean"], expected_first_kl / 2, places=14)
        self.assertAlmostEqual(result["rms_logit_delta"], log_ratio / math.sqrt(2), places=14)
        self.assertEqual(result["top1"]["agreement_percent"], 50)
        self.assertEqual(result["certain"]["agreement_percent"], 50)
        self.assertIsNone(result["uncertain"]["agreement_percent"])

    def test_common_logit_offset_distribution_invariance(self):
        # Integers are exactly representable in F32; both logits get the same offset.
        self.pair(golden=[[1, 0], [2, 0]], candidate=[[6, 5], [7, 5]])
        result = SCORE.score(self.golden, self.candidate)
        self.assertEqual(result["max_absolute_logit_delta"], 5)
        self.assertEqual(result["rms_logit_delta"], 5)
        self.assertEqual(result["kl_golden_candidate"]["maximum"], 0)
        self.assertEqual(result["golden_nll"], result["candidate_nll"])
        self.assertEqual(result["top1"]["agreement_percent"], 100)

    def test_tied_candidate_uses_lowest_id(self):
        self.pair(golden=[[0, math.log(2)]], candidate=[[0, 0]])
        result = SCORE.score(self.golden, self.candidate)
        self.assertEqual(result["per_position"][0]["golden_top1"], 1)
        self.assertEqual(result["per_position"][0]["candidate_top1"], 0)
        self.assertEqual(result["top1"]["agreement_percent"], 0)

    def test_threshold_is_strict_and_empty_bucket_is_null(self):
        self.pair(golden=[[1, 0]], candidate=[[1, 0]])
        result = SCORE.score(self.golden, self.candidate, certain_margin=1)
        self.assertEqual(result["certain"]["positions"], 0)
        self.assertEqual(result["certain"]["share"], 0)
        self.assertIsNone(result["certain"]["agreement_percent"])
        self.assertEqual(result["uncertain"]["agreement_percent"], 100)

    def test_large_nll_reports_perplexity_overflow_without_nonfinite_json(self):
        self.pair(golden=[[0, -2000]], candidate=[[-2000, 0]])
        result = SCORE.score(self.golden, self.candidate)
        self.assertEqual(result["candidate_nll"], 2000)
        self.assertIsNone(result["candidate_perplexity"])
        self.assertTrue(result["candidate_perplexity_overflow"])
        self.assertEqual(result["kl_golden_candidate"]["mean"], 2000)
        json.dumps(result, allow_nan=False)

    def test_extreme_f32_common_offsets_keep_small_normalizer(self):
        self.pair(golden=[[1e38, 1e38]], candidate=[[2e38, 2e38]])
        result = SCORE.score(self.golden, self.candidate)
        self.assertAlmostEqual(result["candidate_nll"], math.log(2), places=15)
        self.assertEqual(result["kl_golden_candidate"]["maximum"], 0)
        self.assertTrue(math.isfinite(result["rms_logit_delta"]))

    def test_free_running_first_token_and_length_divergences(self):
        self.assertEqual(SCORE.free_divergence([1, 2, 3], [1, 0, 3])["first_divergence_index"], 1)
        self.assertEqual(SCORE.free_divergence([1, 2], [1])["first_divergence_kind"], "length")
        self.assertEqual(SCORE.free_divergence([1, 2], [1])["first_divergence_index"], 1)
        self.assertEqual(SCORE.free_divergence([1], [1, 2])["first_divergence_index"], 1)
        self.assertEqual(SCORE.free_divergence([1], [])["first_divergence_index"], 0)
        self.assertIsNone(SCORE.free_divergence([1, 2], [1, 2])["first_divergence_index"])

    def test_optional_free_file_and_native_argmax_witness(self):
        self.pair(candidate=[[0, math.log(3)], [math.log(3), 0]])
        (self.candidate / "native_drawn_ids.json").write_text("[1,0]\n")
        (self.candidate / "free_token_ids.json").write_text("[1,1]\n")
        result = SCORE.score(self.golden, self.candidate)
        self.assertEqual(result["free_running"]["first_divergence_index"], 0)
        self.assertIn("native_drawn_ids.json", result["input_sha256"]["candidate"])
        (self.candidate / "native_drawn_ids.json").write_text("[0,0]\n")
        with self.assertRaisesRegex(SCORE.ScoreError, "native T=0"):
            SCORE.score(self.golden, self.candidate)

    def test_bad_thresholds_refuse(self):
        self.pair()
        for threshold in (-1, math.nan, math.inf, True):
            with self.subTest(threshold=threshold), self.assertRaises(SCORE.ScoreError):
                SCORE.score(self.golden, self.candidate, threshold)

    def test_invalid_capture_metadata_refuses(self):
        self.pair()
        path = self.candidate / "metadata.json"
        valid = path.read_text()
        mutations = [lambda meta: meta.update(schema_version=2),
                     lambda meta: meta.update(schema_version=True),
                     lambda meta: meta.update(vocabulary_size=True),
                     lambda meta: meta.update(vocabulary_size=1),
                     lambda meta: meta.update(generated_positions=0),
                     lambda meta: meta.update(temperature=1),
                     lambda meta: meta.update(temperature=False),
                     lambda meta: meta["alignment"].update(mode="autoregressive_greedy"),
                     lambda meta: meta["alignment"].update(prompt_ids_sha256="0" * 64),
                     lambda meta: meta["alignment"].update(continuation_ids_sha256="0" * 64),
                     lambda meta: meta["alignment"].update(conditioning_ids_sha256="0" * 64)]
        for index, mutation in enumerate(mutations):
            path.write_text(valid)
            self.update_metadata(self.candidate, mutation)
            with self.subTest(mutation=index), self.assertRaises(SCORE.ScoreError):
                SCORE.score(self.golden, self.candidate)
        path.write_text(valid)

    def test_wrong_prompt_or_continuation_with_recomputed_hash_refuses(self):
        self.pair()
        (self.candidate / "prompt_ids.json").write_text("[1]\n")
        (self.candidate / "conditioning_ids.json").write_text("[1,0]\n")
        self.update_metadata(self.candidate, lambda meta: meta["alignment"].update(
            prompt_ids_sha256=SCORE.canonical_ids_digest([1]),
            conditioning_ids_sha256=SCORE.canonical_ids_digest([1, 0])))
        with self.assertRaisesRegex(SCORE.ScoreError, "prompt IDs differ"):
            SCORE.score(self.golden, self.candidate)

    def test_wrong_continuation_with_recomputed_hash_refuses(self):
        self.pair()
        (self.candidate / "token_ids.json").write_text("[1,1]\n")
        (self.candidate / "conditioning_ids.json").write_text("[0,1]\n")
        self.update_metadata(self.candidate, lambda meta: meta["alignment"].update(
            continuation_ids_sha256=SCORE.canonical_ids_digest([1, 1]),
            conditioning_ids_sha256=SCORE.canonical_ids_digest([0, 1])))
        with self.assertRaisesRegex(SCORE.ScoreError, "golden continuation"):
            SCORE.score(self.golden, self.candidate)

    def test_mandatory_actual_input_witness_and_recurrence(self):
        self.pair()
        path = self.candidate / "conditioning_ids.json"
        valid = path.read_text()
        path.unlink()
        with self.assertRaisesRegex(SCORE.ScoreError, "actual-input witness is required"):
            SCORE.score(self.golden, self.candidate)
        path.write_text("[0,1]\n")
        with self.assertRaisesRegex(SCORE.ScoreError, "actual conditioning IDs"):
            SCORE.score(self.golden, self.candidate)
        path.write_text(valid)

    def test_missing_truncated_extra_or_nonfinite_logits_refuse(self):
        self.pair()
        path = self.candidate / "logits.f32"
        valid = path.read_bytes()
        path.unlink()
        with self.assertRaises(OSError):
            SCORE.score(self.golden, self.candidate)
        for size in (len(valid) - 1, len(valid) + 1):
            path.write_bytes((valid + b"x")[:size])
            with self.subTest(size=size), self.assertRaisesRegex(SCORE.ScoreError, "byte count"):
                SCORE.score(self.golden, self.candidate)
        for value in (math.nan, math.inf, -math.inf):
            np.asarray([[value, 0], [1, 0]], dtype="<f4").tofile(path)
            with self.subTest(value=value), self.assertRaisesRegex(SCORE.ScoreError, "nonfinite logit"):
                SCORE.score(self.golden, self.candidate)

    def test_invalid_ids_and_json_refuse(self):
        self.pair()
        path = self.candidate / "token_ids.json"
        for text in ("[true,0]", "[0.0,0]", "[-1,0]", "[2,0]", "{}", "[0]"):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(SCORE.ScoreError):
                SCORE.score(self.golden, self.candidate)
        path.write_text("[0,0]")
        metadata = self.candidate / "metadata.json"
        metadata.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(SCORE.ScoreError, "duplicate JSON"):
            SCORE.score(self.golden, self.candidate)
        metadata.write_text('{"schema_version":NaN}')
        with self.assertRaisesRegex(SCORE.ScoreError, "nonfinite JSON"):
            SCORE.score(self.golden, self.candidate)

    def test_golden_output_must_be_its_argmax(self):
        self.fixture(self.golden, [[0, 1]], tokens=[0])
        with self.assertRaisesRegex(SCORE.ScoreError, "golden T=0 token"):
            SCORE.score(self.golden, self.golden)

    def test_output_is_write_once_and_missing_free_file_refuses(self):
        self.pair()
        output = self.root / "score.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(SCORE.main([str(self.golden), str(self.candidate), "--output", str(output)]), 0)
        before = output.read_bytes()
        with self.assertRaisesRegex(SCORE.ScoreError, "output exists"):
            SCORE.main([str(self.golden), str(self.candidate), "--output", str(output)])
        self.assertEqual(output.read_bytes(), before)
        with self.assertRaisesRegex(SCORE.ScoreError, "free-running IDs file is missing"):
            SCORE.main([str(self.golden), str(self.candidate), "--free-token-ids", str(self.root / "absent.json")])


if __name__ == "__main__":
    unittest.main()
