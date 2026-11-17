#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score every case of an `anemone capture` root against its golden dataset and aggregate the scores into a quality
record (B3: the quality record B1's EXL3 map descriptors cite).

    golden_capture_score.py GOLDEN_ROOT CAPTURE_ROOT --output RECORD_JSON [--compact] [--scores FOLDER]
                            [--certain-margin MARGIN]

Each line of CAPTURE_ROOT/cases.jsonl (the capture's finished cases, in run order) is scored with
tools/golden_score.py: GOLDEN_ROOT/cases/<key> against CAPTURE_ROOT/cases/<key>, in full mode (the golden's logits.f32),
or with --compact against the golden's topk64.bin. A case the scorer refuses fails the whole record, as do a case
listed twice, a capture whose lines are not every case CAPTURE_MANIFEST.json plans (an unfinished capture is not
scored) and a GOLDEN_ROOT whose MANIFEST.json is not the one the capture read (its SHA-256, or both none). --scores keeps
each case's complete score JSON (FOLDER/<set>/<id>.json, create-only).

The record weights every position equally (token weight): NLL is the mean of the positions' NLL, perplexity its exp, and
top-1 agreement and the certain bucket are position counts; in full mode the KL, the RMS logit delta (pooled) and the
largest |logit delta| too. Beside them it names the capture: the golden MANIFEST.json's SHA-256 and the executable's,
the plan and argv digests, placement, weights, map, the EXL3 map's descriptor digest (B1), rows and context (CAPTURE_MANIFEST.json and the cases'
metadata.json), the cases and positions, and the scorer's own SHA-256. The output is create-only.

Measurement tool: not part of the engine. It needs NumPy, as golden_score.py does.
"""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import sys
from pathlib import Path

SCORER_PATH = Path(__file__).resolve().parents[1] / "golden_score.py"
LOADER = importlib.machinery.SourceFileLoader("golden_score", str(SCORER_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
SCORE = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(SCORE)
RECORD_SCHEMA = 1
PERCENT = 100.0
CAPTURE_FIELDS = ("executable_sha256", "argv_sha256", "pack_header_sha256", "descriptor_sha256", "rows",
                  "context_tokens", "version", "placement", "weights", "map")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted(scores, name):
    """The position-weighted mean of a per-case mean."""
    total = sum(score["positions"] for score in scores)
    return math.fsum(score[name] * score["positions"] for score in scores) / total


def bucket(scores, name):
    positions = sum(score[name]["positions"] for score in scores)
    agreements = sum(score[name]["agreements"] for score in scores)
    return dict(positions=positions, agreements=agreements,
                agreement_percent=PERCENT * agreements / positions if positions else None)


def record_of(golden_root, capture_root, compact, certain_margin, scores_folder):
    golden_root, capture_root = Path(golden_root), Path(capture_root)
    manifest = json.loads((capture_root / "CAPTURE_MANIFEST.json").read_text())
    lines = [json.loads(line) for line in (capture_root / "cases.jsonl").read_text().splitlines() if line]
    SCORE.require(lines, "%s: cases.jsonl lists no finished case" % capture_root)
    keys = [line["case"] for line in lines]
    SCORE.require(len(set(keys)) == len(keys), "cases.jsonl lists a case twice: %s" % sorted(
        key for key in set(keys) if keys.count(key) > 1)[:4])
    SCORE.require(len(keys) == manifest["cases"], "cases.jsonl holds %d cases, the capture plans %d: an unfinished "
                  "capture is not scored" % (len(keys), manifest["cases"]))
    golden_manifest = golden_root / "MANIFEST.json"
    scored = digest_file(golden_manifest) if golden_manifest.exists() else None
    SCORE.require(scored == manifest.get("golden_manifest_sha256"), "the golden scored is not the golden captured: "
                  "MANIFEST.json's SHA-256 %s, the capture's %s" % (scored, manifest.get("golden_manifest_sha256")))
    scores, cases, captures = [], [], set()
    for line in lines:
        key = line["case"]
        candidate = capture_root / "cases" / key
        metadata = json.loads((candidate / "metadata.json").read_text())
        SCORE.require(not metadata["capture"]["truncated"], "%s: a truncated capture (--capture_positions) is not "
                      "scored: its positions are not the golden's" % key)
        captures.add(json.dumps({name: metadata["capture"].get(name) for name in CAPTURE_FIELDS}, sort_keys=True))
        score = (SCORE.score_compact if compact else SCORE.score)(golden_root / "cases" / key, candidate, certain_margin)
        if scores_folder is not None:
            path = Path(scores_folder) / (key + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(score, indent=2, sort_keys=True, allow_nan=False) + "\n")
        scores.append(score)
        cases.append(dict(case=key, positions=score["positions"], candidate_nll=score["candidate_nll"],
                          golden_nll=score["golden_nll"], top1=score["top1"],
                          first_divergence=(score.get("free_running") or {}).get("first_divergence_index"),
                          logits_sha256=metadata["capture"]["logits_sha256"]))
    SCORE.require(len(captures) == 1, "the cases were captured by different runs: %d capture records" % len(captures))
    positions = sum(score["positions"] for score in scores)
    candidate_nll, golden_nll = weighted(scores, "candidate_nll"), weighted(scores, "golden_nll")
    certain = [score["certain"] for score in scores]
    record = dict(schema_version=RECORD_SCHEMA, reference="compact" if compact else "full",
                  scorer_sha256=digest_file(SCORER_PATH), golden_manifest_sha256=manifest.get("golden_manifest_sha256"),
                  plan_sha256=manifest.get("plan_sha256"), keys_sha256=manifest.get("keys_sha256"),
                  capture=json.loads(captures.pop()), cases=len(scores), positions=positions,
                  weighting="every position equally (token weight)", candidate_nll=candidate_nll,
                  candidate_perplexity=SCORE.perplexity(candidate_nll), golden_nll=golden_nll,
                  golden_perplexity=SCORE.perplexity(golden_nll), delta_nll=candidate_nll - golden_nll,
                  top1=bucket(scores, "top1"),
                  certain=dict(threshold_logit_margin=certain_margin,
                               positions=sum(entry["positions"] for entry in certain),
                               agreements=sum(entry["agreements"] for entry in certain)),
                  uncertain=bucket(scores, "uncertain"), per_case=cases)
    if not compact:
        squared = math.fsum(score["rms_logit_delta"] ** 2 * score["positions"] for score in scores)
        record.update(kl_golden_candidate_mean=math.fsum(score["kl_golden_candidate"]["mean"] * score["positions"]
                                                         for score in scores) / positions,
                      rms_logit_delta=math.sqrt(squared / positions),
                      max_absolute_logit_delta=max(score["max_absolute_logit_delta"] for score in scores))
    else:
        record.update(kl_top_k_tail_mean=math.fsum(score["kl_top_k_tail"]["mean"] * score["positions"]
                                                   for score in scores) / positions)
    return record


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("golden_root", type=Path)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="the record, written once")
    parser.add_argument("--compact", action="store_true", help="score against each golden case's topk64.bin")
    parser.add_argument("--scores", type=Path, help="keep each case's score JSON here (create-only)")
    parser.add_argument("--certain-margin", type=float, default=SCORE.DEFAULT_CERTAIN_MARGIN)
    options = parser.parse_args(arguments)
    SCORE.require(not options.output.exists(), "output exists already: %s" % options.output)
    record = record_of(options.golden_root, options.capture_root, options.compact, options.certain_margin,
                       options.scores)
    with options.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({name: record[name] for name in ("reference", "cases", "positions", "candidate_nll",
                                                       "golden_nll", "delta_nll", "top1")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, OverflowError, KeyError, TypeError) as error:
        print("golden_capture_score refused: %s" % error, file=sys.stderr)
        raise SystemExit(1)
