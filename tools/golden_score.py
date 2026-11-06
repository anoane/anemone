#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score aligned T=0 captures against a retained golden generation.

Each directory contains metadata.json, prompt_ids.json, token_ids.json and logits.f32.
The logits are little-endian F32 [generated_positions, vocabulary_size]; row i predicts
(token_ids[i]) after the prompt and token_ids[:i]. Metadata schema 1 binds canonical
compact-JSON SHA256 hashes of the prompt and continuation in alignment. A candidate
must declare teacher_forced, with the golden IDs as its continuation. Golden self
comparison is supported directly. conditioning_ids.json is required for both captures
and must equal the prompt followed by all but the last generated token. The alignment
also binds its canonical SHA256. Producer provenance must establish that these are
actual observed inputs, rather than a reconstruction from requested IDs.

Generation is T=0; NLL/PPL/KL use the standard unit-temperature softmax distribution
over full logits, rather than the degenerate greedy point mass. Metrics use F64
max-shifted log probabilities and math.fsum. No logits are changed.
Top-1 ties choose the lowest token ID, matching the greedy sampler. Perplexity is exp
of mean token NLL (nats), not mean per-position perplexity. --certain-margin uses a
strict golden top1-minus-top2 > threshold test. No stochastic or task-quality claim.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
FLOAT_BYTES = 4
MINIMUM_VOCABULARY = 2
DEFAULT_CERTAIN_MARGIN = 1.0
PERCENT = 100.0
PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 100)
MAXIMUM_EXPONENT = math.log(sys.float_info.max)
KL_ROUNDING_TOLERANCE = 1e-12
MAXIMUM_REPORTED_DIVERGENCES = 16


class ScoreError(ValueError):
    """A capture is incomplete, malformed or cannot establish aligned positions."""


def require(condition, message):
    if not condition:
        raise ScoreError(message)


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_ids_digest(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def read_json(path):
    def unique_members(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, f"{path}: duplicate JSON member {name}")
            result[name] = value
        return result

    def bad_constant(value):
        raise ScoreError(f"{path}: nonfinite JSON constant {value}")

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_members, parse_constant=bad_constant)


def ids_file(path, vocabulary_size=None):
    values = read_json(path)
    require(isinstance(values, list), f"{path}: token IDs must be an array")
    require(all(type(value) is int and value >= 0 and
                (vocabulary_size is None or value < vocabulary_size) for value in values),
            f"{path}: every token ID must be a vocabulary integer")
    return values


def load_capture(directory):
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    require(isinstance(metadata, dict), f"{directory}: metadata must be an object")
    require(type(metadata.get("schema_version")) is int and metadata["schema_version"] == SCHEMA_VERSION, f"{directory}: unsupported schema_version")
    vocabulary_size = metadata.get("vocabulary_size")
    positions = metadata.get("generated_positions")
    require(type(vocabulary_size) is int and vocabulary_size >= MINIMUM_VOCABULARY,
            f"{directory}: invalid vocabulary_size")
    require(type(positions) is int and positions > 0, f"{directory}: generated_positions must be positive")
    require(type(metadata.get("temperature")) in (int, float) and metadata["temperature"] == 0,
            f"{directory}: scoring requires explicit temperature=0")
    prompt_ids = ids_file(directory / "prompt_ids.json", vocabulary_size)
    token_ids = ids_file(directory / "token_ids.json", vocabulary_size)
    require(prompt_ids and len(token_ids) == positions, f"{directory}: prompt/continuation lengths disagree")
    alignment = metadata.get("alignment")
    require(isinstance(alignment, dict), f"{directory}: missing alignment object")
    require(alignment.get("mode") in ("autoregressive_greedy", "teacher_forced"),
            f"{directory}: unknown alignment mode")
    require(alignment.get("prompt_ids_sha256") == canonical_ids_digest(prompt_ids),
            f"{directory}: prompt hash does not match actual prompt IDs")
    require(alignment.get("continuation_ids_sha256") == canonical_ids_digest(token_ids),
            f"{directory}: continuation hash does not match actual token IDs")
    witness_path = directory / "conditioning_ids.json"
    require(witness_path.is_file(), f"{directory}: actual-input witness is required")
    witnessed = witness_path.is_file()
    if witnessed:
        conditioning_ids = ids_file(witness_path, vocabulary_size)
        require(conditioning_ids == prompt_ids + token_ids[:-1],
                f"{directory}: actual conditioning IDs do not match the aligned sequence")
        witness_digest = alignment.get("conditioning_ids_sha256")
        require(witness_digest == canonical_ids_digest(conditioning_ids),
                f"{directory}: actual-input witness hash mismatch")
    logits_path = directory / "logits.f32"
    require(logits_path.stat().st_size == positions * vocabulary_size * FLOAT_BYTES,
            f"{directory}: logits.f32 byte count is not generated_positions*vocabulary_size*4")
    logits = np.memmap(logits_path, mode="r", dtype="<f4", shape=(positions, vocabulary_size))
    return dict(directory=str(directory.resolve()), metadata=metadata, prompt_ids=prompt_ids,
                token_ids=token_ids, positions=positions, vocabulary_size=vocabulary_size,
                logits=logits, actual_input_witness_present=witnessed)


def log_distribution(row):
    """Stable F64 log probabilities, retaining small normalizers under large common offsets."""
    shifted = row - float(np.max(row))
    normalizer = math.log(math.fsum(np.exp(shifted)))
    return shifted - normalizer


def perplexity(nll):
    return math.exp(nll) if nll <= MAXIMUM_EXPONENT else None


def distribution(values):
    ordered = sorted(float(value) for value in values)
    require(ordered and all(math.isfinite(value) for value in ordered), "metric distribution is empty/nonfinite")
    def quantile(percentile):
        location = (len(ordered) - 1) * percentile / PERCENT
        lower = int(location)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)
    return dict(count=len(ordered), minimum=ordered[0], maximum=ordered[-1],
                mean=math.fsum(ordered) / len(ordered),
                percentiles={str(percentile): quantile(percentile) for percentile in PERCENTILES})


def agreement_bucket(rows):
    count = len(rows)
    matched = sum(row["top1_equal"] for row in rows)
    return dict(positions=count, agreements=matched,
                agreement_percent=PERCENT * matched / count if count else None)


def free_divergence(golden_ids, free_ids):
    compared = min(len(golden_ids), len(free_ids))
    mismatches = [index for index in range(compared) if golden_ids[index] != free_ids[index]]
    first = mismatches[0] if mismatches else compared if len(golden_ids) != len(free_ids) else None
    return dict(first_divergence_index=first, indexing="zero-based generated token; common length if one is a prefix",
                first_divergence_kind="token" if mismatches else "length" if first is not None else None,
                compared_positions=compared, golden_positions=len(golden_ids), free_positions=len(free_ids),
                equal_tokens_in_shared_length=sum(golden_ids[index] == free_ids[index] for index in range(compared)),
                first_mismatching_positions=mismatches[:MAXIMUM_REPORTED_DIVERGENCES],
                complete_sequence_equal=first is None)


def score(golden_directory, candidate_directory, certain_margin=DEFAULT_CERTAIN_MARGIN,
          free_token_ids_path=None):
    require(type(certain_margin) in (int, float) and math.isfinite(certain_margin) and certain_margin >= 0,
            "certain margin must be finite and nonnegative")
    golden = load_capture(golden_directory)
    candidate = load_capture(candidate_directory)
    require(golden["metadata"]["alignment"]["mode"] == "autoregressive_greedy",
            "golden capture must be an autoregressive_greedy generation")
    same_capture = Path(golden_directory).resolve() == Path(candidate_directory).resolve()
    require(same_capture or candidate["metadata"]["alignment"]["mode"] == "teacher_forced",
            "candidate must be teacher-forced on golden IDs; score free-running IDs separately")
    require(golden["vocabulary_size"] == candidate["vocabulary_size"] and
            golden["positions"] == candidate["positions"], "capture row/vocabulary shapes differ")
    require(golden["prompt_ids"] == candidate["prompt_ids"], "candidate prompt IDs differ from golden")
    require(golden["token_ids"] == candidate["token_ids"], "candidate is not conditioned on the golden continuation")
    native_drawn_path = Path(candidate_directory) / "native_drawn_ids.json"
    native_drawn = ids_file(native_drawn_path, golden["vocabulary_size"]) if native_drawn_path.is_file() else None
    require(native_drawn is None or len(native_drawn) == golden["positions"], "native drawn-ID count differs")
    rows = []
    squared_delta_sums = []
    vocabulary_size = golden["vocabulary_size"]
    for position, token_id in enumerate(golden["token_ids"]):
        original = np.asarray(golden["logits"][position], dtype=np.float64)
        comparison = np.asarray(candidate["logits"][position], dtype=np.float64)
        require(np.isfinite(original).all() and np.isfinite(comparison).all(),
                f"position {position}: nonfinite logit")
        original_top = int(np.argmax(original))
        candidate_top = int(np.argmax(comparison))
        require(native_drawn is None or native_drawn[position] == candidate_top,
                f"position {position}: native T=0 sampler ID is not the candidate logit argmax")
        require(token_id == original_top, f"position {position}: golden T=0 token is not its lowest-ID argmax")
        top_two = np.partition(original, vocabulary_size - MINIMUM_VOCABULARY)[-MINIMUM_VOCABULARY:]
        margin = float(np.max(top_two) - np.min(top_two))
        original_log = log_distribution(original)
        candidate_log = log_distribution(comparison)
        original_nll, candidate_nll = -float(original_log[token_id]), -float(candidate_log[token_id])
        difference = comparison - original
        squared_sum = math.fsum(difference * difference)
        squared_delta_sums.append(squared_sum)
        divergence = math.fsum(np.exp(original_log) * (original_log - candidate_log))
        require(divergence >= -KL_ROUNDING_TOLERANCE,
                f"position {position}: materially negative KL ({divergence})")
        rows.append(dict(position=position, absolute_position=len(golden["prompt_ids"]) - 1 + position,
                         golden_token_id=token_id, golden_top1=original_top, candidate_top1=candidate_top,
                         top1_equal=original_top == candidate_top, golden_top1_margin=margin,
                         certain=margin > certain_margin, golden_nll=original_nll, candidate_nll=candidate_nll,
                         candidate_perplexity=perplexity(candidate_nll),
                         candidate_perplexity_overflow=candidate_nll > MAXIMUM_EXPONENT,
                         max_absolute_logit_delta=float(np.max(np.abs(difference))),
                         rms_logit_delta=math.sqrt(squared_sum / vocabulary_size),
                         kl_golden_candidate=max(0.0, divergence)))
    golden_nll = math.fsum(row["golden_nll"] for row in rows) / len(rows)
    candidate_nll = math.fsum(row["candidate_nll"] for row in rows) / len(rows)
    certain = [row for row in rows if row["certain"]]
    uncertain = [row for row in rows if not row["certain"]]
    result = dict(schema_version=SCHEMA_VERSION, golden_directory=golden["directory"],
                  candidate_directory=candidate["directory"], positions=len(rows), vocabulary_size=vocabulary_size,
                  temperature=0, probability_temperature=1.0, metrics_precision="F64 max-shifted log probabilities; math.fsum; full vocabulary",
                  alignment=dict(prompt_ids_equal=True, continuation_ids_equal=True,
                                 candidate_mode=candidate["metadata"]["alignment"]["mode"],
                                 witness_required=True,
                                 golden_actual_input_witness_present=golden["actual_input_witness_present"],
                                 candidate_actual_input_witness_present=candidate["actual_input_witness_present"],
                                 limit="capture-producer provenance must establish observed inputs; hashes alone do not"),
                  golden_nll=golden_nll, golden_perplexity=perplexity(golden_nll),
                  candidate_nll=candidate_nll, candidate_perplexity=perplexity(candidate_nll),
                  candidate_perplexity_overflow=candidate_nll > MAXIMUM_EXPONENT,
                  top1=agreement_bucket(rows), max_absolute_logit_delta=max(row["max_absolute_logit_delta"] for row in rows),
                  rms_logit_delta=math.sqrt(math.fsum(squared_delta_sums) / (len(rows) * vocabulary_size)),
                  kl_golden_candidate=distribution([row["kl_golden_candidate"] for row in rows]),
                  metric_distributions={name: distribution([row[name] for row in rows]) for name in
                                        ("candidate_nll", "golden_top1_margin", "max_absolute_logit_delta",
                                         "rms_logit_delta")},
                  certain=dict(threshold_logit_margin=certain_margin, comparison="strictly greater than threshold",
                               share=PERCENT * len(certain) / len(rows), share_unit="percent",
                               **agreement_bucket(certain)),
                  uncertain=agreement_bucket(uncertain), per_position=rows)
    free_path = Path(free_token_ids_path) if free_token_ids_path else Path(candidate_directory) / "free_token_ids.json"
    result["free_running"] = free_divergence(golden["token_ids"], ids_file(free_path, vocabulary_size)) \
        if free_path.is_file() else None
    names = ("metadata.json", "prompt_ids.json", "token_ids.json", "conditioning_ids.json", "logits.f32")
    result["input_sha256"] = {label: {name: digest_file(Path(folder) / name) for name in names}
                              for label, folder in (("golden", golden_directory), ("candidate", candidate_directory))}
    if native_drawn is not None:
        result["input_sha256"]["candidate"]["native_drawn_ids.json"] = digest_file(native_drawn_path)
    if free_path.is_file():
        result["input_sha256"]["free_running_ids"] = digest_file(free_path)
    return result


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("golden", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--certain-margin", type=float, default=DEFAULT_CERTAIN_MARGIN)
    parser.add_argument("--free-token-ids", type=Path, help="separate T=0 free-run IDs for divergence reporting")
    parser.add_argument("--require-witness", action="store_true", help="compatibility spelling; input witnesses are always required")
    parser.add_argument("--output", type=Path, help="write JSON once; an existing output is refused")
    options = parser.parse_args(arguments)
    if options.output is not None:
        require(not options.output.exists(), f"output exists already: {options.output}")
    if options.free_token_ids is not None:
        require(options.free_token_ids.is_file(), f"free-running IDs file is missing: {options.free_token_ids}")
    result = score(options.golden, options.candidate, options.certain_margin,
                   options.free_token_ids)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if options.output is None:
        sys.stdout.write(text)
    else:
        with options.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
        print(json.dumps({name: result[name] for name in
                          ("positions", "candidate_nll", "candidate_perplexity", "top1",
                           "max_absolute_logit_delta", "rms_logit_delta", "kl_golden_candidate",
                           "certain", "uncertain", "free_running")}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, OverflowError, KeyError, TypeError) as error:
        print(f"golden_score refused: {error}", file=sys.stderr)
        raise SystemExit(1)
