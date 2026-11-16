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

Compact reference: --write-compact stores a golden's topk64.bin, and --compact scores
a candidate against it, so a golden directory needs no logits.f32. Little-endian, a
64-byte header (magic ANEMTOPK, version 1, top_k, positions, vocabulary size, header
bytes, then zeros), then per position: top_k IDs (int32, highest logit first, ties by
lowest ID), their F64 log probabilities, and F64 logsumexp, golden-token log
probability, top1-minus-top2 logit margin and the log of the probability mass outside
the top_k (-inf when there is none). The candidate's NLL/PPL, top-1 agreement and the
certain share/agreement equal full mode exactly. Top-k overlap, golden-token-in-
candidate-top-k and a top-k KL whose (top_k + 1)-th bucket holds the tail are compact-
only; that bucketed KL is a lower bound of the full-vocabulary KL. Max/RMS logit
deltas and the full KL need the full golden logits.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
FLOAT_BYTES = 4
MINIMUM_VOCABULARY = 2
COMPACT_FILE = "topk64.bin"
COMPACT_MAGIC = b"ANEMTOPK"
COMPACT_VERSION = 1
COMPACT_TOP_K = 64
COMPACT_HEADER = struct.Struct("<8s6I32x")
COMPACT_FIELDS = (("top_ids", "<i4", True), ("top_logprobs", "<f8", True), ("logsumexp", "<f8", False),
                  ("golden_logprob", "<f8", False), ("top1_margin", "<f8", False), ("tail_logprob", "<f8", False))
COMPACT_MASS_TOLERANCE = 1e-9
CAPTURE_FILES = ("metadata.json", "prompt_ids.json", "token_ids.json", "conditioning_ids.json")
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


def load_capture(directory, with_logits=True):
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
    logits = None
    if with_logits:
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


def top_ids(row, count):
    """The count highest IDs of row, highest first, ties by lowest ID (the greedy sampler's order)."""
    threshold = np.partition(row, row.size - count)[row.size - count]
    above = np.flatnonzero(row > threshold)
    chosen = np.concatenate((above, np.flatnonzero(row == threshold)[:count - above.size]))
    return chosen[np.lexsort((chosen, -row[chosen]))]


def outside_log_mass(log_row, ids):
    """Log of the probability mass outside ids, from F64 log probabilities; -inf when it is zero."""
    outside = np.ones(log_row.size, dtype=bool)
    outside[ids] = False
    mass = math.fsum(np.exp(log_row[outside]).tolist())
    return math.log(mass) if mass > 0 else -math.inf


def aligned_captures(golden_directory, candidate_directory, certain_margin, golden_logits=True):
    require(type(certain_margin) in (int, float) and math.isfinite(certain_margin) and certain_margin >= 0,
            "certain margin must be finite and nonnegative")
    golden = load_capture(golden_directory, golden_logits)
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
    return golden, candidate, native_drawn


def candidate_row(candidate, position, native_drawn):
    comparison = np.asarray(candidate["logits"][position], dtype=np.float64)
    require(np.isfinite(comparison).all(), f"position {position}: nonfinite logit")
    candidate_top = int(np.argmax(comparison))
    require(native_drawn is None or native_drawn[position] == candidate_top,
            f"position {position}: native T=0 sampler ID is not the candidate logit argmax")
    return comparison, candidate_top


def likelihood_summary(rows):
    golden_nll = math.fsum(row["golden_nll"] for row in rows) / len(rows)
    candidate_nll = math.fsum(row["candidate_nll"] for row in rows) / len(rows)
    return dict(golden_nll=golden_nll, golden_perplexity=perplexity(golden_nll),
                candidate_nll=candidate_nll, candidate_perplexity=perplexity(candidate_nll),
                candidate_perplexity_overflow=candidate_nll > MAXIMUM_EXPONENT, top1=agreement_bucket(rows))


def certainty_summary(rows, certain_margin):
    certain = [row for row in rows if row["certain"]]
    return dict(certain=dict(threshold_logit_margin=certain_margin, comparison="strictly greater than threshold",
                             share=PERCENT * len(certain) / len(rows), share_unit="percent",
                             **agreement_bucket(certain)),
                uncertain=agreement_bucket([row for row in rows if not row["certain"]]))


def alignment_summary(golden, candidate):
    return dict(prompt_ids_equal=True, continuation_ids_equal=True,
                candidate_mode=candidate["metadata"]["alignment"]["mode"], witness_required=True,
                golden_actual_input_witness_present=golden["actual_input_witness_present"],
                candidate_actual_input_witness_present=candidate["actual_input_witness_present"],
                limit="capture-producer provenance must establish observed inputs; hashes alone do not")


def add_receipts(result, golden, golden_directory, candidate_directory, golden_names, native_drawn,
                 free_token_ids_path):
    vocabulary_size = golden["vocabulary_size"]
    free_path = Path(free_token_ids_path) if free_token_ids_path else Path(candidate_directory) / "free_token_ids.json"
    result["free_running"] = free_divergence(golden["token_ids"], ids_file(free_path, vocabulary_size)) \
        if free_path.is_file() else None
    result["input_sha256"] = {label: {name: digest_file(Path(folder) / name) for name in names}
                              for label, folder, names in (("golden", golden_directory, golden_names),
                                                           ("candidate", candidate_directory,
                                                            CAPTURE_FILES + ("logits.f32",)))}
    if native_drawn is not None:
        result["input_sha256"]["candidate"]["native_drawn_ids.json"] = \
            digest_file(Path(candidate_directory) / "native_drawn_ids.json")
    if free_path.is_file():
        result["input_sha256"]["free_running_ids"] = digest_file(free_path)
    return result


def score(golden_directory, candidate_directory, certain_margin=DEFAULT_CERTAIN_MARGIN,
          free_token_ids_path=None):
    golden, candidate, native_drawn = aligned_captures(golden_directory, candidate_directory, certain_margin)
    rows = []
    squared_delta_sums = []
    vocabulary_size = golden["vocabulary_size"]
    for position, token_id in enumerate(golden["token_ids"]):
        original = np.asarray(golden["logits"][position], dtype=np.float64)
        require(np.isfinite(original).all(), f"position {position}: nonfinite logit")
        comparison, candidate_top = candidate_row(candidate, position, native_drawn)
        original_top = int(np.argmax(original))
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
    result = dict(schema_version=SCHEMA_VERSION, golden_directory=golden["directory"],
                  candidate_directory=candidate["directory"], positions=len(rows), vocabulary_size=vocabulary_size,
                  temperature=0, probability_temperature=1.0, metrics_precision="F64 max-shifted log probabilities; math.fsum; full vocabulary",
                  alignment=alignment_summary(golden, candidate), **likelihood_summary(rows),
                  max_absolute_logit_delta=max(row["max_absolute_logit_delta"] for row in rows),
                  rms_logit_delta=math.sqrt(math.fsum(squared_delta_sums) / (len(rows) * vocabulary_size)),
                  kl_golden_candidate=distribution([row["kl_golden_candidate"] for row in rows]),
                  metric_distributions={name: distribution([row[name] for row in rows]) for name in
                                        ("candidate_nll", "golden_top1_margin", "max_absolute_logit_delta",
                                         "rms_logit_delta")},
                  **certainty_summary(rows, certain_margin), per_position=rows)
    return add_receipts(result, golden, golden_directory, candidate_directory,
                        CAPTURE_FILES + ("logits.f32",), native_drawn, free_token_ids_path)


def compact_reference(golden_directory):
    """The golden's compact arrays, from its full F32 logits; the log probabilities are full mode's F64 values."""
    golden = load_capture(golden_directory)
    require(golden["metadata"]["alignment"]["mode"] == "autoregressive_greedy",
            "golden capture must be an autoregressive_greedy generation")
    positions, top_k = golden["positions"], min(COMPACT_TOP_K, golden["vocabulary_size"])
    arrays = {name: np.empty((positions, top_k) if per_rank else positions, dtype=dtype)
              for name, dtype, per_rank in COMPACT_FIELDS}
    for position, token_id in enumerate(golden["token_ids"]):
        row = np.asarray(golden["logits"][position], dtype=np.float64)
        require(np.isfinite(row).all(), f"position {position}: nonfinite logit")
        ids = top_ids(row, top_k)
        require(token_id == int(ids[0]), f"position {position}: golden T=0 token is not its lowest-ID argmax")
        maximum = float(np.max(row))
        shifted = row - maximum
        normalizer = math.log(math.fsum(np.exp(shifted).tolist()))
        log_row = shifted - normalizer
        arrays["top_ids"][position] = ids
        arrays["top_logprobs"][position] = log_row[ids]
        arrays["logsumexp"][position] = maximum + normalizer
        arrays["golden_logprob"][position] = log_row[token_id]
        arrays["top1_margin"][position] = float(row[ids[0]] - row[ids[1]]) if top_k > 1 else math.inf
        arrays["tail_logprob"][position] = outside_log_mass(log_row, ids)
    return golden, top_k, arrays


def write_compact(golden_directory, output):
    golden, top_k, arrays = compact_reference(golden_directory)
    with Path(output).open("xb") as stream:
        stream.write(COMPACT_HEADER.pack(COMPACT_MAGIC, COMPACT_VERSION, top_k, golden["positions"],
                                         golden["vocabulary_size"], COMPACT_HEADER.size, 0))
        for name, dtype, _ in COMPACT_FIELDS:
            stream.write(np.ascontiguousarray(arrays[name], dtype=dtype).tobytes())
    return dict(positions=golden["positions"], top_k=top_k, vocabulary_size=golden["vocabulary_size"],
                bytes=Path(output).stat().st_size, sha256=digest_file(output))


def read_compact(path, positions, vocabulary_size):
    data = Path(path).read_bytes()
    require(len(data) >= COMPACT_HEADER.size, f"{path}: shorter than the compact header")
    magic, version, top_k, rows, vocabulary, header_bytes, reserved = COMPACT_HEADER.unpack_from(data)
    require(magic == COMPACT_MAGIC and version == COMPACT_VERSION and header_bytes == COMPACT_HEADER.size and
            reserved == 0 and not any(data[COMPACT_HEADER.size - 32:COMPACT_HEADER.size]),
            f"{path}: not a version {COMPACT_VERSION} compact reference")
    require(rows == positions and vocabulary == vocabulary_size, f"{path}: header disagrees with the metadata")
    require(MINIMUM_VOCABULARY <= top_k <= vocabulary, f"{path}: invalid top_k")
    arrays, offset = {}, COMPACT_HEADER.size
    for name, dtype, per_rank in COMPACT_FIELDS:
        count = rows * top_k if per_rank else rows
        require(len(data) >= offset + count * np.dtype(dtype).itemsize, f"{path}: truncated at {name}")
        arrays[name] = np.frombuffer(data, dtype, count, offset).reshape((rows, top_k) if per_rank else rows)
        offset += count * np.dtype(dtype).itemsize
    require(len(data) == offset, f"{path}: bytes after the last array")
    ids, logprobs = arrays["top_ids"], arrays["top_logprobs"]
    require(((ids >= 0) & (ids < vocabulary)).all(), f"{path}: top ID outside the vocabulary")
    require(all(np.unique(row).size == top_k for row in ids), f"{path}: repeated top ID")
    require(np.isfinite(logprobs).all() and (logprobs <= 0).all(), f"{path}: invalid top log probability")
    descending = logprobs[:, :-1] > logprobs[:, 1:]
    tied_in_order = (logprobs[:, :-1] == logprobs[:, 1:]) & (ids[:, :-1] < ids[:, 1:])
    require((descending | tied_in_order).all(), f"{path}: top IDs are not highest first with ties by lowest ID")
    for name in ("logsumexp", "golden_logprob", "top1_margin"):
        require(np.isfinite(arrays[name]).all(), f"{path}: nonfinite {name}")
    require((arrays["top1_margin"] >= 0).all() and (arrays["golden_logprob"] <= 0).all(),
            f"{path}: invalid margin or golden log probability")
    tails = arrays["tail_logprob"]
    require(((tails <= 0) | np.isneginf(tails)).all() and not np.isnan(tails).any(), f"{path}: invalid tail")
    for position in range(rows):
        mass = math.fsum(np.exp(logprobs[position]).tolist()) + math.exp(tails[position])
        require(abs(mass - 1) <= COMPACT_MASS_TOLERANCE, f"position {position}: compact mass {mass} is not 1")
    return top_k, arrays


def score_compact(golden_directory, candidate_directory, certain_margin=DEFAULT_CERTAIN_MARGIN,
                  free_token_ids_path=None):
    golden, candidate, native_drawn = aligned_captures(golden_directory, candidate_directory, certain_margin,
                                                       golden_logits=False)
    top_k, compact = read_compact(Path(golden_directory) / COMPACT_FILE, golden["positions"],
                                  golden["vocabulary_size"])
    rows = []
    for position, token_id in enumerate(golden["token_ids"]):
        comparison, candidate_top = candidate_row(candidate, position, native_drawn)
        ids, golden_log = compact["top_ids"][position], compact["top_logprobs"][position]
        golden_top = int(ids[0])
        require(token_id == golden_top, f"position {position}: golden T=0 token is not its lowest-ID argmax")
        golden_logprob = float(compact["golden_logprob"][position])
        require(golden_logprob == float(golden_log[0]), f"position {position}: golden log probability disagrees")
        margin = float(compact["top1_margin"][position])
        candidate_log = log_distribution(comparison)
        candidate_nll = -float(candidate_log[token_id])
        candidate_ids = top_ids(comparison, top_k)
        golden_tail, candidate_tail = float(compact["tail_logprob"][position]), outside_log_mass(candidate_log, ids)
        require(math.isfinite(candidate_tail) or not math.isfinite(golden_tail),
                f"position {position}: the candidate has no mass outside the golden top-k")
        divergence = math.fsum((np.exp(golden_log) * (golden_log - candidate_log[ids])).tolist())
        if math.isfinite(golden_tail):
            divergence += math.exp(golden_tail) * (golden_tail - candidate_tail)
        require(divergence >= -KL_ROUNDING_TOLERANCE,
                f"position {position}: materially negative top-k KL ({divergence})")
        rows.append(dict(position=position, absolute_position=len(golden["prompt_ids"]) - 1 + position,
                         golden_token_id=token_id, golden_top1=golden_top, candidate_top1=candidate_top,
                         top1_equal=golden_top == candidate_top, golden_top1_margin=margin,
                         certain=margin > certain_margin, golden_nll=-golden_logprob, candidate_nll=candidate_nll,
                         candidate_perplexity=perplexity(candidate_nll),
                         candidate_perplexity_overflow=candidate_nll > MAXIMUM_EXPONENT,
                         top_k_overlap=np.intersect1d(ids, candidate_ids).size / top_k,
                         golden_token_in_candidate_top_k=bool(token_id in candidate_ids),
                         kl_top_k_tail=max(0.0, divergence)))
    in_top_k = [row["golden_token_in_candidate_top_k"] for row in rows]
    result = dict(schema_version=SCHEMA_VERSION, reference="compact", golden_directory=golden["directory"],
                  candidate_directory=candidate["directory"], positions=len(rows),
                  vocabulary_size=golden["vocabulary_size"], top_k=top_k, temperature=0,
                  probability_temperature=1.0,
                  metrics_precision="candidate: F64 max-shifted log probabilities, math.fsum, full vocabulary; "
                                    "golden: the compact reference's F64 values",
                  alignment=alignment_summary(golden, candidate), **likelihood_summary(rows),
                  top_k_agreement=dict(top_k=top_k, overlap_mean=math.fsum(row["top_k_overlap"] for row in rows) /
                                       len(rows), golden_token_in_candidate_top_k=sum(in_top_k),
                                       golden_token_in_candidate_top_k_percent=PERCENT * sum(in_top_k) / len(rows)),
                  kl_top_k_tail=dict(distribution([row["kl_top_k_tail"] for row in rows]),
                                     definition=f"KL(golden || candidate) over the golden's top {top_k} IDs and one "
                                                "bucket holding the rest of the vocabulary",
                                     bound="a lower bound of the full-vocabulary KL (data-processing inequality)"),
                  not_available="max/RMS logit deltas and the full-vocabulary KL need the full golden logits",
                  metric_distributions={name: distribution([row[name] for row in rows]) for name in
                                        ("candidate_nll", "golden_top1_margin")},
                  **certainty_summary(rows, certain_margin), per_position=rows)
    return add_receipts(result, golden, golden_directory, candidate_directory, CAPTURE_FILES + (COMPACT_FILE,),
                        native_drawn, free_token_ids_path)

def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("golden", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--certain-margin", type=float, default=DEFAULT_CERTAIN_MARGIN)
    parser.add_argument("--free-token-ids", type=Path, help="separate T=0 free-run IDs for divergence reporting")
    parser.add_argument("--require-witness", action="store_true", help="compatibility spelling; input witnesses are always required")
    parser.add_argument("--output", type=Path, help="write JSON once; an existing output is refused")
    parser.add_argument("--compact", action="store_true", help=f"score against the golden's {COMPACT_FILE}")
    parser.add_argument("--write-compact", type=Path, metavar="PATH",
                        help=f"write the golden's compact reference ({COMPACT_FILE} layout) once and stop")
    options = parser.parse_args(arguments)
    if options.write_compact is not None:
        require(options.candidate is None and not options.compact, "--write-compact takes the golden alone")
        print(json.dumps(write_compact(options.golden, options.write_compact), sort_keys=True))
        return 0
    require(options.candidate is not None, "a candidate capture is required")
    if options.output is not None:
        require(not options.output.exists(), f"output exists already: {options.output}")
    if options.free_token_ids is not None:
        require(options.free_token_ids.is_file(), f"free-running IDs file is missing: {options.free_token_ids}")
    result = (score_compact if options.compact else score)(options.golden, options.candidate, options.certain_margin,
                                                           options.free_token_ids)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if options.output is None:
        sys.stdout.write(text)
    else:
        with options.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
        names = ("top_k_agreement", "kl_top_k_tail") if options.compact else \
            ("max_absolute_logit_delta", "rms_logit_delta", "kl_golden_candidate")
        print(json.dumps({name: result[name] for name in
                          ("positions", "candidate_nll", "candidate_perplexity", "top1", "certain", "uncertain",
                           "free_running") + names}, sort_keys=True, allow_nan=False))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, OverflowError, KeyError, TypeError) as error:
        print(f"golden_score refused: {error}", file=sys.stderr)
        raise SystemExit(1)
