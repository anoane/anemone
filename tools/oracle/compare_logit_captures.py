#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How far the logits of one forward capture lie from another's over the same tokens, position by position.

Takes two capture folders written by reference_oracle.py --mode forward --all-logits: the BASELINE (for the
question this was written for, the reference with the checkpoint's own weights) and the OTHER (the same reference
with the weights of a GGUF, or its control: the checkpoint's own values multiplied through the stand-in a GGUF
capture uses, see gguf_weights.py). Either may also be a capture of port_oracle_capture.py --capture logits: the
exllamav3 port with its own weight format, which differs from the reference in its arithmetic as well as in its
weights. For every position p that has a next token, 0 to n - 2, with t
the token at p + 1:
  * KL(baseline || other) and KL(other || baseline) over the whole vocabulary, in nats: each row of logits is
    turned into log-probabilities in float64 (the row minus its maximum, minus the logarithm of the sum of the
    exponentials), and KL(P || Q) is the sum of P (log P - log Q). A sum that rounding leaves below zero is zero;
  * whether the baseline's most likely token is a most likely token of the other too (none of the other's logits
    is strictly larger than its), and whether it is among the other's 5 and 10 most likely (fewer than 5, or 10,
    are strictly larger). Of a row's tied largest logits the baseline's "most likely token" is the lowest id: an
    engine that returns half-precision logits (the port) ties now and then, so a tie counts as agreement here,
    and how often each capture ties at its top is counted beside it;
  * whether t is a most likely token of each row (its logit is the row's largest);
  * the negative log-likelihood of t under each row, and their difference, other minus baseline.
Over all scored positions, and over each quarter of them in order: mean, median, 90th and 99th percentile
(numpy's, linear between neighbours) and maximum of each KL; the shares of the yes-or-no scores; the mean
negative log-likelihood and the perplexity (its exponential) of each capture and the ratio of the perplexities,
other over baseline; mean, median, percentiles and maximum of the difference of the negative log-likelihoods and
the mean of its magnitude. The table is printed; --output-json receives it with the scores of every position.

    compare_logit_captures.py BASELINE_FOLDER OTHER_FOLDER --output-json FILE

Refused: a capture whose manifest does not say complete and forward (or logits, the port's), that has no logits of
every row or whose tensors.bin is not as long as its manifest says; two captures of different token ids, lengths
or vocabularies; a sequence of one token; a logit that is not finite; an output file that exists already.
What it does not show: which capture is right (it measures a distance); anything beyond these tokens, which are
teacher-forced from position 0; the last position, which has no next token and is not scored. Two captures of
the SAME weights measure the noise floor of the reference on this machine, which every other figure should be
read against.
Measurement tool: not part of the engine.
"""
import argparse
import json
import math
import os
import sys

import numpy

TENSORS_FILE_NAME = "tensors.bin"
MANIFEST_FILE_NAME = "manifest.json"
FORWARD_CAPTURE = "forward"                  # manifest["conditions"]["capture"] of reference_oracle.py --mode forward
PORT_LOGITS_CAPTURE = "logits"               # and of port_oracle_capture.py --capture logits: the same two records
TOKEN_IDS_RECORD, TOKEN_IDS_DTYPE, TOKEN_IDS_DIMENSIONS = "input.token_ids", "int64", 2       # [sequence, position]
LOGITS_RECORD, LOGITS_DTYPE, LOGITS_DIMENSIONS = "logits.all_rows", "float32", 3   # [sequence, position, vocabulary]
LITTLE_ENDIAN_DTYPES = {"int64": "<i8", "float32": "<f4"}      # tensors.bin holds little-endian bytes
SEQUENCES = 1
FEWEST_TOKENS = 2                            # one position with a next token
SLAB_ROWS = 64                               # positions scored at a time: 66 MB a row set in float64 at 129,280 columns
TOP_RANKS = (5, 10)
PERCENTILES = (50, 90, 99)
POSITION_RANGES = 4                          # the scored positions in order, cut into quarters
KL_SCORES = (("kl_baseline_other", "KL(baseline || other)"), ("kl_other_baseline", "KL(other || baseline)"))
YES_OR_NO_SCORES = (("same_top_token", "same most likely token"),
                    *((f"baseline_top_in_other_top_{rank}", f"baseline's top token in other's top {rank}")
                      for rank in TOP_RANKS),
                    ("baseline_top_is_next_token", "baseline's top token is the next token"),
                    ("other_top_is_next_token", "other's top token is the next token"),
                    ("baseline_top_is_tied", "baseline has more than one top token"),
                    ("other_top_is_tied", "other has more than one top token"))
NLL_SCORES = ("nll_baseline", "nll_other")
JSON_INDENT_SPACES = 1
LABEL_WIDTH, COLUMN_WIDTH = 46, 16
EXIT_CODE_SUCCESS, EXIT_CODE_FAILURE = 0, 1


def load_capture(folder):
    """The token ids [positions], the logits [positions, vocabulary] as a read-only map, and what the capture's
    manifest says of its weights. Anything but a complete capture with logits of every row, the reference's
    (forward) or the port's (logits), is refused."""
    with open(os.path.join(folder, MANIFEST_FILE_NAME), encoding = "utf8") as handle:
        manifest = json.load(handle)
    conditions = manifest.get("conditions") or {}
    if manifest.get("complete") is not True or conditions.get("capture") not in (FORWARD_CAPTURE, PORT_LOGITS_CAPTURE):
        raise ValueError(f"{folder}: the manifest does not say a complete {FORWARD_CAPTURE} capture of the reference "
                         f"or {PORT_LOGITS_CAPTURE} capture of the port")
    tensors_path = os.path.join(folder, TENSORS_FILE_NAME)
    file_bytes = os.path.getsize(tensors_path)
    if sum(record["byte_count"] for record in manifest["tensors"]) != file_bytes:
        raise ValueError(f"{folder}: {TENSORS_FILE_NAME} holds {file_bytes} bytes, the manifest's records another sum")
    records = {record["name"]: record for record in manifest["tensors"]}

    def mapped(name, dtype_name, dimensions):
        record = records.get(name)
        if record is None:
            raise ValueError(f"{folder}: no record {name} (a capture with --all-logits is needed)")
        dtype, shape = numpy.dtype(LITTLE_ENDIAN_DTYPES[dtype_name]), tuple(record["shape"])
        if record["dtype"] != dtype_name or len(shape) != dimensions or shape[0] != SEQUENCES \
                or record["byte_count"] != math.prod(shape) * dtype.itemsize \
                or not 0 <= record["byte_offset"] <= file_bytes - record["byte_count"]:
            raise ValueError(f"{folder}: record {name} is not {dtype_name} of {dimensions} dimensions for "
                             f"{SEQUENCES} sequence, or its bytes are not inside {TENSORS_FILE_NAME}")
        return numpy.memmap(tensors_path, dtype = dtype, mode = "r", offset = record["byte_offset"], shape = shape[1:])

    token_ids = numpy.array(mapped(TOKEN_IDS_RECORD, TOKEN_IDS_DTYPE, TOKEN_IDS_DIMENSIONS))
    logits = mapped(LOGITS_RECORD, LOGITS_DTYPE, LOGITS_DIMENSIONS)
    arguments = conditions.get("arguments") or {}
    sequence = arguments.get("request_index") if conditions["capture"] == PORT_LOGITS_CAPTURE \
        else arguments.get("sequence")       # which entry of its tokens JSON the capture ran
    if conditions["capture"] == PORT_LOGITS_CAPTURE:                     # another engine with its own weight format
        weights = f"{os.path.basename(str(arguments.get('model')).rstrip(os.sep))} in the exllamav3 port"
    elif arguments.get("weights_gguf") is None:
        weights = "the checkpoint's own, in the reference's own kernels"
    else:                                    # a control multiplies the checkpoint's values through the same stand-in
        take, layers = arguments.get("gguf_take"), arguments.get("gguf_expert_layers")
        only = "" if take is None and layers is None else \
            f" (from the file only: {take or 'every part'}{'' if layers is None else f', experts of layers {layers}'})"
        values = "the checkpoint's own values through the stand-in of a GGUF capture" \
            if arguments.get("gguf_control") else os.path.basename(arguments["weights_gguf"]) + only
        weights = f"{values}, linear inputs {arguments.get('gguf_linear_inputs')}"
    return token_ids, logits, {"folder": folder, "weights": weights, "tokens": len(token_ids), "sequence": sequence}


def log_probabilities(logits):
    """float64 rows of logits -> rows of log-probabilities."""
    shifted = logits - logits.max(axis = 1, keepdims = True)
    return shifted - numpy.log(numpy.exp(shifted).sum(axis = 1, keepdims = True))


def score_positions(baseline, other, token_ids):
    """Every score of every position that has a next token, as arrays [positions]."""
    positions = len(token_ids) - 1
    scores = {name: numpy.empty(positions, dtype = numpy.float64) for name in (*dict(KL_SCORES), *NLL_SCORES)}
    scores.update((name, numpy.empty(positions, dtype = bool)) for name, _ in YES_OR_NO_SCORES)
    for start in range(0, positions, SLAB_ROWS):
        rows = slice(start, min(start + SLAB_ROWS, positions))
        first = numpy.asarray(baseline[rows], dtype = numpy.float64)
        second = numpy.asarray(other[rows], dtype = numpy.float64)
        if not (numpy.isfinite(first).all() and numpy.isfinite(second).all()):
            raise ValueError(f"a logit of positions {rows.start} to {rows.stop - 1} is not finite")
        log_first, log_second = log_probabilities(first), log_probabilities(second)
        log_ratio = log_first - log_second
        scores["kl_baseline_other"][rows] = (numpy.exp(log_first) * log_ratio).sum(axis = 1).clip(min = 0.0)
        scores["kl_other_baseline"][rows] = (numpy.exp(log_second) * -log_ratio).sum(axis = 1).clip(min = 0.0)
        row_index, next_tokens = numpy.arange(rows.stop - rows.start), token_ids[rows.start + 1:rows.stop + 1]
        top_first = first.argmax(axis = 1)                                 # of tied logits, the lowest token id
        strictly_larger = (second > second[row_index, top_first][:, None]).sum(axis = 1)
        scores["same_top_token"][rows] = strictly_larger == 0              # a tie at the other's top is no change
        for rank in TOP_RANKS:
            scores[f"baseline_top_in_other_top_{rank}"][rows] = strictly_larger < rank
        scores["baseline_top_is_next_token"][rows] = first[row_index, next_tokens] == first.max(axis = 1)
        scores["other_top_is_next_token"][rows] = second[row_index, next_tokens] == second.max(axis = 1)
        scores["baseline_top_is_tied"][rows] = (first == first.max(axis = 1, keepdims = True)).sum(axis = 1) > 1
        scores["other_top_is_tied"][rows] = (second == second.max(axis = 1, keepdims = True)).sum(axis = 1) > 1
        scores["nll_baseline"][rows] = -log_first[row_index, next_tokens]
        scores["nll_other"][rows] = -log_second[row_index, next_tokens]
    return scores


def spread(values):
    """Mean, the percentiles and the maximum of an array, as plain numbers."""
    return {"mean": float(values.mean()),
            **{f"percentile_{percent}": float(numpy.percentile(values, percent)) for percent in PERCENTILES},
            "maximum": float(values.max())}


def summarize(scores, rows):
    """The figures of one range of positions."""
    difference = scores["nll_other"][rows] - scores["nll_baseline"][rows]
    means = {name: float(scores[name][rows].mean()) for name in NLL_SCORES}
    return {"first_position": rows.start, "last_position": rows.stop - 1, "positions": rows.stop - rows.start,
            **{name: spread(scores[name][rows]) for name, _ in KL_SCORES},
            **{f"share_{name}": float(scores[name][rows].mean()) for name, _ in YES_OR_NO_SCORES},
            "mean_nll_baseline": means["nll_baseline"], "mean_nll_other": means["nll_other"],
            "perplexity_baseline": math.exp(means["nll_baseline"]), "perplexity_other": math.exp(means["nll_other"]),
            "perplexity_ratio_other_over_baseline": math.exp(means["nll_other"] - means["nll_baseline"]),
            "nll_other_minus_baseline": {**spread(difference), "mean_magnitude": float(numpy.abs(difference).mean())}}


def print_table(summaries):
    """One line per figure, one column per range of positions."""
    def line(label, values, layout):
        print(f"{label:{LABEL_WIDTH}s}" + "".join(f"{format(value, layout):>{COLUMN_WIDTH}s}" for value in values))

    ranges = list(summaries.values())
    line("", list(summaries), "s")
    line("positions", [f"{summary['first_position']}..{summary['last_position']}" for summary in ranges], "s")
    for name, label in KL_SCORES:
        for figure in ranges[0][name]:
            line(f"{label} {figure.replace('_', ' ')}", [summary[name][figure] for summary in ranges], ".6f")
    for name, label in YES_OR_NO_SCORES:
        line(f"share: {label}", [summary[f"share_{name}"] for summary in ranges], ".4f")
    for name in ("mean_nll_baseline", "mean_nll_other", "perplexity_baseline", "perplexity_other",
                 "perplexity_ratio_other_over_baseline"):
        line(name.replace("_", " "), [summary[name] for summary in ranges], ".5f")
    for figure in ranges[0]["nll_other_minus_baseline"]:
        line(f"nll other - baseline {figure.replace('_', ' ')}",
             [summary["nll_other_minus_baseline"][figure] for summary in ranges], ".5f")


def compare(baseline_folder, other_folder, output_path):
    if os.path.exists(output_path):
        raise ValueError(f"the output file exists already: {output_path}")
    baseline_tokens, baseline_logits, baseline = load_capture(baseline_folder)
    other_tokens, other_logits, other = load_capture(other_folder)
    if baseline_logits.shape != other_logits.shape or not numpy.array_equal(baseline_tokens, other_tokens):
        raise ValueError(f"the two captures are not of the same token ids: {len(baseline_tokens)} and "
                         f"{len(other_tokens)} tokens, logits of shape {baseline_logits.shape} and "
                         f"{other_logits.shape}")
    tokens, vocabulary = baseline_logits.shape
    if tokens != len(baseline_tokens) or tokens < FEWEST_TOKENS or int(baseline_tokens.min()) < 0 \
            or int(baseline_tokens.max()) >= vocabulary:
        raise ValueError(f"{tokens} rows of logits over {vocabulary} tokens for {len(baseline_tokens)} token ids: at "
                         f"least {FEWEST_TOKENS} tokens, a row for each and ids inside the vocabulary are needed")
    scores = score_positions(baseline_logits, other_logits, baseline_tokens)
    positions = tokens - 1
    summaries = {"all": summarize(scores, slice(0, positions))}
    for number, part in enumerate(numpy.array_split(numpy.arange(positions), POSITION_RANGES), 1):
        if len(part):
            summaries[f"quarter_{number}"] = summarize(scores, slice(int(part[0]), int(part[-1]) + 1))
    print(f"baseline: {baseline['folder']} ({baseline['weights']})")
    print(f"other:    {other['folder']} ({other['weights']})")
    print(f"{tokens} tokens, {positions} positions scored, {vocabulary} tokens in the vocabulary; KL in nats")
    print_table(summaries)
    with open(output_path, "x", encoding = "utf8") as handle:
        json.dump({"baseline": baseline, "other": other, "tokens": tokens, "positions_scored": positions,
                   "vocabulary": vocabulary, "kl_unit": "nat", "percentiles": list(PERCENTILES),
                   "summary": summaries, "per_position": {name: values.tolist() for name, values in scores.items()}},
                  handle, indent = JSON_INDENT_SPACES)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev = False)
    parser.add_argument("baseline_folder", help = "capture with the logits of every row: the weights compared against")
    parser.add_argument("other_folder", help = "capture with the logits of every row, of the same token ids")
    parser.add_argument("--output-json", required = True, help = "receives the table and the scores of every position")
    arguments = parser.parse_args()
    try:
        compare(arguments.baseline_folder, arguments.other_folder, arguments.output_json)
    except (ValueError, KeyError, OSError) as failure:
        print(f"compare_logit_captures FAILED: {type(failure).__name__}: {failure}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
