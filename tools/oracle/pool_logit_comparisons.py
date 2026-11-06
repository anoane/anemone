#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pools the comparisons compare_logit_captures.py wrote for several sequences of one text into one set of figures.

    pool_logit_comparisons.py COMPARISON_JSON... --output-json FILE [--first-position 512]
                              [--published-other VALIDATION_JSON]

Each COMPARISON_JSON is the --output-json of compare_logit_captures.py for one sequence (a chunk of the text),
all of the same pair of weights, given in the order of the chunks; a chunk is printed under the number of its
sequence in the tokens JSON its captures ran (its place among the arguments where a comparison does not say). Of
every chunk the positions from
--first-position on are used (0 by default: all of them); with chunks of 1,024 tokens and --first-position 512
these are the 511 positions per chunk that llama.cpp's perplexity tool scores, each given at least 512 tokens of
context. Printed, per chunk and over all chunks pooled position by position: the perplexity of each capture and
their ratio, the mean of each KL, and the share of positions where both captures have the same most likely token
(the sums of the negative log-likelihoods go to --output-json); pooled also the median, the 90th and 99th
percentile and the maximum of KL(baseline || other), three more yes-or-no shares (the baseline's top token among
the other's top 5, each capture's top token being the next token), and the mean over chunks of the logarithm of
the perplexity ratio with its standard error (the chunks as the sample: positions of one chunk depend on each
other, chunks of one text much less).
--published-other names a JSON with "chunks": [{"chunk", "scored_tokens", "nll", "ppl"}, ...], chunks 0, 1, ... in
order, that another runtime published for the OTHER capture's weights over the same positions (the validation
file of the GSQ-RCO release): its perplexities are printed beside the OTHER capture's own, with their ratio. The
two need not be equal: that runtime computes the same weights in a float32 graph with float32 key caches, a
reference capture keeps the reference's bfloat16 streams and quantized keys. A ratio near 1 on every chunk shows
that both read the same weights the same way; it is not a bit-level proof.

Refused: comparison files that are not of the same two kinds of weights or of different lengths, the same file
twice, a --first-position that leaves no position, a published file with another number of chunks or of scored
tokens, whose chunks are not numbered in order or are not these sequences, an output file that exists already.
What it does not show: anything about text other than this one; the standard error over a handful of chunks is
itself a rough figure.
Measurement tool: not part of the engine.
"""
import argparse
import json
import math
import os
import sys

import numpy

PERCENTILES = (50, 90, 99)
KL_SCORES = ("kl_baseline_other", "kl_other_baseline")
SHARE_SCORES = ("same_top_token", "baseline_top_in_other_top_5", "baseline_top_is_next_token",
                "other_top_is_next_token")
TIE_SCORES = ("baseline_top_is_tied", "other_top_is_tied")       # absent from comparisons made before 2026-10-11
NLL_SCORES = ("nll_baseline", "nll_other")
FEWEST_CHUNKS_FOR_A_STANDARD_ERROR = 2
JSON_INDENT_SPACES = 1
EXIT_CODE_SUCCESS, EXIT_CODE_FAILURE = 0, 1


def read_comparison(path, first_position):
    """The scores of one chunk from --first-position on, and what the file says was compared."""
    with open(path, encoding = "utf8") as handle:
        comparison = json.load(handle)
    scores = {name: numpy.asarray(comparison["per_position"][name], dtype = numpy.float64)[first_position:]
              for name in (*KL_SCORES, *SHARE_SCORES, *NLL_SCORES, *TIE_SCORES) if name in comparison["per_position"]}
    positions = int(comparison["positions_scored"])
    if positions <= first_position or any(len(values) != positions - first_position for values in scores.values()):
        raise ValueError(f"{path}: {positions} positions scored, none left from position {first_position} on, or "
                         "its per-position scores have another length")
    pair = (comparison["baseline"]["weights"], comparison["other"]["weights"], positions)
    sequences = {comparison["baseline"].get("sequence"), comparison["other"].get("sequence")}
    return scores, pair, sequences.pop() if len(sequences) == 1 else None   # the entry of the tokens JSON, if known


def figures(scores):
    """Sums, perplexities, mean KLs and shares of a set of positions."""
    positions = len(scores["nll_baseline"])
    sums = {name: float(scores[name].sum()) for name in NLL_SCORES}
    return {"positions": positions, "nll_sum_baseline": sums["nll_baseline"], "nll_sum_other": sums["nll_other"],
            "perplexity_baseline": math.exp(sums["nll_baseline"] / positions),
            "perplexity_other": math.exp(sums["nll_other"] / positions),
            "log_perplexity_ratio_other_over_baseline": (sums["nll_other"] - sums["nll_baseline"]) / positions,
            **{f"mean_{name}": float(scores[name].mean()) for name in KL_SCORES},
            **{f"share_{name}": float(scores[name].mean()) for name in (*SHARE_SCORES, *TIE_SCORES) if name in scores}}


def pool(paths, first_position, published_path, output_path):
    if os.path.exists(output_path):
        raise ValueError(f"the output file exists already: {output_path}")
    if len({os.path.realpath(path) for path in paths}) != len(paths):
        raise ValueError("a comparison file is given twice: a chunk counts once")
    chunks, pairs, sequences = zip(*(read_comparison(path, first_position) for path in paths))
    labels = [str(number) if sequence is None else str(sequence) for number, sequence in enumerate(sequences)]
    if len(set(pairs)) != 1:
        raise ValueError(f"the comparison files are not of one pair of weights and one length: {sorted(set(pairs))}")
    per_chunk = [figures(scores) for scores in chunks]
    pooled_scores = {name: numpy.concatenate([scores[name] for scores in chunks]) for name in chunks[0]
                     if all(name in scores for scores in chunks)}
    pooled = figures(pooled_scores)
    pooled["kl_baseline_other"] = {
        **{f"percentile_{percent}": float(numpy.percentile(pooled_scores["kl_baseline_other"], percent))
           for percent in PERCENTILES}, "maximum": float(pooled_scores["kl_baseline_other"].max())}
    log_ratios = numpy.asarray([chunk["log_perplexity_ratio_other_over_baseline"] for chunk in per_chunk])
    pooled["mean_over_chunks_of_log_perplexity_ratio"] = float(log_ratios.mean())
    pooled["its_standard_error"] = float(log_ratios.std(ddof = 1) / math.sqrt(len(log_ratios))) \
        if len(log_ratios) >= FEWEST_CHUNKS_FOR_A_STANDARD_ERROR else None
    published = None
    if published_path is not None:
        with open(published_path, encoding = "utf8") as handle:
            published = json.load(handle)["chunks"]
        numbers = [theirs["chunk"] for theirs in published]
        if len(published) != len(per_chunk) or numbers != list(range(len(published))) \
                or labels != [str(number) for number in numbers] \
                or any(int(theirs["scored_tokens"]) != ours["positions"] for theirs, ours in zip(published, per_chunk)):
            raise ValueError(f"{published_path}: chunks {numbers} of "
                             f"{sorted({int(theirs['scored_tokens']) for theirs in published})} scored tokens; "
                             f"these are the sequences {labels} of {per_chunk[0]['positions']} positions")
    baseline_weights, other_weights, positions = pairs[0]
    print(f"baseline: {baseline_weights}\nother:    {other_weights}")
    print(f"{len(paths)} chunks of {positions} scored positions, used from position {first_position} on: "
          f"{pooled['positions']} positions; KL in nats")
    print(f"{'chunk':>6s}{'positions':>10s}{'ppl baseline':>14s}{'ppl other':>12s}{'ratio':>9s}"
          f"{'KL(b||o)':>11s}{'KL(o||b)':>11s}{'same top':>10s}" + ("" if published is None else
                                                                  f"{'published ppl':>15s}{'other / published':>19s}"))
    for number, chunk in enumerate((*per_chunk, pooled)):
        label = "all" if chunk is pooled else labels[number]
        line = (f"{label:>6s}{chunk['positions']:10d}{chunk['perplexity_baseline']:14.5f}"
                f"{chunk['perplexity_other']:12.5f}{math.exp(chunk['log_perplexity_ratio_other_over_baseline']):9.5f}"
                f"{chunk['mean_kl_baseline_other']:11.6f}{chunk['mean_kl_other_baseline']:11.6f}"
                f"{chunk['share_same_top_token']:10.4f}")
        if published is not None:
            theirs = math.exp(sum(float(entry["nll"]) for entry in published) / pooled["positions"]) \
                if chunk is pooled else float(published[number]["ppl"])
            chunk["perplexity_other_published"] = theirs
            line += f"{theirs:15.5f}{chunk['perplexity_other'] / theirs:19.5f}"
        print(line)
    spread = pooled["kl_baseline_other"]
    print("KL(baseline || other) over all positions: " + ", ".join(
        f"percentile {percent} {spread[f'percentile_{percent}']:.6f}" for percent in PERCENTILES)
        + f", maximum {spread['maximum']:.6f}")
    print("shares over all positions: " + ", ".join(
        f"{name.replace('_', ' ')} {pooled[f'share_{name}']:.4f}" for name in (*SHARE_SCORES, *TIE_SCORES)
        if f"share_{name}" in pooled))
    error = pooled["its_standard_error"]
    print("logarithm of the perplexity ratio, mean over chunks: "
          f"{pooled['mean_over_chunks_of_log_perplexity_ratio']:+.5f}"
          + ("" if error is None else f" with a standard error of {error:.5f}"))
    with open(output_path, "x", encoding = "utf8") as handle:
        json.dump({"baseline": baseline_weights, "other": other_weights, "comparisons": list(paths),
                   "first_position": first_position, "kl_unit": "nat", "published_other": published_path,
                   "chunks": per_chunk, "all": pooled}, handle, indent = JSON_INDENT_SPACES)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev = False)
    parser.add_argument("comparisons", nargs = "+", help = "output files of compare_logit_captures.py, in chunk order")
    parser.add_argument("--output-json", required = True, help = "receives the figures of every chunk and of all")
    parser.add_argument("--first-position", type = int, default = 0, help = "the first position of a chunk that counts")
    parser.add_argument("--published-other", default = None, help = "another runtime's chunks for the other's weights")
    arguments = parser.parse_args()
    if arguments.first_position < 0:
        parser.error("--first-position is below 0")
    try:
        pool(arguments.comparisons, arguments.first_position, arguments.published_other, arguments.output_json)
    except (ValueError, KeyError, OSError) as failure:
        print(f"pool_logit_comparisons FAILED: {type(failure).__name__}: {failure}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
