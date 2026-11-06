#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Holds a capture of Anemone to the u16 reference captures of stepwise_capture.py: the gates of the first milestone
(design/first_token_plan.md 3.2 to 3.4, unit U13b).

Both folders are in the format of reference_oracle.py (tensors.bin and manifest.json; bfloat16 values held as
float32) with the record names that stepwise_capture.py gives (its module text lists them). The engine's folders
(unit U14's teacher-forced harness, `anemone generate`) use the same names, shapes and dtypes, the records one after
another in tensors.bin with no gap; each gate below names what the ENGINE's folder must hold, and a record it names
that is missing is a FAIL. FLOOR is the JSON of stepwise_capture.py --mode floor, measured on the same text before
any engine run, with the figures below (FLOOR_CONSTANTS) recorded in it: a floor made with other figures is refused,
so none changes without a new floor, which takes the owner's word once an engine run was judged by the old one; one
made before FLIP_SIGMAS replaced the epsilon rule (2026-10-18) records that rule's figures in its place
(EPSILON_RULE_CONSTANTS) and is taken, as what it measured is the same. Its digest is printed. One gate per run:
  layer           M1a (3.3). ORACLE: the stepwise capture of the text; ENGINE: every layer teacher-forced on it,
                  holding input.token_ids, input.embeddings, layer_L.block.output [1, p, 4, 5120],
                  layer_L.mlp.routed_expert_ids [p, 6] and layer_L.mlp.biased_scores [p, 384] of every layer, and
                  layer_L.position_P.attn_site.normed at every capture point the oracle holds. Per layer, the routed
                  experts of each row are the six best of its biased scores, the engine's of its own, and over the
                  rows (positions) whose six are the reference's six: the mean relative rms of the streams leaving the
                  layer at most the larger of twice the floor's mean at that layer and A_MIN, and no row above the
                  larger of four times the floor's largest row and 4 A_MIN. Over the rows whose experts differ: the
                  engine's flips E, its picks that the reference did not make (one per position and slot), and the
                  floor's flips F at that layer are held as two counts of one rate: E - F at most z sqrt(E + F) for
                  z = FLIP_SIGMAS, that is E at most F + (z^2 + z sqrt(z^2 + 8 F)) / 2, and each flip (FLIP) has a
                  margin, the reference's biased score of its sixth pick less that of the engine's pick, of at most
                  four times the floor's largest router-score difference at that layer (the floor's own flips lie
                  within twice it); their streams are not compared. POOLED: the same count rule on the sums of E and F
                  over the layers that held. In place: the embeddings of every position bit for bit, and at every
                  capture point the attention site's normed input (the first kernel after the injected streams) within
                  one bfloat16 place on 99.9% of its values. Every other record both hold, except the frequencies
                  (table gate), logits (teacher_forced gate) and generated ids (tokens gate), is printed as INFO with
                  its figures, also where its shape or dtype is another (the prompt branch's window at position 0, the
                  ring's 128 slots).
  table           The rotation step's tables (3.2, the table gate of 3.5). ORACLE: a stepwise capture of
                  --frequencies; ENGINE: every frequencies record it holds, at the same sampled positions: the words
                  of each theta bit for bit, cos and sin within one float32 place. No floor.
  tokens          M1b rules 1 and 3 (3.4). ORACLE: a stepwise capture with generated tokens, or a request
                  reference; ENGINE: the engine's run of the same prompt, holding input.token_ids, generate.token_ids
                  (ending with the end token where the run stopped on it, as the oracle's) and logits.all_rows, or
                  top lists that hold the reference's two best ids, of every position before the first difference.
                  The generated ids are equal and as many, or the first difference comes at a step where the
                  reference's margin m (between its two best logits) is below delta, twice the twin's 99.9th
                  percentile of |dm| (after it the runs part, so their lengths are not compared); over the positions
                  before that step, prompt included, the 99th percentile of the engine's |dm| (the change of m when
                  the engine's logits of the same two tokens are read) is at most twice the twin's.
  teacher_forced  M1b rule 2. ORACLE: the stepwise text with the logits of every position; ENGINE: the engine
                  teacher-forced on the same tokens, holding input.token_ids and logits.all_rows: mean KL(u16
                  stepwise || engine) at most twice the twin's, both with their chunk standard errors, and the same
                  top token at no fewer positions than the twin's share less 2 points.
  repeat          M1b rule 4 and M1c rule 5. ORACLE and ENGINE: two captures of the engine (a second run on the same
                  layout; steps of 8 rows against steps of 1): every record of either, bit for bit, in both. No floor.
  long_context    G8 (wide_prefill.md 8). ORACLE: stepwise_capture.py --mode long, the u16 prompt path of a long text
                  with teacher-forced steps after it (no twin's); FLOOR: the twin's JSON of the same run (the same
                  mode with --floor-gguf: the oracle's identity but for the twin, and its flagged positions); ENGINE:
                  the engine's run of the same tokens, holding input.token_ids, logits.flagged_positions and
                  logits.flagged_rows (the logits of every flagged position, whole), and per index layer it selects
                  at layer_L.index.picks with the rows it holds of the oracle's, layer_L.index.positions or else
                  index.sampled_positions (layer_L.index.tied_at_cut and .picked_at_cut where it notes its own ties).
                  It holds every sampled row the layer selects for: all on a layer of rate 2; on a direct layer L
                  (rate 1, DIRECT_RATE) those of its cone, from F + (n - 1 - L)(w - 1) rows before the prompt's end
                  (F flagged rows, n layers, a window of w: 2,541 rows at layer 20, wide_prefill.md 2.1); a row it
                  lacks there fails the layer (missing_rows). Over the flagged positions: mean KL at most twice the
                  twin's and the same top token no fewer than the twin's share less 2 points, as teacher_forced; at
                  layers LONG_CONTEXT_LAYERS, over the sampled rows both hold whose 512th score ties no more entries
                  than were picked of them (on either side; the others are counted apart, those of the engine's notes
                  alone also as engine_tied_rows), the oracle's picks the engine picked too: a mean share of at least
                  PICKS_SHARED_LEAST. Every other index layer both note is printed as INFO.

    compare_engine_capture.py --gate layer --oracle FOLDER --engine FOLDER --floor FLOOR_JSON [--output-json FILE]
    compare_engine_capture.py --gate table --oracle FOLDER --engine FOLDER [--break word]

Printed: GATE (layer gate, one per layer, then a FLIP line per flip, then one POOLED), RECORD (one per record, layer
and capture position: verdict PASS, FAIL or INFO, the largest error of any row included), MARGIN and FIGURE (the
token and logit rules), PICKS (long_context, one per index layer), VERDICT (one, last).
A break (--break) changes the ENGINE's data in memory so that its gate must fail on real captures: stream (layer
gate: the streams leaving layer 20 scaled by 1 + 2^-5), word (table gate: one frequency word one place off), token
(tokens gate: the generated id at the step of the reference's largest margin replaced by its second-best token),
logits (teacher_forced gate: 4 nats taken from the reference's top token in every row of the engine's), bit
(repeat gate: one bit of the engine's first record), flagged (long_context: the logits break on its flagged rows),
picks (long_context: every other pick of layer 20's rows taken away). The verdict then says reported for what
failed as asked, and the exit code is 0 only where the gate failed and nothing else did.
Refused: a manifest that does not say complete; a record outside tensors.bin, of a length that is not its shape's,
of an unknown dtype, given twice, or with a gap before it; captures of other token ids or sampled positions; a
value that is not finite; a bfloat16 record whose float32 words have low bits; a floor of another text or made with
other figures; for long_context, an oracle that is no u16 long capture, a floor of another run of it, notes that
are not one a noted row; an output file that exists. Exit 0 when the gate passed (or a break reported as asked), 1
otherwise.
Measurement tool: not part of the engine.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

import numpy

import compare_logit_captures as compare
import state_precision_true as unit_13b

TENSORS_FILE_NAME, MANIFEST_FILE_NAME = compare.TENSORS_FILE_NAME, compare.MANIFEST_FILE_NAME
LITTLE_ENDIAN_DTYPES = {"float16": "<f2", "float32": "<f4", "float64": "<f8", "int8": "i1", "uint8": "u1",
                        "int16": "<i2", "int32": "<i4", "int64": "<i8"}
GATES = (LAYER, TABLE, TOKENS, TEACHER_FORCED, REPEAT, LONG_CONTEXT) = ("layer", "table", "tokens", "teacher_forced",
                                                                       "repeat", "long_context")
GATES_WITHOUT_FLOOR = (TABLE, REPEAT)
BREAKS = {"stream": LAYER, "word": TABLE, "token": TOKENS, "logits": TEACHER_FORCED, "bit": REPEAT,
          "flagged": LONG_CONTEXT, "picks": LONG_CONTEXT}
BREAK_LAYER = 20                             # 3.5: the layer gate names layer 20; long_context's picks break too
BREAK_STREAM_SCALE = 1 + 2.0 ** -5           # a relative rms of 2^-5: 32 times A_MIN
BREAK_LOGIT_SHIFT = 4.0                      # nats taken from each row's top token
# ---- the gates' figures (first_token_plan.md 3.3, 3.4); fixed before any engine run ----
A_MIN = 2.0 ** -10                           # relative rms: about a tenth of the values one bfloat16 place off
MEAN_FACTOR, LARGEST_FACTOR = 2, 4           # of the floor's mean and largest row at the layer (and of A_MIN), and of
                                             # its largest router-score difference (a flip's margin)
FLIP_SIGMAS = 3.5                            # a flip is a near tie that noise breaks, rare among the 12,288 picks of
                                             # a layer: E and F as Poisson counts, E - F against its spread from both,
                                             # sqrt(E + F). Exact: a floor drawn again fails a layer, or the pooled
                                             # sums, with probability below 2.33e-4 at any rate (the normal tail,
                                             # approached from below), one of 40 layers 0.65% of the time at this
                                             # floor's rates; F + c sqrt(F + 1) takes c = 8 for 0.45% and is looser
                                             # above 20 flips. With the margin it replaces 3.3's epsilon, which the
                                             # floor itself failed at layers 2 and 7, and its 0.5% of picks, which the
                                             # floor holds to within one pick at layer 28
ROUTER_PERCENTILE = 99.9                     # the floor's percentile of the router-score difference (printed)
# what a floor made before FLIP_SIGMAS holds in its place (the module text)
EPSILON_RULE_CONSTANTS = {"EPSILON_FACTOR": 2, "EPSILON_MIN": 2.0 ** -16, "DIFFERING_PAIRS_LIMIT": 0.005}
IN_PLACE_PLACES, IN_PLACE_SHARE = 1, 0.999   # 3.2, mix finish: within one bfloat16 place on 99.9% of the values
COS_SIN_PLACES = 1                           # 3.2, rotation step: within one float32 place of freqs_cis
DELTA_FACTOR, DELTA_PERCENTILE = 2, 99.9     # delta: twice the twin's 99.9th percentile of |dm|
MARGIN_FACTOR, MARGIN_PERCENTILE = 2, 99     # rule 1: the engine's 99th percentile of |dm| at most twice the twin's
KL_FACTOR, SAME_TOP_SHORTFALL = 2, 0.02      # rule 2: twice the twin's mean KL; its same-top share less 2 points
FLOOR_CONSTANTS = ("A_MIN", "MEAN_FACTOR", "LARGEST_FACTOR", "FLIP_SIGMAS", "ROUTER_PERCENTILE", "IN_PLACE_PLACES",
                   "IN_PLACE_SHARE", "COS_SIN_PLACES", "DELTA_FACTOR", "DELTA_PERCENTILE", "MARGIN_FACTOR",
                   "MARGIN_PERCENTILE", "KL_FACTOR", "SAME_TOP_SHORTFALL")
LONG_CONTEXT_LAYERS = (2, 8, 14, 20)         # G8 (wide_prefill.md 8): the picks judged at these layers
PICKS_SHARED_LEAST = 0.99                    # G8: the mean share of the u16 arm's picks at the sampled rows
LONG_CONTEXT_CONSTANTS = ("LONG_CONTEXT_LAYERS", "PICKS_SHARED_LEAST")     # a long-context floor carries these too
DIRECT_RATE = 1                              # wide_prefill.md 2.3: direct layers select only for the rows of the cone
LONG_CAPTURE_MODE = "long"                   # stepwise_capture.py --mode long, the oracle's conditions say it
# ---- record names (stepwise_capture.py's module text) ----
TOKEN_IDS, EMBEDDINGS, ALL_LOGITS = "input.token_ids", "input.embeddings", "logits.all_rows"
TOP_IDS, TOP_VALUES, GENERATED_IDS = "logits.top_ids", "logits.top_values", "generate.token_ids"
FLAGGED_POSITIONS, FLAGGED_ROWS = "logits.flagged_positions", "logits.flagged_rows"
SAMPLED_POSITIONS, INDEX_NOTE = "index.sampled_positions", "layer_{layer}.index.{record}"
INDEX_NOTE_RECORDS = ("picks", "cut_score", "tied_at_cut", "picked_at_cut")
LAYER_POSITIONS, JUDGED_NOTES = "positions", ("picks", "tied_at_cut", "picked_at_cut")  # the engine's rows of a layer
INDEX_PICKS_NAME = re.compile(r"layer_(?P<layer>\d+)\.index\.picks")
LOGITS_PREFIX, FREQUENCIES_PREFIX = "logits.", "frequencies."
NORMED_RECORD = "attn_site.normed"           # the first kernel after the injected streams
POINT_NAME = re.compile(r"layer_(?P<layer>\d+)\.position_(?P<position>\d+)\.(?P<record>.+)")
WORDS_NAME = re.compile(r"frequencies\.theta_\d+\.words")
COS_SIN_NAME = re.compile(r"frequencies\.theta_\d+\.cos_sin_.+")
POSITIONS_NAME = re.compile(r"frequencies\.theta_\d+\.positions")
# ---- what a capture point holds (stepwise_capture.py's module text) ----
KIND_RECORDS = {"window": "attn.window_row", "pool": "compressor.pool_entry", "keys": "indexer.key",
                "queries": "indexer.query"}       # state_precision.py's kinds: the record of each, after the rounding
POINT_RECORDS = {   # what a capture point holds of its layer at its step, by record; the row axes are left out
    "attn_site.carried_pre": "collapse weights the attention site collapses with: the previous layer's FFN site's",
    "attn_site.streams": "streams entering the attention site (the block's input) [4, 5120]",
    "attn_site.mixes": "the site's hyper mixes, linear(x, hc_fn) times the row's rsqrt (model.py:952-954) [24]",
    "attn_site.pre": "pre of the site (kernel.py:421-460), carried to the FFN site's collapse [4]",
    "attn_site.post": "post of the site [4]", "attn_site.comb": "comb of the site, [j][k]: stream j into k [4, 4]",
    "attn_site.collapsed": "collapse of the streams, bfloat16 (model.py:959)", "attn_site.normed": "attn_norm's output",
    "attn.wq_a": "wq_a's output [1280]", "attn.qr": "q_norm's output [1280]",
    "attn.wq_b": "wq_b's output before RoPE [64, 512]", "attn.q": "the queries the attention reads, after RoPE",
    "attn.sink": "the sinks the attention reads, the layer's attn_sink, float32 [64] (model.py:639, :780)",
    "attn.wkv": "wkv's output [512]", "attn.kv_normed": "kv_norm's output before RoPE, float32 (u16)",
    "attn.window_row_unrounded": "the window row after RoPE, before the u16 rounding",
    "attn.window_row": "the window row as the ring holds it: FP16 values in float32",
    "attn.topk_idxs": "the rows the attention reads: window slots, then pool entries plus the window's length",
    "attn.output_rotated": "the attention's output before the inverse RoPE (model.py:780) [64, 512]",
    "attn.output": "the attention's output after the inverse RoPE (model.py:781), before any rounding",
    "attn.output_rounded": "the output as wo_a reads it where the reference rounds it, E4M3 per 32 values with a "
                           "power-of-two scale, in bfloat16 (46cc8310:787; in no capture of an earlier model.py)",
    "attn.wo_a": "the grouped product with wo_a (model.py:787) [8192]", "attn.wo_b": "wo_b's output [5120]",
    "compressor.wkv": "the compressor's wkv output [512]", "compressor.wgate": "the compressor's wgate output",
    "compressor.kv_state": "the compressor's kv slots after the step [2, 512]",
    "compressor.score_state": "the compressor's score slots after the step [2, 512]",
    "compressor.latent": "the compressor norm's output, float32 (u16): the pool entry before RoPE",
    "compressor.pool_entry_unrounded": "the latent after RoPE at its group's first position, before rounding",
    "compressor.pool_entry": "the pool entry as the pool holds it: FP16 values in float32",
    "indexer.wq_b": "the indexer's wq_b output before RoPE [32, 128]", "indexer.query_unrounded": "after RoPE",
    "indexer.query": "the index queries the scores read: FP16 values in float32",
    "indexer.weights_projection": "weights_proj's output [32]",
    "indexer.weights": "the head weights the scores read: weights_proj's output times the scale (model.py:555)",
    "indexer.latent_bfloat16": "the latent wk reads, rounded to bfloat16 [512]", "indexer.wk": "wk's output [128]",
    "indexer.k_normed": "k_norm's output before RoPE, float32", "indexer.key_unrounded": "the key after RoPE",
    "indexer.key": "the index key as the owner's cache holds it: FP16 values in float32",
    "indexer.scores": "index scores over the visible entries, formed again from the queries, keys and weights",
    "indexer.selection": "the indexer's picks as entries (the window's length taken off), ascending, -1 unused",
    "ffn_site.streams": "streams entering the FFN site (after the attention site's post)",
    "ffn_site.mixes": "the FFN site's hyper mixes", "ffn_site.pre": "pre of the FFN site, carried to the next layer",
    "ffn_site.post": "post of the FFN site", "ffn_site.comb": "comb of the FFN site",
    "ffn_site.collapsed": "collapse with the attention site's pre", "ffn_site.normed": "ffn_norm's output",
    "router.logits": "the gate's linear output, float32 [384]", "router.scores": "sqrt(softplus(logits))",
    "router.biased_scores": "scores plus the bias", "router.expert_ids": "the gate's ids [6]",
    "router.route_weights": "the gate's weights [6]",
    "experts.pair_outputs": "each routed expert's output, bfloat16, in ascending expert order [6, 5120]",
    "experts.row_sum": "the routed outputs added from +0 in that order, float32 (model.py:893-900)",
    "shared_expert.output": "the shared expert's output", "ffn.output": "the expert layer's output (model.py:903-904)",
    "ngram.projection": "the n-gram rows' product with wkv [25600]",
    "snapshot.ring": "the layer's window ring after the step [128, 512]",
    "snapshot.pool": "the owner's pool entries after the step", "snapshot.keys": "the owner's index keys after it",
    "snapshot.kv_state": "the compressor's kv slots", "snapshot.score_state": "the compressor's score slots"}
EVERY_LAYER_RECORDS = {name for name in POINT_RECORDS if name.split(".")[0] in (
    "attn_site", "ffn_site", "router", "experts", "shared_expert", "ffn")} | {
    "attn.wq_a", "attn.qr", "attn.wq_b", "attn.q", "attn.sink", "attn.wkv", "attn.kv_normed", "attn.window_row",
    "attn.topk_idxs", "attn.output_rotated", "attn.output", "attn.wo_a", "attn.wo_b"}
# ---- the floor's JSON (stepwise_capture.py --mode floor writes it) ----
FLOOR_KIND, FLOOR_TWIN, FLOOR_LAYERS, FLOOR_TOKENS_SHA256 = "floor", "twin", "layers", "tokens_sha256"
DIGEST_DTYPE = "<i4"                         # make_text_tokens.py's digest: the ids as little-endian int32
FLOAT32, BFLOAT16 = "float32", "bfloat16"
SIGN_OF = {FLOAT32: 2 ** 31, BFLOAT16: 2 ** 15}
BFLOAT16_SHIFT, BFLOAT16_LOW_BITS = 16, 0xFFFF
NOT_NEAR = 2 ** 40                           # places between two words of which one is not finite
BEST_COUNT = 2                               # the two best logits, whose difference is the margin m
SLAB_ROWS = 64                               # rows of logits read at a time: 66 MB in float64
SLAB_VALUES = 2 ** 24                        # values of a record compared at a time by the repeat gate
NOT_COMPARED = "na"
EXIT_CODE_SUCCESS, EXIT_CODE_FAILURE = 0, 1
JSON_INDENT_SPACES = 1


class CaptureFolder:
    """A capture folder, read-only: its records by name, each as a little-endian numpy map of tensors.bin. A break
    puts a changed copy of a record in `overrides`, which array() gives instead."""

    def __init__(self, folder):
        with open(os.path.join(folder, MANIFEST_FILE_NAME), encoding = "utf8") as handle:
            manifest = json.load(handle)
        if manifest.get("complete") is not True:
            raise ValueError(f"{folder}: the manifest does not say complete")
        self.folder, self.path, self.records, self.overrides = folder, os.path.join(folder, TENSORS_FILE_NAME), {}, {}
        end = 0
        for record in sorted(manifest["tensors"], key = lambda record: record["byte_offset"]):
            dtype = LITTLE_ENDIAN_DTYPES.get(record["dtype"])
            if record["name"] in self.records or dtype is None or record["byte_offset"] != end \
                    or record["byte_count"] != math.prod(record["shape"]) * numpy.dtype(dtype).itemsize:
                raise ValueError(f"{folder}: record {record['name']} is given twice, of an unknown dtype, after a gap "
                                 "or of a length that is not its shape's")
            self.records[record["name"]], end = record, end + record["byte_count"]
        if end != os.path.getsize(self.path):
            raise ValueError(f"{folder}: {TENSORS_FILE_NAME} holds {os.path.getsize(self.path)} bytes, the records "
                             f"{end}")
        self.conditions, self.results = manifest.get("conditions") or {}, manifest.get("results") or {}

    def has(self, name):
        return name in self.records

    def array(self, name):
        if name in self.overrides:
            return self.overrides[name]
        record = self.records.get(name)
        if record is None:
            raise ValueError(f"{self.folder}: no record {name}")
        dtype, shape = numpy.dtype(LITTLE_ENDIAN_DTYPES[record["dtype"]]), tuple(record["shape"])
        if not record["byte_count"]:
            return numpy.zeros(shape, dtype)
        return numpy.memmap(self.path, dtype = dtype, mode = "r", offset = record["byte_offset"], shape = shape)

    def token_ids(self):
        ids = numpy.asarray(self.array(TOKEN_IDS)).reshape(-1)
        if not numpy.issubdtype(ids.dtype, numpy.integer):
            raise ValueError(f"{self.folder}: {TOKEN_IDS} is not of integers")
        return ids.astype(numpy.int64)


def tokens_sha256(token_ids):
    return hashlib.sha256(numpy.asarray(token_ids, dtype = DIGEST_DTYPE).tobytes()).hexdigest()


def floor_constants(epsilon_rule = False):
    """The gates' figures a floor is made with and must carry (the module text); with epsilon_rule, those of a floor
    made before FLIP_SIGMAS replaced the epsilon rule."""
    constants = {name: globals()[name] for name in FLOOR_CONSTANTS}
    if epsilon_rule:
        del constants["FLIP_SIGMAS"]
        constants.update(EPSILON_RULE_CONSTANTS)
    return constants


def long_context_constants():
    """What a long-context floor is made with and must carry (the module text)."""
    return {name: list(value) if isinstance(value, tuple) else value
            for name, value in ((name, globals()[name]) for name in LONG_CONTEXT_CONSTANTS)}


def flips_bound(floor_flips):
    """The most flips a layer may have where the floor has floor_flips: E - F <= z sqrt(E + F) solved for E."""
    return floor_flips + (FLIP_SIGMAS ** 2 + FLIP_SIGMAS * math.sqrt(FLIP_SIGMAS ** 2 + 8 * floor_flips)) / 2


def shape_text(shape):
    return "x".join(str(size) for size in shape) or "scalar"


def finite(values, what):
    values = numpy.asarray(values, dtype = numpy.float64)
    if not numpy.isfinite(values).all():
        raise ValueError(f"{what}: a value is not finite")
    return values


def places(reference, other, kind, what):
    """Per value, how many places of `kind` (float32, or bfloat16 held in float32 words) lie between the two; a
    value that is not finite on either side, with other bits than the other side's, lies NOT_NEAR places away."""
    def ordered(values):
        bits = numpy.ascontiguousarray(values, dtype = "<f4").view("<u4").astype(numpy.int64)
        if kind == BFLOAT16:
            if (bits & BFLOAT16_LOW_BITS).any():
                raise ValueError(f"{what}: a bfloat16 record holds float32 words with low bits")
            bits >>= BFLOAT16_SHIFT
        return numpy.where(bits >= SIGN_OF[kind], SIGN_OF[kind] - bits, bits)      # sign and magnitude to a line
    first, second = ordered(reference), ordered(other)
    apart = numpy.abs(first - second)
    not_finite = ~(numpy.isfinite(numpy.asarray(reference, numpy.float32)) & numpy.isfinite(numpy.asarray(other,
                                                                                                        numpy.float32)))
    return numpy.where(not_finite & (first != second), NOT_NEAR, apart)


def share(held):
    """The share of True values; 1 where there is none."""
    return float(held.mean()) if held.size else 1.0


def same_rows(reference, other):
    """Per row (the first axis), whether every byte is the same: of two arrays of one dtype and shape."""
    first, second = (numpy.ascontiguousarray(values).view(numpy.uint8).reshape(len(values), -1)
                     for values in (reference, other))
    return (first == second).all(axis = 1)


def relative_rms_rows(reference, other):
    """Per row (the first axis), the rms of the difference over the rms of the reference's values."""
    reference = finite(reference, "reference").reshape(len(reference), -1)
    other = finite(other, "other").reshape(reference.shape)
    difference = numpy.sqrt(((other - reference) ** 2).sum(axis = 1))
    norm = numpy.sqrt((reference ** 2).sum(axis = 1))
    zero_norm = numpy.where(difference > 0, numpy.inf, 0.0)
    return numpy.divide(difference, norm, out = zero_norm, where = norm > 0)


def least_picks(ids, biased, what):
    """Per row, the slot of its pick of least biased score, the last of equals (the sixth of picks in descending
    order); refused where an expert not picked scores above it, so that the picks are not the row's best (ties taken
    either way)."""
    picked, others = numpy.take_along_axis(biased, ids, axis = 1), biased.copy()
    numpy.put_along_axis(others, ids, -numpy.inf, axis = 1)
    least = ids.shape[1] - 1 - picked[:, ::-1].argmin(axis = 1)
    if (picked[numpy.arange(len(ids)), least] < others.max(axis = 1)).any():
        raise ValueError(f"{what}'s routed ids are not the best of its biased scores")
    return least


def layer_figures(reference_streams, other_streams, reference_ids, other_ids, reference_biased, other_biased = None):
    """One layer of a teacher-forced run against the stepwise reference: the figures of 3.3 (the module text). Each
    flip: [position, the other's pick that the reference did not make, the reference's sixth pick (of least biased
    score), the margin of their biased scores in the reference]."""
    reference_ids, other_ids = numpy.asarray(reference_ids, numpy.int64), numpy.asarray(other_ids, numpy.int64)
    reference_biased = finite(reference_biased, "biased scores")
    other_biased = None if other_biased is None else finite(other_biased, "biased scores")
    if reference_ids.shape != other_ids.shape or reference_ids.ndim != 2 or not reference_ids.shape[1] \
            or reference_biased.ndim != 2 or reference_biased.shape[0] != len(reference_ids) \
            or (other_biased is not None and other_biased.shape != reference_biased.shape):
        raise ValueError(f"routed ids of shapes {reference_ids.shape} and {other_ids.shape} for scores of shapes "
                         f"{reference_biased.shape} and {None if other_biased is None else other_biased.shape}")
    sorted_reference, sorted_other = numpy.sort(reference_ids, axis = 1), numpy.sort(other_ids, axis = 1)
    for ids in (sorted_reference, sorted_other):
        if ids.size and (ids[:, 0].min() < 0 or ids[:, -1].max() >= reference_biased.shape[1]
                         or (numpy.diff(ids, axis = 1) == 0).any()):
            raise ValueError(f"a routed id outside the {reference_biased.shape[1]} experts, or twice in a row")
    rows = numpy.arange(len(reference_ids))
    sixths = reference_ids[rows, least_picks(reference_ids, reference_biased, "the reference")]
    if other_biased is not None:
        least_picks(other_ids, other_biased, "the other run")
    same = (sorted_reference == sorted_other).all(axis = 1)
    relative = relative_rms_rows(reference_streams, other_streams)
    flips = [[int(row), int(expert), int(sixths[row]), float(reference_biased[row, sixths[row]]
                                                             - reference_biased[row, expert])]
             for row in rows[~same] for expert in other_ids[row] if expert not in reference_ids[row]]
    figures = {"rows": len(rows), "same_id_rows": int(same.sum()), "differing_pairs": len(flips),
               "differing_pair_share": len(flips) / reference_ids.size,
               "mean_relative_rms_same_ids": float(relative[same].mean()) if same.any() else 0.0,
               "largest_row_relative_rms_same_ids": float(relative[same].max()) if same.any() else 0.0,
               "largest_margin": max((flip[-1] for flip in flips), default = 0.0), "flips": flips,
               "relative_rms": relative.tolist()}
    if other_biased is not None:
        difference = numpy.abs(other_biased - reference_biased)
        figures.update(router_difference_p999 = float(numpy.percentile(difference, ROUTER_PERCENTILE)),
                       router_difference_largest = float(difference.max()))
    return figures


def best_two(rows):
    """Per row of logits, the ids of the two best (the lowest id first among equals) and their values."""
    ids = numpy.empty((len(rows), BEST_COUNT), numpy.int64)
    for start in range(0, len(rows), SLAB_ROWS):
        slab = finite(numpy.array(rows[start:start + SLAB_ROWS], dtype = numpy.float64), "logits")     # a copy
        ids[start:start + len(slab), 0] = slab.argmax(axis = 1)
        slab[numpy.arange(len(slab)), ids[start:start + len(slab), 0]] = -numpy.inf
        ids[start:start + len(slab), 1] = slab.argmax(axis = 1)
    return ids, values_at(rows, ids)


def values_at(rows, ids):
    """Per row, the logits of the given ids, in float64."""
    values = numpy.empty(ids.shape, numpy.float64)
    for start in range(0, len(ids), SLAB_ROWS):
        part = slice(start, min(start + SLAB_ROWS, len(ids)))
        values[part] = numpy.take_along_axis(numpy.asarray(rows[part], numpy.float64), ids[part], axis = 1)
    return finite(values, "logits")


def margin_changes(reference_ids, reference_values, other_values):
    """Per row, the change dm of the margin between the reference's two best tokens when the other's logits of the
    same two tokens are read, and the reference's margin m."""
    margin = reference_values[:, 0] - reference_values[:, 1]
    return (other_values[:, 0] - other_values[:, 1]) - margin, margin


def best_two_of(capture, positions):
    """The reference's two best ids and values for its first `positions` rows: from its top lists (written in
    descending order, the lowest id first among equals) or from its logits of every row."""
    if capture.has(TOP_IDS):
        ids = numpy.asarray(capture.array(TOP_IDS)[:positions, :BEST_COUNT], numpy.int64)
        return ids, finite(capture.array(TOP_VALUES)[:positions, :BEST_COUNT], "top logits")
    return best_two(capture.array(ALL_LOGITS)[0, :positions])


def logits_at(capture, ids):
    """The capture's logits of the given ids per row: from its logits of every row, else from its top lists, which
    must then hold them."""
    if capture.has(ALL_LOGITS):
        return values_at(capture.array(ALL_LOGITS)[0, :len(ids)], ids)
    top_ids, top_values = capture.array(TOP_IDS)[:len(ids)], capture.array(TOP_VALUES)[:len(ids)]
    found = top_ids[:, :, None] == ids[:, None, :]
    if not found.any(axis = 1).all():
        raise ValueError(f"{capture.folder}: its top lists lack a token the reference's two best name")
    return finite((numpy.asarray(top_values, numpy.float64)[:, :, None] * found).sum(axis = 1), "top logits")


def logit_figures(reference, other, token_ids):
    """KL(reference || other) over the rows with a next token, with its chunk standard error; the same top token;
    per row the change dm of the reference's margin m. The floor's figures of the twin and the prompt path
    (stepwise_capture.py), and the engine's in the teacher_forced gate: one computation for both."""
    scores = compare.score_positions(reference, other, token_ids)
    kl = unit_13b.paired(scores["kl_baseline_other"])
    ids, values = best_two(reference)
    changes, margins = margin_changes(ids, values, values_at(other, ids))
    p99, p999 = (float(numpy.percentile(numpy.abs(changes), percentile))
                 for percentile in (MARGIN_PERCENTILE, DELTA_PERCENTILE))
    return {"positions_scored": len(scores["kl_baseline_other"]), "mean_kl": kl["mean"],
            "mean_kl_standard_error": kl["standard_error"], "chunks": kl["chunks"], "kl_resolvable": kl["resolvable"],
            "same_top": float(scores["same_top_token"].mean()), "margin_change_p99": p99, "margin_change_p999": p999,
            "delta": DELTA_FACTOR * p999, "per_position": {
                "kl": scores["kl_baseline_other"].tolist(), "same_top": scores["same_top_token"].tolist(),
                "margin": margins.tolist(), "margin_change": changes.tolist()}}


class Verdicts:
    """PASS, FAIL or INFO per line; with a break, the line of what was to fail says reported."""

    def __init__(self, targeted):
        self.targeted, self.failed, self.reported = targeted, [], []

    def __call__(self, held, key):
        if held:
            return "PASS"
        (self.reported if key in self.targeted else self.failed).append(key)
        return "reported" if key in self.targeted else "FAIL"


def judge_record(name, reference, other, verdicts, layer = None, position = None):
    """One record both captures hold: its contract where 3.2 gives one in place (a record of another shape or
    dtype then fails), else INFO with its figures."""
    where = ("" if layer is None else f" layer={layer}") + ("" if position is None else f" position={position}")
    record = name if position is None else POINT_NAME.fullmatch(name)["record"]
    reference, other = numpy.asarray(reference), numpy.asarray(other)
    bit_equal = name == EMBEDDINGS or bool(WORDS_NAME.fullmatch(name))
    within_places = bool(COS_SIN_NAME.fullmatch(name)) or record == NORMED_RECORD
    dtypes = f"dtypes={reference.dtype},{other.dtype}"
    if reference.shape != other.shape or ((bit_equal or within_places) and reference.dtype != other.dtype):
        print(f"RECORD name={record}{where} shapes={shape_text(reference.shape)},{shape_text(other.shape)} {dtypes} "
              f"verdict={verdicts(False, name) if bit_equal or within_places else 'INFO'}")
    elif bit_equal:
        rows = (reference[0], other[0]) if name == EMBEDDINGS else (reference, other)      # per position, per word
        differing = int((~same_rows(*rows)).sum())
        print(f"RECORD name={record}{where} rows={len(rows[0])} differing_rows={differing} bound=0 "
              f"verdict={verdicts(differing == 0, name)}")
    elif within_places:
        kind, bound, least = (FLOAT32, COS_SIN_PLACES, 1.0) if COS_SIN_NAME.fullmatch(name) \
            else (BFLOAT16, IN_PLACE_PLACES, IN_PLACE_SHARE)
        apart = places(reference, other, kind, name)
        within = share(apart <= bound)
        print(f"RECORD name={record}{where} values={apart.size} largest_places={int(apart.max()) if apart.size else 0} "
              f"kind={kind} within={within:.6f} bound_places={bound} bound_share={least} "
              f"verdict={verdicts(within >= least, name)}")
    elif numpy.issubdtype(reference.dtype, numpy.floating):                # INFO: over the values finite on both
        first, second = (numpy.asarray(values, numpy.float64).reshape(-1) for values in (reference, other))
        both = numpy.isfinite(first) & numpy.isfinite(second)
        alike = (first == second) | (numpy.isnan(first) & numpy.isnan(second))     # as values: the dtypes may differ
        relative = relative_rms_rows(first[both].reshape(1, -1), second[both].reshape(1, -1))[0] if both.any() else 0.0
        largest = float(numpy.abs(second[both] - first[both]).max()) if both.any() else 0.0
        same_bits = f"{share(same_rows(reference.reshape(-1, 1), other.reshape(-1, 1))):.6f}" \
            if reference.dtype == other.dtype else NOT_COMPARED
        print(f"RECORD name={record}{where} values={first.size} relative_rms={relative:.3e} "
              f"largest_absolute={largest:.3e} not_finite_mismatched={int((~both & ~alike).sum())} "
              f"same_bits={same_bits} {dtypes} verdict=INFO")
    else:
        print(f"RECORD name={record}{where} values={reference.size} same={share(reference == other):.6f} {dtypes} "
              "verdict=INFO")


def layer_gate(oracle, engine, floor, verdicts, record):
    """3.3 per layer, then every other record both hold (the module text)."""
    layers = sorted(int(name.split(".")[0].removeprefix("layer_")) for name in oracle.records
                    if name.endswith(".block.output"))
    pooled = {"layers": 0, "flips": 0, "floor_flips": 0}                 # the layers that held on their own
    for layer in layers:
        name, ids_name, biased_name = (f"layer_{layer}.{part}" for part in (
            "block.output", "mlp.routed_expert_ids", "mlp.biased_scores"))
        missing = next((each for each in (name, ids_name, biased_name) if not engine.has(each)), None)
        if missing is not None:
            print(f"GATE gate={LAYER} layer={layer} missing={missing} verdict={verdicts(False, name)}")
            continue
        try:
            figures = layer_figures(oracle.array(name)[0], engine.array(name)[0], oracle.array(ids_name),
                                    engine.array(ids_name), oracle.array(biased_name), engine.array(biased_name))
        except ValueError as failure:                                      # a value that is not finite, a shape
            print(f"GATE gate={LAYER} layer={layer} error={'_'.join(str(failure).split())} "
                  f"verdict={verdicts(False, name)}")
            continue
        floor_layer = floor[FLOOR_LAYERS][str(layer)]
        floor_flips, floor_largest = floor_layer["differing_pairs"], floor_layer["router_difference_largest"]
        if type(floor_flips) is not int or floor_flips < 0 or floor_layer["rows"] != figures["rows"] \
                or not isinstance(floor_largest, (int, float)) or not 0 <= floor_largest < math.inf:
            raise ValueError(f"the floor's layer {layer} holds no count of flips over {figures['rows']} rows")
        mean_bound = max(MEAN_FACTOR * floor_layer["mean_relative_rms_same_ids"], A_MIN)
        largest_bound = max(LARGEST_FACTOR * floor_layer["largest_row_relative_rms_same_ids"], LARGEST_FACTOR * A_MIN)
        flips_most, margin_bound = flips_bound(floor_flips), LARGEST_FACTOR * floor_largest
        held = figures["mean_relative_rms_same_ids"] <= mean_bound \
            and figures["largest_row_relative_rms_same_ids"] <= largest_bound \
            and figures["differing_pairs"] <= flips_most and figures["largest_margin"] <= margin_bound
        print(f"GATE gate={LAYER} layer={layer} rows={figures['rows']} same_id_rows={figures['same_id_rows']} "
              f"mean_relative_rms={figures['mean_relative_rms_same_ids']:.3e} mean_bound={mean_bound:.3e} "
              f"largest_row={figures['largest_row_relative_rms_same_ids']:.3e} largest_bound={largest_bound:.3e} "
              f"flips={figures['differing_pairs']} floor_flips={floor_flips} flips_bound={flips_most:.1f} "
              f"flip_share={figures['differing_pair_share']:.5f} largest_margin={figures['largest_margin']:.3e} "
              f"margin_bound={margin_bound:.3e} largest_difference={figures['router_difference_largest']:.3e} "
              f"floor_largest_difference={floor_largest:.3e} verdict={verdicts(held, name)}")
        for position, expert, sixth, margin in figures["flips"]:
            over_floor = margin / floor_largest if floor_largest else math.inf
            print(f"FLIP gate={LAYER} layer={layer} position={position} expert={expert} reference_sixth={sixth} "
                  f"margin={margin:.3e} over_floor_largest={over_floor:.2f}")
        record["layers"][layer] = {key: value for key, value in figures.items() if key != "relative_rms"} | {
            "floor_flips": floor_flips, "flips_bound": flips_most, "margin_bound": margin_bound, "held": held}
        if held:
            pooled.update(layers = pooled["layers"] + 1, flips = pooled["flips"] + figures["differing_pairs"],
                          floor_flips = pooled["floor_flips"] + floor_flips)
    pooled["flips_bound"] = flips_bound(pooled["floor_flips"])
    held = pooled["flips"] <= pooled["flips_bound"]
    print(f"POOLED gate={LAYER} layers={pooled['layers']} flips={pooled['flips']} floor_flips={pooled['floor_flips']} "
          f"flips_bound={pooled['flips_bound']:.1f} verdict={verdicts(held, 'pooled')}")
    record["pooled"] = pooled
    for name in sorted(oracle.records):                                    # the in-place records, required
        point = POINT_NAME.fullmatch(name)
        if (name == EMBEDDINGS or (point is not None and point["record"] == NORMED_RECORD)) and not engine.has(name):
            print(f"RECORD name={name} missing=yes verdict={verdicts(False, name)}")
    for name in sorted(set(oracle.records) & set(engine.records)):
        point = POINT_NAME.fullmatch(name)
        if name.endswith((".block.output", ".mlp.routed_expert_ids")) or name in (TOKEN_IDS, GENERATED_IDS) \
                or name.startswith((LOGITS_PREFIX, FREQUENCIES_PREFIX)):
            continue                                       # judged above or by another gate, or the run's input
        judge_record(name, oracle.array(name), engine.array(name), verdicts,
                     None if point is None else int(point["layer"]), None if point is None else int(point["position"]))


def table_gate(oracle, engine, verdicts, record):
    """The rotation step's tables (the module text): every frequencies record of the oracle, in the engine's too."""
    names = sorted(name for name in oracle.records if name.startswith(FREQUENCIES_PREFIX))
    if not any(WORDS_NAME.fullmatch(name) for name in names):
        raise ValueError(f"{oracle.folder}: no frequency words: not a capture of --frequencies")
    for name in (name for name in names if POSITIONS_NAME.fullmatch(name)):  # the input of the sampled tables
        if not engine.has(name) or not numpy.array_equal(numpy.asarray(oracle.array(name), numpy.int64),
                                                         numpy.asarray(engine.array(name), numpy.int64)):
            raise ValueError(f"the two captures are not of the same sampled positions: {name}")
    for name in (name for name in names if not POSITIONS_NAME.fullmatch(name)):
        if engine.has(name):
            judge_record(name, oracle.array(name), engine.array(name), verdicts)
        else:
            print(f"RECORD gate={TABLE} name={name} missing=yes verdict={verdicts(False, name)}")
    record["records"] = names


def logit_rows(capture):
    return capture.array(ALL_LOGITS).shape[1] if capture.has(ALL_LOGITS) else capture.array(TOP_IDS).shape[0]


def tokens_gate(oracle, engine, floor, verdicts, record):
    """Rules 1 and 3 of 3.4 (the module text). Generated token j (from 0) is chosen from the logits of position
    prompt - 1 + j, so a difference at token j leaves the positions before that one to the margin rule."""
    prompt = int(oracle.results["prompt_tokens"])
    reference, other = (numpy.asarray(capture.array(GENERATED_IDS), numpy.int64).reshape(-1)
                        for capture in (oracle, engine))
    if not numpy.array_equal(oracle.token_ids()[:prompt], engine.token_ids()[:prompt]):
        raise ValueError("the two captures are not of the same prompt")
    common = min(len(reference), len(other))
    differing = numpy.flatnonzero(reference[:common] != other[:common])
    first = int(differing[0]) if len(differing) else None
    stop = min(logit_rows(oracle), logit_rows(engine), prompt - 1 + (common if first is None else first))
    ids, values = best_two_of(oracle, stop)
    changes, _ = margin_changes(ids, values, logits_at(engine, ids))
    twin = floor[FLOOR_TWIN]
    engine_p99 = float(numpy.percentile(numpy.abs(changes), MARGIN_PERCENTILE)) if stop else 0.0
    bound = MARGIN_FACTOR * twin["margin_change_p99"]
    print(f"MARGIN gate={TOKENS} positions={stop} engine_margin_change_p99={engine_p99:.4e} bound={bound:.4e} "
          f"twin_margin_change_p99={twin['margin_change_p99']:.4e} verdict={verdicts(engine_p99 <= bound, 'margin')}")
    margin = None
    if first is not None:                                                  # the reference's margin where they part
        _, step_values = best_two_of(oracle, prompt + first)
        margin = float(step_values[-1, 0] - step_values[-1, 1])
    held = len(reference) == len(other) if first is None else margin < twin["delta"]   # parted: lengths differ freely
    print(f"TOKENS gate={TOKENS} generated={len(reference)},{len(other)} "
          f"first_difference_step={'none' if first is None else first + 1} "
          + ("" if first is None else f"position={prompt - 1 + first} reference={reference[first]} "
             f"engine={other[first]} reference_margin={margin:.4e} ")
          + f"delta={twin['delta']:.4e} verdict={verdicts(held, 'token')}")
    record.update(first_difference = first, positions_judged = stop, engine_margin_change_p99 = engine_p99,
                  margin_changes = changes.tolist())


def teacher_forced_gate(oracle, engine, floor, verdicts, record):
    """Rule 2 of 3.4 (the module text)."""
    token_ids = oracle.token_ids()
    if not numpy.array_equal(token_ids, engine.token_ids()):
        raise ValueError("the two captures are not of the same token ids")
    figures = logit_figures(oracle.array(ALL_LOGITS)[0], engine.array(ALL_LOGITS)[0], token_ids)
    held_to_the_twin(TEACHER_FORCED, figures, floor[FLOOR_TWIN], verdicts)
    record.update(figures)


def held_to_the_twin(gate, figures, twin, verdicts):
    """Rule 2's two lines: mean KL at most twice the twin's, the same top token no fewer than its share less 2."""
    kl_bound, same_bound = KL_FACTOR * twin["mean_kl"], twin["same_top"] - SAME_TOP_SHORTFALL
    print(f"FIGURE gate={gate} positions={figures['positions_scored']} mean_kl={figures['mean_kl']:.6f} "
          f"mean_kl_standard_error={figures['mean_kl_standard_error']:.6f} chunks={figures['chunks']} "
          f"kl_bound={kl_bound:.6f} twin_mean_kl={twin['mean_kl']:.6f} "
          f"twin_mean_kl_standard_error={twin['mean_kl_standard_error']:.6f} "
          f"verdict={verdicts(figures['mean_kl'] <= kl_bound, 'kl')}")
    print(f"FIGURE gate={gate} same_top={figures['same_top']:.4f} same_top_bound={same_bound:.4f} "
          f"twin_same_top={twin['same_top']:.4f} verdict={verdicts(figures['same_top'] >= same_bound, 'same_top')}")


def shared_shares(reference, other):
    """Per row of two [rows, picks] arrays of entries (-1: none), the share of the reference's entries the other
    holds too: 1 where the reference holds none. Raises ValueError where a row holds an entry twice."""
    both, single = numpy.sort(numpy.concatenate((reference, other), axis = 1), axis = 1), \
        [numpy.sort(each, axis = 1) for each in (reference, other)]
    if any(((each[:, 1:] == each[:, :-1]) & (each[:, 1:] >= 0)).any() for each in single):
        raise ValueError("a row picks an entry twice")
    shared = ((both[:, 1:] == both[:, :-1]) & (both[:, 1:] >= 0)).sum(axis = 1)
    held = (reference >= 0).sum(axis = 1)
    return numpy.where(held > 0, shared / numpy.maximum(held, 1), 1.0)


def selecting_rows(oracle):
    """The oracle's prompt rows N, flagged rows F and per layer L the first prompt row whose picks an engine holds
    (the module text): 0, or on a direct layer N - F - (n - 1 - L)(w - 1) and at least 0. Raises ValueError where
    the oracle's conditions lack one of these figures."""
    identity, sizes = (oracle.conditions.get(name) for name in ("identity", "model_arguments"))
    identity, sizes = (part if isinstance(part, dict) else {} for part in (identity, sizes))
    prompt, flagged, layers, window, ratios = (identity.get("prompt_tokens"), identity.get("flagged_rows"),
                                               sizes.get("n_layers"), sizes.get("window_size"),
                                               sizes.get("compress_ratios"))
    if not all(type(value) is int and value > 0 for value in (prompt, flagged, layers, window)) \
            or not isinstance(ratios, list) or len(ratios) < layers:
        raise ValueError(f"{oracle.folder}: its conditions lack the prompt's rows, the flagged rows, or the model's "
                         "layers, window and rates")
    return prompt, flagged, {layer: max(prompt - flagged - (layers - 1 - layer) * (window - 1), 0)
                             if ratios[layer] == DIRECT_RATE else 0 for layer in range(layers)}


def noted_rows(capture, name):
    """A list of rows (index.sampled_positions, layer_L.index.positions). Raises ValueError where it is not one of
    integers, each once."""
    rows = numpy.asarray(capture.array(name))
    if rows.ndim != 1 or not numpy.issubdtype(rows.dtype, numpy.integer) or len(numpy.unique(rows)) != len(rows):
        raise ValueError(f"{capture.folder}: {name} is not a list of rows of integers, each once")
    return rows.astype(numpy.int64)


def index_notes(capture, layer, count, rows, ties_optional):
    """An index layer's JUDGED_NOTES at `rows` of its `count` noted rows: picks, then both ties or (ties_optional)
    neither, as zeros. Raises ValueError for notes not one a noted row (picks: a row of entries from -1), ties
    below 0 or more picked at the cut than tied there."""
    present = tuple(record for record in JUDGED_NOTES if capture.has(INDEX_NOTE.format(layer = layer, record = record)))
    if present != JUDGED_NOTES and not (ties_optional and present == JUDGED_NOTES[:1]):
        raise ValueError(f"{capture.folder}: layer {layer} notes {present}: picks, then both ties or neither")
    notes = {record: numpy.zeros(len(rows), numpy.int64) for record in JUDGED_NOTES[1:]}
    for record in present:
        name, picks = INDEX_NOTE.format(layer = layer, record = record), record == JUDGED_NOTES[0]
        values = numpy.asarray(capture.array(name))
        if values.shape[:1] != (count,) or values.ndim != (2 if picks else 1) \
                or not numpy.issubdtype(values.dtype, numpy.integer) or (values < (-1 if picks else 0)).any():
            raise ValueError(f"{capture.folder}: {name} is not one value of integers from {-1 if picks else 0} (picks: "
                             "one row of entries) a noted row")
        notes[record] = values[rows]
    if (notes["picked_at_cut"] > notes["tied_at_cut"]).any():
        raise ValueError(f"{capture.folder}: layer {layer} notes more entries picked at the cut than tied there")
    return notes


def figure_text(figure):
    return " ".join(f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
                    for name, value in figure.items())


def long_context_figures(oracle, engine):
    """G8's figures (the module text), one computation for the gate and the twin's floor: the logit figures over
    the flagged positions, and per index layer the oracle notes the picks shared at the sampled rows both hold and
    the sampled rows the engine lacks where the layer selects (None where the engine notes none). Raises ValueError
    for captures of other tokens or flagged positions, flagged positions that are not the prompt's last F rows and
    the steps with a next token each, what selecting_rows and index_notes refuse."""
    token_ids, (prompt, flagged, first_rows) = oracle.token_ids(), selecting_rows(oracle)
    positions = numpy.asarray(oracle.array(FLAGGED_POSITIONS), numpy.int64)
    if not numpy.array_equal(token_ids, engine.token_ids()) \
            or not numpy.array_equal(positions, numpy.asarray(engine.array(FLAGGED_POSITIONS), numpy.int64)) \
            or not len(positions) or positions[0] != prompt - flagged or positions[-1] + 1 >= len(token_ids) \
            or not numpy.array_equal(positions, numpy.arange(positions[0], positions[0] + len(positions))):
        raise ValueError("the two captures are not of the same tokens and flagged positions, or these are not the "
                         "prompt's last flagged rows and the steps, consecutive, with a next token each")
    figures = logit_figures(oracle.array(FLAGGED_ROWS)[0], engine.array(FLAGGED_ROWS)[0],
                            token_ids[positions[0]:positions[-1] + 2])
    sampled = noted_rows(oracle, SAMPLED_POSITIONS)
    every_layer = noted_rows(engine, SAMPLED_POSITIONS) if engine.has(SAMPLED_POSITIONS) \
        else numpy.empty(0, numpy.int64)
    picks = {}
    for layer in sorted(int(match["layer"]) for match in map(INDEX_PICKS_NAME.fullmatch, oracle.records) if match):
        if not engine.has(INDEX_NOTE.format(layer = layer, record = JUDGED_NOTES[0])):
            picks[layer] = None
            continue
        own = INDEX_NOTE.format(layer = layer, record = LAYER_POSITIONS)
        held = noted_rows(engine, own) if engine.has(own) else every_layer
        _, rows, other_rows = numpy.intersect1d(sampled, held, assume_unique = True, return_indices = True)
        mine, theirs = index_notes(oracle, layer, len(sampled), rows, False), \
            index_notes(engine, layer, len(held), other_rows, True)
        tied, engine_tied = (notes["tied_at_cut"] > notes["picked_at_cut"] for notes in (mine, theirs))
        judged = ~(tied | engine_tied)
        shares = shared_shares(mine["picks"][judged], theirs["picks"][judged])
        picks[layer] = {"rows": int(len(shares)), "missing_rows": len(numpy.setdiff1d(
                            sampled[sampled >= first_rows[layer]], held, assume_unique = True)),
                        "tied_rows": int((~judged).sum()), "engine_tied_rows": int((engine_tied & ~tied).sum()),
                        "mean_shared": float(shares.mean()) if len(shares) else 0.0,
                        "least_shared": float(shares.min()) if len(shares) else 0.0}
    return figures, picks


def long_context_gate(oracle, engine, floor, verdicts, record):
    """G8 (the module text)."""
    identity, twin_identity = oracle.conditions.get("identity"), floor.get("identity")
    if oracle.conditions.get("mode") != LONG_CAPTURE_MODE or not isinstance(identity, dict) \
            or identity.get("twin") is not None:
        raise ValueError("--oracle is no u16 long capture of stepwise_capture.py")
    if floor.get("long_context_constants") != long_context_constants() or not isinstance(twin_identity, dict) \
            or not twin_identity.get("twin") or {**twin_identity, "twin": None} != identity \
            or floor.get("positions") != len(oracle.array(FLAGGED_POSITIONS)):
        raise ValueError("--floor is no long-context floor of this tool's figures, or a twin's of another run than the "
                         f"oracle's (identity, flagged positions): {floor.get('long_context_constants')}")
    figures, picks = long_context_figures(oracle, engine)
    held_to_the_twin(LONG_CONTEXT, figures, floor[FLOOR_TWIN], verdicts)
    for layer, figure in picks.items():
        judged, key = layer in LONG_CONTEXT_LAYERS, f"picks_layer_{layer}"
        held = figure is not None and figure["rows"] > 0 and not figure["missing_rows"] \
            and figure["mean_shared"] >= PICKS_SHARED_LEAST
        print(f"PICKS gate={LONG_CONTEXT} layer={layer} {'missing=yes' if figure is None else figure_text(figure)} "
              f"bound={PICKS_SHARED_LEAST} verdict={verdicts(held, key) if judged else 'INFO'}")
    missing = [layer for layer in LONG_CONTEXT_LAYERS if layer not in picks]
    if missing:
        raise ValueError(f"the oracle notes no picks of layers {missing}")
    record.update(figures, picks = picks)


def same_bytes(first, second):
    """Whether two records hold the same dtype, shape and bytes, read a slab at a time."""
    first, second = numpy.asarray(first).reshape(-1), numpy.asarray(second).reshape(-1)
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    return all(numpy.array_equal(first[start:start + SLAB_VALUES].view(numpy.uint8),
                                 second[start:start + SLAB_VALUES].view(numpy.uint8))
               for start in range(0, len(first), SLAB_VALUES))


def repeat_gate(oracle, engine, verdicts, record):
    """Rules 4 and 5 of 3.4: every record of either capture, bit for bit, in both."""
    for name in sorted(set(oracle.records) | set(engine.records)):
        held = oracle.has(name) and engine.has(name) \
            and oracle.records[name]["shape"] == engine.records[name]["shape"] \
            and same_bytes(oracle.array(name), engine.array(name))
        record.setdefault("differing", []).extend([] if held else [name])
        print(f"RECORD gate={REPEAT} name={name} in_both={'yes' if oracle.has(name) and engine.has(name) else 'no'} "
              f"verdict={verdicts(held, name)}")


def apply_break(break_name, oracle, engine):
    """The ENGINE's data changed as the module text says. Returns the keys of the verdicts that must fail."""
    if break_name == "stream":
        name = f"layer_{BREAK_LAYER}.block.output"
        engine.overrides[name] = numpy.asarray(engine.array(name), numpy.float32) * numpy.float32(BREAK_STREAM_SCALE)
        return {name}
    if break_name == "word":
        name = min((name for name in engine.records if WORDS_NAME.fullmatch(name)), default = None)
        if name is None:
            raise ValueError(f"{engine.folder}: no frequency words to break")
        words = numpy.array(engine.array(name), dtype = "<f4")
        words.view("<u4")[0] += 1
        engine.overrides[name] = words
        return {name}
    if break_name == "token":
        prompt, generated = int(oracle.results["prompt_tokens"]), numpy.array(engine.array(GENERATED_IDS), numpy.int64)
        ids, values = best_two_of(oracle, prompt - 1 + len(generated))
        step = int(numpy.argmax((values[:, 0] - values[:, 1])[prompt - 1:]))
        generated[step] = ids[prompt - 1 + step, 1]
        engine.overrides[GENERATED_IDS] = generated
        return {"token"}
    if break_name in ("logits", "flagged"):
        name = ALL_LOGITS if break_name == "logits" else FLAGGED_ROWS
        rows = numpy.array(engine.array(name), dtype = numpy.float32)
        top = (best_two_of(oracle, rows.shape[1]) if name == ALL_LOGITS else best_two(oracle.array(name)[0]))[0][:, 0]
        rows[0, numpy.arange(rows.shape[1]), top] -= numpy.float32(BREAK_LOGIT_SHIFT)
        engine.overrides[name] = rows
        return {"kl", "same_top"}
    if break_name == "picks":
        name = INDEX_NOTE.format(layer = BREAK_LAYER, record = "picks")
        picks = numpy.array(engine.array(name))
        picks[:, ::2] = -1
        engine.overrides[name] = picks
        return {f"picks_layer_{BREAK_LAYER}"}
    name = min(engine.records, key = lambda name: engine.records[name]["byte_offset"])
    first = numpy.array(engine.array(name))
    first.reshape(-1).view(numpy.uint8)[0] ^= 1
    engine.overrides[name] = first
    return {name}


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev = False)
    parser.add_argument("--gate", required = True, choices = GATES)
    parser.add_argument("--oracle", required = True, type = os.path.abspath, help = "the reference's capture folder")
    parser.add_argument("--engine", required = True, type = os.path.abspath, help = "the engine's capture folder")
    parser.add_argument("--floor", default = None, type = os.path.abspath, help = "stepwise_capture.py's floor JSON")
    parser.add_argument("--output-json", default = None, type = os.path.abspath, help = "receives the figures")
    parser.add_argument("--break", dest = "break_name", choices = tuple(BREAKS), default = None)
    arguments = parser.parse_args()
    if (arguments.floor is None) != (arguments.gate in GATES_WITHOUT_FLOOR) \
            or (arguments.break_name is not None and BREAKS[arguments.break_name] != arguments.gate) \
            or (arguments.output_json is not None and os.path.exists(arguments.output_json)):
        parser.error("--floor goes with every gate but table and repeat; a --break goes with its own gate (the module "
                     "text); --output-json names a file that does not exist")
    sys.stdout.reconfigure(line_buffering = True)
    record, verdicts = {"gate": arguments.gate, "break": arguments.break_name, "layers": {}}, Verdicts(set())
    try:
        oracle, engine = CaptureFolder(arguments.oracle), CaptureFolder(arguments.engine)
        floor = None
        if arguments.floor is not None:
            with open(arguments.floor, "rb") as handle:
                floor_bytes = handle.read()
            floor, record["floor_sha256"] = json.loads(floor_bytes), hashlib.sha256(floor_bytes).hexdigest()
            if not isinstance(floor, dict) or floor.get("kind") != FLOOR_KIND or floor.get("complete") is not True \
                    or (arguments.gate in (LAYER, TEACHER_FORCED, LONG_CONTEXT)
                        and floor.get(FLOOR_TOKENS_SHA256) != tokens_sha256(oracle.token_ids())):
                raise ValueError("--floor is not a complete floor, or not of the oracle capture's tokens")
            if floor.get("constants") not in (floor_constants(), floor_constants(epsilon_rule = True)):
                raise ValueError(f"--floor was made with other figures than this tool's: {floor.get('constants')} "
                                 f"against {floor_constants()}")
            record["floor_of_the_epsilon_rule"] = floor["constants"] != floor_constants()
            print(f"FLOOR file={os.path.basename(arguments.floor)} sha256={record['floor_sha256']} "
                  f"of_the_epsilon_rule={'yes' if record['floor_of_the_epsilon_rule'] else 'no'} "
                  f"delta={floor[FLOOR_TWIN]['delta']:.4e} twin_mean_kl={floor[FLOOR_TWIN]['mean_kl']:.6f}")
        if arguments.gate == LAYER and not numpy.array_equal(oracle.token_ids(), engine.token_ids()):
            raise ValueError("the two captures are not of the same token ids")
        verdicts.targeted = set() if arguments.break_name is None else apply_break(arguments.break_name, oracle, engine)
        if arguments.gate in GATES_WITHOUT_FLOOR:
            {TABLE: table_gate, REPEAT: repeat_gate}[arguments.gate](oracle, engine, verdicts, record)
        else:
            {LAYER: layer_gate, TOKENS: tokens_gate, TEACHER_FORCED: teacher_forced_gate,
             LONG_CONTEXT: long_context_gate}[arguments.gate](oracle, engine, floor, verdicts, record)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as failure:
        print(f"compare_engine_capture FAILED: {type(failure).__name__}: {failure}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    passed = not verdicts.failed and (arguments.break_name is None or bool(verdicts.reported))
    record.update(failed = verdicts.failed, reported = verdicts.reported, passed = passed)
    print(f"VERDICT gate={arguments.gate} break={arguments.break_name or 'none'} failed={len(verdicts.failed)} "
          f"reported={len(verdicts.reported)} verdict={'PASS' if passed else 'FAIL'}")
    if arguments.output_json is not None:
        with open(arguments.output_json, "x", encoding = "utf8") as handle:
            json.dump(record, handle, indent = JSON_INDENT_SPACES)
    return EXIT_CODE_SUCCESS if passed else EXIT_CODE_FAILURE


if __name__ == "__main__":
    sys.exit(main())
