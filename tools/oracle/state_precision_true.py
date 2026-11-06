#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which attention state predicts the TEXT better in DeepSeek's reference, not only which lies nearer the reference
(speed campaign, unit 13b, after unit 13's state_precision.py), and what the reference's shared index-key slot does
to a decode.

The reference rounds four kinds of attention state in place: index keys (model.py:546) and queries (:552) to E2M1
with a power-of-two scale per 32, window rows (:707) to E4M3 with a power-of-two scale per 32, pool entries (:760)
to E2M1 with an E4M3 scale per 16. Through reference_oracle.py's session and state_precision.py's switch, this tool
passes the first --tokens tokens of one sequence from position 0 as the reference is (R, the baseline), then once
per run of --runs, and scores every position from 2,048 on (every index layer selects there) on the TRUE next token.
A run is a form and the kinds it applies to, joined by + ("u16:keys+window"), or "reference" (R again):
  ubf16  the kinds left as computed, in bfloat16: unit 13's runs, the bridge to its figures;
  u16    the kinds unrounded at FP16 precision: rounded to FP16 and held in float32, which holds every FP16 value;
  f8     the kinds rounded afresh, from the same unrounded values, to E4M3 with a power-of-two scale per 32
         (act_quant's in-place rule, kernel.py:81-86). For window rows, which R holds in that grid already, f8 is
         R's rounding of values the norm and RoPE did not first round to bfloat16: it differs from R only where
         that rounding moved a value across a boundary of the E4M3 grid;
  rw     the kinds' scopes (below) held as wide as u16 holds them, every value R's: the control of a u16 or f8 run
         of the same scopes, which differs from it by the values alone.
R is also the profiles' 8-bit trained knob: each E2M1 value is an exact E4M3 code, and the knob keeps the trained
scale beside it (a power of two per 32 for keys and queries, an E4M3 value per 16 for pool entries); window rows are
E4M3 codes already. So 8-bit records hold R's values exactly, and that knob has no run of its own.
How u16 and f8 reach their values: the reference's tensors are bfloat16 (8 significant bits; FP16 has 11), so the
norm of a kind (k_norm, kv_norm, the compressor's: model.py:544, :705, :485 and :462) gets its input in float32,
computes as before and keeps its float32 result; RoPE then runs on float32 (model.py:392-406) and the switch rounds
where R would. The queries leave the indexer's wq_b in float32. The linears before the norms still return
bfloat16 (wk, wkv, wq_b), as the engine's trunk linears write bfloat16 values (linear_records.h), and the pooled
value is still rounded before its norm (:485). So u16 is FP16 storage of values computed from bfloat16 linear
outputs, not an 11-bit state: a norm only rescales its row and keeps its input's bfloat16 error (up to 2^-9 of a
value), and the 64 query values without RoPE are wq_b's bfloat16 outputs exactly. u16 and ubf16 differ by the
bfloat16 rounding at a norm's output and after RoPE; every u16 and f8 VERDICT line says linear_outputs=bfloat16.
What else that changes, within the scopes of a run (keys or queries: the index scope; window or pool: the
attention scope) and nowhere else; the rw run of the same scopes measures all of it:
  * index scope: every indexer's queries and the owners' key caches (model.py:522) are float32, so the score
    of each head (:556) is no longer rounded to bfloat16, nor its sum over heads (:557), and fewer scores tie;
  * attention scope: the window and pool caches (model.py:665, :672) are float32, which the reference's attention
    kernel does not take (kernel.py:329-330); attention then runs in a stand-in on the reference's card
    (wide_sparse_attention below), the same products exact in float32, sums in another order;
  * with the pool unrounded, the latent that feeds wk (model.py:544) is rounded to bfloat16 before it, the value R
    gives wk bit for bit.
The floor (--floor-gguf FILE): after the runs, the reference with the float32 product stand-in of batches f04 and
f05 (gguf_weights.py's control, inputs quantized as the reference does) on the same tokens and positions.

    state_precision_true.py --checkpoint CHECKPOINT --tokens-json FILE --attention-card N [--sequence 0]
            [--tokens 8192] [--runs ...] [--floor-gguf FILE] [--bridge-i13] [--output-json FILE] [--fit-only]
            [--pass-timeout-seconds S] [--break unchanged|narrow_buffers|outside|fp16_range|repeat|bridge]
    state_precision_true.py ... --mode decode [--prefill-tokens 4096] [--decode-steps 2048]
            --decode-budget-seconds S [--output-json FILE] [--pass-timeout-seconds S] [--break slot]

Printed (positions: from 2,048 to the last with a next token; chunks: those positions in order, cut into chunks of
about 512; a difference with its standard error over chunks also gives the 95% quantile of Student's t for chunks
less one degrees of freedom, resolvable= (that quantile times the standard error) and lag1= (the correlation of
neighbouring chunk means, na below 3 chunks): chunks of one text are not independent, so the error is a lower
bound; candidate_ figures: over the positions at which the candidate stage of the index layers after layer 20
prunes, from 16,384 on (model.py:569-575, :598-610), where 2 scored positions or more reach it):
  FIT       the card memory a pass needs (i13's peaks fitted, the index layer of rate 1 at its widest scores), the
            card's, and whether the candidate stage prunes: what does not fit is refused before anything is loaded;
  RUN       per pass: seconds, peak card memory, per kind values left out / asked, changed against R's rounding of
            the same input, held beyond bfloat16, not R's in a kind left to R; FP16 subnormals, values flushed to 0;
  CHECK     switch: state_precision.py's counts, and values changed against R in every kind the run names (--break
            unchanged puts R's values back); kept: in every kind outside the run the values R's definition gives
            for the input rounded to bfloat16 (reference_quantization.py, nothing of the switch's path), also where
            the kind is held wide (--break outside negates one value of each such call); buffers: in a run's
            scopes every buffer and the values of the queries and of a u16 or f8 run's kinds are float32 (--break
            narrow_buffers widens nothing); wide_values: a u16 kind holds values no bfloat16 holds, rounded and
            stored (both breaks above); fp16_range: no u16 value beyond FP16 (--break fp16_range counts the values
            times 2^17); repeat: R again has R's bits (--break repeat gives it another last token); bridge
            (--bridge-i13, i13's 8,192 tokens by digest): every ubf16 run gives i13's figures to its last digit
            (--break bridge holds it to a figure 10 digits off);
  SELECTED  per index layer, the share of R's picks a run picked too (mean, least, rows with all of them);
  FIGURE    mean NLL of the true token under R and under the run, the paired difference with its standard error,
            mean KL(R || run) in nats, the share with R's most likely token, the selected rows shared;
  CONTROL   a u16 or f8 run against the rw run of its scopes: the paired difference and its standard error;
  FLOOR     the floor's paired difference against R;
  VERDICT   per u16 and f8 run: better, worse or not_resolved on the true tokens against R and its control, and
            why not (reason= standard_error, or floor: not past the floor's own difference against R);
  CHANCE    per text and length: the verdicts, those resolved, and those chance alone would resolve at the 5% level
            if chunks were independent.
--mode decode, on the reference as it is: the slot check (prefill up to 8 positions before the first position at
which a layer of rate 2 scores the slot with more entries than it picks, 1,026 here, then 16 decode steps, twice:
as is, and with index_k published before every owner's indexer so that each layer scores its own keys; the two
must have the same bits before that position and differ at each later position where the layer emits nothing;
--break slot publishes nothing. Where model.py publishes every owner's keys itself (from 4c98d615 on: the
REFERENCE_FORMS of reference_oracle.py), the two arms are that reference and must have the same bits at every step,
and --break slot points the index layers that own no keys at the first owner's instead); then a prefill of P + N
tokens, and per arm a prefill of P and N decode steps on the text's own tokens, each step compared with the long
prefill's row: DECODE per arm (NLL, KL, same top token, KL where the rate-2 layers emit nothing and where they emit),
PAIRED (as is minus published, NLL and KL), VERDICT question=slot. CHECK name=decode_keys, in both phases: at every
step each index layer scored the keys of the owner the reference's order gives (as is: the last owner to emit,
model.py:537-548, or as published where model.py publishes; published: the last owner before it).
N falls, and says so, where the first 64 steps show that both arms would not end within the budget; where fewer
than 1,024 steps would, the tool ends with exit code 4, a figure it could not take, not a failed check. A pass (a
prefill) that lasts longer than --pass-timeout-seconds ends the process with the stacks of its threads.
--output-json receives every figure and the per-position scores. Refused: a run named twice or not of a form and
kinds, two rw runs of one scope; --tokens that leave fewer than 2 scored positions; a break with other runs than its
own (unchanged, narrow_buffers, fp16_range: one u16 run; outside: one u16 run of fewer than four kinds; repeat:
"reference" alone; bridge: one ubf16 run of i13 with --bridge-i13), with --floor-gguf, or in the other mode;
--bridge-i13 on other tokens than i13's; an output file that exists; what reference_oracle.py refuses of a tokens
file; a checkpoint whose config.json lacks the sizes the fit reads.
Not shown: Anemone's own arithmetic; a decode of a run other than R; a state from float32 linear outputs; the
candidate stage in a pass of 16,384 tokens or fewer (FIT and VERDICT say candidate_stage=idle); the YaRN range past
65,536; anything past the lengths that fit one card.
Measurement tool: not part of the engine.
"""
import argparse
import collections
import contextlib
import faulthandler
import functools
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
import time

import numpy
import torch

import compare_logit_captures as compare
import reference_oracle as oracle
import reference_quantization
import state_precision as unit_13

BATCH, MEASURE, DECODE = "i13b", "measure", "decode"
BREAK_NAMES = (BREAK_UNCHANGED, BREAK_NARROW_BUFFERS, BREAK_OUTSIDE, BREAK_FP16_RANGE, BREAK_REPEAT, BREAK_BRIDGE,
               BREAK_SLOT) = ("unchanged", "narrow_buffers", "outside", "fp16_range", "repeat", "bridge", "slot")
BREAK_TARGETS = {BREAK_UNCHANGED: ("switch", "wide_values"), BREAK_NARROW_BUFFERS: ("buffers", "wide_values"),
                 BREAK_OUTSIDE: ("kept",), BREAK_FP16_RANGE: ("fp16_range",), BREAK_REPEAT: ("repeat",),
                 BREAK_BRIDGE: ("bridge",), BREAK_SLOT: ("slot", "decode_keys")}         # the checks each must fail
PASS_BREAKS = (BREAK_UNCHANGED, BREAK_NARROW_BUFFERS, BREAK_OUTSIDE, BREAK_FP16_RANGE, BREAK_REPEAT)   # end at the pass
FP16_RANGE_BREAK_SCALE = 2.0 ** 17           # a normed value above 0.5 times this passes FP16's largest, 65,504
BRIDGE_BREAK_DIGITS = 10                     # --break bridge: i13's mean KL moved by 10 of its last digits
KINDS, KEYS, QUERIES, WINDOW, POOL = unit_13.KINDS, unit_13.KEYS, unit_13.QUERIES, unit_13.WINDOW, unit_13.POOL
UBF16, U16, F8, HELD_WIDE = FORMS = ("ubf16", "u16", "f8", "rw")
REFERENCE, BASELINE, FLOOR = unit_13.REFERENCE, "r", "floor"
INDEX_SCOPE, ATTENTION_SCOPE = "index", "attention"
SCOPE_OF = {KEYS: INDEX_SCOPE, QUERIES: INDEX_SCOPE, WINDOW: ATTENTION_SCOPE, POOL: ATTENTION_SCOPE}
EVERY_KIND = "+".join(KINDS)
DEFAULT_RUNS = ",".join([REFERENCE] + [f"{HELD_WIDE}:{kinds}" for kinds in ("keys+queries", "window+pool", EVERY_KIND)]
                        + [f"{form}:{kinds}" for form in (UBF16, U16, F8) for kinds in (*KINDS, EVERY_KIND)])
FIRST_POSITION, DEFAULT_TOKENS = unit_13.FIRST_POSITION, unit_13.DEFAULT_TOKENS
CHUNK_POSITIONS, FEWEST_CHUNKS, LAG_ONE_FEWEST_CHUNKS = 512, 2, 3
STUDENT_T_975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
                 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
                 2.045, 2.042)               # Student's t at 0.975 for 1 to 30 degrees of freedom; beyond, the 30's
CHANCE_LEVEL = 0.05                          # two-sided: what one verdict resolves by chance alone
ALL_POSITIONS, CANDIDATE_POSITIONS = "all", "candidate"
WIDE_DTYPE, NARROW_DTYPE = torch.float32, oracle.REFERENCE_DEFAULT_DTYPE
FP16_SMALLEST_NORMAL = torch.finfo(torch.float16).tiny                    # 2 ** -14
F8_BLOCK_VALUES = 32
BLOCK_SIZE_ARGUMENT, FP4_SCALE_DTYPE_ARGUMENT = 1, 3    # of the partials of state_precision.py's Switch
ATTENTION_ROWS_AT_A_TIME = 256               # 335 MB of selected rows in float32 at 640 rows of 512
KERNEL_LOWEST_MAXIMUM = -1e30                # kernel.py:352-355: the running maximum starts there
I13_TOKENS = 8192                            # batch i13: heldout.txt, the begin token and its first 8,191 tokens
I13_TOKENS_SHA256 = "247e13b333ef09592273e4f72647e4f289530e0306a4b5c6b2a362f63e38fb23"     # i13 tokens.log
I13_FIGURES = {frozenset({KEYS}): (0.016457, 0.9569), frozenset({QUERIES}): (0.016857, 0.9601),   # i13 measure_0.log:
               frozenset({WINDOW}): (0.017143, 0.9560), frozenset({POOL}): (0.017391, 0.9593),   # mean KL, same top
               frozenset(KINDS): (0.019033, 0.9557)}
I13_LAST_DIGITS = (1e-6, 1e-4)               # of each figure as i13 printed it: the bridge allows half of one
HALF = 0.5
DIGEST_DTYPE = "<i4"                         # make_text_tokens.py's digest: the ids as little-endian int32
I13_PASS_PEAKS = ((4096, 16.76), (8192, 24.03))      # GiB allocated at most by one pass of batch i13 (its RUN lines)
SCORE_TENSORS_PER_HEAD = 2                   # model.py:556-557: a head's scores and their weighted copy
SCORE_MASK_BYTES = 1                         # a boolean per (row, entry): the candidates' mask (model.py:575)
NARROW_SCORE_BYTES, WIDE_SCORE_BYTES = 2, 4
FLOOR_MATRIX_GIB = 20.5                      # gguf_weights.py: the trunk's 8-bit matrices kept in float32
FIT_MARGIN_GIB = 4.0                         # the CUDA context, the allocator's slack, the attention stand-in's rows
FIT_SETTINGS = ("index_n_heads", "candidate_source_layer", "candidate_topk_blocks", "candidate_block_size",
                "n_layers", "hc_mult", "dim")             # the long run's range and fit read the last three
EXIT_CODE_DOES_NOT_FIT, EXIT_CODE_BUDGET_TOO_SMALL = 3, 4
DEFAULT_PREFILL_TOKENS, DEFAULT_DECODE_STEPS = 4096, 2048
SLOT_CHECK_STEPS_BEFORE, SLOT_CHECK_STEPS, SLOT_STEPS_SHOWN_AFTER = 8, 16, 2
AS_IS, PUBLISHED = DECODE_ARMS = ("as_is", "published")
PROJECTION_STEPS = 64                        # decode steps of the first arm timed before its length is settled
PASS_TIMEOUT_MOST_SECONDS = 86400            # a day: above any pass, far within what the watchdog's clock takes
FEWEST_DECODE_STEPS = CHUNK_POSITIONS * FEWEST_CHUNKS
NOT_READ = "na"
VALUE_COUNTS = ("changed", "beyond_bfloat16", "not_r", "fp16_subnormal", "fp16_flushed", "fp16_overflow")  # per kind
LINE_SAFE = str.maketrans({character: "_" for character in " =\t\n"})
JSON_INDENT_SPACES = 1

Run = collections.namedtuple("Run", ("label", "form", "kinds", "scopes"))


class DecodeBudgetTooSmall(Exception):
    """The budget leaves fewer decode steps than a figure needs: the decode was not taken, nothing failed."""


def parse_runs(text):
    """The runs of --runs, or None for a text that names one twice, names no form and kinds, or two rw runs that
    hold the same scopes wide (one control each)."""
    runs = []
    for label in text.split(","):
        form, _, kinds_text = label.partition(":")
        kinds = kinds_text.split("+")
        if label == REFERENCE:
            runs.append(Run(label, REFERENCE, frozenset(), frozenset()))
        elif form in FORMS and len(set(kinds)) == len(kinds) and set(kinds) <= set(KINDS):
            scopes = frozenset() if form == UBF16 else frozenset(SCOPE_OF[kind] for kind in kinds)
            runs.append(Run(label, form, frozenset(kinds), scopes))
        else:
            return None
    named, controls = [(run.form, run.kinds) for run in runs], [run.scopes for run in runs if run.form == HELD_WIDE]
    return runs if len(set(named)) == len(named) and len(set(controls)) == len(controls) else None


def break_fits(break_name, runs, bridge_i13):
    """Whether a measure break comes with the runs it is shown on (the module text lists them)."""
    if break_name is None:
        return True
    if len(runs) != 1:
        return False
    run = runs[0]
    return {BREAK_UNCHANGED: run.form == U16, BREAK_NARROW_BUFFERS: run.form == U16, BREAK_FP16_RANGE: run.form == U16,
            BREAK_OUTSIDE: run.form == U16 and run.kinds != frozenset(KINDS), BREAK_REPEAT: run.label == REFERENCE,
            BREAK_BRIDGE: run.form == UBF16 and run.kinds in I13_FIGURES and bridge_i13}.get(break_name, False)


def broken_as_asked(checks, break_name):
    """A break's step ends as asked when every check it aims at reported and every other one held."""
    targets = BREAK_TARGETS[break_name]
    return not any(checks.get(name, True) for name in targets) \
        and all(held for name, held in checks.items() if name not in targets)


def verdict_text(name, held, break_name):
    return "ok" if held else "reported" if name in BREAK_TARGETS.get(break_name, ()) else "FAILED"


def rounded_as_the_reference(values, quantize):
    """R's values for `values`, in a new bfloat16 tensor: the reference's kernels take bfloat16 alone (kernel.py:41,
    :128), and a wider tensor is rounded to it first, as the reference's own tensors round (model.py:405)."""
    narrow = values.to(NARROW_DTYPE, copy = True)
    quantize.func(narrow, *quantize.args[1:], **quantize.keywords)        # the oracle's checked call, on the copy
    return narrow


def as_the_reference(values, quantize):
    """R in place, on a tensor of any width."""
    if values.dtype == NARROW_DTYPE:
        return quantize()                                                  # the reference's own call, as unit 13's
    return values.copy_(rounded_as_the_reference(values, quantize))


def defined_r(kind, narrow, quantize):
    """R from reference_quantization.py's definitions of kernel.py, for the bfloat16 values `narrow` of one call:
    what CHECK name=kept holds a kind outside the run to, sharing nothing with the switch's path to its values."""
    block_size = quantize.args[BLOCK_SIZE_ARGUMENT]
    if kind == WINDOW:
        return reference_quantization.defined_act_quant_in_place(narrow, block_size)
    return reference_quantization.defined_fp4_act_quant_in_place(narrow, block_size,
                                                                 quantize.args[FP4_SCALE_DTYPE_ARGUMENT])


def beyond_bfloat16(values):
    """How many values a bfloat16 could not hold."""
    return int((values != values.to(NARROW_DTYPE).to(values.dtype)).sum())


class FormSwitch(unit_13.Switch):
    """state_precision.py's switch, which skips the rounding of the kinds a run names, with the value of the run's
    form in its place, and R's rounding also for a tensor held wider than bfloat16."""

    def __init__(self, reference_model, break_name):
        super().__init__(reference_model, False)                           # unit 13's break is not this tool's
        self.break_name = break_name

    def begin(self, run):
        super().begin(frozenset() if run.form in (HELD_WIDE, REFERENCE) else run.kinds)   # rw leaves nothing out
        self.form, self.counts = run.form, {name: dict.fromkeys(KINDS, 0) for name in VALUE_COUNTS}
        self.widths = {kind: set() for kind in KINDS}
        self.moves_outside = self.break_name == BREAK_OUTSIDE and run.form == U16      # never in R's own pass

    def in_place(self, kind, values, quantize):
        outside = kind not in self.kinds
        before = values.to(NARROW_DTYPE, copy = True) if outside else None
        held = super().in_place(kind, values, functools.partial(as_the_reference, values, quantize))
        self.widths[kind].add(values.dtype)
        counts = self.counts
        if outside:
            if self.moves_outside:                                         # --break outside: one value negated
                flat, largest = values.view(-1), int(values.abs().argmax())
                flat[largest] = -flat[largest]
            counts["not_r"][kind] += int((values != defined_r(kind, before, quantize).to(values.dtype)).sum())
            return held
        reference = rounded_as_the_reference(values, quantize)
        if self.break_name == BREAK_UNCHANGED:                             # R's values back
            values.copy_(reference)
        elif self.form == U16:
            rounded = values.to(torch.float16)
            ranged = (values * FP16_RANGE_BREAK_SCALE).to(torch.float16) \
                if self.break_name == BREAK_FP16_RANGE else rounded
            counts["fp16_overflow"][kind] += int((ranged.isinf() & values.isfinite()).sum())
            counts["fp16_subnormal"][kind] += int(((rounded != 0) & (rounded.abs() < FP16_SMALLEST_NORMAL)).sum())
            counts["fp16_flushed"][kind] += int(((rounded == 0) & (values != 0)).sum())
            values.copy_(rounded)
        elif self.form == F8:
            values.copy_(reference_quantization.defined_act_quant_in_place(values, F8_BLOCK_VALUES))
        counts["changed"][kind] += int((values != reference.to(values.dtype)).sum())
        counts["beyond_bfloat16"][kind] += beyond_bfloat16(values)
        return values


def wide_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Stands for kernel.sparse_attn (kernel.py:311-403) in a run whose attention scope is wide: per row and head,
    the scores of the selected rows in float32 (a bfloat16 query times a value of 11 significant bits is exact
    there), scaled; one softmax over them with the row's largest score, the sink added to its sum (:383); the
    weights rounded to bfloat16, as the kernel rounds them before its second product (:377); the weighted sum in
    float32, divided, returned in bfloat16. The kernel takes a running maximum per 64 rows instead, and sums in
    another order: the rw run prices both."""
    sequences, rows, heads, width = q.shape
    if sequences != oracle.SEQUENCES or kv.shape[0] != sequences or kv.shape[-1] != width \
            or topk_idxs.shape[:2] != (sequences, rows) or attn_sink.shape != (heads,):
        raise RuntimeError(f"attention of shapes {tuple(q.shape)}, {tuple(kv.shape)}, {tuple(attn_sink.shape)}, "
                           f"{tuple(topk_idxs.shape)}: not what model.py:780 passes")
    if topk_idxs.numel() and not -1 <= int(topk_idxs.min()) <= int(topk_idxs.max()) < kv.shape[1]:
        raise RuntimeError(f"a selected row outside the {kv.shape[1]} rows of the attention's cache")
    output, sink = torch.empty_like(q), attn_sink.float()
    for first in range(0, rows, ATTENTION_ROWS_AT_A_TIME):
        part = slice(first, min(first + ATTENTION_ROWS_AT_A_TIME, rows))
        picks = topk_idxs[0, part].long()
        selected = kv[0][picks.clamp_min(0)].float()                       # [rows, picks, width]
        scores = torch.einsum("rhd,rkd->rhk", q[0, part].float(), selected) * softmax_scale
        scores = scores.masked_fill((picks < 0).unsqueeze(1), -math.inf)
        largest = scores.amax(dim = -1, keepdim = True).clamp_min(KERNEL_LOWEST_MAXIMUM)
        weights = torch.exp(scores - largest)
        total = weights.sum(dim = -1) + torch.exp(sink - largest.squeeze(-1))
        summed = torch.einsum("rhk,rkd->rhd", weights.to(NARROW_DTYPE).float(), selected)
        output[0, part] = (summed / total.unsqueeze(-1)).to(q.dtype)
    return output


def layers_of(session):
    """The attentions, the indexers by layer, the indexers that own keys and the attentions that pool."""
    attentions = [block.attn for block in session.model.layers]
    indexers = {attention.layer_id: attention.indexer for attention in attentions if attention.is_index_source}
    owners = {layer: indexer for layer, indexer in indexers.items() if indexer.owns_k}
    return attentions, indexers, owners, [attention for attention in attentions if attention.is_kv_source]


@contextlib.contextmanager
def conditions(session, run, narrow_buffers):
    """What a run holds wide while it passes, put back after it (the module text says what and why). With
    --break narrow_buffers nothing is widened: its forms round into the reference's bfloat16 tensors."""
    reference_model = session.reference.model
    attentions, indexers, owners, sources = layers_of(session)
    scopes = frozenset() if narrow_buffers else run.scopes
    widened = run.kinds if run.form in (U16, F8) and not narrow_buffers else frozenset()
    handles, swapped, attention = [], [], reference_model.sparse_attn

    def hold_wide(module, name):                                           # a registered buffer, replaced by name
        swapped.append((module, name, getattr(module, name)))
        setattr(module, name, torch.zeros_like(getattr(module, name), dtype = WIDE_DTYPE))

    def wide_input(module, inputs):
        return (inputs[0].to(WIDE_DTYPE), *inputs[1:])

    def narrow_input(module, inputs):
        return (inputs[0].to(NARROW_DTYPE), *inputs[1:])

    try:
        if INDEX_SCOPE in scopes:
            handles += [indexer.wq_b.register_forward_hook(lambda module, inputs, output: output.to(WIDE_DTYPE))
                        for indexer in indexers.values()]
            for indexer in owners.values():
                hold_wide(indexer, "k_cache")
        if ATTENTION_SCOPE in scopes:
            for each in attentions:
                hold_wide(each, "window_kv_cache")
            for each in sources:
                hold_wide(each, "compress_kv_cache")
            reference_model.sparse_attn = wide_sparse_attention
        if KEYS in widened:
            handles += [indexer.k_norm.register_forward_pre_hook(wide_input) for indexer in owners.values()]
        if WINDOW in widened:
            handles += [each.kv_norm.register_forward_pre_hook(wide_input) for each in attentions]
        if POOL in widened:
            handles += [each.compressor.norm.register_forward_pre_hook(wide_input) for each in sources]
            handles += [indexer.wk.register_forward_pre_hook(narrow_input) for indexer in owners.values()]
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, name, buffer in swapped:
            setattr(module, name, buffer)
        reference_model.sparse_attn = attention


@contextlib.contextmanager
def deadline(seconds):
    """A pass that lasts past `seconds` (None: no limit) ends the process with exit code 1 and the stacks of its
    threads, wherever it waits: the standard library's watchdog thread needs no answer from the main thread."""
    if seconds is not None:
        faulthandler.dump_traceback_later(seconds, exit = True)
    try:
        yield
    finally:
        if seconds is not None:
            faulthandler.cancel_dump_traceback_later()


def scope_buffers(session):
    """Per kind, the reference's buffers that hold it."""
    attentions, _, owners, sources = layers_of(session)
    return {KEYS: [indexer.k_cache for indexer in owners.values()], QUERIES: [],
            WINDOW: [each.window_kv_cache for each in attentions], POOL: [each.compress_kv_cache for each in sources]}


def buffers_held(buffers, run, switch):
    """CHECK name=buffers, while the run's buffers are in place: see the module text."""
    held = True
    for kind in (kind for kind in KINDS if SCOPE_OF[kind] in run.scopes):
        held &= all(buffer.dtype == WIDE_DTYPE for buffer in buffers[kind])
        if kind == QUERIES or (run.form in (U16, F8) and kind in run.kinds):
            held &= switch.widths[kind] == {WIDE_DTYPE}
    return held


def wide_values_held(buffers, run, switch):
    """CHECK name=wide_values of a u16 run, while its buffers are in place: see the module text."""
    return all(switch.counts["beyond_bfloat16"][kind] > 0
               and (kind == QUERIES or sum(map(beyond_bfloat16, buffers[kind])) > 0) for kind in run.kinds)


def paired(difference):
    """The mean of a per-position difference and its standard error over chunks of about CHUNK_POSITIONS positions
    in order, the quantile of Student's t a resolved difference must pass in units of it, and the correlation of
    neighbouring chunk means: positions of one chunk depend on each other, chunks less, but not nothing."""
    means = numpy.array([chunk.mean() for chunk in numpy.array_split(
        difference, max(FEWEST_CHUNKS, round(len(difference) / CHUNK_POSITIONS)))])
    centered = means - means.mean()
    spread = float((centered * centered).sum())
    standard_error = float(means.std(ddof = 1) / math.sqrt(len(means)))
    quantile = STUDENT_T_975[min(len(means) - 1, len(STUDENT_T_975)) - 1]
    lag_one = float((centered[1:] * centered[:-1]).sum() / spread) \
        if len(means) >= LAG_ONE_FEWEST_CHUNKS and spread > 0 else None
    return {"mean": float(difference.mean()), "standard_error": standard_error, "chunks": len(means),
            "quantile": quantile, "resolvable": quantile * standard_error, "lag1_correlation": lag_one,
            "chunk_means": means.tolist()}


def verdict_of(figure, floor = None):
    """Lower is better (a negative log-likelihood, a distance). Returns the verdict and why: a difference is
    resolved past its resolvable difference and, where a floor was measured under the same conditions, past the
    floor's own difference against R."""
    if not abs(figure["mean"]) > figure["resolvable"]:
        return "not_resolved", "standard_error"
    if floor is not None and not abs(figure["mean"]) > abs(floor["mean"]):
        return "not_resolved", "floor"
    return ("better" if figure["mean"] < 0 else "worse"), "resolved"


def text_of(figure, name):
    lag_one = figure["lag1_correlation"]
    return (f"{name}={figure['mean']:+.6f} {name}_standard_error={figure['standard_error']:.6f} "
            f"{name}_chunks={figure['chunks']} {name}_resolvable={figure['resolvable']:.6f} "
            f"{name}_lag1={NOT_READ if lag_one is None else f'{lag_one:+.2f}'}")


def first_candidate_position(settings):
    """The first position at which the candidate stage prunes (model.py:569-575, :598-610): a row of the source's
    rate reaches more entries than candidate_topk_blocks blocks of candidate_block_size hold. None where the model
    has no such stage."""
    source = settings["candidate_source_layer"]
    if source < 0:
        return None
    entries = settings["candidate_topk_blocks"] * settings["candidate_block_size"]
    return (entries + 1) * settings["compress_ratios"][source] - 1        # (position + 1) // rate passes entries


def candidate_offset_of(first_candidate, tokens):
    """Where the scored positions the candidate stage prunes at begin, as an index into the per-position scores of
    a pass of `tokens` tokens; None where fewer than FEWEST_CHUNKS scored positions reach the stage."""
    if first_candidate is None:
        return None
    first = max(first_candidate, FIRST_POSITION)
    return first - FIRST_POSITION if first + FEWEST_CHUNKS < tokens else None


def candidate_stage_text(first_candidate, tokens):
    if candidate_offset_of(first_candidate, tokens) is None:
        return "candidate_stage=idle"
    return f"candidate_stage=active candidate_from={first_candidate}"


@torch.inference_mode()                                                    # as the reference's own forward
def measure(arguments, session, token_ids, text, record):
    """The runs, the floor, the checks and the figures. Returns whether the tool ended as its command line asks."""
    model, tokens, break_name = session.model, len(token_ids), arguments.break_name
    _, indexers, _, _ = layers_of(session)
    picks, head = {}, {}
    for layer, indexer in indexers.items():                # hooks on the reference's own modules, left in place
        indexer.register_forward_hook(lambda module, inputs, output, layer = layer:
                                      picks.__setitem__(layer, output[0][FIRST_POSITION:]))
    model.head.register_forward_pre_hook(lambda _, inputs: head.update(rows = inputs[0][0]))
    switch = FormSwitch(session.reference.model, break_name)
    expected = unit_13.expected_values(session, tokens)
    input_ids = torch.tensor([token_ids], dtype = torch.long, device = oracle.CARD_DEVICE)
    repeated_ids = input_ids.clone()                                       # what the run "reference" reads
    if break_name == BREAK_REPEAT:                                         # another last token
        repeated_ids[0, -1] = (repeated_ids[0, -1] + 1) % session.model_arguments.vocab_size
    next_tokens, label = numpy.asarray(token_ids[FIRST_POSITION:]), f"batch={BATCH} text={text} tokens={tokens}"
    narrow_buffers, results, failed = break_name == BREAK_NARROW_BUFFERS, record["runs"], False
    stage = candidate_stage_text(arguments.first_candidate_position, tokens)
    candidate_offset = candidate_offset_of(arguments.first_candidate_position, tokens)

    def one_pass(run):
        """Prints RUN and the checks; returns the logits and whether every check held."""
        with conditions(session, run, narrow_buffers):
            switch.begin(run)
            oracle.forget_earlier_runs(session)
            torch.cuda.reset_peak_memory_stats()
            started = oracle.synchronized_clock()
            with deadline(arguments.pass_timeout_seconds):
                model.forward(repeated_ids if run.label == REFERENCE else input_ids, 0)   # generate.py:69-74
            seconds = oracle.synchronized_clock() - started
            counts, card_gib = switch.counts, torch.cuda.max_memory_allocated() / oracle.BYTES_PER_GIB
            checks = {"switch": unit_13.switch_held(switch, expected)
                      and all(counts["changed"][kind] > 0 for kind in switch.kinds),
                      "kept": not any(counts["not_r"].values())}
            buffers = scope_buffers(session)
            if run.scopes:
                checks["buffers"] = buffers_held(buffers, run, switch)
            if run.form == U16:
                checks["wide_values"] = wide_values_held(buffers, run, switch)
        rows = head["rows"]
        if run.form == U16:
            checks["fp16_range"] = not any(counts["fp16_overflow"].values())
        if switch.kinds:                                                   # a run that changes values changes these
            checks["switch"] &= not oracle.same_bits(results[BASELINE]["rows"], rows)
        logits = model.head.forward(rows[None, FIRST_POSITION:], full_logits = True)[0].cpu().numpy()   # no hook
        if run.label == REFERENCE:
            checks["repeat"] = oracle.same_bits(results[BASELINE]["rows"], rows) \
                and numpy.array_equal(baseline_logits.view(numpy.uint32), logits.view(numpy.uint32))
        shown = VALUE_COUNTS if run.form == U16 else VALUE_COUNTS[:VALUE_COUNTS.index("fp16_subnormal")]
        print(f"RUN {label} run={run.label} seconds={seconds:.1f} card_gib={card_gib:.2f} "
              + " ".join(f"{kind}={switch.left[kind]}/{switch.asked[kind]}"
                         + "".join(f",{name}:{counts[name][kind]}" for name in shown) for kind in KINDS))
        for name, held in checks.items():
            print(f"CHECK {label} run={run.label} name={name} verdict={verdict_text(name, held, break_name)}")
        results[run.label] = {"form": run.form, "kinds": sorted(run.kinds), "seconds": seconds, "card_gib": card_gib,
                              "checks": checks, "rows": rows, "picks": dict(picks), "values": {kind: {
                                  "left": switch.left[kind], "asked": switch.asked[kind],
                                  **{name: counts[name][kind] for name in VALUE_COUNTS}} for kind in KINDS}}
        return logits, all(checks.values())

    def differences(nll, other):
        """A paired difference over every scored position, and over those the candidate stage prunes at."""
        difference = nll - other
        return {ALL_POSITIONS: paired(difference),
                CANDIDATE_POSITIONS: None if candidate_offset is None else paired(difference[candidate_offset:])}

    def described(pair, name):
        return text_of(pair[ALL_POSITIONS], name) + ("" if pair[CANDIDATE_POSITIONS] is None else
                                                     " " + text_of(pair[CANDIDATE_POSITIONS], f"candidate_{name}"))

    def figures(run_label, logits):
        """SELECTED and FIGURE of a run against R; its per-position scores into the record."""
        result, baseline = results[run_label], results[BASELINE]
        scores = compare.score_positions(baseline_logits, logits, next_tokens)
        shares = {}
        for layer, indexer in indexers.items():
            share = unit_13.shared_picks(baseline["picks"][layer], result["picks"][layer])
            shares[layer] = {"mean": float(share.mean()), "least": float(share.min()),
                             "same_rows": float((share == 1).double().mean())}
            print(f"SELECTED {label} run={run_label} layer={layer} rate={indexer.compress_ratio} rows={len(share)} "
                  f"picks={result['picks'][layer].size(-1)} mean_overlap={shares[layer]['mean']:.4f} "
                  f"least_overlap={shares[layer]['least']:.4f} same_rows={shares[layer]['same_rows']:.4f}")
        nll, kl, same_top = scores["nll_other"], scores["kl_baseline_other"], scores["same_top_token"]
        layer_means = [share["mean"] for share in shares.values()]
        result.update(nll = nll, mean_nll = float(nll.mean()), nll_minus_r = differences(nll, scores["nll_baseline"]),
                      mean_kl_r_run = float(kl.mean()), same_top = float(same_top.mean()), selected = shares,
                      same_bits = oracle.same_bits(baseline["rows"], result["rows"]),
                      per_position = {"nll": nll.tolist(), "kl_r_run": kl.tolist(), "same_top": same_top.tolist()})
        candidate_text = "candidate_positions=0"
        if candidate_offset is not None:
            result["candidate"] = {"positions": len(nll) - candidate_offset,
                                   "mean_kl_r_run": float(kl[candidate_offset:].mean()),
                                   "same_top": float(same_top[candidate_offset:].mean())}
            candidate_text = (f"candidate_positions={result['candidate']['positions']} "
                              f"candidate_mean_kl={result['candidate']['mean_kl_r_run']:.6f} "
                              f"candidate_same_top={result['candidate']['same_top']:.4f}")
        record.setdefault("nll_r", scores["nll_baseline"].tolist())
        for name in ("rows", "picks"):                     # on the card: R's alone are kept for the runs after
            result.pop(name)
        print(f"FIGURE {label} run={run_label} positions={len(nll)} mean_nll_r={scores['nll_baseline'].mean():.6f} "
              f"mean_nll={result['mean_nll']:.6f} {described(result['nll_minus_r'], 'nll_minus_r')} "
              f"mean_kl={result['mean_kl_r_run']:.6f} same_top={result['same_top']:.4f} "
              f"selected_shared={numpy.mean(layer_means):.4f} least_layer={min(layer_means):.4f} "
              f"same_bits={unit_13.NO_OR_YES[result['same_bits']]} {candidate_text}")

    runs = parse_runs(arguments.runs)
    baseline_logits, held = one_pass(Run(BASELINE, REFERENCE, frozenset(), frozenset()))
    if not held:
        return False
    for run in runs:
        logits, held = one_pass(run)
        if break_name in PASS_BREAKS:                                      # such a pass ends the tool, either way
            return broken_as_asked(results[run.label]["checks"], break_name)
        if not held:
            return False
        figures(run.label, logits)
    if arguments.floor_gguf is not None:                   # last: the stand-in stays for the rest of the process
        torch.cuda.empty_cache()                                           # what the wide passes held, for its matrices
        gguf_weights = importlib.import_module("gguf_weights")             # as reference_oracle.py imports it
        record["floor_install"] = gguf_weights.install(
            session.reference.model, model, session.stages, oracle.HostResidentExpert, oracle.HostResidentTable,
            arguments.floor_gguf, gguf_weights.INPUTS_AS_THE_REFERENCE, ())          # no part: the control
        logits, held = one_pass(Run(FLOOR, REFERENCE, frozenset(), frozenset()))
        if not held:
            return False
        figures(FLOOR, logits)
    nll = {name: result["nll"] for name, result in results.items() if "nll" in result}
    controls = {run.scopes: run.label for run in runs if run.form == HELD_WIDE}
    for run in (run for run in runs if run.form in (U16, F8) and run.scopes in controls):
        control = controls[run.scopes]
        results[run.label]["nll_minus_control"] = differences(nll[run.label], nll[control])
        print(f"CONTROL {label} run={run.label} control={control} "
              + described(results[run.label]["nll_minus_control"], "nll_minus_control"))
    for run in (run for run in runs if run.form == UBF16 and arguments.bridge_i13 and run.kinds in I13_FIGURES):
        theirs, ours = I13_FIGURES[run.kinds], (results[run.label]["mean_kl_r_run"], results[run.label]["same_top"])
        if break_name == BREAK_BRIDGE:                                     # a figure i13 did not print
            theirs = (theirs[0] + BRIDGE_BREAK_DIGITS * I13_LAST_DIGITS[0], theirs[1])
        same = all(abs(mine - figure) <= HALF * digit for mine, figure, digit in zip(ours, theirs, I13_LAST_DIGITS))
        results[run.label]["checks"]["bridge"], failed = same, failed or not same
        print(f"CHECK {label} run={run.label} name=bridge mean_kl={ours[0]:.6f} i13_mean_kl={theirs[0]:.6f} "
              f"same_top={ours[1]:.4f} i13_same_top={theirs[1]:.4f} verdict={verdict_text('bridge', same, break_name)}")
        if break_name == BREAK_BRIDGE:
            return broken_as_asked(results[run.label]["checks"], break_name)
    floor = results.get(FLOOR, {}).get("nll_minus_r")
    if floor is not None:
        print(f"FLOOR {label} {described(floor, 'floor_nll_minus_r')} {stage}")
    judged = []
    for run in (run for run in runs if run.form in (U16, F8)):
        result, parts = results[run.label], []
        result["verdict"] = {}
        for against, name in (("r", "nll_minus_r"), ("control", "nll_minus_control")):
            pair = result.get(name)
            for positions, prefix in ((ALL_POSITIONS, ""), (CANDIDATE_POSITIONS, "candidate_")):
                figure, key = None if pair is None else pair[positions], prefix + (
                    "verdict" if against == "r" else "against_control")
                floor_figure = None if floor is None else floor[positions]
                if figure is None:
                    if positions == ALL_POSITIONS:
                        parts.append(f"{key}={NOT_READ}")
                    continue
                verdict, reason = verdict_of(figure, floor_figure)
                result["verdict"][key] = {"verdict": verdict, "reason": reason}
                judged.append(verdict)
                parts.append(f"{text_of(figure, prefix + name)} {key}={verdict} {key}_reason={reason}"
                             + ("" if floor_figure is None or against != "r"
                                else f" {prefix}floor_nll_minus_r={floor_figure['mean']:+.6f}"))
        print(f"VERDICT {label} question={run.form} kinds={run.label.partition(':')[2]} linear_outputs=bfloat16 "
              f"{stage} floor={unit_13.NO_OR_YES[floor is not None]} " + " ".join(parts))
    resolved = sum(verdict != "not_resolved" for verdict in judged)
    record["chance"] = {"verdicts": len(judged), "resolved": resolved, "level": CHANCE_LEVEL}
    print(f"CHANCE {label} verdicts={len(judged)} resolved={resolved} level={CHANCE_LEVEL} "
          f"by_chance_if_independent={CHANCE_LEVEL * len(judged):.1f}")
    return not failed


def first_slot_position(session, owners):
    """The first decode position where an owner of keys that pools emits none, so that its layer scores the
    slot (model.py:537, :554), and sees more entries than it picks, so that the slot decides its picks (:579)."""
    topk, positions = session.model_arguments.index_topk, []
    for rate in {indexer.compress_ratio for indexer in owners.values() if indexer.compress_ratio > 1}:
        position = rate * (topk + 1) - 1                                   # (position + 1) // rate is topk + 1
        while (position + 1) % rate == 0:                                  # an emitting position: the next one
            position += 1
        positions.append(position)
    if not positions:
        raise RuntimeError("no index layer that owns keys pools more than one token: the slot is never scored")
    return min(positions)


def keys_expected(indexers, owners, prefill, steps, published):
    """Per decode step after a prefill of `prefill` tokens, whose keys each index layer scores in the reference's
    order: an owner writes the slot where it emits (model.py:537-548) and, published (by a hook, or by model.py
    itself from 4c98d615 on), before it scores at every step; a layer scores what the slot holds (:554). After the
    prefill it holds the last owner's that emitted."""
    emitted = [layer for layer, owner in sorted(owners.items()) if owner.compress_ratio <= prefill]
    if not emitted:
        raise RuntimeError(f"no owner of keys emits in a prefill of {prefill} tokens")
    last, expected = emitted[-1], []
    for position in range(prefill, prefill + steps):
        scored = {}
        for layer in sorted(indexers):
            if layer in owners and (published or (position + 1) % owners[layer].compress_ratio == 0):
                last = layer
            scored[layer] = last
        expected.append(scored)
    return expected


def slot_held(session, positions, same, first, silent):
    """CHECK name=slot, by the reference's form (the module text): per decode step at `positions`, whether the arms
    had the same bits (`same`), the first slot position and whether a layer of rate 2 emitted nothing (`silent`)."""
    if oracle.reference_form(session.reference).publishes_index_keys:
        return bool(same.all())
    return bool(same[positions < first].all() and not same[(positions >= first) & silent].any())


def misdirected(session, indexers, owners):
    """--break slot in the published arm where model.py publishes every owner's keys itself: before they score, the
    index layers that own no keys pointed at the first owner's. Returns the hooks' handles (none where model.py
    keeps the slot: the break then publishes nothing)."""
    if not oracle.reference_form(session.reference).publishes_index_keys:
        return []
    shared, first = session.reference.model.shared_attn, owners[min(owners)]
    return [indexer.register_forward_pre_hook(lambda module, inputs: setattr(shared, "index_k", first.k_cache))
            for layer, indexer in indexers.items() if layer not in owners]


@torch.inference_mode()
def decode(arguments, session, token_ids, text, record):
    """The slot check, then both arms against one prefill. Returns whether the tool ended as its command line
    asks; raises DecodeBudgetTooSmall where the budget leaves too few steps."""
    model, shared, started = session.model, session.reference.model.shared_attn, time.perf_counter()
    _, indexers, owners, _ = layers_of(session)
    publishes = oracle.reference_form(session.reference).publishes_index_keys
    scored, head, label = {}, {}, f"batch={BATCH} text={text}"
    for layer, indexer in indexers.items():                # whose keys each layer scored at its last call
        indexer.register_forward_hook(lambda module, inputs, output, layer = layer: scored.__setitem__(
            layer, next((owner for owner, other in owners.items() if other.k_cache is shared.index_k), "none")))
    model.head.register_forward_pre_hook(lambda _, inputs: head.update(rows = inputs[0][0]))
    input_ids = torch.tensor([token_ids], dtype = torch.long, device = oracle.CARD_DEVICE)
    pooling = {layer: indexer.compress_ratio for layer, indexer in owners.items() if indexer.compress_ratio > 1}

    def publish(indexer, inputs):                                          # each owner's own keys, at every step
        shared.index_k = indexer.k_cache

    def arm(published, prefill, steps, choose = None):
        """A prefill of `prefill` tokens, then decode steps on the text's own tokens. Returns the logits of every
        step [steps, vocabulary], whose keys each layer scored at each step, and the seconds of the prefill."""
        broken = published and arguments.break_name == BREAK_SLOT
        handles = misdirected(session, indexers, owners) if broken else \
            [indexer.register_forward_pre_hook(publish) for indexer in owners.values()] if published else []
        logits, keys = [], []
        try:
            oracle.forget_earlier_runs(session)
            clock = oracle.synchronized_clock()
            with deadline(arguments.pass_timeout_seconds):
                model.forward(input_ids[:, :prefill], 0)
            prefill_seconds, clock = oracle.synchronized_clock() - clock, oracle.synchronized_clock()
            while len(logits) < steps:
                position = prefill + len(logits)                           # generate.py:69-74, one token a step
                logits.append(model.forward(input_ids[:, position:position + 1], position)[1][0].cpu())
                keys.append(dict(scored))
                if choose is not None and len(logits) == PROJECTION_STEPS:
                    steps = choose((oracle.synchronized_clock() - clock) / PROJECTION_STEPS, prefill_seconds)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack(logits).numpy(), keys, prefill_seconds

    def keys_held(phase, prefill, arms_keys):
        """CHECK name=decode_keys: per arm, the (step, layer) pairs whose scored keys are not keys_expected's."""
        differing = {}
        for name, keys in arms_keys.items():
            wanted = keys_expected(indexers, owners, prefill, len(keys), name == PUBLISHED or publishes)
            differing[name] = sum(step_keys.get(layer) != owner for step_keys, step_wanted in zip(keys, wanted)
                                  for layer, owner in step_wanted.items())
        held = not any(differing.values())
        print(f"CHECK {label} name=decode_keys phase={phase} positions={prefill}.."
              f"{prefill + len(next(iter(arms_keys.values()))) - 1} "
              + " ".join(f"{name}_differing={count}" for name, count in differing.items())
              + f" verdict={verdict_text('decode_keys', held, arguments.break_name)}")
        record.setdefault("decode_keys", {})[phase] = differing
        return held

    first = first_slot_position(session, owners)
    check_prefill = first - SLOT_CHECK_STEPS_BEFORE
    if check_prefill < 1 or check_prefill + SLOT_CHECK_STEPS > len(token_ids):
        raise RuntimeError(f"the slot check decodes positions {check_prefill} to "
                           f"{check_prefill + SLOT_CHECK_STEPS - 1} of a sequence of {len(token_ids)} tokens: "
                           "--prefill-tokens and --decode-steps are short")
    as_is, as_is_keys, _ = arm(False, check_prefill, SLOT_CHECK_STEPS)
    fixed, fixed_keys, _ = arm(True, check_prefill, SLOT_CHECK_STEPS)
    for step in range(SLOT_CHECK_STEPS_BEFORE - 1, SLOT_CHECK_STEPS_BEFORE + SLOT_STEPS_SHOWN_AFTER):
        for layer in pooling:                                              # around the first slot position
            print(f"SLOT {label} position={check_prefill + step} layer={layer} rate={pooling[layer]} "
                  f"as_is_keys_of_layer={as_is_keys[step][layer]} published_keys_of_layer={fixed_keys[step][layer]}")
    positions = check_prefill + numpy.arange(SLOT_CHECK_STEPS)
    same = numpy.array([numpy.array_equal(row.view(numpy.uint32), other.view(numpy.uint32))
                        for row, other in zip(as_is, fixed)])
    silent = numpy.array([any((position + 1) % rate for rate in pooling.values()) for position in positions])
    held = slot_held(session, positions, same, first, silent)
    differing = positions[~same]
    print(f"CHECK {label} name=slot first_slot_position={first} positions={positions[0]}..{positions[-1]} "
          f"first_different={differing[0] if len(differing) else 'none'} "
          f"verdict={verdict_text('slot', held, arguments.break_name)}")
    checks = {"slot": held, "decode_keys": keys_held("slot_check", check_prefill, {AS_IS: as_is_keys,
                                                                                   PUBLISHED: fixed_keys})}
    record["slot_check"] = {"first_slot_position": first, "positions": positions.tolist(), "same_bits": same.tolist(),
                            "checks": checks}
    if arguments.break_name == BREAK_SLOT:
        return broken_as_asked(checks, BREAK_SLOT)
    if not all(checks.values()):
        return False

    prefill, asked = arguments.prefill_tokens, arguments.decode_steps
    oracle.forget_earlier_runs(session)
    with deadline(arguments.pass_timeout_seconds):
        model.forward(input_ids[:, :prefill + asked], 0)                   # the prefill every step is compared with
    prefill_logits = model.head.forward(head["rows"][None, prefill:prefill + asked], full_logits = True)[0]
    prefill_logits = prefill_logits.cpu().numpy()

    def choose(seconds_per_step, prefill_seconds):                         # both arms within the budget
        left = arguments.decode_budget_seconds - (time.perf_counter() - started)
        steps = min(asked, int((left - prefill_seconds + PROJECTION_STEPS * seconds_per_step)
                               / (len(DECODE_ARMS) * seconds_per_step)))
        print(f"DECODE {label} steps_asked={asked} steps_chosen={steps} seconds_per_step={seconds_per_step:.3f} "
              f"budget_seconds={arguments.decode_budget_seconds:.0f} reason={'asked' if steps == asked else 'budget'}")
        if steps < FEWEST_DECODE_STEPS:
            raise DecodeBudgetTooSmall(f"the budget leaves {steps} decode steps an arm, fewer than "
                                       f"{FEWEST_DECODE_STEPS}")
        return steps

    as_is, as_is_keys, _ = arm(False, prefill, asked, choose)
    steps = len(as_is)
    fixed, fixed_keys, _ = arm(True, prefill, steps)
    if not keys_held("arms", prefill, {AS_IS: as_is_keys, PUBLISHED: fixed_keys}):
        return False
    next_tokens = numpy.asarray(token_ids[prefill:prefill + steps + 1])
    positions = prefill + numpy.arange(steps)
    silent = numpy.array([any((position + 1) % rate for rate in pooling.values()) for position in positions])
    arms = dict(zip(DECODE_ARMS, (compare.score_positions(prefill_logits[:steps], logits, next_tokens)
                                  for logits in (as_is, fixed))))
    between = compare.score_positions(fixed, as_is, next_tokens)
    for name, scores in arms.items():
        print(f"DECODE {label} arm={name} prefill={prefill} steps={steps} "
              f"mean_nll_prefill={scores['nll_baseline'].mean():.6f} mean_nll={scores['nll_other'].mean():.6f} "
              f"mean_kl={scores['kl_baseline_other'].mean():.6f} same_top={scores['same_top_token'].mean():.4f} "
              f"mean_kl_silent={scores['kl_baseline_other'][silent].mean():.6f} "
              f"mean_kl_emitting={scores['kl_baseline_other'][~silent].mean():.6f}")
    nll, kl = (paired(arms[AS_IS][score] - arms[PUBLISHED][score]) for score in ("nll_other", "kl_baseline_other"))
    print(f"PAIRED {label} as_is_minus_published {text_of(nll, 'nll')} {text_of(kl, 'kl')} "
          f"kl_published_as_is_silent={between['kl_baseline_other'][silent].mean():.6f} "
          f"kl_published_as_is_emitting={between['kl_baseline_other'][~silent].mean():.6f} "
          f"same_top={between['same_top_token'].mean():.4f}")
    verdicts = {"nll": verdict_of(nll), "kl": verdict_of(kl)}
    print(f"VERDICT {label} question=slot verdict="
          + ("not_resolved" if all(verdict == "not_resolved" for verdict, _ in verdicts.values()) else "measurable")
          + "".join(f" {name}={verdict} {name}_reason={reason}" for name, (verdict, reason) in verdicts.items()))
    record.update(prefill = prefill, steps_asked = asked, steps = steps, nll_as_is_minus_published = nll,
                  kl_as_is_minus_published = kl, verdicts = verdicts, positions_silent = silent.tolist(),
                  arms = {name: {"mean_nll": float(scores["nll_other"].mean()),
                                 "mean_kl": float(scores["kl_baseline_other"].mean()),
                                 "same_top": float(scores["same_top_token"].mean()),
                                 "per_position": {score: scores[score].tolist() for score in (
                                     "nll_other", "kl_baseline_other", "same_top_token")}}
                          for name, scores in arms.items()},
                  nll_prefill = arms[AS_IS]["nll_baseline"].tolist())
    return True


def read_settings(checkpoint):
    """What the fit and the candidate stage read of the reference's config.json, read before anything is loaded."""
    path = os.path.join(checkpoint, oracle.INFERENCE_FOLDER_NAME, oracle.MODEL_ARGUMENTS_FILE_NAME)
    with open(path, encoding = "utf8") as handle:
        settings = json.load(handle)
    ratios = settings.get("compress_ratios") if isinstance(settings, dict) else None
    if not isinstance(ratios, list) or not all(type(settings.get(name)) is int for name in FIT_SETTINGS) \
            or min(settings["index_n_heads"], settings["hc_mult"], settings["dim"]) < 1 \
            or settings["candidate_source_layer"] >= len(ratios) \
            or not 0 < settings["n_layers"] <= len(ratios) \
            or not all(type(rate) is int and rate >= 0 for rate in ratios) \
            or (settings["candidate_source_layer"] >= 0 and not (type(ratios[settings["candidate_source_layer"]]) is int
                and min(ratios[settings["candidate_source_layer"]], settings["candidate_topk_blocks"],
                        settings["candidate_block_size"]) > 0)):
        raise ValueError(f"{path} lacks one of {', '.join(FIT_SETTINGS)} and compress_ratios, or holds one out of "
                         "range")
    return settings


def card_bytes_needed(arguments, runs, settings):
    """The peak card memory of the costliest pass: what the indexer of rate 1 holds at once over a prefill of n
    tokens (its scores per head, their weighted copy and their sum, model.py:556-557, plus a mask) grows as n * n;
    the rest is fitted to the two peaks of batch i13, whose scores were bfloat16."""
    heads = settings["index_n_heads"]

    def pair(score_bytes):
        return (SCORE_TENSORS_PER_HEAD * heads + 1) * score_bytes + SCORE_MASK_BYTES

    (short, _), (long, _) = I13_PASS_PEAKS
    rest = [gib * oracle.BYTES_PER_GIB - pair(NARROW_SCORE_BYTES) * tokens * tokens
            for tokens, gib in I13_PASS_PEAKS]
    per_token = (rest[1] - rest[0]) / (long - short)
    base = rest[0] - per_token * short
    tokens = arguments.tokens
    widest = WIDE_SCORE_BYTES if any(INDEX_SCOPE in run.scopes for run in runs) else NARROW_SCORE_BYTES
    needed = base + per_token * tokens + pair(widest) * tokens * tokens
    if arguments.floor_gguf is not None:
        needed = max(needed, base + per_token * tokens + pair(NARROW_SCORE_BYTES) * tokens * tokens
                     + FLOOR_MATRIX_GIB * oracle.BYTES_PER_GIB)
    return needed, widest


def read_entry(path, sequence):
    """The name and token ids of one entry of a tokens file, read before anything is loaded (the oracle checks
    the ids when it loads them)."""
    with open(path, encoding = "utf8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list) or not 0 <= sequence < len(entries):
        raise ValueError(f"{path} holds no entry {sequence}")
    entry = entries[sequence]
    token_ids = entry.get("token_ids") if isinstance(entry, dict) else entry
    name = str(entry.get("name")) if isinstance(entry, dict) and entry.get("name") else f"sequence_{sequence}"
    if not isinstance(token_ids, list) or not all(type(token_id) is int for token_id in token_ids):
        raise ValueError(f"entry {sequence} of {path} is not a list of token ids")
    return name.translate(LINE_SAFE), token_ids


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev = False)
    parser.add_argument("--checkpoint", required = True, type = os.path.abspath, help = "the Hugging Face folder")
    parser.add_argument("--tokens-json", required = True, type = os.path.abspath, help = "of make_text_tokens.py")
    parser.add_argument("--attention-card", required = True, help = "CUDA_VISIBLE_DEVICES value of a CMP 170HX")
    parser.add_argument("--sequence", type = int, default = 0, help = "which entry of the tokens file")
    parser.add_argument("--mode", choices = (MEASURE, DECODE), default = MEASURE)
    parser.add_argument("--tokens", type = int, default = None, help = f"measure: its first N ({DEFAULT_TOKENS})")
    parser.add_argument("--runs", default = DEFAULT_RUNS, help = "measure: form:kinds+kinds or reference, by commas")
    parser.add_argument("--floor-gguf", default = None, type = os.path.abspath, help = "measure: the floor's GGUF")
    parser.add_argument("--bridge-i13", action = "store_true", help = "measure: hold the ubf16 runs to i13's figures")
    parser.add_argument("--prefill-tokens", type = int, default = DEFAULT_PREFILL_TOKENS, help = "decode: P")
    parser.add_argument("--decode-steps", type = int, default = DEFAULT_DECODE_STEPS, help = "decode: N at most")
    parser.add_argument("--decode-budget-seconds", type = float, default = None, help = "decode: from after the load")
    parser.add_argument("--pass-timeout-seconds", type = float, default = None, help = "a pass that lasts longer ends")
    parser.add_argument("--output-json", default = None, type = os.path.abspath, help = "receives every figure")
    parser.add_argument("--fit-only", action = "store_true", help = "print FIT and end: 0 fits, 3 does not")
    parser.add_argument("--break", dest = "break_name", choices = BREAK_NAMES, default = None)
    arguments = parser.parse_args(namespace = argparse.Namespace(**unit_13.ORACLE_SETTINGS))
    sys.stdout.reconfigure(line_buffering = True)                          # a step that is cut short keeps its lines
    measuring, runs = arguments.mode == MEASURE, parse_runs(arguments.runs)
    if measuring:
        arguments.tokens = DEFAULT_TOKENS if arguments.tokens is None else arguments.tokens
    else:
        decode_tokens = arguments.prefill_tokens + arguments.decode_steps + 1      # the last step's next token
        budget = arguments.decode_budget_seconds
        if arguments.tokens is not None or arguments.prefill_tokens < 1 \
                or arguments.decode_steps < FEWEST_DECODE_STEPS or decode_tokens > arguments.maximum_sequence_tokens \
                or (arguments.break_name != BREAK_SLOT and not (budget is not None and 0 < budget < math.inf)):
            parser.error(f"--mode {DECODE} takes no --tokens, --prefill-tokens of 1 or more, --decode-steps of at "
                         f"least {FEWEST_DECODE_STEPS} that fit --maximum-sequence-tokens with the prefill, and a "
                         "positive --decode-budget-seconds")
        arguments.tokens = decode_tokens
    timeout = arguments.pass_timeout_seconds
    if runs is None or arguments.sequence < 0 \
            or not FIRST_POSITION + FEWEST_CHUNKS < arguments.tokens <= arguments.maximum_sequence_tokens \
            or (measuring and not break_fits(arguments.break_name, runs, arguments.bridge_i13)) \
            or (arguments.break_name is not None and arguments.floor_gguf is not None) \
            or (arguments.break_name is not None and (arguments.break_name == BREAK_SLOT) == measuring) \
            or ((arguments.bridge_i13 or arguments.floor_gguf is not None) and not measuring) \
            or (arguments.bridge_i13 and arguments.tokens != I13_TOKENS) \
            or not (timeout is None or 0 < timeout <= PASS_TIMEOUT_MOST_SECONDS):
        parser.error(f"--runs takes each run once, {REFERENCE} or a form of {', '.join(FORMS)} with kinds of "
                     f"{', '.join(KINDS)} joined by +, and one {HELD_WIDE} run per scope; --sequence is 0 or more; "
                     f"--tokens is above {FIRST_POSITION + FEWEST_CHUNKS}; a --break of --mode {MEASURE} goes with "
                     f"its own run alone (the module text lists them), {BREAK_SLOT} with --mode {DECODE}, none with "
                     f"--floor-gguf; --floor-gguf and --bridge-i13 go with --mode {MEASURE}, the bridge at "
                     f"{I13_TOKENS} tokens; --pass-timeout-seconds is positive, at most {PASS_TIMEOUT_MOST_SECONDS}")
    if arguments.floor_gguf is not None and not os.path.isfile(arguments.floor_gguf):
        parser.error(f"--floor-gguf names no file: {arguments.floor_gguf}")
    if arguments.output_json is not None and (os.path.exists(arguments.output_json)
                                              or not os.path.isdir(os.path.dirname(arguments.output_json))):
        parser.error(f"--output-json exists already or its folder does not: {arguments.output_json}")
    try:
        settings = read_settings(arguments.checkpoint)
    except (ValueError, OSError) as failure:
        parser.error(str(failure))
    arguments.first_candidate_position = first_candidate_position(settings)
    text, digest = NOT_READ, None                          # --fit-only reads no tokens: a length is all it asks
    if not arguments.fit_only:
        try:
            text, entry_ids = read_entry(arguments.tokens_json, arguments.sequence)
            digest = hashlib.sha256(numpy.asarray(entry_ids[:arguments.tokens], dtype = DIGEST_DTYPE).tobytes())
        except (ValueError, OverflowError, OSError) as failure:
            parser.error(str(failure))
        digest = digest.hexdigest()
        if arguments.bridge_i13 and digest != I13_TOKENS_SHA256:
            parser.error(f"--bridge-i13: the first {I13_TOKENS} ids have the digest {digest}, not i13's")
    if not torch.cuda.is_available() or arguments.card >= torch.cuda.device_count():
        parser.error(f"there is no card {arguments.card}: the fit is read from the card")
    needed, score_bytes = card_bytes_needed(arguments, runs if measuring else [], settings)
    card_bytes = torch.cuda.get_device_properties(arguments.card).total_memory
    fits = needed + FIT_MARGIN_GIB * oracle.BYTES_PER_GIB <= card_bytes
    print(f"FIT batch={BATCH} mode={arguments.mode} text={text} tokens={arguments.tokens} "
          f"index_scores={'float32' if score_bytes == WIDE_SCORE_BYTES else 'bfloat16'} "
          f"floor={unit_13.NO_OR_YES[arguments.floor_gguf is not None]} "
          f"estimate_gib={needed / oracle.BYTES_PER_GIB:.2f} margin_gib={FIT_MARGIN_GIB:.1f} "
          f"card_gib={card_bytes / oracle.BYTES_PER_GIB:.2f} "
          f"{candidate_stage_text(arguments.first_candidate_position, arguments.tokens)} "
          f"verdict={'fits' if fits else 'does_not_fit'}")
    if arguments.fit_only or not fits:
        return oracle.EXIT_CODE_SUCCESS if fits else EXIT_CODE_DOES_NOT_FIT
    record = {"batch": BATCH, "mode": arguments.mode, "complete": False, "text": text, "sequence": arguments.sequence,
              "tokens": arguments.tokens, "token_ids_sha256": digest, "first_position": FIRST_POSITION,
              "chunk_positions": CHUNK_POSITIONS, "student_t_975": STUDENT_T_975,
              "first_candidate_position": arguments.first_candidate_position, "command_line": sys.argv,
              "torch_version": torch.__version__, "runs": {}}
    exit_code = oracle.EXIT_CODE_FAILURE
    try:
        with tempfile.TemporaryDirectory() as folder:                      # open_session copies the tokenizer there
            arguments.output_directory = folder
            session, token_ids = oracle.open_session(arguments, {})
            record.update(card = torch.cuda.get_device_name(arguments.card),
                          reference_files_sha256 = session.reference.file_digests)
            record["complete"] = (measure if measuring else decode)(arguments, session, token_ids, text, record)
            exit_code = oracle.EXIT_CODE_SUCCESS if record["complete"] else oracle.EXIT_CODE_FAILURE
    except DecodeBudgetTooSmall as failure:
        print(f"DECODE batch={BATCH} text={text} verdict=budget_too_small {' '.join(str(failure).split())}")
        record["budget_too_small"], exit_code = str(failure), EXIT_CODE_BUDGET_TOO_SMALL
    finally:
        if arguments.output_json is not None:
            for result in record["runs"].values():                         # tensors on the card are not numbers
                for name in ("rows", "picks", "nll"):
                    result.pop(name, None)
            with open(arguments.output_json, "x", encoding = "utf8") as handle:
                json.dump(record, handle, indent = JSON_INDENT_SPACES)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
