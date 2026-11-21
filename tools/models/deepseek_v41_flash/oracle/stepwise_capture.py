#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DeepSeek's reference fed one token a step from position 0, its attention state at FP16 precision and the index
keys of each owner published before its indexer: the path of Anemone's first milestone, recorded for its gates
(design/first_token_plan.md 3.1, unit U13a).

The reference is reference_oracle.py's session (every activation quantization held to its definition) under the
u16 run of state_precision_true.py (13b): its switch, its conditions and its checks, imported. Written again here
until 13b gives them as functions: the two arms of its slot check (its decode() goes on to a measure of its own)
and its two break helpers (bound to its break names). "u16" below is that run over all four kinds of state.
Published keys: before every indexer that owns keys, the shared slot is pointed at that indexer's own keys
(model.py:548, :554), so no layer ever scores another group's keys, as the engine does. A step is model.forward of
one token at position p (generate.py:69-74); at p = 0 the reference takes its prompt branch with one row, which
gives what its decode branch would. Modes:
  check     the switch proved first: a u16 pass of the prompt path with its CHECK lines (switch, kept, buffers,
            wide_values, fp16_range); then the slot check of a u16 decode: a prefill to 8 positions before the first
            where a layer of rate 2 scores more entries than it picks (1,026), 16 steps with the shared slot as the
            reference keeps it and 16 with published keys: the same bits before that position, other bits at every
            later one where a layer of rate 2 emits nothing (CHECK slot; where model.py publishes every owner's keys
            itself, 13b's slot_held: the same bits at every step), and each layer scoring the keys the reference's
            order gives (CHECK decode_keys). --break unchanged puts R's values back (switch and wide_values must
            report), --break slot publishes nothing, or points the layers that own no keys at the first owner's
            where model.py publishes them (13b's misdirected; slot and decode_keys must report);
  stepwise  the capture: the given tokens, then --new-tokens greedy ones (the lowest id among the best; the end
            token stops it). CHECK lines: the switch's, against the counts of one token a step (an indexer of rate r
            quantizes no query before an entry is visible, model.py:729); router_recomputed (the gate's scores formed
            again from its own linear output give its ids and weights bit for bit, every layer and step);
            selection_recomputed and moe_recomputed (at the capture points, the index scores formed again give the
            indexer's picks and the pair outputs summed again the expert layer's output); steps (every block once a
            step), points (every capture point holds every record of every layer), greedy, finite, frequencies (the
            words give the model's own tables bit for bit). --break recomputed rotates the ids it compares;
  floor     before any engine run, on the text of a stepwise capture (--stepwise-capture): (b) the layer floor, the
            u16 prompt path with every block fed the capture's input (the streams leaving the block before and the
            carried weights; at an n-gram layer before its n-gram add), so each layer's state builds from captured
            inputs only, against the capture's outputs per layer: relative rms of the streams leaving it per row,
            over the rows whose six experts are the capture's, the 99.9th percentile of |biased score difference|,
            the differing (position, slot) share; (a) the twin, the u16 prompt path through the float32 product
            stand-in of batch f05 (gguf_weights.py's control, inputs as the reference quantizes them) against the
            stepwise capture: mean KL with chunk standard errors, same top token, per position dm, the change of
            the margin m between the capture's two best logits; delta = 2 x the 99.9th percentile of |dm|; and the
            prompt path alone against the capture, the same figures. --output-json receives what
            compare_engine_capture.py reads, with its FLOOR_CONSTANTS, which it holds to its own;
  requests  per entry of a tokens file, the u16 prompt path over its tokens and then --new-tokens greedy decode
            steps with published keys, one capture folder per entry under --output-directory, named by the entry
            (letters, digits, _ . -, not first a dot; each once; not reference_oracle.py's tokenizer folder);
  slab_check long_context.py's stand-in for the indexers held to the reference's own: the u16 prompt path over
            --prompt-tokens tokens and SLAB_CHECK_STEPS teacher-forced steps (ARM lines) with the reference's
            indexers, with the stand-in in slabs of --slab-rows rows and, --break slab, with it broken (each slab's
            rows see one entry fewer); every indexer output, the candidates, every state buffer, the head's input
            rows and every logit row compared bit for bit (SLAB): the stand-in must give the reference's bits (CHECK
            slab), the broken one must not (CHECK slab_break: reported). The candidate stage prunes from 16,384
            tokens, or with --candidate-blocks N (both arms) from N x 8: it then prunes at a length both hold;
  long      G8's long-context reference (wide_prefill.md 8, unit U24): the u16 prompt path over --prompt-tokens
            tokens with the stand-in, the head over the last --flagged-rows rows in one call, then --teacher-forced
            decode steps fed the text's next tokens with published keys; or a part of it, the layers --layers A..B,
            whose state the next part takes from --part-directory (long_context.py's module text); with
            --floor-gguf the twin: the same through the floor's float32 product stand-in, whose figures against
            --reference-capture (the u16 capture of the same tokens, by compare_engine_capture.py's long-context
            computation) --output-json receives as the floor G8 reads. --proof-tokens N: slab_check over the
            first N tokens before it, nothing run unless it holds, with --proof-candidate-blocks B the candidate
            stage cut to B blocks there alone (refused where it would not prune within N), so that the proof
            prunes as a long pass does. CHECK lines: the switch's, notes (every index layer noted every sampled row
            and step), decode_keys (as check's, published), finite.
            slab_check and long refuse, before anything is loaded, a pass (long: or its proof) the card does not
            hold (FIT, exit 3).

    stepwise_capture.py --checkpoint CHECKPOINT --attention-card N --mode check --tokens-json TEXT [--tokens 2048]
    stepwise_capture.py ... --mode stepwise --tokens-json TEXT --tokens 2048 --all-logits --frequencies \
                            --output-directory CAPTURE
    stepwise_capture.py ... --mode stepwise --prompt-json PROMPT --new-tokens 64 --all-logits --output-directory CAPTURE
    stepwise_capture.py ... --mode floor --tokens-json TEXT --tokens 2048 --stepwise-capture CAPTURE \
                            --floor-gguf GGUF --output-json FLOOR
    stepwise_capture.py ... --mode requests --tokens-json REQUESTS --new-tokens 64 --output-directory FOLDER
    stepwise_capture.py ... --mode slab_check --tokens-json TEXT --prompt-tokens 2148 [--slab-rows 256]
                            [--candidate-blocks 64] [--break slab]
    stepwise_capture.py ... --mode long --tokens-json TEXT --prompt-tokens 32768 [--teacher-forced 64]
                            [--flagged-rows 128] [--layers A..B --part-directory PARTS] [--output-directory CAPTURE]
                            [--floor-gguf GGUF --reference-capture CAPTURE --output-json FLOOR]
                            [--proof-tokens 2148 [--proof-candidate-blocks 64]]

A capture folder is reference_oracle.py's (tensors.bin, manifest.json; bfloat16 held as float32, the record says
reference_dtype). Records of every position (p from 0 to the last step, at which logits were computed):
  input.token_ids [1, p] (the token fed at each step), input.embeddings [1, p, 5120];
  layer_L.block.output [1, p, 4, 5120] and layer_L.block.output_collapse_weights [1, p, 4] (the carried weights),
  layer_L.mlp.routed_expert_ids and .route_weights [p, 6] (in the gate's order), layer_L.mlp.biased_scores [p, 384];
  layer_L.ngram.row_ids [1, p, 24], .rows [1, p, 24, 256] (the table rows, dequantized) and .output [1, p, 4, 5120]
  (the streams after the n-gram add) of layers 1 and 14;
  logits.top_ids, logits.top_values [p, 16] (descending, the lowest id first among equals), logits.log_sum_exp [p]
  (float64), with --all-logits logits.all_rows [1, p, 129280]; generate.token_ids [k] with new tokens: token j
  (from 0) is the argmax of the logits of position prompt_tokens - 1 + j.
Capture points (stepwise_recorder.py records them), at layers CAPTURE_LAYERS and positions CAPTURE_POSITIONS and
at the positions of generated steps CAPTURE_GENERATED_STEPS (step k chooses token k - 1, at position
prompt_tokens - 2 + k): layer_L.position_P.<record> for every record of POINT_RECORDS (compare_engine_capture.py)
the layer computes then, and position_P.final.normed. Snapshots after the step of SNAPSHOT_POSITIONS:
layer_L.position_P.snapshot.ring [128, 512] of every capture layer, .pool [entries, 512], .keys [entries, 128],
.kv_state and .score_state [2, 512] of the four owners.
With --frequencies: frequencies.theta_T.words [32] (the tensor before torch.outer, model.py:376-388, formed on the
card as the model was), .cos_sin_below_16384 [16384, 32, 2] (the model's freqs_cis as cos, sin), .positions [64]
and .cos_sin_sampled [64, 32, 2] (positions up to 1,048,575, from the same words).
A requests folder: the same token and logit records (the prompt's from the head over every row; its results say
whether that gave the last row the bits of the forward's own, which chose the first token), generate.token_ids.
A long folder (written by the part that ends the model; N prompt tokens, T steps, F flagged rows): input.token_ids
[1, N + T + 1] (the prompt, the tokens fed, the token after the last), logits.flagged_positions [F + T] (N - F to
N + T - 1), logits.flagged_rows [1, F + T, 129280] (float32, whole); index.sampled_positions [R] (64 prompt rows
spread from the first at which every index layer picks fewer entries than it sees, the flagged prompt rows, every
step) and per index layer L layer_L.index.picks [R, 512] (entries, ascending, -1 none), .cut_score [R] (the 512th
score; -inf where a row sees fewer), .tied_at_cut [R] (entries scoring exactly the cut), .picked_at_cut [R].
Printed: FIT, RUN, CHECK, SLOT, STEPWISE, LAYER, FLOOR, REQUEST, ARM, SLAB, BLOCK, LONG, PICKS and one VERDICT. Exit 0
when every check held (with a break: when the checks it aims at reported and every other held).
Refused: a mode without the options it needs or with those of another; a break of another mode; tokens that do not
fit the slot check, or --maximum-sequence-tokens; a capture folder or output file that exists; a stepwise capture of
other tokens or without the logits of every row for the floor; a part of layers already run there, earlier parts of
another run, or a reference capture of others for the twin; what reference_oracle.py refuses.
Not shown: the published reference (8-bit state; M2d's arm); the candidate stage below 16,385 entries; the stand-in
against the reference at a length whose reference pass the card does not hold (slab_check shows it where it fits);
anything of Anemone's own arithmetic. Measurement tool: not part of the engine.
"""
import argparse
import collections
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
import types

import numpy
import torch

import compare_engine_capture as gates
import long_context as context
import reference_oracle as oracle
import state_precision as unit_13
import state_precision_true as unit_13b
import stepwise_recorder as recording

MODES = (CHECK, STEPWISE, FLOOR, REQUESTS, SLAB_CHECK, LONG) = ("check", "stepwise", "floor", "requests", "slab_check",
                                                                "long")
BREAK_RECOMPUTED, BREAK_SLAB = "recomputed", "slab"
BREAKS = {unit_13b.BREAK_UNCHANGED: CHECK, unit_13b.BREAK_SLOT: CHECK, BREAK_RECOMPUTED: STEPWISE,
          BREAK_SLAB: SLAB_CHECK}
BREAK_TARGETS = {**unit_13b.BREAK_TARGETS, BREAK_RECOMPUTED: ("router_recomputed",)}
RUN_LABEL = f"{unit_13b.U16}:{unit_13b.EVERY_KIND}"
TOP_LOGITS = 16
FREQUENCY_ROWS, FREQUENCY_SAMPLES, FREQUENCY_LAST_POSITION = 16384, 64, 2 ** 20 - 1
REQUEST_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*")   # a folder's name, never . or ..
STREAM_AXIS = 2                              # the streams are [batch, sequence, stream, dim] (model.py:1258)
REQUESTS_LONGEST_TOKENS = 16384              # batch i13b: a u16 pass of 16,384 tokens fits the PRO with the floor
HEAD_ROWS_AT_A_TIME = 256                    # 132 MB of float32 logits
DEFAULT_STEP_TIMEOUT_SECONDS, DEFAULT_PASS_TIMEOUT_SECONDS = 600, 1800
PROGRESS_STEPS = 256                         # a STEPWISE line every so many steps
JSON_INDENT_SPACES = 1
SLAB_CHECK_STEPS = 4                         # decode steps after the prompt in each arm: the stand-in's other branch
ARMS = (REFERENCE_ARM, SLABS_ARM, BROKEN_ARM) = ("reference", "slabs", "broken")
DEFAULT_TEACHER_FORCED, DEFAULT_FLAGGED_ROWS = 64, 128      # wide_prefill.md 8, G8
LAYER_RANGE = re.compile(r"(?P<first>\d+)\.\.(?P<last>\d+)")
NOTE_TEXT = {"picks": "the picks as entries, ascending, -1 none", "cut_score": "the 512th score, -inf: no cut",
             "tied_at_cut": "entries scoring exactly the cut", "picked_at_cut": "of them, picked"}


def verdict_text(name, held, break_name):
    return "ok" if held else "reported" if name in BREAK_TARGETS.get(break_name, ()) else "FAILED"


def broken_as_asked(checks, break_name):
    targets = BREAK_TARGETS[break_name]
    return not any(checks.get(name, True) for name in targets) and all(
        held for name, held in checks.items() if name not in targets)


def stepwise_expected_values(session, positions):
    """unit_13.expected_values for steps of one token from position 0: the same, but an indexer of rate r is called,
    and quantizes its queries, only at steps where an entry is visible (model.py:729-735), from position r - 1 on."""
    expected, sizes = unit_13.expected_values(session, positions), session.model_arguments
    _, indexers, _, _ = unit_13b.layers_of(session)
    expected[unit_13b.QUERIES] = sum(max(positions - indexer.compress_ratio + 1, 0) for indexer in indexers.values()) \
        * sizes.index_n_heads * sizes.index_head_dim
    return expected


@contextlib.contextmanager
def u16_state(session, switch, run, published):
    """The u16 run's conditions (13b), the switch begun, every buffer fresh and, published, each owner's keys put in
    the shared slot before its indexer runs."""
    owners = unit_13b.layers_of(session)[2].values()
    shared = session.reference.model.shared_attn
    with unit_13b.conditions(session, run, False):
        switch.begin(run)
        oracle.forget_earlier_runs(session)
        handles = [owner.register_forward_pre_hook(lambda module, inputs: setattr(shared, "index_k", module.k_cache))
                   for owner in owners] if published else []
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def switch_checks(session, switch, run, expected, label, break_name):
    """13b's checks of a pass, while its buffers are in place, over the kinds it is expected to hand over (a part of
    long's may hold no block of a kind); prints its RUN and CHECK lines."""
    counts, buffers = switch.counts, unit_13b.scope_buffers(session)
    present = frozenset(kind for kind in run.kinds if expected[kind])
    shown = run._replace(kinds = present, scopes = frozenset(unit_13b.SCOPE_OF[kind] for kind in present))
    checks = {"switch": unit_13.switch_held(switch, expected)
              and all(counts["changed"][kind] > 0 for kind in present & switch.kinds),
              "kept": not any(counts["not_r"].values()), "buffers": unit_13b.buffers_held(buffers, shown, switch),
              "wide_values": unit_13b.wide_values_held(buffers, shown, switch),
              "fp16_range": not any(counts["fp16_overflow"].values())}
    print(f"RUN {label} run={run.label} card_gib={torch.cuda.max_memory_allocated() / oracle.BYTES_PER_GIB:.2f} "
          + " ".join(f"{kind}={switch.left[kind]}/{switch.asked[kind]}" + "".join(
              f",{name}:{counts[name][kind]}" for name in unit_13b.VALUE_COUNTS) for kind in unit_13b.KINDS))
    for name, held in checks.items():
        print(f"CHECK {label} name={name} verdict={verdict_text(name, held, break_name)}")
    return checks


def head_rows(session):
    """A hook keeping the head's input of the last pass; returns the dict it fills and its handle."""
    kept = {}
    return kept, session.model.head.register_forward_pre_hook(lambda _, inputs: kept.update(rows = inputs[0][0]))


def every_row_logits(session, rows):
    """The head over each row of a pass (model.py:1008-1012 with full_logits), a slab at a time."""
    for first in range(0, len(rows), HEAD_ROWS_AT_A_TIME):
        yield session.model.head.forward(rows[None, first:first + HEAD_ROWS_AT_A_TIME], full_logits = True)[0]


def top_of(logits):
    """The top 16 of a row of logits (descending, the lowest id first among equals) and its log-sum-exp."""
    values, ids = torch.sort(logits, descending = True, stable = True)
    return ids[..., :TOP_LOGITS].cpu(), values[..., :TOP_LOGITS].cpu(), torch.logsumexp(logits.double(), dim = -1).cpu()


@torch.inference_mode()
def check(arguments, session, switch, run, token_ids, label):
    """The switch's checks on the prompt path, then the slot check of a u16 decode (the module text)."""
    model, break_name = session.model, arguments.break_name
    input_ids = torch.tensor([token_ids], device = oracle.CARD_DEVICE)
    checks = {}
    if break_name != unit_13b.BREAK_SLOT:
        torch.cuda.reset_peak_memory_stats()
        with u16_state(session, switch, run, True), unit_13b.deadline(arguments.pass_timeout_seconds):
            model.forward(input_ids, 0)
            checks.update(switch_checks(session, switch, run, unit_13.expected_values(session, len(token_ids)),
                                        f"{label} pass=prompt_path", break_name))
        if break_name is not None:
            return broken_as_asked(checks, break_name)
    _, indexers, owners, _ = unit_13b.layers_of(session)
    shared, pooling = session.reference.model.shared_attn, {layer: indexer.compress_ratio for layer, indexer in
                                                            owners.items() if indexer.compress_ratio > 1}
    publishes = oracle.reference_form(session.reference).publishes_index_keys
    first = unit_13b.first_slot_position(session, owners)
    prefill, steps, scored = first - unit_13b.SLOT_CHECK_STEPS_BEFORE, unit_13b.SLOT_CHECK_STEPS, {}
    if prefill < 1 or prefill + steps > len(token_ids):
        raise RuntimeError(f"the slot check decodes positions {prefill} to {prefill + steps - 1}: "
                           f"{len(token_ids)} tokens do not reach them")
    handles = [indexer.register_forward_hook(lambda module, inputs, output, layer = layer: scored.__setitem__(
        layer, next((owner for owner, other in owners.items() if other.k_cache is shared.index_k), "none")))
        for layer, indexer in indexers.items()]
    arms = {}
    try:
        for published in (False, True):
            logits, keys, broken = [], [], published and break_name == unit_13b.BREAK_SLOT
            handles += unit_13b.misdirected(session, indexers, owners) if broken else []    # the last arm's
            with u16_state(session, switch, run, published and not broken):
                with unit_13b.deadline(arguments.pass_timeout_seconds):
                    model.forward(input_ids[:, :prefill], 0)
                for position in range(prefill, prefill + steps):
                    with unit_13b.deadline(arguments.step_timeout_seconds):
                        logits.append(model.forward(input_ids[:, position:position + 1], position)[1][0].cpu())
                    keys.append(dict(scored))
            arms[published] = (torch.stack(logits).numpy(), keys)
    finally:
        for handle in handles:
            handle.remove()
    positions = prefill + numpy.arange(steps)
    same = numpy.array([numpy.array_equal(row.view(numpy.uint32), other.view(numpy.uint32))
                        for row, other in zip(arms[False][0], arms[True][0])])
    silent = numpy.array([any((position + 1) % rate for rate in pooling.values()) for position in positions])
    for step, layer in ((step, layer) for step in range(unit_13b.SLOT_CHECK_STEPS_BEFORE - 1,
                                                        unit_13b.SLOT_CHECK_STEPS_BEFORE + 2) for layer in pooling):
        print(f"SLOT {label} position={prefill + step} layer={layer} as_is_keys_of_layer="
              f"{arms[False][1][step][layer]} published_keys_of_layer={arms[True][1][step][layer]}")
    checks["slot"] = unit_13b.slot_held(session, positions, same, first, silent)
    differing = positions[~same]
    print(f"CHECK {label} name=slot first_slot_position={first} positions={positions[0]}..{positions[-1]} "
          f"first_different={differing[0] if len(differing) else 'none'} "
          f"verdict={verdict_text('slot', checks['slot'], break_name)}")
    wrong = {published: sum(step_keys.get(layer) != owner for step_keys, wanted in zip(
        arms[published][1], unit_13b.keys_expected(indexers, owners, prefill, steps, published or publishes))
        for layer, owner in wanted.items()) for published in arms}
    checks["decode_keys"] = not any(wrong.values())
    print(f"CHECK {label} name=decode_keys as_is_differing={wrong[False]} published_differing={wrong[True]} "
          f"verdict={verdict_text('decode_keys', checks['decode_keys'], break_name)}")
    return broken_as_asked(checks, break_name) if break_name is not None else all(checks.values())


def capture_positions_of(prompt_tokens, steps, new_tokens):
    """The positions of CAPTURE_POSITIONS a run reaches, and those of its generated steps CAPTURE_GENERATED_STEPS
    (stepwise_recorder.py)."""
    return {position for position in recording.CAPTURE_POSITIONS if position < steps} | {
        prompt_tokens - 2 + step for step in recording.CAPTURE_GENERATED_STEPS if new_tokens and step <= new_tokens}


def frequencies(session, writer, label):
    """The words of each theta as the reference forms them (the module text): the tensor torch.outer receives in
    model.py:388 while precompute_freqs_cis runs again on the card, as at the model's construction. Returns whether
    both calls received the same words and the tables they give are the model's own, bit for bit."""
    reference_model, sizes, held, outer = session.reference.model, session.model_arguments, True, torch.outer
    attentions = {}
    for block in session.model.layers:                                     # one of each kind: window only, compressed
        attentions.setdefault(bool(block.attn.compress_ratio), block.attn)
    sampled = numpy.unique(numpy.round(numpy.geomspace(FREQUENCY_ROWS, FREQUENCY_LAST_POSITION, FREQUENCY_SAMPLES)))
    if len(sampled) != FREQUENCY_SAMPLES:
        raise RuntimeError("the sampled positions are not distinct")
    sampled = torch.from_numpy(sampled.astype(numpy.int64))
    for compressed, attention in sorted(attentions.items()):
        original, theta = (sizes.original_seq_len, sizes.compress_rope_theta) if compressed else (0, sizes.rope_theta)
        words = []

        def recorded_outer(positions, frequencies):
            words.append(frequencies.clone())
            return outer(positions, frequencies)

        def table(rows):
            torch.outer = recorded_outer
            try:
                with torch.device(oracle.CARD_DEVICE):                    # as build_model constructs the model
                    return reference_model.precompute_freqs_cis.__wrapped__(
                        sizes.rope_head_dim, rows, original, theta, sizes.rope_factor, sizes.beta_fast, sizes.beta_slow)
            finally:
                torch.outer = outer

        own = torch.view_as_real(attention.freqs_cis)
        short, long = torch.view_as_real(table(len(own))), torch.view_as_real(table(FREQUENCY_LAST_POSITION + 1))
        same = len(words) == 2 and oracle.same_bits(words[0], words[1]) and oracle.same_bits(short, own) \
            and oracle.same_bits(long[:len(own)], own) and len(own) >= FREQUENCY_ROWS
        held &= same
        name, facts = f"frequencies.theta_{int(theta)}", dict(theta = theta, original_seq_len = original)
        for part, values, description in (
                ("words", words[0], "the words the angles are positions times (model.py:376-387), on the card"),
                (f"cos_sin_below_{FREQUENCY_ROWS}", own[:FREQUENCY_ROWS], "the model's freqs_cis: [position, 32, 2]"),
                ("positions", sampled, "positions sampled up to 1,048,575"),
                ("cos_sin_sampled", long[sampled.to(long.device)], "cos and sin there, from the same words")):
            writer.add(f"{name}.{part}", values, None, description, **facts)
        print(f"CHECK {label} name=frequencies theta={theta} yarn={'yes' if original else 'no'} "
              f"verdict={verdict_text('frequencies', same, None)}")
    return held


@torch.inference_mode()
def stepwise(arguments, session, switch, run, token_ids, writer, label):
    """The stepwise capture (the module text). Returns its checks."""
    model, sizes, given, new = session.model, session.model_arguments, token_ids, arguments.new_tokens
    steps = len(given) + new - 1 if new else len(given)
    captured = capture_positions_of(len(given), steps, new)
    snapshots = {position for position in recording.SNAPSHOT_POSITIONS if position < steps}
    recorder = recording.StepRecorder(session, writer, switch, steps, captured, snapshots,
                            arguments.break_name == BREAK_RECOMPUTED)
    checks = {"frequencies": frequencies(session, writer, label)} if arguments.frequencies else {}
    tops, fed, generated, greedy, finite = [], [], [], True, True
    all_logits = torch.empty((steps, sizes.vocab_size), dtype = torch.float32) if arguments.all_logits else None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with u16_state(session, switch, run, True):
        switch.recorder = recorder
        recorder.install()
        try:
            for position in range(steps):
                fed.append(given[position] if position < len(given) else generated[-1])
                recorder.begin_step(position)
                with unit_13b.deadline(arguments.step_timeout_seconds):
                    logits = model.forward(torch.tensor([[fed[-1]]], device = oracle.CARD_DEVICE), position)[1]
                finite &= bool(torch.isfinite(logits).all())
                tops.append(top_of(logits))
                if all_logits is not None:
                    all_logits[position].copy_(logits[0])
                choice = int(torch.argmax(logits[0]))                      # the first of the largest
                greedy &= choice == int(tops[-1][0][0, 0])
                if position in snapshots:
                    recorder.snapshot(position)
                if position % PROGRESS_STEPS == 0:
                    print(f"STEPWISE {label} position={position} seconds={time.perf_counter() - started:.1f}")
                if new and position >= len(given) - 1:
                    generated.append(choice)
                    if choice == session.tokenizer.eos_token_id or len(generated) == new:
                        break
            done = len(fed)
            checks.update(switch_checks(session, switch, run, stepwise_expected_values(session, done), label,
                                        arguments.break_name))
        finally:
            recorder.remove()
            switch.recorder = None
    seconds, counts, router = time.perf_counter() - started, recorder.counts, recorder.router
    reached = {(layer, position) for layer in recording.CAPTURE_LAYERS for position in captured if position < done}
    checks.update(
        steps = bool((recorder.ran[:, :done] == 1).all() and not recorder.ran[:, done:].any()),
        router_recomputed = router.differing == 0 and router.calls == len(model.layers) * done,
        selection_recomputed = counts["selection_differing"] == 0 and counts["selection"] > 0,
        moe_recomputed = counts["moe_differing"] == 0 and counts["moe"] == len(reached),
        points = set(recorder.points) == reached
        and all(gates.EVERY_LAYER_RECORDS <= recorder.points[key] for key in reached),
        greedy = greedy, finite = finite)
    calls = {"router_recomputed": (router.calls, router.differing),
             **{f"{kind}_recomputed": (counts[kind], counts[f"{kind}_differing"]) for kind in ("selection", "moe")}}
    for name in (*calls, "steps", "points", "greedy", "finite"):
        figures = f"calls={calls[name][0]} differing={calls[name][1]} " if name in calls else ""
        print(f"CHECK {label} name={name} {figures}verdict={verdict_text(name, checks[name], arguments.break_name)}")
    add, top_records = writer.add, [torch.cat([top[part] for top in tops]) for part in range(len(tops[0]))]
    add(gates.TOKEN_IDS, torch.tensor([fed]), None, "the token fed at each step: given, then generated")
    add(gates.EMBEDDINGS, recorder.embeddings[None, :done], None, "the embedding row of each step's token")
    for layer in range(len(model.layers)):
        add(f"layer_{layer}.block.output", recorder.streams[layer, None, :done], layer, "streams leaving the block")
        add(f"layer_{layer}.block.output_collapse_weights", recorder.carried[layer, None, :done], layer, "carried")
        add(f"layer_{layer}.mlp.routed_expert_ids", recorder.ids[layer, :done], layer, "the gate's ids, in its order")
        add(f"layer_{layer}.mlp.route_weights", recorder.weights[layer, :done], layer, "the gate's weights")
        add(f"layer_{layer}.mlp.biased_scores", recorder.biased[layer, :done], layer, "sqrt(softplus) plus the bias")
    for table, layer in enumerate(recorder.engram_layers):
        add(f"layer_{layer}.ngram.row_ids", recorder.row_ids[None, :done, table], layer, "the n-gram row ids")
        add(f"layer_{layer}.ngram.rows", recorder.engram_rows[table, None, :done], layer, "the rows, dequantized")
        add(f"layer_{layer}.ngram.output", recorder.engram_output[table, None, :done], layer, "after the n-gram add")
    for name, values in zip((gates.TOP_IDS, gates.TOP_VALUES, "logits.log_sum_exp"), top_records):
        add(name, values, None, "of each step's logits: the 16 largest (the lowest id first), their sum's log")
    if all_logits is not None:
        add(gates.ALL_LOGITS, all_logits[None, :done], None, "the logits of every step: [sequence, position, vocab]")
    if new:
        add(gates.GENERATED_IDS, torch.tensor(generated, dtype = torch.int64), None, "token j: the argmax of "
            "position prompt_tokens - 1 + j")
    writer.results.update(prompt_tokens = len(given), steps = done, generated = len(generated),
                          stopped_on_the_end_token = bool(generated)
                          and generated[-1] == session.tokenizer.eos_token_id,
                          seconds = seconds, capture_positions = sorted({position for _, position in reached}),
                          snapshot_positions = sorted(snapshots), checks = checks,
                          text = session.tokenizer.decode(generated) if generated else None)
    print(f"STEPWISE {label} steps={done} prompt_tokens={len(given)} generated={len(generated)} "
          f"seconds={seconds:.1f} seconds_per_step={seconds / done:.3f} "
          f"card_gib={torch.cuda.max_memory_allocated() / oracle.BYTES_PER_GIB:.2f} capture_points={len(reached)}")
    return checks


def layer_floor(arguments, session, switch, run, input_ids, capture, label, checks):
    """(b) of the floor (the module text). Returns the figures by layer."""
    model, sizes, figures, exact, router_of = session.model, session.model_arguments, {}, [], {}
    engram_layers = {block.layer_id for block in model.layers if block.engram is not None}
    router = recording.RouterWatch(session, lambda layer, *parts: router_of.__setitem__(layer, parts), False)

    def card(name):
        return torch.from_numpy(numpy.array(capture.array(name))).to(oracle.CARD_DEVICE)   # a writable copy

    def streams_leaving(layer):                                            # bfloat16 values held as float32
        held = card(f"layer_{layer}.block.output")
        narrow = held.to(torch.bfloat16)
        exact.append(oracle.same_bits(narrow.float(), held))
        return narrow

    def block_begins(block, inputs):
        layer = block.layer_id
        if layer == 0:                                                     # the reference's own embedding, captured
            embeddings = card(gates.EMBEDDINGS).to(torch.bfloat16).unsqueeze(STREAM_AXIS)     # as model.py:1258
            checks["embedding"] = oracle.same_bits(inputs[recording.BLOCK_STREAMS_ARGUMENT],
                                                   embeddings.repeat(1, 1, sizes.hc_mult, 1))
            return None
        arguments = list(inputs)
        if layer not in engram_layers:                                     # there the n-gram hook injected them
            arguments[recording.BLOCK_STREAMS_ARGUMENT] = streams_leaving(layer - 1)
        arguments[recording.BLOCK_PRE_MIX_ARGUMENT] = card(f"layer_{layer - 1}.block.output_collapse_weights")
        return tuple(arguments)

    def block_ends(block, inputs, output):
        layer = block.layer_id
        _, _, biased, indices, _ = router_of.pop(layer)
        figures[layer] = gates.layer_figures(
            capture.array(f"layer_{layer}.block.output")[0], output[0][0].float().cpu().numpy(),
            capture.array(f"layer_{layer}.mlp.routed_expert_ids"), indices.cpu().numpy(),
            capture.array(f"layer_{layer}.mlp.biased_scores"), biased.cpu().numpy())
        print(f"LAYER {label} layer={layer} " + " ".join(f"{name}={figures[layer][name]:.4e}" for name in (
            "mean_relative_rms_same_ids", "largest_row_relative_rms_same_ids", "router_difference_p999",
            "router_difference_largest", "differing_pair_share")) + f" same_id_rows={figures[layer]['same_id_rows']}")

    handles = []
    with u16_state(session, switch, run, True):
        router.install()
        try:
            for block in model.layers:
                handles += [block.register_forward_pre_hook(block_begins), block.register_forward_hook(block_ends)]
                if block.engram is not None:
                    handles.append(block.engram.register_forward_pre_hook(
                        lambda engram, inputs: (streams_leaving(engram.layer_id - 1), *inputs[1:])))
            with unit_13b.deadline(arguments.pass_timeout_seconds):
                model.forward(input_ids, 0)
            checks.update({f"layer_floor_{name}": held for name, held in switch_checks(
                session, switch, run, unit_13.expected_values(session, input_ids.size(1)),
                f"{label} pass=layer_floor", None).items()})
        finally:
            for handle in handles:
                handle.remove()
            router.remove()
    checks.update(injected_exact = bool(exact) and all(exact), layers = len(figures) == len(model.layers),
                  layer_floor_router_recomputed = router.differing == 0 and router.calls == len(model.layers))
    return figures


@torch.inference_mode()
def floor(arguments, session, switch, run, token_ids, label):
    """The floor (the module text): the prompt path, the layer floor, then the twin, which installs the stand-in
    for the rest of the process. Returns whether every check held."""
    capture = gates.CaptureFolder(arguments.stepwise_capture)
    if capture.conditions.get("mode") != STEPWISE or capture.conditions.get("run") != RUN_LABEL \
            or not capture.has(gates.ALL_LOGITS) or not numpy.array_equal(capture.token_ids(), token_ids):
        raise RuntimeError("--stepwise-capture is no u16 stepwise capture of these tokens with every row's logits")
    model, checks = session.model, {}
    input_ids, tokens = torch.tensor([token_ids], device = oracle.CARD_DEVICE), numpy.asarray(token_ids)
    kept, handle = head_rows(session)

    def logits_of_pass(name):
        torch.cuda.reset_peak_memory_stats()
        with u16_state(session, switch, run, True), unit_13b.deadline(arguments.pass_timeout_seconds):
            model.forward(input_ids, 0)
            checks.update({f"{name}_{check}": held for check, held in switch_checks(
                session, switch, run, unit_13.expected_values(session, len(token_ids)), f"{label} pass={name}",
                None).items()})
            return numpy.concatenate([part.cpu().numpy() for part in every_row_logits(session, kept["rows"])])

    record = {"kind": gates.FLOOR_KIND, "complete": False, gates.FLOOR_TOKENS_SHA256: gates.tokens_sha256(tokens),
              "positions": len(tokens), "stepwise_capture": capture.conditions.get("arguments"),
              "constants": gates.floor_constants()}
    try:
        reference = capture.array(gates.ALL_LOGITS)[0]
        prompt_logits = logits_of_pass("prompt_path")
        record[gates.FLOOR_LAYERS] = {str(layer): figures for layer, figures in layer_floor(
            arguments, session, switch, run, input_ids, capture, label, checks).items()}
        torch.cuda.empty_cache()
        gguf_weights = importlib.import_module("gguf_weights")             # as reference_oracle.py imports it
        record["floor_install"] = gguf_weights.install(
            session.reference.model, model, session.stages, oracle.HostResidentExpert, oracle.HostResidentTable,
            arguments.floor_gguf, gguf_weights.INPUTS_AS_THE_REFERENCE, ())             # no part taken: the control
        twin_logits = logits_of_pass("twin")
    finally:
        handle.remove()
    for name, logits in (("prompt_path", prompt_logits), (gates.FLOOR_TWIN, twin_logits)):
        figures = record[name] = gates.logit_figures(reference, logits, tokens)
        print(f"FLOOR {label} arm={name} positions={figures['positions_scored']} mean_kl={figures['mean_kl']:.6f} "
              f"mean_kl_standard_error={figures['mean_kl_standard_error']:.6f} chunks={figures['chunks']} "
              f"same_top={figures['same_top']:.4f} margin_change_p99={figures['margin_change_p99']:.4e} "
              f"margin_change_p999={figures['margin_change_p999']:.4e} delta={figures['delta']:.4e}")
    for name in ("embedding", "injected_exact", "layers", "layer_floor_router_recomputed"):
        print(f"CHECK {label} name={name} verdict={verdict_text(name, checks.get(name, False), None)}")
    record.update(checks = checks, complete = all(checks.values()) and "embedding" in checks,
                  command_line = sys.argv, reference_files_sha256 = session.reference.file_digests)
    with open(arguments.output_json, "x", encoding = "utf8") as handle_out:
        json.dump(record, handle_out, indent = JSON_INDENT_SPACES)
    return record["complete"]


@torch.inference_mode()
def request(arguments, session, switch, run, token_ids, writer, label):
    """One request reference (the module text). Returns its checks."""
    model, prompt, end_token = session.model, len(token_ids), session.tokenizer.eos_token_id
    kept, handle = head_rows(session)
    tops, fed, generated = [], list(token_ids), []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with u16_state(session, switch, run, True):
            with unit_13b.deadline(arguments.pass_timeout_seconds):
                logits = model.forward(torch.tensor([token_ids], device = oracle.CARD_DEVICE), 0)[1]
            for part in every_row_logits(session, kept["rows"]):
                tops.append(top_of(part))
            last_row_same = oracle.same_bits(part[-1], logits[0])
            generated.append(int(torch.argmax(logits[0])))
            while len(generated) < arguments.new_tokens and generated[-1] != end_token:
                fed.append(generated[-1])
                with unit_13b.deadline(arguments.step_timeout_seconds):
                    logits = model.forward(torch.tensor([[fed[-1]]], device = oracle.CARD_DEVICE), len(fed) - 1)[1]
                tops.append(top_of(logits))
                generated.append(int(torch.argmax(logits[0])))
            prompt_expected = unit_13.expected_values(session, prompt)
            later, earlier = stepwise_expected_values(session, len(fed)), stepwise_expected_values(session, prompt)
            checks = switch_checks(session, switch, run, {kind: prompt_expected[kind] + later[kind] - earlier[kind]
                                                          for kind in prompt_expected}, label, None)
    finally:
        handle.remove()
    top_ids, top_values, log_sum_exp = (torch.cat([top[part] for top in tops]) for part in range(len(tops[0])))
    checks["finite"] = bool(torch.isfinite(top_values).all() and torch.isfinite(log_sum_exp).all())
    writer.add(gates.TOKEN_IDS, torch.tensor([fed]), None, "the prompt, then the tokens fed to the decode steps")
    for name, values in zip((gates.TOP_IDS, gates.TOP_VALUES, "logits.log_sum_exp"),
                            (top_ids, top_values, log_sum_exp)):
        writer.add(name, values, None, "the prompt's (the head over every row), then the decode steps'")
    writer.add(gates.GENERATED_IDS, torch.tensor(generated, dtype = torch.int64), None, "the first from the forward's "
               "own last row, then one per decode step")
    writer.results.update(prompt_tokens = prompt, decode_steps = len(fed) - prompt, generated = len(generated),
                          last_of_all_rows_has_the_bits_of_last_row = last_row_same, checks = checks,
                          seconds = time.perf_counter() - started, text = session.tokenizer.decode(generated))
    print(f"REQUEST {label} prompt_tokens={prompt} generated={len(generated)} "
          f"seconds={time.perf_counter() - started:.1f} last_row_same_bits={'yes' if last_row_same else 'no'} "
          f"card_gib={torch.cuda.max_memory_allocated() / oracle.BYTES_PER_GIB:.2f}")
    print(f"CHECK {label} name=finite verdict={verdict_text('finite', checks['finite'], None)}")
    return checks


def expected_values(session, prompt_tokens, layers, steps):
    """The values a prompt through the blocks `layers` hands to the in-place quantizations, then `steps` decode
    steps through every block: unit_13.expected_values of those blocks, stepwise_expected_values for the steps."""
    blocks = types.SimpleNamespace(layers = [session.model.layers[layer] for layer in layers])
    part = types.SimpleNamespace(model = blocks, model_arguments = session.model_arguments,
                                 reference = session.reference)
    prompt = unit_13.expected_values(part, prompt_tokens)
    later, earlier = (stepwise_expected_values(session, prompt_tokens + steps),
                      stepwise_expected_values(session, prompt_tokens))
    return {kind: prompt[kind] + later[kind] - earlier[kind] for kind in prompt}


def slab_check(arguments, session, switch, run, token_ids, label, prompt, candidate_blocks):
    """slab_arms over a prompt of `prompt` tokens with every indexer's candidate stage cut to candidate_blocks blocks
    (None: as published; fewer blocks prune sooner), as published again after. Returns whether the check held as it
    must."""
    indexers, settings = unit_13b.layers_of(session)[1], unit_13b.read_settings(arguments.checkpoint)
    published = {layer: indexer.candidate_topk_blocks for layer, indexer in indexers.items()}
    settings["candidate_topk_blocks"] = candidate_blocks or settings["candidate_topk_blocks"]
    first_candidate = unit_13b.first_candidate_position(settings)
    try:
        for indexer in indexers.values():
            indexer.candidate_topk_blocks = candidate_blocks or indexer.candidate_topk_blocks
        return slab_arms(arguments, session, switch, run, token_ids, label, prompt,
                         0 if first_candidate is None else max(prompt - first_candidate, 0))
    finally:
        for layer, indexer in indexers.items():
            indexer.candidate_topk_blocks = published[layer]


@torch.inference_mode()
def slab_arms(arguments, session, switch, run, token_ids, label, prompt, pruning):
    """The arms of the module text over a prompt of `prompt` tokens, of which `pruning` rows the candidate stage
    prunes, and their comparison. Returns whether the check held as it must."""
    model, (_, indexers, _, _) = session.model, unit_13b.layers_of(session)
    input_ids = torch.tensor([token_ids], device = oracle.CARD_DEVICE)
    expected, reference, held = expected_values(session, prompt, range(len(model.layers)), SLAB_CHECK_STEPS), None, True
    for arm in ARMS if arguments.break_name == BREAK_SLAB else ARMS[:-1]:  # each compared as soon as it ends
        outputs, candidates, seconds = collections.defaultdict(list), [], {}
        slabs = None if arm == REFERENCE_ARM else context.SlabIndexers(session, arguments.slab_rows, (),
                                                                       arm == BROKEN_ARM)
        hooks = context.arm_recorders(session, outputs, candidates) + context.block_timer(session, seconds)
        kept, handle = head_rows(session)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with u16_state(session, switch, run, True):
                if slabs is not None:
                    slabs.install()
                try:
                    with unit_13b.deadline(arguments.pass_timeout_seconds):
                        logits = [model.forward(input_ids[:, :prompt], 0)[1].cpu()]
                    rows, prompt_seconds = kept["rows"].cpu(), context.seconds_text(seconds)
                    for position in range(prompt, prompt + SLAB_CHECK_STEPS):
                        with unit_13b.deadline(arguments.step_timeout_seconds):
                            logits.append(model.forward(input_ids[:, position:position + 1], position)[1].cpu())
                    buffers = {name: buffer.cpu() for name, buffer in model.named_buffers()
                               if name.split(".")[-1] not in context.CONSTANT_BUFFERS}
                    checks = switch_checks(session, switch, run, expected, f"{label} arm={arm}", None)
                finally:
                    if slabs is not None:
                        slabs.remove()
        finally:
            handle.remove()
            for hook in hooks:
                hook.remove()
        held &= all(checks.values())
        print(f"ARM {label} arm={arm} prompt_tokens={prompt} steps={SLAB_CHECK_STEPS} slab_rows="
              f"{arguments.slab_rows if slabs else 'none'} seconds={time.perf_counter() - started:.1f} "
              f"card_gib={torch.cuda.max_memory_allocated() / oracle.BYTES_PER_GIB:.2f} block_seconds={prompt_seconds}")
        torch.cuda.empty_cache()
        if reference is None:
            reference = (outputs, candidates, rows, torch.cat(logits), buffers)
            continue
        counts = context.arm_differences(reference, (outputs, candidates, rows, torch.cat(logits), buffers),
                                         sorted(indexers))
        for name, count in counts.items():
            print(f"SLAB {label} arm={arm} record={name} differing={count}")
        same, check = not any(counts.values()), "slab" if arm == SLABS_ARM else "slab_break"
        verdict = ("ok" if same else "FAILED") if arm == SLABS_ARM else ("FAILED" if same else "reported")
        print(f"CHECK {label} name={check} pruning_rows={pruning} verdict={verdict}")
        held &= verdict != "FAILED"
    return held


@torch.inference_mode()
def long(arguments, session, switch, run, token_ids, writer, label):
    """The long-context reference, a part of it or its twin (the module text). Returns its checks."""
    model, sizes, shared = session.model, session.model_arguments, session.reference.model.shared_attn
    prompt, steps, flagged, (first, last) = (arguments.prompt_tokens, arguments.teacher_forced, arguments.flagged_rows,
                                             arguments.layer_range)
    final, sampled = last == len(model.layers) - 1, context.sampled_positions(sizes, prompt, flagged)
    identity, (_, indexers, owners, _) = context.identity_of(arguments, session, token_ids, sampled, RUN_LABEL), \
        unit_13b.layers_of(session)
    if writer is not None:
        writer.manifest["conditions"]["identity"] = identity           # what twin_floor matches
    if arguments.proof_tokens and not slab_check(arguments, session, switch, run, token_ids, f"{label} proof=yes",
                                                 arguments.proof_tokens, arguments.proof_candidate_blocks):
        return {"proof": False}                                            # the switch did not hold: nothing is run
    slabs = context.SlabIndexers(session, arguments.slab_rows, sampled, False)
    input_ids, scored, keys, rows, earlier = torch.tensor([token_ids], device = oracle.CARD_DEVICE), {}, [], [], {}
    kept, handle = head_rows(session) if final else ({}, None)
    hooks, seconds = [indexer.register_forward_hook(lambda module, inputs, output, layer = layer: scored.__setitem__(
        layer, next((owner for owner, other in owners.items() if other.k_cache is shared.index_k), "none")))
        for layer, indexer in indexers.items()], {}
    hooks += context.block_timer(session, seconds, label)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with u16_state(session, switch, run, True):
            restored, earlier = context.restore_parts(arguments.part_directory, first, session, identity) \
                if first else (None, {})
            slabs.install()
            try:
                with context.layer_range(session, first, last, restored) as left, \
                        unit_13b.deadline(arguments.pass_timeout_seconds):
                    own = model.forward(input_ids[:, :prompt], 0)[1][0]
                restored, prompt_seconds, block_seconds = None, time.perf_counter() - started, \
                    context.seconds_text(seconds)
                if final:
                    rows.append(model.head.forward(kept["rows"][None, prompt - flagged:], full_logits = True)[0])
                    last_row_same = oracle.same_bits(rows[0][-1], own)
                    for position in range(prompt, prompt + steps):
                        with unit_13b.deadline(arguments.step_timeout_seconds):
                            rows.append(model.forward(input_ids[:, position:position + 1], position)[1])
                        keys.append(dict(scored))
                checks = switch_checks(session, switch, run, expected_values(session, prompt, range(first, last + 1),
                                                                             steps if final else 0), label, None)
                if not final:
                    context.save_part(context.part_folder(arguments.part_directory, first, last), session, first,
                                      last, left, {layer: slabs.noted(layer) for layer in indexers
                                                   if first <= layer <= last}, identity,
                                      {"checks": checks, "prompt_seconds": prompt_seconds}, all(checks.values()))
            finally:
                slabs.remove()
    finally:
        for hook in hooks + ([] if handle is None else [handle]):
            hook.remove()
    print(f"LONG {label} layers={first}..{last} prompt_tokens={prompt} steps={steps if final else 0} "
          f"slab_rows={arguments.slab_rows} prompt_seconds={prompt_seconds:.1f} seconds="
          f"{time.perf_counter() - started:.1f} card_gib={torch.cuda.max_memory_allocated() / oracle.BYTES_PER_GIB:.2f}"
          + (f" last_row_same_bits={unit_13.NO_OR_YES[last_row_same]}" if final else "")
          + f" block_seconds={block_seconds}")
    if not final:
        return checks
    positions = numpy.concatenate((sampled, numpy.arange(prompt, prompt + steps)))
    noted = {layer: tuple(torch.cat(parts) for parts in zip(*(([earlier[layer]] if layer < first else [])
                                                                 + [slabs.noted(layer)]))) for layer in indexers}
    wrong = sum(step_keys.get(layer) != owner for step_keys, wanted in zip(
        keys, unit_13b.keys_expected(indexers, owners, prompt, steps, True)) for layer, owner in wanted.items())
    logits = torch.cat([row.float().cpu() for row in rows])
    checks.update(notes = all(numpy.array_equal(noted[layer][0].numpy(), positions)
                              and all(len(values) == len(positions) for values in noted[layer][1:]) for layer in noted),
                  decode_keys = wrong == 0 and len(keys) == steps, finite = bool(torch.isfinite(logits).all()))
    for name in ("notes", "decode_keys", "finite"):
        print(f"CHECK {label} name={name} verdict={verdict_text(name, checks[name], None)}")
    writer.add(gates.TOKEN_IDS, torch.tensor([token_ids]), None, "the prompt, the tokens fed, the token after them")
    writer.add(gates.FLAGGED_POSITIONS, torch.arange(prompt - flagged, prompt + steps), None, "the flagged positions")
    writer.add(gates.FLAGGED_ROWS, logits[None], None, "the prompt's last rows (one head call), then each step's")
    writer.add(gates.SAMPLED_POSITIONS, torch.from_numpy(positions), None, "the noted rows: sampled, then the steps")
    for layer in sorted(noted):
        for record, values in zip(gates.INDEX_NOTE_RECORDS, noted[layer][1:]):
            writer.add(gates.INDEX_NOTE.format(layer = layer, record = record), values, layer, NOTE_TEXT[record])
    writer.results.update(prompt_tokens = prompt, teacher_forced = steps, flagged_rows = flagged,
                          layers_here = [first, last], proof_tokens = arguments.proof_tokens, checks = checks,
                          prompt_seconds = prompt_seconds, seconds = time.perf_counter() - started,
                          last_row_of_the_head_call_has_the_forward_s_bits = last_row_same)
    return checks


def conditions_of(arguments, session, tokens):
    return {"capture": arguments.mode, "mode": arguments.mode, "run": RUN_LABEL, "keys": "published per owner",
            "command_line": sys.argv, "arguments": dict(sorted(vars(arguments).items())),
            "torch_version": torch.__version__, "card": torch.cuda.get_device_name(arguments.card),
            "default_dtype": str(torch.get_default_dtype()), "tokens": tokens,
            "reference_files_sha256": session.reference.file_digests,
            "model_arguments": dataclasses.asdict(session.model_arguments), "load": session.load_summary,
            "capture_layers": recording.CAPTURE_LAYERS, "capture_positions": recording.CAPTURE_POSITIONS,
            "capture_generated_steps": recording.CAPTURE_GENERATED_STEPS,
            "snapshot_positions": recording.SNAPSHOT_POSITIONS}


def closed(writer, complete):
    """writer.close, whose failure (a tensors.bin of another size than recorded) is printed, not raised, so that a
    failure before it stays the one reported. Returns whether it closed."""
    try:
        writer.close(complete)
        return True
    except Exception as failure:
        print(f"stepwise_capture FAILED: closing {writer.tensors_path}: {type(failure).__name__}: {failure}",
              file = sys.stderr)
        return False


def request_entries(arguments):
    """The entries of a requests file, (name, token ids) each, read before anything is loaded. Raises ValueError for
    none, a name that is no folder's own (the module text) or is given twice, a length out of range, a folder there."""
    with open(arguments.tokens_json, encoding = "utf8") as handle:
        count = len(json.load(handle))
    entries = [unit_13b.read_entry(arguments.tokens_json, sequence) for sequence in range(count)]
    names = [name for name, _ in entries]
    for name, token_ids in entries:
        if not REQUEST_NAME.fullmatch(name) or name == oracle.TOKENIZER_FOLDER_NAME or names.count(name) != 1 \
                or not 1 < len(token_ids) <= REQUESTS_LONGEST_TOKENS \
                or len(token_ids) + arguments.new_tokens > arguments.maximum_sequence_tokens \
                or os.path.exists(os.path.join(arguments.output_directory, name)):
            raise ValueError(f"entry {name!r}: a name that is no folder's own or is given twice, a length out of "
                             "range, or its folder exists")
    if not entries:
        raise ValueError("no entry")
    return entries


def requests(arguments, session, switch, run, entries, label):
    """Every entry of request_entries, one folder each. Returns whether every check of every entry held."""
    vocabulary, held = session.model_arguments.vocab_size, True
    if not all(0 <= token < vocabulary for _, token_ids in entries for token in token_ids):
        raise RuntimeError(f"an entry holds an id outside the vocabulary of {vocabulary}")
    for name, token_ids in entries:
        writer, checks = oracle.CaptureWriter(os.path.join(arguments.output_directory, name)), {}
        try:
            writer.manifest["conditions"] = {**conditions_of(arguments, session, len(token_ids)), "request": name}
            checks = request(arguments, session, switch, run, token_ids, writer, f"{label} request={name}")
        finally:
            written = closed(writer, bool(checks) and all(checks.values()))
        held &= written and bool(checks) and all(checks.values())
    return held


def check_arguments(parser, arguments):
    """Refuses, before anything is loaded, what can be refused then (the module text). Returns the entries of a
    requests file (request_entries), else None."""
    allowed = {CHECK: set(), STEPWISE: {"prompt_json", "new_tokens", "all_logits", "frequencies", "output_directory"},
               FLOOR: {"stepwise_capture", "floor_gguf", "output_json"}, REQUESTS: {"new_tokens", "output_directory"},
               SLAB_CHECK: {"prompt_tokens", "slab_rows", "candidate_blocks"},
               LONG: {"prompt_tokens", "teacher_forced", "flagged_rows", "slab_rows", "layers", "part_directory",
                      "output_directory", "floor_gguf", "reference_capture", "output_json", "proof_tokens",
                      "proof_candidate_blocks"}}
    needed = {CHECK: set(), STEPWISE: {"output_directory"}, FLOOR: allowed[FLOOR],
              REQUESTS: {"new_tokens", "output_directory"}, SLAB_CHECK: {"prompt_tokens"}, LONG: {"prompt_tokens"}}
    given = {name for name in set().union(*allowed.values()) if getattr(arguments, name)}
    mode, timeouts = arguments.mode, (arguments.step_timeout_seconds, arguments.pass_timeout_seconds)
    if not needed[mode] <= given <= allowed[mode] \
            or (arguments.tokens_json is None) == (arguments.prompt_json is None) \
            or (arguments.break_name is not None and BREAKS[arguments.break_name] != mode) \
            or (mode == REQUESTS and (arguments.tokens is not None or arguments.sequence != 0)) \
            or (mode in (SLAB_CHECK, LONG) and (arguments.tokens is not None or arguments.tokens_json is None)) \
            or arguments.sequence < 0 or arguments.new_tokens < 0 \
            or (arguments.tokens is not None and arguments.tokens < 1) \
            or not all(0 < timeout <= unit_13b.PASS_TIMEOUT_MOST_SECONDS for timeout in timeouts):
        parser.error(f"--mode {mode} takes {', '.join(sorted(allowed[mode])) or 'none'} of the mode options and needs "
                     f"{', '.join(sorted(needed[mode])) or 'none'}; exactly one of --tokens-json and --prompt-json; a "
                     "break of its own mode; requests takes neither --tokens nor --sequence, slab_check and long take "
                     "--prompt-tokens and --tokens-json; counts and timeouts in range (the module text)")
    if mode == STEPWISE and any(os.path.exists(os.path.join(arguments.output_directory, name))
                                for name in (oracle.TENSORS_FILE_NAME, oracle.MANIFEST_FILE_NAME)):
        parser.error(f"--output-directory holds a capture already: {arguments.output_directory}")
    if mode == FLOOR and (not os.path.isfile(arguments.floor_gguf) or not os.path.isdir(arguments.stepwise_capture)
                          or os.path.exists(arguments.output_json)
                          or not os.path.isdir(os.path.dirname(arguments.output_json))):
        parser.error("--floor-gguf names no file, --stepwise-capture no folder, or --output-json exists already")
    if mode != REQUESTS:
        return None
    try:
        return request_entries(arguments)
    except (OSError, ValueError, TypeError) as failure:                    # a file of another shape included
        parser.error(f"--tokens-json {arguments.tokens_json}: {type(failure).__name__}: {failure}")


def long_arguments(parser, arguments):
    """The options of slab_check and long checked and completed before anything is loaded (the module text), and
    the FIT line. Returns the exit code of a pass the card does not hold, else None."""
    try:
        settings = unit_13b.read_settings(arguments.checkpoint)
    except (ValueError, OSError) as failure:
        parser.error(str(failure))
    layers, slab_check_mode = settings["n_layers"], arguments.mode == SLAB_CHECK
    match = LAYER_RANGE.fullmatch(arguments.layers or f"0..{layers - 1}")
    first, last = (int(match["first"]), int(match["last"])) if match else (layers, 0)
    def given(value, default):                                             # 0 is refused below, never the default
        return default if value is None else value
    prompt, slab = arguments.prompt_tokens, given(arguments.slab_rows, context.DEFAULT_SLAB_ROWS)
    steps = SLAB_CHECK_STEPS if slab_check_mode else given(arguments.teacher_forced, DEFAULT_TEACHER_FORCED)
    flagged, final = given(arguments.flagged_rows, DEFAULT_FLAGGED_ROWS), last == layers - 1
    folder, twin = arguments.part_directory, arguments.floor_gguf is not None
    floor_given = {twin and final, arguments.reference_capture is not None, arguments.output_json is not None}
    cut, proof_prunes = arguments.proof_candidate_blocks, arguments.proof_candidate_blocks is None
    if cut is not None and cut > 0 and arguments.proof_tokens is not None:  # a cut stage must prune in the proof
        first_candidate = unit_13b.first_candidate_position(dict(settings, candidate_topk_blocks = cut))
        proof_prunes = first_candidate is not None and first_candidate < arguments.proof_tokens
    if not (first <= last < layers and 0 < flagged <= prompt and min(steps, slab) > 0 and proof_prunes
            and 0 < given(arguments.proof_tokens, 1) < prompt and given(arguments.candidate_blocks, 1) > 0) \
            or (folder is not None) != (first > 0 or not final) or (folder is not None and not os.path.isdir(folder)) \
            or (not final and os.path.exists(context.part_folder(folder, first, last))) \
            or (final and not slab_check_mode) != (arguments.output_directory is not None) \
            or len(floor_given) > 1 or (twin and not os.path.isfile(arguments.floor_gguf)) \
            or (arguments.reference_capture is not None and not os.path.isdir(arguments.reference_capture)) \
            or (arguments.output_json is not None and (os.path.exists(arguments.output_json)
                                                       or not os.path.isdir(os.path.dirname(arguments.output_json)))) \
            or (arguments.output_directory is not None and any(os.path.exists(os.path.join(
                arguments.output_directory, name)) for name in (oracle.TENSORS_FILE_NAME, oracle.MANIFEST_FILE_NAME))):
        parser.error(f"--layers of 0..{layers - 1}; --flagged-rows within the prompt; positive counts; "
                     "--part-directory (a folder without this part) where a part is not the whole; --output-directory "
                     "(no capture) where it ends the model; the twin's --floor-gguf, there with --reference-capture "
                     "and --output-json; --proof-candidate-blocks with --proof-tokens, a stage that prunes within them")
    arguments.layer_range, arguments.slab_rows, arguments.flagged_rows = (first, last), slab, flagged
    arguments.teacher_forced, arguments.final_part = steps, final and not slab_check_mode
    arguments.tokens = arguments.maximum_sequence_tokens = prompt + steps + (not slab_check_mode)
    if not torch.cuda.is_available() or arguments.card >= torch.cuda.device_count():
        parser.error(f"there is no card {arguments.card}: the fit is read from the card")
    fits = context.long_fit(arguments, settings, slab_check_mode, RUN_LABEL,
                            arguments.proof_tokens + SLAB_CHECK_STEPS if arguments.proof_tokens else 0)
    return None if fits else unit_13b.EXIT_CODE_DOES_NOT_FIT


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev = False)
    parser.add_argument("--checkpoint", required = True, type = os.path.abspath, help = "the Hugging Face folder")
    parser.add_argument("--attention-card", required = True, help = "CUDA_VISIBLE_DEVICES value of a CMP 170HX")
    parser.add_argument("--mode", required = True, choices = MODES)
    parser.add_argument("--tokens-json", default = None, type = os.path.abspath, help = "a list of token id lists")
    parser.add_argument("--prompt-json", default = None, type = os.path.abspath, help = "stepwise: as generate.py")
    parser.add_argument("--sequence", type = int, default = 0, help = "which entry of the JSON file")
    parser.add_argument("--thinking-mode", choices = ("chat", "thinking"), default = oracle.DEFAULT_THINKING_MODE)
    parser.add_argument("--tokens", type = int, default = None, help = "its first N tokens")
    parser.add_argument("--new-tokens", type = int, default = 0, help = "greedy tokens after the given ones")
    parser.add_argument("--all-logits", action = "store_true", help = "stepwise: the logits of every step")
    parser.add_argument("--frequencies", action = "store_true", help = "stepwise: the words and tables of both thetas")
    parser.add_argument("--output-directory", default = None, type = os.path.abspath, help = "the capture's folder")
    parser.add_argument("--stepwise-capture", default = None, type = os.path.abspath, help = "floor: of the text")
    parser.add_argument("--floor-gguf", default = None, type = os.path.abspath, help = "floor: the twin's GGUF")
    parser.add_argument("--output-json", default = None, type = os.path.abspath, help = "floor, twin: receives it")
    parser.add_argument("--prompt-tokens", type = int, default = None, help = "slab_check, long: the prompt's length")
    parser.add_argument("--teacher-forced", type = int, default = None, help = "long: decode steps fed the text")
    parser.add_argument("--flagged-rows", type = int, default = None, help = "long: the prompt's last rows' logits")
    parser.add_argument("--slab-rows", type = int, default = None, help = "slab_check, long: rows a slab of scores")
    parser.add_argument("--proof-tokens", type = int, default = None, help = "long: slab_check over so many first")
    parser.add_argument("--candidate-blocks", type = int, default = None, help = "slab_check: the stage's blocks")
    parser.add_argument("--proof-candidate-blocks", type = int, default = None, help = "long: the proof's stage")
    parser.add_argument("--layers", default = None, help = "long: first..last, the part's layers")
    parser.add_argument("--part-directory", default = None, type = os.path.abspath, help = "long: the parts' folder")
    parser.add_argument("--reference-capture", default = None, type = os.path.abspath, help = "twin: the u16 capture")
    parser.add_argument("--step-timeout-seconds", type = float, default = DEFAULT_STEP_TIMEOUT_SECONDS)
    parser.add_argument("--pass-timeout-seconds", type = float, default = DEFAULT_PASS_TIMEOUT_SECONDS)
    parser.add_argument("--break", dest = "break_name", choices = tuple(BREAKS), default = None)
    parser.add_argument("--traceback", action = "store_true", help = "print the Python traceback of a failure")
    arguments = parser.parse_args(namespace = argparse.Namespace(**unit_13.ORACLE_SETTINGS, self_test = False))
    entries = check_arguments(parser, arguments)
    sys.stdout.reconfigure(line_buffering = True)                          # a step that is cut short keeps its lines
    refused = long_arguments(parser, arguments) if arguments.mode in (SLAB_CHECK, LONG) else None
    if refused is not None:
        return refused
    exit_code, writer, held = oracle.EXIT_CODE_FAILURE, None, False
    label = f"tool=stepwise_capture mode={arguments.mode}"
    try:
        text = "prompt" if arguments.tokens_json is None or arguments.mode == REQUESTS \
            else unit_13b.read_entry(arguments.tokens_json, arguments.sequence)[0]
        label = f"{label} text={text}"
        with tempfile.TemporaryDirectory() as folder:                      # open_session copies the tokenizer there
            if arguments.mode in (CHECK, FLOOR, SLAB_CHECK) or (arguments.mode == LONG
                                                                and not arguments.final_part):
                arguments.output_directory = folder
            if arguments.mode == STEPWISE or (arguments.mode == LONG and arguments.final_part):   # refuses a capture
                writer = oracle.CaptureWriter(arguments.output_directory)                        # there, unloaded
            session, token_ids = oracle.open_session(arguments, {})
            if len(token_ids) + arguments.new_tokens > arguments.maximum_sequence_tokens:
                raise RuntimeError("the tokens and the new tokens do not fit --maximum-sequence-tokens")
            run = unit_13b.parse_runs(RUN_LABEL)[0]
            switch = recording.CaptureSwitch(session.reference.model, arguments.break_name)
            if arguments.mode == CHECK:
                held = check(arguments, session, switch, run, token_ids, label)
            elif arguments.mode == STEPWISE:
                writer.manifest["conditions"] = conditions_of(arguments, session, len(token_ids))
                checks = stepwise(arguments, session, switch, run, token_ids, writer, label)
                held = broken_as_asked(checks, arguments.break_name) if arguments.break_name else all(checks.values())
            elif arguments.mode == FLOOR:
                held = floor(arguments, session, switch, run, token_ids, label)
            elif arguments.mode == SLAB_CHECK:
                held = slab_check(arguments, session, switch, run, token_ids, label, arguments.prompt_tokens,
                                  arguments.candidate_blocks)
            elif arguments.mode == LONG:
                if arguments.floor_gguf is not None:                       # the twin, as floor installs it: with
                    torch.cuda.empty_cache()                               # autograd on, each float32 matrix kept
                    with torch.no_grad():                                  # its two factors (41 GiB more)
                        importlib.import_module("gguf_weights").install(
                            session.reference.model, session.model, session.stages, oracle.HostResidentExpert,
                            oracle.HostResidentTable, arguments.floor_gguf,
                            importlib.import_module("gguf_weights").INPUTS_AS_THE_REFERENCE, ())
                if writer is not None:
                    writer.manifest["conditions"] = conditions_of(arguments, session, len(token_ids))
                held = all(long(arguments, session, switch, run, token_ids, writer, label).values())
            else:
                held = requests(arguments, session, switch, run, entries, label)
            exit_code = oracle.EXIT_CODE_SUCCESS if held else oracle.EXIT_CODE_FAILURE
    except Exception as failure:
        if arguments.traceback:
            traceback.print_exc()
        print(f"stepwise_capture FAILED: {type(failure).__name__}: {failure}", file = sys.stderr)
    finally:                                                               # a capture of a break is never complete
        if writer is not None and not closed(writer, held and arguments.break_name is None):
            exit_code = oracle.EXIT_CODE_FAILURE
    if exit_code == oracle.EXIT_CODE_SUCCESS and arguments.mode == LONG and arguments.reference_capture is not None:
        try:
            context.twin_floor(arguments.reference_capture, arguments.output_directory, arguments.output_json,
                               label, LONG)
        except Exception as failure:                                       # the capture stays; no floor is written
            exit_code = oracle.EXIT_CODE_FAILURE
            print(f"stepwise_capture FAILED: the twin's floor: {type(failure).__name__}: {failure}", file = sys.stderr)
    print(f"VERDICT {label} break={arguments.break_name or 'none'} "
          f"verdict={'ok' if exit_code == oracle.EXIT_CODE_SUCCESS else 'FAILED'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
