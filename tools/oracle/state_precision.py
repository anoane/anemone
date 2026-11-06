#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the precision of the attention state changes in DeepSeek's reference on a context long enough for its index
to select, and whose index keys a layer scores on a decode step in which it emits none (speed campaign, unit 13).

The reference quantizes four kinds of attention state in place (model.py): the index KEYS (:546) and QUERIES
(:552) to 4-bit floats per 32 values, the WINDOW rows (:707) to 8-bit floats per 32, the POOL entries (:760) to
4-bit floats per 16; Anemone keeps all four in half precision. Through reference_oracle.py's session, which holds
every quantization that runs to its definition, this tool passes the first --tokens tokens of one sequence from
position 0: as the reference is (the baseline), then once per run of --runs. A run names the kinds it leaves as
computed, in bfloat16, joined by +; "reference" leaves none: the reference against itself, the floor.

    state_precision.py --checkpoint CHECKPOINT --tokens-json FILE --attention-card N [--sequence 0]
                       [--tokens 8192] [--runs reference,keys,...,keys+queries+window+pool] [--break switch]

Printed, a run against the baseline from position 2,048 on (from there every index layer picks 512 of more):
  RUN       per pass: seconds, the card's peak memory, per kind the values left out / the values asked to quantize;
  CHECK     name=switch: those counts are what the model's structure gives, for the run's kinds and no other, or
            the tool ends. --break switch books every call under the next kind's name: exit 0 only if it reported;
  SELECTED  per index layer, of the baseline's picks: the share the run picked too (mean and least over the rows)
            and the share of rows with exactly the baseline's picks;
  FIGURE    by compare_logit_captures.py's own function, over the positions with a next token: mean KL(baseline ||
            run) in nats, the share with the same most likely token; and whether the head's input has the same bits;
  SLOT      after the baseline, a decode step at each of the next two positions on the reference's own tokens: per
            index layer, whether it emitted index keys and which layer's key cache it scored (:537, :548, :554).
Refused: a run that is not "reference" or kinds joined by +, or given twice; a --sequence below 0; --tokens that
leave no scored position or the decode steps no room; what reference_oracle.py refuses of a tokens file.
The attention output, which model.py rounds in place too from 46cc8310 on (:787), is no state: the switch rounds it in
every run, counts its values apart and holds them to the count the reference's form gives (reference_oracle.py).
Not shown: Anemone's own state (a kind left out keeps the reference's bfloat16 values of 8 significant bits; the
engine's half-precision rows hold 11); candidate blocks (layers 24 to 36), idle below 16,385 tokens; decode steps.
Measurement tool: not part of the engine.
"""
import argparse
import functools
import os
import sys
import tempfile

import numpy
import torch

import compare_logit_captures as compare
import reference_oracle as oracle

BATCH, BREAK_NAMES = "i13", ("switch",)
KEYS, QUERIES, WINDOW, POOL = KINDS = ("keys", "queries", "window", "pool")
REFERENCE, BASELINE = "reference", "baseline"
DEFAULT_RUNS = "reference,keys,queries,window,pool,keys+queries,window+pool,keys+queries+window+pool"
KEY_DIMENSIONS = 3                           # model.py:544: [sequence, entry, value]; a query has a head axis more
OUTPUT, OUTPUT_DIMENSIONS = "output", 4      # the attention output: [batch, sequence, group, value] (46cc8310:786)
LATENT_ARGUMENT = 2                          # model.py:527: forward(x, qr, latent, start_pos, offset)
DEFAULT_TOKENS, FIRST_POSITION = 8192, 2048
SLOT_STEPS, NO_OR_YES = 2, ("no", "yes")     # the steps: consecutive positions, one even and one odd
ORACLE_SETTINGS = dict(card = 0, modes = (), weights_gguf = None,     # what open_session reads of a command line
                       maximum_sequence_tokens = oracle.DEFAULT_MAXIMUM_SEQUENCE_TOKENS)     # beside this tool's own


class Switch:
    """Under model.py's two names, before the oracle's stand-ins: an in-place call of a kind left out is skipped."""

    def __init__(self, reference_model, broken):
        self.checked_act_quant, self.checked_fp4_act_quant = reference_model.act_quant, reference_model.fp4_act_quant
        reference_model.act_quant, reference_model.fp4_act_quant = self.act_quant, self.fp4_act_quant
        self.broken = broken

    def begin(self, kinds):
        self.kinds, self.asked, self.left = kinds, dict.fromkeys(KINDS, 0), dict.fromkeys(KINDS, 0)
        self.outputs = 0                                                   # values of the attention output rounded

    def in_place(self, kind, values, quantize):
        kind = KINDS[(KINDS.index(kind) + self.broken) % len(KINDS)]       # --break switch: the next kind's name
        self.asked[kind] += values.numel()
        if kind not in self.kinds:
            return quantize()
        self.left[kind] += values.numel()
        return values                                                      # as kernel.py:123 and :203 return it

    def act_quant(self, x, block_size, scale_fmt, scale_dtype, inplace = False):       # model.py:707; :187, :197
        quantize = functools.partial(self.checked_act_quant, x, block_size, scale_fmt, scale_dtype, inplace)
        if inplace and x.dim() == OUTPUT_DIMENSIONS:                       # the attention output, never left out
            return self.output_rounded(x, quantize)
        return self.in_place(WINDOW, x, quantize) if inplace else quantize()

    def output_rounded(self, values, quantize):
        self.outputs += values.numel()
        return quantize()

    def fp4_act_quant(self, x, block_size, inplace, scale_dtype = torch.float8_e8m0fnu):   # :546, :552; :760
        kind = POOL if scale_dtype == torch.float8_e4m3fn else KEYS if x.dim() == KEY_DIMENSIONS else QUERIES
        quantize = functools.partial(self.checked_fp4_act_quant, x, block_size, inplace, scale_dtype)
        return self.in_place(kind, x, quantize)


def expected_values(session, tokens):
    """Per kind, the values a pass over `tokens` tokens from position 0 hands to the in-place quantizations."""
    sizes, attentions = session.model_arguments, [block.attn for block in session.model.layers]
    indexers = [attention.indexer for attention in attentions if attention.is_index_source]
    key_rows = [tokens // indexer.compress_ratio for indexer in indexers if indexer.owns_k]
    pool_rows = [tokens // attention.compress_ratio for attention in attentions if attention.is_kv_source]
    rounds = oracle.reference_form(session.reference).rounds_attention_output
    return {KEYS: sum(key_rows) * sizes.index_head_dim,                    # model.py:544, :550, :705, :747
            QUERIES: len(indexers) * tokens * sizes.index_n_heads * sizes.index_head_dim,
            WINDOW: len(attentions) * tokens * sizes.head_dim, POOL: sum(pool_rows) * sizes.head_dim,
            OUTPUT: len(attentions) * tokens * sizes.n_heads * sizes.head_dim if rounds else 0}


def switch_held(switch, expected):
    """Whether a pass asked each kind for its expected values and left out those of the pass's kinds, and no other,
    and rounded the attention output's expected values."""
    return switch.outputs == expected[OUTPUT] and all(
        (switch.asked[kind], switch.left[kind]) == (expected[kind], expected[kind] * (kind in switch.kinds))
        for kind in KINDS)


def shared_picks(baseline_picks, picks):
    """Per row, the share of a run's picks the baseline picked too: a row's picks are distinct, so what is twice in
    both together is in both."""
    both = torch.cat((baseline_picks, picks), dim = -1).sort(dim = -1).values
    return ((both[:, 1:] == both[:, :-1]) & (both[:, 1:] >= 0)).sum(dim = -1).double() / picks.size(-1)


@torch.inference_mode()                                                    # as the reference's own forward
def measure(arguments, session, token_ids):
    """Prints every line of the module text. Returns whether the run ended as its command line requires."""
    model, tokens, last, head = session.model, len(token_ids), {}, {}
    shared, attentions = session.reference.model.shared_attn, [block.attn for block in model.layers]
    indexers = {attention.layer_id: attention.indexer for attention in attentions if attention.is_index_source}
    expected = expected_values(session, tokens)

    def indexer_ran(layer, indexer, inputs, picks):        # its picks, whether it emitted keys, whose keys it scored
        scored = [owner for owner, other in indexers.items() if other.owns_k and other.k_cache is shared.index_k]
        last[layer] = (picks[0], inputs[LATENT_ARGUMENT] is not None, scored[0] if scored else "none")

    for layer, indexer in indexers.items():                # hooks on the reference's own modules, left in place
        indexer.register_forward_hook(functools.partial(indexer_ran, layer))
    model.head.register_forward_pre_hook(lambda _, inputs: head.update(rows = inputs[0][0]))
    switch = Switch(session.reference.model, arguments.break_name is not None)
    input_ids = torch.tensor([token_ids], dtype = torch.long, device = oracle.CARD_DEVICE)
    for run in (BASELINE, *arguments.runs):
        label = f"batch={BATCH} sequence={arguments.sequence} run={run}"
        oracle.forget_earlier_runs(session)
        switch.begin(frozenset(run.split("+")) & set(KINDS))
        torch.cuda.reset_peak_memory_stats()
        started = oracle.synchronized_clock()
        next_ids = model.forward(input_ids, 0)[0]                          # generate.py:69-74, the prompt's call
        print(f"RUN {label} tokens={tokens} seconds={oracle.synchronized_clock() - started:.1f} "
              f"card_gib={torch.cuda.max_memory_allocated() / oracle.BYTES_PER_GIB:.2f} "
              + " ".join(f"{kind}={switch.left[kind]}/{switch.asked[kind]}" for kind in KINDS))
        held = switch_held(switch, expected)
        print(f"CHECK {label} name=switch verdict={'ok' if held else 'reported' if switch.broken else 'FAILED'}")
        if switch.broken or not held:                                      # such a pass is not what its name says
            return switch.broken and not held
        picks, rows = {layer: record[0][FIRST_POSITION:] for layer, record in last.items()}, head["rows"]
        logits = model.head.forward(rows[None, FIRST_POSITION:], full_logits = True)[0].cpu().numpy()   # no hook
        if run == BASELINE:
            baseline_picks, baseline_rows, baseline_logits = picks, rows, logits
            for position in range(tokens, tokens + SLOT_STEPS):
                next_ids = model.forward(next_ids.view(oracle.SEQUENCES, 1), position)[0]     # generate.py:69-74
                for layer, (_, emitted, scored) in last.items():
                    print(f"SLOT {label} position={position} layer={layer} rate={indexers[layer].compress_ratio} "
                          f"emits={NO_OR_YES[emitted]} keys_of_layer={scored}")
            continue
        for layer, indexer in indexers.items():
            share = shared_picks(baseline_picks[layer], picks[layer])
            print(f"SELECTED {label} layer={layer} rate={indexer.compress_ratio} rows={len(share)} "
                  f"picks={picks[layer].size(-1)} mean_overlap={share.mean():.4f} least_overlap={share.min():.4f} "
                  f"same_rows={(share == 1).float().mean():.4f}")
        scores = compare.score_positions(baseline_logits, logits, numpy.asarray(token_ids[FIRST_POSITION:]))
        kl, same_top = scores["kl_baseline_other"], scores["same_top_token"]
        print(f"FIGURE {label} positions={len(kl)} mean_kl={kl.mean():.6f} same_top={same_top.mean():.4f} "
              f"same_bits={NO_OR_YES[oracle.same_bits(baseline_rows, rows)]}")
    print(f"VERDICT batch={BATCH} figure=state_precision verdict=measured")
    return True


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required = True, type = os.path.abspath, help = "the Hugging Face folder")
    parser.add_argument("--tokens-json", required = True, type = os.path.abspath, help = "of make_text_tokens.py")
    parser.add_argument("--attention-card", required = True, help = "CUDA_VISIBLE_DEVICES value of a CMP 170HX")
    parser.add_argument("--sequence", type = int, default = 0, help = "which entry of the tokens file")
    parser.add_argument("--tokens", type = int, default = DEFAULT_TOKENS, help = "its first N tokens")
    parser.add_argument("--runs", default = DEFAULT_RUNS, help = "separated by commas: reference, or kinds joined by +")
    parser.add_argument("--break", dest = "break_name", choices = BREAK_NAMES, default = None)
    arguments = parser.parse_args(namespace = argparse.Namespace(**ORACLE_SETTINGS))
    sys.stdout.reconfigure(line_buffering = True)                          # a step that is cut short keeps its lines
    arguments.runs, most_tokens = arguments.runs.split(","), arguments.maximum_sequence_tokens - SLOT_STEPS
    if len(set(arguments.runs)) != len(arguments.runs) or arguments.sequence < 0 \
            or not all(run == REFERENCE or set(run.split("+")) <= set(KINDS) for run in arguments.runs) \
            or not FIRST_POSITION + 1 < arguments.tokens <= most_tokens:   # one position with a next token at least
        parser.error(f"--runs takes each run once, {REFERENCE} or of {', '.join(KINDS)} joined by +; --sequence is "
                     f"0 or more; --tokens is above {FIRST_POSITION + 1} and at most {most_tokens}")
    with tempfile.TemporaryDirectory() as folder:                          # open_session copies the tokenizer there
        arguments.output_directory = folder
        session, token_ids = oracle.open_session(arguments, {})
        return oracle.EXIT_CODE_SUCCESS if measure(arguments, session, token_ids) else oracle.EXIT_CODE_FAILURE


if __name__ == "__main__":
    sys.exit(main())
