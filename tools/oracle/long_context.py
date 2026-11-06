#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The u16 prompt path of DeepSeek's reference at lengths whose index scores no card here holds whole, and in parts
across processes where one would run too long: what stepwise_capture.py --mode long and --mode slab_check run
(design/wide_prefill.md 8, gate G8, unit U24).

The reference forms each index layer's scores of a prompt as one tensor of rows x heads x entries, then its
weighted copy (model.py:556-557): in the u16 run's float32, 64 GiB a tensor for a layer of rate 2 at 32,768 tokens
and 128 GiB for one of rate 1, which no card holds (state_precision_true.py's FIT). SlabIndexers stands in for
every indexer's forward with model.py:527-580 as written, the scores of a prompt formed for a slab of rows at a
time: the same products, sums, masks, candidate blocks and selections per row, the candidates of the candidate
source kept for every row as model.py:569 keeps them. stepwise_capture.py --mode slab_check holds it to the
reference's own bits where the reference fits. Along the way it notes, at the sampled rows and at every decode
step, each index layer's picks, its 512th score (the cut), the entries scoring exactly the cut and how many of them
it picked: a row whose cut ties more entries than it picked chose among them by the kernel's order, and G8 counts
it apart.
Parts: the prompt path runs its layers in order over every row (model.py:1259-1267), so a part is a range of
layers. A part that does not end the model leaves in its folder (reference_oracle.py's capture format) the streams
leaving its last block with their carried weights, every state buffer of its blocks, what the shared runtime holds
(model.py:1166-1176: the selection the layers after an index source reuse, the candidate source's blocks, and whose
pool and keys the slot points at) and its notes; the next part takes them, with every earlier part's buffers, and
gives the streams to its first block (to its n-gram add where it has one), as stepwise_capture.py's layer floor
injects captured streams. Every record carries its byte sum, checked when it is read back; a part refuses the
folders of another run (the identity: tokens, lengths, slabs, sampled rows, the twin, the reference's files).
Measurement tool: not part of the engine.
"""
import contextlib
import functools
import hashlib
import json
import os
import re
import sys
import types

import numpy
import torch

import compare_engine_capture as gates
import reference_oracle as oracle
import state_precision_true as unit_13b
import stepwise_recorder as recording

DEFAULT_SLAB_ROWS = 256                      # 2 GiB of float32 scores a tensor at 65,536 entries: 8 GiB at 262,144
SPREAD_ROWS = 64                             # prompt rows sampled evenly before the flagged ones
STREAMS, CARRIED = "part.streams", "part.carried"
TOPK_IDXS, CANDIDATE_BLOCKS = "part.shared.topk_idxs", "part.shared.candidate_blocks"
BUFFER_RECORD = "layer_{layer}.buffer.{name}"
CONSTANT_BUFFERS = ("freqs_cis",)            # a table each attention builds once (model.py:690-698): never state
PART_NAME = re.compile(r"layers_(?P<first>\d+)_(?P<last>\d+)")
SLAB_CHECK_ROWS = 4096                       # the candidates compared and expanded a slab of rows at a time
BYTE_SUM = "byte_sum"
JSON_INDENT_SPACES = 1
NOTE_RECORDS = gates.INDEX_NOTE_RECORDS      # picks, cut_score, tied_at_cut, picked_at_cut
FIRST_BLOCK_PEAKS = ((32833, 31.82), (131137, 86.87))   # GiB allocated at most when block 0 of a u16 long
                                                       # pass ends (BLOCK lines of batches m2_long_reference_c
                                                       # and m2_long_part_0__19_0__19, snapshot 20261018-114118)


def sampled_positions(sizes, prompt_tokens, flagged_rows):
    """The prompt rows whose selections are noted: SPREAD_ROWS spread evenly from the first row at which every index
    layer picks fewer entries than it sees, then the flagged rows (the last flagged_rows of the prompt)."""
    rates = [rate for rate in sizes.compress_ratios[:sizes.n_layers] if rate]
    first = (sizes.index_topk + 1) * max(rates) - 1                        # (p + 1) // rate passes index_topk there
    spread = numpy.linspace(first, prompt_tokens - flagged_rows - 1, SPREAD_ROWS).round().astype(numpy.int64) \
        if prompt_tokens - flagged_rows > first else numpy.empty(0, numpy.int64)
    return numpy.unique(numpy.concatenate((spread, numpy.arange(max(prompt_tokens - flagged_rows, 0), prompt_tokens))))


class SlabIndexers:
    """model.py:527-580 in place of every indexer's forward (the module text), with the notes of the sampled rows.
    broken: each slab's rows of a prompt see one entry fewer than they do (slab_check's broken arm)."""

    def __init__(self, session, slab_rows, sampled, broken):
        if slab_rows < 1:
            raise RuntimeError(f"slabs of {slab_rows} rows")
        self.reference_model, self.slab_rows, self.broken = session.reference.model, slab_rows, broken
        self.publishes = oracle.reference_form(session.reference).publishes_index_keys
        self.sampled, self.indexers = numpy.asarray(sampled, numpy.int64), unit_13b.layers_of(session)[1]
        self.notes = {layer: [] for layer in self.indexers}

    def install(self):
        for layer, indexer in self.indexers.items():
            indexer.forward = functools.partial(self.forward, layer, indexer)

    def remove(self):
        for indexer in self.indexers.values():
            if "forward" in vars(indexer):
                del indexer.forward

    def forward(self, layer, indexer, x, qr, latent, start_pos, offset):
        """model.py:527-580, the scores of a prompt a slab of rows at a time; layer: the indexer's block's."""
        model, shared = self.reference_model, self.reference_model.shared_attn
        if model.world_size != 1 or indexer.freqs_cis is None or x.size(0) != oracle.SEQUENCES:
            raise RuntimeError("an indexer of several ranks or sequences, or before its attention gave it its table")
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = indexer.compress_ratio, indexer.rope_head_dim, start_pos + seqlen
        if indexer.owns_k and latent is not None:
            freqs = indexer.freqs_cis[: seqlen - seqlen % ratio : ratio] if start_pos == 0 \
                else indexer.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            k = indexer.k_norm(indexer.wk(latent))
            model.apply_rotary_emb(k[..., -rd:], freqs)
            model.fp4_act_quant(k, model.fp4_block_size, True)
            indexer.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        if indexer.owns_k and (latent is not None or self.publishes):     # 46cc8310:548-549 publishes at every step
            shared.index_k = indexer.k_cache
        q = indexer.wq_b(qr).unflatten(-1, (indexer.n_local_heads, indexer.index_head_dim))
        model.apply_rotary_emb(q[..., -rd:], indexer.freqs_cis[start_pos:end_pos])
        model.fp4_act_quant(q, model.fp4_block_size, True)
        index_k = shared.index_k[:bsz, : end_pos // ratio]
        weights = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads ** -0.5)
        visible = (torch.arange(1, seqlen + 1, device = x.device) // ratio).unsqueeze(-1) if start_pos == 0 \
            else end_pos // ratio
        entries, topk = end_pos // ratio, min(indexer.index_topk, end_pos // ratio)
        picks = torch.empty((bsz, seqlen, topk), dtype = torch.int64, device = x.device)
        candidates = torch.empty((bsz, seqlen, entries), dtype = torch.bool, device = x.device) \
            if indexer.is_candidate_source else None
        for first in range(0, seqlen, self.slab_rows):
            rows = slice(first, min(first + self.slab_rows, seqlen))
            score = torch.einsum("bshd,btd->bsht", q[:, rows], index_k)
            score = (score.relu_() * weights[:, rows].unsqueeze(-1)).sum(dim = 2)
            seen = visible
            if start_pos == 0:
                seen = visible[rows] - int(self.broken)
                score.masked_fill_(torch.arange(seqlen // ratio, device = x.device) >= seen, -torch.inf)
            if indexer.is_candidate_source:
                candidates[:, rows] = model.select_candidate_blocks(
                    score, seen, indexer.candidate_topk_blocks, indexer.candidate_block_size)
            elif indexer.uses_candidates:
                score = score.masked_fill(~shared.candidates[:, rows], -torch.inf)
            picks[:, rows] = score.topk(topk, dim = -1, sorted = False).indices.sort(dim = -1).values
            self.note(layer, start_pos or first, start_pos > 0, score[0], picks[0, rows], seen)
        if indexer.is_candidate_source:
            shared.candidates = candidates
        return torch.where(picks < visible, picks + offset, -1).int()

    def note(self, layer, first_position, decoding, score, picks, seen):
        """The notes (the module text) of the sampled rows of a prompt's slab from first_position, or of a decode
        step's row at it: [rows, entries] scores after every mask, [rows, topk] picks before the offset."""
        rows = numpy.zeros(1, numpy.int64) if decoding else self.sampled[
            (self.sampled >= first_position) & (self.sampled < first_position + len(picks))] - first_position
        if not len(rows):
            return
        local = torch.as_tensor(rows, device = score.device)
        row_scores, row_picks = score[local], picks[local]
        picked = row_scores.gather(-1, row_picks)
        cut = picked.amin(dim = -1)                                       # -inf where a row sees fewer than topk
        finite = torch.isfinite(cut)
        tied = torch.where(finite, (row_scores == cut.unsqueeze(-1)).sum(dim = -1), 0)
        picked_at_cut = torch.where(finite, (picked == cut.unsqueeze(-1)).sum(dim = -1), 0)
        limit = seen[local] if torch.is_tensor(seen) else seen
        self.notes[layer].append((torch.from_numpy(rows + first_position), torch.where(   # positions on the host
            row_picks < limit, row_picks, -1).int().cpu(), cut.float().cpu(), tied.int().cpu(),
            picked_at_cut.int().cpu()))

    def noted(self, layer):
        """A layer's notes in the order they were taken: positions, picks, cut scores, tied and picked at the cut."""
        notes = self.notes[layer]
        return tuple(torch.cat([note[part] for note in notes]) for part in range(len(NOTE_RECORDS) + 1)) \
            if notes else None


def identity_of(arguments, session, token_ids, sampled, run_label):
    """What every part of one long run of stepwise_capture.py shares, and its twin but for the twin's file."""
    return {"tokens_sha256": gates.tokens_sha256(token_ids), "prompt_tokens": arguments.prompt_tokens,
            "teacher_forced": arguments.teacher_forced, "flagged_rows": arguments.flagged_rows,
            "slab_rows": arguments.slab_rows, "run": run_label,
            "reference_files_sha256": session.reference.file_digests,
            "sampled_sha256": hashlib.sha256(numpy.asarray(sampled, numpy.int64).tobytes()).hexdigest(),
            "twin": None if arguments.floor_gguf is None else os.path.basename(arguments.floor_gguf)}


def state_buffers(block):
    """A block's buffers that hold state: every one but the tables it builds once."""
    return [(name, buffer) for name, buffer in block.named_buffers() if name.split(".")[-1] not in CONSTANT_BUFFERS]


@contextlib.contextmanager
def layer_range(session, first, last, restored):
    """model.layers cut to the blocks first..last for one forward (model.py:1259 runs what it holds), restored:
    None from layer 0, else the streams and carried weights given to block `first` (to its n-gram add where it has
    one, which runs before it: model.py:1263). Yields a dict that receives what a block before the model's last
    leaves: streams and carried."""
    model, blocks = session.model, session.model.layers
    left, handles = {}, []

    def block_begins(block, inputs):
        arguments = list(inputs)
        if block.engram is None:
            arguments[recording.BLOCK_STREAMS_ARGUMENT] = restored[0]
        arguments[recording.BLOCK_PRE_MIX_ARGUMENT] = restored[1]
        return tuple(arguments)

    try:
        if restored is not None:
            handles.append(blocks[first].register_forward_pre_hook(block_begins))
            if blocks[first].engram is not None:
                handles.append(blocks[first].engram.register_forward_pre_hook(
                    lambda engram, inputs: (restored[0], *inputs[1:])))
        if last < len(blocks) - 1:
            handles.append(blocks[last].register_forward_hook(
                lambda block, inputs, output: left.update(streams = output[0], carried = output[1])))
        model.layers = torch.nn.ModuleList(list(blocks)[first:last + 1])
        yield left
    finally:
        model.layers = blocks
        for handle in handles:
            handle.remove()


def candidate_blocks(candidates, block_size):
    """The candidate mask of every row (model.py:612: each block's flag repeated over its entries) as one flag per
    block, checked to give the mask back a slab of rows at a time."""
    blocks = candidates[..., ::block_size].contiguous()
    for first in range(0, candidates.size(1), SLAB_CHECK_ROWS):
        rows = slice(first, first + SLAB_CHECK_ROWS)
        if not torch.equal(expanded(blocks[:, rows], block_size, candidates.size(-1)), candidates[:, rows]):
            raise RuntimeError("the candidate mask is not one flag per block")
    return blocks


def expanded(blocks, block_size, entries):
    return blocks.repeat_interleave(block_size, dim = -1)[..., :entries]


def part_folder(directory, first, last):
    return os.path.join(directory, f"layers_{first}_{last}")


def earlier_parts(directory, first):
    """The folders of the parts that ran layers 0 to first - 1, in order, one part from each layer they start at:
    (first, last, folder) each. Raises RuntimeError where they leave a layer out, overlap or are two at a layer."""
    found = {}
    for name in sorted(os.listdir(directory)):
        match = PART_NAME.fullmatch(name)
        if match is not None:
            found.setdefault(int(match["first"]), []).append((int(match["last"]), os.path.join(directory, name)))
    chain, layer = [], 0
    while layer < first:
        ends = found.get(layer, [])
        if len(ends) != 1 or not layer <= ends[0][0] < first:
            raise RuntimeError(f"{directory}: not one part from layer {layer} that ends before layer {first}")
        chain.append((layer, *ends[0]))
        layer = ends[0][0] + 1
    return chain


def save_part(folder, session, first, last, left, notes, identity, results, held):
    """What a part leaves (the module text) into folder, which must not hold a capture; its manifest says complete
    only where every record was written and the part's checks held. notes: {layer: SlabIndexers.noted(layer)} of
    the part's index layers (None where a layer noted no row)."""
    model, shared, sizes = session.model, session.reference.model.shared_attn, session.model_arguments
    attentions, indexers, owners, _ = unit_13b.layers_of(session)
    writer, written = oracle.CaptureWriter(folder), False
    notes = {layer: noted for layer, noted in notes.items() if noted is not None}

    def add(name, tensor, layer, description):
        writer.add(name, tensor, layer, description, **{BYTE_SUM: int(oracle.byte_sum(tensor))})

    try:
        writer.manifest["conditions"] = {"identity": identity, "first": first, "last": last}
        add(STREAMS, left["streams"].view(torch.int16), last, "the streams leaving the part's last block, bfloat16")
        add(CARRIED, left["carried"], last, "the collapse weights they carry (model.py:986)")
        for layer in range(first, last + 1):
            for name, buffer in state_buffers(model.layers[layer]):
                add(BUFFER_RECORD.format(layer = layer, name = name), buffer, layer, "the block's state after the part")
            for record, values in zip(NOTE_RECORDS, notes[layer][1:] if layer in notes else ()):
                add(gates.INDEX_NOTE.format(layer = layer, record = record), values, layer, "the part's notes")
        if shared.topk_idxs is not None:
            add(TOPK_IDXS, shared.topk_idxs, None, "the picks the layers after an index source reuse (model.py:726)")
        if shared.candidates is not None:
            add(CANDIDATE_BLOCKS, candidate_blocks(shared.candidates, sizes.candidate_block_size).to(torch.uint8),
                None, "the candidate source's blocks, a flag each (model.py:569)")
        writer.results.update(results, pool_of_layer = next((attention.layer_id for attention in attentions
                                                             if attention.is_kv_source
                                                             and attention.compress_kv_cache is shared.compress_kv),
                                                            None),
                              keys_of_layer = next((layer for layer, owner in owners.items()
                                                    if owner.k_cache is shared.index_k), None),
                              noted = {str(layer): notes[layer][0].tolist() for layer in notes})
        written = held
    finally:
        writer.close(written)


def restore_parts(directory, first, session, identity):
    """Inside the run's conditions, once they have put fresh buffers in place: every earlier part's buffers into the
    model's and the shared runtime as the last of them left it. Returns the streams and carried weights for block
    `first` and the notes of the earlier index layers ({layer: (positions, picks, cut, tied, picked)}). Raises
    RuntimeError for a part of another run, a record missing or not what was written."""
    model, shared, sizes = session.model, session.reference.model.shared_attn, session.model_arguments
    attentions, indexers, owners, _ = unit_13b.layers_of(session)
    notes, capture, chain = {}, None, earlier_parts(directory, first)
    if not chain:
        raise RuntimeError("a part from layer 0 restores nothing")
    for part_first, part_last, folder in chain:
        capture = gates.CaptureFolder(folder)
        if capture.conditions != {"identity": identity, "first": part_first, "last": part_last}:
            raise RuntimeError(f"{folder}: a part of another run, or of other layers")
        for layer in range(part_first, part_last + 1):
            for name, buffer in state_buffers(model.layers[layer]):
                values = read_back(capture, BUFFER_RECORD.format(layer = layer, name = name))
                if values.dtype != buffer.dtype or values.shape != buffer.shape:
                    raise RuntimeError(f"{folder}: layer {layer}'s {name} is of another dtype or shape")
                buffer.copy_(values)
            if str(layer) in capture.results.get("noted", {}):
                positions = torch.from_numpy(numpy.asarray(capture.results["noted"][str(layer)], numpy.int64))
                notes[layer] = (positions, *(read_back(capture, gates.INDEX_NOTE.format(
                    layer = layer, record = record)).cpu() for record in NOTE_RECORDS))
    shared.topk_idxs = read_back(capture, TOPK_IDXS) if capture.has(TOPK_IDXS) else None
    shared.candidates = None
    if capture.has(CANDIDATE_BLOCKS):
        blocks, entries = read_back(capture, CANDIDATE_BLOCKS).bool(), first_candidate_entries(session, capture)
        shared.candidates = torch.cat([expanded(blocks[:, rows], sizes.candidate_block_size, entries) for rows in (
            slice(row, row + SLAB_CHECK_ROWS) for row in range(0, blocks.size(1), SLAB_CHECK_ROWS))], dim = 1)
    pool, keys = capture.results.get("pool_of_layer"), capture.results.get("keys_of_layer")
    shared.compress_kv = None if pool is None else attentions[pool].compress_kv_cache
    shared.index_k = None if keys is None else owners[keys].k_cache
    return (read_back(capture, STREAMS).view(torch.bfloat16), read_back(capture, CARRIED)), notes


def read_back(capture, name):
    """A record of a part's folder on the card, in the dtype it was written from, checked against its byte sum."""
    if not capture.has(name):
        raise RuntimeError(f"{capture.folder}: no record {name}")
    record = capture.records[name]
    values = torch.from_numpy(numpy.array(capture.array(name))).to(oracle.CARD_DEVICE)
    if record.get("reference_dtype") == gates.BFLOAT16:
        values = values.to(torch.bfloat16)
    if int(oracle.byte_sum(values)) != record.get(BYTE_SUM):
        raise RuntimeError(f"{capture.folder}: {name} is not what was written")
    return values


def first_candidate_entries(session, capture):
    """The entries the candidate source scores over a prompt: the prompt's tokens over its rate."""
    sizes = session.model_arguments
    return capture.conditions["identity"]["prompt_tokens"] // sizes.compress_ratios[sizes.candidate_source_layer]


def card_bytes_needed(tokens, slab_rows, settings, layers, twin):
    """The peak card memory of a long pass or part of layers (first, last): what block 0 holds over every row (the
    model, the state, the streams and a block's work), linear in the tokens through FIRST_BLOCK_PEAKS (13.4 GiB and
    0.57 MiB a token); the float32 scores of one slab at the part's widest index layer; the candidates of every row
    where the part reaches the candidate source (a later part restores them whole); the streams a resumed part
    reads; the twin's matrices. Each term is added to the peak, which those batches' index layers stayed under."""
    (short, short_gib), (long, long_gib) = FIRST_BLOCK_PEAKS
    per_token = (long_gib - short_gib) * oracle.BYTES_PER_GIB / (long - short)
    (first, last), ratios, source = layers, settings["compress_ratios"], settings["candidate_source_layer"]
    rates = [rate for rate in ratios[first:last + 1] if rate]
    pair = (unit_13b.SCORE_TENSORS_PER_HEAD * settings["index_n_heads"] + 1) * unit_13b.WIDE_SCORE_BYTES \
        + unit_13b.SCORE_MASK_BYTES
    scores = pair * slab_rows * (tokens // min(rates)) if rates else 0
    candidates = tokens * (tokens // ratios[source]) if 0 <= source <= last else 0
    streams = tokens * settings["hc_mult"] * settings["dim"] * oracle.REFERENCE_DEFAULT_DTYPE.itemsize
    return short_gib * oracle.BYTES_PER_GIB + per_token * (tokens - short) + scores + candidates \
        + streams * (first > 0) + unit_13b.FLOOR_MATRIX_GIB * oracle.BYTES_PER_GIB * twin


def differing_rows(first, second):
    """Of two tensors [1, rows, ...]: the rows that differ in a bit, every row where the shapes differ."""
    if first.shape != second.shape:
        return first.shape[1] if first.dim() > 1 else 1
    return int((first != second).reshape(first.shape[0] * first.shape[1], -1).any(dim = -1).sum())


def arm_differences(reference, other, layers):
    """slab_check's comparison of two arms (outputs by layer, candidates, head rows, logits, buffers by name): per
    record, the rows that differ (or the buffers), every call one arm made and the other did not counted."""
    def calls(first, second):
        return sum(map(differing_rows, first, second)) + abs(len(first) - len(second))
    counts = {f"layer_{layer}": calls(reference[0][layer], other[0][layer]) for layer in layers}
    counts.update(candidates = calls(reference[1], other[1]), head_rows = differing_rows(reference[2][None],
                                                                                          other[2][None]),
                  logits = differing_rows(reference[3][None], other[3][None]),
                  buffers = sum(not oracle.same_bits(buffer, other[4].get(name, buffer[:0]))
                                for name, buffer in reference[4].items()) + len(set(other[4]) - set(reference[4])))
    return counts


def twin_floor(reference_folder, twin_folder, output_json, label, mode):
    """The twin's figures against the reference capture (both of stepwise_capture.py's `mode`), written to
    output_json as the floor G8 reads (compare_engine_capture.py's long-context computation, one for both)."""
    reference, twin = gates.CaptureFolder(reference_folder), gates.CaptureFolder(twin_folder)
    theirs, mine = (capture.conditions.get("identity") or {} for capture in (reference, twin))
    if reference.conditions.get("mode") != mode or theirs.get("twin") is not None or not mine.get("twin") \
            or {**theirs, "twin": mine["twin"]} != mine:
        raise RuntimeError("--reference-capture is no u16 long capture of these tokens, lengths, slabs and rows")
    figures, picks = gates.long_context_figures(reference, twin)
    record = {"kind": gates.FLOOR_KIND, "complete": True, gates.FLOOR_TOKENS_SHA256: theirs["tokens_sha256"],
              "positions": len(reference.array(gates.FLAGGED_POSITIONS)), "constants": gates.floor_constants(),
              "long_context_constants": gates.long_context_constants(), gates.FLOOR_TWIN: figures,
              "picks": {str(layer): figure for layer, figure in picks.items()}, "identity": mine,
              "command_line": sys.argv}
    print(f"FLOOR {label} arm={gates.FLOOR_TWIN} positions={figures['positions_scored']} "
          f"mean_kl={figures['mean_kl']:.6f} mean_kl_standard_error={figures['mean_kl_standard_error']:.6f} "
          f"chunks={figures['chunks']} same_top={figures['same_top']:.4f} "
          f"margin_change_p999={figures['margin_change_p999']:.4e} delta={figures['delta']:.4e}")
    for layer, figure in picks.items():
        print(f"PICKS {label} arm={gates.FLOOR_TWIN} layer={layer} {gates.figure_text(figure)}")
    with open(output_json, "x", encoding = "utf8") as handle:
        json.dump(record, handle, indent = JSON_INDENT_SPACES)


def block_timer(session, seconds, label = None):
    """Hooks on every block (registered while model.layers holds them all) that add the seconds of each call, the
    card synchronized at its start and end, to seconds[layer]; with a label, a BLOCK line after each call of a
    prompt (more than one row), so that a pass cut short still shows how far it came. Returns the handles."""
    started = {}

    def begins(block, inputs):
        started[block.layer_id] = oracle.synchronized_clock()

    def ends(block, inputs, output):
        took = oracle.synchronized_clock() - started[block.layer_id]
        seconds[block.layer_id] = seconds.get(block.layer_id, 0.0) + took
        if label is not None and output[0].size(1) > 1:
            print(f"BLOCK {label} layer={block.layer_id} seconds={took:.1f} card_gib="
                  f"{torch.cuda.max_memory_allocated() / oracle.BYTES_PER_GIB:.2f}", flush = True)

    return [handle for block in session.model.layers for handle in (
        block.register_forward_pre_hook(begins), block.register_forward_hook(ends))]


def arm_recorders(session, outputs, candidates):
    """slab_check's hooks on one arm: every indexer's output into outputs[layer], the candidate source's candidates
    into candidates, both on the host. Returns the handles."""
    shared, indexers = session.reference.model.shared_attn, unit_13b.layers_of(session)[1]
    return [indexer.register_forward_hook(lambda module, inputs, output, layer = layer: outputs[layer].append(
        output.cpu())) for layer, indexer in indexers.items()] + [indexer.register_forward_hook(
            lambda module, inputs, output: candidates.append(shared.candidates.cpu()))
        for indexer in indexers.values() if indexer.is_candidate_source]


def seconds_text(seconds):
    return ",".join(f"{seconds[layer]:.1f}" for layer in sorted(seconds))


def long_fit(arguments, settings, slab_check_mode, run_label, proof_tokens):
    """stepwise_capture.py's FIT line of a long run or a slab_check: the card memory of its costliest pass (with
    slab_check, the reference's own indexers: state_precision_true.py's fit; with a long run's proof over
    proof_tokens tokens and steps, 0 for none, that fit there with the twin's matrices where it is installed)
    against the card's. Returns whether it fits with that tool's margin."""
    (first, last), twin, runs = arguments.layer_range, arguments.floor_gguf is not None, unit_13b.parse_runs(run_label)
    needed = unit_13b.card_bytes_needed(arguments, runs, settings)[0] if slab_check_mode \
        else card_bytes_needed(arguments.tokens, arguments.slab_rows, settings, (first, last), twin)
    proof = unit_13b.card_bytes_needed(types.SimpleNamespace(tokens = proof_tokens, floor_gguf = None), runs,
                                       settings)[0] + unit_13b.FLOOR_MATRIX_GIB * oracle.BYTES_PER_GIB * twin \
        if proof_tokens and not slab_check_mode else 0.0
    card_bytes = torch.cuda.get_device_properties(arguments.card).total_memory
    fits = max(needed, proof) + unit_13b.FIT_MARGIN_GIB * oracle.BYTES_PER_GIB <= card_bytes
    print(f"FIT tool=stepwise_capture mode={arguments.mode} tokens={arguments.tokens} layers={first}..{last} "
          f"slab_rows={arguments.slab_rows} twin={'yes' if twin else 'no'} estimate_gib="
          f"{needed / oracle.BYTES_PER_GIB:.2f} proof_tokens={proof_tokens} proof_estimate_gib="
          f"{proof / oracle.BYTES_PER_GIB:.2f} margin_gib={unit_13b.FIT_MARGIN_GIB:.1f} card_gib="
          f"{card_bytes / oracle.BYTES_PER_GIB:.2f} verdict={'fits' if fits else 'does_not_fit'}")
    return fits
