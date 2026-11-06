#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is the exact prefill shortcut of DeepSeek-V4.1-Flash bit-equal to a full prefill, on today's engine?

Layers 0-19 have to see every prompt row. A layer of 20-39 reads from the past only the compressed pool, which
layer 20 writes from its input, and its own window of the last 127 rows. So after a prompt of N rows layer l >= 20
owes output only for the last 1 + (39 - l) * 127 rows, with its window warmed by the 127 rows before them
(research/model.md 3.10). This tool runs the exllamav3 port (the frozen copy that serves today) over one prompt in
the passes below, each from an emptied cache, and compares what generation reads afterwards: every pool entry and
pool mirror, the last 127 window rows of every layer, the compressor rows, the n-gram token ids, the last row's
logits.

  port full          model.forward, all 40 layers, chunks of 4096 rows: the reference.
  port repeat        the same again: what the port itself changes between two identical passes.
  control            the walk the passes below are made of, in the port's own calls. The tool calls the port's
                     modules itself: layers 21-39 one after another over the rows from the chunk boundary before the
                     decoder rows, each fed from host memory with what the layer before left. Here the calls are
                     the port's chunks, and collapse and norm run over the whole last chunk, so every call has the
                     shape "port full" gave it and every tensor has to equal "port full". If not, the walk is wrong.
  same calls         all 40 layers over all rows, the calls of layers 21-39 cut where the shortcut cuts them.
  same calls repeat  the same again: what two identical passes of the cut calls change. The first of the two also
                     takes the port's first-use launch tuning of the new call shapes.
  shortcut           layers 0-19 over all rows in the same chunks. Layer 20 in the same chunks too, because its
                     attention is what writes the pool: attention only up to the chunks that hold the decoder rows,
                     whole block there. Layers 21-39 only over their rows: 127 warm-up rows through attention only,
                     starting from an empty window, then the output rows through the whole block.
  broken shortcut    only with --break-on-purpose: the shortcut with the last layer's warm-up one row short.

The port chooses its kernels by the rows of a call (linears at 144 and 1024 rows, experts by rows per expert), so
"port full against shortcut" also measures the call boundaries. "same calls against shortcut" has the same calls on
both sides and measures the shortcut alone. Experts held in RAM are not repeatable in prefill unless
EXL3_MOE_STREAM_FUSED_T=0 and EXL3_MOE_RECON_DET=1 are exported first (the port's doc/env_vars.md): with
--ram-layers or --ram-split the tool stops before it loads the model when they are not, unless
--accept-unrepeatable-experts is given.

    EXL3_MOE_STREAM_FUSED_T=0 EXL3_MOE_RECON_DET=1 run_with_port.sh ../prefill_cone/port_prefill_cone_check.py
        --tokens 3400 --layers-per-device 12,18,10 --ram-layers 33..39
        --output-directory ~/anemone/prefill_cone/3400                              (then 34000 and 100000)
    the same with --break-on-purpose and a directory of its own, ~/anemone/prefill_cone/3400_broken: a missing
    warm-up row of the last layer can change only the last row's logits, and the comparison has to say so

Exit codes:
  0  the shortcut equals "port full"
  3  it differs from "port full" but equals "same calls": only the call boundaries change the port's bits
  1  it differs from both
  2  a failure, or conditions the tool refuses
  5  "control" differs from "port full" while "port repeat" does not: the walk is wrong, the shortcut is not run
  6  a pass differs from its repeat: nothing is proven
  with --break-on-purpose, in place of 0 and 3 (1 then means: the unbroken shortcut differs from "same calls"):
  4  the unbroken shortcut equals "same calls" and the broken one differs from it, only in the last row's logits
  7  the broken one equals "same calls" too (the check is blind), or differs in another tensor

The output directory, which must not hold a capture yet, gets tensors.bin (raw little-endian tensors: the whole
"port full" pass, and of the other passes the tensors that differ from it) and manifest.json ("tensors": name,
dtype, shape, byte_offset, byte_count of each; "conditions"; "comparisons"). Both are created before the model is
loaded and taken away again when the run fails (exit code 2).
Measurement tool: not part of the engine.
"""
import argparse
import collections
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import sys
import time
import traceback

import torch

sys.dont_write_bytecode = True               # nothing is written outside the output directory, compiled imports included
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "measure"))
from kernel_time_trace import (CACHE_PAGE_TOKENS, DEFAULT_CORPUS_DIRECTORY, DEFAULT_MODEL_DIRECTORY,
                               PREFILL_CHUNK_TOKENS, load_corpus_tokens, load_model)

TENSORS_FILE_NAME = "tensors.bin"
MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_INDENT_SPACES = 1
# recorded with every capture: the port's switches and tuning folder, the cards it sees, the libraries under it
RECORDED_ENVIRONMENT_PREFIXES = ("EXL3_", "EXLLAMA", "CUDA_", "TORCH_", "TRITON_")
# what the port's doc/env_vars.md asks for before experts held in RAM give the same prefill bits twice
REPEATABLE_RAM_EXPERT_SETTINGS = {"EXL3_MOE_STREAM_FUSED_T": "0", "EXL3_MOE_RECON_DET": "1"}
NO_EXPERTS_IN_RAM = "0"                      # the value of --ram-layers and --ram-split that keeps every expert on a card
TOKEN_ID_HASH_TYPE = "<i8"                   # the prompt's SHA-256 is taken over its token ids as little-endian 64-bit integers
PLACEMENT_LINE_START = "placement"           # first word of the placement line load_model prints
EXIT_CODE_EQUAL = 0                          # the shortcut equals the port's full prefill
EXIT_CODE_DIFFERENT = 1                      # it differs from the port's full prefill and from the same-calls pass
EXIT_CODE_FAILURE = 2                        # an error, or conditions the tool refuses
EXIT_CODE_ONLY_CALL_BOUNDARIES_DIFFER = 3    # it differs from the port's full prefill, not from the same-calls pass
EXIT_CODE_BROKEN_CONE_SEEN = 4               # --break-on-purpose: the missing row is reported, and only where it can show
EXIT_CODE_CONTROL_DIFFERS = 5                # the tool's own walk does not give the port's bits: the shortcut is not run
EXIT_CODE_NOT_REPEATABLE = 6                 # a pass differs from its own repeat: nothing is proven
EXIT_CODE_BROKEN_CONE_NOT_SEEN = 7           # --break-on-purpose: the missing row is not reported, or not only where it can show
MINIMUM_PROMPT_TOKENS = 1                    # checked before the load; the real floor, known after it, is the decoder rows
PROMPT_START_ROW = 0                         # every pass starts at the prompt's first row, on an emptied cache
SEQUENCES_PER_CALL = 1                       # one prompt: the batch of every call holds one sequence
FIRST_SEQUENCE = 0                           # its index in the batch and its row of the block table
ROWS_OF_THE_LAST_POSITION = 1                # the prompt's last row: all that the last layer owes
OWN_ROWS_IN_A_WINDOW = 1                     # a row's window holds the row itself and the rows before it
LAST_ROW = -1                                # index of the last row of a call
LOGIT_ROWS = 1                               # the head runs on the last row of a call only, as in the port's warm-up passes
ROWS_CUT_ON_PURPOSE = 1                      # --break-on-purpose: rows taken from the last layer's warm-up
ONLY_LAYER_INSTANCE = 0                      # the port numbers repeats of a layer; this model has none
ENGRAM_LAYER_NUMBER_OFFSET = 1               # the port numbers the n-gram layer before layer l as -(l + 1)
INTEGER_TYPE_OF_BYTES = {2: torch.int16, 4: torch.int32, 8: torch.int64}       # to compare floating values bit by bit
# the passes; CONTROL, SAME_CALLS and SHORTCUT also name the three ways run_hand_walk cuts the decoder calls
PORT_FULL, PORT_REPEAT, CONTROL = "port full", "port repeat", "control"
SAME_CALLS, SAME_CALLS_REPEAT = "same calls", "same calls repeat"
SHORTCUT, BROKEN_SHORTCUT = "shortcut", "broken shortcut"
LAST_ROW_LOGITS = "logits_of_last_row"       # the one captured tensor a missing warm-up row of the last layer can change
# names the port gives to what the harness reads from it
ATTENTION_ONLY_PARAMETER = "prefill"         # a block given this returns after its attention: pool and window written, no experts
PREVIOUS_SITE_WEIGHTS_PARAMETER = "hc_pre"   # mixing weights of the site before; every site collapses the streams with them
SHARED_SELECTION_PARAMETER = "dsv4_shared"   # {state slot: {selection, candidates}}, read by the layers that reuse them
SELECTION_KEY = "topk"                       # (pool entries selected per row, how many are valid)
CANDIDATES_KEY = "cand"                      # per row, the candidate blocks layer 20 dropped (True = dropped)
LOGITS_CAPABILITY = "logits_output"          # the port's forward gives a module with this capability the last rows only
ENGRAM_IDS_ATTRIBUTE = "ids"                 # the n-gram layers' ring of compressed token ids
POOL_PARTS = (("pool_c", "plain_part"), ("pool_r", "rotated_part"), ("pool_idx", "index_keys"))
COMPRESSOR_PARTS = (("comp_buf_kv", "compressor_rows"), ("comp_buf_gate", "compressor_gate_rows"))

# where the shortcut cuts one layer after the first decoder layer: warm-up rows from the first, output rows from the second
LayerCut = collections.namedtuple("LayerCut", ("first_warm_up_row", "first_output_row"))


class TensorFileWriter:
    """Raw little-endian tensors, one after another in tensors.bin, listed in manifest.json for C programs. Both files
    are created at once, so that a directory that cannot take them stops the run before the model is loaded."""

    def __init__(self, output_directory):
        self.records, self.byte_offset, self.created_files = [], 0, []
        try:
            for file_name in (TENSORS_FILE_NAME, MANIFEST_FILE_NAME):
                self.created_files.append(open(os.path.join(output_directory, file_name), "xb"))
        except BaseException:
            self.remove()
            raise
        self.tensors_file, self.manifest_file = self.created_files

    def add(self, name, tensor):
        values = tensor.contiguous().numpy()
        values = values.astype(values.dtype.newbyteorder("<"), copy = False)
        values.tofile(self.tensors_file)
        self.records.append({"name": name, "dtype": values.dtype.name, "shape": list(values.shape),
                             "byte_offset": self.byte_offset, "byte_count": values.nbytes})
        self.byte_offset += values.nbytes

    def close(self, conditions, comparisons):
        manifest = {"tensors": self.records, "conditions": conditions, "comparisons": comparisons}
        self.manifest_file.write(json.dumps(manifest, indent = MANIFEST_INDENT_SPACES).encode("utf8"))
        for created_file in self.created_files:
            created_file.close()
        self.created_files = []                      # complete: nothing left for remove() to take away

    def remove(self):
        """After a failure: what this run created goes away, so that half a capture does not block the next run."""
        for created_file in self.created_files:
            created_file.close()
            os.remove(created_file.name)


class CarriedRows:
    """What a layer hands to the next one for a run of rows, kept in host memory: the four streams, the mixing weights
    of its last site, the pool entries selected per row and the candidate blocks dropped per row."""

    def __init__(self, first_row, streams, site_weights, selected_entries, selected_count, dropped_blocks):
        self.first_row = first_row
        self.streams = streams                       # (1, rows, streams, hidden) float32
        self.site_weights = site_weights             # (1, rows, streams) float32
        self.selected_entries = selected_entries     # (rows, entries) int32
        self.selected_count = selected_count
        self.dropped_blocks = dropped_blocks         # (rows, blocks) bool, or None below the port's candidate threshold

    def rows(self, first_row, end_row):
        """The part for the rows from first_row up to end_row, the row after the last one."""
        first_carried_index, end_carried_index = first_row - self.first_row, end_row - self.first_row
        if first_carried_index < 0 or first_carried_index >= end_carried_index or end_carried_index > self.streams.shape[1]:
            raise SystemExit(f"rows {first_row}..{end_row} were not handed down: the layer before gave "
                             f"{self.first_row}..{self.first_row + self.streams.shape[1]}")
        return CarriedRows(first_row, self.streams[:, first_carried_index:end_carried_index],
                           self.site_weights[:, first_carried_index:end_carried_index],
                           self.selected_entries[first_carried_index:end_carried_index], self.selected_count,
                           None if self.dropped_blocks is None else self.dropped_blocks[first_carried_index:end_carried_index])

    @staticmethod
    def join(pieces):
        masks = [piece.dropped_blocks for piece in pieces]
        if len({mask is None for mask in masks}) != 1 or len({piece.selected_count for piece in pieces}) != 1 \
                or len({piece.selected_entries.shape[1] for piece in pieces}) != 1:
            raise SystemExit("the rows the decoder layers run over cross a threshold of the port's index selection "
                             "(its top-k size or its candidate blocks): choose another --tokens")
        dropped_blocks = None
        if masks[0] is not None:
            # a mask ends at the last block of its own call; later blocks are after its rows, so they count as dropped
            dropped_blocks = torch.ones((sum(mask.shape[0] for mask in masks), max(mask.shape[1] for mask in masks)),
                                        dtype = torch.bool)
            row = 0
            for mask in masks:
                dropped_blocks[row:row + mask.shape[0], :mask.shape[1]] = mask
                row += mask.shape[0]
        return CarriedRows(pieces[0].first_row, torch.cat([piece.streams for piece in pieces], dim = 1),
                           torch.cat([piece.site_weights for piece in pieces], dim = 1),
                           torch.cat([piece.selected_entries for piece in pieces], dim = 0), pieces[0].selected_count,
                           dropped_blocks)


class LayerMap:
    """Where the decoder sits in the port's module list, and the model's window, read from the loaded model."""

    def __init__(self, model, config):
        self.entries = model.fwd_modules             # (module, instance, index), as the port's forward walks them
        self.block_position = {module.layer_idx: position for position, (module, _, _) in enumerate(self.entries)
                               if type(module).__name__ == "TransformerBlock"}
        if len(self.block_position) != config.num_hidden_layers:
            raise SystemExit(f"the port walks {len(self.block_position)} different blocks, its configuration counts "
                             f"{config.num_hidden_layers} layers")
        self.first_decoder_layer = max(config.kv_source_layer_ids)        # owns the pool every later layer reads
        self.last_layer = max(self.block_position)
        decoder_layers = sorted(layer for layer in self.block_position if layer >= self.first_decoder_layer)
        self.decoder_layer_count = len(decoder_layers)
        self.cut_layers = [layer for layer in decoder_layers if layer != self.first_decoder_layer]
        self.previous_rows = config.sliding_window - OWN_ROWS_IN_A_WINDOW  # window rows before the current one
        first_decoder_position, last_block_position = (self.block_position[self.first_decoder_layer],
                                                       self.block_position[self.last_layer])
        for layer in self.cut_layers:
            if self.block_position[layer] - first_decoder_position != layer - self.first_decoder_layer:
                raise SystemExit("another module sits between the decoder blocks of the port: this walk would skip it")
            if self.block(layer).attn.kv_source != self.first_decoder_layer:
                raise SystemExit(f"layer {layer} does not read the pool of layer {self.first_decoder_layer}")
        self.index_entries = self.block(self.first_decoder_layer).attn.index_topk
        # the port's walk in parts: up to the block that writes the decoder pool, up to the last block, and after it
        positioned_entries = list(enumerate(self.entries))
        self.entries_through_first_decoder_layer = [entry for position, entry in positioned_entries
                                                    if position <= first_decoder_position]
        self.entries_through_last_layer = [entry for position, entry in positioned_entries if position <= last_block_position]
        self.entries_after_last_layer = [entry for position, entry in positioned_entries if position > last_block_position]

    def block(self, layer):
        return self.entries[self.block_position[layer]][0]

    def output_rows(self, layer):
        return ROWS_OF_THE_LAST_POSITION + (self.last_layer - layer) * self.previous_rows

    def decoder_cuts(self, total_rows):
        """{layer: LayerCut} of the layers after the first decoder layer."""
        return {layer: LayerCut(total_rows - self.output_rows(layer) - self.previous_rows, total_rows - self.output_rows(layer))
                for layer in self.cut_layers}


def chunk_start(row):
    """First row of the port's chunk that holds `row`."""
    return row // PREFILL_CHUNK_TOKENS * PREFILL_CHUNK_TOKENS


def tail_start_row(cuts):
    """The chunk boundary before the decoder rows. Up to it every hand walk runs in the port's chunks; from it the
    layers after the first decoder layer run one after another, in the calls of the walk."""
    return chunk_start(cuts[min(cuts)].first_warm_up_row)


def begin_host_pass(model):
    """What the port's forward does before its first module: wake the workers of experts held in RAM."""
    for host in getattr(model.config, "moe_cpu_hosts", {}).values():
        host.begin_pass()


def call_parameters(cache, batch_rows, first_row, state):
    return {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (SEQUENCES_PER_CALL, batch_rows), "past_len": first_row,
            "recurrent_states": [state]}


def empty_the_cache(cache):
    """Zero every pool, pool mirror and n-gram id ring. Window and compressor rows are zeroed by get_new_state."""
    for pool in cache.layers.values():
        for tensor in pool.get_tensors():
            tensor.zero_()
    for layer_state in cache.get_all_recurrent_layers().values():
        if hasattr(layer_state, ENGRAM_IDS_ATTRIBUTE):
            getattr(layer_state, ENGRAM_IDS_ATTRIBUTE).zero_()


def capture_generation_state(cache, state, total_rows, live_rows, window_start_of_layer, logits):
    """{name: host tensor} of what generation reads after the prompt. A window row of absolute position p sits at
    index p - window start; the start is the job state's, or the layer's own where the walk gave it one."""
    captured = {}
    for (layer, _), pool in sorted(cache.layers.items()):
        entry_count = total_rows // pool.compress_rate
        holders = [("", pool)] + [(f"_mirror_on_{str(device).replace(':', '_')}", mirror)
                                  for device, mirror in sorted(pool.mirrors.items(), key = lambda item: str(item[0]))]
        for suffix, holder in holders:
            for attribute, part in POOL_PARTS:
                entries = getattr(holder, attribute)
                if entries is not None:     # the batch-shape block table is the identity: entry e is flat row e
                    captured[f"pool_of_layer_{layer}{suffix}_{part}"] = \
                        entries.view(-1, entries.shape[-1])[:entry_count].to("cpu", copy = True)
    for (layer, _), layer_state in sorted(cache.get_all_recurrent_layers().items()):
        if hasattr(layer_state, ENGRAM_IDS_ATTRIBUTE):
            captured[f"engram_token_ids_before_layer_{-(layer + ENGRAM_LAYER_NUMBER_OFFSET)}"] = \
                getattr(layer_state, ENGRAM_IDS_ATTRIBUTE)[state.slot].to("cpu", copy = True)
            continue
        first_window_index = total_rows - live_rows - window_start_of_layer.get(layer, state.window_beg)
        if first_window_index < 0 or first_window_index + live_rows > layer_state.ring_rows:
            raise SystemExit(f"layer {layer}: its last {live_rows} window rows are not in the port's buffer "
                             f"(they would start at index {first_window_index} of {layer_state.ring_rows})")
        captured[f"window_rows_of_layer_{layer}"] = \
            layer_state.ring[state.slot, first_window_index:first_window_index + live_rows].to("cpu", copy = True)
        for attribute, part in COMPRESSOR_PARTS:
            rows = getattr(layer_state, attribute)
            if rows is not None:
                captured[f"{part}_of_layer_{layer}"] = rows[state.slot].to("cpu", copy = True)
    captured[LAST_ROW_LOGITS] = logits[FIRST_SEQUENCE, LAST_ROW].to("cpu", copy = True)
    return captured


def run_port_full(model, cache, token_ids, batch_rows, layer_map):
    """The reference: the port's own forward over every layer, in chunks, the head on the last row of each chunk."""
    empty_the_cache(cache)
    state = cache.get_new_state()
    try:
        for first_row in range(PROMPT_START_ROW, token_ids.shape[1], PREFILL_CHUNK_TOKENS):
            parameters = call_parameters(cache, batch_rows, first_row, state)
            parameters["last_tokens_only"] = LOGIT_ROWS
            logits = model.forward(token_ids[:, first_row:first_row + PREFILL_CHUNK_TOKENS], parameters)
        return capture_generation_state(cache, state, token_ids.shape[1], layer_map.previous_rows, {}, logits)
    finally:
        cache.release_state(state)


def collect_rows(first_row, streams, parameters, slot):
    """A block's output rows and what the port left in the call's parameters for the next block, copied to the host."""
    from exllamav3.modules.hyperconnections import hc_flush
    hc_flush(parameters)                                    # a stream update the port may have left for the next site
    shared = (parameters.get(SHARED_SELECTION_PARAMETER) or {}).get(slot) or {}
    if SELECTION_KEY not in shared:
        raise SystemExit(f"the call at row {first_row} selected no pool entries: the prompt is too short for the "
                         "port's top-k selection")
    selected_entries, selected_count = shared[SELECTION_KEY]
    dropped_blocks = shared.get(CANDIDATES_KEY)
    return CarriedRows(first_row, streams.cpu(), parameters[PREVIOUS_SITE_WEIGHTS_PARAMETER].cpu(), selected_entries.cpu(),
                       selected_count, None if dropped_blocks is None else dropped_blocks.cpu())


def run_decoder_call(model, cache, token_ids, batch_rows, entry, layer_state, carried, first_row, end_row, attention_only):
    """One decoder block over the rows from first_row up to end_row, at the position of its own copy of the job state."""
    from exllamav3.cache.recurrent_util import advance_recurrent_states
    module, instance, _ = entry
    if layer_state.position != first_row:       # the port's attention takes its position from the state, not from the call
        raise SystemExit(f"{module.key}: the call starts at row {first_row}, its state stands at row {layer_state.position}")
    rows = carried.rows(first_row, end_row)
    parameters = call_parameters(cache, batch_rows, first_row, layer_state)
    model.prepare_inputs(token_ids[:, first_row:end_row], parameters)
    parameters[PREVIOUS_SITE_WEIGHTS_PARAMETER] = rows.site_weights
    shared = {SELECTION_KEY: (rows.selected_entries, rows.selected_count)}
    if rows.dropped_blocks is not None:
        shared[CANDIDATES_KEY] = rows.dropped_blocks
    parameters[SHARED_SELECTION_PARAMETER] = {layer_state.slot: shared}
    parameters["layer_instance"] = instance
    if attention_only:
        parameters[ATTENTION_ONLY_PARAMETER] = True
    begin_host_pass(model)
    streams = module.forward(module.prepare_for_device(rows.streams, parameters), parameters)
    output = None if attention_only else collect_rows(first_row, streams, parameters, layer_state.slot)
    advance_recurrent_states(token_ids[:, first_row:end_row], parameters, model)
    return output


def run_hand_walk(model, cache, token_ids, batch_rows, layer_map, cuts, walk, layer_broken_on_purpose = None):
    """The tool's own walk over the port's modules. Up to the chunk boundary before the decoder rows it runs in the
    port's chunks: every layer, or in the shortcut the layers up to the first decoder layer, that one through its
    attention only. From there the first decoder layer goes on in the port's chunks and the layers after it run one
    after another, fed from host memory, in the calls `walk` names:
      CONTROL     the port's chunks, whole blocks; collapse and norm over the port's whole last chunk.
      SAME_CALLS  the port's chunks up to the layer's first warm-up row, its warm-up rows, its output rows: whole
                  blocks; collapse and norm on the last row.
      SHORTCUT    the layer's warm-up rows through attention only from an empty window, then its output rows;
                  collapse and norm on the last row. `layer_broken_on_purpose` gets ROWS_CUT_ON_PURPOSE fewer
                  warm-up rows."""
    from exllamav3.cache.recurrent_util import advance_recurrent_states
    total_rows = token_ids.shape[1]
    last_prompt_row = total_rows - ROWS_OF_THE_LAST_POSITION
    shortcut = walk == SHORTCUT
    tail_start = tail_start_row(cuts)
    first_decoder_block = layer_map.block(layer_map.first_decoder_layer)
    empty_the_cache(cache)
    state = cache.get_new_state()
    try:
        tail_state, tail_chunks = None, []
        for first_row in range(PROMPT_START_ROW, total_rows, PREFILL_CHUNK_TOKENS):
            chunk_ids = token_ids[:, first_row:first_row + PREFILL_CHUNK_TOKENS]
            in_tail = first_row >= tail_start
            if first_row == tail_start:
                tail_state = copy.copy(state)               # where the layers after the first decoder layer go on from
            parameters = call_parameters(cache, batch_rows, first_row, state)
            streams = model.prepare_inputs(chunk_ids, parameters)
            begin_host_pass(model)
            for module, instance, _ in (layer_map.entries_through_first_decoder_layer if shortcut or in_tail
                                        else layer_map.entries_through_last_layer):
                parameters["layer_instance"] = instance
                if shortcut and not in_tail and module is first_decoder_block:
                    parameters[ATTENTION_ONLY_PARAMETER] = True
                streams = module.forward(module.prepare_for_device(streams, parameters), parameters)
            if in_tail:
                tail_chunks.append(collect_rows(first_row, streams, parameters, state.slot))
            advance_recurrent_states(chunk_ids, parameters, model)
        if shortcut:
            # a layer that reads the decoder pool on another card copies, in its own forward, the entries of the rows
            # it runs; the entries of the rows it never runs are copied here with the port's own function
            pool = cache.layers[(layer_map.first_decoder_layer, ONLY_LAYER_INSTANCE)]
            for mirror in pool.mirrors.values():
                mirror.pull(pool, parameters["block_table"][FIRST_SEQUENCE], PROMPT_START_ROW // pool.compress_rate,
                            total_rows // pool.compress_rate)

        carried = CarriedRows.join(tail_chunks)
        window_start_of_layer = {}
        for layer in layer_map.cut_layers:
            cut = cuts[layer]
            if shortcut:
                first_warm_up_row = cut.first_warm_up_row
                if layer == layer_broken_on_purpose:
                    first_warm_up_row += ROWS_CUT_ON_PURPOSE
                layer_state = copy.copy(state)
                layer_state.position = layer_state.window_beg = first_warm_up_row       # an empty window: no earlier row is read
                calls = [(first_warm_up_row, cut.first_output_row, True), (cut.first_output_row, total_rows, False)]
            else:
                layer_state = copy.copy(tail_state)
                chunked_end_row = total_rows if walk == CONTROL else cut.first_warm_up_row
                calls = [(first_row, min(first_row + PREFILL_CHUNK_TOKENS, chunked_end_row), False)
                         for first_row in range(tail_start, chunked_end_row, PREFILL_CHUNK_TOKENS)]
                if walk == SAME_CALLS:
                    calls += [(cut.first_warm_up_row, cut.first_output_row, False), (cut.first_output_row, total_rows, False)]
            outputs = [run_decoder_call(model, cache, token_ids, batch_rows, layer_map.entries[layer_map.block_position[layer]],
                                        layer_state, carried, first_row, end_row, attention_only)
                       for first_row, end_row, attention_only in calls]
            if layer_state.position != total_rows:
                raise SystemExit(f"layer {layer} stopped at row {layer_state.position} of {total_rows}")
            carried = CarriedRows.join([output for output in outputs if output is not None])
            window_start_of_layer[layer] = layer_state.window_beg

        # the control collapses and norms the port's whole last chunk, as the port's forward does; the head takes the last row
        first_collapse_row = chunk_start(last_prompt_row) if walk == CONTROL else last_prompt_row
        collapse_input = carried.rows(first_collapse_row, total_rows)
        parameters = call_parameters(cache, batch_rows, first_collapse_row, state)
        model.prepare_inputs(token_ids[:, first_collapse_row:], parameters)
        parameters[PREVIOUS_SITE_WEIGHTS_PARAMETER] = collapse_input.site_weights
        logits = collapse_input.streams
        for module, instance, _ in layer_map.entries_after_last_layer:
            parameters["layer_instance"] = instance
            if module.caps.get(LOGITS_CAPABILITY):
                logits = logits[..., -LOGIT_ROWS:, :].contiguous()
            logits = module.forward(module.prepare_for_device(logits, parameters), parameters)
        return capture_generation_state(cache, state, total_rows, layer_map.previous_rows, window_start_of_layer, logits)
    finally:
        cache.release_state(state)


def bit_pattern(tensor):
    return tensor.view(INTEGER_TYPE_OF_BYTES[tensor.element_size()]) if tensor.is_floating_point() else tensor


def compare_passes(reference, other):
    """{name: {"elements", "differing_elements", "largest_absolute_difference"}} of two passes, bit by bit. All three
    are None for a tensor that cannot be compared: missing on one side, or of another shape or type."""
    results = {}
    for name in list(reference) + [name for name in other if name not in reference]:
        expected, actual = reference.get(name), other.get(name)
        if expected is None or actual is None or expected.shape != actual.shape or expected.dtype != actual.dtype:
            results[name] = {"elements": None, "differing_elements": None, "largest_absolute_difference": None}
            continue
        differing = bit_pattern(expected) != bit_pattern(actual)
        differing_elements = int(differing.sum())
        largest = float((expected[differing].double() - actual[differing].double()).abs().max()) if differing_elements else 0.0
        results[name] = {"elements": expected.numel(), "differing_elements": differing_elements,
                         "largest_absolute_difference": largest if math.isfinite(largest) else None}
    return results


def print_comparison(title, results):
    print(f"\n== {title}")
    for name, result in results.items():
        if result["elements"] is None:
            print(f"  {name}: NOT COMPARABLE (missing on one side, or another shape or type)")
        elif not result["differing_elements"]:
            print(f"  {name}: equal")
        else:
            largest = result["largest_absolute_difference"]
            print(f"  {name}: {result['differing_elements']} of {result['elements']} elements differ, largest absolute "
                  "difference " + ("not finite" if largest is None else f"{largest:.6g}"))
    print(f"  {sum(1 for result in results.values() if result['differing_elements'] == 0)} of {len(results)} tensors equal")


def equal_or_different(differing_tensor_names):
    return "DIFFERENT" if differing_tensor_names else "EQUAL"


def run_and_judge(pass_arguments, cuts, layer_broken_on_purpose, captured_passes, comparisons):
    """Runs the passes, each over `pass_arguments` (model, cache, token_ids, batch_rows, layer_map), prints their
    comparisons and keeps both. Returns the exit code and the lines that say why."""

    def run(pass_name, run_function, *walk_arguments):
        started = time.time()
        captured_passes[pass_name] = run_function(*pass_arguments, *walk_arguments)
        print(f"pass \"{pass_name}\": {time.time() - started:.0f} s, {len(captured_passes[pass_name])} tensors", flush = True)

    def differing_tensor_names(reference_name, other_name, meaning):
        title = f"{reference_name} against {other_name}"
        results = comparisons[title.replace(" ", "_")] = compare_passes(captured_passes[reference_name],
                                                                        captured_passes[other_name])
        print_comparison(f"{title}: {meaning}", results)
        return [name for name, result in results.items() if result["differing_elements"] != 0]

    run(PORT_FULL, run_port_full)
    run(PORT_REPEAT, run_port_full)
    port_repeat_differs = differing_tensor_names(PORT_FULL, PORT_REPEAT, "what two identical passes of the port change")
    run(CONTROL, run_hand_walk, cuts, CONTROL)
    control_differs = differing_tensor_names(PORT_FULL, CONTROL,
                                             "the tool's own walk in the port's own calls, which must change nothing")
    if control_differs and not port_repeat_differs:
        return EXIT_CODE_CONTROL_DIFFERS, (
            "CONTROL FAILED: the tool's own walk over the port's modules, in calls of the port's own shapes, does not give "
            "the port's bits. The walk is wrong: the shortcut was not run and is not judged")
    run(SAME_CALLS, run_hand_walk, cuts, SAME_CALLS)
    differing_tensor_names(PORT_FULL, SAME_CALLS, "what the call boundaries alone change")
    run(SAME_CALLS_REPEAT, run_hand_walk, cuts, SAME_CALLS)
    same_calls_repeat_differs = differing_tensor_names(SAME_CALLS, SAME_CALLS_REPEAT,
                                                       "what two identical passes of the cut calls change")
    run(SHORTCUT, run_hand_walk, cuts, SHORTCUT)
    differs_from_port_full = differing_tensor_names(PORT_FULL, SHORTCUT, "the question, call boundaries included")
    differs_from_same_calls = differing_tensor_names(SAME_CALLS, SHORTCUT, "the shortcut alone")
    if layer_broken_on_purpose is not None:
        run(BROKEN_SHORTCUT, run_hand_walk, cuts, SHORTCUT, layer_broken_on_purpose)
        broken_differs_in = differing_tensor_names(
            SAME_CALLS, BROKEN_SHORTCUT, f"the warm-up of layer {layer_broken_on_purpose} {ROWS_CUT_ON_PURPOSE} row short")
    result = (f"RESULT shortcut against the port's full prefill: {equal_or_different(differs_from_port_full)}"
              f"; against the full prefill cut into the same calls: {equal_or_different(differs_from_same_calls)}"
              f"; control against the port's full prefill: {equal_or_different(control_differs)}"
              f"; two identical passes of the port: {equal_or_different(port_repeat_differs)}"
              f", of the same calls: {equal_or_different(same_calls_repeat_differs)}")
    if port_repeat_differs or same_calls_repeat_differs:
        return EXIT_CODE_NOT_REPEATABLE, result + (
            "\nNOT JUDGED: a pass differs from its own repeat, so no difference above proves anything. The first pass over "
            "new call shapes takes the port's launch tuning (run again); experts in RAM need "
            + " ".join(f"{name}={value}" for name, value in REPEATABLE_RAM_EXPERT_SETTINGS.items()))
    if layer_broken_on_purpose is None:
        if not differs_from_port_full:
            return EXIT_CODE_EQUAL, result
        return (EXIT_CODE_DIFFERENT if differs_from_same_calls else EXIT_CODE_ONLY_CALL_BOUNDARIES_DIFFER), result
    if differs_from_same_calls:
        return EXIT_CODE_DIFFERENT, result + ("\nBROKEN ON PURPOSE: NOT JUDGED, the unbroken shortcut already differs from "
                                              "the same-calls pass")
    if broken_differs_in == [LAST_ROW_LOGITS]:
        return EXIT_CODE_BROKEN_CONE_SEEN, result + ("\nBROKEN ON PURPOSE: the comparison reports the missing row, as it "
                                                     "must, and only in the last row's logits")
    return EXIT_CODE_BROKEN_CONE_NOT_SEEN, result + "\nBROKEN ON PURPOSE: " + (
        "NOT SEEN, the check cannot tell a cone that is one row short" if not broken_differs_in else
        "SEEN IN THE WRONG PLACE, a missing warm-up row of the last layer can change only the last row's logits, but "
        "these differ: " + ", ".join(broken_differs_in))


def load_and_compare(arguments, writer):
    batch_rows = -(-arguments.tokens // CACHE_PAGE_TOKENS) * CACHE_PAGE_TOKENS
    # load_model sizes the cache from this; the port's load runs one whole chunk through the cache
    arguments.cache_tokens = max(batch_rows, PREFILL_CHUNK_TOKENS)
    load_report = io.StringIO()
    try:
        with contextlib.redirect_stdout(load_report):
            config, model, cache, tokenizer = load_model(arguments)
    finally:
        print(load_report.getvalue(), end = "", flush = True)
    import exllamav3
    layer_map = LayerMap(model, config)
    cuts = layer_map.decoder_cuts(arguments.tokens)
    first_cut_layer, last_layer = min(cuts), max(cuts)
    first_cut, last_cut = cuts[first_cut_layer], cuts[last_layer]
    decoder_rows = arguments.tokens - first_cut.first_warm_up_row
    if first_cut.first_warm_up_row <= layer_map.index_entries:
        raise SystemExit(f"--tokens must be above {decoder_rows + layer_map.index_entries}: the decoder rows "
                         f"({decoder_rows}) have to start past the port's top-k selection size ({layer_map.index_entries})")
    token_ids = load_corpus_tokens(tokenizer, arguments.corpus, arguments.tokens)
    layer_broken_on_purpose = last_layer if arguments.break_on_purpose else None
    tail_start = tail_start_row(cuts)
    output_layer_rows = sum(layer_map.output_rows(layer) for layer in cuts)
    print(f"{arguments.tokens} prompt rows; layer {first_cut_layer} warms up from row {first_cut.first_warm_up_row} and gives "
          f"output from row {first_cut.first_output_row}, layer {last_layer} from rows {last_cut.first_warm_up_row} and "
          f"{last_cut.first_output_row}"
          + (f"; BROKEN ON PURPOSE in one more pass: layer {layer_broken_on_purpose} warms up {ROWS_CUT_ON_PURPOSE} row late"
             if arguments.break_on_purpose else ""))
    print(f"decoder work, in layer-rows: port full {arguments.tokens * layer_map.decoder_layer_count} through whole blocks; "
          f"shortcut {arguments.tokens - tail_start + output_layer_rows} through whole blocks (layer "
          f"{layer_map.first_decoder_layer} from chunk boundary {tail_start}) and "
          f"{tail_start + len(cuts) * layer_map.previous_rows} through attention only", flush = True)
    conditions = {
        "command_line": sys.argv,
        "port_copy": exllamav3.__file__,
        "placement": next((line for line in load_report.getvalue().splitlines() if line.startswith(PLACEMENT_LINE_START)), None),
        "environment": {name: value for name, value in sorted(os.environ.items())
                        if name.startswith(RECORDED_ENVIRONMENT_PREFIXES)},
        "torch_version": torch.__version__,
        "cards": [torch.cuda.get_device_name(card) for card in range(torch.cuda.device_count())],
        "prompt_tokens": arguments.tokens,
        "prompt_token_ids_sha256": hashlib.sha256(token_ids.cpu().numpy().astype(TOKEN_ID_HASH_TYPE).tobytes()).hexdigest(),
        "chunk_tokens": PREFILL_CHUNK_TOKENS,
        "cut_of_layer": {str(layer): cut._asdict() for layer, cut in cuts.items()},
        "layer_broken_on_purpose": layer_broken_on_purpose,
    }

    captured_passes, comparisons = {}, {}
    exit_code, verdict = run_and_judge((model, cache, token_ids, batch_rows, layer_map), cuts, layer_broken_on_purpose,
                                       captured_passes, comparisons)
    port_full = captured_passes[PORT_FULL]
    for pass_name, captured in captured_passes.items():
        against_port_full = None if captured is port_full else compare_passes(port_full, captured)
        for name, tensor in captured.items():
            if against_port_full is None or against_port_full[name]["differing_elements"] != 0:
                writer.add(f"{pass_name.replace(' ', '_')}/{name}", tensor)
    writer.close(conditions, comparisons)
    print(f"\n{len(writer.records)} tensors, {writer.byte_offset} bytes in {arguments.output_directory}\n{verdict}")
    return exit_code


def check(arguments):
    if arguments.tokens < MINIMUM_PROMPT_TOKENS:
        raise SystemExit("--tokens must be positive")
    settings_not_exported = [f"{name}={value}" for name, value in REPEATABLE_RAM_EXPERT_SETTINGS.items()
                             if os.environ.get(name) != value]
    experts_in_ram = arguments.ram_layers != NO_EXPERTS_IN_RAM or arguments.ram_split != NO_EXPERTS_IN_RAM
    if experts_in_ram and settings_not_exported and not arguments.accept_unrepeatable_experts:
        raise SystemExit("--ram-layers and --ram-split keep experts in RAM, and their prefill does not give the same bits "
                         f"twice unless the port finds {' '.join(settings_not_exported)} in its environment (doc/env_vars.md): "
                         "no comparison would prove anything. Export them before the run, or pass "
                         "--accept-unrepeatable-experts to run all the same")
    for file_name in (TENSORS_FILE_NAME, MANIFEST_FILE_NAME):
        if os.path.exists(os.path.join(arguments.output_directory, file_name)):
            raise SystemExit(f"{arguments.output_directory} already holds {file_name}: give an empty directory")
    os.makedirs(arguments.output_directory, exist_ok = True)
    writer = TensorFileWriter(arguments.output_directory)
    try:
        return load_and_compare(arguments, writer)
    except BaseException:
        writer.remove()
        raise


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type = int, required = True, help = "prompt length, taken from the corpus")
    parser.add_argument("--output-directory", type = os.path.abspath, required = True,
                        help = "receives tensors.bin and manifest.json; must not hold a capture yet")
    parser.add_argument("--break-on-purpose", action = "store_true",
                        help = "add a shortcut pass with the last layer's warm-up one row short: only the last row's logits may "
                               "differ, and must")
    parser.add_argument("--accept-unrepeatable-experts", action = "store_true",
                        help = "run with experts in RAM although the port's settings for a repeatable prefill are not exported")
    parser.add_argument("--model", default = DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--corpus", default = DEFAULT_CORPUS_DIRECTORY)
    parser.add_argument("--ram-layers", default = NO_EXPERTS_IN_RAM,
                        help = "layers whose experts stay in RAM, as the port's -mcl takes them")
    parser.add_argument("--ram-split", default = NO_EXPERTS_IN_RAM,
                        help = "experts per layer kept in RAM, as the port's -mcs takes them")
    parser.add_argument("--layers-per-device", default = None)
    parser.add_argument("--gigabytes-per-device", default = None)
    try:
        with torch.inference_mode():
            return check(parser.parse_args())
    except SystemExit as stop:
        if not isinstance(stop.code, str):          # argparse: 0 after --help, 2 after a usage error
            raise
        print(f"FAILED: {stop.code}", file = sys.stderr)
    except Exception:
        traceback.print_exc()
        print("FAILED: the error above stopped the check", file = sys.stderr)
    return EXIT_CODE_FAILURE


if __name__ == "__main__":
    sys.exit(main())
