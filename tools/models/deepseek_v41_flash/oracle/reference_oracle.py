#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Captures of DeepSeek's own reference implementation: the ground truth every Anemone kernel is gated against.

Builds the reference model of a DeepSeek-V4.1-Flash checkpoint (the Hugging Face folder as downloaded: the
safetensors shards, the tokenizer, inference/ and encoding/) on ONE card, straight from the shards, and performs
one capture:
  * forward   one pass over a token sequence from position 0, the reference's prompt path: the input embeddings,
              the residual streams that enter and leave every layer with their collapse weights, the routed expert
              ids and route weights of every layer and position, the n-gram row ids of the engram layers, the
              logits of the last position, and of every position with --all-logits;
  * linears   the same pass, recording every call of the reference's quantized linear in the layers of --layers:
              its input, the input as the reference quantized it with the scales, and its output. Of the routed
              experts, --expert-calls (position, expert) pairs per layer, the first rows of each call;
  * generate  greedy generation of --new-tokens tokens with the reference's own generate(): every step's token id
              and its logit, the --top-k ids and logits, the text; with --twice it is repeated and compared bit by bit.

    reference_oracle.py --checkpoint CHECKPOINT --output-directory FOLDER --self-test
    reference_oracle.py --checkpoint CHECKPOINT --output-directory FOLDER --mode forward --tokens-json FILE
    reference_oracle.py --checkpoint CHECKPOINT --output-directory FOLDER --mode linears --prompt-json FILE
    reference_oracle.py ... --mode generate --tokens-json FILE --new-tokens 200 --twice
    reference_oracle.py ... --mode forward --all-logits --weights-gguf FILE --gguf-linear-inputs as-the-reference

--tokens-json is a list of token id lists, or port_oracle_capture.py's list of {"name", "token_ids"}; --prompt-json
is what the reference's generate.py takes (a case, a list of cases or a bare message list), encoded by
CHECKPOINT/encoding and the checkpoint's tokenizer through generate.py's own prepare_case. --sequence picks the
entry of either file and --tokens N its first N tokens. --self-test runs the three captures on the first tokens
of a fixed prompt and prints the shapes of what it wrote.

The reference is imported from CHECKPOINT/inference, never from a copy, and no file of it is edited. The model is
built and driven as generate.py does it (default dtype, construction on the card, default device, model.forward
from position 0, generate()), with these differences, none of which touches the arithmetic of a forward:
  * no convert.py pass: every tensor of the shards goes to the parameter convert.py would have stored it under for
    one rank, with convert.py's own changes (the 4-bit bytes of the routed experts read as float4_e2m1fn_x2, the
    attention output projection wo_a multiplied by its block scales into bfloat16) and the widening of bfloat16 to
    float32 that the reference's load performs. A checkpoint tensor no parameter takes, a parameter no tensor
    fills, a shape or a dtype that does not agree: each stops the tool with the name;
  * the routed experts (the backbone's and the draft head's) are never on the card as a whole. Each stays in its
    shard, memory-mapped read-only: nothing is copied at start, and memory that can be given back cannot run the
    host out, which page-locked copies beside the tables could. The reference's Expert.forward runs unchanged on
    one Expert of the reference on the card, whose six parameters receive the called expert's bytes before the
    call and are overwritten by the next;
  * the two n-gram tables stay memory-mapped as well. A lookup gathers the rows it needs on the host, moves them
    into a table of just those rows built by the reference's class on the card, and the reference's own forward
    reads and dequantizes them through renumbered row ids;
  * with --attention-card N the reference's sparse attention kernel, unchanged, runs in a child process on card N
    (reference_attention_helper.py says why: the kernel needs more shared memory than an RTX PRO 6000 allows);
  * the reference's two activation quantizations (act_quant and fp4_act_quant) are held to their DEFINITIONS at
    every call (reference_quantization.py says why and how): as tilelang 0.1.8 compiles act_quant, a warp can
    overwrite another's output with raw input bytes, so the input of every quantized linear is taken from the
    definition; the reference's kernel runs beside it, and each of its bytes must be the definition's or the
    byte that fault leaves. The in-place calls run the reference's kernels and must leave the definition's bits.
Checked, and the tool stops: shape and dtype of every tensor against its parameter; the sum of the bytes of every
tensor moved to the card against the same sum of the shard's bytes (at load, and at every later move of an expert
or of table rows); one expert "on the card for good" against "moved in for the call" on random input, bit for bit.
Every kernel is compiled by a short pass that is not timed.
With --weights-gguf FILE (modes forward and generate) every weight of a pass comes from a GGUF quantization of the
same model instead, once the checkpoint is loaded and the file has been checked against it: gguf_weights.py says
how the reference takes them, what --gguf-linear-inputs chooses and what such a capture does and does not show.
With --gguf-control as well, nothing is taken from the file: the same stand-in multiplies the CHECKPOINT's own
values, which is the floor a GGUF capture's distance from the reference is read against. --gguf-take names the
parts taken from the file instead of all four (trunk, parameters, tables, experts, separated by commas) and
--gguf-expert-layers the layers whose routed experts are (0..13,20 for instance); the rest keeps the checkpoint's
values through the same stand-in, so that a capture shows what one part of the file costs.

FOLDER receives tensors.bin and manifest.json in the format of port_oracle_capture.py (check_capture_folders.py
reads both), with --self-test one folder per mode under FOLDER, and reference_tokenizer/: the two files convert.py
copies beside the converted weights, all that generate.py loads the tokenizer from. What numpy has no dtype for is
written as follows, and the record says so in "reference_dtype": bfloat16 values as float32 (exact), float8_e4m3fn
and float8_e8m0fnu values as their bytes (uint8). A run that fails leaves a manifest that says "complete": false.
Printed: the time of every stage and of every layer (with the moves of its experts, which the first pass over a
prompt reads from the drive), each capture's own peak card memory, the peak host memory; no path but the given ones.
Measurement tool: not part of the engine.
"""
import argparse
import collections
import contextlib
import dataclasses
import functools
import glob
import hashlib
import importlib
import json
import math
import os
import re
import resource
import shutil
import sys
import time
import traceback
import types
import warnings

import numpy
import torch

import reference_attention_helper
import reference_quantization

# ---- The capture folder: the format of port_oracle_capture.py -------------------------------------------------------
TENSORS_FILE_NAME = "tensors.bin"
MANIFEST_FILE_NAME = "manifest.json"
REQUIRED_BYTE_ORDER = "little"               # tensors.bin holds the host's bytes as they are
MANIFEST_INDENT_SPACES = 1
NUMPY_NAME_BY_DTYPE = {torch.float16: "float16", torch.float32: "float32", torch.float64: "float64",
                       torch.int8: "int8", torch.uint8: "uint8", torch.int16: "int16", torch.int32: "int32",
                       torch.int64: "int64"}
CODE_BYTE_DTYPE_NAMES = {torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e8m0fnu: "float8_e8m0fnu"}
EXIT_CODE_SUCCESS, EXIT_CODE_FAILURE = 0, 1

# ---- The checkpoint and the reference inside it ---------------------------------------------------------------------
INFERENCE_FOLDER_NAME = "inference"
ENCODING_FOLDER_NAME = "encoding"
MODEL_ARGUMENTS_FILE_NAME = "config.json"    # of the inference folder: what generate.py's --config is given
TOKENIZER_FOLDER_NAME, TOKENIZER_FILE_NAMES = "reference_tokenizer", ("tokenizer.json", "tokenizer_config.json")
REFERENCE_MODULE_NAMES = ("model", "kernel", "engram", "vision", "image_processor", "generate", "encoding")
SHARD_PATTERN = "*.safetensors"
SHARD_INDEX_FILE_NAME = "model.safetensors.index.json"
SHARD_HEADER_LENGTH_BYTES = 8                # a shard starts with the length of its JSON header
SHARD_METADATA_KEY = "__metadata__"
SHARD_DTYPES = {"F32": (torch.float32, 4), "BF16": (torch.bfloat16, 2), "F8_E4M3": (torch.float8_e4m3fn, 1),
                "F8_E8M0": (torch.float8_e8m0fnu, 1), "I8": (torch.int8, 1)}       # torch dtype, bytes of one element
BLOCK_SCALED_WEIGHT_SUFFIX = "wo_a.weight"   # convert.py:157: stored multiplied by its block scales ...
BLOCK_SCALES_SUFFIX = "wo_a.scale"           # ... which convert.py:159 then leaves out
CONVERT_BLOCK_SIZES = ((32, 32), (128, 128)) # convert.py:164
ROUTED_EXPERT_DTYPE = torch.float4_e2m1fn_x2 # model.py:872: what MoE gives its routed experts and not its shared one
ROUTED_EXPERT_DTYPE_NAME = "fp4"             # ModelArgs.expert_dtype of the checkpoint's own format (model.py:55)
QUANTIZED_WEIGHT_DTYPES = (torch.float4_e2m1fn_x2, torch.float8_e4m3fn)            # the branches of model.py:186-205
PLACE_CARD, PLACE_HOST_EXPERT, PLACE_HOST_TABLE = "card", "host expert", "host table"   # where a tensor's bytes stay
LAYER_WEIGHT_NAME = re.compile(r"layers\.(?P<layer>\d+)\.(?P<name_in_layer>.+)\.weight")
ROUTED_EXPERT_LINEAR_NAME = re.compile(r"ffn\.experts\.(?P<expert>\d+)\.w\d+")
DRAFT_BLOCK_NAME_PARTS = 2                   # "mtp" and the block's number: what convert.py:109 splits off
NUMBER_IN_A_NAME = re.compile(r"\d+")
# What each published model.py does where a check of these tools depends on it, by the file's sha256 (git blobs
# 3e7dc222 of 2026-09-10, 4c98d615 of 2026-10-12 08:37, 46cc8310 of 09:38, which 1017 ships): whether every owner of
# index keys puts them in the shared slot at every step, not only where it emits (46cc8310:548-549), and whether the
# attention output is rounded to E4M3 per 32 values with a power-of-two scale before wo_a (46cc8310:787, :1073).
ReferenceForm = collections.namedtuple("ReferenceForm", ("publishes_index_keys", "rounds_attention_output"))
REFERENCE_FORMS = {"4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65": ReferenceForm(False, False),
                   "d70d25b369bed39e1077ff465224af54b7a01349383d9b65008f10e7637d1812": ReferenceForm(True, False),
                   "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d": ReferenceForm(True, True)}

# ---- How generate.py runs the model ---------------------------------------------------------------------------------
CARD_DEVICE = "cuda"                         # generate.py:130 and :134: the current card, set as generate.py:116 does
REFERENCE_DEFAULT_DTYPE = torch.bfloat16     # generate.py:118
GREEDY_TEMPERATURE = 0.0                     # model.py:1288-1289: sample() returns the argmax
SEQUENCES = 1                                # generate.py:125
DEFAULT_MAXIMUM_SEQUENCE_TOKENS = 65536      # generate.py:126
DEFAULT_THINKING_MODE = "chat"               # generate.py:207

# ---- This tool ------------------------------------------------------------------------------------------------------
DEFAULT_LINEAR_LAYERS = "0,1,14,20,39"       # first, both engram layers, the first decoder layer, last
DEFAULT_EXPERT_CALLS = 64
EXPERT_ROWS_PER_CALL_LIMIT = 4               # rows recorded of one expert call, so that the pairs span many experts
DEFAULT_TOP_K = 16
GGUF_LINEAR_INPUTS = ("unquantized", "as-the-reference",   # gguf_weights.py: what a quantized linear multiplies
                      "int8-per-256", "int8-per-32")
GGUF_PARTS = ("trunk", "parameters", "tables", "experts")   # gguf_weights.py: what a pass can take from the file
ENDS_OF_A_RANGE = (1, 2)                     # --gguf-expert-layers: a layer, or first..last
HIGHEST_LAYER_NUMBER_READ = 1023             # more than any model has: a typed range is never expanded beyond it
RUNS_WITH_TWICE = 2
WARM_UP_PREFILL_TOKENS = 8                   # whole groups of every compressor, so every indexer runs
WARM_UP_DECODE_STEPS = 2                     # one step that leaves a group of two open and one that completes it
SELF_CHECK_ROWS = 33                         # one more than the 32 rows the kernels take at a time
SELF_CHECK_SEED = 20261010
SELF_TEST_MODES = ("forward", "linears", "generate")
SELF_TEST_TOKENS = 64
SELF_TEST_NEW_TOKENS = 8
SELF_TEST_EXPERT_CALLS = 4
SELF_TEST_PROMPT = (
    "Explain, step by step and in plain words, how a hash table that resolves collisions by open addressing with "
    "linear probing inserts a key, finds a key and deletes a key. Say what a tombstone is and why deleting without "
    "one breaks later searches, how the load factor changes the expected number of probes, and when the table "
    "should grow. Then write a small C function that looks a key up in such a table of 64-bit integers, with every "
    "bound checked, and describe one test that would catch an off-by-one error in the wrap-around at the end of "
    "the table.")
STAGE_COUNTERS = {"moves": "experts_moved", "moved_bytes": "expert_bytes_moved", "move_seconds": "expert_move_seconds"}
STREAMS_RECORDED_PER_LAYER = 2               # the streams that enter a block and the streams that leave it
FLOAT32_BYTES = 4
NAMES_IN_A_MESSAGE = 8
BYTES_PER_GIB = 2 ** 30
RESIDENT_SET_UNIT_BYTES = 1024               # getrusage reports the peak resident set in KiB on Linux

Slot = collections.namedtuple("Slot", ("place", "shape", "dtype", "target"))
ShardTensor = collections.namedtuple("ShardTensor", ("shard_index", "dtype_name", "shape", "byte_offset", "byte_count"))


def byte_sum(tensor):
    """The checksum of the self-checks: the sum of a tensor's bytes, as a 0-dimensional tensor where it lives."""
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).sum(dtype = torch.int64)


def same_bits(first, second):
    return first.shape == second.shape and first.dtype == second.dtype and torch.equal(
        first.contiguous().reshape(-1).view(torch.uint8), second.contiguous().reshape(-1).view(torch.uint8))


def synchronized_clock():
    torch.cuda.synchronize()
    return time.perf_counter()


@contextlib.contextmanager
def timed_stage(stage_seconds, name):
    started = time.perf_counter()
    yield
    stage_seconds[name] = synchronized_clock() - started
    print(f"{name}: {stage_seconds[name]:.3f} s", flush = True)


class CaptureWriter:
    """tensors.bin, written tensor by tensor as each is captured, and manifest.json, written when the capture ends:
    port_oracle_capture.py's writer, which here also keeps the time it took and knows the reference's dtypes."""

    def __init__(self, output_directory):
        os.makedirs(output_directory, exist_ok = True)
        self.manifest_path = os.path.join(output_directory, MANIFEST_FILE_NAME)
        self.tensors_path = os.path.join(output_directory, TENSORS_FILE_NAME)
        self.tensors_file = open(self.tensors_path, "xb")                  # refuses to overwrite a capture
        self.records, self.results, self.names, self.byte_offset, self.seconds = [], {}, set(), 0, 0.0
        self.manifest = {"complete": False, "conditions": None, "results": self.results, "tensors": self.records}

    def add(self, name, tensor, layer, description, **facts):
        started = time.perf_counter()
        if name in self.names:
            raise RuntimeError(f"tensor recorded twice: {name}")
        tensor = tensor.detach().cpu()
        if tensor.dtype == torch.bfloat16:                                 # every bfloat16 value is a float32 value
            tensor, facts = tensor.float(), {**facts, "reference_dtype": "bfloat16"}
        elif tensor.dtype in CODE_BYTE_DTYPE_NAMES:
            tensor, facts = tensor.contiguous().view(torch.uint8), \
                {**facts, "reference_dtype": CODE_BYTE_DTYPE_NAMES[tensor.dtype]}
        if tensor.dtype not in NUMPY_NAME_BY_DTYPE:
            raise RuntimeError(f"{name}: no numpy name for {tensor.dtype}")
        values = tensor.contiguous().numpy()
        values.tofile(self.tensors_file)
        self.records.append({"name": name, "dtype": NUMPY_NAME_BY_DTYPE[tensor.dtype], "shape": list(values.shape),
                             "byte_offset": self.byte_offset, "byte_count": values.nbytes, "layer": layer,
                             "description": description, **facts})
        self.names.add(name)
        self.byte_offset += values.nbytes
        self.seconds += time.perf_counter() - started

    def close(self, complete):
        self.tensors_file.close()
        bytes_on_disk = os.path.getsize(self.tensors_path)
        self.manifest["complete"] = complete and bytes_on_disk == self.byte_offset
        with open(self.manifest_path, "x", encoding = "utf8") as handle:
            json.dump(self.manifest, handle, indent = MANIFEST_INDENT_SPACES)
        if bytes_on_disk != self.byte_offset:
            raise RuntimeError(f"{TENSORS_FILE_NAME} holds {bytes_on_disk} bytes, {self.byte_offset} were recorded")


# ---- What stands in the reference model for the parts that do not fit the card ---------------------------------------
class ExpertStage:
    """The place on the card where a routed expert stands while it is called: one Expert of the routed shape, built
    by the reference, whose parameters receive the bytes of whichever expert the reference calls next."""

    def __init__(self, build_expert):
        self.build_expert = build_expert
        self.expert = build_expert().requires_grad_(False)                 # its bytes are replaced before every call
        self.slots = dict(self.expert.named_parameters())                  # w1.weight, w1.scale, ... (model.py:836-838)
        self.occupant_name, self.moves, self.moved_bytes, self.move_seconds = None, 0, 0, 0.0

    def run(self, host_expert, rows, route_weights):
        started = time.perf_counter()
        for slot_name, parameter in self.slots.items():
            parameter.copy_(host_expert.host_tensors[slot_name])
        if int(sum(byte_sum(parameter) for parameter in self.slots.values())) != host_expert.host_byte_sum():
            raise RuntimeError(f"{host_expert.checkpoint_name}: the bytes on the card are not the bytes of the shard")
        self.occupant_name = host_expert.checkpoint_name
        self.moves, self.moved_bytes = self.moves + 1, self.moved_bytes + host_expert.byte_count
        self.move_seconds += time.perf_counter() - started
        return self.expert(rows, route_weights)                            # model.py:841-851, the reference's forward


class HostResidentExpert(torch.nn.Module):
    """Stands where the reference builds a routed Expert (model.py:873-885). Its three matrices and their scales
    stay in host memory, as read-only views of the shard's bytes; a call moves them to the stage and runs the
    reference's Expert there. It holds no parameter: the loader fills host_bytes and host_tensors by slot name."""

    def __init__(self, stage):
        super().__init__()
        self.stage = stage                                                 # not a module: nothing of it is registered
        self.checkpoint_name, self.host_bytes, self.host_tensors, self.byte_count, self.sum_read = None, {}, {}, 0, None

    def host_byte_sum(self):
        if self.sum_read is None:                                          # read from the shard once, kept after
            self.sum_read = sum(int(byte_sum(raw)) for raw in self.host_bytes.values())
        return self.sum_read

    def forward(self, rows, route_weights = None):                         # as model.py:841 is called
        return self.stage.run(self, rows, route_weights)


class HostResidentTable(torch.nn.Module):
    """Stands where the reference builds an n-gram table (model.py:343). The values and the scales stay in the
    shard, memory-mapped; a lookup gathers the rows it needs there, moves them into a table of just those rows that
    the reference's class builds on the card, and the reference's forward (model.py:312-325) reads and dequantizes
    them through the renumbered row ids: the same rows through the same arithmetic."""

    def __init__(self, table_class, row_count, row_width):                 # after the class, as model.py:299
        super().__init__()
        self.table_class, self.row_count, self.row_width = table_class, row_count, row_width
        self.checkpoint_name, self.host_rows, self.rows_moved = None, {}, 0

    def forward(self, indices):
        if int(indices.min()) < 0 or int(indices.max()) >= self.row_count:
            # with one rank the reference holds every row (model.py:303-305); it would answer zeros here (:313-321)
            raise RuntimeError(f"{self.checkpoint_name}: a row id outside the table's {self.row_count} rows")
        row_ids, renumbered = torch.unique(indices, return_inverse = True)
        host_row_ids = row_ids.cpu().numpy()
        with torch.device(indices.device):
            rows_on_the_card = self.table_class(len(host_row_ids), self.row_width).requires_grad_(False)
        for slot_name, parameter in rows_on_the_card.named_parameters():
            gathered = torch.from_numpy(self.host_rows[slot_name][host_row_ids])
            parameter.copy_(gathered.view(parameter.dtype))
            if int(byte_sum(parameter)) != int(byte_sum(gathered)):
                raise RuntimeError(f"{self.checkpoint_name}.{slot_name}: the rows on the card are not the shard's")
        self.rows_moved += len(host_row_ids)
        return rows_on_the_card(renumbered)


def build_model(reference_model, model_arguments, tokenizer):
    """The reference's Transformer, built on the card as generate.py:130-131 builds it. While it is built, the two
    names its constructors look up for the parts that do not fit the card answer with the stand-ins above."""
    stages = {}
    expert_class, table_class = reference_model.Expert, reference_model.ParallelEngramEmbedding

    def make_expert(width, inner_width, dtype = None, swiglu_limit = 0.0): # as model.py:834 is called (:875, :887)
        if dtype != ROUTED_EXPERT_DTYPE:                                   # the shared expert: the reference's own
            return expert_class(width, inner_width, dtype = dtype, swiglu_limit = swiglu_limit)
        shape = (width, inner_width, swiglu_limit)
        if shape not in stages:
            stages[shape] = ExpertStage(functools.partial(expert_class, width, inner_width, dtype = dtype,
                                                          swiglu_limit = swiglu_limit))
        return HostResidentExpert(stages[shape])

    reference_model.Expert = make_expert
    reference_model.ParallelEngramEmbedding = functools.partial(HostResidentTable, table_class)
    try:
        with torch.device(CARD_DEVICE):
            model = reference_model.Transformer(model_arguments, tokenizer)
    finally:
        reference_model.Expert, reference_model.ParallelEngramEmbedding = expert_class, table_class
    if (reference_model.world_size, reference_model.rank) != (1, 0):
        raise RuntimeError("the reference did not build for one rank (model.py:1189-1190)")
    return model, list(stages.values()), table_class


# ---- Loading straight from the shards --------------------------------------------------------------------------------
def read_shard_headers(checkpoint_directory):
    """Every tensor of the checkpoint by its stored name: dtype, shape and where its bytes lie, from the shards' own
    headers. A name stored twice, a dtype this tool does not map, a size that does not agree with the shape or a
    set of names that differs from the index file (convert.py:86-87 and :148-152) stops the tool."""
    shard_paths = sorted(glob.glob(os.path.join(checkpoint_directory, SHARD_PATTERN)))
    if not shard_paths:
        raise RuntimeError(f"no {SHARD_PATTERN} file in the checkpoint folder")
    tensors = {}
    for shard_index, shard_path in enumerate(shard_paths):
        shard_name, shard_bytes = os.path.basename(shard_path), os.path.getsize(shard_path)
        with open(shard_path, "rb") as handle:
            header_bytes = int.from_bytes(handle.read(SHARD_HEADER_LENGTH_BYTES), REQUIRED_BYTE_ORDER)
            if header_bytes > shard_bytes - SHARD_HEADER_LENGTH_BYTES:
                raise RuntimeError(f"{shard_name}: a header of {header_bytes} bytes in a file of {shard_bytes}")
            header = json.loads(handle.read(header_bytes))
        data_offset = SHARD_HEADER_LENGTH_BYTES + header_bytes
        for stored_name, entry in header.items():
            if stored_name == SHARD_METADATA_KEY:
                continue
            first_byte, end_byte = entry["data_offsets"]
            if stored_name in tensors or entry["dtype"] not in SHARD_DTYPES:
                raise RuntimeError(f"{stored_name} in {shard_name}: stored twice, or of a dtype this tool does not "
                                   f"map ({entry['dtype']})")
            if not 0 <= first_byte <= end_byte <= shard_bytes - data_offset \
                    or end_byte - first_byte != math.prod(entry["shape"]) * SHARD_DTYPES[entry["dtype"]][1]:
                raise RuntimeError(f"{stored_name}: bytes {first_byte} to {end_byte} of {shard_name} do not hold "
                                   f"{entry['dtype']} {entry['shape']}")
            tensors[stored_name] = ShardTensor(shard_index, entry["dtype"], tuple(entry["shape"]),
                                               data_offset + first_byte, end_byte - first_byte)
    index_path = os.path.join(checkpoint_directory, SHARD_INDEX_FILE_NAME)
    if os.path.exists(index_path):
        with open(index_path, encoding = "utf8") as handle:
            indexed_names = set(json.load(handle)["weight_map"])
        if indexed_names != set(tensors):
            raise RuntimeError(f"the shards and {SHARD_INDEX_FILE_NAME} differ in {len(indexed_names ^ set(tensors))} "
                               f"names, first {sorted(indexed_names ^ set(tensors))[:NAMES_IN_A_MESSAGE]}")
    return shard_paths, tensors


def converted_name(stored_name):
    """The name convert.py stores a checkpoint tensor under, or None for a tensor it drops (convert.py:104-115).
    Its lines 116-146 change nothing for one rank: every key of its `mapping` maps to itself, and a tensor cut into
    one piece per rank is the whole tensor."""
    name = stored_name.removeprefix("model.")                              # convert.py:105-106
    if name.startswith("mtp.") and name.split(".", DRAFT_BLOCK_NAME_PARTS)[-1] in ("embed.weight", "head.weight"):
        return None                                                        # convert.py:109-110: tied to the backbone's
    name = name.replace("self_attn", "attn")                               # convert.py:111
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")                                  # convert.py:112-113
    return name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")   # convert.py:114-115


def multiplied_by_block_scales(name, weight, scale):
    """What convert.py stores for an attention output projection (convert.py:157-173): every value times the scale
    of its block in float32, on the processor, rounded to bfloat16. Said here in other operations with the same
    two operands of every product, so the same bits."""
    if weight.dim() != scale.dim() or weight.shape[0] % scale.shape[0] or weight.shape[1] % scale.shape[1]:
        raise RuntimeError(f"{name}: scales of shape {tuple(scale.shape)} for values of shape {tuple(weight.shape)}")
    block_sizes = (weight.shape[0] // scale.shape[0], weight.shape[1] // scale.shape[1])
    if block_sizes not in CONVERT_BLOCK_SIZES:
        raise RuntimeError(f"{name}: blocks of {block_sizes}, convert.py takes {CONVERT_BLOCK_SIZES}")
    scales = scale.float().repeat_interleave(block_sizes[0], dim = 0).repeat_interleave(block_sizes[1], dim = 1)
    return (weight.float() * scales).to(torch.bfloat16)


def expected_slots(model, table_class):
    """Everything the checkpoint has to fill, by the name convert.py would store it under: the parameters the
    reference model declares, and what its stand-ins declare in the reference's own shapes and dtypes."""
    slots = {name: Slot(PLACE_CARD, tuple(parameter.shape), parameter.dtype, parameter)
             for name, parameter in model.named_parameters(remove_duplicate = False)}
    for module_name, module in model.named_modules():
        if isinstance(module, HostResidentExpert):
            place, rows, declared = PLACE_HOST_EXPERT, None, module.stage.slots
        elif isinstance(module, HostResidentTable):                        # a table of one row gives the row shapes
            place, rows = PLACE_HOST_TABLE, module.row_count               # and dtypes of model.py:309-310; with one
            declared = dict(table_class(1, module.row_width).named_parameters())   # rank every row is held (:303)
        else:
            continue
        module.checkpoint_name = module_name
        for slot_name, parameter in declared.items():
            shape = tuple(parameter.shape) if rows is None else (rows, *parameter.shape[1:])
            slots[f"{module_name}.{slot_name}"] = Slot(place, shape, parameter.dtype, (module, slot_name))
    return slots


def widening(name, stored_dtype, stored_shape, slot):
    """Whether a tensor reaches its parameter widened from bfloat16 to float32, as the copy of the reference's load
    widens it (model.py:446, :448, :1006, :1093, vision.py:28), and not bit for bit. Anything else stops the tool."""
    if tuple(stored_shape) != slot.shape:
        raise RuntimeError(f"{name}: the checkpoint holds shape {tuple(stored_shape)}, the model declares {slot.shape}")
    if stored_dtype == slot.dtype:
        return False
    if slot.place == PLACE_HOST_EXPERT and stored_dtype == torch.int8 and slot.dtype == ROUTED_EXPERT_DTYPE:
        return False                                                       # convert.py:174-181: the same bytes, renamed
    if slot.place == PLACE_CARD and stored_dtype == torch.bfloat16 and slot.dtype == torch.float32:
        return True
    raise RuntimeError(f"{name}: the checkpoint holds {stored_dtype}, the model declares {slot.dtype}")


def load_checkpoint(slots, shard_paths, tensors):
    """Fills every slot from the shards and checks what reached the card against the shard's bytes."""
    shard_maps = [numpy.memmap(shard_path, dtype = numpy.uint8, mode = "r") for shard_path in shard_paths]
    names, dropped = {}, []
    for stored_name in tensors:
        name = converted_name(stored_name)
        if name is None:
            dropped.append(stored_name)
        elif name in names:
            raise RuntimeError(f"{stored_name} and {names[name]} are both convert.py's {name}")
        else:
            names[name] = stored_name

    def read(name):                                                        # fresh memory: the bytes, and the tensor
        stored = tensors[names[name]]
        raw = torch.from_numpy(numpy.fromfile(shard_paths[stored.shard_index], dtype = numpy.uint8,
                                              count = stored.byte_count, offset = stored.byte_offset))
        if raw.numel() != stored.byte_count:
            raise RuntimeError(f"{names[name]}: {raw.numel()} of {stored.byte_count} bytes could be read")
        return raw, raw.view(SHARD_DTYPES[stored.dtype_name][0]).view(stored.shape)

    filled, counts = set(), collections.Counter()
    for name in sorted(names):
        stored = tensors[names[name]]
        if name.endswith(BLOCK_SCALES_SUFFIX) \
                and name.removesuffix(BLOCK_SCALES_SUFFIX) + BLOCK_SCALED_WEIGHT_SUFFIX in names:
            continue                                                       # read with its values, below
        if name not in slots:
            raise RuntimeError(f"checkpoint tensor {names[name]} (convert.py's {name}) is no parameter of the model")
        slot = slots[name]
        if slot.place == PLACE_CARD:
            raw, value = read(name)
            source_sum = int(byte_sum(raw))
            if name.endswith(BLOCK_SCALED_WEIGHT_SUFFIX):
                scale_name = name.removesuffix(BLOCK_SCALED_WEIGHT_SUFFIX) + BLOCK_SCALES_SUFFIX
                if scale_name not in names:
                    raise RuntimeError(f"checkpoint tensor {names[name]} has no block scales (convert.py:159)")
                value = multiplied_by_block_scales(name, value, read(scale_name)[1])
                source_sum = int(byte_sum(value))                          # of what convert.py would have stored
                counts["tensors_multiplied_by_their_block_scales"] += 1
            widened = widening(name, value.dtype, value.shape, slot)
            slot.target.copy_(value)
            if int(byte_sum(slot.target.to(value.dtype) if widened else slot.target)) != source_sum:
                raise RuntimeError(f"{name}: the bytes on the card are not the bytes of the shard")
            filled.add(id(slot.target))                                    # by identity: tied names share a parameter
            counts.update(tensors_on_the_card = 1, tensors_widened_from_bfloat16_to_float32 = int(widened),
                          bytes_on_the_card = slot.target.numel() * slot.target.element_size())
        else:
            widening(name, SHARD_DTYPES[stored.dtype_name][0], stored.shape, slot)
            module, slot_name = slot.target
            extent = slice(stored.byte_offset, stored.byte_offset + stored.byte_count)
            mapped = numpy.asarray(shard_maps[stored.shard_index][extent]) # the shard's own bytes, read-only
            if slot.place == PLACE_HOST_EXPERT:
                module.host_bytes[slot_name] = torch.from_numpy(mapped)
                module.host_tensors[slot_name] = module.host_bytes[slot_name].view(slot.dtype).view(slot.shape)
                module.byte_count += stored.byte_count
            else:
                module.host_rows[slot_name] = mapped.reshape(slot.shape)
            filled.add(name)
            counts.update(tensors_in_host_memory = 1, bytes_in_host_memory = stored.byte_count)
    unfilled = [name for name, slot in slots.items()
                if (id(slot.target) if slot.place == PLACE_CARD else name) not in filled]
    if unfilled:
        raise RuntimeError(f"{len(unfilled)} parameters of the model are not in the checkpoint, first "
                           f"{unfilled[:NAMES_IN_A_MESSAGE]}")
    return {"shards": len(shard_paths), "tensors": len(tensors), **dict(sorted(counts.items())),
            "tensors_dropped_as_convert_drops_them": sorted(dropped)}


# ---- The reference, as generate.py sets it up ------------------------------------------------------------------------
def import_reference(checkpoint_directory):
    """The reference's modules, imported from the checkpoint itself and from nowhere else, with their digests."""
    folders = [os.path.realpath(os.path.join(checkpoint_directory, name))
               for name in (INFERENCE_FOLDER_NAME, ENCODING_FOLDER_NAME)]
    sys.dont_write_bytecode = True                                         # nothing is written into the checkpoint
    sys.path.insert(0, folders[0])
    try:                                                                   # generate.py:15 adds the encoding folder
        reference_model, reference_generate = importlib.import_module("model"), importlib.import_module("generate")
    except ImportError as failure:
        raise RuntimeError(f"the reference cannot be imported: {failure}") from failure
    digests = {}
    for module_name in REFERENCE_MODULE_NAMES:
        module_path = os.path.abspath(sys.modules[module_name].__file__)   # the folder decides: the files of a download
        if os.path.realpath(os.path.dirname(module_path)) not in folders:  # may be links into a store of their own
            raise RuntimeError(f"module {module_name} was not imported from the checkpoint's reference")
        with open(module_path, "rb") as handle:
            digests[os.path.basename(module_path)] = hashlib.sha256(handle.read()).hexdigest()
    return types.SimpleNamespace(model = reference_model, generate = reference_generate,
                                 encoding = sys.modules["encoding"], file_digests = digests)


def reference_form(reference):
    """REFERENCE_FORMS' entry of the imported model.py; another model.py stops the tool that asks, whose checks
    would not know what to expect of it: read its changes and add it first."""
    digest = reference.file_digests.get("model.py")
    if digest not in REFERENCE_FORMS:
        raise RuntimeError(f"model.py of sha256 {digest} is no version whose index-key slot and attention output "
                           "these tools know (REFERENCE_FORMS)")
    return REFERENCE_FORMS[digest]


def read_token_ids(arguments, reference, tokenizer, model_arguments):
    """The token ids of the run: a sequence of --tokens-json as it is, or a case of --prompt-json through the
    reference's own prepare_case (generate.py:90-96: encoding.py's encode_case, then the checkpoint's tokenizer)."""
    if arguments.tokens_json is not None:
        with open(arguments.tokens_json, encoding = "utf8") as handle:
            entries = json.load(handle)
    else:
        entries = [{"messages": [{"role": "user", "content": SELF_TEST_PROMPT}]}] if arguments.self_test \
            else reference.encoding.load_cases(arguments.prompt_json)      # generate.py:165
    if not isinstance(entries, list) or not arguments.sequence < len(entries):
        raise RuntimeError(f"the JSON file holds no entry {arguments.sequence}")
    entry = entries[arguments.sequence]
    if arguments.tokens_json is not None:
        token_ids = entry.get("token_ids") if isinstance(entry, dict) else entry
    else:
        _, token_ids, _, images = reference.generate.prepare_case(entry, arguments.thinking_mode, tokenizer,
                                                                  model_arguments)
        if images is not None:
            raise RuntimeError("the prompt holds images: this tool captures text prompts only")
    if not isinstance(token_ids, list) or not token_ids \
            or not all(type(token_id) is int and 0 <= token_id < model_arguments.vocab_size for token_id in token_ids):
        raise RuntimeError(f"entry {arguments.sequence} is not a list of token ids below {model_arguments.vocab_size}")
    if arguments.tokens is not None:
        if len(token_ids) < arguments.tokens:
            raise RuntimeError(f"the sequence has {len(token_ids)} tokens, --tokens asks for {arguments.tokens}")
        token_ids = token_ids[:arguments.tokens]
    new_tokens = arguments.new_tokens if "generate" in arguments.modes else 0
    if len(token_ids) + new_tokens > arguments.maximum_sequence_tokens:
        raise RuntimeError(f"{len(token_ids)} tokens and {new_tokens} new ones do not fit --maximum-sequence-tokens "
                           f"{arguments.maximum_sequence_tokens}")
    return token_ids


def check_staged_expert(model):
    """One routed expert on the card for good against the same expert moved in for the call, on random input: the
    same bits, also after another expert has stood on the stage in between."""
    host_experts = (module for module in model.modules() if isinstance(module, HostResidentExpert))
    first, second = next(host_experts), next(host_experts)
    with torch.device(CARD_DEVICE):
        resident = first.stage.build_expert().requires_grad_(False)
    for slot_name, parameter in resident.named_parameters():
        parameter.copy_(first.host_tensors[slot_name])
    if int(sum(byte_sum(parameter) for parameter in resident.parameters())) != first.host_byte_sum():
        raise RuntimeError(f"{first.checkpoint_name}: the bytes on the card are not the bytes of the shard")
    generator = torch.Generator(device = CARD_DEVICE).manual_seed(SELF_CHECK_SEED)
    rows = torch.randn(SELF_CHECK_ROWS, resident.w1.in_features, generator = generator, device = CARD_DEVICE,
                       dtype = torch.float32).to(REFERENCE_DEFAULT_DTYPE)
    route_weights = torch.rand(SELF_CHECK_ROWS, 1, generator = generator, device = CARD_DEVICE, dtype = torch.float32)
    with torch.inference_mode():                                           # as the reference's forward (model.py:1241)
        expected, staged, other = resident(rows, route_weights), first(rows, route_weights), second(rows, route_weights)
        passed = bool(torch.isfinite(expected.float()).all()) and bool(expected.float().abs().sum() > 0) \
            and same_bits(expected, staged) and same_bits(expected, first(rows, route_weights)) \
            and not same_bits(expected, other)
    if not passed:
        raise RuntimeError(f"{first.checkpoint_name}: moved in for the call it does not give the bits it gives on "
                           f"the card for good, or its output is zero, not finite or {second.checkpoint_name}'s")


def warm_up(model, token_ids):
    """Runs every kernel of the prompt path and of the decode path once, so that none is compiled in a timed pass."""
    count = WARM_UP_PREFILL_TOKENS + WARM_UP_DECODE_STEPS
    tokens = torch.tensor([[token_ids[index % len(token_ids)] for index in range(count)]], dtype = torch.long,
                          device = CARD_DEVICE)
    model.forward(tokens[:, :WARM_UP_PREFILL_TOKENS], 0)
    for position in range(WARM_UP_PREFILL_TOKENS, count):
        model.forward(tokens[:, position:position + 1], position)


def open_session(arguments, stage_seconds):
    """Everything up to the first capture: the reference imported, the model built and loaded, the checks passed."""
    if not torch.cuda.is_available() or arguments.card >= torch.cuda.device_count():
        raise RuntimeError(f"there is no card {arguments.card}")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("float32 products are set to reduced precision (TF32): the reference runs without it")
    reference = import_reference(arguments.checkpoint)
    quantization_counts = reference_quantization.install(reference.model)
    if arguments.attention_card is not None:
        reference_attention_helper.install(reference.model, arguments.checkpoint, arguments.attention_card)
    torch.cuda.set_device(arguments.card)                                  # generate.py:116
    torch.set_default_dtype(REFERENCE_DEFAULT_DTYPE)                       # generate.py:118
    with open(os.path.join(arguments.checkpoint, INFERENCE_FOLDER_NAME, MODEL_ARGUMENTS_FILE_NAME),
              encoding = "utf8") as handle:
        model_arguments = reference.model.ModelArgs(**json.load(handle))   # generate.py:121-122
    model_arguments.temperature = GREEDY_TEMPERATURE                       # generate.py:123
    model_arguments.max_batch_size = SEQUENCES
    model_arguments.max_seq_len = arguments.maximum_sequence_tokens
    if model_arguments.expert_dtype != ROUTED_EXPERT_DTYPE_NAME:
        raise RuntimeError(f"expert_dtype is {model_arguments.expert_dtype!r}: this tool runs the checkpoint's own "
                           f"{ROUTED_EXPERT_DTYPE_NAME} experts")
    if "linears" in arguments.modes and arguments.layer_list[-1] >= model_arguments.n_layers:
        raise RuntimeError(f"--layers names layer {arguments.layer_list[-1]} of {model_arguments.n_layers}")
    tokenizer_folder = os.path.join(arguments.output_directory, TOKENIZER_FOLDER_NAME)   # generate.py:128 loads the
    os.makedirs(tokenizer_folder, exist_ok = True)       # tokenizer from the converted folder, which holds these two
    for file_name in TOKENIZER_FILE_NAMES:               # files of the checkpoint and no other (convert.py:184-189)
        shutil.copyfile(os.path.join(arguments.checkpoint, file_name), os.path.join(tokenizer_folder, file_name))
    tokenizer = reference.generate.AutoTokenizer.from_pretrained(tokenizer_folder, local_files_only = True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("the checkpoint's tokenizer names no end token (generate.py:157)")
    token_ids = read_token_ids(arguments, reference, tokenizer, model_arguments)
    print(f"{len(token_ids)} tokens, the first {token_ids[:NAMES_IN_A_MESSAGE]}", flush = True)
    with timed_stage(stage_seconds, "model construction"):
        model, stages, table_class = build_model(reference.model, model_arguments, tokenizer)
    with timed_stage(stage_seconds, "trunk load"), torch.no_grad():
        shard_paths, tensors = read_shard_headers(arguments.checkpoint)
        load_summary = load_checkpoint(expected_slots(model, table_class), shard_paths, tensors)
    print(", ".join(f"{count} {what.replace('_', ' ')}" for what, count in load_summary.items()
                    if isinstance(count, int)), flush = True)
    torch.set_default_device(CARD_DEVICE)                                  # generate.py:134
    with timed_stage(stage_seconds, "expert moved in against expert on the card"), torch.no_grad():
        check_staged_expert(model)
    if arguments.weights_gguf is not None:                                 # imported only here: without the option
        with timed_stage(stage_seconds, "weights from the GGUF"), torch.no_grad():   # nothing of this tool changes
            parts = () if arguments.gguf_control else arguments.gguf_parts or GGUF_PARTS
            load_summary["weights_from_the_gguf"] = importlib.import_module("gguf_weights").install(
                reference.model, model, stages, HostResidentExpert, HostResidentTable, arguments.weights_gguf,
                arguments.gguf_linear_inputs, parts, arguments.gguf_expert_layer_list)
    torch.cuda.empty_cache()                                               # what the load's checks held for a moment
    fresh_buffers = {name: buffer.clone() for name, buffer in model.named_buffers()}   # before anything has run
    with timed_stage(stage_seconds, "kernel compilation and warm-up (in no time below)"):
        warm_up(model, token_ids)
    return types.SimpleNamespace(reference = reference, model = model, model_arguments = model_arguments,
                                 tokenizer = tokenizer, stages = stages, load_summary = load_summary,
                                 fresh_buffers = fresh_buffers, quantization_counts = quantization_counts), token_ids


def forget_earlier_runs(session):
    """Puts every buffer of the model back to what it held before its first forward, as a new process would find
    it: the reference keeps caches between calls (attention, index, pooling and n-gram state) and does not clear
    them at position 0. Not shown to change a result: the runs that differed (c02, c03) differed by the
    reference's act_quant kernel (c04, c05)."""
    with torch.no_grad():
        for name, buffer in session.model.named_buffers():
            buffer.copy_(session.fresh_buffers[name])


# ---- Observing a pass ------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def observed_pass(session, writer, record_streams, keep_head_input):
    """Hooks on the reference's own modules for one pass from position 0: what a capture records of it, and the time
    of every layer without the time the records took. Nothing of the reference is replaced."""
    model, handles = session.model, []
    observation = types.SimpleNamespace(layers = [], routing = {}, head_input = None)

    def expert_moves():
        return {name: sum(getattr(stage, counter) for stage in session.stages)
                for counter, name in STAGE_COUNTERS.items()}

    def embedding_ran(_, arguments, output):
        if record_streams:
            writer.add("input.embeddings", output, None, "rows of the embedding table for the token ids "
                       "(model.py:1253): [sequence, position, hidden]")

    def row_ids_ran(hash_state, arguments, output):
        for table_index, layer in enumerate(hash_state.layout.layer_ids):
            writer.add(f"layer_{layer}.ngram.row_ids", output[:, :, table_index], layer, "rows of this layer's n-gram "
                       "table read for every position (engram.py:184): [sequence, position, n-gram size and head]")

    def watch(block):
        layer, entry = block.layer_id, {"layer": block.layer_id}

        def lookup_begins(lookup, arguments):
            entry.update(ngram_started = synchronized_clock(), ngram_rows_moved = -lookup.embed.rows_moved)

        def lookup_ends(lookup, arguments, output):
            entry["ngram_seconds"] = synchronized_clock() - entry.pop("ngram_started")
            entry["ngram_rows_moved"] += lookup.embed.rows_moved

        def block_begins(_, arguments):                                    # model.py:1267: h, start_pos, pre_mix, mask
            if record_streams:
                writer.add(f"layer_{layer}.block.input", arguments[0], layer, "residual streams entering the block, "
                           "after the layer's n-gram lookup where it has one: [sequence, position, stream, hidden]")
                writer.add(f"layer_{layer}.block.input_collapse_weights", arguments[2], layer, "weights that "
                           "collapse the entering streams for this block's attention (model.py:983); the first "
                           "block's are 1 for stream 0: [sequence, position, stream]")
            entry.update(started = synchronized_clock(), writer_seconds = writer.seconds, moves = expert_moves())

        def block_ends(_, arguments, output):
            ended = synchronized_clock()
            before = entry.pop("moves")
            entry.update({name: after - before[name] for name, after in expert_moves().items()},
                         seconds = ended - entry.pop("started") - (writer.seconds - entry.pop("writer_seconds")))
            observation.layers.append(entry)
            if record_streams:
                writer.add(f"layer_{layer}.block.output", output[0], layer, "residual streams leaving the block: "
                           "[sequence, position, stream, hidden]")
                writer.add(f"layer_{layer}.block.output_collapse_weights", output[1], layer, "weights computed at "
                           "this block's MLP site, which collapse the streams at the next site (model.py:994)")

        def gate_ran(_, arguments, output):                                # model.py:827: weights, indices
            observation.routing[layer] = output[1]
            writer.add(f"layer_{layer}.mlp.routed_expert_ids", output[1], layer, "routed experts of every position, "
                       "in the order the gate picked them (model.py:822): [position, expert]")
            writer.add(f"layer_{layer}.mlp.route_weights", output[0], layer, "weight of every routed expert, "
                       "normalised and scaled as the experts receive it (model.py:823-826): [position, expert]")

        handles.extend((block.register_forward_pre_hook(block_begins), block.register_forward_hook(block_ends),
                        block.ffn.gate.register_forward_hook(gate_ran)))
        if block.engram is not None:
            handles.extend((block.engram.register_forward_pre_hook(lookup_begins),
                            block.engram.register_forward_hook(lookup_ends)))

    handles.append(model.embed.register_forward_hook(embedding_ran))
    if model.engram_hash is not None:
        handles.append(model.engram_hash.register_forward_hook(row_ids_ran))
    if keep_head_input:
        handles.append(model.head.register_forward_pre_hook(
            lambda _, arguments: setattr(observation, "head_input", arguments[0])))
    for block in model.layers:
        watch(block)
    try:
        yield observation
    finally:
        for handle in handles:
            handle.remove()


class LinearRecorder:
    """Stands in for model.py's linear() and act_quant() while a pass is recorded. Both call the reference's own
    function and keep what went in and what came out: the input, the quantized input and its scales exactly as
    linear() obtained them from act_quant (model.py:187 and :197), and what its product returned."""

    def __init__(self, session, writer, layers, expert_calls, routing):
        self.reference_model, self.writer, self.layers, self.routing = session.reference.model, writer, layers, routing
        self.reference_linear, self.reference_act_quant = self.reference_model.linear, self.reference_model.act_quant
        self.names = {id(parameter): name for name, parameter in session.model.named_parameters()}
        self.staged = {id(parameter): (stage, slot_name) for stage in session.stages
                       for slot_name, parameter in stage.slots.items()}
        self.expert_rows_left = dict.fromkeys(layers, expert_calls)
        self.rows_of_expert_call, self.open_call, self.calls = {}, None, collections.Counter()

    def target_of(self, inputs, weight):
        """Name, layer, name inside the layer, rows and facts of a call; None for a call that is not recorded."""
        if weight.dtype not in QUANTIZED_WEIGHT_DTYPES:
            return None
        name, rows, facts = self.names.get(id(weight)), slice(None), {}
        if name is None:
            if id(weight) not in self.staged:
                raise RuntimeError("the reference ran a quantized linear on a weight this tool cannot name")
            stage, slot_name = self.staged[id(weight)]
            name = f"{stage.occupant_name}.{slot_name}"
        match = LAYER_WEIGHT_NAME.fullmatch(name)
        if match is None or int(match["layer"]) not in self.layers:
            return None
        layer, name_in_layer = int(match["layer"]), match["name_in_layer"]
        expert = ROUTED_EXPERT_LINEAR_NAME.fullmatch(name_in_layer)
        if expert is not None:
            call = (layer, int(expert["expert"]))
            if call not in self.rows_of_expert_call:                       # decided at the call's first linear
                row_count = min(inputs.shape[0], self.expert_rows_left[layer], EXPERT_ROWS_PER_CALL_LIMIT)
                self.expert_rows_left[layer] -= row_count
                self.rows_of_expert_call[call] = row_count
            row_count = self.rows_of_expert_call[call]
            if row_count == 0:
                return None
            positions = torch.where(self.routing[layer] == call[1])[0][:row_count].tolist()   # model.py:899
            rows, facts = slice(0, row_count), {"rows_of_the_call": inputs.shape[0], "positions": positions}
        return name, layer, name_in_layer, rows, facts

    def linear(self, inputs, weight, bias = None):                         # as model.py:181 is called
        target = self.target_of(inputs, weight)
        if target is None:
            return self.reference_linear(inputs, weight, bias)
        name, layer, name_in_layer, rows, facts = target
        self.open_call = {}
        try:
            output = self.reference_linear(inputs, weight, bias)
        finally:
            open_call, self.open_call = self.open_call, None
        if "quantized" not in open_call:
            raise RuntimeError(f"{name}: the reference's linear did not quantize its input")
        facts = {**facts, "weight": name, "weight_dtype": str(weight.dtype).removeprefix("torch.")}
        quantized_input, input_scales = open_call["quantized"]
        for part, tensor, description in (
                ("input", inputs, "input of the linear, before the reference quantizes it"),
                ("quantized_input", quantized_input, "the input as the reference quantized it: one byte per value"),
                ("input_scales", input_scales, "scales of the quantized input: one byte for every "
                                               f"{self.reference_model.fp8_block_size} values of a row"),
                ("output", output, "what the reference's product returned")):
            self.writer.add(f"layer_{layer}.linear.{name_in_layer}.{part}", tensor[rows], layer, description, **facts)
        self.calls[layer] += 1
        return output

    def act_quant(self, *positional, **named):
        result = self.reference_act_quant(*positional, **named)
        if self.open_call is not None:
            self.open_call["quantized"] = result
        return result

    @contextlib.contextmanager
    def installed(self):
        reference_model = self.reference_model
        reference_model.linear, reference_model.act_quant = self.linear, self.act_quant
        try:
            yield
        finally:
            reference_model.linear, reference_model.act_quant = self.reference_linear, self.reference_act_quant


# ---- The captures ----------------------------------------------------------------------------------------------------
def capture_pass(session, writer, token_ids, arguments, record_linears = False):
    forget_earlier_runs(session)
    model, model_arguments, positions = session.model, session.model_arguments, len(token_ids)
    stream_values = 0 if record_linears else model_arguments.dim * (
        1 + STREAMS_RECORDED_PER_LAYER * model_arguments.n_layers * model_arguments.hc_mult)
    bytes_needed = positions * FLOAT32_BYTES * (stream_values + arguments.all_logits * model_arguments.vocab_size)
    bytes_free = shutil.disk_usage(os.path.dirname(writer.tensors_path)).free
    if bytes_free < bytes_needed:                                          # the streams and the logits: the bulk
        raise RuntimeError(f"at least {bytes_needed} bytes are needed in the capture folder, {bytes_free} are free")
    input_ids = torch.tensor([token_ids], dtype = torch.long, device = CARD_DEVICE)
    writer.add("input.token_ids", input_ids, None, "token ids of the sequence, in order")
    with observed_pass(session, writer, not record_linears, arguments.all_logits) as observation:
        recorder = LinearRecorder(session, writer, arguments.layer_list, arguments.expert_calls, observation.routing)
        with (recorder.installed() if record_linears else contextlib.nullcontext()):
            started, writer_seconds = synchronized_clock(), writer.seconds
            output_ids, logits, _ = model.forward(input_ids, 0)            # generate.py:69-74, the prompt's call
            seconds = synchronized_clock() - started - (writer.seconds - writer_seconds)
    if len(observation.layers) != len(model.layers) or tuple(logits.shape) != (SEQUENCES, model_arguments.vocab_size) \
            or not bool(torch.isfinite(logits).all()):
        raise RuntimeError(f"{len(observation.layers)} layers ran; the logits have shape {tuple(logits.shape)} or "
                           "are not all finite")
    silent_layers = [layer for layer in arguments.layer_list if record_linears and not recorder.calls[layer]]
    if silent_layers:
        raise RuntimeError(f"no quantized linear was recorded in layers {silent_layers}")
    writer.add("logits.last_row", logits.view(SEQUENCES, 1, -1), None, "logits of the last position, as the "
               "reference's forward returns them (model.py:1269): [sequence, 1, vocabulary]")
    writer.results.update(positions = positions, next_token_id = int(output_ids[0]), seconds_of_the_pass = seconds,
                          layers = observation.layers, linear_calls_recorded = sum(recorder.calls.values()))
    if arguments.all_logits:
        every_row = model.head(observation.head_input, full_logits = True) # model.py:1008-1012 over every position
        writer.add("logits.all_rows", every_row, None, "logits of every position, from the reference's head given "
                   "the rows of the pass with full_logits: [sequence, position, vocabulary]")
        writer.results["last_of_all_rows_has_the_bits_of_last_row"] = same_bits(every_row[:, -1], logits)
    print(f"pass over {positions} tokens: {seconds:.3f} s without the time the records took")
    for entry in observation.layers:
        lookup = f"; n-gram lookup before it {entry['ngram_seconds']:.3f} s for {entry['ngram_rows_moved']} rows" \
            if "ngram_seconds" in entry else ""
        print(f"layer {entry['layer']:2d}: {entry['seconds']:8.3f} s, of which {entry['expert_move_seconds']:7.3f} s "
              f"moved {entry['experts_moved']:3d} experts ({entry['expert_bytes_moved'] / BYTES_PER_GIB:5.2f} GiB) "
              f"to the card{lookup}")


def capture_generate(session, writer, token_ids, arguments):
    model, end_token_id = session.model, session.tokenizer.eos_token_id
    writer.add("input.token_ids", torch.tensor([token_ids], dtype = torch.long), None, "token ids of the prompt")
    runs = []
    for run in range(1, (RUNS_WITH_TWICE if arguments.twice else 1) + 1):
        forget_earlier_runs(session)
        rows, clocks = [], []

        def head_ran(_, head_arguments, output):                           # once per call of the model's forward
            ended = synchronized_clock()
            rows.append(output.detach().cpu())
            clocks.append((ended, time.perf_counter()))

        handle = model.head.register_forward_hook(head_ran)
        started = synchronized_clock()
        try:                                                               # generate.py:157 and :188, one prompt
            completion = session.reference.generate.generate(model, [token_ids], arguments.new_tokens, end_token_id)[0]
        finally:
            handle.remove()
        logits = torch.cat(rows)                                           # [step, vocabulary]
        chosen = completion + [end_token_id] * (len(rows) - len(completion))   # generate.py:84-85 cuts at the end token
        if not len(completion) <= len(rows) <= len(completion) + 1 or logits.argmax(dim = -1).tolist() != chosen:
            raise RuntimeError(f"run {run}: {len(rows)} steps do not explain generate()'s {len(completion)} tokens")
        chosen_ids = torch.tensor(chosen, dtype = torch.long, device = logits.device)
        top_logits, top_ids = torch.topk(logits, arguments.top_k, dim = -1)
        step_seconds = [clocks[0][0] - started] + [after[0] - before[1] for before, after in zip(clocks, clocks[1:])]
        writer.add(f"generate.run_{run}.token_ids", chosen_ids, None, "token the reference chose at every step, "
                   "greedy (model.py:1288-1289); the end token is the last entry when the run stopped on it")
        writer.add(f"generate.run_{run}.chosen_token_logits", logits.gather(1, chosen_ids.unsqueeze(1)).squeeze(1),
                   None, "logit of the chosen token at every step")
        writer.add(f"generate.run_{run}.top_token_ids", top_ids, None, "ids of the largest logits: [step, rank]")
        writer.add(f"generate.run_{run}.top_logits", top_logits, None, "the largest logits: [step, rank]")
        if arguments.all_logits:
            writer.add(f"generate.run_{run}.logits", logits, None, "logits of every step: [step, vocabulary]")
        runs.append((chosen, logits))
        text = writer.results[f"run_{run}_text"] = session.tokenizer.decode(completion)   # as generate.py:158
        writer.results[f"run_{run}"] = {"steps": len(rows), "stopped_on_the_end_token": len(rows) > len(completion),
                                        "prompt_seconds": step_seconds[0], "decode_step_seconds": step_seconds[1:]}
        print(f"run {run}: {len(rows)} steps; prompt of {len(token_ids)} tokens {step_seconds[0]:.3f} s; decode steps "
              f"{sum(step_seconds[1:]) / max(len(rows) - 1, 1):.3f} s each on average; it wrote {text!r}", flush = True)
    if len(runs) > 1:
        (first_ids, first_logits), (second_ids, second_logits) = runs
        steps = min(len(first_logits), len(second_logits))
        differing = [step for step in range(steps) if not same_bits(first_logits[step], second_logits[step])]
        largest = float((first_logits[:steps].double() - second_logits[:steps].double()).abs().max())
        comparison = {"token_ids_identical": first_ids == second_ids,
                      "logits_identical": same_bits(first_logits, second_logits),
                      "first_step_with_different_logits": differing[0] if differing else None,
                      "steps_with_different_logits": len(differing), "largest_logit_difference": largest}
        writer.results["runs_compared"] = comparison
        print(f"the two runs: token ids {'identical' if comparison['token_ids_identical'] else 'DIFFERENT'}, logits "
              f"{'identical bit for bit' if comparison['logits_identical'] else 'DIFFERENT'}; {len(differing)} of "
              f"{steps} steps differ, largest difference of a logit {largest!r}")


CAPTURES = {"forward": capture_pass, "linears": functools.partial(capture_pass, record_linears = True),
            "generate": capture_generate}


def run(arguments):
    stage_seconds = {}
    writers = {mode: CaptureWriter(arguments.capture_folders[mode]) for mode in arguments.modes}   # before the load
    complete = set()
    try:
        session, token_ids = open_session(arguments, stage_seconds)
        for mode in arguments.modes:
            writer = writers[mode]
            writer.manifest["conditions"] = {
                "capture": mode, "command_line": sys.argv, "arguments": dict(sorted(vars(arguments).items())),
                "torch_version": torch.__version__, "card": torch.cuda.get_device_name(arguments.card),
                "default_dtype": str(torch.get_default_dtype()), "tokens": len(token_ids),
                "reference_files_sha256": session.reference.file_digests,
                "model_arguments": dataclasses.asdict(session.model_arguments), "load": session.load_summary}
            torch.cuda.reset_peak_memory_stats()                           # the card's peaks below: this capture's own
            session.quantization_counts.clear()
            with torch.inference_mode(), timed_stage(stage_seconds, f"capture {mode}"):
                CAPTURES[mode](session, writer, token_ids, arguments)
            quant = writer.results["quantization_kernels_against_their_definitions"] = \
                dict(session.quantization_counts)
            print(f"{mode}: the reference's act_quant kernel returned other bytes than its definition in "
                  f"{quant.get('calls_the_kernel_got_wrong', 0)} of {quant.get('calls', 0)} calls "
                  f"({quant.get('bytes_the_kernel_got_wrong', 0)} of {quant.get('bytes', 0)} bytes, every one the "
                  f"input byte its known fault leaves); in place, {quant.get('in_place_calls', 0)} calls of "
                  f"act_quant and {quant.get('four_bit_in_place_calls', 0)} of fp4_act_quant left their "
                  "definitions bit for bit", flush = True)
            peaks = {"card_bytes_allocated_peak": torch.cuda.max_memory_allocated(),
                     "card_bytes_reserved_peak": torch.cuda.max_memory_reserved(),
                     "host_resident_bytes_peak_with_mapped_shard_pages":
                         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RESIDENT_SET_UNIT_BYTES}
            writer.results.update(stage_seconds = dict(stage_seconds), **peaks)
            complete.add(mode)
            allocated, reserved, resident = (count / BYTES_PER_GIB for count in peaks.values())
            print(f"{mode}: {len(writer.records)} tensors, {writer.byte_offset} bytes; peak card memory "
                  f"{allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved; peak host memory {resident:.2f} GiB "
                  "resident, mapped shard pages included", flush = True)
            if arguments.self_test:                                        # the shapes, one line per kind of record
                kinds = collections.Counter((NUMBER_IN_A_NAME.sub("*", record["name"]), record["dtype"],
                                             str(record["shape"])) for record in writer.records)
                for (pattern, dtype, shape), count in kinds.items():
                    print(f"  {mode}: {count:5d} x {dtype:8s} {shape:28s} {pattern}")
    finally:
        for mode, writer in writers.items():
            writer.close(mode in complete)


def check_arguments(parser, arguments):
    """Refuses, before anything is imported or loaded, what can be refused then."""
    if sys.byteorder != REQUIRED_BYTE_ORDER:
        parser.error(f"this host is {sys.byteorder}-endian: tensors.bin must be {REQUIRED_BYTE_ORDER}-endian")
    for needed in (os.path.join(INFERENCE_FOLDER_NAME, name) for name in ("generate.py", MODEL_ARGUMENTS_FILE_NAME)):
        if not os.path.isfile(os.path.join(arguments.checkpoint, needed)):
            parser.error(f"--checkpoint has no {needed}: the Hugging Face folder is expected")
    if arguments.self_test:
        if arguments.mode is not None or arguments.tokens_json is not None or arguments.prompt_json is not None:
            parser.error("--self-test takes no --mode, --tokens-json or --prompt-json: it runs its own prompt")
        arguments.modes, arguments.twice, arguments.all_logits = SELF_TEST_MODES, True, True
        arguments.tokens, arguments.new_tokens = SELF_TEST_TOKENS, SELF_TEST_NEW_TOKENS
        arguments.expert_calls = SELF_TEST_EXPERT_CALLS
    elif arguments.mode is None or (arguments.tokens_json is None) == (arguments.prompt_json is None):
        parser.error("--mode and exactly one of --tokens-json and --prompt-json are needed (or --self-test)")
    else:
        arguments.modes = (arguments.mode,)
    selection = arguments.gguf_take is not None or arguments.gguf_expert_layers is not None
    if (arguments.weights_gguf is None) != (arguments.gguf_linear_inputs is None) \
            or ((arguments.gguf_control or selection) and arguments.weights_gguf is None) \
            or (arguments.gguf_control and selection):
        parser.error("--weights-gguf and --gguf-linear-inputs are given together or not at all; --gguf-control, "
                     "--gguf-take and --gguf-expert-layers only with them, and --gguf-control without the other two")
    arguments.gguf_parts = None if arguments.gguf_take is None else tuple(sorted(set(arguments.gguf_take.split(","))))
    arguments.gguf_expert_layer_list = None
    if arguments.gguf_parts is not None and not set(arguments.gguf_parts) <= set(GGUF_PARTS):
        parser.error(f"--gguf-take {arguments.gguf_take}: parts among {', '.join(GGUF_PARTS)}, separated by commas")
    if arguments.gguf_expert_layers is not None:
        if arguments.gguf_parts is not None and "experts" not in arguments.gguf_parts:
            parser.error("--gguf-expert-layers names layers whose routed experts are taken: --gguf-take leaves "
                         "the experts out")
        texts = [text.split("..") for text in arguments.gguf_expert_layers.split(",")]
        if any(len(ends) not in ENDS_OF_A_RANGE or not all(end.isascii() and end.isdigit() for end in ends)
               for ends in texts):
            parser.error(f"--gguf-expert-layers {arguments.gguf_expert_layers}: layers and ranges first..last in "
                         "plain digits, separated by commas")
        ranges = [[int(end) for end in ends] for ends in texts]
        if any(ends[-1] < ends[0] or ends[-1] > HIGHEST_LAYER_NUMBER_READ for ends in ranges):
            parser.error(f"--gguf-expert-layers {arguments.gguf_expert_layers}: a range runs backwards or passes "
                         f"layer {HIGHEST_LAYER_NUMBER_READ}")
        arguments.gguf_expert_layer_list = sorted({layer for ends in ranges for layer in range(ends[0], ends[-1] + 1)})
    if arguments.weights_gguf is not None and (arguments.self_test or "linears" in arguments.modes
                                               or not os.path.isfile(arguments.weights_gguf)):
        parser.error("--weights-gguf names a GGUF file and goes with --mode forward or generate: the linears "
                     "capture records the reference's own quantized products")
    arguments.capture_folders = {mode: os.path.join(arguments.output_directory, mode) if arguments.self_test
                                 else arguments.output_directory for mode in arguments.modes}
    for folder in arguments.capture_folders.values():
        for file_name in (TENSORS_FILE_NAME, MANIFEST_FILE_NAME):
            if os.path.exists(os.path.join(folder, file_name)):
                parser.error(f"--output-directory holds a {file_name} already: {folder}")
    try:
        arguments.layer_list = sorted({int(text) for text in arguments.layers.split(",")})
    except ValueError:
        parser.error(f"--layers {arguments.layers}: layer numbers separated by commas are expected")
    if arguments.layer_list[0] < 0 or arguments.card < 0 or arguments.sequence < 0 or arguments.expert_calls < 0 \
            or arguments.top_k < 1 or arguments.maximum_sequence_tokens < 1 \
            or (arguments.tokens is not None and arguments.tokens < 1):
        parser.error("--layers, --card, --sequence, --expert-calls, --top-k, --tokens or --maximum-sequence-tokens "
                     "is out of range")
    if "generate" in arguments.modes and (arguments.new_tokens is None or arguments.new_tokens < 1):
        parser.error("--mode generate needs --new-tokens of at least 1")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev = False)
    parser.add_argument("--checkpoint", required = True, type = os.path.abspath, help = "the Hugging Face folder")
    parser.add_argument("--output-directory", required = True, type = os.path.abspath, help = "receives the capture")
    parser.add_argument("--card", type = int, default = 0, help = "index of the card the reference runs on")
    parser.add_argument("--mode", choices = tuple(CAPTURES), default = None)
    parser.add_argument("--self-test", action = "store_true", help = "all three modes on a fixed prompt's first tokens")
    parser.add_argument("--tokens-json", default = None, type = os.path.abspath, help = "JSON list of token id lists")
    parser.add_argument("--prompt-json", default = None, type = os.path.abspath, help = "chat messages, as generate.py")
    parser.add_argument("--sequence", type = int, default = 0, help = "which entry of the JSON file")
    parser.add_argument("--thinking-mode", choices = ("chat", "thinking"), default = DEFAULT_THINKING_MODE)
    parser.add_argument("--tokens", type = int, default = None, help = "use only the first N tokens of the sequence")
    parser.add_argument("--maximum-sequence-tokens", type = int, default = DEFAULT_MAXIMUM_SEQUENCE_TOKENS,
                        help = "the reference's max_seq_len: prompt and new tokens must fit")
    parser.add_argument("--all-logits", action = "store_true", help = "logits of every position, or of every step")
    parser.add_argument("--layers", default = DEFAULT_LINEAR_LAYERS, help = "linears: the layers recorded")
    parser.add_argument("--expert-calls", type = int, default = DEFAULT_EXPERT_CALLS, help = "linears: pairs per layer")
    parser.add_argument("--new-tokens", type = int, default = None, help = "generate: tokens generated")
    parser.add_argument("--top-k", type = int, default = DEFAULT_TOP_K, help = "generate: largest logits kept per step")
    parser.add_argument("--twice", action = "store_true", help = "generate: run twice and compare bit by bit")
    parser.add_argument("--traceback", action = "store_true", help = "print the Python traceback of a failure")
    parser.add_argument("--attention-card", default = None, help = "CUDA_VISIBLE_DEVICES value of the attention card")
    parser.add_argument("--weights-gguf", default = None, type = os.path.abspath,
                        help = "take the weights from it (all of them, unless the three options below say less)")
    parser.add_argument("--gguf-control", action = "store_true",
                        help = "with --weights-gguf: multiply by the CHECKPOINT's values through the same stand-in")
    parser.add_argument("--gguf-take", default = None, help = "with --weights-gguf: only these parts from the file")
    parser.add_argument("--gguf-expert-layers", default = None,
                        help = "with --weights-gguf: routed experts from the file in these layers only")
    parser.add_argument("--gguf-linear-inputs", choices = GGUF_LINEAR_INPUTS, default = None,
                        help = "with --weights-gguf: the inputs of the quantized linears, as they are or quantized")
    arguments = parser.parse_args()
    check_arguments(parser, arguments)
    warnings.showwarning = lambda message, category, *location, **named: \
        print(f"warning ({category.__name__}): {message}", file = sys.stderr)      # without the path of a library file
    warnings.filterwarnings("ignore", message = "The given NumPy array is not writable")   # the mapped shards
    try:
        run(arguments)
    except Exception as failure:
        if arguments.traceback:
            traceback.print_exc()
        print(f"reference_oracle FAILED: {type(failure).__name__}: {failure}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
