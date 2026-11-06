#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Converts a folder of safetensors files into ONE Anemone pack, format version 2, and checks a pack.

    make_pack.py convert <source folder> <output pack file> [--jobs N] [--overwrite]
                         [--reference <checkpoint folder> | --without-tables]
    make_pack.py verify <pack file> [--jobs N]
    make_pack.py self-test

The format is include/anemone/pack.h; the unit kinds are model_seam.h's and the digest is pack_digest.h's. What the
three commands do about a model's quantized weights (weight_format_seam.h) is in kernel_layouts.py, beside this
file, and in its text; what they do about its n-gram tables (engram.h) is in engram_tables.py and in its text.

convert reads the header of every *.safetensors file of the folder in sorted order (a name stored in more than one
file is the LAST file's, as the engine that wrote these folders takes it; one line says so), sorts every tensor
into a unit with the model table below (a name no rule takes stops the run; the n-gram tables get host-only units
of their own behind the others) and lays the whole pack out before it writes a byte. Every unit is one extent, in
which the tensors lie in the order of their names, each on a multiple of 64 bytes, zero-filled between them and up
to the next block. A quantized weight and its scales are ONE tensor, a matrix of a weight format in the layout its
kernels read; every other tensor is plain, its bytes as the folder stores them. The file is created at its final
size under a temporary name beside the destination; --jobs worker processes (default: the number of processors)
each assemble one extent in memory (of an n-gram table, one piece), digest it and write it at its offset; then
come the tables of the pack, the header, fsync, the rules of verify read back from the file, and the rename. A run
that fails leaves no file under the name of the pack, and the temporary file holds no header until every extent is
written. An existing output is refused without --overwrite, and the free space is checked first.

verify is a second reading, which shares the digest and the constants of the format with convert and none of its
layout: every rule pack.h lists for anemone_pack_open, in the order of that list and each with its own message, read
at byte offsets that are written out a second time below; then the rules of engram.h; then the digest of every
extent, recomputed by the worker processes, which also look in every matrix for a code that is no number. It stops
at the first rule or digest that does not hold. The two rules about the storage reader are one about the file's size.

self-test proves the tool without a model: the known answer of the digest, the digest in pieces and against plain
integer arithmetic, a small folder it writes itself (the real name shapes, a repeated name, quantized weights of
both formats) converted, verified and compared byte for byte (kernel_layouts.py, with the matrices' own cases), and
what must be refused shown refused: an output that exists, six folders that break a rule of convert, one bit of an
extent, one tensor record's offset, a file one block short; then the n-gram tables (engram_tables.py).

Needs Python 3.9 and numpy, and for n-gram tables what engram_tables.py names. Exit status 0, 1 after any failure,
2 for a command line it cannot read. Convert tool: not part of the engine.
"""
import argparse
import collections
import concurrent.futures
import contextlib
import glob
import hashlib
import io
import json
import math
import multiprocessing
import os
import re
import shutil
import struct
import sys
import tempfile
import time

import numpy

import engram_tables                                       # used inside functions only: the two files import each other
import front_end_tables
import kernel_layouts                                      # the same

BYTES_PER_GIB = 2 ** 30
UINT32_MAXIMUM = 2 ** 32 - 1
UINT64_MODULUS = 2 ** 64
EXIT_STATUS_FAILURE = 1

# ---- The format: include/anemone/pack.h, model_seam.h, weight_format_seam.h ----------------------------------------
PACK_MAGIC = b"ANEMONE\0"
PACK_FORMAT_VERSION = 2
PACK_BLOCK_BYTES = 4096
PACK_MAXIMUM_TABLES_BYTES = 2 ** 30
PACK_MAXIMUM_DIMENSIONS = 4
PACK_TENSOR_RESERVED_WORDS = 1
PACK_NO_LAYER = UINT32_MAXIMUM
MODEL_NAME_CAPACITY = 64
MATRIX_ALIGNMENT_BYTES = 64
WEIGHT_FORMAT_NONE = 0                                     # enum anemone_weight_format: a plain tensor
UNIT_KIND_NAMES = ("stage", "on-demand", "host-only")      # enum anemone_placement_unit_kind, by value
UNIT_STAGE, UNIT_ON_DEMAND, UNIT_HOST_ONLY = range(len(UNIT_KIND_NAMES))
# safetensors dtype: (enum anemone_pack_element_type, bytes of one element)
ELEMENT_TYPES = {"F16": (0, 2), "BF16": (1, 2), "F32": (2, 4), "I8": (3, 1), "U8": (4, 1), "I16": (5, 2),
                 "I32": (6, 4), "I64": (7, 8), "F8_E4M3": (8, 1), "F8_E8M0": (9, 1)}
ELEMENT_BYTE_COUNTS = dict(ELEMENT_TYPES.values())         # by element type

# The records as convert writes them: the structs of pack.h, little-endian, in the order of their members.
HEADER_RECORD = struct.Struct("<8sII64sQQQIIIIQ")          # struct anemone_pack_header
EXTENT_RECORD = struct.Struct("<QQQII")                    # struct anemone_pack_extent
UNIT_RECORD = struct.Struct("<II")                         # struct anemone_pack_unit_record
TENSOR_RECORD = struct.Struct("<QQIIIII4IIII")             # struct anemone_pack_tensor

# The records as verify reads them, member: (byte offset, byte count). Written out a second time on purpose, so
# that a slip in the formats above does not pass its own check.
HEADER_FIELDS = {"magic": (0, 8), "format_version": (8, 4), "block_byte_count": (12, 4), "model_name": (16, 64),
                 "file_byte_count": (80, 8), "tables_byte_count": (88, 8), "tables_digest": (96, 8),
                 "extent_count": (104, 4), "unit_count": (108, 4), "tensor_count": (112, 4),
                 "name_byte_count": (116, 4), "header_digest": (120, 8)}
HEADER_BYTE_STRING_FIELDS = ("magic", "model_name")        # every other member of the header is a number
EXTENT_FIELDS = {"byte_offset": (0, 8), "byte_count": (8, 8), "digest": (16, 8), "unit_index": (24, 4),
                 "reserved_zero": (28, 4)}
UNIT_FIELDS = {"unit_kind": (0, 4), "layer_index": (4, 4)}
TENSOR_FIELDS = {"byte_offset_in_extent": (0, 8), "byte_count": (8, 8), "extent_index": (16, 4),
                 "name_byte_offset": (20, 4), "name_byte_count": (24, 4), "element_type": (28, 4),
                 "dimension_count": (32, 4), "dimensions": (36, 16), "weight_format": (52, 4),
                 "bits_per_weight": (56, 4), "reserved_zero": (60, 4)}
HEADER_BYTES, EXTENT_RECORD_BYTES, UNIT_RECORD_BYTES, TENSOR_RECORD_BYTES, DIMENSION_BYTES = 128, 32, 8, 64, 4

# ---- The digest: include/anemone/pack_digest.h ---------------------------------------------------------------------
DIGEST_WORD_BYTES = 8
DIGEST_WORD_TYPE = "<u8"
DIGEST_POSITION_MULTIPLIER = 0x9E3779B97F4A7C15
DIGEST_FIRST_MULTIPLIER, DIGEST_SECOND_MULTIPLIER = 0xBB67AE8584CAA73B, 0xA54FF53A5F1D36F1
DIGEST_FIRST_SHIFT, DIGEST_SECOND_SHIFT, DIGEST_THIRD_SHIFT = 32, 29, 32
DIGEST_KNOWN_ANSWER_BYTES = bytes(range(16))
DIGEST_KNOWN_ANSWER = 0x9F861E2075053703
DIGEST_PIECE_WORDS = 2 ** 15                               # digested at a time: every temporary is 256 KiB
DIGEST_TEXT_FORMAT = "#018x"

# ---- The model table -------------------------------------------------------------------------------------------------
# What model_seam.h calls a unit, as rules over tensor names: the groups of a model in the order they run. A pattern
# takes a name it matches whole. A group whose patterns hold a block number yields, for every number from 0, one
# stage unit (stage) and then one on-demand unit for every expert number (expert); a group without one is a single
# stage unit. Names that match left_out belong to no unit of the table. Another model is another table.
NUMBER = "(?:0|[1-9][0-9]{0,8})"                           # as a name writes it: no sign, no leading zero, below 10^9
BLOCK = "(?P<block>" + NUMBER + ")"
EXPERT = "(?P<expert>" + NUMBER + ")"
UnitGroup = collections.namedtuple("UnitGroup", ("label", "has_layer_index", "stage", "expert", "left_out"))
MODEL_NAME = "DeepSeek-V4.1-Flash"
MODEL_UNIT_GROUPS = (
    UnitGroup("vision", False, r"(?:vision|aligner)\..*|image_start|image_end|image_newline", None, None),
    UnitGroup("embedding", False, r"embed\.weight", None, None),
    UnitGroup("layer", True, rf"layers\.{BLOCK}\.(?!ffn\.experts\.).*",
              rf"layers\.{BLOCK}\.ffn\.experts\.{EXPERT}\..*",
              rf"layers\.{NUMBER}\.engram\.embed\.(?:weight|scale)"),      # the n-gram tables: engram_tables.py
    UnitGroup("head", False, r"norm\.weight|head\..*", None, None),
    UnitGroup("draft block", False, rf"mtp\.{BLOCK}\.(?!ffn\.experts\.).*",
              rf"mtp\.{BLOCK}\.ffn\.experts\.{EXPERT}\..*", None),
)

# ---- Reading and writing ---------------------------------------------------------------------------------------------
SAFETENSORS_SUFFIX = ".safetensors"
SAFETENSORS_HEADER_LENGTH_BYTES = 8
SAFETENSORS_HEADER_LENGTH_LIMIT = 512 * 2 ** 20            # a header larger than this is not a header
SAFETENSORS_METADATA_NAME = "__metadata__"
SAFETENSORS_DATA_OFFSET_COUNT = 2                          # where a tensor begins and where it ends
TEMPORARY_SUFFIX = ".partial"
OUTPUT_FILE_MODE = 0o644
TRANSFER_LIMIT_BYTES = 2 ** 31 - 2 ** 12                   # the most Linux moves in one read or write call
VERIFY_PIECE_BYTES = 2 ** 20                               # the most of an extent verify holds while it digests it
PROGRESS_INTERVAL_SECONDS = 10.0
JOBS_LIMIT = 256                                           # a pool starts every process it is allowed
REPEATED_NAME_LINE = "repeated name, the last file wins:"

SourceTensor = collections.namedtuple("SourceTensor", ("path", "byte_offset", "byte_count", "dtype", "shape"))
Unit = collections.namedtuple("Unit", ("kind", "layer_index", "label", "names"))
# pieces, one for a tensor: (path, or what a file beside this one makes it from, byte offset, byte count, extent offset)
Extent = collections.namedtuple("Extent", ("byte_offset", "byte_count", "unit_index", "pieces"))
TensorRow = collections.namedtuple("TensorRow", ("name", "extent_index", "byte_offset_in_extent", "byte_count",
                                                 "element_type", "dimensions", "weight_format", "bits_per_weight"))
PackLayout = collections.namedtuple("PackLayout", ("extents", "tensor_rows", "name_byte_count", "tables_byte_count",
                                                   "file_byte_count"))

# ---- The self-test's model: the name shapes of the three source folders at toy sizes; every element type appears ----
# Its quantized weights: the smallest matrix of each weight format, and one of 8 bits with a half last scale block.
SELF_TEST_JOBS = 2
SELF_TEST_LAYER_COUNT = 2
SELF_TEST_EXPERTS_PER_LAYER = 11                           # two digits: units run by number, records sort by bytes
SELF_TEST_DRAFT_BLOCK_COUNT = 2
SELF_TEST_EXPERTS_PER_DRAFT_BLOCK = 2
SELF_TEST_VISION = (("vision.patch_embed.proj.weight", "BF16", (2, 12)),
                    ("vision.patch_embed.proj.bias", "BF16", (2,)),
                    ("vision.blocks.0.norm1.weight", "F32", (2,)),
                    ("vision.blocks.0.attn.wqkv.weight", "BF16", (6, 2)),
                    ("vision.blocks.0.attn.wqkv.bias", "BF16", (6,)),
                    ("vision.blocks.0.attn.wo.weight", "BF16", (2, 2)),
                    ("vision.blocks.0.attn.wo.bias", "BF16", (2,)),
                    ("vision.blocks.0.norm2.weight", "F32", (2,)),
                    ("vision.blocks.0.mlp.w1.weight", "BF16", (6, 2)),
                    ("vision.blocks.0.mlp.w2.weight", "BF16", (2, 3)),
                    ("vision.norm.weight", "F32", (2,)),
                    ("aligner.w1.weight", "BF16", (5, 18)), ("aligner.w1.bias", "BF16", (5,)),
                    ("aligner.w2.weight", "BF16", (5, 5)), ("aligner.w2.bias", "BF16", (5,)),
                    ("image_start", "BF16", (5,)), ("image_end", "BF16", (5,)), ("image_newline", "BF16", (5,)))
SELF_TEST_EMBEDDING = (("embed.weight", "BF16", (120000, 5)),)         # an extent longer than a piece of verify
SELF_TEST_LAYER = (("attn_norm.weight", "BF16", (5,)), ("attn.wq_a.weight", "F8_E4M3", (16, 32)),
                   ("attn.wq_a.scale", "F8_E8M0", (1, 1)), ("hc_attn_fn", "F32", (3, 4)),
                   ("ffn.gate.weight", "F16", (SELF_TEST_EXPERTS_PER_LAYER, 5)),
                   ("ffn.gate.bias_vl", "F32", (SELF_TEST_EXPERTS_PER_LAYER,)),
                   ("ffn.shared_experts.w1.weight", "F8_E4M3", (48, 64)),
                   ("ffn.shared_experts.w1.scale", "F8_E8M0", (2, 2)), ("engram.q_weight", "BF16", (4, 5)))
SELF_TEST_EXPERT = (("w1.weight", "I8", (16, 32)), ("w1.scale", "F8_E8M0", (16, 2)), ("w2.trellis", "I16", (2, 3, 4)),
                    ("w2.suh", "F16", (5,)), ("w2.mul1", "I32", ()))
SELF_TEST_HEAD = (("norm.weight", "BF16", (5,)), ("head.trellis", "I16", (2, 3, 4)), ("head.mul1", "I32", ()))
SELF_TEST_DRAFT_BLOCK = (("attn_norm.weight", "BF16", (5,)), ("markov_head.embed.weight", "U8", (3, 7)),
                         ("hc_attn_base", "I64", (3,)), ("tid2eid", "I8", (3, 4)))     # I8 under such a name: plain
SELF_TEST_REPEATED_NAME = "layers.1.engram.q_weight"       # stored twice: the last file's copy is SELF_TEST_LAYER's
SELF_TEST_REPEATED_EARLIER = (SELF_TEST_REPEATED_NAME, "F32", (4, 5))
SELF_TEST_FILE_NAMES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                        "model-00003-of-00003.safetensors")            # the last: the repeated name
SELF_TEST_STRAY_FILE_NAME = "model-00004-of-00004.safetensors"
# One more tensor with which convert must refuse the folder, and what the refusal says: a name under the experts
# with no number, a layer too far, an expert too many, an expert too far, a dimension too many, a dimension too large.
SELF_TEST_REFUSED_TENSORS = (
    (("layers.0.ffn.experts.first.w1.weight", "F16", (1,)), "takes the tensor 'layers.0.ffn.experts.first.w1.weight'"),
    ((f"layers.{SELF_TEST_LAYER_COUNT + 1}.hc_attn_fn", "F16", (1,)), f"layer {SELF_TEST_LAYER_COUNT} is missing"),
    ((f"layers.1.ffn.experts.{SELF_TEST_EXPERTS_PER_LAYER}.w1.weight", "F16", (1,)),
     f"layer 1 holds {SELF_TEST_EXPERTS_PER_LAYER + 1} experts, layer 0 holds {SELF_TEST_EXPERTS_PER_LAYER}"),
    ((f"layers.0.ffn.experts.{SELF_TEST_EXPERTS_PER_LAYER + 1}.w1.weight", "F16", (1,)),
     "layer 0: its expert numbers are not contiguous from 0"),
    (("head.weight", "F16", (1,) * (PACK_MAXIMUM_DIMENSIONS + 1)), f"{PACK_MAXIMUM_DIMENSIONS + 1} dimensions"),
    (("head.weight", "U8", (UINT32_MAXIMUM + 1, 0)), f"a dimension of {UINT32_MAXIMUM + 1} does not fit 32 bits"),
)
SELF_TEST_SAMPLE_EXTRA_WORDS = 1000                        # the digest sample is longer than one piece by this
SELF_TEST_FIRST_PIECE_WORDS = 24
SELF_TEST_FAR_WORD_INDEX = 2 ** 40                         # a position no 32 bits count


class PackError(Exception):
    """A source folder or a pack that breaks a rule; the message names the rule and the place."""


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def rate_text(byte_count, seconds):
    rate = byte_count / BYTES_PER_GIB / seconds if seconds > 0 else 0.0
    return f"{seconds:.1f} s, {rate:.2f} GiB/s"


def digest_of(data, first_word_index = 0):
    """The digest of pack_digest.h over a whole number of 64-bit words, in numpy's unsigned arithmetic, which wraps
    modulo 2^64. first_word_index is where the first word lies in its extent, so the digests of the pieces of an
    extent add up, modulo 2^64, to the digest of the whole."""
    words = numpy.frombuffer(data, dtype = DIGEST_WORD_TYPE)
    digest = 0
    for piece_start in range(0, len(words), DIGEST_PIECE_WORDS):
        piece = words[piece_start:piece_start + DIGEST_PIECE_WORDS]
        first_position = first_word_index + piece_start
        positions = numpy.arange(first_position, first_position + len(piece), dtype = numpy.uint64)
        mixed = piece ^ (positions * numpy.uint64(DIGEST_POSITION_MULTIPLIER))
        mixed ^= mixed >> numpy.uint64(DIGEST_FIRST_SHIFT)
        mixed *= numpy.uint64(DIGEST_FIRST_MULTIPLIER)
        mixed ^= mixed >> numpy.uint64(DIGEST_SECOND_SHIFT)
        mixed *= numpy.uint64(DIGEST_SECOND_MULTIPLIER)
        mixed ^= mixed >> numpy.uint64(DIGEST_THIRD_SHIFT)
        digest = (digest + int(mixed.sum(dtype = numpy.uint64))) % UINT64_MODULUS
    return digest


def read_exactly(handle, byte_offset, destination, path):
    """Fills destination, a memoryview of bytes, from an unbuffered file: a read may stop short and is continued."""
    handle.seek(byte_offset)
    byte_count_read = 0
    while byte_count_read < len(destination):
        step = handle.readinto(destination[byte_count_read:byte_count_read + TRANSFER_LIMIT_BYTES])
        if not step:
            raise PackError(f"{path}: ends before byte {byte_offset + len(destination)}")
        byte_count_read += step


def write_exactly(descriptor, byte_offset, source, path):
    """Writes all of source at byte_offset: one positioned write for anything below TRANSFER_LIMIT_BYTES, unless
    the write stops short; then it is continued."""
    source = memoryview(source)
    byte_count_written = 0
    while byte_count_written < len(source):
        step = os.pwrite(descriptor, source[byte_count_written:byte_count_written + TRANSFER_LIMIT_BYTES],
                         byte_offset + byte_count_written)
        if step <= 0:
            raise PackError(f"{path}: a write at byte {byte_offset + byte_count_written} moved nothing")
        byte_count_written += step


def run_in_workers(worker, tasks, task_byte_counts, jobs, activity):
    """One task per extent in a pool of worker processes; returns the results in the order of the tasks. The first
    task that fails ends the run with its error: what has not begun is cancelled. A worker that dies is an error
    too, never a wait without end. A worker is a fresh interpreter (spawn) on every Python: it inherits neither
    threads nor open files from this process."""
    results, byte_count_done, byte_count_in_all = [], 0, sum(task_byte_counts)
    last_report = time.monotonic()
    pool = concurrent.futures.ProcessPoolExecutor(max_workers = jobs, mp_context = multiprocessing.get_context("spawn"))
    try:
        for result, byte_count in zip(pool.map(worker, tasks), task_byte_counts):
            results.append(result)
            byte_count_done += byte_count
            if time.monotonic() - last_report >= PROGRESS_INTERVAL_SECONDS:
                last_report = time.monotonic()
                print(f"  {activity}: {byte_count_done / BYTES_PER_GIB:.1f} of "
                      f"{byte_count_in_all / BYTES_PER_GIB:.1f} GiB", file = sys.stderr, flush = True)
    finally:
        pool.shutdown(wait = True, cancel_futures = True)
    return results


# ---- convert ---------------------------------------------------------------------------------------------------------
def object_without_repeats(pairs):
    """For json.loads: a JSON object as a dictionary, refused when one name appears in it twice."""
    members = dict(pairs)
    if len(members) != len(pairs):
        raise ValueError("a name appears twice in one JSON object")
    return members


def is_list_of_counts(value):
    return isinstance(value, list) and all(type(item) is int and item >= 0 for item in value)


def read_safetensors_header(path):
    """The tensors of one safetensors file, name: SourceTensor, with everything its header says checked against the
    file. What a pack demands of a tensor (element type, dimensions) is checked where a tensor enters the pack."""
    file_byte_count = os.path.getsize(path)
    with open(path, "rb") as handle:
        length_bytes = handle.read(SAFETENSORS_HEADER_LENGTH_BYTES)
        if len(length_bytes) != SAFETENSORS_HEADER_LENGTH_BYTES:
            raise PackError(f"{path}: too short for a safetensors file")
        (header_length,) = struct.unpack("<Q", length_bytes)
        data_start = SAFETENSORS_HEADER_LENGTH_BYTES + header_length
        if header_length == 0 or header_length > SAFETENSORS_HEADER_LENGTH_LIMIT or data_start > file_byte_count:
            raise PackError(f"{path}: implausible header length {header_length}")
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise PackError(f"{path}: ends inside its header")
    try:
        header = json.loads(header_bytes.decode("utf-8"), object_pairs_hook = object_without_repeats)
    except (ValueError, RecursionError) as error:
        raise PackError(f"{path}: its header cannot be read: {error}") from error
    if not isinstance(header, dict):
        raise PackError(f"{path}: its header is no JSON object")
    tensors = {}
    for name, entry in header.items():
        if name == SAFETENSORS_METADATA_NAME:
            continue
        where = f"{path}: tensor {name!r}"
        if not isinstance(entry, dict):
            raise PackError(f"{where}: its entry is no JSON object")
        dtype, shape, data_offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if not isinstance(dtype, str) or not is_list_of_counts(shape) or not is_list_of_counts(data_offsets) \
                or len(data_offsets) != SAFETENSORS_DATA_OFFSET_COUNT:
            raise PackError(f"{where}: its dtype, shape or data_offsets is missing or malformed")
        first_byte, end_byte = data_offsets
        if not first_byte <= end_byte <= file_byte_count - data_start:
            raise PackError(f"{where} lies outside the file")
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise PackError(f"{where}: its name cannot be stored: {error}") from error
        tensors[name] = SourceTensor(path, data_start + first_byte, end_byte - first_byte, dtype, tuple(shape))
    return tensors


def read_source_folder(source_directory):
    """Every tensor of a folder, name: SourceTensor. A name stored in more than one file is the LAST file's in sorted
    order, which is what the engine that wrote these folders does; one line is printed for each."""
    paths = sorted(glob.glob(os.path.join(glob.escape(source_directory), "*" + SAFETENSORS_SUFFIX)))
    if not paths:
        raise PackError(f"{source_directory}: no *{SAFETENSORS_SUFFIX} file")
    sources = {}
    for path in paths:
        for name, source in read_safetensors_header(path).items():
            if name in sources:
                print(f"{REPEATED_NAME_LINE} {name} in {os.path.basename(sources[name].path)} and "
                      f"{os.path.basename(path)}")
            sources[name] = source
    return sources


def plan_units(unit_groups, names):
    """Sorts tensor names into units with a model table. Returns the units in the order they run and, sorted, the
    names the table leaves out. Fatal: a name no rule takes (nothing is dropped silently), a group no tensor belongs
    to, block or expert numbers that are not contiguous from 0, an expert group without experts, and blocks of one
    group that hold different numbers of experts."""
    patterns = [[pattern and re.compile(pattern, re.DOTALL) for pattern in (group.left_out, group.expert, group.stage)]
                for group in unit_groups]
    stage_names = [collections.defaultdict(list) for _ in unit_groups]             # per group, block: names
    expert_names = [collections.defaultdict(lambda: collections.defaultdict(list))  # per group, block: expert: names
                    for _ in unit_groups]
    left_out_names = []
    for name in names:
        for (left_out_pattern, expert_pattern, stage_pattern), stages, experts in zip(patterns, stage_names,
                                                                                      expert_names):
            if left_out_pattern and left_out_pattern.fullmatch(name):
                left_out_names.append(name)
                break
            match = expert_pattern and expert_pattern.fullmatch(name)
            if match:
                experts[int(match["block"])][int(match["expert"])].append(name)
                break
            match = stage_pattern.fullmatch(name)
            if match:
                stages[int(match.groupdict().get("block", 0))].append(name)
                break
        else:
            raise PackError(f"no rule of the model table takes the tensor {name!r}")

    units = []
    for group, (_, expert_pattern, stage_pattern), stages, experts in zip(unit_groups, patterns, stage_names,
                                                                          expert_names):
        if not stages:
            raise PackError(f"{group.label}: no tensor of the folder belongs to it")
        expert_count = len(experts.get(0, ()))                 # every block holds as many experts as the first
        if expert_pattern and not expert_count:
            raise PackError(f"{group.label} 0 holds no expert")
        for block in range(max(max(stages), max(experts, default = 0)) + 1):
            label = f"{group.label} {block}" if "block" in stage_pattern.groupindex else group.label
            if block not in stages:
                raise PackError(f"{label} is missing or holds experts only: its numbers must be contiguous from 0")
            experts_of_block = experts.get(block, {})
            if sorted(experts_of_block) != list(range(len(experts_of_block))):
                raise PackError(f"{label}: its expert numbers are not contiguous from 0")
            if len(experts_of_block) != expert_count:
                raise PackError(f"{label} holds {len(experts_of_block)} experts, {group.label} 0 holds {expert_count}")
            layer_index = block if group.has_layer_index else PACK_NO_LAYER
            units.append(Unit(UNIT_STAGE, layer_index, label, stages[block]))
            units += [Unit(UNIT_ON_DEMAND, layer_index, f"{label} expert {expert}", experts_of_block[expert])
                      for expert in range(expert_count)]
    return units, sorted(left_out_names)


def element_type_of(name, source):
    """The element type of a source tensor as a pack stores it, after everything a pack demands of a tensor has been
    checked. A tensor without dimensions holds one element."""
    where = f"{source.path}: tensor {name!r}"
    if source.dtype not in ELEMENT_TYPES:
        raise PackError(f"{where}: a pack stores no element type {source.dtype!r}")
    if len(source.shape) > PACK_MAXIMUM_DIMENSIONS:
        raise PackError(f"{where}: {len(source.shape)} dimensions, a pack stores at most {PACK_MAXIMUM_DIMENSIONS}")
    if any(dimension > UINT32_MAXIMUM for dimension in source.shape):
        raise PackError(f"{where}: a dimension of {max(source.shape)} does not fit 32 bits")
    element_type, element_byte_count = ELEMENT_TYPES[source.dtype]
    if math.prod(source.shape) * element_byte_count != source.byte_count:
        raise PackError(f"{where}: {source.byte_count} bytes stored, the shape {list(source.shape)} of {source.dtype} "
                        f"needs {math.prod(source.shape) * element_byte_count}")
    return element_type


def lay_out(units, sources):
    """Every size and offset of a pack, before a byte of it is written. In this format version a unit is one extent,
    so the index of a unit is the index of its extent."""
    extent_byte_counts, extent_pieces, tensor_rows = [], [], []
    for unit_index, unit in enumerate(units):
        pieces, end_in_extent = [], 0
        for name_bytes, name in sorted((name.encode("utf-8"), name) for name in unit.names):
            source, (weight_format, bits_per_weight) = sources[name], kernel_layouts.format_of(sources[name])
            byte_offset_in_extent = round_up(end_in_extent, MATRIX_ALIGNMENT_BYTES)
            # a matrix: with_matrices has checked its two tensors, and its bytes are not its elements' but its format's
            element_type = ELEMENT_TYPES[source.dtype][0] if weight_format else element_type_of(name, source)
            tensor_rows.append(TensorRow(name_bytes, unit_index, byte_offset_in_extent, source.byte_count,
                                         element_type, source.shape, weight_format, bits_per_weight))
            if source.byte_count:
                pieces.append((source.path, source.byte_offset, source.byte_count, byte_offset_in_extent))
            end_in_extent = byte_offset_in_extent + source.byte_count
        if not end_in_extent:
            raise PackError(f"the unit {unit.label} holds no bytes: an extent cannot be empty")
        extent_byte_counts.append(round_up(end_in_extent, PACK_BLOCK_BYTES))
        extent_pieces.append(pieces)
    tensor_rows.sort(key = lambda row: row.name)               # as unsigned bytes, a shorter name before its longer
    name_byte_count = sum(len(row.name) for row in tensor_rows)
    tables_byte_count = round_up(len(units) * (EXTENT_RECORD.size + UNIT_RECORD.size)
                                 + len(tensor_rows) * TENSOR_RECORD.size + name_byte_count, PACK_BLOCK_BYTES)
    if tables_byte_count > PACK_MAXIMUM_TABLES_BYTES:          # which also keeps every count inside its 32 bits
        raise PackError(f"the tables need {tables_byte_count} bytes, a pack allows {PACK_MAXIMUM_TABLES_BYTES}")
    extents, byte_offset = [], PACK_BLOCK_BYTES + tables_byte_count
    for unit_index, (byte_count, pieces) in enumerate(zip(extent_byte_counts, extent_pieces)):
        extents.append(Extent(byte_offset, byte_count, unit_index, pieces))
        byte_offset += byte_count
    return PackLayout(extents, tensor_rows, name_byte_count, tables_byte_count, byte_offset)


def write_extent(task):
    """Worker process: one extent assembled in memory from the source files (a plain tensor as it is stored, a matrix
    laid out by kernel_layouts.py), digested, and written at its offset. Returns the digest."""
    pack_path, extent_byte_offset, extent_byte_count, pieces = task
    extent_bytes = memoryview(bytearray(extent_byte_count))    # zero-filled: what lies between and behind the tensors
    source_files = {}
    try:
        for source, source_byte_offset, byte_count, byte_offset_in_extent in pieces:
            kernel_layouts.read_tensor(source, source_byte_offset, source_files,
                                       extent_bytes[byte_offset_in_extent:byte_offset_in_extent + byte_count])
    finally:
        for source_file in source_files.values():
            source_file.close()
    digest = digest_of(extent_bytes)
    descriptor = os.open(pack_path, os.O_WRONLY)
    try:
        write_exactly(descriptor, extent_byte_offset, extent_bytes, pack_path)
    finally:
        os.close(descriptor)
    return digest


def tables_of(layout, units, extent_digests):
    """The tables as a pack stores them: the extent records, the unit records, the tensor records, the bytes of the
    names, zero-filled to the end of their last block."""
    parts = [EXTENT_RECORD.pack(extent.byte_offset, extent.byte_count, digest, extent.unit_index, 0)
             for extent, digest in zip(layout.extents, extent_digests)]
    parts += [UNIT_RECORD.pack(unit.kind, unit.layer_index) for unit in units]
    name_byte_offset = 0
    for row in layout.tensor_rows:
        unused_dimensions = (0,) * (PACK_MAXIMUM_DIMENSIONS - len(row.dimensions))
        parts.append(TENSOR_RECORD.pack(row.byte_offset_in_extent, row.byte_count, row.extent_index, name_byte_offset,
                                        len(row.name), row.element_type, len(row.dimensions), *row.dimensions,
                                        *unused_dimensions, row.weight_format, row.bits_per_weight,
                                        *(0,) * PACK_TENSOR_RESERVED_WORDS))
        name_byte_offset += len(row.name)
    parts += [row.name for row in layout.tensor_rows]
    tables = b"".join(parts).ljust(layout.tables_byte_count, b"\0")
    if len(tables) != layout.tables_byte_count:
        raise RuntimeError(f"the tables hold {len(tables)} bytes, their layout says {layout.tables_byte_count}")
    return tables


def sealed_header_block(model_name, file_byte_count, tables_byte_count, tables_digest, extent_count, unit_count,
                        tensor_count, name_byte_count):
    """Block 0: the header, zero-filled to the end of the block, with the digest of the block as it is before the
    digest is written into it."""
    members = (PACK_MAGIC, PACK_FORMAT_VERSION, PACK_BLOCK_BYTES, model_name, file_byte_count, tables_byte_count,
               tables_digest, extent_count, unit_count, tensor_count, name_byte_count)
    unsealed_block = HEADER_RECORD.pack(*members, 0).ljust(PACK_BLOCK_BYTES, b"\0")
    return HEADER_RECORD.pack(*members, digest_of(unsealed_block)).ljust(PACK_BLOCK_BYTES, b"\0")


def convert(source_directory, output_path, jobs, overwrite, reference_directory = None, without_tables = False,
            table_hashing = None, table_piece_block_count = None):
    """Writes the pack of a folder of safetensors files and prints what it holds. table_*: the self-test's."""
    started = time.monotonic()
    if os.path.isdir(output_path):
        raise PackError(f"{output_path} is a directory")
    if os.path.lexists(output_path) and not overwrite:
        raise PackError(f"{output_path} exists: --overwrite replaces it")
    model_name = MODEL_NAME.encode("ascii")
    if len(model_name) >= MODEL_NAME_CAPACITY:                 # the header must hold the name and its NUL
        raise PackError(f"the model name {MODEL_NAME} does not fit the header")
    sources = read_source_folder(source_directory)
    if os.path.exists(output_path) and any(os.path.samefile(output_path, path)
                                           for path in {source.path for source in sources.values()}):
        raise PackError(f"{output_path} is one of the source files")
    units, left_out_names = plan_units(MODEL_UNIT_GROUPS, sources)
    units, sources, matrices_line = kernel_layouts.with_matrices(units, sources)
    units, sources, front_end_line = front_end_tables.with_tables(
        units, sources, reference_directory or source_directory)
    units, sources, tables_line = engram_tables.with_tables(
        units, sources, left_out_names, reference_directory or source_directory, without_tables, table_hashing,
        table_piece_block_count)
    layout = lay_out(units, sources)
    output_directory = os.path.dirname(os.path.abspath(output_path))
    free_byte_count = shutil.disk_usage(output_directory).free
    if free_byte_count < layout.file_byte_count:
        raise PackError(f"{output_directory}: the pack needs {layout.file_byte_count} bytes, {free_byte_count} are "
                        "free")

    temporary_path = output_path + TEMPORARY_SUFFIX
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, OUTPUT_FILE_MODE)
    try:
        try:
            os.ftruncate(descriptor, layout.file_byte_count)
            extent_digests = engram_tables.write_extents(temporary_path, layout.extents, jobs)
            tables = tables_of(layout, units, extent_digests)
            write_exactly(descriptor, PACK_BLOCK_BYTES, tables, temporary_path)
            # The header goes last: until here the file does not begin like a pack.
            write_exactly(descriptor, 0, sealed_header_block(
                model_name, layout.file_byte_count, layout.tables_byte_count, digest_of(tables), len(layout.extents),
                len(units), len(layout.tensor_rows), layout.name_byte_count), temporary_path)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        engram_tables.check_pack(temporary_path, *read_pack_tables(temporary_path)[1:])   # before it is named a pack
        if os.path.lexists(output_path) and not overwrite:
            raise PackError(f"{output_path} appeared during the run: --overwrite replaces it")
        os.replace(temporary_path, output_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary_path)
        raise

    kind_counts = ", ".join(f"{sum(unit.kind == kind for unit in units)} {kind_name}"
                            for kind, kind_name in enumerate(UNIT_KIND_NAMES))
    extent_byte_count = sum(extent.byte_count for extent in layout.extents)
    print(f"{output_path}: {MODEL_NAME}, {len(units)} units ({kind_counts}), {len(layout.extents)} extents, "
          f"{len(layout.tensor_rows)} tensors")
    print(f"  extents {extent_byte_count} bytes, tables {layout.tables_byte_count} bytes, file "
          f"{layout.file_byte_count} bytes ({layout.file_byte_count / BYTES_PER_GIB:.2f} GiB)")
    print(matrices_line)
    print(front_end_line)
    print(tables_line)
    print(f"  {rate_text(layout.file_byte_count, time.monotonic() - started)}")


# ---- verify ----------------------------------------------------------------------------------------------------------
def field_bytes(record, fields, name):
    byte_offset, byte_count = fields[name]
    return record[byte_offset:byte_offset + byte_count]


def field_number(record, fields, name):
    return int.from_bytes(field_bytes(record, fields, name), "little")


def records_of(tables, first_byte, record_count, record_byte_count):
    return (tables[byte_offset:byte_offset + record_byte_count]
            for byte_offset in range(first_byte, first_byte + record_count * record_byte_count, record_byte_count))


def read_pack_tables(pack_path):
    """The header, extents, units and tensors of a pack, each as a dictionary of its members (a tensor with its
    name), after every rule that pack.h lists for anemone_pack_open has been checked in the order of that list.
    The first rule that does not hold is a PackError that names it."""
    def broken(rule):
        return PackError(f"{pack_path}: {rule}")

    # the magic, the format version and the block size are the ones of pack.h, and the model name holds a NUL
    file_byte_count = os.path.getsize(pack_path)
    with open(pack_path, "rb") as handle:
        block = handle.read(PACK_BLOCK_BYTES)
        if len(block) != PACK_BLOCK_BYTES:
            raise broken(f"the file holds {len(block)} bytes: less than the block of the header")
        header = {name: field_number(block, HEADER_FIELDS, name) for name in HEADER_FIELDS
                  if name not in HEADER_BYTE_STRING_FIELDS}
        magic, model_name = field_bytes(block, HEADER_FIELDS, "magic"), field_bytes(block, HEADER_FIELDS, "model_name")
        if magic != PACK_MAGIC:
            raise broken(f"the magic is {magic!r}, not {PACK_MAGIC!r}")
        if header["format_version"] != PACK_FORMAT_VERSION:
            raise broken(f"the format version is {header['format_version']}, not {PACK_FORMAT_VERSION}")
        if header["block_byte_count"] != PACK_BLOCK_BYTES:
            raise broken(f"the block size is {header['block_byte_count']}, not {PACK_BLOCK_BYTES}")
        if b"\0" not in model_name:
            raise broken("the model name holds no NUL")
        header["model_name"] = model_name.split(b"\0")[0]
        # file_byte_count is the size of the file
        if header["file_byte_count"] != file_byte_count:
            raise broken(f"file_byte_count is {header['file_byte_count']}, the file holds {file_byte_count} bytes")
        # tables_byte_count is within the limit and exactly what the four counts need, rounded up to a block
        extent_count, unit_count, tensor_count = header["extent_count"], header["unit_count"], header["tensor_count"]
        counted_byte_count = (extent_count * EXTENT_RECORD_BYTES + unit_count * UNIT_RECORD_BYTES
                              + tensor_count * TENSOR_RECORD_BYTES + header["name_byte_count"])
        if header["tables_byte_count"] > PACK_MAXIMUM_TABLES_BYTES:
            raise broken(f"tables_byte_count is {header['tables_byte_count']}, above {PACK_MAXIMUM_TABLES_BYTES}")
        if header["tables_byte_count"] != round_up(counted_byte_count, PACK_BLOCK_BYTES):
            raise broken(f"tables_byte_count is {header['tables_byte_count']}, the four counts need "
                         f"{counted_byte_count} bytes, rounded up to a block")
        # the digest of the header (of block 0 with header_digest as zero) and the digest of the tables hold
        digest_byte_offset, digest_byte_count = HEADER_FIELDS["header_digest"]
        unsealed_block = block[:digest_byte_offset] + bytes(digest_byte_count) + block[digest_byte_offset
                                                                                        + digest_byte_count:]
        if digest_of(unsealed_block) != header["header_digest"]:
            raise broken("the digest of the header does not hold")
        tables = handle.read(header["tables_byte_count"])
    if len(tables) != header["tables_byte_count"]:
        raise broken("the tables end outside the file")
    if digest_of(tables) != header["tables_digest"]:
        raise broken("the digest of the tables does not hold")
    # there is at least one extent and one unit
    if not extent_count:
        raise broken("there is no extent")
    if not unit_count:
        raise broken("there is no unit")

    # every extent is a whole number of blocks and not empty; the first begins right after the tables, each begins
    # where the one before ends, and the last ends at the end of the file
    extents, end_of_previous = [], PACK_BLOCK_BYTES + header["tables_byte_count"]
    for extent_index, record in enumerate(records_of(tables, 0, extent_count, EXTENT_RECORD_BYTES)):
        extent = {name: field_number(record, EXTENT_FIELDS, name) for name in EXTENT_FIELDS}
        if not extent["byte_count"]:
            raise broken(f"extent {extent_index} is empty")
        if extent["byte_count"] % PACK_BLOCK_BYTES:
            raise broken(f"extent {extent_index}: {extent['byte_count']} bytes are no whole number of blocks")
        if extent["byte_offset"] != end_of_previous:
            raise broken(f"extent {extent_index} begins at byte {extent['byte_offset']}, not at {end_of_previous} "
                         "where " + ("the extent before ends" if extent_index else "the tables end"))
        end_of_previous += extent["byte_count"]
        extents.append(extent)
    if end_of_previous != file_byte_count:
        raise broken(f"the last extent ends at byte {end_of_previous}, the file at {file_byte_count}")
    # unit_index is 0 in the first extent, unit_count - 1 in the last, the same or one more from one to the next
    if extents[0]["unit_index"]:
        raise broken(f"extent 0 belongs to unit {extents[0]['unit_index']}, not to unit 0")
    for extent_index in range(1, extent_count):
        unit_index, unit_index_before = extents[extent_index]["unit_index"], extents[extent_index - 1]["unit_index"]
        if unit_index not in (unit_index_before, unit_index_before + 1):
            raise broken(f"extent {extent_index} belongs to unit {unit_index}, the extent before to unit "
                         f"{unit_index_before}")
    if extents[-1]["unit_index"] != unit_count - 1:
        raise broken(f"the last extent belongs to unit {extents[-1]['unit_index']}, the last unit is {unit_count - 1}")
    # a unit's kind is known, and an on-demand unit has a stage unit before it
    units, stage_unit_seen = [], False
    units_first_byte = extent_count * EXTENT_RECORD_BYTES
    for unit_index, record in enumerate(records_of(tables, units_first_byte, unit_count, UNIT_RECORD_BYTES)):
        unit = {name: field_number(record, UNIT_FIELDS, name) for name in UNIT_FIELDS}
        if unit["unit_kind"] >= len(UNIT_KIND_NAMES):
            raise broken(f"unit {unit_index}: its kind {unit['unit_kind']} is not known")
        if unit["unit_kind"] == UNIT_ON_DEMAND and not stage_unit_seen:
            raise broken(f"unit {unit_index} is on-demand and no stage unit lies before it")
        stage_unit_seen = stage_unit_seen or unit["unit_kind"] == UNIT_STAGE
        units.append(unit)

    tensors, name_before = [], None
    tensors_first_byte = units_first_byte + unit_count * UNIT_RECORD_BYTES
    names_first_byte = tensors_first_byte + tensor_count * TENSOR_RECORD_BYTES
    names = tables[names_first_byte:names_first_byte + header["name_byte_count"]]
    for tensor_index, record in enumerate(records_of(tables, tensors_first_byte, tensor_count, TENSOR_RECORD_BYTES)):
        tensor = {name: field_number(record, TENSOR_FIELDS, name) for name in TENSOR_FIELDS}
        dimension_bytes = field_bytes(record, TENSOR_FIELDS, "dimensions")
        tensor["dimensions"] = tuple(int.from_bytes(dimension_bytes[first_byte:first_byte + DIMENSION_BYTES], "little")
                                     for first_byte in range(0, len(dimension_bytes), DIMENSION_BYTES))
        where = f"tensor {tensor_index}"
        # a tensor's extent exists; it lies inside it and begins on a multiple of the matrix alignment
        if tensor["extent_index"] >= extent_count:
            raise broken(f"{where}: its extent {tensor['extent_index']} does not exist")
        if tensor["byte_offset_in_extent"] + tensor["byte_count"] > extents[tensor["extent_index"]]["byte_count"]:
            raise broken(f"{where} does not lie inside its extent {tensor['extent_index']}")
        if tensor["byte_offset_in_extent"] % MATRIX_ALIGNMENT_BYTES:
            raise broken(f"{where} begins at byte {tensor['byte_offset_in_extent']} of its extent, on no multiple of "
                         f"{MATRIX_ALIGNMENT_BYTES}")
        # it has at most the maximum of dimensions and the unused ones are zero; its element type is known
        if tensor["dimension_count"] > PACK_MAXIMUM_DIMENSIONS:
            raise broken(f"{where}: {tensor['dimension_count']} dimensions, at most {PACK_MAXIMUM_DIMENSIONS}")
        if any(tensor["dimensions"][tensor["dimension_count"]:]):
            raise broken(f"{where}: a dimension it does not use is not zero")
        if tensor["element_type"] not in ELEMENT_BYTE_COUNTS:
            raise broken(f"{where}: its element type {tensor['element_type']} is not known")
        # a plain tensor's byte count is the product of its dimensions times the size of an element, and its
        # bits_per_weight is zero; a matrix of a weight format has a format this build carries, two dimensions,
        # elements of UINT8 and exactly its format's bytes for its shape and bits_per_weight (kernel_layouts.py)
        needed_byte_count = (math.prod(tensor["dimensions"][:tensor["dimension_count"]])
                             * ELEMENT_BYTE_COUNTS[tensor["element_type"]])
        if tensor["weight_format"] != WEIGHT_FORMAT_NONE:
            if kernel_layouts.broken_matrix_rule(tensor):
                raise broken(where + kernel_layouts.broken_matrix_rule(tensor))
        elif tensor["byte_count"] != needed_byte_count:
            raise broken(f"{where}: {tensor['byte_count']} bytes, its dimensions and element type need "
                         f"{needed_byte_count}")
        elif tensor["bits_per_weight"]:
            raise broken(f"{where} is plain and its bits_per_weight is {tensor['bits_per_weight']}, not 0")
        # its name is not empty, lies inside the name bytes and sorts after the name before it
        if not tensor["name_byte_count"]:
            raise broken(f"{where}: its name is empty")
        if tensor["name_byte_offset"] + tensor["name_byte_count"] > header["name_byte_count"]:
            raise broken(f"{where}: its name lies outside the name bytes")
        tensor["name"] = names[tensor["name_byte_offset"]:tensor["name_byte_offset"] + tensor["name_byte_count"]]
        if name_before is not None and tensor["name"] <= name_before:
            raise broken(f"{where}: its name {tensor['name']!r} does not sort after the name before it")
        name_before = tensor["name"]
        tensors.append(tensor)

    # every reserved field is zero; and what the layout of pack.h calls zero-filled is
    for extent_index, extent in enumerate(extents):
        if extent["reserved_zero"]:
            raise broken(f"extent {extent_index}: its reserved field is not zero")
    for tensor_index, tensor in enumerate(tensors):
        if tensor["reserved_zero"]:
            raise broken(f"tensor {tensor_index}: a reserved field is not zero")
    if any(block[HEADER_BYTES:]):
        raise broken("block 0 is not zero-filled behind the header")
    if any(tables[counted_byte_count:]):
        raise broken("the tables are not zero-filled behind the names")
    return header, extents, units, tensors


def check_extent_digest(task):
    """Worker process: the digest of one extent of a pack, recomputed piece by piece and compared with its record;
    then what a matrix of the extent may not hold (checked_ranges: kernel_layouts.py)."""
    pack_path, extent_index, extent_byte_offset, extent_byte_count, stored_digest, checked_ranges = task
    digest, forbidden = 0, None
    piece_bytes = memoryview(bytearray(min(extent_byte_count, VERIFY_PIECE_BYTES)))
    with open(pack_path, "rb", buffering = 0) as handle:
        for piece_byte_offset in range(0, extent_byte_count, VERIFY_PIECE_BYTES):
            piece = piece_bytes[:min(VERIFY_PIECE_BYTES, extent_byte_count - piece_byte_offset)]
            read_exactly(handle, extent_byte_offset + piece_byte_offset, piece, pack_path)
            digest = (digest + digest_of(piece, piece_byte_offset // DIGEST_WORD_BYTES)) % UINT64_MODULUS
            forbidden = forbidden or kernel_layouts.forbidden_value_in(piece, piece_byte_offset, checked_ranges)
    if digest != stored_digest:
        raise PackError(f"{pack_path}: extent {extent_index}: the digest of its bytes is "
                        f"{digest:{DIGEST_TEXT_FORMAT}}, its record says {stored_digest:{DIGEST_TEXT_FORMAT}}")
    if forbidden:
        raise PackError(f"{pack_path}: extent {extent_index}: {forbidden}")


def verify(pack_path, jobs):
    """Checks a pack whole and prints what it holds; returns its header, extents, units and tensors. The first rule
    or digest that does not hold is a PackError."""
    started = time.monotonic()
    header, extents, units, tensors = read_pack_tables(pack_path)
    tables_text = engram_tables.check_pack(pack_path, extents, units, tensors)
    checked_ranges, matrices_text = kernel_layouts.checked_ranges(tensors)
    run_in_workers(check_extent_digest,
                   [(pack_path, extent_index, extent["byte_offset"], extent["byte_count"], extent["digest"],
                     checked_ranges.get(extent_index, ())) for extent_index, extent in enumerate(extents)],
                   [extent["byte_count"] for extent in extents], jobs, "verified")
    print(f"{pack_path}: every rule of pack.h and the digest of every extent hold: "
          f"{header['model_name'].decode('utf-8', 'replace')}, {len(units)} units, {len(extents)} extents, "
          f"{len(tensors)} tensors, {matrices_text}, {tables_text}, {header['file_byte_count']} bytes; "
          f"{rate_text(header['file_byte_count'], time.monotonic() - started)}")
    return header, extents, units, tensors


# ---- self-test -------------------------------------------------------------------------------------------------------
def reference_digest(data, first_word_index):
    """pack_digest.h word by word in Python's integers: what the numpy arithmetic of digest_of must equal."""
    digest = 0
    for word_index in range(len(data) // DIGEST_WORD_BYTES):
        word = int.from_bytes(data[word_index * DIGEST_WORD_BYTES:(word_index + 1) * DIGEST_WORD_BYTES], "little")
        value = word ^ ((first_word_index + word_index) * DIGEST_POSITION_MULTIPLIER % UINT64_MODULUS)
        value ^= value >> DIGEST_FIRST_SHIFT
        value = value * DIGEST_FIRST_MULTIPLIER % UINT64_MODULUS
        value ^= value >> DIGEST_SECOND_SHIFT
        value = value * DIGEST_SECOND_MULTIPLIER % UINT64_MODULUS
        value ^= value >> DIGEST_THIRD_SHIFT
        digest = (digest + value) % UINT64_MODULUS
    return digest


def self_test_units():
    """The self-test's model, unit by unit in the order the units run: (unit kind, layer index, tensors), a tensor
    being (name, dtype, shape)."""
    def named(prefix, tensors):
        return tuple((prefix + name, dtype, shape) for name, dtype, shape in tensors)

    units = [(UNIT_STAGE, PACK_NO_LAYER, SELF_TEST_VISION), (UNIT_STAGE, PACK_NO_LAYER, SELF_TEST_EMBEDDING)]
    for layer in range(SELF_TEST_LAYER_COUNT):
        units.append((UNIT_STAGE, layer, named(f"layers.{layer}.", SELF_TEST_LAYER)))
        units += [(UNIT_ON_DEMAND, layer, named(f"layers.{layer}.ffn.experts.{expert}.", SELF_TEST_EXPERT))
                  for expert in range(SELF_TEST_EXPERTS_PER_LAYER)]
    units.append((UNIT_STAGE, PACK_NO_LAYER, SELF_TEST_HEAD))
    for block in range(SELF_TEST_DRAFT_BLOCK_COUNT):
        units.append((UNIT_STAGE, PACK_NO_LAYER, named(f"mtp.{block}.", SELF_TEST_DRAFT_BLOCK)))
        units += [(UNIT_ON_DEMAND, PACK_NO_LAYER, named(f"mtp.{block}.ffn.experts.{expert}.", SELF_TEST_EXPERT))
                  for expert in range(SELF_TEST_EXPERTS_PER_DRAFT_BLOCK)]
    return units


def write_safetensors_file(path, tensors):
    """Writes (name, dtype, shape) tensors as one safetensors file, their bytes drawn from the names of the file and
    of the tensor (codes and scales: none that is no number); returns name: bytes."""
    header, contents, data_byte_count = {SAFETENSORS_METADATA_NAME: {"format": "pt"}}, {}, 0
    for name, dtype, shape in tensors:
        _, element_byte_count = ELEMENT_TYPES[dtype]
        byte_count = math.prod(shape) * element_byte_count
        drawn = hashlib.shake_256(f"{os.path.basename(path)}:{name}".encode("utf-8")).digest(byte_count)
        contents[name] = kernel_layouts.self_test_values(dtype, drawn)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [data_byte_count, data_byte_count + byte_count]}
        data_byte_count += byte_count
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(contents.values()))
    return contents


def flip_lowest_bit(path, byte_offset):
    with open(path, "r+b") as handle:
        handle.seek(byte_offset)
        (value,) = handle.read(1)
        handle.seek(byte_offset)
        handle.write(bytes((value ^ 1,)))


def reseal(pack_path):
    """Recomputes both digests of a pack the self-test changed on purpose, so that the change reaches the rule it
    was made for and is not stopped at a digest. Every other byte of the header stays as the change left it."""
    with open(pack_path, "r+b") as handle:
        block = bytearray(handle.read(PACK_BLOCK_BYTES))
        tables = handle.read(field_number(block, HEADER_FIELDS, "tables_byte_count"))
        # the header's own digest comes last: of the block with zero, the digest of no bytes, in its place
        for name, digested in (("tables_digest", tables), ("header_digest", b""), ("header_digest", block)):
            byte_offset, byte_count = HEADER_FIELDS[name]
            block[byte_offset:byte_offset + byte_count] = digest_of(digested).to_bytes(byte_count, "little")
        handle.seek(0)
        handle.write(block)


def self_test():
    """Proves the digest, a conversion and what verify and convert must refuse; returns the exit status."""
    failures = []

    def expect(condition, what):
        if not condition:
            failures.append(what)
            print(f"FAILED: {what}", file = sys.stderr)

    def expect_refusal(action, rule_text, what):
        """A case passes only if the action is refused AND the refusal says what the case was made for."""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                action()
        except PackError as refusal:
            expect(rule_text in str(refusal), f"{what}: refused, but with: {refusal}")
        else:
            expect(False, f"{what}: not refused")

    sample = hashlib.shake_256(b"digest sample").digest((DIGEST_PIECE_WORDS + SELF_TEST_SAMPLE_EXTRA_WORDS)
                                                        * DIGEST_WORD_BYTES)
    first_piece = sample[:SELF_TEST_FIRST_PIECE_WORDS * DIGEST_WORD_BYTES]
    expect(digest_of(DIGEST_KNOWN_ANSWER_BYTES) == DIGEST_KNOWN_ANSWER, "the known answer of pack_digest.h")
    expect(digest_of(b"") == 0, "the digest of no words is zero")
    expect(digest_of(sample) == reference_digest(sample, 0), "the digest equals plain integer arithmetic")
    expect(digest_of(first_piece, SELF_TEST_FAR_WORD_INDEX) == reference_digest(first_piece, SELF_TEST_FAR_WORD_INDEX),
           "the digest equals plain integer arithmetic far into an extent")
    expect((digest_of(first_piece) + digest_of(sample[len(first_piece):], SELF_TEST_FIRST_PIECE_WORDS))
           % UINT64_MODULUS == digest_of(sample), "a digest computed in pieces equals the whole")

    with tempfile.TemporaryDirectory() as folder:
        source_folder = os.path.join(folder, "source")
        pack_path, broken_path = os.path.join(folder, "model.pack"), os.path.join(folder, "broken.pack")
        os.mkdir(source_folder)
        stored_units = self_test_units()
        tensors_in_units = [tensor for _, _, unit_tensors in stored_units for tensor in unit_tensors]
        repeated = [tensor for tensor in tensors_in_units if tensor[0] == SELF_TEST_REPEATED_NAME]
        shared_out = [tensor for tensor in tensors_in_units if tensor[0] != SELF_TEST_REPEATED_NAME]
        shard_count = len(SELF_TEST_FILE_NAMES) - 1
        file_tensors = [shared_out[shard_index::shard_count] for shard_index in range(shard_count)]
        file_tensors[0] = file_tensors[0] + [SELF_TEST_REPEATED_EARLIER]
        file_tensors.append(repeated)
        stored_bytes = {}
        for file_name, tensors_of_file in zip(SELF_TEST_FILE_NAMES, file_tensors):     # in sorted order: the last wins
            stored_bytes.update(write_safetensors_file(os.path.join(source_folder, file_name), tensors_of_file))

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            convert(source_folder, pack_path, SELF_TEST_JOBS, False)
            header, extents, units, _ = verify(pack_path, SELF_TEST_JOBS)
        expect(f"{REPEATED_NAME_LINE} {SELF_TEST_REPEATED_NAME} in " in printed.getvalue(),
               "a repeated name is printed")
        expect(header["model_name"] == MODEL_NAME.encode("ascii"), "the header names the model")
        expect([(unit["unit_kind"], unit["layer_index"]) for unit in units]
               == [(kind, layer_index) for kind, layer_index, _ in stored_units],
               "the units are the model's, in the order they run")
        expect(len(extents) == len(stored_units), "every unit is one extent")
        with open(pack_path, "rb") as handle:
            pack_bytes = handle.read()
        # every tensor and every extent of the pack against a second writing of them, and the matrices' own cases
        kernel_layouts.self_test(folder, source_folder, pack_path, stored_units, stored_bytes, expect, expect_refusal)

        expect_refusal(lambda: convert(source_folder, pack_path, SELF_TEST_JOBS, False),
                       "exists: --overwrite replaces it", "an output that exists, without --overwrite")
        with contextlib.redirect_stdout(io.StringIO()):
            convert(source_folder, pack_path, SELF_TEST_JOBS, True)
        with open(pack_path, "rb") as handle:
            expect(handle.read() == pack_bytes, "a second conversion writes the same bytes")
        stray_path = os.path.join(source_folder, SELF_TEST_STRAY_FILE_NAME)
        for stray_tensor, rule_text in SELF_TEST_REFUSED_TENSORS:
            write_safetensors_file(stray_path, (stray_tensor,))
            expect_refusal(lambda: convert(source_folder, broken_path, SELF_TEST_JOBS, False), rule_text,
                           f"a folder that also holds {stray_tensor[0]} {list(stray_tensor[-1])}")
            expect(not os.path.lexists(broken_path) and not os.path.lexists(broken_path + TEMPORARY_SUFFIX),
                   f"the conversion refused for {stray_tensor[0]} leaves no file")
        os.remove(stray_path)

        def refused_by_verify(rule_text, what):
            expect_refusal(lambda: verify(broken_path, SELF_TEST_JOBS), rule_text, what)

        shutil.copyfile(pack_path, broken_path)
        flip_lowest_bit(broken_path, extents[-1]["byte_offset"])
        refused_by_verify(f"extent {len(extents) - 1}: the digest of its bytes", "one bit of an extent changed")
        shutil.copyfile(pack_path, broken_path)
        # the first member of the first tensor record: byte_offset_in_extent, a multiple of 64 until its bit flips
        flip_lowest_bit(broken_path,
                        PACK_BLOCK_BYTES + len(extents) * EXTENT_RECORD.size + len(units) * UNIT_RECORD.size)
        refused_by_verify("the digest of the tables", "one tensor record's offset changed")
        reseal(broken_path)
        refused_by_verify("on no multiple of", "one tensor record's offset changed and the digests made to fit")
        shutil.copyfile(pack_path, broken_path)
        os.truncate(broken_path, os.path.getsize(broken_path) - PACK_BLOCK_BYTES)
        refused_by_verify("file_byte_count is", "the file one block short")
        engram_tables.self_test(folder, source_folder, pack_path, expect, expect_refusal)

    if failures:
        return EXIT_STATUS_FAILURE
    print("make_pack self-test passed")
    return 0


def job_count(text):
    """The type of --jobs."""
    if not text.isascii() or not text.isdigit() or not 1 <= int(text) <= JOBS_LIMIT:
        raise argparse.ArgumentTypeError(f"{text!r} is no number from 1 to {JOBS_LIMIT}")
    return int(text)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest = "command", required = True)
    convert_arguments = commands.add_parser("convert", help = "write the pack of a folder of safetensors files")
    convert_arguments.add_argument("source_folder")
    convert_arguments.add_argument("output_pack_file")
    convert_arguments.add_argument("--overwrite", action = "store_true", help = "replace an output that exists")
    table_arguments = convert_arguments.add_mutually_exclusive_group()
    table_arguments.add_argument("--reference", metavar = "FOLDER", help = "the checkpoint with the tables' own code")
    table_arguments.add_argument("--without-tables", action = "store_true", help = "leave the n-gram tables out")
    verify_arguments = commands.add_parser("verify", help = "check every rule and every digest of a pack")
    verify_arguments.add_argument("pack_file")
    for command_arguments in (convert_arguments, verify_arguments):
        command_arguments.add_argument("--jobs", type = job_count, default = min(os.cpu_count() or 1, JOBS_LIMIT),
                                       help = "worker processes (default: the number of processors)")
    commands.add_parser("self-test", help = "prove the tool on a small model it writes itself")
    arguments = parser.parse_args()
    try:
        if arguments.command == "convert":
            convert(arguments.source_folder, arguments.output_pack_file, arguments.jobs, arguments.overwrite,
                    arguments.reference, arguments.without_tables)
        elif arguments.command == "verify":
            verify(arguments.pack_file, arguments.jobs)
        else:
            return self_test()
    except (PackError, OSError, concurrent.futures.BrokenExecutor) as failure:
        print(f"make_pack {arguments.command} FAILED: {failure}", file = sys.stderr)
        return EXIT_STATUS_FAILURE
    return 0


if __name__ == "__main__":
    import make_pack                                       # ONE copy of this file serves it and engram_tables.py
    sys.exit(make_pack.main())
