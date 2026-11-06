#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scans a W4A16 AutoRound folder of DeepSeek-V4.1-Flash (ledger row M14): whether its files are the repository's,
what it holds, what of it is the checkpoint's own, and how it fits the cards with 1M-token state. Reads both folders
and writes nothing.

    w4a16_pack_scan.py files <folder> --repository <owner/name> [--revision R] [--jobs N]
    w4a16_pack_scan.py scan <folder> --reference <checkpoint folder> [--jobs N]
    w4a16_pack_scan.py scales <folder> --reference <checkpoint folder> [--jobs N]

files lists the repository at the commit its revision names (huggingface_hub, which only this command imports) and
checks every file of it in the folder: an LFS file by its size and sha256, any other by its size and git blob id.
It names the files missing from the folder and those the repository does not hold (the download cache aside), and
the commits the folder's download records name.

scan reads the headers of both folders, and of their tensors only what it compares:
  - classes: every tensor name with its numbers as N; count, element types, shapes, bytes. The AutoRound matrices
    (qweight I32 of 8 codes a word along the inputs, qzeros I32 of 8 along the outputs, scales F16 [groups,
    outputs]) are checked against one another, which gives their group size; the n-gram tables' too. Every expert
    matrix of the checkpoint has one here of its name and shape, and none is here alone.
  - zero points: every qzeros word of the folder. Then a sample of matrices decoded with the zero points 7, 8 and 9
    and with the stored ones, against the checkpoint's 4-bit values (E2M1 times 2^(scale - 127), the low half of a
    byte first), and each group scale over the checkpoint's power of two; a sample of n-gram rows decoded in the
    four readings of a 4-bit code against the checkpoint's 8-bit rows.
  - trunk: every tensor that is no routed expert and no n-gram table, against the checkpoint's of its name: of the
    same element type and shape, byte for byte; bfloat16 here where the checkpoint holds E4M3 with a power-of-two
    scale per 32 x 32 block, against that product made bfloat16, bit for bit (and how many of those products a
    bfloat16 cannot hold).
  - fit: the folder as make_pack.py would lay it out (its model table: one stage unit for vision, the embedding,
    each layer, the head and each draft block, one on-demand unit for each expert), and the same with the trunk
    taken from the checkpoint and the qzeros left out where the scan found both lossless; placed as placement.c
    places, on the PRO with two and with four 170HX, both orders; with the 1M-token attention state of every card.

scales reads every group scale of every expert matrix, and the checkpoint's scale byte of the same 32 weights, and
counts the groups whose scale is exactly +-0.5 or +-0.75 times that power of two (the rule scan's sample shows),
those it is not, and the bytes that are no number; the classes of a layer's first matrix are computed a second time
from the values, as a check of the table that classifies them.

Every read of the two folders goes past the page cache (O_DIRECT where the system has it): the machine's serving
keeps its tables there. make_pack.py reads the safetensors headers through it, so each file is put out of it after
(posix_fadvise; pages a process maps stay). Needs Python 3.9 and numpy; files needs huggingface_hub. Exit status 0,
1 after any failure, a file that does not match or a group outside the rule, 2 for a command line it cannot read.
Convert tool: not part of the engine.
"""
import argparse
import collections
import concurrent.futures
import glob
import hashlib
import json
import math
import mmap
import os
import re
import sys
import time

import numpy

import kernel_layouts                                      # the checkpoint's 8-bit matrices as a pack stores them
import make_pack                                           # the safetensors reader, the model table, the layout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "golden"))
import check_pack_linears as golden                        # the values of the E4M3 and E2M1 codes

BYTES_PER_GIB = make_pack.BYTES_PER_GIB
BYTES_PER_MIB = 2 ** 20
EXIT_STATUS_FAILURE = 1
DEFAULT_JOBS = 2                                           # files read at once: serving shares the drive
PERCENT = 100.0

# ---- Reading past the page cache -------------------------------------------------------------------------------------
DIRECT_FLAG = getattr(os, "O_DIRECT", 0)                   # Linux; elsewhere plain reads
DIRECT_ALIGNMENT_BYTES = 4096                              # O_DIRECT: offset, length and buffer on this boundary
HASH_PIECE_BYTES = 64 * BYTES_PER_MIB
COMPARE_PIECE_BYTES = 256 * BYTES_PER_MIB

# ---- The repository --------------------------------------------------------------------------------------------------
DOWNLOAD_CACHE_FOLDER = os.path.join(".cache", "huggingface")
DOWNLOAD_RECORD_FOLDER = os.path.join(DOWNLOAD_CACHE_FOLDER, "download")
DOWNLOAD_RECORD_SUFFIX = ".metadata"                       # its first line: the commit the file was taken from
GIT_BLOB_HEADER = "blob {}\0"                              # git's object id of a file: sha1 of this and the bytes

# ---- The folder: AutoRound's auto_gptq packing, the n-gram tables ----------------------------------------------------
QUANTIZATION_CONFIG_FILE = "quantization_config.json"
PACKED_SUFFIX, ZEROS_SUFFIX, GROUP_SCALES_SUFFIX = ".qweight", ".qzeros", ".scales"
PACKED_DTYPE, GROUP_SCALES_DTYPE, BFLOAT16_DTYPE = "I32", "F16", "BF16"
CODE_BITS = 4
CODE_MASK = 2 ** CODE_BITS - 1
CODES_PER_WORD = 32 // CODE_BITS
CODES_PER_BYTE = 8 // CODE_BITS
CODE_SHIFTS = numpy.arange(CODES_PER_WORD, dtype = numpy.uint32) * CODE_BITS
SIGNED_CODE_OFFSET = 2 ** CODE_BITS                        # two's complement: a code at or above half is this less
CODE_MIDDLE = 2 ** (CODE_BITS - 1)                         # the zero of an offset code
ZERO_POINTS_TRIED = (CODE_MIDDLE - 1, CODE_MIDDLE, CODE_MIDDLE + 1)
NUMBER = make_pack.NUMBER
EXPERT_PATTERN = re.compile(rf"(?:layers|mtp)\.{NUMBER}\.ffn\.experts\.{NUMBER}\.w[123]\.(?:qweight|qzeros|scales)")
CHECKPOINT_EXPERT_PATTERN = re.compile(rf"(?:layers|mtp)\.{NUMBER}\.ffn\.experts\.{NUMBER}\.w[123]\.(?:weight|scale)")
TABLE_PATTERN = re.compile(rf"layers\.{NUMBER}\.engram\.embed\.(?:weight|scale)")
CLASS_NUMBER = re.compile(rf"\.{NUMBER}(?=\.|$)")
SHAPES_SHOWN = 2                                           # of a class
ZERO_WORDS_SHOWN = 4

# ---- The checkpoint --------------------------------------------------------------------------------------------------
WEIGHT_SUFFIX, SCALE_SUFFIX = ".weight", ".scale"
E4M3_DTYPE, E8M0_DTYPE, E2M1_PAIRS_DTYPE = "F8_E4M3", "F8_E8M0", "I8"
E8M0_BIAS = golden.E8M0_BIAS
E8M0_NOT_A_NUMBER = 0xFF                                   # the scale byte that stands for no number
SCALE_BLOCK = golden.E4M3_OUTPUTS_PER_SCALE                # an 8-bit weight's scale covers 32 x 32
INPUTS_PER_SCALE = kernel_layouts.INPUTS_PER_SCALE         # a 4-bit weight's, 32 inputs of one output
E4M3_VALUES = golden.e4m3_values()                         # float64 by code; NaN for the two that are no number
E2M1_VALUES = golden.code_values(kernel_layouts.WEIGHT_FORMAT_E2M1_SCALED)
FLOAT32_LOW_HALF_MASK = 0xFFFF                             # what a bfloat16 drops of a float32
BFLOAT16_SHIFT = 16
COMPARE_ROWS = 2048                                        # of an 8-bit matrix at a time: a multiple of SCALE_BLOCK
UNSIGNED_OF_BYTES = {1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}
HALF_BYTES = 2                                             # of an F16 or a BF16 element

# ---- The samples: which matrices and rows are decoded ----------------------------------------------------------------
SAMPLE_LAYERS = (0, 19, 39)                                # the first, the last encoder, the last decoder layer
SAMPLE_EXPERTS = (0, 191, 383)
SAMPLE_DRAFT_EXPERTS = ((0, 0),)                           # (draft block, expert)
SAMPLE_MATRICES = ("w1", "w2", "w3")
TABLE_SAMPLE_PLACES = 4                                    # row ranges spread evenly over a table
TABLE_SAMPLE_ROWS = 4096                                   # of each range, at most: ranges never overlap

# ---- The scale rule: every group of every expert ---------------------------------------------------------------------
# A group's F16 scale over the checkpoint's power of two of the same 32 weights, the rule scan's sample finds: the
# scale that puts the group's value of largest magnitude, 4 or 6 times that power, at -8 (code 0 less the zero point
# 8). A group is classified by its key, the checkpoint's scale byte above the 16 bits of its F16 scale, in a table of
# every key.
RULE_RATIOS = (0.5, -0.5, 0.75, -0.75)                     # classes 0 to 3, in this order
RULE_CLASS_OTHER, RULE_CLASS_NOT_A_NUMBER = len(RULE_RATIOS), len(RULE_RATIOS) + 1   # outside the rule
RULE_CLASS_COUNT = len(RULE_RATIOS) + 2
FLOAT16_BITS = 16
OTHER_RATIOS_KEPT = 16                                     # of a layer's groups outside the rule, the most common
OTHER_RATIOS_SHOWN = 8
GROUPS_OUTSIDE_SHOWN = 4

# ---- The fit ---------------------------------------------------------------------------------------------------------
# Card sizes as the driver reports them [b02 read_rate.log]; a budget is that less what the backend keeps
# (backend_cuda.c MEMORY_RESERVED_BYTES) and what e02's load kept free (--card_memory_kept_free_mib=8192), which the
# attention state and the scratch of a card live in (DECISIONS 1, design/gsq_rco_full.md 7).
CARD_BYTES = {"PRO": round(95.01 * BYTES_PER_GIB), "170HX": round(63.39 * BYTES_PER_GIB)}
BACKEND_RESERVED_BYTES = 256 * BYTES_PER_MIB
KEPT_FREE_BYTES = 8 * BYTES_PER_GIB
CARD_SETS = ((("170HX", "170HX", "PRO"), "PRO + two 170HX, e02's order"),
             (("PRO", "170HX", "170HX"), "PRO + two 170HX, the PRO first"),
             (("170HX",) * 4 + ("PRO",), "PRO + four 170HX, the PRO last"),
             (("PRO",) + ("170HX",) * 4, "PRO + four 170HX, the PRO first"))
# The draft head's 1M-token cache on the card of its first block, as design/gsq_rco_full.md 7 counts it (DECISIONS 9:
# 10.1 GiB with its cache, less 7.39 of weights); research/model.md 3.11 puts what the engine needs at 384 KiB.
DRAFT_CACHE_BYTES = round(2.7 * BYTES_PER_GIB)
DRAFT_CACHE_UNIT = "draft block 0"
# The attention state at 1,048,576 tokens (DECISIONS 1): the pools written at layers 2, 8, 14 (one entry for two
# tokens) and 20 (one a token), each read by its own layer and those after it up to the next pool's (layer 20's up
# to the last); a pool entry and its index key; the window of 128 rows of every layer. The draft blocks' windows
# are inside DRAFT_CACHE_BYTES.
CONTEXT_TOKENS = 2 ** 20
POOL_TOKENS_PER_ENTRY = {2: 2, 8: 2, 14: 2, 20: 1}         # by the layer that writes the pool
WINDOW_ROWS = 128
STATE_WIDTHS = ((16, 1024 + 256, 1024), (8, 544 + 132, 528))    # bits; bytes of an entry with its key; of a row
DEFAULT_STATE_BITS = 16

Matrix = collections.namedtuple("Matrix", ("outputs", "inputs", "group_size"))
Placement = collections.namedtuple("Placement", ("card_of_unit", "kept_per_owner", "bytes_lacking"))


class ScanError(Exception):
    """A folder that is not what the scan reads; the message names the place."""


def gib(byte_count):
    return byte_count / BYTES_PER_GIB


def share(part, whole):
    return PERCENT * part / whole if whole else 0.0


# ---- Reading ---------------------------------------------------------------------------------------------------------
class DirectFiles:
    """Files read past the page cache, opened once each; closed on leaving the with block."""

    def __init__(self):
        self.descriptors = {}

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        for descriptor in self.descriptors.values():
            os.close(descriptor)

    def read_into(self, path, byte_offset, buffer):
        """Fills an aligned buffer from an aligned offset; returns the bytes read, fewer only at the end of the
        file. A read that stops short on the alignment is continued; one that stops between two boundaries is the
        end of the file (another call there would be refused as unaligned)."""
        if path not in self.descriptors:
            self.descriptors[path] = os.open(path, os.O_RDONLY | DIRECT_FLAG)
        view, byte_count_read = memoryview(buffer), 0
        while byte_count_read < len(view):
            step = os.preadv(self.descriptors[path], [view[byte_count_read:byte_count_read +
                                                           make_pack.TRANSFER_LIMIT_BYTES]],
                             byte_offset + byte_count_read)
            byte_count_read += step
            if step == 0 or step % DIRECT_ALIGNMENT_BYTES:
                break
        return byte_count_read

    def read(self, source, first_byte = 0, byte_count = None):
        """The bytes [first_byte, first_byte + byte_count) of a tensor (all of it by default), numpy uint8."""
        byte_count = source.byte_count - first_byte if byte_count is None else byte_count
        if first_byte < 0 or byte_count < 0 or first_byte + byte_count > source.byte_count:
            raise ScanError(f"{source.path}: bytes {first_byte} to {first_byte + byte_count} of a tensor of "
                            f"{source.byte_count}")
        start = source.byte_offset + first_byte
        aligned_start = start // DIRECT_ALIGNMENT_BYTES * DIRECT_ALIGNMENT_BYTES
        aligned_end = make_pack.round_up(start + byte_count, DIRECT_ALIGNMENT_BYTES)
        buffer = mmap.mmap(-1, max(aligned_end - aligned_start, DIRECT_ALIGNMENT_BYTES))
        if self.read_into(source.path, aligned_start, buffer) < start + byte_count - aligned_start:
            raise ScanError(f"{source.path}: ends before byte {start + byte_count}")
        return numpy.frombuffer(buffer, dtype = numpy.uint8, count = byte_count, offset = start - aligned_start)


def whole_file(path):
    """A small file whole, bytes, read past the page cache as the tensors are."""
    with DirectFiles() as files:
        return files.read(make_pack.SourceTensor(path, 0, os.path.getsize(path), None, None)).tobytes()


def read_folder(folder):
    """make_pack.py's reading of every header of a folder. It reads them through the page cache, so each of its
    *.safetensors files is put out of it after, whole: read-ahead takes more than the header (on this machine's
    drive 128 pages stay of a header of 383,312 bytes)."""
    try:
        return make_pack.read_source_folder(folder)
    finally:
        if hasattr(os, "posix_fadvise"):                   # where it is missing, so is O_DIRECT: plain reads
            for path in glob.glob(os.path.join(glob.escape(folder), "*" + make_pack.SAFETENSORS_SUFFIX)):
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(descriptor)


def powers_of(codes):
    """The values of E8M0 scale bytes, float64: 2^(c - 127), no number for the byte that stands for none."""
    exponents = numpy.asarray(codes).astype(numpy.int64)
    return numpy.where(exponents == E8M0_NOT_A_NUMBER, numpy.nan, numpy.ldexp(1.0, exponents - E8M0_BIAS))


def hash_file(task):
    """Worker thread: (path, size the repository gives, its sha256 or None, its git blob id) -> (verdict, digest).
    An LFS file is checked by sha256, any other by its git blob id."""
    path, size, sha256, blob_id = task
    found_size = os.path.getsize(path)
    if found_size != size:
        return f"MISMATCH size {found_size}, the repository's {size}", None
    digest = hashlib.sha256() if sha256 else hashlib.sha1(GIT_BLOB_HEADER.format(size).encode("ascii"))
    buffer, byte_offset = mmap.mmap(-1, HASH_PIECE_BYTES), 0
    with DirectFiles() as files:
        while byte_offset < size:
            step = min(files.read_into(path, byte_offset, buffer), size - byte_offset)
            if step <= 0:
                return f"MISMATCH ends at byte {byte_offset} of {size}", None
            digest.update(memoryview(buffer)[:step])
            byte_offset += step
    expected = sha256 or blob_id
    found = digest.hexdigest()
    return ("ok" if found == expected else f"MISMATCH {found}, the repository's {expected}"), found


# ---- files -----------------------------------------------------------------------------------------------------------
def check_files(folder, repository, revision, jobs):
    """Prints a line for every file of the repository and one for each the folder lacks or adds; returns the count of
    files that are missing, do not match or are not the repository's (scan would read a stray *.safetensors file)."""
    try:
        import huggingface_hub
    except ImportError as error:
        raise ScanError(f"files needs huggingface_hub: {error}") from error
    try:
        api = huggingface_hub.HfApi()
        commit = api.model_info(repository, revision = revision).sha
        entries = list(api.list_repo_tree(repository, revision = commit, recursive = True))
    except Exception as error:                             # the hub's errors share no base class with OSError
        raise ScanError(f"the repository {repository} at {revision} cannot be listed: {error}") from error
    remote = {entry.path: entry for entry in entries if isinstance(entry, huggingface_hub.hf_api.RepoFile)}
    if not remote:
        raise ScanError(f"the repository {repository} at {revision} lists no file")
    local, recorded_commits = set(), collections.Counter()
    for directory, _, file_names in os.walk(folder):
        for file_name in file_names:
            path = os.path.relpath(os.path.join(directory, file_name), folder)
            if not path.startswith(DOWNLOAD_CACHE_FOLDER + os.sep):
                local.add(path)
            elif path.startswith(DOWNLOAD_RECORD_FOLDER + os.sep) and path.endswith(DOWNLOAD_RECORD_SUFFIX):
                record = whole_file(os.path.join(folder, path)).decode("utf-8", errors = "replace")
                recorded_commits[record.split("\n", 1)[0].strip()] += 1
    print(f"{repository} at {revision}: commit {commit}, {len(remote)} files, "
          f"{sum(entry.size for entry in remote.values())} bytes")
    print("  the folder's download records name: " +
          (", ".join(f"{name} ({count} files)" for name, count in recorded_commits.most_common()) or "none"))
    tasks = [(os.path.join(folder, path), entry.size, entry.lfs.sha256 if entry.lfs else None, entry.blob_id)
             for path, entry in sorted(remote.items()) if path in local]
    started, matching = time.monotonic(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers = jobs) as pool:
        for (path, size, sha256, _), (verdict, digest) in zip(tasks, pool.map(hash_file, tasks)):
            matching += verdict == "ok"
            print(f"  {verdict:<8} {'sha256' if sha256 else 'git id'} {digest or '-'} {size:>15} "
                  f"{os.path.relpath(path, folder)}", flush = True)
    missing, stray = sorted(set(remote) - local), sorted(local - set(remote))
    for path in missing:
        print(f"  MISSING  {path}")
    for path in stray:
        print(f"  NOT IN THE REPOSITORY {path}")
    checked = sum(size for _, size, _, _ in tasks)
    print(f"  {matching} of {len(remote)} files match; {len(tasks) - matching} do not, {len(missing)} missing, "
          f"{len(stray)} not in the repository; {make_pack.rate_text(checked, time.monotonic() - started)}")
    return len(remote) - matching + len(stray)


# ---- scan: classes ---------------------------------------------------------------------------------------------------
def kind_of(name):
    if EXPERT_PATTERN.fullmatch(name):
        return "draft expert" if name.startswith("mtp.") else "routed expert"
    return "n-gram table" if TABLE_PATTERN.fullmatch(name) else "trunk"


def print_classes(sources):
    """One line for every class of names, and the bytes of each kind."""
    classes = collections.defaultdict(lambda: [0, set(), collections.Counter(), 0])
    kinds = collections.Counter()
    for name, source in sources.items():
        entry = classes[CLASS_NUMBER.sub(".N", name)]
        entry[0] += 1
        entry[1].add(source.dtype)
        entry[2][source.shape] += 1
        entry[3] += source.byte_count
        kinds[kind_of(name)] += source.byte_count
    print(f"classes: {len(classes)}, {len(sources)} tensors, {sum(kinds.values())} bytes "
          f"({gib(sum(kinds.values())):.3f} GiB)")
    print(f"  {'count':>6} {'type':<8} {'bytes':>15} {'GiB':>9}  class  shapes")
    for class_name in sorted(classes):
        count, dtypes, shapes, byte_count = classes[class_name]
        shown = ", ".join(str(list(shape)) for shape, _ in shapes.most_common(SHAPES_SHOWN))
        more = f" and {len(shapes) - SHAPES_SHOWN} more" if len(shapes) > SHAPES_SHOWN else ""
        print(f"  {count:>6} {'/'.join(sorted(dtypes)):<8} {byte_count:>15} {gib(byte_count):>9.3f}  {class_name}  "
              f"{shown}{more}")
    print("  by kind: " + "; ".join(f"{kind} {gib(byte_count):.3f} GiB" for kind, byte_count in sorted(kinds.items())))


def autoround_matrices(sources):
    """Every AutoRound matrix of the folder, its name less the suffix: Matrix, its three tensors checked against one
    another. The first that does not agree is a ScanError."""
    matrices = {}
    for name in sorted(sources):
        if not EXPERT_PATTERN.fullmatch(name):
            continue
        base = name.rsplit(".", 1)[0]
        if base in matrices:
            continue
        packed, zeros, scales = (sources.get(base + suffix) for suffix in (PACKED_SUFFIX, ZEROS_SUFFIX,
                                                                            GROUP_SCALES_SUFFIX))
        if packed is None or zeros is None or scales is None:
            raise ScanError(f"{base}: qweight, qzeros and scales are not all there")
        if (packed.dtype, zeros.dtype, scales.dtype) != (PACKED_DTYPE, PACKED_DTYPE, GROUP_SCALES_DTYPE) or \
                not all(len(tensor.shape) == 2 for tensor in (packed, zeros, scales)):
            raise ScanError(f"{base}: not I32, I32 and F16 matrices")
        for tensor_name, tensor in zip((PACKED_SUFFIX, ZEROS_SUFFIX, GROUP_SCALES_SUFFIX), (packed, zeros, scales)):
            make_pack.element_type_of(base + tensor_name, tensor)          # its bytes are its shape's
        inputs, outputs = packed.shape[0] * CODES_PER_WORD, packed.shape[1]
        groups = scales.shape[0]
        if scales.shape[1] != outputs or not groups or inputs % groups or outputs % CODES_PER_WORD or \
                zeros.shape != (groups, outputs // CODES_PER_WORD):
            raise ScanError(f"{base}: qweight {list(packed.shape)}, qzeros {list(zeros.shape)} and scales "
                            f"{list(scales.shape)} do not agree")
        matrices[base] = Matrix(outputs, inputs, inputs // groups)
    return matrices


def table_layouts(sources):
    """The n-gram tables, their name less the suffix: (rows, codes a row, group size), checked."""
    tables = {}
    for name in sorted(sources):
        if TABLE_PATTERN.fullmatch(name) and name.endswith(WEIGHT_SUFFIX):
            base = name[:-len(WEIGHT_SUFFIX)]
            codes, scales = sources[name], sources.get(base + SCALE_SUFFIX)
            if scales is None or (codes.dtype, scales.dtype) != ("U8", GROUP_SCALES_DTYPE) or \
                    len(codes.shape) != 2 or len(scales.shape) != 2 or codes.shape[0] != scales.shape[0] or \
                    not scales.shape[1] or codes.shape[1] * CODES_PER_BYTE % scales.shape[1]:
                raise ScanError(f"{base}: no U8 [rows, bytes] with F16 [rows, groups] beside it")
            make_pack.element_type_of(name, codes)                         # its bytes are its shape's
            make_pack.element_type_of(base + SCALE_SUFFIX, scales)
            codes_per_row = codes.shape[1] * CODES_PER_BYTE
            tables[base] = (codes.shape[0], codes_per_row, codes_per_row // scales.shape[1])
    return tables


def print_quantization(folder, matrices, tables):
    by_kind = collections.Counter(("draft" if base.startswith("mtp.") else "layers") for base in matrices)
    group_sizes = collections.Counter(matrix.group_size for matrix in matrices.values())
    shapes = collections.Counter((matrix.outputs, matrix.inputs) for matrix in matrices.values())
    print(f"AutoRound matrices: {len(matrices)} ({by_kind['layers']} in the layers, {by_kind['draft']} in the draft "
          f"head), {CODE_BITS}-bit codes {CODES_PER_WORD} to an I32 word, group sizes " +
          ", ".join(f"{size} ({count})" for size, count in sorted(group_sizes.items())) + "; outputs x inputs " +
          ", ".join(f"{outputs} x {inputs} ({count})" for (outputs, inputs), count in sorted(shapes.items())))
    for base, (rows, codes_per_row, group_size) in tables.items():
        print(f"n-gram table {base}: {rows} rows of {codes_per_row} {CODE_BITS}-bit codes "
              f"({codes_per_row // CODES_PER_BYTE} bytes) and {codes_per_row // group_size} F16 scales: group size "
              f"{group_size}")
    config_path = os.path.join(folder, QUANTIZATION_CONFIG_FILE)
    if os.path.exists(config_path):
        try:
            config = json.loads(whole_file(config_path).decode("utf-8"))
        except (ValueError, RecursionError) as error:
            raise ScanError(f"{config_path}: cannot be read: {error}") from error
        if not isinstance(config, dict):
            raise ScanError(f"{config_path}: no JSON object")
        stated = {key: config.get(key) for key in ("quant_method", "packing_format", "bits", "group_size", "sym",
                                                    "data_type", "iters")}
        agrees = stated["bits"] == CODE_BITS and set(group_sizes) == {stated["group_size"]}
        print(f"  {QUANTIZATION_CONFIG_FILE}: {stated}; {'agrees' if agrees else 'DOES NOT AGREE'} with the tensors")


# ---- scan: zero points, decoded matrices and rows --------------------------------------------------------------------
def zero_words(sources, matrices):
    """Every qzeros word of the folder, counted by value; read in the order the files hold them."""
    counts = collections.Counter()
    tensors = sorted((sources[base + ZEROS_SUFFIX] for base in matrices), key = lambda source: (source.path,
                                                                                                source.byte_offset))
    with DirectFiles() as files:
        for source in tensors:
            values, value_counts = numpy.unique(files.read(source).view("<u4"), return_counts = True)
            counts.update(dict(zip(values.tolist(), value_counts.tolist())))
    return counts, sum(source.byte_count for source in tensors)


def unpacked_codes(words, along_rows):
    """The 4-bit codes of I32 words, code k of a word at bits 4 k: along_rows puts them on consecutive rows
    ([words, columns] -> [8 words, columns]), else on consecutive columns."""
    words = words.astype(numpy.uint32)
    if along_rows:
        return ((words[:, None, :] >> CODE_SHIFTS[None, :, None]) & CODE_MASK).reshape(-1, words.shape[1])
    return ((words[:, :, None] >> CODE_SHIFTS[None, None, :]) & CODE_MASK).reshape(words.shape[0], -1)


def decode_against_checkpoint(task):
    """Worker process: one AutoRound matrix against the checkpoint's 4-bit matrix of its name. Returns a dictionary:
    per zero point tried and for the stored zeros, the weights equal to the checkpoint's and the squared difference;
    the checkpoint's squared norm and weight count; the group scales over the checkpoint's power of two, counted;
    and the weights equal with the stored zeros, by that ratio."""
    _, matrix, packed, zeros, scales, codes_source, exponents_source = task
    outputs, inputs, group_size = matrix
    with DirectFiles() as files:
        codes = unpacked_codes(files.read(packed).view("<u4").reshape(inputs // CODES_PER_WORD, outputs), True).T
        stored_zeros = unpacked_codes(files.read(zeros).view("<u4").reshape(-1, outputs // CODES_PER_WORD), False)
        group_scales = files.read(scales).view("<f2").reshape(-1, outputs).astype(numpy.float64)
        checkpoint_codes = files.read(codes_source).reshape(outputs, inputs // CODES_PER_BYTE)
        powers = powers_of(files.read(exponents_source).reshape(outputs, inputs // INPUTS_PER_SCALE))
    checkpoint_codes = numpy.stack((checkpoint_codes & CODE_MASK, checkpoint_codes >> CODE_BITS), axis = -1)
    original = E2M1_VALUES[checkpoint_codes.reshape(outputs, inputs)] * numpy.repeat(powers, INPUTS_PER_SCALE, 1)
    per_weight_scales = numpy.repeat(group_scales.T, group_size, axis = 1)
    sums = {"original": float(numpy.square(original).sum()), "weights": original.size}
    equal_by_ratio = collections.Counter()
    for zero_point in ZERO_POINTS_TRIED + ("stored",):
        zero = numpy.repeat(stored_zeros.T, group_size, axis = 1) if zero_point == "stored" else zero_point
        decoded = (codes.astype(numpy.int64) - zero) * per_weight_scales
        equal = decoded == original
        sums[zero_point] = (int(equal.sum()), float(numpy.square(decoded - original).sum()))
        if zero_point == "stored" and group_size == INPUTS_PER_SCALE:
            ratios = group_scales.T / powers                               # [outputs, groups]
            equal_in_group = equal.reshape(outputs, -1, group_size).sum(axis = 2)
            values, inverse, counts = numpy.unique(ratios, return_inverse = True, return_counts = True)
            equal_counts = numpy.bincount(inverse.ravel(), weights = equal_in_group.ravel(), minlength = len(values))
            equal_by_ratio.update(dict(zip(values.tolist(), equal_counts.astype(numpy.int64).tolist())))
            sums["ratios"] = collections.Counter(dict(zip(values.tolist(), counts.tolist())))
    sums["equal by ratio"] = equal_by_ratio
    return sums


def expert_pairs(matrices, reference):
    """Every AutoRound matrix beside the checkpoint's 4-bit matrix of its name, base: (Matrix, its E2M1 codes, its
    scale bytes), each checked. An expert matrix of one folder only, or of another shape, is a ScanError."""
    checkpoint_bases = {name.rsplit(".", 1)[0] for name in reference if CHECKPOINT_EXPERT_PATTERN.fullmatch(name)}
    for alone, where in ((sorted(set(matrices) - checkpoint_bases), "here"),
                         (sorted(checkpoint_bases - set(matrices)), "in the checkpoint")):
        if alone:
            raise ScanError(f"{len(alone)} expert matrices only {where}, the first {alone[0]}")
    pairs = {}
    for base, matrix in sorted(matrices.items()):
        codes, exponents = reference.get(base + WEIGHT_SUFFIX), reference.get(base + SCALE_SUFFIX)
        if codes is None or exponents is None:
            raise ScanError(f"{base}: the checkpoint holds no {WEIGHT_SUFFIX} and {SCALE_SUFFIX} of it")
        make_pack.element_type_of(base + WEIGHT_SUFFIX, codes)
        make_pack.element_type_of(base + SCALE_SUFFIX, exponents)
        if (codes.dtype, exponents.dtype) != (E2M1_PAIRS_DTYPE, E8M0_DTYPE) or matrix.inputs % INPUTS_PER_SCALE or \
                codes.shape != (matrix.outputs, matrix.inputs // CODES_PER_BYTE) or \
                exponents.shape != (matrix.outputs, matrix.inputs // INPUTS_PER_SCALE):
            raise ScanError(f"{base}: the checkpoint's {list(codes.shape)} and {list(exponents.shape)} are no 4-bit "
                            f"matrix of {matrix.outputs} x {matrix.inputs}")
        pairs[base] = (matrix, codes, exponents)
    return pairs


def print_expert_decode(sources, pairs, jobs):
    """The sample of SAMPLE_*, decoded in worker processes, and the sums printed."""
    bases = [f"layers.{layer}.ffn.experts.{expert}.{name}" for layer in SAMPLE_LAYERS for expert in SAMPLE_EXPERTS
             for name in SAMPLE_MATRICES]
    bases += [f"mtp.{block}.ffn.experts.{expert}.{name}" for block, expert in SAMPLE_DRAFT_EXPERTS
              for name in SAMPLE_MATRICES]
    tasks = []
    for base in bases:
        if base not in pairs:
            raise ScanError(f"{base}: a matrix of the sample that neither folder holds")
        matrix, codes, exponents = pairs[base]
        tasks.append((base, matrix, sources[base + PACKED_SUFFIX], sources[base + ZEROS_SUFFIX],
                      sources[base + GROUP_SCALES_SUFFIX], codes, exponents))
    results = make_pack.run_in_workers(decode_against_checkpoint, tasks, [task[2].byte_count for task in tasks],
                                       jobs, "decoding experts")
    weights = sum(result["weights"] for result in results)
    original_norm = math.sqrt(sum(result["original"] for result in results))
    drafts = ", ".join(f"draft block {block} expert {expert}" for block, expert in SAMPLE_DRAFT_EXPERTS)
    print(f"experts decoded against the checkpoint's 4-bit values: {len(tasks)} matrices, {weights} weights (layers "
          f"{', '.join(map(str, SAMPLE_LAYERS))}, experts {', '.join(map(str, SAMPLE_EXPERTS))}; {drafts}; "
          f"{', '.join(SAMPLE_MATRICES)} of each)")
    for zero_point in ZERO_POINTS_TRIED + ("stored",):
        equal = sum(result[zero_point][0] for result in results)
        error = math.sqrt(sum(result[zero_point][1] for result in results))
        print(f"  zero point {zero_point}: {share(equal, weights):.2f}% of the weights equal the checkpoint's; "
              f"difference {share(error, original_norm):.3f}% of its norm")
    ratios, equal_by_ratio = collections.Counter(), collections.Counter()
    for result in results:
        ratios.update(result.get("ratios", {}))
        equal_by_ratio.update(result["equal by ratio"])
    groups = sum(ratios.values())
    if groups:
        print("  a group's scale over the checkpoint's power of two, share of groups (and of their weights equal "
              "with the stored zeros): " + "; ".join(
                  f"{ratio:g}: {share(count, groups):.2f}% "
                  f"({share(equal_by_ratio[ratio], count * INPUTS_PER_SCALE):.1f}%)"
                  for ratio, count in sorted(ratios.items())))


def print_table_decode(sources, reference, tables):
    """A sample of rows of every n-gram table decoded in the four readings of a 4-bit code against the checkpoint's
    E4M3 rows times their power-of-two scales."""
    readings = [(order, coding) for order in ("low half first", "high half first")
                for coding in ("two's complement", "offset 8")]
    for base, (rows, codes_per_row, group_size) in tables.items():
        names = (base + WEIGHT_SUFFIX, base + SCALE_SUFFIX)
        checkpoint_codes, checkpoint_exponents = (reference.get(name) for name in names)
        if checkpoint_codes is None or checkpoint_exponents is None or checkpoint_codes.dtype != E4M3_DTYPE or \
                checkpoint_codes.shape != (rows, codes_per_row) or checkpoint_exponents.dtype != E8M0_DTYPE or \
                len(checkpoint_exponents.shape) != 2 or checkpoint_exponents.shape[0] != rows or \
                not checkpoint_exponents.shape[1] or codes_per_row % checkpoint_exponents.shape[1]:
            raise ScanError(f"{base}: the checkpoint holds no table of {rows} rows of {codes_per_row} values")
        for name, tensor in zip(names, (checkpoint_codes, checkpoint_exponents)):
            make_pack.element_type_of(name, tensor)                         # its bytes are its shape's
        checkpoint_group = codes_per_row // checkpoint_exponents.shape[1]
        sums = collections.defaultdict(lambda: [0, 0.0])
        original_square, negative_scales, sampled_rows = 0.0, 0, 0
        with DirectFiles() as files:
            for place in range(TABLE_SAMPLE_PLACES):
                first = rows * place // TABLE_SAMPLE_PLACES
                count = min(TABLE_SAMPLE_ROWS, rows * (place + 1) // TABLE_SAMPLE_PLACES - first)
                if not count:
                    continue
                codes = files.read(sources[names[0]], first * codes_per_row // CODES_PER_BYTE,
                                   count * codes_per_row // CODES_PER_BYTE).reshape(count, -1)
                scale_row_bytes = codes_per_row // group_size * HALF_BYTES
                scales = files.read(sources[names[1]], first * scale_row_bytes,
                                    count * scale_row_bytes).view("<f2").astype(numpy.float64)
                scales = numpy.repeat(scales.reshape(count, -1), group_size, axis = 1)
                original = E4M3_VALUES[files.read(checkpoint_codes, first * codes_per_row,
                                                  count * codes_per_row).reshape(count, -1)]
                exponents = files.read(checkpoint_exponents, first * checkpoint_exponents.shape[1],
                                       count * checkpoint_exponents.shape[1]).reshape(count, -1)
                original = original * numpy.repeat(powers_of(exponents), checkpoint_group, 1)
                original_square += float(numpy.square(original).sum())
                negative_scales += int((scales < 0).sum()) // group_size
                sampled_rows += count
                halves = (codes & CODE_MASK, codes >> CODE_BITS)
                for order, coding in readings:
                    values = numpy.stack(halves if order.startswith("low") else halves[::-1], axis = -1)
                    values = values.reshape(count, -1).astype(numpy.int64)
                    values = numpy.where(values >= CODE_MIDDLE, values - SIGNED_CODE_OFFSET, values) \
                        if coding.startswith("two") else values - CODE_MIDDLE
                    decoded = values * scales
                    sums[(order, coding)][0] += int((decoded == original).sum())
                    sums[(order, coding)][1] += float(numpy.square(decoded - original).sum())
        values_sampled = sampled_rows * codes_per_row
        print(f"n-gram table {base}: {sampled_rows} rows in {TABLE_SAMPLE_PLACES} ranges against the checkpoint's; "
              f"negative group scales {share(negative_scales, values_sampled // group_size):.2f}%")
        for reading in readings:
            equal, square = sums[reading]
            print(f"  {reading[0]}, {reading[1]}: {share(equal, values_sampled):.2f}% of the values equal; "
                  f"difference {share(math.sqrt(square), math.sqrt(original_square)):.3f}% of the norm")


# ---- scan: the trunk -------------------------------------------------------------------------------------------------
def compare_tensor(task):
    """Worker process: (kind, name, the folder's tensor, the checkpoint's, its scales or None) -> (name, kind,
    elements, elements equal, values a bfloat16 cannot hold, the first difference or None)."""
    kind, name, mine, theirs, scales = task
    equal, unrepresentable, first_difference = 0, 0, None
    with DirectFiles() as files:
        if kind == "same":
            element_bytes = make_pack.ELEMENT_TYPES[mine.dtype][1] if mine.dtype in make_pack.ELEMENT_TYPES else 1
            elements = mine.byte_count // element_bytes
            piece = COMPARE_PIECE_BYTES // element_bytes * element_bytes
            for first_byte in range(0, mine.byte_count, piece):
                count = min(piece, mine.byte_count - first_byte)
                left, right = (files.read(source, first_byte, count).view(UNSIGNED_OF_BYTES[element_bytes])
                               for source in (mine, theirs))
                same = left == right
                equal += int(same.sum())
                if first_difference is None and not same.all():
                    first_difference = f"element {first_byte // element_bytes + int(numpy.argmin(same))}"
            return name, kind, elements, equal, unrepresentable, first_difference
        outputs, inputs = theirs.shape
        elements = outputs * inputs
        for first_row in range(0, outputs, COMPARE_ROWS):
            rows = min(COMPARE_ROWS, outputs - first_row)
            codes = files.read(theirs, first_row * inputs, rows * inputs).reshape(rows, inputs)
            block_rows = slice(first_row // SCALE_BLOCK, -(-(first_row + rows) // SCALE_BLOCK))
            exponents = files.read(scales, block_rows.start * scales.shape[1],
                                   (block_rows.stop - block_rows.start) * scales.shape[1])
            powers = numpy.ldexp(1.0, exponents.reshape(-1, scales.shape[1]).astype(numpy.int64) - E8M0_BIAS)
            powers = numpy.repeat(numpy.repeat(powers, SCALE_BLOCK, axis = 0)[:rows], SCALE_BLOCK, axis = 1)
            values = E4M3_VALUES[codes] * powers[:, :inputs]
            single = values.astype(numpy.float32)
            bits = single.view(numpy.uint32)
            exact = (single.astype(numpy.float64) == values) & ((bits & FLOAT32_LOW_HALF_MASK) == 0)
            unrepresentable += int(numpy.count_nonzero(~exact & ~numpy.isnan(values)))
            expected = (bits >> BFLOAT16_SHIFT).astype(numpy.uint16)
            found = files.read(mine, first_row * inputs * HALF_BYTES, rows * inputs * HALF_BYTES)
            found = found.view("<u2").reshape(rows, inputs)
            same = expected == found
            equal += int(same.sum())
            if first_difference is None and not same.all():
                row, column = numpy.argwhere(~same)[0].tolist()
                first_difference = (f"[{first_row + row}, {column}]: {found[row, column]:#06x}, the checkpoint's "
                                    f"code {codes[row, column]:#04x} times 2^{math.log2(powers[row, column]):g} gives "
                                    f"{expected[row, column]:#06x}")
    return name, kind, elements, equal, unrepresentable, first_difference


def trunk_tasks(sources, reference):
    """The comparisons of the trunk, and the names that have no counterpart (or another kind of one)."""
    tasks, unmatched, taken_scales = [], [], set()
    for name in sorted(sources):
        if kind_of(name) != "trunk":
            continue
        mine, theirs = sources[name], reference.get(name)
        scale_name = name[:-len(WEIGHT_SUFFIX)] + SCALE_SUFFIX if name.endswith(WEIGHT_SUFFIX) else None
        scales = reference.get(scale_name) if scale_name else None
        if theirs is None:
            unmatched.append(f"{name}: not in the checkpoint")
            continue
        for tensor_name, tensor in ((name, mine), (name, theirs), (scale_name, scales)):
            if tensor is not None:
                make_pack.element_type_of(tensor_name, tensor)             # its bytes are its shape's
        if (mine.dtype, mine.shape) == (theirs.dtype, theirs.shape):
            tasks.append(("same", name, mine, theirs, None))
        elif (mine.dtype, theirs.dtype) == (BFLOAT16_DTYPE, E4M3_DTYPE) and scales is not None and \
                mine.shape == theirs.shape and len(theirs.shape) == 2 and theirs.shape[1] % SCALE_BLOCK == 0 and \
                scales.dtype == E8M0_DTYPE and \
                scales.shape == (-(-theirs.shape[0] // SCALE_BLOCK), theirs.shape[1] // SCALE_BLOCK):
            tasks.append(("8-bit", name, mine, theirs, scales))
            taken_scales.add(scale_name)
        else:
            unmatched.append(f"{name}: {mine.dtype} {list(mine.shape)} here, {theirs.dtype} {list(theirs.shape)} "
                             "in the checkpoint")
    for name in sorted(reference):
        if name not in sources and name not in taken_scales and not CHECKPOINT_EXPERT_PATTERN.fullmatch(name):
            unmatched.append(f"{name}: only in the checkpoint")
    return tasks, unmatched


def print_trunk(sources, reference, jobs):
    """Returns the names of the matrices that are bfloat16 here and 8-bit in the checkpoint, when every one of them
    and every other trunk tensor is the checkpoint's; else None."""
    tasks, unmatched = trunk_tasks(sources, reference)
    results = make_pack.run_in_workers(compare_tensor, tasks, [task[2].byte_count for task in tasks], jobs,
                                       "comparing the trunk")
    lossless = not unmatched
    for kind, title in (("8-bit", "bfloat16 here, E4M3 with scales in the checkpoint"),
                        ("same", "the same element type and shape in both")):
        rows = [result for result in results if result[1] == kind]
        classes = collections.Counter(CLASS_NUMBER.sub(".N", result[0]) for result in rows)
        elements, equal = sum(result[2] for result in rows), sum(result[3] for result in rows)
        unrepresentable = sum(result[4] for result in rows)
        byte_count = sum(task[2].byte_count for task in tasks if task[0] == kind)
        print(f"trunk, {title}: {len(rows)} tensors, {elements} elements, {gib(byte_count):.3f} GiB here; "
              f"{equal} equal bit for bit ({share(equal, elements):.6f}%)" +
              (f"; checkpoint values a bfloat16 cannot hold: {unrepresentable}" if kind == "8-bit" else ""))
        print("  " + ", ".join(f"{class_name} {count}" for class_name, count in sorted(classes.items())))
        lossless = lossless and unrepresentable == 0
        for name, _, count, equal_count, _, first_difference in rows:
            if equal_count != count:
                lossless = False
                print(f"  DIFFERS {name}: {count - equal_count} of {count}; first at {first_difference}")
    for line in unmatched:
        print(f"  UNMATCHED {line}")
    return [result[0] for result in results if result[1] == "8-bit"] if lossless else None


# ---- scan: the fit ---------------------------------------------------------------------------------------------------
def unit_bytes(sources):
    """The units of make_pack.py's model table over these tensors, in the order they run, with the bytes of each
    as a pack lays it out; and the bytes of what the table leaves out (the n-gram tables)."""
    units, left_out = make_pack.plan_units(make_pack.MODEL_UNIT_GROUPS, sources)
    units, sources, _ = kernel_layouts.with_matrices(units, sources)
    layout = make_pack.lay_out(units, sources)
    return units, [extent.byte_count for extent in layout.extents], sum(sources[name].byte_count for name in left_out)


def place(units, byte_counts, budgets):
    """placement.c's placement: stage units in order onto the cards in order, each with its first N on-demand
    units, N the largest that fits; the draft cache rides on its unit."""
    asked_by_stage, stage = {}, None
    for index, (unit, byte_count) in enumerate(zip(units, byte_counts)):
        if unit.kind == make_pack.UNIT_STAGE:
            stage = index
            asked_by_stage[stage] = [byte_count + (DRAFT_CACHE_BYTES if unit.label == DRAFT_CACHE_UNIT else 0)]
        elif unit.kind == make_pack.UNIT_ON_DEMAND:
            asked_by_stage[stage].append(asked_by_stage[stage][-1] + byte_count)       # prefix sums

    def walk(kept):
        card, left, lacking, card_of_unit, kept_of_owner = 0, budgets[0], 0, [], kept
        for index, unit in enumerate(units):
            where = None
            if unit.kind == make_pack.UNIT_STAGE:
                asked = asked_by_stage[index][min(kept, len(asked_by_stage[index]) - 1)]
                while asked > left and card + 1 < len(budgets):
                    card, left = card + 1, budgets[card + 1]
                lacking += max(asked - left, 0)
                left, kept_of_owner, where = max(left - asked, 0), 0, card
            elif unit.kind == make_pack.UNIT_ON_DEMAND and kept_of_owner < kept:
                kept_of_owner, where = kept_of_owner + 1, card
            card_of_unit.append(where)
        return card_of_unit, lacking

    most = max(len(asked) - 1 for asked in asked_by_stage.values())
    card_of_unit, lacking = walk(0)
    if lacking:
        return Placement(card_of_unit, 0, lacking)
    fitting, highest = 0, most
    while fitting < highest:
        middle = highest - (highest - fitting) // 2
        if walk(middle)[1] == 0:
            fitting = middle
        else:
            highest = middle - 1
    return Placement(walk(fitting)[0], fitting, 0)


def state_bytes(layers, bits):
    """The 1M-token attention state of a card that holds these layers, at 16 or 8 bits."""
    _, entry_bytes, row_bytes = next(width for width in STATE_WIDTHS if width[0] == bits)
    sources = sorted(POOL_TOKENS_PER_ENTRY)
    pools = {max(source for source in sources if source <= layer) for layer in layers if layer >= sources[0]}
    return sum(CONTEXT_TOKENS // POOL_TOKENS_PER_ENTRY[pool] * entry_bytes for pool in pools) + \
        len(layers) * WINDOW_ROWS * row_bytes


def print_fit(title, units, byte_counts, table_bytes):
    stage_bytes = sum(count for unit, count in zip(units, byte_counts) if unit.kind == make_pack.UNIT_STAGE)
    expert_bytes = sum(byte_counts) - stage_bytes
    layer_units = [(unit, count) for unit, count in zip(units, byte_counts)
                   if unit.layer_index != make_pack.PACK_NO_LAYER]
    layer_count = len({unit.layer_index for unit, _ in layer_units})
    print(f"fit, {title}: wanted on the cards {gib(sum(byte_counts) + DRAFT_CACHE_BYTES):.2f} GiB (stage units "
          f"{gib(stage_bytes):.2f}, experts {gib(expert_bytes):.2f}, the draft cache {gib(DRAFT_CACHE_BYTES):.2f}); "
          f"the {layer_count} layers {gib(sum(count for _, count in layer_units)):.2f}; n-gram tables in host memory "
          f"{gib(table_bytes):.2f}")
    for card_names, card_title in CARD_SETS:
        budgets = [CARD_BYTES[name] - BACKEND_RESERVED_BYTES - KEPT_FREE_BYTES for name in card_names]
        placement = place(units, byte_counts, budgets)
        if placement.bytes_lacking:
            print(f"  {card_title}: the stage units alone do not fit: the last card lacks "
                  f"{gib(placement.bytes_lacking):.2f} GiB")
            continue
        host = [(unit, count) for unit, count, card in zip(units, byte_counts, placement.card_of_unit) if card is None]
        draft_host = sum(count for unit, count in host if unit.layer_index == make_pack.PACK_NO_LAYER)
        print(f"  {card_title} (budgets {', '.join(f'{gib(budget):.2f}' for budget in budgets)} GiB): every stage "
              f"unit keeps its first {placement.kept_per_owner} experts on its card; experts in host memory "
              f"{gib(sum(count for _, count in host)):.2f} GiB ({gib(draft_host):.2f} of the draft head's)")
        for card, (card_name, budget) in enumerate(zip(card_names, budgets)):
            held = [(unit, count) for unit, count, where in zip(units, byte_counts, placement.card_of_unit)
                    if where == card]
            stages = [unit for unit, _ in held if unit.kind == make_pack.UNIT_STAGE]
            if not stages:
                print(f"    {card_name:<6} nothing")
                continue
            layers = [unit.layer_index for unit in stages if unit.layer_index != make_pack.PACK_NO_LAYER]
            held_bytes = sum(count for _, count in held) + DRAFT_CACHE_BYTES * any(unit.label == DRAFT_CACHE_UNIT
                                                                                  for unit in stages)
            states = [state_bytes(layers, bits) for bits, _, _ in STATE_WIDTHS]
            print(f"    {card_name:<6} {stages[0].label} .. {stages[-1].label}: {gib(held_bytes):.2f} of "
                  f"{gib(budget):.2f} GiB; 1M-token state {gib(states[0]):.2f} GiB at {DEFAULT_STATE_BITS} bits "
                  f"({gib(states[1]):.2f} at 8) of the {gib(KEPT_FREE_BYTES):.0f} GiB kept free")


# ---- scales ----------------------------------------------------------------------------------------------------------
def rule_classes():
    """The class of every key (the checkpoint's scale byte << 16 | the bits of an F16 scale), uint8 [2^24]: the index
    in RULE_RATIOS of the ratio the scale is, exactly, of 2^(byte - 127); RULE_CLASS_OTHER for every other scale, and
    RULE_CLASS_NOT_A_NUMBER for the byte that stands for no number."""
    table = numpy.full(golden.BYTE_VALUES << FLOAT16_BITS, RULE_CLASS_OTHER, dtype = numpy.uint8)
    table[E8M0_NOT_A_NUMBER << FLOAT16_BITS:(E8M0_NOT_A_NUMBER + 1) << FLOAT16_BITS] = RULE_CLASS_NOT_A_NUMBER
    codes = numpy.arange(golden.BYTE_VALUES, dtype = numpy.uint32)
    codes = codes[codes != E8M0_NOT_A_NUMBER]
    with numpy.errstate(over = "ignore"):                  # a scale beyond the F16 range: no F16 is it
        for ratio_index, ratio in enumerate(RULE_RATIOS):
            scales = ratio * powers_of(codes)
            halves = scales.astype(numpy.float16)
            exact = halves.astype(numpy.float64) == scales
            table[(codes[exact] << FLOAT16_BITS) | halves.view(numpy.uint16)[exact]] = ratio_index
    return table


def direct_classes(bits, codes):
    """rule_classes' classes of groups computed from their values instead of the table: the check of the table.
    bits: the F16 scales as uint16; codes: the checkpoint's scale bytes of the same groups, in the same shape."""
    scales, powers = bits.view(numpy.float16).astype(numpy.float64), powers_of(codes)
    classes = numpy.full(bits.shape, RULE_CLASS_OTHER, dtype = numpy.uint8)
    for ratio_index, ratio in enumerate(RULE_RATIOS):
        classes[scales == ratio * powers] = ratio_index
    classes[codes == E8M0_NOT_A_NUMBER] = RULE_CLASS_NOT_A_NUMBER
    return classes


def scale_rule_of_unit(task):
    """Worker process: the matrices of one layer or draft block, (base, Matrix, its F16 scales, the checkpoint's
    scale bytes) each. Returns the groups of each class, the matrices with a group outside the rule, the ratios of
    the groups outside it whose byte is a number (the OTHER_RATIOS_KEPT most common, counted) with the count of the
    rest, and the first group outside the rule of the first such matrices."""
    table = rule_classes()
    counts = numpy.zeros(RULE_CLASS_COUNT, dtype = numpy.int64)
    ratios, matrices_outside, examples = collections.Counter(), 0, []
    with DirectFiles() as files:
        for index, (base, matrix, scales, exponents) in enumerate(task):
            groups = matrix.inputs // matrix.group_size
            bits = files.read(scales).view("<u2").reshape(groups, matrix.outputs)
            codes = numpy.ascontiguousarray(files.read(exponents).reshape(matrix.outputs, groups).T)    # as bits
            keys = codes.astype(numpy.uint32) << FLOAT16_BITS
            keys |= bits
            classes = table[keys]
            if index == 0 and not numpy.array_equal(classes, direct_classes(bits, codes)):
                raise ScanError(f"{base}: the table of the scale rule and the values of its scales disagree")
            counts += numpy.bincount(classes.ravel(), minlength = RULE_CLASS_COUNT)
            outside = classes >= RULE_CLASS_OTHER
            if not outside.any():
                continue
            matrices_outside += 1
            other = classes == RULE_CLASS_OTHER
            values, value_counts = numpy.unique(bits[other].view(numpy.float16).astype(numpy.float64) /
                                                powers_of(codes[other]), return_counts = True)
            ratios.update(dict(zip(values.tolist(), value_counts.tolist())))
            if len(examples) < GROUPS_OUTSIDE_SHOWN:
                group, output = divmod(int(outside.argmax()), matrix.outputs)
                code = int(codes[group, output])
                power = "no number" if code == E8M0_NOT_A_NUMBER else f"2^{code - E8M0_BIAS}"
                examples.append(f"{base} [output {output}, group {group}]: scale "
                                f"{float(bits[group, output].view(numpy.float16)):g}, the checkpoint's {power}")
    kept = dict(ratios.most_common(OTHER_RATIOS_KEPT))
    return counts, matrices_outside, kept, sum(ratios.values()) - sum(kept.values()), examples


def check_scale_rule(folder, reference_folder, jobs):
    """scales: the groups of every class by kind of expert, printed; returns the groups outside the rule."""
    started = time.monotonic()
    sources, reference = read_folder(folder), read_folder(reference_folder)
    pairs = expert_pairs(autoround_matrices(sources), reference)
    group_sizes = sorted({matrix.group_size for matrix, _, _ in pairs.values()})
    if group_sizes != [INPUTS_PER_SCALE]:
        raise ScanError(f"group sizes {group_sizes}: the rule holds a group of {INPUTS_PER_SCALE} weights against the "
                        "checkpoint's scale of the same weights")
    units = collections.defaultdict(list)                  # by layer or draft block, in the order the folder holds them
    for base, (matrix, _, exponents) in pairs.items():
        units[base.split(".ffn.", 1)[0]].append((base, matrix, sources[base + GROUP_SCALES_SUFFIX], exponents))
    tasks = sorted((sorted(entries, key = lambda entry: (entry[2].path, entry[2].byte_offset))
                    for entries in units.values()), key = lambda task: (task[0][2].path, task[0][2].byte_offset))
    task_byte_counts = [sum(entry[2].byte_count + entry[3].byte_count for entry in task) for task in tasks]
    results = make_pack.run_in_workers(scale_rule_of_unit, tasks, task_byte_counts, jobs, "reading the scales")

    by_kind = collections.defaultdict(lambda: [numpy.zeros(RULE_CLASS_COUNT, dtype = numpy.int64), 0, 0])
    ratios, ratios_not_kept, examples = collections.Counter(), 0, []
    for task, (counts, matrices_outside, kept, not_kept, task_examples) in zip(tasks, results):
        sums = by_kind[kind_of(task[0][0] + GROUP_SCALES_SUFFIX)]
        sums[0] += counts
        sums[1] += len(task)
        sums[2] += matrices_outside
        ratios.update(kept)
        ratios_not_kept += not_kept
        examples += task_examples
    print(f"scale rule: every group of every expert matrix, its scale over the checkpoint's power of two of the same "
          f"{INPUTS_PER_SCALE} weights")
    for kind, (counts, matrix_count, matrices_outside) in sorted(by_kind.items()):
        groups, holding = int(counts.sum()), int(counts[:len(RULE_RATIOS)].sum())
        print(f"  {kind}s: {matrix_count} matrices, {groups} groups: " +
              ", ".join(f"{ratio:g} {share(counts[index], groups):.2f}%" for index, ratio in enumerate(RULE_RATIOS)) +
              f"; the rule holds in {holding} ({share(holding, groups):.6f}%); outside it {groups - holding} groups "
              f"in {matrices_outside} matrices, {counts[RULE_CLASS_NOT_A_NUMBER]} of them with no number for a "
              "scale byte")
    all_counts = sum(counts for counts, _, _ in by_kind.values())
    groups, outside = int(all_counts.sum()), int(all_counts[RULE_CLASS_OTHER:].sum())
    print(f"  all: the rule holds in {groups - outside} of {groups} groups ({share(groups - outside, groups):.6f}%)")
    if ratios:
        print("  ratios outside the rule, the most common: " +
              "; ".join(f"{ratio:g} ({count})" for ratio, count in ratios.most_common(OTHER_RATIOS_SHOWN)) +
              (f"; {ratios_not_kept} groups in ratios a layer did not keep" if ratios_not_kept else ""))
    for example in examples[:GROUPS_OUTSIDE_SHOWN]:
        print(f"  OUTSIDE {example}")
    print(f"scales: {make_pack.rate_text(sum(task_byte_counts), time.monotonic() - started)}")
    return outside


# ---- scan ------------------------------------------------------------------------------------------------------------
def scan(folder, reference_folder, jobs):
    started = time.monotonic()
    sources, reference = read_folder(folder), read_folder(reference_folder)
    print_classes(sources)
    matrices, tables = autoround_matrices(sources), table_layouts(sources)
    print_quantization(folder, matrices, tables)
    pairs = expert_pairs(matrices, reference)
    print(f"expert matrices: the checkpoint's {len(pairs)} are the folder's, name for name and shape for shape")

    zero_counts, zero_byte_count = zero_words(sources, matrices)
    zero_count = sum(zero_counts.values())
    print(f"qzeros: {zero_count} words in {len(matrices)} tensors ({gib(zero_byte_count):.3f} GiB), "
          f"{len(zero_counts)} distinct: " + ", ".join(f"{word:#010x} ({count})" for word, count in
                                                         zero_counts.most_common(ZERO_WORDS_SHOWN)))
    print_expert_decode(sources, pairs, jobs)
    print_table_decode(sources, reference, tables)
    trunk_names = print_trunk(sources, reference, jobs)

    units, byte_counts, table_bytes = unit_bytes(sources)
    print_fit("the folder as published (bfloat16 trunk, qzeros kept)", units, byte_counts, table_bytes)
    zeros_are_one_word = len(zero_counts) == 1
    if trunk_names is None or not zeros_are_one_word:
        print("fit with the checkpoint's trunk and no qzeros: not computed, the scan did not find both lossless")
    else:
        engine_form = dict(sources)
        for name in trunk_names:
            scale_name = name[:-len(WEIGHT_SUFFIX)] + SCALE_SUFFIX
            engine_form[name], engine_form[scale_name] = reference[name], reference[scale_name]
        for base in matrices:
            del engine_form[base + ZEROS_SUFFIX]
        units, byte_counts, table_bytes = unit_bytes(engine_form)
        print_fit(f"the checkpoint's 8-bit trunk ({len(trunk_names)} matrices) and no qzeros (one zero point, "
                  f"{next(iter(zero_counts)):#010x})", units, byte_counts, table_bytes)
    print(f"scan: {time.monotonic() - started:.1f} s")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest = "command", required = True)
    files_arguments = commands.add_parser("files", help = "check every file against the repository's metadata")
    files_arguments.add_argument("folder")
    files_arguments.add_argument("--repository", required = True, help = "owner/name on the hub")
    files_arguments.add_argument("--revision", default = "main", help = "branch, tag or commit (default: main)")
    scan_arguments = commands.add_parser("scan", help = "classes, zero points, the trunk against the checkpoint, fit")
    scales_arguments = commands.add_parser("scales", help = "every expert group scale against the checkpoint's")
    for command_arguments in (scan_arguments, scales_arguments):
        command_arguments.add_argument("folder")
        command_arguments.add_argument("--reference", required = True, metavar = "FOLDER",
                                       help = "the original checkpoint")
    for command_arguments in (files_arguments, scan_arguments, scales_arguments):
        command_arguments.add_argument("--jobs", type = make_pack.job_count, default = DEFAULT_JOBS,
                                       help = f"files, tensors or layers read at once (default {DEFAULT_JOBS})")
    arguments = parser.parse_args()
    try:
        if arguments.command == "files":
            return EXIT_STATUS_FAILURE if check_files(arguments.folder, arguments.repository, arguments.revision,
                                                      arguments.jobs) else 0
        if arguments.command == "scales":
            return EXIT_STATUS_FAILURE if check_scale_rule(arguments.folder, arguments.reference, arguments.jobs) else 0
        scan(arguments.folder, arguments.reference, arguments.jobs)
    except (ScanError, make_pack.PackError, OSError, concurrent.futures.BrokenExecutor) as failure:
        print(f"w4a16_pack_scan {arguments.command} FAILED: {failure}", file = sys.stderr)
        return EXIT_STATUS_FAILURE
    return 0


if __name__ == "__main__":
    import w4a16_pack_scan                                 # the worker processes find their functions in it
    sys.exit(w4a16_pack_scan.main())
