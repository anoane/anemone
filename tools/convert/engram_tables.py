# SPDX-License-Identifier: AGPL-3.0-or-later
"""The n-gram tables of a model in an Anemone pack: what make_pack.py, beside this file, does about them.

It is not called on its own: make_pack.py's convert, verify and self-test call it, and tools/golden imports the
reference through it. The format is include/anemone/engram.h.

convert. A folder that stores n-gram tables (each as "<table name>.weight", F8_E4M3 [rows, 256], and
"<table name>.scale", F8_E8M0 [rows, 8]) gets them into its pack in host-only units behind every other unit, so
that no on-demand unit follows them: one unit of one extent for each table, in the order of the reference's
settings, and last one unit of one extent with the constants of the hashing.
  - A table's extent holds its rows: a row is the 256 value bytes of the checkpoint's row and then its 8 scale
    bytes, 15 rows lie at the start of every block, and every other byte is zero. Its one tensor is
    "<table name>.rows", UINT8, [blocks, 4096]. A table is larger than memory, so the worker processes write it in
    pieces of whole blocks: a worker reads the value bytes and the scale bytes of its rows from the source files,
    interleaves them, writes the blocks at their place and returns their share of the extent's digest, which is a
    sum over words counted from the start of the extent. No process holds more than one piece.
  - The constants are not derived here: the model's own reference code computes them. engram.py is imported from
    the reference folder (--reference <checkpoint folder>, in the layout of the published checkpoint:
    inference/engram.py, inference/config.json and the tokenizer's files; default: the source folder, if it holds
    them). Its build_compressed_token_map runs on the checkpoint's tokenizer, its EngramLayout.from_args on the
    engram settings of config.json, its compute_hash_multipliers on both, and the offsets are added up as its
    NgramHashState.__init__ adds them. The compressed vocabulary is compared with config.json as the reference
    compares it, and every value with the bounds of engram.h, before a byte is written. This needs Python 3.10 and
    torch, sympy, tokenizers and transformers, which are imported here and only when tables are converted.
  - --without-tables leaves the tables out and prints one line for each of their tensors. Tables without a usable
    reference folder, and tables stored in another form, stop the run: nothing is left out silently.

verify. After the rules of pack.h and before the digests, the rules of engram.h that a file alone can show: the
seven constants with their element types and shapes, in one extent of a host-only unit of one extent whose bytes
give its digest; the counts within the limits; no negative id, no product of an id and a multiplier beyond 64
bits, every multiplier odd and positive, no prime below 2, every range of rows inside its table; every table in a
host-only unit of its own, with one extent of exactly the blocks its rows need and the one tensor above; the zero
fill behind the rows of a sample of each table's blocks and of its last block. A pack without "engram.token_map"
has no tables. That the rows are the ones a token needs is the engine's to prove, against the file
tools/golden/make_engram_golden.py writes.

self-test. make_pack.py's self-test ends here: two tables of toy sizes are added to its folder and the constants
are given directly. The reference is NOT imported: no stand-in can prove the import of the real one, which the
first real conversion does. Shown: the refusal without a reference folder, --without-tables, the layout of the
tables against a second writing of it, the order of the units, the digest of a table written in pieces, the
constants byte for byte, what convert must refuse of a folder and of constants, and every rule of verify above
failing once on a changed copy, except the two that no small change of a pack reaches: a unit of several extents
and the size verify allows the constants (not shown).
"""
import collections
import contextlib
import importlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import types

import numpy

import make_pack                                           # used inside functions only: the two files import each other

# ---- The format: include/anemone/engram.h ----------------------------------------------------------------------------
VALUE_BYTES = 256
SCALE_BYTES = 8
ROW_BYTES = VALUE_BYTES + SCALE_BYTES
MAXIMUM_TABLES = 4
MAXIMUM_LOOKBACK = 8
MAXIMUM_ROWS_PER_TOKEN = 64
SMALLEST_PRIME = 2
INT32_MAXIMUM = 2 ** 31 - 1
INT64_MAXIMUM = 2 ** 63 - 1
ROWS_TENSOR_SUFFIX = ".rows"
ROWS_DTYPE = "U8"
TOKEN_MAP = "engram.token_map"
PAD_ID = "engram.pad_id"
MULTIPLIERS = "engram.multipliers"
PRIMES = "engram.primes"
OFFSETS = "engram.offsets"
ROW_COUNTS = "engram.row_counts"
TABLE_UNITS = "engram.table_units"
# The constants of the hashing, name: (safetensors dtype, number of dimensions), in the order engram.h lists them.
# This file holds them as a dictionary name: numpy int64 array, whatever the element type a pack gives them.
CONSTANT_FORMS = {TOKEN_MAP: ("I32", 1), PAD_ID: ("I32", 1), MULTIPLIERS: ("I64", 2), PRIMES: ("I64", 3),
                  OFFSETS: ("I64", 2), ROW_COUNTS: ("I64", 1), TABLE_UNITS: ("I32", 1)}
CONSTANT_NUMPY_TYPES = {"I32": "<i4", "I64": "<i8"}

# ---- A table as a checkpoint stores it, and the reference folder -----------------------------------------------------
TABLE_NAME = re.compile(r"layers\.(?P<layer>0|[1-9][0-9]{0,8})\.engram\.embed")
VALUES_SUFFIX = ".weight"
SCALES_SUFFIX = ".scale"
VALUES_DTYPE = "F8_E4M3"
SCALES_DTYPE = "F8_E8M0"
INFERENCE_FOLDER_NAME = "inference"
REFERENCE_MODULE_FILE_NAME = "engram.py"
REFERENCE_SETTINGS_FILE_NAME = "config.json"               # of the inference folder: the reference's ModelArgs as JSON
REFERENCE_TOKENIZER_FILE_NAME = "tokenizer.json"
REFERENCE_MODULE_NAME = "anemone_reference_engram"         # what the reference's engram.py is called in this process
REFERENCE_REQUIREMENTS = ("torch", "sympy", "tokenizers", "transformers")
REFERENCE_COUNT_SETTINGS = ("engram_max_ngram_size", "engram_vocab_size", "engram_n_heads", "engram_head_dim",
                            "engram_pad_id", "engram_compressed_vocab_size")
REFERENCE_LIST_SETTINGS = ("engram_layer_ids", "engram_num_embeddings")
REFERENCE_ADVICE = "give --reference <checkpoint folder>, or --without-tables to leave the tables out"
LEFT_OUT_LINE = "left out (an n-gram table):"

# ---- Writing and checking --------------------------------------------------------------------------------------------
TABLE_PIECE_BLOCKS = 2 ** 14                               # 64 MiB of a table: a worker holds that and its source rows
FILL_SAMPLE_BLOCKS = 2 ** 10                               # the blocks of each table whose zero fill verify reads
CONSTANTS_BYTE_LIMIT = 2 ** 30                             # verify holds the constants' extent in memory
# The hashing of a model's tables: the layer of each table, the constants (without the table units, which the layout
# of the pack gives), and where they come from, for messages.
Hashing = collections.namedtuple("Hashing", ("layer_ids", "constants", "origin"))
# What stands in a SourceTensor's path for a tensor that no file stores, and so first in a piece of
# make_pack.lay_out: the two stored tensors a table's rows are made from, or the bytes themselves.
TableRows = collections.namedtuple("TableRows", ("values", "scales", "row_count", "piece_block_count"))
DerivedBytes = collections.namedtuple("DerivedBytes", ("data",))

# ---- The self-test's tables: one on each layer of make_pack.py's toy model -------------------------------------------
SELF_TEST_LAYER_IDS = (0, 1)
# name: (shape, values). The primes are distinct, the offsets their running sums and the row counts their sums.
SELF_TEST_CONSTANTS = {
    TOKEN_MAP: ((10,), (0, 1, 2, 2, 3, 4, 4, 5, 6, 7)),
    PAD_ID: ((1,), (2,)),
    MULTIPLIERS: ((2, 3), ((3, 5, 7), (9, 11, 13))),
    PRIMES: ((2, 2, 2), (((5, 7), (11, 13)), ((17, 19), (23, 31)))),
    OFFSETS: ((2, 4), ((0, 5, 12, 23), (0, 17, 36, 59))),
    ROW_COUNTS: ((2,), (36, 90)),
}
SELF_TEST_ROWS_PER_BLOCK = 15                              # engram.h: 264 bytes to a row, 4,096 to a block
SELF_TEST_BLOCK_COUNTS = (3, 6)                            # a last block of 6 rows, and one that is full
SELF_TEST_PIECE_BLOCKS = 2                                 # the first table in pieces of 2 blocks and of 1
SELF_TEST_FILE_NAMES = ("model-00004-of-00005.safetensors", "model-00005-of-00005.safetensors")   # values, scales
SELF_TEST_STRAY_FILE_NAME = "model-00006-of-00006.safetensors"


def rows_per_block():
    """ANEMONE_ENGRAM_ROWS_PER_BLOCK: the rows at the start of every block."""
    return make_pack.PACK_BLOCK_BYTES // ROW_BYTES


def block_count_of(row_count):
    """The blocks the rows of a table fill."""
    return make_pack.round_up(row_count, rows_per_block()) // rows_per_block()


def check_constants(constants, where):
    """Every bound engram.h states for the constants of the hashing (of the table units, when they are among them,
    the shape only): the shapes against one another and the limits, then the values. The first that does not hold
    is a PackError that names it."""
    def broken(rule):
        return make_pack.PackError(f"{where}: {rule}")

    for name, array in constants.items():
        if array.ndim != CONSTANT_FORMS[name][1]:
            raise broken(f"{name} has {array.ndim} dimensions, not {CONSTANT_FORMS[name][1]}")
    table_count, lookback = constants[MULTIPLIERS].shape
    head_count = constants[PRIMES].shape[-1]
    rows_per_token = (lookback - 1) * head_count
    for count, limit, what in ((table_count, MAXIMUM_TABLES, "tables"),
                               (lookback, MAXIMUM_LOOKBACK, "tokens of lookback"),
                               (rows_per_token, MAXIMUM_ROWS_PER_TOKEN, "rows per token")):
        if not 1 <= count <= limit:
            raise broken(f"{count} {what}: engram.h allows 1 to {limit}")
    shapes = {TOKEN_MAP: constants[TOKEN_MAP].shape, PAD_ID: (1,), MULTIPLIERS: (table_count, lookback),
              PRIMES: (table_count, lookback - 1, head_count), OFFSETS: (table_count, rows_per_token),
              ROW_COUNTS: (table_count,), TABLE_UNITS: (table_count,)}
    for name, array in constants.items():
        if array.shape != shapes[name]:
            raise broken(f"{name} has the shape {list(array.shape)}, not {list(shapes[name])}")
    if not constants[TOKEN_MAP].size:
        raise broken(f"{TOKEN_MAP} is empty")

    token_map, pad_id, multipliers = constants[TOKEN_MAP], int(constants[PAD_ID][0]), constants[MULTIPLIERS]
    largest_id, largest_multiplier = max(int(token_map.max()), pad_id), int(multipliers.max())
    if int(token_map.min()) < 0 or pad_id < 0:
        raise broken("a token_map entry or pad_id is negative")
    if largest_id > INT32_MAXIMUM:
        raise broken(f"the id {largest_id} does not fit the INT32 of token_map and pad_id")
    if int(multipliers.min()) <= 0 or not (multipliers & 1).all():
        raise broken("a multiplier is even or not positive")
    if largest_id * largest_multiplier > INT64_MAXIMUM:
        raise broken(f"the id {largest_id} times the multiplier {largest_multiplier} does not fit signed 64 bits")
    if int(constants[PRIMES].min()) < SMALLEST_PRIME:
        raise broken(f"a prime is below {SMALLEST_PRIME}")
    primes_of_tables = constants[PRIMES].reshape(table_count, rows_per_token).tolist()
    offsets_of_tables, row_counts = constants[OFFSETS].tolist(), constants[ROW_COUNTS].tolist()
    for table_index, row_count in enumerate(row_counts):       # in Python's integers: no sum wraps
        for row_index, (prime, offset) in enumerate(zip(primes_of_tables[table_index], offsets_of_tables[table_index])):
            if offset < 0 or offset + prime > row_count:
                raise broken(f"table {table_index}: the range {row_index} (offset {offset}, prime {prime}) does not "
                             f"lie in its {row_count} rows")


# ---- convert: the reference ------------------------------------------------------------------------------------------
def failure_text(failure):
    """What stopped the reference's code or a module it needs, as ONE line: such messages often hold line breaks."""
    return " ".join(f"{type(failure).__name__}: {failure}".split())


def read_settings(settings_path):
    """The engram settings of the reference's config.json under the names its ModelArgs gives them (model.py:106-115),
    each checked for its kind: what engram.py reads of its args."""
    try:
        with open(settings_path, encoding = "utf-8") as handle:
            config = json.load(handle, object_pairs_hook = make_pack.object_without_repeats)
    except (ValueError, RecursionError) as failure:
        raise make_pack.PackError(f"{settings_path} cannot be read: {failure}") from failure
    settings = types.SimpleNamespace()
    for name in REFERENCE_COUNT_SETTINGS + REFERENCE_LIST_SETTINGS:
        value = config.get(name) if isinstance(config, dict) else None
        if not (make_pack.is_list_of_counts(value) and value if name in REFERENCE_LIST_SETTINGS
                else type(value) is int and value >= 0):
            raise make_pack.PackError(f"{settings_path}: {name} is missing, or is no "
                                      + ("list of counts" if name in REFERENCE_LIST_SETTINGS else "count"))
        setattr(settings, name, value)
    if len(settings.engram_layer_ids) != len(settings.engram_num_embeddings) \
            or len(set(settings.engram_layer_ids)) != len(settings.engram_layer_ids):
        raise make_pack.PackError(f"{settings_path}: engram_layer_ids {settings.engram_layer_ids} does not name one "
                                  f"layer for each of the {len(settings.engram_num_embeddings)} tables")
    if settings.engram_head_dim != VALUE_BYTES:
        raise make_pack.PackError(f"{settings_path}: engram_head_dim is {settings.engram_head_dim}, a row of this "
                                  f"format holds {VALUE_BYTES} values")
    return settings


def open_reference(reference_directory):
    """The reference's engram.py as a module, the engram settings of its config.json, and the checkpoint's tokenizer
    as the reference's generate.py loads it (generate.py:128). A folder that lacks one of the three, a Python that
    lacks a module they need, and whatever else stops the import are each a PackError of one line. Nothing is
    written into the folder."""
    inference_directory = os.path.join(reference_directory, INFERENCE_FOLDER_NAME)
    module_path = os.path.join(inference_directory, REFERENCE_MODULE_FILE_NAME)
    settings_path = os.path.join(inference_directory, REFERENCE_SETTINGS_FILE_NAME)
    for path in (module_path, settings_path, os.path.join(reference_directory, REFERENCE_TOKENIZER_FILE_NAME)):
        if not os.path.isfile(path):
            raise make_pack.PackError(f"{reference_directory} is no reference folder: there is no file {path}")
    settings = read_settings(settings_path)
    modules = {}
    for module_name in REFERENCE_REQUIREMENTS:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except Exception as failure:                           # whatever an installation that is broken raises
            raise make_pack.PackError(f"the reference's engram.py needs the Python module {module_name}, which this "
                                      f"Python cannot import: {failure_text(failure)}") from failure
    writes_no_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        specification = importlib.util.spec_from_file_location(REFERENCE_MODULE_NAME, module_path)
        reference = importlib.util.module_from_spec(specification)
        sys.modules[REFERENCE_MODULE_NAME] = reference         # a dataclass looks its module up while it is defined
        specification.loader.exec_module(reference)
        tokenizer = modules["transformers"].AutoTokenizer.from_pretrained(reference_directory, local_files_only = True)
    except Exception as failure:                               # the reference's code and its tokenizer: any failure
        raise make_pack.PackError(f"{reference_directory}: its engram.py or its tokenizer cannot be loaded: "
                                  f"{failure_text(failure)}") from failure
    finally:
        sys.dont_write_bytecode = writes_no_bytecode
    return reference, settings, tokenizer


def hashing_from_reference(reference_directory):
    """The hashing of a model's tables, computed by the reference's own code: the steps of engram.py's
    NgramHashState.__init__ (engram.py:145-154), each done by the function of engram.py that does it there."""
    try:
        reference, settings, tokenizer = open_reference(reference_directory)
    except make_pack.PackError as failure:
        raise make_pack.PackError(f"the folder stores n-gram tables, and their constants cannot be computed: "
                                  f"{failure}; {REFERENCE_ADVICE}") from failure
    try:
        token_map, compressed_vocabulary_size = reference.build_compressed_token_map(tokenizer)
        layout = reference.EngramLayout.from_args(settings)
        layer_ids, primes, row_counts = tuple(layout.layer_ids), layout.primes, layout.num_embeddings
        multipliers = reference.compute_hash_multipliers(layer_ids, layout.max_ngram_size,
                                                         compressed_vocabulary_size).tolist()
    except Exception as failure:                               # the reference's code: any failure is one line
        raise make_pack.PackError(f"{reference_directory}: its engram.py did not give the constants: "
                                  f"{failure_text(failure)}") from failure
    if compressed_vocabulary_size != settings.engram_compressed_vocab_size:        # engram.py:146
        raise make_pack.PackError(f"{reference_directory}: its tokenizer gives a compressed vocabulary of "
                                  f"{compressed_vocabulary_size}, its config.json says "
                                  f"{settings.engram_compressed_vocab_size}: every multiplier derives from it")
    if not isinstance(token_map, list) or settings.engram_pad_id >= len(token_map):
        raise make_pack.PackError(f"{reference_directory}: engram_pad_id {settings.engram_pad_id} is no token id")
    try:
        primes_of_tables = [[prime for primes_of_size in table for prime in primes_of_size] for table in primes]
        offsets = [numpy.cumsum([0, *primes_of_table[:-1]]).tolist() for primes_of_table in primes_of_tables]
        values = {TOKEN_MAP: token_map, PAD_ID: [token_map[settings.engram_pad_id]], MULTIPLIERS: multipliers,
                  PRIMES: primes, OFFSETS: offsets, ROW_COUNTS: row_counts}
        constants = {name: numpy.array(value, dtype = numpy.int64) for name, value in values.items()}
    except (OverflowError, TypeError, ValueError) as failure:
        raise make_pack.PackError(f"{reference_directory}: a constant its engram.py gave is no array of 64-bit "
                                  f"integers: {failure}") from failure
    return Hashing(layer_ids, constants, reference_directory)


# ---- convert: the units and their bytes ------------------------------------------------------------------------------
def stored_tables(sources, left_out_names):
    """The tables a folder stores, layer: (table name, values, scales, row count), from the names the model table
    left out. A table must be its two tensors in the one form this file converts."""
    parts_of_tables = collections.defaultdict(dict)
    for name in left_out_names:
        table_name, _, part = name.rpartition(".")
        if not TABLE_NAME.fullmatch(table_name):
            raise make_pack.PackError(f"the model table leaves the tensor {name!r} out, and it is no part of an n-gram "
                                      "table")
        parts_of_tables[table_name]["." + part] = sources[name]
    tables = {}
    for table_name, parts in parts_of_tables.items():
        values, scales = parts.get(VALUES_SUFFIX), parts.get(SCALES_SUFFIX)
        row_count = values.shape[0] if values is not None and values.shape else 0
        if set(parts) != {VALUES_SUFFIX, SCALES_SUFFIX} or not row_count \
                or (values.dtype, values.shape, values.byte_count) != (VALUES_DTYPE, (row_count, VALUE_BYTES),
                                                                       row_count * VALUE_BYTES) \
                or (scales.dtype, scales.shape, scales.byte_count) != (SCALES_DTYPE, (row_count, SCALE_BYTES),
                                                                       row_count * SCALE_BYTES):
            stored = ", ".join(f"{suffix} {part.dtype} {list(part.shape)} in {part.byte_count} bytes"
                               for suffix, part in sorted(parts.items()))
            raise make_pack.PackError(
                f"the n-gram table {table_name} is stored as {stored}: this converter takes {VALUES_SUFFIX} "
                f"{VALUES_DTYPE} [rows, {VALUE_BYTES}] with {SCALES_SUFFIX} {SCALES_DTYPE} [rows, {SCALE_BYTES}]; "
                "--without-tables leaves the tables out")
        tables[int(TABLE_NAME.fullmatch(table_name)["layer"])] = (table_name, values, scales, row_count)
    return tables


def with_tables(units, sources, left_out_names, reference_directory, without_tables, hashing, piece_block_count):
    """convert's step between the model table and the layout. Returns the units and the sources with the tables'
    units and the constants' unit behind them, and the line of convert's summary about the tables. With
    without_tables, and for a folder that stores no table, nothing is added and every left-out tensor is printed.
    hashing and piece_block_count are the self-test's: the hashing given in place of a reference folder, and tables
    cut into pieces of fewer blocks."""
    if without_tables or not left_out_names:
        for name in left_out_names:
            print(f"{LEFT_OUT_LINE} {name}, {sources[name].byte_count} bytes")
        byte_count = sum(sources[name].byte_count for name in left_out_names)
        return units, sources, (f"  left out: {len(left_out_names)} tensors, {byte_count} bytes "
                                f"({byte_count / make_pack.BYTES_PER_GIB:.2f} GiB)")
    tables = stored_tables(sources, left_out_names)
    if hashing is None:
        hashing = hashing_from_reference(reference_directory)
    constants = dict(hashing.constants)
    check_constants(constants, f"{hashing.origin}: the constants of the n-gram tables")
    row_counts = constants[ROW_COUNTS].tolist()
    if sorted(tables) != sorted(hashing.layer_ids) or len(hashing.layer_ids) != len(row_counts):
        raise make_pack.PackError(f"the folder stores n-gram tables of the layers {sorted(tables)}, {hashing.origin} "
                                  f"gives {len(row_counts)} tables on the layers {list(hashing.layer_ids)}")

    units, sources, block_bytes = list(units), dict(sources), make_pack.PACK_BLOCK_BYTES
    constants[TABLE_UNITS] = numpy.arange(len(units), len(units) + len(row_counts), dtype = numpy.int64)
    for layer_id, row_count in zip(hashing.layer_ids, row_counts):
        table_name, values, scales, stored_row_count = tables[layer_id]
        rows_name, block_count = table_name + ROWS_TENSOR_SUFFIX, block_count_of(row_count)
        if stored_row_count != row_count:
            raise make_pack.PackError(f"{table_name} stores {stored_row_count} rows, {hashing.origin} gives the table "
                                      f"of layer {layer_id} {row_count} rows")
        if rows_name in sources:
            raise make_pack.PackError(f"the folder stores a tensor {rows_name!r}: the name of the rows of a table")
        rows = TableRows(values, scales, row_count, piece_block_count or TABLE_PIECE_BLOCKS)
        sources[rows_name] = make_pack.SourceTensor(rows, 0, block_count * block_bytes, ROWS_DTYPE,
                                                    (block_count, block_bytes))
        units.append(make_pack.Unit(make_pack.UNIT_HOST_ONLY, layer_id, f"n-gram table of layer {layer_id}",
                                    [rows_name]))
    for name, (dtype, _) in CONSTANT_FORMS.items():
        data = constants[name].astype(CONSTANT_NUMPY_TYPES[dtype]).tobytes()
        sources[name] = make_pack.SourceTensor(DerivedBytes(data), 0, len(data), dtype, constants[name].shape)
    units.append(make_pack.Unit(make_pack.UNIT_HOST_ONLY, make_pack.PACK_NO_LAYER, "n-gram constants",
                                list(CONSTANT_FORMS)))
    byte_count = sum(block_count_of(row_count) for row_count in row_counts) * block_bytes
    return units, sources, (
        f"  n-gram tables: {len(row_counts)} on the layers {list(hashing.layer_ids)} with {row_counts} rows, "
        f"{byte_count} bytes ({byte_count / make_pack.BYTES_PER_GIB:.2f} GiB); {constants[TOKEN_MAP].size} token ids, "
        f"{constants[MULTIPLIERS].shape[1]} tokens of lookback, {constants[PRIMES].shape[-1]} heads, from "
        f"{hashing.origin}")


def digest_and_write(pack_path, byte_offset, data, first_word_index):
    """Writes bytes of an extent at their place in the pack and returns their share of the extent's digest."""
    digest = make_pack.digest_of(data, first_word_index)
    descriptor = os.open(pack_path, os.O_WRONLY)
    try:
        make_pack.write_exactly(descriptor, byte_offset, data, pack_path)
    finally:
        os.close(descriptor)
    return digest


def read_rows(source, first_row, row_count, row_bytes):
    """Rows of a stored tensor, (its file, where it begins there), as an array [rows, bytes of one row]."""
    source_path, source_byte_offset = source
    data = memoryview(bytearray(row_count * row_bytes))
    with open(source_path, "rb", buffering = 0) as handle:
        make_pack.read_exactly(handle, source_byte_offset + first_row * row_bytes, data, source_path)
    return numpy.frombuffer(data, dtype = numpy.uint8).reshape(row_count, row_bytes)


def write_table_piece(task):
    """Worker process: a run of whole blocks of one table, made from the rows of the checkpoint (row r lies in block
    r / rows per block at byte (r % rows per block) * ROW_BYTES, its values and then its scales; every other byte is
    zero) and written at its place. Returns the piece's share of the digest of the table's extent."""
    pack_path, extent_byte_offset, values, scales, table_row_count, first_block, block_count = task
    first_row = first_block * rows_per_block()
    row_count = min(block_count * rows_per_block(), table_row_count - first_row)
    value_rows = read_rows(values, first_row, row_count, VALUE_BYTES)
    scale_rows = read_rows(scales, first_row, row_count, SCALE_BYTES)
    piece_bytes = bytearray(block_count * make_pack.PACK_BLOCK_BYTES)
    blocks = numpy.frombuffer(piece_bytes, dtype = numpy.uint8).reshape(block_count, make_pack.PACK_BLOCK_BYTES)
    for row_in_block in range(rows_per_block()):               # the rows that lie at one place of their blocks
        first_byte, values_of_place = row_in_block * ROW_BYTES, value_rows[row_in_block::rows_per_block()]
        blocks[:len(values_of_place), first_byte:first_byte + VALUE_BYTES] = values_of_place
        blocks[:len(values_of_place), first_byte + VALUE_BYTES:first_byte + ROW_BYTES] = \
            scale_rows[row_in_block::rows_per_block()]
    piece_byte_offset = first_block * make_pack.PACK_BLOCK_BYTES
    return digest_and_write(pack_path, extent_byte_offset + piece_byte_offset, piece_bytes,
                            piece_byte_offset // make_pack.DIGEST_WORD_BYTES)


def write_derived_extent(task):
    """Worker process: one extent whose tensors come from memory, (bytes, byte offset in the extent) each, assembled
    with zero fill, digested and written. Returns the digest."""
    pack_path, extent_byte_offset, extent_byte_count, tensors = task
    extent_bytes = bytearray(extent_byte_count)
    for data, byte_offset_in_extent in tensors:
        extent_bytes[byte_offset_in_extent:byte_offset_in_extent + len(data)] = data
    return digest_and_write(pack_path, extent_byte_offset, extent_bytes, 0)


def run_task(task):
    """Worker process: one task of write_extents, a worker function and its task."""
    function, task_of_function = task
    return function(task_of_function)


def write_extents(pack_path, extents, jobs):
    """What convert hands the worker processes once a pack may hold tables: an extent of stored tensors whole
    (make_pack.write_extent), a table in pieces of whole blocks, the constants from memory. Returns the digest of
    every extent; a table's is the sum, modulo 2^64, of the shares of its pieces."""
    tasks, task_byte_counts, extent_indices, block_bytes = [], [], [], make_pack.PACK_BLOCK_BYTES
    for extent_index, extent in enumerate(extents):
        derived, tasks_of_extent = extent.pieces[0][0], []     # an extent holds bytes, so it has a piece
        if isinstance(derived, TableRows):
            values, scales = ((source.path, source.byte_offset) for source in (derived.values, derived.scales))
            for first_block in range(0, extent.byte_count // block_bytes, derived.piece_block_count):
                block_count = min(derived.piece_block_count, extent.byte_count // block_bytes - first_block)
                tasks_of_extent.append(((write_table_piece, (pack_path, extent.byte_offset, values, scales,
                                                             derived.row_count, first_block, block_count)),
                                        block_count * block_bytes))
        elif isinstance(derived, DerivedBytes):
            tensors = [(source.data, byte_offset_in_extent) for source, _, _, byte_offset_in_extent in extent.pieces]
            tasks_of_extent.append(((write_derived_extent, (pack_path, extent.byte_offset, extent.byte_count, tensors)),
                                    extent.byte_count))
        else:
            tasks_of_extent.append(((make_pack.write_extent, (pack_path, extent.byte_offset, extent.byte_count,
                                                              extent.pieces)), extent.byte_count))
        for task, task_byte_count in tasks_of_extent:
            tasks.append(task)
            task_byte_counts.append(task_byte_count)
            extent_indices.append(extent_index)
    digests = [0] * len(extents)
    shares = make_pack.run_in_workers(run_task, tasks, task_byte_counts, jobs, "written")
    for extent_index, share in zip(extent_indices, shares):
        digests[extent_index] = (digests[extent_index] + share) % make_pack.UINT64_MODULUS
    return digests


# ---- verify ----------------------------------------------------------------------------------------------------------
def check_pack(pack_path, extents, units, tensors):
    """verify's rules for the n-gram tables: what engram.h demands of a pack as far as a file alone can show it, on
    what make_pack.read_pack_tables returns and with none of convert's layout. Returns what verify prints about the
    tables. The first rule that does not hold is a PackError that names it."""
    def broken(rule):
        return make_pack.PackError(f"{pack_path}: n-gram tables: {rule}")

    def only_extent_of(unit_index, what):
        if units[unit_index]["unit_kind"] != make_pack.UNIT_HOST_ONLY:
            raise broken(f"{what}: unit {unit_index} is not host-only")
        if len(extent_indices_of_units[unit_index]) != 1:
            raise broken(f"{what}: unit {unit_index} has {len(extent_indices_of_units[unit_index])} extents, not one")
        return extent_indices_of_units[unit_index][0]

    tensors_by_name = {tensor["name"]: tensor for tensor in tensors}
    if TOKEN_MAP.encode("ascii") not in tensors_by_name:
        return "no n-gram tables"
    extent_indices_of_units = collections.defaultdict(list)
    for extent_index, extent in enumerate(extents):
        extent_indices_of_units[extent["unit_index"]].append(extent_index)

    # every constant is there, with its element type and its number of dimensions, and holds an element
    constant_tensors = {}
    for name, (dtype, dimension_count) in CONSTANT_FORMS.items():
        tensor = tensors_by_name.get(name.encode("ascii"))
        if tensor is None:
            raise broken(f"the constant {name} is missing")
        if (tensor["element_type"], tensor["dimension_count"]) != (make_pack.ELEMENT_TYPES[dtype][0], dimension_count):
            raise broken(f"the constant {name} is not {dtype} with {dimension_count} dimensions")
        if not tensor["byte_count"]:
            raise broken(f"the constant {name} is empty")
        constant_tensors[name] = tensor
    # they all lie in one extent of a host-only unit of one extent, and the bytes of that extent give its digest
    constants_extent_indices = {tensor["extent_index"] for tensor in constant_tensors.values()}
    if len(constants_extent_indices) != 1:
        raise broken(f"the constants do not all lie in one extent: they lie in {sorted(constants_extent_indices)}")
    (constants_extent_index,) = constants_extent_indices
    constants_extent = extents[constants_extent_index]
    only_extent_of(constants_extent["unit_index"], "the constants")
    if constants_extent["byte_count"] > CONSTANTS_BYTE_LIMIT:
        raise broken(f"extent {constants_extent_index}, which holds the constants, has "
                     f"{constants_extent['byte_count']} bytes: this tool reads at most {CONSTANTS_BYTE_LIMIT}")
    extent_bytes, block_bytes = bytearray(constants_extent["byte_count"]), make_pack.PACK_BLOCK_BYTES
    rows_of_a_block = block_bytes // ROW_BYTES
    with open(pack_path, "rb", buffering = 0) as handle:
        make_pack.read_exactly(handle, constants_extent["byte_offset"], memoryview(extent_bytes), pack_path)
        if make_pack.digest_of(extent_bytes) != constants_extent["digest"]:
            raise broken(f"the bytes of extent {constants_extent_index}, which holds the constants, do not give its "
                         "digest")
        constants = {}
        for name, tensor in constant_tensors.items():
            dimensions = tensor["dimensions"][:tensor["dimension_count"]]
            constants[name] = numpy.frombuffer(
                extent_bytes, dtype = CONSTANT_NUMPY_TYPES[CONSTANT_FORMS[name][0]], count = math.prod(dimensions),
                offset = tensor["byte_offset_in_extent"]).reshape(dimensions).astype(numpy.int64)
        # the counts, the shapes and the values within their bounds
        check_constants(constants, f"{pack_path}: n-gram tables")

        # every table: its own host-only unit of one extent, which is exactly the blocks its rows need, holds the one
        # tensor of engram.h and is zero behind the rows of each block (of a sample of them, and of the last)
        row_counts, table_units = constants[ROW_COUNTS].tolist(), constants[TABLE_UNITS].tolist()
        if len(set(table_units)) != len(table_units):
            raise broken(f"two tables name one unit: {TABLE_UNITS} is {table_units}")
        block = memoryview(bytearray(block_bytes))
        for table_index, (unit_index, row_count) in enumerate(zip(table_units, row_counts)):
            what = f"table {table_index}"
            if not 0 <= unit_index < len(units):
                raise broken(f"{what}: there is no unit {unit_index}")
            extent_index = only_extent_of(unit_index, what)
            block_count = (row_count + rows_of_a_block - 1) // rows_of_a_block
            if extents[extent_index]["byte_count"] != block_count * block_bytes:
                raise broken(f"{what}: extent {extent_index} holds {extents[extent_index]['byte_count']} bytes, its "
                             f"{row_count} rows need {block_count} blocks")
            tensors_of_extent = [tensor for tensor in tensors if tensor["extent_index"] == extent_index]
            rows_tensor = tensors_of_extent[0] if len(tensors_of_extent) == 1 else None
            if rows_tensor is None or rows_tensor["byte_offset_in_extent"] \
                    or len(rows_tensor["name"]) <= len(ROWS_TENSOR_SUFFIX) \
                    or not rows_tensor["name"].endswith(ROWS_TENSOR_SUFFIX.encode("ascii")) \
                    or rows_tensor["element_type"] != make_pack.ELEMENT_TYPES[ROWS_DTYPE][0] \
                    or rows_tensor["dimensions"][:rows_tensor["dimension_count"]] != (block_count, block_bytes):
                raise broken(f"{what}: the one tensor of extent {extent_index} is not \"<table name>"
                             f"{ROWS_TENSOR_SUFFIX}\", UINT8, [{block_count}, {block_bytes}], at byte 0")
            sample_count = min(FILL_SAMPLE_BLOCKS, block_count)
            sample = {sample_index * block_count // sample_count for sample_index in range(sample_count)}
            for block_index in sorted(sample | {block_count - 1}):
                make_pack.read_exactly(handle, extents[extent_index]["byte_offset"] + block_index * block_bytes, block,
                                       pack_path)
                rows_of_block = min(rows_of_a_block, row_count - block_index * rows_of_a_block)
                if any(block[rows_of_block * ROW_BYTES:]):
                    raise broken(f"{what}: block {block_index} of its extent is not zero behind its {rows_of_block} "
                                 "rows")
    return f"{len(row_counts)} n-gram tables of {row_counts} rows by the rules of engram.h that a file can show"


# ---- self-test -------------------------------------------------------------------------------------------------------
def flattened(values):
    """The numbers of nested tuples in the order a pack stores them: the last index runs fastest."""
    return [number for value in values for number in flattened(value)] if isinstance(values, tuple) else [values]


def expected_table_extent(value_bytes, scale_bytes, row_count):
    """A table's extent laid out a second time, in plain Python and from the sentence of engram.h: row r lies in
    block r / ROWS_PER_BLOCK at byte (r % ROWS_PER_BLOCK) * ROW_BYTES; the rest is zero."""
    block_bytes = make_pack.PACK_BLOCK_BYTES
    rows_of_a_block = block_bytes // (VALUE_BYTES + SCALE_BYTES)
    extent = bytearray(-(-row_count // rows_of_a_block) * block_bytes)
    for row in range(row_count):
        first_byte = row // rows_of_a_block * block_bytes + row % rows_of_a_block * (VALUE_BYTES + SCALE_BYTES)
        extent[first_byte:first_byte + VALUE_BYTES + SCALE_BYTES] = (
            value_bytes[row * VALUE_BYTES:(row + 1) * VALUE_BYTES]
            + scale_bytes[row * SCALE_BYTES:(row + 1) * SCALE_BYTES])
    return bytes(extent)


def self_test(folder, source_folder, plain_pack_path, expect, expect_refusal):
    """The tables' part of make_pack.py's self-test, on its toy model: source_folder stores no table yet and
    plain_pack_path is its pack. expect and expect_refusal are make_pack.self_test's. What is shown: the text at the
    top of this file."""
    jobs, block_bytes = make_pack.SELF_TEST_JOBS, make_pack.PACK_BLOCK_BYTES
    pack_path, broken_path = os.path.join(folder, "tables.pack"), os.path.join(folder, "broken_tables.pack")
    row_counts = SELF_TEST_CONSTANTS[ROW_COUNTS][1]
    table_names = [f"layers.{layer_id}.engram.embed" for layer_id in SELF_TEST_LAYER_IDS]
    stored = {}                                                # the tables as a checkpoint stores them, name: bytes
    for file_name, suffix, dtype, row_bytes in zip(SELF_TEST_FILE_NAMES, (VALUES_SUFFIX, SCALES_SUFFIX),
                                                   (VALUES_DTYPE, SCALES_DTYPE), (VALUE_BYTES, SCALE_BYTES)):
        stored.update(make_pack.write_safetensors_file(
            os.path.join(source_folder, file_name),
            [(table_name + suffix, dtype, (row_count, row_bytes)) for table_name, row_count in zip(table_names,
                                                                                                   row_counts)]))
    constants = {name: numpy.array(values, dtype = numpy.int64) for name, (_, values) in SELF_TEST_CONSTANTS.items()}
    hashing = Hashing(SELF_TEST_LAYER_IDS, constants, "the self-test")

    def convert(**table_arguments):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            make_pack.convert(source_folder, pack_path, jobs, True, **table_arguments)
        return printed.getvalue()

    def read(path):
        with open(path, "rb") as handle:
            return handle.read()

    def bytes_of(pack, extent):
        return pack[extent["byte_offset"]:extent["byte_offset"] + extent["byte_count"]]

    # tables and no reference folder; --without-tables
    expect(rows_per_block() == SELF_TEST_ROWS_PER_BLOCK, "a block holds the rows engram.h says")
    for reference_directory in (None, folder):
        expect_refusal(lambda: make_pack.convert(source_folder, pack_path, jobs, False, reference_directory),
                       REFERENCE_ADVICE, f"tables, and for a reference folder {reference_directory}, which is none")
    expect(not os.path.lexists(pack_path) and not os.path.lexists(pack_path + make_pack.TEMPORARY_SUFFIX),
           "a conversion refused for its tables leaves no file")
    printed, plain_bytes = convert(without_tables = True), read(plain_pack_path)
    expect(all(f"{LEFT_OUT_LINE} {name}, " in printed for name in stored), "every tensor left out is printed")
    expect(read(pack_path) == plain_bytes, "--without-tables writes the pack of the folder without its tables")

    # the tables in the pack: the units, the rows, the digests of the pieces, the constants
    convert(table_hashing = hashing, table_piece_block_count = SELF_TEST_PIECE_BLOCKS)
    with contextlib.redirect_stdout(io.StringIO()):
        _, extents, units, tensors = make_pack.verify(pack_path, jobs)
    pack_bytes, (_, plain_extents, plain_units, _) = read(pack_path), make_pack.read_pack_tables(plain_pack_path)
    packed = {tensor["name"].decode("utf-8"): tensor for tensor in tensors}
    table_units = list(range(len(plain_units), len(plain_units) + len(table_names)))
    expect(units == plain_units + [{"unit_kind": make_pack.UNIT_HOST_ONLY, "layer_index": layer_index}
                                   for layer_index in (*SELF_TEST_LAYER_IDS, make_pack.PACK_NO_LAYER)],
           "behind the units of the model: one host-only unit for each table, then the constants', and no other")
    expect(len(extents) == len(units), "every unit is one extent, a table's too")
    expect(all(bytes_of(pack_bytes, extent) == bytes_of(plain_bytes, plain_extent)
               for extent, plain_extent in zip(extents, plain_extents)),
           "the extents of the model are those of the pack without tables")
    for table_name, row_count, block_count, unit_index in zip(table_names, row_counts, SELF_TEST_BLOCK_COUNTS,
                                                              table_units):
        expected = expected_table_extent(stored[table_name + VALUES_SUFFIX], stored[table_name + SCALES_SUFFIX],
                                         row_count)
        extent, tensor = extents[unit_index], packed.get(table_name + ROWS_TENSOR_SUFFIX)
        expect(len(expected) == block_count * block_bytes and extent["unit_index"] == unit_index
               and bytes_of(pack_bytes, extent) == expected,
               f"{table_name}: its extent is its rows, {SELF_TEST_ROWS_PER_BLOCK} to a block, values then scales, and "
               "zero elsewhere")
        expect(extent["digest"] == make_pack.digest_of(expected),
               f"{table_name}: the digests of its pieces add up to the digest of its extent")
        expect(tensor is not None and (tensor["extent_index"], tensor["byte_offset_in_extent"], tensor["element_type"])
               == (unit_index, 0, make_pack.ELEMENT_TYPES[ROWS_DTYPE][0])
               and tensor["dimensions"][:tensor["dimension_count"]] == (block_count, block_bytes),
               f"{table_name}{ROWS_TENSOR_SUFFIX}: its extent, its element type and its shape")
    expected_values = {**{name: flattened(values) for name, (_, values) in SELF_TEST_CONSTANTS.items()},
                       TABLE_UNITS: table_units}
    expected_shapes = {**{name: shape for name, (shape, _) in SELF_TEST_CONSTANTS.items()},
                       TABLE_UNITS: (len(table_units),)}
    expected = b""
    for name in sorted(CONSTANT_FORMS, key = str.encode):
        element_type, element_bytes = make_pack.ELEMENT_TYPES[CONSTANT_FORMS[name][0]]
        expected += bytes(-len(expected) % make_pack.MATRIX_ALIGNMENT_BYTES) + b"".join(
            value.to_bytes(element_bytes, "little", signed = True) for value in expected_values[name])
        tensor = packed.get(name)
        expect(tensor is not None and (tensor["extent_index"], tensor["element_type"])
               == (len(extents) - 1, element_type)
               and tensor["dimensions"][:tensor["dimension_count"]] == expected_shapes[name],
               f"{name}: its extent, its element type and its shape")
    expect(bytes_of(pack_bytes, extents[-1]) == expected + bytes(-len(expected) % block_bytes),
           "the constants: in the order of their names, little-endian, aligned and zero-filled")
    convert(table_hashing = hashing)
    expect(read(pack_path) == pack_bytes, "tables written whole and tables written in pieces are one pack")

    # what convert refuses: a table stored in another form, and constants that do not fit the folder or engram.h
    stray_path = os.path.join(source_folder, SELF_TEST_STRAY_FILE_NAME)
    make_pack.write_safetensors_file(stray_path, [(table_names[0] + VALUES_SUFFIX, ROWS_DTYPE,
                                                   (row_counts[0], VALUE_BYTES))])
    expect_refusal(lambda: convert(table_hashing = hashing), "this converter takes", "a table stored in another form")
    os.remove(stray_path)
    _, lookback = constants[MULTIPLIERS].shape
    for refused_layer_ids, changed_name, changed_array, rule_text, what in (
            (SELF_TEST_LAYER_IDS, ROW_COUNTS, constants[ROW_COUNTS] + rows_per_block(),
             f"stores {row_counts[0]} rows", "constants for tables of other row counts"),
            (tuple(layer_id + 1 for layer_id in SELF_TEST_LAYER_IDS), ROW_COUNTS, constants[ROW_COUNTS],
             "n-gram tables of the layers", "constants for the tables of other layers"),
            (SELF_TEST_LAYER_IDS, TOKEN_MAP, constants[TOKEN_MAP] + INT32_MAXIMUM, "does not fit the INT32",
             "constants with an id that 32 bits do not hold"),
            (SELF_TEST_LAYER_IDS, MULTIPLIERS, numpy.ones((MAXIMUM_TABLES + 1, lookback), dtype = numpy.int64),
             "tables: engram.h allows", "constants for more tables than engram.h allows")):
        refused_hashing = Hashing(refused_layer_ids, {**constants, changed_name: changed_array}, "the self-test")
        expect_refusal(lambda: convert(table_hashing = refused_hashing), rule_text, what)
    expect(read(pack_path) == pack_bytes, "a conversion that is refused leaves the pack that was there")

    # what verify refuses: every rule shown failing once, on a copy of the pack with a few bytes changed
    unit_records = block_bytes + len(extents) * make_pack.EXTENT_RECORD_BYTES
    tensor_records = unit_records + len(units) * make_pack.UNIT_RECORD_BYTES
    name_bytes = tensor_records + len(tensors) * make_pack.TENSOR_RECORD_BYTES

    def number(value, byte_count):
        return value.to_bytes(byte_count, "little", signed = value < 0)

    def in_tensor_record(name, field, value):
        byte_offset, byte_count = make_pack.TENSOR_FIELDS[field]
        return (tensor_records + list(packed).index(name) * make_pack.TENSOR_RECORD_BYTES + byte_offset,
                number(value, byte_count))

    def kind_of_unit(unit_index, unit_kind):
        byte_offset, byte_count = make_pack.UNIT_FIELDS["unit_kind"]
        return unit_records + unit_index * make_pack.UNIT_RECORD_BYTES + byte_offset, number(unit_kind, byte_count)

    def in_constant(name, element_index, value):
        tensor, element_bytes = packed[name], make_pack.ELEMENT_TYPES[CONSTANT_FORMS[name][0]][1]
        return (extents[tensor["extent_index"]]["byte_offset"] + tensor["byte_offset_in_extent"]
                + element_index * element_bytes, number(value, element_bytes))

    def behind_rows(table_index, block_index, rows_of_block):
        return (extents[table_units[table_index]]["byte_offset"] + block_index * block_bytes
                + rows_of_block * ROW_BYTES, number(1, 1))

    def refused_by_verify(what, rule_text, changes, digests_made_to_fit = True):
        """verify on a copy of the pack with bytes changed, (byte offset in the file, bytes) each. With the digests
        made to fit (of every extent a change lies in, of the tables, of the header) a change reaches its rule."""
        shutil.copyfile(pack_path, broken_path)
        with open(broken_path, "r+b") as handle:
            for byte_offset, replacement in changes:
                handle.seek(byte_offset)
                handle.write(replacement)
            for extent_index, extent in enumerate(extents if digests_made_to_fit else ()):
                if any(0 <= byte_offset - extent["byte_offset"] < extent["byte_count"] for byte_offset, _ in changes):
                    handle.seek(extent["byte_offset"])
                    digest = make_pack.digest_of(handle.read(extent["byte_count"]))
                    digest_byte_offset, digest_byte_count = make_pack.EXTENT_FIELDS["digest"]
                    handle.seek(block_bytes + extent_index * make_pack.EXTENT_RECORD_BYTES + digest_byte_offset)
                    handle.write(number(digest, digest_byte_count))
        if digests_made_to_fit:
            make_pack.reseal(broken_path)
        expect_refusal(lambda: make_pack.verify(broken_path, jobs), rule_text, what)

    last_block, int64_bytes = SELF_TEST_BLOCK_COUNTS[0] - 1, make_pack.ELEMENT_TYPES["I64"][1]
    for what, rule_text, changes in (
            ("a constant under another name", f"the constant {PAD_ID} is missing",
             [(name_bytes + packed[PAD_ID]["name_byte_offset"] + len(PAD_ID) - 1, b"e")]),
            ("a constant of another element type", f"the constant {PAD_ID} is not",
             [in_tensor_record(PAD_ID, "element_type", make_pack.ELEMENT_TYPES["F32"][0])]),
            ("a constant without an element", f"the constant {PAD_ID} is empty",
             [in_tensor_record(PAD_ID, "byte_count", 0), in_tensor_record(PAD_ID, "dimensions", 0)]),
            ("a constant in another extent", "the constants do not all lie in one extent",
             [in_tensor_record(ROW_COUNTS, "extent_index", 0)]),
            ("the constants in a stage unit", f"the constants: unit {len(units) - 1} is not host-only",
             [kind_of_unit(len(units) - 1, make_pack.UNIT_STAGE)]),
            ("a constant of another shape", f"{ROW_COUNTS} has the shape",
             [in_tensor_record(ROW_COUNTS, "byte_count", int64_bytes), in_tensor_record(ROW_COUNTS, "dimensions", 1)]),
            # the record of the multipliers stretched over the zero fill and the tensor behind it: [limit + 1, lookback]
            ("more tables than engram.h allows", "tables: engram.h allows",
             [in_tensor_record(MULTIPLIERS, "byte_count", (MAXIMUM_TABLES + 1) * lookback * int64_bytes),
              in_tensor_record(MULTIPLIERS, "dimensions",
                               MAXIMUM_TABLES + 1 + lookback * (make_pack.UINT32_MAXIMUM + 1))]),
            ("a negative id", "is negative", [in_constant(TOKEN_MAP, 1, -1)]),
            ("a multiplier that is not positive", "a multiplier is even or not positive",
             [in_constant(MULTIPLIERS, 1, 0)]),
            ("a multiplier too large for the ids", "does not fit signed 64 bits",
             [in_constant(MULTIPLIERS, 0, INT64_MAXIMUM)]),
            ("a prime below the smallest", "a prime is below", [in_constant(PRIMES, 0, 1)]),
            ("a range of rows behind the last row", "does not lie in its", [in_constant(OFFSETS, 1, row_counts[0])]),
            ("two tables in one unit", "two tables name one unit", [in_constant(TABLE_UNITS, 1, table_units[0])]),
            ("a table in a unit that is not there", f"table 0: there is no unit {len(units)}",
             [in_constant(TABLE_UNITS, 0, len(units))]),
            ("a table in a stage unit", f"table 0: unit {table_units[0]} is not host-only",
             [kind_of_unit(table_units[0], make_pack.UNIT_STAGE)]),
            ("a table of more rows than its extent holds", "rows need",
             [in_constant(ROW_COUNTS, 1, row_counts[1] + 1)]),
            ("the tensor of a table with another element type", f"table 0: the one tensor of extent {table_units[0]}",
             [in_tensor_record(table_names[0] + ROWS_TENSOR_SUFFIX, "element_type", make_pack.ELEMENT_TYPES["I8"][0])]),
            ("a byte in the fill of a block", "table 0: block 1 of its extent is not zero behind its",
             [behind_rows(0, 1, rows_per_block())]),
            ("a byte behind the last row of a table", f"table 0: block {last_block} of its extent is not zero behind",
             [behind_rows(0, last_block, row_counts[0] - last_block * rows_per_block())])):
        refused_by_verify(what, rule_text, changes)
    refused_by_verify("one bit of a constant and no digest made to fit", "which holds the constants, do not give its",
                      [in_constant(TOKEN_MAP, 0, 1)], digests_made_to_fit = False)
