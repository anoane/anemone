#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Writes the gate file of the linear: what the reference fed to its quantized linears and what they returned, in
ONE binary file that tests/linear_gate_card_test.c reads without a line of JSON.

    make_linear_gate.py <capture folder> <output file>

The capture folder is one that tools/oracle/reference_oracle.py --mode linears wrote: manifest.json and tensors.bin.
It is read with the reader of check_pack_linears.py, imported from beside this file: every rule that tool holds a
capture to holds here (the capture is complete and of that mode; every record lies in tensors.bin with the shape,
the dtype and the reference's dtype of its part; a linear has each part once, all of one weight). That reader leaves
out the input of a linear before the reference quantized it, which the gate needs: the engine quantizes that input
itself and must arrive at the reference's bytes. Its record is found in the manifest here, under the name of the
linear with ".input", and held to the same rules.

Every captured linear becomes one record, in the order of the capture. No pack is read: a record names its weight
as the capture does, which is the name of the matrix in a pack, and the test looks it up there.

The output, little-endian. Nothing lies between its parts but zero bytes up to the next multiple of 8, where every
array begins:
    header      64 bytes:
                  bytes  0 ..  7   the magic "LINEARGT"
                  bytes  8 .. 11   uint32 format version 1
                  bytes 12 .. 15   uint32 linears: the number of records
                  bytes 16 .. 23   uint64 the size of the whole file in bytes
                  bytes 24 .. 27   uint32 the size of a record in bytes: 256
                  bytes 28 .. 31   uint32 the values of an input that share one scale: 32
                  bytes 32 .. 63   zero
    records     256 bytes each, from byte 64 on:
                  bytes   0 ..  95  the name of the weight, as the capture and the pack have it; zero-filled, at
                                    most 95 bytes of printable ASCII
                  bytes  96 .. 159  the name of the linear in the capture ("layer_0.linear.attn.wq_a"); zero-filled,
                                    at most 63 bytes
                  bytes 160 .. 199  the kind of the linear: its name inside its layer, with "*" for the number of a
                                    routed expert ("ffn.experts.*.w1"); zero-filled, at most 39 bytes
                  bytes 200 .. 203  uint32 weight format: 3 for 4-bit weights (ANEMONE_WEIGHT_FORMAT_E2M1_SCALED),
                                    5 for 8-bit weights (ANEMONE_WEIGHT_FORMAT_E4M3_SCALED)
                  bytes 204 .. 207  uint32 rows: the vectors the call multiplied
                  bytes 208 .. 211  uint32 input size: the values of a row that goes in, a multiple of 32
                  bytes 212 .. 215  uint32 output size: the values of a row that comes out
                  bytes 216 .. 223  uint64 where the input lies in the file
                  bytes 224 .. 231  uint64 where the quantized input lies
                  bytes 232 .. 239  uint64 where the input scales lie
                  bytes 240 .. 247  uint64 where the output lies
                  bytes 248 .. 255  zero
    arrays      for each record, in the order of the records:
                  input            uint16 [rows][input size]: the bfloat16 bits of the input before the reference
                                   quantized it
                  quantized input  uint8 [rows][input size]: the E4M3 byte of every value, in the order of the values
                  input scales     uint8 [rows][input size / 32]: the E8M0 byte of every 32 values of a row
                  output           uint16 [rows][output size]: the bfloat16 bits of what the reference's product
                                   returned
A bfloat16 is the upper half of the float32 of the same number, which is how the capture stores it: the lower half
of every such float32 must be zero and the number finite, or nothing is written. A file holds at most 65,536 records,
a call at most 1,048,576 rows, a row at most 65,536 values: the limits the test bounds every count with.

It prints one line per kind of linear, the counts and the SHA-256 of the file. The file is written under a temporary
name beside the destination and renamed when it is whole.

Needs Python 3.9 and numpy. Exit status 0, 1 after any failure, 2 for a command line it cannot read.
Test tool: not part of the engine.
"""
import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import struct
import sys

import numpy

import check_pack_linears                                  # the reader of a capture folder; beside this file
from check_pack_linears import PackError, make_pack

GATE_MAGIC = b"LINEARGT"
GATE_FORMAT_VERSION = 1
GATE_HEADER = struct.Struct("<8sIIQII32x")                 # magic, version, linears, file bytes, record bytes, block
GATE_RECORD = struct.Struct("<96s64s40sIIIIQQQQ8x")        # the three names, format, rows, sizes, the four offsets
WEIGHT_NAME_CAPACITY = 96                                  # of each name: its bytes and at least one zero
LINEAR_NAME_CAPACITY = 64
KIND_NAME_CAPACITY = 40
ARRAY_ALIGNMENT_BYTES = 8
VALUES_PER_SCALE = check_pack_linears.INPUTS_PER_SCALE     # 32
LINEAR_COUNT_LIMIT = 2 ** 16
ROW_COUNT_LIMIT = 2 ** 20
ROW_SIZE_LIMIT = 2 ** 16
INPUT_PART = check_pack_linears.UNREAD_PART                # "input": the part the reader of the capture leaves out
INPUT_DTYPES = ("float32", "bfloat16")                     # of its record: the dtype, and the reference's
BFLOAT16_TYPE = "<u2"
PRINTABLE_ASCII = range(0x20, 0x7F)
TEMPORARY_SUFFIX = ".partial"
EXIT_STATUS_FAILURE = 1

Gate = collections.namedtuple("Gate", ("linear", "input_record", "row_count", "input_size", "output_size",
                                       "weight_format"))


def name_field(name, capacity, what):
    """A name as the bytes of its field, without the zero fill struct adds."""
    encoded = name.encode("utf-8")
    if not encoded or len(encoded) >= capacity or any(byte not in PRINTABLE_ASCII for byte in encoded):
        raise PackError(f"{what} {name!r} is not 1 to {capacity - 1} bytes of printable ASCII")
    return encoded


def input_records_of(capture_folder):
    """The record of every linear's input before quantization, by the name of the linear: found in the manifest and
    held to what read_capture holds the other parts to. read_capture has read the manifest and found it sound."""
    manifest_path = os.path.join(capture_folder, check_pack_linears.MANIFEST_FILE_NAME)
    tensors_path = os.path.join(capture_folder, check_pack_linears.TENSORS_FILE_NAME)
    with open(manifest_path, encoding = "utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook = make_pack.object_without_repeats)
    tensors_byte_count = os.path.getsize(tensors_path)
    records = {}
    for record in manifest["tensors"]:
        name = record.get("name") if isinstance(record, dict) else None
        match = check_pack_linears.LINEAR_RECORD_NAME.fullmatch(name) if isinstance(name, str) else None
        if match is None or match["part"] != INPUT_PART:
            continue
        where = f"{manifest_path}: {name}"
        shape, byte_offset, byte_count = record.get("shape"), record.get("byte_offset"), record.get("byte_count")
        if (record.get("dtype"), record.get("reference_dtype")) != INPUT_DTYPES:
            raise PackError(f"{where}: it is not {INPUT_DTYPES[0]} for the reference's {INPUT_DTYPES[1]}")
        if not make_pack.is_list_of_counts(shape) or not make_pack.is_list_of_counts([byte_offset, byte_count]) \
                or not shape or not byte_count \
                or byte_count != math.prod(shape) * numpy.dtype(INPUT_DTYPES[0]).itemsize \
                or byte_offset + byte_count > tensors_byte_count:
            raise PackError(f"{where}: its shape, byte_offset and byte_count are not those of values that lie in "
                            f"{tensors_path}, which holds {tensors_byte_count} bytes")
        if match["linear"] in records:
            raise PackError(f"{where}: recorded twice")
        records[match["linear"]] = record
    return records


def gates_of(capture_folder):
    """Every captured linear with the record of its input and its sizes, in the order of the capture, and the path
    of tensors.bin."""
    linears, tensors_path = check_pack_linears.read_capture(capture_folder)
    input_records = input_records_of(capture_folder)
    if not linears or len(linears) > LINEAR_COUNT_LIMIT:
        raise PackError(f"{capture_folder} holds {len(linears)} captured linears: 1 to {LINEAR_COUNT_LIMIT}")
    gates = []
    for linear in linears:
        input_record = input_records.get(linear.name)
        weight_format = check_pack_linears.WEIGHT_FORMATS.get(linear.weight_dtype)
        if input_record is None or weight_format is None:
            raise PackError(f"{linear.name}: the capture holds no input before quantization for it, or its weight "
                            f"is of {linear.weight_dtype}, which is no weight format of a pack")
        if (input_record.get("weight"), input_record.get("weight_dtype")) != (linear.weight, linear.weight_dtype):
            raise PackError(f"{linear.name}: its input was recorded for another weight than its output")
        shapes = {part: record["shape"] for part, record in linear.parts.items()}
        row_shape, input_size = input_record["shape"][:-1], input_record["shape"][-1]
        output_size, row_count = shapes["output"][-1], math.prod(input_record["shape"][:-1])
        if shapes["quantized_input"] != row_shape + [input_size] \
                or shapes["input_scales"] != row_shape + [input_size // VALUES_PER_SCALE] \
                or shapes["output"] != row_shape + [output_size] or input_size % VALUES_PER_SCALE:
            raise PackError(f"{linear.name}: its input, quantized input, input scales and output are of the shapes "
                            f"{input_record['shape']}, {shapes['quantized_input']}, {shapes['input_scales']} and "
                            f"{shapes['output']}: not the same rows of one linear with a scale to every "
                            f"{VALUES_PER_SCALE} values")
        if not 1 <= row_count <= ROW_COUNT_LIMIT or not 1 <= input_size <= ROW_SIZE_LIMIT \
                or not 1 <= output_size <= ROW_SIZE_LIMIT:
            raise PackError(f"{linear.name}: {row_count} rows of {input_size} values in and {output_size} out: at "
                            f"most {ROW_COUNT_LIMIT} rows of {ROW_SIZE_LIMIT} values")
        gates.append(Gate(linear, input_record, row_count, input_size, output_size, weight_format))
    return gates, tensors_path


def bfloat16_bits(values, where):
    """float32 values that are bfloat16 numbers, as the 16 bits of each: the upper half of its float32."""
    patterns = numpy.ascontiguousarray(values).view(check_pack_linears.FLOAT32_BITS)
    if (patterns & check_pack_linears.BFLOAT16_UNUSED_MASK).any() or not numpy.isfinite(values).all():
        raise PackError(f"{where} holds a value that is no finite bfloat16")
    return (patterns >> check_pack_linears.BFLOAT16_UNUSED_BITS).astype(BFLOAT16_TYPE)


def aligned(byte_count):
    return make_pack.round_up(byte_count, ARRAY_ALIGNMENT_BYTES)


def array_byte_counts(gate):
    """The bytes of the four arrays of a record, in the order they are written."""
    return (gate.row_count * gate.input_size * numpy.dtype(BFLOAT16_TYPE).itemsize,
            gate.row_count * gate.input_size, gate.row_count * (gate.input_size // VALUES_PER_SCALE),
            gate.row_count * gate.output_size * numpy.dtype(BFLOAT16_TYPE).itemsize)


def make_gate(capture_folder, output_path):
    """Reads the capture, lays the file out, writes it and prints what it holds."""
    gates, tensors_path = gates_of(capture_folder)
    records, byte_offset = [], aligned(GATE_HEADER.size + len(gates) * GATE_RECORD.size)
    for gate in gates:
        offsets = []
        for byte_count in array_byte_counts(gate):
            offsets.append(byte_offset)
            byte_offset = aligned(byte_offset + byte_count)
        records.append(GATE_RECORD.pack(
            name_field(gate.linear.weight, WEIGHT_NAME_CAPACITY, "the weight"),
            name_field(gate.linear.name, LINEAR_NAME_CAPACITY, "the linear"),
            name_field(gate.linear.kind, KIND_NAME_CAPACITY, "the kind of linear"),
            gate.weight_format, gate.row_count, gate.input_size, gate.output_size, *offsets))
    file_byte_count = byte_offset
    header = GATE_HEADER.pack(GATE_MAGIC, GATE_FORMAT_VERSION, len(gates), file_byte_count, GATE_RECORD.size,
                              VALUES_PER_SCALE)
    digest, written_byte_count = hashlib.sha256(), 0
    kinds = collections.Counter()
    temporary_path = output_path + TEMPORARY_SUFFIX

    def write(handle, content):
        """Writes one part and the zero bytes up to the next multiple of the alignment."""
        nonlocal written_byte_count
        content = bytes(content) + bytes(aligned(len(content)) - len(content))
        handle.write(content)
        digest.update(content)
        written_byte_count += len(content)

    try:
        with open(temporary_path, "wb") as output, open(tensors_path, "rb", buffering = 0) as capture:
            write(output, header + b"".join(records))
            for gate in gates:
                linear = gate.linear
                unquantized = check_pack_linears.read_part(capture, gate.input_record, tensors_path)
                quantized_input, input_scales, captured = (
                    check_pack_linears.read_part(capture, linear.parts[part], tensors_path)
                    for part in check_pack_linears.PARTS)
                arrays = (bfloat16_bits(unquantized, f"{tensors_path}: {linear.name}.input"), quantized_input,
                          input_scales, bfloat16_bits(captured, f"{tensors_path}: {linear.name}.output"))
                for array, byte_count in zip(arrays, array_byte_counts(gate)):
                    content = numpy.ascontiguousarray(array).tobytes()
                    if len(content) != byte_count:
                        raise PackError(f"{linear.name}: an array of {len(content)} bytes where its record says "
                                        f"{byte_count}")
                    write(output, content)
                kinds[linear.kind] += 1
            if written_byte_count != file_byte_count:
                raise PackError(f"{written_byte_count} bytes were written where the header says {file_byte_count}")
        os.replace(temporary_path, output_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary_path)
        raise
    for kind, count in sorted(kinds.items()):
        rows = sum(gate.row_count for gate in gates if gate.linear.kind == kind)
        print(f"  {kind}: {count} linears, {rows} rows")
    print(f"{output_path}: {len(gates)} linears of {len(kinds)} kinds, {file_byte_count} bytes")
    print(f"  sha256 {digest.hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_folder")
    parser.add_argument("output_file")
    arguments = parser.parse_args()
    try:
        make_gate(arguments.capture_folder, arguments.output_file)
    except (PackError, OSError) as failure:
        print(f"make_linear_gate FAILED: {failure}", file = sys.stderr)
        return EXIT_STATUS_FAILURE
    return 0


if __name__ == "__main__":
    sys.exit(main())
