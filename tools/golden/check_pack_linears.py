#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proves on real data that a converted pack holds the weights the reference computed with.

    check_pack_linears.py <pack file> <capture folder>

The capture folder is one that tools/oracle/reference_oracle.py --mode linears wrote: manifest.json and tensors.bin,
which hold for every recorded call of the reference's quantized linear (model.py:181) the input as the reference
quantized it (E4M3 bytes), the scales of that input (E8M0 bytes, one for every 32 inputs of a row) and what the
reference's product returned (bfloat16 values, written as float32). The record of each names the weight it ran on.

For every captured linear whose weight is a matrix of a weight format in the pack, the matrix is read from the pack
(the tables through make_pack.py's read_pack_tables, the bytes through kernel_layouts.py's read_matrix, which is the
inverse of the conversion and written apart from it), and every captured row is computed again in float64:

    y[o] = sum over the groups k of 32 inputs of
           (sum over the inputs i of group k of W[o][i] * x[i]) * 2^(s[o][k] - 127) * 2^(a[k] - 127)

W are the values of the matrix's codes and s its scales (at 4 bits one for every output and group, at 8 bits one for
every 32 outputs and group); x are the values of the quantized input and a its scales. The sum of a group is exact
in float64 (its terms are multiples of 2^-18 below 2^18, so 41 bits hold any partial sum), a power of two scales it
exactly, and the groups are added in their order: the result is the same on every machine. y is rounded to bfloat16,
to nearest with ties to even, and compared with the capture.

The reference adds in float32 and in another order. So a small share of its outputs lies one bfloat16 step from
this result, and an output that nearly cancels may lie several steps from it while the two differ by next to
nothing: neither is a failure. A wrong half of a byte, a wrong arrangement of the scales or a read_matrix that does
not undo the conversion gives other numbers altogether. A captured linear FAILS when
  - fewer than 99% of its outputs lie within one last place of the capture. With the right weights an output leaves
    that range only where its sum nearly cancels; a number that has nothing to do with the reference's lies in it 3
    times in 128 at best, the three neighbouring fractions of the right sign and exponent. Or when
  - one of its outputs differs from the capture by more than 1/16 of the largest magnitude among the linear's
    captured outputs. A step of bfloat16 is at most 1/128 of its value, and an output that nearly cancels differs
    by little however many steps that is; a wrong number is off by about the size of an output. This threshold is
    what shows a fault in a few outputs only, which the share above would let pass.
Both thresholds are chosen far from either side and are not measured ones: the lines below print what a run finds.
One line is printed for each kind of linear (the routed experts of all numbers are one kind): the share of outputs
equal bit for bit, the share within one last place, the largest difference in last places and as a share of its
linear's largest output, and the linear it lies in. Then the verdict.

Also a failure: a captured linear whose weight is no matrix of a weight format in the pack (the reference ran it
quantized, so the converter must have made it one), or whose format or shape is not the capture's; a capture that is
not complete, not of --mode linears, or whose records do not agree with tensors.bin. No failure, but counted in the
verdict: the matrices of the pack that no captured linear ran on. The capture records the layers and expert calls it
was asked for, and never attn.wo_a, which the reference multiplies by its scales into bfloat16 (its convert.py:157)
and runs without quantizing the input (model.py:207).

What this does NOT read: the digests of the pack's extents (make_pack.py verify does), and the unquantized input of
the capture. What it does NOT show: that a stored byte lies where the kernels read it. read_matrix undoes
store_matrix, so a place both of them get wrong alike passes here; the lane order is held to weight_format_seam.h by
make_pack.py's self-test, and the kernels to it by the engine's own tests. It needs numpy and no torch. Exit status
0, 1 after any failure, 2 for a command line it cannot read. Test tool: not part of the engine.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import time

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "convert"))
import kernel_layouts                                      # the inverse of the conversion
import make_pack                                           # the reader of a pack's tables

# ---- The capture folder: the format of tools/oracle/reference_oracle.py ---------------------------------------------
MANIFEST_FILE_NAME = "manifest.json"
TENSORS_FILE_NAME = "tensors.bin"
CAPTURE_KIND = "linears"                                   # conditions.capture of the manifest
LINEAR_RECORD_NAME = re.compile(r"(?P<linear>layer_(?:0|[1-9][0-9]*)\.linear\.(?P<name_in_layer>.+))"
                                r"\.(?P<part>input|quantized_input|input_scales|output)")
UNREAD_PART = "input"                                      # before the reference quantizes it
# The parts of a linear this tool reads: (numpy dtype of the record, what the reference holds the values in)
PARTS = {"quantized_input": ("uint8", "float8_e4m3fn"), "input_scales": ("uint8", "float8_e8m0fnu"),
         "output": ("float32", "bfloat16")}
ROUTED_EXPERT_NUMBER = re.compile(r"(?<=ffn\.experts\.)(?:0|[1-9][0-9]*)(?=\.)")
ANY_EXPERT = "*"                                           # in place of the number, in the name of a kind
# weight_dtype of a record, as model.py:186 and :196 tell the two apart: the weight format of the pack's matrix
WEIGHT_FORMATS = {"float4_e2m1fn_x2": kernel_layouts.WEIGHT_FORMAT_E2M1_SCALED,
                  "float8_e4m3fn": kernel_layouts.WEIGHT_FORMAT_E4M3_SCALED}

# ---- The numbers: include/anemone/weight_format_seam.h, and bfloat16 -------------------------------------------------
INPUTS_PER_SCALE = 32
E4M3_OUTPUTS_PER_SCALE = 32
E8M0_BIAS = 127                                            # a scale byte c stands for 2^(c - 127)
BYTE_VALUES = 256
E4M3_SIGN_BIT = 0x80
E4M3_MAGNITUDE_MASK = 0x7F                                 # all of it set: no number
E4M3_MANTISSA_BITS = 3
E4M3_MANTISSA_STEPS = 2 ** E4M3_MANTISSA_BITS
E4M3_EXPONENT_MASK = 0xF
E4M3_EXPONENT_BIAS = 7
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # of the codes 0 to 7; bit 3 is the sign
BFLOAT16_FRACTION_BITS = 7
BFLOAT16_SMALLEST_EXPONENT = -126                          # below 2^-126 the last place stays 2^-133
BFLOAT16_UNUSED_BITS = 16                                  # of the float32 that holds a bfloat16: its low half, zero
BFLOAT16_UNUSED_MASK = 0xFFFF
BFLOAT16_SIGN_BIT = 0x8000
BFLOAT16_MAGNITUDE_MASK = 0x7FFF
FLOAT32_BITS = "<u4"

# ---- The verdict -----------------------------------------------------------------------------------------------------
WITHIN_ONE_SHARE_REQUIRED = 0.99                           # of the outputs of every linear
LARGEST_DIFFERENCE_ALLOWED = 1 / 16                        # of the largest captured magnitude of the linear
FAILED_LINEARS_SHOWN = 8
PROGRESS_INTERVAL_SECONDS = 10.0
PERCENT = 100.0
EXIT_STATUS_FAILURE = 1
PackError = make_pack.PackError                            # one line on the standard error stream, exit status 1

Linear = collections.namedtuple("Linear", ("name", "kind", "weight", "weight_dtype", "parts"))
Comparison = collections.namedtuple("Comparison", ("output_count", "equal_count", "within_one_count",
                                                   "largest_last_places", "largest_share", "failure"))


def e4m3_values():
    """The value of every E4M3 byte, float64 [256]: a sign, 4 exponent bits of bias 7 and 3 mantissa bits; exponent
    0 is subnormal, and the two codes with every magnitude bit set are no number."""
    codes = numpy.arange(BYTE_VALUES)
    exponents, mantissas = (codes >> E4M3_MANTISSA_BITS) & E4M3_EXPONENT_MASK, codes & (E4M3_MANTISSA_STEPS - 1)
    significands = numpy.where(exponents == 0, mantissas, mantissas + E4M3_MANTISSA_STEPS) / E4M3_MANTISSA_STEPS
    magnitudes = numpy.ldexp(significands, numpy.maximum(exponents, 1) - E4M3_EXPONENT_BIAS)
    values = numpy.where(codes & E4M3_SIGN_BIT, -magnitudes, magnitudes)
    values[(codes & E4M3_MAGNITUDE_MASK) == E4M3_MAGNITUDE_MASK] = numpy.nan
    return values


def code_values(weight_format):
    """The value of every code of a weight format, float64, by code."""
    if weight_format == kernel_layouts.WEIGHT_FORMAT_E4M3_SCALED:
        return e4m3_values()
    return numpy.array(E2M1_MAGNITUDES + tuple(-magnitude for magnitude in E2M1_MAGNITUDES))


def products_of(weight_format, codes, scales, quantized_input, input_scales):
    """y of the text at the top, float64 [rows, outputs]. codes: uint8 [outputs, inputs]; scales: uint8 in the
    checkpoint's arrangement; quantized_input: uint8 [rows, inputs]; input_scales: uint8 [rows, inputs / 32]."""
    output_size, input_size = codes.shape
    weight_values, input_values = code_values(weight_format), e4m3_values()
    weight_exponents = scales.astype(numpy.int64) - E8M0_BIAS                      # [outputs or blocks, groups]
    if weight_format == kernel_layouts.WEIGHT_FORMAT_E4M3_SCALED:                  # the last block: whole or half
        weight_exponents = numpy.repeat(weight_exponents, E4M3_OUTPUTS_PER_SCALE, axis = 0)[:output_size]
    input_exponents = input_scales.astype(numpy.int64) - E8M0_BIAS                 # [rows, groups]
    products = numpy.zeros((quantized_input.shape[0], output_size))
    for group in range(input_size // INPUTS_PER_SCALE):
        inputs = slice(group * INPUTS_PER_SCALE, (group + 1) * INPUTS_PER_SCALE)
        # both operands carry their scale, a power of two: every product of the group has the same one, so their
        # sum is exact whatever the order numpy adds them in
        weights = numpy.ldexp(weight_values[codes[:, inputs]], weight_exponents[:, group, None])
        activations = numpy.ldexp(input_values[quantized_input[:, inputs]], input_exponents[:, group, None])
        products += activations @ weights.T
    return products


def rounded_to_bfloat16(values):
    """float64 values rounded to bfloat16, to nearest with ties to even, as float32. A value's last place is 2^-7 of
    its leading bit, and 2^-133 for every value below 2^-126; what rounds to 2^128 or beyond becomes infinite."""
    _, exponents = numpy.frexp(values)                         # a value is m * 2^exponent with 1/2 <= |m| < 1
    last_places = numpy.maximum(exponents - 1, BFLOAT16_SMALLEST_EXPONENT) - BFLOAT16_FRACTION_BITS
    rounded = numpy.ldexp(numpy.rint(numpy.ldexp(values, -last_places)), last_places)
    with numpy.errstate(over = "ignore"):
        return rounded.astype(numpy.float32)


def last_place_numbers(bfloat16_values):
    """bfloat16 values, held in float32, as whole numbers that are one apart for neighbouring values: the 15 bits of
    exponent and fraction, negative for a negative value. Both zeros are 0."""
    patterns = (numpy.ascontiguousarray(bfloat16_values).view(FLOAT32_BITS) >> BFLOAT16_UNUSED_BITS).astype(numpy.int64)
    magnitudes = patterns & BFLOAT16_MAGNITUDE_MASK
    return numpy.where(patterns & BFLOAT16_SIGN_BIT, -magnitudes, magnitudes)


def compare(products, captured):
    """One linear's float64 products against the capture's output, float32 [rows, outputs] of bfloat16 values: a
    Comparison, whose failure is the text of the threshold it misses, or None."""
    computed = rounded_to_bfloat16(products)
    last_places = numpy.abs(last_place_numbers(computed) - last_place_numbers(captured))
    with numpy.errstate(invalid = "ignore"):
        largest_difference = float(numpy.abs(computed.astype(numpy.float64) - captured.astype(numpy.float64)).max())
    largest_output = float(numpy.abs(captured).max())
    within_one_count = int((last_places <= 1).sum())
    largest_share = largest_difference / largest_output if largest_output else float(largest_difference != 0.0)
    failure = None
    if within_one_count < WITHIN_ONE_SHARE_REQUIRED * captured.size:
        failure = (f"{PERCENT * within_one_count / captured.size:.4f}% of its outputs within one last place, not "
                   f"{PERCENT * WITHIN_ONE_SHARE_REQUIRED:g}%")
    elif not largest_share <= LARGEST_DIFFERENCE_ALLOWED:      # written so that a difference that is no number fails
        failure = (f"an output differs by {PERCENT * largest_share:.4f}% of the largest output, more than "
                   f"{PERCENT * LARGEST_DIFFERENCE_ALLOWED:g}%")
    equal_count = int((numpy.ascontiguousarray(computed).view(FLOAT32_BITS)
                       == numpy.ascontiguousarray(captured).view(FLOAT32_BITS)).sum())
    return Comparison(captured.size, equal_count, within_one_count, int(last_places.max()), largest_share, failure)


def read_capture(capture_folder):
    """The linears of a capture folder, in the order they were recorded, each with the records of the parts this
    tool reads, after everything a record says has been checked against the two files. Returns them and the path of
    tensors.bin."""
    manifest_path = os.path.join(capture_folder, MANIFEST_FILE_NAME)
    tensors_path = os.path.join(capture_folder, TENSORS_FILE_NAME)
    try:
        with open(manifest_path, encoding = "utf-8") as handle:
            manifest = json.load(handle, object_pairs_hook = make_pack.object_without_repeats)
    except (ValueError, RecursionError) as failure:
        raise PackError(f"{manifest_path} cannot be read: {failure}") from failure
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise PackError(f"{manifest_path} does not say \"complete\": true: the capture did not end well")
    conditions, records = manifest.get("conditions"), manifest.get("tensors")
    if not isinstance(conditions, dict) or conditions.get("capture") != CAPTURE_KIND or not isinstance(records, list):
        raise PackError(f"{manifest_path} is no capture of reference_oracle.py --mode {CAPTURE_KIND}")
    tensors_byte_count = os.path.getsize(tensors_path)
    parts_of_linears = collections.defaultdict(dict)
    for record in records:
        name = record.get("name") if isinstance(record, dict) else None
        match = LINEAR_RECORD_NAME.fullmatch(name) if isinstance(name, str) else None
        if match is None or match["part"] == UNREAD_PART:
            continue                                           # no part of a linear, or its input before quantization
        where = f"{manifest_path}: {name}"
        dtype, reference_dtype = PARTS[match["part"]]
        shape, byte_offset, byte_count = record.get("shape"), record.get("byte_offset"), record.get("byte_count")
        if (record.get("dtype"), record.get("reference_dtype")) != (dtype, reference_dtype):
            raise PackError(f"{where}: it is not {dtype} for the reference's {reference_dtype}")
        if not make_pack.is_list_of_counts(shape) or not make_pack.is_list_of_counts([byte_offset, byte_count]) \
                or not shape or not byte_count or byte_count != math.prod(shape) * numpy.dtype(dtype).itemsize \
                or byte_offset + byte_count > tensors_byte_count:
            raise PackError(f"{where}: its shape, byte_offset and byte_count are not those of values that lie in "
                            f"{tensors_path}, which holds {tensors_byte_count} bytes")
        if not isinstance(record.get("weight"), str) or not isinstance(record.get("weight_dtype"), str):
            raise PackError(f"{where}: it does not name its weight and the dtype of its weight")
        if match["part"] in parts_of_linears[match["linear"]]:
            raise PackError(f"{where}: recorded twice")
        parts_of_linears[match["linear"]][match["part"]] = (record, match["name_in_layer"])
    linears = []
    for name, parts in parts_of_linears.items():
        weights = {(record["weight"], record["weight_dtype"]) for record, _ in parts.values()}
        if set(parts) != set(PARTS) or len(weights) != 1:
            raise PackError(f"{manifest_path}: {name}: its records are not {', '.join(PARTS)} of one weight")
        (weight, weight_dtype), (_, name_in_layer) = next(iter(weights)), parts["output"]
        linears.append(Linear(name, ROUTED_EXPERT_NUMBER.sub(ANY_EXPERT, name_in_layer), weight, weight_dtype,
                              {part: record for part, (record, _) in parts.items()}))
    return linears, tensors_path


def read_part(handle, record, tensors_path):
    """The values of one record of the capture as rows: [rows, the record's last dimension]. handle: tensors.bin,
    open without a buffer."""
    values = numpy.empty(math.prod(record["shape"]), dtype = record["dtype"])
    make_pack.read_exactly(handle, record["byte_offset"], memoryview(values).cast("B"), tensors_path)
    return values.reshape(-1, record["shape"][-1])


def check(pack_path, capture_folder):
    """Compares every captured linear with the pack and prints the lines of the text at the top; returns the exit
    status."""
    started = last_report = time.monotonic()
    _, extents, _, tensors = make_pack.read_pack_tables(pack_path)
    matrices = {tensor["name"].decode("utf-8", "replace"): tensor for tensor in tensors
                if tensor["weight_format"] != make_pack.WEIGHT_FORMAT_NONE}
    linears, tensors_path = read_capture(capture_folder)
    if not linears:
        raise PackError(f"{capture_folder} holds no captured linear")
    kinds, failures, covered = collections.defaultdict(list), [], set()
    with open(pack_path, "rb", buffering = 0) as pack, open(tensors_path, "rb", buffering = 0) as capture:
        for linear_index, linear in enumerate(linears):
            tensor, weight_format = matrices.get(linear.weight), WEIGHT_FORMATS.get(linear.weight_dtype)
            if tensor is None or tensor["weight_format"] != weight_format:
                failures.append(f"{linear.name}: the reference ran it on {linear.weight} as {linear.weight_dtype}, and "
                                "the pack holds no matrix of that name and that format")
                continue
            output_size, input_size = tensor["dimensions"][:tensor["dimension_count"]]
            quantized_input, input_scales, captured = (read_part(capture, linear.parts[part], tensors_path)
                                                       for part in PARTS)
            if (quantized_input.shape, input_scales.shape, captured.shape) != (
                    (len(captured), input_size), (len(captured), input_size // INPUTS_PER_SCALE),
                    (len(captured), output_size)):
                failures.append(f"{linear.name}: its rows are {quantized_input.shape[-1]} inputs, "
                                f"{input_scales.shape[-1]} scales and {captured.shape[-1]} outputs, the pack's "
                                f"{linear.weight} has {output_size} outputs by {input_size} inputs")
                continue
            if (numpy.ascontiguousarray(captured).view(FLOAT32_BITS) & BFLOAT16_UNUSED_MASK).any() \
                    or not numpy.isfinite(captured).all():
                raise PackError(f"{tensors_path}: {linear.name}.output holds a value that is no finite bfloat16")
            stored = bytearray(tensor["byte_count"])
            make_pack.read_exactly(pack, extents[tensor["extent_index"]]["byte_offset"]
                                   + tensor["byte_offset_in_extent"], memoryview(stored), pack_path)
            codes, scales = kernel_layouts.read_matrix(weight_format, output_size, input_size, stored)
            comparison = compare(products_of(weight_format, codes, scales, quantized_input, input_scales), captured)
            kinds[linear.kind].append((linear.name, comparison))
            covered.add(linear.weight)
            if comparison.failure:
                failures.append(f"{linear.name} ({linear.weight}): {comparison.failure}")
            if time.monotonic() - last_report >= PROGRESS_INTERVAL_SECONDS:
                last_report = time.monotonic()
                print(f"  compared {linear_index + 1} of {len(linears)} linears", file = sys.stderr, flush = True)

    for kind, comparisons in sorted(kinds.items()):
        output_count = sum(comparison.output_count for _, comparison in comparisons)
        equal_count = sum(comparison.equal_count for _, comparison in comparisons)
        within_one_count = sum(comparison.within_one_count for _, comparison in comparisons)
        worst_name, worst = max(comparisons, key = lambda named: (named[1].largest_last_places, named[1].largest_share))
        print(f"{kind}: {len(comparisons)} linears, {output_count} outputs: "
              f"{PERCENT * equal_count / output_count:.4f}% equal bit for bit, "
              f"{PERCENT * within_one_count / output_count:.4f}% within one last place; the largest difference: "
              f"{worst.largest_last_places} last places, {PERCENT * worst.largest_share:.4f}% of the largest output of "
              f"{worst_name}")
    for failure in failures[:FAILED_LINEARS_SHOWN]:
        print(f"FAILED: {failure}", file = sys.stderr)
    if len(failures) > FAILED_LINEARS_SHOWN:
        print(f"FAILED: and {len(failures) - FAILED_LINEARS_SHOWN} more linears", file = sys.stderr)
    compared_count = sum(len(comparisons) for comparisons in kinds.values())
    print(f"check_pack_linears {'FAILED' if failures else 'PASSED'}: {len(failures)} of {len(linears)} captured "
          f"linears fail; {compared_count} linears of {len(kinds)} kinds were compared, each to pass with "
          f"{PERCENT * WITHIN_ONE_SHARE_REQUIRED:g}% of its outputs within one last place of the reference and none "
          f"further from it than {PERCENT * LARGEST_DIFFERENCE_ALLOWED:g}% of its largest output; "
          f"{len(matrices) - len(covered)} of the pack's {len(matrices)} matrices of a weight format are covered by "
          f"no captured linear; {time.monotonic() - started:.1f} s")
    return EXIT_STATUS_FAILURE if failures else 0


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack_file")
    parser.add_argument("capture_folder")
    arguments = parser.parse_args()
    try:
        return check(arguments.pack_file, arguments.capture_folder)
    except (PackError, OSError) as failure:
        print(f"check_pack_linears FAILED: {failure}", file = sys.stderr)
        return EXIT_STATUS_FAILURE


if __name__ == "__main__":
    sys.exit(main())
