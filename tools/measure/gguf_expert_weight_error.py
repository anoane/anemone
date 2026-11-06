#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How far the routed experts of a GGUF quantization of DeepSeek-V4.1-Flash lie from the checkpoint's own weights.

A first, cheap look at a re-quantized model (the GSQ-RCO GGUF stores the routed experts as Q2_K in some layers and
Q3_K in others): for a sample of experts, each of the three matrices is decoded from the GGUF and from the
checkpoint, and their difference is printed relative to the checkpoint's matrix (the root of the summed squares of
the difference over that of the weights) with the cosine between the two. It is NOT the model's error: the method
that made the file fits layer outputs, not weights, so a large weight error may cost little. The model's error is
measured by the oracle on logits.

    gguf_expert_weight_error.py <gguf file> <checkpoint folder> [--layers 0,1,20] [--experts 0,7,200]

The GGUF is read by this file alone, from the format's definition: a header (metadata, then for every tensor its
name, dimensions innermost first, type and offset into the data, which begins at the next multiple of the
alignment behind the header), and blocks of 256 weights along a row:
  Q2_K  84 bytes: 16 bytes of 4-bit scale (low) and 4-bit minimum (high) for the 16 groups of 16 weights; 64 bytes
        of 2-bit codes; two float16, d and dmin. A weight is d * scale * code - dmin * minimum.
  Q3_K  110 bytes: 32 bytes with the high bit of every code; 64 bytes with the two low bits; 12 bytes holding
        sixteen 6-bit scales; one float16 d. A weight is d * (scale - 32) * (code - 4 if its high bit is clear).
In both, the codes of a half of 128 weights lie in 32 bytes: group 2 j takes bits 2 j and 2 j + 1 of bytes 0..15,
group 2 j + 1 the same bits of bytes 16..31.
The checkpoint's experts are 4-bit codes, two to a byte with the low half first, times a power-of-two scale per
output and 32 inputs.
Measurement tool: not part of the engine.
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy

GGUF_MAGIC = b"GGUF"
GGUF_VERSIONS_READ = (2, 3)                  # version 1 counts in 32 bits: its header would be misread, not refused
GGUF_DEFAULT_ALIGNMENT = 32
GGUF_SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
                       12: "<d"}
GGUF_STRING, GGUF_ARRAY = 8, 9
GGUF_TYPE_Q2_K, GGUF_TYPE_Q3_K = 10, 11
BLOCK_WEIGHTS = 256
GROUP_WEIGHTS = 16
HALF_WEIGHTS = 128
HALF_CODE_BYTES = 32
Q2_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES = 84, 110
FLOAT16_BYTES = 2
CODE_BITS, CODE_MASK = 2, 3                  # the 2-bit codes of both types
HALF_BYTE_BITS, HALF_BYTE_MASK = 4, 15
Q2_K_SCALE_BYTES, Q2_K_CODE_BYTES = 16, 64   # a Q2_K block: scales and minimums, codes, then d and dmin
Q2_K_D_OFFSET = Q2_K_SCALE_BYTES + Q2_K_CODE_BYTES
Q2_K_DMIN_OFFSET = Q2_K_D_OFFSET + FLOAT16_BYTES
Q3_K_HIGH_BIT_BYTES, Q3_K_CODE_BYTES, Q3_K_SCALE_BYTES = 32, 64, 12    # a Q3_K block: high bits, low bits, scales, d
Q3_K_SCALE_OFFSET = Q3_K_HIGH_BIT_BYTES + Q3_K_CODE_BYTES
Q3_K_D_OFFSET = Q3_K_SCALE_OFFSET + Q3_K_SCALE_BYTES
Q3_K_CODE_STEP_WITHOUT_HIGH_BIT = 4          # subtracted from a code whose high bit is clear
Q3_K_SCALE_BIAS = 32                         # a 6-bit scale is stored plus 32
Q3_K_LOW_HALF_WORDS = 2
Q3_K_SCALE_WORD_BYTES = 4                    # the scales' low halves lie in two words of four bytes, the high pairs
Q3_K_SCALES_PER_HALF_BYTE = 8                # in a third: scales 0..7 take the low halves, 8..15 the high ones
BYTE_BITS = 8
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAGNITUDE_MASK, E2M1_SIGN_BIT = 7, 8
E8M0_BIAS = 127
WEIGHTS_PER_SCALE = 32
SHARD_PATTERN = "model-*.safetensors"
SHARD_HEADER_LENGTH_BYTES = 8
MATRICES = (("gate", "w1"), ("up", "w3"), ("down", "w2"))       # the GGUF's name, the checkpoint's


def read_gguf_tensors(path):
    """name -> (dimensions innermost first, type, byte offset in the file) for every tensor of a GGUF."""
    with open(path, "rb") as handle:
        def take(layout):
            return struct.unpack(layout, handle.read(struct.calcsize(layout)))[0]

        def take_string():
            return handle.read(take("<Q")).decode("utf8", "replace")

        def take_value(kind):
            if kind in GGUF_SCALAR_FORMATS:
                return take(GGUF_SCALAR_FORMATS[kind])
            if kind == GGUF_STRING:
                return take_string()
            if kind != GGUF_ARRAY:
                raise ValueError(f"a metadata value of the unknown kind {kind}")
            inner, count = take("<I"), take("<Q")
            if inner in GGUF_SCALAR_FORMATS:
                handle.seek(struct.calcsize(GGUF_SCALAR_FORMATS[inner]) * count, os.SEEK_CUR)
                return None
            return [take_value(inner) for _ in range(count)]

        if handle.read(len(GGUF_MAGIC)) != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = take("<I")
        if version not in GGUF_VERSIONS_READ:
            raise ValueError(f"a GGUF of version {version}: versions {GGUF_VERSIONS_READ} are read")
        tensor_count, metadata_count = take("<Q"), take("<Q")
        alignment = GGUF_DEFAULT_ALIGNMENT
        for _ in range(metadata_count):
            key, kind = take_string(), take("<I")
            value = take_value(kind)
            if key == "general.alignment":
                alignment = value
        tensors = {}
        for _ in range(tensor_count):
            name, dimension_count = take_string(), take("<I")
            dimensions = [take("<Q") for _ in range(dimension_count)]
            tensors[name] = (dimensions, take("<I"), take("<Q"))
        data_start = -(-handle.tell() // alignment) * alignment
    return {name: (dimensions, kind, data_start + offset) for name, (dimensions, kind, offset) in tensors.items()}


def group_codes(code_bytes):
    """[blocks, 64] bytes of 2-bit codes -> [blocks, 16 groups, 16 weights], as the format orders them."""
    sides = HALF_CODE_BYTES // GROUP_WEIGHTS                               # groups 2 j and 2 j + 1 share bit positions
    halves = code_bytes.reshape(-1, BLOCK_WEIGHTS // HALF_WEIGHTS, sides, GROUP_WEIGHTS)   # [block, half, side, 16]
    shifts = (numpy.arange(HALF_WEIGHTS // (sides * GROUP_WEIGHTS), dtype = numpy.uint8) * CODE_BITS) \
        .reshape(1, 1, -1, 1, 1)
    picked = (halves[:, :, None, :, :] >> shifts) & CODE_MASK              # [block, half, j, side, 16]
    return picked.reshape(-1, BLOCK_WEIGHTS // GROUP_WEIGHTS, GROUP_WEIGHTS)


def decode_q2_k(raw):
    blocks = raw.reshape(-1, Q2_K_BLOCK_BYTES)
    scale_bytes = blocks[:, :Q2_K_SCALE_BYTES]
    codes = group_codes(blocks[:, Q2_K_SCALE_BYTES:Q2_K_D_OFFSET]).astype(numpy.float32)
    d = blocks[:, Q2_K_D_OFFSET:Q2_K_DMIN_OFFSET].copy().view("<f2").astype(numpy.float32)             # [block, 1]
    dmin = blocks[:, Q2_K_DMIN_OFFSET:Q2_K_BLOCK_BYTES].copy().view("<f2").astype(numpy.float32)
    scale = (d * (scale_bytes & HALF_BYTE_MASK).astype(numpy.float32))[:, :, None]
    minimum = (dmin * (scale_bytes >> HALF_BYTE_BITS).astype(numpy.float32))[:, :, None]
    return (scale * codes - minimum).reshape(-1)


def decode_q3_k(raw):
    blocks = raw.reshape(-1, Q3_K_BLOCK_BYTES)
    # the high bit of weight w of group 2 j + side of half h is bit 4 h + j of byte side * 16 + w
    sides, halves = Q3_K_HIGH_BIT_BYTES // GROUP_WEIGHTS, BLOCK_WEIGHTS // HALF_WEIGHTS
    bits = numpy.unpackbits(blocks[:, :Q3_K_HIGH_BIT_BYTES], axis = 1, bitorder = "little") \
        .reshape(-1, sides, GROUP_WEIGHTS, halves, BYTE_BITS // halves)    # [block, side, weight, half, j]
    high = bits.transpose(0, 3, 4, 1, 2).reshape(-1, BLOCK_WEIGHTS // GROUP_WEIGHTS, GROUP_WEIGHTS)
    codes = group_codes(blocks[:, Q3_K_HIGH_BIT_BYTES:Q3_K_SCALE_OFFSET]).astype(numpy.int32) \
        - numpy.where(high != 0, 0, Q3_K_CODE_STEP_WITHOUT_HIGH_BIT)
    # sixteen 6-bit scales in 12 bytes: scale k has its low four bits in a half of byte k % 4 of the first word
    # (k / 4 even) or the second (odd), the low half for k < 8 and the high half above; its two high bits are bits
    # 2 (k / 4) and 2 (k / 4) + 1 of byte 8 + k % 4
    scale_bytes = blocks[:, Q3_K_SCALE_OFFSET:Q3_K_D_OFFSET]
    scale_index = numpy.arange(BLOCK_WEIGHTS // GROUP_WEIGHTS)
    byte, word = scale_index % Q3_K_SCALE_WORD_BYTES, scale_index // Q3_K_SCALE_WORD_BYTES
    source = scale_bytes[:, byte + Q3_K_SCALE_WORD_BYTES * (word % Q3_K_LOW_HALF_WORDS)]
    low = numpy.where(scale_index < Q3_K_SCALES_PER_HALF_BYTE, source & HALF_BYTE_MASK, source >> HALF_BYTE_BITS)
    high_pairs = scale_bytes[:, Q3_K_LOW_HALF_WORDS * Q3_K_SCALE_WORD_BYTES + byte]
    high_pair = (high_pairs >> (CODE_BITS * word).astype(numpy.uint8)) & CODE_MASK
    scales = (low | (high_pair << HALF_BYTE_BITS)).astype(numpy.int32) - Q3_K_SCALE_BIAS
    d = blocks[:, Q3_K_D_OFFSET:Q3_K_BLOCK_BYTES].copy().view("<f2").astype(numpy.float32)
    return ((d * scales.astype(numpy.float32))[:, :, None] * codes.astype(numpy.float32)).reshape(-1)


def read_shard_headers(checkpoint_folder):
    tensors = {}
    for path in sorted(glob.glob(os.path.join(checkpoint_folder, SHARD_PATTERN))):
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(SHARD_HEADER_LENGTH_BYTES))[0]
            header = json.loads(handle.read(length))
        for name, record in header.items():
            if name != "__metadata__":
                tensors[name] = (path, SHARD_HEADER_LENGTH_BYTES + length + record["data_offsets"][0], record["shape"])
    return tensors


def checkpoint_expert_matrix(shards, layer, expert, name):
    """The checkpoint's matrix [outputs, inputs] in float32: codes times their power-of-two scales."""
    path, offset, shape = shards[f"layers.{layer}.ffn.experts.{expert}.{name}.weight"]
    packed = numpy.fromfile(path, dtype = numpy.uint8, count = shape[0] * shape[1], offset = offset).reshape(shape)
    codes = numpy.stack((packed & HALF_BYTE_MASK, packed >> HALF_BYTE_BITS), axis = -1) \
        .reshape(shape[0], -1)                                              # the low half first
    values = numpy.array(E2M1_MAGNITUDES, dtype = numpy.float32)[codes & E2M1_MAGNITUDE_MASK] \
        * numpy.where(codes & E2M1_SIGN_BIT, numpy.float32(-1.0), numpy.float32(1.0))
    path, offset, scale_shape = shards[f"layers.{layer}.ffn.experts.{expert}.{name}.scale"]
    scale_codes = numpy.fromfile(path, dtype = numpy.uint8, count = scale_shape[0] * scale_shape[1],
                                 offset = offset).reshape(scale_shape).astype(numpy.int32)
    scales = numpy.ldexp(numpy.float32(1.0), scale_codes - E8M0_BIAS).astype(numpy.float32)
    return (values.reshape(shape[0], -1, WEIGHTS_PER_SCALE) * scales[:, :, None]).reshape(shape[0], -1)


def gguf_expert_matrix(gguf_path, tensors, layer, expert, name):
    dimensions, kind, offset = tensors[f"blk.{layer}.ffn_{name}_exps.weight"]
    inputs, outputs, experts = dimensions                                  # innermost first
    block_bytes, decode = {GGUF_TYPE_Q2_K: (Q2_K_BLOCK_BYTES, decode_q2_k),
                           GGUF_TYPE_Q3_K: (Q3_K_BLOCK_BYTES, decode_q3_k)}[kind]
    expert_bytes = outputs * inputs // BLOCK_WEIGHTS * block_bytes
    raw = numpy.fromfile(gguf_path, dtype = numpy.uint8, count = expert_bytes, offset = offset + expert * expert_bytes)
    return decode(raw).reshape(outputs, inputs), {GGUF_TYPE_Q2_K: "Q2_K", GGUF_TYPE_Q3_K: "Q3_K"}[kind]


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf_file")
    parser.add_argument("checkpoint_folder")
    parser.add_argument("--layers", default = "0,1,2,10,20,30,39")
    parser.add_argument("--experts", default = "0,7,200")
    arguments = parser.parse_args()
    tensors = read_gguf_tensors(arguments.gguf_file)
    shards = read_shard_headers(arguments.checkpoint_folder)
    for layer in (int(text) for text in arguments.layers.split(",")):
        for matrix_name, checkpoint_name in MATRICES:
            errors, cosines, kind = [], [], "?"
            for expert in (int(text) for text in arguments.experts.split(",")):
                quantized, kind = gguf_expert_matrix(arguments.gguf_file, tensors, layer, expert, matrix_name)
                original = checkpoint_expert_matrix(shards, layer, expert, checkpoint_name)
                if quantized.shape != original.shape:
                    raise ValueError(f"layer {layer} {matrix_name}: {quantized.shape} against {original.shape}")
                difference = (quantized.astype(numpy.float64) - original).ravel()
                reference = original.astype(numpy.float64).ravel()
                errors.append(float(numpy.sqrt(difference @ difference / (reference @ reference))))
                cosines.append(float(quantized.astype(numpy.float64).ravel() @ reference
                                     / numpy.sqrt((quantized.astype(numpy.float64).ravel() ** 2).sum()
                                                  * (reference @ reference))))
            print(f"layer {layer:2d} {matrix_name:4s} {kind}: relative error "
                  + " ".join(f"{error:.4f}" for error in errors) + "; cosine "
                  + " ".join(f"{cosine:.4f}" for cosine in cosines), flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
