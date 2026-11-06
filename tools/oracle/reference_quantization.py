#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The activation quantizations of DeepSeek's reference as its kernel.py DEFINES them, computed by PyTorch, and the
stand-ins that hold the reference's compiled kernels to those definitions at every call.

Why. reference_divergence.py found that the reference does not repeat itself on an RTX PRO 6000, and
reference_kernel_order_check.py found why: the kernel that tilelang 0.1.8 builds from act_quant_kernel keeps its
input tile (32 rows of 32 bfloat16 values: thread t stores 16 bytes at byte 16 t) and its output tile (32 rows of
32 bytes: thread t stores 8 bytes at byte 8 t) in ONE shared buffer, with no barrier between the two stores. A
warp that stores its input after another warp has stored its output replaces that output by raw input bytes.
Output byte a of the tile lies where input byte a lies, so what the fault leaves is known exactly: the output
byte of (row r, column c) of a 32-by-32 tile becomes byte a % 2 of input value a // 2, with a = 32 r + c. Rows 0
to 7 of a tile lie under input values of their own warp and cannot be hit. (The reverse case, an output store
landing between a thread's input store and its read-back, would change scales and rows 4 to 15; it was never
seen in 1.1 million calls, and nothing in the generated source forbids it.)

So a result that is to be a ground truth cannot come from that kernel. The definitions below follow kernel.py
line by line; the stand-ins run the reference's kernel as well and compare every byte:
  * act_quant returning new tensors (the input of every quantized linear, model.py:187 and :197): answered by the
    definition. A byte of the kernel that is not the definition's must lie in rows 8 to 31 of its tile and be
    exactly the input byte the fault leaves there, and every scale must be the definition's, or the tool stops:
    a difference of any other kind would mean that the definition, not the kernel, is wrong.
  * act_quant in place (the attention's keys, model.py:707, :1042, :1062) and fp4_act_quant in place (the
    indexer's and the compressed keys, model.py:546, :552, :760): the reference's kernel runs on the tensor, and
    the tensor must then equal the definition applied to a copy taken before, bit for bit, or the tool stops.
    Input and output of these kernels are tiles of the same element size, so a thread's output lies where its own
    input lay; whether a build is in fact safe is what this comparison shows at every call.
  * act_quant in place on the attention output, from 46cc8310 on (the backbone's and the draft head's,
    46cc8310:787, :1073): held as the keys are.
Any other way of calling the two functions stops the tool: nothing here defines its answer.
The definitions are for finite values. The generated kernels use CUDA's max and min, which return the other
operand for a NaN, where PyTorch's amax and clamp keep the NaN: on a NaN or an infinity the two are expected to
differ and the tool to stop.
Measurement tool: not part of the engine.
"""
import collections

import torch

LARGEST_E4M3 = 448.0                         # kernel.py:45-46
SMALLEST_E4M3_BLOCK_MAGNITUDE = 1e-4         # kernel.py:76
LARGEST_E2M1 = 6.0                           # kernel.py:131
SMALLEST_E2M1_BLOCK_MAGNITUDE_E4M3_SCALES = 6 * 2.0 ** -9      # kernel.py:162
SMALLEST_E2M1_BLOCK_MAGNITUDE_POWER_OF_TWO_SCALES = 6 * 2.0 ** -126   # kernel.py:165
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)     # by code: 2 exponent bits and 1 mantissa bit
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)       # between each magnitude and the next
FLOAT32_EXPONENT_SHIFT, FLOAT32_EXPONENT_MASK, FLOAT32_EXPONENT_BIAS = 23, 0xFF, 127
TILE_ROWS = 32                               # kernel.py:48, blk_m: the rows one block of the kernel takes
TILE_COLUMNS = 32                            # the block size the known fault was measured with (model.py:27)
TILE_ROWS_NEVER_OVERWRITTEN = 8              # the rows of a tile held by its first warp: 32 threads, 4 per row
INPUT_VALUE_BYTES = 2                        # bfloat16


def same_bits(first, second):
    return first.shape == second.shape and first.dtype == second.dtype and torch.equal(
        first.contiguous().reshape(-1).view(torch.uint8), second.contiguous().reshape(-1).view(torch.uint8))


def in_blocks(values, block_size):
    """[rows, blocks of a row, block size] in float32: the kernels compute in float32 on rows of view(-1, N)."""
    numbers = values.to(torch.float32)
    return numbers.reshape(-1, numbers.shape[-1] // block_size, block_size)


def power_of_two_at_or_above(magnitude, inverse_of_the_largest):
    """fast_round_scale (kernel.py:22-37): the float32 product's exponent, one more unless the product is a power
    of two itself, by its bits."""
    product = magnitude * torch.tensor(inverse_of_the_largest, dtype = torch.float32, device = magnitude.device)
    product_bits = product.view(torch.int32)
    exponent = ((product_bits >> FLOAT32_EXPONENT_SHIFT) & FLOAT32_EXPONENT_MASK) - FLOAT32_EXPONENT_BIAS \
        + (product_bits & ((1 << FLOAT32_EXPONENT_SHIFT) - 1) != 0).to(torch.int32)
    return ((exponent + FLOAT32_EXPONENT_BIAS) << FLOAT32_EXPONENT_SHIFT).view(torch.float32)


def e4m3_with_scales(values, block_size):
    """kernel.py:74-90 with scales rounded to a power of two: per row and block the largest magnitude (at least
    1e-4) times 1/448, rounded up to a power of two; every value divided by it, clamped to 448 and cast to an
    8-bit float. Returns the 8-bit floats in blocks and the scales as float32."""
    blocks = in_blocks(values, block_size)
    scale = power_of_two_at_or_above(blocks.abs().amax(dim = -1).clamp_min(SMALLEST_E4M3_BLOCK_MAGNITUDE),
                                     1.0 / LARGEST_E4M3)
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-LARGEST_E4M3, LARGEST_E4M3).to(torch.float8_e4m3fn)
    return quantized, scale


def defined_act_quant(values, block_size):
    """act_quant returning new tensors: the quantized values and their scales as the kernel stores them."""
    quantized, scale = e4m3_with_scales(values, block_size)
    return quantized.reshape(values.shape), scale.reshape(*values.shape[:-1], -1).to(torch.float8_e8m0fnu)


def defined_act_quant_in_place(values, block_size):
    """act_quant in place (kernel.py:81-86): every value quantized as above and multiplied by its scale again."""
    quantized, scale = e4m3_with_scales(values, block_size)
    return (quantized.to(torch.float32) * scale.unsqueeze(-1)).reshape(values.shape).to(values.dtype)


def nearest_e2m1(values):
    """The nearest 4-bit float (sign, 2 exponent bits, 1 mantissa bit) of float32 values within 6 of zero, a tie
    going to the code that is even; the sign is kept."""
    magnitude = values.abs()
    midpoints = torch.tensor(E2M1_MIDPOINTS, dtype = torch.float32, device = values.device)
    code_below = torch.bucketize(magnitude, midpoints, right = False)      # a tie takes the code below it
    code_above = torch.bucketize(magnitude, midpoints, right = True)       # a tie takes the code above it
    code = torch.where(code_below % 2 == 0, code_below, code_above)        # the two differ at a tie only
    magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype = torch.float32, device = values.device)
    return torch.copysign(magnitudes[code], values)


def defined_fp4_act_quant_in_place(values, block_size, scale_dtype):
    """fp4_act_quant in place (kernel.py:155-172): per row and block a scale, an 8-bit float of the largest
    magnitude over 6 (compressed keys) or that quotient rounded up to a power of two (the indexer); every value
    divided by it, clamped to 6, rounded to a 4-bit float and multiplied by the scale again."""
    blocks = in_blocks(values, block_size)
    magnitude = blocks.abs().amax(dim = -1)
    if scale_dtype == torch.float8_e4m3fn:
        scale = (magnitude.clamp_min(SMALLEST_E2M1_BLOCK_MAGNITUDE_E4M3_SCALES) / LARGEST_E2M1).to(
            torch.float8_e4m3fn).to(torch.float32)
    else:
        scale = power_of_two_at_or_above(magnitude.clamp_min(SMALLEST_E2M1_BLOCK_MAGNITUDE_POWER_OF_TWO_SCALES),
                                         1.0 / LARGEST_E2M1)
    quantized = nearest_e2m1((blocks / scale.unsqueeze(-1)).clamp(-LARGEST_E2M1, LARGEST_E2M1))
    return (quantized * scale.unsqueeze(-1)).reshape(values.shape).to(values.dtype)


def input_bytes_the_fault_leaves(values, positions):
    """For positions [count, 2] of (row, column) in the quantized output of act_quant on bfloat16 values: the
    input byte that lies at each output byte's place in the kernel's shared tile (the module text derives it)."""
    rows, columns = positions[:, 0], positions[:, 1]
    address = rows % TILE_ROWS * TILE_COLUMNS + columns % TILE_COLUMNS     # the byte's place in its tile
    value_in_tile = address // INPUT_VALUE_BYTES
    source_rows = rows - rows % TILE_ROWS + value_in_tile // TILE_COLUMNS
    source_columns = columns - columns % TILE_COLUMNS + value_in_tile % TILE_COLUMNS
    input_bytes = values.contiguous().reshape(-1, values.shape[-1]).view(torch.uint8)
    return input_bytes[source_rows, source_columns * INPUT_VALUE_BYTES + address % INPUT_VALUE_BYTES]


def unexplained_bytes(values, kernel_quantized, defined_quantized, block_size):
    """Compares the kernel's quantized bytes with the definition's. Returns how many differ and how many of those
    are NOT what the known fault leaves (another row of the tile, another byte, or another block size)."""
    width = values.shape[-1]
    kernel_bytes = kernel_quantized.reshape(-1, width).view(torch.uint8)
    positions = (kernel_bytes != defined_quantized.reshape(-1, width).view(torch.uint8)).nonzero()
    if len(positions) == 0:
        return 0, 0
    if block_size != TILE_COLUMNS or values.dtype != torch.bfloat16:
        return len(positions), len(positions)
    explained = (positions[:, 0] % TILE_ROWS >= TILE_ROWS_NEVER_OVERWRITTEN) & (
        kernel_bytes[positions[:, 0], positions[:, 1]] == input_bytes_the_fault_leaves(values, positions))
    return len(positions), len(positions) - int(explained.sum())


def checked_act_quant(reference_act_quant, counts):
    """What stands where model.py calls act_quant (the module text says what each kind of call is held to)."""
    def act_quant(x, block_size = 128, scale_fmt = None, scale_dtype = torch.float32, inplace = False):
        if scale_fmt is None or scale_dtype != torch.float8_e8m0fnu:
            raise RuntimeError(f"act_quant was called with scale_fmt {scale_fmt!r} and scales of {scale_dtype}: only "
                               "scales rounded to a power of two and stored as float8_e8m0fnu have a definition "
                               "here (model.py:29-30)")
        if inplace:
            before = x.clone()
            result = reference_act_quant(x, block_size, scale_fmt, scale_dtype, True)
            if not same_bits(x, defined_act_quant_in_place(before, block_size)):
                raise RuntimeError(f"act_quant in place on a tensor of shape {tuple(x.shape)}: the reference's "
                                   "kernel did not leave what kernel.py:81-86 defines")
            counts.update(in_place_calls = 1, in_place_values = x.numel())
            return result
        kernel_quantized, kernel_scales = reference_act_quant(x, block_size, scale_fmt, scale_dtype)
        quantized, scales = defined_act_quant(x, block_size)
        wrong, unexplained = unexplained_bytes(x, kernel_quantized, quantized, block_size)
        if unexplained or not same_bits(kernel_scales, scales):
            raise RuntimeError(f"act_quant on an input of shape {tuple(x.shape)}: {unexplained} of the {wrong} bytes "
                               "in which the reference's kernel and its definition differ are not the input bytes "
                               "the kernel's known fault leaves in rows 8 to 31 of a tile, or a scale differs: "
                               "the definition here, or the account of the fault, is wrong")
        counts.update(calls = 1, calls_the_kernel_got_wrong = int(wrong > 0), bytes = quantized.numel(),
                      bytes_the_kernel_got_wrong = wrong)
        return quantized, scales
    return act_quant


def checked_fp4_act_quant(reference_fp4_act_quant, counts):
    """What stands where model.py calls fp4_act_quant: its three calls are all in place."""
    def fp4_act_quant(x, block_size = 32, inplace = False, scale_dtype = torch.float8_e8m0fnu):
        if not inplace or scale_dtype not in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
            raise RuntimeError(f"fp4_act_quant was called with inplace {inplace} and scales of {scale_dtype}: only "
                               "its in-place form has a definition here (model.py:546, :552, :760)")
        before = x.clone()
        result = reference_fp4_act_quant(x, block_size, True, scale_dtype)
        if not same_bits(x, defined_fp4_act_quant_in_place(before, block_size, scale_dtype)):
            raise RuntimeError(f"fp4_act_quant in place on a tensor of shape {tuple(x.shape)} with scales of "
                               f"{scale_dtype}: the reference's kernel did not leave what kernel.py:155-172 defines")
        counts.update(four_bit_in_place_calls = 1, four_bit_in_place_values = x.numel())
        return result
    return fp4_act_quant


def install(reference_model):
    """Puts the stand-ins where model.py looks its two quantization functions up. Returns their counters."""
    counts = collections.Counter()
    reference_model.act_quant = checked_act_quant(reference_model.act_quant, counts)
    reference_model.fp4_act_quant = checked_fp4_act_quant(reference_model.fp4_act_quant, counts)
    return counts
