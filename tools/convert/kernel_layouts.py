# SPDX-License-Identifier: AGPL-3.0-or-later
"""The matrices of a weight format in an Anemone pack: the two layouts the kernels read, and what make_pack.py,
beside this file, does about them.

It is not called on its own: make_pack.py's convert, verify and self-test call it, and
tools/golden/check_pack_linears.py reads the matrices of a pack back through it. The formats are
include/anemone/weight_format_seam.h, whose two LANE ORDER paragraphs are the specification of every index here; the
rules of a tensor record are include/anemone/pack.h's.

The layouts. A checkpoint stores a quantized weight as two tensors, "<name>.weight" and "<name>.scale":
  - 4 bits, E2M1_SCALED: I8 [outputs, inputs / 2], two codes to a byte, and F8_E8M0 [outputs, inputs / 32]. The code
    of input 2 j is the LOW four bits of byte j and the code of input 2 j + 1 the high four. The reference decides
    that where it widens these very bytes to 8-bit weights without loss (its convert.py:32-34: low = x & 0x0F,
    high = (x >> 4) & 0x0F, and torch.stack([low, high], dim=-1) puts low first). Where it runs the 4-bit bytes as
    they are (convert.py:181 views them as float4_e2m1fn_x2; kernel.py:512 and :537-540 read them as a tensor of
    FP4), which half of a byte is which element is left to the compiler of its kernels: check_pack_linears.py is
    what proves it, on the reference's own products.
  - 8 bits, E4M3_SCALED: F8_E4M3 [outputs, inputs] and F8_E8M0 [ceil(outputs / 32), inputs / 32].
store_matrix makes the stored bytes of a format from the bytes of the two tensors; read_matrix makes the codes
[outputs, inputs] and the scales in the checkpoint's arrangement from stored bytes. The two are written apart, each
from the header: store_matrix asks of every stored byte which codes it holds, read_matrix asks of every code where
it is stored, and neither uses an index of the other. A layout is the same from one tile of 16 outputs to the next,
so an index map covers ONE tile; it is built once for an input size (40,960 entries for the 5,120 inputs of an
expert's first matrix), checked to be a rearrangement, and applied to every tile of a matrix by one numpy take. No
weight is visited in Python.

convert (with_matrices, read_tensor). A tensor of a unit is a quantized weight when its element type is F8_E4M3,
or I8 under a name that ends in ".weight"; an I8 tensor of another name is plain integers. A quantized weight must
be "<name>.weight" with two dimensions, of a size its format stores (outputs a multiple of 16; inputs a multiple of
64 at 4 bits, of 32 at 8 bits), with "<name>.scale" in its unit as F8_E8M0 of the shape above. It becomes ONE tensor
of the pack, "<name>.weight": UINT8 [outputs, inputs] with its weight format and its bits per weight in its record,
its bytes the format's layout; its scales are inside it and "<name>.scale" is no tensor of the pack. Whatever else
holds F8_E4M3 or F8_E8M0 elements (codes without scales, scales that no weight takes, another name, another
shape) stops the run with one line that names the tensor: nothing is left out or stored plain silently. The worker
processes lay a matrix out, and refuse a weight code that is no number or a scale code that is no power of two
(below). convert prints one line with the count of the matrices of each format.

verify (broken_matrix_rule, checked_ranges, forbidden_value_in). The rules of pack.h for the record of a matrix,
each with its own message: a format this build carries, two dimensions, elements of UINT8, the format's bits per
weight, a size the format stores, exactly the format's bytes for that size. They are written from the header a
second time and share no table with convert. Then, while the worker processes digest an extent, the one thing a
matrix's own bytes can show without its source: no E4M3 weight has all seven magnitude bits set (0x7F, 0xFF) and
no scale is 0xFF. Those are the codes of E4M3 and of E8M0 that are no number. The header gives every code a value
by its bits alone, and the kernels place bits, so such a code would be computed with: as 480 or as 2^128, where the
reference computes with no number. The 16 codes of E2M1 are all numbers, so the words of a 4-bit matrix are not
looked at. That the codes are the checkpoint's is not verify's to show: the digests keep the bytes convert wrote,
and check_pack_linears.py compares them with the reference.

self-test. The matrices' part of make_pack.py's self-test, on its toy model, whose quantized weights are the
smallest matrix of each format and one of 8 bits with a whole scale block and a half one, and whose scale tensors
each begin with the scale 2^0 (0x7F: a scale like any other, and no weight code). Shown: the byte counts of the
real model's shapes as counted by hand; the sizes no format stores; codes and scales at places counted by hand from
the header's own numbers; store_matrix against literal_stored_bytes, a third writing of the layouts one code at a
time in plain Python, and read_matrix giving the checkpoint's codes and scales back, on five shapes of several
tiles, loads and scale blocks, and both refusing a caller's mistake; the worker's search of an extent cut into
pieces of every size; in the toy pack the record and the bytes of every tensor and the layout of every extent, each
against a second writing (expected_in_pack), and every matrix read back through read_matrix; convert's line; what
convert must refuse of a folder, and of a file whose header gives a tensor other bytes than its shape; and every
rule of verify above failing once on a changed copy of the pack (the sizes one by one, both codes that are no
number, the first weight and the last), with the format version 1 that is no longer read and the reserved word of
the new record.
"""
import collections
import contextlib
import functools
import hashlib
import io
import itertools
import math
import os
import shutil

import numpy

import make_pack                                           # used inside functions only: the two files import each other

# ---- The formats: include/anemone/weight_format_seam.h ---------------------------------------------------------------
WEIGHT_FORMAT_E2M1_SCALED = 3                              # enum anemone_weight_format, by value
WEIGHT_FORMAT_E4M3_SCALED = 5
# The letters of the header: lane = 4 g + t; r and h are the register and the half of the weight operand a nibble
# of a word is placed in, b is a byte of a lane.
OUTPUT_HALVES = 2                                          # r % 2, b % 2: a lane holds outputs g and g + 8 of its tile
LANE_GROUPS = 8                                            # g
LANE_POSITIONS = 4                                         # t
LANES = LANE_GROUPS * LANE_POSITIONS                       # 32
TILE_OUTPUTS = OUTPUT_HALVES * LANE_GROUPS                 # 16
INPUT_PARITIES = 2                                         # h, (b / 2) % 2: the inputs 2 t and 2 t + 1
REGISTER_INPUTS = LANE_POSITIONS * INPUT_PARITIES          # 8: the inputs one r / 2, or one b / 4, covers
INPUTS_PER_SCALE = 32
E2M1_REGISTER_PAIRS = 2                                    # r / 2
E2M1_REGISTERS = OUTPUT_HALVES * E2M1_REGISTER_PAIRS       # 4: r
E2M1_STEP_INPUTS = E2M1_REGISTER_PAIRS * REGISTER_INPUTS   # 16: the inputs of one word of a lane
E2M1_STEPS = 4                                             # the words of a lane in one load
E2M1_LOAD_INPUTS = E2M1_STEPS * E2M1_STEP_INPUTS           # 64
E2M1_SCALES_PER_LOAD = E2M1_LOAD_INPUTS // INPUTS_PER_SCALE                    # 2: q
E2M1_CODES_PER_BYTE = 2
E2M1_WORD_BYTES = 4                                        # of a word of codes, and of a word of scales
E2M1_WORD_NIBBLES = E2M1_WORD_BYTES * E2M1_CODES_PER_BYTE  # 8
E4M3_INPUT_QUARTERS = 4                                    # b / 4: the inputs 8 (b / 4) .. + 7 of a scale group
E4M3_QUARTER_BYTES = INPUT_PARITIES * OUTPUT_HALVES        # 4: the bytes of a lane with one b / 4
E4M3_LANE_BYTES = E4M3_INPUT_QUARTERS * E4M3_QUARTER_BYTES                     # 16
NIBBLE_BITS = 4
LOW_NIBBLE = 0x0F
HIGH_NIBBLE = 0xF0
BYTE_VALUES = 256
E4M3_NOT_A_NUMBER_BITS = 0x7F                              # all seven magnitude bits: the two codes that are no number
E8M0_NOT_A_NUMBER = 0xFF                                   # the E8M0 code that is no power of two
# What convert knows of a format: its name, its one width, what must divide the inputs of a matrix (TILE_OUTPUTS
# divides the outputs of both), the safetensors dtype a checkpoint stores its codes in, the codes a byte of the
# checkpoint holds, the outputs that share a scale, and the bits that are all set in no code (None: every code is a
# number).
Format = collections.namedtuple("Format", ("name", "bits_per_weight", "input_multiple", "codes_dtype",
                                           "codes_per_byte", "outputs_per_scale", "forbidden_code_bits"))
FORMATS = {
    WEIGHT_FORMAT_E2M1_SCALED: Format("E2M1_SCALED", 4, E2M1_LOAD_INPUTS, "I8", E2M1_CODES_PER_BYTE, 1, None),
    WEIGHT_FORMAT_E4M3_SCALED: Format("E4M3_SCALED", 8, INPUTS_PER_SCALE, "F8_E4M3", 1, 32, E4M3_NOT_A_NUMBER_BITS),
}

# ---- A quantized weight as a checkpoint stores it, and as its pack does ----------------------------------------------
WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".scale"
SCALES_DTYPE = "F8_E8M0"
INTEGER_DTYPE = "I8"                                       # 4-bit codes under "<name>.weight", plain integers elsewhere
MATRIX_DTYPE = "U8"                                        # the element type in the record of a matrix
# What stands in a SourceTensor's path for a matrix, and so first in a piece of make_pack.lay_out: the two stored
# tensors (a SourceTensor each) its bytes are laid out from.
MatrixSource = collections.namedtuple("MatrixSource", ("name", "weight_format", "output_size", "input_size",
                                                       "weights", "scales"))

# ---- verify: weight_format_seam.h a second time, on purpose; nothing below this line is used by convert --------------
# enum anemone_weight_format of the formats this build carries: (name, bits per weight, what divides the outputs,
# what divides the inputs, codes to a stored byte, outputs to a scale, the bits all set in no code or None).
VerifiedForm = collections.namedtuple("VerifiedForm", ("name", "bits_per_weight", "output_multiple", "input_multiple",
                                                       "codes_per_byte", "outputs_per_scale", "forbidden_code_bits"))
VERIFIED_FORMS = {3: VerifiedForm("E2M1_SCALED", 4, 16, 64, 2, 1, None),
                  5: VerifiedForm("E4M3_SCALED", 8, 16, 32, 1, 32, 0x7F)}
VERIFIED_INPUTS_PER_SCALE = 32
VERIFIED_FORBIDDEN_SCALE_BITS = 0xFF
MATRIX_DIMENSIONS = 2                                      # outputs, then inputs

# ---- The self-test ---------------------------------------------------------------------------------------------------
# (weight format, outputs, inputs): the smallest of each format, and shapes of several tiles, loads and scale groups;
# the last two of 8 bits end in a scale block of 16 outputs.
SELF_TEST_SHAPES = ((WEIGHT_FORMAT_E2M1_SCALED, 16, 64), (WEIGHT_FORMAT_E2M1_SCALED, 48, 192),
                    (WEIGHT_FORMAT_E4M3_SCALED, 16, 32), (WEIGHT_FORMAT_E4M3_SCALED, 48, 64),
                    (WEIGHT_FORMAT_E4M3_SCALED, 80, 96))
# (weight format, outputs, inputs, stored bytes): shapes of the model and of the toy, their bytes counted by hand
# from the two stored_byte_count sentences of the header.
SELF_TEST_BYTE_COUNTS = (
    (WEIGHT_FORMAT_E2M1_SCALED, 2304, 5120, 6266880), (WEIGHT_FORMAT_E2M1_SCALED, 5120, 2304, 6266880),
    (WEIGHT_FORMAT_E2M1_SCALED, 16, 64, 544), (WEIGHT_FORMAT_E4M3_SCALED, 512, 5120, 2624000),
    (WEIGHT_FORMAT_E4M3_SCALED, 32768, 1280, 41984000), (WEIGHT_FORMAT_E4M3_SCALED, 16, 32, 513),
    (WEIGHT_FORMAT_E4M3_SCALED, 48, 64, 3076))
# Places taken by hand from the two LANE ORDER paragraphs of the header, in the header's own numbers: of a matrix of
# (weight format, outputs, inputs), codes as (stored byte, 0 for its low four bits or 1 for its high four or None for
# the whole byte, output, input) and scales as (byte behind the codes, output or block of 32 outputs, scale group).
# Most lie where one digit of a formula turns over; W[9][70] and W[20][45], and W[9][38] and W[36][21], lie inside a
# tile, with every digit at work: e.g. W[20][45] at 4 bits is tile 1, g 4, r % 2 0; load 0, step 2, r / 2 1, t 2, h 1:
# word ((1 * 2 + 0) * 32 + 18) * 4 + 2 = 330 at bit 4 * 2 + 16 = 24, the low four bits of byte 4 * 330 + 3.
SELF_TEST_PLACES = (
    (WEIGHT_FORMAT_E2M1_SCALED, 32, 128,
     ((0, 0, 0, 0), (0, 1, 8, 0), (1, 0, 0, 8), (1, 1, 8, 8), (2, 0, 0, 1), (3, 1, 8, 9), (4, 0, 0, 16), (16, 0, 0, 2),
      (64, 0, 1, 0), (511, 1, 15, 63), (512, 0, 0, 64), (624, 1, 9, 70), (1024, 0, 16, 0), (1323, 0, 20, 45),
      (2047, 1, 31, 127)),
     ((0, 0, 0), (1, 8, 0), (2, 0, 1), (3, 8, 1), (4, 1, 0), (32, 0, 2), (37, 9, 2), (64, 16, 0), (82, 20, 1),
      (127, 31, 3))),
    (WEIGHT_FORMAT_E4M3_SCALED, 48, 64,
     ((0, None, 0, 0), (1, None, 8, 0), (2, None, 0, 1), (3, None, 8, 1), (4, None, 0, 8), (15, None, 8, 25),
      (16, None, 0, 2), (64, None, 1, 0), (511, None, 15, 31), (512, None, 0, 32), (625, None, 9, 38),
      (1024, None, 16, 0), (2346, None, 36, 21), (3071, None, 47, 63)),
     ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1))))
SELF_TEST_UNCARRIED_FORMAT = 1                             # ANEMONE_WEIGHT_FORMAT_FLOAT16: a format, and not carried
SELF_TEST_NEGATIVE_NOT_A_NUMBER = 0xFF                     # E4M3: the code that is no number with its sign bit set
SELF_TEST_EXTENT_BYTES = 10                                # the toy extent forbidden_value_in is cut up in every way
SELF_TEST_RANGE = (3, 6)                                   # its checked range: the first byte and the end
SELF_TEST_STRAY_NAME = "head.stray"                        # the quantized weight of the folders convert must refuse
SELF_TEST_OTHER_SCALE_SHAPE = (2, 1)                       # no shape of the scales of the smallest matrix of 8 bits


# ---- The formats -----------------------------------------------------------------------------------------------------
def shape_refusal(weight_format, output_size, input_size):
    """Why a format stores no matrix of a size, in a few words, or None for a size it stores: the sizes that
    stored_byte_count of weight_format_seam.h refuses, and those the 32 bits of a record's dimension do not hold."""
    for size, multiple, what in ((output_size, TILE_OUTPUTS, "outputs"),
                                 (input_size, FORMATS[weight_format].input_multiple, "inputs")):
        if not 0 < size <= make_pack.UINT32_MAXIMUM:
            return f"{size} {what} are none, or more than 32 bits count"
        if size % multiple:
            return f"{size} {what} are no multiple of {multiple}"
    return None


def checkpoint_shapes(weight_format, output_size, input_size):
    """The shapes a checkpoint stores a matrix of a format in: of its codes, then of its scales."""
    form = FORMATS[weight_format]
    return ((output_size, input_size // form.codes_per_byte),
            (-(-output_size // form.outputs_per_scale), input_size // INPUTS_PER_SCALE))


def stored_byte_count(weight_format, output_size, input_size):
    """stored_byte_count of weight_format_seam.h for a size the format stores: the codes, and the scales behind
    them."""
    weight_shape, scale_shape = checkpoint_shapes(weight_format, output_size, input_size)
    return math.prod(weight_shape) + math.prod(scale_shape)


def format_of(source):
    """The weight format and the bits per weight in the record of a tensor of the layout: of a matrix its format's,
    of a plain tensor ANEMONE_WEIGHT_FORMAT_NONE and 0."""
    if not isinstance(source.path, MatrixSource):
        return make_pack.WEIGHT_FORMAT_NONE, 0
    return source.path.weight_format, FORMATS[source.path.weight_format].bits_per_weight


# ---- The layouts: from a checkpoint's bytes to stored bytes ----------------------------------------------------------
def digits_of(numbers, *radices):
    """numbers as the digits of a mixed radix, the most significant first: what no radix bounds, then one digit for
    each radix."""
    digits = []
    for radix in reversed(radices):
        numbers, digit = numpy.divmod(numbers, radix)
        digits.append(digit)
    return (numbers, *reversed(digits))


def number_of(first_digit, *digits_with_radices):
    """The number of a mixed radix from its digits, the most significant first; each digit but the first comes with
    its radix."""
    number = first_digit
    for digit, radix in digits_with_radices:
        number = number * radix + digit
    return number


def tile_map(indices, element_count):
    """An index map of one tile, ready for numpy.take: checked to name every one of element_count places once."""
    indices = numpy.ascontiguousarray(indices, dtype = numpy.intp)
    if indices.shape != (element_count,) or not numpy.array_equal(numpy.sort(indices), numpy.arange(element_count)):
        raise RuntimeError("an index map of a weight format is no rearrangement of its tile")
    indices.flags.writeable = False
    return indices


@functools.lru_cache(maxsize = None)
def e2m1_stored_from(input_size):
    """store_matrix's two maps of one tile of E2M1_SCALED.
    Codes: for every byte of the tile's words in stored order, [load][lane][step][byte of the word], where its two
    codes lie in the tile's codes arranged as [input parity][lane group][byte of the checkpoint's row], an
    arrangement in which a byte holds the codes of ONE input for the outputs g (low four bits) and g + 8. Byte
    2 h + r / 2 of a word holds the nibbles at bits 4 r + 16 h with r % 2 = 0 and 1: the input 64 load + 16 step
    + 8 (r / 2) + 2 t + h, which is half h of the checkpoint's byte 32 load + 8 step + 4 (r / 2) + t.
    Scales: for every scale byte in stored order, [load][lane group][byte of the word], where it lies in the
    checkpoint's [output][scale group]: byte 2 q + r is output g + 8 r and scale group 2 load + q."""
    byte_columns, groups = input_size // E2M1_CODES_PER_BYTE, input_size // INPUTS_PER_SCALE
    load, lane_group, lane_position, step, input_parity, register_pair = digits_of(
        numpy.arange(TILE_OUTPUTS * byte_columns), LANE_GROUPS, LANE_POSITIONS, E2M1_STEPS, INPUT_PARITIES,
        E2M1_REGISTER_PAIRS)
    byte_column = number_of(load, (step, E2M1_STEPS), (register_pair, E2M1_REGISTER_PAIRS),
                            (lane_position, LANE_POSITIONS))
    codes_from = number_of(input_parity, (lane_group, LANE_GROUPS), (byte_column, byte_columns))
    load, lane_group, scale_in_load, output_half = digits_of(
        numpy.arange(TILE_OUTPUTS * groups), LANE_GROUPS, E2M1_SCALES_PER_LOAD, OUTPUT_HALVES)
    scale_group = number_of(load, (scale_in_load, E2M1_SCALES_PER_LOAD))
    scales_from = number_of(output_half, (lane_group, LANE_GROUPS), (scale_group, groups))
    return tile_map(codes_from, TILE_OUTPUTS * byte_columns), tile_map(scales_from, TILE_OUTPUTS * groups)


@functools.lru_cache(maxsize = None)
def e4m3_stored_from(input_size):
    """store_matrix's map of one tile of E4M3_SCALED: for every stored byte, [scale group][lane][byte of the lane],
    where it lies in the tile's bytes of the checkpoint, [output][input]. Byte b of lane 4 g + t is output
    g + 8 (b % 2) and input 32 group + 8 (b / 4) + 2 t + (b / 2) % 2."""
    group, lane_group, lane_position, input_quarter, input_parity, output_half = digits_of(
        numpy.arange(TILE_OUTPUTS * input_size), LANE_GROUPS, LANE_POSITIONS, E4M3_INPUT_QUARTERS, INPUT_PARITIES,
        OUTPUT_HALVES)
    column = number_of(group, (input_quarter, E4M3_INPUT_QUARTERS), (lane_position, LANE_POSITIONS),
                       (input_parity, INPUT_PARITIES))
    return tile_map(number_of(output_half, (lane_group, LANE_GROUPS), (column, input_size)),
                    TILE_OUTPUTS * input_size)


def matrix_parts(weight_format, output_size, input_size, parts, what):
    """The byte buffers of a matrix as numpy arrays of uint8, after its size and theirs have been checked: a caller
    that passes another size made a mistake, which is a ValueError."""
    arrays = [numpy.frombuffer(part, dtype = numpy.uint8) for part in parts]
    refusal = shape_refusal(weight_format, output_size, input_size)
    byte_counts = [math.prod(shape) for shape in checkpoint_shapes(weight_format, output_size, input_size)]
    if refusal or [array.size for array in arrays] not in (byte_counts, [sum(byte_counts)]):
        raise ValueError(f"{what}: {refusal or 'the bytes are not those of a matrix'}: "
                         f"{FORMATS[weight_format].name}, {output_size} outputs by {input_size} inputs")
    return arrays


def store_matrix(weight_format, output_size, input_size, weights, scales):
    """The stored bytes of one matrix, a numpy array of uint8: its codes in lane order, then its scales. weights
    and scales are the bytes of the checkpoint's two tensors, in any buffer."""
    weights, scales = matrix_parts(weight_format, output_size, input_size, (weights, scales), "store_matrix")
    tiles = output_size // TILE_OUTPUTS
    if weight_format == WEIGHT_FORMAT_E4M3_SCALED:
        codes = weights.reshape(tiles, TILE_OUTPUTS * input_size).take(e4m3_stored_from(input_size), axis = 1)
        return numpy.concatenate((codes.reshape(-1), scales))  # a scale block [block][group]: as the checkpoint's
    codes_from, scales_from = e2m1_stored_from(input_size)
    halves = weights.reshape(tiles, OUTPUT_HALVES, LANE_GROUPS, input_size // E2M1_CODES_PER_BYTE)
    first_outputs, second_outputs = halves[:, 0], halves[:, 1]     # outputs g and g + 8 of every tile
    paired = numpy.empty_like(halves)                          # [tile][input parity][lane group][byte of the row]
    paired[:, 0] = (first_outputs & LOW_NIBBLE) | ((second_outputs & LOW_NIBBLE) << NIBBLE_BITS)
    paired[:, 1] = (first_outputs >> NIBBLE_BITS) | (second_outputs & HIGH_NIBBLE)
    codes = paired.reshape(tiles, -1).take(codes_from, axis = 1)
    return numpy.concatenate((codes.reshape(-1), scales.reshape(tiles, -1).take(scales_from, axis = 1).reshape(-1)))


# ---- The layouts: from stored bytes back to a checkpoint's codes -----------------------------------------------------
@functools.lru_cache(maxsize = None)
def e2m1_places_of(input_size):
    """read_matrix's two maps of one tile of E2M1_SCALED.
    Codes: for every code of the tile, [output i of 16][input c], its place among the nibbles of the tile's words,
    where nibble n is the low four bits of byte n / 2 for an even n and its high four for an odd one, so that the
    nibble at bit 4 r + 16 h of word w is nibble 8 w + r + 4 h. From i = g + 8 (r % 2) and c = 64 load + 16 step
    + 8 (r / 2) + 2 t + h.
    Scales: for every scale of the tile, [output i][scale group k], its place among the tile's scale bytes: byte
    2 q + r of word [load][g], from i = g + 8 r and k = 2 load + q."""
    groups = input_size // INPUTS_PER_SCALE
    output, column = numpy.divmod(numpy.arange(TILE_OUTPUTS * input_size), input_size)
    lane_group, output_half = output % LANE_GROUPS, output // LANE_GROUPS
    load, step = column // E2M1_LOAD_INPUTS, column % E2M1_LOAD_INPUTS // E2M1_STEP_INPUTS
    register_pair = column % E2M1_STEP_INPUTS // REGISTER_INPUTS
    lane_position, input_parity = column % REGISTER_INPUTS // INPUT_PARITIES, column % INPUT_PARITIES
    word = ((load * LANE_GROUPS + lane_group) * LANE_POSITIONS + lane_position) * E2M1_STEPS + step
    operand_register = output_half + OUTPUT_HALVES * register_pair
    code_places = word * E2M1_WORD_NIBBLES + operand_register + E2M1_REGISTERS * input_parity
    output, group = numpy.divmod(numpy.arange(TILE_OUTPUTS * groups), groups)
    scale_word = group // E2M1_SCALES_PER_LOAD * LANE_GROUPS + output % LANE_GROUPS
    scale_places = (scale_word * E2M1_WORD_BYTES + OUTPUT_HALVES * (group % E2M1_SCALES_PER_LOAD)
                    + output // LANE_GROUPS)
    return tile_map(code_places, TILE_OUTPUTS * input_size), tile_map(scale_places, TILE_OUTPUTS * groups)


@functools.lru_cache(maxsize = None)
def e4m3_places_of(input_size):
    """read_matrix's map of one tile of E4M3_SCALED: for every code of the tile, [output i of 16][input c], its
    place among the tile's stored bytes. From i = g + 8 (b % 2) and c = 32 group + 8 (b / 4) + 2 t + (b / 2) % 2:
    lane 4 g + t of the group, and byte b = 4 (b / 4) + 2 ((b / 2) % 2) + b % 2 of the lane."""
    output, column = numpy.divmod(numpy.arange(TILE_OUTPUTS * input_size), input_size)
    group, in_group = column // INPUTS_PER_SCALE, column % INPUTS_PER_SCALE
    lane = output % LANE_GROUPS * LANE_POSITIONS + in_group % REGISTER_INPUTS // INPUT_PARITIES
    byte_in_lane = ((in_group // REGISTER_INPUTS * INPUT_PARITIES + in_group % INPUT_PARITIES) * OUTPUT_HALVES
                    + output // LANE_GROUPS)
    return tile_map((group * LANES + lane) * E4M3_LANE_BYTES + byte_in_lane, TILE_OUTPUTS * input_size)


def read_matrix(weight_format, output_size, input_size, stored):
    """The inverse of store_matrix, written from the header on its own: the codes of a matrix, uint8 [outputs,
    inputs] with one code to a byte, and its scales, uint8 in the checkpoint's arrangement ([outputs, inputs / 32]
    at 4 bits, [ceil(outputs / 32), inputs / 32] at 8 bits), from its stored bytes in any buffer."""
    (stored,) = matrix_parts(weight_format, output_size, input_size, (stored,), "read_matrix")
    tiles, groups = output_size // TILE_OUTPUTS, input_size // INPUTS_PER_SCALE
    if weight_format == WEIGHT_FORMAT_E4M3_SCALED:
        code_byte_count = output_size * input_size             # the bytes of the whole matrix lie first
        codes = stored[:code_byte_count].reshape(tiles, -1).take(e4m3_places_of(input_size), axis = 1)
        return codes.reshape(output_size, input_size), stored[code_byte_count:].reshape(-1, groups)
    code_places, scale_places = e2m1_places_of(input_size)
    code_byte_count = output_size * input_size // E2M1_CODES_PER_BYTE              # the words of the whole matrix
    nibbles = numpy.empty((code_byte_count, E2M1_CODES_PER_BYTE), dtype = numpy.uint8)
    nibbles[:, 0] = stored[:code_byte_count] & LOW_NIBBLE
    nibbles[:, 1] = stored[:code_byte_count] >> NIBBLE_BITS
    codes = nibbles.reshape(tiles, -1).take(code_places, axis = 1)
    scales = stored[code_byte_count:].reshape(tiles, -1).take(scale_places, axis = 1)
    return codes.reshape(output_size, input_size), scales.reshape(output_size, groups)


# ---- convert ---------------------------------------------------------------------------------------------------------
def with_matrices(units, sources):
    """convert's step between the model table and the n-gram tables: the rules of the text at the top of this file.
    Returns the units without the scales that went into a matrix, the sources with every quantized weight replaced
    by its matrix (a MatrixSource in place of its path, the format's byte count, UINT8 [outputs, inputs]), and the
    line of convert's summary. The first tensor that breaks a rule is a PackError that names it."""
    units, sources, counts = list(units), dict(sources), collections.Counter()
    formats_by_dtype = {form.codes_dtype: weight_format for weight_format, form in FORMATS.items()}
    for unit_index, unit in enumerate(units):
        names_of_unit, scale_names = set(unit.names), set()
        for name in unit.names:
            source, weight_format = sources[name], formats_by_dtype.get(sources[name].dtype)
            if weight_format is None or (source.dtype == INTEGER_DTYPE and not name.endswith(WEIGHT_SUFFIX)):
                continue                                       # plain
            form, where = FORMATS[weight_format], f"{source.path}: tensor {name!r}"
            make_pack.element_type_of(name, source)
            if not name.endswith(WEIGHT_SUFFIX):
                raise make_pack.PackError(f"{where}: {source.dtype} elements are the codes of a quantized weight, "
                                          f"and its name does not end in {WEIGHT_SUFFIX}")
            if len(source.shape) != MATRIX_DIMENSIONS:
                raise make_pack.PackError(f"{where}: a quantized weight of {len(source.shape)} dimensions is no "
                                          "matrix [outputs, inputs]")
            output_size, input_size = source.shape[0], source.shape[1] * form.codes_per_byte
            refusal = shape_refusal(weight_format, output_size, input_size)
            if refusal:
                raise make_pack.PackError(f"{where}: {form.name} stores no matrix of {output_size} outputs by "
                                          f"{input_size} inputs: {refusal}")
            scale_name = name[:-len(WEIGHT_SUFFIX)] + SCALE_SUFFIX
            if scale_name not in names_of_unit:
                raise make_pack.PackError(f"{where}: a quantized weight, and its unit holds no scales {scale_name!r}")
            scales, (_, scale_shape) = sources[scale_name], checkpoint_shapes(weight_format, output_size, input_size)
            make_pack.element_type_of(scale_name, scales)
            if (scales.dtype, scales.shape) != (SCALES_DTYPE, scale_shape):
                raise make_pack.PackError(f"{where}: its scales {scale_name!r} are {scales.dtype} "
                                          f"{list(scales.shape)}, {form.name} takes {SCALES_DTYPE} {list(scale_shape)}")
            matrix = MatrixSource(name, weight_format, output_size, input_size, source, scales)
            sources[name] = make_pack.SourceTensor(matrix, 0, stored_byte_count(weight_format, output_size, input_size),
                                                   MATRIX_DTYPE, (output_size, input_size))
            scale_names.add(scale_name)
            counts[weight_format] += 1
        for name in unit.names:
            if sources[name].dtype == SCALES_DTYPE and name not in scale_names:
                raise make_pack.PackError(f"{sources[name].path}: tensor {name!r}: {SCALES_DTYPE} elements are "
                                          "scales, and no quantized weight of its unit takes them")
        units[unit_index] = unit._replace(names = [name for name in unit.names if name not in scale_names])
    counted = ", ".join(f"{counts[weight_format]} {form.name}" for weight_format, form in FORMATS.items())
    return units, sources, (f"  matrices of a weight format: {counted}; the scales of each lie inside it: "
                            f"{sum(counts.values())} tensors of the folder are no tensors of the pack")


def read_tensor(source, source_byte_offset, source_files, destination):
    """Worker process: fills destination, a memoryview of the bytes one tensor has in its extent, from the folder. A
    plain tensor (source: the path of its file) is read as it is stored; a matrix (source: a MatrixSource) is laid
    out from the bytes of its two tensors, which may hold no code its format gives no value. source_files, path:
    open file, gains the files this call opens; the caller closes them."""
    def read(path, byte_offset, part):
        if path not in source_files:
            source_files[path] = open(path, "rb", buffering = 0)
        make_pack.read_exactly(source_files[path], byte_offset, part, path)

    if not isinstance(source, MatrixSource):
        read(source, source_byte_offset, destination)
        return
    weights, scales = (numpy.empty(part.byte_count, dtype = numpy.uint8) for part in (source.weights, source.scales))
    for part, part_bytes in ((source.weights, weights), (source.scales, scales)):
        read(part.path, part.byte_offset, memoryview(part_bytes))
    forbidden_code_bits, forbidden = FORMATS[source.weight_format].forbidden_code_bits, None
    if forbidden_code_bits is not None and ((weights & forbidden_code_bits) == forbidden_code_bits).any():
        forbidden = "a weight code that is no number"
    elif (scales == E8M0_NOT_A_NUMBER).any():
        forbidden = "a scale code that is no power of two"
    if forbidden:
        raise make_pack.PackError(f"{source.weights.path}: tensor {source.name!r} or its scales: {forbidden}")
    stored = store_matrix(source.weight_format, source.output_size, source.input_size, weights, scales)
    if stored.size != len(destination):
        raise RuntimeError(f"{source.name}: laid out in {stored.size} bytes, the layout of the pack gives it "
                           f"{len(destination)}")
    numpy.frombuffer(destination, dtype = numpy.uint8)[:] = stored


# ---- verify: nothing of the conversion above is used here ------------------------------------------------------------
def verified_byte_counts(weight_format, output_size, input_size):
    """The bytes of a matrix's codes and of the scales behind them, by the two stored_byte_count sentences of the
    header: at 4 bits outputs * inputs / 2 and outputs * inputs / 32, at 8 bits outputs * inputs and
    ((outputs + 31) / 32) * (inputs / 32)."""
    form = VERIFIED_FORMS[weight_format]
    scale_rows = (output_size + form.outputs_per_scale - 1) // form.outputs_per_scale
    return output_size * input_size // form.codes_per_byte, scale_rows * (input_size // VERIFIED_INPUTS_PER_SCALE)


def broken_matrix_rule(tensor):
    """The first rule of pack.h that the record of a matrix breaks, as the text that follows "tensor <index>" in
    verify's message, or None. tensor: a record with a weight format, as make_pack.read_pack_tables holds it. The
    last three are what stored_byte_count of the format answers for the record's shape and bits per weight."""
    form = VERIFIED_FORMS.get(tensor["weight_format"])
    if form is None:
        return f": this build carries no weight format {tensor['weight_format']}"
    if tensor["dimension_count"] != MATRIX_DIMENSIONS:
        return f" is a matrix of {form.name} with {tensor['dimension_count']} dimensions, not {MATRIX_DIMENSIONS}"
    if tensor["element_type"] != make_pack.ELEMENT_TYPES[MATRIX_DTYPE][0]:
        return (f" is a matrix of {form.name} and its element type is {tensor['element_type']}, not "
                f"{make_pack.ELEMENT_TYPES[MATRIX_DTYPE][0]}")
    output_size, input_size = tensor["dimensions"][:MATRIX_DIMENSIONS]
    if tensor["bits_per_weight"] != form.bits_per_weight:
        return f": {tensor['bits_per_weight']} bits per weight, {form.name} has {form.bits_per_weight}"
    if not output_size or not input_size or output_size % form.output_multiple or input_size % form.input_multiple:
        return (f": {form.name} stores no matrix of {output_size} outputs by {input_size} inputs: neither is zero, and "
                f"they are multiples of {form.output_multiple} and of {form.input_multiple}")
    needed_byte_count = sum(verified_byte_counts(tensor["weight_format"], output_size, input_size))
    if tensor["byte_count"] != needed_byte_count:
        return (f": {tensor['byte_count']} bytes, a matrix of {form.name} of {output_size} outputs by {input_size} "
                f"inputs holds {needed_byte_count}")
    return None


def checked_ranges(tensors):
    """What verify's worker processes look for in the matrices of a pack, whose records make_pack.read_pack_tables
    has checked. Returns extent index: [(first byte in the extent, end, bits, what a find is called)], where no byte
    from first to end may have all of bits set, and what verify prints about the matrices."""
    ranges, counts = collections.defaultdict(list), collections.Counter()
    for tensor_index, tensor in enumerate(tensors):
        form = VERIFIED_FORMS.get(tensor["weight_format"])
        if form is None:
            continue                                           # plain
        counts[form.name] += 1
        what = f"tensor {tensor_index}, a matrix of {form.name}, holds"
        code_byte_count, _ = verified_byte_counts(tensor["weight_format"], *tensor["dimensions"][:MATRIX_DIMENSIONS])
        scales_first_byte = tensor["byte_offset_in_extent"] + code_byte_count
        end_byte = tensor["byte_offset_in_extent"] + tensor["byte_count"]
        if form.forbidden_code_bits is not None:
            ranges[tensor["extent_index"]].append((tensor["byte_offset_in_extent"], scales_first_byte,
                                                   form.forbidden_code_bits, f"{what} a weight code that is no number"))
        ranges[tensor["extent_index"]].append((scales_first_byte, end_byte, VERIFIED_FORBIDDEN_SCALE_BITS,
                                               f"{what} a scale code that is no power of two"))
    counted = ", ".join(f"{counts[form.name]} of {form.name}" for form in VERIFIED_FORMS.values())
    return dict(ranges), f"matrices of a weight format without a code that is no number: {counted}"


def forbidden_value_in(piece, piece_byte_offset, ranges):
    """Worker process: what the first of an extent's checked ranges with a forbidden byte inside piece is called, or
    None. piece: a memoryview of the bytes of the extent from piece_byte_offset on."""
    for first_byte, end_byte, bits, what in ranges:
        first, end = max(first_byte - piece_byte_offset, 0), min(end_byte - piece_byte_offset, len(piece))
        if first < end and ((numpy.frombuffer(piece[first:end], dtype = numpy.uint8) & bits) == bits).any():
            return what
    return None


# ---- self-test -------------------------------------------------------------------------------------------------------
def checkpoint_codes(weight_format, weights):
    """The codes of a checkpoint's weight bytes in the order of their inputs, one to a byte, in plain Python: a byte
    of 4-bit codes holds the even input in its low four bits."""
    if FORMATS[weight_format].codes_per_byte == 1:
        return bytes(weights)
    return bytes(code for byte in weights for code in (byte & LOW_NIBBLE, byte >> NIBBLE_BITS))


def literal_stored_bytes(weight_format, output_size, input_size, weights, scales):
    """The stored bytes of a matrix a third time, for the self-test: one code at a time, in plain Python and in the
    order of the header's sentences, with none of the maps above. weights and scales are the bytes of the
    checkpoint's two tensors."""
    tiles, groups, stored = output_size // TILE_OUTPUTS, input_size // INPUTS_PER_SCALE, bytearray()
    lanes = tuple(itertools.product(range(LANE_GROUPS), range(LANE_POSITIONS)))        # lane 4 g + t, in order
    if weight_format == WEIGHT_FORMAT_E4M3_SCALED:
        # bytes [tile][group][lane][16]: byte b of a lane is
        # W[16 tile + g + 8 (b % 2)][32 group + 8 (b / 4) + 2 t + (b / 2) % 2]; scales [block][group] behind them
        for tile, group, (lane_group, lane_position), byte in itertools.product(
                range(tiles), range(groups), lanes, range(E4M3_LANE_BYTES)):
            output = TILE_OUTPUTS * tile + lane_group + LANE_GROUPS * (byte % OUTPUT_HALVES)
            column = (INPUTS_PER_SCALE * group + REGISTER_INPUTS * (byte // E4M3_QUARTER_BYTES)
                      + INPUT_PARITIES * lane_position + byte // OUTPUT_HALVES % INPUT_PARITIES)
            stored.append(weights[output * input_size + column])
        return bytes(stored) + bytes(scales)

    def code(output, column):                                  # the checkpoint's 4-bit code of W[output][column]
        byte = weights[output * (input_size // E2M1_CODES_PER_BYTE) + column // E2M1_CODES_PER_BYTE]
        return byte >> NIBBLE_BITS if column % E2M1_CODES_PER_BYTE else byte & LOW_NIBBLE

    # words [tile][load][lane][step]: the nibble at bit 4 r + 16 h of a word is
    # W[16 tile + g + 8 (r % 2)][64 load + 16 step + 2 t + 8 (r / 2) + h]
    loads = input_size // E2M1_LOAD_INPUTS
    for tile, load, (lane_group, lane_position), step in itertools.product(
            range(tiles), range(loads), lanes, range(E2M1_STEPS)):
        word = 0
        for operand_register, input_parity in itertools.product(range(E2M1_REGISTERS), range(INPUT_PARITIES)):
            output = TILE_OUTPUTS * tile + lane_group + LANE_GROUPS * (operand_register % OUTPUT_HALVES)
            column = (E2M1_LOAD_INPUTS * load + E2M1_STEP_INPUTS * step + INPUT_PARITIES * lane_position
                      + REGISTER_INPUTS * (operand_register // OUTPUT_HALVES) + input_parity)
            word |= code(output, column) << (NIBBLE_BITS * (operand_register + E2M1_REGISTERS * input_parity))
        stored += word.to_bytes(E2M1_WORD_BYTES, "little")
    # scales [tile][load][g]: byte 2 q + r of a word is the scale of inputs 64 load + 32 q .. + 31 of output
    # 16 tile + g + 8 r
    for tile, load, lane_group, scale_in_load, output_half in itertools.product(
            range(tiles), range(loads), range(LANE_GROUPS), range(E2M1_SCALES_PER_LOAD), range(OUTPUT_HALVES)):
        output = TILE_OUTPUTS * tile + lane_group + LANE_GROUPS * output_half
        stored.append(scales[output * groups + E2M1_SCALES_PER_LOAD * load + scale_in_load])
    return bytes(stored)


def self_test_values(dtype, data):
    """The self-test's hash bytes as values a matrix may hold: a weight code that is no number and a scale code that
    is no power of two each become the code below. The first scale of a tensor is 2^0, the byte 0x7F: a scale like
    any other, which neither convert nor verify may take for the weight code of those bits. Bytes of another element
    type stay as they are."""
    forbidden = [E8M0_NOT_A_NUMBER] if dtype == SCALES_DTYPE else [
        form.forbidden_code_bits for form in FORMATS.values()
        if form.codes_dtype == dtype and form.forbidden_code_bits is not None]
    for bits in forbidden:
        data = data.translate(bytes(value - ((value & bits) == bits) for value in range(BYTE_VALUES)))
    if dtype == SCALES_DTYPE and data:
        data = bytes((E4M3_NOT_A_NUMBER_BITS,)) + data[1:]
    return data


def expected_in_pack(units, stored_bytes):
    """The self-test's model as its pack must hold it, said here once more and without with_matrices. units: (unit
    kind, layer index, tensors), a tensor being (name, dtype, shape) as the folder stores it; stored_bytes, name:
    the bytes the folder stores. Returns the units with every quantized weight as its matrix, (name, "U8", (outputs,
    inputs)), and without its scales; name: the bytes of every tensor of the pack (of a matrix:
    literal_stored_bytes); and for the matrices name: (weight format, bits per weight)."""
    formats_by_dtype = {form.codes_dtype: weight_format for weight_format, form in FORMATS.items()}
    packed_units, packed_contents, matrices = [], {}, {}
    for unit_kind, layer_index, tensors in units:
        packed_tensors = []
        for name, dtype, shape in tensors:
            weight_format = formats_by_dtype.get(dtype) if name.endswith(WEIGHT_SUFFIX) else None
            if dtype == SCALES_DTYPE:
                continue                                       # inside its matrix
            if weight_format is None:
                packed_tensors.append((name, dtype, shape))
                packed_contents[name] = stored_bytes[name]
                continue
            form = FORMATS[weight_format]
            output_size, input_size = shape[0], shape[1] * form.codes_per_byte
            packed_tensors.append((name, MATRIX_DTYPE, (output_size, input_size)))
            packed_contents[name] = literal_stored_bytes(
                weight_format, output_size, input_size, stored_bytes[name],
                stored_bytes[name[:-len(WEIGHT_SUFFIX)] + SCALE_SUFFIX])
            matrices[name] = (weight_format, form.bits_per_weight)
        packed_units.append((unit_kind, layer_index, tuple(packed_tensors)))
    return packed_units, packed_contents, matrices


def self_test(folder, source_folder, pack_path, stored_units, stored_bytes, expect, expect_refusal):
    """The matrices' part of make_pack.py's self-test, on its toy model: pack_path is the pack of source_folder,
    which stores no n-gram table yet, and stored_units and stored_bytes are the model as the folder stores it (the
    arguments of expected_in_pack). expect and expect_refusal are make_pack.self_test's. What is shown: the text at
    the top of this file."""
    jobs, broken_path = make_pack.SELF_TEST_JOBS, os.path.join(folder, "broken_matrices.pack")

    def read(path):
        with open(path, "rb") as handle:
            return handle.read()

    # the layouts alone: the byte counts, the sizes no format stores, a caller's mistake, the places counted by hand,
    # the three writings, the cuts of an extent
    for weight_format, output_size, input_size, byte_count in SELF_TEST_BYTE_COUNTS:
        expect(shape_refusal(weight_format, output_size, input_size) is None
               and stored_byte_count(weight_format, output_size, input_size) == byte_count
               and sum(verified_byte_counts(weight_format, output_size, input_size)) == byte_count,
               f"{FORMATS[weight_format].name}, {output_size} by {input_size}: {byte_count} bytes for convert and for "
               "verify")
    for output_size, input_size, rule_text in (
            (0, INPUTS_PER_SCALE, "0 outputs are none"), (TILE_OUTPUTS, 0, "0 inputs are none"),
            (TILE_OUTPUTS, make_pack.UINT32_MAXIMUM + 1, "more than 32 bits count")):
        expect(rule_text in str(shape_refusal(WEIGHT_FORMAT_E4M3_SCALED, output_size, input_size)),
               f"no format stores a matrix of {output_size} outputs by {input_size} inputs: {rule_text}")

    def refused(function, *arguments):
        """Whether store_matrix or read_matrix refuses its arguments by its own check, as its caller's mistake."""
        try:
            function(*arguments)
        except (ValueError, IndexError) as refusal:
            return str(refusal).startswith(function.__name__)
        return False

    half_tile = TILE_OUTPUTS // OUTPUT_HALVES
    expect(refused(store_matrix, WEIGHT_FORMAT_E4M3_SCALED, half_tile, INPUTS_PER_SCALE,
                   bytes(half_tile * INPUTS_PER_SCALE), bytes(1))
           and refused(read_matrix, WEIGHT_FORMAT_E4M3_SCALED, TILE_OUTPUTS, INPUTS_PER_SCALE, bytes(TILE_OUTPUTS)),
           "store_matrix refuses a size its format does not store, read_matrix bytes that are not a matrix's")

    def drawn(weight_format, output_size, input_size):
        """What a case is called, and hash bytes for the two tensors a checkpoint stores a matrix in."""
        what = f"{FORMATS[weight_format].name}, {output_size} by {input_size}"
        return (what, *(hashlib.shake_256(f"{what}: {part}".encode("utf-8")).digest(math.prod(shape)) for part, shape
                        in zip(("weights", "scales"), checkpoint_shapes(weight_format, output_size, input_size))))

    for weight_format, output_size, input_size, code_places, scale_places in SELF_TEST_PLACES:
        what, weights, scales = drawn(weight_format, output_size, input_size)
        stored = store_matrix(weight_format, output_size, input_size, weights, scales).tobytes()
        codes, groups = checkpoint_codes(weight_format, weights), input_size // INPUTS_PER_SCALE
        expect(all((stored[byte] if half is None else (stored[byte] >> (NIBBLE_BITS * half)) & LOW_NIBBLE)
                   == codes[output * input_size + column] for byte, half, output, column in code_places)
               and all(stored[len(stored) - len(scales) + byte] == scales[row * groups + group]
                       for byte, row, group in scale_places),
               f"{what}: codes and scales lie at the places counted by hand from the header's numbers")
    for weight_format, output_size, input_size in SELF_TEST_SHAPES:
        what, weights, scales = drawn(weight_format, output_size, input_size)
        shapes = checkpoint_shapes(weight_format, output_size, input_size)
        stored = store_matrix(weight_format, output_size, input_size, weights, scales).tobytes()
        expect(stored == literal_stored_bytes(weight_format, output_size, input_size, weights, scales)
               and len(stored) == stored_byte_count(weight_format, output_size, input_size),
               f"{what}: store_matrix writes what the sentences of the header say, code by code")
        codes, read_scales = read_matrix(weight_format, output_size, input_size, stored)
        expect((codes.shape, read_scales.shape) == ((output_size, input_size), shapes[1])
               and codes.tobytes() == checkpoint_codes(weight_format, weights) and read_scales.tobytes() == scales,
               f"{what}: read_matrix gives the codes and the scales of the checkpoint back")
    range_first_byte, range_end_byte = SELF_TEST_RANGE
    ranges = [(range_first_byte, range_end_byte, E8M0_NOT_A_NUMBER, "found")]
    for forbidden_byte, piece_byte_count in itertools.product(range(SELF_TEST_EXTENT_BYTES),
                                                              range(1, SELF_TEST_EXTENT_BYTES + 1)):
        extent = bytearray(SELF_TEST_EXTENT_BYTES)
        extent[forbidden_byte] = E8M0_NOT_A_NUMBER
        finds = [forbidden_value_in(memoryview(extent)[first_byte:first_byte + piece_byte_count], first_byte, ranges)
                 for first_byte in range(0, SELF_TEST_EXTENT_BYTES, piece_byte_count)]
        expect(finds.count("found") == (range_first_byte <= forbidden_byte < range_end_byte)
               and finds.count(None) == len(finds) - finds.count("found"),
               f"a forbidden byte at {forbidden_byte} of an extent read in pieces of {piece_byte_count}: found once "
               f"if it lies in the range {list(SELF_TEST_RANGE)}, and not otherwise")

    # the toy pack: every tensor and every extent against a second writing of them, without convert's arithmetic
    _, extents, units, tensors = make_pack.read_pack_tables(pack_path)
    pack_bytes, packed = read(pack_path), {tensor["name"].decode("utf-8"): tensor for tensor in tensors}
    expected_units, expected_bytes, matrices = expected_in_pack(stored_units, stored_bytes)
    expect(sorted(packed) == sorted(expected_bytes) and len(tensors) == len(expected_bytes),
           "the pack holds every tensor once: a quantized weight as one matrix, and no tensor of scales")

    def first_byte_of(name):
        return extents[packed[name]["extent_index"]]["byte_offset"] + packed[name]["byte_offset_in_extent"]

    for unit_index, ((_, _, unit_tensors), unit_extent) in enumerate(zip(expected_units, extents)):
        for name, dtype, shape in unit_tensors:
            if name not in packed:
                continue                                       # reported above
            tensor, (element_type, _) = packed[name], make_pack.ELEMENT_TYPES[dtype]
            bytes_in_pack = pack_bytes[first_byte_of(name):first_byte_of(name) + tensor["byte_count"]]
            expect(extents[tensor["extent_index"]]["unit_index"] == unit_index
                   and tensor["element_type"] == element_type
                   and tensor["dimensions"][:tensor["dimension_count"]] == shape
                   and (tensor["weight_format"], tensor["bits_per_weight"])
                   == matrices.get(name, (make_pack.WEIGHT_FORMAT_NONE, 0)) and bytes_in_pack == expected_bytes[name],
                   f"{name}: its unit, its element type, its shape, its weight format with its bits per weight, and "
                   "its bytes (the last file's)")
        expected_extent = b""
        for name in sorted((name for name, _, _ in unit_tensors), key = str.encode):
            expected_extent += bytes(-len(expected_extent) % make_pack.MATRIX_ALIGNMENT_BYTES) + expected_bytes[name]
        expected_extent += bytes(-len(expected_extent) % make_pack.PACK_BLOCK_BYTES)
        expect(pack_bytes[unit_extent["byte_offset"]:][:unit_extent["byte_count"]] == expected_extent,
               f"unit {unit_index}: its extent is its tensors in the order of their names, aligned and zero-filled")

    # its matrices: every one read back through read_matrix; convert's line
    shapes = {name: shape for _, _, unit_tensors in expected_units for name, _, shape in unit_tensors}
    matrix_names = {weight_format: [] for weight_format in FORMATS}
    for name, (weight_format, _) in matrices.items():
        if name not in packed or packed[name]["byte_count"] != stored_byte_count(weight_format, *shapes[name]):
            continue                                           # reported above
        matrix_names[weight_format].append(name)
        codes, scales = read_matrix(weight_format, *shapes[name],
                                    pack_bytes[first_byte_of(name):first_byte_of(name) + packed[name]["byte_count"]])
        expect(codes.tobytes() == checkpoint_codes(weight_format, stored_bytes[name])
               and scales.tobytes() == stored_bytes[name[:-len(WEIGHT_SUFFIX)] + SCALE_SUFFIX],
               f"{name}: read back through read_matrix, the codes and the scales the folder stores")
    expect(all(matrix_names.values()), "the toy pack holds a matrix of each format")
    if not all(matrix_names.values()):
        return                                                 # nothing below can be shown without them
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        make_pack.convert(source_folder, broken_path, jobs, False)
    expect(read(broken_path) == pack_bytes and all(
        f"{len(names)} {FORMATS[weight_format].name}" in printed.getvalue()
        for weight_format, names in matrix_names.items()), "convert prints the count of the matrices of each format")
    os.remove(broken_path)

    # what convert refuses of a folder: (what the folder also holds, its tensors, the byte of the file that is
    # changed as (distance from the end, value) or None, what the refusal says)
    e2m1, e4m3 = FORMATS[WEIGHT_FORMAT_E2M1_SCALED], FORMATS[WEIGHT_FORMAT_E4M3_SCALED]
    weight_name, scale_name = SELF_TEST_STRAY_NAME + WEIGHT_SUFFIX, SELF_TEST_STRAY_NAME + SCALE_SUFFIX
    stray = []                                                 # for each format: the two tensors of its smallest matrix
    for weight_format, form in FORMATS.items():
        weight_shape, scale_shape = checkpoint_shapes(weight_format, TILE_OUTPUTS, form.input_multiple)
        stray.append(((weight_name, form.codes_dtype, weight_shape), (scale_name, SCALES_DTYPE, scale_shape)))
    (e2m1_weight, e2m1_scale), (e4m3_weight, e4m3_scale) = stray
    e4m3_file_bytes = math.prod(e4m3_weight[-1]) + math.prod(e4m3_scale[-1])       # the file ends with these
    refused_folders = (
        ("a weight of 8 bits without scales", (e4m3_weight,), None, f"its unit holds no scales {scale_name!r}"),
        ("a weight of 4 bits without scales", (e2m1_weight,), None, f"its unit holds no scales {scale_name!r}"),
        ("a weight with scales of another shape",
         (e4m3_weight, (scale_name, SCALES_DTYPE, SELF_TEST_OTHER_SCALE_SHAPE)), None,
         f"{e4m3.name} takes {SCALES_DTYPE} {list(e4m3_scale[-1])}"),
        ("a weight with scales of another element type", (e4m3_weight, (scale_name, MATRIX_DTYPE, e4m3_scale[-1])),
         None, f"its scales {scale_name!r} are {MATRIX_DTYPE}"),
        ("a weight of half a tile of outputs",
         ((weight_name, e4m3.codes_dtype, (half_tile, e4m3.input_multiple)), e4m3_scale), None,
         f"{half_tile} outputs are no multiple of {TILE_OUTPUTS}"),
        ("a weight of 4 bits of one scale group of inputs",
         ((weight_name, e2m1.codes_dtype, (TILE_OUTPUTS, INPUTS_PER_SCALE // E2M1_CODES_PER_BYTE)),
          (scale_name, SCALES_DTYPE, (TILE_OUTPUTS, 1))), None,
         f"{INPUTS_PER_SCALE} inputs are no multiple of {e2m1.input_multiple}"),
        ("a weight of one dimension", ((weight_name, e4m3.codes_dtype, (TILE_OUTPUTS,)),), None, "is no matrix"),
        ("codes under another name", ((SELF_TEST_STRAY_NAME, e4m3.codes_dtype, e4m3_weight[-1]),), None,
         f"its name does not end in {WEIGHT_SUFFIX}"),
        ("scales without a weight", (e4m3_scale,), None, "no quantized weight of its unit takes them"),
        ("a weight code that is no number", (e4m3_weight, e4m3_scale), (e4m3_file_bytes, E4M3_NOT_A_NUMBER_BITS),
         "a weight code that is no number"),
        ("the negative weight code that is no number", (e4m3_weight, e4m3_scale),
         (e4m3_file_bytes, SELF_TEST_NEGATIVE_NOT_A_NUMBER), "a weight code that is no number"),
        ("a scale that is no power of two beside a weight of 8 bits", (e4m3_weight, e4m3_scale),
         (1, E8M0_NOT_A_NUMBER), "a scale code that is no power of two"),
        ("a scale that is no power of two beside a weight of 4 bits", (e2m1_weight, e2m1_scale),
         (1, E8M0_NOT_A_NUMBER), "a scale code that is no power of two"))
    stray_path = os.path.join(source_folder, make_pack.SELF_TEST_STRAY_FILE_NAME)
    for what, stray_tensors, changed_byte, rule_text in refused_folders:
        make_pack.write_safetensors_file(stray_path, stray_tensors)
        if changed_byte is not None:                           # the file ends with the bytes of its tensors, in order
            distance_from_end, value = changed_byte
            with open(stray_path, "r+b") as handle:
                handle.seek(-distance_from_end, os.SEEK_END)
                handle.write(bytes((value,)))
        expect_refusal(lambda: make_pack.convert(source_folder, broken_path, jobs, False), rule_text,
                       f"a folder that also holds {what}")
        expect(not os.path.lexists(broken_path) and not os.path.lexists(broken_path + make_pack.TEMPORARY_SUFFIX),
               f"the conversion refused for {what} leaves no file")
    os.remove(stray_path)
    # a file whose header gives a tensor of a matrix other bytes than its shape: refused before a worker reads it
    stray_unit = [make_pack.Unit(make_pack.UNIT_STAGE, make_pack.PACK_NO_LAYER, "stray", [weight_name, scale_name])]
    whole = {name: make_pack.SourceTensor(stray_path, 0, math.prod(shape), dtype, shape)
             for name, dtype, shape in (e4m3_weight, e4m3_scale)}
    for name, source in whole.items():
        short = {**whole, name: source._replace(byte_count = source.byte_count - 1)}
        expect_refusal(lambda: with_matrices(stray_unit, short),
                       f"tensor {name!r}: {source.byte_count - 1} bytes stored", f"{name} stored one byte short")

    # what verify refuses: every rule shown failing once, on a copy of the pack with a few bytes changed
    tensor_records = (make_pack.PACK_BLOCK_BYTES + len(extents) * make_pack.EXTENT_RECORD_BYTES
                      + len(units) * make_pack.UNIT_RECORD_BYTES)

    def in_header(field, value):
        byte_offset, byte_count = make_pack.HEADER_FIELDS[field]
        return byte_offset, value.to_bytes(byte_count, "little")

    def in_record(name, field, value):
        byte_offset, byte_count = make_pack.TENSOR_FIELDS[field]
        return (tensor_records + list(packed).index(name) * make_pack.TENSOR_RECORD_BYTES + byte_offset,
                value.to_bytes(byte_count, "little"))

    def in_matrix(name, byte_index, value):
        return first_byte_of(name) + byte_index, bytes((value,))

    def refused_by_verify(what, rule_text, changes):
        """verify on a copy of the pack with bytes changed, (byte offset in the file, bytes) each, and the digests
        made to fit: of every extent a change lies in, of the tables, of the header. So a change reaches its rule."""
        shutil.copyfile(pack_path, broken_path)
        with open(broken_path, "r+b") as handle:
            for byte_offset, replacement in changes:
                handle.seek(byte_offset)
                handle.write(replacement)
            for extent_index, extent in enumerate(extents):
                if any(0 <= byte_offset - extent["byte_offset"] < extent["byte_count"] for byte_offset, _ in changes):
                    handle.seek(extent["byte_offset"])
                    digest = make_pack.digest_of(handle.read(extent["byte_count"]))
                    digest_byte_offset, digest_byte_count = make_pack.EXTENT_FIELDS["digest"]
                    handle.seek(make_pack.PACK_BLOCK_BYTES + extent_index * make_pack.EXTENT_RECORD_BYTES
                                + digest_byte_offset)
                    handle.write(digest.to_bytes(digest_byte_count, "little"))
        make_pack.reseal(broken_path)
        expect_refusal(lambda: make_pack.verify(broken_path, jobs), rule_text, what)

    (e2m1_name, *_), (e4m3_name, *_) = (matrix_names[weight_format] for weight_format in FORMATS)
    plain_name = next(name for name, tensor in packed.items() if not tensor["weight_format"])
    e4m3_tensor, former_version = packed[e4m3_name], make_pack.PACK_FORMAT_VERSION - 1
    output_size, input_size = shapes[e4m3_name]
    for what, rule_text, changes in (
            ("a pack of the former format version", f"the format version is {former_version}, not",
             [in_header("format_version", former_version)]),
            ("a plain tensor with bits per weight", "is plain and its bits_per_weight is",
             [in_record(plain_name, "bits_per_weight", e4m3.bits_per_weight)]),
            ("a matrix of a format this build does not carry",
             f"this build carries no weight format {SELF_TEST_UNCARRIED_FORMAT}",
             [in_record(e4m3_name, "weight_format", SELF_TEST_UNCARRIED_FORMAT)]),
            ("a matrix of three dimensions", f"with {MATRIX_DIMENSIONS + 1} dimensions, not {MATRIX_DIMENSIONS}",
             [in_record(e4m3_name, "dimension_count", MATRIX_DIMENSIONS + 1)]),
            ("a matrix of another element type", f"a matrix of {e4m3.name} and its element type is",
             [in_record(e4m3_name, "element_type", make_pack.ELEMENT_TYPES[INTEGER_DTYPE][0])]),
            ("a matrix of 8 bits that says 4", f"{e2m1.bits_per_weight} bits per weight, {e4m3.name} has",
             [in_record(e4m3_name, "bits_per_weight", e2m1.bits_per_weight)]),
            ("a matrix of half a tile of outputs", f"{e4m3.name} stores no matrix of {output_size // OUTPUT_HALVES} ",
             [in_record(e4m3_name, "dimensions",
                        output_size // OUTPUT_HALVES + input_size * (make_pack.UINT32_MAXIMUM + 1))]),
            # three more sizes no format stores, each with the byte count made to fit: this rule alone refuses them
            ("a matrix of no outputs", f"{e4m3.name} stores no matrix of 0 outputs",
             [in_record(e4m3_name, "dimensions", input_size * (make_pack.UINT32_MAXIMUM + 1)),
              in_record(e4m3_name, "byte_count", 0)]),
            ("a matrix of no inputs", f"{e4m3.name} stores no matrix of {output_size} outputs by 0 inputs",
             [in_record(e4m3_name, "dimensions", output_size), in_record(e4m3_name, "byte_count", 0)]),
            ("a matrix of 4 bits of one scale group of inputs",
             f"{e2m1.name} stores no matrix of {TILE_OUTPUTS} outputs by {INPUTS_PER_SCALE} inputs",
             [in_record(e2m1_name, "dimensions", TILE_OUTPUTS + INPUTS_PER_SCALE * (make_pack.UINT32_MAXIMUM + 1)),
              in_record(e2m1_name, "byte_count",
                        sum(verified_byte_counts(WEIGHT_FORMAT_E2M1_SCALED, TILE_OUTPUTS, INPUTS_PER_SCALE)))]),
            ("a matrix one byte short", f"{e4m3_tensor['byte_count'] - 1} bytes, a matrix of {e4m3.name}",
             [in_record(e4m3_name, "byte_count", e4m3_tensor["byte_count"] - 1)]),
            ("a tensor record with its reserved word set", "a reserved field is not zero",
             [in_record(e4m3_name, "reserved_zero", 1)]),
            ("a weight code that is no number in a matrix", f"a matrix of {e4m3.name}, holds a weight code",
             [in_matrix(e4m3_name, 0, E4M3_NOT_A_NUMBER_BITS)]),
            ("the negative weight code that is no number in the last weight of a matrix",
             f"a matrix of {e4m3.name}, holds a weight code",
             [in_matrix(e4m3_name, output_size * input_size - 1, SELF_TEST_NEGATIVE_NOT_A_NUMBER)]),
            ("a scale that is no power of two in a matrix of 8 bits", f"a matrix of {e4m3.name}, holds a scale code",
             [in_matrix(e4m3_name, e4m3_tensor["byte_count"] - 1, E8M0_NOT_A_NUMBER)]),
            ("a scale that is no power of two in a matrix of 4 bits", f"a matrix of {e2m1.name}, holds a scale code",
             [in_matrix(e2m1_name, packed[e2m1_name]["byte_count"] - 1, E8M0_NOT_A_NUMBER)])):
        refused_by_verify(what, rule_text, changes)
