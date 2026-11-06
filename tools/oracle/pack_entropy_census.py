#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How many bits the scales of the 4-bit experts and the codes of the 8-bit trunk of a pack need without loss.

    pack_entropy_census.py <pack file> [--matrices PATTERN] [--break-recoding] [--break-blocks]

The experts' scales (one E8M0 byte for every output and 32 inputs: 0.25 of an expert's 4.25 bits a weight). Every
row of scales of every 4-bit matrix is coded as another engine codes its own: a base for the row, ONE bit for every
scale (the base, or one above), and what is neither as an exception; the base is the one that leaves the fewest.
Priced: a row costs its base byte and a count of its exceptions, an exception its place in the row and its byte.
One line for each layer (and draft block) and one for all: the share of exceptions, the bits a scale costs so
coded, and the bits held by a scale's distance from its row's base: the floor of a code that knows the base and
takes every scale alone, by one table for all the rows counted. A code that also reads a scale's neighbours (the
scales of the same inputs in the rows beside it) is not held by that floor.
Checked in every run: every row is decoded again from its base, its bits and its list of exceptions and from
nothing else, and must be the stored row. --break-recoding shows the check failing: the decoder reads every bit
inverted, and every row must be reported.

The trunk's codes (E4M3 bytes, one scale for every 32 outputs by 32 inputs). For every such block of every 8-bit
matrix, how often it holds each of the 256 codes. One line for each kind of matrix and one for all: the share of
blocks that hold at most 16, 32, 64 and 128 different codes (4 to 7 bits a weight as indexes into a table of the
block's own), the median count of different codes (of an even number of blocks the upper of the middle two, as
expert_sharing.py takes a median), and the bits a weight takes by an ideal code with a table for every block, by
one with a table for every matrix, and as indexes of fixed width with the table's bytes counted (the plain byte
where that is less). The tables of the two ideal codes are NOT in their bits (a block's own would be up to 256
counts for its 1,024 codes): those two figures are floors, not sizes.
Checked in every run: in every 8-bit matrix the first, a middle and the last block (of 32 outputs and, where a
matrix ends in one, of 16) are counted again, code by code from the matrix as read, and must hold each code as
often as the census found; and the census must have cut the matrix into as many blocks as it stores scales.
--break-blocks shows the check failing: one is added to the recount of the first of those blocks in every matrix,
and that one block of each must be reported.
The exit status is 0 only if both checks found what the run asks of them: nothing, or with its flag exactly what
the flag says. A check with nothing to look at (no matrix of its format among those taken) says not_run. A run
with a flag is no result: its VERDICT lines say so.

Refused: what make_pack.read_pack_tables refuses of a pack; --matrices (fnmatch over the tensors' names, such as
"layers.0.*") that takes no matrix of a weight format; --break-recoding where it takes no 4-bit matrix and
--break-blocks where it takes none of 8 bits.
What it does not show: what a kernel pays to read scales coded so (another engine measured 13.7% of its decode
before it fused the expansion into its product), or that a card then holds more experts; a last block of 16
outputs counts as a block. It reads the scales of every expert (16 GiB of this model's pack) and the whole trunk
(7 GiB). Needs numpy. Exit status 0, 1 after a failure, 2 for a command line it cannot read.
Measurement tool: not part of the engine.
"""
import argparse
import fnmatch
import os
import re
import sys
import time

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "convert"))
import kernel_layouts                                      # where a pack stores the scales and the codes of a matrix
import make_pack                                           # the reader of a pack's tables

BYTE_VALUES, BITS_PER_BYTE = 256, 8
SCALE_INPUTS = kernel_layouts.INPUTS_PER_SCALE             # 32 inputs share a scale, in both formats
BLOCK_OUTPUTS = kernel_layouts.FORMATS[kernel_layouts.WEIGHT_FORMAT_E4M3_SCALED].outputs_per_scale    # 32; the last: 16
OUTPUTS_IN_BLOCK_AXIS, BLOCKS_OF_INPUTS_AXIS = 1, 2        # of [block of outputs][output][block of inputs][input]
INDEX_BITS = (4, 5, 6, 7)                                  # a block fits n bits if it holds at most 2^n different codes
BITS_OF_AN_INDEX = numpy.array([max(count - 1, 0).bit_length() for count in range(BYTE_VALUES + 1)])   # by that count
SCALE_FACTS = ("matrices", "rows", "scales", "exceptions", "coded_bits", "rows_not_decoded")   # then the distances
CODE_FACTS = ("matrices", "codes", "block_bits", "matrix_bits", "indexed_bits", "recounts",
              "recounts_that_differ")                      # then the blocks by their codes
EXPERT_OWNER = re.compile(r"(?P<owner>.+)\.ffn\.experts\..+")                      # a layer, or a draft block
NUMBER_IN_A_NAME = re.compile(r"(?<=\.)[0-9]+(?=\.)")
NUMBERS_AND_TEXT = re.compile(r"([0-9]+)")
PROGRESS_INTERVAL_SECONDS = 10.0
PERCENT = 100.0
EXIT_STATUS_FAILURE = 1
PackError = make_pack.PackError


def counts_by_row(values):
    """How often every row of uint8 [rows, n] holds each byte: int64 [rows, 256]."""
    keys = numpy.arange(len(values))[:, None] * BYTE_VALUES + values
    return numpy.bincount(keys.ravel(), minlength = len(values) * BYTE_VALUES).reshape(len(values), BYTE_VALUES)


def ideal_bits(counts):
    """The bits of every row's symbols under an ideal code of the row's own table, summed over the rows; counts
    [rows, symbols] holds how often a row has each symbol."""
    totals = counts.sum(axis = 1)
    return float((totals * numpy.log2(numpy.maximum(totals, 1))).sum()
                 - (counts * numpy.log2(numpy.maximum(counts, 1))).sum())


def coded_rows(scales):
    """Every row of scales, uint8 [rows, n], as the text at the top codes it: the bases [rows], one bit for every
    scale [rows, n] (set: one above the base) and the exceptions as (row, place, scale) arrays."""
    counts = counts_by_row(scales)
    bases = (counts[:, :-1] + counts[:, 1:]).argmax(axis = 1)  # the base that the most scales are, or are one above
    distances = scales.astype(numpy.int64) - bases[:, None]
    exception_rows, exception_places = numpy.nonzero((distances < 0) | (distances > 1))
    return bases, distances == 1, (exception_rows, exception_places, scales[exception_rows, exception_places])


def decoded_rows(bases, bits, exceptions, bits_inverted):
    """The rows again, int64 [rows, n], from what coded_rows returned and nothing else."""
    exception_rows, exception_places, exception_scales = exceptions
    decoded = bases[:, None] + (bits != bits_inverted)
    decoded[exception_rows, exception_places] = exception_scales
    return decoded


def scale_facts(scales, bits_inverted):
    """The scales of one matrix, uint8 [outputs, groups], coded row by row as the text at the top says: SCALE_FACTS,
    then how often a scale lies at each distance from its row's base (-256 to 255), int64 in all."""
    row_count, scales_per_row = scales.shape
    bases, bits, exceptions = coded_rows(scales)
    exception_count = len(exceptions[0])
    row_bits = BITS_PER_BYTE + scales_per_row.bit_length()     # the base, and a count from 0 to scales_per_row
    exception_bits = max((scales_per_row - 1).bit_length(), 1) + BITS_PER_BYTE     # its place, and the scale itself
    coded_bits = row_count * row_bits + bits.size + exception_count * exception_bits
    rows_not_decoded = int((decoded_rows(bases, bits, exceptions, bits_inverted) != scales).any(axis = 1).sum())
    facts = [1, row_count, scales.size, exception_count, coded_bits, rows_not_decoded]
    return numpy.concatenate((facts, numpy.bincount((scales.astype(numpy.int64) - bases[:, None] + BYTE_VALUES).ravel(),
                                                    minlength = BYTE_VALUES + BYTE_VALUES)))


def code_facts(codes, scale_count, recount_broken):
    """The codes of one 8-bit matrix, uint8 [outputs, inputs], which stores scale_count scales, block by block:
    CODE_FACTS, then how many blocks hold 0 to 256 different codes, float64 in all (whole numbers, and bits)."""
    whole_outputs = len(codes) - len(codes) % BLOCK_OUTPUTS
    block_bits = indexed_bits = 0.0
    recounts = recounts_that_differ = 0                        # of the check: blocks, and the number of blocks
    matrix_counts, blocks_by_codes = numpy.zeros((1, BYTE_VALUES)), numpy.zeros(BYTE_VALUES + 1)
    for part, block_outputs in ((codes[:whole_outputs], BLOCK_OUTPUTS),
                                (codes[whole_outputs:], len(codes) - whole_outputs)):
        if not len(part):
            continue                                           # no whole block, or no last block of fewer outputs
        block_codes, blocks_of_inputs = block_outputs * SCALE_INPUTS, part.shape[1] // SCALE_INPUTS
        blocks = part.reshape(len(part) // block_outputs, block_outputs, blocks_of_inputs, SCALE_INPUTS)
        counts = counts_by_row(blocks.swapaxes(OUTPUTS_IN_BLOCK_AXIS, BLOCKS_OF_INPUTS_AXIS).reshape(-1, block_codes))
        different = (counts > 0).sum(axis = 1)
        blocks_by_codes += numpy.bincount(different, minlength = BYTE_VALUES + 1)
        matrix_counts += counts.sum(axis = 0)
        block_bits += ideal_bits(counts)
        indexed_bits += float(numpy.minimum(BITS_OF_AN_INDEX[different] * block_codes + BITS_PER_BYTE * different,
                                            BITS_PER_BYTE * block_codes).sum())
        for block in sorted({0, len(counts) // 2, len(counts) - 1}):   # of this part, in the order of its counts
            first_output, first_input = (place * size for place, size in zip(divmod(block, blocks_of_inputs),
                                                                              (block_outputs, SCALE_INPUTS)))
            recount = [0] * BYTE_VALUES
            for code in part[first_output:first_output + block_outputs,
                             first_input:first_input + SCALE_INPUTS].ravel().tolist():
                recount[code] += 1
            if recount_broken and not recounts:                # the first block recounted of the matrix
                recount[0] += 1
            recounts, recounts_that_differ = recounts + 1, recounts_that_differ + (counts[block].tolist() != recount)
    recounts, recounts_that_differ = recounts + 1, recounts_that_differ + (blocks_by_codes.sum() != scale_count)
    return numpy.concatenate(([1, codes.size, block_bits, ideal_bits(matrix_counts), indexed_bits, recounts,
                               recounts_that_differ], blocks_by_codes))


def scale_line(label, facts):
    matrix_count, row_count, scale_count, exception_count, coded_bits, _ = facts[:len(SCALE_FACTS)].tolist()
    return (f"{label}: {matrix_count} matrices, {scale_count} scales in {row_count} rows: "
            f"{PERCENT * exception_count / scale_count:.3f}% exceptions, {coded_bits / scale_count:.3f} bits a scale "
            f"as a base, a bit and exceptions ({BITS_PER_BYTE} today); a scale's distance from its row's base holds "
            f"{ideal_bits(facts[None, len(SCALE_FACTS):]) / scale_count:.3f} bits")


def code_line(label, facts):
    matrix_count, code_count, block_bits, matrix_bits, indexed_bits, _, _ = facts[:len(CODE_FACTS)].tolist()
    blocks_by_codes = facts[len(CODE_FACTS):]
    block_count = blocks_by_codes.sum()
    within = ", ".join(f"{PERCENT * blocks_by_codes[:(1 << bits) + 1].sum() / block_count:.2f}% at most {1 << bits}"
                       for bits in INDEX_BITS)
    median = int(numpy.searchsorted(numpy.cumsum(blocks_by_codes), block_count // 2, side = "right"))
    return (f"{label}: {matrix_count:.0f} matrices, {block_count:.0f} blocks, by the different codes a block holds: "
            f"{within}, {median} at the median; a weight takes {block_bits / code_count:.3f} bits by an ideal code of "
            f"each block's table and {matrix_bits / code_count:.3f} of each matrix's (neither table counted), "
            f"{indexed_bits / code_count:.3f} as indexes with their table counted ({BITS_PER_BYTE} today)")


def natural_order(named):
    return [int(part) if part.isdigit() else part for part in NUMBERS_AND_TEXT.split(named[0])]


def check_line(name, differing, of, as_required, broken):
    """One CHECK line; as_required: whether the check found what this run asks of it. Returns as_required."""
    verdict = "FAILED" if not as_required else "reported" if broken else "ok" if of else "not_run"
    print(f"CHECK name={name} differing={differing} of={of} verdict={verdict}")
    return as_required


def census(pack_path, pattern, recoding_broken, blocks_broken):
    """Reads every matrix the pattern takes and prints the lines of the text at the top; returns the exit status."""
    started = last_report = time.monotonic()
    _, extents, _, tensors = make_pack.read_pack_tables(pack_path)
    matrices = [(tensor["name"].decode("utf-8", "replace"), tensor) for tensor in tensors
                if tensor["weight_format"] != make_pack.WEIGHT_FORMAT_NONE]
    chosen = [(name, tensor) for name, tensor in matrices if fnmatch.fnmatchcase(name, pattern)]
    expert_format = kernel_layouts.WEIGHT_FORMAT_E2M1_SCALED
    expert_count = sum(tensor["weight_format"] == expert_format for _, tensor in chosen)
    if not chosen or recoding_broken and not expert_count or blocks_broken and expert_count == len(chosen):
        raise PackError(f"--matrices {pattern} takes none of the {len(matrices)} matrices of a weight format, or none "
                        "of the format whose check a flag breaks: 4 bits for --break-recoding, 8 for --break-blocks")
    owners, kinds, expert_byte_count, blocks_as_required = {}, {}, 0, True
    with open(pack_path, "rb", buffering = 0) as pack:
        for matrix_index, (name, tensor) in enumerate(chosen):
            output_size, input_size = tensor["dimensions"][:kernel_layouts.MATRIX_DIMENSIONS]
            is_expert = tensor["weight_format"] == expert_format
            scale_count = output_size * (input_size // SCALE_INPUTS)
            stored = numpy.empty(scale_count if is_expert else tensor["byte_count"], dtype = numpy.uint8)
            end_byte = (extents[tensor["extent_index"]]["byte_offset"] + tensor["byte_offset_in_extent"]
                        + tensor["byte_count"])
            make_pack.read_exactly(pack, end_byte - stored.size, memoryview(stored), pack_path)
            if is_expert:                                      # only its scales are read: they lie behind its codes
                _, scale_places = kernel_layouts.e2m1_places_of(input_size)
                scales = stored.reshape(output_size // kernel_layouts.TILE_OUTPUTS, -1).take(scale_places, axis = 1)
                owner = EXPERT_OWNER.fullmatch(name)
                owner = owner["owner"] if owner else name
                owners[owner] = owners.get(owner, 0) + scale_facts(scales.reshape(output_size, -1), recoding_broken)
                expert_byte_count += tensor["byte_count"]
            else:
                codes, scales = kernel_layouts.read_matrix(tensor["weight_format"], output_size, input_size, stored)
                kind, facts = NUMBER_IN_A_NAME.sub("*", name), code_facts(codes, scales.size, blocks_broken)
                kinds[kind] = kinds.get(kind, 0) + facts
                blocks_as_required &= facts[CODE_FACTS.index("recounts_that_differ")] == int(blocks_broken)
            if time.monotonic() - last_report >= PROGRESS_INTERVAL_SECONDS:
                last_report = time.monotonic()
                print(f"  read {matrix_index + 1} of {len(chosen)} matrices", file = sys.stderr, flush = True)

    scope = f"matrices={pattern}"
    result_verdict = "not_a_result" if recoding_broken or blocks_broken else "measured"
    for owner, facts in sorted(owners.items(), key = natural_order):
        print(scale_line(owner, facts))
    row_count = rows_not_decoded = recounts = recounts_that_differ = 0
    if owners:
        all_facts = sum(owners.values())
        _, row_count, scale_count, exception_count, coded_bits, rows_not_decoded = all_facts[:len(SCALE_FACTS)].tolist()
        saved_byte_count = scale_count - coded_bits / BITS_PER_BYTE
        print(scale_line(f"the experts of all {len(owners)}", all_facts))
        print(f"so coded, the scales' {scale_count / make_pack.BYTES_PER_GIB:.3f} GiB lose "
              f"{saved_byte_count / make_pack.BYTES_PER_GIB:.3f} GiB: "
              f"{PERCENT * saved_byte_count / expert_byte_count:.2f}% of the bytes of these experts")
        print(f"VERDICT figure=expert_scale_bits {scope} value={coded_bits / scale_count:.3f} "
              f"exceptions={exception_count / scale_count:.3e} saved_share={saved_byte_count / expert_byte_count:.4f} "
              f"verdict={result_verdict}")
    for kind, facts in sorted(kinds.items()):
        print(code_line(kind, facts))
    if kinds:
        all_facts = sum(kinds.values())
        print(code_line(f"the 8-bit matrices of all {len(kinds)} kinds", all_facts))
        _, code_count, block_bits, _, indexed_bits, recounts, recounts_that_differ = (
            all_facts[:len(CODE_FACTS)].tolist())
        print(f"VERDICT figure=trunk_code_bits_indexed {scope} value={indexed_bits / code_count:.3f} "
              f"ideal_by_block_table_not_counted={block_bits / code_count:.3f} verdict={result_verdict}")
    rows_as_required = rows_not_decoded == row_count if recoding_broken else not rows_not_decoded
    held = [check_line("scale_rows_decoded_back", rows_not_decoded, row_count, rows_as_required, recoding_broken),
            check_line("trunk_blocks_recounted", int(recounts_that_differ), int(recounts), bool(blocks_as_required),
                       blocks_broken)]
    print(f"pack_entropy_census: {len(chosen)} matrices in {time.monotonic() - started:.1f} s")
    return 0 if all(held) else EXIT_STATUS_FAILURE


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack_file")
    parser.add_argument("--matrices", default = "*", metavar = "PATTERN", help = "only the tensors it matches")
    parser.add_argument("--break-recoding", action = "store_true", help = "the first check must report every row")
    parser.add_argument("--break-blocks", action = "store_true", help = "the second must report one block a matrix")
    arguments = parser.parse_args()
    try:
        return census(arguments.pack_file, arguments.matrices, arguments.break_recoding, arguments.break_blocks)
    except (PackError, OSError) as failure:
        print(f"pack_entropy_census FAILED: {failure}", file = sys.stderr)
        return EXIT_STATUS_FAILURE


if __name__ == "__main__":
    sys.exit(main())
