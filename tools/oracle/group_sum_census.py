#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the 32 products of one group of a quantized linear need more than a float32, in the reference's captures.

    group_sum_census.py <pack file> <capture folder of --mode linears> [--kinds PATTERN]
                        [--break-scales] [--break-sum] [--break-recount] [--break-facts]

A product kernel adds the 32 products of one output and one block of 32 inputs before any scale (linear_records.h,
"A product"): a card in float32, by two tensor-core instructions of 16 products, the second on the sum of the first;
a processor kernel as one whole number that it converts to float32. Both operands are whole numbers of a unit (an
E2M1 weight in halves, an E4M3 weight or input in 2^-9), so the two have the same bits wherever no addition has to
round. For every captured linear the matrix is read from the pack and the exact sum of every group of every captured
row is formed in float64 (below 2^41 units: exact in any order). Counted per kind of linear and per weight format:
  at or above 2^24      sums of 2^24 units or more: what a debug build of the processor kernel counts (unit 8b).
                        With 8-bit weights the unit is 2^-18 and nearly every sum is one: read the others there;
  of no float32         sums that are no float32 number: converting the whole number rounds, and so must a card;
  first 16 no float32   groups whose first instruction's sum is none: today's kernel hands its second instruction a
                        rounded number, which a form that adds the 32 in one instruction does not;
  that may round        groups whose 32 products add up IN MAGNITUDE to 2^24 or more of their own unit, the largest
                        power of two that divides every one of them. In every other group each addition is exact in
                        any order (linear_records.h:46-48, at the largest unit its words allow): a card's sum IS the
                        whole number. Between the two counts above and this one the answer is the instruction's own
                        (unit 3a);
  and of the outputs (one of a row), those with a group that must round (either of the two counts above) and those
  with a group that may.
The gate judges every linear alone (linear_gate_card_test: 99.9% of ITS outputs) and the share of a kind can hide
one linear, so every line also says how many of its linears have more than 0.1% of their outputs with a group that
must round, how many with one that may, and the largest such share with the linear it lies in.

Checked in every run:
  - the sums, scaled by both sides' scale codes and added in float64, against the capture's outputs by
    check_pack_linears.py's comparison (with its tolerance: the reference adds in float32): the sums are those of
    the reference's own products. --break-scales: the scale codes of the weights are not read, and every linear
    must be reported;
  - every sum, bit for bit, against the same sum formed from check_pack_linears.py's tables of VALUES: the
    tolerance above cannot see a wrong small code, which moves these counts and hardly an output. --break-sum: one
    unit is added to the first of those sums of every linear, and that one sum of each must be reported;
  - in three groups of every linear (the first, the middle one, the last), for its first rows and outputs and for
    the place of the group's largest magnitudes: the first instruction's sum, the sum, the magnitudes, the unit and
    the four facts counted, against Python's whole numbers made from those tables of values. --break-recount: one
    is added to the first recounted number of every linear, and that one number of each must be reported;
  - in every group the facts counted must agree with each other: a group whose sum or whose first instruction's sum
    is no float32 is one that may round, and a sum that is no float32 is one of 2^24 units or more. --break-facts:
    one group more is counted as disagreeing in every linear, and that one of each must be reported.
The exit status is 0 only if every check found what the run asks of it: nothing, or with its flag exactly what the
flag says. A run with a flag is no result: its VERDICT lines say so.

Refused: what make_pack.read_pack_tables refuses of a pack and check_pack_linears.read_capture of a capture; a
captured linear whose weight is no matrix of that name, format and shape in the pack; a captured input or a matrix
of 8 bits that holds a code that is no number (no table of values has one); --kinds (fnmatch over the kinds,
"ffn.experts.*") that takes no captured linear.
What it does not show: that an instruction multiplies the placed codes exactly, or how it rounds (unit 3a); that its
first 16 products are the group's first 16 inputs (the kernel's header says so); anything beyond the capture, one
prompt of 435 tokens, five layers, about 320 rows of routed experts: a count of 0 is a rate below one in the groups
counted. Needs numpy. Exit status 0, 1 after a failure, 2 for a command line it cannot read.
Measurement tool: not part of the engine.
"""
import argparse
import collections
import fnmatch
import math
import os
import sys
import time

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "golden"))
import check_pack_linears as golden                        # the capture's reader and the comparison with its outputs

kernel_layouts, make_pack, PackError = golden.kernel_layouts, golden.make_pack, golden.PackError
GROUP_INPUTS = golden.INPUTS_PER_SCALE                     # 32: one scale of the vector, one multiply-add of a chain
INSTRUCTION_INPUTS = 16                                    # m16n8k16: the products one instruction adds
INSTRUCTIONS_PER_GROUP = 2                                 # what the recount cuts a group by
FLOAT32_WHOLE_BITS = 24                                    # every whole number below 2^24 is a float32 number
FLOAT32_WHOLE_LIMIT = 2.0 ** FLOAT32_WHOLE_BITS
E4M3_HIDDEN_BIT = golden.E4M3_MANTISSA_STEPS               # a normal byte: (8 + mantissa) << (exponent field - 1)
E4M3_UNIT_EXPONENT, E2M1_UNIT_EXPONENT = -9, -1            # an E4M3 byte counts in 2^-9, an E2M1 code in halves
E2M1_DOUBLED_MAGNITUDES = (0, 1, 2, 3, 4, 6, 8, 12)        # of the codes 0 to 7; bit 3 is the sign
WHOLE_NUMBER_DTYPE = numpy.int32                           # an E4M3 byte is at most 480 * 512 units
MARK_BITS = 6                                              # 2^6 is more than the 32 products of a group: product_units
UNIT_OF_ZEROS = 1                                          # of a group whose products are all 0: every unit divides it
FLOAT64_EXPONENT_SHIFT, FLOAT64_EXPONENT_BIAS, FLOAT64_EXPONENT_FIELDS = 52, 1023, 2048    # of a float64's bits
GATE_OUTPUT_SHARE = 0.001                                  # of a linear's outputs: what the gate lets differ
RECOUNTED_ROWS, RECOUNTED_OUTPUTS = 2, 16                  # the first of a linear, in each of its recounted groups
FORMAT_NAMES = {kernel_layouts.WEIGHT_FORMAT_E2M1_SCALED: "4-bit", kernel_layouts.WEIGHT_FORMAT_E4M3_SCALED: "8-bit"}
FACTS = ("groups", "groups at or above 2^24", "groups of no float32", "groups whose first 16 are no float32",
         "groups that may round",
         "outputs", "outputs with a group that must round", "outputs with a group that may round")
FACTS_OF_A_GROUP = FACTS.index("outputs")                  # census_of adds these up group by group
FACTS_BY_LINEAR = FACTS[FACTS_OF_A_GROUP + 1:]             # held to the gate's share linear by linear
PERCENT, FAILED_LINEARS_SHOWN, PROGRESS_INTERVAL_SECONDS = golden.PERCENT, golden.FAILED_LINEARS_SHOWN, 10.0

Census = collections.namedtuple("Census", ("counts", "largest_sum", "products", "differing_sums", "sum_count",
                                           "differing_recounts", "recount_count", "disagreeing_groups"))


def e4m3_whole_numbers():
    """What every E4M3 byte stands for in units of 2^-9, by its fields as the kernels place them: the mantissa alone
    under the exponent field 0, else the mantissa with its hidden bit, shifted by the field less 1."""
    codes = numpy.arange(golden.BYTE_VALUES, dtype = WHOLE_NUMBER_DTYPE)
    exponents = (codes >> golden.E4M3_MANTISSA_BITS) & golden.E4M3_EXPONENT_MASK
    mantissas = codes & (E4M3_HIDDEN_BIT - 1)
    normal_magnitudes = (mantissas + E4M3_HIDDEN_BIT) << numpy.maximum(exponents - 1, 0)
    magnitudes = numpy.where(exponents == 0, mantissas, normal_magnitudes)
    return numpy.where((codes & golden.E4M3_SIGN_BIT) != 0, -magnitudes, magnitudes)


def weight_whole_numbers(weight_format):
    """The whole number of every code of a weight format, by code, and the exponent of its unit."""
    if weight_format == kernel_layouts.WEIGHT_FORMAT_E4M3_SCALED:
        return e4m3_whole_numbers(), E4M3_UNIT_EXPONENT
    magnitudes = numpy.array(E2M1_DOUBLED_MAGNITUDES, dtype = WHOLE_NUMBER_DTYPE)
    return numpy.concatenate((magnitudes, -magnitudes)), E2M1_UNIT_EXPONENT


def lookups_of(whole_numbers):
    """Two tables by code, float64: the whole numbers themselves, and their marks for product_units: 64^-z for a
    number whose lowest set bit is 2^z, and 0 for 0."""
    marks = [math.ldexp(1.0, -MARK_BITS * ((number & -number).bit_length() - 1)) if number else 0.0
             for number in whole_numbers.tolist()]
    return whole_numbers.astype(numpy.float64), numpy.array(marks)


def units_by_exponent_field():
    """For product_units: the unit 2^u of a group whose marks add up to a float64 with each exponent field, that is
    to a sum from 2^(field - 1023) to under twice that: u is the least whole number with 64^-u at or under the sum.
    Field 0 is the sum 0."""
    floor_log2 = numpy.arange(FLOAT64_EXPONENT_FIELDS) - FLOAT64_EXPONENT_BIAS
    units = numpy.ldexp(1.0, numpy.maximum(-(floor_log2 // MARK_BITS), 0))
    units[0] = UNIT_OF_ZEROS
    return units


UNITS_BY_EXPONENT_FIELD = units_by_exponent_field()


def product_units(input_marks, weight_marks):
    """The largest power of two that divides every product of a row's 32 inputs and an output's 32 weights, float64
    [rows, outputs], from the marks of both ([rows, 32] and [outputs, 32]); UNIT_OF_ZEROS where every product is 0.
    A product's mark is 64^-(the zeros under its own lowest set bit), and the sum of a group's 32 marks lies from
    64^-u to 32 * 64^-u for the smallest u among them: below 64^-(u - 1) in whatever order it is added and rounded.
    The sum's exponent is read from its bits (it is not negative): frexp over a product of the trunk's size costs
    several times the matrix product itself."""
    mark_sums = input_marks @ weight_marks.T
    return UNITS_BY_EXPONENT_FIELD.take(mark_sums.view(numpy.int64) >> FLOAT64_EXPONENT_SHIFT)


def is_no_float32(sums):
    return sums.astype(numpy.float32).astype(numpy.float64) != sums


def is_float32_whole_number(whole_number):
    """Whether float32 holds a whole number of Python: at most 24 bits from its highest set bit to its lowest."""
    magnitude = abs(whole_number)
    return (magnitude // ((magnitude & -magnitude) or 1)).bit_length() <= FLOAT32_WHOLE_BITS


def recounted_facts(weight_values, input_values, weight_unit_exponent):
    """One output's 32 weights and one row's 32 inputs of a group once more: both as check_pack_linears' VALUES,
    made whole numbers of their units here and added as Python adds whole numbers. In the order of census_of's
    arrays: the first instruction's sum, the sum, the magnitudes, the unit, then the four facts counted."""
    products = [int(math.ldexp(weight, -weight_unit_exponent)) * int(math.ldexp(value, -E4M3_UNIT_EXPONENT))
                for weight, value in zip(weight_values, input_values)]
    first_sum, whole_sum = sum(products[:len(products) // INSTRUCTIONS_PER_GROUP]), sum(products)
    magnitude = sum(abs(product) for product in products)
    unit = min((product & -product for product in products if product), default = UNIT_OF_ZEROS)
    return (first_sum, whole_sum, magnitude, unit, abs(whole_sum) >= 1 << FLOAT32_WHOLE_BITS,
            not is_float32_whole_number(whole_sum), not is_float32_whole_number(first_sum),
            magnitude >= unit << FLOAT32_WHOLE_BITS)


def census_of(weight_format, codes, quantized_input, scales_of_group, sum_broken, recount_broken, facts_broken):
    """One linear, the codes of its matrix uint8 [outputs, inputs] and of its captured rows uint8 [rows, inputs]: a
    Census. counts: FACTS, int64; largest_sum: in units; products: float64 [rows, outputs], every group's sum times
    the two powers of two of scales_of_group(group), [rows, 1] and [1, outputs] (two exact multiplications: a
    ldexp over a product of the trunk's size costs ten of them); then what the last three checks of the text at the
    top found: the sums that differ, of how many; the recounted numbers that differ, of how many; and the groups
    whose facts disagree, of as many groups as there are sums."""
    weight_numbers, weight_unit_exponent = weight_whole_numbers(weight_format)
    (weight_table, weight_marks), (input_table, input_marks) = (lookups_of(numbers) for numbers
                                                                in (weight_numbers, e4m3_whole_numbers()))
    weight_values, input_values = golden.code_values(weight_format), golden.e4m3_values()
    unit = math.ldexp(1.0, weight_unit_exponent + E4M3_UNIT_EXPONENT)      # of a sum: it multiplies exactly
    group_count, shape = codes.shape[1] // GROUP_INPUTS, (len(quantized_input), len(codes))
    recounted_rows, recounted_outputs = min(RECOUNTED_ROWS, shape[0]), min(RECOUNTED_OUTPUTS, shape[1])
    counts, largest_sum, products = numpy.zeros(FACTS_OF_A_GROUP, dtype = numpy.int64), 0.0, numpy.zeros(shape)
    must_round, may_round = numpy.zeros(shape, dtype = bool), numpy.zeros(shape, dtype = bool)     # by output
    differing_sums = differing_recounts = recount_count = disagreeing_groups = 0
    for group in range(group_count):
        inputs = slice(group * GROUP_INPUTS, (group + 1) * GROUP_INPUTS)
        group_codes, group_input_codes = codes[:, inputs], quantized_input[:, inputs]
        group_weights, group_inputs = weight_table[group_codes], input_table[group_input_codes]
        first_sums = group_inputs[:, :INSTRUCTION_INPUTS] @ group_weights[:, :INSTRUCTION_INPUTS].T
        sums = first_sums + group_inputs[:, INSTRUCTION_INPUTS:] @ group_weights[:, INSTRUCTION_INPUTS:].T
        magnitudes = numpy.abs(group_inputs) @ numpy.abs(group_weights).T
        units = product_units(input_marks[group_input_codes], weight_marks[group_codes])
        sum_magnitudes = numpy.abs(sums)
        at_or_above, sum_rounds = sum_magnitudes >= FLOAT32_WHOLE_LIMIT, is_no_float32(sums)
        first_rounds, sum_may_round = is_no_float32(first_sums), magnitudes >= FLOAT32_WHOLE_LIMIT * units
        facts = (at_or_above, sum_rounds, first_rounds, sum_may_round)     # in the order of FACTS
        counts += [sums.size] + [int(numpy.count_nonzero(fact)) for fact in facts]
        sum_must_round = sum_rounds | first_rounds
        must_round |= sum_must_round
        may_round |= sum_may_round
        disagreeing_groups += (int(numpy.count_nonzero((sum_must_round & ~sum_may_round) | (sum_rounds & ~at_or_above)))
                               + (facts_broken and group == 0))
        largest_sum = max(largest_sum, float(sum_magnitudes.max()))
        row_scales, output_scales = scales_of_group(group)
        products += sums * row_scales * output_scales

        sums_again = input_values[group_input_codes] @ weight_values[group_codes].T        # in ones, not in units
        if sum_broken and group == 0:
            sums_again[0, 0] += unit
        differing_sums += int(numpy.count_nonzero(sums * unit != sums_again))
        if group not in (0, group_count // 2, group_count - 1):
            continue
        places = [(row, output) for row in range(recounted_rows) for output in range(recounted_outputs)]
        places.append(numpy.unravel_index(magnitudes.argmax(), shape))             # where a sum rounds first
        for place_index, (row, output) in enumerate(places):
            expected = recounted_facts(weight_values[group_codes[output]].tolist(),
                                       input_values[group_input_codes[row]].tolist(), weight_unit_exponent)
            if recount_broken and group == 0 and place_index == 0:
                expected = (expected[0] + 1,) + expected[1:]
            found = [array[row, output] for array in (first_sums, sums, magnitudes, units) + facts]
            differing_recounts += sum(bool(one != other) for one, other in zip(found, expected))
            recount_count += len(expected)
    counts = numpy.append(counts, [products.size, numpy.count_nonzero(must_round), numpy.count_nonzero(may_round)])
    return Census(counts, largest_sum, products, differing_sums, products.size * group_count, differing_recounts,
                  recount_count, disagreeing_groups)


def add_linear(total, name, counts, largest_sum):
    """One linear into the total of its kind or of its format, a dictionary: its linears, counts and largest sum,
    and by_linear, for each of FACTS_BY_LINEAR: the linears with more than the gate's share of their outputs, the
    largest share and the linear it lies in."""
    total["linears"], total["counts"] = total.get("linears", 0) + 1, total.get("counts", 0) + counts
    total["largest_sum"] = max(total.get("largest_sum", 0.0), largest_sum)
    by_linear = total.setdefault("by_linear", {fact: {"past": 0, "share": 0.0, "linear": name}
                                                for fact in FACTS_BY_LINEAR})
    for fact, count in zip(FACTS_BY_LINEAR, counts[-len(FACTS_BY_LINEAR):].tolist()):
        share = count / int(counts[FACTS_OF_A_GROUP])
        by_linear[fact]["past"] += share > GATE_OUTPUT_SHARE
        if share > by_linear[fact]["share"]:
            by_linear[fact].update(share = share, linear = name)


def print_total(weight_format, kind, total, kinds_pattern, verdict):
    """The line of one kind, or of a format with its VERDICT lines (kind None)."""
    name, unit_exponent = FORMAT_NAMES[weight_format], weight_whole_numbers(weight_format)[1] + E4M3_UNIT_EXPONENT
    linear_count, counts, largest_sum = total["linears"], total["counts"], total["largest_sum"]
    counted = dict(zip(FACTS, counts.tolist()))
    shares = [(fact, count, counted[fact.split()[0]]) for fact, count in counted.items() if " " in fact]
    by_linear = [(fact, worst["past"], worst["share"], worst["linear"]) for fact, worst in total["by_linear"].items()]
    print(f"{kind or f'all linears of {name} weights'}: {linear_count} linears, {counted['groups']} groups in "
          f"units of 2^{unit_exponent} and {counted['outputs']} outputs: "
          + ", ".join(f"{count} ({PERCENT * count / count_of:.5f}%) {fact}" for fact, count, count_of in shares)
          + f"; the largest sum 2^{math.log2(max(largest_sum, 1.0)):.2f} units; linears with more than "
          f"{PERCENT * GATE_OUTPUT_SHARE:g}% of their "
          + " and ".join(f"{fact}: {past} (the most {PERCENT * share:.5f}%, in {linear})"
                         for fact, past, share, linear in by_linear))
    if kind is not None:
        return
    scope = f"format={name} kinds={kinds_pattern}"
    for fact, count, count_of in shares:
        print(f"VERDICT figure={fact.replace(' ', '_')} {scope} value={count} of={count_of} "
              f"share={count / count_of:.3e} verdict={verdict}")
    for fact, past, share, linear in by_linear:
        print(f"VERDICT figure=linears_past_the_gate_by_{fact.replace(' ', '_')} {scope} value={past} "
              f"of={linear_count} largest_share={share:.3e} linear={linear} verdict={verdict}")


def check_line(name, differing, of, as_required, broken):
    """One CHECK line; as_required: whether the check found what this run asks of it. Returns as_required."""
    verdict = "FAILED" if not as_required else "reported" if broken else "ok"
    print(f"CHECK name={name} differing={differing} of={of} verdict={verdict}")
    return as_required


def census(pack_path, capture_folder, kinds_pattern, scales_broken, sum_broken, recount_broken, facts_broken):
    """Counts every captured linear the pattern takes and prints the lines of the text at the top; returns the exit
    status."""
    started = last_report = time.monotonic()
    _, extents, _, tensors = make_pack.read_pack_tables(pack_path)
    matrices = {tensor["name"].decode("utf-8", "replace"): tensor for tensor in tensors
                if tensor["weight_format"] != make_pack.WEIGHT_FORMAT_NONE}
    captured_linears, tensors_path = golden.read_capture(capture_folder)
    linears = [linear for linear in captured_linears if fnmatch.fnmatchcase(linear.kind, kinds_pattern)]
    if not linears:
        raise PackError(f"--kinds {kinds_pattern} takes none of the {len(captured_linears)} linears of "
                        f"{capture_folder}")
    totals, failed, sum_checks, recount_checks, fact_checks = {}, [], [], [], []
    with open(pack_path, "rb", buffering = 0) as pack, open(tensors_path, "rb", buffering = 0) as capture:
        for linear_index, linear in enumerate(linears):
            tensor, weight_format = matrices.get(linear.weight), golden.WEIGHT_FORMATS.get(linear.weight_dtype)
            quantized_input, input_scales, outputs = (golden.read_part(capture, linear.parts[part], tensors_path)
                                                      for part in golden.PARTS)
            output_size, input_size = tensor["dimensions"][:kernel_layouts.MATRIX_DIMENSIONS] if tensor else (0, 0)
            rows_expected = [(len(outputs), input_size), (len(outputs), input_size // GROUP_INPUTS),
                             (len(outputs), output_size)]
            if tensor is None or tensor["weight_format"] != weight_format \
                    or [quantized_input.shape, input_scales.shape, outputs.shape] != rows_expected:
                raise PackError(f"{linear.name}: the pack holds no matrix {linear.weight} of {linear.weight_dtype} "
                                f"with the {quantized_input.shape[1]} inputs and {outputs.shape[1]} outputs of its "
                                "captured rows")
            stored = bytearray(tensor["byte_count"])
            make_pack.read_exactly(pack, extents[tensor["extent_index"]]["byte_offset"]
                                   + tensor["byte_offset_in_extent"], memoryview(stored), pack_path)
            codes, scales = kernel_layouts.read_matrix(weight_format, output_size, input_size, stored)
            is_e4m3 = weight_format == kernel_layouts.WEIGHT_FORMAT_E4M3_SCALED
            for bytes_of_e4m3 in (quantized_input, codes) if is_e4m3 else (quantized_input,):
                if ((bytes_of_e4m3 & golden.E4M3_MAGNITUDE_MASK) == golden.E4M3_MAGNITUDE_MASK).any():
                    raise PackError(f"{linear.name}: its captured input or {linear.weight} holds a code that is no "
                                    "number")
            unit_exponent = weight_whole_numbers(weight_format)[1] + E4M3_UNIT_EXPONENT
            weight_scales = numpy.full_like(scales, golden.E8M0_BIAS) if scales_broken else scales
            weight_powers = numpy.ldexp(1.0, weight_scales.astype(numpy.int64) - golden.E8M0_BIAS + unit_exponent)
            if is_e4m3:                                        # the last block of 32 outputs: whole or half
                weight_powers = numpy.repeat(weight_powers, golden.E4M3_OUTPUTS_PER_SCALE, axis = 0)[:output_size]
            input_powers = numpy.ldexp(1.0, input_scales.astype(numpy.int64) - golden.E8M0_BIAS)
            counted = census_of(weight_format, codes, quantized_input,
                                lambda group: (input_powers[:, group, None], weight_powers[None, :, group]),
                                sum_broken, recount_broken, facts_broken)
            failure = golden.compare(counted.products, outputs).failure
            if failure:
                failed.append(f"{linear.name} ({linear.weight}): {failure}")
            sum_checks.append((counted.differing_sums, counted.sum_count))
            recount_checks.append((counted.differing_recounts, counted.recount_count))
            fact_checks.append((counted.disagreeing_groups, counted.sum_count))
            for key in ((weight_format, linear.kind), (weight_format, None)):          # the kind, and all of its format
                add_linear(totals.setdefault(key, {}), linear.name, counted.counts, counted.largest_sum)
            if time.monotonic() - last_report >= PROGRESS_INTERVAL_SECONDS:
                last_report = time.monotonic()
                print(f"  counted {linear_index + 1} of {len(linears)} linears", file = sys.stderr, flush = True)

    broken = scales_broken or sum_broken or recount_broken or facts_broken
    result_verdict = "not_a_result" if broken else "measured"
    in_order = sorted(totals.items(), key = lambda total: (total[0][0], total[0][1] is None, total[0][1] or ""))
    for (weight_format, kind), total in in_order:              # a format's kinds, then the format
        print_total(weight_format, kind, total, kinds_pattern, result_verdict)
    captured_as_required = len(failed) == len(linears) if scales_broken else not failed
    for failure in () if captured_as_required else failed[:FAILED_LINEARS_SHOWN]:
        print(f"FAILED: {failure}", file = sys.stderr)
    held = [check_line("sums_against_the_captured_outputs", len(failed), len(linears), captured_as_required,
                       scales_broken)]
    for name, found, broken in (("sums_equal_a_second_writing", sum_checks, sum_broken),
                                ("sample_recounted_in_whole_numbers", recount_checks, recount_broken),
                                ("facts_agree_in_every_group", fact_checks, facts_broken)):
        held.append(check_line(name, sum(differing for differing, _ in found), sum(of for _, of in found),
                               all(differing == int(broken) for differing, _ in found), broken))
    print(f"group_sum_census: {len(linears)} linears in {time.monotonic() - started:.1f} s")
    return 0 if all(held) else golden.EXIT_STATUS_FAILURE


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack_file")
    parser.add_argument("capture_folder")
    parser.add_argument("--kinds", default = "*", metavar = "PATTERN", help = "only the kinds of linear it matches")
    parser.add_argument("--break-scales", action = "store_true", help = "the first check must report every linear")
    parser.add_argument("--break-sum", action = "store_true", help = "the second check must report one sum a linear")
    parser.add_argument("--break-recount", action = "store_true", help = "the third must report one number a linear")
    parser.add_argument("--break-facts", action = "store_true", help = "the fourth must report one group a linear")
    arguments = parser.parse_args()
    try:
        return census(arguments.pack_file, arguments.capture_folder, arguments.kinds, arguments.break_scales,
                      arguments.break_sum, arguments.break_recount, arguments.break_facts)
    except (PackError, OSError) as failure:
        print(f"group_sum_census FAILED: {failure}", file = sys.stderr)
        return golden.EXIT_STATUS_FAILURE


if __name__ == "__main__":
    sys.exit(main())
