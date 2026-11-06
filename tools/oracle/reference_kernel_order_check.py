#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What is wrong with the activation quantization kernels of DeepSeek's reference, kernel by kernel.

reference_divergence.py found that two passes over one prompt first differ at the output of the reference's own
act_quant: same input, other bytes. This tool looks at the kernels alone, on random input of the model's widths.

Without an option, three tests of act_quant returning new tensors, at several row counts:
  order    the kernel is queued where PyTorch's next operation does not wait for it. Test: a copy of the output
           taken at once (queued by PyTorch right after the call) against the output read after the card has been
           waited for. They differ only if the copy ran before the kernel had written everything.
  bounds   the kernel takes 32 rows at a time; with a row count that is not a multiple of 32 it could read and
           write behind its tensors. Test: the kernel is given the first rows of larger buffers whose remaining
           rows hold a pattern; the pattern must be untouched. It shows a write behind the rows; a read behind
           them would not show.
  repeat   the kernel itself gives other bytes for the same input. Test: the settled outputs of repeated calls.
--wait-before-reading waits for the card before the copy of the order test is taken: a difference seen only
without it is one of order. Where neither run shows a difference the two cases have not been told apart.

    reference_kernel_order_check.py <checkpoint>/inference [--calls N] [--wait-before-reading]
    reference_kernel_order_check.py <checkpoint>/inference --describe ROWS [--calls N] [--pass-config KEY=true|false]...
    reference_kernel_order_check.py <checkpoint>/inference --in-place [--calls N]
    reference_kernel_order_check.py <checkpoint>/inference --sources FOLDER

--describe ROWS compares every call of act_quant with its definition computed by PyTorch
(reference_quantization.py) and tests the account of the fault given there: every wrong byte must lie in rows 8
to 31 of its 32-row tile and be exactly the input byte that lies at its place in the kernel's shared memory.
--in-place does the same for the three in-place quantizations as the model calls them (the attention's keys, the
indexer's keys and queries, the compressed keys), whose definitions leave no room for a fault: every bit must be
the definition's. From two rows on, the input holds a row of zeros and 32 tiny negative values beside the random
ones; a single row is random.
--sources writes the CUDA source the reference's compiler generated for the four kernels and prints the lines
that touch shared memory. --pass-config adds an option of the compiler to the ones kernel.py passes
(kernel.py:9-12) before any kernel is compiled; the kernel's source is not changed.
Exit code 0 when nothing differed from what the mode expects: all three tests passed, every byte was the
definition's (--describe: so a kernel with the fault ends with 1), every bit was (--in-place).
Measurement tool: not part of the engine.
"""
import argparse
import json
import os
import sys

import torch

import reference_quantization as quantization

ROW_COUNTS = (1, 31, 32, 33, 64, 435, 448, 4096)
IN_PLACE_ROW_COUNTS = (1, 4, 31, 33, 435, 4096, 27840)       # the last: 435 positions of 64 heads, the indexer's
BLOCK_SIZE = 32                              # model.py:27-28, fp8_block_size and fp4_block_size
COMPRESSED_KEY_BLOCK_SIZE = 16               # model.py:760
SCALE_FORMAT = "ue8m0"                       # model.py:29: any value but None rounds scales up to a power of two
GUARD_ROWS = 64                              # two whole steps of the kernel behind the rows it is given
GUARD_BYTE = 0x5A
DEFAULT_CALLS = 20
SEED = 20261010
INPUT_SPREAD = 3.0
TINY_NEGATIVE = -1.0e-30                     # below the block floors of act_quant (1e-4) and of the compressed keys
                                             # (6 * 2**-9): a block of it rounds to zero there, and the sign of that
                                             # zero is on trial. Above the indexer's floor (6 * 2**-126): there the
                                             # block becomes -6 * 2**-102, and the zeros on trial are the random
                                             # values within a quarter of their block's scale
WRONG_BYTES_PRINTED = 8
PERCENT = 100.0
MODEL_ARGUMENTS_FILE_NAME = "config.json"
SHARED_MEMORY_WORDS = ("shmem", "__syncthreads", "__shared__")
EXIT_CODE_PASSED, EXIT_CODE_NOT_PASSED = 0, 1


def bits(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def random_rows(row_count, width):
    return (torch.randn(row_count, width, dtype = torch.float32, device = "cuda") * INPUT_SPREAD).to(torch.bfloat16)


def with_edge_values(rows):
    """From two rows on: row 0 becomes zeros and the first 32 values of row 1 tiny negative ones. A single row stays
    random: as zeros it would be the whole input, and zeros come out of the quantization whatever the scale is."""
    if rows.shape[0] > 1:
        rows[0] = 0.0
        rows[1, :BLOCK_SIZE] = TINY_NEGATIVE
        rows[1, BLOCK_SIZE:BLOCK_SIZE + BLOCK_SIZE] = -0.0             # and the next 32 values zeros with a sign
    return rows


def describe(kernel, row_count, width, calls):
    """Every call of act_quant against the definition, and every wrong byte against the input byte the account of
    the fault predicts at its place. True when no byte of any call was wrong."""
    rows = random_rows(row_count, width)
    defined_quantized, defined_scales = quantization.defined_act_quant(rows, BLOCK_SIZE)
    defined_bytes = defined_quantized.view(torch.uint8)
    print(f"{row_count} rows of {width} against the definition, {calls} calls:")
    every_call_as_defined, wrong_in_all, unexplained_in_all, calls_wrong = True, 0, 0, 0
    for call in range(calls):
        quantized, scales = kernel.act_quant(rows, BLOCK_SIZE, SCALE_FORMAT, torch.float8_e8m0fnu)
        torch.cuda.synchronize()
        wrong = (quantized.view(torch.uint8) != defined_bytes).nonzero()
        wrong_scales = int((scales.view(torch.uint8) != defined_scales.view(torch.uint8)).sum())
        _, unexplained = quantization.unexplained_bytes(rows, quantized, defined_quantized, BLOCK_SIZE)
        line = f"  call {call:2d}: {len(wrong)} of {quantized.numel()} quantized bytes " \
               f"({PERCENT * len(wrong) / quantized.numel():.2f}%) and {wrong_scales} scale bytes are not the " \
               "definition's"
        if len(wrong):
            wrong_rows, wrong_columns = wrong[:, 0], wrong[:, 1]
            tiles = torch.unique(wrong_rows // BLOCK_SIZE * (width // BLOCK_SIZE) + wrong_columns // BLOCK_SIZE)
            row_groups = torch.unique(wrong_rows % BLOCK_SIZE // quantization.TILE_ROWS_NEVER_OVERWRITTEN).tolist()
            line += f", in {len(tiles)} of the kernel's tiles, in the 8-row groups {row_groups} of a tile; " \
                    f"{unexplained} of them are NOT the input byte predicted at their place; the first: " + ", ".join(
                        f"[{row},{column}] {int(quantized.view(torch.uint8)[row, column]):#04x} for "
                        f"{int(defined_bytes[row, column]):#04x}"
                        for row, column in wrong[:WRONG_BYTES_PRINTED].tolist())
        every_call_as_defined = every_call_as_defined and len(wrong) == 0 and wrong_scales == 0
        wrong_in_all, unexplained_in_all = wrong_in_all + len(wrong), unexplained_in_all + unexplained
        calls_wrong += int(len(wrong) > 0)
        print(line, flush = True)
    print(f"  in all: {calls_wrong} of {calls} calls with wrong bytes; {wrong_in_all} wrong bytes, of which "
          f"{wrong_in_all - unexplained_in_all} are the predicted input byte in rows 8 to 31 of a tile and "
          f"{unexplained_in_all} are not")
    return every_call_as_defined


def in_place(kernel, model_arguments, calls):
    """The three in-place quantizations as model.py calls them, against their definitions. True when every bit of
    every call was the definition's."""
    cases = (
        ("act_quant in place, the attention's keys (model.py:707)", model_arguments.head_dim,
         lambda rows: kernel.act_quant(rows, BLOCK_SIZE, SCALE_FORMAT, torch.float8_e8m0fnu, True),
         lambda rows: quantization.defined_act_quant_in_place(rows, BLOCK_SIZE)),
        ("fp4_act_quant in place, the indexer (model.py:546, :552)", model_arguments.index_head_dim,
         lambda rows: kernel.fp4_act_quant(rows, BLOCK_SIZE, True),
         lambda rows: quantization.defined_fp4_act_quant_in_place(rows, BLOCK_SIZE, torch.float8_e8m0fnu)),
        ("fp4_act_quant in place, the compressed keys (model.py:760)", model_arguments.head_dim,
         lambda rows: kernel.fp4_act_quant(rows, COMPRESSED_KEY_BLOCK_SIZE, True, scale_dtype = torch.float8_e4m3fn),
         lambda rows: quantization.defined_fp4_act_quant_in_place(rows, COMPRESSED_KEY_BLOCK_SIZE,
                                                                  torch.float8_e4m3fn)))
    every_bit_as_defined = True
    for name, width, quantize, define in cases:
        print(f"{name}, rows of {width}:")
        for row_count in IN_PLACE_ROW_COUNTS:
            rows = with_edge_values(random_rows(row_count, width))
            defined = define(rows)
            calls_differing, values_differing, rows_in_tile, first = 0, 0, set(), None
            for _ in range(calls):
                work = rows.clone()
                quantize(work)
                torch.cuda.synchronize()
                differing = (work.view(torch.int16) != defined.view(torch.int16)).nonzero()
                if len(differing):
                    calls_differing, values_differing = calls_differing + 1, values_differing + len(differing)
                    rows_in_tile.update(torch.unique(differing[:, 0] % quantization.TILE_ROWS).tolist())
                    first = first or [(row, column, float(rows[row, column]), float(work[row, column]),
                                       float(defined[row, column]))
                                      for row, column in differing[:WRONG_BYTES_PRINTED].tolist()]
            line = f"  {row_count:5d} rows: {calls_differing} of {calls} calls left other bits than the definition"
            if calls_differing:
                line += f" ({values_differing} values, rows {sorted(rows_in_tile)} of a tile); the first as [row, " \
                        f"column] input -> kernel, definition: " + ", ".join(
                            f"[{row},{column}] {value!r} -> {left!r}, {defined_value!r}"
                            for row, column, value, left, defined_value in first)
            every_bit_as_defined = every_bit_as_defined and calls_differing == 0
            print(line, flush = True)
    return every_bit_as_defined


def write_sources(kernel, model_arguments, folder):
    """The generated source of the four quantization kernels, and their lines that touch shared memory."""
    os.makedirs(folder, exist_ok = True)
    compiled = {
        "act_quant_new_tensors": kernel.act_quant_kernel(model_arguments.dim, BLOCK_SIZE, scale_dtype = kernel.FE8M0,
                                                         round_scale = True, inplace = False),
        "act_quant_in_place": kernel.act_quant_kernel(model_arguments.head_dim, BLOCK_SIZE, scale_dtype = kernel.FE8M0,
                                                      round_scale = True, inplace = True),
        "fp4_act_quant_in_place_indexer": kernel.fp4_quant_kernel(model_arguments.index_head_dim, BLOCK_SIZE,
                                                                  scale_dtype = kernel.FE8M0, inplace = True),
        "fp4_act_quant_in_place_compressed_keys": kernel.fp4_quant_kernel(
            model_arguments.head_dim, COMPRESSED_KEY_BLOCK_SIZE, scale_dtype = kernel.FP8, inplace = True)}
    for name, kernel_object in compiled.items():
        source = kernel_object.get_kernel_source()
        with open(os.path.join(folder, name + ".cu"), "w", encoding = "utf8") as handle:
            handle.write(source)
        print(f"{name}.cu: {len(source.splitlines())} lines; the ones that touch shared memory:")
        for number, line in enumerate(source.splitlines(), start = 1):
            if any(word in line for word in SHARED_MEMORY_WORDS):
                print(f"  {number:4d}: {' '.join(line.split())[:150]}")


def three_tests(kernel, width, calls, wait_before_reading):
    scale_dtype = torch.float8_e8m0fnu
    passed = True
    for row_count in ROW_COUNTS:
        rows = random_rows(row_count, width)
        early_differs, repeat_differs, first_settled = 0, 0, None
        bytes_differing_early = 0
        for _ in range(calls):
            quantized, scales = kernel.act_quant(rows, BLOCK_SIZE, SCALE_FORMAT, scale_dtype)
            if wait_before_reading:
                torch.cuda.synchronize()
            early = (bits(quantized).clone(), bits(scales).clone())     # queued by PyTorch right behind the call
            torch.cuda.synchronize()
            settled = (bits(quantized).clone(), bits(scales).clone())
            differing = sum(int((first != second).sum()) for first, second in zip(early, settled))
            early_differs += int(differing > 0)
            bytes_differing_early += differing
            if first_settled is None:
                first_settled = settled
            repeat_differs += int(not all(torch.equal(first, second) for first, second in zip(first_settled, settled)))
        # bounds: the compiled kernel on the first rows of larger buffers
        compiled = kernel.act_quant_kernel(width, BLOCK_SIZE, scale_dtype = kernel.FE8M0, round_scale = True,
                                           inplace = False)
        wide_rows = torch.full((row_count + GUARD_ROWS, width), 1.0e4, dtype = torch.bfloat16, device = "cuda")
        wide_rows[:row_count] = rows
        wide_quantized = torch.empty(row_count + GUARD_ROWS, width, dtype = torch.float8_e4m3fn, device = "cuda")
        wide_scales = torch.empty(row_count + GUARD_ROWS, width // BLOCK_SIZE, dtype = scale_dtype, device = "cuda")
        bits(wide_quantized).fill_(GUARD_BYTE)
        bits(wide_scales).fill_(GUARD_BYTE)
        compiled(wide_rows[:row_count], wide_quantized[:row_count], wide_scales[:row_count])
        torch.cuda.synchronize()
        guard_bytes_written = int((bits(wide_quantized[row_count:]) != GUARD_BYTE).sum()) \
            + int((bits(wide_scales[row_count:]) != GUARD_BYTE).sum())
        print(f"{row_count:5d} rows: order: {early_differs:2d} of {calls} calls read early differ from the settled "
              f"output ({bytes_differing_early} bytes); repeat: {repeat_differs} settled outputs differ from the "
              f"first; bounds: {guard_bytes_written} bytes written behind the rows", flush = True)
        passed = passed and early_differs == 0 and repeat_differs == 0 and guard_bytes_written == 0
    print("the reference's kernel passed the three tests" if passed else
          "the reference's kernel did NOT pass: see the lines above")
    return passed


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inference_folder")
    parser.add_argument("--calls", type = int, default = DEFAULT_CALLS)
    parser.add_argument("--wait-before-reading", action = "store_true")
    parser.add_argument("--describe", type = int, default = None, metavar = "ROWS")
    parser.add_argument("--in-place", action = "store_true")
    parser.add_argument("--sources", default = None, metavar = "FOLDER")
    parser.add_argument("--pass-config", action = "append", default = [], metavar = "KEY=true|false")
    arguments = parser.parse_args()
    sys.dont_write_bytecode = True              # nothing is written into the checkpoint
    sys.path.insert(0, arguments.inference_folder)
    import kernel                               # the reference's kernel.py
    import model                                # and its model.py, for the widths of the checkpoint
    for option in arguments.pass_config:
        key, _, value = option.partition("=")
        if value not in ("true", "false"):
            parser.error(f"--pass-config {option}: KEY=true or KEY=false is expected")
        kernel.pass_configs[key] = value == "true"
    with open(os.path.join(arguments.inference_folder, MODEL_ARGUMENTS_FILE_NAME), encoding = "utf8") as handle:
        model_arguments = model.ModelArgs(**json.load(handle))
    torch.set_default_dtype(torch.bfloat16)     # generate.py:118
    torch.manual_seed(SEED)
    print(f"card: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; {arguments.calls} calls per case; "
          f"compiler options beside kernel.py's own: {arguments.pass_config or 'none'}"
          f"{'; the card is waited for before every read' if arguments.wait_before_reading else ''}")
    if arguments.sources is not None:
        write_sources(kernel, model_arguments, arguments.sources)
        return EXIT_CODE_PASSED
    if arguments.describe is not None:
        passed = describe(kernel, arguments.describe, model_arguments.dim, arguments.calls)
    elif arguments.in_place:
        passed = in_place(kernel, model_arguments, arguments.calls)
    else:
        passed = three_tests(kernel, model_arguments.dim, arguments.calls, arguments.wait_before_reading)
    return EXIT_CODE_PASSED if passed else EXIT_CODE_NOT_PASSED


if __name__ == "__main__":
    sys.exit(main())
