#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the fork's engram layers spend fetching their rows: the figure the new engine's row lookup stands beside.

    run_with_port.sh fork_engram_lookup_time.py --layout <name> --gigabytes-per-device 58,60,95
        --ram-split 22..39:16 --tables ram|page_cache [--break <check>]

Loads the frozen exllamav3 port as port_load_only.py does (a cache of 1,048,576 tokens: the layout's limits are
sized for it), the two engram tables in host memory (ram: EXL3_NGRAM_STREAM=0) or read from their files through
the page cache (page_cache: the port's default, the recipe's way). It drives model.forward with the same corpus
tokens in two passes: a piece of 4,096 rows untimed, 4 timed (a prefill piece), 16 steps of one row untimed, 240
timed (a decode token). Tables that are files are put out of the page cache first: pass 1 reads its rows from the
drive, pass 2 from the page cache. Printed per pass and kind of step, summed over the engram layers and tables:
  BAR figure=engram_gather:..  ms a step inside RowTable._gather_rows: the read of the rows alone, on the host;
  BAR figure=engram_lookup:..  ms a step inside RowTable.lookup between two waits for the layer's card: the
                               hashing, that read, the upload and the decoding on the card;
  BAR figure=engram_rows:..    the distinct rows a step read, a row's values and its scales counted apart.
The lines begin with VOID, not BAR, where a check failed (exit status 1) or was broken on purpose. Checked in every
run, each shown failing by --break <name> (exit status 0 only if that check alone reported):
  tables_as_asked  every table is where --tables says (the break sets the port's variable the other way);
  lookups_counted  each engram layer looked its rows up once a step (the break times a function no step calls);
  first_pass_cold  where the tables are files, at most 16 MiB of them was in the page cache when pass 1 began
                   (the break reads 64 MiB of each back in); refused with --tables ram (exit status 2).
What it does not show: the Generator's steps (drafting, sampling); a lookup without the waits (a part that is waited for
is longer); the engram layers' linear and gate; another drive or file system. Measurement tool: not part of the engine.
"""
import argparse
import ctypes
import os
import sys
import time

import torch

from kernel_time_trace import DEFAULT_CORPUS_DIRECTORY, DEFAULT_MODEL_DIRECTORY, load_corpus_tokens, load_model

CACHE_TOKENS = 1048576
CACHE_PAGE_TOKENS = 256                      # the port's cache page: a prefill piece begins on one
PREFILL_PIECE_ROWS = 4096
UNTIMED_DECODE_STEPS = 16
PASS_STEPS = (                               # (kind, steps, rows a step) of a pass; a pass ends on a cache page
    ("untimed", 1, PREFILL_PIECE_ROWS),      # kernel picks, and the three rows whose n-grams span two passes
    (f"prefill_piece_{PREFILL_PIECE_ROWS}", 4, PREFILL_PIECE_ROWS),
    ("untimed", UNTIMED_DECODE_STEPS, 1),
    ("decode_token", CACHE_PAGE_TOKENS - UNTIMED_DECODE_STEPS, 1))
PASSES = 2
COLD_LIMIT_BYTES = 16 << 20
BREAK_READ_BYTES = 64 << 20
SYSCALL_CACHESTAT = 451                      # x86-64, Linux 6.5 on: (descriptor, range, counts, flags)
CACHESTAT_WORDS = (2, 5)                     # of its range (zeros: the whole file) and counts (the first: pages held)
CHECK_NAMES = ("tables_as_asked", "lookups_counted", "first_pass_cold")
MILLISECONDS_PER_SECOND = 1000.0


def timed(inner, totals, card = None):
    """inner, adding to totals its seconds (between two waits for card, if given), calls and first argument's size."""
    wait = (lambda: torch.cuda.synchronize(card)) if card is not None else (lambda: None)
    def call(*positional, **named):
        wait()
        started = time.perf_counter()
        result = inner(*positional, **named)
        wait()
        totals["seconds"] += time.perf_counter() - started
        totals["calls"] += 1
        totals["rows"] += positional[0].numel()                    # of _gather_rows: the rows it read
        return result
    return call


def page_cache_bytes(path, read_back_bytes):
    """Puts a file out of the page cache, reads its first read_back_bytes back in: what the kernel then holds."""
    whole_file, counts = ((ctypes.c_uint64 * words)() for words in CACHESTAT_WORDS)
    with open(path, "rb") as file:
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.pread(file.fileno(), read_back_bytes, 0)
        if ctypes.CDLL(None, use_errno = True).syscall(ctypes.c_long(SYSCALL_CACHESTAT), ctypes.c_int(file.fileno()),
                                                       whole_file, counts, ctypes.c_uint(0)):
            raise OSError(ctypes.get_errno(), f"cachestat of {path}")
    return counts[0] * os.sysconf("SC_PAGE_SIZE")


def measure(arguments):
    _, model, cache, tokenizer = load_model(arguments)
    embeddings = [module.embed for module in model.modules if type(module).__name__ == "EngramLayer"]
    tables = [table for embed in embeddings for table in (embed.table, embed.scale) if table is not None]
    gather, lookup = {}, {}
    for table in tables:
        table._gather_rows = timed(table._gather_rows, gather)
    timed_function = "fetch" if arguments.broken_check == "lookups_counted" else "lookup"
    for embed in embeddings:
        setattr(embed.table, timed_function, timed(getattr(embed.table, timed_function), lookup, embed.device))
    differing = {"tables_as_asked": (sum(table.on_disk != (arguments.tables == "page_cache") for table in tables),
                                     len(tables))}
    files = sorted({store.filename for table in tables if table.on_disk for store in table.stores})
    if files:
        read_back_bytes = BREAK_READ_BYTES if arguments.broken_check == "first_pass_cold" else 0
        cached_bytes = sum(page_cache_bytes(path, read_back_bytes) for path in files)
        print(f"TABLES files={len(files)} page_cache_bytes_before_pass_1={cached_bytes}", flush = True)
        differing["first_pass_cold"] = (int(cached_bytes > COLD_LIMIT_BYTES), 1)
    pass_tokens = sum(steps * rows for _, steps, rows in PASS_STEPS)
    token_ids = load_corpus_tokens(tokenizer, DEFAULT_CORPUS_DIRECTORY, pass_tokens).repeat(1, PASSES)
    state, position, lines, miscounted = cache.get_new_state(), 0, [], 0
    for pass_number in range(1, PASSES + 1):
        for kind, steps, rows in PASS_STEPS:
            for totals in (gather, lookup):
                totals.update(seconds = 0.0, calls = 0, rows = 0)
            for _ in range(steps):
                model.forward(token_ids[:, position:position + rows], {
                    "attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, token_ids.shape[1]),
                    "past_len": position, "recurrent_states": [state]})
                position += rows
            if kind == "untimed":
                continue
            miscounted += lookup["calls"] != steps * len(embeddings)
            for figure, value, unit in (("gather", MILLISECONDS_PER_SECOND * gather["seconds"] / steps, "ms"),
                                        ("lookup", MILLISECONDS_PER_SECOND * lookup["seconds"] / steps, "ms"),
                                        ("rows", gather["rows"] / steps, "rows")):
                lines.append(f"layout={arguments.layout} drafting=none figure=engram_{figure}:{arguments.tables}:"
                             f"pass_{pass_number}:{kind} value={value:.3f} unit={unit} n={steps}")
    differing["lookups_counted"] = (miscounted, PASSES * sum(kind != "untimed" for kind, _, _ in PASS_STEPS))
    failed = False
    for name, (count, of) in differing.items():
        must_report = arguments.broken_check == name
        verdict = "FAILED" if (count == 0 and of > 0) == must_report else "reported" if must_report else "ok"
        failed |= verdict == "FAILED"
        print(f"CHECK name={name} differing={count} of={of} verdict={verdict}")
    first_word = "VOID" if failed or arguments.broken_check else "BAR"
    print("\n".join(f"{first_word} {line}" for line in lines))
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", required = True, help = "its name, for the lines printed")
    parser.add_argument("--gigabytes-per-device", required = True, help = "the limit of each card, as the layout's")
    parser.add_argument("--ram-split", default = "0", help = "experts a layer kept in RAM, as the port's -mcs")
    parser.add_argument("--tables", choices = ("ram", "page_cache"), required = True)
    parser.add_argument("--break", dest = "broken_check", choices = CHECK_NAMES, help = "the check that must report")
    parser.set_defaults(model = DEFAULT_MODEL_DIRECTORY, ram_layers = "0", layers_per_device = None,
                        cache_tokens = CACHE_TOKENS)                 # load_model's
    arguments = parser.parse_args()
    if arguments.broken_check == "first_pass_cold" and arguments.tables == "ram":
        parser.error("--break first_pass_cold needs --tables page_cache: tables in RAM have no file to be cold")
    in_ram = (arguments.tables == "ram") != (arguments.broken_check == "tables_as_asked")
    os.environ["EXL3_NGRAM_STREAM"] = "0" if in_ram else "1"       # the port reads it when its config is built
    with torch.inference_mode():
        return measure(arguments)


if __name__ == "__main__":
    sys.exit(main())
