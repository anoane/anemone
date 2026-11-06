#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Records the n-gram row ids that DeepSeek's reference computes for a stream of tokens: the golden file that
Anemone's anemone_engram_row_ids (include/anemone/engram.h) is compared with.

    make_engram_golden.py <checkpoint folder> <output file> --tokens-json FILE [--sequence I] [--split K]
    make_engram_golden.py <checkpoint folder> <output file> --random N [--seed S] [--split K]

The checkpoint folder is the published one: inference/engram.py, inference/config.json and the tokenizer's files.
engram.py is imported from it, through tools/convert/engram_tables.py and so exactly as the pack converter imports
it, and its NgramHashState is built on the CPU from the engram settings of config.json and, for the two runtime
limits of ModelArgs (model.py:51-52), one sequence of as many tokens as the stream. The state is called as the
model calls it (model.py:1249-1252): once for the whole stream as a prompt from position 0, with the mask of the
tokens that are not dead if any is dead, and with none otherwise. With --split K a second, fresh state is called as
generate.py drives the model: the first K tokens as the prompt, then every further token alone at its position
and without a mask. An image lies in the prompt only (generate.py:62), so a dead token at or behind K is refused.
Both ways must give the same ids, and every id must be a row of its table, or nothing is written.

The stream is entry --sequence (default 0) of --tokens-json: a list of token ids, or an object with "token_ids"
and, if tokens are dead (they lie inside an image), "dead_flags" with a 0 or a 1 for each token. The file may also
be one such entry alone. Or the stream is --random N: N tokens drawn with --seed (default 1), single uniformly
random ids mixed with runs of one repeated id and with stretches of dead tokens, whose ids are random too.

The output, little-endian, with nothing between its parts:
    header      32 bytes: the magic "ENGRAMID", uint32 format version 1, uint32 tables, uint32 rows per token,
                uint32 vocabulary (a token id is below it), uint64 tokens
    token ids   uint32 [tokens]
    dead flags  uint8 [tokens], 1 for a dead token
    row ids     int64 [tokens, tables, rows per token], in the order anemone_engram_row_ids gives them
It prints the counts and the SHA-256 of the file.

Needs Python 3.10, numpy, torch, sympy, tokenizers and transformers. Exit status 0, 1 after any failure, 2 for a
command line it cannot read.
Test tool: not part of the engine.
"""
import argparse
import contextlib
import hashlib
import json
import os
import random
import struct
import sys

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "convert"))
import engram_tables                                       # the reference, imported as the pack converter imports it

GOLDEN_MAGIC = b"ENGRAMID"
GOLDEN_FORMAT_VERSION = 1
GOLDEN_HEADER = struct.Struct("<8sIIIIQ")                  # magic, version, tables, rows per token, vocabulary, tokens
TOKEN_ID_TYPE = "<u4"
DEAD_FLAG_TYPE = "u1"
ROW_ID_TYPE = "<i8"
TOKEN_COUNT_LIMIT = 2 ** 24                                # the state's cache and the whole output are held in memory
SEQUENCES = 1                                              # ModelArgs.max_batch_size: one stream
DEFAULT_SEED = 1
RUN_CHANCE = 0.15                                          # of each draw of --random: a run of one repeated id
DEAD_CHANCE = 0.05                                         # and: a stretch of dead tokens
RUN_SHORTEST = 2
RUN_LONGEST = 6                                            # longer than any n-gram of the model
DEAD_LONGEST = 5
TEMPORARY_SUFFIX = ".partial"
EXIT_STATUS_FAILURE = 1
PackError = engram_tables.make_pack.PackError              # one line on the standard error stream, exit status 1


def read_stream(path, sequence_index):
    """The token ids and the dead flags of one entry of --tokens-json."""
    try:
        with open(path, encoding = "utf-8") as handle:
            content = json.load(handle)
    except (ValueError, RecursionError) as failure:
        raise PackError(f"{path} cannot be read: {failure}") from failure
    if isinstance(content, dict) or (isinstance(content, list) and all(type(item) is int for item in content)):
        content = [content]                                    # one entry alone
    if not isinstance(content, list) or sequence_index >= len(content):
        raise PackError(f"{path} holds no entry {sequence_index}")
    entry = content[sequence_index]
    token_ids = entry.get("token_ids") if isinstance(entry, dict) else entry
    if not engram_tables.make_pack.is_list_of_counts(token_ids) or not token_ids:
        raise PackError(f"{path}: entry {sequence_index} holds no list of token ids")
    dead_flags = entry.get("dead_flags", [0] * len(token_ids)) if isinstance(entry, dict) else [0] * len(token_ids)
    if not isinstance(dead_flags, list) or len(dead_flags) != len(token_ids) \
            or not all(type(flag) in (int, bool) and flag in (0, 1) for flag in dead_flags):
        raise PackError(f"{path}: the dead_flags of entry {sequence_index} are no 0 or 1 for each of its "
                        f"{len(token_ids)} tokens")
    return token_ids, [int(flag) for flag in dead_flags]


def random_stream(token_count, vocabulary_size, seed, dead_end):
    """The token ids and the dead flags of --random: single uniformly random ids, runs of one repeated id, and
    stretches of dead tokens with random ids, which lie before position dead_end."""
    generator, token_ids, dead_flags = random.Random(seed), [], []
    while len(token_ids) < token_count:
        draw, dead_room = generator.random(), dead_end - len(token_ids)
        if draw < RUN_CHANCE:
            token_ids += [generator.randrange(vocabulary_size)] * generator.randint(RUN_SHORTEST, RUN_LONGEST)
        elif draw < RUN_CHANCE + DEAD_CHANCE and dead_room > 0:
            stretch = min(generator.randint(1, DEAD_LONGEST), dead_room, token_count - len(token_ids))
            token_ids += [generator.randrange(vocabulary_size) for _ in range(stretch)]
            dead_flags += [1] * stretch
        else:
            token_ids.append(generator.randrange(vocabulary_size))
        dead_flags += [0] * (len(token_ids) - len(dead_flags))
    return token_ids[:token_count], dead_flags[:token_count]


def reference_row_ids(reference, settings, tokenizer, token_ids, dead_flags, prompt_token_count):
    """The row ids of a stream from a fresh NgramHashState of the reference, as a numpy array [tokens, tables, rows
    per token], with the state's layout and the size of its token map. The first prompt_token_count tokens are one
    call, as the model's pass over a prompt is (model.py:1249-1252: positional arguments, the mask of the tokens
    that are not dead, or None for a prompt without an image); each further token is a call of its own at its
    position and without a mask, as a pass of generate.py's loop is (generate.py:67-74)."""
    import torch                                               # open_reference has imported it, or has said why not
    layout = reference.EngramLayout.from_args(settings)        # model.py:1197
    state = reference.NgramHashState(settings, layout, tokenizer)                  # model.py:1198-1199, on the CPU
    ids = torch.tensor([token_ids], dtype = torch.long)
    is_alive = ~torch.tensor([dead_flags], dtype = torch.bool) if any(dead_flags) else None
    passes = [state(ids[:, :prompt_token_count], 0, None if is_alive is None else is_alive[:, :prompt_token_count])]
    for position in range(prompt_token_count, len(token_ids)):
        passes.append(state(ids[:, position:position + 1], position, None))
    return torch.cat(passes, dim = 1)[0].numpy(), layout, state.token_map.numel()


def make_golden(arguments):
    """Builds the stream, has the reference compute its row ids, checks them and writes the file."""
    reference, settings, tokenizer = engram_tables.open_reference(arguments.checkpoint_folder)
    vocabulary_size = len(tokenizer)                           # what engram.py's token map has an entry for
    if not vocabulary_size:
        raise PackError(f"{arguments.checkpoint_folder}: its tokenizer holds no token")
    if arguments.tokens_json is not None:
        token_ids, dead_flags = read_stream(arguments.tokens_json, arguments.sequence)
    else:
        dead_end = arguments.random if arguments.split is None else min(arguments.split, arguments.random)
        token_ids, dead_flags = random_stream(arguments.random, vocabulary_size, arguments.seed, dead_end)
    token_count, split = len(token_ids), arguments.split
    if token_count > TOKEN_COUNT_LIMIT or max(token_ids) >= vocabulary_size:
        raise PackError(f"the stream holds {token_count} tokens and the id {max(token_ids)}: at most "
                        f"{TOKEN_COUNT_LIMIT} tokens and ids below {vocabulary_size}")
    if split is not None and (split > token_count or any(dead_flags[split:])):
        raise PackError(f"--split {split}: the stream holds {token_count} tokens, and none at or behind the split "
                        "may be dead (an image lies in the prompt only)")

    settings.max_batch_size, settings.max_seq_len = SEQUENCES, token_count         # ModelArgs' two runtime limits
    try:
        row_ids, layout, map_size = reference_row_ids(reference, settings, tokenizer, token_ids, dead_flags,
                                                      token_count)
        split_row_ids = row_ids if split is None else reference_row_ids(reference, settings, tokenizer, token_ids,
                                                                        dead_flags, split)[0]
        table_count, rows_per_token = len(layout.layer_ids), (layout.max_ngram_size - 1) * layout.n_heads
        row_counts = numpy.array(layout.num_embeddings, dtype = numpy.int64).reshape(1, table_count, 1)
        is_sound = (map_size == vocabulary_size and row_ids.dtype == numpy.int64
                    and row_ids.shape == (token_count, table_count, rows_per_token)
                    and bool(((row_ids >= 0) & (row_ids < row_counts)).all()))
        is_same = numpy.array_equal(split_row_ids, row_ids)
        row_id_bytes = row_ids.astype(ROW_ID_TYPE).tobytes()
    except Exception as failure:                               # the reference's code: any failure is one line
        raise PackError(f"{arguments.checkpoint_folder}: its engram.py did not give the row ids: "
                        f"{engram_tables.failure_text(failure)}") from failure
    if not is_sound:
        raise PackError(f"the reference gave row ids of the shape {list(row_ids.shape)} for {token_count} tokens, "
                        f"{table_count} tables and {rows_per_token} rows per token, or an id that is no row of its "
                        f"table, or a token map of {map_size} entries for {vocabulary_size} token ids")
    if not is_same:
        raise PackError(f"--split {split}: the reference gives other row ids for a prompt of {split} tokens and "
                        "single tokens behind it than for the whole stream as one prompt")

    content = b"".join((
        GOLDEN_HEADER.pack(GOLDEN_MAGIC, GOLDEN_FORMAT_VERSION, table_count, rows_per_token, vocabulary_size,
                           token_count),
        numpy.array(token_ids, dtype = TOKEN_ID_TYPE).tobytes(),
        numpy.array(dead_flags, dtype = DEAD_FLAG_TYPE).tobytes(), row_id_bytes))
    temporary_path = arguments.output_file + TEMPORARY_SUFFIX
    try:
        with open(temporary_path, "wb") as handle:
            handle.write(content)
        os.replace(temporary_path, arguments.output_file)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary_path)
        raise
    print(f"{arguments.output_file}: {token_count} tokens ({sum(dead_flags)} dead), {table_count} tables, "
          f"{rows_per_token} rows per token, a vocabulary of {vocabulary_size}, {len(content)} bytes; "
          + ("not split" if split is None else f"split at {split}: the same row ids"))
    print(f"  sha256 {hashlib.sha256(content).hexdigest()}")


def number_from(smallest):
    """The type of an option that is a number from smallest to the most tokens a stream may hold."""
    def number(text):
        if not text.isascii() or not text.isdigit() or not smallest <= int(text) <= TOKEN_COUNT_LIMIT:
            raise argparse.ArgumentTypeError(f"{text!r} is no number from {smallest} to {TOKEN_COUNT_LIMIT}")
        return int(text)
    return number


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_folder")
    parser.add_argument("output_file")
    stream_arguments = parser.add_mutually_exclusive_group(required = True)
    stream_arguments.add_argument("--tokens-json", metavar = "FILE", help = "the stream: token ids, dead flags")
    stream_arguments.add_argument("--random", type = number_from(1), metavar = "N", help = "the stream: N tokens drawn")
    parser.add_argument("--sequence", type = number_from(0), default = 0, metavar = "I",
                        help = "the entry of --tokens-json, from 0")
    parser.add_argument("--seed", type = number_from(0), default = DEFAULT_SEED, metavar = "S", help = "of --random")
    parser.add_argument("--split", type = number_from(1), metavar = "K",
                        help = "also as a prompt of K tokens and single tokens behind it")
    arguments = parser.parse_args()
    try:
        make_golden(arguments)
    except (PackError, OSError) as failure:
        print(f"make_engram_golden FAILED: {failure}", file = sys.stderr)
        return EXIT_STATUS_FAILURE
    return 0


if __name__ == "__main__":
    sys.exit(main())
