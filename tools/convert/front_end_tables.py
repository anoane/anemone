#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tokenizer's tables of Anemone's front end, from a checkpoint's tokenizer.json: what include/anemone/tokenizer.h
reads, and the step of make_pack.py's convert that puts them into the pack.

    front_end_tables.py write <tokenizer.json> <output file> [--overwrite]

write converts the one form of tokenizer.json this file knows and refuses any other, naming what differs: a byte-
level BPE model without dropout, unknown token, prefixes or suffixes, byte fallback or ignored merges; no normalizer
(an empty sequence); the pre-tokenizer that the engine's third split rewrites by hand, compared whole; byte-level
decoding. It then proves, before a byte is written:
  - every id below the vocabulary size has a token: an added token's content, or the model vocabulary's string;
    a token's bytes are what the reference's ByteLevel decoder makes of it alone (its characters mapped back to
    bytes when all of them are of the byte alphabet, else its UTF-8), at least one and at most 1,024 of them;
  - the 256 characters of the byte alphabet are tokens, and each one's bytes are its byte;
  - a merge joins two tokens into the token of their strings one after the other, whose bytes are theirs in that
    order, and no pair is merged twice (the reference would keep the last; the engine refuses);
  - an added token is no single word, strips nothing, and its bytes are its content, the bytes of all of them at most
    tokenizer.h's 1 MiB; the reference finds the raw added tokens first and then, in the text between them, the
    normalized ones: that one leftmost-longest look over all of them finds the same, because no raw token lies
    inside a normalized one and no end of a normalized token is the start of a raw one;
  - no code point is a number and of the second split's range at once.
The classes of every code point come from the reference's own regular expression engine: the tokenizers library
splits a text of all code points (surrogates left out: UTF-8 cannot hold them) on a pattern of each class, its
matches removed, and what is left are the code points outside the class. The patterns of the first two splits are
read from tokenizer.json itself; the others are the third split's \\p{L}, \\p{M}, [\\p{P}\\p{S}] and \\s.

The file is the header and the arrays of tokenizer.h, little-endian. In a pack it is the plain tensor
"front_end.tables" (U8, one dimension) of a host-only unit of its own: with_tables() below is convert's step, beside
engram_tables.with_tables (make_pack.py calls it before that one, so that the n-gram constants stay last).

Needs Python 3.9 and the tokenizers library (the oracle's environment). Exit status 0, 1 after any failure, 2 for a
command line it cannot read. Convert tool: not part of the engine.
"""
import argparse
import json
import os
import struct
import sys

MAGIC = b"ANEMTOKN"
FORMAT_VERSION = 1
HEADER = struct.Struct("<8sIIIIII8x")                      # struct anemone_tokenizer_tables_header, reserved zero
BYTE_VALUES = 256
CODE_POINTS = 0x110000
CLASS_BLOCK_CODE_POINTS = 256
SURROGATES = range(0xD800, 0xE000)
MAXIMUM_VOCABULARY = 2 ** 21
MAXIMUM_TOKEN_BYTES = 1024
MAXIMUM_ADDED_TOKEN_BYTES = 2 ** 20
TENSOR_NAME = "front_end.tables"
UNIT_LABEL = "front-end tables"
EXIT_STATUS_FAILURE, EXIT_STATUS_USAGE = 1, 2

# enum anemone_tokenizer_class
LETTER, MARK, NUMBER, SYMBOL, WHITESPACE, SECOND_SPLIT = (1 << bit for bit in range(6))
THIRD_SPLIT_CLASSES = ((LETTER, r"\p{L}"), (MARK, r"\p{M}"), (SYMBOL, r"[\p{P}\p{S}]"), (WHITESPACE, r"\s"))

# The one form this file converts. The third split is rewritten by hand in source/tokenizer.c; the patterns of the
# first two are read from here, the first must group at most three of its matches as tokenizer.c does.
THIRD_SPLIT_PATTERN = ("[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|"
                       "[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
                       " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+")
FIRST_SPLIT_PATTERN = "\\p{N}{1,3}"


def split(pattern):
    return {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False}


EXPECTED = {
    "normalizer": {"type": "Sequence", "normalizers": []},
    "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
    "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
              "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False},
}
SECOND_SPLIT_INDEX = 1
BYTE_LEVEL_STEP = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}


class TablesError(Exception):
    """A tokenizer.json this file does not convert, or tables that break a rule of tokenizer.h."""


def byte_alphabet():
    """The reference's byte-level alphabet: a character for each byte, printable ones as themselves."""
    kept = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    characters, extra = {}, 0
    for byte in range(BYTE_VALUES):
        if byte in kept:
            characters[byte] = chr(byte)
        else:
            characters[byte] = chr(BYTE_VALUES + extra)
            extra += 1
    return characters


def check_form(tokenizer):
    for key, expected in EXPECTED.items():
        found = tokenizer.get(key)
        if key == "model":
            found = {name: (found or {}).get(name) for name in expected}
        if found != expected:
            raise TablesError(f"{key} is {json.dumps(found)}; this converter knows {json.dumps(expected)}")
    if tokenizer["model"].get("ignore_merges"):
        raise TablesError("the model ignores merges: this converter knows BPE that merges every piece")
    steps = (tokenizer.get("pre_tokenizer") or {}).get("pretokenizers") or []
    second = steps[SECOND_SPLIT_INDEX]["pattern"].get("Regex", "") if len(steps) > SECOND_SPLIT_INDEX else ""
    expected_steps = [split(FIRST_SPLIT_PATTERN), split(second), split(THIRD_SPLIT_PATTERN), BYTE_LEVEL_STEP]
    if tokenizer.get("pre_tokenizer", {}).get("type") != "Sequence" or steps != expected_steps or \
            not (second.startswith("[") and second.endswith("]+") and "]" not in second[1:-2]):
        raise TablesError(f"the pre-tokenizer is {json.dumps(tokenizer.get('pre_tokenizer'), ensure_ascii = False)}; "
                          "this converter knows the three splits it compares whole, the second one character class")
    return second


def token_bytes_of(string, alphabet_bytes):
    """What the reference's ByteLevel decoder makes of a token alone."""
    if all(character in alphabet_bytes for character in string):
        return bytes(alphabet_bytes[character] for character in string)
    return string.encode("utf-8")


def code_point_classes(second_split_pattern):
    """The classes of every code point, as the reference's regular expression engine reads its patterns."""
    from tokenizers import Regex, pre_tokenizers           # the reference's library: imported only here
    code_points = [code_point for code_point in range(CODE_POINTS) if code_point not in SURROGATES]
    text = "".join(map(chr, code_points))
    classes = bytearray(CODE_POINTS)
    patterns = ((NUMBER, FIRST_SPLIT_PATTERN), (SECOND_SPLIT, second_split_pattern)) + THIRD_SPLIT_CLASSES
    for bit, pattern in patterns:
        splitter = pre_tokenizers.Split(Regex(pattern), behavior = "removed")
        left = {ord(character) for piece, _ in splitter.pre_tokenize_str(text) for character in piece}
        for code_point in code_points:
            if code_point not in left:
                classes[code_point] |= bit
    if any(value & NUMBER and value & SECOND_SPLIT for value in classes):
        raise TablesError("a code point is a number and of the second split's range: tokenizer.c cannot split it")
    return classes


def check_added_tokens(added, token_bytes):
    """The flags tokenizer.c does not implement, and the proof that one look finds what the reference's two find."""
    for token in added:
        if token["single_word"] or token["lstrip"] or token["rstrip"]:
            raise TablesError(f"the added token {token['content']!r} is a single word or strips: not converted")
        if token_bytes[token["id"]] != token["content"].encode("utf-8"):
            raise TablesError(f"the added token {token['content']!r} decodes to other bytes than its content")
    if sum(len(token_bytes[token["id"]]) for token in added) > MAXIMUM_ADDED_TOKEN_BYTES:
        raise TablesError(f"the added tokens hold more than {MAXIMUM_ADDED_TOKEN_BYTES} bytes together")
    raw = [token["content"].encode("utf-8") for token in added if not token["normalized"]]
    normalized = [token["content"].encode("utf-8") for token in added if token["normalized"]]
    for inner in normalized:
        for outer in raw:
            if outer in inner or any(inner.endswith(outer[:length])
                                     for length in range(1, min(len(outer), len(inner)))):
                raise TablesError(f"the raw added token {outer!r} overlaps the normalized {inner!r}: one look over "
                                  "both would not find what the reference's two find")


def tables_of(tokenizer_path):
    """The bytes of the tables, and a line that says what they hold."""
    with open(tokenizer_path, encoding = "utf8") as handle:
        tokenizer = json.load(handle)
    second_split_pattern = check_form(tokenizer)
    model, added = tokenizer["model"], tokenizer.get("added_tokens") or []
    strings = {token_id: string for string, token_id in model["vocab"].items()}
    strings.update({token["id"]: token["content"] for token in added})
    vocabulary_size = max(strings) + 1
    if len(strings) != vocabulary_size or min(strings) != 0 or vocabulary_size > MAXIMUM_VOCABULARY:
        raise TablesError(f"the ids are not 0 to {vocabulary_size - 1} each once, or more than {MAXIMUM_VOCABULARY}")
    alphabet = byte_alphabet()
    alphabet_bytes = {character: byte for byte, character in alphabet.items()}
    token_bytes = [token_bytes_of(strings[token_id], alphabet_bytes) for token_id in range(vocabulary_size)]
    if any(not 0 < len(value) <= MAXIMUM_TOKEN_BYTES for value in token_bytes):
        raise TablesError(f"a token has no byte or more than {MAXIMUM_TOKEN_BYTES}")
    vocabulary = model["vocab"]
    if any(alphabet[byte] not in vocabulary for byte in range(BYTE_VALUES)):
        raise TablesError("a character of the byte alphabet is no token")
    byte_token_ids = [vocabulary[alphabet[byte]] for byte in range(BYTE_VALUES)]
    merges, pairs = [], set()
    for merge in model["merges"]:
        left, right = merge.split(" ") if isinstance(merge, str) else merge
        result = vocabulary.get(left + right)
        if left not in vocabulary or right not in vocabulary or result is None:
            raise TablesError(f"the merge {left!r} {right!r} joins what is not in the vocabulary")
        triple = (vocabulary[left], vocabulary[right], result)
        if token_bytes[result] != token_bytes[triple[0]] + token_bytes[triple[1]] or triple[:2] in pairs:
            raise TablesError(f"the merge {left!r} {right!r} is not its tokens' bytes, or merges its pair again")
        pairs.add(triple[:2])
        merges.append(triple)
    check_added_tokens(added, token_bytes)
    classes = code_point_classes(second_split_pattern)
    blocks, block_of = {}, []
    for first in range(0, CODE_POINTS, CLASS_BLOCK_CODE_POINTS):
        block = bytes(classes[first:first + CLASS_BLOCK_CODE_POINTS])
        block_of.append(blocks.setdefault(block, len(blocks)))
    ends, total = [], 0
    for value in token_bytes:
        total += len(value)
        ends.append(total)
    added_ids = sorted(token["id"] for token in added)
    parts = [HEADER.pack(MAGIC, FORMAT_VERSION, vocabulary_size, len(merges), len(added_ids), len(blocks), total),
             struct.pack(f"<{vocabulary_size}I", *ends),
             struct.pack(f"<{3 * len(merges)}I", *(word for triple in merges for word in triple)),
             struct.pack(f"<{len(added_ids)}I", *added_ids), struct.pack(f"<{BYTE_VALUES}I", *byte_token_ids),
             struct.pack(f"<{len(block_of)}H", *block_of), b"".join(blocks), b"".join(token_bytes)]
    data = b"".join(parts)
    return data, (f"  front-end tables: {vocabulary_size} tokens, {len(merges)} merges, {len(added_ids)} added tokens, "
                  f"{len(blocks)} blocks of classes, {len(data)} bytes, from {tokenizer_path}")


def with_tables(units, sources, reference_directory):
    """make_pack.py convert's step: a host-only unit of one extent with the tensor TENSOR_NAME, from the
    tokenizer.json of reference_directory (none when it has no such file, as a converter's test folder). Returns
    the units, the sources and the line of convert's summary."""
    import engram_tables                                   # the pack converter's own helpers, beside this file
    import make_pack
    tokenizer_path = os.path.join(reference_directory, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        return units, sources, f"  front-end tables: none, {reference_directory} has no tokenizer.json"
    try:
        data, line = tables_of(tokenizer_path)
    except (TablesError, OSError, KeyError, ValueError) as error:
        raise make_pack.PackError(f"front-end tables: {error}") from error
    if TENSOR_NAME in sources:
        raise make_pack.PackError(f"the folder stores a tensor {TENSOR_NAME!r}: the name of the front-end tables")
    sources = dict(sources)
    sources[TENSOR_NAME] = make_pack.SourceTensor(engram_tables.DerivedBytes(data), 0, len(data), "U8", (len(data),))
    return list(units) + [make_pack.Unit(make_pack.UNIT_HOST_ONLY, make_pack.PACK_NO_LAYER, UNIT_LABEL,
                                         [TENSOR_NAME])], sources, line


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest = "command", required = True)
    write = commands.add_parser("write", help = "convert a tokenizer.json into a tables file")
    write.add_argument("tokenizer_json")
    write.add_argument("output")
    write.add_argument("--overwrite", action = "store_true")
    arguments = parser.parse_args()
    try:
        if os.path.lexists(arguments.output) and not arguments.overwrite:
            raise TablesError(f"{arguments.output} exists: --overwrite replaces it")
        data, line = tables_of(arguments.tokenizer_json)
        temporary = arguments.output + ".partial"
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, arguments.output)
    except (TablesError, OSError, KeyError, ValueError) as error:
        print(f"front_end_tables.py: {error}", file = sys.stderr)
        return EXIT_STATUS_FAILURE
    print(line.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
