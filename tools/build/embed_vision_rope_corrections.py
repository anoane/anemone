#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build-only: convert the qualified packed RoPE data into a numeric initializer under build/.

Usage: embed_vision_rope_corrections.py <input .hex> <output .inc>
The committed input contains data only. The generated initializer is never checked in.
"""
import hashlib
import os
from pathlib import Path
import struct
import sys

POSITIONS, FREQUENCIES, FUNCTIONS = 3063, 16, 2
BITS_PER_VALUE, WORD_BITS, WORD_BYTES = 3, 32, 4
HEX_CHARACTERS = "0123456789abcdef"
HEX_WORD_DIGITS = WORD_BYTES * 2
NEGATIVE_ZERO, VALUE_MASK = 4, 7
VALUES = POSITIONS * FREQUENCIES * FUNCTIONS
WORDS = VALUES * BITS_PER_VALUE // WORD_BITS
WORDS_PER_LINE = 8


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    source, output = (Path(value) for value in sys.argv[1:])
    expected_bytes = WORDS * (HEX_WORD_DIGITS + 1)
    if source.stat().st_size != expected_bytes:
        raise ValueError(f"{source}: expected {expected_bytes} canonical hexadecimal text bytes")
    payload = source.read_bytes()
    records = payload.decode("ascii").splitlines()
    if len(records) != WORDS or any(len(record) != HEX_WORD_DIGITS or
            any(letter not in HEX_CHARACTERS for letter in record) for record in records):
        raise ValueError(f"{source}: expected {WORDS} lines of eight lowercase hexadecimal digits")
    if payload != ("\n".join(records) + "\n").encode("ascii"):
        raise ValueError(f"{source}: expected canonical LF-terminated hexadecimal words")
    words = tuple(int(record, 16) for record in records)
    for index in range(VALUES):
        word, shift = divmod(index * BITS_PER_VALUE, WORD_BITS)
        code = words[word] >> shift
        if shift > WORD_BITS - BITS_PER_VALUE:
            code |= words[word + 1] << (WORD_BITS - shift)
        if code & VALUE_MASK == NEGATIVE_ZERO:
            raise ValueError(f"{source}: noncanonical negative-zero correction at value {index}")
    lines = ["/* SPDX-License-Identifier: AGPL-3.0-or-later */",
             "/* Generated numeric data; never edit or check in this build product.",
             f" * Input text SHA-256: {hashlib.sha256(payload).hexdigest()}",
             f" * Packed data SHA-256: {hashlib.sha256(struct.pack('<' + 'I' * WORDS, *words)).hexdigest()} */"]
    lines.extend(["__device__ __constant__ static const uint32_t",
                  "    vision_rope_corrections[VISION_ROPE_CORRECTION_WORDS] = {"])
    for first in range(0, WORDS, WORDS_PER_LINE):
        lines.append("\t" + ", ".join(f"0x{word:08X}u" for word in words[first:first + WORDS_PER_LINE]) + ",")
    lines.append("};")
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
