#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turns a text file into the tokens JSON that reference_oracle.py --tokens-json takes, with the checkpoint's own
tokenizer, cut into sequences of a given number of tokens: a longer, plain text for the logit comparisons of
batch_f03 and batch_f05.

    make_text_tokens.py --checkpoint CHECKPOINT --text FILE [FILE...] --tokens 2048 --output tokens.json
                        [--name NAME] [--chunks 8] [--without-begin-token] [--expected-sha256 HEX]

The text is encoded without any special token; unless --without-begin-token is given, the tokenizer's
begin-of-sequence token is put before it, because every prompt of the reference's encoder starts with it. The ids
are then cut, in order and without overlap, into --chunks sequences of --tokens tokens each (one by default); what
is left over is not used. With --without-begin-token --tokens 1024 --chunks 8 these are the contexts llama.cpp's
perplexity tool reads from a text for a model whose GGUF declares add_bos = false, which is how the author of the
GSQ-RCO file measured it. Several text files are each cut that way, one after another: --chunks sequences from
the start of every file (source files of one kind, for a set that is not one long text). The digest printed is the
SHA-256 of all the ids used, in order, as little-endian int32: what that author's validation records as
token_ids_sha256. With --expected-sha256 another digest is refused. Refused as well: a text of fewer tokens than
asked for, two texts of one file name (a sequence is named after its file), --name with more than one text, and
an output file that exists. The file written is a list of {"name", "token_ids"}, one entry per sequence:
reference_oracle.py takes it with --tokens-json and --sequence.
What it does not do: no chat format is applied, so the model reads the text as a bare continuation, which is what
a perplexity over a text wants and not what a served request looks like; the first token of a later chunk is not
replaced by the begin token, which llama.cpp does only for a model that declares add_bos = true.
Measurement tool: not part of the engine.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

TOKENS_SHOWN = 8
DIGEST_TOKEN_FORMAT = "<i"                   # one id of the digest: little-endian int32, as llama.cpp stores tokens
JSON_INDENT_SPACES = 1
EXIT_CODE_SUCCESS, EXIT_CODE_FAILURE = 0, 1


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev = False)
    parser.add_argument("--checkpoint", required = True, help = "the Hugging Face folder: its tokenizer is used")
    parser.add_argument("--text", required = True, nargs = "+", help = "UTF-8 text files")
    parser.add_argument("--tokens", required = True, type = int, help = "length of a sequence, begin token included")
    parser.add_argument("--chunks", type = int, default = 1, help = "number of sequences cut from the text, in order")
    parser.add_argument("--without-begin-token", action = "store_true", help = "the text's own tokens only")
    parser.add_argument("--expected-sha256", default = None, help = "refuse ids whose digest is another")
    parser.add_argument("--output", required = True, help = "the tokens JSON to write")
    parser.add_argument("--name", default = None, help = "name of the entry; the text file's name by default")
    arguments = parser.parse_args()
    names = [arguments.name] if arguments.name else [os.path.basename(path) for path in arguments.text]
    if arguments.tokens < 1 or arguments.chunks < 1 or os.path.exists(arguments.output) \
            or len(names) != len(arguments.text) or len(set(names)) != len(names):
        print("make_text_tokens FAILED: --tokens or --chunks is below 1, the output file exists already, --name is "
              "given for several texts, or two texts have one file name", file = sys.stderr)
        return EXIT_CODE_FAILURE
    from transformers import AutoTokenizer                                 # as the reference's generate.py:9
    tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint, local_files_only = True)
    if tokenizer.bos_token_id is None and not arguments.without_begin_token:
        print("make_text_tokens FAILED: the checkpoint's tokenizer names no begin-of-sequence token", file = sys.stderr)
        return EXIT_CODE_FAILURE
    begin = [] if arguments.without_begin_token else [tokenizer.bos_token_id]
    tokens_per_text = arguments.tokens * arguments.chunks
    token_ids, entries = [], []
    for name, path in zip(names, arguments.text):
        with open(path, encoding = "utf8", newline = "") as handle:      # the file's own line ends, untranslated
            text = handle.read()
        of_text = [int(token_id) for token_id in begin + tokenizer.encode(text, add_special_tokens = False)]
        if len(of_text) < tokens_per_text:
            print(f"make_text_tokens FAILED: {name} gives {len(of_text)} tokens, {tokens_per_text} were asked for",
                  file = sys.stderr)
            return EXIT_CODE_FAILURE
        of_text = of_text[:tokens_per_text]
        token_ids += of_text
        entries += [{"name": name if arguments.chunks == 1 else f"{name} chunk {chunk}",
                     "token_ids": of_text[chunk * arguments.tokens:(chunk + 1) * arguments.tokens]}
                    for chunk in range(arguments.chunks)]
        print(f"{name}: {arguments.chunks} sequences of {arguments.tokens} tokens, the first ids "
              f"{of_text[:TOKENS_SHOWN]}; the text used ends with {tokenizer.decode(of_text[-TOKENS_SHOWN:])!r}")
    tokens_used = len(token_ids)
    digest = hashlib.sha256(b"".join(struct.pack(DIGEST_TOKEN_FORMAT, token_id) for token_id in token_ids)).hexdigest()
    if arguments.expected_sha256 is not None and digest != arguments.expected_sha256.lower():
        print(f"make_text_tokens FAILED: the {tokens_used} ids have the digest {digest}, not the expected "
              f"{arguments.expected_sha256}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    with open(arguments.output, "x", encoding = "utf8") as handle:
        json.dump(entries, handle, indent = JSON_INDENT_SPACES)
    print(f"SHA-256 of the {tokens_used} ids as little-endian int32: {digest}"
          + ("" if arguments.expected_sha256 is None else ", as expected"))
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
