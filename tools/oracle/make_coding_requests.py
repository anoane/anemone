#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turns the nine C-coding requests of the earlier benchmarks into token-id requests for port_oracle_capture.py.

Each request is rendered the way the serving stack renders it (the chat template of the served pack, thinking on,
effort high) and tokenized with the model's tokenizer. For each request it also asks DeepSeek's reference encoder
for the same conversation and records whether the two prompts are the same text: the engine's own front end has
to match the reference encoder, so a difference here is a finding about today's serving stack.

    make_coding_requests.py --workloads-directory ~/exp4090/cmds --served-model-directory <EXL3 pack> \
        --reference-encoding-directory <official checkpoint>/encoding --output requests.json

Measurement tool: not part of the engine.
"""
import argparse
import copy
import json
import os
import sys

REFERENCE_THINKING_MODE = "thinking"
REFERENCE_REASONING_EFFORT = "high"
TOKENIZER_FILE_NAME = "tokenizer.json"
EXIT_CODE_USAGE = 2
DIFFERENCE_CONTEXT_CHARACTERS = 60


def first_difference(first_text, second_text):
    for position, (first_character, second_character) in enumerate(zip(first_text, second_text)):
        if first_character != second_character:
            return position
    return None if len(first_text) == len(second_text) else min(len(first_text), len(second_text))


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads-directory", required = True)
    parser.add_argument("--served-model-directory", required = True)
    parser.add_argument("--reference-encoding-directory", required = True)
    parser.add_argument("--output", required = True)
    arguments = parser.parse_args()
    for directory in (arguments.workloads_directory, arguments.served_model_directory, arguments.reference_encoding_directory):
        if not os.path.isdir(directory):
            print(f"not a directory: {directory}", file = sys.stderr)
            return EXIT_CODE_USAGE
    sys.dont_write_bytecode = True
    sys.path.insert(0, arguments.workloads_directory)
    sys.path.insert(0, arguments.reference_encoding_directory)
    import draft_workloads
    import encoding
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(arguments.served_model_directory, TOKENIZER_FILE_NAME))

    served_prompts = {name: text for name, _, text, _ in draft_workloads.render(arguments.served_model_directory)}
    requests, same_as_reference = [], 0
    for name, kind, messages, tools, new_token_limit in draft_workloads.requests():
        reference_messages = copy.deepcopy(messages)
        for message in reference_messages:                       # the reference takes tool-call arguments as JSON text
            for tool_call in message.get("tool_calls") or ():
                if not isinstance(tool_call["function"]["arguments"], str):
                    tool_call["function"]["arguments"] = json.dumps(tool_call["function"]["arguments"], ensure_ascii = False)
        if tools:
            reference_messages[0]["tools"] = tools
        try:
            reference_prompt = encoding.encode_messages(reference_messages, thinking_mode = REFERENCE_THINKING_MODE,
                                                        reasoning_effort = REFERENCE_REASONING_EFFORT)
            difference = first_difference(served_prompts[name], reference_prompt)
            comparison = "same text" if difference is None else (
                f"differs at character {difference}: served {served_prompts[name][difference:difference + DIFFERENCE_CONTEXT_CHARACTERS]!r} "
                f"reference {reference_prompt[difference:difference + DIFFERENCE_CONTEXT_CHARACTERS]!r}")
        except Exception as error:                                # a refusal by the reference is a finding too
            comparison = f"reference encoder refused: {type(error).__name__}: {error}"
        same_as_reference += comparison == "same text"
        token_ids = tokenizer.encode(served_prompts[name], add_special_tokens = False).ids
        requests.append({"name": name, "kind": kind, "new_token_limit_of_the_benchmark": new_token_limit,
                         "prompt_against_reference_encoder": comparison, "token_ids": token_ids})
        print(f"{name:12s} {len(token_ids):6d} tokens; against the reference encoder: {comparison[:150]}")
    with open(arguments.output, "w", encoding = "utf8") as handle:
        json.dump(requests, handle)
    print(f"{len(requests)} requests written to {arguments.output}; {same_as_reference} render to the reference encoder's text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
