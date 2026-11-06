#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The golden sets of Anemone's front end, from the checkpoint's own tokenizer and encoder: what
tests/front_end_golden_test.c holds the tokenizer (include/anemone/tokenizer.h) and the chat template
(include/anemone/chat_template.h) to.

    make_front_end_golden.py --checkpoint <folder> --output-directory <folder> [--text <file> ...] [--overwrite]

tokenizer_golden.bin holds texts and the ids tokenizer.json gives them (tokenizers' Tokenizer.encode, no special
token added), and the text the ids decode to. The texts: every --text file (the engine's sources, prose, the held-out
text of the oracle, code in other languages), each whole; every added token alone, all of them in a row, inside
text and cut short; ill-formed UTF-8, whose reference text is Python's bytes.decode(errors="replace"); long runs of
one character; thousands of short random strings of whitespace, digits, marks, symbols and scripts, and of random
code points of every plane; emoji and CJK; and last a prompt of at least 1,048,576 tokens made of the --text files.
Each text's ids decode, by the reference's decode with special tokens kept, to its reference text, or nothing is
written.

template_golden.bin holds conversations and the prompt the encoder (the checkpoint's encoding/encoding.py) renders,
with its ids. The conversations: the encoder's own test pairs (encoding/tests: each must render to its shipped
output, or nothing is written), the requests of make_golden_set.py beside this file, and the cases below, which
cover every role, the merges and the order of tool results, tools with namespaces and numbers of every spelling,
calls whose arguments are objects, JSON, JSON of JSON or neither, content arrays, tasks, the reasoning levels with
thinking on and off, earlier reasoning dropped and kept (tool results alone open no turn: the encoder of 2026-10-16,
3c7c357d), the images of tool results in the order of the calls, a message continued, long histories, what the encoder
refuses (recorded as refused; a prompt that is no UTF-8, which the tokenizer cannot take, among them), and what the
template's contract refuses though the encoder renders it. A conversation is written in its JSON as given, and
again compact and ASCII-escaped (its prompt the same), unless it is spelled by hand. The encoder must be git blob
3c7c357d, the one the template follows (chat_template.h), or nothing is written.

Both files, little-endian, no padding:
    tokenizer   "ANEMTOKG", uint32 version 1, uint32 cases; per case: uint32 name bytes, name, uint64 text bytes,
                text, uint64 reference text bytes, reference text, uint32 ids, uint32 ids[]
    template    "ANEMCHTG", uint32 version 1, uint32 cases; per case: uint32 name bytes, name, uint32 flags (1
                thinking, 2 drops thinking, 4 continues the last message, 8 has tools, 16 refused by the encoder),
                uint32 reasoning effort, uint64 messages JSON bytes, JSON, uint64 tools JSON bytes, JSON, uint64
                prompt bytes, prompt, uint32 ids, uint32 ids[]; 32 marks what the encoder renders and Anemone's
                template refuses by its contract (chat_template.h says what and why): a case whose name says so
With --corrupt it also writes each file again with one id changed, *_corrupt.bin, which the test must report.

Needs Python 3.9 and the tokenizers library (the oracle's environment). Exit status 0, 1 after any failure, 2 for a
command line it cannot read. Test tool: not part of the engine.
"""
import argparse
import copy
import glob
import hashlib
import json
import math
import os
import random
import struct
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_golden_set                                     # its requests, beside this file

TOKENIZER_MAGIC, TEMPLATE_MAGIC, FORMAT_VERSION = b"ANEMTOKG", b"ANEMCHTG", 1
THINKING, DROPS_THINKING, CONTINUES, HAS_TOOLS, REFUSED, REFUSED_BY_CONTRACT = 1, 2, 4, 8, 16, 32
REFUSED_PREFIX, CONTRACT_PREFIX = "refused:", "refused by contract:"
EFFORTS = {None: 75, "low": 50, "high": 75, "max": 100}
LONG_PROMPT_TOKENS = 1048576
IMAGE_PLACEHOLDER = "<\uff5cdeepseek_image\uff5c>"            # the encoder refuses text that holds it
LONG_CONVERSATION_BYTES = 4500000                          # about 1.2 million tokens of the corpus below
RANDOM_SEED = 20261012
RANDOM_STRINGS = 3000
RANDOM_CODE_POINT_STRINGS = 2000
LONGEST_RANDOM_STRING = 30
RUN_BYTES = 200000
SURROGATES = range(0xD800, 0xE000)
DEEPEST_ARRAYS = 127                                       # inside a call's arguments object: json.h's depth in all
EXIT_STATUS_FAILURE = 1
TEMPLATE_ENCODER_BLOB = "3c7c357dc9ae3ed37adacb8572a2e9498b7d1bcb"   # encoding.py of 2026-10-16 (chat_template.h)

PIECES = [" ", "  ", "   ", "\t", "\n", "\r\n", "\r", "\n\n", " \n", " ", "　", " ", "\u0085", "\x0b",
          "\x0c", "​", "0", "1", "12", "123", "4567", "٣", "½", "²", "Ⅻ", "①", "a", "B",
          "Zz", "word", "é", "é", "́", "⃣", "️", "!", "(", ")", "'s", "'S", "_", "-", ".",
          ",", "#", "$", "+", "<", ">", "/", "\\", "\"", "*", "\x01", "\x7f", "日", "本", "ア",
          "ー", "・", "ぁ", "゛", "゠", "가", "ا", "א", "ก", "\U0001f600",
          "\U0001f1ee\U0001f1f9", "\U0001f468‍\U0001f469‍\U0001f467", "\U0001f44d\U0001f3fd",
          "\U000e0001", "", "\U0010fffd", "￿", "�", "’", "“", "—", "«"]
INVALID_UTF8 = [b"\x80", b"\xbf", b"\xc0\x80", b"\xc1\xbf", b"\xc2", b"\xc2\x41", b"\xe0\x80\x80", b"\xe0\xa0",
                b"\xe0\xa0\x41", b"\xed\xa0\x80", b"\xed\xbf\xbf", b"\xef\xbf", b"\xf0\x80\x80\x80", b"\xf0\x90\x80",
                b"\xf4\x90\x80\x80", b"\xf5\x80\x80\x80", b"\xfe", b"\xff", b"\xf8\x88\x80\x80\x80",
                b"abc\xffdef", b"\xe4\xbd\xa0\xe5\xa5", b"<\xef\xbd\x9cUser\xef\xbd\x9c>\xff",
                b"\xc3\xa9\x80\x80\xc3\xa9", b"x" * 7 + b"\xe2\x82" + b"y" * 9, bytes(range(256))]


def ids_of(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens = False).ids


def tokenizer_case(tokenizer, name, data):
    """One text: raw bytes, which the reference reads as Python's decode(errors="replace") reads them."""
    text = data.decode("utf-8", "replace")
    ids = ids_of(tokenizer, text)
    if tokenizer.decode(ids, skip_special_tokens = False) != text:
        raise RuntimeError(f"{name}: the reference's ids do not decode to its text")
    return name, data, text.encode("utf-8"), ids


def random_strings(generator, count, draw):
    return ["".join(draw() for _ in range(generator.randint(1, LONGEST_RANDOM_STRING))) for _ in range(count)]


def random_code_point(generator):
    plane = generator.choice((0, 0, 0, 1, 2, 3, 14, 15, 16))
    while True:
        code_point = plane * 0x10000 + generator.randrange(0x10000)
        if code_point not in SURROGATES:
            return chr(code_point)


def tokenizer_cases(tokenizer, text_files):
    generator = random.Random(RANDOM_SEED)
    added = sorted(tokenizer.get_added_tokens_decoder().items())
    contents = [token.content for _, token in added]
    files = []
    for path in text_files:
        with open(path, "rb") as handle:
            files.append((os.path.basename(path), handle.read()))
    cases = [(f"file {name}", data) for name, data in files]
    cases += [(f"added token {token_id}", token.content.encode()) for token_id, token in added]
    cases += [("added tokens in a row", "".join(contents).encode()),
              ("added tokens inside text", " x ".join(contents).encode()),
              ("added tokens cut short", " ".join(content[:-1] for content in contents).encode()),
              ("added tokens nested", "".join("<" + content + content[1:] for content in contents).encode())]
    cases += [(f"ill-formed UTF-8 {index}", data) for index, data in enumerate(INVALID_UTF8)]
    for character in ("a", " ", "\n", "\t", "!", "1", "你", "é", "\U0001f600", "́", ".\n", " a", "ab"):
        cases.append((f"run of {character!r}", (character * (RUN_BYTES // len(character.encode()))).encode()))
    cases += [(f"random pieces {index}", text.encode())
              for index, text in enumerate(random_strings(generator, RANDOM_STRINGS, lambda: generator.choice(PIECES)))]
    cases += [(f"random code points {index}", text.encode()) for index, text in enumerate(
        random_strings(generator, RANDOM_CODE_POINT_STRINGS, lambda: random_code_point(generator)))]
    cases += [("emoji", "Family \U0001f468‍\U0001f469‍\U0001f467‍\U0001f466, flags \U0001f1ee\U0001f1f9"
                        "\U0001f1ef\U0001f1f5, skin \U0001f44b\U0001f3ff, keycap 1️⃣ #️⃣, "
                        "❤️\U0001f525\U0001f680".encode()),
              ("CJK", "你好，世界！日本語のテキスト、"
                      "「引用」・カタカナー。한국어 "
                      "文字 123 １２３ abc".encode())]
    corpus = b"\n".join(data for _, data in files) or b"word "
    long_prompt, token_count = b"", 0
    while token_count < LONG_PROMPT_TOKENS:
        long_prompt += corpus + b"\n"
        token_count = len(ids_of(tokenizer, long_prompt.decode("utf-8", "replace")))
    cases.append((f"long prompt of {token_count} tokens", long_prompt))
    return [tokenizer_case(tokenizer, name, data) for name, data in cases]


# ---- conversations ---------------------------------------------------------------------------------------------------
def function_tool(name, parameters, description = "A tool.", **outer):
    tool = {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
    tool.update(outer)
    return tool


def call(name, arguments, call_id = None, function_id = None, **outer):
    """A tool call; arguments is JSON text (a str) or any other value, given as is; ids of any type."""
    value = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call_id is not None:
        value["id"] = call_id
    if function_id is not None:
        value["function"]["id"] = function_id
    value.update(outer)
    return value


def conversation(name, messages, tools = None, thinking = False, effort = 75, drops = True, continues = False,
                 spelled = None, spelled_tools = None, one_spelling = False):
    """A case: the encoder's arguments, and the JSON of messages and tools when it is spelled by hand."""
    return {"name": name, "messages": messages, "tools": tools, "thinking": thinking, "effort": effort,
            "drops": drops, "continues": continues, "spelled": spelled, "spelled_tools": spelled_tools,
            "one_spelling": one_spelling}


def long_conversation(text_files):
    """A conversation of at least LONG_PROMPT_TOKENS tokens: the text files as user messages, each answered."""
    texts = []
    for path in text_files:
        with open(path, "rb") as handle:
            texts.append(handle.read().decode("utf-8", "replace").replace(IMAGE_PLACEHOLDER, "<image>"))
    messages, byte_count = [{"role": "system", "content": "You review code and text."}], 0
    while texts and byte_count < LONG_CONVERSATION_BYTES:
        for text in texts:
            messages += [{"role": "user", "content": text}, {"role": "assistant", "content": "Read.",
                                                              "reasoning_content": "Next."}]
            byte_count += len(text.encode())
    messages.append({"role": "user", "content": "Which of them has a bug?"})
    return conversation("long conversation", messages, thinking = True, one_spelling = True)


def own_conversations():
    weather = function_tool("get_weather", {"type": "object", "properties": {"city": {"type": "string"}}})
    system = {"role": "system", "content": "You are terse."}
    user = {"role": "user", "content": "Weather in Turin and Oslo?"}
    calls = {"role": "assistant", "reasoning_content": "Two calls.", "content": None,
             "tool_calls": [call("get_weather", '{"city": "Turin"}', "a"),
                            call("get_weather", '{"city": "Oslo"}', "b")]}
    turin = {"role": "tool", "tool_call_id": "a", "content": "21 C"}
    oslo = {"role": "tool", "tool_call_id": "b", "content": "9 C"}
    answer = {"role": "assistant", "reasoning_content": "Both in.", "content": "Turin is warmer."}
    history = [system]
    for turn in range(150):
        history += [{"role": "user", "content": f"Question {turn}: what is {turn} squared?"},
                    {"role": "assistant", "reasoning_content": f"{turn} times {turn}.", "content": str(turn * turn)}]
    history.append({"role": "user", "content": "And the sum of all of them?"})
    many_tools = [function_tool(f"tool_{index}", {"type": "object", "properties": {
        f"p{index}": {"type": "integer", "minimum": index * 0.25, "maximum": 2 ** index}}}) for index in range(64)]
    deep = {"leaf": "é\U0001f600"}
    for depth in range(40):
        deep = {f"level{depth}": [deep, depth, depth / 3]}
    numbers_tools = ('[{"type":"function","function":{"name":"numbers","description":"Numbers \\u00e9\\ud83d\\ude00 '
                     '\\u0000\\u001f\\"\\\\\\/\\b\\f\\n\\r\\t\\u2028","parameters":{"a":1E400,"b":-0,"c":-0.0,'
                     '"d":0.1,"e":1e-7,"f":5.960464477539063e-08,"g":123456789012345678901234567890,"h":1.0,'
                     '"i":5e-324,"j":1.7976931348623157e308,"k":1e16,"l":1e15,"m":0.0001,"n":0.00001,"o":NaN,'
                     '"p":-Infinity,"q":100,"r":2.5E+3,"s":true,"t":false,"u":null,"v":[],"w":{},"x":[1,[2,[3]]],'
                     '"\\u0079":"escaped key","z":-1.5e-300}}}]')
    namespaced = [function_tool("lookup", {"type": "object"}, namespace = "search"),
                  function_tool("lookup", {"type": "object"}, namespace = {"name": "files", "description": "Files."}),
                  function_tool("files::read", {"type": "object"}, namespace = {"name": "files"}),
                  {"type": "function", "function": {"name": "inner", "namespace": "deep", "parameters": {}}},
                  {"type": "function", "function": {"name": "nameless_namespace"}, "namespace": {"name": None,
                                                                                                 "description": "D."}},
                  {"type": "function", "namespace": {"name": "x", "description": "Only the namespace's."},
                   "function": {"name": "no_description", "parameters": {"type": "object"}}}]
    argument_forms = {"role": "assistant", "content": "", "tool_calls": [
        call("f", '{"text": "line\\none </\\uff5cDSML\\uff5c parameter>", "n": 3, "x": 2.50, "ok": true, '
                  '"none": null, "list": [1, "two", {"three": 3.0}], "empty": {}, "\\u00e9": "\\u00e9"}', "1"),
        call("f", json.dumps(json.dumps({"double": "encoded", "n": 1})), "2"),
        call("f", json.dumps(json.dumps(json.dumps({"triple": 1}))), "3"),
        call("f", "not JSON at all", "4"), call("f", "[1, 2]", "5"), call("f", "", "6"),
        call("f", "  {\"a\" : 1}  ", "7"), call("f", {"object": {"nested": [True, None]}}, "8"), call("f", 5, "9"),
        call("f", None, "10"), call("f", "{}", "11"), call("f", "\"just a string\"", "12"),
        call("f", "{\"a\": NaN, \"b\": 1e999}", "13"),
        call("f", '{"deep": ' + "[" * DEEPEST_ARRAYS + "]" * DEEPEST_ARRAYS + "}", "14"),
        call("f", "[" * (DEEPEST_ARRAYS + 2) + "]" * (DEEPEST_ARRAYS + 2), "15"),
        call("f", '["\\ud800", 1]', "16"), call("f", "{}", [], function_id = "17")]}
    later_results = [user, calls, turin, oslo, {"role": "assistant", "content": "One more."},
                     {"role": "tool", "content": "none"}, oslo, turin, {"role": "tool", "tool_call_id": "b"}]
    image_user = {"role": "user", "content": [{"type": "text", "text": "Look:"},
                                              {"type": "image_url", "image_url": "data:image/png;base64,AAAA"},
                                              {"type": "image_url", "image_url": {"url": "x.png", "detail": "low"}},
                                              {"type": "text"}]}
    turns = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
             {"role": "user", "content": "q2"},
             {"role": "assistant", "reasoning_content": "r2", "tool_calls": [call("lookup", "{}", "c1")]},
             {"role": "tool", "tool_call_id": "c1", "content": "value"}]
    image_results = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "history.png"}}]},
                     {"role": "assistant", "tool_calls": [call(name, "{}", name) for name in ("a", "b")]}] + [
        {"role": "tool", "tool_call_id": name, "content": [{"type": "text", "text": f"result {name}"},
                                                          {"type": "image_url", "image_url": {"url": f"{name}.png"}}]}
        for name in ("b", "a")]
    blocks_user = {"role": "user", "content": [
        {"type": "text", "text": "Results:"},
        {"type": "tool_result", "tool_use_id": "b", "content": [{"type": "text", "text": "nine"},
                                                                {"type": "image_url", "image_url": {"url": "x.png"}},
                                                                {"type": "other"}]},
        {"type": "tool_result", "tool_use_id": "a", "content": "twenty-one"},
        {"type": "tool_result", "tool_use_id": "zz", "content": None}]}
    cases = [
        conversation("empty conversation", []),
        conversation("empty content", [{"role": "user", "content": ""}], thinking = True),
        conversation("missing content", [{"role": "user"}, {"role": "assistant"}, {"role": "user", "content": "?"}]),
        conversation("content array of text", [{"role": "system", "content": [{"type": "text", "text": "S1"},
                                                                              {"type": "text", "text": "S2"}]},
                                               {"role": "user", "content": [{"type": "text", "text": "a"},
                                                                            {"type": "text", "text": "b"}]}]),
        conversation("content arrays empty", [{"role": "user", "content": []}, {"role": "user", "content": "b"},
                                              {"role": "latest_reminder", "content": []},
                                              {"role": "assistant", "content": [], "reasoning_content": None}]),
        conversation("images", [image_user], thinking = True, effort = 100),
        conversation("refused by contract: tool results as content parts", [user, calls, blocks_user],
                     tools = [weather], thinking = True),
        conversation("refused by contract: other content parts", [{"role": "user", "content": [
            {"type": "image", "data": "AAAA"}, {"type": "input_audio", "input_audio": {}}, {"text": "no type"}]}]),
        conversation("consecutive users merged", [{"role": "user", "content": "one"},
                                                  {"role": "user", "content": "two"},
                                                  {"role": "user", "content": "three", "task": "query"}]),
        conversation("user with a task not merged", [{"role": "user", "content": "one", "task": "domain"},
                                                     {"role": "user", "content": "two"}]),
        conversation("tool result after a task", [{"role": "user", "content": "x", "task": "authority"},
                                                  {"role": "tool", "tool_call_id": "a", "content": "r"}]),
        conversation("tool messages first", [turin, oslo, answer], thinking = True),
        conversation("tool content forms", [user, calls, {"role": "tool", "tool_call_id": "a", "content": None},
                                            {"role": "tool", "tool_call_id": "b", "content": [
                                                {"type": "text", "text": "x"},
                                                {"type": "image_url", "image_url": "u"}]},
                                            {"role": "tool", "content": []}, {"role": "tool"}], tools = [weather]),
        conversation("tool results reordered", [system, user, calls, oslo, turin, answer], tools = [weather],
                     thinking = True),
        conversation("tool results unknown and repeated ids", [system, user, {
            "role": "assistant", "content": "", "tool_calls": [call("get_weather", "{}", "a"),
                                                               call("get_weather", "{}", "b"),
                                                               call("get_weather", "{}", "a"),
                                                               {"type": "function", "function": {"name": "get_weather",
                                                                "arguments": "{}", "id": "c"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "C"},
            {"role": "tool", "tool_call_id": "zz", "content": "Z"},
            {"role": "tool", "tool_call_id": "a", "content": "A"}, {"role": "tool", "content": "none"},
            {"role": "tool", "tool_call_id": "b", "content": "B"}], tools = [weather]),
        conversation("two rounds of calls", [system, user, calls, oslo, turin,
                                             {"role": "assistant", "content": "More.", "reasoning_content": "Again.",
                                              "tool_calls": [call("get_weather", '{"city": "Rome"}', "c")]},
                                             {"role": "tool", "tool_call_id": "c", "content": "25 C"}, answer],
                     tools = [weather], thinking = True, drops = False),
        conversation("calls without tools, thinking dropped", [user, calls, turin, oslo, answer,
                                                               {"role": "user", "content": "Thanks"}], thinking = True),
        conversation("tool results open no turn, thinking dropped", turns, thinking = True),
        conversation("tool results open no turn, thinking kept", turns, thinking = True, drops = False),
        conversation("tool results open no turn, chat mode", turns),
        conversation("the next turn drops the reasoning of the calls", turns + [{"role": "user", "content": "q3"}],
                     thinking = True),
        conversation("tool results after an empty user array open no turn",
                     [user, calls, {"role": "user", "content": []}, turin, oslo], thinking = True),
        conversation("an empty user array opens a turn", [user, answer, {"role": "user", "content": []}],
                     thinking = True),
        conversation("tool results alone, no turn at all", [turin, oslo, answer, turin], thinking = True),
        conversation("tool result images in the order of the calls", image_results),
        conversation("tool result images in the order of the calls, thinking", image_results, thinking = True),
        conversation("latest reminder", [system, {"role": "latest_reminder", "content": "2026-10-12, Turin"}, user]),
        conversation("latest reminder after a user", [user, {"role": "latest_reminder"}], thinking = True),
        conversation("mid-conversation system last", [system, user, answer, {"role": "system", "content": "Be brief."}],
                     thinking = True, effort = 42),
        conversation("response format", [{"role": "system", "content": "JSON only.", "response_format": {
            "type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object", "x": [0.5, 1e-9, -3]}}}},
                                         user]),
        conversation("many tools", [system, user], tools = many_tools, thinking = True),
        conversation("deeply nested tool schema", [system, user], tools = [function_tool("deep", deep)]),
        conversation("numbers and escapes in a tool schema", None, spelled = '[{"role": "system", "content": "s"}]',
                     spelled_tools = numbers_tools),
        conversation("names with one separator", [system, user, {"role": "assistant", "content": "", "tool_calls": [
            call("files::read", "{}", "1"), call("::odd", "{}", "2")]}],
                     tools = [function_tool("files::read", {"type": "object"}), function_tool("::odd", {})]),
        conversation("refused by contract: namespaced tools", [system, user], tools = namespaced),
        conversation("refused by contract: namespaced calls", [system, user, {
            "role": "assistant", "content": "", "tool_calls": [call("lookup", "{}", "1", namespace = "search"),
            {"type": "function", "function": {"name": "lookup", "arguments": "{}", "namespace": "files"}},
            call("search::lookup", "{}", "3"), call("search::lookup", "{}", "4", namespace = "search"),
            call("plain", "{}", "5", namespace = "")]}], tools = namespaced, thinking = True),
        conversation("argument forms", [system, user, argument_forms], tools = [weather]),
        conversation("long history thinking dropped", history, thinking = True),
        conversation("long history thinking kept", history, thinking = True, drops = False),
        conversation("long history chat", history),
        conversation("unicode and control characters", [{"role": "user", "content":
            "NUL \u0000 SOH \u0001 DEL \u007f   ‏שלום é \U0001f469‍\U0001f4bb"
            " \"quotes\" \\ back"}], thinking = True),
        conversation("continue the answer", [system, user, {"role": "assistant", "content": "The weather in"}],
                     continues = True),
        conversation("continue the answer thinking", [user, {"role": "assistant", "reasoning_content": "Hmm",
                                                             "content": "Partial"}], thinking = True, continues = True),
        conversation("continue with calls", [user, dict(calls)], tools = [weather], thinking = True, continues = True),
        conversation("task title after an answer", [user, {"role": "assistant", "content": "A", "task": "title"}]),
        conversation("task read_url", [{"role": "user", "content": "https://example.org", "task": "read_url"}]),
        conversation("task action thinking", [{"role": "user", "content": "Search?", "task": "action"},
                                              {"role": "assistant", "content": "Search", "reasoning_content": "r"}],
                     thinking = True),
        conversation("reasoning kept without tools", [system, user, answer, {"role": "user", "content": "Why?"}],
                     thinking = True, drops = False),
        conversation("reasoning null and missing", [user,
                                                    {"role": "assistant", "content": "x", "reasoning_content": None},
                                                    {"role": "user", "content": "y"},
                                                    {"role": "assistant", "content": "z"}],
                     thinking = True, drops = False),
        conversation("effort lowest", [user], thinking = True, effort = 1),
        conversation("effort highest without system", [user], thinking = True, effort = 100),
        conversation("effort in chat mode", [system, user], effort = 1),
        conversation("assistant first", [answer, user], thinking = True),
        conversation("members repeated and escaped", None, spelled = '[{"role":"user","content":"first","content":'
                     '"last","\\u0072ole":"user","extra":{"ignored":[1,2]}},'
                     '{"r\\u006fle":"assistant","content":"ok"}]'),
        conversation("JSON spacing", None, spelled = ' \n[ \t{ "role" : "user" ,\r\n "content" : "spaced" } ] \n'),
        conversation("tool results sorted after a later answer", later_results, tools = [weather], thinking = True),
        conversation("refused by contract: a key twice in a call's arguments", [user, {
            "role": "assistant", "content": "", "tool_calls": [call("f", '{"city": "Turin", "city": "Oslo"}', "a")]}]),
        conversation("refused by contract: a key twice inside a parameter", None, spelled = '[{"role": "assistant", '
                     '"tool_calls": [{"function": {"name": "f", "arguments": {"b": {"c": 1, "\\u0063": [2]}}}}]}]'),
        conversation("refused by contract: a key twice in a tool", None,
                     spelled = '[{"role": "system", "content": "s"}]', spelled_tools = '[{"type": "function", '
                     '"function": {"name": "f", "parameters": {"type": "object"}, "parameters": {}}}]'),
        conversation("refused by contract: a key twice in a response format", None, spelled = '[{"role": "system", '
                     '"content": "s", "response_format": {"a": 1, "a": 2}}]'),
        conversation("refused by contract: arguments nested deeper than json.h reads", [user, {
            "role": "assistant", "content": "", "tool_calls": [
                call("f", '{"deep": ' + "[" * (DEEPEST_ARRAYS + 1) + "]" * (DEEPEST_ARRAYS + 1) + "}", "a")]}]),
        conversation("refused: a lone surrogate in a call's arguments object", [user, {
            "role": "assistant", "content": "", "tool_calls": [call("f", '{"s": "\\ud800"}', "a")]}]),
        conversation("refused: a call's id that is a list", [user, {"role": "assistant", "content": "", "tool_calls": [
            call("f", "{}", [1])]}]),
        conversation("refused by contract: a call's id that is a number", [user, {
            "role": "assistant", "content": "", "tool_calls": [call("f", "{}", 7)]}]),
        conversation("refused by contract: the encoder's own content_blocks", [{
            "role": "user", "content": "u", "content_blocks": [{"type": "text", "text": "b"}]}]),
        conversation("refused by contract: the encoder's own wo_eos", [user, {"role": "assistant", "content": "a",
                                                                               "wo_eos": True}]),
        # what the encoder refuses
        conversation("refused: image placeholder in text", [{"role": "user", "content": "<｜deepseek_image｜>"}]),
        conversation("refused: image placeholder in reasoning", [user, {"role": "assistant", "content": "a",
                                                                         "reasoning_content": "<｜deepseek_image｜>"}]),
        conversation("refused: unknown role", [{"role": "developer", "content": "x"}, user]),
        conversation("refused: call without function", [user, {"role": "assistant", "tool_calls": [{"id": "x"}]}]),
        conversation("refused: conflicting namespaces", [user, {"role": "assistant", "tool_calls": [
            call("a::b", "{}", namespace = "c")]}]),
        conversation("refused: name with two separators", [system, user], tools = [function_tool("a::b::c", {})]),
        conversation("refused: image without source", [{"role": "user", "content": [{"type": "image_url"}]}]),
        conversation("refused: user content null", [{"role": "user", "content": None}]),
        conversation("refused: unknown task", [{"role": "user", "content": "x", "task": "bogus"}]),
        conversation("refused: tools without messages", [], tools = [weather]),
        conversation("refused: a message that is no object", None, spelled = '["hello"]'),
        conversation("refused: tool without function", [system], tools = [{"type": "function"}]),
    ]
    return cases


def template_encoder(checkpoint):
    """The checkpoint's encoding/encoding.py, run from the bytes read here once they are the template's git blob."""
    path = os.path.join(checkpoint, "encoding", "encoding.py")
    with open(path, "rb") as handle:
        source = handle.read()
    blob = hashlib.sha1(b"blob %d\0" % len(source) + source).hexdigest()
    if blob != TEMPLATE_ENCODER_BLOB:
        raise RuntimeError(f"{path} is git blob {blob}, not {TEMPLATE_ENCODER_BLOB}: the template follows that "
                           "encoder alone (chat_template.h), and another one's prompts would be no golden set of it")
    encoding = types.ModuleType("encoding")
    encoding.__file__ = path
    exec(compile(source, path, "exec"), encoding.__dict__)
    return encoding


def official_conversations(encoding, tests_directory):
    """The encoder's own test pairs, each held to its shipped output."""
    cases = []
    for input_path in sorted(glob.glob(os.path.join(tests_directory, "test_input_*.json"))):
        number = os.path.basename(input_path)[len("test_input_"):-len(".json")]
        with open(os.path.join(tests_directory, f"test_output_{number}.txt"), encoding = "utf8") as handle:
            shipped = handle.read()
        with open(input_path, encoding = "utf8") as handle:
            raw = json.load(handle)
        case = encoding.load_cases(input_path)[0]
        prompt, _ = encoding.encode_case(case, thinking_mode = "chat")
        if prompt != shipped:
            raise RuntimeError(f"the encoder does not reproduce its own pair {number}")
        messages = raw["messages"] if isinstance(raw, dict) else raw
        tools = raw.get("tools") if isinstance(raw, dict) else None
        effort = case.get("reasoning_effort")
        cases.append(conversation(f"official pair {number}", messages, tools = tools,
                                  thinking = (case.get("thinking_mode") or "chat") == "thinking",
                                  effort = EFFORTS[effort] if effort in EFFORTS else effort))
    return cases


def request_conversations():
    """make_golden_set.py's requests, as cases."""
    cases = []
    for request in make_golden_set.build_requests():
        effort = request.get("reasoning_effort")
        refusal = CONTRACT_PREFIX + " " if "namespace" in json.dumps(request) else ""
        cases.append(conversation(f"{refusal}request {request['name']}", request["messages"],
                                  tools = request.get("tools"),
                                  thinking = request["thinking_mode"] == "thinking",
                                  effort = EFFORTS[effort] if effort in EFFORTS else effort,
                                  drops = request.get("drop_thinking", True)))
    return cases


def render(encoding, case, messages_text, tools_text):
    """The encoder's prompt for a case, or None when it refuses it."""
    try:
        messages = json.loads(messages_text)
        if tools_text is not None:
            if not messages:
                raise ValueError("A case with tools must contain at least one message")
            messages[0]["tools"] = json.loads(tools_text)
        if case["continues"]:
            messages[-1]["wo_eos"] = True
        prompt = encoding.encode_messages(messages, thinking_mode = "thinking" if case["thinking"] else "chat",
                                          drop_thinking = case["drops"], reasoning_effort = case["effort"])
        prompt.encode("utf-8")                            # one the tokenizer cannot take (a lone surrogate) is none
        return prompt
    except Exception:                                     # a refusal of the encoder is part of the contract
        return None


def template_records(encoding, tokenizer, cases):
    records = []
    for case in cases:
        spellings = [(case["spelled"], case["spelled_tools"], "")] if case["messages"] is None else \
            [(json.dumps(case["messages"], ensure_ascii = False), None, "")] if case["one_spelling"] else [
            (json.dumps(case["messages"], ensure_ascii = False),
             None if case["tools"] is None else json.dumps(case["tools"], ensure_ascii = False), ""),
            (json.dumps(case["messages"], separators = (",", ":")),
             None if case["tools"] is None else json.dumps(case["tools"], separators = (",", ":")), ", compact ASCII"),
            (json.dumps(case["messages"], ensure_ascii = False, indent = 2),
             None if case["tools"] is None else json.dumps(case["tools"], ensure_ascii = False, indent = 2),
             ", indented")]
        for messages_text, tools_text, spelling in spellings:
            prompt = render(encoding, copy.deepcopy(case), messages_text, tools_text)
            by_contract = case["name"].startswith(CONTRACT_PREFIX)
            flags = (THINKING if case["thinking"] else 0) | (DROPS_THINKING if case["drops"] else 0) | \
                    (CONTINUES if case["continues"] else 0) | (HAS_TOOLS if tools_text is not None else 0) | \
                    (REFUSED if prompt is None else 0) | (REFUSED_BY_CONTRACT if by_contract else 0)
            if case["name"].startswith(REFUSED_PREFIX) != (prompt is None):
                raise RuntimeError(f"{case['name']}{spelling}: the encoder's answer is not what the case expects")
            ids = [] if prompt is None else ids_of(tokenizer, prompt)
            records.append((case["name"] + spelling, flags, case["effort"], messages_text.encode(),
                            b"" if tools_text is None else tools_text.encode(),
                            b"" if prompt is None else prompt.encode(), ids))
    return records


def write_file(path, magic, records, pack_record, overwrite):
    if os.path.lexists(path) and not overwrite:
        raise RuntimeError(f"{path} exists: --overwrite replaces it")
    with open(path + ".partial", "wb") as handle:
        handle.write(magic + struct.pack("<II", FORMAT_VERSION, len(records)))
        for record in records:
            handle.write(pack_record(*record))
    os.replace(path + ".partial", path)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def sized(data, width = "<Q"):
    return struct.pack(width, len(data)) + data


def pack_tokenizer_record(name, text, reference_text, ids):
    return sized(name.encode(), "<I") + sized(text) + sized(reference_text) + \
        struct.pack(f"<I{len(ids)}I", len(ids), *ids)


def pack_template_record(name, flags, effort, messages, tools, prompt, ids):
    return sized(name.encode(), "<I") + struct.pack("<II", flags, effort) + sized(messages) + sized(tools) + \
        sized(prompt) + struct.pack(f"<I{len(ids)}I", len(ids), *ids)


def corrupted(records, ids_index):
    """The records with one id changed: of the record with the most ids, its middle one."""
    target = max(range(len(records)), key = lambda index: len(records[index][ids_index]))
    record = list(records[target])
    ids = list(record[ids_index])
    ids[len(ids) // 2] ^= 1
    record[ids_index] = ids
    return records[:target] + [tuple(record)] + records[target + 1:]


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required = True)
    parser.add_argument("--output-directory", required = True)
    parser.add_argument("--text", nargs = "*", default = [])
    parser.add_argument("--overwrite", action = "store_true")
    parser.add_argument("--corrupt", action = "store_true", help = "also write each file with one id changed")
    arguments = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(arguments.checkpoint, "tokenizer.json"))
    try:
        encoding = template_encoder(arguments.checkpoint)
        cases = official_conversations(encoding, os.path.join(arguments.checkpoint, "encoding", "tests")) + \
            request_conversations() + own_conversations() + [long_conversation(arguments.text)]
        template = template_records(encoding, tokenizer, cases)
        tokenizer_records = tokenizer_cases(tokenizer, arguments.text)
        os.makedirs(arguments.output_directory, exist_ok = True)
        outputs = [("tokenizer_golden.bin", TOKENIZER_MAGIC, tokenizer_records, pack_tokenizer_record, 3),
                   ("template_golden.bin", TEMPLATE_MAGIC, template, pack_template_record, 6)]
        for file_name, magic, records, pack_record, ids_index in outputs:
            path = os.path.join(arguments.output_directory, file_name)
            digest = write_file(path, magic, records, pack_record, arguments.overwrite)
            print(f"{path}: {len(records)} cases, {sum(len(record[ids_index]) for record in records)} ids, "
                  f"sha256 {digest}")
            if arguments.corrupt:
                path = path.replace(".bin", "_corrupt.bin")
                write_file(path, magic, corrupted(records, ids_index), pack_record, True)
                print(f"{path}: one id changed")
        refused = sum(1 for record in template if record[1] & REFUSED)
        print(f"conversations: {len(cases)}, spellings: {len(template)}, refused by the encoder: {refused}")
    except (RuntimeError, OSError) as error:
        print(f"make_front_end_golden.py: {error}", file = sys.stderr)
        return EXIT_STATUS_FAILURE
    return 0


if __name__ == "__main__":
    sys.exit(main())
