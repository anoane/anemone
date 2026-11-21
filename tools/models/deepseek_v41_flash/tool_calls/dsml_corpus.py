#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The cases of the DSML grammar's proofs, each with what dsml_model.py says it gives (no dependencies).

    dsml_corpus.py variants                      the variant corpus, generated from the grammar
    dsml_corpus.py fuzz --seed S --count N       random mutations, splices and truncations of it
    dsml_corpus.py reference-check ENCODING_PY   G3: every reference spelling against encoding.py's own parse
    dsml_corpus.py summary FILE                  the counts of a corpus file (cases, kinds, repairs)
    dsml_corpus.py g3-listings FILE BASE TOLERANT   G3 on dsml_dump.c's listings of FILE: the cases the tolerant
                                                 reading repairs nothing in give v0.0.5-alpha's (BASE) output
                                                 bytes, whole answers of every feeding and API

A corpus is written to standard output, one case a block of lines, every byte string in hexadecimal:
    case NAME / tools HEX,HEX ("-": the request renders no tool) / stops HEX,HEX / thinking 0|1 / mode tolerant /
    finish stop|length / stream HEX / content HEX / reasoning HEX / calls HEX / expect_finish R /
    repairs N1 ... N11 (dsml_model.REPAIR_KINDS) / end
tests/engine/server/api/chat_tool_calls_test.c reads it: build/<build>/chat_tool_calls_test FILE. The output is the same for the same
arguments (the fuzz's random source is seeded).

The variants: each place of each head (the five runs of spaces, both bars, the word, the keyword and its case) in
every spelling, every quote and string form and a missing string attribute in both orders, each closing left out on
its own and all of them, each start context (the first byte, after </think>, after text with no, one or two line
feeds or a space, inside reasoning, a bare invoke), one to three invokes, values empty, multi-line UTF-8, JSON or
holding "<" and DSML-like text, names folded, unknown or namespaced, stray heads, stops and limits, and requests
that render no tool.
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # its model, beside it
import dsml_model as model  # noqa: E402

WIDE, NARROW = model.WIDE_BAR, b"|"
TOOLS = [b"read", b"bash", b"list_agents", b"job", b"job_kill", b"write"]
HEADS = [(kind, closing) for kind in (model.BLOCK, model.INVOKE, model.PARAMETER) for closing in (False, True)]
REFERENCE_SPACES = (b"", b"", b"", b"", b" ")
SPACINGS = (b"", b" ", b"  ", b"\t", b"\n", b"\r\n", b" " * model.SPACE_MAX, b" " * (model.SPACE_MAX + 1))
SLOTS = 5
CLOSING_SLOT = 1
MUTATION_ALPHABET = (b"<", b">", b"/", b"|", WIDE, b" ", b"\n", b'"', b"'", b"=", b"DSML", b"dsml", b"calls",
                     b"invoke", b"parameter", b"name", b"string", b"true", b"false", b"x", b"\xff", b"\xe2\x82")
MOST_MUTATIONS = 6
JOIN_HEADS = (b"<|DSML|invoke name=bash>", b"</" + WIDE + b"DSML" + WIDE + b" calls>", b"< | dsml | parameter name=p>",
              b"<" + WIDE + b"DSML" + WIDE + b"\ninvoke>")  # cut in two: a head text between tags may complete
FUZZ_STOPS = (b"@@END@@",)


class Style:
    """How a block is spelt: per head its spaces, bars, word and keyword; the tags' attributes; what is left out."""

    def __init__(self, **changes):
        self.spaces = {head: REFERENCE_SPACES for head in HEADS}
        self.bars = {head: (WIDE, WIDE) for head in HEADS}
        self.words = {head: b"DSML" for head in HEADS}
        self.keywords = dict(model.REFERENCE_KEYWORDS)
        self.invoke_attributes = lambda name: b' name="' + name + b'">'
        self.parameter_attributes = lambda name, string: b' name="' + name + b'" string="' + string + b'">'
        self.closing_attributes = b">"
        self.left_out, self.separator, self.opens_block = set(), b"\n", True
        for key, value in changes.items():
            setattr(self, key, value)

    def head(self, kind, closing):
        spaces, bars = self.spaces[(kind, closing)], self.bars[(kind, closing)]
        return (b"<" + spaces[0] + (b"/" + spaces[1] if closing else b"") + bars[0] + spaces[2] +
                self.words[(kind, closing)] + spaces[3] + bars[1] + spaces[4] + self.keywords[kind])


def block(invokes, style=None):
    """invokes: (name, [(parameter, string, value)]) each; string None leaves the attribute out."""
    style = style or Style()
    out = (style.head(model.BLOCK, False) + b">" + style.separator) if style.opens_block else b""
    for count, (name, parameters) in enumerate(invokes):
        out += style.head(model.INVOKE, False) + style.invoke_attributes(name) + style.separator
        for index, (parameter, string, value) in enumerate(parameters):
            attributes = (style.parameter_attributes(parameter, string) if string is not None else
                          b' name="' + parameter + b'">')
            out += style.head(model.PARAMETER, False) + attributes + value
            if ("parameter", count, index) not in style.left_out and "parameter" not in style.left_out:
                out += style.head(model.PARAMETER, True) + style.closing_attributes
            out += style.separator
        if "invoke" not in style.left_out:
            out += style.head(model.INVOKE, True) + style.closing_attributes + style.separator
    if "calls" not in style.left_out and style.opens_block:
        out += style.head(model.BLOCK, True) + style.closing_attributes
    return out


BASE_CALL = [(b"read", [(b"path", b"true", b"src/a.c"), (b"limit", b"false", b"10")])]


def variants():
    """(name, stream, settings) of every variant."""
    cases = []

    def add(name, stream, **settings):
        cases.append((name, stream, settings))

    add("reference", b"\n\n" + block(BASE_CALL))
    for kind, closing in HEADS:
        for slot in range(SLOTS):
            if slot == CLOSING_SLOT and not closing:
                continue
            for spacing in SPACINGS:
                spaces = list(REFERENCE_SPACES)
                spaces[slot] = spacing
                style = Style()
                style.spaces[(kind, closing)] = tuple(spaces)
                add(f"spaces {kind}{'/' if closing else ''} slot {slot} {spacing!r}",
                    b"\n\n" + block(BASE_CALL, style))
        for position in (0, 1):
            style = Style()
            bars = [WIDE, WIDE]
            bars[position] = NARROW
            style.bars[(kind, closing)] = tuple(bars)
            add(f"bar {kind}{'/' if closing else ''} {position} ascii", b"\n\n" + block(BASE_CALL, style))
    for word in (b"dsml", b"Dsml", b"DsMl"):
        add(f"word {word!r}", b"\n\n" + block(BASE_CALL, Style(words={head: word for head in HEADS})))
    add("ascii everywhere, no spaces", b"\n" + block(BASE_CALL, Style(
        bars={head: (NARROW, NARROW) for head in HEADS}, spaces={head: (b"",) * SLOTS for head in HEADS})))
    for keyword in (b"calls", b"function_calls", b"tool_calls"):
        for spelt in (keyword, keyword.upper(), keyword.capitalize()):
            add(f"block keyword {spelt!r}", b"\n\n" + block(BASE_CALL, Style(
                keywords=dict(model.REFERENCE_KEYWORDS, **{model.BLOCK: spelt}))))
    for kind in (model.INVOKE, model.PARAMETER):
        for spelt in (model.REFERENCE_KEYWORDS[kind].upper(), model.REFERENCE_KEYWORDS[kind].capitalize()):
            add(f"{kind} keyword {spelt!r}", b"\n\n" + block(BASE_CALL, Style(
                keywords=dict(model.REFERENCE_KEYWORDS, **{kind: spelt}))))
    invoke_forms = {
        "single quotes": lambda name: b" name='" + name + b"'>",
        "bare": lambda name: b" name=" + name + b">",
        "spaces around =": lambda name: b" name = \"" + name + b"\" >",
        "attribute in capitals": lambda name: b' NAME="' + name + b'">',
        "an unknown attribute before": lambda name: b' id="7" name="' + name + b'">',
        "an unknown attribute after": lambda name: b' name="' + name + b'" async>',
        "a quote left open": lambda name: b' name="' + name + b">",
        "no attribute": lambda name: b">",
        "an empty name": lambda name: b' name="">',
        "spaces in the name": lambda name: b' name=" ' + name + b' ">',
        "no space before name": lambda name: b'name="' + name + b'">',
        "no closing >": lambda name: b' name="' + name + b'"',
    }
    for label, form in invoke_forms.items():
        add(f"invoke {label}", b"\n\n" + block(BASE_CALL, Style(invoke_attributes=form)))
    parameter_forms = {
        "single quotes": lambda name, string: b" name='" + name + b"' string='" + string + b"'>",
        "bare": lambda name, string: b" name=" + name + b" string=" + string + b">",
        "string first": lambda name, string: b' string="' + string + b'" name="' + name + b'">',
        "spaces around =": lambda name, string: b' name = "' + name + b'" string = "' + string + b'" >',
        "a quote left open": lambda name, string: b' name="' + name + b'" string="' + string + b">",
        "an unknown attribute": lambda name, string: b' name="' + name + b'" type="x" string="' + string + b'">',
        "no closing >": lambda name, string: b' name="' + name + b'" string="' + string + b'"',
    }
    for label, form in parameter_forms.items():
        add(f"parameter {label}", b"\n\n" + block(BASE_CALL, Style(parameter_attributes=form)))
    for string in (b"True", b"False", b"1", b"0", b"TRUE", b"yes", None):
        for value in (b"10", b"src/a.c", b'{"a": [1, null]}', b'"quoted"', b"NaN", b"{bad", b" 7 ", b"true"):
            add(f"string {string!r} value {value!r}",
                b"\n\n" + block([(b"read", [(b"v", string, value)])]))
    for left_out in (("parameter", 0, 0), ("parameter", 0, 1), "parameter", "invoke", "calls"):
        add(f"left out {left_out}", b"\n\n" + block(BASE_CALL, Style(left_out={left_out})))
    add("every closing left out", b"\n\n" + block(BASE_CALL, Style(left_out={"parameter", "invoke", "calls"})))
    for separator in (b"", b" ", b"\r\n", b"\n\n", b" \t \n"):
        add(f"separator {separator!r}", b"\n\n" + block(BASE_CALL, Style(separator=separator)))
    contexts = {"first byte": b"", "one line feed": b"\n", "text": b"Done.", "text, one line feed": b"Done.\n",
                "text, two line feeds": b"Done.\n\n", "text, a space": b"Done. ", "text, three line feeds": b"x\n\n\n",
                "text, space and two line feeds": b"x \n\n", "two line feeds and a space": b"x\n\n "}
    for label, before in contexts.items():
        for opens in (True, False):
            stream = before + block(BASE_CALL, Style(opens_block=opens))
            add(f"context {label}{'' if opens else ', bare invoke'}", stream)
            add(f"context {label}{'' if opens else ', bare invoke'}, after </think>", b"plan</think>" + stream,
                thinking=True)
    add("in reasoning", b"plan\n\n" + block(BASE_CALL), thinking=True)
    add("in reasoning, no line feeds", b"plan" + block(BASE_CALL), thinking=True)
    add("in reasoning, bare invoke", b"plan " + block(BASE_CALL, Style(opens_block=False)), thinking=True)
    second = (b"bash", [(b"command", b"true", b"ls -l")])
    third = (b"list_agents", [])
    for invokes in ([second], [BASE_CALL[0], second], [BASE_CALL[0], second, third]):
        add(f"{len(invokes)} invokes", b"\n\n" + block(invokes))
        add(f"{len(invokes)} invokes, ascii, no closings", b"\n" + block(invokes, Style(
            bars={head: (NARROW, NARROW) for head in HEADS}, left_out={"parameter", "invoke"})))
    values = {"empty": b"", "multi-line UTF-8": "línea 1\n第二行\n\U0001f642".encode(), "a <": b"a < b <",
              "DSML-like text": b"x |DSML| y <|DSML| foo </|DSML|> " + model.DSML,
              "a whole marker (the reference keeps it)": b"a <" + model.DSML + b" invoke name=\"x\"> b",
              "a trailing line feed": b"text\n", "quotes and backslashes": b'say "hi" \\ there',
              "a JSON object": b'{"k": "v", "n": [1, 2]}'}
    for label, value in values.items():
        add(f"value {label}", b"\n\n" + block([(b"write", [(b"text", b"true", value)])]))
        add(f"value {label}, tolerant", b"\n" + block([(b"write", [(b"text", b"true", value)])], Style(
            bars={head: (NARROW, NARROW) for head in HEADS})))
    for name in (b"READ", b"Read", b"list-agents", b"List_Agents", b"j", b"readfile", b"files::read", b"rea"):
        add(f"name {name!r}", b"\n\n" + block([(name, [(b"path", b"true", b"a")])]))
        add(f"name {name!r}, tolerant", b"\n" + block([(name, [(b"path", b"true", b"a")])]))
    add("a repeated parameter", b"\n\n" + block([(b"read", [(b"a", b"true", b"1"), (b"b", b"false", b"2"),
                                                            (b"a", b"true", b"3")])]))
    add("text between tags", b"\n\n<" + model.DSML + b" calls>\nLet me look.\n" + block(BASE_CALL)[len(
        b"<" + model.DSML + b" calls>\n"):])
    add("text after the last invoke", block(BASE_CALL, Style(left_out={"calls"})) + b"That is all.")
    add("a stray closing head in content", b"text </" + model.DSML + b" parameter> more")
    add("a stray parameter head in content", b"see <|DSML| parameter name=x> here")
    add("a stray closing in a block", b"\n\n" + block(BASE_CALL).replace(
        b"</" + model.DSML + b" invoke>", b"</" + model.DSML + b" parameter></" + model.DSML + b" invoke>"))
    add("an empty block", b"x\n\n<" + model.DSML + b" calls>\n</" + model.DSML + b" calls>")
    add("a block of text", b"x\n<|DSML|calls>\nno call here\n</|DSML|calls>")
    add("an unnamed invoke and a named one", b"\n<|DSML|invoke>\n<|DSML|invoke name=read>")
    add("a parameter before any invoke", b"\n<|DSML|calls><|DSML|parameter name=a>1<|DSML|invoke name=bash>")
    cut_name = (b"<" + model.DSML + b" calls> <" + model.DSML + b" invoke name=\"j</" + model.DSML + b" parameter> <" +
             model.DSML + b" parameter name=\"description\" string=\"true\">Stop the background search</" +
             model.DSML + b" parameter> </" + model.DSML + b" invoke> </" + model.DSML + b" calls>")
    for before in (b"", b"\n", b"\n\n", b"Done. "):
        add(f"a name cut by a hole, after {before!r}", before + cut_name)
    for stops in ([b"\n"], [b"Done"], [b"END"], [b"<"]):
        add(f"stop {stops!r}", b"Done.\n\n" + block(BASE_CALL), stops=stops)
    full = b"\n\n" + block(BASE_CALL)
    for cut in range(1, len(full), 7):
        add(f"cut at {cut} by the limit", full[:cut], finish="length")
        add(f"cut at {cut} at the end", full[:cut])
    # Reproduced faults of the tolerant reading, kept.
    dsml = model.DSML
    value_cut = (b"\n\n<" + dsml + b" calls>\n<" + dsml + b' invoke name="bash">\n<' + dsml +
                 b' parameter name="command" string="true">rm -rf ./build/tmp_cache')
    after_call = b"\n\n<" + dsml + b" calls>\n<" + dsml + b' invoke name="read">\n</' + dsml + b" invoke>\n"
    add("reproduced: a value cut by the limit", value_cut, finish="length")
    add("reproduced: a value cut at the end", value_cut)
    add("reproduced: a head cut by the limit", value_cut[:value_cut.index(b" parameter") + len(b" param")],
        finish="length")
    add("reproduced: an invoke cut by the limit after a call", after_call + value_cut[len(b"\n\n<" + dsml + b" calls>\n"):],
        finish="length")
    add("reproduced: an exact block's closing spelled otherwise",
        b"\n\n<" + dsml + b" calls>\n<" + dsml + b'invoke name="read">\n</' + dsml + b"invoke>\n</" + dsml +
        b'calls>\nI have asked for the file; next I will run the build.\n\n<|DSML|invoke name="bash"></|DSML|invoke>')
    add("reproduced: a head the joined text forms", b"abc<|DSML|\n<|DSML|invoke name=read></|DSML|invoke>invoke name=bash>")
    add("reproduced: an exact block's untrimmed name",
        b"\n\n<" + dsml + b" calls>\n<" + dsml + b' invoke name=" read">\n</' + dsml + b" invoke>\n</" + dsml + b" calls>")
    add("reproduced: a long declared name before the folded one",
        b"\n\n<|DSML|calls>\n<|DSML|invoke name=JOB-KILL></|DSML|invoke>", tools=[b"x" * 1100, b"job_kill"])
    add("reproduced: an unnamed invoke's value",
        b"\n<|DSML|invoke>\n<|DSML|parameter name=a>lost value</|DSML|parameter>\n<|DSML|invoke name=read>")
    add("reproduced: text after the last exact invoke", after_call + b"Done now")
    # A closing head without its "<": the corpus case that found it (its value took the closing in), then each kind
    # in each spelling in an otherwise exact block.
    add("reproduced: a parameter closing without its <, as fuzz 5/456 found it",
        b"Done.\n\n<" + dsml + b' calls>\n<' + dsml + b' invoke name="read">\n<' + dsml +
        b' parameter name="pat\xe2\x82h" string="true">src/a./' + dsml + b' parameter>\n<' + WIDE + b"ML" + WIDE +
        b' parameter name="limit" string=lse">10</' + dsml + b" parameter>\n</" + dsml + b" invoke>\n</" + dsml +
        b" calls>", stops=[b"END"])
    for kind in (b"parameter", b"invoke", b"calls"):
        for index, form in enumerate((b"/" + dsml + b" ", b"/" + dsml, b"/ " + WIDE + b" DSML " + WIDE + b" ",
                                      b"/|DSML| ")):
            closing = {other: b"</" + dsml + b" " + other + b">" for other in (b"parameter", b"invoke", b"calls")}
            closing[kind] = form + kind + b">"
            add(f"reproduced: a {kind.decode()} closing without its <, form {index}",
                b"\n\n<" + dsml + b" calls>\n<" + dsml + b' invoke name="read">\n<' + dsml +
                b' parameter name="path" string="true">src/a.c' + closing[b"parameter"] + b"\n" + closing[b"invoke"] +
                b"\n" + closing[b"calls"] + b" tail")
    for name, stream, _ in list(cases[:3]):
        add(f"{name}, no tool rendered", stream, tools=None)
    return cases


def record(name, stream, tools=TOOLS, thinking=False, stops=(), finish="stop"):
    result = model.answer(stream, tools=tools, thinking=thinking, stops=stops, finish=finish)
    hexed = lambda values: ",".join(value.hex() for value in values)
    return "".join((f"case {name}\n", f"tools {hexed(tools) if tools is not None else '-'}\n",
                    f"stops {hexed(stops)}\n", f"thinking {int(thinking)}\n", "mode tolerant\n",
                    f"finish {finish}\n", f"stream {stream.hex()}\n", f"content {result['content'].hex()}\n",
                    f"reasoning {result['reasoning'].hex()}\n", f"calls {result['calls'].hex()}\n",
                    f"expect_finish {result['finish']}\n",
                    "repairs " + " ".join(str(count) for count in result["repairs"]) + "\n", "end\n"))


def fuzz(seed, count):
    """Random mutations of the variants: bytes put in, taken out or changed (structure bytes weighted), two cases
    spliced, and truncations at the end or by the limit."""
    source, rng = variants(), random.Random(seed)
    for index in range(count):
        name, stream, settings = source[rng.randrange(len(source))]
        stream = bytearray(stream)
        for _ in range(rng.randrange(1, MOST_MUTATIONS + 1)):
            at = rng.randrange(len(stream) + 1)
            action = rng.randrange(5)
            if action == 4:  # a head cut in two, its parts apart: text joined around a block may complete it
                head = rng.choice(JOIN_HEADS)
                cut, later = rng.randrange(1, len(head)), rng.randrange(at, len(stream) + 1)
                stream[later:later] = head[cut:]
                stream[at:at] = head[:cut]
            elif action == 0:
                stream[at:at] = rng.choice(MUTATION_ALPHABET)
            elif action == 1 and stream:
                del stream[at:at + rng.randrange(1, 4)]
            elif action == 2 and stream:
                stream[min(at, len(stream) - 1)] = rng.randrange(256)
            else:
                other = source[rng.randrange(len(source))][1]
                stream[at:] = other[rng.randrange(len(other) + 1):]
        settings = dict(settings)
        if rng.randrange(4) == 0:
            settings["finish"] = "length"
            stream = stream[:rng.randrange(len(stream) + 1)]
        if rng.randrange(8) == 0:
            settings["stops"] = list(FUZZ_STOPS)
        yield f"fuzz {seed}/{index} of {name}", bytes(stream), settings


def reference_check(encoding_path):
    """Every variant encoding.py's parse_message_from_completion_text reads as calls, or as text with no head of the
    grammar in it: the model gives its content, reasoning and calls (name, namespace, arguments), byte for byte."""
    specification = importlib.util.spec_from_file_location("reference_encoding", encoding_path)
    encoding = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(encoding)
    taken = tolerated = folded = 0
    for name, stream, settings in variants():
        if settings.get("tools", TOOLS) is None or settings.get("finish") == "length" or settings.get("stops"):
            continue
        thinking = settings.get("thinking", False)
        try:
            parsed = encoding.parse_message_from_completion_text(
                stream.decode("utf-8") + encoding.eos_token, "thinking" if thinking else "chat")
        except (AssertionError, ValueError, UnicodeDecodeError, IndexError):
            continue
        result = model.answer(stream, tools=TOOLS, thinking=thinking)
        has_head = model.next_head(stream, 0, len(stream))[0] == model.WHOLE
        if not parsed["tool_calls"] and has_head:
            tolerated += 1  # text the reference takes as content though it holds a head: G1's side, not G3's
            continue
        taken += 1
        def resolved(call):
            """The reference's name trimmed and as the declared tools resolve it (the spec's own rules, not a
            reading)."""
            namespace, name = call.get("namespace"), call["function"]["name"]
            full = (namespace + "::" + name if namespace is not None else name).encode()
            full = model.resolve(full[slice(*model.trimmed(full, 0, len(full)))], TOOLS)[0].decode()
            return full.split("::", 1)[-1], namespace

        theirs = [(*resolved(call), call["function"]["arguments"]) for call in parsed["tool_calls"]]
        folded += sum(1 for call, (name, _, _) in zip(parsed["tool_calls"], theirs) if call["function"]["name"] != name)
        mine = [(call["function"]["name"], call.get("namespace"), call["function"]["arguments"])
                for call in json.loads(b"[" + result["calls"] + b"]")]
        if (theirs != mine or parsed["content"].encode() != result["content"] or
                parsed["reasoning_content"].encode() != result["reasoning"]):
            print(f"G3 DIFFERS: {name}\n  reference {parsed['content']!r} {theirs}\n  model "
                  f"{result['content']!r} {mine}")
            return 1
    print(f"G3: all {taken} variants the reference reads give its content, reasoning and calls; {tolerated} it "
          f"takes as text though they hold a head are read as calls (G1); {folded} names trimmed or resolved to a declared tool")
    return 0


def summary(path):
    cases, calls, repairs = 0, 0, [0] * len(model.REPAIR_KINDS)
    with open(path) as handle:
        for line in handle:
            if line.startswith("case "):
                cases += 1
            elif line.startswith("calls "):
                calls += bytes.fromhex(line.split()[1] if len(line.split()) > 1 else "").count(b'{"id":')
            elif line.startswith("repairs "):
                repairs = [total + int(count) for total, count in zip(repairs, line.split()[1:])]
    print(f"{cases} cases, {calls} calls; repairs " +
          ", ".join(f"{kind} {count}" for kind, count in zip(model.REPAIR_KINDS, repairs)))
    return 0


def g3_listings(path, base_path, tolerant_path):
    """The zero-repair cases of a corpus whose whole (not streamed) outputs differ between the two listings; a stream's
    chunks may part differently (a held space), its joined text is the whole answer's (chat_tool_calls_test). Each
    is labelled by its intended cause: a stop string (the head's verdict first), a name with spaces (trimmed)."""
    clean, stops, trims, names, index = set(), {}, {}, {}, -1
    with open(path) as handle:
        for line in handle:
            if line.startswith("case "):
                index += 1
                names[index] = line[len("case "):].strip()
            elif line.startswith("stops "):
                stops[index] = bool(line.split()[1:])
            elif line.startswith("stream "):
                trims[index] = any(model.trimmed(value, 0, len(value)) != (0, len(value)) for value in
                                   re.findall(rb'name="([^"]*)"', bytes.fromhex(line.split()[1]
                                                                                 if len(line.split()) > 1 else "")))
            elif line.startswith("repairs ") and not any(int(count) for count in line.split()[1:]):
                clean.add(index)

    def whole(listing):
        rows = {}
        with open(listing) as handle:
            for line in handle:
                fields = line.split()
                if len(fields) > 5 and fields[4] == "0":
                    rows.setdefault(int(fields[0]), []).append(fields[:6])
        return rows

    base, tolerant = whole(base_path), whole(tolerant_path)
    differing = sorted(case for case in clean if base.get(case) != tolerant.get(case))
    print(f"G3 listings: {len(clean)} zero-repair cases, {sum(len(base.get(case, [])) for case in clean)} whole-answer "
          f"runs; output bytes differ in {len(differing)}: " +
          ", ".join(f"{names[case]}{' (a stop string)' if stops.get(case) else ' (a name trimmed)' if trims.get(case) else ''}"
                    for case in differing))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("variants", "fuzz", "reference-check", "summary", "g3-listings"))
    parser.add_argument("path", nargs="?")
    parser.add_argument("listings", nargs="*")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=1000)
    arguments = parser.parse_args()
    if arguments.action == "reference-check":
        return reference_check(arguments.path)
    if arguments.action == "summary":
        return summary(arguments.path)
    if arguments.action == "g3-listings":
        return g3_listings(arguments.path, *arguments.listings)
    cases = variants() if arguments.action == "variants" else fuzz(arguments.seed, arguments.count)
    out = sys.stdout
    for name, stream, settings in cases:
        out.write(record(name, stream, **settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
