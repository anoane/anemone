#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The DSML tool-call grammar of source/engine/server/api/chat_completions.c, as a model of the whole text (no dependencies).

    answer(stream, tools=[b"name", ...], thinking=False, stops=(), mode="tolerant", finish="stop") -> dict

The model reads a completion's bytes as the answer layer reads them, and returns what a whole Chat Completions
answer holds: the content and reasoning text (U+FFFD for what is not UTF-8), the tool_calls objects as answer.calls
holds them, the finish reason and the count of each repair kind (REPAIR_KINDS, the order of the metric's kinds).
tools None is a request that rendered no tool: nothing is read as a call. Every input is held to G1 (no marker in
the text but those a repair explains, on the joined text) and G2 (each byte is content, reasoning, a call block's
structure or dropped, once; the content and reasoning are their parts; only whitespace or what a closed block or a
stop stopped is dropped).
mode "reference" is v0.0.5-alpha's rule, fed in one piece.

The tolerant grammar (chat_completions.h's Calls):
  - A head is "<" WS ["/" WS] bar WS "DSML" WS bar WS keyword: WS a run of at most SPACE_MAX spaces, tabs, CRs or
    LFs, bar U+FF5C or "|", "DSML" and the keyword in any case; a closing head may leave its "<" out; calls, function_calls and tool_calls open or close
    the block, invoke a call, parameter an argument. A tag runs from its head to its ">", or to the next head or
    the end; attributes in any order, with spaces around "=", quoted with " or ', or bare. A quote that does not
    close before the next head or the end ends at its first ">", which ends the tag, else at that head or the end.
  - In the content, and in reasoning without a thinking budget, an opening block or invoke head starts a block;
    the marker whitespace before it is trimmed, up to its last two line feeds and at most SPACE_MAX bytes. Any other
    head there is text, counted stray.
  - A block opened by the reference's exact head is read by the reference's rule first (v0.0.5-alpha's call_read): an
    invoke it accepts is its reading (G3), its name trimmed. From the first it refuses, where it is unfinished at the
    end, or at a whole closing block head in any spelling in what it takes or after where it waits, the tolerant
    reading takes the rest of the block. At a request limit the reference's EOF forms give no call.
  - Tolerant reading: a parameter ends at the next head of any kind; an invoke at the next invoke, its closing or
    the block's closing, or the end; the block at its closing or the end. At a request limit (finish "length") an
    invoke the end closed is no call: it goes out as content text, counted cut_at_limit. A value is the bytes between its tag's
    end and that head. Names are trimmed and resolved against the declared tools: exactly, else ignoring case and
    -/_, never by prefix. A repeated parameter keeps its first place and its last value. string true/false/1/0
    reads as JSON string or as written; a parameter with no name is passed over; without it a value is written as it
    is when it is JSON (a number, true,
    false, null, an object or an array) and as a string else. Text between tags is content, given out before the
    next call or at the end; a head it forms with the content before it is counted stray. A closed block stops
    generation; a block with no named call goes out as text.
"""
import json
import re

SPACE_MAX = 16  # DSML_MARKER_SPACE_MAX
LINE_FEEDS_TRIMMED = 2
MARKER_SPACES = b" \t\r\n"
PYTHON_ASCII_SPACES = b" \t\n\r\x0c\x0b\x1c\x1d\x1e\x1f"
UNICODE_SPACES = tuple(bytes.fromhex(value) for value in (
    "c285", "c2a0", "e19a80", "e28080", "e28081", "e28082", "e28083", "e28084", "e28085", "e28086", "e28087",
    "e28088", "e28089", "e2808a", "e280a8", "e280a9", "e280af", "e2819f", "e38080"))
WIDE_BAR = "\uff5c".encode()
BARS = (WIDE_BAR, b"|")
HEAD_MOST = len(b"</" + WIDE_BAR + b"dsml" + WIDE_BAR + b"function_calls") + 5 * SPACE_MAX  # MARKER_HEAD_MOST_BYTES
BLOCK, INVOKE, PARAMETER = "block", "invoke", "parameter"
KEYWORDS = ((b"calls", BLOCK), (b"function_calls", BLOCK), (b"tool_calls", BLOCK), (b"invoke", INVOKE),
            (b"parameter", PARAMETER))
REFERENCE_KEYWORDS = {BLOCK: b"calls", INVOKE: b"invoke", PARAMETER: b"parameter"}
DSML = WIDE_BAR + b"DSML" + WIDE_BAR
REASONING_END = b"</think>"
CALLS_BEGIN = b"\n\n<" + DSML + b" calls"
CALLS_END = b"</" + DSML + b" calls>"
INVOKE_BEGIN = b"<" + DSML + b" invoke"
INVOKE_END = b"</" + DSML + b" invoke"
PARAMETER_BEGIN = b"<" + DSML + b" parameter"
PARAMETER_END = b"/" + DSML + b" parameter"
NAME_OPENING, NAME_CLOSING = b'name="', b'">'
NAMESPACE_SEPARATOR = b"::"
PARAMETER_NAME_OPENING = b' name="'
STRING_TRUE, STRING_FALSE = b'" string="true">', b'" string="false">'
TRUE_WORDS, FALSE_WORDS = (b"true", b"1"), (b"false", b"0")
REPAIR_KINDS = ("block_start", "marker_spelling", "attribute", "closing_implied", "marker_stray",
                "parameter_repeated", "name_folded", "name_unknown", "invoke_unnamed", "text_in_block",
                "block_as_text", "cut_at_limit")
TOOL_NAME_MOST = 1024  # TOOL_NAME_MOST_BYTES: a longer declared name is not kept, and the names are then not whole
REPLACEMENT = "\ufffd".encode()
JSON_MAXIMUM_DEPTH = 128
JSON_MOST_INTEGER_DIGITS = 4300
WHOLE, PARTIAL, NONE = "whole", "partial", "none"
UNFINISHED, CALL_WHOLE, CALLS_CLOSED, CALLS_MALFORMED = "unfinished", "whole", "closed", "malformed"


# ---- bytes --------------------------------------------------------------------------------------------------

def python_space_at(text, at, end):
    """The bytes of the Python whitespace character at at (0: none), as v0.0.5-alpha's spaces_skipped reads it."""
    if at >= end:
        return 0
    if text[at] in PYTHON_ASCII_SPACES:
        return 1
    for space in UNICODE_SPACES:
        if at + len(space) <= end and text.startswith(space, at):
            return len(space)
    return 0


def spaces_skipped(text, at, end):
    while python_space_at(text, at, end) != 0:
        at += python_space_at(text, at, end)
    return at


def trimmed(text, begin, end):
    """The span without the Python whitespace at both its ends."""
    begin = spaces_skipped(text, begin, end)
    last, at = begin, begin
    while at < end:
        space = python_space_at(text, at, end)
        at += space if space != 0 else 1
        last = at if space == 0 else last
    return begin, last


def utf8_sequence(text, at, end):
    """(bytes, is_valid) of the sequence at at, as tokenizer.c's sequence_length reads it."""
    lead = text[at]
    length = 1 if lead < 0xC2 or lead >= 0xF5 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
    low = 0xA0 if lead == 0xE0 else 0x90 if lead == 0xF0 else 0x80
    high = 0x9F if lead == 0xED else 0x8F if lead == 0xF4 else 0xBF
    valid = lead < 0x80 or length > 1
    for index in range(1, length):
        if at + index >= end or not low <= text[at + index] <= high:
            return index, False
        low, high = 0x80, 0xBF
    return length, valid


def decoded(raw):
    """text_given_out's text of one run: U+FFFD for each ill-formed part."""
    out, at = bytearray(), 0
    while at < len(raw):
        length, valid = utf8_sequence(raw, at, len(raw))
        out += raw[at:at + length] if valid else REPLACEMENT
        at += length
    return bytes(out)


def json_string(raw):
    """json_write_string: '"', '\\' and control bytes escaped, every other byte as it is."""
    escapes = {0x22: b'\\"', 0x5C: b"\\\\", 0x08: b"\\b", 0x0C: b"\\f", 0x0A: b"\\n", 0x0D: b"\\r", 0x09: b"\\t"}
    out = bytearray(b'"')
    for byte in raw:
        out += escapes.get(byte, b"\\u%04x" % byte if byte < 0x20 else bytes((byte,)))
    return bytes(out + b'"')


def is_json_value(raw):
    """Whether a value without a string attribute is written as it is: a whole document json.h reads as valid (as
    json.loads does, at most JSON_MAXIMUM_DEPTH deep, no lone surrogate escape, integers of at most
    JSON_MOST_INTEGER_DIGITS digits), a number, true, false, null, an object or an array, holding no NaN or
    Infinity."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    stripped = text.strip(" \t\n\r")
    if not stripped or stripped[0] not in "{[-0123456789tfn":
        return False
    depth, most, in_string, escaped = 0, 0, False, False
    for character in text:
        if in_string:
            escaped, in_string = (False, True) if escaped else (character == "\\", character != '"')
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            most = max(most, depth)
        elif character in "]}":
            depth -= 1
    if most > JSON_MAXIMUM_DEPTH:
        return False

    def refused(_):
        raise ValueError("NaN or Infinity")

    try:
        value = json.loads(text, parse_constant=refused)
    except (ValueError, RecursionError):
        return False

    def has_lone_surrogate(item):
        if isinstance(item, str):
            return any(0xD800 <= ord(character) <= 0xDFFF for character in item)
        if isinstance(item, list):
            return any(has_lone_surrogate(element) for element in item)
        if isinstance(item, dict):
            return any(has_lone_surrogate(key) or has_lone_surrogate(element) for key, element in item.items())
        return False

    if has_lone_surrogate(value):
        return False
    for number in re.finditer(r'"(?:\\.|[^"\\])*"|-?(\d+)(?![.\deE])', text):
        if number.group(1) is not None and len(number.group(1)) > JSON_MOST_INTEGER_DIGITS:
            return False
    return True


# ---- heads and tags -----------------------------------------------------------------------------------------

class Head:
    def __init__(self, begin, end, closing, kind, exact):
        self.begin, self.end, self.closing, self.kind, self.exact = begin, end, closing, kind, exact

    def opens_block(self):
        return not self.closing and self.kind in (BLOCK, INVOKE)


def head_at(text, at, end):
    """(WHOLE, Head) for a whole head at at, (PARTIAL, None) when the text ends inside what may still be one, else
    (NONE, None). A closing head may leave its "<" out (the reference's own PARAMETER_END has none): it then begins
    at its "/" and is never spelled exactly."""
    bare, position = text[at] == ord("/"), at + 1

    def spaces(position):
        start = position
        while position < end and text[position] in MARKER_SPACES:
            position += 1
        return position if position - start <= SPACE_MAX else None

    def literal(position, options, folds):
        """(WHOLE, end, option), PARTIAL or NONE for the first option the bytes at position spell."""
        for option in options:
            available = text[position:min(end, position + len(option))]
            if (available.lower() if folds else available) == (option.lower() if folds else option)[:len(available)]:
                return (WHOLE, position + len(option), option) if len(available) == len(option) else \
                    (PARTIAL, None, None)
        return NONE, None, None

    closing, position = bare, spaces(position)
    if not bare and position is not None and position < end and text[position] == ord("/"):
        closing, position = True, spaces(position + 1)
    steps = ((BARS, False, True), ((b"dsml",), True, True), (BARS, False, True),
             ([word for word, _ in KEYWORDS], True, False))
    for options, folds, spaced in steps:
        if position is None:
            return NONE, None
        if position == end:
            return PARTIAL, None
        found, position, option = literal(position, options, folds)
        if found != WHOLE:
            return found, None
        position = spaces(position) if spaced else position
    kind = dict(KEYWORDS)[option]
    exact = text[at:position] == (b"</" if closing else b"<") + DSML + b" " + REFERENCE_KEYWORDS[kind]
    return WHOLE, Head(at, position, closing, kind, exact)


def heads_in(raw):
    """The whole heads of a text, as the G1 check counts them."""
    heads, position = 0, 0
    while True:
        found, value = next_head(raw, position, len(raw))
        if found != WHOLE:
            return heads
        heads, position = heads + 1, value.end


def closing_between(text, at, until, end):
    """Whether a whole closing block head, in any spelling, begins from at to before until."""
    position = at
    while True:
        found, value = next_head(text, position, end)
        if found != WHOLE or value.begin >= until:
            return False
        if value.closing and value.kind == BLOCK:
            return True
        position = value.end


def next_head(text, at, end):
    """(WHOLE, head) of the first whole head at or after at; (PARTIAL, begin) when none is whole before the end and
    one may still start at begin; else (NONE, end)."""
    while True:
        starts = [found for found in (text.find(b"<", at, end), text.find(b"/", at, end)) if found >= 0]
        if not starts:
            return NONE, end
        at = min(starts)
        found, head = head_at(text, at, end)
        if found == WHOLE:
            return WHOLE, head
        if found == PARTIAL:
            return PARTIAL, at
        at += 1


class Tag:
    def __init__(self, end, attributes, text):
        self.end, self.attributes, self.text = end, attributes, text

    def value(self, wanted):
        """The last value of the attribute (its name in any case), None when it has none."""
        found = None
        for name, value in self.attributes:
            found = value if name == wanted and value is not None else found
        return found


def tag_read(text, head, end, is_last):
    """The tag after head: its attributes and where it ends (after its ">", at the next head, or at the end); None
    while bytes still to come may change it."""
    attributes, position = [], head.end

    def stop_at(position):
        """'head' at a whole head, 'wait' at one that may still be, 'end' at the end, else None."""
        if position == end:
            return "end"
        if text[position] in b"</":
            found, _ = head_at(text, position, end)
            return "head" if found == WHOLE else "wait" if found == PARTIAL and not is_last else None
        return None

    def ended(at):
        return Tag(at, attributes, text[head.end:at])

    while True:
        position = spaces_skipped(text, position, end)
        stop = stop_at(position)
        if stop in ("wait",) or (stop == "end" and not is_last):
            return None
        if stop is not None:
            return ended(position)
        if text[position] == ord(">"):
            return ended(position + 1)
        name_end = position
        while (stop_at(name_end) is None and python_space_at(text, name_end, end) == 0
               and text[name_end] not in b"=>\"'"):
            name_end += 1
        if stop_at(name_end) == "wait" or (stop_at(name_end) == "end" and not is_last):
            return None
        if name_end == position:
            attributes.append((None, None))  # a stray "=" or quote
            position += 1
            continue
        name = text[position:name_end].lower()
        position = spaces_skipped(text, name_end, end)
        if stop_at(position) == "wait" or (stop_at(position) == "end" and not is_last):
            return None
        if position == end or text[position] != ord("="):
            attributes.append((name, None))
            continue
        position = spaces_skipped(text, position + 1, end)
        stop = stop_at(position)
        if stop == "wait" or (stop == "end" and not is_last):
            return None
        if stop is None and text[position] in b"\"'":
            quote, value_begin, scan, greater = text[position], position + 1, position + 1, None
            while stop_at(scan) is None and text[scan] != quote:
                greater = scan if greater is None and text[scan] == ord(">") else greater
                scan += 1
            stop = stop_at(scan)
            if stop == "wait" or (stop == "end" and not is_last):
                return None
            if stop is None:
                attributes.append((name, text[value_begin:scan]))
                position = scan + 1
                continue
            if greater is not None:
                attributes.append((name, text[value_begin:greater]))
                return ended(greater + 1)
            attributes.append((name, text[value_begin:scan]))
            position = scan
            continue
        value_end = position
        while (stop_at(value_end) is None and python_space_at(text, value_end, end) == 0
               and text[value_end] != ord(">")):
            value_end += 1
        if stop_at(value_end) == "wait" or (stop_at(value_end) == "end" and not is_last):
            return None
        attributes.append((name, text[position:value_end]))
        position = value_end


def string_form(tag):
    value = tag.value(b"string")
    word = value.lower() if value is not None else None
    return True if word in TRUE_WORDS else False if word in FALSE_WORDS else None


# ---- the reference's reading: v0.0.5-alpha's call_read ------------------------------------------------------------

def first_found(text, at, end, first, second):
    at_first, at_second = text.find(first, at, end), text.find(second, at, end)
    if at_first < 0 and at_second < 0:
        return end, 0
    which = 1 if at_second >= 0 and (at_first < 0 or at_second < at_first) else 0
    return (at_second if which else at_first), which


def is_call_separator(text, begin, end):
    return begin < end and text[begin] == ord(">") and spaces_skipped(text, begin + 1, end) == end


def parameter_read(text, begin, end):
    if end > begin and text[end - 1] == ord("\n"):
        end -= 1
    name_begin = begin + len(PARAMETER_NAME_OPENING)
    if (end - begin < len(PARAMETER_NAME_OPENING) + 1 or text[end - 1] != ord("<")
            or not text.startswith(PARAMETER_NAME_OPENING, begin)):
        return None
    true_at, false_at = text.find(STRING_TRUE, name_begin, end - 1), text.find(STRING_FALSE, name_begin, end - 1)
    is_string = true_at >= 0 and (false_at < 0 or true_at < false_at)
    mark = true_at if is_string else false_at
    if mark < 0:
        return None
    value_begin = mark + len(STRING_TRUE if is_string else STRING_FALSE)
    return text[name_begin:mark], text[value_begin:end - 1], is_string


def member_text(name, value, is_string):
    return json_string(name) + b": " + (json_string(value) if is_string else value)


def reference_call(text, read, end, is_last):
    """(status, name, arguments, read after): one invoke as call_read reads it."""
    at, which = first_found(text, read, end, INVOKE_BEGIN, CALLS_END)
    if at == end:
        return UNFINISHED, None, None, read
    if not is_call_separator(text, read, at):
        return CALLS_MALFORMED, None, None, read
    if which == 1:
        return CALLS_CLOSED, None, None, at + len(CALLS_END)
    name_begin = at + len(INVOKE_BEGIN)
    at, which = first_found(text, name_begin, end, PARAMETER_BEGIN, INVOKE_END)
    if at == end and not is_last:
        return UNFINISHED, None, None, read
    name_begin = spaces_skipped(text, name_begin, at)
    if at - name_begin < len(NAME_OPENING) + len(NAME_CLOSING) or not text.startswith(NAME_OPENING, name_begin):
        return CALLS_MALFORMED, None, None, read
    name_begin += len(NAME_OPENING)
    name_end = name_begin
    while True:
        closing = text.find(NAME_CLOSING, name_end, at)
        if closing < 0:
            return CALLS_MALFORMED, None, None, read
        name_end = closing
        if spaces_skipped(text, name_end + len(NAME_CLOSING), at) == at:
            break
        name_end += 1
    name = text[name_begin:name_end]
    qualified = name.find(NAMESPACE_SEPARATOR)
    if qualified >= 0 and name.find(NAMESPACE_SEPARATOR, qualified + len(NAMESPACE_SEPARATOR)) >= 0:
        return CALLS_MALFORMED, None, None, read
    members, names = [], []
    which = 1 if at == end else which
    while which == 0:
        content_begin = at + len(PARAMETER_BEGIN)
        content_end = text.find(PARAMETER_END, content_begin, end)
        if content_end < 0:
            return UNFINISHED, None, None, read
        parameter = parameter_read(text, content_begin, content_end)
        if parameter is None or parameter[0] in names:
            return CALLS_MALFORMED, None, None, read
        names.append(parameter[0])
        members.append(member_text(*parameter))
        after = content_end + len(PARAMETER_END)
        at, which = first_found(text, after, end, PARAMETER_BEGIN, INVOKE_END)
        if at == end and not is_last:
            return UNFINISHED, None, None, read
        if not is_call_separator(text, after, at):
            return CALLS_MALFORMED, None, None, read
        which = 1 if at == end else which
    after_invoke = at + len(INVOKE_END) if at < end else at
    if after_invoke == end and not is_last:
        return UNFINISHED, None, None, read
    if after_invoke < end and text[after_invoke] != ord(">"):
        return CALLS_MALFORMED, None, None, read
    return CALL_WHOLE, name, b"{" + b", ".join(members) + b"}", after_invoke


# ---- the tolerant reading of one unit ------------------------------------------------------------------------

class Unit:
    """One step of a block's tolerant reading from a place between tags: kind "call", "unnamed", "closed" or
    "end", where the next begins, the text between tags it passed, and its repairs."""

    def __init__(self):
        self.kind, self.name, self.arguments, self.next = None, None, None, None
        self.junk, self.repairs = [], dict.fromkeys(REPAIR_KINDS, 0)
        self.spans = []  # the bytes it reads as structure: its invoke, its tags

    def text_between(self, text, begin, end):
        begin, end = trimmed(text, begin, end)
        if end > begin:
            self.junk.append((begin, end))
            self.repairs["text_in_block"] += 1

    def tag_counted(self, head, tag, exact_text):
        self.repairs["marker_spelling"] += 0 if head.exact else 1
        self.repairs["attribute"] += 0 if tag.text == exact_text else 1


def resolve(name, declared):
    """The declared tool a name is, exactly or ignoring case and -/_, and the repair; the name itself when none
    (name_unknown only when every declared name is kept: none longer than TOOL_NAME_MOST)."""
    kept = [tool for tool in declared if len(tool) <= TOOL_NAME_MOST]
    if not declared or name in kept:
        return name, None
    folded = name.lower().replace(b"-", b"_")
    for tool in kept:
        if tool.lower().replace(b"-", b"_") == folded:
            return tool, "name_folded"
    return name, "name_unknown" if len(kept) == len(declared) else None


def invoke_read(text, end, is_last, unit, head, declared, limited=False):
    """The invoke from head (an invoke head, or a parameter head, which implies one without a name) to its end. At a
    request limit (limited, is_last) an invoke the end cut, closed by no head, is kind "cut": text, never a call."""
    name, parameters, parameter, position, is_cut = None, [], None, head.begin, False
    repairs_before, junk_before = dict(unit.repairs), len(unit.junk)
    if head.kind == INVOKE:
        tag = tag_read(text, head, end, is_last)
        if tag is None:
            return None
        found = tag.value(b"name")
        unit.tag_counted(head, tag, b' name="' + (found or b"") + b'">')
        if found is not None:
            name = found[slice(*trimmed(found, 0, len(found)))]
        position = tag.end
    while True:
        found, value = next_head(text, position, end)
        if found != WHOLE and not is_last:
            return None
        if found != WHOLE:
            if parameter is not None:
                parameter[3] = end
                unit.repairs["closing_implied"] += 1
            else:
                unit.text_between(text, position, end)
            unit.repairs["closing_implied"] += 1
            unit.next, is_cut = end, limited
            break
        marker = value
        if parameter is not None:
            parameter[3] = marker.begin
            if marker.closing and marker.kind == PARAMETER:
                tag = tag_read(text, marker, end, is_last)
                if tag is None:
                    return None
                unit.tag_counted(marker, tag, b">")
                parameter, position = None, tag.end
                continue
            parameter = None
            unit.repairs["closing_implied"] += 1
        else:
            unit.text_between(text, position, marker.begin)
        if marker.kind == PARAMETER and not marker.closing:
            tag = tag_read(text, marker, end, is_last)
            if tag is None:
                return None
            parameter_name, written = tag.value(b"name"), tag.value(b"string")
            unit.tag_counted(marker, tag, b' name="' + (parameter_name or b"") + b'" string="' + (written or b"") +
                             b'">' if written in (b"true", b"false") and parameter_name is not None else None)
            parameter_name = b"" if parameter_name is None else parameter_name[
                slice(*trimmed(parameter_name, 0, len(parameter_name)))]
            parameter = [parameter_name, string_form(tag), tag.end, None]
            parameters.append(parameter)
            position = tag.end
            continue
        if marker.kind == INVOKE and not marker.closing or marker.kind == BLOCK and marker.closing:
            unit.repairs["closing_implied"] += 1
            unit.next = marker.begin
            break
        tag = tag_read(text, marker, end, is_last)
        if tag is None:
            return None
        unit.tag_counted(marker, tag, b">")
        if marker.kind == INVOKE:
            unit.next = tag.end
            break
        unit.repairs["marker_stray"] += 1  # an opening block head or a closing parameter one inside the invoke
        position = tag.end
    if not name:
        unit.kind = "unnamed"
        unit.repairs["invoke_unnamed"] += 1
        unit.spans.append((head.begin, unit.next))
        return unit
    if is_cut:
        unit.repairs, unit.junk = repairs_before, unit.junk[:junk_before]
        unit.repairs["cut_at_limit"] += 1
        unit.kind, unit.next = "cut", head.begin
        return unit
    unit.spans.append((head.begin, unit.next))
    resolved, repair = resolve(name, declared)
    if repair is not None:
        unit.repairs[repair] += 1
    values = {}
    for parameter_name, form, value_begin, value_end in parameters:
        if not parameter_name:
            continue  # a parameter with no name is passed over (its tag is counted)
        if parameter_name in values:
            unit.repairs["parameter_repeated"] += 1
        values[parameter_name] = (form, text[value_begin:value_end])  # the first place, the last value
    members = [member_text(key, value, form is True or form is None and not is_json_value(value))
               for key, (form, value) in values.items()]
    unit.kind, unit.name, unit.arguments = "call", resolved, b"{" + b", ".join(members) + b"}"
    return unit


def tolerant_unit(text, at, end, is_last, declared, opening, limited=False):
    """The next unit from at; None while unfinished. opening is the block's own opening head's place."""
    unit, position = Unit(), at
    while True:
        found, value = next_head(text, position, end)
        if found != WHOLE and not is_last:
            return None
        if found != WHOLE:
            unit.text_between(text, position, end)
            unit.kind, unit.next = "end", end
            return unit
        marker = value
        unit.text_between(text, position, marker.begin)
        if marker.kind in (INVOKE, PARAMETER) and not marker.closing:
            return invoke_read(text, end, is_last, unit, marker, declared, limited)
        tag = tag_read(text, marker, end, is_last)
        if tag is None:
            return None
        unit.tag_counted(marker, tag, b">")
        unit.spans.append((marker.begin, tag.end))
        if marker.kind == BLOCK and marker.closing:
            unit.kind, unit.next = "closed", tag.end
            return unit
        if marker.begin != opening:
            unit.repairs["marker_stray"] += 1
        position = tag.end


# ---- the answer ---------------------------------------------------------------------------------------------

def spaces_trimmed(text, begin, head_begin):
    """How many bytes of marker whitespace right before head_begin (none before begin) the block trims: up to the
    last two line feeds, at most SPACE_MAX."""
    length, feeds = 0, 0
    while head_begin - length > begin and text[head_begin - length - 1] in MARKER_SPACES:
        length += 1
        feeds += 1 if text[head_begin - length] == ord("\n") else 0
        if feeds == LINE_FEEDS_TRIMMED:
            break
    return min(length, SPACE_MAX)


class Answer:
    def __init__(self, stream, tools, thinking, stops, mode, finish, identifier):
        self.text, self.tools_on, self.declared = stream, tools is not None, list(tools or [])
        self.thinking, self.stops, self.mode, self.finish_reason = thinking, list(stops), mode, finish
        self.identifier = identifier
        self.content, self.reasoning, self.calls = bytearray(), bytearray(), []
        self.repairs = dict.fromkeys(REPAIR_KINDS, 0)
        self.stopped, self.parts = False, []  # parts: (begin, end, label), every byte once (G2)
        self.raw_content, self.structure, self.cut_heads = bytearray(), [], 0

    def given(self, begin, end, is_reasoning):
        (self.reasoning if is_reasoning else self.content).extend(decoded(self.text[begin:end]))
        if not is_reasoning:
            self.raw_content += self.text[begin:end]
        self.parts.append((begin, end, "reasoning" if is_reasoning else "content"))
        position = begin
        while self.tools_on and self.mode == "tolerant":
            found, value = next_head(self.text, position, end)
            if found != WHOLE:
                break
            self.repairs["marker_stray"] += 1
            position = value.end

    def call(self, name, arguments):
        qualified = name.find(NAMESPACE_SEPARATOR)
        bare = name[qualified + len(NAMESPACE_SEPARATOR):] if qualified >= 0 else name
        self.calls.append(b'{"id":' + json_string(self.identifier + b"-%d" % len(self.calls)) +
                          b',"type":"function","function":{"name":' + json_string(bare) + b',"arguments":' +
                          json_string(arguments) + b"}" +
                          (b',"namespace":' + json_string(name[:qualified]) if qualified >= 0 else b"") + b"}")

    def delimiter(self, at, is_reasoning):
        """(begin, end, kind, head) of the first delimiter at or after at; a stop wins a tie."""
        end, best = len(self.text), None
        candidates = [(self.text.find(REASONING_END, at), len(REASONING_END), "reasoning_end")] if is_reasoning \
            else [(self.text.find(stop, at), len(stop), "stop") for stop in self.stops]
        for found, length, kind in candidates:
            if found >= 0 and (best is None or found < best[0]):
                best = (found, found + length, kind, None)
        if self.tools_on and self.mode == "reference":
            found = self.text.find(CALLS_BEGIN, at)
            if found >= 0 and (best is None or found < best[0]):
                best = (found, found + len(CALLS_BEGIN), "block", None)
        position = at
        while self.tools_on and self.mode == "tolerant":
            found, head = next_head(self.text, position, end)
            if found != WHOLE:
                break
            if head.opens_block():
                begin = head.begin - spaces_trimmed(self.text, at, head.begin)
                if best is None or begin < best[0]:
                    best = (begin, head.end, "block", head)
                break
            position = head.end
        return best

    def run(self):
        at, is_reasoning, end = 0, self.thinking, len(self.text)
        while at < end:
            found = self.delimiter(at, is_reasoning)
            if found is None:
                self.given(at, end, is_reasoning)
                break
            begin, after, kind, head = found
            self.given(at, begin, is_reasoning)
            if kind == "reasoning_end":
                self.parts.append((begin, after, "delimiter"))
                at, is_reasoning = after, False
            elif kind == "stop":
                self.parts.append((begin, end, "stopped"))
                self.stopped = True
                break
            else:
                (self.reference_block if self.mode == "reference" else self.block)(begin, after, head)
                break
        return self.result()

    def joined(self, raw):
        """Text between tags given out after the content so far: each head it forms, alone or across the join
        (G1 on the client's joined text), counted stray, as join_counted counts it."""
        window = bytes(self.raw_content[-(SPACE_MAX + HEAD_MOST):])
        join, at = window + raw, 0
        while True:
            starts = [found for found in (join.find(b"<", at), join.find(b"/", at)) if found >= 0]
            if not starts:
                break
            at = min(starts)
            found, head = head_at(join, at, len(join))
            if found == WHOLE and head.end > len(window):
                self.repairs["marker_stray"] += 1
                at = head.end
                continue
            at += 1
        self.raw_content += raw
        self.content += decoded(raw)

    def block(self, begin, after, head):
        """The block from begin (its trimmed whitespace) to its closing or the end."""
        text, end = self.text, len(self.text)
        limited = self.finish_reason == "length"
        repairs = dict.fromkeys(REPAIR_KINDS, 0)
        reference_opened = head.exact and head.kind == BLOCK
        repairs["block_start"] = 0 if reference_opened and text[begin:head.begin] == b"\n\n" else 1
        tolerant, cursor, calls_before = not reference_opened, head.begin if not reference_opened else head.end, \
            len(self.calls)
        pending, junk, closed_at, cut_at, finished = bytearray(), [], None, None, False
        structure = [(head.begin, head.end)] if reference_opened else []
        for is_last in (False, True):
            while not finished:
                if not tolerant:
                    status, name, arguments, read = reference_call(text, cursor, end, is_last and not limited)
                    if status == CALL_WHOLE and closing_between(text, cursor, read, end):
                        status = CALLS_MALFORMED  # a closing block head in what it takes ends its reading
                    if status == CALL_WHOLE:
                        resolved, repair = resolve(name[slice(*trimmed(name, 0, len(name)))], self.declared)
                        if repair is not None:
                            repairs[repair] += 1
                        self.call(resolved, arguments)
                        structure.append((cursor, read))
                        cursor = read
                    elif status == CALLS_CLOSED:
                        structure.append((cursor, read))
                        closed_at, self.stopped, finished = read, not is_last, True
                    elif status == CALLS_MALFORMED or is_last or closing_between(text, cursor, end, end):
                        tolerant, closes = True, text[cursor:cursor + 1] == b">"
                        if len(self.calls) != calls_before and closes:
                            structure.append((cursor, cursor + 1))  # the last call's ">"
                        cursor = head.begin if len(self.calls) == calls_before else cursor + (1 if closes else 0)
                    else:
                        break
                    continue
                unit = tolerant_unit(text, cursor, end, is_last, self.declared, head.begin, limited)
                if unit is None:
                    break
                for kind in REPAIR_KINDS:
                    repairs[kind] += unit.repairs[kind]
                junk += unit.junk
                structure += unit.spans
                pending += b"".join(text[a:b] for a, b in unit.junk)
                cursor = unit.next
                if unit.kind == "call":
                    if pending:
                        self.joined(bytes(pending))
                    pending = bytearray()
                    self.call(unit.name, unit.arguments)
                elif unit.kind in ("closed", "end", "cut"):
                    closed_at = unit.next if unit.kind == "closed" else None
                    cut_at = unit.next if unit.kind == "cut" else None
                    repairs["closing_implied"] += 1 if unit.kind == "end" and head.kind == BLOCK else 0
                    self.stopped, finished = unit.kind == "closed" and not is_last, True
            if finished:
                break
        block_end = closed_at if closed_at is not None else end
        if len(self.calls) == calls_before:
            repairs = dict.fromkeys(REPAIR_KINDS, 0)
            repairs["block_as_text"] = 1
            self.content += decoded(text[begin:block_end])
            self.parts.append((begin, block_end, "content"))
        else:
            if pending:
                self.joined(bytes(pending))
            structure_end = cut_at if cut_at is not None else block_end
            self.parts += [(begin, head.begin, "trimmed")] + [(a, b, "content") for a, b in junk]
            position = head.begin
            for junk_begin, junk_end in junk:
                self.parts.append((position, junk_begin, "block"))  # the call block's own bytes
                position = junk_end
            self.parts.append((position, structure_end, "block"))
            if cut_at is not None:
                self.content += decoded(text[cut_at:end])
                self.raw_content += text[cut_at:end]
                self.cut_heads += heads_in(decoded(text[cut_at:end]))
                self.parts.append((cut_at, end, "content"))
            self.structure += structure
        self.parts.append((block_end, end, "stopped"))  # what a closed block stopped
        for kind in REPAIR_KINDS:
            self.repairs[kind] += repairs[kind]

    def reference_block(self, begin, after, _head):
        """v0.0.5-alpha fed in one piece: the reference's reading; a closed block ends the answer; with no call the
        block goes out as text; after given out calls only the unread rest does."""
        text, end, cursor, closed, calls_before = self.text, len(self.text), after, False, len(self.calls)
        for is_last in (False, True):
            while not closed:
                status, name, arguments, read = reference_call(text, cursor, end, is_last)
                if status == CALL_WHOLE:
                    self.call(name, arguments)
                    cursor = read
                    continue
                if status == CALLS_CLOSED:
                    closed, cursor, self.stopped = True, read, not is_last
                break
        if len(self.calls) == calls_before:
            self.content += decoded(text[begin:end])
            self.parts.append((begin, end, "content"))
            self.repairs["block_as_text"] += 1
        elif not closed and cursor < end:
            rest = cursor + (1 if text[cursor:cursor + 1] == b">" else 0)
            self.content += decoded(text[rest:end])
            self.parts += [(begin, rest, "block"), (rest, end, "content")]
            self.repairs["block_as_text"] += 1
        else:
            self.parts += [(begin, cursor, "block"), (cursor, end, "stopped" if self.stopped else "block")]

    def result(self):
        parts, position = sorted(part for part in self.parts if part[1] > part[0]), 0
        for begin, end, _ in parts:
            assert begin == position, ("G2: a byte not read exactly once", begin, position)
            position = end
        assert position == len(self.text), ("G2: bytes left", position, len(self.text))
        if decoded(self.text) == self.text:  # the text the parts give out is what the answer holds
            for label, held in (("content", self.content), ("reasoning", self.reasoning)):
                assert b"".join(self.text[a:b] for a, b, kind in parts if kind == label) == bytes(held), \
                    ("G2: the " + label + " is not its parts", bytes(held))
        for begin, end, kind in parts:
            assert kind != "stopped" or self.stopped, ("G2: bytes dropped, though nothing stopped", begin, end)
            assert kind != "trimmed" or all(byte in MARKER_SPACES for byte in self.text[begin:end]), \
                ("G2: trimmed bytes that are no marker whitespace", begin, end)
        if self.mode == "tolerant":  # every byte of a block with calls is structure it read, or whitespace
            covered = bytearray(len(self.text))
            for begin, end in self.structure:
                covered[begin:end] = b"\x01" * (end - begin)
            for begin, end, kind in parts:
                at = begin
                while kind == "block" and at < end:
                    space = python_space_at(self.text, at, end)
                    assert covered[at] or space, ("G2: a block byte neither read nor whitespace", at)
                    at += 1 if covered[at] or not space else space
        reason = "stop" if self.stopped else self.finish_reason
        reason = "tool_calls" if self.calls and reason == "stop" else reason
        if self.tools_on and self.mode == "tolerant" and not self.repairs["block_as_text"]:
            heads = heads_in(bytes(self.content)) + heads_in(bytes(self.reasoning))
            assert heads <= self.repairs["marker_stray"] + self.cut_heads, \
                "G1: a marker in the text that no repair explains"
        return {"content": bytes(self.content), "reasoning": bytes(self.reasoning), "calls": b",".join(self.calls),
                "call_count": len(self.calls), "finish": reason,
                "repairs": [self.repairs[kind] for kind in REPAIR_KINDS]}


def answer(stream, tools=None, thinking=False, stops=(), mode="tolerant", finish="stop", identifier=b"chatcmpl-1"):
    return Answer(stream, tools, thinking, stops, mode, finish, identifier).run()
