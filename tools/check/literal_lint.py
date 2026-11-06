#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fails on text the engine's rules forbid, and on anything this tool cannot read: it never passes what it skipped.

    literal_lint.py <directory or file> [<directory or file> ...]

Reads every .c, .h, .cu and .cuh file (any other file is reported; a directory named check_cases is left alone
unless it is named on the command line) and reports:

  - A bare number. Only 0 and 1 may appear in code, spelled 0 or 1 with an integer suffix (0u, 1ull), or 0.0 and
    1.0 with an optional f or l. Every other spelling is reported, the ones this tool does not know included:
    4096, 0x40, 0x1.8p4, 0b101, 1'000, 1e9, 2.5f32. A character constant spelled as a number ('\\x14') is one.
    A number is allowed where it gets its name, and only there: in an object-like #define of a constant (a macro
    with parameters, or one whose text is a statement, is code); between the braces of an enum definition (enum,
    an optional tag, the brace); between the braces of a `static const` table. Even there a template argument
    (stamp<8>) is reported, and so is a table this tool cannot follow: one that never closes, that opens inside
    another, or that a preprocessor line cuts.
  - A comment, string or character constant that does not close (no literal crosses a line end), and a character
    outside ASCII in code.
  - Forbidden text, searched in the raw text with the comments: whatever silences a check or changes results.
  - `++`, `--` or an assignment in the argument of a hook: a build without the hook does not evaluate it.
  - A prototype, a static function or a table member that returns result_status without MUST_CHECK
    (the definition of a function with a prototype inherits it).

Check tool: not part of the engine.
"""
import os
import re
import sys

ENGINE_SUFFIXES = (".c", ".h", ".cu", ".cuh")
CASES_DIRECTORY_NAME = "check_cases"
EXIT_STATUS_USAGE = 2

# One scanner. A number is tried before a character constant, so a digit separator never opens one; no literal
# crosses a line end, so a quote that does not close hides nothing and is reported.
TOKEN = re.compile("|".join((
    r"(?P<comment>/\*.*?\*/|//[^\n]*)",
    r'(?P<string>"(?:\\[^\n]|[^"\\\n])*")',
    r"(?P<number>(?<![A-Za-z0-9_])\.?[0-9](?:[eEpP][+-]|'(?=[0-9A-Za-z_])|[0-9A-Za-z_.])*)",
    r"(?P<character>'(?:\\[^\n]|[^'\\\n])*')",
    r"""(?P<unclosed>/\*|["'])""",
)), re.DOTALL)
# Allowed by spelling, not by value: a form this tool does not know is reported, never skipped.
ALLOWED_NUMBER = re.compile(r"[01](?:[uU](?:l|L|ll|LL)?|(?:l|L|ll|LL)[uU]?)?|[01]\.0?[fFlL]?")
NUMERIC_CHARACTER = re.compile(r"'\\(?:x[0-9A-Fa-f]+|[0-7]+)'")
TERMINATOR_CHARACTER = r"'\0'"
NOT_BLANKED = re.compile(r"[^\n]")
OUTSIDE_ASCII = re.compile(r"[^\x00-\x7f]+")

# Where a number gets its name.
CONSTANT_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+[A-Za-z_]\w*(?![\w(])(?P<text>(?:\\\n|[^\n])*)",
                             re.MULTILINE | re.ASCII)
SIDE_EFFECT = re.compile(r"\+\+|--|<<=|>>=|(?<![=!<>])[-+*/%&|^]?=(?!=)")
NOT_A_CONSTANT = re.compile(r"[;{}\[\]]|\b(?:return|if|else|while|for|do|switch|case|goto|break|continue)\b|"
                            + SIDE_EFFECT.pattern, re.ASCII)
ENUMERATION_START = re.compile(r"\benum\b\s*(?:[A-Za-z_]\w*\s*)?\{", re.ASCII)
CONSTANT_TABLE_START = re.compile(r"\bstatic\s+const\b[^;{}=]*=\s*\{", re.ASCII)
PREPROCESSOR_LINE = re.compile(r"^[ \t]*#", re.MULTILINE)
TEMPLATE_ARGUMENTS = re.compile(r"\w\s*<(?![<=])(?P<arguments>[\w\s,.:+*/'-]*)>", re.ASCII)

HOOK_USE = re.compile(r"\b(?:ASSERT(?:_TEXT)?|COUNTER_ADD|FAULT_POINT|LOG_[A-Z_]+|TIMER_[A-Z_]+)\s*\(", re.ASCII)
STATUS_RETURNED = re.compile(r"(?P<must_check>\bMUST_CHECK\s+)?"
                             r"(?P<specifiers>(?:\b(?:static|inline|extern)\s+)*)"
                             r"\bresult_status\b\s*(?:(?P<pointer>\(\s*\*[\w\s]*\))|\w+)\s*\(", re.ASCII)
BODY_FOLLOWS = re.compile(r"\s*\{")

# Searched in the raw text, comments included: each is a way to pass a check without meeting it.
FORBIDDEN_TEXT = (
    (re.compile(r"NOLINT"), "it silences clang-tidy"),
    (re.compile(r"cppcheck-suppress"), "it silences cppcheck"),
    (re.compile(r"GCC[\s\\]+(?:diagnostic|system_header)"), "it silences compiler warnings"),
    (re.compile(r"GCC[\s\\]+optimize|\b_*optimize_*\s*\("), "it compiles one function with other flags"),
    (re.compile(r"no_sanitize|no_address_safety_analysis"), "it hides code from a sanitizer"),
    (re.compile(r"""[<"]c?assert(?:\.h)?[>"]"""), "it is a second assertion: use ASSERT"),
    (re.compile(r"fast-math|use_fast_math|FP_CONTRACT"), "it changes results"),
)


def scan(text):
    """Returns the text with comments, strings and character constants blanked (numbers stay), the numbers as
    (offset, spelling), and the offsets of comments and quotes that never close."""
    pieces, numbers, unclosed_offsets, position = [], [], [], 0
    for match in TOKEN.finditer(text):
        token = match.group(0)
        is_number = match.lastgroup == "number" or (match.lastgroup == "character" and token != TERMINATOR_CHARACTER
                                                    and NUMERIC_CHARACTER.fullmatch(token) is not None)
        if is_number:
            numbers.append((match.start(), token))
        elif match.lastgroup == "unclosed":
            unclosed_offsets.append(match.start())
        pieces += [text[position:match.start()], token if is_number else NOT_BLANKED.sub(" ", token)]
        position = match.end()
    return "".join(pieces + [text[position:]]), numbers, unclosed_offsets


def closing_offset(code, opening_offset, opening, closing):
    """The offset just past the bracket that closes the one at opening_offset, or None if nothing closes it."""
    depth = 0
    for offset in range(opening_offset, len(code)):
        if code[offset] == opening:
            depth += 1
        elif code[offset] == closing:
            depth -= 1
            if depth == 0:
                return offset + 1
    return None


def named_spans(code, report):
    """The spans of the code in which a number is being given a name."""
    spans = [match.span("text") for match in CONSTANT_DEFINE.finditer(code)
             if NOT_A_CONSTANT.search(match.group("text")) is None]
    opening_offsets = sorted(match.end() - 1 for start in (ENUMERATION_START, CONSTANT_TABLE_START)
                             for match in start.finditer(code))
    end_of_last_table = 0
    for opening_offset in opening_offsets:
        if opening_offset < end_of_last_table:
            report(opening_offset, "a constant table or enumeration opens inside another one")
            continue
        end_of_last_table = closing_offset(code, opening_offset, "{", "}")
        if end_of_last_table is None:
            report(opening_offset, "a constant table or enumeration never closes: the rest of the file is unchecked")
            end_of_last_table = len(code)
        cut = PREPROCESSOR_LINE.search(code, opening_offset, end_of_last_table)
        if cut is not None:
            report(cut.start(), "a preprocessor line inside a constant table or enumeration: it cannot be followed")
        spans.append((opening_offset, end_of_last_table))
    return spans


def lint_text(text):
    """Every problem of one file's text, as sorted (line number, message)."""
    problems = []

    def report(offset, message):
        problems.append((text.count("\n", 0, offset) + 1, message))

    for forbidden, reason in FORBIDDEN_TEXT:
        for match in forbidden.finditer(text):
            report(match.start(), f'forbidden text "{" ".join(match.group(0).split())}": {reason}')
    code, numbers, unclosed_offsets = scan(text)
    for offset in unclosed_offsets:
        report(offset, "a comment, string or character constant that does not close")
    for match in OUTSIDE_ASCII.finditer(code):
        report(match.start(), "a character outside ASCII in code")

    spans = named_spans(code, report)
    template_spans = [match.span("arguments") for start, end in spans
                      for match in TEMPLATE_ARGUMENTS.finditer(code, start, end)]
    for offset, spelling in numbers:
        if ALLOWED_NUMBER.fullmatch(spelling) is not None:
            continue
        if any(start <= offset < end for start, end in template_spans):
            report(offset, f"bare number {spelling} as a template argument")
        elif not any(start <= offset < end for start, end in spans):
            report(offset, f"bare number {spelling}")

    for match in HOOK_USE.finditer(code):
        arguments_end = closing_offset(code, match.end() - 1, "(", ")")
        if arguments_end is None:
            report(match.start(), "the argument list of a hook never closes")
            continue
        side_effect = SIDE_EFFECT.search(code, match.end(), arguments_end)
        if side_effect is not None:
            report(side_effect.start(), f"{side_effect.group(0)} in the argument of a hook: "
                                        "a build without the hook does not evaluate it")
    for match in STATUS_RETURNED.finditer(code):
        if match.group("must_check") is not None:
            continue
        parameters_end = closing_offset(code, match.end() - 1, "(", ")")
        # The definition of a function with external linkage inherits the attribute from its prototype, which the
        # compiler demands (-Wmissing-prototypes) and this loop checks. Everything else carries it itself.
        inherits = (match.group("pointer") is None and "static" not in match.group("specifiers")
                    and parameters_end is not None and BODY_FOLLOWS.match(code, parameters_end) is not None)
        if not inherits:
            report(match.end(), "result_status returned without MUST_CHECK")
    return sorted(problems)


def files_under(top, problems):
    """Every file under a directory, in a fixed order; a file is itself."""
    if os.path.isfile(top):
        return [top]
    if not os.path.isdir(top):
        problems.append(f"{top}: neither a file nor a directory")
        return []
    found = []
    for directory, directory_names, names in os.walk(top):
        directory_names[:] = sorted(name for name in directory_names if name != CASES_DIRECTORY_NAME)
        found += [os.path.join(directory, name) for name in sorted(names)]
    return found


def main():
    tops = sys.argv[1:]
    if not tops:
        print(__doc__, file = sys.stderr)
        return EXIT_STATUS_USAGE
    problems, file_count = [], 0
    for top in tops:
        for path in files_under(top, problems):
            if not path.endswith(ENGINE_SUFFIXES):
                problems.append(f"{path}: not a {' '.join(ENGINE_SUFFIXES)} file: this tool cannot read it")
                continue
            file_count += 1
            try:
                with open(path, encoding = "utf8") as handle:
                    text = handle.read()
            except UnicodeDecodeError:
                problems.append(f"{path}: not UTF-8 text")
                continue
            problems += [f"{path}:{line_number}: {message}" for line_number, message in lint_text(text)]
    if not file_count:
        problems.append("no file to lint")
    for problem in problems:
        print(problem, file = sys.stderr)
    print(f"literal lint: {file_count} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
