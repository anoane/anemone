# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression for direct recursion, banned copies and the retired key matcher.

Run from the repository root. This lexical check does not prove callback graphs.
"""
from pathlib import Path
import re
import sys

LITERALS = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
problems = []
for directory in ('source', 'include/anemone', 'tests'):
    for path in sorted(Path(directory).iterdir()):
        if path.suffix not in ('.c', '.h'):
            continue
        source = path.read_text()
        clean = LITERALS.sub(lambda match: ''.join('\n' if c == '\n' else ' ' for c in match[0]), source)
        clean = re.sub(r'^\s*#(?:[^\n]*\\\n)*[^\n]*',
                       lambda match: ''.join('\n' if c == '\n' else ' ' for c in match[0]), clean, flags=re.M)
        for match in re.finditer(r'\b(strcpy|strcat|server_key_matches)\s*\(', clean):
            problems.append(f'{path}:{clean.count(chr(10), 0, match.start()) + 1}: {match[1]} remains')
        depth, begin, previous = 0, None, 0
        for position, byte in enumerate(clean):
            if byte == '{':
                if depth == 0:
                    match = re.search(r'\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$', clean[previous:position])
                    begin = (match[1], position) if match else None
                depth += 1
            elif byte == '}':
                depth -= 1
                if depth == 0:
                    if begin:
                        name, start = begin
                        for call in re.finditer(r'\b' + re.escape(name) + r'\s*\(', clean[start:position]):
                            at = start + call.start()
                            if not re.search(r'(?:\.|->)\s*$', clean[start:at]):
                                problems.append(f'{path}:{clean.count(chr(10), 0, at) + 1}: recursive {name}')
                    begin, previous = None, position + 1
            elif byte == ';' and depth == 0:
                previous = position + 1
for problem in problems:
    print(problem)
sys.exit(bool(problems))
