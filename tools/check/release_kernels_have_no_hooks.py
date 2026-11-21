#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reject optional device logging/assertion/sanitizer instrumentation in release modules.

    release_kernels_have_no_hooks.py [--required MODULE=FUNCTION,...]... MODULE.cubin...

Inspect SASS and ELF symbols, without creating a CUDA context. Algorithmic checks,
atomics and bounded-progress clocks are not optional instrumentation and stay valid.
A --required line names the functions of a module (its cubin's file name) that keep
required launch/resource guards and communication deadlines (its module.mk's
REQUIRED_TRAPS_<module>): a trap is allowed in them only. Optional instrumentation
is forbidden in them as in every other release function.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

CUOBJDUMP = '/usr/local/cuda/bin/cuobjdump'
FORBIDDEN = re.compile(r'vprintf|__assertfail|__assert_fail|__asan_|__ubsan_|__tsan_|__sanitizer_', re.I)
TRAP = re.compile(r'\b(?:BPT\.TRAP|TRAP)\b')


def unexpected_traps(path, sass, required):
    allowed = required.get(path.stem, set())
    problems = []
    current_function = ''
    for line in sass.splitlines():
        if 'Function :' in line:
            current_function = line.split('Function :', 1)[1].strip()
        if TRAP.search(line) and current_function not in allowed:
            problems.append(current_function or '<unnamed>')
    return sorted(set(problems))


def inspect(path, required):
    results = [subprocess.run([CUOBJDUMP, option, str(path)], capture_output=True, text=True, check=True).stdout
               for option in ('--dump-sass', '--dump-elf-symbols')]
    sass, symbols = results
    if 'Function :' not in sass:
        raise ValueError('module has no disassembled function')
    problems = []
    traps = unexpected_traps(path, sass, required)
    if traps: problems.append('holds a trap outside required runtime checks: ' + ', '.join(traps))
    matches = sorted(set(FORBIDDEN.findall(sass + symbols)))
    if matches: problems.append('holds optional device instrumentation: ' + ', '.join(matches))
    return dict(path=str(path), functions=sass.count('Function :'), problems=problems)


def main():
    arguments, required = sys.argv[1:], {}
    while len(arguments) >= 2 and arguments[0] == '--required' and '=' in arguments[1]:
        module, functions = arguments[1].split('=', 1)
        required[module] = set(filter(None, functions.split(',')))
        arguments = arguments[2:]
    if not arguments or any(argument.startswith('--') for argument in arguments):
        print(__doc__, file=sys.stderr)
        return 2
    records = []
    for argument in arguments:
        try:
            row = inspect(Path(argument), required)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            row = dict(path=argument, problems=['cannot inspect: ' + str(error)])
        records.append(row)
        for problem in row['problems']: print(f'{argument} {problem}', file=sys.stderr)
    passed = all(not row['problems'] for row in records)
    print(json.dumps(dict(passed=passed, modules=records)))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
