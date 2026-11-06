#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reject optional device logging/assertion/sanitizer instrumentation in release modules.

Inspect SASS and ELF symbols, without creating a CUDA context. Algorithmic checks,
atomics and bounded-progress clocks are not optional instrumentation and stay valid.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

CUOBJDUMP = '/usr/local/cuda/bin/cuobjdump'
FORBIDDEN = re.compile(r'vprintf|__assertfail|__assert_fail|__asan_|__ubsan_|__tsan_|__sanitizer_', re.I)
TRAP = re.compile(r'\b(?:BPT\.TRAP|TRAP)\b')
# These functions retain required launch/resource guards and communication deadlines.
# Optional instrumentation is forbidden in them as in every other release function.
REQUIRED_CHECK_FUNCTIONS = {
    'kernel_bf16_compressed_ring': {'bf16_compressed_ring'},
    'kernel_bf16_round_once_ring': {'bf16_round_once_ring'},
    'kernel_draft_share': {'draft_share_wait'},
    'kernel_exl3_tcp': {'exl3_tcp_route'},
    'kernel_sampler_shard': {'sampler_shard'},
    'kernel_ordered_collective': {
        'ordered_all_reduce', 'ordered_two_shot',
        'ordered_reduce_scatter', 'ordered_gather',
    },
    'kernel_ordered_push': {'ordered_push'},
    'kernel_tp_exchange': {
        'tp_exchange', 'tp_exchange_ce_owner_ring_sum', 'tp_exchange_ce_owner_sum',
        'tp_exchange_done', 'tp_exchange_owner_publish', 'tp_exchange_owner_ring_publish',
        'tp_exchange_owner_ring_sum', 'tp_exchange_owner_sum', 'tp_exchange_publish',
        'tp_exchange_ready', 'tp_exchange_sum',
    },
    'kernel_exact_ring': {'exact_ring', 'exact_ring_unequal'},
    'kernel_ordered_push_parallel': {'ordered_push_parallel_publish',
                                     'ordered_push_parallel_reduce'},
    'kernel_ordered_push_two': {'ordered_push_two'},
    'kernel_expert_exchange': {
        'expert_peer_rows_send', 'expert_peer_jobs_take',
        'expert_peer_pairs_copy',
    },
    'kernel_hyper_connection': {'peer_hyper_post', 'peer_expert_join'},
    'kernel_peer_site_front': {
        'peer_site_front_1', 'peer_site_front_8',
        'peer_host_site_front_1', 'peer_host_site_front_8',
    },
    'kernel_trunk_linear': {
        'peer_expert_shared_join_' + str(slices) for slices in (1, 2, 4, 8, 16)
    },
    'kernel_trunk_linear_native': {
        'peer_expert_shared_join_' + str(slices) for slices in (1, 2, 4, 8, 16)
    },
}


def unexpected_traps(path, sass):
    allowed = REQUIRED_CHECK_FUNCTIONS.get(path.stem, set())
    problems = []
    current_function = ''
    for line in sass.splitlines():
        if 'Function :' in line:
            current_function = line.split('Function :', 1)[1].strip()
        if TRAP.search(line) and current_function not in allowed:
            problems.append(current_function or '<unnamed>')
    return sorted(set(problems))


def inspect(path):
    results = [subprocess.run([CUOBJDUMP, option, str(path)], capture_output=True, text=True, check=True).stdout
               for option in ('--dump-sass', '--dump-elf-symbols')]
    sass, symbols = results
    if 'Function :' not in sass:
        raise ValueError('module has no disassembled function')
    problems = []
    traps = unexpected_traps(path, sass)
    if traps: problems.append('holds a trap outside required runtime checks: ' + ', '.join(traps))
    matches = sorted(set(FORBIDDEN.findall(sass + symbols)))
    if matches: problems.append('holds optional device instrumentation: ' + ', '.join(matches))
    return dict(path=str(path), functions=sass.count('Function :'), problems=problems)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    records = []
    for argument in sys.argv[1:]:
        try:
            row = inspect(Path(argument))
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            row = dict(path=argument, problems=['cannot inspect: ' + str(error)])
        records.append(row)
        for problem in row['problems']: print(f'{argument} {problem}', file=sys.stderr)
    passed = all(not row['problems'] for row in records)
    print(json.dumps(dict(passed=passed, modules=records)))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
