# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the isolated sanitizer fixtures, including both allocation failures and hostile snapshot paths."""
from pathlib import Path
import subprocess
import tempfile

PROGRAMS = Path('build/debug/check_cases/quality')
CASES = ('snapshot_size', 'tp_load_null', 'tp_load_callback', 'exl3_allocation',
         'engine_descriptor', 'router_failure', 'tune_weighted_copy', 'replica_receipt_binding',
         'replica_lag_choice')
PATH_CASES = ('fifo_read', 'fifo_load', 'symlink_read', 'symlink_load',
              'nonregular_read', 'nonregular_match', 'partial_regular', 'partial_symlink',
              'directory_safe', 'directory_group', 'directory_other', 'directory_owner',
              'directory_symlink', 'directory_version_group', 'directory_version_other',
              'directory_version_owner', 'directory_version_symlink')
TIMEOUT_SECONDS = 30


def run(name, *arguments):
    print('QUALITY_REGRESSION', name, *arguments, flush=True)
    subprocess.run([str(PROGRAMS / name), *arguments], check=True, timeout=TIMEOUT_SECONDS)


for case in CASES:
    run(case)
run('exl3_allocation', '2')
for case in PATH_CASES:
    with tempfile.TemporaryDirectory(prefix='anemone_snapshot_paths_') as directory:
        run('snapshot_paths', case, directory)
