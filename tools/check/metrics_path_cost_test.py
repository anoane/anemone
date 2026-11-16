#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guard the metrics control-flow/cost boundaries that timing tests cannot reliably detect.

Runtime engine tests separately reconcile each landed step with one observation.
These checks protect early continues and multi-step drains against regression.
"""
import argparse
from pathlib import Path


def block_after(text, marker):
    begin = text.index('{', text.index(marker))
    depth = 1
    for end in range(begin + 1, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[begin + 1:end]
    raise ValueError(f'unclosed block: {marker}')


def check(root):
    failures = []
    engine = (root / 'source/engine.c').read_text()
    worker = block_after(engine, 'static void *stepping_thread(void *argument)\n{')
    for marker in ('if (restoring < slot_count)', 'if (pending < slot_count)',
                   'while (engine->flight_count != 0)'):
        branch = block_after(worker, marker)
        if 'engine_live_iteration(engine);' not in branch:
            failures.append(f'landed-step observation missing before leaving {marker}')
        if 'if (engine->metrics_enabled)' not in branch:
            failures.append(f'metrics-off publication changed in {marker}')
        if marker.startswith('while') and 'if (observe_metrics)' not in branch:
            failures.append('metrics-off image drain acquires extra publication locks')
    server = (root / 'source/server.c').read_text()
    tokens = block_after(server, 'if (!is_final && server->metrics.enabled)')
    if 'event_seconds' not in tokens or 'monotonic_seconds()' in tokens:
        failures.append('token observation must reuse the single event clock read')
    metrics = (root / 'source/server_metrics.c').read_text()
    observed = block_after(metrics, 'static void observed(')
    if 'bounds_for(' in observed or 'memcpy(' in observed or 'token_bounds(' in observed:
        failures.append('observation rebuilds or copies histogram bounds')
    if 'void server_metrics_prepare(' not in metrics or 'token_context!=server->description.context_token_limit' not in ''.join(block_after(metrics, 'void server_metrics_prepare(').split()):
        failures.append('token bounds must be cached by immutable context identity')
    if failures:
        raise ValueError(' ; '.join(failures))
    print('metrics paths/cost: restore, session, each image-drain step; one token clock; cached bounds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    check(parser.parse_args().source_root)
