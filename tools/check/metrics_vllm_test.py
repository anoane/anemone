#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare real exposition to pinned upstream metadata and evaluate every dashboard query.

The PromQL fixture uses a zero baseline followed by the supplied measured counters.
It tests query compatibility, not model quality, native timing or throughput.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from metrics_exposition_test import LABEL, SAMPLE, validate

ROOT = Path(__file__).resolve().parents[2]


def parse(text):
    validate(text)
    families, samples = {}, []
    for line in text.splitlines():
        if line.startswith('# HELP '):
            name, help_text = line[len('# HELP '):].split(' ', 1)
            families.setdefault(name, {})['help'] = help_text
        elif line.startswith('# TYPE '):
            name, kind = line[len('# TYPE '):].split(' ')
            families.setdefault(name, {})['type'] = kind
        else:
            name, raw, value = SAMPLE.fullmatch(line).groups()
            labels = {m[1]: m[2] for m in LABEL.finditer(raw or '')}
            family = name if name in families else next(
                (name[:-len(s)] for s in ('_bucket', '_sum', '_count')
                 if name.endswith(s) and name[:-len(s)] in families), None)
            samples.append(dict(name=name, labels=labels, raw=raw or '', value=float(value), family=family))
    return families, samples


def token_buckets(context):
    result, scale = [], 1
    while scale <= context:
        for mantissa in (1, 2, 5):
            if scale * mantissa > context:
                return result
            result.append(float(scale * mantissa))
        scale *= 10
    return result


def bound_spelling(value):
    if math.isinf(value):
        return '+Inf'
    # prometheus_client.utils.floatToGoString: exponent notation starts at 1e6.
    return format(value, '.0e') if value >= 1e6 else repr(float(value))


def compare(text, context, speculation, *, model='anemone', slots=1, depth=1,
            prefix_reuse=None, require_process=False, messages=False):
    contract = json.loads((ROOT / 'tools/metrics/vllm_contract.json').read_text())
    actual, samples = parse(text)
    expected = {name: data for name, data in contract['families'].items()
                if speculation or not data.get('conditional')}
    lazy = {'http_requests_created', 'http_request_size_bytes_created',
            'http_response_size_bytes_created', 'http_request_duration_seconds_created'}
    http_children = any(sample['name'] == 'http_requests_total' for sample in samples)
    if not http_children:
        expected = {name: data for name, data in expected.items() if name not in lazy}
    if not require_process:
        expected = {name: data for name, data in expected.items()
                    if not name.startswith('process_') or name in actual}
    observed = {name for name in actual if not name.startswith('anemone_')}
    if observed != set(expected):
        raise ValueError(f"family set mismatch: missing={set(expected)-observed}, extra={observed-set(expected)}")
    escaped_model = model.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    fixed_values = {
        'engine': {'0'}, 'model_name': {escaped_model},
        'finished_reason': {'stop', 'length', 'abort', 'error', 'repetition'},
        'source': {'local_compute', 'local_cache_hit', 'external_kv_transfer'},
        'position': {str(index) for index in range(depth)},
        'status': {'2xx', '3xx', '4xx', '5xx'},
        'handler': {'/v1/models', '/v1/chat/completions', '/v1/anemone/stats', '/v1/responses', 'none'} | ({'/v1/messages'} if messages else set()),
        'method': {'GET', 'POST', 'HEAD', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'OTHER'},
        'enable_prefix_caching': {'True', 'False'} if prefix_reuse is None else {str(prefix_reuse)},
        'kv_cache_size_tokens': {str(context * slots)},
        'kv_cache_max_concurrency': {str(float(slots))},
    }
    for name, data in expected.items():
        if actual[name]['type'] != data['type'] or actual[name]['help'] != data['help']:
            raise ValueError(f'TYPE/HELP mismatch: {name}')
        selected = [sample for sample in samples if sample['family'] == name]
        if not selected and (http_children or not name.startswith('http_')):
            raise ValueError(f'no samples for {name}')
        for sample in selected:
            wanted = set(data['labels']) | ({'le'} if sample['name'].endswith('_bucket') else set())
            if set(sample['labels']) != wanted:
                raise ValueError(f"label keys mismatch: {sample['name']}: {sample['labels']}")
            for key, value in sample['labels'].items():
                if key == 'le':
                    continue
                allowed = fixed_values.get(key)
                if key == 'reason':
                    allowed = ({'capacity', 'deferred'} if 'waiting_by_reason' in name else
                               {'max_num_queued_reqs', 'max_num_queued_tokens'})
                if allowed is None or value not in allowed:
                    raise ValueError(f'label value mismatch: {name}: {key}={value!r}, expected {allowed}')
        if data['type'] == 'histogram':
            bounds = data['buckets']
            if isinstance(bounds, str):
                bounds = token_buckets(context)
            bounds = list(map(float, bounds)) + [math.inf]
            groups = {}
            for sample in selected:
                if 'le' in sample['labels']:
                    key = tuple(sorted((k, v) for k, v in sample['labels'].items() if k != 'le'))
                    groups.setdefault(key, []).append(sample['labels']['le'])
            if (not groups and (http_children or not name.startswith('http_'))) or any(
                    values != [bound_spelling(value) for value in bounds] for values in groups.values()):
                raise ValueError(f'buckets mismatch: {name}')
    exported = {sample['name'] for sample in samples}
    # Upstream mixes base families and sample suffixes; require every asserted item.
    for name in contract['expected_metrics_v1']:
        if name not in exported and name not in actual and name + '_total' not in exported:
            raise ValueError(f'upstream expected-metrics list missing: {name}')
    reasons = {s['labels']['finished_reason'] for s in samples if s['name'] == 'vllm:request_success_total'}
    if reasons != {'stop', 'length', 'abort', 'error', 'repetition'}:
        raise ValueError('FinishReason strings differ from pinned vLLM')
    if speculation:
        positions = {sample['labels']['position'] for sample in samples
                     if sample['name'] == 'vllm:spec_decode_num_accepted_tokens_per_pos_total'}
        if positions != fixed_values['position']:
            raise ValueError('speculative position children differ from configured depth')
    return samples, len(expected)


def dashboard(samples, promtool):
    source = json.loads((ROOT / 'tools/metrics/vllm_dashboard_queries.json').read_text())
    models = {sample['labels']['model_name'] for sample in samples if 'model_name' in sample['labels']}
    if len(models) != 1:
        raise ValueError('one served model identity required')
    model = next(iter(models))
    tests = []
    for expression in source['expressions']:
        expression = expression.replace('$model_name', model).replace('$__rate_interval', '5m')
        if '$' in expression:
            raise ValueError('unresolved Grafana variable')
        tests.append({'expr': f'count(({expression}) >= 0) > bool 0', 'eval_time': '3m',
                      'exp_samples': [{'labels': '{}', 'value': 1}]})
    series = []
    for sample in samples:
        # Values are the real terminal scrape, not invented engine measurements.
        value = sample['value']
        is_counter = (sample['name'].endswith(('_total', '_bucket', '_count', '_sum'))
                      and not sample['name'].startswith('process_'))
        values = f'0 0 {value!r} {value!r}' if is_counter else ' '.join([repr(value)] * 4)
        series.append({'series': sample['name'] + '{' + sample['raw'] + '}', 'values': values})
    config = {'rule_files': [], 'evaluation_interval': '1m',
              'tests': [{'interval': '1m', 'input_series': series, 'promql_expr_test': tests}]}
    with tempfile.TemporaryDirectory(prefix='metrics-promql-') as temporary:
        path = Path(temporary) / 'dashboard.json'
        path.write_text(json.dumps(config))  # JSON is a YAML subset accepted by promtool.
        subprocess.run([promtool, 'test', 'rules', str(path)], check=True, timeout=60)
    return len(tests)


def negative_controls(text, context, speculation, **identity):
    controls = [text.replace('Number of generation tokens processed.', 'Wrong help.'),
                text.replace('le="0.005"', 'le="0.004"'),
                text.replace('le="1.0"', 'le="1"')]
    _, samples = parse(text)
    values = {}
    for sample in samples:
        if sample['name'].startswith('anemone_'):
            continue
        for key, value in sample['labels'].items():
            if key != 'le':
                values.setdefault(key, value)
    for key, value in values.items():
        controls.append(text.replace(f'{key}="{value}"', f'{key}="invalid"'))
    concurrency = str(float(identity.get('slots', 1)))
    controls.append(text.replace(f'kv_cache_max_concurrency="{concurrency}"',
                                 f'kv_cache_max_concurrency="{int(float(concurrency))}"'))
    for index, broken in enumerate(controls):
        if broken == text:
            raise ValueError(f'negative control {index} did not mutate the fixture')
        try:
            compare(broken, context, speculation, **identity)
        except ValueError:
            continue
        raise ValueError(f'upstream parity negative control {index} passed')
    return len(controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exposition', type=Path)
    parser.add_argument('--context', required=True, type=int)
    parser.add_argument('--speculation', action='store_true')
    parser.add_argument('--model', default='anemone')
    parser.add_argument('--slots', type=int, default=1)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--require-process', action='store_true')
    parser.add_argument('--messages', action='store_true', help='Allow the opt-in Messages HTTP handler label')
    parser.add_argument('--prefix-reuse', choices=('True', 'False'))
    parser.add_argument('--promtool', required=True)
    args = parser.parse_args()
    identity = dict(model=args.model, slots=args.slots, depth=args.depth,
                    prefix_reuse=None if args.prefix_reuse is None else args.prefix_reuse == 'True',
                    require_process=args.require_process, messages=args.messages)
    text = args.exposition.read_text()
    samples, count = compare(text, args.context, args.speculation, **identity)
    rejected = negative_controls(text, args.context, args.speculation, **identity)
    expressions = dashboard(samples, args.promtool)
    print(f'vLLM parity: {count} families; upstream expected set; TYPE/HELP/label keys and values/raw buckets; '
          f'{expressions} dashboard expressions return finite data; {rejected} negative controls rejected')


if __name__ == '__main__':
    main()
