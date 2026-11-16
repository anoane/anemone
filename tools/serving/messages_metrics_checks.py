# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared CPU/native assertions for terminal Messages accounting in the existing metrics collectors.

Every check compares all of its families at once; a mismatch raises one AssertionError that lists every observed
number with its expectation, so a single failure shows the whole terminal.
"""
import json
import operator

COMPARISONS = {'==': operator.eq, '>': operator.gt, '>=': operator.ge}
REQUEST_HISTOGRAMS = ('time_to_first_token_seconds', 'e2e_request_latency_seconds', 'request_queue_time_seconds',
                      'request_prefill_time_seconds', 'request_decode_time_seconds', 'request_inference_time_seconds',
                      'request_time_per_output_token_seconds', 'request_prompt_tokens', 'request_generation_tokens',
                      'request_max_num_generation_tokens', 'request_params_max_tokens', 'request_params_n',
                      'request_num_preemptions', 'request_prefill_kv_computed_tokens')
ERROR_HISTOGRAMS = ('e2e_request_latency_seconds', 'request_queue_time_seconds', 'request_prefill_time_seconds',
                    'request_decode_time_seconds', 'request_inference_time_seconds', 'request_prompt_tokens',
                    'request_generation_tokens', 'request_num_preemptions', 'request_params_n')


def value(samples, name, **labels):
    return sum(sample['value'] for sample in samples if sample['name'] == name
               and all(sample['labels'].get(key) == val for key, val in labels.items()))


def delta(before, after, name, **labels):
    return value(after, name, **labels) - value(before, name, **labels)


def verify(before, after, expectations):
    """expectations: (kind, name, labels, comparison, expected), kind 'delta' or 'value' (of after)."""
    rows = []
    for kind, name, labels, comparison, expected in expectations:
        observed = delta(before, after, name, **labels) if kind == 'delta' else value(after, name, **labels)
        rows.append(dict(kind=kind, name=name, labels=labels, observed=observed, comparison=comparison,
                         expected=expected, ok=COMPARISONS[comparison](observed, expected)))
    failed = [row for row in rows if not row['ok']]
    if failed:
        raise AssertionError('metrics differ in ' + ', '.join(row['name'] for row in failed) + '; observed: ' +
                             json.dumps(rows))
    return rows


def http_expectations(status, handler='/v1/messages', response_bytes=None):
    rows = [('delta', 'http_requests_total', dict(handler=handler, method='POST', status=status), '==', 1)]
    rows += [('delta', name, dict(handler=handler), '==', 1) for name in
             ('http_request_duration_seconds_count', 'http_request_size_bytes_count', 'http_response_size_bytes_count')]
    rows.append(('delta', 'http_request_size_bytes_sum', dict(handler=handler), '>', 0))
    # Upstream counts Content-Length; chunked responses have a measured size of zero.
    rows.append(('delta', 'http_response_size_bytes_sum', dict(handler=handler),
                 *(('>=', 0) if response_bytes is None else ('==', response_bytes))))
    rows.append(('delta', 'http_request_duration_highr_seconds_count', {}, '==', 1))
    return rows


def check_http(before, after, status, handler='/v1/messages', response_bytes=None):
    return verify(before, after, http_expectations(status, handler, response_bytes))


def check_success(before, after, usage, finish='stop', handler='/v1/messages', response_bytes=None):
    prompt = usage['input_tokens'] + usage.get('cache_read_input_tokens', 0)
    output = usage['output_tokens']
    rows = http_expectations('2xx', handler, response_bytes) + [
        ('delta', 'anemone_requests_admitted_total', {}, '==', 1),
        ('delta', 'vllm:request_success_total', dict(finished_reason=finish), '==', 1),
        ('delta', 'vllm:prompt_tokens_total', {}, '==', prompt),
        ('delta', 'vllm:generation_tokens_total', {}, '==', output)]
    rows += [('delta', 'vllm:' + name + '_count', {}, '==', 1) for name in REQUEST_HISTOGRAMS]
    rows += [('delta', 'vllm:inter_token_latency_seconds_count', {}, '==', max(0, output - 1)),
             ('delta', 'vllm:request_prompt_tokens_sum', {}, '==', prompt),
             ('delta', 'vllm:request_generation_tokens_sum', {}, '==', output),
             ('delta', 'vllm:prompt_tokens_by_source_total', dict(source='local_compute'), '==',
              usage['input_tokens']),
             ('delta', 'vllm:prompt_tokens_by_source_total', dict(source='local_cache_hit'), '==',
              usage.get('cache_read_input_tokens', 0)),
             ('value', 'vllm:num_requests_running', {}, '==', 0),
             ('value', 'vllm:num_requests_waiting', {}, '==', 0)]
    return verify(before, after, rows)


def check_refusal(before, after):
    rows = http_expectations('4xx') + [('delta', 'anemone_request_refusals_total', dict(status_class='4xx'), '==', 1)]
    rows += [('delta', name, {}, '==', 0) for name in
             ('vllm:request_success_total', 'vllm:prompt_tokens_total', 'vllm:generation_tokens_total',
              'vllm:e2e_request_latency_seconds_count', 'vllm:request_generation_tokens_count')]
    return verify(before, after, rows)


def error_expectations(status):
    """A server-side error terminal: counted as an error, never as a client cancellation, with every terminal
    histogram observed."""
    rows = http_expectations(status) + [
        ('delta', 'vllm:request_success_total', dict(finished_reason='error'), '==', 1),
        ('delta', 'anemone_requests_finished_total', dict(finished_reason='error'), '==', 1),
        ('delta', 'anemone_requests_finished_total', dict(finished_reason='cancelled'), '==', 0)]
    return rows + [('delta', 'vllm:' + name + '_count', {}, '==', 1) for name in ERROR_HISTOGRAMS]


def check_error_terminal(before, after, status='5xx'):
    return verify(before, after, error_expectations(status))


def check_engine_error(before, after):
    return verify(before, after, error_expectations('5xx') + [
        ('delta', 'vllm:generation_tokens_total', {}, '==', 0),
        ('delta', 'vllm:time_to_first_token_seconds_count', {}, '==', 0)])
