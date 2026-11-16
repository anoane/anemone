#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real HTTP contract checks over the native CPU engine fixture; no CUDA/model runs."""
import concurrent.futures
import contextlib
import copy
import http.client
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from messages_sdk_smoke import LEGACY_VERSION, SCENARIOS, run_sdk_scenario

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'check'))
from metrics_vllm_test import parse as parse_metrics, compare as compare_metrics
from messages_metrics_checks import value as metric_value, check_success, check_refusal, check_engine_error, check_error_terminal

BINARY = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path('build/release/server_messages_test').resolve()
sys.argv = sys.argv[:1]
KEY = 'alice_' + 'a' * 64
SECOND_KEY = 'bob_' + 'b' * 64
BASE = {'model': 'anemone', 'max_tokens': 128, 'messages': [{'role': 'user', 'content': 'Hello'}]}
TOOL = {'name': 'lookup', 'description': 'Find a city', 'input_schema': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}
CHECKS = 0

@contextlib.contextmanager
def fixture(mode='text'):
    with tempfile.TemporaryDirectory(prefix='anemone_messages_') as directory:
        key = Path(directory) / 'disposable_keys'
        key.write_text(KEY + '\n' + SECOND_KEY + '\n')
        key.chmod(0o600)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        with (Path(directory) / 'stderr').open('w+') as errors:
            process = subprocess.Popen([str(BINARY), '--client-fixture', str(port), mode, str(key)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        errors.seek(0)
                        raise RuntimeError('fixture failed: ' + errors.read())
                    try:
                        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                        connection.request('GET', '/health')
                        response = connection.getresponse()
                        ready = response.status == 200
                        response.read()
                        connection.close()
                        if ready:
                            break
                    except OSError:
                        time.sleep(0.02)
                else:
                    raise TimeoutError('fixture readiness')
                yield port
            finally:
                process.stdin.close()
                try:
                    status = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    raise
                process.stdout.close()
                if status != 0:
                    errors.seek(0)
                    raise RuntimeError('fixture exit ' + str(status) + ': ' + errors.read())


def request(port, body=None, *, headers=None, path='/v1/messages', method='POST'):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
    fields = {'x-api-key': KEY, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'}
    if headers is not None:
        fields = headers
    connection.request(method, path, None if method == 'GET' else json.dumps(BASE if body is None else body, ensure_ascii=False).encode(), fields)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def events(wire):
    output = []
    for frame in wire.decode().split('\n\n'):
        if not frame or frame.startswith(':'):
            continue
        lines = frame.splitlines()
        assert len(lines) == 2 and lines[0].startswith('event: ') and lines[1].startswith('data: '), frame
        value = json.loads(lines[1][6:])
        assert value['type'] == lines[0][7:]
        output.append(value)
    return output


def raw_exchange(port, head):
    with socket.create_connection(('127.0.0.1', port), timeout=15) as connection:
        connection.sendall(head)
        output = bytearray()
        while piece := connection.recv(65536):
            output.extend(piece)
        return bytes(output)


def reconstruct(wire):
    sequence = events(wire)
    assert sequence[0]['type'] == 'message_start'
    message = sequence[0]['message']
    assert message['content'] == [] and message['usage']['output_tokens'] == 0
    opened = None
    argument = ''
    for event in sequence[1:]:
        kind = event['type']
        if kind == 'content_block_start':
            assert opened is None and event['index'] == len(message['content'])
            opened = event['content_block']
            message['content'].append(opened)
            argument = ''
        elif kind == 'content_block_delta':
            assert opened is not None and event['index'] == len(message['content']) - 1
            delta = event['delta']
            if delta['type'] == 'text_delta':
                opened['text'] += delta['text']
            elif delta['type'] == 'thinking_delta':
                opened['thinking'] += delta['thinking']
            elif delta['type'] == 'signature_delta':
                opened['signature'] += delta['signature']
            else:
                assert opened['type'] == 'tool_use' and delta['type'] == 'input_json_delta'
                argument += delta['partial_json']
        elif kind == 'content_block_stop':
            assert opened is not None and event['index'] == len(message['content']) - 1
            if opened['type'] == 'tool_use':
                opened['input'] = json.loads(argument)
            opened = None
        elif kind == 'message_delta':
            assert opened is None
            message.update(event['delta'])
            message['usage'].update(event['usage'])
        else:
            assert kind == 'message_stop'
    assert opened is None and sequence[-1]['type'] == 'message_stop'
    return message


class MessagesContract(unittest.TestCase):
    def check_message(self, message, stop='end_turn'):
        self.assertTrue(message['id'].startswith('msg-'))
        self.assertEqual((message['type'], message['role'], message['model']), ('message', 'assistant', 'anemone'))
        self.assertEqual(message['stop_reason'], stop)
        self.assertGreater(message['usage']['input_tokens'], 0)
        self.assertGreater(message['usage']['output_tokens'], 0)
        self.assertNotIn('cache_read_input_tokens', message['usage'])

    def test_text_stream_stops_limits_and_versions(self):
        with fixture() as port:
            for streamed in (False, True):
                for stops, limit, expected, stop in (([], 128, 'Hi', 'end_turn'), (['i'], 128, 'H', 'stop_sequence'), ([], 1, 'H', 'max_tokens')):
                    body = dict(BASE, stream=streamed, max_tokens=limit, stop_sequences=stops)
                    status, headers, wire = request(port, body)
                    self.assertEqual(status, 200)
                    message = reconstruct(wire) if streamed else json.loads(wire)
                    self.check_message(message, stop)
                    self.assertEqual(message['content'], [{'type': 'text', 'text': expected}])
                    self.assertEqual(message['stop_sequence'], 'i' if stops else None)
                    self.assertNotIn(b'[DONE]', wire)
                    self.assertEqual(message['usage']['output_tokens'], 3 if stop == 'end_turn' else 2 if stops else 1)
            status, _, wire = request(port, headers={'x-api-key': KEY, 'anthropic-version': '2026-10-21'})
            self.assertEqual(status, 200)
            self.check_message(json.loads(wire))
            for field in ('temperature', 'top_p', 'top_k'):
                self.assertEqual(request(port, dict(BASE, **{field: 1}), headers={'x-api-key': KEY, 'anthropic-version': '2026-10-21'})[0], 400)
            self.assertEqual(request(port, dict(BASE, temperature=0, top_p=1))[0], 200)
            # OpenAI authentication and response grammar remain intact on the same listener.
            status, _, wire = request(port, {'model': 'test-model', 'messages': BASE['messages'], 'max_tokens': 1, 'reasoning_effort': 0},
                                      path='/v1/chat/completions', headers={'Authorization': 'Bearer ' + KEY})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(wire)['object'], 'chat.completion')

    def test_errors_and_authentication(self):
        with fixture() as port:
            for headers, expected in (({}, 400), ({'anthropic-version': '2023-06-01'}, 401),
                                      ({'anthropic-version': '2023-06-01', 'x-api-key': 'wrong-placeholder'}, 401),
                                      ({'anthropic-version': 'invalid', 'x-api-key': KEY}, 400),
                                      ({'anthropic-version': '2023-06-01', 'x-api-key': KEY, 'Authorization': 'Bearer ' + KEY}, 400),
                                      ({'anthropic-version': '2023-06-01', 'x-api-key': KEY, 'anthropic-beta': 'anything'}, 400)):
                status, _, wire = request(port, headers=headers)
                self.assertEqual(status, expected)
                self.assertEqual(json.loads(wire)['type'], 'error')
                self.assertNotIn(KEY.encode(), wire)
            self.assertEqual(request(port, headers={'anthropic-version': '2023-06-01', 'Authorization': 'Bearer ' + KEY})[0], 200)
            for changes in ({'top_k': -1}, {'thinking': {'type': 'enabled', 'budget_tokens': 1024}}, {'thinking': {'type': 'adaptive', 'budget_tokens': 2}},
                            {'temperature': 1.1}, {'temperature': None}, {'top_p': 0}, {'max_tokens': 0}, {'max_tokens': 1.5},
                            {'stream': None}, {'stop_sequences': None}, {'stop_sequences': ['']}, {'metadata': {}},
                            {'messages': []}, {'messages': [{'role': 'system', 'content': 'x'}]},
                            {'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}]},
                            {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'x', 'cache_control': {}}]}]},
                            {'tools': [dict(TOOL, strict=True)]}, {'tool_choice': {'type': 'any'}}, {'unknown': False}):
                status, _, wire = request(port, dict(BASE, **changes))
                self.assertEqual(status, 400, changes)
                self.assertEqual(json.loads(wire)['error']['type'], 'invalid_request_error')
            self.assertEqual(request(port, dict(BASE, model='unavailable-model'))[0], 404)
            self.assertEqual(request(port, dict(BASE, thinking={'type': 'disabled'}))[0], 200)
            for suffix, status in ((b'Content-Length: 0\r\n', 400), (b'Transfer-Encoding: chunked\r\n', 501),
                                   (b'Content-Length: 33554433\r\n', 413), (b'', 411),
                                   (b'Content-Length: 0\r\nanthropic-version: 2023-06-01\r\n', 400),
                                   (b'Content-Length: 0\r\nx-api-key: duplicate-placeholder\r\n', 400)):
                with socket.create_connection(('127.0.0.1', port), timeout=5) as sock:
                    sock.sendall(b'POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nanthropic-version: 2023-06-01\r\nx-api-key: ' + KEY.encode() + b'\r\n' + suffix + b'\r\n')
                    raw = b''
                    while chunk := sock.recv(65536):
                        raw += chunk
                    self.assertEqual(int(raw.split(b' ')[1]), status)
                    self.assertEqual(json.loads(raw.split(b'\r\n\r\n', 1)[1])['type'], 'error')
            for raw in (b'{', b'{"model":"anemone","model":"anemone"}', b'NaN', json.dumps(dict(BASE, temperature=float('nan'))).encode()):
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                connection.request('POST', '/v1/messages', raw, {'x-api-key': KEY, 'anthropic-version': '2023-06-01'})
                response = connection.getresponse()
                self.assertEqual(response.status, 400)
                response.read(); connection.close()

    def test_tool_turns_and_stream_equivalence(self):
        with fixture('tool') as port:
            messages = []
            for streamed in (False, True):
                status, _, wire = request(port, dict(BASE, max_tokens=1024, tools=[TOOL], stream=streamed))
                self.assertEqual(status, 200)
                message = reconstruct(wire) if streamed else json.loads(wire)
                self.check_message(message, 'tool_use')
                call = message['content'][0]
                self.assertEqual((call['type'], call['name'], call['input']), ('tool_use', 'lookup', {'city': 'Rome'}))
                self.assertNotIn(b'DSML', wire)
                history = BASE['messages'] + [{'role': 'assistant', 'content': message['content']},
                            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call['id'], 'content': 'ok'}]}]
                self.assertEqual(request(port, dict(BASE, messages=history, max_tokens=1, tools=[TOOL]))[0], 200)
                bad = copy.deepcopy(history)
                bad[-1]['content'][0]['tool_use_id'] = 'missing'
                self.assertEqual(request(port, dict(BASE, messages=bad, tools=[TOOL]))[0], 400)
                bad = copy.deepcopy(history)
                bad[-1]['content'].append(bad[-1]['content'][0])
                self.assertEqual(request(port, dict(BASE, messages=bad, tools=[TOOL]))[0], 400)
                self.assertEqual(request(port, dict(BASE, messages=history[:-1] + BASE['messages'], tools=[TOOL]))[0], 400)
                self.assertEqual(request(port, dict(BASE, messages=[history[-1]], tools=[TOOL]))[0], 400)
                messages.append(message)
            for message in messages:
                message.pop('id'); message['content'][0].pop('id')
            self.assertEqual(*messages)
        with fixture('invalid_tool') as port:
            for streamed in (False, True):
                status, _, wire = request(port, dict(BASE, max_tokens=1024, tools=[TOOL], stream=streamed))
                self.assertEqual(status, 200 if streamed else 500)
                self.assertNotIn(b'message_stop', wire)
                value = events(wire)[-1] if streamed else json.loads(wire)
                self.assertEqual(value['error']['type'], 'api_error')

    def test_default_off(self):
        with fixture('disabled') as port:
            variants = [b'Host: localhost\r\nContent-Length: 0\r\n',
                        b'Host: localhost\r\nTransfer-Encoding: chunked\r\n',
                        b'Host: localhost\r\nContent-Length: 33554433\r\n',
                        b'Content-Length: 0\r\n',
                        b'Host: localhost\r\nExpect: wrong\r\n',
                        b'Host: localhost\r\n' + b'X-Test: x\r\n' * 65,
                        b'Host: localhost\r\nanthropic-version: invalid\r\nanthropic-beta: x\r\n'
                        b'x-api-key: ignored\r\nContent-Length: 0\r\n']
            for fields in variants:
                with self.subTest(fields=fields[:80]):
                    expected = raw_exchange(port, b'POST /unknown HTTP/1.1\r\n' + fields + b'\r\n')
                    actual = raw_exchange(port, b'POST /v1/messages HTTP/1.1\r\n' + fields + b'\r\n')
                    self.assertEqual(actual, expected)

    def test_merged_metrics_and_messages(self):
        with fixture('both') as port:
            self.assertEqual(request(port)[0], 200)
            status, _, body = request(port, method='GET', path='/metrics', headers={'Authorization': 'Bearer ' + KEY})
            self.assertEqual(status, 200)
            self.assertIn(b'# HELP', body)
            def refusals(wire):
                rows = [line for line in wire.decode().splitlines()
                        if line.startswith('anemone_request_refusals_total{') and 'status_class="4xx"' in line]
                self.assertEqual(len(rows), 1)
                return float(rows[0].rsplit(' ', 1)[1])
            before = refusals(body)
            self.assertEqual(request(port, headers={'anthropic-version': 'invalid', 'x-api-key': KEY})[0], 400)
            self.assertEqual(request(port, headers={'anthropic-version': '2023-06-01', 'x-api-key': 'invalid_placeholder'})[0], 401)
            status, _, body = request(port, method='GET', path='/metrics', headers={'Authorization': 'Bearer ' + KEY})
            self.assertEqual(status, 200)
            self.assertEqual(refusals(body), before + 2)

    def test_messages_metrics_parity(self):
        with fixture('reasoning_metrics') as port:
            def scrape():
                status, _, wire = request(port, method='GET', path='/metrics', headers={'Authorization': 'Bearer ' + KEY})
                self.assertEqual(status, 200)
                text = wire.decode()
                return text, parse_metrics(text)[1]

            def finished(before, status, handler='/v1/messages'):
                count = metric_value(before, 'http_requests_total', handler=handler, method='POST', status=status)
                deadline = time.monotonic() + 5
                while True:
                    text, after = scrape()
                    if metric_value(after, 'http_requests_total', handler=handler, method='POST', status=status) == count + 1:
                        return text, after
                    if time.monotonic() > deadline:
                        self.fail('Messages HTTP accounting did not retire')
                    time.sleep(.01)

            # The same collector assertions also hold for the existing Chat endpoint.
            _, before = scrape()
            chat = dict(model='test-model', messages=BASE['messages'], max_tokens=8, reasoning_effort=0)
            status, _, wire = request(port, chat, path='/v1/chat/completions', headers={'Authorization': 'Bearer ' + KEY})
            self.assertEqual(status, 200)
            usage = json.loads(wire)['usage']
            _, after = finished(before, '2xx', '/v1/chat/completions')
            check_success(before, after, {'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens']},
                          handler='/v1/chat/completions', response_bytes=len(wire))
            for streaming in (False, True):
                for thinking in (False, True):
                    body = copy.deepcopy(BASE)
                    body.update(max_tokens=1200, stream=streaming, thinking={'type': 'enabled', 'budget_tokens': 1024} if thinking else {'type': 'disabled'})
                    _, before = scrape()
                    status, _, wire = request(port, body)
                    self.assertEqual(status, 200)
                    message = reconstruct(wire) if streaming else json.loads(wire)
                    self.assertEqual(any(block['type'] == 'thinking' for block in message['content']), thinking)
                    _, after = finished(before, '2xx')
                    # Non-stream responses carry Content-Length; chunked streams are measured as zero upstream.
                    check_success(before, after, message['usage'], response_bytes=None if streaming else len(wire))
            _, before = scrape()
            invalid = copy.deepcopy(BASE); invalid['max_tokens'] = 0
            self.assertEqual(request(port, invalid)[0], 400)
            text, after = finished(before, '4xx')
            check_refusal(before, after)
            compare_metrics(text, 16384, False, model='test-model', slots=4, prefix_reuse=False, messages=True)
            if artifact := os.environ.get('ANEMONE_MESSAGES_METRICS_ARTIFACT'):
                Path(artifact).write_text(text)

    def test_messages_metrics_invalid_tool(self):
        """Unsupported tool output answered with api_error is an error terminal, not a client cancellation, also
        when the connection closes before the engine's final event (invalid_tool_detached, the native order)."""
        for mode, streamed in itertools.product(('invalid_tool_metrics', 'invalid_tool_detached'), (False, True)):
            with self.subTest(mode=mode, streamed=streamed), fixture(mode) as port:
                self.invalid_tool_terminal(port, streamed)

    def invalid_tool_terminal(self, port, streamed):
        def scrape():
            status, _, wire = request(port, method='GET', path='/metrics', headers={'Authorization': 'Bearer ' + KEY})
            self.assertEqual(status, 200)
            return parse_metrics(wire.decode())[1]
        before = scrape()
        status, _, wire = request(port, dict(BASE, max_tokens=1024, tools=[TOOL], stream=streamed))
        self.assertEqual(status, 200 if streamed else 500)
        http_status = '2xx' if streamed else '5xx'
        # The answer completes before the engine's final event, so the terminal retires separately.
        retired = [('http_requests_total', dict(handler='/v1/messages', method='POST', status=http_status)),
                   ('anemone_requests_finished_total', {})]
        deadline = time.monotonic() + 5
        while True:
            after = scrape()
            if all(metric_value(after, name, **labels) == metric_value(before, name, **labels) + 1
                   for name, labels in retired):
                break
            if time.monotonic() > deadline:
                self.fail('Messages adapter error accounting did not retire')
            time.sleep(.01)
        check_error_terminal(before, after, http_status)

    def test_metrics_only_keeps_messages_disabled(self):
        with fixture('metrics_only') as port:
            auth = ('Authorization: Bearer ' + KEY + '\r\n').encode()
            suffix = b' HTTP/1.1\r\nHost: x\r\n' + auth + b'anthropic-version: invalid\r\nContent-Length: 2\r\n\r\n{}'
            expected = raw_exchange(port, b'POST /unknown' + suffix)
            self.assertTrue(expected.startswith(b'HTTP/1.1 404 '))
            self.assertEqual(raw_exchange(port, b'POST /v1/messages' + suffix), expected)
            status, _, wire = request(port, method='GET', path='/metrics', headers={'Authorization': 'Bearer ' + KEY})
            self.assertEqual(status, 200)
            samples = parse_metrics(wire.decode())[1]
            self.assertEqual(metric_value(samples, 'http_requests_total', handler='none', method='POST', status='4xx'), 2)
            self.assertFalse(any(x['labels'].get('handler') == '/v1/messages' for x in samples))

    def test_metrics_query_path_parity(self):
        with fixture('both') as port:
            for handler in ('/v1/chat/completions', '/v1/messages'):
                status, _, _ = request(port, path=handler + '?extra=1', headers={'Authorization': 'Bearer ' + KEY})
                self.assertEqual(status, 404)  # Shared router rejects query-bearing targets.
                deadline = time.monotonic() + 5
                while True:
                    status, _, wire = request(port, method='GET', path='/metrics', headers={'Authorization': 'Bearer ' + KEY})
                    self.assertEqual(status, 200)
                    samples = parse_metrics(wire.decode())[1]
                    if metric_value(samples, 'http_requests_total', handler=handler, method='POST', status='4xx') == 1:
                        break
                    if time.monotonic() > deadline:
                        self.fail('Query path handler accounting did not retire')
                    time.sleep(.01)
            self.assertEqual(metric_value(samples, 'vllm:request_success_total'), 0)
            self.assertEqual(metric_value(samples, 'anemone_requests_admitted_total'), 0)

    def test_messages_metrics_engine_error(self):
        with fixture('metrics_error') as port:
            def scrape():
                status, _, wire = request(port, method='GET', path='/metrics', headers={'Authorization': 'Bearer ' + KEY})
                self.assertEqual(status, 200)
                return parse_metrics(wire.decode())[1]
            before = scrape()
            self.assertEqual(request(port)[0], 500)
            deadline = time.monotonic() + 5
            while True:
                after = scrape()
                if metric_value(after, 'http_requests_total', handler='/v1/messages', method='POST', status='5xx') == 1:
                    break
                if time.monotonic() > deadline:
                    self.fail('Messages error HTTP accounting did not retire')
                time.sleep(.01)
            check_engine_error(before, after)

    def test_openai_falsy_top_k_identity(self):
        with fixture() as port:
            body = {'model': 'test-model', 'messages': BASE['messages'], 'max_tokens': 8, 'reasoning_effort': 0}
            baseline = request(port, body, path='/v1/chat/completions', headers={'Authorization': 'Bearer ' + KEY})
            expected = json.loads(baseline[2])['choices']
            for value in ('0.0', '-0', 'false', '""', '[]', '{}', 'null'):
                wire = json.dumps(body)[:-1] + ',"top_k":' + value + '}'
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
                connection.request('POST', '/v1/chat/completions', wire, {'Authorization': 'Bearer ' + KEY})
                response = connection.getresponse()
                result = response.read(); connection.close()
                self.assertEqual(response.status, 200, value)
                self.assertEqual(json.loads(result)['choices'], expected)

    def test_adaptive_real_context_room(self):
        with fixture('reasoning') as port:
            status, _, wire = request(port, dict(BASE, max_tokens=64000, thinking={'type': 'adaptive'},
                                                output_config={'effort': 'high'}))
            self.assertEqual(status, 200)
            value = json.loads(wire)
            room = 16384 - value['usage']['input_tokens']
            self.assertLessEqual(value['usage']['output_tokens_details']['thinking_tokens'], (room - 1) * 3 // 4)
            self.assertEqual(value['content'][-1]['text'], 'A')
            self.assertEqual(request(port, dict(BASE, max_tokens=64000,
                thinking={'type': 'enabled', 'budget_tokens': 63999}))[0], 400)

    def test_context_window_stop_reason(self):
        with fixture('slow') as port:
            for streamed in (False, True):
                status, _, wire = request(port, dict(BASE, messages=[{'role': 'user', 'content': 'x' * 15000}],
                                                    max_tokens=64000, stream=streamed))
                self.assertEqual(status, 200)
                value = reconstruct(wire) if streamed else json.loads(wire)
                self.assertEqual(value['stop_reason'], 'model_context_window_exceeded')
                self.assertEqual(value['usage']['input_tokens'] + value['usage']['output_tokens'], 16384)

    def test_tool_suffix_stays_replayable(self):
        with fixture('tool_suffix') as port:
            for streamed in (False, True):
                for maximum in (210, 1024):
                    status, _, wire = request(port, dict(BASE, max_tokens=maximum, tools=[TOOL], stream=streamed))
                    self.assertEqual(status, 200)
                    value = reconstruct(wire) if streamed else json.loads(wire)
                    self.assertEqual([item['type'] for item in value['content']], ['tool_use'])
                    self.assertNotIn(b'DSML', wire)
                    self.assertIn(value['stop_reason'], ('tool_use', 'max_tokens'))
                    calls = value['content']
                    history = BASE['messages'] + [{'role': 'assistant', 'content': calls}, {'role': 'user', 'content': [
                        {'type': 'tool_result', 'tool_use_id': item['id'], 'content': 'ok'} for item in calls]}]
                    self.assertEqual(request(port, dict(BASE, max_tokens=1, messages=history, tools=[TOOL]))[0], 200)

    def test_oversized_input_is_client_error(self):
        with fixture() as port:
            status, _, _ = request(port, dict(BASE, messages=[{'role': 'user', 'content': 'x' * (12 * 1024 * 1024)}]))
            self.assertIn(status, (400, 413))

    def test_oversized_normalized_body_is_client_error(self):
        with fixture() as port:
            history = BASE['messages'] + [{'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'oversized', 'name': 'lookup', 'input': {'payload': '"' * (9 * 1024 * 1024)}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'oversized', 'content': 'ok'}]}]
            self.assertEqual(request(port, dict(BASE, messages=history, tools=[TOOL]))[0], 413)

    def test_budgeted_reasoning_boundaries(self):
        for mode in ('reasoning_utf8', 'reasoning_literal'):
            with self.subTest(mode=mode), fixture(mode) as port:
                for streamed in (False, True):
                    status, _, wire = request(port, dict(BASE, max_tokens=1030, stream=streamed,
                        thinking={'type': 'enabled', 'budget_tokens': 1024}))
                    self.assertEqual(status, 200)
                    value = reconstruct(wire) if streamed else json.loads(wire)
                    self.assertEqual(value['content'][-1], {'type': 'text', 'text': 'A'})
                    self.assertNotIn('\ufffd', value['content'][0]['thinking'])
                    self.assertLessEqual(value['usage']['output_tokens_details']['thinking_tokens'], 1024)
                    if mode == 'reasoning_literal':
                        self.assertIn('</think>', value['content'][0]['thinking'])
                    else:
                        self.assertTrue(value['content'][0]['thinking'].startswith('\U0001f600'))
    def test_invalid_reasoning_output_does_not_poison_serving(self):
        with fixture('reasoning_invalid') as port:
            for streamed in (False, True):
                status, _, wire = request(port, dict(BASE, max_tokens=1030, stream=streamed,
                    thinking={'type': 'enabled', 'budget_tokens': 1024}))
                self.assertEqual(status, 200 if streamed else 500)
                value = events(wire)[-1] if streamed else json.loads(wire)
                self.assertEqual(value['error']['type'], 'api_error')
                self.assertEqual(request(port, dict(BASE, thinking={'type': 'disabled'}))[0], 200)

    def test_budgeted_calls_in_reasoning_are_refused(self):
        with fixture('reasoning_call') as port:
            for streamed in (False, True):
                status, _, wire = request(port, dict(BASE, max_tokens=1030, tools=[TOOL], stream=streamed,
                    thinking={'type': 'enabled', 'budget_tokens': 1024}))
                self.assertEqual(status, 200 if streamed else 500)
                value = events(wire)[-1] if streamed else json.loads(wire)
                self.assertEqual(value['error']['type'], 'api_error')
                self.assertNotIn(b'"type":"tool_use"', wire)

    def test_concurrency_queue_and_disconnect(self):
        with fixture('slow') as port:
            clients = []
            try:
                for index in range(4):
                    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
                    connection.request('POST', '/v1/messages', json.dumps(dict(BASE, stream=True, max_tokens=2048)),
                                       {'x-api-key': KEY, 'anthropic-version': '2023-06-01'})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    response.read(1)
                    clients.append((connection, response))
                # Waiters occupy the same per-user FIFO as Chat requests.
                for index in range(2):
                    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
                    body = json.dumps(BASE).encode()
                    sock.sendall(b'POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nanthropic-version: 2023-06-01\r\nx-api-key: ' + KEY.encode() + b'\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
                    clients.append((sock, None))
                time.sleep(0.05)
                status, headers, wire = request(port)
                self.assertEqual(status, 503)
                self.assertIn('Retry-After', headers)
                self.assertEqual(json.loads(wire)['error']['type'], 'overloaded_error')
                # Bob retains admission despite Alice's full per-user queue.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(request, port, dict(BASE, max_tokens=1), headers={'x-api-key': SECOND_KEY, 'anthropic-version': '2023-06-01'})
                    time.sleep(0.05)
                    for connection, response in reversed(clients):
                        if response is not None: response.close()
                        connection.close()
                    clients.clear()
                    self.assertEqual(future.result(timeout=10)[0], 200)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    status, _, wire = request(port, method='GET', path='/v1/anemone/stats', headers={'Authorization': 'Bearer ' + KEY})
                    self.assertEqual(status, 200)
                    stats = json.loads(wire)['global']
                    if stats['slots_busy'] == 0 and stats['queued_requests'] == 0:
                        break
                    time.sleep(0.02)
                self.assertEqual(stats['slots_busy'], 0)
                self.assertEqual(stats['queued_requests'], 0)
                self.assertGreaterEqual(stats['cancelled_requests'], 4)
                self.assertEqual(request(port, dict(BASE, max_tokens=1))[0], 200)
                def short(_):
                    return request(port, dict(BASE, max_tokens=1))[0]
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    self.assertEqual(list(pool.map(short, range(4))), [200] * 4)
            finally:
                for connection, response in clients:
                    if response is not None: response.close()
                    connection.close()

    def test_shared_top_k_controls(self):
        with fixture('sampling') as port:
            for top_k in (0, 1, 4, 129280):
                for streamed in (False, True):
                    status, _, wire = request(port, dict(BASE, top_k=top_k, stream=streamed, temperature=0.7, top_p=0.5))
                    self.assertEqual(status, 200)
                    value = reconstruct(wire) if streamed else json.loads(wire)
                    self.assertEqual(value['content'][0]['text'], 'K' if top_k else 'U')
                body = {'model': 'test-model', 'messages': BASE['messages'], 'max_tokens': 8,
                        'reasoning_effort': 0, 'top_k': top_k, 'temperature': 0.7, 'top_p': 0.5}
                status, _, wire = request(port, body, path='/v1/chat/completions', headers={'Authorization': 'Bearer ' + KEY})
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(wire)['choices'][0]['message']['content'], 'K' if top_k else 'U')
            for invalid in (-1, 1.5, 2097153, None):
                self.assertEqual(request(port, dict(BASE, top_k=invalid))[0], 400)

    def test_forced_close_serial_and_speculative(self):
        for mode in ('reasoning_single', 'reasoning_speculative'):
            with fixture(mode) as port:
                status, _, wire = request(port, dict(BASE, max_tokens=1030, thinking={'type': 'enabled', 'budget_tokens': 1024}, stream=True))
                self.assertEqual(status, 200)
                value = reconstruct(wire)
                self.assertEqual(value['content'][0]['thinking'], 'r' * 1023)
                self.assertEqual(value['content'][1]['text'], 'A')
                self.assertEqual(value['usage']['output_tokens_details']['thinking_tokens'], 1024)

    def test_thinking_budgets_adaptive_and_switch_cache(self):
        with fixture('reasoning') as port:
            for streamed in (False, True):
                body = dict(BASE, max_tokens=1028, thinking={'type': 'enabled', 'budget_tokens': 1024}, stream=streamed)
                status, _, wire = request(port, body)
                self.assertEqual(status, 200)
                message = reconstruct(wire) if streamed else json.loads(wire)
                self.check_message(message)
                self.assertEqual(message['content'][0]['type'], 'thinking')
                self.assertEqual(message['content'][0]['thinking'], 'r' * 1023)
                self.assertEqual(message['content'][0]['signature'], 'anemone-v1-effort-75')
                self.assertEqual(message['content'][1], {'type': 'text', 'text': 'A'})
                self.assertEqual(message['usage']['output_tokens_details']['thinking_tokens'], 1024)
                self.assertEqual(message['usage']['output_tokens'], 1026)
                self.assertNotIn(b'</think>', wire)
            for effort, native in (('low', 50), ('medium', 75), ('high', 75), ('xhigh', 75), ('max', 100)):
                status, _, wire = request(port, dict(BASE, max_tokens=16, thinking={'type': 'adaptive'}, output_config={'effort': effort}, stream=True))
                self.assertEqual(status, 200)
                message = reconstruct(wire)
                self.assertEqual(message['content'][0]['signature'], 'anemone-v1-effort-' + str(native))
                self.assertEqual(message['content'][-1]['text'], 'A')
                self.assertLess(message['usage']['output_tokens_details']['thinking_tokens'], 16)
            history = [{'role': 'user', 'content': 'Retain this conversation: ' + 'context ' * 96}]
            hits = []
            for setting in ({'thinking': {'type': 'disabled'}, 'max_tokens': 8},
                            {'thinking': {'type': 'enabled', 'budget_tokens': 1024}, 'max_tokens': 1030},
                            {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'low'}, 'max_tokens': 16},
                            {'thinking': {'type': 'disabled'}, 'max_tokens': 8}):
                status, _, wire = request(port, dict(BASE, messages=history, **setting), headers={'x-api-key': SECOND_KEY, 'anthropic-version': '2023-06-01'})
                self.assertEqual(status, 200)
                message = json.loads(wire)
                self.assertEqual(message['content'][-1]['text'], 'A')
                status, _, wire = request(port, method='GET', path='/v1/anemone/stats', headers={'Authorization': 'Bearer ' + KEY})
                stats = json.loads(wire)
                # The fixture's final event is published before the completed HTTP response.
                record = next(item for item in stats['finished'] if item['identity']['request_tag'] == max(row['identity']['request_tag'] for row in stats['finished']))
                hits.append(record['prompt']['reused_tokens'])
                history += [{'role': 'assistant', 'content': message['content']}, {'role': 'user', 'content': 'Continue this same conversation.'}]
            self.assertEqual(hits[0], 0)
            self.assertTrue(all(count > 512 for count in hits[1:]), hits)
            print('thinking_switch_prefix_hits=' + json.dumps(hits), flush=True)
            try:
                import anthropic
            except ImportError:
                return
            with anthropic.Anthropic(api_key=KEY, base_url=f'http://127.0.0.1:{port}', max_retries=0,
                                     default_headers={'anthropic-version': '2023-06-01'}) as client:
                with client.messages.stream(**dict(BASE, max_tokens=1028, thinking={'type': 'enabled', 'budget_tokens': 1024})) as stream:
                    value = stream.get_final_message()
                    self.assertEqual(value.content[0].type, 'thinking')
                    self.assertEqual(value.content[1].text, 'A')
                reply = client.messages.create(**dict(BASE, max_tokens=16, thinking={'type': 'adaptive'}, output_config={'effort': 'max'}))
                self.assertEqual(reply.content[0].signature, 'anemone-v1-effort-100')

    def test_native_sdk_scenarios(self):
        try:
            import anthropic
        except ImportError:
            self.skipTest('official SDK is not installed in this Python environment')
        modes = {'text': 'text', 'sampling': 'sampling', 'versions': 'text',
                 'thinking': 'reasoning', 'tools': 'tool'}
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario), fixture(modes[scenario]) as port:
                with anthropic.Anthropic(api_key=KEY, base_url=f'http://127.0.0.1:{port}', max_retries=0,
                        default_headers={'anthropic-version': LEGACY_VERSION}) as client:
                    result = run_sdk_scenario(client, scenario)
                    if scenario == 'text':
                        self.assertEqual(result['complete']['content'][0]['text'], 'Hi')
                        self.assertEqual(result['streamed']['content'][0]['text'], 'Hi')
                    elif scenario == 'sampling':
                        self.assertEqual(result['sampled']['content'][0]['text'], 'K')
                        self.assertEqual(result['streamed']['content'][0]['text'], 'K')
                    elif scenario == 'versions':
                        self.assertEqual(result['sampling_refused'], ['temperature', 'top_p', 'top_k'])
                    elif scenario == 'thinking':
                        self.assertEqual(result['enabled_streamed']['content'][1]['text'], 'A')
                        self.assertEqual(result['adaptive']['content'][1]['text'], 'A')
                    elif scenario == 'tools':
                        self.assertEqual(result['proposal']['content'][0]['input'], {'city': 'Rome'})

    def test_official_sdk(self):
        try:
            import anthropic
        except ImportError:
            self.skipTest('official SDK is not installed in this Python environment')
        with fixture() as port:
            with anthropic.Anthropic(api_key=KEY, base_url=f'http://127.0.0.1:{port}', max_retries=0,
                                     default_headers={'anthropic-version': '2023-06-01'}) as client:
                message = client.messages.create(**BASE)
                self.assertEqual(message.content[0].text, 'Hi')
                with client.messages.stream(**BASE) as stream:
                    self.assertEqual(''.join(stream.text_stream), 'Hi')
                    self.assertEqual(stream.get_final_message().stop_reason, 'end_turn')
                with self.assertRaises(anthropic.BadRequestError):
                    client.messages.create(**BASE, thinking={'type': 'enabled', 'budget_tokens': 1})
        with fixture('tool') as port:
            with anthropic.Anthropic(api_key=KEY, base_url=f'http://127.0.0.1:{port}', max_retries=0,
                                     default_headers={'anthropic-version': '2023-06-01'}) as client:
                body = dict(BASE, max_tokens=1024, tools=[TOOL])
                with client.messages.stream(**body) as stream:
                    message = stream.get_final_message()
                    self.assertEqual(message.stop_reason, 'tool_use')
                    self.assertEqual(message.content[0].input, {'city': 'Rome'})
                body['messages'] = BASE['messages'] + [{'role': 'assistant', 'content': [item.model_dump() for item in message.content]},
                        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': message.content[0].id, 'content': 'ok'}]}]
                body['max_tokens'] = 1
                self.assertEqual(client.messages.create(**body).stop_reason, 'max_tokens')

if __name__ == '__main__':
    unittest.main(verbosity=2)
