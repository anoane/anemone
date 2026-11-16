# SPDX-License-Identifier: AGPL-3.0-or-later
"""SDK scenarios shared by CPU contract fixtures and admitted native smoke jobs."""

LEGACY_VERSION = '2023-06-01'
CURRENT_VERSION = '2026-10-21'
BASE = {
    'model': 'anemone',
    'max_tokens': 16,
    'messages': [{'role': 'user', 'content': 'Reply with one short greeting.'}],
    # SDK 2.6 removed typed sampling keywords. The legacy API version still accepts them on the wire.
    'extra_body': {'temperature': 0},
}
TOOL = {
    'name': 'lookup',
    'description': 'Look up a city',
    'input_schema': {'type': 'object', 'properties': {'city': {'type': 'string'}}},
}
SCENARIOS = ('text', 'sampling', 'versions', 'thinking', 'tools')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked(message):
    value = message.model_dump()
    require(value['type'] == 'message' and value['model'] == 'anemone', 'SDK Message identity')
    require(value['stop_reason'] in ('end_turn', 'max_tokens', 'model_context_window_exceeded', 'tool_use', 'stop_sequence'), 'SDK stop reason')
    require(value['usage']['input_tokens'] > 0 and value['usage']['output_tokens'] > 0, 'SDK token usage')
    return value


def run_sdk_scenario(client, scenario):
    """The client uses LEGACY_VERSION by default; current-version probes override its header explicitly."""
    if scenario == 'text':
        complete = checked(client.messages.create(**BASE))
        with client.messages.stream(**BASE) as stream:
            streamed = checked(stream.get_final_message())
        return {'complete': complete, 'streamed': streamed}
    if scenario == 'sampling':
        sampled = checked(client.messages.create(**dict(
            BASE, extra_body={'temperature': 1, 'top_p': 0.6, 'top_k': 5})))
        with client.messages.stream(**dict(BASE, extra_body={'temperature': 0, 'top_k': 1})) as stream:
            streamed = checked(stream.get_final_message())
        return {'sampled': sampled, 'streamed': streamed}
    if scenario == 'versions':
        import anthropic
        current = dict(BASE, extra_body={}, extra_headers={'anthropic-version': CURRENT_VERSION})
        complete = checked(client.messages.create(**current))
        refused = []
        for field in ('temperature', 'top_p', 'top_k'):
            try:
                client.messages.create(**dict(current, extra_body={field: 1}))
            except anthropic.BadRequestError as error:
                require(error.status_code == 400 and error.response.json()['error']['type'] == 'invalid_request_error',
                        'SDK current-version refusal shape')
                refused.append(field)
            else:
                raise RuntimeError('Current version accepted ' + field)
        return {'complete': complete, 'sampling_refused': refused}
    if scenario == 'thinking':
        with client.messages.stream(**dict(BASE, max_tokens=1056,
                thinking={'type': 'enabled', 'budget_tokens': 1024})) as stream:
            enabled = checked(stream.get_final_message())
        require(enabled['content'][0]['type'] == 'thinking', 'SDK thinking stream block')
        require(enabled['usage']['output_tokens_details']['thinking_tokens'] <= 1024, 'SDK thinking budget')
        adaptive = checked(client.messages.create(**dict(BASE, max_tokens=64,
            thinking={'type': 'adaptive'}, output_config={'effort': 'high'})))
        require(adaptive['content'][0]['type'] == 'thinking', 'SDK adaptive block')
        return {'enabled_streamed': enabled, 'adaptive': adaptive}
    if scenario == 'tools':
        question = {'role': 'user', 'content': 'Call lookup with city Rome. Do not answer directly.'}
        proposed = client.messages.create(**dict(BASE, max_tokens=256, tools=[TOOL], messages=[question]))
        proposal = checked(proposed)
        require(proposed.stop_reason == 'tool_use', 'Native model did not produce the requested tool call')
        results = [{'type': 'tool_result', 'tool_use_id': item.id, 'content': 'Rome is in Italy.'}
                   for item in proposed.content if item.type == 'tool_use']
        require(len(results) > 0, 'SDK tool-use block missing')
        follow = dict(BASE, tools=[TOOL], messages=[question,
            {'role': 'assistant', 'content': [item.model_dump() for item in proposed.content]},
            {'role': 'user', 'content': results}])
        return {'proposal': proposal, 'follow': checked(client.messages.create(**follow))}
    raise ValueError('Unknown SDK scenario: ' + scenario)
