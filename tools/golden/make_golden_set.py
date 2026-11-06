#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Builds the golden set for Anemone's front end from DeepSeek's reference encoder.

For every request below, and for the five test pairs shipped with the official checkpoint, it records the prompt
text the reference encoder produces, its token ids, and the images in prompt order. For every completion text it
records what the reference parser returns. Anemone's own template and parser must reproduce these byte for byte.

    make_golden_set.py --encoding-directory <checkpoint>/encoding --tokenizer-json <checkpoint>/tokenizer.json \
                       --output-directory <folder>
    make_golden_set.py ... --self-test      # shows the checks failing on a corrupted expectation

Measurement and test tool: not part of the engine.
"""
import argparse
import copy
import glob
import json
import os
import sys

THINKING = "thinking"
CHAT = "chat"
REASONING_EFFORT_LEVELS = (1, 25, 50, 75, 100, "low", "high", "max")
ONE_PIXEL_PNG_DATA_URL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
                          "60e6kgAAAABJRU5ErkJggg==")

C_SOURCE_SAMPLE = ("#include <stddef.h>\r\n\r\nstatic size_t count_set_bits(unsigned long long value)\r\n{\r\n"
                   "\tsize_t count = 0;\r\n\twhile (value) { value &= value - 1; count++; }\r\n\treturn count;\r\n}\r\n")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                      "units": {"type": "string", "enum": ["metric", "imperial"]}},
                       "required": ["city"]},
    },
}
FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "range": {"type": "object", "properties": {"first_line": {"type": "integer"},
                                                                                 "last_line": {"type": "integer"}}},
                                      "encodings": {"type": "array", "items": {"type": "string"}}},
                       "required": ["path"]},
    },
}
NAMESPACED_TOOL = {
    "type": "function",
    "namespace": {"name": "search", "description": "Search tools."},
    "function": {"name": "lookup", "description": "Look up a value.",
                 "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}},
}


def tool_call(call_id, name, arguments, namespace = None):
    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii = False)}}
    if namespace:
        call["namespace"] = namespace
    return call


def build_requests():
    """Our own requests: every feature the API has to serve, each as one named case."""
    system = {"role": "system", "content": "You are a careful C programmer."}
    first_user = {"role": "user", "content": "Count the set bits of a 64-bit value."}
    first_assistant = {"role": "assistant", "reasoning_content": "Kernighan's loop clears one bit per step.",
                       "content": "Use value &= value - 1 in a loop and count the steps."}
    second_user = {"role": "user", "content": "Show it as code.\n\n" + C_SOURCE_SAMPLE}
    requests = [
        {"name": "plain_chat_single_turn", "thinking_mode": CHAT, "messages": [first_user]},
        {"name": "plain_chat_with_system", "thinking_mode": CHAT, "messages": [system, first_user]},
        {"name": "thinking_default_effort", "thinking_mode": THINKING, "messages": [system, first_user]},
        {"name": "thinking_without_system", "thinking_mode": THINKING, "reasoning_effort": 75, "messages": [first_user]},
        {"name": "thinking_two_turns_earlier_reasoning_dropped", "thinking_mode": THINKING, "drop_thinking": True,
         "messages": [system, first_user, first_assistant, second_user]},
        {"name": "thinking_two_turns_earlier_reasoning_preserved", "thinking_mode": THINKING, "drop_thinking": False,
         "messages": [system, first_user, first_assistant, second_user]},
        {"name": "chat_two_turns_with_reasoning_sent_back", "thinking_mode": CHAT,
         "messages": [system, first_user, first_assistant, second_user]},
        {"name": "unicode_and_carriage_returns", "thinking_mode": CHAT,
         "messages": [{"role": "user", "content": "Übersetze: 你好，世界 🌍\ttab\r\nzeile zwei — „Zitat“"}]},
        {"name": "mid_conversation_system_message", "thinking_mode": THINKING,
         "messages": [system, first_user, first_assistant, {"role": "system", "content": "From now on answer in one line."}, second_user]},
        {"name": "latest_reminder", "thinking_mode": THINKING,
         "messages": [system, {"role": "latest_reminder", "content": "Today is 2026-10-08."}, first_user]},
        {"name": "conversation_resent_base", "thinking_mode": THINKING, "messages": [system, first_user]},
        {"name": "conversation_resent_extended", "thinking_mode": THINKING,
         "messages": [system, first_user, first_assistant, second_user]},
    ]
    for effort in REASONING_EFFORT_LEVELS:
        requests.append({"name": f"thinking_effort_{effort}", "thinking_mode": THINKING, "reasoning_effort": effort,
                         "messages": [system, first_user]})
    weather_question = {"role": "user", "content": "Weather in Turin and in Oslo for three days?"}
    weather_calls = {"role": "assistant", "reasoning_content": "Two cities, two calls.", "content": "",
                     "tool_calls": [tool_call("call_1", "get_weather", {"city": "Turin", "days": 3, "units": "metric"}),
                                    tool_call("call_2", "get_weather", {"city": "Oslo", "days": 3})]}
    turin_result = {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"city": "Turin", "high_c": [21, 22, 19]})}
    oslo_result = {"role": "tool", "tool_call_id": "call_2", "content": json.dumps({"city": "Oslo", "high_c": [9, 8, 11]})}
    weather_answer = {"role": "assistant", "reasoning_content": "Both results are in.", "content": "Turin is warmer on all three days."}
    requests += [
        {"name": "tools_offered_no_call_yet", "thinking_mode": THINKING, "tools": [WEATHER_TOOL, FILE_TOOL],
         "messages": [system, weather_question]},
        {"name": "tools_offered_chat_mode", "thinking_mode": CHAT, "tools": [WEATHER_TOOL], "messages": [system, weather_question]},
        {"name": "tools_two_calls_with_results", "thinking_mode": THINKING, "tools": [WEATHER_TOOL],
         "messages": [system, weather_question, weather_calls, turin_result, oslo_result]},
        {"name": "tools_results_sent_out_of_order", "thinking_mode": THINKING, "tools": [WEATHER_TOOL],
         "messages": [system, weather_question, weather_calls, oslo_result, turin_result]},
        {"name": "tools_follow_up_turn_keeps_reasoning", "thinking_mode": THINKING, "tools": [WEATHER_TOOL],
         "messages": [system, weather_question, weather_calls, turin_result, oslo_result, weather_answer,
                      {"role": "user", "content": "And which is drier?"}]},
        {"name": "tools_nested_arguments", "thinking_mode": THINKING, "tools": [FILE_TOOL],
         "messages": [system, {"role": "user", "content": "Read lines 10 to 20 of main.c."},
                      {"role": "assistant", "reasoning_content": "", "content": "",
                       "tool_calls": [tool_call("call_1", "read_file", {"path": "main.c", "range": {"first_line": 10, "last_line": 20},
                                                                          "encodings": ["utf-8", "latin-1"]})]},
                      {"role": "tool", "tool_call_id": "call_1", "content": C_SOURCE_SAMPLE}]},
        {"name": "tools_namespaced", "thinking_mode": THINKING, "tools": [NAMESPACED_TOOL],
         "messages": [system, {"role": "user", "content": "Look up 'anemone'."},
                      {"role": "assistant", "reasoning_content": "Use the search tool.", "content": "",
                       "tool_calls": [tool_call("call_1", "lookup", {"query": "anemone", "limit": 2}, namespace = "search")]},
                      {"role": "tool", "tool_call_id": "call_1", "content": "[\"sea anemone\", \"wood anemone\"]"}]},
        {"name": "image_one", "thinking_mode": CHAT,
         "messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this picture?"},
                                                   {"type": "image_url", "image_url": {"url": "examples/images/carrots.jpeg"}}]}]},
        {"name": "image_two_interleaved_thinking", "thinking_mode": THINKING, "reasoning_effort": "max",
         "messages": [system, {"role": "user", "content": [{"type": "text", "text": "First:"},
                                                           {"type": "image_url", "image_url": {"url": "examples/images/carrots.jpeg"}},
                                                           {"type": "text", "text": "second:"},
                                                           {"type": "image_url", "image_url": {"url": "examples/images/corn.jpeg"}},
                                                           {"type": "text", "text": "Which is orange?"}]}]},
        {"name": "image_as_base64_data_url", "thinking_mode": CHAT,
         "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG_DATA_URL}},
                                                   {"type": "text", "text": "Describe it."}]}]},
    ]
    return requests


def build_completions():
    """Model output texts and the mode they were produced in; the reference parser gives the expected message."""
    end = "<｜end▁of▁sentence｜>"
    one_call = ('\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
                '<｜DSML｜ parameter name="city" string="true">Turin</｜DSML｜ parameter>\n'
                '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>\n'
                '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    two_calls = one_call.replace("</｜DSML｜ calls>", '<｜DSML｜ invoke name="read_file">\n'
                                 '<｜DSML｜ parameter name="path" string="true">main.c</｜DSML｜ parameter>\n'
                                 '<｜DSML｜ parameter name="range" string="false">{"first_line": 10, "last_line": 20}</｜DSML｜ parameter>\n'
                                 '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    namespaced_call = one_call.replace('name="get_weather"', 'name="search::lookup"')
    return [
        {"name": "thinking_then_answer", "thinking_mode": THINKING, "text": "Count with a loop.</think>Use Kernighan's method." + end},
        {"name": "thinking_empty_reasoning", "thinking_mode": THINKING, "text": "</think>Done." + end},
        {"name": "chat_answer_only", "thinking_mode": CHAT, "text": "Use Kernighan's method." + end},
        {"name": "chat_answer_with_code", "thinking_mode": CHAT, "text": "Here:\n\n" + C_SOURCE_SAMPLE + end},
        {"name": "thinking_one_tool_call", "thinking_mode": THINKING, "text": "Need the weather.</think>Checking." + one_call + end},
        {"name": "thinking_two_tool_calls", "thinking_mode": THINKING, "text": "Two lookups.</think>" + two_calls + end},
        {"name": "chat_one_tool_call", "thinking_mode": CHAT, "text": one_call + end},
        {"name": "thinking_namespaced_tool_call", "thinking_mode": THINKING, "text": "Search.</think>" + namespaced_call + end},
        {"name": "malformed_missing_end_token", "thinking_mode": CHAT, "text": "No end token"},
        {"name": "malformed_unclosed_thinking", "thinking_mode": THINKING, "text": "Never closes" + end},
    ]


def encode_request(encoding, tokenizer, request):
    messages = copy.deepcopy(request["messages"])
    if "tools" in request:
        messages[0]["tools"] = request["tools"]
    options = {"thinking_mode": request["thinking_mode"], "return_multi_modal_data": True}
    for optional in ("reasoning_effort", "drop_thinking", "context"):
        if request.get(optional) is not None:
            options[optional] = request[optional]
    record = {"name": request["name"], "request": {key: value for key, value in request.items() if key != "name"}}
    try:
        prompt, media = encoding.encode_messages(messages, **options)
    except Exception as error:                                   # a refusal by the reference is part of the contract
        record["reference_error"] = f"{type(error).__name__}: {error}"
        return record
    record["expected_prompt"] = prompt
    record["expected_token_ids"] = tokenizer.encode(prompt, add_special_tokens = False).ids
    record["expected_image_count"] = len(media["images"])
    return record


def encode_official_pairs(encoding, tokenizer, tests_directory):
    """The checkpoint's own test pairs: the reference must reproduce its expected text, or this script is wrong."""
    records = []
    for input_path in sorted(glob.glob(os.path.join(tests_directory, "test_input_*.json"))):
        case_number = os.path.basename(input_path)[len("test_input_"):-len(".json")]
        with open(os.path.join(tests_directory, f"test_output_{case_number}.txt"), encoding = "utf8") as handle:
            expected_text = handle.read()
        prompts = []
        for case in encoding.load_cases(input_path):
            prompt, images = encoding.encode_case(case, thinking_mode = CHAT)
            prompts.append(prompt)
        produced_text = "\n".join(prompts)
        records.append({"name": f"official_pair_{case_number}", "input_file": os.path.basename(input_path),
                        "expected_prompt": produced_text, "matches_shipped_expectation": produced_text.strip() == expected_text.strip(),
                        "expected_token_ids": tokenizer.encode(produced_text, add_special_tokens = False).ids})
    return records


def parse_completion(encoding, completion):
    record = dict(completion)
    try:
        record["expected_message"] = encoding.parse_message_from_completion_text(completion["text"], thinking_mode = completion["thinking_mode"])
    except Exception as error:
        record["reference_error"] = f"{type(error).__name__}: {error}"
    return record


def shared_prefix_length(first_ids, second_ids):
    length = 0
    for first, second in zip(first_ids, second_ids):
        if first != second:
            break
        length += 1
    return length


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoding-directory", required = True)
    parser.add_argument("--tokenizer-json", required = True)
    parser.add_argument("--output-directory", required = True)
    parser.add_argument("--self-test", action = "store_true", help = "corrupt one expectation and show the comparison failing")
    arguments = parser.parse_args()

    sys.path.insert(0, arguments.encoding_directory)
    os.chdir(os.path.dirname(os.path.abspath(arguments.encoding_directory)))      # image paths are relative to the checkpoint
    import encoding
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(arguments.tokenizer_json)

    official = encode_official_pairs(encoding, tokenizer, os.path.join(arguments.encoding_directory, "tests"))
    prompts = [encode_request(encoding, tokenizer, request) for request in build_requests()]
    completions = [parse_completion(encoding, completion) for completion in build_completions()]

    by_name = {record["name"]: record for record in prompts}
    base, extended = by_name["conversation_resent_base"], by_name["conversation_resent_extended"]
    cache_pairs = [{"first": base["name"], "second": extended["name"],
                    "shared_prefix_tokens": shared_prefix_length(base["expected_token_ids"], extended["expected_token_ids"]),
                    "first_tokens": len(base["expected_token_ids"]), "second_tokens": len(extended["expected_token_ids"])}]

    if arguments.self_test:
        official[0]["matches_shipped_expectation"] = (official[0]["expected_prompt"] + "x").strip() == official[0]["expected_prompt"].strip()

    os.makedirs(arguments.output_directory, exist_ok = True)
    for file_name, content in (("golden_official_pairs.json", official), ("golden_prompts.json", prompts),
                               ("golden_completions.json", completions), ("golden_cache_pairs.json", cache_pairs)):
        with open(os.path.join(arguments.output_directory, file_name), "w", encoding = "utf8") as handle:
            json.dump(content, handle, ensure_ascii = False, indent = 1)

    failed_official = [record["name"] for record in official if not record["matches_shipped_expectation"]]
    refused = [record["name"] for record in prompts if "reference_error" in record]
    print(f"official pairs: {len(official)}, not reproduced: {failed_official or 'none'}")
    print(f"our requests: {len(prompts)}, refused by the reference: {len(refused)}")
    for record in prompts:
        if "reference_error" in record:
            print(f"  refused {record['name']}: {record['reference_error'][:160]}")
    print(f"prompt tokens in all: {sum(len(record.get('expected_token_ids', ())) for record in prompts)}")
    print(f"completions: {len(completions)}, rejected by the reference parser: "
          f"{[record['name'] for record in completions if 'reference_error' in record]}")
    print(f"cache pair: {cache_pairs[0]}")
    special_token_check = tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>x<｜Assistant｜><think>", add_special_tokens = False).ids
    print(f"special tokens are single ids: {len(special_token_check) == 5} ({special_token_check})")
    return 1 if failed_official else 0


if __name__ == "__main__":
    sys.exit(main())
