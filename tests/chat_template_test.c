/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * chat_template_test: source/chat_template.c and source/json.c against chat_template.h and json.h. The encoder's own
 * prompts are front_end_golden_test's; here are prompts written out by hand for one case of each feature, JSON values
 * spelled as Python writes them, every refusal of the contract once, the exact capacity, the limits of depth (in a
 * response format and in a call's arguments) and of calls, the fault point, and conversations changed at random
 * byte by byte, which must render within their capacity or be refused, and nothing else.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/chat_template.h"
#include "anemone/hooks.h"
#include "anemone/status.h"
#include "json.h"

#define TEST_TIME_LIMIT_SECONDS 600
#define BEGIN "<｜begin▁of▁sentence｜>"
#define USER "<｜User｜>"
#define ASSISTANT "<｜Assistant｜>"
#define END "<｜end▁of▁sentence｜>"
#define EFFORT(level) "<｜System｜>Reasoning Effort: " level \
    " (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
#define DSML "｜DSML｜"
#define PROMPT_CAPACITY 65536
#define LONG_JSON_CAPACITY 262144
#define DEFAULT_EFFORT 75
#define LOWEST_EFFORT CHAT_TEMPLATE_LOWEST_EFFORT
#define HIGHEST_EFFORT CHAT_TEMPLATE_HIGHEST_EFFORT
#define ARRAYS_AROUND_A_FORMAT 2                  /* the messages array and the message object */
#define FORMAT_OPENING "[{\"role\":\"system\",\"content\":\"s\",\"response_format\":"
#define FORMAT_CLOSING "}]"
#define ARGUMENTS_OPENING "[{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"f\",\"arguments\":\""
#define ARGUMENTS_CLOSING "\"}}]}]"
#define CALLS(parameters) "\n\n<" DSML " calls>\n<" DSML " invoke name=\"f\">\n" parameters "\n</" DSML " invoke>\n</" \
    DSML " calls>"
#define IMAGE "<｜deepseek_image｜>"
/* A turn whose answer made a call, its result after it, no tools declared: the encoder's own test of 2026-10-16. */
#define TURNS_OPENING "[{\"role\":\"user\",\"content\":\"q1\"},{\"role\":\"assistant\",\"content\":\"a1\"," \
    "\"reasoning_content\":\"r1\"},{\"role\":\"user\",\"content\":\"q2\"},{\"role\":\"assistant\","             \
    "\"reasoning_content\":\"r2\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":"   \
    "\"f\",\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"value\"}"
#define TURNS TURNS_OPENING "]"
#define TURNS_THEN(message) TURNS_OPENING "," message "]"
/* pi-ai replay: every assistant has reasoning_content, including an empty tool-call assistant. */
#define HARNESS_LOOP_OPENING "[{\"role\":\"system\",\"content\":\"S\"},{\"role\":\"user\",\"content\":\"q0\"}," \
    "{\"role\":\"assistant\",\"content\":\"a0\"," \
    "\"reasoning_content\":\"old\"},{\"role\":\"user\",\"content\":\"q1\"},{\"role\":\"assistant\",\"content\":null," \
    "\"reasoning_content\":\"\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\"," \
    "\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"first\\nvalue\"}," \
    "{\"role\":\"assistant\",\"content\":\"checking\",\"reasoning_content\":\"  r2\\n\\u00e9  \",\"tool_calls\":[" \
    "{\"id\":\"c2\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]}," \
    "{\"role\":\"tool\",\"tool_call_id\":\"c2\",\"content\":[{\"type\":\"text\",\"text\":\"second\"}]}"
#define HARNESS_LOOP HARNESS_LOOP_OPENING "]"
#define HARNESS_NEXT_USER HARNESS_LOOP_OPENING ",{\"role\":\"user\",\"content\":\"q2\"}]"
#define HARNESS_OLD_DROPPED "S" USER "q0" ASSISTANT "</think>a0" END USER "q1" ASSISTANT
#define HARNESS_OLD_KEPT "S" USER "q0" ASSISTANT "<think>old</think>a0" END USER "q1" ASSISTANT
#define HARNESS_CHAT_LOOP "</think>" CALLS("") END USER "<tool_result>first\nvalue</tool_result>" ASSISTANT \
    "</think>checking" CALLS("") END USER "<tool_result>second</tool_result>"
#define HARNESS_THINKING_LOOP "<think></think>" CALLS("") END USER "<tool_result>first\nvalue</tool_result>" \
    ASSISTANT "<think>  r2\n\xc3\xa9  </think>checking" CALLS("") END USER "<tool_result>second</tool_result>"
#define UNSUPPORTED STATUS_UNSUPPORTED
#define WHAT_CAPACITY 256
#define FUZZ_ROUNDS 30000
#define MOST_MUTATIONS 4
#define FAULT_ON_FIRST_CALL 1
#define RANDOM_SEED 0x2545F4914F6CDD1Dull
#define SPLITMIX_INCREMENT 0x9E3779B97F4A7C15ull
#define SPLITMIX_FIRST_MULTIPLIER 0xBF58476D1CE4E5B9ull
#define SPLITMIX_SECOND_MULTIPLIER 0x94D049BB133111EBull
#define SPLITMIX_FIRST_SHIFT 30
#define SPLITMIX_SECOND_SHIFT 27
#define SPLITMIX_THIRD_SHIFT 31
#define INVALID STATUS_INVALID_ARGUMENT

enum option_flag { THINKING = 1, DROPS = 2, CONTINUES = 4 };
enum mutation { FLIP, INSERT, DELETE, REPEAT, CUT, MUTATION_COUNT };

struct prompt_case {
    const char *what;
    const char *messages;
    const char *tools;                            /* NULL for none */
    unsigned flags;
    const char *prompt;                           /* NULL: refused */
};

static const struct prompt_case prompt_cases[] = {
    {"a user in chat mode", "[{\"role\": \"user\", \"content\": \"hi\"}]", NULL, 0,
     BEGIN USER "hi" ASSISTANT "</think>"},
    {"a system and a user, thinking", "[{\"role\":\"system\",\"content\":\"S\"},{\"role\":\"user\",\"content\":\"Q\"}]",
     NULL, THINKING, BEGIN EFFORT("75") "S" USER "Q" ASSISTANT "<think>"},
    {"earlier reasoning dropped", "[{\"role\":\"user\",\"content\":\"q1\"},{\"role\":\"assistant\",\"content\":\"a1\","
     "\"reasoning_content\":\"r1\"},{\"role\":\"user\",\"content\":\"q2\"}]", NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "q1" ASSISTANT "</think>a1" END USER "q2" ASSISTANT "<think>"},
    {"earlier reasoning kept", "[{\"role\":\"user\",\"content\":\"q1\"},{\"role\":\"assistant\",\"content\":\"a1\","
     "\"reasoning_content\":\"r1\"},{\"role\":\"user\",\"content\":\"q2\"}]", NULL, THINKING,
     BEGIN EFFORT("75") USER "q1" ASSISTANT "<think>r1</think>a1" END USER "q2" ASSISTANT "<think>"},
    {"calls, and their results in the order of the calls", "[{\"role\":\"user\",\"content\":\"u\"},"
     "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"a\",\"type\":\"function\",\"function\":"
     "{\"name\":\"f\",\"arguments\":{\"x\":1}}},{\"id\":\"b\",\"function\":{\"name\":\"f\",\"arguments\":"
     "\"{\\\"s\\\": \\\"t\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"b\",\"content\":\"B\"},"
     "{\"role\":\"tool\",\"tool_call_id\":\"a\",\"content\":\"A\"}]", NULL, 0,
     BEGIN USER "u" ASSISTANT "</think>\n\n<" DSML " calls>\n<" DSML " invoke name=\"f\">\n<" DSML " parameter "
     "name=\"x\" string=\"false\">1</" DSML " parameter>\n</" DSML " invoke>\n<" DSML " invoke name=\"f\">\n<" DSML
     " parameter name=\"s\" string=\"true\">t</" DSML " parameter>\n</" DSML " invoke>\n</" DSML " calls>" END USER
     "<tool_result>A</tool_result>\n\n<tool_result>B</tool_result>" ASSISTANT "</think>"},
    {"an answer continued", "[{\"role\":\"user\",\"content\":\"u\"},{\"role\":\"assistant\",\"content\":\"Par\"}]",
     NULL, CONTINUES, BEGIN USER "u" ASSISTANT "</think>Par"},
    {"JSON values as Python writes them", "[{\"role\":\"system\",\"content\":\"\",\"response_format\":{\"a\":[1E400,"
     "-0,-0.0,0.1,5e-324,1e16,1e15,0.0001,0.00001,2.5E+3,5.960464477539063e-08,1.7976931348623157e308,NaN,-Infinity,"
     "123456789012345678901234567890,\"\\u00e9\\ud83d\\ude00\\u0001\\\"\\\\\\/\\t\",true,null,{},[]],\"b\":{\"c\":"
     "\"d\"}}}]", NULL, 0,
     BEGIN "<｜System｜>\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
     "{\"a\": [Infinity, 0, -0.0, 0.1, 5e-324, 1e+16, 1000000000000000.0, 0.0001, 1e-05, 2500.0, "
     "5.960464477539063e-08, 1.7976931348623157e+308, NaN, -Infinity, 123456789012345678901234567890, "
     "\"\xC3\xA9\xF0\x9F\x98\x80\\u0001\\\"\\\\/\\t\", true, null, {}, []], \"b\": {\"c\": \"d\"}}"},
    {"a reminder, and a task", "[{\"role\":\"latest_reminder\",\"content\":\"R\"},{\"role\":\"user\",\"content\":"
     "\"q\",\"task\":\"query\"}]", NULL, 0, BEGIN "<｜latest_reminder｜>R" USER "q<｜query｜>"},
    {"text and an image", "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":"
     "\"image_url\",\"image_url\":{\"url\":\"x\"}}]}]", NULL, 0,
     BEGIN USER "a\n\n<｜deepseek_image｜>" ASSISTANT "</think>"},
    {"two users merged", "[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"user\",\"content\":\"b\"}]", NULL, 0,
     BEGIN USER "a\n\nb" ASSISTANT "</think>"},
    {"tool results open no turn: the reasoning of the calls is kept", TURNS, NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "q1" ASSISTANT "</think>a1" END USER "q2" ASSISTANT "<think>r2</think>" CALLS("") END
     USER "<tool_result>value</tool_result>" ASSISTANT "<think>"},
    {"the next turn drops it", TURNS_THEN("{\"role\":\"user\",\"content\":\"q3\"}"), NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "q1" ASSISTANT "</think>a1" END USER "q2" ASSISTANT "</think>" CALLS("") END
     USER "<tool_result>value</tool_result>\n\nq3" ASSISTANT "<think>"},
    {"harness multi-step replay, chat", HARNESS_LOOP, NULL, 0,
     BEGIN "<｜System｜>" HARNESS_OLD_DROPPED HARNESS_CHAT_LOOP ASSISTANT "</think>"},
    {"harness multi-step replay, chat with drops", HARNESS_LOOP, NULL, DROPS,
     BEGIN "<｜System｜>" HARNESS_OLD_DROPPED HARNESS_CHAT_LOOP ASSISTANT "</think>"},
    {"harness multi-step replay, thinking kept", HARNESS_LOOP, NULL, THINKING,
     BEGIN EFFORT("75") HARNESS_OLD_KEPT HARNESS_THINKING_LOOP ASSISTANT "<think>"},
    {"harness multi-step replay, only old turn dropped", HARNESS_LOOP, NULL, THINKING | DROPS,
     BEGIN EFFORT("75") HARNESS_OLD_DROPPED HARNESS_THINKING_LOOP ASSISTANT "<think>"},
    {"harness next user, chat", HARNESS_NEXT_USER, NULL, 0,
     BEGIN "<｜System｜>" HARNESS_OLD_DROPPED HARNESS_CHAT_LOOP "\n\nq2" ASSISTANT "</think>"},
    {"harness next user, chat with drops", HARNESS_NEXT_USER, NULL, DROPS,
     BEGIN "<｜System｜>" HARNESS_OLD_DROPPED HARNESS_CHAT_LOOP "\n\nq2" ASSISTANT "</think>"},
    {"harness next user, thinking kept", HARNESS_NEXT_USER, NULL, THINKING,
     BEGIN EFFORT("75") HARNESS_OLD_KEPT HARNESS_THINKING_LOOP "\n\nq2" ASSISTANT "<think>"},
    {"harness next user, all earlier reasoning dropped", HARNESS_NEXT_USER, NULL, THINKING | DROPS,
     BEGIN EFFORT("75") HARNESS_OLD_DROPPED HARNESS_CHAT_LOOP "\n\nq2" ASSISTANT "<think>"},
    {"harness empty replayed reasoning, thinking", "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\","
     "\"content\":\"a\",\"reasoning_content\":\"\"}]", NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "q" ASSISTANT "<think></think>a" END},
    {"harness missing replayed reasoning, thinking", "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\","
     "\"content\":\"a\"}]", NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "q" ASSISTANT "<think></think>a" END},
    {"harness null replayed reasoning, thinking", "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\","
     "\"content\":\"a\",\"reasoning_content\":null}]", NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "q" ASSISTANT "<think></think>a" END},
    {"tool results after an empty user array open no turn", "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":"
     "\"assistant\",\"reasoning_content\":\"r\",\"tool_calls\":[{\"id\":\"c\",\"function\":{\"name\":\"f\","
     "\"arguments\":\"{}\"}}]},{\"role\":\"user\",\"content\":[]},{\"role\":\"tool\",\"tool_call_id\":\"c\","
     "\"content\":\"v\"}]", NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "q" ASSISTANT "<think>r</think>" CALLS("") END USER "<tool_result>v</tool_result>"
     ASSISTANT "<think>"},
    {"an empty user array opens a turn", "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\","
     "\"content\":\"a\",\"reasoning_content\":\"r\"},{\"role\":\"user\",\"content\":[]}]", NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "q" ASSISTANT "</think>a" END USER ASSISTANT "<think>"},
    {"the images of tool results in the order of the calls", "[{\"role\":\"user\",\"content\":[{\"type\":"
     "\"image_url\",\"image_url\":{\"url\":\"h.png\"}}]},{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"a\","
     "\"function\":{\"name\":\"f\",\"arguments\":{}}},{\"id\":\"b\",\"function\":{\"name\":\"f\",\"arguments\":{}}}"
     "]},{\"role\":\"tool\",\"tool_call_id\":\"b\",\"content\":[{\"type\":\"text\",\"text\":\"B\"},{\"type\":"
     "\"image_url\",\"image_url\":{\"url\":\"b.png\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"a\",\"content\":["
     "{\"type\":\"text\",\"text\":\"A\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"a.png\"}}]}]", NULL, 0,
     BEGIN USER IMAGE ASSISTANT "</think>\n\n<" DSML " calls>\n<" DSML " invoke name=\"f\">\n\n</" DSML " invoke>\n<"
     DSML " invoke name=\"f\">\n\n</" DSML " invoke>\n</" DSML " calls>" END USER "<tool_result>A\n\n" IMAGE
     "</tool_result>\n\n<tool_result>B\n\n" IMAGE "</tool_result>" ASSISTANT "</think>"},
    {"no message", " [ ] ", NULL, 0, BEGIN},
    {"invalid JSON", "[{\"role\":\"user\",\"content\":\"a\"}", NULL, 0, NULL},
    {"no array", "{}", NULL, 0, NULL},
    {"text behind the document", "[] x", NULL, 0, NULL},
    {"an escape Python does not read", "[\"\\x41\"]", NULL, 0, NULL},
    {"a control character in a string", "[\"\x01\"]", NULL, 0, NULL},
    {"invalid UTF-8 in a string", "[\"\xFF\"]", NULL, 0, NULL},
    {"a lone high surrogate", "[{\"role\":\"user\",\"content\":\"\\ud800\"}]", NULL, 0, NULL},
    {"a lone low surrogate", "[{\"role\":\"user\",\"content\":\"\\udc00x\"}]", NULL, 0, NULL},
    {"a number with a leading zero", "[01]", NULL, 0, NULL},
    {"a number without digits after its point", "[1.]", NULL, 0, NULL},
    {"a number without digits in its exponent", "[1e]", NULL, 0, NULL},
    {"a word cut short", "[tru]", NULL, 0, NULL},
    {"a comma before a closing bracket", "[{\"role\":\"user\",},]", NULL, 0, NULL},
    {"a role the encoder does not know", "[{\"role\":\"bogus\",\"content\":\"x\"}]", NULL, 0, NULL},
    {"a developer in chat mode", "[{\"role\":\"developer\",\"content\":\"S\"}]", NULL, 0,
     BEGIN "<｜System｜>S"},
    {"an escaped developer role", "[{\"role\":\"devel\\u006fper\",\"content\":\"S\"}]", NULL, THINKING,
     BEGIN EFFORT("75") "S"},
    {"a mid-conversation developer opens an answer", "[{\"role\":\"user\",\"content\":\"Q\"},"
     "{\"role\":\"assistant\",\"content\":\"A\",\"reasoning_content\":\"old\"},"
     "{\"role\":\"developer\",\"content\":\"S\"}]", NULL, THINKING | DROPS,
     BEGIN EFFORT("75") USER "Q" ASSISTANT "</think>A" END "<｜System｜>S" ASSISTANT "<think>"},
    {"no role", "[{\"content\":\"x\"}]", NULL, 0, NULL},
    {"a message that is no object", "[\"hello\"]", NULL, 0, NULL},
    {"the image placeholder in text", "[{\"role\":\"user\",\"content\":\"<｜deepseek_image｜>\"}]", NULL, 0, NULL},
    {"the image placeholder in reasoning", "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\","
     "\"reasoning_content\":\"x<｜deepseek_image｜>\"}]", NULL, 0, NULL},
    {"a user content that is null", "[{\"role\":\"user\",\"content\":null}]", NULL, 0, NULL},
    {"a content that is a number", "[{\"role\":\"system\",\"content\":5}]", NULL, 0, NULL},
    {"a task the encoder does not know", "[{\"role\":\"user\",\"content\":\"q\",\"task\":\"bogus\"}]", NULL, 0, NULL},
    {"a call without its function", "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\","
     "\"tool_calls\":[{\"id\":\"x\"}]}]", NULL, 0, NULL},
    {"a call without arguments", "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\","
     "\"tool_calls\":[{\"function\":{\"name\":\"f\"}}]}]", NULL, 0, NULL},
    {"a name with two separators", "[{\"role\":\"system\",\"content\":\"s\"}]",
     "[{\"type\":\"function\",\"function\":{\"name\":\"a::b::c\"}}]", 0, NULL},
    {"a tool namespace, refused by contract", "[{\"role\":\"system\",\"content\":\"s\"}]",
     "[{\"type\":\"function\",\"namespace\":\"n\",\"function\":{\"name\":\"f\"}}]", 0, NULL},
    {"tools that are no array", "[{\"role\":\"system\",\"content\":\"s\"}]", "{\"a\": 1}", 0, NULL},
    {"tools on no message", "[]", "[]", 0, NULL},
    {"tools that are no JSON", "[{\"role\":\"system\",\"content\":\"s\"}]", "[", 0, NULL},
    {"bad tools on a user message are still checked", "[{\"role\":\"user\",\"content\":\"q\"}]", "[5]", 0, NULL},
    {"a content part of another kind, refused by contract", "[{\"role\":\"user\",\"content\":[{\"type\":"
     "\"input_audio\"}]}]", NULL, 0, NULL},
    {"an image without its source", "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\"}]}]", NULL, 0, NULL},
    {"an answer continued that is no answer", "[{\"role\":\"user\",\"content\":\"u\"}]", NULL, CONTINUES, NULL},
    {"tool results sorted by the calls of the assistant before, also after another answer",
     "[{\"role\":\"user\",\"content\":\"u\"},{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"a\",\"function\":"
     "{\"name\":\"f\",\"arguments\":{}}},{\"function\":{\"id\":\"b\",\"name\":\"f\",\"arguments\":{}}},{\"id\":\"a\","
     "\"function\":{\"name\":\"f\",\"arguments\":{}}}]},{\"role\":\"tool\",\"tool_call_id\":\"a\",\"content\":\"A\"},"
     "{\"role\":\"tool\",\"tool_call_id\":\"b\",\"content\":\"B\"},{\"role\":\"assistant\",\"content\":\"x\"},"
     "{\"role\":\"tool\",\"tool_call_id\":\"a\",\"content\":\"A2\"},{\"role\":\"tool\",\"content\":\"N\"},"
     "{\"role\":\"tool\",\"tool_call_id\":\"b\",\"content\":\"B2\"}]", NULL, 0,
     BEGIN USER "u" ASSISTANT "</think>\n\n<" DSML " calls>\n<" DSML " invoke name=\"f\">\n\n</" DSML " invoke>\n<" DSML
     " invoke name=\"f\">\n\n</" DSML " invoke>\n<" DSML " invoke name=\"f\">\n\n</" DSML " invoke>\n</" DSML " calls>"
     END USER "<tool_result>B</tool_result>\n\n<tool_result>A</tool_result>" ASSISTANT "</think>x" END USER
     "<tool_result>N</tool_result>\n\n<tool_result>B2</tool_result>\n\n<tool_result>A2</tool_result>" ASSISTANT
     "</think>"},
    {"a call's id that is falsy gives way to its function's", "[{\"role\":\"assistant\",\"tool_calls\":[{\"id\":[],"
     "\"function\":{\"id\":\"b\",\"name\":\"f\",\"arguments\":\"[\\\"\\\\ud800\\\"]\"}}]}]", NULL, 0,
     BEGIN CALLS("<" DSML " parameter name=\"arguments\" string=\"true\">[\"\\ud800\"]</" DSML " parameter>") END},
    {"calls that are no array", "[{\"role\":\"assistant\",\"tool_calls\":\"[{\\\"id\\\": \\\"x\\\"}]\"}]", NULL, 0,
     NULL},
    {"a call's id that is a list", "[{\"role\":\"assistant\",\"tool_calls\":[{\"id\":[1],\"function\":{\"name\":"
     "\"f\",\"arguments\":{}}}]}]", NULL, 0, NULL},
    {"a call's id that is a number, refused by contract", "[{\"role\":\"assistant\",\"tool_calls\":[{\"id\":7,"
     "\"function\":{\"name\":\"f\",\"arguments\":{}}}]}]", NULL, 0, NULL},
    {"a key twice in a call's arguments string, refused by contract", ARGUMENTS_OPENING
     "{\\\"a\\\": 1, \\\"a\\\": 2}" ARGUMENTS_CLOSING, NULL, 0, NULL},
    {"a key twice inside a parameter, refused by contract", "[{\"role\":\"assistant\",\"tool_calls\":[{\"function\":"
     "{\"name\":\"f\",\"arguments\":{\"b\":{\"c\":1,\"\\u0063\":2}}}}]}]", NULL, 0, NULL},
    {"a key twice in a tool, refused by contract", "[{\"role\":\"system\",\"content\":\"s\"}]",
     "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"name\":\"g\"}}]", 0, NULL},
    {"a key twice in a response format, refused by contract", FORMAT_OPENING "{\"a\":[{\"b\":1}],\"a\":2}"
     FORMAT_CLOSING, NULL, 0, NULL},
    {"keys alike in objects of their own", FORMAT_OPENING "{\"a\":{\"a\":{\"a\":1}},\"b\":[{\"a\":2},{\"a\":3}]}"
     FORMAT_CLOSING, NULL, 0, BEGIN "<｜System｜>s\n\n## Response Format:\n\nYou MUST strictly adhere to the following "
     "schema to reply:\n{\"a\": {\"a\": {\"a\": 1}}, \"b\": [{\"a\": 2}, {\"a\": 3}]}"},
    {"a lone half of a pair in a call's arguments object", ARGUMENTS_OPENING "{\\\"s\\\": \\\"\\\\ud800\\\"}"
     ARGUMENTS_CLOSING, NULL, 0, NULL},
    {"the encoder's own member content_blocks, refused by contract", "[{\"role\":\"user\",\"content\":\"u\","
     "\"content_blocks\":[]}]", NULL, 0, NULL},
    {"the encoder's own member wo_eos, refused by contract", "[{\"role\":\"user\",\"content\":\"u\"},"
     "{\"role\":\"assistant\",\"content\":\"a\",\"wo_eos\":false}]", NULL, 0, NULL}};

static int failures = 0;
static uint64_t random_state = RANDOM_SEED;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static uint64_t next_random(void)
{
    uint64_t value = (random_state += SPLITMIX_INCREMENT);
    value = (value ^ (value >> SPLITMIX_FIRST_SHIFT)) * SPLITMIX_FIRST_MULTIPLIER;
    value = (value ^ (value >> SPLITMIX_SECOND_SHIFT)) * SPLITMIX_SECOND_MULTIPLIER;
    return value ^ (value >> SPLITMIX_THIRD_SHIFT);
}

static struct chat_template_options options_of(unsigned flags)
{
	return (struct chat_template_options){ DEFAULT_EFFORT,
					       (flags & THINKING) != 0,
					       (flags & DROPS) != 0,
					       (flags & CONTINUES) != 0 };
}

static MUST_CHECK result_status render(const char *messages,
				       size_t messages_bytes, const char *tools,
				       size_t tools_bytes, unsigned flags,
				       char *prompt, size_t capacity,
				       size_t *length)
{
	const struct chat_template_options options = options_of(flags);
	return chat_template_render(messages, messages_bytes, tools,
				    tools_bytes, &options, prompt, capacity,
				    length);
}

static void test_prompt_cases(void)
{
    static char prompt[PROMPT_CAPACITY];
    char what[WHAT_CAPACITY] = {0};
    for (size_t index = 0; index < sizeof(prompt_cases) / sizeof(prompt_cases[0]); index++) {
        const struct prompt_case *const prompt_case = &prompt_cases[index];
        size_t length = 0;
	const result_status status = render(
		prompt_case->messages, strlen(prompt_case->messages),
		prompt_case->tools,
		prompt_case->tools != NULL ? strlen(prompt_case->tools) : 0,
		prompt_case->flags, prompt, PROMPT_CAPACITY, &length);
	(void)snprintf(what, sizeof(what), "%s: %s", prompt_case->prompt != NULL ? "the prompt" : "a refusal",
                       prompt_case->what);
	expect(prompt_case->prompt == NULL ?
		       status == INVALID :
		       status == STATUS_OK &&
			       length == strlen(prompt_case->prompt) &&
			       memcmp(prompt, prompt_case->prompt, length) == 0,
	       what);
	if (prompt_case->prompt != NULL && length > 0 &&
	    status == STATUS_OK) { /* the capacity is exact */
		size_t shorter = 0;
		expect(render(prompt_case->messages,
			      strlen(prompt_case->messages), prompt_case->tools,
			      prompt_case->tools != NULL ?
				      strlen(prompt_case->tools) :
				      0,
			      prompt_case->flags, prompt, length - 1,
			      &shorter) == UNSUPPORTED &&
			       shorter == 0 &&
			       render(prompt_case->messages,
				      strlen(prompt_case->messages),
				      prompt_case->tools,
				      prompt_case->tools != NULL ?
					      strlen(prompt_case->tools) :
					      0,
				      prompt_case->flags, prompt, length,
				      &shorter) == STATUS_OK &&
			       shorter == length,
		       "a prompt one byte longer than the capacity is unsupported, one as long is not");
	} else if (prompt_case->prompt == NULL) {
		expect(render(prompt_case->messages,
			      strlen(prompt_case->messages), prompt_case->tools,
			      prompt_case->tools != NULL ?
				      strlen(prompt_case->tools) :
				      0,
			      prompt_case->flags, prompt, 0,
			      &length) == INVALID,
		       "a request refused is refused whatever the capacity");
	}
    }
}

/* The API alias follows every system branch, including a later turn's reasoning boundary. */
static void test_developer_role(void)
{
    static const struct { const char *before, *after; } cases[] = {
        {"[{\"role\":\"", "\",\"content\":\"S\"}]"},
        {"[{\"role\":\"", "\",\"content\":null},{\"role\":\"user\",\"content\":\"Q\"}]"},
        {"[{\"role\":\"", "\"},{\"role\":\"user\",\"content\":\"Q\"}]"},
        {"[{\"role\":\"", "\",\"content\":[{\"type\":\"text\",\"text\":\"S\\n\\u00e9\"}],"
         "\"response_format\":{\"type\":\"json_object\"}}]"},
        {"[{\"role\":\"", "\",\"content\":\"S\",\"tools\":[{\"function\":{\"name\":\"local\"}}]},"
         "{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"assistant\",\"content\":\"A\"}]"},
        {"[{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"assistant\",\"content\":\"A\","
         "\"reasoning_content\":\"old\"},{\"role\":\"", "\",\"content\":\"later\"}]"},
        {"[{\"role\":\"system\",\"content\":\"first\"},{\"role\":\"", "\",\"content\":\"second\"},"
         "{\"role\":\"assistant\",\"content\":\"A\",\"reasoning_content\":\"R\"}]"},
        {"[{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"", "\",\"content\":\"later\","
         "\"tools\":[{\"function\":{\"name\":\"local\"}}],\"response_format\":{\"type\":\"json_object\"}},"
         "{\"role\":\"user\",\"content\":\"next\"}]"}
    };
    static const char *const tools[] = {NULL, "[]", "[{\"function\":{\"name\":\"request\"}}]"};
    static const unsigned modes[] = {0, THINKING, DROPS, THINKING | DROPS, CONTINUES, THINKING | CONTINUES,
                                    DROPS | CONTINUES, THINKING | DROPS | CONTINUES};
    static char system_messages[PROMPT_CAPACITY], developer_messages[PROMPT_CAPACITY];
    static char expected[PROMPT_CAPACITY], actual[PROMPT_CAPACITY];
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        const int system_bytes = snprintf(system_messages, sizeof(system_messages), "%s%s%s",
                                           cases[index].before, "system", cases[index].after);
        const int developer_bytes = snprintf(developer_messages, sizeof(developer_messages), "%s%s%s",
                                              cases[index].before, "developer", cases[index].after);
        if (system_bytes <= 0 || (size_t)system_bytes >= sizeof(system_messages) || developer_bytes <= 0 ||
            (size_t)developer_bytes >= sizeof(developer_messages)) {
            expect(false, "role comparison messages fit");
            continue;
        }
        for (size_t tool = 0; tool < sizeof(tools) / sizeof(tools[0]); tool++) {
            for (size_t mode = 0; mode < sizeof(modes) / sizeof(modes[0]); mode++) {
                size_t expected_bytes = 0, actual_bytes = 0;
                const size_t tool_bytes = tools[tool] != NULL ? strlen(tools[tool]) : 0;
		const result_status system_status =
			render(system_messages, (size_t)system_bytes,
			       tools[tool], tool_bytes, modes[mode], expected,
			       sizeof(expected), &expected_bytes);
		const result_status developer_status =
			render(developer_messages, (size_t)developer_bytes,
			       tools[tool], tool_bytes, modes[mode], actual,
			       sizeof(actual), &actual_bytes);
		expect(system_status == developer_status &&
			       expected_bytes == actual_bytes &&
			       (system_status == INVALID ||
				(system_status == STATUS_OK &&
				 memcmp(expected, actual, expected_bytes) == 0)),
		       "developer equals system for first and later messages, tools, thinking and continuation");
	    }
        }
    }
}

/* A tool's schema written again as Python would, inside the encoder's text of tools. */
static void test_tools(void)
{
    static char prompt[PROMPT_CAPACITY];
    static const char messages[] = "[{\"role\":\"system\",\"content\":\"S\"},{\"role\":\"user\",\"content\":\"U\"}]";
    static const char tools[] = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":"
                                "\"object\",\"minimum\":0.50}}}]";
    static const char schema[] = "### Available Tool Schemas\n\n{\"name\": \"f\", \"parameters\": {\"type\": "
                                 "\"object\", \"minimum\": 0.5}}\n\nYou MUST strictly follow";
    size_t length = 0;
    expect(render(messages, strlen(messages), tools, strlen(tools), 0, prompt,
		  PROMPT_CAPACITY, &length) == STATUS_OK &&
		   memmem(prompt, length, schema, strlen(schema)) != NULL &&
		   memcmp(prompt + length -
				  strlen(USER "U" ASSISTANT "</think>"),
			  USER "U" ASSISTANT "</think>",
			  strlen(USER "U" ASSISTANT "</think>")) == 0,
	   "a tool's schema in the system message");
}

/* Request tools with no system message are exactly an explicit empty-system prepend, in every mode. */
static void test_tools_without_system(void)
{
    static const char tools[] = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{}}}]";
    static const char *const messages[] = {
        "[{\"role\":\"user\",\"content\":\"U\"}]",
        "[{\"role\":\"user\",\"content\":\"U\"},{\"role\":\"user\",\"content\":\"V\"}]",
        "[{\"role\":\"assistant\",\"content\":\"A\",\"reasoning_content\":\"R\"}]",
        "[{\"role\":\"tool\",\"content\":\"result\"}]",
        "[{\"role\":\"latest_reminder\",\"content\":\"reminder\"}]",
        TURNS,
        TURNS_THEN("{\"role\":\"user\",\"content\":\"next\"}"),
        "[{\"role\":\"user\",\"content\":\"U\"},{\"role\":\"system\",\"content\":\"later\"},"
        "{\"role\":\"user\",\"content\":\"V\"}]"
    };
    static const unsigned modes[] = {0, THINKING, DROPS, THINKING | DROPS};
    static char actual[PROMPT_CAPACITY], expected[PROMPT_CAPACITY], explicit_messages[PROMPT_CAPACITY];
    for (size_t message = 0; message < sizeof(messages) / sizeof(messages[0]); message++) {
        const int written = snprintf(explicit_messages, sizeof(explicit_messages),
            "[{\"role\":\"system\",\"content\":\"\"},%s", messages[message] + 1);
        expect(written > 0 && (size_t)written < sizeof(explicit_messages), "explicit prepend fits");
        for (size_t mode = 0; mode < sizeof(modes) / sizeof(modes[0]); mode++) {
            size_t actual_bytes = 0, expected_bytes = 0;
	    expect(render(messages[message], strlen(messages[message]), tools,
			  strlen(tools), modes[mode], actual, sizeof(actual),
			  &actual_bytes) == STATUS_OK &&
			   render(explicit_messages, strlen(explicit_messages),
				  tools, strlen(tools), modes[mode], expected,
				  sizeof(expected),
				  &expected_bytes) == STATUS_OK &&
			   actual_bytes == expected_bytes &&
			   memcmp(actual, expected, actual_bytes) == 0,
		   "tools without first system equal the explicit prepend byte for byte");
	    expect(render(messages[message], strlen(messages[message]), "[]",
			  strlen("[]"), modes[mode], actual, sizeof(actual),
			  &actual_bytes) == STATUS_OK &&
			   render(messages[message], strlen(messages[message]),
				  NULL, 0, modes[mode], expected,
				  sizeof(expected),
				  &expected_bytes) == STATUS_OK &&
			   actual_bytes == expected_bytes &&
			   memcmp(actual, expected, actual_bytes) == 0,
		   "empty request tools remain the no-tools conversation");
	}
    }
}

/* The reference keeps all assistant reasoning when any message declares tools, even with drops requested. */
static void test_harness_declared_tools(void)
{
    static char kept[PROMPT_CAPACITY], dropped[PROMPT_CAPACITY];
    static const char tools[] = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":"
                                "\"object\"}}}]";
    static const char *const messages[] = {HARNESS_LOOP, HARNESS_NEXT_USER};
    static const unsigned modes[] = {0, THINKING};
    static const char old_thought[] = "<think>old</think>a0";
    static const char empty_thought[] = "q1" ASSISTANT "<think></think>";
    static const char later_thought[] = "<think>  r2\n\xc3\xa9  </think>checking";
    for (size_t message = 0; message < sizeof(messages) / sizeof(messages[0]); message++) {
        for (size_t mode = 0; mode < sizeof(modes) / sizeof(modes[0]); mode++) {
            size_t kept_length = 0, dropped_length = 0;
	    const result_status kept_status =
		    render(messages[message], strlen(messages[message]), tools,
			   strlen(tools), modes[mode], kept, sizeof(kept),
			   &kept_length);
	    const result_status dropped_status =
		    render(messages[message], strlen(messages[message]), tools,
			   strlen(tools), modes[mode] | DROPS, dropped,
			   sizeof(dropped), &dropped_length);
	    expect(kept_status == STATUS_OK && dropped_status == STATUS_OK &&
			   kept_length == dropped_length &&
			   memcmp(kept, dropped, kept_length) == 0,
		   "harness tools override drops_thinking in both modes and across user turns");
	    if ((modes[mode] & THINKING) != 0) {
                expect(memmem(kept, kept_length, old_thought, strlen(old_thought)) != NULL &&
                       memmem(kept, kept_length, empty_thought, strlen(empty_thought)) != NULL &&
                       memmem(kept, kept_length, later_thought, strlen(later_thought)) != NULL,
                       "harness tools keep old, empty and whitespace-preserving reasoning byte for byte");
            }
        }
    }
}

/* Messages of opening, before, a run of bytes, after and closing: rendered or not. */
static bool renders_run(const char *opening, const char *before, char repeated, size_t count, const char *after,
                        const char *closing)
{
    static char messages[LONG_JSON_CAPACITY], prompt[LONG_JSON_CAPACITY];
    size_t length = 0, used = (size_t)snprintf(messages, sizeof(messages), "%s%s", opening, before);
    memset(messages + used, repeated, count);
    used += count;
    used += (size_t)snprintf(messages + used, sizeof(messages) - used, "%s%s", after, closing);
    return render(messages, used, NULL, 0, 0, prompt, sizeof(prompt),
		  &length) == STATUS_OK;
}

/* A system message whose response format is a run of bytes, around it what makes it a value: rendered or not. */
static bool renders_format(const char *before, char repeated, size_t count, const char *after)
{
    return renders_run(FORMAT_OPENING, before, repeated, count, after, FORMAT_CLOSING);
}

/* An assistant message of calls, or a user message and tool results, count of them: rendered or not. */
static bool renders_many(bool is_calls, int count)
{
    static char messages[LONG_JSON_CAPACITY], prompt[LONG_JSON_CAPACITY];
    size_t length = 0, used = (size_t)snprintf(messages, sizeof(messages), is_calls ? "[{\"role\":\"assistant\","
                                               "\"tool_calls\":[" : "[{\"role\":\"user\",\"content\":\"u\"}");
    for (int index = 0; index < count; index++) {
        used += (size_t)snprintf(messages + used, sizeof(messages) - used, is_calls ? "%s{\"function\":{\"name\":"
                                 "\"f\",\"arguments\":{}}}" : "%s{\"role\":\"tool\",\"content\":\"r\"}",
                                 is_calls && index == 0 ? "" : ",");
    }
    used += (size_t)snprintf(messages + used, sizeof(messages) - used, is_calls ? "]}]" : "]");
    return render(messages, used, NULL, 0, 0, prompt, sizeof(prompt),
		  &length) == STATUS_OK;
}

static void test_arguments_and_limits(void)
{
    static char prompt[PROMPT_CAPACITY];
    size_t length = 0;
    enum { MOST_ARRAYS = JSON_MAXIMUM_DEPTH - ARRAYS_AROUND_A_FORMAT };
    const struct chat_template_options effort_too_low = { LOWEST_EFFORT - 1,
							  true, true, false };
    const struct chat_template_options effort_too_high = { HIGHEST_EFFORT + 1,
							   true, true, false };
    const char *const one = "[{\"role\":\"user\",\"content\":\"u\"}]";
    expect(chat_template_render(one, strlen(one), NULL, 0, &effort_too_low,
				prompt, PROMPT_CAPACITY, &length) == INVALID &&
		   chat_template_render(one, strlen(one), NULL, 0,
					&effort_too_high, prompt,
					PROMPT_CAPACITY, &length) == INVALID,
	   "a reasoning effort outside 1 to 100");
    expect(render(NULL, 0, NULL, 0, 0, prompt, PROMPT_CAPACITY, &length) ==
			   INVALID &&
		   render(one, strlen(one), NULL, 1, 0, prompt, PROMPT_CAPACITY,
			  &length) == INVALID &&
		   render(one, strlen(one), NULL, 0, 0, NULL, PROMPT_CAPACITY,
			  &length) == INVALID &&
		   render(one, strlen(one), NULL, 0, 0, prompt, PROMPT_CAPACITY,
			  NULL) == INVALID &&
		   chat_template_render(one, strlen(one), NULL, 0, NULL, prompt,
					PROMPT_CAPACITY, &length) == INVALID,
	   "a NULL argument");
    char closing[JSON_MAXIMUM_DEPTH + 1] = {
	    0
    }; /* as many closing brackets as json.h allows */
    memset(closing, ']', JSON_MAXIMUM_DEPTH);
    expect(renders_format("", '[', MOST_ARRAYS, closing + ARRAYS_AROUND_A_FORMAT) &&
           !renders_format("", '[', MOST_ARRAYS + 1, closing + ARRAYS_AROUND_A_FORMAT - 1),
           "arrays nested as deep as json.h allows, and one more");
    expect(renders_run(ARGUMENTS_OPENING, "{\\\"d\\\": ", '[',
		       JSON_MAXIMUM_DEPTH - 1, closing + 1,
		       "}" ARGUMENTS_CLOSING) &&
		   !renders_run(ARGUMENTS_OPENING, "{\\\"d\\\": ", '[',
				JSON_MAXIMUM_DEPTH, closing,
				"}" ARGUMENTS_CLOSING) &&
		   renders_run(ARGUMENTS_OPENING, "", '[',
			       JSON_MAXIMUM_DEPTH + 1, "", ARGUMENTS_CLOSING),
	   "an object of arguments nested as deep as json.h allows and one more, and deeper arrays as they are");
    expect(renders_format("", '7', JSON_MOST_INTEGER_DIGITS, "") &&
		   !renders_format("", '7', JSON_MOST_INTEGER_DIGITS + 1, "") &&
		   renders_format("0.", '7', JSON_MOST_INTEGER_DIGITS + 1, ""),
	   "an integer of as many digits as Python's int() reads and one more, and a float of more");
    expect(renders_many(true, CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS) &&
		   !renders_many(true, CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS + 1),
	   "as many calls as the limit, and one more");
    expect(renders_many(false, CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS) &&
		   !renders_many(false, CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS + 1),
	   "as many tool results in a merged user message as the limit, and one more");
#if CHECKING_HOOKS_ENABLED
    expect(fault_arm("chat_template_memory", FAULT_ON_FIRST_CALL),
	   "a lack of memory is armed");
    expect(render(one, strlen(one), NULL, 0, 0, prompt, PROMPT_CAPACITY,
		  &length) == STATUS_OUT_OF_MEMORY &&
		   fault_disarm() == 1,
	   "a render that lacks memory says so");
#endif
}

/* A random change: a byte set to anything, one of the characters of JSON put in, a byte taken out, a run of bytes
 * written twice, the rest cut off. */
static size_t mutated(char *text, size_t length, size_t capacity)
{
    static const char inserted[] = "{}[],:\"\\0123456789-+.eEntfuaIN \t\xFF\xC3\xED\xA0";
    const size_t place = length > 0 ? (size_t)(next_random() % length) : 0;
    switch ((enum mutation)(next_random() % MUTATION_COUNT)) {
    case FLIP:
        if (length > 0) {
            text[place] = (char)next_random();
        }
        return length;
    case INSERT:
        if (length < capacity) {
            memmove(text + place + 1, text + place, length - place);
            text[place] = inserted[next_random() % (sizeof(inserted) - 1)];
            return length + 1;
        }
        return length;
    case DELETE:
        if (length > 0) {
            memmove(text + place, text + place + 1, length - place - 1);
            return length - 1;
        }
        return length;
    case REPEAT: {
        const size_t run = length > place ? (size_t)(next_random() % (length - place)) : 0;
        if (length + run <= capacity) {
            memmove(text + place + run, text + place, length - place);
            return length + run;
        }
        return length;
    }
    case CUT:
    case MUTATION_COUNT:
        return place;
    }
    return length;
}

static void test_fuzz(void)
{
    static char text[LONG_JSON_CAPACITY], prompt[PROMPT_CAPACITY];
    static const char tools[] = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"x\":[1.5,"
                                "\"y\",null]}}}]";
    int rendered = 0, misbehaved = 0;
    for (int round = 0; round < FUZZ_ROUNDS; round++) {
        const struct prompt_case *const seed = &prompt_cases[next_random() % (sizeof(prompt_cases) /
                                                                             sizeof(prompt_cases[0]))];
        size_t length = strlen(seed->messages), prompt_length = 0;
        memcpy(text, seed->messages, length);
        for (uint64_t mutation = 1 + next_random() % MOST_MUTATIONS; mutation > 0; mutation--) {
            length = mutated(text, length, sizeof(text));
        }
        const size_t capacity = (size_t)(next_random() % PROMPT_CAPACITY);
        const bool has_tools = (next_random() & 1u) == 0;
	const result_status status =
		render(text, length, has_tools ? tools : NULL,
		       has_tools ? strlen(tools) : 0, (unsigned)next_random(),
		       prompt, capacity, &prompt_length);
	rendered += status == STATUS_OK ? 1 : 0;
	misbehaved += (status == STATUS_OK && prompt_length <= capacity) ||
				      status == INVALID ||
				      status == UNSUPPORTED ?
			      0 :
			      1;
    }
    expect(misbehaved == 0 && rendered > 0, "a changed conversation renders within its capacity or is refused");
}

int main(void)
{
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    test_prompt_cases();
    test_developer_role();
    test_tools();
    test_tools_without_system();
    test_harness_declared_tools();
    test_arguments_and_limits();
    test_fuzz();
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf(
	    "chat_template_test passed (%zu cases, %d changed conversations, checking hooks %s)\n",
	    sizeof(prompt_cases) / sizeof(prompt_cases[0]), FUZZ_ROUNDS,
	    CHECKING_HOOKS_ENABLED ? "enabled" : "compiled out");
    return EXIT_SUCCESS;
}
