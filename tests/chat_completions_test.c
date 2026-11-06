/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * chat_completions_test: source/chat_completions.c against chat_completions.h. A table of request bodies, each taken
 * or refused with its status and the member it names (every refusal of the contract, every limit, JSON json.loads
 * refuses or this reader does not take); the values a request sets and the defaults it keeps. Answers, byte for byte:
 * whole and streamed, with the usage; reasoning split from content, the end of reasoning and a stop string cut
 * across tokens, stop strings looked for in the content only; UTF-8 held back across tokens, ill-formed bytes as
 * U+FFFD, control characters escaped; the end token counted and never written; a failure. Refusals with the head
 * each status asks for, the listings, the buffer's limit, and the fault point chat_memory. A seed: the request's, the
 * server's, or drawn afresh for each request where neither sets one, and the fault point chat_random.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/chat_completions.h"
#include "anemone/hooks.h"
#include "json.h"

#define MODEL "test-model"
#define CREATED 1700000000
#define IDENTIFIER "chatcmpl-1"
#define PROMPT_TOKENS 3u
#define DEFAULT_SEED 7u
#define REQUEST_SEED 9u
#define DEFAULT_MAX_TOKENS 100u
#define DEFAULT_TEMPERATURE 0.5f
#define DEFAULT_TOP_P 0.875f
#define SET_TEMPERATURE 0.25f
#define SET_TOP_P 0.5f
#define SET_MAX_TOKENS 5u
#define HIGH_EFFORT 75u
#define LOW_EFFORT 50u
#define MAX_EFFORT 100u
#define FORTY_TWO 42u
#define TEXT_CAPACITY 65536u
#define BODY_CAPACITY 1048576u
#define MESSAGE_TEXT "{\"role\":\"user\",\"content\":\"hi\"},"
#define DEEP_OPENING "{\"a\":["
#define STOP_TWO_BYTES 2u
#define ACCENTED_STOP_BYTES 3u
#define TWO_STOPS 2u
#define THIRD_TOKEN 2u
#define FOURTH_TOKEN 3u
#define REASONING_PHASE_TOKENS 2u
#define GENERATED_BEFORE_STOP 2u
#define LISTED_MODELS_WITH_ALIAS 2u
#define OK_STATUS 0u
#define BAD HTTP_BAD_REQUEST
#define NOT_FOUND HTTP_NOT_FOUND
#define CHUNK "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":" \
              "\"test-model\",\"choices\":["

static int failures = 0;
static const char *context_text = "";
static const struct chat_request defaults = { .seed = DEFAULT_SEED,
					      .max_tokens = DEFAULT_MAX_TOKENS,
					      .temperature =
						      DEFAULT_TEMPERATURE,
					      .top_p = DEFAULT_TOP_P,
					      .is_seeded = true };

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static uint32_t read_request(const char *body, size_t byte_count,
			     struct chat_request *request,
			     struct chat_refusal *refusal)
{
	*refusal = (struct chat_refusal){ 0 };
	return chat_request_read(body, byte_count, MODEL, NULL, &defaults,
				 request, refusal);
}

static void test_request_table(void)
{
    struct request_case { const char *body; uint32_t status; const char *member; };
    static const struct request_case cases[] = {
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", OK_STATUS, NULL},
        {"{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", OK_STATUS, NULL},
        {"{\"model\":\"other\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", NOT_FOUND, "\"model\""},
        {"{\"model\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", NOT_FOUND, "\"model\""},
        {"{}", BAD, NULL},
        {"{\"messages\":null}", BAD, NULL},
        {"{\"messages\":[]}", BAD, "\"messages\""},
        {"{\"messages\":\"hi\"}", BAD, "\"messages\""},
        {"{\"messages\":[\"hi\"]}", BAD, "\"messages\""},
        {"{\"messages\":[{\"role\":\"tool\",\"content\":\"x\",\"tool_call_id\":\"a\"}]}", OK_STATUS, NULL},
        {"{\"messages\":[{\"role\":\"developer\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"messages\":[{\"role\":\"devel\\u006fper\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"a\"},"
         "{\"role\":\"developer\",\"content\":\"later\"}]}", OK_STATUS, NULL},
        {"{\"messages\":[{\"role\":\"bogus\",\"content\":\"x\"}]}", BAD, "\"role\""},
        {"{\"messages\":[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"a\"}]}]}",
         OK_STATUS, NULL},
        {"{\"messages\":[{\"role\":\"assistant\",\"content\":\"x\",\"tool_calls\":[]}]}", OK_STATUS, NULL},
        {"{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"}]}]}", OK_STATUS, NULL},
        {"{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}]}",
         BAD, "\"content\""},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"x\",\"name\":\"me\"}]}", OK_STATUS, NULL},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"x\",\"audio\":{\"id\":\"a\"}}]}", BAD, "\"audio\""},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"x\",\"function_call\":{\"name\":\"f\"}}]}", BAD,
         "\"function_call\""},
        {"{\"stream\":\"yes\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"stream\""},
        {"{\"stream_options\":{\"include_usage\":1},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"stream_options\""},
        {"{\"stream_options\":{\"other\":true},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"stream_options\""},
        {"{\"stream_options\":{\"other\":false},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
         BAD, "\"stream_options\""},
        {"{\"max_tokens\":0,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"max_tokens\""},
        {"{\"max_tokens\":-1,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"max_tokens\""},
        {"{\"max_tokens\":1.5,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"max_tokens\""},
        {"{\"max_tokens\":1e2,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"max_tokens\""},
        {"{\"max_completion_tokens\":4294967296,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"max_completion_tokens\""},
        {"{\"max_tokens\":4294967295,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"temperature\":2.5,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"temperature\""},
        {"{\"temperature\":-0.1,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"temperature\""},
        {"{\"temperature\":\"1\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"temperature\""},
        {"{\"temperature\":NaN,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"temperature\""},
        {"{\"temperature\":-Infinity,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"temperature\""},
        {"{\"temperature\":2,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"top_p\":0,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"top_p\""},
        {"{\"top_p\":1e-50,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"top_p\""},
        {"{\"top_p\":1.5,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"top_p\""},
        {"{\"top_p\":1,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"seed\":-1,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"seed\""},
        {"{\"seed\":18446744073709551616,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"seed\""},
        {"{\"stop\":\"\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"stop\""},
        {"{\"stop\":[1],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"stop\""},
        {"{\"stop\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"stop\""},
        {"{\"stop\":[\"a\",\"b\",\"c\",\"d\",\"e\"],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"stop\""},
        {"{\"stop\":[\"a\",\"b\",\"c\",\"d\"],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"n\":2,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"n\""},
        {"{\"n\":1,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"reasoning_effort\":\"turbo\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"reasoning_effort\""},
        {"{\"reasoning_effort\":\"medium\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS,
         NULL},
        {"{\"reasoning_effort\":101,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"reasoning_effort\""},
        {"{\"reasoning_effort\":-1,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"reasoning_effort\""},
        {"{\"anemone_draft_usage\":\"compact\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"anemone_draft_usage\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"anemone_draft_usage\":\"full\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"anemone_draft_usage\":\"f\\u0075ll\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
         OK_STATUS, NULL},
        {"{\"anemone_draft_usage\":\"Full\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"anemone_draft_usage\""},
        {"{\"anemone_draft_usage\":\"\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"anemone_draft_usage\""},
        {"{\"anemone_draft_usage\":0,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"anemone_draft_usage\""},
        {"{\"anemone_draft_usage\":false,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"anemone_draft_usage\""},
        {"{\"anemone_draft_usage\":[],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"anemone_draft_usage\""},
        {"{\"anemone_draft_usage\":{},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"anemone_draft_usage\""},
        {"{\"tools\":[],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"tools\":[{\"type\":\"function\"}],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
         OK_STATUS, NULL},
        {"{\"tools\":{},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"tools\""},
        {"{\"functions\":[{\"name\":\"f\"}],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"functions\""},
        {"{\"functions\":[{}],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"functions\""},
        {"{\"tool_choice\":\"required\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"tool_choice\""},
        {"{\"tool_choice\":{\"type\":\"function\"},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"tool_choice\""},
        {"{\"tool_choice\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", OK_STATUS, NULL},
        {"{\"response_format\":{\"type\":\"json_object\"},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"response_format\""},
        {"{\"response_format\":{\"type\":\"text\"},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
         OK_STATUS, NULL},
        {"{\"logprobs\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD, "\"logprobs\""},
        {"{\"logprobs\":false,\"presence_penalty\":0,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
         OK_STATUS, NULL},
        {"{\"presence_penalty\":0.5,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", BAD,
         "\"presence_penalty\""},
        {"{\"user\":\"u\",\"metadata\":{\"a\":\"b\"},\"store\":true,\"messages\":[{\"role\":\"user\",\"content\":"
         "\"x\"}]}", BAD, "\"store\""},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"messages\":[{\"role\":\"user\",\"content\":\"y\"}]}",
         BAD, NULL},
        {"{\"messages\":[{\"role\":\"user\",\"role\":\"user\",\"content\":\"y\"}]}", BAD, NULL},
        {"{\"messages\":", BAD, NULL},
        {"[]", BAD, NULL},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]} x", BAD, NULL},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"\xff\"}]}", BAD, NULL},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"\\ud800\"}]}", BAD, NULL},
        {"{\"temperature\":null,\"max_tokens\":null,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
         OK_STATUS, NULL}};
    for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        const struct request_case *const test_case = &cases[index];
	struct chat_request request = { 0 };
	struct chat_refusal refusal = { 0 };
	context_text = test_case->body;
        const uint32_t status = read_request(test_case->body, strlen(test_case->body), &request, &refusal);
        const size_t member_bytes = refusal.member_end - refusal.member_begin;
        expect(status == test_case->status && (status == 0 || refusal.status == status), "the status");
        expect(status == 0 || (test_case->member == NULL ? member_bytes == 0
                                                         : member_bytes == strlen(test_case->member) &&
                                                           memcmp(test_case->body + refusal.member_begin,
                                                                  test_case->member, member_bytes) == 0),
               "the member it names");
        expect(status == 0 || (refusal.message != NULL && refusal.message[0] != '\0'), "a message for people");
        if (test_case->member != NULL && strcmp(test_case->member, "\"role\"") == 0) {
            expect(refusal.message != NULL && strstr(refusal.message, "system, developer, user, assistant or tool")
                   != NULL, "role refusal lists developer");
        }
        if (test_case->member != NULL && strcmp(test_case->member, "\"reasoning_effort\"") == 0) {
            expect(refusal.message != NULL && strstr(refusal.message, "none, off,") != NULL,
                   "effort refusal lists the off alias");
        }
    }
}

/* text behind the first length bytes of a body, which stays NUL-terminated; the new length. */
static size_t appended(char *body, size_t length, const char *text)
{
    const size_t count = strlen(text);
    if (length + count >= BODY_CAPACITY) {
        expect(false, "room in the body");
        return length;
    }
    memcpy(body + length, text, count + 1u);
    return length + count;
}

/* A body of count messages, in body; its length. */
static size_t messages_body(char *body, uint32_t count)
{
    size_t length = appended(body, 0, "{\"messages\":[");
    for (uint32_t index = 0; index < count; index++) {
        length = appended(body, length, MESSAGE_TEXT);
    }
    return appended(body, length - 1, "]}");                         /* over the last comma */
}

/* Bodies built here: too many messages, too deep, a stop too long, a body above the limit. */
static void test_request_limits(void)
{
    char *const body = malloc(BODY_CAPACITY);
    struct chat_request request = { 0 };
    struct chat_refusal refusal = { 0 };
    if (body == NULL) {
        expect(false, "memory for the bodies");
        return;
    }
    context_text = "4,097 messages, and 4,096";
    size_t length = messages_body(body, CHAT_MOST_MESSAGES + 1u);
    expect(read_request(body, length, &request, &refusal) == BAD, "refused");
    length = messages_body(body, CHAT_MOST_MESSAGES);
    expect(read_request(body, length, &request, &refusal) == OK_STATUS, "taken");
    context_text = "a body deeper than json.h reads";
    length = appended(body, 0, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"m\":");
    for (uint32_t depth = 0; depth < JSON_MAXIMUM_DEPTH; depth++) {
	    length = appended(body, length, DEEP_OPENING);
    }
    for (uint32_t depth = 0; depth < JSON_MAXIMUM_DEPTH; depth++) {
	    length = appended(body, length, "]}");
    }
    length = appended(body, length, "}");
    expect(read_request(body, length, &request, &refusal) == BAD, "refused");
    context_text = "a stop of 257 bytes, and one of 256";
    length = appended(body, 0, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":\"");
    for (uint32_t index = 0; index <= CHAT_MOST_STOP_BYTES; index++) {
	    length = appended(body, length, "a");
    }
    length = appended(body, length, "\"}");
    expect(read_request(body, length, &request, &refusal) == BAD, "refused");
    length = appended(body, length - strlen("a\"}"), "\"}");          /* one a less */
    expect(read_request(body, length, &request, &refusal) == OK_STATUS &&
		   request.stop_byte_counts[0] == CHAT_MOST_STOP_BYTES,
	   "taken");
    context_text = "a body above the limit";
    expect(read_request(body, CHAT_MOST_BODY_BYTES + 1u, &request, &refusal) ==
		   HTTP_SERVER_ERROR,
	   "refused before it is read");
    free(body);
}

static void test_request_values(void)
{
    static const char body[] = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,"
        "\"stream_options\":{\"include_usage\":true},\"max_tokens\":10,\"max_completion_tokens\":5,"
        "\"temperature\":0.25,\"top_p\":0.5,\"seed\":18446744073709551615,\"stop\":[\"\\u00e9\\n\",\"bc\"],"
        "\"reasoning_effort\":\"high\"}";
    static const char plain[] = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":42}";
    static const char none[] = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"none\"}";
    struct chat_request request = { 0 };
    struct chat_refusal refusal = { 0 };
    context_text = "the values a request sets";
    expect(read_request(body, strlen(body), &request, &refusal) == OK_STATUS, "taken");
    expect(request.is_streamed && request.includes_usage && request.max_tokens == SET_MAX_TOKENS &&
           request.temperature == SET_TEMPERATURE && request.top_p == SET_TOP_P && request.seed == UINT64_MAX &&
           request.reasoning_effort == HIGH_EFFORT, "stream, usage, the smaller limit, the sampler, the effort");
    expect(request.stop_count == TWO_STOPS && request.stop_byte_counts[0] == ACCENTED_STOP_BYTES &&
           memcmp(request.stops[0], "\xc3\xa9\n", ACCENTED_STOP_BYTES) == 0 &&
           request.stop_byte_counts[1] == STOP_TWO_BYTES && memcmp(request.stops[1], "bc", STOP_TWO_BYTES) == 0,
           "the stops, unescaped");
    expect(request.messages_end - request.messages_begin == strlen("[{\"role\":\"user\",\"content\":\"hi\"}]") &&
           body[request.messages_begin] == '[', "the messages' span");
    context_text = "the defaults a request keeps";
    expect(read_request(plain, strlen(plain), &request, &refusal) == OK_STATUS && !request.is_streamed &&
           request.seed == DEFAULT_SEED && request.max_tokens == DEFAULT_MAX_TOKENS &&
           request.temperature == DEFAULT_TEMPERATURE && request.top_p == DEFAULT_TOP_P &&
           request.reasoning_effort == FORTY_TWO && request.stop_count == 0, "taken, with the defaults");
    expect(read_request(none, strlen(none), &request, &refusal) == OK_STATUS && request.reasoning_effort == 0,
           "\"none\" turns thinking off");
#if CHECKING_HOOKS_ENABLED
    context_text = "the fault point chat_memory in a request";
    expect(fault_arm("chat_memory", 1), "armed");
    expect(read_request(plain, strlen(plain), &request, &refusal) ==
		   HTTP_SERVER_ERROR,
	   "memory refused: 500");
    expect(fault_disarm() == 1u, "reached");
#endif
}

static void test_deepseek_controls(void)
{
    static const struct {
        const char *members;
        uint32_t default_effort, status, effort;
    } cases[] = {
        {"", HIGH_EFFORT, OK_STATUS, HIGH_EFFORT},
        {",\"reasoning_effort\":null", HIGH_EFFORT, OK_STATUS, HIGH_EFFORT},
        {",\"reasoning_effort\":\"off\"", HIGH_EFFORT, OK_STATUS, 0},
        {",\"reasoning_effort\":\"none\"", HIGH_EFFORT, OK_STATUS, 0},
        {",\"reasoning_effort\":0", HIGH_EFFORT, OK_STATUS, 0},
        {",\"thinking\":{\"type\":\"enabled\"}", HIGH_EFFORT, OK_STATUS, HIGH_EFFORT},
        {",\"thinking\":{\"type\":\"enabled\"},\"reasoning_effort\":\"low\"", 0, OK_STATUS, LOW_EFFORT},
        {",\"reasoning_effort\":\"max\",\"thinking\":{\"type\":\"enabled\"}", 0, OK_STATUS, MAX_EFFORT},
        {",\"thinking\":{\"type\":\"disabled\"},\"reasoning_effort\":\"high\"", HIGH_EFFORT, OK_STATUS, 0},
        {",\"reasoning_effort\":\"high\",\"thinking\":{\"type\":\"disabled\"}", HIGH_EFFORT, OK_STATUS, 0},
        {",\"thinking\":{\"type\":\"disabled\"}", HIGH_EFFORT, OK_STATUS, 0},
        {",\"thinking\":{\"type\":\"enabled\"}", 0, BAD, 0},
        {",\"thinking\":{\"type\":\"enabled\"},\"reasoning_effort\":\"none\"", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":{\"type\":\"enabled\"},\"reasoning_effort\":\"off\"", HIGH_EFFORT, BAD, 0},
        {",\"reasoning_effort\":\"off\",\"thinking\":{\"type\":\"enabled\"}", HIGH_EFFORT, BAD, 0},
        {",\"reasoning_effort\":\"off\",\"thinking\":{\"type\":\"disabled\"}", HIGH_EFFORT, OK_STATUS, 0},
        {",\"thinking\":{\"type\":\"enabled\"},\"reasoning_effort\":0", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":{\"type\":\"enabled\"},\"reasoning_effort\":null", HIGH_EFFORT, OK_STATUS, HIGH_EFFORT},
        {",\"thinking\":null", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":true", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":0", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":\"enabled\"", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":[]", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":{}", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":{\"type\":null}", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":{\"type\":false}", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":{\"type\":\"auto\"}", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":0}", HIGH_EFFORT, BAD, 0},
        {",\"thinking\":{\"clear_thinking\":false,\"type\":\"enabled\"}", HIGH_EFFORT, BAD, 0},
    };
    char body[TEXT_CAPACITY];
    for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
	    struct chat_request settings = defaults,
				request = { .max_tokens = FORTY_TWO };
	    struct chat_refusal refusal = { 0 };
	    settings.reasoning_effort = cases[index].default_effort;
	    (void)snprintf(
		    body, sizeof(body),
		    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]%s}",
		    cases[index].members);
	    context_text = body;
	    const uint32_t status = chat_request_read(body, strlen(body), MODEL,
						      NULL, &settings, &request,
						      &refusal);
	    expect(status == cases[index].status, "mode accepted or refused");
	    if (status == 0) {
		    expect(request.reasoning_effort == cases[index].effort,
			   "resolved effort independent of member order");
	    } else {
		    expect(request.max_tokens == FORTY_TWO &&
				   refusal.message != NULL &&
				   refusal.member_end - refusal.member_begin ==
					   strlen("\"thinking\"") &&
				   memcmp(body + refusal.member_begin,
					  "\"thinking\"",
					  strlen("\"thinking\"")) == 0,
			   "named 400 leaves output untouched");
	    }
    }
    static const struct { const char *name; uint32_t effort; } levels[] = {
        {"none", 0}, {"off", 0}, {"minimal", LOW_EFFORT}, {"low", LOW_EFFORT}, {"medium", HIGH_EFFORT},
        {"high", HIGH_EFFORT},
        {"xhigh", HIGH_EFFORT}, {"max", MAX_EFFORT}};
    for (uint32_t index = 0; index < sizeof(levels) / sizeof(levels[0]); index++) {
	    struct chat_request request = { 0 };
	    struct chat_refusal refusal = { 0 };
	    (void)snprintf(
		    body, sizeof(body),
		    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
		    "\"reasoning_effort\":\"%s\"}",
		    levels[index].name);
	    context_text = levels[index].name;
	    expect(read_request(body, strlen(body), &request, &refusal) == 0 &&
			   request.reasoning_effort == levels[index].effort,
		   "all harness API names keep their mapping");
    }
}

/* Route controls pi-ai can emit directly or via compat/samplingParams. Refusals must name the field. */
static void test_harness_fields(void)
{
    static const struct { const char *members, *param; } cases[] = {
        {"\"store\":false", NULL}, {"\"store\":true", "store"}, {"\"store\":0", "store"},
        {"\"parallel_tool_calls\":true", NULL}, {"\"parallel_tool_calls\":false", "parallel_tool_calls"},
        {"\"parallel_tool_calls\":0", "parallel_tool_calls"},
        {"\"service_tier\":\"default\"", NULL}, {"\"service_tier\":\"auto\"", NULL},
        {"\"service_tier\":\"priority\"", "service_tier"},
        {"\"prompt_cache_key\":\"session-a\",\"metadata\":{},\"user\":\"owner\"", NULL},
        {"\"prompt_cache_retention\":\"24h\"", "prompt_cache_retention"},
        {"\"prompt_cache_retention\":false", "prompt_cache_retention"},
        {"\"priority\":0", "priority"}, {"\"priority\":1", "priority"},
        {"\"thinking_token_budget\":0", "thinking_token_budget"},
        {"\"thinking_budget\":100", "thinking_budget"},
        {"\"thinking_budget_tokens\":100", "thinking_budget_tokens"},
        {"\"provider\":{\"order\":[\"x\"]}", "provider"},
        {"\"providerOptions\":{\"gateway\":{\"only\":[\"x\"]}}", "providerOptions"},
        {"\"tool_stream\":true", "tool_stream"},
        {"\"stream_options\":{\"include_usage\":false}", NULL},
        {"\"stream_options\":{\"include_usage\":0}", "stream_options"},
        {"\"response_format\":{\"type\":\"json_schema\"}", "response_format"},
        {"\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"read\"}}", "tool_choice"},
        {"\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"strict\":false}}]", NULL},
        {"\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"strict\":true}}]", "tools"},
        {"\"tools\":[{\"type\":\"custom\",\"custom\":{\"name\":\"read\"}}]", "tools"},
        {"\"tools\":[{\"function\":{\"name\":\"read\"},\"cache_control\":{\"type\":\"ephemeral\"}}]", "tools"},
        {"\"messages\":[{\"role\":\"assistant\",\"content\":\"x\",\"reasoning_content\":\"\"}]", NULL},
        {"\"messages\":[{\"role\":\"assistant\",\"content\":\"x\",\"reasoning_details\":[]}]", NULL},
        {"\"messages\":[{\"role\":\"assistant\",\"content\":\"x\",\"reasoning_details\":[{}]}]", "reasoning_details"},
        {"\"messages\":[{\"role\":\"assistant\",\"content\":\"x\",\"reasoning\":\"plan\"}]", "reasoning"},
        {"\"messages\":[{\"role\":\"assistant\",\"content\":\"x\",\"reasoning_text\":\"plan\"}]", "reasoning_text"},
        {"\"messages\":[{\"role\":\"developer\",\"content\":\"x\"}]", NULL},
        {"\"messages\":[{\"role\":\"bogus\",\"content\":\"x\"}]", "role"},
        {"\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"x\","
         "\"cache_control\":{\"type\":\"ephemeral\",\"ttl\":\"1h\"}}]}]", "cache_control"},
        {"\"messages\":[{\"role\":\"user\",\"content\":\"x\",\"cache_control\":{\"type\":\"ephemeral\"}}]",
         "cache_control"},
    };
    char body[TEXT_CAPACITY];
    for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
	    struct chat_request request = { 0 };
	    struct chat_refusal refusal = { 0 };
	    const char *const messages =
		    strncmp(cases[index].members, "\"messages\"",
			    strlen("\"messages\"")) == 0 ?
			    "" :
			    "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],";
	    (void)snprintf(body, sizeof(body), "{%s%s}", messages,
			   cases[index].members);
	    context_text = body;
	    const uint32_t status =
		    read_request(body, strlen(body), &request, &refusal);
	    expect(status == (cases[index].param == NULL ? OK_STATUS : BAD),
		   "field accepted or refused");
	    if (cases[index].param != NULL) {
		    char param[CHAT_PREFIX_CAPACITY];
		    (void)snprintf(param, sizeof(param), "\"%s\"",
				   cases[index].param);
		    expect(refusal.message != NULL &&
				   refusal.member_end - refusal.member_begin ==
					   strlen(param) &&
				   memcmp(body + refusal.member_begin, param,
					  strlen(param)) == 0,
			   "clear error names its field");
	    }
    }
}

static void test_draft_request_values(void)
{
    static const struct { const char *value; uint32_t mode; bool keeps_default;
    } cases[] = { { NULL, 0, true },
		  { "null", 0, true },
		  { "\"compact\"", CHAT_DRAFT_USAGE_COMPACT, false },
		  { "\"none\"", CHAT_DRAFT_USAGE_NONE, false },
		  { "\"full\"", CHAT_DRAFT_USAGE_FULL, false } };
    char body[TEXT_CAPACITY];
    for (uint32_t mode = CHAT_DRAFT_USAGE_COMPACT;
	 mode <= CHAT_DRAFT_USAGE_FULL; mode++) {
	    struct chat_request settings = defaults;
	    settings.draft_usage = mode;
	    for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
		 index++) {
		    struct chat_request request = { 0 };
		    struct chat_refusal refusal = { 0 };
		    (void)snprintf(
			    body, sizeof(body),
			    "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]%s%s}",
			    cases[index].value != NULL ?
				    ",\"anemone_draft_usage\":" :
				    "",
			    cases[index].value != NULL ? cases[index].value :
							 "");
		    context_text = body;
		    expect(chat_request_read(body, strlen(body), MODEL, NULL,
					     &settings, &request,
					     &refusal) == OK_STATUS &&
				   request.draft_usage ==
					   (cases[index].keeps_default ?
						    mode :
						    cases[index].mode),
			   "draft mode is explicit or inherited, including null and zero-initialized compact");
	    }
    }
    struct chat_request settings = defaults,
			request = { .max_tokens = FORTY_TWO };
    struct chat_refusal refusal = { 0 };
    settings.draft_usage = CHAT_DRAFT_USAGE_FULL + 1u;
    expect(chat_request_read(body, strlen(body), MODEL, NULL, &settings,
			     &request, &refusal) == HTTP_SERVER_ERROR &&
		   request.max_tokens == FORTY_TWO,
	   "invalid default mode is refused untouched");
}

/* An answer to tokens (NULL: the end token), finished with reason or failed without; the output, NUL-terminated, in
 * text; the index of the token a stop string ended it at, or token_count. */
static void answer_of(const struct chat_request *request,
		      const char *const *tokens, uint32_t token_count,
		      const char *reason, char *text, uint32_t *stopped_at)
{
	struct chat_answer answer = { 0 };
	chat_answer_begin(&answer, request, MODEL, IDENTIFIER, CREATED,
			  PROMPT_TOKENS);
	*stopped_at = token_count;
	for (uint32_t index = 0; index < token_count; index++) {
		if (!answer.is_finished && !answer.stop_matched &&
		    answer.is_reasoning && !answer.is_calling) {
			answer.reasoning_token_count++;
		}
		const bool is_stopped = chat_answer_token(
			&answer, tokens[index],
			tokens[index] != NULL ?
				(uint32_t)strlen(tokens[index]) :
				0);
		*stopped_at = is_stopped && *stopped_at == token_count ?
				      index :
				      *stopped_at;
	}
    if (reason != NULL) {
	    chat_answer_finish(&answer, reason);
    } else {
	    chat_answer_fail(&answer);
    }
    const size_t count = answer.output.bytes != NULL ? answer.output.count - answer.output.sent : 0;
    expect(!answer.output.is_short && count < TEXT_CAPACITY, "the answer fits");
    memcpy(text, count != 0 && count < TEXT_CAPACITY ? answer.output.bytes + answer.output.sent : "",
           count < TEXT_CAPACITY ? count : 0);
    text[count < TEXT_CAPACITY ? count : 0] = '\0';
    chat_answer_free(&answer);
}

/* Whether a whole answer is the body expected, behind a head that says its length. */
static bool is_whole(const char *text, const char *body)
{
	char head[CHAT_PREFIX_CAPACITY];
	(void)snprintf(
		head, sizeof(head),
		"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
		"Connection: close\r\n\r\n",
		strlen(body));
	return strncmp(text, head, strlen(head)) == 0 &&
	       strcmp(text + strlen(head), body) == 0;
}

/* Calls in DSML, as encoding.py's parse_message_from_completion_text reads them. */
#define DSML "｜DSML｜"
#define CALL_OPEN "\n\n<" DSML " calls>\n<" DSML " invoke name=\"read\">\n"
#define CALL_PATH "<" DSML " parameter name=\"path\" string=\"true\">a \"b\".c</" DSML " parameter>\n"
#define CALL_LIMIT "<" DSML " parameter name=\"limit\" string=\"false\">10</" DSML " parameter>\n"
#define CALL_CLOSE "</" DSML " invoke>\n</" DSML " calls>"
#define CALL_END_PIECE 5u                       /* of call in test_calls: the piece that closes the block */
#define CALL_ARGUMENTS "\"arguments\":\"{\\\"path\\\": \\\"a \\\\\\\"b\\\\\\\".c\\\", \\\"limit\\\": 10}\""

static void test_calls(void)
{
    static char text[TEXT_CAPACITY];
    static const char *const call[] = {"Reading.", "\n\n<" DSML " calls", ">\n<" DSML " invoke name=\"read\">\n",
                                       CALL_PATH, CALL_LIMIT, "</" DSML " invoke>\n</" DSML " calls>", "late"};
    static const char *const thought_call[] = {"plan", "</think>", CALL_OPEN CALL_PATH CALL_CLOSE};
    static const char *const malformed[] = {"x", CALL_OPEN "<" DSML " parameter name=\"p\">", "y"};
    static const char *const unclosed[] = {"x", CALL_OPEN "<" DSML " parameter name=\"path\" string=\"true\">a"};
    struct chat_request request = { .max_tokens = DEFAULT_MAX_TOKENS,
				    .top_p = 1.0f,
				    .tools_end = 1 };
    uint32_t stopped_at = 0;
    context_text = "a whole answer that calls";
    answer_of(&request, call, sizeof(call) / sizeof(call[0]), "length", text, &stopped_at);
    expect(stopped_at == CALL_END_PIECE &&
           strstr(text, "\"message\":{\"role\":\"assistant\",\"content\":\"Reading.\",\"tool_calls\":[{\"id\":"
                        "\"chatcmpl-1-0\",\"type\":\"function\",\"function\":{\"name\":\"read\"," CALL_ARGUMENTS
                        "}}]},\"finish_reason\":\"tool_calls\"") != NULL,
           "the content before the block, the call's arguments as the reference decodes them, ended at its end");
    context_text = "a streamed call after reasoning";
    request.is_streamed = true;
    request.reasoning_effort = HIGH_EFFORT;
    answer_of(&request, thought_call, sizeof(thought_call) / sizeof(thought_call[0]), "stop", text, &stopped_at);
    expect(strstr(text, "\"delta\":{\"reasoning_content\":\"plan\"}") != NULL &&
           strstr(text, "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-1-0\",\"type\":\"function\","
                        "\"function\":{\"name\":\"read\","
                        "\"arguments\":\"{\\\"path\\\": \\\"a \\\\\\\"b\\\\\\\".c\\\"}\""
                        "}}]}") != NULL &&
           strstr(text, "\"finish_reason\":\"tool_calls\"") != NULL && strstr(text, "DSML") == NULL,
           "one tool_calls delta of index 0, finished tool_calls");
    context_text = "a malformed block";
    request.is_streamed = false;
    request.reasoning_effort = 0;
    answer_of(&request, malformed, sizeof(malformed) / sizeof(malformed[0]), "stop", text, &stopped_at);
    expect(strstr(text, "\"content\":\"x\\n\\n<" DSML " calls>\\n<" DSML " invoke name=\\\"read\\\">\\n<" DSML
                        " parameter name=\\\"p\\\">y\"},\"finish_reason\":\"stop\"") != NULL,
           "given out as the content the model wrote");
    context_text = "a block cut by the limit";
    answer_of(&request, unclosed, sizeof(unclosed) / sizeof(unclosed[0]), "length", text, &stopped_at);
    expect(strstr(text, "\"tool_calls\"") == NULL && strstr(text, "\"finish_reason\":\"length\"") != NULL,
           "no call whole: no tool_calls");
}

#define EMPTY_CALL_JSON "{\"id\":\"chatcmpl-1-0\",\"type\":\"function\",\"function\":{\"name\":\"list_agents\"," \
                        "\"arguments\":\"{}\"}}"
#define READ_CALL_JSON "{\"id\":\"chatcmpl-1-0\",\"type\":\"function\",\"function\":{\"name\":\"read\"," \
                       CALL_ARGUMENTS "}}"
#define NEXT_CALL_JSON "{\"id\":\"chatcmpl-1-1\",\"type\":\"function\",\"function\":{\"name\":\"list_agents\"," \
                       "\"arguments\":\"{}\"}}"
#define EMPTY_INVOKE "<" DSML " invoke name=\"list_agents\">"
#define INVOKE_CLOSE "</" DSML " invoke>"
#define CALLS_CLOSE "</" DSML " calls>"

static size_t occurrences(const char *text, const char *word)
{
    size_t count = 0;
    for (const char *at = strstr(text, word); at != NULL; at = strstr(at + strlen(word), word)) {
        count++;
    }
    return count;
}

/* Every two-chunk split, then one byte per token, for both response modes. No prefix may end generation. */
static void test_call_splits(const char *block, const char *calls, uint32_t call_count, bool closes)
{
    const size_t length = strlen(block);
    for (uint32_t streamed = 0; streamed <= 1; streamed++) {
        for (size_t split = 0; split <= length + 1; split++) {
		struct chat_request request = { .tools_end = 1,
						.is_streamed = streamed != 0,
						.includes_usage = true };
		struct chat_answer answer = { 0 };
		chat_answer_begin(&answer, &request, MODEL, IDENTIFIER, CREATED,
				  PROMPT_TOKENS);
		answer.defers_stop_finish = true;
		size_t position = 0;
		uint32_t tokens = 0;
		while (position < length) {
			const size_t piece = split > length ? 1 :
					     position < split ?
							      split - position :
							      length - position;
			position += piece;
			tokens++;
			const bool stopped = chat_answer_token(
				&answer, block + position - piece,
				(uint32_t)piece);
			expect(stopped == (closes && position == length),
			       "only the whole closing calls tag stops");
		}
            if (!closes) {
		    expect(!chat_answer_token(&answer, NULL, 0),
			   "EOS does not turn an open block into a stop");
		    tokens++;
            }
            expect(closes || answer.call_count == 0, "truncated reference forms recover only at finalization");
	    chat_answer_finish(&answer, "stop");
	    chat_answer_finish(&answer, "length");
	    expect(answer.call_count == call_count && answer.completion_token_count == tokens,
                   "each call once, every generated token counted");
	    chat_buffer_write(&answer.output, "", 1);
	    const char *const output = answer.output.bytes != NULL ? answer.output.bytes + answer.output.sent : "";
            expect(!answer.output.is_short && strstr(output, "DSML") == NULL &&
                   strstr(output, "\"finish_reason\":\"tool_calls\"") != NULL, "calls instead of raw DSML");
            if (streamed == 0) {
                expect(answer.calls.count == strlen(calls) && memcmp(answer.calls.bytes, calls, strlen(calls)) == 0,
                       "the reference calls and arguments byte for byte");
            } else {
                expect(occurrences(output, "\"tool_calls\":[") == call_count &&
                       occurrences(output, "data: [DONE]") == 1 &&
                       strstr(output, "\"id\":\"chatcmpl-1-0\"") != NULL &&
                       (call_count == 1 || strstr(output, "\"index\":1,\"id\":\"chatcmpl-1-1\"") != NULL),
                       "ordered deltas, stable ids, usage and one DONE");
            }
	    chat_answer_free(&answer);
	}
    }
}

static void test_tolerant_calls(void)
{
    static const struct { const char *block, *calls; uint32_t count; bool closes; } cases[] = {
        {"\n\n<" DSML " calls>\n" EMPTY_INVOKE "\n\n</" DSML " invoke", EMPTY_CALL_JSON, 1, false},
        {"\n\n<" DSML " calls>" EMPTY_INVOKE INVOKE_CLOSE CALLS_CLOSE, EMPTY_CALL_JSON, 1, true},
        {"\n\n<" DSML " calls>\n" EMPTY_INVOKE "\n" INVOKE_CLOSE "\n" CALLS_CLOSE,
         EMPTY_CALL_JSON, 1, true},
        {"\n\n<" DSML " calls>\n\n" EMPTY_INVOKE "\n\n" INVOKE_CLOSE "\n\n" CALLS_CLOSE,
         EMPTY_CALL_JSON, 1, true},
        {"\n\n<" DSML " calls>\n\n\n" EMPTY_INVOKE "\n\n\n" INVOKE_CLOSE "\n\n\n" CALLS_CLOSE,
         EMPTY_CALL_JSON, 1, true},
        {"\n\n<" DSML " calls>\r\n" EMPTY_INVOKE "\r\n\r\n" INVOKE_CLOSE "\r\n" CALLS_CLOSE,
         EMPTY_CALL_JSON, 1, true},
        {"\n\n<" DSML " calls> \t" EMPTY_INVOKE " \t" INVOKE_CLOSE " \t" CALLS_CLOSE,
         EMPTY_CALL_JSON, 1, true},
        {CALL_OPEN CALL_PATH CALL_LIMIT INVOKE_CLOSE "\r\n \t" EMPTY_INVOKE "\n\n" INVOKE_CLOSE
         "\f\v " CALLS_CLOSE, READ_CALL_JSON "," NEXT_CALL_JSON, TWO_STOPS, true},
        {"\n\n<" DSML " calls>\n" EMPTY_INVOKE "\n", EMPTY_CALL_JSON, 1, false},
        {CALL_OPEN CALL_PATH, "{\"id\":\"chatcmpl-1-0\",\"type\":\"function\",\"function\":{\"name\":\"read\","
         "\"arguments\":\"{\\\"path\\\": \\\"a \\\\\\\"b\\\\\\\".c\\\"}\"}}", 1, false}};
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        context_text = cases[index].block;
        test_call_splits(cases[index].block, cases[index].calls, cases[index].count, cases[index].closes);
    }
}

static void test_malformed_calls(void)
{
    static const char *const blocks[] = {
        "\n\n<" DSML " calls>junk" EMPTY_INVOKE INVOKE_CLOSE CALLS_CLOSE,
        "\n\n<" DSML " calls>\n<" DSML " invoke Name=\"bad\">\n" INVOKE_CLOSE CALLS_CLOSE,
        CALL_OPEN "<" DSML " parameter name=\"p\" string=\"TRUE\">v</" DSML " parameter>\n"
        INVOKE_CLOSE CALLS_CLOSE,
        CALL_OPEN CALL_PATH CALL_PATH INVOKE_CLOSE CALLS_CLOSE,
        CALL_OPEN "<" DSML " parameter name=\"p\" string=\"true\">v</" DSML " parameter>bad<"};
    for (size_t index = 0; index < sizeof(blocks) / sizeof(blocks[0]); index++) {
        for (uint32_t streamed = 0; streamed <= 1; streamed++) {
		struct chat_request request = { .tools_end = 1,
						.is_streamed = streamed != 0 };
		struct chat_answer answer = { 0 };
		context_text = blocks[index];
		chat_answer_begin(&answer, &request, MODEL, IDENTIFIER, CREATED,
				  PROMPT_TOKENS);
		for (size_t position = 0; position < strlen(blocks[index]);
		     position++) {
			expect(!chat_answer_token(
				       &answer, blocks[index] + position, 1) &&
				       !answer.is_finished,
			       "malformed bytes do not cancel generation mid-tag");
		}
            static const char tail[] = "complete the tag> and reach EOS";
	    expect(!chat_answer_token(&answer, tail, sizeof(tail) - 1),
		   "the entire later token is kept");
	    expect(!chat_answer_token(&answer, NULL, 0), "EOS is counted");
	    chat_answer_finish(&answer, "stop");
	    chat_buffer_write(&answer.output, "", 1);
	    const char *const output = answer.output.bytes != NULL ? answer.output.bytes + answer.output.sent : "";
            expect(!answer.are_calls_closed && answer.call_count == 0 && strstr(output, tail) != NULL &&
                   strstr(output, "DSML") != NULL && strstr(output, "\"finish_reason\":\"stop\"") != NULL,
                   "the whole malformed block, including the completed tail, is returned as text");
            if (streamed == 0) {
                expect(answer.content.count == strlen(blocks[index]) + strlen(tail) &&
                       memcmp(answer.content.bytes, blocks[index], strlen(blocks[index])) == 0,
                       "fallback preserves all original bytes");
            }
	    chat_answer_free(&answer);
	}
    }
}

static void test_malformed_call_suffix(void)
{
    static const char block[] = CALL_OPEN CALL_PATH INVOKE_CLOSE "\n<" DSML " invoke name=bad>\n"
                                INVOKE_CLOSE "\n" CALLS_CLOSE;
    static char text[TEXT_CAPACITY];
    static const char *const tokens[] = {block, "finish the malformed tail>"};
    struct chat_request request = { .tools_end = 1 };
    for (uint32_t streamed = 0; streamed <= 1; streamed++) {
        for (uint32_t limited = 0; limited <= 1; limited++) {
            request.is_streamed = streamed != 0;
            uint32_t stopped_at = 0;
            context_text = "a malformed suffix after an emitted call, through EOS or the request limit";
            answer_of(&request, tokens, sizeof(tokens) / sizeof(tokens[0]), limited != 0 ? "length" : "stop",
                      text, &stopped_at);
            expect(stopped_at == sizeof(tokens) / sizeof(tokens[0]) &&
                   occurrences(text, "\"name\":\"read\"") == 1 &&
                   strstr(text, "finish the malformed tail>") != NULL &&
                   strstr(text, limited != 0 ? "\"finish_reason\":\"length\""
                                            : "\"finish_reason\":\"tool_calls\"") != NULL,
                   "the valid call once, the entire unread tail as text, the final event's finish reason");
        }
    }
}

static void test_answers(void)
{
    static char text[TEXT_CAPACITY];
    static const char *const hello[] = {"Hel", "lo", NULL};
    static const char *const hi[] = {"Hi"};
    static const char *const thinking[] = {"ab", "</think>", "cd"};
    static const char *const cut_end[] = {"x</th", "ink>y"};
    static const char *const cut_stop[] = {"ab", "cE", "N", "Dzz"};
    static const char *const stop_after_reasoning[] = {"bab", "</think>", "xay"};
    static const char *const accent[] = {"\xc3", "\xa9"};
    static const char *const ill_formed[] = {"\xff", "a", "b\xc3"};
    static const char *const controls[] = {"\n\t\x01\"\\"};
    struct chat_request request = { .max_tokens = DEFAULT_MAX_TOKENS,
				    .top_p = 1.0f };
    uint32_t stopped_at = 0;
    context_text = "a whole answer";
    answer_of(&request, hello, sizeof(hello) / sizeof(hello[0]), "stop", text, &stopped_at);
    expect(is_whole(text, "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"created\":1700000000,\"model\":"
                    "\"test-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":"
                    "\"Hello\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":3,"
                    "\"total_tokens\":6,\"prompt_tokens_details\":{\"cached_tokens\":0},"
                    "\"completion_tokens_details\":{\"reasoning_tokens\":0}}}\n"),
           "its bytes: the end token counted, not written");
    context_text = "a streamed answer with its usage";
    request.is_streamed = request.includes_usage = true;
    answer_of(&request, hi, 1, "length", text, &stopped_at);
    expect(strcmp(text, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                  "Connection: close\r\n\r\n"
                  CHUNK "{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n"
                  CHUNK "{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n"
                  CHUNK "{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"
                  CHUNK "],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4,"
                  "\"prompt_tokens_details\":{\"cached_tokens\":0},"
                  "\"completion_tokens_details\":{\"reasoning_tokens\":0}}}\n\n"
                  "data: [DONE]\n\n") == 0, "its bytes");
    context_text = "the end of reasoning cut across tokens";
    request.includes_usage = false;
    request.reasoning_effort = HIGH_EFFORT;
    answer_of(&request, cut_end, sizeof(cut_end) / sizeof(cut_end[0]), "stop", text, &stopped_at);
    expect(strstr(text, "\"delta\":{\"reasoning_content\":\"x\"}") != NULL &&
           strstr(text, "\"delta\":{\"content\":\"y\"}") != NULL && strstr(text, "think") == NULL,
           "the reasoning, then the content, no delimiter");
    context_text = "a whole answer that thinks";
    request.is_streamed = false;
    answer_of(&request, thinking, sizeof(thinking) / sizeof(thinking[0]), "stop", text, &stopped_at);
    expect(strstr(text, "\"content\":\"cd\",\"reasoning_content\":\"ab\"}") != NULL, "content and reasoning apart");
    context_text = "a stop string in the reasoning and in the content";
    request.stop_count = 1;
    request.stop_byte_counts[0] = 1;
    request.stops[0][0] = 'a';
    answer_of(&request, stop_after_reasoning, sizeof(stop_after_reasoning) / sizeof(stop_after_reasoning[0]), "length",
              text, &stopped_at);
    expect(stopped_at == THIRD_TOKEN && strstr(text, "\"content\":\"x\",\"reasoning_content\":\"bab\"},"
                                                     "\"finish_reason\":\"stop\"") != NULL &&
           strstr(text, "\"completion_tokens\":3,") != NULL, "the content stops, the reasoning does not");
    context_text = "a stop string cut across tokens";
    request.reasoning_effort = 0;
    request.stop_byte_counts[0] = (uint32_t)strlen("END");
    memcpy(request.stops[0], "END", strlen("END"));
    answer_of(&request, cut_stop, sizeof(cut_stop) / sizeof(cut_stop[0]), "length", text, &stopped_at);
    expect(stopped_at == FOURTH_TOKEN && strstr(text, "\"content\":\"abc\"},\"finish_reason\":\"stop\"") != NULL,
           "the text before it, finished at the token that completed it");
    context_text = "UTF-8 held across tokens";
    request.stop_count = 0;
    request.is_streamed = true;
    answer_of(&request, accent, sizeof(accent) / sizeof(accent[0]), "stop", text, &stopped_at);
    expect(strstr(text, "\"delta\":{\"content\":\"\xc3\xa9\"}") != NULL && strstr(text, "\xef\xbf\xbd") == NULL,
           "one chunk of the whole character");
    context_text = "ill-formed bytes, and one unfinished at the end";
    request.is_streamed = false;
    answer_of(&request, ill_formed, sizeof(ill_formed) / sizeof(ill_formed[0]), "stop", text, &stopped_at);
    expect(strstr(text, "\"content\":\"\xef\xbf\xbd" "ab\xef\xbf\xbd\"") != NULL, "each as U+FFFD");
    context_text = "control characters";
    answer_of(&request, controls, 1, "stop", text, &stopped_at);
    expect(strstr(text, "\"content\":\"\\n\\t\\u0001\\\"\\\\\"") != NULL, "escaped as json.dumps does");
    context_text = "a whole answer that fails";
    answer_of(&request, hi, 1, NULL, text, &stopped_at);
    expect(strncmp(text, "HTTP/1.1 500 Internal Server Error\r\n",
		   strlen("HTTP/1.1 500")) == 0 &&
		   strstr(text, CHAT_ENGINE_FAILED) != NULL &&
		   strstr(text, "\"server_error\"") != NULL,
	   "a 500 that says why");
    context_text = "a stream that fails";
    request.is_streamed = true;
    answer_of(&request, hi, 1, NULL, text, &stopped_at);
    expect(strstr(text,
		  "data: {\"error\":{\"message\":\"" CHAT_ENGINE_FAILED) !=
			   NULL &&
		   strstr(text, "[DONE]") == NULL,
	   "an error event and no [DONE]");
    test_calls();
    test_tolerant_calls();
    test_malformed_calls();
    test_malformed_call_suffix();
}

/* What output holds to be sent, NUL-terminated, in text. */
static const char *text_of(const struct chat_buffer *output, char *text)
{
    const size_t count = output->bytes != NULL && !output->is_short ? output->count - output->sent : 0;
    memcpy(text, count != 0 && count < TEXT_CAPACITY ? output->bytes + output->sent : "",
           count < TEXT_CAPACITY ? count : 0);
    text[count < TEXT_CAPACITY ? count : 0] = '\0';
    return text;
}

static bool answer_json_is_valid(const char *text, bool streamed)
{
    const char *const body = strstr(text, "\r\n\r\n");
    if (body == NULL) {
        return false;
    }
    struct json_value document;
    if (!streamed) {
	    return json_read_document(body + strlen("\r\n\r\n"),
				      strlen(body + strlen("\r\n\r\n")),
				      &document) == JSON_DOCUMENT_VALID;
    }
    bool has_json = false;
    const char *event = body;
    while ((event = strstr(event, "data: ")) != NULL) {
        const char *const payload = event + strlen("data: ");
        const char *const end = strstr(payload, "\n\n");
        if (end == NULL) {
            return false;
        }
        if ((size_t)(end - payload) != strlen("[DONE]") || memcmp(payload, "[DONE]", strlen("[DONE]")) != 0) {
		if (json_read_document(payload, (size_t)(end - payload),
				       &document) != JSON_DOCUMENT_VALID) {
			return false;
		}
	    has_json = true;
        }
        event = end + strlen("\n\n");
    }
    return has_json;
}

static void test_reasoning_usage(void)
{
    static const char *const plain[] = {"a long explanatory answer", NULL};
    static const char *const thought[] = {"many characters in one token", "</think>", "answer", NULL};
    static const char *const budget[] = {"many reasoning characters", "more reasoning"};
    static const char *const reasoning_end[] = {"reasoning", NULL};
    static const char *const call[] = {"a plan", "</think>", CALL_OPEN CALL_PATH CALL_CLOSE};
    static const struct {
        const char *const *tokens;
        uint32_t count, effort, reasoning;
        bool tools;
        const char *finish;
    } cases[] = {
        {plain, sizeof(plain) / sizeof(plain[0]), 0, 0, false, "stop"},
        {thought, sizeof(thought) / sizeof(thought[0]), HIGH_EFFORT, REASONING_PHASE_TOKENS,
         false, "stop"},
        {budget, sizeof(budget) / sizeof(budget[0]), HIGH_EFFORT, REASONING_PHASE_TOKENS,
         false, "length"},
        {reasoning_end, sizeof(reasoning_end) / sizeof(reasoning_end[0]), HIGH_EFFORT, REASONING_PHASE_TOKENS,
         false, "stop"},
        {call, sizeof(call) / sizeof(call[0]), HIGH_EFFORT, REASONING_PHASE_TOKENS, true, "stop"}};
    static char text[TEXT_CAPACITY];
    char expected[CHAT_PREFIX_CAPACITY];
    for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        for (uint32_t streamed = 0; streamed <= 1; streamed++) {
		struct chat_request request = {
			.reasoning_effort = cases[index].effort,
			.is_streamed = streamed != 0,
			.includes_usage = true,
			.tools_end = cases[index].tools ? 1u : 0u
		};
		uint32_t stopped_at = 0;
		context_text =
			"reasoning token-phase usage, including delimiter, EOS, budget and tools";
		answer_of(&request, cases[index].tokens, cases[index].count,
			  cases[index].finish, text, &stopped_at);
		(void)snprintf(
			expected, sizeof(expected),
			"\"completion_tokens_details\":{\"reasoning_tokens\":%u}",
			cases[index].reasoning);
		expect(strstr(text, expected) != NULL,
		       "whole and SSE serialize token counts, never character counts");
		(void)snprintf(expected, sizeof(expected),
			       "\"completion_tokens\":%u,", cases[index].count);
		expect(strstr(text, expected) != NULL &&
			       answer_json_is_valid(text, streamed != 0),
		       "completion usage stays inclusive and JSON is valid");
        }
    }
    struct chat_answer answer = { .reasoning_token_count = FORTY_TWO };
    chat_answer_begin(&answer, &defaults, MODEL, IDENTIFIER, CREATED,
		      PROMPT_TOKENS);
    expect(answer.reasoning_token_count == 0, "begin resets a reused answer's reasoning count");
    chat_answer_free(&answer);
}

static void test_refusals_and_listings(void)
{
    static const char body[] = "{\"logprobs\":true}";
    static char text[TEXT_CAPACITY];
    struct chat_buffer output = { 0 };
    struct chat_refusal refusal = { HTTP_METHOD_NOT_ALLOWED, NULL, NULL, 0, 0 };
    context_text = "405 on each route";
    chat_refusal_write(&output, &refusal, NULL, CHAT_ROUTE_COMPLETIONS);
    expect(strncmp(text_of(&output, text), "HTTP/1.1 405 Method Not Allowed\r\n", strlen("HTTP/1.1 405")) == 0 &&
           strstr(text, "Allow: POST\r\n") != NULL, "Allow: POST");
    chat_refusal_write(&output, &refusal, NULL, CHAT_ROUTE_HEALTH);
    expect(strstr(text_of(&output, text), "Allow: GET\r\n") != NULL, "Allow: GET");
    context_text = "401 and 503";
    refusal.status = HTTP_UNAUTHORIZED;
    chat_refusal_write(&output, &refusal, NULL, CHAT_ROUTE_MODELS);
    expect(strstr(text_of(&output, text), "WWW-Authenticate: Bearer\r\n") != NULL &&
           strstr(text, "\"type\":\"authentication_error\"") != NULL, "the scheme, the type");
    refusal.status = HTTP_UNAVAILABLE;
    chat_refusal_write(&output, &refusal, NULL, CHAT_ROUTE_COMPLETIONS);
    expect(strstr(text_of(&output, text),
		  "Retry-After: " CHAT_RETRY_AFTER "\r\n") != NULL,
	   "Retry-After");
    context_text = "a refusal that names a member, with a code";
    refusal = (struct chat_refusal){ HTTP_BAD_REQUEST, "said so", "a_code", 1,
				     strlen("{\"logprobs\"") };
    chat_refusal_write(&output, &refusal, body, CHAT_ROUTE_COMPLETIONS);
    expect(strstr(text_of(&output, text), "\r\n\r\n{\"error\":{\"message\":\"said so\",\"type\":"
                  "\"invalid_request_error\",\"param\":\"logprobs\",\"code\":\"a_code\"}}\n") != NULL, "its body");
    context_text = "the listings";
    chat_listing_write(&output, CHAT_ROUTE_HEALTH, MODEL, NULL, CREATED, 1);
    expect(strstr(text_of(&output, text), "Content-Length: 16\r\n") != NULL &&
           strstr(text, "\r\n\r\n{\"status\":\"ok\"}\n") != NULL, "health");
    chat_listing_write(&output, CHAT_ROUTE_MODELS, MODEL, NULL, CREATED,
		       DEFAULT_MAX_TOKENS);
    expect(strstr(text_of(&output, text), "{\"object\":\"list\",\"data\":[{\"id\":\"test-model\",\"object\":"
                  "\"model\",\"created\":1700000000,\"owned_by\":\"anemone\",\"context_length\":100,"
                  "\"input_modalities\":[\"text\",\"image\"]}]}\n") != NULL,
           "the model");
    chat_listing_write(&output, CHAT_ROUTE_MODELS, MODEL, "alias", CREATED,
		       DEFAULT_MAX_TOKENS);
    (void)text_of(&output, text);
    const char *const listing = strstr(text, "\r\n\r\n");
    struct json_value document;
    expect(listing != NULL &&
		   json_read_document(listing + strlen("\r\n\r\n"),
				      strlen(listing + strlen("\r\n\r\n")),
				      &document) == JSON_DOCUMENT_VALID &&
		   occurrences(text,
			       "\"input_modalities\":[\"text\",\"image\"]") ==
			   LISTED_MODELS_WITH_ALIAS &&
		   strstr(text, "\"id\":\"test-model\"") != NULL &&
		   strstr(text, "\"id\":\"alias\"") != NULL,
	   "both aliases expose the same text and image capability in valid JSON");
    context_text = "the buffer's limit";
    expect(chat_buffer_room(&output, CHAT_MOST_OUTPUT_BYTES + 1u) == NULL &&
		   output.is_short,
	   "beyond it, no room and short");
    chat_buffer_free(&output);
#if CHECKING_HOOKS_ENABLED
    context_text = "the fault point chat_memory in a buffer";
    expect(fault_arm("chat_memory", 1), "armed");
    expect(chat_buffer_room(&output, 1) == NULL && output.is_short,
	   "memory refused: short");
    expect(fault_disarm() == 1u, "reached");
    chat_buffer_free(&output);
#endif
}

/* Where the server sets no seed: the request's, else one drawn for each request. */
static void test_seeds(void)
{
    static const char plain[] = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    static const char seeded[] = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"seed\":9}";
    static const char null_seed[] = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"seed\":null}";
    const struct chat_request unseeded = { .max_tokens = DEFAULT_MAX_TOKENS,
					   .top_p = DEFAULT_TOP_P };
    struct chat_request first = { 0 }, second = { 0 }, third = { 0 };
    struct chat_refusal refusal = { 0 };
    context_text = "seeds where the server sets none";
    expect(chat_request_read(seeded, strlen(seeded), MODEL, NULL, &unseeded,
			     &first, &refusal) == OK_STATUS &&
		   first.seed == REQUEST_SEED,
	   "the request's");
    expect(chat_request_read(plain, strlen(plain), MODEL, NULL, &unseeded,
			     &first, &refusal) == OK_STATUS &&
		   chat_request_read(plain, strlen(plain), MODEL, NULL,
				     &unseeded, &second,
				     &refusal) == OK_STATUS &&
		   chat_request_read(null_seed, strlen(null_seed), MODEL, NULL,
				     &unseeded, &third, &refusal) == OK_STATUS,
	   "taken");
    expect(first.seed != second.seed && first.seed != third.seed && second.seed != third.seed,
           "a fresh one for each request without a seed, null or left out");
#if CHECKING_HOOKS_ENABLED
    context_text = "the fault point chat_random";
    expect(fault_arm("chat_random", 1), "armed");
    expect(chat_request_read(seeded, strlen(seeded), MODEL, NULL, &unseeded,
			     &first, &refusal) == OK_STATUS,
	   "a request with its seed draws none");
    expect(chat_request_read(plain, strlen(plain), MODEL, NULL, &unseeded,
			     &first, &refusal) == HTTP_SERVER_ERROR &&
		   refusal.status == HTTP_SERVER_ERROR,
	   "no seed to draw: 500");
    expect(fault_disarm() == 1u, "reached once, by the request without a seed");
#endif
}

static void test_draft_usage(void)
{
    static char text[TEXT_CAPACITY];
    for (uint32_t streamed = 0; streamed <= 1; streamed++) {
	    for (uint32_t mode = CHAT_DRAFT_USAGE_COMPACT;
		 mode <= CHAT_DRAFT_USAGE_FULL; mode++) {
		    context_text =
			    "draft modes preserve output/usage after a deferred stop, whole and SSE";
		    struct chat_request request = { .top_p = 1.0f,
						    .stop_count = 1,
						    .stop_byte_counts = { 1 },
						    .stops = { { 'x' } },
						    .draft_usage = mode,
						    .is_streamed = streamed !=
								   0,
						    .includes_usage = true };
		    struct chat_answer answer = { 0 };
		    chat_answer_begin(&answer, &request, MODEL, IDENTIFIER,
				      CREATED, PROMPT_TOKENS);
		    answer.defers_stop_finish = true;
		    expect(!chat_answer_token(&answer, "a", 1),
			   "ordinary content is unchanged");
		    expect(chat_answer_token(&answer, "x", 1) &&
				   !answer.is_finished,
			   "stop waits for engine final");
		    expect(!chat_answer_token(&answer, "y", 1) &&
				   answer.completion_token_count ==
					   GENERATED_BEFORE_STOP,
			   "events following a stop add no content or usage tokens");
		    answer.has_draft_statistics = true;
		    answer.draft_command.drawing_rounds = DEFAULT_SEED;
		    answer.draft_command.depth_rounds[0] = DEFAULT_SEED;
		    answer.draft_command.source[DRAFT_LOOKUP].rounds =
			    SET_MAX_TOKENS;
		    answer.draft_command.source[DRAFT_LOOKUP].proposed[0] =
			    SET_MAX_TOKENS;
		    answer.draft_command.source[DRAFT_LOOKUP].reached[0] =
			    PROMPT_TOKENS;
		    answer.draft_command.source[DRAFT_LOOKUP].accepted[0] =
			    STOP_TWO_BYTES;
		    answer.draft_command.source[DRAFT_DSPARK].rounds =
			    STOP_TWO_BYTES;
		    answer.draft_command.source[DRAFT_DSPARK].proposed[0] =
			    STOP_TWO_BYTES;
		    answer.draft_command.source[DRAFT_DSPARK].reached[0] = 1;
		    answer.draft_command.source[DRAFT_DSPARK].accepted[0] = 1;
		    answer.draft_command.confidence[1][0][0] =
			    (struct draft_outcomes){ DEFAULT_SEED,
						     SET_MAX_TOKENS,
						     PROMPT_TOKENS,
						     STOP_TWO_BYTES };
		    answer.draft_total = answer.draft_command;
		    answer.draft_total.drawing_rounds = FORTY_TWO;
		    chat_answer_finish(&answer, "length");
		    (void)text_of(&answer.output, text);
		    const char *const body = strstr(text, "\r\n\r\n");
		    if (streamed == 0 && body != NULL) {
			    (void)printf(
				    "draft usage fixture: mode=%u response_body_bytes=%zu\n",
				    mode, strlen(body + strlen("\r\n\r\n")));
		    }
		    expect(answer_json_is_valid(text, streamed != 0) &&
				   strstr(text,
					  "\"completion_tokens\":2,\"total_tokens\":5,") !=
					   NULL &&
				   strstr(text,
					  "\"completion_tokens_details\":{\"reasoning_tokens\":0}") !=
					   NULL &&
				   strstr(text, "\"finish_reason\":\"stop\"") !=
					   NULL &&
				   strstr(text,
					  streamed != 0 ?
						  "\"delta\":{\"content\":\"a\"}" :
						  "\"content\":\"a\"") != NULL,
			   "all modes keep ordinary output and standard token usage, in valid JSON");
		    if (mode == CHAT_DRAFT_USAGE_NONE) {
			    expect(strstr(text, "\"anemone_draft\"") == NULL,
				   "none omits the extension");
		    } else {
			    expect(strstr(text,
					  "\"anemone_draft\":{\"version\":1,\"command\":{\"drawing_rounds\":7,") !=
						   NULL &&
					   strstr(text,
						  "\"sources\":[{\"rounds\":5,\"proposed\":[5,0,0,0,0,0,0],"
						  "\"reached\":[3,0,0,0,0,0,0],\"accepted\":[2,0,0,0,0,0,0]},"
						  "{\"rounds\":2,") != NULL &&
					   strstr(text,
						  "\"accepted\":[1,0,0,0,0,0,0]") !=
						   NULL,
				   "compact and full retain every owner-script counter, fixed positions and source order");
			    if (mode == CHAT_DRAFT_USAGE_FULL) {
				    expect(strstr(text, "[7,5,3,2]") != NULL &&
						   strstr(text,
							  "\"depth_rounds\":[7,0,0,0,0,0,0,0]") !=
							   NULL &&
						   strstr(text,
							  "\"total\":{\"drawing_rounds\":42") !=
							   NULL,
					   "full preserves both scopes, depth and confidence outcomes");
			    } else {
				    expect(strstr(text, "\"total\":") == NULL &&
						   strstr(text,
							  "\"confidence\":") ==
							   NULL &&
						   strstr(text,
							  "\"depth_rounds\":") ==
							   NULL,
					   "compact contains this request only, without calibration arrays");
			    }
		    }
		    chat_answer_free(&answer);
	    }
    }
    for (uint32_t mode = CHAT_DRAFT_USAGE_COMPACT;
	 mode <= CHAT_DRAFT_USAGE_FULL; mode++) {
	    struct chat_request request = { .draft_usage = mode,
					    .is_streamed = true };
	    struct chat_answer answer = { 0 };
	    chat_answer_begin(&answer, &request, MODEL, IDENTIFIER, CREATED,
			      PROMPT_TOKENS);
	    answer.has_draft_statistics = true;
	    chat_answer_finish(&answer, "stop");
	    expect(strstr(text_of(&answer.output, text), "\"usage\":") ==
				   NULL &&
			   strstr(text, "[DONE]") != NULL,
		   "SSE without include_usage omits all usage in every draft mode");
	    chat_answer_free(&answer);
    }
}

int main(void)
{
    test_request_table();
    test_request_limits();
    test_request_values();
    test_deepseek_controls();
    test_harness_fields();
    test_draft_request_values();
    test_seeds();
    test_answers();
    test_reasoning_usage();
    test_draft_usage();
    test_refusals_and_listings();
    (void)printf("chat_completions_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
