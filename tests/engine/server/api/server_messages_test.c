/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU-only protocol conversion checks and a real native HTTP/engine fixture for clients. */
#include "engine/server/server.h"
#include "models/deepseek_v41_flash/prompt/ds41f_chat_template.h"
#include "engine/library/sampling/sampler_shard_records.h"
#include "common/hooks.h"
#include "support/serving_fixture.h"
#include "engine/server/api/server_messages.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CONTEXT 16384u
#define OUTPUT_LIMIT 4096u
#define DEFAULT_EFFORT 75u
#define QUEUE_DEPTH 4u
#define USER_QUEUE_DEPTH 2u
#define TOP_TWO 2u
#define LONG_TOKENS 2048u
#define SLOW_STEP_NS 1000000ull
#define DETACHED_STEP_NS \
	250000000ull /* the client reads the api_error and closes before the final event */
#define TOOL_PREFIX \
	"\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"city\" " \
	"string=\"true\">Rome</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
#define CUT_INVOKE \
	"\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">bad"
#define CUT_TOOLS \
	"\"tools\":[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\"}}]"
#define CUT_ANSWER_IDENTIFIER "msg_cut"
#define CUT_PROMPT_TOKENS 3u
#define HEAD_END "\r\n\r\n"
#define TEXT_AFTER_CALLS "text after tool_use is unsupported"
#define CUT_ANSWER_BLOCKS 2u /* the whole call, then the cut one as text */
#define PORT_ARGUMENT 2
#define MODE_ARGUMENT 3
#define FIXTURE_ARGUMENTS 4
#define KEY_ARGUMENTS 5
#define DECIMAL_BASE 10
#define RENDER_BYTES 32768u
#define COMPACT_PHASES 2u
#define EMOJI_THIRD_INDEX 2u
#define EMOJI_FOURTH_INDEX 3u
#define UTF8_TEST_EXTRA 2u
#define COMPACT_SESSION_BYTES 20u
#define COMPACT_EXCHANGE_BYTES 2560u
#define REGRESSION_BUDGET 1024u
#define UTF8_TAIL 1022u
#define EMOJI_BYTES 4u
#define EMOJI_LEAD 0xF0u
#define EMOJI_SECOND 0x9Fu
#define EMOJI_THIRD 0x98u
#define EMOJI_LAST 0x80u
#define RESPONSE_PREFIX_BYTES 128u
	_Alignas(struct json_bytes) static char reader_bytes[RENDER_BYTES];
static struct json_scratch reader_workspace = { .bytes = reader_bytes,
						.capacity =
							sizeof(reader_bytes) };
_Alignas(struct json_bytes) static char answer_scratch[RENDER_BYTES];
static int failures;
static const struct chat_request defaults = { .max_tokens = OUTPUT_LIMIT,
					      .top_p = 1.0f,
					      .is_seeded = true,
					      .reasoning_effort =
						      DEFAULT_EFFORT };
static void expect(bool condition, const char *label)
{
	if (!condition) {
		(void)printf("FAILED: %s\n", label);
		failures++;
	}
}
static void conversion_tests(void)
{
	static const char *const valid[] = {
		"{\"model\":\"anemone\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
		"{\"model\":\"anemone\",\"max_tokens\":8,\"system\":[{\"type\":\"text\",\"text\":\"s\"}],"
		"\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\","
		"\"text\":\"b\"}]}]}",
		"{\"model\":\"anemone\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"q\"},"
		"{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"lookup\","
		"\"input\":{\"city\":\"Rome\"}}]},{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
		"\"tool_use_id\":\"toolu_1\",\"content\":\"ok\"}]}],\"tools\":[{\"name\":\"lookup\","
		"\"input_schema\":{\"type\":\"object\"}}]}"
	};
	for (size_t index = 0; index < sizeof(valid) / sizeof(valid[0]);
	     index++) {
		struct chat_buffer body = { 0 };
		struct chat_request request = { 0 };
		struct chat_refusal refusal = { 0 };
		expect(server_messages_read(valid[index], strlen(valid[index]),
					    true, &defaults, &request, &body,
					    &reader_workspace, &refusal) == 0 &&
			       request.is_messages &&
			       request.reasoning_effort == 0,
		       "valid normalization disables native thinking");
		static char rendered[RENDER_BYTES];
		size_t count = 0;
		const struct model_prompt_options options = {
			.reasoning_effort = DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT,
			.drops_thinking = true
		};
		expect(ds41f_chat_template_render(
			       body.bytes + request.messages_begin,
			       request.messages_end - request.messages_begin,
			       request.tools_end > request.tools_begin ?
				       body.bytes + request.tools_begin :
				       NULL,
			       request.tools_end - request.tools_begin,
			       &options, rendered, sizeof(rendered),
			       &count) == STATUS_OK,
		       "normalized history renders with shared template");
		chat_buffer_free(&body);
	}
	static const char *const invalid[] = {
		"{}",
		"[]",
		"{",
		"{\"messages\":[],\"messages\":[]}",
		"{\"model\":\"anemone\",\"max_tokens\":0,\"messages\":[]}",
		"{\"model\":\"anemone\",\"max_tokens\":8,\"messages\":[{\"role\":\"assistant\",\"content\":\"x\"}]}"
	};
	for (size_t index = 0; index < sizeof(invalid) / sizeof(invalid[0]);
	     index++) {
		struct chat_buffer body = { 0 };
		struct chat_request request = { 0 };
		struct chat_refusal refusal = { 0 };
		expect(server_messages_read(
			       invalid[index], strlen(invalid[index]), true,
			       &defaults, &request, &body, &reader_workspace,
			       &refusal) == HTTP_BAD_REQUEST,
		       "malformed request refused");
		chat_buffer_free(&body);
	}
}
static void allocation_refusal(void)
{
#if CHECKING_HOOKS_ENABLED
	static const char body[] =
		"{\"model\":\"anemone\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
	struct chat_buffer normalized = { 0 };
	struct chat_request request = { 0 };
	struct chat_refusal refusal = { 0 };
	expect(fault_arm("chat_memory", 1), "allocation failure armed");
	const uint32_t status = server_messages_read(
		body, sizeof(body) - 1u, true, &defaults, &request, &normalized,
		&reader_workspace, &refusal);
	const uint64_t reached = fault_disarm();
	expect(status == HTTP_SERVER_ERROR && reached != 0,
	       "actual allocation failure remains a server error");
	chat_buffer_free(&normalized);
#endif
}

static void stable_history_tests(void)
{
	static const char body[] =
		"{\"model\":\"anemone\",\"max_tokens\":2048,\"messages\":[{\"role\":\"user\",\"content\":\"First "
		"question\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\","
		"\"thinking\":\"retained reasoning\",\"signature\":\"anemone-v1-effort-100\"},{\"type\":\"text\","
		"\"text\":\"prior answer\"}]},{\"role\":\"user\",\"content\":\"NEWQUESTION\"}]}";
	struct chat_buffer normalized = { 0 };
	struct chat_request request = { 0 };
	struct chat_refusal refusal = { 0 };
	expect(server_messages_read(body, strlen(body), true, &defaults,
				    &request, &normalized, &reader_workspace,
				    &refusal) == 0,
	       "thinking history normalized");
	static const uint32_t levels[] = { 0, 50, 75, 100 };
	static char first[RENDER_BYTES], rendered[RENDER_BYTES];
	size_t shared = 0;
	for (size_t index = 0; index < sizeof(levels) / sizeof(levels[0]);
	     index++) {
		const struct model_prompt_options options = {
			.stable_turns = true,
			.is_thinking = levels[index] != 0,
			.reasoning_effort = levels[index] != 0 ?
						    levels[index] :
						    DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT
		};
		size_t count = 0;
		expect(ds41f_chat_template_render(
			       normalized.bytes + request.messages_begin,
			       request.messages_end - request.messages_begin,
			       NULL, 0, &options, rendered, sizeof(rendered),
			       &count) == STATUS_OK,
		       "stable per-turn render");
		const char *last = memmem(rendered, count, "NEWQUESTION",
					  strlen("NEWQUESTION"));
		expect(last != NULL, "latest user prefix exists");
		if (index == 0 && last != NULL) {
			shared = (size_t)(last - rendered) +
				 strlen("NEWQUESTION");
			memcpy(first, rendered, shared);
		}
		expect(shared != 0 && memcmp(first, rendered, shared) == 0,
		       "entire history byte-identical across next-turn modes");
		expect(memmem(rendered, shared, "Reasoning Effort: 100",
			      strlen("Reasoning Effort: 100")) != NULL,
		       "prior turn effort remains its own native effort");
		expect(memmem(rendered, shared, "retained reasoning",
			      strlen("retained reasoning")) != NULL,
		       "supplied thinking preserved independently of next mode");
	}
	chat_buffer_free(&normalized);
}

/* A prior turn's signature effort is read by a checked parse: digits only, at most 3 (unsigned overflow of a long
 * digit run, undefined for sscanf, falls back to the default effort instead of wrapping into range). */
static void signature_effort_tests(void)
{
	static const struct {
		const char *signature, *effort;
	} cases[] = { { "anemone-v1-effort-60", "Reasoning Effort: 60" },
		      { "anemone-v1-effort-4294967396",
			"Reasoning Effort: 75" },
		      { "anemone-v1-effort-6x", "Reasoning Effort: 75" },
		      { "anemone-v1-effort- 60", "Reasoning Effort: 75" } };
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		static char body[RENDER_BYTES], rendered[RENDER_BYTES];
		const int length = snprintf(
			body, sizeof(body),
			"{\"model\":\"anemone\",\"max_tokens\":2048,\"messages\":[{\"role\":\"user\","
			"\"content\":\"First\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\","
			"\"thinking\":\"kept\",\"signature\":\"%s\"},{\"type\":\"text\",\"text\":\"prior\"}]},"
			"{\"role\":\"user\",\"content\":\"NEXT\"}]}",
			cases[index].signature);
		struct chat_buffer normalized = { 0 };
		struct chat_request request = { 0 };
		struct chat_refusal refusal = { 0 };
		const bool read =
			length > 0 && (size_t)length < sizeof(body) &&
			server_messages_read(body, (size_t)length, true,
					     &defaults, &request, &normalized,
					     &reader_workspace, &refusal) == 0;
		const struct model_prompt_options options = {
			.stable_turns = true,
			.is_thinking = true,
			.reasoning_effort = DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT
		};
		size_t count = 0;
		const char *const next =
			read && ds41f_chat_template_render(
					normalized.bytes +
						request.messages_begin,
					request.messages_end -
						request.messages_begin,
					NULL, 0, &options, rendered,
					sizeof(rendered), &count) == STATUS_OK ?
				memmem(rendered, count, "NEXT",
				       strlen("NEXT")) :
				NULL;
		expect(next != NULL &&
			       memmem(rendered, (size_t)(next - rendered),
				      cases[index].effort,
				      strlen(cases[index].effort)) != NULL,
		       "a prior turn's signature effort is a checked, bounded parse");
		chat_buffer_free(&normalized);
	}
}

static void review_regressions(void)
{
	expect(STEP_SESSION_BYTES == COMPACT_SESSION_BYTES,
	       "session settings carry only the required words");
	expect(SAMPLER_SHARD_PHASES == COMPACT_PHASES &&
		       SAMPLER_SHARD_TOTAL_EXCHANGE_BYTES ==
			       COMPACT_EXCHANGE_BYTES,
	       "sharded exchange uses two phases and one bounded entry barrier");
	static const struct {
		uint32_t status;
		const char *reason;
	} reasons[] = { { HTTP_REQUEST_TIMEOUT, "Request Timeout" },
			{ HTTP_LENGTH_REQUIRED, "Length Required" },
			{ HTTP_CONTENT_TOO_LARGE, "Content Too Large" },
			{ HTTP_EXPECTATION_FAILED, "Expectation Failed" },
			{ HTTP_FIELDS_TOO_LARGE,
			  "Request Header Fields Too Large" },
			{ HTTP_NOT_IMPLEMENTED, "Not Implemented" } };
	for (size_t index = 0; index < sizeof(reasons) / sizeof(reasons[0]);
	     index++) {
		struct chat_buffer output = { 0 };
		const struct chat_refusal refusal = {
			.status = reasons[index].status
		};
		server_messages_refuse(&output, &refusal);
		char expected[RESPONSE_PREFIX_BYTES];
		const int count = snprintf(expected, sizeof(expected),
					   "HTTP/1.1 %u %s\r\n",
					   reasons[index].status,
					   reasons[index].reason);
		expect(count > 0 && (size_t)count < sizeof(expected) &&
			       output.count >= (size_t)count &&
			       memcmp(output.bytes + output.sent, expected,
				      (size_t)count) == 0,
		       "Messages uses the shared HTTP reason phrase");
		chat_buffer_free(&output);
	}
}

/* A member of an object, or an empty span. */
static struct json_value member_of(struct json_value object, const char *name)
{
	struct json_value key = { 0 }, value = { 0 };
	for (size_t cursor = object.begin;
	     object.text != NULL && object.text[object.begin] == '{' &&
	     json_next(object, &cursor, &key, &value);) {
		if (json_string_is(&reader_workspace, key, name))
			return value;
	}
	return (struct json_value){ 0 };
}

/* A replay of a turn whose content is assistant (a JSON array) and whose next user turn is user, read as the server
 * reads it. */
static uint32_t replay_read(const char *assistant, const char *user,
			    struct chat_buffer *converted,
			    struct chat_refusal *refusal)
{
	static char body[RENDER_BYTES];
	struct chat_request request = { 0 };
	const int length = snprintf(
		body, sizeof(body),
		"{\"model\":\"anemone\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"q\"},"
		"{\"role\":\"assistant\",\"content\":%s},{\"role\":\"user\",\"content\":%s}]," CUT_TOOLS
		"}",
		assistant, user);
	return length > 0 && (size_t)length < sizeof(body) ?
		       server_messages_read(body, (size_t)length, true,
					    &defaults, &request, converted,
					    &reader_workspace, refusal) :
		       HTTP_SERVER_ERROR;
}

/* v0.0.6's R13: at a request limit a cut call goes out as text after the whole calls (the fixture's tool_suffix). A
 * verbatim replay of that answer reads as the same turn with its text first, the Chat form of the answer; any other
 * block after such text is refused at the text, as before. */
static void cut_turn_replay_tests(void)
{
	static const char asked[] =
		"{\"model\":\"anemone\",\"max_tokens\":210,"
		"\"messages\":[{\"role\":\"user\",\"content\":\"q\"}]," CUT_TOOLS
		"}";
	static char stream[RENDER_BYTES], verbatim[RENDER_BYTES],
		text_first[RENDER_BYTES], results[RENDER_BYTES],
		refused[RENDER_BYTES];
	static const char *const after_text[] = {
		",{\"type\":\"tool_use\",\"id\":\"toolu_2\",\"name\":\"lookup\",\"input\":{}}",
		",{\"type\":\"thinking\",\"thinking\":\"t\"}",
		",{\"type\":\"image\"}", ",\"text\""
	};
	struct chat_buffer asked_body = { 0 }, names = { 0 };
	struct chat_request request = { 0 };
	struct chat_refusal refusal = { 0 };
	struct chat_answer answer = { 0 };
	const char *const close = strstr(TOOL_PREFIX, "\n</｜DSML｜ calls>");
	const int length = snprintf(stream, sizeof(stream), "%.*s%s",
				    (int)(close - TOOL_PREFIX), TOOL_PREFIX,
				    CUT_INVOKE);
	expect(length > 0 && (size_t)length < sizeof(stream) &&
		       server_messages_read(asked, sizeof(asked) - 1u, true,
					    &defaults, &request, &asked_body,
					    &reader_workspace, &refusal) == 0,
	       "the cut turn's request is read");
	chat_tool_names_kept(asked_body.bytes + request.tools_begin,
			     request.tools_end - request.tools_begin, &names);
	chat_answer_begin(&answer, &request, MESSAGES_MODEL,
			  CUT_ANSWER_IDENTIFIER, 0, CUT_PROMPT_TOKENS);
	answer.tool_names = &names;
	answer.messages.scratch = answer_scratch;
	answer.messages.scratch_capacity = sizeof(answer_scratch);
	(void)chat_answer_token(&answer, stream, (uint32_t)length);
	chat_answer_finish(&answer, "length");
	const char *const head_end =
		answer.output.bytes != NULL ?
			memmem(answer.output.bytes + answer.output.sent,
			       answer.output.count - answer.output.sent,
			       HEAD_END, strlen(HEAD_END)) :
			NULL;
	struct json_value document = { 0 }, call = { 0 }, cut = { 0 },
			  block = { 0 };
	if (head_end != NULL &&
	    json_read_document(head_end + strlen(HEAD_END),
			       (size_t)(answer.output.bytes +
					answer.output.count - head_end) -
				       strlen(HEAD_END),
			       &document) != JSON_DOCUMENT_VALID)
		document = (struct json_value){ 0 };
	const struct json_value content = member_of(document, "content");
	uint32_t blocks = 0;
	for (size_t cursor = content.begin;
	     content.text != NULL && json_next(content, &cursor, NULL, &block);
	     blocks++)
		*(blocks == 0 ? &call : &cut) = block;
	const struct json_value cut_text = member_of(cut, "text"),
				id = member_of(call, "id");
	expect(blocks == CUT_ANSWER_BLOCKS &&
		       json_string_is(&reader_workspace,
				      member_of(call, "type"), "tool_use") &&
		       json_string_is(&reader_workspace, member_of(cut, "type"),
				      "text") &&
		       cut_text.text != NULL &&
		       strncmp(cut_text.text + cut_text.begin,
			       "\"<｜DSML｜ invoke",
			       strlen("\"<｜DSML｜ invoke")) == 0,
	       "R13's answer: the whole call, then the cut call as text");
	const int formed[] = {
		snprintf(verbatim, sizeof(verbatim), "%.*s",
			 (int)(content.end - content.begin),
			 content.text != NULL ? content.text + content.begin :
						""),
		snprintf(text_first, sizeof(text_first), "[%.*s,%.*s]",
			 (int)(cut.end - cut.begin),
			 cut.text != NULL ? cut.text + cut.begin : "",
			 (int)(call.end - call.begin),
			 call.text != NULL ? call.text + call.begin : ""),
		snprintf(
			results, sizeof(results),
			"[{\"type\":\"tool_result\",\"tool_use_id\":%.*s,\"content\":\"ok\"}]",
			(int)(id.end - id.begin),
			id.text != NULL ? id.text + id.begin : "")
	};
	for (size_t index = 0; index < sizeof(formed) / sizeof(formed[0]);
	     index++)
		expect(formed[index] > 0 &&
			       (size_t)formed[index] < sizeof(verbatim),
		       "the replays are formed");
	struct chat_buffer replayed = { 0 }, reordered = { 0 };
	expect(replay_read(verbatim, results, &replayed, &refusal) == 0 &&
		       replay_read(text_first, results, &reordered, &refusal) ==
			       0 &&
		       replayed.count == reordered.count &&
		       memcmp(replayed.bytes, reordered.bytes,
			      replayed.count) == 0,
	       "a verbatim replay reads as the turn with its text first");
	chat_buffer_free(&replayed);
	chat_buffer_free(&reordered);
	expect(replay_read(verbatim, "\"next\"", &replayed, &refusal) ==
			       HTTP_BAD_REQUEST &&
		       strcmp(refusal.message,
			      "missing immediate tool_result") == 0,
	       "the replayed calls still need their results");
	chat_buffer_free(&replayed);
	for (size_t index = 0;
	     index < sizeof(after_text) / sizeof(after_text[0]); index++) {
		const int count = snprintf(
			refused, sizeof(refused), "%.*s%s]",
			(int)(content.end - content.begin) - 1,
			content.text != NULL ? content.text + content.begin :
					       "",
			after_text[index]);
		expect(count > 0 && (size_t)count < sizeof(refused) &&
			       replay_read(refused, results, &replayed,
					   &refusal) == HTTP_BAD_REQUEST &&
			       strcmp(refusal.message, TEXT_AFTER_CALLS) == 0,
		       "a block after the text after the calls is refused at the text");
		chat_buffer_free(&replayed);
	}
	chat_answer_free(&answer);
	chat_buffer_free(&names);
	chat_buffer_free(&asked_body);
}

static int fixture(int argc, char **argv)
{
	char *end = NULL;
	const unsigned long port =
		strtoul(argv[PORT_ARGUMENT], &end, DECIMAL_BASE);
	if (*end != '\0' || port == 0 || port > SERVER_MOST_PORT)
		return EXIT_FAILURE;
	static const uint32_t hello[] = { 'H', 'i', TEST_SERVING_END_TOKEN };
	uint32_t script[OUTPUT_LIMIT];
	uint32_t reasoning_script[OUTPUT_LIMIT];
	uint32_t reasoning_count = 0;
	/* invalid_tool_metrics: the invalid tool output with metrics enabled; invalid_tool_detached: the same, with the
	 * steps after the failing call slow, so the connection closes before the engine's final event (as natively) */
	const bool detached =
		strcmp(argv[MODE_ARGUMENT], "invalid_tool_detached") == 0;
	const bool invalid_tool =
		strcmp(argv[MODE_ARGUMENT], "invalid_tool") == 0 ||
		strcmp(argv[MODE_ARGUMENT], "invalid_tool_metrics") == 0 ||
		detached;
	const bool metrics =
		strcmp(argv[MODE_ARGUMENT], "both") == 0 ||
		strcmp(argv[MODE_ARGUMENT], "invalid_tool_metrics") == 0 ||
		detached ||
		strcmp(argv[MODE_ARGUMENT], "reasoning_metrics") == 0 ||
		strcmp(argv[MODE_ARGUMENT], "metrics_error") == 0 ||
		strcmp(argv[MODE_ARGUMENT], "metrics_only") == 0;
	test_serving_set_metrics(metrics);
	test_serving_set_reasoning_script(NULL, 0);
	const char *const tool =
		invalid_tool ?
			"\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"city\" "
			"string=\"false\">NaN</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>" :
			TOOL_PREFIX;
	const uint32_t *selected = hello;
	if (strcmp(argv[MODE_ARGUMENT], "tool") == 0 || invalid_tool) {
		for (size_t index = 0; index < strlen(tool); index++)
			script[index] = (unsigned char)tool[index];
		script[strlen(tool)] = TEST_SERVING_END_TOKEN;
		selected = script;
	} else if (strcmp(argv[MODE_ARGUMENT], "slow") == 0) {
		for (uint32_t index = 0; index < LONG_TOKENS; index++)
			script[index] = 'x';
		script[LONG_TOKENS] = TEST_SERVING_END_TOKEN;
		selected = script;
	}
	/* tool_suffix_reference: the same bytes under --dsml_parsing=reference, v0.0.5-alpha's rule */
	const bool reference =
		strcmp(argv[MODE_ARGUMENT], "tool_suffix_reference") == 0;
	struct chat_request fixture_defaults = defaults;
	fixture_defaults.is_calls_reference = reference;
	if (strcmp(argv[MODE_ARGUMENT], "tool_suffix") == 0 || reference) {
		const char *const close =
			strstr(TOOL_PREFIX, "\n</｜DSML｜ calls>");
		const size_t prefix = (size_t)(close - TOOL_PREFIX);
		static const char suffix[] = CUT_INVOKE;
		for (size_t index = 0; index < prefix + sizeof(suffix) - 1u;
		     index++)
			script[index] =
				(unsigned char)(index < prefix ?
							TOOL_PREFIX[index] :
							suffix[index - prefix]);
		script[prefix + sizeof(suffix) - 1u] = TEST_SERVING_END_TOKEN;
		selected = script;
	}
	if (strcmp(argv[MODE_ARGUMENT], "reasoning_literal") == 0 ||
	    strcmp(argv[MODE_ARGUMENT], "reasoning_call") == 0) {
		const char *const prefix =
			strcmp(argv[MODE_ARGUMENT], "reasoning_call") == 0 ?
				TOOL_PREFIX :
				"ordinary </think> quote ";
		for (size_t index = 0; index < strlen(prefix); index++)
			reasoning_script[reasoning_count++] =
				(unsigned char)prefix[index];
	} else if (strcmp(argv[MODE_ARGUMENT], "reasoning_invalid") == 0) {
		reasoning_script[0] = EMOJI_LEAD;
		reasoning_script[1] = 'r';
		reasoning_count = UTF8_TEST_EXTRA;
	} else if (strcmp(argv[MODE_ARGUMENT], "reasoning_utf8") == 0) {
		for (uint32_t index = 0; index < UTF8_TAIL; index++)
			reasoning_script[index] = 'r';
		reasoning_script[0] = EMOJI_LEAD;
		reasoning_script[1] = EMOJI_SECOND;
		reasoning_script[EMOJI_THIRD_INDEX] = EMOJI_THIRD;
		reasoning_script[EMOJI_FOURTH_INDEX] = EMOJI_LAST;
		reasoning_script[UTF8_TAIL] = EMOJI_LEAD;
		reasoning_script[UTF8_TAIL + 1u] = EMOJI_SECOND;
		reasoning_script[REGRESSION_BUDGET] = EMOJI_THIRD;
		reasoning_script[REGRESSION_BUDGET + 1u] = EMOJI_LAST;
		reasoning_count = REGRESSION_BUDGET + UTF8_TEST_EXTRA;
	}
	test_serving_set_reasoning_script(reasoning_script, reasoning_count);
	const uint32_t *const scripts[] = { selected };
	test_serving_set_sampling_model(
		strcmp(argv[MODE_ARGUMENT], "sampling") == 0);
	test_serving_set_reasoning_model(strncmp(argv[MODE_ARGUMENT],
						 "reasoning",
						 strlen("reasoning")) == 0);
	test_serving_set_engine(
		strcmp(argv[MODE_ARGUMENT], "reasoning_single") == 0 ?
			1u :
			TEST_SERVING_MOST_SLOTS,
		1, TEST_SERVING_ROWS,
		strcmp(argv[MODE_ARGUMENT], "reasoning_speculative") == 0 ?
			TOP_TWO :
			0);
	test_serving_set_step_delay(
		strcmp(argv[MODE_ARGUMENT], "slow") == 0 ? SLOW_STEP_NS : 0);
	struct inference_engine *engine = NULL;
	struct http_server *server = NULL;
	struct text_tokenizer *tokenizer = NULL;
	uint64_t table_bytes = 0;
	void *const tables = test_serving_tables(&table_bytes);
	const struct server_options options = {
		.bind_address = SERVER_DEFAULT_ADDRESS,
		.port = (uint32_t)port,
		.key_file = argc == KEY_ARGUMENTS ? argv[FIXTURE_ARGUMENTS] :
						    NULL,
		.anthropic_messages =
			strcmp(argv[MODE_ARGUMENT], "disabled") != 0 &&
			strcmp(argv[MODE_ARGUMENT], "metrics_only") != 0,
		.metrics = metrics,
		.queue_depth = QUEUE_DEPTH,
		.user_queue_depth = USER_QUEUE_DEPTH,
		.disable_prefix_reuse =
			strcmp(argv[MODE_ARGUMENT], "reasoning") != 0,
		.defaults = fixture_defaults
	};
	const bool opened =
		tables != NULL &&
		tokenizer_open(tables, table_bytes, &tokenizer) == STATUS_OK &&
		test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK &&
		server_open(&options, &server) == STATUS_OK;
	result_status status = STATUS_INVALID_ARGUMENT;
	if (opened) {
		if (strcmp(argv[MODE_ARGUMENT], "metrics_error") == 0)
			test_serving_set_fault_position(0);
		test_serving_set_step_delay(
			strcmp(argv[MODE_ARGUMENT], "slow") == 0 ?
				SLOW_STEP_NS :
				0);
		if (detached) {
			static const char invoke_end[] = "</｜DSML｜ invoke>";
			const char *const failing = strstr(tool, invoke_end);
			test_serving_set_tail_delay(
				failing != NULL ?
					(uint32_t)(failing - tool) +
						(uint32_t)strlen(invoke_end) :
					0,
				DETACHED_STEP_NS);
		}
		(void)puts("READY cpu-only");
		status = server_run(server, engine, tokenizer, STDIN_FILENO);
	}
	server_close(server);
	test_serving_close();
	tokenizer_close(tokenizer);
	free(tables);
	return status == STATUS_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
int main(int argc, char **argv)
{
	(void)setvbuf(stdout, NULL, _IONBF, 0);
	if ((argc == FIXTURE_ARGUMENTS || argc == KEY_ARGUMENTS) &&
	    strcmp(argv[1], "--client-fixture") == 0)
		return fixture(argc, argv);
	conversion_tests();
	stable_history_tests();
	signature_effort_tests();
	cut_turn_replay_tests();
	allocation_refusal();
	review_regressions();
	(void)printf("Messages conversion: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
