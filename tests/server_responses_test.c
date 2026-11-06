/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Responses conversion must render exactly the corresponding Chat history; SSE must reconstruct whole output. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/server.h"

#include "anemone/chat_template.h"
#include "anemone/hooks.h"
#include "anemone/tokenizer.h"
#include "server_responses.h"
#include "serving_fixture.h"

#define TEXT_BYTES 131072u
#define CREATED 1700000000
#define PROMPT_TOKENS 123u
#define CACHED_TOKENS 17u
#define OUTPUT_LIMIT 100u
#define REASONING_EFFORT 75u
#define EXPECTED_REASONS 2u
#define EXPECTED_OUTPUT_ITEMS 3u
#define BAD HTTP_BAD_REQUEST
#define CLIENT_FIXTURE_PORT 8337u
#define CLIENT_OUTPUT_FACTOR 4u
#define CLIENT_MODE_ARGUMENTS 2
#define MEMORY_FAULT_CASES 20u

static int failures;
static const struct chat_request defaults = { .max_tokens = OUTPUT_LIMIT,
					      .top_p = 1.0f,
					      .is_seeded = true };

static void expect(bool okay, const char *what)
{
	if (!okay) {
		(void)printf("FAILED: %s\n", what);
		failures++;
	}
}

static struct json_value member(struct json_value object, const char *name)
{
	char memory[TEXT_BYTES];
	struct json_scratch scratch = { memory, sizeof(memory), 0, false };
	struct json_value key = { 0 }, value = { 0 };
	for (size_t at = object.begin; json_next(object, &at, &key, &value);) {
		if (json_string_is(&scratch, key, name)) {
			return value;
		}
	}
	return (struct json_value){ 0 };
}

static bool is_string(struct json_value value, const char *word)
{
	char memory[TEXT_BYTES];
	struct json_scratch scratch = { memory, sizeof(memory), 0, false };
	return json_string_is(&scratch, value, word);
}

static void test_conversion(const struct text_tokenizer *tokenizer)
{
	static const struct {
		const char *responses, *chat;
	} cases[] = {
		{ "{\"input\":[{\"role\":\"assistant\",\"content\":[{\"type\":\"input_text\",\"text\":\"old reply\"}]},"
		  "{\"role\":\"user\",\"content\":\"continue\"}]}",
		  "{\"messages\":[{\"role\":\"assistant\",\"content\":\"old reply\"},"
		  "{\"role\":\"user\",\"content\":\"continue\"}]}" },

		{ "{\"input\":\"hello <think>\\n你好\",\"store\":false}",
		  "{\"messages\":[{\"role\":\"user\",\"content\":\"hello <think>\\n你好\"}]}" },
		{ "{\"instructions\":\"s\",\"input\":[{\"role\":\"developer\",\"content\":\"d\"},"
		  "{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"a\"},"
		  "{\"type\":\"input_text\",\"text\":\"b\"}]}],\"max_output_tokens\":9,\"reasoning\":{\"effort"
		  "\":\"high\"}}",
		  "{\"messages\":[{\"role\":\"system\",\"content\":\"s\"},{\"role\":\"developer\",\"content\":\"d\"},"
		  "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\",\"tex"
		  "t\":\"b\"}]}],"
		  "\"max_tokens\":9,\"reasoning_effort\":\"high\"}" },
		{ "{\"input\":[{\"role\":\"user\",\"content\":\"hi\"},{\"type\":\"reasoning\",\"id\":\"ignored\","
		  "\"summary\":[{\"type\":\"summary_text\",\"text\":\"r\"}],\"encrypted_content\":null},"
		  "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text"
		  "\":\"checking\"}]},"
		  "{\"type\":\"function_call\",\"call_id\":\"call1\",\"name\":\"exec\",\"namespace\":\"functions\","
		  "\"arguments\":\"{\\\"cmd\\\":\\\"pwd\\\"}\"},{\"type\":\"function_call\",\"call_id\":\"call2\","
		  "\"name\":\"read\",\"arguments\":\"{}\"},{\"type\":\"function_call_output\",\"call_id\":\"call2\","
		  "\"output\":\"ok2\"},{\"type\":\"function_call_output\",\"call_id\":\"call1\",\"output\":\"ok1\"}],"
		  "\"tools\":[{\"type\":\"namespace\",\"name\":\"functions\",\"tools\":[{\"type\":\"function\""
		  ",\"name\":\"exec\","
		  "\"description\":\"run\",\"parameters\":{\"type\":\"object\",\"properties\":{}},\"strict\":false}]}],"
		  "\"include\":[\"reasoning.encrypted_content\"],\"text\":{\"format\":{\"type\":\"text\"}},\"s"
		  "tore\":false}",
		  "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":"
		  "\"checking\","
		  "\"reasoning_content\":\"r\",\"tool_calls\":[{\"id\":\"call1\",\"type\":\"function\",\"function\":{"
		  "\"name\":\"functions::exec\",\"arguments\":\"{\\\"cmd\\\":\\\"pwd\\\"}\"}},{\"id\":\"call2"
		  "\",\"type\":\"function\","
		  "\"function\":{\"name\":\"read\",\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"tool_call_id"
		  "\":\"call2\","
		  "\"content\":\"ok2\"},{\"role\":\"tool\",\"tool_call_id\":\"call1\",\"content\":\"ok1\"}],\""
		  "tools\":[{\"type\":"
		  "\"function\",\"function\":{\"name\":\"functions::exec\",\"description\":\"run\",\"parameter"
		  "s\":{\"type\":\"object\","
		  "\"properties\":{}},\"strict\":false}}],\"response_format\":{\"type\":\"text\"}}" }
	};
	static char rendered[TEXT_BYTES], equivalent[TEXT_BYTES];
	static uint32_t tokens[TEXT_BYTES], other_tokens[TEXT_BYTES];
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		struct chat_buffer body = { 0 };
		struct chat_request responses = { 0 }, chat = { 0 };
		struct chat_refusal refusal = { 0 };
		expect(server_responses_read(
			       cases[index].responses,
			       strlen(cases[index].responses), "model", NULL,
			       &defaults, &responses, &body, &refusal) == 0 &&
			       responses.is_responses,
		       "convert valid full history");
		expect(chat_request_read(cases[index].chat,
					 strlen(cases[index].chat), "model",
					 NULL, &defaults, &chat, &refusal) == 0,
		       "equivalent Chat valid");
		expect(responses.reasoning_effort == chat.reasoning_effort &&
			       responses.max_tokens == chat.max_tokens,
		       "generation settings preserved");
		const struct chat_template_options options = {
			.reasoning_effort =
				responses.reasoning_effort != 0 ?
					responses.reasoning_effort :
					CHAT_TEMPLATE_HIGHEST_EFFORT,
			.is_thinking = responses.reasoning_effort != 0,
			.drops_thinking = true
		};
		struct chat_template_spans spans = { 0 }, other = { 0 };
		size_t count = 0, other_count = 0;
		const char *const reference = cases[index].chat;
		expect(chat_template_render_spans(
			       body.bytes + responses.messages_begin,
			       responses.messages_end -
				       responses.messages_begin,
			       responses.tools_end > responses.tools_begin ?
				       body.bytes + responses.tools_begin :
				       NULL,
			       responses.tools_end - responses.tools_begin,
			       &options, rendered, sizeof(rendered), &count,
			       &spans) == 0,
		       "Responses renders");
		expect(chat_template_render_spans(
			       reference + chat.messages_begin,
			       chat.messages_end - chat.messages_begin,
			       chat.tools_end > chat.tools_begin ?
				       reference + chat.tools_begin :
				       NULL,
			       chat.tools_end - chat.tools_begin, &options,
			       equivalent, sizeof(equivalent), &other_count,
			       &other) == 0,
		       "Chat renders");
		expect(count == other_count &&
			       memcmp(rendered, equivalent, count) == 0,
		       "rendered prompt bytes identical");
		uint32_t token_count = 0, other_token_count = 0;
		expect(tokenizer_encode_spans(tokenizer, rendered, count,
					      spans.spans, spans.count, tokens,
					      TEXT_BYTES, &token_count) == 0 &&
			       tokenizer_encode_spans(
				       tokenizer, equivalent, other_count,
				       other.spans, other.count, other_tokens,
				       TEXT_BYTES, &other_token_count) == 0,
		       "both tokenize");
		expect(token_count == other_token_count &&
			       memcmp(tokens, other_tokens,
				      (size_t)token_count * sizeof(*tokens)) ==
				       0,
		       "trusted token vectors identical");
		free(spans.spans);
		free(other.spans);
		chat_buffer_free(&body);
	}
}

static void test_refusals(void)
{
	static const struct {
		const char *body, *param;
	} cases[] = {
		{ "{\"input\":\"x\",\"metadata\":{\"value\":[NaN]}}",
		  "metadata" },
		{ "{\"input\":\"x\",\"metadata\":{\"value\":-Infinity}}",
		  "metadata" },
		{ "{\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"f\","
		  "\"parameters\":{\"default\":Infinity}}]}",
		  "tools" },

		{ "{\"input\":\"x\",\"text\":{\"verbosity\":\"high\"}}",
		  "text" },
		{ "{\"input\":\"x\",\"reasoning\":{\"summary\":\"concise\"}}",
		  "reasoning" },
		{ "{\"input\":\"x\",\"reasoning\":{\"unknown\":1}}",
		  "reasoning" },
		{ "{\"input\":\"x\",\"text\":{\"unknown\":1}}", "text" },
		{ "{\"input\":\"x\",\"stream_options\":{\"include_obfuscation\":0}}",
		  "stream_options" },
		{ "{\"input\":\"x\",\"stream_options\":{\"include_usage\":\"yes\"}}",
		  "stream_options" },

		{ "{\"input\":\"x\",\"store\":true}", "store" },
		{ "{\"input\":\"x\",\"store\":0}", "store" },
		{ "{\"input\":\"x\",\"previous_response_id\":\"resp_1\"}",
		  "previous_response_id" },
		{ "{\"input\":\"x\",\"previous_response_id\":\"\"}",
		  "previous_response_id" },
		{ "{\"input\":\"x\",\"conversation\":\"c\"}", "conversation" },
		{ "{\"input\":\"x\",\"max_output_tokens\":-1}",
		  "max_output_tokens" },
		{ "{\"input\":\"x\",\"text\":{\"format\":{\"type\":\"json_schema\"}}}",
		  "text" },
		{ "{\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"a\",\"strict\":true}]}",
		  "tools" },
		{ "{\"input\":\"x\",\"background\":true}", "background" },
		{ "{\"input\":\"x\",\"truncation\":\"auto\"}", "truncation" },
		{ "{\"input\":\"x\",\"stream_options\":{\"include_obfuscation\":true}}",
		  "stream_options" },
		{ "{\"input\":\"x\",\"model\":\"other\"}", "model" },
		{ "{\"input\":[{\"type\":\"item_reference\",\"id\":\"m\"}]}",
		  NULL },
		{ "{\"input\":[{\"type\":\"reasoning\",\"encrypted_content\":\"opaque\"}]}",
		  NULL },
		{ "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"x\"}]}]}",
		  NULL },
		{ "{\"input\":\"x\",\"input\":\"y\"}", NULL },
		{ "{\"input\":\"x\",\"metadata\":{\"a\":1,\"a\":2}}", NULL }
	};
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		struct chat_buffer converted = { 0 }, wire = { 0 };
		struct chat_request request = { 0 };
		struct chat_refusal refusal = { 0 };
		const uint32_t status = server_responses_read(
			cases[index].body, strlen(cases[index].body), "model",
			NULL, &defaults, &request, &converted, &refusal);
		expect(status == BAD || status == HTTP_NOT_FOUND,
		       "unsupported/stateful input refused");
		chat_refusal_write(&wire, &refusal, cases[index].body,
				   CHAT_ROUTE_RESPONSES);
		const char *const head = memmem(wire.bytes, wire.count,
						"\r\n\r\n", strlen("\r\n\r\n"));
		struct json_value doc = { 0 };
		expect(head != NULL &&
			       json_read_document(
				       head + strlen("\r\n\r\n"),
				       wire.count -
					       (size_t)(head +
							strlen("\r\n\r\n") -
							wire.bytes),
				       &doc) == JSON_DOCUMENT_VALID,
		       "error JSON uses original-body spans");
		const struct json_value param =
			member(member(doc, "error"), "param");
		expect(cases[index].param == NULL ||
			       is_string(param, cases[index].param),
		       "correct original parameter named");
		chat_buffer_free(&wire);
		chat_buffer_free(&converted);
	}
}

static void test_echo_fields(void)
{
	static const char body[] =
		"{\"input\":\"x\",\"instructions\":\"say \\\"hi\\\"\",\"metadata\":{\"session_id\":\"owned\"},"
		"\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{}}],\"tool_choice\":\"none\","
		"\"temperature\":0.25,\"top_p\":0.5}";
	struct chat_buffer converted = { 0 }, fields = { 0 };
	struct chat_request request = { 0 };
	struct chat_refusal refusal = { 0 };
	expect(server_responses_read_fields(body, strlen(body), "model", NULL,
					    &defaults, &request, &converted,
					    &fields, &refusal) == 0 &&
		       fields.count != 0,
	       "read owned echo fields");
	// Original and converted histories can be erased before generation; required echoes remain independently owned.
	chat_buffer_free(&converted);
	struct chat_answer answer = { 0 };
	answer.pending_responses_fields = &fields;
	chat_answer_begin(&answer, &request, "model", "resp-echo", CREATED,
			  PROMPT_TOKENS);
	expect(answer.pending_responses_fields == NULL,
	       "begin consumes pending echoes");
	(void)chat_answer_token(&answer, "x", 1);
	chat_answer_finish(&answer, "stop");
	const char *head = memmem(answer.output.bytes, answer.output.count,
				  "\r\n\r\n", strlen("\r\n\r\n"));
	struct json_value doc = { 0 };
	expect(head != NULL &&
		       json_read_document(head + strlen("\r\n\r\n"),
					  answer.output.count -
						  (size_t)(head +
							   strlen("\r\n\r\n") -
							   answer.output.bytes),
					  &doc) == JSON_DOCUMENT_VALID,
	       "echo response JSON valid");
	expect(is_string(member(doc, "instructions"), "say \"hi\"") &&
		       is_string(member(doc, "tool_choice"), "none") &&
		       is_string(member(member(doc, "metadata"), "session_id"),
				 "owned"),
	       "instructions/metadata/choice echoed");
	const struct json_value tools = member(doc, "tools");
	size_t cursor = tools.begin;
	struct json_value tool = { 0 };
	expect(json_next(tools, &cursor, NULL, &tool) &&
		       is_string(member(tool, "name"), "f"),
	       "tools echoed");
	expect(member(doc, "temperature").text != NULL &&
		       member(doc, "top_p").text != NULL,
	       "sampling settings present");
	chat_answer_free(&answer);
	chat_buffer_free(&fields);
}

static void test_answers(void)
{
	static const char call[] =
		"\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"functions::exec\">\n"
		"<｜DSML｜ parameter name=\"cmd\" string=\"true\">pwd</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜D"
		"SML｜ calls>";
	for (uint32_t streaming = 0; streaming < EXPECTED_REASONS;
	     streaming++) {
		struct chat_request request = defaults;
		request.is_responses = true;
		request.is_streamed = streaming != 0;
		request.reasoning_effort = REASONING_EFFORT;
		request.tools_declared = true;
		request.tools_end = 1;
		struct chat_answer answer = { 0 };
		chat_answer_begin(&answer, &request, "model", "resp-test",
				  CREATED, PROMPT_TOKENS);
		answer.cached_token_count = CACHED_TOKENS;
		expect(!chat_answer_token(&answer, "r", 1), "reasoning text");
		answer.reasoning_token_count = EXPECTED_REASONS;
		expect(!chat_answer_token(&answer, "</think>hello",
					  (uint32_t)strlen("</think>hello")),
		       "content follows reasoning");
		expect(chat_answer_token(&answer, call, (uint32_t)strlen(call)),
		       "closed tool block stops");
		expect(answer.is_finished && answer.call_count == 1 &&
			       !answer.output.is_short,
		       "one call, complete output");
		if (answer.output.bytes == NULL) {
			(void)printf(
				"formatter empty: finished=%d short=%d responses=%d streamed=%d calls=%u\n",
				answer.is_finished, answer.output.is_short,
				request.is_responses, request.is_streamed,
				answer.call_count);
			chat_answer_free(&answer);
			continue;
		}
		expect(memmem(answer.output.bytes, answer.output.count, "DSML",
			      strlen("DSML")) == NULL,
		       "no raw tool syntax");
		const char *const marker =
			streaming != 0 ? "event: response.completed\ndata: " :
					 "\r\n\r\n";
		const char *const at = memmem(answer.output.bytes,
					      answer.output.count, marker,
					      strlen(marker));
		struct json_value doc = { 0 };
		const size_t begin = at != NULL ?
					     (size_t)(at + strlen(marker) -
						      answer.output.bytes) :
					     0;
		expect(at != NULL &&
			       json_read_document(answer.output.bytes + begin,
						  answer.output.count - begin,
						  &doc) == JSON_DOCUMENT_VALID,
		       "complete response/event valid JSON");
		const struct json_value response =
			streaming != 0 ? member(doc, "response") : doc;
		expect(is_string(member(response, "status"), "completed"),
		       "tool handoff is completed");
		const struct json_value output = member(response, "output");
		struct json_value item = { 0 };
		size_t cursor = output.begin, count = 0;
		while (json_next(output, &cursor, NULL, &item)) {
			count++;
		}
		expect(count == EXPECTED_OUTPUT_ITEMS &&
			       is_string(member(item, "type"),
					 "function_call") &&
			       is_string(member(item, "name"), "exec") &&
			       is_string(member(item, "namespace"),
					 "functions"),
		       "reasoning, message and qualified function call retained");
		if (streaming != 0) {
			expect(memmem(answer.output.bytes, answer.output.count,
				      "event: response.function_call_arguments.delta",
				      strlen("event: response.function_call_arguments.delta")) !=
				       NULL,
			       "native argument event");
			expect(memmem(answer.output.bytes, answer.output.count,
				      "[DONE]", strlen("[DONE]")) == NULL,
			       "Responses has its own terminal event");
		}
		chat_answer_free(&answer);
		request.tools_declared = false;
		request.tools_end = 0;
		chat_answer_begin(&answer, &request, "model", "resp-limit",
				  CREATED, PROMPT_TOKENS);
		expect(!chat_answer_token(&answer, "x", 1),
		       "limited reasoning");
		chat_answer_finish(&answer, "length");
		expect(memmem(answer.output.bytes, answer.output.count,
			      "\"reason\":\"max_output_tokens\"",
			      strlen("\"reason\":\"max_output_tokens\"")) !=
			       NULL,
		       "token limit signals incomplete");
		chat_answer_free(&answer);
		chat_answer_begin(&answer, &request, "model", "resp-fail",
				  CREATED, PROMPT_TOKENS);
		(void)chat_answer_token(&answer, "partial",
					(uint32_t)strlen("partial"));
		chat_answer_fail(&answer);
		expect(answer.is_finished &&
			       memmem(answer.output.bytes, answer.output.count,
				      streaming != 0 ? "response.failed" :
						       "500 Internal",
				      strlen(streaming != 0 ?
						     "response.failed" :
						     "500 Internal")) != NULL,
		       "engine failures preserve Responses or HTTP error format");
		if (streaming != 0) {
			expect(memmem(answer.output.bytes, answer.output.count,
				      "\"type\":\"reasoning\",\"status\":\"incomplete\"",
				      strlen("\"type\":\"reasoning\",\"status\":\"incomplete\"")) !=
				       NULL,
			       "failed response marks partial items incomplete");
		}
		chat_answer_free(&answer);
	}
}

static void test_memory_failures(void)
{
#if CHECKING_HOOKS_ENABLED
	static const char body[] =
		"{\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{}}]}";
	for (uint32_t fault = 1; fault <= MEMORY_FAULT_CASES; fault++) {
		struct chat_buffer converted = { 0 }, fields = { 0 };
		struct chat_request request = { 0 };
		struct chat_refusal refusal = { 0 };
		expect(fault_arm("chat_memory", fault),
		       "arm conversion memory failure");
		const uint32_t status = server_responses_read_fields(
			body, strlen(body), "model", NULL, &defaults, &request,
			&converted, &fields, &refusal);
		const uint64_t reached = fault_disarm();
		expect(status == (reached >= fault ? HTTP_SERVER_ERROR : 0),
		       "conversion fails closed on memory loss");
		chat_buffer_free(&converted);
		chat_buffer_free(&fields);
		request = defaults;
		request.is_responses = true;
		request.is_streamed = true;
		struct chat_answer answer = { 0 };
		expect(fault_arm("chat_memory", fault),
		       "arm formatter memory failure");
		chat_answer_begin(&answer, &request, "model", "resp-fault",
				  CREATED, PROMPT_TOKENS);
		(void)chat_answer_token(&answer, "x", 1);
		chat_answer_finish(&answer, "stop");
		const uint64_t writes = fault_disarm();
		expect(answer.output.is_short == (writes >= fault),
		       "formatter marks memory failure unsendable");
		chat_answer_free(&answer);
	}
#endif
}

/* Optional external-client mode uses the same CPU engine as the serving suite, never a CUDA backend. */
static int client_fixture(void)
{
	static const char tool_text[] =
		"\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"functions::exec_command\">\n"
		"<｜DSML｜ parameter name=\"cmd\" string=\"true\">printf codex_fixture</｜DSML｜ parameter>\n"
		"</｜DSML｜ invoke>\n</｜DSML｜ calls>";
	uint32_t tool[sizeof(tool_text) + EXPECTED_REASONS];
	tool[0] = 'r';
	tool[1] = TEST_SERVING_REASONING_END_TOKEN;
	for (size_t index = 0; index < strlen(tool_text); index++) {
		tool[index + EXPECTED_REASONS] =
			(unsigned char)tool_text[index];
	}
	tool[strlen(tool_text) + EXPECTED_REASONS] = TEST_SERVING_END_TOKEN;
	static const uint32_t final[] = { 's',
					  TEST_SERVING_REASONING_END_TOKEN,
					  'd',
					  'o',
					  'n',
					  'e',
					  TEST_SERVING_END_TOKEN };
	const uint32_t *const scripts[] = { tool, final };
	struct inference_engine *engine = NULL;
	struct http_server *server = NULL;
	struct text_tokenizer *tokenizer = NULL;
	uint64_t table_bytes = 0;
	void *const tables = test_serving_tables(&table_bytes);
	const struct server_options options = {
		.bind_address = SERVER_DEFAULT_ADDRESS,
		.port = CLIENT_FIXTURE_PORT,
		.disable_prefix_reuse = true,
		.defaults = { .max_tokens = OUTPUT_LIMIT * CLIENT_OUTPUT_FACTOR,
			      .top_p = 1.0f,
			      .is_seeded = true }
	};
	const bool opened =
		tables != NULL &&
		tokenizer_open(tables, table_bytes, &tokenizer) == 0 &&
		test_serving_open(TEST_SERVING_STATE_POSITIONS, scripts,
				  EXPECTED_REASONS, &engine) == 0 &&
		server_open(&options, &server) == 0;
	result_status status = STATUS_INVALID_ARGUMENT;
	if (opened) {
		(void)puts("READY cpu-fixture port=8337");
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
	if (argc == CLIENT_MODE_ARGUMENTS &&
	    strcmp(argv[1], "--client-fixture") == 0) {
		return client_fixture();
	}
	uint64_t table_bytes = 0;
	void *const tables = test_serving_tables(&table_bytes);
	struct text_tokenizer *tokenizer = NULL;
	expect(tables != NULL &&
		       tokenizer_open(tables, table_bytes, &tokenizer) == 0,
	       "host tokenizer");
	if (tokenizer != NULL) {
		test_conversion(tokenizer);
	}
	test_refusals();
	test_echo_fields();
	test_answers();
	test_memory_failures();
	tokenizer_close(tokenizer);
	free(tables);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
