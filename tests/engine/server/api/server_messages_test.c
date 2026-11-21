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
		static const char suffix[] =
			"\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">bad";
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
	allocation_refusal();
	review_regressions();
	(void)printf("Messages conversion: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
