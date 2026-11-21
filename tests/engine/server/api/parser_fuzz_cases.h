/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Fixed corpus bytes; emitted under build/ by the seed runner. */
#ifndef PARSER_FUZZ_CASES_H
#define PARSER_FUZZ_CASES_H
#include <stddef.h>

#define FUZZ_HTTP 0
#define FUZZ_CHAT 1
#define FUZZ_TEMPLATE 2
#define FUZZ_JSON 3
#define FUZZ_MESSAGES 4

#ifndef PARSER_FUZZING

struct test_parser_seed {
	unsigned target;
	const char *bytes;
	size_t count;
};
static const char test_seed_0[] = "GET /health HTTP/1.1\r\nHost: test\r\n\r\n";
static const char test_seed_1[] =
	"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n"
	"Content-Length: 2\r\n\r\n{}";
static const char test_seed_2[] =
	"GET /health HTTP/1.0\r\nAuthorization: Bearer plac"
	"eholder\r\n\r\n";
static const char test_seed_3[] =
	"GET /health HTTP/1.1\r\nHost: test\r\nContent-Length"
	": 9999999999999999999\r\n\r\n";
static const char test_seed_4[] =
	"{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}";
static const char test_seed_5[] =
	"{\"messages\":[],\"seed\":18446744073709551616,\"temp"
	"erature\":1e999}";
static const char test_seed_6[] =
	"{\"messages\":[{\"role\":\"assistant\",\"content\":null,"
	"\"tool_calls\":[{\"id\":\"a\",\"type\":\"function\",\"funct"
	"ion\":{\"name\":\"test\",\"arguments\":\"{}\"}}]}]}";
static const char test_seed_7[] =
	"{\"messages\":[{\"role\":\"user\",\"content\":\"\\uD800\"}]"
	",\"messages\":null}";
static const char test_seed_8[] = "[{\"role\":\"user\",\"content\":\"hello\"}]";
static const char test_seed_9[] =
	"[{\"role\":\"system\",\"content\":\"test\"},{\"role\":\"ass"
	"istant\",\"content\":\"answer\",\"reasoning_content\":\""
	"think\"}]";
static const char test_seed_10[] =
	"[{\"role\":\"user\",\"content\":\"hello\"}]\000[{\"type\":\"fu"
	"nction\",\"function\":{\"name\":\"test\",\"parameters\":{"
	"\"type\":\"object\",\"properties\":{}}}}]";
static const char test_seed_11[] =
	"{\"a\":[null,true,false,NaN,Infinity,-Infinity,1e-"
	"100,0.000001],\"b\":\"\\uD834\\uDD1E\"}";
static const char test_seed_12[] = "{\"a\":1,\"\\u0061\":2}";
static const char test_seed_13[] =
	"GET /health HTTP/1.1\r\nHost: test\r\nAuthor"
	"ization: Bearer placeholder1\r\nAuthorizat"
	"ion: Bearer placeholder2\r\n\r\n";
static const char test_seed_14[] = "POST /v1/chat/completions HTTP/1.1\r\nHost"
				   ": test\r\nContent-Length: -1\r\n\r\n";
static const char test_seed_15[] = "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
				   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
				   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
				   "[[[[[[[[0]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
				   "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
				   "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
				   "]]]]]]]]]]]]]]]]]";
static const char test_seed_16[] = "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
				   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
				   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
				   "[[[[[[[[[0]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
				   "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
				   "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
				   "]]]]]]]]]]]]]]]]]]]";
static const char test_seed_17[] =
	"{\"messages\":[],\"ignored\":[[[[[[[[[[[[[[["
	"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
	"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
	"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[0]]]]]]]"
	"]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
	"]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
	"]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
	"}";
static const char test_seed_18[] =
	"[{\"role\":\"user\",\"content\":\"\\u0000\\uD834\\"
	"uDD1E\"}]";
static const char test_seed_19[] =
	"{\"model\":\"anemone\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\","
	"\"content\":\"hello\"}]}";
static const char test_seed_20[] =
	"{\"model\":\"anemone\",\"max_tokens\":64,\"tools\":[{\"name\":\"lookup\","
	"\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":"
	"\"string\"}}}}],\"messages\":[{\"role\":\"user\",\"content\":\"weather\"},"
	"{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":"
	"\"plan\",\"signature\":\"anemone-v1-effort-75\"},{\"type\":\"tool_use\","
	"\"id\":\"toolu_1\",\"name\":\"lookup\",\"input\":{\"city\":\"Rome\"}}]},"
	"{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":"
	"\"toolu_1\",\"content\":\"sunny\"}]}],\"thinking\":{\"type\":\"enabled\","
	"\"budget_tokens\":1024}}";
static const struct test_parser_seed test_parser_seeds[] = {
	{ FUZZ_HTTP, test_seed_0, sizeof(test_seed_0) - 1u },
	{ FUZZ_HTTP, test_seed_1, sizeof(test_seed_1) - 1u },
	{ FUZZ_HTTP, test_seed_2, sizeof(test_seed_2) - 1u },
	{ FUZZ_HTTP, test_seed_3, sizeof(test_seed_3) - 1u },
	{ FUZZ_CHAT, test_seed_4, sizeof(test_seed_4) - 1u },
	{ FUZZ_CHAT, test_seed_5, sizeof(test_seed_5) - 1u },
	{ FUZZ_CHAT, test_seed_6, sizeof(test_seed_6) - 1u },
	{ FUZZ_CHAT, test_seed_7, sizeof(test_seed_7) - 1u },
	{ FUZZ_TEMPLATE, test_seed_8, sizeof(test_seed_8) - 1u },
	{ FUZZ_TEMPLATE, test_seed_9, sizeof(test_seed_9) - 1u },
	{ FUZZ_TEMPLATE, test_seed_10, sizeof(test_seed_10) - 1u },
	{ FUZZ_JSON, test_seed_11, sizeof(test_seed_11) - 1u },
	{ FUZZ_JSON, test_seed_12, sizeof(test_seed_12) - 1u },
	{ FUZZ_HTTP, test_seed_13, sizeof(test_seed_13) - 1u },
	{ FUZZ_HTTP, test_seed_14, sizeof(test_seed_14) - 1u },
	{ FUZZ_JSON, test_seed_15, sizeof(test_seed_15) - 1u },
	{ FUZZ_JSON, test_seed_16, sizeof(test_seed_16) - 1u },
	{ FUZZ_CHAT, test_seed_17, sizeof(test_seed_17) - 1u },
	{ FUZZ_TEMPLATE, test_seed_18, sizeof(test_seed_18) - 1u },
	{ FUZZ_MESSAGES, test_seed_19, sizeof(test_seed_19) - 1u },
	{ FUZZ_MESSAGES, test_seed_20, sizeof(test_seed_20) - 1u },
};
#endif
#endif
