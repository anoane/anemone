/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Namespace parsing and free-slot affinity at fixed times; no model or device required. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "server_internal.h"
#include "server_responses.h"

#define REQUEST_BYTES 16384u
#define TEST_TOKEN_COUNT 16u
#define EXACT_PROMPT_TOKENS (TEST_TOKEN_COUNT - 1u)
#define TEST_CONTEXT 32u
#define TEST_SLOTS 2u
#define TEST_GRACE 30u
#define TEST_NOW 100.0
#define TEST_RECENT 99.0
#define TEST_OLD 1.0
#define TEST_NAMESPACE_PREFIX_BYTES 2u
#define TEST_NAMESPACE_TAG_BYTES 3u
#define TEST_APPENDED_FIELD_BYTES (TEST_NAMESPACE_TAG_BYTES + 1u)
#define TEST_INVALID_UTF8_LEAD 0xffu

struct identity_case {
	const char *first_members, *second_members;
	const char *first_headers, *second_headers;
};

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: server cache: %s\n", message);
		failures++;
	}
}

static bool identity_read(struct http_server *server,
			  struct connection *connection, const char *user,
			  const char *members, const char *headers)
{
	char body[REQUEST_BYTES], input[REQUEST_BYTES];
	const bool responses = strstr(members, "\"client_metadata\"") != NULL;
	(void)snprintf(body, sizeof(body),
		       "{\"%s\":[{\"role\":\"user\",\"content\":\"hi\"}]%s}",
		       responses ? "input" : "messages", members);
	const int bytes = snprintf(
		input, sizeof(input),
		"POST %s HTTP/1.1\r\nHost: local\r\n%sContent-Length: %zu\r\n\r\n%s",
		responses ? "/v1/responses" : "/v1/chat/completions", headers,
		strlen(body), body);
	expect(bytes > 0 && (size_t)bytes < sizeof(input), "test request fits");
	connection->input.bytes = input;
	connection->input.count = (size_t)bytes;
	(void)snprintf(connection->live.user, sizeof(connection->live.user),
		       "%s", user);
	expect(server_head_read(input, (size_t)bytes, &connection->head) &&
		       connection->head.refusal == 0,
	       "test head validates");
	const struct chat_request defaults = { .is_seeded = true,
					       .top_p = 1.0f,
					       .max_tokens = 1 };
	struct chat_refusal refusal = { 0 };
	struct chat_buffer converted = { 0 }, fields = { 0 };
	const uint32_t status =
		responses ? server_responses_read_fields(
				    body, strlen(body), "model", NULL,
				    &defaults, &connection->request, &converted,
				    &fields, &refusal) :
			    chat_request_read(body, strlen(body), "model", NULL,
					      &defaults, &connection->request,
					      &refusal);
	const bool valid =
		status == 0 &&
		server_cache_namespace_read(server, connection, &refusal) == 0;
	chat_buffer_free(&converted);
	chat_buffer_free(&fields);
	connection->input = (struct chat_buffer){ 0 };
	return valid;
}

static void test_identity(struct http_server *server, struct connection *first,
			  struct connection *second)
{
	expect(identity_read(server, first, "alice",
			     ",\"session_id\":\"A\",\"cache_salt\":\"salt\"",
			     ""),
	       "session and salt accepted");
	expect(identity_read(
		       server, second, "alice",
		       ",\"cache_salt\":\"salt\",\"session_id\":\"\\u0041\"",
		       "") &&
		       server_cache_namespace_equal(&first->cache_namespace,
						    &second->cache_namespace),
	       "member order and equivalent Unicode escapes preserve identity");
	expect(identity_read(server, second, "bob",
			     ",\"session_id\":\"A\",\"cache_salt\":\"salt\"",
			     "") &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace),
	       "the authenticated user cannot be overridden by client identity");
	expect(identity_read(server, second, "alice",
			     ",\"session_id\":\"B\",\"cache_salt\":\"salt\"",
			     "") &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace),
	       "the same salt does not merge distinct sessions");
	expect(identity_read(server, first, "alice",
			     ",\"metadata\":{\"session_id\":\"A\",\"trace\":1}",
			     "X-Session-Id: A\r\nX-Client-Request-Id: one\r\n"),
	       "metadata and header identity");
	expect(identity_read(
		       server, second, "alice",
		       ",\"metadata\":{\"trace\":2,\"session_id\":\"A\"}",
		       "x-client-request-id: two\r\nx-session-id: A\r\n") &&
		       server_cache_namespace_equal(&first->cache_namespace,
						    &second->cache_namespace),
	       "header case/order, request IDs and unrelated metadata do not split the namespace");
	expect(identity_read(server, second, "alice",
			     ",\"metadata\":{\"session_id\":\"B\"}",
			     "x-session-id: A\r\n") &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace),
	       "all supplied session identities participate");
	expect(!identity_read(server, second, "alice", ",\"session_id\":42",
			      ""),
	       "numeric identity refused");
	expect(!identity_read(server, second, "alice",
			      ",\"metadata\":{\"session_id\":[]}", ""),
	       "array metadata identity refused");
	expect(!identity_read(server, second, "alice", "",
			      "x-session-id: A\r\nX-Session-Id: A\r\n"),
	       "duplicate identity headers refused even when their values match");
	expect(identity_read(server, first, "alice", "", "") &&
		       identity_read(
			       server, second, "alice",
			       ",\"session_id\":null,\"prompt_cache_key\":\"\"",
			       "") &&
		       server_cache_namespace_equal(&first->cache_namespace,
						    &second->cache_namespace) &&
		       first->cache_namespace.has_session == 0,
	       "missing, empty and null signals retain per-user scope");
	server->cache_scope = SERVER_CACHE_USER;
	expect(identity_read(server, first, "alice", ",\"session_id\":\"A\"",
			     "") &&
		       identity_read(server, second, "alice",
				     ",\"session_id\":\"B\"", "") &&
		       server_cache_namespace_equal(&first->cache_namespace,
						    &second->cache_namespace),
	       "the user-only fallback is selectable");
	server->cache_scope = SERVER_CACHE_GLOBAL;
	expect(identity_read(server, first, "alice", ",\"session_id\":\"A\"",
			     "") &&
		       identity_read(server, second, "bob",
				     ",\"session_id\":\"B\"", "") &&
		       server_cache_namespace_equal(&first->cache_namespace,
						    &second->cache_namespace),
	       "the legacy global fallback is explicit and selectable");
	server->cache_scope = SERVER_CACHE_SESSION;
	first->authenticated_user = true;
	expect(identity_read(server, first, "-", "", "") &&
		       identity_read(server, second, "-", "", "") &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace),
	       "authenticated legacy usernames cannot import an anonymous namespace after a restart");
	first->authenticated_user = false;
	expect(identity_read(server, first, "alice",
			     ",\"user_id\":\"deepseek-a\"", "") &&
		       identity_read(server, second, "alice",
				     ",\"user_id\":\"deepseek-b\"", "") &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace),
	       "DeepSeek's top-level user_id isolates end users behind the same API user");
	expect(identity_read(server, first, "alice",
			     ",\"metadata\":{\"user_id\":\"a\"}", "") &&
		       identity_read(server, second, "alice",
				     ",\"metadata\":{\"user_id\":\"b\"}", "") &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace),
	       "DeepSeek's Anthropic-style metadata.user_id is also recognized");
	char maximum[SERVER_CACHE_FIELD_BYTES + 1u], members[REQUEST_BYTES];
	memset(maximum, 'a', SERVER_CACHE_FIELD_BYTES);
	maximum[SERVER_CACHE_FIELD_BYTES] = '\0';
	(void)snprintf(members, sizeof(members), ",\"prompt_cache_key\":\"%s\"",
		       maximum);
	expect(identity_read(server, first, "alice", members, ""),
	       "the maximum cache key is accepted without truncation");
	(void)snprintf(members, sizeof(members),
		       ",\"prompt_cache_key\":\"%sa\"", maximum);
	expect(!identity_read(server, second, "alice", members, ""),
	       "an oversized key is refused without truncation");
	expect(identity_read(server, first, "alice",
			     ",\"session_id\":\"A\\u0000B\"", "") &&
		       identity_read(server, second, "alice",
				     ",\"session_id\":\"A\"", "") &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace),
	       "embedded zero bytes cannot collapse distinct identities");
}

static void test_selection(struct http_server *server, struct connection *first,
			   struct connection *second)
{
	uint32_t tokens[TEST_TOKEN_COUNT],
		histories[TEST_SLOTS * TEST_CONTEXT] = { 0 }, kept = 0;
	for (uint32_t index = 0; index < TEST_TOKEN_COUNT; index++) {
		tokens[index] = index;
	}
	server->description.session_slot_count = TEST_SLOTS;
	server->description.context_token_limit = TEST_CONTEXT;
	server->histories = histories;
	server->session_grace_seconds = TEST_GRACE;
	expect(identity_read(server, first, "alice", ",\"session_id\":\"A\"",
			     "") &&
		       identity_read(server, second, "bob",
				     ",\"session_id\":\"A\"", ""),
	       "selection fixture identities");
	first->token_ids = tokens;
	first->token_count = TEST_TOKEN_COUNT;
	first->waiting_snapshot_slot = NO_WAITING_SNAPSHOT;
	memcpy(histories, tokens, sizeof(tokens));
	server->slot_histories[0] =
		(struct slot_history){ .token_count = TEST_TOKEN_COUNT,
				       .processed_count = TEST_TOKEN_COUNT,
				       .reach_count = TEST_TOKEN_COUNT,
				       .cache_namespace =
					       second->cache_namespace,
				       .last_active_seconds = TEST_RECENT };
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) ==
			       1 &&
		       kept == 0,
	       "another user's identical prefix is neither reused nor evicted while an empty slot exists");
	server->slot_histories[0].cache_namespace = first->cache_namespace;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) ==
			       0 &&
		       kept > 0,
	       "the same session keeps its valid history");
	server->slot_histories[0].cache_namespace = second->cache_namespace;
	server->slot_histories[1] = server->slot_histories[0];
	server->slot_histories[1].last_active_seconds = TEST_OLD;
	server->slot_histories[1].last_used = 1;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) ==
			       1 &&
		       kept == 0,
	       "grace protects the recent slot despite its older LRU ordinal");
	server->session_grace_seconds = 0;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) ==
			       0 &&
		       kept == 0,
	       "zero grace restores LRU, without waiving namespace isolation");
	server->session_grace_seconds = TEST_GRACE;
	server->slot_histories[1].last_active_seconds = TEST_RECENT;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) ==
			       0 &&
		       kept == 0,
	       "all protected free slots still permit dispatch");
	server->active[0].token_ids = tokens;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) == 1,
	       "active slots cannot be evicted by affinity");
	server->active[0].token_ids = NULL;
	server->slot_histories[1].cache_namespace = first->cache_namespace;
	first->waiting_snapshot_slot = WAITING_SNAPSHOT_READY + 1u;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) == 1,
	       "a completed snapshot prerequisite retains its granted slot");
	first->token_ids = NULL;
	server->histories = NULL;
}

static void test_exact_selection(struct http_server *server,
				 struct connection *first,
				 struct connection *second)
{
	uint32_t tokens[EXACT_PROMPT_TOKENS],
		histories[TEST_SLOTS * TEST_CONTEXT] = { 0 }, kept = 0;
	for (uint32_t index = 0; index < EXACT_PROMPT_TOKENS; index++) {
		tokens[index] = index;
	}
	expect(identity_read(server, first, "alice", ",\"session_id\":\"A\"",
			     "") &&
		       identity_read(server, second, "alice",
				     ",\"session_id\":\"B\"", ""),
	       "combined policy identities");
	server->histories = histories;
	server->session_grace_seconds = 0;
	server->exact_prefix_reuse = SERVER_PREFIX_REPEAT;
	memset(server->slot_histories, 0, sizeof(server->slot_histories));
	first->token_ids = tokens;
	first->token_count = EXACT_PROMPT_TOKENS;
	first->waiting_snapshot_slot = NO_WAITING_SNAPSHOT;
	memcpy(histories, tokens, sizeof(tokens));
	memcpy(histories + TEST_CONTEXT, tokens, sizeof(tokens));
	server->slot_histories[0] =
		(struct slot_history){ .token_count = EXACT_PROMPT_TOKENS,
				       .processed_count = EXACT_PROMPT_TOKENS,
				       .reach_count = EXACT_PROMPT_TOKENS,
				       .cold_prompt_count = EXACT_PROMPT_TOKENS,
				       .cache_namespace =
					       second->cache_namespace };
	server->slot_histories[1].cache_namespace = first->cache_namespace;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) ==
			       1 &&
		       kept == 0,
	       "cold-plan provenance cannot admit an identical prefix from another session");
	server->slot_histories[1] = server->slot_histories[0];
	server->slot_histories[1].cache_namespace = first->cache_namespace;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) ==
			       1 &&
		       kept == EXACT_PROMPT_TOKENS - 1u,
	       "matching namespace and cold provenance both permit a closed repeat");
	server->slot_histories[1].cold_prompt_count = 0;
	(void)server_chosen_session_slot(server, first, TEST_NOW, &kept);
	expect(kept == 0,
	       "namespace affinity cannot waive missing cold provenance");
	server->slot_histories[1].cold_prompt_count = EXACT_PROMPT_TOKENS;
	server->slot_histories[1].from_snapshot = true;
	(void)server_chosen_session_slot(server, first, TEST_NOW, &kept);
	expect(kept == 0,
	       "namespace affinity cannot make imported state numerically eligible");
	expect(identity_read(server, second, "bob", ",\"session_id\":\"A\"",
			     ""),
	       "combined policy second API user");
	server->slot_histories[0].cache_namespace = second->cache_namespace;
	first->waiting_snapshot_slot = WAITING_SNAPSHOT_READY;
	expect(server_chosen_session_slot(server, first, TEST_NOW, &kept) ==
			       1 &&
		       kept == 0,
	       "a ready slot from another user cannot bypass namespace and numerical gates");
	first->token_ids = NULL;
	server->histories = NULL;
	server->exact_prefix_reuse = SERVER_PREFIX_LEGACY;
}

static void test_fork_owners(struct http_server *server,
			     struct connection *first,
			     struct connection *second)
{
	static const struct identity_case hard[] = {
		{ ",\"user\":\"one\"", ",\"user\":\"two\"", "", "" },
		{ ",\"safety_identifier\":\"one\"",
		  ",\"safety_identifier\":\"two\"", "", "" },
		{ ",\"cache_salt\":\"one\"", ",\"cache_salt\":\"two\"", "",
		  "" },
		{ ",\"user_id\":\"one\"", ",\"user_id\":\"two\"", "", "" },
		{ ",\"metadata\":{\"user_id\":\"one\"}",
		  ",\"metadata\":{\"user_id\":\"two\"}", "", "" },
		{ "", "", "x-openwebui-user-id: one\r\n",
		  "x-openwebui-user-id: two\r\n" },
		{ "", "", "x-opencode-project: one\r\n",
		  "x-opencode-project: two\r\n" },
		{ "", "", "openai-project: one\r\n",
		  "openai-project: two\r\n" },
		{ "", "", "openai-organization: one\r\n",
		  "openai-organization: two\r\n" }
	};
	static const struct identity_case soft[] = {
		{ ",\"prompt_cache_key\":\"one\"",
		  ",\"prompt_cache_key\":\"two\"", "", "" },
		{ ",\"session_id\":\"one\"", ",\"session_id\":\"two\"", "",
		  "" },
		{ ",\"conversation_id\":\"one\"",
		  ",\"conversation_id\":\"two\"", "", "" },
		{ ",\"metadata\":{\"session_id\":\"one\",\"cache_key\":\"one\"}",
		  ",\"metadata\":{\"session_id\":\"two\",\"cache_key\":\"two\"}",
		  "", "" },
		{ "", "", "session-id: one\r\nthread-id: one\r\n",
		  "session-id: two\r\nthread-id: two\r\n" },
		{ "", "", "x-openwebui-chat-id: one\r\nx-task-id: one\r\n",
		  "x-openwebui-chat-id: two\r\nx-task-id: two\r\n" }
	};
	server->session_fork = 1;
	first->authenticated_user = second->authenticated_user = true;
	for (size_t index = 0; index < sizeof(hard) / sizeof(hard[0]);
	     index++) {
		expect(identity_read(server, first, "alice",
				     hard[index].first_members,
				     hard[index].first_headers) &&
			       identity_read(server, second, "alice",
					     hard[index].second_members,
					     hard[index].second_headers) &&
			       !server_cache_owner_equal(
				       &first->cache_namespace,
				       &second->cache_namespace),
		       "fork ownership preserves every explicit user, salt, project and organization boundary");
	}
	for (size_t index = 0; index < sizeof(soft) / sizeof(soft[0]);
	     index++) {
		expect(identity_read(server, first, "alice",
				     soft[index].first_members,
				     soft[index].first_headers) &&
			       identity_read(server, second, "alice",
					     soft[index].second_members,
					     soft[index].second_headers) &&
			       !server_cache_namespace_equal(
				       &first->cache_namespace,
				       &second->cache_namespace) &&
			       server_cache_owner_equal(
				       &first->cache_namespace,
				       &second->cache_namespace),
		       "only known session/cache affinity tags may differ inside one fork owner");
	}
	expect(identity_read(server, first, "alice", ",\"cache_salt\":\"one\"",
			     "") &&
		       identity_read(server, second, "alice", "", "") &&
		       !server_cache_owner_equal(&first->cache_namespace,
						 &second->cache_namespace),
	       "omitting a hard identity cannot bypass its boundary");
	expect(identity_read(server, first, "alice", "", "") &&
		       identity_read(server, second, "bob", "", "") &&
		       !server_cache_owner_equal(&first->cache_namespace,
						 &second->cache_namespace),
	       "a lineage hint can never authorize another API user");
	second->authenticated_user = false;
	expect(identity_read(server, second, "alice", "", "") &&
		       !server_cache_owner_equal(&first->cache_namespace,
						 &second->cache_namespace) &&
		       !server_cache_owner_equal(&second->cache_namespace,
						 &second->cache_namespace),
	       "anonymous identities cannot expand to cross-session donors");
	second->authenticated_user = true;
	expect(identity_read(server, first, "alice",
			     ",\"session_id\":\"parent\"", ""),
	       "owner corruption fixture");
	struct cache_namespace invalid = first->cache_namespace;
	invalid.byte_count = SERVER_CACHE_NAMESPACE_BYTES + 1u;
	expect(!server_cache_owner_equal(&first->cache_namespace, &invalid),
	       "oversized stored namespace refuses");
	invalid = first->cache_namespace;
	invalid.byte_count--;
	expect(!server_cache_owner_equal(&first->cache_namespace, &invalid),
	       "truncated stored TLV refuses");
	invalid = first->cache_namespace;
	invalid.bytes[TEST_NAMESPACE_PREFIX_BYTES + 1u] = 0;
	invalid.bytes[TEST_NAMESPACE_PREFIX_BYTES + TEST_NAMESPACE_PREFIX_BYTES] =
		0;
	expect(!server_cache_owner_equal(&first->cache_namespace, &invalid),
	       "zero-length stored owner refuses");
	invalid = first->cache_namespace;
	uint32_t cursor = invalid.byte_count;
	expect(cursor + TEST_APPENDED_FIELD_BYTES <= sizeof(invalid.bytes),
	       "unknown stored field fits test storage");
	invalid.bytes[cursor++] = UINT8_MAX;
	invalid.bytes[cursor++] = 1;
	invalid.bytes[cursor++] = 0;
	invalid.bytes[cursor++] = 'x';
	invalid.byte_count = cursor;
	expect(!server_cache_owner_equal(&first->cache_namespace, &invalid),
	       "unknown stored tag refuses");
	invalid = first->cache_namespace;
	invalid.has_session = 0;
	expect(!server_cache_owner_equal(&first->cache_namespace, &invalid),
	       "inconsistent stored flags refuse");
	invalid = first->cache_namespace;
	const size_t session_tag = TEST_NAMESPACE_PREFIX_BYTES +
				   TEST_NAMESPACE_TAG_BYTES + strlen("alice");
	invalid.bytes[session_tag] = 0;
	expect(!server_cache_owner_equal(&first->cache_namespace, &invalid),
	       "duplicate or noncanonical stored tag order refuses");
	invalid = first->cache_namespace;
	invalid.bytes[session_tag + TEST_NAMESPACE_TAG_BYTES] =
		TEST_INVALID_UTF8_LEAD;
	expect(!server_cache_owner_equal(&first->cache_namespace, &invalid),
	       "invalid UTF-8 in an otherwise soft stored identity refuses");
	server->cache_scope = SERVER_CACHE_USER;
	expect(!identity_read(server, second, "alice", "", ""),
	       "forks cannot run in a collapsed compatibility scope");
	server->cache_scope = SERVER_CACHE_SESSION;
	server->session_fork = 0;
}

static void test_fork_hints(struct http_server *server,
			    struct connection *first, struct connection *second)
{
	server->session_fork = 1;
	first->authenticated_user = second->authenticated_user = true;
	expect(identity_read(
		       server, first, "alice",
		       ",\"client_metadata\":{\"session_id\":\"shared-root\",\"thread_id\":\"parent-thread\"}",
		       "session-id: shared-root\r\nthread-id: parent-thread\r\n") &&
		       identity_read(
			       server, second, "alice",
			       ",\"client_metadata\":{\"session_id\":\"shared-root\",\"thread_id\":\"child-thread\","
			       "\"x-codex-parent-thread-id\":\"parent-thread\"}",
			       "session-id: shared-root\r\nthread-id: child-thread\r\n"
			       "x-codex-parent-thread-id: parent-thread\r\n") &&
		       server_cache_owner_equal(&first->cache_namespace,
						&second->cache_namespace) &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace) &&
		       second->fork_parent.kind == CACHE_FORK_THREAD &&
		       server_cache_fork_hint_matches(&second->fork_parent,
						      &first->cache_namespace),
	       "Codex header/body parent thread hints agree while child identity stays distinct");
	expect(identity_read(server, second, "alice", "",
			     "x-codex-parent-thread-id: shared-root\r\n") &&
		       !server_cache_fork_hint_matches(&second->fork_parent,
						       &first->cache_namespace),
	       "a Codex parent thread cannot accidentally match the family's shared root session");
	expect(identity_read(
		       server, first, "alice",
		       ",\"prompt_cache_key\":\"root\",\"client_metadata\":{\"session_id\":\"root\","
		       "\"thread_id\":\"parent\"}",
		       "") &&
		       identity_read(
			       server, second, "alice",
			       ",\"prompt_cache_key\":\"root\",\"client_metadata\":{\"session_id\":\"root\","
			       "\"thread_id\":\"child\","
			       "\"x-codex-parent-thread-id\":\"parent\"}",
			       "") &&
		       !server_cache_namespace_equal(&first->cache_namespace,
						     &second->cache_namespace) &&
		       server_cache_fork_hint_matches(&second->fork_parent,
						      &first->cache_namespace),
	       "stable body identities prevent child collapse when a proxy strips session headers");
	server->session_fork = 0;
	expect(identity_read(
		       server, first, "alice",
		       ",\"prompt_cache_key\":\"root\",\"client_metadata\":{\"thread_id\":\"parent\"}",
		       "") &&
		       identity_read(
			       server, second, "alice",
			       ",\"prompt_cache_key\":\"root\",\"client_metadata\":{\"thread_id\":\"child\","
			       "\"x-codex-parent-thread-id\":42}",
			       "x-anemone-fork-of: one\r\nx-anemone-fork-of: two\r\n") &&
		       server_cache_namespace_equal(&first->cache_namespace,
						    &second->cache_namespace) &&
		       second->fork_parent.kind == CACHE_FORK_NONE &&
		       second->fork_parent.byte_count == 0,
	       "fork zero preserves old namespace bytes and ignores new metadata/hints");
	server->session_fork = 1;
	expect(identity_read(server, first, "alice", "",
			     "session_id: parent-session\r\n") &&
		       identity_read(
			       server, second, "alice", "",
			       "X-Anemone-Fork-Of: \tparent-session \t\r\n") &&
		       second->fork_parent.kind == CACHE_FORK_SESSION &&
		       server_cache_fork_hint_matches(&second->fork_parent,
						      &first->cache_namespace),
	       "pi's generic parent hint uses normalized HTTP value and session aliases");
	expect(identity_read(
		       server, first, "alice", "",
		       "session-id: other\r\nthread-id: parent-session\r\n") &&
		       !server_cache_fork_hint_matches(&second->fork_parent,
						       &first->cache_namespace),
	       "generic session hints do not bind thread-only aliases");
	expect(identity_read(server, first, "alice",
			     ",\"session_id\":\"child\"",
			     "x-anemone-fork-of: one\r\n") &&
		       identity_read(server, second, "alice",
				     ",\"session_id\":\"child\"",
				     "x-anemone-fork-of: two\r\n") &&
		       server_cache_namespace_equal(&first->cache_namespace,
						    &second->cache_namespace),
	       "donor hints never become a namespace equality component");
	const struct cache_namespace before = second->cache_namespace;
	const struct cache_fork_hint hint_before = second->fork_parent;
	expect(!identity_read(
		       server, second, "alice", "",
		       "x-anemone-fork-of: one\r\nX-Anemone-Fork-Of: one\r\n") &&
		       server_cache_namespace_equal(&before,
						    &second->cache_namespace) &&
		       hint_before.kind == second->fork_parent.kind &&
		       hint_before.byte_count ==
			       second->fork_parent.byte_count &&
		       memcmp(hint_before.bytes, second->fork_parent.bytes,
			      hint_before.byte_count) == 0,
	       "duplicate hint headers refuse transactionally even when equal");
	expect(!identity_read(
		       server, second, "alice",
		       ",\"client_metadata\":{\"x-codex-parent-thread-id\":\"body\"}",
		       "x-codex-parent-thread-id: header\r\n"),
	       "disagreeing Codex header/body hints refuse");
	expect(!identity_read(server, second, "alice",
			      ",\"client_metadata\":[]", ""),
	       "fork client metadata must be an object when present");
	expect(!identity_read(
		       server, second, "alice",
		       ",\"client_metadata\":{\"x-codex-parent-thread-id\":42}",
		       ""),
	       "nonstrings cannot name a parent");
	expect(!identity_read(server, second, "alice", "",
			      "x-anemone-fork-of: \r\n"),
	       "empty hints refuse");
	expect(!identity_read(
		       server, second, "alice",
		       ",\"client_metadata\":{\"x-codex-parent-thread-id\":\"A\\u0000B\"}",
		       ""),
	       "embedded control bytes cannot truncate a donor hint");
	expect(identity_read(server, first, "alice",
			     ",\"client_metadata\":{\"thread_id\":\"\\u2603\"}",
			     "") &&
		       identity_read(
			       server, second, "alice",
			       ",\"client_metadata\":{\"x-codex-parent-thread-id\":\"\\u2603\"}",
			       "") &&
		       server_cache_fork_hint_matches(&second->fork_parent,
						      &first->cache_namespace),
	       "decoded UTF-8 parent bytes match without normalization or truncation");
	char maximum[SERVER_CACHE_FIELD_BYTES + 1u], headers[REQUEST_BYTES];
	memset(maximum, 'p', SERVER_CACHE_FIELD_BYTES);
	maximum[SERVER_CACHE_FIELD_BYTES] = '\0';
	(void)snprintf(headers, sizeof(headers), "x-anemone-fork-of: %s\r\n",
		       maximum);
	expect(identity_read(server, second, "alice", "", headers) &&
		       second->fork_parent.byte_count ==
			       SERVER_CACHE_FIELD_BYTES,
	       "a maximum-length parent hint is retained exactly");
	(void)snprintf(headers, sizeof(headers), "x-anemone-fork-of: %sp\r\n",
		       maximum);
	expect(!identity_read(server, second, "alice", "", headers),
	       "an oversized parent hint refuses without truncation");
	server->session_fork = 0;
	first->authenticated_user = second->authenticated_user = false;
}

int main(void)
{
	struct http_server *const server = calloc(1, sizeof(*server));
	struct connection *const first = calloc(1, sizeof(*first)),
				 *const second = calloc(1, sizeof(*second));
	if (server == NULL || first == NULL || second == NULL) {
		free(server);
		free(first);
		free(second);
		return EXIT_FAILURE;
	}
	test_identity(server, first, second);
	test_selection(server, first, second);
	test_exact_selection(server, first, second);
	test_fork_owners(server, first, second);
	test_fork_hints(server, first, second);
	free(server);
	free(first);
	free(second);
	if (failures != 0) {
		return EXIT_FAILURE;
	}
	(void)puts(
		"server cache: exact namespaces and nonblocking affinity passed");
	return EXIT_SUCCESS;
}
