/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Fork policy and publication tests. Synthetic selection histories exercise authorization and the actual cold
 * planner only; they make no numerical claim. Lifecycle tests use independently generated host-model state. */
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "engine_internal.h"
#include "host_memory_backend.h"
#include "server_internal.h"
#include "serving_fixture.h"
#include "step_driver.h"

#define CONTEXT 16384u
#define SLOTS 4u
#define SOURCE_SLOT 0u
#define DESTINATION_SLOT 1u
#define ALTERNATE_SLOT 2u
#define OLDEST_SLOT 3u
#define TEST_MIDDLE_USE 2u
#define UNQUALIFIED_FLIGHTS 2u
#define INVALID_PREFIX_QUALIFICATION 2u
#define LIFECYCLE_SLOTS 2u
#define GENERATION_EVENTS 2u
#define IDENTITY_CONNECTION 2u
#define PROMPT 6004u
#define HOST_PROMPT 513u
#define HINT_PREFIX 3072u
#define REQUEST_BYTES 4096u
#define TOKEN_VALUES 256u
#define TOKEN_STRIDE 37u
#define TOKEN_OFFSET 11u
#define SOURCE_TAG 71u
#define CHILD_TAG 72u
#define TEST_NOW 100.0
#define TEST_RECENT 99.0
#define TEST_OLD 1.0
#define TEST_GRACE 30u
#define WAIT_MILLISECONDS 10000
#define WAIT_ATTEMPTS 3u
#define TEST_SECONDS 60u

enum completion_case {
	COMPLETE_SUCCESS,
	COMPLETE_COPY_REFUSED,
	COMPLETE_SOURCE_STALE,
	COMPLETE_DESTINATION_STALE,
	COMPLETE_CANCELLED
};

static const uint32_t sizes[] = { 64, 128, 256, 512, 1024 };
static struct execution_program programs[sizeof(sizes) / sizeof(sizes[0])];
static const struct execution_program
	*pointers[sizeof(programs) / sizeof(programs[0])];
static struct step_driver planner_driver;
static struct inference_engine planner;
static struct http_server *server;
static uint32_t request_tokens[CONTEXT], histories[SLOTS * CONTEXT];
static struct backend_operations refusing_backend;
static const uint32_t script[] = { TEST_SERVING_END_TOKEN };
static const uint32_t *const scripts[] = { script };
static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: server forks: %s\n", message);
		failures++;
	}
}

static bool identity(struct connection *connection, const char *user,
		     bool authenticated, const char *members,
		     const char *headers)
{
	char body[REQUEST_BYTES], input[REQUEST_BYTES];
	(void)snprintf(
		body, sizeof(body),
		"{\"messages\":[{\"role\":\"user\",\"content\":\"synthetic\"}]%s}",
		members);
	const int bytes = snprintf(
		input, sizeof(input),
		"POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n%sContent-Length: %zu\r\n\r\n%s",
		headers, strlen(body), body);
	if (bytes <= 0 || (size_t)bytes >= sizeof(input))
		return false;
	connection->input.bytes = input;
	connection->input.count = (size_t)bytes;
	connection->authenticated_user = authenticated;
	(void)snprintf(connection->live.user, sizeof(connection->live.user),
		       "%s", user);
	const struct chat_request defaults = { .is_seeded = true,
					       .top_p = 1.0f,
					       .max_tokens = 1 };
	struct chat_refusal refusal = { 0 };
	const bool valid =
		server_head_read(input, (size_t)bytes, &connection->head) &&
		connection->head.refusal == 0 &&
		chat_request_read(body, strlen(body), "model", NULL, &defaults,
				  &connection->request, &refusal) == 0 &&
		server_cache_namespace_read(server, connection, &refusal) == 0;
	connection->input = (struct chat_buffer){ 0 };
	return valid;
}

static void reset_server(struct inference_engine *engine, uint32_t count)
{
	memset(server, 0, sizeof(*server));
	memset(histories, 0, sizeof(histories));
	server->engine = engine;
	server->histories = histories;
	server->session_fork = 1;
	server->cache_scope = SERVER_CACHE_SESSION;
	server->description.context_token_limit = CONTEXT;
	server->description.session_slot_count = count;
	server->description.prefix_cold_plan_qualified = 1;
	server->description.prefill_prefix_stable = 1;
	server->exact_prefix_reuse = SERVER_PREFIX_LEGACY;
	for (uint32_t index = 0; index < CONTEXT; index++)
		request_tokens[index] =
			(index * TOKEN_STRIDE + TOKEN_OFFSET) % TOKEN_VALUES;
	struct connection *const child = &server->connections[0];
	expect(identity(
		       child, "alice", true,
		       ",\"session_id\":\"child\",\"cache_salt\":\"one\"",
		       "openai-project: alpha\r\nx-anemone-fork-of: parent\r\n"),
	       "child identity validates");
	child->token_ids = request_tokens;
	child->token_count = PROMPT;
	child->phase = PHASE_WAITING;
	child->waiting_snapshot_slot = NO_WAITING_SNAPSHOT;
}

static void source_identity(uint32_t slot, const char *user, bool authenticated,
			    const char *members, const char *headers)
{
	struct connection *const scratch =
		&server->connections[IDENTITY_CONNECTION];
	expect(identity(scratch, user, authenticated, members, headers),
	       "source identity validates");
	server->slot_histories[slot].cache_namespace = scratch->cache_namespace;
}

static void source_set(uint32_t slot, const char *session)
{
	char members[REQUEST_BYTES];
	(void)snprintf(members, sizeof(members),
		       ",\"session_id\":\"%s\",\"cache_salt\":\"one\"",
		       session);
	server->slot_histories[slot] =
		(struct slot_history){ .token_count = PROMPT,
				       .processed_count = PROMPT - 1u,
				       .reach_count = PROMPT,
				       .cold_prompt_count = PROMPT,
				       .last_used = slot + 1u,
				       .last_active_seconds = TEST_OLD };
	memcpy(histories + (size_t)slot * CONTEXT, request_tokens,
	       sizeof(request_tokens));
	source_identity(slot, "alice", true, members,
			"openai-project: alpha\r\n");
}

static void selection_reset(void)
{
	reset_server(&planner, SLOTS);
	source_set(SOURCE_SLOT, "parent");
}

static uint32_t destination(uint32_t proposed)
{
	return server_fork_destination(server, &server->connections[0],
				       proposed, TEST_NOW);
}

static void selection_policy(void)
{
	selection_reset();
	expect(destination(SOURCE_SLOT) == DESTINATION_SLOT,
	       "with grace zero a sole donor is preserved in favor of an empty destination");
	uint32_t kept = UINT32_MAX;
	expect(server_chosen_session_slot(server, &server->connections[0],
					  TEST_NOW,
					  &kept) == DESTINATION_SLOT &&
		       kept == 0,
	       "namespace selection reserves a destination without legacy donor reuse");
	server->connections[0].cache_namespace =
		server->slot_histories[0].cache_namespace;
	expect(server_chosen_session_slot(server, &server->connections[0],
					  TEST_NOW, &kept) == SOURCE_SLOT &&
		       kept != 0,
	       "same-session legacy selection is unchanged");

	selection_reset();
	server->session_grace_seconds = TEST_GRACE;
	source_set(1, "recent");
	source_set(ALTERNATE_SLOT, "old");
	source_set(OLDEST_SLOT, "oldest");
	server->slot_histories[1].last_active_seconds = TEST_RECENT;
	server->slot_histories[ALTERNATE_SLOT].last_used = TEST_MIDDLE_USE;
	server->slot_histories[OLDEST_SLOT].last_used = 1;
	expect(destination(0) == OLDEST_SLOT,
	       "alternate destination honors grace then LRU");
	server->slot_histories[ALTERNATE_SLOT].cache_namespace =
		server->connections[0].cache_namespace;
	expect(destination(0) == ALTERNATE_SLOT,
	       "same-session affinity wins an equal-protection destination tie");
	server->slot_histories[1].last_active_seconds = TEST_OLD;
	server->slot_histories[1].last_used = 0;
	server->slot_histories[ALTERNATE_SLOT].cache_namespace =
		server->slot_histories[OLDEST_SLOT].cache_namespace;
	expect(destination(0) == 1,
	       "expired grace returns to ordinary LRU order");

	selection_reset();
	for (uint32_t slot = 1; slot < SLOTS; slot++)
		server->active[slot].token_ids = request_tokens;
	expect(destination(0) == 0,
	       "without an independent spare slot destination remains the cold candidate");
	expect(!server_fork_before(server, 0, &server->connections[0]) &&
		       server->connections[0].fork_attempted,
	       "no-spare admission cold-misses once without querying fake engine state");
}

/* A completed exact copy retains its own admission policy only through its tagged READY continuation. Global
 * repeat policy remains unchanged for an ordinary same-session request without that successful-copy marker. */
static void ready_clone_policy(void)
{
	selection_reset();
	struct connection *const child = &server->connections[0];
	struct slot_history *const copied =
		&server->slot_histories[DESTINATION_SLOT];
	*copied = (struct slot_history){ .token_count = HINT_PREFIX,
					 .processed_count = HINT_PREFIX,
					 .reach_count = PROMPT,
					 .cold_prompt_count = PROMPT,
					 .cache_namespace =
						 child->cache_namespace,
					 .last_used = TEST_MIDDLE_USE };
	memcpy(histories + (size_t)DESTINATION_SLOT * CONTEXT, request_tokens,
	       (size_t)HINT_PREFIX * sizeof(uint32_t));
	child->waiting_snapshot_slot =
		WAITING_SNAPSHOT_READY + DESTINATION_SLOT;
	child->fork_attempted = true;
	child->fork_prepared = true;
	server->exact_prefix_reuse = SERVER_PREFIX_REPEAT;
	const uint32_t certified = server_prefix_kept_mode(
		server, DESTINATION_SLOT, request_tokens, PROMPT,
		child->image_identity, SERVER_PREFIX_COLD_PLAN);
	expect(certified == HINT_PREFIX &&
		       server_prefix_kept(server, DESTINATION_SLOT,
					  request_tokens, PROMPT,
					  child->image_identity) == 0,
	       "a partial exact-copy boundary is eligible in mode2 and ineligible under global repeat mode1");
	uint32_t kept = UINT32_MAX;
	const uint32_t modes[] = { SERVER_PREFIX_LEGACY, SERVER_PREFIX_REPEAT,
				   SERVER_PREFIX_COLD_PLAN };
	for (uint32_t mode = 0; mode < sizeof(modes) / sizeof(modes[0]);
	     mode++) {
		server->exact_prefix_reuse = modes[mode];
		const uint32_t ordinary = server_prefix_kept(
			server, DESTINATION_SLOT, request_tokens, PROMPT,
			child->image_identity);
		child->fork_prepared = true;
		expect(server_chosen_session_slot(server, child, TEST_NOW,
						  &kept) == DESTINATION_SLOT &&
			       kept == certified,
		       "a successful stable READY clone retains exact mode2 under every global policy");
		child->fork_prepared = false;
		expect(server_chosen_session_slot(server, child, TEST_NOW,
						  &kept) == DESTINATION_SLOT &&
			       kept == ordinary,
		       "ordinary READY continuation without copy proof still obeys its global policy");
	}
	server->exact_prefix_reuse = SERVER_PREFIX_REPEAT;
	child->fork_prepared = true;
	copied->cache_namespace =
		server->slot_histories[SOURCE_SLOT].cache_namespace;
	(void)server_chosen_session_slot(server, child, TEST_NOW, &kept);
	expect(kept == 0,
	       "successful-copy proof cannot bypass READY namespace equality");
	copied->cache_namespace = child->cache_namespace;
	server->slot_read_leased[DESTINATION_SLOT] = true;
	expect(server_chosen_session_slot(server, child, TEST_NOW, &kept) !=
			       DESTINATION_SLOT &&
		       kept == 0,
	       "successful-copy proof cannot select a READY destination under another read lease");
}

static void qualification_policy(void)
{
	const uint32_t modes[] = { SERVER_PREFIX_LEGACY, SERVER_PREFIX_REPEAT,
				   SERVER_PREFIX_COLD_PLAN };
	const uint32_t refused[] = { 0, INVALID_PREFIX_QUALIFICATION };
	for (uint32_t mode = 0; mode < sizeof(modes) / sizeof(modes[0]);
	     mode++) {
		for (uint32_t index = 0;
		     index < sizeof(refused) / sizeof(refused[0]); index++) {
			selection_reset();
			server->exact_prefix_reuse = modes[mode];
			server->description.prefix_cold_plan_qualified =
				refused[index];
			expect(destination(SOURCE_SLOT) == SOURCE_SLOT &&
				       !server_fork_before(
					       server, DESTINATION_SLOT,
					       &server->connections[0]) &&
				       !server->connections[0].fork_attempted,
			       "unknown or malformed recipe capability cannot select or submit a donor "
			       "under any global policy");
		}
		selection_reset();
		server->exact_prefix_reuse = modes[mode];
		expect(destination(SOURCE_SLOT) == DESTINATION_SLOT &&
			       server->exact_prefix_reuse == modes[mode],
		       "a qualified exact donor remains available without changing legacy, repeat or cold-plan policy");
	}
	selection_reset();
	server->session_fork = 0;
	server->description.prefix_cold_plan_qualified = 0;
	struct connection *const child = &server->connections[0];
	child->cache_namespace =
		server->slot_histories[SOURCE_SLOT].cache_namespace;
	uint32_t kept = 0;
	expect(server_chosen_session_slot(server, child, TEST_NOW, &kept) ==
			       SOURCE_SLOT &&
		       kept != 0,
	       "disabled forks leave same-session legacy reuse unchanged when qualification is unknown");
}

static void legacy_fork_refusal(void)
{
	const uint32_t modes[] = { SERVER_PREFIX_LEGACY, SERVER_PREFIX_REPEAT,
				   SERVER_PREFIX_COLD_PLAN };
	uint32_t *const history_before = malloc(sizeof(histories));
	expect(history_before != NULL,
	       "legacy refusal history backup allocates");
	if (history_before == NULL)
		return;
	for (uint32_t mode = 0; mode < sizeof(modes) / sizeof(modes[0]);
	     mode++) {
		selection_reset();
		server->description.prefill_prefix_stable = 0;
		planner.description.prefill_prefix_stable = 0;
		server->exact_prefix_reuse = modes[mode];
		struct slot_history before[SLOTS];
		bool leased_before[SLOTS];
		struct active_command active_before[SLOTS];
		memcpy(before, server->slot_histories, sizeof(before));
		memcpy(leased_before, server->slot_read_leased,
		       sizeof(leased_before));
		memcpy(active_before, server->active, sizeof(active_before));
		memcpy(history_before, histories, sizeof(histories));
		struct connection *const child = &server->connections[0];
		expect(destination(SOURCE_SLOT) == SOURCE_SLOT &&
			       !server_fork_before(server, DESTINATION_SLOT,
						   child) &&
			       !child->fork_attempted && !child->fork_prepared,
		       "qualified legacy form never retargets or submits a cross-session fork "
		       "under any global policy");
		expect(memcmp(before, server->slot_histories, sizeof(before)) ==
				       0 &&
			       memcmp(leased_before, server->slot_read_leased,
				      sizeof(leased_before)) == 0 &&
			       memcmp(active_before, server->active,
				      sizeof(active_before)) == 0 &&
			       memcmp(history_before, histories,
				      sizeof(histories)) == 0,
		       "legacy fork refusal preserves histories, namespaces, leases and active commands");
		child->cache_namespace =
			server->slot_histories[SOURCE_SLOT].cache_namespace;
		server->session_fork = 0;
		uint32_t ordinary_kept = UINT32_MAX;
		const uint32_t ordinary_slot = server_chosen_session_slot(
			server, child, TEST_NOW, &ordinary_kept);
		server->session_fork = 1;
		uint32_t enabled_kept = UINT32_MAX;
		expect(server_chosen_session_slot(server, child, TEST_NOW,
						  &enabled_kept) ==
				       ordinary_slot &&
			       enabled_kept == ordinary_kept,
		       "same-session legacy selection is identical with forks disabled or enabled");
	}
	planner.description.prefill_prefix_stable = 1;
	free(history_before);
}

static void hard_owner_policy(void)
{
	static const struct {
		const char *user, *members, *headers;
		bool authenticated;
	} cases[] = {
		{ "bob", ",\"session_id\":\"parent\",\"cache_salt\":\"one\"",
		  "openai-project: alpha\r\n", true },
		{ "alice", ",\"session_id\":\"parent\",\"cache_salt\":\"two\"",
		  "openai-project: alpha\r\n", true },
		{ "alice", ",\"session_id\":\"parent\",\"cache_salt\":\"one\"",
		  "openai-project: beta\r\n", true },
		{ "alice",
		  ",\"session_id\":\"parent\",\"cache_salt\":\"one\",\"user\":\"end-user\"",
		  "openai-project: alpha\r\n", true },
		{ "alice", ",\"session_id\":\"parent\",\"cache_salt\":\"one\"",
		  "openai-project: alpha\r\n", false }
	};
	for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		selection_reset();
		source_identity(0, cases[index].user,
				cases[index].authenticated,
				cases[index].members, cases[index].headers);
		expect(destination(0) == 0,
		       "API user, salt, project, end-user or authentication mismatch defeats a matching hint");
	}
	selection_reset();
	server->connections[0].authenticated_user = false;
	expect(identity(&server->connections[0], "alice", false,
			",\"session_id\":\"child\",\"cache_salt\":\"one\"",
			"openai-project: alpha\r\n"),
	       "anonymous child identity validates");
	source_identity(0, "alice", false,
			",\"session_id\":\"parent\",\"cache_salt\":\"one\"",
			"openai-project: alpha\r\n");
	expect(destination(0) == 0,
	       "matching anonymous identities never authorize cross-session sharing");
}

static void donor_guards(void)
{
	selection_reset();
	histories[0] ^= 1u;
	expect(destination(0) == 0,
	       "a forged correct-name hint cannot waive token comparison");
	selection_reset();
	server->slot_histories[0].cold_prompt_count = 0;
	expect(destination(0) == 0,
	       "unknown cold provenance misses even with an explicit hint");
	selection_reset();
	server->slot_histories[0].from_snapshot = true;
	expect(destination(0) == 0,
	       "imported snapshot provenance cannot donate");
	selection_reset();
	server->slot_histories[0].cold_prompt_count++;
	expect(destination(0) == 0,
	       "a child shorter than its donor's canonical prompt cannot inherit that provenance");
	selection_reset();
	server->slot_histories[0].image_identity[0] = 1;
	server->connections[0].image_identity[0] = 1;
	expect(destination(0) == 0,
	       "even matching image identities are outside exact fork qualification");
	selection_reset();
	server->active[0].token_ids = request_tokens;
	expect(destination(0) == 0, "an active donor cannot be leased");
	selection_reset();
	server->slot_read_leased[0] = true;
	expect(destination(0) == 0,
	       "an already read-leased donor cannot be leased twice");
	selection_reset();
	const uint32_t prefix = server_prefix_kept_mode(
		server, 0, request_tokens, PROMPT,
		server->connections[0].image_identity, SERVER_PREFIX_COLD_PLAN);
	expect(prefix != 0,
	       "ordinary selection fixture has a genuine planner boundary");
	server->slot_histories[0].reach_count =
		prefix + SERVING_KEPT_LAG + SERVING_KEPT_GRANULE;
	server->slot_histories[0].checkpoint_count = prefix;
	expect(server_prefix_kept_mode(server, 0, request_tokens, PROMPT,
				       server->connections[0].image_identity,
				       SERVER_PREFIX_COLD_PLAN) == prefix &&
		       destination(0) == 0,
	       "checkpoint-only eligibility remains outside ordinary server fork policy");
	selection_reset();
	server->connections[0].request.max_tokens = 0;
	expect(destination(0) == 0, "prefill-only requests never fork");
	selection_reset();
	server->session_fork = 0;
	expect(destination(0) == 0,
	       "the default-off knob retains old selection");
	selection_reset();
	server->disable_prefix_reuse = true;
	expect(destination(0) == 0, "disabled reuse forbids forks");
	for (uint32_t scope = SERVER_CACHE_USER; scope <= SERVER_CACHE_GLOBAL;
	     scope++) {
		selection_reset();
		server->cache_scope = scope;
		expect(destination(0) == 0,
		       "compatibility cache scopes cannot enable cross-session forks");
	}
	selection_reset();
	server->connections[0].fork_attempted = true;
	expect(destination(0) == 0,
	       "a finished or refused attempt cannot retry selection");
	selection_reset();
	planner.description.chunks_in_flight = UNQUALIFIED_FLIGHTS;
	expect(destination(0) == 0,
	       "unqualified multi-flight prefill cannot donate");
	planner.description.chunks_in_flight = 1;
}

static void hint_preference(void)
{
	selection_reset();
	source_set(1, "preferred");
	histories[CONTEXT + HINT_PREFIX] ^= 1u;
	struct connection *const child = &server->connections[0];
	expect(identity(
		       child, "alice", true,
		       ",\"session_id\":\"child\",\"cache_salt\":\"one\"",
		       "openai-project: alpha\r\nx-anemone-fork-of: preferred\r\n"),
	       "preferred-parent hint validates");
	const uint32_t hinted = server_prefix_kept_mode(
		server, 1, request_tokens, PROMPT, child->image_identity,
		SERVER_PREFIX_COLD_PLAN);
	const uint32_t larger = server_prefix_kept_mode(
		server, 0, request_tokens, PROMPT, child->image_identity,
		SERVER_PREFIX_COLD_PLAN);
	expect(hinted != 0 && hinted < larger,
	       "preferred donor intentionally shares a shorter exact prefix");
	expect(destination(1) == ALTERNATE_SLOT,
	       "an eligible hinted parent outranks a larger unhinted exact prefix");
	child->fork_parent = (struct cache_fork_hint){ 0 };
	expect(destination(1) == 1,
	       "without a hint the longest eligible exact donor wins");
	child->fork_parent =
		(struct cache_fork_hint){ .byte_count =
						  sizeof("preferred") - 1u,
					  .kind = CACHE_FORK_SESSION,
					  .bytes = "preferred" };
	source_identity(1, "bob", true,
			",\"session_id\":\"preferred\",\"cache_salt\":\"one\"",
			"openai-project: alpha\r\n");
	expect(destination(0) == ALTERNATE_SLOT,
	       "an ineligible hinted owner falls back to another same-owner exact donor");

	selection_reset();
	source_set(1, "other");
	source_identity(
		1, "alice", true,
		",\"session_id\":\"shared-root\",\"cache_salt\":\"one\"",
		"openai-project: alpha\r\nthread-id: worker\r\n");
	expect(identity(
		       child, "alice", true,
		       ",\"session_id\":\"child\",\"cache_salt\":\"one\"",
		       "openai-project: alpha\r\nx-codex-parent-thread-id: worker\r\n"),
	       "typed Codex parent-thread hint validates");
	expect(destination(1) == ALTERNATE_SLOT,
	       "Codex lineage prefers the matching parent thread within the hard owner");
	expect(identity(
		       child, "alice", true,
		       ",\"session_id\":\"child\",\"cache_salt\":\"one\"",
		       "openai-project: alpha\r\nx-codex-parent-thread-id: shared-root\r\n"),
	       "a Codex root-shaped hint remains a typed thread hint");
	expect(destination(1) == 1,
	       "a shared root session identifier cannot impersonate a Codex parent thread");
}

static struct generation_event next_event(struct inference_engine *engine)
{
	struct generation_event event = { .event_kind =
						  GENERATION_EVENT_FAILED };
	int descriptor = -1;
	expect(engine_event_descriptor(engine, &descriptor) == STATUS_OK,
	       "event descriptor opens");
	for (uint32_t attempt = 0; attempt < WAIT_ATTEMPTS; attempt++) {
		bool available = false;
		if (engine_next_event(engine, &event, &available) != STATUS_OK)
			break;
		if (available)
			return event;
		struct pollfd polling = { descriptor, POLLIN, 0 };
		if (poll(&polling, 1, WAIT_MILLISECONDS) <= 0)
			break;
	}
	expect(false, "command produces an event before its deadline");
	return event;
}

static struct generation_event generated(struct inference_engine *engine,
					 uint32_t slot, uint32_t kept,
					 uint32_t tag)
{
	const struct generation_command_record command = {
		.new_token_ids = request_tokens + kept,
		.session_slot = slot,
		.kept_token_count = kept,
		.new_token_count = HOST_PROMPT - kept,
		.generated_token_limit = 1,
		.sampler_top_p = 1.0f,
		.request_tag = tag
	};
	const result_status status = engine_submit(engine, &command);
	expect(status == STATUS_OK, "host generation is admitted");
	struct generation_event event = { .event_kind =
						  GENERATION_EVENT_FAILED };
	if (status != STATUS_OK)
		return event;
	for (uint32_t count = 0; count < GENERATION_EVENTS; count++) {
		event = next_event(engine);
		if (event.event_kind != GENERATION_EVENT_TOKEN)
			return event;
		histories[(size_t)slot * CONTEXT + HOST_PROMPT] =
			event.token_id;
	}
	expect(false, "one-token generation has a final event");
	return event;
}

MUST_CHECK static result_status
refused_copy(struct device_card *card, const struct kernel_launch *launches,
	     uint32_t count)
{
	if (launches != NULL && count == 1u &&
	    launches[0].kernel == BACKEND_COPY)
		return STATUS_INJECTED_FAULT;
	return test_host_memory_backend.launch_run(card, launches, count);
}

static void completion_policy(enum completion_case which)
{
	struct inference_engine *engine = NULL;
	test_serving_set_interleaved(0);
	test_serving_set_engine(LIFECYCLE_SLOTS, 1, TEST_SERVING_MOST_ROWS, 0);
	test_serving_set_prefill(1, 0, 0);
	test_serving_set_cone(0);
	test_serving_set_function_model(true, TEST_SERVING_WINDOW - 1u);
	test_serving_set_function_end(UINT32_MAX);
	test_serving_set_ring(TEST_SERVING_STATE_POSITIONS,
			      TEST_SERVING_WINDOW);
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "host engine opens");
	if (engine == NULL)
		return;
	/* The fixture exposes no setter for this immutable planner choice. Set it before any admission, under lock;
	 * this host byte model checks lifecycle and copy ownership, not real-model floating-point equivalence. */
	(void)pthread_mutex_lock(&engine->lock);
	engine->description.chunks_in_flight = 1;
	engine->description.prefix_cold_plan_qualified = 1;
	engine->description.prefill_prefix_stable = 1;
	(void)pthread_mutex_unlock(&engine->lock);
	reset_server(engine, LIFECYCLE_SLOTS);
	source_set(SOURCE_SLOT, "parent");
	struct connection *const child = &server->connections[0];
	child->token_count = HOST_PROMPT;
	const struct generation_event warmed =
		generated(engine, SOURCE_SLOT, 0, SOURCE_TAG);
	expect(warmed.event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       warmed.status == 0,
	       "donor is actually generated to completion");
	struct session_state source_state = { 0 };
	expect(engine_session_state(engine, SOURCE_SLOT, &source_state) ==
		       STATUS_OK,
	       "real donor metadata is available");
	struct slot_history *const history =
		&server->slot_histories[SOURCE_SLOT];
	history->token_count = warmed.session_token_count;
	history->processed_count = warmed.session_token_count - 1u;
	history->reach_count = source_state.reach_count;
	history->decoder_from = source_state.decoder_from;
	history->checkpoint_count = source_state.checkpoint_count;
	history->cold_prompt_count = HOST_PROMPT;
	history->saved_count = SERVING_KEPT_GRANULE;
	const struct slot_history donor_before = *history;
	unsigned char source_before[CONTEXT];
	const unsigned char *const donor_state = test_serving_state();
	expect(donor_state != NULL, "generated donor state is available");
	if (donor_state == NULL) {
		test_serving_close();
		return;
	}
	memcpy(source_before, donor_state, sizeof(source_before));
	request_tokens[HOST_PROMPT - 1u] ^= 1u;
	const struct backend_operations *const original_backend =
		engine->driver->backend;
	if (which == COMPLETE_COPY_REFUSED) {
		(void)pthread_mutex_lock(&engine->lock);
		refusing_backend = *original_backend;
		refusing_backend.launch_run = refused_copy;
		engine->driver->backend = &refusing_backend;
		(void)pthread_mutex_unlock(&engine->lock);
	}
	const bool admitted =
		server_fork_before(server, DESTINATION_SLOT, child);
	expect(admitted, "controller admits a real independent engine clone");
	if (!admitted) {
		test_serving_close();
		return;
	}
	struct active_command *const active = &server->active[DESTINATION_SLOT];
	child->waiting_snapshot_slot = DESTINATION_SLOT;
	child->waiting_snapshot_tag = active->request_tag;
	expect(active->token_ids != request_tokens &&
		       active->token_ids != histories &&
		       memcmp(active->token_ids, histories,
			      (size_t)active->prompt_token_count *
				      sizeof(uint32_t)) == 0,
	       "active destination owns a separate immutable copy of the admitted prefix tokens");
	expect(server->slot_read_leased[SOURCE_SLOT] && child->fork_attempted &&
		       !child->fork_prepared &&
		       server->slot_histories[DESTINATION_SLOT]
				       .cache_namespace.byte_count == 0,
	       "source is leased while the destination namespace remains unpublished");
	struct generation_event event = next_event(engine);
	if (which == COMPLETE_COPY_REFUSED) {
		expect(event.event_kind == GENERATION_EVENT_FAILED &&
			       event.status == STATUS_INJECTED_FAULT,
		       "a real backend copy refusal reaches the controller");
		(void)pthread_mutex_lock(&engine->lock);
		engine->driver->backend = original_backend;
		(void)pthread_mutex_unlock(&engine->lock);
	} else {
		expect(event.event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT &&
			       event.status == 0,
		       "real clone has one successful final event");
	}
	/* Deliberately stale front-end correlation certificates and a cancelled final notification test refusal to
	 * publish. These do not simulate an engine generation race or claim engine cancellation/drain coverage. */
	if (which == COMPLETE_SOURCE_STALE)
		active->fork_source_generation++;
	if (which == COMPLETE_DESTINATION_STALE)
		active->fork_destination_generation++;
	if (which == COMPLETE_CANCELLED)
		event.event_kind = GENERATION_EVENT_CANCELLED;
	const bool prepared =
		server_fork_final(server, DESTINATION_SLOT, &event, TEST_NOW);
	expect(prepared == (which == COMPLETE_SUCCESS) &&
		       child->fork_prepared == prepared &&
		       !server->slot_read_leased[SOURCE_SLOT],
	       "only a matching successful clone publishes and every final releases its lease");
	expect(memcmp(&donor_before, history, sizeof(donor_before)) == 0 &&
		       memcmp(source_before, test_serving_state(),
			      sizeof(source_before)) == 0,
	       "every controller outcome preserves donor namespace, bytes, saved coverage, provenance and affinity");
	const uint32_t prefix = active->prompt_token_count;
	if (prepared) {
		const struct slot_history *const target =
			&server->slot_histories[DESTINATION_SLOT];
		expect(server_cache_namespace_equal(&target->cache_namespace,
						    &child->cache_namespace) &&
			       target->token_count == prefix &&
			       target->processed_count == prefix &&
			       target->saved_count == 0 &&
			       target->checkpoint_count == 0 &&
			       target->cold_prompt_count == HOST_PROMPT,
		       "successful publication assigns only the child's independent prefix and inherited certificate");
	} else {
		expect(server->slot_histories[DESTINATION_SLOT].token_count ==
				       0 &&
			       server->slot_histories[DESTINATION_SLOT]
					       .cache_namespace.byte_count == 0,
		       "refused publication retains no destination token or namespace certificate");
	}
	free(active->token_ids);
	*active = (struct active_command){ .connection = NO_CONNECTION };
	expect(!server_fork_before(server, DESTINATION_SLOT, child),
	       "completion never retries its copy prerequisite");
	const struct generation_event continued = generated(
		engine, DESTINATION_SLOT, prepared ? prefix : 0u, CHILD_TAG);
	expect(continued.event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       continued.status == 0,
	       "published prefix continues, or rejected publication performs a real cold row-zero restart");
	expect(memcmp(source_before, test_serving_state(),
		      sizeof(source_before)) == 0,
	       "child continuation or cold restart cannot mutate donor device state");
	test_serving_close();
}

int main(void)
{
	(void)alarm(TEST_SECONDS);
	server = calloc(1, sizeof(*server));
	if (server == NULL)
		return EXIT_FAILURE;
	const uint32_t count = sizeof(sizes) / sizeof(sizes[0]);
	for (uint32_t index = 0; index < count; index++) {
		programs[index] =
			(struct execution_program){ .row_count = sizes[index],
						    .kind = PROGRAM_FULL };
		pointers[index] = &programs[index];
	}
	planner_driver = (struct step_driver){ .programs = pointers,
					       .program_count = count };
	planner.driver = &planner_driver;
	planner.description = (struct engine_description_record){
		.context_token_limit = CONTEXT,
		.session_slot_count = SLOTS,
		.chunks_in_flight = 1,
		.prefill_first = 1,
		.prefill_prefix_stable = 1,
		.prefix_cold_plan_qualified = 1
	};
	planner.full_rows = sizes[count - 1u];
	selection_policy();
	ready_clone_policy();
	qualification_policy();
	legacy_fork_refusal();
	hard_owner_policy();
	donor_guards();
	hint_preference();
	for (uint32_t which = COMPLETE_SUCCESS; which <= COMPLETE_CANCELLED;
	     which++)
		completion_policy((enum completion_case)which);
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
	test_serving_set_prefill(0, 0, 0);
	test_serving_set_function_model(false, 0);
	free(server);
	(void)printf("server forks: %s\n", failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
