/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Snapshot policy and metadata through the engine and server session interfaces, over the host serving model. */
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "anemone/front_end_seam.h"
#include "anemone/chat_images.h"
#include "anemone/model_layout.h"
#include "server_internal.h"
#include "serving_fixture.h"

#define CONTEXT 16384u
#define FIRST_COUNT 5000u
#define FAR_REACH 8840u
#define LONGER_COUNT 5280u
#define PREFIX_COUNT (FIRST_COUNT - 256u)
#define SNAPSHOT_BUDGET (1ull << 20u)
#define PATH_BYTES 512u
#define POLL_MILLISECONDS 5000
#define IDLE_TIME 121.0
#define TOKEN_VALUES 251u
#define TOKEN_STRIDE 37u
#define SAVE_MINIMUM 1024u
#define CANCEL_PROMPT 8192u
#define CONE_ROWS 256u
#define STEP_DELAY 100000ull
#define WAIT_PAUSE 1000000l
#define WAIT_ATTEMPTS 5000u
#define SNAPSHOT_MAGIC_BYTES 16u
#define SNAPSHOT_VERSION 4u
#define LEGACY_VERSION 1u
#define FILE_MODE 0600

struct test_snapshot_head {
	char magic[SNAPSHOT_MAGIC_BYTES];
	uint64_t identity, payload_bytes;
	uint32_t version, token_count, reach_count, decoder_from;
	uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES];
	struct cache_namespace cache_namespace;
};

static uint32_t tokens[CONTEXT];
static uint32_t histories[CONTEXT];
static const uint32_t script[] = { TEST_SERVING_END_TOKEN };
static const uint32_t *const scripts[] = { script };
static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: serving snapshots: %s\n", message);
		failures++;
	}
}

static struct generation_event final_taken(struct inference_engine *engine,
					   uint32_t *token_out)
{
	struct generation_event event = { 0 };
	int descriptor = -1;
	expect(engine_event_descriptor(engine, &descriptor) == STATUS_OK,
	       "event descriptor");
	for (;;) {
		struct pollfd entry = { descriptor, POLLIN, 0 };
		bool available = false;
		if (poll(&entry, 1, POLL_MILLISECONDS) <= 0 ||
		    engine_next_event(engine, &event, &available) !=
			    STATUS_OK ||
		    !available) {
			expect(false, "a command finishes before its deadline");
			return (struct generation_event){ 0 };
		}
		if (event.event_kind != GENERATION_EVENT_TOKEN) {
			return event;
		}
		if (token_out != NULL) {
			*token_out = event.token_id;
		}
	}
}

static void prefilling(struct inference_engine *engine, uint32_t kept,
		       uint32_t count)
{
	const struct generation_command_record command = {
		.new_token_ids = tokens + kept,
		.kept_token_count = kept,
		.new_token_count = count - kept,
		.sampler_top_p = 1.0f
	};
	expect(engine_submit(engine, &command) == STATUS_OK,
	       "prefill submitted");
	expect(final_taken(engine, NULL).event_kind ==
		       GENERATION_EVENT_FINISHED_AT_LIMIT,
	       "prefill completed");
}

static void store_opened(struct http_server *server,
			 struct inference_engine *engine, const char *directory)
{
	*server = (struct http_server){
		.engine = engine,
		.histories = histories,
		.session_save_tokens = SERVER_DEFAULT_SAVE_TOKENS,
		.session_load_tokens = SERVER_DEFAULT_LOAD_TOKENS,
		.session_idle_save_seconds = SERVER_DEFAULT_IDLE_SAVE_SECONDS,
		.session_directory = directory,
		.session_directory_bytes = SNAPSHOT_BUDGET
	};
	server->slot_histories[0].cache_namespace.byte_count = 1;
	expect(engine_describe(engine, &server->description) == STATUS_OK,
	       "engine described");
	expect(server_sessions_open(server) == STATUS_OK,
	       "session store opens");
}

static void store_command_finished(struct http_server *server)
{
	const struct generation_event event = final_taken(server->engine, NULL);
	expect(server_session_final(server, 0, &event),
	       "session final handled");
	expect(event.event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT,
	       "session command succeeds");
	free(server->active[0].token_ids);
	server->active[0] =
		(struct active_command){ .connection = NO_CONNECTION };
}

static void store_saved(struct http_server *server, uint32_t count)
{
	struct session_state state = { 0 };
	double nearest = IDLE_TIME;
	expect(engine_session_state(server->engine, 0, &state) == STATUS_OK,
	       "idle slot state");
	memcpy(histories, tokens, (size_t)count * sizeof(uint32_t));
	server->slot_histories[0] =
		(struct slot_history){ .token_count = count,
				       .processed_count = count,
				       .cache_namespace = { .byte_count = 1 },
				       .reach_count = state.reach_count };
	server_sessions_idle(server, IDLE_TIME, &nearest);
	expect(server->active[0].token_ids != NULL, "idle save begins");
	if (server->active[0].token_ids != NULL) {
		store_command_finished(server);
	}
}

static uint32_t snapshots_counted(const char *directory, uint32_t *first_reach,
				  uint32_t *decoder_from)
{
	char path[PATH_BYTES];
	(void)snprintf(path, sizeof(path), "%s/v4", directory);
	DIR *const listing = opendir(path);
	uint32_t count = 0;
	expect(listing != NULL, "version directory exists");
	for (const struct dirent *entry = listing != NULL ? readdir(listing) :
							    NULL;
	     entry != NULL; entry = readdir(listing)) {
		if (strstr(entry->d_name, ".snapshot") != NULL) {
			struct test_snapshot_head head = { 0 };
			(void)snprintf(path, sizeof(path), "%s/v4/%s",
				       directory, entry->d_name);
			const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
			expect(descriptor >= 0 &&
				       read(descriptor, &head, sizeof(head)) ==
					       (ssize_t)sizeof(head),
			       "snapshot head");
			expect(head.version == SNAPSHOT_VERSION,
			       "snapshot records version four");
			if (head.token_count == FIRST_COUNT &&
			    first_reach != NULL) {
				*first_reach = head.reach_count;
			}
			if (decoder_from != NULL) {
				*decoder_from = head.decoder_from;
			}
			if (descriptor >= 0) {
				(void)close(descriptor);
			}
			count++;
		}
	}
	if (listing != NULL) {
		expect(closedir(listing) == 0, "version directory closes");
	}
	return count;
}

static void directory_removed(const char *directory)
{
	char path[PATH_BYTES], version[PATH_BYTES];
	(void)snprintf(version, sizeof(version), "%s/v4", directory);
	DIR *const listing = opendir(version);
	for (const struct dirent *entry = listing != NULL ? readdir(listing) :
							    NULL;
	     entry != NULL; entry = readdir(listing)) {
		if (entry->d_name[0] != '.') {
			(void)snprintf(path, sizeof(path), "%s/v4/%s",
				       directory, entry->d_name);
			expect(unlink(path) == 0,
			       "snapshot removed after test");
		}
	}
	if (listing != NULL) {
		expect(closedir(listing) == 0, "cleanup listing closes");
	}
	expect(rmdir(version) == 0 && rmdir(directory) == 0,
	       "test directory removed");
}

static void test_snapshot_replacement(void)
{
	/* Keep a valid indexed entry, then replace its namespace or token bytes on disk. */
	for (unsigned changes_tokens = 0; changes_tokens <= 1;
	     changes_tokens++) {
		char directory[] = "/tmp/anemone_snapshot_replacement_XXXXXX";
		char path[PATH_BYTES];
		struct inference_engine *engine = NULL;
		struct http_server server;
		expect(mkdtemp(directory) != NULL,
		       "replacement directory created");
		test_serving_set_cone(0);
		test_serving_set_function_model(false, 0);
		expect(test_serving_open(CONTEXT, scripts, 1, &engine) ==
			       STATUS_OK,
		       "replacement engine opens");
		store_opened(&server, engine, directory);
		prefilling(engine, 0, FIRST_COUNT);
		store_saved(&server, FIRST_COUNT);
		(void)snprintf(path, sizeof(path), "%s/v4", directory);
		DIR *const listing = opendir(path);
		bool changed = false;
		for (const struct dirent *entry =
			     listing != NULL ? readdir(listing) : NULL;
		     entry != NULL; entry = readdir(listing)) {
			if (strstr(entry->d_name, ".snapshot") == NULL) {
				continue;
			}
			(void)snprintf(path, sizeof(path), "%s/v4/%s",
				       directory, entry->d_name);
			const int descriptor = open(path, O_RDWR | O_CLOEXEC);
			struct test_snapshot_head head = { 0 };
			changed = descriptor >= 0 &&
				  pread(descriptor, &head, sizeof(head), 0) ==
					  (ssize_t)sizeof(head);
			if (changed && changes_tokens != 0) {
				const uint32_t different = tokens[0] ^ 1u;
				changed = pwrite(descriptor, &different,
						 sizeof(different),
						 (off_t)sizeof(head)) ==
					  (ssize_t)sizeof(different);
			} else if (changed) {
				head.cache_namespace.bytes[0] =
					SERVER_CACHE_USER;
				changed = pwrite(descriptor, &head,
						 sizeof(head),
						 0) == (ssize_t)sizeof(head);
			}
			if (descriptor >= 0) {
				(void)close(descriptor);
			}
			break;
		}
		if (listing != NULL) {
			(void)closedir(listing);
		}
		expect(changed,
		       "indexed snapshot file changed after admission");
		expect(!server_session_before(&server, 0, tokens,
					      PREFIX_COUNT + 1u, 0),
		       "a changed namespace or token prefix is refused before engine load");
		if (server.active[0].token_ids != NULL) {
			store_command_finished(&server);
		}
		expect(snapshots_counted(directory, NULL, NULL) == 1,
		       "a replaced file is forgotten without deleting another identity's file");
		server_sessions_close(&server);
		test_serving_close();
		directory_removed(directory);
	}
}

static void test_snapshot_coverage(void)
{
	char directory[] = "/tmp/anemone_snapshot_coverage_XXXXXX";
	struct inference_engine *engine = NULL;
	struct http_server server;
	uint32_t first_reach = 0;
	expect(mkdtemp(directory) != NULL, "coverage directory created");
	test_serving_set_cone(0);
	test_serving_set_function_model(false, 0);
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "coverage engine opens");
	store_opened(&server, engine, directory);
	prefilling(engine, 0, FIRST_COUNT);
	store_saved(&server, FIRST_COUNT);
	prefilling(engine, 0, FAR_REACH);
	prefilling(engine, FIRST_COUNT - SERVING_KEPT_GRANULE, FIRST_COUNT);
	store_saved(&server, FIRST_COUNT);
	expect(snapshots_counted(directory, &first_reach, NULL) == 1 &&
		       first_reach == FIRST_COUNT,
	       "an identical prefix retains the snapshot with the lower reach");
	prefilling(engine, FIRST_COUNT - SERVING_KEPT_GRANULE, LONGER_COUNT);
	store_saved(&server, LONGER_COUNT);
	expect(snapshots_counted(directory, &first_reach, NULL) ==
			       SERVING_PAIR_ROWS &&
		       first_reach == FIRST_COUNT,
	       "a longer snapshot preserves an older snapshot's earlier reusable prefixes");
	server_sessions_close(&server);
	store_opened(&server, engine, directory);
	uint8_t different_image[SERVER_IMAGE_IDENTITY_BYTES] = { 1 };
	expect(!server_session_before_images(&server, 0, tokens,
					     PREFIX_COUNT + 1u, 0,
					     different_image),
	       "a byte-identical token prefix with another image digest cannot load a snapshot");
	expect(server_session_before(&server, 0, tokens, PREFIX_COUNT + 1u, 0),
	       "the earlier prefix loads");
	expect(server.active[0].prompt_token_count == FIRST_COUNT &&
		       server.active[0].reach_count == FIRST_COUNT,
	       "the retained snapshot serves the prefix outside the replacement's interval");
	if (server.active[0].token_ids != NULL) {
		store_command_finished(&server);
	}
	static const uint8_t same_image[SERVER_IMAGE_IDENTITY_BYTES] = { 0 };
	expect(server_prefix_kept(&server, 0, tokens, PREFIX_COUNT + 1u,
				  same_image) != 0,
	       "matching tokens and image digest retain a reusable in-memory prefix");
	expect(server_prefix_kept(&server, 0, tokens, PREFIX_COUNT + 1u,
				  different_image) == 0,
	       "another image digest never retains that in-memory prefix");
	const uint32_t enabled_kept = server_prefix_kept(
		&server, 0, tokens, PREFIX_COUNT + 1u, same_image);
	server.disable_prefix_reuse = true;
	expect(server_prefix_kept(&server, 0, tokens, PREFIX_COUNT + 1u,
				  same_image) == 0 &&
		       server_prefix_kept(&server, 0, tokens, PREFIX_COUNT + 1u,
					  different_image) == 0,
	       "disabled GPU prefix reuse keeps no shared BOS, role or prompt tokens");
	server.disable_prefix_reuse = false;
	expect(enabled_kept != 0 &&
		       server_prefix_kept(&server, 0, tokens, PREFIX_COUNT + 1u,
					  same_image) == enabled_kept,
	       "the default prefix policy is unchanged after the explicit disabled control");
	uint8_t image_bytes[] = { 'a', 'b', 'c' };
	struct chat_images images = { .count = 1 };
	images.images[0].bytes = image_bytes;
	images.images[0].byte_count = sizeof(image_bytes);
	uint8_t cmp_identity[CHAT_IMAGE_IDENTITY_BYTES] = { 0 },
		pro_identity[CHAT_IMAGE_IDENTITY_BYTES] = { 0 };
	expect(chat_images_identity(&images, ARITHMETIC_CLASS_SM_80,
				    cmp_identity) == STATUS_OK &&
		       chat_images_identity(&images,
					    ARITHMETIC_CLASS_SM_120_NATIVE,
					    pro_identity) == STATUS_OK,
	       "same image bytes have identities for each encoder arithmetic class");
	memcpy(server.slot_histories[0].image_identity, cmp_identity,
	       sizeof(cmp_identity));
	expect(server_prefix_kept(&server, 0, tokens, PREFIX_COUNT + 1u,
				  cmp_identity) != 0 &&
		       server_prefix_kept(&server, 0, tokens, PREFIX_COUNT + 1u,
					  pro_identity) == 0,
	       "identical image bytes reuse their matching encoder arithmetic and reject another class");
	server_sessions_close(&server);
	test_serving_close();
	directory_removed(directory);
}

static void test_version_isolation(void)
{
	char directory[] = "/tmp/anemone_snapshot_versions_XXXXXX";
	char path[PATH_BYTES];
	char unscoped_directory[PATH_BYTES], unscoped_path[PATH_BYTES];
	static const char unscoped_bytes[] =
		"unscoped version-three snapshot fixture";
	struct inference_engine *engine = NULL;
	struct http_server server;
	struct test_snapshot_head head = { .version = LEGACY_VERSION };
	memcpy(head.magic, "anemone snapshot", SNAPSHOT_MAGIC_BYTES);
	expect(mkdtemp(directory) != NULL, "version directory created");
	(void)snprintf(unscoped_directory, sizeof(unscoped_directory), "%s/v3",
		       directory);
	(void)snprintf(unscoped_path, sizeof(unscoped_path),
		       "%s/v3/unscoped.snapshot", directory);
	expect(mkdir(unscoped_directory, S_IRWXU) == 0,
	       "old unscoped snapshot directory");
	const int unscoped =
		open(unscoped_path, O_WRONLY | O_CREAT | O_CLOEXEC, FILE_MODE);
	expect(unscoped >= 0 && write(unscoped, unscoped_bytes,
				      sizeof(unscoped_bytes)) ==
					(ssize_t)sizeof(unscoped_bytes),
	       "unscoped snapshot marker");
	if (unscoped >= 0) {
		expect(close(unscoped) == 0, "unscoped marker closes");
	}
	(void)snprintf(path, sizeof(path), "%s/rollback.snapshot", directory);
	const int descriptor =
		open(path, O_WRONLY | O_CREAT | O_CLOEXEC, FILE_MODE);
	expect(descriptor >= 0 && write(descriptor, &head, sizeof(head)) ==
					  (ssize_t)sizeof(head),
	       "legacy file written");
	if (descriptor >= 0) {
		expect(close(descriptor) == 0, "legacy file closes");
	}
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "version engine opens");
	store_opened(&server, engine, directory);
	expect(!server_session_before(&server, 0, tokens, FIRST_COUNT, 0),
	       "old unscoped snapshots cannot supply scoped state");
	server_sessions_close(&server);
	struct stat status;
	expect(stat(path, &status) == 0 &&
		       status.st_size == (off_t)sizeof(head),
	       "production's version one file survives");
	const int restored = open(path, O_RDONLY | O_CLOEXEC);
	struct test_snapshot_head after = { 0 };
	expect(restored >= 0 &&
		       read(restored, &after, sizeof(after)) ==
			       (ssize_t)sizeof(after) &&
		       memcmp(&head, &after, sizeof(head)) == 0,
	       "legacy snapshot bytes unchanged for rollback");
	if (restored >= 0) {
		(void)close(restored);
	}
	expect(unlink(path) == 0, "legacy test file removed");
	char unscoped_after[sizeof(unscoped_bytes)] = { 0 };
	const int old_file = open(unscoped_path, O_RDONLY | O_CLOEXEC);
	expect(old_file >= 0 &&
		       read(old_file, unscoped_after, sizeof(unscoped_after)) ==
			       (ssize_t)sizeof(unscoped_after) &&
		       memcmp(unscoped_after, unscoped_bytes,
			      sizeof(unscoped_bytes)) == 0,
	       "the version-three namespace-less snapshot remains byte-for-byte untouched");
	if (old_file >= 0) {
		expect(close(old_file) == 0,
		       "unscoped preserved marker closes");
	}
	expect(unlink(unscoped_path) == 0 && rmdir(unscoped_directory) == 0,
	       "unscoped test fixture removed");
	test_serving_close();
	directory_removed(directory);
}

/* FRONT rows leave the decoder boundary beyond the cancelled prefix. Saving the prefix must preserve that fact:
 * continuing its entire prefix still needs cone replay before a draw, as a fresh prefill's bits demonstrate. */
static void test_cancelled_prefill(void)
{
	char directory[] = "/tmp/anemone_snapshot_cancelled_XXXXXX";
	struct inference_engine *engine = NULL;
	struct http_server server;
	struct session_state saved = { 0 }, loaded = { 0 };
	uint32_t saved_decoder = 0, resumed_token = UINT32_MAX,
		 fresh_token = UINT32_MAX;
	expect(mkdtemp(directory) != NULL,
	       "cancelled prefill directory created");
	test_serving_set_cone(CONE_ROWS);
	test_serving_set_ring(MODEL_RING_SLOTS, MODEL_WINDOW);
	test_serving_set_function_model(true, 0);
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "cancelled prefill engine opens");
	store_opened(&server, engine, directory);
	test_serving_set_step_delay(STEP_DELAY);
	const struct generation_command_record command = {
		.new_token_ids = tokens,
		.new_token_count = CANCEL_PROMPT,
		.generated_token_limit = 1,
		.sampler_top_p = 1.0f
	};
	expect(engine_submit(engine, &command) == STATUS_OK,
	       "long prefill begins");
	for (uint32_t attempt = 0;
	     attempt < WAIT_ATTEMPTS &&
	     test_serving_front_rows_seen() < SAVE_MINIMUM;
	     attempt++) {
		const struct timespec pause = { 0, WAIT_PAUSE };
		expect(nanosleep(&pause, NULL) == 0, "waiting for FRONT rows");
	}
	expect(test_serving_front_rows_seen() >= SAVE_MINIMUM &&
		       engine_cancel(engine, 0) == STATUS_OK,
	       "cancel after enough FRONT rows to save");
	const struct generation_event cancelled = final_taken(engine, NULL);
	const uint32_t count = cancelled.session_token_count;
	expect(cancelled.event_kind == GENERATION_EVENT_CANCELLED &&
		       test_serving_rows_seen() == 0,
	       "prefill cancelled before any decoder rows");
	expect(engine_session_state(engine, 0, &saved) == STATUS_OK &&
		       saved.decoder_from > count,
	       "the actual decoder boundary is beyond the processed prefix");
	store_saved(&server, count);
	expect(snapshots_counted(directory, NULL, &saved_decoder) == 1 &&
		       saved_decoder == saved.decoder_from,
	       "the version four snapshot head persists the real decoder boundary");
	server_sessions_close(&server);
	test_serving_close();
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "engine restarts after cancellation");
	store_opened(&server, engine, directory);
	expect(server_session_before(&server, 0, tokens, count + 1u, 0),
	       "cancelled prefix snapshot selected");
	if (server.active[0].token_ids != NULL) {
		store_command_finished(&server);
	}
	expect(engine_session_state(engine, 0, &loaded) == STATUS_OK &&
		       loaded.decoder_from == saved.decoder_from,
	       "load restores the boundary beyond its token count");
	expect(server.slot_histories[0].from_snapshot &&
		       server.slot_histories[0].cold_prompt_count == 0,
	       "a real snapshot load retains unknown cold provenance for the exact policy");
	const struct generation_command_record continued = {
		.new_token_ids = tokens + count,
		.kept_token_count = count,
		.new_token_count = 1,
		.generated_token_limit = 1,
		.sampler_top_p = 1.0f
	};
	expect(engine_submit(engine, &continued) == STATUS_OK,
	       "entire loaded prefix kept");
	expect(final_taken(engine, &resumed_token).status == STATUS_OK &&
		       test_serving_rows_seen() > 1,
	       "a short continuation replays the decoder cone");
	server_sessions_close(&server);
	test_serving_close();
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "fresh reference engine opens");
	const struct generation_command_record fresh = {
		.new_token_ids = tokens,
		.new_token_count = count + 1u,
		.generated_token_limit = 1,
		.sampler_top_p = 1.0f
	};
	expect(engine_submit(engine, &fresh) == STATUS_OK,
	       "fresh prefill submitted");
	expect(final_taken(engine, &fresh_token).status == STATUS_OK &&
		       resumed_token == fresh_token,
	       "snapshot continuation draws the fresh prefill's exact token");
	test_serving_close();
	test_serving_set_cone(0);
	test_serving_set_ring(TEST_SERVING_STATE_POSITIONS,
			      TEST_SERVING_WINDOW);
	test_serving_set_function_model(false, 0);
	directory_removed(directory);
}

static void test_exact_prefix_policy(void)
{
	const uint32_t prompt = LINEAR_MAXIMUM_ROWS + 1u,
		       boundary = prompt - 1u;
	static const uint8_t no_image[SERVER_IMAGE_IDENTITY_BYTES] = { 0 };
	struct http_server server = { .histories = histories,
				      .exact_prefix_reuse = true,
				      .description = { .context_token_limit =
							       CONTEXT } };
	memcpy(histories, tokens, (size_t)prompt * sizeof(*tokens));
	struct slot_history *const history = &server.slot_histories[0];
	*history = (struct slot_history){ .token_count = prompt,
					  .processed_count = boundary,
					  .reach_count = prompt };
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) == 0,
	       "unknown numerical provenance declines an otherwise matching repeat");
	history->cold_prompt_count = prompt;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) ==
		       boundary,
	       "same cold plan retains every row before the final THIN row at a closed pair");
	history->decoder_from = boundary + 1u;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) == 0,
	       "a stale decoder boundary cannot cause an unplanned cone replay");
	history->decoder_from = 0;
	history->cold_prompt_count = prompt - 1u;
	expect(server_prefix_kept(&server, 0, tokens, prompt - 1u, no_image) ==
		       0,
	       "an even-length prompt has no closed pair immediately before its final row");
	history->cold_prompt_count = prompt;
	tokens[SERVING_PAIR_ROWS]++;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) == 0,
	       "a partial shared prefix cannot substitute for a complete cold prompt boundary");
	server.exact_prefix_reuse = false;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) ==
		       SERVING_PAIR_ROWS,
	       "the old shared-prefix policy remains selectable");
	server.exact_prefix_reuse = true;
	tokens[SERVING_PAIR_ROWS]--;
	history->processed_count--;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) == 0,
	       "a cancelled prefill that did not process the boundary declines reuse");
	history->processed_count++;
	history->from_snapshot = true;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) == 0,
	       "imported state cannot manufacture cold-plan provenance");
	history->from_snapshot = false;
	uint8_t image[SERVER_IMAGE_IDENTITY_BYTES] = { 1 };
	memcpy(history->image_identity, image, sizeof(image));
	expect(server_prefix_kept(&server, 0, tokens, prompt, image) == 0,
	       "image reuse remains cold until its plan provenance is represented");
	memset(history->image_identity, 0, sizeof(history->image_identity));
	history->reach_count = boundary + SERVING_KEPT_LAG + 1u;
	history->checkpoint_count = boundary - SERVING_KEPT_GRANULE;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) == 0,
	       "an older checkpoint cannot bypass the exact boundary rule");
	history->checkpoint_count = boundary;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) ==
		       boundary,
	       "the exact prompt checkpoint remains reusable beyond ordinary ring reach");
	server.disable_prefix_reuse = true;
	expect(server_prefix_kept(&server, 0, tokens, prompt, no_image) == 0,
	       "explicit zero reuse still overrides the exact policy");
}

int main(void)
{
	for (uint32_t position = 0; position < CONTEXT; position++) {
		tokens[position] =
			(position * TOKEN_STRIDE + 1u) % TOKEN_VALUES;
	}
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
	test_snapshot_coverage();
	test_snapshot_replacement();
	test_version_isolation();
	test_cancelled_prefill();
	test_exact_prefix_policy();
	if (failures != 0) {
		return EXIT_FAILURE;
	}
	(void)puts(
		"serving snapshots: coverage, version isolation and cancelled prefill passed");
	return EXIT_SUCCESS;
}
