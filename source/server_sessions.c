/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * server_sessions.c: the server's snapshots of sessions in a directory, so that a session outlives a restart or the
 * slot another session takes (front_end_seam.h's session commands; step.h's snapshots). A snapshot file holds a head
 * (the identity it fits, its token count, reach and decoder boundary, its payload's bytes), its tokens and the engine's
 * payload; it is
 * written as <name>.partial and renamed once whole. At run the store reads the head and tokens of every snapshot that
 * fits the engine in its version's subdirectory; other versions' directories and files are untouched. The policy,
 * one session command at a time:
 *   - before a request takes a slot whose session it would lose (it keeps less than SERVING_KEPT_LAG behind
 *     what the slot ran), that session is saved if at least session_save_tokens of it are in no snapshot;
 *   - a snapshot that keeps session_load_tokens more of the request than its slot does is loaded into the slot first;
 *   - after session_idle_save_seconds with no command, unsaved sessions are saved one by one; at the end, all of them.
 * A new snapshot supersedes those whose reusable prefixes it covers entirely; beyond the byte
 * budget the least recently used go. A failed load removes its snapshot; a failed save counts the session as saved,
 * so that neither is tried again and again.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "server_internal.h"

#define OK STATUS_OK
#define END_SAVE_SECONDS 600.0                      /* every session saved at the end, at most */
#define MOST_SNAPSHOTS 64u
#define NAME_CAPACITY 64u
#define PATH_CAPACITY 4096u
#define SNAPSHOT_MAGIC "anemone snapshot"
#define SNAPSHOT_MAGIC_BYTES 16u
#define SNAPSHOT_VERSION 4u
#define SNAPSHOT_SUBDIRECTORY "v4"
#define SNAPSHOT_SUFFIX ".snapshot"
#define PARTIAL_SUFFIX ".partial"
#define DIRECTORY_MODE 0700
#define FILE_MODE 0600
#define HASH_OFFSET 0xCBF29CE484222325ull           /* FNV-1a, 64 bits */
#define HASH_PRIME 0x100000001B3ull
#define MILLISECONDS_PER_SECOND 1000.0

struct snapshot_head {
    char magic[SNAPSHOT_MAGIC_BYTES];
    uint64_t identity;
    uint64_t payload_bytes;                         /* 0 until the payload is whole */
    uint32_t version;
    uint32_t token_count;
    uint32_t reach_count;                           /* the saving slot's (front_end_seam.h) */
    uint32_t decoder_from;                          /* may exceed token_count after a cancelled FRONT prefill */
    uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES];
    struct cache_namespace cache_namespace;
};

struct snapshot {
    char name[NAME_CAPACITY];                       /* in the directory */
    uint32_t *token_ids;
    uint32_t token_count, reach_count, decoder_from;
    uint64_t file_bytes;
    uint64_t last_used;
    uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES];
    struct cache_namespace cache_namespace;
};

struct session_store {
    char directory[PATH_CAPACITY];
    uint64_t identity, byte_budget, uses;
    uint32_t context_limit, snapshot_count;
    struct snapshot snapshots[MOST_SNAPSHOTS];
    char saving[NAME_CAPACITY];                     /* the snapshot a save writes, with its tokens */
    uint32_t *saving_token_ids;
    uint32_t saving_token_count, saving_reach_count, saving_decoder_from;
    uint8_t saving_image_identity[SERVER_IMAGE_IDENTITY_BYTES];
    struct cache_namespace saving_namespace;
};

static uint64_t hash_of(uint64_t hash, const void *bytes, size_t byte_count)
{
    for (size_t index = 0; index < byte_count; index++) {
        hash = (hash ^ ((const unsigned char *)bytes)[index]) * HASH_PRIME;
    }
    return hash;
}

static bool path_of(const struct session_store *store, const char *name, const char *suffix, char path[PATH_CAPACITY])
{
    const int length = snprintf(path, PATH_CAPACITY, "%s/%s%s", store->directory, name, suffix);
    return length > 0 && (size_t)length < PATH_CAPACITY;
}

/* Whole reads and writes at an offset of a file; false for less. */
static bool bytes_moved(int descriptor, void *bytes, size_t byte_count, off_t offset, bool is_write)
{
    while (byte_count != 0) {
        const ssize_t moved = is_write ? pwrite(descriptor, bytes, byte_count, offset)
                                       : pread(descriptor, bytes, byte_count, offset);
        if (moved <= 0 && !(moved < 0 && errno == EINTR)) {
            return false;
        }
        bytes = (unsigned char *)bytes + (moved > 0 ? moved : 0);
        byte_count -= moved > 0 ? (size_t)moved : 0;
        offset += moved > 0 ? moved : 0;
    }
    return true;
}

/* The tokens a request keeps of a session of these tokens and reach, as kept_tokens keeps of a slot that ran them. */
static uint32_t snapshot_kept(const uint32_t *held, uint32_t held_count, uint32_t reach_count,
                              const uint32_t *token_ids, uint32_t token_count)
{
    const uint32_t most = held_count < token_count ? held_count : token_count;
    uint32_t common = 0;
    while (common < most && held[common] == token_ids[common]) {
        common++;
    }
    uint32_t kept = common < token_count ? common : token_count - 1u;
    kept -= kept % SERVING_KEPT_GRANULE;
    return reach_count - kept <= SERVING_KEPT_LAG ? kept : 0u;
}

static void snapshot_forgotten(struct session_store *store, uint32_t index)
{
    free(store->snapshots[index].token_ids);
    store->snapshots[index] = store->snapshots[--store->snapshot_count];
}

static void snapshot_removed(struct session_store *store, uint32_t index)
{
    char path[PATH_CAPACITY];
    if (path_of(store, store->snapshots[index].name, SNAPSHOT_SUFFIX, path)) {
        (void)unlink(path);
    }
    snapshot_forgotten(store, index);
}

/* A snapshot file's head and tokens, if it fits the store and is whole; invalid files of this version are removed. */
static void snapshot_read(struct session_store *store, const char *file_name)
{
    char path[PATH_CAPACITY];
    struct snapshot_head head;
    struct stat status;
    const size_t name_length = strlen(file_name) - strlen(SNAPSHOT_SUFFIX);
    struct snapshot *const snapshot = &store->snapshots[store->snapshot_count];
    if (name_length >= NAME_CAPACITY || store->snapshot_count == MOST_SNAPSHOTS ||
        snprintf(path, sizeof(path), "%s/%s", store->directory, file_name) >= (int)sizeof(path)) {
        return;
    }
    const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    const bool has_head = descriptor >= 0 && bytes_moved(descriptor, &head, sizeof(head), 0, false);
    const bool other_version = has_head && memcmp(head.magic, SNAPSHOT_MAGIC, SNAPSHOT_MAGIC_BYTES) == 0 &&
                               head.version != SNAPSHOT_VERSION;
    bool fits = has_head && fstat(descriptor, &status) == 0 &&
		memcmp(head.magic, SNAPSHOT_MAGIC, SNAPSHOT_MAGIC_BYTES) == 0 &&
		head.version == SNAPSHOT_VERSION &&
		head.identity == store->identity && head.token_count != 0 &&
		head.token_count <= head.reach_count &&
		head.reach_count <= store->context_limit &&
		head.decoder_from <= store->context_limit &&
		head.cache_namespace.byte_count != 0 &&
		head.cache_namespace.byte_count <=
			SERVER_CACHE_NAMESPACE_BYTES &&
		head.cache_namespace.has_session <= 1 &&
		head.cache_namespace.bytes[0] < SERVER_CACHE_SCOPES &&
		head.payload_bytes != 0 &&
		(uint64_t)status.st_size ==
			sizeof(head) +
				(uint64_t)head.token_count * sizeof(uint32_t) +
				head.payload_bytes;
    uint32_t *const token_ids = fits ? malloc((size_t)head.token_count * sizeof(uint32_t)) : NULL;
    fits = token_ids != NULL &&
           bytes_moved(descriptor, token_ids, (size_t)head.token_count * sizeof(uint32_t), sizeof(head), false);
    if (descriptor >= 0) {
        (void)close(descriptor);
    }
    if (!fits) {
        free(token_ids);
        if (!other_version) {
            (void)unlink(path);
        }
        return;
    }
    *snapshot = (struct snapshot){.token_ids = token_ids, .token_count = head.token_count,
                                  .reach_count = head.reach_count, .file_bytes = (uint64_t)status.st_size,
                                  .decoder_from = head.decoder_from,
                                  .last_used = ++store->uses};
    memcpy(snapshot->image_identity, head.image_identity, sizeof(snapshot->image_identity));
    snapshot->cache_namespace = head.cache_namespace;
    memcpy(snapshot->name, file_name, name_length);
    store->snapshot_count++;
}

result_status server_sessions_open(struct http_server *server)
{
    if (server->session_directory == NULL) {
        return OK;
    }
    struct session_store *const store = calloc(1, sizeof(*store));
    if (store == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    const bool has_path = snprintf(store->directory, sizeof(store->directory), "%s/%s", server->session_directory,
                                   SNAPSHOT_SUBDIRECTORY) < (int)sizeof(store->directory);
    DIR *const directory = !has_path ||
                          (mkdir(server->session_directory, DIRECTORY_MODE) != 0 && errno != EEXIST) ||
                          (mkdir(store->directory, DIRECTORY_MODE) != 0 && errno != EEXIST)
                              ? NULL : opendir(store->directory);
    if (directory == NULL) {
	    LOG_ERROR("server: the session directory cannot be opened");
	    free(store);
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    store->identity = hash_of(server->description.snapshot_identity ^ HASH_OFFSET,
                              server->description.front_end_tables, server->description.front_end_table_byte_count);
    store->byte_budget = server->session_directory_bytes;
    store->context_limit = server->description.context_token_limit;
    for (const struct dirent *entry = readdir(directory); entry != NULL; entry = readdir(directory)) {
        const size_t length = strlen(entry->d_name);
        char path[PATH_CAPACITY];
        if (length > strlen(SNAPSHOT_SUFFIX) &&
            strcmp(entry->d_name + length - strlen(SNAPSHOT_SUFFIX), SNAPSHOT_SUFFIX) == 0) {
            snapshot_read(store, entry->d_name);
        } else if (length > strlen(PARTIAL_SUFFIX) &&
                   strcmp(entry->d_name + length - strlen(PARTIAL_SUFFIX), PARTIAL_SUFFIX) == 0 &&
                   snprintf(path, sizeof(path), "%s/%s", store->directory, entry->d_name) < (int)sizeof(path)) {
            (void)unlink(path);                      /* a save that never ended */
        }
    }
    (void)closedir(directory);
    server->store = store;
    return OK;
}

void server_sessions_close(struct http_server *server)
{
    struct session_store *const store = server->store;
    for (uint32_t index = 0; store != NULL && index < store->snapshot_count; index++) {
        free(store->snapshots[index].token_ids);
    }
    if (store != NULL) {
        free(store->saving_token_ids);
    }
    free(store);
    server->store = NULL;
}

/* Bind the opened inode to its indexed identity and tokens. A filename can outlive that binding after another
 * process replaces a cache file; such a file is a miss, not permission to load or delete its replacement. */
static bool snapshot_matches(int descriptor, const struct session_store *store,
			     const struct snapshot *snapshot,
			     uint32_t *token_ids, size_t token_bytes)
{
	struct snapshot_head head;
	struct stat status;
	const uint64_t front_bytes = sizeof(head) + (uint64_t)token_bytes;
	return fstat(descriptor, &status) == 0 &&
	       (uint64_t)status.st_size == snapshot->file_bytes &&
	       snapshot->file_bytes > front_bytes &&
	       bytes_moved(descriptor, &head, sizeof(head), 0, false) &&
	       memcmp(head.magic, SNAPSHOT_MAGIC, SNAPSHOT_MAGIC_BYTES) == 0 &&
	       head.version == SNAPSHOT_VERSION &&
	       head.identity == store->identity &&
	       head.token_count == snapshot->token_count &&
	       head.reach_count == snapshot->reach_count &&
	       head.decoder_from == snapshot->decoder_from &&
	       head.payload_bytes == snapshot->file_bytes - front_bytes &&
	       memcmp(head.image_identity, snapshot->image_identity,
		      sizeof(head.image_identity)) == 0 &&
	       head.cache_namespace.has_session ==
		       snapshot->cache_namespace.has_session &&
	       server_cache_namespace_equal(&head.cache_namespace,
					    &snapshot->cache_namespace) &&
	       bytes_moved(descriptor, token_ids, token_bytes,
			   (off_t)sizeof(head), false) &&
	       memcmp(token_ids, snapshot->token_ids, token_bytes) == 0;
}

/* A session command for the slot: a save of what it ran (to the kept granule), or a load of snapshot `snapshot`. False
 * if none began: a save then counts as done, a snapshot that cannot be loaded goes. */
static bool session_started(struct http_server *server, uint32_t session_slot,
			    uint32_t operation, uint32_t snapshot)
{
    struct session_store *const store = server->store;
    struct slot_history *const history = &server->slot_histories[session_slot];
    const bool is_save = operation == SESSION_SAVE;
    const uint32_t token_count =
	    is_save ? history->processed_count -
			      history->processed_count % SERVING_KEPT_GRANULE :
		      store->snapshots[snapshot].token_count;
    struct session_state state = { 0 };
    if (is_save &&
	engine_session_state(server->engine, session_slot, &state) != OK) {
	    history->saved_count = history->processed_count;
	    return false;
    }
    const uint32_t reach_count = is_save ? state.reach_count : store->snapshots[snapshot].reach_count;
    const uint32_t decoder_from = is_save ? state.decoder_from : store->snapshots[snapshot].decoder_from;
    const uint32_t *const source = is_save ? server->histories + (size_t)session_slot *
                                                                     server->description.context_token_limit
                                           : store->snapshots[snapshot].token_ids;
    const size_t token_bytes = (size_t)token_count * sizeof(uint32_t);
    const off_t payload = (off_t)(sizeof(struct snapshot_head) + token_bytes);
    char path[PATH_CAPACITY];
    uint32_t *const token_ids = token_count != 0 ? malloc(token_bytes) : NULL;
    int descriptor = -1;
    bool is_replaced = false;
    if (token_ids != NULL && is_save) {
	    memcpy(token_ids, source, token_bytes);
    }
    if (token_ids != NULL && is_save) {
	    struct snapshot_head head = { .identity = store->identity,
					  .version = SNAPSHOT_VERSION,
					  .token_count = token_count,
					  .reach_count = reach_count,
					  .decoder_from = decoder_from,
					  .cache_namespace =
						  history->cache_namespace };
	    memcpy(head.image_identity, history->image_identity,
		   sizeof(head.image_identity));
	    memcpy(head.magic, SNAPSHOT_MAGIC, SNAPSHOT_MAGIC_BYTES);
	    (void)snprintf(store->saving, sizeof(store->saving),
			   "%016" PRIx64 "-%016" PRIx64 "-%" PRIu32,
			   hash_of(HASH_OFFSET, head.cache_namespace.bytes,
				   head.cache_namespace.byte_count),
			   hash_of(hash_of(HASH_OFFSET, head.image_identity,
					   sizeof(head.image_identity)),
				   token_ids, token_bytes),
			   token_count);
	    descriptor = path_of(store, store->saving, PARTIAL_SUFFIX, path) ?
				 open(path,
				      O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
				      FILE_MODE) :
				 -1;
	    if (descriptor >= 0 &&
		(!bytes_moved(descriptor, &head, sizeof(head), 0, true) ||
		 !bytes_moved(descriptor, token_ids, token_bytes, sizeof(head),
			      true) ||
		 lseek(descriptor, payload, SEEK_SET) != payload)) {
		    (void)close(descriptor);
		    (void)unlink(path);
		    descriptor = -1;
	    }
    } else if (token_ids != NULL && path_of(store, store->snapshots[snapshot].name, SNAPSHOT_SUFFIX, path)) {
        descriptor = open(path, O_RDONLY | O_CLOEXEC);
	is_replaced = descriptor >= 0 &&
		      !snapshot_matches(descriptor, store,
					&store->snapshots[snapshot], token_ids,
					token_bytes);
	if (descriptor >= 0 &&
	    (is_replaced || lseek(descriptor, payload, SEEK_SET) != payload)) {
		(void)close(descriptor);
		descriptor = -1;
	}
	store->snapshots[snapshot].last_used = ++store->uses;
    }
    const uint32_t request_tag = server_next_tag(server);
    const struct session_command command = {
	    .token_ids = is_save ? NULL : token_ids,
	    .descriptor = descriptor,
	    .session_slot = session_slot,
	    .token_count = token_count,
	    .operation = operation,
	    .request_tag = request_tag,
	    .reach_count = is_save ? 0 : reach_count,
	    .decoder_from = is_save ? 0 : decoder_from
    };
    if (descriptor < 0 || engine_session(server->engine, &command) != OK) {
	    LOG_ERROR("server: a session could not be %s",
		      is_save ? "saved" : "loaded");
	    if (descriptor >= 0) {
		    (void)close(descriptor);
	    }
	    if (descriptor >= 0 && is_save &&
		path_of(store, store->saving, PARTIAL_SUFFIX, path)) {
		    (void)unlink(path);
	    }
	    free(token_ids);
	    history->saved_count = is_save ? history->processed_count :
					     history->saved_count;
	    if (!is_save) {
		    if (is_replaced) {
			    snapshot_forgotten(store, snapshot);
		    } else {
			    snapshot_removed(store, snapshot);
		    }
	    }
	    return false;
    }
    server->active[session_slot] = (struct active_command){
        .connection = NO_CONNECTION, .token_ids = token_ids, .request_tag = request_tag,
        .prompt_token_count = is_save ? history->token_count : token_count, .operation = operation,
        .descriptor = descriptor, .reach_count = reach_count};
    memcpy(server->active[session_slot].image_identity,
           is_save ? history->image_identity : store->snapshots[snapshot].image_identity,
           sizeof(history->image_identity));
    server->active[session_slot].cache_namespace =
	    is_save ? history->cache_namespace :
		      store->snapshots[snapshot].cache_namespace;
    if (is_save) {
        memcpy(store->saving_image_identity, history->image_identity, sizeof(history->image_identity));
	store->saving_namespace = history->cache_namespace;
    }
    store->saving_token_ids = is_save ? token_ids : store->saving_token_ids;
    store->saving_token_count = is_save ? token_count : store->saving_token_count;
    store->saving_reach_count = is_save ? reach_count : store->saving_reach_count;
    store->saving_decoder_from = is_save ? decoder_from : store->saving_decoder_from;
    return true;
}

static bool session_before(struct http_server *server, uint32_t session_slot,
			   const uint32_t *token_ids, uint32_t token_count,
			   uint32_t kept, const uint8_t *image_identity,
			   const struct cache_namespace *identity)
{
    struct session_store *const store = server->store;
    const struct slot_history *const history = &server->slot_histories[session_slot];
    if (store == NULL) {
        return false;
    }
    const bool same =
	    server_cache_namespace_equal(&history->cache_namespace, identity);
    if (history->processed_count - history->saved_count >=
		server->session_save_tokens &&
	(!same ||
	 (kept <= history->processed_count &&
	  history->processed_count - kept >= server->session_save_tokens)) &&
	store->saving_token_ids == NULL &&
	session_started(server, session_slot, SESSION_SAVE, 0)) {
	    return true;
    }
    uint32_t best = MOST_SNAPSHOTS, best_kept = 0;
    for (uint32_t index = 0; index < store->snapshot_count; index++) {
        const struct snapshot *const candidate = &store->snapshots[index];
	const uint32_t snapshot_keeps =
		!server_cache_namespace_equal(&candidate->cache_namespace,
					      identity) ||
				memcmp(candidate->image_identity,
				       image_identity,
				       sizeof(candidate->image_identity)) != 0 ?
			0 :
			snapshot_kept(candidate->token_ids,
				      candidate->token_count,
				      candidate->reach_count, token_ids,
				      token_count);
	best = snapshot_keeps > best_kept ? index : best;
	best_kept = snapshot_keeps > best_kept ? snapshot_keeps : best_kept;
    }
    return best < MOST_SNAPSHOTS && best_kept > kept &&
	   best_kept - kept >= server->session_load_tokens &&
	   session_started(server, session_slot, SESSION_LOAD, best);
}

bool server_session_before_namespace(struct http_server *server,
				     uint32_t session_slot,
				     const struct connection *connection,
				     uint32_t kept)
{
	return session_before(server, session_slot, connection->token_ids,
			      connection->token_count, kept,
			      connection->image_identity,
			      &connection->cache_namespace);
}

bool server_session_before_images(
	struct http_server *server, uint32_t session_slot,
	const uint32_t *token_ids, uint32_t token_count, uint32_t kept,
	const uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES])
{
	return session_before(
		server, session_slot, token_ids, token_count, kept,
		image_identity,
		&server->slot_histories[session_slot].cache_namespace);
}

bool server_session_before(struct http_server *server, uint32_t session_slot,
			   const uint32_t *token_ids, uint32_t token_count,
			   uint32_t kept)
{
	static const uint8_t no_images[SERVER_IMAGE_IDENTITY_BYTES] = { 0 };
	return server_session_before_images(server, session_slot, token_ids,
					    token_count, kept, no_images);
}

/* The first positive reusable granule. Together with token_count it bounds every prefix a snapshot serves. */
static uint32_t first_kept(uint32_t reach_count)
{
	const uint32_t first = reach_count > SERVING_KEPT_LAG ?
				       reach_count - SERVING_KEPT_LAG :
				       1u;
	return (first + SERVING_KEPT_GRANULE - 1u) / SERVING_KEPT_GRANULE *
	       SERVING_KEPT_GRANULE;
}

static bool saving_covers(const struct session_store *store, const struct snapshot *older)
{
	return server_cache_namespace_equal(&older->cache_namespace,
					    &store->saving_namespace) &&
	       memcmp(older->image_identity, store->saving_image_identity,
		      sizeof(older->image_identity)) == 0 &&
	       older->token_count <= store->saving_token_count &&
	       first_kept(store->saving_reach_count) <=
		       first_kept(older->reach_count) &&
	       memcmp(older->token_ids, store->saving_token_ids,
		      (size_t)older->token_count * sizeof(uint32_t)) == 0;
}

/* A save that ended: its file kept and indexed, what it supersedes and what passes the budget removed; or removed. */
static void save_ended(struct session_store *store, int descriptor, bool is_saved)
{
    char partial[PATH_CAPACITY], whole[PATH_CAPACITY];
    const off_t end = lseek(descriptor, 0, SEEK_CUR);
    uint64_t payload = (uint64_t)end - sizeof(struct snapshot_head) -
                       (uint64_t)store->saving_token_count * sizeof(uint32_t);
    const bool has_paths = path_of(store, store->saving, PARTIAL_SUFFIX, partial) &&
                           path_of(store, store->saving, SNAPSHOT_SUFFIX, whole);
    is_saved = is_saved && end > 0 && payload != 0 && has_paths &&
               bytes_moved(descriptor, &payload, sizeof(payload), offsetof(struct snapshot_head, payload_bytes),
                           true) &&
               fsync(descriptor) == 0;
    (void)close(descriptor);
    for (uint32_t index = 0; is_saved && index < store->snapshot_count; index++) {
        const struct snapshot *const older = &store->snapshots[index];
        if (strcmp(older->name, store->saving) == 0 &&
            (!saving_covers(store, older) || store->saving_reach_count >= older->reach_count)) {
            is_saved = false;                       /* an identical prefix keeps its lower reach */
        }
    }
    bool replaces = false;
    for (uint32_t index = 0; is_saved && index < store->snapshot_count; index++) {
        replaces = replaces || saving_covers(store, &store->snapshots[index]);
    }
    is_saved = is_saved && (store->snapshot_count < MOST_SNAPSHOTS || replaces) && rename(partial, whole) == 0;
    for (uint32_t index = store->snapshot_count; is_saved && index-- > 0;) {
        const struct snapshot *const older = &store->snapshots[index];
        if (saving_covers(store, older)) {
            if (strcmp(older->name, store->saving) == 0) {
                snapshot_forgotten(store, index);     /* rename already replaced this file atomically */
            } else {
                snapshot_removed(store, index);
            }
        }
    }
    if (is_saved) {
        struct snapshot *const snapshot = &store->snapshots[store->snapshot_count++];
        *snapshot = (struct snapshot){.token_ids = store->saving_token_ids, .token_count = store->saving_token_count,
                                      .reach_count = store->saving_reach_count, .file_bytes = (uint64_t)end,
                                      .decoder_from = store->saving_decoder_from,
                                      .last_used = ++store->uses};
        memcpy(snapshot->image_identity, store->saving_image_identity, sizeof(snapshot->image_identity));
	snapshot->cache_namespace = store->saving_namespace;
	memcpy(snapshot->name, store->saving, sizeof(snapshot->name));
	store->saving_token_ids = NULL; /* the snapshot's now */
    } else if (has_paths) {
        (void)unlink(partial);
    }
    uint64_t total = 0;
    for (uint32_t index = 0; index < store->snapshot_count; index++) {
        total += store->snapshots[index].file_bytes;
    }
    while (total > store->byte_budget && store->snapshot_count > 1) {
        uint32_t oldest = 0;
        for (uint32_t index = 1; index < store->snapshot_count; index++) {
            oldest = store->snapshots[index].last_used < store->snapshots[oldest].last_used ? index : oldest;
        }
        total -= store->snapshots[oldest].file_bytes;
        snapshot_removed(store, oldest);
    }
}

bool server_session_final(struct http_server *server, uint32_t session_slot,
			  const struct generation_event *event)
{
    struct active_command *const active = &server->active[session_slot];
    struct slot_history *const history = &server->slot_histories[session_slot];
    struct session_store *const store = server->store;
    const bool is_saved = event->event_kind ==
			  GENERATION_EVENT_FINISHED_AT_LIMIT;
    if (active->operation == 0 || store == NULL) {
        return false;
    }
    if (active->operation == SESSION_SAVE) {
	    save_ended(store, active->descriptor, is_saved);
	    history->saved_count =
		    history->processed_count; /* saved, or given up */
	    active->token_ids = store->saving_token_ids == active->token_ids ?
					active->token_ids :
					NULL;
	    store->saving_token_ids =
		    NULL; /* the caller frees them, unless the snapshot keeps them */
    } else {
	    (void)close(active->descriptor);
	    if (is_saved) {
		    memcpy(server->histories +
				   (size_t)session_slot *
					   server->description
						   .context_token_limit,
			   active->token_ids,
			   (size_t)active->prompt_token_count *
				   sizeof(uint32_t));
	    }
	    for (uint32_t index = 0; !is_saved && index < store->snapshot_count;
		 index++) {
		    if (store->snapshots[index].token_count ==
				active->prompt_token_count &&
			server_cache_namespace_equal(
				&store->snapshots[index].cache_namespace,
				&active->cache_namespace) &&
			memcmp(store->snapshots[index].image_identity,
			       active->image_identity,
			       sizeof(active->image_identity)) == 0 &&
			memcmp(store->snapshots[index].token_ids,
			       active->token_ids,
			       (size_t)active->prompt_token_count *
				       sizeof(uint32_t)) == 0) {
			    snapshot_removed(
				    store,
				    index); /* it failed to load: never again */
			    break;
		    }
	    }
	    *history = (struct slot_history){
		    .token_count = is_saved ? active->prompt_token_count : 0,
		    .processed_count = is_saved ? active->prompt_token_count :
						  0,
		    .last_used = ++server->commands_started,
		    .saved_count = is_saved ? active->prompt_token_count : 0,
		    .reach_count = is_saved ? active->reach_count : 0,
		    .cache_namespace = active->cache_namespace,
		    .from_snapshot = is_saved
	    };
	    if (is_saved) {
		    memcpy(history->image_identity, active->image_identity,
			   sizeof(history->image_identity));
	    }
    }
    server->engine_failed = server->engine_failed ||
			    event->status == STATUS_DEVICE_ERROR ||
			    event->status == STATUS_TIMED_OUT;
    return true;
}

/* The slot whose session holds the most tokens in no snapshot, at least session_save_tokens; the slot count if none. */
static uint32_t unsaved_slot(const struct http_server *server)
{
	uint32_t chosen = server->description.session_slot_count,
		 most = server->session_save_tokens - 1u;
	for (uint32_t session_slot = 0;
	     session_slot < server->description.session_slot_count;
	     session_slot++) {
		const struct slot_history *const history =
			&server->slot_histories[session_slot];
		if (server->active[session_slot].token_ids == NULL &&
		    !server->slot_read_leased[session_slot] &&
		    history->processed_count - history->saved_count > most) {
			chosen = session_slot;
			most = history->processed_count - history->saved_count;
		}
	}
    return chosen;
}

void server_sessions_idle(struct http_server *server, double now,
			  double *nearest)
{
	if (server->store == NULL || server->engine_failed ||
	    server->waiting_count != 0 || server_commands_are_active(server) ||
	    unsaved_slot(server) == server->description.session_slot_count) {
		return;
	}
    if (now >= server->idle_since + server->session_idle_save_seconds) {
	    (void)session_started(server, unsaved_slot(server), SESSION_SAVE,
				  0);
    } else if (server->idle_since + server->session_idle_save_seconds <
	       *nearest) {
	    *nearest = server->idle_since + server->session_idle_save_seconds;
    }
}

void server_sessions_kept_at_end(struct http_server *server,
				 int event_descriptor)
{
    while (server->store != NULL && !server->engine_failed) {
        const uint32_t session_slot = unsaved_slot(server);
	if (!server_commands_are_active(server) &&
	    (session_slot == server->description.session_slot_count ||
	     !session_started(server, session_slot, SESSION_SAVE, 0))) {
		if (session_slot == server->description.session_slot_count) {
			break; /* nothing left to save */
		}
		continue; /* that one is given up: the next */
	}
	struct generation_event event = { 0 };
	bool is_available = false;
        struct pollfd poll_entry = {event_descriptor, POLLIN, 0};
	if (poll(&poll_entry, 1,
		 (int)(END_SAVE_SECONDS * MILLISECONDS_PER_SECOND)) <= 0 ||
	    engine_next_event(server->engine, &event, &is_available) != OK) {
		LOG_ERROR("server: a session save at the end did not finish");
		break;
	}
	for (uint32_t slot = 0; is_available && slot < server->description.session_slot_count; slot++) {
            struct active_command *const active = &server->active[slot];
	    if (active->token_ids != NULL &&
		active->request_tag == event.request_tag &&
		event.event_kind != GENERATION_EVENT_TOKEN &&
		server_session_final(server, slot, &event)) {
		    free(active->token_ids);
		    *active = (struct active_command){ .connection =
							       NO_CONNECTION };
	    }
	}
    }
}
