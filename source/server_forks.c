/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * server_forks.c: poll-thread admission and publication of independent, same-owner exact-prefix copies.
 * A hint prefers an eligible donor; it never supplies ownership, numerical provenance or mutable shared state.
 */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "server_internal.h"

#define OK STATUS_OK

struct fork_donor {
	uint32_t slot, kept;
	bool hinted;
};

static bool forks_enabled(const struct http_server *server,
			  const struct connection *connection)
{
	return server->session_fork != 0 &&
	       server->description.prefix_cold_plan_qualified == 1u &&
	       server->description.prefill_prefix_stable != 0 &&
	       server->cache_scope == SERVER_CACHE_SESSION &&
	       !server->disable_prefix_reuse && !server->engine_failed &&
	       !connection->fork_attempted &&
	       connection->request.max_tokens != 0 &&
	       server->description.session_slot_count > 1u;
}

/* Only idle, separately owned canonical histories can donate. Legacy cold-plan closure alone did not preserve
 * selected-logit bits; require the qualified stable form as well as exact mode, independently of same-session policy.
 * Equal hard ownership precedes both the hint and all token/image/provenance comparisons. */
static struct fork_donor donor_chosen(const struct http_server *server,
				      const struct connection *connection,
				      uint32_t excluded)
{
	struct fork_donor chosen = {
		.slot = server->description.session_slot_count
	};
	if (server->description.prefix_cold_plan_qualified != 1u ||
	    server->description.prefill_prefix_stable == 0)
		return chosen;
	for (uint32_t slot = 0; slot < server->description.session_slot_count;
	     slot++) {
		const struct slot_history *const history =
			&server->slot_histories[slot];
		if (slot == excluded ||
		    server->active[slot].token_ids != NULL ||
		    server->slot_read_leased[slot] ||
		    server_cache_namespace_equal(&history->cache_namespace,
						 &connection->cache_namespace) ||
		    !server_cache_owner_equal(&history->cache_namespace,
					      &connection->cache_namespace)) {
			continue;
		}
		const uint32_t kept = server_prefix_kept_mode(
			server, slot, connection->token_ids,
			connection->token_count, connection->image_identity,
			SERVER_PREFIX_COLD_PLAN);
		/* The first server path copies ordinary state only. The engine's separate checkpoint-copy contract
		 * does not make checkpoint-only donor eligibility part of this policy. */
		if (kept == 0 || history->reach_count < kept ||
		    history->reach_count - kept > SERVING_KEPT_LAG)
			continue;
		const bool hinted = server_cache_fork_hint_matches(
			&connection->fork_parent, &history->cache_namespace);
		if (chosen.kept == 0 || (hinted && !chosen.hinted) ||
		    (hinted == chosen.hinted && kept > chosen.kept)) {
			chosen = (struct fork_donor){ .slot = slot,
						      .kept = kept,
						      .hinted = hinted };
		}
	}
	return chosen;
}

/* Destination choice remains independent of donor preference. The ordinary selector already chose a suitable
 * idle destination unless it chose the donor itself; then retain its grace/affinity/LRU ordering among the rest. */
uint32_t server_fork_destination(const struct http_server *server,
				 const struct connection *connection,
				 uint32_t proposed, double now)
{
	const uint32_t count = server->description.session_slot_count;
	if (!forks_enabled(server, connection) || proposed >= count)
		return proposed;
	const struct fork_donor donor = donor_chosen(server, connection, count);
	if (donor.kept == 0 || donor.slot != proposed)
		return proposed;
	uint32_t chosen = count;
	bool chosen_protected = true, chosen_affinity = false;
	for (uint32_t slot = 0; slot < count; slot++) {
		if (slot == donor.slot ||
		    server->active[slot].token_ids != NULL ||
		    server->slot_read_leased[slot])
			continue;
		const struct slot_history *const history =
			&server->slot_histories[slot];
		const bool same = server_cache_namespace_equal(
			&history->cache_namespace,
			&connection->cache_namespace);
		const bool protected =
			!same && history->token_count != 0 &&
			server->session_grace_seconds != 0 &&
			now < history->last_active_seconds +
					server->session_grace_seconds;
		const bool affinity =
			same && connection->cache_namespace.has_session != 0;
		const bool older =
			chosen != count &&
			history->last_used <
				server->slot_histories[chosen].last_used;
		if (chosen == count || (chosen_protected && !protected) ||
		    (chosen_protected == protected &&
		     ((affinity && !chosen_affinity) ||
		      (affinity == chosen_affinity && older)))) {
			chosen = slot;
			chosen_protected = protected;
			chosen_affinity = affinity;
		}
	}
	return chosen != count ? chosen : proposed;
}

/* Called after snapshot prerequisites, before a cold generation. A refusal or a failed copy is attempted once;
 * the caller can continue cold without waiting for an unavailable donor or retrying the same prerequisite. */
bool server_fork_before(struct http_server *server, uint32_t destination,
			struct connection *connection)
{
	if (!forks_enabled(server, connection))
		return false;
	connection->fork_attempted = true;
	connection->fork_prepared = false;
	if (destination >= server->description.session_slot_count ||
	    server->active[destination].token_ids != NULL ||
	    server->slot_read_leased[destination]) {
		return false;
	}
	const struct fork_donor donor =
		donor_chosen(server, connection, destination);
	if (donor.kept == 0)
		return false;
	struct session_state source = { 0 }, target = { 0 };
	if (engine_session_state(server->engine, donor.slot, &source) != OK ||
	    engine_session_state(server->engine, destination, &target) != OK) {
		return false;
	}
	const struct slot_history *const history =
		&server->slot_histories[donor.slot];
	if (source.reach_count != history->reach_count ||
	    source.decoder_from != history->decoder_from ||
	    source.checkpoint_count != history->checkpoint_count ||
	    source.reach_count < donor.kept ||
	    source.reach_count - donor.kept > SERVING_KEPT_LAG) {
		return false;
	}
	const size_t bytes = (size_t)donor.kept * sizeof(uint32_t);
	uint32_t *const prefix = malloc(bytes);
	if (prefix == NULL)
		return false;
	memcpy(prefix,
	       server->histories +
		       (size_t)donor.slot *
			       server->description.context_token_limit,
	       bytes);
	const uint32_t tag = server_next_tag(server);
	const struct session_clone_command command = {
		.source_generation = source.generation,
		.destination_generation = target.generation,
		.source_slot = donor.slot,
		.destination_slot = destination,
		.token_count = donor.kept,
		.source_kind = SESSION_CLONE_ORDINARY,
		.request_tag = tag
	};
	server->slot_read_leased[donor.slot] = true;
	const result_status status =
		engine_session_clone(server->engine, &command);
	if (status != OK) {
		server->slot_read_leased[donor.slot] = false;
		free(prefix);
		server->engine_failed = server->engine_failed ||
					status == STATUS_DEVICE_ERROR ||
					status == STATUS_TIMED_OUT;
		return false;
	}
	server->active[destination] = (struct active_command){
		.connection = (uint32_t)(connection - server->connections),
		.token_ids = prefix,
		.request_tag = tag,
		.prompt_token_count = donor.kept,
		.operation = SESSION_CLONE,
		.descriptor = -1,
		.fork_source_slot = donor.slot,
		.fork_source_generation = source.generation,
		.fork_destination_generation = target.generation,
		.cache_namespace = connection->cache_namespace
	};
	memcpy(server->active[destination].image_identity,
	       history->image_identity, sizeof(history->image_identity));
	return true;
}

/* The engine has drained every copy before its final event; the poller still owns the source read lease. The
 * return value controls the queued request's READY marker. The caller frees active.token_ids and clears active. */
bool server_fork_final(struct http_server *server, uint32_t destination,
		       const struct generation_event *event, double now)
{
	struct active_command *const active = &server->active[destination];
	const uint32_t source_slot = active->fork_source_slot;
	struct session_state source = { 0 }, target = { 0 };
	const result_status target_status =
		engine_session_state(server->engine, destination, &target);
	bool prepared =
		active->operation == SESSION_CLONE &&
		source_slot < server->description.session_slot_count &&
		source_slot != destination &&
		server->slot_read_leased[source_slot] &&
		event->request_tag == active->request_tag &&
		event->event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT &&
		event->status == OK &&
		event->session_token_count == active->prompt_token_count;
	if (prepared) {
		prepared = target_status == OK &&
			   engine_session_state(server->engine, source_slot,
						&source) == OK &&
			   source.generation ==
				   active->fork_source_generation &&
			   active->fork_destination_generation != UINT64_MAX &&
			   target.generation ==
				   active->fork_destination_generation + 1u;
	}
	if (prepared) {
		const struct slot_history *const donor =
			&server->slot_histories[source_slot];
		prepared = source.reach_count >= active->prompt_token_count &&
			   target.reach_count == source.reach_count &&
			   source.reach_count - active->prompt_token_count <=
				   SERVING_KEPT_LAG &&
			   target.decoder_from == source.decoder_from &&
			   target.checkpoint_count == 0 &&
			   server_cache_owner_equal(&donor->cache_namespace,
						    &active->cache_namespace);
	}
	if (prepared) {
		const struct slot_history donor =
			server->slot_histories[source_slot];
		memcpy(server->histories +
			       (size_t)destination *
				       server->description.context_token_limit,
		       active->token_ids,
		       (size_t)active->prompt_token_count * sizeof(uint32_t));
		server->slot_histories[destination] = (struct slot_history){
			.token_count = active->prompt_token_count,
			.processed_count = active->prompt_token_count,
			.last_used = ++server->commands_started,
			.reach_count = target.reach_count,
			.checkpoint_count = target.checkpoint_count,
			.decoder_from = target.decoder_from,
			.cold_prompt_count = donor.cold_prompt_count,
			.cache_namespace = active->cache_namespace,
			.last_active_seconds = now
		};
		memcpy(server->slot_histories[destination].image_identity,
		       active->image_identity, sizeof(active->image_identity));
	} else {
		/* Accepted clone commands destroy the old destination even when cancellation prevents publication. */
		server->slot_histories[destination] =
			(struct slot_history){ 0 };
	}
	if (source_slot < server->description.session_slot_count)
		server->slot_read_leased[source_slot] = false;
	if (active->connection < SERVER_MOST_CONNECTIONS) {
		struct connection *const connection =
			&server->connections[active->connection];
		if (connection->waiting_snapshot_slot == destination &&
		    connection->waiting_snapshot_tag == active->request_tag) {
			connection->fork_prepared = prepared;
		}
	}
	server->engine_failed = server->engine_failed ||
				event->status == STATUS_DEVICE_ERROR ||
				event->status == STATUS_TIMED_OUT;
	LOG_INFORMATION(
		"server: fork source=%u destination=%u tokens=%u prepared=%u bytes=%" PRIu64
		" copy_ns=%" PRIu64 " wait_ns=%" PRIu64,
		source_slot, destination, active->prompt_token_count,
		prepared ? 1u : 0u, target.clone.byte_count,
		target.clone.copy_ns, target.clone.wait_ns);
	return prepared;
}
