/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Private all-owner result publication boundary. Model GPU BATCH admission is independent.
 * Borrow immutable, naturally aligned arrays/header through return. The output must be disjoint from inputs.
 * Caller supplies valid CPU storage for all declared entries and serializes publication/readers.
 *
 * round_capacity 0 or 1 retains the legacy single record at index0 and allows at most one DRAFTS owner.
 * round_capacity==session_capacity instead supplies one unchanged88-byte record per session slot. Other
 * capacities are refused. Only DRAFTS owners' records are read; inactive table entries may remain stale.
 * Results/header and every input row are validated before any round is published. For each DRAFTS owner,
 * rows are consecutive by position, earlier rows are ordinary and later rows are PROPOSED. Sessions may interleave,
 * but each session's input positions remain strictly ascending and consecutive; EOS ends acceptance.
 * A future draft block must fit the original conservative anchor+proposed+1+draft_positions context bound.
 * Proposals/confidences in each active round retain the legacy validation, including zero unused tails.
 * Result logits are not changed or newly constrained; legacy token validation remains authoritative.
 *
 * No heap allocation, CUDA, event emission or mutation of caller metadata occurs. Fixed stack scratch holds
 * at most eight sessions. Success replaces the complete output, using NO_SESSION records for inactive slots.
 * Failure leaves every output byte, including the valid mask and padding, untouched. This is a serialized
 * all-or-nothing boundary, not a lock-free multiword publication protocol. All owners must pass before events.
 */
#ifndef STEP_ROUNDS_H
#define STEP_ROUNDS_H

#include <stdbool.h>
#include "anemone/status.h"
#include "anemone/step_records.h"

struct step_round_description {
	const struct step_row_record *rows;
	const struct step_row_result *results;
	const struct step_result *result_header;
	const struct step_round *rounds;
	uint64_t sequence;
	uint32_t row_count;
	uint32_t context_token_limit;
	uint32_t session_capacity;
	uint32_t vocabulary_size;
	uint32_t draft_positions;
	uint32_t end_token_id;
	uint32_t round_capacity;
	uint32_t reserved_zero;
};

struct step_round_table {
	struct step_round rounds[STEP_BATCH_MAXIMUM_SESSIONS];
	uint32_t valid_mask;
};

MUST_CHECK result_status
step_rounds_validate(const struct step_round_description *description,
		     struct step_round_table *output);

struct step_driver;

/* BATCH only: immutable token/EOS domain before the first submitted step. No public runtime knob.
 * Missing/invalid/late attachment or a different domain refuses without changing driver fields.
 * An identical cold attachment is idempotent, allowing engine-create resource-failure retries. */
MUST_CHECK result_status step_rounds_domain_attach(struct step_driver *driver,
						   uint32_t vocabulary_size,
						   uint32_t end_token_id);

MUST_CHECK result_status step_batch_result_take(
	struct step_driver *driver, const unsigned char *result_bytes,
	uint64_t sequence, const struct step_row_record *rows,
	uint32_t row_count, struct step_row_result *results_out);

bool step_rounds_is_batch(const struct step_driver *driver);
#endif
