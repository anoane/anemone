/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft_reference.h: the host reference of the draft's own kernels (draft_records.h), written from the definitions of
 * dspark_v1/DESIGN.md C5, C6 and C8 and the orders draft_records.h states, with no call into the engine and no formula
 * of a kernel: the oracle of their tests. Test support: no part of the engine.
 *
 * Contract:
 *   - test_draft_verify is what the verify kernel writes of a step and the results its sampler drew, the step's live
 *     rows the smaller of its row_count and record_row_count (the record's): the round's members sequence ..
 *     reserved_zero (proposals and confidence_logits left as they are), the draft's step block and draft_positions
 *     rows, and draft_positions views of STEP_DRAFT_VIEW_BYTES. The emitted count is found by walking the
 *     proposals one after another (the definition of keyed match: DESIGN 7.1), not by a ballot. A step without exactly
 *     one DRAFTS row is the release modules' answer: session STEP_NO_SESSION, e 0, blocks of no rows (only the
 *     first DRAFTS row counts where there are more).
 *   - test_draft_markov_bias adds the Markov bias of token to logits_row [vocabulary_size] in place, in float32 as
 *     draft_records.h orders it (lane l's fmaf chain of 8 terms, the lanes added by exchanges at distance 16 .. 1,
 *     one addition to the logit: a card's bits), and copies W1's row of the token to markov_row [256].
 *     test_draft_markov_bias_exact is the same bias in double, for a check of its size.
 *   - test_draft_confidence is the raw score of a position from its collapse row [5,120], its Markov row [256] and
 *     the weights [5,376], in float32 as draft_records.h orders it (a card's bits); test_draft_confidence_exact the
 *     same in double.
 *   - Every function may be called from any thread.
 */
#ifndef TEST_DRAFT_REFERENCE_H
#define TEST_DRAFT_REFERENCE_H

#include <stdint.h>

#include "anemone/draft_records.h"

void test_draft_verify(const struct step_block *block,
		       const struct step_row_record *rows,
		       const struct step_row_result *results,
		       uint32_t record_row_count, uint32_t draft_positions,
		       uint32_t end_token_id, uint32_t noise_token_id,
		       struct step_round *round, struct step_block *draft_block,
		       struct step_row_record *draft_rows,
		       unsigned char *views);
void test_draft_markov_bias(const uint16_t *embedding, const uint16_t *head,
			    uint32_t vocabulary_size, uint32_t token,
			    float *logits_row, uint16_t *markov_row);
double test_draft_markov_bias_exact(const uint16_t *embedding,
				    const uint16_t *head, uint32_t token,
				    uint32_t entry);
float test_draft_confidence(const uint16_t *collapse_row,
			    const uint16_t *markov_row,
			    const uint16_t *weights);
double test_draft_confidence_exact(const uint16_t *collapse_row,
				   const uint16_t *markov_row,
				   const uint16_t *weights);

#endif
