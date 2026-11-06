/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Startup numerical contract and decode_max's proposed tensor geometry. Host C11 only.
 *
 * exact selects original PP arithmetic, including the experimental output-sharded TP4 recipe. relaxed permits
 * the existing fixed alternate CMP trunk/expert
 * sums; fast currently permits the same sums on PP. Neither changes weight, cache or target-head formats.
 * A specified cmp_order overrides relaxed/fast's default. Explicit exact refuses a relaxed CMP order.
 * An absent numerics option retains the legacy cmp_order option, resolving its effective contract to relaxed.
 * TP2_PP2_PRO defaults to reference local products; its separate relaxation is sharded products/rank sums.
 *
 * TP2_PP2_PRO, TP4_PRO and TP2_SINGLE_SINGLE_PRO are explicit experimental text placements, with a native PRO
 * tail on layers29..39/head/DSpark. Local CMP order is reference. Exact TP4 replicates attention/shared work,
 * shards gate/up outputs into576 channels and down outputs into1280, and losslessly gathers hidden/FP32 sums.
 * Relaxed TP4 uses512/640/512/640 input-axis expert widths; TP2 uses1152. Its attention heads split evenly;
 * KV/router/mHC replicate. The explicit BF16 wire adds hop rounding. Main enforces state/scheduling limits.
 * TP4_EXL3 selects four class-80 cards for all40 target layers plus an unsharded
 * head/drafter on CMP0. Its reference local products and lossless full-K output
 * shards are exact within the selected EXL3 checkpoint; original-weight output
 * equivalence and every serving gate remain separate. Its optional expert order
 * is admitted separately by main, never inferred from relaxed CMP trunk order.
 * Admission qualifies no serving preset or speedup. TP5 remains a design-only unsupported geometry.
 * No arithmetic or placement is silently substituted. Main and host admission tests are the first users.
 *
 * Prepare functions allocate nothing, touch no card, and leave the complete output untouched on failure.
 * NULL arguments, invalid enumerators and inconsistent options are INVALID_ARGUMENT. A valid unsupported
 * execution contract is UNSUPPORTED. Text parsers take the option value exactly, with no surrounding spaces.
 */
#ifndef STARTUP_POLICY_H
#define STARTUP_POLICY_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/linear.h"
#include "anemone/status.h"

#define TENSOR_PLACEMENT_CARDS 5u
#define TENSOR_PLACEMENT_CMP_CARDS 4u

enum numerics_code { NUMERICS_EXACT, NUMERICS_RELAXED, NUMERICS_FAST };
enum execution_placement_code {
	EXECUTION_PP5,
	EXECUTION_TP5,
	EXECUTION_TP4_PRO,
	EXECUTION_TP2_PP2_PRO,
	EXECUTION_TP2_SINGLE_SINGLE_PRO,
	EXECUTION_TP4_EXL3
};

struct startup_policy_request {
	enum numerics_code numerics;
	enum execution_placement_code placement;
	enum cmp_order_code cmp_order;
	bool numerics_was_given;
	bool cmp_order_was_given;
};

struct startup_policy {
	enum numerics_code numerics;
	enum execution_placement_code placement;
	enum cmp_order_code cmp_order;
};

struct tensor_shard_geometry {
	uint32_t card;
	uint32_t hidden_begin, hidden_count;
	uint32_t attention_head_begin, attention_head_count;
	uint32_t expert_intermediate_begin, expert_intermediate_count;
};

struct tensor_placement_design {
	struct tensor_shard_geometry ranks[TENSOR_PLACEMENT_CARDS];
	uint32_t rank_count;
	uint32_t pro_card; /* target head and original DSpark draft */
	uint32_t tensor_layer_count; /* prefix layers; the remaining target layers run wholly on the PRO */
	uint32_t target_layer_count;
	uint32_t reductions_per_layer; /* attention/down and MoE/down; replicated norms, mHC and router */
	bool kernel_aligned; /* discrete shape check only; does not establish runtime admission */
};

MUST_CHECK result_status numerics_from_text(const char *text,
					    enum numerics_code *numerics_out);
MUST_CHECK result_status execution_placement_from_text(
	const char *text, enum execution_placement_code *placement_out);
MUST_CHECK result_status
startup_policy_prepare(const struct startup_policy_request *request,
		       struct startup_policy *policy_out);
MUST_CHECK result_status
startup_policy_admit(const struct startup_policy *policy);

/* Requires exactly four class-80 cards and one class-120/native card, in any order. Ranks are CMPs in their
 * input order, then PRO for TP5. TP4 has four CMP ranks. Only relaxed/fast tensor designs are provided; the
 * modeled exact tensor counterfactual never earns runtime equivalence and returns UNSUPPORTED here too.
 * TP4_EXL3 accepts four-card arguments here but returns UNSUPPORTED without inventing a design.
 * The pair placements return UNSUPPORTED rather than masquerading as a single four-rank group.
 * Shards cover hidden5120, attention64 and expert2304 exactly. Small low-rank state, norms, router and mHC
 * are replicated by the design; that replication still needs an allocator/state implementation and fit proof.
 */
MUST_CHECK result_status tensor_placement_design_prepare(
	const struct startup_policy *policy, const uint32_t *arithmetic_classes,
	uint32_t card_count, struct tensor_placement_design *design_out);

#endif
