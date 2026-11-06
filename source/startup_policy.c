/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stddef.h>
#include <string.h>

#include "anemone/attention_core_records.h"
#include "anemone/backend_seam.h"
#include "anemone/startup_policy.h"

#define TENSOR_HIDDEN_WIDTH 5120u
#define TENSOR_ATTENTION_HEADS 64u
#define TENSOR_EXPERT_WIDTH 2304u
#define TARGET_LAYERS 40u
#define CMP_PREFIX_LAYERS 29u
#define EXPERT_PAIR_RANKS 2u
#define TP4_SMALL_EXPERT_WIDTH 512u
#define TP4_LARGE_EXPERT_WIDTH 640u
#define HIDDEN_REDUCTIONS_PER_LAYER 2u
#define UNEVEN_CMP_HIDDEN_WIDTH 896u
#define UNEVEN_CMP_ATTENTION_HEADS 11u
#define UNEVEN_CMP_EXPERT_WIDTH 432u

static bool numerics_valid(enum numerics_code numerics)
{
	return numerics == NUMERICS_EXACT || numerics == NUMERICS_RELAXED ||
	       numerics == NUMERICS_FAST;
}

static bool placement_valid(enum execution_placement_code placement)
{
	return placement == EXECUTION_PP5 || placement == EXECUTION_TP5 ||
	       placement == EXECUTION_TP4_PRO ||
	       placement == EXECUTION_TP2_PP2_PRO ||
	       placement == EXECUTION_TP2_SINGLE_SINGLE_PRO ||
	       placement == EXECUTION_TP4_EXL3;
}

static bool policy_valid(const struct startup_policy *policy)
{
	return policy != NULL && numerics_valid(policy->numerics) &&
	       placement_valid(policy->placement) &&
	       (policy->cmp_order == CMP_ORDER_REFERENCE ||
		policy->cmp_order == CMP_ORDER_RELAXED) &&
	       (policy->numerics != NUMERICS_EXACT ||
		policy->cmp_order == CMP_ORDER_REFERENCE);
}

result_status numerics_from_text(const char *text,
				 enum numerics_code *numerics_out)
{
	if (text == NULL || numerics_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	enum numerics_code numerics;
	if (strcmp(text, "exact") == 0) {
		numerics = NUMERICS_EXACT;
	} else if (strcmp(text, "relaxed") == 0) {
		numerics = NUMERICS_RELAXED;
	} else if (strcmp(text, "fast") == 0) {
		numerics = NUMERICS_FAST;
	} else {
		return STATUS_INVALID_ARGUMENT;
	}
	*numerics_out = numerics;
	return STATUS_OK;
}

result_status
execution_placement_from_text(const char *text,
			      enum execution_placement_code *placement_out)
{
	if (text == NULL || placement_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	enum execution_placement_code placement;
	if (strcmp(text, "pp5") == 0) {
		placement = EXECUTION_PP5;
	} else if (strcmp(text, "tp5") == 0) {
		placement = EXECUTION_TP5;
	} else if (strcmp(text, "tp4_pro") == 0) {
		placement = EXECUTION_TP4_PRO;
	} else if (strcmp(text, "tp2_pp2_pro") == 0) {
		placement = EXECUTION_TP2_PP2_PRO;
	} else if (strcmp(text, "tp4_exl3") == 0) {
		placement = EXECUTION_TP4_EXL3;
	} else if (strcmp(text, "tp2_single_single_pro") == 0) {
		placement = EXECUTION_TP2_SINGLE_SINGLE_PRO;
	} else {
		return STATUS_INVALID_ARGUMENT;
	}
	*placement_out = placement;
	return STATUS_OK;
}

result_status
startup_policy_prepare(const struct startup_policy_request *request,
		       struct startup_policy *policy_out)
{
	if (request == NULL || policy_out == NULL ||
	    !numerics_valid(request->numerics) ||
	    !placement_valid(request->placement) ||
	    (request->cmp_order != CMP_ORDER_REFERENCE &&
	     request->cmp_order != CMP_ORDER_RELAXED)) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct startup_policy policy = {
		.numerics = request->numerics,
		.placement = request->placement,
		.cmp_order =
			request->cmp_order_was_given ? request->cmp_order :
			request->numerics == NUMERICS_EXACT ||
					request->placement ==
						EXECUTION_TP2_PP2_PRO ||
					request->placement ==
						EXECUTION_TP4_PRO ||
					request->placement ==
						EXECUTION_TP4_EXL3 ||
					request->placement ==
						EXECUTION_TP2_SINGLE_SINGLE_PRO ?
						       CMP_ORDER_REFERENCE :
						       CMP_ORDER_RELAXED
	};
	if (!request->numerics_was_given &&
	    policy.cmp_order == CMP_ORDER_RELAXED) {
		policy.numerics = NUMERICS_RELAXED;
	}
	if (!policy_valid(&policy)) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (policy.placement != EXECUTION_PP5 &&
	    policy.numerics == NUMERICS_EXACT &&
	    policy.placement != EXECUTION_TP4_PRO &&
	    policy.placement != EXECUTION_TP4_EXL3) {
		return STATUS_UNSUPPORTED;
	}
	*policy_out = policy;
	return STATUS_OK;
}

result_status startup_policy_admit(const struct startup_policy *policy)
{
	if (!policy_valid(policy)) {
		return STATUS_INVALID_ARGUMENT;
	}
	/* Experimental placements; main separately enforces native PRO, text, state and scheduling limits. */
	const bool tensor_pair =
		(policy->placement == EXECUTION_TP2_PP2_PRO ||
		 policy->placement == EXECUTION_TP4_PRO ||
		 policy->placement == EXECUTION_TP4_EXL3 ||
		 policy->placement == EXECUTION_TP2_SINGLE_SINGLE_PRO) &&
		(policy->numerics != NUMERICS_EXACT ||
		 policy->placement == EXECUTION_TP4_PRO ||
		 policy->placement == EXECUTION_TP4_EXL3) &&
		policy->cmp_order == CMP_ORDER_REFERENCE;
	return policy->placement == EXECUTION_PP5 || tensor_pair ?
		       STATUS_OK :
		       STATUS_UNSUPPORTED;
}

result_status tensor_placement_design_prepare(
	const struct startup_policy *policy, const uint32_t *arithmetic_classes,
	uint32_t card_count, struct tensor_placement_design *design_out)
{
	if (!policy_valid(policy) || arithmetic_classes == NULL ||
	    design_out == NULL ||
	    card_count != (policy->placement == EXECUTION_TP4_EXL3 ?
				   TENSOR_PLACEMENT_CMP_CARDS :
				   TENSOR_PLACEMENT_CARDS)) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (policy->placement == EXECUTION_PP5 ||
	    policy->placement == EXECUTION_TP4_EXL3 ||
	    policy->placement == EXECUTION_TP2_PP2_PRO ||
	    policy->placement == EXECUTION_TP2_SINGLE_SINGLE_PRO ||
	    policy->numerics == NUMERICS_EXACT) {
		return STATUS_UNSUPPORTED;
	}
	struct tensor_placement_design design = {
		.target_layer_count = TARGET_LAYERS,
		.tensor_layer_count = policy->placement == EXECUTION_TP5 ?
					      TARGET_LAYERS :
					      CMP_PREFIX_LAYERS,
		.reductions_per_layer = HIDDEN_REDUCTIONS_PER_LAYER,
		.kernel_aligned = true
	};
	bool pro_found = false;
	for (uint32_t card = 0; card < card_count; card++) {
		if (arithmetic_classes[card] == ARITHMETIC_CLASS_SM_80 &&
		    design.rank_count < TENSOR_PLACEMENT_CMP_CARDS) {
			design.ranks[design.rank_count].card = card;
			design.rank_count++;
		} else if (!pro_found &&
			   (arithmetic_classes[card] ==
				    ARITHMETIC_CLASS_SM_120 ||
			    arithmetic_classes[card] ==
				    ARITHMETIC_CLASS_SM_120_NATIVE)) {
			design.pro_card = card;
			pro_found = true;
		} else {
			return STATUS_UNSUPPORTED;
		}
	}
	if (!pro_found || design.rank_count != TENSOR_PLACEMENT_CMP_CARDS) {
		return STATUS_UNSUPPORTED;
	}
	const bool uneven = policy->placement == EXECUTION_TP5;
	const uint32_t cmp_hidden =
		uneven ? UNEVEN_CMP_HIDDEN_WIDTH :
			 TENSOR_HIDDEN_WIDTH / design.rank_count;
	const uint32_t cmp_heads =
		uneven ? UNEVEN_CMP_ATTENTION_HEADS :
			 TENSOR_ATTENTION_HEADS / design.rank_count;
	const uint32_t cmp_expert =
		uneven ? UNEVEN_CMP_EXPERT_WIDTH :
			 TENSOR_EXPERT_WIDTH / design.rank_count;
	uint32_t hidden_begin = 0, head_begin = 0, expert_begin = 0;
	if (uneven) {
		design.ranks[design.rank_count].card = design.pro_card;
		design.rank_count++;
	}
	for (uint32_t rank = 0; rank < design.rank_count; rank++) {
		struct tensor_shard_geometry *const shard = &design.ranks[rank];
		const bool last = rank + 1u == design.rank_count;
		shard->hidden_begin = hidden_begin;
		shard->hidden_count =
			last ? TENSOR_HIDDEN_WIDTH - hidden_begin : cmp_hidden;
		shard->attention_head_begin = head_begin;
		shard->attention_head_count =
			last ? TENSOR_ATTENTION_HEADS - head_begin : cmp_heads;
		shard->expert_intermediate_begin = expert_begin;
		shard->expert_intermediate_count =
			uneven ? (last ? TENSOR_EXPERT_WIDTH - expert_begin :
					 cmp_expert) :
				 (rank % EXPERT_PAIR_RANKS == 0 ?
					  TP4_SMALL_EXPERT_WIDTH :
					  TP4_LARGE_EXPERT_WIDTH);
		design.kernel_aligned =
			design.kernel_aligned &&
			shard->attention_head_begin %
					ATTENTION_CORE_HEADS_PER_BLOCK ==
				0 &&
			shard->attention_head_count %
					ATTENTION_CORE_HEADS_PER_BLOCK ==
				0 &&
			shard->expert_intermediate_begin %
					LINEAR_VALUES_PER_LOAD ==
				0 &&
			shard->expert_intermediate_count %
					LINEAR_VALUES_PER_LOAD ==
				0;
		hidden_begin += shard->hidden_count;
		head_begin += shard->attention_head_count;
		expert_begin += shard->expert_intermediate_count;
	}
	*design_out = design;
	return STATUS_OK;
}
